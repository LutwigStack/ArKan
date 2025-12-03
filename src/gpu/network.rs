//! GPU Network implementation.
//!
//! This module provides [`GpuNetwork`] which wraps a CPU KAN network
//! with GPU-accelerated forward pass.

use crate::error::{ArkanError, ArkanResult};
use crate::gpu::backend::WgpuBackend;
use crate::gpu::layer::GpuLayer;
use crate::gpu::pipeline::{PipelineCache, WORKGROUP_SIZE, workgroup_count};
use crate::gpu::workspace::GpuWorkspace;
use crate::network::KanNetwork;
use std::sync::Arc;

/// GPU-accelerated KAN Network.
///
/// `GpuNetwork` holds GPU resources for a KAN network and provides
/// GPU-accelerated forward pass. It wraps a CPU network and syncs
/// weights as needed.
///
/// # Example
///
/// ```rust,ignore
/// use arkan::{KanConfig, KanNetwork};
/// use arkan::gpu::{WgpuBackend, WgpuOptions, GpuNetwork};
///
/// // Create CPU network
/// let config = KanConfig::default_poker();
/// let cpu_network = KanNetwork::new(config);
///
/// // Initialize GPU backend
/// let backend = WgpuBackend::init(WgpuOptions::default())?;
///
/// // Create GPU network from CPU network
/// let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network)?;
///
/// // Forward pass on GPU
/// let input = vec![0.5f32; 21];
/// let output = gpu_network.forward_batch(&input, 1)?;
/// ```
pub struct GpuNetwork {
    /// Reference to the wgpu backend.
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    
    /// GPU layers.
    pub layers: Vec<GpuLayer>,
    
    /// Pipeline cache for compute operations.
    pipeline_cache: PipelineCache,
    
    /// Input dimension.
    pub input_dim: usize,
    /// Output dimension.
    pub output_dim: usize,
    /// Layer dimensions.
    layer_dims: Vec<usize>,
    
    /// Workspace bind group layout.
    workspace_layout: wgpu::BindGroupLayout,
}

impl GpuNetwork {
    /// Creates a GPU network from a CPU network.
    ///
    /// Uploads all layer weights and bias to GPU memory.
    pub fn from_cpu(backend: &WgpuBackend, cpu_network: &KanNetwork) -> ArkanResult<Self> {
        let device = backend.device_arc();
        let queue = backend.queue_arc();
        
        // Upload all layers
        let mut layers = Vec::with_capacity(cpu_network.layers.len());
        for cpu_layer in &cpu_network.layers {
            // Validate weights fit in GPU memory
            backend.validate_layer_weights(cpu_layer.weights.len())?;
            
            let gpu_layer = GpuLayer::from_cpu_layer(&device, cpu_layer)?;
            layers.push(gpu_layer);
        }
        
        // Create pipeline cache
        let pipeline_cache = PipelineCache::new(Arc::clone(&device));
        
        // Get dimensions
        let input_dim = cpu_network.config.input_dim;
        let output_dim = cpu_network.config.output_dim;
        let layer_dims: Vec<_> = cpu_network.layers.iter()
            .map(|l| l.out_dim)
            .collect();
        
        // Create workspace layout
        let workspace_layout = GpuLayer::create_workspace_bind_group_layout(&device);
        
        Ok(Self {
            device,
            queue,
            layers,
            pipeline_cache,
            input_dim,
            output_dim,
            layer_dims,
            workspace_layout,
        })
    }
    
    /// Creates a GPU workspace for this network.
    pub fn create_workspace(&self, max_batch: usize) -> ArkanResult<GpuWorkspace> {
        GpuWorkspace::new(&self.device, max_batch, self.input_dim, self.output_dim)
    }
    
    /// Performs forward pass on GPU.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data [batch_size * input_dim].
    /// * `batch_size` - Number of samples in the batch.
    /// * `workspace` - GPU workspace for intermediate buffers.
    ///
    /// # Returns
    ///
    /// Output data [batch_size * output_dim].
    pub fn forward_batch(
        &mut self,
        input: &[f32],
        batch_size: usize,
        workspace: &mut GpuWorkspace,
    ) -> ArkanResult<Vec<f32>> {
        // Validate input
        let expected_input_len = batch_size * self.input_dim;
        if input.len() != expected_input_len {
            return Err(ArkanError::shape_mismatch(
                &[batch_size, self.input_dim],
                &[input.len() / self.input_dim, self.input_dim],
            ));
        }
        
        // Ensure workspace capacity
        workspace.ensure_capacity(&self.device, batch_size)?;
        
        // Upload input
        workspace.upload_input(&self.queue, input)?;
        
        // Execute forward pass
        self.execute_forward(batch_size, workspace)?;
        
        // Download output
        workspace.download_output(&self.device, &self.queue, batch_size)
    }
    
    /// Executes the forward pass computation.
    fn execute_forward(
        &mut self,
        batch_size: usize,
        workspace: &mut GpuWorkspace,
    ) -> ArkanResult<()> {
        if self.layers.is_empty() {
            return Ok(());
        }
        
        // For single layer network
        if self.layers.len() == 1 {
            return self.execute_single_layer_forward(0, batch_size, workspace);
        }
        
        // Multi-layer: need intermediate buffers
        workspace.ensure_intermediates(&self.device, &self.layer_dims, batch_size)?;
        
        // Execute layers sequentially
        let num_layers = self.layers.len();
        for i in 0..num_layers {
            self.execute_layer_forward_multi(i, num_layers, batch_size, workspace)?;
        }
        
        Ok(())
    }
    
    /// Executes forward pass for a single layer network.
    fn execute_single_layer_forward(
        &mut self,
        layer_idx: usize,
        batch_size: usize,
        workspace: &mut GpuWorkspace,
    ) -> ArkanResult<()> {
        let layer = &mut self.layers[layer_idx];
        
        // Update batch size in uniforms
        layer.update_batch_size(&self.queue, batch_size);
        
        // Get or create pipeline
        let pipeline = self.pipeline_cache
            .get_forward_simple_pipeline(&layer.bind_group_layout)?;
        
        // Create workspace bind group
        let workspace_bg = workspace.get_or_create_bind_group(&self.device, &self.workspace_layout)?;
        
        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Forward Encoder"),
        });
        
        // Dispatch compute
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Forward Pass"),
                timestamp_writes: None,
            });
            
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &layer.bind_group, &[]);
            pass.set_bind_group(1, workspace_bg, &[]);
            
            // One thread per output element
            let total_outputs = batch_size * layer.out_dim;
            let num_workgroups = workgroup_count(total_outputs, WORKGROUP_SIZE);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        
        // Submit
        self.queue.submit(std::iter::once(encoder.finish()));
        
        Ok(())
    }
    
    /// Executes forward pass for one layer in a multi-layer network.
    fn execute_layer_forward_multi(
        &mut self,
        layer_idx: usize,
        num_layers: usize,
        batch_size: usize,
        workspace: &mut GpuWorkspace,
    ) -> ArkanResult<()> {
        // First update uniforms for the layer
        self.layers[layer_idx].update_batch_size(&self.queue, batch_size);
        
        // Get layer info we need before borrowing
        let out_dim = self.layers[layer_idx].out_dim;
        
        // Determine input and output buffers
        let (input_buffer, output_buffer) = if layer_idx == 0 {
            // First layer: input from workspace.input, output to intermediate[0]
            (
                workspace.input.as_ref().ok_or_else(|| ArkanError::buffer("No input buffer"))?,
                &workspace.intermediates[0],
            )
        } else if layer_idx == num_layers - 1 {
            // Last layer: input from intermediate[n-2], output to workspace.output
            (
                &workspace.intermediates[layer_idx - 1],
                workspace.output.as_ref().ok_or_else(|| ArkanError::buffer("No output buffer"))?,
            )
        } else {
            // Middle layer: intermediate[i-1] -> intermediate[i]
            let (left, right) = workspace.intermediates.split_at(layer_idx);
            (&left[layer_idx - 1], &right[0])
        };
        
        // Create temporary bind group for this layer's I/O
        let io_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Layer I/O BindGroup"),
            layout: &self.workspace_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.buffer.as_entire_binding(),
                },
            ],
        });
        
        // Get bind group layout reference for pipeline creation
        let bind_group_layout = &self.layers[layer_idx].bind_group_layout;
        
        // Get pipeline
        let pipeline = self.pipeline_cache
            .get_forward_simple_pipeline(bind_group_layout)?;
        
        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("Forward Layer {} Encoder", layer_idx)),
        });
        
        // Dispatch
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Forward Layer {} Pass", layer_idx)),
                timestamp_writes: None,
            });
            
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.layers[layer_idx].bind_group, &[]);
            pass.set_bind_group(1, &io_bind_group, &[]);
            
            let total_outputs = batch_size * out_dim;
            let num_workgroups = workgroup_count(total_outputs, WORKGROUP_SIZE);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        
        // Submit
        self.queue.submit(std::iter::once(encoder.finish()));
        
        // Wait for this layer to complete before next
        self.device.poll(wgpu::Maintain::Wait);
        
        Ok(())
    }
    
    /// Syncs weights from CPU network to GPU.
    pub fn sync_weights(&mut self, cpu_network: &KanNetwork) -> ArkanResult<()> {
        if cpu_network.layers.len() != self.layers.len() {
            return Err(ArkanError::validation(
                "CPU and GPU network layer count mismatch"
            ));
        }
        
        for (gpu_layer, cpu_layer) in self.layers.iter_mut().zip(&cpu_network.layers) {
            gpu_layer.update_weights(&self.queue, cpu_layer);
            gpu_layer.update_bias(&self.queue, cpu_layer);
        }
        
        Ok(())
    }
    
    /// Returns the total parameter count.
    pub fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.param_count()).sum()
    }
    
    /// Returns layer count.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

impl std::fmt::Debug for GpuNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuNetwork")
            .field("input_dim", &self.input_dim)
            .field("output_dim", &self.output_dim)
            .field("num_layers", &self.layers.len())
            .field("param_count", &self.param_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    // GPU tests require actual GPU, run with: cargo test --features gpu -- --ignored
}
