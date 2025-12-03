//! GPU Network implementation.
//!
//! This module provides [`GpuNetwork`] which wraps a CPU KAN network
//! with GPU-accelerated forward pass.

use crate::error::{ArkanError, ArkanResult};
use crate::gpu::backend::WgpuBackend;
use crate::gpu::layer::GpuLayer;
use crate::gpu::pipeline::{PipelineCache, WORKGROUP_SIZE, workgroup_count};
use crate::gpu::workspace::GpuWorkspace;
use crate::network::{KanNetwork, TrainOptions};
use crate::optimizer::{Adam, SGD};
use crate::loss::{masked_mse, masked_cross_entropy};
use std::sync::Arc;
use wgpu::util::DeviceExt;

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
/// let config = KanConfig::preset();
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
            // Validate weights fit in GPU memory (uses padded vec4 size)
            backend.validate_layer_weights(
                cpu_layer.out_dim,
                cpu_layer.in_dim,
                cpu_layer.global_basis_size,
            )?;
            
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
    
    /// Performs forward pass for a single input on GPU.
    ///
    /// This is an optimized version for batch_size=1, which is common
    /// in inference scenarios like game AI.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data `(input_dim,)`.
    /// * `workspace` - GPU workspace for intermediate buffers.
    ///
    /// # Returns
    ///
    /// Output data `(output_dim,)`.
    pub fn forward_single(
        &mut self,
        input: &[f32],
        workspace: &mut GpuWorkspace,
    ) -> ArkanResult<Vec<f32>> {
        if input.len() != self.input_dim {
            return Err(ArkanError::shape_mismatch(
                &[self.input_dim],
                &[input.len()],
            ));
        }
        
        // Use forward_batch with batch_size=1
        self.forward_batch(input, 1, workspace)
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
    ///
    /// Uses cached bind groups from workspace to avoid recreation overhead.
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
        
        // Get or create cached bind group for this layer's I/O
        let io_bind_group = workspace.get_or_create_layer_bind_group(
            &self.device,
            &self.workspace_layout,
            layer_idx,
            num_layers,
        )?;
        
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
            pass.set_bind_group(1, io_bind_group, &[]);
            
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
    
    // ==================== Softmax ====================
    
    /// Applies softmax normalization to the output buffer in-place.
    ///
    /// Softmax transforms logits into a probability distribution:
    /// softmax(x_i) = exp(x_i) / sum(exp(x_j))
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of samples in the batch.
    /// * `workspace` - GPU workspace containing the output buffer to normalize.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Forward pass followed by softmax
    /// let logits = gpu_network.forward_batch(&inputs, batch_size, &mut workspace)?;
    /// gpu_network.apply_softmax(batch_size, &mut workspace)?;
    /// let probabilities = workspace.download_output(&device, &queue, batch_size)?;
    /// ```
    pub fn apply_softmax(
        &mut self,
        batch_size: usize,
        workspace: &mut GpuWorkspace,
    ) -> ArkanResult<()> {
        // Create uniform buffer for softmax config
        // struct Uniforms { num_elements: u32, dim: u32, batch_size: u32, _padding: u32 }
        let config_data = [
            (batch_size * self.output_dim) as u32, // num_elements
            self.output_dim as u32,                 // dim
            batch_size as u32,                      // batch_size
            0u32,                                   // padding
        ];
        
        let config_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Softmax Config"),
            contents: bytemuck::cast_slice(&config_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        
        // Get output buffer reference
        let output = workspace.output.as_ref()
            .ok_or_else(|| ArkanError::validation("No output buffer in workspace"))?;
        
        // Get layout and create bind group before getting pipeline
        let softmax_layout = self.pipeline_cache.get_softmax_layout();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Softmax BindGroup"),
            layout: softmax_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.buffer.as_entire_binding(),
                },
            ],
        });
        
        // Get softmax pipeline (after layout usage is done)
        let pipeline = self.pipeline_cache.get_softmax_pipeline()?;
        
        // Create command encoder and dispatch
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Softmax Encoder"),
        });
        
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Softmax Pass"),
                timestamp_writes: None,
            });
            
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            
            // One workgroup per batch sample
            let num_workgroups = workgroup_count(batch_size, WORKGROUP_SIZE);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        
        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        
        Ok(())
    }
    
    /// Performs forward pass with softmax on GPU.
    ///
    /// Combines forward_batch and apply_softmax into a single operation.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data [batch_size * input_dim].
    /// * `batch_size` - Number of samples in the batch.
    /// * `workspace` - GPU workspace for intermediate buffers.
    ///
    /// # Returns
    ///
    /// Output probabilities [batch_size * output_dim] summing to 1.0 per sample.
    pub fn forward_batch_softmax(
        &mut self,
        input: &[f32],
        batch_size: usize,
        workspace: &mut GpuWorkspace,
    ) -> ArkanResult<Vec<f32>> {
        // Run forward pass
        let _ = self.forward_batch(input, batch_size, workspace)?;
        
        // Apply softmax in-place
        self.apply_softmax(batch_size, workspace)?;
        
        // Download output
        workspace.download_output(&self.device, &self.queue, batch_size)
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
    
    // ==================== Training Methods ====================
    
    /// Performs forward pass with training data saved for backward.
    ///
    /// This variant saves z_values and span_indices for each layer,
    /// which are needed for the backward pass.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data [batch_size * input_dim].
    /// * `batch_size` - Number of samples in the batch.
    /// * `workspace` - GPU workspace for intermediate and training buffers.
    ///
    /// # Returns
    ///
    /// Output data [batch_size * output_dim].
    pub fn forward_batch_training(
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
        
        // Prepare training buffers (z_values, span_indices per layer)
        // Build layer_dims array: [in_dim, out_dim_0, out_dim_1, ...]
        let mut layer_dims = vec![self.input_dim];
        for layer in &self.layers {
            layer_dims.push(layer.out_dim);
        }
        workspace.prepare_training(&self.device, &layer_dims, batch_size)?;
        
        // Upload input
        workspace.upload_input(&self.queue, input)?;
        
        // Execute forward pass with training data saving
        self.execute_forward_training(batch_size, workspace)?;
        
        // Download output
        workspace.download_output(&self.device, &self.queue, batch_size)
    }
    
    /// Executes the forward training pass.
    fn execute_forward_training(
        &mut self,
        batch_size: usize,
        workspace: &mut GpuWorkspace,
    ) -> ArkanResult<()> {
        if self.layers.is_empty() {
            return Ok(());
        }
        
        // Get training workspace layout
        let training_layout = self.pipeline_cache.get_training_workspace_layout();
        // Need to clone reference before mutable borrow
        let training_layout_ref = unsafe { &*(training_layout as *const _) };
        
        // For single layer network
        if self.layers.len() == 1 {
            return self.execute_single_layer_forward_training(
                0, batch_size, workspace, training_layout_ref
            );
        }
        
        // Multi-layer: need intermediate buffers
        workspace.ensure_intermediates(&self.device, &self.layer_dims, batch_size)?;
        
        // Execute layers sequentially
        let num_layers = self.layers.len();
        for i in 0..num_layers {
            self.execute_layer_forward_training_multi(
                i, num_layers, batch_size, workspace, training_layout_ref
            )?;
        }
        
        Ok(())
    }
    
    /// Executes forward training for a single layer network.
    fn execute_single_layer_forward_training(
        &mut self,
        layer_idx: usize,
        batch_size: usize,
        workspace: &mut GpuWorkspace,
        training_layout: &wgpu::BindGroupLayout,
    ) -> ArkanResult<()> {
        let layer = &mut self.layers[layer_idx];
        let in_dim = layer.in_dim;
        let out_dim = layer.out_dim;
        
        // Update batch size in uniforms
        layer.update_batch_size(&self.queue, batch_size);
        
        // Get or create pipeline
        let pipeline = self.pipeline_cache
            .get_forward_training_pipeline(&layer.bind_group_layout)?;
        
        // Create training bind group (input, output, z_values, span_indices)
        let training_bg = workspace.get_or_create_training_bind_group(
            &self.device,
            training_layout,
            layer_idx,
            in_dim,
            batch_size,
        )?;
        
        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Forward Training Encoder"),
        });
        
        // Dispatch compute
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Forward Training Pass"),
                timestamp_writes: None,
            });
            
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &layer.bind_group, &[]);
            pass.set_bind_group(1, training_bg, &[]);
            
            // One thread per output element
            let total_outputs = batch_size * out_dim;
            let num_workgroups = workgroup_count(total_outputs, WORKGROUP_SIZE);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        
        // Submit
        self.queue.submit(std::iter::once(encoder.finish()));
        
        Ok(())
    }
    
    /// Executes forward training for one layer in multi-layer network.
    fn execute_layer_forward_training_multi(
        &mut self,
        layer_idx: usize,
        num_layers: usize,
        batch_size: usize,
        workspace: &mut GpuWorkspace,
        training_layout: &wgpu::BindGroupLayout,
    ) -> ArkanResult<()> {
        let layer = &mut self.layers[layer_idx];
        let in_dim = layer.in_dim;
        let out_dim = layer.out_dim;
        
        // Update batch size
        layer.update_batch_size(&self.queue, batch_size);
        
        // Get or create pipeline
        let bind_group_layout = &layer.bind_group_layout;
        let pipeline = self.pipeline_cache
            .get_forward_training_pipeline(bind_group_layout)?;
        
        // Create training bind group for this layer's I/O
        let training_bg = workspace.get_or_create_training_layer_bind_group(
            &self.device,
            training_layout,
            layer_idx,
            num_layers,
            in_dim,
            batch_size,
        )?;
        
        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("Forward Training Layer {} Encoder", layer_idx)),
        });
        
        // Dispatch
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Forward Training Layer {} Pass", layer_idx)),
                timestamp_writes: None,
            });
            
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.layers[layer_idx].bind_group, &[]);
            pass.set_bind_group(1, training_bg, &[]);
            
            let total_outputs = batch_size * out_dim;
            let num_workgroups = workgroup_count(total_outputs, WORKGROUP_SIZE);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        
        // Submit and wait
        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        
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
    
    // ==================== Backward Pass ====================
    
    /// Performs backward pass on GPU.
    ///
    /// Computes weight gradients and input gradients for all layers using GPU shaders.
    /// Must be called after `forward_batch_training()`.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Gradient of loss w.r.t. output [batch_size * output_dim].
    /// * `batch_size` - Number of samples in the batch.
    /// * `workspace` - GPU workspace with training buffers from forward_batch_training.
    /// * `grad_weights` - Output: gradients for each layer's weights.
    /// * `grad_biases` - Output: gradients for each layer's biases.
    ///
    /// # Returns
    ///
    /// Gradient of loss w.r.t. input (for chaining with other layers).
    pub fn backward_batch(
        &mut self,
        grad_output: &[f32],
        batch_size: usize,
        workspace: &mut GpuWorkspace,
        grad_weights: &mut Vec<Vec<f32>>,
        grad_biases: &mut Vec<Vec<f32>>,
    ) -> ArkanResult<Vec<f32>> {
        // Validate
        let expected_len = batch_size * self.output_dim;
        if grad_output.len() != expected_len {
            return Err(ArkanError::shape_mismatch(
                &[batch_size, self.output_dim],
                &[grad_output.len() / self.output_dim, self.output_dim],
            ));
        }
        
        // Prepare output vectors
        grad_weights.clear();
        grad_biases.clear();
        grad_weights.reserve(self.layers.len());
        grad_biases.reserve(self.layers.len());
        
        // Prepare gradient buffers in workspace
        let layer_specs: Vec<_> = self.layers.iter()
            .map(|l| (l.in_dim, l.out_dim, l.basis_padded))
            .collect();
        workspace.prepare_grad_buffers(&self.device, &layer_specs)?;
        
        // Ensure std_inv is set (default to 1.0 if not set)
        if workspace.std_inv.len() != self.layers.len() {
            let default_std_inv: Vec<Vec<f32>> = self.layers.iter()
                .map(|l| vec![1.0f32; l.in_dim])
                .collect();
            workspace.set_std_inv(&self.device, &self.queue, &default_std_inv)?;
        }
        
        // Zero gradient buffers
        workspace.zero_grad_buffers(&self.queue);
        
        // Upload grad_output to GPU
        workspace.upload_grad_output(&self.queue, grad_output)?;
        
        // Process layers in reverse order
        let num_layers = self.layers.len();
        
        for layer_idx in (0..num_layers).rev() {
            let compute_input_grad = layer_idx > 0;
            
            // Execute GPU backward for this layer
            self.backward_layer_gpu(layer_idx, batch_size, workspace, compute_input_grad)?;
            
            // Download gradients from GPU
            let layer_grad_weights = workspace.download_grad_weights(&self.device, &self.queue, layer_idx)?;
            let layer_grad_bias = workspace.download_grad_bias(&self.device, &self.queue, layer_idx)?;
            
            grad_weights.push(layer_grad_weights);
            grad_biases.push(layer_grad_bias);
            
            // If computing input grad, copy grad_input to grad_output for next layer
            if compute_input_grad {
                let grad_input = workspace.download_grad_input(
                    &self.device, &self.queue, batch_size, self.layers[layer_idx].in_dim
                )?;
                workspace.upload_grad_output(&self.queue, &grad_input)?;
            }
        }
        
        // Reverse the gradient vectors (they were computed in reverse order)
        grad_weights.reverse();
        grad_biases.reverse();
        
        // Download final grad_input
        let grad_input = workspace.download_grad_input(
            &self.device, &self.queue, batch_size, self.input_dim
        )?;
        
        Ok(grad_input)
    }
    
    /// Executes backward pass for a single layer on GPU.
    fn backward_layer_gpu(
        &mut self,
        layer_idx: usize,
        batch_size: usize,
        workspace: &GpuWorkspace,
        compute_input_grad: bool,
    ) -> ArkanResult<()> {
        let layer = &self.layers[layer_idx];
        let in_dim = layer.in_dim;
        let out_dim = layer.out_dim;
        let basis_padded = layer.basis_padded;
        
        // Create backward uniforms buffer
        let backward_uniforms = crate::gpu::uniforms::BackwardUniforms::new(
            &layer.uniforms,
            compute_input_grad,
        );
        let uniforms_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Backward Uniforms"),
            contents: bytemuck::bytes_of(&backward_uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        
        // Update batch_size in uniforms
        let mut uniforms = layer.uniforms.clone();
        uniforms.batch_size = batch_size as u32;
        self.queue.write_buffer(&layer.uniforms_buffer, 0, bytemuck::bytes_of(&uniforms));
        
        // Create backward bind group layout (Group 0: weights + uniforms)
        let backward_layer_layout = GpuLayer::create_backward_bind_group_layout(&self.device);
        let backward_layer_bind_group = layer.create_backward_bind_group(
            &self.device, &backward_layer_layout, &uniforms_buffer
        );
        
        // Create backward workspace bind group (Group 1)
        // Need to use unsafe to break the borrow chain
        let backward_workspace_layout = self.pipeline_cache.get_backward_workspace_layout();
        let backward_workspace_layout_ptr = backward_workspace_layout as *const _;
        let backward_workspace_bind_group = self.create_backward_workspace_bind_group(
            workspace, layer_idx, unsafe { &*backward_workspace_layout_ptr }
        )?;
        
        // Get pipelines
        let weights_pipeline = self.pipeline_cache
            .get_backward_weights_pipeline(&backward_layer_layout)?;
        
        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Backward Pass"),
        });
        
        // Dispatch weight gradients shader
        {
            let total_weights = out_dim * in_dim * basis_padded;
            let workgroups = workgroup_count(total_weights, WORKGROUP_SIZE);
            
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Backward Weights"),
                timestamp_writes: None,
            });
            pass.set_pipeline(weights_pipeline);
            pass.set_bind_group(0, &backward_layer_bind_group, &[]);
            pass.set_bind_group(1, &backward_workspace_bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        // Dispatch bias gradients shader
        {
            let bias_layout = self.pipeline_cache.create_bias_bind_group_layout();
            let bias_uniforms = crate::gpu::uniforms::BiasUniforms::new(out_dim as u32, batch_size as u32);
            let bias_uniforms_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Bias Uniforms"),
                contents: bytemuck::bytes_of(&bias_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            
            let grad_output_buf = workspace.grad_output.as_ref()
                .ok_or_else(|| ArkanError::buffer("grad_output not allocated"))?;
            let grad_bias_buf = workspace.grad_bias.get(layer_idx)
                .ok_or_else(|| ArkanError::buffer("grad_bias not allocated"))?;
            
            let bias_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bias Backward BindGroup"),
                layout: &bias_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: bias_uniforms_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: grad_output_buf.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: grad_bias_buf.buffer.as_entire_binding(),
                    },
                ],
            });
            
            let bias_pipeline = self.pipeline_cache.get_backward_bias_pipeline(&bias_layout)?;
            let workgroups = workgroup_count(out_dim, WORKGROUP_SIZE);
            
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Backward Bias"),
                timestamp_writes: None,
            });
            pass.set_pipeline(bias_pipeline);
            pass.set_bind_group(0, &bias_bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        // Dispatch input gradients shader (if needed)
        if compute_input_grad {
            let input_pipeline = self.pipeline_cache
                .get_backward_input_pipeline(&backward_layer_layout)?;
            
            let total_inputs = batch_size * in_dim;
            let workgroups = workgroup_count(total_inputs, WORKGROUP_SIZE);
            
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Backward Input"),
                timestamp_writes: None,
            });
            pass.set_pipeline(input_pipeline);
            pass.set_bind_group(0, &backward_layer_bind_group, &[]);
            pass.set_bind_group(1, &backward_workspace_bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        // Submit
        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        
        Ok(())
    }
    
    /// Creates backward workspace bind group (Group 1).
    fn create_backward_workspace_bind_group(
        &self,
        workspace: &GpuWorkspace,
        layer_idx: usize,
        layout: &wgpu::BindGroupLayout,
    ) -> ArkanResult<wgpu::BindGroup> {
        let z_values = workspace.z_values.get(layer_idx)
            .ok_or_else(|| ArkanError::buffer("z_values not allocated"))?;
        let span_indices = workspace.span_indices.get(layer_idx)
            .ok_or_else(|| ArkanError::buffer("span_indices not allocated"))?;
        let grad_output = workspace.grad_output.as_ref()
            .ok_or_else(|| ArkanError::buffer("grad_output not allocated"))?;
        let grad_weights = workspace.grad_weights.get(layer_idx)
            .ok_or_else(|| ArkanError::buffer("grad_weights not allocated"))?;
        let grad_bias = workspace.grad_bias.get(layer_idx)
            .ok_or_else(|| ArkanError::buffer("grad_bias not allocated"))?;
        let grad_input = workspace.grad_input.as_ref()
            .ok_or_else(|| ArkanError::buffer("grad_input not allocated"))?;
        let std_inv = workspace.std_inv.get(layer_idx)
            .ok_or_else(|| ArkanError::buffer("std_inv not allocated"))?;
        
        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Backward Workspace BindGroup"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: z_values.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: span_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grad_output.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grad_weights.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: grad_bias.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: grad_input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: std_inv.buffer.as_entire_binding(),
                },
            ],
        }))
    }
    
    // ==================== Training Step ====================
    
    /// Performs a complete training step on GPU with MSE loss.
    ///
    /// This method combines forward pass, loss computation, backward pass,
    /// and optimizer step into a single call.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data [batch_size * input_dim].
    /// * `target` - Target data [batch_size * output_dim].
    /// * `batch_size` - Number of samples in the batch.
    /// * `workspace` - GPU workspace for training buffers.
    /// * `optimizer` - Adam optimizer with state.
    /// * `cpu_network` - CPU network to update weights on.
    ///
    /// # Returns
    ///
    /// The loss value for this batch.
    pub fn train_step_mse(
        &mut self,
        input: &[f32],
        target: &[f32],
        batch_size: usize,
        workspace: &mut GpuWorkspace,
        optimizer: &mut Adam,
        cpu_network: &mut KanNetwork,
    ) -> ArkanResult<f32> {
        // Validate target shape
        let expected_target_len = batch_size * self.output_dim;
        if target.len() != expected_target_len {
            return Err(ArkanError::shape_mismatch(
                &[batch_size, self.output_dim],
                &[target.len() / self.output_dim, self.output_dim],
            ));
        }
        
        // 1. Forward pass with training data
        let output = self.forward_batch_training(input, batch_size, workspace)?;
        
        // 2. Compute loss and gradient
        let (loss, grad_output) = masked_mse(&output, target, None);
        
        // 3. Backward pass
        let mut grad_weights = Vec::new();
        let mut grad_biases = Vec::new();
        let _grad_input = self.backward_batch(
            &grad_output,
            batch_size,
            workspace,
            &mut grad_weights,
            &mut grad_biases,
        )?;
        
        // 4. Optimizer step on CPU network
        optimizer.step(cpu_network, &grad_weights, &grad_biases, Some(1.0));
        
        // 5. Sync weights from CPU to GPU
        self.sync_weights(cpu_network)?;
        
        Ok(loss)
    }
    
    /// Performs a training step with cross-entropy loss.
    pub fn train_step_cross_entropy(
        &mut self,
        input: &[f32],
        target: &[f32],
        batch_size: usize,
        workspace: &mut GpuWorkspace,
        optimizer: &mut Adam,
        cpu_network: &mut KanNetwork,
    ) -> ArkanResult<f32> {
        // Validate target shape
        let expected_target_len = batch_size * self.output_dim;
        if target.len() != expected_target_len {
            return Err(ArkanError::shape_mismatch(
                &[batch_size, self.output_dim],
                &[target.len() / self.output_dim, self.output_dim],
            ));
        }
        
        // 1. Forward pass with training data
        let output = self.forward_batch_training(input, batch_size, workspace)?;
        
        // 2. Compute loss and gradient
        let (loss, grad_output) = masked_cross_entropy(&output, target, None);
        
        // 3. Backward pass
        let mut grad_weights = Vec::new();
        let mut grad_biases = Vec::new();
        let _grad_input = self.backward_batch(
            &grad_output,
            batch_size,
            workspace,
            &mut grad_weights,
            &mut grad_biases,
        )?;
        
        // 4. Optimizer step on CPU network
        optimizer.step(cpu_network, &grad_weights, &grad_biases, Some(1.0));
        
        // 5. Sync weights from CPU to GPU
        self.sync_weights(cpu_network)?;
        
        Ok(loss)
    }
    
    /// Performs a training step with options (gradient clipping, weight decay) using Adam.
    ///
    /// This method provides parity with the CPU [`KanNetwork::train_step_with_options`].
    ///
    /// # Arguments
    ///
    /// * `input` - Input data [batch_size * input_dim].
    /// * `target` - Target data [batch_size * output_dim].
    /// * `mask` - Optional mask [batch_size * output_dim] (1.0 = active, 0.0 = ignore).
    /// * `batch_size` - Number of samples in the batch.
    /// * `workspace` - GPU workspace for training buffers.
    /// * `optimizer` - Adam optimizer with state.
    /// * `cpu_network` - CPU network to update weights on.
    /// * `opts` - Training options (gradient clipping, weight decay).
    ///
    /// # Returns
    ///
    /// The loss value for this batch.
    pub fn train_step_with_options(
        &mut self,
        input: &[f32],
        target: &[f32],
        mask: Option<&[f32]>,
        batch_size: usize,
        workspace: &mut GpuWorkspace,
        optimizer: &mut Adam,
        cpu_network: &mut KanNetwork,
        opts: &TrainOptions,
    ) -> ArkanResult<f32> {
        // Validate target shape
        let expected_target_len = batch_size * self.output_dim;
        if target.len() != expected_target_len {
            return Err(ArkanError::shape_mismatch(
                &[batch_size, self.output_dim],
                &[target.len() / self.output_dim, self.output_dim],
            ));
        }
        
        // 1. Forward pass with training data
        let output = self.forward_batch_training(input, batch_size, workspace)?;
        
        // 2. Compute loss and gradient
        let (loss, grad_output) = masked_mse(&output, target, mask);
        
        // 3. Backward pass
        let mut grad_weights = Vec::new();
        let mut grad_biases = Vec::new();
        let _grad_input = self.backward_batch(
            &grad_output,
            batch_size,
            workspace,
            &mut grad_weights,
            &mut grad_biases,
        )?;
        
        // 4. Apply weight decay if specified (AdamW-style decoupled decay)
        if opts.weight_decay > 0.0 {
            let lr = optimizer.config.lr;
            for layer in &mut cpu_network.layers {
                for w in layer.weights.as_mut_slice() {
                    *w *= 1.0 - lr * opts.weight_decay;
                }
            }
        }
        
        // 5. Optimizer step on CPU network with gradient clipping
        optimizer.step(cpu_network, &grad_weights, &grad_biases, opts.max_grad_norm);
        
        // 6. Sync weights from CPU to GPU
        self.sync_weights(cpu_network)?;
        
        Ok(loss)
    }
    
    /// Performs a training step using SGD optimizer.
    ///
    /// This method provides parity with CPU training using SGD.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data [batch_size * input_dim].
    /// * `target` - Target data [batch_size * output_dim].
    /// * `batch_size` - Number of samples in the batch.
    /// * `workspace` - GPU workspace for training buffers.
    /// * `optimizer` - SGD optimizer.
    /// * `cpu_network` - CPU network to update weights on.
    ///
    /// # Returns
    ///
    /// The loss value for this batch.
    pub fn train_step_sgd(
        &mut self,
        input: &[f32],
        target: &[f32],
        batch_size: usize,
        workspace: &mut GpuWorkspace,
        optimizer: &mut SGD,
        cpu_network: &mut KanNetwork,
    ) -> ArkanResult<f32> {
        // Validate target shape
        let expected_target_len = batch_size * self.output_dim;
        if target.len() != expected_target_len {
            return Err(ArkanError::shape_mismatch(
                &[batch_size, self.output_dim],
                &[target.len() / self.output_dim, self.output_dim],
            ));
        }
        
        // 1. Forward pass with training data
        let output = self.forward_batch_training(input, batch_size, workspace)?;
        
        // 2. Compute loss and gradient
        let (loss, grad_output) = masked_mse(&output, target, None);
        
        // 3. Backward pass
        let mut grad_weights = Vec::new();
        let mut grad_biases = Vec::new();
        let _grad_input = self.backward_batch(
            &grad_output,
            batch_size,
            workspace,
            &mut grad_weights,
            &mut grad_biases,
        )?;
        
        // 4. Optimizer step on CPU network
        optimizer.step(cpu_network, &grad_weights, &grad_biases, None);
        
        // 5. Sync weights from CPU to GPU
        self.sync_weights(cpu_network)?;
        
        Ok(loss)
    }
    
    /// Performs a training step using SGD optimizer with options.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data [batch_size * input_dim].
    /// * `target` - Target data [batch_size * output_dim].
    /// * `mask` - Optional mask [batch_size * output_dim] (1.0 = active, 0.0 = ignore).
    /// * `batch_size` - Number of samples in the batch.
    /// * `workspace` - GPU workspace for training buffers.
    /// * `optimizer` - SGD optimizer.
    /// * `cpu_network` - CPU network to update weights on.
    /// * `opts` - Training options (gradient clipping, weight decay).
    ///
    /// # Returns
    ///
    /// The loss value for this batch.
    pub fn train_step_sgd_with_options(
        &mut self,
        input: &[f32],
        target: &[f32],
        mask: Option<&[f32]>,
        batch_size: usize,
        workspace: &mut GpuWorkspace,
        optimizer: &mut SGD,
        cpu_network: &mut KanNetwork,
        opts: &TrainOptions,
    ) -> ArkanResult<f32> {
        // Validate target shape
        let expected_target_len = batch_size * self.output_dim;
        if target.len() != expected_target_len {
            return Err(ArkanError::shape_mismatch(
                &[batch_size, self.output_dim],
                &[target.len() / self.output_dim, self.output_dim],
            ));
        }
        
        // 1. Forward pass with training data
        let output = self.forward_batch_training(input, batch_size, workspace)?;
        
        // 2. Compute loss and gradient
        let (loss, grad_output) = masked_mse(&output, target, mask);
        
        // 3. Backward pass
        let mut grad_weights = Vec::new();
        let mut grad_biases = Vec::new();
        let _grad_input = self.backward_batch(
            &grad_output,
            batch_size,
            workspace,
            &mut grad_weights,
            &mut grad_biases,
        )?;
        
        // 4. Apply weight decay override if specified in opts
        if opts.weight_decay > 0.0 {
            let lr = optimizer.lr;
            for layer in &mut cpu_network.layers {
                for w in layer.weights.as_mut_slice() {
                    *w *= 1.0 - lr * opts.weight_decay;
                }
            }
        }
        
        // 5. Optimizer step on CPU network with gradient clipping
        optimizer.step(cpu_network, &grad_weights, &grad_biases, opts.max_grad_norm);
        
        // 6. Sync weights from CPU to GPU
        self.sync_weights(cpu_network)?;
        
        Ok(loss)
    }
    
    /// Syncs weights from CPU network to GPU layers.
    ///
    /// Call this after modifying weights on the CPU side (e.g., after optimizer step).
    pub fn sync_weights_from_cpu(&mut self, cpu_network: &KanNetwork) -> ArkanResult<()> {
        self.sync_weights(cpu_network)
    }
    
    /// Syncs weights from GPU layers back to CPU network.
    ///
    /// This is an alias for `sync_weights_to_cpu` for naming consistency.
    pub fn sync_weights_gpu_to_cpu(&self, cpu_network: &mut KanNetwork) -> ArkanResult<()> {
        self.sync_weights_to_cpu(cpu_network)
    }
    
    /// Syncs weights from CPU network to GPU layers.
    ///
    /// This is an alias for `sync_weights_from_cpu` for naming consistency.
    pub fn sync_weights_cpu_to_gpu(&mut self, cpu_network: &KanNetwork) -> ArkanResult<()> {
        self.sync_weights(cpu_network)
    }
    
    /// Downloads weights from GPU to CPU network.
    ///
    /// Call this to get the trained weights back to the CPU after GPU training.
    pub fn sync_weights_to_cpu(&self, cpu_network: &mut KanNetwork) -> ArkanResult<()> {
        if cpu_network.layers.len() != self.layers.len() {
            return Err(ArkanError::validation(
                "CPU and GPU network layer count mismatch"
            ));
        }
        
        for (gpu_layer, cpu_layer) in self.layers.iter().zip(&mut cpu_network.layers) {
            // Download and unpack weights
            let packed = gpu_layer.weights.download(&self.device, &self.queue)?;
            
            // Unpack vec4 format to original format
            let unpacked = Self::unpack_weights_vec4(
                &packed,
                cpu_layer.out_dim,
                cpu_layer.in_dim,
                cpu_layer.global_basis_size,
                gpu_layer.basis_padded,
                gpu_layer.basis_vec4s,
            );
            
            cpu_layer.weights.copy_from_slice(&unpacked);
            
            // Download bias
            let bias_data = gpu_layer.bias.download(&self.device, &self.queue)?;
            cpu_layer.bias.copy_from_slice(&bias_data[..cpu_layer.out_dim]);
        }
        
        Ok(())
    }
    
    /// Unpacks weights from vec4 GPU format to original CPU format.
    fn unpack_weights_vec4(
        packed: &[f32],
        out_dim: usize,
        in_dim: usize,
        global_basis: usize,
        _basis_padded: usize,
        basis_vec4s: usize,
    ) -> Vec<f32> {
        let mut unpacked = vec![0.0f32; out_dim * in_dim * global_basis];
        
        for o in 0..out_dim {
            for i in 0..in_dim {
                let src_vec4_base = (o * in_dim + i) * basis_vec4s;
                let dst_offset = (o * in_dim + i) * global_basis;
                
                for k in 0..global_basis {
                    let src_offset = src_vec4_base * 4 + k;
                    unpacked[dst_offset + k] = packed[src_offset];
                }
            }
        }
        
        unpacked
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
