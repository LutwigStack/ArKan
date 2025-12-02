//! Full KAN network with multi-layer support.

use crate::buffer::Workspace;
use crate::config::KanConfig;
use crate::layer::KanLayer;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Complete KAN network with multiple layers.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KanNetwork {
    /// Configuration.
    pub config: KanConfig,
    
    /// Layers: input→hidden[0]→...→hidden[n]→output.
    pub layers: Vec<KanLayer>,
    
    /// Cached layer dimensions for quick access.
    layer_dims: Vec<usize>,
}

impl KanNetwork {
    /// Creates a new KAN network from configuration.
    pub fn new(config: KanConfig) -> Self {
        let layer_dims = config.layer_dims();
        let mut layers = Vec::with_capacity(layer_dims.len() - 1);
        
        for i in 0..layer_dims.len() - 1 {
            let in_dim = layer_dims[i];
            let out_dim = layer_dims[i + 1];
            layers.push(KanLayer::from_config(in_dim, out_dim, &config));
        }
        
        Self {
            config,
            layers,
            layer_dims,
        }
    }
    
    /// Creates network from configuration (alias).
    pub fn from_config(config: KanConfig) -> Self {
        Self::new(config)
    }
    
    /// Returns the number of layers.
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
    
    /// Returns total number of trainable parameters.
    pub fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.param_count()).sum()
    }
    
    /// Sets normalization statistics for the input layer.
    pub fn set_input_normalization(&mut self, mean: &[f32], std: &[f32]) {
        if !self.layers.is_empty() {
            self.layers[0].set_normalization(mean, std);
        }
    }
    
    /// Forward pass for a single sample.
    /// 
    /// # Arguments
    /// * `input` - Input features [input_dim]
    /// * `output` - Output buffer [output_dim]
    /// * `workspace` - Pre-allocated workspace
    pub fn forward_single(&self, input: &[f32], output: &mut [f32], workspace: &mut Workspace) {
        debug_assert_eq!(input.len(), self.config.input_dim);
        debug_assert_eq!(output.len(), self.config.output_dim);
        
        if self.layers.is_empty() {
            return;
        }
        
        // Reserve workspace for batch_size=1
        workspace.reserve(1, &self.config);
        
        // Calculate max dimension for ping-pong buffers
        let max_dim = self.layer_dims.iter().copied().max().unwrap_or(1);
        workspace.layer_output.resize(max_dim);
        workspace.layer_input.resize(max_dim);
        
        // Get max basis_aligned across all layers
        let max_basis = self.layers.iter().map(|l| l.basis_aligned).max().unwrap_or(8);
        workspace.basis_values.resize(max_basis);
        
        if self.layers.len() == 1 {
            // Single layer: input → output
            let basis_buf = &mut workspace.basis_values.as_mut_slice()[..self.layers[0].basis_aligned];
            self.layers[0].forward_single(input, output, basis_buf);
        } else {
            // Multi-layer: use ping-pong buffers
            let mut use_output_as_current = true;
            
            // First layer: input → layer_output
            {
                let layer = &self.layers[0];
                let out_slice = &mut workspace.layer_output.as_mut_slice()[..layer.out_dim];
                let basis_buf = &mut workspace.basis_values.as_mut_slice()[..layer.basis_aligned];
                layer.forward_single(input, out_slice, basis_buf);
            }
            
            // Hidden layers: ping-pong
            for i in 1..self.layers.len() - 1 {
                let layer = &self.layers[i];
                
                // Copy current to z_buffer to avoid borrow issues
                let in_size = layer.in_dim;
                workspace.z_buffer.resize(in_size);
                if use_output_as_current {
                    workspace.z_buffer.as_mut_slice().copy_from_slice(
                        &workspace.layer_output.as_slice()[..in_size]
                    );
                } else {
                    workspace.z_buffer.as_mut_slice().copy_from_slice(
                        &workspace.layer_input.as_slice()[..in_size]
                    );
                }
                
                // Forward to the other buffer
                let out_slice = if use_output_as_current {
                    &mut workspace.layer_input.as_mut_slice()[..layer.out_dim]
                } else {
                    &mut workspace.layer_output.as_mut_slice()[..layer.out_dim]
                };
                let basis_buf = &mut workspace.basis_values.as_mut_slice()[..layer.basis_aligned];
                layer.forward_single(workspace.z_buffer.as_slice(), out_slice, basis_buf);
                
                use_output_as_current = !use_output_as_current;
            }
            
            // Last layer: current buffer → output
            {
                let layer = self.layers.last().unwrap();
                let in_size = layer.in_dim;
                
                // Copy to z_buffer
                workspace.z_buffer.resize(in_size);
                if use_output_as_current {
                    workspace.z_buffer.as_mut_slice().copy_from_slice(
                        &workspace.layer_output.as_slice()[..in_size]
                    );
                } else {
                    workspace.z_buffer.as_mut_slice().copy_from_slice(
                        &workspace.layer_input.as_slice()[..in_size]
                    );
                }
                
                let basis_buf = &mut workspace.basis_values.as_mut_slice()[..layer.basis_aligned];
                layer.forward_single(workspace.z_buffer.as_slice(), output, basis_buf);
            }
        }
    }

    /// Forward pass for a batch of samples.
    /// 
    /// # Arguments
    /// * `input` - Input batch [batch_size * input_dim], Row-Major
    /// * `output` - Output buffer [batch_size * output_dim]
    /// * `workspace` - Pre-allocated workspace
    pub fn forward_batch(
        &self,
        input: &[f32],
        output: &mut [f32],
        workspace: &mut Workspace,
    ) {
        let batch_size = input.len() / self.config.input_dim;
        debug_assert_eq!(input.len(), batch_size * self.config.input_dim);
        debug_assert_eq!(output.len(), batch_size * self.config.output_dim);
        
        if self.layers.is_empty() {
            return;
        }
        
        workspace.reserve(batch_size, &self.config);
        
        if self.layers.len() == 1 {
            // Single layer network
            workspace.prepare_forward(batch_size, &self.config);
            self.layers[0].forward_batch(input, output, workspace);
        } else {
            // Multi-layer: use ping-pong buffers
            self.forward_batch_multilayer(input, output, workspace, batch_size);
        }
    }
    
    /// Multi-layer forward pass using ping-pong buffers.
    /// 
    /// Uses layer_output and layer_input as alternating buffers.
    /// Minimal allocations after initial reserve().
    fn forward_batch_multilayer(
        &self,
        input: &[f32],
        output: &mut [f32],
        workspace: &mut Workspace,
        batch_size: usize,
    ) {
        // Calculate max dimension for ping-pong buffers
        let max_hidden = self.layer_dims.iter()
            .copied()
            .max()
            .unwrap_or(self.config.input_dim);
        
        // Ensure ping-pong buffers are sized
        let ping_pong_size = batch_size * max_hidden;
        workspace.layer_output.resize(ping_pong_size);
        workspace.layer_input.resize(ping_pong_size);
        
        // Track which buffer currently holds the latest output
        let mut use_output_as_current = true;
        
        // First layer: external input → layer_output
        {
            let layer = &self.layers[0];
            let out_size = batch_size * layer.out_dim;
            
            // Use temporary vec for output to avoid borrow conflict
            let mut temp_out = vec![0.0f32; out_size];
            layer.forward_batch(input, &mut temp_out, workspace);
            workspace.layer_output.as_mut_slice()[..out_size].copy_from_slice(&temp_out);
        }
        
        // Hidden layers: ping-pong between layer_output and layer_input
        for i in 1..self.layers.len() - 1 {
            let layer = &self.layers[i];
            let in_size = batch_size * layer.in_dim;
            let out_size = batch_size * layer.out_dim;
            
            // Copy input to owned vec to avoid borrow issues
            let input_copy: Vec<f32> = if use_output_as_current {
                workspace.layer_output.as_slice()[..in_size].to_vec()
            } else {
                workspace.layer_input.as_slice()[..in_size].to_vec()
            };
            
            // Forward to temp, then copy to target buffer
            let mut temp_out = vec![0.0f32; out_size];
            layer.forward_batch(&input_copy, &mut temp_out, workspace);
            
            if use_output_as_current {
                workspace.layer_input.as_mut_slice()[..out_size].copy_from_slice(&temp_out);
            } else {
                workspace.layer_output.as_mut_slice()[..out_size].copy_from_slice(&temp_out);
            }
            
            use_output_as_current = !use_output_as_current;
        }
        
        // Last layer: current buffer → external output
        {
            let layer = self.layers.last().unwrap();
            let in_size = batch_size * layer.in_dim;
            
            // Copy to owned vec
            let input_copy: Vec<f32> = if use_output_as_current {
                workspace.layer_output.as_slice()[..in_size].to_vec()
            } else {
                workspace.layer_input.as_slice()[..in_size].to_vec()
            };
            
            layer.forward_batch(&input_copy, output, workspace);
        }
    }
    
    /// Full training step: forward + backward + update.
    /// 
    /// Returns the loss value.
    pub fn train_step(
        &mut self,
        input: &[f32],
        target: &[f32],
        mask: Option<&[f32]>,
        learning_rate: f32,
        workspace: &mut Workspace,
    ) -> f32 {
        let batch_size = input.len() / self.config.input_dim;
        let output_dim = self.config.output_dim;
        
        // Forward pass
        let mut predictions = vec![0.0f32; batch_size * output_dim];
        self.forward_batch(input, &mut predictions, workspace);
        
        // Compute loss and output gradient
        let (loss, output_grad) = compute_masked_mse_loss(
            &predictions,
            target,
            mask,
            batch_size,
            output_dim,
        );
        
        // Backward pass through all layers (simplified: no proper backprop yet)
        // For now, just update the last layer
        // TODO: Implement full backpropagation
        
        // Copy last layer input before mutable borrow
        let last_layer_input: Vec<f32> = if self.layers.len() == 1 {
            input.to_vec()
        } else {
            workspace.layer_input.as_slice().to_vec()
        };
        
        if let Some(layer) = self.layers.last_mut() {
            let mut weight_grad = vec![0.0f32; layer.weights.len()];
            let mut bias_grad = vec![0.0f32; layer.bias.len()];
            
            layer.backward(
                &last_layer_input,
                &output_grad,
                None,
                &mut weight_grad,
                &mut bias_grad,
                workspace,
            );
            
            // SGD update
            let scale = learning_rate / batch_size as f32;
            for (w, g) in layer.weights.as_mut_slice().iter_mut().zip(weight_grad.iter()) {
                *w -= scale * g;
            }
            for (b, g) in layer.bias.as_mut_slice().iter_mut().zip(bias_grad.iter()) {
                *b -= scale * g;
            }
        }
        
        loss
    }
    
    /// Creates a workspace sized for this network.
    pub fn create_workspace(&self, max_batch: usize) -> Workspace {
        let mut ws = Workspace::new(&self.config);
        ws.reserve(max_batch, &self.config);
        ws
    }
    
    /// Saves network to bytes using bincode.
    #[cfg(feature = "serde")]
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }
    
    /// Loads network from bytes.
    #[cfg(feature = "serde")]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }
}

impl Clone for KanNetwork {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            layers: self.layers.clone(),
            layer_dims: self.layer_dims.clone(),
        }
    }
}

/// Computes masked MSE loss and gradient.
fn compute_masked_mse_loss(
    predictions: &[f32],
    targets: &[f32],
    mask: Option<&[f32]>,
    batch_size: usize,
    output_dim: usize,
) -> (f32, Vec<f32>) {
    let mut loss = 0.0f32;
    let mut grad = vec![0.0f32; batch_size * output_dim];
    let mut count = 0.0f32;
    
    for b in 0..batch_size {
        for o in 0..output_dim {
            let idx = b * output_dim + o;
            let m = mask.map(|m| m[idx]).unwrap_or(1.0);
            
            if m > 0.0 {
                let diff = predictions[idx] - targets[idx];
                loss += m * diff * diff;
                grad[idx] = 2.0 * m * diff;
                count += m;
            }
        }
    }
    
    if count > 0.0 {
        loss /= count;
    }
    
    (loss, grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_network_creation() {
        let config = KanConfig::default_poker();
        let network = KanNetwork::new(config);
        
        // 21 → 64 → 64 → 24: 3 layers
        assert_eq!(network.num_layers(), 3);
        assert_eq!(network.layers[0].in_dim, 21);
        assert_eq!(network.layers[0].out_dim, 64);
        assert_eq!(network.layers[2].out_dim, 24);
    }
    
    #[test]
    fn test_network_forward_single() {
        let config = KanConfig::default_poker();
        let network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(1);
        
        let input = vec![0.5f32; 21];
        let mut output = vec![0.0f32; 24];
        
        network.forward_single(&input, &mut output, &mut workspace);
        
        assert!(output.iter().all(|&x| x.is_finite()));
    }
    
    #[test]
    fn test_network_forward_batch() {
        let config = KanConfig::default_poker();
        let network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(32);
        
        let batch_size = 16;
        let input: Vec<f32> = (0..batch_size * 21)
            .map(|i| (i as f32 * 0.01) % 1.0)
            .collect();
        let mut output = vec![0.0f32; batch_size * 24];
        
        network.forward_batch(&input, &mut output, &mut workspace);
        
        assert!(output.iter().all(|&x| x.is_finite()));
    }
    
    #[test]
    fn test_network_train_step() {
        let config = KanConfig::default_poker();
        let mut network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(16);
        
        let batch_size = 8;
        let input: Vec<f32> = vec![0.5; batch_size * 21];
        let target: Vec<f32> = vec![0.1; batch_size * 24];
        
        let loss1 = network.train_step(&input, &target, None, 0.01, &mut workspace);
        let loss2 = network.train_step(&input, &target, None, 0.01, &mut workspace);
        
        // Loss should decrease after training
        assert!(loss2 < loss1, "Loss should decrease: {} -> {}", loss1, loss2);
    }
    
    #[test]
    fn test_network_param_count() {
        let config = KanConfig {
            input_dim: 4,
            output_dim: 2,
            hidden_dims: vec![8],
            grid_size: 5,
            spline_order: 3,
            grid_range: (-1.0, 1.0),
            input_mean: vec![0.0; 4],
            input_std: vec![1.0; 4],
            multithreading_threshold: 1024,
            simd_width: 8,
        };
        
        let network = KanNetwork::new(config);
        
        // Layer 1: 4 → 8, basis_aligned=8
        // Weights: 8 * 4 * 8 = 256, Bias: 8 → 264
        // Layer 2: 8 → 2
        // Weights: 2 * 8 * 8 = 128, Bias: 2 → 130
        // Total: 394
        
        let params = network.param_count();
        assert!(params > 0);
        assert_eq!(params, 264 + 130);
    }
    
    #[test]
    fn test_single_layer_network() {
        let config = KanConfig {
            input_dim: 4,
            output_dim: 2,
            hidden_dims: vec![],  // No hidden layers
            grid_size: 5,
            spline_order: 3,
            grid_range: (-1.0, 1.0),
            input_mean: vec![0.0; 4],
            input_std: vec![1.0; 4],
            multithreading_threshold: 1024,
            simd_width: 8,
        };
        
        let network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(8);
        
        assert_eq!(network.num_layers(), 1);
        
        let input = vec![0.5f32; 4];
        let mut output = vec![0.0f32; 2];
        network.forward_single(&input, &mut output, &mut workspace);
        
        assert!(output.iter().all(|&x| x.is_finite()));
    }
}
