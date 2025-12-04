//! GPU Optimizer implementations.
//!
//! This module provides GPU-accelerated optimizers that run weight updates
//! entirely on the GPU, avoiding expensive CPU-GPU transfers during training.
//!
//! # Overview
//!
//! Two optimizers are provided:
//! - [`GpuAdam`] - Adam optimizer with momentum and RMSprop
//! - [`GpuSgd`] - SGD with optional momentum
//!
//! Both optimizers use compute shaders to update weights directly on the GPU.

use crate::error::{ArkanError, ArkanResult};
use crate::gpu::shaders::{ADAM_SHADER, SGD_SHADER};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Uniform buffer for Adam optimizer shader.
///
/// This struct must match the AdamUniforms in ADAM_SHADER exactly (std140 layout).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AdamUniforms {
    /// Learning rate.
    pub lr: f32,
    /// First moment decay (beta1).
    pub beta1: f32,
    /// Second moment decay (beta2).
    pub beta2: f32,
    /// Epsilon for numerical stability.
    pub epsilon: f32,
    /// Weight decay (decoupled).
    pub weight_decay: f32,
    /// Bias correction factor for first moment: 1 / (1 - beta1^t).
    pub beta1_correction: f32,
    /// Bias correction factor for second moment: 1 / (1 - beta2^t).
    pub beta2_correction: f32,
    /// Number of parameters.
    pub num_params: u32,
}

impl AdamUniforms {
    /// Creates new Adam uniforms with bias correction for timestep t.
    pub fn new(
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        t: usize,
        num_params: usize,
    ) -> Self {
        let t = t.max(1) as i32;
        let beta1_correction = 1.0 / (1.0 - beta1.powi(t));
        let beta2_correction = 1.0 / (1.0 - beta2.powi(t));

        Self {
            lr,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            beta1_correction,
            beta2_correction,
            num_params: num_params as u32,
        }
    }
}

/// GPU Adam optimizer configuration.
#[derive(Debug, Clone, Copy)]
pub struct GpuAdamConfig {
    /// Learning rate (alpha).
    pub lr: f32,
    /// First moment decay (beta1).
    pub beta1: f32,
    /// Second moment decay (beta2).
    pub beta2: f32,
    /// Epsilon for numerical stability.
    pub epsilon: f32,
    /// Weight decay (decoupled, AdamW style).
    pub weight_decay: f32,
}

impl Default for GpuAdamConfig {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        }
    }
}

impl GpuAdamConfig {
    /// Creates config with specified learning rate.
    pub fn with_lr(lr: f32) -> Self {
        Self {
            lr,
            ..Default::default()
        }
    }

    /// Creates config with learning rate and weight decay (AdamW).
    pub fn with_decay(lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            weight_decay,
            ..Default::default()
        }
    }
}

/// GPU Adam optimizer state for a single parameter buffer.
///
/// Stores first moment (m) and second moment (v) on the GPU.
pub struct GpuAdamLayerState {
    /// First moment buffer (mean of gradients).
    pub m: wgpu::Buffer,
    /// Second moment buffer (mean of squared gradients).
    pub v: wgpu::Buffer,
    /// Number of parameters.
    pub num_params: usize,
}

impl GpuAdamLayerState {
    /// Creates new Adam state for a parameter buffer of given size.
    ///
    /// Both moments are initialized to zero.
    pub fn new(device: &wgpu::Device, num_params: usize) -> Self {
        let zeros = vec![0.0f32; num_params];

        let m = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Adam M Buffer"),
            contents: bytemuck::cast_slice(&zeros),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let v = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Adam V Buffer"),
            contents: bytemuck::cast_slice(&zeros),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        Self { m, v, num_params }
    }

    /// Resets the state to zero.
    pub fn reset(&self, queue: &wgpu::Queue) {
        let zeros = vec![0.0f32; self.num_params];
        queue.write_buffer(&self.m, 0, bytemuck::cast_slice(&zeros));
        queue.write_buffer(&self.v, 0, bytemuck::cast_slice(&zeros));
    }
}

/// GPU Adam optimizer for a complete network.
///
/// Maintains Adam state (m, v) for all weights and biases on the GPU,
/// along with cached pipeline and bind groups for efficient updates.
pub struct GpuAdam {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    /// Configuration.
    pub config: GpuAdamConfig,

    /// Adam states for each layer's weights.
    weight_states: Vec<GpuAdamLayerState>,
    /// Adam states for each layer's biases.
    bias_states: Vec<GpuAdamLayerState>,

    /// Timestep counter for bias correction.
    pub t: usize,

    /// Cached compute pipeline.
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout.
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuAdam {
    /// Creates a new GPU Adam optimizer for a network.
    ///
    /// # Arguments
    ///
    /// * `device` - wgpu device
    /// * `queue` - wgpu queue
    /// * `layer_sizes` - Vec of (num_weights, num_biases) for each layer
    /// * `config` - Adam configuration
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        layer_sizes: &[(usize, usize)],
        config: GpuAdamConfig,
    ) -> Self {
        // Create Adam states for each layer
        let mut weight_states = Vec::with_capacity(layer_sizes.len());
        let mut bias_states = Vec::with_capacity(layer_sizes.len());

        for &(num_weights, num_biases) in layer_sizes {
            weight_states.push(GpuAdamLayerState::new(&device, num_weights));
            bias_states.push(GpuAdamLayerState::new(&device, num_biases));
        }

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Adam Bind Group Layout"),
            entries: &[
                // params (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // grads (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // m (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // v (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // config (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Adam Shader"),
            source: wgpu::ShaderSource::Wgsl(ADAM_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Adam Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Adam Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("adam_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            config,
            weight_states,
            bias_states,
            t: 0,
            pipeline,
            bind_group_layout,
        }
    }

    /// Performs one Adam update step for a parameter buffer.
    ///
    /// This is an internal method that creates a bind group and dispatches
    /// the compute shader for a single parameter tensor.
    fn step_buffer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params: &wgpu::Buffer,
        grads: &wgpu::Buffer,
        state: &GpuAdamLayerState,
        uniforms_buffer: &wgpu::Buffer,
    ) {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Adam Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grads.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: state.m.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: state.v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniforms_buffer.as_entire_binding(),
                },
            ],
        });

        let workgroups = (state.num_params as u32).div_ceil(256);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Adam Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }

    /// Increments timestep and performs Adam update for all network parameters.
    ///
    /// # Arguments
    ///
    /// * `layer_params` - Vec of (weights_buffer, bias_buffer) for each layer
    /// * `layer_grads` - Vec of (weight_grads_buffer, bias_grads_buffer) for each layer
    ///
    /// # Note
    ///
    /// This submits GPU work and does NOT wait for completion.
    /// Call `queue.submit()` after this to actually execute.
    pub fn step(
        &mut self,
        layer_params: &[(&wgpu::Buffer, &wgpu::Buffer)],
        layer_grads: &[(&wgpu::Buffer, &wgpu::Buffer)],
    ) -> ArkanResult<()> {
        if layer_params.len() != self.weight_states.len() {
            return Err(ArkanError::shape_mismatch(
                &[self.weight_states.len()],
                &[layer_params.len()],
            ));
        }

        // Increment timestep
        self.t += 1;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Adam Encoder"),
            });

        // Process each layer
        for (i, ((weights, bias), (weight_grads, bias_grads))) in
            layer_params.iter().zip(layer_grads.iter()).enumerate()
        {
            // Create uniforms for weights
            let weight_uniforms = AdamUniforms::new(
                self.config.lr,
                self.config.beta1,
                self.config.beta2,
                self.config.epsilon,
                self.config.weight_decay,
                self.t,
                self.weight_states[i].num_params,
            );

            let weight_uniforms_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Adam Weight Uniforms"),
                        contents: bytemuck::bytes_of(&weight_uniforms),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            self.step_buffer(
                &mut encoder,
                weights,
                weight_grads,
                &self.weight_states[i],
                &weight_uniforms_buffer,
            );

            // Create uniforms for biases
            let bias_uniforms = AdamUniforms::new(
                self.config.lr,
                self.config.beta1,
                self.config.beta2,
                self.config.epsilon,
                0.0, // No weight decay on biases typically
                self.t,
                self.bias_states[i].num_params,
            );

            let bias_uniforms_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Adam Bias Uniforms"),
                        contents: bytemuck::bytes_of(&bias_uniforms),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            self.step_buffer(
                &mut encoder,
                bias,
                bias_grads,
                &self.bias_states[i],
                &bias_uniforms_buffer,
            );
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Resets optimizer state (moments and timestep).
    pub fn reset(&mut self) {
        self.t = 0;
        for state in &self.weight_states {
            state.reset(&self.queue);
        }
        for state in &self.bias_states {
            state.reset(&self.queue);
        }
    }

    /// Gets the current learning rate.
    pub fn get_lr(&self) -> f32 {
        self.config.lr
    }

    /// Sets the learning rate.
    pub fn set_lr(&mut self, lr: f32) {
        self.config.lr = lr;
    }
}

// =============================================================================
// SGD Optimizer
// =============================================================================

/// Uniform buffer for SGD optimizer shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SgdUniforms {
    /// Learning rate.
    pub lr: f32,
    /// Momentum coefficient.
    pub momentum: f32,
    /// Weight decay (decoupled).
    pub weight_decay: f32,
    /// Number of parameters.
    pub num_params: u32,
}

/// GPU SGD optimizer configuration.
#[derive(Debug, Clone, Copy)]
pub struct GpuSgdConfig {
    /// Learning rate.
    pub lr: f32,
    /// Momentum coefficient (0.0 for vanilla SGD).
    pub momentum: f32,
    /// Weight decay.
    pub weight_decay: f32,
}

impl Default for GpuSgdConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            momentum: 0.9,
            weight_decay: 0.0,
        }
    }
}

impl GpuSgdConfig {
    /// Creates config with specified learning rate.
    pub fn with_lr(lr: f32) -> Self {
        Self {
            lr,
            ..Default::default()
        }
    }
}

/// GPU SGD optimizer state for a single parameter buffer.
pub struct GpuSgdLayerState {
    /// Velocity buffer (momentum accumulator).
    pub velocity: wgpu::Buffer,
    /// Number of parameters.
    pub num_params: usize,
}

impl GpuSgdLayerState {
    /// Creates new SGD state for a parameter buffer.
    pub fn new(device: &wgpu::Device, num_params: usize) -> Self {
        let zeros = vec![0.0f32; num_params];

        let velocity = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SGD Velocity Buffer"),
            contents: bytemuck::cast_slice(&zeros),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            velocity,
            num_params,
        }
    }

    /// Resets velocity to zero.
    pub fn reset(&self, queue: &wgpu::Queue) {
        let zeros = vec![0.0f32; self.num_params];
        queue.write_buffer(&self.velocity, 0, bytemuck::cast_slice(&zeros));
    }
}

/// GPU SGD optimizer with momentum.
pub struct GpuSgd {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    /// Configuration.
    pub config: GpuSgdConfig,

    /// SGD states for each layer's weights.
    weight_states: Vec<GpuSgdLayerState>,
    /// SGD states for each layer's biases.
    bias_states: Vec<GpuSgdLayerState>,

    /// Cached compute pipeline.
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout.
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuSgd {
    /// Creates a new GPU SGD optimizer.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        layer_sizes: &[(usize, usize)],
        config: GpuSgdConfig,
    ) -> Self {
        let mut weight_states = Vec::with_capacity(layer_sizes.len());
        let mut bias_states = Vec::with_capacity(layer_sizes.len());

        for &(num_weights, num_biases) in layer_sizes {
            weight_states.push(GpuSgdLayerState::new(&device, num_weights));
            bias_states.push(GpuSgdLayerState::new(&device, num_biases));
        }

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SGD Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SGD Shader"),
            source: wgpu::ShaderSource::Wgsl(SGD_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SGD Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SGD Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("sgd_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            config,
            weight_states,
            bias_states,
            pipeline,
            bind_group_layout,
        }
    }

    /// Performs one SGD update step.
    pub fn step(
        &mut self,
        layer_params: &[(&wgpu::Buffer, &wgpu::Buffer)],
        layer_grads: &[(&wgpu::Buffer, &wgpu::Buffer)],
    ) -> ArkanResult<()> {
        if layer_params.len() != self.weight_states.len() {
            return Err(ArkanError::shape_mismatch(
                &[self.weight_states.len()],
                &[layer_params.len()],
            ));
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("SGD Encoder"),
            });

        for (i, ((weights, bias), (weight_grads, bias_grads))) in
            layer_params.iter().zip(layer_grads.iter()).enumerate()
        {
            // Update weights
            let weight_uniforms = SgdUniforms {
                lr: self.config.lr,
                momentum: self.config.momentum,
                weight_decay: self.config.weight_decay,
                num_params: self.weight_states[i].num_params as u32,
            };

            let weight_uniforms_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("SGD Weight Uniforms"),
                        contents: bytemuck::bytes_of(&weight_uniforms),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SGD Weight Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: weights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: weight_grads.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.weight_states[i].velocity.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: weight_uniforms_buffer.as_entire_binding(),
                    },
                ],
            });

            let workgroups = (self.weight_states[i].num_params as u32).div_ceil(256);

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SGD Weight Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            // Update biases
            let bias_uniforms = SgdUniforms {
                lr: self.config.lr,
                momentum: self.config.momentum,
                weight_decay: 0.0, // No weight decay on biases
                num_params: self.bias_states[i].num_params as u32,
            };

            let bias_uniforms_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("SGD Bias Uniforms"),
                        contents: bytemuck::bytes_of(&bias_uniforms),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SGD Bias Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: bias.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: bias_grads.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.bias_states[i].velocity.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: bias_uniforms_buffer.as_entire_binding(),
                    },
                ],
            });

            let workgroups = (self.bias_states[i].num_params as u32).div_ceil(256);

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SGD Bias Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Resets optimizer state.
    pub fn reset(&mut self) {
        for state in &self.weight_states {
            state.reset(&self.queue);
        }
        for state in &self.bias_states {
            state.reset(&self.queue);
        }
    }

    /// Gets the current learning rate.
    pub fn get_lr(&self) -> f32 {
        self.config.lr
    }

    /// Sets the learning rate.
    pub fn set_lr(&mut self, lr: f32) {
        self.config.lr = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_uniforms_size() {
        assert_eq!(std::mem::size_of::<AdamUniforms>(), 32);
    }

    #[test]
    fn test_adam_uniforms_bias_correction() {
        let uniforms = AdamUniforms::new(0.001, 0.9, 0.999, 1e-8, 0.0, 1, 100);

        // At t=1:
        // beta1_correction = 1 / (1 - 0.9^1) = 1 / 0.1 = 10.0
        // beta2_correction = 1 / (1 - 0.999^1) = 1 / 0.001 ≈ 1000.0
        // Note: f32 precision limits require larger tolerance for beta2_correction
        assert!((uniforms.beta1_correction - 10.0).abs() < 1e-5);
        assert!(
            (uniforms.beta2_correction - 1000.0).abs() < 10.0,
            "beta2_correction = {}, expected ~1000.0",
            uniforms.beta2_correction
        );
    }

    #[test]
    fn test_sgd_uniforms_size() {
        assert_eq!(std::mem::size_of::<SgdUniforms>(), 16);
    }

    #[test]
    fn test_gpu_adam_config_default() {
        let config = GpuAdamConfig::default();
        assert_eq!(config.lr, 0.001);
        assert_eq!(config.beta1, 0.9);
        assert_eq!(config.beta2, 0.999);
    }

    #[test]
    fn test_gpu_sgd_config_default() {
        let config = GpuSgdConfig::default();
        assert_eq!(config.lr, 0.01);
        assert_eq!(config.momentum, 0.9);
    }
}
