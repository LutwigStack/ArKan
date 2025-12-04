//! GPU Compute Pipeline management.
//!
//! This module provides compute pipeline creation and management for GPU operations.

use crate::error::ArkanResult;
use crate::gpu::layer::GpuLayer;
use crate::gpu::shaders;
use std::collections::HashMap;
use std::sync::Arc;

/// Cached compute pipelines for different operations.
pub struct PipelineCache {
    device: Arc<wgpu::Device>,
    /// Forward pass pipelines cached by spline order (2-5)
    forward_pipelines: HashMap<usize, wgpu::ComputePipeline>,
    /// Forward training pipelines cached by spline order (2-5)
    forward_training_pipelines: HashMap<usize, wgpu::ComputePipeline>,
    /// Forward pass pipeline (legacy, order=3 only)
    forward_pipeline: Option<wgpu::ComputePipeline>,
    /// Forward simple pipeline (alternative implementation)
    forward_simple_pipeline: Option<wgpu::ComputePipeline>,
    /// Softmax pipeline
    softmax_pipeline: Option<wgpu::ComputePipeline>,
    /// Softmax bind group layout
    softmax_layout: Option<wgpu::BindGroupLayout>,
    /// Pipeline layout for forward pass
    forward_layout: Option<wgpu::PipelineLayout>,
    /// Bind group layout for workspace (Group 1)
    workspace_layout: Option<wgpu::BindGroupLayout>,

    // Training pipelines
    /// Forward training pipeline (saves z_values and span_indices)
    forward_training_pipeline: Option<wgpu::ComputePipeline>,
    /// Backward weights pipeline
    backward_weights_pipeline: Option<wgpu::ComputePipeline>,
    /// Backward input gradients pipeline
    backward_input_pipeline: Option<wgpu::ComputePipeline>,
    /// Backward bias pipeline
    backward_bias_pipeline: Option<wgpu::ComputePipeline>,
    /// Bind group layout for training (Group 1)
    training_workspace_layout: Option<wgpu::BindGroupLayout>,
    /// Bind group layout for backward pass (Group 1)
    backward_workspace_layout: Option<wgpu::BindGroupLayout>,
}

impl PipelineCache {
    /// Creates a new pipeline cache.
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self {
            device,
            forward_pipelines: HashMap::new(),
            forward_training_pipelines: HashMap::new(),
            forward_pipeline: None,
            forward_simple_pipeline: None,
            softmax_pipeline: None,
            softmax_layout: None,
            forward_layout: None,
            workspace_layout: None,
            forward_training_pipeline: None,
            backward_weights_pipeline: None,
            backward_input_pipeline: None,
            backward_bias_pipeline: None,
            training_workspace_layout: None,
            backward_workspace_layout: None,
        }
    }

    /// Gets or creates the forward pass pipeline for a specific spline order.
    ///
    /// Uses dynamically generated WGSL shaders optimized for the given order.
    /// Pipelines are cached by order for reuse.
    pub fn get_forward_pipeline_for_order(
        &mut self,
        layer_layout: &wgpu::BindGroupLayout,
        order: usize,
    ) -> ArkanResult<&wgpu::ComputePipeline> {
        // Check cache first
        if !self.forward_pipelines.contains_key(&order) {
            self.create_forward_pipeline_for_order(layer_layout, order)?;
        }
        Ok(self.forward_pipelines.get(&order).unwrap())
    }

    /// Gets or creates the forward pass pipeline (legacy, order=3 only).
    pub fn get_forward_pipeline(
        &mut self,
        layer_layout: &wgpu::BindGroupLayout,
    ) -> ArkanResult<&wgpu::ComputePipeline> {
        if self.forward_pipeline.is_none() {
            self.create_forward_pipeline(layer_layout)?;
        }
        Ok(self.forward_pipeline.as_ref().unwrap())
    }

    /// Gets or creates the forward training pipeline for a specific spline order.
    pub fn get_forward_training_pipeline_for_order(
        &mut self,
        layer_layout: &wgpu::BindGroupLayout,
        order: usize,
    ) -> ArkanResult<&wgpu::ComputePipeline> {
        if !self.forward_training_pipelines.contains_key(&order) {
            self.create_forward_training_pipeline_for_order(layer_layout, order)?;
        }
        Ok(self.forward_training_pipelines.get(&order).unwrap())
    }

    /// Gets or creates the forward simple pipeline.
    pub fn get_forward_simple_pipeline(
        &mut self,
        layer_layout: &wgpu::BindGroupLayout,
    ) -> ArkanResult<&wgpu::ComputePipeline> {
        if self.forward_simple_pipeline.is_none() {
            self.create_forward_simple_pipeline(layer_layout)?;
        }
        Ok(self.forward_simple_pipeline.as_ref().unwrap())
    }

    /// Gets the workspace bind group layout.
    pub fn get_workspace_layout(&mut self) -> &wgpu::BindGroupLayout {
        if self.workspace_layout.is_none() {
            self.workspace_layout =
                Some(GpuLayer::create_workspace_bind_group_layout(&self.device));
        }
        self.workspace_layout.as_ref().unwrap()
    }

    fn create_forward_pipeline(&mut self, layer_layout: &wgpu::BindGroupLayout) -> ArkanResult<()> {
        // Create workspace layout if needed
        if self.workspace_layout.is_none() {
            self.workspace_layout =
                Some(GpuLayer::create_workspace_bind_group_layout(&self.device));
        }

        // Create shader module
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Forward Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::FORWARD_SHADER.into()),
            });

        // Create pipeline layout
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Forward Pipeline Layout"),
                bind_group_layouts: &[layer_layout, self.workspace_layout.as_ref().unwrap()],
                push_constant_ranges: &[],
            });

        // Create compute pipeline
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Forward Pipeline"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("forward_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        self.forward_layout = Some(layout);
        self.forward_pipeline = Some(pipeline);

        Ok(())
    }

    /// Creates a forward pipeline for a specific spline order using dynamic shader generation.
    fn create_forward_pipeline_for_order(
        &mut self,
        layer_layout: &wgpu::BindGroupLayout,
        order: usize,
    ) -> ArkanResult<()> {
        // Create workspace layout if needed
        if self.workspace_layout.is_none() {
            self.workspace_layout =
                Some(GpuLayer::create_workspace_bind_group_layout(&self.device));
        }

        // Generate shader for specific order
        let shader_source = shaders::generate_forward_shader(order)?;

        // Create shader module
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("Forward Shader Order {}", order)),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        // Create or reuse pipeline layout
        if self.forward_layout.is_none() {
            let layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Forward Pipeline Layout"),
                    bind_group_layouts: &[layer_layout, self.workspace_layout.as_ref().unwrap()],
                    push_constant_ranges: &[],
                });
            self.forward_layout = Some(layout);
        }

        // Create compute pipeline
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("Forward Pipeline Order {}", order)),
                layout: Some(self.forward_layout.as_ref().unwrap()),
                module: &shader,
                entry_point: Some("forward_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        self.forward_pipelines.insert(order, pipeline);

        Ok(())
    }

    fn create_forward_simple_pipeline(
        &mut self,
        layer_layout: &wgpu::BindGroupLayout,
    ) -> ArkanResult<()> {
        // Create workspace layout if needed
        if self.workspace_layout.is_none() {
            self.workspace_layout =
                Some(GpuLayer::create_workspace_bind_group_layout(&self.device));
        }

        // Create shader module
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Forward Simple Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::FORWARD_SIMPLE_SHADER.into()),
            });

        // Create pipeline layout
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Forward Simple Pipeline Layout"),
                bind_group_layouts: &[layer_layout, self.workspace_layout.as_ref().unwrap()],
                push_constant_ranges: &[],
            });

        // Create compute pipeline
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Forward Simple Pipeline"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("forward_simple"),
                compilation_options: Default::default(),
                cache: None,
            });

        self.forward_simple_pipeline = Some(pipeline);

        Ok(())
    }

    /// Returns the forward pipeline layout.
    pub fn forward_layout(&self) -> Option<&wgpu::PipelineLayout> {
        self.forward_layout.as_ref()
    }

    // ==================== Softmax Pipeline ====================

    /// Gets the softmax bind group layout (creates if needed).
    pub fn get_softmax_layout(&mut self) -> &wgpu::BindGroupLayout {
        if self.softmax_layout.is_none() {
            self.softmax_layout = Some(self.device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("Softmax BindGroupLayout"),
                    entries: &[
                        // binding 0: config (uniform)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 1: data (read-write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                },
            ));
        }
        self.softmax_layout.as_ref().unwrap()
    }

    /// Gets or creates the softmax pipeline.
    ///
    /// The softmax pipeline normalizes outputs to probability distribution:
    /// softmax(x_i) = exp(x_i) / sum(exp(x_j))
    ///
    /// Uses bind group layout with uniform config and read-write data buffer.
    pub fn get_softmax_pipeline(&mut self) -> ArkanResult<&wgpu::ComputePipeline> {
        if self.softmax_pipeline.is_none() {
            self.create_softmax_pipeline()?;
        }
        Ok(self.softmax_pipeline.as_ref().unwrap())
    }

    fn create_softmax_pipeline(&mut self) -> ArkanResult<()> {
        // Ensure layout is created first
        let _ = self.get_softmax_layout();
        let softmax_layout = self.softmax_layout.as_ref().unwrap();

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Softmax Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::SOFTMAX_SHADER.into()),
            });

        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Softmax Pipeline Layout"),
                bind_group_layouts: &[softmax_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Softmax Pipeline"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("softmax_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        self.softmax_pipeline = Some(pipeline);
        Ok(())
    }

    // ==================== Training Pipelines ====================

    /// Creates the bind group layout for training forward pass.
    /// Group 1: input, output, z_values, span_indices
    pub fn get_training_workspace_layout(&mut self) -> &wgpu::BindGroupLayout {
        if self.training_workspace_layout.is_none() {
            self.training_workspace_layout = Some(self.device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("Training Workspace BindGroupLayout"),
                    entries: &[
                        // binding 0: input (read)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 1: output (write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 2: z_values (write)
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
                        // binding 3: span_indices (write)
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
                    ],
                },
            ));
        }
        self.training_workspace_layout.as_ref().unwrap()
    }

    /// Creates the bind group layout for backward pass.
    /// Group 1: z_values, span_indices, grad_output, grad_weights, grad_bias, grad_input, std_inv
    pub fn get_backward_workspace_layout(&mut self) -> &wgpu::BindGroupLayout {
        if self.backward_workspace_layout.is_none() {
            self.backward_workspace_layout = Some(self.device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("Backward Workspace BindGroupLayout"),
                    entries: &[
                        // binding 0: z_values (read)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 1: span_indices (read)
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
                        // binding 2: grad_output (read)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 3: grad_weights (read-write)
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
                        // binding 4: grad_bias (read-write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 5: grad_input (read-write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 6: std_inv (read)
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                },
            ));
        }
        self.backward_workspace_layout.as_ref().unwrap()
    }

    /// Gets or creates the forward training pipeline.
    pub fn get_forward_training_pipeline(
        &mut self,
        layer_layout: &wgpu::BindGroupLayout,
    ) -> ArkanResult<&wgpu::ComputePipeline> {
        if self.forward_training_pipeline.is_none() {
            self.create_forward_training_pipeline(layer_layout)?;
        }
        Ok(self.forward_training_pipeline.as_ref().unwrap())
    }

    fn create_forward_training_pipeline(
        &mut self,
        layer_layout: &wgpu::BindGroupLayout,
    ) -> ArkanResult<()> {
        let training_layout = self.get_training_workspace_layout();
        // Need to clone since we mutably borrow self above
        let training_layout_ref = unsafe { &*(training_layout as *const _) };

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Forward Training Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::FORWARD_TRAINING_SHADER.into()),
            });

        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Forward Training Pipeline Layout"),
                bind_group_layouts: &[layer_layout, training_layout_ref],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Forward Training Pipeline"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("forward_training_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        self.forward_training_pipeline = Some(pipeline);
        Ok(())
    }

    /// Creates a forward training pipeline for a specific spline order.
    fn create_forward_training_pipeline_for_order(
        &mut self,
        layer_layout: &wgpu::BindGroupLayout,
        order: usize,
    ) -> ArkanResult<()> {
        let training_layout = self.get_training_workspace_layout();
        let training_layout_ref = unsafe { &*(training_layout as *const _) };

        let shader_source = shaders::generate_forward_training_shader(order)?;
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("Forward Training Shader Order {}", order)),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Forward Training Pipeline Layout"),
                bind_group_layouts: &[layer_layout, training_layout_ref],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("Forward Training Pipeline Order {}", order)),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("forward_training_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        self.forward_training_pipelines.insert(order, pipeline);
        Ok(())
    }

    /// Gets or creates the backward weights pipeline.
    pub fn get_backward_weights_pipeline(
        &mut self,
        layer_layout: &wgpu::BindGroupLayout,
    ) -> ArkanResult<&wgpu::ComputePipeline> {
        if self.backward_weights_pipeline.is_none() {
            self.create_backward_weights_pipeline(layer_layout)?;
        }
        Ok(self.backward_weights_pipeline.as_ref().unwrap())
    }

    fn create_backward_weights_pipeline(
        &mut self,
        layer_layout: &wgpu::BindGroupLayout,
    ) -> ArkanResult<()> {
        let backward_layout = self.get_backward_workspace_layout();
        // Need to clone since we mutably borrow self above
        let backward_layout_ref = unsafe { &*(backward_layout as *const _) };

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Backward Weights Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::BACKWARD_WEIGHTS_SHADER.into()),
            });

        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Backward Weights Pipeline Layout"),
                bind_group_layouts: &[layer_layout, backward_layout_ref],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Backward Weights Pipeline"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("backward_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        self.backward_weights_pipeline = Some(pipeline);
        Ok(())
    }

    /// Gets or creates the backward input pipeline.
    pub fn get_backward_input_pipeline(
        &mut self,
        layer_layout: &wgpu::BindGroupLayout,
    ) -> ArkanResult<&wgpu::ComputePipeline> {
        if self.backward_input_pipeline.is_none() {
            self.create_backward_input_pipeline(layer_layout)?;
        }
        Ok(self.backward_input_pipeline.as_ref().unwrap())
    }

    fn create_backward_input_pipeline(
        &mut self,
        layer_layout: &wgpu::BindGroupLayout,
    ) -> ArkanResult<()> {
        let backward_layout = self.get_backward_workspace_layout();
        let backward_layout_ref = unsafe { &*(backward_layout as *const _) };

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Backward Input Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::BACKWARD_INPUT_SHADER.into()),
            });

        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Backward Input Pipeline Layout"),
                bind_group_layouts: &[layer_layout, backward_layout_ref],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Backward Input Pipeline"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("backward_input_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        self.backward_input_pipeline = Some(pipeline);
        Ok(())
    }

    /// Gets or creates the backward bias pipeline.
    pub fn get_backward_bias_pipeline(
        &mut self,
        bias_layout: &wgpu::BindGroupLayout,
    ) -> ArkanResult<&wgpu::ComputePipeline> {
        if self.backward_bias_pipeline.is_none() {
            self.create_backward_bias_pipeline(bias_layout)?;
        }
        Ok(self.backward_bias_pipeline.as_ref().unwrap())
    }

    fn create_backward_bias_pipeline(
        &mut self,
        bias_layout: &wgpu::BindGroupLayout,
    ) -> ArkanResult<()> {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Backward Bias Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::BACKWARD_BIAS_SHADER.into()),
            });

        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Backward Bias Pipeline Layout"),
                bind_group_layouts: &[bias_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Backward Bias Pipeline"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("backward_bias_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        self.backward_bias_pipeline = Some(pipeline);
        Ok(())
    }

    /// Creates bias bind group layout for backward pass.
    pub fn create_bias_bind_group_layout(&self) -> wgpu::BindGroupLayout {
        self.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bias Backward BindGroupLayout"),
                entries: &[
                    // binding 0: uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 1: grad_output (read)
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
                    // binding 2: grad_bias (write)
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
                ],
            })
    }
}

impl std::fmt::Debug for PipelineCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineCache")
            .field(
                "forward_orders_cached",
                &self.forward_pipelines.keys().collect::<Vec<_>>(),
            )
            .field("has_forward", &self.forward_pipeline.is_some())
            .field(
                "has_forward_simple",
                &self.forward_simple_pipeline.is_some(),
            )
            .field("has_softmax", &self.softmax_pipeline.is_some())
            .field(
                "has_forward_training",
                &self.forward_training_pipeline.is_some(),
            )
            .field(
                "has_backward_weights",
                &self.backward_weights_pipeline.is_some(),
            )
            .field(
                "has_backward_input",
                &self.backward_input_pipeline.is_some(),
            )
            .field("has_backward_bias", &self.backward_bias_pipeline.is_some())
            .finish()
    }
}

/// Computes the workgroup count for a given number of elements.
#[inline]
pub fn workgroup_count(total: usize, workgroup_size: usize) -> u32 {
    ((total + workgroup_size - 1) / workgroup_size) as u32
}

/// Default workgroup size for compute shaders.
pub const WORKGROUP_SIZE: usize = 64;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workgroup_count() {
        assert_eq!(workgroup_count(1, 64), 1);
        assert_eq!(workgroup_count(64, 64), 1);
        assert_eq!(workgroup_count(65, 64), 2);
        assert_eq!(workgroup_count(128, 64), 2);
        assert_eq!(workgroup_count(129, 64), 3);
    }
}
