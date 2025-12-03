//! GPU Compute Pipeline management.
//!
//! This module provides compute pipeline creation and management for GPU operations.

use crate::error::ArkanResult;
use crate::gpu::layer::GpuLayer;
use crate::gpu::shaders;
use std::sync::Arc;

/// Cached compute pipelines for different operations.
pub struct PipelineCache {
    device: Arc<wgpu::Device>,
    /// Forward pass pipeline
    forward_pipeline: Option<wgpu::ComputePipeline>,
    /// Forward simple pipeline (alternative implementation)
    forward_simple_pipeline: Option<wgpu::ComputePipeline>,
    /// Softmax pipeline
    softmax_pipeline: Option<wgpu::ComputePipeline>,
    /// Pipeline layout for forward pass
    forward_layout: Option<wgpu::PipelineLayout>,
    /// Bind group layout for workspace (Group 1)
    workspace_layout: Option<wgpu::BindGroupLayout>,
}

impl PipelineCache {
    /// Creates a new pipeline cache.
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self {
            device,
            forward_pipeline: None,
            forward_simple_pipeline: None,
            softmax_pipeline: None,
            forward_layout: None,
            workspace_layout: None,
        }
    }

    /// Gets or creates the forward pass pipeline.
    pub fn get_forward_pipeline(
        &mut self,
        layer_layout: &wgpu::BindGroupLayout,
    ) -> ArkanResult<&wgpu::ComputePipeline> {
        if self.forward_pipeline.is_none() {
            self.create_forward_pipeline(layer_layout)?;
        }
        Ok(self.forward_pipeline.as_ref().unwrap())
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
            self.workspace_layout = Some(GpuLayer::create_workspace_bind_group_layout(&self.device));
        }
        self.workspace_layout.as_ref().unwrap()
    }

    fn create_forward_pipeline(
        &mut self,
        layer_layout: &wgpu::BindGroupLayout,
    ) -> ArkanResult<()> {
        // Create workspace layout if needed
        if self.workspace_layout.is_none() {
            self.workspace_layout = Some(GpuLayer::create_workspace_bind_group_layout(&self.device));
        }

        // Create shader module
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Forward Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::FORWARD_SHADER.into()),
        });

        // Create pipeline layout
        let layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Forward Pipeline Layout"),
            bind_group_layouts: &[layer_layout, self.workspace_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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

    fn create_forward_simple_pipeline(
        &mut self,
        layer_layout: &wgpu::BindGroupLayout,
    ) -> ArkanResult<()> {
        // Create workspace layout if needed
        if self.workspace_layout.is_none() {
            self.workspace_layout = Some(GpuLayer::create_workspace_bind_group_layout(&self.device));
        }

        // Create shader module
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Forward Simple Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::FORWARD_SIMPLE_SHADER.into()),
        });

        // Create pipeline layout
        let layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Forward Simple Pipeline Layout"),
            bind_group_layouts: &[layer_layout, self.workspace_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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
}

impl std::fmt::Debug for PipelineCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineCache")
            .field("has_forward", &self.forward_pipeline.is_some())
            .field("has_forward_simple", &self.forward_simple_pipeline.is_some())
            .field("has_softmax", &self.softmax_pipeline.is_some())
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
