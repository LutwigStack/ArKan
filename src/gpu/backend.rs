//! GPU Backend initialization and device management.
//!
//! This module provides [`WgpuBackend`] for initializing and managing
//! the wgpu device and queue.

use crate::error::{ArkanError, ArkanResult};
use std::sync::Arc;

/// Power preference for GPU adapter selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PowerPreference {
    /// Prefer low power consumption (integrated GPU).
    LowPower,
    /// Prefer high performance (discrete GPU).
    #[default]
    HighPerformance,
}

impl From<PowerPreference> for wgpu::PowerPreference {
    fn from(pref: PowerPreference) -> Self {
        match pref {
            PowerPreference::LowPower => wgpu::PowerPreference::LowPower,
            PowerPreference::HighPerformance => wgpu::PowerPreference::HighPerformance,
        }
    }
}

/// Options for initializing the wgpu backend.
#[derive(Debug, Clone)]
pub struct WgpuOptions {
    /// Power preference for adapter selection.
    pub power_preference: PowerPreference,
    /// Preferred backend (Vulkan, DX12, Metal, etc.).
    /// If None, wgpu will auto-select the best available.
    pub backend: Option<wgpu::Backends>,
    /// Force a specific adapter by name (substring match).
    pub force_adapter_name: Option<String>,
    /// Required features.
    pub required_features: wgpu::Features,
    /// Required limits (minimum).
    pub required_limits: wgpu::Limits,
}

impl Default for WgpuOptions {
    fn default() -> Self {
        Self {
            power_preference: PowerPreference::HighPerformance,
            backend: None,
            force_adapter_name: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        }
    }
}

impl WgpuOptions {
    /// Creates options optimized for compute workloads.
    pub fn compute() -> Self {
        Self {
            power_preference: PowerPreference::HighPerformance,
            backend: None,
            force_adapter_name: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_storage_buffer_binding_size: 1 << 30, // 1GB
                max_buffer_size: 1 << 30,
                max_compute_workgroup_size_x: 256,
                max_compute_workgroup_size_y: 256,
                max_compute_workgroup_size_z: 64,
                max_compute_invocations_per_workgroup: 256,
                ..wgpu::Limits::default()
            },
        }
    }

    /// Creates options for low-memory environments.
    pub fn low_memory() -> Self {
        Self {
            power_preference: PowerPreference::LowPower,
            backend: None,
            force_adapter_name: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        }
    }
}

/// The main GPU backend struct.
///
/// Holds the wgpu device, queue, and adapter info. This is the entry point
/// for all GPU operations in ArKan.
///
/// # Example
///
/// ```rust,ignore
/// use arkan::gpu::{WgpuBackend, WgpuOptions};
///
/// let backend = WgpuBackend::init(WgpuOptions::default())?;
/// println!("Using GPU: {}", backend.adapter_info().name);
/// ```
pub struct WgpuBackend {
    /// The wgpu instance.
    pub instance: wgpu::Instance,
    /// The selected adapter.
    pub adapter: wgpu::Adapter,
    /// The wgpu device for resource creation.
    pub device: Arc<wgpu::Device>,
    /// The wgpu queue for command submission.
    pub queue: Arc<wgpu::Queue>,
    /// Cached adapter info.
    adapter_info: wgpu::AdapterInfo,
    /// Device limits.
    limits: wgpu::Limits,
}

impl WgpuBackend {
    /// Initializes the GPU backend with the given options.
    ///
    /// # Arguments
    ///
    /// * `options` - Configuration options for device selection.
    ///
    /// # Returns
    ///
    /// An initialized `WgpuBackend`, or an error if initialization fails.
    ///
    /// # Errors
    ///
    /// - `ArkanError::AdapterNotFound` - No suitable GPU adapter found.
    /// - `ArkanError::DeviceRequestFailed` - Failed to create device with requested limits.
    pub fn init(options: WgpuOptions) -> ArkanResult<Self> {
        // Create instance
        let backends = options.backend.unwrap_or(wgpu::Backends::all());
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        // Request adapter
        let adapter = pollster::block_on(Self::request_adapter(&instance, &options))?;
        let adapter_info = adapter.get_info();

        log::info!(
            "Selected GPU adapter: {} ({:?})",
            adapter_info.name,
            adapter_info.backend
        );

        // Check limits
        let adapter_limits = adapter.limits();
        Self::check_limits(&adapter_limits, &options.required_limits)?;

        // Request device
        let (device, queue) = pollster::block_on(Self::request_device(&adapter, &options))?;

        let limits = device.limits();

        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
            limits,
        })
    }

    async fn request_adapter(
        instance: &wgpu::Instance,
        options: &WgpuOptions,
    ) -> ArkanResult<wgpu::Adapter> {
        // First, try with the specified power preference
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: options.power_preference.into(),
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await;

        let adapter = match adapter {
            Some(a) => a,
            None => {
                // Try any available adapter
                instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::None,
                        compatible_surface: None,
                        force_fallback_adapter: false,
                    })
                    .await
                    .ok_or_else(|| {
                        ArkanError::adapter_not_found(
                            "No GPU adapters available. Ensure GPU drivers are installed.",
                        )
                    })?
            }
        };

        // Check if we need to filter by name
        if let Some(ref name_filter) = options.force_adapter_name {
            let info = adapter.get_info();
            if !info
                .name
                .to_lowercase()
                .contains(&name_filter.to_lowercase())
            {
                // Try to find adapter by name
                let adapters: Vec<_> = instance
                    .enumerate_adapters(wgpu::Backends::all())
                    .into_iter()
                    .collect();
                for a in adapters {
                    let info = a.get_info();
                    if info
                        .name
                        .to_lowercase()
                        .contains(&name_filter.to_lowercase())
                    {
                        return Ok(a);
                    }
                }
                return Err(ArkanError::adapter_not_found(format!(
                    "No adapter matching '{}' found",
                    name_filter
                )));
            }
        }

        Ok(adapter)
    }

    async fn request_device(
        adapter: &wgpu::Adapter,
        options: &WgpuOptions,
    ) -> ArkanResult<(wgpu::Device, wgpu::Queue)> {
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("ArKan GPU Device"),
                    required_features: options.required_features,
                    required_limits: options.required_limits.clone(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await?;

        Ok((device, queue))
    }

    fn check_limits(adapter: &wgpu::Limits, required: &wgpu::Limits) -> ArkanResult<()> {
        // Check critical limits for compute
        if adapter.max_storage_buffer_binding_size < required.max_storage_buffer_binding_size {
            return Err(ArkanError::unsupported_limits(format!(
                "max_storage_buffer_binding_size: adapter has {}, required {}",
                adapter.max_storage_buffer_binding_size, required.max_storage_buffer_binding_size
            )));
        }

        if adapter.max_buffer_size < required.max_buffer_size {
            return Err(ArkanError::unsupported_limits(format!(
                "max_buffer_size: adapter has {}, required {}",
                adapter.max_buffer_size, required.max_buffer_size
            )));
        }

        if adapter.max_compute_workgroup_size_x < required.max_compute_workgroup_size_x {
            return Err(ArkanError::unsupported_limits(format!(
                "max_compute_workgroup_size_x: adapter has {}, required {}",
                adapter.max_compute_workgroup_size_x, required.max_compute_workgroup_size_x
            )));
        }

        Ok(())
    }

    /// Returns information about the selected adapter.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Returns the device limits.
    pub fn limits(&self) -> &wgpu::Limits {
        &self.limits
    }

    /// Returns the maximum storage buffer size in bytes.
    pub fn max_storage_buffer_size(&self) -> u64 {
        self.limits.max_storage_buffer_binding_size as u64
    }

    /// Returns whether a buffer size (in bytes) is supported.
    pub fn supports_buffer_size(&self, size_bytes: u64) -> bool {
        size_bytes <= self.max_storage_buffer_size()
    }

    /// Validates that a layer's weight buffer can fit in GPU memory.
    ///
    /// # Arguments
    ///
    /// * `out_dim` - Output dimension of the layer.
    /// * `in_dim` - Input dimension of the layer.
    /// * `global_basis_size` - Number of basis functions (grid_size + order).
    ///
    /// # Note
    ///
    /// This calculates the actual GPU buffer size including vec4 padding:
    /// - basis_padded = align4(global_basis_size)
    /// - basis_vec4s = ceil(basis_padded / 4)
    /// - GPU buffer size = out_dim * in_dim * basis_vec4s * 4 floats
    pub fn validate_layer_weights(
        &self,
        out_dim: usize,
        in_dim: usize,
        global_basis_size: usize,
    ) -> ArkanResult<()> {
        // Calculate padded GPU buffer size (same as GpuLayer::from_cpu_layer)
        let basis_padded = crate::gpu::pad_to_vec4(global_basis_size);
        let basis_vec4s = (basis_padded + 3) / 4;
        let gpu_weight_count = out_dim * in_dim * basis_vec4s * 4;
        let size_bytes = (gpu_weight_count * std::mem::size_of::<f32>()) as u64;

        if !self.supports_buffer_size(size_bytes) {
            return Err(ArkanError::unsupported_limits(format!(
                "Layer weights ({}x{}x{} -> {} vec4s, {} bytes) exceed max storage buffer size ({} bytes)",
                out_dim, in_dim, global_basis_size,
                gpu_weight_count / 4,
                size_bytes,
                self.max_storage_buffer_size()
            )));
        }
        Ok(())
    }

    /// Polls the device for completed operations.
    pub fn poll(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Returns a reference to the device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Returns a reference to the queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Returns a clone of the device Arc.
    pub fn device_arc(&self) -> Arc<wgpu::Device> {
        Arc::clone(&self.device)
    }

    /// Returns a clone of the queue Arc.
    pub fn queue_arc(&self) -> Arc<wgpu::Queue> {
        Arc::clone(&self.queue)
    }
}

impl std::fmt::Debug for WgpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuBackend")
            .field("adapter", &self.adapter_info.name)
            .field("backend", &self.adapter_info.backend)
            .field("device_type", &self.adapter_info.device_type)
            .field(
                "max_storage_buffer",
                &self.limits.max_storage_buffer_binding_size,
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_options_default() {
        let opts = WgpuOptions::default();
        assert_eq!(opts.power_preference, PowerPreference::HighPerformance);
        assert!(opts.backend.is_none());
    }

    #[test]
    fn test_options_compute() {
        let opts = WgpuOptions::compute();
        assert_eq!(opts.power_preference, PowerPreference::HighPerformance);
        assert!(opts.required_limits.max_storage_buffer_binding_size >= 1 << 30);
    }

    // GPU tests require actual GPU, run with: cargo test --features gpu -- --ignored
    #[test]
    #[ignore]
    fn test_backend_init() {
        let backend = WgpuBackend::init(WgpuOptions::default()).expect("Failed to init backend");
        println!("Adapter: {:?}", backend.adapter_info());
        assert!(!backend.adapter_info().name.is_empty());
    }
}
