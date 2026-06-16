use super::GpuError;

/// Owned GPU device and queue. Created via `GpuContext::new`.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Initialise a GPU context synchronously (pollster::block_on).
    ///
    /// Returns `Err(GpuError::NotAvailable)` when no adapter is found (headless
    /// CI, macOS without Metal, etc.). Callers should handle this gracefully.
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(async {
            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .map_err(|e| GpuError::NotAvailable(e.to_string()))?;

            // Request native limits — the Hz compute pass uses 8 storage buffers,
            // which exceeds wgpu's conservative default of 8 only on some drivers.
            // Requesting adapter.limits() unlocks the hardware maximum (16+ on
            // Metal/Vulkan), making 8-storage-buffer passes safe on all targets.
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    required_limits: adapter.limits(),
                    ..Default::default()
                })
                .await
                .map_err(|e| GpuError::Internal(e.to_string()))?;

            Ok(Self { device, queue })
        })
    }
}
