// GPU-accelerated FDTD solvers via wgpu compute shaders.
// Enabled only with the `gpu-wgpu` feature.

use thiserror::Error;

/// Error type for GPU operations.
#[derive(Debug, Error)]
pub enum GpuError {
    #[error("No GPU adapter available: {0}")]
    NotAvailable(String),
    #[error("GPU device error: {0}")]
    Internal(String),
}

pub mod buffers;
pub mod context;
pub mod fdtd_2d_gpu;
pub mod fdtd_2d_tm_gpu;

pub use fdtd_2d_gpu::Fdtd2dGpu;
pub use fdtd_2d_tm_gpu::Fdtd2dTmGpu;

pub mod fdtd_3d_gpu;
pub use fdtd_3d_gpu::Fdtd3dGpu;
