use bytemuck::{Pod, Zeroable};

use super::GpuError;

// ── Uniform struct layouts ────────────────────────────────────────────────────

/// Per-step source injection uniform.
/// Padded to 16 bytes for std140 compatibility.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SrcUniform {
    pub flat_idx: u32, // j * nx + i
    pub _p0: u32,
    pub _p1: u32,
    pub val: f32,
}

/// Simulation grid dimensions and cell spacing, used by all compute passes.
/// 32 bytes (2 × 16-byte blocks).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SimDims {
    pub nx: u32,
    pub ny: u32,
    pub dx: f32,
    pub dy: f32,
    pub dt: f32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// 3D simulation dimension + spacing uniform (32 bytes, two 16-byte blocks).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimDims3d {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub dx: f32,
    pub dy: f32,
    pub dz: f32,
    pub dt: f32,
    pub _p0: u32,
}

// ── Buffer helpers ─────────────────────────────────────────────────────────

/// Create a GPU storage buffer initialised from data, read-only from the shader.
pub fn storage_init(device: &wgpu::Device, data: &[f32], label: &str) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

/// Create a GPU storage buffer initialised from data, writable by CPU (COPY_DST).
pub fn storage_init_updatable(device: &wgpu::Device, data: &[f32], label: &str) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

/// Create a GPU storage buffer initialised to zero, read/write from shader,
/// readable back to CPU (COPY_SRC).
pub fn storage_rw_field(device: &wgpu::Device, n: usize, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Create a GPU storage buffer for auxiliary (ψ) arrays: read/write, no CPU readback needed.
pub fn storage_rw_aux(device: &wgpu::Device, n: usize, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

/// Create a uniform buffer from a Pod struct.
pub fn uniform_from<T: Pod>(device: &wgpu::Device, value: &T, label: &str) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(value),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

/// Create a CPU-readable staging buffer.
pub fn staging(device: &wgpu::Device, byte_size: u64, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// BindGroupLayoutEntry for a storage buffer.
pub fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// BindGroupLayoutEntry for a uniform buffer.
pub fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Read back a GPU storage buffer (with COPY_SRC usage) to `Vec<f32>`.
pub fn readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    byte_size: u64,
    label: &str,
) -> Result<Vec<f32>, GpuError> {
    let stg = staging(device, byte_size, label);
    let mut enc =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    enc.copy_buffer_to_buffer(src, 0, &stg, 0, byte_size);
    queue.submit(std::iter::once(enc.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    stg.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| GpuError::Internal(format!("poll: {e}")))?;
    rx.recv()
        .map_err(|e| GpuError::Internal(format!("recv: {e}")))?
        .map_err(|e| GpuError::Internal(format!("map_async: {e}")))?;

    let mapped = stg.slice(..).get_mapped_range();
    let floats: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    stg.unmap();
    Ok(floats)
}
