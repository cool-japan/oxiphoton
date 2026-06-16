use crate::fdtd::boundary::pml::Cpml;
use crate::fdtd::config::{BoundaryConfig, Dimensions, GridSpacing};
use crate::fdtd::courant::courant_dt;

use super::{
    buffers::{
        readback, storage_entry, storage_init, storage_rw_aux, storage_rw_field, uniform_entry,
        uniform_from, SimDims3d, SrcUniform,
    },
    context::GpuContext,
    GpuError,
};

const SHADER_INJECT: &str = include_str!("shaders/fdtd3d_inject.wgsl");
const SHADER_H: &str = include_str!("shaders/fdtd3d_h.wgsl");
const SHADER_E: &str = include_str!("shaders/fdtd3d_e.wgsl");

/// 3D FDTD solver accelerated via wgpu compute shaders.
///
/// Fields (Ex, Ey, Ez, Hx, Hy, Hz) and CPML psi arrays reside on-GPU across
/// `step()` calls; no per-step readback. Six field components are packed as two
/// `vec4<f32>` buffers (E and H), with auxiliary psi arrays stored as stride-6
/// flat f32 buffers. Coefficients are precomputed in f64 (identical to `Fdtd3d`)
/// then cast to f32 for upload.
///
/// Step order: inject_ez → H update → E update.
///
/// # GPU availability
/// If no adapter is found, `new` returns `Err(GpuError::NotAvailable)`.
/// The caller is responsible for graceful fallback to `Fdtd3d`.
// GPU buffers are owned for lifetime management — bind groups hold wgpu
// references to them, so they must outlive the bind groups. The compiler
// cannot track this cross-API ownership, hence the allow.
#[allow(dead_code)]
pub struct Fdtd3dGpu {
    ctx: GpuContext,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dt: f64,
    pub time_step: usize,

    // Field buffers packed as vec4<f32> per cell (STORAGE | COPY_SRC for readback)
    // E: (ex, ey, ez, 0), H: (hx, hy, hz, 0)
    buf_e: wgpu::Buffer,
    buf_h: wgpu::Buffer,

    // psi auxiliary buffers: stride-6 per cell, STORAGE only
    // psi_h[6*idx + {0=hx_y,1=hx_z,2=hy_x,3=hy_z,4=hz_x,5=hz_y}]
    // psi_e[6*idx + {0=ex_y,1=ex_z,2=ey_x,3=ey_z,4=ez_x,5=ez_y}]
    buf_psi_h: wgpu::Buffer,
    buf_psi_e: wgpu::Buffer,

    // Material buffer: vec4<f32> per cell = (eps_r, mu_r, sigma_e, sigma_m)
    buf_mat: wgpu::Buffer,

    // PML coefficient buffers, stride-6: [b_e, c_e, kappa_e, b_h, c_h, kappa_h]
    buf_pml_x: wgpu::Buffer,
    buf_pml_y: wgpu::Buffer,
    buf_pml_z: wgpu::Buffer,

    // Uniform buffers
    buf_dims: wgpu::Buffer,
    buf_src: wgpu::Buffer,

    // Compute pipelines
    pipeline_inject: wgpu::ComputePipeline,
    pipeline_h: wgpu::ComputePipeline,
    pipeline_e: wgpu::ComputePipeline,

    // Bind groups (one per pipeline)
    bg_inject: wgpu::BindGroup,
    bg_h: wgpu::BindGroup,
    bg_e: wgpu::BindGroup,

    // Pending source for next step
    src_flat_idx: u32,
    src_val: f32,
}

impl Fdtd3dGpu {
    /// Create a GPU 3D FDTD solver with the same parameters as `Fdtd3d::new`.
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        boundary: &BoundaryConfig,
    ) -> Result<Self, GpuError> {
        let spacing = GridSpacing { dx, dy, dz };
        let dt = 0.99 * courant_dt(Dimensions::ThreeD { nx, ny, nz }, spacing, 1.0);

        let pml_x = Cpml::new(
            nx,
            boundary.pml_cells,
            dx,
            dt,
            boundary.pml_m,
            boundary.pml_r0,
        );
        let pml_y = Cpml::new(
            ny,
            boundary.pml_cells,
            dy,
            dt,
            boundary.pml_m,
            boundary.pml_r0,
        );
        let pml_z = Cpml::new(
            nz,
            boundary.pml_cells,
            dz,
            dt,
            boundary.pml_m,
            boundary.pml_r0,
        );

        let ctx = GpuContext::new()?;
        let dev = &ctx.device;

        let n = nx * ny * nz;

        // Field buffers: vec4<f32> per cell = 16 bytes → pass n*4 to get n*16 bytes
        let buf_e = storage_rw_field(dev, n * 4, "e3d");
        let buf_h = storage_rw_field(dev, n * 4, "h3d");

        // psi buffers: 6 f32 per cell = 24 bytes → pass n*6 to get n*24 bytes
        let buf_psi_h = storage_rw_aux(dev, n * 6, "psi_h3d");
        let buf_psi_e = storage_rw_aux(dev, n * 6, "psi_e3d");

        // Material buffer: vec4<f32> per cell = 16 bytes → pass n*4
        // Initial: vacuum (eps_r=1, mu_r=1, sigma_e=0, sigma_m=0)
        let cpu_mat: Vec<f32> = {
            let mut v = vec![0.0f32; n * 4];
            for i in 0..n {
                v[i * 4] = 1.0; // eps_r
                v[i * 4 + 1] = 1.0; // mu_r
                                    // sigma_e = 0, sigma_m = 0
            }
            v
        };
        let buf_mat = storage_init(dev, &cpu_mat, "mat3d");

        // PML buffers: stride-6 per axis
        let buf_pml_x = storage_init(dev, &pack_pml_3d(&pml_x, nx), "pml_x3d");
        let buf_pml_y = storage_init(dev, &pack_pml_3d(&pml_y, ny), "pml_y3d");
        let buf_pml_z = storage_init(dev, &pack_pml_3d(&pml_z, nz), "pml_z3d");

        // Uniform buffers
        let sim_dims = SimDims3d {
            nx: nx as u32,
            ny: ny as u32,
            nz: nz as u32,
            dx: dx as f32,
            dy: dy as f32,
            dz: dz as f32,
            dt: dt as f32,
            _p0: 0,
        };
        let buf_dims = uniform_from(dev, &sim_dims, "dims3d");

        let init_src = SrcUniform {
            flat_idx: 0,
            _p0: 0,
            _p1: 0,
            val: 0.0,
        };
        let buf_src = uniform_from(dev, &init_src, "src3d");

        // Shaders
        let sm_inject = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fdtd3d_inject"),
            source: wgpu::ShaderSource::Wgsl(SHADER_INJECT.into()),
        });
        let sm_h = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fdtd3d_h"),
            source: wgpu::ShaderSource::Wgsl(SHADER_H.into()),
        });
        let sm_e = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fdtd3d_e"),
            source: wgpu::ShaderSource::Wgsl(SHADER_E.into()),
        });

        // --- Inject pipeline ---
        // binding 0: e_field (storage rw), binding 1: src (uniform)
        let bgl_inject = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_inject3d"),
            entries: &[storage_entry(0, false), uniform_entry(1)],
        });
        let pl_inject = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_inject3d"),
            bind_group_layouts: &[Some(&bgl_inject)],
            immediate_size: 0,
        });
        let pipeline_inject = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("inject3d"),
            layout: Some(&pl_inject),
            module: &sm_inject,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bg_inject = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_inject3d"),
            layout: &bgl_inject,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_e.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_src.as_entire_binding(),
                },
            ],
        });

        // --- H pipeline ---
        // 0: buf_h rw, 1: buf_e r, 2: psi_h rw, 3: buf_mat r,
        // 4: pml_x r, 5: pml_y r, 6: pml_z r, 7: dims uniform
        let bgl_h = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_h3d"),
            entries: &[
                storage_entry(0, false), // buf_h rw
                storage_entry(1, true),  // buf_e r
                storage_entry(2, false), // psi_h rw
                storage_entry(3, true),  // buf_mat r
                storage_entry(4, true),  // pml_x r
                storage_entry(5, true),  // pml_y r
                storage_entry(6, true),  // pml_z r
                uniform_entry(7),        // dims
            ],
        });
        let pl_h = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_h3d"),
            bind_group_layouts: &[Some(&bgl_h)],
            immediate_size: 0,
        });
        let pipeline_h = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("h3d"),
            layout: Some(&pl_h),
            module: &sm_h,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bg_h = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_h3d"),
            layout: &bgl_h,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_h.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_e.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_psi_h.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_mat.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_pml_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_pml_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buf_pml_z.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buf_dims.as_entire_binding(),
                },
            ],
        });

        // --- E pipeline ---
        // 0: buf_e rw, 1: buf_h r, 2: psi_e rw, 3: buf_mat r,
        // 4: pml_x r, 5: pml_y r, 6: pml_z r, 7: dims uniform
        let bgl_e = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_e3d"),
            entries: &[
                storage_entry(0, false), // buf_e rw
                storage_entry(1, true),  // buf_h r
                storage_entry(2, false), // psi_e rw
                storage_entry(3, true),  // buf_mat r
                storage_entry(4, true),  // pml_x r
                storage_entry(5, true),  // pml_y r
                storage_entry(6, true),  // pml_z r
                uniform_entry(7),        // dims
            ],
        });
        let pl_e = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_e3d"),
            bind_group_layouts: &[Some(&bgl_e)],
            immediate_size: 0,
        });
        let pipeline_e = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("e3d"),
            layout: Some(&pl_e),
            module: &sm_e,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bg_e = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_e3d"),
            layout: &bgl_e,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_e.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_h.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_psi_e.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_mat.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_pml_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_pml_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buf_pml_z.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buf_dims.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            ctx,
            nx,
            ny,
            nz,
            dt,
            time_step: 0,
            buf_e,
            buf_h,
            buf_psi_h,
            buf_psi_e,
            buf_mat,
            buf_pml_x,
            buf_pml_y,
            buf_pml_z,
            buf_dims,
            buf_src,
            pipeline_inject,
            pipeline_h,
            pipeline_e,
            bg_inject,
            bg_h,
            bg_e,
            src_flat_idx: 0,
            src_val: 0.0,
        })
    }

    /// Queue a point-source injection into Ez at (i, j, k) for the next `step()`.
    pub fn inject_ez(&mut self, i: usize, j: usize, k: usize, val: f64) {
        if i < self.nx && j < self.ny && k < self.nz {
            self.src_flat_idx = (k * self.ny * self.nx + j * self.nx + i) as u32;
            self.src_val = val as f32;
        }
    }

    /// Advance one time step on the GPU.
    /// Source from the last `inject_ez` call is applied first, then cleared.
    pub fn step(&mut self) -> Result<(), GpuError> {
        let src_uni = SrcUniform {
            flat_idx: self.src_flat_idx,
            _p0: 0,
            _p1: 0,
            val: self.src_val,
        };
        self.ctx
            .queue
            .write_buffer(&self.buf_src, 0, bytemuck::bytes_of(&src_uni));

        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fdtd3d_step"),
            });

        // Pass 1: inject source into Ez
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inject3d"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.pipeline_inject);
            cp.set_bind_group(0, &self.bg_inject, &[]);
            cp.dispatch_workgroups(1, 1, 1);
        }

        // Pass 2: update H field
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("h3d"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.pipeline_h);
            cp.set_bind_group(0, &self.bg_h, &[]);
            let gx = (self.nx as u32).div_ceil(4);
            let gy = (self.ny as u32).div_ceil(4);
            let gz = (self.nz as u32).div_ceil(4);
            cp.dispatch_workgroups(gx, gy, gz);
        }

        // Pass 3: update E field (reads updated H)
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("e3d"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.pipeline_e);
            cp.set_bind_group(0, &self.bg_e, &[]);
            let gx = (self.nx as u32).div_ceil(4);
            let gy = (self.ny as u32).div_ceil(4);
            let gz = (self.nz as u32).div_ceil(4);
            cp.dispatch_workgroups(gx, gy, gz);
        }

        self.ctx.queue.submit(std::iter::once(enc.finish()));

        self.src_val = 0.0;
        self.time_step += 1;

        Ok(())
    }

    /// Run for `steps` time steps. Fields remain on-GPU; no per-step readback.
    pub fn run(&mut self, steps: usize) -> Result<(), GpuError> {
        for _ in 0..steps {
            self.step()?;
        }
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Internal(format!("poll: {e}")))?;
        Ok(())
    }

    /// Download Ez field from GPU as f64 (flat index: k*ny*nx + j*nx + i).
    pub fn download_ez(&self) -> Result<Vec<f64>, GpuError> {
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Internal(format!("poll before ez dl: {e}")))?;
        let n = self.nx * self.ny * self.nz;
        // buf_e is n vec4<f32> = n*16 bytes
        let f32s = readback(
            &self.ctx.device,
            &self.ctx.queue,
            &self.buf_e,
            (n * 16) as u64,
            "dl_e3d",
        )?;
        // Ez is at component index 2 (x=0, y=1, z=2, w=3)
        Ok(f32s.iter().skip(2).step_by(4).map(|&v| v as f64).collect())
    }

    /// Download Hz field from GPU as f64 (flat index: k*ny*nx + j*nx + i).
    pub fn download_hz(&self) -> Result<Vec<f64>, GpuError> {
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Internal(format!("poll before hz dl: {e}")))?;
        let n = self.nx * self.ny * self.nz;
        let f32s = readback(
            &self.ctx.device,
            &self.ctx.queue,
            &self.buf_h,
            (n * 16) as u64,
            "dl_h3d",
        )?;
        // Hz is at component index 2
        Ok(f32s.iter().skip(2).step_by(4).map(|&v| v as f64).collect())
    }

    /// Download Ex field from GPU as f64.
    pub fn download_ex(&self) -> Result<Vec<f64>, GpuError> {
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Internal(format!("poll before ex dl: {e}")))?;
        let n = self.nx * self.ny * self.nz;
        let f32s = readback(
            &self.ctx.device,
            &self.ctx.queue,
            &self.buf_e,
            (n * 16) as u64,
            "dl_ex3d",
        )?;
        Ok(f32s.iter().step_by(4).map(|&v| v as f64).collect())
    }

    /// Download Ey field from GPU as f64.
    pub fn download_ey(&self) -> Result<Vec<f64>, GpuError> {
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Internal(format!("poll before ey dl: {e}")))?;
        let n = self.nx * self.ny * self.nz;
        let f32s = readback(
            &self.ctx.device,
            &self.ctx.queue,
            &self.buf_e,
            (n * 16) as u64,
            "dl_ey3d",
        )?;
        // Ey is at component index 1
        Ok(f32s.iter().skip(1).step_by(4).map(|&v| v as f64).collect())
    }

    /// Download Hx field from GPU as f64.
    pub fn download_hx(&self) -> Result<Vec<f64>, GpuError> {
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Internal(format!("poll before hx dl: {e}")))?;
        let n = self.nx * self.ny * self.nz;
        let f32s = readback(
            &self.ctx.device,
            &self.ctx.queue,
            &self.buf_h,
            (n * 16) as u64,
            "dl_hx3d",
        )?;
        Ok(f32s.iter().step_by(4).map(|&v| v as f64).collect())
    }

    /// Download Hy field from GPU as f64.
    pub fn download_hy(&self) -> Result<Vec<f64>, GpuError> {
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Internal(format!("poll before hy dl: {e}")))?;
        let n = self.nx * self.ny * self.nz;
        let f32s = readback(
            &self.ctx.device,
            &self.ctx.queue,
            &self.buf_h,
            (n * 16) as u64,
            "dl_hy3d",
        )?;
        // Hy is at component index 1
        Ok(f32s.iter().skip(1).step_by(4).map(|&v| v as f64).collect())
    }
}

// ── PML packing helper ────────────────────────────────────────────────────────

/// Pack Cpml coefficients for 3D shaders with stride-6 per cell:
/// [b_e, c_e, kappa_e, b_h, c_h, kappa_h] for each cell index 0..n
fn pack_pml_3d(pml: &Cpml, n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; 6 * n];
    for i in 0..n {
        v[6 * i] = pml.b_e[i] as f32;
        v[6 * i + 1] = pml.c_e[i] as f32;
        v[6 * i + 2] = pml.kappa_e[i] as f32;
        v[6 * i + 3] = pml.b_h[i] as f32;
        v[6 * i + 4] = pml.c_h[i] as f32;
        v[6 * i + 5] = pml.kappa_h[i] as f32;
    }
    v
}
