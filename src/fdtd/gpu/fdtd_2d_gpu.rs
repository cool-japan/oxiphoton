use crate::fdtd::boundary::pml::Cpml;
use crate::fdtd::config::{BoundaryConfig, Dimensions, GridSpacing};
use crate::fdtd::courant::courant_dt;

use super::{
    buffers::{
        readback, storage_entry, storage_init, storage_init_updatable, storage_rw_aux,
        storage_rw_field, uniform_entry, uniform_from, SimDims, SrcUniform,
    },
    context::GpuContext,
    GpuError,
};

const SHADER_INJECT: &str = include_str!("shaders/te2d_inject.wgsl");
const SHADER_HZ: &str = include_str!("shaders/te2d_hz.wgsl");
const SHADER_EX: &str = include_str!("shaders/te2d_ex.wgsl");
const SHADER_EY: &str = include_str!("shaders/te2d_ey.wgsl");

/// 2D TE FDTD solver accelerated via wgpu compute shaders.
///
/// Fields (Hz, Ex, Ey) and psi arrays reside on-GPU across `run()`; no per-step
/// readback. Coefficients are precomputed in f64 (identical to `Fdtd2dTe`) then
/// cast to f32 for upload, keeping CPML precision under control.
///
/// # GPU availability
/// If no adapter is found, `new` returns `Err(GpuError::NotAvailable)`.
/// The caller is responsible for graceful fallback to `Fdtd2dTe`.
// GPU buffers are owned for lifetime management — the bind groups hold wgpu
// references to them, so they must outlive the bind groups. The compiler
// cannot track this cross-API ownership, hence the allow.
#[allow(dead_code)]
pub struct Fdtd2dGpu {
    ctx: GpuContext,
    pub nx: usize,
    pub ny: usize,
    pub dt: f64,
    pub time_step: usize,

    // Field buffers (STORAGE | COPY_SRC for readback)
    buf_hz: wgpu::Buffer,
    buf_ex: wgpu::Buffer,
    buf_ey: wgpu::Buffer,

    // psi auxiliary buffers (STORAGE only)
    buf_psi_hz_x: wgpu::Buffer,
    buf_psi_hz_y: wgpu::Buffer,
    buf_psi_ex_y: wgpu::Buffer,
    buf_psi_ey_x: wgpu::Buffer,

    // Material buffers (STORAGE | COPY_DST so fill_eps_box can update them)
    buf_mu_hz: wgpu::Buffer,
    buf_eps_ex: wgpu::Buffer,
    buf_eps_ey: wgpu::Buffer,

    // PML coefficient buffers, packed [b, c, kappa] (STORAGE, uploaded once)
    buf_pml_h_x: wgpu::Buffer,
    buf_pml_h_y: wgpu::Buffer,
    buf_pml_e_x: wgpu::Buffer,
    buf_pml_e_y: wgpu::Buffer,

    // Uniform buffers
    buf_dims: wgpu::Buffer,
    buf_src: wgpu::Buffer,

    // Compute pipelines
    pipeline_inject: wgpu::ComputePipeline,
    pipeline_hz: wgpu::ComputePipeline,
    pipeline_ex: wgpu::ComputePipeline,
    pipeline_ey: wgpu::ComputePipeline,

    // Bind groups (one per pipeline)
    bg_inject: wgpu::BindGroup,
    bg_hz: wgpu::BindGroup,
    bg_ex: wgpu::BindGroup,
    bg_ey: wgpu::BindGroup,

    // CPU-side eps copies (f32) — kept in sync for fill_eps_box uploads
    cpu_eps_ex: Vec<f32>,
    cpu_eps_ey: Vec<f32>,

    // Pending source for next step
    src_flat_idx: u32,
    src_val: f32,
}

impl Fdtd2dGpu {
    /// Create a GPU FDTD solver with the same parameters as `Fdtd2dTe::new`.
    pub fn new(
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
        boundary: &BoundaryConfig,
    ) -> Result<Self, GpuError> {
        // Replicate CPU dt and CPML exactly
        let spacing = GridSpacing { dx, dy, dz: dx };
        let dt = 0.99 * courant_dt(Dimensions::TwoD { nx, ny }, spacing, 1.0);
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

        let ctx = GpuContext::new()?;
        let dev = &ctx.device;

        // ── Buffer sizes ──────────────────────────────────────────────────────
        let n_hz = nx * ny;
        let n_ex = (nx + 1) * ny;
        let n_ey = nx * (ny + 1);

        // ── Field buffers ─────────────────────────────────────────────────────
        let buf_hz = storage_rw_field(dev, n_hz, "hz");
        let buf_ex = storage_rw_field(dev, n_ex, "ex");
        let buf_ey = storage_rw_field(dev, n_ey, "ey");

        // ── psi auxiliary buffers ───────────────────────────────────────────────
        let buf_psi_hz_x = storage_rw_aux(dev, n_hz, "psi_hz_x");
        let buf_psi_hz_y = storage_rw_aux(dev, n_hz, "psi_hz_y");
        let buf_psi_ex_y = storage_rw_aux(dev, n_ex, "psi_ex_y");
        let buf_psi_ey_x = storage_rw_aux(dev, n_ey, "psi_ey_x");

        // ── Material buffers (initialised to vacuum = 1.0) ───────────────────
        let cpu_eps_ex = vec![1.0f32; n_ex];
        let cpu_eps_ey = vec![1.0f32; n_ey];
        let cpu_mu_hz = vec![1.0f32; n_hz];

        let buf_eps_ex = storage_init_updatable(dev, &cpu_eps_ex, "eps_ex");
        let buf_eps_ey = storage_init_updatable(dev, &cpu_eps_ey, "eps_ey");
        let buf_mu_hz = storage_init_updatable(dev, &cpu_mu_hz, "mu_hz");

        // ── PML coefficient buffers (packed [b, c, kappa]) ───────────────────
        let buf_pml_h_x = storage_init(dev, &pack_pml_h(&pml_x, nx), "pml_h_x");
        let buf_pml_h_y = storage_init(dev, &pack_pml_h(&pml_y, ny), "pml_h_y");
        let buf_pml_e_x = storage_init(dev, &pack_pml_e(&pml_x, nx), "pml_e_x");
        let buf_pml_e_y = storage_init(dev, &pack_pml_e(&pml_y, ny), "pml_e_y");

        // ── Uniform buffers ───────────────────────────────────────────────────
        let sim_dims = SimDims {
            nx: nx as u32,
            ny: ny as u32,
            dx: dx as f32,
            dy: dy as f32,
            dt: dt as f32,
            _p0: 0,
            _p1: 0,
            _p2: 0,
        };
        let buf_dims = uniform_from(dev, &sim_dims, "sim_dims");

        let init_src = SrcUniform {
            flat_idx: 0,
            _p0: 0,
            _p1: 0,
            val: 0.0,
        };
        let buf_src = uniform_from(dev, &init_src, "src_uniform");

        // ── Shaders ───────────────────────────────────────────────────────────
        let sm_inject = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("te2d_inject"),
            source: wgpu::ShaderSource::Wgsl(SHADER_INJECT.into()),
        });
        let sm_hz = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("te2d_hz"),
            source: wgpu::ShaderSource::Wgsl(SHADER_HZ.into()),
        });
        let sm_ex = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("te2d_ex"),
            source: wgpu::ShaderSource::Wgsl(SHADER_EX.into()),
        });
        let sm_ey = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("te2d_ey"),
            source: wgpu::ShaderSource::Wgsl(SHADER_EY.into()),
        });

        // ── Bind group layouts & pipelines ────────────────────────────────────

        // --- Inject ---
        let bgl_inject = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_inject"),
            entries: &[storage_entry(0, false), uniform_entry(1)],
        });
        let pl_inject = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_inject"),
            bind_group_layouts: &[Some(&bgl_inject)],
            immediate_size: 0,
        });
        let pipeline_inject = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("inject"),
            layout: Some(&pl_inject),
            module: &sm_inject,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bg_inject = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_inject"),
            layout: &bgl_inject,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_hz.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_src.as_entire_binding(),
                },
            ],
        });

        // --- Hz ---
        let bgl_hz = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_hz"),
            entries: &[
                storage_entry(0, false), // hz rw
                storage_entry(1, true),  // ex r
                storage_entry(2, true),  // ey r
                storage_entry(3, false), // psi_hz_x rw
                storage_entry(4, false), // psi_hz_y rw
                storage_entry(5, true),  // mu_hz r
                storage_entry(6, true),  // pml_h_x r
                storage_entry(7, true),  // pml_h_y r
                uniform_entry(8),        // dims
            ],
        });
        let pl_hz = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_hz"),
            bind_group_layouts: &[Some(&bgl_hz)],
            immediate_size: 0,
        });
        let pipeline_hz = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hz"),
            layout: Some(&pl_hz),
            module: &sm_hz,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bg_hz = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_hz"),
            layout: &bgl_hz,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_hz.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_ex.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_ey.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_psi_hz_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_psi_hz_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_mu_hz.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buf_pml_h_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buf_pml_h_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buf_dims.as_entire_binding(),
                },
            ],
        });

        // --- Ex ---
        let bgl_ex = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_ex"),
            entries: &[
                storage_entry(0, false), // ex rw
                storage_entry(1, true),  // hz r
                storage_entry(2, false), // psi_ex_y rw
                storage_entry(3, true),  // eps_ex r
                storage_entry(4, true),  // pml_e_y r
                uniform_entry(5),        // dims
            ],
        });
        let pl_ex = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_ex"),
            bind_group_layouts: &[Some(&bgl_ex)],
            immediate_size: 0,
        });
        let pipeline_ex = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ex"),
            layout: Some(&pl_ex),
            module: &sm_ex,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bg_ex = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_ex"),
            layout: &bgl_ex,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_ex.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_hz.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_psi_ex_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_eps_ex.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_pml_e_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_dims.as_entire_binding(),
                },
            ],
        });

        // --- Ey ---
        let bgl_ey = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_ey"),
            entries: &[
                storage_entry(0, false), // ey rw
                storage_entry(1, true),  // hz r
                storage_entry(2, false), // psi_ey_x rw
                storage_entry(3, true),  // eps_ey r
                storage_entry(4, true),  // pml_e_x r
                uniform_entry(5),        // dims
            ],
        });
        let pl_ey = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_ey"),
            bind_group_layouts: &[Some(&bgl_ey)],
            immediate_size: 0,
        });
        let pipeline_ey = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ey"),
            layout: Some(&pl_ey),
            module: &sm_ey,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bg_ey = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_ey"),
            layout: &bgl_ey,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_ey.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_hz.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_psi_ey_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_eps_ey.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_pml_e_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_dims.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            ctx,
            nx,
            ny,
            dt,
            time_step: 0,
            buf_hz,
            buf_ex,
            buf_ey,
            buf_psi_hz_x,
            buf_psi_hz_y,
            buf_psi_ex_y,
            buf_psi_ey_x,
            buf_mu_hz,
            buf_eps_ex,
            buf_eps_ey,
            buf_pml_h_x,
            buf_pml_h_y,
            buf_pml_e_x,
            buf_pml_e_y,
            buf_dims,
            buf_src,
            pipeline_inject,
            pipeline_hz,
            pipeline_ex,
            pipeline_ey,
            bg_inject,
            bg_hz,
            bg_ex,
            bg_ey,
            cpu_eps_ex,
            cpu_eps_ey,
            src_flat_idx: 0,
            src_val: 0.0,
        })
    }

    /// Queue a point-source injection at (i, j) into Hz for the next `step()`.
    /// Matches the additive semantics of `Fdtd2dTe::inject_hz`.
    pub fn inject_hz(&mut self, i: usize, j: usize, val: f64) {
        if i < self.nx && j < self.ny {
            self.src_flat_idx = (j * self.nx + i) as u32;
            self.src_val = val as f32;
        }
    }

    /// Fill a dielectric box [ix0,ix1) x [iy0,iy1) with eps_r.
    /// Mirrors `Fdtd2dTe::fill_eps_box` staggering exactly.
    pub fn fill_eps_box(&mut self, ix0: usize, ix1: usize, iy0: usize, iy1: usize, eps_r: f64) {
        let eps_r = eps_r as f32;
        let nx = self.nx;
        let ny = self.ny;
        for j in iy0..iy1.min(ny) {
            for i in ix0..ix1.min(nx) {
                self.cpu_eps_ex[(j + 1) * (nx + 1) + i] = eps_r;
                self.cpu_eps_ey[j * nx + i] = eps_r;
            }
        }
        self.ctx
            .queue
            .write_buffer(&self.buf_eps_ex, 0, bytemuck::cast_slice(&self.cpu_eps_ex));
        self.ctx
            .queue
            .write_buffer(&self.buf_eps_ey, 0, bytemuck::cast_slice(&self.cpu_eps_ey));
    }

    /// Advance one time step on the GPU.
    /// Source from the last `inject_hz` call is applied first, then cleared.
    pub fn step(&mut self) -> Result<(), GpuError> {
        // Update source uniform
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
                label: Some("fdtd_step"),
            });

        // Pass 1: inject source into Hz (1x1 dispatch)
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inject"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.pipeline_inject);
            cp.set_bind_group(0, &self.bg_inject, &[]);
            cp.dispatch_workgroups(1, 1, 1);
        }

        // Pass 2: update Hz (reads Ex/Ey from previous step)
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hz"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.pipeline_hz);
            cp.set_bind_group(0, &self.bg_hz, &[]);
            let gx = (self.nx as u32).div_ceil(8);
            let gy = (self.ny as u32).div_ceil(8);
            cp.dispatch_workgroups(gx, gy, 1);
        }

        // Pass 3: update Ex (reads updated Hz — separated by compute pass barrier)
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ex"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.pipeline_ex);
            cp.set_bind_group(0, &self.bg_ex, &[]);
            let gx = ((self.nx + 1) as u32).div_ceil(8);
            let gy = (self.ny as u32).div_ceil(8);
            cp.dispatch_workgroups(gx, gy, 1);
        }

        // Pass 4: update Ey (reads updated Hz)
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ey"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.pipeline_ey);
            cp.set_bind_group(0, &self.bg_ey, &[]);
            let gx = (self.nx as u32).div_ceil(8);
            let gy = ((self.ny + 1) as u32).div_ceil(8);
            cp.dispatch_workgroups(gx, gy, 1);
        }

        self.ctx.queue.submit(std::iter::once(enc.finish()));

        // Clear pending source (CPU-side; next step gets 0 unless inject_hz called)
        self.src_val = 0.0;
        self.time_step += 1;

        Ok(())
    }

    /// Run for `steps` time steps. Fields remain on-GPU; no per-step readback.
    pub fn run(&mut self, steps: usize) -> Result<(), GpuError> {
        for _ in 0..steps {
            self.step()?;
        }
        // Flush and wait for all submitted work
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Internal(format!("poll: {e}")))?;
        Ok(())
    }

    /// Download Hz field from GPU as f64 (row-major, j*nx+i).
    pub fn download_hz(&self) -> Result<Vec<f64>, GpuError> {
        // Ensure all pending GPU work is done before reading
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Internal(format!("poll before hz dl: {e}")))?;
        let n = self.nx * self.ny;
        let f32s = readback(
            &self.ctx.device,
            &self.ctx.queue,
            &self.buf_hz,
            (n * 4) as u64,
            "dl_hz",
        )?;
        Ok(f32s.iter().map(|&v| v as f64).collect())
    }

    /// Download Ex field from GPU as f64 (j*(nx+1)+i).
    pub fn download_ex(&self) -> Result<Vec<f64>, GpuError> {
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Internal(format!("poll before ex dl: {e}")))?;
        let n = (self.nx + 1) * self.ny;
        let f32s = readback(
            &self.ctx.device,
            &self.ctx.queue,
            &self.buf_ex,
            (n * 4) as u64,
            "dl_ex",
        )?;
        Ok(f32s.iter().map(|&v| v as f64).collect())
    }

    /// Download Ey field from GPU as f64 (j*nx+i).
    pub fn download_ey(&self) -> Result<Vec<f64>, GpuError> {
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Internal(format!("poll before ey dl: {e}")))?;
        let n = self.nx * (self.ny + 1);
        let f32s = readback(
            &self.ctx.device,
            &self.ctx.queue,
            &self.buf_ey,
            (n * 4) as u64,
            "dl_ey",
        )?;
        Ok(f32s.iter().map(|&v| v as f64).collect())
    }
}

// ── PML packing helpers ───────────────────────────────────────────────────────

/// Pack Cpml H-field coefficients as [b_h[0..n], c_h[0..n], kappa_h[0..n]].
fn pack_pml_h(pml: &Cpml, n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; 3 * n];
    for k in 0..n {
        v[k] = pml.b_h[k] as f32;
        v[n + k] = pml.c_h[k] as f32;
        v[2 * n + k] = pml.kappa_h[k] as f32;
    }
    v
}

/// Pack Cpml E-field coefficients as [b_e[0..n], c_e[0..n], kappa_e[0..n]].
fn pack_pml_e(pml: &Cpml, n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; 3 * n];
    for k in 0..n {
        v[k] = pml.b_e[k] as f32;
        v[n + k] = pml.c_e[k] as f32;
        v[2 * n + k] = pml.kappa_e[k] as f32;
    }
    v
}
