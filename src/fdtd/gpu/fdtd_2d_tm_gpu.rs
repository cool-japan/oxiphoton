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

const SHADER_INJECT: &str = include_str!("shaders/tm2d_inject.wgsl");
const SHADER_HX: &str = include_str!("shaders/tm2d_hx.wgsl");
const SHADER_HY: &str = include_str!("shaders/tm2d_hy.wgsl");
const SHADER_EZ: &str = include_str!("shaders/tm2d_ez.wgsl");

/// 2D TM FDTD solver accelerated via wgpu compute shaders.
///
/// Fields (Ez, Hx, Hy) and psi arrays reside on-GPU across `step()`; no
/// per-step readback. Coefficients are precomputed in f64 (identical to
/// `Fdtd2dTm`) then cast to f32 for upload, keeping CPML precision under
/// control.
///
/// Step order: inject_ez → Hx update → Hy update → Ez update.
///
/// # GPU availability
/// If no adapter is found, `new` returns `Err(GpuError::NotAvailable)`.
/// The caller is responsible for graceful fallback to `Fdtd2dTm`.
// GPU buffers are owned for lifetime management — bind groups hold wgpu
// references to them, so they must outlive the bind groups. The compiler
// cannot track this cross-API ownership, hence the allow.
#[allow(dead_code)]
pub struct Fdtd2dTmGpu {
    ctx: GpuContext,
    pub nx: usize,
    pub ny: usize,
    pub dt: f64,
    pub time_step: usize,

    // Field buffers (STORAGE | COPY_SRC for readback)
    buf_ez: wgpu::Buffer,
    buf_hx: wgpu::Buffer,
    buf_hy: wgpu::Buffer,

    // psi auxiliary buffers (STORAGE only)
    buf_psi_hx_y: wgpu::Buffer,
    buf_psi_hy_x: wgpu::Buffer,
    buf_psi_ez_x: wgpu::Buffer,
    buf_psi_ez_y: wgpu::Buffer,

    // Material buffer (STORAGE | COPY_DST so fill_eps_box can update it)
    buf_eps_r: wgpu::Buffer,

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
    pipeline_hx: wgpu::ComputePipeline,
    pipeline_hy: wgpu::ComputePipeline,
    pipeline_ez: wgpu::ComputePipeline,

    // Bind groups (one per pipeline)
    bg_inject: wgpu::BindGroup,
    bg_hx: wgpu::BindGroup,
    bg_hy: wgpu::BindGroup,
    bg_ez: wgpu::BindGroup,

    // CPU-side eps copy (f32) — kept in sync for fill_eps_box uploads
    cpu_eps_r: Vec<f32>,

    // Pending source for next step
    src_flat_idx: u32,
    src_val: f32,
}

impl Fdtd2dTmGpu {
    /// Create a GPU TM FDTD solver with the same parameters as `Fdtd2dTm::new`.
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
        let n_ez = nx * ny;
        let n_hx = nx * (ny + 1);
        let n_hy = (nx + 1) * ny;

        // ── Field buffers ─────────────────────────────────────────────────────
        let buf_ez = storage_rw_field(dev, n_ez, "ez");
        let buf_hx = storage_rw_field(dev, n_hx, "hx");
        let buf_hy = storage_rw_field(dev, n_hy, "hy");

        // ── psi auxiliary buffers ─────────────────────────────────────────────
        let buf_psi_hx_y = storage_rw_aux(dev, n_hx, "psi_hx_y");
        let buf_psi_hy_x = storage_rw_aux(dev, n_hy, "psi_hy_x");
        let buf_psi_ez_x = storage_rw_aux(dev, n_ez, "psi_ez_x");
        let buf_psi_ez_y = storage_rw_aux(dev, n_ez, "psi_ez_y");

        // ── Material buffer (initialised to vacuum = 1.0) ─────────────────────
        let cpu_eps_r = vec![1.0f32; n_ez];
        let buf_eps_r = storage_init_updatable(dev, &cpu_eps_r, "eps_r");

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
            label: Some("tm2d_inject"),
            source: wgpu::ShaderSource::Wgsl(SHADER_INJECT.into()),
        });
        let sm_hx = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tm2d_hx"),
            source: wgpu::ShaderSource::Wgsl(SHADER_HX.into()),
        });
        let sm_hy = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tm2d_hy"),
            source: wgpu::ShaderSource::Wgsl(SHADER_HY.into()),
        });
        let sm_ez = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tm2d_ez"),
            source: wgpu::ShaderSource::Wgsl(SHADER_EZ.into()),
        });

        // ── Bind group layouts & pipelines ────────────────────────────────────

        // --- Inject ---
        let bgl_inject = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_inject_tm"),
            entries: &[storage_entry(0, false), uniform_entry(1)],
        });
        let pl_inject = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_inject_tm"),
            bind_group_layouts: &[Some(&bgl_inject)],
            immediate_size: 0,
        });
        let pipeline_inject = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("inject_tm"),
            layout: Some(&pl_inject),
            module: &sm_inject,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bg_inject = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_inject_tm"),
            layout: &bgl_inject,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_ez.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_src.as_entire_binding(),
                },
            ],
        });

        // --- Hx ---
        // Bindings: hx(rw), ez(r), psi_hx_y(rw), pml_h_y(r), dims
        let bgl_hx = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_hx"),
            entries: &[
                storage_entry(0, false), // hx rw
                storage_entry(1, true),  // ez r
                storage_entry(2, false), // psi_hx_y rw
                storage_entry(3, true),  // pml_h_y r
                uniform_entry(4),        // dims
            ],
        });
        let pl_hx = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_hx"),
            bind_group_layouts: &[Some(&bgl_hx)],
            immediate_size: 0,
        });
        let pipeline_hx = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hx"),
            layout: Some(&pl_hx),
            module: &sm_hx,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bg_hx = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_hx"),
            layout: &bgl_hx,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_hx.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_ez.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_psi_hx_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_pml_h_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_dims.as_entire_binding(),
                },
            ],
        });

        // --- Hy ---
        // Bindings: hy(rw), ez(r), psi_hy_x(rw), pml_h_x(r), dims
        let bgl_hy = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_hy"),
            entries: &[
                storage_entry(0, false), // hy rw
                storage_entry(1, true),  // ez r
                storage_entry(2, false), // psi_hy_x rw
                storage_entry(3, true),  // pml_h_x r
                uniform_entry(4),        // dims
            ],
        });
        let pl_hy = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_hy"),
            bind_group_layouts: &[Some(&bgl_hy)],
            immediate_size: 0,
        });
        let pipeline_hy = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hy"),
            layout: Some(&pl_hy),
            module: &sm_hy,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bg_hy = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_hy"),
            layout: &bgl_hy,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_hy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_ez.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_psi_hy_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_pml_h_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_dims.as_entire_binding(),
                },
            ],
        });

        // --- Ez ---
        // Bindings: ez(rw), hx(r), hy(r), psi_ez_x(rw), psi_ez_y(rw), eps_r(r),
        //           pml_e_x(r), pml_e_y(r), dims
        let bgl_ez = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_ez"),
            entries: &[
                storage_entry(0, false), // ez rw
                storage_entry(1, true),  // hx r
                storage_entry(2, true),  // hy r
                storage_entry(3, false), // psi_ez_x rw
                storage_entry(4, false), // psi_ez_y rw
                storage_entry(5, true),  // eps_r r
                storage_entry(6, true),  // pml_e_x r
                storage_entry(7, true),  // pml_e_y r
                uniform_entry(8),        // dims
            ],
        });
        let pl_ez = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_ez"),
            bind_group_layouts: &[Some(&bgl_ez)],
            immediate_size: 0,
        });
        let pipeline_ez = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ez"),
            layout: Some(&pl_ez),
            module: &sm_ez,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bg_ez = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_ez"),
            layout: &bgl_ez,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_ez.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_hx.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_hy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_psi_ez_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_psi_ez_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_eps_r.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buf_pml_e_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buf_pml_e_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
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
            buf_ez,
            buf_hx,
            buf_hy,
            buf_psi_hx_y,
            buf_psi_hy_x,
            buf_psi_ez_x,
            buf_psi_ez_y,
            buf_eps_r,
            buf_pml_h_x,
            buf_pml_h_y,
            buf_pml_e_x,
            buf_pml_e_y,
            buf_dims,
            buf_src,
            pipeline_inject,
            pipeline_hx,
            pipeline_hy,
            pipeline_ez,
            bg_inject,
            bg_hx,
            bg_hy,
            bg_ez,
            cpu_eps_r,
            src_flat_idx: 0,
            src_val: 0.0,
        })
    }

    /// Queue a point-source injection at (i, j) into Ez for the next `step()`.
    /// Matches the additive semantics of `Fdtd2dTm::inject_ez`.
    pub fn inject_ez(&mut self, i: usize, j: usize, val: f64) {
        if i < self.nx && j < self.ny {
            self.src_flat_idx = (j * self.nx + i) as u32;
            self.src_val = val as f32;
        }
    }

    /// Fill a dielectric box [ix0,ix1) x [iy0,iy1) with eps_r.
    /// Mirrors `Fdtd2dTm::fill_eps_box` — no staggering for TM Ez cells.
    pub fn fill_eps_box(&mut self, ix0: usize, ix1: usize, iy0: usize, iy1: usize, eps_r: f64) {
        let eps_r = eps_r as f32;
        let nx = self.nx;
        let ny = self.ny;
        for j in iy0..iy1.min(ny) {
            for i in ix0..ix1.min(nx) {
                self.cpu_eps_r[j * nx + i] = eps_r;
            }
        }
        self.ctx
            .queue
            .write_buffer(&self.buf_eps_r, 0, bytemuck::cast_slice(&self.cpu_eps_r));
    }

    /// Advance one time step on the GPU.
    /// Source from the last `inject_ez` call is applied first (additive), then cleared.
    /// Step order: inject Ez → Hx → Hy → Ez.
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
                label: Some("fdtd_tm_step"),
            });

        // Pass 1: inject source into Ez (1x1 dispatch)
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inject_ez"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.pipeline_inject);
            cp.set_bind_group(0, &self.bg_inject, &[]);
            cp.dispatch_workgroups(1, 1, 1);
        }

        // Pass 2: update Hx (reads Ez from inject pass)
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hx"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.pipeline_hx);
            cp.set_bind_group(0, &self.bg_hx, &[]);
            let gx = (self.nx as u32).div_ceil(8);
            let gy = (self.ny as u32).div_ceil(8);
            cp.dispatch_workgroups(gx, gy, 1);
        }

        // Pass 3: update Hy (reads Ez from inject pass)
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hy"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.pipeline_hy);
            cp.set_bind_group(0, &self.bg_hy, &[]);
            let gx = (self.nx as u32).div_ceil(8);
            let gy = (self.ny as u32).div_ceil(8);
            cp.dispatch_workgroups(gx, gy, 1);
        }

        // Pass 4: update Ez (reads updated Hx and Hy)
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ez"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.pipeline_ez);
            cp.set_bind_group(0, &self.bg_ez, &[]);
            let gx = (self.nx as u32).div_ceil(8);
            let gy = (self.ny as u32).div_ceil(8);
            cp.dispatch_workgroups(gx, gy, 1);
        }

        self.ctx.queue.submit(std::iter::once(enc.finish()));

        // Clear pending source (CPU-side; next step gets 0 unless inject_ez called)
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

    /// Download Ez field from GPU as f64 (row-major, j*nx+i).
    pub fn download_ez(&self) -> Result<Vec<f64>, GpuError> {
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Internal(format!("poll before ez dl: {e}")))?;
        let n = self.nx * self.ny;
        let f32s = readback(
            &self.ctx.device,
            &self.ctx.queue,
            &self.buf_ez,
            (n * 4) as u64,
            "dl_ez",
        )?;
        Ok(f32s.iter().map(|&v| v as f64).collect())
    }

    /// Download Hx field from GPU as f64 (j*nx+i, buffer size nx*(ny+1)).
    pub fn download_hx(&self) -> Result<Vec<f64>, GpuError> {
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Internal(format!("poll before hx dl: {e}")))?;
        let n = self.nx * (self.ny + 1);
        let f32s = readback(
            &self.ctx.device,
            &self.ctx.queue,
            &self.buf_hx,
            (n * 4) as u64,
            "dl_hx",
        )?;
        Ok(f32s.iter().map(|&v| v as f64).collect())
    }

    /// Download Hy field from GPU as f64 (j*(nx+1)+i, buffer size (nx+1)*ny).
    pub fn download_hy(&self) -> Result<Vec<f64>, GpuError> {
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Internal(format!("poll before hy dl: {e}")))?;
        let n = (self.nx + 1) * self.ny;
        let f32s = readback(
            &self.ctx.device,
            &self.ctx.queue,
            &self.buf_hy,
            (n * 4) as u64,
            "dl_hy",
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
