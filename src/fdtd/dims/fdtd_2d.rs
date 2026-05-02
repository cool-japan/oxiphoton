use crate::fdtd::boundary::pml::Cpml;
use crate::fdtd::config::BoundaryConfig;
use crate::fdtd::config::{Dimensions, GridSpacing};
use crate::fdtd::courant::courant_dt;
use crate::fdtd::engine::yee::Yee2dTe;
use crate::fdtd::source::SourceWaveform;
use crate::units::conversion::{EPSILON_0, MU_0};
use num_complex::Complex64;
use std::f64::consts::PI;

/// TFSF (Total-Field/Scattered-Field) source configuration for 2D TE FDTD
///
/// Injects a plane wave traveling in +x direction inside the TFSF box.
pub struct TfsfSource {
    /// Left boundary of TFSF box (i_min in cells)
    pub i_min: usize,
    /// Right boundary (i_max)
    pub i_max: usize,
    /// Bottom boundary (j_min)
    pub j_min: usize,
    /// Top boundary (j_max)
    pub j_max: usize,
    /// 1D auxiliary grid for incident field tracking (size = nx)
    aux_ex: Vec<f64>,
    aux_hy: Vec<f64>,
    aux_eps: Vec<f64>,
    aux_psi_hy: Vec<f64>,
    aux_psi_ex: Vec<f64>,
    /// CPML for aux grid
    aux_pml: Cpml,
    /// Source waveform
    pub waveform: Box<dyn SourceWaveform>,
    /// Source injection position in aux grid
    src_pos: usize,
    /// dt for aux grid (same as main solver)
    dt: f64,
    dx: f64,
}

/// Builder config for TfsfSource
pub struct TfsfConfig {
    pub i_min: usize,
    pub i_max: usize,
    pub j_min: usize,
    pub j_max: usize,
    pub pml_cells: usize,
}

impl TfsfSource {
    pub fn new(
        nx_total: usize,
        dx: f64,
        dt: f64,
        cfg: TfsfConfig,
        waveform: Box<dyn SourceWaveform>,
    ) -> Self {
        let aux_pml = Cpml::new(nx_total, cfg.pml_cells, dx, dt, 3.5, 1e-8);
        let src_pos = cfg.i_min.saturating_sub(5).max(cfg.pml_cells + 1);
        Self {
            i_min: cfg.i_min,
            i_max: cfg.i_max,
            j_min: cfg.j_min,
            j_max: cfg.j_max,
            aux_ex: vec![0.0; nx_total],
            aux_hy: vec![0.0; nx_total],
            aux_eps: vec![1.0; nx_total],
            aux_psi_hy: vec![0.0; nx_total],
            aux_psi_ex: vec![0.0; nx_total],
            aux_pml,
            waveform,
            src_pos,
            dt,
            dx,
        }
    }

    /// Advance the auxiliary 1D grid by one step
    pub fn advance_aux(&mut self, t: f64) {
        let n = self.aux_ex.len();
        let dz = self.dx;
        let dt = self.dt;

        // H update
        for i in 0..n - 1 {
            let dex = self.aux_ex[i + 1] - self.aux_ex[i];
            self.aux_psi_hy[i] =
                self.aux_pml.b_h[i] * self.aux_psi_hy[i] + self.aux_pml.c_h[i] * dex / dz;
            let kappa = self.aux_pml.kappa_h[i];
            self.aux_hy[i] -= dt / MU_0 * (dex / (kappa * dz) + self.aux_psi_hy[i]);
        }

        // Inject source
        if self.src_pos < n {
            self.aux_ex[self.src_pos] += self.waveform.amplitude(t + 0.5 * dt);
        }

        // E update
        for i in 1..n - 1 {
            let dhy = self.aux_hy[i] - self.aux_hy[i - 1];
            self.aux_psi_ex[i] =
                self.aux_pml.b_e[i] * self.aux_psi_ex[i] + self.aux_pml.c_e[i] * dhy / dz;
            let kappa = self.aux_pml.kappa_e[i];
            self.aux_ex[i] -=
                dt / (EPSILON_0 * self.aux_eps[i]) * (dhy / (kappa * dz) + self.aux_psi_ex[i]);
        }
        self.aux_ex[0] = 0.0;
        self.aux_ex[n - 1] = 0.0;
    }

    pub fn inc_ex(&self, i: usize) -> f64 {
        if i < self.aux_ex.len() {
            self.aux_ex[i]
        } else {
            0.0
        }
    }

    pub fn inc_hy(&self, i: usize) -> f64 {
        if i < self.aux_hy.len() {
            self.aux_hy[i]
        } else {
            0.0
        }
    }
}

/// DFT collector for 2D fields at multiple frequencies
#[derive(Clone)]
pub struct DftBox2d {
    pub nx: usize,
    pub ny: usize,
    pub omegas: Vec<f64>,
    /// Hz DFT: [freq][j*nx + i]
    pub hz_dft: Vec<Vec<Complex64>>,
    /// Ex DFT: [freq][j*(nx+1) + i]
    pub ex_dft: Vec<Vec<Complex64>>,
    /// Ey DFT: [freq][j*nx + i]
    pub ey_dft: Vec<Vec<Complex64>>,
}

impl DftBox2d {
    pub fn new(nx: usize, ny: usize, frequencies_hz: &[f64]) -> Self {
        let omegas: Vec<f64> = frequencies_hz.iter().map(|&f| 2.0 * PI * f).collect();
        let nf = omegas.len();
        Self {
            nx,
            ny,
            omegas,
            hz_dft: vec![vec![Complex64::new(0.0, 0.0); nx * ny]; nf],
            ex_dft: vec![vec![Complex64::new(0.0, 0.0); (nx + 1) * ny]; nf],
            ey_dft: vec![vec![Complex64::new(0.0, 0.0); nx * (ny + 1)]; nf],
        }
    }

    pub fn accumulate(&mut self, grid: &Yee2dTe, t: f64, dt: f64) {
        for (k, &omega) in self.omegas.iter().enumerate() {
            let phase = Complex64::new(0.0, -omega * t).exp() * dt;
            for idx in 0..self.nx * self.ny {
                self.hz_dft[k][idx] += grid.hz[idx] * phase;
            }
            for idx in 0..(self.nx + 1) * self.ny {
                self.ex_dft[k][idx] += grid.ex[idx] * phase;
            }
            for idx in 0..self.nx * (self.ny + 1) {
                self.ey_dft[k][idx] += grid.ey[idx] * phase;
            }
        }
    }
}

/// DFT collector for 2D TM fields at multiple frequencies.
///
/// TM mode fields: Ez (nx × ny), Hx (nx × (ny+1)), Hy ((nx+1) × ny).
///
/// Accumulates the running DFT at each time step using:
///   F(ω) += field(t) · exp(-i·ω·t) · dt
///
/// The `ez_field` method returns the complex Ez snapshot at a given
/// frequency bin as a flat `Vec<Complex64>` of length `nx * ny`.
#[derive(Clone)]
pub struct DftBox2dTm {
    pub nx: usize,
    pub ny: usize,
    pub omegas: Vec<f64>,
    /// Stored dt for per-step phaser computation
    pub dt: f64,
    /// Ez DFT accumulators: [freq][j*nx + i]  (nx × ny cells)
    pub ez_dft: Vec<Vec<Complex64>>,
    /// Hx DFT accumulators: [freq][j*nx + i]  (nx × (ny+1) cells)
    pub hx_dft: Vec<Vec<Complex64>>,
    /// Hy DFT accumulators: [freq][j*(nx+1) + i]  ((nx+1) × ny cells)
    pub hy_dft: Vec<Vec<Complex64>>,
    /// Number of accumulated time steps
    pub n_steps: usize,
}

impl DftBox2dTm {
    /// Create a new TM DFT box.
    ///
    /// # Arguments
    /// * `frequencies_hz` – list of frequencies to monitor (Hz)
    /// * `nx`, `ny` – grid dimensions (must match the FDTD solver)
    /// * `dt` – simulation time step (s)
    pub fn new(frequencies_hz: &[f64], nx: usize, ny: usize, dt: f64) -> Self {
        let omegas: Vec<f64> = frequencies_hz.iter().map(|&f| 2.0 * PI * f).collect();
        let nf = omegas.len();
        Self {
            nx,
            ny,
            omegas,
            dt,
            ez_dft: vec![vec![Complex64::new(0.0, 0.0); nx * ny]; nf],
            hx_dft: vec![vec![Complex64::new(0.0, 0.0); nx * (ny + 1)]; nf],
            hy_dft: vec![vec![Complex64::new(0.0, 0.0); (nx + 1) * ny]; nf],
            n_steps: 0,
        }
    }

    /// Accumulate fields at the current time step.
    ///
    /// `step` is the integer time-step counter; the physical time is `step * dt`.
    /// `ez` must have length `nx * ny`, `hx` length `nx * (ny+1)`,
    /// `hy` length `(nx+1) * ny`.
    pub fn accumulate(&mut self, step: usize, ez: &[f64], hx: &[f64], hy: &[f64]) {
        let t = step as f64 * self.dt;
        let dt = self.dt;
        for (k, &omega) in self.omegas.iter().enumerate() {
            let phase = Complex64::new(0.0, -omega * t).exp() * dt;
            let ez_acc = &mut self.ez_dft[k];
            for (acc, &val) in ez_acc.iter_mut().zip(ez.iter()) {
                *acc += val * phase;
            }
            let hx_acc = &mut self.hx_dft[k];
            for (acc, &val) in hx_acc.iter_mut().zip(hx.iter()) {
                *acc += val * phase;
            }
            let hy_acc = &mut self.hy_dft[k];
            for (acc, &val) in hy_acc.iter_mut().zip(hy.iter()) {
                *acc += val * phase;
            }
        }
        self.n_steps += 1;
    }

    /// Return the accumulated complex Ez field at frequency bin `freq_idx`.
    ///
    /// Returns a `Vec<Complex64>` of length `nx * ny` (row-major, j × nx + i).
    /// Returns an empty Vec if `freq_idx` is out of range.
    pub fn ez_field(&self, freq_idx: usize) -> Vec<Complex64> {
        self.ez_dft.get(freq_idx).cloned().unwrap_or_default()
    }

    /// Return the accumulated complex Hx field at frequency bin `freq_idx`.
    pub fn hx_field(&self, freq_idx: usize) -> Vec<Complex64> {
        self.hx_dft.get(freq_idx).cloned().unwrap_or_default()
    }

    /// Return the accumulated complex Hy field at frequency bin `freq_idx`.
    pub fn hy_field(&self, freq_idx: usize) -> Vec<Complex64> {
        self.hy_dft.get(freq_idx).cloned().unwrap_or_default()
    }

    /// Peak |Ez(f)| across all cells for the given frequency bin.
    pub fn peak_ez_magnitude(&self, freq_idx: usize) -> f64 {
        self.ez_dft
            .get(freq_idx)
            .map(|v| v.iter().map(|c| c.norm()).fold(0.0_f64, f64::max))
            .unwrap_or(0.0)
    }
}

/// 2D TE FDTD solver (Hz, Ex, Ey)
///
/// TE mode in 2D: Hz, Ex, Ey nonzero.
///
/// Update equations:
///   dHz/dt = (1/mu) * (dEx/dy - dEy/dx)
///   dEx/dt = (1/eps) * dHz/dy
///   dEy/dt = -(1/eps) * dHz/dx
/// Time-domain field probe for 2D FDTD.
///
/// Records Hz at a specific (i, j) cell location at every time step.
pub struct FieldProbe2d {
    /// Grid cell x-index
    pub i: usize,
    /// Grid cell y-index
    pub j: usize,
    /// Recorded (time_s, Hz) pairs
    pub data: Vec<(f64, f64)>,
}

pub struct Fdtd2dTe {
    pub grid: Yee2dTe,
    pub dt: f64,
    pub time_step: usize,
    /// CPML for x-direction
    pml_x: Cpml,
    /// CPML for y-direction
    pml_y: Cpml,
    /// CPML auxiliary fields
    psi_hz_x: Vec<f64>,
    psi_hz_y: Vec<f64>,
    psi_ex_y: Vec<f64>,
    psi_ey_x: Vec<f64>,
    /// Optional TFSF source
    pub tfsf: Option<TfsfSource>,
    /// DFT monitors
    pub dft_boxes: Vec<DftBox2d>,
    /// Time-domain field probes
    pub field_probes: Vec<FieldProbe2d>,
}

impl Fdtd2dTe {
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64, boundary: &BoundaryConfig) -> Self {
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

        let hz_size = nx * ny;
        let ex_size = (nx + 1) * ny;
        let ey_size = nx * (ny + 1);

        Self {
            grid: Yee2dTe::new(nx, ny, dx, dy),
            dt,
            time_step: 0,
            pml_x,
            pml_y,
            psi_hz_x: vec![0.0; hz_size],
            psi_hz_y: vec![0.0; hz_size],
            psi_ex_y: vec![0.0; ex_size],
            psi_ey_x: vec![0.0; ey_size],
            tfsf: None,
            dft_boxes: Vec::new(),
            field_probes: Vec::new(),
        }
    }

    pub fn current_time(&self) -> f64 {
        self.time_step as f64 * self.dt
    }

    pub fn set_tfsf(&mut self, tfsf: TfsfSource) {
        self.tfsf = Some(tfsf);
    }

    pub fn add_dft_box(&mut self, dft: DftBox2d) {
        self.dft_boxes.push(dft);
    }

    /// Advance one time step
    pub fn step(&mut self) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let dx = self.grid.dx;
        let dy = self.grid.dy;
        let dt = self.dt;
        let t = self.current_time();

        // --- Advance TFSF aux grid ---
        if let Some(ref mut tfsf) = self.tfsf {
            tfsf.advance_aux(t);
        }

        // --- Update Hz field ---
        // Hz[i,j] at (i+0.5, j+0.5)
        // dHz/dt = (1/mu) * ((Ex[i,j+1] - Ex[i,j])/dy - (Ey[i+1,j] - Ey[i,j])/dx)
        for j in 0..ny {
            for i in 0..nx {
                let idx_hz = j * nx + i;
                let idx_ex_top = (j + 1) * (nx + 1) + i;
                let idx_ex_bot = j * (nx + 1) + i;

                let dex_dy = if j + 1 < ny {
                    (self.grid.ex[idx_ex_top] - self.grid.ex[idx_ex_bot]) / dy
                } else {
                    (0.0 - self.grid.ex[idx_ex_bot]) / dy
                };

                let dey_dx = if i + 1 < nx {
                    (self.grid.ey[j * nx + i + 1] - self.grid.ey[j * nx + i]) / dx
                } else {
                    (0.0 - self.grid.ey[j * nx + i]) / dx
                };

                // CPML corrections
                self.psi_hz_x[idx_hz] =
                    self.pml_x.b_h[i] * self.psi_hz_x[idx_hz] + self.pml_x.c_h[i] * dey_dx;
                self.psi_hz_y[idx_hz] =
                    self.pml_y.b_h[j] * self.psi_hz_y[idx_hz] + self.pml_y.c_h[j] * dex_dy;

                let kx = self.pml_x.kappa_h[i];
                let ky = self.pml_y.kappa_h[j];
                let mu = MU_0 * self.grid.mu_hz[idx_hz];

                self.grid.hz[idx_hz] += dt / mu
                    * (dex_dy / ky - dey_dx / kx + self.psi_hz_y[idx_hz] - self.psi_hz_x[idx_hz]);
            }
        }

        // TFSF Hz corrections (Hz update uses Ey, but for +x-propagating TE wave Ey_inc=0,
        // so no Hz correction needed at the left/right TFSF boundaries in this mode)

        // --- Update Ex field ---
        // Ex[i,j] at (i, j+0.5)
        // dEx/dt = (1/eps) * dHz/dy
        for j in 0..ny {
            for i in 0..=nx {
                let idx_ex = j * (nx + 1) + i;

                // Backward difference: Hz[j] - Hz[j-1]; at j=0 PEC gives Hz[-1]=0
                let dhz_dy = if i < nx {
                    if j > 0 {
                        (self.grid.hz[j * nx + i] - self.grid.hz[(j - 1) * nx + i]) / dy
                    } else {
                        self.grid.hz[j * nx + i] / dy
                    }
                } else {
                    0.0
                };

                self.psi_ex_y[idx_ex] =
                    self.pml_y.b_e[j] * self.psi_ex_y[idx_ex] + self.pml_y.c_e[j] * dhz_dy;
                let ky = self.pml_y.kappa_e[j];
                let eps = EPSILON_0 * self.grid.eps_ex[idx_ex];

                self.grid.ex[idx_ex] += dt / eps * (dhz_dy / ky + self.psi_ex_y[idx_ex]);
            }
        }

        // --- TFSF corrections to Ex ---
        if let Some(ref tfsf) = self.tfsf {
            let i_min = tfsf.i_min;
            let j_min = tfsf.j_min;
            let j_max = tfsf.j_max;
            // At i = i_min: Ex[i_min, j] -= dt/(eps*dx) * Hz_inc[i_min-1, j]
            // For +x plane wave: Hz_inc comes from the 1D aux grid
            // This correction accounts for incident Hz at the left TFSF boundary
            if i_min > 0 {
                for j in j_min..j_max {
                    let idx_ex = j * (nx + 1) + i_min;
                    let hz_inc = tfsf.inc_hy(i_min - 1); // Hy in aux = Hz in 2D for x-propagating wave
                    let eps = EPSILON_0 * self.grid.eps_ex[idx_ex];
                    self.grid.ex[idx_ex] -= dt / (eps * dx) * hz_inc;
                }
            }
            // At i = i_max: Ex[i_max, j] += dt/(eps*dx) * Hz_inc[i_max, j]
            for j in j_min..j_max {
                let idx_ex = j * (nx + 1) + tfsf.i_max;
                if tfsf.i_max < nx + 1 {
                    let hz_inc = tfsf.inc_hy(tfsf.i_max);
                    let eps = EPSILON_0 * self.grid.eps_ex[idx_ex];
                    self.grid.ex[idx_ex] += dt / (eps * dx) * hz_inc;
                }
            }
        }

        // --- Update Ey field ---
        // Ey[i,j] at (i+0.5, j)
        // dEy/dt = -(1/eps) * dHz/dx
        for j in 0..=ny {
            for i in 0..nx {
                let idx_ey = j * nx + i;
                // Backward difference: Hz[i] - Hz[i-1]; at i=0 PEC gives Hz[-1]=0
                let dhz_dx = if j < ny {
                    if i > 0 {
                        (self.grid.hz[j * nx + i] - self.grid.hz[j * nx + i - 1]) / dx
                    } else {
                        self.grid.hz[j * nx + i] / dx
                    }
                } else {
                    0.0
                };

                self.psi_ey_x[idx_ey] =
                    self.pml_x.b_e[i] * self.psi_ey_x[idx_ey] + self.pml_x.c_e[i] * dhz_dx;
                let kx = self.pml_x.kappa_e[i];
                let eps = EPSILON_0 * self.grid.eps_ey[idx_ey];

                self.grid.ey[idx_ey] -= dt / eps * (dhz_dx / kx + self.psi_ey_x[idx_ey]);
            }
        }

        self.time_step += 1;

        // --- Record DFT boxes ---
        let t_new = self.current_time();
        for dft in &mut self.dft_boxes {
            dft.accumulate(&self.grid, t_new, dt);
        }
        // --- Record field probes ---
        self.record_probes(t_new);
    }

    /// Run for the given number of steps
    pub fn run(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// Inject a hard source at a point (i, j) into Hz
    pub fn inject_hz(&mut self, i: usize, j: usize, val: f64) {
        if i < self.grid.nx && j < self.grid.ny {
            self.grid.hz[j * self.grid.nx + i] += val;
        }
    }

    /// Fill a rectangular box [ix0..ix1) × [iy0..iy1) with given eps_r.
    pub fn fill_eps_box(&mut self, ix0: usize, ix1: usize, iy0: usize, iy1: usize, eps_r: f64) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        for j in iy0..iy1.min(ny) {
            for i in ix0..ix1.min(nx) {
                self.grid.eps_ex[(j + 1) * (nx + 1) + i] = eps_r;
                self.grid.eps_ey[j * nx + i] = eps_r;
            }
        }
    }

    /// Extract a horizontal slice (fixed row j) of Hz.
    pub fn hz_row(&self, j: usize) -> Vec<f64> {
        let nx = self.grid.nx;
        (0..nx).map(|i| self.grid.hz[j * nx + i]).collect()
    }

    /// Extract a vertical slice (fixed column i) of Hz.
    pub fn hz_col(&self, i: usize) -> Vec<f64> {
        let ny = self.grid.ny;
        let nx = self.grid.nx;
        (0..ny).map(|j| self.grid.hz[j * nx + i]).collect()
    }

    /// Peak |Hz| over the whole grid.
    pub fn peak_hz(&self) -> f64 {
        self.grid
            .hz
            .iter()
            .cloned()
            .fold(0.0_f64, |a, v| a.max(v.abs()))
    }

    /// Peak |Ex| over the whole grid.
    pub fn peak_ex(&self) -> f64 {
        self.grid
            .ex
            .iter()
            .cloned()
            .fold(0.0_f64, |a, v| a.max(v.abs()))
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Parallel field updates (feature-gated)
    // ──────────────────────────────────────────────────────────────────────────

    /// Parallel H-field update for 2D TE mode (Hz) with CPML.
    ///
    /// Uses snapshot/par_iter/sequential-writeback pattern.
    /// Produces the same result as the serial step() Hz-update.
    #[cfg(feature = "parallel")]
    pub fn update_h_parallel(&mut self) {
        use rayon::prelude::*;

        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let dx = self.grid.dx;
        let dy = self.grid.dy;
        let dt = self.dt;

        // Snapshot arrays
        let hz_snap = self.grid.hz.clone();
        let ex_snap = self.grid.ex.clone();
        let ey_snap = self.grid.ey.clone();
        let mu_hz_snap = self.grid.mu_hz.clone();
        let psi_hz_x_snap = self.psi_hz_x.clone();
        let psi_hz_y_snap = self.psi_hz_y.clone();
        let pml_x_b_h = self.pml_x.b_h.clone();
        let pml_x_c_h = self.pml_x.c_h.clone();
        let pml_x_kappa_h = self.pml_x.kappa_h.clone();
        let pml_y_b_h = self.pml_y.b_h.clone();
        let pml_y_c_h = self.pml_y.c_h.clone();
        let pml_y_kappa_h = self.pml_y.kappa_h.clone();

        // Hz update: j in 0..ny, i in 0..nx
        // hz_idx = j * nx + i
        // dHz/dt = (1/mu) * ((Ex[i,j+1] - Ex[i,j])/dy - (Ey[i+1,j] - Ey[i,j])/dx)
        // ex_idx_top = (j+1)*(nx+1)+i, ex_idx_bot = j*(nx+1)+i
        // ey_idx at i+1 = j*nx+(i+1), at i = j*nx+i
        let hz_updates: Vec<(usize, f64, f64, f64)> = (0..ny)
            .into_par_iter()
            .flat_map(|j| {
                let hz_snap = &hz_snap;
                let ex_snap = &ex_snap;
                let ey_snap = &ey_snap;
                let mu_hz_snap = &mu_hz_snap;
                let psi_x_snap = &psi_hz_x_snap;
                let psi_y_snap = &psi_hz_y_snap;
                let b_h_x = &pml_x_b_h;
                let c_h_x = &pml_x_c_h;
                let kappa_h_x = &pml_x_kappa_h;
                let b_h_y = &pml_y_b_h;
                let c_h_y = &pml_y_c_h;
                let kappa_h_y = &pml_y_kappa_h;
                (0..nx)
                    .map(move |i| {
                        let idx_hz = j * nx + i;
                        let idx_ex_top = (j + 1) * (nx + 1) + i;
                        let idx_ex_bot = j * (nx + 1) + i;

                        let dex_dy = if j + 1 < ny {
                            (ex_snap[idx_ex_top] - ex_snap[idx_ex_bot]) / dy
                        } else {
                            (0.0 - ex_snap[idx_ex_bot]) / dy
                        };

                        let dey_dx = if i + 1 < nx {
                            (ey_snap[j * nx + i + 1] - ey_snap[j * nx + i]) / dx
                        } else {
                            (0.0 - ey_snap[j * nx + i]) / dx
                        };

                        let psi_hz_x_new = b_h_x[i] * psi_x_snap[idx_hz] + c_h_x[i] * dey_dx;
                        let psi_hz_y_new = b_h_y[j] * psi_y_snap[idx_hz] + c_h_y[j] * dex_dy;

                        let kx = kappa_h_x[i];
                        let ky = kappa_h_y[j];
                        let mu = MU_0 * mu_hz_snap[idx_hz];

                        let hz_new = hz_snap[idx_hz]
                            + dt / mu * (dex_dy / ky - dey_dx / kx + psi_hz_y_new - psi_hz_x_new);
                        (idx_hz, hz_new, psi_hz_x_new, psi_hz_y_new)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        for (idx, hz_new, psi_x_new, psi_y_new) in hz_updates {
            self.grid.hz[idx] = hz_new;
            self.psi_hz_x[idx] = psi_x_new;
            self.psi_hz_y[idx] = psi_y_new;
        }
    }

    /// Parallel E-field update for 2D TE mode (Ex, Ey) with CPML.
    ///
    /// Uses snapshot/par_iter/sequential-writeback pattern.
    /// Produces the same result as the serial step() Ex/Ey-update.
    #[cfg(feature = "parallel")]
    pub fn update_e_parallel(&mut self) {
        use rayon::prelude::*;

        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let dx = self.grid.dx;
        let dy = self.grid.dy;
        let dt = self.dt;

        let hz_snap = self.grid.hz.clone();
        let ex_snap = self.grid.ex.clone();
        let ey_snap = self.grid.ey.clone();
        let eps_ex_snap = self.grid.eps_ex.clone();
        let eps_ey_snap = self.grid.eps_ey.clone();
        let psi_ex_y_snap = self.psi_ex_y.clone();
        let psi_ey_x_snap = self.psi_ey_x.clone();
        let pml_x_b_e = self.pml_x.b_e.clone();
        let pml_x_c_e = self.pml_x.c_e.clone();
        let pml_x_kappa_e = self.pml_x.kappa_e.clone();
        let pml_y_b_e = self.pml_y.b_e.clone();
        let pml_y_c_e = self.pml_y.c_e.clone();
        let pml_y_kappa_e = self.pml_y.kappa_e.clone();

        // Ex update: j in 0..ny, i in 0..=nx
        // ex_idx = j*(nx+1)+i
        // dEx/dt = (1/eps) * dHz/dy   (backward diff: Hz[j] - Hz[j-1])
        let ex_updates: Vec<(usize, f64, f64)> = (0..ny)
            .into_par_iter()
            .flat_map(|j| {
                let hz_snap = &hz_snap;
                let ex_snap = &ex_snap;
                let eps_ex_snap = &eps_ex_snap;
                let psi_snap = &psi_ex_y_snap;
                let b_e = &pml_y_b_e;
                let c_e = &pml_y_c_e;
                let kappa_e = &pml_y_kappa_e;
                (0..=nx)
                    .map(move |i| {
                        let idx_ex = j * (nx + 1) + i;
                        let dhz_dy = if i < nx {
                            if j > 0 {
                                (hz_snap[j * nx + i] - hz_snap[(j - 1) * nx + i]) / dy
                            } else {
                                hz_snap[j * nx + i] / dy
                            }
                        } else {
                            0.0
                        };
                        let psi_new = b_e[j] * psi_snap[idx_ex] + c_e[j] * dhz_dy;
                        let ky = kappa_e[j];
                        let eps = EPSILON_0 * eps_ex_snap[idx_ex];
                        let ex_new = ex_snap[idx_ex] + dt / eps * (dhz_dy / ky + psi_new);
                        (idx_ex, ex_new, psi_new)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        for (idx, ex_new, psi_new) in ex_updates {
            self.grid.ex[idx] = ex_new;
            self.psi_ex_y[idx] = psi_new;
        }

        // Ey update: j in 0..=ny, i in 0..nx
        // ey_idx = j*nx+i
        // dEy/dt = -(1/eps) * dHz/dx   (backward diff: Hz[i] - Hz[i-1])
        let ey_updates: Vec<(usize, f64, f64)> = (0..=ny)
            .into_par_iter()
            .flat_map(|j| {
                let hz_snap = &hz_snap;
                let ey_snap = &ey_snap;
                let eps_ey_snap = &eps_ey_snap;
                let psi_snap = &psi_ey_x_snap;
                let b_e = &pml_x_b_e;
                let c_e = &pml_x_c_e;
                let kappa_e = &pml_x_kappa_e;
                (0..nx)
                    .map(move |i| {
                        let idx_ey = j * nx + i;
                        let dhz_dx = if j < ny {
                            if i > 0 {
                                (hz_snap[j * nx + i] - hz_snap[j * nx + i - 1]) / dx
                            } else {
                                hz_snap[j * nx + i] / dx
                            }
                        } else {
                            0.0
                        };
                        let psi_new = b_e[i] * psi_snap[idx_ey] + c_e[i] * dhz_dx;
                        let kx = kappa_e[i];
                        let eps = EPSILON_0 * eps_ey_snap[idx_ey];
                        let ey_new = ey_snap[idx_ey] - dt / eps * (dhz_dx / kx + psi_new);
                        (idx_ey, ey_new, psi_new)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        for (idx, ey_new, psi_new) in ey_updates {
            self.grid.ey[idx] = ey_new;
            self.psi_ey_x[idx] = psi_new;
        }

        self.time_step += 1;
    }

    /// Total electromagnetic energy.
    pub fn total_energy(&self) -> f64 {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let dx = self.grid.dx;
        let dy = self.grid.dy;
        let dv = dx * dy;
        let e_hz: f64 = self.grid.hz.iter().map(|h| h * h).sum::<f64>() * 0.5 * MU_0 * dv;
        let e_ex: f64 = self
            .grid
            .ex
            .iter()
            .zip(self.grid.eps_ex.iter())
            .map(|(e, &eps)| eps * e * e)
            .sum::<f64>()
            * 0.5
            * EPSILON_0
            * dv;
        let e_ey: f64 = self
            .grid
            .ey
            .iter()
            .zip(self.grid.eps_ey.iter())
            .map(|(e, &eps)| eps * e * e)
            .sum::<f64>()
            * 0.5
            * EPSILON_0
            * dv;
        let _ = (nx, ny);
        e_hz + e_ex + e_ey
    }

    /// Add a time-domain field probe at (i, j). Returns the probe index.
    ///
    /// After simulation, retrieve the recorded (time, Hz) pairs via
    /// `get_probe_time_series(idx)`.
    pub fn add_field_probe(&mut self, i: usize, j: usize) -> usize {
        let idx = self.field_probes.len();
        self.field_probes.push(FieldProbe2d {
            i,
            j,
            data: Vec::new(),
        });
        idx
    }

    /// Get the time-series data for field probe `idx`.
    ///
    /// Returns a slice of `(time_s, hz_value)` pairs recorded once per step.
    pub fn get_probe_time_series(&self, idx: usize) -> Option<&[(f64, f64)]> {
        self.field_probes.get(idx).map(|p| p.data.as_slice())
    }

    /// Compute the DFT of Hz at a single point (i, j) over a list of angular frequencies.
    ///
    /// Requires that a simulation has been run already (uses the last-stored time series
    /// from a field probe, or runs a fresh integration over the current fields).
    ///
    /// Returns `Vec<(re, im)>` complex DFT values at each requested frequency.
    pub fn dft_at_point(&self, i: usize, j: usize, omega_list: &[f64]) -> Vec<(f64, f64)> {
        let nx = self.grid.nx;
        if i >= nx || j >= self.grid.ny {
            return omega_list.iter().map(|_| (0.0, 0.0)).collect();
        }
        // Search for an existing probe at (i, j)
        let probe_opt = self.field_probes.iter().find(|p| p.i == i && p.j == j);
        if let Some(probe) = probe_opt {
            let data = &probe.data;
            omega_list
                .iter()
                .map(|&omega| {
                    let mut re = 0.0_f64;
                    let mut im = 0.0_f64;
                    for &(t, hz) in data {
                        re += hz * (omega * t).cos();
                        im -= hz * (omega * t).sin();
                    }
                    // Normalize by number of samples
                    let n = data.len().max(1) as f64;
                    (re / n, im / n)
                })
                .collect()
        } else {
            // No probe at this location — return zeros
            omega_list.iter().map(|_| (0.0, 0.0)).collect()
        }
    }

    /// Fill permittivity using a closure: f(i, j) → ε_r.
    ///
    /// Sets both ε_ex and ε_ey in the Yee grid based on the closure value at each (i, j).
    pub fn fill_eps_fn(&mut self, f: impl Fn(usize, usize) -> f64) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        for j in 0..ny {
            for i in 0..nx {
                let eps = f(i, j).max(1.0); // ensure physically valid
                self.grid.eps_ex[(j + 1) * (nx + 1) + i] = eps;
                self.grid.eps_ey[j * nx + i] = eps;
            }
        }
    }

    /// Record field probes (called internally from `step()`).
    fn record_probes(&mut self, t: f64) {
        let nx = self.grid.nx;
        for probe in &mut self.field_probes {
            if probe.i < nx && probe.j < self.grid.ny {
                let hz = self.grid.hz[probe.j * nx + probe.i];
                probe.data.push((t, hz));
            }
        }
    }

    /// Return the RMS value of the Hz field (useful for power estimation).
    pub fn hz_rms(&self) -> f64 {
        let n = self.grid.hz.len();
        if n == 0 {
            return 0.0;
        }
        let sum_sq: f64 = self.grid.hz.iter().map(|h| h * h).sum();
        (sum_sq / n as f64).sqrt()
    }

    /// Return the 2D field snapshot as a flat Vec (Hz values, row-major).
    pub fn hz_snapshot(&self) -> Vec<f64> {
        self.grid.hz.clone()
    }

    /// Return the maximum |Ex| at position (i, j) across Ex grid.
    pub fn ex_at(&self, i: usize, j: usize) -> f64 {
        let nx = self.grid.nx + 1;
        if i < nx && j < self.grid.ny + 1 {
            self.grid.ex[j * nx + i]
        } else {
            0.0
        }
    }

    /// Return Hz at position (i, j).
    pub fn hz_at(&self, i: usize, j: usize) -> f64 {
        let nx = self.grid.nx;
        if i < nx && j < self.grid.ny {
            self.grid.hz[j * nx + i]
        } else {
            0.0
        }
    }
}

/// 2D TM FDTD solver: Ez, Hx, Hy fields.
///
/// TM mode in 2D: Ez, Hx, Hy are nonzero.
///
/// Update equations (Yee TM):
///   dHx/dt = -(1/mu) * dEz/dy
///   dHy/dt = +(1/mu) * dEz/dx
///   dEz/dt = (1/eps) * (dHy/dx - dHx/dy)
pub struct Fdtd2dTm {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    pub dt: f64,
    pub time_step: usize,
    /// Ez field (nx × ny)
    pub ez: Vec<f64>,
    /// Hx field (nx × (ny+1))
    pub hx: Vec<f64>,
    /// Hy field ((nx+1) × ny)
    pub hy: Vec<f64>,
    /// Relative permittivity at Ez cells
    pub eps_r: Vec<f64>,
    /// CPML for x
    pml_x: Cpml,
    /// CPML for y
    pml_y: Cpml,
    psi_ez_x: Vec<f64>,
    psi_ez_y: Vec<f64>,
    psi_hx_y: Vec<f64>,
    psi_hy_x: Vec<f64>,
    /// Time-domain field probes (Ez)
    pub tm_probes: Vec<TmFieldProbe2d>,
}

impl Fdtd2dTm {
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64, boundary: &BoundaryConfig) -> Self {
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
        let n_ez = nx * ny;
        let n_hx = nx * (ny + 1);
        let n_hy = (nx + 1) * ny;
        Self {
            nx,
            ny,
            dx,
            dy,
            dt,
            time_step: 0,
            ez: vec![0.0; n_ez],
            hx: vec![0.0; n_hx],
            hy: vec![0.0; n_hy],
            eps_r: vec![1.0; n_ez],
            pml_x,
            pml_y,
            psi_ez_x: vec![0.0; n_ez],
            psi_ez_y: vec![0.0; n_ez],
            psi_hx_y: vec![0.0; n_hx],
            psi_hy_x: vec![0.0; n_hy],
            tm_probes: Vec::new(),
        }
    }

    pub fn current_time(&self) -> f64 {
        self.time_step as f64 * self.dt
    }

    fn ez_idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }
    fn hx_idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }
    fn hy_idx(&self, i: usize, j: usize) -> usize {
        j * (self.nx + 1) + i
    }

    pub fn inject_ez(&mut self, i: usize, j: usize, val: f64) {
        if i < self.nx && j < self.ny {
            let idx = j * self.nx + i;
            self.ez[idx] += val;
        }
    }

    pub fn fill_eps_box(&mut self, ix0: usize, ix1: usize, iy0: usize, iy1: usize, eps_r: f64) {
        for j in iy0..iy1.min(self.ny) {
            for i in ix0..ix1.min(self.nx) {
                let idx = j * self.nx + i;
                self.eps_r[idx] = eps_r;
            }
        }
    }

    pub fn step(&mut self) {
        let (nx, ny, dx, dy, dt) = (self.nx, self.ny, self.dx, self.dy, self.dt);

        // Update Hx: Hx[i,j] at (i, j+0.5)
        // dHx/dt = -(1/mu) * (Ez[i,j+1] - Ez[i,j]) / dy
        for j in 0..ny {
            for i in 0..nx {
                let idx_hx = self.hx_idx(i, j);
                let ez_j1 = if j + 1 < ny {
                    self.ez[self.ez_idx(i, j + 1)]
                } else {
                    0.0
                };
                let dez_dy = (ez_j1 - self.ez[self.ez_idx(i, j)]) / dy;
                self.psi_hx_y[idx_hx] =
                    self.pml_y.b_h[j] * self.psi_hx_y[idx_hx] + self.pml_y.c_h[j] * dez_dy;
                let ky = self.pml_y.kappa_h[j];
                self.hx[idx_hx] -= dt / MU_0 * (dez_dy / ky + self.psi_hx_y[idx_hx]);
            }
        }

        // Update Hy: Hy[i,j] at (i+0.5, j)
        // dHy/dt = (1/mu) * (Ez[i+1,j] - Ez[i,j]) / dx
        for j in 0..ny {
            for i in 0..nx {
                let idx_hy = self.hy_idx(i, j);
                let ez_i1 = if i + 1 < nx {
                    self.ez[self.ez_idx(i + 1, j)]
                } else {
                    0.0
                };
                let dez_dx = (ez_i1 - self.ez[self.ez_idx(i, j)]) / dx;
                self.psi_hy_x[idx_hy] =
                    self.pml_x.b_h[i] * self.psi_hy_x[idx_hy] + self.pml_x.c_h[i] * dez_dx;
                let kx = self.pml_x.kappa_h[i];
                self.hy[idx_hy] += dt / MU_0 * (dez_dx / kx + self.psi_hy_x[idx_hy]);
            }
        }

        // Update Ez: Ez[i,j] at (i, j)
        // dEz/dt = (1/eps) * ((Hy[i,j] - Hy[i-1,j])/dx - (Hx[i,j] - Hx[i,j-1])/dy)
        for j in 0..ny {
            for i in 0..nx {
                let idx = self.ez_idx(i, j);
                let dhy_dx = if i > 0 {
                    (self.hy[self.hy_idx(i, j)] - self.hy[self.hy_idx(i - 1, j)]) / dx
                } else {
                    self.hy[self.hy_idx(i, j)] / dx
                };
                let dhx_dy = if j > 0 {
                    (self.hx[self.hx_idx(i, j)] - self.hx[self.hx_idx(i, j - 1)]) / dy
                } else {
                    self.hx[self.hx_idx(i, j)] / dy
                };
                self.psi_ez_x[idx] =
                    self.pml_x.b_e[i] * self.psi_ez_x[idx] + self.pml_x.c_e[i] * dhy_dx;
                self.psi_ez_y[idx] =
                    self.pml_y.b_e[j] * self.psi_ez_y[idx] + self.pml_y.c_e[j] * dhx_dy;
                let kx = self.pml_x.kappa_e[i];
                let ky = self.pml_y.kappa_e[j];
                let eps = EPSILON_0 * self.eps_r[idx];
                self.ez[idx] += dt / eps
                    * ((dhy_dx / kx + self.psi_ez_x[idx]) - (dhx_dy / ky + self.psi_ez_y[idx]));
            }
        }
        self.time_step += 1;
        // Record TM probes
        let t_probe = self.current_time();
        self.record_tm_probes(t_probe);
    }

    pub fn run(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    pub fn peak_ez(&self) -> f64 {
        self.ez.iter().cloned().fold(0.0_f64, |a, v| a.max(v.abs()))
    }

    /// Total electromagnetic energy in the TM domain.
    ///
    /// U = 0.5 * (ε₀·ε_r·Ez² + μ₀·(Hx² + Hy²)) summed over all cells × dV.
    pub fn total_energy_tm(&self) -> f64 {
        let dv = self.dx * self.dy;
        let e_ez: f64 = self
            .ez
            .iter()
            .zip(self.eps_r.iter())
            .map(|(e, &eps)| EPSILON_0 * eps * e * e)
            .sum::<f64>()
            * 0.5
            * dv;
        let e_hx: f64 = self.hx.iter().map(|h| h * h).sum::<f64>() * 0.5 * MU_0 * dv;
        let e_hy: f64 = self.hy.iter().map(|h| h * h).sum::<f64>() * 0.5 * MU_0 * dv;
        e_ez + e_hx + e_hy
    }

    /// Add a time-domain field probe at (i, j) — records Ez.
    ///
    /// Returns probe index. Retrieve data with `get_probe_ez_series(idx)`.
    pub fn add_field_probe(&mut self, i: usize, j: usize) -> usize {
        let idx = self.tm_probes.len();
        self.tm_probes.push(TmFieldProbe2d {
            i,
            j,
            data: Vec::new(),
        });
        idx
    }

    /// Get the recorded (time, Ez) pairs for probe `idx`.
    pub fn get_probe_ez_series(&self, idx: usize) -> Option<&[(f64, f64)]> {
        self.tm_probes.get(idx).map(|p| p.data.as_slice())
    }

    /// Fill permittivity using a closure: f(i, j) → ε_r.
    pub fn fill_eps_fn(&mut self, f: impl Fn(usize, usize) -> f64) {
        let nx = self.nx;
        let ny = self.ny;
        for j in 0..ny {
            for i in 0..nx {
                let eps = f(i, j).max(1.0);
                self.eps_r[j * nx + i] = eps;
            }
        }
    }

    /// Ez value at (i, j).
    pub fn ez_at(&self, i: usize, j: usize) -> f64 {
        if i < self.nx && j < self.ny {
            self.ez[j * self.nx + i]
        } else {
            0.0
        }
    }

    /// RMS value of Ez field.
    pub fn ez_rms(&self) -> f64 {
        let n = self.ez.len();
        if n == 0 {
            return 0.0;
        }
        let sum_sq: f64 = self.ez.iter().map(|e| e * e).sum();
        (sum_sq / n as f64).sqrt()
    }

    /// Internal: record TM field probes after each step.
    fn record_tm_probes(&mut self, t: f64) {
        let nx = self.nx;
        for probe in &mut self.tm_probes {
            if probe.i < nx && probe.j < self.ny {
                let ez = self.ez[probe.j * nx + probe.i];
                probe.data.push((t, ez));
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Parallel field updates (feature-gated)
    // ──────────────────────────────────────────────────────────────────────────

    /// Parallel H-field update for 2D TM mode (Hx, Hy) with CPML.
    ///
    /// Uses snapshot/par_iter/sequential-writeback pattern identical to Fdtd3d.
    /// Produces the same result as the serial step() H-update.
    #[cfg(feature = "parallel")]
    pub fn update_h_parallel(&mut self) {
        use rayon::prelude::*;

        let nx = self.nx;
        let ny = self.ny;
        let dx = self.dx;
        let dy = self.dy;
        let dt = self.dt;

        // Snapshot all arrays needed for read access
        let ez_snap = self.ez.clone();
        let hx_snap = self.hx.clone();
        let hy_snap = self.hy.clone();
        let psi_hx_y_snap = self.psi_hx_y.clone();
        let psi_hy_x_snap = self.psi_hy_x.clone();
        let pml_y_b_h = self.pml_y.b_h.clone();
        let pml_y_c_h = self.pml_y.c_h.clone();
        let pml_y_kappa_h = self.pml_y.kappa_h.clone();
        let pml_x_b_h = self.pml_x.b_h.clone();
        let pml_x_c_h = self.pml_x.c_h.clone();
        let pml_x_kappa_h = self.pml_x.kappa_h.clone();

        // Compute Hx updates: Hx[i,j] at j in 0..ny, i in 0..nx
        // hx_idx(i,j) = j * nx + i  (size nx*(ny+1), but we update 0..ny rows)
        let hx_updates: Vec<(usize, f64, f64)> = (0..ny)
            .into_par_iter()
            .flat_map(|j| {
                let ez_snap = &ez_snap;
                let hx_snap = &hx_snap;
                let psi_snap = &psi_hx_y_snap;
                let b_h = &pml_y_b_h;
                let c_h = &pml_y_c_h;
                let kappa_h = &pml_y_kappa_h;
                (0..nx)
                    .map(move |i| {
                        let idx_hx = j * nx + i;
                        let ez_j1 = if j + 1 < ny {
                            ez_snap[(j + 1) * nx + i]
                        } else {
                            0.0
                        };
                        let dez_dy = (ez_j1 - ez_snap[j * nx + i]) / dy;
                        let psi_new = b_h[j] * psi_snap[idx_hx] + c_h[j] * dez_dy;
                        let ky = kappa_h[j];
                        let hx_new = hx_snap[idx_hx] - dt / MU_0 * (dez_dy / ky + psi_new);
                        (idx_hx, hx_new, psi_new)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        for (idx, hx_new, psi_new) in hx_updates {
            self.hx[idx] = hx_new;
            self.psi_hx_y[idx] = psi_new;
        }

        // Compute Hy updates: Hy[i,j] at j in 0..ny, i in 0..nx
        // hy_idx(i,j) = j * (nx+1) + i
        let hy_updates: Vec<(usize, f64, f64)> = (0..ny)
            .into_par_iter()
            .flat_map(|j| {
                let ez_snap = &ez_snap;
                let hy_snap = &hy_snap;
                let psi_snap = &psi_hy_x_snap;
                let b_h = &pml_x_b_h;
                let c_h = &pml_x_c_h;
                let kappa_h = &pml_x_kappa_h;
                (0..nx)
                    .map(move |i| {
                        let idx_hy = j * (nx + 1) + i;
                        let ez_i1 = if i + 1 < nx {
                            ez_snap[j * nx + i + 1]
                        } else {
                            0.0
                        };
                        let dez_dx = (ez_i1 - ez_snap[j * nx + i]) / dx;
                        let psi_new = b_h[i] * psi_snap[idx_hy] + c_h[i] * dez_dx;
                        let kx = kappa_h[i];
                        let hy_new = hy_snap[idx_hy] + dt / MU_0 * (dez_dx / kx + psi_new);
                        (idx_hy, hy_new, psi_new)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        for (idx, hy_new, psi_new) in hy_updates {
            self.hy[idx] = hy_new;
            self.psi_hy_x[idx] = psi_new;
        }
    }

    /// Parallel E-field update for 2D TM mode (Ez) with CPML.
    ///
    /// Uses snapshot/par_iter/sequential-writeback pattern.
    /// Produces the same result as the serial step() E-update.
    #[cfg(feature = "parallel")]
    pub fn update_e_parallel(&mut self) {
        use rayon::prelude::*;

        let nx = self.nx;
        let ny = self.ny;
        let dx = self.dx;
        let dy = self.dy;
        let dt = self.dt;

        let ez_snap = self.ez.clone();
        let hx_snap = self.hx.clone();
        let hy_snap = self.hy.clone();
        let eps_r_snap = self.eps_r.clone();
        let psi_ez_x_snap = self.psi_ez_x.clone();
        let psi_ez_y_snap = self.psi_ez_y.clone();
        let pml_x_b_e = self.pml_x.b_e.clone();
        let pml_x_c_e = self.pml_x.c_e.clone();
        let pml_x_kappa_e = self.pml_x.kappa_e.clone();
        let pml_y_b_e = self.pml_y.b_e.clone();
        let pml_y_c_e = self.pml_y.c_e.clone();
        let pml_y_kappa_e = self.pml_y.kappa_e.clone();

        // Ez update: j in 0..ny, i in 0..nx
        // dEz/dt = (1/eps) * ((Hy[i,j] - Hy[i-1,j])/dx - (Hx[i,j] - Hx[i,j-1])/dy)
        let ez_updates: Vec<(usize, f64, f64, f64)> = (0..ny)
            .into_par_iter()
            .flat_map(|j| {
                let ez_snap = &ez_snap;
                let hx_snap = &hx_snap;
                let hy_snap = &hy_snap;
                let eps_r_snap = &eps_r_snap;
                let psi_x_snap = &psi_ez_x_snap;
                let psi_y_snap = &psi_ez_y_snap;
                let b_e_x = &pml_x_b_e;
                let c_e_x = &pml_x_c_e;
                let kappa_e_x = &pml_x_kappa_e;
                let b_e_y = &pml_y_b_e;
                let c_e_y = &pml_y_c_e;
                let kappa_e_y = &pml_y_kappa_e;
                (0..nx)
                    .map(move |i| {
                        let idx = j * nx + i;
                        // hy_idx(i,j) = j*(nx+1)+i, hy_idx(i-1,j) = j*(nx+1)+(i-1)
                        let dhy_dx = if i > 0 {
                            (hy_snap[j * (nx + 1) + i] - hy_snap[j * (nx + 1) + i - 1]) / dx
                        } else {
                            hy_snap[j * (nx + 1) + i] / dx
                        };
                        // hx_idx(i,j) = j*nx+i, hx_idx(i,j-1) = (j-1)*nx+i
                        let dhx_dy = if j > 0 {
                            (hx_snap[j * nx + i] - hx_snap[(j - 1) * nx + i]) / dy
                        } else {
                            hx_snap[j * nx + i] / dy
                        };
                        let psi_x_new = b_e_x[i] * psi_x_snap[idx] + c_e_x[i] * dhy_dx;
                        let psi_y_new = b_e_y[j] * psi_y_snap[idx] + c_e_y[j] * dhx_dy;
                        let kx = kappa_e_x[i];
                        let ky = kappa_e_y[j];
                        let eps = EPSILON_0 * eps_r_snap[idx];
                        let ez_new = ez_snap[idx]
                            + dt / eps * ((dhy_dx / kx + psi_x_new) - (dhx_dy / ky + psi_y_new));
                        (idx, ez_new, psi_x_new, psi_y_new)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        for (idx, ez_new, psi_x_new, psi_y_new) in ez_updates {
            self.ez[idx] = ez_new;
            self.psi_ez_x[idx] = psi_x_new;
            self.psi_ez_y[idx] = psi_y_new;
        }

        self.time_step += 1;
    }

    /// Total electromagnetic energy (alias for `total_energy_tm`).
    pub fn total_energy(&self) -> f64 {
        self.total_energy_tm()
    }
}

/// Time-domain field probe for 2D TM FDTD.
pub struct TmFieldProbe2d {
    pub i: usize,
    pub j: usize,
    pub data: Vec<(f64, f64)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fdtd::source::{GaussianEnvelope, SourceWaveform};

    fn basic_solver(nx: usize, ny: usize) -> Fdtd2dTe {
        let d = 10e-9;
        Fdtd2dTe::new(nx, ny, d, d, &BoundaryConfig::pml(15))
    }

    #[test]
    fn fdtd2d_runs_without_panic() {
        let mut solver = basic_solver(80, 80);
        solver.run(100);
        assert!(solver.grid.hz.iter().all(|&v| v.is_finite()));
        assert!(solver.grid.ex.iter().all(|&v| v.is_finite()));
        assert!(solver.grid.ey.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn fdtd2d_fields_start_zero() {
        let solver = basic_solver(40, 40);
        assert!(solver.grid.hz.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn fdtd2d_with_point_source() {
        let mut solver = basic_solver(80, 80);
        let pulse = GaussianEnvelope::new(15.0 * solver.dt, 5.0 * solver.dt);
        for step in 0..200 {
            let amp = pulse.amplitude(step as f64 * solver.dt);
            solver.inject_hz(40, 40, amp);
            solver.step();
        }
        let max_hz: f64 = solver.grid.hz.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(max_hz.is_finite());
    }

    #[test]
    fn fdtd2d_hz_row_length() {
        let solver = basic_solver(40, 30);
        let row = solver.hz_row(10);
        assert_eq!(row.len(), 40);
    }

    #[test]
    fn fdtd2d_fill_eps_box() {
        let mut solver = basic_solver(40, 40);
        solver.fill_eps_box(10, 30, 10, 30, 2.25);
        // Check eps_ex was updated in the region
        let idx_ex = 16 * (40 + 1) + 15;
        assert!((solver.grid.eps_ex[idx_ex] - 2.25).abs() < 1e-10);
    }

    #[test]
    fn fdtd2d_total_energy_zero_initially() {
        let solver = basic_solver(40, 40);
        assert_eq!(solver.total_energy(), 0.0);
    }

    #[test]
    fn fdtd2d_tm_runs_without_panic() {
        let d = 20e-9;
        let mut solver = Fdtd2dTm::new(60, 60, d, d, &BoundaryConfig::pml(10));
        let pulse = GaussianEnvelope::new(20.0 * solver.dt, 6.0 * solver.dt);
        for step in 0..200 {
            let amp = pulse.amplitude(step as f64 * solver.dt);
            solver.inject_ez(30, 30, amp);
            solver.step();
        }
        assert!(solver.ez.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn fdtd2d_tm_fields_start_zero() {
        let d = 10e-9;
        let solver = Fdtd2dTm::new(40, 40, d, d, &BoundaryConfig::pml(10));
        assert!(solver.ez.iter().all(|&v| v == 0.0));
        assert!(solver.hx.iter().all(|&v| v == 0.0));
        assert!(solver.hy.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn fdtd2d_tm_fill_eps_box() {
        let d = 10e-9;
        let mut solver = Fdtd2dTm::new(40, 40, d, d, &BoundaryConfig::pml(8));
        solver.fill_eps_box(10, 30, 10, 30, 4.0);
        assert!((solver.eps_r[15 * 40 + 15] - 4.0).abs() < 1e-10);
    }
}
