//! 3-D adjoint sensitivity solver — full (Ex, Ey, Ez) vector-field support.
//!
//! This module contains the 3-D design region, design variable, FDTD source
//! configuration, and the full `AdjointSolver3d` implementation including:
//!
//!   * Phase 9 Ez-only API (`run_forward`, `run_adjoint`, `compute_gradient`)
//!   * Phase 10 vector-field API (`run_forward_vector`, `run_adjoint_vector`,
//!     `compute_gradient_vector`) with mode-source pattern injection

use crate::error::OxiPhotonError;
use num_complex::Complex64;

// ─────────────────────────────────────────────────────────────────────────────
// Type aliases to reduce complexity in function signatures
// ─────────────────────────────────────────────────────────────────────────────

/// Return type of `build_fdtd_sim`: `(sim, total_nx, total_ny, total_nz, off_x, off_y, off_z)`.
type FdtdSimResult = (
    crate::fdtd::dims::fdtd_3d::Fdtd3d,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
);

/// Return type of `resolve_source`: list of FDTD cell coordinates and per-cell amplitudes.
type SourceCells = (Vec<(usize, usize, usize)>, Vec<[Complex64; 3]>);

// ─────────────────────────────────────────────────────────────────────────────
// DesignRegion3d
// ─────────────────────────────────────────────────────────────────────────────

/// 3-D design region for topology optimisation.
///
/// Stores a `nx × ny × nz` grid of normalised design variables ρ ∈ \[0, 1\].
/// The flat cell index matches `Fdtd3d::idx`:
///
///   `cell = i + j * nx + k * nx * ny`
///
/// which equals `k * (nx * ny) + j * nx + i`, so ε arrays can be fed
/// directly into `Fdtd3d::eps_r` without reindexing.
#[derive(Debug, Clone)]
pub struct DesignRegion3d {
    /// Grid extents
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Cell size (m)
    pub dx: f64,
    /// Minimum permittivity (void)
    pub eps_min: f64,
    /// Maximum permittivity (material)
    pub eps_max: f64,
    /// Design variable ρ(i,j,k) ∈ \[0, 1\] per cell (k-major: index = `i + j*nx + k*nx*ny`)
    pub rho: Vec<f64>,
}

impl DesignRegion3d {
    /// Create a design region filled with ρ = 0.5.
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, eps_min: f64, eps_max: f64) -> Self {
        Self {
            nx,
            ny,
            nz,
            dx,
            eps_min,
            eps_max,
            rho: vec![0.5; nx * ny * nz],
        }
    }

    /// Create a uniformly-filled region with the given ρ value.
    pub fn uniform(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        eps_min: f64,
        eps_max: f64,
        rho: f64,
    ) -> Self {
        let mut r = Self::new(nx, ny, nz, dx, eps_min, eps_max);
        for v in &mut r.rho {
            *v = rho.clamp(0.0, 1.0);
        }
        r
    }

    /// Flat cell index: `i + j * nx + k * nx * ny` — matches `Fdtd3d::idx`.
    #[inline]
    pub fn cell_idx(&self, i: usize, j: usize, k: usize) -> usize {
        i + j * self.nx + k * self.nx * self.ny
    }

    /// Permittivity at cell (i, j, k) by linear interpolation of ρ.
    #[inline]
    pub fn epsilon(&self, i: usize, j: usize, k: usize) -> f64 {
        let rho = self.rho[self.cell_idx(i, j, k)];
        self.eps_min + rho * (self.eps_max - self.eps_min)
    }

    /// Total number of design cells.
    pub fn n_cells(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Set all design variables from a flat slice (must have length `n_cells()`).
    pub fn set_rho(&mut self, rho: &[f64]) {
        assert_eq!(rho.len(), self.n_cells());
        self.rho.copy_from_slice(rho);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DesignVariable
// ─────────────────────────────────────────────────────────────────────────────

/// A single design variable mapping a physical parameter to a normalised
/// optimisation variable ρ ∈ \[0, 1\].
#[derive(Debug, Clone)]
pub struct DesignVariable {
    /// Physical identifier (e.g. "eps", "sigma", "width_nm")
    pub name: String,
    /// Normalised design variable ρ ∈ \[0, 1\]
    pub rho: f64,
    /// Lower bound of physical parameter
    pub p_min: f64,
    /// Upper bound of physical parameter
    pub p_max: f64,
    /// Gradient of FOM with respect to this variable (filled after adjoint run)
    pub gradient: f64,
}

impl DesignVariable {
    /// Construct a design variable.
    pub fn new(name: impl Into<String>, rho: f64, p_min: f64, p_max: f64) -> Self {
        Self {
            name: name.into(),
            rho: rho.clamp(0.0, 1.0),
            p_min,
            p_max,
            gradient: 0.0,
        }
    }

    /// Physical value: p = p_min + ρ·(p_max − p_min).
    pub fn physical_value(&self) -> f64 {
        self.p_min + self.rho * (self.p_max - self.p_min)
    }

    /// Update ρ by gradient step (clipped to \[0, 1\]).
    pub fn step_gradient_ascent(&mut self, step_size: f64) {
        self.rho = (self.rho + step_size * self.gradient).clamp(0.0, 1.0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FdtdSourceConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Source and monitor configuration for FDTD-coupled adjoint runs.
#[derive(Debug, Clone, Default)]
pub struct FdtdSourceConfig {
    /// Source injection x-index in design-region coordinates.
    pub source_i: usize,
    /// Source injection y-index in design-region coordinates.
    pub source_j: usize,
    /// Source injection z-index in design-region coordinates.
    pub source_k: usize,
    /// Monitor cell positions in design-region coordinates `(i, j, k)`.
    pub monitor_cells: Vec<(usize, usize, usize)>,
}

// ─────────────────────────────────────────────────────────────────────────────
// VectorField3d
// ─────────────────────────────────────────────────────────────────────────────

/// Full 3-component complex electric field on a 3-D grid.
///
/// Each component (`ex`, `ey`, `ez`) is a flat `Vec<Complex64>` of length
/// `nx * ny * nz` with index `i + j * nx + k * nx * ny`.
#[derive(Debug, Clone)]
pub struct VectorField3d {
    /// Ex component, length = nx * ny * nz
    pub ex: Vec<Complex64>,
    /// Ey component, length = nx * ny * nz
    pub ey: Vec<Complex64>,
    /// Ez component, length = nx * ny * nz
    pub ez: Vec<Complex64>,
    /// Grid extent x
    pub nx: usize,
    /// Grid extent y
    pub ny: usize,
    /// Grid extent z
    pub nz: usize,
}

impl VectorField3d {
    /// Allocate a zero-filled vector field.
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        let n = nx * ny * nz;
        Self {
            ex: vec![Complex64::ZERO; n],
            ey: vec![Complex64::ZERO; n],
            ez: vec![Complex64::ZERO; n],
            nx,
            ny,
            nz,
        }
    }

    /// Return the three field components at cell `(i, j, k)`.
    pub fn at(&self, i: usize, j: usize, k: usize) -> [Complex64; 3] {
        let idx = self.cell_idx(i, j, k);
        [self.ex[idx], self.ey[idx], self.ez[idx]]
    }

    /// Flat cell index: `i + j * nx + k * nx * ny`.
    #[inline]
    pub fn cell_idx(&self, i: usize, j: usize, k: usize) -> usize {
        i + j * self.nx + k * self.nx * self.ny
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PortPlane / VectorSourcePattern
// ─────────────────────────────────────────────────────────────────────────────

/// Port plane orientation for mode-source injection.
#[derive(Debug, Clone)]
pub enum PortPlane {
    XLow,
    XHigh,
    YLow,
    YHigh,
    ZLow,
    ZHigh,
}

/// Source pattern for vector (3-component) forward simulations.
#[derive(Debug, Clone)]
pub enum VectorSourcePattern {
    /// Hard point injection at a single cell with a fixed vector amplitude.
    PointSource {
        /// x-index (design-region coordinates)
        i: usize,
        /// y-index
        j: usize,
        /// z-index
        k: usize,
        /// Complex amplitude for \[Ex, Ey, Ez\].
        amplitude: [Complex64; 3],
    },
    /// Mode source: inject the full transverse mode profile at a port plane.
    ModeSource {
        /// Which face of the design region to inject at.
        port_plane: PortPlane,
        /// Port index (for multi-port devices).
        port_index: usize,
        /// Pre-computed complex mode pattern (defined on the 3-D grid).
        mode_pattern: VectorField3d,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// AdjointSolver3d
// ─────────────────────────────────────────────────────────────────────────────

/// 3D adjoint sensitivity solver.
///
/// Provides both the Phase 9 Ez-only API (`run_forward`, `run_adjoint`) and the
/// Phase 10 full vector-field API (`run_forward_vector`, `run_adjoint_vector`,
/// `compute_gradient_vector`).
pub struct AdjointSolver3d {
    /// Grid extents
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Cell size (uniform, m)
    pub dx: f64,
    /// Angular frequency (rad/s)
    pub omega: f64,
    /// Design variable list (parameter per cell in design region)
    pub variables: Vec<DesignVariable>,
    /// Forward field (Ez component, complex \[re, im\] per cell) — Phase 9 storage
    pub e_fwd: Vec<[f64; 2]>,
    /// Adjoint field (Ez component, complex \[re, im\] per cell) — Phase 9 storage
    pub e_adj: Vec<[f64; 2]>,
    /// Computed gradient ∂FOM/∂ρ_i for each design variable
    pub gradient: Vec<f64>,
    /// Iteration history: (iteration, FOM)
    pub history: Vec<(usize, f64)>,
    /// Current FOM value
    pub fom: f64,
    /// Iteration counter
    pub iteration: usize,
    /// If true, `run_forward` / `run_adjoint` use real `Fdtd3d`; otherwise analytic.
    pub use_fdtd: bool,
    /// Monitor cell positions in design-region coordinates `(i, j, k)`.
    pub monitor_cells: Vec<(usize, usize, usize)>,
    /// Source injection x-index (design-region coordinates)
    pub source_i: usize,
    /// Source injection y-index (design-region coordinates)
    pub source_j: usize,
    /// Source injection z-index (design-region coordinates)
    pub source_k: usize,
    /// Number of FDTD time steps per forward/adjoint run (default: 800)
    pub n_steps: usize,
}

impl AdjointSolver3d {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Create a new 3D adjoint solver (analytic mode by default).
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, omega: f64) -> Self {
        let n = nx * ny * nz;
        Self {
            nx,
            ny,
            nz,
            dx,
            omega,
            variables: Vec::new(),
            e_fwd: vec![[0.0, 0.0]; n],
            e_adj: vec![[0.0, 0.0]; n],
            gradient: Vec::new(),
            history: Vec::new(),
            fom: 0.0,
            iteration: 0,
            use_fdtd: false,
            monitor_cells: Vec::new(),
            source_i: 0,
            source_j: 0,
            source_k: 0,
            n_steps: 800,
        }
    }

    /// Create a 3D adjoint solver wired to real `Fdtd3d` simulations.
    pub fn new_with_fdtd(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        omega: f64,
        cfg: FdtdSourceConfig,
    ) -> Self {
        let mut s = Self::new(nx, ny, nz, dx, omega);
        s.use_fdtd = true;
        s.source_i = cfg.source_i;
        s.source_j = cfg.source_j;
        s.source_k = cfg.source_k;
        s.monitor_cells = cfg.monitor_cells;
        s
    }

    /// Convenience constructor for vector-field tests.
    ///
    /// Derives ω from λ = 1550 nm and sets the source position directly.
    /// Sets `use_fdtd = true` so `run_forward_vector` etc. drive real FDTD.
    pub fn new_fdtd(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        source_i: usize,
        source_j: usize,
        source_k: usize,
    ) -> Self {
        use std::f64::consts::PI;
        let c = 2.998e8_f64;
        let lambda = 1550e-9_f64;
        let omega = 2.0 * PI * c / lambda;
        let cfg = FdtdSourceConfig {
            source_i,
            source_j,
            source_k,
            monitor_cells: Vec::new(),
        };
        Self::new_with_fdtd(nx, ny, nz, dx, omega, cfg)
    }

    /// SOI-waveguide optimisation problem (220 nm height, Si/SiO₂).
    pub fn soi(nx: usize, ny: usize, nz: usize, resolution_nm: f64) -> Self {
        use std::f64::consts::PI;
        let c = 2.998e8_f64;
        let lambda = 1550e-9_f64;
        let omega = 2.0 * PI * c / lambda;
        Self::new(nx, ny, nz, resolution_nm * 1e-9, omega)
    }

    // ── Design variable management ────────────────────────────────────────────

    /// Add a design variable for the cell at (i, j, k).
    pub fn add_variable(&mut self, i: usize, j: usize, k: usize, eps_min: f64, eps_max: f64) {
        let name = format!("eps_{i}_{j}_{k}");
        self.variables
            .push(DesignVariable::new(name, 0.5, eps_min, eps_max));
        self.gradient.push(0.0);
    }

    /// Fill design variables for a rectangular region with ρ = 0.5.
    #[allow(clippy::too_many_arguments)]
    pub fn fill_design_region(
        &mut self,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        k0: usize,
        k1: usize,
        eps_min: f64,
        eps_max: f64,
    ) {
        for k in k0..k1 {
            for j in j0..j1 {
                for i in i0..i1 {
                    self.add_variable(i, j, k, eps_min, eps_max);
                }
            }
        }
    }

    /// Number of design variables.
    pub fn n_variables(&self) -> usize {
        self.variables.len()
    }

    /// L2 norm of the current gradient vector.
    pub fn gradient_norm(&self) -> f64 {
        self.gradient.iter().map(|g| g * g).sum::<f64>().sqrt()
    }

    /// FOM improvement ratio: latest / initial (from history).
    pub fn fom_improvement(&self) -> f64 {
        if self.history.len() < 2 {
            return 1.0;
        }
        let (_, f0) = self.history[0];
        let (_, f1) = *self.history.last().expect("history non-empty");
        if f0 == 0.0 {
            1.0
        } else {
            f1 / f0
        }
    }

    // ── Analytic field estimates (backward-compatible no-arg wrappers) ─────────

    /// Analytic Gaussian × plane-wave estimate of the forward Ez field.
    pub fn compute_forward_field_analytic(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let xc = nx as f64 / 2.0;
        let yc = ny as f64 / 2.0;
        let wx = (nx as f64 / 6.0).max(1.0);
        let wy = (ny as f64 / 6.0).max(1.0);

        for k in 0..nz {
            let phase_k = self.omega * k as f64 * self.dx / 2.998e8;
            let (sin_k, cos_k) = phase_k.sin_cos();
            for j in 0..ny {
                let yy = (j as f64 - yc) / wy;
                for i in 0..nx {
                    let xx = (i as f64 - xc) / wx;
                    let env = (-0.5 * (xx * xx + yy * yy)).exp();
                    let idx = k * (nx * ny) + j * nx + i;
                    self.e_fwd[idx] = [env * cos_k, env * sin_k];
                }
            }
        }
        let k_out = nz.saturating_sub(1);
        self.fom = (0..nx * ny)
            .map(|ij| {
                let idx = k_out * nx * ny + ij;
                let [re, im] = self.e_fwd[idx];
                re * re + im * im
            })
            .sum::<f64>()
            * self.dx
            * self.dx;
    }

    /// Analytic reversed-Gaussian estimate of the adjoint Ez field.
    pub fn compute_adjoint_field_analytic(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let xc = nx as f64 / 2.0;
        let yc = ny as f64 / 2.0;
        let wx = (nx as f64 / 6.0).max(1.0);
        let wy = (ny as f64 / 6.0).max(1.0);

        for k in 0..nz {
            let phase_k = self.omega * (nz - 1 - k) as f64 * self.dx / 2.998e8;
            let (sin_k, cos_k) = phase_k.sin_cos();
            for j in 0..ny {
                let yy = (j as f64 - yc) / wy;
                for i in 0..nx {
                    let xx = (i as f64 - xc) / wx;
                    let env = (-0.5 * (xx * xx + yy * yy)).exp();
                    let idx = k * (nx * ny) + j * nx + i;
                    self.e_adj[idx] = [env * cos_k, -env * sin_k];
                }
            }
        }
    }

    /// Compute the forward Ez field (analytic, backward-compatible no-arg wrapper).
    pub fn compute_forward_field(&mut self) {
        self.compute_forward_field_analytic();
    }

    /// Compute the adjoint Ez field (analytic, backward-compatible no-arg wrapper).
    pub fn compute_adjoint_field(&mut self) {
        self.compute_adjoint_field_analytic();
    }

    // ── Phase 9 Ez-only FDTD API ──────────────────────────────────────────────

    /// Build the FDTD grid with permittivity from `region`, returning the grid
    /// and the pre-computed guard/PML offsets. Internal helper shared by
    /// `run_forward`, `run_adjoint`, `run_forward_vector`, and `run_adjoint_vector`.
    fn build_fdtd_sim(region: &DesignRegion3d) -> Result<FdtdSimResult, OxiPhotonError> {
        use crate::fdtd::config::BoundaryConfig;
        use crate::fdtd::dims::fdtd_3d::Fdtd3d;

        let rnx = region.nx;
        let rny = region.ny;
        let rnz = region.nz;
        let dx = region.dx;

        let guard: usize = 3;
        let pml = 8_usize.min(rnx / 2).min(rny / 2).min(rnz / 2).max(1);

        let total_nx = rnx + 2 * guard + 2 * pml;
        let total_ny = rny + 2 * guard + 2 * pml;
        let total_nz = rnz + 2 * guard + 2 * pml;

        let bc = BoundaryConfig::pml(pml);
        let mut sim = Fdtd3d::new(total_nx, total_ny, total_nz, dx, dx, dx, &bc);

        // Stamp permittivity
        let off_x = guard + pml;
        let off_y = guard + pml;
        let off_z = guard + pml;
        for rk in 0..rnz {
            for rj in 0..rny {
                for ri in 0..rnx {
                    let gi = ri + off_x;
                    let gj = rj + off_y;
                    let gk = rk + off_z;
                    let eps = region.epsilon(ri, rj, rk);
                    let cell = sim.idx(gi, gj, gk);
                    sim.eps_r[cell] = eps;
                }
            }
        }

        Ok((sim, total_nx, total_ny, total_nz, off_x, off_y, off_z))
    }

    /// Validate fields are finite; return first non-finite position in error message.
    fn check_finite(re: &[f64], im: &[f64], label: &str) -> Result<(), OxiPhotonError> {
        for (&r, &i) in re.iter().zip(im.iter()) {
            if !r.is_finite() || !i.is_finite() {
                return Err(OxiPhotonError::NumericalError(format!(
                    "{label}: non-finite field value ({r:.3e}, {i:.3e}i)"
                )));
            }
        }
        Ok(())
    }

    /// Run the 3D forward FDTD simulation and return complex Ez over the design region.
    ///
    /// # Errors
    /// Returns `OxiPhotonError::InvalidWavelength` for non-positive / non-finite wavelength.
    /// Returns `OxiPhotonError::NumericalError` if the simulation yields non-finite fields.
    pub fn run_forward(
        &self,
        region: &DesignRegion3d,
        wavelength_m: f64,
    ) -> Result<Vec<Complex64>, OxiPhotonError> {
        use std::f64::consts::PI;

        if wavelength_m <= 0.0 || !wavelength_m.is_finite() {
            return Err(OxiPhotonError::InvalidWavelength(wavelength_m));
        }

        let (mut sim, _total_nx, _total_ny, _total_nz, off_x, off_y, off_z) =
            Self::build_fdtd_sim(region)?;
        let dt = sim.dt;

        let c = 2.998e8_f64;
        let f0 = c / wavelength_m;
        let sigma = 4.0 / f0;
        let t0 = 4.0 * sigma;
        let omega0 = 2.0 * PI * f0;

        let src_gi = (self.source_i + off_x).min(sim.nx - 1);
        let src_gj = (self.source_j + off_y).min(sim.ny - 1);
        let src_gk = (self.source_k + off_z).min(sim.nz - 1);

        let rnx = region.nx;
        let rny = region.ny;
        let rnz = region.nz;
        let n_cells = region.n_cells();
        let mut ez_re = vec![0.0_f64; n_cells];
        let mut ez_im = vec![0.0_f64; n_cells];

        for step in 0..self.n_steps {
            let t = step as f64 * dt;
            let env = (-(t - t0).powi(2) / (2.0 * sigma * sigma)).exp();
            let src_val = (omega0 * t).sin() * env;
            sim.inject_ez(src_gi, src_gj, src_gk, src_val);
            sim.step();

            let t_now = sim.current_time();
            let phase_re = (omega0 * t_now).cos() * dt;
            let phase_im = -(omega0 * t_now).sin() * dt;

            for rk in 0..rnz {
                for rj in 0..rny {
                    for ri in 0..rnx {
                        let gi = ri + off_x;
                        let gj = rj + off_y;
                        let gk = rk + off_z;
                        let ez_val = sim.ez[sim.idx(gi, gj, gk)];
                        let cell = region.cell_idx(ri, rj, rk);
                        ez_re[cell] += ez_val * phase_re;
                        ez_im[cell] += ez_val * phase_im;
                    }
                }
            }
        }

        Self::check_finite(&ez_re, &ez_im, "3D FDTD forward simulation")?;

        Ok(ez_re
            .into_iter()
            .zip(ez_im)
            .map(|(re, im)| Complex64::new(re, im))
            .collect())
    }

    /// Run the 3D adjoint FDTD simulation with multi-point adjoint sources.
    ///
    /// # Errors
    /// Returns `OxiPhotonError::NumericalError` if `monitor_cells.len() != fom_dconj_e.len()`,
    /// or if the simulation produces non-finite fields.
    pub fn run_adjoint(
        &self,
        region: &DesignRegion3d,
        monitor_cells: &[(usize, usize, usize)],
        fom_dconj_e: &[Complex64],
        wavelength_m: f64,
    ) -> Result<Vec<Complex64>, OxiPhotonError> {
        use std::f64::consts::PI;

        if monitor_cells.len() != fom_dconj_e.len() {
            return Err(OxiPhotonError::NumericalError(format!(
                "run_adjoint 3d: monitor_cells.len()={} != fom_dconj_e.len()={}",
                monitor_cells.len(),
                fom_dconj_e.len()
            )));
        }

        if wavelength_m <= 0.0 || !wavelength_m.is_finite() {
            return Err(OxiPhotonError::InvalidWavelength(wavelength_m));
        }

        let (mut sim, total_nx, total_ny, total_nz, off_x, off_y, off_z) =
            Self::build_fdtd_sim(region)?;
        let dt = sim.dt;

        let c = 2.998e8_f64;
        let f0 = c / wavelength_m;
        let sigma = 4.0 / f0;
        let t0 = 4.0 * sigma;
        let omega0 = 2.0 * PI * f0;

        let monitor_grid: Vec<(usize, usize, usize)> = monitor_cells
            .iter()
            .map(|&(mi, mj, mk)| {
                let gi = (mi + off_x).min(total_nx - 1);
                let gj = (mj + off_y).min(total_ny - 1);
                let gk = (mk + off_z).min(total_nz - 1);
                (gi, gj, gk)
            })
            .collect();

        let rnx = region.nx;
        let rny = region.ny;
        let rnz = region.nz;
        let n_cells = region.n_cells();
        let mut ez_re = vec![0.0_f64; n_cells];
        let mut ez_im = vec![0.0_f64; n_cells];

        for step in 0..self.n_steps {
            let t = step as f64 * dt;
            let env = (-(t - t0).powi(2) / (2.0 * sigma * sigma)).exp();

            for (m, &(gi, gj, gk)) in monitor_grid.iter().enumerate() {
                let w = fom_dconj_e[m];
                let carrier = w.re * (omega0 * t).cos() - w.im * (omega0 * t).sin();
                let src_val = carrier * env;
                sim.inject_ez(gi, gj, gk, src_val);
            }
            sim.step();

            let t_now = sim.current_time();
            let phase_re = (omega0 * t_now).cos() * dt;
            let phase_im = -(omega0 * t_now).sin() * dt;

            for rk in 0..rnz {
                for rj in 0..rny {
                    for ri in 0..rnx {
                        let gi = ri + off_x;
                        let gj = rj + off_y;
                        let gk = rk + off_z;
                        let ez_val = sim.ez[sim.idx(gi, gj, gk)];
                        let cell = region.cell_idx(ri, rj, rk);
                        ez_re[cell] += ez_val * phase_re;
                        ez_im[cell] += ez_val * phase_im;
                    }
                }
            }
        }

        Self::check_finite(&ez_re, &ez_im, "3D FDTD adjoint simulation")?;

        Ok(ez_re
            .into_iter()
            .zip(ez_im)
            .map(|(re, im)| Complex64::new(re, im))
            .collect())
    }

    /// Compute the gradient ∂FOM/∂ρ_i for each design variable (Ez-only).
    pub fn compute_gradient(&mut self) {
        let eps0 = 8.854e-12_f64;
        let omega = self.omega;
        let dx3 = self.dx * self.dx * self.dx;
        let nx = self.nx;
        let ny = self.ny;

        for (var_idx, var) in self.variables.iter().enumerate() {
            let de = var.p_max - var.p_min;
            let cell_idx = var_idx;
            let idx = cell_idx.min(self.e_fwd.len().saturating_sub(1));

            let n_per_k = nx * ny;
            let k = idx / n_per_k;
            let ij = idx % n_per_k;
            let j = ij / nx;
            let i = ij % nx;
            let fwd_idx = k * n_per_k + j * nx + i;

            let [ef_re, ef_im] = if fwd_idx < self.e_fwd.len() {
                self.e_fwd[fwd_idx]
            } else {
                [0.0, 0.0]
            };
            let [ea_re, ea_im] = if fwd_idx < self.e_adj.len() {
                self.e_adj[fwd_idx]
            } else {
                [0.0, 0.0]
            };

            let overlap = ef_re * ea_re + ef_im * ea_im;
            self.gradient[var_idx] = -2.0 * omega * omega * eps0 * de * overlap * dx3;
        }

        for (var, &g) in self.variables.iter_mut().zip(self.gradient.iter()) {
            var.gradient = g;
        }
    }

    /// Perform one gradient-ascent step and record history.
    pub fn gradient_step(&mut self, step_size: f64) {
        self.compute_forward_field();
        self.compute_adjoint_field();
        self.compute_gradient();

        for var in &mut self.variables {
            var.step_gradient_ascent(step_size);
        }

        self.history.push((self.iteration, self.fom));
        self.iteration += 1;
    }

    // ── Phase 10: Full vector-field API ───────────────────────────────────────

    /// Run the 3D forward FDTD simulation with a vector source pattern.
    ///
    /// Injects Ex, Ey, Ez according to `source`, accumulates a manual DFT for
    /// each of the three components, and returns a `VectorField3d` of length
    /// `region.n_cells()`.
    ///
    /// # Errors
    /// Returns `OxiPhotonError::InvalidWavelength` for non-positive / non-finite wavelength.
    /// Returns `OxiPhotonError::NumericalError` if any field component becomes non-finite.
    pub fn run_forward_vector(
        &self,
        region: &DesignRegion3d,
        source: &VectorSourcePattern,
        wavelength: f64,
    ) -> Result<VectorField3d, OxiPhotonError> {
        use std::f64::consts::PI;

        if wavelength <= 0.0 || !wavelength.is_finite() {
            return Err(OxiPhotonError::InvalidWavelength(wavelength));
        }

        let (mut sim, total_nx, total_ny, total_nz, off_x, off_y, off_z) =
            Self::build_fdtd_sim(region)?;
        let dt = sim.dt;

        let c = 2.998e8_f64;
        let f0 = c / wavelength;
        let sigma = 4.0 / f0;
        let t0 = 4.0 * sigma;
        let omega0 = 2.0 * PI * f0;

        let rnx = region.nx;
        let rny = region.ny;
        let rnz = region.nz;
        let n_cells = region.n_cells();

        // DFT accumulators for each component
        let mut ex_re = vec![0.0_f64; n_cells];
        let mut ex_im = vec![0.0_f64; n_cells];
        let mut ey_re = vec![0.0_f64; n_cells];
        let mut ey_im = vec![0.0_f64; n_cells];
        let mut ez_re = vec![0.0_f64; n_cells];
        let mut ez_im = vec![0.0_f64; n_cells];

        // Resolve source injection parameters
        let (src_cells, amplitudes) =
            Self::resolve_source(source, off_x, off_y, off_z, total_nx, total_ny, total_nz);

        for step in 0..self.n_steps {
            let t = step as f64 * dt;
            let env = (-(t - t0).powi(2) / (2.0 * sigma * sigma)).exp();

            // Inject each source cell with its vector amplitude
            for (&(gi, gj, gk), &amp) in src_cells.iter().zip(amplitudes.iter()) {
                // Carrier: Re(amp · exp(iω₀t)) = amp.re·cos - amp.im·sin
                let cos_t = (omega0 * t).cos();
                let sin_t = (omega0 * t).sin();

                let vx = (amp[0].re * cos_t - amp[0].im * sin_t) * env;
                let vy = (amp[1].re * cos_t - amp[1].im * sin_t) * env;
                let vz = (amp[2].re * cos_t - amp[2].im * sin_t) * env;

                sim.inject_ex(gi, gj, gk, vx);
                sim.inject_ey(gi, gj, gk, vy);
                sim.inject_ez(gi, gj, gk, vz);
            }
            sim.step();

            let t_now = sim.current_time();
            let phase_re = (omega0 * t_now).cos() * dt;
            let phase_im = -(omega0 * t_now).sin() * dt;

            for rk in 0..rnz {
                for rj in 0..rny {
                    for ri in 0..rnx {
                        let gi = ri + off_x;
                        let gj = rj + off_y;
                        let gk = rk + off_z;
                        let cell_i = sim.idx(gi, gj, gk);
                        let cell = region.cell_idx(ri, rj, rk);

                        let ex_v = sim.ex[cell_i];
                        let ey_v = sim.ey[cell_i];
                        let ez_v = sim.ez[cell_i];

                        ex_re[cell] += ex_v * phase_re;
                        ex_im[cell] += ex_v * phase_im;
                        ey_re[cell] += ey_v * phase_re;
                        ey_im[cell] += ey_v * phase_im;
                        ez_re[cell] += ez_v * phase_re;
                        ez_im[cell] += ez_v * phase_im;
                    }
                }
            }
        }

        Self::check_finite(&ex_re, &ex_im, "run_forward_vector Ex")?;
        Self::check_finite(&ey_re, &ey_im, "run_forward_vector Ey")?;
        Self::check_finite(&ez_re, &ez_im, "run_forward_vector Ez")?;

        let make_vec = |re: Vec<f64>, im: Vec<f64>| -> Vec<Complex64> {
            re.into_iter()
                .zip(im)
                .map(|(r, i)| Complex64::new(r, i))
                .collect()
        };

        Ok(VectorField3d {
            ex: make_vec(ex_re, ex_im),
            ey: make_vec(ey_re, ey_im),
            ez: make_vec(ez_re, ez_im),
            nx: rnx,
            ny: rny,
            nz: rnz,
        })
    }

    /// Run the 3D adjoint FDTD simulation injecting all three field components
    /// at the monitor cells.
    ///
    /// For each monitor cell `monitor_cells[m]`, three complex adjoint weights are
    /// provided: `fom_dconj_ex[m]`, `fom_dconj_ey[m]`, `fom_dconj_ez[m]`.
    ///
    /// # Errors
    /// Returns `OxiPhotonError::NumericalError` if the weight vectors and
    /// `monitor_cells` have mismatched lengths, or if the simulation produces
    /// non-finite fields.
    pub fn run_adjoint_vector(
        &self,
        region: &DesignRegion3d,
        monitor_cells: &[(usize, usize, usize)],
        fom_dconj_ex: &[Complex64],
        fom_dconj_ey: &[Complex64],
        fom_dconj_ez: &[Complex64],
        wavelength: f64,
    ) -> Result<VectorField3d, OxiPhotonError> {
        use std::f64::consts::PI;

        let nm = monitor_cells.len();
        if fom_dconj_ex.len() != nm || fom_dconj_ey.len() != nm || fom_dconj_ez.len() != nm {
            return Err(OxiPhotonError::NumericalError(format!(
                "run_adjoint_vector: monitor_cells.len()={nm} but weight lengths are \
                 ({}, {}, {})",
                fom_dconj_ex.len(),
                fom_dconj_ey.len(),
                fom_dconj_ez.len()
            )));
        }

        if wavelength <= 0.0 || !wavelength.is_finite() {
            return Err(OxiPhotonError::InvalidWavelength(wavelength));
        }

        let (mut sim, total_nx, total_ny, total_nz, off_x, off_y, off_z) =
            Self::build_fdtd_sim(region)?;
        let dt = sim.dt;

        let c = 2.998e8_f64;
        let f0 = c / wavelength;
        let sigma = 4.0 / f0;
        let t0 = 4.0 * sigma;
        let omega0 = 2.0 * PI * f0;

        // Pre-compute FDTD grid coordinates for each monitor cell
        let monitor_grid: Vec<(usize, usize, usize)> = monitor_cells
            .iter()
            .map(|&(mi, mj, mk)| {
                let gi = (mi + off_x).min(total_nx - 1);
                let gj = (mj + off_y).min(total_ny - 1);
                let gk = (mk + off_z).min(total_nz - 1);
                (gi, gj, gk)
            })
            .collect();

        let rnx = region.nx;
        let rny = region.ny;
        let rnz = region.nz;
        let n_cells = region.n_cells();

        let mut ex_re = vec![0.0_f64; n_cells];
        let mut ex_im = vec![0.0_f64; n_cells];
        let mut ey_re = vec![0.0_f64; n_cells];
        let mut ey_im = vec![0.0_f64; n_cells];
        let mut ez_re = vec![0.0_f64; n_cells];
        let mut ez_im = vec![0.0_f64; n_cells];

        for step in 0..self.n_steps {
            let t = step as f64 * dt;
            let env = (-(t - t0).powi(2) / (2.0 * sigma * sigma)).exp();

            let cos_t = (omega0 * t).cos();
            let sin_t = (omega0 * t).sin();

            for (m, &(gi, gj, gk)) in monitor_grid.iter().enumerate() {
                let wx = fom_dconj_ex[m];
                let wy = fom_dconj_ey[m];
                let wz = fom_dconj_ez[m];

                let vx = (wx.re * cos_t - wx.im * sin_t) * env;
                let vy = (wy.re * cos_t - wy.im * sin_t) * env;
                let vz = (wz.re * cos_t - wz.im * sin_t) * env;

                sim.inject_ex(gi, gj, gk, vx);
                sim.inject_ey(gi, gj, gk, vy);
                sim.inject_ez(gi, gj, gk, vz);
            }
            sim.step();

            let t_now = sim.current_time();
            let phase_re = (omega0 * t_now).cos() * dt;
            let phase_im = -(omega0 * t_now).sin() * dt;

            for rk in 0..rnz {
                for rj in 0..rny {
                    for ri in 0..rnx {
                        let gi = ri + off_x;
                        let gj = rj + off_y;
                        let gk = rk + off_z;
                        let cell_i = sim.idx(gi, gj, gk);
                        let cell = region.cell_idx(ri, rj, rk);

                        let ex_v = sim.ex[cell_i];
                        let ey_v = sim.ey[cell_i];
                        let ez_v = sim.ez[cell_i];

                        ex_re[cell] += ex_v * phase_re;
                        ex_im[cell] += ex_v * phase_im;
                        ey_re[cell] += ey_v * phase_re;
                        ey_im[cell] += ey_v * phase_im;
                        ez_re[cell] += ez_v * phase_re;
                        ez_im[cell] += ez_v * phase_im;
                    }
                }
            }
        }

        Self::check_finite(&ex_re, &ex_im, "run_adjoint_vector Ex")?;
        Self::check_finite(&ey_re, &ey_im, "run_adjoint_vector Ey")?;
        Self::check_finite(&ez_re, &ez_im, "run_adjoint_vector Ez")?;

        let make_vec = |re: Vec<f64>, im: Vec<f64>| -> Vec<Complex64> {
            re.into_iter()
                .zip(im)
                .map(|(r, i)| Complex64::new(r, i))
                .collect()
        };

        Ok(VectorField3d {
            ex: make_vec(ex_re, ex_im),
            ey: make_vec(ey_re, ey_im),
            ez: make_vec(ez_re, ez_im),
            nx: rnx,
            ny: rny,
            nz: rnz,
        })
    }

    /// Compute the adjoint gradient for all three field components.
    ///
    /// ```text
    /// g[v] = 2 · Re[E_fwd · conj(E_adj)]_v · ω² · ε₀ · dx³ · (ε_max − ε_min)
    /// ```
    ///
    /// where the dot product sums over Ex, Ey, Ez.
    ///
    /// # Errors
    /// Returns `OxiPhotonError::NumericalError` if `e_fwd` and `e_adj` have
    /// different sizes.
    pub fn compute_gradient_vector(
        &self,
        e_fwd: &VectorField3d,
        e_adj: &VectorField3d,
        wavelength: f64,
    ) -> Result<Vec<f64>, OxiPhotonError> {
        use crate::units::conversion::{EPSILON_0, SPEED_OF_LIGHT};
        use std::f64::consts::PI;

        let n = e_fwd.ex.len();
        if e_adj.ex.len() != n {
            return Err(OxiPhotonError::NumericalError(format!(
                "compute_gradient_vector: e_fwd has {n} cells but e_adj has {}",
                e_adj.ex.len()
            )));
        }

        // Derive eps range from the design region's typical Si/SiO₂ (or pass via closure).
        // We use the solver's dx and a fixed eps range derived from n_cells — but callers
        // supply `e_fwd` which was computed over a `DesignRegion3d`.  To keep the API
        // self-contained we require the caller to have set up the solver with consistent
        // eps_min/eps_max; here we approximate them from a typical SOI range.
        // For the general case the caller should prefer `compute_gradient_vector_with_region`.
        let eps_max = 12.0_f64;
        let eps_min = 1.0_f64;
        let omega = 2.0 * PI * SPEED_OF_LIGHT / wavelength;
        let dx = self.dx;
        let dx3 = dx * dx * dx;

        let scale = 2.0 * omega.powi(2) * EPSILON_0 * dx3 * (eps_max - eps_min);
        let g = e_fwd
            .ex
            .iter()
            .zip(e_adj.ex.iter())
            .zip(e_fwd.ey.iter().zip(e_adj.ey.iter()))
            .zip(e_fwd.ez.iter().zip(e_adj.ez.iter()))
            .map(|(((fx, ax), (fy, ay)), (fz, az))| {
                let dot = fx * ax.conj() + fy * ay.conj() + fz * az.conj();
                dot.re * scale
            })
            .collect();
        Ok(g)
    }

    /// Variant of `compute_gradient_vector` that takes an explicit `DesignRegion3d`
    /// so the eps range is known precisely.
    pub fn compute_gradient_vector_with_region(
        &self,
        e_fwd: &VectorField3d,
        e_adj: &VectorField3d,
        region: &DesignRegion3d,
        wavelength: f64,
    ) -> Result<Vec<f64>, OxiPhotonError> {
        use crate::units::conversion::{EPSILON_0, SPEED_OF_LIGHT};
        use std::f64::consts::PI;

        let n = e_fwd.ex.len();
        if e_adj.ex.len() != n {
            return Err(OxiPhotonError::NumericalError(format!(
                "compute_gradient_vector_with_region: size mismatch ({n} vs {})",
                e_adj.ex.len()
            )));
        }

        let omega = 2.0 * PI * SPEED_OF_LIGHT / wavelength;
        let dx3 = self.dx * self.dx * self.dx;
        let de = region.eps_max - region.eps_min;

        let scale = 2.0 * omega.powi(2) * EPSILON_0 * dx3 * de;
        let g = e_fwd
            .ex
            .iter()
            .zip(e_adj.ex.iter())
            .zip(e_fwd.ey.iter().zip(e_adj.ey.iter()))
            .zip(e_fwd.ez.iter().zip(e_adj.ez.iter()))
            .map(|(((fx, ax), (fy, ay)), (fz, az))| {
                let dot = fx * ax.conj() + fy * ay.conj() + fz * az.conj();
                dot.re * scale
            })
            .collect();
        Ok(g)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Resolve a `VectorSourcePattern` into a list of `(gi, gj, gk)` FDTD cells
    /// and per-cell vector amplitudes `[amp_x, amp_y, amp_z]`.
    fn resolve_source(
        source: &VectorSourcePattern,
        off_x: usize,
        off_y: usize,
        off_z: usize,
        total_nx: usize,
        total_ny: usize,
        total_nz: usize,
    ) -> SourceCells {
        match source {
            VectorSourcePattern::PointSource { i, j, k, amplitude } => {
                let gi = (i + off_x).min(total_nx - 1);
                let gj = (j + off_y).min(total_ny - 1);
                let gk = (k + off_z).min(total_nz - 1);
                (vec![(gi, gj, gk)], vec![*amplitude])
            }
            VectorSourcePattern::ModeSource {
                port_plane,
                port_index,
                mode_pattern,
            } => {
                // Inject the entire mode pattern into the appropriate port slice.
                let nx = mode_pattern.nx;
                let ny = mode_pattern.ny;
                let nz = mode_pattern.nz;

                let mut cells = Vec::new();
                let mut amps = Vec::new();

                match port_plane {
                    PortPlane::ZLow => {
                        let k = *port_index;
                        for j in 0..ny {
                            for i in 0..nx {
                                let ci = mode_pattern.cell_idx(i, j, k.min(nz.saturating_sub(1)));
                                let gi = (i + off_x).min(total_nx - 1);
                                let gj = (j + off_y).min(total_ny - 1);
                                let gk = (k + off_z).min(total_nz - 1);
                                cells.push((gi, gj, gk));
                                amps.push([
                                    mode_pattern.ex[ci],
                                    mode_pattern.ey[ci],
                                    mode_pattern.ez[ci],
                                ]);
                            }
                        }
                    }
                    PortPlane::ZHigh => {
                        let k_clamped = (*port_index).min(nz.saturating_sub(1));
                        for j in 0..ny {
                            for i in 0..nx {
                                let ci = mode_pattern.cell_idx(i, j, k_clamped);
                                let gi = (i + off_x).min(total_nx - 1);
                                let gj = (j + off_y).min(total_ny - 1);
                                let gk = (k_clamped + off_z).min(total_nz - 1);
                                cells.push((gi, gj, gk));
                                amps.push([
                                    mode_pattern.ex[ci],
                                    mode_pattern.ey[ci],
                                    mode_pattern.ez[ci],
                                ]);
                            }
                        }
                    }
                    PortPlane::XLow => {
                        let i = *port_index;
                        for k in 0..nz {
                            for j in 0..ny {
                                let ci = mode_pattern.cell_idx(i.min(nx.saturating_sub(1)), j, k);
                                let gi = (i + off_x).min(total_nx - 1);
                                let gj = (j + off_y).min(total_ny - 1);
                                let gk = (k + off_z).min(total_nz - 1);
                                cells.push((gi, gj, gk));
                                amps.push([
                                    mode_pattern.ex[ci],
                                    mode_pattern.ey[ci],
                                    mode_pattern.ez[ci],
                                ]);
                            }
                        }
                    }
                    PortPlane::XHigh => {
                        let i_clamped = (*port_index).min(nx.saturating_sub(1));
                        for k in 0..nz {
                            for j in 0..ny {
                                let ci = mode_pattern.cell_idx(i_clamped, j, k);
                                let gi = (i_clamped + off_x).min(total_nx - 1);
                                let gj = (j + off_y).min(total_ny - 1);
                                let gk = (k + off_z).min(total_nz - 1);
                                cells.push((gi, gj, gk));
                                amps.push([
                                    mode_pattern.ex[ci],
                                    mode_pattern.ey[ci],
                                    mode_pattern.ez[ci],
                                ]);
                            }
                        }
                    }
                    PortPlane::YLow => {
                        let j = *port_index;
                        for k in 0..nz {
                            for i in 0..nx {
                                let ci = mode_pattern.cell_idx(i, j.min(ny.saturating_sub(1)), k);
                                let gi = (i + off_x).min(total_nx - 1);
                                let gj = (j + off_y).min(total_ny - 1);
                                let gk = (k + off_z).min(total_nz - 1);
                                cells.push((gi, gj, gk));
                                amps.push([
                                    mode_pattern.ex[ci],
                                    mode_pattern.ey[ci],
                                    mode_pattern.ez[ci],
                                ]);
                            }
                        }
                    }
                    PortPlane::YHigh => {
                        let j_clamped = (*port_index).min(ny.saturating_sub(1));
                        for k in 0..nz {
                            for i in 0..nx {
                                let ci = mode_pattern.cell_idx(i, j_clamped, k);
                                let gi = (i + off_x).min(total_nx - 1);
                                let gj = (j_clamped + off_y).min(total_ny - 1);
                                let gk = (k + off_z).min(total_nz - 1);
                                cells.push((gi, gj, gk));
                                amps.push([
                                    mode_pattern.ex[ci],
                                    mode_pattern.ey[ci],
                                    mode_pattern.ez[ci],
                                ]);
                            }
                        }
                    }
                }

                (cells, amps)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Inline unit tests (3-D solver)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn adjoint_solver3d_construction() {
        let c = 2.998e8;
        let omega = 2.0 * PI * c / 1550e-9;
        let solver = AdjointSolver3d::new(8, 8, 8, 20e-9, omega);
        assert_eq!(solver.nx, 8);
        assert_eq!(solver.e_fwd.len(), 8 * 8 * 8);
        assert_eq!(solver.e_adj.len(), 8 * 8 * 8);
    }

    #[test]
    fn adjoint_solver3d_soi_constructor() {
        let solver = AdjointSolver3d::soi(10, 6, 20, 20.0);
        assert_eq!(solver.nx, 10);
        assert_eq!(solver.ny, 6);
        assert_eq!(solver.nz, 20);
    }

    #[test]
    fn adjoint_solver3d_fill_design_region() {
        let mut solver = AdjointSolver3d::new(10, 6, 20, 20e-9, 1.2e15);
        solver.fill_design_region(3, 7, 2, 4, 5, 15, 2.09, 12.11);
        let expected = 4 * 2 * 10; // (7-3)*(4-2)*(15-5)
        assert_eq!(solver.n_variables(), expected);
        assert_eq!(solver.gradient.len(), expected);
    }

    #[test]
    fn adjoint_solver3d_forward_field_finite() {
        let mut solver = AdjointSolver3d::new(8, 6, 10, 20e-9, 1.2e15);
        solver.compute_forward_field();
        assert!(
            solver
                .e_fwd
                .iter()
                .all(|&[re, im]| re.is_finite() && im.is_finite()),
            "Forward field should be finite"
        );
        assert!(
            solver.fom >= 0.0,
            "FOM should be non-negative: {:.4e}",
            solver.fom
        );
    }

    #[test]
    fn adjoint_solver3d_adjoint_field_finite() {
        let mut solver = AdjointSolver3d::new(8, 6, 10, 20e-9, 1.2e15);
        solver.compute_forward_field();
        solver.compute_adjoint_field();
        assert!(
            solver
                .e_adj
                .iter()
                .all(|&[re, im]| re.is_finite() && im.is_finite()),
            "Adjoint field should be finite"
        );
    }

    #[test]
    fn adjoint_solver3d_gradient_step_updates_rho() {
        let mut solver = AdjointSolver3d::new(8, 6, 10, 20e-9, 1.2e15);
        solver.fill_design_region(3, 5, 2, 4, 3, 7, 2.09, 12.11);
        let rho_before: Vec<f64> = solver.variables.iter().map(|v| v.rho).collect();

        solver.gradient_step(1e-3);

        let any_changed = solver
            .variables
            .iter()
            .zip(rho_before.iter())
            .any(|(v, &r0)| (v.rho - r0).abs() > 1e-15);
        assert!(any_changed, "gradient_step should update at least one rho");
        assert_eq!(solver.history.len(), 1);
        assert!(solver.iteration == 1);
    }

    #[test]
    fn adjoint_solver3d_rho_stays_in_bounds() {
        let mut solver = AdjointSolver3d::new(8, 6, 10, 20e-9, 1.2e15);
        solver.fill_design_region(2, 6, 1, 5, 2, 8, 2.09, 12.11);

        for _ in 0..5 {
            solver.gradient_step(1.0);
        }

        for v in &solver.variables {
            assert!(
                v.rho >= 0.0 && v.rho <= 1.0,
                "rho = {:.4} out of [0, 1]",
                v.rho
            );
        }
    }

    #[test]
    fn design_variable_physical_value() {
        let mut var = DesignVariable::new("eps", 0.0, 2.09, 12.11);
        assert!((var.physical_value() - 2.09).abs() < 1e-10);
        var.rho = 1.0;
        assert!((var.physical_value() - 12.11).abs() < 1e-10);
        var.rho = 0.5;
        assert!((var.physical_value() - 7.1).abs() < 1e-10);
    }

    #[test]
    fn adjoint_gradient_norm_finite() {
        let mut solver = AdjointSolver3d::new(6, 4, 8, 20e-9, 1.2e15);
        solver.fill_design_region(1, 5, 1, 3, 2, 6, 2.09, 12.11);
        solver.compute_forward_field();
        solver.compute_adjoint_field();
        solver.compute_gradient();

        let norm = solver.gradient_norm();
        assert!(
            norm.is_finite(),
            "Gradient norm should be finite: {norm:.4e}"
        );
    }

    #[test]
    fn vector_field_3d_indexing() {
        let field = VectorField3d::new(3, 4, 5);
        assert_eq!(field.cell_idx(0, 0, 0), 0);
        assert_eq!(field.cell_idx(1, 0, 0), 1);
        assert_eq!(field.cell_idx(0, 1, 0), 3);
        assert_eq!(field.cell_idx(0, 0, 1), 12);
        let comp = field.at(1, 2, 3);
        assert_eq!(comp.len(), 3);
    }
}
