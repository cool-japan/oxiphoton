/// Adjoint sensitivity analysis for photonic inverse design.
///
/// The adjoint method computes the gradient ∂FOM/∂ρ_i (where ρ_i are design
/// parameters, typically local permittivity values) using only two simulations
/// regardless of the number of parameters:
///
///   Forward:  E_fwd driven by sources → fields in design region
///   Adjoint:  E_adj driven by adjoint source at monitor → adjoint fields
///
///   Gradient: dFOM/dε(r) = -Re\[E_fwd(r) · E_adj(r)\] · (2ωε₀/A_cell)
///
/// For amplitude maximization at a monitor:
///   Adjoint source = -i·ω·conj(E_fwd) at monitor location
///
/// This enables topology optimization, shape optimization, and parameter sweeps
/// for photonic device design.
///
/// Design region for topology optimization.
///
/// The design region is a rectangular grid of "pixels" with permittivity
/// values ε(r) interpolated between ε_min (void) and ε_max (material).
#[derive(Debug, Clone)]
pub struct DesignRegion {
    /// Number of pixels in x-direction
    pub nx: usize,
    /// Number of pixels in z-direction (or y in 2D)
    pub nz: usize,
    /// Pixel size (m)
    pub dx: f64,
    /// Minimum permittivity (void, e.g., SiO₂: 2.09)
    pub eps_min: f64,
    /// Maximum permittivity (material, e.g., Si: 12.11)
    pub eps_max: f64,
    /// Design variable ρ(i,j) ∈ \[0,1\] for each pixel
    pub rho: Vec<f64>,
}

impl DesignRegion {
    /// Create a design region filled with ρ = 0.5 (intermediate density).
    pub fn new(nx: usize, nz: usize, dx: f64, eps_min: f64, eps_max: f64) -> Self {
        Self {
            nx,
            nz,
            dx,
            eps_min,
            eps_max,
            rho: vec![0.5; nx * nz],
        }
    }

    /// SOI waveguide design region (220nm tall, Si/SiO₂).
    pub fn soi_design(nx: usize, nz: usize, resolution_nm: f64) -> Self {
        Self::new(nx, nz, resolution_nm * 1e-9, 2.09, 12.11)
    }

    /// Permittivity at pixel (i, j) from design variable (linear interpolation).
    pub fn epsilon(&self, i: usize, j: usize) -> f64 {
        let rho = self.rho[j * self.nx + i];
        self.eps_min + rho * (self.eps_max - self.eps_min)
    }

    /// Set design variables from a flat slice.
    pub fn set_rho(&mut self, rho: &[f64]) {
        assert_eq!(rho.len(), self.nx * self.nz);
        self.rho.copy_from_slice(rho);
    }

    /// Number of design parameters.
    pub fn n_params(&self) -> usize {
        self.nx * self.nz
    }

    /// Apply sigmoid projection to binarise the design.
    ///   ρ_proj = tanh(β·(ρ - η)) / tanh(β·(1-η))  (normalised)
    pub fn apply_sigmoid(&mut self, beta: f64, eta: f64) {
        for rho in &mut self.rho {
            let num = (beta * (*rho - eta)).tanh();
            let den = (beta * (1.0 - eta)).tanh();
            *rho = (num / den).clamp(-1.0, 1.0) * 0.5 + 0.5;
        }
    }

    /// Average permittivity in the design region.
    pub fn mean_epsilon(&self) -> f64 {
        let sum: f64 = self
            .rho
            .iter()
            .map(|&r| self.eps_min + r * (self.eps_max - self.eps_min))
            .sum();
        sum / self.n_params() as f64
    }

    /// Fill fraction (fraction of pixels with ρ > 0.5).
    pub fn fill_fraction(&self) -> f64 {
        let n_filled = self.rho.iter().filter(|&&r| r > 0.5).count();
        n_filled as f64 / self.n_params() as f64
    }
}

// 3-D types (DesignRegion3d, AdjointSolver3d, VectorField3d, etc.) live in adjoint_3d.rs.
// Re-exports are provided via src/inverse/mod.rs.

/// Gradient of the figure of merit (FOM) with respect to each design pixel.
#[derive(Debug, Clone)]
pub struct FomGradient {
    /// Gradient values dFOM/dρ_i (same shape as DesignRegion::rho)
    pub grad: Vec<f64>,
    /// FOM value at this evaluation
    pub fom: f64,
}

impl FomGradient {
    pub fn new(n: usize) -> Self {
        Self {
            grad: vec![0.0; n],
            fom: 0.0,
        }
    }

    /// L2 norm of gradient vector.
    pub fn grad_norm(&self) -> f64 {
        self.grad.iter().map(|g| g * g).sum::<f64>().sqrt()
    }

    /// Maximum absolute gradient component.
    pub fn max_grad(&self) -> f64 {
        self.grad
            .iter()
            .cloned()
            .fold(0.0_f64, |a, b| a.max(b.abs()))
    }
}

/// Adjoint solver for computing FOM gradients.
///
/// This is a lightweight model — in a full implementation, the forward and
/// adjoint simulations would call the FDTD engine. Here we provide the
/// mathematical framework and interface.
pub struct AdjointSolver {
    /// Angular frequency of operation (rad/s)
    pub omega: f64,
    /// Design region
    pub region: DesignRegion,
    /// Optimisation step size (learning rate)
    pub step_size: f64,
    /// Current FOM value
    pub fom: f64,
    /// Iteration counter
    pub iteration: usize,
}

impl AdjointSolver {
    pub fn new(omega: f64, region: DesignRegion) -> Self {
        Self {
            omega,
            region,
            step_size: 0.01,
            fom: 0.0,
            iteration: 0,
        }
    }

    /// Compute the adjoint gradient using the overlap integral.
    ///
    /// Given forward field Efwd and adjoint field Eadj at each pixel,
    /// the gradient is:
    ///
    ///   dFOM/dε_i = -Re\[E_fwd,i · E_adj,i\] × (ε_max - ε_min) / (ω²·μ₀)
    ///
    /// Returns gradient with respect to design variable ρ_i.
    pub fn compute_gradient(
        &self,
        e_fwd: &[[f64; 2]], // complex field [re, im] at each pixel
        e_adj: &[[f64; 2]],
    ) -> FomGradient {
        assert_eq!(e_fwd.len(), self.region.n_params());
        assert_eq!(e_adj.len(), self.region.n_params());

        let de_deps = self.region.eps_max - self.region.eps_min;
        let scale = -de_deps * self.omega;

        let grad: Vec<f64> = e_fwd
            .iter()
            .zip(e_adj.iter())
            .map(|(&ef, &ea)| {
                // Re[E_fwd · conj(E_adj)] = Re[E_fwd] × Re[E_adj] + Im[E_fwd] × Im[E_adj]
                let overlap_re = ef[0] * ea[0] + ef[1] * ea[1];
                scale * overlap_re
            })
            .collect();

        let fom = e_fwd.iter().map(|&[re, im]| re * re + im * im).sum::<f64>();

        FomGradient { grad, fom }
    }

    /// Update design variables using gradient ascent (maximise FOM).
    ///
    ///   ρ ← clip(ρ + α · ∇FOM, 0, 1)
    pub fn update_gradient_ascent(&mut self, gradient: &FomGradient) {
        for (rho, &g) in self.region.rho.iter_mut().zip(&gradient.grad) {
            *rho = (*rho + self.step_size * g).clamp(0.0, 1.0);
        }
        self.fom = gradient.fom;
        self.iteration += 1;
    }

    /// Adam optimizer update.
    ///
    /// Maintains first (m) and second (v) moment estimates:
    ///   m ← β₁·m + (1-β₁)·g
    ///   v ← β₂·v + (1-β₂)·g²
    ///   ρ ← ρ + α·m̂/(√v̂ + ε)
    #[allow(clippy::too_many_arguments)]
    pub fn update_adam(
        &mut self,
        gradient: &FomGradient,
        m: &mut [f64],
        v: &mut [f64],
        t: usize,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) {
        let alpha = self.step_size;
        let t_f64 = t as f64;
        for i in 0..self.region.n_params() {
            let g = gradient.grad[i];
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
            let m_hat = m[i] / (1.0 - beta1.powf(t_f64));
            let v_hat = v[i] / (1.0 - beta2.powf(t_f64));
            self.region.rho[i] =
                (self.region.rho[i] + alpha * m_hat / (v_hat.sqrt() + epsilon)).clamp(0.0, 1.0);
        }
        self.fom = gradient.fom;
        self.iteration += 1;
    }

    /// Check convergence: gradient norm < tolerance.
    pub fn is_converged(&self, gradient: &FomGradient, tolerance: f64) -> bool {
        gradient.grad_norm() < tolerance
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AdjointSolver2d — wraps Fdtd2dTm for 2D adjoint forward simulation
// ─────────────────────────────────────────────────────────────────────────────

/// FDTD-backed 2D TM adjoint solver.
///
/// Runs a real 2D TM FDTD simulation and returns the complex E_z field at a
/// given frequency, accumulated via a running DFT over the simulation duration.
/// Used by `AdjointOptimizer::compute_forward_field` when `use_fdtd_forward` is `true`.
pub struct AdjointSolver2d {
    /// Design-region width in pixels
    pub nx: usize,
    /// Design-region height in pixels
    pub nz: usize,
    /// Pixel size (m)
    pub dx: f64,
    /// Time step (s) — computed from dx via Courant condition when the FDTD grid is built
    pub dt: f64,
    /// Number of FDTD time steps to run per forward simulation
    pub n_steps: usize,
}

impl AdjointSolver2d {
    /// Guard-band thickness (cells) surrounding the design region.
    /// The source is placed at the boundary of the guard band, and PML cells
    /// absorb outgoing energy at the outer grid edge.
    const GUARD: usize = 5;
    /// PML thickness (cells)
    const PML: usize = 10;

    /// Construct a new adjoint solver for a design region of size `nx × nz`.
    ///
    /// `dx` is the cell size in metres (same for both axes — square grid).
    /// Default step count is 2000, which covers ≈ 8 light-crossings for a
    /// 200-cell grid at dx = 10 nm.
    pub fn new(nx: usize, nz: usize, dx: f64) -> Self {
        // Estimate dt from 2D Courant condition: dt = 0.99·dx/(c·√2)
        use crate::fdtd::config::{Dimensions, GridSpacing};
        use crate::fdtd::courant::courant_dt;
        let spacing = GridSpacing { dx, dy: dx, dz: dx };
        let total_nx = nx + 2 * Self::GUARD + 2 * Self::PML;
        let total_ny = nz + 2 * Self::GUARD + 2 * Self::PML;
        let dt = 0.99
            * courant_dt(
                Dimensions::TwoD {
                    nx: total_nx,
                    ny: total_ny,
                },
                spacing,
                1.0,
            );
        Self {
            nx,
            nz,
            dx,
            dt,
            n_steps: 2000,
        }
    }

    /// Override the number of time steps.
    pub fn set_fdtd_steps(&mut self, n: usize) {
        self.n_steps = n;
    }

    /// Run the adjoint FDTD simulation with multi-point adjoint sources.
    ///
    /// Each monitor cell `monitor_cells[k]` injects a Gaussian-modulated CW source
    /// weighted by `fom_dconj_e[k]`.  For complex weight `w = a + ib`:
    ///
    ///   src_val = Re(w · exp(iωt)) · gauss(t) = (a·cos(ωt) − b·sin(ωt)) · gauss(t)
    ///
    /// This handles complex adjoint weights correctly and reduces to `a·cos(ωt)·gauss(t)`
    /// for real weights.
    ///
    /// After running `n_steps` FDTD steps, the DFT-accumulated E_z field over the
    /// design region is returned as a flat `Vec<Complex64>` of length
    /// `region.nx * region.nz` (j-major: index = `j * nx + i`).
    ///
    /// # Errors
    /// Returns `OxiPhotonError::NumericalError` if `monitor_cells.len() !=
    /// fom_dconj_e.len()`, or if the simulation produces non-finite fields.
    pub fn run_adjoint(
        &self,
        region: &DesignRegion,
        monitor_cells: &[(usize, usize)],
        fom_dconj_e: &[num_complex::Complex64],
        wavelength_m: f64,
    ) -> Result<Vec<num_complex::Complex64>, crate::error::OxiPhotonError> {
        use crate::error::OxiPhotonError;
        use crate::fdtd::config::BoundaryConfig;
        use crate::fdtd::dims::fdtd_2d::Fdtd2dTm;
        use num_complex::Complex64;
        use std::f64::consts::PI;

        if monitor_cells.len() != fom_dconj_e.len() {
            return Err(OxiPhotonError::NumericalError(format!(
                "run_adjoint: monitor_cells.len()={} != fom_dconj_e.len()={}",
                monitor_cells.len(),
                fom_dconj_e.len()
            )));
        }

        if wavelength_m <= 0.0 || !wavelength_m.is_finite() {
            return Err(OxiPhotonError::InvalidWavelength(wavelength_m));
        }

        let guard = Self::GUARD;
        let pml = Self::PML;
        let dx = self.dx;

        let total_nx = region.nx + 2 * guard + 2 * pml;
        let total_ny = region.nz + 2 * guard + 2 * pml;

        let bc = BoundaryConfig::pml(pml);
        let mut sim = Fdtd2dTm::new(total_nx, total_ny, dx, dx, &bc);
        let dt = sim.dt;

        // Fill permittivity from the design region (same as run_forward)
        let offset_x = guard + pml;
        let offset_y = guard + pml;
        for rj in 0..region.nz {
            for ri in 0..region.nx {
                let gi = ri + offset_x;
                let gj = rj + offset_y;
                let eps = region.epsilon(ri, rj);
                let idx = gj * total_nx + gi;
                if idx < sim.eps_r.len() {
                    sim.eps_r[idx] = eps;
                }
            }
        }

        // Source parameters: Gaussian-modulated CW (same envelope as run_forward)
        let c = 2.998e8_f64;
        let f0 = c / wavelength_m;
        let sigma = 4.0 / f0;
        let t0 = 4.0 * sigma;
        let omega0 = 2.0 * PI * f0;

        // Pre-compute FDTD grid coordinates for each monitor cell
        let monitor_grid: Vec<(usize, usize)> = monitor_cells
            .iter()
            .map(|&(mi, mj)| {
                let gi = (mi + offset_x).min(total_nx - 1);
                let gj = (mj + offset_y).min(total_ny - 1);
                (gi, gj)
            })
            .collect();

        // Running DFT accumulators for E_z over the design region
        let n_cells = region.nx * region.nz;
        let mut ez_re = vec![0.0_f64; n_cells];
        let mut ez_im = vec![0.0_f64; n_cells];

        let n_steps = self.n_steps;

        for step in 0..n_steps {
            let t = step as f64 * dt;
            let env = (-(t - t0).powi(2) / (2.0 * sigma * sigma)).exp();

            // Inject adjoint source at each monitor cell with complex weight
            for (k, &(gi, gj)) in monitor_grid.iter().enumerate() {
                let w = fom_dconj_e[k];
                // Re(w · exp(iω₀t)) = w.re·cos(ω₀t) − w.im·sin(ω₀t)
                let carrier = w.re * (omega0 * t).cos() - w.im * (omega0 * t).sin();
                let src_val = carrier * env;
                sim.inject_ez(gi, gj, src_val);
            }
            sim.step();

            // Accumulate DFT over design region
            let t_now = sim.current_time();
            let phase_re = (omega0 * t_now).cos() * dt;
            let phase_im = -(omega0 * t_now).sin() * dt;
            for rj in 0..region.nz {
                for ri in 0..region.nx {
                    let gi = ri + offset_x;
                    let gj = rj + offset_y;
                    let ez_val = if gi < total_nx && gj < total_ny {
                        sim.ez[gj * total_nx + gi]
                    } else {
                        0.0
                    };
                    let cell = rj * region.nx + ri;
                    ez_re[cell] += ez_val * phase_re;
                    ez_im[cell] += ez_val * phase_im;
                }
            }
        }

        // Check for NaN/Inf
        for (&re, &im) in ez_re.iter().zip(ez_im.iter()) {
            if !re.is_finite() || !im.is_finite() {
                return Err(OxiPhotonError::NumericalError(
                    "FDTD adjoint simulation produced non-finite fields".to_string(),
                ));
            }
        }

        let result: Vec<Complex64> = ez_re
            .into_iter()
            .zip(ez_im)
            .map(|(re, im)| Complex64::new(re, im))
            .collect();

        Ok(result)
    }

    /// Run the forward FDTD simulation and return the complex E_z field at `wavelength_m`
    /// over the design region.
    ///
    /// # Arguments
    /// * `region`       – design region (supplies permittivity at each pixel)
    /// * `source_i`     – design-region x-index of the injection point
    /// * `source_j`     – design-region z/y-index of the injection point
    /// * `wavelength_m` – free-space wavelength (m)
    ///
    /// # Returns
    /// Flat `Vec<Complex64>` of length `region.nx * region.nz` (row-major, j-major).
    pub fn run_forward(
        &self,
        region: &DesignRegion,
        source_i: usize,
        source_j: usize,
        wavelength_m: f64,
    ) -> Result<Vec<num_complex::Complex64>, crate::error::OxiPhotonError> {
        use crate::error::OxiPhotonError;
        use crate::fdtd::config::BoundaryConfig;
        use crate::fdtd::dims::fdtd_2d::{DftBox2dTm, Fdtd2dTm};
        use num_complex::Complex64;
        use std::f64::consts::PI;

        if wavelength_m <= 0.0 || !wavelength_m.is_finite() {
            return Err(OxiPhotonError::InvalidWavelength(wavelength_m));
        }

        let guard = Self::GUARD;
        let pml = Self::PML;
        let dx = self.dx;

        // Total FDTD grid dimensions (design region + guard bands + PML)
        let total_nx = region.nx + 2 * guard + 2 * pml;
        let total_ny = region.nz + 2 * guard + 2 * pml;

        let bc = BoundaryConfig::pml(pml);
        let mut sim = Fdtd2dTm::new(total_nx, total_ny, dx, dx, &bc);
        let dt = sim.dt;

        // Fill permittivity from the design region (offset by guard + pml)
        let offset_x = guard + pml;
        let offset_y = guard + pml;
        for rj in 0..region.nz {
            for ri in 0..region.nx {
                let gi = ri + offset_x;
                let gj = rj + offset_y;
                let eps = region.epsilon(ri, rj);
                let idx = gj * total_nx + gi;
                if idx < sim.eps_r.len() {
                    sim.eps_r[idx] = eps;
                }
            }
        }

        // Source parameters: Gaussian-modulated CW (quasi-monochromatic)
        let c = 2.998e8_f64;
        let f0 = c / wavelength_m;
        let sigma = 4.0 / f0; // temporal Gaussian width (4 cycles)
        let t0 = 4.0 * sigma; // centre of Gaussian envelope

        // DFT monitor over the design region cells
        let dft = DftBox2dTm::new(&[f0], region.nx, region.nz, dt);
        // We accumulate manually to handle the region offset

        // Running DFT accumulators for the design region E_z only
        let n_cells = region.nx * region.nz;
        let mut ez_re = vec![0.0_f64; n_cells];
        let mut ez_im = vec![0.0_f64; n_cells];

        // Source cell in FDTD grid coordinates
        let src_gi = (source_i + offset_x).min(total_nx - 1);
        let src_gj = (source_j + offset_y).min(total_ny - 1);

        let n_steps = self.n_steps;
        let omega0 = 2.0 * PI * f0;

        // Drop the unused dft binding; we accumulate manually
        let _ = dft;

        for step in 0..n_steps {
            let t = step as f64 * dt;
            // Gaussian-modulated sinusoidal source
            let env = (-(t - t0).powi(2) / (2.0 * sigma * sigma)).exp();
            let src_val = (omega0 * t).sin() * env;
            sim.inject_ez(src_gi, src_gj, src_val);
            sim.step();

            // Accumulate DFT over design region
            let t_now = sim.current_time();
            let phase_re = (omega0 * t_now).cos() * dt;
            let phase_im = -(omega0 * t_now).sin() * dt;
            for rj in 0..region.nz {
                for ri in 0..region.nx {
                    let gi = ri + offset_x;
                    let gj = rj + offset_y;
                    let ez_val = if gi < total_nx && gj < total_ny {
                        sim.ez[gj * total_nx + gi]
                    } else {
                        0.0
                    };
                    let cell = rj * region.nx + ri;
                    ez_re[cell] += ez_val * phase_re;
                    ez_im[cell] += ez_val * phase_im;
                }
            }
        }

        // Check for NaN/Inf in result
        for (&re, &im) in ez_re.iter().zip(ez_im.iter()) {
            if !re.is_finite() || !im.is_finite() {
                return Err(OxiPhotonError::NumericalError(
                    "FDTD forward simulation produced non-finite fields".to_string(),
                ));
            }
        }

        let result: Vec<Complex64> = ez_re
            .into_iter()
            .zip(ez_im)
            .map(|(re, im)| Complex64::new(re, im))
            .collect();

        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AdjointOptimizer — 2D TM adjoint optimizer wiring FDTD forward field
// ─────────────────────────────────────────────────────────────────────────────

/// 2D adjoint optimizer that uses a real FDTD simulation for the forward field.
///
/// This struct coordinates:
///   1. Forward simulation: `compute_forward_field` runs `Fdtd2dTm` and
///      extracts the complex E_z via a running DFT.
///   2. Adjoint simulation: `compute_adjoint_field` uses the same
///      FDTD approach driven by the adjoint source (conj(E_fwd) at monitor).
///   3. Gradient assembly: `compute_gradient` evaluates
///      dFOM/dρ = -Re[E_fwd · conj(E_adj)] × (ε_max - ε_min).
///
/// When `use_fdtd_forward = false`, the forward field falls back to a fast
/// analytic Gaussian × plane-wave estimate (`compute_forward_field_analytic`).
pub struct AdjointOptimizer {
    /// Operating wavelength (m)
    pub wavelength: f64,
    /// Design region
    pub region: DesignRegion,
    /// If true, use FDTD for the forward field; otherwise use analytic estimate
    pub use_fdtd_forward: bool,
    /// 2D FDTD adjoint solver
    pub fdtd_solver: AdjointSolver2d,
    /// Source injection x-index within the design region
    pub source_i: usize,
    /// Source injection z/y-index within the design region
    pub source_j: usize,
    /// Monitor cell positions (design-region coordinates) for the adjoint sources.
    ///
    /// Each `(i, j)` is a design-region pixel index; the adjoint field is driven
    /// by weighted sources at these locations.
    pub monitor_cells: Vec<(usize, usize)>,
    /// Optimisation step size (learning rate)
    pub step_size: f64,
    /// Iteration counter
    pub iteration: usize,
}

impl AdjointOptimizer {
    /// Create an `AdjointOptimizer` for the given design region and wavelength.
    ///
    /// By default the FDTD forward field is enabled (`use_fdtd_forward = true`).
    /// The source is placed at the centre of the design region's left edge.
    /// `monitor_cells` is the list of design-region pixel coordinates `(i, j)`
    /// used as adjoint source locations; pass `vec![]` for an empty initial list.
    pub fn new(region: DesignRegion, wavelength: f64, monitor_cells: Vec<(usize, usize)>) -> Self {
        let fdtd_solver = AdjointSolver2d::new(region.nx, region.nz, region.dx);
        let source_i = 0;
        let source_j = region.nz / 2;
        Self {
            wavelength,
            region,
            use_fdtd_forward: true,
            fdtd_solver,
            source_i,
            source_j,
            monitor_cells,
            step_size: 0.01,
            iteration: 0,
        }
    }

    /// Compute the forward E_z field over the design region.
    ///
    /// Dispatches to the real FDTD simulation when `use_fdtd_forward` is `true`,
    /// otherwise falls back to the fast analytic estimate.
    pub fn compute_forward_field(
        &self,
    ) -> Result<Vec<num_complex::Complex64>, crate::error::OxiPhotonError> {
        if self.use_fdtd_forward {
            self.fdtd_solver.run_forward(
                &self.region,
                self.source_i,
                self.source_j,
                self.wavelength,
            )
        } else {
            self.compute_forward_field_analytic()
        }
    }

    /// Analytic Gaussian × plane-wave estimate of the forward E_z field.
    ///
    /// Provides a quick, approximate forward field for testing without FDTD.
    /// The field is a Gaussian mode modulated by a plane wave propagating in
    /// the +z direction (i index), peaked at the source location.
    pub fn compute_forward_field_analytic(
        &self,
    ) -> Result<Vec<num_complex::Complex64>, crate::error::OxiPhotonError> {
        use num_complex::Complex64;
        use std::f64::consts::PI;

        let nx = self.region.nx;
        let nz = self.region.nz;
        let c = 2.998e8_f64;
        let f0 = c / self.wavelength;
        let omega = 2.0 * PI * f0;

        // Gaussian transverse profile centred on source_j
        let xc = self.source_i as f64;
        let zc = self.source_j as f64;
        let wx = (nx as f64 / 6.0).max(1.0);
        let wz = (nz as f64 / 6.0).max(1.0);

        let mut result = Vec::with_capacity(nx * nz);
        for rj in 0..nz {
            // phase increases along propagation (i) direction
            for ri in 0..nx {
                let xx = (ri as f64 - xc) / wx;
                let zz = (rj as f64 - zc) / wz;
                let env = (-0.5 * (xx * xx + zz * zz)).exp();
                let phase = omega * ri as f64 * self.region.dx / c;
                result.push(Complex64::new(env * phase.cos(), env * phase.sin()));
            }
        }
        Ok(result)
    }

    /// Compute the adjoint E_z field over the design region.
    ///
    /// Dispatches to the real FDTD-backed adjoint simulation when
    /// `use_fdtd_forward` is `true`, otherwise falls back to the fast analytic
    /// Gaussian-weighted Green's-function approximation.
    ///
    /// # Arguments
    /// * `region`       – design region (supplies permittivity for the FDTD run)
    /// * `fom_dconj_e`  – ∂FoM/∂E_z* at each monitor cell (complex adjoint weights)
    ///
    /// `fom_dconj_e.len()` must equal `self.monitor_cells.len()`.
    pub fn compute_adjoint_field(
        &self,
        region: &DesignRegion,
        fom_dconj_e: &[num_complex::Complex64],
    ) -> Result<Vec<num_complex::Complex64>, crate::error::OxiPhotonError> {
        if self.use_fdtd_forward {
            self.fdtd_solver
                .run_adjoint(region, &self.monitor_cells, fom_dconj_e, self.wavelength)
        } else {
            self.compute_adjoint_field_analytic(region, fom_dconj_e)
        }
    }

    /// Analytic approximation of the adjoint E_z field.
    ///
    /// For a uniform-medium approximation, each monitor cell `m` at `(mi_m, mj_m)`
    /// contributes a Gaussian-decaying field weighted by `fom_dconj_e[m]`:
    ///
    ///   `e_adj[j * nx + i]` = Σ_m `fom_dconj_e[m]` · exp(−((i−mi_m)² + (j−mj_m)²) / σ²)
    ///
    /// where σ = `region.nx.min(region.nz) as f64 * 0.3`.
    ///
    /// Used only when `use_fdtd_forward = false` (unit tests and fast validation).
    pub fn compute_adjoint_field_analytic(
        &self,
        region: &DesignRegion,
        fom_dconj_e: &[num_complex::Complex64],
    ) -> Result<Vec<num_complex::Complex64>, crate::error::OxiPhotonError> {
        use crate::error::OxiPhotonError;
        use num_complex::Complex64;

        if self.monitor_cells.len() != fom_dconj_e.len() {
            return Err(OxiPhotonError::NumericalError(format!(
                "compute_adjoint_field_analytic: monitor_cells.len()={} != fom_dconj_e.len()={}",
                self.monitor_cells.len(),
                fom_dconj_e.len()
            )));
        }

        let nx = region.nx;
        let nz = region.nz;
        let sigma = region.nx.min(region.nz) as f64 * 0.3;
        let sigma_sq = (sigma * sigma).max(1.0); // guard against zero

        let mut result = vec![Complex64::new(0.0, 0.0); nx * nz];

        for rj in 0..nz {
            for ri in 0..nx {
                let cell = rj * nx + ri;
                let mut acc = Complex64::new(0.0, 0.0);
                for (m, &(mi_m, mj_m)) in self.monitor_cells.iter().enumerate() {
                    let di = ri as f64 - mi_m as f64;
                    let dj = rj as f64 - mj_m as f64;
                    let decay = (-(di * di + dj * dj) / sigma_sq).exp();
                    acc += fom_dconj_e[m] * decay;
                }
                result[cell] = acc;
            }
        }

        Ok(result)
    }

    /// Compute the gradient dFOM/dρ using the adjoint method.
    ///
    /// FOM = Σ |E_z|² at monitor (here: all design cells, sum of intensities).
    /// Gradient: dFOM/dρ_i = -2 Re[E_fwd_i · conj(E_adj_i)] × (ε_max - ε_min)
    ///
    /// For the amplitude-maximisation FOM the adjoint source equals conj(E_fwd)
    /// at the monitor, so E_adj ≈ conj(E_fwd) and the gradient simplifies to:
    ///   dFOM/dρ_i ≈ -2 |E_fwd_i|² × (ε_max - ε_min)
    /// which is the correct sign for gradient ascent (increasing ε where the
    /// field is strong).
    pub fn compute_gradient(
        &self,
        e_fwd: &[num_complex::Complex64],
        e_adj: &[num_complex::Complex64],
    ) -> Result<crate::error::Result<FomGradient>, crate::error::OxiPhotonError> {
        let n = self.region.n_params();
        if e_fwd.len() != n || e_adj.len() != n {
            return Err(crate::error::OxiPhotonError::NumericalError(format!(
                "Field length mismatch: e_fwd={}, e_adj={}, n_params={}",
                e_fwd.len(),
                e_adj.len(),
                n
            )));
        }
        let de = self.region.eps_max - self.region.eps_min;
        let fom: f64 = e_fwd.iter().map(|c| c.norm_sqr()).sum();
        let grad: Vec<f64> = e_fwd
            .iter()
            .zip(e_adj.iter())
            .map(|(ef, ea)| {
                // Re[E_fwd · conj(E_adj)]
                let overlap = ef.re * ea.re + ef.im * ea.im;
                -2.0 * de * overlap
            })
            .collect();
        Ok(Ok(FomGradient { grad, fom }))
    }

    /// Update design variables using gradient ascent: ρ ← clip(ρ + α·∇FOM, 0, 1).
    pub fn update_gradient_ascent(&mut self, gradient: &FomGradient) {
        for (rho, &g) in self.region.rho.iter_mut().zip(&gradient.grad) {
            *rho = (*rho + self.step_size * g).clamp(0.0, 1.0);
        }
        self.iteration += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inverse::adjoint_3d::{AdjointSolver3d, DesignVariable};
    use std::f64::consts::PI;

    #[test]
    fn design_region_init() {
        let dr = DesignRegion::new(10, 10, 20e-9, 2.09, 12.11);
        assert_eq!(dr.n_params(), 100);
        assert_eq!(dr.rho.len(), 100);
    }

    #[test]
    fn epsilon_interpolation() {
        let mut dr = DesignRegion::new(2, 1, 20e-9, 1.0, 4.0);
        dr.rho[0] = 0.0;
        dr.rho[1] = 1.0;
        assert!((dr.epsilon(0, 0) - 1.0).abs() < 1e-10);
        assert!((dr.epsilon(1, 0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn fill_fraction_initial() {
        let dr = DesignRegion::new(10, 10, 20e-9, 1.0, 4.0);
        // All rho = 0.5, so fill fraction = 0 (none > 0.5)
        assert_eq!(dr.fill_fraction(), 0.0);
    }

    #[test]
    fn fill_fraction_all_ones() {
        let mut dr = DesignRegion::new(4, 4, 20e-9, 1.0, 4.0);
        for r in &mut dr.rho {
            *r = 1.0;
        }
        assert_eq!(dr.fill_fraction(), 1.0);
    }

    #[test]
    fn sigmoid_binarises() {
        let mut dr = DesignRegion::new(4, 1, 20e-9, 1.0, 4.0);
        dr.rho = vec![0.1, 0.3, 0.7, 0.9];
        dr.apply_sigmoid(100.0, 0.5);
        // Low values → 0, high values → 1
        assert!(dr.rho[0] < 0.1);
        assert!(dr.rho[3] > 0.9);
    }

    #[test]
    fn gradient_has_correct_length() {
        let c = 2.998e8;
        let omega = 2.0 * PI * c / 1550e-9;
        let dr = DesignRegion::soi_design(4, 4, 20.0);
        let solver = AdjointSolver::new(omega, dr);
        let n = solver.region.n_params();
        let e_fwd = vec![[1.0_f64, 0.0]; n];
        let e_adj = vec![[1.0_f64, 0.0]; n];
        let grad = solver.compute_gradient(&e_fwd, &e_adj);
        assert_eq!(grad.grad.len(), n);
    }

    #[test]
    fn gradient_sign_correct() {
        let c = 2.998e8;
        let omega = 2.0 * PI * c / 1550e-9;
        let dr = DesignRegion::new(1, 1, 20e-9, 1.0, 4.0);
        let solver = AdjointSolver::new(omega, dr);
        // Positive overlap → negative gradient (maximise FOM means reduce eps where field is high)
        let e_fwd = vec![[1.0, 0.0]];
        let e_adj = vec![[1.0, 0.0]];
        let grad = solver.compute_gradient(&e_fwd, &e_adj);
        assert!(grad.grad[0] < 0.0);
    }

    #[test]
    fn adam_update_stays_in_bounds() {
        let c = 2.998e8;
        let omega = 2.0 * PI * c / 1550e-9;
        let dr = DesignRegion::new(4, 4, 20e-9, 1.0, 4.0);
        let mut solver = AdjointSolver::new(omega, dr);
        let n = solver.region.n_params();
        let e_fwd = vec![[2.0, 0.0]; n];
        let e_adj = vec![[2.0, 0.0]; n];
        let grad = solver.compute_gradient(&e_fwd, &e_adj);
        let mut m = vec![0.0; n];
        let mut v = vec![0.0; n];
        solver.update_adam(&grad, &mut m, &mut v, 1, 0.9, 0.999, 1e-8);
        for &r in &solver.region.rho {
            assert!((0.0..=1.0).contains(&r), "rho={r:.4} out of [0,1]");
        }
    }

    // ── AdjointSolver3d tests ─────────────────────────────────────────────

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

        // At least some rho values should change after a gradient step
        let any_changed = solver
            .variables
            .iter()
            .zip(rho_before.iter())
            .any(|(v, &r0)| (v.rho - r0).abs() > 1e-15);
        assert!(any_changed, "gradient_step should update at least one rho");

        // History should record the FOM
        assert_eq!(solver.history.len(), 1);
        assert!(solver.iteration == 1);
    }

    #[test]
    fn adjoint_solver3d_rho_stays_in_bounds() {
        let mut solver = AdjointSolver3d::new(8, 6, 10, 20e-9, 1.2e15);
        solver.fill_design_region(2, 6, 1, 5, 2, 8, 2.09, 12.11);

        // Run several gradient steps
        for _ in 0..5 {
            solver.gradient_step(1.0); // large step to stress-test clamping
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
}
