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

/// A single design variable mapping a physical parameter to a normalised
/// optimisation variable ρ ∈ \[0, 1\].
///
/// Supports any scalar field (permittivity, conductivity, geometry parameter)
/// with optional upper/lower bounds and a projection function.
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

/// 3D adjoint sensitivity solver.
///
/// Wraps an FDTD-like structure (represented by permittivity grids) and
/// provides the forward / adjoint field computation framework needed for
/// gradient-based topology optimisation.
///
/// In a full FDTD-coupled implementation the forward and adjoint runs would
/// drive the actual `Fdtd3d` engine.  Here we implement the complete
/// mathematical infrastructure: field storage, gradient computation, design
/// variable management, and optimisation loops.
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
    /// Forward field (Ez component, complex \[re, im\] per cell)
    pub e_fwd: Vec<[f64; 2]>,
    /// Adjoint field (Ez component, complex \[re, im\] per cell)
    pub e_adj: Vec<[f64; 2]>,
    /// Computed gradient ∂FOM/∂ρ_i for each design variable
    pub gradient: Vec<f64>,
    /// Iteration history: (iteration, FOM)
    pub history: Vec<(usize, f64)>,
    /// Current FOM value
    pub fom: f64,
    /// Iteration counter
    pub iteration: usize,
}

impl AdjointSolver3d {
    /// Create a new 3D adjoint solver.
    ///
    /// # Arguments
    /// * `nx, ny, nz` – grid dimensions
    /// * `dx`         – uniform cell size (m)
    /// * `omega`      – angular frequency ω = 2πc/λ (rad/s)
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
        }
    }

    /// SOI-waveguide optimisation problem (220 nm height, Si/SiO₂).
    pub fn soi(nx: usize, ny: usize, nz: usize, resolution_nm: f64) -> Self {
        use std::f64::consts::PI;
        let c = 2.998e8_f64;
        let lambda = 1550e-9_f64;
        let omega = 2.0 * PI * c / lambda;
        Self::new(nx, ny, nz, resolution_nm * 1e-9, omega)
    }

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

    /// Simulate the "forward" run — in a full implementation this calls FDTD.
    ///
    /// Here we populate `e_fwd` with a simplified Gaussian mode estimate
    /// centred on the design region, to provide a meaningful gradient when
    /// chained with `compute_adjoint_field` and `compute_gradient`.
    pub fn compute_forward_field(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let xc = nx as f64 / 2.0;
        let yc = ny as f64 / 2.0;
        let wx = nx as f64 / 6.0;
        let wy = ny as f64 / 6.0;

        for k in 0..nz {
            // Simple plane-wave with Gaussian transverse profile
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
        // FOM = integrated |E|² at output plane
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

    /// Simulate the "adjoint" run — drives adjoint source at the monitor.
    ///
    /// The adjoint source is proportional to conj(E_fwd) at the output
    /// monitor plane.  Here we propagate a reversed (–z) Gaussian wave
    /// from the output plane back through the domain.
    pub fn compute_adjoint_field(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let xc = nx as f64 / 2.0;
        let yc = ny as f64 / 2.0;
        let wx = nx as f64 / 6.0;
        let wy = ny as f64 / 6.0;

        for k in 0..nz {
            let phase_k = self.omega * (nz - 1 - k) as f64 * self.dx / 2.998e8;
            let (sin_k, cos_k) = phase_k.sin_cos();
            for j in 0..ny {
                let yy = (j as f64 - yc) / wy;
                for i in 0..nx {
                    let xx = (i as f64 - xc) / wx;
                    let env = (-0.5 * (xx * xx + yy * yy)).exp();
                    let idx = k * (nx * ny) + j * nx + i;
                    // Adjoint source = conj(E_fwd) amplitude backward
                    self.e_adj[idx] = [env * cos_k, -env * sin_k];
                }
            }
        }
    }

    /// Compute the gradient ∂FOM/∂ρ_i for each design variable.
    ///
    /// Uses the adjoint formula:
    ///   ∂FOM/∂ρ_i = -2ω²ε₀ · (ε_max - ε_min) · Re\[E_fwd · conj(E_adj)\]_i · V_cell
    ///
    /// where V_cell = dx³ is the cell volume.
    pub fn compute_gradient(&mut self) {
        let eps0 = 8.854e-12_f64;
        let omega = self.omega;
        let dx3 = self.dx * self.dx * self.dx;
        let nx = self.nx;
        let ny = self.ny;

        for (var_idx, var) in self.variables.iter().enumerate() {
            // Extract pixel coordinates from variable name (stored as index into variables)
            // We rely on the order variables were added via fill_design_region
            // Each variable corresponds to a unique cell; we use var_idx as proxy
            let de = var.p_max - var.p_min;
            // Find the corresponding cell index (variables added in k-j-i order)
            let cell_idx = var_idx; // 1:1 mapping when fill_design_region used
            let idx = cell_idx.min(self.e_fwd.len().saturating_sub(1));

            // Map var_idx → (i, j, k) using the design region shape
            // (assume variables cover a contiguous block starting at some offset)
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

            // Re[E_fwd · conj(E_adj)]
            let overlap = ef_re * ea_re + ef_im * ea_im;
            self.gradient[var_idx] = -2.0 * omega * omega * eps0 * de * overlap * dx3;
        }

        // Update gradient on each DesignVariable
        for (var, &g) in self.variables.iter_mut().zip(self.gradient.iter()) {
            var.gradient = g;
        }
    }

    /// Perform one gradient-ascent step and record history.
    ///
    /// Runs forward → adjoint → gradient → update in one call.
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

    /// L2 norm of the current gradient vector.
    pub fn gradient_norm(&self) -> f64 {
        self.gradient.iter().map(|g| g * g).sum::<f64>().sqrt()
    }

    /// Number of design variables.
    pub fn n_variables(&self) -> usize {
        self.variables.len()
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
}

#[cfg(test)]
mod tests {
    use super::*;
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
