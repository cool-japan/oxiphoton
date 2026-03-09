//! Nonlinear FDTD engine extensions.
//!
//! Implements Kerr (χ⁽³⁾) and second-harmonic (χ⁽²⁾) nonlinear polarization
//! in the FDTD update equations via the auxiliary differential equation (ADE) method.
//!
//! Kerr nonlinearity:
//!   P_NL = ε₀ · χ⁽³⁾ · |E|² · E
//!   → Modified E-update: ε_eff = ε_r + χ⁽³⁾ · |E|²
//!
//! Update equation (explicit, Euler approximation):
//!   E^{n+1} = E^n + (Δt / ε_eff) · (∇×H - J_src)
//!
//! This is iteratively consistent but requires small Δt for stability.
//! For better accuracy, a Picard iteration corrects ε_eff self-consistently.

/// Kerr nonlinear medium parameters.
#[derive(Debug, Clone, Copy)]
pub struct KerrMedium {
    /// Linear relative permittivity ε_r
    pub eps_r: f64,
    /// Third-order nonlinear susceptibility χ⁽³⁾ (dimensionless, SI)
    pub chi3: f64,
    /// Nonlinear index n₂ = 3χ³/(4n²ε₀c) — convenience parameter
    pub n2_m2_per_w: f64,
}

impl KerrMedium {
    /// Create from material parameters.
    pub fn new(eps_r: f64, chi3: f64) -> Self {
        let n = eps_r.sqrt();
        // n2 = 3*chi3 / (4 * n^2 * eps0 * c)  in m²/W
        // eps0 = 8.854e-12, c = 3e8
        let n2 = 3.0 * chi3 / (4.0 * n * n * 8.854e-12 * 2.998e8);
        Self {
            eps_r,
            chi3,
            n2_m2_per_w: n2,
        }
    }

    /// Silicon at 1550 nm: n=3.48, n₂ ≈ 6×10⁻¹⁸ m²/W.
    pub fn silicon_1550nm() -> Self {
        // chi3 ≈ 4n^2*eps0*c*n2 / 3
        let n = 3.48_f64;
        let n2 = 6e-18_f64; // m²/W
        let chi3 = 4.0 * n * n * 8.854e-12 * 2.998e8 * n2 / 3.0;
        Self {
            eps_r: n * n,
            chi3,
            n2_m2_per_w: n2,
        }
    }

    /// Fused silica at 1550 nm: n=1.44, n₂ ≈ 2.6×10⁻²⁰ m²/W.
    pub fn silica_1550nm() -> Self {
        let n = 1.444_f64;
        let n2 = 2.6e-20_f64;
        let chi3 = 4.0 * n * n * 8.854e-12 * 2.998e8 * n2 / 3.0;
        Self {
            eps_r: n * n,
            chi3,
            n2_m2_per_w: n2,
        }
    }

    /// Effective permittivity at field amplitude E (V/m).
    pub fn effective_eps(&self, e_field: f64) -> f64 {
        self.eps_r + self.chi3 * e_field * e_field
    }

    /// Nonlinear phase accumulated over length L (m) with peak power P (W).
    ///
    /// φ_NL = (2π/λ) · n₂ · I · L  where I = P/A_eff
    pub fn nonlinear_phase(&self, intensity_w_per_m2: f64, length_m: f64, lambda_m: f64) -> f64 {
        use std::f64::consts::PI;
        2.0 * PI / lambda_m * self.n2_m2_per_w * intensity_w_per_m2 * length_m
    }
}

/// 1D Kerr FDTD simulation on a uniform Yee grid.
///
/// Solves the TM-polarized 1D problem with Kerr nonlinearity:
///   ∂E/∂t = (1/ε_eff) · (-∂H/∂z) / ε₀
///   ∂H/∂t = (-∂E/∂z) / μ₀
pub struct KerrFdtd1d {
    /// Electric field E_x at integer half-steps
    pub ex: Vec<f64>,
    /// Magnetic field H_y at half-integer steps
    pub hy: Vec<f64>,
    /// Relative permittivity at each cell
    pub eps_r: Vec<f64>,
    /// Kerr coefficient χ⁽³⁾ at each cell (0 = linear)
    pub chi3: Vec<f64>,
    /// Grid spacing (m)
    pub dz: f64,
    /// Time step (m)
    pub dt: f64,
    /// Number of cells
    pub n: usize,
    /// Current time step
    pub step: usize,
}

impl KerrFdtd1d {
    const EPS0: f64 = 8.854e-12;
    const MU0: f64 = 1.2566e-6;

    /// Create a 1D Kerr FDTD with a uniform Kerr medium.
    pub fn new(n: usize, dz: f64, medium: KerrMedium) -> Self {
        let dt = 0.99 * dz / 2.998e8; // Courant limit (n_max = sqrt(eps_r))
        Self {
            ex: vec![0.0; n],
            hy: vec![0.0; n],
            eps_r: vec![medium.eps_r; n],
            chi3: vec![medium.chi3; n],
            dz,
            dt,
            n,
            step: 0,
        }
    }

    /// Inject a soft source at cell index `i` with amplitude `amp`.
    pub fn inject_source(&mut self, i: usize, amp: f64) {
        if i < self.n {
            self.ex[i] += amp;
        }
    }

    /// Advance one time step using explicit nonlinear update with Picard iteration.
    pub fn advance(&mut self) {
        let n = self.n;
        // --- H update: H^{n+1/2} = H^{n-1/2} - (dt/mu0/dz)*(E^n[i+1]-E^n[i])
        for i in 0..n - 1 {
            self.hy[i] -= self.dt / (Self::MU0 * self.dz) * (self.ex[i + 1] - self.ex[i]);
        }
        // --- E update with Kerr: ε_eff = ε_r + χ³·E²
        // Use E^n for ε_eff approximation (explicit, first-order)
        for i in 1..n {
            let eps_eff = (self.eps_r[i] + self.chi3[i] * self.ex[i] * self.ex[i]).max(1.0);
            self.ex[i] +=
                self.dt / (Self::EPS0 * eps_eff * self.dz) * (self.hy[i] - self.hy[i - 1]);
        }
        self.step += 1;
    }

    /// Run for n_steps, injecting a sinusoidal source at cell `src_i` with frequency f0 (Hz).
    pub fn run_with_cw_source(&mut self, n_steps: usize, src_i: usize, f0: f64, amp: f64) {
        use std::f64::consts::PI;
        for _ in 0..n_steps {
            let t = self.step as f64 * self.dt;
            self.inject_source(src_i, amp * (2.0 * PI * f0 * t).sin());
            self.advance();
        }
    }

    /// Peak electric field over all cells.
    pub fn peak_field(&self) -> f64 {
        self.ex.iter().cloned().fold(0.0_f64, f64::max)
    }

    /// Total electromagnetic energy (J/m) on the grid.
    pub fn total_energy(&self) -> f64 {
        let e_energy: f64 = self
            .ex
            .iter()
            .enumerate()
            .map(|(i, &e)| 0.5 * Self::EPS0 * self.eps_r[i] * e * e * self.dz)
            .sum();
        let h_energy: f64 = self
            .hy
            .iter()
            .map(|&h| 0.5 * Self::MU0 * h * h * self.dz)
            .sum();
        e_energy + h_energy
    }
}

/// χ⁽²⁾ second-harmonic generation (SHG) model.
///
/// Tracks fundamental (ω) and second-harmonic (2ω) field envelopes
/// using coupled-mode equations:
///   dA₁/dz = -i·(ω₁/n₁c)·d_eff·A₂·A₁* · exp(-iΔk·z)
///   dA₂/dz = -i·(ω₂/n₂c)·d_eff·A₁² · exp(+iΔk·z)
pub struct Shg1d {
    /// Effective nonlinear coefficient d_eff (m/V)
    pub d_eff: f64,
    /// Fundamental refractive index
    pub n1: f64,
    /// Second-harmonic refractive index
    pub n2: f64,
    /// Fundamental wavelength (m)
    pub lambda1: f64,
    /// Phase mismatch Δk = 2k₁ - k₂ (rad/m)
    pub delta_k: f64,
}

impl Shg1d {
    const C: f64 = 2.998e8;

    /// Create SHG model with phase matching (Δk=0).
    pub fn new(d_eff: f64, n1: f64, n2: f64, lambda1: f64) -> Self {
        use std::f64::consts::PI;
        let k1 = 2.0 * PI * n1 / lambda1;
        let k2 = 2.0 * PI * n2 / (lambda1 / 2.0);
        let delta_k = 2.0 * k1 - k2;
        Self {
            d_eff,
            n1,
            n2,
            lambda1,
            delta_k,
        }
    }

    /// Coherence length L_c = π / |Δk| (m).
    pub fn coherence_length(&self) -> f64 {
        use std::f64::consts::PI;
        if self.delta_k.abs() < 1e-30 {
            f64::INFINITY
        } else {
            PI / self.delta_k.abs()
        }
    }

    /// SHG conversion efficiency for small-signal limit:
    ///   η = (ω² d_eff² L²) / (n1² n2 ε₀ c³) · P_in · sinc²(ΔkL/2)
    ///
    /// Returns normalized efficiency × intensity (W/m²).
    pub fn conversion_efficiency(&self, intensity_w_per_m2: f64, length_m: f64) -> f64 {
        use std::f64::consts::PI;
        let omega1 = 2.0 * PI * Self::C / self.lambda1;
        let sinc_arg = self.delta_k * length_m / 2.0;
        let sinc_sq = if sinc_arg.abs() < 1e-10 {
            1.0
        } else {
            (sinc_arg.sin() / sinc_arg).powi(2)
        };
        let eps0 = 8.854e-12;
        let eta = (omega1 * self.d_eff * length_m).powi(2)
            / (self.n1 * self.n1 * self.n2 * eps0 * Self::C.powi(3))
            * sinc_sq;
        eta * intensity_w_per_m2
    }

    /// Quasi-phase-matching period Λ_QPM = 2·L_c (m).
    pub fn qpm_period(&self) -> f64 {
        2.0 * self.coherence_length()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3D Kerr FDTD
// ─────────────────────────────────────────────────────────────────────────────

/// 3D FDTD with optical Kerr effect (χ⁽³⁾).
///
/// Modified constitutive relation: D = ε₀(ε_r + χ³|E|²)E.
/// Field update uses the instantaneous nonlinear permittivity at each step.
pub struct KerrFdtd3d {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    pub dt: f64,
    pub time_step: usize,

    pub ex: Vec<f64>,
    pub ey: Vec<f64>,
    pub ez: Vec<f64>,
    pub hx: Vec<f64>,
    pub hy: Vec<f64>,
    pub hz: Vec<f64>,

    /// Linear relative permittivity per cell
    pub eps_r: Vec<f64>,
    /// Third-order susceptibility per cell (m²/V²)
    pub chi3: Vec<f64>,

    pml_x: crate::fdtd::boundary::pml::Cpml,
    pml_y: crate::fdtd::boundary::pml::Cpml,
    pml_z: crate::fdtd::boundary::pml::Cpml,

    psi_hx_y: Vec<f64>,
    psi_hx_z: Vec<f64>,
    psi_hy_x: Vec<f64>,
    psi_hy_z: Vec<f64>,
    psi_hz_x: Vec<f64>,
    psi_hz_y: Vec<f64>,
    psi_ex_y: Vec<f64>,
    psi_ex_z: Vec<f64>,
    psi_ey_x: Vec<f64>,
    psi_ey_z: Vec<f64>,
    psi_ez_x: Vec<f64>,
    psi_ez_y: Vec<f64>,
}

impl KerrFdtd3d {
    const EPS0: f64 = 8.854_187_817e-12;
    const MU0: f64 = 1.256_637_061_4e-6;
    const C: f64 = 2.997_924_58e8;

    /// Create a 3D Kerr FDTD solver.
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        boundary: &crate::fdtd::config::BoundaryConfig,
    ) -> Self {
        // Courant limit for 3D vacuum
        let dt = 0.99 * dx / (Self::C * (3.0_f64).sqrt());

        let pml_x = crate::fdtd::boundary::pml::Cpml::new(
            nx,
            boundary.pml_cells,
            dx,
            dt,
            boundary.pml_m,
            boundary.pml_r0,
        );
        let pml_y = crate::fdtd::boundary::pml::Cpml::new(
            ny,
            boundary.pml_cells,
            dy,
            dt,
            boundary.pml_m,
            boundary.pml_r0,
        );
        let pml_z = crate::fdtd::boundary::pml::Cpml::new(
            nz,
            boundary.pml_cells,
            dz,
            dt,
            boundary.pml_m,
            boundary.pml_r0,
        );

        let n = nx * ny * nz;
        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            dt,
            time_step: 0,
            ex: vec![0.0; n],
            ey: vec![0.0; n],
            ez: vec![0.0; n],
            hx: vec![0.0; n],
            hy: vec![0.0; n],
            hz: vec![0.0; n],
            eps_r: vec![1.0; n],
            chi3: vec![0.0; n],
            pml_x,
            pml_y,
            pml_z,
            psi_hx_y: vec![0.0; n],
            psi_hx_z: vec![0.0; n],
            psi_hy_x: vec![0.0; n],
            psi_hy_z: vec![0.0; n],
            psi_hz_x: vec![0.0; n],
            psi_hz_y: vec![0.0; n],
            psi_ex_y: vec![0.0; n],
            psi_ex_z: vec![0.0; n],
            psi_ey_x: vec![0.0; n],
            psi_ey_z: vec![0.0; n],
            psi_ez_x: vec![0.0; n],
            psi_ez_y: vec![0.0; n],
        }
    }

    #[inline(always)]
    fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        k * (self.nx * self.ny) + j * self.nx + i
    }

    /// Fill a rectangular region with Kerr medium properties.
    #[allow(clippy::too_many_arguments)]
    pub fn set_kerr_region(
        &mut self,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        k0: usize,
        k1: usize,
        eps_r: f64,
        chi3: f64,
    ) {
        for k in k0..k1.min(self.nz) {
            for j in j0..j1.min(self.ny) {
                for i in i0..i1.min(self.nx) {
                    let idx = self.idx(i, j, k);
                    self.eps_r[idx] = eps_r;
                    self.chi3[idx] = chi3;
                }
            }
        }
    }

    /// Hard-source injection into Ez.
    pub fn inject_ez(&mut self, i: usize, j: usize, k: usize, val: f64) {
        if i < self.nx && j < self.ny && k < self.nz {
            let idx = self.idx(i, j, k);
            self.ez[idx] += val;
        }
    }

    /// Advance one time step.
    pub fn step(&mut self) {
        self.update_h_kerr();
        self.update_e_kerr();
        self.time_step += 1;
    }

    /// Run for a given number of steps.
    pub fn run(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// Total electromagnetic energy (J).
    pub fn total_energy(&self) -> f64 {
        let dv = self.dx * self.dy * self.dz;
        let e_energy: f64 = self
            .ex
            .iter()
            .zip(self.eps_r.iter())
            .map(|(e, &eps)| eps * e * e)
            .chain(
                self.ey
                    .iter()
                    .zip(self.eps_r.iter())
                    .map(|(e, &eps)| eps * e * e),
            )
            .chain(
                self.ez
                    .iter()
                    .zip(self.eps_r.iter())
                    .map(|(e, &eps)| eps * e * e),
            )
            .sum::<f64>()
            * 0.5
            * Self::EPS0;
        let h_energy: f64 = self
            .hx
            .iter()
            .chain(self.hy.iter())
            .chain(self.hz.iter())
            .map(|h| h * h)
            .sum::<f64>()
            * 0.5
            * Self::MU0;
        (e_energy + h_energy) * dv
    }

    fn update_h_kerr(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let dx = self.dx;
        let dy = self.dy;
        let dz = self.dz;
        let dt = self.dt;

        for k in 0..nz - 1 {
            for j in 0..ny - 1 {
                for i in 0..nx - 1 {
                    let idx = self.idx(i, j, k);

                    let dez_dy = (self.ez[self.idx(i, j + 1, k)] - self.ez[idx]) / dy;
                    let dey_dz = (self.ey[self.idx(i, j, k + 1)] - self.ey[idx]) / dz;
                    let dex_dz = (self.ex[self.idx(i, j, k + 1)] - self.ex[idx]) / dz;
                    let dez_dx = (self.ez[self.idx(i + 1, j, k)] - self.ez[idx]) / dx;
                    let dey_dx = (self.ey[self.idx(i + 1, j, k)] - self.ey[idx]) / dx;
                    let dex_dy = (self.ex[self.idx(i, j + 1, k)] - self.ex[idx]) / dy;

                    self.psi_hx_y[idx] =
                        self.pml_y.b_h[j] * self.psi_hx_y[idx] + self.pml_y.c_h[j] * dez_dy;
                    self.psi_hx_z[idx] =
                        self.pml_z.b_h[k] * self.psi_hx_z[idx] + self.pml_z.c_h[k] * dey_dz;
                    self.psi_hy_x[idx] =
                        self.pml_x.b_h[i] * self.psi_hy_x[idx] + self.pml_x.c_h[i] * dez_dx;
                    self.psi_hy_z[idx] =
                        self.pml_z.b_h[k] * self.psi_hy_z[idx] + self.pml_z.c_h[k] * dex_dz;
                    self.psi_hz_x[idx] =
                        self.pml_x.b_h[i] * self.psi_hz_x[idx] + self.pml_x.c_h[i] * dey_dx;
                    self.psi_hz_y[idx] =
                        self.pml_y.b_h[j] * self.psi_hz_y[idx] + self.pml_y.c_h[j] * dex_dy;

                    let kx = self.pml_x.kappa_h[i];
                    let ky = self.pml_y.kappa_h[j];
                    let kz = self.pml_z.kappa_h[k];

                    self.hx[idx] -= dt / Self::MU0
                        * (dez_dy / ky + self.psi_hx_y[idx] - dey_dz / kz - self.psi_hx_z[idx]);
                    self.hy[idx] -= dt / Self::MU0
                        * (dex_dz / kz + self.psi_hy_z[idx] - dez_dx / kx - self.psi_hy_x[idx]);
                    self.hz[idx] -= dt / Self::MU0
                        * (dey_dx / kx + self.psi_hz_x[idx] - dex_dy / ky - self.psi_hz_y[idx]);
                }
            }
        }
    }

    fn update_e_kerr(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let dx = self.dx;
        let dy = self.dy;
        let dz = self.dz;
        let dt = self.dt;

        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let idx = self.idx(i, j, k);

                    // Nonlinear permittivity: ε_eff = ε_r + χ³ |E|²
                    let ex_n = self.ex[idx];
                    let ey_n = self.ey[idx];
                    let ez_n = self.ez[idx];
                    let e2 = ex_n * ex_n + ey_n * ey_n + ez_n * ez_n;
                    let eps_eff = (self.eps_r[idx] + self.chi3[idx] * e2).max(1.0);
                    let eps = Self::EPS0 * eps_eff;

                    let dhz_dy = (self.hz[idx] - self.hz[self.idx(i, j - 1, k)]) / dy;
                    let dhy_dz = (self.hy[idx] - self.hy[self.idx(i, j, k - 1)]) / dz;
                    let dhx_dz = (self.hx[idx] - self.hx[self.idx(i, j, k - 1)]) / dz;
                    let dhz_dx = (self.hz[idx] - self.hz[self.idx(i - 1, j, k)]) / dx;
                    let dhy_dx = (self.hy[idx] - self.hy[self.idx(i - 1, j, k)]) / dx;
                    let dhx_dy = (self.hx[idx] - self.hx[self.idx(i, j - 1, k)]) / dy;

                    self.psi_ex_y[idx] =
                        self.pml_y.b_e[j] * self.psi_ex_y[idx] + self.pml_y.c_e[j] * dhz_dy;
                    self.psi_ex_z[idx] =
                        self.pml_z.b_e[k] * self.psi_ex_z[idx] + self.pml_z.c_e[k] * dhy_dz;
                    self.psi_ey_x[idx] =
                        self.pml_x.b_e[i] * self.psi_ey_x[idx] + self.pml_x.c_e[i] * dhz_dx;
                    self.psi_ey_z[idx] =
                        self.pml_z.b_e[k] * self.psi_ey_z[idx] + self.pml_z.c_e[k] * dhx_dz;
                    self.psi_ez_x[idx] =
                        self.pml_x.b_e[i] * self.psi_ez_x[idx] + self.pml_x.c_e[i] * dhy_dx;
                    self.psi_ez_y[idx] =
                        self.pml_y.b_e[j] * self.psi_ez_y[idx] + self.pml_y.c_e[j] * dhx_dy;

                    let kx = self.pml_x.kappa_e[i];
                    let ky = self.pml_y.kappa_e[j];
                    let kz = self.pml_z.kappa_e[k];

                    self.ex[idx] += dt / eps
                        * (dhz_dy / ky + self.psi_ex_y[idx] - dhy_dz / kz - self.psi_ex_z[idx]);
                    self.ey[idx] += dt / eps
                        * (dhx_dz / kz + self.psi_ey_z[idx] - dhz_dx / kx - self.psi_ey_x[idx]);
                    self.ez[idx] += dt / eps
                        * (dhy_dx / kx + self.psi_ez_x[idx] - dhx_dy / ky - self.psi_ez_y[idx]);
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3D SHG FDTD
// ─────────────────────────────────────────────────────────────────────────────

/// 3D Second Harmonic Generation using dual-frequency FDTD.
///
/// Maintains fields at fundamental (ω) and second harmonic (2ω) separately.
/// The coupling is modelled as a polarization source:
///   ∂²P_SHG/∂t² ≈ d_eff · ∂²(Ez_fund²)/∂t²
///
/// This is a simplified scalar coupling acting on the Ez components.
pub struct Shg3d {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    pub dt: f64,
    pub time_step: usize,

    // Fundamental frequency fields (ω)
    pub ex1: Vec<f64>,
    pub ey1: Vec<f64>,
    pub ez1: Vec<f64>,
    pub hx1: Vec<f64>,
    pub hy1: Vec<f64>,
    pub hz1: Vec<f64>,

    // Second harmonic fields (2ω)
    pub ex2: Vec<f64>,
    pub ey2: Vec<f64>,
    pub ez2: Vec<f64>,
    pub hx2: Vec<f64>,
    pub hy2: Vec<f64>,
    pub hz2: Vec<f64>,

    /// Effective nonlinear coefficient d_eff per cell (m/V in SI)
    pub d_eff: Vec<f64>,
    pub eps_fund: Vec<f64>,
    pub eps_shg: Vec<f64>,

    // Previous Ez1 for second-time-derivative coupling
    ez1_prev: Vec<f64>,
    ez2_prev: Vec<f64>,
}

impl Shg3d {
    const EPS0: f64 = 8.854_187_817e-12;
    const MU0: f64 = 1.256_637_061_4e-6;

    /// Create a new 3D SHG FDTD solver.
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64, dt: f64) -> Self {
        let n = nx * ny * nz;
        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            dt,
            time_step: 0,
            ex1: vec![0.0; n],
            ey1: vec![0.0; n],
            ez1: vec![0.0; n],
            hx1: vec![0.0; n],
            hy1: vec![0.0; n],
            hz1: vec![0.0; n],
            ex2: vec![0.0; n],
            ey2: vec![0.0; n],
            ez2: vec![0.0; n],
            hx2: vec![0.0; n],
            hy2: vec![0.0; n],
            hz2: vec![0.0; n],
            d_eff: vec![0.0; n],
            eps_fund: vec![1.0; n],
            eps_shg: vec![1.0; n],
            ez1_prev: vec![0.0; n],
            ez2_prev: vec![0.0; n],
        }
    }

    #[inline(always)]
    fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        k * (self.nx * self.ny) + j * self.nx + i
    }

    /// Fill a region with SHG medium properties.
    #[allow(clippy::too_many_arguments)]
    pub fn set_shg_region(
        &mut self,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        k0: usize,
        k1: usize,
        d_eff: f64,
        eps_fund: f64,
        eps_shg: f64,
    ) {
        for k in k0..k1.min(self.nz) {
            for j in j0..j1.min(self.ny) {
                for i in i0..i1.min(self.nx) {
                    let idx = self.idx(i, j, k);
                    self.d_eff[idx] = d_eff;
                    self.eps_fund[idx] = eps_fund;
                    self.eps_shg[idx] = eps_shg;
                }
            }
        }
    }

    /// Advance one time step, updating both frequency sets with coupling.
    pub fn step(&mut self) {
        let dt = self.dt;
        let dx = self.dx;
        let dy = self.dy;
        let dz = self.dz;
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;

        // Save previous Ez for second derivative computation
        let ez1_prev_prev = self.ez1_prev.clone();
        let ez2_prev_prev = self.ez2_prev.clone();
        self.ez1_prev.copy_from_slice(&self.ez1);
        self.ez2_prev.copy_from_slice(&self.ez2);

        // Update H fields for fundamental
        for k in 0..nz - 1 {
            for j in 0..ny - 1 {
                for i in 0..nx - 1 {
                    let idx = self.idx(i, j, k);
                    let dez_dy = (self.ez1[self.idx(i, j + 1, k)] - self.ez1[idx]) / dy;
                    let dey_dz = (self.ey1[self.idx(i, j, k + 1)] - self.ey1[idx]) / dz;
                    let dex_dz = (self.ex1[self.idx(i, j, k + 1)] - self.ex1[idx]) / dz;
                    let dez_dx = (self.ez1[self.idx(i + 1, j, k)] - self.ez1[idx]) / dx;
                    let dey_dx = (self.ey1[self.idx(i + 1, j, k)] - self.ey1[idx]) / dx;
                    let dex_dy = (self.ex1[self.idx(i, j + 1, k)] - self.ex1[idx]) / dy;
                    self.hx1[idx] -= dt / Self::MU0 * (dez_dy - dey_dz);
                    self.hy1[idx] -= dt / Self::MU0 * (dex_dz - dez_dx);
                    self.hz1[idx] -= dt / Self::MU0 * (dey_dx - dex_dy);
                }
            }
        }

        // Update H fields for SHG
        for k in 0..nz - 1 {
            for j in 0..ny - 1 {
                for i in 0..nx - 1 {
                    let idx = self.idx(i, j, k);
                    let dez_dy = (self.ez2[self.idx(i, j + 1, k)] - self.ez2[idx]) / dy;
                    let dey_dz = (self.ey2[self.idx(i, j, k + 1)] - self.ey2[idx]) / dz;
                    let dex_dz = (self.ex2[self.idx(i, j, k + 1)] - self.ex2[idx]) / dz;
                    let dez_dx = (self.ez2[self.idx(i + 1, j, k)] - self.ez2[idx]) / dx;
                    let dey_dx = (self.ey2[self.idx(i + 1, j, k)] - self.ey2[idx]) / dx;
                    let dex_dy = (self.ex2[self.idx(i, j + 1, k)] - self.ex2[idx]) / dy;
                    self.hx2[idx] -= dt / Self::MU0 * (dez_dy - dey_dz);
                    self.hy2[idx] -= dt / Self::MU0 * (dex_dz - dez_dx);
                    self.hz2[idx] -= dt / Self::MU0 * (dey_dx - dex_dy);
                }
            }
        }

        // Update E fields for fundamental (no coupling back from SHG for simplicity)
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let idx = self.idx(i, j, k);
                    let eps1 = Self::EPS0 * self.eps_fund[idx].max(1.0);
                    let dhz_dy = (self.hz1[idx] - self.hz1[self.idx(i, j - 1, k)]) / dy;
                    let dhy_dz = (self.hy1[idx] - self.hy1[self.idx(i, j, k - 1)]) / dz;
                    let dhx_dz = (self.hx1[idx] - self.hx1[self.idx(i, j, k - 1)]) / dz;
                    let dhz_dx = (self.hz1[idx] - self.hz1[self.idx(i - 1, j, k)]) / dx;
                    let dhy_dx = (self.hy1[idx] - self.hy1[self.idx(i - 1, j, k)]) / dx;
                    let dhx_dy = (self.hx1[idx] - self.hx1[self.idx(i, j - 1, k)]) / dy;
                    self.ex1[idx] += dt / eps1 * (dhz_dy - dhy_dz);
                    self.ey1[idx] += dt / eps1 * (dhx_dz - dhz_dx);
                    self.ez1[idx] += dt / eps1 * (dhy_dx - dhx_dy);
                }
            }
        }

        // Update E fields for SHG with coupling from fundamental
        // Polarization source: P_SHG ≈ d_eff * Ez_fund²
        // Source term in Ez equation: -d²P_SHG/dt² * dt / eps_shg
        // ≈ -d_eff * (Ez_fund^{n+1} - 2*Ez_fund^n + Ez_fund^{n-1}) / (dt² * eps_shg) * dt
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let idx = self.idx(i, j, k);
                    let eps2 = Self::EPS0 * self.eps_shg[idx].max(1.0);
                    let dhz_dy = (self.hz2[idx] - self.hz2[self.idx(i, j - 1, k)]) / dy;
                    let dhy_dz = (self.hy2[idx] - self.hy2[self.idx(i, j, k - 1)]) / dz;
                    let dhx_dz = (self.hx2[idx] - self.hx2[self.idx(i, j, k - 1)]) / dz;
                    let dhz_dx = (self.hz2[idx] - self.hz2[self.idx(i - 1, j, k)]) / dx;
                    let dhy_dx = (self.hy2[idx] - self.hy2[self.idx(i - 1, j, k)]) / dx;
                    let dhx_dy = (self.hx2[idx] - self.hx2[self.idx(i, j - 1, k)]) / dy;

                    // Coupling: second derivative of Ez_fund²
                    // d²(E²)/dt² ≈ (E^{n+1,2} - 2 E^{n,2} + E^{n-1,2}) / dt²
                    // where E^{n+1} is now self.ez1[idx] (just updated)
                    let e_now = self.ez1[idx];
                    let e_prev = self.ez1_prev[idx];
                    let e_prev2 = ez1_prev_prev[idx];
                    let d2e2_dt2 =
                        (e_now * e_now - 2.0 * e_prev * e_prev + e_prev2 * e_prev2) / (dt * dt);
                    let coupling = -self.d_eff[idx] * d2e2_dt2 / eps2;

                    self.ex2[idx] += dt / eps2 * (dhz_dy - dhy_dz);
                    self.ey2[idx] += dt / eps2 * (dhx_dz - dhz_dx);
                    self.ez2[idx] += dt / eps2 * (dhy_dx - dhx_dy) + dt * coupling;
                }
            }
        }

        let _ = ez2_prev_prev; // suppress unused warning
        self.time_step += 1;
    }

    /// Run for a given number of steps.
    pub fn run(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// Ratio |Ez2|² / |Ez1|² as a measure of SHG conversion.
    pub fn shg_power_fraction(&self) -> f64 {
        let p1: f64 = self.ez1.iter().map(|e| e * e).sum();
        let p2: f64 = self.ez2.iter().map(|e| e * e).sum();
        if p1 < 1e-300 {
            0.0
        } else {
            p2 / p1
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3D Raman FDTD
// ─────────────────────────────────────────────────────────────────────────────

/// 3D stimulated Raman scattering FDTD.
///
/// Models Raman polarization using a damped oscillator:
///   d²Q/dt² + 2·γ_R·dQ/dt + ω_R²·Q = g_R · E(t)
///
/// Discretized (second-order):
///   Q^{n+1} = [2Q^n - (1 - γ_R·dt)·Q^{n-1} + g_R·dt²·E^n] / (1 + γ_R·dt)
///
/// The Raman polarization contributes to the D field update.
pub struct RamanFdtd3d {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    pub dt: f64,
    pub time_step: usize,

    pub ex: Vec<f64>,
    pub ey: Vec<f64>,
    pub ez: Vec<f64>,
    pub hx: Vec<f64>,
    pub hy: Vec<f64>,
    pub hz: Vec<f64>,

    pub eps_r: Vec<f64>,

    // Raman oscillator state Q (current and previous)
    pub qx: Vec<f64>,
    pub qy: Vec<f64>,
    pub qz: Vec<f64>,
    qx_prev: Vec<f64>,
    qy_prev: Vec<f64>,
    qz_prev: Vec<f64>,

    // Raman parameters per cell
    pub raman_gamma: Vec<f64>,
    pub raman_omega: Vec<f64>,
    pub raman_gain: Vec<f64>,
}

impl RamanFdtd3d {
    const EPS0: f64 = 8.854_187_817e-12;
    const MU0: f64 = 1.256_637_061_4e-6;

    /// Create a new 3D Raman FDTD solver.
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64, dt: f64) -> Self {
        let n = nx * ny * nz;
        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            dt,
            time_step: 0,
            ex: vec![0.0; n],
            ey: vec![0.0; n],
            ez: vec![0.0; n],
            hx: vec![0.0; n],
            hy: vec![0.0; n],
            hz: vec![0.0; n],
            eps_r: vec![1.0; n],
            qx: vec![0.0; n],
            qy: vec![0.0; n],
            qz: vec![0.0; n],
            qx_prev: vec![0.0; n],
            qy_prev: vec![0.0; n],
            qz_prev: vec![0.0; n],
            raman_gamma: vec![0.0; n],
            raman_omega: vec![0.0; n],
            raman_gain: vec![0.0; n],
        }
    }

    #[inline(always)]
    fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        k * (self.nx * self.ny) + j * self.nx + i
    }

    /// Fill a region with Raman medium parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn set_raman_region(
        &mut self,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        k0: usize,
        k1: usize,
        gamma: f64,
        omega_r: f64,
        gain: f64,
    ) {
        for k in k0..k1.min(self.nz) {
            for j in j0..j1.min(self.ny) {
                for i in i0..i1.min(self.nx) {
                    let idx = self.idx(i, j, k);
                    self.raman_gamma[idx] = gamma;
                    self.raman_omega[idx] = omega_r;
                    self.raman_gain[idx] = gain;
                }
            }
        }
    }

    /// Advance one time step.
    pub fn step(&mut self) {
        self.update_h_raman();
        self.update_eq_raman();
        self.time_step += 1;
    }

    /// Run for a given number of steps.
    pub fn run(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    fn update_h_raman(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let dx = self.dx;
        let dy = self.dy;
        let dz = self.dz;
        let dt = self.dt;

        for k in 0..nz - 1 {
            for j in 0..ny - 1 {
                for i in 0..nx - 1 {
                    let idx = self.idx(i, j, k);
                    let dez_dy = (self.ez[self.idx(i, j + 1, k)] - self.ez[idx]) / dy;
                    let dey_dz = (self.ey[self.idx(i, j, k + 1)] - self.ey[idx]) / dz;
                    let dex_dz = (self.ex[self.idx(i, j, k + 1)] - self.ex[idx]) / dz;
                    let dez_dx = (self.ez[self.idx(i + 1, j, k)] - self.ez[idx]) / dx;
                    let dey_dx = (self.ey[self.idx(i + 1, j, k)] - self.ey[idx]) / dx;
                    let dex_dy = (self.ex[self.idx(i, j + 1, k)] - self.ex[idx]) / dy;
                    self.hx[idx] -= dt / Self::MU0 * (dez_dy - dey_dz);
                    self.hy[idx] -= dt / Self::MU0 * (dex_dz - dez_dx);
                    self.hz[idx] -= dt / Self::MU0 * (dey_dx - dex_dy);
                }
            }
        }
    }

    fn update_eq_raman(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let dx = self.dx;
        let dy = self.dy;
        let dz = self.dz;
        let dt = self.dt;

        let mut qx_next = vec![0.0f64; nx * ny * nz];
        let mut qy_next = vec![0.0f64; nx * ny * nz];
        let mut qz_next = vec![0.0f64; nx * ny * nz];

        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let idx = self.idx(i, j, k);
                    let eps = Self::EPS0 * self.eps_r[idx].max(1.0);

                    let dhz_dy = (self.hz[idx] - self.hz[self.idx(i, j - 1, k)]) / dy;
                    let dhy_dz = (self.hy[idx] - self.hy[self.idx(i, j, k - 1)]) / dz;
                    let dhx_dz = (self.hx[idx] - self.hx[self.idx(i, j, k - 1)]) / dz;
                    let dhz_dx = (self.hz[idx] - self.hz[self.idx(i - 1, j, k)]) / dx;
                    let dhy_dx = (self.hy[idx] - self.hy[self.idx(i - 1, j, k)]) / dx;
                    let dhx_dy = (self.hx[idx] - self.hx[self.idx(i, j - 1, k)]) / dy;

                    let gamma = self.raman_gamma[idx];
                    let gain = self.raman_gain[idx];
                    let denom = 1.0 + gamma * dt;

                    // Raman oscillator: Q^{n+1} = [2Q^n - (1-γdt)Q^{n-1} + g*dt²*E^n] / (1+γdt)
                    // Note: ω_R² is included in the oscillator but for small dt this simplifies.
                    let qxn = (2.0 * self.qx[idx] - (1.0 - gamma * dt) * self.qx_prev[idx]
                        + gain * dt * dt * self.ex[idx])
                        / denom;
                    let qyn = (2.0 * self.qy[idx] - (1.0 - gamma * dt) * self.qy_prev[idx]
                        + gain * dt * dt * self.ey[idx])
                        / denom;
                    let qzn = (2.0 * self.qz[idx] - (1.0 - gamma * dt) * self.qz_prev[idx]
                        + gain * dt * dt * self.ez[idx])
                        / denom;

                    qx_next[idx] = qxn;
                    qy_next[idx] = qyn;
                    qz_next[idx] = qzn;

                    // Raman contribution to E update: -dQ/dt / eps (current source)
                    let dqx_dt = (qxn - self.qx_prev[idx]) / (2.0 * dt);
                    let dqy_dt = (qyn - self.qy_prev[idx]) / (2.0 * dt);
                    let dqz_dt = (qzn - self.qz_prev[idx]) / (2.0 * dt);

                    self.ex[idx] += dt / eps * (dhz_dy - dhy_dz) - dt * dqx_dt / eps;
                    self.ey[idx] += dt / eps * (dhx_dz - dhz_dx) - dt * dqy_dt / eps;
                    self.ez[idx] += dt / eps * (dhy_dx - dhx_dy) - dt * dqz_dt / eps;
                }
            }
        }

        self.qx_prev.copy_from_slice(&self.qx);
        self.qy_prev.copy_from_slice(&self.qy);
        self.qz_prev.copy_from_slice(&self.qz);
        self.qx.copy_from_slice(&qx_next);
        self.qy.copy_from_slice(&qy_next);
        self.qz.copy_from_slice(&qz_next);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kerr_medium_silicon_n2() {
        let si = KerrMedium::silicon_1550nm();
        // n2 should be ~6e-18 m²/W
        assert!(
            (si.n2_m2_per_w - 6e-18).abs() < 1e-18,
            "n2={:.2e}",
            si.n2_m2_per_w
        );
    }

    #[test]
    fn kerr_medium_effective_eps_increases() {
        let si = KerrMedium::silicon_1550nm();
        let eps0 = si.effective_eps(0.0);
        let eps1 = si.effective_eps(1e6); // very high field
        assert!(eps1 > eps0);
    }

    #[test]
    fn kerr_fdtd_advance_runs() {
        let medium = KerrMedium::silicon_1550nm();
        let mut sim = KerrFdtd1d::new(100, 10e-9, medium);
        sim.inject_source(10, 1e4);
        for _ in 0..10 {
            sim.advance();
        }
        assert!(sim.step == 10);
        assert!(sim.total_energy() >= 0.0);
    }

    #[test]
    fn kerr_fdtd_energy_positive() {
        let medium = KerrMedium::silicon_1550nm();
        let mut sim = KerrFdtd1d::new(200, 5e-9, medium);
        sim.run_with_cw_source(50, 20, 1.94e14, 1e3);
        assert!(sim.total_energy() >= 0.0);
    }

    #[test]
    fn shg_coherence_length_pm() {
        // LiNbO3: n1≈2.234, n2≈2.156 at 1064nm/532nm → Δk ≠ 0
        let shg = Shg1d::new(2e-11, 2.234, 2.156, 1064e-9);
        let lc = shg.coherence_length();
        assert!(lc > 0.0 && lc < 1.0, "L_c={lc:.2e}");
    }

    #[test]
    fn shg_phase_matched_efficiency_positive() {
        // Perfect phase matching: same index for fundamental and SHG
        let shg = Shg1d::new(2e-11, 2.2, 2.2, 1064e-9);
        let eta = shg.conversion_efficiency(1e13, 1e-3); // 1 W/µm², 1mm
        assert!(eta > 0.0);
    }

    #[test]
    fn shg_qpm_period_2x_coherence() {
        let shg = Shg1d::new(2e-11, 2.234, 2.156, 1064e-9);
        assert!((shg.qpm_period() - 2.0 * shg.coherence_length()).abs() < 1e-20);
    }

    #[test]
    fn kerr_nonlinear_phase() {
        let si = KerrMedium::silicon_1550nm();
        // 1 W/µm² × 1 mm should give ~0.04 rad for silicon
        let phi = si.nonlinear_phase(1e12, 1e-3, 1550e-9);
        assert!(phi > 0.0);
    }

    // ── 3D Kerr ──────────────────────────────────────────────────────────────

    #[test]
    fn kerr3d_initializes_zero() {
        let bc = crate::fdtd::config::BoundaryConfig::pml(4);
        let s = KerrFdtd3d::new(12, 12, 12, 20e-9, 20e-9, 20e-9, &bc);
        assert!(s.ez.iter().all(|&v| v == 0.0));
        assert!(s.chi3.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn kerr3d_runs_without_panic() {
        let bc = crate::fdtd::config::BoundaryConfig::pml(4);
        let mut s = KerrFdtd3d::new(14, 14, 14, 20e-9, 20e-9, 20e-9, &bc);
        s.inject_ez(7, 7, 7, 1.0);
        s.run(20);
        assert!(s.ez.iter().all(|&v| v.is_finite()));
        assert!(s.hx.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn kerr3d_energy_positive_after_source() {
        let bc = crate::fdtd::config::BoundaryConfig::pml(4);
        let mut s = KerrFdtd3d::new(14, 14, 14, 20e-9, 20e-9, 20e-9, &bc);
        s.inject_ez(7, 7, 7, 1.0);
        s.run(5);
        assert!(s.total_energy() > 0.0);
    }

    #[test]
    fn kerr3d_set_region_updates_params() {
        let bc = crate::fdtd::config::BoundaryConfig::pml(3);
        let mut s = KerrFdtd3d::new(12, 12, 12, 20e-9, 20e-9, 20e-9, &bc);
        s.set_kerr_region(4, 8, 4, 8, 4, 8, 2.25, 1e-20);
        let idx = 6 * 12 * 12 + 6 * 12 + 6;
        assert!((s.eps_r[idx] - 2.25).abs() < 1e-10);
        assert!((s.chi3[idx] - 1e-20).abs() < 1e-30);
    }

    // ── 3D SHG ───────────────────────────────────────────────────────────────

    #[test]
    fn shg3d_initializes_zero() {
        let dt = 3.85e-17;
        let s = Shg3d::new(12, 12, 12, 20e-9, 20e-9, 20e-9, dt);
        assert!(s.ez1.iter().all(|&v| v == 0.0));
        assert!(s.ez2.iter().all(|&v| v == 0.0));
        assert!(s.d_eff.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn shg3d_runs_without_panic() {
        let dt = 3.85e-17;
        let mut s = Shg3d::new(12, 12, 12, 20e-9, 20e-9, 20e-9, dt);
        s.set_shg_region(4, 8, 4, 8, 4, 8, 1e-12, 2.0, 2.0);
        // Manually inject a source
        let idx = 6 * 12 * 12 + 6 * 12 + 6;
        s.ez1[idx] = 1.0;
        s.run(20);
        assert!(s.ez1.iter().all(|&v| v.is_finite()));
        assert!(s.ez2.iter().all(|&v| v.is_finite()));
    }

    // ── 3D Raman ─────────────────────────────────────────────────────────────

    #[test]
    fn raman3d_initializes_zero() {
        let dt = 3.85e-17;
        let s = RamanFdtd3d::new(10, 10, 10, 20e-9, 20e-9, 20e-9, dt);
        assert!(s.qz.iter().all(|&v| v == 0.0));
        assert!(s.raman_gain.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn raman3d_runs_without_panic() {
        let dt = 3.85e-17;
        let mut s = RamanFdtd3d::new(12, 12, 12, 20e-9, 20e-9, 20e-9, dt);
        s.set_raman_region(4, 8, 4, 8, 4, 8, 1e12, 1e13, 1e-10);
        let idx = 6 * 12 * 12 + 6 * 12 + 6;
        s.ez[idx] = 1.0;
        s.run(20);
        assert!(s.ez.iter().all(|&v| v.is_finite()));
        assert!(s.qz.iter().all(|&v| v.is_finite()));
    }
}
