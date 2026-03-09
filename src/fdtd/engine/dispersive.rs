use crate::fdtd::boundary::pml::Cpml;
use crate::fdtd::config::BoundaryConfig;
use crate::fdtd::config::{Dimensions, GridSpacing};
use crate::fdtd::courant::courant_dt;
use crate::units::conversion::{EPSILON_0, MU_0};

/// Drude-model parameters for a dispersive metal.
///
/// ε(ω) = ε_inf - ωp² / (ω² + i·γ·ω)
///
/// ADE polarization equation: d²P/dt² + γ·dP/dt = ε₀·ωp²·E
#[derive(Debug, Clone, Copy)]
pub struct DrudeParams {
    /// High-frequency permittivity (ε_∞)
    pub eps_inf: f64,
    /// Plasma frequency (rad/s)
    pub omega_p: f64,
    /// Collision frequency / damping (rad/s)
    pub gamma: f64,
}

impl DrudeParams {
    /// Gold parameters (approximate, valid near visible)
    pub fn gold() -> Self {
        Self {
            eps_inf: 9.54,
            omega_p: 1.367e16,
            gamma: 1.224e14,
        }
    }

    /// Silver parameters (approximate)
    pub fn silver() -> Self {
        Self {
            eps_inf: 3.7,
            omega_p: 1.38e16,
            gamma: 2.73e13,
        }
    }
}

/// 1D dispersive FDTD using Auxiliary Differential Equations (ADE) for a Drude material.
///
/// Solves TEM (Ex, Hy) with a Drude polarization current:
///   d²P/dt² + γ·dP/dt = ε₀·ωp²·Ex
///
/// Discretized (second-order accurate):
///   P^{n+1} = [2P^n - (1 - γ·dt/2)P^{n-1} + ε₀·ωp²·dt²·E^n] / (1 + γ·dt/2)
///
/// Modified E update:
///   E^{n+1} = E^n + dt/(ε₀·ε_∞·dz)·ΔHy - (P^{n+1} - P^{n-1})/(2·ε₀·ε_∞)
pub struct Fdtd1dDrude {
    pub nz: usize,
    pub dz: f64,
    pub dt: f64,
    pub time_step: usize,

    pub ex: Vec<f64>,
    pub hy: Vec<f64>,

    /// Current polarization P^n
    pub px: Vec<f64>,
    /// Previous polarization P^{n-1}
    px_prev: Vec<f64>,

    /// Relative background permittivity (ε_∞) at each cell
    pub eps_inf: Vec<f64>,
    /// Drude params at each cell (None = vacuum)
    drude: Vec<Option<DrudeParams>>,

    pml: Cpml,
    psi_hy: Vec<f64>,
    psi_ex: Vec<f64>,
}

impl Fdtd1dDrude {
    pub fn new(nz: usize, dz: f64, boundary: &BoundaryConfig) -> Self {
        let dt = 0.99
            * courant_dt(
                Dimensions::OneD { nz },
                GridSpacing { dx: dz, dy: dz, dz },
                1.0,
            );
        let pml = Cpml::new(
            nz,
            boundary.pml_cells,
            dz,
            dt,
            boundary.pml_m,
            boundary.pml_r0,
        );
        Self {
            nz,
            dz,
            dt,
            time_step: 0,
            ex: vec![0.0; nz],
            hy: vec![0.0; nz],
            px: vec![0.0; nz],
            px_prev: vec![0.0; nz],
            eps_inf: vec![1.0; nz],
            drude: vec![None; nz],
            pml,
            psi_hy: vec![0.0; nz],
            psi_ex: vec![0.0; nz],
        }
    }

    /// Fill a slab [z_start, z_end) with a Drude material.
    pub fn fill_drude(&mut self, z_start: f64, z_end: f64, params: DrudeParams) {
        let i0 = (z_start / self.dz).floor() as usize;
        let i1 = ((z_end / self.dz).ceil() as usize).min(self.nz);
        for i in i0..i1 {
            self.eps_inf[i] = params.eps_inf;
            self.drude[i] = Some(params);
        }
    }

    pub fn current_time(&self) -> f64 {
        self.time_step as f64 * self.dt
    }

    /// Hard-source injection into Ex at position `pos`.
    pub fn inject_ex(&mut self, pos: usize, val: f64) {
        if pos < self.nz {
            self.ex[pos] += val;
        }
    }

    pub fn step(&mut self) {
        let nz = self.nz;
        let dz = self.dz;
        let dt = self.dt;

        // --- Update Hy (n → n+½) ---
        for i in 0..nz - 1 {
            let dex = self.ex[i + 1] - self.ex[i];
            self.psi_hy[i] = self.pml.b_h[i] * self.psi_hy[i] + self.pml.c_h[i] * dex / dz;
            let kappa = self.pml.kappa_h[i];
            self.hy[i] -= dt / MU_0 * (dex / (kappa * dz) + self.psi_hy[i]);
        }

        // --- Update polarization P^{n+1} and Ex^{n+1} ---
        let mut px_next = vec![0.0f64; nz];
        #[allow(clippy::needless_range_loop)]
        for i in 1..nz - 1 {
            let dhy = self.hy[i] - self.hy[i - 1];
            self.psi_ex[i] = self.pml.b_e[i] * self.psi_ex[i] + self.pml.c_e[i] * dhy / dz;
            let kappa = self.pml.kappa_e[i];
            let eps_inf = self.eps_inf[i];

            if let Some(d) = self.drude[i] {
                // ADE update for Drude polarization
                let a1 = 1.0 + d.gamma * dt / 2.0;
                let a2 = 1.0 - d.gamma * dt / 2.0;
                let f = EPSILON_0 * d.omega_p * d.omega_p * dt * dt;
                px_next[i] = (2.0 * self.px[i] - a2 * self.px_prev[i] + f * self.ex[i]) / a1;

                // Modified E update including polarization
                // ε₀·ε_∞·∂Ex/∂t = ∂Hy/∂z - ∂Px/∂t
                // Using: ∂Px/∂t ≈ (Px^{n+1} - Px^{n-1}) / (2dt)
                let dpx_dt = (px_next[i] - self.px_prev[i]) / (2.0 * dt);
                self.ex[i] += dt / (EPSILON_0 * eps_inf) * (dhy / (kappa * dz) + self.psi_ex[i])
                    - dpx_dt / (EPSILON_0 * eps_inf) * dt;
            } else {
                // Standard non-dispersive update
                self.ex[i] -= dt / (EPSILON_0 * eps_inf) * (dhy / (kappa * dz) + self.psi_ex[i]);
                px_next[i] = 0.0;
            }
        }
        self.ex[0] = 0.0;
        self.ex[nz - 1] = 0.0;

        // Advance polarization buffers
        self.px_prev.copy_from_slice(&self.px);
        self.px.copy_from_slice(&px_next);

        self.time_step += 1;
    }

    pub fn run(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }
}

/// Lorentz oscillator parameters for dispersive dielectric.
///
/// ε(ω) = ε_∞ + Σ_p  Δεp·ω₀p² / (ω₀p² - ω² - i·δp·ω)
#[derive(Debug, Clone)]
pub struct LorentzParams {
    pub eps_inf: f64,
    /// List of (delta_eps, omega_0, delta) for each oscillator
    pub oscillators: Vec<(f64, f64, f64)>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Precomputed ADE coefficients for 3D Drude (per-cell)
// ─────────────────────────────────────────────────────────────────────────────

/// Precomputed update coefficients for 3D Drude ADE (per cell).
///
/// Avoids recomputing expensive divisions inside the hot loop.
pub struct AdeCoeffs3d {
    /// ca = 1/(1 + sigma*dt/(2*eps0*eps_inf))  (no conductivity here → 1.0)
    pub ca: Vec<f64>,
    /// cb = dt / (eps0 * eps_inf)  for curl contribution
    pub cb: Vec<f64>,
    /// a1 = 1 / (1 + gamma*dt/2)
    pub a1: Vec<f64>,
    /// a2 = (1 - gamma*dt/2) / (1 + gamma*dt/2)
    pub a2: Vec<f64>,
    /// f_drude = eps0 * omega_p^2 * dt^2 / (1 + gamma*dt/2)
    pub f_drude: Vec<f64>,
}

impl AdeCoeffs3d {
    /// Build ADE coefficients from per-cell arrays and dt.
    pub fn for_drude(n: usize, eps_inf: &[f64], drude: &[Option<DrudeParams>], dt: f64) -> Self {
        let mut ca = vec![1.0f64; n];
        let mut cb = vec![0.0f64; n];
        let mut a1 = vec![1.0f64; n];
        let mut a2 = vec![0.0f64; n];
        let mut f_drude = vec![0.0f64; n];

        for idx in 0..n {
            let ei = eps_inf[idx].max(1.0);
            // ca is 1 here (no separate conductivity σ in Drude E-update)
            ca[idx] = 1.0;
            cb[idx] = dt / (EPSILON_0 * ei);

            if let Some(d) = drude[idx] {
                let half_gdt = d.gamma * dt * 0.5;
                let denom = 1.0 + half_gdt;
                a1[idx] = 1.0 / denom;
                a2[idx] = (1.0 - half_gdt) / denom;
                f_drude[idx] = EPSILON_0 * d.omega_p * d.omega_p * dt * dt / denom;
            }
        }

        Self {
            ca,
            cb,
            a1,
            a2,
            f_drude,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3D Drude ADE FDTD
// ─────────────────────────────────────────────────────────────────────────────

/// 3D FDTD with Drude dispersive material using ADE method.
///
/// Supports mixed cells: some vacuum, some Drude metal.
/// All 6 field components (Ex,Ey,Ez,Hx,Hy,Hz) updated with full vector Maxwell.
/// Drude polarization P is maintained per-cell per-component (Px, Py, Pz).
pub struct Fdtd3dDrude {
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

    /// Background permittivity ε_∞ per cell
    pub eps_inf: Vec<f64>,
    pub mu_r: Vec<f64>,

    /// Drude parameters per cell (None = vacuum)
    pub drude: Vec<Option<DrudeParams>>,

    /// Polarization components (current)
    pub px: Vec<f64>,
    pub py: Vec<f64>,
    pub pz: Vec<f64>,
    /// Polarization components (previous step)
    px_prev: Vec<f64>,
    py_prev: Vec<f64>,
    pz_prev: Vec<f64>,

    // CPML per axis
    pml_x: Cpml,
    pml_y: Cpml,
    pml_z: Cpml,

    // 12 CPML psi arrays
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

impl Fdtd3dDrude {
    /// Create a new 3D Drude FDTD solver.
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        boundary: &BoundaryConfig,
    ) -> Self {
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
            eps_inf: vec![1.0; n],
            mu_r: vec![1.0; n],
            drude: vec![None; n],
            px: vec![0.0; n],
            py: vec![0.0; n],
            pz: vec![0.0; n],
            px_prev: vec![0.0; n],
            py_prev: vec![0.0; n],
            pz_prev: vec![0.0; n],
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

    /// Fill a rectangular box with Drude material.
    #[allow(clippy::too_many_arguments)]
    pub fn fill_drude_box(
        &mut self,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        k0: usize,
        k1: usize,
        params: DrudeParams,
    ) {
        for k in k0..k1.min(self.nz) {
            for j in j0..j1.min(self.ny) {
                for i in i0..i1.min(self.nx) {
                    let idx = self.idx(i, j, k);
                    self.eps_inf[idx] = params.eps_inf;
                    self.drude[idx] = Some(params);
                }
            }
        }
    }

    /// Hard-source injection into Ez at (i,j,k).
    pub fn inject_ez(&mut self, i: usize, j: usize, k: usize, val: f64) {
        if i < self.nx && j < self.ny && k < self.nz {
            let idx = self.idx(i, j, k);
            self.ez[idx] += val;
        }
    }

    /// Hard-source injection into Ex at (i,j,k).
    pub fn inject_ex(&mut self, i: usize, j: usize, k: usize, val: f64) {
        if i < self.nx && j < self.ny && k < self.nz {
            let idx = self.idx(i, j, k);
            self.ex[idx] += val;
        }
    }

    /// Hard-source injection into Ey at (i,j,k).
    pub fn inject_ey(&mut self, i: usize, j: usize, k: usize, val: f64) {
        if i < self.nx && j < self.ny && k < self.nz {
            let idx = self.idx(i, j, k);
            self.ey[idx] += val;
        }
    }

    /// Advance one full time step: H update → P update → E update.
    pub fn step(&mut self) {
        self.update_h_drude();
        self.update_ep_drude();
        self.time_step += 1;
    }

    /// Run for a given number of steps.
    pub fn run(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// Current simulation time (s).
    pub fn current_time(&self) -> f64 {
        self.time_step as f64 * self.dt
    }

    /// Total electromagnetic energy (J) in the domain.
    pub fn total_energy(&self) -> f64 {
        let dv = self.dx * self.dy * self.dz;
        use crate::units::conversion::MU_0;
        let e_energy: f64 = {
            let ex: f64 = self
                .ex
                .iter()
                .zip(self.eps_inf.iter())
                .map(|(e, &ei)| ei * e * e)
                .sum();
            let ey: f64 = self
                .ey
                .iter()
                .zip(self.eps_inf.iter())
                .map(|(e, &ei)| ei * e * e)
                .sum();
            let ez: f64 = self
                .ez
                .iter()
                .zip(self.eps_inf.iter())
                .map(|(e, &ei)| ei * e * e)
                .sum();
            (ex + ey + ez) * 0.5 * EPSILON_0
        };
        let h_energy: f64 = {
            let hx: f64 = self
                .hx
                .iter()
                .zip(self.mu_r.iter())
                .map(|(h, &mu)| mu * h * h)
                .sum();
            let hy: f64 = self
                .hy
                .iter()
                .zip(self.mu_r.iter())
                .map(|(h, &mu)| mu * h * h)
                .sum();
            let hz: f64 = self
                .hz
                .iter()
                .zip(self.mu_r.iter())
                .map(|(h, &mu)| mu * h * h)
                .sum();
            (hx + hy + hz) * 0.5 * MU_0
        };
        (e_energy + h_energy) * dv
    }

    /// Peak |Ez| field value.
    pub fn peak_ez(&self) -> f64 {
        self.ez.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
    }

    fn update_h_drude(&mut self) {
        use crate::units::conversion::MU_0;
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
                    let mu = MU_0 * self.mu_r[idx];

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

                    self.hx[idx] -= dt / mu
                        * (dez_dy / ky + self.psi_hx_y[idx] - dey_dz / kz - self.psi_hx_z[idx]);
                    self.hy[idx] -= dt / mu
                        * (dex_dz / kz + self.psi_hy_z[idx] - dez_dx / kx - self.psi_hy_x[idx]);
                    self.hz[idx] -= dt / mu
                        * (dey_dx / kx + self.psi_hz_x[idx] - dex_dy / ky - self.psi_hz_y[idx]);
                }
            }
        }
    }

    fn update_ep_drude(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let dx = self.dx;
        let dy = self.dy;
        let dz = self.dz;
        let dt = self.dt;

        // Allocate next polarization buffers
        let mut px_next = vec![0.0f64; nx * ny * nz];
        let mut py_next = vec![0.0f64; nx * ny * nz];
        let mut pz_next = vec![0.0f64; nx * ny * nz];

        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let idx = self.idx(i, j, k);
                    let eps_inf = self.eps_inf[idx].max(1.0);
                    let eps = EPSILON_0 * eps_inf;

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

                    let curl_hx =
                        dhz_dy / ky + self.psi_ex_y[idx] - dhy_dz / kz - self.psi_ex_z[idx];
                    let curl_hy =
                        dhx_dz / kz + self.psi_ey_z[idx] - dhz_dx / kx - self.psi_ey_x[idx];
                    let curl_hz =
                        dhy_dx / kx + self.psi_ez_x[idx] - dhx_dy / ky - self.psi_ez_y[idx];

                    if let Some(d) = self.drude[idx] {
                        // ADE for Drude polarization
                        // P^{n+1} = [2P^n - (1 - γdt/2)P^{n-1} + ε₀ωp²dt²E^n] / (1 + γdt/2)
                        let half_gdt = d.gamma * dt * 0.5;
                        let denom = 1.0 + half_gdt;
                        let a2_fac = 1.0 - half_gdt;
                        let f_p = EPSILON_0 * d.omega_p * d.omega_p * dt * dt;

                        let pxn = (2.0 * self.px[idx] - a2_fac * self.px_prev[idx]
                            + f_p * self.ex[idx])
                            / denom;
                        let pyn = (2.0 * self.py[idx] - a2_fac * self.py_prev[idx]
                            + f_p * self.ey[idx])
                            / denom;
                        let pzn = (2.0 * self.pz[idx] - a2_fac * self.pz_prev[idx]
                            + f_p * self.ez[idx])
                            / denom;

                        px_next[idx] = pxn;
                        py_next[idx] = pyn;
                        pz_next[idx] = pzn;

                        // E update: ε₀ε_∞∂E/∂t = curl_H - ∂P/∂t
                        // ∂P/∂t ≈ (P^{n+1} - P^{n-1}) / (2dt)
                        let dpx_dt = (pxn - self.px_prev[idx]) / (2.0 * dt);
                        let dpy_dt = (pyn - self.py_prev[idx]) / (2.0 * dt);
                        let dpz_dt = (pzn - self.pz_prev[idx]) / (2.0 * dt);

                        self.ex[idx] += dt / eps * curl_hx - dt * dpx_dt / eps;
                        self.ey[idx] += dt / eps * curl_hy - dt * dpy_dt / eps;
                        self.ez[idx] += dt / eps * curl_hz - dt * dpz_dt / eps;
                    } else {
                        // Vacuum / non-dispersive
                        self.ex[idx] += dt / eps * curl_hx;
                        self.ey[idx] += dt / eps * curl_hy;
                        self.ez[idx] += dt / eps * curl_hz;
                    }
                }
            }
        }

        // Advance polarization buffers
        self.px_prev.copy_from_slice(&self.px);
        self.py_prev.copy_from_slice(&self.py);
        self.pz_prev.copy_from_slice(&self.pz);
        self.px.copy_from_slice(&px_next);
        self.py.copy_from_slice(&py_next);
        self.pz.copy_from_slice(&pz_next);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3D Lorentz ADE FDTD
// ─────────────────────────────────────────────────────────────────────────────

/// 3D FDTD with Lorentz oscillator dispersive material (ADE method).
///
/// Single-pole implementation; extend `oscillators` for multi-pole.
pub struct Fdtd3dLorentz {
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

    /// Background permittivity ε_∞ per cell
    pub eps_inf: Vec<f64>,

    /// Lorentz params per cell (None = vacuum); uses first oscillator only for ADE update.
    lorentz: Vec<Option<LorentzParams>>,

    // Polarization for first oscillator
    pub px: Vec<f64>,
    pub py: Vec<f64>,
    pub pz: Vec<f64>,
    pub px_prev: Vec<f64>,
    pub py_prev: Vec<f64>,
    pub pz_prev: Vec<f64>,

    pml_x: Cpml,
    pml_y: Cpml,
    pml_z: Cpml,

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

impl Fdtd3dLorentz {
    /// Create a new 3D Lorentz FDTD solver.
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        boundary: &BoundaryConfig,
    ) -> Self {
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
            eps_inf: vec![1.0; n],
            lorentz: vec![None; n],
            px: vec![0.0; n],
            py: vec![0.0; n],
            pz: vec![0.0; n],
            px_prev: vec![0.0; n],
            py_prev: vec![0.0; n],
            pz_prev: vec![0.0; n],
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

    /// Fill a rectangular box with Lorentz dispersive material.
    #[allow(clippy::too_many_arguments)]
    pub fn fill_lorentz_box(
        &mut self,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        k0: usize,
        k1: usize,
        params: LorentzParams,
    ) {
        for k in k0..k1.min(self.nz) {
            for j in j0..j1.min(self.ny) {
                for i in i0..i1.min(self.nx) {
                    let idx = self.idx(i, j, k);
                    self.eps_inf[idx] = params.eps_inf;
                    self.lorentz[idx] = Some(params.clone());
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

    /// Hard-source injection into Ex.
    pub fn inject_ex(&mut self, i: usize, j: usize, k: usize, val: f64) {
        if i < self.nx && j < self.ny && k < self.nz {
            let idx = self.idx(i, j, k);
            self.ex[idx] += val;
        }
    }

    /// Hard-source injection into Ey.
    pub fn inject_ey(&mut self, i: usize, j: usize, k: usize, val: f64) {
        if i < self.nx && j < self.ny && k < self.nz {
            let idx = self.idx(i, j, k);
            self.ey[idx] += val;
        }
    }

    /// Advance one full time step.
    pub fn step(&mut self) {
        self.update_h_lorentz();
        self.update_ep_lorentz();
        self.time_step += 1;
    }

    /// Run for a given number of steps.
    pub fn run(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// Current simulation time (s).
    pub fn current_time(&self) -> f64 {
        self.time_step as f64 * self.dt
    }

    /// Total electromagnetic energy (J).
    pub fn total_energy(&self) -> f64 {
        use crate::units::conversion::MU_0;
        let dv = self.dx * self.dy * self.dz;
        let e_energy: f64 = {
            let ex: f64 = self
                .ex
                .iter()
                .zip(self.eps_inf.iter())
                .map(|(e, &ei)| ei * e * e)
                .sum();
            let ey: f64 = self
                .ey
                .iter()
                .zip(self.eps_inf.iter())
                .map(|(e, &ei)| ei * e * e)
                .sum();
            let ez: f64 = self
                .ez
                .iter()
                .zip(self.eps_inf.iter())
                .map(|(e, &ei)| ei * e * e)
                .sum();
            (ex + ey + ez) * 0.5 * EPSILON_0
        };
        let h_energy: f64 = {
            let hx: f64 = self.hx.iter().map(|h| h * h).sum();
            let hy: f64 = self.hy.iter().map(|h| h * h).sum();
            let hz: f64 = self.hz.iter().map(|h| h * h).sum();
            (hx + hy + hz) * 0.5 * MU_0
        };
        (e_energy + h_energy) * dv
    }

    /// Peak |Ez| field value.
    pub fn peak_ez(&self) -> f64 {
        self.ez.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
    }

    fn update_h_lorentz(&mut self) {
        use crate::units::conversion::MU_0;
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
                    let mu = MU_0;

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

                    self.hx[idx] -= dt / mu
                        * (dez_dy / ky + self.psi_hx_y[idx] - dey_dz / kz - self.psi_hx_z[idx]);
                    self.hy[idx] -= dt / mu
                        * (dex_dz / kz + self.psi_hy_z[idx] - dez_dx / kx - self.psi_hy_x[idx]);
                    self.hz[idx] -= dt / mu
                        * (dey_dx / kx + self.psi_hz_x[idx] - dex_dy / ky - self.psi_hz_y[idx]);
                }
            }
        }
    }

    fn update_ep_lorentz(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let dx = self.dx;
        let dy = self.dy;
        let dz = self.dz;
        let dt = self.dt;

        let mut px_next = vec![0.0f64; nx * ny * nz];
        let mut py_next = vec![0.0f64; nx * ny * nz];
        let mut pz_next = vec![0.0f64; nx * ny * nz];

        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let idx = self.idx(i, j, k);
                    let eps_inf = self.eps_inf[idx].max(1.0);
                    let eps = EPSILON_0 * eps_inf;

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

                    let curl_hx =
                        dhz_dy / ky + self.psi_ex_y[idx] - dhy_dz / kz - self.psi_ex_z[idx];
                    let curl_hy =
                        dhx_dz / kz + self.psi_ey_z[idx] - dhz_dx / kx - self.psi_ey_x[idx];
                    let curl_hz =
                        dhy_dx / kx + self.psi_ez_x[idx] - dhx_dy / ky - self.psi_ez_y[idx];

                    if let Some(lor) = &self.lorentz[idx] {
                        // Use first oscillator for ADE
                        // P^{n+1} = [(2 - ω₀²dt²)P^n - (1 - δdt/2)P^{n-1} + ε₀Δε ω₀²dt²E^n]
                        //           / (1 + δdt/2)
                        let (delta_eps, omega_0, delta) = if lor.oscillators.is_empty() {
                            (0.0_f64, 1.0_f64, 0.0_f64)
                        } else {
                            lor.oscillators[0]
                        };
                        let o2dt2 = omega_0 * omega_0 * dt * dt;
                        let half_ddt = delta * dt * 0.5;
                        let denom = 1.0 + half_ddt;
                        let f_lor = EPSILON_0 * delta_eps * o2dt2;

                        let pxn = ((2.0 - o2dt2) * self.px[idx]
                            - (1.0 - half_ddt) * self.px_prev[idx]
                            + f_lor * self.ex[idx])
                            / denom;
                        let pyn = ((2.0 - o2dt2) * self.py[idx]
                            - (1.0 - half_ddt) * self.py_prev[idx]
                            + f_lor * self.ey[idx])
                            / denom;
                        let pzn = ((2.0 - o2dt2) * self.pz[idx]
                            - (1.0 - half_ddt) * self.pz_prev[idx]
                            + f_lor * self.ez[idx])
                            / denom;

                        px_next[idx] = pxn;
                        py_next[idx] = pyn;
                        pz_next[idx] = pzn;

                        let dpx_dt = (pxn - self.px_prev[idx]) / (2.0 * dt);
                        let dpy_dt = (pyn - self.py_prev[idx]) / (2.0 * dt);
                        let dpz_dt = (pzn - self.pz_prev[idx]) / (2.0 * dt);

                        self.ex[idx] += dt / eps * curl_hx - dt * dpx_dt / eps;
                        self.ey[idx] += dt / eps * curl_hy - dt * dpy_dt / eps;
                        self.ez[idx] += dt / eps * curl_hz - dt * dpz_dt / eps;
                    } else {
                        self.ex[idx] += dt / eps * curl_hx;
                        self.ey[idx] += dt / eps * curl_hy;
                        self.ez[idx] += dt / eps * curl_hz;
                    }
                }
            }
        }

        self.px_prev.copy_from_slice(&self.px);
        self.py_prev.copy_from_slice(&self.py);
        self.pz_prev.copy_from_slice(&self.pz);
        self.px.copy_from_slice(&px_next);
        self.py.copy_from_slice(&py_next);
        self.pz.copy_from_slice(&pz_next);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drude_solver_initializes() {
        let solver = Fdtd1dDrude::new(200, 5e-9, &BoundaryConfig::pml(20));
        assert!(solver.ex.iter().all(|&v| v == 0.0));
        assert!(solver.hy.iter().all(|&v| v == 0.0));
        assert!(solver.px.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn drude_solver_runs_vacuum() {
        let mut s = Fdtd1dDrude::new(200, 5e-9, &BoundaryConfig::pml(20));
        // Inject a pulse
        for step in 0..100 {
            let t = step as f64 * s.dt;
            let amp = (-(t - 20.0 * s.dt).powi(2) / (2.0 * (5.0 * s.dt).powi(2))).exp();
            s.inject_ex(40, amp);
            s.step();
        }
        assert!(s.ex.iter().all(|&v| v.is_finite()));
        assert!(s.hy.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn drude_gold_fills_region() {
        let mut s = Fdtd1dDrude::new(200, 5e-9, &BoundaryConfig::pml(20));
        let gold = DrudeParams::gold();
        s.fill_drude(500e-9, 600e-9, gold);
        // Check that cells in that range have Drude params
        let i = (550e-9 / 5e-9) as usize;
        assert!(s.drude[i].is_some());
        assert!((s.eps_inf[i] - 9.54).abs() < 1e-6);
    }

    #[test]
    fn drude_solver_runs_with_metal() {
        let mut s = Fdtd1dDrude::new(300, 5e-9, &BoundaryConfig::pml(20));
        s.fill_drude(500e-9, 600e-9, DrudeParams::gold());
        for step in 0..200 {
            let t = step as f64 * s.dt;
            let amp = (-(t - 30.0 * s.dt).powi(2) / (2.0 * (8.0 * s.dt).powi(2))).exp();
            s.inject_ex(50, amp);
            s.step();
        }
        // Fields should remain finite (not blow up)
        assert!(
            s.ex.iter().all(|&v| v.is_finite()),
            "Ex has non-finite values"
        );
        assert!(
            s.hy.iter().all(|&v| v.is_finite()),
            "Hy has non-finite values"
        );
    }

    #[test]
    fn drude_params_gold_has_physical_values() {
        let g = DrudeParams::gold();
        // omega_p ~ 1e16 rad/s for noble metals
        assert!(g.omega_p > 1e15 && g.omega_p < 1e17);
        // eps_inf > 1 for gold background
        assert!(g.eps_inf > 1.0);
        // gamma << omega_p (low damping)
        assert!(g.gamma < g.omega_p);
    }

    #[test]
    fn fdtd3d_drude_initializes_zero() {
        let s = Fdtd3dDrude::new(16, 16, 16, 20e-9, 20e-9, 20e-9, &BoundaryConfig::pml(4));
        assert!(s.ez.iter().all(|&v| v == 0.0));
        assert!(s.px.iter().all(|&v| v == 0.0));
        assert_eq!(s.drude.iter().filter(|d| d.is_some()).count(), 0);
    }

    #[test]
    fn fdtd3d_drude_runs_vacuum() {
        let mut s = Fdtd3dDrude::new(16, 16, 16, 20e-9, 20e-9, 20e-9, &BoundaryConfig::pml(4));
        let dt = s.dt;
        for step in 0..20 {
            let t = step as f64 * dt;
            let amp = (-(t - 5.0 * dt).powi(2) / (2.0 * (2.0 * dt).powi(2))).exp();
            s.inject_ez(8, 8, 8, amp);
            s.step();
        }
        assert!(s.ez.iter().all(|&v| v.is_finite()));
        assert!(s.hx.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn fdtd3d_drude_fill_box_sets_material() {
        let mut s = Fdtd3dDrude::new(20, 20, 20, 10e-9, 10e-9, 10e-9, &BoundaryConfig::pml(4));
        s.fill_drude_box(5, 15, 5, 15, 5, 15, DrudeParams::gold());
        let idx = s.idx(10, 10, 10);
        assert!(s.drude[idx].is_some());
        assert!((s.eps_inf[idx] - 9.54).abs() < 1e-6);
        let idx2 = s.idx(2, 2, 2);
        assert!(s.drude[idx2].is_none());
    }

    #[test]
    fn fdtd3d_drude_energy_positive_after_source() {
        let mut s = Fdtd3dDrude::new(16, 16, 16, 20e-9, 20e-9, 20e-9, &BoundaryConfig::pml(4));
        s.inject_ez(8, 8, 8, 1.0);
        s.run(5);
        assert!(s.total_energy() > 0.0);
    }

    #[test]
    fn fdtd3d_drude_ade_coeffs_correctness() {
        let n = 4;
        let eps_inf = vec![9.54f64; n];
        let drude = vec![Some(DrudeParams::gold()); n];
        let dt = 1e-17;
        let coeffs = AdeCoeffs3d::for_drude(n, &eps_inf, &drude, dt);
        for i in 0..n {
            // a1 must be positive and < 1 (since gamma > 0)
            assert!(coeffs.a1[i] > 0.0 && coeffs.a1[i] < 1.0);
            // f_drude must be positive
            assert!(coeffs.f_drude[i] > 0.0);
            // cb = dt / (eps0 * eps_inf)
            let expected_cb = dt / (crate::units::conversion::EPSILON_0 * 9.54);
            assert!((coeffs.cb[i] - expected_cb).abs() < 1e-30);
        }
    }

    #[test]
    fn fdtd3d_lorentz_initializes_zero() {
        let s = Fdtd3dLorentz::new(14, 14, 14, 20e-9, 20e-9, 20e-9, &BoundaryConfig::pml(4));
        assert!(s.ez.iter().all(|&v| v == 0.0));
        assert!(s.px.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn fdtd3d_lorentz_runs_vacuum() {
        let mut s = Fdtd3dLorentz::new(14, 14, 14, 20e-9, 20e-9, 20e-9, &BoundaryConfig::pml(4));
        let dt = s.dt;
        for step in 0..20 {
            let t = step as f64 * dt;
            let amp = (-(t - 5.0 * dt).powi(2) / (2.0 * (2.0 * dt).powi(2))).exp();
            s.inject_ez(7, 7, 7, amp);
            s.step();
        }
        assert!(s.ez.iter().all(|&v| v.is_finite()));
        assert!(s.hx.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn fdtd3d_lorentz_fill_box_sets_material() {
        let mut s = Fdtd3dLorentz::new(20, 20, 20, 10e-9, 10e-9, 10e-9, &BoundaryConfig::pml(4));
        let lor = LorentzParams {
            eps_inf: 2.25,
            oscillators: vec![(1.0, 2.0e14, 1.0e12)],
        };
        s.fill_lorentz_box(5, 15, 5, 15, 5, 15, lor);
        let idx = s.idx(10, 10, 10);
        assert!(s.lorentz[idx].is_some());
        assert!((s.eps_inf[idx] - 2.25).abs() < 1e-6);
    }
}
