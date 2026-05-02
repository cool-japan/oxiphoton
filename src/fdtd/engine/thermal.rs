//! Thermal FDTD engine: thermo-optic coupling and heat equation solver.
//!
//! # Physics
//!
//! Optical properties depend on temperature through the thermo-optic effect:
//!
//!   n(T) = n₀ + (dn/dT) · ΔT
//!   ε(T) = n(T)²
//!
//! Thermal transport is governed by the parabolic heat equation:
//!
//!   ∂T/∂t = α ∇²T + Q/(ρ c_p)
//!
//! where α is the thermal diffusivity (m²/s), Q is a volumetric heat source
//! (W/m³, e.g. Joule heating σ|E|²), and ρ c_p is the heat capacity per volume.
//!
//! # Coupling strategy
//!
//! [`ThermoOpticCoupler`] alternates between electromagnetic FDTD steps (fast,
//! optical time scale ~fs) and thermal steps (slow, thermal time scale ~ns–μs).
//! After every `optical_steps_per_thermal` EM steps one thermal step is taken and
//! the permittivity array is recomputed, feeding back into the EM solver.

use crate::error::OxiPhotonError;

// ─── ThermalFdtd3d ───────────────────────────────────────────────────────────

/// Temperature-aware permittivity grid for 3D FDTD.
///
/// Stores a temperature field and thermo-optic coefficients (dn/dT) per cell.
/// The effective permittivity at each cell is recomputed from the current
/// temperature whenever \[`apply_temperature_field`\] is called.
///
/// # Field layout
/// All arrays have length `nx * ny * nz`; index mapping is
///   `idx(i, j, k) = i * ny * nz + j * nz + k`.
#[derive(Debug, Clone)]
pub struct ThermalFdtd3d {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64, // m
    pub dy: f64, // m
    pub dz: f64, // m

    /// Temperature field T\[i,j,k\] in Kelvin.
    pub temperature: Vec<f64>,

    /// Reference (base) temperature in Kelvin.
    pub base_temperature: f64,

    /// Thermo-optic coefficient dn/dT \[1/K\] per cell.
    pub dn_dt: Vec<f64>,

    /// Base (T = T_base) relative permittivity per cell.
    pub eps_base: Vec<f64>,

    /// Current (temperature-corrected) relative permittivity per cell.
    pub eps_current: Vec<f64>,
}

impl ThermalFdtd3d {
    /// Create a new thermal FDTD grid.
    ///
    /// All cells are initialised to `base_temp`, vacuum permittivity (ε = 1),
    /// and zero thermo-optic coefficient.
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64, base_temp: f64) -> Self {
        let n = nx * ny * nz;
        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            temperature: vec![base_temp; n],
            base_temperature: base_temp,
            dn_dt: vec![0.0; n],
            eps_base: vec![1.0; n],
            eps_current: vec![1.0; n],
        }
    }

    /// Linear index for grid cell (i, j, k).
    #[inline]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        i * self.ny * self.nz + j * self.nz + k
    }

    /// Set the thermo-optic coefficient dn/dT for all cells inside the box
    /// [i0, i1) × [j0, j1) × [k0, k1).
    #[allow(clippy::too_many_arguments)]
    pub fn set_thermo_optic(
        &mut self,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        k0: usize,
        k1: usize,
        dn_dt: f64,
    ) {
        for i in i0..i1.min(self.nx) {
            for j in j0..j1.min(self.ny) {
                for k in k0..k1.min(self.nz) {
                    let idx = self.idx(i, j, k);
                    self.dn_dt[idx] = dn_dt;
                }
            }
        }
    }

    /// Set the base permittivity for all cells inside the box
    /// [i0, i1) × [j0, j1) × [k0, k1).
    #[allow(clippy::too_many_arguments)]
    pub fn set_eps_region(
        &mut self,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        k0: usize,
        k1: usize,
        eps: f64,
    ) {
        for i in i0..i1.min(self.nx) {
            for j in j0..j1.min(self.ny) {
                for k in k0..k1.min(self.nz) {
                    let idx = self.idx(i, j, k);
                    self.eps_base[idx] = eps;
                    self.eps_current[idx] = eps;
                }
            }
        }
    }

    /// Recompute the current permittivity from the temperature field.
    ///
    ///   n(T) = n₀ + (dn/dT) · (T − T_base)
    ///   ε(T) = n(T)²
    ///
    /// where n₀ = sqrt(ε_base).
    pub fn apply_temperature_field(&mut self) {
        let t_base = self.base_temperature;
        for idx in 0..(self.nx * self.ny * self.nz) {
            let n0 = self.eps_base[idx].max(0.0).sqrt();
            let delta_t = self.temperature[idx] - t_base;
            let n = n0 + self.dn_dt[idx] * delta_t;
            // Clamp to physically meaningful values (n > 0)
            let n_clamped = n.max(1e-6);
            self.eps_current[idx] = n_clamped * n_clamped;
        }
    }

    /// Set all cells to the same temperature.
    pub fn set_uniform_temperature(&mut self, temp_k: f64) {
        for t in self.temperature.iter_mut() {
            *t = temp_k;
        }
    }

    /// Load a temperature profile from an external slice.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] if `profile.len()` does not
    /// match the grid size.
    pub fn set_temperature_profile(&mut self, profile: &[f64]) -> Result<(), OxiPhotonError> {
        let expected = self.nx * self.ny * self.nz;
        if profile.len() != expected {
            return Err(OxiPhotonError::NumericalError(format!(
                "Temperature profile length {} does not match grid size {}",
                profile.len(),
                expected
            )));
        }
        self.temperature.copy_from_slice(profile);
        Ok(())
    }

    /// Local refractive index sqrt(ε_current) at cell (i, j, k).
    pub fn refractive_index_at(&self, i: usize, j: usize, k: usize) -> f64 {
        let idx = self.idx(i, j, k);
        self.eps_current[idx].max(0.0).sqrt()
    }

    /// Maximum temperature deviation from the base temperature across all cells.
    pub fn max_delta_t(&self) -> f64 {
        self.temperature
            .iter()
            .map(|&t| (t - self.base_temperature).abs())
            .fold(0.0_f64, f64::max)
    }

    /// Spatial average temperature over the full domain.
    pub fn mean_temperature(&self) -> f64 {
        let n = self.temperature.len();
        if n == 0 {
            return self.base_temperature;
        }
        self.temperature.iter().sum::<f64>() / n as f64
    }
}

// ─── HeatSolver3d ────────────────────────────────────────────────────────────

/// Explicit Euler finite-difference solver for the 3D heat equation.
///
///   ∂T/∂t = α ∇²T + Q/(ρ c_p)
///
/// The heat source term `Q` (W/m³) typically comes from Joule heating in the
/// electromagnetic simulation:  Q = σ |E|².
///
/// # Stability
/// The explicit scheme is conditionally stable.  Use \[`check_cfl_stability`\] to
/// verify that the chosen `dt` and `dx` satisfy the CFL condition before running.
///
/// # Boundary conditions
/// * Interior: standard six-neighbour Laplacian stencil.
/// * Domain boundary: Newton convection  `q = h (T − T_ambient)` modelled as a
///   Robin BC (ghost-cell elimination).  Set `convection_coefficient = 0` for
///   adiabatic (Neumann) boundaries.
///
/// The Robin BC stability constraint is independent of the Laplacian CFL:
///   dt < ρ c_p · ds / (2 · h)
/// where `ds` is the grid spacing perpendicular to the face and `ρ c_p` is the
/// volumetric heat capacity at the face cell.
#[derive(Debug, Clone)]
pub struct HeatSolver3d {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64, // m
    pub dy: f64, // m
    pub dz: f64, // m
    /// Thermal time step (s).  Must satisfy CFL: dt < min(dx²,dy²,dz²) / (6 α_max).
    pub dt: f64,

    /// Temperature field T\[i,j,k\] in Kelvin.
    pub temperature: Vec<f64>,

    /// Thermal diffusivity α(x,y,z) in m²/s per cell.
    /// Typical values: Si ≈ 8.8e-5, SiO₂ ≈ 8.3e-7, air ≈ 2.1e-5.
    pub thermal_diffusivity: Vec<f64>,

    /// Volumetric heat capacity ρ·c_p (x,y,z) in J/(m³·K) per cell.
    /// Default: 1.0e6 J/(m³·K) (approximately water-like).
    /// Used by the Robin convective BC to convert surface flux to temperature rate.
    pub volumetric_heat_capacity: Vec<f64>,

    /// Volumetric heat source Q(x,y,z) already divided by ρ c_p, in K/s per cell.
    pub heat_source: Vec<f64>,

    /// Ambient temperature for convective BC (K).
    pub ambient_temperature: f64,

    /// Newton convection coefficient h in W/(m²·K).
    /// Set to 0.0 for adiabatic boundaries.
    pub convection_coefficient: f64,
}

impl HeatSolver3d {
    /// Create a new heat solver initialised to `ambient_temp` everywhere.
    ///
    /// Default thermal diffusivity is that of air (~2.1e-5 m²/s); override
    /// material regions with \[`set_diffusivity_region`\].
    /// Default volumetric heat capacity is 1.0e6 J/(m³·K); override with
    /// \[`set_rho_cp_region`\].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        ambient_temp: f64,
    ) -> Self {
        let n = nx * ny * nz;
        // Default diffusivity: air
        let alpha_air = 2.1e-5_f64;
        // Default ρ·c_p: ~water-like (1e6 J/(m³·K))
        let rho_cp_default = 1.0e6_f64;
        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            dt,
            temperature: vec![ambient_temp; n],
            thermal_diffusivity: vec![alpha_air; n],
            volumetric_heat_capacity: vec![rho_cp_default; n],
            heat_source: vec![0.0; n],
            ambient_temperature: ambient_temp,
            convection_coefficient: 0.0,
        }
    }

    /// Create a 3D thermal grid pre-configured for silicon.
    ///
    /// α_Si ≈ 8.8e-5 m²/s (α = k/(ρ·c_p) = 148 / (2330 × 712))
    /// ρ·c_p_Si ≈ 1.66e6 J/(m³·K)
    #[allow(clippy::too_many_arguments)]
    pub fn for_silicon(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        ambient_temp: f64,
    ) -> Self {
        let mut sim = Self::new(nx, ny, nz, dx, dy, dz, dt, ambient_temp);
        let n = nx * ny * nz;
        let alpha_si = 8.8e-5_f64;
        let rho_cp_si = 1.66e6_f64;
        sim.thermal_diffusivity = vec![alpha_si; n];
        sim.volumetric_heat_capacity = vec![rho_cp_si; n];
        sim
    }

    /// Create a 3D thermal grid pre-configured for copper.
    ///
    /// α_Cu ≈ 1.17e-4 m²/s; ρ·c_p_Cu ≈ 3.44e6 J/(m³·K)
    #[allow(clippy::too_many_arguments)]
    pub fn for_copper(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        ambient_temp: f64,
    ) -> Self {
        let mut sim = Self::new(nx, ny, nz, dx, dy, dz, dt, ambient_temp);
        let n = nx * ny * nz;
        let alpha_cu = 1.17e-4_f64;
        let rho_cp_cu = 3.44e6_f64;
        sim.thermal_diffusivity = vec![alpha_cu; n];
        sim.volumetric_heat_capacity = vec![rho_cp_cu; n];
        sim
    }

    /// Linear index for cell (i, j, k).
    #[inline]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        i * self.ny * self.nz + j * self.nz + k
    }

    /// Six-neighbour Laplacian of `temp` at interior cell (i, j, k).
    ///
    /// At domain boundaries a zero-flux (Neumann) condition is applied by
    /// mirroring the interior value (ghost-cell approach).
    fn laplacian(&self, temp: &[f64], i: usize, j: usize, k: usize) -> f64 {
        let get =
            |ii: usize, jj: usize, kk: usize| temp[ii * self.ny * self.nz + jj * self.nz + kk];

        // Mirror indices at boundaries (zero-flux)
        let ip = if i + 1 < self.nx { i + 1 } else { i };
        let im = if i > 0 { i - 1 } else { i };
        let jp = if j + 1 < self.ny { j + 1 } else { j };
        let jm = if j > 0 { j - 1 } else { j };
        let kp = if k + 1 < self.nz { k + 1 } else { k };
        let km = if k > 0 { k - 1 } else { k };

        let t_c = get(i, j, k);
        let d2x = (get(ip, j, k) - 2.0 * t_c + get(im, j, k)) / (self.dx * self.dx);
        let d2y = (get(i, jp, k) - 2.0 * t_c + get(i, jm, k)) / (self.dy * self.dy);
        let d2z = (get(i, j, kp) - 2.0 * t_c + get(i, j, km)) / (self.dz * self.dz);

        d2x + d2y + d2z
    }

    /// Advance one explicit Euler thermal step.
    ///
    ///   T_new\[i,j,k\] = T\[i,j,k\] + dt · (α\[i,j,k\] · ∇²T + Q\[i,j,k\])
    ///
    /// The heat source array is assumed to already carry the 1/(ρ c_p) factor.
    ///
    /// # Stability constraints
    ///
    /// **Laplacian CFL**: `dt < min(dx², dy², dz²) / (6 · α_max)`.
    ///
    /// **Robin BC CFL** (when `convection_coefficient > 0`):
    /// `dt < ρ·c_p · ds / (2 · h)` for each face, where `ds` is the grid
    /// spacing perpendicular to that face and `h` is the convection coefficient.
    pub fn step(&mut self) {
        // Work on a copy so updates don't pollute the stencil within this step
        let t_old = self.temperature.clone();
        let mut t_new = t_old.clone();

        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    let idx = self.idx(i, j, k);
                    let lap = self.laplacian(&t_old, i, j, k);
                    let alpha = self.thermal_diffusivity[idx];
                    let q = self.heat_source[idx];
                    t_new[idx] = t_old[idx] + self.dt * (alpha * lap + q);
                }
            }
        }

        // Apply Newton/Robin convective BC on the six axis-aligned faces.
        // Discretized form (explicit forward Euler, ghost-cell elimination):
        //   ΔT[face] = -2 · h · dt · (T_old[face] - T_amb) / (ρ·c_p[face] · ds)
        // Stability constraint: dt < ρ·c_p · ds / (2·h).
        if self.convection_coefficient > 0.0 {
            let h = self.convection_coefficient;
            let t_amb = self.ambient_temperature;
            let dt = self.dt;

            // Face perpendicular to X (i=0 and i=nx-1)
            for j in 0..self.ny {
                for k in 0..self.nz {
                    let i0 = self.idx(0, j, k);
                    let i1 = self.idx(self.nx - 1, j, k);
                    let rcp0 = self.volumetric_heat_capacity[i0];
                    let rcp1 = self.volumetric_heat_capacity[i1];
                    if rcp0 > 1e-30 {
                        t_new[i0] += -2.0 * h * dt * (t_old[i0] - t_amb) / (rcp0 * self.dx);
                    }
                    if rcp1 > 1e-30 {
                        t_new[i1] += -2.0 * h * dt * (t_old[i1] - t_amb) / (rcp1 * self.dx);
                    }
                }
            }
            // Face perpendicular to Y (j=0 and j=ny-1)
            for i in 0..self.nx {
                for k in 0..self.nz {
                    let j0 = self.idx(i, 0, k);
                    let j1 = self.idx(i, self.ny - 1, k);
                    let rcp0 = self.volumetric_heat_capacity[j0];
                    let rcp1 = self.volumetric_heat_capacity[j1];
                    if rcp0 > 1e-30 {
                        t_new[j0] += -2.0 * h * dt * (t_old[j0] - t_amb) / (rcp0 * self.dy);
                    }
                    if rcp1 > 1e-30 {
                        t_new[j1] += -2.0 * h * dt * (t_old[j1] - t_amb) / (rcp1 * self.dy);
                    }
                }
            }
            // Face perpendicular to Z (k=0 and k=nz-1)
            for i in 0..self.nx {
                for j in 0..self.ny {
                    let k0 = self.idx(i, j, 0);
                    let k1 = self.idx(i, j, self.nz - 1);
                    let rcp0 = self.volumetric_heat_capacity[k0];
                    let rcp1 = self.volumetric_heat_capacity[k1];
                    if rcp0 > 1e-30 {
                        t_new[k0] += -2.0 * h * dt * (t_old[k0] - t_amb) / (rcp0 * self.dz);
                    }
                    if rcp1 > 1e-30 {
                        t_new[k1] += -2.0 * h * dt * (t_old[k1] - t_amb) / (rcp1 * self.dz);
                    }
                }
            }
        }

        self.temperature = t_new;
    }

    /// Inject Joule heating from FDTD fields.
    ///
    ///   Q\[i,j,k\] = σ\[i,j,k\] · |E|²\[i,j,k\]   (W/m³)
    ///
    /// The result is stored in `self.heat_source`.  Divide by ρ c_p externally
    /// if you need the temperature-rate form.
    pub fn inject_heat_source(&mut self, sigma: &[f64], e_field_sq: &[f64]) {
        let n = self.nx * self.ny * self.nz;
        let n_sigma = sigma.len().min(n);
        let n_esq = e_field_sq.len().min(n);
        for idx in 0..n {
            let s = if idx < n_sigma { sigma[idx] } else { 0.0 };
            let esq = if idx < n_esq { e_field_sq[idx] } else { 0.0 };
            self.heat_source[idx] = s * esq;
        }
    }

    /// Set thermal diffusivity α for all cells inside [i0,i1) × [j0,j1) × [k0,k1).
    #[allow(clippy::too_many_arguments)]
    pub fn set_diffusivity_region(
        &mut self,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        k0: usize,
        k1: usize,
        alpha: f64,
    ) {
        for i in i0..i1.min(self.nx) {
            for j in j0..j1.min(self.ny) {
                for k in k0..k1.min(self.nz) {
                    let idx = self.idx(i, j, k);
                    self.thermal_diffusivity[idx] = alpha;
                }
            }
        }
    }

    /// Set volumetric heat capacity ρ·c_p for all cells inside
    /// [i0,i1) × [j0,j1) × [k0,k1).
    ///
    /// Units: J/(m³·K).  Typical values: Si ≈ 1.66e6, Cu ≈ 3.44e6, air ≈ 1.2e3.
    #[allow(clippy::too_many_arguments)]
    pub fn set_rho_cp_region(
        &mut self,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        k0: usize,
        k1: usize,
        rho_cp: f64,
    ) {
        for i in i0..i1.min(self.nx) {
            for j in j0..j1.min(self.ny) {
                for k in k0..k1.min(self.nz) {
                    let idx = self.idx(i, j, k);
                    self.volumetric_heat_capacity[idx] = rho_cp;
                }
            }
        }
    }

    /// Load an initial temperature profile from an external slice.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] if the slice length mismatches
    /// the grid size.
    pub fn set_temperature(&mut self, profile: &[f64]) -> Result<(), OxiPhotonError> {
        let expected = self.nx * self.ny * self.nz;
        if profile.len() != expected {
            return Err(OxiPhotonError::NumericalError(format!(
                "Temperature profile length {} does not match grid size {}",
                profile.len(),
                expected
            )));
        }
        self.temperature.copy_from_slice(profile);
        Ok(())
    }

    /// Peak temperature in the domain.
    pub fn max_temperature(&self) -> f64 {
        self.temperature
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// CFL stability check for the explicit forward-Euler scheme.
    ///
    ///   dt < min(dx², dy², dz²) / (6 α_max)
    ///
    /// Returns `true` when the current `dt` satisfies the stability criterion.
    pub fn check_cfl_stability(&self) -> bool {
        let alpha_max = self
            .thermal_diffusivity
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max);
        if alpha_max < 1e-30 {
            return true; // zero diffusivity → trivially stable
        }
        let ds_sq_min = self.dx.powi(2).min(self.dy.powi(2)).min(self.dz.powi(2));
        self.dt < ds_sq_min / (6.0 * alpha_max)
    }

    /// Steady-state criterion: maximum |∂T/∂t| < `threshold_k_per_s`.
    ///
    /// Computes the approximate rate of change using the current temperature
    /// field without advancing the solver.
    pub fn is_steady_state(&self, threshold_k_per_s: f64) -> bool {
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    let idx = self.idx(i, j, k);
                    let alpha = self.thermal_diffusivity[idx];
                    let lap = self.laplacian(&self.temperature, i, j, k);
                    let q = self.heat_source[idx];
                    let rate = (alpha * lap + q).abs();
                    if rate >= threshold_k_per_s {
                        return false;
                    }
                }
            }
        }
        true
    }
}

// ─── ThermoOpticCoupler ───────────────────────────────────────────────────────

/// Bidirectionally coupled thermo-optic FDTD simulation.
///
/// Handles the multiscale coupling between the fast electromagnetic solver
/// (femtosecond time steps) and the slow thermal solver (nanosecond time steps):
///
/// 1. Run `optical_steps_per_thermal` electromagnetic FDTD steps.
/// 2. Compute Joule heating Q = σ|E|² from the accumulated E-field.
/// 3. Advance the heat equation by one thermal step.
/// 4. Update the permittivity map from the new temperature field.
/// 5. Repeat.
pub struct ThermoOpticCoupler {
    /// Thermo-optic permittivity grid (feeds into the EM solver).
    pub thermal: ThermalFdtd3d,
    /// Parabolic heat equation solver.
    pub heat_solver: HeatSolver3d,
    /// Number of EM time steps between consecutive thermal updates.
    pub optical_steps_per_thermal: usize,
    /// Electrical conductivity σ(x,y,z) in S/m (for Joule heating).
    pub sigma_e: Vec<f64>,
}

impl ThermoOpticCoupler {
    /// Create a new coupler.
    ///
    /// # Arguments
    /// * `thermal`                  – Thermo-optic permittivity grid.
    /// * `heat_solver`              – Configured heat equation solver.
    /// * `optical_steps_per_thermal` – How many EM steps per thermal step.
    /// * `sigma_e`                  – Electrical conductivity per cell (S/m).
    pub fn new(
        thermal: ThermalFdtd3d,
        heat_solver: HeatSolver3d,
        optical_steps_per_thermal: usize,
        sigma_e: Vec<f64>,
    ) -> Self {
        Self {
            thermal,
            heat_solver,
            optical_steps_per_thermal,
            sigma_e,
        }
    }

    /// Copy the current temperature from the heat solver into the thermo-optic
    /// grid, then recompute all permittivities.
    pub fn update_optical_from_thermal(&mut self) {
        // Lengths should match (same grid) — copy defensively
        let n = self
            .thermal
            .temperature
            .len()
            .min(self.heat_solver.temperature.len());
        self.thermal.temperature[..n].copy_from_slice(&self.heat_solver.temperature[..n]);
        self.thermal.apply_temperature_field();
    }

    /// Compute the cell-wise |E|² = Ex² + Ey² + Ez² from the three field components.
    ///
    /// # Panics (non-production path)
    /// In debug mode, mismatched slice lengths produce an array of zeros for the
    /// shorter components rather than a panic.
    pub fn compute_e_field_squared(ex: &[f64], ey: &[f64], ez: &[f64]) -> Vec<f64> {
        let n = ex.len().max(ey.len()).max(ez.len());
        let mut out = vec![0.0_f64; n];
        for idx in 0..n {
            let vx = if idx < ex.len() { ex[idx] } else { 0.0 };
            let vy = if idx < ey.len() { ey[idx] } else { 0.0 };
            let vz = if idx < ez.len() { ez[idx] } else { 0.0 };
            out[idx] = vx * vx + vy * vy + vz * vz;
        }
        out
    }

    /// Run one complete thermal update cycle:
    ///
    /// 1. Compute |E|² from the provided field components.
    /// 2. Inject Joule heating into the heat solver.
    /// 3. Advance the heat equation by one step.
    /// 4. Update the thermo-optic permittivity.
    pub fn thermal_update_step(&mut self, ex: &[f64], ey: &[f64], ez: &[f64]) {
        let e_sq = Self::compute_e_field_squared(ex, ey, ez);
        // Clone sigma_e to avoid borrow issues
        let sigma_clone = self.sigma_e.clone();
        self.heat_solver.inject_heat_source(&sigma_clone, &e_sq);
        self.heat_solver.step();
        self.update_optical_from_thermal();
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const BASE_T: f64 = 300.0; // K

    fn small_thermal_grid() -> ThermalFdtd3d {
        ThermalFdtd3d::new(4, 4, 4, 1e-6, 1e-6, 1e-6, BASE_T)
    }

    fn small_heat_solver(dt: f64) -> HeatSolver3d {
        HeatSolver3d::new(4, 4, 4, 1e-6, 1e-6, 1e-6, dt, BASE_T)
    }

    // ── ThermalFdtd3d tests ──────────────────────────────────────────────────

    #[test]
    fn test_thermal_fdtd_creation() {
        let grid = small_thermal_grid();
        // All cells should start at base temperature
        for &t in &grid.temperature {
            assert!(
                (t - BASE_T).abs() < 1e-10,
                "Initial temperature should be {BASE_T} K, got {t}"
            );
        }
        // Permittivity should default to vacuum
        for &eps in &grid.eps_current {
            assert!(
                (eps - 1.0).abs() < 1e-10,
                "Default permittivity should be 1.0, got {eps}"
            );
        }
    }

    #[test]
    fn test_set_uniform_temperature() {
        let mut grid = small_thermal_grid();
        let new_t = 400.0_f64;
        grid.set_uniform_temperature(new_t);
        for &t in &grid.temperature {
            assert!(
                (t - new_t).abs() < 1e-10,
                "All cells should be {new_t} K, got {t}"
            );
        }
    }

    #[test]
    fn test_apply_temperature_field() {
        let mut grid = small_thermal_grid();
        // Set a non-trivial background permittivity
        let eps_base = 12.25; // n₀ = 3.5 (silicon-like)
        grid.set_eps_region(0, 4, 0, 4, 0, 4, eps_base);

        // Set dn/dT = 1.8e-4 K⁻¹ (silicon thermo-optic coefficient)
        let dn_dt = 1.8e-4_f64;
        grid.set_thermo_optic(0, 4, 0, 4, 0, 4, dn_dt);

        // Raise temperature by 100 K
        let delta_t = 100.0;
        grid.set_uniform_temperature(BASE_T + delta_t);
        grid.apply_temperature_field();

        // n(T) = 3.5 + 1.8e-4 * 100 = 3.5180
        // ε(T) = 3.5180² ≈ 12.3763
        let n0 = eps_base.sqrt();
        let n_expected = n0 + dn_dt * delta_t;
        let eps_expected = n_expected * n_expected;

        for &eps in &grid.eps_current {
            assert!(
                (eps - eps_expected).abs() < 1e-6,
                "ε(T) should be {eps_expected:.6}, got {eps:.6}"
            );
        }
        // Permittivity should be larger than the base value (positive dn/dT)
        assert!(
            eps_expected > eps_base,
            "ε(T) should increase for positive dn/dT and positive ΔT"
        );
    }

    #[test]
    fn test_thermal_fdtd_profile_size_error() {
        let mut grid = small_thermal_grid();
        // Wrong-size profile → error
        let bad_profile = vec![350.0_f64; 3]; // 3 ≠ 64
        let result = grid.set_temperature_profile(&bad_profile);
        assert!(
            result.is_err(),
            "Should return error for mismatched profile size"
        );
    }

    // ── HeatSolver3d tests ───────────────────────────────────────────────────

    #[test]
    fn test_heat_solver_creation() {
        // Choose dt that satisfies CFL for default diffusivity (air ~2.1e-5 m²/s)
        // CFL: dt < dx² / (6 α) = (1e-6)² / (6 * 2.1e-5) ≈ 7.94e-9 s
        let dt = 5.0e-9_f64;
        let solver = small_heat_solver(dt);
        assert!(
            solver.check_cfl_stability(),
            "CFL should be satisfied for dt={dt:.2e} with default air diffusivity"
        );
        // Initial temperatures should equal ambient
        for &t in &solver.temperature {
            assert!(
                (t - BASE_T).abs() < 1e-10,
                "Initial temperature should be {BASE_T} K"
            );
        }
    }

    #[test]
    fn test_heat_solver_step() {
        // Create a 6×6×6 grid with a hot spot at centre; verify it diffuses
        let nx = 6;
        let ny = 6;
        let nz = 6;
        let dx = 1e-6_f64;
        let alpha_si = 8.8e-5_f64; // silicon thermal diffusivity (m²/s)
                                   // CFL limit: dx²/(6α) ≈ 1.89e-9 s → use 1e-9 s
        let dt = 1.0e-9_f64;

        let mut solver = HeatSolver3d::new(nx, ny, nz, dx, dx, dx, dt, BASE_T);
        solver.set_diffusivity_region(0, nx, 0, ny, 0, nz, alpha_si);

        // Place a hot spot at cell (3, 3, 3)
        let hot_idx = solver.idx(3, 3, 3);
        solver.temperature[hot_idx] = BASE_T + 200.0;

        let t_before = solver.temperature[hot_idx];
        solver.step();
        let t_after = solver.temperature[hot_idx];

        // Hot spot should cool (energy diffuses outward)
        assert!(
            t_after < t_before,
            "Hot spot should cool after one step: {t_before:.2} → {t_after:.2}"
        );
        // Neighbours should warm up
        let nb_idx = solver.idx(4, 3, 3);
        assert!(
            solver.temperature[nb_idx] > BASE_T,
            "Neighbour cell should warm above ambient after hot-spot step"
        );
    }

    #[test]
    fn test_cfl_stability() {
        let dx = 1e-6_f64;
        let alpha = 8.8e-5_f64; // silicon
                                // Stable dt
        let dt_stable = 1.0e-9_f64;
        let solver_ok = HeatSolver3d::new(4, 4, 4, dx, dx, dx, dt_stable, BASE_T);
        // We need the solver to use silicon diffusivity
        let mut solver_ok = solver_ok;
        solver_ok.set_diffusivity_region(0, 4, 0, 4, 0, 4, alpha);
        assert!(
            solver_ok.check_cfl_stability(),
            "dt={dt_stable:.2e} should be stable for silicon"
        );

        // Unstable dt (10× too large)
        let dt_bad = 1.0e-7_f64;
        let mut solver_bad = HeatSolver3d::new(4, 4, 4, dx, dx, dx, dt_bad, BASE_T);
        solver_bad.set_diffusivity_region(0, 4, 0, 4, 0, 4, alpha);
        assert!(
            !solver_bad.check_cfl_stability(),
            "dt={dt_bad:.2e} should be unstable for silicon"
        );
    }

    #[test]
    fn test_inject_heat_source() {
        let mut solver = small_heat_solver(1e-9);
        let n = 4 * 4 * 4;
        let sigma = vec![1.0e4_f64; n]; // 10 kS/m
        let e_sq = vec![1.0e6_f64; n]; // |E|² = (1000 V/m)²

        solver.inject_heat_source(&sigma, &e_sq);

        // Q = σ |E|² = 1e4 * 1e6 = 1e10 W/m³ at every cell
        let expected_q = 1.0e10_f64;
        for (idx, &q) in solver.heat_source.iter().enumerate() {
            assert!(
                (q - expected_q).abs() < 1e3,
                "Heat source at {idx} should be {expected_q:.2e}, got {q:.2e}"
            );
        }
    }

    #[test]
    fn test_thermo_optic_coupler_e_field_squared() {
        let n = 8;
        let ex = vec![1.0_f64; n];
        let ey = vec![2.0_f64; n];
        let ez = vec![3.0_f64; n];
        let e_sq = ThermoOpticCoupler::compute_e_field_squared(&ex, &ey, &ez);
        // |E|² = 1 + 4 + 9 = 14
        for &v in &e_sq {
            assert!((v - 14.0).abs() < 1e-10, "|E|² should be 14.0, got {v}");
        }
    }
}
