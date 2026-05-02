//! 1D Boltzmann-statistics drift-diffusion solver for solar cells.
//!
//! Implements a self-consistent Poisson + electron + hole continuity solver
//! using Scharfetter-Gummel flux discretisation and a damped Newton method
//! with backtracking line-search.
//!
//! # Physical fidelity
//! * Boltzmann statistics (valid for doping < ~1e18 cm⁻³).
//! * SRH + radiative + Auger recombination at every node.
//! * Ohmic boundary conditions (Dirichlet).
//! * Monochromatic or spectrally-resolved generation profile G(z).
//!
//! # References
//! Sze & Ng, "Physics of Semiconductor Devices", 3rd ed. (2006), Ch. 2.
//! Scharfetter & Gummel, IEEE TED 16(1), 64-77 (1969).
//! Selberherr, "Analysis and Simulation of Semiconductor Devices" (1984).

pub mod bandgap_narrowing;
pub mod continuity;
pub mod fermi_dirac;
pub mod material;
pub mod newton;
pub mod poisson;
pub mod recombination;

pub use bandgap_narrowing::BgnModel;
pub use fermi_dirac::{f_half, f_minus_half, joyce_dixon_eta};
pub use material::{SemiconductorMaterial, StatisticsModel};

use crate::error::OxiPhotonError;
use material::Q;

// ─── Doping profile ───────────────────────────────────────────────────────────

/// Doping profile specifying donor and acceptor concentrations at each node.
#[derive(Debug, Clone)]
pub struct DopingProfile {
    /// Ionised donor density N_d⁺ (cm⁻³) at each node.
    pub nd: Vec<f64>,
    /// Ionised acceptor density N_a⁻ (cm⁻³) at each node.
    pub na: Vec<f64>,
}

impl DopingProfile {
    /// Simple p-n junction: left half p-type (acceptor density `na_cm3`),
    /// right half n-type (donor density `nd_cm3`).
    ///
    /// This profile steps abruptly at the midpoint with no grading.
    pub fn pn_junction(n_nodes: usize, na_cm3: f64, nd_cm3: f64) -> Self {
        let mid = n_nodes / 2;
        let mut na = vec![0.0_f64; n_nodes];
        let mut nd = vec![0.0_f64; n_nodes];
        for item in na.iter_mut().take(mid) {
            *item = na_cm3;
        }
        for item in nd.iter_mut().take(n_nodes).skip(mid) {
            *item = nd_cm3;
        }
        Self { na, nd }
    }
}

// ─── Device facade ────────────────────────────────────────────────────────────

/// 1D drift-diffusion solar-cell device.
///
/// Encapsulates the material, doping profile, and current solution state (ψ, n, p).
/// The device supports dark and illuminated IV sweeps, and monochromatic IQE
/// computation.
///
/// # Example
/// ```
/// # use oxiphoton::solar::drift_diffusion::{DriftDiffusionDevice, DopingProfile, SemiconductorMaterial};
/// let mat = SemiconductorMaterial::silicon();
/// let doping = DopingProfile::pn_junction(100, 1e16, 1e16);
/// let mut device = DriftDiffusionDevice::new(mat, doping, 5e-4, 100)
///     .expect("device creation should succeed");
/// let _n_iters = device.solve_equilibrium().expect("equilibrium solve");
/// ```
pub struct DriftDiffusionDevice {
    /// Material parameters.
    pub material: SemiconductorMaterial,
    /// Doping profile.
    pub doping: DopingProfile,
    /// Total device thickness (cm).
    pub thickness_cm: f64,
    /// Number of grid nodes.
    pub n_nodes: usize,
    /// Device temperature (K).
    pub temp_k: f64,
    /// Node positions x_i (cm).
    pub x_cm: Vec<f64>,
    /// Grid spacings `dx[i]` = x_{i+1} - x_i (cm), length n_nodes-1.
    pub dx_cm: Vec<f64>,
    /// Electrostatic potential (V) at each node.
    pub psi: Vec<f64>,
    /// Electron density (cm⁻³) at each node.
    pub n_carriers: Vec<f64>,
    /// Hole density (cm⁻³) at each node.
    pub p_carriers: Vec<f64>,
}

impl DriftDiffusionDevice {
    /// Construct a new 1D drift-diffusion device on a uniform grid.
    ///
    /// The initial guess uses the equilibrium quadratic solve:
    ///   n = ½(N_net + √(N_net² + 4nᵢ²)),  p = nᵢ²/n
    ///
    /// # Errors
    /// Returns `NumericalError` if `n_nodes < 3`.
    pub fn new(
        material: SemiconductorMaterial,
        doping: DopingProfile,
        thickness_cm: f64,
        n_nodes: usize,
    ) -> Result<Self, OxiPhotonError> {
        if n_nodes < 3 {
            return Err(OxiPhotonError::NumericalError(
                "n_nodes must be >= 3".to_string(),
            ));
        }
        let dx_uniform = thickness_cm / (n_nodes - 1) as f64;
        let x_cm: Vec<f64> = (0..n_nodes).map(|i| i as f64 * dx_uniform).collect();
        let dx_cm = vec![dx_uniform; n_nodes - 1];
        let temp_k = 300.0;

        let vt = material.vt_at(temp_k);
        let ni = material.ni_cm3;
        let mut psi = vec![0.0_f64; n_nodes];
        let mut n_c = vec![0.0_f64; n_nodes];
        let mut p_c = vec![0.0_f64; n_nodes];

        // Compute per-node equilibrium carrier concentrations using numerically stable
        // quadratic formula (avoids catastrophic cancellation at heavy doping).
        // We build a temporary dummy device to call equil_n, but instead use the
        // same numerically stable logic directly here.
        for i in 0..n_nodes {
            let ni_eff = material
                .n_ie_squared(temp_k, doping.nd[i], doping.na[i])
                .sqrt();
            let net = doping.nd[i] - doping.na[i];
            let disc = (0.25 * net * net + ni_eff * ni_eff).sqrt();
            let n_i = if net >= 0.0 {
                0.5 * net + disc
            } else {
                let denom = -0.5 * net + disc;
                if denom > 0.0 {
                    ni_eff * ni_eff / denom
                } else {
                    ni_eff * 1e-10
                }
            };
            n_c[i] = n_i.max(1e-10 * ni);
            p_c[i] = ni_eff * ni_eff / n_c[i];
            psi[i] = vt * (n_c[i] / ni_eff).ln();
        }

        Ok(Self {
            material,
            doping,
            thickness_cm,
            n_nodes,
            temp_k,
            x_cm,
            dx_cm,
            psi,
            n_carriers: n_c,
            p_carriers: p_c,
        })
    }

    /// Compute equilibrium electron density from doping via the quadratic formula.
    ///
    /// Uses the numerically stable form to avoid catastrophic cancellation at heavy doping:
    ///   For p-type (net < 0): n = ni² / (0.5·|net| + √(0.25·net² + ni²))
    ///   For n-type (net ≥ 0): n = 0.5·net + √(0.25·net² + ni²)
    fn equil_n(&self, na: f64, nd: f64, ni: f64) -> f64 {
        let net = nd - na;
        let disc = (0.25 * net * net + ni * ni).sqrt();
        let n = if net >= 0.0 {
            // n-type or intrinsic: direct formula is stable (both terms positive)
            0.5 * net + disc
        } else {
            // p-type: use rationalised form to avoid n ≈ 0 - 0 cancellation
            // n·(0.5·|net| + disc) = ni²  →  n = ni² / (0.5·|net| + disc)
            let denom = -0.5 * net + disc; // = 0.5*|net| + disc, always > 0
            if denom > 0.0 {
                ni * ni / denom
            } else {
                ni * 1e-10
            }
        };
        n.max(1e-10 * ni)
    }

    /// Solve the dark equilibrium problem (zero applied bias, zero generation).
    ///
    /// Sets ohmic BCs, then calls the Newton solver.
    ///
    /// # Returns
    /// Number of Newton iterations on convergence.
    pub fn solve_equilibrium(&mut self) -> Result<usize, OxiPhotonError> {
        let vt = self.material.vt_at(self.temp_k);

        // Left contact (p-type ohmic)
        let na_left = self.doping.na[0].max(1.0);
        let nd_left = self.doping.nd[0];
        let ni_eff_left = self
            .material
            .n_ie_squared(self.temp_k, nd_left, na_left)
            .sqrt();
        let n_eq_left = self.equil_n(na_left, nd_left, ni_eff_left);
        let p_eq_left = ni_eff_left * ni_eff_left / n_eq_left;
        let psi_left = vt * (n_eq_left / ni_eff_left).ln();

        // Right contact (n-type ohmic)
        let na_right = self.doping.na[self.n_nodes - 1];
        let nd_right = self.doping.nd[self.n_nodes - 1].max(1.0);
        let ni_eff_right = self
            .material
            .n_ie_squared(self.temp_k, nd_right, na_right)
            .sqrt();
        let n_eq_right = self.equil_n(na_right, nd_right, ni_eff_right);
        let p_eq_right = ni_eff_right * ni_eff_right / n_eq_right;
        let psi_right = vt * (n_eq_right / ni_eff_right).ln();

        // Set boundary nodes
        self.psi[0] = psi_left;
        self.n_carriers[0] = n_eq_left;
        self.p_carriers[0] = p_eq_left;
        self.psi[self.n_nodes - 1] = psi_right;
        self.n_carriers[self.n_nodes - 1] = n_eq_right;
        self.p_carriers[self.n_nodes - 1] = p_eq_right;

        newton::solve_equilibrium_gummel(
            &mut self.psi,
            &mut self.n_carriers,
            &mut self.p_carriers,
            &self.doping.nd,
            &self.doping.na,
            &self.dx_cm,
            &self.material,
            self.temp_k,
            psi_left,
            psi_right,
            n_eq_left,
            p_eq_left,
            n_eq_right,
            p_eq_right,
        )
    }

    /// Compute ohmic contact conditions for a given node's doping.
    fn contact_bc(&self, na: f64, nd: f64) -> (f64, f64, f64) {
        let vt = self.material.vt_at(self.temp_k);
        let ni_eff = self.material.n_ie_squared(self.temp_k, nd, na).sqrt();
        let n_eq = self.equil_n(na, nd, ni_eff);
        let p_eq = ni_eff * ni_eff / n_eq;
        let psi_eq = vt * (n_eq / ni_eff).ln();
        (psi_eq, n_eq, p_eq)
    }

    /// Solve the dark IV characteristic over the given voltage grid.
    ///
    /// Each bias point warm-starts from the previous solution.
    /// The equilibrium solution is computed first if needed.
    ///
    /// # Returns
    /// Vector of (V, J) pairs in V and A/cm².
    pub fn solve_dark_iv(&mut self, v_grid: &[f64]) -> Result<Vec<(f64, f64)>, OxiPhotonError> {
        self.solve_equilibrium()?;
        let gen = vec![0.0_f64; self.n_nodes];
        self.solve_iv_sweep(v_grid, &gen)
    }

    /// Solve the illuminated IV characteristic over the given voltage grid.
    ///
    /// The generation profile `generation` (cm⁻³ s⁻¹) must have length equal to `n_nodes`.
    /// Each bias point warm-starts from the previous solution.
    ///
    /// Uses generation ramping to find a good short-circuit initial state before sweeping V.
    ///
    /// # Returns
    /// Vector of (V, J) pairs in V and A/cm².
    pub fn solve_illuminated_iv(
        &mut self,
        v_grid: &[f64],
        generation: &[f64],
    ) -> Result<Vec<(f64, f64)>, OxiPhotonError> {
        if generation.len() != self.n_nodes {
            return Err(OxiPhotonError::NumericalError(format!(
                "generation profile length {} != n_nodes {}",
                generation.len(),
                self.n_nodes
            )));
        }
        // Start from equilibrium
        self.solve_equilibrium()?;

        // Ramp generation in log-space to find the illuminated short-circuit point.
        // This improves convergence when G creates large excess carrier density
        // (high injection) by gradually stepping toward the target generation.
        let g_max = generation.iter().cloned().fold(0.0_f64, f64::max);
        if g_max > 0.0 {
            let ni = self.material.ni_cm3;
            let vt = self.material.vt_at(self.temp_k);

            // Determine number of ramp steps based on injection ratio
            let na_left = self.doping.na[0].max(1.0);
            let nd_left = self.doping.nd[0];
            let ni_eff_left = self
                .material
                .n_ie_squared(self.temp_k, nd_left, na_left)
                .sqrt();
            let n_eq_left = self.equil_n(na_left, nd_left, ni_eff_left);
            let psi_left_eq = vt * (n_eq_left / ni_eff_left).ln();
            let na_right = self.doping.na[self.n_nodes - 1];
            let nd_right = self.doping.nd[self.n_nodes - 1].max(1.0);
            let ni_eff_right = self
                .material
                .n_ie_squared(self.temp_k, nd_right, na_right)
                .sqrt();
            let n_eq_right = self.equil_n(na_right, nd_right, ni_eff_right);
            let psi_right_eq = vt * (n_eq_right / ni_eff_right).ln();

            // Estimate excess carriers at max generation (use bulk ni for ramp sizing heuristic)
            let tau_est = self.material.tau_n_s.min(self.material.tau_p_s);
            let excess_ratio = g_max * tau_est / ni;
            // Number of log-steps: 1 step per decade of excess ratio
            let n_ramp = (1.0 + excess_ratio.log10().max(0.0)).ceil() as usize;
            let n_ramp = n_ramp.clamp(1, 8);

            for step in 1..=n_ramp {
                let frac = step as f64 / n_ramp as f64;
                let gen_ramp: Vec<f64> = generation.iter().map(|&g| g * frac).collect();

                // Set BCs for V=0 (short circuit)
                self.psi[0] = psi_left_eq;
                self.n_carriers[0] = self.equil_n(na_left, nd_left, ni_eff_left);
                self.p_carriers[0] = ni_eff_left * ni_eff_left / self.n_carriers[0];
                self.psi[self.n_nodes - 1] = psi_right_eq;
                self.n_carriers[self.n_nodes - 1] = n_eq_right;
                self.p_carriers[self.n_nodes - 1] = ni_eff_right * ni_eff_right / n_eq_right;

                newton::newton_solve(
                    &mut self.psi,
                    &mut self.n_carriers,
                    &mut self.p_carriers,
                    &self.doping.nd,
                    &self.doping.na,
                    &gen_ramp,
                    &self.dx_cm,
                    &self.material,
                    self.temp_k,
                    psi_left_eq,
                    psi_right_eq,
                    200,
                    1e-7,
                )?;
            }
        }

        self.solve_iv_sweep(v_grid, generation)
    }

    /// Internal IV sweep implementation used by both dark and illuminated solvers.
    ///
    /// When consecutive bias points differ by more than one thermal voltage (VT),
    /// the step is sub-divided with intermediate Gummel solves (bias ramping).
    /// This ensures convergence for large forward-bias jumps where a single
    /// Gummel call from equilibrium (V=0) to V=0.4V (≈15 VT) would stall.
    fn solve_iv_sweep(
        &mut self,
        v_grid: &[f64],
        gen: &[f64],
    ) -> Result<Vec<(f64, f64)>, OxiPhotonError> {
        let mut iv = Vec::with_capacity(v_grid.len());

        let na_left = self.doping.na[0].max(1.0);
        let nd_left = self.doping.nd[0];
        let (psi_left_eq, n_eq_left, p_eq_left) = self.contact_bc(na_left, nd_left);

        let na_right = self.doping.na[self.n_nodes - 1];
        let nd_right = self.doping.nd[self.n_nodes - 1].max(1.0);
        let (psi_right_eq, n_eq_right, p_eq_right) = self.contact_bc(na_right, nd_right);

        let vt = self.material.vt_at(self.temp_k);
        // Maximum allowed single-step bias change.
        // Gummel has linear convergence; each outer iteration resolves ~1 VT of
        // potential change. Sub-stepping at 0.5 VT intervals ensures ~200 Gummel
        // iterations are more than sufficient to converge each sub-step.
        let max_step_v = 0.5 * vt;

        // Track the current applied bias so intermediate ramp steps are relative.
        let mut v_prev = 0.0_f64;

        for &v in v_grid {
            // Determine number of sub-steps needed to move from v_prev to v.
            let dv = v - v_prev;
            let n_sub = if dv.abs() > max_step_v {
                (dv.abs() / max_step_v).ceil() as usize
            } else {
                1
            };

            for sub in 1..=n_sub {
                let v_sub = v_prev + dv * (sub as f64 / n_sub as f64);
                let psi_left = psi_left_eq + v_sub;
                let psi_right = psi_right_eq;

                // Update boundary nodes (warm-start from previous sub-step).
                self.psi[0] = psi_left;
                self.n_carriers[0] = n_eq_left;
                self.p_carriers[0] = p_eq_left;
                self.psi[self.n_nodes - 1] = psi_right;
                self.n_carriers[self.n_nodes - 1] = n_eq_right;
                self.p_carriers[self.n_nodes - 1] = p_eq_right;

                newton::newton_solve(
                    &mut self.psi,
                    &mut self.n_carriers,
                    &mut self.p_carriers,
                    &self.doping.nd,
                    &self.doping.na,
                    gen,
                    &self.dx_cm,
                    &self.material,
                    self.temp_k,
                    psi_left,
                    psi_right,
                    200,
                    1e-8,
                )?;
            }

            let j = self.terminal_current();
            iv.push((v, j));
            v_prev = v;
        }
        Ok(iv)
    }

    /// Compute the monochromatic internal quantum efficiency (IQE).
    ///
    /// Generates a Beer-Lambert profile G(z) = α · exp(−α·z) for the given
    /// absorption coefficient `alpha_cm` (cm⁻¹), solves the illuminated IV at
    /// short-circuit (V = 0), and returns:
    ///
    ///   IQE = J_sc / (q · ∫G dz)
    ///
    /// Clamped to [0, 1].
    pub fn compute_iqe(&mut self, alpha_cm: f64) -> Result<f64, OxiPhotonError> {
        let gen: Vec<f64> = self
            .x_cm
            .iter()
            .map(|&x| alpha_cm * (-alpha_cm * x).exp())
            .collect();

        // Integrate G over device using trapezoidal rule
        let mut total_gen = 0.0_f64;
        for i in 0..self.n_nodes - 1 {
            total_gen += 0.5 * (gen[i] + gen[i + 1]) * self.dx_cm[i];
        }

        let v_sc = vec![0.0_f64];
        let iv = self.solve_illuminated_iv(&v_sc, &gen)?;
        let j_sc = iv[0].1.abs();
        let denom = (Q * total_gen).max(1e-100);
        Ok((j_sc / denom).clamp(0.0, 1.0))
    }

    /// Extract J_sc and V_oc from an IV curve.
    ///
    /// * J_sc: |J| at V ≈ 0.
    /// * V_oc: voltage where J changes sign (linear interpolation).
    pub fn extract_jsc_voc(&self, iv: &[(f64, f64)]) -> (f64, f64) {
        let j_sc = iv
            .iter()
            .find(|(v, _)| v.abs() < 1e-6)
            .map(|(_, j)| j.abs())
            .unwrap_or(0.0);

        let v_oc = iv
            .windows(2)
            .filter(|w| w[0].1 * w[1].1 < 0.0)
            .map(|w| {
                let (v1, j1) = w[0];
                let (v2, j2) = w[1];
                // Guard against division by zero
                let dj = j2 - j1;
                if dj.abs() < 1e-30 {
                    v1
                } else {
                    v1 - j1 * (v2 - v1) / dj
                }
            })
            .next()
            .unwrap_or(0.0);

        (j_sc, v_oc)
    }

    /// Compute the terminal current J_n + J_p at the left contact (node 0→1 half-node).
    pub fn terminal_current(&self) -> f64 {
        use continuity::{sg_flux_n, sg_flux_p};
        let vt = self.material.vt_at(self.temp_k);
        let dn = self.material.dn_cm2_s(self.temp_k);
        let dp = self.material.dp_cm2_s(self.temp_k);
        let dx0 = self.dx_cm[0];

        let j_n = sg_flux_n(
            self.n_carriers[0],
            self.n_carriers[1],
            self.psi[0],
            self.psi[1],
            vt,
            dn,
            dx0,
        );
        let j_p = sg_flux_p(
            self.p_carriers[0],
            self.p_carriers[1],
            self.psi[0],
            self.psi[1],
            vt,
            dp,
            dx0,
        );
        j_n + j_p
    }

    /// Compute the local current J_n + J_p at half-node i+½.
    ///
    /// Used for current-continuity diagnostics.
    pub fn terminal_current_at(&self, i: usize) -> f64 {
        use continuity::{sg_flux_n, sg_flux_p};
        if i >= self.n_nodes - 1 {
            return 0.0;
        }
        let vt = self.material.vt_at(self.temp_k);
        let dn = self.material.dn_cm2_s(self.temp_k);
        let dp = self.material.dp_cm2_s(self.temp_k);
        let dxi = self.dx_cm[i];

        let j_n = sg_flux_n(
            self.n_carriers[i],
            self.n_carriers[i + 1],
            self.psi[i],
            self.psi[i + 1],
            vt,
            dn,
            dxi,
        );
        let j_p = sg_flux_p(
            self.p_carriers[i],
            self.p_carriers[i + 1],
            self.psi[i],
            self.psi[i + 1],
            vt,
            dp,
            dxi,
        );
        j_n + j_p
    }
}
