//! Polariton laser and Bose-Einstein condensation dynamics.
//!
//! Polariton lasers differ fundamentally from conventional photon lasers:
//! condensation occurs in the lower polariton branch without population inversion,
//! the threshold is orders of magnitude lower, and the emitted light is coherent
//! due to the macroscopic occupation of the k=0 state.
//!
//! # Mean-field rate equations
//!
//! The coupled equations for the condensate density n_c and the reservoir
//! (high-k polariton / exciton) density n_R are:
//!
//! ```text
//! dn_c/dt = (W_R · n_R - 1/τ_LP) · n_c + W_R · n_R
//! dn_R/dt = P - (W_R · n_R + 1/τ_R) · n_R - W_R · n_R · n_c
//! ```
//!
//! where W_R is the stimulated scattering rate (bosonic enhancement),
//! τ_LP is the lower-polariton lifetime, and τ_R is the reservoir lifetime.
//!
//! # References
//! - I. Carusotto & C. Ciuti, Rev. Mod. Phys. 85, 299 (2013)
//! - A. Baas et al., Phys. Rev. Lett. 93, 236401 (2004) (bistability)
//! - J. Kasprzak et al., Nature 443, 409 (2006) (first polariton BEC)

use std::f64::consts::PI;

// ─── PolaritonCondensate ──────────────────────────────────────────────────────

/// Mean-field model for a polariton condensate in a planar microcavity.
///
/// Combines Gross-Pitaevskii dynamics for the coherent condensate with
/// Boltzmann rate equations for the incoherent reservoir of high-momentum
/// polaritons and free carriers.
#[derive(Debug, Clone)]
pub struct PolaritonCondensate {
    /// Vacuum Rabi splitting ħΩ_R (eV) — determines polariton character.
    pub rabi_splitting_ev: f64,
    /// Lower-polariton lifetime τ_LP (ps).
    /// Typical range: 1 ps (reflectivity R ~ 99%) to 100 ps (R ~ 99.99%).
    pub polariton_lifetime_ps: f64,
    /// Reservoir (high-k polariton / free exciton) lifetime τ_R (ps).
    pub reservoir_lifetime_ps: f64,
    /// Reservoir-to-condensate stimulated scattering rate W_R (ps⁻¹·µm²).
    /// Encodes the bosonic enhancement of stimulated scattering.
    pub relaxation_rate_ps: f64,
    /// Polariton-polariton interaction coefficient g_X (eV·µm²).
    /// Drives energy blueshift of the condensate.
    pub blueshift_coeff: f64,
    /// Continuous-wave pump rate P (polaritons per ps per µm²).
    pub pumping_rate: f64,
    /// Lattice / phonon bath temperature T (K).
    pub temperature_k: f64,
}

impl PolaritonCondensate {
    /// Threshold pump rate P_th for polariton condensation.
    ///
    /// Obtained from the steady-state condition dN_c/dt = 0 at N_c = 0:
    ///
    /// ```text
    /// P_th = (1/τ_LP) / (W_R · τ_R)
    /// ```
    ///
    /// Below P_th the condensate is zero; above it macroscopic occupation builds up.
    pub fn threshold_pump_rate(&self) -> f64 {
        let tau_lp = self.polariton_lifetime_ps;
        let tau_r = self.reservoir_lifetime_ps;
        let w_r = self.relaxation_rate_ps;
        if tau_lp <= 0.0 || tau_r <= 0.0 || w_r <= 0.0 {
            return f64::INFINITY;
        }
        1.0 / (tau_lp * w_r * tau_r)
    }

    /// Steady-state condensate density n_c (normalised) above threshold.
    ///
    /// ```text
    /// n_c = max(0, (P/P_th - 1) × P_th × τ_LP)
    /// ```
    pub fn condensate_density(&self) -> f64 {
        let p_th = self.threshold_pump_rate();
        if p_th <= 0.0 || !p_th.is_finite() {
            return 0.0;
        }
        let ratio = self.pumping_rate / p_th;
        if ratio <= 1.0 {
            0.0
        } else {
            (ratio - 1.0) * p_th * self.polariton_lifetime_ps
        }
    }

    /// Steady-state reservoir density n_R.
    ///
    /// At threshold the reservoir saturates at n_R^th = 1/(W_R · τ_LP).
    /// Above threshold the reservoir is clamped at this value.
    pub fn reservoir_density(&self) -> f64 {
        let w_r = self.relaxation_rate_ps;
        let tau_lp = self.polariton_lifetime_ps;
        if w_r <= 0.0 || tau_lp <= 0.0 {
            return 0.0;
        }
        // Reservoir saturates at threshold value
        let n_r_th = 1.0 / (w_r * tau_lp);
        let p_th = self.threshold_pump_rate();
        if self.pumping_rate <= p_th {
            // Below threshold: n_R ≈ P × τ_R
            self.pumping_rate * self.reservoir_lifetime_ps
        } else {
            n_r_th
        }
    }

    /// Energy blueshift of the condensate due to polariton-polariton interactions.
    ///
    /// ```text
    /// ΔE = g_X × n_c    [eV]
    /// ```
    ///
    /// This interaction-driven blueshift is the main nonlinear optical response
    /// of polariton condensates.
    pub fn blueshift_ev(&self) -> f64 {
        self.blueshift_coeff * self.condensate_density()
    }

    /// Time evolution of condensate and reservoir densities via Euler integration.
    ///
    /// Integrates the coupled rate equations:
    ///
    /// ```text
    /// dn_c/dt = (W_R n_R - 1/τ_LP) n_c + W_R n_R
    /// dn_R/dt = P - (W_R n_c + W_R + 1/τ_R) n_R
    /// ```
    ///
    /// Returns `Vec<(time_ps, n_condensate, n_reservoir)>`.
    pub fn time_evolution(&self, t_max_ps: f64, dt_ps: f64) -> Vec<(f64, f64, f64)> {
        if t_max_ps <= 0.0 || dt_ps <= 0.0 {
            return Vec::new();
        }
        let n_steps = ((t_max_ps / dt_ps) as usize).min(1_000_000);
        let mut result = Vec::with_capacity(n_steps + 1);

        let tau_lp = self.polariton_lifetime_ps.max(f64::EPSILON);
        let tau_r = self.reservoir_lifetime_ps.max(f64::EPSILON);
        let w_r = self.relaxation_rate_ps;
        let p = self.pumping_rate;

        let mut n_c = 0.0_f64;
        let mut n_r = 0.0_f64;
        let mut t = 0.0_f64;

        result.push((t, n_c, n_r));

        for _ in 0..n_steps {
            // Rate equations (bosonic stimulation included)
            let dn_c = (w_r * n_r - 1.0 / tau_lp) * n_c + w_r * n_r;
            let dn_r = p - (w_r * n_c + w_r + 1.0 / tau_r) * n_r;

            n_c = (n_c + dt_ps * dn_c).max(0.0);
            n_r = (n_r + dt_ps * dn_r).max(0.0);
            t += dt_ps;
            result.push((t, n_c, n_r));
        }
        result
    }

    /// Light-intensity curve (L-I equivalent) vs pump rate.
    ///
    /// Returns `Vec<(pump_rate, output_photon_flux)>` where the output flux is
    /// proportional to the condensate density divided by the photon lifetime.
    pub fn li_curve(&self, pump_range: (f64, f64), n_points: usize) -> Vec<(f64, f64)> {
        let n = n_points.max(2);
        let (p_lo, p_hi) = pump_range;
        let tau_lp = self.polariton_lifetime_ps.max(f64::EPSILON);

        (0..n)
            .map(|i| {
                let p = p_lo + (p_hi - p_lo) * (i as f64) / ((n - 1) as f64);
                let mut cond = self.clone();
                cond.pumping_rate = p;
                let n_c = cond.condensate_density();
                let flux = n_c / tau_lp; // photon output ∝ n_c / τ_LP
                (p, flux)
            })
            .collect()
    }

    /// Polariton laser linewidth via a Schawlow-Townes analog.
    ///
    /// Above threshold the linewidth narrows as 1/n_c (phase diffusion):
    ///
    /// ```text
    /// Δω ≈ 1 / (τ_LP · n_c)    [rad/ps]
    /// ```
    ///
    /// Returns the linewidth in Hz.  Returns `f64::INFINITY` below threshold.
    pub fn linewidth_hz(&self) -> f64 {
        let n_c = self.condensate_density();
        if n_c < 1.0 {
            return f64::INFINITY;
        }
        let tau_lp_s = self.polariton_lifetime_ps * 1e-12;
        // Δω = 1/(τ_LP · n_c) in rad/s → Hz
        1.0 / (tau_lp_s * n_c) / (2.0 * PI)
    }

    /// Check whether the system has crossed the condensation threshold.
    pub fn is_above_threshold(&self) -> bool {
        self.pumping_rate > self.threshold_pump_rate()
    }
}

// ─── PolaritonBistability ────────────────────────────────────────────────────

/// Polariton optical bistability from the interaction-induced nonlinearity.
///
/// When driving a polariton system with a coherent laser slightly blue-detuned
/// from the lower polariton resonance, the interaction blueshift shifts the LP
/// into resonance — creating a self-consistent (bistable) feedback.
///
/// The steady-state equation for polariton density n is:
///
/// ```text
/// n × [(Γ/2)² + (δ − g·n)²] = P_in
/// ```
///
/// where δ is the laser-LP detuning, g is the interaction strength, Γ is the
/// LP linewidth, and P_in is the incident power (in normalised units).
/// This cubic equation has 1 or 3 real solutions.
#[derive(Debug, Clone)]
pub struct PolaritonBistability {
    /// Polariton-polariton interaction coefficient g (meV·µm²).
    pub interaction_coeff: f64,
    /// LP lifetime τ_LP (ps) → Γ = 1/τ_LP (ps⁻¹).
    pub lifetime_ps: f64,
    /// Laser-LP detuning δ = ω_laser − ω_LP (meV); positive means blue-detuned.
    pub detuning_mev: f64,
    /// LP linewidth HWHM Γ/2 (meV).
    pub linewidth_mev: f64,
}

impl PolaritonBistability {
    /// Half linewidth Γ/2 (meV) from the lifetime.
    fn half_linewidth_mev(&self) -> f64 {
        self.linewidth_mev
    }

    /// Solve the cubic steady-state equation for polariton density.
    ///
    /// Returns 1 or 3 real solutions (sorted ascending).
    /// Uses Cardano's method for the depressed cubic.
    pub fn steady_states(&self, pump_power: f64) -> Vec<f64> {
        let delta = self.detuning_mev;
        let g = self.interaction_coeff;
        let gamma_half = self.half_linewidth_mev();
        let p = pump_power;

        // Expand n · [(Γ/2)² + (δ - g·n)²] = P
        // → g²n³ - 2gδ n² + (δ² + Γ²/4)n - P = 0
        // Coefficients: g²·n³ + b·n² + c·n + d = 0
        let a3 = g * g;
        let a2 = -2.0 * g * delta;
        let a1 = delta * delta + gamma_half * gamma_half;
        let a0 = -p;

        if a3.abs() < f64::EPSILON {
            // Degenerate — linear
            if a1.abs() > f64::EPSILON {
                return vec![(p / a1).max(0.0)];
            }
            return vec![0.0];
        }

        // Normalise: t³ + pt² + qt + r = 0
        let p_coef = a2 / a3;
        let q_coef = a1 / a3;
        let r_coef = a0 / a3;

        // Depress: t = u - p/3
        let p3 = p_coef / 3.0;
        let qq = q_coef - p_coef * p_coef / 3.0;
        let rr = r_coef + 2.0 * p_coef.powi(3) / 27.0 - p_coef * q_coef / 3.0;

        // Discriminant
        let disc = -(4.0 * qq.powi(3) + 27.0 * rr * rr);

        if disc > 0.0 {
            // Three distinct real roots (bistable regime)
            let m = 2.0 * (-qq / 3.0).sqrt();
            let theta = (3.0 * rr / (2.0 * qq) * (-3.0 / qq).sqrt()).acos() / 3.0;
            let mut roots = vec![
                m * theta.cos() - p3,
                m * (theta + 2.0 * PI / 3.0).cos() - p3,
                m * (theta + 4.0 * PI / 3.0).cos() - p3,
            ];
            roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            roots.into_iter().filter(|&x| x >= -1e-6).map(|x| x.max(0.0)).collect()
        } else {
            // One real root (monostable regime)
            let aa = -(rr / 2.0 + (rr * rr / 4.0 + qq.powi(3) / 27.0).abs().sqrt().copysign(rr))
                .abs()
                .cbrt()
                .copysign(-(rr / 2.0 + (rr * rr / 4.0 + qq.powi(3) / 27.0).abs().sqrt().copysign(rr)));
            let bb = if aa.abs() > f64::EPSILON { -qq / (3.0 * aa) } else { 0.0 };
            let root = aa + bb - p3;
            vec![root.max(0.0)]
        }
    }

    /// Check whether the system is in the bistable (hysteresis) regime.
    ///
    /// Bistability requires that the detuning exceeds sqrt(3) × Γ/2.
    pub fn is_bistable(&self) -> bool {
        self.detuning_mev.abs() > 3.0_f64.sqrt() * self.linewidth_mev
    }

    /// Switching thresholds (lower switch-on, upper switch-off pump powers).
    ///
    /// Returns `None` if the system is monostable.
    pub fn switching_thresholds(&self) -> Option<(f64, f64)> {
        if !self.is_bistable() {
            return None;
        }
        let delta = self.detuning_mev;
        let g = self.interaction_coeff;
        let gamma_half = self.half_linewidth_mev();

        // Turning points of P(n) curve: dP/dn = 0
        // P(n) = n · [(Γ/2)² + (δ - g·n)²]
        // dP/dn = (Γ/2)² + (δ - g·n)² - 2g·n·(δ - g·n) = 0
        // → 3g²n² - 4gδ·n + δ² + Γ²/4 = 0
        let a = 3.0 * g * g;
        let b = -4.0 * g * delta;
        let c = delta * delta + gamma_half * gamma_half;
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 {
            return None;
        }
        let n1 = (-b - disc.sqrt()) / (2.0 * a);
        let n2 = (-b + disc.sqrt()) / (2.0 * a);
        if n1 < 0.0 || n2 < 0.0 {
            return None;
        }
        // P values at turning points
        let p_fn = |n: f64| n * (gamma_half * gamma_half + (delta - g * n).powi(2));
        Some((p_fn(n1).min(p_fn(n2)), p_fn(n1).max(p_fn(n2))))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn standard_condensate(pump: f64) -> PolaritonCondensate {
        PolaritonCondensate {
            rabi_splitting_ev: 0.005,        // 5 meV Rabi splitting
            polariton_lifetime_ps: 10.0,     // 10 ps LP lifetime
            reservoir_lifetime_ps: 100.0,    // 100 ps reservoir lifetime
            relaxation_rate_ps: 0.01,        // W_R = 0.01 ps⁻¹
            blueshift_coeff: 2e-3,           // g_X interaction
            pumping_rate: pump,
            temperature_k: 10.0,
        }
    }

    #[test]
    fn threshold_positive_and_finite() {
        let cond = standard_condensate(0.1);
        let p_th = cond.threshold_pump_rate();
        assert!(p_th > 0.0 && p_th.is_finite(), "Threshold should be finite positive, got {}", p_th);
    }

    #[test]
    fn below_threshold_no_condensate() {
        let p_th = standard_condensate(0.0).threshold_pump_rate();
        let cond = standard_condensate(p_th * 0.5);
        let n_c = cond.condensate_density();
        assert!(n_c == 0.0, "Below threshold: condensate should be 0, got {}", n_c);
    }

    #[test]
    fn above_threshold_condensate_grows() {
        let p_th = standard_condensate(0.0).threshold_pump_rate();
        let cond1 = standard_condensate(p_th * 2.0);
        let cond2 = standard_condensate(p_th * 4.0);
        assert!(
            cond2.condensate_density() > cond1.condensate_density(),
            "Condensate density should increase with pump"
        );
    }

    #[test]
    fn time_evolution_reaches_steady_state() {
        let p_th = standard_condensate(0.0).threshold_pump_rate();
        let cond = standard_condensate(p_th * 3.0);
        let traj = cond.time_evolution(2000.0, 0.1);
        assert!(!traj.is_empty());
        // Final condensate density should be positive and close to steady-state
        let (_, n_c_final, _) = traj.last().copied().unwrap_or((0.0, 0.0, 0.0));
        let n_c_ss = cond.condensate_density();
        // Allow 20% relative tolerance (Euler integration, not exact)
        assert!(
            (n_c_final - n_c_ss).abs() / (n_c_ss + 1e-10) < 0.2,
            "Time evolution n_c={} vs SS n_c={}", n_c_final, n_c_ss
        );
    }

    #[test]
    fn li_curve_monotonic_above_threshold() {
        let p_th = standard_condensate(0.0).threshold_pump_rate();
        let cond = standard_condensate(0.0);
        let curve = cond.li_curve((p_th * 0.5, p_th * 5.0), 50);
        assert_eq!(curve.len(), 50);
        // Check non-negativity
        for &(_, flux) in &curve {
            assert!(flux >= 0.0, "Negative flux in L-I curve");
        }
        // Output should be higher at high pump than at threshold
        let flux_low = curve.first().map(|&(_, f)| f).unwrap_or(0.0);
        let flux_high = curve.last().map(|&(_, f)| f).unwrap_or(0.0);
        assert!(flux_high > flux_low, "L-I curve not increasing");
    }

    #[test]
    fn linewidth_narrows_above_threshold() {
        let p_th = standard_condensate(0.0).threshold_pump_rate();
        let cond_low = standard_condensate(p_th * 1.5);
        let cond_high = standard_condensate(p_th * 10.0);
        let lw_low = cond_low.linewidth_hz();
        let lw_high = cond_high.linewidth_hz();
        assert!(lw_high < lw_low, "Linewidth should narrow with increasing pump");
        assert!(lw_low.is_finite());
    }

    #[test]
    fn bistability_criterion() {
        // Bistable: δ > √3 × Γ/2
        let bistable = PolaritonBistability {
            interaction_coeff: 0.01,
            lifetime_ps: 10.0,
            detuning_mev: 2.0,    // large detuning
            linewidth_mev: 0.5,   // narrow linewidth → √3 × 0.5 ≈ 0.87 < 2.0
        };
        assert!(bistable.is_bistable(), "Should be bistable");

        // Monostable: δ < √3 × Γ/2
        let monostable = PolaritonBistability {
            interaction_coeff: 0.01,
            lifetime_ps: 10.0,
            detuning_mev: 0.5,    // small detuning
            linewidth_mev: 1.0,   // √3 × 1.0 ≈ 1.73 > 0.5
        };
        assert!(!monostable.is_bistable(), "Should be monostable");
    }

    #[test]
    fn steady_states_monostable_count() {
        let mono = PolaritonBistability {
            interaction_coeff: 0.01,
            lifetime_ps: 10.0,
            detuning_mev: 0.3,
            linewidth_mev: 1.0,
        };
        let roots = mono.steady_states(0.5);
        assert_eq!(roots.len(), 1, "Monostable regime should give 1 root, got {}", roots.len());
    }
}
