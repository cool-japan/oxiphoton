//! Four-wave mixing (FWM) in optical fibers.
//!
//! FWM is a third-order nonlinear interaction in which three photons at
//! frequencies ω₁, ω₂, ω₃ combine to generate a fourth at ω₄:
//!   ω₄ = ω₁ + ω₂ - ω₃
//!
//! For degenerate FWM (ω₁ = ω₂ = ωp, pump):
//!   ω_signal = ωp + Ω
//!   ω_idler  = ωp - Ω
//!
//! Phase matching condition:
//!   Δk = β(ωs) + β(ωi) - 2β(ωp) + 2γP_p = 0
//!
//! For D ≈ 0 (near ZDW), Δk ≈ -β₂·Ω² + 2γP_p
//! → Phase matching bandwidth: Ω²_pm = 2γP_p / |β₂|
//!
//! FWM efficiency (undepleted pump approximation):
//!   η_FWM = (γ·P_p·L_eff)² · sinc²(Δk·L/2)

use std::f64::consts::PI;

/// Four-wave mixing model (degenerate: single pump).
#[derive(Debug, Clone, Copy)]
pub struct FwmFiber {
    /// Nonlinear coefficient γ (W⁻¹m⁻¹)
    pub gamma: f64,
    /// Group velocity dispersion β₂ (s²/m) at pump wavelength
    pub beta2: f64,
    /// Fiber attenuation α (m⁻¹)
    pub alpha: f64,
    /// Fiber length L (m)
    pub length: f64,
}

impl FwmFiber {
    /// Create FWM model.
    pub fn new(gamma: f64, beta2: f64, alpha_db_per_km: f64, length_m: f64) -> Self {
        let alpha = alpha_db_per_km * 1e-3 / (10.0 / std::f64::consts::LN_10);
        Self {
            gamma,
            beta2,
            alpha,
            length: length_m,
        }
    }

    /// Effective length.
    pub fn effective_length(&self) -> f64 {
        if self.alpha < 1e-30 {
            return self.length;
        }
        (1.0 - (-self.alpha * self.length).exp()) / self.alpha
    }

    /// Linear phase mismatch Δk_linear (m⁻¹) at frequency offset Ω (rad/s).
    ///
    ///   Δk_linear = β₂·Ω²
    pub fn phase_mismatch_linear(&self, omega_offset: f64) -> f64 {
        self.beta2 * omega_offset * omega_offset
    }

    /// Total phase mismatch Δk = Δk_linear + 2γP_p.
    pub fn total_phase_mismatch(&self, omega_offset: f64, pump_power_w: f64) -> f64 {
        self.phase_mismatch_linear(omega_offset) + 2.0 * self.gamma * pump_power_w
    }

    /// Phase-matching frequency offset Ω_pm (rad/s) for given pump power.
    ///
    ///   Ω_pm = √(-2γP_p / β₂)  [requires β₂ < 0, i.e., anomalous dispersion]
    pub fn phase_matching_offset(&self, pump_power_w: f64) -> Option<f64> {
        let discriminant = -2.0 * self.gamma * pump_power_w / self.beta2;
        if discriminant > 0.0 {
            Some(discriminant.sqrt())
        } else {
            None
        }
    }

    /// FWM efficiency η (linear, not dB) for parametric amplification.
    ///
    ///   η = (γ·P_p)² · sinh²(g·L_eff) / g²   with g = √((γP_p)² - (Δk/2)²)
    ///   (parametric gain coefficient g)
    pub fn parametric_gain(&self, omega_offset: f64, pump_power_w: f64) -> f64 {
        let dk = self.total_phase_mismatch(omega_offset, pump_power_w) / 2.0;
        let gp = self.gamma * pump_power_w;
        let g_sq = gp * gp - dk * dk;
        let l_eff = self.effective_length();
        if g_sq > 0.0 {
            let g = g_sq.sqrt();
            (gp / g * g.sinh().abs() * l_eff.signum()).powi(2).max(0.0) * (g * l_eff).sinh().powi(2)
                / (g * l_eff).powi(2)
                * (g * l_eff).powi(2)
        } else {
            // Phase-mismatched: oscillatory
            let kappa = (-g_sq).sqrt();
            (gp * (kappa * l_eff).sin() / kappa).powi(2)
        }
    }

    /// FWM sidebands wavelengths (m) given pump wavelength and offset.
    pub fn sideband_wavelengths(&self, pump_wavelength_m: f64, omega_offset: f64) -> (f64, f64) {
        use crate::units::conversion::SPEED_OF_LIGHT;
        let omega_p = 2.0 * PI * SPEED_OF_LIGHT / pump_wavelength_m;
        let omega_s = omega_p + omega_offset;
        let omega_i = omega_p - omega_offset;
        let lambda_s = 2.0 * PI * SPEED_OF_LIGHT / omega_s;
        let lambda_i = 2.0 * PI * SPEED_OF_LIGHT / omega_i;
        (lambda_s, lambda_i)
    }

    /// Conversion efficiency (dB) for signal after FWM interaction.
    pub fn conversion_efficiency_db(&self, omega_offset: f64, pump_power_w: f64) -> f64 {
        let eta = self.parametric_gain(omega_offset, pump_power_w);
        10.0 * eta.log10()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// FwmPhaseMatching: non-degenerate (two-pump) phase matching
// ──────────────────────────────────────────────────────────────────────────────

/// Phase-matching analysis for non-degenerate FWM with two pumps.
///
/// Energy conservation: ω_p1 + ω_p2 = ω_s + ω_i
/// Momentum conservation: β_p1 + β_p2 - β_s - β_i + 2γP = 0
///
/// `beta1_*` are the inverse group velocities (s/m) at each wavelength.
#[derive(Debug, Clone, Copy)]
pub struct FwmPhaseMatching {
    /// Inverse group velocity at pump 1 (s/m)
    pub beta1_pump1: f64,
    /// Inverse group velocity at pump 2 (s/m)
    pub beta1_pump2: f64,
    /// Inverse group velocity at signal (s/m)
    pub beta1_signal: f64,
    /// Inverse group velocity at idler (s/m)
    pub beta1_idler: f64,
}

impl FwmPhaseMatching {
    /// Create a phase-matching analysis object.
    pub fn new(beta1_pump1: f64, beta1_pump2: f64, beta1_signal: f64, beta1_idler: f64) -> Self {
        Self {
            beta1_pump1,
            beta1_pump2,
            beta1_signal,
            beta1_idler,
        }
    }

    /// Linear phase mismatch Δβ_linear (m⁻¹) from group-velocity dispersion.
    ///
    ///   Δβ_linear = β₂ · (Ω_s² + Ω_i² - Ω_p1² - Ω_p2²) / 2   (Taylor expansion)
    ///
    /// Approximated here via the provided `dk_linear` parameter (pre-computed
    /// externally or from Taylor-expanded dispersion).
    pub fn phase_mismatch(&self, beta2: f64, dk_linear: f64, gamma: f64, pump_power: f64) -> f64 {
        // Total: Δβ = Δβ_linear + Δβ_GVD·β₂ + 2γP (nonlinear contribution)
        let gvd_contribution = beta2
            * (self.beta1_signal * self.beta1_signal + self.beta1_idler * self.beta1_idler
                - self.beta1_pump1 * self.beta1_pump1
                - self.beta1_pump2 * self.beta1_pump2)
            / 2.0;
        dk_linear + gvd_contribution + 2.0 * gamma * pump_power
    }

    /// Returns `true` when |Δβ_total| ≤ tolerance.
    pub fn is_phase_matched(
        &self,
        beta2: f64,
        gamma: f64,
        pump_power: f64,
        tolerance: f64,
    ) -> bool {
        self.phase_mismatch(beta2, 0.0, gamma, pump_power).abs() <= tolerance
    }

    /// Idler wavelength from energy conservation.
    ///
    ///   1/λ_i = 1/λ_p1 + 1/λ_p2 - 1/λ_s
    ///
    /// All wavelengths in meters.
    pub fn idler_wavelength(lambda_pump1: f64, lambda_pump2: f64, lambda_signal: f64) -> f64 {
        let inv = 1.0 / lambda_pump1 + 1.0 / lambda_pump2 - 1.0 / lambda_signal;
        if inv.abs() < 1e-30 {
            return f64::INFINITY;
        }
        1.0 / inv
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ParametricAmplifier: OPA/OPG gain model
// ──────────────────────────────────────────────────────────────────────────────

/// Fiber optical parametric amplifier (FOPA) gain model.
///
/// Based on the undepleted-pump approximation for degenerate FWM:
///
///   G_s(L) = cosh²(g·L) + (Δβ/2g)²·sinh²(g·L)
///
/// where g = √((γP)² - (Δβ/2)²) is the parametric gain coefficient.
///
/// For perfect phase matching (Δβ=0): G_s = cosh²(γ·P·L).
#[derive(Debug, Clone, Copy)]
pub struct ParametricAmplifier {
    /// Nonlinear coefficient γ (W⁻¹m⁻¹)
    pub gamma: f64,
    /// Pump power P (W)
    pub pump_power: f64,
    /// Total phase mismatch Δβ (m⁻¹)
    pub phase_mismatch: f64,
}

impl ParametricAmplifier {
    /// Create a parametric amplifier model.
    pub fn new(gamma: f64, pump_power: f64, phase_mismatch: f64) -> Self {
        Self {
            gamma,
            pump_power,
            phase_mismatch,
        }
    }

    /// Parametric gain coefficient g (m⁻¹).
    ///
    ///   g = √((γP)² - (Δβ/2)²)
    ///
    /// Returns 0 if the system is below threshold (no parametric amplification).
    pub fn gain_coefficient(&self) -> f64 {
        let gp = self.gamma * self.pump_power;
        let half_dk = self.phase_mismatch / 2.0;
        let g_sq = gp * gp - half_dk * half_dk;
        if g_sq > 0.0 {
            g_sq.sqrt()
        } else {
            0.0
        }
    }

    /// Signal power gain G = cosh²(g·L) for perfect phase matching (Δβ = 0).
    ///
    /// When Δβ ≠ 0, use the full expression with `signal_gain_full`.
    pub fn signal_gain(&self, length: f64) -> f64 {
        let g = self.gain_coefficient();
        if g < 1e-30 {
            // Below threshold or zero phase mismatch with trivial case:
            // use small-g limit cosh²(gL) ≈ 1 + (gL)²/... → cosh is still valid
            (g * length).cosh().powi(2)
        } else {
            (g * length).cosh().powi(2)
        }
    }

    /// Full signal gain including phase-mismatch penalty.
    ///
    ///   G_s = cosh²(g·L) + (Δβ/(2g))²·sinh²(g·L)   when g > 0
    ///   G_s = 1 + (γP·L)²·sinc²(Δβ·L/2)             when g = 0 (below threshold)
    pub fn signal_gain_full(&self, length: f64) -> f64 {
        let gp = self.gamma * self.pump_power;
        let half_dk = self.phase_mismatch / 2.0;
        let g_sq = gp * gp - half_dk * half_dk;
        if g_sq > 0.0 {
            let g = g_sq.sqrt();
            let cosh_gl = (g * length).cosh();
            let sinh_gl = (g * length).sinh();
            cosh_gl * cosh_gl + (half_dk / g).powi(2) * sinh_gl * sinh_gl
        } else if g_sq < 0.0 {
            // Phase-mismatched oscillation
            let kappa = (-g_sq).sqrt();
            let sin_kl = (kappa * length).sin();
            let cos_kl = (kappa * length).cos();
            cos_kl * cos_kl + (gp / kappa).powi(2) * sin_kl * sin_kl
        } else {
            // g = 0: perfect cancellation at threshold
            1.0 + (gp * length).powi(2)
        }
    }

    /// Phase-matched 3-dB gain bandwidth (Hz).
    ///
    /// Near the phase-matched frequency the gain drops by 3 dB when:
    ///   |β₂| · Ω² / 2 ≈ γP  →  Ω_bw = √(2γP/|β₂|)
    ///
    /// Converts to Hz: f_bw = Ω_bw / (2π).
    ///
    /// `pump_wavelength` is in meters; used with `beta2` (s²/m) to compute
    /// the frequency extent.
    pub fn bandwidth_hz(&self, beta2: f64, pump_wavelength: f64) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        if beta2.abs() < 1e-60 {
            return f64::INFINITY;
        }
        let omega_p = 2.0 * PI * SPEED_OF_LIGHT / pump_wavelength;
        // Phase matching: β₂·Ω² = 2γP_p → Ω = √(2γP/|β₂|)
        let _ = omega_p; // omega_p is the pump angular frequency, used for reference
        let omega_bw = (2.0 * self.gamma * self.pump_power / beta2.abs()).sqrt();
        omega_bw / (2.0 * PI)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwm_phase_mismatch_zero_at_zero_offset() {
        let f = FwmFiber::new(1.3e-3, -20e-27, 0.2, 1e3);
        assert!(f.phase_mismatch_linear(0.0).abs() < 1e-30);
    }

    #[test]
    fn fwm_phase_matching_in_anomalous_dispersion() {
        let f = FwmFiber::new(1.3e-3, -20e-27, 0.2, 1e3);
        let pm = f.phase_matching_offset(100e-3);
        assert!(pm.is_some(), "Should find PM in anomalous dispersion");
    }

    #[test]
    fn fwm_no_phase_matching_in_normal_dispersion() {
        let f = FwmFiber::new(1.3e-3, 20e-27, 0.2, 1e3); // normal dispersion
        let pm = f.phase_matching_offset(100e-3);
        assert!(pm.is_none());
    }

    #[test]
    fn fwm_sideband_wavelengths_symmetric() {
        let f = FwmFiber::new(1.3e-3, -20e-27, 0.2, 1e3);
        let pump_wl = 1550e-9;
        let offset = 2.0 * PI * 1e12; // 1 THz
        let (ls, li) = f.sideband_wavelengths(pump_wl, offset);
        // Signal shorter, idler longer than pump
        assert!(ls < pump_wl, "Signal wavelength should be < pump");
        assert!(li > pump_wl, "Idler wavelength should be > pump");
    }

    #[test]
    fn fwm_effective_length_positive() {
        let f = FwmFiber::new(1.3e-3, -20e-27, 0.2, 80e3);
        assert!(f.effective_length() > 0.0);
    }

    // ── FwmPhaseMatching tests ────────────────────────────────────────────────

    #[test]
    fn idler_wavelength_conservation() {
        // 1/λ_i = 1/λ_p1 + 1/λ_p2 - 1/λ_s
        let lp1 = 1550e-9_f64;
        let lp2 = 1550e-9_f64; // degenerate pump
        let ls = 1540e-9_f64;
        let li = FwmPhaseMatching::idler_wavelength(lp1, lp2, ls);
        let check = 1.0 / li - (1.0 / lp1 + 1.0 / lp2 - 1.0 / ls);
        assert!(
            check.abs() < 1e6,
            "Energy conservation violated: residual={check:.3e}"
        );
        // Idler should be red-shifted compared to pump (signal is blue-shifted)
        assert!(
            li > lp1,
            "Idler should be longer than pump when signal is shorter, li={li:.2e}"
        );
    }

    #[test]
    fn parametric_gain_positive() {
        // Perfect phase matching → cosh²(gL) ≥ 1
        let amp = ParametricAmplifier::new(1e-2, 1.0, 0.0); // γ=0.01, P=1W, Δβ=0
        let g = amp.signal_gain(1e3); // 1 km
        assert!(
            g > 1.0,
            "Parametric gain should exceed 1 with PM and nonzero γ, got {g}"
        );
    }

    #[test]
    fn phase_mismatch_zero() {
        // When β₂ = 0 and γ·P compensates dk_linear: Δβ_total = 2γP + dk_linear
        // For phase matching: dk_linear = -2γP
        let gamma = 1.3e-3;
        let pump = 100e-3; // 100 mW
        let dk_linear = -2.0 * gamma * pump; // exact compensation
        let pm = FwmPhaseMatching::new(5e-9, 5e-9, 5e-9, 5e-9);
        let dm = pm.phase_mismatch(0.0, dk_linear, gamma, pump);
        assert!(dm.abs() < 1e-12, "Phase mismatch should be zero: {dm:.3e}");
    }

    #[test]
    fn parametric_amp_gain_coefficient_zero_below_threshold() {
        // When Δβ/2 > γP, gain_coefficient returns 0
        let amp = ParametricAmplifier::new(1e-3, 0.1, 10.0); // Δβ/2 = 5 >> γP = 1e-4
        let g = amp.gain_coefficient();
        assert_eq!(g, 0.0, "Below threshold g should be 0, got {g}");
    }

    #[test]
    fn parametric_bandwidth_finite_for_nonzero_beta2() {
        let amp = ParametricAmplifier::new(1e-3, 1.0, 0.0);
        let bw = amp.bandwidth_hz(20e-27, 1550e-9);
        assert!(
            bw > 0.0 && bw < f64::INFINITY,
            "Bandwidth should be finite: {bw:.3e}"
        );
    }

    #[test]
    fn idler_wavelength_degenerate_symmetric() {
        // Degenerate case λ_p1=λ_p2=λ_p, λ_s=λ_p-Δ → λ_i=λ_p+Δ (approx for small Δ)
        let lp = 1550e-9_f64;
        let delta = 10e-9_f64;
        let ls = lp - delta;
        let li = FwmPhaseMatching::idler_wavelength(lp, lp, ls);
        // 1/li = 2/lp - 1/ls
        let expected = 1.0 / (2.0 / lp - 1.0 / ls);
        let rel_err = (li - expected).abs() / expected;
        assert!(
            rel_err < 1e-10,
            "Idler wavelength mismatch: got {li:.2e}, expected {expected:.2e}"
        );
    }
}
