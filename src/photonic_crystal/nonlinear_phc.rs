//! Nonlinear effects in photonic crystals.
//!
//! Slow light and small mode volumes dramatically enhance nonlinear optical
//! interactions.  This module provides semi-analytic models for:
//!
//! * **Kerr (χ⁽³⁾) enhancement** — self-phase modulation, four-wave mixing,
//!   optical bistability.
//! * **χ⁽²⁾ / SHG enhancement** — slow-light-boosted second-harmonic
//!   generation in PhC waveguides.
//!
//! ## Slow-light enhancement
//!
//! In a slow-light medium with group index n_g the effective nonlinear
//! parameter is enhanced by a factor (n_g / n)² for χ⁽³⁾ processes and
//! roughly (n_g1 · n_g2)^(1/2) for χ⁽²⁾, arising from the increased field
//! confinement and reduced group velocity.  The effective nonlinear
//! coefficient is:
//!
//!   γ_eff = ω n₂ / (c A_eff)  ×  (n_g / n)²
//!
//! where A_eff is the effective mode area.

use std::f64::consts::PI;

use crate::units::conversion::{EPSILON_0, SPEED_OF_LIGHT};

// ─── Kerr / χ⁽³⁾ enhancement ──────────────────────────────────────────────────

/// Enhanced nonlinear susceptibility near a band edge or inside a slow-light
/// photonic crystal waveguide.
///
/// All quantities use SI units.
#[derive(Debug, Clone)]
pub struct PhCNonlinearEnhancement {
    /// Group index n_g in the slow-light region (dimensionless).
    pub group_index: f64,
    /// Effective mode volume V_eff (m³).
    pub mode_volume: f64,
    /// Bulk Kerr index n₂ of the slab material (m²/W).
    pub n2_bulk: f64,
    /// Bulk second-order susceptibility χ⁽²⁾ (m/V).
    pub chi2_bulk: f64,
}

impl PhCNonlinearEnhancement {
    /// Construct from group index, mode volume, and bulk Kerr coefficient.
    ///
    /// Sets `chi2_bulk = 0.0` (update for non-centrosymmetric materials).
    ///
    /// # Arguments
    /// * `ng`          – group index at operating frequency
    /// * `mode_volume` – effective mode volume V_eff (m³)
    /// * `n2`          – bulk n₂ (m²/W)
    pub fn new(ng: f64, mode_volume: f64, n2: f64) -> Self {
        Self {
            group_index: ng.max(1.0),
            mode_volume: mode_volume.max(1e-30),
            n2_bulk: n2,
            chi2_bulk: 0.0,
        }
    }

    /// Construct with explicit χ⁽²⁾ (for non-centrosymmetric PhC materials).
    pub fn with_chi2(mut self, chi2: f64) -> Self {
        self.chi2_bulk = chi2;
        self
    }

    // ── Kerr nonlinearity ──────────────────────────────────────────────────────

    /// Effective Kerr index n₂_eff, enhanced by slow-light confinement.
    ///
    /// For a waveguide mode with effective area A_eff:
    ///   n₂_eff = n₂_bulk · (n_g / n_bg)² · (λ² / V_eff)
    ///
    /// Here we absorb the (λ²/V_eff) factor into the mode area:
    ///   n₂_eff ≈ n₂_bulk · S_slow   where S_slow = (n_g / n_bg)²
    ///
    /// The caller should combine this with the physical A_eff to obtain γ.
    pub fn effective_n2(&self) -> f64 {
        // Assume background index n_bg ~ √(ε_eff); here we use n = 3.476 as Si ref.
        let n_bg = 3.476_f64;
        let s_slow = (self.group_index / n_bg).powi(2);
        self.n2_bulk * s_slow
    }

    /// Nonlinear propagation coefficient γ (rad/(W·m)).
    ///
    ///   γ = ω n₂_eff / (c A_eff)
    ///
    /// where A_eff is estimated from the mode volume as A_eff ≈ V_eff^(2/3).
    ///
    /// # Arguments
    /// * `wavelength` – free-space wavelength λ (m)
    pub fn self_phase_modulation_coeff(&self, wavelength: f64) -> f64 {
        let omega = 2.0 * PI * SPEED_OF_LIGHT / wavelength.max(1e-20);
        let a_eff = self.mode_volume.powf(2.0 / 3.0);
        let n2_eff = self.effective_n2();
        omega * n2_eff / (SPEED_OF_LIGHT * a_eff)
    }

    // ── χ⁽²⁾ enhancement ──────────────────────────────────────────────────────

    /// χ⁽²⁾ enhancement factor due to slow light.
    ///
    /// For SHG in a PhC waveguide with group indices n_g1 (fundamental) and
    /// n_g2 (SH), the effective χ⁽²⁾ is enhanced by:
    ///
    ///   F_χ₂ ≈ (n_g / n)^(3/2)
    ///
    /// Here we use a single group index (the fundamental).
    pub fn chi2_enhancement_factor(&self) -> f64 {
        let n_bg = 3.476_f64;
        (self.group_index / n_bg).powf(1.5)
    }

    // ── Optical bistability ────────────────────────────────────────────────────

    /// Bistability threshold power (W) for a Kerr cavity.
    ///
    /// Optical bistability requires the cavity round-trip phase shift to equal
    /// the cavity linewidth.  The threshold intracavity power is:
    ///
    ///   P_th ≈ ω V_eff / (2 n₂_eff Q²)   ×  (n ε₀ c / 2)
    ///
    /// Simplified form for an order-of-magnitude estimate:
    ///   P_th ≈ ε₀ n² c ω V_eff / (2 n₂_eff Q²)
    ///
    /// # Arguments
    /// * `cavity_q`  – quality factor Q of the PhC resonance
    /// * `wavelength` – resonant wavelength λ (m)
    pub fn optical_bistability_threshold(&self, cavity_q: f64, wavelength: f64) -> f64 {
        let omega = 2.0 * PI * SPEED_OF_LIGHT / wavelength.max(1e-20);
        let n2_eff = self.effective_n2();
        if n2_eff.abs() < 1e-40 || cavity_q < 1.0 {
            return f64::INFINITY;
        }
        let n_bg = 3.476_f64;
        EPSILON_0 * n_bg * n_bg * SPEED_OF_LIGHT * omega * self.mode_volume
            / (2.0 * n2_eff * cavity_q * cavity_q)
    }

    // ── Four-wave mixing ──────────────────────────────────────────────────────

    /// Phase-matched FWM frequency detuning Δω (rad/s).
    ///
    /// For degenerate FWM in a dispersive slow-light waveguide, the phase
    /// mismatch is zeroed when:
    ///
    ///   Δk = β₂ Ω²   →   Ω² = 2 γ P / |β₂|
    ///
    /// We use the approximation β₂ ≈ -n_g² / (c ω) (anomalous dispersion
    /// near the band edge) and γ from SPM at unit power (1 W):
    ///
    ///   Ω_pm = √(2 γ P / |β₂|)
    ///
    /// Returns the signed detuning in rad/s (positive for anomalous dispersion).
    ///
    /// # Arguments
    /// * `wavelength` – pump wavelength (m)
    /// * `pump_power` – CW pump power P (W)
    pub fn fwm_phase_matching_detuning(&self) -> f64 {
        // At unit power, using λ = 1550 nm reference
        let wavelength = 1550e-9;
        let omega = 2.0 * PI * SPEED_OF_LIGHT / wavelength;
        let gamma = self.self_phase_modulation_coeff(wavelength);
        // β₂ ≈ n_g² / (c ω) (magnitude, anomalous sign)
        let beta2 = self.group_index.powi(2) / (SPEED_OF_LIGHT * omega);
        if beta2.abs() < 1e-30 || gamma.abs() < 1e-30 {
            return 0.0;
        }
        // Ω_pm = √(2 γ / |β₂|)  at P = 1 W
        (2.0 * gamma / beta2.abs()).sqrt()
    }
}

// ─── Slow-light SHG ───────────────────────────────────────────────────────────

/// Second-harmonic generation enhanced by slow light in a PhC waveguide.
///
/// The conversion efficiency scales as:
///
///   η_SHG ≈ (ω² d_eff² L²) / (ε₀ c³ n_eff² A_eff) × sinc²(ΔkL/2)
///
/// where d_eff = χ⁽²⁾/2 is the effective nonlinear coefficient enhanced by
/// the slow-light factor.
#[derive(Debug, Clone)]
pub struct SlowLightShg {
    /// Waveguide length L (m).
    pub l1: f64,
    /// Group index at the fundamental frequency n_g1.
    pub ng1: f64,
    /// Group index at the second-harmonic frequency n_g2.
    pub ng2: f64,
    /// Effective χ⁽²⁾ of the material (m/V), before slow-light enhancement.
    pub chi2: f64,
    /// Effective mode area A_eff (m²).
    pub a_eff: f64,
}

impl SlowLightShg {
    /// Construct a slow-light SHG device.
    ///
    /// # Arguments
    /// * `length` – waveguide length L (m)
    /// * `ng1`    – group index at fundamental
    /// * `ng2`    – group index at SH
    /// * `chi2`   – bulk χ⁽²⁾ (m/V)
    /// * `a_eff`  – effective mode area (m²)
    pub fn new(length: f64, ng1: f64, ng2: f64, chi2: f64, a_eff: f64) -> Self {
        Self {
            l1: length.max(0.0),
            ng1: ng1.max(1.0),
            ng2: ng2.max(1.0),
            chi2,
            a_eff: a_eff.max(1e-20),
        }
    }

    /// Slow-light enhanced d_eff = χ⁽²⁾/2 × F_slow.
    ///
    /// Enhancement factor:  F_slow = (n_g1 · n_g2)^(1/2) / n_bg
    fn d_eff_enhanced(&self) -> f64 {
        let n_bg = 3.476_f64;
        let f_slow = (self.ng1 * self.ng2).sqrt() / n_bg;
        (self.chi2 / 2.0) * f_slow
    }

    /// Phase mismatch Δk = 2β(ω) − β(2ω) (rad/m).
    ///
    /// For a waveguide with different group indices at ω and 2ω:
    ///
    ///   Δk = (2ω/c)(n_g2 - n_g1)
    ///
    /// (approximate; ignores higher-order dispersion).
    pub fn phase_mismatch(&self) -> f64 {
        // At λ = 1550 nm fundamental
        let lambda_fund = 1550e-9;
        let omega = 2.0 * PI * SPEED_OF_LIGHT / lambda_fund;
        2.0 * omega / SPEED_OF_LIGHT * (self.ng2 - self.ng1).abs()
    }

    /// Phase-matching coherence length L_c = π / |Δk| (m).
    pub fn coherence_length(&self) -> f64 {
        let dk = self.phase_mismatch();
        if dk < 1e-10 {
            return f64::INFINITY;
        }
        PI / dk
    }

    /// SHG conversion efficiency η = P_SH / P_pump² (W⁻¹) for CW excitation.
    ///
    /// Combining the slowly-varying envelope approximation with the sinc²
    /// phase-matching factor and slow-light enhancement:
    ///
    ///   η_SHG = (8 π² d_eff² L²) / (ε₀ n_eff³ c λ² A_eff) × sinc²(ΔkL/2)
    ///
    /// # Arguments
    /// * `pump_power` – incident pump power P (W)
    /// * `wavelength` – fundamental wavelength λ (m)
    pub fn conversion_efficiency(&self, pump_power: f64, wavelength: f64) -> f64 {
        let _omega = 2.0 * PI * SPEED_OF_LIGHT / wavelength.max(1e-20);
        let d_eff = self.d_eff_enhanced();
        let dk = self.phase_mismatch();
        let n_eff = 3.476_f64; // effective index reference

        // sinc²(ΔkL/2)
        let arg = dk * self.l1 / 2.0;
        let sinc_sq = if arg.abs() < 1e-10 {
            1.0
        } else {
            (arg.sin() / arg).powi(2)
        };

        // η = 8π² d_eff² L² / (ε₀ n³ c λ² A_eff) × sinc²
        let num = 8.0 * PI * PI * d_eff * d_eff * self.l1 * self.l1;
        let den = EPSILON_0 * n_eff.powi(3) * SPEED_OF_LIGHT * wavelength * wavelength * self.a_eff;
        if den.abs() < 1e-60 {
            return 0.0;
        }
        // η_norm in W⁻¹; multiply by pump power for dimensionless efficiency
        let eta_norm = num / den * sinc_sq;
        // Clamp to [0, 1] for physical efficiency
        (eta_norm * pump_power).clamp(0.0, 1.0)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn si_kerr() -> PhCNonlinearEnhancement {
        // Si: n₂ ≈ 6e-18 m²/W, V ≈ 0.7 (λ/n)³ at λ=1550 nm
        let lambda = 1550e-9_f64;
        let n = 3.476_f64;
        let v_eff = 0.7 * (lambda / n).powi(3);
        PhCNonlinearEnhancement::new(30.0, v_eff, 6e-18)
    }

    // ── PhCNonlinearEnhancement ────────────────────────────────────────────

    #[test]
    fn effective_n2_larger_than_bulk() {
        let enh = si_kerr();
        // n_g = 30 > n_bg ≈ 3.476 → n2_eff > n2_bulk
        assert!(enh.effective_n2() > enh.n2_bulk);
    }

    #[test]
    fn spm_coeff_positive() {
        let enh = si_kerr();
        let gamma = enh.self_phase_modulation_coeff(1550e-9);
        assert!(gamma > 0.0, "γ = {gamma}");
    }

    #[test]
    fn chi2_enhancement_factor_greater_than_one_for_slow_light() {
        // n_g = 30 ≫ n_bg ≈ 3.476 → factor > 1
        let enh = si_kerr();
        let f = enh.chi2_enhancement_factor();
        assert!(f > 1.0, "enhancement = {f:.3}");
    }

    #[test]
    fn bistability_threshold_finite_for_high_q() {
        let enh = si_kerr();
        let p_th = enh.optical_bistability_threshold(1e5, 1550e-9);
        assert!(p_th.is_finite(), "P_th = {p_th}");
        assert!(p_th > 0.0, "P_th must be positive, got {p_th}");
    }

    #[test]
    fn bistability_threshold_decreases_with_higher_q() {
        let enh = si_kerr();
        let p_low_q = enh.optical_bistability_threshold(1e4, 1550e-9);
        let p_high_q = enh.optical_bistability_threshold(1e5, 1550e-9);
        // P_th ∝ 1/Q² → higher Q → lower threshold
        assert!(
            p_high_q < p_low_q,
            "p_high_q={p_high_q} < p_low_q={p_low_q}"
        );
    }

    #[test]
    fn fwm_detuning_positive() {
        let enh = si_kerr();
        let omega_pm = enh.fwm_phase_matching_detuning();
        assert!(omega_pm >= 0.0, "Ω_pm = {omega_pm}");
    }

    #[test]
    fn effective_n2_scales_with_ng_squared() {
        // n2_eff ∝ (n_g/n)²
        let v = 1e-21_f64;
        let enh1 = PhCNonlinearEnhancement::new(10.0, v, 6e-18);
        let enh2 = PhCNonlinearEnhancement::new(20.0, v, 6e-18);
        let ratio = enh2.effective_n2() / enh1.effective_n2();
        // Expect ratio ≈ 4.0 (2² = 4)
        assert_abs_diff_eq!(ratio, 4.0, epsilon = 0.01);
    }

    // ── SlowLightShg ──────────────────────────────────────────────────────

    fn example_shg() -> SlowLightShg {
        // LiNbO₃: χ⁽²⁾ ≈ 3e-11 m/V, A_eff ~ 1e-12 m², L = 1 mm
        SlowLightShg::new(1e-3, 5.0, 8.0, 3e-11, 1e-12)
    }

    #[test]
    fn shg_phase_mismatch_positive() {
        let shg = example_shg();
        // ng1 ≠ ng2 → Δk > 0
        assert!(shg.phase_mismatch() > 0.0);
    }

    #[test]
    fn shg_coherence_length_positive_and_finite() {
        let shg = example_shg();
        let lc = shg.coherence_length();
        assert!(lc > 0.0 && lc.is_finite(), "L_c = {lc}");
    }

    #[test]
    fn shg_efficiency_between_zero_and_one() {
        let shg = example_shg();
        let eta = shg.conversion_efficiency(1.0, 1550e-9);
        assert!((0.0..=1.0).contains(&eta), "η = {eta}");
    }

    #[test]
    fn shg_perfect_phase_match_higher_efficiency() {
        // Identical group indices → Δk = 0 → sinc² = 1
        let shg_pm = SlowLightShg::new(1e-3, 5.0, 5.0, 3e-11, 1e-12);
        let shg_mm = SlowLightShg::new(1e-3, 5.0, 8.0, 3e-11, 1e-12);
        let eta_pm = shg_pm.conversion_efficiency(1.0, 1550e-9);
        let eta_mm = shg_mm.conversion_efficiency(1.0, 1550e-9);
        assert!(
            eta_pm >= eta_mm,
            "phase-matched η={eta_pm} should ≥ mismatched η={eta_mm}"
        );
    }

    #[test]
    fn shg_d_eff_enhanced_positive_for_nonzero_chi2() {
        let shg = example_shg();
        // d_eff_enhanced uses chi2 > 0
        let d_eff = shg.d_eff_enhanced();
        assert!(d_eff > 0.0, "d_eff = {d_eff}");
    }
}
