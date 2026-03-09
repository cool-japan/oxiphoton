//! WDM transmission impairments: PMD, SRS tilt, XPM, and FWM.
//!
//! Provides analytical models for the major physical impairments in
//! WDM optical transmission systems.
//!
//! # Impairment Summary
//! | Impairment | Type       | Mitigation          |
//! |-----------|-----------|---------------------|
//! | PMD       | Linear     | DSP (MIMO)          |
//! | SRS       | Inelastic  | Power equalization  |
//! | XPM       | Nonlinear  | Dispersion, spacing |
//! | FWM       | Nonlinear  | Dispersion, spacing |
//!
//! # References
//! - G. Agrawal, "Fiber-Optic Communication Systems," 5th ed., Wiley.
//! - ITU-T G.663 — Application-related aspects of optical amplifier devices.

/// Speed of light in vacuum \[m/s\]
const C_M_PER_S: f64 = 2.997_924_58e8;

// ─────────────────────────────────────────────────────────────────────────────
// PmdAnalysis
// ─────────────────────────────────────────────────────────────────────────────

/// Polarization mode dispersion (PMD) analysis for a single fiber link.
///
/// PMD arises from fiber birefringence and causes the two polarization modes
/// to travel at slightly different group velocities. The differential group
/// delay (DGD) is a Maxwellian random variable.
///
/// Mean DGD: `⟨τ⟩ = PMD_coeff × √L`
#[derive(Debug, Clone)]
pub struct PmdAnalysis {
    /// PMD coefficient \[ps/√km\] (typ. 0.01–0.1 ps/√km for modern fiber).
    pub pmd_coefficient_ps_per_sqrt_km: f64,
    /// Fiber link length \[km\].
    pub fiber_length_km: f64,
}

impl PmdAnalysis {
    /// Create a new PMD analysis for the given fiber.
    pub fn new(coeff: f64, length_km: f64) -> Self {
        Self {
            pmd_coefficient_ps_per_sqrt_km: coeff,
            fiber_length_km: length_km,
        }
    }

    /// Mean DGD \[ps\]: `⟨τ⟩ = PMD_coeff × √L`.
    pub fn mean_dgd_ps(&self) -> f64 {
        self.pmd_coefficient_ps_per_sqrt_km * self.fiber_length_km.sqrt()
    }

    /// PMD-induced outage probability.
    ///
    /// For a Maxwellian-distributed DGD with mean ⟨τ⟩, the probability that
    /// the instantaneous DGD exceeds `τ_threshold` is:
    /// ```text
    ///   P_out ≈ erfc(τ_threshold / (⟨τ⟩ × √(8/3π))) / 2
    /// ```
    /// Reference: Foschini & Poole, JLT 9(11), 1991.
    pub fn outage_probability(&self, threshold_ps: f64) -> f64 {
        let mean_dgd = self.mean_dgd_ps();
        if mean_dgd <= 0.0 {
            return 0.0;
        }
        // Maxwellian distribution: argument to erfc
        let arg = threshold_ps / (mean_dgd * (8.0 / (3.0 * std::f64::consts::PI)).sqrt());
        erfc_approx(arg) / 2.0
    }

    /// System DGD tolerance \[ps\].
    ///
    /// Standard rule: DGD tolerance < 10% of the symbol period.
    /// ```text
    ///   τ_tol = 0.1 / baud_rate \[ps\]  (baud in Gbaud)
    /// ```
    pub fn tolerance_ps(&self, symbol_rate_gbaud: f64) -> f64 {
        if symbol_rate_gbaud <= 0.0 {
            return f64::INFINITY;
        }
        0.1 / symbol_rate_gbaud * 1e3 // 0.1/baud [ns] × 1000 = ps
    }

    /// PMD-limited reach \[km\].
    ///
    /// Maximum fiber length before PMD tolerance is exceeded on average:
    /// ```text
    ///   L_max = (DGD_tol / PMD_coeff)²
    /// ```
    pub fn pmd_limited_reach_km(&self, symbol_rate_gbaud: f64) -> f64 {
        let dgd_tol = self.tolerance_ps(symbol_rate_gbaud);
        if self.pmd_coefficient_ps_per_sqrt_km <= 0.0 {
            return f64::INFINITY;
        }
        (dgd_tol / self.pmd_coefficient_ps_per_sqrt_km).powi(2)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SrsTilt
// ─────────────────────────────────────────────────────────────────────────────

/// Stimulated Raman scattering (SRS) tilt in WDM systems.
///
/// In multi-channel WDM, SRS transfers power from shorter-wavelength
/// (higher-frequency) channels to longer-wavelength channels, creating
/// a power tilt across the WDM band.
///
/// The power tilt (dB) is approximately linear with the total WDM bandwidth.
#[derive(Debug, Clone)]
pub struct SrsTilt {
    /// Number of WDM channels.
    pub n_channels: usize,
    /// Channel spacing \[GHz\].
    pub channel_spacing_ghz: f64,
    /// Total launched power into the fiber \[dBm\].
    pub total_power_dbm: f64,
    /// Fiber length \[km\].
    pub fiber_length_km: f64,
    /// Raman gain slope \[dB/(W·km·THz)\] (typ. ≈ 0.04).
    pub raman_gain_slope: f64,
}

impl SrsTilt {
    /// Create a new SRS tilt model.
    pub fn new(n_ch: usize, spacing_ghz: f64, total_power_dbm: f64, length_km: f64) -> Self {
        Self {
            n_channels: n_ch,
            channel_spacing_ghz: spacing_ghz,
            total_power_dbm,
            fiber_length_km: length_km,
            raman_gain_slope: 0.04,
        }
    }

    /// Total WDM bandwidth \[THz\].
    pub fn total_bandwidth_thz(&self) -> f64 {
        self.channel_spacing_ghz * 1e-3 * self.n_channels as f64
    }

    /// Fiber effective length \[km\] (using α = 0.046/km for SMF).
    fn effective_length_km(&self) -> f64 {
        let alpha = 0.046_f64; // 1/km, ≈ 0.2 dB/km
        (1.0 - (-alpha * self.fiber_length_km).exp()) / alpha
    }

    /// Peak-to-peak power tilt \[dB\] due to SRS.
    ///
    /// ```text
    ///   ΔP_tilt ≈ g_R_slope × P_total × Δf_total × L_eff
    /// ```
    /// where `P_total` is in watts, `Δf_total` in THz, `L_eff` in km.
    pub fn power_tilt_db(&self) -> f64 {
        let p_total_w = 1e-3 * 10.0_f64.powf(self.total_power_dbm / 10.0);
        let bw_thz = self.total_bandwidth_thz();
        let l_eff = self.effective_length_km();
        self.raman_gain_slope * p_total_w * bw_thz * l_eff
    }

    /// Power deviation of channel `k` relative to the center channel \[dB\].
    ///
    /// Channels below center lose power; channels above center gain power.
    /// The tilt is linear across the band:
    /// ```text
    ///   ΔP_k = tilt × (k - N/2) / N
    /// ```
    pub fn channel_power_deviation_db(&self, channel: usize) -> f64 {
        if self.n_channels == 0 {
            return 0.0;
        }
        let tilt = self.power_tilt_db();
        let center = (self.n_channels as f64 - 1.0) / 2.0;
        tilt * (channel as f64 - center) / self.n_channels as f64
    }

    /// SRS threshold power \[W\] (single pump → signal transfer).
    ///
    /// Simplified: `P_th ≈ 16 × A_eff / (g_R × L_eff)`.
    /// The bulk Raman gain coefficient `g_R ≈ 1e-13 m/W`.
    pub fn srs_threshold_w(&self, effective_area_um2: f64) -> f64 {
        let g_r = 1e-13_f64; // m/W (peak Raman gain coefficient for silica)
        let a_eff_m2 = effective_area_um2 * 1e-12;
        let l_eff_m = self.effective_length_km() * 1e3;
        16.0 * a_eff_m2 / (g_r * l_eff_m)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// XpmPenalty
// ─────────────────────────────────────────────────────────────────────────────

/// Cross-phase modulation (XPM) penalty model.
///
/// XPM occurs when intensity fluctuations in one WDM channel modulate the
/// phase of adjacent channels through the Kerr effect. In dispersive fibers,
/// the channels walk off each other, reducing the XPM efficiency.
#[derive(Debug, Clone)]
pub struct XpmPenalty {
    /// Number of interfering channels (excluding the probe channel).
    pub n_interfering_channels: usize,
    /// Channel spacing \[GHz\].
    pub channel_spacing_ghz: f64,
    /// Chromatic dispersion \[ps/(nm·km)\].
    pub fiber_dispersion: f64,
    /// Fiber link length \[km\].
    pub fiber_length_km: f64,
    /// Per-channel launch power \[dBm\].
    pub launch_power_dbm: f64,
    /// Fiber nonlinear coefficient \[1/(W·km)\].
    pub nonlinear_coeff: f64,
}

impl XpmPenalty {
    /// Create an XPM penalty model with default nonlinear coefficient (SMF-28).
    pub fn new(n_ch: usize, spacing_ghz: f64, disp: f64, length_km: f64, power_dbm: f64) -> Self {
        Self {
            n_interfering_channels: n_ch,
            channel_spacing_ghz: spacing_ghz,
            fiber_dispersion: disp,
            fiber_length_km: length_km,
            launch_power_dbm: power_dbm,
            nonlinear_coeff: 1.3, // γ ≈ 1.3 W⁻¹km⁻¹ for SMF-28
        }
    }

    /// Walk-off length \[km\] between adjacent channels.
    ///
    /// ```text
    ///   L_W = T_bit / (D × Δλ)
    /// ```
    /// where `Δλ` is the channel wavelength spacing.
    /// Using `Δλ ≈ λ² × Δf / c` and `T_bit = 1/baud_rate`.
    ///
    /// Simplified: `L_W = c / (λ² × D × Δf²) × T_bit`
    pub fn walk_off_length_km(&self) -> f64 {
        // λ = 1550 nm, Δf in THz → Δλ = λ² Δf/c
        let lambda_m = 1.55e-6_f64;
        let df_hz = self.channel_spacing_ghz * 1e9;
        let delta_lambda_m = lambda_m * lambda_m * df_hz / C_M_PER_S;
        let d_s_per_m2 = self.fiber_dispersion.abs() * 1e-6; // ps/(nm·km) → s/m²
        if d_s_per_m2 * delta_lambda_m <= 0.0 {
            return f64::INFINITY;
        }
        // L_W = 1 / (D * Δλ) in km — we need a reference bit period (1 ps here)
        1e-12 / (d_s_per_m2 / 1e3 * delta_lambda_m)
    }

    /// Fiber effective length \[km\].
    fn effective_length_km(&self) -> f64 {
        let alpha = 0.046_f64;
        (1.0 - (-alpha * self.fiber_length_km).exp()) / alpha
    }

    /// XPM phase noise variance \[rad²\].
    ///
    /// Simplified GN model for XPM:
    /// ```text
    ///   σ²_XPM ≈ (2γ)² × P² × L_eff² × N_ch / (|D| × Δf)²
    /// ```
    pub fn phase_variance_rad2(&self) -> f64 {
        let p_w = 1e-3 * 10.0_f64.powf(self.launch_power_dbm / 10.0);
        let gamma = self.nonlinear_coeff;
        let l_eff = self.effective_length_km();
        let df_thz = self.channel_spacing_ghz * 1e-3;
        let d = self.fiber_dispersion.abs();

        if d < 1e-12 || df_thz < 1e-12 {
            return 0.0;
        }
        let n = self.n_interfering_channels as f64;
        // σ² ∝ (2γ·P·L_eff)² × N / (D × Δf)²
        let numerator = (2.0 * gamma * p_w * l_eff).powi(2) * n;
        let denominator = (d * df_thz).powi(2);
        numerator / denominator
    }

    /// XPM-induced SNR penalty \[dB\].
    ///
    /// `penalty = 10·log10(1 + SNR × σ²_XPM)` ≈ `10·log10(σ²_XPM)` in weak-penalty regime.
    pub fn snr_penalty_db(&self) -> f64 {
        let sigma2 = self.phase_variance_rad2();
        if sigma2 <= 0.0 {
            return 0.0;
        }
        // Penalty ≈ 10 log10(1 + σ²_XPM × SNR_target)
        // Simplified: use σ²_XPM directly as noise contribution
        10.0 * (1.0 + sigma2).log10()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FwmEfficiency
// ─────────────────────────────────────────────────────────────────────────────

/// Four-wave mixing (FWM) efficiency model.
///
/// FWM generates new frequency components at `f_ijk = f_i + f_j - f_k`
/// that fall on other WDM channels, causing coherent crosstalk.
/// The efficiency depends strongly on phase mismatch (hence dispersion).
#[derive(Debug, Clone)]
pub struct FwmEfficiency {
    /// Channel spacing \[GHz\].
    pub channel_spacing_ghz: f64,
    /// Fiber dispersion \[ps/(nm·km)\] — higher dispersion suppresses FWM.
    pub fiber_dispersion: f64,
    /// Fiber length \[km\].
    pub fiber_length_km: f64,
    /// Fiber nonlinear coefficient `γ` \[1/(W·km)\].
    pub nonlinear_coeff: f64,
}

impl FwmEfficiency {
    /// Create an FWM efficiency model.
    pub fn new(spacing_ghz: f64, dispersion: f64, length_km: f64, gamma: f64) -> Self {
        Self {
            channel_spacing_ghz: spacing_ghz,
            fiber_dispersion: dispersion,
            fiber_length_km: length_km,
            nonlinear_coeff: gamma,
        }
    }

    /// Phase mismatch `Δβ` \[1/m\].
    ///
    /// For equally spaced channels with spacing `Δf`:
    /// ```text
    ///   Δβ = -2π × λ² × D × Δf² / c   \[1/m\]
    /// ```
    pub fn phase_mismatch(&self) -> f64 {
        let lambda_m = 1.55e-6_f64;
        let df_hz = self.channel_spacing_ghz * 1e9;
        // D in ps/(nm·km) → s/m²: D_SI = D × 1e-6 s/m²
        let d_si = self.fiber_dispersion.abs() * 1e-6;
        // Δβ = -2π λ² D Δf² / c
        -2.0 * std::f64::consts::PI * lambda_m * lambda_m * d_si * df_hz * df_hz / C_M_PER_S
    }

    /// FWM efficiency `η` (dimensionless, range 0–1).
    ///
    /// ```text
    ///   η = α² / (α² + Δβ²) × |1 - exp(-(α + iΔβ)L)|² / L²
    /// ```
    /// Simplified (small α approximation):
    /// ```text
    ///   η ≈ (α·L_eff / (αL + iΔβ·L))² ≈ 1 / (1 + (Δβ/α)²)
    /// ```
    pub fn efficiency(&self, alpha_per_km: f64) -> f64 {
        let alpha_per_m = alpha_per_km / 1e3;
        let delta_beta = self.phase_mismatch().abs();
        let alpha2 = alpha_per_m * alpha_per_m;
        let db2 = delta_beta * delta_beta;
        // η = α² / (α² + Δβ²)
        let base_efficiency = alpha2 / (alpha2 + db2).max(1e-60);
        // Modulate by oscillatory term: |1 - exp(-(α+iΔβ)L)|²
        let al = alpha_per_m * self.fiber_length_km * 1e3;
        let dbl = delta_beta * self.fiber_length_km * 1e3;
        let decay = (-al).exp();
        let osc = 1.0 - 2.0 * decay * dbl.cos() + decay * decay;
        base_efficiency * osc / (al * al).max(1e-60)
    }

    /// FWM crosstalk \[dB\] at the signal channel.
    ///
    /// ```text
    ///   P_FWM = (γ·P₀)² × η × L² × P₀
    ///   XT_FWM = P_FWM / P_signal = (γ·L·P₀)² × η
    /// ```
    pub fn crosstalk_db(&self, launch_power_dbm: f64, alpha_per_km: f64) -> f64 {
        let p0_w = 1e-3 * 10.0_f64.powf(launch_power_dbm / 10.0);
        let gamma = self.nonlinear_coeff;
        let eta = self.efficiency(alpha_per_km);
        let l_km = self.fiber_length_km;
        // FWM crosstalk ratio
        let xt_linear = (gamma * l_km * p0_w).powi(2) * eta;
        if xt_linear <= 0.0 {
            return f64::NEG_INFINITY;
        }
        10.0 * xt_linear.log10()
    }

    /// Minimum safe channel spacing \[GHz\] for which `efficiency <= max_efficiency`.
    ///
    /// Uses binary search over spacing from 1–400 GHz.
    pub fn minimum_safe_spacing_ghz(&self, max_efficiency: f64) -> f64 {
        if max_efficiency >= 1.0 {
            return 0.0;
        }
        let alpha_per_km = 0.046_f64;
        let mut lo = 1.0_f64;
        let mut hi = 400.0_f64;
        for _ in 0..100 {
            let mid = (lo + hi) / 2.0;
            let mut test = self.clone();
            test.channel_spacing_ghz = mid;
            let eta = test.efficiency(alpha_per_km);
            if eta > max_efficiency {
                lo = mid;
            } else {
                hi = mid;
            }
            if (hi - lo) < 0.01 {
                break;
            }
        }
        (lo + hi) / 2.0
    }
}

/// Complementary error function approximation (Abramowitz & Stegun 7.1.26).
fn erfc_approx(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc_approx(-x);
    }
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    (-x * x).exp() * poly
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── PmdAnalysis ──────────────────────────────────────────────────────────

    #[test]
    fn pmd_mean_dgd_formula() {
        let pmd = PmdAnalysis::new(0.1, 100.0);
        // ⟨τ⟩ = 0.1 × √100 = 1.0 ps
        assert_abs_diff_eq!(pmd.mean_dgd_ps(), 1.0, epsilon = 1e-9);
    }

    #[test]
    fn pmd_tolerance_at_10gbaud() {
        let pmd = PmdAnalysis::new(0.05, 400.0);
        // tol = 0.1 / 10 Gbaud × 1000 = 10 ps
        let tol = pmd.tolerance_ps(10.0);
        assert_abs_diff_eq!(tol, 10.0, epsilon = 1e-9);
    }

    #[test]
    fn pmd_limited_reach_decreases_with_higher_rate() {
        let pmd = PmdAnalysis::new(0.1, 100.0);
        let reach_10g = pmd.pmd_limited_reach_km(10.0);
        let reach_100g = pmd.pmd_limited_reach_km(100.0);
        assert!(
            reach_100g < reach_10g,
            "higher rate → shorter PMD-limited reach"
        );
    }

    #[test]
    fn pmd_outage_zero_for_zero_coeff() {
        let pmd = PmdAnalysis::new(0.0, 100.0);
        assert_abs_diff_eq!(pmd.outage_probability(1.0), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn pmd_outage_probability_between_zero_and_half() {
        let pmd = PmdAnalysis::new(0.1, 100.0); // mean DGD = 1 ps
        let p_out = pmd.outage_probability(2.0); // threshold = 2 ps
        assert!((0.0..=0.5).contains(&p_out), "P_out = {p_out:.6}");
    }

    // ── SrsTilt ──────────────────────────────────────────────────────────────

    #[test]
    fn srs_tilt_positive() {
        let srs = SrsTilt::new(32, 100.0, 20.0, 80.0);
        let tilt = srs.power_tilt_db();
        assert!(tilt > 0.0, "tilt = {tilt:.4} dB");
    }

    #[test]
    fn srs_tilt_increases_with_power() {
        let srs_low = SrsTilt::new(32, 100.0, 10.0, 80.0);
        let srs_high = SrsTilt::new(32, 100.0, 20.0, 80.0);
        assert!(srs_high.power_tilt_db() > srs_low.power_tilt_db());
    }

    #[test]
    fn srs_channel_deviation_zero_at_center() {
        let srs = SrsTilt::new(32, 100.0, 20.0, 80.0);
        // For n_channels = 32, center is between channels 15 and 16
        // Channel 16 (index 16): deviation ≈ small positive
        // Channel 0: most negative, channel 31: most positive
        let dev_low = srs.channel_power_deviation_db(0);
        let dev_high = srs.channel_power_deviation_db(31);
        assert!(
            dev_low < 0.0,
            "low-frequency channel loses power: {dev_low}"
        );
        assert!(
            dev_high > 0.0,
            "high-frequency channel gains power: {dev_high}"
        );
    }

    #[test]
    fn srs_threshold_positive() {
        let srs = SrsTilt::new(32, 100.0, 0.0, 100.0);
        let p_th = srs.srs_threshold_w(80.0);
        assert!(p_th > 0.0 && p_th < 10.0, "P_th = {p_th:.4} W");
    }

    // ── XpmPenalty ───────────────────────────────────────────────────────────

    #[test]
    fn xpm_walk_off_length_positive() {
        let xpm = XpmPenalty::new(7, 100.0, 17.0, 80.0, 0.0);
        let l_w = xpm.walk_off_length_km();
        assert!(l_w > 0.0, "walk-off length = {l_w:.2} km");
    }

    #[test]
    fn xpm_walk_off_decreases_with_dispersion() {
        let xpm_low_d = XpmPenalty::new(7, 100.0, 2.0, 80.0, 0.0);
        let xpm_high_d = XpmPenalty::new(7, 100.0, 17.0, 80.0, 0.0);
        assert!(
            xpm_high_d.walk_off_length_km() < xpm_low_d.walk_off_length_km(),
            "higher dispersion → shorter walk-off"
        );
    }

    #[test]
    fn xpm_phase_variance_increases_with_power() {
        let xpm_low = XpmPenalty::new(7, 100.0, 17.0, 80.0, -5.0);
        let xpm_high = XpmPenalty::new(7, 100.0, 17.0, 80.0, 5.0);
        assert!(xpm_high.phase_variance_rad2() > xpm_low.phase_variance_rad2());
    }

    #[test]
    fn xpm_penalty_nonnegative() {
        let xpm = XpmPenalty::new(7, 100.0, 17.0, 80.0, 0.0);
        assert!(xpm.snr_penalty_db() >= 0.0);
    }

    // ── FwmEfficiency ────────────────────────────────────────────────────────

    #[test]
    fn fwm_phase_mismatch_negative() {
        // D > 0 (normal dispersion) → Δβ < 0
        let fwm = FwmEfficiency::new(100.0, 17.0, 80.0, 1.3);
        assert!(fwm.phase_mismatch() < 0.0);
    }

    #[test]
    fn fwm_efficiency_between_zero_and_one() {
        let fwm = FwmEfficiency::new(100.0, 17.0, 80.0, 1.3);
        let eta = fwm.efficiency(0.046);
        assert!((0.0..=1.0).contains(&eta), "η = {eta:.6}");
    }

    #[test]
    fn fwm_efficiency_higher_dispersion_lowers_fwm() {
        let fwm_low_d = FwmEfficiency::new(100.0, 1.0, 80.0, 1.3);
        let fwm_high_d = FwmEfficiency::new(100.0, 17.0, 80.0, 1.3);
        let eta_low = fwm_low_d.efficiency(0.046);
        let eta_high = fwm_high_d.efficiency(0.046);
        assert!(
            eta_high < eta_low,
            "higher D → lower FWM: η_low={eta_low:.4}, η_high={eta_high:.4}"
        );
    }

    #[test]
    fn fwm_minimum_safe_spacing_positive() {
        let fwm = FwmEfficiency::new(100.0, 17.0, 80.0, 1.3);
        let spacing = fwm.minimum_safe_spacing_ghz(0.01);
        assert!(
            spacing > 0.0 && spacing < 400.0,
            "spacing = {spacing:.2} GHz"
        );
    }

    #[test]
    fn fwm_crosstalk_negative_db() {
        let fwm = FwmEfficiency::new(100.0, 17.0, 80.0, 1.3);
        let xt = fwm.crosstalk_db(0.0, 0.046); // 0 dBm = 1 mW
                                               // For typical parameters, FWM crosstalk should be well below 0 dB
        assert!(xt < 0.0, "crosstalk = {xt:.2} dB");
    }
}
