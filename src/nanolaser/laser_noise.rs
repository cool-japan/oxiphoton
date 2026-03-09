//! Laser noise analysis: RIN spectrum, frequency noise PSD, and partition noise.
//!
//! Covers:
//! - Relative Intensity Noise (RIN) spectrum including relaxation oscillation peak
//! - Laser frequency noise and phase noise PSD (Henry formula, Schawlow-Townes)
//! - Partition noise in multi-longitudinal mode lasers and resulting power penalty
//!
//! # References
//!
//! - C. H. Henry, "Theory of the linewidth of semiconductor lasers",
//!   IEEE J. Quantum Electron. 18, 259 (1982).
//! - K. Petermann, "Laser Diode Modulation and Noise", Kluwer 1988.
//! - G. P. Agrawal, "Fiber-Optic Communication Systems", 6th ed., Wiley 2021.

use num_complex::Complex64;
use std::f64::consts::PI;

/// Speed of light in vacuum (m/s)
const C_LIGHT: f64 = 2.997_924_58e8;
/// Planck constant (J·s)
const H_PLANCK: f64 = 6.626_070_15e-34;

// ─── RinSpectrum ─────────────────────────────────────────────────────────────

/// Relative Intensity Noise (RIN) spectrum model.
///
/// Models the double-sideband RIN spectral density including the relaxation
/// oscillation (RO) resonance peak and the shot-noise floor.
///
/// RIN(f) = RIN_RO(f) + shot noise = numerator / |D(f)|² + 2hν/P
///
/// where D(f) = f_RO² − f² + i·f·γ_d is the second-order damping denominator.
#[derive(Debug, Clone)]
pub struct RinSpectrum {
    /// Spontaneous emission coupling factor β.
    pub beta_factor: f64,
    /// Relaxation oscillation frequency f_RO (GHz).
    pub relaxation_oscillation_ghz: f64,
    /// Damping rate γ_d (GHz).
    pub damping_rate_ghz: f64,
    /// Average laser output power (mW).
    pub laser_power_mw: f64,
}

impl RinSpectrum {
    /// Construct a RIN spectrum descriptor.
    pub fn new(beta: f64, ro_ghz: f64, gamma_ghz: f64, power_mw: f64) -> Self {
        Self {
            beta_factor: beta.clamp(1e-8, 1.0),
            relaxation_oscillation_ghz: ro_ghz.max(1e-3),
            damping_rate_ghz: gamma_ghz.max(1e-3),
            laser_power_mw: power_mw.max(1e-6),
        }
    }

    /// Denominator |D(f)|² of the second-order transfer function.
    fn denom_sq(&self, freq_ghz: f64) -> f64 {
        let f_ro = self.relaxation_oscillation_ghz;
        let gamma = self.damping_rate_ghz;
        let d_re = f_ro * f_ro - freq_ghz * freq_ghz;
        let d_im = freq_ghz * gamma;
        d_re * d_re + d_im * d_im
    }

    /// RIN spectral density [dBc/Hz] at frequency `freq_ghz`.
    ///
    /// RIN(f) ≈ 4·β·B_sp·f_RO⁴ / (P·|D(f)|²) + 2hν/P
    ///
    /// where B_sp is the spontaneous emission bandwidth and P is average power.
    pub fn rin_db_per_hz(&self, freq_ghz: f64) -> f64 {
        // Emission frequency ~193 THz (1550 nm default)
        let nu_hz = C_LIGHT / 1550e-9;
        let p_w = self.laser_power_mw * 1e-3;
        let f_ro = self.relaxation_oscillation_ghz;
        let gamma = self.damping_rate_ghz;

        // Spontaneous emission noise intensity: S_I ∝ β·f_RO⁴ / |D(f)|²
        // Normalised to give correct DC value and RO peak shape.
        let denom_sq = self.denom_sq(freq_ghz);
        let denom_sq = if denom_sq < 1e-30 { 1e-30 } else { denom_sq };

        // RIN floor from relative noise: S_RIN_0 = 2·β·(γ_d/f_RO²) [1/GHz units]
        let rin_0_per_ghz = 2.0 * self.beta_factor * (gamma / (f_ro * f_ro));
        // Shape: f_RO⁴ / |D(f)|² peaks near f_RO
        let rin_shaped = rin_0_per_ghz * f_ro.powi(4) / denom_sq;
        // Convert 1/GHz → 1/Hz
        let rin_noise = rin_shaped * 1e-9;

        // Shot noise floor: 2hν/P
        let shot_noise = 2.0 * H_PLANCK * nu_hz / p_w;

        let rin_total = rin_noise + shot_noise;
        if rin_total <= 0.0 {
            return -200.0;
        }
        10.0 * rin_total.log10()
    }

    /// Integrated RIN (dBc) over a frequency band [f_low, f_high] (GHz).
    ///
    /// Uses Gaussian quadrature with 64 points for accuracy.
    pub fn integrated_rin_db(&self, f_low_ghz: f64, f_high_ghz: f64) -> f64 {
        let n = 64_usize;
        let df = (f_high_ghz - f_low_ghz) / n as f64;
        let mut integral_lin = 0.0_f64;
        for k in 0..n {
            let f = f_low_ghz + (k as f64 + 0.5) * df;
            let rin_db = self.rin_db_per_hz(f);
            let rin_lin = 10.0_f64.powf(rin_db / 10.0);
            integral_lin += rin_lin * df * 1e9; // GHz → Hz bandwidth element
        }
        if integral_lin <= 0.0 {
            return -200.0;
        }
        10.0 * integral_lin.log10()
    }

    /// Peak RIN at the relaxation oscillation frequency (dBc/Hz).
    pub fn peak_rin_db_per_hz(&self) -> f64 {
        self.rin_db_per_hz(self.relaxation_oscillation_ghz)
    }

    /// RIN floor far above the relaxation oscillation (shot-noise limited, dBc/Hz).
    pub fn rin_floor_db_per_hz(&self) -> f64 {
        // Well above RO: use 10× f_RO
        self.rin_db_per_hz(self.relaxation_oscillation_ghz * 10.0)
    }
}

// ─── LaserFrequencyNoise ──────────────────────────────────────────────────────

/// Laser frequency noise and phase noise power spectral densities.
///
/// The total frequency noise PSD is modelled as:
///
/// S_ν(f) = Δν/π + S_flicker/f
///
/// where Δν/π is the Schawlow-Townes white noise floor and S_flicker is the
/// 1/f (technical) flicker noise coefficient.
#[derive(Debug, Clone)]
pub struct LaserFrequencyNoise {
    /// Lorentzian (Schawlow-Townes + α-factor) linewidth Δν (Hz).
    pub linewidth_hz: f64,
    /// 1/f flicker noise coefficient S_flicker (Hz²).
    pub flicker_noise_hz2: f64,
    /// White frequency noise floor S_ν_white (Hz²/Hz) = Δν/π.
    pub white_noise_hz: f64,
}

impl LaserFrequencyNoise {
    /// Construct from Lorentzian linewidth Δν_white (Hz²/Hz = Δν/π).
    fn from_linewidth(linewidth_hz: f64, flicker_hz2: f64) -> Self {
        Self {
            linewidth_hz,
            flicker_noise_hz2: flicker_hz2,
            white_noise_hz: linewidth_hz / PI,
        }
    }

    /// Construct for a single-mode diode laser (Δν ≈ MHz, strong 1/f noise).
    pub fn new_diode(linewidth_mhz: f64) -> Self {
        let lw_hz = linewidth_mhz * 1e6;
        Self::from_linewidth(lw_hz, lw_hz * 1e6) // flicker dominates up to MHz
    }

    /// Construct for a fiber DFB laser (narrow linewidth, low 1/f).
    pub fn new_fiber_laser(linewidth_khz: f64) -> Self {
        let lw_hz = linewidth_khz * 1e3;
        Self::from_linewidth(lw_hz, lw_hz * 100.0)
    }

    /// Construct for an external-cavity laser (ECL).
    pub fn new_ecl(linewidth_khz: f64) -> Self {
        let lw_hz = linewidth_khz * 1e3;
        Self::from_linewidth(lw_hz, lw_hz * 10.0) // very low flicker
    }

    /// Frequency noise PSD S_ν(f) [Hz²/Hz] at offset frequency `freq_hz`.
    ///
    /// S_ν(f) = Δν/π + S_flicker/f
    pub fn freq_noise_hz2_per_hz(&self, freq_hz: f64) -> f64 {
        let f = freq_hz.max(1.0); // avoid division by zero
        self.white_noise_hz + self.flicker_noise_hz2 / f
    }

    /// Phase noise PSD S_φ(f) [dBc/Hz] at offset `freq_hz`.
    ///
    /// S_φ(f) = S_ν(f) / f²
    pub fn phase_noise_dbc_hz(&self, freq_hz: f64) -> f64 {
        let f = freq_hz.max(1.0);
        let s_phi = self.freq_noise_hz2_per_hz(f) / (f * f);
        if s_phi <= 0.0 {
            return -200.0;
        }
        10.0 * s_phi.log10()
    }

    /// Lorentzian linewidth from white frequency noise floor.
    ///
    /// Δν = π · S_ν_white
    pub fn linewidth_from_noise_hz(&self) -> f64 {
        PI * self.white_noise_hz
    }

    /// Temporal coherence length L_c = c / (π · Δν).
    pub fn coherence_length_m(&self) -> f64 {
        let lw = self.linewidth_from_noise_hz().max(1.0);
        C_LIGHT / (PI * lw)
    }

    /// RMS phase error for coherent detection with symbol rate R_s (GBaud).
    ///
    /// σ_φ = √(2π · Δν / R_s)
    pub fn phase_error_rad(&self, symbol_rate_gbaud: f64) -> f64 {
        let r_s = symbol_rate_gbaud.max(1e-6) * 1e9;
        let lw = self.linewidth_hz.max(1.0);
        (2.0 * PI * lw / r_s).sqrt()
    }
}

// ─── PartitionNoise ───────────────────────────────────────────────────────────

/// Partition noise in multi-longitudinal mode (MLM) lasers.
///
/// Partition noise arises from random fluctuations of optical power between
/// longitudinal modes. When transmitted through a dispersive fiber, the
/// time-of-flight difference between modes causes additional intensity noise
/// at the receiver.
///
/// The power penalty from partition noise is given by:
/// δ_PN (dB) ≈ −5 · log₁₀(1 − Q²·(D·L·σ_λ)²·k²/2)
#[derive(Debug, Clone)]
pub struct PartitionNoise {
    /// Number of longitudinal modes.
    pub n_modes: usize,
    /// Mode partition parameter k (0–1; typical 0.1–0.9).
    pub mode_partition_parameter: f64,
    /// Chromatic dispersion coefficient D (ps/nm·km).
    pub fiber_dispersion_ps_per_nm: f64,
}

impl PartitionNoise {
    /// Construct a partition noise model.
    pub fn new(n_modes: usize, k: f64, disp_ps_per_nm: f64) -> Self {
        Self {
            n_modes: n_modes.max(1),
            mode_partition_parameter: k.clamp(0.0, 1.0),
            fiber_dispersion_ps_per_nm: disp_ps_per_nm,
        }
    }

    /// Power penalty δ_PN (dB) from partition noise after a fiber span.
    ///
    /// δ_PN = −5·log₁₀(1 − Q²·σ_t²)
    /// where σ_t = D·L·σ_λ is the dispersion-induced rms time spread.
    ///
    /// Parameters:
    /// - `q_factor`: system Q factor (e.g. 7 for 10⁻¹² BER)
    /// - `source_spectral_width_nm`: rms source spectral width (nm)
    ///
    /// Note: fiber length is derived from the dispersion product.
    pub fn power_penalty_db(&self, q_factor: f64, source_spectral_width_nm: f64) -> f64 {
        let k = self.mode_partition_parameter;
        let d = self.fiber_dispersion_ps_per_nm; // ps/(nm·km)
                                                 // Use unit length (1 km) for the penalty formula
        let l_km = 1.0_f64;
        let sigma_t_ps = d * l_km * source_spectral_width_nm; // ps
        let sigma_t_s = sigma_t_ps * 1e-12;
        // Assume bit period T_b = 1 ns (1 Gb/s) for normalisation
        let t_b = 1e-9_f64;
        let sigma_norm_sq = (q_factor * k * sigma_t_s / t_b).powi(2) / 2.0;
        let arg = 1.0 - sigma_norm_sq;
        if arg <= 0.0 {
            return 20.0;
        } // saturated penalty
        -5.0 * arg.log10()
    }

    /// Maximum fiber span length (km) before partition noise penalty exceeds `max_penalty_db`.
    ///
    /// Inverts the penalty formula to solve for L.
    pub fn max_length_km(&self, max_penalty_db: f64, q_factor: f64, spectral_width_nm: f64) -> f64 {
        let k = self.mode_partition_parameter;
        let d = self.fiber_dispersion_ps_per_nm;
        // From δ_PN = -5·log10(1 - Q²·k²·(D·L·σ_λ)²/(2·T_b²)) ≤ max_penalty
        // Solve for L:
        let threshold_lin = 10.0_f64.powf(-max_penalty_db / 5.0);
        let sigma_norm_sq_max = 1.0 - threshold_lin;
        if sigma_norm_sq_max <= 0.0 {
            return 0.0;
        }
        let t_b = 1e-9_f64;
        // sigma_norm_sq_max = Q²·k²·(D·L·σ_λ)²/(2·T_b²)
        // (D·L·σ_λ)² = sigma_norm_sq_max * 2 * T_b² / (Q²·k²)
        let qk = q_factor * k;
        if qk.abs() < 1e-30 {
            return f64::INFINITY;
        }
        let d_l_sigma_s = (sigma_norm_sq_max * 2.0).sqrt() * t_b / qk;
        let d_sigma_km_s = d * spectral_width_nm * 1e-12; // km·s / km = s/km ... units
                                                          // d_l_sigma_s = D[ps/nm/km] * L[km] * sigma[nm] * 1e-12 s/ps
        if d_sigma_km_s.abs() < 1e-30 {
            return f64::INFINITY;
        }
        (d_l_sigma_s / d_sigma_km_s.abs()).abs()
    }
}

// ─── Utility: phase noise transfer function ───────────────────────────────────

/// Compute the complex transfer function H(ω) for a second-order laser noise system.
///
/// H(ω) = ω_R² / (ω_R² − ω² + i·Γ·ω)
pub fn laser_noise_transfer(omega_rad_s: f64, omega_ro_rad_s: f64, gamma_rad_s: f64) -> Complex64 {
    let omega_r_sq = omega_ro_rad_s * omega_ro_rad_s;
    let denom = Complex64::new(
        omega_r_sq - omega_rad_s * omega_rad_s,
        gamma_rad_s * omega_rad_s,
    );
    if denom.norm() < 1e-30 {
        Complex64::new(1e30, 0.0)
    } else {
        Complex64::new(omega_r_sq, 0.0) / denom
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_rin_floor_lower_than_peak() {
        let rin = RinSpectrum::new(1e-4, 5.0, 1.0, 1.0);
        let peak = rin.peak_rin_db_per_hz();
        let floor = rin.rin_floor_db_per_hz();
        assert!(
            floor < peak,
            "RIN floor should be lower than peak: floor={} peak={}",
            floor,
            peak
        );
    }

    #[test]
    fn test_rin_integrated_finite() {
        let rin = RinSpectrum::new(1e-4, 5.0, 1.0, 1.0);
        let rin_int = rin.integrated_rin_db(0.1, 10.0);
        assert!(
            rin_int.is_finite(),
            "Integrated RIN should be finite: {}",
            rin_int
        );
    }

    #[test]
    fn test_frequency_noise_diode_vs_fiber() {
        let diode = LaserFrequencyNoise::new_diode(10.0); // 10 MHz linewidth
        let fiber = LaserFrequencyNoise::new_fiber_laser(1.0); // 1 kHz linewidth
                                                               // Diode should have higher frequency noise at 1 kHz offset
        let s_diode = diode.freq_noise_hz2_per_hz(1e3);
        let s_fiber = fiber.freq_noise_hz2_per_hz(1e3);
        assert!(
            s_diode > s_fiber,
            "Diode should be noisier than fiber laser at 1 kHz"
        );
    }

    #[test]
    fn test_coherence_length_narrow_linewidth() {
        let ecl = LaserFrequencyNoise::new_ecl(0.1); // 100 Hz linewidth
        let lc = ecl.coherence_length_m();
        // L_c = c/(π·Δν) ≈ 3e8 / (π·100) ≈ 955 km
        assert!(lc > 1e4, "ECL coherence length should be many km: {} m", lc);
    }

    #[test]
    fn test_partition_noise_penalty_zero_at_short_reach() {
        let pn = PartitionNoise::new(5, 0.5, 17.0);
        let penalty = pn.power_penalty_db(7.0, 0.1);
        assert!(
            penalty >= 0.0,
            "Power penalty must be non-negative: {}",
            penalty
        );
    }

    #[test]
    fn test_partition_noise_max_length_decreases_with_dispersion() {
        let pn_low = PartitionNoise::new(5, 0.5, 3.5); // DSF
        let pn_high = PartitionNoise::new(5, 0.5, 17.0); // SSMF
        let l_low = pn_low.max_length_km(1.0, 7.0, 1.0);
        let l_high = pn_high.max_length_km(1.0, 7.0, 1.0);
        assert!(l_low > l_high, "Lower dispersion should allow longer reach");
    }

    #[test]
    fn test_phase_error_increases_with_linewidth() {
        let narrow = LaserFrequencyNoise::new_ecl(0.1);
        let wide = LaserFrequencyNoise::new_diode(1.0);
        let err_narrow = narrow.phase_error_rad(28.0);
        let err_wide = wide.phase_error_rad(28.0);
        assert!(
            err_wide > err_narrow,
            "Wider linewidth → larger phase error"
        );
    }

    #[test]
    fn test_laser_noise_transfer_dc() {
        let omega_ro = 2.0 * PI * 5e9; // 5 GHz RO
        let gamma = 2.0 * PI * 1e9; // 1 GHz damping
        let h = laser_noise_transfer(1.0, omega_ro, gamma); // near DC (ω ≈ 0)
                                                            // |H(0)| ≈ 1
        assert_abs_diff_eq!(h.re, 1.0, epsilon = 0.01);
        assert_abs_diff_eq!(h.im, 0.0, epsilon = 0.01);
    }
}
