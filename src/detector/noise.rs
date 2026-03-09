//! Detector noise analysis toolkit and photon-counting statistics.
//!
//! Provides fundamental noise formulas (shot, Johnson/thermal, 1/f),
//! cascaded noise-figure calculation (Friis formula), blackbody radiance,
//! and Poissonian photon-counting statistics for single-photon applications.

use crate::error::OxiPhotonError;

// ── Physical constants ────────────────────────────────────────────────────────
const KB: f64 = 1.380_649e-23; // J/K
const E_CHARGE: f64 = 1.602_176_634e-19; // C
const H_PLANCK: f64 = 6.626_070_15e-34; // J·s
const C0: f64 = 2.997_924_58e8; // m/s

// ── NoiseAnalysis — static toolkit ───────────────────────────────────────────

/// Collection of static detector-noise formulas.
pub struct NoiseAnalysis;

impl NoiseAnalysis {
    /// Shot noise RMS current in bandwidth `bandwidth_hz` (A).
    ///
    /// i_shot = √(2 · e · I · B)
    pub fn shot_noise_a(current_a: f64, bandwidth_hz: f64) -> f64 {
        (2.0 * E_CHARGE * current_a * bandwidth_hz).sqrt()
    }

    /// Johnson (thermal) noise RMS voltage in bandwidth `bandwidth_hz` (V).
    ///
    /// v_J = √(4 · k_B · T · R · B)
    pub fn johnson_noise_v(resistance_ohm: f64, temperature_k: f64, bandwidth_hz: f64) -> f64 {
        (4.0 * KB * temperature_k * resistance_ohm * bandwidth_hz).sqrt()
    }

    /// Johnson (thermal) noise RMS current in bandwidth `bandwidth_hz` (A).
    ///
    /// i_J = v_J / R = √(4 · k_B · T · B / R)
    pub fn johnson_noise_a(resistance_ohm: f64, temperature_k: f64, bandwidth_hz: f64) -> f64 {
        (4.0 * KB * temperature_k * bandwidth_hz / resistance_ohm).sqrt()
    }

    /// 1/f (flicker) noise RMS current (A) — simplified single-pole model.
    ///
    /// i_flicker ≈ I · K · √(B / f_corner)
    ///
    /// Here K is a dimensionless technology-dependent constant (~10⁻³–10⁻⁵)
    /// and `corner_freq_hz` is the 1/f corner frequency.
    pub fn flicker_noise_a(
        current_a: f64,
        k_factor: f64,
        bandwidth_hz: f64,
        corner_freq_hz: f64,
    ) -> f64 {
        if corner_freq_hz <= 0.0 {
            return 0.0;
        }
        current_a * k_factor * (bandwidth_hz / corner_freq_hz).sqrt()
    }

    /// Total noise RMS current from quadrature (RSS) sum of components (A).
    pub fn total_noise_a(components: &[f64]) -> f64 {
        let sum_sq: f64 = components.iter().map(|&x| x * x).sum();
        sum_sq.sqrt()
    }

    /// Noise figure (dB) from linear noise factor F.
    ///
    /// NF = 10 · log₁₀(F)
    pub fn noise_figure_db(f_linear: f64) -> f64 {
        10.0 * f_linear.log10()
    }

    /// Cascaded noise figure via Friis formula.
    ///
    /// F_total = F₁ + (F₂ − 1)/G₁ + (F₃ − 1)/(G₁·G₂) + …
    ///
    /// `noise_figures` and `gains` are linear (not dB) quantities.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] when the slices have
    /// mismatched lengths (must be equal) or are empty.
    pub fn cascaded_noise_figure(
        noise_figures: &[f64],
        gains: &[f64],
    ) -> Result<f64, OxiPhotonError> {
        if noise_figures.is_empty() {
            return Err(OxiPhotonError::NumericalError(
                "noise_figures slice must not be empty".into(),
            ));
        }
        if noise_figures.len() != gains.len() {
            return Err(OxiPhotonError::NumericalError(format!(
                "noise_figures length ({}) must equal gains length ({})",
                noise_figures.len(),
                gains.len()
            )));
        }
        let mut f_total = noise_figures[0];
        let mut cumulative_gain = gains[0];
        for i in 1..noise_figures.len() {
            f_total += (noise_figures[i] - 1.0) / cumulative_gain;
            cumulative_gain *= gains[i];
        }
        Ok(f_total)
    }

    /// Equivalent noise bandwidth of a first-order RC lowpass filter (Hz).
    ///
    /// B_n = (π/2) · f_3dB
    pub fn rc_noise_bandwidth_hz(f_3db_hz: f64) -> f64 {
        std::f64::consts::FRAC_PI_2 * f_3db_hz
    }

    /// Photon-number standard deviation for a coherent (Poissonian) state.
    ///
    /// σ_n = √n̄
    pub fn photon_number_uncertainty(mean_photon_number: f64) -> f64 {
        mean_photon_number.sqrt()
    }

    /// Planck spectral radiance at wavelength `lambda_nm` and temperature
    /// `temperature_k` (W · m⁻² · sr⁻¹ · nm⁻¹).
    ///
    /// B_λ = 2hc² / λ⁵ · 1/(exp(hc/(λ·k_B·T)) − 1)  (converted to per nm)
    pub fn blackbody_radiance_w_per_m2_per_sr_per_nm(lambda_nm: f64, temperature_k: f64) -> f64 {
        let lambda_m = lambda_nm * 1e-9;
        let exponent = H_PLANCK * C0 / (lambda_m * KB * temperature_k);
        let denom = exponent.exp() - 1.0;
        if denom <= 0.0 {
            return 0.0;
        }
        // 2hc²/λ⁵ in SI (W·m⁻²·sr⁻¹·m⁻¹), converted to per nm (*1e-9)
        let radiance_per_m = 2.0 * H_PLANCK * C0 * C0 / (lambda_m.powi(5) * denom);
        radiance_per_m * 1e-9 // per m → per nm
    }
}

// ── DetectorNoiseModel ────────────────────────────────────────────────────────

/// Parametric noise model for a complete photodetection front-end.
///
/// Captures the three dominant noise contributions:
/// - Shot noise (signal-power dependent)
/// - Thermal (Johnson) noise floor
/// - 1/f (flicker) noise at low frequencies
#[derive(Debug, Clone)]
pub struct DetectorNoiseModel {
    /// Fraction of total noise attributable to shot noise at the nominal
    /// operating point (informational).
    pub shot_noise_fraction: f64,
    /// Thermal noise power spectral density (A² / Hz).
    pub thermal_noise_a2_per_hz: f64,
    /// 1/f corner frequency (Hz).
    pub flicker_corner_hz: f64,
    /// Total detection bandwidth (Hz).
    pub total_bandwidth_hz: f64,
}

impl DetectorNoiseModel {
    /// Construct a noise model.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] when any parameter is
    /// negative or not finite.
    pub fn new(
        thermal_noise_a2_per_hz: f64,
        flicker_corner_hz: f64,
        total_bandwidth_hz: f64,
    ) -> Result<Self, OxiPhotonError> {
        if thermal_noise_a2_per_hz < 0.0 || !thermal_noise_a2_per_hz.is_finite() {
            return Err(OxiPhotonError::NumericalError(
                "thermal_noise_a2_per_hz must be non-negative and finite".into(),
            ));
        }
        if total_bandwidth_hz <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "total_bandwidth_hz must be positive".into(),
            ));
        }
        Ok(Self {
            shot_noise_fraction: 0.0, // updated dynamically
            thermal_noise_a2_per_hz,
            flicker_corner_hz,
            total_bandwidth_hz,
        })
    }

    /// Crossover signal current above which shot noise dominates thermal noise.
    ///
    /// At crossover: 2·e·I·R = I_th²/Hz → I_cross = I_th² / (2·e·R)
    /// where `responsivity` is the detector responsivity (A/W).
    pub fn shot_noise_dominated_power_w(&self, responsivity: f64) -> f64 {
        if responsivity == 0.0 {
            return f64::INFINITY;
        }
        // I_cross such that shot noise density = thermal noise density
        // 2·e·I_cross = thermal_noise_a2_per_hz
        let i_cross = self.thermal_noise_a2_per_hz / (2.0 * E_CHARGE);
        i_cross / responsivity
    }

    /// Total noise current spectral density at frequency `freq_hz` and
    /// signal current `signal_current_a` (A/√Hz).
    pub fn noise_current_density_a_per_sqrt_hz(&self, signal_current_a: f64, freq_hz: f64) -> f64 {
        // Shot noise (white)
        let i_shot_sq = 2.0 * E_CHARGE * signal_current_a;
        // Thermal noise (white)
        let i_th_sq = self.thermal_noise_a2_per_hz;
        // 1/f noise: S_1f = I² * K_f / f  (K_f fitted to flicker_corner)
        let i_flicker_sq = if freq_hz > 0.0 && self.flicker_corner_hz > 0.0 {
            i_th_sq * self.flicker_corner_hz / freq_hz
        } else {
            0.0
        };
        (i_shot_sq + i_th_sq + i_flicker_sq).sqrt()
    }

    /// Integrated RMS noise current over the detection bandwidth (A).
    pub fn integrated_noise_a(&self, signal_current_a: f64) -> f64 {
        // White components: integrate over B directly
        let i_shot_sq_hz = 2.0 * E_CHARGE * signal_current_a;
        let i_th_sq_hz = self.thermal_noise_a2_per_hz;
        let white = (i_shot_sq_hz + i_th_sq_hz) * self.total_bandwidth_hz;

        // 1/f component: ∫_{f_low}^{B} S_1f df ≈ I_th² * f_c * ln(B/f_low)
        let flicker =
            if self.flicker_corner_hz > 0.0 && self.total_bandwidth_hz > self.flicker_corner_hz {
                let f_low = self.flicker_corner_hz * 0.001; // approx lower bound
                i_th_sq_hz * self.flicker_corner_hz * (self.total_bandwidth_hz / f_low).ln()
            } else {
                0.0
            };

        (white + flicker).sqrt()
    }

    /// Minimum detectable power in the shot-noise limit (SNR = 1).
    ///
    /// MDP_shot = √(2·e·B) / R  \[W\]
    pub fn shot_noise_mdp_w(&self, responsivity: f64) -> f64 {
        if responsivity == 0.0 {
            return f64::INFINITY;
        }
        let i_noise = (2.0 * E_CHARGE * self.total_bandwidth_hz).sqrt();
        i_noise / responsivity
    }

    /// Minimum detectable power in the thermal-noise limit (SNR = 1).
    ///
    /// MDP_thermal = √(I_th² · B) / R  \[W\]
    pub fn thermal_noise_mdp_w(&self, responsivity: f64) -> f64 {
        if responsivity == 0.0 {
            return f64::INFINITY;
        }
        let i_noise = (self.thermal_noise_a2_per_hz * self.total_bandwidth_hz).sqrt();
        i_noise / responsivity
    }
}

// ── PhotonCounting — static toolkit ─────────────────────────────────────────

/// Static toolkit for Poissonian photon-counting statistics.
pub struct PhotonCounting;

impl PhotonCounting {
    /// Poisson probability P(n; μ) = μⁿ · exp(−μ) / n!.
    ///
    /// Uses log-domain computation to avoid overflow for large n or μ.
    pub fn poisson_probability(n: usize, mean: f64) -> f64 {
        if mean < 0.0 {
            return 0.0;
        }
        if mean == 0.0 {
            return if n == 0 { 1.0 } else { 0.0 };
        }
        // log P(n;μ) = n·ln(μ) - μ - ln(n!)
        let log_p = n as f64 * mean.ln() - mean - Self::log_factorial(n);
        log_p.exp()
    }

    /// Probability that at least one photon is detected given mean photon
    /// number `mean_photons` and detection efficiency η.
    ///
    /// P_det = 1 − exp(−η · μ)
    pub fn detection_probability(mean_photons: f64, efficiency: f64) -> f64 {
        1.0 - (-(efficiency * mean_photons)).exp()
    }

    /// False positive (dark-click) probability within a time window.
    ///
    /// P_false = 1 − exp(−DCR · T)  ≈ DCR · T  for small DCR·T.
    pub fn false_positive_rate(dark_count_rate: f64, time_window_s: f64) -> f64 {
        1.0 - (-(dark_count_rate * time_window_s)).exp()
    }

    /// Optimal threshold n_th that minimises total error (miss + false alarm)
    /// for a Poissonian signal with mean `signal_mean` versus Poissonian
    /// background with mean `dark_mean`.
    ///
    /// Scanning n from 1 upward and choosing the threshold that maximises
    /// (P_detect − P_false_alarm).
    pub fn optimal_threshold(signal_mean: f64, dark_mean: f64) -> usize {
        // The optimal threshold is where the likelihood ratio equals 1:
        // P_signal(n) / P_dark(n) = 1  →  n·ln(μ_s/μ_d) = μ_s − μ_d
        // n_opt = (μ_s − μ_d) / ln(μ_s/μ_d)
        if signal_mean <= dark_mean || dark_mean == 0.0 || signal_mean == 0.0 {
            return 1;
        }
        let n_opt = (signal_mean - dark_mean) / (signal_mean / dark_mean).ln();
        (n_opt.round() as usize).max(1)
    }

    /// Bit error rate for binary photon-counting communication
    /// (0 = no photon, 1 = photon).
    ///
    /// BER = ½ · (P_miss + P_false_alarm)
    /// using the optimal threshold from \[`Self::optimal_threshold`\].
    pub fn photon_counting_ber(signal_mean: f64, dark_mean: f64) -> f64 {
        let n_th = Self::optimal_threshold(signal_mean, dark_mean);

        // P_miss: probability that signal pulse gives < n_th detections
        let p_detect = (n_th..=1000)
            .map(|n| Self::poisson_probability(n, signal_mean))
            .sum::<f64>();
        let p_miss = 1.0 - p_detect;

        // P_false: probability that dark counts give >= n_th detections
        let p_false: f64 = (n_th..=1000)
            .map(|n| Self::poisson_probability(n, dark_mean))
            .sum();

        0.5 * (p_miss + p_false)
    }

    /// Required mean photon number per pulse to achieve a target BER, given
    /// a fixed dark count mean per time window.
    ///
    /// Solved by binary search over μ_signal ∈ \[dark_mean, 1000\].
    pub fn required_photons_for_ber(ber_target: f64, dark_mean: f64) -> f64 {
        // Binary search: find μ_s such that BER(μ_s, dark_mean) = ber_target
        let mut lo = dark_mean.max(1e-3);
        let mut hi = 1000.0_f64;
        for _ in 0..64 {
            let mid = 0.5 * (lo + hi);
            let ber = Self::photon_counting_ber(mid, dark_mean);
            if ber > ber_target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    /// Natural logarithm of n! using exact summation for small n and the
    /// Lanczos approximation (via lgamma) for larger n.
    fn log_factorial(n: usize) -> f64 {
        // For small n, sum exactly: ln(n!) = ln(1) + ln(2) + … + ln(n)
        const EXACT_LIMIT: usize = 20;
        if n <= EXACT_LIMIT {
            (1..=n).map(|k| (k as f64).ln()).sum()
        } else {
            // Stirling series with higher-order correction terms for accuracy
            let nf = n as f64;
            nf * nf.ln() - nf + 0.5 * (2.0 * std::f64::consts::PI * nf).ln() + 1.0 / (12.0 * nf)
                - 1.0 / (360.0 * nf * nf * nf)
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_shot_noise_scaling() {
        // i_shot ∝ sqrt(I): quadruple current → double noise
        let i1 = NoiseAnalysis::shot_noise_a(1e-6, 1e9);
        let i4 = NoiseAnalysis::shot_noise_a(4e-6, 1e9);
        assert_relative_eq!(i4 / i1, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_johnson_noise_temp_dependence() {
        // Johnson noise ∝ sqrt(T): quadruple temperature → double noise
        let v1 = NoiseAnalysis::johnson_noise_v(1000.0, 300.0, 1e9);
        let v4 = NoiseAnalysis::johnson_noise_v(1000.0, 1200.0, 1e9);
        assert_relative_eq!(v4 / v1, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_friis_formula() {
        // Two stages: F1=2 (NF=3dB), G1=10, F2=3 (NF≈4.8dB)
        // F_total = 2 + (3-1)/10 = 2.2
        let f_total =
            NoiseAnalysis::cascaded_noise_figure(&[2.0, 3.0], &[10.0, 10.0]).expect("valid inputs");
        assert_relative_eq!(f_total, 2.2, epsilon = 1e-12);
        // Total NF > first stage NF
        assert!(f_total > 2.0, "cascaded NF should exceed first stage");
    }

    #[test]
    fn test_poisson_probability_normalization() {
        // Sum of P(n; μ) for n=0..∞ must equal 1; use a small mean so the
        // tail is negligible well before n=100 and Stirling is accurate.
        let mean = 3.5_f64;
        // Extend to n=500 to capture essentially the full distribution and
        // allow for Stirling rounding in each term (cumulative < 1e-4).
        let sum: f64 = (0..=500)
            .map(|n| PhotonCounting::poisson_probability(n, mean))
            .sum();
        // The true tail probability beyond n=500 for μ=3.5 is negligible.
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "Poisson sum = {sum}, expected ≈ 1.0 (tol 1e-4)"
        );
    }

    #[test]
    fn test_detection_probability() {
        // For η=1 and large μ, P_det → 1
        let p = PhotonCounting::detection_probability(20.0, 1.0);
        assert!(p > 0.999, "P_det = {p}, expected > 0.999");
        // For η=0, P_det = 0
        let p0 = PhotonCounting::detection_probability(10.0, 0.0);
        assert_relative_eq!(p0, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_photon_counting_ber_decreases_with_signal() {
        let dark = 0.1;
        let ber_low = PhotonCounting::photon_counting_ber(2.0, dark);
        let ber_high = PhotonCounting::photon_counting_ber(10.0, dark);
        assert!(
            ber_high < ber_low,
            "BER should decrease with increasing signal: {ber_high} >= {ber_low}"
        );
    }

    #[test]
    fn test_total_noise_quadrature_sum() {
        // Components: 3, 4 → total should be 5
        let total = NoiseAnalysis::total_noise_a(&[3.0, 4.0]);
        assert_relative_eq!(total, 5.0, epsilon = 1e-12);
    }
}
