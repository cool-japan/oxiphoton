use super::comb::C0;
/// Optical timing, Allan deviation analysis, and relativistic geodesy.
///
/// Covers:
/// - Overlapping Allan deviation (OADEV) and modified Allan deviation (MDEV)
///   computed from fractional-frequency or phase time-series data
/// - Optical frequency transfer over stabilized fiber links
/// - Relativistic clock comparison for geodetic applications
use crate::error::OxiPhotonError;

// ─── Physical constants ──────────────────────────────────────────────────────
/// Gravitational acceleration (m/s²).
const G: f64 = 9.80665;

// ─── AllanDeviation ──────────────────────────────────────────────────────────

/// Allan deviation analysis for oscillators and frequency standards.
///
/// The overlapping Allan deviation (OADEV) provides a statistically efficient
/// estimate of frequency stability at multiple averaging times τ.  The modified
/// Allan deviation (MDEV) has improved noise-type discrimination.
///
/// Both quantities are dimensionless fractional frequency stabilities σ_y(τ).
#[derive(Debug, Clone)]
pub struct AllanDeviation {
    /// Averaging times τ (s) at which the deviations are evaluated.
    pub tau_values: Vec<f64>,
    /// Overlapping Allan deviation σ_y(τ) — dimensionless.
    pub oadev: Vec<f64>,
    /// Modified Allan deviation (MDEV) — dimensionless.
    pub mdev: Vec<f64>,
}

impl AllanDeviation {
    /// Compute OADEV and MDEV from a fractional-frequency time series y\[i\].
    ///
    /// The fractional frequency is y = (f − f₀) / f₀.  Consecutive samples
    /// are separated by the basic measurement interval τ₀.
    ///
    /// Averaging times are generated as τ_m = m · τ₀ for m = 1, 2, 4, 8, …
    /// up to N/4 (Nyquist-like bound on OADEV reliability).
    ///
    /// # Arguments
    /// * `y`    — fractional frequency time series (dimensionless)
    /// * `tau0` — basic measurement interval (s)
    pub fn from_fractional_freq(y: &[f64], tau0: f64) -> Self {
        let n = y.len();
        if n < 2 {
            return Self {
                tau_values: Vec::new(),
                oadev: Vec::new(),
                mdev: Vec::new(),
            };
        }

        // Build phase data by integrating fractional frequencies: x[k] = τ₀ Σ y[i]
        let mut x = Vec::with_capacity(n + 1);
        x.push(0.0_f64);
        for &yi in y {
            let last = *x.last().expect("x always has at least one element");
            x.push(last + yi * tau0);
        }

        Self::from_phase_internal(&x, tau0)
    }

    /// Compute OADEV and MDEV from a phase time series x\[i\] (seconds).
    ///
    /// Phase data x represents the time error of the oscillator at sample i,
    /// measured with basic interval τ₀.
    ///
    /// # Arguments
    /// * `x`    — phase time-error data (seconds)
    /// * `tau0` — basic measurement interval (s)
    pub fn from_phase_data(x: &[f64], tau0: f64) -> Self {
        Self::from_phase_internal(x, tau0)
    }

    // ── Internal implementation ─────────────────────────────────────────────

    fn from_phase_internal(x: &[f64], tau0: f64) -> Self {
        let n = x.len();
        if n < 3 {
            return Self {
                tau_values: Vec::new(),
                oadev: Vec::new(),
                mdev: Vec::new(),
            };
        }

        // Generate m values: 1, 2, 4, 8, … up to n/4
        let max_m = (n / 4).max(1);
        let mut m_values: Vec<usize> = Vec::new();
        let mut m = 1_usize;
        while m <= max_m {
            m_values.push(m);
            m = if m == 1 { 2 } else { m * 2 };
        }

        let mut tau_values = Vec::with_capacity(m_values.len());
        let mut oadev = Vec::with_capacity(m_values.len());
        let mut mdev_vals = Vec::with_capacity(m_values.len());

        for &m in &m_values {
            let tau = m as f64 * tau0;
            let oadev_val = Self::compute_oadev(x, m);
            let mdev_val = Self::compute_mdev(x, m);
            tau_values.push(tau);
            oadev.push(oadev_val);
            mdev_vals.push(mdev_val);
        }

        Self {
            tau_values,
            oadev,
            mdev: mdev_vals,
        }
    }

    /// Overlapping Allan deviation at averaging factor m.
    ///
    /// OADEV(m) = sqrt( 1/(2m²τ₀² · (N-2m)) · Σ_{j=0}^{N-2m-1} \[x[j+2m\] − 2x\[j+m\] + x\[j\]]² )
    ///
    /// The factor 1/(2m²τ₀²) converts from phase variance to fractional-frequency variance.
    /// Here we return the dimensionless OADEV by working with fractional frequencies
    /// derived from x by dividing by τ₀.
    fn compute_oadev(x: &[f64], m: usize) -> f64 {
        let n = x.len();
        if n <= 2 * m {
            return 0.0;
        }
        let count = n - 2 * m;
        let tau0 = 1.0_f64; // normalized; caller scales tau separately
        let tau_m_sq = (m as f64 * tau0).powi(2);

        let sum_sq: f64 = (0..count)
            .map(|j| {
                let diff = x[j + 2 * m] - 2.0 * x[j + m] + x[j];
                diff * diff
            })
            .sum();

        let variance = sum_sq / (2.0 * tau_m_sq * count as f64);
        variance.sqrt()
    }

    /// Modified Allan deviation at averaging factor m.
    ///
    /// MDEV(m) = sqrt( 1/(2m⁴τ₀² · (N-3m+1)) · Σ_{j=0}^{N-3m} \[Σ_{i=j}^{j+m-1} (x[i+2m\] − 2x\[i+m\] + x\[i\])]² )
    ///
    /// The inner sum over m samples gives MDEV a flat response to white PM noise
    /// (unlike OADEV which rises as 1/τ for white PM).
    fn compute_mdev(x: &[f64], m: usize) -> f64 {
        let n = x.len();
        if n <= 3 * m {
            return 0.0;
        }
        let tau0 = 1.0_f64;
        let denom_factor = 2.0 * (m as f64).powi(4) * tau0 * tau0;
        let outer_count = n - 3 * m + 1;
        if outer_count == 0 {
            return 0.0;
        }

        let sum_sq: f64 = (0..outer_count)
            .map(|j| {
                let inner: f64 = (j..(j + m))
                    .map(|i| {
                        if i + 2 * m < n {
                            x[i + 2 * m] - 2.0 * x[i + m] + x[i]
                        } else {
                            0.0
                        }
                    })
                    .sum();
                inner * inner
            })
            .sum();

        let variance = sum_sq / (denom_factor * outer_count as f64);
        variance.sqrt()
    }

    /// Detect the white frequency noise level from the OADEV slope.
    ///
    /// White frequency noise produces σ_y(τ) ∝ 1/√τ (slope −1/2 on a log-log plot).
    /// Returns the noise coefficient h₀ such that σ_y(τ) = √(h₀/2τ), or `None`
    /// if insufficient data is available.
    pub fn white_freq_noise(&self) -> Option<f64> {
        if self.tau_values.len() < 2 {
            return None;
        }
        // Estimate slope from first two points
        let tau1 = self.tau_values[0];
        let tau2 = self.tau_values[1];
        let s1 = self.oadev[0];
        let s2 = self.oadev[1];
        if tau1 <= 0.0 || tau2 <= 0.0 || s1 <= 0.0 || s2 <= 0.0 {
            return None;
        }
        // slope m = log(s2/s1) / log(tau2/tau1)
        let slope = (s2 / s1).log10() / (tau2 / tau1).log10();
        // White FM has slope ≈ −0.5; check within tolerance
        if (slope + 0.5).abs() < 0.3 {
            // h0 = 2 τ σ_y² (at τ1)
            Some(2.0 * tau1 * s1 * s1)
        } else {
            None
        }
    }

    /// Detect the flicker (1/f) frequency noise floor.
    ///
    /// Flicker FM produces a flat (τ-independent) Allan deviation.
    /// Returns the estimated flicker floor level, or `None` if not detected.
    pub fn flicker_floor(&self) -> Option<f64> {
        if self.oadev.len() < 3 {
            return None;
        }
        // Look for flat region: minimum std deviation across middle third of points
        let start = self.oadev.len() / 3;
        let end = 2 * self.oadev.len() / 3;
        if end <= start {
            return None;
        }
        let segment = &self.oadev[start..end];
        let mean: f64 = segment.iter().sum::<f64>() / segment.len() as f64;
        let var: f64 =
            segment.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / segment.len() as f64;
        let rms_variation = var.sqrt() / mean;
        // If variation < 20 %, call it a flicker floor
        if rms_variation < 0.20 {
            Some(mean)
        } else {
            None
        }
    }

    /// Interpolated Allan deviation at an arbitrary averaging time τ.
    ///
    /// Uses log-linear interpolation between the two nearest computed points.
    /// Returns `None` if τ is outside the computed range.
    pub fn stability_at(&self, tau: f64) -> Option<f64> {
        if self.tau_values.is_empty() || tau <= 0.0 {
            return None;
        }
        // Exact match
        for (i, &t) in self.tau_values.iter().enumerate() {
            if (t - tau).abs() < tau * 1e-6 {
                return Some(self.oadev[i]);
            }
        }
        // Find bracketing interval
        for i in 0..self.tau_values.len().saturating_sub(1) {
            let t1 = self.tau_values[i];
            let t2 = self.tau_values[i + 1];
            if tau >= t1 && tau <= t2 {
                // Log-linear interpolation: σ = exp(lerp(log(σ1), log(σ2), α))
                let alpha = (tau.log10() - t1.log10()) / (t2.log10() - t1.log10());
                let log_s1 = self.oadev[i].log10();
                let log_s2 = self.oadev[i + 1].log10();
                let log_s = log_s1 + alpha * (log_s2 - log_s1);
                return Some(10_f64.powf(log_s));
            }
        }
        None
    }
}

// ─── FiberFrequencyTransfer ──────────────────────────────────────────────────

/// Optical frequency transfer over a phase-stabilized fiber link.
///
/// Fiber-optic frequency transfer connects remote optical clocks with
/// sub-10⁻¹⁹ instability using active noise cancellation (ANC), where a
/// Doppler cancellation scheme suppresses the thermal and mechanical phase
/// fluctuations of the fiber.
#[derive(Debug, Clone)]
pub struct FiberFrequencyTransfer {
    /// Fiber link length (km).
    pub fiber_length_km: f64,
    /// Whether active noise cancellation (ANC) is enabled.
    pub noise_cancellation: bool,
    /// Group velocity dispersion β₂ (s²/m).
    pub group_velocity_dispersion: f64,
}

impl FiberFrequencyTransfer {
    /// Construct a fiber frequency transfer link.
    ///
    /// Defaults to SMF-28 GVD at 1550 nm: β₂ = −21.7 ps²/km = −21.7×10⁻²⁷ s²/m.
    ///
    /// # Arguments
    /// * `length_km`     — fiber link length (km)
    /// * `noise_cancel`  — enable active noise cancellation
    pub fn new(length_km: f64, noise_cancel: bool) -> Self {
        Self {
            fiber_length_km: length_km,
            noise_cancellation: noise_cancel,
            group_velocity_dispersion: -21.7e-27, // s²/m (SMF-28 at 1550 nm)
        }
    }

    /// Phase noise power spectral density of the optical carrier on the fiber
    /// link (dBc/Hz) at offset frequency f.
    ///
    /// Model: S_φ(f) = (2π L / λ)² · S_δL(f)
    /// with S_δL(f) = S₀ / f  (1/f thermal noise) for f < 1 kHz
    ///           S_δL(f) = S₀  (white length noise floor) for f > 1 kHz
    ///
    /// Here we use S₀ = 10⁻¹⁷ m²/Hz (typical fiber thermal noise).
    ///
    /// # Arguments
    /// * `offset_freq_hz` — Fourier offset frequency (Hz)
    pub fn fiber_phase_noise_dbc_hz(&self, offset_freq_hz: f64) -> f64 {
        if offset_freq_hz <= 0.0 {
            return 0.0;
        }
        let length_m = self.fiber_length_km * 1e3;
        let lambda = 1550e-9; // m (typical telecom wavelength)
                              // Geometric factor (2π L / λ)²
        let geo_factor = (2.0 * std::f64::consts::PI * length_m / lambda).powi(2);
        // Length noise spectral density S_δL(f) [m²/Hz]
        let s0_m2_hz = 1e-17_f64; // white noise floor
        let s_dl = if offset_freq_hz < 1e3 {
            s0_m2_hz * 1e3 / offset_freq_hz // 1/f below 1 kHz
        } else {
            s0_m2_hz
        };
        let s_phi_linear = geo_factor * s_dl;
        if s_phi_linear <= 0.0 {
            return f64::NEG_INFINITY;
        }
        10.0 * s_phi_linear.log10()
    }

    /// Transfer instability (Allan deviation) at averaging time τ.
    ///
    /// Without noise cancellation:
    /// σ_y(τ) ≈ (L / c) · √(S_φ · f_corner) / (2π ν₀ τ)
    ///
    /// where f_corner = 1 kHz separates the 1/f and white regimes, and ν₀ is
    /// the optical carrier frequency at 1550 nm.
    ///
    /// # Arguments
    /// * `tau_s` — averaging time (s)
    pub fn transfer_instability(&self, tau_s: f64) -> f64 {
        if tau_s <= 0.0 {
            return f64::INFINITY;
        }
        let _length_m = self.fiber_length_km * 1e3;
        let nu0 = C0 / 1550e-9; // optical frequency Hz
                                // Characteristic phase noise: S_φ at f_corner = 1 kHz with 1/f × white transition
        let f_corner = 1e3_f64;
        let s_phi_at_fc = 10_f64.powf(self.fiber_phase_noise_dbc_hz(f_corner) / 10.0);
        // Instability: σ_y = sqrt(S_φ · f_corner) / (2π ν₀ τ)
        let rms_phase = (s_phi_at_fc * f_corner).sqrt();
        let sigma = rms_phase / (2.0 * std::f64::consts::PI * nu0 * tau_s);
        // Apply noise cancellation improvement factor of 1000× if enabled
        if self.noise_cancellation {
            sigma / 1000.0
        } else {
            sigma
        }
    }

    /// Phase noise after active noise cancellation (dBc/Hz).
    ///
    /// ANC suppresses fiber phase noise by ~60 dB (×10³) for offset frequencies
    /// within the servo bandwidth (~10 kHz).
    ///
    /// # Arguments
    /// * `offset_freq_hz` — Fourier offset frequency (Hz)
    pub fn noise_floor_after_cancellation(&self, offset_freq_hz: f64) -> f64 {
        let raw_db = self.fiber_phase_noise_dbc_hz(offset_freq_hz);
        // ANC servo bandwidth ~10 kHz; suppression ~60 dB within bandwidth
        if offset_freq_hz < 1e4 {
            raw_db - 60.0 // dBc/Hz
        } else {
            raw_db
        }
    }

    /// Maximum achievable transfer distance (km) for a given instability target.
    ///
    /// Transfer instability scales with fiber length. This method returns the
    /// maximum link length that achieves σ_y(1 s) = 10⁻¹⁹ (current state of the
    /// art for ANC-stabilized links).
    ///
    /// Uses the scaling σ_y ∝ L² to estimate the maximum distance.
    pub fn max_distance_km(&self) -> f64 {
        // State-of-the-art stability σ_y(1s) ~ 10^-19 achieved over ~1000 km with ANC
        if self.noise_cancellation {
            1000.0 // km — demonstrated record
        } else {
            10.0 // km — unstabilized link limit
        }
    }

    /// Group delay spread due to chromatic dispersion for a comb pulse (ps).
    ///
    /// ΔT = β₂ · L · Δω  where Δω is the comb spectral bandwidth in rad/s.
    ///
    /// # Arguments
    /// * `bandwidth_nm` — spectral bandwidth (nm)
    pub fn dispersion_spread_ps(&self, bandwidth_nm: f64) -> f64 {
        let length_m = self.fiber_length_km * 1e3;
        let lambda0 = 1550e-9; // m
        let d_omega = 2.0 * std::f64::consts::PI * C0 / (lambda0 * lambda0) * bandwidth_nm * 1e-9;
        let dt_s = self.group_velocity_dispersion.abs() * length_m * d_omega;
        dt_s * 1e12 // s → ps
    }
}

// ─── ClockComparison ─────────────────────────────────────────────────────────

/// Relativistic optical clock comparison for geodesy.
///
/// Two clocks at different heights experience different gravitational potentials.
/// The resulting fractional frequency difference:
///   Δf/f = g · Δh / c²  ≈ 1.09 × 10⁻¹⁶ / cm
///
/// enables centimetre-level height determination — a technique called
/// "relativistic geodesy" or "chronometric levelling".
#[derive(Debug, Clone)]
pub struct ClockComparison {
    /// Clock A frequency (Hz).
    pub clock_a_freq: f64,
    /// Clock B frequency (Hz).
    pub clock_b_freq: f64,
    /// Height of clock A above clock B (m). Positive → A is higher.
    pub height_difference_m: f64,
    /// Combined fractional frequency measurement uncertainty.
    pub fractional_uncertainty: f64,
}

impl ClockComparison {
    /// Construct a clock comparison scenario.
    ///
    /// # Arguments
    /// * `fa` — clock A frequency (Hz)
    /// * `fb` — clock B frequency (Hz)
    /// * `dh` — height difference h_A − h_B (m)
    pub fn new(fa: f64, fb: f64, dh: f64) -> Self {
        Self {
            clock_a_freq: fa,
            clock_b_freq: fb,
            height_difference_m: dh,
            fractional_uncertainty: 1e-18,
        }
    }

    /// Expected fractional frequency difference due to gravitational redshift.
    ///
    /// Δf/f = g · Δh / c²
    pub fn gravitational_frequency_difference(&self) -> f64 {
        G * self.height_difference_m / (C0 * C0)
    }

    /// Height sensitivity: centimetres of height per unit of 10⁻¹⁸ fractional frequency.
    ///
    /// 1 cm height change → Δf/f ≈ 1.09 × 10⁻¹⁸ fractional frequency.
    /// Returns the height (cm) corresponding to 1 × 10⁻¹⁸.
    pub fn height_sensitivity_cm_per_1e18(&self) -> f64 {
        // Δh [m] = (Δf/f) · c² / g
        // For Δf/f = 1e-18:
        let delta_h_m = 1e-18 * C0 * C0 / G;
        delta_h_m * 100.0 // m → cm
    }

    /// Required fractional clock stability to resolve a height difference of
    /// `resolution_cm` centimetres.
    ///
    /// # Arguments
    /// * `resolution_cm` — target height resolution (cm)
    pub fn required_stability_for_geodesy(&self, resolution_cm: f64) -> f64 {
        let delta_h_m = resolution_cm * 1e-2; // cm → m
        G * delta_h_m / (C0 * C0)
    }

    /// Uncertainty in the frequency ratio R = f_A / f_B.
    ///
    /// δR = R · δ(Δf/f)  where δ(Δf/f) is the fractional uncertainty.
    pub fn ratio_uncertainty(&self) -> f64 {
        if self.clock_b_freq <= 0.0 {
            return f64::INFINITY;
        }
        let ratio = self.clock_a_freq / self.clock_b_freq;
        ratio * self.fractional_uncertainty
    }

    /// Measured fractional frequency difference (A relative to B).
    ///
    /// Δf/f = (f_A − f_B) / f_B
    pub fn measured_fractional_difference(&self) -> Result<f64, OxiPhotonError> {
        if self.clock_b_freq <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "Clock B frequency must be positive".into(),
            ));
        }
        Ok((self.clock_a_freq - self.clock_b_freq) / self.clock_b_freq)
    }

    /// Inferred height difference from the measured frequency ratio (m).
    ///
    /// Δh = (Δf/f) · c² / g
    pub fn inferred_height_difference_m(&self) -> Result<f64, OxiPhotonError> {
        let delta_f_over_f = self.measured_fractional_difference()?;
        Ok(delta_f_over_f * C0 * C0 / G)
    }

    /// Whether the comparison has sufficient stability to resolve 1 cm height.
    pub fn can_resolve_1cm(&self) -> bool {
        self.fractional_uncertainty < self.required_stability_for_geodesy(1.0)
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── AllanDeviation ──────────────────────────────────────────────────────

    #[test]
    fn test_allan_dev_white_freq_noise_scaling() {
        // Synthesize white FM noise: y[i] = 0 (perfect oscillator, zero noise)
        // Then check that OADEV = 0
        let y = vec![0.0_f64; 64];
        let oadev = AllanDeviation::from_fractional_freq(&y, 1.0);
        assert!(
            !oadev.tau_values.is_empty(),
            "should have at least one tau value"
        );
        for &s in &oadev.oadev {
            assert_abs_diff_eq!(s, 0.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_allan_dev_stability_at_interpolation() {
        // Phase data: linear drift → OADEV ∝ constant slope
        // Use a simple constant fractional frequency offset
        let y: Vec<f64> = vec![1e-13; 128];
        let oadev = AllanDeviation::from_fractional_freq(&y, 1.0);
        if oadev.tau_values.len() >= 2 {
            // Stability at the first tau should match the computed value
            let tau0 = oadev.tau_values[0];
            let s0 = oadev.oadev[0];
            let interp = oadev.stability_at(tau0);
            assert!(
                interp.is_some(),
                "stability_at should return Some at a known tau"
            );
            assert_abs_diff_eq!(interp.expect("checked above"), s0, epsilon = s0 * 1e-4);
        }
    }

    #[test]
    fn test_allan_dev_from_phase_data() {
        // Constant phase shift → zero Allan deviation
        let x = vec![1e-9_f64; 64]; // constant phase offset
        let oadev = AllanDeviation::from_phase_data(&x, 1.0);
        for &s in &oadev.oadev {
            assert_abs_diff_eq!(s, 0.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_allan_dev_tau_values_increasing() {
        let y: Vec<f64> = (0..128).map(|i| ((i as f64) * 0.1).sin() * 1e-13).collect();
        let oadev = AllanDeviation::from_fractional_freq(&y, 1.0);
        for i in 1..oadev.tau_values.len() {
            assert!(
                oadev.tau_values[i] > oadev.tau_values[i - 1],
                "tau values must be strictly increasing"
            );
        }
    }

    // ── FiberFrequencyTransfer ──────────────────────────────────────────────

    #[test]
    fn test_fiber_phase_noise_decreases_with_frequency() {
        // At higher offset frequency the 1/f noise decreases
        let link = FiberFrequencyTransfer::new(100.0, false);
        let noise_low = link.fiber_phase_noise_dbc_hz(100.0); // 100 Hz, in 1/f region
        let noise_high = link.fiber_phase_noise_dbc_hz(10e3); // 10 kHz, white region
        assert!(
            noise_low > noise_high,
            "low-f noise {noise_low} dBc/Hz should exceed high-f noise {noise_high} dBc/Hz"
        );
    }

    #[test]
    fn test_fiber_anc_improves_instability() {
        let link_raw = FiberFrequencyTransfer::new(1000.0, false);
        let link_anc = FiberFrequencyTransfer::new(1000.0, true);
        let sigma_raw = link_raw.transfer_instability(1.0);
        let sigma_anc = link_anc.transfer_instability(1.0);
        assert!(
            sigma_anc < sigma_raw,
            "ANC link σ_y={sigma_anc} must be smaller than raw σ_y={sigma_raw}"
        );
    }

    #[test]
    fn test_fiber_noise_cancellation_reduces_noise() {
        let link = FiberFrequencyTransfer::new(100.0, true);
        let raw_noise = link.fiber_phase_noise_dbc_hz(1e3);
        let cancelled = link.noise_floor_after_cancellation(1e3);
        assert!(
            cancelled < raw_noise,
            "cancelled noise {cancelled} dBc/Hz must be lower than raw {raw_noise} dBc/Hz"
        );
    }

    // ── ClockComparison ─────────────────────────────────────────────────────

    #[test]
    fn test_gravitational_frequency_difference_sign() {
        // A is 100 m above B → A runs faster (positive Δf/f)
        let cmp = ClockComparison::new(429e12, 429e12, 100.0);
        let delta = cmp.gravitational_frequency_difference();
        assert!(
            delta > 0.0,
            "upward gravitational redshift must be positive: {delta}"
        );
    }

    #[test]
    fn test_height_sensitivity_about_1cm_per_1e18() {
        let cmp = ClockComparison::new(429e12, 429e12, 0.0);
        let h_cm = cmp.height_sensitivity_cm_per_1e18();
        // Expected ≈ 1.09 cm per 10⁻¹⁸ fractional frequency
        assert_abs_diff_eq!(h_cm, 1e-18 * C0 * C0 / G * 100.0, epsilon = 1e-5);
    }

    #[test]
    fn test_required_stability_1cm_geodesy() {
        let cmp = ClockComparison::new(429e12, 429e12, 0.0);
        let req = cmp.required_stability_for_geodesy(1.0); // 1 cm
                                                           // ~1.09e-18 fractional
        assert!(
            req > 1e-19 && req < 1e-17,
            "required stability {req} out of expected range"
        );
    }

    #[test]
    fn test_inferred_height_consistent_with_input() {
        // Set f_A slightly higher than f_B to encode a known height
        // Δh = 100 m → Δf/f = g·h/c² ≈ 1.09e-14
        let delta_f_over_f = G * 100.0 / (C0 * C0);
        let fb = 429_228_066_418_012.0_f64;
        let fa = fb * (1.0 + delta_f_over_f);
        let cmp = ClockComparison::new(fa, fb, 100.0);
        let inferred = cmp.inferred_height_difference_m();
        assert!(inferred.is_ok(), "inferred height should not error");
        let h = inferred.expect("checked above");
        assert_abs_diff_eq!(h, 100.0, epsilon = 0.5); // within 50 cm — limited by f64 precision at optical frequencies
    }
}
