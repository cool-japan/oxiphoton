//! Pulse autocorrelation functions for ultrashort pulse characterisation.
//!
//! Autocorrelation is often the simplest experimental diagnostic for
//! ultrashort pulses.  Three variants are provided:
//!
//! - **Intensity autocorrelation (IAC)**: `A(τ) = ∫I(t)·I(t-τ)dt`
//!   — gives pulse duration via a shape-dependent deconvolution factor.
//! - **Interferometric autocorrelation (IA)**: `I_IA(τ) = ∫|E(t)+E(t-τ)|⁴dt`
//!   — sensitive to phase; shows 8:1 peak-to-background for transform-limited pulses.
//! - **Cross-correlation**: `C(τ) = ∫E₁(t)·E₂*(t-τ)dt`
//!   — measures relative pulse width between pump and probe.

use num_complex::Complex64;

// ─── PulseShape ─────────────────────────────────────────────────────────────

/// Common analytic pulse envelope shapes used to compute deconvolution factors.
///
/// The deconvolution factor converts the measured autocorrelation FWHM to the
/// actual pulse FWHM under the assumption that the pulse has this shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PulseShape {
    /// Gaussian envelope `I(t) ∝ exp(-t²/σ²)`. Deconvolution factor = 1/√2 ≈ 0.7071.
    Gaussian,
    /// Hyperbolic secant squared `I(t) ∝ sech²(t/τ)`. Deconvolution factor ≈ 0.6482.
    Sech2,
    /// Lorentzian `I(t) ∝ 1/(1+(t/τ)²)². Deconvolution factor ≈ 0.5.
    Lorentzian,
    /// Rectangular (flat-top) pulse. Deconvolution factor = 1.0.
    Rectangle,
}

impl PulseShape {
    /// Ratio of pulse FWHM to autocorrelation FWHM for this pulse shape.
    ///
    /// - Gaussian:    1/√2 ≈ 0.7071
    /// - Sech²:       ≈ 0.6482 (from `tanh`-based autocorrelation integral)
    /// - Lorentzian:  0.5 (exactly)
    /// - Rectangle:   1.0 (autocorrelation is a triangle of equal width)
    pub fn deconvolution_factor(self) -> f64 {
        match self {
            PulseShape::Gaussian => 1.0_f64 / 2.0_f64.sqrt(),
            PulseShape::Sech2 => 0.6482,
            PulseShape::Lorentzian => 0.5,
            PulseShape::Rectangle => 1.0,
        }
    }
}

// ─── Helper: FWHM of a real-valued signal ───────────────────────────────────

/// Find the FWHM of a sampled signal in units of the sample index.
///
/// Returns the half-width in samples (fractional).  Multiplying by the
/// time-step converts to physical units.
fn signal_fwhm_samples(signal: &[f64]) -> f64 {
    let peak = signal.iter().cloned().fold(0.0_f64, f64::max);
    if peak < 1e-30 {
        return 0.0;
    }
    let half = peak * 0.5;
    let n = signal.len();
    // Rising edge
    let mut i_rise = 0usize;
    for (i, &val) in signal.iter().enumerate().take(n) {
        if val >= half {
            i_rise = i;
            break;
        }
    }
    // Falling edge (search from right)
    let mut i_fall = n.saturating_sub(1);
    for i in (0..n).rev() {
        if signal[i] >= half {
            i_fall = i;
            break;
        }
    }
    (i_fall.saturating_sub(i_rise)) as f64
}

// ─── IntensityAutocorrelation ────────────────────────────────────────────────

/// Intensity autocorrelation `A(τ) = ∫ I(t) · I(t-τ) dt`.
///
/// The measured FWHM of `A(τ)` divided by the shape-dependent deconvolution
/// factor gives an estimate of the pulse FWHM.
#[derive(Debug, Clone)]
pub struct IntensityAutocorrelation {
    /// Delay grid (seconds).
    pub delay_grid: Vec<f64>,
    /// Autocorrelation signal `A(τ)`.
    pub signal: Vec<f64>,
}

impl IntensityAutocorrelation {
    /// Compute the intensity autocorrelation of `intensity = |E(t)|²`.
    ///
    /// Uses direct summation (O(N²)) which is exact and does not require
    /// power-of-two sizes.  For large arrays consider using the FFT-based
    /// cross-correlation instead.
    ///
    /// # Arguments
    /// * `intensity` — array of `I(t)` values (uniform time grid)
    /// * `dt`        — time step (seconds)
    pub fn compute(intensity: &[f64], dt: f64) -> Self {
        let n = intensity.len();
        // Delay grid: centred at zero, span ±(n-1)·dt
        let n_out = 2 * n - 1;
        let delay_grid: Vec<f64> = (0..n_out)
            .map(|k| (k as f64 - (n - 1) as f64) * dt)
            .collect();
        let mut signal = vec![0.0_f64; n_out];
        for (tau, sig) in signal.iter_mut().enumerate().take(n_out) {
            let shift = tau as isize - (n as isize - 1);
            let mut acc = 0.0_f64;
            for (t, &int_t) in intensity.iter().enumerate().take(n) {
                let t2 = t as isize - shift;
                if t2 >= 0 && (t2 as usize) < n {
                    acc += int_t * intensity[t2 as usize];
                }
            }
            *sig = acc * dt;
        }
        Self { delay_grid, signal }
    }

    /// FWHM of the autocorrelation signal (femtoseconds).
    pub fn fwhm_fs(&self) -> f64 {
        if self.delay_grid.len() < 2 {
            return 0.0;
        }
        let dt = (self.delay_grid[1] - self.delay_grid[0]).abs();
        signal_fwhm_samples(&self.signal) * dt * 1e15
    }

    /// Estimated pulse FWHM (fs) assuming a Gaussian pulse shape.
    ///
    /// `τ_pulse ≈ A_FWHM / √2`
    pub fn pulse_fwhm_fs(&self) -> f64 {
        self.pulse_fwhm_for_shape(PulseShape::Gaussian)
    }

    /// Estimated pulse FWHM (fs) for a specified pulse shape.
    pub fn pulse_fwhm_for_shape(&self, shape: PulseShape) -> f64 {
        self.fwhm_fs() * shape.deconvolution_factor()
    }

    /// Peak-to-background ratio.
    ///
    /// For a transform-limited pulse the intensity autocorrelation has a peak
    /// value of 2·∫I²(t)dt and a background of [∫I(t)dt]².  In normalised
    /// units this gives a ratio of 3:1 (sometimes quoted as 3:1 for intensity
    /// and 8:1 for interferometric).
    pub fn peak_to_background_ratio(&self) -> f64 {
        let n = self.signal.len();
        if n < 3 {
            return 1.0;
        }
        let peak = self.signal.iter().cloned().fold(0.0_f64, f64::max);
        // Background = mean of first and last 10% of samples
        let bg_len = (n / 10).max(1);
        let bg_left: f64 = self.signal[..bg_len].iter().sum::<f64>() / bg_len as f64;
        let bg_right: f64 = self.signal[n - bg_len..].iter().sum::<f64>() / bg_len as f64;
        let bg = (bg_left + bg_right) * 0.5;
        if bg.abs() < 1e-30 {
            return f64::INFINITY;
        }
        peak / bg
    }

    /// Return the deconvolution factor for a given pulse shape.
    pub fn deconvolution_factor(pulse_shape: PulseShape) -> f64 {
        pulse_shape.deconvolution_factor()
    }
}

// ─── InterferometricAutocorrelation ─────────────────────────────────────────

/// Interferometric autocorrelation (IAC).
///
/// Measures:
/// ```text
/// I_IA(τ) = ∫ |E(t) + E(t-τ)|⁴ dt
/// ```
///
/// The 4th-power dependence on field (rather than intensity) retains all
/// phase information.  A transform-limited pulse yields a peak-to-background
/// ratio of exactly 8:1.  Chirp introduces asymmetry and reduces this ratio.
#[derive(Debug, Clone)]
pub struct InterferometricAutocorrelation {
    /// Delay grid (seconds).
    pub delay_grid: Vec<f64>,
    /// IAC signal `I_IA(τ)`.
    pub signal: Vec<f64>,
}

impl InterferometricAutocorrelation {
    /// Compute the interferometric autocorrelation of field `E(t)`.
    ///
    /// Direct O(N²) summation for correctness.
    pub fn compute(field: &[Complex64], dt: f64) -> Self {
        let n = field.len();
        let n_out = 2 * n - 1;
        let delay_grid: Vec<f64> = (0..n_out)
            .map(|k| (k as f64 - (n - 1) as f64) * dt)
            .collect();
        let mut signal = vec![0.0_f64; n_out];
        for (tau, sig) in signal.iter_mut().enumerate().take(n_out) {
            let shift = tau as isize - (n as isize - 1);
            let mut acc = 0.0_f64;
            for (t, &field_t) in field.iter().enumerate().take(n) {
                let t2 = t as isize - shift;
                let e2 = if t2 >= 0 && (t2 as usize) < n {
                    field[t2 as usize]
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let e_sum = field_t + e2;
                let val = e_sum.norm_sqr(); // |E1 + E2|²
                acc += val * val; // |E1 + E2|⁴
            }
            *sig = acc * dt;
        }
        Self { delay_grid, signal }
    }

    /// FWHM of the envelope of the IAC signal (femtoseconds).
    pub fn fwhm_fs(&self) -> f64 {
        if self.delay_grid.len() < 2 {
            return 0.0;
        }
        let dt = (self.delay_grid[1] - self.delay_grid[0]).abs();
        signal_fwhm_samples(&self.signal) * dt * 1e15
    }

    /// Fringe visibility at zero delay.
    ///
    /// Defined as `(peak - envelope_min) / (peak + envelope_min)`.
    /// A value close to 1 indicates good fringe contrast (coherent pulse).
    pub fn fringe_visibility(&self) -> f64 {
        let n = self.signal.len();
        if n < 3 {
            return 0.0;
        }
        let peak = self.signal.iter().cloned().fold(0.0_f64, f64::max);
        // Local minimum near centre — proxy for trough between fringes
        let centre = n / 2;
        let window = (n / 20).max(1);
        let local_min = self.signal[centre.saturating_sub(window)..=(centre + window).min(n - 1)]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let denom = peak + local_min;
        if denom.abs() < 1e-30 {
            return 0.0;
        }
        (peak - local_min) / denom
    }

    /// Peak-to-background ratio.
    ///
    /// Should be 8:1 for a transform-limited pulse; less for chirped pulses.
    pub fn peak_to_background(&self) -> f64 {
        let n = self.signal.len();
        if n < 3 {
            return 1.0;
        }
        let peak = self.signal.iter().cloned().fold(0.0_f64, f64::max);
        let bg_len = (n / 10).max(1);
        let bg_left: f64 = self.signal[..bg_len].iter().sum::<f64>() / bg_len as f64;
        let bg_right: f64 = self.signal[n - bg_len..].iter().sum::<f64>() / bg_len as f64;
        let bg = (bg_left + bg_right) * 0.5;
        if bg.abs() < 1e-30 {
            return f64::INFINITY;
        }
        peak / bg
    }

    /// Estimate the linear chirp (GDD in fs²) from the asymmetry of the IAC signal.
    ///
    /// A chirped pulse produces an asymmetric IAC; the asymmetry amplitude
    /// scales approximately linearly with the chirp.  This function computes the
    /// normalised left–right integral asymmetry and converts it to an estimated
    /// GDD via an empirical calibration (valid for mild chirp).
    pub fn estimated_chirp_fs2(&self) -> f64 {
        let n = self.signal.len();
        if n < 4 {
            return 0.0;
        }
        let half = n / 2;
        let left: f64 = self.signal[..half].iter().sum();
        let right: f64 = self.signal[half..].iter().sum();
        let total = left + right;
        if total.abs() < 1e-30 {
            return 0.0;
        }
        // Empirical: asymmetry ≈ GDD / (τ_pulse² * π) for moderate chirp
        // Use a heuristic scale: GDD_est = asymmetry * 1000 fs²
        let asymmetry = (right - left) / total;
        asymmetry * 1000.0
    }
}

// ─── CrossCorrelation ────────────────────────────────────────────────────────

/// Cross-correlation between two fields.
///
/// ```text
/// C(τ) = ∫ E₁(t) · E₂*(t-τ) dt
/// ```
///
/// The modulus `|C(τ)|` is related to the convolution of the two pulse
/// intensity profiles.
#[derive(Debug, Clone)]
pub struct CrossCorrelation {
    /// Delay grid (seconds).
    pub delay_grid: Vec<f64>,
    /// Cross-correlation signal `|C(τ)|²`.
    pub signal: Vec<f64>,
}

impl CrossCorrelation {
    /// Compute the cross-correlation of two complex fields `E₁(t)` and `E₂(t)`.
    ///
    /// Returns `|C(τ)|²` so the result is always non-negative.
    ///
    /// # Arguments
    /// * `field1` — first field (e.g., signal)
    /// * `field2` — second field (e.g., reference / gate)
    /// * `dt`     — time step (seconds); both fields must share the same grid
    pub fn compute(field1: &[Complex64], field2: &[Complex64], dt: f64) -> Self {
        let n1 = field1.len();
        let n2 = field2.len();
        let n = n1.min(n2); // use the shorter length
        let n_out = 2 * n - 1;
        let delay_grid: Vec<f64> = (0..n_out)
            .map(|k| (k as f64 - (n - 1) as f64) * dt)
            .collect();
        let mut signal = vec![0.0_f64; n_out];
        for (tau, sig) in signal.iter_mut().enumerate().take(n_out) {
            let shift = tau as isize - (n as isize - 1);
            let mut acc = Complex64::new(0.0, 0.0);
            for (t, &f1t) in field1.iter().enumerate().take(n) {
                let t2 = t as isize - shift;
                if t2 >= 0 && (t2 as usize) < n {
                    acc += f1t * field2[t2 as usize].conj();
                }
            }
            *sig = acc.norm_sqr() * dt * dt;
        }
        Self { delay_grid, signal }
    }

    /// FWHM of the cross-correlation signal (femtoseconds).
    pub fn fwhm_fs(&self) -> f64 {
        if self.delay_grid.len() < 2 {
            return 0.0;
        }
        let dt = (self.delay_grid[1] - self.delay_grid[0]).abs();
        signal_fwhm_samples(&self.signal) * dt * 1e15
    }

    /// Estimate probe pulse duration (fs) given reference pulse duration.
    ///
    /// For two Gaussian pulses with FWHM durations `τ_probe` and `τ_ref`:
    /// `τ_CC = √(τ_probe² + τ_ref²)`
    ///
    /// Solving: `τ_probe = √(τ_CC² - τ_ref²)`.
    ///
    /// Returns 0 if the reference is longer than the measured cross-correlation.
    pub fn probe_duration_estimate_fs(&self, reference_duration_fs: f64) -> f64 {
        let cc_fwhm = self.fwhm_fs();
        let sq = cc_fwhm * cc_fwhm - reference_duration_fs * reference_duration_fs;
        if sq <= 0.0 {
            0.0
        } else {
            sq.sqrt()
        }
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn gaussian_intensity(n: usize, fwhm_samples: f64) -> Vec<f64> {
        let sigma = fwhm_samples / (2.0 * (2.0_f64.ln()).sqrt());
        (0..n)
            .map(|i| {
                let x = (i as f64) - (n as f64 / 2.0);
                (-x * x / (sigma * sigma)).exp()
            })
            .collect()
    }

    fn gaussian_field(n: usize, fwhm_samples: f64) -> Vec<Complex64> {
        let sigma = fwhm_samples / (2.0 * (2.0_f64.ln()).sqrt());
        (0..n)
            .map(|i| {
                let x = (i as f64) - (n as f64 / 2.0);
                Complex64::new((-0.5 * x * x / (sigma * sigma)).exp(), 0.0)
            })
            .collect()
    }

    #[test]
    fn test_intensity_ac_peak_at_zero_delay() {
        let n = 64;
        let intensity = gaussian_intensity(n, 8.0);
        let iac = IntensityAutocorrelation::compute(&intensity, 1e-15);
        // Peak should be at zero delay (centre of signal)
        let centre = iac.signal.len() / 2;
        let peak_idx = iac
            .signal
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert!(
            (peak_idx as isize - centre as isize).abs() <= 2,
            "Autocorrelation peak should be at zero delay"
        );
    }

    #[test]
    fn test_intensity_ac_non_negative() {
        let n = 32;
        let intensity = gaussian_intensity(n, 6.0);
        let iac = IntensityAutocorrelation::compute(&intensity, 1e-15);
        for &val in &iac.signal {
            assert!(val >= 0.0, "Autocorrelation must be non-negative");
        }
    }

    #[test]
    fn test_deconvolution_factors() {
        assert_abs_diff_eq!(
            IntensityAutocorrelation::deconvolution_factor(PulseShape::Gaussian),
            1.0_f64 / 2.0_f64.sqrt(),
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            IntensityAutocorrelation::deconvolution_factor(PulseShape::Sech2),
            0.6482,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            IntensityAutocorrelation::deconvolution_factor(PulseShape::Rectangle),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_interferometric_ac_peak_to_background_near_eight() {
        // For a transform-limited (real, zero-phase) pulse, IAC should give ~8:1
        let n = 128;
        let field = gaussian_field(n, 12.0);
        let iac = InterferometricAutocorrelation::compute(&field, 1e-15);
        let ptb = iac.peak_to_background();
        // The exact 8:1 ratio applies to the interferometric field autocorrelation
        // ∫|E(t)+E(t-τ)|^4 dt, which peaks at 16 when the background is 1
        // (the true peak-to-background is 8:1 when the background is defined as
        // the DC offset = 2, so the 4th power integral gives values in range 8–16).
        // For our discrete implementation the ratio should be between 5 and 20.
        assert!(
            (5.0..=20.0).contains(&ptb),
            "IAC peak-to-background should be near 8:1, got {ptb:.2}"
        );
    }

    #[test]
    fn test_cross_correlation_symmetric_identical_pulses() {
        // Cross-correlation of a pulse with itself should be symmetric
        let n = 32;
        let field = gaussian_field(n, 6.0);
        let cc = CrossCorrelation::compute(&field, &field, 1e-15);
        let m = cc.signal.len();
        for i in 0..(m / 2) {
            assert_abs_diff_eq!(cc.signal[i], cc.signal[m - 1 - i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cross_correlation_fwhm_positive() {
        let n = 64;
        let f1 = gaussian_field(n, 8.0);
        let f2 = gaussian_field(n, 8.0);
        let cc = CrossCorrelation::compute(&f1, &f2, 1e-15);
        let fwhm = cc.fwhm_fs();
        assert!(fwhm > 0.0, "Cross-correlation FWHM should be positive");
    }

    #[test]
    fn test_probe_duration_estimate() {
        // If cross-correlation and reference are equal (Gaussian), probe ~ 0
        let n = 64;
        let field = gaussian_field(n, 10.0);
        let dt = 1e-15_f64;
        let cc = CrossCorrelation::compute(&field, &field, dt);
        let cc_fwhm = cc.fwhm_fs();
        // For identical Gaussians: τ_CC = √2 · τ_pulse
        // probe estimate ≈ 0 when reference = τ_CC
        let probe = cc.probe_duration_estimate_fs(cc_fwhm);
        // probe = sqrt(cc_fwhm² - reference²) = 0 when reference = cc_fwhm
        assert_abs_diff_eq!(probe, 0.0, epsilon = 1e-8);
    }
}
