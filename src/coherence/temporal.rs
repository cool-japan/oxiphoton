/// Temporal Coherence — autocorrelation functions and power spectral density.
///
/// Implements the Wiener-Khinchin theorem connecting the autocorrelation
/// function Γ(τ) = <E*(t) E(t+τ)> to the power spectral density S(ω).
///
/// Key quantities:
/// - Coherence time  τ_c = ∫|γ(τ)|² dτ
/// - Coherence length lc  = c · τ_c
/// - Visibility in Michelson interferometer V = |γ(path_diff/c)|
use std::f64::consts::PI;

use num_complex::Complex64;

/// Speed of light in vacuum \[m/s\].
const C: f64 = 2.997_924_58e8;

/// Error type for temporal coherence calculations.
#[derive(Debug, Clone, PartialEq)]
pub enum TemporalCoherenceError {
    /// Input arrays have different lengths.
    LengthMismatch { expected: usize, got: usize },
    /// A parameter lies outside its valid domain.
    InvalidParameter(String),
    /// The spectrum or autocorrelation is all-zero (undefined coherence time).
    ZeroSpectrum,
}

impl std::fmt::Display for TemporalCoherenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch { expected, got } => {
                write!(f, "Length mismatch: expected {expected}, got {got}")
            }
            Self::InvalidParameter(msg) => write!(f, "Invalid parameter: {msg}"),
            Self::ZeroSpectrum => write!(f, "Spectrum is identically zero"),
        }
    }
}

impl std::error::Error for TemporalCoherenceError {}

// ─────────────────────────────────────────────────────────────────────────────
// TemporalCoherence
// ─────────────────────────────────────────────────────────────────────────────

/// Temporal coherence function Γ(τ) = <E*(t) E(t+τ)>.
///
/// Computed from the power spectral density via the Wiener-Khinchin theorem:
///   Γ(τ) = ∫ S(ω) exp(iωτ) dω / (2π)
///
/// The autocorrelation is stored for a uniform grid of delay values τ.
#[derive(Debug, Clone)]
pub struct TemporalCoherence {
    /// Delay values τ \[s\] (uniform grid, zero-centred).
    pub taus: Vec<f64>,
    /// Complex autocorrelation Γ(τ).
    pub gamma: Vec<Complex64>,
}

impl TemporalCoherence {
    /// Compute the temporal coherence from a discrete power spectrum.
    ///
    /// Uses the Wiener-Khinchin theorem via direct quadrature:
    ///   Γ(τ) = Σ_k S(ω_k) exp(i ω_k τ) Δω / (2π)
    ///
    /// The delay grid covers \[-τ_max, τ_max\] with the same number of points N
    /// as the frequency grid.
    ///
    /// # Errors
    /// - `LengthMismatch` if `freqs.len() ≠ spectrum.len()`
    /// - `InvalidParameter` if N < 2 or Δω ≤ 0
    /// - `ZeroSpectrum` if the total spectral power is zero
    pub fn from_spectrum(freqs: &[f64], spectrum: &[f64]) -> Result<Self, TemporalCoherenceError> {
        let n = freqs.len();
        if spectrum.len() != n {
            return Err(TemporalCoherenceError::LengthMismatch {
                expected: n,
                got: spectrum.len(),
            });
        }
        if n < 2 {
            return Err(TemporalCoherenceError::InvalidParameter(
                "at least 2 frequency samples are required".into(),
            ));
        }

        let d_omega = (freqs[n - 1] - freqs[0]) / (n as f64 - 1.0);
        if d_omega <= 0.0 {
            return Err(TemporalCoherenceError::InvalidParameter(
                "frequency grid must be strictly increasing".into(),
            ));
        }

        let total_power: f64 = spectrum.iter().sum::<f64>() * d_omega;
        if total_power <= 0.0 {
            return Err(TemporalCoherenceError::ZeroSpectrum);
        }

        // Build delay grid τ ∈ [-τ_max, τ_max] with τ_max = π/Δω.
        let tau_max = PI / d_omega;
        let taus: Vec<f64> = (0..n)
            .map(|k| -tau_max + 2.0 * tau_max * k as f64 / (n as f64 - 1.0))
            .collect();

        // Shift to baseband before computing the DFT-style quadrature.
        //
        // The full Wiener-Khinchin result is:
        //   Γ(τ) = exp(i ω₀ τ) · Γ_bb(τ)
        // where Γ_bb is the baseband autocorrelation computed with shifted
        // frequencies (ω_k − ω₀).  At τ = 0 we recover Γ(0) = Γ_bb(0) > 0,
        // avoiding catastrophic numerical cancellation that occurs when the
        // absolute optical frequency (~3×10¹⁴ rad/s) drives the phase argument
        // to enormous values even for τ ≈ 0.
        let omega_center = freqs[n / 2];

        // Wiener-Khinchin via direct DFT-style quadrature (baseband).
        let mut gamma: Vec<Complex64> = Vec::with_capacity(n);
        for &tau in &taus {
            let mut acc = Complex64::new(0.0, 0.0);
            for k in 0..n {
                let phase = (freqs[k] - omega_center) * tau;
                acc += Complex64::new(0.0, phase).exp() * spectrum[k];
            }
            gamma.push(acc * d_omega / (2.0 * PI));
        }

        Ok(Self { taus, gamma })
    }

    /// Coherence time τ_c = ∫|γ(τ)|² dτ (trapezoidal rule).
    ///
    /// Uses the normalised degree of coherence γ(τ) = Γ(τ)/Γ(0).
    ///
    /// Returns `None` if Γ(0) ≈ 0.
    pub fn coherence_time(&self) -> Option<f64> {
        let n = self.taus.len();
        if n < 2 {
            return None;
        }
        let gamma0 = self.gamma[n / 2].norm(); // centre of zero-centred grid
        if gamma0 < f64::EPSILON {
            return None;
        }
        // Trapezoidal integration of |γ(τ)|².
        let mut integral = 0.0_f64;
        for i in 0..(n - 1) {
            let g1 = (self.gamma[i].norm() / gamma0).powi(2);
            let g2 = (self.gamma[i + 1].norm() / gamma0).powi(2);
            let d_tau = self.taus[i + 1] - self.taus[i];
            integral += 0.5 * (g1 + g2) * d_tau;
        }
        Some(integral)
    }

    /// Coherence length lc = c · τ_c \[m\].
    pub fn coherence_length(&self) -> Option<f64> {
        self.coherence_time().map(|tc| C * tc)
    }

    /// Degree of temporal coherence |γ(τ)| = |Γ(τ)| / |Γ(0)|.
    ///
    /// Returns 0 when Γ(0) ≈ 0.  Uses nearest-neighbour interpolation.
    pub fn degree_of_temporal_coherence(&self, tau: f64) -> f64 {
        if self.taus.is_empty() {
            return 0.0;
        }
        let n = self.taus.len();
        let gamma0_norm = self.gamma[n / 2].norm();
        if gamma0_norm < f64::EPSILON {
            return 0.0;
        }
        // Nearest-neighbour look-up.
        let idx = self
            .taus
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((*a - tau).abs())
                    .partial_cmp(&((*b - tau).abs()))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(n / 2);
        (self.gamma[idx].norm() / gamma0_norm).min(1.0)
    }

    /// Access the zero-delay value Γ(0) (total power up to normalisation).
    pub fn gamma_zero(&self) -> Complex64 {
        let n = self.taus.len();
        if n == 0 {
            return Complex64::new(0.0, 0.0);
        }
        self.gamma[n / 2]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PowerSpectralDensity
// ─────────────────────────────────────────────────────────────────────────────

/// Spectral shape model for computing analytical coherence properties.
#[derive(Debug, Clone)]
pub enum SpectralShape {
    /// Gaussian: S(ω) = exp(−(ω−ω₀)²/(2σ²))
    Gaussian { center_freq: f64, bandwidth: f64 },
    /// Lorentzian: S(ω) = (Δω/2)² / ((ω−ω₀)² + (Δω/2)²)
    Lorentzian { center_freq: f64, linewidth: f64 },
    /// Flat-top (rectangular): S(ω) = 1 for ω ∈ \[f_low, f_high\], else 0.
    FlatTop { f_low: f64, f_high: f64 },
}

/// Discretised power spectral density S(ω) sampled on a uniform frequency grid.
///
/// Can be constructed analytically from standard line-shape models.
#[derive(Debug, Clone)]
pub struct PowerSpectralDensity {
    /// Angular frequencies \[rad/s\].
    pub freqs: Vec<f64>,
    /// Spectral density values S(ωk) [normalised to unit peak].
    pub values: Vec<f64>,
    /// Underlying spectral shape (for analytical coherence time).
    pub shape: SpectralShape,
}

impl PowerSpectralDensity {
    /// Gaussian PSD centred at `center_freq` with 1/e half-bandwidth `bandwidth` \[rad/s\].
    pub fn gaussian(center_freq: f64, bandwidth: f64) -> Self {
        let n = 2048_usize;
        let omega_span = 8.0 * bandwidth;
        let freqs: Vec<f64> = (0..n)
            .map(|k| center_freq - omega_span / 2.0 + omega_span * k as f64 / (n as f64 - 1.0))
            .collect();
        let values: Vec<f64> = freqs
            .iter()
            .map(|&w| {
                let x = (w - center_freq) / bandwidth;
                (-x * x / 2.0).exp()
            })
            .collect();
        Self {
            freqs,
            values,
            shape: SpectralShape::Gaussian {
                center_freq,
                bandwidth,
            },
        }
    }

    /// Lorentzian PSD centred at `center_freq` with FWHM linewidth `linewidth` \[rad/s\].
    pub fn lorentzian(center_freq: f64, linewidth: f64) -> Self {
        let n = 2048_usize;
        let gamma = linewidth / 2.0;
        let omega_span = 40.0 * gamma;
        let freqs: Vec<f64> = (0..n)
            .map(|k| center_freq - omega_span / 2.0 + omega_span * k as f64 / (n as f64 - 1.0))
            .collect();
        let values: Vec<f64> = freqs
            .iter()
            .map(|&w| {
                let dw = w - center_freq;
                gamma * gamma / (dw * dw + gamma * gamma)
            })
            .collect();
        Self {
            freqs,
            values,
            shape: SpectralShape::Lorentzian {
                center_freq,
                linewidth,
            },
        }
    }

    /// Flat-top (rectangular) PSD over \[f_low, f_high\] \[rad/s\].
    pub fn flat_top(f_low: f64, f_high: f64) -> Self {
        let n = 2048_usize;
        let margin = (f_high - f_low) * 0.1;
        let freqs: Vec<f64> = (0..n)
            .map(|k| {
                (f_low - margin) + (f_high - f_low + 2.0 * margin) * k as f64 / (n as f64 - 1.0)
            })
            .collect();
        let values: Vec<f64> = freqs
            .iter()
            .map(|&w| if w >= f_low && w <= f_high { 1.0 } else { 0.0 })
            .collect();
        Self {
            freqs,
            values,
            shape: SpectralShape::FlatTop { f_low, f_high },
        }
    }

    /// Analytical coherence time for the stored spectral shape.
    ///
    /// - Gaussian:    τ_c = √(2π) / σ
    /// - Lorentzian:  τ_c = 1 / (π Δν_HWHM) = 2 / (Δω_FWHM)  (Lorentz FT gives exp decay)
    /// - Flat-top:    τ_c = 2π / (f_high − f_low) × sinc-squared integral → 2π/Δω
    pub fn coherence_time(&self) -> f64 {
        match &self.shape {
            SpectralShape::Gaussian { bandwidth, .. } => (2.0 * PI).sqrt() / bandwidth,
            SpectralShape::Lorentzian { linewidth, .. } => {
                // Γ(τ) = exp(−|τ| Δω/2) → τ_c = ∫|γ|² dτ = 2/(Δω)
                2.0 / linewidth
            }
            SpectralShape::FlatTop { f_low, f_high } => {
                // sinc²-autocorrelation: τ_c = 2π / Δω
                2.0 * PI / (f_high - f_low)
            }
        }
    }

    /// Coherence length lc = c · τ_c \[m\].
    pub fn coherence_length(&self) -> f64 {
        C * self.coherence_time()
    }

    /// Convert to a `TemporalCoherence` via numerical Wiener-Khinchin transform.
    pub fn to_temporal_coherence(&self) -> Result<TemporalCoherence, TemporalCoherenceError> {
        TemporalCoherence::from_spectrum(&self.freqs, &self.values)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MichelsonVisibility
// ─────────────────────────────────────────────────────────────────────────────

/// Fringe visibility in a Michelson interferometer.
///
/// V = (I_max − I_min) / (I_max + I_min)
///
/// For equal-amplitude beams V = |γ(τ)| where τ = path_diff / c.
pub struct MichelsonVisibility;

impl MichelsonVisibility {
    /// Compute the fringe visibility for a given path difference.
    ///
    /// # Parameters
    /// - `gamma`         — temporal coherence function.
    /// - `path_diff_m`   — optical path difference \[m\].
    /// - `wavelength`    — centre wavelength \[m\].
    ///
    /// Returns the visibility V ∈ \[0, 1\].
    pub fn compute(
        gamma: &TemporalCoherence,
        path_diff_m: f64,
        wavelength: f64,
    ) -> Result<f64, TemporalCoherenceError> {
        if wavelength <= 0.0 {
            return Err(TemporalCoherenceError::InvalidParameter(
                "wavelength must be positive".into(),
            ));
        }
        let tau = path_diff_m / C;
        Ok(gamma.degree_of_temporal_coherence(tau))
    }

    /// Compute visibility analytically from a `PowerSpectralDensity`.
    ///
    /// For a Gaussian spectrum: V(Δl) = exp(−(π Δl / lc)²/2).
    /// For Lorentzian:          V(Δl) = exp(−Δl / lc).
    /// For flat-top:            V(Δl) = |sinc(Δl / lc)|.
    pub fn compute_analytic(psd: &PowerSpectralDensity, path_diff_m: f64) -> f64 {
        let lc = psd.coherence_length();
        match &psd.shape {
            SpectralShape::Gaussian { .. } => {
                let x = path_diff_m / lc;
                (-PI * PI * x * x / 2.0).exp()
            }
            SpectralShape::Lorentzian { .. } => (-path_diff_m.abs() / lc).exp(),
            SpectralShape::FlatTop { f_low, f_high } => {
                let delta_omega = f_high - f_low;
                let tau = path_diff_m / C;
                let x = delta_omega * tau / 2.0;
                if x.abs() < f64::EPSILON {
                    1.0
                } else {
                    (x.sin() / x).abs()
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_gaussian_psd(center: f64, bw: f64) -> PowerSpectralDensity {
        PowerSpectralDensity::gaussian(center, bw)
    }

    #[test]
    fn gaussian_psd_coherence_time_analytic() {
        let bw = 1e12_f64; // 1 THz bandwidth
        let psd = make_gaussian_psd(3e14, bw);
        let tc_analytic = (2.0 * PI).sqrt() / bw;
        assert_abs_diff_eq!(psd.coherence_time(), tc_analytic, epsilon = 1e-20);
    }

    #[test]
    fn lorentzian_psd_coherence_time_analytic() {
        let linewidth = 1e9_f64; // 1 GHz
        let psd = PowerSpectralDensity::lorentzian(3e14, linewidth);
        let tc_analytic = 2.0 / linewidth;
        assert_abs_diff_eq!(psd.coherence_time(), tc_analytic, epsilon = 1e-20);
    }

    #[test]
    fn flat_top_psd_coherence_time_analytic() {
        let f_low = 2.9e14_f64;
        let f_high = 3.1e14_f64;
        let psd = PowerSpectralDensity::flat_top(f_low, f_high);
        let tc_analytic = 2.0 * PI / (f_high - f_low);
        assert_abs_diff_eq!(psd.coherence_time(), tc_analytic, epsilon = 1e-25);
    }

    #[test]
    fn temporal_coherence_from_spectrum_gamma0_is_positive() {
        let bw = 1e12_f64;
        let psd = make_gaussian_psd(3e14, bw);
        let tc = psd.to_temporal_coherence().expect("should succeed");
        assert!(tc.gamma_zero().re > 0.0, "Γ(0) must be positive");
    }

    #[test]
    fn degree_of_temporal_coherence_at_zero_is_unity() {
        let psd = make_gaussian_psd(3e14, 1e12);
        let tc = psd.to_temporal_coherence().expect("ok");
        let mu = tc.degree_of_temporal_coherence(0.0);
        assert_abs_diff_eq!(mu, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn michelson_visibility_zero_path_diff_is_unity() {
        let psd = make_gaussian_psd(3e14, 1e12);
        let tc = psd.to_temporal_coherence().expect("ok");
        let v = MichelsonVisibility::compute(&tc, 0.0, 633e-9).expect("ok");
        assert_abs_diff_eq!(v, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn michelson_visibility_analytic_gaussian_decays() {
        let psd = make_gaussian_psd(3e14, 1e12);
        let lc = psd.coherence_length();
        let v0 = MichelsonVisibility::compute_analytic(&psd, 0.0);
        let v1 = MichelsonVisibility::compute_analytic(&psd, lc);
        assert_abs_diff_eq!(v0, 1.0, epsilon = 1e-12);
        assert!(v1 < v0, "visibility must decay with path difference");
    }

    #[test]
    fn michelson_visibility_analytic_lorentzian_at_lc() {
        let linewidth = 1e9_f64;
        let psd = PowerSpectralDensity::lorentzian(3e14, linewidth);
        let lc = psd.coherence_length();
        let v = MichelsonVisibility::compute_analytic(&psd, lc);
        // For Lorentzian: V(lc) = exp(-1).
        assert_abs_diff_eq!(v, (-1.0_f64).exp(), epsilon = 1e-10);
    }

    #[test]
    fn coherence_length_is_positive() {
        let psd = make_gaussian_psd(3e14, 1e12);
        assert!(psd.coherence_length() > 0.0);
    }
}
