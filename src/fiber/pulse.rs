/// Optical pulse representations and characterisation tools.
///
/// Provides time-domain (`OpticalPulse`) and frequency-domain (`SpectralPulse`)
/// representations of optical pulses, along with constructors for common pulse
/// shapes (Gaussian, sech, chirped Gaussian) and physical characterisation
/// methods (FWHM, energy, instantaneous frequency, chirp).
///
/// Physical constants
use std::f64::consts::PI;

const C0: f64 = 2.997_924_58e8; // m/s — speed of light in vacuum
const HBAR: f64 = 1.054_571_8e-34; // J·s — reduced Planck constant

use crate::error::OxiPhotonError;
use num_complex::Complex64;

// ---------------------------------------------------------------------------
// Internal FFT (Cooley–Tukey radix-2) — kept private to this module
// ---------------------------------------------------------------------------

/// Cooley–Tukey radix-2 FFT / IFFT.
///
/// `x` must have a length that is a power of two; if it is not, the caller
/// should zero-pad before calling.  When `inverse` is `true` the result is
/// divided by `n` to produce the un-normalised IDFT.
pub(crate) fn fft_radix2(x: &[Complex64], inverse: bool) -> Vec<Complex64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return x.to_vec();
    }
    // Bit-reversal permutation
    let mut out = x.to_vec();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            out.swap(i, j);
        }
    }
    // Butterfly stages
    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let angle_step = sign * PI / half as f64;
        let wlen = Complex64::new(angle_step.cos(), angle_step.sin());
        let mut k = 0;
        while k < n {
            let mut w = Complex64::new(1.0, 0.0);
            for p in 0..half {
                let u = out[k + p];
                let v = out[k + p + half] * w;
                out[k + p] = u + v;
                out[k + p + half] = u - v;
                w *= wlen;
            }
            k += len;
        }
        len <<= 1;
    }
    if inverse {
        let inv_n = 1.0 / n as f64;
        for v in &mut out {
            *v *= inv_n;
        }
    }
    out
}

/// Zero-pad `x` to the next power-of-two length ≥ `x.len()`, then FFT.
pub(crate) fn fft_pow2(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    let m = n.next_power_of_two();
    let mut padded = x.to_vec();
    padded.resize(m, Complex64::new(0.0, 0.0));
    fft_radix2(&padded, false)
}

/// IFFT (assumes `x.len()` is already a power of two).
#[allow(dead_code)]
pub(crate) fn ifft_pow2(x: &[Complex64]) -> Vec<Complex64> {
    fft_radix2(x, true)
}

/// Angular-frequency axis (rad/s) for an `n`-point FFT with time step `dt`.
///
/// Returns frequencies in *fftshift* order (negative first, then positive)
/// which matches the convention used when the pulse is centred in the time
/// window.  For internal SSFM use we keep the *standard* (un-shifted) order.
pub(crate) fn omega_array_unshifted(n: usize, dt: f64) -> Vec<f64> {
    let df = 1.0 / (n as f64 * dt);
    (0..n)
        .map(|i| {
            let fi = if i < n / 2 {
                i as f64 * df
            } else {
                (i as f64 - n as f64) * df
            };
            2.0 * PI * fi
        })
        .collect()
}

// ---------------------------------------------------------------------------
// OpticalPulse
// ---------------------------------------------------------------------------

/// Optical pulse represented as a complex envelope A(t) in the retarded time
/// frame.  Power is |A(t)|² (W), energy is ∫|A|² dt (J).
#[derive(Debug, Clone)]
pub struct OpticalPulse {
    /// Time array (s) — uniformly spaced.
    pub t: Vec<f64>,
    /// Complex envelope A(t) (√W).
    pub amplitude: Vec<Complex64>,
    /// Centre wavelength λ₀ (nm).
    pub center_wavelength_nm: f64,
    /// Time step Δt (s).
    pub dt: f64,
}

impl OpticalPulse {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Create a pulse from pre-computed time and amplitude arrays.
    ///
    /// Returns `Err` if the lengths mismatch or `t` is empty.
    pub fn new(
        t: Vec<f64>,
        amplitude: Vec<Complex64>,
        lambda_nm: f64,
    ) -> Result<Self, OxiPhotonError> {
        if t.is_empty() {
            return Err(OxiPhotonError::NumericalError(
                "time array must not be empty".into(),
            ));
        }
        if t.len() != amplitude.len() {
            return Err(OxiPhotonError::NumericalError(format!(
                "time array length {} != amplitude length {}",
                t.len(),
                amplitude.len()
            )));
        }
        let dt = if t.len() > 1 { t[1] - t[0] } else { 1.0 };
        Ok(Self {
            t,
            amplitude,
            center_wavelength_nm: lambda_nm,
            dt,
        })
    }

    /// Uniformly-spaced time array centred at zero with total span
    /// `t_window_ps` (ps) and `n_pts` samples.
    fn make_time_array(n_pts: usize, t_window_ps: f64) -> Vec<f64> {
        let t_window_s = t_window_ps * 1.0e-12;
        let dt = t_window_s / n_pts as f64;
        let half = t_window_s / 2.0;
        (0..n_pts).map(|i| i as f64 * dt - half).collect()
    }

    /// Gaussian pulse:  A(t) = √P₀ · exp(−t²/(2T₀²))
    ///
    /// T₀ = FWHM_ps / (2 √(ln 2)) so that the intensity FWHM matches
    /// `fwhm_ps`.
    pub fn gaussian(
        n_pts: usize,
        t_window_ps: f64,
        peak_power_w: f64,
        fwhm_ps: f64,
        lambda_nm: f64,
    ) -> Self {
        let t = Self::make_time_array(n_pts, t_window_ps);
        let t0_s = fwhm_ps * 1.0e-12 / (2.0 * 2.0_f64.ln().sqrt());
        let a0 = peak_power_w.max(0.0).sqrt();
        let dt = if t.len() > 1 { t[1] - t[0] } else { 1.0 };
        let amplitude: Vec<Complex64> = t
            .iter()
            .map(|&ti| {
                let env = (-ti * ti / (2.0 * t0_s * t0_s)).exp();
                Complex64::new(a0 * env, 0.0)
            })
            .collect();
        Self {
            t,
            amplitude,
            center_wavelength_nm: lambda_nm,
            dt,
        }
    }

    /// Hyperbolic-secant (soliton) pulse:  A(t) = √P₀ · sech(t/T₀)
    ///
    /// T₀ = FWHM / (2 · ln(1+√2)) so that the intensity FWHM matches
    /// `fwhm_ps`.
    pub fn sech(
        n_pts: usize,
        t_window_ps: f64,
        peak_power_w: f64,
        fwhm_ps: f64,
        lambda_nm: f64,
    ) -> Self {
        let t = Self::make_time_array(n_pts, t_window_ps);
        let ln_factor = 2.0 * (1.0 + 2.0_f64.sqrt()).ln();
        let t0_s = fwhm_ps * 1.0e-12 / ln_factor;
        let a0 = peak_power_w.max(0.0).sqrt();
        let dt = if t.len() > 1 { t[1] - t[0] } else { 1.0 };
        let amplitude: Vec<Complex64> = t
            .iter()
            .map(|&ti| {
                let env = if t0_s > 0.0 {
                    1.0 / (ti / t0_s).cosh()
                } else {
                    0.0
                };
                Complex64::new(a0 * env, 0.0)
            })
            .collect();
        Self {
            t,
            amplitude,
            center_wavelength_nm: lambda_nm,
            dt,
        }
    }

    /// Chirped Gaussian pulse:
    ///   A(t) = √P₀ · exp(−(1+iC)·t²/(2T₀²))
    ///
    /// where `chirp_c` is the dimensionless chirp parameter C.  Positive C
    /// gives a down-chirped (red-leading) pulse.
    pub fn chirped_gaussian(
        n_pts: usize,
        t_window_ps: f64,
        peak_power_w: f64,
        fwhm_ps: f64,
        chirp_c: f64,
        lambda_nm: f64,
    ) -> Self {
        let t = Self::make_time_array(n_pts, t_window_ps);
        let t0_s = fwhm_ps * 1.0e-12 / (2.0 * 2.0_f64.ln().sqrt());
        let a0 = peak_power_w.max(0.0).sqrt();
        let dt = if t.len() > 1 { t[1] - t[0] } else { 1.0 };
        let amplitude: Vec<Complex64> = t
            .iter()
            .map(|&ti| {
                let arg = Complex64::new(1.0 + chirp_c * chirp_c, -chirp_c);
                // A = a0 * exp(-(1+iC)*t²/(2T0²))
                // = a0 * exp(-t²(1+iC)/(2T0²))
                // real part of exponent: -(1+C²)t²/(2T0²) * (1/(1+C²)) ... simplified:
                // Actually: A = a0 * exp(-(1+iC)*t²/(2T0²))
                let exponent =
                    -(1.0 + Complex64::new(0.0, chirp_c)) * (ti * ti / (2.0 * t0_s * t0_s));
                let _ = arg; // suppress unused warning
                Complex64::new(a0, 0.0) * exponent.exp()
            })
            .collect();
        Self {
            t,
            amplitude,
            center_wavelength_nm: lambda_nm,
            dt,
        }
    }

    // -----------------------------------------------------------------------
    // Pulse characterisation
    // -----------------------------------------------------------------------

    /// Peak power max|A(t)|² (W).
    pub fn peak_power(&self) -> f64 {
        self.amplitude
            .iter()
            .map(|a| a.norm_sqr())
            .fold(0.0_f64, f64::max)
    }

    /// Pulse energy E = ∫|A|² dt (J).
    pub fn energy_j(&self) -> f64 {
        self.amplitude.iter().map(|a| a.norm_sqr()).sum::<f64>() * self.dt
    }

    /// RMS pulse width σ_t (s): σ_t² = ⟨(t−⟨t⟩)²⟩ weighted by |A|².
    pub fn rms_width_s(&self) -> f64 {
        let power: Vec<f64> = self.amplitude.iter().map(|a| a.norm_sqr()).collect();
        let total: f64 = power.iter().sum();
        if total < 1.0e-60 {
            return 0.0;
        }
        let mean: f64 = power
            .iter()
            .zip(self.t.iter())
            .map(|(&p, &ti)| ti * p)
            .sum::<f64>()
            / total;
        let var: f64 = power
            .iter()
            .zip(self.t.iter())
            .map(|(&p, &ti)| {
                let d = ti - mean;
                d * d * p
            })
            .sum::<f64>()
            / total;
        var.sqrt()
    }

    /// FWHM of the intensity profile |A(t)|² (ps).
    ///
    /// Locates the half-maximum points by linear interpolation.  Returns 0 if
    /// the pulse is too weak.
    pub fn fwhm_ps(&self) -> f64 {
        let power: Vec<f64> = self.amplitude.iter().map(|a| a.norm_sqr()).collect();
        let peak = power.iter().cloned().fold(0.0_f64, f64::max);
        if peak < 1.0e-60 {
            return 0.0;
        }
        let half_max = peak / 2.0;
        // Find first crossing from below
        let left_idx = power
            .windows(2)
            .enumerate()
            .find(|(_, w)| w[0] <= half_max && w[1] >= half_max)
            .map(|(i, w)| {
                let frac = (half_max - w[0]) / (w[1] - w[0]).max(1.0e-60);
                self.t[i] + frac * self.dt
            });
        // Find last crossing from above
        let right_idx = power
            .windows(2)
            .enumerate()
            .rev()
            .find(|(_, w)| w[0] >= half_max && w[1] <= half_max)
            .map(|(i, w)| {
                let frac = (w[0] - half_max) / (w[0] - w[1]).max(1.0e-60);
                self.t[i] + frac * self.dt
            });
        match (left_idx, right_idx) {
            (Some(l), Some(r)) if r > l => (r - l) * 1.0e12, // convert s → ps
            _ => 0.0,
        }
    }

    /// Centre of mass ⟨t⟩ (ps), weighted by |A(t)|².
    pub fn center_of_mass_ps(&self) -> f64 {
        let power: Vec<f64> = self.amplitude.iter().map(|a| a.norm_sqr()).collect();
        let total: f64 = power.iter().sum();
        if total < 1.0e-60 {
            return 0.0;
        }
        let mean: f64 = power
            .iter()
            .zip(self.t.iter())
            .map(|(&p, &ti)| ti * p)
            .sum::<f64>()
            / total;
        mean * 1.0e12 // s → ps
    }

    /// Instantaneous frequency ν(t) = ν₀ − (1/2π) dφ/dt (Hz).
    ///
    /// The carrier contribution ν₀ is subtracted so the result is the *offset*
    /// frequency from the centre frequency.  Phase is unwrapped numerically
    /// via finite differences.
    pub fn instantaneous_frequency(&self) -> Vec<f64> {
        let n = self.amplitude.len();
        if n < 2 {
            return vec![0.0; n];
        }
        let phase: Vec<f64> = self.amplitude.iter().map(|a| a.arg()).collect();
        // Unwrap phase
        let mut unwrapped = phase.clone();
        for i in 1..n {
            let diff = unwrapped[i] - unwrapped[i - 1];
            let correction = -((diff + PI) / (2.0 * PI)).floor() * 2.0 * PI;
            unwrapped[i] += correction;
        }
        // Numerical derivative dφ/dt, central differences
        let mut freq = vec![0.0; n];
        for i in 1..(n - 1) {
            freq[i] = (unwrapped[i + 1] - unwrapped[i - 1]) / (2.0 * self.dt * 2.0 * PI);
        }
        // Forward / backward difference at boundaries
        freq[0] = (unwrapped[1] - unwrapped[0]) / (self.dt * 2.0 * PI);
        freq[n - 1] = (unwrapped[n - 1] - unwrapped[n - 2]) / (self.dt * 2.0 * PI);
        freq
    }

    /// Local chirp C(t) = −T₀² d²φ/dt² (dimensionless, normalised by T₀²
    /// = FWHM² / (8 ln 2) for a Gaussian).
    ///
    /// Positive C at t<0 means frequency increases with time (up-chirp) for
    /// the standard convention.
    pub fn chirp(&self) -> Vec<f64> {
        let n = self.amplitude.len();
        let phase: Vec<f64> = self.amplitude.iter().map(|a| a.arg()).collect();
        // Unwrap
        let mut unwrapped = phase;
        for i in 1..n {
            let diff = unwrapped[i] - unwrapped[i - 1];
            let correction = -((diff + PI) / (2.0 * PI)).floor() * 2.0 * PI;
            unwrapped[i] += correction;
        }
        // Second derivative d²φ/dt²
        let mut d2phi = vec![0.0; n];
        for i in 1..(n - 1) {
            d2phi[i] =
                (unwrapped[i + 1] - 2.0 * unwrapped[i] + unwrapped[i - 1]) / (self.dt * self.dt);
        }
        d2phi[0] = d2phi[1];
        d2phi[n - 1] = d2phi[n - 2];
        // Normalise by T0²: use RMS width as T0 proxy
        let t0_sq = {
            let w = self.rms_width_s();
            if w < 1.0e-30 {
                1.0
            } else {
                w * w
            }
        };
        d2phi.iter().map(|&d| -t0_sq * d).collect()
    }

    /// Intensity profile |A(t)|² (W).
    pub fn power(&self) -> Vec<f64> {
        self.amplitude.iter().map(|a| a.norm_sqr()).collect()
    }

    /// Apply phase modulation in-place: A(t) → A(t) · exp(i·φ(t)).
    ///
    /// `phi` must have the same length as the pulse.
    pub fn apply_phase(&mut self, phi: &[f64]) -> Result<(), OxiPhotonError> {
        if phi.len() != self.amplitude.len() {
            return Err(OxiPhotonError::NumericalError(format!(
                "phase array length {} != pulse length {}",
                phi.len(),
                self.amplitude.len()
            )));
        }
        for (a, &p) in self.amplitude.iter_mut().zip(phi.iter()) {
            let phase_factor = Complex64::new(0.0, p).exp();
            *a *= phase_factor;
        }
        Ok(())
    }

    /// Angular centre frequency ω₀ = 2πc/λ₀ (rad/s).
    pub fn omega0_rad_s(&self) -> f64 {
        2.0 * PI * C0 / (self.center_wavelength_nm * 1.0e-9)
    }

    /// Centre frequency ν₀ = c/λ₀ (Hz).
    pub fn nu0_hz(&self) -> f64 {
        C0 / (self.center_wavelength_nm * 1.0e-9)
    }

    /// Number of photons per pulse: E / (ħ·ω₀).
    pub fn photon_count(&self) -> f64 {
        let omega0 = self.omega0_rad_s();
        if omega0 < 1.0e-10 || HBAR < 1.0e-60 {
            return 0.0;
        }
        self.energy_j() / (HBAR * omega0)
    }
}

// ---------------------------------------------------------------------------
// SpectralPulse
// ---------------------------------------------------------------------------

/// Optical pulse in the frequency domain: Ã(ω) = FFT[A(t)].
///
/// Frequencies are angular offsets from the centre frequency (rad/s).
#[derive(Debug, Clone)]
pub struct SpectralPulse {
    /// Angular frequency offsets ω − ω₀ (rad/s) in standard (un-shifted) FFT order.
    pub omega: Vec<f64>,
    /// Complex spectrum Ã(ω) (√(W·s)).
    pub spectrum: Vec<Complex64>,
    /// Centre wavelength λ₀ (nm).
    pub center_wavelength_nm: f64,
    /// Time step used to compute the FFT (retained for future bandwidth calculations).
    #[allow(dead_code)]
    dt: f64,
    /// Number of original time samples (before power-of-two padding).
    n_orig: usize,
}

impl SpectralPulse {
    /// Compute the spectrum by FFT of a time-domain pulse.
    pub fn from_pulse(pulse: &OpticalPulse) -> Self {
        let n_orig = pulse.amplitude.len();
        let spectrum = fft_pow2(&pulse.amplitude);
        let n = spectrum.len();
        let omega = omega_array_unshifted(n, pulse.dt);
        Self {
            omega,
            spectrum,
            center_wavelength_nm: pulse.center_wavelength_nm,
            dt: pulse.dt,
            n_orig,
        }
    }

    /// Power spectral density |Ã(ω)|² · Δt² (W·s²) — proportional to the
    /// energy spectral density.
    pub fn power_spectrum(&self) -> Vec<f64> {
        self.spectrum.iter().map(|s| s.norm_sqr()).collect()
    }

    /// Wavelength array (nm) corresponding to each spectral bin.
    ///
    /// Uses λ = 2πc / (ω₀ + ωₘ) where ω₀ is the centre angular frequency.
    pub fn wavelength_array_nm(&self) -> Vec<f64> {
        let omega0 = 2.0 * PI * C0 / (self.center_wavelength_nm * 1.0e-9);
        self.omega
            .iter()
            .map(|&dw| {
                let omega_total = omega0 + dw;
                if omega_total.abs() < 1.0e-10 {
                    f64::INFINITY
                } else {
                    2.0 * PI * C0 / omega_total * 1.0e9
                }
            })
            .collect()
    }

    /// RMS spectral bandwidth σ_ω (rad/s).
    pub fn rms_bandwidth_rad_s(&self) -> f64 {
        let psd = self.power_spectrum();
        let total: f64 = psd.iter().sum();
        if total < 1.0e-60 {
            return 0.0;
        }
        let mean_omega: f64 = psd
            .iter()
            .zip(self.omega.iter())
            .map(|(&p, &w)| w * p)
            .sum::<f64>()
            / total;
        let var: f64 = psd
            .iter()
            .zip(self.omega.iter())
            .map(|(&p, &w)| {
                let d = w - mean_omega;
                d * d * p
            })
            .sum::<f64>()
            / total;
        var.sqrt()
    }

    /// RMS bandwidth in nm (converted from rad/s via λ = 2πc/ω).
    pub fn bandwidth_nm(&self) -> f64 {
        let sigma_omega = self.rms_bandwidth_rad_s();
        let lambda0_m = self.center_wavelength_nm * 1.0e-9;
        // Δλ ≈ (λ₀²/(2πc)) · Δω
        sigma_omega * lambda0_m * lambda0_m / (2.0 * PI * C0) * 1.0e9
    }

    /// Time-bandwidth product σ_t · σ_ω (dimensionless, RMS definition).
    ///
    /// For a transform-limited Gaussian pulse this equals 0.5.
    pub fn time_bandwidth_product(&self, pulse: &OpticalPulse) -> f64 {
        let sigma_t = pulse.rms_width_s();
        let sigma_omega = self.rms_bandwidth_rad_s();
        sigma_t * sigma_omega
    }

    /// FWHM bandwidth in nm (from intensity spectrum).
    pub fn fwhm_bandwidth_nm(&self) -> f64 {
        let psd = self.power_spectrum();
        let peak = psd.iter().cloned().fold(0.0_f64, f64::max);
        if peak < 1.0e-60 {
            return 0.0;
        }
        let half_max = peak / 2.0;
        let lambda = self.wavelength_array_nm();
        // filter out non-finite wavelengths and sort
        let mut pairs: Vec<(f64, f64)> = psd
            .iter()
            .zip(lambda.iter())
            .filter(|(_, &lam)| lam.is_finite() && lam > 0.0)
            .map(|(&p, &lam)| (lam, p))
            .collect();
        if pairs.len() < 2 {
            return 0.0;
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let left = pairs
            .windows(2)
            .find(|w| w[0].1 <= half_max && w[1].1 >= half_max)
            .map(|w| {
                let frac = (half_max - w[0].1) / (w[1].1 - w[0].1).max(1.0e-60);
                w[0].0 + frac * (w[1].0 - w[0].0)
            });
        let right = pairs
            .windows(2)
            .rev()
            .find(|w| w[0].1 >= half_max && w[1].1 <= half_max)
            .map(|w| {
                let frac = (w[0].1 - half_max) / (w[0].1 - w[1].1).max(1.0e-60);
                w[0].0 + frac * (w[1].0 - w[0].0)
            });
        match (left, right) {
            (Some(l), Some(r)) if r > l => r - l,
            _ => 0.0,
        }
    }

    /// Number of spectral samples (including zero-padding).
    pub fn len(&self) -> usize {
        self.spectrum.len()
    }

    /// `true` if there are no spectral samples.
    pub fn is_empty(&self) -> bool {
        self.spectrum.is_empty()
    }

    /// Original (un-padded) time-domain sample count.
    pub fn n_orig(&self) -> usize {
        self.n_orig
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Helper: max absolute difference between slices.
    fn max_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn test_gaussian_pulse_fwhm() {
        // Create a 1 ps FWHM Gaussian and verify fwhm_ps() ≈ 1
        let pulse = OpticalPulse::gaussian(4096, 50.0, 1.0, 1.0, 1550.0);
        let fwhm = pulse.fwhm_ps();
        assert!(
            (fwhm - 1.0).abs() < 0.05,
            "Gaussian FWHM expected ~1 ps, got {fwhm:.4} ps"
        );
    }

    #[test]
    fn test_sech_pulse_energy() {
        // Sech: E = ∫P₀·sech²(t/T₀) dt = 2·P₀·T₀
        let fwhm_ps = 1.0_f64;
        let p0 = 1.0_f64; // W
        let ln_fac = 2.0 * (1.0 + 2.0_f64.sqrt()).ln();
        let t0_s = fwhm_ps * 1.0e-12 / ln_fac;
        let expected_energy = 2.0 * p0 * t0_s;
        let pulse = OpticalPulse::sech(4096, 50.0, p0, fwhm_ps, 1550.0);
        let energy = pulse.energy_j();
        let rel_err = (energy - expected_energy).abs() / expected_energy;
        assert!(
            rel_err < 0.02,
            "Sech energy: expected {expected_energy:.3e} J, got {energy:.3e} J (rel_err={rel_err:.4})"
        );
    }

    #[test]
    fn test_pulse_peak_power() {
        let p0 = 42.5_f64;
        let pulse = OpticalPulse::gaussian(2048, 50.0, p0, 2.0, 1550.0);
        let peak = pulse.peak_power();
        // Peak should be ≈ P₀ (within numerical grid accuracy)
        assert!(
            (peak - p0).abs() / p0 < 0.01,
            "Peak power: expected {p0}, got {peak}"
        );
    }

    #[test]
    fn test_chirped_gaussian_tbp() {
        // A chirped Gaussian has a larger time-bandwidth product than
        // a transform-limited one (TBP ≥ 0.5 for RMS σ_t·σ_ω).
        let unchirped = OpticalPulse::gaussian(2048, 80.0, 1.0, 2.0, 1550.0);
        let chirped = OpticalPulse::chirped_gaussian(2048, 80.0, 1.0, 2.0, 5.0, 1550.0);
        let sp_unchirped = SpectralPulse::from_pulse(&unchirped);
        let sp_chirped = SpectralPulse::from_pulse(&chirped);
        let tbp_unchirped = sp_unchirped.time_bandwidth_product(&unchirped);
        let tbp_chirped = sp_chirped.time_bandwidth_product(&chirped);
        assert!(
            tbp_chirped > tbp_unchirped,
            "Chirped TBP ({tbp_chirped:.4}) should exceed unchirped TBP ({tbp_unchirped:.4})"
        );
    }

    #[test]
    fn test_fft_roundtrip() {
        let n = 64_usize;
        let x: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.3).sin(), (i as f64 * 0.7).cos()))
            .collect();
        let spectrum = fft_radix2(&x, false);
        let recovered = fft_radix2(&spectrum, true);
        for (orig, rec) in x.iter().zip(recovered.iter()) {
            let err = (orig - rec).norm();
            assert!(err < 1.0e-10, "FFT roundtrip error: {err:.2e}");
        }
    }

    #[test]
    fn test_energy_conservation_fft() {
        // Parseval's theorem: Σ|x|² = (1/N) Σ|X|²
        let n = 128_usize;
        let x: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect();
        let energy_time: f64 = x.iter().map(|a| a.norm_sqr()).sum();
        let spectrum = fft_radix2(&x, false);
        let energy_freq: f64 = spectrum.iter().map(|s| s.norm_sqr()).sum::<f64>() / n as f64;
        assert_relative_eq!(energy_time, energy_freq, max_relative = 1.0e-10);
    }

    #[test]
    fn test_pulse_apply_phase() {
        let mut pulse = OpticalPulse::gaussian(256, 20.0, 1.0, 1.0, 1550.0);
        let original_power: Vec<f64> = pulse.power();
        let phi: Vec<f64> = (0..256).map(|i| i as f64 * 0.01).collect();
        pulse.apply_phase(&phi).expect("apply_phase failed");
        let new_power: Vec<f64> = pulse.power();
        // Phase modulation must not change the intensity profile.
        let diff = max_diff(&original_power, &new_power);
        assert!(
            diff < 1.0e-12,
            "Phase modulation changed power by {diff:.2e}"
        );
    }

    #[test]
    fn test_spectral_pulse_bandwidth_positive() {
        let pulse = OpticalPulse::gaussian(1024, 50.0, 1.0, 1.0, 1550.0);
        let sp = SpectralPulse::from_pulse(&pulse);
        let bw = sp.bandwidth_nm();
        assert!(bw > 0.0, "Spectral bandwidth must be positive, got {bw}");
    }

    #[test]
    fn test_center_of_mass_near_zero() {
        // A symmetric pulse centred at t=0 should have CoM ≈ 0.
        let pulse = OpticalPulse::gaussian(2048, 50.0, 1.0, 2.0, 1550.0);
        let com = pulse.center_of_mass_ps();
        assert!(com.abs() < 0.01, "CoM should be near 0, got {com:.4} ps");
    }

    #[test]
    fn test_photon_count_positive() {
        let pulse = OpticalPulse::gaussian(512, 20.0, 1.0, 1.0, 1550.0);
        let n_photons = pulse.photon_count();
        assert!(n_photons > 0.0, "Photon count must be positive");
    }
}
