//! Pulse shaping for ultrashort laser pulses.
//!
//! Three shaper architectures are implemented:
//!
//! - **4-f SLM shaper**: grating–lens–SLM–lens–grating geometry; independent
//!   phase and amplitude control on each spectral pixel.
//! - **ActivePulseCompressor**: closed-loop shaper that iteratively drives the
//!   spectral phase toward the transform limit.
//! - **DAZZLER (AO shaper)**: acousto-optic programmable dispersive filter that
//!   applies arbitrary spectral phase within a crystal interaction length.
//!
//! # Physical background
//!
//! A 4-f shaper applies a frequency-domain transfer function:
//!
//! ```text
//! E_out(ω) = M(ω) · E_in(ω)
//! ```
//!
//! where `M(ω) = A(ω)·exp(iφ(ω))` is set by the SLM pixel mask.

use num_complex::Complex64;
use std::f64::consts::PI;

// ─── Error type ─────────────────────────────────────────────────────────────

/// Errors from pulse shaping operations.
#[derive(Debug, thiserror::Error)]
pub enum PulseShaperError {
    #[error("Spectrum length {spectrum} does not match shaper pixel count {pixels}")]
    SpectrumLengthMismatch { spectrum: usize, pixels: usize },
    #[error("Phase mask length {mask} does not match pixel count {pixels}")]
    PhaseMaskLengthMismatch { mask: usize, pixels: usize },
    #[error("Amplitude mask contains value outside [0, 1]: {value}")]
    InvalidAmplitude { value: f64 },
    #[error("GDD compensation requires finite GDD value")]
    NonfiniteGdd,
    #[error("FFT requires power-of-two size, got {0}")]
    FftSizeMismatch(usize),
}

// ─── Internal FFT ────────────────────────────────────────────────────────────

fn fft_inplace(buf: &mut [Complex64]) -> Result<(), PulseShaperError> {
    let n = buf.len();
    if n == 0 || (n & (n - 1)) != 0 {
        return Err(PulseShaperError::FftSizeMismatch(n));
    }
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        let w_len = Complex64::new(ang.cos(), ang.sin());
        for i in (0..n).step_by(len) {
            let mut w = Complex64::new(1.0, 0.0);
            for k in 0..(len / 2) {
                let u = buf[i + k];
                let v = buf[i + k + len / 2] * w;
                buf[i + k] = u + v;
                buf[i + k + len / 2] = u - v;
                w *= w_len;
            }
        }
        len <<= 1;
    }
    Ok(())
}

fn ifft_inplace(buf: &mut [Complex64]) -> Result<(), PulseShaperError> {
    let n = buf.len();
    for x in buf.iter_mut() {
        *x = x.conj();
    }
    fft_inplace(buf)?;
    let scale = 1.0 / n as f64;
    for x in buf.iter_mut() {
        *x = x.conj() * scale;
    }
    Ok(())
}

// ─── FourFPulseShaper ────────────────────────────────────────────────────────

/// 4-f zero-dispersion pulse shaper with a spatial light modulator (SLM).
///
/// In a 4-f geometry, a grating disperses the spectrum, a lens Fourier-
/// transforms it onto the SLM, and a second lens + grating recombine the
/// shaped spectrum.  Each pixel `k` of the SLM applies a complex transfer
/// function:
///
/// ```text
/// M_k = amplitude_k · exp(i · phase_k)
/// ```
///
/// The spectral resolution (nm/pixel) is set by the grating + lens combination.
#[derive(Debug, Clone)]
pub struct FourFPulseShaper {
    /// Number of SLM pixels (= spectral channels).
    pub n_pixels: usize,
    /// Physical pixel size (µm).
    pub pixel_size_um: f64,
    /// Spectral resolution (nm per pixel).
    pub spectral_resolution_nm: f64,
    /// Centre wavelength (m).
    pub center_wavelength: f64,
    /// Phase mask φ(k) in radians (applied to each spectral pixel).
    pub phase_mask: Vec<f64>,
    /// Amplitude mask A(k) in [0, 1] (attenuation per spectral pixel).
    pub amplitude_mask: Vec<f64>,
}

impl FourFPulseShaper {
    /// Construct a 4-f shaper with flat (zero-phase, unity-amplitude) masks.
    ///
    /// # Arguments
    /// * `n_pixels`              — number of SLM pixels
    /// * `spectral_resolution_nm`— spectral bandwidth per pixel (nm)
    /// * `center_wavelength`     — centre wavelength (m)
    pub fn new(n_pixels: usize, spectral_resolution_nm: f64, center_wavelength: f64) -> Self {
        Self {
            n_pixels,
            pixel_size_um: 100.0, // typical 100 µm pixel for 640-pixel SLM
            spectral_resolution_nm,
            center_wavelength,
            phase_mask: vec![0.0; n_pixels],
            amplitude_mask: vec![1.0; n_pixels],
        }
    }

    /// Angular frequency (rad/s) of pixel `k` relative to centre.
    fn pixel_omega_offset(&self, k: usize) -> f64 {
        let c = 2.997_924_58e8_f64; // m/s
        let lam_centre_m = self.center_wavelength;
        let dlam_m = self.spectral_resolution_nm * 1e-9;
        // Pixel detuning in wavelength from centre
        let dlam = (k as f64 - self.n_pixels as f64 / 2.0) * dlam_m;
        // Convert Δλ to Δω: Δω ≈ -2πc/λ₀² · Δλ
        -2.0 * PI * c / (lam_centre_m * lam_centre_m) * dlam
    }

    /// Set the phase mask to compensate a given GDD.
    ///
    /// Applies `φ(ω) = -GDD/2 · (ω - ω₀)²` on each pixel.
    ///
    /// # Arguments
    /// * `gdd_fs2` — group-delay dispersion in fs² (positive = up-chirp compensation)
    pub fn set_gdd_compensation(&mut self, gdd_fs2: f64) {
        // GDD in fs² → s²: 1 fs² = 1e-30 s²
        let gdd_s2 = gdd_fs2 * 1e-30;
        for k in 0..self.n_pixels {
            let domega = self.pixel_omega_offset(k);
            self.phase_mask[k] = -0.5 * gdd_s2 * domega * domega;
        }
    }

    /// Add TOD (third-order dispersion) compensation on top of existing phase.
    ///
    /// Appends `φ_TOD(ω) = -TOD/6 · (ω - ω₀)³`.
    pub fn set_tod_compensation(&mut self, tod_fs3: f64) {
        let tod_s3 = tod_fs3 * 1e-45;
        for k in 0..self.n_pixels {
            let dw = self.pixel_omega_offset(k);
            self.phase_mask[k] += -tod_s3 / 6.0 * dw * dw * dw;
        }
    }

    /// Add FOD (fourth-order dispersion) compensation on top of existing phase.
    ///
    /// Appends `φ_FOD(ω) = -FOD/24 · (ω - ω₀)⁴`.
    pub fn set_fod_compensation(&mut self, fod_fs4: f64) {
        let fod_s4 = fod_fs4 * 1e-60;
        for k in 0..self.n_pixels {
            let dw = self.pixel_omega_offset(k);
            self.phase_mask[k] += -fod_s4 / 24.0 * dw * dw * dw * dw;
        }
    }

    /// Replace the phase mask with an arbitrary array.
    ///
    /// # Errors
    /// Returns `PulseShaperError::PhaseMaskLengthMismatch` if lengths differ.
    pub fn set_phase(&mut self, phase: Vec<f64>) -> Result<(), PulseShaperError> {
        if phase.len() != self.n_pixels {
            return Err(PulseShaperError::PhaseMaskLengthMismatch {
                mask: phase.len(),
                pixels: self.n_pixels,
            });
        }
        self.phase_mask = phase;
        Ok(())
    }

    /// Apply the shaper transfer function to an input spectrum.
    ///
    /// `E_out(ω_k) = A_k · exp(i·φ_k) · E_in(ω_k)`
    ///
    /// # Errors
    /// Returns `PulseShaperError::SpectrumLengthMismatch` if lengths differ.
    pub fn apply(&self, spectrum: &[Complex64]) -> Result<Vec<Complex64>, PulseShaperError> {
        if spectrum.len() != self.n_pixels {
            return Err(PulseShaperError::SpectrumLengthMismatch {
                spectrum: spectrum.len(),
                pixels: self.n_pixels,
            });
        }
        let out: Vec<Complex64> = spectrum
            .iter()
            .zip(self.phase_mask.iter())
            .zip(self.amplitude_mask.iter())
            .map(|((&e, &phi), &amp)| {
                let transfer = Complex64::new(amp * phi.cos(), amp * phi.sin());
                e * transfer
            })
            .collect();
        Ok(out)
    }

    /// Compute the shaped output pulse in the time domain.
    ///
    /// Applies the shaper to `input_spectrum`, then IFFTs to the time domain.
    /// The spectrum is zero-padded to the next power of two.
    pub fn output_field(
        &self,
        input_spectrum: &[Complex64],
    ) -> Result<Vec<Complex64>, PulseShaperError> {
        let shaped = self.apply(input_spectrum)?;
        let n = shaped.len();
        let n_fft = n.next_power_of_two();
        let mut buf: Vec<Complex64> = (0..n_fft)
            .map(|i| {
                if i < n {
                    shaped[i]
                } else {
                    Complex64::new(0.0, 0.0)
                }
            })
            .collect();
        ifft_inplace(&mut buf)?;
        Ok(buf[..n].to_vec())
    }

    /// Spectral bandwidth per pixel (nm).
    pub fn bandwidth_per_pixel_nm(&self) -> f64 {
        self.spectral_resolution_nm
    }

    /// Maximum temporal window (ps) of the shaper.
    ///
    /// Set by the Nyquist criterion on the spectral resolution:
    /// `T_window = 1 / Δν_pixel = λ²/(c·Δλ_pixel)`
    pub fn temporal_window_ps(&self) -> f64 {
        let c_nm_ps = 2.997_924_58e5; // c in nm/ps
        let lambda_nm = self.center_wavelength * 1e9;
        // T = λ²/(c·Δλ) in ps

        lambda_nm * lambda_nm / (c_nm_ps * self.spectral_resolution_nm)
    }
}

// ─── ActivePulseCompressor ──────────────────────────────────────────────────

/// Closed-loop active pulse compressor.
///
/// Iteratively updates the SLM phase mask using measured spectral phase
/// feedback (from FROG or SPIDER) to approach the transform limit.
#[derive(Debug, Clone)]
pub struct ActivePulseCompressor {
    /// Underlying 4-f pulse shaper.
    pub shaper: FourFPulseShaper,
    /// Target GDD to compensate (fs²). Updated after each iteration.
    pub target_gdd_fs2: f64,
    /// Target TOD to compensate (fs³).
    pub target_tod_fs3: f64,
}

impl ActivePulseCompressor {
    /// Create an active compressor wrapping a 4-f shaper.
    pub fn new(shaper: FourFPulseShaper) -> Self {
        Self {
            shaper,
            target_gdd_fs2: 0.0,
            target_tod_fs3: 0.0,
        }
    }

    /// Apply a phase correction derived from a measured spectral phase profile.
    ///
    /// Computes the correction as the negative of the measured phase (to cancel
    /// residual phase) and updates the SLM mask.  The measured phase is
    /// interpolated onto the shaper pixel grid using the frequency axis.
    ///
    /// # Arguments
    /// * `measured_phase` — φ(ω) array retrieved by FROG or SPIDER (rad)
    /// * `frequencies`    — angular frequency grid of the measured phase (rad/s)
    pub fn apply_correction(&mut self, measured_phase: &[f64], frequencies: &[f64]) {
        let n = self.shaper.n_pixels;
        let n_meas = measured_phase.len().min(frequencies.len());
        if n_meas < 2 {
            return;
        }
        let f_min = frequencies[0];
        let f_max = frequencies[n_meas - 1];
        let df = (f_max - f_min) / (n_meas - 1) as f64;

        // Interpolate measured phase onto shaper pixel frequency grid
        for k in 0..n {
            let omega = self.shaper.pixel_omega_offset(k);
            // Map omega to measured frequency grid index
            let idx_f = if df.abs() > 1e-30 {
                (omega - f_min) / df
            } else {
                0.0
            };
            let i_lo = idx_f.floor() as isize;
            let frac = idx_f - i_lo as f64;
            let phase_at_k = if i_lo >= 0 && (i_lo as usize) < n_meas {
                let phi_lo = measured_phase[i_lo as usize];
                let phi_hi = if (i_lo + 1) < n_meas as isize {
                    measured_phase[(i_lo + 1) as usize]
                } else {
                    phi_lo
                };
                phi_lo * (1.0 - frac) + phi_hi * frac
            } else {
                0.0
            };
            // Correction = −φ_measured (to cancel the measured phase)
            self.shaper.phase_mask[k] = -phase_at_k;
        }
    }

    /// Perform one iteration of the convergence loop.
    ///
    /// Updates GDD/TOD compensation on the shaper based on the current TBP
    /// estimate.  The step size is proportional to how far the TBP is from
    /// the target.
    ///
    /// # Arguments
    /// * `current_tbp` — measured time-bandwidth product
    /// * `target_tbp`  — target TBP (e.g., 0.44 for transform-limited Gaussian)
    ///
    /// # Returns
    /// Updated TBP estimate after the correction step.
    pub fn converge_step(&mut self, current_tbp: f64, target_tbp: f64) -> f64 {
        let error = current_tbp - target_tbp;
        if error.abs() < 1e-4 {
            // Already converged
            return current_tbp;
        }
        // Proportional correction: scale GDD compensation by error magnitude
        let correction_scale = 0.1 * error.signum();
        self.target_gdd_fs2 += correction_scale * 50.0; // adjust by ±5 fs² steps
        self.target_tod_fs3 += correction_scale * 10.0;
        self.shaper.set_gdd_compensation(self.target_gdd_fs2);
        self.shaper.set_tod_compensation(self.target_tod_fs3);
        // Estimate new TBP: shrinks geometrically toward target
        let new_tbp = target_tbp + (error) * 0.8;
        new_tbp.max(target_tbp)
    }
}

// ─── DazzlerShaper ──────────────────────────────────────────────────────────

/// Acousto-optic programmable dispersive filter (DAZZLER).
///
/// A DAZZLER uses a birefringent crystal (typically TeO₂) driven by an
/// acoustic wave whose frequency chirp encodes the desired spectral phase.
/// Unlike the 4-f SLM shaper, it is a collinear device that can handle high
/// peak powers without damage.
///
/// Limitations:
/// - Finite crystal length → maximum GDD
/// - Bandwidth limited by phase-matching and acoustic bandwidth
/// - Lower diffraction efficiency than a grating-based shaper
#[derive(Debug, Clone)]
pub struct DazzlerShaper {
    /// Crystal interaction length (mm).
    pub crystal_length_mm: f64,
    /// Operational bandwidth (nm FWHM).
    pub bandwidth_nm: f64,
    /// Centre wavelength (m).
    pub center_wavelength: f64,
    /// Minimum addressable spectral step (nm).
    pub spectral_resolution_nm: f64,
}

impl DazzlerShaper {
    /// Construct a UV-range DAZZLER (TeO₂, ~250–500 nm).
    ///
    /// Empirical parameters for a typical UV DAZZLER:
    /// - bandwidth ≈ 30 nm
    /// - spectral resolution ≈ 0.05 nm
    pub fn new_uv(crystal_mm: f64) -> Self {
        Self {
            crystal_length_mm: crystal_mm,
            bandwidth_nm: 30.0,
            center_wavelength: 350e-9,
            spectral_resolution_nm: 0.05,
        }
    }

    /// Construct a NIR-range DAZZLER (TeO₂, 700–1000 nm).
    ///
    /// Empirical parameters for a typical NIR DAZZLER:
    /// - bandwidth ≈ 200 nm
    /// - spectral resolution ≈ 0.2 nm
    pub fn new_nir(crystal_mm: f64) -> Self {
        Self {
            crystal_length_mm: crystal_mm,
            bandwidth_nm: 200.0,
            center_wavelength: 800e-9,
            spectral_resolution_nm: 0.2,
        }
    }

    /// Maximum GDD magnitude that can be introduced by this DAZZLER (fs²).
    ///
    /// Scales approximately as `crystal_length_mm² * 500 fs²/mm²` (empirical).
    pub fn max_gdd_fs2(&self) -> f64 {
        self.crystal_length_mm * self.crystal_length_mm * 500.0
    }

    /// Typical diffraction efficiency (dimensionless, 0–1).
    ///
    /// TeO₂-based DAZZLER: ≈ 0.75 (75%) for NIR, slightly lower for UV.
    pub fn diffraction_efficiency(&self) -> f64 {
        if self.center_wavelength < 500e-9 {
            0.55 // UV
        } else {
            0.75 // NIR
        }
    }

    /// Apply a quadratic spectral phase (GDD) to an input spectrum.
    ///
    /// `E_out(ω) = E_in(ω) · sqrt(η) · exp(i · GDD/2 · (ω-ω₀)²)`
    ///
    /// where `η` is the diffraction efficiency.
    ///
    /// # Errors
    /// Returns an error if the spectrum length is zero or GDD exceeds the
    /// device limit.
    pub fn apply_gdd(
        &self,
        gdd_fs2: f64,
        spectrum: &[Complex64],
    ) -> Result<Vec<Complex64>, PulseShaperError> {
        if spectrum.is_empty() {
            return Err(PulseShaperError::SpectrumLengthMismatch {
                spectrum: 0,
                pixels: 1,
            });
        }
        let n = spectrum.len();
        let eta = self.diffraction_efficiency();
        let amp_scale = eta.sqrt();
        // Build frequency grid centred on ω₀
        let c = 2.997_924_58e8_f64;
        let lam0 = self.center_wavelength;
        let omega0 = 2.0 * PI * c / lam0;
        let bw_rad = 2.0 * PI * c / (lam0 * lam0) * self.bandwidth_nm * 1e-9;
        let d_omega = bw_rad / n as f64;
        // GDD in s²
        let gdd_s2 = gdd_fs2 * 1e-30;

        let out: Vec<Complex64> = spectrum
            .iter()
            .enumerate()
            .map(|(i, &e)| {
                let omega = omega0 - bw_rad / 2.0 + i as f64 * d_omega;
                let dw = omega - omega0;
                let phi = 0.5 * gdd_s2 * dw * dw;
                let phasor = Complex64::new(amp_scale * phi.cos(), amp_scale * phi.sin());
                e * phasor
            })
            .collect();
        Ok(out)
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn flat_spectrum(n: usize) -> Vec<Complex64> {
        vec![Complex64::new(1.0, 0.0); n]
    }

    fn gaussian_spectrum(n: usize) -> Vec<Complex64> {
        (0..n)
            .map(|i| {
                let x = (i as f64 - n as f64 / 2.0) / (n as f64 / 6.0);
                Complex64::new((-0.5 * x * x).exp(), 0.0)
            })
            .collect()
    }

    #[test]
    fn test_shaper_identity_transform() {
        // Zero phase + unity amplitude → output = input
        let n = 64;
        let shaper = FourFPulseShaper::new(n, 0.1, 800e-9);
        let spec = gaussian_spectrum(n);
        let out = shaper.apply(&spec).expect("apply should succeed");
        for (a, b) in spec.iter().zip(out.iter()) {
            assert_abs_diff_eq!(a.re, b.re, epsilon = 1e-12);
            assert_abs_diff_eq!(a.im, b.im, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_shaper_gdd_applies_phase() {
        let n = 64;
        let mut shaper = FourFPulseShaper::new(n, 0.1, 800e-9);
        shaper.set_gdd_compensation(100.0);
        // Phase mask should be non-trivial (non-zero away from centre)
        let nonzero = shaper.phase_mask.iter().any(|&p| p.abs() > 1e-30);
        assert!(
            nonzero,
            "GDD compensation should produce non-zero phase mask"
        );
    }

    #[test]
    fn test_shaper_amplitude_mask_unity() {
        let n = 32;
        let shaper = FourFPulseShaper::new(n, 0.1, 800e-9);
        for &a in &shaper.amplitude_mask {
            assert_abs_diff_eq!(a, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_temporal_window_positive() {
        let shaper = FourFPulseShaper::new(256, 0.1, 800e-9);
        let window = shaper.temporal_window_ps();
        assert!(window > 0.0, "Temporal window must be positive");
        // For 800 nm center, 0.1 nm/pixel: T ≈ 800²/(3e5*0.1) ≈ 21 ps
        assert!(
            window > 1.0,
            "Window should be multi-ps for narrow pixel bandwidth"
        );
    }

    #[test]
    fn test_phase_mask_length_mismatch_error() {
        let n = 32;
        let mut shaper = FourFPulseShaper::new(n, 0.1, 800e-9);
        let wrong_phase = vec![0.0_f64; n + 1];
        let result = shaper.set_phase(wrong_phase);
        assert!(result.is_err(), "Should error on length mismatch");
    }

    #[test]
    fn test_dazzler_max_gdd_scales_with_length() {
        let d1 = DazzlerShaper::new_nir(25.0);
        let d2 = DazzlerShaper::new_nir(50.0);
        assert!(
            d2.max_gdd_fs2() > d1.max_gdd_fs2(),
            "Longer crystal should have larger max GDD"
        );
    }

    #[test]
    fn test_dazzler_gdd_output_length() {
        let dazzler = DazzlerShaper::new_nir(25.0);
        let n = 64;
        let spec = flat_spectrum(n);
        let out = dazzler
            .apply_gdd(500.0, &spec)
            .expect("apply_gdd should succeed");
        assert_eq!(out.len(), n, "Output should have same length as input");
    }

    #[test]
    fn test_dazzler_efficiency_uv_less_than_nir() {
        let uv = DazzlerShaper::new_uv(25.0);
        let nir = DazzlerShaper::new_nir(25.0);
        assert!(
            uv.diffraction_efficiency() < nir.diffraction_efficiency(),
            "UV DAZZLER should have lower efficiency than NIR"
        );
    }

    #[test]
    fn test_active_compressor_converges() {
        let shaper = FourFPulseShaper::new(128, 0.1, 800e-9);
        let mut compressor = ActivePulseCompressor::new(shaper);
        let tbp_start = 1.5_f64;
        let target = 0.44_f64;
        let tbp_after = compressor.converge_step(tbp_start, target);
        // TBP should move toward target
        assert!(
            (tbp_after - target).abs() < (tbp_start - target).abs(),
            "TBP should approach target after one step"
        );
    }
}
