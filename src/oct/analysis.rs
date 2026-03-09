//! OCT signal processing and analysis tools.
//!
//! Provides:
//! - [`AScanProcessor`]: processes raw spectral data into depth-resolved A-scans
//! - [`DopplerOct`]: phase-resolved velocity measurement
//! - [`OctMetrics`]: signal quality assessment
//!
//! All processing is pure Rust with no external FFT dependency (uses the
//! radix-2 FFT bundled in the `spectral_domain` sibling module).

use crate::error::OxiPhotonError;
use crate::oct::spectral_domain::SdOct;
use num_complex::Complex64;
use std::f64::consts::PI;

// Re-export the internal FFT for local use
fn fft_inplace(buf: &mut [Complex64]) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two());

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
        let half = len / 2;
        let w_step = -2.0 * PI / len as f64;
        for chunk_start in (0..n).step_by(len) {
            for k in 0..half {
                let angle = w_step * k as f64;
                let w = Complex64::new(angle.cos(), angle.sin());
                let u = buf[chunk_start + k];
                let v = w * buf[chunk_start + k + half];
                buf[chunk_start + k] = u + v;
                buf[chunk_start + k + half] = u - v;
            }
        }
        len <<= 1;
    }
}

fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

// ---------------------------------------------------------------------------
// Window functions
// ---------------------------------------------------------------------------

/// Window functions for sidelobe suppression in A-scan processing.
///
/// The choice of window trades axial resolution for sidelobe rejection:
/// - Rectangular: best resolution, worst sidelobes (−13 dB)
/// - Hann: 50% resolution loss, −32 dB sidelobes
/// - Hamming: similar to Hann but non-zero endpoints
/// - Gaussian: tunable via σ
/// - Blackman-Harris: −92 dB sidelobes, ~2× resolution penalty
#[derive(Debug, Clone)]
pub enum WindowFunction {
    /// No windowing (uniform weights = 1)
    Rectangular,
    /// Hann (raised cosine) window
    Hann,
    /// Hamming window (slightly raised baseline)
    Hamming,
    /// Gaussian window with parameter σ (relative to half-window length, ∈ (0, 0.5])
    Gaussian {
        /// Standard deviation as a fraction of (N-1)/2; typical values 0.3–0.5
        sigma: f64,
    },
    /// 4-term Blackman-Harris window (−92 dB sidelobe level)
    BlackmanHarris,
}

impl WindowFunction {
    /// Evaluate window coefficient at position `i` of `n` total samples.
    fn coefficient(&self, i: usize, n: usize) -> f64 {
        let x = i as f64 / (n - 1).max(1) as f64; // x ∈ [0, 1]
        match self {
            WindowFunction::Rectangular => 1.0,
            WindowFunction::Hann => 0.5 * (1.0 - (2.0 * PI * x).cos()),
            WindowFunction::Hamming => 0.54 - 0.46 * (2.0 * PI * x).cos(),
            WindowFunction::Gaussian { sigma } => {
                let center = 0.5;
                let s = *sigma;
                let arg = (x - center) / s;
                (-0.5 * arg * arg).exp()
            }
            WindowFunction::BlackmanHarris => {
                // Coefficients from Harris (1978)
                let a0 = 0.358_750_287;
                let a1 = 0.488_290_653;
                let a2 = 0.141_279_508;
                let a3 = 0.011_681_552;
                a0 - a1 * (2.0 * PI * x).cos() + a2 * (4.0 * PI * x).cos()
                    - a3 * (6.0 * PI * x).cos()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// A-scan processor
// ---------------------------------------------------------------------------

/// A-scan processor: converts a raw spectral interferogram into a depth profile.
///
/// Encapsulates the full processing chain:
/// 1. DC removal (background subtraction)
/// 2. Dispersion compensation (optional)
/// 3. Windowing
/// 4. k-space resampling
/// 5. FFT → magnitude² → dB
///
/// The processor is configured with the spectral range and sample medium index
/// so that the output depth axis is calibrated in physical units \[μm\].
#[derive(Debug, Clone)]
pub struct AScanProcessor {
    /// Number of spectral sampling points
    pub n_pts: usize,
    /// Minimum wavelength \[nm\]
    pub lambda_min_nm: f64,
    /// Maximum wavelength \[nm\]
    pub lambda_max_nm: f64,
    /// Refractive index of sample medium
    pub sample_index: f64,
    /// Window function applied before FFT
    pub window: WindowFunction,
}

impl AScanProcessor {
    /// Create a new A-scan processor.
    ///
    /// # Parameters
    /// - `n_pts`: number of spectral points (spectrometer pixels)
    /// - `lambda_min_nm`, `lambda_max_nm`: wavelength range \[nm\]
    /// - `n_sample`: sample medium refractive index
    /// - `window`: window function for sidelobe control
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] if the wavelength range is invalid
    /// or the pixel count is zero.
    pub fn new(
        n_pts: usize,
        lambda_min_nm: f64,
        lambda_max_nm: f64,
        n_sample: f64,
        window: WindowFunction,
    ) -> Result<Self, OxiPhotonError> {
        if n_pts == 0 {
            return Err(OxiPhotonError::NumericalError(
                "n_pts must be > 0".to_string(),
            ));
        }
        if lambda_min_nm <= 0.0 || lambda_max_nm <= lambda_min_nm {
            return Err(OxiPhotonError::NumericalError(format!(
                "invalid wavelength range: [{lambda_min_nm}, {lambda_max_nm}] nm"
            )));
        }
        if n_sample < 1.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "sample index must be >= 1.0, got {n_sample}"
            )));
        }
        Ok(Self {
            n_pts,
            lambda_min_nm,
            lambda_max_nm,
            sample_index: n_sample,
            window,
        })
    }

    /// Generate window coefficient vector of length `n_pts`.
    pub fn window_coefficients(&self) -> Vec<f64> {
        (0..self.n_pts)
            .map(|i| self.window.coefficient(i, self.n_pts))
            .collect()
    }

    /// Process a raw spectral interferogram into a depth profile \[dB\].
    ///
    /// # Steps
    /// 1. Remove DC (subtract mean)
    /// 2. Apply window
    /// 3. Zero-pad to next power of two
    /// 4. FFT
    /// 5. Return positive-frequency magnitude-squared in dB
    ///
    /// Output length = next_pow2(n_pts) / 2.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] if input length ≠ `n_pts`.
    pub fn process(&self, raw_spectrum: &[f64]) -> Result<Vec<f64>, OxiPhotonError> {
        if raw_spectrum.len() != self.n_pts {
            return Err(OxiPhotonError::NumericalError(format!(
                "input length {} ≠ n_pts {}",
                raw_spectrum.len(),
                self.n_pts
            )));
        }

        // Step 1: remove DC
        let dc_removed = self.remove_dc(raw_spectrum);

        // Step 2: apply window
        let win = self.window_coefficients();
        let windowed: Vec<Complex64> = dc_removed
            .iter()
            .zip(win.iter())
            .map(|(&v, &w)| Complex64::new(v * w, 0.0))
            .collect();

        // Step 3: zero-pad
        let fft_len = next_pow2(self.n_pts);
        let mut buf = vec![Complex64::new(0.0, 0.0); fft_len];
        for (i, &v) in windowed.iter().enumerate() {
            buf[i] = v;
        }

        // Step 4: FFT
        fft_inplace(&mut buf);

        // Step 5: magnitude² → dB (positive half only)
        let result = buf[..fft_len / 2]
            .iter()
            .map(|c| {
                let p = c.norm_sqr();
                if p > 1e-30 {
                    10.0 * p.log10()
                } else {
                    -300.0
                }
            })
            .collect();

        Ok(result)
    }

    /// Depth axis \[μm\] corresponding to the processed A-scan output.
    ///
    /// Depth step: dz = π / (n · k_range)
    /// where k_range = 2π/λ_min − 2π/λ_max.
    pub fn depth_axis_um(&self) -> Vec<f64> {
        let lambda_min_um = self.lambda_min_nm * 1e-3;
        let lambda_max_um = self.lambda_max_nm * 1e-3;
        let k_range = 2.0 * PI / lambda_min_um - 2.0 * PI / lambda_max_um;
        let dz = PI / (self.sample_index * k_range);
        let fft_len = next_pow2(self.n_pts);
        (0..fft_len / 2).map(|i| i as f64 * dz).collect()
    }

    /// Find peaks in an A-scan profile.
    ///
    /// A sample is considered a peak if it exceeds both its neighbours by at least
    /// `min_prominence_db` dB.
    ///
    /// Returns `(depth_um, amplitude_dB)` pairs, sorted by amplitude descending.
    pub fn find_peaks(&self, a_scan_db: &[f64], min_prominence_db: f64) -> Vec<(f64, f64)> {
        let depths = self.depth_axis_um();
        let n = a_scan_db.len().min(depths.len());
        let mut peaks = Vec::new();
        for i in 1..n.saturating_sub(1) {
            let amp = a_scan_db[i];
            let left = a_scan_db[i - 1];
            let right = a_scan_db[i + 1];
            if amp > left && amp > right {
                // Compute local prominence: difference from surrounding floor
                let local_min = left.min(right);
                let prominence = amp - local_min;
                if prominence >= min_prominence_db {
                    peaks.push((depths[i], amp));
                }
            }
        }
        // Sort by amplitude descending
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        peaks
    }

    /// Remove DC component (subtract mean from each sample).
    pub fn remove_dc(&self, spectrum: &[f64]) -> Vec<f64> {
        if spectrum.is_empty() {
            return Vec::new();
        }
        let mean = spectrum.iter().sum::<f64>() / spectrum.len() as f64;
        spectrum.iter().map(|&v| v - mean).collect()
    }

    /// Dispersion compensation via phase correction in the spectral domain.
    ///
    /// Applies the conjugate dispersion phase to the complex spectrum:
    ///   E_comp(k) = E(k) · exp(−i · \[a2·(k−k₀)² + a3·(k−k₀)³\])
    ///
    /// The real part of the corrected spectrum is returned (for subsequent FFT).
    ///
    /// # Parameters
    /// - `spectrum`: real-valued spectral interferogram
    /// - `a2_um2`: group delay dispersion (GDD) coefficient \[μm²\] (β₂ L / 2)
    /// - `a3_um3`: third-order dispersion (TOD) coefficient \[μm³\]
    pub fn dispersion_compensate(&self, spectrum: &[f64], a2_um2: f64, a3_um3: f64) -> Vec<f64> {
        let n = spectrum.len();
        // Build k array linearly over [k_min, k_max]
        let lambda_min_um = self.lambda_min_nm * 1e-3;
        let lambda_max_um = self.lambda_max_nm * 1e-3;
        let k_min = 2.0 * PI / lambda_max_um;
        let k_max = 2.0 * PI / lambda_min_um;
        let k0 = (k_min + k_max) / 2.0;
        let dk = (k_max - k_min) / (n - 1).max(1) as f64;

        spectrum
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let k = k_min + i as f64 * dk;
                let dk_rel = k - k0;
                // Correction phase: negate to compensate sample dispersion
                let phase = -(a2_um2 * dk_rel * dk_rel + a3_um3 * dk_rel * dk_rel * dk_rel);
                let correction = Complex64::new(phase.cos(), phase.sin());
                let e = Complex64::new(s, 0.0) * correction;
                e.re
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Doppler OCT
// ---------------------------------------------------------------------------

/// Doppler OCT: phase-sensitive velocity measurement.
///
/// Blood flow and other laminar/turbulent flows create a Doppler shift in the
/// detected signal.  By comparing the phase of consecutive A-scans at each
/// depth, the axial velocity component can be recovered:
///
///   v_z = Δφ · λ₀ / (4π · n · Δt)
///
/// where Δφ is the inter-A-scan phase difference and Δt = 1 / f_A_scan.
///
/// Reference: Chen et al., Opt. Lett. 22 (1997) 64.
#[derive(Debug, Clone)]
pub struct DopplerOct {
    /// Underlying SD-OCT system
    pub sd_oct: SdOct,
    /// A-scan acquisition rate \[Hz\]
    pub a_scan_rate_hz: f64,
}

impl DopplerOct {
    /// Create a Doppler OCT analyser.
    ///
    /// # Parameters
    /// - `sd_oct`: configured SD-OCT system
    /// - `rate_hz`: A-scan repetition rate \[Hz\]
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] if rate is non-positive.
    pub fn new(sd_oct: SdOct, rate_hz: f64) -> Result<Self, OxiPhotonError> {
        if rate_hz <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "A-scan rate must be positive, got {rate_hz} Hz"
            )));
        }
        Ok(Self {
            sd_oct,
            a_scan_rate_hz: rate_hz,
        })
    }

    /// Maximum detectable axial velocity (phase aliasing limit) \[mm/s\].
    ///
    /// v_max = λ₀ / (4·n·Δt)  where Δt = 1/f_A
    ///
    /// At this velocity, Δφ = π (Nyquist limit for phase).
    pub fn max_velocity_mm_per_s(&self) -> f64 {
        let lambda0_um = self.sd_oct.center_wavelength_nm * 1e-3;
        let delta_t = 1.0 / self.a_scan_rate_hz;
        // v [μm/s] = λ₀ [μm] / (4 · n · Δt [s])
        let v_um_per_s = lambda0_um / (4.0 * self.sd_oct.sample_index * delta_t);
        v_um_per_s * 1e-3 // convert μm/s → mm/s
    }

    /// Minimum detectable axial velocity \[mm/s\] given a phase noise floor \[rad\].
    ///
    /// v_min = phase_noise · λ₀ / (4π · n · Δt)
    pub fn min_velocity_mm_per_s(&self, phase_noise_rad: f64) -> f64 {
        let lambda0_um = self.sd_oct.center_wavelength_nm * 1e-3;
        let delta_t = 1.0 / self.a_scan_rate_hz;
        let v_um_per_s =
            phase_noise_rad * lambda0_um / (4.0 * PI * self.sd_oct.sample_index * delta_t);
        v_um_per_s * 1e-3
    }

    /// Convert inter-A-scan phase difference to axial velocity \[mm/s\].
    ///
    /// v_z = Δφ · λ₀ / (4π · n · Δt)
    pub fn phase_to_velocity_mm_per_s(&self, delta_phi_rad: f64) -> f64 {
        let lambda0_um = self.sd_oct.center_wavelength_nm * 1e-3;
        let delta_t = 1.0 / self.a_scan_rate_hz;
        let v_um_per_s =
            delta_phi_rad * lambda0_um / (4.0 * PI * self.sd_oct.sample_index * delta_t);
        v_um_per_s * 1e-3
    }

    /// Doppler bandwidth \[Hz\] (= half the A-scan rate, Nyquist theorem).
    ///
    /// BW_D = f_A / 2
    pub fn doppler_bandwidth_hz(&self) -> f64 {
        self.a_scan_rate_hz / 2.0
    }

    /// Velocity resolution \[mm/s\] for a given linear SNR.
    ///
    /// δv = (λ₀ / (4π·n·Δt)) · (1/√SNR)   \[Cramer-Rao bound for phase noise\]
    pub fn velocity_resolution_mm_per_s(&self, snr_linear: f64) -> f64 {
        if snr_linear <= 0.0 {
            return f64::INFINITY;
        }
        let lambda0_um = self.sd_oct.center_wavelength_nm * 1e-3;
        let delta_t = 1.0 / self.a_scan_rate_hz;
        let v_um_per_s =
            lambda0_um / (4.0 * PI * self.sd_oct.sample_index * delta_t * snr_linear.sqrt());
        v_um_per_s * 1e-3
    }

    /// Compute inter-A-scan phase difference at each depth pixel.
    ///
    /// Δφ\[i\] = arg(A₂\[i\] · conj(A₁\[i\]))
    ///
    /// This is the standard complex cross-correlation phase estimator, which is
    /// unbiased and has minimum variance at high SNR.
    pub fn phase_difference(a_scan1: &[Complex64], a_scan2: &[Complex64]) -> Vec<f64> {
        a_scan1
            .iter()
            .zip(a_scan2.iter())
            .map(|(&e1, &e2)| {
                // Cross-correlation: e2 · conj(e1)
                let cross = e2 * e1.conj();
                cross.arg()
            })
            .collect()
    }

    /// Generate a Poiseuille (laminar) flow velocity profile.
    ///
    /// v(r) = v_max · (1 − r²/R²),  r ∈ \[0, R_max\]
    ///
    /// # Parameters
    /// - `v_max_mm_per_s`: centreline (peak) velocity \[mm/s\]
    /// - `r_max_um`: vessel radius \[μm\]
    /// - `n_pts`: number of radial sample points
    ///
    /// Returns `(radius_um, velocity_mm_per_s)` pairs from r = 0 to r = R.
    pub fn poiseuille_profile(v_max_mm_per_s: f64, r_max_um: f64, n_pts: usize) -> Vec<(f64, f64)> {
        if n_pts == 0 || r_max_um <= 0.0 {
            return Vec::new();
        }
        (0..n_pts)
            .map(|i| {
                let r = i as f64 / (n_pts - 1).max(1) as f64 * r_max_um;
                let v = v_max_mm_per_s * (1.0 - (r / r_max_um).powi(2));
                (r, v)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// OCT signal quality metrics
// ---------------------------------------------------------------------------

/// Collection of signal quality metrics for OCT A-scans.
///
/// All methods operate on dB-scaled A-scan vectors produced by
/// [`AScanProcessor::process`] or \[`SdOct::compute_a_scan`\].
pub struct OctMetrics;

impl OctMetrics {
    /// Signal-to-noise ratio \[dB\] at a specific A-scan index.
    ///
    /// SNR = signal_amplitude\[signal_idx\] − mean(noise_region)
    ///
    /// # Parameters
    /// - `a_scan_db`: dB-scaled A-scan
    /// - `signal_idx`: index of the signal peak
    /// - `noise_region`: `(start, end)` index range for noise estimation (exclusive end)
    pub fn snr_db(a_scan_db: &[f64], signal_idx: usize, noise_region: (usize, usize)) -> f64 {
        if a_scan_db.is_empty() || signal_idx >= a_scan_db.len() {
            return f64::NEG_INFINITY;
        }
        let signal = a_scan_db[signal_idx];
        let noise = Self::noise_floor_db(a_scan_db, noise_region);
        signal - noise
    }

    /// Dynamic range \[dB\]: peak amplitude minus noise floor.
    ///
    /// DR = max(A-scan) − min(A-scan)
    pub fn dynamic_range_db(a_scan_db: &[f64]) -> f64 {
        if a_scan_db.is_empty() {
            return 0.0;
        }
        let max = a_scan_db.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = a_scan_db.iter().cloned().fold(f64::INFINITY, f64::min);
        (max - min).max(0.0)
    }

    /// Measure axial resolution \[μm\] from a measured A-scan peak via FWHM.
    ///
    /// Locates the peak, then walks outward to find the −3 dB (half-power) points.
    /// Uses linear interpolation for sub-pixel accuracy.
    ///
    /// # Parameters
    /// - `a_scan`: dB A-scan (positive-depth half)
    /// - `depth_axis`: corresponding depth values \[μm\] (same length as `a_scan`)
    ///
    /// Returns 0.0 if the peak cannot be found or the half-power points fall outside
    /// the array bounds.
    pub fn measure_axial_resolution_um(a_scan: &[f64], depth_axis: &[f64]) -> f64 {
        let n = a_scan.len().min(depth_axis.len());
        if n < 3 {
            return 0.0;
        }
        // Find index of global maximum
        let (peak_idx, &peak_val) = a_scan[..n]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        let half_max = peak_val - 3.0; // −3 dB = FWHM threshold

        // Walk left from peak
        let left_depth = (1..=peak_idx)
            .rev()
            .find(|&i| a_scan[i - 1] <= half_max)
            .map(|i| {
                // Linear interpolation between i-1 and i
                let d0 = depth_axis[i - 1];
                let d1 = depth_axis[i];
                let a0 = a_scan[i - 1];
                let a1 = a_scan[i];
                if (a1 - a0).abs() < 1e-12 {
                    d0
                } else {
                    d0 + (half_max - a0) / (a1 - a0) * (d1 - d0)
                }
            })
            .unwrap_or(depth_axis[0]);

        // Walk right from peak
        let right_depth = (peak_idx..n - 1)
            .find(|&i| a_scan[i + 1] <= half_max)
            .map(|i| {
                let d0 = depth_axis[i];
                let d1 = depth_axis[i + 1];
                let a0 = a_scan[i];
                let a1 = a_scan[i + 1];
                if (a1 - a0).abs() < 1e-12 {
                    d1
                } else {
                    d0 + (half_max - a0) / (a1 - a0) * (d1 - d0)
                }
            })
            .unwrap_or(depth_axis[n - 1]);

        (right_depth - left_depth).abs()
    }

    /// Contrast-to-noise ratio \[dB\].
    ///
    /// CNR = 20·log10(|signal_mean − background_mean| / noise_std)
    pub fn cnr_db(signal_mean: f64, background_mean: f64, noise_std: f64) -> f64 {
        if noise_std <= 0.0 {
            return f64::INFINITY;
        }
        let contrast = (signal_mean - background_mean).abs();
        if contrast <= 0.0 {
            return f64::NEG_INFINITY;
        }
        20.0 * (contrast / noise_std).log10()
    }

    /// Noise floor \[dB\]: mean of A-scan values in the specified index range.
    ///
    /// # Parameters
    /// - `a_scan_db`: dB A-scan
    /// - `noise_region`: `(start, end)` index range for noise estimation (exclusive end)
    pub fn noise_floor_db(a_scan_db: &[f64], noise_region: (usize, usize)) -> f64 {
        let (start, end) = noise_region;
        let end = end.min(a_scan_db.len());
        if start >= end {
            return f64::NEG_INFINITY;
        }
        let slice = &a_scan_db[start..end];
        slice.iter().sum::<f64>() / slice.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_sd_oct() -> SdOct {
        SdOct::new(830.0, 70.0, 1024, 10.0, 0.1, 1.35).unwrap()
    }

    fn make_processor() -> AScanProcessor {
        AScanProcessor::new(1024, 795.0, 865.0, 1.35, WindowFunction::Hann).unwrap()
    }

    // -------------------------------------------------------------------------
    // test_window_hann_endpoints
    // -------------------------------------------------------------------------
    #[test]
    fn test_window_hann_endpoints() {
        let proc = make_processor();
        let win = proc.window_coefficients();
        assert!(!win.is_empty());
        assert_relative_eq!(win[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(*win.last().unwrap(), 0.0, epsilon = 1e-10);
    }

    // -------------------------------------------------------------------------
    // test_window_rectangular_uniform
    // -------------------------------------------------------------------------
    #[test]
    fn test_window_rectangular_uniform() {
        let proc =
            AScanProcessor::new(64, 795.0, 865.0, 1.35, WindowFunction::Rectangular).unwrap();
        let win = proc.window_coefficients();
        for &w in &win {
            assert_relative_eq!(w, 1.0, epsilon = 1e-12);
        }
    }

    // -------------------------------------------------------------------------
    // test_dc_removal
    // -------------------------------------------------------------------------
    #[test]
    fn test_dc_removal() {
        let proc = make_processor();
        // Construct a signal with a known non-zero mean
        let spectrum: Vec<f64> = (0..1024).map(|i| 5.0 + (i as f64 * 0.01).sin()).collect();
        let dc_removed = proc.remove_dc(&spectrum);
        let mean = dc_removed.iter().sum::<f64>() / dc_removed.len() as f64;
        assert_relative_eq!(mean, 0.0, epsilon = 1e-9);
    }

    // -------------------------------------------------------------------------
    // test_a_scan_processor_output_length
    // -------------------------------------------------------------------------
    #[test]
    fn test_a_scan_processor_output_length() {
        let proc = make_processor();
        let spectrum = vec![1.0f64; 1024];
        let out = proc.process(&spectrum).unwrap();
        // Output should be next_pow2(1024)/2 = 512
        assert_eq!(out.len(), 512);
    }

    // -------------------------------------------------------------------------
    // test_doppler_max_velocity
    // -------------------------------------------------------------------------
    #[test]
    fn test_doppler_max_velocity() {
        let sd_oct = make_sd_oct();
        let rate_hz = 20_000.0; // 20 kHz A-scan rate
        let dop = DopplerOct::new(sd_oct.clone(), rate_hz).unwrap();
        // v_max = λ₀ / (4·n·Δt) with Δt = 1/rate
        let lambda0_um = sd_oct.center_wavelength_nm * 1e-3;
        let delta_t = 1.0 / rate_hz;
        let expected_mm_per_s = lambda0_um / (4.0 * sd_oct.sample_index * delta_t) * 1e-3;
        assert_relative_eq!(
            dop.max_velocity_mm_per_s(),
            expected_mm_per_s,
            epsilon = 1e-6
        );
    }

    // -------------------------------------------------------------------------
    // test_phase_to_velocity_conversion
    // -------------------------------------------------------------------------
    #[test]
    fn test_phase_to_velocity_conversion() {
        let sd_oct = make_sd_oct();
        let dop = DopplerOct::new(sd_oct, 20_000.0).unwrap();
        // At Δφ = π → v = v_max
        let v_max = dop.max_velocity_mm_per_s();
        let v_from_phase = dop.phase_to_velocity_mm_per_s(PI);
        assert_relative_eq!(v_from_phase, v_max, epsilon = 1e-6);
    }

    // -------------------------------------------------------------------------
    // test_poiseuille_profile_max_at_center
    // -------------------------------------------------------------------------
    #[test]
    fn test_poiseuille_profile_max_at_center() {
        let v_max = 10.0; // mm/s
        let r_max = 50.0; // μm
        let n_pts = 101;
        let profile = DopplerOct::poiseuille_profile(v_max, r_max, n_pts);
        assert_eq!(profile.len(), n_pts);
        // First point is at r=0 and should have v = v_max
        let (r0, v0) = profile[0];
        assert_relative_eq!(r0, 0.0, epsilon = 1e-12);
        assert_relative_eq!(v0, v_max, epsilon = 1e-10);
        // Last point at r=R should have v ≈ 0
        let (r_last, v_last) = *profile.last().unwrap();
        assert_relative_eq!(r_last, r_max, epsilon = 1e-10);
        assert_relative_eq!(v_last, 0.0, epsilon = 1e-10);
        // Verify monotonic decrease
        let velocities: Vec<f64> = profile.iter().map(|&(_, v)| v).collect();
        for i in 1..velocities.len() {
            assert!(
                velocities[i] <= velocities[i - 1] + 1e-10,
                "velocity not monotonically decreasing at i={i}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // test_dynamic_range_db_positive
    // -------------------------------------------------------------------------
    #[test]
    fn test_dynamic_range_db_positive() {
        // A-scan with a clear peak
        let mut a_scan = vec![-80.0f64; 512];
        a_scan[100] = -20.0; // peak
        let dr = OctMetrics::dynamic_range_db(&a_scan);
        assert!(dr > 0.0, "dynamic range should be positive, got {dr}");
        assert_relative_eq!(dr, 60.0, epsilon = 1e-10);
    }
}
