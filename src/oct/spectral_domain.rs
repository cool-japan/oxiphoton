//! Spectral-domain OCT (SD-OCT) system model and related OCT variants.
//!
//! SD-OCT uses a broadband light source with a spectrometer-based detector to
//! simultaneously acquire all depth information without mechanical depth scanning.
//! The depth (A-scan) profile is retrieved via Fourier transform of the spectral
//! interference fringe pattern.
//!
//! Physics summary:
//! - Axial resolution:  δz = (2·ln2/π) · λ₀²/(n·Δλ)   \[FWHM, Gaussian source\]
//! - Lateral resolution: δx = 0.61·λ₀/NA               [Rayleigh criterion]
//! - Max depth:         z_max = λ₀²/(4·n·δλ_pixel)     [Nyquist limit]
//! - Sensitivity (SD):  S ≈ ρ·P_s/(hν·BW)              \[shot-noise limited\]
//!
//! References:
//! - Wojtkowski et al., Opt. Express 12 (2004) 2404
//! - Leitgeb et al., Opt. Express 11 (2003) 889

use crate::error::OxiPhotonError;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Speed of light in vacuum \[μm/fs\] — expressed in SI as m/s
const C0_M_PER_S: f64 = 2.997_924_58e8;
/// Planck constant \[J·s\]
const HBAR_J_S: f64 = 6.626_070_15e-34;
/// Elementary charge \[C\]
const E_CHARGE: f64 = 1.602_176_634e-19;

// ---------------------------------------------------------------------------
// Utility: minimal radix-2 FFT (no external crate needed for modest N)
// ---------------------------------------------------------------------------

/// In-place Cooley-Tukey radix-2 DIT FFT.
/// `buf` length must be a power of two.
fn fft_inplace(buf: &mut [Complex64]) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two(), "FFT length must be a power of two");

    // Bit-reversal permutation
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

    // Butterfly stages
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

/// Round `n` up to the nearest power of two (minimum 2).
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
// SD-OCT
// ---------------------------------------------------------------------------

/// Spectral-domain OCT system model.
///
/// Simulates the core optical parameters and signal processing chain of an
/// SD-OCT instrument: broadband source, beam splitter, sample/reference arms,
/// and a grating-based spectrometer.
#[derive(Debug, Clone)]
pub struct SdOct {
    /// Centre wavelength λ₀ \[nm\]
    pub center_wavelength_nm: f64,
    /// Full-width at half-maximum source bandwidth Δλ \[nm\]
    pub bandwidth_nm: f64,
    /// Number of spectrometer pixels
    pub n_pixels: usize,
    /// Spectrometer pixel size \[μm\] (used for sensitivity estimate)
    pub pixel_size_um: f64,
    /// Objective numerical aperture (determines lateral resolution)
    pub objective_na: f64,
    /// Refractive index of sample medium (affects axial resolution and depth)
    pub sample_index: f64,
    /// Fraction of source power directed to the reference arm (typically 0.5)
    pub reference_power_fraction: f64,
}

impl SdOct {
    /// Construct a new SD-OCT system.
    ///
    /// # Parameters
    /// - `lambda0_nm`: centre wavelength \[nm\]
    /// - `bw_nm`: source FWHM bandwidth \[nm\]
    /// - `n_pixels`: spectrometer pixel count
    /// - `pixel_um`: spectrometer pixel pitch \[μm\]
    /// - `na`: objective numerical aperture
    /// - `n_sample`: refractive index of sample medium
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::InvalidWavelength`] if wavelength or bandwidth
    /// are not positive, or [`OxiPhotonError::NumericalError`] for invalid NA / index.
    pub fn new(
        lambda0_nm: f64,
        bw_nm: f64,
        n_pixels: usize,
        pixel_um: f64,
        na: f64,
        n_sample: f64,
    ) -> Result<Self, OxiPhotonError> {
        if lambda0_nm <= 0.0 || !lambda0_nm.is_finite() {
            return Err(OxiPhotonError::InvalidWavelength(lambda0_nm * 1e-9));
        }
        if bw_nm <= 0.0 || !bw_nm.is_finite() {
            return Err(OxiPhotonError::NumericalError(format!(
                "bandwidth must be positive, got {bw_nm} nm"
            )));
        }
        if na <= 0.0 || na >= 1.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "NA must be in (0, 1), got {na}"
            )));
        }
        if n_sample < 1.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "sample refractive index must be >= 1.0, got {n_sample}"
            )));
        }
        Ok(Self {
            center_wavelength_nm: lambda0_nm,
            bandwidth_nm: bw_nm,
            n_pixels,
            pixel_size_um: pixel_um,
            objective_na: na,
            sample_index: n_sample,
            reference_power_fraction: 0.5,
        })
    }

    /// Axial (depth) resolution \[μm\], FWHM for a Gaussian source.
    ///
    /// δz = (2·ln2 / π) · λ₀² / (n · Δλ)
    ///
    /// All wavelength quantities in the same unit; result in μm when λ₀ \[μm\] used.
    pub fn axial_resolution_um(&self) -> f64 {
        let lambda0_um = self.center_wavelength_nm * 1e-3;
        let bw_um = self.bandwidth_nm * 1e-3;
        (2.0 * 2_f64.ln() / PI) * lambda0_um * lambda0_um / (self.sample_index * bw_um)
    }

    /// Lateral resolution \[μm\] (Rayleigh criterion).
    ///
    /// δx = 0.61 · λ₀ / NA
    pub fn lateral_resolution_um(&self) -> f64 {
        0.61 * self.center_wavelength_nm * 1e-3 / self.objective_na
    }

    /// Confocal parameter (depth of focus) \[μm\].
    ///
    /// b = π · δx² / λ₀   (1/e² half-width of Gaussian beam waist, doubled for
    /// Rayleigh range convention)
    pub fn depth_of_focus_um(&self) -> f64 {
        let dx = self.lateral_resolution_um();
        let lambda0_um = self.center_wavelength_nm * 1e-3;
        PI * dx * dx / lambda0_um
    }

    /// Maximum unambiguous imaging depth \[μm\].
    ///
    /// z_max = λ₀² / (4 · n · δλ_pixel),  δλ_pixel = Δλ / N_pixels
    pub fn max_depth_um(&self) -> f64 {
        let lambda0_um = self.center_wavelength_nm * 1e-3;
        let d_lambda_pixel_um = self.bandwidth_nm * 1e-3 / self.n_pixels as f64;
        lambda0_um * lambda0_um / (4.0 * self.sample_index * d_lambda_pixel_um)
    }

    /// Wavenumber array \[1/μm\] for each spectrometer pixel, linearly spaced in k-space.
    ///
    /// k = 2π/λ,  k spans \[k_min, k_max\] symmetrically around k₀ = 2π/λ₀.
    pub fn k_array_per_um(&self) -> Vec<f64> {
        let lambda0_um = self.center_wavelength_nm * 1e-3;
        let bw_um = self.bandwidth_nm * 1e-3;
        let lambda_min_um = lambda0_um - bw_um / 2.0;
        let lambda_max_um = lambda0_um + bw_um / 2.0;
        let k_min = 2.0 * PI / lambda_max_um;
        let k_max = 2.0 * PI / lambda_min_um;
        let dk = (k_max - k_min) / (self.n_pixels - 1).max(1) as f64;
        (0..self.n_pixels).map(|i| k_min + i as f64 * dk).collect()
    }

    /// Wavelength array \[nm\] for each spectrometer pixel.
    ///
    /// Linearly spaced from λ₀ − Δλ/2 to λ₀ + Δλ/2.
    pub fn wavelength_array_nm(&self) -> Vec<f64> {
        let lambda_min = self.center_wavelength_nm - self.bandwidth_nm / 2.0;
        let lambda_max = self.center_wavelength_nm + self.bandwidth_nm / 2.0;
        let step = (lambda_max - lambda_min) / (self.n_pixels - 1).max(1) as f64;
        (0..self.n_pixels)
            .map(|i| lambda_min + i as f64 * step)
            .collect()
    }

    /// Source spectrum (Gaussian envelope) sampled at k-space pixels.
    ///
    /// S(k) = exp(−(k − k₀)² / (2·σ_k²))
    ///
    /// where σ_k corresponds to the FWHM bandwidth Δλ converted to k-space.
    pub fn source_spectrum(&self) -> Vec<f64> {
        let lambda0_um = self.center_wavelength_nm * 1e-3;
        let bw_um = self.bandwidth_nm * 1e-3;
        let k0 = 2.0 * PI / lambda0_um;
        // Convert FWHM Δλ to Δk: Δk ≈ (2π/λ₀²)·Δλ
        let dk_fwhm = 2.0 * PI / (lambda0_um * lambda0_um) * bw_um;
        let sigma_k = dk_fwhm / (2.0 * 2_f64.ln().sqrt());
        self.k_array_per_um()
            .iter()
            .map(|&k| {
                let dk = k - k0;
                (-dk * dk / (2.0 * sigma_k * sigma_k)).exp()
            })
            .collect()
    }

    /// Spectral interference fringe for a single reflector at depth `depth_um`.
    ///
    /// I(k) = S(k) · \[R_r + R_s + 2·√(R_r·R_s)·cos(2·k·n·z)\]
    ///
    /// The factor 2 in the argument accounts for the double-pass path difference.
    pub fn interference_fringe(&self, depth_um: f64, r_sample: f64, r_ref: f64) -> Vec<f64> {
        let spectrum = self.source_spectrum();
        let k_arr = self.k_array_per_um();
        let n = self.sample_index;
        k_arr
            .iter()
            .zip(spectrum.iter())
            .map(|(&k, &s)| {
                let dc = r_ref + r_sample;
                let interference = 2.0 * (r_ref * r_sample).sqrt() * (2.0 * k * n * depth_um).cos();
                s * (dc + interference)
            })
            .collect()
    }

    /// Simulate interference from multiple sample layers (coherent superposition).
    ///
    /// I(k) = S(k) · |√R_r · exp(0) + Σⱼ √Rⱼ · exp(i·2·k·n·zⱼ)|²
    ///
    /// # Parameters
    /// - `layers`: slice of `(depth_um, reflectivity)` pairs
    /// - `r_ref`: reference arm reflectivity
    pub fn multi_layer_fringe(&self, layers: &[(f64, f64)], r_ref: f64) -> Vec<f64> {
        let spectrum = self.source_spectrum();
        let k_arr = self.k_array_per_um();
        let n = self.sample_index;

        k_arr
            .iter()
            .zip(spectrum.iter())
            .map(|(&k, &s)| {
                // Reference field: amplitude √R_r, phase 0
                let e_ref = Complex64::new(r_ref.sqrt(), 0.0);
                // Sum of sample fields
                let e_sample: Complex64 = layers
                    .iter()
                    .map(|&(z, r)| {
                        let phase = 2.0 * k * n * z;
                        Complex64::new(r.sqrt(), 0.0) * Complex64::new(phase.cos(), phase.sin())
                    })
                    .sum();
                let total = e_ref + e_sample;
                s * total.norm_sqr()
            })
            .collect()
    }

    /// Compute A-scan from a spectral interference fringe.
    ///
    /// Processing steps:
    /// 1. Subtract DC (mean of fringe)
    /// 2. Apply Hann window (sidelobe suppression)
    /// 3. Zero-pad to next power of two
    /// 4. FFT
    /// 5. Take magnitude squared → intensity in dB
    ///
    /// Returns `(depth_um, intensity_dB)` pairs for the positive-depth half.
    pub fn compute_a_scan(&self, fringe: &[f64]) -> Vec<(f64, f64)> {
        let n = fringe.len();
        // Step 1: Remove DC
        let mean = fringe.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = fringe.iter().map(|&v| v - mean).collect();

        // Step 2: Hann window
        let windowed: Vec<Complex64> = centered
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
                Complex64::new(v * w, 0.0)
            })
            .collect();

        // Step 3: Zero-pad to next power of two
        let fft_len = next_pow2(n);
        let mut buf = vec![Complex64::new(0.0, 0.0); fft_len];
        for (i, &v) in windowed.iter().enumerate() {
            buf[i] = v;
        }

        // Step 4: FFT
        fft_inplace(&mut buf);

        // Step 5: Magnitude squared → dB, map to depth axis
        // Depth axis: z = i / (2·n·k_range / (2π)) for i in 0..fft_len/2
        let k_arr = self.k_array_per_um();
        let k_range = k_arr.last().copied().unwrap_or(1.0) - k_arr.first().copied().unwrap_or(0.0);
        // dz step from k-space FFT: dz = π / (n · k_range)
        // total depth range = fft_len/2 · dz
        let dz = PI / (self.sample_index * k_range);

        let half = fft_len / 2;
        (0..half)
            .map(|i| {
                let depth = i as f64 * dz;
                let power = buf[i].norm_sqr();
                let db = if power > 1e-30 {
                    10.0 * power.log10()
                } else {
                    -300.0
                };
                (depth, db)
            })
            .collect()
    }

    /// Resample fringe from λ-space to k-space (linear in k).
    ///
    /// The spectrometer natively measures in λ-space; FFT requires linear-k spacing.
    /// Uses linear interpolation.
    pub fn resample_to_k_space(&self, fringe_lambda: &[f64]) -> Vec<f64> {
        let n = fringe_lambda.len();
        let lambda_min = self.center_wavelength_nm - self.bandwidth_nm / 2.0;
        let lambda_max = self.center_wavelength_nm + self.bandwidth_nm / 2.0;
        // Lambda array (linearly spaced in λ, matching fringe_lambda)
        let dlambda = (lambda_max - lambda_min) / (n - 1).max(1) as f64;

        // Target: linearly spaced k array
        let k_arr = self.k_array_per_um();

        k_arr
            .iter()
            .map(|&k| {
                // k → λ [nm] = 2π/k (k in 1/μm → λ in μm → ×1000 → nm)
                let lambda_um = 2.0 * PI / k;
                let lambda_nm = lambda_um * 1e3;
                // Find position in λ array
                let idx_f = (lambda_nm - lambda_min) / dlambda;
                let idx_lo = idx_f.floor() as isize;
                let frac = idx_f - idx_lo as f64;
                if idx_lo < 0 {
                    fringe_lambda[0]
                } else if idx_lo as usize + 1 >= n {
                    fringe_lambda[n - 1]
                } else {
                    let lo = idx_lo as usize;
                    fringe_lambda[lo] * (1.0 - frac) + fringe_lambda[lo + 1] * frac
                }
            })
            .collect()
    }

    /// Shot-noise–limited sensitivity \[dB\] for specular reflector (R = 1).
    ///
    /// SD-OCT sensitivity: S_SD ≈ ρ·P_s / (2·e·BW)
    ///
    /// where ρ is detector quantum efficiency (responsivity), P_s is source power,
    /// e is elementary charge, and BW is detection bandwidth.
    ///
    /// # Parameters
    /// - `source_power_mw`: source power \[mW\]
    /// - `detector_efficiency`: quantum efficiency η ∈ (0, 1]
    pub fn sensitivity_db(&self, source_power_mw: f64, detector_efficiency: f64) -> f64 {
        let p_w = source_power_mw * 1e-3;
        let lambda0_m = self.center_wavelength_nm * 1e-9;
        // Photon energy: E = h·c/λ
        let photon_energy = HBAR_J_S * C0_M_PER_S / lambda0_m;
        // Responsivity [A/W] = η·e/(h·ν)
        let responsivity = detector_efficiency * E_CHARGE / photon_energy;
        // Detection bandwidth BW ≈ A-scan rate; use 1 Hz for normalised sensitivity
        // Typically BW_eff = Δf (acquisition bandwidth per pixel × N_pixels/2)
        // For sensitivity figure: SNR_max = ρ·P / (2·e·BW)
        let bw_hz = 1.0; // normalised; gives sensitivity in dB·Hz
        let snr_linear = responsivity * p_w / (2.0 * E_CHARGE * bw_hz);
        10.0 * snr_linear.log10()
    }

    /// Shot-noise–limited SNR \[dB\] for a sample reflector with reflectivity R.
    ///
    /// SNR = sensitivity_dB + 10·log10(R)
    pub fn snr_db(&self, reflectivity: f64, source_power_mw: f64, det_efficiency: f64) -> f64 {
        self.sensitivity_db(source_power_mw, det_efficiency)
            + 10.0 * reflectivity.max(1e-20).log10()
    }

    /// Roll-off: sensitivity reduction with depth due to finite spectrometer pixel bandwidth.
    ///
    /// R(z) = 20·log10\[sinc(z / z_max)\]  \[dB\]
    ///
    /// At z = 0: 0 dB; decreasing monotonically toward z_max.
    pub fn roll_off_db(&self, depth_um: f64) -> f64 {
        let z_max = self.max_depth_um();
        if z_max <= 0.0 {
            return 0.0;
        }
        let x = depth_um / z_max;
        // sinc(x) = sin(π·x)/(π·x); at x=0 sinc=1
        let sinc_val = if x.abs() < 1e-12 {
            1.0
        } else {
            (PI * x).sin() / (PI * x)
        };
        20.0 * sinc_val.abs().max(1e-20).log10()
    }
}

// ---------------------------------------------------------------------------
// TD-OCT
// ---------------------------------------------------------------------------

/// Time-domain OCT system model.
///
/// In TD-OCT the reference mirror is scanned to match each depth sequentially.
/// The coherence gate selects light from the matching optical path length.
/// Sensitivity advantage of SD-OCT over TD-OCT: ~N·ln(N) (for N pixels).
#[derive(Debug, Clone)]
pub struct TdOct {
    /// Centre wavelength λ₀ \[nm\]
    pub center_wavelength_nm: f64,
    /// Source FWHM bandwidth Δλ \[nm\]
    pub bandwidth_nm: f64,
    /// Objective NA
    pub na: f64,
    /// Refractive index of sample medium
    pub sample_index: f64,
}

impl TdOct {
    /// Construct a new TD-OCT model.
    pub fn new(
        lambda0_nm: f64,
        bw_nm: f64,
        na: f64,
        n_sample: f64,
    ) -> Result<Self, OxiPhotonError> {
        if lambda0_nm <= 0.0 || !lambda0_nm.is_finite() {
            return Err(OxiPhotonError::InvalidWavelength(lambda0_nm * 1e-9));
        }
        if bw_nm <= 0.0 || !bw_nm.is_finite() {
            return Err(OxiPhotonError::NumericalError(format!(
                "bandwidth must be positive, got {bw_nm} nm"
            )));
        }
        Ok(Self {
            center_wavelength_nm: lambda0_nm,
            bandwidth_nm: bw_nm,
            na,
            sample_index: n_sample,
        })
    }

    /// Axial resolution \[μm\] (same formula as SD-OCT).
    pub fn axial_resolution_um(&self) -> f64 {
        let lambda0_um = self.center_wavelength_nm * 1e-3;
        let bw_um = self.bandwidth_nm * 1e-3;
        (2.0 * 2_f64.ln() / PI) * lambda0_um * lambda0_um / (self.sample_index * bw_um)
    }

    /// Coherence length \[μm\] (1/e half-width of coherence envelope).
    ///
    /// l_c = (2·ln2 / π) · λ₀² / Δλ   (in sample medium: divide by n)
    ///
    /// This is identical to axial_resolution_um but without the sample index factor
    /// when defined in free-space terms; here we include n for the in-medium value.
    pub fn coherence_length_um(&self) -> f64 {
        self.axial_resolution_um()
    }

    /// Coherence gate signal strength for a reflector at a given path difference.
    ///
    /// The interference signal follows a Gaussian envelope modulated at k₀:
    ///
    /// I(Δl) = √R_s · exp(−(Δl / l_c)²) · cos(2·k₀·n·Δl/2)
    ///
    /// Returns the envelope amplitude (ignoring carrier oscillation).
    ///
    /// # Parameters
    /// - `path_difference_um`: optical path difference Δl \[μm\]
    /// - `r_sample`: sample reflectivity
    pub fn coherence_gate_signal(&self, path_difference_um: f64, r_sample: f64) -> f64 {
        let lc = self.coherence_length_um();
        if lc <= 0.0 {
            return 0.0;
        }
        let norm = path_difference_um / lc;
        r_sample.sqrt() * (-norm * norm).exp()
    }
}

// ---------------------------------------------------------------------------
// SS-OCT
// ---------------------------------------------------------------------------

/// Swept-source OCT (SS-OCT) system model.
///
/// In SS-OCT, a narrowband tunable laser sweeps across the source bandwidth.
/// Unlike SD-OCT, no spectrometer is required; a single balanced photodetector
/// acquires the time-resolved interferogram as the laser tunes.
///
/// Key advantage: longer coherence length at each instant → deeper imaging,
/// and balanced detection for RIN suppression.
#[derive(Debug, Clone)]
pub struct SsOct {
    /// Centre wavelength λ₀ \[nm\]
    pub center_wavelength_nm: f64,
    /// Total sweep bandwidth Δλ \[nm\]
    pub sweep_bandwidth_nm: f64,
    /// Sweep repetition rate \[kHz\] (= A-scan rate)
    pub sweep_rate_khz: f64,
    /// Number of sampling points per sweep
    pub n_samples_per_sweep: usize,
    /// Objective NA
    pub na: f64,
    /// Refractive index of sample medium
    pub sample_index: f64,
}

impl SsOct {
    /// Construct a new SS-OCT model.
    pub fn new(
        lambda0_nm: f64,
        bw_nm: f64,
        rate_khz: f64,
        n_samples: usize,
        na: f64,
        n_sample: f64,
    ) -> Result<Self, OxiPhotonError> {
        if lambda0_nm <= 0.0 || !lambda0_nm.is_finite() {
            return Err(OxiPhotonError::InvalidWavelength(lambda0_nm * 1e-9));
        }
        if bw_nm <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "sweep bandwidth must be positive, got {bw_nm} nm"
            )));
        }
        if rate_khz <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "sweep rate must be positive, got {rate_khz} kHz"
            )));
        }
        Ok(Self {
            center_wavelength_nm: lambda0_nm,
            sweep_bandwidth_nm: bw_nm,
            sweep_rate_khz: rate_khz,
            n_samples_per_sweep: n_samples,
            na,
            sample_index: n_sample,
        })
    }

    /// Axial resolution \[μm\] (identical formula to SD-OCT / TD-OCT).
    pub fn axial_resolution_um(&self) -> f64 {
        let lambda0_um = self.center_wavelength_nm * 1e-3;
        let bw_um = self.sweep_bandwidth_nm * 1e-3;
        (2.0 * 2_f64.ln() / PI) * lambda0_um * lambda0_um / (self.sample_index * bw_um)
    }

    /// A-scan acquisition rate \[kHz\] (equals the laser sweep rate).
    pub fn a_scan_rate_khz(&self) -> f64 {
        self.sweep_rate_khz
    }

    /// Maximum unambiguous imaging depth \[μm\].
    ///
    /// z_max = λ₀² / (4·n·δλ_sample),  δλ_sample = Δλ / N_samples
    pub fn max_depth_um(&self) -> f64 {
        let lambda0_um = self.center_wavelength_nm * 1e-3;
        let d_lambda_um = self.sweep_bandwidth_nm * 1e-3 / self.n_samples_per_sweep as f64;
        lambda0_um * lambda0_um / (4.0 * self.sample_index * d_lambda_um)
    }

    /// Sensitivity advantage of SS-OCT over TD-OCT \[dB\].
    ///
    /// SS-OCT (like SD-OCT) gains sensitivity because all spectral components are
    /// measured simultaneously.  The advantage ≈ 10·log10(N/2) where N is the number
    /// of resolution elements, which for practical systems is typically 20–30 dB.
    pub fn sensitivity_advantage_db(&self) -> f64 {
        let n_res = self.n_samples_per_sweep as f64 / 2.0;
        10.0 * n_res.max(1.0).log10()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn standard_sd_oct() -> SdOct {
        SdOct::new(830.0, 70.0, 1024, 10.0, 0.1, 1.35).expect("valid SD-OCT params")
    }

    // -------------------------------------------------------------------------
    // test_sd_oct_axial_resolution
    // -------------------------------------------------------------------------
    #[test]
    fn test_sd_oct_axial_resolution() {
        // λ₀ = 830 nm, Δλ = 70 nm, n = 1.35
        // δz = (2·ln2/π) · (0.83)² / (1.35 · 0.07) = (0.4413) · 0.6889 / 0.0945 ≈ 3.22 μm
        let oct = standard_sd_oct();
        let dz = oct.axial_resolution_um();
        // Theoretical: (2·ln2/π) · 0.83² / (1.35 · 0.07)
        let expected = (2.0 * 2_f64.ln() / PI) * 0.83_f64.powi(2) / (1.35 * 0.07);
        assert_relative_eq!(dz, expected, epsilon = 1e-6);
        // Sanity: should be in the range 2–10 μm for these parameters
        assert!(
            dz > 1.0 && dz < 15.0,
            "axial resolution {dz} μm out of expected range"
        );
    }

    // -------------------------------------------------------------------------
    // test_lateral_resolution
    // -------------------------------------------------------------------------
    #[test]
    fn test_lateral_resolution() {
        let oct = standard_sd_oct();
        let dx = oct.lateral_resolution_um();
        // 0.61 · 0.83 μm / 0.1 = 5.063 μm
        let expected = 0.61 * 0.83 / 0.1;
        assert_relative_eq!(dx, expected, epsilon = 1e-6);
    }

    // -------------------------------------------------------------------------
    // test_max_depth
    // -------------------------------------------------------------------------
    #[test]
    fn test_max_depth() {
        // More pixels → larger max depth
        let oct_small = SdOct::new(830.0, 70.0, 512, 10.0, 0.1, 1.35).unwrap();
        let oct_large = SdOct::new(830.0, 70.0, 2048, 10.0, 0.1, 1.35).unwrap();
        assert!(oct_large.max_depth_um() > oct_small.max_depth_um());
        // Check formula directly
        let lambda0_um = 0.83;
        let d_lambda_um = 0.070 / 1024.0;
        let expected = lambda0_um * lambda0_um / (4.0 * 1.35 * d_lambda_um);
        assert_relative_eq!(standard_sd_oct().max_depth_um(), expected, epsilon = 1e-6);
    }

    // -------------------------------------------------------------------------
    // test_interference_fringe_has_modulation
    // -------------------------------------------------------------------------
    #[test]
    fn test_interference_fringe_has_modulation() {
        let oct = standard_sd_oct();
        let fringe = oct.interference_fringe(100.0, 0.01, 0.9);
        assert_eq!(fringe.len(), oct.n_pixels);
        let min = fringe.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = fringe.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        // The fringe should exhibit modulation (not flat)
        assert!(
            max - min > 1e-6,
            "fringe has no modulation: min={min}, max={max}"
        );
    }

    // -------------------------------------------------------------------------
    // test_a_scan_peak_at_correct_depth
    // -------------------------------------------------------------------------
    #[test]
    fn test_a_scan_peak_at_correct_depth() {
        // Place reflector at 100 μm; peak of A-scan should be near 100 μm
        let oct = SdOct::new(830.0, 70.0, 1024, 10.0, 0.1, 1.35).unwrap();
        let fringe = oct.interference_fringe(100.0, 0.01, 0.9);
        let a_scan = oct.compute_a_scan(&fringe);
        // Find index of maximum
        let (peak_idx, _) = a_scan
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap())
            .expect("a_scan is non-empty");
        let peak_depth = a_scan[peak_idx].0;
        // Allow ±20 μm tolerance (limited by pixel count and windowing)
        assert!(
            (peak_depth - 100.0).abs() < 20.0,
            "peak at {peak_depth} μm, expected ≈ 100 μm"
        );
    }

    // -------------------------------------------------------------------------
    // test_sensitivity_db_positive
    // -------------------------------------------------------------------------
    #[test]
    fn test_sensitivity_db_positive() {
        let oct = standard_sd_oct();
        let s = oct.sensitivity_db(1.0, 0.8); // 1 mW, 80% QE
        assert!(s > 0.0, "sensitivity should be positive dB, got {s}");
    }

    // -------------------------------------------------------------------------
    // test_roll_off_zero_depth
    // -------------------------------------------------------------------------
    #[test]
    fn test_roll_off_zero_depth() {
        let oct = standard_sd_oct();
        let ro = oct.roll_off_db(0.0);
        assert_relative_eq!(ro, 0.0, epsilon = 1e-6);
    }

    // -------------------------------------------------------------------------
    // test_roll_off_increases_with_depth
    // -------------------------------------------------------------------------
    #[test]
    fn test_roll_off_increases_with_depth() {
        let oct = standard_sd_oct();
        let z_max = oct.max_depth_um();
        let ro_near = oct.roll_off_db(z_max * 0.1);
        let ro_far = oct.roll_off_db(z_max * 0.8);
        // Roll-off becomes more negative with depth
        assert!(
            ro_far < ro_near,
            "roll-off should decrease with depth: near={ro_near}, far={ro_far}"
        );
    }

    // -------------------------------------------------------------------------
    // test_coherence_length_td_oct
    // -------------------------------------------------------------------------
    #[test]
    fn test_coherence_length_td_oct() {
        let td = TdOct::new(830.0, 70.0, 0.1, 1.35).unwrap();
        let lc = td.coherence_length_um();
        let expected = (2.0 * 2_f64.ln() / PI) * 0.83_f64.powi(2) / (1.35 * 0.07);
        assert_relative_eq!(lc, expected, epsilon = 1e-6);
    }
}
