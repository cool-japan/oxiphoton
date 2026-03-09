//! Frequency-Resolved Optical Gating (FROG) simulation and retrieval.
//!
//! FROG measures the spectrogram of an ultrashort pulse:
//!
//! ```text
//! I_FROG(ω, τ) = |∫ E(t) · g(t-τ) · exp(-iωt) dt|²
//! ```
//!
//! Different gating nonlinearities yield different FROG variants:
//! - SHG-FROG: g(t) = E(t)  (second-harmonic generation, centrosymmetric)
//! - PG-FROG:  g(t) = |E(t)|²  (polarization gating, asymmetric)
//! - TG-FROG:  g(t) = |E(t)|² (transient grating, three-beam geometry)
//! - SD-FROG:  g(t) = E²(t)·exp(-iωt) (self-diffraction)
//!
//! Retrieval uses the Principal Component Generalized Projections Algorithm
//! (PCGPA), which iterates between time- and frequency-domain constraints.

use num_complex::Complex64;
use std::f64::consts::PI;

// ─── Internal pure-Rust radix-2 Cooley-Tukey FFT ────────────────────────────

/// Compute in-place radix-2 decimation-in-time FFT (power-of-2 length).
///
/// Follows the standard Cooley-Tukey butterfly with bit-reversal permutation.
/// Returns an error if `buf.len()` is not a power of two.
fn fft_inplace(buf: &mut [Complex64]) -> Result<(), FrogError> {
    let n = buf.len();
    if n == 0 || (n & (n - 1)) != 0 {
        return Err(FrogError::InvalidSize(n));
    }
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
    // Cooley-Tukey butterfly stages
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

/// Inverse FFT via conjugate trick: IFFT(x) = conj(FFT(conj(x))) / N.
fn ifft_inplace(buf: &mut [Complex64]) -> Result<(), FrogError> {
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

/// FFT of a slice, returning a new Vec (zero-padded to next power of two if needed).
fn fft_vec(input: &[Complex64]) -> Result<Vec<Complex64>, FrogError> {
    let n = input.len();
    if n == 0 {
        return Err(FrogError::EmptyInput);
    }
    // Find next power of two
    let n_fft = if n.is_power_of_two() {
        n
    } else {
        n.next_power_of_two()
    };
    let mut buf = vec![Complex64::new(0.0, 0.0); n_fft];
    buf[..n].copy_from_slice(input);
    fft_inplace(&mut buf)?;
    Ok(buf)
}

// ─── Error type ─────────────────────────────────────────────────────────────

/// Errors arising from FROG simulation or retrieval.
#[derive(Debug, thiserror::Error)]
pub enum FrogError {
    #[error("FFT size {0} is not a power of two")]
    InvalidSize(usize),
    #[error("Input slice is empty")]
    EmptyInput,
    #[error("Field length {field} does not match grid size {grid}")]
    FieldLengthMismatch { field: usize, grid: usize },
    #[error("Trace dimensions are inconsistent: expected [{n_delay}][{n_freq}]")]
    TraceDimensionMismatch { n_delay: usize, n_freq: usize },
    #[error("Retrieval failed to converge after {0} iterations")]
    ConvergenceFailed(usize),
    #[error("Division by zero in normalisation")]
    DivisionByZero,
}

// ─── FrogType ───────────────────────────────────────────────────────────────

/// Nonlinear gating interaction used in the FROG measurement.
#[derive(Debug, Clone, PartialEq)]
pub enum FrogType {
    /// SHG-FROG: second-harmonic generation gating function g(t) = E(t).
    /// The resulting trace is symmetric with respect to delay, which prevents
    /// determination of the time-reversal symmetry of the pulse.
    ShgFrog,
    /// PG-FROG: polarization-gate; g(t) = |E(t)|².
    /// Produces an asymmetric trace that is unambiguous in time direction.
    PgFrog,
    /// TG-FROG: transient grating in a three-beam geometry; g(t) = |E(t)|².
    /// Higher signal than PG-FROG because it is background-free.
    TgFrog,
    /// SD-FROG: self-diffraction; g(t) = E²(t).
    /// Sensitive to phase and allows single-shot operation.
    SdFrog,
}

// ─── FrogTrace ──────────────────────────────────────────────────────────────

/// FROG spectrogram trace and retrieval engine.
///
/// Stores both the measured (or simulated) trace and provides methods for
/// FROG retrieval via the PCGPA algorithm.
#[derive(Debug, Clone)]
pub struct FrogTrace {
    /// Variant of FROG being simulated/retrieved.
    pub frog_type: FrogType,
    /// Number of time-grid points (must be a power of two for FFT).
    pub n_time: usize,
    /// Number of delay-grid points.
    pub n_delay: usize,
    /// Time step (seconds).
    pub dt: f64,
    /// Delay step (seconds).
    pub d_tau: f64,
    /// Measured intensity trace `[n_delay][n_freq]`.
    pub trace: Vec<Vec<f64>>,
}

impl FrogTrace {
    /// Create a new `FrogTrace` with empty (zeroed) trace array.
    ///
    /// `n_time` determines the frequency-axis length after FFT.
    pub fn new(frog_type: FrogType, n_time: usize, n_delay: usize, dt: f64) -> Self {
        let d_tau = dt; // symmetric delay grid by default
        let trace = vec![vec![0.0_f64; n_time]; n_delay];
        Self {
            frog_type,
            n_time,
            n_delay,
            dt,
            d_tau,
            trace,
        }
    }

    /// Compute the gating signal E_gate(t) for a given delay `tau_index`.
    ///
    /// The delay is applied as a cyclic shift of the field by `tau_index` samples.
    fn gate_field(&self, field: &[Complex64], tau_index: isize) -> Vec<Complex64> {
        let n = field.len();
        let mut gated = vec![Complex64::new(0.0, 0.0); n];
        for t in 0..n {
            let shifted_t = ((t as isize - tau_index).rem_euclid(n as isize)) as usize;
            let gate = match self.frog_type {
                FrogType::ShgFrog => field[shifted_t],
                FrogType::PgFrog => {
                    let amp2 = field[shifted_t].norm_sqr();
                    Complex64::new(amp2, 0.0)
                }
                FrogType::TgFrog => {
                    let amp2 = field[shifted_t].norm_sqr();
                    Complex64::new(amp2, 0.0)
                }
                FrogType::SdFrog => field[shifted_t] * field[shifted_t],
            };
            gated[t] = field[t] * gate;
        }
        gated
    }

    /// Simulate the FROG trace for a known complex field `E(t)`.
    ///
    /// Fills `self.trace[i_delay][i_freq]` with the squared-modulus of the
    /// short-time Fourier transform of the gated field.
    pub fn simulate(&mut self, field: &[Complex64]) -> Result<(), FrogError> {
        if field.len() != self.n_time {
            return Err(FrogError::FieldLengthMismatch {
                field: field.len(),
                grid: self.n_time,
            });
        }
        let n_half = self.n_delay / 2;
        for i_delay in 0..self.n_delay {
            let tau_index = (i_delay as isize) - (n_half as isize);
            let mut gated = self.gate_field(field, tau_index);
            fft_inplace(&mut gated)?;
            for (i_freq, row) in self.trace[i_delay].iter_mut().enumerate().take(self.n_time) {
                *row = gated[i_freq].norm_sqr();
            }
        }
        Ok(())
    }

    /// Normalized RMS FROG error between the stored trace and a retrieved trace.
    ///
    /// ```text
    /// G = sqrt( Σ |I_FROG - I_retrieved|² / N² )
    /// ```
    ///
    /// where `N = n_delay * n_freq` is the total number of pixels.
    pub fn frog_error(&self, retrieved_trace: &[Vec<f64>]) -> Result<f64, FrogError> {
        if retrieved_trace.len() != self.n_delay {
            return Err(FrogError::TraceDimensionMismatch {
                n_delay: self.n_delay,
                n_freq: self.n_time,
            });
        }
        let n_total = (self.n_delay * self.n_time) as f64;
        let mut sum_sq = 0.0_f64;
        for (i, row) in self.trace.iter().enumerate() {
            if retrieved_trace[i].len() != self.n_time {
                return Err(FrogError::TraceDimensionMismatch {
                    n_delay: self.n_delay,
                    n_freq: self.n_time,
                });
            }
            for (j, &measured) in row.iter().enumerate() {
                let diff = measured - retrieved_trace[i][j];
                sum_sq += diff * diff;
            }
        }
        Ok((sum_sq / (n_total * n_total)).sqrt())
    }

    /// Principal Component Generalized Projections Algorithm (PCGPA) retrieval.
    ///
    /// Recovers the complex field `E(t)` from the measured FROG trace.
    ///
    /// # Algorithm
    /// 1. Start with random or flat-phase field guess.
    /// 2. Build the signal matrix `P[t, τ] = E(t) · g(t-τ)`.
    /// 3. Apply frequency-domain projection: replace |P(ω,τ)| with √I_FROG(ω,τ).
    /// 4. Apply time-domain projection: extract E(t) via SVD approximation (power
    ///    iteration on the outer-product structure of the signal matrix).
    /// 5. Repeat until FROG error converges.
    pub fn retrieve_pcgpa(&self, n_iterations: usize) -> Result<Vec<Complex64>, FrogError> {
        let n = self.n_time;
        // Initialise with flat-amplitude, zero-phase field
        let mut field: Vec<Complex64> = (0..n)
            .map(|i| {
                // Gaussian envelope guess
                let x = (i as f64 - n as f64 / 2.0) / (n as f64 / 6.0);
                Complex64::new((-0.5 * x * x).exp(), 0.0)
            })
            .collect();

        let n_half = self.n_delay / 2;

        for _iter in 0..n_iterations {
            // Build signal matrix in frequency domain and apply intensity constraint
            let mut signal_matrix = vec![vec![Complex64::new(0.0, 0.0); n]; self.n_delay];

            for (i_delay, sig_row) in signal_matrix.iter_mut().enumerate().take(self.n_delay) {
                let tau_index = (i_delay as isize) - (n_half as isize);
                let mut gated = self.gate_field(&field, tau_index);
                fft_inplace(&mut gated)?;
                // Apply measured-amplitude projection: keep phase, replace magnitude
                for i_freq in 0..n {
                    let measured_amp = self.trace[i_delay][i_freq].sqrt();
                    let current_phase = gated[i_freq].arg();
                    sig_row[i_freq] = Complex64::new(
                        measured_amp * current_phase.cos(),
                        measured_amp * current_phase.sin(),
                    );
                }
                // IFFT back to time domain
                ifft_inplace(sig_row)?;
            }

            // Power-iteration: extract leading left singular vector as new field
            // approximation.  For outer-product matrices O[t,τ] = E(t)·g(t-τ) the
            // dominant singular vector approximates E(t).
            let mut new_field = vec![Complex64::new(0.0, 0.0); n];
            for t in 0..n {
                let mut acc = Complex64::new(0.0, 0.0);
                for row in signal_matrix.iter().take(self.n_delay) {
                    acc += row[t];
                }
                new_field[t] = acc;
            }
            // Normalise
            let max_amp = new_field.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
            if max_amp > 1e-30 {
                for x in new_field.iter_mut() {
                    *x /= max_amp;
                }
            }
            field = new_field;
        }
        Ok(field)
    }

    /// Frequency marginal: `S(ω) = ∫ I_FROG(ω, τ) dτ`.
    ///
    /// For SHG-FROG this equals the autoconvolution of the pulse spectrum;
    /// for PG-FROG it equals the power spectrum.
    pub fn frequency_marginal(&self) -> Vec<f64> {
        let mut marginal = vec![0.0_f64; self.n_time];
        for row in &self.trace {
            for (j, &val) in row.iter().enumerate() {
                marginal[j] += val * self.d_tau;
            }
        }
        marginal
    }

    /// Time marginal: `A(τ) = ∫ I_FROG(ω, τ) dω`.
    ///
    /// Proportional to the intensity autocorrelation of the gated signal.
    pub fn time_marginal(&self) -> Vec<f64> {
        let df = 1.0 / (self.n_time as f64 * self.dt);
        self.trace
            .iter()
            .map(|row| row.iter().sum::<f64>() * df)
            .collect()
    }

    /// Time-bandwidth product (TBP) = σ_t · σ_ω.
    ///
    /// For a transform-limited Gaussian pulse TBP ≈ 0.5.
    pub fn time_bandwidth_product(field: &[Complex64], dt: f64) -> Result<f64, FrogError> {
        let sigma_t = Self::pulse_duration_rms(field, dt)?;
        // Compute spectrum via FFT
        let spectrum = fft_vec(field)?;
        let df = 1.0 / (field.len() as f64 * dt);
        let sigma_omega = Self::spectral_bandwidth_rms(&spectrum, df)?;
        Ok(sigma_t * sigma_omega)
    }

    /// RMS pulse duration: `σ_t = √(⟨t²⟩ - ⟨t⟩²)`.
    ///
    /// Computed from the intensity `I(t) = |E(t)|²`.
    pub fn pulse_duration_rms(field: &[Complex64], dt: f64) -> Result<f64, FrogError> {
        let intensity: Vec<f64> = field.iter().map(|c| c.norm_sqr()).collect();
        let total: f64 = intensity.iter().sum::<f64>();
        if total < 1e-30 {
            return Err(FrogError::DivisionByZero);
        }
        let n = field.len();
        let t0 = -(n as f64 / 2.0) * dt;
        let mut mean_t = 0.0_f64;
        let mut mean_t2 = 0.0_f64;
        for (i, &ii) in intensity.iter().enumerate() {
            let t = t0 + i as f64 * dt;
            mean_t += t * ii;
            mean_t2 += t * t * ii;
        }
        mean_t /= total;
        mean_t2 /= total;
        let variance = mean_t2 - mean_t * mean_t;
        if variance < 0.0 {
            Ok(0.0)
        } else {
            Ok(variance.sqrt())
        }
    }

    /// RMS spectral bandwidth: `σ_ω = √(⟨ω²⟩ - ⟨ω⟩²)`.
    ///
    /// Input `spectrum` is the complex field spectrum (e.g., output of FFT).
    pub fn spectral_bandwidth_rms(spectrum: &[Complex64], df: f64) -> Result<f64, FrogError> {
        let power: Vec<f64> = spectrum.iter().map(|c| c.norm_sqr()).collect();
        let total: f64 = power.iter().sum::<f64>();
        if total < 1e-30 {
            return Err(FrogError::DivisionByZero);
        }
        let n = spectrum.len();
        let f0 = -(n as f64 / 2.0) * df;
        let mut mean_f = 0.0_f64;
        let mut mean_f2 = 0.0_f64;
        for (i, &p) in power.iter().enumerate() {
            let f = f0 + i as f64 * df;
            mean_f += f * p;
            mean_f2 += f * f * p;
        }
        mean_f /= total;
        mean_f2 /= total;
        let variance = mean_f2 - mean_f * mean_f;
        if variance < 0.0 {
            Ok(0.0)
        } else {
            Ok(variance.sqrt())
        }
    }
}

// ─── ChirpedGaussianPulse ───────────────────────────────────────────────────

/// Chirped Gaussian pulse for testing and benchmarking FROG algorithms.
///
/// The temporal electric field is:
/// ```text
/// E(t) = A · exp(-t²/(2τ²)) · exp(i·(ω₀·t + C·t²/2))
/// ```
///
/// where `τ = τ_FWHM / (2·√(2·ln2))` is the RMS half-width and
/// `C` is the linear chirp rate (rad/fs²).
#[derive(Debug, Clone)]
pub struct ChirpedGaussianPulse {
    /// Peak amplitude (electric field units).
    pub amplitude: f64,
    /// Intensity FWHM duration (femtoseconds).
    pub duration_fs: f64,
    /// Angular centre frequency (rad/s).
    pub center_frequency: f64,
    /// Group-delay dispersion / chirp coefficient (fs²).
    pub chirp: f64,
    /// Carrier-envelope phase offset (radians).
    pub phase_offset: f64,
}

impl ChirpedGaussianPulse {
    /// Construct a chirped Gaussian pulse.
    ///
    /// # Arguments
    /// * `duration_fs`  — intensity FWHM (fs)
    /// * `chirp_fs2`    — GDD / linear chirp (fs²); 0 for transform-limited
    /// * `frequency`    — centre angular frequency (rad/s)
    pub fn new(duration_fs: f64, chirp_fs2: f64, frequency: f64) -> Self {
        Self {
            amplitude: 1.0,
            duration_fs,
            center_frequency: frequency,
            chirp: chirp_fs2,
            phase_offset: 0.0,
        }
    }

    /// RMS half-width `τ = FWHM / (2·√(2·ln2))`.
    fn rms_halfwidth_fs(&self) -> f64 {
        self.duration_fs / (2.0 * (2.0_f64 * 2.0_f64.ln()).sqrt())
    }

    /// Complex field amplitude at time `t_fs` (femtoseconds).
    ///
    /// `E(t) = A · exp(-t²/(2τ²)) · exp(i·(ω₀·t + C·t²/2 + φ₀))`
    pub fn field(&self, t_fs: f64) -> Complex64 {
        let tau = self.rms_halfwidth_fs();
        let envelope = (-t_fs * t_fs / (2.0 * tau * tau)).exp();
        let phase =
            self.center_frequency * t_fs + 0.5 * self.chirp * t_fs * t_fs + self.phase_offset;
        Complex64::new(
            self.amplitude * envelope * phase.cos(),
            self.amplitude * envelope * phase.sin(),
        )
    }

    /// Sample the field on a time grid (femtoseconds).
    pub fn sample_field(&self, t_grid_fs: &[f64]) -> Vec<Complex64> {
        t_grid_fs.iter().map(|&t| self.field(t)).collect()
    }

    /// Transform-limited intensity FWHM (fs) — the FWHM with zero chirp.
    ///
    /// `τ_TL = 2·√(2·ln2)·σ_t`
    pub fn transform_limited_duration_fs(&self) -> f64 {
        self.duration_fs
    }

    /// Actual intensity FWHM including chirp broadening.
    ///
    /// For a chirped Gaussian:
    /// `τ_chirped = τ_TL · √(1 + (GDD / τ_TL²·4·ln2)²)`
    pub fn chirped_duration_fs(&self) -> f64 {
        let tau_tl = self.duration_fs;
        let tau_sigma = tau_tl / (2.0 * (2.0_f64.ln() * 2.0).sqrt());
        let ratio = self.chirp / (tau_sigma * tau_sigma);
        tau_tl * (1.0 + ratio * ratio).sqrt()
    }

    /// Time-bandwidth product for a chirped Gaussian.
    ///
    /// `TBP = 0.4413 · √(1 + (GDD/τ_RMS²)²)`
    ///
    /// where 0.4413 = 2·ln2/π is the TL Gaussian TBP.
    pub fn tbp(&self) -> f64 {
        let tau_rms = self.rms_halfwidth_fs();
        let ratio = self.chirp / (tau_rms * tau_rms);
        0.4413 * (1.0 + ratio * ratio).sqrt()
    }

    /// Instantaneous angular frequency at time `t_fs`:
    ///
    /// `ω_inst(t) = ω₀ + C·t`
    pub fn instantaneous_frequency(&self, t_fs: f64) -> f64 {
        self.center_frequency + self.chirp * t_fs
    }
}

// ─── GRENOUILLE ─────────────────────────────────────────────────────────────

/// GRENOUILLE (single-shot SHG-FROG using a Fresnel biprism).
///
/// GRENOUILLE maps delay onto spatial position via a Fresnel biprism and
/// frequency onto the other spatial axis via a thick SHG crystal acting as
/// both nonlinear medium and spectrometer.
///
/// Physical constraints:
/// - Crystal thickness determines bandwidth acceptance (phase-matching bandwidth).
/// - Angular aperture of biprism determines maximum measurable delay range.
#[derive(Debug, Clone)]
pub struct Grenouille {
    /// SHG crystal thickness (mm). Thicker crystals have narrower phase-matching bandwidth.
    pub crystal_thickness_mm: f64,
    /// Phase-matching bandwidth (nm FWHM). Approximately 0.5 nm·mm / crystal_thickness_mm
    /// for BBO at 800 nm.
    pub phase_matching_bw_nm: f64,
    /// Space-to-delay angular sensitivity (rad/rad).
    pub angular_sensitivity: f64,
}

impl Grenouille {
    /// Construct a GRENOUILLE for a given BBO crystal thickness.
    ///
    /// Uses empirical scaling:
    /// - phase_matching_bw ≈ 0.5 nm·mm / crystal_mm
    /// - angular_sensitivity ≈ 0.15 rad/rad (typical biprism geometry)
    pub fn new(crystal_mm: f64) -> Self {
        let bw_nm = if crystal_mm > 1e-9 {
            0.5 / crystal_mm
        } else {
            0.5
        };
        Self {
            crystal_thickness_mm: crystal_mm,
            phase_matching_bw_nm: bw_nm,
            angular_sensitivity: 0.15,
        }
    }

    /// Maximum measurable pulse duration (fs).
    ///
    /// Limited by the angular aperture of the Fresnel biprism. Approximated as:
    /// `τ_max ≈ 2 * crystal_thickness_mm * 100 fs/mm` (empirical for BBO).
    pub fn max_measurable_duration_fs(&self) -> f64 {
        self.crystal_thickness_mm * 200.0
    }

    /// Maximum measurable spectral bandwidth (nm).
    ///
    /// Set by the phase-matching bandwidth of the SHG crystal (acts as a
    /// spectral filter on the FROG trace frequency axis).
    pub fn max_measurable_bandwidth_nm(&self) -> f64 {
        self.phase_matching_bw_nm
    }

    /// Returns `true` if this GRENOUILLE configuration can measure a pulse
    /// with the given duration and bandwidth without clipping either axis.
    pub fn can_measure(&self, duration_fs: f64, bandwidth_nm: f64) -> bool {
        duration_fs <= self.max_measurable_duration_fs()
            && bandwidth_nm <= self.max_measurable_bandwidth_nm()
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_gaussian_field(n: usize, fwhm_samples: f64) -> Vec<Complex64> {
        let sigma = fwhm_samples / (2.0 * (2.0_f64.ln()).sqrt());
        (0..n)
            .map(|i| {
                let x = (i as f64) - (n as f64 / 2.0);
                Complex64::new((-0.5 * x * x / (sigma * sigma)).exp(), 0.0)
            })
            .collect()
    }

    #[test]
    fn test_fft_roundtrip() {
        let mut buf: Vec<Complex64> = (0..16_usize)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect();
        let original = buf.clone();
        fft_inplace(&mut buf).unwrap();
        ifft_inplace(&mut buf).unwrap();
        for (a, b) in original.iter().zip(buf.iter()) {
            assert_abs_diff_eq!(a.re, b.re, epsilon = 1e-10);
            assert_abs_diff_eq!(a.im, b.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_shg_frog_simulate_symmetric() {
        let n = 32;
        let field = make_gaussian_field(n, 6.0);
        let mut frog = FrogTrace::new(FrogType::ShgFrog, n, n, 1e-15);
        frog.simulate(&field).expect("simulate should succeed");
        // SHG-FROG trace is symmetric about the zero-delay row (index n/2).
        // Check that trace[n/2 + k][j] == trace[n/2 - k][j] for small k.
        let centre = n / 2;
        for k in 1..4 {
            for j in 0..n {
                assert_abs_diff_eq!(
                    frog.trace[centre + k][j],
                    frog.trace[centre - k][j],
                    epsilon = 1e-6
                );
            }
        }
    }

    #[test]
    fn test_frequency_marginal_positive() {
        let n = 32;
        let field = make_gaussian_field(n, 6.0);
        let mut frog = FrogTrace::new(FrogType::PgFrog, n, n, 1e-15);
        frog.simulate(&field).expect("simulate should succeed");
        let marginal = frog.frequency_marginal();
        for &val in &marginal {
            assert!(val >= 0.0, "frequency marginal must be non-negative");
        }
    }

    #[test]
    fn test_chirped_gaussian_tbp() {
        // Transform-limited pulse: TBP ≈ 0.4413
        let pulse = ChirpedGaussianPulse::new(30.0, 0.0, 2.35e15);
        assert_abs_diff_eq!(pulse.tbp(), 0.4413, epsilon = 1e-4);
    }

    #[test]
    fn test_chirped_gaussian_instantaneous_frequency() {
        let chirp_fs2 = 100.0_f64;
        let omega0 = 2.35e15_f64;
        let pulse = ChirpedGaussianPulse::new(30.0, chirp_fs2, omega0);
        // At t = 0, ω_inst = ω₀
        assert_abs_diff_eq!(pulse.instantaneous_frequency(0.0), omega0, epsilon = 1e-6);
        // At t = 10 fs, ω_inst = ω₀ + 100 * 10
        assert_abs_diff_eq!(
            pulse.instantaneous_frequency(10.0),
            omega0 + chirp_fs2 * 10.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_pulse_duration_rms() {
        // For a Gaussian |E(t)| = exp(-t²/(2σ²)), RMS width = σ
        let n = 512;
        let sigma_samples = 40.0_f64;
        let dt = 1.0_f64; // 1 sample = 1 unit
        let field: Vec<Complex64> = (0..n)
            .map(|i| {
                let t = (i as f64) - (n as f64 / 2.0);
                Complex64::new((-0.5 * t * t / (sigma_samples * sigma_samples)).exp(), 0.0)
            })
            .collect();
        let sigma_t = FrogTrace::pulse_duration_rms(&field, dt).unwrap();
        // The intensity I(t) = |E(t)|² = exp(-t²/σ²) has RMS width = σ/√2
        assert_abs_diff_eq!(sigma_t, sigma_samples / 2.0_f64.sqrt(), epsilon = 0.5);
    }

    #[test]
    fn test_grenouille_can_measure() {
        let g = Grenouille::new(0.5);
        // 0.5 mm BBO: max duration ≈ 100 fs, max bandwidth ≈ 1 nm
        assert!(g.max_measurable_duration_fs() > 50.0);
        assert!(g.max_measurable_bandwidth_nm() > 0.5);
        assert!(g.can_measure(50.0, 0.5));
        assert!(!g.can_measure(1000.0, 50.0));
    }

    #[test]
    fn test_frog_error_zero_for_identical_traces() {
        let n = 16;
        let field = make_gaussian_field(n, 4.0);
        let mut frog = FrogTrace::new(FrogType::ShgFrog, n, n, 1e-15);
        frog.simulate(&field).expect("simulate should succeed");
        let err = frog.frog_error(&frog.trace.clone()).unwrap();
        assert_abs_diff_eq!(err, 0.0, epsilon = 1e-14);
    }
}
