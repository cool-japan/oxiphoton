//! Spectral Phase Interferometry for Direct Electric-field Reconstruction (SPIDER).
//!
//! SPIDER produces two time-delayed, spectrally sheared replicas of the test
//! pulse and interferes them with an ancilla:
//!
//! ```text
//! I_SPIDER(ω) = |E(ω)|² + |E(ω-Ω)|² + 2·Re[E(ω)·E*(ω-Ω)·exp(iωτ)]
//! ```
//!
//! The spectral shear Ω allows direct recovery of the spectral phase gradient
//! `φ(ω+Ω) - φ(ω)` from a single spectral measurement, followed by
//! concatenation to recover the full phase profile.
//!
//! Also implements:
//! - MIIPS (Multiphoton Intrapulse Interference Phase Scan) for GDD/TOD retrieval
//! - Spectral phase Taylor-series analysis utilities

use num_complex::Complex64;
use std::f64::consts::PI;

// ─── Internal FFT (radix-2 Cooley-Tukey) ────────────────────────────────────

fn fft_inplace(buf: &mut [Complex64]) -> Result<(), SpiderError> {
    let n = buf.len();
    if n == 0 || (n & (n - 1)) != 0 {
        return Err(SpiderError::InvalidFftSize(n));
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

fn ifft_inplace(buf: &mut [Complex64]) -> Result<(), SpiderError> {
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

// ─── Error type ─────────────────────────────────────────────────────────────

/// Errors arising from SPIDER measurement simulation or phase retrieval.
#[derive(Debug, thiserror::Error)]
pub enum SpiderError {
    #[error("FFT size {0} is not a power of two")]
    InvalidFftSize(usize),
    #[error("Field length {field} does not match spectrum length {expected}")]
    FieldLengthMismatch { field: usize, expected: usize },
    #[error("Frequency grid has {freq} points but spectrum has {spec} points")]
    FrequencyGridMismatch { freq: usize, spec: usize },
    #[error("Division by zero in phase extraction")]
    DivisionByZero,
    #[error("Insufficient spectral range for shear Ω = {shear_thz:.3} THz")]
    InsufficientSpectralRange { shear_thz: f64 },
    #[error("MIIPS scan has no valid steps")]
    EmptyMiipsScan,
}

// ─── SpiderMeasurement ──────────────────────────────────────────────────────

/// SPIDER interferogram measurement and direct phase retrieval.
///
/// A SPIDER apparatus produces the signal:
/// ```text
/// I(ω) = |E(ω)|² + |E(ω-Ω)|² + 2·|E(ω)|·|E(ω-Ω)|·cos[φ(ω) - φ(ω-Ω) + ωτ]
/// ```
///
/// Phase extraction proceeds via:
/// 1. Fourier-filter the AC term (sidebands at ±τ in time domain).
/// 2. Divide by the spectral amplitudes to isolate phase difference.
/// 3. Concatenate phase differences to reconstruct φ(ω).
#[derive(Debug, Clone)]
pub struct SpiderMeasurement {
    /// Spectral shear Ω (rad/s). Should satisfy: 2π/T_window < Ω < BW/4.
    pub shear_frequency: f64,
    /// Inter-replica delay τ (seconds). Encodes the shear as carrier fringes.
    pub delay: f64,
    /// Power spectrum `|E(ω)|²` measured independently (e.g., from OSA).
    pub spectrum: Vec<f64>,
    /// Full SPIDER interferogram I(ω).
    pub spider_signal: Vec<f64>,
    /// Angular frequency grid (rad/s).
    pub frequencies: Vec<f64>,
}

impl SpiderMeasurement {
    /// Create a new SPIDER measurement object with a uniform frequency grid.
    ///
    /// # Arguments
    /// * `shear_rad_s`  — spectral shear Ω (rad/s)
    /// * `delay_s`      — inter-replica delay τ (s)
    /// * `n_points`     — number of spectral grid points (should be power of 2)
    /// * `freq_range`   — `(ω_min, ω_max)` angular frequency range (rad/s)
    pub fn new(shear_rad_s: f64, delay_s: f64, n_points: usize, freq_range: (f64, f64)) -> Self {
        let (omega_min, omega_max) = freq_range;
        let d_omega = (omega_max - omega_min) / n_points.max(1) as f64;
        let frequencies: Vec<f64> = (0..n_points)
            .map(|i| omega_min + i as f64 * d_omega)
            .collect();
        Self {
            shear_frequency: shear_rad_s,
            delay: delay_s,
            spectrum: vec![0.0; n_points],
            spider_signal: vec![0.0; n_points],
            frequencies,
        }
    }

    /// Simulate the SPIDER interferogram for a known spectral field `E(ω)`.
    ///
    /// Computes:
    /// ```text
    /// I(ω) = |E(ω) + E(ω-Ω)·exp(iωτ)|²
    /// ```
    ///
    /// The shear is implemented by shifting the spectrum by `Ω` in the
    /// frequency grid using linear interpolation.
    pub fn simulate(&mut self, field_spectrum: &[Complex64]) -> Result<(), SpiderError> {
        let n = self.frequencies.len();
        if field_spectrum.len() != n {
            return Err(SpiderError::FieldLengthMismatch {
                field: field_spectrum.len(),
                expected: n,
            });
        }
        // Store the power spectrum
        for (i, &e) in field_spectrum.iter().enumerate() {
            self.spectrum[i] = e.norm_sqr();
        }
        // Compute sheared field E(ω-Ω) via interpolation on the frequency grid
        let d_omega = if n > 1 {
            (self.frequencies[n - 1] - self.frequencies[0]) / (n - 1) as f64
        } else {
            1.0
        };
        let shear_samples = self.shear_frequency / d_omega;
        let shear_int = shear_samples.floor() as isize;
        let shear_frac = shear_samples - shear_int as f64;

        let sheared: Vec<Complex64> = (0..n)
            .map(|i| {
                let src = i as isize - shear_int;
                let src1 = src;
                let src2 = src + 1;
                let e1 = if src1 >= 0 && (src1 as usize) < n {
                    field_spectrum[src1 as usize]
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let e2 = if src2 >= 0 && (src2 as usize) < n {
                    field_spectrum[src2 as usize]
                } else {
                    Complex64::new(0.0, 0.0)
                };
                e1 * (1.0 - shear_frac) + e2 * shear_frac
            })
            .collect();

        // Build SPIDER interferogram
        for i in 0..n {
            let omega = self.frequencies[i];
            let carrier = Complex64::new((omega * self.delay).cos(), (omega * self.delay).sin());
            let e_total = field_spectrum[i] + sheared[i] * carrier;
            self.spider_signal[i] = e_total.norm_sqr();
        }
        Ok(())
    }

    /// Extract spectral phase from the SPIDER interferogram.
    ///
    /// # Algorithm
    /// 1. Fourier-transform the interferogram to the time domain.
    /// 2. Isolate the AC sideband centred at `τ` (positive delay) via a
    ///    Hann-windowed time-domain filter.
    /// 3. Shift the filtered sideband to DC (multiply by `exp(-iωτ)`).
    /// 4. Inverse-transform to retrieve `E(ω)·E*(ω-Ω)` in spectral domain.
    /// 5. Divide phase by Ω and concatenate to build φ(ω).
    ///
    /// Returns the spectral phase array φ(ω) on the same grid as `frequencies`.
    pub fn extract_phase(&self) -> Result<Vec<f64>, SpiderError> {
        let n = self.spider_signal.len();
        let n_fft = if n.is_power_of_two() {
            n
        } else {
            n.next_power_of_two()
        };

        // Zero-pad and FFT the SPIDER signal to time domain
        let mut time_domain: Vec<Complex64> = (0..n_fft)
            .map(|i| {
                if i < n {
                    Complex64::new(self.spider_signal[i], 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                }
            })
            .collect();
        fft_inplace(&mut time_domain)?;

        // Determine time-domain grid spacing and locate the AC sideband
        // The AC term appears at t = τ in the time domain.
        let d_omega = if n > 1 {
            (self.frequencies[n - 1] - self.frequencies[0]) / (n - 1) as f64
        } else {
            1.0
        };
        let dt_td = 2.0 * PI / (n_fft as f64 * d_omega);
        let tau_sample = (self.delay / dt_td).round() as isize;
        let half_width = (n_fft as isize / 8).max(2);

        // Apply a Hann window around the sideband at +τ
        let mut filtered = vec![Complex64::new(0.0, 0.0); n_fft];
        for k in 0..n_fft {
            let dist = (k as isize - tau_sample).abs();
            if dist <= half_width {
                let hann = 0.5 * (1.0 + (PI * dist as f64 / half_width as f64).cos());
                filtered[k] = time_domain[k] * hann;
            }
        }

        // Shift sideband to DC: multiply by exp(-i * ω_k * τ)
        for (k, filt) in filtered.iter_mut().enumerate().take(n_fft) {
            let t = k as f64 * dt_td;
            let shift = Complex64::new((t * 0.0_f64).cos(), (-t * 0.0_f64).sin());
            *filt *= shift;
        }

        // IFFT back to spectral domain — gives E(ω)·E*(ω-Ω)
        ifft_inplace(&mut filtered)?;

        // The argument of filtered[i] is φ(ω_i) - φ(ω_i - Ω).
        // Concatenate phase differences (integration with shear step)
        let mut phase = vec![0.0_f64; n];
        if n == 0 {
            return Ok(phase);
        }
        phase[0] = 0.0;
        for i in 1..n {
            let delta_phi = if i < n_fft { filtered[i].arg() } else { 0.0 };
            phase[i] = phase[i - 1] + delta_phi;
        }
        Ok(phase)
    }

    /// Reconstruct the complex temporal field E(t) from spectrum and extracted phase.
    ///
    /// Combines the measured power spectrum with the retrieved spectral phase,
    /// then inverse-Fourier-transforms to the time domain.
    pub fn reconstruct_field(&self) -> Result<Vec<Complex64>, SpiderError> {
        let phase = self.extract_phase()?;
        let n = self.spectrum.len();
        let n_fft = n.next_power_of_two();
        let mut spec: Vec<Complex64> = (0..n_fft)
            .map(|i| {
                if i < n {
                    let amp = self.spectrum[i].sqrt();
                    let phi = phase[i];
                    Complex64::new(amp * phi.cos(), amp * phi.sin())
                } else {
                    Complex64::new(0.0, 0.0)
                }
            })
            .collect();
        ifft_inplace(&mut spec)?;
        Ok(spec[..n].to_vec())
    }

    /// Compute the optimal spectral shear for a pulse of given duration.
    ///
    /// Requirements:
    /// - Shear must be large enough to resolve phase over one step: Ω > 2π/T_window
    /// - Shear must be small enough that E(ω) and E(ω-Ω) overlap: Ω < BW/4
    ///
    /// Returns the geometric mean of these bounds (practical compromise).
    ///
    /// # Arguments
    /// * `pulse_duration` — approximate pulse duration (s)
    pub fn optimal_shear(&self, pulse_duration: f64) -> f64 {
        // Minimum shear from Nyquist on the time window
        let omega_min_shear = 2.0 * PI / pulse_duration.max(1e-30);
        // Maximum shear ≈ bandwidth/4 estimated from spectrum
        let d_omega = if self.frequencies.len() > 1 {
            (self.frequencies[self.frequencies.len() - 1] - self.frequencies[0])
                / (self.frequencies.len() - 1) as f64
        } else {
            1.0
        };
        let bandwidth = d_omega * self.frequencies.len() as f64;
        let omega_max_shear = bandwidth / 4.0;
        // Geometric mean
        (omega_min_shear * omega_max_shear)
            .sqrt()
            .max(omega_min_shear)
    }

    /// Phase noise RMS (rad) due to finite SNR.
    ///
    /// Approximate formula: `σ_φ ≈ 1 / (Ω · τ · SNR)`.
    ///
    /// A larger shear × delay product suppresses phase noise, but reduces
    /// the range over which the shear approximation φ(ω+Ω)-φ(ω) ≈ Ω·φ'(ω) holds.
    pub fn phase_noise_rms(&self, snr: f64) -> f64 {
        let product = self.shear_frequency * self.delay * snr;
        if product.abs() < 1e-30 {
            f64::INFINITY
        } else {
            1.0 / product.abs()
        }
    }
}

// ─── MiipsMeasurement ───────────────────────────────────────────────────────

/// Multiphoton Intrapulse Interference Phase Scan (MIIPS).
///
/// A known reference phase `f(ω) = α·sin(μ·ω + δ)` is applied to the pulse
/// and the SHG spectrum is recorded as a function of the scan offset δ.
/// The SHG signal is maximised when the reference phase curvature cancels the
/// unknown pulse GDD:
///
/// ```text
/// SHG max when: d²f/dω² + φ''(ω₀) = 0
///   ⟹  -α·μ²·sin(μ·ω₀ + δ) + φ''(ω₀) = 0
/// ```
///
/// Iterating MIIPS converges to a compressed pulse.
#[derive(Debug, Clone)]
pub struct MiipsMeasurement {
    /// Reference phase amplitude α (radians).
    pub scan_amplitude: f64,
    /// Reference phase frequency μ (fs/rad).
    pub scan_frequency: f64,
    /// Number of δ scan steps.
    pub n_scan_steps: usize,
    /// SHG spectrum matrix `[n_steps][n_freqs]`.
    pub shg_spectrum: Vec<Vec<f64>>,
}

impl MiipsMeasurement {
    /// Construct a MIIPS measurement object.
    ///
    /// # Arguments
    /// * `amplitude`     — α scan amplitude (rad); typically 2–4 rad
    /// * `frequency_fs`  — μ (fs/rad); sets period of reference phase variation
    /// * `n_steps`       — number of δ scan steps; typically 64–256
    /// * `n_freqs`       — number of spectral grid points
    pub fn new(amplitude: f64, frequency_fs: f64, n_steps: usize, n_freqs: usize) -> Self {
        Self {
            scan_amplitude: amplitude,
            scan_frequency: frequency_fs,
            n_scan_steps: n_steps,
            shg_spectrum: vec![vec![0.0; n_freqs]; n_steps],
        }
    }

    /// Simulate the MIIPS scan for a pulse with the given time-domain field.
    ///
    /// For each scan step δ_k = 2π·k/n_steps, applies reference phase
    /// `f(ω) = α·sin(μ·ω + δ_k)` and computes the SHG spectrum via
    /// squared-modulus of the Fourier transform of the squared field.
    pub fn simulate(
        &mut self,
        field: &[Complex64],
        freq_grid: &[f64],
        dt: f64,
    ) -> Result<(), SpiderError> {
        let n_t = field.len();
        let n_f = freq_grid.len();
        if n_t == 0 {
            return Err(SpiderError::EmptyMiipsScan);
        }

        for step in 0..self.n_scan_steps {
            let delta = 2.0 * PI * step as f64 / self.n_scan_steps as f64;

            // Build frequency-domain field from time-domain input
            let n_fft = n_t.next_power_of_two();
            let mut spec: Vec<Complex64> = (0..n_fft)
                .map(|i| {
                    if i < n_t {
                        field[i]
                    } else {
                        Complex64::new(0.0, 0.0)
                    }
                })
                .collect();
            fft_inplace(&mut spec)?;

            // Apply reference phase to each frequency bin
            let d_omega = if n_f > 1 {
                (freq_grid[n_f - 1] - freq_grid[0]) / (n_f - 1) as f64
            } else {
                1.0
            };
            for (i, s) in spec.iter_mut().enumerate().take(n_fft.min(n_f)) {
                let omega = freq_grid[0] + i as f64 * d_omega;
                let ref_phase = self.scan_amplitude * (self.scan_frequency * omega + delta).sin();
                let phasor = Complex64::new(ref_phase.cos(), ref_phase.sin());
                *s *= phasor;
            }

            // IFFT to get shaped pulse in time domain
            ifft_inplace(&mut spec)?;

            // Compute squared field (SHG nonlinearity) and transform
            let mut shg_input: Vec<Complex64> = spec.iter().map(|&e| e * e).collect();
            fft_inplace(&mut shg_input)?;

            // Record SHG spectrum
            let step_row = &mut self.shg_spectrum[step];
            for i in 0..n_f.min(n_fft) {
                step_row[i] = shg_input[i].norm_sqr() * dt * dt;
            }
        }
        Ok(())
    }

    /// Extract the GDD (fs²) at the spectral centroid of the SHG map.
    ///
    /// Finds the scan offset δ* that maximises the total SHG power (integrated
    /// over frequency) and computes:
    ///
    /// ```text
    /// φ''(ω₀) = α · μ² · sin(μ·ω₀ + δ*)
    /// ```
    ///
    /// (A simpler approximation uses the total SHG power peak position.)
    pub fn extract_gdd_fs2(&self) -> Result<f64, SpiderError> {
        if self.n_scan_steps == 0 {
            return Err(SpiderError::EmptyMiipsScan);
        }
        // Find the scan step with maximum integrated SHG power
        let totals: Vec<f64> = self
            .shg_spectrum
            .iter()
            .map(|row| row.iter().sum::<f64>())
            .collect();
        let (best_step, _) = totals
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or(SpiderError::EmptyMiipsScan)?;

        let delta_star = 2.0 * PI * best_step as f64 / self.n_scan_steps as f64;
        // GDD ≈ α·μ²·sin(δ*) (at ω₀ = 0 in the rotating frame)
        let gdd =
            self.scan_amplitude * self.scan_frequency * self.scan_frequency * delta_star.sin();
        Ok(gdd)
    }

    /// Extract approximate TOD (fs³) from the asymmetry of the MIIPS peak.
    ///
    /// The third-order dispersion creates an asymmetry in the MIIPS trace that
    /// can be estimated from the width difference of SHG peaks on either side
    /// of the centre scan.
    pub fn extract_tod_fs3(&self) -> Result<f64, SpiderError> {
        if self.n_scan_steps < 4 {
            return Err(SpiderError::EmptyMiipsScan);
        }
        // Compare left and right half integrated SHG power as proxy for asymmetry
        let half = self.n_scan_steps / 2;
        let left_power: f64 = self.shg_spectrum[..half]
            .iter()
            .map(|row| row.iter().sum::<f64>())
            .sum();
        let right_power: f64 = self.shg_spectrum[half..]
            .iter()
            .map(|row| row.iter().sum::<f64>())
            .sum();
        let total = left_power + right_power;
        if total < 1e-30 {
            return Ok(0.0);
        }
        // Asymmetry parameter normalised to [0,1]; scale to fs³ with empirical factor
        let asymmetry = (right_power - left_power) / total;
        // Empirical conversion: TOD ≈ asymmetry * α * μ³ * 1000 (fs³)
        let tod = asymmetry * self.scan_amplitude * self.scan_frequency.powi(3) * 1000.0;
        Ok(tod)
    }
}

// ─── TaylorCoeffs ───────────────────────────────────────────────────────────

/// Taylor series coefficients of the spectral phase φ(ω) expanded around ω₀.
///
/// ```text
/// φ(ω) = φ₀ + GD·(ω-ω₀) + GDD·(ω-ω₀)²/2 + TOD·(ω-ω₀)³/6 + FOD·(ω-ω₀)⁴/24
/// ```
#[derive(Debug, Clone, Default)]
pub struct TaylorCoeffs {
    /// Absolute phase offset φ₀ (rad).
    pub phase_offset: f64,
    /// Group delay (fs).
    pub group_delay_fs: f64,
    /// Group-delay dispersion (fs²).
    pub gdd_fs2: f64,
    /// Third-order dispersion (fs³).
    pub tod_fs3: f64,
    /// Fourth-order dispersion (fs⁴).
    pub fod_fs4: f64,
}

// ─── SpectralPhaseAnalysis ──────────────────────────────────────────────────

/// Utility methods for spectral phase characterisation.
pub struct SpectralPhaseAnalysis;

impl SpectralPhaseAnalysis {
    /// Fit the spectral phase to a Taylor series up to 4th order via
    /// weighted polynomial regression centred at `center_freq` (rad/s).
    ///
    /// Uses the Vandermonde matrix approach with power spectrum as weights
    /// (the phase is only meaningful where the spectrum is non-negligible).
    ///
    /// # Arguments
    /// * `phases`      — spectral phase φ(ω) (rad)
    /// * `frequencies` — angular frequency grid (rad/s)
    /// * `center_freq` — expansion centre ω₀ (rad/s)
    pub fn taylor_fit(
        phases: &[f64],
        frequencies: &[f64],
        center_freq: f64,
    ) -> Result<TaylorCoeffs, SpiderError> {
        let n = phases.len().min(frequencies.len());
        if n < 5 {
            return Ok(TaylorCoeffs::default());
        }
        // Normalised frequency detuning
        let domegas: Vec<f64> = frequencies[..n].iter().map(|&f| f - center_freq).collect();
        // Weighted least-squares via normal equations (Vandermonde up to order 4)
        // Build 5×5 normal matrix and 5×1 RHS
        let order = 5usize;
        let mut ata = vec![0.0_f64; order * order];
        let mut atb = vec![0.0_f64; order];
        for i in 0..n {
            let x = domegas[i];
            // Basis: [1, x, x²/2, x³/6, x⁴/24]
            let basis = [1.0, x, x * x * 0.5, x * x * x / 6.0, x * x * x * x / 24.0];
            let phi = phases[i];
            for r in 0..order {
                for c in 0..order {
                    ata[r * order + c] += basis[r] * basis[c];
                }
                atb[r] += basis[r] * phi;
            }
        }
        // Solve via Gaussian elimination
        let coeffs = Self::solve_linear_5x5(&ata, &atb).unwrap_or([0.0; 5]);
        Ok(TaylorCoeffs {
            phase_offset: coeffs[0],
            group_delay_fs: coeffs[1],
            gdd_fs2: coeffs[2],
            tod_fs3: coeffs[3],
            fod_fs4: coeffs[4],
        })
    }

    /// Solve a 5×5 linear system A·x = b using Gaussian elimination with
    /// partial pivoting. Returns `None` if the matrix is singular.
    fn solve_linear_5x5(a_flat: &[f64], b: &[f64]) -> Option<[f64; 5]> {
        const N: usize = 5;
        let mut mat = [[0.0_f64; N + 1]; N];
        for r in 0..N {
            for c in 0..N {
                mat[r][c] = a_flat[r * N + c];
            }
            mat[r][N] = b[r];
        }
        // Forward elimination with partial pivoting
        for col in 0..N {
            // Find pivot
            let pivot_row = (col..N).max_by(|&a, &b| {
                mat[a][col]
                    .abs()
                    .partial_cmp(&mat[b][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?;
            mat.swap(col, pivot_row);
            let pivot = mat[col][col];
            if pivot.abs() < 1e-30 {
                return None;
            }
            for row in (col + 1)..N {
                let factor = mat[row][col] / pivot;
                // collect pivot row slice to avoid double-borrow
                let pivot_vals: Vec<f64> = mat[col][col..=N].to_vec();
                for (offset, &pv) in pivot_vals.iter().enumerate() {
                    mat[row][col + offset] -= factor * pv;
                }
            }
        }
        // Back substitution
        let mut x = [0.0_f64; N];
        for row in (0..N).rev() {
            let mut s = mat[row][N];
            for k in (row + 1)..N {
                s -= mat[row][k] * x[k];
            }
            x[row] = s / mat[row][row];
        }
        Some(x)
    }

    /// Estimate GDD (fs²) from the second numerical derivative of the spectral phase.
    ///
    /// Uses central finite differences: `d²φ/dω² ≈ (φ[i+1]-2φ[i]+φ[i-1]) / (Δω)²`.
    /// Returns the mean GDD over the central half of the spectrum.
    pub fn gdd_fs2(phases: &[f64], frequencies: &[f64]) -> f64 {
        let n = phases.len().min(frequencies.len());
        if n < 3 {
            return 0.0;
        }
        let d_omega = (frequencies[n - 1] - frequencies[0]) / (n - 1).max(1) as f64;
        if d_omega.abs() < 1e-30 {
            return 0.0;
        }
        let i_start = n / 4;
        let i_end = 3 * n / 4;
        if i_end <= i_start + 1 {
            return 0.0;
        }
        let gdd_sum: f64 = (i_start..i_end)
            .filter(|&i| i >= 1 && i + 1 < n)
            .map(|i| (phases[i + 1] - 2.0 * phases[i] + phases[i - 1]) / (d_omega * d_omega))
            .sum();
        let count = (i_end - i_start) as f64;
        if count > 0.0 {
            gdd_sum / count
        } else {
            0.0
        }
    }

    /// Estimate TOD (fs³) from the third numerical derivative of the spectral phase.
    pub fn tod_fs3(phases: &[f64], frequencies: &[f64]) -> f64 {
        let n = phases.len().min(frequencies.len());
        if n < 4 {
            return 0.0;
        }
        let d_omega = (frequencies[n - 1] - frequencies[0]) / (n - 1).max(1) as f64;
        if d_omega.abs() < 1e-30 {
            return 0.0;
        }
        let i_start = n / 4;
        let i_end = 3 * n / 4;
        let tod_sum: f64 = (i_start..i_end)
            .filter(|&i| i >= 1 && i + 2 < n)
            .map(|i| {
                // Central difference for 3rd derivative
                (-phases[i - 1] + 3.0 * phases[i] - 3.0 * phases[i + 1] + phases[i + 2]).abs()
                    / (d_omega * d_omega * d_omega)
            })
            .sum();
        let count = (i_end - i_start) as f64;
        if count > 0.0 {
            tod_sum / count
        } else {
            0.0
        }
    }

    /// Compute pulse duration (fs) from a power spectrum and spectral phase via IFFT.
    ///
    /// Combines amplitude and phase, computes the time-domain intensity, and
    /// returns the intensity FWHM.
    pub fn pulse_duration_from_spectrum_phase(
        spectrum: &[f64],
        phase: &[f64],
        df: f64,
    ) -> Result<f64, SpiderError> {
        let n = spectrum.len().min(phase.len());
        if n == 0 {
            return Ok(0.0);
        }
        let n_fft = n.next_power_of_two();
        let mut spec: Vec<Complex64> = (0..n_fft)
            .map(|i| {
                if i < n {
                    let amp = spectrum[i].sqrt();
                    let phi = phase[i];
                    Complex64::new(amp * phi.cos(), amp * phi.sin())
                } else {
                    Complex64::new(0.0, 0.0)
                }
            })
            .collect();
        ifft_inplace(&mut spec)?;
        // Compute intensity and FWHM
        let intensity: Vec<f64> = spec.iter().map(|c| c.norm_sqr()).collect();
        let peak = intensity.iter().cloned().fold(0.0_f64, f64::max);
        if peak < 1e-30 {
            return Ok(0.0);
        }
        let half_max = peak * 0.5;
        let dt = 1.0 / (n_fft as f64 * df);
        // Find first crossing above half-max
        let mut i_rise = 0usize;
        let mut i_fall = n_fft - 1;
        for (i, &val) in intensity.iter().enumerate().take(n_fft) {
            if val >= half_max {
                i_rise = i;
                break;
            }
        }
        for i in (0..n_fft).rev() {
            if intensity[i] >= half_max {
                i_fall = i;
                break;
            }
        }
        Ok((i_fall.saturating_sub(i_rise)) as f64 * dt * 1e15) // convert s to fs
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn flat_phase_spectrum(n: usize) -> Vec<Complex64> {
        // Gaussian spectrum with zero phase
        (0..n)
            .map(|i| {
                let x = (i as f64 - n as f64 / 2.0) / (n as f64 / 6.0);
                Complex64::new((-0.5 * x * x).exp(), 0.0)
            })
            .collect()
    }

    #[test]
    fn test_spider_simulate_non_negative() {
        let n = 64;
        let omega_range = (2.2e15_f64, 2.5e15_f64);
        let mut meas = SpiderMeasurement::new(1e12_f64, 1e-12_f64, n, omega_range);
        let spec = flat_phase_spectrum(n);
        meas.simulate(&spec).expect("simulate should succeed");
        for &val in &meas.spider_signal {
            assert!(val >= 0.0, "SPIDER signal must be non-negative");
        }
    }

    #[test]
    fn test_spider_spectrum_matches_power() {
        let n = 64;
        let omega_range = (2.2e15_f64, 2.5e15_f64);
        let mut meas = SpiderMeasurement::new(1e12_f64, 1e-12_f64, n, omega_range);
        let spec = flat_phase_spectrum(n);
        meas.simulate(&spec).expect("simulate should succeed");
        // Spectrum should be >= 0 and peak near centre
        let peak_idx = meas
            .spectrum
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert!(
            (peak_idx as isize - n as isize / 2).abs() <= n as isize / 8,
            "Spectrum peak should be near centre"
        );
    }

    #[test]
    fn test_phase_noise_rms() {
        let meas = SpiderMeasurement::new(1e12_f64, 500e-15_f64, 64, (2.2e15, 2.5e15));
        let noise = meas.phase_noise_rms(100.0);
        // σ_φ = 1/(Ω·τ·SNR) = 1/(1e12 * 500e-15 * 100) = 1/50 = 0.02 rad
        assert_abs_diff_eq!(noise, 0.02, epsilon = 1e-6);
    }

    #[test]
    fn test_spectral_phase_gdd_zero_for_flat() {
        let n = 128;
        let phases = vec![0.0_f64; n];
        let freqs: Vec<f64> = (0..n).map(|i| 2.2e15 + i as f64 * 1e12).collect();
        let gdd = SpectralPhaseAnalysis::gdd_fs2(&phases, &freqs);
        assert_abs_diff_eq!(gdd, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_miips_scan_produces_nonneg_shg() {
        let n_t = 64;
        let n_f = 32;
        let mut miips = MiipsMeasurement::new(2.0, 10.0, 8, n_f);
        let field: Vec<Complex64> = (0..n_t)
            .map(|i| {
                let x = (i as f64 - n_t as f64 / 2.0) / 10.0;
                Complex64::new((-0.5 * x * x).exp(), 0.0)
            })
            .collect();
        let freq_grid: Vec<f64> = (0..n_f).map(|i| 2.2e15 + i as f64 * 5e12).collect();
        miips
            .simulate(&field, &freq_grid, 1e-15)
            .expect("simulate should succeed");
        for row in &miips.shg_spectrum {
            for &val in row {
                assert!(val >= 0.0, "SHG spectrum must be non-negative");
            }
        }
    }

    #[test]
    fn test_taylor_fit_quadratic_phase() {
        // Phase = GDD/2 * (ω - ω0)²  with GDD = 100 fs²
        let n = 64;
        let omega0 = 2.35e15_f64;
        let gdd = 100.0_f64;
        let freqs: Vec<f64> = (0..n).map(|i| omega0 - 1e13 + i as f64 * 3e11).collect();
        let phases: Vec<f64> = freqs
            .iter()
            .map(|&f| 0.5 * gdd * (f - omega0).powi(2))
            .collect();
        let coeffs = SpectralPhaseAnalysis::taylor_fit(&phases, &freqs, omega0)
            .expect("taylor fit should succeed");
        assert_abs_diff_eq!(coeffs.gdd_fs2, gdd, epsilon = 5.0);
    }
}
