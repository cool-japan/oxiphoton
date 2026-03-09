//! Core DSP algorithms for photonic communications.
//!
//! Includes Q-factor / BER calculations, OSNR conversion utilities, FEC
//! coding models, Gram–Schmidt orthonormalisation, a pure-Rust Cooley–Tukey
//! radix-2 DFT/iDFT implementation, Welch PSD estimation, and eye-diagram
//! statistics.

use num_complex::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Error function approximations
// ---------------------------------------------------------------------------

/// Complementary error function approximation (Abramowitz & Stegun 7.1.26).
///
/// Maximum absolute error < 1.5 × 10⁻⁷ for all x ≥ 0.
pub fn erfc_approx(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc_approx(-x);
    }
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    poly * (-(x * x)).exp()
}

/// Inverse error function via rational approximation + Halley refinement.
///
/// Uses the rational approximation from Peter Acklam's method (2002), then
/// refines with one Halley step for accuracy.
pub fn erfinv_approx(y: f64) -> f64 {
    let y = y.clamp(-1.0 + 1e-15, 1.0 - 1e-15);
    // Remap erf to erfinv: erf(x) = y → erfinv(y) = x
    // Use the rational approximation for the complementary CDF
    let p = (y + 1.0) / 2.0; // p in (0,1)
    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    // Rational approximation (Acklam coefficients)
    let c = [2.515_517, 0.802_853, 0.010_328_f64];
    let d = [1.432_788, 0.189_269, 0.001_308_f64];
    let x0 = t - (c[0] + t * (c[1] + t * c[2])) / (1.0 + t * (d[0] + t * (d[1] + t * d[2])));
    let x0 = if p < 0.5 { -x0 } else { x0 };

    // Convert: erfinv(y) = erfinv(2p-1) = x0 / sqrt(2)
    let mut x = x0 / 2.0_f64.sqrt();

    // Halley refinement step: x ← x − (erf(x)−y) / (2/√π · exp(−x²) · (1 + x·(erf(x)−y)))
    let sqrt_pi_inv = 1.0 / PI.sqrt();
    for _ in 0..3 {
        let erf_x = 1.0 - erfc_approx(x);
        let fx = erf_x - y;
        let fpx = 2.0 * sqrt_pi_inv * (-(x * x)).exp();
        let fppx = -2.0 * x * fpx;
        // Halley step: x ← x − 2·f·f' / (2·f'² − f·f'')
        let denom = 2.0 * fpx * fpx - fx * fppx;
        if denom.abs() < 1e-30 {
            break;
        }
        x -= 2.0 * fx * fpx / denom;
    }
    x
}

// ---------------------------------------------------------------------------
// Q-factor and BER
// ---------------------------------------------------------------------------

/// Convert Q-factor to BER.
///
/// BER = erfc(Q / √2) / 2
pub fn q_factor_to_ber(q: f64) -> f64 {
    erfc_approx(q / 2.0_f64.sqrt()) / 2.0
}

/// Convert BER to Q-factor.
///
/// Q = √2 · erfinv(1 − 2·BER)
pub fn ber_to_q_factor(ber: f64) -> f64 {
    2.0_f64.sqrt() * erfinv_approx(1.0 - 2.0 * ber)
}

// ---------------------------------------------------------------------------
// OSNR utilities
// ---------------------------------------------------------------------------

/// Convert OSNR (dB) to linear per-symbol SNR.
///
/// SNR = OSNR_linear × B_ref / B_sym  (for single polarisation)
/// B_ref = reference noise bandwidth (default: 12.5 GHz = 0.1 nm at 1550 nm).
pub fn osnr_to_snr(osnr_db: f64, baud_rate_gbaud: f64, noise_bw_ghz: f64) -> f64 {
    let osnr = 10.0_f64.powf(osnr_db / 10.0);
    let b_sym = baud_rate_gbaud;
    if b_sym > 0.0 {
        osnr * noise_bw_ghz / b_sym
    } else {
        0.0
    }
}

/// BER from OSNR for DP-QPSK (coherent detection).
///
/// BER ≈ erfc(√(OSNR · B_ref / (2 · B_sym))) / 2
pub fn ber_dp_qpsk_from_osnr(osnr_db: f64, baud_gbaud: f64) -> f64 {
    let osnr = 10.0_f64.powf(osnr_db / 10.0);
    let b_ref = 12.5_f64; // GHz
    let b_sym = baud_gbaud;
    let arg = (osnr * b_ref / (2.0 * b_sym)).sqrt();
    erfc_approx(arg) / 2.0
}

/// Required OSNR (dB) for a target BER with a given modulation format.
///
/// Supported `modulation` strings: `"dp-qpsk"`, `"dp-16qam"`, `"dp-64qam"`.
pub fn required_osnr(target_ber: f64, baud_gbaud: f64, modulation: &str) -> f64 {
    let b_ref = 12.5_f64; // GHz
    match modulation.to_lowercase().as_str() {
        "dp-qpsk" => {
            // target_ber = erfc(√x)/2  →  erfc(√x) = 2·BER
            let erfc_val = 2.0 * target_ber;
            // erfc(x) = erfc_val → x = erfc⁻¹(erfc_val)
            // erfc⁻¹(v) = erfinv(1-v)
            let sqrt_x = erfinv_approx(1.0 - erfc_val).max(0.0);
            let snr_req = sqrt_x * sqrt_x;
            10.0 * (snr_req * 2.0 * baud_gbaud / b_ref).log10()
        }
        "dp-16qam" => {
            // BER ≈ (3/8)·erfc(√x) → erfc(√x) = (8/3)·BER
            let erfc_val = (8.0 / 3.0) * target_ber;
            let sqrt_x = erfinv_approx(1.0 - erfc_val).max(0.0);
            let snr_req = sqrt_x * sqrt_x;
            10.0 * (snr_req * 5.0 * baud_gbaud / b_ref).log10()
        }
        "dp-64qam" => {
            // BER ≈ (7/24)·erfc(√x) where x = SNR/(2*(64-1)) ≈ rough model
            let erfc_val = (24.0 / 7.0) * target_ber;
            let sqrt_x = erfinv_approx(1.0 - erfc_val).max(0.0);
            let snr_req = sqrt_x * sqrt_x;
            10.0 * (snr_req * 7.0 * baud_gbaud / b_ref).log10()
        }
        _ => {
            // Default: use DP-QPSK formula
            let erfc_val = 2.0 * target_ber;
            let sqrt_x = erfinv_approx(1.0 - erfc_val).max(0.0);
            let snr_req = sqrt_x * sqrt_x;
            10.0 * (snr_req * 2.0 * baud_gbaud / b_ref).log10()
        }
    }
}

// ---------------------------------------------------------------------------
// FEC code models
// ---------------------------------------------------------------------------

/// Forward Error Correction code model.
#[derive(Debug, Clone)]
pub struct FecCode {
    /// Code rate k/n (fraction of information bits).
    pub code_rate: f64,
    /// Net coding gain in dB at BER = 10⁻¹⁵.
    pub coding_gain_db: f64,
    /// Pre-FEC BER threshold (input BER that gives BER = 10⁻¹⁵ output).
    pub pre_fec_threshold: f64,
}

impl FecCode {
    /// RS(255,239): 7% overhead, ≈ 6.2 dB net coding gain.
    pub fn reed_solomon_255_239() -> Self {
        Self {
            code_rate: 239.0 / 255.0,
            coding_gain_db: 6.2,
            pre_fec_threshold: 3.8e-3,
        }
    }

    /// Soft-decision FEC with 20% overhead and ≈ 11 dB coding gain.
    pub fn sd_fec_20_percent() -> Self {
        Self {
            code_rate: 1.0 / 1.2,
            coding_gain_db: 11.0,
            pre_fec_threshold: 2.0e-2,
        }
    }

    /// oFEC standard (25% overhead, ≈ 11.5 dB net coding gain).
    pub fn ofec_25_percent() -> Self {
        Self {
            code_rate: 0.8,
            coding_gain_db: 11.5,
            pre_fec_threshold: 2.7e-2,
        }
    }

    /// Net coding gain (same as `coding_gain_db` field for clarity).
    pub fn net_coding_gain_db(&self) -> f64 {
        self.coding_gain_db
    }

    /// Post-FEC BER estimate given a pre-FEC BER.
    ///
    /// Simplified model: if pre_fec_ber < threshold → post ≈ 10⁻¹⁵;
    /// otherwise BER degrades exponentially above threshold.
    pub fn post_fec_ber(&self, pre_fec_ber: f64) -> f64 {
        if pre_fec_ber <= self.pre_fec_threshold {
            1e-15
        } else {
            // Exponential degradation model
            let excess = pre_fec_ber / self.pre_fec_threshold;
            1e-15 * excess.powi(10)
        }
    }

    /// FEC overhead as a percentage: (1/rate − 1) × 100.
    pub fn overhead_percent(&self) -> f64 {
        (1.0 / self.code_rate - 1.0) * 100.0
    }
}

// ---------------------------------------------------------------------------
// Gram–Schmidt orthonormalisation
// ---------------------------------------------------------------------------

/// Gram–Schmidt orthonormalisation of a set of complex vectors.
///
/// Returns an orthonormal basis spanning the same subspace.
pub fn gram_schmidt(vectors: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let mut basis: Vec<Vec<Complex64>> = Vec::new();
    for v in vectors {
        let mut u = v.clone();
        // Project out all existing basis vectors
        for e in &basis {
            let proj_coeff = inner_product(e, &u);
            for (ui, &ei) in u.iter_mut().zip(e.iter()) {
                *ui -= proj_coeff * ei;
            }
        }
        let norm = inner_product(&u, &u).re.sqrt();
        if norm > 1e-14 {
            let normed: Vec<Complex64> = u.iter().map(|&x| x / norm).collect();
            basis.push(normed);
        }
    }
    basis
}

/// Compute the Hermitian inner product ⟨a, b⟩ = Σ a_k* · b_k.
fn inner_product(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    a.iter()
        .zip(b.iter())
        .fold(Complex64::new(0.0, 0.0), |acc, (&ai, &bi)| {
            acc + ai.conj() * bi
        })
}

// ---------------------------------------------------------------------------
// Cooley–Tukey radix-2 DFT / iDFT
// ---------------------------------------------------------------------------

/// Pure-Rust Cooley–Tukey radix-2 FFT (iterative, decimation-in-time).
///
/// Requires that `x.len()` is a power of 2.  If not, the input is
/// zero-padded to the next power of two.
pub fn dft(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    // Pad to power of 2
    let m = n.next_power_of_two();
    let mut a: Vec<Complex64> = x.to_vec();
    a.resize(m, Complex64::new(0.0, 0.0));
    fft_inplace(&mut a, false);
    a
}

/// Pure-Rust inverse DFT (normalised by 1/N).
///
/// Input length is padded to the next power of two if necessary.
pub fn idft(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let m = n.next_power_of_two();
    let mut a: Vec<Complex64> = x.to_vec();
    a.resize(m, Complex64::new(0.0, 0.0));
    fft_inplace(&mut a, true);
    let scale = 1.0 / m as f64;
    for s in &mut a {
        *s *= scale;
    }
    a
}

/// In-place iterative Cooley–Tukey radix-2 FFT.
///
/// `inverse = true` uses twiddle factors exp(+j·2π/N) (IFFT convention).
fn fft_inplace(a: &mut [Complex64], inverse: bool) {
    let n = a.len();
    // Bit-reversal permutation
    let bits = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = bit_reverse(i, bits);
        if i < j {
            a.swap(i, j);
        }
    }
    // Cooley–Tukey butterfly stages
    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    let mut len = 2_usize;
    while len <= n {
        let ang = sign * 2.0 * PI / len as f64;
        let wlen = Complex64::new(ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let mut w = Complex64::new(1.0, 0.0);
            for j in 0..(len / 2) {
                let u = a[i + j];
                let v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Reverse the lowest `bits` bits of `x`.
fn bit_reverse(x: usize, bits: usize) -> usize {
    let mut r = 0_usize;
    let mut v = x;
    for _ in 0..bits {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    r
}

// ---------------------------------------------------------------------------
// Welch PSD estimator
// ---------------------------------------------------------------------------

/// Window function type for the Welch PSD estimator.
#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    /// No windowing (rectangular).
    Rectangular,
    /// Hann (raised cosine) window.
    Hann,
    /// Hamming window.
    Hamming,
    /// Blackman window.
    Blackman,
}

/// Welch method power spectral density estimator.
#[derive(Debug, Clone)]
pub struct WelchPsd {
    /// Segment (FFT) length.
    pub segment_length: usize,
    /// Overlap between successive segments (samples).
    pub overlap: usize,
    /// Window function.
    pub window: WindowType,
}

impl WelchPsd {
    /// Construct a new Welch PSD estimator.
    pub fn new(n: usize, overlap: usize, window: WindowType) -> Self {
        Self {
            segment_length: n,
            overlap,
            window,
        }
    }

    /// Compute the PSD of `signal` sampled at `sample_rate` Hz.
    ///
    /// Returns `(frequencies_hz, psd_linear)` where frequencies are centred
    /// on DC (negative frequencies first after FFT shift).
    pub fn compute(&self, signal: &[Complex64], sample_rate: f64) -> (Vec<f64>, Vec<f64>) {
        let n = self.segment_length;
        let hop = if n > self.overlap {
            n - self.overlap
        } else {
            1
        };
        let win = self.window_coefficients();
        let win_power: f64 = win.iter().map(|&w| w * w).sum::<f64>() / n as f64;

        let mut psd_accum = vec![0.0_f64; n.next_power_of_two()];
        let mut n_segments = 0_usize;

        let mut start = 0;
        while start + n <= signal.len() {
            let segment: Vec<Complex64> = signal[start..start + n]
                .iter()
                .enumerate()
                .map(|(k, &s)| s * win[k])
                .collect();
            let spectrum = dft(&segment);
            let m = spectrum.len();
            for (k, &s) in spectrum.iter().enumerate() {
                psd_accum[k % m] += s.norm_sqr();
            }
            n_segments += 1;
            start += hop;
        }

        if n_segments == 0 {
            return (Vec::new(), Vec::new());
        }

        let m = n.next_power_of_two();
        let scale = 1.0 / (n_segments as f64 * sample_rate * win_power * m as f64);
        let psd: Vec<f64> = psd_accum.iter().map(|&p| p * scale).collect();

        // Frequency axis (shifted so DC is at centre)
        let df = sample_rate / m as f64;
        let freqs: Vec<f64> = (0..m)
            .map(|k| {
                if k <= m / 2 {
                    k as f64 * df
                } else {
                    (k as f64 - m as f64) * df
                }
            })
            .collect();

        (freqs, psd)
    }

    /// Compute the window coefficient vector for the selected window type.
    pub fn window_coefficients(&self) -> Vec<f64> {
        let n = self.segment_length;
        (0..n)
            .map(|k| {
                let x = 2.0 * PI * k as f64 / (n as f64 - 1.0).max(1.0);
                match self.window {
                    WindowType::Rectangular => 1.0,
                    WindowType::Hann => 0.5 * (1.0 - x.cos()),
                    WindowType::Hamming => 0.54 - 0.46 * x.cos(),
                    WindowType::Blackman => 0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos(),
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Eye diagram statistics
// ---------------------------------------------------------------------------

/// Statistical characterisation of an optical eye diagram.
#[derive(Debug, Clone)]
pub struct EyeDiagram {
    /// Normalised eye opening height (0 = closed, 1 = fully open).
    pub eye_opening: f64,
    /// Amplitude at the zero-crossing point.
    pub eye_crossing: f64,
    /// RMS timing jitter in picoseconds.
    pub jitter_rms_ps: f64,
    /// Inter-symbol interference penalty in dB.
    pub isi_db: f64,
}

impl EyeDiagram {
    /// Analyse a sampled signal and extract eye-diagram statistics.
    ///
    /// `signal`: sampled real-valued waveform.
    /// `symbol_rate`: symbol rate in GHz.
    /// `samples_per_symbol`: oversampling ratio.
    pub fn analyze(signal: &[f64], symbol_rate: f64, samples_per_symbol: usize) -> Self {
        if signal.is_empty() || samples_per_symbol == 0 {
            return Self {
                eye_opening: 0.0,
                eye_crossing: 0.0,
                jitter_rms_ps: 0.0,
                isi_db: 0.0,
            };
        }

        let sps = samples_per_symbol;
        let n_sym = signal.len() / sps;

        // Sample at the optimal sampling instant (centre of symbol)
        let centre = sps / 2;
        let mut samples_at_centre: Vec<f64> = (0..n_sym)
            .filter_map(|i| signal.get(i * sps + centre).copied())
            .collect();
        samples_at_centre.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n_s = samples_at_centre.len();
        if n_s == 0 {
            return Self {
                eye_opening: 0.0,
                eye_crossing: 0.0,
                jitter_rms_ps: 0.0,
                isi_db: 0.0,
            };
        }

        let max_val = samples_at_centre[n_s - 1];
        let min_val = samples_at_centre[0];
        let range = max_val - min_val;

        // Separate into logic-0 and logic-1 populations
        let mid = (max_val + min_val) / 2.0;
        let ones: Vec<f64> = samples_at_centre
            .iter()
            .copied()
            .filter(|&v| v >= mid)
            .collect();
        let zeros: Vec<f64> = samples_at_centre
            .iter()
            .copied()
            .filter(|&v| v < mid)
            .collect();

        let mean_one = ones.iter().sum::<f64>() / ones.len().max(1) as f64;
        let mean_zero = zeros.iter().sum::<f64>() / zeros.len().max(1) as f64;
        let std_one = std_dev(&ones);
        let std_zero = std_dev(&zeros);

        let eye_opening_abs = (mean_one - std_one) - (mean_zero + std_zero);
        let eye_opening = if range > 0.0 {
            (eye_opening_abs / range).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let eye_crossing = (mean_one + mean_zero) / 2.0;

        // RMS jitter: approximate from transition samples
        let t_sym_ps = if symbol_rate > 0.0 {
            1e3 / symbol_rate
        } else {
            1.0
        };
        let jitter_rms_ps = (std_one + std_zero) / 2.0 / range.max(1e-30) * t_sym_ps;

        // ISI penalty: ratio of eye opening to ideal
        let isi_db = if eye_opening > 0.0 {
            -10.0 * eye_opening.log10()
        } else {
            30.0
        };

        Self {
            eye_opening,
            eye_crossing,
            jitter_rms_ps,
            isi_db,
        }
    }

    /// Estimate Q-factor from the eye statistics.
    ///
    /// Q ≈ (mean_one − mean_zero) / (σ_one + σ_zero)
    /// Approximated here as eye_opening / (1 − eye_opening + ε).
    pub fn q_factor(&self) -> f64 {
        if self.eye_opening <= 0.0 {
            return 0.0;
        }
        self.eye_opening / (1.0 - self.eye_opening + 1e-6)
    }

    /// Power penalty relative to an ideal eye (in dB).
    pub fn power_penalty_db(&self) -> f64 {
        self.isi_db
    }
}

/// Standard deviation of a slice of f64 values.
fn std_dev(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    var.sqrt()
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn erfc_at_zero() {
        // erfc(0) = 1
        assert_abs_diff_eq!(erfc_approx(0.0), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn erfc_symmetry() {
        // erfc(-x) = 2 - erfc(x)
        let x = 1.5;
        assert_abs_diff_eq!(erfc_approx(-x), 2.0 - erfc_approx(x), epsilon = 1e-10);
    }

    #[test]
    fn q_factor_ber_roundtrip() {
        let q = 6.0;
        let ber = q_factor_to_ber(q);
        let q_back = ber_to_q_factor(ber);
        assert_abs_diff_eq!(q_back, q, epsilon = 0.01);
    }

    #[test]
    fn dft_idft_roundtrip() {
        let n = 8;
        let signal: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new(k as f64, -(k as f64) * 0.5))
            .collect();
        let spectrum = dft(&signal);
        let recovered = idft(&spectrum);
        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(a.re, b.re, epsilon = 1e-9);
            assert_abs_diff_eq!(a.im, b.im, epsilon = 1e-9);
        }
    }

    #[test]
    fn dft_parseval() {
        // Parseval: Σ|x_n|² = (1/N) Σ|X_k|²
        let n = 16;
        let signal: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64).sin(), (k as f64).cos()))
            .collect();
        let power_time: f64 = signal.iter().map(|s| s.norm_sqr()).sum();
        let spectrum = dft(&signal);
        let m = spectrum.len() as f64;
        let power_freq: f64 = spectrum.iter().map(|s| s.norm_sqr()).sum::<f64>() / m;
        assert_abs_diff_eq!(power_time, power_freq, epsilon = 1e-8);
    }

    #[test]
    fn fec_rs_overhead() {
        let fec = FecCode::reed_solomon_255_239();
        let overhead = fec.overhead_percent();
        // Expected ≈ 6.72%
        assert!(
            (overhead - 6.72).abs() < 0.1,
            "RS overhead mismatch: {}",
            overhead
        );
    }

    #[test]
    fn gram_schmidt_orthogonality() {
        let v1 = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let v2 = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
        let basis = gram_schmidt(&[v1, v2]);
        assert_eq!(basis.len(), 2);
        // Orthogonality: |⟨e1, e2⟩| < ε
        let ip = inner_product(&basis[0], &basis[1]);
        assert_abs_diff_eq!(ip.norm(), 0.0, epsilon = 1e-12);
        // Normalisation
        assert_abs_diff_eq!(inner_product(&basis[0], &basis[0]).re, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(inner_product(&basis[1], &basis[1]).re, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn welch_psd_length() {
        let signal: Vec<Complex64> = (0..256)
            .map(|k| Complex64::new((2.0 * PI * 0.1 * k as f64).sin(), 0.0))
            .collect();
        let welch = WelchPsd::new(64, 32, WindowType::Hann);
        let (freqs, psd) = welch.compute(&signal, 1.0);
        assert_eq!(freqs.len(), psd.len());
        assert!(!psd.is_empty());
    }

    #[test]
    fn eye_diagram_open() {
        // Construct a clean NRZ signal (alternating 0/1)
        let sps = 16;
        let n_sym = 32;
        let mut signal = Vec::with_capacity(n_sym * sps);
        for sym in 0..n_sym {
            let level = if sym % 2 == 0 { 1.0 } else { 0.0 };
            for _ in 0..sps {
                signal.push(level);
            }
        }
        let eye = EyeDiagram::analyze(&signal, 10.0, sps);
        assert!(
            eye.eye_opening > 0.5,
            "Eye should be open: {}",
            eye.eye_opening
        );
    }

    #[test]
    fn osnr_to_snr_scaling() {
        // At OSNR = 20 dB and equal bandwidths, SNR should equal OSNR
        let snr = osnr_to_snr(20.0, 10.0, 10.0);
        let osnr_linear = 10.0_f64.powf(2.0); // 10^2 = 100
        assert_abs_diff_eq!(snr, osnr_linear, epsilon = 0.01);
    }
}
