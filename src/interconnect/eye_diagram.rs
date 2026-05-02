//! Eye-diagram simulation for optical interconnect links.
//!
//! Generates realistic eye diagrams from PRBS bit sequences, applies raised-cosine
//! pulse shaping, and propagates through the link S21 channel response. Computes
//! eye opening, Q-factor, jitter, and BER estimate.
//!
//! # Algorithm overview
//!
//! 1. Generate PRBS bits (ITU-T O.150 Galois LFSR).
//! 2. Map to NRZ symbols and upsample.
//! 3. Convolve with raised-cosine filter.
//! 4. FFT → apply S21 channel response (with Hermitian symmetry) → IFFT.
//! 5. Add AWGN calibrated to the requested OSNR.
//! 6. Fold into 2-UI eye traces.
//! 7. Extract eye metrics.

#![cfg(feature = "interconnect")]

use num_complex::Complex64;
use oxifft::{fft, ifft, Complex as OxiComplex};
use std::f64::consts::PI;

use crate::comms::modulation::ModulationFormat;
use crate::error::OxiPhotonError;
use crate::interconnect::sparam_link::SiPhLink;

// Reference bandwidth for OSNR (0.1 nm at 1550 nm)
const B_REF_HZ: f64 = 12.5e9;

// ─────────────────────────────────────────────────────────────────────────────
// PRBS generator
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a PRBS bit sequence using a Galois LFSR (ITU-T O.150).
///
/// Supported orders: 7 (x⁷+x⁶+1), 9 (x⁹+x⁵+1), 11 (x¹¹+x⁹+1), 15 (x¹⁵+x¹⁴+1).
///
/// The LFSR state is initialised to all-ones.  Output is the LSB of the state
/// after each clock, giving the standard ITU-T O.150 bit sequence.
///
/// # Errors
///
/// Returns [`OxiPhotonError::NumericalError`] for unsupported PRBS orders.
pub fn prbs(order: u8, n_bits: usize) -> Result<Vec<u8>, OxiPhotonError> {
    // Galois LFSR parameters: (register width, feedback mask).
    //
    // ITU-T O.150 Galois LFSR feedback polynomials (verified against period counts):
    //   PRBS-7:  x^7+x^6+1   width=7,  mask=0x44  → 64 ones in period 127
    //   PRBS-9:  x^9+x^5+1   width=9,  mask=0x108 → 256 ones in period 511
    //   PRBS-11: x^11+x^9+1  width=11, mask=0x500 → 1024 ones in period 2047
    //   PRBS-15: x^15+x^14+1 width=15, mask=0x6000 → 16384 ones in period 32767
    //
    // Galois LFSR step: if LSB==1, shift right and XOR with mask; else just shift right.
    let (shift_width, galois_mask): (u8, u32) = match order {
        7 => (7, 0x44),
        9 => (9, 0x108),
        11 => (11, 0x500),
        15 => (15, 0x6000),
        _ => {
            return Err(OxiPhotonError::NumericalError(format!(
                "Unsupported PRBS order {order}; supported: 7, 9, 11, 15"
            )))
        }
    };

    let state_max: u32 = (1u32 << shift_width) - 1;
    let mut state: u32 = state_max; // all-ones init

    let mut bits = Vec::with_capacity(n_bits);
    for _ in 0..n_bits {
        let output_bit = (state & 1) as u8;
        bits.push(output_bit);
        // Galois LFSR step
        let feedback = state & 1;
        state >>= 1;
        if feedback == 1 {
            state ^= galois_mask;
        }
        // Keep only the relevant bits
        state &= state_max;
    }

    Ok(bits)
}

// ─────────────────────────────────────────────────────────────────────────────
// Raised-cosine filter
// ─────────────────────────────────────────────────────────────────────────────

/// Raised-cosine FIR filter impulse response.
///
/// `h[k]` = sinc(k/T) · cos(π·β·k/T) / (1 − (2·β·k/T)²)
///
/// sampled at integer sample positions k from −(n_taps/2) to +(n_taps/2).
/// T = `samples_per_bit` (oversampling factor).
///
/// The limit at |k| = T/(2β) is evaluated via L'Hôpital's rule:
///   h = (π/4) · sinc(1/(2β))
///
/// The result is normalised so that Σ `h[k]` = 1.
pub fn raised_cosine_filter(rolloff: f64, samples_per_bit: usize, n_taps: usize) -> Vec<f64> {
    let t = samples_per_bit as f64;
    let center = (n_taps / 2) as isize;

    let h: Vec<f64> = (0..n_taps)
        .map(|i| {
            let k = i as isize - center;
            let kf = k as f64;

            if k == 0 {
                // k=0: sinc(0)·... limit → 1
                1.0
            } else {
                let kt = kf / t; // k / T
                let sinc_val = (PI * kt).sin() / (PI * kt);
                let denom_arg = 2.0 * rolloff * kt; // 2·β·k/T
                let denom = 1.0 - denom_arg * denom_arg;

                if denom.abs() < 1e-9 {
                    // At the Nyquist singularity: |2βk/T| = 1 → k = ±T/(2β)
                    // L'Hôpital limit: h = (π/4)·sinc(1/(2β))
                    let x = 1.0 / (2.0 * rolloff);
                    let sinc_x = if x.abs() < 1e-12 {
                        1.0
                    } else {
                        (PI * x).sin() / (PI * x)
                    };
                    (PI / 4.0) * sinc_x
                } else {
                    let cos_val = (PI * rolloff * kt).cos();
                    sinc_val * cos_val / denom
                }
            }
        })
        .collect();

    // Normalise so sum = 1
    let sum: f64 = h.iter().sum();
    if sum.abs() < 1e-30 {
        h
    } else {
        h.iter().map(|&v| v / sum).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Erfc approximation (Abramowitz & Stegun 7.1.26)
// ─────────────────────────────────────────────────────────────────────────────

/// Complementary error function approximation (Abramowitz & Stegun 7.1.26).
///
/// Error < 1.5e-7 for x ≥ 0.  Extended to x < 0 via erfc(-x) = 2 - erfc(x).
pub(crate) fn erfc_approx(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc_approx(-x);
    }
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    poly * (-x * x).exp()
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the eye-diagram simulation.
#[derive(Debug, Clone)]
pub struct EyeDiagramConfig {
    /// Bit rate (Gb/s)
    pub bit_rate_gbps: f64,
    /// Oversampling factor (samples per bit), e.g. 32
    pub samples_per_bit: usize,
    /// Number of PRBS bits to simulate, e.g. 512
    pub n_bits: usize,
    /// Modulation format (OOK or PAM-4 supported)
    pub modulation: ModulationFormat,
    /// Raised-cosine roll-off factor β ∈ [0, 1]
    pub rolloff: f64,
    /// OSNR in dB (0.1 nm reference bandwidth)
    pub osnr_db: f64,
    /// PRBS order (7, 9, 11, or 15)
    pub prbs_order: u8,
}

impl Default for EyeDiagramConfig {
    fn default() -> Self {
        Self {
            bit_rate_gbps: 25.0,
            samples_per_bit: 32,
            n_bits: 512,
            modulation: ModulationFormat::Ook,
            rolloff: 0.35,
            osnr_db: 20.0,
            prbs_order: 7,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Result
// ─────────────────────────────────────────────────────────────────────────────

/// Result of the eye-diagram simulation.
#[derive(Debug, Clone)]
pub struct EyeDiagramResult {
    /// Time axis for a 2-UI window (ps), length = 2 × samples_per_bit
    pub time_axis_ps: Vec<f64>,
    /// Eye traces: each trace is one 2-UI window \[sample\]
    pub traces: Vec<Vec<f64>>,
    /// Vertical eye opening at the optimal sample point (signal units)
    pub eye_opening_v: f64,
    /// Q-factor = (μ₁ − μ₀) / (σ₁ + σ₀)
    pub q_factor: f64,
    /// RMS jitter of zero-crossings (ps)
    pub jitter_rms_ps: f64,
    /// BER estimate = 0.5 × erfc(Q/√2)
    pub ber_estimate: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Convert `num_complex::Complex64` to `oxifft::Complex<f64>`.
#[inline]
fn to_oxi(c: Complex64) -> OxiComplex<f64> {
    OxiComplex::new(c.re, c.im)
}

/// Linearly interpolate S21 from the link cascade at a given frequency.
///
/// The input `freqs_hz` are the evaluation frequencies of `s_params`.
/// `s21_vals[k]` = S21 at `freqs_hz[k]`.
///
/// Outside the range, the boundary value is returned.
fn interpolate_s21(f_query: f64, freqs_hz: &[f64], s21_vals: &[Complex64]) -> Complex64 {
    let n = freqs_hz.len();
    if n == 0 {
        return Complex64::new(1.0, 0.0);
    }
    if f_query <= freqs_hz[0] {
        return s21_vals[0];
    }
    if f_query >= freqs_hz[n - 1] {
        return s21_vals[n - 1];
    }
    // Binary search for surrounding interval
    let mut lo = 0usize;
    let mut hi = n - 1;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if freqs_hz[mid] <= f_query {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let f_lo = freqs_hz[lo];
    let f_hi = freqs_hz[hi];
    let df = f_hi - f_lo;
    if df < 1e-30 {
        return s21_vals[lo];
    }
    let t = (f_query - f_lo) / df;
    s21_vals[lo] * (1.0 - t) + s21_vals[hi] * t
}

/// Generate NRZ-OOK symbol waveform from a bit sequence, upsampled by `sps`.
///
/// Bit 0 → amplitude 0.0 (optical low), Bit 1 → amplitude 1.0 (optical high).
/// Each bit is held for `sps` samples (sample-and-hold NRZ).
fn bits_to_ook_nrz(bits: &[u8], sps: usize) -> Vec<f64> {
    let n = bits.len() * sps;
    let mut out = vec![0.0f64; n];
    for (bit_idx, &b) in bits.iter().enumerate() {
        let level = if b == 1 { 1.0 } else { 0.0 };
        let start = bit_idx * sps;
        for s in out[start..start + sps].iter_mut() {
            *s = level;
        }
    }
    out
}

/// Generate PAM-4 symbol waveform from a bit sequence, upsampled by `sps`.
///
/// Pairs of bits (b0, b1) → four levels: (0,0)→0, (0,1)→1/3, (1,0)→2/3, (1,1)→1.
/// If the bit count is odd, the last bit is padded to make an even count.
fn bits_to_pam4_nrz(bits: &[u8], sps: usize) -> Vec<f64> {
    let n_syms = bits.len().div_ceil(2);
    let n = n_syms * sps;
    let mut out = vec![0.0f64; n];
    let levels = [0.0_f64, 1.0 / 3.0, 2.0 / 3.0, 1.0];
    for sym_idx in 0..n_syms {
        let b0 = bits.get(2 * sym_idx).copied().unwrap_or(0) as usize;
        let b1 = bits.get(2 * sym_idx + 1).copied().unwrap_or(0) as usize;
        let level_idx = b0 * 2 + b1;
        let level = levels[level_idx.min(3)];
        let start = sym_idx * sps;
        for s in out[start..start + sps].iter_mut() {
            *s = level;
        }
    }
    out
}

/// Direct convolution of signal with filter (time-domain).
///
/// Output length = signal.len() (causal, centred, truncated to match).
fn convolve_full(signal: &[f64], filter: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let m = filter.len();
    let offset = m / 2; // centre the filter
    let mut out = vec![0.0f64; n];
    for (i, o) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (j, &h) in filter.iter().enumerate() {
            let si = i + j;
            if si >= offset && si - offset < n {
                acc += signal[si - offset] * h;
            }
        }
        *o = acc;
    }
    out
}

/// Add Gaussian noise samples to a waveform in-place.
///
/// Uses a simple Box-Muller transform with a linear-congruential PRNG to avoid
/// external dependencies.
fn add_awgn(waveform: &mut [f64], noise_variance: f64, seed: u64) {
    if noise_variance <= 0.0 {
        return;
    }
    let sigma = noise_variance.sqrt();
    let mut state = seed ^ 0xDEAD_BEEF_1234_5678;

    let lcg = |s: &mut u64| -> f64 {
        *s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Take upper 32 bits, map to [0, 1)
        let hi = (*s >> 32) as f64;
        hi / 4_294_967_296.0
    };

    let mut i = 0usize;
    while i < waveform.len() {
        // Box-Muller: two uniform samples → two Gaussian samples
        let u1 = lcg(&mut state).max(1e-15);
        let u2 = lcg(&mut state);
        let r = (-2.0 * u1.ln()).sqrt() * sigma;
        let theta = 2.0 * PI * u2;
        waveform[i] += r * theta.cos();
        if i + 1 < waveform.len() {
            waveform[i + 1] += r * theta.sin();
        }
        i += 2;
    }
}

/// Compute RMS jitter from zero-crossing times of eye traces.
///
/// For each trace we look for a rising zero-crossing near sample `sps` (the
/// midpoint of the 2-UI window, i.e., the bit transition).  Crossing time is
/// linearly interpolated between adjacent samples.
fn compute_jitter_rms_ps(traces: &[Vec<f64>], sps: usize, ui_ps: f64) -> f64 {
    let dt_ps = ui_ps / sps as f64; // ps per sample
    let search_start = sps / 2;
    let search_end = 3 * sps / 2;

    let mut crossings = Vec::new();

    for trace in traces {
        let n = trace.len();
        // Look for zero-crossing (0.5 threshold for OOK) in vicinity of the bit edge
        for k in search_start..search_end.min(n.saturating_sub(1)) {
            let threshold = 0.5;
            let before = trace[k] - threshold;
            let after = trace[k + 1] - threshold;
            if before < 0.0 && after >= 0.0 {
                // Rising zero-crossing: interpolate
                let frac = (-before) / (after - before).max(1e-30);
                let t_cross_ps = (k as f64 + frac) * dt_ps;
                crossings.push(t_cross_ps);
                break;
            }
        }
    }

    if crossings.len() < 2 {
        return 0.0;
    }

    let mean = crossings.iter().sum::<f64>() / crossings.len() as f64;
    let variance = crossings
        .iter()
        .map(|&t| (t - mean) * (t - mean))
        .sum::<f64>()
        / crossings.len() as f64;
    variance.sqrt()
}

/// Compute OOK eye metrics from traces sampled at index `sample_idx`.
///
/// Returns `(eye_opening, q_factor)`.
fn ook_eye_metrics(traces: &[Vec<f64>], sample_idx: usize) -> (f64, f64) {
    let threshold = 0.5;
    let mut high: Vec<f64> = Vec::new();
    let mut low: Vec<f64> = Vec::new();

    for trace in traces {
        if let Some(&v) = trace.get(sample_idx) {
            if v >= threshold {
                high.push(v);
            } else {
                low.push(v);
            }
        }
    }

    if high.is_empty() || low.is_empty() {
        return (0.0, 0.0);
    }

    let min_high = high.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_low = low.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let eye_opening = min_high - max_low;

    let mu1 = high.iter().sum::<f64>() / high.len() as f64;
    let mu0 = low.iter().sum::<f64>() / low.len() as f64;
    let var1 = high.iter().map(|&v| (v - mu1).powi(2)).sum::<f64>() / high.len() as f64;
    let var0 = low.iter().map(|&v| (v - mu0).powi(2)).sum::<f64>() / low.len() as f64;
    let sigma1 = var1.sqrt();
    let sigma0 = var0.sqrt();

    let denom = sigma1 + sigma0;
    let q_factor = if denom < 1e-30 {
        f64::INFINITY
    } else {
        (mu1 - mu0) / denom
    };

    (eye_opening.max(0.0), q_factor.max(0.0))
}

/// Compute PAM-4 eye metrics from traces sampled at index `sample_idx`.
///
/// PAM-4 has three eye openings (between levels 0/1, 1/2, 2/3).
/// Returns the minimum opening and a Q-factor based on the worst eye.
fn pam4_eye_metrics(traces: &[Vec<f64>], sample_idx: usize) -> (f64, f64) {
    // Four levels ≈ 0, 1/3, 2/3, 1 with ± spread
    let thresholds = [1.0 / 6.0, 0.5, 5.0 / 6.0];
    let level_centers = [0.0_f64, 1.0 / 3.0, 2.0 / 3.0, 1.0];

    let mut groups: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

    for trace in traces {
        if let Some(&v) = trace.get(sample_idx) {
            // Assign to nearest level
            let idx = level_centers
                .iter()
                .enumerate()
                .min_by(|(_, &a), (_, &b)| {
                    (v - a)
                        .abs()
                        .partial_cmp(&(v - b).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            groups[idx].push(v);
        }
    }

    let mut min_opening = f64::INFINITY;
    let mut min_q = f64::INFINITY;

    for eye_idx in 0..3 {
        let hi = &groups[eye_idx + 1];
        let lo = &groups[eye_idx];
        if hi.is_empty() || lo.is_empty() {
            continue;
        }
        let min_hi = hi.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_lo = lo.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let opening = (min_hi - max_lo).max(0.0);
        if opening < min_opening {
            min_opening = opening;
        }

        let mu1 = hi.iter().sum::<f64>() / hi.len() as f64;
        let mu0 = lo.iter().sum::<f64>() / lo.len() as f64;
        let var1 = hi.iter().map(|&v| (v - mu1).powi(2)).sum::<f64>() / hi.len() as f64;
        let var0 = lo.iter().map(|&v| (v - mu0).powi(2)).sum::<f64>() / lo.len() as f64;
        let sigma1 = var1.sqrt();
        let sigma0 = var0.sqrt();
        let denom = sigma1 + sigma0;
        let q = if denom < 1e-30 {
            f64::INFINITY
        } else {
            (mu1 - mu0) / denom
        };
        if q < min_q {
            min_q = q;
        }
        let _threshold = thresholds[eye_idx]; // decision level — documented for future use
    }

    let opening = if min_opening == f64::INFINITY {
        0.0
    } else {
        min_opening
    };
    let q = if min_q == f64::INFINITY {
        0.0
    } else {
        min_q.max(0.0)
    };
    (opening, q)
}

// ─────────────────────────────────────────────────────────────────────────────
// Main simulation entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Simulate an eye diagram for the given SiPh link and configuration.
///
/// # Arguments
///
/// * `link` — cascaded SiPh link (provides frequency-domain S21).
/// * `freqs_hz` — frequency sweep points for the link S-parameter evaluation.
/// * `config` — eye-diagram simulation parameters.
///
/// # Errors
///
/// Returns [`OxiPhotonError::NumericalError`] if the configuration is invalid
/// (e.g. zero samples, unsupported PRBS order).
pub fn simulate_eye(
    link: &SiPhLink,
    freqs_hz: &[f64],
    config: &EyeDiagramConfig,
) -> Result<EyeDiagramResult, OxiPhotonError> {
    // ── Validate configuration ────────────────────────────────────────────────
    if config.n_bits < 4 {
        return Err(OxiPhotonError::NumericalError(
            "n_bits must be at least 4".to_string(),
        ));
    }
    if config.samples_per_bit < 2 {
        return Err(OxiPhotonError::NumericalError(
            "samples_per_bit must be at least 2".to_string(),
        ));
    }
    if config.bit_rate_gbps <= 0.0 {
        return Err(OxiPhotonError::NumericalError(
            "bit_rate_gbps must be positive".to_string(),
        ));
    }

    let sps = config.samples_per_bit;
    let n_bits = config.n_bits;

    // ── Step 1: Generate PRBS bits ────────────────────────────────────────────
    let bits = prbs(config.prbs_order, n_bits)?;

    // ── Step 2: Map bits to NRZ symbols and upsample ─────────────────────────
    let waveform_raw: Vec<f64> = match config.modulation {
        ModulationFormat::Pam4 => bits_to_pam4_nrz(&bits, sps),
        _ => bits_to_ook_nrz(&bits, sps),
    };
    let n_samples = waveform_raw.len();

    // ── Step 3: Raised-cosine filter ─────────────────────────────────────────
    let n_taps = 10 * sps + 1;
    let h_rc = raised_cosine_filter(config.rolloff, sps, n_taps);
    let waveform_shaped = convolve_full(&waveform_raw, &h_rc);

    // ── Step 4: FFT of shaped waveform ────────────────────────────────────────
    // Convert to oxifft complex (imaginary part = 0)
    let waveform_oxi: Vec<OxiComplex<f64>> = waveform_shaped
        .iter()
        .map(|&v| OxiComplex::new(v, 0.0))
        .collect();
    let waveform_freq = fft(&waveform_oxi);
    let n_fft = waveform_freq.len(); // should equal n_samples

    // ── Step 5: Compute S21 at each FFT bin ──────────────────────────────────
    // Sample rate and frequency resolution
    let f_s = config.bit_rate_gbps * 1e9 * sps as f64; // Hz
    let df = f_s / n_fft as f64;

    // Precompute S21 at link evaluation frequencies
    let s_params = link.cascade(freqs_hz);
    let s21_vals: Vec<Complex64> = s_params.iter().map(|&[_, s21, _, _]| s21).collect();

    // Apply S21 at each FFT bin (with Hermitian symmetry for real signal)
    let mut waveform_channel: Vec<OxiComplex<f64>> = waveform_freq.clone();
    for (k, wf_bin) in waveform_channel.iter_mut().enumerate() {
        // Physical frequency: positive for k ≤ N/2, negative for k > N/2
        let f_k = if k <= n_fft / 2 {
            k as f64 * df
        } else {
            (k as f64 - n_fft as f64) * df
        };
        // Apply Hermitian symmetry: S21(-f) = conj(S21(f)) for real systems
        let s21_at_fk = if f_k >= 0.0 {
            interpolate_s21(f_k, freqs_hz, &s21_vals)
        } else {
            interpolate_s21(-f_k, freqs_hz, &s21_vals).conj()
        };
        let s21_oxi = to_oxi(s21_at_fk);
        // OxiComplex multiplication in-place
        let (re, im) = (wf_bin.re, wf_bin.im);
        *wf_bin = OxiComplex::new(
            re * s21_oxi.re - im * s21_oxi.im,
            re * s21_oxi.im + im * s21_oxi.re,
        );
    }

    // ── Step 6: IFFT → time domain ────────────────────────────────────────────
    let waveform_complex = ifft(&waveform_channel);

    // Extract real part (imaginary should be ~0 due to Hermitian symmetry)
    let mut waveform_time: Vec<f64> = waveform_complex.iter().map(|c| c.re).collect();

    // oxifft ifft normalises by 1/N — scale back to match forward FFT convention
    // (fft_bpm.rs calls fft then ifft and treats result as identity, so ifft does /N)
    // The shaped waveform's DC level should match after round-trip; no re-scaling needed.

    // ── Step 7: Add AWGN calibrated to OSNR ──────────────────────────────────
    // OSNR (linear) = signal_power / (2 * noise_variance * B_ref / B_s)
    // signal_power ≈ mean(waveform²) over the waveform
    let signal_power: f64 = {
        let sum_sq: f64 = waveform_time.iter().map(|&v| v * v).sum();
        sum_sq / n_samples as f64
    };
    let signal_power = signal_power.max(1e-30);

    let osnr_lin = 10.0_f64.powf(config.osnr_db / 10.0);
    let b_s = config.bit_rate_gbps * 1e9;
    // noise_variance = signal_power * B_ref / (2 * OSNR * B_s)
    let noise_variance = signal_power * B_REF_HZ / (2.0 * osnr_lin * b_s);

    add_awgn(&mut waveform_time, noise_variance, 0xC0CA_C01A);

    // ── Step 8: Eye folding ───────────────────────────────────────────────────
    // UI period in ps
    let ui_ps = 1e12 / (config.bit_rate_gbps * 1e9);
    // 2-UI window = 2 * sps samples
    let window = 2 * sps;

    // Build time axis for one 2-UI window
    let dt_ps = ui_ps / sps as f64;
    let time_axis_ps: Vec<f64> = (0..window).map(|i| i as f64 * dt_ps).collect();

    // Skip guard bits at start and end to avoid transients
    let guard_bits = 10;
    let start_sample = guard_bits * sps;
    let end_sample = n_samples.saturating_sub(guard_bits * sps);

    let mut traces: Vec<Vec<f64>> = Vec::new();

    // Slice into 2-UI segments
    let mut pos = start_sample;
    while pos + window <= end_sample {
        let trace: Vec<f64> = waveform_time[pos..pos + window].to_vec();
        traces.push(trace);
        pos += window;
    }

    // ── Step 9: Compute eye metrics ───────────────────────────────────────────
    // Sample at the centre of the first UI (optimal sampling point)
    let sample_idx = sps / 2;

    let (eye_opening_v, q_factor) = match config.modulation {
        ModulationFormat::Pam4 => pam4_eye_metrics(&traces, sample_idx),
        _ => ook_eye_metrics(&traces, sample_idx),
    };

    let jitter_rms_ps = compute_jitter_rms_ps(&traces, sps, ui_ps);

    let ber_estimate = 0.5 * erfc_approx(q_factor / 2.0_f64.sqrt());

    Ok(EyeDiagramResult {
        time_axis_ps,
        traces,
        eye_opening_v,
        q_factor,
        jitter_rms_ps,
        ber_estimate,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erfc_at_zero_is_one() {
        let v = erfc_approx(0.0);
        assert!((v - 1.0).abs() < 1e-6, "erfc(0) should be 1, got {v}");
    }

    #[test]
    fn erfc_symmetry() {
        let v1 = erfc_approx(1.0);
        let v2 = erfc_approx(-1.0);
        assert!((v1 + v2 - 2.0).abs() < 1e-6, "erfc(-x) = 2 - erfc(x)");
    }

    #[test]
    fn raised_cosine_normalised() {
        let h = raised_cosine_filter(0.35, 16, 10 * 16 + 1);
        let sum: f64 = h.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Filter should sum to 1, got {sum}"
        );
    }

    #[test]
    fn prbs7_unsupported_order_errors() {
        assert!(prbs(5, 100).is_err());
    }

    #[test]
    fn ook_nrz_correct_levels() {
        let bits = vec![0u8, 1, 0, 1];
        let out = bits_to_ook_nrz(&bits, 4);
        assert_eq!(out.len(), 16);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[4], 1.0);
    }
}
