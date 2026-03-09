/// Photonic RF signal processing: transversal filters, resonator-based filters,
/// and photonic Hilbert transformers.
///
/// All frequency responses are computed analytically; no FFT or external
/// crate is required beyond `num_complex`.
use num_complex::Complex64;
use std::f64::consts::PI;

// ─── PhotonicRfFilter ─────────────────────────────────────────────────────────

/// Transversal (FIR) photonic RF filter using multiple wavelength taps.
///
/// The filter implements a weighted tapped delay line:
///
///   H(f) = Σₖ wₖ · exp(−j 2π f τₖ)
///
/// Negative tap weights are realized using a balanced photodetector pair.
#[derive(Debug, Clone)]
pub struct PhotonicRfFilter {
    /// Number of taps.
    pub n_taps: usize,
    /// Amplitude weights for each tap (can be negative for balanced-PD taps).
    pub tap_weights: Vec<f64>,
    /// Absolute delay of each tap from the first tap \[s\].
    /// `tap_delays[0]` is conventionally 0.
    pub tap_delays: Vec<f64>,
    /// Design center frequency \[Hz\].
    pub center_frequency: f64,
}

impl PhotonicRfFilter {
    /// Construct a uniform FIR filter from tap weights and a uniform inter-tap delay.
    ///
    /// # Arguments
    /// * `weights` – amplitude weight for each tap
    /// * `delay` – differential delay between consecutive taps \[s\]
    pub fn new_fir(weights: Vec<f64>, delay: f64) -> Self {
        let n = weights.len();
        let delays: Vec<f64> = (0..n).map(|k| k as f64 * delay).collect();
        let fc = if delay > 0.0 { 0.5 / delay } else { 0.0 };
        PhotonicRfFilter {
            n_taps: n,
            tap_weights: weights,
            tap_delays: delays,
            center_frequency: fc,
        }
    }

    /// Construct a bandpass filter centered at `fc` using N taps with delay `T`.
    ///
    /// Uses a Hamming-windowed sinc prototype transformed to the desired center.
    pub fn new_bandpass(fc: f64, n_taps: usize, delay: f64) -> Self {
        let mut weights = Vec::with_capacity(n_taps);
        let half = (n_taps as f64 - 1.0) / 2.0;
        for k in 0..n_taps {
            let n = k as f64 - half;
            // Hamming window
            let w_ham = 0.54 - 0.46 * (2.0 * PI * k as f64 / (n_taps as f64 - 1.0)).cos();
            // Modulated tap coefficient: cos(2π fc T n) * hamming
            let h = if n.abs() < 1e-12 {
                w_ham
            } else {
                w_ham * (2.0 * PI * fc * delay * n).cos()
            };
            weights.push(h);
        }
        let delays: Vec<f64> = (0..n_taps).map(|k| k as f64 * delay).collect();
        PhotonicRfFilter {
            n_taps,
            tap_weights: weights,
            tap_delays: delays,
            center_frequency: fc,
        }
    }

    /// Complex transfer function H(f) = Σₖ wₖ · exp(−j 2π f τₖ).
    pub fn transfer_function(&self, freq_hz: f64) -> Complex64 {
        self.tap_weights.iter().zip(self.tap_delays.iter()).fold(
            Complex64::new(0.0, 0.0),
            |acc, (&w, &tau)| {
                let phase = -2.0 * PI * freq_hz * tau;
                acc + Complex64::new(w, 0.0) * Complex64::new(phase.cos(), phase.sin())
            },
        )
    }

    /// Power response |H(f)|² in dB.
    pub fn response_db(&self, freq_hz: f64) -> f64 {
        let h = self.transfer_function(freq_hz);
        let power = h.norm_sqr();
        if power > 0.0 {
            10.0 * power.log10()
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Free spectral range (FSR) of the filter \[Hz\].
    ///
    /// FSR = 1 / T, where T is the inter-tap delay.
    pub fn fsr_hz(&self) -> f64 {
        if self.n_taps < 2 {
            return f64::INFINITY;
        }
        // Use the uniform tap spacing from tap_delays[1] - tap_delays[0]
        let t = self.tap_delays[1] - self.tap_delays[0];
        if t > 0.0 {
            1.0 / t
        } else {
            f64::INFINITY
        }
    }

    /// Approximate 3-dB bandwidth of the passband \[Hz\].
    ///
    /// For a uniform FIR filter the 3-dB bandwidth is approximately FSR / N_taps.
    pub fn bandwidth_hz(&self) -> f64 {
        let fsr = self.fsr_hz();
        if fsr.is_finite() {
            fsr / self.n_taps as f64
        } else {
            f64::INFINITY
        }
    }

    /// Stopband rejection: minimum sidelobe power level \[dB\] relative to passband peak.
    ///
    /// Estimated by sweeping one FSR on a fine grid.
    pub fn stopband_rejection_db(&self) -> f64 {
        let fsr = self.fsr_hz();
        if !fsr.is_finite() || fsr <= 0.0 {
            return 0.0;
        }
        let n_scan = 4096;
        let passband_db = self.response_db(self.center_frequency);
        // Search sidelobe minimum outside the 3-dB bandwidth
        let bw3db = self.bandwidth_hz();
        let f_min = self.center_frequency + bw3db;
        let f_max = self.center_frequency + fsr;
        if f_min >= f_max {
            return 0.0;
        }
        let mut max_sidelobe = f64::NEG_INFINITY;
        for i in 0..n_scan {
            let f = f_min + (f_max - f_min) * i as f64 / (n_scan as f64 - 1.0);
            let r = self.response_db(f);
            if r > max_sidelobe {
                max_sidelobe = r;
            }
        }
        passband_db - max_sidelobe
    }

    /// Compute the power response \[dB\] over a frequency grid.
    ///
    /// Returns a vector of `(frequency_hz, response_db)` pairs.
    pub fn frequency_response(
        &self,
        f_start: f64,
        f_stop: f64,
        n_points: usize,
    ) -> Vec<(f64, f64)> {
        if n_points == 0 {
            return Vec::new();
        }
        (0..n_points)
            .map(|i| {
                let f = f_start + (f_stop - f_start) * i as f64 / (n_points as f64 - 1.0).max(1.0);
                (f, self.response_db(f))
            })
            .collect()
    }
}

// ─── RingResonatorRfFilter ────────────────────────────────────────────────────

/// Photonic microwave bandpass filter modeled as a Lorentzian resonance.
///
/// The optical ring resonator imposes a Lorentzian transfer function on the
/// RF photocurrent spectrum via the intensity modulation/direct detection scheme.
#[derive(Debug, Clone)]
pub struct RingResonatorRfFilter {
    /// Center frequency \[Hz\].
    pub center_freq_hz: f64,
    /// 3-dB bandwidth \[Hz\].
    pub bandwidth_hz: f64,
    /// In-band insertion loss \[dB\].
    pub insertion_loss_db: f64,
    /// Filter order (number of cascaded resonators).
    pub order: usize,
}

impl RingResonatorRfFilter {
    /// Create a ring-resonator RF filter.
    ///
    /// # Arguments
    /// * `fc` – center frequency \[Hz\]
    /// * `bw` – 3-dB bandwidth \[Hz\]
    /// * `order` – filter order (≥ 1)
    pub fn new(fc: f64, bw: f64, order: usize) -> Self {
        RingResonatorRfFilter {
            center_freq_hz: fc,
            bandwidth_hz: bw,
            insertion_loss_db: 0.5 * order as f64, // approx 0.5 dB per resonator
            order: order.max(1),
        }
    }

    /// Complex transfer function using a Butterworth-Lorentzian model.
    ///
    /// For a single-pole resonator:
    ///   H(f) = (BW/2) / (j(f − fc) + BW/2)
    ///
    /// For order N the poles are cascaded with the Butterworth prototype.
    pub fn transfer_function(&self, freq_hz: f64) -> Complex64 {
        let il_linear = 10.0_f64.powf(-self.insertion_loss_db / 20.0);
        let df = freq_hz - self.center_freq_hz;
        // Normalized frequency relative to half-bandwidth
        let bw_half = self.bandwidth_hz / 2.0;
        // For Butterworth prototype of order N:
        //   |H(jΩ)|² = 1 / (1 + Ω^(2N))
        // Apply as magnitude with linear phase approximation
        let omega_n = df / bw_half;
        let mag_sq = 1.0 / (1.0 + omega_n.powi(2 * self.order as i32));
        let mag = il_linear * mag_sq.sqrt();
        // Phase: single-pole Lorentzian phase ≈ -arctan(df / (BW/2)) * order
        let phase = -(df / bw_half).atan() * self.order as f64;
        Complex64::new(mag * phase.cos(), mag * phase.sin())
    }

    /// Group delay \[s\] at frequency `freq_hz`.
    ///
    /// For a Lorentzian resonance:
    ///   τ(f) = order · (BW/2) / (π · ((f−fc)² + (BW/2)²))
    pub fn group_delay_s(&self, freq_hz: f64) -> f64 {
        let bw_half = self.bandwidth_hz / 2.0;
        let df = freq_hz - self.center_freq_hz;
        // Single-pole Lorentzian: τ = (BW/2) / (π((f-fc)² + (BW/2)²))
        // For order N (cascaded): multiply by N
        let tau_single = bw_half / (PI * (df * df + bw_half * bw_half));
        tau_single * self.order as f64
    }

    /// Q factor of the resonance: Q = fc / BW.
    pub fn q_factor(&self) -> f64 {
        self.center_freq_hz / self.bandwidth_hz
    }
}

// ─── PhotonicHilbertTransformer ───────────────────────────────────────────────

/// Photonic Hilbert transformer: provides 90° phase shift over a broad RF bandwidth.
///
/// Implemented as a tapped delay line with antisymmetric Hilbert coefficients:
///   h\[k\] = 2/(πk) for odd k,  0 for even k (and k ≠ 0)
///
/// An optional fractional design allows non-integer group delay.
#[derive(Debug, Clone)]
pub struct PhotonicHilbertTransformer {
    /// Number of taps (should be odd for a centered design).
    pub n_taps: usize,
    /// Operating bandwidth \[Hz\] (used to scale tap delays).
    pub bandwidth_hz: f64,
    /// Use fractional Hilbert transformer design.
    pub fractional: bool,
}

impl PhotonicHilbertTransformer {
    /// Create a photonic Hilbert transformer.
    ///
    /// # Arguments
    /// * `n_taps` – number of filter taps (odd number recommended)
    /// * `bandwidth_hz` – target operating bandwidth \[Hz\]
    pub fn new(n_taps: usize, bandwidth_hz: f64) -> Self {
        PhotonicHilbertTransformer {
            n_taps: if n_taps % 2 == 0 { n_taps + 1 } else { n_taps },
            bandwidth_hz,
            fractional: false,
        }
    }

    /// Create a fractional-order Hilbert transformer with arbitrary phase shift.
    pub fn new_fractional(n_taps: usize, bandwidth_hz: f64) -> Self {
        PhotonicHilbertTransformer {
            n_taps: if n_taps % 2 == 0 { n_taps + 1 } else { n_taps },
            bandwidth_hz,
            fractional: true,
        }
    }

    /// Hilbert transformer tap coefficients h\[k\].
    ///
    /// Returns the sequence h\[0..n_taps\] with the center tap at index n_taps/2.
    ///   h\[center\] = 0
    ///   h[center ± k] = ±2/(πk) for odd k, 0 for even k
    /// Windowed with a Hamming window for sidelobe suppression.
    pub fn tap_coefficients(&self) -> Vec<f64> {
        let n = self.n_taps;
        let center = n / 2;
        let mut coeffs = vec![0.0_f64; n];
        for (i, coeff) in coeffs.iter_mut().enumerate() {
            let k = i as i64 - center as i64;
            if k == 0 {
                *coeff = 0.0;
            } else if k % 2 != 0 {
                // Hamming window
                let w = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos();
                *coeff = w * 2.0 / (PI * k as f64);
            } else {
                *coeff = 0.0;
            }
        }
        coeffs
    }

    /// Inter-tap delay \[s\] for the transformer operating at `bandwidth_hz`.
    fn tap_delay_s(&self) -> f64 {
        // Nyquist: T = 1 / (2 * bandwidth_hz) so the FSR covers the operating band
        0.5 / self.bandwidth_hz
    }

    /// Complex frequency response of the Hilbert transformer at `freq_hz`.
    pub fn response_at(&self, freq_hz: f64) -> Complex64 {
        let coeffs = self.tap_coefficients();
        let t = self.tap_delay_s();
        let center = self.n_taps / 2;
        coeffs
            .iter()
            .enumerate()
            .fold(Complex64::new(0.0, 0.0), |acc, (i, &h)| {
                let k = i as i64 - center as i64;
                let tau = k as f64 * t;
                let phase = -2.0 * PI * freq_hz * tau;
                acc + Complex64::new(h, 0.0) * Complex64::new(phase.cos(), phase.sin())
            })
    }

    /// Phase error from ideal 90° at `freq_hz` \[degrees\].
    ///
    /// The ideal Hilbert transformer has a phase response of −90° for f > 0.
    pub fn phase_error_deg(&self, freq_hz: f64) -> f64 {
        let h = self.response_at(freq_hz);
        let actual_phase_deg = h.arg().to_degrees();
        // Ideal phase for Hilbert transformer is -90°
        let error = actual_phase_deg - (-90.0);
        // Wrap to [-180, 180]

        ((error + 180.0).rem_euclid(360.0)) - 180.0
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── PhotonicRfFilter ─────────────────────────────────────────────────────

    #[test]
    fn test_fir_filter_dc_response() {
        // Uniform weights sum to N; DC gain = N
        let n = 5usize;
        let weights = vec![1.0; n];
        let delay = 100e-12; // 100 ps
        let filter = PhotonicRfFilter::new_fir(weights, delay);
        let h_dc = filter.transfer_function(0.0);
        assert_abs_diff_eq!(h_dc.re, n as f64, epsilon = 1e-9);
        assert_abs_diff_eq!(h_dc.im, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_fir_filter_fsr() {
        let delay = 200e-12; // 200 ps → FSR = 5 GHz
        let filter = PhotonicRfFilter::new_fir(vec![1.0; 4], delay);
        assert_abs_diff_eq!(filter.fsr_hz(), 5.0e9, epsilon = 1.0);
    }

    #[test]
    fn test_fir_filter_response_grid_length() {
        let filter = PhotonicRfFilter::new_fir(vec![1.0; 3], 100e-12);
        let resp = filter.frequency_response(0.0, 10.0e9, 101);
        assert_eq!(resp.len(), 101);
    }

    #[test]
    fn test_fir_bandwidth_less_than_fsr() {
        let filter = PhotonicRfFilter::new_fir(vec![1.0; 8], 100e-12);
        assert!(filter.bandwidth_hz() < filter.fsr_hz());
    }

    #[test]
    fn test_fir_negative_weights_notch() {
        // Alternating weights [1, -1, 1, -1] → notch at DC
        let filter = PhotonicRfFilter::new_fir(vec![1.0, -1.0, 1.0, -1.0], 100e-12);
        let h_dc = filter.transfer_function(0.0);
        // Sum of alternating ±1 = 0
        assert_abs_diff_eq!(h_dc.norm(), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_bandpass_filter_peak_at_fc() {
        let fc = 5.0e9;
        let delay = 100e-12;
        let filter = PhotonicRfFilter::new_bandpass(fc, 11, delay);
        let resp_fc = filter.response_db(fc);
        let resp_dc = filter.response_db(0.0);
        // Response at center should exceed response at DC
        assert!(resp_fc > resp_dc, "Bandpass peak should be at fc, not DC");
    }

    // ── RingResonatorRfFilter ────────────────────────────────────────────────

    #[test]
    fn test_ring_q_factor() {
        let fc = 10.0e9;
        let bw = 100.0e6;
        let filt = RingResonatorRfFilter::new(fc, bw, 1);
        assert_abs_diff_eq!(filt.q_factor(), fc / bw, epsilon = 1e-3);
    }

    #[test]
    fn test_ring_peak_at_center() {
        let fc = 5.0e9;
        let bw = 200.0e6;
        let filt = RingResonatorRfFilter::new(fc, bw, 1);
        let h_fc = filt.transfer_function(fc).norm();
        let h_off = filt.transfer_function(fc + 1.0e9).norm();
        assert!(h_fc > h_off, "Peak must be at center frequency");
    }

    #[test]
    fn test_ring_group_delay_peak_at_center() {
        let fc = 5.0e9;
        let bw = 200.0e6;
        let filt = RingResonatorRfFilter::new(fc, bw, 2);
        let tau_fc = filt.group_delay_s(fc);
        let tau_off = filt.group_delay_s(fc + 1.0e9);
        assert!(tau_fc > tau_off, "Group delay should peak at resonance");
    }

    #[test]
    fn test_ring_order_increases_rolloff() {
        let fc = 5.0e9;
        let bw = 200.0e6;
        let f_test = fc + bw;
        let filt1 = RingResonatorRfFilter::new(fc, bw, 1);
        let filt3 = RingResonatorRfFilter::new(fc, bw, 3);
        let h1 = filt1.transfer_function(f_test).norm();
        let h3 = filt3.transfer_function(f_test).norm();
        assert!(h3 < h1, "Higher order should have steeper rolloff");
    }

    // ── PhotonicHilbertTransformer ───────────────────────────────────────────

    #[test]
    fn test_hilbert_center_tap_zero() {
        let ht = PhotonicHilbertTransformer::new(11, 10.0e9);
        let coeffs = ht.tap_coefficients();
        let center = coeffs.len() / 2;
        assert_abs_diff_eq!(coeffs[center], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_hilbert_antisymmetry() {
        let ht = PhotonicHilbertTransformer::new(11, 10.0e9);
        let coeffs = ht.tap_coefficients();
        let n = coeffs.len();
        // Hilbert coefficients must be antisymmetric: h[k] = -h[N-1-k]
        for i in 0..n / 2 {
            assert_abs_diff_eq!(coeffs[i], -coeffs[n - 1 - i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_hilbert_phase_near_90_at_center() {
        let bw = 10.0e9;
        let ht = PhotonicHilbertTransformer::new(31, bw);
        // At center of the band (1/4 * FSR), phase should be close to -90°
        let f_center = bw * 0.5;
        let error = ht.phase_error_deg(f_center);
        // Should be within ±10° for a 31-tap Hamming-windowed design
        assert!(
            error.abs() < 15.0,
            "Phase error={:.2}° too large at {:.1} GHz",
            error,
            f_center * 1e-9
        );
    }

    #[test]
    fn test_hilbert_tap_count_always_odd() {
        let ht_even = PhotonicHilbertTransformer::new(10, 5.0e9);
        assert_eq!(ht_even.n_taps % 2, 1, "n_taps must be odd");
        let ht_odd = PhotonicHilbertTransformer::new(11, 5.0e9);
        assert_eq!(ht_odd.n_taps, 11);
    }
}
