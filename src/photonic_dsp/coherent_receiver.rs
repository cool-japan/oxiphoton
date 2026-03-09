//! Coherent optical receiver DSP chain.
//!
//! Models the 90° optical hybrid, digital carrier phase estimation,
//! frequency offset estimation and compensation, chromatic dispersion
//! compensation (frequency-domain), and polarisation demultiplexing via the
//! constant modulus algorithm.

use num_complex::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// 90° optical hybrid
// ---------------------------------------------------------------------------

/// 90° optical hybrid used at the front-end of a coherent receiver.
///
/// Mixes the received signal `E_s` with a local oscillator `E_lo` to
/// produce four quadrature outputs.  Non-ideal behaviour is captured by
/// insertion loss, phase imbalance, and amplitude imbalance.
#[derive(Debug, Clone)]
pub struct OpticalHybrid {
    /// Insertion loss in dB (≥ 0).
    pub insertion_loss_db: f64,
    /// Phase imbalance of the 90° branch in degrees (ideal = 0°).
    pub phase_imbalance_deg: f64,
    /// Amplitude imbalance between I and Q arms in dB (ideal = 0 dB).
    pub amplitude_imbalance_db: f64,
}

impl OpticalHybrid {
    /// Construct an ideal 90° hybrid (no losses or imbalances).
    pub fn new() -> Self {
        Self {
            insertion_loss_db: 0.0,
            phase_imbalance_deg: 0.0,
            amplitude_imbalance_db: 0.0,
        }
    }

    /// Compute the four detector output fields of the 90° hybrid.
    ///
    /// Output ordering: [I+, I−, Q+, Q−] (balanced detector pairs).
    pub fn outputs(&self, signal: Complex64, lo: Complex64) -> [Complex64; 4] {
        let loss = 10.0_f64.powf(-self.insertion_loss_db / 20.0);
        let phase_err = self.phase_imbalance_deg.to_radians();
        let amp_imb = 10.0_f64.powf(self.amplitude_imbalance_db / 20.0);

        // 3-dB splitting factor
        let split = (0.5_f64).sqrt() * loss;
        // Ideal 90° phase shift for Q branch, perturbed by imbalance
        let q_phase = Complex64::new(0.0, PI / 2.0 + phase_err).exp();

        let e_s = signal * split;
        let e_lo = lo * split;

        // I branch: E_s + E_lo  and  E_s − E_lo
        let i_plus = e_s + e_lo;
        let i_minus = e_s - e_lo;

        // Q branch: E_s + j·E_lo  (with phase and amplitude imbalances)
        let e_lo_q = lo * split * amp_imb * q_phase;
        let q_plus = e_s + e_lo_q;
        let q_minus = e_s - e_lo_q;

        [i_plus, i_minus, q_plus, q_minus]
    }

    /// Extract the complex IQ signal after balanced detection.
    ///
    /// Returns I + j·Q where I and Q are the balanced detector outputs.
    pub fn iq_signal(&self, signal: Complex64, lo: Complex64) -> Complex64 {
        let [i_plus, i_minus, q_plus, q_minus] = self.outputs(signal, lo);
        let i_bal = i_plus - i_minus;
        let q_bal = q_plus - q_minus;
        Complex64::new(i_bal.re, q_bal.re)
    }

    /// Common-mode rejection ratio (CMRR) in dB estimated from imbalances.
    ///
    /// CMRR ≈ 20·log₁₀(1 / sin(Δφ/2)) for small phase imbalance Δφ.
    pub fn cmrr_db(&self) -> f64 {
        let phi = self.phase_imbalance_deg.to_radians();
        let amp = 10.0_f64.powf(self.amplitude_imbalance_db / 20.0);
        // Combined CMRR (approximate)
        let cmrr_linear = 1.0 / ((phi / 2.0).sin().abs().max(1e-10) + (amp - 1.0).abs().max(1e-10));
        20.0 * cmrr_linear.log10()
    }
}

impl Default for OpticalHybrid {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Carrier phase estimation (Viterbi–Viterbi)
// ---------------------------------------------------------------------------

/// Digital carrier phase estimation using the Viterbi–Viterbi (VV) algorithm.
///
/// Estimates the laser phase noise by raising received QPSK/M-PSK symbols to
/// the M-th power to remove data modulation, then averaging the argument.
#[derive(Debug, Clone)]
pub struct CarrierPhaseEstimation {
    /// Number of symbols averaged per block.
    pub block_size: usize,
    /// PSK order used for phase removal (4 for QPSK, 8 for 8PSK).
    pub m_psk_order: usize,
}

impl CarrierPhaseEstimation {
    /// Construct a new Viterbi–Viterbi phase estimator.
    pub fn new(block_size: usize, order: usize) -> Self {
        Self {
            block_size,
            m_psk_order: order,
        }
    }

    /// Estimate the carrier phase from a block of received symbols.
    ///
    /// φ̂ = (1/M) · arg(Σ_{k=1}^{N} x_k^M)
    pub fn estimate_phase(&self, symbols: &[Complex64]) -> f64 {
        if symbols.is_empty() {
            return 0.0;
        }
        let m = self.m_psk_order as f64;
        let n = symbols.len().min(self.block_size);
        let sum = symbols[..n]
            .iter()
            .fold(Complex64::new(0.0, 0.0), |acc, &x| {
                acc + complex_pow(x, self.m_psk_order)
            });
        sum.im.atan2(sum.re) / m
    }

    /// Apply phase correction to a sequence of symbols.
    ///
    /// The block-averaged phase estimate is used to rotate each symbol in the
    /// block.
    pub fn correct(&self, symbols: &[Complex64]) -> Vec<Complex64> {
        symbols
            .chunks(self.block_size)
            .flat_map(|block| {
                let phi = self.estimate_phase(block);
                let _correction = Complex64::new(-phi.cos(), -phi.sin()); // exp(-jφ)
                                                                          // Use proper rotation: multiply by exp(-jφ)
                let rot = Complex64::new((-phi).cos(), (-phi).sin());
                block.iter().map(move |&s| s * rot).collect::<Vec<_>>()
            })
            .collect()
    }

    /// Maximum tolerable laser linewidth (Hz) for this estimator.
    ///
    /// Δν_max ≈ (β_2 / π) · R_sym² / (M² − 1)
    /// where β_2 = 1/(2·N·block_size) is the filter bandwidth fraction.
    pub fn max_linewidth_hz(&self, symbol_rate_gbaud: f64) -> f64 {
        let r_sym = symbol_rate_gbaud * 1e9;
        let m = self.m_psk_order as f64;
        let n = self.block_size as f64;
        // Simplified: Δν_max ≈ R_sym / (2π · N · (M²-1))
        r_sym / (2.0 * PI * n * (m * m - 1.0))
    }
}

/// Raise a complex number to an integer power.
fn complex_pow(z: Complex64, n: usize) -> Complex64 {
    if n == 0 {
        return Complex64::new(1.0, 0.0);
    }
    let mut result = Complex64::new(1.0, 0.0);
    let mut base = z;
    let mut exp = n;
    while exp > 0 {
        if exp & 1 == 1 {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }
    result
}

// ---------------------------------------------------------------------------
// Frequency offset estimation
// ---------------------------------------------------------------------------

/// Frequency offset estimator for coherent receivers.
///
/// Uses the 4th-power method to estimate the carrier frequency offset between
/// transmitter and local oscillator lasers.
#[derive(Debug, Clone)]
pub struct FrequencyOffsetEstimator {
    /// Maximum expected frequency offset in GHz.
    pub max_offset_ghz: f64,
    /// Number of symbols used for averaging.
    pub n_symbols: usize,
}

impl FrequencyOffsetEstimator {
    /// Construct a new frequency offset estimator.
    pub fn new(max_offset_ghz: f64) -> Self {
        Self {
            max_offset_ghz,
            n_symbols: 1024,
        }
    }

    /// Estimate the frequency offset from QPSK symbols using the 4th-power
    /// differential method.
    ///
    /// Δf ≈ (1/(4·2π·T)) · arg(Σ x_k^4 · conj(x_{k-1}^4))
    pub fn estimate_offset_hz(&self, symbols: &[Complex64]) -> f64 {
        if symbols.len() < 2 {
            return 0.0;
        }
        let n = symbols.len().min(self.n_symbols);
        let sum: Complex64 = (1..n).fold(Complex64::new(0.0, 0.0), |acc, k| {
            acc + complex_pow(symbols[k], 4) * complex_pow(symbols[k - 1], 4).conj()
        });
        sum.im.atan2(sum.re) / (4.0 * 2.0 * PI)
        // Note: caller should divide by symbol period T for Hz
    }

    /// Apply frequency offset compensation to the symbol sequence.
    ///
    /// Each symbol k is rotated by exp(−j·2π·Δf·k·T):
    /// x_k_comp = x_k · exp(−j·2π·Δf·k·T)
    pub fn compensate(
        &self,
        symbols: &[Complex64],
        offset_hz: f64,
        symbol_rate: f64,
    ) -> Vec<Complex64> {
        let t_sym = if symbol_rate > 0.0 {
            1.0 / symbol_rate
        } else {
            1e-12
        };
        symbols
            .iter()
            .enumerate()
            .map(|(k, &s)| {
                let phase = -2.0 * PI * offset_hz * k as f64 * t_sym;
                s * Complex64::new(phase.cos(), phase.sin())
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Chromatic dispersion compensator
// ---------------------------------------------------------------------------

/// Frequency-domain chromatic dispersion compensator.
///
/// Applies the dispersion transfer function H_CD(ω) = exp(j·β₂/2·ω²·L) to
/// the received signal in the frequency domain, effectively unwinding the
/// group-velocity dispersion accumulated over the fibre span.
#[derive(Debug, Clone)]
pub struct CdCompensator {
    /// Total accumulated dispersion coefficient D in ps/nm.
    pub dispersion_ps_per_nm: f64,
    /// Fibre span length in km.
    pub fiber_length_km: f64,
    /// Symbol rate in GBaud.
    pub symbol_rate_gbaud: f64,
    /// Centre wavelength in metres.
    pub center_wavelength: f64,
}

impl CdCompensator {
    /// Construct a new CD compensator.
    pub fn new(disp: f64, length_km: f64, rate_gbaud: f64, wavelength: f64) -> Self {
        Self {
            dispersion_ps_per_nm: disp,
            fiber_length_km: length_km,
            symbol_rate_gbaud: rate_gbaud,
            center_wavelength: wavelength,
        }
    }

    /// Group-velocity dispersion parameter β₂ in s²/m derived from D·L.
    fn beta2_s2_per_m(&self) -> f64 {
        // D [ps/(nm·km)] = −(λ²/2πc) · β₂
        // β₂ = −D · λ² / (2π·c)
        // D·L [ps/nm] → convert to [s/m]: ×1e-12 / (1e-9 · 1e3) = ×1e-6
        let d_si = self.dispersion_ps_per_nm * 1e-12 / (1e-9 * 1e3); // s/(m·m)
        let lambda = self.center_wavelength;
        let c = 3e8_f64;
        -d_si * lambda * lambda / (2.0 * PI * c) * self.fiber_length_km * 1e3
    }

    /// CD compensation transfer function H_CD(ω) = exp(j·β₂/2·ω²·L).
    ///
    /// `omega` is the angular frequency offset from the carrier (rad/s).
    pub fn transfer_function(&self, omega: f64) -> Complex64 {
        let beta2_l = self.beta2_s2_per_m(); // β₂ · L (s²)
        let phase = beta2_l / 2.0 * omega * omega;
        Complex64::new(phase.cos(), phase.sin())
    }

    /// Apply CD compensation in the frequency domain.
    ///
    /// Uses a simple DFT-based approach: transform → multiply → inverse.
    pub fn compensate(&self, signal: &[Complex64]) -> Vec<Complex64> {
        use crate::photonic_dsp::dsp_algorithms::{dft, idft};
        let n = signal.len();
        if n == 0 {
            return Vec::new();
        }
        let spectrum = dft(signal);
        let fs = self.symbol_rate_gbaud * 1e9; // sample rate in Hz
        let df = fs / n as f64;
        let compensated: Vec<Complex64> = spectrum
            .iter()
            .enumerate()
            .map(|(k, &s)| {
                let freq = if k <= n / 2 {
                    k as f64 * df
                } else {
                    (k as f64 - n as f64) * df
                };
                let omega = 2.0 * PI * freq;
                s * self.transfer_function(omega)
            })
            .collect();
        idft(&compensated)
    }

    /// Number of time-domain taps required for an equivalent FIR equalizer.
    ///
    /// N_taps ≈ |D| · Δλ · f_sym · L + 1
    /// where Δλ is the signal bandwidth in nm.
    pub fn required_taps(&self) -> usize {
        let d = self.dispersion_ps_per_nm.abs();
        let fs = self.symbol_rate_gbaud; // GBaud
        let lambda_nm = self.center_wavelength * 1e9;
        let c_nm_ps = 3e8 * 1e9 / 1e12; // speed of light in nm/ps
        let bw_nm = fs * 1e9 * lambda_nm * lambda_nm / (c_nm_ps * 1e12);
        let taps = (d * bw_nm * self.fiber_length_km).ceil() as usize + 1;
        taps.max(1)
    }
}

// ---------------------------------------------------------------------------
// Polarisation demultiplexer (CMA)
// ---------------------------------------------------------------------------

/// 2×2 polarisation demultiplexer using the constant modulus algorithm.
///
/// Tracks the fibre's time-varying polarisation rotation and birefringence
/// to separate X and Y polarisation tributaries.
#[derive(Debug, Clone)]
pub struct PolDemux {
    /// Current 2×2 Jones matrix equalizer [W_xx, W_xy; W_yx, W_yy].
    pub jones_matrix: [[Complex64; 2]; 2],
    /// CMA step size.
    pub mu: f64,
    /// Number of training iterations used so far.
    pub n_iterations: usize,
}

impl PolDemux {
    /// Construct a new polarisation demultiplexer with identity initialisation.
    pub fn new(mu: f64) -> Self {
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        Self {
            jones_matrix: [[one, zero], [zero, one]],
            mu,
            n_iterations: 0,
        }
    }

    /// Update the Jones matrix via the CMA for one symbol pair.
    ///
    /// `x`: received 2-component field [Ex, Ey].
    /// `y_desired`: target (ideal) 2-component field (not used in blind CMA,
    ///              but included for supervised/semi-blind variants).
    pub fn update_cma(&mut self, x: [Complex64; 2], _y_desired: [Complex64; 2]) {
        // Current outputs
        let y = self.apply(x);
        // CMA error for each output: e_k = y_k · (1 − |y_k|²)
        let e = [
            y[0] * (1.0 - y[0].norm_sqr()),
            y[1] * (1.0 - y[1].norm_sqr()),
        ];
        // Update W: W ← W + μ · e · x^H
        for (i, &e_i) in e.iter().enumerate() {
            for (j, &x_j) in x.iter().enumerate() {
                self.jones_matrix[i][j] += self.mu * e_i * x_j.conj();
            }
        }
        self.n_iterations += 1;
    }

    /// Apply the current Jones matrix to an input field pair.
    pub fn apply(&self, input: [Complex64; 2]) -> [Complex64; 2] {
        [
            self.jones_matrix[0][0] * input[0] + self.jones_matrix[0][1] * input[1],
            self.jones_matrix[1][0] * input[0] + self.jones_matrix[1][1] * input[1],
        ]
    }

    /// Check convergence: |W†W − I|_F < threshold.
    pub fn is_converged(&self, threshold: f64) -> bool {
        let w = &self.jones_matrix;
        // Compute W†W
        let wh_w_00 = w[0][0].conj() * w[0][0] + w[1][0].conj() * w[1][0];
        let wh_w_01 = w[0][0].conj() * w[0][1] + w[1][0].conj() * w[1][1];
        let wh_w_10 = w[0][1].conj() * w[0][0] + w[1][1].conj() * w[1][0];
        let wh_w_11 = w[0][1].conj() * w[0][1] + w[1][1].conj() * w[1][1];
        // Distance from identity: |W†W - I|_F²
        let fro_sq = (wh_w_00 - Complex64::new(1.0, 0.0)).norm_sqr()
            + wh_w_01.norm_sqr()
            + wh_w_10.norm_sqr()
            + (wh_w_11 - Complex64::new(1.0, 0.0)).norm_sqr();
        fro_sq.sqrt() < threshold
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn optical_hybrid_ideal_iq() {
        let hybrid = OpticalHybrid::new();
        // Pure real signal and LO → IQ should give real output
        let signal = Complex64::new(1.0, 0.0);
        let lo = Complex64::new(1.0, 0.0);
        let iq = hybrid.iq_signal(signal, lo);
        // I component should be non-zero (2 × Re{E_s · E_lo*})
        assert!(iq.re.abs() > 0.0 || iq.im.abs() > 0.0);
    }

    #[test]
    fn optical_hybrid_default() {
        let h1 = OpticalHybrid::new();
        let h2 = OpticalHybrid::default();
        assert_abs_diff_eq!(h1.insertion_loss_db, h2.insertion_loss_db, epsilon = 1e-15);
    }

    #[test]
    fn viterbi_viterbi_zero_phase() {
        // Symbols on QPSK constellation with zero phase noise → estimate should be ~0
        let symbols: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, -1.0),
        ];
        let cpe = CarrierPhaseEstimation::new(4, 4);
        let phase = cpe.estimate_phase(&symbols);
        // Sum of x^4 = 4 → arg = 0 → phase estimate = 0
        assert_abs_diff_eq!(phase, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn viterbi_viterbi_phase_correction() {
        let symbols: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, -1.0),
        ];
        let cpe = CarrierPhaseEstimation::new(4, 4);
        let corrected = cpe.correct(&symbols);
        assert_eq!(corrected.len(), symbols.len());
    }

    #[test]
    fn freq_offset_compensator_identity() {
        // Zero offset → symbols unchanged
        let symbols: Vec<Complex64> = (0..10).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let foe = FrequencyOffsetEstimator::new(10.0);
        let out = foe.compensate(&symbols, 0.0, 32e9);
        for (a, b) in symbols.iter().zip(out.iter()) {
            assert_abs_diff_eq!(a.re, b.re, epsilon = 1e-10);
        }
    }

    #[test]
    fn cd_compensator_transfer_function_unity_at_dc() {
        let cd = CdCompensator::new(1000.0, 80.0, 32.0, 1550e-9);
        let h0 = cd.transfer_function(0.0);
        assert_abs_diff_eq!(h0.norm(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn cd_compensator_required_taps_positive() {
        let cd = CdCompensator::new(1000.0, 80.0, 32.0, 1550e-9);
        assert!(cd.required_taps() >= 1);
    }

    #[test]
    fn pol_demux_identity_init() {
        let pd = PolDemux::new(1e-3);
        let input = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
        let output = pd.apply(input);
        assert_abs_diff_eq!(output[0].re, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(output[1].im, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn pol_demux_identity_converged() {
        let pd = PolDemux::new(1e-3);
        // Identity matrix satisfies W†W = I
        assert!(pd.is_converged(1e-10));
    }
}
