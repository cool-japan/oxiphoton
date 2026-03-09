//! Advanced optical modulation formats for coherent photonic communications.
//!
//! Includes QAM constellation geometry, probabilistic shaping, DP-QPSK,
//! DP-16QAM, and OFDM modulation.

use num_complex::Complex64;

// ---------------------------------------------------------------------------
// Helper: Q-function and erfc approximation (Abramowitz & Stegun 7.1.26)
// ---------------------------------------------------------------------------

/// Complementary error function approximation (Abramowitz & Stegun 7.1.26).
/// Maximum absolute error < 1.5 × 10⁻⁷.
fn erfc_as(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc_as(-x);
    }
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    poly * (-(x * x)).exp()
}

/// Q-function: Q(x) = erfc(x/√2) / 2.
fn q_func(x: f64) -> f64 {
    erfc_as(x / 2.0_f64.sqrt()) / 2.0
}

// ---------------------------------------------------------------------------
// Constellation point
// ---------------------------------------------------------------------------

/// A single point in a QAM constellation.
#[derive(Debug, Clone, Copy)]
pub struct ConstellationPoint {
    /// In-phase coordinate (normalised).
    pub i: f64,
    /// Quadrature coordinate (normalised).
    pub q: f64,
    /// Gray-coded bit label for this point.
    pub bits: u32,
}

impl ConstellationPoint {
    /// Complex representation of the symbol.
    pub fn symbol(&self) -> Complex64 {
        Complex64::new(self.i, self.q)
    }

    /// Squared Euclidean distance to another point.
    pub fn distance_sq(&self, other: &Self) -> f64 {
        let di = self.i - other.i;
        let dq = self.q - other.q;
        di * di + dq * dq
    }
}

// ---------------------------------------------------------------------------
// Gray code helper
// ---------------------------------------------------------------------------

/// Convert binary index to Gray code.
fn to_gray(n: u32) -> u32 {
    n ^ (n >> 1)
}

// ---------------------------------------------------------------------------
// QAM constellation
// ---------------------------------------------------------------------------

/// Square M-QAM constellation (M = 4, 16, 64, 256, 1024, …).
///
/// The constellation is built on a square grid with Gray coding.
#[derive(Debug, Clone)]
pub struct QamConstellation {
    /// Modulation order M (must be a perfect square and a power of 4).
    pub order: usize,
    /// All M constellation points.
    pub points: Vec<ConstellationPoint>,
    /// Peak power (max |symbol|²).
    pub peak_power: f64,
}

impl QamConstellation {
    /// Construct an M-QAM constellation with average power = 1.
    ///
    /// `order` must be a power of 4 (4, 16, 64, 256, …).
    pub fn new(order: usize) -> Self {
        assert!(
            order >= 4 && (order as f64).sqrt().fract() == 0.0,
            "QAM order must be a perfect square ≥ 4"
        );
        let sqrt_m = (order as f64).sqrt() as usize;
        let bits_per_dim = (sqrt_m as f64).log2() as u32;
        let levels: Vec<f64> = (0..sqrt_m)
            .map(|k| -((sqrt_m as f64) - 1.0) + 2.0 * k as f64)
            .collect();

        let mut points = Vec::with_capacity(order);
        for (qi, &q_val) in levels.iter().enumerate() {
            for (ii, &i_val) in levels.iter().enumerate() {
                // Gray code per dimension, then interleave
                let gray_i = to_gray(ii as u32);
                let gray_q = to_gray(qi as u32);
                // Interleave bits: I bits occupy even bit positions, Q odd
                let mut bits = 0u32;
                for b in 0..bits_per_dim {
                    bits |= ((gray_i >> b) & 1) << (2 * b);
                    bits |= ((gray_q >> b) & 1) << (2 * b + 1);
                }
                points.push(ConstellationPoint {
                    i: i_val,
                    q: q_val,
                    bits,
                });
            }
        }

        let avg_power: f64 =
            points.iter().map(|p| p.i * p.i + p.q * p.q).sum::<f64>() / order as f64;
        let norm = avg_power.sqrt().max(1e-30);
        for pt in &mut points {
            pt.i /= norm;
            pt.q /= norm;
        }

        let peak_power = points
            .iter()
            .map(|p| p.i * p.i + p.q * p.q)
            .fold(0.0_f64, f64::max);

        Self {
            order,
            points,
            peak_power,
        }
    }

    /// Normalise the constellation so that average symbol power = 1.
    pub fn normalize(&mut self) {
        let avg_power: f64 = self
            .points
            .iter()
            .map(|p| p.i * p.i + p.q * p.q)
            .sum::<f64>()
            / self.order as f64;
        let norm = avg_power.sqrt().max(1e-30);
        for pt in &mut self.points {
            pt.i /= norm;
            pt.q /= norm;
        }
        self.peak_power = self
            .points
            .iter()
            .map(|p| p.i * p.i + p.q * p.q)
            .fold(0.0_f64, f64::max);
    }

    /// Modulate a bit sequence into a complex symbol.
    ///
    /// `bits` must have exactly log₂(M) elements (MSB first).
    pub fn modulate(&self, bits: &[bool]) -> Complex64 {
        let bps = (self.order as f64).log2() as usize;
        let mut label = 0u32;
        for (b, &bit) in bits.iter().enumerate().take(bps) {
            if bit {
                label |= 1 << (bps - 1 - b);
            }
        }
        // Find the point whose Gray-coded label matches
        self.points
            .iter()
            .find(|p| p.bits == label)
            .map(|p| p.symbol())
            .unwrap_or_else(|| self.points[label as usize % self.points.len()].symbol())
    }

    /// Demodulate a received complex symbol by nearest-neighbour decision.
    ///
    /// Returns log₂(M) bits (MSB first).
    pub fn demodulate(&self, symbol: Complex64) -> Vec<bool> {
        let bps = (self.order as f64).log2() as usize;
        let nearest = self
            .points
            .iter()
            .min_by(|a, b| {
                let da = (a.symbol() - symbol).norm_sqr();
                let db = (b.symbol() - symbol).norm_sqr();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(&self.points[0]);
        (0..bps)
            .rev()
            .map(|b| ((nearest.bits >> b) & 1) == 1)
            .collect()
    }

    /// Symbol error rate for AWGN channel.
    ///
    /// Approximation: P_s ≈ 4·(1 − 1/√M)·Q(√(3·SNR/(M−1)))
    pub fn ser_awgn(&self, snr_linear: f64) -> f64 {
        let m = self.order as f64;
        let arg = (3.0 * snr_linear / (m - 1.0)).sqrt();
        4.0 * (1.0 - 1.0 / m.sqrt()) * q_func(arg)
    }

    /// Bit error rate for AWGN channel with Gray coding.
    ///
    /// `snr_per_bit_db` is Eb/N₀ in dB.
    pub fn ber_awgn(&self, snr_per_bit_db: f64) -> f64 {
        let bps = (self.order as f64).log2();
        let eb_n0 = 10.0_f64.powf(snr_per_bit_db / 10.0);
        let snr = eb_n0 * bps;
        self.ser_awgn(snr) / bps
    }

    /// Spectral efficiency: log₂(M) bits per symbol per Hz.
    pub fn spectral_efficiency_bps_per_hz(&self) -> f64 {
        (self.order as f64).log2()
    }

    /// Peak-to-Average Power Ratio of the constellation in dB.
    pub fn papr_db(&self) -> f64 {
        if self.peak_power > 0.0 {
            10.0 * self.peak_power.log10()
        } else {
            0.0
        }
    }

    /// Error Vector Magnitude (EVM) as a percentage.
    ///
    /// EVM% = 100 · √(mean(|r_k − s_k|²) / mean(|s_k|²))
    pub fn evm_percent(&self, symbols: &[Complex64], reference: &[Complex64]) -> f64 {
        if symbols.is_empty() || symbols.len() != reference.len() {
            return 0.0;
        }
        let n = symbols.len() as f64;
        let error_power: f64 = symbols
            .iter()
            .zip(reference.iter())
            .map(|(&s, &r)| (s - r).norm_sqr())
            .sum::<f64>()
            / n;
        let ref_power: f64 = reference.iter().map(|r| r.norm_sqr()).sum::<f64>() / n;
        if ref_power > 0.0 {
            100.0 * (error_power / ref_power).sqrt()
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Probabilistic constellation shaping
// ---------------------------------------------------------------------------

/// Probabilistically shaped constellation using a Maxwell-Boltzmann (MB)
/// distribution over the base QAM constellation.
///
/// P_k ∝ exp(−ν · |a_k|²), where ν is the shaping parameter (inverse
/// temperature).  Higher ν concentrates probability on low-energy points.
#[derive(Debug, Clone)]
pub struct ShapedConstellation {
    /// Underlying square QAM constellation.
    pub base_constellation: QamConstellation,
    /// Shaping parameter ν (≥ 0; ν = 0 → uniform, ν → ∞ → single point).
    pub nu: f64,
    /// Maxwell-Boltzmann probability of each constellation point.
    pub probabilities: Vec<f64>,
}

impl ShapedConstellation {
    /// Construct a shaped M-QAM constellation with shaping parameter `nu`.
    pub fn new(order: usize, nu: f64) -> Self {
        let base = QamConstellation::new(order);
        let weights: Vec<f64> = base
            .points
            .iter()
            .map(|p| (-nu * (p.i * p.i + p.q * p.q)).exp())
            .collect();
        let z: f64 = weights.iter().sum::<f64>().max(1e-300);
        let probabilities = weights.iter().map(|&w| w / z).collect();
        Self {
            base_constellation: base,
            nu,
            probabilities,
        }
    }

    /// Shannon entropy of the shaped distribution (bits per symbol).
    pub fn entropy_bits_per_symbol(&self) -> f64 {
        self.probabilities
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.log2())
            .sum()
    }

    /// Shaping gain relative to uniform distribution (dB).
    ///
    /// Approximated as the reduction in average symbol energy (dB).
    pub fn shaping_gain_db(&self) -> f64 {
        let avg_power_shaped = self.average_power();
        let avg_power_uniform: f64 = self
            .base_constellation
            .points
            .iter()
            .map(|p| p.i * p.i + p.q * p.q)
            .sum::<f64>()
            / self.base_constellation.order as f64;
        if avg_power_shaped > 0.0 && avg_power_uniform > 0.0 {
            10.0 * (avg_power_uniform / avg_power_shaped).log10()
        } else {
            0.0
        }
    }

    /// Effective SNR gain from shaping at a given operating SNR (dB).
    ///
    /// Simplified model: gain ≈ shaping_gain_db / 2 (half of power reduction
    /// translates to SNR improvement in Gaussian noise channels).
    pub fn effective_snr_gain_db(&self, _snr_db: f64) -> f64 {
        self.shaping_gain_db() / 2.0
    }

    /// Weighted average symbol power under the MB distribution.
    pub fn average_power(&self) -> f64 {
        self.base_constellation
            .points
            .iter()
            .zip(self.probabilities.iter())
            .map(|(p, &prob)| prob * (p.i * p.i + p.q * p.q))
            .sum()
    }
}

// ---------------------------------------------------------------------------
// DP-QPSK
// ---------------------------------------------------------------------------

/// Dual-Polarisation QPSK modulation format helper.
pub struct DpQpsk;

impl DpQpsk {
    /// Symbol rate in GBaud for a given bit rate in Gbps.
    ///
    /// DP-QPSK carries 4 bits/symbol (2 pol × 2 bits), so:
    /// R_sym = R_bit / 4.
    pub fn symbol_rate_gbaud(bit_rate_gbps: f64) -> f64 {
        bit_rate_gbps / 4.0
    }

    /// Spectral efficiency: 4 bits/symbol.
    pub fn spectral_efficiency() -> f64 {
        4.0
    }

    /// BER from OSNR for DP-QPSK (coherent detection, no FEC).
    ///
    /// BER ≈ erfc(√(OSNR·B_ref/(2·B_sym))) / 2
    /// where B_ref = 12.5 GHz (0.1 nm reference bandwidth at 1550 nm).
    pub fn ber_from_osnr(osnr_db: f64, baud_rate_gbaud: f64) -> f64 {
        let osnr = 10.0_f64.powf(osnr_db / 10.0);
        let b_ref = 12.5e9; // 0.1 nm reference bandwidth
        let b_sym = baud_rate_gbaud * 1e9;
        let snr_per_bit = osnr * b_ref / (2.0 * b_sym);
        erfc_as(snr_per_bit.sqrt()) / 2.0
    }

    /// Required OSNR (dB) for a given target BER.
    ///
    /// Inverted analytically: SNR = [erfinv(1-2·BER)]² → OSNR from SNR.
    pub fn required_osnr_db(target_ber: f64) -> f64 {
        use crate::photonic_dsp::dsp_algorithms::erfinv_approx;
        let b_ref = 12.5e9;
        let b_sym = 32e9; // default 32 GBaud
        let q = erfinv_approx(1.0 - 2.0 * target_ber);
        let snr_per_bit = q * q;
        10.0 * (snr_per_bit * 2.0 * b_sym / b_ref).log10()
    }
}

// ---------------------------------------------------------------------------
// DP-16QAM
// ---------------------------------------------------------------------------

/// Dual-Polarisation 16QAM modulation format helper.
pub struct Dp16Qam;

impl Dp16Qam {
    /// Spectral efficiency: 8 bits/symbol (2 pol × 4 bits).
    pub fn spectral_efficiency() -> f64 {
        8.0
    }

    /// BER from OSNR for DP-16QAM.
    ///
    /// BER ≈ (3/8) · erfc(√(OSNR · B_ref / (5 · B_sym)))
    pub fn ber_from_osnr(osnr_db: f64, baud_rate_gbaud: f64) -> f64 {
        let osnr = 10.0_f64.powf(osnr_db / 10.0);
        let b_ref = 12.5e9;
        let b_sym = baud_rate_gbaud * 1e9;
        let snr_eff = osnr * b_ref / (5.0 * b_sym);
        (3.0 / 8.0) * erfc_as(snr_eff.sqrt())
    }

    /// Required OSNR (dB) for a given target BER (16QAM, 32 GBaud default).
    pub fn required_osnr_db(target_ber: f64) -> f64 {
        let b_ref = 12.5e9;
        let b_sym = 32e9;
        // Invert: target_ber = (3/8)·erfc(√x)  →  erfc(√x) = 8/3·target_ber
        let erfc_val = (8.0 / 3.0) * target_ber;
        // Use erfinv approximation: erfc(x) = 1 - erf(x); need erfc⁻¹
        // erfc(√x) = v → √x = erfc⁻¹(v); erfc⁻¹(v) = erfinv(1-v)
        use crate::photonic_dsp::dsp_algorithms::erfinv_approx;
        let sqrt_x = erfinv_approx(1.0 - erfc_val).max(0.0);
        let snr_eff = sqrt_x * sqrt_x;
        10.0 * (snr_eff * 5.0 * b_sym / b_ref).log10()
    }

    /// Probabilistic shaping gain for DP-16QAM (dB) as a function of ν.
    ///
    /// Approximated as: gain ≈ (1.53 · ν) / (log₂(16) / 2) dB.
    pub fn shaping_gain_db(nu: f64) -> f64 {
        // For 16-QAM: theoretical max shaping gain ≈ 1.53 dB
        // Linear model with saturation at theoretical limit
        let max_gain = 1.53_f64;
        max_gain * (1.0 - (-nu).exp())
    }
}

// ---------------------------------------------------------------------------
// OFDM modulator
// ---------------------------------------------------------------------------

/// Optical OFDM (Orthogonal Frequency Division Multiplexing) modulator.
///
/// Models key OFDM parameters for WDM-OFDM superchannels.
#[derive(Debug, Clone)]
pub struct OfdmModulator {
    /// Number of data subcarriers.
    pub n_subcarriers: usize,
    /// Subcarrier spacing in GHz.
    pub subcarrier_spacing_ghz: f64,
    /// Cyclic prefix length as a fraction of the OFDM symbol duration.
    pub cyclic_prefix_fraction: f64,
    /// QAM order carried on each subcarrier.
    pub qam_order: usize,
}

impl OfdmModulator {
    /// Construct a new OFDM modulator.
    pub fn new(n_sc: usize, spacing_ghz: f64, cp_frac: f64, qam_order: usize) -> Self {
        Self {
            n_subcarriers: n_sc,
            subcarrier_spacing_ghz: spacing_ghz,
            cyclic_prefix_fraction: cp_frac,
            qam_order,
        }
    }

    /// OFDM symbol duration (useful part only) in picoseconds.
    pub fn symbol_duration_ps(&self) -> f64 {
        // T_u = 1 / Δf
        1e3 / self.subcarrier_spacing_ghz // 1/(GHz) in ps
    }

    /// Cyclic prefix duration in picoseconds.
    pub fn cp_duration_ps(&self) -> f64 {
        self.symbol_duration_ps() * self.cyclic_prefix_fraction
    }

    /// Total OFDM signal bandwidth in GHz (subcarrier_spacing × n_subcarriers).
    pub fn total_bandwidth_ghz(&self) -> f64 {
        self.subcarrier_spacing_ghz * self.n_subcarriers as f64
    }

    /// Spectral efficiency in bits/s/Hz.
    ///
    /// SE = log₂(M) · N_sc / (N_sc + N_cp) / (1 + CP_fraction)
    pub fn spectral_efficiency_bps_per_hz(&self) -> f64 {
        let bps = (self.qam_order as f64).log2();
        bps / (1.0 + self.cyclic_prefix_fraction)
    }

    /// OFDM PAPR in dB.
    ///
    /// Theoretical worst-case: PAPR ≈ 10·log₁₀(N_sc).
    pub fn papr_db(&self) -> f64 {
        10.0 * (self.n_subcarriers as f64).log10()
    }

    /// BER in AWGN for the underlying QAM on each subcarrier.
    ///
    /// `snr_per_bit_db` is the Eb/N₀ in dB.
    pub fn ber_awgn(&self, snr_per_bit_db: f64) -> f64 {
        let qam = QamConstellation::new(self.qam_order);
        qam.ber_awgn(snr_per_bit_db)
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
    fn qpsk_spectral_efficiency() {
        let qam = QamConstellation::new(4);
        assert_abs_diff_eq!(qam.spectral_efficiency_bps_per_hz(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn qam16_demodulate_roundtrip() {
        let qam = QamConstellation::new(16);
        let bits_in: Vec<bool> = vec![false, true, true, false]; // some 4-bit label
        let symbol = qam.modulate(&bits_in);
        let bits_out = qam.demodulate(symbol);
        assert_eq!(bits_in, bits_out);
    }

    #[test]
    fn qam_ser_high_snr() {
        // At very high SNR, SER should approach zero
        let qam = QamConstellation::new(16);
        let ser = qam.ser_awgn(1000.0);
        assert!(ser < 1e-10, "SER at high SNR: {}", ser);
    }

    #[test]
    fn shaped_constellation_entropy() {
        // Uniform (nu=0) → entropy = log2(M)
        let shaped = ShapedConstellation::new(16, 0.0);
        let expected_entropy = 4.0_f64; // log2(16)
        assert_abs_diff_eq!(
            shaped.entropy_bits_per_symbol(),
            expected_entropy,
            epsilon = 1e-6
        );
    }

    #[test]
    fn shaped_constellation_shaping_gain_positive() {
        // With positive nu, shaped average power < uniform average power
        let shaped = ShapedConstellation::new(16, 1.0);
        let gain = shaped.shaping_gain_db();
        assert!(
            gain >= 0.0,
            "Shaping gain should be non-negative, got {}",
            gain
        );
    }

    #[test]
    fn dp_qpsk_symbol_rate() {
        assert_abs_diff_eq!(DpQpsk::symbol_rate_gbaud(100.0), 25.0, epsilon = 1e-10);
    }

    #[test]
    fn ofdm_bandwidth() {
        let ofdm = OfdmModulator::new(128, 0.125, 0.25, 16);
        assert_abs_diff_eq!(ofdm.total_bandwidth_ghz(), 16.0, epsilon = 1e-6);
    }

    #[test]
    fn ofdm_papr() {
        let ofdm = OfdmModulator::new(128, 0.125, 0.25, 16);
        // 10 * log10(128) ≈ 21.07 dB
        assert_abs_diff_eq!(ofdm.papr_db(), 21.072, epsilon = 0.01);
    }

    #[test]
    fn evm_zero_noise() {
        let qam = QamConstellation::new(4);
        let symbols: Vec<Complex64> = qam.points.iter().map(|p| p.symbol()).collect();
        let evm = qam.evm_percent(&symbols, &symbols);
        assert_abs_diff_eq!(evm, 0.0, epsilon = 1e-10);
    }
}
