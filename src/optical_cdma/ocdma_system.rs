//! OCDMA transceiver models and multiple-access interference (MAI) analysis.
//!
//! Implements:
//! * **Incoherent OCDMA** — on-off keying (OOK) with unipolar OOC codewords.
//! * **Coherent OCDMA** — bipolar ±1 spreading codes with matched-filter
//!   correlation reception.
//! * **MAI analyser** — Gaussian-approximation BER and capacity estimation
//!   for systems with K simultaneous users.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Helper: Q-function
// ---------------------------------------------------------------------------

/// Q-function using the Abramowitz & Stegun 26.2.17 polynomial approximation.
///
/// `Q(x) = 0.5 · erfc(x / √2)`.  Error < 7.5 × 10⁻⁸ for x ≥ 0.
fn q_function(x: f64) -> f64 {
    crate::optical_cdma::performance::q_function(x)
}

// ---------------------------------------------------------------------------
// Incoherent OCDMA (OOK)
// ---------------------------------------------------------------------------

/// Incoherent OCDMA transceiver using an OOC unipolar codeword.
///
/// Bit "1" is transmitted as the codeword chip sequence; bit "0" is all
/// zeros.  The receiver integrates received chips over one symbol period and
/// compares against a threshold equal to the code weight `w`.
#[derive(Debug, Clone)]
pub struct IncoherentOcdma {
    /// OOC codeword (0/1 chip values), length n, weight w.
    pub code: Vec<u8>,
    /// Chip rate in chips per second.
    pub chip_rate_hz: f64,
    /// Optical power per chip when a "1" chip is transmitted (W).
    pub power_per_chip_w: f64,
}

impl IncoherentOcdma {
    /// Create a new incoherent OCDMA transceiver.
    pub fn new(code: Vec<u8>, chip_rate_hz: f64, power_per_chip_w: f64) -> Self {
        Self {
            code,
            chip_rate_hz,
            power_per_chip_w,
        }
    }

    /// Encode one information bit into a chip sequence.
    ///
    /// Bit "1" → codeword; bit "0" → all-zeros of the same length.
    pub fn encode(&self, bit: u8) -> Vec<u8> {
        if bit != 0 {
            self.code.clone()
        } else {
            vec![0u8; self.code.len()]
        }
    }

    /// Decode a received chip-level signal into an information bit.
    ///
    /// The correlator sums chip magnitudes and compares against threshold.
    /// Default threshold is the code weight minus 0.5 (midpoint between
    /// clean "0" and "1" decisions).
    pub fn decode(&self, received: &[f64]) -> u8 {
        let weight: usize = self.code.iter().map(|&b| b as usize).sum();
        let correlation: f64 = received
            .iter()
            .zip(self.code.iter())
            .map(|(&r, &c)| r * c as f64)
            .sum();
        // Threshold at weight - 0.5
        let threshold = weight as f64 - 0.5;
        if correlation >= threshold {
            1
        } else {
            0
        }
    }

    /// Information bit rate in bits per second.
    pub fn bit_rate_hz(&self) -> f64 {
        self.chip_rate_hz / self.code.len() as f64
    }

    /// Peak optical power when transmitting a "1" bit (W).
    ///
    /// All `w` active chips contribute simultaneously → P_peak = w × P_chip.
    pub fn peak_power_w(&self) -> f64 {
        let weight: usize = self.code.iter().map(|&b| b as usize).sum();
        weight as f64 * self.power_per_chip_w
    }

    /// Time-averaged optical power (W).
    ///
    /// Assuming equally likely bits:
    /// `P_avg = (1/2) × w × P_chip × (w / n)` (active chips fraction × 50% bit-1 prob.)
    pub fn average_power_w(&self) -> f64 {
        let n = self.code.len();
        let weight: usize = self.code.iter().map(|&b| b as usize).sum();
        // Each chip is on with probability (w/n) × 0.5 (bit "1")
        0.5 * weight as f64 * self.power_per_chip_w * weight as f64 / n as f64
    }

    /// Code weight (number of "1" chips).
    pub fn weight(&self) -> usize {
        self.code.iter().map(|&b| b as usize).sum()
    }

    /// Code length (chips per symbol).
    pub fn code_length(&self) -> usize {
        self.code.len()
    }
}

// ---------------------------------------------------------------------------
// MAI analyser
// ---------------------------------------------------------------------------

/// Multiple Access Interference (MAI) analyser for incoherent OCDMA.
///
/// Models the statistical interference from K − 1 simultaneous users onto
/// user 0 using both exact binomial and Gaussian approximations.
#[derive(Debug, Clone)]
pub struct MaiAnalyzer {
    /// Codewords of all active users (user 0 is the reference).
    pub codes: Vec<Vec<u8>>,
    /// Number of active users.
    pub n_users: usize,
}

impl MaiAnalyzer {
    /// Create a new MAI analyser from a set of user codewords.
    pub fn new(codes: Vec<Vec<u8>>) -> Self {
        let n_users = codes.len();
        Self { codes, n_users }
    }

    /// Cross-correlation coefficient between user `j` and the reference user 0.
    ///
    /// Returns the maximum cyclic cross-correlation normalised by the weight
    /// of user 0's code.
    pub fn mai_probability(&self, user_j: usize, threshold: usize) -> f64 {
        if user_j >= self.codes.len() || self.codes.is_empty() {
            return 0.0;
        }
        let c0 = &self.codes[0];
        let cj = &self.codes[user_j];
        let n = c0.len().min(cj.len());
        let w0: usize = c0.iter().map(|&b| b as usize).sum();
        if w0 == 0 {
            return 0.0;
        }
        // Maximum cyclic cross-correlation over all shifts
        let max_xcorr = (0..n)
            .map(|tau| {
                (0..n)
                    .map(|i| (cj[i] as usize) * (c0[(i + tau) % n] as usize))
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0);

        // P(MAI hits threshold) ≈ max_xcorr / w0, capped at 1
        let raw = max_xcorr as f64 / (w0 as f64 * threshold.max(1) as f64);
        raw.min(1.0)
    }

    /// Variance of the total MAI contribution from K−1 interferers.
    ///
    /// Uses the Gaussian approximation: each interfering user contributes
    /// independently with mean μ_I and variance σ²_I derived from their
    /// cross-correlation statistics with user 0.
    ///
    /// `σ²_total = (K-1) × <λ_c²> / n²`
    ///
    /// where `<λ_c²>` is the mean squared cross-correlation and n is code
    /// length.
    pub fn total_mai_variance(&self, k_users: usize) -> f64 {
        if self.codes.is_empty() {
            return 0.0;
        }
        let n = self.codes[0].len();
        if n == 0 || k_users <= 1 {
            return 0.0;
        }
        // Estimate mean squared cross-correlation from available code pairs
        let c0 = &self.codes[0];
        let n_pairs = (self.codes.len() - 1).max(1);
        let mean_sq_xcorr: f64 = self.codes[1..]
            .iter()
            .map(|cj| {
                let max_xc = (0..n)
                    .map(|tau| {
                        (0..n)
                            .map(|i| (cj[i] as usize) * (c0[(i + tau) % n] as usize))
                            .sum::<usize>()
                    })
                    .max()
                    .unwrap_or(0) as f64;
                max_xc * max_xc
            })
            .sum::<f64>()
            / n_pairs as f64;

        (k_users - 1) as f64 * mean_sq_xcorr / (n as f64 * n as f64)
    }

    /// BER under MAI plus additive Gaussian noise, using the Gaussian
    /// approximation.
    ///
    /// `P_e = Q( (w − threshold) / sqrt(σ²_MAI + σ²_shot) )`
    ///
    /// where threshold is set to (w − 0.5).
    pub fn ber_with_mai(&self, snr_per_chip: f64, k_users: usize) -> f64 {
        if self.codes.is_empty() {
            return 0.5;
        }
        let w: f64 = self.codes[0].iter().map(|&b| b as f64).sum();
        let n = self.codes[0].len() as f64;
        if w < 1.0 || n < 1.0 {
            return 0.5;
        }
        // Shot noise variance at chip level: σ²_shot = w / SNR_chip
        let sigma_sq_shot = w / snr_per_chip.max(1e-12);
        let sigma_sq_mai = self.total_mai_variance(k_users);
        let sigma = (sigma_sq_shot + sigma_sq_mai).sqrt();
        // Decision threshold at w - 0.5; SNR argument = 0.5 / σ
        let snr_arg = (w - 0.5) / sigma.max(1e-30);
        // BER = 0.5 × P_e|bit0 + 0.5 × P_e|bit1 ≈ Q(snr_arg)
        q_function(snr_arg)
    }

    /// Maximum number of simultaneous users for BER below `target_ber`.
    ///
    /// Iterates from 1 upward until BER exceeds the target, returning the
    /// last valid user count.  Hard-capped at 1000 users.
    pub fn max_users_for_ber(&self, snr_per_chip: f64, target_ber: f64) -> usize {
        let mut k = 1usize;
        loop {
            let ber = self.ber_with_mai(snr_per_chip, k);
            if ber > target_ber {
                return k.saturating_sub(1).max(1);
            }
            k += 1;
            if k > 1000 {
                return 1000;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Coherent OCDMA
// ---------------------------------------------------------------------------

/// Coherent OCDMA transceiver using bipolar (±1) spreading codes.
///
/// Information bits are differentially spread by multiplying with the bipolar
/// codeword.  At the receiver, the received signal is correlated with the
/// stored replica (matched filter), producing a processing gain equal to the
/// code length.
#[derive(Debug, Clone)]
pub struct CoherentOcdma {
    /// Bipolar (±1) spreading code.
    pub code: Vec<i8>,
    /// Duration of one chip in seconds.
    pub chip_duration_s: f64,
}

impl CoherentOcdma {
    /// Create a new coherent OCDMA transceiver.
    pub fn new(code: Vec<i8>, chip_duration_s: f64) -> Self {
        Self {
            code,
            chip_duration_s,
        }
    }

    /// Encode an information symbol (`+1` or `−1`) into chip samples.
    ///
    /// The transmitted chip sequence is `bit × code[k]` for each chip k.
    pub fn encode(&self, bit: i8) -> Vec<i8> {
        self.code.iter().map(|&c| bit * c).collect()
    }

    /// Matched-filter correlator output.
    ///
    /// `y = Σ_k received[k] × code[k]`
    ///
    /// For a noise-free matched signal this returns ±N (code length).
    pub fn correlate(&self, received: &[f64]) -> f64 {
        received
            .iter()
            .zip(self.code.iter())
            .map(|(&r, &c)| r * c as f64)
            .sum()
    }

    /// Processing gain (linear ratio).
    ///
    /// Equals the code length N (number of chips).
    pub fn snr_gain(&self) -> f64 {
        self.code.len() as f64
    }

    /// Processing gain in dB.
    ///
    /// `PG_dB = 10 · log10(N)`
    pub fn processing_gain_db(&self) -> f64 {
        10.0 * (self.code.len() as f64).log10()
    }

    /// Symbol rate in symbols per second.
    pub fn symbol_rate_sps(&self) -> f64 {
        if self.chip_duration_s > 0.0 {
            1.0 / (self.chip_duration_s * self.code.len() as f64)
        } else {
            0.0
        }
    }

    /// BER for BPSK coherent OCDMA after despreading, assuming AWGN.
    ///
    /// `P_e = Q(sqrt(2 × N × Eb/N0))`
    pub fn ber_bpsk(&self, eb_n0_linear: f64) -> f64 {
        let n = self.code.len() as f64;
        q_function((2.0 * n * eb_n0_linear).sqrt())
    }
}

/// Re-export PI for use in sibling modules.
#[allow(dead_code)]
const _PI: f64 = PI;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ooc_code() -> Vec<u8> {
        // (7, 3, 1, 1) OOC codeword at positions 0,1,3
        vec![1, 1, 0, 1, 0, 0, 0]
    }

    #[test]
    fn incoherent_encode_decode_roundtrip() {
        let code = make_ooc_code();
        let trx = IncoherentOcdma::new(code.clone(), 1e9, 1e-3);
        let chips1 = trx.encode(1);
        let chips0 = trx.encode(0);

        // Convert to f64 for decoder input
        let rx1: Vec<f64> = chips1.iter().map(|&b| b as f64).collect();
        let rx0: Vec<f64> = chips0.iter().map(|&b| b as f64).collect();

        assert_eq!(trx.decode(&rx1), 1);
        assert_eq!(trx.decode(&rx0), 0);
    }

    #[test]
    fn incoherent_bit_rate() {
        let code = make_ooc_code(); // length 7
        let trx = IncoherentOcdma::new(code, 7e9, 1e-3);
        // bit_rate = chip_rate / n = 7e9 / 7 = 1e9
        assert!((trx.bit_rate_hz() - 1e9).abs() < 1.0);
    }

    #[test]
    fn incoherent_peak_average_power() {
        let code = make_ooc_code(); // weight 3, length 7
        let p_chip = 1e-3;
        let trx = IncoherentOcdma::new(code, 1e9, p_chip);
        assert!((trx.peak_power_w() - 3e-3).abs() < 1e-12);
        // avg = 0.5 × 3 × 1e-3 × 3/7
        let expected_avg = 0.5 * 3.0 * p_chip * 3.0 / 7.0;
        assert!((trx.average_power_w() - expected_avg).abs() < 1e-15);
    }

    #[test]
    fn mai_analyzer_variance_zero_one_user() {
        let c0 = make_ooc_code();
        let analyzer = MaiAnalyzer::new(vec![c0]);
        // Only 1 user → no MAI
        assert_eq!(analyzer.total_mai_variance(1), 0.0);
    }

    #[test]
    fn mai_analyzer_ber_decreases_with_snr() {
        let c0 = make_ooc_code();
        let c1 = vec![0u8, 0, 1, 0, 1, 1, 0]; // second user
        let analyzer = MaiAnalyzer::new(vec![c0, c1]);
        let ber_low = analyzer.ber_with_mai(5.0, 2);
        let ber_high = analyzer.ber_with_mai(20.0, 2);
        assert!(ber_low > ber_high, "BER should decrease with SNR");
    }

    #[test]
    fn coherent_encode_decode() {
        let code = vec![1i8, -1, 1, 1, -1, 1, -1, -1];
        let trx = CoherentOcdma::new(code.clone(), 1e-9);
        let chips = trx.encode(1i8);
        let rx: Vec<f64> = chips.iter().map(|&c| c as f64).collect();
        let corr = trx.correlate(&rx);
        // Should equal N = 8 (all chips aligned)
        assert!((corr - 8.0).abs() < 1e-10);
    }

    #[test]
    fn coherent_processing_gain() {
        let code: Vec<i8> = vec![1; 64];
        let trx = CoherentOcdma::new(code, 1e-9);
        let pg = trx.processing_gain_db();
        // 10 log10(64) ≈ 18.06 dB
        assert!((pg - 18.06).abs() < 0.1, "PG_dB = {}", pg);
    }

    #[test]
    fn mai_max_users_for_ber_monotone() {
        let c0 = make_ooc_code();
        let c1 = vec![0u8, 1, 0, 1, 1, 0, 0];
        let analyzer = MaiAnalyzer::new(vec![c0, c1]);
        let cap_tight = analyzer.max_users_for_ber(10.0, 1e-3);
        let cap_loose = analyzer.max_users_for_ber(10.0, 0.1);
        // Looser BER target → at least as many users
        assert!(cap_loose >= cap_tight);
    }
}
