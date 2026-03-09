//! Performance analysis for OCDMA systems.
//!
//! Provides:
//! * Q-function and BER utilities.
//! * OOK-OCDMA BER models (Gaussian approximation and exact binomial sum).
//! * Capacity estimation tools.
//! * Multiple-access scheme capacity comparison (OCDMA vs TDMA vs WDM).

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Q-function
// ---------------------------------------------------------------------------

/// Q-function: `Q(x) = 0.5 · erfc(x / √2)`.
///
/// Uses the Abramowitz & Stegun 26.2.17 polynomial approximation.
/// Maximum absolute error < 7.5 × 10⁻⁸ for x ≥ 0.
///
/// For x < 0 the identity `Q(x) = 1 − Q(−x)` is applied.
pub fn q_function(x: f64) -> f64 {
    if x < 0.0 {
        return 1.0 - q_function(-x);
    }
    let t = 1.0 / (1.0 + 0.2316419 * x);
    let poly = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    poly * (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Complementary error function approximation (Abramowitz & Stegun 7.1.26).
///
/// `erfc(x) = 2 · Q(x · √2)` — consistent with the Q-function above.
pub fn erfc_approx(x: f64) -> f64 {
    2.0 * q_function(x * 2.0_f64.sqrt())
}

// ---------------------------------------------------------------------------
// OOK-OCDMA BER
// ---------------------------------------------------------------------------

/// BER analysis for incoherent OOK-OCDMA with MAI.
///
/// Parameters:
/// * `code_length` n  — chips per symbol.
/// * `code_weight` w  — number of "1" chips per codeword.
/// * `n_users` K      — number of simultaneous active users (including self).
/// * `snr_per_chip`   — SNR measured at individual chip receiver (linear).
#[derive(Debug, Clone)]
pub struct OokOcdmaBer {
    /// Code length n.
    pub code_length: usize,
    /// Code weight w.
    pub code_weight: usize,
    /// Number of active users K (including the desired user).
    pub n_users: usize,
    /// SNR per chip (linear ratio, signal power / noise power at chip level).
    pub snr_per_chip: f64,
}

impl OokOcdmaBer {
    /// Create a new BER analysis instance.
    pub fn new(code_length: usize, code_weight: usize, n_users: usize, snr_per_chip: f64) -> Self {
        Self {
            code_length,
            code_weight,
            n_users,
            snr_per_chip,
        }
    }

    /// BER using the Gaussian approximation for MAI + shot noise.
    ///
    /// Model: given that K users are active, K−1 interferers each contribute
    /// an MAI term.  The total noise at the correlator output is modelled as
    /// Gaussian with:
    ///
    /// * Signal component = w (weight of desired user's code).
    /// * MAI variance = (K−1) × λ_c² / n  (λ_c = w²/n assumed average).
    /// * Shot-noise variance = w / SNR_chip.
    ///
    /// Decision threshold at w − 0.5 (midpoint).
    ///
    /// `P_e = Q(SNR_eff)` where `SNR_eff = (w − threshold) / σ_total`.
    pub fn ber_gaussian(&self) -> f64 {
        let n = self.code_length as f64;
        let w = self.code_weight as f64;
        let k = self.n_users as f64;
        if w < 1.0 || n < 1.0 {
            return 0.5;
        }
        // Mean cross-correlation λ_c ≈ w² / n
        let lambda_c = w * w / n;
        // MAI variance from K−1 interferers
        let sigma_sq_mai = (k - 1.0).max(0.0) * lambda_c * lambda_c / n;
        // Shot noise variance
        let sigma_sq_shot = w / self.snr_per_chip.max(1e-30);
        let sigma = (sigma_sq_mai + sigma_sq_shot).sqrt();
        // Threshold gap
        let gap = (w - 0.5).max(0.0);
        q_function(gap / sigma.max(1e-30))
    }

    /// BER using the exact binomial sum for MAI.
    ///
    /// For each possible MAI count m (number of interfering users whose
    /// codes produce a "hit" on the reference chip set), the BER is
    /// weighted by the binomial probability of exactly m hits out of K−1
    /// interferers.
    ///
    /// `P_e = Σ_{m=0}^{K-1} P(m hits) × P_e(m)`
    ///
    /// where `P(m hits) = C(K-1, m) × p^m × (1-p)^{K-1-m}`
    /// and `p = w² / (n · w) = w / n` (probability an interferer contributes
    /// a hit to the correlator, given OOC λ_c = 1 bound).
    ///
    /// `P_e(m) = Q((w − m − 0.5) / σ_shot)` for m < w, else 1.
    pub fn ber_exact(&self) -> f64 {
        let n = self.code_length;
        let w = self.code_weight;
        let k = self.n_users;
        if w == 0 || n == 0 {
            return 0.5;
        }

        let p_hit = w as f64 / n as f64; // probability each interferer contributes a hit
        let sigma_shot = (w as f64 / self.snr_per_chip.max(1e-30)).sqrt();
        let n_interferers = k.saturating_sub(1);

        let mut ber = 0.0f64;
        // Precompute log-binomial coefficients to avoid overflow
        for m in 0..=n_interferers {
            let log_binom = log_binomial(n_interferers, m);
            let log_p = m as f64 * p_hit.max(1e-300).ln()
                + (n_interferers - m) as f64 * (1.0 - p_hit).max(1e-300).ln();
            let prob_m = (log_binom + log_p).exp();
            // Conditional BER: threshold is w - 0.5 - m (MAI shifts the signal down)
            let gap = w as f64 - 0.5 - m as f64;
            let cond_ber = if gap <= 0.0 {
                1.0
            } else {
                q_function(gap / sigma_shot.max(1e-30))
            };
            ber += prob_m * cond_ber;
        }
        ber.clamp(0.0, 1.0)
    }

    /// Processing gain `PG = n / w` (linear).
    pub fn processing_gain(&self) -> f64 {
        if self.code_weight == 0 {
            return 0.0;
        }
        self.code_length as f64 / self.code_weight as f64
    }

    /// Processing gain in dB.
    pub fn processing_gain_db(&self) -> f64 {
        10.0 * self.processing_gain().max(1e-300).log10()
    }

    /// Maximum number of simultaneous users for BER below `target_ber`.
    ///
    /// Uses the Gaussian BER estimate.  Iterates from K = 1 upward,
    /// returning the largest K where BER ≤ target.  Hard cap: 10 000.
    pub fn capacity_for_ber(&self, target_ber: f64) -> usize {
        let mut best = 1usize;
        for k in 1..=10_000usize {
            let inst = Self {
                code_length: self.code_length,
                code_weight: self.code_weight,
                n_users: k,
                snr_per_chip: self.snr_per_chip,
            };
            if inst.ber_gaussian() <= target_ber {
                best = k;
            } else {
                break;
            }
        }
        best
    }
}

// ---------------------------------------------------------------------------
// Multiple-access capacity comparison
// ---------------------------------------------------------------------------

/// Comparative capacity analysis for OCDMA, TDMA, and WDM.
///
/// All three schemes are assumed to share the same total bandwidth
/// `bandwidth_hz`.  The number of logical channels is `n_channels`.
#[derive(Debug, Clone)]
pub struct MultipleAccessComparison {
    /// Total available bandwidth (Hz).
    pub bandwidth_hz: f64,
    /// Number of logical channels (users or wavelengths).
    pub n_channels: usize,
    /// Per-channel SNR in dB.
    pub snr_db: f64,
}

impl MultipleAccessComparison {
    /// Create a new comparison instance.
    pub fn new(bandwidth_hz: f64, n_channels: usize, snr_db: f64) -> Self {
        Self {
            bandwidth_hz,
            n_channels,
            snr_db,
        }
    }

    /// SNR as a linear power ratio.
    fn snr_linear(&self) -> f64 {
        10.0_f64.powf(self.snr_db / 10.0)
    }

    /// TDMA aggregate throughput.
    ///
    /// Each user gets `1/K` of the time, full bandwidth.
    /// `C_TDMA = B × log2(1 + SNR)` (Shannon bound on full bandwidth).
    pub fn tdma_capacity_bps(&self) -> f64 {
        let snr = self.snr_linear();
        self.bandwidth_hz * (1.0 + snr).log2()
    }

    /// WDM aggregate throughput.
    ///
    /// N channels each with bandwidth `B/N` and the same per-channel SNR.
    /// `C_WDM = N × (B/N) × log2(1 + SNR) = C_TDMA` (Shannon equivalent).
    pub fn wdm_capacity_bps(&self) -> f64 {
        if self.n_channels == 0 {
            return 0.0;
        }
        let ch_bw = self.bandwidth_hz / self.n_channels as f64;
        let snr = self.snr_linear();
        self.n_channels as f64 * ch_bw * (1.0 + snr).log2()
    }

    /// OCDMA aggregate throughput.
    ///
    /// With code length n and weight w, the processing gain is n/w and
    /// the effective per-user bandwidth is `B × w / n`.  The MAI-limited
    /// capacity is approximated by the Gaussian model for K users:
    ///
    /// `C_OCDMA ≈ K × (B/n) × log2(1 + SNR_eff)`
    ///
    /// where `SNR_eff = SNR / (1 + (K−1) × w² / n²)`.
    pub fn ocdma_capacity_bps(&self, code_length: usize, code_weight: usize) -> f64 {
        if code_length == 0 || code_weight == 0 || self.n_channels == 0 {
            return 0.0;
        }
        let n = code_length as f64;
        let w = code_weight as f64;
        let k = self.n_channels as f64;
        let snr = self.snr_linear();
        let pg = n / w;
        // Effective SNR after MAI
        let mai_factor = 1.0 + (k - 1.0).max(0.0) * w * w / (n * n);
        let snr_eff = snr * pg / mai_factor;
        let chip_rate = self.bandwidth_hz / n;
        k * chip_rate * (1.0 + snr_eff).log2()
    }

    /// Return the name of the scheme with the highest aggregate capacity.
    pub fn best_scheme(&self, code_length: usize, code_weight: usize) -> &'static str {
        let tdma = self.tdma_capacity_bps();
        let wdm = self.wdm_capacity_bps();
        let ocdma = self.ocdma_capacity_bps(code_length, code_weight);
        if ocdma >= tdma && ocdma >= wdm {
            "OCDMA"
        } else if wdm >= tdma {
            "WDM"
        } else {
            "TDMA"
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: log-binomial coefficient
// ---------------------------------------------------------------------------

/// Compute `log C(n, k) = log(n!) - log(k!) - log((n-k)!)` using Stirling
/// summation via the exact log-gamma identity for integer arguments.
fn log_binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    if k == 0 || k == n {
        return 0.0;
    }
    // log C(n,k) = Σ_{i=0}^{k-1} log(n-i) - log(i+1)
    (0..k)
        .map(|i| ((n - i) as f64).ln() - ((i + 1) as f64).ln())
        .sum()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q_function_known_values() {
        // Q(0) = 0.5
        assert!((q_function(0.0) - 0.5).abs() < 1e-6);
        // Q(1) ≈ 0.1587
        assert!((q_function(1.0) - 0.1587).abs() < 1e-3);
        // Q(3) ≈ 1.35e-3
        assert!((q_function(3.0) - 1.35e-3).abs() < 1e-4);
        // Q(x) + Q(-x) = 1
        assert!((q_function(2.0) + q_function(-2.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ook_ber_gaussian_single_user() {
        // With K=1 (no MAI), BER should be very low at high SNR
        let ber = OokOcdmaBer::new(13, 3, 1, 30.0).ber_gaussian();
        assert!(
            ber < 0.01,
            "Single-user BER at SNR=30 should be low: {}",
            ber
        );
    }

    #[test]
    fn ook_ber_increases_with_users() {
        let ber1 = OokOcdmaBer::new(100, 5, 1, 20.0).ber_gaussian();
        let ber5 = OokOcdmaBer::new(100, 5, 5, 20.0).ber_gaussian();
        assert!(ber5 > ber1, "BER should increase with more users");
    }

    #[test]
    fn processing_gain_values() {
        let inst = OokOcdmaBer::new(100, 5, 1, 10.0);
        assert!((inst.processing_gain() - 20.0).abs() < 1e-10);
        assert!((inst.processing_gain_db() - 10.0 * 20.0f64.log10()).abs() < 1e-8);
    }

    #[test]
    fn capacity_for_ber_monotone() {
        let inst = OokOcdmaBer::new(100, 5, 1, 20.0);
        let cap_tight = inst.capacity_for_ber(1e-3);
        let cap_loose = inst.capacity_for_ber(0.1);
        assert!(
            cap_loose >= cap_tight,
            "Looser BER target ≥ tighter target capacity"
        );
    }

    #[test]
    fn ook_ber_exact_consistent_single_user() {
        // For K=1 exact and Gaussian should agree reasonably closely
        let inst = OokOcdmaBer::new(31, 4, 1, 15.0);
        let ber_g = inst.ber_gaussian();
        let ber_e = inst.ber_exact();
        // Both should be small; ratio within a factor 100 (models differ)
        let ratio = if ber_g > 1e-15 { ber_e / ber_g } else { 1.0 };
        assert!(
            ratio < 100.0 && ratio > 0.0,
            "Exact/Gaussian ratio = {}",
            ratio
        );
    }

    #[test]
    fn mac_tdma_equals_wdm() {
        // Shannon: TDMA and WDM have equal aggregate capacity
        let mac = MultipleAccessComparison::new(10e9, 4, 20.0);
        let diff = (mac.tdma_capacity_bps() - mac.wdm_capacity_bps()).abs();
        assert!(
            diff < 1.0,
            "TDMA and WDM capacity should be equal: diff = {}",
            diff
        );
    }

    #[test]
    fn mac_ocdma_positive() {
        let mac = MultipleAccessComparison::new(10e9, 4, 20.0);
        let cap = mac.ocdma_capacity_bps(31, 4);
        assert!(cap > 0.0, "OCDMA capacity must be positive");
    }
}
