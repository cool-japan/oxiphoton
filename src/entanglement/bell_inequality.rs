//! Bell inequality tests and loophole-free analysis.
//!
//! Implements:
//! - CHSH inequality |S| ≤ 2 (classical), quantum max = 2√2
//! - Clauser-Horne (CH) inequality (detection-efficiency analysis)
//! - Mermin inequality (three-party GHZ entanglement)
//! - Loophole-free Bell test analysis
//!
//! References:
//! - Bell, Physics 1, 195 (1964)
//! - Clauser, Horne, Shimony, Holt, PRL 23, 880 (1969): CHSH
//! - Clauser & Horne, PRD 10, 526 (1974): CH
//! - Mermin, PRL 65, 1838 (1990): Mermin inequality
//! - Hensen et al., Nature 526, 682 (2015): loophole-free Bell test

use std::f64::consts::{PI, SQRT_2};

/// Classical limit of the CHSH parameter |S| ≤ 2.
pub const CHSH_CLASSICAL_BOUND: f64 = 2.0;

/// Tsirelson bound: quantum maximum |S| = 2√2.
pub const CHSH_TSIRELSON_BOUND: f64 = 2.0 * SQRT_2;

/// Detection efficiency threshold for CHSH loophole closure: η > 2/(√2 + 1) ≈ 0.8284.
pub const CHSH_DETECTION_THRESHOLD: f64 = 2.0 / (SQRT_2 + 1.0);

// ─── CHSH test ────────────────────────────────────────────────────────────────

/// Clauser-Horne-Shimony-Holt (CHSH) Bell inequality test.
///
/// Measures two-party correlations E(a, b) for two pairs of measurement settings
/// {a, a'} on Alice and {b, b'} on Bob.  The CHSH parameter is:
/// S = E(a,b) − E(a,b') + E(a',b) + E(a',b')
///
/// Local hidden variable theories require |S| ≤ 2.  Quantum mechanics allows
/// |S| ≤ 2√2 (Tsirelson bound).
#[derive(Debug, Clone)]
pub struct ChshTest {
    /// Measurement angles in degrees: `[a, a']` for Alice, `[b, b']` for Bob.
    pub settings: [[f64; 2]; 2],
    /// Correlation values: `correlations[i][j] = E(settings[0][i], settings[1][j])`.
    pub correlations: [[f64; 2]; 2],
}

impl ChshTest {
    /// Optimal CHSH settings that maximise the violation for a maximally entangled state.
    ///
    /// For |Ψ-⟩ with E(Δθ) = −cos(2Δθ), the maximal violation S = −2√2 is achieved with:
    /// a = 0°, a' = 45°, b = 22.5°, b' = 67.5°.
    ///
    /// S = E(a,b) − E(a,b') + E(a',b) + E(a',b') = −2√2 (|S| = 2√2 = Tsirelson bound).
    pub fn optimal_settings() -> Self {
        let a = 0.0_f64;
        let a2 = 45.0_f64;
        let b = 22.5_f64;
        let b2 = 67.5_f64;
        // Quantum correlation for |Ψ-⟩: E(Δθ) = −cos(2Δθ)
        let e = |alpha: f64, beta: f64| -> f64 { Self::quantum_correlation(alpha - beta) };
        Self {
            settings: [[a, a2], [b, b2]],
            correlations: [[e(a, b), e(a, b2)], [e(a2, b), e(a2, b2)]],
        }
    }

    /// Construct from four measured correlation values.
    ///
    /// # Arguments
    /// - `e_ab`  : E(a, b)
    /// - `e_ab2` : E(a, b')
    /// - `e_a2b` : E(a', b)
    /// - `e_a2b2`: E(a', b')
    pub fn from_correlations(e_ab: f64, e_ab2: f64, e_a2b: f64, e_a2b2: f64) -> Self {
        Self {
            settings: [[0.0, 45.0], [22.5, -22.5]],
            correlations: [[e_ab, e_ab2], [e_a2b, e_a2b2]],
        }
    }

    /// CHSH parameter S = E(a,b) − E(a,b') + E(a',b) + E(a',b').
    pub fn s_parameter(&self) -> f64 {
        let e_ab = self.correlations[0][0];
        let e_ab2 = self.correlations[0][1];
        let e_a2b = self.correlations[1][0];
        let e_a2b2 = self.correlations[1][1];
        e_ab - e_ab2 + e_a2b + e_a2b2
    }

    /// Returns `true` if the classical bound |S| > 2 is violated.
    pub fn violates_classical(&self) -> bool {
        self.s_parameter().abs() > CHSH_CLASSICAL_BOUND
    }

    /// Violation significance (number of standard deviations above the classical bound).
    ///
    /// Statistical uncertainty on E(a,b) ≈ 1/√N for N measurements per setting.
    /// σ_S ≈ 4/√N (four correlation terms, each ≈ 1/√N).
    pub fn violation_sigmas(&self, n_measurements: usize) -> f64 {
        if n_measurements == 0 {
            return 0.0;
        }
        let sigma_s = 4.0 / (n_measurements as f64).sqrt();
        let excess = self.s_parameter().abs() - CHSH_CLASSICAL_BOUND;
        if sigma_s > 0.0 {
            excess / sigma_s
        } else {
            0.0
        }
    }

    /// Quantum correlation for a maximally entangled state: E(θ) = −cos(2Δθ).
    ///
    /// Δθ is the relative angle between Alice and Bob's measurement directions (degrees).
    pub fn quantum_correlation(angle_deg: f64) -> f64 {
        let theta = angle_deg * PI / 180.0;
        -(2.0 * theta).cos()
    }

    /// Suggest measurement angles that maximise |S| for a state with given Bell-state fidelity F.
    ///
    /// The quantum bound for a partially mixed state is |S_max| = 2√2 * F.
    /// Returns [a, a', b, b'] in degrees.
    pub fn optimal_angles_for_state(fidelity: f64) -> [f64; 4] {
        // For a Werner-like state with fidelity F to |Ψ-⟩, the optimal angles are the
        // same as for the pure Bell state; only the magnitude of S is reduced.
        let _ = fidelity; // angles are state-independent for rotationally symmetric states
        [0.0, 45.0, 22.5, -22.5]
    }

    /// Expected CHSH parameter for a state with given Bell-state fidelity.
    pub fn expected_s_for_fidelity(fidelity: f64) -> f64 {
        CHSH_TSIRELSON_BOUND * fidelity.clamp(0.0, 1.0)
    }
}

// ─── CH test ─────────────────────────────────────────────────────────────────

/// Clauser-Horne (CH) inequality test.
///
/// The CH inequality does not require the fair-sampling assumption and directly
/// tests local hidden variable theories using raw coincidence and single counts.
///
/// Classical bound: S_CH ≤ 0
/// S_CH = p(a,b) + p(a,b') + p(a',b) − p(a',b') − p(a) − p(b)
#[derive(Debug, Clone)]
pub struct ChTest {
    /// Coincidence counts: `n_coinc[i][j]` for settings (a_i, b_j), i,j ∈ {0,1}.
    pub n_coinc: [[f64; 2]; 2],
    /// Single counts at Alice for settings [a, a'].
    pub n_single_a: [f64; 2],
    /// Single counts at Bob for settings [b, b'].
    pub n_single_b: [f64; 2],
    /// Estimated accidental coincidences (subtracted from n_coinc).
    pub n_accidental: f64,
}

impl ChTest {
    /// Construct from coincidence and single count arrays.
    pub fn new(coinc: [[f64; 2]; 2], sing_a: [f64; 2], sing_b: [f64; 2]) -> Self {
        Self {
            n_coinc: coinc,
            n_single_a: sing_a,
            n_single_b: sing_b,
            n_accidental: 0.0,
        }
    }

    /// Normalise coincidence count to joint probability (relative to singles product).
    fn joint_prob(&self, i: usize, j: usize) -> f64 {
        let n_a = self.n_single_a[i];
        let n_b = self.n_single_b[j];
        let denom = (n_a * n_b).max(1.0);
        (self.n_coinc[i][j] - self.n_accidental).max(0.0) / denom
    }

    /// Marginal probability p(a_i) = N_single_a[i] / N_total (normalised to max).
    fn marginal_a(&self) -> f64 {
        let max_s: f64 = self
            .n_single_a
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        self.n_single_a[0] / max_s.max(1.0)
    }

    /// Marginal probability p(b_j) = N_single_b[j] / N_total (normalised to max).
    fn marginal_b(&self) -> f64 {
        let max_s: f64 = self
            .n_single_b
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        self.n_single_b[0] / max_s.max(1.0)
    }

    /// CH parameter S_CH = p(a,b) + p(a,b') + p(a',b) − p(a',b') − p(a) − p(b).
    ///
    /// Classical bound: S_CH ≤ 0.
    pub fn ch_parameter(&self) -> f64 {
        self.joint_prob(0, 0) + self.joint_prob(0, 1) + self.joint_prob(1, 0)
            - self.joint_prob(1, 1)
            - self.marginal_a()
            - self.marginal_b()
    }

    /// Returns `true` if the CH classical bound S_CH > 0 is violated.
    pub fn violates_classical(&self) -> bool {
        self.ch_parameter() > 0.0
    }

    /// Detection efficiency threshold: η > 2/(√2 + 1) ≈ 82.84% (Garg & Mermin 1987).
    pub fn detection_efficiency_threshold(&self) -> f64 {
        CHSH_DETECTION_THRESHOLD
    }

    /// Effective detection efficiency estimated from singles-to-coincidence ratio.
    ///
    /// η_eff = 2 * N_coinc(a,b) / (N_single_a + N_single_b)
    pub fn effective_efficiency(&self) -> f64 {
        let coinc = self.n_coinc[0][0].max(0.0);
        let na = self.n_single_a[0];
        let nb = self.n_single_b[0];
        let denom = na + nb;
        if denom > 0.0 {
            2.0 * coinc / denom
        } else {
            0.0
        }
    }
}

// ─── Mermin inequality (3-party) ─────────────────────────────────────────────

/// Mermin inequality test for three-party GHZ entanglement.
///
/// For a GHZ state |GHZ⟩ = (|000⟩ + |111⟩)/√2 the Mermin parameter is:
/// M = ⟨XYY⟩ + ⟨YXY⟩ + ⟨YYX⟩ − ⟨XXX⟩
///
/// Classical bound: |M| ≤ 2.
/// Quantum prediction for |GHZ⟩: M = 4 (maximal violation).
#[derive(Debug, Clone)]
pub struct MerminTest {
    /// Number of parties (currently only 3-party is implemented)
    pub n_parties: usize,
    /// Measured correlators: [⟨XYY⟩, ⟨YXY⟩, ⟨YYX⟩, ⟨XXX⟩]
    pub correlations: Vec<f64>,
}

impl MerminTest {
    /// Construct with ideal GHZ-state quantum predictions (M = 4).
    pub fn new_ghz() -> Self {
        Self {
            n_parties: 3,
            // Quantum predictions: XYY=YXY=YYX=1, XXX=-1
            correlations: vec![1.0, 1.0, 1.0, -1.0],
        }
    }

    /// Mermin parameter M = ⟨XYY⟩ + ⟨YXY⟩ + ⟨YYX⟩ − ⟨XXX⟩.
    ///
    /// Classical bound: |M| ≤ 2.
    /// GHZ quantum maximum: |M| = 4.
    pub fn mermin_parameter(&self) -> f64 {
        if self.correlations.len() < 4 {
            return 0.0;
        }
        // M = C_xyy + C_yxy + C_yyx - C_xxx
        self.correlations[0] + self.correlations[1] + self.correlations[2] - self.correlations[3]
    }

    /// Classical bound on the Mermin parameter (= 2 for 3 parties).
    pub fn classical_bound(&self) -> f64 {
        match self.n_parties {
            3 => 2.0,
            n => 2.0_f64.powi((n as i32) / 2), // general even-n Mermin bound
        }
    }

    /// Quantum maximum for GHZ state (= 4 for 3 parties).
    pub fn quantum_maximum(&self) -> f64 {
        match self.n_parties {
            3 => 4.0,
            n => 2.0_f64.powi(((n + 1) as i32) / 2),
        }
    }

    /// Ratio |M| / classical_bound.  > 1 means the inequality is violated.
    pub fn violation_ratio(&self) -> f64 {
        self.mermin_parameter().abs() / self.classical_bound().max(1e-30)
    }

    /// Returns `true` if the Mermin classical bound is violated.
    pub fn violates_classical(&self) -> bool {
        self.mermin_parameter().abs() > self.classical_bound()
    }
}

// ─── Loophole-free analysis ───────────────────────────────────────────────────

/// Analysis of whether a Bell test closes all major loopholes.
///
/// Three loopholes must be addressed for a conclusive Bell test:
/// 1. Detection/fair-sampling loophole: η > η_threshold (≈ 82.84% for CHSH)
/// 2. Locality loophole: measurement events space-like separated
/// 3. Freedom-of-choice loophole: setting choices independent of hidden variables
#[derive(Debug, Clone)]
pub struct LoopholeFreeAnalysis {
    /// Measured detection efficiency η (per detector, symmetric)
    pub detection_efficiency: f64,
    /// Whether the locality loophole is closed (space-like separation enforced)
    pub locality_loophole_closed: bool,
    /// Whether freedom-of-choice is addressed (fast random setting selection)
    pub freedom_of_choice: bool,
    /// Whether fair-sampling is verified or detection loophole closed
    pub fair_sampling: bool,
}

impl LoopholeFreeAnalysis {
    /// Construct with given detection efficiency; locality and freedom-of-choice
    /// default to `false` (conservative).
    pub fn new(efficiency: f64) -> Self {
        Self {
            detection_efficiency: efficiency.clamp(0.0, 1.0),
            locality_loophole_closed: false,
            freedom_of_choice: false,
            fair_sampling: false,
        }
    }

    /// Returns `true` if all three major loopholes are closed.
    pub fn all_loopholes_closed(&self) -> bool {
        let detection_closed = self.detection_efficiency >= self.required_efficiency();
        detection_closed && self.locality_loophole_closed && self.freedom_of_choice
    }

    /// Required detection efficiency to close the detection loophole for CHSH:
    /// η_threshold = 2 / (√2 + 1) ≈ 82.84%.
    pub fn required_efficiency(&self) -> f64 {
        CHSH_DETECTION_THRESHOLD
    }

    /// Estimated p-value under the local hidden variable model.
    ///
    /// Uses the bound: p ≤ exp(−n_violations²/(2n_trials)) where
    /// n_violations is the excess number of events beyond the LHV bound.
    ///
    /// This is a simplified (conservative) estimate; rigorous bounds require
    /// game-theoretic or martingale-based methods (see Gill 2003).
    pub fn p_value(&self, s_param: f64, n_trials: usize) -> f64 {
        if n_trials == 0 {
            return 1.0;
        }
        let excess = s_param.abs() - CHSH_CLASSICAL_BOUND;
        if excess <= 0.0 {
            return 1.0; // No violation
        }
        // Per-trial LHV probability bound: p_trial ≤ (1 + excess/2)^{-1} approximately
        // For many trials: p ≈ exp(−n * excess² / 8)
        let exponent = -(n_trials as f64) * excess.powi(2) / 8.0;
        exponent.exp().clamp(0.0, 1.0)
    }

    /// Separation distance (km) required to close the locality loophole for a given
    /// coincidence window T_w (ns): d > c * T_w / 2.
    pub fn required_separation_km(coincidence_window_ns: f64) -> f64 {
        let c = 3e8_f64; // m/s
        c * coincidence_window_ns * 1e-9 / 2.0 / 1e3 // km
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chsh_optimal_settings_tsirelson() {
        let chsh = ChshTest::optimal_settings();
        let s = chsh.s_parameter();
        // Should approach Tsirelson bound 2√2 ≈ 2.8284
        assert!(
            (s.abs() - CHSH_TSIRELSON_BOUND).abs() < 1e-6,
            "Optimal CHSH S should equal Tsirelson bound, got {s}"
        );
    }

    #[test]
    fn test_chsh_violates_classical() {
        let chsh = ChshTest::optimal_settings();
        assert!(
            chsh.violates_classical(),
            "Optimal CHSH settings should violate classical bound"
        );
    }

    #[test]
    fn test_chsh_no_violation() {
        // Classical state: all correlations = 0
        let chsh = ChshTest::from_correlations(0.5, 0.5, 0.5, 0.5);
        let s = chsh.s_parameter();
        // S = 0.5 - 0.5 + 0.5 + 0.5 = 1.0 < 2
        assert!(
            !chsh.violates_classical(),
            "S={s} should not violate classical bound"
        );
    }

    #[test]
    fn test_chsh_quantum_correlation() {
        // E(0°) = −cos(0) = −1
        let e0 = ChshTest::quantum_correlation(0.0);
        assert!((e0 + 1.0).abs() < 1e-10, "E(0°) = -1");
        // E(45°) = −cos(90°) = 0
        let e45 = ChshTest::quantum_correlation(45.0);
        assert!(e45.abs() < 1e-10, "E(45°) = 0");
        // E(22.5°) = −cos(45°) = −1/√2
        let e225 = ChshTest::quantum_correlation(22.5);
        assert!(
            (e225 + 1.0 / std::f64::consts::SQRT_2).abs() < 1e-10,
            "E(22.5°) = -1/√2"
        );
    }

    #[test]
    fn test_ch_efficiency_threshold() {
        let ch = ChTest::new(
            [[100.0, 80.0], [90.0, 85.0]],
            [200.0, 195.0],
            [200.0, 200.0],
        );
        let threshold = ch.detection_efficiency_threshold();
        assert!(
            (threshold - CHSH_DETECTION_THRESHOLD).abs() < 1e-6,
            "Detection threshold = 2/(√2+1)"
        );
    }

    #[test]
    fn test_mermin_ghz_violation() {
        let mermin = MerminTest::new_ghz();
        // GHZ quantum prediction: M = 4, classical bound = 2
        assert_eq!(mermin.n_parties, 3);
        let m = mermin.mermin_parameter();
        assert!((m - 4.0).abs() < 1e-9, "GHZ Mermin parameter = 4, got {m}");
        assert!(
            mermin.violates_classical(),
            "GHZ state should violate Mermin inequality"
        );
        let ratio = mermin.violation_ratio();
        assert!((ratio - 2.0).abs() < 1e-9, "Violation ratio = 2 for GHZ");
    }

    #[test]
    fn test_loophole_free_p_value() {
        let analysis = LoopholeFreeAnalysis::new(0.90);
        // With S = 2√2 and 10^6 trials, p-value should be astronomically small
        let p = analysis.p_value(CHSH_TSIRELSON_BOUND, 1_000_000);
        assert!(
            p < 1e-10,
            "p-value for 10^6 trials and Tsirelson bound should be tiny, got {p}"
        );
    }

    #[test]
    fn test_loophole_free_detection_threshold() {
        let mut analysis = LoopholeFreeAnalysis::new(0.90);
        analysis.locality_loophole_closed = true;
        analysis.freedom_of_choice = true;
        assert!(
            analysis.all_loopholes_closed(),
            "90% efficiency should close detection loophole"
        );

        let analysis_low = LoopholeFreeAnalysis::new(0.70);
        assert!(
            !analysis_low.all_loopholes_closed(),
            "70% efficiency is below threshold"
        );
    }

    #[test]
    fn test_chsh_violation_sigmas() {
        let chsh = ChshTest::optimal_settings();
        let sigmas = chsh.violation_sigmas(10_000);
        // S ≈ 2√2, excess ≈ 0.828, σ_S ≈ 4/100 = 0.04 → ~20 σ
        assert!(sigmas > 10.0, "Should be highly significant, got {sigmas}σ");
    }
}
