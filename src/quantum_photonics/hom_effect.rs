//! Hong-Ou-Mandel (HOM) two-photon interference.
//!
//! The HOM effect arises when two indistinguishable photons enter a 50:50 beam splitter
//! from different input ports: they always exit together (bunching), causing a dip in
//! coincidence counts as a function of relative arrival time delay.
//!
//! # References
//! - Hong, Ou, Mandel (1987) *Phys. Rev. Lett.* 59, 2044
//! - Santori et al. (2002) *Nature* 419, 594 — semiconductor quantum dot HOM
//! - Trivedi et al. (2020) — multi-photon HOM generalisation

/// Hong-Ou-Mandel interferometer on a (partially) balanced beam splitter.
///
/// The coincidence probability as a function of relative time delay τ is:
///
/// P_coinc(τ) = (R² + T²) − 2RT · V · |g(τ)|²
///
/// where g(τ) is the normalised temporal overlap integral and V is the
/// photon indistinguishability (visibility).
#[derive(Debug, Clone)]
pub struct HomInterferometer {
    /// Reflectivity R ∈ [0, 1].  Transmissivity T = 1 − R.
    pub reflectivity: f64,
    /// Indistinguishability / mode overlap V ∈ [0, 1].
    pub mode_overlap: f64,
}

impl HomInterferometer {
    /// Construct an ideal 50:50 HOM interferometer with perfect indistinguishability.
    pub fn new() -> Self {
        Self {
            reflectivity: 0.5,
            mode_overlap: 1.0,
        }
    }

    /// Transmissivity T = 1 − R.
    #[inline]
    pub fn transmissivity(&self) -> f64 {
        1.0 - self.reflectivity
    }

    /// Coincidence probability for time delay `time_delay_s` and photon
    /// coherence time `coherence_time_s`.
    ///
    /// The temporal overlap integral for Lorentzian photons:
    ///   |g(τ)|² = exp(−|τ| / τ_c)   where τ_c = coherence_time_s.
    ///
    /// P_coinc(τ) = (R² + T²) − 2RT · V · exp(−|τ|/τ_c)
    pub fn coincidence_probability(&self, time_delay_s: f64, coherence_time_s: f64) -> f64 {
        let r = self.reflectivity;
        let t = self.transmissivity();
        let v = self.mode_overlap;
        let g2: f64 = if coherence_time_s > 1e-30 {
            (-time_delay_s.abs() / coherence_time_s).exp()
        } else {
            if time_delay_s.abs() < 1e-30 {
                1.0
            } else {
                0.0
            }
        };
        r * r + t * t - 2.0 * r * t * v * g2
    }

    /// HOM dip visibility V_dip = (P_max − P_min) / P_max.
    ///
    /// P_max = R² + T² (at large delay), P_min = R² + T² − 2RT·V (at zero delay).
    ///
    /// V_dip = 2RT·V / (R² + T²).
    pub fn dip_visibility(&self) -> f64 {
        let r = self.reflectivity;
        let t = self.transmissivity();
        let denom = r * r + t * t;
        if denom < f64::EPSILON {
            return 0.0;
        }
        2.0 * r * t * self.mode_overlap / denom
    }

    /// Probability of both photons exiting from the same port (bunching) at zero delay.
    ///
    /// P_bunch = (1 + V) / 2   for a 50:50 BS.
    pub fn bunching_probability(&self) -> f64 {
        (1.0 + self.mode_overlap) / 2.0
    }

    /// Probability of the photons exiting from different ports (antibunching) at zero delay.
    ///
    /// P_anti = (1 − V) / 2.
    pub fn antibunching_probability(&self) -> f64 {
        (1.0 - self.mode_overlap) / 2.0
    }

    /// Infer photon indistinguishability from the measured HOM dip visibility.
    ///
    /// Inverse of dip_visibility(): V = vis * (R² + T²) / (2RT).
    /// For a 50:50 BS (R = T = 1/2): V = visibility.
    pub fn indistinguishability_from_visibility(visibility: f64) -> f64 {
        // Using R = T = 0.5: dip_visibility = V (indistinguishability) exactly
        visibility.clamp(0.0, 1.0)
    }

    /// Coherence length corresponding to a coherence time τ_c and speed of light c.
    ///
    /// l_c = c · τ_c.
    pub fn coherence_length_m(coherence_time_s: f64) -> f64 {
        2.997_924_58e8 * coherence_time_s
    }

    /// HOM dip half-width at half-maximum (HWHM) in seconds.
    ///
    /// For Lorentzian photons, the dip profile is exp(−|τ|/τ_c),
    /// so HWHM = τ_c · ln 2.
    pub fn hom_dip_hwhm_s(coherence_time_s: f64) -> f64 {
        coherence_time_s * 2.0_f64.ln()
    }
}

impl Default for HomInterferometer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── MultiPhotonHom ───────────────────────────────────────────────────────────

/// Multi-photon HOM: N indistinguishable photons on an m-mode beam splitter network.
///
/// When N identical photons enter N distinct input ports of a balanced network,
/// the probability of N-fold coincidence (all in different output modes) is suppressed
/// relative to the distinguishable case by a factor that depends on the permanent of
/// the unitary submatrix.
#[derive(Debug, Clone)]
pub struct MultiPhotonHom {
    /// Number of photons N.
    pub n_photons: usize,
    /// Number of modes m.
    pub n_modes: usize,
    /// Photon indistinguishability I ∈ [0, 1].
    pub indistinguishability: f64,
}

impl MultiPhotonHom {
    /// Construct a multi-photon HOM configuration.
    pub fn new(n: usize, m: usize, indist: f64) -> Self {
        Self {
            n_photons: n,
            n_modes: m,
            indistinguishability: indist.clamp(0.0, 1.0),
        }
    }

    /// Probability of N-fold coincidence (one photon per output mode).
    ///
    /// For fully indistinguishable photons on a balanced n-mode splitter,
    /// the coincidence probability is suppressed by the Hong-Ou-Mandel effect.
    /// For two photons (n=2, m=2): P_coinc = (1 − I)/2.
    /// For N photons: approximated as  P_coinc ≈ (1 − I)^(N(N−1)/2) / C(m, N).
    pub fn n_fold_coincidence_prob(&self) -> f64 {
        let n = self.n_photons;
        let m = self.n_modes;
        if n == 0 || m < n {
            return 0.0;
        }
        if n == 1 {
            return 1.0 / m as f64;
        }
        let pairs = (n * (n - 1) / 2) as f64;
        let suppression = (1.0 - self.indistinguishability).powf(pairs);
        let comb = combinatorial_n_choose_k(m, n) as f64;
        if comb < f64::EPSILON {
            return 0.0;
        }
        suppression / comb
    }

    /// Probability of all photons exiting the same output mode (full bunching).
    ///
    /// Approximated as: P_bunch ≈ I^(N(N−1)/2) / m.
    pub fn bunching_prob(&self) -> f64 {
        let n = self.n_photons;
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return 1.0 / self.n_modes as f64;
        }
        let pairs = (n * (n - 1) / 2) as f64;
        self.indistinguishability.powf(pairs) / self.n_modes as f64
    }
}

/// Binomial coefficient C(n, k) for use in probability calculations.
fn combinatorial_n_choose_k(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result = 1u64;
    for i in 0..k {
        result = result.saturating_mul(n as u64 - i as u64) / (i as u64 + 1);
    }
    result
}

// ─── IndistinguishabilityMeasurement ─────────────────────────────────────────

/// Photon indistinguishability measurement via HOM interferometry.
///
/// The corrected indistinguishability M = V_HOM / V_classical accounts for
/// non-unity single-photon purity (g²(0) > 0) and optical losses.
#[derive(Debug, Clone)]
pub struct IndistinguishabilityMeasurement {
    /// Raw HOM visibility (from same-source, adjacent-pulse interference).
    pub raw_visibility: f64,
    /// Classical visibility reference (from different-time, non-interfering pulses).
    pub classical_visibility: f64,
}

impl IndistinguishabilityMeasurement {
    /// Construct from raw and classical visibility measurements.
    pub fn new(raw_vis: f64, classical_vis: f64) -> Self {
        Self {
            raw_visibility: raw_vis.clamp(0.0, 1.0),
            classical_visibility: classical_vis.clamp(f64::EPSILON, 1.0),
        }
    }

    /// Corrected indistinguishability M = V_raw / V_classical.
    ///
    /// This accounts for beam-splitter imbalance, loss asymmetry, and
    /// multi-photon contamination encoded in the classical reference.
    pub fn indistinguishability(&self) -> f64 {
        (self.raw_visibility / self.classical_visibility).clamp(0.0, 1.0)
    }

    /// Construct from second-order coherence g²(0) and raw HOM visibility.
    ///
    /// The classical visibility for a source with g²(0) = g2 is:
    ///   V_classical = 1 − g²(0) / 2
    ///
    /// (from the two-photon coherence correction in a Hanbury-Brown–Twiss setup).
    pub fn from_g2(g2_zero: f64, raw_visibility: f64) -> Self {
        let classical_vis = 1.0 - g2_zero / 2.0;
        Self::new(raw_visibility, classical_vis.max(f64::EPSILON))
    }

    /// Single-photon purity estimated from g²(0): P = 1 − g²(0).
    pub fn single_photon_purity(g2_zero: f64) -> f64 {
        (1.0 - g2_zero).clamp(0.0, 1.0)
    }

    /// Effective indistinguishability including purity correction:
    ///   M_eff = M · (1 − g²(0))
    pub fn effective_indistinguishability(&self, g2_zero: f64) -> f64 {
        let m = self.indistinguishability();
        let purity = Self::single_photon_purity(g2_zero);
        (m * purity).clamp(0.0, 1.0)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_hom_ideal_zero_delay() {
        // Perfect HOM dip: P_coinc(τ=0) = R² + T² − 2RT·V = 0 for R=T=0.5, V=1
        let hom = HomInterferometer::new();
        let p = hom.coincidence_probability(0.0, 1e-12);
        assert!(approx_eq(p, 0.0, 1e-12), "ideal HOM: P={p} ≠ 0");
    }

    #[test]
    fn test_hom_large_delay() {
        // At large delay, g(τ)→0 → P_coinc = R² + T² = 0.5 for balanced BS
        let hom = HomInterferometer::new();
        let p = hom.coincidence_probability(1e6, 1e-12);
        // R²+T² = 0.25 + 0.25 = 0.5
        assert!(approx_eq(p, 0.5, 1e-12));
    }

    #[test]
    fn test_hom_visibility_balanced_perfect() {
        // For R=T=0.5, V=1: dip_visibility = 2*0.5*0.5*1 / (0.25+0.25) = 0.5/0.5 = 1.0
        let hom = HomInterferometer::new();
        assert!(approx_eq(hom.dip_visibility(), 1.0, 1e-12));
    }

    #[test]
    fn test_hom_visibility_partial_indist() {
        // V = 0.8, balanced BS: dip_visibility = 0.8
        let hom = HomInterferometer {
            reflectivity: 0.5,
            mode_overlap: 0.8,
        };
        assert!(approx_eq(hom.dip_visibility(), 0.8, 1e-12));
    }

    #[test]
    fn test_bunching_antibunching_sum_to_one() {
        let hom = HomInterferometer {
            reflectivity: 0.5,
            mode_overlap: 0.9,
        };
        let sum = hom.bunching_probability() + hom.antibunching_probability();
        assert!(approx_eq(sum, 1.0, 1e-12));
    }

    #[test]
    fn test_indistinguishability_from_visibility_trivial() {
        // For ideal 50:50 BS: visibility = indistinguishability
        let v = HomInterferometer::indistinguishability_from_visibility(0.95);
        assert!(approx_eq(v, 0.95, 1e-12));
    }

    #[test]
    fn test_multi_photon_hom_two_photons_full_indist() {
        // Two fully indistinguishable photons: bunching_prob = 1/2, coincidence → 0
        let mhom = MultiPhotonHom::new(2, 2, 1.0);
        assert!(approx_eq(mhom.bunching_prob(), 0.5, 1e-12));
        assert!(mhom.n_fold_coincidence_prob() < 1e-12);
    }

    #[test]
    fn test_multi_photon_hom_distinguishable() {
        // Fully distinguishable photons: coincidence is not suppressed
        let mhom = MultiPhotonHom::new(2, 2, 0.0);
        // P_coinc = 1 / C(2,2) = 1 (since only one way to distribute 2 photons in 2 modes
        // with 1 each when fully distinguishable)
        assert!(mhom.n_fold_coincidence_prob() > 0.0);
    }

    #[test]
    fn test_indistinguishability_measurement_correction() {
        let meas = IndistinguishabilityMeasurement::new(0.90, 0.95);
        let m = meas.indistinguishability();
        // 0.90 / 0.95 ≈ 0.947
        assert!(approx_eq(m, 0.90 / 0.95, 1e-10));
    }

    #[test]
    fn test_indistinguishability_from_g2() {
        // g²(0) = 0.05: classical_vis = 1 - 0.025 = 0.975
        let meas = IndistinguishabilityMeasurement::from_g2(0.05, 0.90);
        let m = meas.indistinguishability();
        assert!(approx_eq(m, 0.90 / 0.975, 1e-10));
    }

    #[test]
    fn test_hom_dip_hwhm() {
        let tau_c = 1e-12; // 1 ps coherence time
        let hwhm = HomInterferometer::hom_dip_hwhm_s(tau_c);
        // HWHM = τ_c * ln(2) ≈ 0.693e-12
        assert!(approx_eq(hwhm, tau_c * 2.0_f64.ln(), 1e-20));
    }
}
