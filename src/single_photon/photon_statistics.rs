//! Photon statistics for solid-state single-photon emitters.
//!
//! Provides:
//! - Second-order coherence function g²(τ) for single emitters
//! - Hanbury Brown-Twiss (HBT) experiment simulation
//! - Photon number distributions (Fock, coherent, thermal) and derived moments
//!
//! # Physical background
//! The key figure of merit for a single-photon source is g²(0) < 0.5,
//! which certifies non-classical light.  For an ideal two-level emitter:
//!   g²(τ) = 1 − exp(−|τ|/τ_lifetime)
//! Bunching (g²(τ) > 1 for τ > 0) indicates shelving to dark states.
//!
//! # References
//! - Mandel & Wolf, "Optical Coherence and Quantum Optics" (1995)
//! - Michler et al., Science 290, 2282 (2000) — first QD single-photon source
//! - Santori et al., Nature 419, 594 (2002) — indistinguishable photons from QD

use std::f64::consts::PI;

// ─── Physical constants ────────────────────────────────────────────────────────

// ─── G2Function ───────────────────────────────────────────────────────────────

/// Second-order coherence function g²(τ) measured in an HBT experiment.
///
/// Contains the time-resolved coincidence histogram and the derived g²(0).
#[derive(Debug, Clone)]
pub struct G2Function {
    /// Time delay grid (seconds)
    pub time_grid: Vec<f64>,
    /// g²(τ) values on the grid
    pub g2_values: Vec<f64>,
    /// Total measurement integration time (s)
    pub integration_time_s: f64,
    /// Mean detected photon rate (counts/s)
    pub count_rate_cps: f64,
}

impl G2Function {
    /// Construct g²(τ) for an ideal two-level single-photon emitter.
    ///
    /// `lifetime_ns`: excited-state radiative lifetime (ns)
    /// `n_time_points`: number of points in the delay grid (±5 τ_lifetime)
    ///
    /// The model function is:
    ///   g²(τ) = 1 − exp(−|τ|/τ_lifetime)
    pub fn new_single_photon_emitter(lifetime_ns: f64, n_time_points: usize) -> Self {
        let n = n_time_points.max(2);
        let tau_s = lifetime_ns * 1e-9;
        let t_max = 5.0 * tau_s;
        let dt = 2.0 * t_max / (n as f64 - 1.0);

        let mut time_grid = Vec::with_capacity(n);
        let mut g2_values = Vec::with_capacity(n);
        for i in 0..n {
            let t = -t_max + i as f64 * dt;
            time_grid.push(t);
            // Pure dephasing = 0 for this model
            let g2 = Self::compute_g2(lifetime_ns, 0.0, t * 1e9); // convert back to ns
            g2_values.push(g2);
        }

        Self {
            time_grid,
            g2_values,
            integration_time_s: 3600.0,
            count_rate_cps: 1e6,
        }
    }

    /// Compute g²(τ) for an emitter with given radiative lifetime and pure dephasing.
    ///
    /// Model:
    ///   g²(τ) = 1 − exp(−|τ|/τ_rad) · exp(−|τ|·Γ_pure)
    ///
    /// where Γ_pure = 1/(pure_dephasing_ns * 1e-9) is the pure dephasing rate.
    /// The cross term ensures g²(0) → 0 and g²(∞) → 1.
    ///
    /// `lifetime_ns`: radiative lifetime (ns)
    /// `pure_dephasing_ns`: pure dephasing time 1/γ_pure (ns); 0 = no dephasing
    /// `delay_ns`: time delay τ (ns)
    pub fn compute_g2(lifetime_ns: f64, pure_dephasing_ns: f64, delay_ns: f64) -> f64 {
        let tau_abs = delay_ns.abs();
        if lifetime_ns <= 0.0 {
            return 1.0;
        }
        let decay_rad = (-tau_abs / lifetime_ns).exp();
        let decay_pure = if pure_dephasing_ns > 0.0 {
            (-tau_abs / pure_dephasing_ns).exp()
        } else {
            1.0
        };
        1.0 - decay_rad * decay_pure
    }

    /// Extract g²(0) from the stored histogram.
    ///
    /// Returns the value at the nearest point to τ = 0.
    pub fn g2_zero(&self) -> f64 {
        if self.time_grid.is_empty() || self.g2_values.is_empty() {
            return 0.0;
        }
        // Find index closest to τ = 0
        let zero_idx = self
            .time_grid
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.abs()
                    .partial_cmp(&b.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.g2_values[zero_idx]
    }

    /// Single-photon purity from g²(0): P = 1 − g²(0).
    ///
    /// P = 1 for an ideal SPE, P = 0 for a fully classical (coherent/thermal) source.
    pub fn single_photon_purity(&self) -> f64 {
        (1.0 - self.g2_zero()).clamp(0.0, 1.0)
    }

    /// Compute g²(0) using the peak-area method.
    ///
    /// `center_area`: area of the τ = 0 coincidence peak
    /// `lateral_area`: mean area of lateral peaks (|τ| = n · T_rep)
    ///
    /// g²(0) = A_centre / A_lateral
    pub fn g2_zero_from_areas(center_area: f64, lateral_area: f64) -> f64 {
        if lateral_area <= 0.0 {
            return 0.0;
        }
        (center_area / lateral_area).clamp(0.0, 2.0)
    }

    /// Check whether bunching (g²(τ) > 1) exists at any non-zero delay.
    ///
    /// Bunching is a signature of shelving to metastable dark states.
    pub fn has_bunching(&self) -> bool {
        self.time_grid
            .iter()
            .zip(self.g2_values.iter())
            .any(|(t, g2)| t.abs() > 1e-15 && *g2 > 1.0 + 1e-9)
    }

    /// Hong-Ou-Mandel interference visibility from g²(τ).
    ///
    /// V_HOM = 1 − g²(0) / 2
    ///
    /// For ideal SPE: g²(0)=0 → V=1.  Coherent light: g²(0)=1 → V=0.5.
    pub fn interference_visibility(&self) -> f64 {
        (1.0 - self.g2_zero() / 2.0).clamp(0.0, 1.0)
    }
}

// ─── HbtSetup ─────────────────────────────────────────────────────────────────

/// Hanbury Brown-Twiss interferometer configuration.
///
/// Simulates the experimental parameters of a fibre-based HBT setup with
/// two single-photon detectors and a 50:50 beamsplitter.
#[derive(Debug, Clone)]
pub struct HbtSetup {
    /// Quantum efficiency of detector 1 (0–1)
    pub detector1_efficiency: f64,
    /// Quantum efficiency of detector 2 (0–1)
    pub detector2_efficiency: f64,
    /// IRF (instrument response function) FWHM (ps)
    pub time_resolution_ps: f64,
    /// Coincidence window width (ns)
    pub coincidence_window_ns: f64,
    /// Dark count rate per detector (counts/s)
    pub dark_count_rate_cps: f64,
}

impl HbtSetup {
    /// Create an HBT setup with identical detectors.
    ///
    /// `efficiency`: detector quantum efficiency (0–1)
    /// `timing_resolution_ps`: timing jitter FWHM (ps)
    pub fn new(efficiency: f64, timing_resolution_ps: f64) -> Self {
        Self {
            detector1_efficiency: efficiency.clamp(0.0, 1.0),
            detector2_efficiency: efficiency.clamp(0.0, 1.0),
            time_resolution_ps: timing_resolution_ps.max(1.0),
            coincidence_window_ns: 1.0,
            dark_count_rate_cps: 100.0,
        }
    }

    /// Expected coincidence rate (counts/s) at τ = 0.
    ///
    /// R_c = R₁ · R₂ · Δt · g²(0)
    /// where R₁, R₂ are the detected rates on each detector and Δt is the
    /// coincidence window.
    pub fn coincidence_rate(&self, signal_rate_cps: f64, g2_zero: f64) -> f64 {
        let r1 = signal_rate_cps * self.detector1_efficiency * 0.5 + self.dark_count_rate_cps;
        let r2 = signal_rate_cps * self.detector2_efficiency * 0.5 + self.dark_count_rate_cps;
        let dt = self.coincidence_window_ns * 1e-9;
        r1 * r2 * dt * g2_zero.max(0.0)
    }

    /// Accidental coincidence rate (counts/s).
    ///
    /// R_acc = R₁ · R₂ · Δt (the classical floor, g²(τ → ∞) = 1)
    pub fn accidental_rate(&self, rate_cps: f64) -> f64 {
        let r1 = rate_cps * self.detector1_efficiency * 0.5 + self.dark_count_rate_cps;
        let r2 = rate_cps * self.detector2_efficiency * 0.5 + self.dark_count_rate_cps;
        let dt = self.coincidence_window_ns * 1e-9;
        r1 * r2 * dt
    }

    /// Background-corrected g²(0).
    ///
    /// g²_corr(0) = (R_meas − R_acc) / R_lateral
    pub fn g2_zero_corrected(&self, raw_coincidences: f64, lateral_coincidences: f64) -> f64 {
        if lateral_coincidences <= 0.0 {
            return 0.0;
        }
        ((raw_coincidences - lateral_coincidences) / lateral_coincidences + 1.0).clamp(0.0, 2.0)
    }

    /// Required integration time (s) to achieve a target signal-to-noise ratio.
    ///
    /// For shot-noise limited HBT:
    ///   t = SNR² · R_acc / (R_sig − R_acc)²
    pub fn required_integration_time_s(&self, signal_rate: f64, target_snr: f64) -> f64 {
        let r_acc = self.accidental_rate(signal_rate);
        let r1 = signal_rate * self.detector1_efficiency * 0.5 + self.dark_count_rate_cps;
        let r2 = signal_rate * self.detector2_efficiency * 0.5 + self.dark_count_rate_cps;
        let dt = self.coincidence_window_ns * 1e-9;
        // True coincidence rate for a SPE (g²(0)→0): close to 0 by design
        // SNR-limited by the fluctuations in R_acc: σ = √(R_acc * t)
        // Signal = (R_true - R_acc), Noise = √(R_acc/t)
        // Minimum useful case: r_true = R_acc * (1 - epsilon)
        let r_signal = r1 * r2 * dt; // rate at g²=1 (accidentals)
        if r_signal <= 0.0 || r_acc <= 0.0 {
            return f64::INFINITY;
        }
        target_snr * target_snr * r_acc / r_signal
    }
}

// ─── PhotonNumberDistribution ─────────────────────────────────────────────────

/// Static helpers for photon number probability distributions.
///
/// Computes P(n) for Poissonian, thermal (Bose-Einstein), and Fock states.
pub struct PhotonNumberDistribution;

impl PhotonNumberDistribution {
    /// Poisson (coherent-state) distribution: P(n) = exp(−μ)·μⁿ/n!
    ///
    /// `n`: photon number, `mean`: mean photon number μ
    pub fn poisson(n: usize, mean: f64) -> f64 {
        if mean < 0.0 {
            return 0.0;
        }
        // Use log-sum to avoid factorial overflow
        let log_p = -(mean) + n as f64 * mean.ln().max(f64::NEG_INFINITY) - log_factorial(n);
        log_p.exp()
    }

    /// Thermal (Bose-Einstein) distribution: P(n) = μⁿ / (1+μ)^{n+1}
    ///
    /// Arises from a single-mode thermal field.
    pub fn thermal(n: usize, mean: f64) -> f64 {
        if mean < 0.0 {
            return 0.0;
        }
        let denom = 1.0 + mean;
        mean.powi(n as i32) / denom.powi((n + 1) as i32)
    }

    /// Fock state |1⟩: P(1) = 1, P(n≠1) = 0.
    pub fn fock_one(n: usize) -> f64 {
        if n == 1 {
            1.0
        } else {
            0.0
        }
    }

    /// Mandel Q parameter: Q = (⟨n²⟩ − ⟨n⟩²)/⟨n⟩ − 1.
    ///
    /// - Q < 0: sub-Poissonian (non-classical) — typical for single-photon sources
    /// - Q = 0: Poissonian (coherent state)
    /// - Q > 0: super-Poissonian (thermal, bunched light)
    ///
    /// `distribution`: probability values P(n) for n = 0, 1, 2, …
    pub fn mandel_q(distribution: &[f64]) -> f64 {
        let mean = distribution
            .iter()
            .enumerate()
            .map(|(n, p)| n as f64 * p)
            .sum::<f64>();
        if mean < 1e-30 {
            return 0.0;
        }
        let mean_sq = distribution
            .iter()
            .enumerate()
            .map(|(n, p)| (n as f64).powi(2) * p)
            .sum::<f64>();
        let variance = mean_sq - mean * mean;
        variance / mean - 1.0
    }

    /// Fano factor: F = ⟨Δn²⟩/⟨n⟩ = Q + 1.
    ///
    /// F = 1 for coherent light, F < 1 for sub-Poissonian, F > 1 for thermal.
    pub fn fano_factor(distribution: &[f64]) -> f64 {
        Self::mandel_q(distribution) + 1.0
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Compute ln(n!) using Stirling approximation for large n, exact for n ≤ 20.
fn log_factorial(n: usize) -> f64 {
    const LN_FACTORIALS: [f64; 21] = [
        0.0,                    // 0!
        0.0,                    // 1!
        std::f64::consts::LN_2, // 2!
        1.791_759_469,          // 3!
        3.178_053_830,          // 4!
        4.787_491_743,          // 5!
        6.579_251_212,          // 6!
        8.525_161_361,          // 7!
        10.604_602_902,         // 8!
        12.801_827_480,         // 9!
        15.104_412_573,         // 10!
        17.502_307_846,         // 11!
        19.987_214_496,         // 12!
        22.552_163_853,         // 13!
        25.191_221_181,         // 14!
        27.899_271_384,         // 15!
        30.671_860_106,         // 16!
        33.505_073_451,         // 17!
        36.395_445_208,         // 18!
        39.339_884_187,         // 19!
        42.335_616_461,         // 20!
    ];
    if n <= 20 {
        LN_FACTORIALS[n]
    } else {
        // Stirling: ln(n!) ≈ n*ln(n) - n + 0.5*ln(2πn)
        let nf = n as f64;
        nf * nf.ln() - nf + 0.5 * (2.0 * PI * nf).ln()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─ G2Function ─────────────────────────────────────────────────────────────

    #[test]
    fn test_g2_zero_at_zero_delay() {
        // For an ideal two-level emitter g²(0) = 0
        let g2_at_0 = G2Function::compute_g2(1.0, 0.0, 0.0);
        assert_eq!(g2_at_0, 0.0, "g²(0) should be exactly 0");
    }

    #[test]
    fn test_g2_approaches_unity_at_long_delay() {
        // At τ → ∞ the correlations vanish: g²(∞) = 1
        let g2_inf = G2Function::compute_g2(1.0, 0.0, 50.0); // 50 × τ_lifetime
        assert!(
            (g2_inf - 1.0).abs() < 1e-10,
            "g²(∞) should approach 1; got {g2_inf}"
        );
    }

    #[test]
    fn test_g2_histogram_g2_zero() {
        let hist = G2Function::new_single_photon_emitter(1.0, 101);
        let g2_0 = hist.g2_zero();
        assert!(
            g2_0 < 0.1,
            "g²(0) from histogram should be < 0.1 for SPE; got {g2_0:.4}"
        );
    }

    #[test]
    fn test_single_photon_purity_near_unity() {
        let hist = G2Function::new_single_photon_emitter(1.0, 101);
        let purity = hist.single_photon_purity();
        assert!(
            purity > 0.9,
            "Purity should be > 0.9 for ideal SPE; got {purity:.4}"
        );
    }

    #[test]
    fn test_g2_zero_from_areas() {
        // Perfect SPE: centre area = 0, lateral = 100 → g²(0) = 0
        let g2 = G2Function::g2_zero_from_areas(0.0, 100.0);
        assert_eq!(g2, 0.0);
        // Coherent: centre = lateral → g²(0) = 1
        let g2_coh = G2Function::g2_zero_from_areas(100.0, 100.0);
        assert!((g2_coh - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_interference_visibility_is_bounded() {
        let hist = G2Function::new_single_photon_emitter(1.0, 51);
        let v = hist.interference_visibility();
        assert!(
            (0.0..=1.0).contains(&v),
            "Visibility must be in [0,1]; got {v}"
        );
    }

    // ─ HbtSetup ───────────────────────────────────────────────────────────────

    #[test]
    fn test_hbt_accidentals_positive() {
        let hbt = HbtSetup::new(0.5, 50.0);
        let acc = hbt.accidental_rate(1e6);
        assert!(acc > 0.0, "Accidental rate must be positive; got {acc}");
    }

    #[test]
    fn test_hbt_coincidences_zero_for_ideal_spe() {
        let hbt = HbtSetup::new(0.5, 50.0);
        let r_c = hbt.coincidence_rate(1e6, 0.0);
        // g²(0) = 0 → only dark count contribution (very small)
        assert!(r_c < 1.0, "Coincidences should be near zero for ideal SPE");
    }

    #[test]
    fn test_hbt_corrected_g2_sensible() {
        let hbt = HbtSetup::new(0.5, 50.0);
        // If measured and lateral coincidences are equal → g²(0) = 1
        let g2_c = hbt.g2_zero_corrected(100.0, 100.0);
        assert!(
            (g2_c - 1.0).abs() < 0.01,
            "Corrected g²(0) should be 1 for equal peaks"
        );
    }

    // ─ PhotonNumberDistribution ───────────────────────────────────────────────

    #[test]
    fn test_poisson_normalises_to_one() {
        let mean = 2.5_f64;
        let total: f64 = (0..50)
            .map(|n| PhotonNumberDistribution::poisson(n, mean))
            .sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "Poisson must sum to 1; got {total}"
        );
    }

    #[test]
    fn test_thermal_normalises_to_one() {
        let mean = 1.0_f64;
        let total: f64 = (0..200)
            .map(|n| PhotonNumberDistribution::thermal(n, mean))
            .sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "Thermal distribution must sum to ~1; got {total}"
        );
    }

    #[test]
    fn test_fock_one_distribution() {
        assert_eq!(PhotonNumberDistribution::fock_one(1), 1.0);
        assert_eq!(PhotonNumberDistribution::fock_one(0), 0.0);
        assert_eq!(PhotonNumberDistribution::fock_one(2), 0.0);
    }

    #[test]
    fn test_mandel_q_coherent_is_zero() {
        // Poisson: Q = 0 exactly
        let mean = 3.0_f64;
        let dist: Vec<f64> = (0..50)
            .map(|n| PhotonNumberDistribution::poisson(n, mean))
            .collect();
        let q = PhotonNumberDistribution::mandel_q(&dist);
        assert!(
            q.abs() < 1e-4,
            "Mandel Q for coherent state should be ~0; got {q}"
        );
    }

    #[test]
    fn test_mandel_q_thermal_positive() {
        let mean = 2.0_f64;
        let dist: Vec<f64> = (0..200)
            .map(|n| PhotonNumberDistribution::thermal(n, mean))
            .collect();
        let q = PhotonNumberDistribution::mandel_q(&dist);
        assert!(
            q > 0.0,
            "Thermal state should have Q > 0 (super-Poissonian); got {q}"
        );
    }

    #[test]
    fn test_fano_factor_fock_one_sub_poissonian() {
        // |1⟩ Fock state: F = 0 (zero variance)
        let dist: Vec<f64> = (0..5).map(PhotonNumberDistribution::fock_one).collect();
        let f = PhotonNumberDistribution::fano_factor(&dist);
        assert!(
            f < 1.0,
            "Fano factor for Fock state should be < 1 (sub-Poissonian); got {f}"
        );
    }
}
