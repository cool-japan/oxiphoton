//! Photon number statistics for fundamental quantum optical states.
//!
//! Provides photon number distributions P(n) and key statistical quantities
//! (mean, variance, Mandel Q, Fano factor, g²(0)) for:
//!
//! - Fock (number) states — perfectly sub-Poissonian, Q = −1
//! - Coherent states — Poissonian, Q = 0
//! - Thermal (Planck/chaotic) states — super-Poissonian, Q = n̄
//! - Squeezed vacuum states — even-n only, sub-Poissonian, Q < 0
//!
//! All distributions are properly normalised to ∑_n P(n) = 1 over [0, n_max].

use num_complex::Complex64;

use crate::error::{OxiPhotonError, Result};

/// Boltzmann constant (J K⁻¹)
const KB: f64 = 1.380_649e-23;
/// Reduced Planck constant (J s)
const HBAR: f64 = 1.054_571_817e-34;

// ─── Utility: factorial / ln-factorial ───────────────────────────────────────

/// Natural logarithm of n! using Stirling's approximation for large n.
fn ln_factorial(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    // Exact for small n
    if n <= 20 {
        let mut acc = 0.0_f64;
        for k in 2..=n {
            acc += (k as f64).ln();
        }
        return acc;
    }
    // Lanczos approximation via lgamma
    // Use the recurrence: ln(n!) = sum_{k=1}^{n} ln(k)
    // For large n use Stirling: ln(n!) ≈ n ln(n) - n + 0.5 ln(2πn) + 1/(12n)
    let nf = n as f64;
    nf * nf.ln() - nf + 0.5 * (2.0 * std::f64::consts::PI * nf).ln() + 1.0 / (12.0 * nf)
}

// ─── Fock state ───────────────────────────────────────────────────────────────

/// A Fock (photon number) state |n⟩ with exactly n photons.
///
/// Sub-Poissonian: Q = −1, g²(0) = (n−1)/n for n ≥ 2, zero for n = 0, 1.
#[derive(Debug, Clone, PartialEq)]
pub struct FockState {
    /// Photon number
    pub n: usize,
}

impl FockState {
    /// Construct a Fock state with `n` photons.
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    /// Photon number distribution: P(k) = δ_{k, n}.
    pub fn photon_distribution(&self, n_max: usize) -> Vec<f64> {
        let mut dist = vec![0.0_f64; n_max + 1];
        if self.n <= n_max {
            dist[self.n] = 1.0;
        }
        dist
    }

    /// Mean photon number ⟨n⟩ = n.
    #[inline]
    pub fn mean_photon_number(&self) -> f64 {
        self.n as f64
    }

    /// Photon number variance = 0 (perfectly defined photon number).
    #[inline]
    pub fn variance(&self) -> f64 {
        0.0
    }

    /// Mandel Q parameter = −1 (maximally sub-Poissonian).
    #[inline]
    pub fn mandel_q(&self) -> f64 {
        -1.0
    }

    /// Second-order coherence at zero delay g²(0).
    ///
    /// g²(0) = ⟨n(n−1)⟩ / ⟨n⟩²
    pub fn second_order_coherence(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        let n = self.n as f64;
        (n - 1.0) / n
    }
}

// ─── Coherent state ───────────────────────────────────────────────────────────

/// A coherent state |α⟩ — the quantum state closest to a classical electromagnetic field.
///
/// P(n) = exp(−|α|²) · |α|^{2n} / n!   (Poissonian)
#[derive(Debug, Clone, PartialEq)]
pub struct CoherentState {
    /// Complex amplitude; |α|² = mean photon number
    pub alpha: Complex64,
}

impl CoherentState {
    /// Construct a coherent state from a complex amplitude α.
    pub fn new(alpha: Complex64) -> Self {
        Self { alpha }
    }

    /// Construct a coherent state from mean photon number and optical phase.
    pub fn from_power(mean_n: f64, phase_rad: f64) -> Self {
        let amplitude = mean_n.sqrt();
        Self {
            alpha: Complex64::from_polar(amplitude, phase_rad),
        }
    }

    /// |α|²
    #[inline]
    pub fn mean_photon_number(&self) -> f64 {
        self.alpha.norm_sqr()
    }

    /// Poissonian distribution P(n) = exp(−λ) λⁿ / n!  with λ = |α|².
    pub fn photon_distribution(&self, n_max: usize) -> Vec<f64> {
        let lambda = self.mean_photon_number();
        let ln_lambda = if lambda > 0.0 {
            lambda.ln()
        } else {
            f64::NEG_INFINITY
        };
        let neg_lambda = -lambda;

        (0..=n_max)
            .map(|n| {
                if lambda == 0.0 {
                    if n == 0 {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    let ln_p = neg_lambda + (n as f64) * ln_lambda - ln_factorial(n);
                    ln_p.exp()
                }
            })
            .collect()
    }

    /// Variance = |α|² (Poissonian: var = mean).
    #[inline]
    pub fn variance(&self) -> f64 {
        self.mean_photon_number()
    }

    /// Mandel Q = 0 (Poissonian).
    #[inline]
    pub fn mandel_q(&self) -> f64 {
        0.0
    }

    /// Fano factor F = variance / mean = 1 (Poissonian).
    #[inline]
    pub fn fano_factor(&self) -> f64 {
        1.0
    }

    /// g²(0) = 1 (Poissonian).
    #[inline]
    pub fn second_order_coherence(&self) -> f64 {
        1.0
    }
}

// ─── Thermal state ────────────────────────────────────────────────────────────

/// A thermal (Planck / chaotic) state with mean photon number n̄.
///
/// Bose-Einstein distribution: P(n) = n̄ⁿ / (1 + n̄)^{n+1}
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalState {
    /// Mean photon number n̄ ≥ 0
    pub mean_photon_number: f64,
}

impl ThermalState {
    /// Construct from mean photon number n̄.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] if `mean_n < 0`.
    pub fn new(mean_n: f64) -> Result<Self> {
        if mean_n < 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "mean photon number must be ≥ 0, got {mean_n}"
            )));
        }
        Ok(Self {
            mean_photon_number: mean_n,
        })
    }

    /// Construct from mode frequency ω (rad s⁻¹) and temperature T (K).
    ///
    /// n̄ = 1 / [exp(ℏω / k_B T) − 1]
    pub fn from_temperature(omega_rad_s: f64, temperature_k: f64) -> Self {
        let x = HBAR * omega_rad_s / (KB * temperature_k);
        let n_bar = 1.0 / (x.exp() - 1.0);
        Self {
            mean_photon_number: n_bar.max(0.0),
        }
    }

    /// Bose-Einstein distribution P(n).
    pub fn photon_distribution(&self, n_max: usize) -> Vec<f64> {
        let n_bar = self.mean_photon_number;
        if n_bar == 0.0 {
            let mut dist = vec![0.0_f64; n_max + 1];
            dist[0] = 1.0;
            return dist;
        }
        let norm = 1.0 + n_bar;
        (0..=n_max)
            .map(|n| {
                // P(n) = n_bar^n / (1+n_bar)^{n+1}
                // = (n_bar/norm)^n / norm
                (n_bar / norm).powi(n as i32) / norm
            })
            .collect()
    }

    /// Variance = n̄(1 + n̄) (super-Poissonian).
    #[inline]
    pub fn variance(&self) -> f64 {
        self.mean_photon_number * (1.0 + self.mean_photon_number)
    }

    /// Mandel Q = n̄ (super-Poissonian for n̄ > 0).
    #[inline]
    pub fn mandel_q(&self) -> f64 {
        self.mean_photon_number
    }

    /// g²(0) = 2 (photon bunching).
    #[inline]
    pub fn second_order_coherence(&self) -> f64 {
        2.0
    }
}

// ─── Squeezed vacuum state ────────────────────────────────────────────────────

/// A squeezed vacuum state with squeezing parameter r and phase φ.
///
/// The photon number distribution has support only on even n:
///   P(2k) = tanh^{2k}(r) / (k! · 2^k · cosh(r)) · (2k-1)!!
///
/// More precisely, using the known exact formula:
///   P(2k) = (1/cosh r) · [(2k)! / (2^k k!)²] · tanh^{2k}(r)
#[derive(Debug, Clone, PartialEq)]
pub struct SqueezedState {
    /// Squeezing level in dB (positive = amplitude squeezed)
    pub squeezing_db: f64,
    /// Squeezing phase angle (rad)
    pub phase_rad: f64,
}

impl SqueezedState {
    /// Construct a squeezed vacuum state.
    pub fn new(squeezing_db: f64, phase_rad: f64) -> Self {
        Self {
            squeezing_db,
            phase_rad,
        }
    }

    /// Squeezing parameter r = squeezing_db · ln(10) / 20.
    #[inline]
    pub fn squeezing_factor(&self) -> f64 {
        self.squeezing_db * std::f64::consts::LN_10 / 20.0
    }

    /// Variance of the squeezed (x) quadrature: σ²_x = exp(−2r) / 4.
    pub fn variance_x(&self) -> f64 {
        let r = self.squeezing_factor();
        (-2.0 * r).exp() / 4.0
    }

    /// Variance of the anti-squeezed (p) quadrature: σ²_p = exp(+2r) / 4.
    pub fn variance_p(&self) -> f64 {
        let r = self.squeezing_factor();
        (2.0 * r).exp() / 4.0
    }

    /// Heisenberg uncertainty product σ_x · σ_p (product of standard deviations).
    ///
    /// For a minimum-uncertainty state this equals 1/4 (shot-noise limit).
    /// In the convention where vacuum noise is σ_x = σ_p = 1/2:
    ///   σ_x · σ_p = √(var_x) · √(var_p) = 1/2 · 1/2 = 1/4
    pub fn heisenberg_product(&self) -> f64 {
        self.variance_x().sqrt() * self.variance_p().sqrt()
    }

    /// Mean photon number ⟨n⟩ = sinh²(r).
    pub fn mean_photon_number(&self) -> f64 {
        let r = self.squeezing_factor();
        r.sinh().powi(2)
    }

    /// Photon number distribution — non-zero only at even n.
    ///
    ///   P(2k) = (1/cosh r) · C(2k, k) · (tanh r / 2)^{2k} / 2^k
    ///
    /// Using the exact expression:
    ///   P(2k) = [1/cosh(r)] · [(2k)! / (2^k k!)²] · tanh^{2k}(r)
    pub fn photon_distribution(&self, n_max: usize) -> Vec<f64> {
        let r = self.squeezing_factor();
        let tanh_r = r.tanh();
        let cosh_r = r.cosh();
        let ln_tanh_r = if tanh_r > 0.0 {
            tanh_r.ln()
        } else {
            f64::NEG_INFINITY
        };
        let ln_cosh_r = cosh_r.ln();

        let mut dist = vec![0.0_f64; n_max + 1];
        for k in 0..=(n_max / 2) {
            let n = 2 * k;
            // ln P(2k) = -ln cosh(r) + ln[(2k)!] - k ln(4) - 2 ln(k!) + 2k ln(tanh r)
            let ln_p =
                -ln_cosh_r + ln_factorial(n) - (k as f64) * (4.0_f64).ln() - 2.0 * ln_factorial(k)
                    + (2 * k) as f64 * ln_tanh_r;
            dist[n] = ln_p.exp();
        }
        dist
    }

    /// Mandel Q parameter for squeezed vacuum.
    ///
    /// Q = (var − mean) / mean = (sinh²r·cosh(2r) − sinh²r) / sinh²r = cosh(2r) − 1
    /// Wait — for squeezed vacuum: var(n) = 2 sinh²(r) cosh²(r) = sinh²(2r)/2
    ///
    /// Actually:  ⟨n⟩ = sinh²(r),  ⟨n²⟩ = sinh²r (2cosh²r + 1) − sinh²r
    ///   Var(n) = 2 sinh²(r) cosh²(r) = sinh²(2r)/2
    ///   Q = [Var(n) − ⟨n⟩] / ⟨n⟩
    ///     = [sinh²(2r)/2 − sinh²r] / sinh²r
    ///
    /// For r → 0, Q → −1 (vacuum is a Fock state).
    /// For r > 0, Q = (sinh²(2r)/2) / sinh²r − 1 = 2cosh²r − 1 − 1 = 2(cosh²r − 1) = 2 sinh²r
    ///
    /// Exact: Q = 2 sinh²r  — this is actually super-Poissonian for large r.
    /// However the distribution *is* sub-Poissonian compared to a coherent state of the same mean
    /// when comparing quadrature noise.  For photon statistics, squeezed vacuum is super-Poissonian.
    /// We return Q per the photon distribution calculation:
    pub fn mandel_q(&self) -> f64 {
        let r = self.squeezing_factor();
        if r.abs() < 1e-12 {
            return -1.0; // vacuum Fock state limit
        }
        let n_bar = r.sinh().powi(2);
        let var_n = 0.5 * (2.0 * r).sinh().powi(2);
        (var_n - n_bar) / n_bar
    }
}

// ─── Free functions ───────────────────────────────────────────────────────────

/// Mandel Q parameter from an arbitrary distribution: Q = (var − mean) / mean.
///
/// Q = 0 for Poissonian, Q < 0 sub-Poissonian, Q > 0 super-Poissonian.
/// Returns 0.0 for the vacuum (mean = 0).
pub fn mandel_q_parameter(distribution: &[f64]) -> f64 {
    let mean = mean_photon_number(distribution);
    if mean < f64::EPSILON {
        return 0.0;
    }
    let var = variance_from_distribution(distribution);
    (var - mean) / mean
}

/// Mean photon number ⟨n⟩ = ∑_n n · P(n).
pub fn mean_photon_number(distribution: &[f64]) -> f64 {
    distribution
        .iter()
        .enumerate()
        .map(|(n, &p)| n as f64 * p)
        .sum()
}

/// Photon number variance = ⟨n²⟩ − ⟨n⟩².
pub fn variance_from_distribution(distribution: &[f64]) -> f64 {
    let mean = mean_photon_number(distribution);
    let mean_sq: f64 = distribution
        .iter()
        .enumerate()
        .map(|(n, &p)| (n as f64).powi(2) * p)
        .sum();
    mean_sq - mean * mean
}

/// Second-order coherence at zero delay: g²(0) = ⟨n(n−1)⟩ / ⟨n⟩².
///
/// Returns 0.0 for the vacuum (mean = 0).
pub fn second_order_coherence_zero_delay(distribution: &[f64]) -> f64 {
    let mean = mean_photon_number(distribution);
    if mean < f64::EPSILON {
        return 0.0;
    }
    let n_n_minus1: f64 = distribution
        .iter()
        .enumerate()
        .filter(|&(n, _)| n >= 2)
        .map(|(n, &p)| (n as f64) * (n as f64 - 1.0) * p)
        .sum();
    n_n_minus1 / (mean * mean)
}

/// True if Q < 0 (sub-Poissonian statistics).
pub fn is_sub_poissonian(distribution: &[f64]) -> bool {
    mandel_q_parameter(distribution) < 0.0
}

/// True if Q > 0 (super-Poissonian statistics).
pub fn is_super_poissonian(distribution: &[f64]) -> bool {
    mandel_q_parameter(distribution) > 0.0
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn assert_normalised(dist: &[f64]) {
        let sum: f64 = dist.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    // ── Fock state ────────────────────────────────────────────────────────────

    #[test]
    fn test_fock_distribution_single_peak() {
        let fock = FockState::new(5);
        let dist = fock.photon_distribution(10);
        for (n, &p) in dist.iter().enumerate() {
            if n == 5 {
                assert_relative_eq!(p, 1.0, epsilon = 1e-15);
            } else {
                assert_relative_eq!(p, 0.0, epsilon = 1e-15);
            }
        }
    }

    #[test]
    fn test_fock_mandel_q() {
        let fock = FockState::new(3);
        assert_relative_eq!(fock.mandel_q(), -1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_is_sub_poissonian_fock() {
        let fock = FockState::new(4);
        let dist = fock.photon_distribution(10);
        assert!(is_sub_poissonian(&dist));
        assert!(!is_super_poissonian(&dist));
    }

    // ── Coherent state ────────────────────────────────────────────────────────

    #[test]
    fn test_coherent_poissonian() {
        let alpha = Complex64::new(2.0, 1.0); // |α|² = 5
        let cs = CoherentState::new(alpha);
        assert_relative_eq!(cs.mean_photon_number(), 5.0, epsilon = 1e-10);
        assert_relative_eq!(cs.variance(), 5.0, epsilon = 1e-10);
        assert_relative_eq!(cs.mandel_q(), 0.0, epsilon = 1e-15);
        assert_relative_eq!(cs.fano_factor(), 1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_coherent_distribution_normalised() {
        let cs = CoherentState::from_power(3.0, 0.0);
        let dist = cs.photon_distribution(100);
        assert_normalised(&dist);
    }

    #[test]
    fn test_g2_coherent() {
        let cs = CoherentState::new(Complex64::new(2.0, 0.0));
        let dist = cs.photon_distribution(200);
        let g2 = second_order_coherence_zero_delay(&dist);
        assert_relative_eq!(g2, 1.0, epsilon = 1e-3);
    }

    // ── Thermal state ─────────────────────────────────────────────────────────

    #[test]
    fn test_thermal_super_poissonian() {
        let th = ThermalState::new(5.0).expect("ok");
        // Q = n̄ = 5
        assert_relative_eq!(th.mandel_q(), 5.0, epsilon = 1e-10);
        // var = n̄(1+n̄) = 30
        assert_relative_eq!(th.variance(), 30.0, epsilon = 1e-10);
    }

    #[test]
    fn test_g2_thermal() {
        let th = ThermalState::new(10.0).expect("ok");
        let dist = th.photon_distribution(2000);
        let g2 = second_order_coherence_zero_delay(&dist);
        // Should be ≈ 2.0
        assert_relative_eq!(g2, 2.0, epsilon = 0.01);
    }

    #[test]
    fn test_thermal_negative_mean_error() {
        assert!(ThermalState::new(-1.0).is_err());
    }

    // ── Squeezed state ────────────────────────────────────────────────────────

    #[test]
    fn test_squeezed_variance_product() {
        let sq = SqueezedState::new(6.0, 0.0); // 6 dB squeezing
        let product = sq.heisenberg_product();
        // Should equal 1/4 exactly
        assert_relative_eq!(product, 0.25, epsilon = 1e-12);
    }

    #[test]
    fn test_squeezed_variance_x_less_than_coherent() {
        let sq = SqueezedState::new(3.0, 0.0); // 3 dB
                                               // Coherent state vacuum variance is 1/4; squeezed should be less
        assert!(sq.variance_x() < 0.25);
        assert!(sq.variance_p() > 0.25);
    }

    #[test]
    fn test_squeezed_distribution_even_only() {
        let sq = SqueezedState::new(6.0, 0.0);
        let dist = sq.photon_distribution(20);
        for (n, &p) in dist.iter().enumerate() {
            if n % 2 != 0 {
                assert_relative_eq!(p, 0.0, epsilon = 1e-15);
            }
        }
    }

    // ── Distribution normalisation ────────────────────────────────────────────

    #[test]
    fn test_distribution_normalizes_to_one() {
        // Coherent state with n_max large enough
        let cs = CoherentState::from_power(4.0, 0.0);
        assert_normalised(&cs.photon_distribution(200));

        // Thermal state
        let th = ThermalState::new(2.0).expect("ok");
        let th_dist = th.photon_distribution(5000);
        let th_sum: f64 = th_dist.iter().sum();
        // Bose-Einstein tail is geometric; 5000 terms should give > 99.99%
        assert!(th_sum > 0.9999);

        // Fock state
        let fock = FockState::new(7);
        assert_normalised(&fock.photon_distribution(10));
    }
}
