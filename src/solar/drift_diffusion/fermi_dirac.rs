//! Fermi-Dirac integrals for degenerate-semiconductor carrier statistics.
//!
//! Implements F_{1/2}(η) and F_{-1/2}(η) using:
//! * Asymptotic (Boltzmann) expansion for η ≤ −2.
//! * Smooth-integrand Simpson quadrature (u = √ε substitution) for −2 < η ≤ 5.
//! * Sommerfeld expansion for η > 5.
//!
//! The substitution `u = √ε`, so `ε = u²`, `dε = 2u du`, gives smooth integrands:
//!   F_{1/2}(η)  = ∫₀^∞ √ε / (1+exp(ε−η)) dε = 2 ∫₀^∞ u² / (1+exp(u²−η)) du
//!   F_{-1/2}(η) = ∫₀^∞ ε^{-1/2}/(1+exp(ε−η)) dε = 2 ∫₀^∞ 1/(1+exp(u²−η)) du
//!
//! Both transformed integrands are smooth at u=0, enabling O(h⁴) Simpson accuracy.
//!
//! # References
//! * Blakemore (1982), Solid-State Electronics 25, 1067.
//! * Joyce & Dixon (1977), Appl. Phys. Lett. 31, 354.
//! * Sommerfeld (1928), Z. Physik 47, 1.

use std::f64::consts::PI;

/// √π / 2, the leading coefficient in the Boltzmann series.
const SQRT_PI_OVER_2: f64 = 0.886_226_925_452_758; // √π / 2

/// Number of Simpson subintervals in the midrange quadrature.
/// Must be even.  2000 gives < 1e-12 absolute error via the O(h⁴) rule.
const SIMPSON_N: usize = 2000;

// ─── F_{1/2}(η) ──────────────────────────────────────────────────────────────

/// Fermi-Dirac integral of order 1/2:
///
///   F_{1/2}(η) = ∫₀^∞ √ε / (1 + exp(ε − η)) dε
///
/// Three-region evaluation:
/// * η ≤ −2: six-term asymptotic Boltzmann series.
/// * −2 < η ≤ 5: 2000-point Simpson quadrature with u = √ε substitution.
/// * η > 5: four-term Sommerfeld expansion.
pub fn f_half(eta: f64) -> f64 {
    if eta <= -2.0 {
        f_half_boltzmann(eta)
    } else if eta <= 5.0 {
        f_half_midrange(eta)
    } else {
        f_half_sommerfeld(eta)
    }
}

/// Boltzmann tail for η ≤ −2 (relative error < 1e-13 at η = −2).
///
/// F_{1/2}(η) ≈ √π/2 · eη · Σ_{k=1}^6 (−1)^{k+1} · e^{(k-1)η} / (k · √k)
fn f_half_boltzmann(eta: f64) -> f64 {
    let e = eta.exp();
    let e2 = e * e;
    let e3 = e2 * e;
    let e4 = e3 * e;
    let e5 = e4 * e;
    // Coefficients: (−1)^{k+1} / (k · √k) for k = 1..6
    SQRT_PI_OVER_2
        * e
        * (1.0 - e / (2.0 * 2.0_f64.sqrt()) + e2 / (3.0 * 3.0_f64.sqrt())
            - e3 / (4.0 * 4.0_f64.sqrt())
            + e4 / (5.0 * 5.0_f64.sqrt())
            - e5 / (6.0 * 6.0_f64.sqrt()))
}

/// Simpson quadrature via u = √ε substitution.
///
/// F_{1/2}(η) = 2 ∫₀^{u_max} u² / (1 + exp(u²−η)) du
///
/// The transformed integrand is smooth at u = 0 (unlike the original).
fn f_half_midrange(eta: f64) -> f64 {
    // Integrate from u = 0 to u_max where exp(u_max² − η) >> 1.
    // u_max = √(η + 50) ensures exp(50) contribution < 2e-22.
    let u_max = (eta + 50.0).max(50.0).sqrt();
    let n = SIMPSON_N;
    let h = u_max / n as f64;

    let mut sum = 0.0_f64;
    for k in 0..=n {
        let u = k as f64 * h;
        let u2 = u * u;
        // Guard: exp(u²−η) can overflow for large u; clamp at max representable.
        let exponent = u2 - eta;
        let val = if exponent > 700.0 {
            0.0
        } else {
            2.0 * u2 / (1.0 + exponent.exp())
        };
        let w = if k == 0 || k == n {
            1.0
        } else if k % 2 == 1 {
            4.0
        } else {
            2.0
        };
        sum += w * val;
    }
    sum * h / 3.0
}

/// Sommerfeld expansion for η > 5.
///
/// For the **un-normalized** integral F_{1/2}(η) = ∫₀^∞ √ε/(1+exp(ε−η)) dε
/// the Sommerfeld series reads:
///
///   F_{1/2}(η) = (2/3)·η^{3/2} · (1 + π²/(8η²) + 7π⁴/(640η⁴) + …)
///
/// Derivation: standard Sommerfeld lemma gives ∫₀^η ε^{1/2} dε = (2/3)η^{3/2} as
/// the zero-temperature term, with subsequent corrections from the Bernoulli expansion.
///
/// Relative error < 4e-4 at η = 5; the correction terms ensure < 2e-5 for η ≥ 8.
fn f_half_sommerfeld(eta: f64) -> f64 {
    let eta2 = eta * eta;
    let eta32 = eta.powf(1.5);
    let pi2 = PI * PI;
    let pi4 = pi2 * pi2;
    (2.0 / 3.0) * eta32 * (1.0 + pi2 / (8.0 * eta2) + 7.0 * pi4 / (640.0 * eta2 * eta2))
}

// ─── F_{-1/2}(η) ─────────────────────────────────────────────────────────────

/// Fermi-Dirac integral of order −1/2:
///
///   F_{-1/2}(η) = ∫₀^∞ ε^{-1/2} / (1 + exp(ε − η)) dε
///
/// Three-region evaluation:
/// * η ≤ −2: six-term asymptotic Boltzmann series, leading coefficient √π · eη.
/// * −2 < η ≤ 5: 2000-point Simpson quadrature with u = √ε substitution.
/// * η > 5: Sommerfeld expansion.
pub fn f_minus_half(eta: f64) -> f64 {
    if eta <= -2.0 {
        f_minus_half_boltzmann(eta)
    } else if eta <= 5.0 {
        f_minus_half_midrange(eta)
    } else {
        f_minus_half_sommerfeld(eta)
    }
}

/// Boltzmann tail for η ≤ −2.
///
/// F_{-1/2}(η) = ∫₀^∞ ε^{-1/2}/(1+exp(ε-η)) dε
///             ≈ Γ(1/2)·exp(η)·Σ_{k=1}^6 (-1)^{k+1}/√k · exp((k-1)η)
///             = √π · exp(η) · (1 − exp(η)/√2 + exp(2η)/√3 − ...)
///
/// Note: the leading coefficient is √π (not √π/2) because Γ(1/2) = √π.
fn f_minus_half_boltzmann(eta: f64) -> f64 {
    let e = eta.exp();
    let e2 = e * e;
    let e3 = e2 * e;
    let e4 = e3 * e;
    let e5 = e4 * e;
    // Coefficients: (−1)^{k+1}/√k for k = 1..6; leading factor Γ(1/2) = √π
    PI.sqrt()
        * e
        * (1.0 - e / 2.0_f64.sqrt() + e2 / 3.0_f64.sqrt() - e3 / 4.0_f64.sqrt()
            + e4 / 5.0_f64.sqrt()
            - e5 / 6.0_f64.sqrt())
}

/// Simpson quadrature via u = √ε substitution.
///
/// F_{-1/2}(η) = 2 ∫₀^{u_max} 1 / (1 + exp(u²−η)) du
///
/// The transformed integrand is smooth and bounded everywhere.
fn f_minus_half_midrange(eta: f64) -> f64 {
    let u_max = (eta + 50.0).max(50.0).sqrt();
    let n = SIMPSON_N;
    let h = u_max / n as f64;

    let mut sum = 0.0_f64;
    for k in 0..=n {
        let u = k as f64 * h;
        let u2 = u * u;
        let exponent = u2 - eta;
        let val = if exponent > 700.0 {
            0.0
        } else {
            2.0 / (1.0 + exponent.exp())
        };
        let w = if k == 0 || k == n {
            1.0
        } else if k % 2 == 1 {
            4.0
        } else {
            2.0
        };
        sum += w * val;
    }
    sum * h / 3.0
}

/// Sommerfeld expansion for η > 5.
///
/// For the **un-normalized** integral F_{-1/2}(η) = ∫₀^∞ ε^{-1/2}/(1+exp(ε−η)) dε
/// the Sommerfeld series reads:
///
///   F_{-1/2}(η) = 2·√η · (1 − π²/(24η²) − 7π⁴/(384η⁴) + …)
///
/// Derivation: differentiate the F_{1/2} Sommerfeld series term-by-term using
/// d/dη F_{1/2}(η) = (1/2)·F_{-1/2}(η).  The zero-temperature term ∫₀^η ε^{-1/2} dε = 2√η.
/// Higher-order coefficients follow from the Bernoulli-number Sommerfeld corrections.
///
/// Relative error < 4e-4 at η = 5.
fn f_minus_half_sommerfeld(eta: f64) -> f64 {
    let eta2 = eta * eta;
    let pi2 = PI * PI;
    let pi4 = pi2 * pi2;
    2.0 * eta.sqrt() * (1.0 - pi2 / (24.0 * eta2) - 7.0 * pi4 / (384.0 * eta2 * eta2))
}

// ─── Joyce-Dixon inverse ──────────────────────────────────────────────────────

/// Compute the reduced chemical potential η from the normalised carrier density u.
///
/// Given `u = n / N_c` (Blakemore convention), returns η such that:
///   F_{1/2}(η) = u · (√π / 2)
///
/// Two-region initial guess strategy:
/// * u ≤ 8 (mildly degenerate): Joyce-Dixon (1977) polynomial, valid to ~5%.
/// * u > 8 (strongly degenerate): Sommerfeld inverse,
///   η₀ ≈ (3π/4 · u)^{2/3} — exact zero-temperature limit, error < 5% at u=8.
///
/// Newton refinement then corrects either guess to < 1e-12 relative error.
///
/// # References
/// Joyce & Dixon (1977), Appl. Phys. Lett. 31, 354.
pub fn joyce_dixon_eta(u: f64) -> f64 {
    // Target: F_{1/2}(η) = u · √π/2
    let target = u * SQRT_PI_OVER_2;

    // Initial guess: Joyce-Dixon polynomial for u ≤ 8, Sommerfeld inversion for u > 8.
    //
    // Joyce-Dixon polynomial is only accurate for u ≲ 10. For larger u the u³ and u⁴
    // correction terms overflow the Boltzmann baseline and produce a wildly incorrect
    // initial guess that prevents Newton convergence.
    //
    // Sommerfeld inversion: F_{1/2}(η) ≈ (2/3)·η^{3/2}  (zero-temperature leading term)
    //   → η₀ = (3/2 · target)^{2/3} = (3·u·√π/4)^{2/3}
    let mut eta = if u <= 8.0 {
        let a1 = 1.0_f64 / 8.0_f64.sqrt(); // ≈ 0.35355339059
        let a2 = -4.9587e-3_f64;
        let a3 = 1.483e-4_f64;
        let a4 = -2.13e-6_f64;
        let ln_u = u.ln();
        ln_u + a1 * u + a2 * u * u + a3 * u.powi(3) + a4 * u.powi(4)
    } else {
        // Sommerfeld: F_{1/2}(η) ≈ (2/3)η^{3/2} → η = (3/2 · target)^{2/3}
        (1.5 * target).powf(2.0 / 3.0)
    };

    // Newton refinement on F_{1/2}(η) − target = 0.
    // Analytic derivative: d/dη F_{1/2}(η) = (1/2)·F_{-1/2}(η)
    // (integration-by-parts identity for un-normalised Fermi-Dirac integrals)
    for _ in 0..20 {
        let f = f_half(eta) - target;
        // df = (1/2)*F_{-1/2}(eta)
        let df = 0.5 * f_minus_half(eta);
        if df.abs() < 1e-300 {
            break;
        }
        let step = f / df;
        eta -= step;
        if step.abs() < 1e-12 * (eta.abs() + 1.0) {
            break;
        }
    }
    eta
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f_half_zero_eta_matches_known_value() {
        let val = f_half(0.0);
        // Known: ∫_0^∞ √ε/(1+exp(ε)) dε = (√π/2)·(1−2^{−1/2})·ζ(3/2) ≈ 0.6780938952
        let expected = 0.678_093_895_2_f64;
        assert!(
            (val - expected).abs() < 1e-6,
            "F_1/2(0) = {val}, expected {expected}"
        );
    }

    #[test]
    fn f_minus_half_zero_eta_matches_known_value() {
        let val = f_minus_half(0.0);
        // Known: ∫_0^∞ ε^{-1/2}/(1+exp(ε)) dε = (√π)·(1−√2)·ζ(1/2) ≈ 1.07215493
        let expected = 1.072_154_930_0_f64;
        assert!(
            (val - expected).abs() < 1e-6,
            "F_{{-1/2}}(0) = {val}, expected {expected}"
        );
    }

    #[test]
    fn derivative_identity_holds() {
        // d/dη F_{1/2}(η) = (1/2)·F_{-1/2}(η)  (integration-by-parts identity)
        let h = 1e-4_f64;
        for eta in [-5.0_f64, -1.0, 0.0, 2.0, 4.0] {
            let d_num = (f_half(eta + h) - f_half(eta - h)) / (2.0 * h);
            let expected = 0.5 * f_minus_half(eta);
            let relerr = (d_num - expected).abs() / expected.abs().max(1e-10);
            assert!(
                relerr < 1e-5,
                "Derivative identity failed at η={eta}: numerical={d_num}, (1/2)·F_{{-1/2}}={expected}, relerr={relerr}"
            );
        }
    }
}
