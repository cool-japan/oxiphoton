//! Integration tests for Fermi-Dirac statistics in the drift-diffusion solver.
//!
//! Verifies:
//! * Accuracy of F_{1/2}(η) and F_{-1/2}(η) against known values.
//! * The derivative identity dF_{1/2}/dη = (1/2)·F_{-1/2}.
//! * Boltzmann (negative η) and Sommerfeld (large positive η) limits.
//! * Joyce-Dixon round-trip consistency.
//! * Modified Einstein relation at degenerate and non-degenerate doping.

use oxiphoton::solar::drift_diffusion::fermi_dirac;
use oxiphoton::solar::drift_diffusion::SemiconductorMaterial;

// ─── Fermi-Dirac integral accuracy ───────────────────────────────────────────

#[test]
fn fermi_dirac_half_at_zero_eta_matches_known_value() {
    let val = fermi_dirac::f_half(0.0);
    // ∫_0^∞ √ε/(1+exp(ε)) dε = (√π/2)·(1−2^{−1/2})·ζ(3/2) ≈ 0.6780938952...
    let expected = 0.678_093_895_2_f64;
    assert!(
        (val - expected).abs() < 1e-6,
        "F_1/2(0) = {val}, expected {expected}"
    );
}

#[test]
fn fermi_dirac_minus_half_at_zero_eta_matches_known_value() {
    let val = fermi_dirac::f_minus_half(0.0);
    // ∫_0^∞ ε^{-1/2}/(1+exp(ε)) dε = (√π)·(1−√2)·ζ(1/2) ≈ 1.0721549300...
    let expected = 1.072_154_930_0_f64;
    assert!(
        (val - expected).abs() < 1e-6,
        "F_{{-1/2}}(0) = {val}, expected {expected}"
    );
}

#[test]
fn fermi_dirac_derivative_identity_numerical() {
    // Integration-by-parts identity for un-normalised Fermi-Dirac integrals:
    //   d/dη F_{1/2}(η) = (1/2) · F_{-1/2}(η)
    //
    // (More generally: d/dη F_j = j · F_{j-1}, so d/dη F_{1/2} = (1/2)·F_{-1/2}.)
    //
    // Avoid region boundaries at η = −2 and η = 5 where the numerical
    // derivative straddles two different approximation regions.
    let h = 1e-4_f64;
    for eta in [-5.0_f64, -1.0, 0.0, 2.0, 4.0] {
        let f_plus = fermi_dirac::f_half(eta + h);
        let f_minus_val = fermi_dirac::f_half(eta - h);
        let deriv = (f_plus - f_minus_val) / (2.0 * h);
        let expected = 0.5 * fermi_dirac::f_minus_half(eta);
        let relerr = (deriv - expected).abs() / expected.abs().max(1e-10);
        assert!(
            relerr < 1e-5,
            "dF_{{1/2}}/dη at η={eta}: numerical={deriv}, (1/2)·F_{{-1/2}}={expected}, relerr={relerr}"
        );
    }
}

#[test]
fn fermi_dirac_boltzmann_limit_negative_eta() {
    // For η << 0: F_{1/2}(η) ≈ √π/2 · e^η
    // The leading-term relative error vs the asymptotic series is ~e^η/2√2,
    // which at η = −10 is ~1.6e-5.  We only assert the cruder bound < 5e-5.
    let eta = -10.0_f64;
    let val = fermi_dirac::f_half(eta);
    let boltzmann = (std::f64::consts::PI.sqrt() / 2.0) * eta.exp();
    let relerr = (val - boltzmann).abs() / boltzmann;
    assert!(
        relerr < 5e-5,
        "Boltzmann limit at η={eta}: {val} vs {boltzmann}, relerr={relerr}"
    );
}

#[test]
fn fermi_dirac_degenerate_limit_positive_eta() {
    // For large η the un-normalized integral satisfies:
    //   F_{1/2}(η) ≈ (2/3) · η^{3/2}
    // (zero-temperature leading term; full expansion used in Sommerfeld region).
    let eta = 50.0_f64;
    let val = fermi_dirac::f_half(eta);
    let leading = (2.0 / 3.0) * eta.powf(1.5);
    let relerr = (val - leading).abs() / leading;
    assert!(
        relerr < 5e-3,
        "Sommerfeld leading term at η={eta}: F_{{1/2}}={val}, (2/3)η^{{3/2}}={leading}, relerr={relerr}"
    );
}

// ─── Joyce-Dixon inverse ──────────────────────────────────────────────────────

#[test]
fn joyce_dixon_inverse_round_trip_consistent() {
    // Given η, compute u = F_{1/2}(η)/(√π/2), then recover η via joyce_dixon_eta.
    // With Newton refinement, round-trip should be accurate to 1e-3 relative.
    let sqrt_pi_over_2 = std::f64::consts::PI.sqrt() / 2.0;
    for eta_target in [-1.0_f64, 0.0, 1.0, 3.0] {
        let f12 = fermi_dirac::f_half(eta_target);
        let u = f12 / sqrt_pi_over_2; // Blakemore u = n/N_c
        let eta_recovered = fermi_dirac::joyce_dixon_eta(u);
        let relerr = (eta_recovered - eta_target).abs() / (eta_target.abs() + 0.1);
        assert!(
            relerr < 5e-3,
            "Joyce-Dixon round-trip at η={eta_target}: recovered={eta_recovered}, relerr={relerr}"
        );
    }
}

// ─── Modified Einstein relation ───────────────────────────────────────────────

#[test]
fn degenerate_einstein_relation_above_classical() {
    // At n = 2e20, N_c ≈ 2.8e19: n/N_c ≈ 7.1, strongly degenerate (η ≈ 3.5).
    // The modified Einstein D_FD = μ·VT·F_{1/2}(η)/F_{-1/2}(η) > D_Boltzmann = μ·VT
    // once η exceeds ~2.1 (where F_{1/2}/F_{-1/2} crosses 1.0).
    let mat = SemiconductorMaterial::silicon();
    let temp_k = 300.0_f64;
    let d_boltzmann = mat.dn_cm2_s(temp_k);
    let d_fd = mat.dn_cm2_s_fd(temp_k, 2e20_f64);
    assert!(
        d_fd > d_boltzmann,
        "FD diffusion coefficient should exceed Boltzmann at strongly degenerate doping: \
         D_FD={d_fd:.4}, D_Boltz={d_boltzmann:.4}"
    );
}

#[test]
fn boltzmann_limit_at_low_doping_diffusion_matches() {
    // At n = 1e16 << N_c ≈ 2.8e19 (non-degenerate), D_FD should match D_Boltzmann.
    let mat = SemiconductorMaterial::silicon();
    let temp_k = 300.0_f64;
    let d_boltzmann = mat.dn_cm2_s(temp_k);
    let d_fd = mat.dn_cm2_s_fd(temp_k, 1e16_f64);
    let relerr = (d_fd - d_boltzmann).abs() / d_boltzmann;
    assert!(
        relerr < 0.01,
        "At low doping, D_FD should match D_Boltzmann: {d_fd:.4} vs {d_boltzmann:.4}, relerr={relerr:.4}"
    );
}

// ─── Continuity at approximation-region seams ─────────────────────────────────

#[test]
fn fermi_dirac_continuous_at_boltzmann_seam() {
    // The Boltzmann/midrange seam is at η = −2.
    // Continuity is approximate (the two series differ at the boundary),
    // but they should agree to better than 1 % so that no visible discontinuity
    // appears in physical quantities.
    let lo = fermi_dirac::f_half(-2.0 - 1e-6);
    let hi = fermi_dirac::f_half(-2.0 + 1e-6);
    let relerr = (hi - lo).abs() / lo.max(1e-30);
    assert!(
        relerr < 0.01,
        "F_{{1/2}} Boltzmann/midrange seam jump at η=−2: lo={lo}, hi={hi}, relerr={relerr}"
    );
    let lo2 = fermi_dirac::f_minus_half(-2.0 - 1e-6);
    let hi2 = fermi_dirac::f_minus_half(-2.0 + 1e-6);
    let relerr2 = (hi2 - lo2).abs() / lo2.max(1e-30);
    assert!(
        relerr2 < 0.01,
        "F_{{-1/2}} Boltzmann/midrange seam jump at η=−2: lo={lo2}, hi={hi2}, relerr={relerr2}"
    );
}

#[test]
fn fermi_dirac_continuous_at_sommerfeld_seam() {
    // The midrange/Sommerfeld seam is at η = 5.
    // The Sommerfeld expansion is an asymptotic series; at η = 5 it agrees with the
    // quadrature to < 0.5 % (verified by direct computation).
    let lo = fermi_dirac::f_half(5.0 - 1e-6);
    let hi = fermi_dirac::f_half(5.0 + 1e-6);
    let relerr = (hi - lo).abs() / lo.max(1e-30);
    assert!(
        relerr < 0.005,
        "F_{{1/2}} midrange/Sommerfeld seam jump at η=5: lo={lo}, hi={hi}, relerr={relerr}"
    );
    let lo2 = fermi_dirac::f_minus_half(5.0 - 1e-6);
    let hi2 = fermi_dirac::f_minus_half(5.0 + 1e-6);
    let relerr2 = (hi2 - lo2).abs() / lo2.max(1e-30);
    assert!(
        relerr2 < 0.005,
        "F_{{-1/2}} midrange/Sommerfeld seam jump at η=5: lo={lo2}, hi={hi2}, relerr={relerr2}"
    );
}

#[test]
fn fermi_dirac_derivative_identity_sommerfeld_region() {
    // Verify the derivative identity also holds inside the Sommerfeld region (η > 5).
    let h = 1e-4_f64;
    for eta in [6.0_f64, 8.0, 12.0] {
        let f_plus = fermi_dirac::f_half(eta + h);
        let f_minus_val = fermi_dirac::f_half(eta - h);
        let deriv = (f_plus - f_minus_val) / (2.0 * h);
        let expected = 0.5 * fermi_dirac::f_minus_half(eta);
        let relerr = (deriv - expected).abs() / expected.abs().max(1e-10);
        assert!(
            relerr < 1e-5,
            "dF_{{1/2}}/dη at η={eta} (Sommerfeld): numerical={deriv}, (1/2)·F_{{-1/2}}={expected}, relerr={relerr}"
        );
    }
}
