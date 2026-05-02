//! Tests for the Harmon 1994 effective BGN model for GaAs.
//!
//! Reference: Harmon, Melloch & Lundstrom (1994), Appl. Phys. Lett. 64(4):502.
//! DOI: 10.1063/1.111075. (Purdue ePubs open access.)

use oxiphoton::solar::drift_diffusion::bandgap_narrowing::{harmon1994_gaas_delta_eg_ev, BgnModel};
use oxiphoton::solar::drift_diffusion::SemiconductorMaterial;

#[test]
fn bgn_near_zero_at_intrinsic_doping() {
    // Below the N = 1e14 cm⁻³ threshold the function returns 0.0 exactly.
    // This reflects the Harmon 1994 measurement regime: the model is only validated
    // for N ≥ 1e14 cm⁻³, below which BGN is negligible by definition.
    let delta_below = harmon1994_gaas_delta_eg_ev(1e13, true);
    assert_eq!(
        delta_below, 0.0,
        "Expected exactly 0.0 below threshold (N=1e13), got {:.4e} eV",
        delta_below
    );

    // At the threshold boundary N = 1e14 cm⁻³, the value is small (< 2 meV).
    let delta_at = harmon1994_gaas_delta_eg_ev(1e14, true);
    assert!(
        delta_at < 2e-3,
        "Expected < 2 meV at N=1e14, got {:.4e} eV",
        delta_at
    );
}

#[test]
fn bgn_n_gaas_physically_reasonable_at_1e18() {
    // Harmon 1994: A = 3.23e-8 eV·cm for n-GaAs → ΔEg = 32.3 meV at N = 1e18 cm⁻³
    let delta = harmon1994_gaas_delta_eg_ev(1e18, true);
    assert!(
        (0.020..=0.060).contains(&delta),
        "n-GaAs BGN at 1e18 should be 20–60 meV, got {:.1} meV",
        delta * 1000.0
    );
}

#[test]
fn bgn_p_gaas_physically_reasonable_at_1e18() {
    // Harmon 1994: A = 2.55e-8 eV·cm for p-GaAs → ΔEg = 25.5 meV at N = 1e18 cm⁻³
    let delta = harmon1994_gaas_delta_eg_ev(1e18, false);
    assert!(
        (0.015..=0.050).contains(&delta),
        "p-GaAs BGN at 1e18 should be 15–50 meV, got {:.1} meV",
        delta * 1000.0
    );
}

#[test]
fn bgn_increases_monotonically_with_doping() {
    // BGN must be monotonically increasing with N (N^(1/3) is strictly monotone).
    let n_values = [1e16_f64, 1e17, 1e18, 1e19, 1e20];
    let deltas: Vec<f64> = n_values
        .iter()
        .map(|&n| harmon1994_gaas_delta_eg_ev(n, true))
        .collect();
    for i in 1..deltas.len() {
        assert!(
            deltas[i] > deltas[i - 1],
            "BGN not monotone at index {i}: {:.4e} <= {:.4e}",
            deltas[i],
            deltas[i - 1]
        );
    }
}

#[test]
fn bgn_n_type_larger_than_p_type_for_gaas() {
    // Harmon 1994: n-GaAs coefficient (3.23e-8) > p-GaAs coefficient (2.55e-8)
    let n_gaas = harmon1994_gaas_delta_eg_ev(1e18, true);
    let p_gaas = harmon1994_gaas_delta_eg_ev(1e18, false);
    assert!(
        n_gaas > p_gaas,
        "n-GaAs BGN ({:.1} meV) should exceed p-GaAs ({:.1} meV) per Harmon 1994",
        n_gaas * 1000.0,
        p_gaas * 1000.0
    );
}

#[test]
fn gaas_default_uses_harmon1994() {
    let mat = SemiconductorMaterial::gaas();
    assert_eq!(mat.bgn_model, BgnModel::Harmon1994);
}
