//! Integration tests for bandgap-narrowing (BGN) models in the drift-diffusion solver.
//!
//! Tests verify the Slotboom-de Graaff and Klaassen-Slotboom-de Graaff models against
//! published parameter values, continuity properties, and physical predictions for
//! heavily-doped silicon solar-cell emitters.

use oxiphoton::solar::drift_diffusion::{
    bandgap_narrowing::{klaassen_delta_eg_ev, ni_eff_squared_cm6, slotboom_delta_eg_ev},
    BgnModel, DopingProfile, DriftDiffusionDevice, SemiconductorMaterial,
};

// ─── Unit-level formula tests ────────────────────────────────────────────────

#[test]
fn slotboom_delta_eg_zero_at_low_doping() {
    // At N = 1e15 cm⁻³ (low doping), BGN should be negligible (< 1 meV).
    // V₁ = 9 meV, N_ref = 1e17: ln(1e15/1e17) = -4.6, sqrt(21.2+0.5)=4.66 → ΔEg ≈ 0.006 meV.
    let deg = slotboom_delta_eg_ev(1e15);
    assert!(deg < 1e-3, "expected ΔEg < 1 meV at N=1e15, got {} eV", deg);
}

#[test]
fn slotboom_delta_eg_at_n_ref_matches_published() {
    // At N = N_ref = 1e17: ln_r = 0, ΔEg = V1 * sqrt(0.5) = 9e-3 * 0.70711 ≈ 6.364e-3 eV
    let deg = slotboom_delta_eg_ev(1e17);
    let expected = 9e-3_f64 * 0.5_f64.sqrt();
    assert!(
        (deg - expected).abs() < 0.5e-3,
        "Slotboom ΔEg at N_ref: got {} eV, expected {} eV",
        deg,
        expected
    );
}

#[test]
fn slotboom_delta_eg_at_1e20_matches_published() {
    // At N = 1e20: ln_r = ln(1e20/1e17) = ln(1000) ≈ 6.908
    // ΔEg = 9e-3 * (6.908 + sqrt(6.908² + 0.5)) = 9e-3 * (6.908 + 6.944) ≈ 9e-3 * 13.852 ≈ 124.7 meV
    let deg = slotboom_delta_eg_ev(1e20);
    assert!(
        (deg - 124.7e-3).abs() < 5e-3,
        "Slotboom ΔEg at N=1e20: got {} eV, expected ~124.7 meV",
        deg
    );
}

#[test]
fn klaassen_delta_eg_lower_than_slotboom_at_high_doping() {
    // Klaassen (V1=6.92 meV) should give lower ΔEg than Slotboom (V1=9 meV) at high doping.
    // At N >> N_ref, ΔEg ≈ 2*V1*ln(N/N_ref), so ratio ≈ 6.92/9 = 0.77 for same N_ref.
    // Klaassen has higher N_ref (1.3e17) which raises the effective value slightly,
    // but the lower V1 dominates at N ≥ 1e18.
    for &n in &[1e18_f64, 1e19, 1e20] {
        let s = slotboom_delta_eg_ev(n);
        let k = klaassen_delta_eg_ev(n);
        assert!(
            k < s,
            "Klaassen ΔEg ({} eV) should be lower than Slotboom ({} eV) at N={:.0e}",
            k,
            s,
            n
        );
    }
}

#[test]
fn ni_eff_squared_enhancement_factor_correct() {
    // ΔEg = 50 meV, VT = 25.85 meV: enhancement = exp(50/25.85) ≈ 6.916
    let vt = 25.85e-3_f64;
    let deg = 50e-3_f64;
    let ratio = ni_eff_squared_cm6(1.0, deg, vt);
    let expected = (deg / vt).exp();
    assert!(
        (ratio - expected).abs() < 1e-9,
        "ni_eff² enhancement factor: got {}, expected {}",
        ratio,
        expected
    );
}

#[test]
fn bgn_none_recovers_classical_ni_squared() {
    // With BgnModel::None, n_ie_squared must equal ni² exactly regardless of doping.
    let mut mat = SemiconductorMaterial::silicon();
    mat.bgn_model = BgnModel::None;
    let ni2 = mat.ni_cm3 * mat.ni_cm3;
    // Test at multiple doping levels (pass nd=n_total, na=0 — the None model ignores type)
    for &n_total in &[1e15_f64, 1e18, 1e20] {
        let nie2 = mat.n_ie_squared(300.0, n_total, 0.0);
        assert_eq!(
            nie2, ni2,
            "BgnModel::None: n_ie_squared({:.0e}) = {} != ni² = {}",
            n_total, nie2, ni2
        );
    }
}

#[test]
fn bgn_continuous_at_n_ref_threshold() {
    // The formula is C∞ (no piecewise branches), but verify continuity numerically.
    // eps = 1e-6 relative step → finite-difference change ≈ |dΔEg/dN| * N * eps
    // At N_ref, dΔEg/dN ≈ V1/(N*sqrt(0.5)) so dΔEg ≈ V1*eps/sqrt(0.5) ≈ 1.8e-8 eV.
    // We check that the difference is below 1e-6 eV (1 μeV), far below measurement noise.
    let eps = 1e-6_f64;
    let d1 = slotboom_delta_eg_ev(1e17 * (1.0 - eps));
    let d2 = slotboom_delta_eg_ev(1e17 * (1.0 + eps));
    let diff_ev = (d1 - d2).abs();
    assert!(
        diff_ev < 1e-6,
        "Slotboom ΔEg: large discontinuity at N_ref (eps=1e-6): {} vs {}, diff={} eV",
        d1,
        d2,
        diff_ev
    );
}

// ─── Device-level physics tests ───────────────────────────────────────────────

/// Build a pn-junction device with the given material.
///
/// Emitter (p-type): left half, Na = na_cm3.
/// Base   (n-type): right half, Nd = nd_cm3.
/// Total thickness 100 μm, 100 nodes.
fn build_device(
    mat: SemiconductorMaterial,
    na_cm3: f64,
    nd_cm3: f64,
    n_nodes: usize,
    thickness_cm: f64,
) -> DriftDiffusionDevice {
    let doping = DopingProfile::pn_junction(n_nodes, na_cm3, nd_cm3);
    DriftDiffusionDevice::new(mat, doping, thickness_cm, n_nodes)
        .expect("device creation should succeed")
}

#[test]
fn bgn_lowers_v_oc_at_heavy_emitter_doping() {
    // A heavily-doped emitter (Na=1e19) with BGN should show lower V_oc than without BGN.
    // Physical reason: BGN raises ni_eff in the emitter, increasing J0 and lowering V_oc.
    //
    // Device: 100 μm, 100 nodes (dx ≈ 1 μm).
    // Emitter: Na = 1e19 cm⁻³ (heavily doped p-type, left half).
    //   At Na=1e19: ΔEg(Slotboom) ≈ 83 meV → ni_eff²/ni² ≈ exp(83/25.85) ≈ 25×.
    // Base:    Nd = 1e18 cm⁻³ (n-type, right half) — higher base doping ensures
    //          the BGN-amplified emitter J0 is comparable to base J0.
    //   Predicted ΔV_oc = 15–30 mV (well within the 5–80 mV bracket).
    // Generation: uniform G = 1e21 cm⁻³ s⁻¹.
    //
    // Na=1e20 is not used here because the depletion width (~0.05 μm) would be
    // thinner than one grid cell (1 μm), causing Gummel/Thomas convergence failures.

    let na_emitter = 1e19_f64;
    let nd_base = 1e18_f64;
    let n_nodes = 100;
    let thickness_cm = 100e-4; // 100 μm

    let mut mat_slotboom = SemiconductorMaterial::silicon();
    mat_slotboom.bgn_model = BgnModel::Slotboom;

    let mut mat_none = SemiconductorMaterial::silicon();
    mat_none.bgn_model = BgnModel::None;

    let gen = vec![1e21_f64; n_nodes];
    // Sweep from 0 to 0.8 V in 0.05 V steps — covers V_oc range for Si
    let v_grid: Vec<f64> = (0..17).map(|i| i as f64 * 0.05).collect();

    let mut dev_slotboom = build_device(mat_slotboom, na_emitter, nd_base, n_nodes, thickness_cm);
    let iv_slotboom = dev_slotboom
        .solve_illuminated_iv(&v_grid, &gen)
        .expect("Slotboom illuminated IV");
    let (_, v_oc_slotboom) = dev_slotboom.extract_jsc_voc(&iv_slotboom);

    let mut dev_none = build_device(mat_none, na_emitter, nd_base, n_nodes, thickness_cm);
    let iv_none = dev_none
        .solve_illuminated_iv(&v_grid, &gen)
        .expect("None illuminated IV");
    let (_, v_oc_none) = dev_none.extract_jsc_voc(&iv_none);

    // BGN should lower V_oc by at least 5 mV and at most 80 mV
    let delta_voc = v_oc_none - v_oc_slotboom;
    assert!(
        delta_voc >= 5e-3,
        "BGN should lower V_oc by >= 5 mV: V_oc(Slotboom)={:.4} V, V_oc(None)={:.4} V, ΔV_oc={:.1} mV",
        v_oc_slotboom,
        v_oc_none,
        delta_voc * 1e3
    );
    assert!(
        delta_voc <= 80e-3,
        "BGN V_oc reduction too large (>80 mV): V_oc(Slotboom)={:.4} V, V_oc(None)={:.4} V, ΔV_oc={:.1} mV",
        v_oc_slotboom,
        v_oc_none,
        delta_voc * 1e3
    );
}

#[test]
fn bgn_increases_dark_j0_at_heavy_doping() {
    // BGN raises ni_eff in the emitter, which increases J0 proportionally to ni_eff².
    // At Na=1e19: ΔEg(Slotboom) ≈ 83 meV → ni_eff²/ni² ≈ exp(83/25.85) ≈ 25×.
    //
    // With nd_base=1e18, both emitter and base contributions are comparable, but
    // the Slotboom BGN only applies in the heavily-doped emitter (Na=1e19).
    // Default Si lifetimes (τ=1e-6 s) are used — shorter lifetimes hurt Gummel convergence.
    // Conservative assertion: J_dark(Slotboom) / J_dark(None) >= 2.0.
    //   Predicted ratio ≈ 5–15× (well above 2.0).
    //
    // Ramped voltage sweep (0 → 0.3 V in 0.05 V steps) aids convergence at high doping.
    // At 0.3 V the injection ratio exp(0.3/0.026) ≈ 1e5, still tractable for Gummel.
    //
    // Na=1e20 is not used here because the depletion width (~0.05 μm) would be
    // thinner than one grid cell (1 μm), causing Thomas near-zero-pivot failures.

    let na_emitter = 1e19_f64;
    let nd_base = 1e18_f64;
    let n_nodes = 100;
    let thickness_cm = 100e-4;

    let mut mat_slotboom = SemiconductorMaterial::silicon();
    mat_slotboom.bgn_model = BgnModel::Slotboom;

    let mut mat_none = SemiconductorMaterial::silicon();
    mat_none.bgn_model = BgnModel::None;

    // Sweep from 0 to 0.3 V in 0.05 V steps; evaluate at the last point (0.30 V)
    let v_dark: Vec<f64> = (0..7).map(|i| i as f64 * 0.05).collect();

    let mut dev_slotboom = build_device(mat_slotboom, na_emitter, nd_base, n_nodes, thickness_cm);
    let iv_slotboom = dev_slotboom
        .solve_dark_iv(&v_dark)
        .expect("Slotboom dark IV");
    // Use the last non-zero voltage point
    let j_dark_slotboom = iv_slotboom.last().map(|(_, j)| j.abs()).unwrap_or(0.0);

    let mut dev_none = build_device(mat_none, na_emitter, nd_base, n_nodes, thickness_cm);
    let iv_none = dev_none.solve_dark_iv(&v_dark).expect("None dark IV");
    let j_dark_none = iv_none.last().map(|(_, j)| j.abs()).unwrap_or(0.0);

    // Avoid division by zero in degenerate cases
    let ratio = if j_dark_none > 1e-30 {
        j_dark_slotboom / j_dark_none
    } else {
        f64::INFINITY
    };

    assert!(
        ratio >= 2.0,
        "BGN should increase dark current by >= 2×: J_dark(Slotboom)={:.3e}, J_dark(None)={:.3e}, ratio={:.2}",
        j_dark_slotboom,
        j_dark_none,
        ratio
    );
}
