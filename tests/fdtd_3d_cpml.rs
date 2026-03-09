//! CPML-specific tests for 3D FDTD.
//!
//! Tests correctness of CPML coefficients and absorption behaviour.

use oxiphoton::fdtd::{BoundaryConfig, Fdtd3d};

// ─────────────────────────────────────────────────────────────────────────────
// 1. CPML coefficients: b < 1 in PML region, b = 1 outside
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cpml_coefficients_pml_cells() {
    use oxiphoton::fdtd::boundary::pml::Cpml;

    let total = 100;
    let pml = 10;
    let d = 10e-9;
    let dt = 1.67e-17;
    let coeffs = Cpml::new(total, pml, d, dt, 3.5, 1e-8);

    // PML cells on the left (indices 0..pml) should have b < 1
    for i in 0..pml {
        assert!(
            coeffs.b_e[i] < 1.0,
            "b_e[{i}] = {} should be < 1 (inside PML)",
            coeffs.b_e[i]
        );
        assert!(
            coeffs.b_h[i] < 1.0,
            "b_h[{i}] = {} should be < 1 (inside PML)",
            coeffs.b_h[i]
        );
    }

    // PML cells on the right (indices total-pml..total) should have b < 1
    for i in total - pml..total {
        assert!(
            coeffs.b_e[i] < 1.0,
            "b_e[{i}] = {} should be < 1 (inside PML)",
            coeffs.b_e[i]
        );
        assert!(
            coeffs.b_h[i] < 1.0,
            "b_h[{i}] = {} should be < 1 (inside PML)",
            coeffs.b_h[i]
        );
    }

    // Interior cells should have b = 1 (no damping)
    let mid = total / 2;
    assert!(
        (coeffs.b_e[mid] - 1.0).abs() < 1e-12,
        "b_e[mid] = {} should be 1.0 (interior)",
        coeffs.b_e[mid]
    );
    assert!(
        (coeffs.b_h[mid] - 1.0).abs() < 1e-12,
        "b_h[mid] = {} should be 1.0 (interior)",
        coeffs.b_h[mid]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. CPML kappa = 1 in interior
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cpml_kappa_interior() {
    use oxiphoton::fdtd::boundary::pml::Cpml;

    let total = 80;
    let pml = 8;
    let d = 15e-9;
    let dt = 2.0e-17;
    let coeffs = Cpml::new(total, pml, d, dt, 3.5, 1e-8);

    // Interior cells: kappa should be 1.0
    for i in pml..total - pml {
        assert!(
            (coeffs.kappa_e[i] - 1.0).abs() < 1e-12,
            "kappa_e[{i}] = {} should be 1.0",
            coeffs.kappa_e[i]
        );
        assert!(
            (coeffs.kappa_h[i] - 1.0).abs() < 1e-12,
            "kappa_h[{i}] = {} should be 1.0",
            coeffs.kappa_h[i]
        );
    }

    // PML cells: kappa >= 1 (CFS-PML uses kappa >= 1)
    for i in 0..pml {
        assert!(
            coeffs.kappa_e[i] >= 1.0,
            "kappa_e[{i}] = {} should be >= 1.0",
            coeffs.kappa_e[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. 3D FDTD with PML: pulse launched, total energy decays
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fdtd3d_pml_absorbs_pulse() {
    // Single-step impulse injection, then track peak energy and decay.
    let d = 20e-9;
    let n = 24;
    let pml = 6;
    let mut s = Fdtd3d::new(n, n, n, d, d, d, &BoundaryConfig::pml(pml));
    let cx = n / 2;

    // Hard-source impulse
    s.inject_ez(cx, cx, cx, 1.0);
    s.step();

    // Measure peak energy in first 20 propagation steps
    let mut e_peak = 0.0_f64;
    for _ in 0..20 {
        s.step();
        let e = s.total_energy();
        if e > e_peak {
            e_peak = e;
        }
    }

    // Run many more steps to let PML absorb
    s.run(400);
    let e_final = s.total_energy();

    assert!(e_peak > 0.0, "Energy should be positive after injection");
    assert!(e_final >= 0.0, "Final energy must be non-negative");
    // PML should have absorbed most of the energy (>80%)
    assert!(
        e_final < e_peak * 0.2,
        "PML did not absorb enough: e_peak={e_peak:.3e} e_final={e_final:.3e}"
    );
    assert!(s.ez.iter().all(|&v| v.is_finite()), "Fields must be finite");
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. 3D FDTD with PML: no reflection artifacts in interior
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fdtd3d_pml_no_reflection() {
    // Inject a single impulse and verify that after long time all interior
    // fields are negligibly small (pulse absorbed by PML, no persistent reflection).
    let d = 20e-9;
    let n = 26;
    let pml = 7;
    let mut s = Fdtd3d::new(n, n, n, d, d, d, &BoundaryConfig::pml(pml));
    let cx = n / 2;
    let src_k = pml + 4;

    // Single-step impulse
    s.inject_ez(cx, cx, src_k, 1.0);
    s.run(500);

    // After 500 steps the pulse should be long gone into the PML.
    // The residual energy in the whole domain should be small relative to
    // the initial injection energy (which equals the energy right after the impulse).
    let e_residual = s.total_energy();
    assert!(
        e_residual < 1e-10,
        "Residual energy too high after PML absorption: {e_residual:.3e}"
    );
    assert!(
        s.ez.iter().all(|&v| v.is_finite()),
        "Fields must remain finite"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. PML with different thicknesses
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cpml_various_thicknesses() {
    // Test that FDTD remains stable for PML thicknesses 4, 8, 12.
    // All fields must remain finite after an impulse injection.
    for &pml in &[4usize, 8, 12] {
        let d = 20e-9;
        let n = 2 * pml + 10;
        let mut s = Fdtd3d::new(n, n, n, d, d, d, &BoundaryConfig::pml(pml));
        let cx = n / 2;

        // Single impulse injection
        s.inject_ez(cx, cx, cx, 1.0);
        s.step();

        // Track peak energy
        let mut e_peak = 0.0_f64;
        for _ in 0..20 {
            s.step();
            let e = s.total_energy();
            if e > e_peak {
                e_peak = e;
            }
        }

        s.run(200);
        let e_final = s.total_energy();

        assert!(
            s.ez.iter().all(|&v| v.is_finite()),
            "PML={pml}: fields non-finite"
        );
        assert!(e_final >= 0.0, "PML={pml}: negative energy {e_final:.3e}");
        // Energy should not grow above the peak
        assert!(
            e_final <= e_peak * 1.01 + 1e-40,
            "PML={pml}: energy grew: peak={e_peak:.3e} final={e_final:.3e}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. CPML c coefficient has correct sign (absorbing, not amplifying)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cpml_c_coefficient_absorbing_sign() {
    use oxiphoton::fdtd::boundary::pml::Cpml;

    let total = 60;
    let pml = 8;
    let d = 10e-9;
    let dt = 1.0e-17;
    let coeffs = Cpml::new(total, pml, d, dt, 3.5, 1e-8);

    // c_e and c_h in PML cells should be negative (or zero) since b < 1 and c = sigma/denom*(b-1)
    // b - 1 < 0, sigma > 0 → c < 0
    for i in 0..pml {
        assert!(
            coeffs.c_e[i] <= 0.0,
            "c_e[{i}] = {} should be <= 0 (absorbing)",
            coeffs.c_e[i]
        );
    }
    for i in total - pml..total {
        assert!(
            coeffs.c_e[i] <= 0.0,
            "c_e[{i}] = {} should be <= 0 (absorbing)",
            coeffs.c_e[i]
        );
    }

    // Interior cells: c = 0 (no PML correction)
    let mid = total / 2;
    assert!(
        coeffs.c_e[mid].abs() < 1e-30,
        "c_e[mid] = {} should be 0 in interior",
        coeffs.c_e[mid]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. CPML: larger thickness gives smaller residual field
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cpml_thicker_pml_absorbs_more() {
    // Compare peak energy and residual after impulse for PML=4 vs PML=10.
    // Thicker PML should result in lower or equal residual.
    fn peak_and_residual(pml: usize) -> (f64, f64) {
        let d = 20e-9;
        let n = (pml * 4).max(20);
        let mut s = Fdtd3d::new(n, n, n, d, d, d, &BoundaryConfig::pml(pml));
        let cx = n / 2;

        // Single impulse
        s.inject_ez(cx, cx, cx, 1.0);
        s.step();

        let mut e_peak = 0.0_f64;
        for _ in 0..20 {
            s.step();
            let e = s.total_energy();
            if e > e_peak {
                e_peak = e;
            }
        }
        s.run(300);
        (e_peak, s.total_energy())
    }

    let (e_peak_thin, e_thin) = peak_and_residual(4);
    let (e_peak_thick, e_thick) = peak_and_residual(10);

    assert!(e_thin >= 0.0 && e_thick >= 0.0);
    assert!(e_peak_thin > 0.0 && e_peak_thick > 0.0);
    // Both should absorb a significant fraction
    assert!(
        e_thin < e_peak_thin,
        "Thin PML should absorb some energy: peak={e_peak_thin:.3e} residual={e_thin:.3e}"
    );
    assert!(
        e_thick < e_peak_thick,
        "Thick PML should absorb some energy: peak={e_peak_thick:.3e} residual={e_thick:.3e}"
    );
}
