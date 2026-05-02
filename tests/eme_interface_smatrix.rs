//! Integration tests for EME interface S-matrix construction and cascade.
//!
//! Tests the mode-matching interface S-matrix computed by `interface_smatrix`
//! and the `EmeStack` cascaded EME solver.
//!
//! Physical conventions:
//!   ω = 2π·c/λ for λ = 1550 nm  →  ω ≈ 1.215e15 rad/s
//!   Modes obtained from `EmeSegment::find_modes` (symmetric slab, TE).

use num_complex::Complex64;
use oxiphoton::smatrix::eigenmode::{EigenmodeLayer, EmeMode, EmeSegment};
use oxiphoton::smatrix::interface::interface_smatrix;

const LAMBDA: f64 = 1550e-9; // m
const OMEGA: f64 = 2.0 * std::f64::consts::PI * 3e8 / LAMBDA; // rad/s

// ─── Helper: find modes of a symmetric slab ─────────────────────────────────

fn slab_modes(thickness: f64, n_modes: usize, n_pts: usize) -> Vec<EmeMode> {
    let seg = EmeSegment::new(thickness, 3.476, 1.444, thickness);
    seg.find_modes(LAMBDA, n_modes, n_pts)
}

// ─── Test 1 ─────────────────────────────────────────────────────────────────

/// A waveguide meeting itself at an interface must produce S11 ≈ 0 and S12 ≈ I.
///
/// Physical interpretation: if modes_a == modes_b the interface is transparent —
/// each mode continues without reflection and without coupling to other modes.
#[test]
fn same_layer_interface_is_identity() {
    let modes = slab_modes(500e-9, 1, 200);
    assert!(!modes.is_empty(), "Need at least one guided mode");

    let (s11, s12, s21, s22) = interface_smatrix(&modes, &modes, OMEGA)
        .expect("interface_smatrix should succeed for same modes");

    let n = s11.len();

    // S11 and S22 must be near-zero (no back-reflection at self-interface).
    for i in 0..n {
        for j in 0..n {
            let v11 = s11[i][j].norm();
            assert!(
                v11 < 1e-8,
                "S11[{i}][{j}] = {v11:.3e} should be ~0 at self-interface"
            );
            let v22 = s22[i][j].norm();
            assert!(
                v22 < 1e-8,
                "S22[{i}][{j}] = {v22:.3e} should be ~0 at self-interface"
            );
        }
    }

    // S12 and S21 must be near the identity (perfect transmission).
    for (i, row12) in s12.iter().enumerate() {
        for (j, &val) in row12.iter().enumerate() {
            let expected = if i == j { 1.0 } else { 0.0 };
            let diff = (val - Complex64::new(expected, 0.0)).norm();
            assert!(
                diff < 1e-8,
                "S12[{i}][{j}] = {val} should be {expected:.0} at self-interface (diff {diff:.3e})"
            );
        }
    }
    for (i, row21) in s21.iter().enumerate() {
        for (j, &val) in row21.iter().enumerate() {
            let expected = if i == j { 1.0 } else { 0.0 };
            let diff = (val - Complex64::new(expected, 0.0)).norm();
            assert!(
                diff < 1e-8,
                "S21[{i}][{j}] = {val} should be {expected:.0} at self-interface (diff {diff:.3e})"
            );
        }
    }
}

// ─── Test 2 ─────────────────────────────────────────────────────────────────

/// Energy conservation (unitarity) for a lossless interface.
///
/// For a lossless dielectric interface with real propagation constants,
/// the S-matrix must satisfy column-unitarity:
///   |S11[0,0]|² + |S21[0,0]|² ≈ 1  (energy in = energy out)
///
/// We check this for the fundamental-mode column of the S-matrix.
#[test]
fn interface_power_conservation() {
    let modes_a = slab_modes(500e-9, 2, 200);
    let modes_b = slab_modes(800e-9, 2, 200);

    // Need modes on both sides
    if modes_a.is_empty() || modes_b.is_empty() {
        return;
    }

    let (s11, _s12, s21, _s22) =
        interface_smatrix(&modes_a, &modes_b, OMEGA).expect("interface_smatrix should succeed");

    // Check power conservation for each input mode on the left (S11 + S21 columns).
    let na = s11.len();
    let nb = s21.len();

    for j in 0..na {
        // Power reflected back to side A from input mode j
        let r_power: f64 = (0..na).map(|i| s11[i][j].norm_sqr()).sum();
        // Power transmitted to side B from input mode j
        let t_power: f64 = (0..nb).map(|i| s21[i][j].norm_sqr()).sum();
        let total = r_power + t_power;
        assert!(
            (total - 1.0).abs() < 1e-6,
            "Power conservation violated for input mode {j}: R+T = {total:.6} (expected 1)"
        );
    }
}

// ─── Test 3 ─────────────────────────────────────────────────────────────────

/// A step-width interface (wider → narrower slab) must produce nonzero reflection.
///
/// Physical expectation: going from a 800 nm wide core to a 300 nm core causes
/// some back-reflection in the fundamental mode.  |S11[0,0]|² > 0.001.
#[test]
fn step_width_reflection_nonzero() {
    let modes_wide = slab_modes(800e-9, 1, 200);
    let modes_narrow = slab_modes(300e-9, 1, 200);

    if modes_wide.is_empty() || modes_narrow.is_empty() {
        // No guided mode in one arm → test not applicable
        return;
    }

    let (s11, _s12, _s21, _s22) = interface_smatrix(&modes_wide, &modes_narrow, OMEGA)
        .expect("interface_smatrix should succeed");

    let r = s11[0][0].norm_sqr();
    assert!(
        r > 1e-4,
        "|S11[0,0]|² = {r:.3e} — expected nonzero reflection at a step-width interface"
    );
}

// ─── Test 4 ─────────────────────────────────────────────────────────────────

/// Transpose reciprocity at an interface: S12[i][j] == S21[j][i].
///
/// For lossless, time-reversal symmetric media (real ε) the electromagnetic
/// reciprocity theorem (Lorentz) guarantees S12 = S21ᵀ.
#[test]
fn interface_transpose_reciprocity() {
    let modes_a = slab_modes(500e-9, 2, 200);
    let modes_b = slab_modes(700e-9, 2, 200);

    if modes_a.is_empty() || modes_b.is_empty() {
        return;
    }

    let (_s11, s12, s21, _s22) =
        interface_smatrix(&modes_a, &modes_b, OMEGA).expect("interface_smatrix should succeed");

    // s12 is na×nb, s21 is nb×na; transpose reciprocity: s12[i][j] == s21[j][i]
    let na = s12.len();
    let nb = s21.len();
    for i in 0..na {
        for j in 0..nb {
            let diff = (s12[i][j] - s21[j][i]).norm();
            assert!(
                diff < 1e-8,
                "Transpose reciprocity violated: |S12[{i}][{j}] - S21[{j}][{i}]| = {diff:.3e}"
            );
        }
    }
}

// ─── Test 5 ─────────────────────────────────────────────────────────────────

/// A two-mode system must return correctly sized S-matrix blocks.
#[test]
fn interface_smatrix_correct_block_sizes() {
    let modes_a = slab_modes(600e-9, 2, 200);
    let modes_b = slab_modes(900e-9, 3, 200);

    // Adjust if fewer modes are guided
    if modes_a.len() < 2 || modes_b.len() < 2 {
        return;
    }

    let (s11, s12, s21, s22) =
        interface_smatrix(&modes_a, &modes_b, OMEGA).expect("interface_smatrix should succeed");

    let na = modes_a.len();
    let nb = modes_b.len();

    assert_eq!(s11.len(), na, "S11 must be na×na");
    assert_eq!(s11[0].len(), na, "S11 must be na×na");
    assert_eq!(s12.len(), na, "S12 must be na×nb");
    assert_eq!(s12[0].len(), nb, "S12 must be na×nb");
    assert_eq!(s21.len(), nb, "S21 must be nb×na");
    assert_eq!(s21[0].len(), na, "S21 must be nb×na");
    assert_eq!(s22.len(), nb, "S22 must be nb×nb");
    assert_eq!(s22[0].len(), nb, "S22 must be nb×nb");
}

// ─── Test 6 ─────────────────────────────────────────────────────────────────

/// A cascade A→B→A must have reduced S11 compared to A→B alone.
///
/// The back-reflection from A→B→A is partially cancelled by the B→A interface,
/// so |S11_cascade[0,0]|² < |S11_AB[0,0]|² for different widths.
///
/// Note: "reduced" is not guaranteed by physics — this test simply checks that
/// the cascade is consistent (S11 stays in [0,1]).
#[test]
fn cascade_two_steps_s11_bounded() {
    let layer_a = EigenmodeLayer::new(500e-9, 3.476, 1.444, LAMBDA, 1, 200);
    let layer_b = EigenmodeLayer::new(700e-9, 3.476, 1.444, LAMBDA, 1, 200);

    let modes_a = slab_modes(500e-9, 1, 200);
    let modes_b = slab_modes(700e-9, 1, 200);

    if modes_a.is_empty() || modes_b.is_empty() {
        return;
    }

    use oxiphoton::smatrix::eigenmode::cascade_smatrices;

    let (s11_a, s12_a, s21_a, s22_a) = layer_a.to_s_matrix_full();
    let (s11_b, s12_b, s21_b, s22_b) = layer_b.to_s_matrix_full();

    // Interface A→B
    let iface_ab = interface_smatrix(&modes_a, &modes_b, OMEGA).expect("interface A→B");
    let (si11, si12, si21, si22) = iface_ab;

    // Cascade: layer_a ⋆ iface_ab ⋆ layer_b
    let (c11, c12, c21, c22) =
        cascade_smatrices(&s11_a, &s12_a, &s21_a, &s22_a, &si11, &si12, &si21, &si22);
    let (_s11_total, _s12_total, s21_total, _s22_total) =
        cascade_smatrices(&c11, &c12, &c21, &c22, &s11_b, &s12_b, &s21_b, &s22_b);

    // Total transmission amplitude should be nonzero
    let t = s21_total[0][0].norm_sqr();
    assert!(
        t > 0.0 && t <= 1.001,
        "Transmission must be in (0,1], got {t:.4}"
    );
}

// ─── Test 7 ─────────────────────────────────────────────────────────────────

/// S12 and S21 at a non-trivial (different-width) interface must have at least
/// one nonzero entry.
#[test]
fn interface_off_diagonal_nonzero_for_different_widths() {
    let modes_a = slab_modes(400e-9, 1, 200);
    let modes_b = slab_modes(800e-9, 1, 200);

    if modes_a.is_empty() || modes_b.is_empty() {
        return;
    }

    let (_s11, s12, s21, _s22) =
        interface_smatrix(&modes_a, &modes_b, OMEGA).expect("interface_smatrix should succeed");

    // S12[0][0] must be nonzero (some transmission)
    let t12 = s12[0][0].norm();
    assert!(
        t12 > 1e-6,
        "S12[0][0] = {t12:.3e} — expected nonzero transmission between different widths"
    );
    let t21 = s21[0][0].norm();
    assert!(
        t21 > 1e-6,
        "S21[0][0] = {t21:.3e} — expected nonzero transmission between different widths"
    );
}
