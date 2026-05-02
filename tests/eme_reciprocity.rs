//! Regression-guard tests verifying transpose reciprocity S = Sᵀ for the EME
//! S-matrix infrastructure.
//!
//! For lossless, time-reversal-symmetric waveguide layers the scattering matrix
//! satisfies the transpose (not conjugate-transpose) reciprocity condition:
//!
//!   S12[i][j] == S21[j][i]    for all i, j
//!
//! This is distinct from unitarity (S†S = I); it follows from electromagnetic
//! reciprocity (Lorentz reciprocity theorem) when the permittivity tensor is
//! symmetric (ε = εᵀ) and no magneto-optic material is present.
//!
//! ## Scope of current tests
//!
//! Tests 1 and 2 guard the *diagonal* S-matrix returned by `to_s_matrix_full()`
//! for uniform layers (S11 = S22 = 0, S12 = S21 = diag(exp(iβᵢL))).  For such
//! blocks the difference is identically zero at the bit level — the tests would
//! trivially pass regardless of tolerance.  They are valuable as **regression
//! guards**: any future change that introduces off-diagonal mode coupling, changes
//! the S12 ≠ S21 construction symmetry, or alters the block-zero convention will
//! immediately surface here.
//!
//! Tests 3 and 4 exercise the scalar `SMatrix2x2` path and carry the non-trivial
//! part of the reciprocity check: after multiple Redheffer star-product cascades
//! the scalar invariant s12 == s21 must hold to near machine precision.
//!
//! ## Off-diagonal N×N reciprocity (future work)
//!
//! Full off-diagonal reciprocity testing of the N×N S-matrix blocks (S12[i][j]
//! for i ≠ j) requires constructing inter-segment interface S-matrices via mode
//! overlap integrals between *different* waveguide cross-sections.  That
//! infrastructure (an N×N Fresnel-overlap constructor returning coupled S-matrix
//! blocks) does not yet exist in `cascade_smatrices`.  When added, the guard
//! should be extended here to check S12 ≈ S21ᵀ for each off-diagonal element.

use num_complex::Complex64;
use oxiphoton::smatrix::eigenmode::{cascade_smatrices, EigenmodeLayer, EmeSegment, SMatrix2x2};
use oxiphoton::smatrix::interface::interface_smatrix;

const LAMBDA: f64 = 1550e-9;
const OMEGA: f64 = 2.0 * std::f64::consts::PI * 3e8 / LAMBDA;

// ─── Test 1 ───────────────────────────────────────────────────────────────────

/// Guard: `to_s_matrix_full()` for a uniform layer returns S12 == S21.
///
/// For a uniform layer S12 and S21 are built from the same diagonal phase
/// factors, so S12[i][j] == S21[j][i] is bit-exact.  This test becomes
/// non-trivial as soon as mode coupling (off-diagonal elements) or asymmetric
/// S12/S21 construction is introduced.
///
/// Parameters: n_core = 3.476 (Si), n_clad = 1.444 (SiO₂), λ = 1.55 µm,
/// thickness = 2 µm, 2 modes, 200 grid points.
#[test]
fn to_s_matrix_full_satisfies_transpose_reciprocity() {
    let layer = EigenmodeLayer::new(2e-6, 3.476, 1.444, 1.55e-6, 2, 200);
    let (_s11, s12, s21, _s22) = layer.to_s_matrix_full();

    // S12 and S21 must have the same size.
    assert_eq!(s12.len(), s21.len(), "S12 and S21 must have same row count");

    let n = s12.len();
    for i in 0..n {
        assert_eq!(
            s12[i].len(),
            n,
            "S12 row {i} has wrong length (expected {n})"
        );
        assert_eq!(
            s21[i].len(),
            n,
            "S21 row {i} has wrong length (expected {n})"
        );
    }

    // Check S12[i][j] == S21[j][i] within 1e-12 (transpose reciprocity).
    // For a uniform layer both are diagonal with the same phase on the diagonal,
    // so differences are identically zero.  The guard fires if the construction
    // is ever changed to make them asymmetric.
    for i in 0..n {
        for j in 0..n {
            let diff = (s12[i][j] - s21[j][i]).norm();
            assert!(
                diff < 1e-12,
                "Transpose reciprocity violated: |S12[{i}][{j}] - S21[{j}][{i}]| = {diff:.3e}"
            );
        }
    }
}

// ─── Test 2 ───────────────────────────────────────────────────────────────────

/// Guard: `cascade_smatrices()` of two uniform layers preserves S12 == S21ᵀ.
///
/// Because both input blocks are diagonal (S11 = S22 = 0), the Redheffer product
/// reduces to S21_total = S21_B · D1 · S21_A and S12_total = S12_A · D2 · S12_B
/// with diagonal denominator matrices D1 = D2 = I.  The output is again
/// diagonal, so all differences are identically zero.  The test becomes
/// non-trivial once cross-mode terms or a full LU-based inverse are introduced.
#[test]
fn cascade_smatrices_preserves_reciprocity() {
    let layer_a = EigenmodeLayer::new(2.0e-6, 3.476, 1.444, 1.55e-6, 2, 200);
    let layer_b = EigenmodeLayer::new(2.5e-6, 3.476, 1.444, 1.55e-6, 2, 200);

    let (s11_a, s12_a, s21_a, s22_a) = layer_a.to_s_matrix_full();
    let (s11_b, s12_b, s21_b, s22_b) = layer_b.to_s_matrix_full();

    let (_s11, s12, s21, _s22) = cascade_smatrices(
        &s11_a, &s12_a, &s21_a, &s22_a, &s11_b, &s12_b, &s21_b, &s22_b,
    );

    let n = s12.len();
    assert_eq!(n, s21.len(), "Cascaded S12 and S21 must have same size");

    // Tolerance slightly relaxed to 1e-9 to allow for future introduction of
    // floating-point accumulation in the Redheffer product.
    for i in 0..n {
        for j in 0..n {
            let diff = (s12[i][j] - s21[j][i]).norm();
            assert!(
                diff < 1e-9,
                "Transpose reciprocity violated after cascade: \
                 |S12[{i}][{j}] - S21[{j}][{i}]| = {diff:.3e}"
            );
        }
    }
}

// ─── Test 3 ───────────────────────────────────────────────────────────────────

/// `SMatrix2x2::propagation` must satisfy s12 == s21 for any real β and L.
///
/// For a lossless segment: s11 = s22 = 0, s12 = s21 = exp(i·β·L).
/// The constructor sets both from the same `phase` value, so equality is
/// exact.  This test would fail if the sign convention were ever changed to
/// make s12 = exp(+iβL) and s21 = exp(-iβL) (incorrect for reciprocal media).
#[test]
fn smatrix_2x2_propagation_reciprocity() {
    let beta = 10.0_f64;
    let length = std::f64::consts::PI / 20.0_f64; // β·L = π/2  ⟹  phase = i
    let s = SMatrix2x2::propagation(beta, length);

    // For a lossless propagation segment s11 and s22 must vanish.
    assert!(
        s.s11.norm() < 1e-12,
        "s11 should be zero for propagation segment, got {}",
        s.s11
    );
    assert!(
        s.s22.norm() < 1e-12,
        "s22 should be zero for propagation segment, got {}",
        s.s22
    );

    // Transpose reciprocity: s12 == s21.
    let diff = (s.s12 - s.s21).norm();
    assert!(
        diff < 1e-12,
        "s12 != s21 for SMatrix2x2::propagation: |s12 - s21| = {diff:.3e}"
    );

    // Sanity: both equal the expected phase factor exp(i·π/2) = i.
    let expected_phase = Complex64::new(0.0, beta * length).exp();
    assert!(
        (s.s12 - expected_phase).norm() < 1e-12,
        "s12 = {} does not equal expected phase {}",
        s.s12,
        expected_phase
    );
}

// ─── Test 4 ───────────────────────────────────────────────────────────────────

/// Transpose reciprocity is preserved through a chain of Redheffer cascades.
///
/// The chain is:  p1 → i1 → p2 → i2
///   p1 = propagation(β=10.0, L=0.05)
///   i1 = from_overlap(η=0.9)
///   p2 = propagation(β=12.0, L=0.07)
///   i2 = from_overlap(η=0.8)
///
/// Each individual element has s12 == s21.  After the Redheffer star product
/// the Redheffer recursion preserves this when the denominator (1 - S22_A·S11_B)
/// is the same scalar in both the s12 and s21 update, which is the case for the
/// scalar 2×2 formulation.  The tolerance 1e-12 is tight to catch sign-flip
/// regressions in the cascade.
#[test]
fn smatrix_2x2_cascade_preserves_reciprocity() {
    let p1 = SMatrix2x2::propagation(10.0_f64, 0.05_f64);
    let i1 = SMatrix2x2::from_overlap(0.9_f64);
    let p2 = SMatrix2x2::propagation(12.0_f64, 0.07_f64);
    let i2 = SMatrix2x2::from_overlap(0.8_f64);

    // Verify each stage individually satisfies reciprocity before cascading.
    for (label, s) in &[("p1", &p1), ("i1", &i1), ("p2", &p2), ("i2", &i2)] {
        let d = (s.s12 - s.s21).norm();
        assert!(
            d < 1e-12,
            "Stage {label}: |s12 - s21| = {d:.3e} before cascade"
        );
    }

    // Cascade all four stages.
    let r = p1.cascade(&i1).cascade(&p2).cascade(&i2);

    let diff = (r.s12 - r.s21).norm();
    assert!(
        diff < 1e-12,
        "Transpose reciprocity violated after 4-stage cascade: \
         s12 = {}, s21 = {}, |s12 - s21| = {diff:.3e}",
        r.s12,
        r.s21,
    );
}

// ─── Test 5 ───────────────────────────────────────────────────────────────────

/// Off-diagonal N×N reciprocity for a non-trivial two-layer interface stack.
///
/// Two slabs of different widths (500 nm and 700 nm) in the same Si/SiO₂ system.
/// The interface S-matrix is now built via `interface_smatrix` using mode overlaps.
///
/// Assertions:
///   1. S12 and S21 have at least one entry with |entry| > 1e-6 (non-trivial).
///   2. Transpose reciprocity S12 ≈ S21ᵀ holds to 1e-9.
#[test]
fn non_trivial_interface_satisfies_transpose_reciprocity() {
    let seg_a = EmeSegment::new(500e-9, 3.476, 1.444, 500e-9);
    let seg_b = EmeSegment::new(700e-9, 3.476, 1.444, 700e-9);

    let modes_a = seg_a.find_modes(LAMBDA, 2, 200);
    let modes_b = seg_b.find_modes(LAMBDA, 2, 200);

    if modes_a.is_empty() || modes_b.is_empty() {
        // Not enough guided modes for this wavelength — skip gracefully.
        return;
    }

    let (_s11, s12, s21, _s22) = interface_smatrix(&modes_a, &modes_b, OMEGA)
        .expect("interface_smatrix should succeed for different-width slabs");

    // Guard: at least one entry in S12 must be non-trivially nonzero.
    let max_s12: f64 = s12
        .iter()
        .flat_map(|row| row.iter())
        .map(|v| v.norm())
        .fold(0.0_f64, f64::max);
    assert!(
        max_s12 > 1e-6,
        "S12 should have at least one nonzero entry for a real interface, got max |S12| = {max_s12:.3e}"
    );

    let max_s21: f64 = s21
        .iter()
        .flat_map(|row| row.iter())
        .map(|v| v.norm())
        .fold(0.0_f64, f64::max);
    assert!(
        max_s21 > 1e-6,
        "S21 should have at least one nonzero entry for a real interface, got max |S21| = {max_s21:.3e}"
    );

    // Transpose reciprocity: S12[i][j] == S21[j][i] within 1e-9.
    let na = s12.len();
    let nb = s21.len();
    for i in 0..na {
        for j in 0..nb {
            let diff = (s12[i][j] - s21[j][i]).norm();
            assert!(
                diff < 1e-9,
                "Transpose reciprocity S12[{i}][{j}] ≈ S21[{j}][{i}] violated: \
                 S12={}, S21={}, |diff|={diff:.3e}",
                s12[i][j],
                s21[j][i],
            );
        }
    }
}

// ─── Test 6 ───────────────────────────────────────────────────────────────────

/// Guard: cascading two different uniform layers through an interface preserves
/// transpose reciprocity of the aggregate S12 and S21 blocks.
///
/// Stack: layer_a (500 nm) ⋆ interface(a→b) ⋆ layer_b (700 nm).
/// After cascade, the total S12 and S21 must satisfy S12 ≈ S21ᵀ.
#[test]
fn cascade_with_interface_preserves_reciprocity() {
    let layer_a = EigenmodeLayer::new(500e-9, 3.476, 1.444, LAMBDA, 1, 200);
    let layer_b = EigenmodeLayer::new(700e-9, 3.476, 1.444, LAMBDA, 1, 200);

    let seg_a = EmeSegment::new(500e-9, 3.476, 1.444, 500e-9);
    let seg_b = EmeSegment::new(700e-9, 3.476, 1.444, 700e-9);
    let modes_a = seg_a.find_modes(LAMBDA, 1, 200);
    let modes_b = seg_b.find_modes(LAMBDA, 1, 200);

    if modes_a.is_empty() || modes_b.is_empty() {
        return;
    }

    let (s11_a, s12_a, s21_a, s22_a) = layer_a.to_s_matrix_full();
    let (s11_b, s12_b, s21_b, s22_b) = layer_b.to_s_matrix_full();
    let (si11, si12, si21, si22) =
        interface_smatrix(&modes_a, &modes_b, OMEGA).expect("interface_smatrix should succeed");

    // Cascade: layer_a ⋆ interface ⋆ layer_b
    let (c11, c12, c21, c22) =
        cascade_smatrices(&s11_a, &s12_a, &s21_a, &s22_a, &si11, &si12, &si21, &si22);
    let (_s11_tot, s12_tot, s21_tot, _s22_tot) =
        cascade_smatrices(&c11, &c12, &c21, &c22, &s11_b, &s12_b, &s21_b, &s22_b);

    // Transpose reciprocity on the total blocks.
    let n12 = s12_tot.len();
    let n21 = s21_tot.len();
    for i in 0..n12 {
        for j in 0..n21 {
            let diff = (s12_tot[i][j] - s21_tot[j][i]).norm();
            assert!(
                diff < 1e-9,
                "Cascaded S12[{i}][{j}] ≠ S21[{j}][{i}]: diff = {diff:.3e}"
            );
        }
    }
}
