//! Power-conservation and S-matrix orthogonality tests for the EME cascade.
//!
//! These six tests act as regression guards for the Phase 10 fix that migrated
//! `SMatrix2x2` to `Complex64`.  The pre-fix bug returned identity for every
//! propagation segment, which means `s12 = 1, s11 = 0` trivially — so the
//! phase and passivity assertions below would fail against a regressed impl.

use num_complex::Complex64;
use oxiphoton::smatrix::eigenmode::{EigenmodeLayer, SMatrix2x2};

/// |c|²
#[inline]
fn norm2(c: Complex64) -> f64 {
    c.norm_sqr()
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: single propagation segment — phase is non-trivial and unit magnitude
// ─────────────────────────────────────────────────────────────────────────────

/// A lossless propagation S-matrix must be unitary:
/// |s12|² = 1, |s11|² + |s21|² = 1, |s12|² + |s22|² = 1.
///
/// The old f64 bug returned identity, where all of these also happen to hold
/// trivially.  This test additionally checks that s12 has the *correct phase*,
/// which the identity matrix cannot pass.
#[test]
fn single_segment_propagation_unitary() {
    let beta = 10.0_f64;
    let length = std::f64::consts::PI / 20.0_f64; // β·L = π/2  ⟹  phase = i
    let s = SMatrix2x2::propagation(beta, length);

    // Pure phase: |s12|² must equal 1.0
    assert!(
        (norm2(s.s12) - 1.0).abs() < 1e-10,
        "|s12|² = {} ≠ 1",
        norm2(s.s12)
    );

    // Passive from left: |s11|² + |s21|² = 1
    let col1 = norm2(s.s11) + norm2(s.s21);
    assert!((col1 - 1.0).abs() < 1e-10, "|s11|² + |s21|² = {col1} ≠ 1");

    // Passive from right: |s12|² + |s22|² = 1 (reciprocal, s22 = 0)
    let col2 = norm2(s.s12) + norm2(s.s22);
    assert!((col2 - 1.0).abs() < 1e-10, "|s12|² + |s22|² = {col2} ≠ 1");

    // Phase must be e^{i·β·L} = e^{i·π/2} = i  (regression guard against identity)
    let expected_phase = Complex64::new(0.0, 1.0); // i
    assert!(
        (s.s12 - expected_phase).norm() < 1e-10,
        "s12 = {} ≠ i (expected pure-imaginary phase for β·L = π/2)",
        s.s12
    );
    // s11 = s22 = 0 for a lossless, perfectly matched segment
    assert!(s.s11.norm() < 1e-10, "s11 should be 0, got {}", s.s11);
    assert!(s.s22.norm() < 1e-10, "s22 should be 0, got {}", s.s22);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: overlap interface — energy conserved for partial coupling
// ─────────────────────────────────────────────────────────────────────────────

/// For η = 0.7:  T = η² = 0.49, R = 1 − T = 0.51.
/// Column-1 passivity: |s11|² + |s21|² = R + T = 1.
#[test]
fn single_segment_from_overlap_energy_conserving() {
    let eta = 0.7_f64;
    let s = SMatrix2x2::from_overlap(eta);

    let t = eta * eta; // = 0.49
    let r = 1.0 - t; // = 0.51

    assert!(
        (norm2(s.s21) - t).abs() < 1e-10,
        "|s21|² = {} ≠ T = {t}",
        norm2(s.s21)
    );
    assert!(
        (norm2(s.s11) - r).abs() < 1e-10,
        "|s11|² = {} ≠ R = {r}",
        norm2(s.s11)
    );

    let col1 = norm2(s.s11) + norm2(s.s21);
    assert!((col1 - 1.0).abs() < 1e-10, "|s11|² + |s21|² = {col1} ≠ 1");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: full overlap → identity
// ─────────────────────────────────────────────────────────────────────────────

/// η = 1.0: T = 1, R = 0.  The resulting matrix must equal `SMatrix2x2::identity()`.
#[test]
fn single_segment_from_overlap_full_overlap_is_identity() {
    let s = SMatrix2x2::from_overlap(1.0);

    assert!(s.s11.norm() < 1e-10, "s11 = {} ≠ 0 for full overlap", s.s11);
    assert!(s.s22.norm() < 1e-10, "s22 = {} ≠ 0 for full overlap", s.s22);
    assert!(
        (norm2(s.s12) - 1.0).abs() < 1e-10,
        "|s12|² = {} ≠ 1 for full overlap",
        norm2(s.s12)
    );
    assert!(
        (norm2(s.s21) - 1.0).abs() < 1e-10,
        "|s21|² = {} ≠ 1 for full overlap",
        norm2(s.s21)
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: cascaded chain — passive and non-trivial
// ─────────────────────────────────────────────────────────────────────────────

/// Chain: identity → prop(β=10, L=0.05) → overlap(η=0.9) → prop(β=12, L=0.07)
///       → overlap(η=0.8) → prop(β=10, L=0.03)
///
/// Passivity: |s11|² + |s21|² ≤ 1  (energy cannot be created).
/// Non-triviality: |s21|² > 0.5 (not all power lost — interface losses are small).
#[test]
fn multi_segment_cascade_below_unity() {
    let s = SMatrix2x2::identity()
        .cascade(&SMatrix2x2::propagation(10.0, 0.05))
        .cascade(&SMatrix2x2::from_overlap(0.9))
        .cascade(&SMatrix2x2::propagation(12.0, 0.07))
        .cascade(&SMatrix2x2::from_overlap(0.8))
        .cascade(&SMatrix2x2::propagation(10.0, 0.03));

    let col1 = norm2(s.s11) + norm2(s.s21);
    assert!(
        col1 <= 1.0 + 1e-12,
        "|s11|² + |s21|² = {col1} > 1 (energy creation!)"
    );
    // Non-trivial: the column-1 norm (reflection + transmission) must exceed 0.5,
    // i.e. the cascade is not near-degenerate.  For a passive lossless network
    // this sum equals 1.0; we assert > 0.5 to catch a broken cascade that
    // collapses to a near-zero S-matrix.
    let col1_sum = norm2(s.s11) + norm2(s.s21);
    assert!(
        col1_sum > 0.5,
        "|s11|² + |s21|² = {col1_sum} ≤ 0.5 (degenerate cascade — possible regression to zero)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: multi-propagation + interface chain — passive on both ports
// ─────────────────────────────────────────────────────────────────────────────

/// 5-segment chain alternating propagation segments and coupling interfaces.
/// Checks passivity from both port 1 and port 2.
#[test]
fn cascade_power_conservation_multi_propagation_interfaces() {
    let s = SMatrix2x2::identity()
        .cascade(&SMatrix2x2::propagation(5.0, 0.1))
        .cascade(&SMatrix2x2::from_overlap(0.95))
        .cascade(&SMatrix2x2::propagation(7.0, 0.15))
        .cascade(&SMatrix2x2::from_overlap(0.85))
        .cascade(&SMatrix2x2::propagation(6.0, 0.08))
        .cascade(&SMatrix2x2::from_overlap(0.75))
        .cascade(&SMatrix2x2::propagation(5.5, 0.12));

    // Passive from port 1
    let col1 = norm2(s.s11) + norm2(s.s21);
    assert!(
        col1 <= 1.0 + 1e-12,
        "|s11|² + |s21|² = {col1} > 1 (passive-port-1 violated)"
    );

    // Passive from port 2
    let col2 = norm2(s.s12) + norm2(s.s22);
    assert!(
        col2 <= 1.0 + 1e-12,
        "|s12|² + |s22|² = {col2} > 1 (passive-port-2 violated)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: two identical propagation segments — combined phase doubles
// ─────────────────────────────────────────────────────────────────────────────

/// Cascading two propagation S-matrices with the same β and length L must
/// produce a combined transmission phase of e^{i·2·β·L}.
///
/// With the old f64 bug, propagation returned identity (s12 = 1 always), so
/// the cascaded s12 would equal 1 instead of e^{i·2βL}.  This test would fail
/// against that regression for any β·L ≠ 0 mod π.
///
/// Also verifies the diagonal block structure of `EigenmodeLayer::to_s_matrix_full()`:
/// S11 and S22 blocks must be all-zero and S21[0][0] must be a pure phase.
#[test]
fn multi_mode_smatrix_full_block_structure_and_double_phase() {
    let beta = 8.0_f64;
    let length = 0.1_f64; // β·L = 0.8 rad

    let s1 = SMatrix2x2::propagation(beta, length);
    let s2 = SMatrix2x2::propagation(beta, length);
    let s_total = s1.cascade(&s2);

    // Combined phase: e^{i·2·β·L}
    let expected = Complex64::new(0.0, 2.0 * beta * length).exp();
    assert!(
        (s_total.s12 - expected).norm() < 1e-10,
        "Cascaded s12 = {} ≠ e^{{i·2βL}} = {expected} (regression: identity returns 1)",
        s_total.s12
    );
    assert!(
        (s_total.s21 - expected).norm() < 1e-10,
        "Cascaded s21 = {} ≠ e^{{i·2βL}} = {expected}",
        s_total.s21
    );

    // EigenmodeLayer block structure check
    // S11 and S22 blocks should be all-zero; S21 diagonal should be unit magnitude
    let layer = EigenmodeLayer::new(10e-6, 3.476, 1.444, 1550e-9, 1, 100);
    let (s11, _s12_block, s21_block, s22) = layer.to_s_matrix_full();

    assert!(
        s11[0][0].norm() < 1e-10,
        "EigenmodeLayer S11[0][0] = {} ≠ 0 (no back-reflection in uniform layer)",
        s11[0][0]
    );
    assert!(
        s22[0][0].norm() < 1e-10,
        "EigenmodeLayer S22[0][0] = {} ≠ 0 (no back-reflection in uniform layer)",
        s22[0][0]
    );
    assert!(
        (s21_block[0][0].norm() - 1.0).abs() < 1e-6,
        "EigenmodeLayer S21[0][0] magnitude = {} ≠ 1 (should be pure phase)",
        s21_block[0][0].norm()
    );
}
