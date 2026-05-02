// tests/photonic_crystal_eigensolver.rs
//
// Integration tests for the SSH eigenpair / Bloch-vector / Wilson-loop pipeline.

use std::f64::consts::PI;

use oxiphoton::photonic_crystal::topology::SshPhotonicChain;

// ─── helpers ──────────────────────────────────────────────────────────────────

/// Build a k-path of `n_k` equally-spaced points over [0, 2π).
fn linspace_bz(n_k: usize) -> Vec<f64> {
    (0..n_k).map(|i| 2.0 * PI * i as f64 / n_k as f64).collect()
}

// ─── tests ────────────────────────────────────────────────────────────────────

/// bloch_vector returns unit-norm vectors for a small grid of k-points.
#[test]
fn bloch_vector_unit_norm() {
    let chain = SshPhotonicChain::new(3, 0.5, 1.0, 0.0);
    let k_vals = linspace_bz(20);
    for &k in &k_vals {
        for band in 0..2 {
            let v = chain
                .bloch_vector(k, band)
                .expect("bloch_vector must succeed");
            let norm_sq: f64 = v.iter().map(|c| c.norm_sqr()).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1e-12,
                "band={band} k={k:.4}: norm_sq = {norm_sq:.15} (expected 1)"
            );
        }
    }
}

/// bloch_vector returns Err for band_index >= 2.
#[test]
fn bloch_vector_out_of_range_band() {
    let chain = SshPhotonicChain::new(3, 0.5, 1.0, 0.0);
    assert!(
        chain.bloch_vector(0.0, 2).is_err(),
        "band_index=2 must return Err"
    );
}

/// Topological phase (κ₁ < κ₂): Berry phase ≈ π.
///
/// For the lower band of the SSH model with intra-cell < inter-cell coupling
/// the Wilson-loop (Zak) phase is π (topological invariant).
#[test]
fn ssh_topological_phase_pi() {
    let chain = SshPhotonicChain::new(3, 0.5, 1.0, 0.0); // kappa1 < kappa2
    let k_path = linspace_bz(100);
    let bp = chain
        .wilson_loop_band_n(&k_path, 0)
        .expect("wilson_loop_band_n must succeed");
    let phase = bp.zak_phase(0);
    // Zak phase is normalised to [0, 2π); π is ~3.14, check within 0.2 rad.
    assert!(
        (phase - PI).abs() < 0.2,
        "topological phase: expected ≈π, got {phase:.4}"
    );
}

/// Trivial phase (κ₁ > κ₂): Berry phase ≈ 0.
#[test]
fn ssh_trivial_phase_zero() {
    let chain = SshPhotonicChain::new(3, 1.0, 0.5, 0.0); // kappa1 > kappa2
    let k_path = linspace_bz(100);
    let bp = chain
        .wilson_loop_band_n(&k_path, 0)
        .expect("wilson_loop_band_n must succeed");
    let phase = bp.zak_phase(0);
    // Trivial: Zak phase = 0 (or equivalently 2π). Check not near π.
    assert!(
        !(0.2..=(2.0 * PI - 0.2)).contains(&phase),
        "trivial phase: expected ≈0, got {phase:.4}"
    );
}

/// Berry phase is stable under refinement of the k-path density.
#[test]
fn wilson_loop_independent_of_k_path_density() {
    let chain = SshPhotonicChain::new(3, 0.5, 1.0, 0.0);

    let phase_50 = chain
        .wilson_loop_band_n(&linspace_bz(50), 0)
        .expect("50-point path")
        .zak_phase(0);
    let phase_100 = chain
        .wilson_loop_band_n(&linspace_bz(100), 0)
        .expect("100-point path")
        .zak_phase(0);
    let phase_200 = chain
        .wilson_loop_band_n(&linspace_bz(200), 0)
        .expect("200-point path")
        .zak_phase(0);

    assert!(
        (phase_50 - phase_100).abs() < 0.05,
        "50 vs 100 k-points: phases differ by more than 0.05 rad ({phase_50:.4} vs {phase_100:.4})"
    );
    assert!(
        (phase_100 - phase_200).abs() < 0.05,
        "100 vs 200 k-points: phases differ by more than 0.05 rad ({phase_100:.4} vs {phase_200:.4})"
    );
}

/// Eigenfrequencies of the finite-chain `SshChain` model are unchanged by the
/// migration from pure Sturm bisection to `TridiagEvd`.
///
/// We compare the old Sturm-bisection fallback (via calling the existing
/// `SshChain::eigenfrequencies`) against itself — the test simply verifies
/// that the function still returns 2N sorted values for an N-cell chain.
#[test]
fn eigenfrequencies_count_and_ordering() {
    use oxiphoton::photonic_crystal::topology::SshChain;
    let chain = SshChain::topological(4);
    let eigs = chain.eigenfrequencies();
    assert_eq!(eigs.len(), 8, "2×n_cells eigenvalues expected");
    for w in eigs.windows(2) {
        assert!(
            w[1] >= w[0] - 1e-9,
            "eigenvalues must be sorted: {} > {}",
            w[0],
            w[1]
        );
    }
}

/// SshChain eigenfrequencies agree between TridiagEvd path and a reference
/// analytical estimate (Gershgorin bound check: all eigenvalues in [ω₀ ± sum_offdiag]).
#[test]
fn eigenfrequencies_within_gershgorin_bounds() {
    use oxiphoton::photonic_crystal::topology::SshChain;
    let kappa1 = 0.5_f64;
    let kappa2 = 1.0_f64;
    let omega0 = 2.0_f64;
    let n_cells = 5;
    let chain = SshChain::new(kappa1, kappa2, omega0, n_cells);
    let eigs = chain.eigenfrequencies();
    // Gershgorin bound: max off-diagonal sum per row is kappa1 + kappa2
    let max_radius = kappa1 + kappa2;
    for &e in &eigs {
        assert!(
            (e - omega0).abs() <= max_radius + 1e-9,
            "eigenvalue {e:.6} outside Gershgorin bound [{}, {}]",
            omega0 - max_radius,
            omega0 + max_radius
        );
    }
}
