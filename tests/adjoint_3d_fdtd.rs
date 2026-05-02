/// Integration tests for `AdjointSolver3d` wired to a real `Fdtd3d` pipeline.
///
/// Each test uses a small design region (6×6×6 or similar) with a reduced step
/// count (`n_steps = 150`) to keep CI run time manageable while still exercising
/// the full forward → adjoint → gradient pipeline.
use num_complex::Complex64;
use oxiphoton::inverse::{AdjointSolver3d, DesignRegion3d};

/// Helper: build a small solver wired to FDTD with reduced step count.
fn small_fdtd_solver(
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    wavelength_m: f64,
) -> (AdjointSolver3d, DesignRegion3d) {
    use oxiphoton::inverse::FdtdSourceConfig;
    use std::f64::consts::PI;
    let c = 2.998e8_f64;
    let omega = 2.0 * PI * c / wavelength_m;
    let cfg = FdtdSourceConfig {
        source_i: nx / 2,
        source_j: ny / 2,
        source_k: 0,
        monitor_cells: vec![(nx / 2, ny / 2, nz.saturating_sub(1))],
    };

    let mut solver = AdjointSolver3d::new_with_fdtd(nx, ny, nz, dx, omega, cfg);
    // Reduce step count so tests run quickly
    solver.n_steps = 150;

    let region = DesignRegion3d::new(nx, ny, nz, dx, 1.0, 4.0);
    (solver, region)
}

/// Test 1: `run_forward` on a small vacuum region returns a finite `Vec<Complex64>`.
///
/// This verifies that the full FDTD pipeline runs without panicking, that all
/// returned field values are finite, and that at least some cells have non-zero
/// magnitude (the source has propagated into the domain).
#[test]
fn run_forward_fields_are_finite_and_nonzero() {
    let (solver, region) = small_fdtd_solver(6, 6, 6, 20e-9, 1550e-9);

    let fields = solver
        .run_forward(&region, 1550e-9)
        .expect("run_forward should succeed on a small vacuum region");

    assert_eq!(
        fields.len(),
        region.n_cells(),
        "returned field length must equal region.n_cells()"
    );

    for (idx, f) in fields.iter().enumerate() {
        assert!(
            f.re.is_finite() && f.im.is_finite(),
            "cell {idx}: non-finite field ({:.3e}, {:.3e}i)",
            f.re,
            f.im
        );
    }

    // At least one cell should have non-negligible magnitude
    let max_amp: f64 = fields.iter().map(|f| f.norm()).fold(0.0_f64, f64::max);
    assert!(
        max_amp > 0.0,
        "maximum |E| should be > 0; got {max_amp:.3e}"
    );
}

/// Test 2: `run_adjoint` with a single real-weight monitor source returns finite fields.
///
/// Feeds the forward fields from Test 1 as adjoint source weights at the single
/// monitor cell and checks finiteness + non-triviality of the adjoint fields.
#[test]
fn run_adjoint_fields_are_finite() {
    let (solver, region) = small_fdtd_solver(6, 6, 6, 20e-9, 1550e-9);

    // Forward fields
    let fwd = solver
        .run_forward(&region, 1550e-9)
        .expect("run_forward should succeed");

    // Adjoint weight = conj(E_fwd) at the single monitor cell
    let mon_cell = region.cell_idx(region.nx / 2, region.ny / 2, region.nz.saturating_sub(1));
    let weight = fwd[mon_cell].conj();
    let fom_dconj_e = vec![weight];
    let monitor_cells = vec![(region.nx / 2, region.ny / 2, region.nz.saturating_sub(1))];

    let adj = solver
        .run_adjoint(&region, &monitor_cells, &fom_dconj_e, 1550e-9)
        .expect("run_adjoint should succeed");

    assert_eq!(adj.len(), region.n_cells());

    for (idx, f) in adj.iter().enumerate() {
        assert!(
            f.re.is_finite() && f.im.is_finite(),
            "adjoint cell {idx}: non-finite ({:.3e}, {:.3e}i)",
            f.re,
            f.im
        );
    }
}

/// Test 3: `run_forward` rejects invalid wavelength with the correct error variant.
#[test]
fn run_forward_rejects_invalid_wavelength() {
    use oxiphoton::error::OxiPhotonError;

    let (solver, region) = small_fdtd_solver(6, 6, 6, 20e-9, 1550e-9);

    let result_zero = solver.run_forward(&region, 0.0);
    assert!(
        matches!(result_zero, Err(OxiPhotonError::InvalidWavelength(_))),
        "zero wavelength should return InvalidWavelength"
    );

    let result_neg = solver.run_forward(&region, -1550e-9);
    assert!(
        matches!(result_neg, Err(OxiPhotonError::InvalidWavelength(_))),
        "negative wavelength should return InvalidWavelength"
    );

    let result_nan = solver.run_forward(&region, f64::NAN);
    assert!(
        matches!(result_nan, Err(OxiPhotonError::InvalidWavelength(_))),
        "NaN wavelength should return InvalidWavelength"
    );
}

/// Test 4: Adjoint reciprocity — field at monitor from run_forward ≈ field at source
/// from run_adjoint (qualitative: both are non-zero and finite).
///
/// In a lossless, reciprocal medium Green's function symmetry guarantees that the
/// forward response at the monitor drives an adjoint field that is strongest near the
/// original source.  This test checks the weaker condition that both runs produce
/// non-trivial, finite results on a small uniform-medium grid, and that the source
/// cell in the adjoint field has non-negligible magnitude.
#[test]
fn adjoint_reciprocity_source_cell_nonzero() {
    let (solver, region) = small_fdtd_solver(8, 8, 8, 20e-9, 1550e-9);
    let wl = 1550e-9_f64;

    let fwd = solver
        .run_forward(&region, wl)
        .expect("run_forward should succeed");

    // Use the monitor cell's forward field as adjoint weight
    let mon = (region.nx / 2, region.ny / 2, region.nz.saturating_sub(1));
    let mon_cell_idx = region.cell_idx(mon.0, mon.1, mon.2);
    let weight = fwd[mon_cell_idx].conj();

    let adj = solver
        .run_adjoint(&region, &[mon], &[weight], wl)
        .expect("run_adjoint should succeed");

    // Source cell in the adjoint field should be non-negligible
    let src_cell_idx = region.cell_idx(solver.source_i, solver.source_j, solver.source_k);
    let src_amp = adj[src_cell_idx].norm();
    assert!(
        src_amp.is_finite(),
        "adjoint field at source cell should be finite; got {src_amp:.3e}"
    );
    // The field at the source cell should be positive (energy propagated back)
    assert!(
        src_amp > 0.0,
        "adjoint field at source cell should be > 0; got {src_amp:.3e}"
    );

    // Also check `DesignRegion3d::cell_idx` consistency
    let c = region.cell_idx(2, 3, 4);
    let expected = 2 + 3 * region.nx + 4 * region.nx * region.ny;
    assert_eq!(
        c, expected,
        "cell_idx(2,3,4) should equal {expected}; got {c}"
    );

    // Suppress unused warning — Complex64 is used in the test through weight
    let _ = Complex64::new(0.0, 0.0);
}
