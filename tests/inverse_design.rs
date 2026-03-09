use oxiphoton::inverse::fabrication::{
    apply_overlay_error, check_min_feature_size, proximity_correction,
};
use oxiphoton::inverse::shape::LevelSetField;
/// Inverse design integration tests — fabrication constraints, level set,
/// curvature penalty, convergence history, Nelder-Mead, and proximity correction.
use oxiphoton::inverse::{
    ConvergenceHistory, CurvaturePenalty, FabricationConstraints, LevelSet, NelderMead,
};

// ── FabricationConstraints ────────────────────────────────────────────────────

#[test]
fn fab_constraints_is_binary_all_zeros() {
    let fc = FabricationConstraints::new(100e-9, 50e-9);
    let rho = vec![0.0_f64; 20];
    assert!(fc.is_binary(&rho, 0.05));
}

#[test]
fn fab_constraints_is_binary_all_ones() {
    let fc = FabricationConstraints::new(100e-9, 50e-9);
    let rho = vec![1.0_f64; 20];
    assert!(fc.is_binary(&rho, 0.05));
}

#[test]
fn fab_constraints_is_not_binary_gray() {
    let fc = FabricationConstraints::new(100e-9, 50e-9);
    let rho = vec![0.5_f64; 20];
    assert!(!fc.is_binary(&rho, 0.05));
}

#[test]
fn fab_constraints_binarization_penalty_is_zero_for_binary() {
    let fc = FabricationConstraints::new(100e-9, 50e-9);
    // Alternating 0 and 1 → all binary
    let rho: Vec<f64> = (0..20)
        .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
        .collect();
    let penalty = fc.binarization_penalty(&rho);
    assert!(
        penalty < 1e-12,
        "Binary field should have zero binarization penalty: {penalty}"
    );
}

#[test]
fn fab_constraints_binarization_penalty_positive_for_gray() {
    let fc = FabricationConstraints::new(100e-9, 50e-9);
    let rho = vec![0.5_f64; 20];
    let penalty = fc.binarization_penalty(&rho);
    assert!(
        penalty > 0.0,
        "Gray field should have positive binarization penalty: {penalty}"
    );
}

// ── check_min_feature_size ────────────────────────────────────────────────────

#[test]
fn check_mfs_solid_block_passes() {
    // A fully solid 10×10 block: single feature of size 10 cells → passes any mfs ≤ 10 cells
    let nx = 10_usize;
    let ny = 10_usize;
    let dx = 100e-9_f64; // 100 nm
    let mfs = 200e-9_f64; // 200 nm = 2 cells (satisfied by 10-cell block)
    let rho = vec![1.0_f64; nx * ny];
    assert!(check_min_feature_size(&rho, nx, ny, dx, mfs));
}

#[test]
fn check_mfs_empty_field_passes() {
    let nx = 10_usize;
    let ny = 10_usize;
    let dx = 100e-9_f64;
    let mfs = 200e-9_f64;
    let rho = vec![0.0_f64; nx * ny];
    assert!(check_min_feature_size(&rho, nx, ny, dx, mfs));
}

// ── proximity_correction ──────────────────────────────────────────────────────

#[test]
fn proximity_correction_uniform_field_unchanged() {
    // A uniform density field should remain (approximately) unchanged after
    // proximity correction (convolution with shift kernel).
    let nx = 10_usize;
    let ny = 10_usize;
    let dx = 100e-9_f64;
    let rho = vec![0.5_f64; nx * ny];
    let corrected = proximity_correction(&rho, nx, ny, dx, 100e-9, 0.1);
    // Mean should be close to 0.5 since uniform
    let mean: f64 = corrected.iter().sum::<f64>() / corrected.len() as f64;
    assert!(
        (mean - 0.5).abs() < 0.1,
        "uniform field mean should be ~0.5: {mean}"
    );
}

// ── apply_overlay_error ───────────────────────────────────────────────────────

#[test]
fn apply_overlay_error_zero_shift_identity() {
    let nx = 8_usize;
    let ny = 8_usize;
    let dx = 100e-9_f64;
    let rho: Vec<f64> = (0..nx * ny).map(|i| (i % 3) as f64 * 0.5).collect();
    let result = apply_overlay_error(&rho, nx, ny, dx, 0.0, 0.0);
    // Zero shift → output should equal input
    for (a, b) in rho.iter().zip(result.iter()) {
        assert!(
            (a - b).abs() < 1e-10,
            "Zero shift should be identity: {a} vs {b}"
        );
    }
}

#[test]
fn apply_overlay_error_nonzero_shift_changes_field() {
    let nx = 8_usize;
    let ny = 8_usize;
    let dx = 100e-9_f64;
    // Non-uniform field so a shift will actually change things
    let mut rho = vec![0.0_f64; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            rho[j * nx + i] = if i < nx / 2 { 1.0 } else { 0.0 };
        }
    }
    let shifted = apply_overlay_error(&rho, nx, ny, dx, 1.5 * dx, 0.0);
    let diff: f64 = rho
        .iter()
        .zip(shifted.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 0.0,
        "Non-zero shift should change a non-uniform field"
    );
}

// ── ConvergenceHistory ────────────────────────────────────────────────────────

#[test]
fn convergence_history_push_and_best() {
    let mut hist = ConvergenceHistory::new();
    hist.push(0, 0.5);
    hist.push(1, 0.8);
    hist.push(2, 1.0);
    // best() returns the maximum value
    assert!(
        (hist.best() - 1.0).abs() < 1e-12,
        "best should be 1.0: {}",
        hist.best()
    );
}

#[test]
fn convergence_history_not_converged_at_start() {
    let mut hist = ConvergenceHistory::new();
    hist.push(0, 1.0);
    assert!(!hist.converged(1e-6));
}

#[test]
fn convergence_history_converged_when_flat() {
    let mut hist = ConvergenceHistory::new();
    for i in 0..20 {
        hist.push(i, 0.5);
    }
    assert!(hist.converged(1e-6), "Flat history should be converged");
}

#[test]
fn convergence_history_best_is_maximum() {
    let mut hist = ConvergenceHistory::new();
    for i in 0..10 {
        hist.push(i, i as f64); // 0, 1, 2, ... 9
    }
    // best() returns max
    assert!(
        (hist.best() - 9.0).abs() < 1e-12,
        "best should be 9.0: {}",
        hist.best()
    );
}

// ── NelderMead ────────────────────────────────────────────────────────────────

#[test]
fn nelder_mead_minimizes_quadratic_1d() {
    // Minimize f(x) = (x - 3)^2
    let mut nm = NelderMead::new(vec![0.0]);
    let (result, fval) = nm.run(|x| (x[0] - 3.0).powi(2), 200, 1e-8);
    assert!(
        (result[0] - 3.0).abs() < 0.05,
        "Nelder-Mead should find minimum at x=3: {} (f={})",
        result[0],
        fval
    );
}

#[test]
fn nelder_mead_minimizes_quadratic_2d() {
    // Minimize f(x,y) = (x-1)^2 + (y+2)^2
    let mut nm = NelderMead::new(vec![0.0, 0.0]);
    let (result, fval) = nm.run(|x| (x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2), 400, 1e-8);
    assert!(
        (result[0] - 1.0).abs() < 0.1,
        "x should converge to 1: {} (f={})",
        result[0],
        fval
    );
    assert!(
        (result[1] + 2.0).abs() < 0.1,
        "y should converge to -2: {} (f={})",
        result[1],
        fval
    );
}

// ── LevelSetField ─────────────────────────────────────────────────────────────

#[test]
fn level_set_field_new_construction() {
    let lsf = LevelSetField::new(10, 10, 10e-9);
    // Internal phi should be initialized (defaults to 0)
    let rho = lsf.to_density();
    assert_eq!(rho.len(), 100);
}

#[test]
fn level_set_field_from_density_solid_round_trip() {
    let nx = 8_usize;
    let ny = 8_usize;
    let dx = 50e-9_f64;
    let rho = vec![1.0_f64; nx * ny];
    let lsf = LevelSetField::from_density(&rho, nx, ny, dx);
    let rho_out = lsf.to_density();
    // Solid field → all densities should be 1.0 (or close to it)
    let mean: f64 = rho_out.iter().sum::<f64>() / rho_out.len() as f64;
    assert!(mean > 0.8, "Solid density round-trip: mean={mean}");
}

#[test]
fn level_set_field_from_density_void_round_trip() {
    let nx = 8_usize;
    let ny = 8_usize;
    let dx = 50e-9_f64;
    let rho = vec![0.0_f64; nx * ny];
    let lsf = LevelSetField::from_density(&rho, nx, ny, dx);
    let rho_out = lsf.to_density();
    let mean: f64 = rho_out.iter().sum::<f64>() / rho_out.len() as f64;
    assert!(mean < 0.2, "Void density round-trip: mean={mean}");
}

// ── CurvaturePenalty ──────────────────────────────────────────────────────────

#[test]
fn curvature_penalty_smooth_field_near_zero() {
    let cp = CurvaturePenalty::new(1e6);
    let nx = 10_usize;
    let ny = 10_usize;
    let dx = 50e-9_f64;
    // Linearly varying field has zero curvature
    let rho: Vec<f64> = (0..nx * ny)
        .map(|i| (i as f64) / (nx * ny) as f64)
        .collect();
    let penalty = cp.penalty(&rho, nx, ny, dx);
    assert!(
        penalty < 1.0,
        "Linearly varying field should have low curvature penalty: {penalty}"
    );
}

#[test]
fn curvature_penalty_constant_field_is_zero() {
    let cp = CurvaturePenalty::new(1e6);
    let nx = 8_usize;
    let ny = 8_usize;
    let dx = 50e-9_f64;
    let rho = vec![0.5_f64; nx * ny];
    let penalty = cp.penalty(&rho, nx, ny, dx);
    assert!(
        penalty < 1e-10,
        "Constant field should have zero curvature penalty: {penalty}"
    );
}

#[test]
fn curvature_penalty_circular_boundary_nonzero() {
    // A circular field has high curvature at its boundary → penalty > 0 for small kappa_max
    let nx = 12_usize;
    let ny = 12_usize;
    let dx = 50e-9_f64;
    // Very small kappa_max: any curvature > 0 triggers penalty
    let kappa_max = 1.0; // 1 m^-1 is essentially zero given dx = 50nm → curvature ~ 1/dx ~ 2e7 m^-1
    let cp = CurvaturePenalty::new(kappa_max);
    // Build a soft circular shape (smooth transition, NOT binary) so gradient > 0
    let cx = nx as f64 / 2.0;
    let cy = ny as f64 / 2.0;
    let r = 3.5_f64; // cells
    let rho: Vec<f64> = (0..nx * ny)
        .map(|k| {
            let i = (k % nx) as f64;
            let j = (k / nx) as f64;
            let dist = ((i - cx).powi(2) + (j - cy).powi(2)).sqrt();
            // Smooth step: tanh-based to create gradient at boundary
            let tanh_val = (-(dist - r)).tanh() * 0.5 + 0.5;
            tanh_val.clamp(0.0, 1.0)
        })
        .collect();
    let penalty = cp.penalty(&rho, nx, ny, dx);
    assert!(
        penalty > 0.0,
        "Circular boundary should have positive curvature penalty: {penalty}"
    );
}

// ── LevelSet (basic) ──────────────────────────────────────────────────────────

#[test]
fn level_set_solid_all_inside() {
    let ls = LevelSet::solid(10, 10, 50e-9);
    let indicator = ls.material_indicator();
    // All cells should be inside (indicator = 1)
    assert!(
        indicator.iter().all(|&v| v > 0.5),
        "Solid level set: all cells should be inside"
    );
}

#[test]
fn level_set_void_all_outside() {
    let ls = LevelSet::void(10, 10, 50e-9);
    let indicator = ls.material_indicator();
    assert!(
        indicator.iter().all(|&v| v < 0.5),
        "Void level set: all cells should be outside"
    );
}

#[test]
fn level_set_circle_center_inside() {
    let nx = 20_usize;
    let ny = 20_usize;
    let dx = 50e-9_f64;
    let cx = nx as f64 * dx / 2.0;
    let cy = ny as f64 * dx / 2.0;
    let r = 200e-9_f64; // 4 cells radius
    let ls = LevelSet::circle(nx, ny, dx, cx, cy, r);
    let indicator = ls.material_indicator();
    // Center cell should be inside
    let center_idx = (ny / 2) * nx + nx / 2;
    assert!(
        indicator[center_idx] > 0.5,
        "Center cell of circle should be inside"
    );
}
