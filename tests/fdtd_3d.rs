use oxiphoton::fdtd::{
    BlochFdtd1d, BoundaryConfig, DrudeParams, Fdtd1dDrude, Fdtd3d, NearToFarField2d,
};
use std::f64::consts::PI;

// ── 3D FDTD ──────────────────────────────────────────────────────────────────

#[test]
fn fdtd3d_creates_with_correct_size() {
    let s = Fdtd3d::new(20, 20, 20, 15e-9, 15e-9, 15e-9, &BoundaryConfig::pml(5));
    assert_eq!(s.ex.len(), 20 * 20 * 20);
    assert_eq!(s.ez.len(), 20 * 20 * 20);
}

#[test]
fn fdtd3d_stable_without_source() {
    let mut s = Fdtd3d::new(24, 24, 24, 15e-9, 15e-9, 15e-9, &BoundaryConfig::pml(6));
    s.run(50);
    let max_e: f64 =
        s.ex.iter()
            .chain(s.ey.iter())
            .chain(s.ez.iter())
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
    assert!(
        max_e < 1e-30,
        "Fields should remain zero without source: max_e={max_e:.2e}"
    );
}

#[test]
fn fdtd3d_point_source_generates_field() {
    let mut s = Fdtd3d::new(30, 30, 30, 15e-9, 15e-9, 15e-9, &BoundaryConfig::pml(7));
    let dt = s.dt;
    let tau = 10.0 * dt;
    let t0 = 3.0 * tau;
    for step in 0..80 {
        let t = step as f64 * dt;
        let amp = (-(t - t0).powi(2) / (2.0 * tau * tau)).exp();
        s.inject_ez(15, 15, 15, amp);
        s.step();
    }
    let peak = s.peak_ez();
    assert!(peak.is_finite(), "Peak must be finite");
}

#[test]
fn fdtd3d_courant_number_below_unity() {
    use oxiphoton::units::conversion::SPEED_OF_LIGHT;
    let dx = 20e-9;
    let s = Fdtd3d::new(20, 20, 20, dx, dx, dx, &BoundaryConfig::pml(5));
    let inv = (3.0 / (dx * dx)).sqrt(); // isotropic
    let cn = SPEED_OF_LIGHT * s.dt * inv;
    assert!(cn < 1.0, "Courant number = {cn:.4} >= 1 (unstable)");
}

#[test]
fn fdtd3d_dielectric_fill() {
    let mut s = Fdtd3d::new(30, 30, 30, 10e-9, 10e-9, 10e-9, &BoundaryConfig::pml(5));
    s.fill_box(10, 20, 10, 20, 10, 20, 11.7, 1.0); // Si-like
                                                   // interior cell should have eps=11.7
    let idx = s.nx * s.ny * 15 + s.nx * 15 + 15;
    assert!((s.eps_r[idx] - 11.7).abs() < 1e-10);
}

// ── Dispersive FDTD ──────────────────────────────────────────────────────────

#[test]
fn dispersive_drude_gold_propagation() {
    let mut s = Fdtd1dDrude::new(300, 5e-9, &BoundaryConfig::pml(20));
    s.fill_drude(400e-9, 450e-9, DrudeParams::gold());

    let dt = s.dt;
    let tau = 8.0 * dt;
    for step in 0..200 {
        let t = step as f64 * dt;
        let amp = (-(t - 3.0 * tau).powi(2) / (2.0 * tau * tau)).exp();
        s.inject_ex(60, amp);
        s.step();
    }
    assert!(s.ex.iter().all(|&v| v.is_finite()), "Ex should be finite");
    assert!(s.hy.iter().all(|&v| v.is_finite()), "Hy should be finite");
}

#[test]
fn dispersive_drude_silver_runs() {
    let mut s = Fdtd1dDrude::new(200, 5e-9, &BoundaryConfig::pml(20));
    s.fill_drude(300e-9, 350e-9, DrudeParams::silver());
    s.run(100);
    assert!(s.ex.iter().all(|&v| v.is_finite()));
}

#[test]
fn drude_params_physical_values() {
    let g = DrudeParams::gold();
    let ag = DrudeParams::silver();
    assert!(g.omega_p > 1e15 && g.omega_p < 1e17);
    assert!(ag.omega_p > 1e15 && ag.omega_p < 1e17);
    assert!(g.eps_inf > 1.0);
    assert!(ag.eps_inf > 1.0);
}

// ── Bloch Periodic BC ─────────────────────────────────────────────────────────

#[test]
fn bloch_bc_zero_k_matches_periodic() {
    // At k_B=0, field should satisfy E(x+L) = E(x)
    let mut s = BlochFdtd1d::new(64, 10e-9, 0.0);
    // Inject and run
    for step in 0..50 {
        let t = step as f64 * s.dt;
        s.inject_ex(32, (2.0 * PI * 200e12 * t).sin(), 0.0);
        s.step();
    }
    // Fields should be finite
    assert!(s.ex_re.iter().all(|&v| v.is_finite()));
    assert!(s.hy_re.iter().all(|&v| v.is_finite()));
}

#[test]
fn bloch_bc_courant_stable() {
    use oxiphoton::units::conversion::SPEED_OF_LIGHT;
    let s = BlochFdtd1d::new(100, 5e-9, 1e7);
    let cn = SPEED_OF_LIGHT * s.dt / s.dz;
    assert!(cn <= 1.0, "Courant = {cn:.4}");
}

#[test]
fn bloch_bc_zone_boundary_phase() {
    // At zone boundary k_B = π/L, phase factor = exp(iπ) = -1
    let period = 1e-6;
    let nz = 100;
    let dz = period / nz as f64;
    let k_b = PI / period;
    let s = BlochFdtd1d::new(nz, dz, k_b);
    let ph_re = (k_b * s.period).cos();
    let ph_im = (k_b * s.period).sin();
    assert!(
        (ph_re + 1.0).abs() < 1e-10,
        "Zone boundary phase should be -1"
    );
    assert!(ph_im.abs() < 1e-10);
}

// ── Near-to-Far-Field ─────────────────────────────────────────────────────────

#[test]
fn ntff_zero_fields_zero_pattern() {
    let m = NearToFarField2d::new(80, 80, 20e-9, 20e-9, 15, 65, 15, 65, 2.0 * PI * 200e12);
    let angles = vec![0.0, 45.0, 90.0, 135.0, 180.0];
    let pattern = m.radiation_pattern(&angles);
    assert!(pattern.iter().all(|&p| p < 1e-60));
}

#[test]
fn ntff_pattern_nonnegative() {
    let m = NearToFarField2d::new(80, 80, 20e-9, 20e-9, 15, 65, 15, 65, 2.0 * PI * 200e12);
    let angles: Vec<f64> = (0..36).map(|i| i as f64 * 10.0).collect();
    let pattern = m.radiation_pattern(&angles);
    assert!(pattern.iter().all(|&p| p >= 0.0));
}
