use oxiphoton::fdtd::*;
use oxiphoton::units::Wavelength;

fn make_2d_solver(nx: usize, ny: usize) -> Fdtd2dTe {
    let d = 20e-9;
    Fdtd2dTe::new(nx, ny, d, d, &BoundaryConfig::pml(15))
}

#[test]
fn fdtd2d_zero_initial_fields() {
    let solver = make_2d_solver(60, 60);
    assert!(solver.grid.hz.iter().all(|&v| v == 0.0));
    assert!(solver.grid.ex.iter().all(|&v| v == 0.0));
    assert!(solver.grid.ey.iter().all(|&v| v == 0.0));
}

#[test]
fn fdtd2d_stable_under_run() {
    let mut solver = make_2d_solver(80, 80);
    let pulse = GaussianEnvelope::new(20.0 * solver.dt, 5.0 * solver.dt);

    for step in 0..500 {
        let amp = pulse.amplitude(step as f64 * solver.dt);
        solver.inject_hz(40, 40, amp * 0.01);
        solver.step();
    }

    assert!(solver.grid.hz.iter().all(|&v| v.is_finite()));
    assert!(solver.grid.ex.iter().all(|&v| v.is_finite()));
    assert!(solver.grid.ey.iter().all(|&v| v.is_finite()));
}

#[test]
fn fdtd2d_pml_absorbs_radiation() {
    let nx = 100;
    let ny = 100;
    let mut solver = make_2d_solver(nx, ny);

    // Inject a pulse at center
    let pulse = GaussianEnvelope::new(15.0 * solver.dt, 4.0 * solver.dt);
    for step in 0..100 {
        let amp = pulse.amplitude(step as f64 * solver.dt);
        solver.inject_hz(nx / 2, ny / 2, amp * 0.01);
        solver.step();
    }

    let peak: f64 = solver.grid.hz.iter().map(|v| v.abs()).fold(0.0, f64::max);

    // Run until PML should absorb the radiation
    solver.run(2000);

    let final_max: f64 = solver.grid.hz.iter().map(|v| v.abs()).fold(0.0, f64::max);

    if peak > 0.0 {
        assert!(
            final_max < peak,
            "PML should absorb radiation: peak={peak:.3e}, final={final_max:.3e}"
        );
    }
}

#[test]
fn fdtd2d_dft_box_records_fields() {
    use oxiphoton::units::conversion::SPEED_OF_LIGHT;
    let nx = 80;
    let ny = 80;
    let d = 20e-9;
    let mut solver = Fdtd2dTe::new(nx, ny, d, d, &BoundaryConfig::pml(15));

    let f0 = SPEED_OF_LIGHT / 500e-9;
    let dft = DftBox2d::new(nx, ny, &[f0]);
    solver.add_dft_box(dft);

    let pulse = GaussianEnvelope::new(20.0 * solver.dt, 5.0 * solver.dt);
    for step in 0..1000 {
        let amp = pulse.amplitude(step as f64 * solver.dt);
        solver.inject_hz(nx / 2, ny / 2, amp * 0.01);
        solver.step();
    }

    // DFT box should have accumulated finite values
    let hz_norm: f64 = solver.dft_boxes[0].hz_dft[0].iter().map(|v| v.norm()).sum();
    assert!(hz_norm.is_finite(), "DFT should produce finite values");
}

#[test]
fn fdtd2d_courant_condition() {
    use oxiphoton::fdtd::courant::courant_number;
    let d = 20e-9;
    let solver = make_2d_solver(60, 60);
    let s = courant_number(solver.dt, d, 1.0);
    // For 2D, Courant number should be <= 1/sqrt(2) ~ 0.707
    assert!(s <= 1.0 / 2.0_f64.sqrt() + 1e-10);
    assert!(s > 0.68, "Courant number too conservative: S={s}");
}

#[test]
fn fdtd2d_dielectric_cylinder() {
    use oxiphoton::geometry::Circle2d;
    use oxiphoton::material::Sellmeier;

    let nx = 100;
    let ny = 100;
    let d = 10e-9;
    let mut solver = Fdtd2dTe::new(nx, ny, d, d, &BoundaryConfig::pml(15));

    // Place a dielectric cylinder at center
    let center_x = nx as f64 * d / 2.0;
    let center_y = ny as f64 * d / 2.0;
    let radius = 150e-9;
    let circle = Circle2d::new(center_x, center_y, radius);
    let si = Sellmeier::si();
    solver
        .grid
        .fill_shape(&circle, &si, Wavelength::from_nm(1550.0));

    // Verify permittivity was set inside the circle
    let i_c = nx / 2;
    let j_c = ny / 2;
    let idx = j_c * (nx + 1) + i_c;
    let eps_inside = solver.grid.eps_ex[idx];
    // Si at 1550nm: n ~ 3.476, eps = n^2 ~ 12.08
    assert!(
        eps_inside > 10.0,
        "Permittivity inside Si should be > 10, got {eps_inside}"
    );

    // Run to verify stability with dielectric
    solver.run(200);
    assert!(solver.grid.hz.iter().all(|&v| v.is_finite()));
}
