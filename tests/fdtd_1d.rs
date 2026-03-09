use oxiphoton::fdtd::*;

fn make_solver(nz: usize) -> Fdtd1d {
    let dz = 10e-9;
    Fdtd1d::new(nz, dz, &BoundaryConfig::pml(20))
}

#[test]
fn fdtd1d_zero_initial_fields() {
    let solver = make_solver(200);
    assert!(solver.grid.ex.iter().all(|&v| v == 0.0));
    assert!(solver.grid.hy.iter().all(|&v| v == 0.0));
}

#[test]
fn fdtd1d_stable_under_long_run() {
    let mut solver = make_solver(200);
    let pulse = GaussianEnvelope::new(30.0 * solver.dt, 8.0 * solver.dt);
    solver.add_source(PlaneWaveSource::new(80, Box::new(pulse)));
    solver.run(2000);

    let max_ex: f64 = solver.grid.ex.iter().map(|v| v.abs()).fold(0.0, f64::max);
    assert!(max_ex.is_finite(), "Fields should remain finite");
    // After PML absorption, field should decay significantly
    // This is a stability check, not a numerical accuracy check
    assert!(max_ex < 1e10, "Field amplitude should not blow up");
}

#[test]
fn fdtd1d_pml_absorbs_pulse() {
    // Run a pulse to the edge of the grid and verify PML absorbs it
    let nz = 500;
    let mut solver = make_solver(nz);
    let pulse = GaussianEnvelope::new(30.0 * solver.dt, 8.0 * solver.dt);
    solver.add_source(PlaneWaveSource::new(100, Box::new(pulse)));

    // Run until pulse should have reached the right PML and been absorbed
    let steps = nz * 2;
    solver.run(steps);

    // After absorption, max field should be very small
    let max_ex: f64 = solver.grid.ex.iter().map(|v| v.abs()).fold(0.0, f64::max);
    assert!(
        max_ex < 1e-2,
        "PML should absorb pulse, max_ex={max_ex:.3e}"
    );
}

#[test]
fn fdtd1d_dft_monitor_records_frequency() {
    let nz = 500;
    let mut solver = make_solver(nz);

    let f0 = 300e12; // 300 THz ~ 1000 nm
    let waveform =
        GaussianModulated::from_wavelength(oxiphoton::units::Wavelength::from_nm(1000.0), 3.0);
    solver.add_source(PlaneWaveSource::new(100, Box::new(waveform)));
    solver.add_dft_monitor(DftMonitor1d::new(300, &[f0]));

    solver.run(3000);

    let dft_amplitude = solver.dft_monitors[0].e_dft[0].norm();
    assert!(
        dft_amplitude > 0.0,
        "DFT monitor should record nonzero signal"
    );
}

#[test]
fn fdtd1d_courant_condition() {
    use oxiphoton::fdtd::courant::courant_number;
    let dz = 10e-9;
    let solver = make_solver(200);
    let s = courant_number(solver.dt, dz, 1.0);
    // Courant number should be <= 1 (we use 0.99 factor)
    assert!(s <= 1.0, "Courant condition violated: S={s}");
    assert!(s > 0.98, "Courant number too conservative: S={s}");
}

#[test]
fn fdtd1d_energy_decreases_with_pml() {
    let nz = 300;
    let mut solver = make_solver(nz);
    let pulse = GaussianEnvelope::new(20.0 * solver.dt, 5.0 * solver.dt);
    solver.add_source(PlaneWaveSource::new(80, Box::new(pulse)));

    // Run for a short time and record peak energy
    solver.run(50);
    let peak_energy: f64 = solver.grid.ex.iter().map(|v| v * v).sum::<f64>()
        + solver.grid.hy.iter().map(|v| v * v).sum::<f64>();

    // Run more (pulse should disperse into PML)
    solver.run(2000);
    let final_energy: f64 = solver.grid.ex.iter().map(|v| v * v).sum::<f64>()
        + solver.grid.hy.iter().map(|v| v * v).sum::<f64>();

    if peak_energy > 0.0 {
        assert!(
            final_energy < peak_energy,
            "PML should remove energy: initial={peak_energy:.3e}, final={final_energy:.3e}"
        );
    }
}
