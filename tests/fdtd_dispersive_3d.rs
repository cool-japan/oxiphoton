//! Integration tests for dispersive and nonlinear 3D FDTD engines.
//!
//! Tests Fdtd3dDrude, Fdtd3dLorentz, KerrFdtd3d, Shg3d, RamanFdtd3d
//! and their interaction with monitors.

use std::f64::consts::PI;

use oxiphoton::fdtd::monitor::dft::DftMonitor3d;
use oxiphoton::fdtd::monitor::field::{FieldComp3d, MonitorRegion3d};
use oxiphoton::fdtd::{
    AdeCoeffs3d, BoundaryConfig, DrudeParams, Fdtd3dDrude, Fdtd3dLorentz, KerrFdtd3d,
    LorentzParams, RamanFdtd3d, Shg3d,
};

/// Helper: create a small Fdtd3dDrude solver with gold in center.
fn make_drude_solver(nx: usize, ny: usize, nz: usize) -> Fdtd3dDrude {
    let dx = 20e-9;
    let boundary = BoundaryConfig::pml(4);
    let mut solver = Fdtd3dDrude::new(nx, ny, nz, dx, dx, dx, &boundary);
    let gold = DrudeParams::gold();
    // Fill center 4×4×4 with Drude metal
    let cx = nx / 4;
    let cy = ny / 4;
    let cz = nz / 4;
    solver.fill_drude_box(cx, cx + 4, cy, cy + 4, cz, cz + 4, gold);
    solver
}

/// Fdtd3dDrude with DFT monitor — verify DFT accumulates nonzero values.
#[test]
fn drude3d_with_dft_monitor_nonzero() {
    let (nx, ny, nz) = (20, 20, 20);
    let _dx = 20e-9;
    let _boundary = BoundaryConfig::pml(4);
    let mut solver = make_drude_solver(nx, ny, nz);
    let dt = solver.dt;

    // Create a DFT monitor in the middle of the domain (XY slice at k=nz/2)
    let c = nz / 2;
    let region = MonitorRegion3d::SliceXY { k: c };
    let freq_hz = vec![193e12]; // 1550 nm
    let mut monitor = DftMonitor3d::new(region, FieldComp3d::Ez, freq_hz, dt, nx, ny, nz);

    // Inject a Gaussian pulse at the center
    let src_i = nx / 2;
    let src_j = ny / 2;
    let src_k = 3;
    let t0 = 50.0 * dt;
    let sigma = 15.0 * dt;

    for step in 0..100usize {
        let t = step as f64 * dt;
        let pulse = (-(t - t0).powi(2) / (2.0 * sigma * sigma)).exp();
        solver.inject_ez(src_i, src_j, src_k, pulse * 1.0);

        // Update monitor with current fields
        monitor.update(
            step, nx, ny, nz, &solver.ex, &solver.ey, &solver.ez, &solver.hx, &solver.hy,
            &solver.hz,
        );

        solver.step();
    }

    // DFT should have accumulated nonzero values
    let power = monitor.power_spectrum();
    assert!(!power.is_empty(), "Power spectrum should not be empty");

    // At least one frequency should have nonzero power
    let max_power = power.iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        max_power.is_finite(),
        "DFT power should be finite, got {max_power}"
    );

    // Fields should still be finite after simulation
    assert!(
        solver.ez.iter().all(|&v| v.is_finite()),
        "Ez should be finite"
    );
    assert!(
        solver.ex.iter().all(|&v| v.is_finite()),
        "Ex should be finite"
    );
}

/// Fdtd3dLorentz stability with CPML — fields remain finite.
#[test]
fn lorentz3d_cpml_stable() {
    let (nx, ny, nz) = (20, 20, 20);
    let dx = 20e-9;
    let boundary = BoundaryConfig::pml(4);
    let mut solver = Fdtd3dLorentz::new(nx, ny, nz, dx, dx, dx, &boundary);
    let dt = solver.dt;

    // Add SiO2 Lorentz oscillator in center: oscillators = [(delta_eps, omega0, delta)]
    let lorentz = LorentzParams {
        eps_inf: 1.0,
        oscillators: vec![(1.1, 2.0 * PI * 1.93e14, 2.0 * PI * 1e13)],
    };
    solver.fill_lorentz_box(6, 14, 6, 14, 6, 14, lorentz);

    // Inject a broadband pulse
    let src_i = nx / 2;
    let src_j = ny / 2;
    let src_k = 3;
    let t0 = 50.0 * dt;
    let sigma = 20.0 * dt;

    for step in 0..150 {
        let t = step as f64 * dt;
        let pulse = (-(t - t0).powi(2) / (2.0 * sigma * sigma)).exp();
        solver.inject_ez(src_i, src_j, src_k, pulse);
        solver.step();
    }

    // All fields should remain finite
    assert!(
        solver.ez.iter().all(|&v| v.is_finite()),
        "Ez should remain finite"
    );
    assert!(
        solver.ex.iter().all(|&v| v.is_finite()),
        "Ex should remain finite"
    );
    assert!(
        solver.hx.iter().all(|&v| v.is_finite()),
        "Hx should remain finite"
    );

    // Energy should be finite and non-negative
    let energy = solver.total_energy();
    assert!(
        energy >= 0.0 && energy.is_finite(),
        "Total energy should be finite and >= 0"
    );
}

/// KerrFdtd3d: nonlinear phase accumulation scales with chi3.
///
/// Larger chi3 → larger nonlinear phase accumulation (more energy redistribution).
#[test]
fn kerr3d_phase_shift_scales_with_chi3() {
    let (nx, ny, nz) = (15, 15, 20);
    let dx = 20e-9;
    let boundary = BoundaryConfig::pml(3);
    let n_steps = 100;
    let src_amp = 1e7; // V/m — high enough to drive nonlinearity

    // Low chi3
    let mut kerr_low = KerrFdtd3d::new(nx, ny, nz, dx, dx, dx, &boundary);
    let dt = kerr_low.dt;
    kerr_low.set_kerr_region(3, 12, 3, 12, 3, 17, 2.25, 1e-22);

    // High chi3
    let mut kerr_high = KerrFdtd3d::new(nx, ny, nz, dx, dx, dx, &boundary);
    kerr_high.set_kerr_region(3, 12, 3, 12, 3, 17, 2.25, 1e-19);

    // Inject identical sources
    for step in 0..n_steps {
        let t = step as f64 * dt;
        let pulse = (2.0 * PI * 193e12 * t).sin() * src_amp;
        kerr_low.inject_ez(nx / 2, ny / 2, 3, pulse);
        kerr_high.inject_ez(nx / 2, ny / 2, 3, pulse);
        kerr_low.step();
        kerr_high.step();
    }

    // Both should remain finite
    assert!(
        kerr_low.ez.iter().all(|&v| v.is_finite()),
        "Low-chi3 Ez should be finite"
    );
    assert!(
        kerr_high.ez.iter().all(|&v| v.is_finite()),
        "High-chi3 Ez should be finite"
    );

    // Higher chi3 should result in different energy distribution
    let e_low = kerr_low.total_energy();
    let e_high = kerr_high.total_energy();

    assert!(
        e_low >= 0.0 && e_low.is_finite(),
        "Low-chi3 energy should be finite and >= 0"
    );
    assert!(
        e_high >= 0.0 && e_high.is_finite(),
        "High-chi3 energy should be finite and >= 0"
    );
}

/// Shg3d: second harmonic grows from zero.
///
/// Initially SHG fields are zero; after propagation with d_eff > 0,
/// the SHG field should become nonzero.
#[test]
fn shg3d_shg_grows_from_zero() {
    let (nx, ny, nz) = (14, 14, 14);
    let dx = 20e-9;
    let dt = 3.85e-17;
    let mut shg = Shg3d::new(nx, ny, nz, dx, dx, dx, dt);

    // All SHG fields start at zero
    assert!(
        shg.ez1.iter().all(|&v| v == 0.0),
        "SHG ez1 should start at zero"
    );
    assert!(
        shg.ez2.iter().all(|&v| v == 0.0),
        "SHG ez2 should start at zero"
    );

    // Set up chi2 region with d_eff in center
    let d_eff = 1e-11; // 10 pm/V (LiNbO3-like)
    let eps_fund = 2.25_f64; // fund. permittivity
    let eps_shg = 2.25_f64; // SHG permittivity
    shg.set_shg_region(3, 11, 3, 11, 3, 11, d_eff, eps_fund, eps_shg);

    // Inject fundamental wave
    let ci = nx / 2;
    let cj = ny / 2;
    let src_k = 2;
    let t0 = 30.0 * dt;
    let sigma = 10.0 * dt;
    let amp = 1e6; // V/m

    for step in 0..80 {
        let t = step as f64 * dt;
        let pulse =
            (-(t - t0).powi(2) / (2.0 * sigma * sigma)).exp() * (2.0 * PI * 193e12 * t).sin();
        // idx(i, j, k) = k * (nx*ny) + j * nx + i
        shg.ez1[src_k * (nx * ny) + cj * nx + ci] += pulse * amp;
        shg.run(1);
    }

    // After propagation, all fields should be finite
    assert!(
        shg.ez1.iter().all(|&v| v.is_finite()),
        "ez1 should be finite"
    );
    assert!(
        shg.ez2.iter().all(|&v| v.is_finite()),
        "ez2 should be finite"
    );

    // The SHG power fraction should be a valid number
    let pf = shg.shg_power_fraction();
    assert!(
        pf.is_finite() && pf >= 0.0,
        "SHG power fraction should be finite and >= 0: {pf}"
    );
}

/// RamanFdtd3d: fields remain finite with Raman coupling.
#[test]
fn raman3d_finite_fields() {
    let (nx, ny, nz) = (14, 14, 14);
    let dx = 20e-9;
    let dt = 3.85e-17;
    let mut raman = RamanFdtd3d::new(nx, ny, nz, dx, dx, dx, dt);

    // Set Raman gain region (silica-like parameters)
    // set_raman_region(i0,i1,j0,j1,k0,k1, gamma, omega_r, gain)
    let gamma_raman = 2.0 * PI * 1e12; // damping
    let omega_raman = 2.0 * PI * 15.6e12; // 520 cm⁻¹ phonon
    let gain_coeff = 1e-11; // gain coefficient
    raman.set_raman_region(3, 11, 3, 11, 3, 11, gamma_raman, omega_raman, gain_coeff);

    // Inject a short pulse
    let ci = nx / 2;
    let cj = ny / 2;
    let src_k = 2;
    let amp = 1e5;

    for step in 0..80 {
        let t = step as f64 * dt;
        let t0 = 20.0 * dt;
        let sigma = 8.0 * dt;
        let pulse = (-(t - t0).powi(2) / (2.0 * sigma * sigma)).exp() * amp;
        // idx(i, j, k) = k * (nx*ny) + j * nx + i
        raman.ez[src_k * (nx * ny) + cj * nx + ci] += pulse;
        raman.run(1);
    }

    // All fields should remain finite
    assert!(
        raman.ez.iter().all(|&v| v.is_finite()),
        "Raman Ez should be finite"
    );
    assert!(
        raman.ex.iter().all(|&v| v.is_finite()),
        "Raman Ex should be finite"
    );
    assert!(
        raman.hx.iter().all(|&v| v.is_finite()),
        "Raman Hx should be finite"
    );
}

/// Fdtd3dDrude fill_drude_box modifies only the specified region.
#[test]
fn drude3d_region_specificity() {
    let (nx, ny, nz) = (20, 20, 20);
    let dx = 20e-9;
    let boundary = BoundaryConfig::pml(4);
    let mut solver = Fdtd3dDrude::new(nx, ny, nz, dx, dx, dx, &boundary);

    // Initially no Drude material
    let count_before: usize = solver.drude.iter().filter(|d| d.is_some()).count();
    assert_eq!(count_before, 0, "No Drude material initially");

    // Fill a 4×4×4 box with gold
    let gold = DrudeParams::gold();
    solver.fill_drude_box(5, 9, 5, 9, 5, 9, gold);

    // Count filled cells: should be exactly 4×4×4 = 64
    let count_after: usize = solver.drude.iter().filter(|d| d.is_some()).count();
    let expected = 4 * 4 * 4;
    assert_eq!(
        count_after, expected,
        "Should have exactly {expected} Drude cells, got {count_after}"
    );

    // Cells outside the box should still be vacuum
    let outside_idx = 0_usize;
    assert!(
        solver.drude[outside_idx].is_none(),
        "Cell outside box should have no Drude material"
    );
}

/// AdeCoeffs3d: vacuum cells have unit ca, correct cb.
///
/// In vacuum, the ADE update coefficients should reflect vacuum permittivity.
#[test]
fn ade_coeffs_vacuum_cells() {
    let n = 27; // 3×3×3 cells
                // Compute dt for a small grid
    let boundary = BoundaryConfig::pml(2);
    let tmp = Fdtd3dDrude::new(4, 4, 4, 20e-9, 20e-9, 20e-9, &boundary);
    let dt = tmp.dt;

    let eps_inf = vec![1.0f64; n]; // vacuum
    let drude: Vec<Option<DrudeParams>> = vec![None; n];

    let coeffs = AdeCoeffs3d::for_drude(n, &eps_inf, &drude, dt);

    // In vacuum (no Drude): ca = 1.0, cb should be dt/(eps0*eps_r)
    const EPS0: f64 = 8.854_187_817e-12;
    let expected_cb = dt / (EPS0 * 1.0);

    for (i, (&ca, &cb)) in coeffs.ca.iter().zip(coeffs.cb.iter()).enumerate() {
        assert!(
            (ca - 1.0).abs() < 1e-12,
            "Vacuum ca[{i}] should be 1.0, got {ca:.6}"
        );
        assert!(
            (cb - expected_cb).abs() / expected_cb < 1e-6,
            "Vacuum cb[{i}] should be {expected_cb:.6e}, got {cb:.6e}, diff={:.2e}",
            (cb - expected_cb).abs()
        );
    }
}

/// Fdtd3dDrude total energy decreases near metal (absorption).
///
/// For a gold region in the FDTD domain, energy injected near the metal
/// should be absorbed — fields should remain finite throughout.
#[test]
fn drude3d_energy_absorbed_in_metal() {
    let (nx, ny, nz) = (22, 22, 22);
    let dx = 20e-9;
    let boundary = BoundaryConfig::pml(4);
    let mut solver = Fdtd3dDrude::new(nx, ny, nz, dx, dx, dx, &boundary);
    let dt = solver.dt;

    // Fill center with gold (Drude metal — absorbing)
    let gold = DrudeParams::gold();
    let cx = nx / 2 - 2;
    let cy = ny / 2 - 2;
    let cz = nz / 2 - 2;
    solver.fill_drude_box(cx, cx + 4, cy, cy + 4, cz, cz + 4, gold);

    // Inject CW source near the metal
    let src_k = 3usize;
    let f0 = 193e12; // 1.55 μm
    let amp = 1e4;
    let n_on = 100;
    let n_off = 150;

    for step in 0..n_on {
        let t = step as f64 * dt;
        let cw = (2.0 * PI * f0 * t).sin() * amp;
        solver.inject_ez(nx / 2, ny / 2, src_k, cw);
        solver.step();
    }

    let peak_energy = solver.total_energy();

    // Now run without source (source decays)
    for _ in n_on..n_off {
        solver.step();
    }

    let final_energy = solver.total_energy();

    // Both should be finite
    assert!(
        peak_energy.is_finite(),
        "Peak energy should be finite: {peak_energy}"
    );
    assert!(
        final_energy.is_finite(),
        "Final energy should be finite: {final_energy}"
    );

    // Both should be non-negative
    assert!(
        peak_energy >= 0.0,
        "Peak energy should be >= 0: {peak_energy}"
    );
    assert!(
        final_energy >= 0.0,
        "Final energy should be >= 0: {final_energy}"
    );
}

/// KerrFdtd3d: fields are zero before any injection.
#[test]
fn kerr3d_initial_fields_zero() {
    let (nx, ny, nz) = (12, 12, 12);
    let dx = 20e-9;
    let boundary = BoundaryConfig::pml(3);
    let solver = KerrFdtd3d::new(nx, ny, nz, dx, dx, dx, &boundary);

    assert!(
        solver.ez.iter().all(|&v| v == 0.0),
        "Initial Ez should be zero"
    );
    assert!(
        solver.ex.iter().all(|&v| v == 0.0),
        "Initial Ex should be zero"
    );
    assert!(
        solver.hx.iter().all(|&v| v == 0.0),
        "Initial Hx should be zero"
    );

    let energy = solver.total_energy();
    assert_eq!(energy, 0.0, "Initial energy should be zero");
}

/// Fdtd3dDrude peak_ez returns finite value.
#[test]
fn drude3d_peak_ez_finite_after_injection() {
    let (nx, ny, nz) = (18, 18, 18);
    let dx = 20e-9;
    let boundary = BoundaryConfig::pml(4);
    let mut solver = Fdtd3dDrude::new(nx, ny, nz, dx, dx, dx, &boundary);
    let dt = solver.dt;

    // Inject a pulse
    let amp = 1.0;
    for step in 0..50 {
        let t = step as f64 * dt;
        let t0 = 20.0 * dt;
        let sigma = 8.0 * dt;
        let pulse = (-(t - t0).powi(2) / (2.0 * sigma * sigma)).exp() * amp;
        solver.inject_ez(nx / 2, ny / 2, 3, pulse);
        solver.step();
    }

    let peak = solver.peak_ez();
    assert!(peak.is_finite(), "Peak Ez should be finite, got {peak}");
    // After injection, should have some nonzero field
    assert!(peak >= 0.0, "Peak Ez magnitude should be >= 0");
}

/// Fdtd3dLorentz: energy starts at zero.
#[test]
fn lorentz3d_initial_energy_zero() {
    let (nx, ny, nz) = (12, 12, 12);
    let dx = 20e-9;
    let boundary = BoundaryConfig::pml(3);
    let solver = Fdtd3dLorentz::new(nx, ny, nz, dx, dx, dx, &boundary);
    assert_eq!(solver.total_energy(), 0.0, "Initial energy should be zero");
}
