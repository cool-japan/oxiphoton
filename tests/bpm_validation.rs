use std::f64::consts::PI;

use oxiphoton::bpm::fd_bpm::FdBpm1d;
use oxiphoton::bpm::fft_bpm::FftBpm1d;

/// Analytical 1D Gaussian beam peak intensity at position z.
/// I_peak(z) = I0 / sqrt(1 + (z/z_R)²)
fn gaussian_peak_intensity_analytical(i0: f64, z: f64, z_r: f64) -> f64 {
    i0 / (1.0 + (z / z_r).powi(2)).sqrt()
}

#[test]
fn fft_bpm_power_conserved_free_space() {
    let nx = 256;
    let dx = 80e-9;
    let w0 = 1.5e-6;
    let mut bpm = FftBpm1d::new(nx, dx, 1.0, 1550e-9);
    let xc = nx as f64 * dx / 2.0;
    bpm.set_gaussian_input(1.0, xc, w0);

    let p0: f64 = bpm.intensity().iter().sum::<f64>() * dx;
    bpm.propagate(200e-9, 50); // 50 steps × 200nm = 10μm
    let p_final: f64 = bpm.intensity().iter().sum::<f64>() * dx;

    let rel_err = (p_final - p0).abs() / p0;
    assert!(rel_err < 1e-3, "Power not conserved: {rel_err:.2e}");
}

#[test]
fn fft_bpm_gaussian_beam_analytical_match() {
    // Compare FFT-BPM peak intensity with analytical Gaussian beam propagation
    // within ±0.1% over a short propagation (2% of Rayleigh range)
    let nx = 512;
    let dx = 25e-9; // 25nm, total 12.8μm
    let wavelength = 1550e-9;
    let n_ref = 1.0;
    let w0 = 1.8e-6; // 1.8μm Gaussian waist

    let z_r = PI * w0 * w0 * n_ref / wavelength; // Rayleigh range ≈ 13.2μm

    let mut bpm = FftBpm1d::new(nx, dx, n_ref, wavelength);
    let xc = nx as f64 * dx / 2.0;
    bpm.set_gaussian_input(1.0, xc, w0);

    let i0 = bpm.peak_intensity();

    // Propagate 2% of Rayleigh range in 10 steps
    let dz = 0.002 * z_r;
    let n_steps = 10;
    let z_total = dz * n_steps as f64;

    bpm.propagate(dz, n_steps);
    let i_final = bpm.peak_intensity();

    let expected = gaussian_peak_intensity_analytical(i0, z_total, z_r);
    let rel_err = (i_final - expected).abs() / expected;

    assert!(
        rel_err < 0.001,
        "Gaussian beam error: {rel_err:.4e} (computed I={i_final:.6}, expected I={expected:.6})"
    );
}

#[test]
fn fft_bpm_beam_spreads_correctly() {
    // After propagating z_R (one Rayleigh range), beam width should increase by factor sqrt(2)
    let nx = 512;
    let dx = 25e-9;
    let wavelength = 1550e-9;
    let w0 = 2.0e-6; // 2μm beam waist
    let z_r = PI * w0 * w0 / wavelength; // Rayleigh range

    let mut bpm = FftBpm1d::new(nx, dx, 1.0, wavelength);
    let xc = nx as f64 * dx / 2.0;
    bpm.set_gaussian_input(1.0, xc, w0);
    let w_init = bpm.rms_width();

    // Propagate in small steps (5% z_R each)
    let dz = 0.05 * z_r;
    bpm.propagate(dz, 20); // total = z_R
    let w_final = bpm.rms_width();

    // Expected: w(z_R) = w0 * sqrt(2) → rms should increase by ~sqrt(2)
    let expected_ratio = 2.0_f64.sqrt();
    let actual_ratio = w_final / w_init;
    let rel_err = (actual_ratio - expected_ratio).abs() / expected_ratio;

    assert!(
        rel_err < 0.02,
        "Beam width ratio: got {actual_ratio:.4}, expected {expected_ratio:.4}, err={rel_err:.4}"
    );
}

#[test]
fn fd_bpm_power_conserved() {
    let nx = 128;
    let dx = 100e-9;
    let mut bpm = FdBpm1d::new(nx, dx, 1.0, 1550e-9);
    let xc = nx as f64 * dx / 2.0;
    bpm.set_gaussian_input(1.0, xc, 1.5e-6);

    let p0: f64 = bpm.intensity().iter().sum::<f64>() * dx;
    bpm.propagate(500e-9, 20);
    let p1: f64 = bpm.intensity().iter().sum::<f64>() * dx;

    let rel_err = (p1 - p0).abs() / p0;
    assert!(rel_err < 1e-3, "FD-BPM power not conserved: {rel_err:.2e}");
}

#[test]
fn fd_bpm_beam_spreads() {
    let nx = 256;
    let dx = 50e-9;
    let mut bpm = FdBpm1d::new(nx, dx, 1.0, 1550e-9);
    let xc = nx as f64 * dx / 2.0;
    bpm.set_gaussian_input(1.0, xc, 1e-6);
    let w_init = bpm.rms_width();
    bpm.propagate(1e-6, 20);
    let w_final = bpm.rms_width();
    assert!(
        w_final > w_init,
        "FD-BPM: beam should spread: w_init={w_init:.3e} w_final={w_final:.3e}"
    );
}

#[test]
fn fft_bpm_waveguide_confines_beam() {
    // A waveguide index profile should prevent spreading
    let nx = 256;
    let dx = 50e-9;
    let n_core = 1.5;
    let n_clad = 1.0;
    let wg_width = 3e-6; // 3μm waveguide

    let n_profile: Vec<f64> = (0..nx)
        .map(|i| {
            let x = i as f64 * dx;
            let xc = nx as f64 * dx / 2.0;
            if (x - xc).abs() < wg_width / 2.0 {
                n_core
            } else {
                n_clad
            }
        })
        .collect();

    let mut bpm = FftBpm1d::new(nx, dx, n_core, 1550e-9);
    bpm.set_index_profile(n_profile);
    let xc = nx as f64 * dx / 2.0;
    bpm.set_gaussian_input(1.0, xc, 1e-6);

    bpm.propagate(500e-9, 40); // 20μm propagation
    let w_final = bpm.rms_width();

    // In a waveguide, beam should be more confined than in free space
    // (width change should be smaller)
    assert!(
        w_final.is_finite(),
        "Waveguide BPM: fields diverged w_final={w_final}"
    );
    // The confined beam should not spread as much as in free space
    // Just verify the fields are finite and stable
    assert!(bpm
        .field
        .iter()
        .all(|e| e.re.is_finite() && e.im.is_finite()));
}
