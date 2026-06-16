//! Integration tests for the adaptive RK4IP GNLSE integrator.
//!
//! Tests verify:
//! 1. Photon-number conservation under lossless, Raman-free propagation.
//! 2. Adaptive RK4IP converges to the fixed-step symmetric SSF result.
//! 3. Adaptive step count increases with nonlinearity (solver adapts).
//! 4. RK4IP is more accurate than symmetric SSF at equal step count.

use num_complex::Complex64;
use oxiphoton::fiber::supercontinuum::GnlseSolver;
use std::f64::consts::PI;

/// Build a Gaussian pulse field on an n-point time grid with step dt.
/// Pulse 1/e half-width = t0, centred at the grid midpoint.
fn gaussian_pulse(n: usize, dt: f64, t0: f64, peak_amplitude: f64) -> Vec<Complex64> {
    (0..n)
        .map(|i| {
            let t = (i as f64 - n as f64 / 2.0) * dt;
            Complex64::new(peak_amplitude * (-0.5 * (t / t0).powi(2)).exp(), 0.0)
        })
        .collect()
}

/// L2 norm squared of a complex field (unnormalised).
fn l2_norm_sq(v: &[Complex64]) -> f64 {
    v.iter().map(|a| a.norm_sqr()).sum()
}

/// Relative L2 difference between two fields.
fn rel_l2_diff(a: &[Complex64], b: &[Complex64]) -> f64 {
    let diff_sq: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).norm_sqr())
        .sum();
    let norm_b_sq: f64 = l2_norm_sq(b);
    if norm_b_sq < 1e-300 {
        return 0.0;
    }
    (diff_sq / norm_b_sq).sqrt()
}

// ---------------------------------------------------------------------------
// Test 1 – Photon-number conservation under lossless, Raman-free propagation
// ---------------------------------------------------------------------------
/// For α = 0, fR = 0 (Kerr-only, no loss), the total energy ∫|A(t)|² dt
/// is conserved under the GNLSE since the nonlinear phase is pure imaginary
/// and dispersion is a unitary operator.  We verify this holds to within
/// ±0.1% relative for tol = 1e-6.
#[test]
fn photon_number_conservation_rk4ip() {
    let fiber_length = 0.05; // 5 cm — short enough for fast test
    let mut solver = GnlseSolver::new_silica(
        fiber_length,
        50e-3,   // γ = 50 /W/m (high NL for visible effect)
        -20e-27, // β₂ < 0
        0.1e-39, // β₃
    );
    solver.alpha = 0.0;
    solver.raman_fraction = 0.0;

    let n = solver.n_time_points;
    let dt = solver.dt;
    let t0 = 100e-15; // 100 fs pulse
    let field = gaussian_pulse(n, dt, t0, 10.0_f64.sqrt()); // √10 W amplitude → 10 W peak

    let energy_in: f64 = field.iter().map(|a| a.norm_sqr()).sum::<f64>() * dt;

    let (output, _steps) = solver.propagate_adaptive(&field, 2.0 * PI * 2.998e8 / 1550e-9, 1e-6);

    let energy_out: f64 = output.iter().map(|a| a.norm_sqr()).sum::<f64>() * dt;
    let rel_err = (energy_out - energy_in).abs() / energy_in;

    assert!(
        rel_err < 1e-3,
        "Energy not conserved (lossless Kerr): rel_err = {rel_err:.3e} (must be < 0.1%)"
    );
}

// ---------------------------------------------------------------------------
// Test 2 – Adaptive RK4IP agrees with fixed-step SSF for a short fiber
// ---------------------------------------------------------------------------
/// For a short propagation (0.01 m) with moderate nonlinearity, the output
/// spectra from `propagate_adaptive` (tol = 1e-7) and `propagate` (fixed-step
/// SSF) must agree within 5% RMS relative spectral error.
#[test]
fn adaptive_converges_to_fixed_step() {
    let fiber_length = 0.01;
    let solver = GnlseSolver::new_silica(
        fiber_length,
        10e-3,   // γ
        -20e-27, // β₂
        0.0,     // β₃
    );

    let n = solver.n_time_points;
    let dt = solver.dt;
    let omega0 = 2.0 * PI * 2.998e8 / 1550e-9;
    let t0 = 200e-15;
    let field = gaussian_pulse(n, dt, t0, 5.0_f64.sqrt()); // ~5 W peak

    // Fixed-step symmetric SSF
    let out_ssf = solver.propagate(&field, omega0);

    // Adaptive RK4IP with tight tolerance
    let (out_rk4ip, _steps) = solver.propagate_adaptive(&field, omega0, 1e-7);

    // Compute relative L2 difference on output fields
    let rms_err = rel_l2_diff(&out_rk4ip, &out_ssf);
    assert!(
        rms_err < 0.05,
        "Adaptive RK4IP diverged from fixed-step SSF: rms_err = {rms_err:.3e} (must be < 5%)"
    );
}

// ---------------------------------------------------------------------------
// Test 3 – Adaptive step count increases with nonlinearity
// ---------------------------------------------------------------------------
/// A stronger nonlinear coefficient γ requires the adaptive solver to take
/// more steps (shorter h) to maintain the same tolerance.  We compare two
/// solvers — one with low γ and one with high γ — and check that the
/// high-γ case requires strictly more steps.
#[test]
fn step_count_increases_with_nonlinearity() {
    let fiber_length = 0.05;
    let tol = 1e-5;
    let omega0 = 2.0 * PI * 2.998e8 / 1550e-9;

    let make_field = |n: usize, dt: f64| {
        let t0 = 200e-15;
        gaussian_pulse(n, dt, t0, 3.0_f64.sqrt())
    };

    // Low nonlinearity
    let solver_low = GnlseSolver::new_silica(fiber_length, 1e-3, -20e-27, 0.0);
    let field_low = make_field(solver_low.n_time_points, solver_low.dt);
    let (_, steps_low) = solver_low.propagate_adaptive(&field_low, omega0, tol);

    // High nonlinearity (100× larger γ)
    let solver_high = GnlseSolver::new_silica(fiber_length, 100e-3, -20e-27, 0.0);
    let field_high = make_field(solver_high.n_time_points, solver_high.dt);
    let (_, steps_high) = solver_high.propagate_adaptive(&field_high, omega0, tol);

    assert!(
        steps_high > steps_low,
        "High-γ solver should take more steps than low-γ solver: \
         steps_low = {steps_low}, steps_high = {steps_high}"
    );
}

// ---------------------------------------------------------------------------
// Test 4 – RK4IP is more accurate than symmetric SSF at equal step count
// ---------------------------------------------------------------------------
/// We use a small number of fixed steps (same step size for both methods)
/// and compare both results against a high-accuracy reference from the
/// adaptive solver.  RK4IP should achieve smaller error than SSF.
///
/// We use a highly nonlinear, short-fiber scenario so that truncation error
/// differences are detectable.
#[test]
fn rk4ip_more_accurate_than_ssf() {
    let fiber_length = 0.02;
    let omega0 = 2.0 * PI * 2.998e8 / 1550e-9;
    let tol_ref = 1e-8;

    let n_steps_coarse = 5; // few steps → large local error → easy to see difference

    let mut solver = GnlseSolver::new_silica(fiber_length, 50e-3, -20e-27, 0.0);
    solver.alpha = 0.0;
    solver.raman_fraction = 0.0;
    // Override dz to force a specific step count
    solver.dz = fiber_length / n_steps_coarse as f64;

    let n = solver.n_time_points;
    let dt = solver.dt;
    let t0 = 150e-15;
    let field = gaussian_pulse(n, dt, t0, 8.0_f64.sqrt());

    // ── Reference: adaptive RK4IP at high accuracy ─────────────────────────
    let mut solver_ref = solver.clone();
    solver_ref.dz = fiber_length / 1000.0;
    let (ref_field, _) = solver_ref.propagate_adaptive(&field, omega0, tol_ref);

    // ── Coarse SSF (n_steps_coarse fixed steps) ────────────────────────────
    let ssf_out = solver.propagate(&field, omega0);

    // ── Coarse RK4IP (same step size) ──────────────────────────────────────
    let h = fiber_length / n_steps_coarse as f64;
    let mut rk4ip_field = field.clone();
    for _ in 0..n_steps_coarse {
        rk4ip_field = solver.rk4ip_step(&rk4ip_field, omega0, h);
    }

    let err_ssf = rel_l2_diff(&ssf_out, &ref_field);
    let err_rk4ip = rel_l2_diff(&rk4ip_field, &ref_field);

    assert!(
        err_rk4ip < err_ssf,
        "RK4IP should be more accurate than SSF at equal step size: \
         err_ssf = {err_ssf:.3e}, err_rk4ip = {err_rk4ip:.3e}"
    );
}
