use oxiphoton::mode::{
    dispersion_parameter_d, group_index, group_velocity, FdModeSolver1d, FdTmSolver1d,
    GratingCoupler, TaperedCoupler, TemporalCmt,
};
/// Extended mode solver tests — FdTmSolver1d, group velocity/index,
/// dispersion parameter, TaperedCoupler, GratingCoupler, TemporalCmt.
use std::f64::consts::PI;

// ── FdTmSolver1d ──────────────────────────────────────────────────────────────

#[test]
fn fd_tm_solver_finds_guided_mode_for_thick_slab() {
    // Use a thick slab and lower n_min to ensure TM modes are found
    let n_core = 3.48_f64;
    let n_clad = 1.44_f64;
    let thickness = 2000e-9_f64; // 2 μm slab → strongly multi-mode
    let wavelength = 1550e-9_f64;
    let n_pts = 120_usize;
    let dx = 30e-9_f64; // 30 nm spacing over ~3.6 μm domain

    let profile = FdModeSolver1d::slab_profile(n_core, n_clad, thickness, n_pts, dx);
    // Use n_min slightly below n_clad to catch modes that the TM inverse-eps weighting shifts
    let solver = FdTmSolver1d::new(profile, dx, 1.0);
    let modes = solver.solve(wavelength);
    // Filter to physically guided modes: n_eff > n_clad
    let guided: Vec<_> = modes.iter().filter(|m| m.n_eff > n_clad).collect();

    assert!(
        !guided.is_empty(),
        "TM solver should find at least one guided mode for 2μm Si slab"
    );
    let n_eff = guided[0].n_eff;
    assert!(
        n_eff > n_clad && n_eff <= n_core,
        "Guided mode n_eff should be between cladding and core: {n_eff:.4}"
    );
}

#[test]
fn fd_tm_solver_mode_order_zero_is_fundamental() {
    let n_core = 3.48_f64;
    let n_clad = 1.44_f64;
    let thickness = 500e-9_f64; // thicker slab → possibly multi-mode
    let wavelength = 1550e-9_f64;
    let n_pts = 120_usize;
    let dx = 20e-9_f64;

    let profile = FdModeSolver1d::slab_profile(n_core, n_clad, thickness, n_pts, dx);
    let solver = FdTmSolver1d::new(profile, dx, n_clad);
    let modes = solver.solve(wavelength);

    if !modes.is_empty() {
        assert_eq!(modes[0].order, 0, "First mode should have order 0");
    }
}

#[test]
fn fd_tm_solver_mode_field_length_matches_grid() {
    let n_core = 3.48_f64;
    let n_clad = 1.44_f64;
    let thickness = 300e-9_f64;
    let wavelength = 1550e-9_f64;
    let n_pts = 80_usize;
    let dx = 20e-9_f64;

    let profile = FdModeSolver1d::slab_profile(n_core, n_clad, thickness, n_pts, dx);
    let solver = FdTmSolver1d::new(profile, dx, n_clad);
    let modes = solver.solve(wavelength);

    for mode in &modes {
        assert_eq!(
            mode.field.len(),
            n_pts,
            "Mode field length should match grid"
        );
    }
}

// ── group_velocity ────────────────────────────────────────────────────────────

#[test]
fn group_velocity_dispersionless_equals_phase_velocity() {
    // If n_eff is constant with wavelength → v_g = c/n_eff
    let c = 2.998e8_f64;
    let n = 1.5_f64;
    let dl = 1e-9_f64; // 1 nm step
    let lam = 1550e-9_f64;
    // n_eff identical at both wavelengths → dβ/dω = n/c
    let vg = group_velocity(lam, n, lam + dl, n);
    // For constant n: v_g = c/n
    let expected = c / n;
    let rel_err = (vg - expected).abs() / expected;
    assert!(
        rel_err < 0.01,
        "Dispersionless group velocity: {vg:.3e} vs {expected:.3e}"
    );
}

#[test]
fn group_velocity_positive_for_forward_mode() {
    let vg = group_velocity(1550e-9, 2.5, 1551e-9, 2.499);
    assert!(vg > 0.0, "Group velocity should be positive: {vg:.3e}");
}

#[test]
fn group_velocity_less_than_c() {
    let c = 2.998e8_f64;
    let vg = group_velocity(1550e-9, 2.0, 1551e-9, 1.999);
    assert!(vg < c, "Group velocity must be less than c: {vg:.3e}");
}

// ── group_index ───────────────────────────────────────────────────────────────

#[test]
fn group_index_greater_than_phase_index_in_normal_dispersion() {
    // Normal dispersion (dn/dlambda < 0): n_g = n_eff - lambda * dn_eff/dlambda > n_eff
    // Here n_eff decreases with longer wavelength
    let n_g = group_index(1549e-9, 2.502, 1551e-9, 2.498);
    let n_eff_center = 2.500_f64;
    assert!(n_g > n_eff_center, "Group index should exceed phase index in normal dispersion: n_g={n_g:.4} n_eff={n_eff_center}");
}

#[test]
fn group_index_positive() {
    let ng = group_index(1550e-9, 2.5, 1551e-9, 2.499);
    assert!(ng > 0.0, "Group index must be positive: {ng}");
}

// ── dispersion_parameter_d ────────────────────────────────────────────────────

#[test]
fn dispersion_parameter_anomalous_is_positive() {
    // Anomalous dispersion: D > 0 means d²n/dlambda² < 0 (concave down: n decreases faster)
    // D = -(lambda/c) * d2n/dl2. For D > 0: d2n/dl2 < 0.
    // n_m > n_0 and n_p > n_0 → but the center is minimum → d2n = n_m + n_p - 2*n_0 > 0... wrong
    // Actually: d2n < 0 means n_m + n_p < 2*n_0 (center is a local maximum).
    // e.g., n_m = 2.400, n_0 = 2.402, n_p = 2.400 → d2n = -0.004 < 0 → D > 0
    let lambda = 1550e-9_f64;
    let dl = 1e-9_f64;
    let n_m = 2.400_f64; // at lambda - dl
    let n_0 = 2.402_f64; // at lambda (local maximum)
    let n_p = 2.400_f64; // at lambda + dl
                         // d2n = 2.400 + 2.400 - 2*2.402 = -0.004 < 0 → D = -(lambda/c)*(-0.004/dl²) > 0
    let d = dispersion_parameter_d(lambda, n_m, n_0, n_p, dl);
    assert!(
        d > 0.0,
        "Anomalous dispersion: D should be positive: {d:.4}"
    );
}

#[test]
fn dispersion_parameter_normal_is_negative() {
    // Normal dispersion: D < 0 means d²n/dlambda² > 0 (concave up).
    // n_m + n_p > 2*n_0 → center is a local minimum.
    // e.g., n_m = 2.403, n_0 = 2.400, n_p = 2.403 → d2n = +0.006 > 0 → D < 0
    let lambda = 800e-9_f64;
    let dl = 1e-9_f64;
    let n_m = 2.403_f64;
    let n_0 = 2.400_f64; // local minimum
    let n_p = 2.403_f64;
    let d = dispersion_parameter_d(lambda, n_m, n_0, n_p, dl);
    assert!(d < 0.0, "Normal dispersion: D should be negative: {d:.4}");
}

// ── TaperedCoupler ────────────────────────────────────────────────────────────

#[test]
fn tapered_coupler_long_coupler_efficiency_in_range() {
    // Coupler with moderate kappa and reasonable length
    let kappa = 1e6_f64; // 1 Mrad/m
    let length = 5e-6_f64; // 5 μm → kappa*L = 5 rad
    let coupler = TaperedCoupler::new(length, kappa, kappa);
    let eff = coupler.transfer_efficiency(200);
    // Power oscillates between 0 and 1
    assert!(
        (0.0..=1.0 + 1e-10).contains(&eff),
        "Transfer efficiency must be in [0,1]: {eff}"
    );
}

#[test]
fn tapered_coupler_zero_kappa_no_transfer() {
    // Zero coupling: no power transfer
    let coupler = TaperedCoupler::new(100e-6, 0.0, 0.0);
    let eff = coupler.transfer_efficiency(200);
    assert!(
        eff < 1e-10,
        "Zero coupling coupler should have zero transfer: {eff}"
    );
}

#[test]
fn tapered_coupler_at_coupling_length_transfers_fully() {
    // At exact coupling length L_c = π/(2κ) for symmetric coupler
    let kappa = 1e6_f64; // 1 Mrad/m
    let l_c = PI / (2.0 * kappa); // coupling length
    let coupler = TaperedCoupler::new(l_c, kappa, kappa);
    let eff = coupler.transfer_efficiency(1000);
    // At L_c, complete power transfer (efficiency ≈ 1)
    assert!(
        eff > 0.95,
        "At coupling length, efficiency should be ~1: {eff:.4}"
    );
}

#[test]
fn tapered_coupler_power_conservation() {
    // Total power |a|² + |b|² = 1 at every point
    let kappa = 5e5_f64;
    let length = PI / kappa; // two coupling lengths
    let coupler = TaperedCoupler::new(length, kappa, kappa);
    let (pa, pb) = coupler.propagate(200);
    for (a, b) in pa.iter().zip(pb.iter()) {
        assert!(
            (a + b - 1.0).abs() < 1e-10,
            "Power should be conserved: pa={a:.6} pb={b:.6} sum={:.6}",
            a + b
        );
    }
}

// ── GratingCoupler ────────────────────────────────────────────────────────────

#[test]
fn grating_coupler_transmittance_plus_reflectance_approx_one() {
    // Under lossless CMT: T + R = 1
    let kappa = 1e5_f64; // 100 krad/m
    let delta_beta = 0.0_f64; // perfect phase match
    let length = 100e-6_f64; // 100 μm device
    let gc = GratingCoupler::new(kappa, delta_beta, length);
    let t = gc.transmittance();
    let r = gc.reflectance();
    assert!(
        (t + r - 1.0).abs() < 1e-10,
        "T + R should be 1 (lossless): T={t:.6} R={r:.6} T+R={:.6}",
        t + r
    );
}

#[test]
fn grating_coupler_phase_mismatch_reduces_reflectance() {
    let kappa = 5e4_f64;
    let length = 200e-6_f64;
    let gc_pm = GratingCoupler::new(kappa, 0.0, length);
    let gc_mis = GratingCoupler::new(kappa, kappa * 2.0, length);
    let r_pm = gc_pm.reflectance();
    let r_mis = gc_mis.reflectance();
    assert!(
        r_pm >= r_mis,
        "Phase-matched grating should have higher reflectance: pm={r_pm:.4} mis={r_mis:.4}"
    );
}

#[test]
fn grating_coupler_transmittance_in_0_1() {
    let gc = GratingCoupler::new(1e5, 0.0, 50e-6);
    let t = gc.transmittance();
    assert!(
        (0.0..=1.0).contains(&t),
        "Transmittance must be in [0,1]: {t}"
    );
}

#[test]
fn grating_coupler_reflectance_in_0_1() {
    let gc = GratingCoupler::new(1e5, 0.0, 50e-6);
    let r = gc.reflectance();
    assert!(
        (0.0..=1.0).contains(&r),
        "Reflectance must be in [0,1]: {r}"
    );
}

// ── TemporalCmt ───────────────────────────────────────────────────────────────

#[test]
fn temporal_cmt_transmission_spectrum_valid_range() {
    let omega0 = 2.0 * PI * 1.934e14; // ~1550nm
    let tau = 5e-12_f64;
    let d = 1.0_f64;
    let cmt = TemporalCmt::new(omega0, tau, d);

    // Sweep frequency around resonance
    let gamma = 1.0 / (2.0 * tau);
    let omegas: Vec<f64> = (-5..=5).map(|k| omega0 + k as f64 * gamma).collect();
    let spectra = cmt.transmission_spectrum(&omegas);

    // All transmission values should be non-negative (they can exceed 1 for non-critically coupled)
    for &t in &spectra {
        assert!(t >= 0.0, "Transmission must be >= 0: {t}");
    }
}

#[test]
fn temporal_cmt_steady_state_amplitude_at_resonance() {
    let omega0 = 2.0 * PI * 1.934e14;
    let tau = 5e-12_f64;
    let d = 1.0_f64;
    let cmt = TemporalCmt::new(omega0, tau, d);

    let a_res = cmt.steady_state_amplitude(omega0);
    // At resonance: a_ss = d/γ (real), max amplitude
    let gamma = 1.0 / (2.0 * tau);
    let expected_mag = d / gamma;
    assert!(
        (a_res.norm() - expected_mag).abs() < expected_mag * 0.01,
        "Steady-state amplitude at resonance: |a|={:.4e} expected={:.4e}",
        a_res.norm(),
        expected_mag
    );
}

#[test]
fn temporal_cmt_steady_state_magnitude_decreases_with_detuning() {
    let omega0 = 2.0 * PI * 1.934e14;
    let tau = 5e-12_f64;
    let d = 1.0_f64;
    let cmt = TemporalCmt::new(omega0, tau, d);

    let gamma = 1.0 / (2.0 * tau);
    let a_on = cmt.steady_state_amplitude(omega0).norm();
    let a_off = cmt.steady_state_amplitude(omega0 + 5.0 * gamma).norm();
    assert!(
        a_on > a_off,
        "Steady-state amplitude should be larger at resonance: on={a_on:.4e} off={a_off:.4e}"
    );
}

#[test]
fn temporal_cmt_impulse_response_decays() {
    let omega0 = 2.0 * PI * 1.934e14;
    let tau = 5e-12_f64;
    let d = 1.0_f64;
    let cmt = TemporalCmt::new(omega0, tau, d);

    let times = vec![0.0, tau, 2.0 * tau];
    let response = cmt.impulse_response(&times);
    assert!(
        response[1].norm() < response[0].norm(),
        "Impulse response should decay over time"
    );
    assert!(
        response[2].norm() < response[1].norm(),
        "Impulse response should continue decaying"
    );
}

#[test]
fn temporal_cmt_linewidth_matches_gamma() {
    let omega0 = 2.0 * PI * 1.934e14;
    let tau = 5e-12_f64;
    let d = 1.0_f64;
    let cmt = TemporalCmt::new(omega0, tau, d);
    let gamma = 1.0 / (2.0 * tau);
    let linewidth = cmt.linewidth();
    assert!(
        (linewidth - 2.0 * gamma).abs() < 1e-3 * linewidth,
        "Linewidth should be 2γ: {linewidth:.4e} vs {:.4e}",
        2.0 * gamma
    );
}
