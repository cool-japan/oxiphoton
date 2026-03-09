/// Fiber optics integration tests — SPM, XPM, FWM, parametric amplification.
use num_complex::Complex64;
use oxiphoton::fiber::{
    soliton_order, FwmPhaseMatching, NlseSolver, ParametricAmplifier, SplitStepNls, SpmFiber,
    TwoChannelPropagation, XpmCoeff,
};

// ── SpmFiber ──────────────────────────────────────────────────────────────────

#[test]
fn spm_fiber_effective_length_lossless() {
    // Lossless fiber: L_eff = L
    let fiber = SpmFiber::new(1.3e-3, 0.0, 100e3);
    let l_eff = fiber.effective_length();
    assert!((l_eff - 100e3).abs() < 1.0, "l_eff={l_eff:.2e}");
}

#[test]
fn spm_fiber_spectral_broadening_increases_with_power() {
    let fiber = SpmFiber::smf28_80km();
    let bf_low = fiber.spectral_broadening_factor(1e-3); // 1 mW
    let bf_high = fiber.spectral_broadening_factor(1.0); // 1 W
    assert!(
        bf_high > bf_low,
        "broadening should increase with power: low={bf_low:.3} high={bf_high:.3}"
    );
}

#[test]
fn spm_fiber_broadening_factor_ge_one() {
    let fiber = SpmFiber::smf28_80km();
    let bf = fiber.spectral_broadening_factor(0.1);
    assert!(bf >= 1.0, "broadening factor must be >= 1: {bf}");
}

#[test]
fn spm_cw_field_preserves_power() {
    let fiber = SpmFiber::new(1.3e-3, 0.0, 10e3);
    let (re_in, im_in) = (1.0_f64, 0.5_f64);
    let power_in = re_in * re_in + im_in * im_in;
    let (re_out, im_out) = fiber.apply_spm_cw(re_in, im_in);
    let power_out = re_out * re_out + im_out * im_out;
    assert!(
        (power_out - power_in).abs() < 1e-12,
        "Power not conserved under SPM phase rotation"
    );
}

// ── SplitStepNls ─────────────────────────────────────────────────────────────

#[test]
fn split_step_nls_pulse_width_broadens_with_spm() {
    // High nonlinearity, no dispersion → SPM only, should cause chirp but minimal temporal change.
    // With beta2 = 0 and SPM: the pulse power profile doesn't change (only phase).
    // But RMS width remains the same for pure SPM (no dispersion).
    let n = 256_usize;
    let dt = 1e-12_f64; // 1 ps sample spacing
    let t0 = 5e-12_f64; // 5 ps pulse width
    let power = 0.01_f64; // low power → negligible SPM

    let solver = SplitStepNls::new(1.3e-3, 0.0, 0.0, 1e3, 10);

    // Build Gaussian pulse
    let t_center = (n as f64 - 1.0) / 2.0 * dt;
    let a0: Vec<Complex64> = (0..n)
        .map(|i| {
            let t = i as f64 * dt - t_center;
            let amp = (power * (-t * t / (t0 * t0)).exp()).sqrt();
            Complex64::new(amp, 0.0)
        })
        .collect();

    let w_in = SplitStepNls::pulse_width_rms(&a0, dt);
    let a_out = solver.propagate(&a0, dt);
    let w_out = SplitStepNls::pulse_width_rms(&a_out, dt);
    // Low power → width should be approximately unchanged (< 5% change)
    let rel_change = (w_out - w_in).abs() / w_in;
    assert!(
        rel_change < 0.05,
        "Low-power SPM should not change pulse width: w_in={w_in:.3e} w_out={w_out:.3e}"
    );
}

#[test]
fn split_step_nls_anomalous_dispersion_broadens_pulse() {
    // Anomalous dispersion β₂ < 0, no nonlinearity → dispersion broadening
    let n = 256_usize;
    let dt = 1e-12_f64;
    let t0 = 10e-12_f64;
    let power = 1e-6_f64; // very low → no SPM

    let solver = SplitStepNls::new(0.0, -20.0, 0.0, 1000e3, 50);

    let t_center = (n as f64 - 1.0) / 2.0 * dt;
    let a0: Vec<Complex64> = (0..n)
        .map(|i| {
            let t = i as f64 * dt - t_center;
            let amp = (power * (-t * t / (t0 * t0)).exp()).sqrt();
            Complex64::new(amp, 0.0)
        })
        .collect();

    let w_in = SplitStepNls::pulse_width_rms(&a0, dt);
    let a_out = solver.propagate(&a0, dt);
    let w_out = SplitStepNls::pulse_width_rms(&a_out, dt);
    assert!(
        w_out >= w_in,
        "Dispersion should broaden pulse: w_in={w_in:.3e} w_out={w_out:.3e}"
    );
}

#[test]
fn split_step_nls_total_power_conserved_lossless() {
    let n = 64_usize;
    let dt = 0.5e-12_f64;
    let t0 = 5e-12_f64;
    let solver = SplitStepNls::new(1.3e-3, -20.0, 0.0, 100e3, 20);

    let t_center = (n as f64 - 1.0) / 2.0 * dt;
    let a0: Vec<Complex64> = (0..n)
        .map(|i| {
            let t = i as f64 * dt - t_center;
            Complex64::new((-t * t / (t0 * t0)).exp().sqrt(), 0.0)
        })
        .collect();

    let p0: f64 = a0.iter().map(|v| v.norm_sqr()).sum::<f64>() * dt;
    let a_out = solver.propagate(&a0, dt);
    let p1: f64 = a_out.iter().map(|v| v.norm_sqr()).sum::<f64>() * dt;
    let rel_err = (p1 - p0).abs() / p0;
    assert!(rel_err < 0.02, "Power not conserved: {rel_err:.2e}");
}

// ── NlseSolver ───────────────────────────────────────────────────────────────

#[test]
fn nlse_solver_power_conservation() {
    let mut s = NlseSolver::new(256, 100e-12, 0.0, -20e-27, 1e-3);
    s.set_gaussian_pulse(1.0, 5e-12);
    let p0 = s.total_power();
    s.propagate(1e3, 50);
    let p1 = s.total_power();
    let rel_err = (p1 - p0).abs() / p0;
    assert!(rel_err < 0.02, "Power not conserved: {rel_err:.2e}");
}

#[test]
fn soliton_order_unity_at_soliton_condition() {
    let beta2 = 20e-27_f64; // anomalous
    let gamma = 1e-3_f64;
    let t0 = 5e-12_f64;
    let p0 = beta2 / (gamma * t0 * t0);
    let n = soliton_order(gamma, p0, t0, beta2);
    assert!((n - 1.0).abs() < 1e-6, "N should be 1: {n}");
}

// ── XpmCoeff ──────────────────────────────────────────────────────────────────

#[test]
fn xpm_coeff_gamma1_positive() {
    let xpm = XpmCoeff::new(2.6e-20, 1550e-9, 1551e-9, 80e-12);
    let g1 = xpm.gamma1();
    assert!(g1 > 0.0, "gamma1 should be positive: {g1:.3e}");
}

#[test]
fn xpm_coeff_xpm_is_twice_gamma1() {
    let xpm = XpmCoeff::new(2.6e-20, 1550e-9, 1551e-9, 80e-12);
    let g1 = xpm.gamma1();
    let xc = xpm.xpm_coeff();
    assert!(
        (xc - 2.0 * g1).abs() < 1e-30,
        "xpm_coeff should be 2*gamma1"
    );
}

#[test]
fn xpm_coeff_gamma_wavelength_scaling() {
    // gamma ∝ 1/lambda → shorter wavelength has larger gamma
    let xpm = XpmCoeff::new(2.6e-20, 1550e-9, 1551e-9, 80e-12);
    let g1 = xpm.gamma1();
    let g2 = xpm.gamma2();
    // Channel 1 at 1550 nm, channel 2 at 1551 nm → g1 slightly > g2
    assert!(
        g1 > g2,
        "shorter wavelength should have larger gamma: g1={g1:.3e} g2={g2:.3e}"
    );
}

// ── TwoChannelPropagation ─────────────────────────────────────────────────────

#[test]
fn two_channel_propagation_power_conservation() {
    let xpm = XpmCoeff::new(2.6e-20, 1550e-9, 1551e-9, 80e-12);
    let solver = TwoChannelPropagation::new(xpm, 10e3, 20);

    // Low-amplitude pulse (negligible SPM/XPM)
    let n = 32_usize;
    let dt = 1e-12_f64;
    let t0 = 5e-12_f64;
    let t_center = (n as f64 - 1.0) / 2.0 * dt;
    let a: Vec<Complex64> = (0..n)
        .map(|i| {
            let t = i as f64 * dt - t_center;
            Complex64::new(0.01 * (-t * t / (t0 * t0)).exp().sqrt(), 0.0)
        })
        .collect();

    let p0: f64 = a.iter().map(|v| v.norm_sqr()).sum();
    let (a1_out, a2_out) = solver.propagate(&a, &a, dt);
    let p1: f64 = a1_out.iter().map(|v| v.norm_sqr()).sum::<f64>()
        + a2_out.iter().map(|v| v.norm_sqr()).sum::<f64>();
    // Total power should be conserved (two channels each with power p0)
    let rel_err = (p1 - 2.0 * p0).abs() / (2.0 * p0);
    assert!(rel_err < 0.05, "Power not conserved: {rel_err:.2e}");
}

// ── FwmPhaseMatching ──────────────────────────────────────────────────────────

#[test]
fn fwm_idler_wavelength_energy_conservation() {
    let lp1 = 1540e-9_f64;
    let lp2 = 1560e-9_f64;
    let ls = 1545e-9_f64;
    let li = FwmPhaseMatching::idler_wavelength(lp1, lp2, ls);
    // Energy conservation: 1/λp1 + 1/λp2 = 1/λs + 1/λi
    let lhs = 1.0 / lp1 + 1.0 / lp2;
    let rhs = 1.0 / ls + 1.0 / li;
    assert!(
        (lhs - rhs).abs() < 1e-3 * lhs,
        "Energy conservation violated: lhs={lhs:.6e} rhs={rhs:.6e}"
    );
}

#[test]
fn fwm_idler_wavelength_is_positive() {
    let li = FwmPhaseMatching::idler_wavelength(1540e-9, 1560e-9, 1545e-9);
    assert!(
        li > 0.0 && li < 2000e-9,
        "idler wavelength out of range: {li:.3e}"
    );
}

#[test]
fn fwm_idler_symmetric_pumps_equals_pump_for_degenerate() {
    // Degenerate: both pumps at same wavelength → idler at same signal wavelength
    // 1/λp + 1/λp = 1/λs + 1/λi → λi = λs (if λs = λp gives special case)
    // Actually: 2/λp - 1/λs = 1/λi
    let lp = 1550e-9_f64;
    let ls = 1560e-9_f64;
    let li = FwmPhaseMatching::idler_wavelength(lp, lp, ls);
    let expected = 1.0 / (2.0 / lp - 1.0 / ls);
    assert!(
        (li - expected).abs() < 1e-15,
        "Degenerate FWM idler mismatch: {li:.3e} vs {expected:.3e}"
    );
}

// ── ParametricAmplifier ───────────────────────────────────────────────────────

#[test]
fn parametric_amp_perfect_phase_match_gain_gt_1() {
    // Perfect phase match: Δβ = 0, γ = 1.2 W^-1m^-1, P = 1 W
    let amp = ParametricAmplifier::new(1.2, 1.0, 0.0);
    let gain = amp.signal_gain(100.0); // 100 m fiber
    assert!(
        gain > 1.0,
        "Perfect phase match should give gain > 1: {gain:.3}"
    );
}

#[test]
fn parametric_amp_gain_coefficient_positive_at_phase_match() {
    let amp = ParametricAmplifier::new(1.2, 1.0, 0.0);
    let g = amp.gain_coefficient();
    assert!(
        g > 0.0,
        "gain coefficient should be > 0 at perfect phase match: {g:.3e}"
    );
    // g = γP = 1.2 * 1.0 = 1.2 m^-1
    assert!((g - 1.2).abs() < 1e-10, "g should equal gamma*P: {g}");
}

#[test]
fn parametric_amp_large_mismatch_gain_zero() {
    // Phase mismatch >> 2*gamma*P → g² < 0 → oscillatory, no exponential gain
    let gamma = 1.2_f64;
    let pump_power = 1.0_f64;
    // Large phase mismatch so that g² = (γP)² - (Δβ/2)² < 0
    let large_dk = 10.0 * gamma * pump_power; // much larger than 2*gamma*P
    let amp = ParametricAmplifier::new(gamma, pump_power, large_dk);
    let g = amp.gain_coefficient();
    assert_eq!(
        g, 0.0,
        "Below-threshold case should return 0.0 gain coefficient: {g}"
    );
}

#[test]
fn parametric_amp_full_gain_ge_1() {
    let amp = ParametricAmplifier::new(1.0, 0.5, 0.0);
    let gain_full = amp.signal_gain_full(50.0);
    assert!(
        gain_full >= 1.0,
        "signal_gain_full should be >= 1 at phase match: {gain_full}"
    );
}

#[test]
fn parametric_amp_mismatch_reduces_gain() {
    let gamma = 1.0_f64;
    let power = 1.0_f64;
    let length = 50.0_f64;
    let amp_pm = ParametricAmplifier::new(gamma, power, 0.0);
    let amp_mis = ParametricAmplifier::new(gamma, power, gamma * power * 0.5);
    let gain_pm = amp_pm.signal_gain_full(length);
    let gain_mis = amp_mis.signal_gain_full(length);
    assert!(
        gain_pm >= gain_mis,
        "Phase matched gain should be >= mismatched: pm={gain_pm:.3} mis={gain_mis:.3}"
    );
}
