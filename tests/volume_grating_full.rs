//! Tests for VolumeGrating: Raman-Nath J_m² order spectrum and full
//! off-Bragg Kogelnik reflection formula.
//!
//! Exercises Block E of the Phase-8 roadmap (volume-grating-full-formulas).

use oxiphoton::diffractive::VolumeGrating;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Build a typical holographic VolumeGrating in SI units.
///
/// λ_B = 633 nm, Λ_g = 500 nm, θ_B ≈ 39.2° (from Bragg condition),
/// d = 100 µm, Δn = 1×10⁻³.
fn make_vg() -> VolumeGrating {
    let lambda_b = 633e-9_f64; // 633 nm Bragg wavelength
    let period = 500e-9_f64; // 500 nm grating period
                             // Bragg condition: λ_B = 2·Λ·cos θ_B  →  cos θ_B = λ_B/(2Λ)
    let bragg_angle = (lambda_b / (2.0 * period)).clamp(-1.0, 1.0).acos();
    VolumeGrating::new(
        1e-3,   // Δn
        100e-6, // d = 100 µm
        lambda_b,
        period,
        bragg_angle,
    )
    .expect("valid VolumeGrating parameters")
}

// ---------------------------------------------------------------------------
// 1. bessel_j_m_matches_known_values
// ---------------------------------------------------------------------------

/// Validate Bessel values via VolumeGrating's public raman_nath_modulation /
/// diffraction_orders path, and directly from the bessel_j_integer function
/// by checking J₀(0)=1, J₁(0)=0, and J₀(2.4048)≈0 (first zero).
#[test]
fn bessel_j_m_matches_known_values() {
    let vg = make_vg();

    // --- J_m(ν) at ν→0 using raman_nath_modulation ≈ 0 ---
    // For a very thin grating (tiny d), ν ≈ 0 → J_0(0)=1, J_1(0)=0.
    let tiny_vg = VolumeGrating::new(1e-6, 1e-9, 633e-9, 500e-9, 0.0).expect("tiny vg");
    // ν = 2π·1e-6·1e-9 / 633e-9 ≈ 9.93e-9 ≈ 0
    let orders_tiny = tiny_vg.diffraction_orders(2, 633e-9);
    let eta0 = orders_tiny
        .iter()
        .find(|(m, _)| *m == 0)
        .map(|(_, e)| *e)
        .unwrap_or(0.0);
    let eta1 = orders_tiny
        .iter()
        .find(|(m, _)| *m == 1)
        .map(|(_, e)| *e)
        .unwrap_or(1.0);
    // J_0(~0)² ≈ 1, J_1(~0)² ≈ 0
    assert!(
        (eta0 - 1.0).abs() < 1e-6,
        "J_0²(0) should be ≈1, got {eta0}"
    );
    assert!(eta1 < 1e-12, "J_1²(0) should be ≈0, got {eta1}");

    // --- J_0(2.4048) ≈ 0 (first zero of J_0) ---
    // ν = 2.4048  →  need d such that 2π·Δn·d/λ = 2.4048
    // d = 2.4048·λ/(2π·Δn) with Δn=1e-3, λ=633e-9
    let lambda = 633e-9_f64;
    let delta_n = 1e-3_f64;
    let x_target = 2.4048_f64;
    let d_target = x_target * lambda / (2.0 * PI * delta_n);
    let vg2 = VolumeGrating::new(delta_n, d_target, lambda, 500e-9, 0.0).expect("vg2");
    let nu_check = vg2.raman_nath_modulation(lambda);
    assert!(
        (nu_check - x_target).abs() < 1e-10,
        "ν should be {x_target}, got {nu_check}"
    );
    // J_0²(2.4048) ≈ 0 within 1e-4
    let orders2 = vg2.diffraction_orders(1, lambda);
    let eta0_zero = orders2
        .iter()
        .find(|(m, _)| *m == 0)
        .map(|(_, e)| *e)
        .unwrap_or(1.0);
    assert!(
        eta0_zero < 1e-4,
        "J_0²(2.4048) should be ≈0 (first zero), got {eta0_zero}"
    );

    let _ = vg;
}

// ---------------------------------------------------------------------------
// 2. raman_nath_orders_sum_to_unity
// ---------------------------------------------------------------------------

/// For ν = 2.0 and |m| ≤ 8, the sum Σ J_m²(ν) should be ≈ 1.0 within 1e-3.
///
/// This follows from the Bessel completeness relation Σ_{m=-∞}^∞ J_m²(x) = 1.
#[test]
fn raman_nath_orders_sum_to_unity() {
    // ν = 2.0 → d = 2.0·λ/(2π·Δn)
    let lambda = 633e-9_f64;
    let delta_n = 1e-3_f64;
    let nu_target = 2.0_f64;
    let d = nu_target * lambda / (2.0 * PI * delta_n);
    let vg = VolumeGrating::new(delta_n, d, lambda, 500e-9, 0.0).expect("vg for sum test");

    let orders = vg.diffraction_orders(8, lambda);
    let total: f64 = orders.iter().map(|(_, eta)| eta).sum();
    assert!(
        (total - 1.0).abs() < 1e-3,
        "Sum of J_m²(2.0) for |m|≤8 should be ≈1.0, got {total}"
    );
}

// ---------------------------------------------------------------------------
// 3. raman_nath_first_order_matches_old_thin_formula
// ---------------------------------------------------------------------------

/// For small ν (ν = 0.1), the exact J₁²(ν) ≈ ν²/4 within 1%.
#[test]
fn raman_nath_first_order_matches_old_thin_formula() {
    let lambda = 633e-9_f64;
    let delta_n = 1e-3_f64;
    let nu_small = 0.1_f64;
    let d = nu_small * lambda / (2.0 * PI * delta_n);
    let vg = VolumeGrating::new(delta_n, d, lambda, 500e-9, 0.0).expect("vg for thin formula test");

    let eta1_exact = vg.first_order_efficiency_thin(lambda);
    let eta1_approx = nu_small * nu_small / 4.0; // ν²/4 approximation

    let rel_err = ((eta1_exact - eta1_approx) / eta1_approx).abs();
    assert!(
        rel_err < 0.01,
        "For ν=0.1, J₁²(ν) ≈ ν²/4: got exact={eta1_exact:.6e}, approx={eta1_approx:.6e}, rel_err={rel_err:.4}"
    );
}

// ---------------------------------------------------------------------------
// 4. kogelnik_at_bragg_matches_tanh_squared
// ---------------------------------------------------------------------------

/// At λ = λ_B (perfect Bragg resonance), R = tanh²(κd) to machine precision.
#[test]
fn kogelnik_at_bragg_matches_tanh_squared() {
    let vg = make_vg();
    let lambda_b = vg.bragg_wavelength_m;
    let kappa = PI * vg.delta_n / lambda_b;
    let tanh2_kd = (kappa * vg.thickness_m).tanh().powi(2);

    let r_bragg = vg.reflection_spectrum(lambda_b);

    assert!(
        (r_bragg - tanh2_kd).abs() < 1e-6,
        "R(λ_B) = tanh²(κd) = {tanh2_kd:.8}, got {r_bragg:.8}, diff = {:.2e}",
        (r_bragg - tanh2_kd).abs()
    );
}

// ---------------------------------------------------------------------------
// 5. kogelnik_off_bragg_dips_below_on_bragg
// ---------------------------------------------------------------------------

/// For any λ ≠ λ_B, R(λ) < R(λ_B).
#[test]
fn kogelnik_off_bragg_dips_below_on_bragg() {
    let vg = make_vg();
    let lambda_b = vg.bragg_wavelength_m;
    let r_at_bragg = vg.reflection_spectrum(lambda_b);

    // Test at several detuned wavelengths
    let offsets_nm: &[f64] = &[-10.0, -5.0, 2.0, 5.0, 10.0, 20.0];
    for &delta_nm in offsets_nm {
        let lam = lambda_b + delta_nm * 1e-9;
        if lam <= 0.0 {
            continue;
        }
        let r_off = vg.reflection_spectrum(lam);
        assert!(
            r_off <= r_at_bragg + 1e-10,
            "R(λ_B + {delta_nm}nm) = {r_off:.6} should be < R(λ_B) = {r_at_bragg:.6}"
        );
    }
}

// ---------------------------------------------------------------------------
// 6. kogelnik_reflection_bounded_by_unity
// ---------------------------------------------------------------------------

/// R(λ) ∈ [0, 1] for all reasonable wavelengths.
#[test]
fn kogelnik_reflection_bounded_by_unity() {
    let vg = make_vg();
    let lambda_b = vg.bragg_wavelength_m;

    // Sweep ±100 nm around Bragg wavelength in 1 nm steps
    let mut i = -100_i32;
    while i <= 100 {
        let lam = lambda_b + i as f64 * 1e-9;
        if lam > 0.0 {
            let r = vg.reflection_spectrum(lam);
            assert!(
                (-1e-10..=1.0 + 1e-10).contains(&r),
                "R must be in [0,1], got R({i}nm offset) = {r}"
            );
        }
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// 7. raman_nath_orders_symmetric_about_m_zero
// ---------------------------------------------------------------------------

/// η_{+m} = η_{-m} exactly for all orders (J_{-m}²(x) = J_m²(x) by parity).
#[test]
fn raman_nath_orders_symmetric_about_m_zero() {
    let lambda = 633e-9_f64;
    let delta_n = 1e-3_f64;
    let nu_target = 3.0_f64;
    let d = nu_target * lambda / (2.0 * PI * delta_n);
    let vg = VolumeGrating::new(delta_n, d, lambda, 500e-9, 0.0).expect("vg symmetry test");

    let orders = vg.diffraction_orders(5, lambda);
    for m in 1..=5_i32 {
        let eta_pos = orders
            .iter()
            .find(|(o, _)| *o == m)
            .map(|(_, e)| *e)
            .unwrap_or(f64::NAN);
        let eta_neg = orders
            .iter()
            .find(|(o, _)| *o == -m)
            .map(|(_, e)| *e)
            .unwrap_or(f64::NAN);
        assert!(
            (eta_pos - eta_neg).abs() < 1e-10,
            "η_{m} = {eta_pos:.12} but η_{neg} = {eta_neg:.12} differ for m={m}",
            neg = -m
        );
    }
}
