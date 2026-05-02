//! Integration tests for the textured-Si grating light-trapping model.
//!
//! Tests cover five physical requirements:
//!   1. Planar limit (n_orders=0) matches the internal zeroth-order-only path
//!   2. A real grating (period=600nm, depth=200nm, dc=0.5) enhances Jsc by > 5% over planar
//!   3. The grating cannot exceed the Lambertian statistical limit (lambertian_fraction ≤ 1)
//!   4. A nearly flat grating (depth=10nm) gives negligible enhancement (< 5%)
//!   5. Jsc converges as n_orders increases (< 2% change from 4→5)

use oxiphoton::solar::evaluate_textured_absorption;
use oxiphoton::solar::SolarSpectrum;

// ─── Common parameters ────────────────────────────────────────────────────────

const PERIOD_M: f64 = 600.0e-9; // 600 nm
const DEPTH_M: f64 = 200.0e-9; // 200 nm
const DUTY_CYCLE: f64 = 0.5;
const ABSORBER_THICK_M: f64 = 200.0e-6; // 200 µm (standard wafer)

// ─── Test 1: Planar limit ─────────────────────────────────────────────────────

/// When `n_orders = 0`, only the m=0 (zeroth-order) term contributes.
/// `jsc_ma_cm2` and `jsc_planar_ma_cm2` must be identical because both are computed
/// from the same m={0} order set.
#[test]
fn planar_limit_matches_zeroth_order_only() {
    let am15g = SolarSpectrum::am15g();
    let result = evaluate_textured_absorption(
        PERIOD_M,
        DEPTH_M,
        DUTY_CYCLE,
        ABSORBER_THICK_M,
        &am15g,
        0, // n_orders = 0 → only m=0
    )
    .expect("evaluate_textured_absorption failed");

    let rel_diff =
        (result.jsc_ma_cm2 - result.jsc_planar_ma_cm2).abs() / result.jsc_planar_ma_cm2.max(1e-12);

    assert!(
        rel_diff < 0.001,
        "n_orders=0: jsc_textured={:.4} vs jsc_planar={:.4}, rel_diff={:.6}",
        result.jsc_ma_cm2,
        result.jsc_planar_ma_cm2,
        rel_diff
    );
}

// ─── Test 2: Grating enhances over planar ────────────────────────────────────

/// A realistic grating (period=600nm, depth=200nm, dc=0.5) must give at least 5%
/// enhancement in Jsc relative to the planar (m=0 only) case.
#[test]
fn textured_enhances_over_planar() {
    let am15g = SolarSpectrum::am15g();
    let result = evaluate_textured_absorption(
        PERIOD_M,
        DEPTH_M,
        DUTY_CYCLE,
        ABSORBER_THICK_M,
        &am15g,
        3, // n_orders = 3: m = ±1, ±2, ±3 included
    )
    .expect("evaluate_textured_absorption failed");

    assert!(
        result.enhancement_factor > 1.05,
        "Expected enhancement_factor > 1.05, got {:.4} (Jsc_textured={:.4}, Jsc_planar={:.4})",
        result.enhancement_factor,
        result.jsc_ma_cm2,
        result.jsc_planar_ma_cm2
    );
}

// ─── Test 3: Lambertian fraction ≤ 1.0 ───────────────────────────────────────

/// A grating cannot exceed the Lambertian (statistical) light-trapping limit.
/// lambertian_fraction = jsc_textured / jsc_lambertian must be ≤ 1.0.
#[test]
fn lambertian_fraction_below_unity() {
    let am15g = SolarSpectrum::am15g();
    let result =
        evaluate_textured_absorption(PERIOD_M, DEPTH_M, DUTY_CYCLE, ABSORBER_THICK_M, &am15g, 3)
            .expect("evaluate_textured_absorption failed");

    assert!(
        result.lambertian_fraction <= 1.0,
        "lambertian_fraction={:.4} must be ≤ 1.0 (Jsc_textured={:.4}, Jsc_lambertian={:.4})",
        result.lambertian_fraction,
        result.jsc_ma_cm2,
        result.jsc_lambertian_ma_cm2
    );
}

// ─── Test 4: Shallow grating recovers planar ─────────────────────────────────

/// A very shallow grating (depth=10nm ≪ λ) imprints a tiny phase (<< 1 rad).
/// Nearly all energy stays in m=0, so enhancement_factor must be < 1.05.
#[test]
fn weak_grating_recovers_planar() {
    let am15g = SolarSpectrum::am15g();
    let result = evaluate_textured_absorption(
        PERIOD_M,
        10.0e-9, // 10 nm depth — essentially flat
        DUTY_CYCLE,
        ABSORBER_THICK_M,
        &am15g,
        3,
    )
    .expect("evaluate_textured_absorption failed");

    assert!(
        result.enhancement_factor < 1.05,
        "Shallow grating (10nm): expected enhancement_factor < 1.05, got {:.6}",
        result.enhancement_factor
    );
}

// ─── Test 5: n_orders convergence ────────────────────────────────────────────

/// As n_orders increases, higher diffraction orders carry less and less energy.
/// The relative change in Jsc from n_orders=4 to n_orders=5 must be < 2%.
#[test]
fn n_orders_convergence() {
    let am15g = SolarSpectrum::am15g();

    let r4 =
        evaluate_textured_absorption(PERIOD_M, DEPTH_M, DUTY_CYCLE, ABSORBER_THICK_M, &am15g, 4)
            .expect("n_orders=4 failed");

    let r5 =
        evaluate_textured_absorption(PERIOD_M, DEPTH_M, DUTY_CYCLE, ABSORBER_THICK_M, &am15g, 5)
            .expect("n_orders=5 failed");

    let jsc4 = r4.jsc_ma_cm2;
    let jsc5 = r5.jsc_ma_cm2;

    let rel_change = (jsc5 - jsc4).abs() / jsc4.max(1e-12);

    assert!(
        rel_change < 0.02,
        "Jsc changed by {:.4} ({:.2}%) from n_orders=4 to n_orders=5; expected < 2%",
        jsc5 - jsc4,
        rel_change * 100.0
    );
}
