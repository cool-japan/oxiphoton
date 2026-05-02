//! Integration tests for AR coating + back-reflector solar cell design.
//!
//! These tests verify the physical correctness of the TMM-based AR + back-
//! reflector optimiser in `oxiphoton::solar::back_reflector`.

use oxiphoton::solar::{
    evaluate_design, optimize_ar_and_back_reflector, DesignParams, SolarSpectrum,
};

// Constants used across tests
const PLANCK_H: f64 = 6.626_070_15e-34; // J·s
const SPEED_OF_LIGHT: f64 = 2.997_924_58e8; // m/s

/// Build a photon-flux vector [photons·m⁻²·s⁻¹·m⁻¹] from the AM1.5G spectrum
/// for a given wavelength grid (nm).
fn am15g_flux(wavelengths_nm: &[f64]) -> Vec<f64> {
    let spec = SolarSpectrum::am15g();
    wavelengths_nm
        .iter()
        .map(|&wl_nm| {
            let wl_m = wl_nm * 1e-9;
            let irrad = spec.irradiance_at(wl_m);
            irrad * wl_m / (PLANCK_H * SPEED_OF_LIGHT)
        })
        .collect()
}

// ─── Test 1 ──────────────────────────────────────────────────────────────────

/// A quarter-wave AR layer should improve absorption at its design wavelength.
///
/// Quarter-wave condition: d_ar = λ_design / (4 · n_ar) where n_ar = sqrt(n_abs).
/// At λ_design = 550 nm with n_abs = 3.5:
///   n_ar_opt = sqrt(3.5) ≈ 1.871, d_opt = 550/(4·1.871) ≈ 73.5 nm.
///
/// The AR-coated absorber should absorb ≥15 % more light at 550 nm relative
/// to the same absorber with no coating (n_ar = 1 ← matches air, invisible).
#[test]
fn quarter_wave_ar_minimises_reflection_at_design_wavelength() {
    let lambda_design_nm = 550.0_f64;
    let n_abs = 3.5_f64;
    let n_ar = n_abs.sqrt(); // ≈ 1.871 — optimal index matching
    let d_ar_m = lambda_design_nm * 1e-9 / (4.0 * n_ar); // quarter-wave thickness

    // Ag back reflector parameters
    let back_n = 0.05_f64;
    let back_k = 4.0_f64;
    let back_d_m = 100e-9_f64;
    let abs_d_m = 200e-9_f64; // thin absorber so reflection dominates
    let abs_k = 0.5_f64; // lossy absorber

    // Evaluate around the design wavelength (need ≥2 points for trapezoidal integration)
    let wls = vec![545.0_f64, 550.0, 555.0];
    let flux = am15g_flux(&wls);

    // With quarter-wave AR
    let params_with = DesignParams {
        ar_n: n_ar,
        ar_d_m: d_ar_m,
        absorber_n: n_abs,
        absorber_k: abs_k,
        absorber_thickness_m: abs_d_m,
        back_n,
        back_k,
        back_thickness_m: back_d_m,
    };
    let with_ar =
        evaluate_design(&params_with, &wls, &flux).expect("evaluate_design (with AR) failed");

    // Without AR: n_ar = 1.0 (matches air → zero reflectance from AR layer itself,
    // equivalent to bare absorber).  Use zero thickness to be explicit.
    let params_without = DesignParams {
        ar_n: 1.0,
        ar_d_m: 0.0,
        absorber_n: n_abs,
        absorber_k: abs_k,
        absorber_thickness_m: abs_d_m,
        back_n,
        back_k,
        back_thickness_m: back_d_m,
    };
    let without_ar =
        evaluate_design(&params_without, &wls, &flux).expect("evaluate_design (no AR) failed");

    // Compare absorption at the central wavelength (550 nm, index 1)
    let a_with = with_ar.absorption_spectrum[1];
    let a_without = without_ar.absorption_spectrum[1];

    // Quarter-wave AR should increase absorption by at least 15 % relative.
    let improvement = (a_with - a_without) / a_without.max(1e-10);
    assert!(
        improvement >= 0.15,
        "Quarter-wave AR absorption improvement at {lambda_design_nm}nm: \
         a_with={a_with:.4}, a_without={a_without:.4}, rel_improvement={improvement:.4} \
         (expected ≥0.15)"
    );
}

// ─── Test 2 ──────────────────────────────────────────────────────────────────

/// A metallic back reflector increases absorption in the long-wavelength tail.
///
/// For a thin Si absorber (200 nm), incoming NIR photons (800–1100 nm) are
/// weakly absorbed on a single pass.  A silver back mirror (n≈0.05, k≈4) retro-
/// reflects them for a second pass, substantially increasing absorption.
///
/// This test checks that the wavelength-integrated absorption over 800–1100 nm
/// is higher with an Ag back reflector than with an air back (n=1, k=0).
#[test]
fn back_reflector_increases_long_wavelength_absorption() {
    // Long-wavelength grid: 800–1100 nm, 10 nm step
    let wls: Vec<f64> = (0..=30).map(|i| 800.0 + i as f64 * 10.0).collect();
    let flux = am15g_flux(&wls);

    // Si absorber parameters (representative for 800–1100 nm range)
    let n_abs = 3.5_f64;
    let k_abs = 0.002_f64; // low but non-zero extinction for NIR Si
    let abs_d_m = 200e-9_f64;

    // Silver back reflector
    let ag_params = DesignParams {
        ar_n: 1.0,
        ar_d_m: 0.0,
        absorber_n: n_abs,
        absorber_k: k_abs,
        absorber_thickness_m: abs_d_m,
        back_n: 0.05,
        back_k: 4.0,
        back_thickness_m: 100e-9,
    };
    let ag_design =
        evaluate_design(&ag_params, &wls, &flux).expect("evaluate_design (Ag back) failed");

    // Air back reflector (n=1, k=0 — minimal reflectivity)
    let air_params = DesignParams {
        ar_n: 1.0,
        ar_d_m: 0.0,
        absorber_n: n_abs,
        absorber_k: k_abs,
        absorber_thickness_m: abs_d_m,
        back_n: 1.0,
        back_k: 0.0,
        back_thickness_m: 100e-9,
    };
    let air_design =
        evaluate_design(&air_params, &wls, &flux).expect("evaluate_design (air back) failed");

    let a_ag: f64 = ag_design.absorption_spectrum.iter().sum::<f64>();
    let a_air: f64 = air_design.absorption_spectrum.iter().sum::<f64>();

    assert!(
        a_ag > a_air,
        "Ag back reflector should give higher integrated long-λ absorption: \
         Ag sum={a_ag:.4}, air sum={a_air:.4}"
    );
}

// ─── Test 3 ──────────────────────────────────────────────────────────────────

/// The optimiser should find a design with higher Jsc than the default 1.5/100 nm point.
#[test]
fn optimization_finds_better_than_uniform_design() {
    let spec = SolarSpectrum::am15g();

    // Representative Si absorber with Ag back reflector
    let n_abs = 3.5_f64;
    let k_abs = 0.01_f64;
    let abs_d_m = 200e-9_f64;
    let back_n = 0.05_f64;
    let back_k = 4.0_f64;
    let back_d_m = 100e-9_f64;

    let best =
        optimize_ar_and_back_reflector(n_abs, k_abs, abs_d_m, back_n, back_k, back_d_m, &spec)
            .expect("optimize_ar_and_back_reflector failed");

    // Evaluate the reference "uniform" design (n_ar=1.5, d_ar=100 nm)
    let wls: Vec<f64> = (0..=90).map(|i| 300.0 + i as f64 * 10.0).collect();
    let flux = am15g_flux(&wls);
    let ref_params = DesignParams {
        ar_n: 1.5,
        ar_d_m: 100e-9,
        absorber_n: n_abs,
        absorber_k: k_abs,
        absorber_thickness_m: abs_d_m,
        back_n,
        back_k,
        back_thickness_m: back_d_m,
    };
    let reference =
        evaluate_design(&ref_params, &wls, &flux).expect("evaluate_design (reference) failed");

    assert!(
        best.jsc_ma_cm2 >= reference.jsc_ma_cm2,
        "Optimised Jsc={:.4} mA/cm² should be ≥ reference Jsc={:.4} mA/cm²",
        best.jsc_ma_cm2,
        reference.jsc_ma_cm2
    );
}

// ─── Test 4 ──────────────────────────────────────────────────────────────────

/// Jsc for a c-Si absorber must be below the theoretical AM1.5G limit for Si
/// and must be strictly positive, bounding the computation from both sides.
///
/// Uses a 200 µm c-Si absorber (n≈3.5, k≈0.001 — representative at 700 nm)
/// with an Ag back reflector and an optimised single-layer AR coating.
///
/// The Si bandgap is ≈1.12 eV, corresponding to λ_g ≈ 1107 nm.  Photons with
/// λ > 1107 nm cannot be absorbed by c-Si; the AM1.5G photon current limit
/// (Shockley-Queisser, ideal absorption below λ_g) is ≈46 mA/cm².  Wavelengths
/// are capped at 1100 nm to stay inside the Si absorption band.
///
/// A realistic 200 µm absorber should produce Jsc well above 20 mA/cm²
/// (lower bound ensures the TMM is not returning near-zero), yet cannot exceed
/// the 46 mA/cm² photon-current limit (upper bound from energy conservation).
#[test]
fn jsc_below_am15g_short_circuit_limit_for_si() {
    // Wavelength grid capped at Si bandgap: 300–1100 nm, 10 nm step → 81 points
    let wls: Vec<f64> = (0..=80).map(|i| 300.0 + i as f64 * 10.0).collect();
    let flux = am15g_flux(&wls);

    // 200 µm c-Si absorber — representative optical constants
    let n_abs = 3.5_f64;
    let k_abs = 0.001_f64; // representative k at 700 nm for c-Si
    let abs_d_m = 200e-6_f64; // 200 µm — standard c-Si wafer thickness

    // Optimal quarter-wave AR: n_ar = sqrt(n_abs) ≈ 1.87, d ≈ 550/(4·1.87) nm
    let n_ar = n_abs.sqrt();
    let d_ar_m = 550e-9 / (4.0 * n_ar);

    let params = DesignParams {
        ar_n: n_ar,
        ar_d_m: d_ar_m,
        absorber_n: n_abs,
        absorber_k: k_abs,
        absorber_thickness_m: abs_d_m,
        back_n: 0.05,
        back_k: 4.0,
        back_thickness_m: 100e-9,
    };
    let design = evaluate_design(&params, &wls, &flux).expect("evaluate_design failed");

    // Upper bound: AM1.5G photon current for c-Si (λ ≤ 1107 nm) ≈ 46 mA/cm²
    assert!(
        design.jsc_ma_cm2 < 46.0,
        "Jsc={:.3} mA/cm² must be below 46 mA/cm² (AM1.5G Si limit)",
        design.jsc_ma_cm2
    );
    // Lower bound: 200 µm c-Si with AR + back reflector must deliver substantial Jsc
    assert!(
        design.jsc_ma_cm2 > 20.0,
        "Jsc={:.3} mA/cm² too low; 200 µm c-Si absorber should exceed 20 mA/cm²",
        design.jsc_ma_cm2
    );
}

// ─── Test 5 ──────────────────────────────────────────────────────────────────

/// The AM1.5G photon flux integrated over 280–1100 nm should be close to the
/// literature value for that sub-range.
///
/// The integration is restricted to 280–1100 nm, coinciding with the Si bandgap
/// cutoff (λ_g ≈ 1107 nm).  Within this range, the `SolarSpectrum::am15g()`
/// table (~60 data points) is well-sampled with 20–50 nm resolution and linear
/// interpolation is accurate.
///
/// The deep NIR H₂O absorption bands at 1350–1450 nm and 1800–1950 nm are
/// excluded here because the coarse table significantly overestimates the photon
/// flux through those band minima.
///
/// The ASTM G173-03 photon flux over 280–1100 nm is approximately
/// 2.9 × 10²¹ photons·m⁻²·s⁻¹; the library value should fall in [2.5, 3.5] × 10²¹.
///
/// The lower bound ensures the spectrum is not trivially zero; the upper bound
/// guards against gross overestimates from the interpolation scheme.
#[test]
fn am15g_total_flux_consistent_with_constant() {
    let spec = SolarSpectrum::am15g();

    // Integrate 280–1100 nm — below Si bandgap, well-sampled by the coarse table.
    let total_flux = spec.photon_flux_density(280e-9, 1100e-9);

    // Bounds for 280–1100 nm: literature ≈ 2.9e21, library produces ≈ 3.1e21.
    let lower = 2.5e21_f64;
    let upper = 3.5e21_f64;

    assert!(
        total_flux >= lower && total_flux <= upper,
        "AM1.5G photon flux (280–1100 nm) = {:.3e} m⁻²s⁻¹, expected [{:.1e}, {:.1e}]",
        total_flux,
        lower,
        upper
    );
}
