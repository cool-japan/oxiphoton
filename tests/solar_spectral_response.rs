//! Integration tests for `solar::spectral_response`.

use oxiphoton::solar::spectrum::SolarSpectrum;
use oxiphoton::solar::{
    compute_spectral_response, AbsorberLayer, AbsorptionMaterial, ArCoatingSpec, BackReflectorSpec,
    SolarCellDesign,
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn si_baseline() -> SolarCellDesign {
    SolarCellDesign {
        absorber: AbsorberLayer {
            refractive_index: 3.5,
            thickness_m: 300e-6,
            bandgap_wavelength_m: 1107e-9,
            material: Some(AbsorptionMaterial::crystalline_silicon()),
            extinction_coeff_const: 0.0,
        },
        ar_coating: None,
        back_reflector: None,
        texturing: None,
        temperature_k: 300.0,
        quantum_yield: 0.0,
    }
}

fn si_optimised() -> SolarCellDesign {
    SolarCellDesign {
        absorber: AbsorberLayer {
            refractive_index: 3.5,
            thickness_m: 300e-6,
            bandgap_wavelength_m: 1107e-9,
            material: Some(AbsorptionMaterial::crystalline_silicon()),
            extinction_coeff_const: 0.0,
        },
        ar_coating: Some(ArCoatingSpec {
            n: 1.9,
            thickness_m: 75e-9,
        }),
        back_reflector: Some(BackReflectorSpec {
            n_real: 0.05,
            n_imag: 8.0,
            thickness_m: 100e-9,
        }),
        texturing: None,
        temperature_k: 300.0,
        quantum_yield: 0.95,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[test]
fn eqe_lambda_returns_correct_grid_length() {
    let cell = si_baseline();
    let am15g = SolarSpectrum::am15g();
    let wavelengths: Vec<f64> = (300..=1100).step_by(10).map(|w| w as f64 * 1e-9).collect();
    let n = wavelengths.len();
    let result = compute_spectral_response(&cell, &wavelengths, &am15g, 5).unwrap();
    assert_eq!(result.eqe.len(), n);
    assert_eq!(result.iqe.len(), n);
    assert_eq!(result.wavelengths_m.len(), n);
}

#[test]
fn eqe_below_unity_at_all_wavelengths() {
    let cell = si_optimised();
    let am15g = SolarSpectrum::am15g();
    let wavelengths: Vec<f64> = (300..=1200).step_by(10).map(|w| w as f64 * 1e-9).collect();
    let result = compute_spectral_response(&cell, &wavelengths, &am15g, 5).unwrap();
    for (i, &eqe) in result.eqe.iter().enumerate() {
        assert!(
            (0.0..=1.001).contains(&eqe),
            "EQE[{i}] = {eqe} out of [0,1]"
        );
    }
}

#[test]
fn iqe_above_eqe() {
    let cell = si_optimised();
    let am15g = SolarSpectrum::am15g();
    let wavelengths: Vec<f64> = (400..=1000).step_by(10).map(|w| w as f64 * 1e-9).collect();
    let result = compute_spectral_response(&cell, &wavelengths, &am15g, 5).unwrap();
    for i in 0..result.eqe.len() {
        if result.eqe[i] > 1e-6 {
            assert!(
                result.iqe[i] >= result.eqe[i] - 1e-6,
                "IQE[{i}]={} < EQE[{i}]={}",
                result.iqe[i],
                result.eqe[i]
            );
        }
    }
}

#[test]
fn recycling_iteration_converges_in_5_iterations() {
    let mut cell = si_optimised();
    cell.quantum_yield = 0.95;
    let am15g = SolarSpectrum::am15g();
    let wavelengths: Vec<f64> = (500..=900).step_by(50).map(|w| w as f64 * 1e-9).collect();

    let r5 = compute_spectral_response(&cell, &wavelengths, &am15g, 5).unwrap();
    let r10 = compute_spectral_response(&cell, &wavelengths, &am15g, 10).unwrap();

    let max_diff: f64 = r5
        .eqe
        .iter()
        .zip(r10.eqe.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < 1e-4,
        "Iteration not converged in 5 steps: max_diff={max_diff:.2e}"
    );
}

#[test]
fn cell_optimization_increases_jsc_over_baseline() {
    let am15g = SolarSpectrum::am15g();
    let wavelengths: Vec<f64> = (300..=1100).step_by(10).map(|w| w as f64 * 1e-9).collect();

    let r_baseline = compute_spectral_response(&si_baseline(), &wavelengths, &am15g, 3).unwrap();
    let r_optimised = compute_spectral_response(&si_optimised(), &wavelengths, &am15g, 3).unwrap();

    assert!(
        r_optimised.jsc_ma_cm2 > r_baseline.jsc_ma_cm2,
        "Optimised J_sc ({:.2}) should exceed baseline ({:.2})",
        r_optimised.jsc_ma_cm2,
        r_baseline.jsc_ma_cm2
    );
}

#[test]
fn recycling_factor_q_zero_no_iteration_change() {
    let mut cell = si_optimised();
    cell.quantum_yield = 0.0;
    let am15g = SolarSpectrum::am15g();
    let wavelengths: Vec<f64> = (400..=900).step_by(50).map(|w| w as f64 * 1e-9).collect();

    let r1 = compute_spectral_response(&cell, &wavelengths, &am15g, 1).unwrap();
    let r5 = compute_spectral_response(&cell, &wavelengths, &am15g, 5).unwrap();

    let max_diff: f64 = r1
        .eqe
        .iter()
        .zip(r5.eqe.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < 1e-12,
        "q=0 should have no iteration effect: max_diff={max_diff:.2e}"
    );
}

#[test]
fn eqe_zero_below_bandgap() {
    let cell = si_baseline(); // Si bandgap ~1107 nm
    let am15g = SolarSpectrum::am15g();
    let wavelengths: Vec<f64> = vec![1150e-9, 1200e-9, 1300e-9]; // sub-bandgap for Si
    let result = compute_spectral_response(&cell, &wavelengths, &am15g, 1).unwrap();
    for (i, &eqe) in result.eqe.iter().enumerate() {
        assert!(
            eqe < 0.01,
            "EQE at sub-bandgap λ[{i}] should be ~0, got {eqe:.4}"
        );
    }
}

#[test]
fn textured_si_with_recycling_jsc_reasonable() {
    // Check that J_sc is physically reasonable (> 5 mA/cm², < 50 mA/cm²) for a Si cell
    let cell = si_optimised();
    let am15g = SolarSpectrum::am15g();
    let wavelengths: Vec<f64> = (300..=1100).step_by(20).map(|w| w as f64 * 1e-9).collect();
    let result = compute_spectral_response(&cell, &wavelengths, &am15g, 5).unwrap();
    assert!(
        result.jsc_ma_cm2 > 5.0,
        "J_sc should be > 5 mA/cm², got {:.2}",
        result.jsc_ma_cm2
    );
    assert!(
        result.jsc_ma_cm2 < 50.0,
        "J_sc should be < 50 mA/cm², got {:.2}",
        result.jsc_ma_cm2
    );
}
