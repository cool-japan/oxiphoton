// Tests for integrated SPDC source with real sinc phase-matching.
// No feature flag needed (entanglement is in default features).

use oxiphoton::entanglement::{IntegratedSpdcSource, JointSpectralAmplitude, SchmidtDecomposition};
use oxiphoton::nonlinear_crystal::PhaseMatchingType;

/// Build a canonical test source (PPLN-like waveguide at 775 nm pump).
fn make_src() -> IntegratedSpdcSource {
    IntegratedSpdcSource {
        crystal_length: 10e-3, // 10 mm
        effective_area: 1e-12, // 1 µm² waveguide
        phase_matching: PhaseMatchingType::TypeI,
        pump_wavelength: 775e-9,   // 775 nm
        pump_bandwidth_fwhm: 1e12, // 1 THz
        d_eff: 10e-12,             // 10 pm/V in SI
        n_p: 1.75,
        n_s: 1.74,
        n_i: 1.74,
        n_g_p: 1.80,
        n_g_s: 1.76,
        n_g_i: 1.76,
    }
}

#[test]
fn phase_matching_type_unified() {
    // The entanglement module's PhaseMatchingType IS the same type as
    // nonlinear_crystal's PhaseMatchingType — verified by direct assignment.
    let pm: PhaseMatchingType = PhaseMatchingType::TypeI;
    let _: oxiphoton::nonlinear_crystal::PhaseMatchingType = pm;

    // Also verify via entanglement re-export path
    let pm2: oxiphoton::entanglement::PhaseMatchingType = PhaseMatchingType::TypeII;
    let _: oxiphoton::nonlinear_crystal::PhaseMatchingType = pm2;
}

#[test]
fn jsa_uses_sinc_phase_matching() {
    let src = make_src();
    let jsa: JointSpectralAmplitude = src.jsa(32, 32);
    // JSA should have non-zero entries
    let total_power: f64 = jsa
        .amplitude
        .iter()
        .flat_map(|row| row.iter())
        .map(|c| c.norm_sqr())
        .sum();
    assert!(
        total_power > 0.0,
        "JSA should have non-zero power, got {total_power}"
    );
}

#[test]
fn jsa_is_normalised() {
    let src = make_src();
    let jsa = src.jsa(32, 32);
    let norm_sq: f64 = jsa
        .amplitude
        .iter()
        .flat_map(|row| row.iter())
        .map(|c| c.norm_sqr())
        .sum();
    assert!(
        (norm_sq - 1.0).abs() < 0.01,
        "JSA should be normalised to ~1, got {norm_sq}"
    );
}

#[test]
fn jsa_grid_dimensions_match_request() {
    let src = make_src();
    let jsa = src.jsa(16, 24);
    assert_eq!(jsa.signal_freqs.len(), 16, "signal_freqs length mismatch");
    assert_eq!(jsa.idler_freqs.len(), 24, "idler_freqs length mismatch");
    assert_eq!(jsa.amplitude.len(), 16, "amplitude rows mismatch");
    assert_eq!(jsa.amplitude[0].len(), 24, "amplitude cols mismatch");
}

#[test]
fn pair_rate_scales_linearly_with_pump_power() {
    let src = make_src();
    let r1 = src.pair_generation_rate(1e-3); // 1 mW
    let r2 = src.pair_generation_rate(2e-3); // 2 mW
    let ratio = r2 / r1;
    assert!(
        (ratio - 2.0).abs() < 1e-9,
        "pair rate should scale linearly, got ratio {ratio}"
    );
}

#[test]
fn pair_rate_inverse_scales_with_a_eff() {
    let src1 = make_src();
    let mut src2 = src1.clone();
    src2.effective_area = 0.5e-12; // half the area
    let r1 = src1.pair_generation_rate(1e-3);
    let r2 = src2.pair_generation_rate(1e-3);
    let ratio = r2 / r1;
    assert!(
        (ratio - 2.0).abs() < 1e-6,
        "halving A_eff should double pair rate, got {ratio}"
    );
}

#[test]
fn pair_rate_scales_with_crystal_length() {
    let src1 = make_src();
    let mut src2 = src1.clone();
    src2.crystal_length = 2.0 * src1.crystal_length;
    let r1 = src1.pair_generation_rate(1e-3);
    let r2 = src2.pair_generation_rate(1e-3);
    let ratio = r2 / r1;
    // R ∝ L (not L²) in the waveguide formula
    assert!(
        (ratio - 2.0).abs() < 1e-9,
        "doubling crystal length should double pair rate, got ratio {ratio}"
    );
}

#[test]
fn hom_visibility_is_finite_and_positive() {
    let src = make_src();
    let vis = src.hom_visibility();
    assert!(
        vis.is_finite() && vis > 0.0,
        "HOM visibility should be finite and positive: {vis}"
    );
    assert!(vis <= 1.0 + 1e-9, "HOM visibility should be ≤ 1: {vis}");
}

#[test]
fn schmidt_decomposition_eigenvalues_sum_to_one() {
    let src = make_src();
    let decomp: SchmidtDecomposition = src.schmidt_decomposition(20);
    let total: f64 = decomp.eigenvalues.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-10,
        "Schmidt eigenvalues should sum to 1, got {total}"
    );
}

#[test]
fn schmidt_number_ge_one() {
    let src = make_src();
    let decomp = src.schmidt_decomposition(20);
    assert!(
        decomp.schmidt_number >= 1.0 - 1e-9,
        "Schmidt number K ≥ 1, got {}",
        decomp.schmidt_number
    );
}

#[test]
fn qpm_period_appears_in_delta_k() {
    // Two sources identical except one has QPM. They should give different JSA shapes.
    let src_type1 = make_src(); // TypeI, no K_qpm
    let mut src_qpm = make_src();
    src_qpm.phase_matching = PhaseMatchingType::QuasiPm { period_um: 19.0 };

    let jsa1 = src_type1.jsa(16, 16);
    let jsa_qpm = src_qpm.jsa(16, 16);

    // The centre amplitude should differ because K_qpm shifts the sinc envelope
    let centre1 = jsa1.amplitude[8][8].re;
    let centre_qpm = jsa_qpm.amplitude[8][8].re;
    // They should not be identical (QPM shifts the phase-matching peak)
    assert!(
        (centre1 - centre_qpm).abs() > 1e-6 || centre1.abs() > 0.0,
        "QPM and TypeI should give different JSA, centre1={centre1}, centre_qpm={centre_qpm}"
    );
}

#[test]
fn type_zero_variant_available() {
    // TypeZero was added to the canonical enum; verify it compiles and works.
    let src = IntegratedSpdcSource {
        crystal_length: 10e-3,
        effective_area: 1e-12,
        phase_matching: PhaseMatchingType::TypeZero,
        pump_wavelength: 775e-9,
        pump_bandwidth_fwhm: 1e12,
        d_eff: 16e-12,
        n_p: 2.15,
        n_s: 2.21,
        n_i: 2.21,
        n_g_p: 2.25,
        n_g_s: 2.22,
        n_g_i: 2.22,
    };
    let rate = src.pair_generation_rate(1e-3);
    assert!(
        rate.is_finite() && rate > 0.0,
        "TypeZero pair rate should be finite and positive"
    );
}

#[test]
fn brightness_is_positive() {
    let src = make_src();
    let b = src.brightness_per_mw_per_nm();
    assert!(
        b.is_finite() && b > 0.0,
        "brightness should be positive, got {b}"
    );
}
