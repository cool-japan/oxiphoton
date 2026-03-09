use std::f64::consts::PI;

use oxiphoton::devices::coupler::{
    half_coupler, DirectionalCoupler, GratingCoupler, Mmi1x2, MmiCoupler,
};
use oxiphoton::devices::resonator::RingResonator;
use oxiphoton::devices::waveguide::{SlabWaveguideDevice, StripWaveguide};

// ──────────────────────── Slab Waveguide ────────────────────────

#[test]
fn slab_device_te_modes() {
    let slab = SlabWaveguideDevice::new(3.476, 1.444, 500e-9);
    let modes = slab.te_modes(1550e-9);
    assert!(!modes.is_empty(), "Should find TE modes");
}

#[test]
fn slab_device_v_number() {
    let slab = SlabWaveguideDevice::new(3.476, 1.444, 500e-9);
    let v = slab.v_number(1550e-9);
    assert!(v > 0.0, "V number must be positive");
    // Si slab 500nm at 1550nm: V ≈ 2.8
    assert!(v > 1.0 && v < 10.0, "V={v:.3}");
}

#[test]
fn slab_device_confinement() {
    let slab = SlabWaveguideDevice::new(3.476, 1.444, 500e-9);
    let gamma = slab.confinement_factor_te0(1550e-9).unwrap();
    assert!(gamma > 0.0 && gamma < 1.0);
}

// ──────────────────────── Strip Waveguide ────────────────────────

#[test]
fn strip_waveguide_si_1550_guided() {
    let wg = StripWaveguide::new(3.476, 1.444, 500e-9, 220e-9);
    let n_eff = wg.n_eff_te(1550e-9).expect("Si strip should have TE mode");
    assert!(n_eff > 1.444 && n_eff < 3.476, "n_eff={n_eff:.4}");
}

#[test]
fn strip_waveguide_te_vs_tm() {
    let wg = StripWaveguide::new(3.476, 1.444, 500e-9, 220e-9);
    let n_te = wg.n_eff_te(1550e-9);
    let n_tm = wg.n_eff_tm(1550e-9);
    // Both should find guided modes for Si strip
    assert!(
        n_te.is_some() || n_tm.is_some(),
        "Should find at least one polarization"
    );
}

// ──────────────────────── Ring Resonator ────────────────────────

#[test]
fn ring_resonator_fsr_formula() {
    // FSR = λ² / (n_g · 2πR)
    let r = 5e-6; // 5μm radius
    let n_g = 4.2;
    let lambda = 1550e-9;
    let ring = RingResonator::new(r, 2.4, n_g, 0.05, 0.0);
    let fsr = ring.fsr(lambda);
    let expected = lambda * lambda / (n_g * 2.0 * PI * r);
    let err_nm = (fsr - expected).abs() * 1e9;
    assert!(err_nm < 0.01, "FSR error={err_nm:.4} nm");
}

#[test]
fn ring_resonator_fsr_within_01nm() {
    // Stricter test: within ±0.1nm of analytical FSR
    let r = 5e-6;
    let n_g = 4.2;
    let lambda = 1550e-9;
    let ring = RingResonator::new(r, 2.4, n_g, 0.1, 0.0);
    let fsr_nm = ring.fsr(lambda) * 1e9;
    let analytical_nm = (lambda * lambda / (n_g * 2.0 * PI * r)) * 1e9;
    assert!(
        (fsr_nm - analytical_nm).abs() < 0.1,
        "FSR={fsr_nm:.4} nm, analytical={analytical_nm:.4} nm"
    );
}

#[test]
fn ring_resonator_transmission_resonances() {
    let ring = RingResonator::new(5e-6, 2.4, 4.2, 0.1, 0.0);
    let wl = 1550e-9;
    let reso = ring.resonances(wl, 3);
    assert_eq!(reso.len(), 3, "Should return 3 resonances");
    // Resonances should be within the wavelength range
    for &r in &reso {
        assert!(
            r > 1400e-9 && r < 1700e-9,
            "Resonance at {:.2}nm unexpected",
            r * 1e9
        );
    }
}

#[test]
fn ring_resonator_quality_factor_positive() {
    let ring = RingResonator::new(5e-6, 2.4, 4.2, 0.05, 100.0);
    let q = ring.quality_factor(1550e-9);
    assert!(
        q > 0.0 && q.is_finite(),
        "Q factor should be positive and finite: Q={q:.1}"
    );
}

#[test]
fn ring_resonator_through_drops_at_resonance() {
    // Lossless all-pass ring has |T|=1 everywhere (only phase changes).
    // Use lossy ring so there is a measurable dip at resonance.
    let ring = RingResonator::new(5e-6, 2.4, 4.2, 0.1, 5000.0);
    let wl = 1550e-9;
    let reso = ring.resonances(wl, 1);
    if !reso.is_empty() {
        let t_res = ring.transmission_through(&reso)[0];
        // Off resonance (half FSR away)
        let t_off = ring.transmission_through(&[reso[0] + ring.fsr(wl) / 2.0])[0];
        assert!(
            t_res < t_off,
            "Through transmission must dip at resonance: T_res={t_res:.3} T_off={t_off:.3}"
        );
    }
}

// ──────────────────────── Directional Coupler ────────────────────────

#[test]
fn directional_coupler_full_transfer() {
    let kappa = 2000.0; // 2000/m coupling
    let lc = PI / (2.0 * kappa); // coupling length for full transfer
    let coupler = DirectionalCoupler::new(kappa, 1.5e7, lc);
    assert!(
        coupler.cross_power() > 0.99,
        "Should have >99% power in cross port at L_c"
    );
    assert!(
        coupler.through_power() < 0.01,
        "Should have <1% in through port at L_c"
    );
}

#[test]
fn directional_coupler_50_50_split() {
    let kappa = 1000.0;
    let coupler = half_coupler(kappa, 1.0e7);
    let diff = (coupler.cross_power() - coupler.through_power()).abs();
    assert!(
        diff < 1e-10,
        "Half coupler should be 50/50: diff={diff:.2e}"
    );
}

#[test]
fn directional_coupler_power_conservation() {
    let coupler = DirectionalCoupler::new(500.0, 1.5e7, 2e-3);
    assert!((coupler.through_power() + coupler.cross_power() - 1.0).abs() < 1e-10);
}

#[test]
fn directional_coupler_transfer_matrix_unitary() {
    use num_complex::Complex64;
    let kappa = 800.0;
    let coupler = DirectionalCoupler::new(kappa, 1.5e7, 1.5e-3);
    // Test with two orthogonal inputs
    let e1 = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    let e2 = [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
    let o1 = coupler.propagate(e1);
    let o2 = coupler.propagate(e2);
    let p1 = o1[0].norm_sqr() + o1[1].norm_sqr();
    let p2 = o2[0].norm_sqr() + o2[1].norm_sqr();
    assert!((p1 - 1.0).abs() < 1e-10, "Power1={p1:.6}");
    assert!((p2 - 1.0).abs() < 1e-10, "Power2={p2:.6}");
}

// ──────────────────────── MMI Coupler ────────────────────────

#[test]
fn mmi_coupler_beat_length_positive() {
    let mmi = MmiCoupler::new(6e-6, 2.8, 1550e-9, 1.444);
    assert!(mmi.beat_length() > 0.0);
}

#[test]
fn mmi_1x2_length_positive() {
    let mmi = MmiCoupler::new(6e-6, 2.8, 1550e-9, 1.444);
    assert!(mmi.length_1x2() > 0.0);
}

#[test]
fn mmi_power_split_50_50() {
    let splitter = Mmi1x2::new(6e-6, 2.8, 1550e-9, 1.444, 3e-6);
    assert!((splitter.output_power() - 0.5).abs() < 1e-10);
}

// ──────────────────────── Grating Coupler ────────────────────────

#[test]
fn grating_coupler_design_and_verify() {
    // Design for 10° coupling
    let theta = 10.0_f64.to_radians();
    let gc = GratingCoupler::design(2.4, 1.444, theta, 1550e-9).unwrap();
    let angle_back = gc.coupling_angle(1550e-9).unwrap();
    let err_deg = (angle_back.to_degrees() - theta.to_degrees()).abs();
    assert!(err_deg < 0.001, "Grating angle error={err_deg:.4}°");
}

#[test]
fn grating_coupler_bandwidth_positive() {
    let gc = GratingCoupler::design(2.4, 1.444, 8.0_f64.to_radians(), 1550e-9).unwrap();
    let bw = gc.bandwidth_3db(1550e-9);
    assert!(bw > 0.0 && bw < 100e-9, "Bandwidth={:.2}nm", bw * 1e9);
}

#[test]
fn grating_coupler_period_si_photonics() {
    // Typical Si photonics grating: period ~600-700nm for 8-12° coupling
    let gc = GratingCoupler::design(2.4, 1.444, 8.0_f64.to_radians(), 1550e-9).unwrap();
    let period_nm = gc.period * 1e9;
    assert!(
        period_nm > 400.0 && period_nm < 1500.0,
        "Period={period_nm:.1}nm expected 400–1500nm"
    );
}
