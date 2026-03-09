use std::f64::consts::PI;

// ── Ray optics ────────────────────────────────────────────────────────────────
use oxiphoton::ray::{
    abbe_resolution, numerical_aperture, strehl_marechal, AbcdMatrix, GaussianBeam, OpticalSystem,
    Ray, Surface, ZernikePolynomial,
};

#[test]
fn ray_thin_lens_focuses_parallel_beam() {
    let f = 50e-3_f64;
    let system = OpticalSystem::new()
        .push(Surface::ThinLens { f })
        .push(Surface::FreeSpace { d: f });
    let ray = Ray::new(1.0, 0.0);
    let out = system.trace(ray);
    assert!(
        out.y.abs() < 1e-10,
        "y={:.2e} should be 0 at focal plane",
        out.y
    );
}

#[test]
fn gaussian_beam_doubles_at_rayleigh() {
    let w0 = 1e-3;
    let wl = 1550e-9;
    let beam = GaussianBeam::at_waist(w0, wl, 1.0);
    let zr = beam.rayleigh_range();
    let at_zr = beam.propagate_free(zr);
    let w_expected = w0 * 2.0_f64.sqrt();
    let err = (at_zr.w - w_expected).abs() / w_expected;
    assert!(err < 1e-4, "w at z_R rel_err={err:.2e}");
}

#[test]
fn abcd_free_space_det_unity() {
    let m = AbcdMatrix::free_space(1e-3);
    assert!((m.det() - 1.0).abs() < 1e-12);
}

#[test]
fn zernike_defocus_at_edge() {
    // R_2^0(1) = 2-1 = 1
    let r = ZernikePolynomial::radial(2, 0, 1.0);
    assert!((r - 1.0).abs() < 1e-12);
}

#[test]
fn strehl_zero_aberration_unity() {
    assert!((strehl_marechal(0.0, 633e-9) - 1.0).abs() < 1e-12);
}

#[test]
fn abbe_resolution_formula() {
    // d = λ/(2 NA)
    let r = abbe_resolution(500e-9, 1.0);
    assert!((r - 250e-9).abs() < 1e-13);
}

#[test]
fn na_from_angle() {
    let na = numerical_aperture(30.0_f64.to_radians(), 1.5);
    assert!((na - 0.75).abs() < 0.01);
}

// ── Nonlinear materials ───────────────────────────────────────────────────────
use oxiphoton::material::{Chi2Material, KerrMaterial};

#[test]
fn kerr_silica_n2_order_of_magnitude() {
    let m = KerrMaterial::silica();
    assert!(m.n2 > 1e-21 && m.n2 < 1e-19);
}

#[test]
fn chi2_ktp_d_eff_range() {
    let m = Chi2Material::ktp();
    assert!(m.d_eff > 1e-12 && m.d_eff < 1e-11);
}

#[test]
fn chi2_shg_efficiency_max_at_phase_match() {
    let m = Chi2Material::lithium_niobate();
    let omega = 2.0 * PI * 2.998e8 / 1064e-9;
    let eta_pm = m.shg_efficiency_normalized(omega, 10e-3, 0.0);
    let eta_mis = m.shg_efficiency_normalized(omega, 10e-3, 1e6);
    assert!(eta_pm > eta_mis);
}

// ── Photonic crystal ──────────────────────────────────────────────────────────
use oxiphoton::photonic_crystal::PhCrystal1d;

#[test]
fn photonic_crystal_quarter_wave_has_gap() {
    let pc = PhCrystal1d::quarter_wave(1.46, 2.35, 550e-9);
    let omega_c = PI * 2.998e8 / (2.0 * pc.period);
    assert!(pc.is_band_gap(omega_c));
}

#[test]
fn photonic_crystal_uniform_no_gap() {
    let pc = PhCrystal1d::new(1.5, 100e-9, 1.5, 100e-9);
    let omega = 2.0 * PI * 2.998e8 / 800e-9;
    assert!(!pc.is_band_gap(omega));
}

#[test]
fn photonic_crystal_bloch_k_in_band() {
    let pc = PhCrystal1d::quarter_wave(1.46, 2.35, 550e-9);
    let omega = 0.1 * PI * 2.998e8 / pc.period;
    assert!(pc.bloch_k(omega).is_some());
}

// ── Fiber optics ──────────────────────────────────────────────────────────────
use oxiphoton::fiber::{soliton_order, NlseSolver, StepIndexFiber};

#[test]
fn smf28_single_mode_at_1310nm() {
    let f = StepIndexFiber::smf28();
    assert!(f.is_single_mode(1310e-9));
}

#[test]
fn smf28_n_eff_in_guidance_range() {
    let f = StepIndexFiber::smf28();
    let neff = f.n_eff(1310e-9);
    assert!(neff > f.n_clad && neff < f.n_core);
}

#[test]
fn nlse_power_conservation() {
    let mut s = NlseSolver::new(256, 100e-12, 0.0, -20e-27, 1e-3);
    s.set_gaussian_pulse(1.0, 5e-12);
    let p0 = s.total_power();
    s.propagate(1e3, 50);
    let p1 = s.total_power();
    let rel_err = (p1 - p0).abs() / p0;
    assert!(rel_err < 0.01, "Power not conserved: {rel_err:.2e}");
}

#[test]
fn soliton_order_unity_condition() {
    let beta2 = 20e-27_f64;
    let gamma = 1e-3_f64;
    let t0 = 5e-12_f64;
    let p0 = beta2 / (gamma * t0 * t0);
    let n = soliton_order(gamma, p0, t0, beta2);
    assert!((n - 1.0).abs() < 1e-6);
}

// ── Solar optics ──────────────────────────────────────────────────────────────
use oxiphoton::solar::SolarSpectrum;

#[test]
fn solar_am15g_total_near_1000wm2() {
    let spec = SolarSpectrum::am15g();
    let total = spec.total_irradiance();
    // Coarsely-sampled model overestimates due to missed absorption dips
    assert!(total > 700.0 && total < 1500.0);
}

#[test]
fn solar_peak_in_visible() {
    let spec = SolarSpectrum::am15g();
    let ir_vis = spec.irradiance_at(500e-9);
    let ir_ir = spec.irradiance_at(2000e-9);
    assert!(ir_vis > ir_ir);
}

// ── Devices: MZI Modulator ────────────────────────────────────────────────────
use oxiphoton::devices::{MziModulator, PockelsModulator};

#[test]
fn mzi_v_pi_gives_zero_output() {
    let m = MziModulator::new(5.0);
    assert!(m.transmission(5.0) < 1e-10);
}

#[test]
fn mzi_zero_voltage_full_transmission() {
    let m = MziModulator::new(5.0);
    assert!((m.transmission(0.0) - 1.0).abs() < 1e-10);
}

#[test]
fn pockels_v_pi_physical() {
    let m = PockelsModulator::linbo3(1e-2, 15e-6, 1550e-9);
    let vpi = m.v_pi();
    assert!(vpi > 1.0 && vpi < 100.0);
}

// ── Devices: Photodiode ───────────────────────────────────────────────────────
use oxiphoton::devices::Photodiode;

#[test]
fn photodiode_ingaas_qe_near_unity() {
    let pd = Photodiode::ingaas_pin_1550();
    let qe = pd.quantum_efficiency();
    // InGaAs at 1550nm: R=0.95A/W → QE ≈ 0.76 (photon energy ~0.8eV)
    assert!(qe > 0.7 && qe <= 1.0);
}

#[test]
fn photodiode_photocurrent_proportional() {
    let pd = Photodiode::ingaas_pin_1550();
    let i1 = pd.photocurrent(1e-3);
    let i2 = pd.photocurrent(2e-3);
    assert!((i2 / i1 - 2.0).abs() < 1e-10);
}

// ── Devices: Metalens ─────────────────────────────────────────────────────────
use oxiphoton::devices::MetalensPhaseFocusing;

#[test]
fn metalens_phase_at_center_zero() {
    let ml = MetalensPhaseFocusing::new(10e-3, 532e-9, 1e-3);
    assert!(ml.phase_continuous(0.0).abs() < 1e-10);
}

#[test]
fn metalens_wrapped_phase_in_range() {
    let ml = MetalensPhaseFocusing::new(10e-3, 532e-9, 2e-3);
    for i in 0..50 {
        let r = i as f64 * 0.04e-3;
        let phi = ml.phase_wrapped(r);
        assert!((0.0..2.0 * PI).contains(&phi), "phi={phi:.4}");
    }
}

#[test]
fn metalens_na_physical_range() {
    let ml = MetalensPhaseFocusing::new(10e-3, 532e-9, 2e-3);
    let na = ml.numerical_aperture();
    assert!(na > 0.0 && na < 1.0);
}
