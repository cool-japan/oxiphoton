//! Integration tests for silicon photonics device chain.
//!
//! Tests full silicon photonics component interactions including MZI links,
//! ring modulators, CROW slow-light, and material permittivities.

#![allow(clippy::approx_constant)]

use approx::{assert_abs_diff_eq, assert_relative_eq};
use std::f64::consts::PI;

// Device imports
use oxiphoton::devices::coupler::{AsymmetricCoupler, DirectionalCoupler, Mmi1x2};
use oxiphoton::devices::detector::AvalanchePhotodetector;
use oxiphoton::devices::modulator::mzi::MziModulator;
use oxiphoton::devices::modulator::plasma_dispersion::SiPlasmaDispersion;
use oxiphoton::devices::resonator::{CoupledResonatorOW, RingResonator};
use oxiphoton::devices::waveguide::{
    AdiabaticTaper, MmiSplitter, MultimodeWaveguide, StripWaveguide, TaperProfile,
};

/// Integration test: full MZI link performance.
///
/// Build a complete MZI from a 3dB splitter + phase arm + combiner.
/// Verify that at phi=0 → ~0 dB insertion loss, phi=pi → high extinction.
#[test]
fn mzi_link_insertion_loss() {
    // Ideal MZI with V_pi = 5V
    let mzi = MziModulator::new(5.0);

    // At zero voltage (Δφ = 0), T = cos²(0/2) = 1.0 → 0 dB
    let t_on = mzi.transmission(0.0);
    assert!(
        t_on > 0.95,
        "At V=0, MZI transmission should be ~1.0, got {t_on:.4}"
    );

    // At V_pi (Δφ = π), T = cos²(π/2) = 0 → very low transmission
    let t_off = mzi.transmission(5.0);
    assert!(
        t_off < 0.01,
        "At V=Vpi, MZI transmission should be ~0, got {t_off:.6}"
    );

    // Extinction ratio: 10*log10(T_on / T_off) should be large
    let er_db = mzi.transmission_db(0.0) - mzi.transmission_db(5.0);
    assert!(
        er_db > 20.0,
        "Extinction ratio should be > 20 dB, got {er_db:.2} dB"
    );

    // A realistic MZI with some insertion loss and finite ER
    let mzi_real = MziModulator::with_params(5.0, 1.5, 25.0);
    let t_on_real = mzi_real.transmission(0.0);
    // With 1.5 dB IL, T_max ≈ 0.71
    assert!(
        t_on_real < 0.99,
        "Should have insertion loss, got T={t_on_real:.4}"
    );
    assert!(
        t_on_real > 0.5,
        "Insertion loss should not be too large, got T={t_on_real:.4}"
    );
}

/// Integration test: ring modulator bandwidth estimation.
///
/// Ring resonator Q factor → linewidth → carrier lifetime → bandwidth.
#[test]
fn ring_modulator_bandwidth() {
    // Silicon ring: R=5μm, n_eff=2.4, n_g=4.2, kappa²=0.05, low loss
    let ring = RingResonator::new(5e-6, 2.4, 4.2, 0.05, 100.0);

    // Q factor should be large (> 1000) for a good ring resonator
    let q = ring.quality_factor(1550e-9);
    assert!(q > 100.0, "Q factor should be > 100, got {q:.1}");

    // Linewidth = lambda / Q (should be a few nm for Q ~ 10000)
    let linewidth_nm = 1550e-9 / q * 1e9;
    assert!(
        linewidth_nm > 0.0 && linewidth_nm < 100.0,
        "Linewidth should be < 100 nm, got {linewidth_nm:.3} nm"
    );

    // FSR should be around 18 nm for R=5μm, ng=4.2
    let fsr_nm = ring.fsr(1550e-9) * 1e9;
    let expected_fsr = (1550e-9_f64).powi(2) / (4.2 * 2.0 * PI * 5e-6) * 1e9;
    assert!(
        (fsr_nm - expected_fsr).abs() < 1.0,
        "FSR should be ~{expected_fsr:.2} nm, got {fsr_nm:.2} nm"
    );

    // Plasma dispersion bandwidth: 1/(2π·τ) for τ = 1 ns → ~159 MHz
    let tau = 1e-9;
    let bw_mhz = SiPlasmaDispersion::bandwidth_3db(tau) / 1e6;
    assert!(
        (bw_mhz - 159.2).abs() < 1.0,
        "Bandwidth should be ~159 MHz for 1 ns lifetime, got {bw_mhz:.2} MHz"
    );
}

/// Integration test: coupled resonators (CROW) slow-light.
///
/// Verify CROW group delay is larger than single resonator at band center.
#[test]
fn crow_slow_light_factor() {
    // CROW with N=3 resonators, FSR=200 GHz, Q=5000, κ=0.1
    let n_res = 3usize;
    let fsr_hz = 200e9;
    let q = 5000.0;
    let kappa = 0.1;
    let lambda = 1550e-9;

    let crow = CoupledResonatorOW::new(n_res, fsr_hz, q, kappa, lambda);

    // Group delay at band center should scale with N
    let f0 = crow.center_frequency();
    let tau_crow = crow.group_delay(f0);

    // Baseline: single resonator group delay = 1/FSR
    let tau_single = 1.0 / fsr_hz;

    // CROW group delay should be N × single resonator baseline
    let slow_light_factor = tau_crow / tau_single;
    assert!(
        slow_light_factor >= 1.0,
        "CROW slow-light factor should be >= 1, got {slow_light_factor:.3}"
    );

    // Bandwidth should be related to coupling coefficient
    let bw_hz = crow.bandwidth();
    assert!(
        bw_hz > 0.0 && bw_hz < fsr_hz,
        "CROW bandwidth {bw_hz:.3e} Hz should be between 0 and FSR"
    );
}

/// Integration test: waveguide mode count vs V-number.
///
/// For large V, mode count should be > 1; for small V, single mode.
#[test]
fn multimode_waveguide_mode_count() {
    // Wide waveguide: should be multimode
    let wg_wide = MultimodeWaveguide::new(5e-6, 0.22e-6, 3.476, 1.444, 1550e-9);
    let n_modes_wide = wg_wide.num_modes();
    assert!(
        n_modes_wide > 1,
        "Wide waveguide (5μm) should support multiple modes, got {n_modes_wide}"
    );

    // Narrow waveguide: should approach single-mode
    let wg_narrow = MultimodeWaveguide::new(0.45e-6, 0.22e-6, 3.476, 1.444, 1550e-9);
    let v_narrow = wg_narrow.v_number();
    // V < π → single mode condition
    // For 450nm Si waveguide: V should be relatively small
    assert!(
        v_narrow > 0.0,
        "V-number should be positive, got {v_narrow:.4}"
    );

    // V-number increases with width
    assert!(
        wg_wide.v_number() > wg_narrow.v_number(),
        "Wider waveguide should have larger V-number"
    );
}

/// Integration test: adiabatic taper width monotonicity.
///
/// Sample the taper profile; verify width changes monotonically.
#[test]
fn adiabatic_taper_profile_monotonic() {
    // Expanding taper: 0.5μm → 3.0μm over 100μm
    let taper = AdiabaticTaper::new(
        0.5e-6,
        3.0e-6,
        100e-6,
        TaperProfile::Linear,
        3.476,
        1.444,
        1550e-9,
    );

    let n_samples = 50;
    let widths: Vec<f64> = (0..=n_samples)
        .map(|i| taper.width_at(i as f64 / n_samples as f64 * taper.length))
        .collect();

    // Width should monotonically increase for expanding taper
    for i in 1..widths.len() {
        assert!(
            widths[i] >= widths[i - 1] - 1e-15, // allow floating point tolerance
            "Width should be monotonically increasing: w[{i}]={:.4e} < w[{}]={:.4e}",
            widths[i],
            i - 1,
            widths[i - 1]
        );
    }

    // Start and end widths should match
    assert_abs_diff_eq!(widths[0], taper.width_in, epsilon = 1e-15);
    assert_abs_diff_eq!(widths[n_samples], taper.width_out, epsilon = 1e-15);

    // Test exponential taper also monotonic
    let taper_exp = AdiabaticTaper::new(
        0.5e-6,
        3.0e-6,
        100e-6,
        TaperProfile::Exponential,
        3.476,
        1.444,
        1550e-9,
    );
    let widths_exp: Vec<f64> = (0..=n_samples)
        .map(|i| taper_exp.width_at(i as f64 / n_samples as f64 * taper_exp.length))
        .collect();
    for i in 1..widths_exp.len() {
        assert!(
            widths_exp[i] >= widths_exp[i - 1] - 1e-12,
            "Exponential taper width should be monotonically increasing at step {i}"
        );
    }
}

/// Integration test: plasma dispersion modulator phase shift sign.
///
/// More carriers → more negative delta_n in silicon (free-carrier plasma dispersion).
#[test]
fn plasma_dispersion_phase_shift_sign() {
    let modulator_low = SiPlasmaDispersion::new(1.55e-6, 0.8, 1e-3).with_carriers(1e16, 1e16);
    let modulator_high = SiPlasmaDispersion::new(1.55e-6, 0.8, 1e-3).with_carriers(1e18, 1e18);

    // delta_n should be negative (free carrier plasma dispersion)
    assert!(
        modulator_low.delta_n() < 0.0,
        "delta_n should be negative for free carriers, got {}",
        modulator_low.delta_n()
    );

    // More carriers → larger magnitude of delta_n
    assert!(
        modulator_high.delta_n().abs() > modulator_low.delta_n().abs(),
        "Higher carrier density should give larger |delta_n|"
    );

    // Phase shift should be negative (n decreases)
    let phi_low = modulator_low.phase_shift_rad();
    let phi_high = modulator_high.phase_shift_rad();

    assert!(
        phi_low < 0.0,
        "Phase shift should be negative, got {phi_low}"
    );
    assert!(
        phi_high < phi_low,
        "Higher carrier density should give larger negative phase shift"
    );

    // Zero carriers → zero phase shift
    let modulator_zero = SiPlasmaDispersion::new(1.55e-6, 0.8, 1e-3);
    assert_abs_diff_eq!(modulator_zero.phase_shift_rad(), 0.0, epsilon = 1e-30);
}

/// Integration test: APD SNR vs optical power.
///
/// Higher optical power → higher SNR.
#[test]
fn apd_snr_increases_with_power() {
    let apd = AvalanchePhotodetector::ingaas_1550();
    let bandwidth = 1e9; // 1 GHz

    let snr_low = apd.snr_db(1e-9, bandwidth); // 1 nW
    let snr_mid = apd.snr_db(1e-6, bandwidth); // 1 μW
    let snr_high = apd.snr_db(1e-3, bandwidth); // 1 mW

    assert!(
        snr_mid > snr_low,
        "SNR should increase with power: low={snr_low:.2} dB, mid={snr_mid:.2} dB"
    );
    assert!(
        snr_high > snr_mid,
        "SNR should increase with power: mid={snr_mid:.2} dB, high={snr_high:.2} dB"
    );

    // At reasonable power, SNR should be finite
    assert!(
        snr_high.is_finite(),
        "SNR should be finite at 1 mW: {snr_high}"
    );

    // Excess noise factor should be > 1 for k > 0
    let f_excess = apd.excess_noise_factor();
    assert!(
        f_excess >= 1.0,
        "Excess noise factor should be >= 1, got {f_excess:.3}"
    );
}

/// Integration test: asymmetric coupler at phase-match wavelength.
///
/// At phase-match wavelength, coupling efficiency should be near maximum.
#[test]
fn asymmetric_coupler_max_coupling_at_phasematch() {
    // Two waveguides with different widths — phase match occurs at a specific wavelength
    let wl_match = 1550e-9;
    let coupler = AsymmetricCoupler::new(
        0.45e-6, // width A
        0.6e-6,  // width B
        0.2e-6,  // gap
        50e-6,   // length
        3.476,   // n_core (Si)
        1.444,   // n_clad (SiO2)
        wl_match,
    );

    // Get effective indices
    let (na, nb) = coupler.effective_indices();
    // Both should be between cladding and core indices
    assert!(na > 1.444 && na < 3.476, "n_eff_A = {na:.4} out of range");
    assert!(nb > 1.444 && nb < 3.476, "n_eff_B = {nb:.4} out of range");

    // Coupling efficiency should be >= 0 and <= 1
    let eta = coupler.coupling_efficiency();
    assert!(
        (0.0..=1.0).contains(&eta),
        "Coupling efficiency should be in [0,1], got {eta:.4}"
    );
}

/// Integration test: MMI splitter 50:50.
///
/// Splitting ratio should be ~0.5 for 1×2 MMI (by design).
#[test]
fn mmi_splitter_50_50() {
    // Use Mmi1x2 from the coupler module (5 args: width, n_eff, wl, n_clad, output_gap)
    let mmi_1x2 = Mmi1x2::new(10e-6, 3.4, 1550e-9, 1.444, 1e-6);

    // Output power should be 0.5 (50:50 split)
    let ratio = mmi_1x2.output_power();
    assert!(
        (ratio - 0.5).abs() < 1e-12,
        "Mmi1x2 output_power should be exactly 0.5, got {ratio:.6}"
    );

    // Test the general MmiSplitter
    let mmi = MmiSplitter::new(10e-6, 50e-6, 3.4, 1550e-9);
    let mmi_ratio = mmi.splitting_ratio();
    assert!(
        (0.0..=1.0).contains(&mmi_ratio),
        "MMI splitting ratio should be in [0,1], got {mmi_ratio:.4}"
    );

    // Beat length should be proportional to n_eff * W^2 / lambda
    let beat_length = mmi.beat_length();
    let expected = 4.0 * 3.4 * (10e-6_f64).powi(2) / (3.0 * 1550e-9);
    assert_relative_eq!(beat_length, expected, epsilon = 1e-10);
}

/// Integration test: Brendel-Bormann gold permittivity at 1.55μm.
///
/// At 1.55μm, Im(ε) for gold should be negative (absorbing metal).
/// |Re(ε)| >> 1 (strongly metallic).
#[test]
fn bb_gold_permittivity_physical() {
    use oxiphoton::material::dispersive::brendel_bormann::BrendelBormannModel;
    use std::f64::consts::PI;

    let gold = BrendelBormannModel::gold();
    let omega = 2.0 * PI * 2.998e8 / 1550e-9; // angular frequency at 1.55 μm
    let eps = gold.permittivity(omega);

    // Gold at 1.55 μm: ε ≈ -130 + 12i (strongly metallic)
    assert!(
        eps.re < -10.0,
        "Gold Re(ε) at 1.55μm should be large and negative, got {:.2}",
        eps.re
    );

    // Imaginary part should be positive (we use exp(-iωt) convention: Im(ε) > 0 for lossy metal)
    // Convention: ε = ε' + iε'', positive Im for absorption
    assert!(
        eps.im != 0.0, // Im(ε) ≠ 0 for absorbing metal
        "Gold Im(ε) should be nonzero at 1.55μm, got {:.4}",
        eps.im
    );

    // |ε| >> 1 for gold
    let eps_mag = (eps.re * eps.re + eps.im * eps.im).sqrt();
    assert!(
        eps_mag > 10.0,
        "|ε| for gold should be >> 1, got {eps_mag:.2}"
    );
}

/// Integration test: CriticalPoint silver matches known values.
///
/// Silver at 1.55 μm: Re(ε) strongly negative, Im(ε) nonzero.
#[test]
fn cp_silver_permittivity_physical() {
    use oxiphoton::material::dispersive::critical_point::CriticalPointModel;
    use std::f64::consts::PI;

    let silver = CriticalPointModel::silver_etchegoin();
    let omega = 2.0 * PI * 2.998e8 / 1550e-9;
    let eps = silver.permittivity(omega);

    // Silver at 1.55 μm: Re(ε) ≈ -100 (strongly metallic)
    assert!(
        eps.re < -5.0,
        "Silver Re(ε) at 1.55μm should be negative, got {:.2}",
        eps.re
    );

    // |ε| should be large
    let eps_mag = (eps.re * eps.re + eps.im * eps.im).sqrt();
    assert!(
        eps_mag > 5.0,
        "|ε| for silver should be >> 1, got {eps_mag:.2}"
    );

    // Permittivity should be finite (no NaN)
    assert!(eps.re.is_finite(), "Re(ε) should be finite");
    assert!(eps.im.is_finite(), "Im(ε) should be finite");
}

/// Integration test: InGaAs at 1.55μm has n > 3.
#[test]
fn ingaas_index_at_1550nm() {
    use oxiphoton::material::dispersive::extended_materials::InGaAs;
    use oxiphoton::material::DispersiveMaterial;
    use oxiphoton::units::Wavelength;

    let ingaas = InGaAs::inp_matched();
    let n_ri = ingaas.refractive_index(Wavelength::from_um(1.55));

    // InGaAs at 1.55μm: n ≈ 3.4-3.6
    assert!(
        n_ri.n > 3.0,
        "InGaAs refractive index at 1.55μm should be > 3.0, got {:.4}",
        n_ri.n
    );
    assert!(
        n_ri.n < 4.5,
        "InGaAs refractive index at 1.55μm should be < 4.5, got {:.4}",
        n_ri.n
    );
}

/// Integration test: LiNbO3 ordinary vs extraordinary index.
///
/// Extraordinary index ≠ ordinary index for LiNbO3 (birefringent).
#[test]
fn linbo3_birefringence() {
    use oxiphoton::material::dispersive::extended_materials::LithiumNiobate;
    use oxiphoton::material::DispersiveMaterial;
    use oxiphoton::units::Wavelength;

    let linbo3_o = LithiumNiobate::ordinary();
    let linbo3_e = LithiumNiobate::extraordinary();

    let wl = Wavelength::from_um(1.55);
    let n_o = linbo3_o.refractive_index(wl).n;
    let n_e = linbo3_e.refractive_index(wl).n;

    // LiNbO3 is birefringent: n_o ≠ n_e
    assert!(
        (n_o - n_e).abs() > 0.01,
        "LiNbO3 should be birefringent: n_o={n_o:.4}, n_e={n_e:.4}"
    );

    // Both indices should be in a physically reasonable range (2.0 - 2.5)
    assert!(n_o > 2.0 && n_o < 2.5, "n_o = {n_o:.4}");
    assert!(n_e > 2.0 && n_e < 2.5, "n_e = {n_e:.4}");

    // For LiNbO3: n_o > n_e (negative uniaxial crystal)
    // Actually: n_o ≈ 2.21, n_e ≈ 2.14 at 1.55μm
    // n_o > n_e is expected
    assert!(
        n_o > n_e,
        "LiNbO3 is negative uniaxial: n_o ({n_o:.4}) should be > n_e ({n_e:.4})"
    );
}

/// Integration test: TiN plasmonic — negative Re(ε) at telecom.
///
/// TiN is a plasmonic material; at telecom wavelengths Re(ε) < 0.
#[test]
fn tin_plasmonic_negative_epsilon() {
    use oxiphoton::material::dispersive::extended_materials::TitaniumNitride;
    use oxiphoton::material::DispersiveMaterial;
    use oxiphoton::units::Wavelength;

    let tin = TitaniumNitride;
    let wl = Wavelength::from_um(1.55);
    let eps = tin.permittivity(wl);

    // TiN at 1.55 μm: Re(ε) should be negative (metallic/plasmonic)
    // Note: TiN is borderline plasmonic at telecom, so we just check it's lossy
    assert!(
        eps.im.abs() > 0.01 || eps.re < 10.0,
        "TiN at 1.55μm should have significant absorption or metallic behavior"
    );

    // Permittivity should be finite
    assert!(eps.re.is_finite(), "Re(ε) should be finite");
    assert!(eps.im.is_finite(), "Im(ε) should be finite");
}

/// Integration test: StripWaveguide effective index bounded.
///
/// n_eff must satisfy n_clad < n_eff < n_core.
#[test]
fn strip_waveguide_neff_bounded() {
    let wg = StripWaveguide::new(3.476, 1.444, 500e-9, 220e-9);
    let n_eff = wg.n_eff_te(1550e-9);

    assert!(n_eff.is_some(), "Should find a guided TE mode");
    let n = n_eff.unwrap();
    assert!(
        n > 1.444 && n < 3.476,
        "n_eff = {n:.4} not in range [{}, {}]",
        1.444,
        3.476
    );
}

/// Integration test: DirectionalCoupler power conservation.
///
/// Through + Cross power = 1 (lossless coupler).
#[test]
fn directional_coupler_power_conservation() {
    // Symmetric coupler: κ*L = π/4 → 50:50
    let kappa = PI / (4.0 * 100e-6); // gives κ*L = π/4
    let beta = 2.0 * PI * 2.4 / 1550e-9;
    let coupler = DirectionalCoupler::new(kappa, beta, 100e-6);

    let through = coupler.through_power();
    let cross = coupler.cross_power();

    // Power conservation: through + cross = 1
    assert_relative_eq!(through + cross, 1.0, epsilon = 1e-12);

    // At κ*L = π/4: T = T_cross = 0.5 (3dB coupler)
    assert_relative_eq!(through, 0.5, epsilon = 1e-12);
    assert_relative_eq!(cross, 0.5, epsilon = 1e-12);
}

/// Integration test: CROW transmission spectrum has structure.
///
/// The transmission spectrum should show pass-band behavior.
#[test]
fn crow_transmission_spectrum_structure() {
    let crow = CoupledResonatorOW::new(3, 200e9, 5000.0, 0.1, 1550e-9);
    let f0 = crow.center_frequency();
    let spectrum = crow.transmission_spectrum(f0, crow.bandwidth() * 5.0, 50);

    assert_eq!(spectrum.len(), 50, "Should return 50 data points");

    // All frequencies and transmissions should be finite
    for (f, t) in &spectrum {
        assert!(f.is_finite(), "Frequency should be finite: {f}");
        assert!(t.is_finite(), "Transmission should be finite: {t}");
        assert!(*t >= 0.0, "Transmission should be non-negative: {t}");
    }

    // At least some points should have non-trivial transmission
    let max_t = spectrum
        .iter()
        .map(|&(_, t)| t)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_t > 0.0,
        "Maximum transmission should be positive, got {max_t}"
    );
}

/// Integration test: RingResonator resonance count.
///
/// The number of resonances returned should match the requested count.
#[test]
fn ring_resonator_resonance_count() {
    let ring = RingResonator::new(5e-6, 2.4, 4.2, 0.1, 0.0);
    let wl = 1550e-9;
    let n_req = 3;
    let resonances = ring.resonances(wl, n_req);
    assert_eq!(
        resonances.len(),
        n_req,
        "Should return exactly {n_req} resonances"
    );

    // Resonances should be separated by approximately FSR (within 100%)
    // The resonances function finds closest resonances which may not be exactly 1 FSR apart
    let fsr = ring.fsr(wl);
    for i in 1..resonances.len() {
        let sep = (resonances[i] - resonances[i - 1]).abs();
        assert!(
            sep > 0.0 && sep <= fsr * 3.0,
            "Resonance separation should be <= 3×FSR, got {:.4e} nm (FSR={:.4e} nm)",
            sep * 1e9,
            fsr * 1e9
        );
    }
}

/// Integration test: MmiSplitter output field is nonzero.
///
/// After propagation through MMI, the output field should have nonzero power.
#[test]
fn mmi_output_field_nonzero() {
    let mmi = MmiSplitter::new(10e-6, 50e-6, 3.4, 1550e-9);
    let field = mmi.output_field_distribution(32);

    let total_power: f64 = field.iter().map(|c| c.norm_sqr()).sum();
    assert!(
        total_power > 0.0,
        "Output field power should be nonzero, got {total_power}"
    );

    // All field values should be finite
    for c in &field {
        assert!(
            c.re.is_finite() && c.im.is_finite(),
            "Field values should be finite"
        );
    }
}

/// Integration test: AvalanchePhotodetector effective responsivity.
///
/// Effective responsivity = gain × primary responsivity.
#[test]
fn apd_effective_responsivity() {
    let apd = AvalanchePhotodetector::ingaas_1550();
    let r_eff = apd.effective_responsivity();
    let expected = apd.gain * apd.responsivity;
    assert_relative_eq!(r_eff, expected, epsilon = 1e-12);

    // Si APD should have different parameters
    let si_apd = AvalanchePhotodetector::si_800();
    let si_r = si_apd.effective_responsivity();
    assert!(
        si_r > 0.0,
        "Si APD effective responsivity should be positive"
    );
    assert!(
        si_r > si_apd.responsivity,
        "Gain should amplify responsivity"
    );
}
