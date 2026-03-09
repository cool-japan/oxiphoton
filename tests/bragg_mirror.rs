use approx::assert_relative_eq;
use oxiphoton::prelude::*;

/// Build a Bragg mirror stack: (H L)^N where H=high-index, L=low-index
/// Each layer is lambda/4 optical thickness at the design wavelength
fn bragg_stack(n_high: f64, n_low: f64, design_wavelength: f64, num_pairs: usize) -> Vec<Layer> {
    let d_high = design_wavelength / (4.0 * n_high);
    let d_low = design_wavelength / (4.0 * n_low);

    let mut layers = Vec::with_capacity(2 * num_pairs);
    for _ in 0..num_pairs {
        layers.push(Layer::from_boxed(
            Box::new(ConstantMaterial::from_n("H", n_high)),
            d_high,
        ));
        layers.push(Layer::from_boxed(
            Box::new(ConstantMaterial::from_n("L", n_low)),
            d_low,
        ));
    }
    layers
}

#[test]
fn bragg_mirror_sio2_tio2_10_pairs_high_reflectance() {
    // SiO2/TiO2 Bragg mirror, 10 pairs, design @ 550nm
    // n_SiO2 ~ 1.46 at 550nm, n_TiO2 ~ 2.45 at 550nm
    let n_sio2 = 1.46;
    let n_tio2 = 2.45;
    let design_wl = 550e-9;

    let layers = bragg_stack(n_tio2, n_sio2, design_wl, 10);

    let result = TransferMatrix::solve(
        &layers,
        RefractiveIndex::real(1.0),  // air
        RefractiveIndex::real(1.52), // glass substrate
        Wavelength(design_wl),
        Angle(0.0),
        Polarization::TE,
    );

    assert!(
        result.reflectance > 0.99,
        "Bragg mirror R={:.6} should be > 0.99",
        result.reflectance
    );
    assert_relative_eq!(
        result.reflectance + result.transmittance + result.absorbance,
        1.0,
        epsilon = 1e-10
    );
}

#[test]
fn bragg_mirror_20_pairs_very_high_reflectance() {
    let n_sio2 = 1.46;
    let n_tio2 = 2.45;
    let design_wl = 550e-9;

    let layers = bragg_stack(n_tio2, n_sio2, design_wl, 20);

    let result = TransferMatrix::solve(
        &layers,
        RefractiveIndex::real(1.0),
        RefractiveIndex::real(1.52),
        Wavelength(design_wl),
        Angle(0.0),
        Polarization::TE,
    );

    assert!(
        result.reflectance > 0.9999,
        "20-pair Bragg mirror R={:.8} should be > 0.9999",
        result.reflectance
    );
}

#[test]
fn bragg_mirror_reflectance_spectrum_has_stopband() {
    let n_sio2 = 1.46;
    let n_tio2 = 2.45;
    let design_wl = 550e-9;

    let layers = bragg_stack(n_tio2, n_sio2, design_wl, 10);

    let wavelengths: Vec<Wavelength> = (400..=800)
        .map(|nm| Wavelength::from_nm(nm as f64))
        .collect();

    let results = TransferMatrix::spectrum(
        &layers,
        RefractiveIndex::real(1.0),
        RefractiveIndex::real(1.52),
        &wavelengths,
        Angle(0.0),
        Polarization::TE,
    );

    // Find peak reflectance
    let max_r = results
        .iter()
        .map(|r| r.reflectance)
        .fold(0.0_f64, f64::max);
    assert!(
        max_r > 0.99,
        "Peak reflectance={:.6} should be > 0.99",
        max_r
    );

    // Verify energy conservation for all wavelengths
    for (i, r) in results.iter().enumerate() {
        let sum = r.reflectance + r.transmittance + r.absorbance;
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Energy conservation failed at wavelength index {i}: R+T+A={sum}"
        );
    }

    // Reflectance should be low far from design wavelength
    let r_at_400 = results[0].reflectance;
    assert!(
        r_at_400 < 0.5,
        "R at 400nm={:.4} should be much lower than peak",
        r_at_400
    );
}

#[test]
fn bragg_mirror_energy_conservation_all_angles() {
    let n_sio2 = 1.46;
    let n_tio2 = 2.45;
    let design_wl = 550e-9;

    let layers = bragg_stack(n_tio2, n_sio2, design_wl, 5);

    for deg in (0..=60).step_by(10) {
        for pol in [Polarization::TE, Polarization::TM] {
            let result = TransferMatrix::solve(
                &layers,
                RefractiveIndex::real(1.0),
                RefractiveIndex::real(1.52),
                Wavelength(design_wl),
                Angle::from_degrees(deg as f64),
                pol,
            );
            let sum = result.reflectance + result.transmittance + result.absorbance;
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "Energy conservation failed at {deg} deg, {pol:?}: R+T+A={sum}"
            );
        }
    }
}

#[test]
fn bragg_mirror_with_dispersive_materials() {
    // Use actual Sellmeier materials
    let sio2 = Sellmeier::sio2();
    let tio2 = Sellmeier::tio2();
    let design_wl = 550e-9;

    let n_h = tio2.refractive_index(Wavelength(design_wl)).n;
    let n_l = sio2.refractive_index(Wavelength(design_wl)).n;
    let d_h = design_wl / (4.0 * n_h);
    let d_l = design_wl / (4.0 * n_l);

    let mut layers = Vec::new();
    for _ in 0..10 {
        layers.push(Layer::from_boxed(Box::new(tio2.clone()), d_h));
        layers.push(Layer::from_boxed(Box::new(sio2.clone()), d_l));
    }

    let result = TransferMatrix::solve(
        &layers,
        RefractiveIndex::real(1.0),
        RefractiveIndex::real(1.52),
        Wavelength(design_wl),
        Angle(0.0),
        Polarization::TE,
    );

    assert!(
        result.reflectance > 0.99,
        "Dispersive Bragg mirror R={:.6}",
        result.reflectance
    );
}
