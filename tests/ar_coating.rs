use approx::assert_relative_eq;
use oxiphoton::prelude::*;

#[test]
fn mgf2_ar_coating_on_glass() {
    // MgF2 (n=1.38) single-layer AR coating on glass (n=1.52)
    // Quarter-wave thickness at 550nm
    let n_mgf2 = 1.38;
    let n_glass = 1.52;
    let design_wl = 550e-9;
    let thickness = design_wl / (4.0 * n_mgf2);

    let layer = Layer::from_boxed(
        Box::new(ConstantMaterial::from_n("MgF2", n_mgf2)),
        thickness,
    );

    let result = TransferMatrix::solve(
        &[layer],
        RefractiveIndex::real(1.0),
        RefractiveIndex::real(n_glass),
        Wavelength(design_wl),
        Angle(0.0),
        Polarization::TE,
    );

    // Expected R ~ 0.012 for MgF2 on glass
    // R = ((n_f^2 - n_i*n_s) / (n_f^2 + n_i*n_s))^2
    let r_theory = ((n_mgf2 * n_mgf2 - 1.0 * n_glass) / (n_mgf2 * n_mgf2 + 1.0 * n_glass)).powi(2);

    assert_relative_eq!(result.reflectance, r_theory, epsilon = 1e-10);
    assert!(
        result.reflectance < 0.015,
        "AR coating R={:.6} should be < 0.015",
        result.reflectance
    );
    assert_relative_eq!(
        result.reflectance + result.transmittance + result.absorbance,
        1.0,
        epsilon = 1e-10
    );
}

#[test]
fn perfect_ar_coating() {
    // n_film = sqrt(n1 * n2) gives R = 0
    let n_glass = 1.5;
    let n_film = (1.0_f64 * n_glass).sqrt();
    let design_wl = 550e-9;
    let thickness = design_wl / (4.0 * n_film);

    let layer = Layer::from_boxed(Box::new(ConstantMaterial::from_n("AR", n_film)), thickness);

    let result = TransferMatrix::solve(
        &[layer],
        RefractiveIndex::real(1.0),
        RefractiveIndex::real(n_glass),
        Wavelength(design_wl),
        Angle(0.0),
        Polarization::TE,
    );

    assert_relative_eq!(result.reflectance, 0.0, epsilon = 1e-10);
    assert_relative_eq!(result.transmittance, 1.0, epsilon = 1e-10);
}

#[test]
fn ar_coating_spectrum_v_shape() {
    // AR coating has minimum reflectance at design wavelength
    let n_mgf2 = 1.38;
    let n_glass = 1.52;
    let design_wl = 550e-9;
    let thickness = design_wl / (4.0 * n_mgf2);

    let wavelengths: Vec<Wavelength> = (400..=800)
        .map(|nm| Wavelength::from_nm(nm as f64))
        .collect();

    let results: Vec<_> = wavelengths
        .iter()
        .map(|&wl| {
            let layer = Layer::from_boxed(
                Box::new(ConstantMaterial::from_n("MgF2", n_mgf2)),
                thickness,
            );
            TransferMatrix::solve(
                &[layer],
                RefractiveIndex::real(1.0),
                RefractiveIndex::real(n_glass),
                wl,
                Angle(0.0),
                Polarization::TE,
            )
        })
        .collect();

    // Find minimum reflectance wavelength
    let min_idx = results
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.reflectance.partial_cmp(&b.reflectance).unwrap())
        .unwrap()
        .0;

    let min_wl_nm = wavelengths[min_idx].as_nm();
    assert!(
        (min_wl_nm - 550.0).abs() < 5.0,
        "Minimum reflectance at {:.1}nm, expected ~550nm",
        min_wl_nm
    );

    // Reflectance should be higher away from design wavelength
    assert!(results[0].reflectance > results[min_idx].reflectance);
    assert!(results[results.len() - 1].reflectance > results[min_idx].reflectance);

    // Energy conservation for all wavelengths
    for (i, r) in results.iter().enumerate() {
        let sum = r.reflectance + r.transmittance + r.absorbance;
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Energy conservation failed at wavelength index {i}: R+T+A={sum}"
        );
    }
}

#[test]
fn ar_coating_with_sellmeier_mgf2() {
    let mgf2 = Sellmeier::mgf2();
    let design_wl = Wavelength::from_nm(550.0);
    let n_mgf2 = mgf2.refractive_index(design_wl).n;
    let thickness = design_wl.0 / (4.0 * n_mgf2);

    let layer = Layer::from_boxed(Box::new(mgf2), thickness);

    let result = TransferMatrix::solve(
        &[layer],
        RefractiveIndex::real(1.0),
        RefractiveIndex::real(1.52),
        design_wl,
        Angle(0.0),
        Polarization::TE,
    );

    assert!(
        result.reflectance < 0.02,
        "Sellmeier MgF2 AR coating R={:.6}",
        result.reflectance
    );
    assert_relative_eq!(
        result.reflectance + result.transmittance + result.absorbance,
        1.0,
        epsilon = 1e-10
    );
}
