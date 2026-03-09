use approx::assert_relative_eq;
use oxiphoton::prelude::*;

#[test]
fn fresnel_air_glass_normal_te() {
    // R = ((1 - 1.5) / (1 + 1.5))^2 = 0.04
    let result = TransferMatrix::solve(
        &[],
        RefractiveIndex::real(1.0),
        RefractiveIndex::real(1.5),
        Wavelength::from_nm(550.0),
        Angle(0.0),
        Polarization::TE,
    );
    assert_relative_eq!(result.reflectance, 0.04, epsilon = 1e-12);
    assert_relative_eq!(result.transmittance, 0.96, epsilon = 1e-12);
    assert_relative_eq!(
        result.reflectance + result.transmittance + result.absorbance,
        1.0,
        epsilon = 1e-12
    );
}

#[test]
fn fresnel_air_glass_normal_tm() {
    let result = TransferMatrix::solve(
        &[],
        RefractiveIndex::real(1.0),
        RefractiveIndex::real(1.5),
        Wavelength::from_nm(550.0),
        Angle(0.0),
        Polarization::TM,
    );
    assert_relative_eq!(result.reflectance, 0.04, epsilon = 1e-12);
    assert_relative_eq!(
        result.reflectance + result.transmittance + result.absorbance,
        1.0,
        epsilon = 1e-12
    );
}

#[test]
fn brewster_angle_tm_zero_reflection() {
    let theta_b = (1.5_f64).atan();
    let result = TransferMatrix::solve(
        &[],
        RefractiveIndex::real(1.0),
        RefractiveIndex::real(1.5),
        Wavelength::from_nm(550.0),
        Angle(theta_b),
        Polarization::TM,
    );
    assert_relative_eq!(result.reflectance, 0.0, epsilon = 1e-10);
    assert_relative_eq!(
        result.reflectance + result.transmittance + result.absorbance,
        1.0,
        epsilon = 1e-10
    );
}

#[test]
fn total_internal_reflection_te() {
    // Glass -> Air, angle > critical angle (41.81 deg)
    let theta = Angle::from_degrees(45.0);
    let result = TransferMatrix::solve(
        &[],
        RefractiveIndex::real(1.5),
        RefractiveIndex::real(1.0),
        Wavelength::from_nm(550.0),
        theta,
        Polarization::TE,
    );
    assert_relative_eq!(result.reflectance, 1.0, epsilon = 1e-10);
    assert_relative_eq!(result.transmittance, 0.0, epsilon = 1e-10);
}

#[test]
fn total_internal_reflection_tm() {
    let theta = Angle::from_degrees(45.0);
    let result = TransferMatrix::solve(
        &[],
        RefractiveIndex::real(1.5),
        RefractiveIndex::real(1.0),
        Wavelength::from_nm(550.0),
        theta,
        Polarization::TM,
    );
    assert_relative_eq!(result.reflectance, 1.0, epsilon = 1e-10);
    assert_relative_eq!(result.transmittance, 0.0, epsilon = 1e-10);
}

#[test]
fn fresnel_at_various_angles_energy_conservation() {
    for deg in (0..=80).step_by(5) {
        let theta = Angle::from_degrees(deg as f64);
        for pol in [Polarization::TE, Polarization::TM] {
            let result = TransferMatrix::solve(
                &[],
                RefractiveIndex::real(1.0),
                RefractiveIndex::real(1.5),
                Wavelength::from_nm(550.0),
                theta,
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
fn single_glass_slab_energy_conservation() {
    for deg in (0..=60).step_by(10) {
        let theta = Angle::from_degrees(deg as f64);
        for pol in [Polarization::TE, Polarization::TM] {
            let layer = Layer::from_boxed(Box::new(ConstantMaterial::from_n("Glass", 1.5)), 200e-9);
            let result = TransferMatrix::solve(
                &[layer],
                RefractiveIndex::real(1.0),
                RefractiveIndex::real(1.0),
                Wavelength::from_nm(550.0),
                theta,
                pol,
            );
            let sum = result.reflectance + result.transmittance;
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "Energy conservation failed for glass slab at {deg} deg, {pol:?}: R+T={sum}"
            );
        }
    }
}
