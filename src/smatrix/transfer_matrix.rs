use num_complex::Complex64;
use std::f64::consts::PI;

use crate::material::DispersiveMaterial;
use crate::smatrix::Polarization;
use crate::units::{Angle, RefractiveIndex, Wavelength};

/// A layer in a multilayer thin-film stack
pub struct Layer {
    pub material: Box<dyn DispersiveMaterial>,
    pub thickness: f64, // meters
}

impl Layer {
    pub fn new(material: Box<dyn DispersiveMaterial>, thickness: f64) -> Self {
        Self {
            material,
            thickness,
        }
    }

    pub fn from_boxed(material: Box<dyn DispersiveMaterial>, thickness: f64) -> Self {
        Self::new(material, thickness)
    }

    pub fn from_constant(name: impl Into<String>, n: f64, thickness: f64) -> Self {
        Self::new(Box::new(ConstantMaterial::from_n(name, n)), thickness)
    }
}

/// Wavelength-independent material with constant refractive index
#[derive(Debug, Clone)]
pub struct ConstantMaterial {
    ri: RefractiveIndex,
    mat_name: String,
}

impl ConstantMaterial {
    pub fn new(name: impl Into<String>, n: f64, k: f64) -> Self {
        Self {
            ri: RefractiveIndex::new(n, k),
            mat_name: name.into(),
        }
    }

    pub fn from_n(name: impl Into<String>, n: f64) -> Self {
        Self::new(name, n, 0.0)
    }
}

impl DispersiveMaterial for ConstantMaterial {
    fn refractive_index(&self, _wavelength: Wavelength) -> RefractiveIndex {
        self.ri
    }

    fn name(&self) -> &str {
        &self.mat_name
    }
}

/// Result of transfer matrix calculation
#[derive(Debug, Clone, Copy)]
pub struct TransferMatrixResult {
    pub reflectance: f64,
    pub transmittance: f64,
    pub absorbance: f64,
    pub r_complex: Complex64,
    pub t_complex: Complex64,
}

/// Transfer Matrix Method solver for multilayer thin films
///
/// Computes reflection, transmission, and absorption for stratified media
/// using the 2x2 transfer matrix formalism.
pub struct TransferMatrix;

impl TransferMatrix {
    /// Solve for a single wavelength and angle.
    ///
    /// `layers` — the film stack (excluding semi-infinite substrate and superstrate).
    /// `n_incident` — refractive index of incident medium.
    /// `n_substrate` — refractive index of substrate (exit medium).
    /// `wavelength` — incident wavelength.
    /// `angle` — angle of incidence (in incident medium).
    /// `polarization` — TE or TM.
    pub fn solve(
        layers: &[Layer],
        n_incident: RefractiveIndex,
        n_substrate: RefractiveIndex,
        wavelength: Wavelength,
        angle: Angle,
        polarization: Polarization,
    ) -> TransferMatrixResult {
        let k0 = 2.0 * PI / wavelength.0;
        let n_i = Complex64::new(n_incident.n, n_incident.k);
        let n_s = Complex64::new(n_substrate.n, n_substrate.k);

        // Snell's law: n_i * sin(theta_i) = n_j * sin(theta_j)
        let sin_theta_i = Complex64::new(angle.0.sin(), 0.0);
        let n_i_sin = n_i * sin_theta_i;

        let cos_theta_i = cos_theta(n_i, n_i_sin);
        let cos_theta_s = cos_theta(n_s, n_i_sin);

        // Build total transfer matrix M = D_i^{-1} * [prod_j D_j P_j D_j^{-1}] * D_s
        // Using the convention: M = prod of interface and propagation matrices

        // Start with identity matrix
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let mut m: Mat2x2 = [one, zero, zero, one];

        // Interface: incident -> first layer (or substrate if no layers)
        let mut n_prev = n_i;
        let mut cos_prev = cos_theta_i;

        for layer in layers {
            let ri = layer.material.refractive_index(wavelength);
            let n_j = Complex64::new(ri.n, ri.k);
            let cos_j = cos_theta(n_j, n_i_sin);

            // Interface matrix from prev to j
            let iface = interface_matrix(n_prev, cos_prev, n_j, cos_j, polarization);
            mat_mul_assign(&mut m, &iface);

            // Propagation through layer j
            let phase = k0 * n_j * cos_j * layer.thickness;
            let prop = propagation_matrix(phase);
            mat_mul_assign(&mut m, &prop);

            n_prev = n_j;
            cos_prev = cos_j;
        }

        // Final interface: last layer -> substrate
        let iface = interface_matrix(n_prev, cos_prev, n_s, cos_theta_s, polarization);
        mat_mul_assign(&mut m, &iface);

        // r = m21 / m11, t = 1 / m11
        let r = m[2] / m[0];
        let t = one / m[0];

        let reflectance = r.norm_sqr();

        // Transmittance correction factor for different media
        let eta_factor = compute_eta_factor(n_i, cos_theta_i, n_s, cos_theta_s, polarization);
        let transmittance = t.norm_sqr() * eta_factor.re;

        let absorbance = (1.0 - reflectance - transmittance).max(0.0);

        TransferMatrixResult {
            reflectance,
            transmittance,
            absorbance,
            r_complex: r,
            t_complex: t,
        }
    }

    /// Compute spectrum over a range of wavelengths
    pub fn spectrum(
        layers: &[Layer],
        n_incident: RefractiveIndex,
        n_substrate: RefractiveIndex,
        wavelengths: &[Wavelength],
        angle: Angle,
        polarization: Polarization,
    ) -> Vec<TransferMatrixResult> {
        wavelengths
            .iter()
            .map(|&wl| Self::solve(layers, n_incident, n_substrate, wl, angle, polarization))
            .collect()
    }

    /// Convenience: solve with dispersive incident/substrate materials
    pub fn solve_dispersive(
        layers: &[Layer],
        incident: &dyn DispersiveMaterial,
        substrate: &dyn DispersiveMaterial,
        wavelength: Wavelength,
        angle: Angle,
        polarization: Polarization,
    ) -> TransferMatrixResult {
        let n_i = incident.refractive_index(wavelength);
        let n_s = substrate.refractive_index(wavelength);
        Self::solve(layers, n_i, n_s, wavelength, angle, polarization)
    }

    /// Convenience: spectrum with dispersive incident/substrate materials
    pub fn spectrum_dispersive(
        layers: &[Layer],
        incident: &dyn DispersiveMaterial,
        substrate: &dyn DispersiveMaterial,
        wavelengths: &[Wavelength],
        angle: Angle,
        polarization: Polarization,
    ) -> Vec<TransferMatrixResult> {
        wavelengths
            .iter()
            .map(|&wl| Self::solve_dispersive(layers, incident, substrate, wl, angle, polarization))
            .collect()
    }
}

/// Compute cos(theta_j) from Snell's law: n_j * sin(theta_j) = n_i * sin(theta_i)
fn cos_theta(n_j: Complex64, n_i_sin_theta_i: Complex64) -> Complex64 {
    let sin_j = n_i_sin_theta_i / n_j;
    let cos_sq = Complex64::new(1.0, 0.0) - sin_j * sin_j;
    let cos_j = cos_sq.sqrt();
    // Ensure correct branch: Re(cos) >= 0 for forward propagation
    if cos_j.re < 0.0 {
        -cos_j
    } else {
        cos_j
    }
}

/// Interface (Fresnel) matrix between two media
fn interface_matrix(
    n1: Complex64,
    cos1: Complex64,
    n2: Complex64,
    cos2: Complex64,
    polarization: Polarization,
) -> Mat2x2 {
    let (r, t) = fresnel_coefficients(n1, cos1, n2, cos2, polarization);
    let one = Complex64::new(1.0, 0.0);
    // Interface matrix: (1/t) * [[1, r], [r, 1]]
    let inv_t = one / t;
    [inv_t, inv_t * r, inv_t * r, inv_t]
}

/// Fresnel reflection and transmission coefficients
fn fresnel_coefficients(
    n1: Complex64,
    cos1: Complex64,
    n2: Complex64,
    cos2: Complex64,
    polarization: Polarization,
) -> (Complex64, Complex64) {
    match polarization {
        Polarization::TE => {
            // r_s = (n1*cos1 - n2*cos2) / (n1*cos1 + n2*cos2)
            // t_s = 2*n1*cos1 / (n1*cos1 + n2*cos2)
            let a = n1 * cos1;
            let b = n2 * cos2;
            let r = (a - b) / (a + b);
            let t = (Complex64::new(2.0, 0.0) * a) / (a + b);
            (r, t)
        }
        Polarization::TM => {
            // r_p = (n2*cos1 - n1*cos2) / (n2*cos1 + n1*cos2)
            // t_p = 2*n1*cos1 / (n2*cos1 + n1*cos2)
            let a = n2 * cos1;
            let b = n1 * cos2;
            let r = (a - b) / (a + b);
            let t = (Complex64::new(2.0, 0.0) * n1 * cos1) / (a + b);
            (r, t)
        }
    }
}

/// Propagation matrix through a layer with given phase
fn propagation_matrix(phase: Complex64) -> Mat2x2 {
    let zero = Complex64::new(0.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    let exp_pos = (i * phase).exp();
    let exp_neg = (-i * phase).exp();
    [exp_pos, zero, zero, exp_neg]
}

/// 2x2 complex matrix
type Mat2x2 = [Complex64; 4]; // [m11, m12, m21, m22]

/// 2x2 matrix multiplication: a = a * b
fn mat_mul_assign(a: &mut Mat2x2, b: &Mat2x2) {
    let c11 = a[0] * b[0] + a[1] * b[2];
    let c12 = a[0] * b[1] + a[1] * b[3];
    let c21 = a[2] * b[0] + a[3] * b[2];
    let c22 = a[2] * b[1] + a[3] * b[3];
    a[0] = c11;
    a[1] = c12;
    a[2] = c21;
    a[3] = c22;
}

/// Compute the transmittance correction factor
/// For TE: eta = Re(n_s * cos_s) / Re(n_i * cos_i)
/// For TM: eta = Re(n_s* * cos_s) / Re(n_i* * cos_i)
fn compute_eta_factor(
    n_i: Complex64,
    cos_i: Complex64,
    n_s: Complex64,
    cos_s: Complex64,
    polarization: Polarization,
) -> Complex64 {
    match polarization {
        Polarization::TE => {
            let num = (n_s * cos_s).re;
            let den = (n_i * cos_i).re;
            Complex64::new(num / den, 0.0)
        }
        Polarization::TM => {
            let num = (n_s.conj() * cos_s).re;
            let den = (n_i.conj() * cos_i).re;
            Complex64::new(num / den, 0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn glass() -> ConstantMaterial {
        ConstantMaterial::from_n("Glass", 1.5)
    }

    #[test]
    fn fresnel_air_glass_normal_incidence_te() {
        // R = ((n1-n2)/(n1+n2))^2 = ((1-1.5)/(1+1.5))^2 = 0.04
        let result = TransferMatrix::solve(
            &[],
            RefractiveIndex::real(1.0),
            RefractiveIndex::real(1.5),
            Wavelength::from_nm(550.0),
            Angle(0.0),
            Polarization::TE,
        );
        assert_relative_eq!(result.reflectance, 0.04, epsilon = 1e-12);
        assert_relative_eq!(
            result.reflectance + result.transmittance,
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn fresnel_air_glass_normal_incidence_tm() {
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
            result.reflectance + result.transmittance,
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn brewster_angle_tm() {
        // Brewster angle: theta_B = atan(n2/n1) = atan(1.5) ≈ 56.31°
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
    }

    #[test]
    fn total_internal_reflection() {
        // Glass to air, angle > critical angle
        // Critical angle: sin(theta_c) = n2/n1 = 1/1.5 => theta_c ≈ 41.81°
        let theta = Angle::from_degrees(45.0); // > critical angle
        let result = TransferMatrix::solve(
            &[],
            RefractiveIndex::real(1.5),
            RefractiveIndex::real(1.0),
            Wavelength::from_nm(550.0),
            theta,
            Polarization::TE,
        );
        assert_relative_eq!(result.reflectance, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn energy_conservation_single_layer() {
        let layer = Layer::from_boxed(Box::new(glass()), 100e-9);
        let result = TransferMatrix::solve(
            &[layer],
            RefractiveIndex::real(1.0),
            RefractiveIndex::real(1.0),
            Wavelength::from_nm(550.0),
            Angle(0.0),
            Polarization::TE,
        );
        assert_relative_eq!(
            result.reflectance + result.transmittance + result.absorbance,
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn quarter_wave_ar_coating() {
        // lambda/4 coating: n_film = sqrt(n_substrate) for zero reflection
        // n = sqrt(1.5) ≈ 1.2247
        let n_film = 1.5_f64.sqrt();
        let wl = 550e-9;
        let thickness = wl / (4.0 * n_film); // quarter-wave optical thickness

        let layer = Layer::from_boxed(Box::new(ConstantMaterial::from_n("AR", n_film)), thickness);
        let result = TransferMatrix::solve(
            &[layer],
            RefractiveIndex::real(1.0),
            RefractiveIndex::real(1.5),
            Wavelength(wl),
            Angle(0.0),
            Polarization::TE,
        );
        // Perfect AR: R = 0 when n_film = sqrt(n1 * n2)
        assert_relative_eq!(result.reflectance, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn single_layer_thin_film_consistency() {
        // Verify T+R = 1 for lossless dielectric
        for n in [1.2, 1.5, 2.0, 2.5, 3.0] {
            for d_nm in [50.0, 100.0, 200.0, 500.0] {
                let layer =
                    Layer::from_boxed(Box::new(ConstantMaterial::from_n("film", n)), d_nm * 1e-9);
                let result = TransferMatrix::solve(
                    &[layer],
                    RefractiveIndex::real(1.0),
                    RefractiveIndex::real(1.0),
                    Wavelength::from_nm(550.0),
                    Angle(0.0),
                    Polarization::TE,
                );
                assert!(
                    (result.reflectance + result.transmittance - 1.0).abs() < 1e-10,
                    "Failed for n={n}, d={d_nm}nm: R+T={}",
                    result.reflectance + result.transmittance
                );
            }
        }
    }
}
