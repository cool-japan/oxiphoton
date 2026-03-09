//! Fresnel Zone Plate (FZP), Compound Refractive Lens (CRL), and
//! Kirkpatrick–Baez (KB) mirror optics for X-ray focusing.
//!
//! # References
//! - Attwood, D. *Soft X-rays and Extreme Ultraviolet Radiation*, Cambridge (2007).
//! - Snigirev, A. *et al.*, Nature **384**, 49–51 (1996) – compound refractive lens.
//! - Kirkpatrick, P. & Baez, A.V., J. Opt. Soc. Am. **38**, 766 (1948).

use crate::error::{OxiPhotonError, Result};
use std::f64::consts::PI;

/// Convert photon energy in keV to refractive-index decrement δ for Beryllium.
///
/// Approximation valid from 1–100 keV (NIST).
/// δ(Be) ≈ 2.37e-6 * (λ/1nm)²  (non-resonant limit)
fn beryllium_delta(energy_kev: f64) -> f64 {
    let lambda_nm = 1.23984193 / energy_kev; // λ in nm
    2.37e-6 * lambda_nm * lambda_nm
}

/// Imaginary part β for Beryllium (absorption).
/// β(Be) ≈ 3.8e-10 * (λ/1nm)³ (rough tabulated fit)
fn beryllium_beta(energy_kev: f64) -> f64 {
    let lambda_nm = 1.23984193 / energy_kev;
    3.8e-10 * lambda_nm * lambda_nm * lambda_nm
}

/// Convert photon energy in keV to refractive-index decrement δ for Aluminium.
/// δ(Al) ≈ 7.6e-6 * (λ/1nm)²
fn aluminum_delta(energy_kev: f64) -> f64 {
    let lambda_nm = 1.23984193 / energy_kev;
    7.6e-6 * lambda_nm * lambda_nm
}

/// Imaginary part β for Aluminium.
/// β(Al) ≈ 1.1e-8 * (λ/1nm)³
fn aluminum_beta(energy_kev: f64) -> f64 {
    let lambda_nm = 1.23984193 / energy_kev;
    1.1e-8 * lambda_nm * lambda_nm * lambda_nm
}

// ═══════════════════════════════════════════════════════════════════════════
// Fresnel Zone Plate
// ═══════════════════════════════════════════════════════════════════════════

/// Fresnel Zone Plate (FZP) for X-ray focusing.
///
/// A binary diffractive optical element consisting of alternating opaque and
/// transparent annular zones.  The zone radii follow the thin-lens condition:
///
/// ```text
/// r_n  = sqrt(n · λ · f + n²·λ²/4) ≈ sqrt(n·λ·f)   for n·λ ≪ f
/// Δr_N = r_N / (2N)                                  (outermost zone width)
/// ```
///
/// Physical aperture is `D = 2 r_N`.  Numerical aperture `NA = λ / (2 Δr_N)`.
#[derive(Debug, Clone)]
pub struct FresnelZonePlate {
    /// Design wavelength (m).  Typical X-ray range: 0.1–10 nm.
    pub wavelength: f64,
    /// First-order focal length (m).
    pub focal_length: f64,
    /// Total number of Fresnel zones.
    pub n_zones: usize,
    /// Diffraction order for which the plate is designed (+1 or −1 for focusing).
    pub efficiency_order: i32,
    /// Cached outermost zone width Δr_N (m); computed on construction.
    pub outermost_zone_width: f64,
}

impl FresnelZonePlate {
    /// Construct a Fresnel zone plate.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::InvalidWavelength`] if `wavelength` is not
    /// positive and finite.  Returns [`OxiPhotonError::NumericalError`] for
    /// non-positive `focal_length` or zero `n_zones`.
    pub fn new(wavelength: f64, focal_length: f64, n_zones: usize) -> Result<Self> {
        if !wavelength.is_finite() || wavelength <= 0.0 {
            return Err(OxiPhotonError::InvalidWavelength(wavelength));
        }
        if !focal_length.is_finite() || focal_length <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "focal_length must be positive and finite".into(),
            ));
        }
        if n_zones == 0 {
            return Err(OxiPhotonError::NumericalError(
                "n_zones must be at least 1".into(),
            ));
        }
        // Outermost zone width: Δr_N ≈ sqrt(λ·f/N)  (paraxial)
        let delta_r = (wavelength * focal_length / n_zones as f64).sqrt();
        Ok(Self {
            wavelength,
            focal_length,
            n_zones,
            efficiency_order: 1,
            outermost_zone_width: delta_r,
        })
    }

    /// Radius of the `n`-th zone (exact thin-lens formula):
    ///
    /// ```text
    /// r_n = sqrt(n·λ·f + (n·λ)²/4)
    /// ```
    ///
    /// Returns `None` if `n > n_zones`.
    pub fn zone_radius(&self, n: usize) -> Option<f64> {
        if n > self.n_zones {
            return None;
        }
        let nl = n as f64 * self.wavelength;
        Some((nl * self.focal_length + nl * nl / 4.0).sqrt())
    }

    /// Recompute and return the outermost zone width Δr_N = r_N / (2N).
    ///
    /// Stored on construction; this method recomputes from `zone_radius`.
    pub fn outermost_zone_width(&self) -> f64 {
        match self.zone_radius(self.n_zones) {
            Some(r_n) => r_n / (2.0 * self.n_zones as f64),
            None => self.outermost_zone_width,
        }
    }

    /// Outer diameter D = 2 r_N.
    pub fn diameter(&self) -> f64 {
        self.zone_radius(self.n_zones)
            .unwrap_or(2.0 * self.n_zones as f64 * self.outermost_zone_width)
            * 2.0
    }

    /// Numerical aperture: NA = λ / (2 Δr_N).
    pub fn numerical_aperture(&self) -> f64 {
        self.wavelength / (2.0 * self.outermost_zone_width())
    }

    /// Depth of focus (DOF): ±2 (Δr_N)² / λ.
    ///
    /// Positive half is returned (full DOF = 2 × this value).
    pub fn depth_of_focus(&self) -> f64 {
        let dr = self.outermost_zone_width();
        2.0 * dr * dr / self.wavelength
    }

    /// Diffraction efficiency for an ideal *binary amplitude* zone plate at
    /// order `m`:
    ///
    /// ```text
    /// η_m = 1 / (m·π)²   for odd m
    /// η_0 = 1/4           (zero order, transmitted through open zones)
    /// η_m = 0             for even m ≠ 0
    /// ```
    pub fn diffraction_efficiency(&self, order: i32) -> f64 {
        if order == 0 {
            return 0.25;
        }
        if order % 2 == 0 {
            return 0.0;
        }
        1.0 / (order as f64 * PI).powi(2)
    }

    /// Longitudinal chromatic aberration Δf caused by a bandwidth Δλ:
    ///
    /// ```text
    /// Δf = f · (Δλ / λ)
    /// ```
    ///
    /// The sign follows from df/dλ = −f/λ (negative dispersion).
    pub fn chromatic_aberration(&self, delta_lambda: f64) -> f64 {
        self.focal_length * delta_lambda / self.wavelength
    }

    /// Rayleigh resolution limit: δ = 1.22 · Δr_N.
    pub fn resolution(&self) -> f64 {
        1.22 * self.outermost_zone_width()
    }

    /// Diffraction efficiency of a *phase zone plate* with phase shift φ (rad).
    ///
    /// For a pure phase-shifting zone plate of phase depth φ the first-order
    /// efficiency is:
    ///
    /// ```text
    /// η₁ = (2/π)² · sin²(φ/2)
    /// ```
    ///
    /// At φ = π the theoretical maximum is (2/π)² ≈ 0.405.
    pub fn phase_zone_plate_efficiency(&self, phase_shift: f64) -> f64 {
        let s = (phase_shift / 2.0).sin();
        (2.0 / PI).powi(2) * s * s
    }

    /// Focal length scaled to a different diffraction order `m`:
    /// f_m = f₁ / m.
    pub fn focal_length_order(&self, order: i32) -> Option<f64> {
        if order == 0 {
            return None;
        }
        Some(self.focal_length / order as f64)
    }

    /// Zone plate throughput (fraction of incident photons in first order)
    /// accounting for equal open/blocked zones: first-order η₁ = 1/π² ≈ 0.101.
    pub fn first_order_throughput(&self) -> f64 {
        self.diffraction_efficiency(1)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Compound Refractive Lens (CRL)
// ═══════════════════════════════════════════════════════════════════════════

/// Compound Refractive Lens (CRL) for X-rays.
///
/// A stack of N bi-concave parabolic lenses (since n < 1 for X-rays, concave
/// lenses are converging).  The effective focal length is:
///
/// ```text
/// f = R / (2 · N · δ)
/// ```
///
/// where R is the radius of curvature of each surface, N the number of lenses,
/// and δ = Re(1−n) the refractive index decrement.
#[derive(Debug, Clone)]
pub struct CompoundRefractiveLens {
    /// Number of individual lenses in the stack.
    pub n_lenses: usize,
    /// Radius of curvature of the parabolic lens surfaces (m).
    pub radius: f64,
    /// Refractive index decrement δ (Re part of 1 − n).
    pub delta: f64,
    /// Absorption index β (Im part of n).
    pub beta: f64,
    /// Total material thickness along the optical axis (m).
    pub material_thickness: f64,
}

impl CompoundRefractiveLens {
    /// Beryllium CRL — low-Z, low absorption, ideal for hard X-rays.
    ///
    /// Parameters are tuned to `energy_kev` using a simple power-law model.
    pub fn new_beryllium(n_lenses: usize, radius: f64, energy_kev: f64) -> Self {
        let delta = beryllium_delta(energy_kev);
        let beta = beryllium_beta(energy_kev);
        // Minimum web thickness for a parabolic Be lens: ~ 0.3 mm per lens
        let material_thickness = n_lenses as f64 * 3e-4;
        Self {
            n_lenses,
            radius,
            delta,
            beta,
            material_thickness,
        }
    }

    /// Aluminium CRL — cheap, easily machined alternative.
    pub fn new_aluminum(n_lenses: usize, radius: f64, energy_kev: f64) -> Self {
        let delta = aluminum_delta(energy_kev);
        let beta = aluminum_beta(energy_kev);
        let material_thickness = n_lenses as f64 * 5e-4;
        Self {
            n_lenses,
            radius,
            delta,
            beta,
            material_thickness,
        }
    }

    /// Effective focal length: f = R / (2 N δ).
    ///
    /// Returns `None` if δ or N is zero (no focusing).
    pub fn focal_length(&self) -> Option<f64> {
        if self.delta == 0.0 || self.n_lenses == 0 {
            return None;
        }
        Some(self.radius / (2.0 * self.n_lenses as f64 * self.delta))
    }

    /// Transmission through the CRL stack via Beer–Lambert law:
    ///
    /// ```text
    /// T = exp(−μ · d_total)   where μ = 4π β / λ
    /// ```
    pub fn transmission(&self, wavelength: f64) -> f64 {
        if wavelength <= 0.0 {
            return 0.0;
        }
        let mu = 4.0 * PI * self.beta / wavelength; // m⁻¹
        (-mu * self.material_thickness).exp()
    }

    /// Effective (absorption-limited) aperture radius:
    ///
    /// ```text
    /// r_eff = sqrt(λ / (4π β N / R)) = sqrt(R λ / (4π β N))
    /// ```
    ///
    /// This is the radius at which absorption reduces transmission to 1/e.
    pub fn effective_aperture(&self, wavelength: f64) -> f64 {
        if self.beta <= 0.0 || self.n_lenses == 0 || wavelength <= 0.0 {
            return 0.0;
        }
        let denom = 4.0 * PI * self.beta * self.n_lenses as f64 / self.radius;
        if denom <= 0.0 {
            return 0.0;
        }
        (wavelength / denom).sqrt()
    }

    /// Gain (peak intensity / unfocused beam intensity) at the focal spot.
    ///
    /// ```text
    /// G ≈ T · (π r_eff²) / (λ f)
    /// ```
    pub fn gain(&self, wavelength: f64) -> f64 {
        let t = self.transmission(wavelength);
        let r_eff = self.effective_aperture(wavelength);
        match self.focal_length() {
            Some(f) if f > 0.0 && wavelength > 0.0 => t * PI * r_eff * r_eff / (wavelength * f),
            _ => 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Kirkpatrick–Baez (KB) Mirror Pair
// ═══════════════════════════════════════════════════════════════════════════

/// Kirkpatrick–Baez (KB) elliptical mirror pair for X-ray microbeam focusing.
///
/// Two orthogonally oriented elliptical mirrors focus the beam independently
/// in the horizontal and vertical planes.  Typically used at synchrotron
/// beamlines where diffraction-limited focusing in both transverse directions
/// is required.
#[derive(Debug, Clone)]
pub struct KbMirror {
    /// Mirror length along the beam direction (m).
    pub mirror_length: f64,
    /// Grazing incidence angle (rad).
    pub grazing_angle_rad: f64,
    /// Focal length in the horizontal plane (m).
    pub focal_length_h: f64,
    /// Focal length in the vertical plane (m).
    pub focal_length_v: f64,
    /// RMS surface roughness (nm).
    pub surface_roughness_nm: f64,
}

impl KbMirror {
    /// Construct a KB mirror pair.
    ///
    /// `angle_mrad` is the grazing angle in *milli*-radians (typical range:
    /// 1–10 mrad for hard X-rays).
    pub fn new(length: f64, angle_mrad: f64, f_h: f64, f_v: f64) -> Self {
        Self {
            mirror_length: length,
            grazing_angle_rad: angle_mrad * 1e-3,
            focal_length_h: f_h,
            focal_length_v: f_v,
            surface_roughness_nm: 0.3, // typical state-of-the-art value
        }
    }

    /// Critical angle θ_c for total external reflection:
    ///
    /// ```text
    /// θ_c = sqrt(2δ)
    /// ```
    ///
    /// where δ is the refractive index decrement of the mirror material.
    pub fn critical_angle_rad(&self, delta: f64) -> f64 {
        (2.0 * delta).sqrt()
    }

    /// Fresnel reflectivity near the critical angle.
    ///
    /// For s-polarisation at grazing angle θ the Fresnel reflectivity is:
    ///
    /// ```text
    /// R_s = |( θ - sqrt(θ² - 2δ + 2iβ) ) / ( θ + sqrt(θ² - 2δ + 2iβ) )|²
    /// ```
    ///
    /// This implementation uses the exact two-component formula with a complex
    /// square root.
    pub fn reflectivity(&self, _wavelength: f64, delta: f64, beta: f64) -> f64 {
        use num_complex::Complex64;
        let theta = self.grazing_angle_rad;
        let theta2 = theta * theta;
        let under = Complex64::new(theta2 - 2.0 * delta, 2.0 * beta);
        let sq = under.sqrt();
        let num = Complex64::new(theta, 0.0) - sq;
        let den = Complex64::new(theta, 0.0) + sq;
        if den.norm() < 1e-30 {
            return 1.0;
        }
        (num / den).norm_sqr()
    }

    /// Debye–Waller factor for surface roughness:
    ///
    /// ```text
    /// DW = exp( −( 4π σ sin(θ) / λ )² )
    /// ```
    ///
    /// where σ is the RMS roughness.
    pub fn scattering_loss(&self, wavelength: f64) -> f64 {
        if wavelength <= 0.0 {
            return 0.0;
        }
        let sigma = self.surface_roughness_nm * 1e-9;
        let theta = self.grazing_angle_rad;
        let exponent = (4.0 * PI * sigma * theta.sin() / wavelength).powi(2);
        (-exponent).exp()
    }

    /// Diffraction-limited spot size (half-width):
    ///
    /// ```text
    /// d = 0.886 · λ / NA
    /// ```
    ///
    /// where NA ≈ sin(θ) · L / (2 f).  This uses the vertical focal length.
    pub fn spot_size(&self, wavelength: f64) -> f64 {
        let na = self.grazing_angle_rad.sin() * self.mirror_length / (2.0 * self.focal_length_v);
        if na <= 0.0 || wavelength <= 0.0 {
            return f64::INFINITY;
        }
        0.886 * wavelength / na
    }

    /// Numerical aperture of the mirror: NA = sin(θ) × L / (2 f).
    pub fn numerical_aperture(&self) -> f64 {
        self.grazing_angle_rad.sin() * self.mirror_length / (2.0 * self.focal_length_v)
    }

    /// Total intensity throughput (reflectivity × roughness loss for both mirrors).
    pub fn throughput(&self, wavelength: f64, delta: f64, beta: f64) -> f64 {
        let r = self.reflectivity(wavelength, delta, beta);
        let dw = self.scattering_loss(wavelength);
        // Two mirrors in series
        (r * dw).powi(2)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // Test wavelength: 1 nm (1.24 keV soft X-ray)
    const LAMBDA: f64 = 1e-9;
    const FOCAL: f64 = 0.1; // 100 mm
    const N: usize = 1000;

    #[test]
    fn fzp_construction_valid() {
        let fzp = FresnelZonePlate::new(LAMBDA, FOCAL, N).expect("valid FZP");
        // Δr_N ≈ sqrt(λf/N) = sqrt(1e-9 * 0.1 / 1000) = sqrt(1e-13) = 1e-6.5 ≈ 316 pm
        let expected = (LAMBDA * FOCAL / N as f64).sqrt();
        assert_abs_diff_eq!(fzp.outermost_zone_width, expected, epsilon = 1e-15);
    }

    #[test]
    fn fzp_construction_bad_wavelength() {
        assert!(FresnelZonePlate::new(-1e-9, FOCAL, N).is_err());
        assert!(FresnelZonePlate::new(0.0, FOCAL, N).is_err());
        assert!(FresnelZonePlate::new(f64::NAN, FOCAL, N).is_err());
    }

    #[test]
    fn fzp_zone_radius_paraxial() {
        let fzp = FresnelZonePlate::new(LAMBDA, FOCAL, N).expect("valid FZP");
        // r₁ ≈ sqrt(λ·f) in the paraxial limit
        let r1 = fzp.zone_radius(1).expect("zone 1 exists");
        let paraxial = (LAMBDA * FOCAL).sqrt();
        assert_abs_diff_eq!(r1, paraxial, epsilon = paraxial * 1e-6);
    }

    #[test]
    fn fzp_efficiency_binary_amplitude() {
        let fzp = FresnelZonePlate::new(LAMBDA, FOCAL, N).expect("valid FZP");
        // η₁ = 1/π²
        let eta1 = fzp.diffraction_efficiency(1);
        assert_abs_diff_eq!(eta1, 1.0 / PI.powi(2), epsilon = 1e-12);
        // Even orders vanish
        assert_abs_diff_eq!(fzp.diffraction_efficiency(2), 0.0, epsilon = 1e-15);
        // Zero order = 1/4
        assert_abs_diff_eq!(fzp.diffraction_efficiency(0), 0.25, epsilon = 1e-15);
    }

    #[test]
    fn fzp_phase_efficiency_at_pi() {
        let fzp = FresnelZonePlate::new(LAMBDA, FOCAL, N).expect("valid FZP");
        // At φ = π: η = (2/π)² ≈ 0.4053
        let eta = fzp.phase_zone_plate_efficiency(PI);
        assert_abs_diff_eq!(eta, (2.0 / PI).powi(2), epsilon = 1e-12);
    }

    #[test]
    fn fzp_resolution_and_na() {
        let fzp = FresnelZonePlate::new(LAMBDA, FOCAL, N).expect("valid FZP");
        let na = fzp.numerical_aperture();
        let res = fzp.resolution();
        // resolution = 1.22 · Δr_N = 1.22 · λ / (2 NA)
        assert_abs_diff_eq!(res, 1.22 * LAMBDA / (2.0 * na), epsilon = 1e-15);
    }

    #[test]
    fn crl_be_focal_length() {
        // 10 keV, R = 1 mm = 1e-3 m, N = 20
        let crl = CompoundRefractiveLens::new_beryllium(20, 1e-3, 10.0);
        let f = crl.focal_length().expect("valid focal length");
        // δ(Be,10keV) ≈ 2.37e-6 * (0.12398)² ≈ 3.64e-8
        let lambda_nm = 1.23984193 / 10.0;
        let delta = 2.37e-6 * lambda_nm * lambda_nm;
        let expected = 1e-3 / (2.0 * 20.0 * delta);
        assert_abs_diff_eq!(f, expected, epsilon = 1.0);
    }

    #[test]
    fn crl_transmission_monotone_in_beta() {
        let crl_low_beta = CompoundRefractiveLens {
            n_lenses: 10,
            radius: 1e-3,
            delta: 1e-6,
            beta: 1e-10,
            material_thickness: 1e-3,
        };
        let crl_high_beta = CompoundRefractiveLens {
            n_lenses: 10,
            radius: 1e-3,
            delta: 1e-6,
            beta: 1e-8,
            material_thickness: 1e-3,
        };
        let lambda = 1e-10; // 0.1 nm, hard X-ray
                            // Higher β → lower transmission
        assert!(crl_high_beta.transmission(lambda) < crl_low_beta.transmission(lambda));
    }

    #[test]
    fn kb_critical_angle() {
        let kb = KbMirror::new(0.1, 5.0, 0.2, 0.2);
        // For Pt at 10 keV: δ ≈ 4.5e-5 → θ_c ≈ sqrt(9e-5) ≈ 9.49 mrad
        let delta = 4.5e-5;
        let tc = kb.critical_angle_rad(delta);
        assert_abs_diff_eq!(tc, (2.0 * delta).sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn kb_reflectivity_below_critical_angle() {
        // At θ ≪ θ_c the reflectivity should be close to 1
        let kb = KbMirror {
            mirror_length: 0.1,
            grazing_angle_rad: 1e-4, // very small angle
            focal_length_h: 0.5,
            focal_length_v: 0.5,
            surface_roughness_nm: 0.0,
        };
        let delta = 1e-2; // large δ so θ_c ≫ θ
        let beta = 1e-4;
        let r = kb.reflectivity(1e-10, delta, beta);
        assert!(r > 0.95, "expected near-total reflection, got R={r:.4}");
    }

    #[test]
    fn kb_debye_waller_zero_roughness() {
        let mut kb = KbMirror::new(0.1, 5.0, 0.2, 0.2);
        kb.surface_roughness_nm = 0.0;
        let loss = kb.scattering_loss(1e-10);
        assert_abs_diff_eq!(loss, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn fzp_chromatic_aberration_sign() {
        let fzp = FresnelZonePlate::new(LAMBDA, FOCAL, N).expect("valid FZP");
        // Positive Δλ should give positive Δf
        let da = fzp.chromatic_aberration(1e-12);
        assert!(da > 0.0);
    }
}
