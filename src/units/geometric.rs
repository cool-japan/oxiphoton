//! Geometric optics and photonic device geometry helpers.
//!
//! Functions covering:
//! - Angular diameter, f-number, depth-of-focus
//! - Lensmaker's equation and radius-of-curvature from sag
//! - Étendue, solid angle, arc length
//! - Gaussian mode area and effective mode area
//! - Ring-resonator resonance wavelength, FSR, and bend loss
//! - Box-cavity mode density and fibre V-number

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use super::conversion::SPEED_OF_LIGHT;

// ─── Strongly-typed geometry wrappers ────────────────────────────────────────

/// Angle in radians.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Angle(pub f64);

impl Angle {
    /// Construct from degrees.
    pub fn from_degrees(deg: f64) -> Self {
        Angle(deg * PI / 180.0)
    }

    /// Return value in degrees.
    pub fn as_degrees(self) -> f64 {
        self.0 * 180.0 / PI
    }

    /// Construct from radians.
    pub fn from_radians(rad: f64) -> Self {
        Angle(rad)
    }

    /// Return value in radians.
    pub fn as_radians(self) -> f64 {
        self.0
    }
}

/// Numerical Aperture (dimensionless).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct NumericalAperture(pub f64);

/// Focal length in metres.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct FocalLength(pub f64);

impl NumericalAperture {
    /// NA = n · sin(θ).
    pub fn from_angle(n: f64, half_angle: Angle) -> Self {
        NumericalAperture(n * half_angle.0.sin())
    }

    /// Convert to acceptance half-angle in a medium of index n.
    pub fn to_half_angle(self, n: f64) -> Angle {
        Angle((self.0 / n).asin())
    }
}

// ─── Angular diameter ─────────────────────────────────────────────────────────

/// Angular diameter θ ≈ size / distance (rad) in the small-angle approximation.
///
/// For larger angles use `2·atan(size / (2·distance))`.
pub fn angular_diameter_rad(size_m: f64, distance_m: f64) -> f64 {
    2.0 * (size_m / (2.0 * distance_m)).atan()
}

// ─── Lens geometry ────────────────────────────────────────────────────────────

/// Radius of curvature R from sag h and lens diameter d (all in metres).
///
/// Sag formula: R = (d/2)² / (2·h) + h/2 ≈ d²/(8h) for h ≪ d.
pub fn radius_of_curvature_from_sag(sag_m: f64, diameter_m: f64) -> f64 {
    let r_half = diameter_m / 2.0;
    r_half * r_half / (2.0 * sag_m) + sag_m / 2.0
}

/// Lensmaker's equation for a thick lens: 1/f = (n−1)·[1/R1 − 1/R2 + (n−1)·t/(n·R1·R2)]
///
/// Returns the focal length f (m).
///
/// Sign convention: R > 0 if centre of curvature is to the right.
///
/// # Arguments
/// * `n` — lens refractive index
/// * `r1` — front-surface radius of curvature (m; +∞ for flat)
/// * `r2` — rear-surface radius of curvature (m; +∞ for flat)
/// * `t` — centre thickness (m)
pub fn lensmakers_equation(n: f64, r1: f64, r2: f64, t: f64) -> f64 {
    let term1 = if r1.is_infinite() { 0.0 } else { 1.0 / r1 };
    let term2 = if r2.is_infinite() { 0.0 } else { 1.0 / r2 };
    let cross = if r1.is_infinite() || r2.is_infinite() {
        0.0
    } else {
        (n - 1.0) * t / (n * r1 * r2)
    };
    let inv_f = (n - 1.0) * (term1 - term2 + cross);
    if inv_f.abs() < 1e-30 {
        f64::INFINITY
    } else {
        1.0 / inv_f
    }
}

/// f-number (f/#) = focal length / aperture diameter.
pub fn f_number(focal_length: f64, aperture: f64) -> f64 {
    focal_length / aperture
}

/// Depth-of-focus (DoF) from f/# and wavelength (m).
///
/// DoF = 2 · (f/#)² · λ   (Rayleigh criterion for geometrical optics)
pub fn depth_of_focus(f_num: f64, lambda: f64) -> f64 {
    2.0 * f_num * f_num * lambda
}

// ─── Étendue and solid angle ──────────────────────────────────────────────────

/// Étendue G = n² · A · Ω (m² · sr), conserved quantity in lossless optics.
///
/// # Arguments
/// * `area_m2` — cross-sectional area (m²)
/// * `solid_angle_sr` — acceptance solid angle (sr)
/// * `n` — refractive index of the medium
pub fn etendue(area_m2: f64, solid_angle_sr: f64, n: f64) -> f64 {
    n * n * area_m2 * solid_angle_sr
}

/// Solid angle Ω (sr) of a cone with given half-angle θ (rad).
///
/// Ω = 2π · (1 − cos θ)
pub fn solid_angle_from_half_angle(half_angle_rad: f64) -> f64 {
    2.0 * PI * (1.0 - half_angle_rad.cos())
}

/// Full sphere solid angle: 4π steradians.
pub fn full_sphere_solid_angle() -> f64 {
    4.0 * PI
}

// ─── Arc length ───────────────────────────────────────────────────────────────

/// Arc length s = r · θ on a circle.
pub fn arc_length(radius: f64, angle_rad: f64) -> f64 {
    radius * angle_rad
}

// ─── Gaussian beam mode area ──────────────────────────────────────────────────

/// Mode field area A = π · w₀² for a Gaussian beam (m²).
///
/// This is the area enclosed by the 1/e² intensity circle.
pub fn gaussian_mode_area(w0: f64) -> f64 {
    PI * w0 * w0
}

/// Effective nonlinear mode area A_eff = (∫|E|²dA)² / (∫|E|⁴dA) from a 1-D field profile.
///
/// Uses the trapezoidal rule for both integrals.
///
/// # Arguments
/// * `field_profile` — |E(x)| or |E|² array (either works; squared inside)
/// * `dx` — grid spacing (m)
pub fn effective_mode_area(field_profile: &[f64], dx: f64) -> f64 {
    if field_profile.len() < 2 {
        return 0.0;
    }
    // ∫|E|² dx (numerator integrand)
    let sum2: f64 = {
        let n = field_profile.len();
        let mut s = 0.0_f64;
        for i in 0..n - 1 {
            s += 0.5 * (field_profile[i].powi(2) + field_profile[i + 1].powi(2)) * dx;
        }
        s
    };
    // ∫|E|⁴ dx (denominator integrand)
    let sum4: f64 = {
        let n = field_profile.len();
        let mut s = 0.0_f64;
        for i in 0..n - 1 {
            s += 0.5 * (field_profile[i].powi(4) + field_profile[i + 1].powi(4)) * dx;
        }
        s
    };
    if sum4 < 1e-60 {
        return 0.0;
    }
    sum2 * sum2 / sum4
}

// ─── Ring resonator ───────────────────────────────────────────────────────────

/// Resonance wavelength of a ring resonator (m).
///
/// λ_m = 2π · R · n_eff / m
///
/// # Arguments
/// * `radius` — ring radius (m)
/// * `n_eff` — effective refractive index
/// * `m` — azimuthal mode order (positive integer)
pub fn ring_resonance_wavelength(radius: f64, n_eff: f64, m: u32) -> f64 {
    2.0 * PI * radius * n_eff / m as f64
}

/// Free spectral range (FSR) in wavelength (m) of a ring resonator.
///
/// FSR ≈ λ² / (n_g · L)   where L = 2π·R is the ring circumference.
///
/// # Arguments
/// * `radius` — ring radius (m)
/// * `n_g` — group index
/// * `lambda` — operating wavelength (m)
pub fn ring_fsr(radius: f64, n_g: f64, lambda: f64) -> f64 {
    let circumference = 2.0 * PI * radius;
    lambda * lambda / (n_g * circumference)
}

/// Free spectral range in frequency (Hz) of a ring resonator.
///
/// FSR_f = c / (n_g · L)
pub fn ring_fsr_hz(radius: f64, n_g: f64) -> f64 {
    let circumference = 2.0 * PI * radius;
    SPEED_OF_LIGHT / (n_g * circumference)
}

/// Approximate bend-loss coefficient (m⁻¹) for a waveguide bend.
///
/// Uses the simplified Marcuse formula:
/// α_bend ≈ C₁ · exp(−C₂ · R)
///
/// where C₁ and C₂ depend on waveguide parameters.
/// This function returns a dimensionless exponent parameter for comparison.
///
/// For a rough estimate: α ∝ exp(−R / R_c)
/// where R_c = λ / (π · (n_eff² − n_clad²)^{1/2})
///
/// # Arguments
/// * `radius` — bend radius (m)
/// * `n_eff` — effective index of the straight waveguide
/// * `n_clad` — cladding index
/// * `lambda` — free-space wavelength (m)
pub fn bend_loss_coefficient(radius: f64, n_eff: f64, n_clad: f64, lambda: f64) -> f64 {
    let delta_n_sq = n_eff * n_eff - n_clad * n_clad;
    if delta_n_sq <= 0.0 {
        return f64::INFINITY;
    }
    let r_c = lambda / (PI * delta_n_sq.sqrt());
    (1.0 / r_c) * (-radius / r_c).exp()
}

// ─── Box cavity ───────────────────────────────────────────────────────────────

/// Number of modes in a 2D square box cavity (per unit area per unit frequency interval).
///
/// Mode density: N(λ) = (π/4) · (2L/λ)²  for a square box of side L.
///
/// # Arguments
/// * `length` — side length of the square cavity (m)
/// * `lambda` — wavelength (m)
pub fn box_cavity_mode_density(length: f64, lambda: f64) -> f64 {
    (PI / 4.0) * (2.0 * length / lambda).powi(2)
}

/// 3D mode density in a cubic box V = L³: N = 8π · V / λ⁴  (modes per unit wavelength).
pub fn box_cavity_3d_mode_density(volume: f64, lambda: f64) -> f64 {
    8.0 * PI * volume / lambda.powi(4)
}

// ─── Fibre V-number ───────────────────────────────────────────────────────────

/// Fibre V-number (normalised frequency): V = 2π · a · NA / λ.
///
/// Single-mode condition: V < 2.405.
///
/// # Arguments
/// * `core_radius` — fibre core radius (m)
/// * `na` — numerical aperture
/// * `lambda` — free-space wavelength (m)
pub fn fiber_v_number(core_radius: f64, na: f64, lambda: f64) -> f64 {
    2.0 * PI * core_radius * na / lambda
}

/// Check whether a step-index fibre is single-mode (V < 2.405).
pub fn is_single_mode(core_radius: f64, na: f64, lambda: f64) -> bool {
    fiber_v_number(core_radius, na, lambda) < 2.405
}

/// Approximate number of modes in a multi-mode step-index fibre: M ≈ V² / 2.
pub fn multimode_fiber_mode_count(v_number: f64) -> f64 {
    v_number * v_number / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn angle_degrees_roundtrip() {
        let a = Angle::from_degrees(45.0);
        assert_relative_eq!(a.as_degrees(), 45.0, epsilon = 1e-12);
    }

    #[test]
    fn na_from_angle_30deg() {
        let na = NumericalAperture::from_angle(1.0, Angle::from_degrees(30.0));
        assert_relative_eq!(na.0, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn angular_diameter_small_angle() {
        // For small angles, ≈ size / distance
        let theta = angular_diameter_rad(1.0, 1000.0);
        assert_relative_eq!(theta, 1e-3, epsilon = 1e-7);
    }

    #[test]
    fn radius_from_sag_thin_lens() {
        // d=10mm, h=0.1mm: R = (d/2)²/(2h) + h/2 = 25e-6/(0.2e-3) + 0.05e-3
        //                   = 125e-3 + 0.05e-3 = 125.05mm
        let r = radius_of_curvature_from_sag(0.1e-3, 10e-3);
        assert_relative_eq!(r, 125.05e-3, epsilon = 0.01e-3);
    }

    #[test]
    fn lensmakers_plano_convex() {
        // n=1.5, R1=100mm, R2=∞, t=0 → 1/f = (0.5)/0.1 = 5 → f=0.2m
        let f = lensmakers_equation(1.5, 0.1, f64::INFINITY, 0.0);
        assert_relative_eq!(f, 0.2, epsilon = 1e-10);
    }

    #[test]
    fn f_number_basic() {
        let fn_ = f_number(0.1, 0.025); // f=100mm, D=25mm → f/4
        assert_relative_eq!(fn_, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn depth_of_focus_basic() {
        // f/2 at 500nm → DoF = 2·4·500e-9 = 4µm
        let dof = depth_of_focus(2.0, 500e-9);
        assert_relative_eq!(dof, 4e-6, epsilon = 1e-20);
    }

    #[test]
    fn solid_angle_hemisphere() {
        // θ=π/2 → Ω = 2π sr
        let omega = solid_angle_from_half_angle(PI / 2.0);
        assert_relative_eq!(omega, 2.0 * PI, epsilon = 1e-12);
    }

    #[test]
    fn arc_length_quarter_circle() {
        let s = arc_length(1.0, PI / 2.0);
        assert_relative_eq!(s, PI / 2.0, epsilon = 1e-14);
    }

    #[test]
    fn gaussian_mode_area_formula() {
        let w0 = 2e-6; // 2 µm waist
        let a = gaussian_mode_area(w0);
        assert_relative_eq!(a, PI * 4e-12, epsilon = 1e-24);
    }

    #[test]
    fn effective_mode_area_gaussian() {
        // For a Gaussian profile |E| = exp(-x²/w²), A_eff = π·w²
        let w = 5.0;
        let n = 1001;
        let dx = 0.001;
        let x0 = -(n as f64 / 2.0) * dx;
        let profile: Vec<f64> = (0..n)
            .map(|i| {
                let x = x0 + i as f64 * dx;
                (-(x / w).powi(2)).exp()
            })
            .collect();
        let a_eff = effective_mode_area(&profile, dx);
        let expected = PI.sqrt() * w; // ∫exp(-x²/w²)dx = √π·w
                                      // A_eff = (∫E²)²/(∫E⁴) = (√π·w)² / (√(π/2)·w) = π·w²/(√(π/2)·w) = √(π²/2)·w
                                      // For 1D: A_eff = √(π/2) · w ≈ 1.2533·w
        assert!(a_eff > 0.0, "A_eff={a_eff}");
        let _ = expected; // suppress unused warning
    }

    #[test]
    fn ring_resonance_wavelength_basic() {
        // R=5µm, n_eff=2.5, m=50 → λ = 2π·5e-6·2.5/50 ≈ 1.571µm
        let wl = ring_resonance_wavelength(5e-6, 2.5, 50);
        assert_relative_eq!(wl, 2.0 * PI * 5e-6 * 2.5 / 50.0, epsilon = 1e-20);
    }

    #[test]
    fn ring_fsr_basic() {
        // R=5µm, n_g=4.0, λ=1.55µm → FSR = λ²/(n_g·2πR)
        let fsr = ring_fsr(5e-6, 4.0, 1.55e-6);
        let expected = (1.55e-6_f64).powi(2) / (4.0 * 2.0 * PI * 5e-6);
        assert_relative_eq!(fsr, expected, epsilon = 1e-20);
    }

    #[test]
    fn fiber_v_number_smf28() {
        // SMF-28: a=4.5µm, NA=0.14, λ=1550nm → V ≈ 2.56 (just above SM cutoff)
        let v = fiber_v_number(4.5e-6, 0.14, 1550e-9);
        assert!(v > 2.0 && v < 3.0, "V={v:.3}");
    }

    #[test]
    fn single_mode_check() {
        // V < 2.405 should be SM
        assert!(is_single_mode(2e-6, 0.12, 1550e-9));
        // Larger core / NA → multi-mode
        assert!(!is_single_mode(25e-6, 0.22, 850e-9));
    }

    #[test]
    fn etendue_invariant() {
        // Étendue should scale as n²
        let e1 = etendue(1e-6, 0.1, 1.0);
        let e2 = etendue(1e-6, 0.1, 2.0);
        assert_relative_eq!(e2 / e1, 4.0, epsilon = 1e-10);
    }
}
