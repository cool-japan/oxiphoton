//! Optical engineering utility functions.
//!
//! Covers Fresnel coefficients, critical-angle / Brewster-angle calculations,
//! coherence length, ABCD ray-transfer matrices, Gaussian beam parameters,
//! and numerical-aperture / spot-size helpers.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use super::conversion::{SPEED_OF_LIGHT, Z0};

// ─── Refractive index types ──────────────────────────────────────────────────

/// Complex refractive index: n + ik
/// n = real part (refractive index), k = imaginary part (extinction coefficient)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RefractiveIndex {
    pub n: f64,
    pub k: f64,
}

impl RefractiveIndex {
    /// Create a complex refractive index.
    pub fn new(n: f64, k: f64) -> Self {
        Self { n, k }
    }

    /// Create a purely real refractive index (lossless).
    pub fn real(n: f64) -> Self {
        Self { n, k: 0.0 }
    }

    /// Convert to complex permittivity: ε = (n + ik)² = (n²−k²) + 2nk·i
    pub fn to_permittivity_scalar(&self) -> Complex64 {
        let n = self.n;
        let k = self.k;
        Complex64::new(n * n - k * k, 2.0 * n * k)
    }

    /// Convert to complex refractive index as `Complex64`.
    pub fn to_complex(&self) -> Complex64 {
        Complex64::new(self.n, self.k)
    }

    /// Absorption coefficient α (m⁻¹) at wavelength λ (m).
    ///
    /// α = 4π k / λ
    pub fn absorption_coefficient(&self, lambda_m: f64) -> f64 {
        4.0 * PI * self.k / lambda_m
    }

    /// Intensity penetration depth (1/e depth) = 1/α (m).
    pub fn penetration_depth(&self, lambda_m: f64) -> f64 {
        let alpha = self.absorption_coefficient(lambda_m);
        if alpha < 1e-30 {
            f64::INFINITY
        } else {
            1.0 / alpha
        }
    }
}

// ─── Permittivity / permeability types ───────────────────────────────────────

/// Permittivity (dielectric constant) — supports isotropic and anisotropic.
#[derive(Debug, Clone, Copy)]
pub enum Permittivity {
    Isotropic(Complex64),
    Diagonal([Complex64; 3]),
    Full([[Complex64; 3]; 3]),
}

/// Permeability — usually 1.0 for optical materials.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Permeability(pub f64);

impl Default for Permeability {
    fn default() -> Self {
        Self(1.0)
    }
}

// ─── Fresnel coefficients ────────────────────────────────────────────────────

/// Fresnel power reflectance at normal incidence: R = ((n₁−n₂)/(n₁+n₂))²
///
/// # Example
/// ```
/// use oxiphoton::units::optical::fresnel_r_normal;
/// let r = fresnel_r_normal(1.0, 1.5); // air → glass
/// assert!((r - 0.04).abs() < 0.001);
/// ```
pub fn fresnel_r_normal(n1: f64, n2: f64) -> f64 {
    ((n1 - n2) / (n1 + n2)).powi(2)
}

/// Fresnel power transmittance at normal incidence: T = 1 − R
pub fn fresnel_t_normal(n1: f64, n2: f64) -> f64 {
    1.0 - fresnel_r_normal(n1, n2)
}

/// Fresnel amplitude reflectance for s-polarization (TE) at angle θ_i (rad).
///
/// r_s = (n₁·cosθ_i − n₂·cosθ_t) / (n₁·cosθ_i + n₂·cosθ_t)
///
/// Returns `None` if total-internal reflection would make θ_t imaginary.
pub fn fresnel_rs_amplitude(n1: f64, n2: f64, theta_i_rad: f64) -> Option<f64> {
    let sin_t = n1 / n2 * theta_i_rad.sin();
    if sin_t.abs() > 1.0 {
        return None; // TIR
    }
    let cos_i = theta_i_rad.cos();
    let cos_t = (1.0 - sin_t * sin_t).sqrt();
    Some((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t))
}

/// Fresnel power reflectance for s-polarization (TE).
///
/// Returns 1.0 (total reflection) beyond critical angle.
pub fn fresnel_rs(n1: f64, n2: f64, theta_i_rad: f64) -> f64 {
    match fresnel_rs_amplitude(n1, n2, theta_i_rad) {
        Some(r) => r * r,
        None => 1.0,
    }
}

/// Fresnel amplitude reflectance for p-polarization (TM).
///
/// r_p = (n₂·cosθ_i − n₁·cosθ_t) / (n₂·cosθ_i + n₁·cosθ_t)
pub fn fresnel_rp_amplitude(n1: f64, n2: f64, theta_i_rad: f64) -> Option<f64> {
    let sin_t = n1 / n2 * theta_i_rad.sin();
    if sin_t.abs() > 1.0 {
        return None;
    }
    let cos_i = theta_i_rad.cos();
    let cos_t = (1.0 - sin_t * sin_t).sqrt();
    Some((n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t))
}

/// Fresnel power reflectance for p-polarization (TM).
///
/// Returns 1.0 beyond critical angle.
pub fn fresnel_rp(n1: f64, n2: f64, theta_i_rad: f64) -> f64 {
    match fresnel_rp_amplitude(n1, n2, theta_i_rad) {
        Some(r) => r * r,
        None => 1.0,
    }
}

/// Average (unpolarised) Fresnel power reflectance at angle θ_i.
pub fn fresnel_r_unpolarised(n1: f64, n2: f64, theta_i_rad: f64) -> f64 {
    0.5 * (fresnel_rs(n1, n2, theta_i_rad) + fresnel_rp(n1, n2, theta_i_rad))
}

// ─── Critical / Brewster angles ───────────────────────────────────────────────

/// Brewster angle θ_B = atan(n₂/n₁) (rad).
///
/// At this angle, p-polarisation has zero reflection.
pub fn brewster_angle(n1: f64, n2: f64) -> f64 {
    (n2 / n1).atan()
}

/// Critical angle for total-internal reflection (rad).
///
/// Returns `None` if n1 ≤ n2 (TIR cannot occur).
pub fn critical_angle(n1: f64, n2: f64) -> Option<f64> {
    if n1 <= n2 {
        return None;
    }
    Some((n2 / n1).asin())
}

/// Snell's law: transmitted angle θ_t (rad) for incidence at θ_i (rad).
///
/// Returns `None` if TIR occurs (sin θ_t > 1).
pub fn snell_theta_t(n1: f64, n2: f64, theta_i_rad: f64) -> Option<f64> {
    let sin_t = n1 / n2 * theta_i_rad.sin();
    if sin_t.abs() > 1.0 {
        None
    } else {
        Some(sin_t.asin())
    }
}

// ─── Optical path and coherence ───────────────────────────────────────────────

/// Optical path length (OPL) = n · L (m).
///
/// # Arguments
/// * `geometric_length_m` — physical length of the medium (m)
/// * `n` — refractive index
pub fn optical_path_length(geometric_length_m: f64, n: f64) -> f64 {
    geometric_length_m * n
}

/// Temporal coherence length from spectral linewidth.
///
/// L_coh = λ² / Δλ  (m), where λ and Δλ are both in metres.
///
/// Represents the 1/e decay length of the fringe visibility.
pub fn coherence_length(lambda_m: f64, delta_lambda_m: f64) -> f64 {
    lambda_m * lambda_m / delta_lambda_m
}

// ─── Dispersion ──────────────────────────────────────────────────────────────

/// Group velocity dispersion from group-index measurement at two wavelengths.
///
/// GVD = dn_g/dλ · (−1/(c·λ)) ≈ (n_g2 − n_g1) / (c · (λ2 − λ1)) [s/m²]
///
/// # Arguments
/// * `ng1`, `ng2` — group indices at wavelengths λ₁, λ₂
/// * `lambda1_m`, `lambda2_m` — wavelengths in metres
pub fn gvd_from_group_index(ng1: f64, ng2: f64, lambda1_m: f64, lambda2_m: f64) -> f64 {
    let d_lambda = lambda2_m - lambda1_m;
    if d_lambda.abs() < 1e-30 {
        return 0.0;
    }
    (ng2 - ng1) / (SPEED_OF_LIGHT * d_lambda)
}

/// Phase-matching bandwidth of a nonlinear crystal of length L (m).
///
/// Δω ≈ 0.886 · π / (|β₂| · L) where β₂ is the GVD (s²/m).
/// d_beta here is |β₂| (s²/m).
pub fn phase_matching_bandwidth(l: f64, d_beta: f64) -> f64 {
    if d_beta.abs() < 1e-50 {
        return f64::INFINITY;
    }
    0.886 * PI / (d_beta.abs() * l)
}

// ─── Lens / optics ────────────────────────────────────────────────────────────

/// Abbe number: V = (n_d − 1) / (n_F − n_C).
///
/// Characterises dispersion of an optical glass.
/// n_d at 587.6 nm (yellow He-d), n_F at 486.1 nm (blue H-F), n_C at 656.3 nm (red H-C).
pub fn abbe_number(nd: f64, nf: f64, nc: f64) -> f64 {
    (nd - 1.0) / (nf - nc)
}

/// Numerical aperture from medium index n and acceptance half-angle θ (rad).
///
/// NA = n · sin(θ)
pub fn numerical_aperture(n: f64, half_angle_rad: f64) -> f64 {
    n * half_angle_rad.sin()
}

/// Diffraction-limited spot radius (1/e²) using the Rayleigh criterion.
///
/// r = 0.61 · λ / NA  (Abbe limit for incoherent illumination)
pub fn diffraction_limited_spot(wavelength_m: f64, na: f64) -> f64 {
    0.61 * wavelength_m / na
}

// ─── Gaussian beam ────────────────────────────────────────────────────────────

/// Rayleigh range z_R = π·w₀² / λ (m).
///
/// # Arguments
/// * `w0` — beam waist radius (m)
/// * `lambda_m` — free-space wavelength (m)
pub fn rayleigh_range(w0: f64, lambda_m: f64) -> f64 {
    PI * w0 * w0 / lambda_m
}

/// Gaussian beam radius at distance z from the waist: w(z) = w₀·√(1 + (z/z_R)²)
pub fn gaussian_beam_radius(w0: f64, z: f64, z_r: f64) -> f64 {
    w0 * (1.0 + (z / z_r).powi(2)).sqrt()
}

/// Gaussian beam on-axis peak irradiance (W/m²) from total power P (W) and waist w₀ (m).
///
/// I₀ = 2P / (π·w₀²)
pub fn gaussian_beam_peak_irradiance(power_w: f64, w0: f64) -> f64 {
    2.0 * power_w / (PI * w0 * w0)
}

// ─── ABCD ray-transfer matrices ───────────────────────────────────────────────

/// ABCD matrix for free-space propagation over distance d (m).
///
/// [[1, d],
///  [0, 1]]
pub fn abcd_freespace(d: f64) -> [[f64; 2]; 2] {
    [[1.0, d], [0.0, 1.0]]
}

/// ABCD matrix for a thin lens with focal length f (m).
///
/// [[1,    0 ],
///  [-1/f, 1 ]]
pub fn abcd_thin_lens(f: f64) -> [[f64; 2]; 2] {
    [[1.0, 0.0], [-1.0 / f, 1.0]]
}

/// ABCD matrix for a flat interface between refractive indices n1 and n2.
///
/// [[1, 0    ],
///  [0, n1/n2]]
pub fn abcd_flat_interface(n1: f64, n2: f64) -> [[f64; 2]; 2] {
    [[1.0, 0.0], [0.0, n1 / n2]]
}

/// ABCD matrix for a curved mirror of radius of curvature R (m).
///
/// Equivalent to a lens with f = R/2.
///
/// [[1,    0],
///  [-2/R, 1]]
pub fn abcd_curved_mirror(radius: f64) -> [[f64; 2]; 2] {
    [[1.0, 0.0], [-2.0 / radius, 1.0]]
}

/// Multiply two 2×2 ABCD matrices: result = a · b (left multiplication).
///
/// Beams propagate right-to-left in the product: M = M_n · … · M_1 · M_0.
pub fn abcd_multiply(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

/// Complex beam parameter q after propagation through ABCD system.
///
/// q' = (A·q + B) / (C·q + D)
pub fn abcd_transform_q(mat: [[f64; 2]; 2], q: Complex64) -> Complex64 {
    let a = Complex64::new(mat[0][0], 0.0);
    let b = Complex64::new(mat[0][1], 0.0);
    let c = Complex64::new(mat[1][0], 0.0);
    let d = Complex64::new(mat[1][1], 0.0);
    (a * q + b) / (c * q + d)
}

/// Gaussian beam q-parameter from waist w₀ (m) and wavelength λ (m).
///
/// q = i·z_R = i·π·w₀²/λ  at the beam waist.
pub fn gaussian_q_at_waist(w0: f64, lambda_m: f64) -> Complex64 {
    let z_r = rayleigh_range(w0, lambda_m);
    Complex64::new(0.0, z_r)
}

// ─── Irradiance and power density ────────────────────────────────────────────

/// Time-averaged intensity from peak E-field amplitude in medium of index n.
///
/// I = n·|E|²/(2·Z₀) [W/m²]
///
/// This function is an alias for the canonical `field::irradiance_from_e_field`.
pub fn optical_irradiance_from_e_field(e_amplitude: f64, n: f64) -> f64 {
    0.5 * n * e_amplitude * e_amplitude / Z0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn refractive_index_to_permittivity() {
        let ri = RefractiveIndex::real(1.5);
        let eps = ri.to_permittivity_scalar();
        assert_relative_eq!(eps.re, 2.25, epsilon = 1e-12);
        assert_relative_eq!(eps.im, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn complex_refractive_index_to_permittivity() {
        let ri = RefractiveIndex::new(0.5, 10.0);
        let eps = ri.to_permittivity_scalar();
        assert_relative_eq!(eps.re, -99.75, epsilon = 1e-12);
        assert_relative_eq!(eps.im, 10.0, epsilon = 1e-12);
    }

    #[test]
    fn fresnel_normal_air_glass() {
        // n1=1, n2=1.5 → R = 0.04 exactly
        let r = fresnel_r_normal(1.0, 1.5);
        assert_relative_eq!(r, 0.04, epsilon = 1e-10);
    }

    #[test]
    fn fresnel_t_plus_r_equals_one_normal() {
        let r = fresnel_r_normal(1.0, 1.5);
        let t = fresnel_t_normal(1.0, 1.5);
        assert_relative_eq!(r + t, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn fresnel_rs_at_zero_equals_normal() {
        // At normal incidence, rs and rp should both equal the normal-incidence formula
        let r_normal = fresnel_r_normal(1.0, 1.5);
        let rs = fresnel_rs(1.0, 1.5, 0.0);
        assert_relative_eq!(r_normal, rs, epsilon = 1e-10);
    }

    #[test]
    fn fresnel_tir_at_critical_angle() {
        // For n1=1.5, n2=1.0, TIR above ~41.8°
        let theta_c = critical_angle(1.5, 1.0).expect("critical angle exists");
        let r = fresnel_rs(1.5, 1.0, theta_c + 0.01);
        assert_relative_eq!(r, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn critical_angle_none_when_n1_less_n2() {
        assert!(critical_angle(1.0, 1.5).is_none());
    }

    #[test]
    fn brewster_angle_glass() {
        // n1=1, n2=1.5 → θ_B = atan(1.5) ≈ 56.31°
        let theta_b = brewster_angle(1.0, 1.5);
        assert_relative_eq!(theta_b.to_degrees(), 56.310, epsilon = 0.001);
    }

    #[test]
    fn snell_tir_returns_none() {
        // n1=1.5, n2=1.0, θ_i = 90°
        let result = snell_theta_t(1.5, 1.0, std::f64::consts::FRAC_PI_2);
        assert!(result.is_none());
    }

    #[test]
    fn snell_normal_incidence() {
        let theta_t = snell_theta_t(1.0, 1.5, 0.0).expect("valid");
        assert_relative_eq!(theta_t, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn coherence_length_basic() {
        // λ=1550nm, Δλ=0.1nm → L_coh = (1550e-9)²/0.1e-9 ≈ 24.025 mm
        let lc = coherence_length(1550e-9, 0.1e-9);
        assert_relative_eq!(lc, 24.025e-3, epsilon = 1e-5);
    }

    #[test]
    fn rayleigh_range_basic() {
        // w₀=1µm, λ=1550nm → z_R = π·(1e-6)²/1550e-9 ≈ 2.027 µm
        let z_r = rayleigh_range(1e-6, 1550e-9);
        assert_relative_eq!(z_r, 2.027e-6, epsilon = 1e-9);
    }

    #[test]
    fn abcd_freespace_determinant_one() {
        let m = abcd_freespace(10e-3);
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        assert_relative_eq!(det, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn abcd_thin_lens_determinant_one() {
        let m = abcd_thin_lens(0.05);
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        assert_relative_eq!(det, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn abcd_multiply_identity() {
        let id = [[1.0, 0.0], [0.0, 1.0]];
        let m = abcd_freespace(5e-3);
        let result = abcd_multiply(id, m);
        assert_relative_eq!(result[0][1], m[0][1], epsilon = 1e-14);
    }

    #[test]
    fn abcd_multiply_lens_freespace() {
        // f=100mm lens (applied first), then d=100mm free-space propagation
        // M = M_space · M_lens = [[0, 0.1],[-10, 1]]
        // A = 0, B = f = 0.1, C = -1/f = -10, D = 1
        let lens = abcd_thin_lens(0.1);
        let space = abcd_freespace(0.1);
        let m = abcd_multiply(space, lens);
        assert_relative_eq!(m[0][0], 0.0, epsilon = 1e-14);
        assert_relative_eq!(m[0][1], 0.1, epsilon = 1e-14);
        assert_relative_eq!(m[1][0], -10.0, epsilon = 1e-12);
        assert_relative_eq!(m[1][1], 1.0, epsilon = 1e-14);
    }

    #[test]
    fn abbe_number_bk7() {
        // BK7 glass: nd=1.5168, nF=1.5224, nC=1.5143
        let v = abbe_number(1.5168, 1.5224, 1.5143);
        // V ≈ 64.2
        assert!(v > 60.0 && v < 70.0, "V_BK7={v:.1}");
    }

    #[test]
    fn numerical_aperture_basic() {
        let na = numerical_aperture(1.0, std::f64::consts::FRAC_PI_6); // sin(30°) = 0.5
        assert_relative_eq!(na, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn irradiance_from_e_field_air() {
        // For E=1 V/m in air (n=1): I = 0.5 * 1 * 1 / Z0 = 1/(2*376.73) ≈ 1.327e-3 W/m²
        let i = optical_irradiance_from_e_field(1.0, 1.0);
        assert_relative_eq!(i, 0.5 / Z0, epsilon = 1e-12);
    }

    #[test]
    fn penetration_depth_zero_k() {
        let ri = RefractiveIndex::real(1.5);
        let depth = ri.penetration_depth(1550e-9);
        assert!(depth.is_infinite());
    }
}
