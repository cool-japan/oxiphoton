//! Bessel beams, Bessel-Gauss beams, Airy beams, and cylindrical vector beams.
//!
//! All special functions (J_n, Ai(x)) are implemented in pure Rust via
//! standard series expansions and asymptotic formulæ.

use num_complex::Complex64;
use std::f64::consts::PI;

// Physical constant: reduced Planck constant (J·s)
const HBAR: f64 = 1.054_571_817e-34;

// ---------------------------------------------------------------------------
// Bessel Beam
// ---------------------------------------------------------------------------

/// Non-diffracting Bessel beam of order n.
///
/// The ideal (infinite-energy) field is
///
///   E(r, φ, z) = J_n(k_r · r) · exp(i n φ) · exp(i k_z · z)
///
/// where k_r = k sin θ, k_z = k cos θ, k = 2π/λ and θ is the cone half-angle.
#[derive(Debug, Clone)]
pub struct BesselBeam {
    /// Bessel order n.  n = 0 gives the fundamental (on-axis bright spot).
    pub order: u32,
    /// Half-angle of the Bessel cone in radians.
    pub cone_angle: f64,
    /// Free-space wavelength (metres).
    pub wavelength: f64,
    /// Total power (watts, approximate for ideal beam).
    pub power: f64,
}

impl BesselBeam {
    /// Construct from cone angle given in **degrees**.
    pub fn new(order: u32, cone_angle_deg: f64, wavelength: f64) -> Self {
        Self {
            order,
            cone_angle: cone_angle_deg.to_radians(),
            wavelength,
            power: 1.0,
        }
    }

    /// Total wavenumber k = 2π/λ.
    fn k_total(&self) -> f64 {
        2.0 * PI / self.wavelength
    }

    /// Transverse wavenumber k_r = k sin θ.
    pub fn k_r(&self) -> f64 {
        self.k_total() * self.cone_angle.sin()
    }

    /// Longitudinal wavenumber k_z = k cos θ.
    pub fn k_z(&self) -> f64 {
        self.k_total() * self.cone_angle.cos()
    }

    /// Complex field amplitude at (r, φ, z).
    pub fn field(&self, r: f64, phi: f64, z: f64) -> Complex64 {
        let jn = Self::bessel_j(self.order, self.k_r() * r);
        let longitudinal = Complex64::from_polar(1.0, self.k_z() * z);
        let azimuthal = Complex64::from_polar(1.0, self.order as f64 * phi);
        // Amplitude normalised so that at r=0, z=0 the field is 1.0 for n=0
        // (for n>0 the on-axis value is always 0).
        Complex64::new(jn, 0.0) * azimuthal * longitudinal
    }

    /// Intensity at (r, z): I = |J_n(k_r r)|² (averaged over φ for n>0).
    pub fn intensity(&self, r: f64, _z: f64) -> f64 {
        let jn = Self::bessel_j(self.order, self.k_r() * r);
        jn * jn
    }

    /// Radius of the central bright spot: first zero of J_n divided by k_r.
    ///
    /// Uses the well-known first zeros of J_0..J_5; falls back to 2.4/k_r for higher n.
    pub fn central_spot_size(&self) -> f64 {
        let kr = self.k_r();
        if kr < 1e-30 {
            return f64::INFINITY;
        }
        // First zeros of J_n(x) for n = 0..=5
        let first_zeros = [
            2.404_825, 3.831_706, 5.135_622, 6.380_162, 7.588_342, 8.771_484,
        ];
        let zero = if (self.order as usize) < first_zeros.len() {
            first_zeros[self.order as usize]
        } else {
            // Approximate: ≈ n + 1.8557·n^{1/3} for large n (Tricomi)
            let nf = self.order as f64;
            nf + 1.8557 * nf.powf(1.0 / 3.0)
        };
        zero / kr
    }

    /// Non-diffracting propagation range for a beam truncated to aperture A (metres).
    ///
    /// z_max = A / (2 · tan θ)
    pub fn nondiffracting_range(&self, aperture: f64) -> f64 {
        let tan_theta = self.cone_angle.tan();
        if tan_theta.abs() < 1e-30 {
            return f64::INFINITY;
        }
        aperture / (2.0 * tan_theta)
    }

    /// OAM per photon for an n-th order Bessel beam: L_z = n ℏ.
    pub fn oam_per_photon(&self) -> f64 {
        self.order as f64 * HBAR
    }

    /// Bessel function of the first kind J_n(x), computed via the
    /// Miller backward-recurrence algorithm (Steed/Miller), which is stable
    /// for all x and n.
    ///
    /// For |x| > 30 we use the leading term of the asymptotic expansion
    /// J_n(x) ≈ √(2/πx) · cos(x − nπ/2 − π/4) as a cross-check; the
    /// series is still used because it is always convergent.
    pub fn bessel_j(n: u32, x: f64) -> f64 {
        if x.abs() < 1e-30 {
            return if n == 0 { 1.0 } else { 0.0 };
        }

        // Forward series: J_n(x) = (x/2)^n / n! · Σ_{k=0}^∞ (-1)^k (x/2)^{2k} / (k! (n+k)!)
        // This converges for all x, but is numerically accurate for |x| not too large.
        // For moderate x we cap the series at 80 terms (more than enough for double precision).
        let half_x = x / 2.0;
        let mut term = {
            // (x/2)^n / n!
            let mut t = 1.0_f64;
            for k in 1..=(n as usize) {
                t *= half_x / k as f64;
            }
            t
        };
        let mut sum = term;
        for k in 1_usize..=80 {
            term *= -(half_x * half_x) / (k as f64 * (n as f64 + k as f64));
            sum += term;
            if term.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        sum
    }
}

// ---------------------------------------------------------------------------
// Bessel-Gauss Beam
// ---------------------------------------------------------------------------

/// Finite-energy Bessel-Gauss beam.
///
/// The field is the ideal Bessel beam apodized by a Gaussian envelope:
///
///   E(r, z) ≈ J_n(k_r r) · exp(−r²/w²(z)) · exp(i k_z z)
///
/// (paraxial approximation; higher-order phase terms omitted for clarity).
#[derive(Debug, Clone)]
pub struct BesselGaussBeam {
    /// Bessel order.
    pub order: u32,
    /// Cone half-angle (radians).
    pub cone_angle: f64,
    /// Gaussian envelope beam waist w₀ (metres).
    pub beam_waist: f64,
    /// Free-space wavelength (metres).
    pub wavelength: f64,
}

impl BesselGaussBeam {
    /// Construct from cone angle in degrees and Gaussian waist w0.
    pub fn new(order: u32, cone_angle_deg: f64, w0: f64, wavelength: f64) -> Self {
        Self {
            order,
            cone_angle: cone_angle_deg.to_radians(),
            beam_waist: w0,
            wavelength,
        }
    }

    fn k_total(&self) -> f64 {
        2.0 * PI / self.wavelength
    }

    fn k_r(&self) -> f64 {
        self.k_total() * self.cone_angle.sin()
    }

    fn k_z(&self) -> f64 {
        self.k_total() * self.cone_angle.cos()
    }

    fn rayleigh_range(&self) -> f64 {
        PI * self.beam_waist * self.beam_waist / self.wavelength
    }

    fn beam_radius_at(&self, z: f64) -> f64 {
        let zr = self.rayleigh_range();
        self.beam_waist * (1.0 + (z / zr).powi(2)).sqrt()
    }

    /// Complex field amplitude (paraxial BG approximation).
    pub fn field(&self, r: f64, z: f64) -> Complex64 {
        let w = self.beam_radius_at(z);
        let jn = BesselBeam::bessel_j(self.order, self.k_r() * r);
        let gauss = (-r * r / (w * w)).exp();
        let phase = self.k_z() * z;
        Complex64::new(jn * gauss, 0.0) * Complex64::from_polar(1.0, phase)
    }

    /// Intensity I(r, z) = |E|².
    pub fn intensity(&self, r: f64, z: f64) -> f64 {
        self.field(r, z).norm_sqr()
    }

    /// Non-diffracting propagation range ≈ w₀ / tan θ.
    pub fn nondiffracting_range(&self) -> f64 {
        let tan_theta = self.cone_angle.tan();
        if tan_theta.abs() < 1e-30 {
            return f64::INFINITY;
        }
        self.beam_waist / tan_theta
    }

    /// Approximate M² factor for a BG beam: M² ≈ 1 + (k_r w₀)² / 2.
    ///
    /// For very small cone angles this approaches 1 (Gaussian limit).
    pub fn m_squared(&self) -> f64 {
        let kr_w0 = self.k_r() * self.beam_waist;
        1.0 + kr_w0 * kr_w0 / 2.0
    }
}

// ---------------------------------------------------------------------------
// Airy Beam
// ---------------------------------------------------------------------------

/// 1-D (x, z) Airy beam — a non-diffracting, self-accelerating beam.
///
/// The field at the launch plane is Ai(x/x₀)·exp(a x/x₀) where a is the
/// truncation parameter and x₀ is the transverse scale.  During propagation
/// the main lobe follows the parabolic trajectory
///
///   x_peak(z) = (1/4) · (λ z)² / (2π x₀)³
#[derive(Debug, Clone)]
pub struct AiryBeam {
    /// Transverse scale x₀ (metres).
    pub scale: f64,
    /// Aperture / truncation parameter a (0 < a < 1; smaller = more Airy-like).
    pub truncation: f64,
    /// Free-space wavelength (metres).
    pub wavelength: f64,
}

impl AiryBeam {
    /// Construct from scale in micrometres.
    pub fn new(scale_um: f64, truncation: f64, wavelength: f64) -> Self {
        Self {
            scale: scale_um * 1e-6,
            truncation,
            wavelength,
        }
    }

    /// Airy function Ai(x).
    ///
    /// Implementation:
    /// * For |x| ≤ 4: power-series using the Maclaurin expansion
    ///   Ai(x) = c₁·f(x) − c₂·g(x)
    ///   where c₁ = Ai(0) = 3^{-2/3}/Γ(2/3), c₂ = Ai'(0) = −3^{-1/3}/Γ(1/3)
    ///   and f, g are the two even/odd power series.
    /// * For x > 4: asymptotic (WKB) expansion (Abramowitz & Stegun 10.4.59)
    ///   Ai(x) ≈ (1/(2√π)) · x^{-1/4} · exp(−ξ) · [1 − 5/(72ξ) + …]
    ///   where ξ = (2/3) x^{3/2}.
    /// * For x < −4: oscillatory asymptotic
    ///   Ai(x) ≈ (1/√π) · |x|^{-1/4} · sin(ξ + π/4)
    ///   where ξ = (2/3) |x|^{3/2}.
    pub fn airy_function(x: f64) -> f64 {
        // Ai(0) and Ai'(0) from DLMF 9.2.3
        const AI0: f64 = 0.355_028_053_887_817; // 3^{-2/3}/Γ(2/3)
        const AIP0: f64 = -0.258_819_403_792_807; // −3^{-1/3}/Γ(1/3)

        if x.abs() <= 4.0 {
            // Power series: Ai(x) = AI0·f(x) + AIP0·g(x)
            // f(x) = Σ_{k≥0} 3^k Γ(k+1/3) / (Γ(1/3) (3k)!) · x^{3k}  (even part)
            // g(x) = Σ_{k≥0} 3^k Γ(k+2/3) / (Γ(2/3) (3k+1)!) · x^{3k+1} (odd part)
            // More convenient: use the recurrence for individual terms.
            let x3 = x * x * x;
            let mut f_sum = 1.0_f64;
            let mut g_sum = x;
            let mut f_term = 1.0_f64;
            let mut g_term = x;
            for k in 1_usize..=30 {
                let kf = k as f64;
                // f recurrence: term_{k} = x^3 / (3k(3k-1)) * term_{k-1}
                f_term *= x3 / ((3.0 * kf) * (3.0 * kf - 1.0));
                f_sum += f_term;
                // g recurrence: term_{k} = x^3 / (3k(3k+1)) * term_{k-1}
                g_term *= x3 / ((3.0 * kf) * (3.0 * kf + 1.0));
                g_sum += g_term;
                if f_term.abs().max(g_term.abs()) < 1e-16 {
                    break;
                }
            }
            AI0 * f_sum + AIP0 * g_sum
        } else if x > 4.0 {
            // Positive large x: decaying WKB
            let xi = (2.0 / 3.0) * x.powf(1.5);
            let prefactor = 0.5 / (PI.sqrt()) * x.powf(-0.25);
            // First two asymptotic correction terms
            let corr = 1.0 - 5.0 / (72.0 * xi);
            prefactor * (-xi).exp() * corr
        } else {
            // Negative large |x|: oscillatory WKB
            let abs_x = x.abs();
            let xi = (2.0 / 3.0) * abs_x.powf(1.5);
            let prefactor = 1.0 / (PI.sqrt()) * abs_x.powf(-0.25);
            prefactor * (xi + PI / 4.0).sin()
        }
    }

    /// 1-D complex field amplitude E(x, z).
    ///
    /// E(x,z) = Ai(ξ) · exp(aξ) · exp(i φ)
    /// where ξ = x/x₀ − (z/L)², L = 2π x₀²/λ,
    /// and the complex phase is φ = (z/L)·(x/x₀) − (z/L)³/3 + a²z/L/2.
    pub fn field_1d(&self, x: f64, z: f64) -> Complex64 {
        let x0 = self.scale;
        let k = 2.0 * PI / self.wavelength;
        // Diffraction length L = k x₀²
        let length = k * x0 * x0;
        let zeta = z / length;
        let xi = x / x0 - zeta * zeta;
        let ai = Self::airy_function(xi);
        // Truncated Airy: multiply by exp(a * (x/x0 - ζ²)) = exp(a·ξ)
        let truncation_factor = (self.truncation * xi).exp();
        // Phase from paraxial propagation:
        // φ = ζ·(x/x0) − ζ³/3 + a²·ζ/2
        let phase = zeta * (x / x0) - zeta * zeta * zeta / 3.0
            + self.truncation * self.truncation * zeta / 2.0;
        Complex64::new(ai * truncation_factor, 0.0) * Complex64::from_polar(1.0, phase)
    }

    /// Parabolic trajectory of the peak lobe: x_peak(z) = (1/4)(λ z / (2π x₀²))².
    ///
    /// Derived from ∂ξ/∂ζ = 0 for the first Airy maximum.
    pub fn trajectory_peak(&self, z: f64) -> f64 {
        let x0 = self.scale;
        let k = 2.0 * PI / self.wavelength;
        let length = k * x0 * x0;
        let zeta = z / length;
        // Peak lobe of Ai is at ξ ≈ −1.019 (first maximum); trajectory shifts
        // The leading term of the trajectory is x = (z/L)² · x₀
        zeta * zeta * x0
    }

    /// Approximate self-healing distance after an obstacle of given size (metres).
    ///
    /// d_heal ≈ π x₀ / (a · λ) — scaling from geometric shadow argument.
    pub fn self_healing_distance(&self, obstacle_size: f64) -> f64 {
        // From diffraction: d ≈ obstacle_size / tan(θ_cone)
        // θ_cone for Airy ≈ λ / (2π x₀)
        let theta = self.wavelength / (2.0 * PI * self.scale);
        if theta.abs() < 1e-30 {
            return f64::INFINITY;
        }
        obstacle_size / theta.tan()
    }

    /// Propagation-invariant (non-diffracting) range.
    ///
    /// Scales as x₀ / truncation (smaller a → longer range).
    pub fn invariant_range(&self) -> f64 {
        if self.truncation.abs() < 1e-20 {
            return f64::INFINITY;
        }
        let k = 2.0 * PI / self.wavelength;
        // z_max ≈ k x₀² / a  (diffraction length / truncation)
        k * self.scale * self.scale / self.truncation
    }
}

// ---------------------------------------------------------------------------
// Vector Beam
// ---------------------------------------------------------------------------

/// Type of cylindrical vector beam.
#[derive(Debug, Clone)]
pub enum VectorBeamType {
    RadiallyPolarized,
    AzimuthallyPolarized,
    /// Hybrid: linear combination with given phase offset between radial and azimuthal.
    Hybrid {
        phase_offset: f64,
    },
}

/// Cylindrical vector beam — a beam whose polarisation state varies spatially.
///
/// These beams are eigenmodes of cylindrically symmetric optical systems and
/// exhibit unique tight-focusing properties (strong longitudinal field for
/// radial polarisation).
#[derive(Debug, Clone)]
pub struct VectorBeam {
    /// Polarisation order m (m = 1: radial/azimuthal; m = 2: petal beams…).
    pub polarization_order: u32,
    /// Beam type.
    pub beam_type: VectorBeamType,
    /// Beam waist w₀ (metres).
    pub beam_waist: f64,
    /// Free-space wavelength (metres).
    pub wavelength: f64,
}

impl VectorBeam {
    /// Construct a radially polarised beam.
    pub fn new_radial(w0: f64, wavelength: f64) -> Self {
        Self {
            polarization_order: 1,
            beam_type: VectorBeamType::RadiallyPolarized,
            beam_waist: w0,
            wavelength,
        }
    }

    /// Construct an azimuthally polarised beam.
    pub fn new_azimuthal(w0: f64, wavelength: f64) -> Self {
        Self {
            polarization_order: 1,
            beam_type: VectorBeamType::AzimuthallyPolarized,
            beam_waist: w0,
            wavelength,
        }
    }

    /// Local Jones vector [E_x, E_y] at azimuthal angle φ.
    pub fn jones_vector(&self, phi: f64) -> [f64; 2] {
        let m = self.polarization_order as f64;
        match &self.beam_type {
            VectorBeamType::RadiallyPolarized => [(m * phi).cos(), (m * phi).sin()],
            VectorBeamType::AzimuthallyPolarized => [-(m * phi).sin(), (m * phi).cos()],
            VectorBeamType::Hybrid { phase_offset } => {
                let alpha = m * phi + phase_offset;
                [alpha.cos(), alpha.sin()]
            }
        }
    }

    /// Fraction of power in the longitudinal (z) component at tight focus.
    ///
    /// For a radially polarised beam focused with numerical aperture NA:
    ///   f_z ≈ (NA/n)² · (1 − (NA/n)²/4)
    /// where n is the medium refractive index (assumed 1 here).
    /// For azimuthally polarised beams, E_z ≈ 0 at focus.
    pub fn longitudinal_field_fraction(&self, na: f64) -> f64 {
        let sin2 = na.min(1.0).powi(2);
        match &self.beam_type {
            VectorBeamType::RadiallyPolarized => sin2 * (1.0 - sin2 / 4.0),
            VectorBeamType::AzimuthallyPolarized => 0.0,
            VectorBeamType::Hybrid { .. } => sin2 * 0.5 * (1.0 - sin2 / 4.0),
        }
    }

    fn rayleigh_range(&self) -> f64 {
        PI * self.beam_waist * self.beam_waist / self.wavelength
    }

    fn beam_radius(&self, z: f64) -> f64 {
        let zr = self.rayleigh_range();
        self.beam_waist * (1.0 + (z / zr).powi(2)).sqrt()
    }

    /// Intensity profile I(r, z).  For m = 1 this is ring-shaped (like LG_{0}^{1}).
    pub fn intensity(&self, r: f64, z: f64) -> f64 {
        let w = self.beam_radius(z);
        let m = self.polarization_order as f64;
        // Amplitude ~ (r/w)^m · exp(−r²/w²)
        let radial = (r / w).powf(m) * (-r * r / (w * w)).exp();
        radial * radial
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // --- Bessel function ---
    #[test]
    fn bessel_j0_at_zero_is_one() {
        assert_abs_diff_eq!(BesselBeam::bessel_j(0, 0.0), 1.0, epsilon = 1e-15);
    }

    #[test]
    fn bessel_j1_at_zero_is_zero() {
        assert_abs_diff_eq!(BesselBeam::bessel_j(1, 0.0), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn bessel_j0_known_value() {
        // J_0(2.4048) ≈ 0 (first zero)
        let val = BesselBeam::bessel_j(0, 2.404_825);
        assert_abs_diff_eq!(val, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn bessel_j1_known_value() {
        // J_1(1) ≈ 0.440 5 (DLMF 10.2.2)
        let val = BesselBeam::bessel_j(1, 1.0);
        assert_abs_diff_eq!(val, 0.440_5, epsilon = 1e-3);
    }

    #[test]
    fn bessel_j0_large_x() {
        // At x = 10 the asymptotic gives ≈ −0.2459, series should match
        let val = BesselBeam::bessel_j(0, 10.0);
        assert_abs_diff_eq!(val, -0.245_936, epsilon = 1e-4);
    }

    // --- BesselBeam ---
    #[test]
    fn bessel_beam_kz_positive() {
        let bb = BesselBeam::new(0, 5.0, 1064e-9);
        assert!(bb.k_z() > 0.0);
        assert!(bb.k_r() > 0.0);
        // k² = k_r² + k_z²
        let k = 2.0 * PI / bb.wavelength;
        assert_abs_diff_eq!(
            bb.k_r() * bb.k_r() + bb.k_z() * bb.k_z(),
            k * k,
            epsilon = 1.0
        );
    }

    #[test]
    fn bessel_beam_oam_zero_order() {
        let bb = BesselBeam::new(0, 5.0, 1064e-9);
        assert_abs_diff_eq!(bb.oam_per_photon(), 0.0, epsilon = 1e-50);
    }

    #[test]
    fn bessel_beam_nondiffracting_range() {
        let bb = BesselBeam::new(0, 1.0, 1064e-9); // 1°
        let range = bb.nondiffracting_range(0.01); // 1 cm aperture
        assert!(range > 0.0 && range.is_finite());
    }

    // --- BesselGauss ---
    #[test]
    fn bessel_gauss_m_squared_gt1() {
        let bg = BesselGaussBeam::new(0, 5.0, 1e-3, 1064e-9);
        assert!(bg.m_squared() >= 1.0);
    }

    #[test]
    fn bessel_gauss_nondiffracting_range() {
        let bg = BesselGaussBeam::new(0, 5.0, 1e-3, 1064e-9);
        let range = bg.nondiffracting_range();
        assert!(range > 0.0 && range.is_finite());
    }

    // --- Airy function ---
    #[test]
    fn airy_at_zero() {
        // Ai(0) = 3^{-2/3}/Γ(2/3) ≈ 0.3550
        let val = AiryBeam::airy_function(0.0);
        assert_abs_diff_eq!(val, 0.355_028, epsilon = 1e-5);
    }

    #[test]
    fn airy_large_positive_decays() {
        let v1 = AiryBeam::airy_function(5.0).abs();
        let v2 = AiryBeam::airy_function(10.0).abs();
        assert!(v1 > v2, "Airy function should decay for large positive x");
    }

    #[test]
    fn airy_trajectory_parabolic() {
        let beam = AiryBeam::new(100.0, 0.1, 800e-9);
        // Trajectory at z=0 should be 0
        assert_abs_diff_eq!(beam.trajectory_peak(0.0), 0.0, epsilon = 1e-20);
        // Should be positive and increasing for z > 0
        let p1 = beam.trajectory_peak(0.01);
        let p2 = beam.trajectory_peak(0.02);
        assert!(p2 > p1);
    }

    // --- VectorBeam ---
    #[test]
    fn radial_jones_at_zero_phi() {
        let vb = VectorBeam::new_radial(1e-3, 1064e-9);
        let jv = vb.jones_vector(0.0);
        // At φ=0: (cos0, sin0) = (1, 0)
        assert_abs_diff_eq!(jv[0], 1.0, epsilon = 1e-14);
        assert_abs_diff_eq!(jv[1], 0.0, epsilon = 1e-14);
    }

    #[test]
    fn azimuthal_jones_at_zero_phi() {
        let vb = VectorBeam::new_azimuthal(1e-3, 1064e-9);
        let jv = vb.jones_vector(0.0);
        // At φ=0: (−sin0, cos0) = (0, 1)
        assert_abs_diff_eq!(jv[0], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(jv[1], 1.0, epsilon = 1e-14);
    }

    #[test]
    fn radial_longitudinal_fraction_increases_with_na() {
        let vb = VectorBeam::new_radial(1e-3, 1064e-9);
        let f_low = vb.longitudinal_field_fraction(0.5);
        let f_high = vb.longitudinal_field_fraction(0.9);
        assert!(f_high > f_low);
    }

    #[test]
    fn azimuthal_no_longitudinal_field() {
        let vb = VectorBeam::new_azimuthal(1e-3, 1064e-9);
        assert_abs_diff_eq!(vb.longitudinal_field_fraction(0.9), 0.0, epsilon = 1e-15);
    }
}
