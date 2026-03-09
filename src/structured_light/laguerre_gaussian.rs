//! Laguerre-Gaussian beams and Orbital Angular Momentum (OAM) optics.
//!
//! Implements LG_{p}^{l} beams with full field, intensity, Gouy phase,
//! Laguerre/Hermite polynomials, and OAM-related quantities. All computations
//! are pure Rust with no external linear-algebra or random-number dependencies.

use num_complex::Complex64;
use std::f64::consts::PI;

// Physical constant: reduced Planck constant (J·s)
const HBAR: f64 = 1.054_571_817e-34;

// ---------------------------------------------------------------------------
// Laguerre-Gaussian Beam
// ---------------------------------------------------------------------------

/// Laguerre-Gaussian beam LG_{p}^{l}.
///
/// The field in cylindrical coordinates (r, φ, z) is
///
/// u(r,φ,z) = C · (r√2/w)^|l| · L_p^|l|(2r²/w²) · exp(−r²/w²)
///            · exp(i l φ) · exp(i k z) · exp(−i ψ_G) · exp(−i k r²/(2R))
///
/// where w = w(z), R = R(z), ψ_G = Gouy phase, k = 2π/λ.
#[derive(Debug, Clone)]
pub struct LgBeam {
    /// Azimuthal index (OAM per photon = ℓℏ).  May be negative.
    pub ell: i32,
    /// Radial index (number of additional radial rings beyond the central one).
    pub p: usize,
    /// Beam waist w₀ at the focal plane (metres).
    pub beam_waist: f64,
    /// Free-space wavelength (metres).
    pub wavelength: f64,
    /// Total optical power (watts).
    pub power: f64,
}

impl LgBeam {
    /// Construct an LG beam.  `power` defaults to 1 W.
    pub fn new(ell: i32, p: usize, w0: f64, wavelength: f64) -> Self {
        Self {
            ell,
            p,
            beam_waist: w0,
            wavelength,
            power: 1.0,
        }
    }

    /// Rayleigh range z_R = π w₀² / λ.
    pub fn rayleigh_range(&self) -> f64 {
        PI * self.beam_waist * self.beam_waist / self.wavelength
    }

    /// Beam radius at propagation distance z.
    pub fn beam_radius(&self, z: f64) -> f64 {
        let zr = self.rayleigh_range();
        self.beam_waist * (1.0 + (z / zr).powi(2)).sqrt()
    }

    /// Wave-front radius of curvature R(z).  Returns `f64::INFINITY` at z = 0.
    fn radius_of_curvature(&self, z: f64) -> f64 {
        if z.abs() < 1e-30 {
            return f64::INFINITY;
        }
        let zr = self.rayleigh_range();
        z * (1.0 + (zr / z).powi(2))
    }

    /// Gouy phase ψ_G(z) = (2p + |l| + 1) · arctan(z / z_R).
    pub fn gouy_phase(&self, z: f64) -> f64 {
        let zr = self.rayleigh_range();
        let order = 2.0 * self.p as f64 + self.ell.unsigned_abs() as f64 + 1.0;
        order * (z / zr).atan()
    }

    /// Normalisation constant C such that ∫ |u|² dA = power.
    fn normalization(&self) -> f64 {
        let l_abs = self.ell.unsigned_abs() as f64;
        let _p = self.p as f64;
        // C² = 2 P / (π w₀²) · p! / (p + |l|)!
        let factorial_p = factorial(self.p);
        let factorial_p_plus_l = factorial(self.p + self.ell.unsigned_abs() as usize);
        (2.0 * self.power / (PI * self.beam_waist * self.beam_waist)
            * factorial_p as f64 / factorial_p_plus_l as f64)
            .sqrt()
        // The (r√2/w)^|l| factor in the field naturally scales with w(z), so the
        // normalisation above is evaluated at w₀ and remains power-conserving.
        // (Standard LG normalisation; see Siegman "Lasers", Appendix.)
        * (2.0_f64).powf(l_abs / 2.0) // absorb the √2 factor from (r√2/w)^|l|
    }

    /// Complex field amplitude u(r, φ, z).
    pub fn field(&self, r: f64, phi: f64, z: f64) -> Complex64 {
        let w = self.beam_radius(z);
        let _zr = self.rayleigh_range();
        let k = 2.0 * PI / self.wavelength;
        let l_abs = self.ell.unsigned_abs() as usize;

        // Radial amplitude factor (r/w)^|l| · exp(−r²/w²) · L_p^|l|(2r²/w²)
        let rho2 = 2.0 * r * r / (w * w);
        let radial = (r / w).powi(l_abs as i32)
            * (-r * r / (w * w)).exp()
            * Self::laguerre_polynomial(self.p, l_abs as f64, rho2);

        // Gouy phase and wave-front curvature
        let gouy = self.gouy_phase(z);
        let rc = self.radius_of_curvature(z);
        let curvature_phase = if rc.is_finite() {
            k * r * r / (2.0 * rc)
        } else {
            0.0
        };

        // w₀/w prefactor (from Gouy phase beam spreading — already in normalization? No:
        // the full normalisation including the w(z) dependence is w₀/w times the focal-plane norm)
        let w_factor = self.beam_waist / w;

        // Normalization constant (computed at z=0 reference)
        let c = self.normalization();

        // Phase accumulation
        let phase = k * z - gouy + (self.ell as f64) * phi - curvature_phase;

        // Combine real radial envelope × complex phase
        let amplitude = c * w_factor * radial;
        Complex64::from_polar(amplitude, phase)
    }

    /// Intensity I(r, φ, z) = |u|² (W/m²).
    pub fn intensity(&self, r: f64, phi: f64, z: f64) -> f64 {
        let u = self.field(r, phi, z);
        u.norm_sqr()
    }

    /// Radius of peak intensity ring at z: r_peak = w(z) √(|l|/2).
    pub fn peak_intensity_radius(&self, z: f64) -> f64 {
        let l_abs = self.ell.unsigned_abs() as f64;
        self.beam_radius(z) * (l_abs / 2.0).sqrt()
    }

    /// OAM per photon: L_z = ℓ ℏ (J·s).
    pub fn oam_per_photon(&self) -> f64 {
        self.ell as f64 * HBAR
    }

    /// Beam quality factor M² = 2p + |l| + 1.
    pub fn m_squared(&self) -> f64 {
        2.0 * self.p as f64 + self.ell.unsigned_abs() as f64 + 1.0
    }

    /// Generalised Laguerre polynomial L_p^α(x) computed via three-term recurrence.
    ///
    /// L_0^α(x) = 1
    /// L_1^α(x) = 1 + α − x
    /// L_{k+1}^α(x) = [(2k+1+α−x)·L_k^α(x) − (k+α)·L_{k-1}^α(x)] / (k+1)
    pub fn laguerre_polynomial(p: usize, alpha: f64, x: f64) -> f64 {
        if p == 0 {
            return 1.0;
        }
        let mut l_prev = 1.0_f64;
        let mut l_curr = 1.0 + alpha - x;
        for k in 1..p {
            let kf = k as f64;
            let l_next =
                ((2.0 * kf + 1.0 + alpha - x) * l_curr - (kf + alpha) * l_prev) / (kf + 1.0);
            l_prev = l_curr;
            l_curr = l_next;
        }
        l_curr
    }

    /// Decompose LG_{0}^{±1} into Hermite-Gaussian components.
    ///
    /// LG_{0}^{+1} = (HG_{1,0} + i·HG_{0,1}) / √2
    /// LG_{0}^{-1} = (HG_{1,0} − i·HG_{0,1}) / √2
    ///
    /// Returns `None` unless p == 0 and |ℓ| == 1.
    pub fn to_hg_components(&self) -> Option<Vec<(f64, HgBeam)>> {
        if self.p != 0 || self.ell.unsigned_abs() != 1 {
            return None;
        }
        let hg10 = HgBeam::new(1, 0, self.beam_waist, self.wavelength);
        let hg01 = HgBeam::new(0, 1, self.beam_waist, self.wavelength);
        // coefficients are 1/√2 for both; phase handled by the caller
        Some(vec![
            (1.0 / 2.0_f64.sqrt(), hg10),
            (1.0 / 2.0_f64.sqrt(), hg01),
        ])
    }
}

// ---------------------------------------------------------------------------
// Hermite-Gaussian Beam
// ---------------------------------------------------------------------------

/// Hermite-Gaussian beam HG_{m,n}.
///
/// u(x,y,z) = C · H_m(√2 x/w) · H_n(√2 y/w) · exp(−(x²+y²)/w²)
///            · exp(i k z) · exp(−i ψ_G) · exp(−i k(x²+y²)/(2R))
#[derive(Debug, Clone)]
pub struct HgBeam {
    /// Horizontal order m.
    pub m: usize,
    /// Vertical order n.
    pub n: usize,
    /// Beam waist w₀ (metres).
    pub beam_waist: f64,
    /// Free-space wavelength (metres).
    pub wavelength: f64,
    /// Total power (watts).
    pub power: f64,
}

impl HgBeam {
    /// Construct an HG beam with unit power.
    pub fn new(m: usize, n: usize, w0: f64, wavelength: f64) -> Self {
        Self {
            m,
            n,
            beam_waist: w0,
            wavelength,
            power: 1.0,
        }
    }

    fn rayleigh_range(&self) -> f64 {
        PI * self.beam_waist * self.beam_waist / self.wavelength
    }

    fn beam_radius(&self, z: f64) -> f64 {
        let zr = self.rayleigh_range();
        self.beam_waist * (1.0 + (z / zr).powi(2)).sqrt()
    }

    fn gouy_phase(&self, z: f64) -> f64 {
        let zr = self.rayleigh_range();
        let order = self.m as f64 + self.n as f64 + 1.0;
        order * (z / zr).atan()
    }

    fn radius_of_curvature(&self, z: f64) -> f64 {
        if z.abs() < 1e-30 {
            return f64::INFINITY;
        }
        let zr = self.rayleigh_range();
        z * (1.0 + (zr / z).powi(2))
    }

    /// Normalisation constant.
    fn normalization(&self) -> f64 {
        // C² = (2/π) · P / (w₀² · 2^(m+n) · m! · n!)
        let fm = factorial(self.m) as f64;
        let fn_ = factorial(self.n) as f64;
        let denom = PI / 2.0
            * self.beam_waist
            * self.beam_waist
            * 2.0_f64.powi((self.m + self.n) as i32)
            * fm
            * fn_;
        (self.power / denom).sqrt()
    }

    /// Complex field amplitude u(x, y, z).
    pub fn field(&self, x: f64, y: f64, z: f64) -> Complex64 {
        let w = self.beam_radius(z);
        let k = 2.0 * PI / self.wavelength;
        let rc = self.radius_of_curvature(z);
        let gouy = self.gouy_phase(z);

        let hm = Self::hermite_polynomial(self.m, 2.0_f64.sqrt() * x / w);
        let hn = Self::hermite_polynomial(self.n, 2.0_f64.sqrt() * y / w);

        let radial = (-(x * x + y * y) / (w * w)).exp();
        let curvature_phase = if rc.is_finite() {
            k * (x * x + y * y) / (2.0 * rc)
        } else {
            0.0
        };
        let w_factor = self.beam_waist / w;
        let c = self.normalization();
        let amplitude = c * w_factor * hm * hn * radial;
        let phase = k * z - gouy - curvature_phase;
        Complex64::from_polar(amplitude.abs(), phase) * amplitude.signum()
    }

    /// Intensity I(x, y, z) = |u|².
    pub fn intensity(&self, x: f64, y: f64, z: f64) -> f64 {
        self.field(x, y, z).norm_sqr()
    }

    /// Beam quality factor in x: M²_x = 2m + 1.
    pub fn m_squared_x(&self) -> f64 {
        2.0 * self.m as f64 + 1.0
    }

    /// Beam quality factor in y: M²_y = 2n + 1.
    pub fn m_squared_y(&self) -> f64 {
        2.0 * self.n as f64 + 1.0
    }

    /// Hermite polynomial H_n(x) via three-term recurrence.
    ///
    /// H_0(x) = 1,  H_1(x) = 2x
    /// H_{n+1}(x) = 2x·H_n(x) − 2n·H_{n-1}(x)
    pub fn hermite_polynomial(n: usize, x: f64) -> f64 {
        if n == 0 {
            return 1.0;
        }
        let mut h_prev = 1.0_f64;
        let mut h_curr = 2.0 * x;
        for k in 1..n {
            let h_next = 2.0 * x * h_curr - 2.0 * k as f64 * h_prev;
            h_prev = h_curr;
            h_curr = h_next;
        }
        h_curr
    }
}

// ---------------------------------------------------------------------------
// Optical Vortex
// ---------------------------------------------------------------------------

/// Optical vortex: a beam carrying a phase singularity with topological charge l.
#[derive(Debug, Clone)]
pub struct OpticalVortex {
    /// Topological charge (equals the OAM quantum number ℓ of the inner LG beam).
    pub topological_charge: i32,
    /// The underlying LG_{p=0}^{l} beam providing the amplitude envelope.
    pub beam: LgBeam,
}

impl OpticalVortex {
    /// Construct a pure doughnut optical vortex (p = 0) with the given charge.
    pub fn new(charge: i32, w0: f64, wavelength: f64) -> Self {
        let beam = LgBeam::new(charge, 0, w0, wavelength);
        Self {
            topological_charge: charge,
            beam,
        }
    }

    /// Azimuthal phase at angle φ: φ_vortex = charge · φ.
    pub fn phase(&self, phi: f64) -> f64 {
        self.topological_charge as f64 * phi
    }

    /// True if the beam is a pure doughnut (p = 0, hence no radial rings).
    pub fn is_pure_oam(&self) -> bool {
        self.beam.p == 0
    }

    /// Approximate vortex core radius ≈ w₀ / √|l|.
    /// Returns w₀ for zero charge (no vortex).
    pub fn vortex_core_radius(&self) -> f64 {
        let l_abs = self.topological_charge.unsigned_abs() as f64;
        if l_abs < 1e-10 {
            self.beam.beam_waist
        } else {
            self.beam.beam_waist / l_abs.sqrt()
        }
    }

    /// Number of spiral arms in the interference pattern with a plane wave = |charge|.
    pub fn spiral_pattern_arms(&self) -> usize {
        self.topological_charge.unsigned_abs() as usize
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Integer factorial n! (returns u64; fine for n ≤ 20).
fn factorial(n: usize) -> u64 {
    (1..=n as u64).product::<u64>().max(1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    const W0: f64 = 1e-3; // 1 mm beam waist
    const WL: f64 = 1064e-9; // 1064 nm

    // --- Laguerre polynomial ---
    #[test]
    fn laguerre_p0_is_one() {
        let val = LgBeam::laguerre_polynomial(0, 0.0, std::f64::consts::PI);
        assert_abs_diff_eq!(val, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn laguerre_p1_alpha0() {
        // L_1^0(x) = 1 − x
        let x = 0.5;
        let val = LgBeam::laguerre_polynomial(1, 0.0, x);
        assert_abs_diff_eq!(val, 1.0 - x, epsilon = 1e-14);
    }

    #[test]
    fn laguerre_p2_alpha0() {
        // L_2^0(x) = 1 − 2x + x²/2
        let x = 1.0;
        let val = LgBeam::laguerre_polynomial(2, 0.0, x);
        let expected = 1.0 - 2.0 * x + x * x / 2.0;
        assert_abs_diff_eq!(val, expected, epsilon = 1e-14);
    }

    // --- Hermite polynomial ---
    #[test]
    fn hermite_h0_is_one() {
        let val = HgBeam::hermite_polynomial(0, 1.23);
        assert_abs_diff_eq!(val, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn hermite_h1() {
        // H_1(x) = 2x
        let x = 2.5;
        assert_abs_diff_eq!(HgBeam::hermite_polynomial(1, x), 2.0 * x, epsilon = 1e-14);
    }

    #[test]
    fn hermite_h2() {
        // H_2(x) = 4x² − 2
        let x = 1.0;
        assert_abs_diff_eq!(
            HgBeam::hermite_polynomial(2, x),
            4.0 * x * x - 2.0,
            epsilon = 1e-14
        );
    }

    // --- LgBeam geometry ---
    #[test]
    fn rayleigh_range() {
        let beam = LgBeam::new(1, 0, W0, WL);
        let zr = beam.rayleigh_range();
        assert_abs_diff_eq!(zr, PI * W0 * W0 / WL, epsilon = 1e-6);
    }

    #[test]
    fn beam_radius_at_focus() {
        let beam = LgBeam::new(0, 0, W0, WL);
        assert_abs_diff_eq!(beam.beam_radius(0.0), W0, epsilon = 1e-15);
    }

    #[test]
    fn m_squared_fundamental() {
        let beam = LgBeam::new(0, 0, W0, WL);
        // LG_{0}^{0} = TEM00 → M² = 1
        assert_abs_diff_eq!(beam.m_squared(), 1.0, epsilon = 1e-15);
    }

    #[test]
    fn oam_per_photon_sign() {
        let beam_pos = LgBeam::new(2, 0, W0, WL);
        let beam_neg = LgBeam::new(-2, 0, W0, WL);
        assert!(beam_pos.oam_per_photon() > 0.0);
        assert!(beam_neg.oam_per_photon() < 0.0);
        assert_abs_diff_eq!(
            beam_pos.oam_per_photon(),
            -beam_neg.oam_per_photon(),
            epsilon = 1e-45
        );
    }

    // --- OpticalVortex ---
    #[test]
    fn vortex_spiral_arms() {
        let v = OpticalVortex::new(3, W0, WL);
        assert_eq!(v.spiral_pattern_arms(), 3);
    }

    #[test]
    fn vortex_is_pure_oam() {
        let v = OpticalVortex::new(1, W0, WL);
        assert!(v.is_pure_oam());
    }

    #[test]
    fn vortex_core_radius_positive_charge() {
        let v = OpticalVortex::new(4, W0, WL);
        let expected = W0 / (4.0_f64.sqrt()); // w0/√|l| = w0/2 for l=4
        assert_abs_diff_eq!(v.vortex_core_radius(), expected, epsilon = 1e-10);
    }

    // --- HgBeam ---
    #[test]
    fn hg_m_squared() {
        let hg = HgBeam::new(2, 3, W0, WL);
        assert_abs_diff_eq!(hg.m_squared_x(), 5.0, epsilon = 1e-15);
        assert_abs_diff_eq!(hg.m_squared_y(), 7.0, epsilon = 1e-15);
    }

    #[test]
    fn lg_to_hg_decomposition() {
        let lg = LgBeam::new(1, 0, W0, WL);
        let comps = lg.to_hg_components();
        assert!(comps.is_some());
        let v = comps.unwrap();
        assert_eq!(v.len(), 2);
        // Both coefficients = 1/√2
        assert_abs_diff_eq!(v[0].0, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-14);
        assert_abs_diff_eq!(v[1].0, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-14);
    }

    #[test]
    fn lg_to_hg_none_for_higher_order() {
        let lg = LgBeam::new(2, 0, W0, WL);
        assert!(lg.to_hg_components().is_none());
    }

    #[test]
    fn gouy_phase_at_rayleigh() {
        let beam = LgBeam::new(1, 0, W0, WL);
        let zr = beam.rayleigh_range();
        // ψ_G(zR) = (2·0 + 1 + 1) · π/4 = π/2
        let expected = (2.0 * 0.0 + 1.0 + 1.0) * PI / 4.0;
        assert_abs_diff_eq!(beam.gouy_phase(zr), expected, epsilon = 1e-14);
    }
}
