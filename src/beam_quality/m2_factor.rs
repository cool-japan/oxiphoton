//! M² beam quality factor for real laser beams.
//!
//! The M² factor (beam quality factor, or beam propagation factor) characterises how
//! closely a laser beam resembles an ideal diffraction-limited Gaussian beam.
//!
//! For a real beam with waist w₀ and divergence half-angle θ:
//!
//! ```text
//!   M² = π w₀ θ / λ      (1 for ideal Gaussian)
//!   BPP = w₀ · θ = M² λ/π
//!   zR  = π w₀² / (M² λ)  (Rayleigh range of real beam)
//!   w(z) = w₀ √(1 + M²² (z−z₀)² / zR²)
//! ```
//!
//! Reference: ISO 11146-1:2021

use std::f64::consts::PI;

// Speed of light (m/s) — used for derived calculations
#[allow(dead_code)]
const C_LIGHT: f64 = 2.998_924_58e8;

/// M² beam quality factor and associated propagation parameters for a real laser beam.
///
/// `M² = 1` for an ideal diffraction-limited Gaussian beam; `M² > 1` for real beams.
///
/// Beam radius as a function of propagation distance z:
/// ```text
///   w(z)² = w₀² [ 1 + M²² (z − z₀)² / zR² ]
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BeamQuality {
    /// Vacuum wavelength (m).
    pub wavelength_m: f64,
    /// M² beam quality factor (≥ 1; 1 = ideal Gaussian).
    pub m2: f64,
    /// Beam waist radius w₀ at the focus position (m).
    pub beam_waist_m: f64,
    /// Axial position of the beam waist z₀ (m).
    pub waist_position_m: f64,
}

impl BeamQuality {
    /// Create a new `BeamQuality` instance.
    ///
    /// # Arguments
    /// * `wavelength_m` – Vacuum wavelength in metres.
    /// * `m2` – M² factor (must be ≥ 1; clamped to 1 if less).
    /// * `beam_waist_m` – Beam waist radius at focus (m).
    /// * `waist_position_m` – Axial position of the waist (m).
    pub fn new(wavelength_m: f64, m2: f64, beam_waist_m: f64, waist_position_m: f64) -> Self {
        Self {
            wavelength_m,
            m2: m2.max(1.0),
            beam_waist_m,
            waist_position_m,
        }
    }

    /// Rayleigh range of the real beam.
    ///
    /// `zR = π w₀² / (M² λ)`
    pub fn rayleigh_range_m(&self) -> f64 {
        PI * self.beam_waist_m * self.beam_waist_m / (self.m2 * self.wavelength_m)
    }

    /// Beam radius at axial position `z_m`.
    ///
    /// `w(z) = w₀ √( 1 + M²² (z − z₀)² / zR² )`
    pub fn beam_radius_at(&self, z_m: f64) -> f64 {
        let zr = self.rayleigh_range_m();
        let dz = z_m - self.waist_position_m;
        // M² already folded into zR; per ISO 11146 the propagation equation uses M⁴ in
        // the numerator when zR is the *geometric* Rayleigh range, so we write:
        //   w(z)² = w₀² [1 + (M² dz / zR)²]  with zR = π w₀² / λ (geometric)
        // Here zR is already the *real-beam* Rayleigh range (includes 1/M² factor), so:
        //   w(z)² = w₀² [1 + (dz / zR_real)²]
        let ratio = dz / zr;
        self.beam_waist_m * (1.0 + ratio * ratio).sqrt()
    }

    /// Far-field divergence half-angle (rad).
    ///
    /// `θ = M² λ / (π w₀)`
    pub fn divergence_half_angle_rad(&self) -> f64 {
        self.m2 * self.wavelength_m / (PI * self.beam_waist_m)
    }

    /// Beam parameter product (BPP).
    ///
    /// `BPP = w₀ · θ = M² λ / π`
    pub fn beam_parameter_product(&self) -> f64 {
        self.beam_waist_m * self.divergence_half_angle_rad()
    }

    /// Approximate Strehl ratio based on the M² factor.
    ///
    /// For an aberrated beam the Strehl ratio scales approximately as `S ≈ 1/M⁴`.
    pub fn strehl_ratio(&self) -> f64 {
        let m2sq = self.m2 * self.m2;
        1.0 / (m2sq * m2sq)
    }

    /// Focused spot radius after a thin lens (Gaussian beam optics).
    ///
    /// For a lens of focal length `f` and an input beam radius `w_in` at the lens:
    /// ```text
    ///   w_focus = M² λ f / (π w_in)
    /// ```
    ///
    /// # Arguments
    /// * `focal_length_m` – Focal length of the lens (m).
    /// * `input_beam_radius_m` – Input beam radius at the lens plane (m).
    pub fn focus_spot_through_lens(&self, focal_length_m: f64, input_beam_radius_m: f64) -> f64 {
        self.m2 * self.wavelength_m * focal_length_m / (PI * input_beam_radius_m)
    }

    /// Propagate the beam through an ABCD (ray-transfer) matrix.
    ///
    /// Uses the complex beam parameter `q = z - z₀ + i zR` and the standard
    /// ABCD transformation `q' = (A q + B) / (C q + D)`.
    ///
    /// The M² factor is invariant under propagation through ideal optical systems.
    ///
    /// # Arguments
    /// * `a, b, c, d` – Elements of the 2×2 ray-transfer matrix `[[A,B],[C,D]]`.
    ///
    /// # Returns
    /// A new `BeamQuality` with updated waist and waist position; M² unchanged.
    pub fn propagate_through_abcd(&self, a: f64, b: f64, c: f64, d: f64) -> BeamQuality {
        // Complex q at the current waist (z = z₀): q = i zR
        let zr = self.rayleigh_range_m();
        // q = 0 + i*zr  (at the waist, real part = 0 by convention)
        let qr = 0.0_f64; // real part of q
        let qi = zr; // imaginary part of q

        // q' = (A q + B) / (C q + D)
        // numerator: (A(qr + i qi) + B) = (A qr + B) + i A qi
        let nr = a * qr + b;
        let ni = a * qi;
        // denominator: (C(qr + i qi) + D) = (C qr + D) + i C qi
        let dr = c * qr + d;
        let di = c * qi;

        // q' = (nr + i ni) / (dr + i di)
        let denom_sq = dr * dr + di * di;
        // Guard against degenerate systems
        let qp_r = if denom_sq < 1e-300 {
            0.0
        } else {
            (nr * dr + ni * di) / denom_sq
        };
        let qp_i = if denom_sq < 1e-300 {
            zr
        } else {
            (ni * dr - nr * di) / denom_sq
        };

        // New Rayleigh range = Im(q')
        let zr_new = qp_i.abs();
        // New waist position relative to output plane = -Re(q') + original waist position
        let z0_new = self.waist_position_m + qp_r;

        // New waist radius from zR' = π w₀'² / (M² λ)
        let w0_new = ((zr_new * self.m2 * self.wavelength_m) / PI).sqrt();

        BeamQuality {
            wavelength_m: self.wavelength_m,
            m2: self.m2,
            beam_waist_m: w0_new,
            waist_position_m: z0_new,
        }
    }

    /// Approximate number of transverse spatial modes in the beam.
    ///
    /// `N_modes ≈ M²`
    pub fn n_transverse_modes(&self) -> f64 {
        self.m2
    }

    /// Encircled energy fraction within radius `r_m` at axial position `z_m`.
    ///
    /// For a Gaussian beam:
    /// ```text
    ///   EE(r) = 1 − exp(−2 r² / w(z)²)
    /// ```
    pub fn encircled_energy(&self, z_m: f64, r_m: f64) -> f64 {
        let w = self.beam_radius_at(z_m);
        1.0 - (-2.0 * r_m * r_m / (w * w)).exp()
    }
}

// ─── Mode-specific M² formulae ───────────────────────────────────────────────

/// M² factor for a Laguerre–Gaussian mode LG_p^l.
///
/// For radial index `p` and azimuthal index `l`:
/// ```text
///   M² = 2p + |l| + 1
/// ```
/// A TEM₀₀ (p=0, l=0) gives M² = 1 (ideal Gaussian).
pub fn laguerre_gaussian_m2(p: u32, l: i32) -> f64 {
    (2 * p + l.unsigned_abs() + 1) as f64
}

/// M² factors for a Hermite–Gaussian mode HG_mn.
///
/// ```text
///   M²_x = 2m + 1,  M²_y = 2n + 1
/// ```
/// Returns `(M²_x, M²_y)`.
pub fn hermite_gaussian_m2(m: u32, n: u32) -> (f64, f64) {
    ((2 * m + 1) as f64, (2 * n + 1) as f64)
}

/// Laser brightness (radiance) in W·m⁻²·sr⁻¹.
///
/// Defined as power per unit phase-space area (étendue):
/// ```text
///   B = P / (π · BPP)²  where BPP = M² λ / π
/// ```
///
/// # Arguments
/// * `power_w` – Laser output power (W).
/// * `m2` – M² beam quality factor.
/// * `wavelength_m` – Vacuum wavelength (m).
pub fn brightness_w_per_m2_sr(power_w: f64, m2: f64, wavelength_m: f64) -> f64 {
    let bpp = m2 * wavelength_m / PI;
    power_w / (PI * bpp * bpp)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Ideal Gaussian: BPP = λ/π
    #[test]
    fn ideal_gaussian_bpp() {
        let lambda = 1064e-9;
        let bq = BeamQuality::new(lambda, 1.0, 1e-3, 0.0);
        let bpp = bq.beam_parameter_product();
        let expected = lambda / PI;
        assert!(
            (bpp - expected).abs() / expected < 1e-10,
            "BPP mismatch: got {bpp}, expected {expected}"
        );
    }

    /// Rayleigh range for M²=1: zR = π w₀² / λ
    #[test]
    fn rayleigh_range_ideal() {
        let lambda = 1064e-9;
        let w0 = 1e-3;
        let bq = BeamQuality::new(lambda, 1.0, w0, 0.0);
        let zr = bq.rayleigh_range_m();
        let expected = PI * w0 * w0 / lambda;
        assert!(
            (zr - expected).abs() / expected < 1e-10,
            "zR mismatch: got {zr}, expected {expected}"
        );
    }

    /// At z = z₀ the beam radius equals the waist
    #[test]
    fn radius_at_waist() {
        let bq = BeamQuality::new(1064e-9, 1.5, 0.5e-3, 0.1);
        let w = bq.beam_radius_at(0.1);
        assert!(
            (w - bq.beam_waist_m).abs() < 1e-15,
            "Radius at waist should equal w₀, got {w}"
        );
    }

    /// At z = z₀ ± zR the beam radius is √2 · w₀
    #[test]
    fn radius_at_rayleigh_range() {
        let bq = BeamQuality::new(1064e-9, 1.0, 1e-3, 0.0);
        let zr = bq.rayleigh_range_m();
        let w = bq.beam_radius_at(zr);
        let expected = bq.beam_waist_m * 2.0_f64.sqrt();
        assert!(
            (w - expected).abs() / expected < 1e-10,
            "Radius at zR should be √2 w₀: got {w}, expected {expected}"
        );
    }

    /// LG mode M² values
    #[test]
    fn lg_mode_m2() {
        assert_eq!(laguerre_gaussian_m2(0, 0), 1.0); // TEM₀₀
        assert_eq!(laguerre_gaussian_m2(0, 1), 2.0); // donut beam
        assert_eq!(laguerre_gaussian_m2(1, 0), 3.0);
        assert_eq!(laguerre_gaussian_m2(1, 1), 4.0);
    }

    /// HG mode M² values
    #[test]
    fn hg_mode_m2() {
        assert_eq!(hermite_gaussian_m2(0, 0), (1.0, 1.0)); // TEM₀₀
        assert_eq!(hermite_gaussian_m2(1, 0), (3.0, 1.0));
        assert_eq!(hermite_gaussian_m2(2, 3), (5.0, 7.0));
    }

    /// Encircled energy: 86.5% within 1 waist radius
    #[test]
    fn encircled_energy_one_waist() {
        let bq = BeamQuality::new(1064e-9, 1.0, 1e-3, 0.0);
        let ee = bq.encircled_energy(0.0, bq.beam_waist_m);
        // 1 - exp(-2) ≈ 0.8647
        let expected = 1.0 - (-2.0_f64).exp();
        assert!(
            (ee - expected).abs() < 1e-10,
            "Encircled energy at w: got {ee}, expected {expected}"
        );
    }

    /// Propagation through free space (ABCD = [[1,L],[0,1]]) preserves M²
    #[test]
    fn abcd_free_space_propagation() {
        let bq = BeamQuality::new(1064e-9, 1.2, 0.5e-3, 0.0);
        let propagated = bq.propagate_through_abcd(1.0, 0.5, 0.0, 1.0);
        assert!(
            (propagated.m2 - bq.m2).abs() < 1e-12,
            "M² must be invariant under free-space propagation"
        );
    }

    /// Brightness scales as 1/M²²
    #[test]
    fn brightness_scales_with_m2() {
        let lambda = 1064e-9;
        let power = 1.0;
        let b1 = brightness_w_per_m2_sr(power, 1.0, lambda);
        let b2 = brightness_w_per_m2_sr(power, 2.0, lambda);
        // B ∝ 1/M⁴ since B = P / (M² λ/π)²
        let ratio = b1 / b2;
        assert!(
            (ratio - 4.0).abs() < 1e-10,
            "Brightness ratio M²=1 vs M²=2 should be 4, got {ratio}"
        );
    }
}
