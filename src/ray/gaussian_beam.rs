use std::f64::consts::PI;

/// ABCD (ray transfer) matrix for paraxial optics.
///
/// Represents the 2×2 matrix [[A, B], [C, D]] that transforms a ray vector
/// (height y, angle u) as: [y', u'] = M * [y, u]ᵀ
///
/// Reference: Saleh & Teich, "Fundamentals of Photonics", Ch. 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AbcdMatrix {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

impl AbcdMatrix {
    /// Identity (free ray, no transformation)
    pub fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
        }
    }

    /// Free-space propagation over distance `d` (m)
    pub fn free_space(d: f64) -> Self {
        Self {
            a: 1.0,
            b: d,
            c: 0.0,
            d: 1.0,
        }
    }

    /// Thin lens with focal length `f` (m). Use negative f for diverging.
    pub fn thin_lens(f: f64) -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: -1.0 / f,
            d: 1.0,
        }
    }

    /// Flat interface between media n1 and n2 (paraxial Snell's law)
    pub fn flat_interface(n1: f64, n2: f64) -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: n1 / n2,
        }
    }

    /// Curved interface with radius `r` (positive = center on right) and indices n1→n2
    pub fn curved_interface(r: f64, n1: f64, n2: f64) -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: (n1 - n2) / (r * n2),
            d: n1 / n2,
        }
    }

    /// Matrix multiplication: apply `self` after `other` (other acts first)
    pub fn then(&self, other: &AbcdMatrix) -> AbcdMatrix {
        AbcdMatrix {
            a: self.a * other.a + self.b * other.c,
            b: self.a * other.b + self.b * other.d,
            c: self.c * other.a + self.d * other.c,
            d: self.c * other.b + self.d * other.d,
        }
    }

    /// Determinant (should be n1/n2 for lossless systems, 1.0 for same medium)
    pub fn det(&self) -> f64 {
        self.a * self.d - self.b * self.c
    }
}

/// Gaussian beam parameters at a given cross-section.
///
/// A Gaussian beam is characterised by its complex beam parameter (q-parameter):
///   1/q = 1/R - i·λ/(π·w²·n)
///
/// where R is the radius of curvature, w is the 1/e² intensity beam radius,
/// n is the medium index, and λ is the free-space wavelength.
#[derive(Debug, Clone, Copy)]
pub struct GaussianBeam {
    /// 1/e² beam radius (m)
    pub w: f64,
    /// Radius of curvature of the wavefront (m). Positive = diverging.
    /// Use f64::INFINITY for a flat wavefront (beam waist).
    pub r: f64,
    /// Medium refractive index
    pub n: f64,
    /// Free-space wavelength (m)
    pub wavelength: f64,
}

impl GaussianBeam {
    /// Create a Gaussian beam at its waist (flat wavefront, minimum width).
    pub fn at_waist(w0: f64, wavelength: f64, n: f64) -> Self {
        Self {
            w: w0,
            r: f64::INFINITY,
            n,
            wavelength,
        }
    }

    /// Rayleigh range (m): z_R = π·n·w₀²/λ
    pub fn rayleigh_range(&self) -> f64 {
        PI * self.n * self.w * self.w / self.wavelength
    }

    /// Divergence half-angle (rad): θ = λ/(π·n·w₀)
    pub fn divergence(&self) -> f64 {
        self.wavelength / (PI * self.n * self.w)
    }

    /// Complex beam parameter q = 1/(1/R - i·λ/(π·n·w²))
    pub fn q_parameter(&self) -> num_complex::Complex64 {
        let lambda_n = self.wavelength / self.n;
        let inv_q_re = if self.r.is_infinite() {
            0.0
        } else {
            1.0 / self.r
        };
        let inv_q_im = -lambda_n / (PI * self.w * self.w);
        let inv_q = num_complex::Complex64::new(inv_q_re, inv_q_im);
        num_complex::Complex64::new(1.0, 0.0) / inv_q
    }

    /// Propagate the beam through an ABCD system.
    ///
    /// Applies the ABCD law for Gaussian beams:
    ///   q' = (A·q + B) / (C·q + D)
    pub fn propagate(&self, m: &AbcdMatrix) -> Self {
        let q = self.q_parameter();
        let q_new = (m.a * q + m.b) / (m.c * q + m.d);

        // Extract w and R from new q
        let inv_q_new = num_complex::Complex64::new(1.0, 0.0) / q_new;
        let r_new = if inv_q_new.re.abs() < 1e-30 {
            f64::INFINITY
        } else {
            1.0 / inv_q_new.re
        };
        let lambda_n = self.wavelength / self.n;
        let w_new = (lambda_n / (PI * (-inv_q_new.im).abs())).sqrt();

        Self {
            w: w_new,
            r: r_new,
            n: self.n,
            wavelength: self.wavelength,
        }
    }

    /// Propagate through free space by distance `z` (m).
    pub fn propagate_free(&self, z: f64) -> Self {
        let m = AbcdMatrix::free_space(z);
        self.propagate(&m)
    }

    /// Apply a thin lens of focal length `f`.
    pub fn apply_lens(&self, f: f64) -> Self {
        let m = AbcdMatrix::thin_lens(f);
        self.propagate(&m)
    }

    /// Intensity profile |E(r)|² at transverse radius r from axis.
    pub fn intensity_profile(&self, r: f64, i0: f64) -> f64 {
        i0 * (-2.0 * r * r / (self.w * self.w)).exp()
    }

    /// Peak on-axis intensity given total power P.
    pub fn peak_intensity(&self, power: f64) -> f64 {
        2.0 * power / (PI * self.w * self.w)
    }
}

/// Beam focus after a thin lens: position and waist size.
///
/// For a beam incident on a lens, computes the image-side focus.
/// - Input beam waist w0 at distance `d_in` before the lens.
/// - Returns (d_out, w_focus): image distance and focus waist.
pub fn focus_gaussian(w0: f64, d_in: f64, f: f64, wavelength: f64, n: f64) -> (f64, f64) {
    let beam = GaussianBeam::at_waist(w0, wavelength, n);
    // Propagate to lens
    let at_lens = beam.propagate_free(d_in);
    // Apply lens
    let after_lens = at_lens.apply_lens(f);
    // Find waist: propagate until R = ∞ (waist), i.e., 1/q is purely imaginary.
    // The waist position satisfies: d_out = Re(q') / (|q'|² / Im(q') ... etc.
    // Analytically: d_out = -Re(q') where q' = q_after_lens
    let q_after = after_lens.q_parameter();
    let d_out = q_after.re;
    let w_focus = after_lens.propagate_free(d_out).w;
    (d_out, w_focus)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abcd_identity() {
        let m = AbcdMatrix::identity();
        assert_eq!(m.a, 1.0);
        assert_eq!(m.d, 1.0);
        assert_eq!(m.b, 0.0);
        assert_eq!(m.c, 0.0);
    }

    #[test]
    fn abcd_free_space_det() {
        let m = AbcdMatrix::free_space(1e-3);
        assert!((m.det() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn abcd_thin_lens_det() {
        let m = AbcdMatrix::thin_lens(50e-3);
        assert!((m.det() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn abcd_cascade() {
        // Free space + identity = free space
        let m1 = AbcdMatrix::free_space(10e-3);
        let m2 = AbcdMatrix::identity();
        let combined = m1.then(&m2);
        assert!((combined.b - 10e-3).abs() < 1e-12);
    }

    #[test]
    fn gaussian_beam_rayleigh_range() {
        let w0 = 1e-3; // 1mm waist
        let wl = 1550e-9;
        let beam = GaussianBeam::at_waist(w0, wl, 1.0);
        let zr = beam.rayleigh_range();
        let expected = PI * w0 * w0 / wl;
        assert!((zr - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn gaussian_propagation_doubles_at_rayleigh() {
        let w0 = 1e-3;
        let wl = 1550e-9;
        let beam = GaussianBeam::at_waist(w0, wl, 1.0);
        let zr = beam.rayleigh_range();
        let propagated = beam.propagate_free(zr);
        let expected_w = w0 * 2.0_f64.sqrt();
        let rel_err = (propagated.w - expected_w).abs() / expected_w;
        assert!(
            rel_err < 1e-6,
            "w at z_R should be w0√2, got {:.4e}",
            propagated.w
        );
    }

    #[test]
    fn gaussian_thin_lens_focuses() {
        let w0 = 1e-3;
        let f = 50e-3;
        let wl = 633e-9;
        let beam = GaussianBeam::at_waist(w0, wl, 1.0);
        let after_lens = beam.apply_lens(f);
        // After lens, beam should converge (R < 0)
        assert!(after_lens.r < 0.0, "Lens should make beam converge");
    }

    #[test]
    fn gaussian_intensity_at_center() {
        let beam = GaussianBeam::at_waist(1e-3, 633e-9, 1.0);
        let power = 1e-3; // 1mW
        let i_peak = beam.peak_intensity(power);
        let i_center = beam.intensity_profile(0.0, i_peak);
        assert!((i_center - i_peak).abs() < 1e-12);
    }

    #[test]
    fn gaussian_intensity_at_1_over_e2() {
        // At r = w, intensity drops to 1/e²
        let beam = GaussianBeam::at_waist(1e-3, 633e-9, 1.0);
        let i0 = 1.0;
        let i_edge = beam.intensity_profile(beam.w, i0);
        let expected = i0 * (-2.0f64).exp();
        assert!((i_edge - expected).abs() < 1e-12);
    }
}
