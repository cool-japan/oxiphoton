use std::f64::consts::PI;

/// Metalens phase profile design.
///
/// A metalens focuses light by imposing a spatially varying phase profile on the wavefront.
/// For a flat lens with focal length f, the ideal phase profile is:
///   φ(r) = -(2π/λ) · (sqrt(r² + f²) - f)
///
/// This converts a normally incident plane wave into a converging spherical wave.
/// The phase is wrapped to [0, 2π] and approximated by discrete nanoscatterers
/// (nanoposts, nanopillars) whose geometry is mapped to the required local phase.
///
/// Reference: Khorasaninejad et al., Science 352(6290), 2016.
///
/// Phase profile for a focusing metalens.
///
/// The phase at radius r from the lens center for wavelength λ and focal length f:
///   φ(r) = -(2π/λ) · (sqrt(r² + f²) - f)   (modulo 2π)
pub struct MetalensPhaseFocusing {
    /// Focal length (m)
    pub focal_length: f64,
    /// Design wavelength (m)
    pub wavelength: f64,
    /// Lens aperture radius (m)
    pub aperture_radius: f64,
}

impl MetalensPhaseFocusing {
    pub fn new(focal_length: f64, wavelength: f64, aperture_radius: f64) -> Self {
        Self {
            focal_length,
            wavelength,
            aperture_radius,
        }
    }

    /// Phase profile φ(r) at radius r (m), range [−∞, 0] (not wrapped).
    pub fn phase_continuous(&self, r: f64) -> f64 {
        let f = self.focal_length;
        let k0 = 2.0 * PI / self.wavelength;
        -k0 * ((r * r + f * f).sqrt() - f)
    }

    /// Phase profile wrapped to [0, 2π].
    pub fn phase_wrapped(&self, r: f64) -> f64 {
        self.phase_continuous(r).rem_euclid(2.0 * PI)
    }

    /// Discretize the phase profile onto a grid of n_pts radial points.
    ///
    /// Returns Vec<(r, phi_wrapped)>.
    pub fn phase_profile(&self, n_pts: usize) -> Vec<(f64, f64)> {
        let dr = self.aperture_radius / n_pts as f64;
        (0..n_pts)
            .map(|i| {
                let r = (i as f64 + 0.5) * dr;
                let phi = self.phase_wrapped(r);
                (r, phi)
            })
            .collect()
    }

    /// Numerical aperture: NA = sin(atan(aperture_radius / focal_length))
    pub fn numerical_aperture(&self) -> f64 {
        let theta = (self.aperture_radius / self.focal_length).atan();
        theta.sin()
    }

    /// F-number: f_num = f / D = f / (2·a)
    pub fn f_number(&self) -> f64 {
        self.focal_length / (2.0 * self.aperture_radius)
    }

    /// Ideal Airy disk radius (first zero of Airy pattern): r_Airy = 1.22·λ·f_num
    pub fn airy_disk_radius(&self) -> f64 {
        1.22 * self.wavelength * self.f_number()
    }

    /// Zone radius for the m-th Fresnel zone: r_m = sqrt(m·λ·f + (m·λ/2)²)
    /// Approximation: r_m ≈ sqrt(m·λ·f) for m·λ << f
    pub fn fresnel_zone_radius(&self, m: usize) -> f64 {
        let ml = m as f64 * self.wavelength;
        (ml * self.focal_length + (ml / 2.0).powi(2)).sqrt()
    }

    /// Number of phase wraps (zones) across the aperture: N_zones ≈ a²/(λ·f)
    pub fn n_zones(&self) -> usize {
        let n = self.aperture_radius * self.aperture_radius / (self.wavelength * self.focal_length);
        n.floor() as usize
    }

    /// Phase gradient at radius r: dφ/dr = -(2π/λ) · r / sqrt(r² + f²)
    /// This gives the local grating vector K(r) = dφ/dr that the nanopost must encode.
    pub fn phase_gradient(&self, r: f64) -> f64 {
        let f = self.focal_length;
        let k0 = 2.0 * PI / self.wavelength;
        -k0 * r / (r * r + f * f).sqrt()
    }
}

/// Holographic grating (deflecting) metalens: deflects normally incident beam by angle θ.
///
/// Required phase: φ(x) = (2π/λ)·sin(θ)·x  (linear phase ramp = prism)
pub struct MetalensDeflector {
    /// Deflection angle (rad)
    pub theta: f64,
    /// Design wavelength (m)
    pub wavelength: f64,
    /// Required grating period: Λ = λ/sin(θ)
    pub period: f64,
}

impl MetalensDeflector {
    pub fn new(theta: f64, wavelength: f64) -> Self {
        let period = if theta.abs() < 1e-10 {
            f64::INFINITY
        } else {
            wavelength / theta.abs().sin()
        };
        Self {
            theta,
            wavelength,
            period,
        }
    }

    /// Phase at transverse position x: φ(x) = (2π/λ)·sin(θ)·x (mod 2π)
    pub fn phase_at(&self, x: f64) -> f64 {
        let k0 = 2.0 * PI / self.wavelength;
        let phi = k0 * self.theta.sin() * x;
        phi.rem_euclid(2.0 * PI)
    }

    /// Grating vector magnitude: K = 2π/Λ = k₀·sin(θ)
    pub fn grating_vector(&self) -> f64 {
        2.0 * PI / self.period
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metalens_phase_at_center_is_zero() {
        let ml = MetalensPhaseFocusing::new(10e-3, 532e-9, 1e-3);
        // At r=0: φ = -(2π/λ)(f - f) = 0
        let phi = ml.phase_continuous(0.0);
        assert!(
            phi.abs() < 1e-10,
            "Phase at center should be 0, got {phi:.2e}"
        );
    }

    #[test]
    fn metalens_phase_decreases_with_radius() {
        let ml = MetalensPhaseFocusing::new(10e-3, 532e-9, 1e-3);
        let phi0 = ml.phase_continuous(0.0);
        let phi1 = ml.phase_continuous(0.5e-3);
        assert!(
            phi1 < phi0,
            "Phase should decrease (become more negative) with r"
        );
    }

    #[test]
    fn metalens_wrapped_in_range() {
        let ml = MetalensPhaseFocusing::new(10e-3, 532e-9, 1e-3);
        for i in 0..20 {
            let r = i as f64 * 0.05e-3;
            let phi = ml.phase_wrapped(r);
            assert!(
                (0.0..2.0 * PI).contains(&phi),
                "Wrapped phase={phi:.4} out of [0,2π)"
            );
        }
    }

    #[test]
    fn metalens_na_physical() {
        let ml = MetalensPhaseFocusing::new(10e-3, 532e-9, 1e-3);
        let na = ml.numerical_aperture();
        assert!(na > 0.0 && na < 1.0, "NA={na:.3}");
    }

    #[test]
    fn metalens_airy_disk_positive() {
        let ml = MetalensPhaseFocusing::new(10e-3, 532e-9, 1e-3);
        let r_airy = ml.airy_disk_radius();
        assert!(r_airy > 0.0);
    }

    #[test]
    fn fresnel_zone_radii_increasing() {
        let ml = MetalensPhaseFocusing::new(10e-3, 532e-9, 2e-3);
        let r1 = ml.fresnel_zone_radius(1);
        let r2 = ml.fresnel_zone_radius(2);
        let r3 = ml.fresnel_zone_radius(3);
        assert!(r1 < r2 && r2 < r3);
    }

    #[test]
    fn phase_gradient_at_center_zero() {
        let ml = MetalensPhaseFocusing::new(10e-3, 532e-9, 1e-3);
        let grad = ml.phase_gradient(0.0);
        assert!(grad.abs() < 1e-10);
    }

    #[test]
    fn phase_gradient_magnitude_increases() {
        let ml = MetalensPhaseFocusing::new(10e-3, 532e-9, 2e-3);
        let g1 = ml.phase_gradient(0.5e-3).abs();
        let g2 = ml.phase_gradient(1.0e-3).abs();
        assert!(g2 > g1);
    }

    #[test]
    fn deflector_period_physical() {
        let theta = 15.0_f64.to_radians();
        let d = MetalensDeflector::new(theta, 532e-9);
        assert!(d.period > 0.0 && d.period > d.wavelength);
    }

    #[test]
    fn deflector_phase_periodic() {
        let theta = 15.0_f64.to_radians();
        let d = MetalensDeflector::new(theta, 532e-9);
        let phi0 = d.phase_at(0.0);
        let phi1 = d.phase_at(d.period);
        // After one period, phase should be 0 (mod 2π) → wrapped to same
        assert!(
            (phi0 - phi1).abs() < 1e-6
                || (phi0 - phi1 + 2.0 * PI).abs() < 1e-6
                || (phi0 - phi1 - 2.0 * PI).abs() < 1e-6
        );
    }
}
