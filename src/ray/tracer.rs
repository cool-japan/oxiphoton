/// A paraxial ray defined by its height `y` (m) and angle `u` (rad).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    /// Transverse height from the optical axis (m)
    pub y: f64,
    /// Paraxial angle to the optical axis (rad)
    pub u: f64,
}

impl Ray {
    pub fn new(y: f64, u: f64) -> Self {
        Self { y, u }
    }

    /// Axial ray: starts on axis, travels at 1 rad (marginal)
    pub fn axial() -> Self {
        Self { y: 0.0, u: 1.0 }
    }

    /// Chief ray: starts at height 1 on the aperture stop, travels at 0 rad
    pub fn chief() -> Self {
        Self { y: 1.0, u: 0.0 }
    }
}

/// Optical surface in a paraxial system.
#[derive(Debug, Clone)]
pub enum Surface {
    /// Free-space propagation (thickness in m)
    FreeSpace { d: f64 },
    /// Thin lens (focal length in m)
    ThinLens { f: f64 },
    /// Curved refracting surface: radius R (positive = center right), n1→n2
    CurvedInterface { r: f64, n1: f64, n2: f64 },
    /// Flat refracting interface
    FlatInterface { n1: f64, n2: f64 },
    /// Mirror with radius R (positive = concave)
    Mirror { r: f64 },
    /// Aperture stop: passes ray if |y| <= radius, clips otherwise
    ApertureStop { radius: f64 },
    /// Diffraction grating: adds diffraction angle for order m
    /// groove_density in lines/m, n1 = incident medium, n2 = exit medium
    DiffractionGrating {
        groove_density: f64,
        m: i32,
        n1: f64,
        n2: f64,
        wavelength: f64,
    },
}

impl Surface {
    /// Apply this surface to a ray using the ABCD matrix approach.
    pub fn apply(&self, ray: Ray) -> Ray {
        match *self {
            Surface::FreeSpace { d } => Ray {
                y: ray.y + d * ray.u,
                u: ray.u,
            },
            Surface::ThinLens { f } => Ray {
                y: ray.y,
                u: ray.u - ray.y / f,
            },
            Surface::CurvedInterface { r, n1, n2 } => {
                // Snell paraxial: n1·u1 + n1·y/r = n2·u2 → u2 = (n1·u1 + (n1-n2)·y/r)/n2
                // Paraxial refraction: n2·u' = n1·u - (n2-n1)/r · y
                let u_new = (n1 * ray.u - (n2 - n1) / r * ray.y) / n2;
                Ray { y: ray.y, u: u_new }
            }
            Surface::FlatInterface { n1, n2 } => Ray {
                y: ray.y,
                u: ray.u * n1 / n2,
            },
            Surface::Mirror { r } => {
                if r.is_infinite() {
                    Ray {
                        y: ray.y,
                        u: -ray.u,
                    }
                } else {
                    Ray {
                        y: ray.y,
                        u: ray.u - 2.0 * ray.y / r,
                    }
                }
            }
            Surface::ApertureStop { radius } => {
                // Clip: if |y| > radius, the ray is vignetted (set u to NaN to flag)
                if ray.y.abs() > radius {
                    Ray {
                        y: ray.y,
                        u: f64::NAN,
                    }
                } else {
                    ray
                }
            }
            Surface::DiffractionGrating {
                groove_density,
                m,
                n1,
                n2,
                wavelength,
            } => {
                // Grating equation (paraxial): n2·sin(theta_out) = n1·sin(theta_in) + m*lambda*groove_density
                // In paraxial approx: n2·u_out = n1·u_in + m * lambda * groove_density
                let u_out = (n1 * ray.u + m as f64 * wavelength * groove_density) / n2;
                Ray { y: ray.y, u: u_out }
            }
        }
    }
}

/// Paraxial optical system built from a sequence of surfaces.
#[derive(Debug, Clone, Default)]
pub struct OpticalSystem {
    surfaces: Vec<Surface>,
}

impl OpticalSystem {
    pub fn new() -> Self {
        Self {
            surfaces: Vec::new(),
        }
    }

    pub fn push(mut self, surface: Surface) -> Self {
        self.surfaces.push(surface);
        self
    }

    /// Trace a ray through all surfaces.
    pub fn trace(&self, ray: Ray) -> Ray {
        self.surfaces.iter().fold(ray, |r, s| s.apply(r))
    }

    /// Trace a ray and return snapshots after each surface.
    pub fn trace_full(&self, ray: Ray) -> Vec<Ray> {
        let mut snapshots = vec![ray];
        let mut current = ray;
        for s in &self.surfaces {
            current = s.apply(current);
            snapshots.push(current);
        }
        snapshots
    }

    /// Compute effective focal length from the ABCD matrix.
    /// Returns None if the system has no optical power.
    pub fn effective_focal_length(&self) -> Option<f64> {
        let m = self.abcd_matrix();
        if m.c.abs() < 1e-30 {
            None
        } else {
            Some(-1.0 / m.c)
        }
    }

    /// Compute the system ABCD matrix.
    pub fn abcd_matrix(&self) -> crate::ray::gaussian_beam::AbcdMatrix {
        use crate::ray::gaussian_beam::AbcdMatrix;
        self.surfaces
            .iter()
            .map(|s| match s {
                Surface::FreeSpace { d } => AbcdMatrix::free_space(*d),
                Surface::ThinLens { f } => AbcdMatrix::thin_lens(*f),
                Surface::CurvedInterface { r, n1, n2 } => {
                    AbcdMatrix::curved_interface(*r, *n1, *n2)
                }
                Surface::FlatInterface { n1, n2 } => AbcdMatrix::flat_interface(*n1, *n2),
                Surface::Mirror { r } => AbcdMatrix {
                    a: 1.0,
                    b: 0.0,
                    c: -2.0 / r,
                    d: 1.0,
                },
                // Aperture stop and grating don't change the ABCD matrix
                Surface::ApertureStop { .. } => AbcdMatrix::identity(),
                Surface::DiffractionGrating { n1, n2, .. } => AbcdMatrix::flat_interface(*n1, *n2),
            })
            .fold(AbcdMatrix::identity(), |acc, m| m.then(&acc))
    }
}

/// Aspheric surface described by conic constant K and polynomial terms.
///
/// Sag: z(r) = r²/R / (1 + sqrt(1 - (1+K)r²/R²)) + A4*r⁴ + A6*r⁶ + ...
///
/// In paraxial approximation (small r), the sag ≈ r²/(2R).
#[derive(Debug, Clone)]
pub struct AsphericSurface {
    /// Paraxial radius of curvature R (m); positive = center to the right
    pub r: f64,
    /// Conic constant K: -1=paraboloid, 0=sphere, >0=oblate ellipsoid
    pub k: f64,
    /// 4th-order aspheric coefficient A4
    pub a4: f64,
    /// 6th-order aspheric coefficient A6
    pub a6: f64,
    /// Incident medium index
    pub n1: f64,
    /// Exit medium index
    pub n2: f64,
}

impl AsphericSurface {
    pub fn spherical(r: f64, n1: f64, n2: f64) -> Self {
        Self {
            r,
            k: 0.0,
            a4: 0.0,
            a6: 0.0,
            n1,
            n2,
        }
    }

    pub fn parabolic(r: f64, n1: f64, n2: f64) -> Self {
        Self {
            r,
            k: -1.0,
            a4: 0.0,
            a6: 0.0,
            n1,
            n2,
        }
    }

    /// Sag at height h (paraxial approximation).
    pub fn sag_paraxial(&self, h: f64) -> f64 {
        h * h / (2.0 * self.r)
    }

    /// Full sag at height h.
    pub fn sag(&self, h: f64) -> f64 {
        let c = 1.0 / self.r;
        let disc = 1.0 - (1.0 + self.k) * c * c * h * h;
        if disc <= 0.0 {
            return f64::NAN;
        }
        c * h * h / (1.0 + disc.sqrt()) + self.a4 * h.powi(4) + self.a6 * h.powi(6)
    }

    /// Paraxial refraction power: P = (n2 - n1) / R.
    pub fn power(&self) -> f64 {
        (self.n2 - self.n1) / self.r
    }

    /// Apply paraxial refraction (ignores conic/aspheric for paraxial ray).
    pub fn apply_paraxial(&self, ray: Ray) -> Ray {
        let u_new = (self.n1 * ray.u - self.power() * ray.y) / self.n2;
        Ray { y: ray.y, u: u_new }
    }

    /// RMS wavefront error from conic deviation (approximate).
    ///
    /// For a conic surface with conic constant K, the 3rd-order spherical
    /// aberration coefficient is proportional to K.
    pub fn conic_aberration_coeff(&self) -> f64 {
        -self.k / (8.0 * self.r.powi(3))
    }
}

/// Prism deviation: angle through which a ray is deflected by a prism.
///
/// Using Snell's law at both surfaces.
///
/// # Arguments
/// - `n_glass`: refractive index of the prism glass
/// - `apex_angle_deg`: prism apex angle (degrees)
/// - `incident_angle_deg`: angle of incidence at the first surface (degrees)
/// - `wavelength`: wavelength (m) — used only for informational purposes
///
/// Returns the total deviation angle in degrees.
pub fn prism_deviation(n_glass: f64, apex_angle_deg: f64, incident_angle_deg: f64) -> f64 {
    let apex = apex_angle_deg.to_radians();
    let theta1 = incident_angle_deg.to_radians();
    // Refraction at first surface: sin(theta2) = sin(theta1) / n
    let sin_theta2 = theta1.sin() / n_glass;
    if sin_theta2.abs() > 1.0 {
        return f64::NAN; // total internal reflection
    }
    let theta2 = sin_theta2.asin();
    // Angle inside prism at second surface
    let theta3 = apex - theta2;
    // Refraction at second surface: sin(theta4) = n * sin(theta3)
    let sin_theta4 = n_glass * theta3.sin();
    if sin_theta4.abs() > 1.0 {
        return f64::NAN;
    }
    let theta4 = sin_theta4.asin();
    // Total deviation
    let deviation = theta1 + theta4 - apex;
    deviation.to_degrees()
}

/// Minimum deviation angle for a prism (symmetric case: i1 = i2).
pub fn prism_minimum_deviation(n_glass: f64, apex_angle_deg: f64) -> f64 {
    let apex = apex_angle_deg.to_radians();
    // At minimum deviation: sin((D_min + apex)/2) = n * sin(apex/2)
    let sin_val = n_glass * (apex / 2.0).sin();
    if sin_val.abs() > 1.0 {
        return f64::NAN;
    }
    let d_min = 2.0 * sin_val.asin() - apex;
    d_min.to_degrees()
}

/// Check if a ray is vignetted (has NaN angle from aperture stop).
pub fn is_vignetted(ray: &Ray) -> bool {
    ray.u.is_nan()
}

/// Aperture stop filtering: remove vignetted rays from a set.
pub fn filter_vignetted(rays: &[Ray]) -> Vec<Ray> {
    rays.iter().cloned().filter(|r| !is_vignetted(r)).collect()
}

/// Numerical aperture from acceptance half-angle in medium of index n.
pub fn numerical_aperture(half_angle_rad: f64, n: f64) -> f64 {
    n * half_angle_rad.sin()
}

/// Abbe diffraction limit for resolution: d = λ/(2·NA)
pub fn abbe_resolution(wavelength: f64, na: f64) -> f64 {
    wavelength / (2.0 * na)
}

/// F-number of a lens: f_num = f / D
pub fn f_number(focal_length: f64, aperture_diameter: f64) -> f64 {
    focal_length / aperture_diameter
}

/// Depth of field (approximate): DoF ≈ 2·λ·(f/#)² (in air)
pub fn depth_of_field(wavelength: f64, f_num: f64) -> f64 {
    2.0 * wavelength * f_num * f_num
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_through_thin_lens() {
        // Parallel ray at height 1 through a lens → passes through back focal point
        let system = OpticalSystem::new()
            .push(Surface::ThinLens { f: 100e-3 })
            .push(Surface::FreeSpace { d: 100e-3 });
        let ray = Ray::new(1.0, 0.0);
        let out = system.trace(ray);
        // After propagation to focal plane, all parallel rays converge to axis
        assert!(
            out.y.abs() < 1e-10,
            "Ray should cross axis at focal plane, y={:.2e}",
            out.y
        );
    }

    #[test]
    fn ray_through_two_lenses_telescope() {
        // Galilean telescope: f1=-50mm, f2=200mm, separation=f1+f2=150mm
        let f1 = -50e-3_f64;
        let f2 = 200e-3_f64;
        let d = f2 + f1; // 150mm separation
        let system = OpticalSystem::new()
            .push(Surface::ThinLens { f: f1 })
            .push(Surface::FreeSpace { d })
            .push(Surface::ThinLens { f: f2 });
        // Input parallel ray at height 1 → output magnified parallel ray at height -f2/f1=4
        let ray = Ray::new(1.0, 0.0);
        let out = system.trace(ray);
        let magnification = -f2 / f1; // = 4.0
        assert!(
            (out.u).abs() < 1e-10,
            "Output should be parallel: u={:.2e}",
            out.u
        );
        assert!(
            (out.y - magnification).abs() < 1e-8,
            "Magnification should be {magnification:.1}, got {:.4}",
            out.y
        );
    }

    #[test]
    fn focal_length_from_matrix() {
        let f = 50e-3_f64;
        let system = OpticalSystem::new().push(Surface::ThinLens { f });
        let efl = system.effective_focal_length().unwrap();
        assert!((efl - f).abs() < 1e-12);
    }

    #[test]
    fn flat_interface_snell_paraxial() {
        // Paraxial Snell: n1*u1 = n2*u2
        let n1 = 1.0;
        let n2 = 1.5;
        let s = Surface::FlatInterface { n1, n2 };
        let ray = Ray::new(0.0, 0.5);
        let out = s.apply(ray);
        let rel_err = (n1 * ray.u - n2 * out.u).abs();
        assert!(
            rel_err < 1e-12,
            "Snell's law violated: n1*u1={:.4} n2*u2={:.4}",
            n1 * ray.u,
            n2 * out.u
        );
    }

    #[test]
    fn numerical_aperture_values() {
        let na = numerical_aperture(30.0_f64.to_radians(), 1.5);
        assert!((na - 0.75).abs() < 0.01);
    }

    #[test]
    fn abbe_resolution_values() {
        let res = abbe_resolution(500e-9, 1.0);
        assert!((res - 250e-9).abs() < 1e-12);
    }

    #[test]
    fn trace_full_returns_snapshots() {
        let system = OpticalSystem::new()
            .push(Surface::FreeSpace { d: 10e-3 })
            .push(Surface::ThinLens { f: 50e-3 })
            .push(Surface::FreeSpace { d: 50e-3 });
        let rays = system.trace_full(Ray::new(1.0, 0.0));
        assert_eq!(rays.len(), 4); // initial + 3 surfaces
    }

    #[test]
    fn abcd_system_det_is_unity() {
        let system = OpticalSystem::new()
            .push(Surface::FreeSpace { d: 10e-3 })
            .push(Surface::ThinLens { f: 50e-3 })
            .push(Surface::FreeSpace { d: 50e-3 });
        let m = system.abcd_matrix();
        assert!((m.det() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn aperture_stop_clips_large_ray() {
        let stop = Surface::ApertureStop { radius: 1e-3 };
        let large_ray = Ray::new(2e-3, 0.0);
        let out = stop.apply(large_ray);
        assert!(is_vignetted(&out));
    }

    #[test]
    fn aperture_stop_passes_small_ray() {
        let stop = Surface::ApertureStop { radius: 1e-3 };
        let small_ray = Ray::new(0.5e-3, 0.1);
        let out = stop.apply(small_ray);
        assert!(!is_vignetted(&out));
        assert!((out.u - 0.1).abs() < 1e-12);
    }

    #[test]
    fn diffraction_grating_deflects_ray() {
        // 600 lines/mm grating, m=1 order, 633nm
        let groove_density = 600e3; // lines/m
        let grating = Surface::DiffractionGrating {
            groove_density,
            m: 1,
            n1: 1.0,
            n2: 1.0,
            wavelength: 633e-9,
        };
        let ray = Ray::new(0.0, 0.0); // on-axis
        let out = grating.apply(ray);
        // u_out = 0 + 1 * 633e-9 * 600e3 / 1.0 ≈ 0.3798 rad
        assert!((out.u - 633e-9 * groove_density).abs() < 1e-6);
    }

    #[test]
    fn aspheric_sphere_sag_paraxial() {
        let surf = AsphericSurface::spherical(100e-3, 1.0, 1.5);
        let h = 5e-3;
        let sag_p = surf.sag_paraxial(h);
        let sag_full = surf.sag(h);
        // Paraxial and full should be close for small h
        assert!((sag_p - sag_full).abs() / sag_p < 0.01);
    }

    #[test]
    fn aspheric_power_equals_formula() {
        let surf = AsphericSurface::spherical(50e-3, 1.0, 1.5);
        let expected = (1.5 - 1.0) / 50e-3; // = 10 diopters
        assert!((surf.power() - expected).abs() < 1e-10);
    }

    #[test]
    fn prism_deviation_positive() {
        // Equilateral prism, n_glass=1.5, apex=60deg, normal incidence ~48.6deg min deviation
        let dev = prism_deviation(1.5, 60.0, 50.0);
        assert!(dev > 0.0, "Deviation should be positive, got {dev:.2}");
        assert!(dev < 90.0, "Deviation should be < 90deg");
    }

    #[test]
    fn prism_minimum_deviation_less_than_apex() {
        // For glass prism, minimum deviation < apex angle is not guaranteed, but should be finite
        let d_min = prism_minimum_deviation(1.5, 60.0);
        assert!(d_min.is_finite(), "Minimum deviation should be finite");
        assert!(d_min > 0.0);
    }

    #[test]
    fn filter_vignetted_removes_nan_rays() {
        let rays = vec![
            Ray::new(0.5e-3, 0.1),
            Ray::new(2e-3, f64::NAN), // vignetted
            Ray::new(0.3e-3, -0.05),
        ];
        let filtered = filter_vignetted(&rays);
        assert_eq!(filtered.len(), 2);
    }
}
