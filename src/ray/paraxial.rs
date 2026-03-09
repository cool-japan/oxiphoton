//! Paraxial optics: cardinal points, thick lens analysis, and paraxial image formation.
//!
//! A thick optical system is described by its ABCD (ray transfer) matrix:
//!   \[y'\]   [A  B] \[y\]
//!   \[u'\] = [C  D] \[u\]
//!
//! Cardinal points:
//!   - Front focal point F:  image at infinity, object at f_F from first principal plane
//!   - Back focal point F':  object at infinity, image at f_B from second principal plane
//!   - Principal planes H, H': unit magnification conjugate planes
//!   - Nodal points N, N': unit angular magnification
//!
//! For a system in medium n₁ (input) and n₂ (output):
//!   f_B = -n₂/C,  f_F = n₁/C
//!   BFD = (D·n₂ - n₂·A + ... )/C  (back focal distance from last surface)

use crate::ray::tracer::{Ray, Surface};

/// Cardinal points of an optical system (all distances from the system reference).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CardinalPoints {
    /// Front focal length f_F (m), positive = converging
    pub front_focal_length: f64,
    /// Back focal length f_B (m)
    pub back_focal_length: f64,
    /// Front principal plane H position (m from first surface)
    pub front_principal_plane: f64,
    /// Back principal plane H' position (m from last surface)
    pub back_principal_plane: f64,
    /// Front focal distance (from first surface, m)
    pub front_focal_distance: f64,
    /// Back focal distance (from last surface, m)
    pub back_focal_distance: f64,
}

/// System ABCD matrix for a sequence of surfaces.
///
/// Built by multiplying surface matrices from right (input) to left (output).
#[derive(Debug, Clone, Copy)]
pub struct SystemMatrix {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

impl SystemMatrix {
    /// Identity matrix (free space of zero length).
    pub fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
        }
    }

    /// Free-space propagation matrix for distance d.
    pub fn propagation(d: f64) -> Self {
        Self {
            a: 1.0,
            b: d,
            c: 0.0,
            d: 1.0,
        }
    }

    /// Thin-lens refraction matrix for focal length f.
    pub fn thin_lens(f: f64) -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: -1.0 / f,
            d: 1.0,
        }
    }

    /// Curved interface matrix: radius R (positive = center to right), n1→n2.
    pub fn curved_interface(r: f64, n1: f64, n2: f64) -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: -(n2 - n1) / (r * n1),
            d: n1 / n2,
        }
    }

    /// Multiply two ABCD matrices: self = right (applied first) × left (applied second).
    /// In ray optics: M_total = M_last · ... · M_first
    pub fn then(self, next: SystemMatrix) -> SystemMatrix {
        SystemMatrix {
            a: next.a * self.a + next.b * self.c,
            b: next.a * self.b + next.b * self.d,
            c: next.c * self.a + next.d * self.c,
            d: next.c * self.b + next.d * self.d,
        }
    }

    /// Apply matrix to a ray.
    pub fn apply(&self, ray: Ray) -> Ray {
        Ray {
            y: self.a * ray.y + self.b * ray.u,
            u: self.c * ray.y + self.d * ray.u,
        }
    }

    /// Determinant (should be n1/n2 for a system between media n1 and n2).
    pub fn det(&self) -> f64 {
        self.a * self.d - self.b * self.c
    }

    /// Cardinal points assuming input medium n1=1.0 and output medium n2=1.0.
    pub fn cardinal_points(&self) -> CardinalPoints {
        self.cardinal_points_with_media(1.0, 1.0)
    }

    /// Cardinal points for system between media n1 (input) and n2 (output).
    pub fn cardinal_points_with_media(&self, n1: f64, n2: f64) -> CardinalPoints {
        let c = self.c;
        // EFL (effective focal length in image space)
        let f_b = if c.abs() < 1e-30 {
            f64::INFINITY
        } else {
            -n2 / c
        };
        let f_f = if c.abs() < 1e-30 {
            f64::INFINITY
        } else {
            n1 / c
        };
        // Back focal distance (BFD) from last surface: BFD = -D/C * n2
        let bfd = if c.abs() < 1e-30 {
            f64::INFINITY
        } else {
            -self.d * n2 / c
        };
        // Front focal distance (FFD) from first surface: FFD = A/C * n1
        let ffd = if c.abs() < 1e-30 {
            f64::INFINITY
        } else {
            self.a * n1 / c
        };
        // Principal plane positions
        let h_back = bfd - f_b; // H' from last surface
        let h_front = ffd - f_f; // H from first surface
        CardinalPoints {
            front_focal_length: f_f,
            back_focal_length: f_b,
            front_principal_plane: h_front,
            back_principal_plane: h_back,
            front_focal_distance: ffd,
            back_focal_distance: bfd,
        }
    }

    /// Build system matrix from a slice of surfaces.
    pub fn from_surfaces(surfaces: &[Surface]) -> Self {
        surfaces.iter().fold(Self::identity(), |m, s| {
            let sm = Self::from_surface(s);
            m.then(sm)
        })
    }

    fn from_surface(surface: &Surface) -> Self {
        match *surface {
            Surface::FreeSpace { d } => Self::propagation(d),
            Surface::ThinLens { f } => Self::thin_lens(f),
            Surface::CurvedInterface { r, n1, n2 } => Self::curved_interface(r, n1, n2),
            Surface::FlatInterface { n1, n2 } => Self {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: n1 / n2,
            },
            Surface::Mirror { r } => {
                // Mirror as thin lens equivalent: f = R/2
                if r.is_infinite() {
                    Self::identity()
                } else {
                    Self::thin_lens(r / 2.0)
                }
            }
            Surface::ApertureStop { .. } => Self::identity(),
            Surface::DiffractionGrating { n1, n2, .. } => Self {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: n1 / n2,
            },
        }
    }
}

/// Paraxial image formation using the Gaussian lens formula.
///
/// Thin lens: 1/v - 1/u = 1/f  (with sign convention: u < 0 for real object)
#[derive(Debug, Clone, Copy)]
pub struct ParaxialImager {
    /// Effective focal length f (m)
    pub focal_length: f64,
}

impl ParaxialImager {
    pub fn new(focal_length: f64) -> Self {
        Self { focal_length }
    }

    /// Image distance v for object at distance u (m, negative = real object).
    ///   1/v = 1/f + 1/u
    pub fn image_distance(&self, object_distance: f64) -> f64 {
        let inv_v = 1.0 / self.focal_length + 1.0 / object_distance;
        if inv_v.abs() < 1e-30 {
            f64::INFINITY
        } else {
            1.0 / inv_v
        }
    }

    /// Lateral magnification m = v / u (negative = inverted real image).
    pub fn magnification(&self, object_distance: f64) -> f64 {
        self.image_distance(object_distance) / object_distance
    }

    /// Newton's lens equation: x·x' = f² where x, x' are distances from focal planes.
    /// Returns x' (image position from back focal plane) given x (object from front focal).
    pub fn newtons_equation(&self, x_object: f64) -> f64 {
        self.focal_length * self.focal_length / x_object
    }

    /// Depth of focus Δv for a given depth of field Δu and magnification m.
    ///   Δv = m²·Δu
    pub fn depth_of_focus(&self, depth_of_field: f64, object_distance: f64) -> f64 {
        let m = self.magnification(object_distance);
        m * m * depth_of_field
    }

    /// Angular field of view (half-angle, rad) for a sensor of half-size h at image distance v.
    pub fn field_of_view_half(&self, sensor_half_size: f64, object_distance: f64) -> f64 {
        let v = self.image_distance(object_distance);
        (sensor_half_size / v).atan()
    }
}

/// Chromatic aberration analysis for a single lens.
///
/// Longitudinal chromatic aberration (LCA):
///   LCA = f / V  where V = (n_d - 1)/(n_F - n_C) is the Abbe number.
///
/// An achromatic doublet eliminates LCA by combining crown (high V) and flint (low V):
///   f₁ = f·V₁/(V₁-V₂),  f₂ = -f·V₂/(V₁-V₂)
#[derive(Debug, Clone, Copy)]
pub struct ChromaticAnalysis {
    /// Abbe number (V-number) of the glass
    pub abbe_number: f64,
    /// Focal length at d-line (587.6 nm), m
    pub focal_length: f64,
}

impl ChromaticAnalysis {
    /// Create from glass Abbe number and focal length.
    pub fn new(focal_length: f64, abbe_number: f64) -> Self {
        Self {
            abbe_number,
            focal_length,
        }
    }

    /// Crown glass (BK7): n_d=1.5168, V=64.2.
    pub fn bk7(focal_length: f64) -> Self {
        Self::new(focal_length, 64.2)
    }

    /// Flint glass (F2): n_d=1.6200, V=36.4.
    pub fn f2(focal_length: f64) -> Self {
        Self::new(focal_length, 36.4)
    }

    /// Longitudinal chromatic aberration LCA = f / V (m).
    pub fn lca(&self) -> f64 {
        self.focal_length / self.abbe_number
    }

    /// Required focal length of crown lens in an achromatic doublet with flint abbe number V2.
    ///
    /// Achromatic condition: φ₁/V₁ + φ₂/V₂ = 0 and φ₁ + φ₂ = φ.
    /// Solving: φ₁ = φ·V₁/(V₁-V₂)  →  f₁ = f·(V₁-V₂)/V₁
    pub fn achromat_crown_focal(&self, v2: f64) -> f64 {
        self.focal_length * (self.abbe_number - v2) / self.abbe_number
    }

    /// Required focal length of flint lens in an achromatic doublet.
    ///
    /// φ₂ = -φ·V₂/(V₁-V₂)  →  f₂ = -f·(V₁-V₂)/V₂
    pub fn achromat_flint_focal(&self, v2: f64) -> f64 {
        -self.focal_length * (self.abbe_number - v2) / v2
    }
}

/// Entrance and exit pupil analysis.
#[derive(Debug, Clone, Copy)]
pub struct PupilAnalysis {
    /// F-number (f/D) of the system
    pub f_number: f64,
    /// Entrance pupil diameter (m)
    pub entrance_pupil_diameter: f64,
}

impl PupilAnalysis {
    pub fn new(focal_length: f64, aperture_diameter: f64) -> Self {
        Self {
            f_number: focal_length / aperture_diameter,
            entrance_pupil_diameter: aperture_diameter,
        }
    }

    /// Numerical aperture NA = n·sin(θ) ≈ n·D/(2f) for small angles.
    pub fn numerical_aperture(&self, n: f64) -> f64 {
        n * self.entrance_pupil_diameter / (2.0 * self.f_number * self.entrance_pupil_diameter)
    }

    /// Airy disk radius (m) at wavelength λ: r = 1.22·λ·f_number.
    pub fn airy_radius(&self, wavelength: f64) -> f64 {
        1.22 * wavelength * self.f_number
    }

    /// Rayleigh resolution criterion (m).
    pub fn rayleigh_resolution(&self, wavelength: f64) -> f64 {
        self.airy_radius(wavelength)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thin_lens_cardinal_points() {
        // Single thin lens f=100mm
        let f = 0.1;
        let m = SystemMatrix::thin_lens(f);
        let cp = m.cardinal_points();
        assert!(
            (cp.back_focal_length - f).abs() < 1e-12,
            "BFL={}",
            cp.back_focal_length
        );
        assert!(
            (cp.front_focal_length + f).abs() < 1e-12,
            "FFL={}",
            cp.front_focal_length
        );
    }

    #[test]
    fn system_matrix_from_surfaces() {
        let surfaces = vec![Surface::ThinLens { f: 0.1 }];
        let m = SystemMatrix::from_surfaces(&surfaces);
        assert!((m.c + 10.0).abs() < 1e-10, "C={}", m.c);
    }

    #[test]
    fn paraxial_image_distance_thin_lens() {
        let imager = ParaxialImager::new(0.1); // f=100mm
        let v = imager.image_distance(-0.2); // object 200mm in front
                                             // 1/v = 1/0.1 + 1/(-0.2) = 10 - 5 = 5 → v = 0.2
        assert!((v - 0.2).abs() < 1e-12, "image distance={v}");
    }

    #[test]
    fn magnification_sign() {
        let imager = ParaxialImager::new(0.1);
        let m = imager.magnification(-0.2);
        // v=0.2, u=-0.2, m = 0.2/(-0.2) = -1 (inverted)
        assert!((m + 1.0).abs() < 1e-12, "magnification={m}");
    }

    #[test]
    fn newtons_equation() {
        let imager = ParaxialImager::new(0.1);
        // x=0.1 → x'=f²/x=0.01/0.1=0.1
        let xp = imager.newtons_equation(0.1);
        assert!((xp - 0.1).abs() < 1e-12, "x'={xp}");
    }

    #[test]
    fn chromatic_aberration_achromat() {
        let crown = ChromaticAnalysis::bk7(0.1); // f=100mm BK7
        let v2 = 36.4; // F2 flint
        let f1 = crown.achromat_crown_focal(v2);
        let f2 = crown.achromat_flint_focal(v2);
        // Combined power: 1/f1 + 1/f2 = 1/f
        let combined = 1.0 / f1 + 1.0 / f2;
        assert!((combined - 10.0).abs() < 1e-6, "combined power={combined}");
    }

    #[test]
    fn matrix_multiply_identity() {
        let m = SystemMatrix::identity();
        let m2 = m.then(SystemMatrix::identity());
        assert!((m2.a - 1.0).abs() < 1e-12);
        assert!((m2.d - 1.0).abs() < 1e-12);
        assert!(m2.b.abs() < 1e-12);
        assert!(m2.c.abs() < 1e-12);
    }

    #[test]
    fn propagation_then_lens() {
        // propagation(f).then(thin_lens(f)) computes M = M_lens · M_propagation
        // = [1 0; -1/f 1] · [1 f; 0 1] = [1, f; -1/f, 0]
        let f = 0.1;
        let m = SystemMatrix::propagation(f).then(SystemMatrix::thin_lens(f));
        assert!((m.a - 1.0).abs() < 1e-12, "A={}", m.a);
        assert!((m.b - f).abs() < 1e-12, "B={}", m.b);
        assert!((m.c + 1.0 / f).abs() < 1e-12, "C={}", m.c);
        assert!(m.d.abs() < 1e-12, "D={}", m.d);
    }

    #[test]
    fn airy_radius_f2() {
        let p = PupilAnalysis::new(0.1, 0.05); // f=100mm, D=50mm → f/2
        let r = p.airy_radius(550e-9);
        assert!((r - 1.22 * 550e-9 * 2.0).abs() < 1e-15, "airy={r}");
    }
}
