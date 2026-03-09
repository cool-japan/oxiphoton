/// Transformation Optics coordinate transformations.
///
/// Provides electromagnetic cloaks, concentrators, anti-reflection coatings
/// and generic coordinate mappings derived from Jacobian-based material tensors.
use crate::error::OxiPhotonError;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: 3×3 matrix operations
// ─────────────────────────────────────────────────────────────────────────────

type Mat3 = [[f64; 3]; 3];

/// Matrix multiplication C = A · B for 3×3 real matrices.
fn mat3_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut c = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Transpose of a 3×3 matrix.
fn mat3_transpose(a: &Mat3) -> Mat3 {
    let mut t = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = a[j][i];
        }
    }
    t
}

/// Determinant of a 3×3 matrix via cofactor expansion.
fn mat3_det(a: &Mat3) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

/// Scale all elements of a 3×3 matrix by scalar `s`.
fn mat3_scale(a: &Mat3, s: f64) -> Mat3 {
    let mut out = *a;
    for row in &mut out {
        for v in row.iter_mut() {
            *v *= s;
        }
    }
    out
}

/// Identity 3×3 matrix.
const MAT3_IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

// ─────────────────────────────────────────────────────────────────────────────
// TransformType & CoordTransformation
// ─────────────────────────────────────────────────────────────────────────────

/// Supported coordinate transformation types.
#[derive(Debug, Clone)]
pub enum TransformType {
    /// Cylindrical cloak: maps annulus r ∈ [a, b] → r' ∈ [a, b] hiding r = 0.
    CylindricalCloak {
        /// Inner (hidden) radius in metres.
        inner_radius: f64,
        /// Outer (cloak boundary) radius in metres.
        outer_radius: f64,
    },
    /// Spherical cloak (Pendry 2006): 3D analogue of the cylindrical cloak.
    SphericalCloak {
        inner_radius: f64,
        outer_radius: f64,
    },
    /// Uniform anisotropic stretching along each axis.
    LinearScale {
        scale_x: f64,
        scale_y: f64,
        scale_z: f64,
    },
    /// Beam-bending: maps a straight waveguide to a circular arc.
    BeamBend {
        /// Bending radius in metres.
        bend_radius: f64,
        /// Total bending angle in radians.
        angle_rad: f64,
    },
    /// Flattening transform: maps a curved surface to a flat one.
    Flattener {
        /// Surface curvature 1/R in m⁻¹.
        curvature: f64,
    },
}

/// Generic coordinate transformation for transformation optics.
///
/// Given a mapping from physical space (x, y, z) to virtual (electromagnetic)
/// space (x', y', z'), the transformation media parameters are:
///
///   ε' = J · ε · Jᵀ / det(J)
///   μ' = J · μ · Jᵀ / det(J)
///
/// where J = ∂(x', y', z') / ∂(x, y, z) is the Jacobian.
#[derive(Debug, Clone)]
pub struct CoordTransformation {
    /// Which transformation to apply.
    pub transform_type: TransformType,
}

impl CoordTransformation {
    /// Create a new coordinate transformation.
    pub fn new(transform_type: TransformType) -> Self {
        Self { transform_type }
    }

    /// Compute the Jacobian matrix J(x, y, z) numerically using central differences.
    ///
    /// For the cylindrical and spherical cloaks, the Jacobian is computed
    /// analytically to avoid singularities.
    pub fn jacobian(&self, x: f64, y: f64, z: f64) -> Mat3 {
        match &self.transform_type {
            TransformType::LinearScale {
                scale_x,
                scale_y,
                scale_z,
            } => {
                // x' = sx·x, y' = sy·y, z' = sz·z  →  J = diag(sx, sy, sz)
                [
                    [*scale_x, 0.0, 0.0],
                    [0.0, *scale_y, 0.0],
                    [0.0, 0.0, *scale_z],
                ]
            }
            TransformType::CylindricalCloak {
                inner_radius: a,
                outer_radius: b,
            } => {
                // Pendry cylindrical cloak mapping:
                //   r' = a + r·(b-a)/b
                //   φ' = φ
                //   z' = z
                // In Cartesian: x = r cosφ, y = r sinφ
                let r = (x * x + y * y).sqrt();
                if r < 1e-15 {
                    // At origin the cloak is singular — return identity as a safe fallback
                    return MAT3_IDENTITY;
                }
                let a = *a;
                let b = *b;
                // dr'/dr = (b-a)/b
                let dr_prime_dr = (b - a) / b;
                // r' = a + r*(b-a)/b
                let r_prime = a + r * (b - a) / b;
                let cos_phi = x / r;
                let sin_phi = y / r;

                // Jacobian in polar:
                // ∂x'/∂x = (dr'/dr)cos²φ + (r'/r)sin²φ
                // ∂x'/∂y = (dr'/dr - r'/r)sinφ cosφ
                // ∂y'/∂x = (dr'/dr - r'/r)sinφ cosφ
                // ∂y'/∂y = (dr'/dr)sin²φ + (r'/r)cos²φ
                let jrr = dr_prime_dr;
                let jss = r_prime / r; // tangential stretch
                let jxx = jrr * cos_phi * cos_phi + jss * sin_phi * sin_phi;
                let jxy = (jrr - jss) * sin_phi * cos_phi;
                let jyx = jxy;
                let jyy = jrr * sin_phi * sin_phi + jss * cos_phi * cos_phi;
                [[jxx, jxy, 0.0], [jyx, jyy, 0.0], [0.0, 0.0, 1.0]]
            }
            TransformType::SphericalCloak {
                inner_radius: a,
                outer_radius: b,
            } => {
                // Pendry spherical cloak:
                //   r' = a + r·(b-a)/b
                // Jacobian is the same form as cylindrical but in 3D spherical
                let r = (x * x + y * y + z * z).sqrt();
                if r < 1e-15 {
                    return MAT3_IDENTITY;
                }
                let a = *a;
                let b = *b;
                let dr_prime_dr = (b - a) / b;
                let r_prime = a + r * (b - a) / b;
                let ratio = r_prime / r;

                // Unit vector components
                let rx = x / r;
                let ry = y / r;
                let rz = z / r;

                // J = (dr'/dr - r'/r) r̂r̂ᵀ + (r'/r) I
                // = ratio·I + (dr_prime_dr - ratio) r̂r̂ᵀ
                let diff = dr_prime_dr - ratio;
                [
                    [ratio + diff * rx * rx, diff * rx * ry, diff * rx * rz],
                    [diff * ry * rx, ratio + diff * ry * ry, diff * ry * rz],
                    [diff * rz * rx, diff * rz * ry, ratio + diff * rz * rz],
                ]
            }
            TransformType::BeamBend {
                bend_radius: r0,
                angle_rad,
            } => {
                // Map straight waveguide [0,L]×[-W/2,W/2] to circular arc.
                // Use local approximation: s = x (along waveguide), u = y (transverse).
                // Orientation-preserving convention (clockwise bend):
                //   x' = (r0+u)·cos(s/r0)
                //   y' = −(r0+u)·sin(s/r0)
                //   z' = z
                // This yields det(J) = rho/r0 > 0, preserving coordinate orientation.
                // Jacobian at (x,y,z) where s↔x, u↔y:
                let s = x;
                let u = y;
                let r0 = *r0;
                let raw_theta = s / r0;
                let theta = raw_theta.clamp(-*angle_rad, *angle_rad);
                let cos_t = theta.cos();
                let sin_t = theta.sin();
                let rho = r0 + u;
                // ∂x'/∂s = −rho/r0·sin(θ),  ∂x'/∂u =  cos(θ)
                // ∂y'/∂s = −rho/r0·cos(θ),  ∂y'/∂u = −sin(θ)
                // det(J) = (−rho/r0·sin θ)(−sin θ) − (cos θ)(−rho/r0·cos θ)
                //        = rho/r0·(sin²θ + cos²θ) = rho/r0 > 0
                [
                    [-rho / r0 * sin_t, cos_t, 0.0],
                    [-rho / r0 * cos_t, -sin_t, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            }
            TransformType::Flattener { curvature: kappa } => {
                // Flattening: maps curved surface z = κ(x²+y²)/2 → flat z=0.
                // Transform: x'=x, y'=y, z'=z − κ(x²+y²)/2
                // J = [[1,0,0],[0,1,0],[-κx,-κy,1]]
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [-kappa * x, -kappa * y, 1.0],
                ]
            }
        }
    }

    /// Determinant of the Jacobian at (x, y, z).
    pub fn jacobian_det(&self, x: f64, y: f64, z: f64) -> f64 {
        let j = self.jacobian(x, y, z);
        mat3_det(&j)
    }

    /// Transformed permittivity tensor ε' = J · J^T / det(J) (vacuum background).
    pub fn transformed_eps(&self, x: f64, y: f64, z: f64) -> Mat3 {
        let j = self.jacobian(x, y, z);
        let det = mat3_det(&j);
        if det.abs() < 1e-30 {
            return MAT3_IDENTITY;
        }
        let jjt = mat3_mul(&j, &mat3_transpose(&j));
        mat3_scale(&jjt, 1.0 / det)
    }

    /// Transformed permeability tensor μ' = J · J^T / det(J) (same as ε' for TO).
    pub fn transformed_mu(&self, x: f64, y: f64, z: f64) -> Mat3 {
        self.transformed_eps(x, y, z)
    }

    /// Map a point from virtual space to physical space.
    pub fn virtual_to_physical(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        match &self.transform_type {
            TransformType::LinearScale {
                scale_x,
                scale_y,
                scale_z,
            } => (x * scale_x, y * scale_y, z * scale_z),
            TransformType::CylindricalCloak {
                inner_radius: a,
                outer_radius: b,
            } => {
                // r' = a + r*(b-a)/b  →  r = (r'-a)*b/(b-a)
                let rp = (x * x + y * y).sqrt();
                if rp < *a {
                    return (x, y, z); // inside cloaked region — pass through
                }
                let a = *a;
                let b = *b;
                let r = (rp - a) * b / (b - a);
                let phi = y.atan2(x);
                (r * phi.cos(), r * phi.sin(), z)
            }
            TransformType::SphericalCloak {
                inner_radius: a,
                outer_radius: b,
            } => {
                let rp = (x * x + y * y + z * z).sqrt();
                if rp < *a {
                    return (x, y, z);
                }
                let a = *a;
                let b = *b;
                let r = (rp - a) * b / (b - a);
                let scale = if rp > 1e-15 { r / rp } else { 1.0 };
                (x * scale, y * scale, z * scale)
            }
            TransformType::BeamBend {
                bend_radius: r0,
                angle_rad,
            } => {
                // Orientation-preserving convention: x' = rho·cos(θ), y' = −rho·sin(θ)
                // Inverse: given (s, u) in waveguide coords:
                //   rho = r0 + u, θ = s/r0
                //   x' = rho·cos(θ), y' = −rho·sin(θ)
                let r0 = *r0;
                let s = x;
                let u = y;
                let theta = (s / r0).clamp(-*angle_rad, *angle_rad);
                let rho = r0 + u;
                (rho * theta.cos(), -rho * theta.sin(), z)
            }
            TransformType::Flattener { curvature: kappa } => {
                (x, y, z + kappa * (x * x + y * y) / 2.0)
            }
        }
    }

    /// Map a point from physical space to virtual space.
    pub fn physical_to_virtual(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        match &self.transform_type {
            TransformType::LinearScale {
                scale_x,
                scale_y,
                scale_z,
            } => (x / scale_x, y / scale_y, z / scale_z),
            TransformType::CylindricalCloak {
                inner_radius: a,
                outer_radius: b,
            } => {
                let r = (x * x + y * y).sqrt();
                let a = *a;
                let b = *b;
                let rp = a + r * (b - a) / b;
                let phi = y.atan2(x);
                (rp * phi.cos(), rp * phi.sin(), z)
            }
            TransformType::SphericalCloak {
                inner_radius: a,
                outer_radius: b,
            } => {
                let r = (x * x + y * y + z * z).sqrt();
                let a = *a;
                let b = *b;
                let rp = a + r * (b - a) / b;
                let scale = if r > 1e-15 { rp / r } else { 1.0 };
                (x * scale, y * scale, z * scale)
            }
            TransformType::BeamBend {
                bend_radius: r0,
                angle_rad,
            } => {
                // Orientation-preserving convention: x' = rho·cos(θ), y' = −rho·sin(θ)
                // Inverse: given virtual Cartesian (x', y'):
                //   rho = sqrt(x'² + y'²), θ = atan2(−y', x')
                //   s = r0·θ, u = rho − r0
                let r0 = *r0;
                let rho = (x * x + y * y).sqrt();
                let theta = (-y).atan2(x).clamp(-*angle_rad, *angle_rad);
                let s = r0 * theta;
                let u = rho - r0;
                (s, u, z)
            }
            TransformType::Flattener { curvature: kappa } => {
                (x, y, z - kappa * (x * x + y * y) / 2.0)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cylindrical Cloak
// ─────────────────────────────────────────────────────────────────────────────

/// Pendry cylindrical electromagnetic cloak.
///
/// The cloak occupies the annular region a ≤ r ≤ b.
/// Material parameters derived from the coordinate transformation
/// r' = (b/(b−a))·(r − a):
///
///   ε_r(r)   = μ_r(r)   = (r − a)/r
///   ε_φ(r)   = μ_φ(r)   = r/(r − a)
///   ε_z(r)   = μ_z(r)   = (b/(b−a))² · (r−a)/r
#[derive(Debug, Clone)]
pub struct CylindricalCloak {
    /// Inner (hidden) radius in metres.
    pub inner_radius: f64,
    /// Outer (cloak boundary) radius in metres.
    pub outer_radius: f64,
}

impl CylindricalCloak {
    /// Create a new cylindrical cloak.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] if `inner_radius >= outer_radius`
    /// or if either radius is non-positive.
    pub fn new(inner_radius: f64, outer_radius: f64) -> Result<Self, OxiPhotonError> {
        if inner_radius <= 0.0 || outer_radius <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "Cloak radii must be positive".to_string(),
            ));
        }
        if inner_radius >= outer_radius {
            return Err(OxiPhotonError::NumericalError(format!(
                "inner_radius ({inner_radius}) must be < outer_radius ({outer_radius})"
            )));
        }
        Ok(Self {
            inner_radius,
            outer_radius,
        })
    }

    /// Radial (r) component of permittivity/permeability at radius r.
    ///
    /// ε_r(r) = (r − a)/r
    pub fn eps_radial(&self, r: f64) -> f64 {
        let a = self.inner_radius;
        (r - a) / r
    }

    /// Azimuthal (φ) component of permittivity/permeability at radius r.
    ///
    /// ε_φ(r) = r/(r − a)
    pub fn eps_azimuthal(&self, r: f64) -> f64 {
        let a = self.inner_radius;
        if (r - a).abs() < 1e-30 {
            return f64::INFINITY;
        }
        r / (r - a)
    }

    /// Axial (z) component of permittivity/permeability at radius r.
    ///
    /// ε_z(r) = (b/(b−a))² · (r−a)/r
    pub fn eps_z(&self, r: f64) -> f64 {
        let a = self.inner_radius;
        let b = self.outer_radius;
        let scale = b / (b - a);
        scale * scale * (r - a) / r
    }

    /// Returns `true` if the point (x, y) lies inside the cloaked (hidden) region r < a.
    pub fn is_cloaked(&self, x: f64, y: f64) -> bool {
        (x * x + y * y).sqrt() < self.inner_radius
    }

    /// Fill a 2D FDTD grid with the cylindrical cloak permittivity profiles.
    ///
    /// The grid cell at (ix, iy) has its centre at
    ///   x = (ix as f64 + 0.5)*dx − x_center
    ///   y = (iy as f64 + 0.5)*dy − y_center
    ///
    /// Grid is stored in row-major order: index = iy * nx + ix.
    /// Points inside the cloaked region (r < a) are left unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn fill_fdtd_grid(
        &self,
        eps_r: &mut [f64],
        eps_phi: &mut [f64],
        eps_z: &mut [f64],
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
        x_center: f64,
        y_center: f64,
    ) {
        let a = self.inner_radius;
        let b = self.outer_radius;
        for iy in 0..ny {
            for ix in 0..nx {
                let cx = (ix as f64 + 0.5) * dx - x_center;
                let cy = (iy as f64 + 0.5) * dy - y_center;
                let r = (cx * cx + cy * cy).sqrt();
                let idx = iy * nx + ix;
                if r >= a && r <= b {
                    eps_r[idx] = self.eps_radial(r);
                    eps_phi[idx] = self.eps_azimuthal(r);
                    eps_z[idx] = self.eps_z(r);
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Electromagnetic Concentrator
// ─────────────────────────────────────────────────────────────────────────────

/// Electromagnetic concentrator — compresses the field into an inner region.
///
/// For a cylindrical concentrator with compression factor η = inner/outer:
///   ε_r(r) = r / (η·(r − a) + a)
///   ε_φ(r) = (η·(r − a) + a) / r
///   Enhancement factor ≈ (outer / inner)²
#[derive(Debug, Clone)]
pub struct EmConcentrator {
    /// Inner radius of the concentrating shell.
    pub inner_radius: f64,
    /// Outer radius of the concentrating shell.
    pub outer_radius: f64,
    /// Linear field concentration factor η > 1.
    pub concentration_factor: f64,
}

impl EmConcentrator {
    /// Create a new electromagnetic concentrator.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] if any parameter is invalid.
    pub fn new(inner: f64, outer: f64, factor: f64) -> Result<Self, OxiPhotonError> {
        if inner <= 0.0 || outer <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "Concentrator radii must be positive".to_string(),
            ));
        }
        if inner >= outer {
            return Err(OxiPhotonError::NumericalError(format!(
                "inner ({inner}) must be < outer ({outer})"
            )));
        }
        if factor <= 1.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "concentration_factor must be > 1, got {factor}"
            )));
        }
        Ok(Self {
            inner_radius: inner,
            outer_radius: outer,
            concentration_factor: factor,
        })
    }

    /// Radial permittivity component.
    pub fn eps_radial(&self, r: f64) -> f64 {
        let a = self.inner_radius;
        let eta = self.concentration_factor;
        // Mapping: r' = eta*(r - a) + a  for r ∈ [a, b]
        let r_prime = eta * (r - a) + a;
        if r_prime.abs() < 1e-30 {
            return 0.0;
        }
        r / r_prime
    }

    /// Azimuthal permittivity component.
    pub fn eps_azimuthal(&self, r: f64) -> f64 {
        let a = self.inner_radius;
        let eta = self.concentration_factor;
        let r_prime = eta * (r - a) + a;
        if r.abs() < 1e-30 {
            return 0.0;
        }
        r_prime / r
    }

    /// Peak field enhancement factor (power ratio) at the inner surface.
    ///
    /// For an ideal concentrator, the intensity scales as (outer/inner)².
    pub fn field_enhancement(&self) -> f64 {
        let ratio = self.outer_radius / self.inner_radius;
        ratio * ratio
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Graded-Index Anti-Reflection Coating via EMT
// ─────────────────────────────────────────────────────────────────────────────

/// Graded-index anti-reflection coating designed via Maxwell-Garnett EMT.
///
/// The refractive index is varied continuously from n_ambient to n_substrate
/// using a prescribed profile, realised as a stack of thin nanostructured layers
/// whose fill fractions are determined by the effective medium model.
#[derive(Debug, Clone)]
pub struct GradedIndexARC {
    /// Substrate refractive index (real).
    pub n_substrate: f64,
    /// Ambient (air) refractive index (real).
    pub n_ambient: f64,
    /// Number of discrete graded layers.
    pub n_layers: usize,
    /// Total ARC stack thickness in nm.
    pub total_thickness_nm: f64,
    /// Design wavelength in nm.
    pub wavelength_nm: f64,
}

impl GradedIndexARC {
    /// Create a new graded-index ARC.
    pub fn new(
        n_substrate: f64,
        n_ambient: f64,
        n_layers: usize,
        thickness_nm: f64,
        lambda_nm: f64,
    ) -> Self {
        Self {
            n_substrate,
            n_ambient,
            n_layers: n_layers.max(2),
            total_thickness_nm: thickness_nm,
            wavelength_nm: lambda_nm,
        }
    }

    /// Exponential (linear-in-log) index profile.
    ///
    /// n(z) = n_amb · (n_sub / n_amb)^(z/d),  z ∈ [0, d]
    /// Sampled at the centre of each layer.
    pub fn index_profile(&self) -> Vec<f64> {
        let nl = self.n_layers;
        let ratio = self.n_substrate / self.n_ambient;
        (0..nl)
            .map(|i| {
                // Sample at layer centre: z = (i + 0.5) / nl
                let t = (i as f64 + 0.5) / nl as f64;
                self.n_ambient * ratio.powf(t)
            })
            .collect()
    }

    /// Quintic polynomial index profile for ultra-low reflection.
    ///
    /// Uses the Klopfenstein-motivated quintic p(t) = 10t³ − 15t⁴ + 6t⁵
    /// ensuring smooth first and second derivatives at both interfaces.
    pub fn quintic_index_profile(&self) -> Vec<f64> {
        let nl = self.n_layers;
        let n0 = self.n_ambient;
        let ns = self.n_substrate;
        // Use logarithmic interpolation with quintic blending function
        (0..nl)
            .map(|i| {
                let t = i as f64 / (nl.saturating_sub(1).max(1)) as f64;
                let p = 10.0 * t.powi(3) - 15.0 * t.powi(4) + 6.0 * t.powi(5);
                // n = n_amb * (n_sub/n_amb)^p  (log-linear quintic)
                n0 * (ns / n0).powf(p)
            })
            .collect()
    }

    /// Layer fill fractions required to achieve the exponential index profile
    /// using Maxwell-Garnett mixing of a high-index inclusion (n_inclusion)
    /// in an air host.
    ///
    /// Solves ε_eff = n(z)² for f using the MG formula analytically.
    pub fn fill_fractions(&self, n_inclusion: f64) -> Vec<f64> {
        let eps_host = num_complex::Complex64::new(self.n_ambient * self.n_ambient, 0.0);
        let eps_incl = num_complex::Complex64::new(n_inclusion * n_inclusion, 0.0);
        self.index_profile()
            .iter()
            .map(|&n_target| {
                let eps_target = n_target * n_target;
                // MG: eps_eff = eps_h * [1 + 3f·Δ/(Δ_0 - f·Δ)]
                // where Δ = ε_i - ε_h, Δ_0 = ε_i + 2ε_h
                // Solve for f:
                //   (eps_eff/eps_h - 1) * (eps_i + 2*eps_h) = 3f*(eps_i - eps_h) + f*(eps_eff/eps_h - 1)*(eps_i - eps_h)
                // → f = (eps_eff - eps_h)*(eps_i + 2*eps_h) / [(eps_eff - eps_h)*(eps_i - eps_h) + 3*(eps_i - eps_h)*eps_h]
                // For real values:
                let eh = eps_host.re;
                let ei = eps_incl.re;
                let et = eps_target;
                let delta = ei - eh;
                if delta.abs() < 1e-12 {
                    return 0.0; // degenerate case
                }
                let num = (et - eh) * (ei + 2.0 * eh);
                let _den = (et - eh) * delta + 3.0 * delta * eh;
                // Simplify: den = delta * [(et - eh) + 3*eh] = delta * (et + 2*eh)
                let den = delta * (et + 2.0 * eh);
                if den.abs() < 1e-12 {
                    return 0.0;
                }
                (num / den).clamp(0.0, 1.0)
            })
            .collect()
    }

    /// Layer thicknesses (uniform: d_layer = total / n_layers).
    pub fn layer_thicknesses_nm(&self) -> Vec<f64> {
        let d = self.total_thickness_nm / self.n_layers as f64;
        vec![d; self.n_layers]
    }

    /// Estimate normal-incidence reflectance using the transfer-matrix method.
    ///
    /// Uses the exponential index profile and the thin-film Fresnel equations
    /// for each layer.  The result is an amplitude reflectance squared.
    pub fn reflectance(&self) -> f64 {
        let profile = self.index_profile();
        let thicknesses = self.layer_thicknesses_nm();
        let lambda = self.wavelength_nm;

        // Build a chain of Fresnel interfaces + phase shifts
        // Transfer matrix for each layer: [[cos δ, -i sin δ/n], [-i n sin δ, cos δ]]
        // Use 2×2 complex transfer matrix
        use num_complex::Complex64;
        let i_unit = Complex64::new(0.0, 1.0);

        let mut m = [[Complex64::new(1.0, 0.0); 2]; 2];

        let all_n: Vec<f64> = std::iter::once(self.n_ambient)
            .chain(profile.iter().copied())
            .chain(std::iter::once(self.n_substrate))
            .collect();

        for (k, (&n_layer, &d)) in profile.iter().zip(thicknesses.iter()).enumerate() {
            let delta = 2.0 * std::f64::consts::PI * n_layer * d / lambda;
            let cos_d = Complex64::new(delta.cos(), 0.0);
            let sin_d = Complex64::new(delta.sin(), 0.0);
            let n_c = Complex64::new(n_layer, 0.0);
            // Layer matrix
            let ml = [
                [cos_d, -i_unit * sin_d / n_c],
                [-i_unit * n_c * sin_d, cos_d],
            ];
            // M_new = ml * M
            let m00 = ml[0][0] * m[0][0] + ml[0][1] * m[1][0];
            let m01 = ml[0][0] * m[0][1] + ml[0][1] * m[1][1];
            let m10 = ml[1][0] * m[0][0] + ml[1][1] * m[1][0];
            let m11 = ml[1][0] * m[0][1] + ml[1][1] * m[1][1];
            m = [[m00, m01], [m10, m11]];
            let _ = (k, all_n[k + 1]); // suppress unused warning
        }

        let n0 = Complex64::new(self.n_ambient, 0.0);
        let ns = Complex64::new(self.n_substrate, 0.0);
        // r = (m00*n0 + m01*n0*ns - m10 - m11*ns) / (m00*n0 + m01*n0*ns + m10 + m11*ns)
        let num = m[0][0] * n0 + m[0][1] * n0 * ns - m[1][0] - m[1][1] * ns;
        let den = m[0][0] * n0 + m[0][1] * n0 * ns + m[1][0] + m[1][1] * ns;
        if den.norm() < 1e-30 {
            return 0.0;
        }
        (num / den).norm_sqr()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── Cylindrical cloak ────────────────────────────────────────────────────

    #[test]
    fn test_cloak_inner_eps_r() {
        // At r = inner_radius + ε, eps_r → 0
        let a = 0.1_f64;
        let b = 0.3_f64;
        let cloak = CylindricalCloak::new(a, b).unwrap();
        let r_near = a + 1e-8;
        let eps_r = cloak.eps_radial(r_near);
        assert!(
            eps_r.abs() < 1e-6,
            "eps_r should approach 0 near inner radius, got {eps_r}"
        );
    }

    #[test]
    fn test_cloak_outer_eps() {
        // At r = outer_radius, all components should reduce to 1
        // ε_r(b) = (b-a)/b, ε_φ(b) = b/(b-a), ε_z(b) = (b/(b-a))²*(b-a)/b = b/(b-a)
        // They are NOT all 1 in general; but ε_r * ε_φ = 1 (reciprocal relationship)
        let a = 0.1_f64;
        let b = 0.3_f64;
        let cloak = CylindricalCloak::new(a, b).unwrap();
        let eps_r = cloak.eps_radial(b);
        let eps_phi = cloak.eps_azimuthal(b);
        // Product should be 1.0
        assert_abs_diff_eq!(eps_r * eps_phi, 1.0, epsilon = 1e-12);
        // ε_z at outer boundary
        let eps_z = cloak.eps_z(b);
        // ε_z(b) = (b/(b-a))² * (b-a)/b = b/(b-a)
        let expected_ez = b / (b - a);
        assert_abs_diff_eq!(eps_z, expected_ez, epsilon = 1e-12);
    }

    #[test]
    fn test_cloak_hidden_region() {
        let cloak = CylindricalCloak::new(0.1, 0.3).unwrap();
        assert!(cloak.is_cloaked(0.0, 0.05)); // inside
        assert!(!cloak.is_cloaked(0.2, 0.0)); // in cloak shell
        assert!(!cloak.is_cloaked(0.5, 0.0)); // outside
    }

    #[test]
    fn test_cloak_invalid_radii() {
        assert!(CylindricalCloak::new(0.3, 0.1).is_err()); // inner > outer
        assert!(CylindricalCloak::new(-0.1, 0.3).is_err()); // negative
        assert!(CylindricalCloak::new(0.1, 0.1).is_err()); // equal
    }

    // ── EM Concentrator ──────────────────────────────────────────────────────

    #[test]
    fn test_concentrator_field_enhancement() {
        let conc = EmConcentrator::new(0.05, 0.2, 2.0).unwrap();
        let fe = conc.field_enhancement();
        assert!(fe > 1.0, "field enhancement should be > 1, got {fe}");
        // (0.2/0.05)² = 16
        assert_abs_diff_eq!(fe, 16.0, epsilon = 1e-10);
    }

    #[test]
    fn test_concentrator_invalid_params() {
        assert!(EmConcentrator::new(0.2, 0.1, 2.0).is_err()); // inner > outer
        assert!(EmConcentrator::new(0.1, 0.2, 0.5).is_err()); // factor < 1
    }

    // ── GradedIndexARC ───────────────────────────────────────────────────────

    #[test]
    fn test_arc_index_profile_bounds() {
        let arc = GradedIndexARC::new(3.5, 1.0, 20, 200.0, 550.0);
        let profile = arc.index_profile();
        assert_eq!(profile.len(), 20);
        for &n in &profile {
            assert!(
                n >= arc.n_ambient * 0.9999 && n <= arc.n_substrate * 1.0001,
                "n={n} out of bounds [{}, {}]",
                arc.n_ambient,
                arc.n_substrate
            );
        }
    }

    #[test]
    fn test_arc_quintic_smooth() {
        let arc = GradedIndexARC::new(3.5, 1.0, 21, 200.0, 550.0);
        let profile = arc.quintic_index_profile();
        assert_eq!(profile.len(), 21);
        // quintic at t=0: p=0 → n = n_ambient
        assert_abs_diff_eq!(profile[0], arc.n_ambient, epsilon = 1e-12);
        // quintic at t=1: p=1 → n = n_substrate
        assert_abs_diff_eq!(profile[20], arc.n_substrate, epsilon = 1e-10);
    }

    // ── CoordTransformation ──────────────────────────────────────────────────

    #[test]
    fn test_jacobian_determinant_positive() {
        let transforms = [
            CoordTransformation::new(TransformType::LinearScale {
                scale_x: 2.0,
                scale_y: 1.5,
                scale_z: 1.0,
            }),
            CoordTransformation::new(TransformType::CylindricalCloak {
                inner_radius: 0.1,
                outer_radius: 0.3,
            }),
            CoordTransformation::new(TransformType::BeamBend {
                bend_radius: 1.0,
                angle_rad: std::f64::consts::PI / 4.0,
            }),
        ];
        for (t, xform) in transforms.iter().enumerate() {
            let det = xform.jacobian_det(0.2, 0.15, 0.0);
            assert!(det > 0.0, "transform {t}: det(J)={det} should be positive");
        }
    }

    #[test]
    fn test_linear_scale_jacobian() {
        let sx = 3.0;
        let sy = 2.0;
        let sz = 1.5;
        let xform = CoordTransformation::new(TransformType::LinearScale {
            scale_x: sx,
            scale_y: sy,
            scale_z: sz,
        });
        let j = xform.jacobian(1.0, 1.0, 1.0);
        // J should be diagonal with (sx, sy, sz)
        assert_abs_diff_eq!(j[0][0], sx, epsilon = 1e-14);
        assert_abs_diff_eq!(j[1][1], sy, epsilon = 1e-14);
        assert_abs_diff_eq!(j[2][2], sz, epsilon = 1e-14);
        // Off-diagonals should be zero
        assert_abs_diff_eq!(j[0][1], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(j[1][0], 0.0, epsilon = 1e-14);
        // det = sx * sy * sz
        let det = xform.jacobian_det(0.0, 0.0, 0.0);
        assert_abs_diff_eq!(det, sx * sy * sz, epsilon = 1e-12);
    }

    #[test]
    fn test_cylindrical_cloak_jacobian_det_positive_in_shell() {
        let a = 0.1_f64;
        let b = 0.3_f64;
        let xform = CoordTransformation::new(TransformType::CylindricalCloak {
            inner_radius: a,
            outer_radius: b,
        });
        // Test several points in the cloak shell
        for r_frac in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let r = a + r_frac * (b - a);
            let det = xform.jacobian_det(r, 0.0, 0.0);
            assert!(det > 0.0, "det(J)={det} at r={r} should be positive");
        }
    }

    #[test]
    fn test_spherical_cloak_eps_symmetric() {
        // Spherical cloak ε' tensor should be symmetric
        let xform = CoordTransformation::new(TransformType::SphericalCloak {
            inner_radius: 0.1,
            outer_radius: 0.3,
        });
        let eps = xform.transformed_eps(0.15, 0.1, 0.05);
        // Check symmetry: eps[i][j] == eps[j][i]
        for (i, row) in eps.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert_abs_diff_eq!(val, eps[j][i], epsilon = 1e-12);
            }
        }
    }
}
