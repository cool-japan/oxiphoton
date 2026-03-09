/// Transformation optics — coordinate-transformation-based cloaks and GRIN lenses.
use std::f64::consts::PI;

const C_LIGHT: f64 = 2.99792458e8;

// ---------------------------------------------------------------------------
// Cylindrical Cloak (Pendry–Schurig–Smith)
// ---------------------------------------------------------------------------

/// 2-D cylindrical electromagnetic cloak derived from the coordinate transformation
///
/// r′ = R₁ + r (R₂ − R₁) / R₂
///
/// which maps the annulus R₁ < r′ < R₂ to the full disk 0 < r < R₂.
/// The resulting material parameters are:
///
/// ε_r = μ_r = (r′ − R₁) / r′
/// ε_θ = μ_θ = r′ / (r′ − R₁)
/// ε_z = μ_z = (R₂ / (R₂ − R₁))²
#[derive(Debug, Clone)]
pub struct CylindricalCloak {
    /// Inner radius — boundary of the hidden (cloaked) region (m).
    pub r1_m: f64,
    /// Outer radius — boundary of the cloak (m).
    pub r2_m: f64,
}

impl CylindricalCloak {
    /// Radial permittivity / permeability component at radial position r (m).
    ///
    /// Returns 1.0 outside the cloak shell.
    pub fn eps_r(&self, r_m: f64) -> f64 {
        if r_m < self.r1_m || r_m > self.r2_m {
            return 1.0;
        }
        (r_m - self.r1_m) / r_m
    }

    /// Azimuthal permittivity / permeability component at radial position r (m).
    ///
    /// Returns 1.0 outside the cloak shell.
    pub fn eps_theta(&self, r_m: f64) -> f64 {
        if r_m < self.r1_m || r_m > self.r2_m {
            return 1.0;
        }
        let denom = r_m - self.r1_m;
        if denom.abs() < 1e-30 {
            return f64::INFINITY;
        }
        r_m / denom
    }

    /// Axial permittivity / permeability — constant throughout the cloak shell.
    pub fn eps_z(&self) -> f64 {
        let ratio = self.r2_m / (self.r2_m - self.r1_m).max(1e-30);
        ratio * ratio
    }

    /// The outer boundary of the cloak is always perfectly impedance-matched to free space.
    pub fn is_perfectly_matched_at_outer(&self) -> bool {
        true
    }

    /// Minimum refractive index within the cloak: n_min → 0 as r → R₁⁺.
    pub fn min_refractive_index(&self) -> f64 {
        0.0
    }

    /// Maximum refractive index at r = R₁ + dr (regularised by a small offset dr > 0).
    pub fn max_refractive_index_at(&self, dr: f64) -> f64 {
        self.eps_theta(self.r1_m + dr.abs().max(1e-30)).sqrt()
    }

    /// Idealised phase delay for a ray traversing the cloak shell.
    ///
    /// For a perfect transformation-optics cloak the optical path length through
    /// the shell equals the vacuum path length around the shell exterior, so the
    /// accumulated phase relative to free-space propagation is zero.  In practice
    /// dispersion and finite fabrication precision introduce non-zero delay; this
    /// method returns the ideal (zero) value.
    pub fn phase_delay_rad(&self, _wavelength_m: f64) -> f64 {
        0.0
    }

    /// Ratio of the cloaked scattering cross-section to the uncloaked geometrical
    /// cross-section 2 R₁.  Returns 0.0 for an ideal cloak.
    pub fn scattering_cross_section_reduction(&self) -> f64 {
        0.0
    }

    /// Outer-boundary optical path length (m) — equals 2 R₁ for an ideal cloak.
    pub fn outer_optical_path_m(&self) -> f64 {
        2.0 * self.r1_m
    }

    /// Anisotropy ratio ε_θ / ε_r evaluated at r (diverges at inner boundary).
    pub fn anisotropy_ratio(&self, r_m: f64) -> f64 {
        let er = self.eps_r(r_m);
        if er.abs() < 1e-30 {
            return f64::INFINITY;
        }
        self.eps_theta(r_m) / er
    }
}

// ---------------------------------------------------------------------------
// Carpet / Ground-Plane Cloak (quasi-conformal mapping)
// ---------------------------------------------------------------------------

/// Carpet cloak that hides a bump on a reflecting ground plane using a quasi-conformal
/// coordinate transformation.  Requires only a modest anisotropic index variation.
#[derive(Debug, Clone)]
pub struct CarpetCloak {
    /// Height of the bump above the ground plane (m).
    pub bump_height_m: f64,
    /// Full width of the bump at the ground plane (m).
    pub bump_width_m: f64,
    /// Background refractive index of the cloak medium.
    pub n_background: f64,
}

impl CarpetCloak {
    /// Required fractional index variation Δn/n ≈ h / w.
    pub fn index_variation(&self) -> f64 {
        self.bump_height_m / self.bump_width_m.max(1e-30)
    }

    /// Returns `true` when the required index variation is small enough to be fabricated
    /// with isotropic dielectrics (heuristic threshold: Δn/n < 0.5).
    pub fn is_feasible(&self) -> bool {
        self.index_variation() < 0.5
    }

    /// Approximate relative bandwidth of the carpet cloak:
    ///
    /// Δf / f ≈ 1 / (n_bg × (Δn/n) × aspect_ratio)
    ///
    /// where aspect_ratio = bump_width / bump_height.
    pub fn bandwidth_fraction(&self) -> f64 {
        let delta_n_over_n = self.index_variation();
        if delta_n_over_n.abs() < 1e-30 || self.n_background < 1e-30 {
            return f64::INFINITY;
        }
        let aspect = self.bump_width_m / self.bump_height_m.max(1e-30);
        1.0 / (self.n_background * delta_n_over_n * aspect)
    }

    /// Required index range [n_min, n_max] within the cloak region.
    pub fn index_range(&self) -> (f64, f64) {
        let delta_n = self.n_background * self.index_variation();
        (
            (self.n_background - delta_n).max(1.0),
            self.n_background + delta_n,
        )
    }
}

// ---------------------------------------------------------------------------
// Luneburg Lens
// ---------------------------------------------------------------------------

/// Luneburg lens — a gradient-index (GRIN) sphere with index profile
///
/// n(r) = √(2 − (r/R)²)
///
/// which focuses a plane wave to a point on the opposite surface.
#[derive(Debug, Clone)]
pub struct LuneburgLens {
    /// Sphere radius (m).
    pub radius_m: f64,
    /// Maximum refractive index at the centre.  For a classical Luneburg lens this
    /// equals √2, but generalised designs may use different values.
    pub n_max: f64,
}

impl LuneburgLens {
    /// Refractive index at radial distance r from the centre:
    ///
    /// n(r) = √(2 − (r/R)²)
    pub fn index_at(&self, r_m: f64) -> f64 {
        (2.0 - (r_m / self.radius_m.max(1e-30)).powi(2))
            .max(0.0)
            .sqrt()
    }

    /// Focal point location: parallel rays are focused to the surface (r = R).
    pub fn focal_point_m(&self) -> f64 {
        self.radius_m
    }

    /// Numerical aperture of the Luneburg lens: NA = 1.
    pub fn numerical_aperture(&self) -> f64 {
        1.0
    }

    /// Volume-averaged refractive index (numerical integration with N steps).
    pub fn average_index(&self, n_steps: usize) -> f64 {
        let n = n_steps.max(2);
        let dr = self.radius_m / n as f64;
        let mut sum_n_r2 = 0.0_f64;
        let mut sum_r2 = 0.0_f64;
        for i in 0..n {
            let r = (i as f64 + 0.5) * dr;
            sum_n_r2 += self.index_at(r) * r * r;
            sum_r2 += r * r;
        }
        if sum_r2 < 1e-60 {
            self.n_max
        } else {
            sum_n_r2 / sum_r2
        }
    }

    /// Geometric phase delay for a ray entering at impact parameter b (m).
    ///
    /// Returns the approximate Luneburg phase delay (rad) relative to vacuum.
    pub fn phase_delay_rad(&self, wavelength_m: f64, impact_param_m: f64) -> f64 {
        let b = impact_param_m.abs().min(self.radius_m);
        let r_entry = (self.radius_m * self.radius_m - b * b).max(0.0).sqrt();
        // Approximate path-length enhancement factor from GRIN vs vacuum
        let path_factor = self.index_at(r_entry);
        2.0 * PI / wavelength_m.max(1e-30) * path_factor * 2.0 * r_entry
    }

    /// Free-space wavelength at which the lens diameter equals one wavelength.
    pub fn lambda_diameter_m(&self) -> f64 {
        2.0 * self.radius_m
    }
}

// ---------------------------------------------------------------------------
// Maxwell Fish-Eye Lens (bonus GRIN lens from transformation optics)
// ---------------------------------------------------------------------------

/// Maxwell fish-eye lens: n(r) = 2 n₀ / (1 + (r/R)²).
///
/// Maps a point source on one hemisphere to a perfect image on the antipodal point.
#[derive(Debug, Clone)]
pub struct MaxwellFishEye {
    /// Characteristic radius (m).
    pub radius_m: f64,
    /// Index at the centre (n₀).
    pub n0: f64,
}

impl MaxwellFishEye {
    /// Refractive index at r: n(r) = 2 n₀ / (1 + (r/R)²).
    pub fn index_at(&self, r_m: f64) -> f64 {
        let rho = r_m / self.radius_m.max(1e-30);
        2.0 * self.n0 / (1.0 + rho * rho)
    }

    /// Index at the edge (r = R): n_edge = n₀.
    pub fn index_at_edge(&self) -> f64 {
        self.n0
    }

    /// Phase velocity at r.
    pub fn phase_velocity_at(&self, r_m: f64) -> f64 {
        C_LIGHT / self.index_at(r_m).max(1e-30)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cylindrical_cloak_eps_r_at_inner_boundary() {
        let cloak = CylindricalCloak {
            r1_m: 0.1,
            r2_m: 0.2,
        };
        // At r = R1 + dr → eps_r → 0
        let eps = cloak.eps_r(0.1 + 1e-9);
        assert!(
            eps < 1e-8,
            "eps_r should → 0 at inner boundary, got {}",
            eps
        );
    }

    #[test]
    fn cylindrical_cloak_eps_theta_diverges_near_inner() {
        let cloak = CylindricalCloak {
            r1_m: 0.1,
            r2_m: 0.2,
        };
        let eps = cloak.eps_theta(0.1 + 1e-6);
        assert!(
            eps > 1e4,
            "eps_theta should be very large near R1, got {}",
            eps
        );
    }

    #[test]
    fn cylindrical_cloak_outside_returns_unity() {
        let cloak = CylindricalCloak {
            r1_m: 0.1,
            r2_m: 0.2,
        };
        assert_eq!(cloak.eps_r(0.0), 1.0);
        assert_eq!(cloak.eps_r(0.3), 1.0);
        assert_eq!(cloak.eps_theta(0.3), 1.0);
    }

    #[test]
    fn carpet_cloak_feasibility() {
        let carpet = CarpetCloak {
            bump_height_m: 1e-3,
            bump_width_m: 10e-3,
            n_background: 1.5,
        };
        // Δn/n = 0.1 < 0.5 → feasible
        assert!(carpet.is_feasible());

        let steep = CarpetCloak {
            bump_height_m: 5e-3,
            bump_width_m: 6e-3,
            n_background: 1.5,
        };
        assert!(!steep.is_feasible());
    }

    #[test]
    fn luneburg_index_profile() {
        let lens = LuneburgLens {
            radius_m: 1.0,
            n_max: 2.0_f64.sqrt(),
        };
        let n_center = lens.index_at(0.0);
        let n_edge = lens.index_at(1.0);
        assert!(
            (n_center - 2.0_f64.sqrt()).abs() < 1e-10,
            "n(0) should be √2"
        );
        assert!((n_edge - 1.0).abs() < 1e-10, "n(R) should be 1");
    }

    #[test]
    fn luneburg_focal_point_equals_radius() {
        let lens = LuneburgLens {
            radius_m: 0.05,
            n_max: 2.0_f64.sqrt(),
        };
        assert_eq!(lens.focal_point_m(), 0.05);
        assert_eq!(lens.numerical_aperture(), 1.0);
    }

    #[test]
    fn maxwell_fish_eye_index() {
        let mfe = MaxwellFishEye {
            radius_m: 1.0,
            n0: 2.0,
        };
        assert!((mfe.index_at(0.0) - 4.0).abs() < 1e-10, "n(0) = 2 n0");
        assert!((mfe.index_at(1.0) - 2.0).abs() < 1e-10, "n(R) = n0");
        assert_eq!(mfe.index_at_edge(), 2.0);
    }

    #[test]
    fn cylindrical_cloak_phase_delay_ideal() {
        let cloak = CylindricalCloak {
            r1_m: 0.05,
            r2_m: 0.10,
        };
        assert_eq!(cloak.phase_delay_rad(500e-9), 0.0);
        assert_eq!(cloak.scattering_cross_section_reduction(), 0.0);
    }
}
