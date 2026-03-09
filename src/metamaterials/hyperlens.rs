/// Hyperlens and superlens imaging — sub-diffraction focusing via evanescent amplification.
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Pendry Superlens
// ---------------------------------------------------------------------------

/// Flat slab superlens (Pendry 2000): a slab with ε ≈ μ ≈ −1 amplifies evanescent
/// waves and reconstructs sub-diffraction images.
#[derive(Debug, Clone)]
pub struct PendrySuperLens {
    /// Slab thickness d (m).
    pub slab_thickness_m: f64,
    /// Real part of slab permittivity (ideally −1).
    pub eps_slab: f64,
    /// Operating free-space wavelength (m).
    pub wavelength_m: f64,
}

impl PendrySuperLens {
    /// Free-space wave number k₀ = 2π / λ.
    pub fn k0(&self) -> f64 {
        2.0 * PI / self.wavelength_m.max(1e-30)
    }

    /// Evanescent wave amplification factor |T| for a transverse wave number kx (in units of k₀).
    ///
    /// For an ideal lossless slab with ε = −1 and kx > k₀:
    ///
    /// kz = i √(kx² − k₀²) = i κ
    /// |T| = exp(κ d)   (amplification)
    ///
    /// This returns the ideal amplitude gain; in a real device, loss limits the maximum kx.
    pub fn evanescent_amplification(&self, kx_over_k0: f64) -> f64 {
        let k0 = self.k0();
        let kx = kx_over_k0 * k0;
        if kx_over_k0.abs() <= 1.0 {
            // Propagating wave — transmission amplitude ≈ 1 for matched slab
            1.0
        } else {
            let kappa = (kx * kx - k0 * k0).max(0.0).sqrt();
            (kappa * self.slab_thickness_m).exp()
        }
    }

    /// Sub-diffraction resolution (m) limited by material loss Im(ε).
    ///
    /// The maximum amplified spatial frequency is:
    ///
    /// κ_max ≈ (1 / d) ln(1 / |Im(ε)|)
    ///
    /// giving a resolution Δr ≈ π / κ_max.
    pub fn resolution_m(&self, eps_imag: f64) -> f64 {
        let eps_im = eps_imag.abs().max(1e-30);
        // κ_max d ≈ ln(1/|Im ε|)
        let kappa_max_d = (-eps_im.ln()).max(1e-10);
        let kappa_max = kappa_max_d / self.slab_thickness_m.max(1e-30);
        PI / kappa_max
    }

    /// Image plane distance from the second surface of the slab.
    ///
    /// For a superlens with equal object and image distances, the image forms at
    /// d_image = d_object = d / 2.
    pub fn image_distance_m(&self) -> f64 {
        self.slab_thickness_m / 2.0
    }

    /// Magnification of the flat superlens: always 1.
    pub fn magnification(&self) -> f64 {
        1.0
    }

    /// Maximum spatial frequency supported before the evanescent amplification
    /// becomes limited by loss (heuristic: gain < exp(10)).
    pub fn max_spatial_frequency_k0(&self, eps_imag: f64) -> f64 {
        let eps_im = eps_imag.abs().max(1e-30);
        let kappa_max_d = (-eps_im.ln()).max(1e-10);
        let kappa_max = kappa_max_d / self.slab_thickness_m.max(1e-30);
        let k0 = self.k0();
        kappa_max / k0
    }

    /// Returns `true` when the slab permittivity is close to the perfect-lens value (ε ≈ −1).
    pub fn is_near_perfect_lens(&self, tolerance: f64) -> bool {
        (self.eps_slab + 1.0).abs() < tolerance
    }
}

// ---------------------------------------------------------------------------
// Optical Hyperlens
// ---------------------------------------------------------------------------

/// Cylindrical optical hyperlens — a strongly anisotropic curved metamaterial shell
/// that converts evanescent near-field components into propagating far-field waves,
/// enabling real-time, far-field sub-diffraction imaging with magnification M = R₂/R₁.
#[derive(Debug, Clone)]
pub struct OpticalHyperlens {
    /// Inner radius R₁ — objects are placed at this surface (m).
    pub inner_radius_m: f64,
    /// Outer radius R₂ — images form at this surface (m).
    pub outer_radius_m: f64,
    /// Radial permittivity ε_r (negative for Type-II hyperbolic metamaterial).
    pub eps_radial: f64,
    /// Tangential permittivity ε_θ (positive for Type-II).
    pub eps_tangential: f64,
    /// Operating wavelength in free space (m).
    pub wavelength_m: f64,
}

impl OpticalHyperlens {
    /// Magnification M = R₂ / R₁.
    pub fn magnification(&self) -> f64 {
        self.outer_radius_m / self.inner_radius_m.max(1e-30)
    }

    /// Returns `true` when the medium is hyperbolic: ε_r × ε_θ < 0.
    pub fn is_hyperbolic(&self) -> bool {
        self.eps_radial * self.eps_tangential < 0.0
    }

    /// Minimum feature size resolvable at the inner surface:
    ///
    /// Δr_inner = λ / (2 √(|ε_θ / ε_r|))
    pub fn inner_surface_resolution_m(&self) -> f64 {
        if self.eps_radial.abs() < 1e-30 {
            return f64::INFINITY;
        }
        let factor = (self.eps_tangential.abs() / self.eps_radial.abs()).sqrt();
        self.wavelength_m / (2.0 * factor.max(1e-30))
    }

    /// Conventional diffraction-limited resolution at the outer surface: λ / (2 M).
    pub fn outer_surface_resolution_m(&self) -> f64 {
        self.wavelength_m / (2.0 * self.magnification().max(1e-30))
    }

    /// Maximum transverse spatial frequency (in units of k₀) that can propagate
    /// through the hyperlens for Type-II (ε_r < 0, ε_θ > 0):
    ///
    /// k_max / k₀ = √(|ε_θ| / |ε_r|)
    pub fn max_spatial_frequency(&self) -> f64 {
        (self.eps_tangential.abs() / self.eps_radial.abs().max(1e-30)).sqrt()
    }

    /// Returns `true` when the device is in the hyperbolic propagation regime.
    pub fn is_in_hyperbolic_regime(&self) -> bool {
        self.is_hyperbolic()
    }

    /// Subwavelength resolution improvement factor relative to the classical limit λ/2:
    ///
    /// η = (λ/2) / Δr_inner = √(|ε_θ / ε_r|)
    pub fn resolution_enhancement_factor(&self) -> f64 {
        if self.eps_radial.abs() < 1e-30 {
            return f64::INFINITY;
        }
        (self.eps_tangential.abs() / self.eps_radial.abs()).sqrt()
    }

    /// Effective numerical aperture at the inner surface:
    ///
    /// NA_eff = √(|ε_θ / ε_r|) × (R₁ / R₂)
    ///
    /// This quantifies how many evanescent orders are collected and magnified.
    pub fn effective_numerical_aperture(&self) -> f64 {
        let ratio = (self.eps_tangential.abs() / self.eps_radial.abs().max(1e-30)).sqrt();
        ratio * self.inner_radius_m / self.outer_radius_m.max(1e-30)
    }

    /// Phase accumulated by a ray travelling from R₁ to R₂ in the hyperlens (approximate).
    ///
    /// Δφ ≈ k₀ √(|ε_θ|) (R₂ − R₁)
    pub fn phase_accumulation_rad(&self) -> f64 {
        let k0 = 2.0 * PI / self.wavelength_m.max(1e-30);
        let n_eff = self.eps_tangential.abs().sqrt();
        k0 * n_eff * (self.outer_radius_m - self.inner_radius_m).abs()
    }
}

// ---------------------------------------------------------------------------
// Spherical Superlens (near-field)
// ---------------------------------------------------------------------------

/// Spherical near-field superlens: a spherical shell of double-negative material
/// that focuses evanescent fields around a point source.
#[derive(Debug, Clone)]
pub struct SphericalSuperlens {
    /// Inner radius (m).
    pub r1_m: f64,
    /// Outer radius (m).
    pub r2_m: f64,
    /// Real part of the shell permittivity.
    pub eps_shell: f64,
    /// Real part of the shell permeability.
    pub mu_shell: f64,
    /// Free-space wavelength (m).
    pub wavelength_m: f64,
}

impl SphericalSuperlens {
    /// Shell refractive index n = −√(ε μ) for double-negative material.
    pub fn refractive_index(&self) -> f64 {
        let sign = if self.eps_shell < 0.0 && self.mu_shell < 0.0 {
            -1.0
        } else {
            1.0
        };
        sign * (self.eps_shell.abs() * self.mu_shell.abs()).sqrt()
    }

    /// Returns `true` when both ε and μ are negative.
    pub fn is_double_negative(&self) -> bool {
        self.eps_shell < 0.0 && self.mu_shell < 0.0
    }

    /// Geometric magnification of a concentric shell: M = R₂ / R₁.
    pub fn geometric_magnification(&self) -> f64 {
        self.r2_m / self.r1_m.max(1e-30)
    }

    /// Solid angle subtended by the inner surface from the centre.
    pub fn solid_angle_sr(&self) -> f64 {
        4.0 * PI
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hyperlens_magnification() {
        let hl = OpticalHyperlens {
            inner_radius_m: 250e-9,
            outer_radius_m: 500e-9,
            eps_radial: -3.0,
            eps_tangential: 4.0,
            wavelength_m: 365e-9,
        };
        assert!(
            (hl.magnification() - 2.0).abs() < 1e-10,
            "Magnification should be 2.0, got {}",
            hl.magnification()
        );
        assert!(
            hl.is_hyperbolic(),
            "Should be hyperbolic (ε_r < 0, ε_θ > 0)"
        );
    }

    #[test]
    fn hyperlens_inner_resolution() {
        let hl = OpticalHyperlens {
            inner_radius_m: 250e-9,
            outer_radius_m: 500e-9,
            eps_radial: -3.0,
            eps_tangential: 4.0,
            wavelength_m: 365e-9,
        };
        let res = hl.inner_surface_resolution_m();
        // Δr = λ / (2 √(4/3)) ≈ 365 nm / (2 × 1.155) ≈ 158 nm < λ/2
        assert!(
            res < hl.wavelength_m / 2.0,
            "Inner resolution {} should be sub-diffraction",
            res
        );
    }

    #[test]
    fn superlens_evanescent_amplification_propagating() {
        let sl = PendrySuperLens {
            slab_thickness_m: 40e-9,
            eps_slab: -1.0,
            wavelength_m: 365e-9,
        };
        // Propagating component kx < k0 → amplification = 1
        let amp = sl.evanescent_amplification(0.5);
        assert!(
            (amp - 1.0).abs() < 1e-10,
            "Propagating amp should be 1.0, got {}",
            amp
        );
    }

    #[test]
    fn superlens_evanescent_amplification_grows() {
        let sl = PendrySuperLens {
            slab_thickness_m: 40e-9,
            eps_slab: -1.0,
            wavelength_m: 365e-9,
        };
        let amp_low = sl.evanescent_amplification(2.0);
        let amp_high = sl.evanescent_amplification(4.0);
        assert!(
            amp_high > amp_low,
            "Higher kx should give more amplification"
        );
    }

    #[test]
    fn superlens_resolution_improves_with_less_loss() {
        let sl1 = PendrySuperLens {
            slab_thickness_m: 40e-9,
            eps_slab: -1.0,
            wavelength_m: 365e-9,
        };
        let sl2 = PendrySuperLens {
            slab_thickness_m: 40e-9,
            eps_slab: -1.0,
            wavelength_m: 365e-9,
        };
        let res_low_loss = sl1.resolution_m(0.01);
        let res_high_loss = sl2.resolution_m(0.1);
        assert!(
            res_low_loss < res_high_loss,
            "Lower loss should give finer resolution"
        );
    }

    #[test]
    fn superlens_image_distance() {
        let sl = PendrySuperLens {
            slab_thickness_m: 80e-9,
            eps_slab: -1.0,
            wavelength_m: 365e-9,
        };
        assert!((sl.image_distance_m() - 40e-9).abs() < 1e-20);
        assert_eq!(sl.magnification(), 1.0);
    }

    #[test]
    fn spherical_superlens_double_negative() {
        let ssl = SphericalSuperlens {
            r1_m: 100e-9,
            r2_m: 200e-9,
            eps_shell: -1.5,
            mu_shell: -1.0,
            wavelength_m: 532e-9,
        };
        assert!(ssl.is_double_negative());
        assert!(ssl.refractive_index() < 0.0, "n should be negative for DNM");
    }

    #[test]
    fn hyperlens_max_spatial_frequency() {
        let hl = OpticalHyperlens {
            inner_radius_m: 150e-9,
            outer_radius_m: 450e-9,
            eps_radial: -2.0,
            eps_tangential: 3.0,
            wavelength_m: 400e-9,
        };
        // k_max/k0 = sqrt(3/2) ≈ 1.225
        let kmax = hl.max_spatial_frequency();
        assert!(
            (kmax - (1.5_f64).sqrt()).abs() < 1e-10,
            "k_max should be √(3/2), got {}",
            kmax
        );
    }
}
