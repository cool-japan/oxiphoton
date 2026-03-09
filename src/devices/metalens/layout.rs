//! Metalens layout generation.
//!
//! Converts a phase profile into a 2D array of nanopost diameters,
//! ready for fabrication (GDS export).
//!
//! Layout steps:
//!   1. Define aperture (circular lens of diameter D_lens)
//!   2. For each array site (i·Λ, j·Λ), compute required phase φ(x,y)
//!   3. Look up nanopost diameter from phase library
//!   4. Record (x, y, diameter) for each post
//!
//! The phase profile for a focusing lens:
//!   φ(x,y) = -k₀ · (√(x²+y²+f²) - f) + φ₀
//! where f is the focal length.

use std::f64::consts::PI;

/// A single nanopost element in the metalens layout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NanopostElement {
    /// x-position (m)
    pub x: f64,
    /// y-position (m)
    pub y: f64,
    /// Post diameter (m)
    pub diameter: f64,
}

/// Complete metalens layout: positions and diameters of all nanoposts.
#[derive(Debug, Clone)]
pub struct MetalensLayout {
    /// Array of nanopost elements
    pub posts: Vec<NanopostElement>,
    /// Lens aperture diameter (m)
    pub aperture: f64,
    /// Array pitch Λ (m)
    pub pitch: f64,
    /// Design wavelength (m)
    pub wavelength: f64,
    /// Focal length (m)
    pub focal_length: f64,
}

impl MetalensLayout {
    /// Generate a focusing metalens layout.
    ///
    /// - `aperture`: lens diameter (m)
    /// - `focal_length`: focal length (m)
    /// - `pitch`: array pitch (m)
    /// - `wavelength`: design wavelength (m)
    /// - `phase_fn`: function mapping (x,y) → phase (rad)
    pub fn new_with_phase<F>(
        aperture: f64,
        focal_length: f64,
        pitch: f64,
        wavelength: f64,
        phase_fn: F,
    ) -> Self
    where
        F: Fn(f64, f64) -> f64,
    {
        let r_max = aperture / 2.0;
        let n_half = (r_max / pitch).ceil() as i32;
        let mut posts = Vec::new();

        for iy in -n_half..=n_half {
            for ix in -n_half..=n_half {
                let x = ix as f64 * pitch;
                let y = iy as f64 * pitch;
                let r = (x * x + y * y).sqrt();
                if r > r_max {
                    continue;
                }
                let phi = phase_fn(x, y);
                // Default diameter mapping: D = min + (max-min) × φ/(2π)
                // (in practice, use NanopostLibrary::diameter_for_phase)
                let phi_norm = phi.rem_euclid(2.0 * PI) / (2.0 * PI);
                let d_min = 0.1 * pitch;
                let d_max = 0.8 * pitch;
                let diameter = d_min + phi_norm * (d_max - d_min);
                posts.push(NanopostElement { x, y, diameter });
            }
        }

        Self {
            posts,
            aperture,
            pitch,
            wavelength,
            focal_length,
        }
    }

    /// Generate a standard focusing lens.
    ///
    /// Phase profile: φ(r) = -k₀(√(r²+f²) - f) mod 2π
    pub fn focusing(aperture: f64, focal_length: f64, pitch: f64, wavelength: f64) -> Self {
        let k0 = 2.0 * PI / wavelength;
        Self::new_with_phase(aperture, focal_length, pitch, wavelength, |x, y| {
            let r = (x * x + y * y).sqrt();
            -k0 * ((r * r + focal_length * focal_length).sqrt() - focal_length)
        })
    }

    /// Generate a vortex (OAM) lens with topological charge m.
    ///
    /// Phase profile: φ(x,y) = m·atan2(y, x)
    pub fn vortex(aperture: f64, pitch: f64, wavelength: f64, topological_charge: i32) -> Self {
        let m = topological_charge as f64;
        Self::new_with_phase(aperture, 0.0, pitch, wavelength, |x, y| m * y.atan2(x))
    }

    /// Number of nanoposts in the layout.
    pub fn n_posts(&self) -> usize {
        self.posts.len()
    }

    /// Numerical aperture: NA = sin(θ) ≈ (aperture/2) / f.
    pub fn numerical_aperture(&self) -> f64 {
        if self.focal_length <= 0.0 {
            return 0.0;
        }
        (self.aperture / 2.0) / (self.aperture / 2.0).hypot(self.focal_length)
    }

    /// Diffraction-limited spot size (Airy disk radius): r = 1.22·λ/NA.
    pub fn airy_radius(&self) -> f64 {
        let na = self.numerical_aperture();
        if na < 1e-30 {
            return f64::INFINITY;
        }
        1.22 * self.wavelength / na
    }

    /// Diameter range (min, max) across all posts (m).
    pub fn diameter_range(&self) -> (f64, f64) {
        let d_min = self
            .posts
            .iter()
            .map(|p| p.diameter)
            .fold(f64::INFINITY, f64::min);
        let d_max = self
            .posts
            .iter()
            .map(|p| p.diameter)
            .fold(f64::NEG_INFINITY, f64::max);
        (d_min, d_max)
    }

    /// Bounding box of the layout: (x_min, x_max, y_min, y_max).
    pub fn bounding_box(&self) -> (f64, f64, f64, f64) {
        let x_min = self.posts.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        let x_max = self
            .posts
            .iter()
            .map(|p| p.x)
            .fold(f64::NEG_INFINITY, f64::max);
        let y_min = self.posts.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let y_max = self
            .posts
            .iter()
            .map(|p| p.y)
            .fold(f64::NEG_INFINITY, f64::max);
        (x_min, x_max, y_min, y_max)
    }
}

// ---------------------------------------------------------------------------
// AchromaticMetalens — broadband focusing with dispersion compensation
// ---------------------------------------------------------------------------

/// Achromatic metalens: a flat lens designed to bring two distinct wavelengths
/// to the same focal point by adding a compensating dispersive phase term.
///
/// Phase design:
///   φ_total(r, λ) = φ_hyp(r) + φ_disp(r, λ)
///
/// where
///   φ_hyp(r) = −k₀_c · (√(r²+f²) − f)          (focusing at centre λ)
///   φ_disp(r, λ) = α(r) · (1/λ − 1/λ_c)          (equalises focus at λ₁, λ₂)
///
/// The dispersion coefficient α(r) is chosen so that both λ₁ and λ₂ focus at f:
///   α(r) = −φ_hyp(r) · (λ_c / (1/λ₁ − 1/λ₂)) · (1/λ₁ − 1/λ₂) / (1/λ₁ − 1/λ₂)
///
/// Simplified: α(r) = −(φ_hyp(r,λ₁) − φ_hyp(r,λ₂)) · λ₁·λ₂ / (λ₂−λ₁)
#[derive(Debug, Clone, Copy)]
pub struct AchromaticMetalens {
    /// Focal length (m), constant for both wavelengths.
    pub focal_length: f64,
    /// Lens diameter (m).
    pub diameter: f64,
    /// First design wavelength λ₁ (m).
    pub wavelength1: f64,
    /// Second design wavelength λ₂ (m).
    pub wavelength2: f64,
}

impl AchromaticMetalens {
    /// Create an achromatic metalens.
    pub fn new(focal_length: f64, diameter: f64, wavelength1: f64, wavelength2: f64) -> Self {
        Self {
            focal_length,
            diameter,
            wavelength1,
            wavelength2,
        }
    }

    /// Centre design wavelength λ_c = (λ₁ + λ₂) / 2.
    pub fn centre_wavelength(&self) -> f64 {
        (self.wavelength1 + self.wavelength2) / 2.0
    }

    /// Hyperbolic focusing phase at radius r (m) for the centre wavelength (rad).
    ///
    /// φ_hyp(r) = −k₀_c · (√(r²+f²) − f)
    pub fn phase_profile(&self, r: f64) -> f64 {
        let lambda_c = self.centre_wavelength();
        let k0_c = 2.0 * PI / lambda_c;
        -k0_c * ((r * r + self.focal_length * self.focal_length).sqrt() - self.focal_length)
    }

    /// Dispersion-compensation phase at radius r (m) for wavelength λ.
    ///
    /// Δφ_disp(r, λ) = α(r) · (1/λ − 1/λ_c)
    ///
    /// where α(r) is chosen to equalise focus between λ₁ and λ₂:
    ///   α(r) = [φ_hyp(r,λ₁) − φ_hyp(r,λ₂)] / (1/λ₁ − 1/λ₂)
    ///         = [φ_hyp(r,λ₁) − φ_hyp(r,λ₂)] · λ₁·λ₂ / (λ₂ − λ₁)
    pub fn dispersion_compensation_phase(&self, r: f64, wavelength: f64) -> f64 {
        let l1 = self.wavelength1;
        let l2 = self.wavelength2;
        let lc = self.centre_wavelength();
        let f = self.focal_length;
        // Phase at λ₁ and λ₂
        let k0_1 = 2.0 * PI / l1;
        let k0_2 = 2.0 * PI / l2;
        let path = (r * r + f * f).sqrt() - f;
        let phi1 = -k0_1 * path;
        let phi2 = -k0_2 * path;
        let d_inv_lambda = 1.0 / l1 - 1.0 / l2;
        if d_inv_lambda.abs() < 1e-30 {
            return 0.0;
        }
        let alpha = (phi1 - phi2) / d_inv_lambda;
        // Dispersion compensation relative to centre wavelength
        alpha * (1.0 / wavelength - 1.0 / lc)
    }

    /// Total phase at radius r (m) for wavelength λ.
    pub fn total_phase(&self, r: f64, wavelength: f64) -> f64 {
        self.phase_profile(r) + self.dispersion_compensation_phase(r, wavelength)
    }

    /// Estimated focusing efficiency: 0.5–0.9 depending on implementation quality.
    ///
    /// Uses an analytical model: η ≈ sinc²(Δφ_max / (2π)) where Δφ_max is the
    /// maximum dispersion-compensation phase excursion across the aperture.
    pub fn efficiency_estimate(&self) -> f64 {
        // Evaluate max dispersion phase at edge of aperture
        let r_edge = self.diameter / 2.0;
        let dp_max = self
            .dispersion_compensation_phase(r_edge, self.wavelength1)
            .abs();
        // sinc²(x/2π) model; clamp to [0.5, 0.9] to reflect realistic TiO2/Si implementations
        let x = dp_max / (2.0 * PI);
        let sinc = if x.abs() < 1e-12 {
            1.0
        } else {
            (PI * x).sin() / (PI * x)
        };
        (sinc * sinc).clamp(0.5, 0.9)
    }

    /// Numerical aperture: NA = (D/2) / √((D/2)² + f²).
    pub fn numerical_aperture(&self) -> f64 {
        let r = self.diameter / 2.0;
        r / r.hypot(self.focal_length)
    }
}

// ---------------------------------------------------------------------------
// ZonePlate — binary Fresnel zone plate
// ---------------------------------------------------------------------------

/// Binary Fresnel zone plate for comparison with a metalens.
///
/// The m-th zone radius satisfies: r_m = √(m·λ·f + m²·λ²/4) ≈ √(m·λ·f)
/// for m·λ << f (paraxial approximation).
///
/// Theoretical first-order efficiency: η = 1/π² ≈ 10.1% (binary amplitude)
/// A phase zone plate achieves 4/π² ≈ 40.5%.
#[derive(Debug, Clone, Copy)]
pub struct ZonePlate {
    /// Focal length (m).
    pub focal_length: f64,
    /// Design wavelength (m).
    pub wavelength: f64,
    /// Aperture diameter (m).
    pub diameter: f64,
}

impl ZonePlate {
    /// Create a zone plate.
    pub fn new(focal_length: f64, wavelength: f64, diameter: f64) -> Self {
        Self {
            focal_length,
            wavelength,
            diameter,
        }
    }

    /// Radii of all zones within the aperture (m).
    ///
    /// Exact formula: r_m = √(m·λ·f + (m·λ/2)²)
    pub fn zone_radii(&self) -> Vec<f64> {
        let r_max = self.diameter / 2.0;
        let l = self.wavelength;
        let f = self.focal_length;
        let mut radii = Vec::new();
        let mut m = 1usize;
        loop {
            let r_m = (m as f64 * l * f + (m as f64 * l / 2.0).powi(2)).sqrt();
            if r_m > r_max {
                break;
            }
            radii.push(r_m);
            m += 1;
        }
        radii
    }

    /// Total number of zones within the aperture.
    pub fn n_zones(&self) -> usize {
        self.zone_radii().len()
    }

    /// Theoretical first-order diffraction efficiency of a binary amplitude zone plate.
    ///
    /// η = 1/π² ≈ 0.1013
    pub fn theoretical_efficiency(&self) -> f64 {
        1.0 / (PI * PI)
    }

    /// Numerical aperture: NA = (D/2) / √((D/2)² + f²).
    pub fn numerical_aperture(&self) -> f64 {
        let r = self.diameter / 2.0;
        r / r.hypot(self.focal_length)
    }

    /// Diffraction-limited spot size (Airy disk radius).
    pub fn airy_radius(&self) -> f64 {
        let na = self.numerical_aperture();
        if na < 1e-30 {
            return f64::INFINITY;
        }
        1.22 * self.wavelength / na
    }
}

// ---------------------------------------------------------------------------
// FillFactorMap — radial fill-factor library for metalens unit-cell design
// ---------------------------------------------------------------------------

/// Radial fill-factor map for metalens unit-cell design.
///
/// Maps a required phase at a given radius to a fill factor using a piecewise
/// linear lookup.  The fill factor is linearly interpolated between 0 and 1
/// as the target phase sweeps from 0 to 2π.
///
/// In a real design workflow the lookup would come from rigorous EM simulation
/// (e.g. RCWA) of the unit cell.  This model uses a simple linear approximation.
#[derive(Debug, Clone)]
pub struct FillFactorMap {
    /// Number of radial rings.
    pub n_rings: usize,
    /// Outer diameter of the lens (m).
    pub max_diameter: f64,
}

impl FillFactorMap {
    /// Create a fill-factor map.
    pub fn new(n_rings: usize, max_diameter: f64) -> Self {
        Self {
            n_rings,
            max_diameter,
        }
    }

    /// Ring width (m).
    pub fn ring_width(&self) -> f64 {
        if self.n_rings == 0 {
            return 0.0;
        }
        (self.max_diameter / 2.0) / self.n_rings as f64
    }

    /// Fill factor at radius r (m) for a target phase φ (rad).
    ///
    /// Uses a linear mapping: ff = φ.rem_euclid(2π) / 2π, clamped to [0.1, 0.9].
    ///
    /// This represents the simplest possible unit-cell library, where phase
    /// is monotonically controlled by fill factor.
    pub fn fill_factor_at(&self, _r: f64, target_phase: f64) -> f64 {
        let phi_norm = target_phase.rem_euclid(2.0 * PI) / (2.0 * PI);
        // Map [0,1] → [0.1, 0.9] to stay in a physically realisable range
        (0.1 + phi_norm * 0.8).clamp(0.1, 0.9)
    }

    /// Generate a radial layout: Vec<(radius, fill_factor)> for a focusing lens.
    ///
    /// Phase profile at radius r: φ(r) = −k₀ · (√(r²+f²) − f)
    /// Then fill factor = fill_factor_at(r, φ(r)).
    pub fn generate_layout(&self, focal_length: f64, wavelength: f64) -> Vec<(f64, f64)> {
        if self.n_rings == 0 {
            return Vec::new();
        }
        let k0 = 2.0 * PI / wavelength;
        let r_max = self.max_diameter / 2.0;
        let dr = r_max / self.n_rings as f64;
        (0..self.n_rings)
            .map(|i| {
                // Use ring centre radius
                let r = (i as f64 + 0.5) * dr;
                let phi = -k0 * ((r * r + focal_length * focal_length).sqrt() - focal_length);
                let ff = self.fill_factor_at(r, phi);
                (r, ff)
            })
            .collect()
    }

    /// Phase at a given radius for a focusing lens (rad).
    pub fn focusing_phase_at(&self, r: f64, focal_length: f64, wavelength: f64) -> f64 {
        let k0 = 2.0 * PI / wavelength;
        -k0 * ((r * r + focal_length * focal_length).sqrt() - focal_length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- MetalensLayout (original) ----

    #[test]
    fn metalens_layout_posts_inside_aperture() {
        let layout = MetalensLayout::focusing(10e-6, 20e-6, 1e-6, 1550e-9);
        let r_max = layout.aperture / 2.0;
        for p in &layout.posts {
            let r = (p.x * p.x + p.y * p.y).sqrt();
            assert!(
                r <= r_max + 1e-9,
                "Post at r={r:.2e} outside aperture r_max={r_max:.2e}"
            );
        }
    }

    #[test]
    fn metalens_layout_has_posts() {
        let layout = MetalensLayout::focusing(10e-6, 20e-6, 1e-6, 1550e-9);
        assert!(layout.n_posts() > 0, "No posts generated");
    }

    #[test]
    fn metalens_na_in_range() {
        let layout = MetalensLayout::focusing(10e-6, 20e-6, 1e-6, 1550e-9);
        let na = layout.numerical_aperture();
        assert!(na > 0.0 && na < 1.0, "NA={na:.3}");
    }

    #[test]
    fn metalens_airy_radius_positive() {
        let layout = MetalensLayout::focusing(10e-6, 20e-6, 1e-6, 1550e-9);
        assert!(layout.airy_radius() > 0.0);
    }

    #[test]
    fn metalens_vortex_has_posts() {
        let layout = MetalensLayout::vortex(10e-6, 1e-6, 1550e-9, 1);
        assert!(layout.n_posts() > 0);
    }

    #[test]
    fn metalens_diameter_range_valid() {
        let layout = MetalensLayout::focusing(10e-6, 20e-6, 1e-6, 1550e-9);
        let (d_min, d_max) = layout.diameter_range();
        assert!(d_min >= 0.0);
        assert!(d_max <= layout.pitch);
        assert!(d_min < d_max);
    }

    // ---- AchromaticMetalens ----

    #[test]
    fn achromatic_metalens_centre_wavelength() {
        let am = AchromaticMetalens::new(100e-6, 50e-6, 1310e-9, 1550e-9);
        let lc = am.centre_wavelength();
        assert!(
            (lc - 1430e-9).abs() < 1e-12,
            "λ_c={:.0}nm expected 1430nm",
            lc * 1e9
        );
    }

    #[test]
    fn achromatic_metalens_phase_profile_negative_at_edge() {
        // Focusing phase should be zero at r=0 and negative (converging) at r>0
        let am = AchromaticMetalens::new(100e-6, 50e-6, 1310e-9, 1550e-9);
        let phi_centre = am.phase_profile(0.0);
        let phi_edge = am.phase_profile(25e-6);
        assert!(phi_centre.abs() < 1e-10, "Phase at r=0 should be 0");
        assert!(phi_edge < 0.0, "Focusing phase at edge should be negative");
    }

    #[test]
    fn achromatic_metalens_disp_phase_zero_at_centre_wavelength() {
        let am = AchromaticMetalens::new(100e-6, 50e-6, 1310e-9, 1550e-9);
        let lc = am.centre_wavelength();
        let dp = am.dispersion_compensation_phase(10e-6, lc);
        assert!(
            dp.abs() < 1e-6,
            "Disp compensation at λ_c should be ≈0, got {dp:.2e}"
        );
    }

    #[test]
    fn achromatic_metalens_total_phase_equals_sum() {
        let am = AchromaticMetalens::new(100e-6, 50e-6, 1310e-9, 1550e-9);
        let r = 15e-6;
        let l = 1400e-9;
        let total = am.total_phase(r, l);
        let expected = am.phase_profile(r) + am.dispersion_compensation_phase(r, l);
        assert!((total - expected).abs() < 1e-12);
    }

    #[test]
    fn achromatic_metalens_efficiency_in_range() {
        let am = AchromaticMetalens::new(100e-6, 50e-6, 1310e-9, 1550e-9);
        let eta = am.efficiency_estimate();
        assert!(
            (0.5..=0.9).contains(&eta),
            "Efficiency={eta:.3} out of [0.5, 0.9]"
        );
    }

    #[test]
    fn achromatic_metalens_na_positive() {
        let am = AchromaticMetalens::new(100e-6, 50e-6, 1310e-9, 1550e-9);
        let na = am.numerical_aperture();
        assert!(na > 0.0 && na < 1.0, "NA={na:.3}");
    }

    // ---- ZonePlate ----

    #[test]
    fn zone_plate_zone_radii_increasing() {
        let zp = ZonePlate::new(100e-6, 1550e-9, 50e-6);
        let radii = zp.zone_radii();
        assert!(!radii.is_empty(), "Should have at least one zone");
        for i in 1..radii.len() {
            assert!(
                radii[i] > radii[i - 1],
                "Radii should be strictly increasing"
            );
        }
    }

    #[test]
    fn zone_plate_zone_radii_all_within_aperture() {
        let zp = ZonePlate::new(100e-6, 1550e-9, 50e-6);
        let r_max = zp.diameter / 2.0;
        for &r in &zp.zone_radii() {
            assert!(
                r <= r_max + 1e-12,
                "Zone radius {r:.2e} outside aperture {r_max:.2e}"
            );
        }
    }

    #[test]
    fn zone_plate_n_zones_matches_radii() {
        let zp = ZonePlate::new(100e-6, 1550e-9, 50e-6);
        assert_eq!(zp.n_zones(), zp.zone_radii().len());
    }

    #[test]
    fn zone_plate_theoretical_efficiency_approx_10percent() {
        let zp = ZonePlate::new(100e-6, 1550e-9, 50e-6);
        let eta = zp.theoretical_efficiency();
        // 1/π² ≈ 0.10132
        assert!((eta - 1.0 / (PI * PI)).abs() < 1e-10);
    }

    #[test]
    fn zone_plate_first_zone_radius_formula() {
        // r₁ ≈ √(λ·f) for paraxial limit
        let l = 1550e-9;
        let f = 100e-6;
        let zp = ZonePlate::new(f, l, 50e-6);
        let radii = zp.zone_radii();
        if !radii.is_empty() {
            let r1_paraxial = (l * f).sqrt();
            let r1_exact = radii[0];
            // Should be close (exact differs by (λ/2)² term)
            assert!(
                (r1_exact - r1_paraxial).abs() / r1_paraxial < 0.01,
                "r₁={r1_exact:.2e} expected≈{r1_paraxial:.2e}"
            );
        }
    }

    #[test]
    fn zone_plate_na_positive() {
        let zp = ZonePlate::new(100e-6, 1550e-9, 50e-6);
        let na = zp.numerical_aperture();
        assert!(na > 0.0 && na < 1.0, "NA={na:.3}");
    }

    #[test]
    fn zone_plate_airy_radius_positive() {
        let zp = ZonePlate::new(100e-6, 1550e-9, 50e-6);
        assert!(zp.airy_radius() > 0.0);
    }

    // ---- FillFactorMap ----

    #[test]
    fn fill_factor_map_layout_length_correct() {
        let ffm = FillFactorMap::new(20, 50e-6);
        let layout = ffm.generate_layout(100e-6, 1550e-9);
        assert_eq!(layout.len(), 20);
    }

    #[test]
    fn fill_factor_map_fill_factors_in_range() {
        let ffm = FillFactorMap::new(20, 50e-6);
        let layout = ffm.generate_layout(100e-6, 1550e-9);
        for &(_, ff) in &layout {
            assert!(
                (0.1..=0.9).contains(&ff),
                "Fill factor {ff:.3} out of [0.1, 0.9]"
            );
        }
    }

    #[test]
    fn fill_factor_map_radii_increasing() {
        let ffm = FillFactorMap::new(10, 50e-6);
        let layout = ffm.generate_layout(100e-6, 1550e-9);
        for i in 1..layout.len() {
            assert!(
                layout[i].0 > layout[i - 1].0,
                "Radii should be strictly increasing"
            );
        }
    }

    #[test]
    fn fill_factor_map_fill_factor_at_phase_zero() {
        let ffm = FillFactorMap::new(10, 50e-6);
        let ff = ffm.fill_factor_at(5e-6, 0.0);
        // φ=0 → φ_norm=0 → ff = 0.1
        assert!(
            (ff - 0.1).abs() < 1e-10,
            "ff at φ=0 should be 0.1, got {ff}"
        );
    }

    #[test]
    fn fill_factor_map_fill_factor_at_pi() {
        let ffm = FillFactorMap::new(10, 50e-6);
        let ff = ffm.fill_factor_at(5e-6, PI);
        // φ=π → φ_norm=0.5 → ff = 0.1 + 0.5*0.8 = 0.5
        assert!(
            (ff - 0.5).abs() < 1e-10,
            "ff at φ=π should be 0.5, got {ff}"
        );
    }

    #[test]
    fn fill_factor_map_empty_for_zero_rings() {
        let ffm = FillFactorMap::new(0, 50e-6);
        let layout = ffm.generate_layout(100e-6, 1550e-9);
        assert!(layout.is_empty());
    }

    #[test]
    fn fill_factor_map_ring_width_correct() {
        let ffm = FillFactorMap::new(10, 50e-6);
        let rw = ffm.ring_width();
        assert!(
            (rw - 2.5e-6).abs() < 1e-15,
            "Ring width={rw:.2e}m expected 2.5µm"
        );
    }
}
