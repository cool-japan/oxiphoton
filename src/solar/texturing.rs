//! Light trapping and texturing models for solar cells.
//!
//! Surface texturing increases the optical path length in the absorber layer,
//! enhancing absorption of weakly-absorbed near-bandgap photons.
//!
//! Key structures:
//!   1. Random pyramids (alkaline-etched c-Si): Lambertian scattering
//!   2. Inverted pyramids (PERC cells): lower surface recombination
//!   3. V-grooves: geometric light trapping
//!   4. Nano-cone arrays: broadband AR + trapping
//!   5. Mie resonant particles: Mie scattering for coupling
//!
//! Lambertian limit (Yablonovitch 1982):
//!   Maximum path length enhancement F_max = 4n²  (for n = semiconductor index)
//!   → For c-Si (n=3.5): F_max ≈ 49
//!
//! Enhanced absorptance with path length factor F:
//!   A_text(λ) = 1 - exp(-F·α·L)
//!
//! Photocurrent enhancement:
//!   J_text = ∫ Φ(λ) · A_text(λ) dλ / ∫ Φ(λ) · A_single(λ) dλ

/// Surface texture type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureType {
    /// Planar surface (no texturing) — baseline
    Planar,
    /// Random pyramid texture (alkaline-etched c-Si)
    RandomPyramid,
    /// Inverted pyramid texture (lithographically defined)
    InvertedPyramid,
    /// V-groove texture
    VGroove,
    /// Ideal Lambertian scatterer (maximum randomization)
    Lambertian,
    /// Nano-cone array (subwavelength)
    NanoCone,
}

/// Light trapping model for a textured solar cell.
#[derive(Debug, Clone, Copy)]
pub struct LightTrappingModel {
    /// Texture type
    pub texture: TextureType,
    /// Semiconductor refractive index (at λ ≈ bandgap)
    pub n_semiconductor: f64,
    /// Absorber layer thickness L (m)
    pub thickness: f64,
    /// Front surface reflectance R_front (0–1)
    pub r_front: f64,
    /// Back reflector reflectance R_back (0–1)
    pub r_back: f64,
}

impl LightTrappingModel {
    /// Create a light trapping model.
    pub fn new(texture: TextureType, n_semi: f64, thickness: f64) -> Self {
        Self {
            texture,
            n_semiconductor: n_semi,
            thickness,
            r_front: 0.03, // with ARC
            r_back: 0.9,   // Al back reflector
        }
    }

    /// Standard c-Si cell with random pyramid texture (n=3.5, 180 µm thick).
    pub fn csi_random_pyramid() -> Self {
        Self::new(TextureType::RandomPyramid, 3.5, 180e-6)
    }

    /// Planar c-Si reference (no texturing).
    pub fn csi_planar() -> Self {
        Self::new(TextureType::Planar, 3.5, 180e-6)
    }

    /// Lambertian limit (ideal).
    pub fn lambertian(n_semi: f64, thickness: f64) -> Self {
        Self::new(TextureType::Lambertian, n_semi, thickness)
    }

    /// Yablonovitch limit: maximum path length enhancement F_max = 4n².
    pub fn yablonovitch_limit(&self) -> f64 {
        4.0 * self.n_semiconductor * self.n_semiconductor
    }

    /// Effective path length enhancement factor F for this texture type.
    ///
    /// Ranges from 1 (planar with back reflector) to 4n² (Lambertian).
    pub fn path_length_enhancement(&self) -> f64 {
        let f_max = self.yablonovitch_limit();
        match self.texture {
            TextureType::Planar => 1.0 + self.r_back, // single pass + partial back reflection
            TextureType::RandomPyramid => 0.7 * f_max, // ~70% of Lambertian
            TextureType::InvertedPyramid => 0.8 * f_max,
            TextureType::VGroove => 0.5 * f_max,
            TextureType::Lambertian => f_max,
            TextureType::NanoCone => 0.6 * f_max,
        }
    }

    /// Enhanced absorptance with light trapping.
    ///
    ///   A_text(λ) = 1 - exp(-F·α·L) × (approximate for Lambertian)
    ///
    /// More precise: A = (α·L·F) / (1 + α·L·F + R_back·exp(-α·L·F)...)
    /// Here we use the simplified exponential model.
    pub fn absorptance(&self, alpha_per_m: f64) -> f64 {
        let f = self.path_length_enhancement();
        let al = alpha_per_m * self.thickness;
        1.0 - (1.0 - self.r_front) * (-f * al).exp()
    }

    /// Photocurrent density ratio J_text / J_planar (enhancement factor).
    ///
    /// Estimated by comparing absorptance at a single representative wavelength.
    pub fn current_enhancement(&self, alpha_per_m: f64) -> f64 {
        let planar =
            LightTrappingModel::new(TextureType::Planar, self.n_semiconductor, self.thickness);
        let a_text = self.absorptance(alpha_per_m);
        let a_planar = planar.absorptance(alpha_per_m);
        if a_planar < 1e-10 {
            return 1.0;
        }
        a_text / a_planar
    }

    /// Front surface effective reflectance (accounting for texture geometry).
    ///
    /// Random pyramid: R_eff ≈ 0.6 × R_bare (multiple bounces reduce effective R).
    pub fn effective_front_reflectance(&self, r_bare: f64) -> f64 {
        match self.texture {
            TextureType::Planar => r_bare,
            TextureType::RandomPyramid | TextureType::InvertedPyramid => r_bare * 0.6,
            TextureType::VGroove => r_bare * 0.5,
            TextureType::Lambertian => r_bare * 0.1,
            TextureType::NanoCone => r_bare * 0.3,
        }
    }

    /// Effective path length in µm for near-bandgap light.
    pub fn effective_path_length_um(&self) -> f64 {
        let f = self.path_length_enhancement();
        f * self.thickness * 1e6
    }
}

/// Geometric model for inverted pyramid array.
#[derive(Debug, Clone, Copy)]
pub struct InvertedPyramidArray {
    /// Pyramid pitch P (m) — center-to-center spacing
    pub pitch: f64,
    /// Pyramid half-width w (m)
    pub half_width: f64,
    /// Pyramid depth d (m)
    pub depth: f64,
    /// Tilted facet normal angle (rad from surface normal)
    pub facet_angle: f64,
}

impl InvertedPyramidArray {
    /// Standard inverted pyramid for c-Si (P=5µm, coverage 90%).
    pub fn standard_si() -> Self {
        let p = 5e-6;
        let w = 0.45 * p; // 90% coverage
        Self {
            pitch: p,
            half_width: w,
            depth: w, // 54.7° etch angle for (100) Si
            facet_angle: f64::to_radians(54.7),
        }
    }

    /// Surface area enhancement factor (textured / planar).
    pub fn surface_area_ratio(&self) -> f64 {
        let p = self.pitch;
        // Area of one pyramid = 4 × triangular facets
        let facet_area = 4.0
            * self.half_width
            * (self.depth * self.depth + self.half_width * self.half_width).sqrt();
        let coverage = (2.0 * self.half_width / p).powi(2);
        1.0 + coverage * (facet_area / (p * p) - 1.0)
    }

    /// Average number of reflections before coupling into substrate (≈2 for optimal).
    pub fn average_reflections(&self) -> f64 {
        // For inverted pyramid with tilted facets, light typically reflects 2× before absorption
        let angle_rad = self.facet_angle;
        // After first reflection off facet, redirected light hits opposite facet
        let hit_opposite = (2.0 * angle_rad).cos() < 0.0; // angle > 45°
        if hit_opposite {
            2.0
        } else {
            1.5
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yablonovitch_limit_si() {
        let m = LightTrappingModel::csi_random_pyramid();
        let f_max = m.yablonovitch_limit();
        assert!((f_max - 49.0).abs() < 1.0, "F_max={f_max:.1}");
    }

    #[test]
    fn lambertian_has_max_enhancement() {
        let m_lamb = LightTrappingModel::lambertian(3.5, 180e-6);
        let m_rand = LightTrappingModel::csi_random_pyramid();
        assert!(m_lamb.path_length_enhancement() >= m_rand.path_length_enhancement());
    }

    #[test]
    fn planar_has_minimum_enhancement() {
        let m_plane = LightTrappingModel::csi_planar();
        let m_rand = LightTrappingModel::csi_random_pyramid();
        assert!(m_rand.path_length_enhancement() > m_plane.path_length_enhancement());
    }

    #[test]
    fn absorptance_increased_with_texturing() {
        let m_rand = LightTrappingModel::csi_random_pyramid();
        let m_plane = LightTrappingModel::csi_planar();
        let alpha = 100.0; // m⁻¹ (weak absorption)
        let a_rand = m_rand.absorptance(alpha);
        let a_plane = m_plane.absorptance(alpha);
        assert!(
            a_rand > a_plane,
            "A_texture={a_rand:.4} > A_planar={a_plane:.4}"
        );
    }

    #[test]
    fn current_enhancement_ge_1() {
        let m = LightTrappingModel::csi_random_pyramid();
        let enh = m.current_enhancement(1000.0);
        assert!(enh >= 1.0, "enhancement={enh:.3}");
    }

    #[test]
    fn effective_path_length_positive() {
        let m = LightTrappingModel::csi_random_pyramid();
        assert!(m.effective_path_length_um() > 0.0);
    }

    #[test]
    fn inverted_pyramid_area_ratio_gt_1() {
        let pyr = InvertedPyramidArray::standard_si();
        assert!(pyr.surface_area_ratio() > 1.0);
    }

    #[test]
    fn inverted_pyramid_reflections_positive() {
        let pyr = InvertedPyramidArray::standard_si();
        assert!(pyr.average_reflections() > 0.0);
    }
}
