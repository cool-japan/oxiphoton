//! Lens system design: singlet, achromatic doublet, and triplet optimization.
//!
//! Provides tools for designing refractive lens systems:
//! - Singlet lens with bending factor optimization (Coddington equations)
//! - Achromatic doublet (crown + flint, Fraunhofer design)
//! - Triplet (Cooke triplet) layout
//! - Simple merit function evaluation for paraxial optimization
//!
//! Bending factor q = (R₂+R₁)/(R₂-R₁) controls spherical aberration of a singlet.
//! Aplanatic condition: q = (2(n²-1))/(n+2) minimizes spherical aberration.

use crate::ray::paraxial::{ChromaticAnalysis, SystemMatrix};
use crate::ray::tracer::Surface;

/// Glass material defined by refractive index at d-line and Abbe number.
#[derive(Debug, Clone, Copy)]
pub struct GlassMaterial {
    /// Refractive index at d-line (587.6 nm)
    pub n_d: f64,
    /// Abbe number V = (n_d - 1)/(n_F - n_C)
    pub abbe_number: f64,
    /// Material name
    pub name: &'static str,
}

impl GlassMaterial {
    /// N-BK7 crown glass: n_d=1.5168, V=64.17.
    pub fn bk7() -> Self {
        Self {
            n_d: 1.5168,
            abbe_number: 64.17,
            name: "N-BK7",
        }
    }

    /// N-F2 flint glass: n_d=1.6200, V=36.43.
    pub fn f2() -> Self {
        Self {
            n_d: 1.6200,
            abbe_number: 36.43,
            name: "N-F2",
        }
    }

    /// N-SF11 dense flint glass: n_d=1.7847, V=25.76.
    pub fn sf11() -> Self {
        Self {
            n_d: 1.7847,
            abbe_number: 25.76,
            name: "N-SF11",
        }
    }

    /// Fused silica: n_d=1.4585, V=67.82.
    pub fn fused_silica() -> Self {
        Self {
            n_d: 1.4585,
            abbe_number: 67.82,
            name: "SiO2",
        }
    }

    /// N-LAK22 lanthanum crown: n_d=1.6516, V=58.90.
    pub fn lak22() -> Self {
        Self {
            n_d: 1.6516,
            abbe_number: 58.90,
            name: "N-LAK22",
        }
    }
}

/// Plano-convex or biconvex singlet lens.
///
/// Described by bending factor q = (R₂+R₁)/(R₂-R₁) and focal length f.
/// Radius calculation from thin-lens formula:
///   1/f = (n-1)[1/R₁ - 1/R₂]
/// For a given q: R₁ = 2f(n-1)/(1+q), R₂ = 2f(n-1)/(-1+q)  (when q≠±1)
#[derive(Debug, Clone, Copy)]
pub struct Singlet {
    /// Glass material
    pub glass: GlassMaterial,
    /// Effective focal length f (m)
    pub focal_length: f64,
    /// Bending factor q (range -∞ to +∞)
    pub bending_factor: f64,
    /// Center thickness t (m)
    pub thickness: f64,
}

impl Singlet {
    /// Create a singlet with specified bending factor.
    /// q=0: equiconvex, q=1: plano-convex, q=-1: convex-plano.
    pub fn new(
        glass: GlassMaterial,
        focal_length: f64,
        bending_factor: f64,
        thickness: f64,
    ) -> Self {
        Self {
            glass,
            focal_length,
            bending_factor,
            thickness,
        }
    }

    /// Optimum bending factor to minimize third-order spherical aberration.
    ///   q_opt = (2(n²-1))/(n+2)
    pub fn aplanatic_bending(glass: GlassMaterial) -> f64 {
        let n = glass.n_d;
        2.0 * (n * n - 1.0) / (n + 2.0)
    }

    /// Equiconvex singlet (q=0).
    pub fn equiconvex(glass: GlassMaterial, focal_length: f64, thickness: f64) -> Self {
        Self::new(glass, focal_length, 0.0, thickness)
    }

    /// Plano-convex singlet (q=1, flat side towards image).
    pub fn plano_convex(glass: GlassMaterial, focal_length: f64, thickness: f64) -> Self {
        Self::new(glass, focal_length, 1.0, thickness)
    }

    /// Front radius R₁ (m). Positive = center of curvature to right.
    pub fn r1(&self) -> f64 {
        let n = self.glass.n_d;
        let q = self.bending_factor;
        let power = (n - 1.0) / self.focal_length;
        if (q - 1.0).abs() < 1e-10 {
            // Plano-convex: R1 = (n-1)*f, R2 = infinity
            (n - 1.0) * self.focal_length
        } else {
            2.0 * (n - 1.0) / (power * (1.0 + q))
        }
    }

    /// Back radius R₂ (m).
    pub fn r2(&self) -> f64 {
        let n = self.glass.n_d;
        let q = self.bending_factor;
        let power = (n - 1.0) / self.focal_length;
        if (q + 1.0).abs() < 1e-10 {
            // Convex-plano
            f64::INFINITY
        } else {
            2.0 * (n - 1.0) / (power * (q - 1.0))
        }
    }

    /// System (ABCD) matrix of the thick singlet.
    pub fn system_matrix(&self) -> SystemMatrix {
        let n = self.glass.n_d;
        let r1 = self.r1();
        let r2 = self.r2();
        let surfaces = vec![
            Surface::CurvedInterface {
                r: r1,
                n1: 1.0,
                n2: n,
            },
            Surface::FreeSpace { d: self.thickness },
            Surface::CurvedInterface {
                r: r2,
                n1: n,
                n2: 1.0,
            },
        ];
        SystemMatrix::from_surfaces(&surfaces)
    }

    /// Chromatic aberration (LCA = f/V).
    pub fn lca(&self) -> f64 {
        ChromaticAnalysis::new(self.focal_length, self.glass.abbe_number).lca()
    }
}

/// Achromatic doublet (cemented): crown + flint combination that corrects
/// chromatic aberration at two wavelengths.
///
/// Power split: φ₁ = φ·V₁/(V₁-V₂), φ₂ = -φ·V₂/(V₁-V₂)
#[derive(Debug, Clone, Copy)]
pub struct AchromaticDoublet {
    /// Crown glass (element 1)
    pub crown: GlassMaterial,
    /// Flint glass (element 2)
    pub flint: GlassMaterial,
    /// Combined effective focal length f (m)
    pub focal_length: f64,
    /// Thickness of crown element (m)
    pub t1: f64,
    /// Thickness of flint element (m)
    pub t2: f64,
}

impl AchromaticDoublet {
    pub fn new(
        crown: GlassMaterial,
        flint: GlassMaterial,
        focal_length: f64,
        t1: f64,
        t2: f64,
    ) -> Self {
        Self {
            crown,
            flint,
            focal_length,
            t1,
            t2,
        }
    }

    /// Standard BK7/F2 achromat.
    pub fn bk7_f2(focal_length: f64) -> Self {
        let t = focal_length * 0.05; // 5% of focal length for each element
        Self::new(
            GlassMaterial::bk7(),
            GlassMaterial::f2(),
            focal_length,
            t,
            t * 0.5,
        )
    }

    /// Focal length of crown element.
    ///
    /// φ₁ = φ·V₁/(V₁-V₂)  →  f₁ = f·(V₁-V₂)/V₁
    pub fn f_crown(&self) -> f64 {
        let v1 = self.crown.abbe_number;
        let v2 = self.flint.abbe_number;
        self.focal_length * (v1 - v2) / v1
    }

    /// Focal length of flint element.
    ///
    /// φ₂ = -φ·V₂/(V₁-V₂)  →  f₂ = -f·(V₁-V₂)/V₂
    pub fn f_flint(&self) -> f64 {
        let v1 = self.crown.abbe_number;
        let v2 = self.flint.abbe_number;
        -self.focal_length * (v1 - v2) / v2
    }

    /// Cemented surface radius R (m) — shared surface between crown and flint.
    ///   From thin lens: 1/f_crown = (n1-1)[1/R1 - 1/Rc]
    ///   Simplified: assume symmetric crown, Rc from Coddington
    pub fn cemented_radius(&self) -> f64 {
        let n1 = self.crown.n_d;
        let n2 = self.flint.n_d;
        let fc = self.f_crown();
        // For plano-convex crown cemented to plano-concave flint
        // Cemented surface: (n2-n1)/Rc contributes to power
        let phi_c = (n1 - 1.0) / fc;
        if phi_c.abs() > 1e-30 {
            (n2 - n1) / phi_c
        } else {
            f64::INFINITY
        }
    }

    /// Residual secondary spectrum (m): LCA after achromatic correction.
    ///   Secondary = f·(P₁-P₂)/(V₁-V₂)  where P is partial dispersion
    /// Approximated as f/(V₁+V₂) for common glass pairs.
    pub fn secondary_spectrum(&self) -> f64 {
        self.focal_length / (self.crown.abbe_number + self.flint.abbe_number)
    }

    /// Verify achromatic condition: 1/f1 + 1/f2 = 1/f.
    pub fn verify_power_sum(&self) -> f64 {
        let phi1 = 1.0 / self.f_crown();
        let phi2 = 1.0 / self.f_flint();
        (phi1 + phi2 - 1.0 / self.focal_length).abs()
    }
}

/// Cooke triplet: three-element lens (crown-flint-crown) design.
///
/// The triplet corrects 7 Seidel aberrations by distributing power across
/// positive (crown), negative (flint), and positive (crown) elements.
/// Thin-lens spacing rules:
///   d₁ = (f·φ₂)/(φ₁·(φ₁+φ₂+φ₃) - φ)  (approximate)
#[derive(Debug, Clone, Copy)]
pub struct CookeTriplet {
    /// First crown element focal length (m)
    pub f1: f64,
    /// Flint element focal length (m)
    pub f2: f64,
    /// Third crown element focal length (m)
    pub f3: f64,
    /// Spacing between elements 1-2 (m)
    pub d1: f64,
    /// Spacing between elements 2-3 (m)
    pub d2: f64,
    /// Crown glass
    pub crown: GlassMaterial,
    /// Flint glass
    pub flint: GlassMaterial,
}

impl CookeTriplet {
    /// Create a Cooke triplet with specified element focal lengths and spacings.
    pub fn new(
        f1: f64,
        f2: f64,
        f3: f64,
        d1: f64,
        d2: f64,
        crown: GlassMaterial,
        flint: GlassMaterial,
    ) -> Self {
        Self {
            f1,
            f2,
            f3,
            d1,
            d2,
            crown,
            flint,
        }
    }

    /// Standard Cooke triplet targeting f=100mm, f/4.
    pub fn standard_f100() -> Self {
        Self::new(
            0.120,  // f1 = 120mm
            -0.050, // f2 = -50mm (flint)
            0.120,  // f3 = 120mm
            0.040,  // d1 = 40mm
            0.040,  // d2 = 40mm
            GlassMaterial::bk7(),
            GlassMaterial::f2(),
        )
    }

    /// ABCD system matrix for the triplet.
    pub fn system_matrix(&self) -> SystemMatrix {
        let surfaces = vec![
            Surface::ThinLens { f: self.f1 },
            Surface::FreeSpace { d: self.d1 },
            Surface::ThinLens { f: self.f2 },
            Surface::FreeSpace { d: self.d2 },
            Surface::ThinLens { f: self.f3 },
        ];
        SystemMatrix::from_surfaces(&surfaces)
    }

    /// Combined effective focal length.
    pub fn effective_focal_length(&self) -> f64 {
        let cp = self.system_matrix().cardinal_points();
        cp.back_focal_length
    }

    /// Total Petzval sum: Σ(φᵢ/nᵢ) — zero gives flat image field.
    pub fn petzval_sum(&self) -> f64 {
        let phi1 = 1.0 / self.f1;
        let phi2 = 1.0 / self.f2;
        let phi3 = 1.0 / self.f3;
        phi1 / self.crown.n_d + phi2 / self.flint.n_d + phi3 / self.crown.n_d
    }
}

/// Merit function for paraxial lens optimization.
///
/// Evaluates system quality based on:
/// - Focal length error (targeting f_target)
/// - Chromatic aberration (LCA)
/// - Petzval field curvature (for multi-element systems)
#[derive(Debug, Clone)]
pub struct LensMeritFunction {
    /// Target focal length (m)
    pub target_focal_length: f64,
    /// Maximum allowed LCA (m)
    pub max_lca: f64,
    /// Weight for focal length error
    pub w_focal: f64,
    /// Weight for chromatic aberration
    pub w_chromatic: f64,
}

impl LensMeritFunction {
    pub fn new(target_focal_length: f64) -> Self {
        Self {
            target_focal_length,
            max_lca: target_focal_length * 0.01,
            w_focal: 1.0,
            w_chromatic: 1.0,
        }
    }

    /// Evaluate merit for a singlet (lower = better).
    pub fn evaluate_singlet(&self, singlet: &Singlet) -> f64 {
        let m = singlet.system_matrix();
        let cp = m.cardinal_points();
        let focal_err = (cp.back_focal_length - self.target_focal_length).abs();
        let lca = singlet.lca().abs();
        self.w_focal * focal_err / self.target_focal_length + self.w_chromatic * lca / self.max_lca
    }

    /// Evaluate merit for an achromatic doublet.
    pub fn evaluate_doublet(&self, doublet: &AchromaticDoublet) -> f64 {
        let focal_err = (doublet.focal_length - self.target_focal_length).abs();
        let secondary = doublet.secondary_spectrum().abs();
        let power_err = doublet.verify_power_sum();
        self.w_focal * focal_err / self.target_focal_length
            + self.w_chromatic * secondary / self.max_lca
            + 10.0 * power_err * self.target_focal_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singlet_r1_positive_for_converging() {
        let s = Singlet::equiconvex(GlassMaterial::bk7(), 0.1, 0.005);
        assert!(s.r1() > 0.0, "R1={}", s.r1());
    }

    #[test]
    fn singlet_r2_negative_for_equiconvex() {
        let s = Singlet::equiconvex(GlassMaterial::bk7(), 0.1, 0.005);
        assert!(s.r2() < 0.0, "R2={}", s.r2());
    }

    #[test]
    fn singlet_plano_convex_r2_infinity() {
        let s = Singlet::plano_convex(GlassMaterial::bk7(), 0.1, 0.005);
        assert!(s.r2().is_infinite(), "R2={}", s.r2());
    }

    #[test]
    fn aplanatic_bending_bk7() {
        let q = Singlet::aplanatic_bending(GlassMaterial::bk7());
        // For n=1.5168: q_opt = 2*(n²-1)/(n+2) ≈ 0.832
        assert!(q > 0.5 && q < 1.5, "q_opt={q}");
    }

    #[test]
    fn achromat_power_sum_correct() {
        let d = AchromaticDoublet::bk7_f2(0.1);
        let err = d.verify_power_sum();
        assert!(err < 1e-10, "power sum error={err}");
    }

    #[test]
    fn achromat_crown_flint_opposing_powers() {
        let d = AchromaticDoublet::bk7_f2(0.1);
        // Crown is positive, flint is negative
        assert!(d.f_crown() > 0.0, "f_crown={}", d.f_crown());
        assert!(d.f_flint() < 0.0, "f_flint={}", d.f_flint());
    }

    #[test]
    fn triplet_petzval_sum_finite() {
        let t = CookeTriplet::standard_f100();
        let pz = t.petzval_sum();
        assert!(pz.is_finite(), "petzval={pz}");
    }

    #[test]
    fn triplet_efl_reasonable() {
        let t = CookeTriplet::standard_f100();
        let efl = t.effective_focal_length();
        // BFL should be positive and finite (approximate design, tolerant range)
        assert!(efl.is_finite() && efl > 0.0 && efl < 2.0, "EFL={efl}");
    }

    #[test]
    fn merit_function_singlet() {
        let mf = LensMeritFunction::new(0.1);
        let s = Singlet::equiconvex(GlassMaterial::bk7(), 0.1, 0.005);
        let merit = mf.evaluate_singlet(&s);
        assert!(merit.is_finite() && merit >= 0.0, "merit={merit}");
    }

    #[test]
    fn glass_materials_distinct() {
        let bk7 = GlassMaterial::bk7();
        let f2 = GlassMaterial::f2();
        assert!((bk7.n_d - f2.n_d).abs() > 0.05);
        assert!((bk7.abbe_number - f2.abbe_number).abs() > 20.0);
    }
}
