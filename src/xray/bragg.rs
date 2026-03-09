//! Bragg diffraction from crystal lattice planes.
//!
//! Provides:
//! - Crystal material database with d-spacings and structure factors.
//! - Bragg condition, Darwin rocking-curve width, integrated reflectivity.
//! - Energy resolving power and energy–angle dispersion.
//! - Johann bent-crystal spectrometer geometry.
//!
//! # Conventions
//! All angles are in radians.  Energies returned in eV unless noted.
//! Wavelengths in metres.
//!
//! # References
//! - Als-Nielsen & McMorrow, *Elements of Modern X-ray Physics*, 2nd ed. (2011).
//! - Zachariasen, *Theory of X-ray Diffraction in Crystals*, Dover (1945).

use crate::error::{OxiPhotonError, Result};
use std::f64::consts::PI;

// ─── Physical constants ────────────────────────────────────────────────────

/// Classical electron radius r_e (m).
const R_ELECTRON: f64 = 2.817_940_3e-15;
/// Speed of light (m/s).
const C0: f64 = 2.997_924_58e8;
/// Planck constant (J·s).
const H_PLANCK: f64 = 6.626_070_15e-34;
/// Electron charge (C).
const E_CHARGE: f64 = 1.602_176_634e-19;

/// Convert wavelength in metres to photon energy in eV.
#[inline]
fn lambda_to_ev(lambda_m: f64) -> f64 {
    H_PLANCK * C0 / (lambda_m * E_CHARGE)
}

// ═══════════════════════════════════════════════════════════════════════════
// CrystalMaterial
// ═══════════════════════════════════════════════════════════════════════════

/// Supported crystal materials for Bragg optics.
#[derive(Debug, Clone)]
pub enum CrystalMaterial {
    /// Silicon — cubic, a = 5.4310 Å, diamond structure.
    Silicon,
    /// Germanium — cubic, a = 5.6578 Å, diamond structure.
    Germanium,
    /// Diamond — cubic, a = 3.5668 Å, diamond structure.
    Diamond,
    /// α-Quartz (SiO₂) — hexagonal.
    Quartz,
    /// Lithium niobate (LiNbO₃) — trigonal.
    LithiumNiobate,
    /// User-supplied d-spacing and structure-factor magnitude.
    Custom {
        /// Interplanar d-spacing (m).
        d_spacing: f64,
        /// |F_hkl| in electrons (structure factor magnitude).
        structure_factor: f64,
    },
}

impl CrystalMaterial {
    /// Lattice parameter `a` for cubic crystals (m).
    fn lattice_a(&self) -> Option<f64> {
        match self {
            Self::Silicon => Some(5.431_020_511e-10),
            Self::Germanium => Some(5.658_e-10),
            Self::Diamond => Some(3.5668e-10),
            _ => None,
        }
    }

    /// Interplanar d-spacing for the (h k l) reflection (m).
    ///
    /// Uses the cubic formula d = a / sqrt(h²+k²+l²) for Si, Ge, Diamond.
    /// For Quartz and LiNbO₃ only a selection of common reflections is
    /// tabulated; otherwise falls back to the `Custom` variant.
    pub fn d_spacing(&self, h: i32, k: i32, l: i32) -> f64 {
        match self {
            Self::Silicon | Self::Germanium | Self::Diamond => {
                let a = self.lattice_a().unwrap_or(5.431e-10);
                let denom = ((h * h + k * k + l * l) as f64).sqrt();
                if denom < 1e-10 {
                    return f64::INFINITY;
                }
                a / denom
            }
            Self::Quartz => {
                // Hexagonal: d = a / sqrt(4/3*(h²+hk+k²) + (a/c)²*l²)
                // α-quartz: a = 4.913 Å, c = 5.405 Å
                let a = 4.913e-10_f64;
                let c = 5.405e-10_f64;
                let num =
                    (4.0 / 3.0) * (h * h + h * k + k * k) as f64 + (a / c).powi(2) * (l * l) as f64;
                if num <= 0.0 {
                    return f64::INFINITY;
                }
                a / num.sqrt()
            }
            Self::LithiumNiobate => {
                // Trigonal/rhombohedral; approximate hexagonal setting
                // a = 5.148 Å, c = 13.863 Å (hexagonal)
                let a = 5.148e-10_f64;
                let c = 13.863e-10_f64;
                let num =
                    (4.0 / 3.0) * (h * h + h * k + k * k) as f64 + (a / c).powi(2) * (l * l) as f64;
                if num <= 0.0 {
                    return f64::INFINITY;
                }
                a / num.sqrt()
            }
            Self::Custom { d_spacing, .. } => *d_spacing,
        }
    }

    /// Structure factor magnitude |F_hkl| (in units of electron scattering length).
    ///
    /// For diamond-cubic crystals the structure factor is:
    /// ```text
    /// F = f_atom × |1 + (-1)^(h+k+l) + e^{iπ(h+k)/2} + ...|  (FCC + 2-atom basis)
    /// ```
    /// Simplified: |F| = 8 f_atom if (h+k+l) ≡ 0 mod 4 and all same parity;
    ///             |F| = 4√2 f_atom if (h+k+l) ≡ 2 mod 4; 0 otherwise.
    ///
    /// Here we use approximate atomic form factors f ≈ Z (forward-scattering limit).
    pub fn structure_factor(&self, h: i32, k: i32, l: i32) -> f64 {
        let sum_hkl = h + k + l;
        let all_same_parity = (h % 2 == k % 2) && (k % 2 == l % 2);

        match self {
            Self::Silicon => {
                // Z_Si = 14 (forward-scattering limit, f ≈ Z)
                diamond_cubic_structure_factor(h, k, l, 14.0)
            }
            Self::Germanium => {
                // Z_Ge = 32
                diamond_cubic_structure_factor(h, k, l, 32.0)
            }
            Self::Diamond => {
                // Z_C = 6
                diamond_cubic_structure_factor(h, k, l, 6.0)
            }
            Self::Quartz => {
                // SiO₂ — simplified: |F| ∝ Z_Si + 2·Z_O = 14 + 16 = 30
                if all_same_parity {
                    30.0
                } else {
                    0.0
                }
            }
            Self::LithiumNiobate => {
                // LiNbO₃ — Z_Li=3, Z_Nb=41, Z_O=8; simplified
                if all_same_parity || sum_hkl % 2 == 0 {
                    52.0
                } else {
                    0.0
                }
            }
            Self::Custom {
                structure_factor, ..
            } => *structure_factor,
        }
    }
}

/// Structure factor magnitude |F_hkl| for the diamond-cubic lattice.
///
/// The diamond structure = FCC Bravais lattice + 2-atom basis at (0,0,0)
/// and (1/4, 1/4, 1/4).  The structure factor is:
///
/// ```text
/// F_hkl = f_atom · F_FCC · (1 + e^{iπ(h+k+l)/2})
/// ```
///
/// where F_FCC = 4 for all-even or all-odd indices (0 for mixed parity).
/// The diamond basis phase factor gives:
///
/// - **All-odd** (h+k+l always odd): |1 + e^{iπ·odd/2}| = √2 (always).
/// - **All-even, sum ≡ 0 (mod 4)**: |1 + 1| = 2  →  |F| = 8 Z.
/// - **All-even, sum ≡ 2 (mod 4)**: |1 − 1| = 0  →  forbidden.
fn diamond_cubic_structure_factor(h: i32, k: i32, l: i32, z: f64) -> f64 {
    // FCC extinction rule: mixed parity → F = 0
    let all_odd = h % 2 != 0 && k % 2 != 0 && l % 2 != 0;
    let all_even = h % 2 == 0 && k % 2 == 0 && l % 2 == 0;
    if !all_odd && !all_even {
        return 0.0;
    }
    let sum = h + k + l;
    if all_odd {
        // For all-odd indices the sum is always odd → e^{iπ·odd/2} = ±i
        // |1 ± i| = √2  →  |F| = 4Z·√2
        4.0 * z * (2.0_f64).sqrt()
    } else {
        // all_even: sum is always even
        match sum.unsigned_abs() % 4 {
            0 => 8.0 * z, // |1 + 1| = 2  →  4Z × 2
            2 => 0.0,     // |1 − 1| = 0  →  systematic absence
            _ => 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BraggCrystal
// ═══════════════════════════════════════════════════════════════════════════

/// Bragg crystal for X-ray monochromator or analyser optics.
///
/// The crystal diffracts photons when the Bragg condition is satisfied:
/// ```text
/// m λ = 2 d sin(θ_B)
/// ```
/// The rocking-curve FWHM (Darwin width) for a perfect crystal is:
/// ```text
/// W = 2 r_e λ² |F_hkl| / (π V sin(2θ_B))
/// ```
/// where V is the unit-cell volume.
#[derive(Debug, Clone)]
pub struct BraggCrystal {
    /// Crystal material and its d-spacing model.
    pub material: CrystalMaterial,
    /// Miller index h.
    pub miller_h: i32,
    /// Miller index k.
    pub miller_k: i32,
    /// Miller index l.
    pub miller_l: i32,
    /// Pre-computed interplanar d-spacing (m).
    pub d_spacing: f64,
}

impl BraggCrystal {
    /// Construct a Bragg crystal for the specified (hkl) reflection.
    ///
    /// # Errors
    /// Returns an error if (h,k,l) = (0,0,0) or if the d-spacing is not
    /// finite and positive.
    pub fn new(material: CrystalMaterial, h: i32, k: i32, l: i32) -> Result<Self> {
        if h == 0 && k == 0 && l == 0 {
            return Err(OxiPhotonError::NumericalError(
                "Miller indices (0,0,0) are not a valid reflection".into(),
            ));
        }
        let d = material.d_spacing(h, k, l);
        if !d.is_finite() || d <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "non-positive d-spacing {d:.3e} m for ({h},{k},{l})"
            )));
        }
        Ok(Self {
            d_spacing: d,
            material,
            miller_h: h,
            miller_k: k,
            miller_l: l,
        })
    }

    // ─── Bragg geometry ───────────────────────────────────────────────────

    /// Bragg angle for first-order diffraction:
    /// ```text
    /// θ_B = arcsin( λ / 2d )
    /// ```
    ///
    /// Returns `None` if λ > 2d (no diffraction possible at this energy).
    pub fn bragg_angle(&self, wavelength: f64) -> Option<f64> {
        let arg = wavelength / (2.0 * self.d_spacing);
        if arg.abs() > 1.0 {
            return None;
        }
        Some(arg.asin())
    }

    /// Wavelength selected at a given Bragg angle θ (rad):
    /// ```text
    /// λ = 2 d sin(θ)
    /// ```
    pub fn wavelength_at_angle(&self, theta_rad: f64) -> f64 {
        2.0 * self.d_spacing * theta_rad.sin()
    }

    // ─── Darwin width and reflectivity ────────────────────────────────────

    /// Unit-cell volume V (m³) for the crystal material.
    fn unit_cell_volume(&self) -> f64 {
        match &self.material {
            CrystalMaterial::Silicon => {
                let a = 5.431_020_511e-10_f64;
                a.powi(3)
            }
            CrystalMaterial::Germanium => {
                let a = 5.658e-10_f64;
                a.powi(3)
            }
            CrystalMaterial::Diamond => {
                let a = 3.5668e-10_f64;
                a.powi(3)
            }
            CrystalMaterial::Quartz => {
                // Hexagonal: V = a²c sin(60°) for SiO₂ (α-quartz)
                let a = 4.913e-10_f64;
                let c = 5.405e-10_f64;
                (3.0_f64).sqrt() / 2.0 * a * a * c
            }
            CrystalMaterial::LithiumNiobate => {
                let a = 5.148e-10_f64;
                let c = 13.863e-10_f64;
                (3.0_f64).sqrt() / 2.0 * a * a * c
            }
            CrystalMaterial::Custom { d_spacing, .. } => {
                // Estimate V ~ d³ (rough, consistent with simple-cubic)
                d_spacing.powi(3)
            }
        }
    }

    /// Darwin width (rocking curve FWHM) in radians:
    /// ```text
    /// W = 2 r_e λ² |F_hkl| / ( π V sin(2θ_B) )
    /// ```
    pub fn darwin_width_rad(&self, wavelength: f64) -> f64 {
        let theta = match self.bragg_angle(wavelength) {
            Some(t) => t,
            None => return 0.0,
        };
        let sin2t = (2.0 * theta).sin();
        if sin2t.abs() < 1e-15 {
            return 0.0;
        }
        let f_hkl = self
            .material
            .structure_factor(self.miller_h, self.miller_k, self.miller_l);
        let v = self.unit_cell_volume();
        2.0 * R_ELECTRON * wavelength * wavelength * f_hkl / (PI * v * sin2t)
    }

    /// Integrated reflectivity (rad) for a perfect crystal in Bragg geometry:
    /// ```text
    /// R_int = W · π/2
    /// ```
    ///
    /// This is the kinematic result (valid when absorption is negligible).
    pub fn integrated_reflectivity(&self, wavelength: f64) -> f64 {
        self.darwin_width_rad(wavelength) * PI / 2.0
    }

    // ─── Spectroscopy ─────────────────────────────────────────────────────

    /// Energy–angle dispersion (eV/rad) at the given wavelength:
    /// ```text
    /// dE/dθ = −E · cot(θ)
    /// ```
    pub fn energy_dispersion(&self, wavelength: f64) -> f64 {
        let theta = match self.bragg_angle(wavelength) {
            Some(t) => t,
            None => return 0.0,
        };
        let energy_ev = lambda_to_ev(wavelength);
        -energy_ev / theta.tan()
    }

    /// Energy resolving power E/ΔE:
    /// ```text
    /// E/ΔE = cot(θ_B) / W
    /// ```
    ///
    /// Returns 0 if Darwin width is zero (forbidden reflection or θ_B = 0).
    pub fn resolving_power(&self, wavelength: f64) -> f64 {
        let theta = match self.bragg_angle(wavelength) {
            Some(t) => t,
            None => return 0.0,
        };
        let w = self.darwin_width_rad(wavelength);
        if w <= 0.0 {
            return 0.0;
        }
        1.0 / (theta.tan() * w)
    }

    /// Decide whether a crystal of thickness `t` (m) operates in Bragg
    /// (reflection) geometry or Laue (transmission) geometry.
    ///
    /// The Bragg geometry is used when the extinction length
    /// Λ_ext = π V / (r_e λ |F|) is much less than t; otherwise Laue is
    /// preferred.  The threshold here is t > 10 Λ_ext ⟹ Bragg.
    pub fn is_bragg_geometry(&self, wavelength: f64, crystal_thickness: f64) -> bool {
        let f_hkl = self
            .material
            .structure_factor(self.miller_h, self.miller_k, self.miller_l);
        let v = self.unit_cell_volume();
        if f_hkl <= 0.0 || wavelength <= 0.0 {
            return true; // default to Bragg
        }
        let extinction_length = PI * v / (R_ELECTRON * wavelength * f_hkl);
        crystal_thickness > 10.0 * extinction_length
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Johann Spectrometer
// ═══════════════════════════════════════════════════════════════════════════

/// Johann bent-crystal spectrometer for high-resolution X-ray emission /
/// absorption spectroscopy.
///
/// The crystal is bent to radius R (twice the Rowland circle radius).  The
/// source, crystal, and detector all lie on the Rowland circle of radius R/2.
///
/// Wavelength dispersed: Δθ_source → ΔE at the detector.
#[derive(Debug, Clone)]
pub struct JohannSpectrometer {
    /// The Bragg-diffracting crystal.
    pub crystal: BraggCrystal,
    /// Crystal bending radius (m).
    pub bending_radius: f64,
    /// Source-to-crystal distance (m); equals the Rowland circle radius R/2.
    pub source_distance: f64,
}

impl JohannSpectrometer {
    /// Construct a Johann spectrometer.
    ///
    /// # Errors
    /// Returns an error if `radius` is non-positive.
    pub fn new(crystal: BraggCrystal, radius: f64) -> Result<Self> {
        if radius <= 0.0 || !radius.is_finite() {
            return Err(OxiPhotonError::NumericalError(
                "bending_radius must be positive and finite".into(),
            ));
        }
        let source_distance = radius / 2.0;
        Ok(Self {
            crystal,
            bending_radius: radius,
            source_distance,
        })
    }

    /// Resolving power E/ΔE of the Johann spectrometer.
    ///
    /// The geometric Johann aberration limits the resolution; for a bent
    /// crystal of length L and bending radius R:
    ///
    /// ```text
    /// E/ΔE ≈ cot(θ_B) · R / (L²/(8R·sin(θ_B)))
    /// ```
    ///
    /// For an ideal (perfectly focused) instrument this approaches the Darwin
    /// width limit:
    /// ```text
    /// (E/ΔE)_ideal = cot(θ_B) / W
    /// ```
    ///
    /// Here we return the Darwin-width-limited value as an upper bound.
    pub fn resolving_power(&self, wavelength: f64) -> f64 {
        self.crystal.resolving_power(wavelength)
    }

    /// Energy range accessible by rotating the spectrometer (eV).
    ///
    /// The Bragg angle can vary from a minimum of θ_min (limited by the
    /// geometry) to near 90°.  We assume θ_min = 10° = π/18 rad.
    pub fn energy_range(&self) -> (f64, f64) {
        let d = self.crystal.d_spacing;
        // E_max ↔ θ_B,min (smallest usable angle, ~10°)
        let theta_min = PI / 18.0;
        // E_min ↔ θ_B = 90° → λ = 2d
        let lambda_at_90 = 2.0 * d;
        let lambda_at_min = 2.0 * d * theta_min.sin();
        let e_min = lambda_to_ev(lambda_at_90);
        let e_max = lambda_to_ev(lambda_at_min);
        (e_min, e_max)
    }

    /// Image distance p' (m) from the crystal to the detector on the Rowland
    /// circle:
    ///
    /// ```text
    /// 1/p' + 1/p = 2/(R sin(θ_B))   ← sagittal (Johann, meridional)
    /// p' = p R sin(θ_B) / (2p − R sin(θ_B))
    /// ```
    pub fn image_distance(&self, wavelength: f64) -> Option<f64> {
        let theta = self.crystal.bragg_angle(wavelength)?;
        let r_sin = self.bending_radius * theta.sin();
        let p = self.source_distance;
        let denom = 2.0 * p - r_sin;
        if denom.abs() < 1e-15 {
            return None;
        }
        Some(p * r_sin / denom)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // Si(111): d ≈ 3.136 Å, λ = 0.154 nm (Cu Kα)
    const CU_KALPHA: f64 = 0.154_056e-9; // m

    fn si_111() -> BraggCrystal {
        BraggCrystal::new(CrystalMaterial::Silicon, 1, 1, 1).expect("valid Si(111)")
    }

    #[test]
    fn si_d_spacing_111() {
        let crystal = si_111();
        // Si(111): d = a/√3 = 5.431Å/√3 ≈ 3.136 Å
        let expected = 5.431_020_511e-10 / (3.0_f64).sqrt();
        assert_abs_diff_eq!(crystal.d_spacing, expected, epsilon = 1e-14);
    }

    #[test]
    fn bragg_angle_cu_kalpha() {
        let crystal = si_111();
        let theta = crystal
            .bragg_angle(CU_KALPHA)
            .expect("Cu Kα diffracts on Si(111)");
        // θ_B = arcsin(0.154056/(2*3.136e-1)) — known value ≈ 14.22°
        let expected_rad = (CU_KALPHA / (2.0 * crystal.d_spacing)).asin();
        assert_abs_diff_eq!(theta, expected_rad, epsilon = 1e-10);
    }

    #[test]
    fn bragg_angle_returns_none_for_too_long_wavelength() {
        let crystal = si_111();
        // λ > 2d → no diffraction
        let long_lambda = 2.0 * crystal.d_spacing + 1e-11;
        assert!(crystal.bragg_angle(long_lambda).is_none());
    }

    #[test]
    fn wavelength_at_angle_roundtrip() {
        let crystal = si_111();
        let theta = crystal.bragg_angle(CU_KALPHA).expect("valid angle");
        let lambda_back = crystal.wavelength_at_angle(theta);
        assert_abs_diff_eq!(lambda_back, CU_KALPHA, epsilon = 1e-18);
    }

    #[test]
    fn darwin_width_positive() {
        let crystal = si_111();
        let w = crystal.darwin_width_rad(CU_KALPHA);
        assert!(w > 0.0, "Darwin width should be positive, got {w:.3e}");
        // Si(111) Darwin width ~ 10 µrad for 8 keV X-rays
        assert!(w < 1e-3, "Darwin width unexpectedly large: {w:.3e} rad");
    }

    #[test]
    fn resolving_power_positive() {
        let crystal = si_111();
        let rp = crystal.resolving_power(CU_KALPHA);
        assert!(rp > 0.0 && rp.is_finite());
    }

    #[test]
    fn energy_dispersion_negative() {
        // dE/dθ = −E cot(θ): must be negative (energy decreases with increasing angle)
        let crystal = si_111();
        let disp = crystal.energy_dispersion(CU_KALPHA);
        assert!(
            disp < 0.0,
            "energy dispersion should be negative, got {disp:.3e}"
        );
    }

    #[test]
    fn bad_miller_indices() {
        assert!(BraggCrystal::new(CrystalMaterial::Silicon, 0, 0, 0).is_err());
    }

    #[test]
    fn johann_energy_range_ordered() {
        let crystal = si_111();
        let spec = JohannSpectrometer::new(crystal, 0.5).expect("valid spectrometer");
        let (e_min, e_max) = spec.energy_range();
        assert!(
            e_min < e_max,
            "E_min should be less than E_max: {e_min:.1} vs {e_max:.1}"
        );
    }

    #[test]
    fn johann_image_distance_positive() {
        let crystal = si_111();
        let spec = JohannSpectrometer::new(crystal, 0.5).expect("valid spectrometer");
        let img = spec
            .image_distance(CU_KALPHA)
            .expect("image distance exists");
        assert!(img > 0.0 && img.is_finite());
    }
}
