//! Multilayer X-ray and EUV mirror simulation.
//!
//! Covers:
//! - Mo/Si multilayer mirrors optimised for 13.5 nm EUV (EUVL lithography).
//! - Hard X-ray multilayer mirrors (W/C, W/Si, …).
//! - Scalar coupled-wave (kinematic) model for peak reflectivity and bandwidth.
//! - EUV mirror transmission chains for scanner optics.
//!
//! # Model
//! The kinematic (first Born approximation) reflectivity for a periodic
//! multilayer with N bilayers of period Λ is:
//!
//! ```text
//! r_bilayer = (n_a² − n_b²) / (n_a² + n_b²)   (approximate contrast)
//! R_peak    = tanh²( N |r_bilayer| )
//! ```
//!
//! This is the *scalar* (Parratt-like limit) formula.  For a rigorous
//! calculation, use the full transfer-matrix method.  The approximation is
//! accurate within ~5% for typical Mo/Si stacks.
//!
//! # References
//! - Spiller, E. *Soft X-ray Optics*, SPIE Press (1994).
//! - Born & Wolf, *Principles of Optics*, §1.6 (multilayer reflectance).

use crate::error::{OxiPhotonError, Result};
use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════════════
// XrayMaterial
// ═══════════════════════════════════════════════════════════════════════════

/// Optical constants of a material in the X-ray / EUV regime.
///
/// The complex refractive index is:
/// ```text
/// ñ = 1 − δ + i·β
/// ```
/// where δ is the refractive-index *decrement* (always positive for X-rays)
/// and β is the absorption index.
#[derive(Debug, Clone)]
pub struct XrayMaterial {
    /// Human-readable material name (e.g. `"Mo"`, `"Si"`).
    pub name: String,
    /// Refractive index decrement δ (dimensionless, > 0 for X-rays).
    pub delta: f64,
    /// Absorption index β (dimensionless, ≥ 0).
    pub beta: f64,
    /// Mass density (g cm⁻³).
    pub density: f64,
}

impl XrayMaterial {
    /// Create a custom material with explicit optical constants.
    pub fn new(name: impl Into<String>, delta: f64, beta: f64, density: f64) -> Self {
        Self {
            name: name.into(),
            delta,
            beta,
            density,
        }
    }

    // ─── Pre-tabulated materials at specific wavelengths ──────────────────

    /// Molybdenum at λ = 13.5 nm (EUV).
    ///
    /// CXRO tabulated values: δ = 0.077, β = 0.00607 (Henke et al.)
    pub fn molybdenum_13_5nm() -> Self {
        Self::new("Mo@13.5nm", 0.077, 0.00607, 10.28)
    }

    /// Silicon at λ = 13.5 nm (EUV).
    ///
    /// CXRO tabulated values: δ = 0.001, β = 1.71e-4
    pub fn silicon_13_5nm() -> Self {
        Self::new("Si@13.5nm", 0.001, 1.71e-4, 2.33)
    }

    /// Tungsten at λ = 0.1 nm (hard X-ray, 12.4 keV).
    pub fn tungsten_0_1nm() -> Self {
        Self::new("W@0.1nm", 4.8e-5, 3.6e-6, 19.3)
    }

    /// Carbon at λ = 0.1 nm (hard X-ray).
    pub fn carbon_0_1nm() -> Self {
        Self::new("C@0.1nm", 2.4e-6, 4.1e-10, 2.0)
    }

    /// Ruthenium at λ = 13.5 nm — common capping layer.
    pub fn ruthenium_13_5nm() -> Self {
        Self::new("Ru@13.5nm", 0.0274, 0.00623, 12.4)
    }

    /// Complex refractive index as (real, imag) pair: (1 − δ, β).
    pub fn refractive_index(&self) -> (f64, f64) {
        (1.0 - self.delta, self.beta)
    }

    /// Absorption length (1/e) at the given wavelength (m):
    /// ```text
    /// l_abs = λ / (4π β)
    /// ```
    pub fn absorption_length_m(&self, wavelength_m: f64) -> f64 {
        if self.beta <= 0.0 || wavelength_m <= 0.0 {
            return f64::INFINITY;
        }
        wavelength_m / (4.0 * PI * self.beta)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MultilayerMirror
// ═══════════════════════════════════════════════════════════════════════════

/// Periodic multilayer X-ray / EUV mirror.
///
/// Each bilayer consists of:
/// - **Layer A** (absorber, thickness γ·Λ) — high-Z or high-δ material, e.g. Mo.
/// - **Layer B** (spacer, thickness (1−γ)·Λ) — low-Z material, e.g. Si.
///
/// The Bragg condition for maximum reflectivity at order m is:
/// ```text
/// m λ = 2 Λ cos(θ_i)
/// ```
/// where θ_i is the grazing incidence angle measured from the surface *normal*.
#[derive(Debug, Clone)]
pub struct MultilayerMirror {
    /// Bilayer period Λ (m).
    pub bilayer_period: f64,
    /// Number of bilayers N.
    pub n_bilayers: usize,
    /// Thickness fraction γ of the absorber layer (0 < γ < 1).
    pub gamma: f64,
    /// Absorber material (e.g. Mo).
    pub layer_a: XrayMaterial,
    /// Spacer material (e.g. Si).
    pub layer_b: XrayMaterial,
    /// Design wavelength (m).
    pub wavelength: f64,
    /// Angle of incidence from surface *normal* (rad).
    pub incidence_angle: f64,
}

impl MultilayerMirror {
    /// General constructor.
    ///
    /// # Errors
    /// Returns an error if `bilayer_period` ≤ 0, `n_bilayers` = 0, or
    /// `gamma` ∉ (0, 1).
    pub fn new(
        bilayer_period: f64,
        n_bilayers: usize,
        gamma: f64,
        layer_a: XrayMaterial,
        layer_b: XrayMaterial,
        wavelength: f64,
        incidence_angle: f64,
    ) -> Result<Self> {
        if bilayer_period <= 0.0 || !bilayer_period.is_finite() {
            return Err(OxiPhotonError::NumericalError(
                "bilayer_period must be positive and finite".into(),
            ));
        }
        if n_bilayers == 0 {
            return Err(OxiPhotonError::NumericalError(
                "n_bilayers must be at least 1".into(),
            ));
        }
        if !(0.0..=1.0).contains(&gamma) {
            return Err(OxiPhotonError::NumericalError(
                "gamma must be in [0, 1]".into(),
            ));
        }
        if !wavelength.is_finite() || wavelength <= 0.0 {
            return Err(OxiPhotonError::InvalidWavelength(wavelength));
        }
        Ok(Self {
            bilayer_period,
            n_bilayers,
            gamma,
            layer_a,
            layer_b,
            wavelength,
            incidence_angle,
        })
    }

    /// Optimised Mo/Si multilayer mirror for 13.5 nm EUV.
    ///
    /// Standard parameters: Λ ≈ 6.9 nm, γ = 0.4 (Mo fraction), N = 40,
    /// normal incidence.
    pub fn new_mo_si_euv(n_bilayers: usize) -> Self {
        Self {
            bilayer_period: 6.9e-9,
            n_bilayers,
            gamma: 0.4,
            layer_a: XrayMaterial::molybdenum_13_5nm(),
            layer_b: XrayMaterial::silicon_13_5nm(),
            wavelength: 13.5e-9,
            incidence_angle: 0.0, // normal incidence
        }
    }

    // ─── Core optics ──────────────────────────────────────────────────────

    /// Bragg wavelength at diffraction order `m`:
    /// ```text
    /// λ_m = 2 Λ cos(θ_i) / m
    /// ```
    pub fn bragg_wavelength(&self, order: usize) -> f64 {
        if order == 0 {
            return f64::INFINITY;
        }
        2.0 * self.bilayer_period * self.incidence_angle.cos() / order as f64
    }

    /// Fresnel contrast (reflectance amplitude per bilayer interface):
    /// ```text
    /// r = (δ_a − δ_b) / (2 − δ_a − δ_b)   ≈ (δ_a − δ_b) / 2   for δ ≪ 1
    /// ```
    fn bilayer_amplitude(&self) -> f64 {
        let da = self.layer_a.delta;
        let db = self.layer_b.delta;
        let denom = 2.0 - da - db;
        if denom.abs() < 1e-30 {
            return 0.0;
        }
        (da - db).abs() / denom
    }

    /// Peak reflectivity using the kinematic (scalar) coupled-wave model:
    /// ```text
    /// R_peak = tanh²( N |r| )
    /// ```
    /// where |r| is the single-bilayer Fresnel amplitude contrast.
    pub fn peak_reflectivity(&self) -> f64 {
        let r = self.bilayer_amplitude();
        let x = self.n_bilayers as f64 * r;
        x.tanh().powi(2)
    }

    /// Reflectivity at wavelength `lambda` relative to the Bragg peak.
    ///
    /// Near the Bragg wavelength λ_B = Λ·cos(θ_i) the rocking curve has a
    /// roughly Lorentzian profile:
    ///
    /// ```text
    /// R(λ) ≈ R_peak / (1 + ((λ − λ_B) / (Δλ/2))² )
    /// ```
    ///
    /// where Δλ is the FWHM bandwidth.
    pub fn reflectivity_at(&self, lambda: f64) -> f64 {
        let lambda_b = self.bragg_wavelength(1);
        let fwhm = self.bandwidth_fwhm();
        if fwhm <= 0.0 {
            if (lambda - lambda_b).abs() < 1e-30 {
                return self.peak_reflectivity();
            }
            return 0.0;
        }
        let half = fwhm / 2.0;
        let x = (lambda - lambda_b) / half;
        self.peak_reflectivity() / (1.0 + x * x)
    }

    /// Relative bandwidth (FWHM) of the reflectivity peak:
    /// ```text
    /// Δλ/λ ≈ 2 γ (1−γ) |δ_a − δ_b| / δ_avg
    /// ```
    ///
    /// Absolute FWHM = (Δλ/λ) × λ_B.
    pub fn bandwidth_fwhm(&self) -> f64 {
        let da = self.layer_a.delta;
        let db = self.layer_b.delta;
        let d_avg = self.gamma * da + (1.0 - self.gamma) * db;
        if d_avg <= 0.0 {
            return 0.0;
        }
        let relative = 2.0 * self.gamma * (1.0 - self.gamma) * (da - db).abs() / d_avg;
        let lambda_b = self.bragg_wavelength(1);
        relative * lambda_b
    }

    /// 1/e penetration depth into the multilayer stack (number of bilayers):
    ///
    /// ```text
    /// N_pen = 1 / (2 |r_bilayer|)
    /// ```
    ///
    /// Depth in metres = N_pen × Λ.
    pub fn penetration_depth(&self) -> f64 {
        let r = self.bilayer_amplitude();
        if r <= 0.0 {
            return f64::INFINITY;
        }
        let n_pen = 1.0 / (2.0 * r);
        n_pen * self.bilayer_period
    }

    /// Thermal wavelength shift caused by a temperature change Δ T (K):
    ///
    /// ```text
    /// Δλ = Λ · α · ΔT
    /// ```
    ///
    /// where α is the linear coefficient of thermal expansion of the stack.
    pub fn thermal_shift(&self, delta_temp: f64, thermal_expansion: f64) -> f64 {
        self.bilayer_period * thermal_expansion * delta_temp
    }

    /// Absorption-weighted effective number of bilayers contributing to the
    /// reflectivity (saturation indicator):
    ///
    /// ```text
    /// N_eff = min(N, N_pen / Λ)
    /// ```
    pub fn effective_bilayers(&self) -> f64 {
        let r = self.bilayer_amplitude();
        if r <= 0.0 {
            return self.n_bilayers as f64;
        }
        let n_pen = 1.0 / (2.0 * r); // bilayer units
        (self.n_bilayers as f64).min(n_pen)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// EUV Mirror
// ═══════════════════════════════════════════════════════════════════════════

/// EUV lithography (EUVL) mirror stack designed for 13.5 nm.
///
/// Real EUVL scanners use six to eight such mirrors.  Each mirror
/// achieves ~67–70% peak reflectivity; the system throughput is the product
/// of all mirror reflectivities.
#[derive(Debug, Clone)]
pub struct EuvMirror {
    /// The underlying Mo/Si multilayer.
    pub multilayer: MultilayerMirror,
    /// Substrate material label (e.g. `"fused silica"`, `"ULE glass"`).
    pub substrate: String,
    /// Ruthenium capping layer thickness (nm).  Typical: 2–3 nm.
    pub capping_layer_nm: f64,
}

impl EuvMirror {
    /// Standard EUVL mirror: 40 Mo/Si bilayers + 2.5 nm Ru capping.
    pub fn new_standard() -> Self {
        Self {
            multilayer: MultilayerMirror::new_mo_si_euv(40),
            substrate: "ULE glass".to_string(),
            capping_layer_nm: 2.5,
        }
    }

    /// Peak reflectivity including the attenuation from the Ru capping layer.
    ///
    /// The capping layer absorbs a fraction of the incident wave before it
    /// reaches the Mo/Si stack.  The transmitted amplitude through the capping
    /// is:
    ///
    /// ```text
    /// T_cap = exp(−2π β_Ru d_cap / λ)
    /// ```
    ///
    /// where d_cap is the capping layer thickness.  The reflectivity is then:
    /// ```text
    /// R_total ≈ T_cap² · R_multilayer
    /// ```
    pub fn peak_reflectivity_with_capping(&self) -> f64 {
        let lambda = self.multilayer.wavelength;
        let d_cap = self.capping_layer_nm * 1e-9;
        let ru = XrayMaterial::ruthenium_13_5nm();
        let mu_cap = 4.0 * PI * ru.beta / lambda; // absorption coefficient m⁻¹
        let t_cap_sq = (-2.0 * mu_cap * d_cap).exp();
        t_cap_sq * self.multilayer.peak_reflectivity()
    }

    /// Throughput of a chain of `n_mirrors` identical mirrors.
    ///
    /// ```text
    /// T_chain = R_mirror^n_mirrors
    /// ```
    pub fn n_mirror_transmission(&self, n_mirrors: usize) -> f64 {
        self.peak_reflectivity_with_capping().powi(n_mirrors as i32)
    }

    /// Estimate collector étendue-weighted flux reduction for a 6-mirror scanner.
    pub fn six_mirror_scanner_transmission(&self) -> f64 {
        self.n_mirror_transmission(6)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn mo_si_bragg_wavelength_normal_incidence() {
        // At normal incidence θ = 0 → cos(θ) = 1 → λ_B = 2Λ/1
        let mirror = MultilayerMirror::new_mo_si_euv(40);
        let lambda_b = mirror.bragg_wavelength(1);
        // Λ = 6.9 nm → λ_B = 13.8 nm ≈ 13.5 nm (within design tolerance)
        assert!(
            (lambda_b - 13.8e-9).abs() < 1e-11,
            "λ_B should be ~13.8 nm, got {:.3e} m",
            lambda_b
        );
    }

    #[test]
    fn mo_si_peak_reflectivity_reasonable() {
        let mirror = MultilayerMirror::new_mo_si_euv(40);
        let r = mirror.peak_reflectivity();
        // Kinematic model should give R > 0.5 for 40 bilayers Mo/Si
        assert!(r > 0.5, "peak reflectivity should exceed 0.5, got {r:.4}");
        assert!(r <= 1.0, "reflectivity cannot exceed 1, got {r:.4}");
    }

    #[test]
    fn multilayer_construction_bad_period() {
        let mat = XrayMaterial::molybdenum_13_5nm();
        let res = MultilayerMirror::new(-1e-9, 40, 0.4, mat.clone(), mat, 13.5e-9, 0.0);
        assert!(res.is_err());
    }

    #[test]
    fn multilayer_construction_bad_gamma() {
        let mat = XrayMaterial::molybdenum_13_5nm();
        let res = MultilayerMirror::new(6.9e-9, 40, 1.5, mat.clone(), mat, 13.5e-9, 0.0);
        assert!(res.is_err());
    }

    #[test]
    fn bandwidth_increases_with_contrast() {
        // Higher contrast (larger |δ_a − δ_b|) → wider bandwidth
        let mo = XrayMaterial::molybdenum_13_5nm(); // δ = 0.077
        let si = XrayMaterial::silicon_13_5nm(); // δ = 0.001
        let mut high_contrast = MultilayerMirror::new_mo_si_euv(40);
        high_contrast.layer_a = mo;
        high_contrast.layer_b = si.clone();
        // Custom low-contrast mirror: same δ materials
        let low_contrast = MultilayerMirror {
            layer_a: XrayMaterial::new("Lo_A", 0.01, 1e-4, 5.0),
            layer_b: XrayMaterial::new("Lo_B", 0.009, 1e-4, 5.0),
            ..MultilayerMirror::new_mo_si_euv(40)
        };
        assert!(
            high_contrast.bandwidth_fwhm() > low_contrast.bandwidth_fwhm(),
            "higher contrast should yield wider bandwidth"
        );
    }

    #[test]
    fn penetration_depth_finite_for_nonzero_contrast() {
        let mirror = MultilayerMirror::new_mo_si_euv(40);
        let depth = mirror.penetration_depth();
        assert!(depth.is_finite() && depth > 0.0);
    }

    #[test]
    fn euv_mirror_capping_reduces_reflectivity() {
        let euv = EuvMirror::new_standard();
        let r_with = euv.peak_reflectivity_with_capping();
        let r_without = euv.multilayer.peak_reflectivity();
        // Capping absorbs some light → should reduce reflectivity
        assert!(
            r_with <= r_without,
            "capping layer should reduce reflectivity: {r_with:.4} vs {r_without:.4}"
        );
    }

    #[test]
    fn n_mirror_transmission_monotone_decreasing() {
        let euv = EuvMirror::new_standard();
        let t6 = euv.n_mirror_transmission(6);
        let t8 = euv.n_mirror_transmission(8);
        assert!(t8 < t6, "more mirrors → less transmission");
    }

    #[test]
    fn thermal_shift_proportional() {
        let mirror = MultilayerMirror::new_mo_si_euv(40);
        let alpha = 1e-6; // CTE 1 ppm/K
        let dt1 = 10.0;
        let dt2 = 20.0;
        let s1 = mirror.thermal_shift(dt1, alpha);
        let s2 = mirror.thermal_shift(dt2, alpha);
        assert_abs_diff_eq!(s2, 2.0 * s1, epsilon = 1e-25);
    }

    #[test]
    fn material_absorption_length_silicon() {
        let si = XrayMaterial::silicon_13_5nm();
        let l_abs = si.absorption_length_m(13.5e-9);
        // l_abs = λ/(4π β) = 13.5e-9/(4π·1.71e-4) ≈ 6.29e-6 m ≈ 6.3 μm
        let expected = 13.5e-9 / (4.0 * PI * 1.71e-4);
        assert_abs_diff_eq!(l_abs, expected, epsilon = 1e-10);
    }
}
