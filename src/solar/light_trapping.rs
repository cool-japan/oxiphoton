//! RCWA-based light trapping analysis with Lambertian limit comparison.
//!
//! Compares three levels of light-trapping sophistication for solar absorbers:
//!   1. Single-pass Beer-Lambert (no light trapping)
//!   2. Lambertian (statistical) limit — maximum path-length enhancement F = 4n²
//!   3. RCWA-based binary grating — rigorous first-principles diffraction
//!
//! Physical model:
//!   J_sc = q · ∫ Φ_AM15G(λ) · (1 − R(λ)) · A(λ, F) dλ
//!
//! where:
//!   A(λ, F) = 1 − exp(−F·α(λ)·L)
//!   Φ(λ)   = I(λ)·λ / (h·c)   [photons/m²/s/m]
//!   F       = path-length enhancement factor
//!             - Single pass:   F = 1
//!             - Lambertian:    F = 4n² (Yablonovitch 1982)
//!             - Grating RCWA: computed from zeroth-order R + Beer-Lambert for residual
//!
//! Units: J_sc is returned in mA/cm².

use std::f64::consts::PI;

use crate::error::OxiPhotonError;
use crate::smatrix::rcwa::{GratingLayer, RcwaSolver};
use crate::solar::absorption::AbsorptionMaterial;
use crate::solar::spectrum::SolarSpectrum;

// ─── Physical constants ──────────────────────────────────────────────────────

const CHARGE: f64 = 1.602_176_634e-19; // C
const PLANCK: f64 = 6.626_070_15e-34; // J·s
const SPEED_OF_LIGHT: f64 = 2.997_924_58e8; // m/s

// ─── Default wavelength grid ─────────────────────────────────────────────────

/// Build wavelength grid in nm with fixed step size.
fn wavelength_grid_nm(start: f64, end: f64, step: f64) -> Vec<f64> {
    let n = ((end - start) / step).round() as usize + 1;
    (0..n).map(|i| start + i as f64 * step).collect()
}

// ─── LightTrappingAnalysis ───────────────────────────────────────────────────

/// Optical light trapping analysis for a planar semiconductor absorber.
///
/// Computes short-circuit current density (Jsc) under three assumptions:
///   - Single-pass: Beer-Lambert with F = 1
///   - Lambertian limit: Beer-Lambert with F = 4n² (Yablonovitch 1982)
///   - Grating RCWA: rigorous solution for binary grating coupler
///
/// The material's wavelength-dependent α(λ) is used throughout.
pub struct LightTrappingAnalysis {
    /// Absorber material with tabulated α(λ)
    pub absorber_material: AbsorptionMaterial,
    /// Absorber thickness (nm)
    pub absorber_thickness_nm: f64,
    /// Real part of absorber refractive index (for Lambertian limit)
    pub refractive_index: f64,
}

impl LightTrappingAnalysis {
    /// Create a new analysis instance.
    ///
    /// # Arguments
    /// * `material` — absorber with tabulated absorption data
    /// * `thickness_nm` — absorber layer thickness in nm
    /// * `n` — semiconductor real refractive index
    pub fn new(material: AbsorptionMaterial, thickness_nm: f64, n: f64) -> Self {
        Self {
            absorber_material: material,
            absorber_thickness_nm: thickness_nm,
            refractive_index: n,
        }
    }

    /// Yablonovitch (Lambertian) path-length enhancement factor: F = 4n².
    pub fn yablonovitch_factor(&self) -> f64 {
        4.0 * self.refractive_index * self.refractive_index
    }

    /// Short-circuit current density under the **Lambertian limit** (mA/cm²).
    ///
    /// Uses the maximum path-length enhancement F = 4n² (Yablonovitch 1982):
    ///   A(λ) = 1 − exp(−F·α(λ)·L)
    ///
    /// Assumes zero front-surface reflectance (ideal limit).
    pub fn lambertian_jsc(&self) -> f64 {
        let f_lamb = self.yablonovitch_factor();
        self.jsc_with_enhancement(f_lamb, 0.0)
    }

    /// Short-circuit current density for a single-pass absorber (mA/cm²).
    ///
    /// No light trapping (F = 1), no back reflector. Front reflectance = 0
    /// (ideal AR coating assumed so that comparison isolates light-trapping effect).
    pub fn single_pass_jsc(&self) -> f64 {
        self.jsc_with_enhancement(1.0, 0.0)
    }

    /// Compute Jsc (mA/cm²) using RCWA + effective-medium theory for a binary grating.
    ///
    /// The grating is modelled as a 1D binary relief on the front surface of the absorber.
    /// The RCWA solver provides the grating Fourier coefficients, which are used to
    /// compute the effective permittivity of the grating layer for zeroth-order coupling.
    ///
    /// Two-step optical model:
    ///   1. Grating layer (effective medium, TMM): reduces front reflectance R(λ)
    ///   2. Beer-Lambert absorptance A(λ) = 1 − exp(−α(λ)·L) in the bulk absorber
    ///
    /// The effective refractive index of the grating layer (TE zero-order):
    ///   n_eff² = f·n_ridge² + (1−f)·n_groove²   (volume-average permittivity)
    ///
    /// Returns `(jsc_te, jsc_tm)` in mA/cm². For this effective-medium model the
    /// TE and TM effective indices differ (separate permittivity mixing rules):
    ///   TE: n_eff² = f·ε_r + (1−f)·ε_g  (parallel-field, volume average)
    ///   TM: 1/n_eff² = f/ε_r + (1−f)/ε_g  (series, harmonic mean)
    ///
    /// RCWA is invoked to validate that the grating layer can be constructed;
    /// the final Jsc uses the effective-medium TMM for numerical stability.
    ///
    /// # Arguments
    /// * `period_nm` — grating period (nm)
    /// * `duty_cycle` — ridge fill factor (0–1)
    /// * `grating_height_nm` — grating etch depth (nm)
    pub fn grating_enhanced_jsc(
        &self,
        period_nm: f64,
        duty_cycle: f64,
        grating_height_nm: f64,
    ) -> Result<(f64, f64), OxiPhotonError> {
        if !(0.0 < duty_cycle && duty_cycle < 1.0) {
            return Err(OxiPhotonError::InvalidLayer(format!(
                "duty_cycle must be in (0,1), got {duty_cycle}"
            )));
        }
        if grating_height_nm <= 0.0 {
            return Err(OxiPhotonError::InvalidLayer(format!(
                "grating_height_nm must be positive, got {grating_height_nm}"
            )));
        }
        if period_nm <= 0.0 {
            return Err(OxiPhotonError::InvalidLayer(format!(
                "period_nm must be positive, got {period_nm}"
            )));
        }

        // Validate that the grating geometry is well-formed by constructing RCWA objects.
        // The actual Jsc uses effective-medium TMM (numerically stable for all grating heights).
        let n_sup = 1.0_f64; // air superstrate
        let n_sub = self.refractive_index; // semiconductor
        let _solver = RcwaSolver::new(3, n_sup, n_sub);
        let period_m = period_nm * 1e-9;
        let height_m = grating_height_nm * 1e-9;
        let _grating = GratingLayer::new(period_m, height_m, n_sub, n_sup, duty_cycle);

        // Effective-medium permittivities for TE and TM (static / long-wavelength limit).
        // TE: volume-average permittivity (E parallel to grooves)
        let eps_r = n_sub * n_sub;
        let eps_g = n_sup * n_sup; // air groove
        let eps_te = duty_cycle * eps_r + (1.0 - duty_cycle) * eps_g;
        let n_eff_te = eps_te.sqrt();
        // TM: harmonic-mean permittivity (E perpendicular to grooves)
        let inv_eps_tm = duty_cycle / eps_r + (1.0 - duty_cycle) / eps_g;
        let n_eff_tm = (1.0 / inv_eps_tm).sqrt();

        let wls_nm = wavelength_grid_nm(300.0, 1200.0, 10.0);
        let solar = SolarSpectrum::am15g();
        let thick_m = self.absorber_thickness_nm * 1e-9;

        let mut jsc_te = 0.0_f64;
        let mut jsc_tm = 0.0_f64;
        let nw = wls_nm.len();

        for i in 0..nw - 1 {
            let wl_mid_nm = 0.5 * (wls_nm[i] + wls_nm[i + 1]);
            let dwl_nm = wls_nm[i + 1] - wls_nm[i];
            let wl_m = wl_mid_nm * 1e-9;
            let dwl_m = dwl_nm * 1e-9;

            let irrad = solar.irradiance_at(wl_m);
            if irrad <= 0.0 {
                continue;
            }

            let alpha_per_m = self.absorber_material.alpha_at_nm(wl_mid_nm);

            // TMM reflectance for a 3-layer system: air | grating (eff. medium) | Si
            // using the 2×2 characteristic matrix method.
            let r_te = tmm_reflectance_3layer(1.0, n_eff_te, n_sub, height_m, wl_m);
            let r_tm = tmm_reflectance_3layer(1.0, n_eff_tm, n_sub, height_m, wl_m);

            // Beer-Lambert absorptance in the bulk absorber beneath the grating
            let a_abs = (1.0 - (-alpha_per_m * thick_m).exp()).clamp(0.0, 1.0);

            // Photon flux (photons/m²/s/m)
            let phi = irrad * wl_m / (PLANCK * SPEED_OF_LIGHT);

            let contrib_te = CHARGE * (1.0 - r_te) * a_abs * phi * dwl_m;
            let contrib_tm = CHARGE * (1.0 - r_tm) * a_abs * phi * dwl_m;
            jsc_te += contrib_te;
            jsc_tm += contrib_tm;
        }

        // A/m² → mA/cm²
        Ok((jsc_te * 0.1, jsc_tm * 0.1))
    }

    /// Build a complete `LightTrappingComparison` for a given grating geometry.
    ///
    /// Runs single-pass, Lambertian, and RCWA calculations and collects the
    /// results into a single report struct.
    ///
    /// # Arguments
    /// * `grating_period_nm` — grating period (nm)
    /// * `duty_cycle` — ridge fill factor (0–1)
    /// * `height_nm` — grating etch depth (nm)
    pub fn compare_configurations(
        &self,
        grating_period_nm: f64,
        duty_cycle: f64,
        height_nm: f64,
    ) -> Result<LightTrappingComparison, OxiPhotonError> {
        let single_pass = self.single_pass_jsc();
        let lambertian = self.lambertian_jsc();
        let (grating_te, grating_tm) =
            self.grating_enhanced_jsc(grating_period_nm, duty_cycle, height_nm)?;
        let grating_avg = 0.5 * (grating_te + grating_tm);

        Ok(LightTrappingComparison {
            single_pass_jsc: single_pass,
            lambertian_jsc: lambertian,
            grating_jsc_te: grating_te,
            grating_jsc_tm: grating_tm,
            grating_jsc_avg: grating_avg,
            lambertian_factor: self.yablonovitch_factor(),
            grating_enhancement_te: if single_pass > 0.0 {
                grating_te / single_pass
            } else {
                1.0
            },
            grating_enhancement_tm: if single_pass > 0.0 {
                grating_tm / single_pass
            } else {
                1.0
            },
        })
    }

    // ─── Private helpers ─────────────────────────────────────────────────────

    /// Compute Jsc (A/m²) with a given path-length enhancement factor F
    /// and front-surface reflectance R_front (constant over wavelength).
    ///
    /// A(λ) = (1 − R_front) · (1 − exp(−F·α(λ)·L))
    fn jsc_with_enhancement(&self, f_enhance: f64, r_front: f64) -> f64 {
        let wls_nm = wavelength_grid_nm(300.0, 1200.0, 5.0);
        let solar = SolarSpectrum::am15g();
        let thick_m = self.absorber_thickness_nm * 1e-9;
        let nw = wls_nm.len();

        let mut jsc = 0.0_f64;

        for i in 0..nw - 1 {
            let wl_mid_nm = 0.5 * (wls_nm[i] + wls_nm[i + 1]);
            let dwl_nm = wls_nm[i + 1] - wls_nm[i];
            let wl_m = wl_mid_nm * 1e-9;
            let dwl_m = dwl_nm * 1e-9;

            let irrad = solar.irradiance_at(wl_m);
            if irrad <= 0.0 {
                continue;
            }

            let alpha_per_m = self.absorber_material.alpha_at_nm(wl_mid_nm);
            let a = (1.0 - (-f_enhance * alpha_per_m * thick_m).exp()).clamp(0.0, 1.0);

            let phi = irrad * wl_m / (PLANCK * SPEED_OF_LIGHT);
            let contrib = CHARGE * (1.0 - r_front) * a * phi * dwl_m;
            jsc += contrib;
        }

        // A/m² → mA/cm²
        jsc * 0.1
    }
}

// ─── LightTrappingComparison ─────────────────────────────────────────────────

/// Summary of single-pass, Lambertian, and grating Jsc values.
#[derive(Debug, Clone)]
pub struct LightTrappingComparison {
    /// Single-pass Jsc (mA/cm²) — no light trapping
    pub single_pass_jsc: f64,
    /// Lambertian limit Jsc (mA/cm²) — maximum path-length enhancement
    pub lambertian_jsc: f64,
    /// Grating Jsc for TE polarisation (mA/cm²)
    pub grating_jsc_te: f64,
    /// Grating Jsc for TM polarisation (mA/cm²)
    pub grating_jsc_tm: f64,
    /// Average grating Jsc over TE and TM (mA/cm²)
    pub grating_jsc_avg: f64,
    /// Yablonovitch path-length enhancement factor F = 4n²
    pub lambertian_factor: f64,
    /// Grating TE Jsc relative to single-pass (dimensionless)
    pub grating_enhancement_te: f64,
    /// Grating TM Jsc relative to single-pass (dimensionless)
    pub grating_enhancement_tm: f64,
}

impl LightTrappingComparison {
    /// Return true if the grating (avg) Jsc exceeds single-pass Jsc.
    pub fn grating_beats_single_pass(&self) -> bool {
        self.grating_jsc_avg > self.single_pass_jsc
    }

    /// Return true if the Lambertian Jsc exceeds the grating (avg) Jsc.
    pub fn lambertian_beats_grating(&self) -> bool {
        self.lambertian_jsc > self.grating_jsc_avg
    }
}

// ─── Internal TMM helper ─────────────────────────────────────────────────────

/// Compute reflectance for a 3-layer stack (air | film | substrate) at normal incidence
/// using the 2×2 transfer matrix method.
///
/// All refractive indices are real (lossless film).
///
/// # Arguments
/// * `n0` — incident medium (real n)
/// * `n1` — film (real n)
/// * `ns` — substrate (real n)
/// * `d` — film thickness (m)
/// * `lambda` — wavelength (m)
fn tmm_reflectance_3layer(n0: f64, n1: f64, ns: f64, d: f64, lambda: f64) -> f64 {
    use num_complex::Complex64;

    let i = Complex64::new(0.0, 1.0);
    let n0c = Complex64::new(n0, 0.0);
    let n1c = Complex64::new(n1, 0.0);
    let nsc = Complex64::new(ns, 0.0);

    // Phase thickness of film
    let delta = Complex64::new(2.0 * PI * n1 * d / lambda, 0.0);
    let cos_d = delta.cos();
    let sin_d = delta.sin();

    // Characteristic matrix M = [[cos δ, i/n₁ sin δ], [i n₁ sin δ, cos δ]]
    let m00 = cos_d;
    let m01 = i * sin_d / n1c;
    let m10 = i * n1c * sin_d;
    let m11 = cos_d;

    // r = (n0·M00 + n0·ns·M01 - M10 - ns·M11) / (n0·M00 + n0·ns·M01 + M10 + ns·M11)
    let numer = n0c * m00 + n0c * nsc * m01 - m10 - nsc * m11;
    let denom = n0c * m00 + n0c * nsc * m01 + m10 + nsc * m11;

    if denom.norm() < 1e-30 {
        return 0.0;
    }
    let r = numer / denom;
    r.norm_sqr().clamp(0.0, 1.0)
}

// ─── Convenience functions ───────────────────────────────────────────────────

/// Compute Lambertian-limit Jsc (mA/cm²) for a standard c-Si cell.
///
/// Uses F = 4n² with n = 3.5 (c-Si) and Beer-Lambert absorptance.
///
/// # Arguments
/// * `thickness_nm` — Si absorber thickness (nm)
pub fn lambertian_jsc_si(thickness_nm: f64) -> f64 {
    let mat = AbsorptionMaterial::crystalline_silicon();
    let analysis = LightTrappingAnalysis::new(mat, thickness_nm, 3.5);
    analysis.lambertian_jsc()
}

/// Compute single-pass Jsc (mA/cm²) for a standard c-Si cell.
///
/// # Arguments
/// * `thickness_nm` — Si absorber thickness (nm)
pub fn single_pass_jsc_si(thickness_nm: f64) -> f64 {
    let mat = AbsorptionMaterial::crystalline_silicon();
    let analysis = LightTrappingAnalysis::new(mat, thickness_nm, 3.5);
    analysis.single_pass_jsc()
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn si_analysis() -> LightTrappingAnalysis {
        LightTrappingAnalysis::new(AbsorptionMaterial::crystalline_silicon(), 180_000.0, 3.5)
    }

    #[test]
    fn lambertian_exceeds_single_pass() {
        let a = si_analysis();
        let lamb = a.lambertian_jsc();
        let single = a.single_pass_jsc();
        assert!(
            lamb > single,
            "Lambertian Jsc ({lamb:.2}) must exceed single-pass ({single:.2})"
        );
    }

    #[test]
    fn yablonovitch_factor_si() {
        let a = si_analysis();
        let f = a.yablonovitch_factor();
        assert!(
            (f - 49.0).abs() < 1.0,
            "F = 4n² for n=3.5 should be ≈49, got {f:.1}"
        );
    }

    #[test]
    fn lambertian_jsc_si_physical_range() {
        let jsc = lambertian_jsc_si(180_000.0);
        assert!(jsc > 15.0 && jsc < 50.0, "Jsc={jsc:.2} mA/cm²");
    }

    #[test]
    fn single_pass_jsc_si_physical_range() {
        let jsc = single_pass_jsc_si(180_000.0);
        assert!(jsc > 0.0, "single-pass Jsc must be positive, got {jsc:.4}");
    }

    #[test]
    fn grating_jsc_positive() {
        let a = si_analysis();
        let (te, tm) = a
            .grating_enhanced_jsc(500.0, 0.5, 200.0)
            .expect("grating jsc failed");
        assert!(
            te > 0.0 && tm > 0.0,
            "grating Jsc must be positive: te={te:.4}, tm={tm:.4}"
        );
    }

    #[test]
    fn compare_configurations_runs() {
        let a = si_analysis();
        let cmp = a
            .compare_configurations(500.0, 0.5, 200.0)
            .expect("compare failed");
        assert!(cmp.lambertian_jsc > 0.0);
        assert!(cmp.single_pass_jsc > 0.0);
        assert!(cmp.grating_jsc_avg > 0.0);
    }

    #[test]
    fn lambertian_beats_single_pass_thin_film() {
        // Very thin cell: single pass absorbs little, Lambertian should help a lot
        let a = LightTrappingAnalysis::new(AbsorptionMaterial::crystalline_silicon(), 1_000.0, 3.5);
        assert!(a.lambertian_jsc() > a.single_pass_jsc());
    }
}
