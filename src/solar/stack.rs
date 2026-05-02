//! TMM-based full-stack solar cell optical model.
//!
//! Implements a multilayer solar cell stack with:
//! - Complex (lossy) refractive indices via the transfer matrix method
//! - AM1.5G-integrated short-circuit current density (Jsc)
//! - ARC thickness optimization for maximum Jsc
//!
//! The transfer matrix is computed per-wavelength to handle dispersive materials.
//! For crystalline Si, the wavelength-dependent extinction coefficient k(λ) is
//! derived from the tabulated α(λ) via: k = α·λ/(4π).

use num_complex::Complex64;
use std::f64::consts::PI;

use crate::error::OxiPhotonError;
use crate::solar::absorption::AbsorptionMaterial;
use crate::solar::spectrum::SolarSpectrum;

// ─── Physical constants ─────────────────────────────────────────────────────

const CHARGE: f64 = 1.602_176_634e-19; // C
const PLANCK: f64 = 6.626_070_15e-34; // J·s
const SPEED_OF_LIGHT: f64 = 2.997_924_58e8; // m/s

// ─── StackLayer ─────────────────────────────────────────────────────────────

/// Material kind for a stack layer — either fixed optical constants or dispersive.
#[derive(Debug, Clone)]
enum LayerMaterial {
    /// Wavelength-independent (n, k) pair.
    Constant { n: f64, k: f64 },
    /// Wavelength-dependent via AbsorptionMaterial alpha table.
    Absorber(AbsorptionMaterial),
}

/// A single layer in the solar cell optical stack.
#[derive(Debug, Clone)]
pub struct StackLayer {
    /// Refractive index (real part) at 1000 nm reference
    pub n: f64,
    /// Extinction coefficient k (0 for transparent layers)
    pub k: f64,
    /// Thickness in nm (0 for semi-infinite media)
    pub thickness_nm: f64,
    /// Material label for reporting
    pub name: &'static str,
    /// Internal material description for wavelength-dependent k
    material: LayerMaterial,
}

impl StackLayer {
    /// Compute complex refractive index ñ = n - ik at wavelength_nm.
    ///
    /// For absorbing materials derived from an `AbsorptionMaterial`, k is
    /// computed from the tabulated α via k = α(λ)·λ/(4π).
    pub fn n_complex(&self, wavelength_nm: f64) -> (f64, f64) {
        match &self.material {
            LayerMaterial::Constant { n, k } => (*n, *k),
            LayerMaterial::Absorber(mat) => {
                let alpha_per_m = mat.alpha_at_nm(wavelength_nm);
                let lambda_m = wavelength_nm * 1e-9;
                let k = alpha_per_m * lambda_m / (4.0 * PI);
                (self.n, k)
            }
        }
    }

    /// Preset: air (semi-infinite entrance or exit medium).
    pub fn air() -> Self {
        Self {
            n: 1.0,
            k: 0.0,
            thickness_nm: 0.0,
            name: "Air",
            material: LayerMaterial::Constant { n: 1.0, k: 0.0 },
        }
    }

    /// Preset: SiNx ARC (n≈2.0, lossless in visible/NIR).
    pub fn sinx(thickness_nm: f64) -> Self {
        Self {
            n: 2.0,
            k: 0.0,
            thickness_nm,
            name: "SiNx",
            material: LayerMaterial::Constant { n: 2.0, k: 0.0 },
        }
    }

    /// Preset: crystalline silicon absorber with wavelength-dependent absorption.
    ///
    /// The extinction coefficient k(λ) is derived from the AbsorptionMaterial
    /// alpha table for c-Si.
    pub fn c_si(thickness_nm: f64) -> Self {
        Self {
            n: 3.5,
            k: 0.0, // nominal; actual k comes from dispersive alpha
            thickness_nm,
            name: "c-Si",
            material: LayerMaterial::Absorber(AbsorptionMaterial::crystalline_silicon()),
        }
    }

    /// Preset: Al back reflector (metal, high k, semi-infinite).
    pub fn al_reflector() -> Self {
        Self {
            n: 1.0,
            k: 6.0,
            thickness_nm: 0.0,
            name: "Al",
            material: LayerMaterial::Constant { n: 1.0, k: 6.0 },
        }
    }
}

// ─── SolarCellStack ─────────────────────────────────────────────────────────

/// A multilayer solar cell optical stack.
pub struct SolarCellStack {
    /// Layers from front (illuminated side) to back.
    /// First and last layers are semi-infinite (thickness_nm ignored).
    pub layers: Vec<StackLayer>,
}

impl SolarCellStack {
    /// Standard c-Si cell: Air | SiNx ARC (arc_nm) | c-Si (absorber_um µm) | Al back reflector.
    ///
    /// # Arguments
    /// * `arc_nm` — SiNx ARC thickness in nm
    /// * `absorber_um` — c-Si absorber thickness in µm
    pub fn c_si_standard(arc_nm: f64, absorber_um: f64) -> Self {
        Self {
            layers: vec![
                StackLayer::air(),
                StackLayer::sinx(arc_nm),
                StackLayer::c_si(absorber_um * 1000.0), // µm → nm
                StackLayer::al_reflector(),
            ],
        }
    }

    /// Compute spectrally-resolved reflectance R(λ), transmittance T(λ), absorptance A(λ)
    /// using the 2×2 transfer matrix method for normal incidence.
    ///
    /// The first and last layers are treated as semi-infinite media.
    /// Interior layers (indices 1 … N-2) have finite thickness.
    ///
    /// Returns `Vec<(R, T, A)>` where A = 1 - R - T.
    pub fn optical_response(
        &self,
        wavelengths_nm: &[f64],
    ) -> Result<Vec<(f64, f64, f64)>, OxiPhotonError> {
        if self.layers.len() < 2 {
            return Err(OxiPhotonError::InvalidLayer(
                "Stack must have at least 2 layers (entrance + exit)".into(),
            ));
        }

        let results = wavelengths_nm
            .iter()
            .map(|&wl_nm| compute_tmm_normal(&self.layers, wl_nm))
            .collect();
        Ok(results)
    }

    /// Short-circuit current density Jsc (mA/cm²) integrated over AM1.5G spectrum.
    ///
    /// Uses TMM to obtain the reflectance R(λ) at each wavelength, then computes
    /// the Si-absorbed fraction using a Beer-Lambert double-pass model.
    /// Assumes IQE = 1 (all photons absorbed in Si generate carriers).
    ///
    /// # Arguments
    /// * `wavelengths_nm` — wavelength grid (nm) for integration
    pub fn jsc_am15g(&self, wavelengths_nm: &[f64]) -> Result<f64, OxiPhotonError> {
        if wavelengths_nm.len() < 2 {
            return Err(OxiPhotonError::NumericalError(
                "Need at least 2 wavelength points for integration".into(),
            ));
        }

        // Build AM1.5G spectrum for interpolation
        let solar = SolarSpectrum::am15g();

        // Locate the Si absorber layer (first layer with an Absorber material)
        let si_layer_opt = self
            .layers
            .iter()
            .find(|l| matches!(l.material, LayerMaterial::Absorber(_)));

        // Compute TMM reflectance at each wavelength
        let optical = self.optical_response(wavelengths_nm)?;

        let mut jsc_sum = 0.0_f64;
        let n = wavelengths_nm.len();

        for i in 0..n - 1 {
            let wl0_nm = wavelengths_nm[i];
            let wl1_nm = wavelengths_nm[i + 1];
            let wl_mid_nm = 0.5 * (wl0_nm + wl1_nm);
            let dwl_nm = wl1_nm - wl0_nm;
            let dwl_m = dwl_nm * 1e-9;
            let wl_m = wl_mid_nm * 1e-9;

            // R at midpoint (average of adjacent values)
            let r_mid = 0.5 * (optical[i].0 + optical[i + 1].0);

            // AM1.5G irradiance at midpoint (W/m²/m)
            let irrad = solar.irradiance_at(wl_m);

            // Absorbed fraction in Si layer
            let a_si = match si_layer_opt {
                Some(si_layer) => {
                    let thick_m = si_layer.thickness_nm * 1e-9;
                    let (_, k_si) = si_layer.n_complex(wl_mid_nm);
                    let alpha = 4.0 * PI * k_si / wl_m;
                    // Double-pass Beer-Lambert (back reflector)
                    1.0 - (-2.0 * alpha * thick_m).exp()
                }
                None => optical[i].2, // Fall back to total absorptance
            };

            // Photon flux density (photons/m²/s/m) = I(λ) · λ / (h·c)
            let phi = irrad * wl_m / (PLANCK * SPEED_OF_LIGHT);

            // Jsc contribution: q · (1-R) · A_Si · Φ(λ) · dλ
            let contrib = CHARGE * (1.0 - r_mid) * a_si * phi * dwl_m;
            jsc_sum += contrib;
        }

        // Convert A/m² → mA/cm²: × 1000 (mA/A) / 10000 (m²/cm²) = × 0.1
        Ok(jsc_sum * 0.1)
    }

    /// Optimize the ARC layer thickness to maximize Jsc using a linear scan.
    ///
    /// # Arguments
    /// * `layer_idx` — index of the ARC layer in `self.layers`
    /// * `thickness_range_nm` — (min, max) search range in nm
    /// * `n_steps` — number of thickness values to evaluate
    ///
    /// Returns `(optimal_thickness_nm, optimal_jsc_mA_cm2)`.
    pub fn optimize_arc_thickness(
        &self,
        layer_idx: usize,
        thickness_range_nm: (f64, f64),
        n_steps: usize,
    ) -> Result<(f64, f64), OxiPhotonError> {
        if layer_idx >= self.layers.len() {
            return Err(OxiPhotonError::InvalidLayer(format!(
                "Layer index {layer_idx} out of bounds (stack has {} layers)",
                self.layers.len()
            )));
        }
        if n_steps < 2 {
            return Err(OxiPhotonError::NumericalError(
                "n_steps must be at least 2".into(),
            ));
        }

        let (t_min, t_max) = thickness_range_nm;
        let step = (t_max - t_min) / (n_steps - 1) as f64;

        // Default integration wavelength grid: 300-1200 nm at 5 nm resolution
        let wls: Vec<f64> = (0..=180).map(|i| 300.0 + i as f64 * 5.0).collect();

        let mut best_t = t_min;
        let mut best_jsc = f64::NEG_INFINITY;

        for step_i in 0..n_steps {
            let thickness = t_min + step_i as f64 * step;

            // Build a modified stack with the new thickness
            let mut layers = self.layers.clone();
            layers[layer_idx].thickness_nm = thickness;
            let stack = SolarCellStack { layers };

            let jsc = stack.jsc_am15g(&wls)?;
            if jsc > best_jsc {
                best_jsc = jsc;
                best_t = thickness;
            }
        }

        Ok((best_t, best_jsc))
    }
}

// ─── Internal TMM computation ────────────────────────────────────────────────

/// Compute (R, T, A) using the 2×2 characteristic matrix method at normal incidence.
///
/// The standard Hecht/Born-Wolf characteristic matrix for each layer j:
///
/// ```text
/// M_j = [[cos(δ_j),      i/ñ_j · sin(δ_j)],
///         [i·ñ_j·sin(δ_j), cos(δ_j)       ]]
/// ```
///
/// where δ_j = 2π ñ_j d_j / λ (complex phase thickness).
///
/// For a stack with N layers:
///   - Layers 0 and N-1 are semi-infinite (only their n̂ values matter)
///   - Layers 1 … N-2 are finite (contribute propagation matrices)
///
/// Numerator/denominator of r and t:
/// ```text
/// denom = (ñ₀·M[0,0] + ñ₀·ñ_N·M[0,1] + M[1,0] + ñ_N·M[1,1])
/// r     = (ñ₀·M[0,0] + ñ₀·ñ_N·M[0,1] - M[1,0] - ñ_N·M[1,1]) / denom
/// t     = 2·ñ₀ / denom
/// ```
fn compute_tmm_normal(layers: &[StackLayer], wavelength_nm: f64) -> (f64, f64, f64) {
    let lambda_m = wavelength_nm * 1e-9;

    // Entrance medium (semi-infinite)
    let (n0_r, n0_i) = layers[0].n_complex(wavelength_nm);
    let n0 = Complex64::new(n0_r, -n0_i); // ñ = n - ik (physics sign convention)

    // Exit medium (semi-infinite)
    let last = layers.len() - 1;
    let (ns_r, ns_i) = layers[last].n_complex(wavelength_nm);
    let ns = Complex64::new(ns_r, -ns_i);

    // Start with identity matrix M = I
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let i = Complex64::new(0.0, 1.0);

    // M = [[m00, m01], [m10, m11]]
    let mut m00 = one;
    let mut m01 = zero;
    let mut m10 = zero;
    let mut m11 = one;

    // Multiply in each interior layer (indices 1 .. last-1)
    for layer in layers.iter().take(last).skip(1) {
        let d_m = layer.thickness_nm * 1e-9;
        let (nj_r, nj_i) = layer.n_complex(wavelength_nm);
        let nj = Complex64::new(nj_r, -nj_i); // ñ_j = n_j - i·k_j

        // Complex phase: δ_j = 2π ñ_j d_j / λ
        let delta = Complex64::new(2.0 * PI, 0.0) * nj * d_m / lambda_m;

        let cos_d = delta.cos();
        let sin_d = delta.sin();

        // Layer matrix M_j
        let lm00 = cos_d;
        let lm01 = i * sin_d / nj;
        let lm10 = i * nj * sin_d;
        let lm11 = cos_d;

        // M = M · M_j
        let new_m00 = m00 * lm00 + m01 * lm10;
        let new_m01 = m00 * lm01 + m01 * lm11;
        let new_m10 = m10 * lm00 + m11 * lm10;
        let new_m11 = m10 * lm01 + m11 * lm11;
        m00 = new_m00;
        m01 = new_m01;
        m10 = new_m10;
        m11 = new_m11;
    }

    // Compute amplitude reflection and transmission
    // denom = ñ₀·M[0,0] + ñ₀·ñ_N·M[0,1] + M[1,0] + ñ_N·M[1,1]
    let denom = n0 * m00 + n0 * ns * m01 + m10 + ns * m11;

    if denom.norm() < 1e-30 {
        return (0.0, 0.0, 0.0);
    }

    // r_amp = (ñ₀·M[0,0] + ñ₀·ñ_N·M[0,1] - M[1,0] - ñ_N·M[1,1]) / denom
    let numer_r = n0 * m00 + n0 * ns * m01 - m10 - ns * m11;
    let r_amp = numer_r / denom;
    let r_norm_sq = r_amp.norm_sqr();
    // Guard against NaN/Inf from numerically ill-conditioned metallic substrates
    let reflectance = if r_norm_sq.is_finite() {
        r_norm_sq.clamp(0.0, 1.0)
    } else {
        1.0 // treat as fully reflective (metallic limit)
    };

    // t_amp = 2·ñ₀ / denom
    let t_amp = (Complex64::new(2.0, 0.0) * n0) / denom;
    // T = Re(ñ_N)/Re(ñ₀) · |t|²  (energy flux ratio for absorbing media)
    let transmittance = if n0.re > 1e-10 {
        let t_raw = ns.re / n0.re * t_amp.norm_sqr();
        if t_raw.is_finite() {
            t_raw.clamp(0.0, 1.0 - reflectance)
        } else {
            0.0
        }
    } else {
        0.0
    };

    let absorptance = (1.0 - reflectance - transmittance).max(0.0);
    (reflectance, transmittance, absorptance)
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn air_glass_normal_incidence_r_near_4_percent() {
        let layers = vec![
            StackLayer::air(),
            StackLayer {
                n: 1.5,
                k: 0.0,
                thickness_nm: 0.0,
                name: "Glass",
                material: LayerMaterial::Constant { n: 1.5, k: 0.0 },
            },
        ];
        let (r, _t, a) = compute_tmm_normal(&layers, 550.0);
        assert!((r - 0.04).abs() < 0.005, "Expected R≈4%, got {:.4}", r);
        assert!(a < 0.001, "Expected A≈0 for lossless, got {:.4}", a);
    }

    #[test]
    fn sinx_arc_reduces_reflection() {
        let bare = vec![
            StackLayer::air(),
            StackLayer {
                n: 3.5,
                k: 0.0,
                thickness_nm: 0.0,
                name: "Si",
                material: LayerMaterial::Constant { n: 3.5, k: 0.0 },
            },
        ];
        let with_arc = vec![
            StackLayer::air(),
            StackLayer::sinx(80.0),
            StackLayer {
                n: 3.5,
                k: 0.0,
                thickness_nm: 0.0,
                name: "Si",
                material: LayerMaterial::Constant { n: 3.5, k: 0.0 },
            },
        ];
        let (r_bare, _, _) = compute_tmm_normal(&bare, 550.0);
        let (r_arc, _, _) = compute_tmm_normal(&with_arc, 550.0);
        assert!(
            r_arc < r_bare,
            "ARC should reduce reflectance: bare={r_bare:.3}, arc={r_arc:.3}"
        );
    }

    #[test]
    fn c_si_stack_jsc_in_physical_range() {
        let stack = SolarCellStack::c_si_standard(80.0, 300.0);
        let wls: Vec<f64> = (300..=1200).map(|w| w as f64).collect();
        let jsc = stack.jsc_am15g(&wls).expect("Jsc computation failed");
        assert!(
            (10.0..=50.0).contains(&jsc),
            "Expected Jsc in 10-50 mA/cm², got {jsc:.2}"
        );
    }

    #[test]
    fn layer_k_dispersive_at_short_wavelength() {
        let si = StackLayer::c_si(100.0);
        let (_, k_400) = si.n_complex(400.0);
        let (_, k_1000) = si.n_complex(1000.0);
        // k should be larger at 400 nm (more absorbing) than at 1000 nm
        assert!(
            k_400 > k_1000,
            "k(400nm)={k_400:.4} should exceed k(1000nm)={k_1000:.4}"
        );
    }
}
