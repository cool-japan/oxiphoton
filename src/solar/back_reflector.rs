//! AR coating + back-reflector design optimised for maximum AM1.5G short-circuit current.
//!
//! ## Physical model
//!
//! Stack layout (front to back):
//! ```text
//! air (semi-infinite)
//!  → AR layer  (n_ar,  k=0,    d_ar)
//!  → absorber  (n_abs, k_abs,  d_abs)
//!  → back reflector (n_br, k_br, d_br)
//! air (semi-infinite substrate)
//! ```
//!
//! The transfer-matrix method (TMM) is applied at normal incidence for each
//! wavelength.  The total absorptance in the stack is approximated as
//!
//! ```text
//! A_stack(λ) = 1 − R(λ) − T(λ)
//! ```
//!
//! Under the **EQE = A ideal-collection** assumption every photon absorbed
//! anywhere in the stack is treated as contributing to the photocurrent.
//! For a thick metallic back reflector T ≈ 0 and the dominant loss channel
//! is front reflection, so `A_stack ≈ A_absorber` to an excellent approximation
//! in the visible/NIR.
//!
//! ## Short-circuit current
//!
//! J_sc \[mA/cm²\] is calculated from the AM1.5G photon flux Φ(λ) as
//!
//! ```text
//! J_sc = q · ∫ A(λ) · Φ(λ) dλ   (trapezoidal rule)
//! ```
//!
//! where Φ(λ) is the spectral photon flux density in photons·m⁻²·s⁻¹·m⁻¹.
//!
//! The result is converted from A/m² to mA/cm² by multiplying by 0.1.

use num_complex::Complex64;
use std::f64::consts::PI;

use crate::error::OxiPhotonError;
use crate::solar::SolarSpectrum;

// ─── Physical constants ──────────────────────────────────────────────────────

const ELECTRON_CHARGE: f64 = 1.602_176_634e-19; // C
const PLANCK_H: f64 = 6.626_070_15e-34; // J·s
const SPEED_OF_LIGHT: f64 = 2.997_924_58e8; // m/s

// ─── Result type ─────────────────────────────────────────────────────────────

/// Result of a single AR + back-reflector design evaluation.
///
/// Contains the optimal (or evaluated) parameters together with the
/// computed absorption spectrum and integrated short-circuit current.
#[derive(Debug, Clone)]
pub struct ArBackReflectorDesign {
    /// AR layer refractive index (real, lossless).
    pub ar_n: f64,
    /// AR layer physical thickness (nm).
    pub ar_thickness_nm: f64,
    /// Integrated short-circuit current density (mA/cm²).
    ///
    /// Computed as J_sc = q·∫A(λ)·Φ(λ)dλ with Φ from AM1.5G.
    pub jsc_ma_cm2: f64,
    /// Wavelength-averaged absorptance A(λ) over the evaluated range.
    pub mean_absorption: f64,
    /// Wavelength grid used for the evaluation (nm).
    pub wavelengths_nm: Vec<f64>,
    /// Absorptance spectrum A(λ) at each wavelength in `wavelengths_nm`.
    ///
    /// Computed as 1 − R − T for the full stack; under the EQE=A ideal
    /// assumption this equals the external quantum efficiency.
    pub absorption_spectrum: Vec<f64>,
}

// ─── Design parameters ────────────────────────────────────────────────────────

/// Parameters for a single AR + back-reflector stack evaluation.
///
/// Groups all optical constants and thicknesses so that `evaluate_design`
/// does not exceed the clippy `too_many_arguments` limit.
#[derive(Debug, Clone)]
pub struct DesignParams {
    /// AR layer refractive index (real, lossless).
    pub ar_n: f64,
    /// AR layer physical thickness (m).
    pub ar_d_m: f64,
    /// Absorber real refractive index.
    pub absorber_n: f64,
    /// Absorber extinction coefficient.
    pub absorber_k: f64,
    /// Absorber physical thickness (m).
    pub absorber_thickness_m: f64,
    /// Back-reflector real refractive index.
    pub back_n: f64,
    /// Back-reflector extinction coefficient.
    pub back_k: f64,
    /// Back-reflector physical thickness (m).
    pub back_thickness_m: f64,
}

// ─── Single design evaluation ────────────────────────────────────────────────

/// Evaluate a single AR + back-reflector design at fixed optical constants.
///
/// Uses the transfer-matrix method (2×2 characteristic matrix, normal
/// incidence) to compute the spectral absorptance in the stack at each
/// wavelength and then integrates against the supplied photon flux to obtain
/// the short-circuit current.
///
/// ## Stack layout
///
/// ```text
/// air (n=1, semi-infinite) → AR (params.ar_n, k=0, d=params.ar_d_m)
///   → absorber (params.absorber_n+i·params.absorber_k, d=params.absorber_thickness_m)
///   → back reflector (params.back_n+i·params.back_k, d=params.back_thickness_m)
/// air (n=1, semi-infinite substrate)
/// ```
///
/// ## Units
///
/// All thickness fields in `params` must be in **metres**.
/// `wavelengths_nm` must be in **nanometres**.
/// `photon_flux` must be in **photons·m⁻²·s⁻¹·m⁻¹** (spectral photon-flux
/// density per unit wavelength in metres), matching the output of
/// [`SolarSpectrum::photon_flux`].
///
/// ## EQE = A ideal assumption
///
/// The absorptance A(λ) = 1 − R(λ) − T(λ) is used directly as EQE.  This
/// is a physically rigorous upper bound assuming ideal carrier collection.
///
/// # Errors
///
/// Returns [`OxiPhotonError::InvalidLayer`] if fewer than 2 wavelength points
/// are supplied.
pub fn evaluate_design(
    params: &DesignParams,
    wavelengths_nm: &[f64],
    photon_flux: &[f64],
) -> Result<ArBackReflectorDesign, OxiPhotonError> {
    if wavelengths_nm.len() < 2 {
        return Err(OxiPhotonError::InvalidLayer(
            "evaluate_design requires at least 2 wavelength points".into(),
        ));
    }
    if wavelengths_nm.len() != photon_flux.len() {
        return Err(OxiPhotonError::InvalidLayer(format!(
            "wavelengths_nm ({}) and photon_flux ({}) must have the same length",
            wavelengths_nm.len(),
            photon_flux.len()
        )));
    }

    let n_wl = wavelengths_nm.len();
    let mut absorption_spectrum = Vec::with_capacity(n_wl);

    let stack = TmmStack {
        n0: 1.0,
        k0: 0.0,
        n1: params.ar_n,
        k1: 0.0,
        d1_m: params.ar_d_m,
        n2: params.absorber_n,
        k2: params.absorber_k,
        d2_m: params.absorber_thickness_m,
        n3: params.back_n,
        k3: params.back_k,
        d3_m: params.back_thickness_m,
        n4: 1.0,
        k4: 0.0,
    };

    for &wl_nm in wavelengths_nm {
        let wl_m = wl_nm * 1e-9;
        let a = tmm_stack_absorptance(&stack, wl_m);
        absorption_spectrum.push(a);
    }

    // Trapezoidal integration: J_sc [A/m²] = q · Σ A_mid · Φ_mid · dλ_m
    let mut jsc_a_m2 = 0.0_f64;
    for i in 0..n_wl - 1 {
        let a_mid = 0.5 * (absorption_spectrum[i] + absorption_spectrum[i + 1]);
        let phi_mid = 0.5 * (photon_flux[i] + photon_flux[i + 1]);
        let dwl_nm = wavelengths_nm[i + 1] - wavelengths_nm[i];
        let dwl_m = dwl_nm * 1e-9;
        jsc_a_m2 += ELECTRON_CHARGE * a_mid * phi_mid * dwl_m;
    }

    // Convert A/m² → mA/cm² (×0.1)
    let jsc_ma_cm2 = jsc_a_m2 * 0.1;

    let mean_absorption = if n_wl > 0 {
        absorption_spectrum.iter().sum::<f64>() / n_wl as f64
    } else {
        0.0
    };

    Ok(ArBackReflectorDesign {
        ar_n: params.ar_n,
        ar_thickness_nm: params.ar_d_m * 1e9,
        jsc_ma_cm2,
        mean_absorption,
        wavelengths_nm: wavelengths_nm.to_vec(),
        absorption_spectrum,
    })
}

// ─── Grid-search optimiser ───────────────────────────────────────────────────

/// Optimise AR coating + back-reflector for maximum AM1.5G J_sc.
///
/// Performs a dense grid search over AR refractive index and thickness:
///
/// | Parameter    | Range          | Step    | Points |
/// |-------------|----------------|---------|--------|
/// | `n_ar`      | 1.20 – 2.50    | 0.05    | 27     |
/// | `d_ar` (nm) | 50 – 200       | 5       | 31     |
///
/// The integration wavelength grid is 300 – 1200 nm in 10 nm steps (91
/// points), which gives ≈ 27 × 31 × 91 ≈ 76 000 TMM evaluations.
///
/// ## Arguments
///
/// * `absorber_n` — real part of the absorber refractive index.
/// * `absorber_k` — extinction coefficient of the absorber.
/// * `absorber_thickness_m` — absorber thickness (m).
/// * `back_n` — real part of the back-reflector refractive index.
/// * `back_k` — extinction coefficient of the back reflector.
/// * `back_thickness_m` — back-reflector physical thickness (m).
/// * `am15g` — AM1.5G solar spectrum used to build the photon-flux weighting.
///
/// ## Returns
///
/// The [`ArBackReflectorDesign`] at the grid point with the highest J_sc.
///
/// # Errors
///
/// Returns [`OxiPhotonError::NumericalError`] if the optimiser fails to find
/// any finite J_sc value (should not occur under normal circumstances).
pub fn optimize_ar_and_back_reflector(
    absorber_n: f64,
    absorber_k: f64,
    absorber_thickness_m: f64,
    back_n: f64,
    back_k: f64,
    back_thickness_m: f64,
    am15g: &SolarSpectrum,
) -> Result<ArBackReflectorDesign, OxiPhotonError> {
    // Wavelength grid: 300–1200 nm, 10 nm step → 91 points
    let wavelengths_nm: Vec<f64> = (0..=90).map(|i| 300.0 + i as f64 * 10.0).collect();

    // Build photon-flux array [photons·m⁻²·s⁻¹·m⁻¹] once
    let photon_flux: Vec<f64> = wavelengths_nm
        .iter()
        .map(|&wl_nm| {
            let wl_m = wl_nm * 1e-9;
            let irrad = am15g.irradiance_at(wl_m); // W/m²/m
                                                   // Φ(λ) = E(λ) · λ / (h·c)
            irrad * wl_m / (PLANCK_H * SPEED_OF_LIGHT)
        })
        .collect();

    // Grid: n_ar 1.20 → 2.50 (step 0.05, 27 values)
    //       d_ar 50 → 200 nm (step 5 nm, 31 values)
    let n_ar_values: Vec<f64> = (0..27).map(|i| 1.20 + i as f64 * 0.05).collect();
    let d_ar_nm_values: Vec<f64> = (0..31).map(|i| 50.0 + i as f64 * 5.0).collect();

    let mut best_jsc = f64::NEG_INFINITY;
    let mut best_n_ar = n_ar_values[0];
    let mut best_d_ar_nm = d_ar_nm_values[0];

    for &n_ar in &n_ar_values {
        for &d_ar_nm in &d_ar_nm_values {
            let d_ar_m = d_ar_nm * 1e-9;

            let params = DesignParams {
                ar_n: n_ar,
                ar_d_m: d_ar_m,
                absorber_n,
                absorber_k,
                absorber_thickness_m,
                back_n,
                back_k,
                back_thickness_m,
            };
            let design = evaluate_design(&params, &wavelengths_nm, &photon_flux)?;

            if design.jsc_ma_cm2 > best_jsc {
                best_jsc = design.jsc_ma_cm2;
                best_n_ar = n_ar;
                best_d_ar_nm = d_ar_nm;
            }
        }
    }

    if !best_jsc.is_finite() {
        return Err(OxiPhotonError::NumericalError(
            "optimize_ar_and_back_reflector: no finite Jsc found during grid search".into(),
        ));
    }

    // Re-evaluate best point to return the full design struct
    let best_d_ar_m = best_d_ar_nm * 1e-9;
    let best_params = DesignParams {
        ar_n: best_n_ar,
        ar_d_m: best_d_ar_m,
        absorber_n,
        absorber_k,
        absorber_thickness_m,
        back_n,
        back_k,
        back_thickness_m,
    };
    evaluate_design(&best_params, &wavelengths_nm, &photon_flux)
}

// ─── Internal TMM ─────────────────────────────────────────────────────────────

/// Parameter bundle for a 5-region TMM stack (normal incidence).
///
/// Layout:
/// 1. entrance medium (semi-infinite): (n0, k0)
/// 2. AR coating (finite):             (n1, k1, d1_m)
/// 3. absorber   (finite):             (n2, k2, d2_m)
/// 4. back reflector (finite):         (n3, k3, d3_m)
/// 5. exit medium (semi-infinite):     (n4, k4)
struct TmmStack {
    n0: f64,
    k0: f64,
    n1: f64,
    k1: f64,
    d1_m: f64,
    n2: f64,
    k2: f64,
    d2_m: f64,
    n3: f64,
    k3: f64,
    d3_m: f64,
    n4: f64,
    k4: f64,
}

/// Compute stack absorptance A = 1 − R − T using the 2×2 characteristic-matrix
/// method for a 4-region stack at normal incidence.
///
/// The characteristic matrix for layer j is:
/// ```text
/// M_j = [[cos(δ_j),      i/ñ_j · sin(δ_j)],
///        [i·ñ_j·sin(δ_j), cos(δ_j)       ]]
/// ```
/// where δ_j = 2π ñ_j d_j / λ and ñ_j = n_j − i·k_j.
///
/// The total matrix M = M_1 · M_2 · M_3.  Then:
/// ```text
/// r = (ñ₀·M₀₀ + ñ₀·ñ₄·M₀₁ − M₁₀ − ñ₄·M₁₁)
///   / (ñ₀·M₀₀ + ñ₀·ñ₄·M₀₁ + M₁₀ + ñ₄·M₁₁)
/// t = 2·ñ₀ / (ñ₀·M₀₀ + ñ₀·ñ₄·M₀₁ + M₁₀ + ñ₄·M₁₁)
/// R = |r|², T = Re(ñ₄)/Re(ñ₀) · |t|²
/// A = 1 − R − T
/// ```
fn tmm_stack_absorptance(stack: &TmmStack, lambda_m: f64) -> f64 {
    // Complex refractive indices (physics sign: ñ = n − i·k for absorbing media)
    let nc0 = Complex64::new(stack.n0, -stack.k0);
    let nc1 = Complex64::new(stack.n1, -stack.k1);
    let nc2 = Complex64::new(stack.n2, -stack.k2);
    let nc3 = Complex64::new(stack.n3, -stack.k3);
    let nc4 = Complex64::new(stack.n4, -stack.k4);

    let im = Complex64::new(0.0, 1.0);

    // Build total characteristic matrix M = M1 · M2 · M3
    let m = {
        let m1 = char_matrix(nc1, stack.d1_m, lambda_m, im);
        let m2 = char_matrix(nc2, stack.d2_m, lambda_m, im);
        let m3 = char_matrix(nc3, stack.d3_m, lambda_m, im);
        mat2_mul(mat2_mul(m1, m2), m3)
    };

    let denom = nc0 * m[0][0] + nc0 * nc4 * m[0][1] + m[1][0] + nc4 * m[1][1];

    if denom.norm() < 1e-30 {
        // Degenerate case (e.g. zero-thickness stack in a limit)
        return 0.0;
    }

    let numer_r = nc0 * m[0][0] + nc0 * nc4 * m[0][1] - m[1][0] - nc4 * m[1][1];
    let r_amp = numer_r / denom;
    let r_sq = r_amp.norm_sqr();
    let reflectance = if r_sq.is_finite() {
        r_sq.clamp(0.0, 1.0)
    } else {
        1.0
    };

    let t_amp = (Complex64::new(2.0, 0.0) * nc0) / denom;
    let transmittance = if nc0.re > 1e-10 {
        let t_raw = nc4.re / nc0.re * t_amp.norm_sqr();
        if t_raw.is_finite() {
            t_raw.clamp(0.0, 1.0 - reflectance)
        } else {
            0.0
        }
    } else {
        0.0
    };

    (1.0 - reflectance - transmittance).max(0.0)
}

/// Build the 2×2 characteristic matrix for a single layer.
///
/// ```text
/// M = [[cos(δ),       i/ñ · sin(δ)],
///      [i·ñ · sin(δ), cos(δ)      ]]
/// ```
/// where δ = 2π ñ d / λ.
fn char_matrix(nc: Complex64, d_m: f64, lambda_m: f64, im: Complex64) -> [[Complex64; 2]; 2] {
    let delta = Complex64::new(2.0 * PI, 0.0) * nc * d_m / lambda_m;
    let cos_d = delta.cos();
    let sin_d = delta.sin();

    [[cos_d, im * sin_d / nc], [im * nc * sin_d, cos_d]]
}

/// Multiply two 2×2 complex matrices.
fn mat2_mul(a: [[Complex64; 2]; 2], b: [[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A layer specification used only in unit tests (n, k, thickness).
    struct LayerSpec {
        n: f64,
        k: f64,
        d_m: f64,
    }

    /// Sanity: absorptance ∈ [0, 1] for a variety of parameter sets.
    #[test]
    fn absorptance_bounded_for_various_stacks() {
        let cases = [
            (
                LayerSpec {
                    n: 1.5,
                    k: 0.0,
                    d_m: 100e-9,
                },
                LayerSpec {
                    n: 3.5,
                    k: 0.01,
                    d_m: 200e-9,
                },
                LayerSpec {
                    n: 0.05,
                    k: 4.0,
                    d_m: 100e-9,
                },
            ),
            (
                LayerSpec {
                    n: 2.0,
                    k: 0.0,
                    d_m: 75e-9,
                },
                LayerSpec {
                    n: 4.0,
                    k: 0.1,
                    d_m: 50e-9,
                },
                LayerSpec {
                    n: 1.5,
                    k: 0.0,
                    d_m: 50e-9,
                },
            ),
            (
                LayerSpec {
                    n: 1.2,
                    k: 0.0,
                    d_m: 50e-9,
                },
                LayerSpec {
                    n: 2.0,
                    k: 0.5,
                    d_m: 300e-9,
                },
                LayerSpec {
                    n: 0.05,
                    k: 4.0,
                    d_m: 200e-9,
                },
            ),
        ];
        for (ar, abs, back) in &cases {
            let stack = TmmStack {
                n0: 1.0,
                k0: 0.0,
                n1: ar.n,
                k1: ar.k,
                d1_m: ar.d_m,
                n2: abs.n,
                k2: abs.k,
                d2_m: abs.d_m,
                n3: back.n,
                k3: back.k,
                d3_m: back.d_m,
                n4: 1.0,
                k4: 0.0,
            };
            let a = tmm_stack_absorptance(&stack, 550e-9);
            assert!(
                (0.0..=1.0).contains(&a),
                "A={a:.4} out of range for ar.n={}, abs.k={}, back.k={}",
                ar.n,
                abs.k,
                back.k
            );
        }
    }

    /// Energy conservation: R + T + A = 1 for a lossless stack.
    #[test]
    fn energy_conservation_lossless_stack() {
        // All k=0 → pure dielectric stack: A must be negligible
        let stack = TmmStack {
            n0: 1.0,
            k0: 0.0,
            n1: 1.5,
            k1: 0.0,
            d1_m: 100e-9,
            n2: 2.0,
            k2: 0.0,
            d2_m: 200e-9,
            n3: 1.2,
            k3: 0.0,
            d3_m: 50e-9,
            n4: 1.0,
            k4: 0.0,
        };
        let a = tmm_stack_absorptance(&stack, 550e-9);
        // For a lossless stack, A should be zero (or very close to it due to floating point)
        assert!(a < 1e-10, "Lossless stack should have A≈0, got {a:.2e}");
    }

    /// TMM with zero-thickness AR layer should match bare absorber interface.
    #[test]
    fn zero_thickness_ar_matches_bare() {
        let stack_zero_ar = TmmStack {
            n0: 1.0,
            k0: 0.0,
            n1: 1.5,
            k1: 0.0,
            d1_m: 0.0, // AR with zero thickness → invisible
            n2: 3.5,
            k2: 0.02,
            d2_m: 200e-9,
            n3: 0.05,
            k3: 4.0,
            d3_m: 100e-9,
            n4: 1.0,
            k4: 0.0,
        };
        let stack_bare = TmmStack {
            n0: 1.0,
            k0: 0.0,
            n1: 1.0,
            k1: 0.0,
            d1_m: 0.0, // AR with n=1.0 (matches air) → also invisible
            n2: 3.5,
            k2: 0.02,
            d2_m: 200e-9,
            n3: 0.05,
            k3: 4.0,
            d3_m: 100e-9,
            n4: 1.0,
            k4: 0.0,
        };
        let a_zero_ar = tmm_stack_absorptance(&stack_zero_ar, 600e-9);
        let a_bare = tmm_stack_absorptance(&stack_bare, 600e-9);
        // Both should give similar absorptance (allow small TMM numerical differences)
        assert!(
            (a_zero_ar - a_bare).abs() < 0.05,
            "Zero-AR A={a_zero_ar:.4} vs bare A={a_bare:.4}"
        );
    }

    /// evaluate_design returns bounded Jsc for a realistic Si absorber.
    #[test]
    fn evaluate_design_returns_finite_jsc() {
        let wls: Vec<f64> = (0..=90).map(|i| 300.0 + i as f64 * 10.0).collect();
        let spec = SolarSpectrum::am15g();
        let flux: Vec<f64> = wls
            .iter()
            .map(|&wl_nm| {
                let wl_m = wl_nm * 1e-9;
                spec.irradiance_at(wl_m) * wl_m / (PLANCK_H * SPEED_OF_LIGHT)
            })
            .collect();

        let params = DesignParams {
            ar_n: 1.9,
            ar_d_m: 75e-9,
            absorber_n: 3.5,
            absorber_k: 0.01,
            absorber_thickness_m: 200e-6,
            back_n: 0.05,
            back_k: 4.0,
            back_thickness_m: 100e-9,
        };
        let design = evaluate_design(&params, &wls, &flux).expect("evaluate_design failed");

        // A 200 µm absorber with k=0.01 absorbs nearly all photons over 300–1200 nm.
        // The AM1.5G photon current integrated over that range is ~55–58 mA/cm²,
        // so a realistic upper bound (allowing for some reflection loss) is 58 mA/cm².
        assert!(
            design.jsc_ma_cm2 > 0.0 && design.jsc_ma_cm2 < 58.0,
            "Jsc={:.2} mA/cm²",
            design.jsc_ma_cm2
        );
        assert_eq!(design.wavelengths_nm.len(), wls.len());
        assert_eq!(design.absorption_spectrum.len(), wls.len());
    }
}
