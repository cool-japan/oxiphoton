//! Textured-Si light-trapping: 1D grating diffraction (front) + oblique Beer-Lambert (back).
//!
//! Physical model:
//!
//! **Front grating** — thin phase-grating (Fraunhofer) model for a rectangular 1D grating:
//!   - The grating impresses a phase profile t(x) = exp(iφ(x)) where
//!     φ(x) = Δφ for 0 < x < f·Λ and 0 otherwise.
//!   - Δφ = 2π · (n_si − 1) · d / λ  (optical path difference at grating depth d)
//!   - Fourier coefficient amplitude |c_m|:
//!     - m = 0: |c_0|² = (1−f)² + f² + 2f(1−f)cos(Δφ)
//!     - m ≠ 0: |c_m|² = 4sin²(Δφ/2) · sin²(πmf) / (πm)²
//!   - Energy conservation: Σ_m |c_m|² = 1  (lossless phase screen)
//!
//! **Diffraction angles** — grating equation in transmission (light entering Si):
//!   sin(θ_m) = m·λ / (n_si·Λ)
//!   Evanescent orders (|sin θ_m| ≥ 1) are discarded.
//!
//! **Back stack** — each transmitted order m propagates at angle θ_m in Si.
//!   For a double-pass absorber with perfect back reflector:
//!   A_m(λ) = 1 − exp(−2·α(λ)·L / cos(θ_m))
//!
//! **J_sc integration**:
//!   J_sc = q · ∫ Σ_m |c_m|² · A_m(λ) · Φ_AM15G(λ) dλ
//!
//! **Lambertian reference** — uses the `lambertian_jsc_si` function (F = 4n², Beer-Lambert).
//!
//! **Planar reference** — uses only the m=0 order with straight through Si.

use std::f64::consts::PI;

use crate::error::OxiPhotonError;
use crate::solar::absorption::AbsorptionMaterial;
use crate::solar::light_trapping::lambertian_jsc_si;
use crate::solar::spectrum::SolarSpectrum;

// ─── Physical constants ───────────────────────────────────────────────────────

const CHARGE_C: f64 = 1.602_176_634e-19; // C
const PLANCK_JS: f64 = 6.626_070_15e-34; // J·s
const SPEED_OF_LIGHT_MS: f64 = 2.997_924_58e8; // m/s

/// Representative c-Si refractive index (real part) used in diffraction angle computation.
const N_SI: f64 = 3.5;

// ─── Result type ──────────────────────────────────────────────────────────────

/// Result of the textured-absorption evaluation.
#[derive(Debug, Clone)]
pub struct TexturedAbsorptionResult {
    /// Textured Jsc (mA/cm²): grating-coupled multi-order absorption
    pub jsc_ma_cm2: f64,
    /// Planar (zeroth-order-only) Jsc (mA/cm²)
    pub jsc_planar_ma_cm2: f64,
    /// Lambertian-limit Jsc (mA/cm²) from `lambertian_jsc_si`
    pub jsc_lambertian_ma_cm2: f64,
    /// Enhancement factor: jsc_textured / jsc_planar
    pub enhancement_factor: f64,
    /// Lambertian fraction: jsc_textured / jsc_lambertian
    pub lambertian_fraction: f64,
    /// Wavelength grid used (nm)
    pub wavelengths_nm: Vec<f64>,
    /// Absorption spectrum A(λ) for the textured case
    pub absorption_spectrum: Vec<f64>,
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Evaluate textured-Si absorption using a 1D rectangular phase grating (front) and a
/// double-pass Beer-Lambert absorber with perfect back reflector (back stack).
///
/// # Arguments
/// * `period_m`            — grating period Λ (m)
/// * `depth_m`             — grating etch depth d (m)
/// * `duty_cycle`          — ridge fill factor f (0 < f < 1)
/// * `absorber_thickness_m`— Si absorber thickness L (m)
/// * `am15g`               — AM1.5G solar spectrum (used for J_sc integration)
/// * `n_orders`            — maximum diffraction order index N (m = ±1..±N considered)
///
/// # Returns
/// A `TexturedAbsorptionResult` containing J_sc values and the absorption spectrum, or an
/// `OxiPhotonError` for invalid parameter combinations.
///
/// # Deviation note
/// The existing `RcwaSolver::solve` zeroes out all non-zeroth-order transmission efficiencies
/// (single-layer approximation only).  Per the deviation policy this function therefore uses a
/// first-principles thin-phase-grating Fourier model, which is physically exact in the
/// Fraunhofer / scalar diffraction limit for subwavelength-to-wavelength-scale gratings.
pub fn evaluate_textured_absorption(
    period_m: f64,
    depth_m: f64,
    duty_cycle: f64,
    absorber_thickness_m: f64,
    am15g: &SolarSpectrum,
    n_orders: usize,
) -> Result<TexturedAbsorptionResult, OxiPhotonError> {
    // ── Parameter validation ──────────────────────────────────────────────────
    if period_m <= 0.0 {
        return Err(OxiPhotonError::InvalidLayer(format!(
            "period_m must be positive, got {period_m:.3e}"
        )));
    }
    if depth_m < 0.0 {
        return Err(OxiPhotonError::InvalidLayer(format!(
            "depth_m must be non-negative, got {depth_m:.3e}"
        )));
    }
    if !(0.0 < duty_cycle && duty_cycle < 1.0) {
        return Err(OxiPhotonError::InvalidLayer(format!(
            "duty_cycle must be in (0,1), got {duty_cycle}"
        )));
    }
    if absorber_thickness_m <= 0.0 {
        return Err(OxiPhotonError::InvalidLayer(format!(
            "absorber_thickness_m must be positive, got {absorber_thickness_m:.3e}"
        )));
    }

    // ── Wavelength grid: 300–1200 nm in 20 nm steps ──────────────────────────
    let wavelengths_nm = wavelength_grid_nm(300.0, 1200.0, 20.0);
    let nw = wavelengths_nm.len();

    // ── Absorption material (c-Si) ────────────────────────────────────────────
    let si_material = AbsorptionMaterial::crystalline_silicon();

    // ── Per-wavelength absorption spectra ────────────────────────────────────
    let mut abs_textured = vec![0.0_f64; nw];
    let mut abs_planar = vec![0.0_f64; nw];

    for (i, &wl_nm) in wavelengths_nm.iter().enumerate() {
        let wl_m = wl_nm * 1.0e-9;

        // Grating phase depth Δφ at this wavelength
        let delta_phi = 2.0 * PI * (N_SI - 1.0) * depth_m / wl_m;

        // Absorption coefficient (m⁻¹) in Si at this wavelength
        let alpha = si_material.alpha_at_nm(wl_nm);

        // Compute textured absorption (sum over all propagating orders)
        abs_textured[i] = absorption_sum(
            period_m,
            duty_cycle,
            delta_phi,
            alpha,
            absorber_thickness_m,
            wl_m,
            n_orders,
        );

        // Planar: zeroth order only
        abs_planar[i] = absorption_sum(
            period_m,
            duty_cycle,
            delta_phi,
            alpha,
            absorber_thickness_m,
            wl_m,
            0, // only m=0
        );
    }

    // ── J_sc integration (trapezoidal) ───────────────────────────────────────
    let jsc_textured = integrate_jsc(&wavelengths_nm, &abs_textured, am15g);
    let jsc_planar = integrate_jsc(&wavelengths_nm, &abs_planar, am15g);

    // ── Lambertian reference ─────────────────────────────────────────────────
    let thickness_nm = absorber_thickness_m * 1.0e9;
    let jsc_lambertian = lambertian_jsc_si(thickness_nm);

    // ── Derived metrics ───────────────────────────────────────────────────────
    let enhancement_factor = if jsc_planar > 1.0e-12 {
        jsc_textured / jsc_planar
    } else {
        1.0
    };
    let lambertian_fraction = if jsc_lambertian > 1.0e-12 {
        jsc_textured / jsc_lambertian
    } else {
        0.0
    };

    Ok(TexturedAbsorptionResult {
        jsc_ma_cm2: jsc_textured,
        jsc_planar_ma_cm2: jsc_planar,
        jsc_lambertian_ma_cm2: jsc_lambertian,
        enhancement_factor,
        lambertian_fraction,
        wavelengths_nm,
        absorption_spectrum: abs_textured,
    })
}

// ─── Diffraction model ────────────────────────────────────────────────────────

/// Compute the summed absorption A(λ) = Σ_m |c_m|² · A_m(λ) for all propagating orders
/// from m = −n_orders to +n_orders.
///
/// The thin-phase-grating (Fraunhofer) model gives exact Fourier coefficients for a
/// rectangular phase profile t(x) = exp(iΔφ) for 0 < x < f·Λ, 1 otherwise.
///
/// For m = 0:
///   |c_0|² = (1−f)² + f² + 2f(1−f)·cos(Δφ)
///
/// For m ≠ 0:
///   |c_m|² = 4·sin²(Δφ/2) · sin²(π·m·f) / (π·m)²
///
/// Energy conservation: Σ_{m=-∞}^{+∞} |c_m|² = 1 (lossless phase screen).
///
/// Evanescent orders (|sin θ_m| ≥ 1) are skipped.
fn absorption_sum(
    period_m: f64,
    duty_cycle: f64,
    delta_phi: f64,
    alpha: f64,
    thickness_m: f64,
    wl_m: f64,
    n_orders: usize,
) -> f64 {
    let f = duty_cycle;
    let sin_half = (delta_phi * 0.5).sin();
    let sin_half_sq = sin_half * sin_half;

    let mut total = 0.0_f64;

    let orders_range = -(n_orders as i64)..=(n_orders as i64);

    for m in orders_range {
        let cm_sq = phase_grating_order_power(m, f, sin_half_sq, delta_phi);
        if cm_sq <= 0.0 {
            continue;
        }

        // Grating equation: sin(θ_m) = m·λ / (n_si·Λ)
        let sin_theta_m = (m as f64) * wl_m / (N_SI * period_m);

        // Skip evanescent (total-internal-reflection) orders
        if sin_theta_m.abs() >= 1.0 {
            continue;
        }

        let cos_theta_m = (1.0 - sin_theta_m * sin_theta_m).sqrt();

        // Double-pass Beer-Lambert absorption with oblique path (perfect back reflector)
        // A_m = 1 − exp(−2·α·L / cos(θ_m))
        let a_m = absorptance_double_pass_oblique(alpha, thickness_m, cos_theta_m);

        total += cm_sq * a_m;
    }

    total
}

/// Power fraction of order m for a rectangular phase grating.
///
/// Uses the closed-form Fourier coefficients of the phase transmittance
/// t(x) = exp(iΔφ) for 0 < x < f·Λ, 1 otherwise.
///
/// # Arguments
/// * `m`           — diffraction order index
/// * `f`           — duty cycle (fill factor)
/// * `sin_half_sq` — sin²(Δφ/2), precomputed for efficiency
/// * `delta_phi`   — total phase depth Δφ = 2π(n−1)d/λ
fn phase_grating_order_power(m: i64, f: f64, sin_half_sq: f64, delta_phi: f64) -> f64 {
    if m == 0 {
        // |c_0|² = (1−f)² + f² + 2f(1−f)cos(Δφ)
        let cos_phi = delta_phi.cos();
        (1.0 - f) * (1.0 - f) + f * f + 2.0 * f * (1.0 - f) * cos_phi
    } else {
        // |c_m|² = 4·sin²(Δφ/2) · sin²(π·m·f) / (π·m)²
        let sin_pi_mf = (PI * m as f64 * f).sin();
        4.0 * sin_half_sq * sin_pi_mf * sin_pi_mf / (PI * m as f64) / (PI * m as f64)
    }
}

/// Double-pass absorptance in a slab of thickness L at angle θ (given cos θ).
///
/// A = 1 − exp(−2·α·L / cos(θ))
///
/// Clamps output to [0, 1].
fn absorptance_double_pass_oblique(alpha: f64, thickness_m: f64, cos_theta: f64) -> f64 {
    if cos_theta <= 0.0 || !cos_theta.is_finite() {
        return 0.0;
    }
    let exponent = -2.0 * alpha * thickness_m / cos_theta;
    (1.0 - exponent.exp()).clamp(0.0, 1.0)
}

// ─── J_sc integration helper ─────────────────────────────────────────────────

/// Compute J_sc (mA/cm²) from a wavelength grid (nm) and absorption spectrum A(λ).
///
/// J_sc = q · ∫ A(λ) · Φ_AM15G(λ) dλ
///
/// Uses the midpoint rule on adjacent wavelength intervals.
fn integrate_jsc(wavelengths_nm: &[f64], absorption: &[f64], am15g: &SolarSpectrum) -> f64 {
    let nw = wavelengths_nm.len();
    if nw < 2 {
        return 0.0;
    }

    let mut jsc = 0.0_f64;

    for i in 0..nw - 1 {
        let wl_mid_nm = 0.5 * (wavelengths_nm[i] + wavelengths_nm[i + 1]);
        let dwl_nm = wavelengths_nm[i + 1] - wavelengths_nm[i];
        let wl_mid_m = wl_mid_nm * 1.0e-9;
        let dwl_m = dwl_nm * 1.0e-9;

        let irrad = am15g.irradiance_at(wl_mid_m);
        if irrad <= 0.0 {
            continue;
        }

        // Average absorption over the interval
        let a_mid = 0.5 * (absorption[i] + absorption[i + 1]);

        // Photon flux Φ(λ) = I(λ) · λ / (h·c)
        let phi = irrad * wl_mid_m / (PLANCK_JS * SPEED_OF_LIGHT_MS);

        jsc += CHARGE_C * a_mid * phi * dwl_m;
    }

    // Convert A/m² → mA/cm²: ×0.1
    jsc * 0.1
}

// ─── Wavelength grid helper ───────────────────────────────────────────────────

/// Build a uniform wavelength grid from `start` to `end` (nm) with step `step` (nm).
fn wavelength_grid_nm(start: f64, end: f64, step: f64) -> Vec<f64> {
    let n = ((end - start) / step).round() as usize + 1;
    (0..n).map(|i| start + i as f64 * step).collect()
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify energy conservation: Σ|c_m|² = 1 for any Δφ and f.
    #[test]
    fn phase_grating_energy_conservation() {
        let f = 0.5_f64;
        let delta_phi = 1.5_f64;
        let sin_half_sq = (delta_phi * 0.5).sin().powi(2);
        let n_check = 200;
        let mut sum = 0.0_f64;
        for m in -(n_check as i64)..=(n_check as i64) {
            sum += phase_grating_order_power(m, f, sin_half_sq, delta_phi);
        }
        // Should converge close to 1.0 for large n_check
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Energy not conserved: Σ|c_m|² = {sum:.6}"
        );
    }

    /// Shallow grating (Δφ → 0): nearly all power in m=0.
    #[test]
    fn shallow_grating_zeroth_order_dominates() {
        let f = 0.5_f64;
        let delta_phi = 0.001_f64;
        let sin_half_sq = (delta_phi * 0.5).sin().powi(2);
        let c0 = phase_grating_order_power(0, f, sin_half_sq, delta_phi);
        assert!(
            c0 > 0.999,
            "Shallow grating: |c_0|² should be ≈1, got {c0:.6}"
        );
    }

    /// Double-pass oblique absorptance increases with angle (longer path through absorber).
    #[test]
    fn oblique_path_increases_absorption() {
        let alpha = 1.0e4; // m⁻¹
        let thickness = 200.0e-6; // 200 µm
        let a_normal = absorptance_double_pass_oblique(alpha, thickness, 1.0);
        let a_oblique = absorptance_double_pass_oblique(alpha, thickness, 0.9);
        assert!(
            a_oblique >= a_normal,
            "Oblique A={a_oblique:.6} should be >= normal A={a_normal:.6}"
        );
    }
}
