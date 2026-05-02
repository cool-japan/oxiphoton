//! Bandgap-narrowing (BGN) models for heavily-doped silicon and GaAs.
//!
//! For heavily-doped emitters (N > ~1e18 cm⁻³) bandgap narrowing causes an
//! effective increase in the intrinsic carrier concentration:
//!
//!   n_ie² = n_i² · exp(ΔEg / V_T)
//!
//! This reduces V_oc by 10–30 mV and increases dark current J0 significantly.
//!
//! # References
//! * Slotboom-de Graaff 1976 (Solid-State Electronics 19, 857)
//! * Klaassen-Slotboom-de Graaff 1992 (Solid-State Electronics 35, 125)
//! * Harmon, Melloch & Lundstrom 1994, Appl. Phys. Lett. 64(4):502

/// Selectable BGN model for a semiconductor material.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BgnModel {
    /// No bandgap narrowing (classical n_i).
    None,
    /// Slotboom-de Graaff 1976: V₁=9 meV, N_ref=1e17 cm⁻³.
    Slotboom,
    /// Klaassen-Slotboom-de Graaff 1992: V₁=6.92 meV, N_ref=1.3e17 cm⁻³.
    Klaassen,
    /// Harmon-Melloch-Lundstrom 1994 effective BGN for GaAs (N^(1/3) model).
    ///
    /// Uses measured effective bandgap shrinkage (appropriate for minority-carrier
    /// transport and open-circuit voltage calculations in solar cells).
    /// Source: Harmon, Melloch & Lundstrom (1994), Appl. Phys. Lett. 64(4):502.
    Harmon1994,
}

/// Shared ΔEg formula with model-specific parameters.
///
/// ΔEg(N) = V₁ · (ln(N/N_ref) + √(ln²(N/N_ref) + 0.5))
///
/// This formula is ALWAYS POSITIVE for any N > 0 because sqrt(x²+0.5) > |x| for all x.
fn delta_eg_ev_impl(v1_ev: f64, n_ref_cm3: f64, n_total_cm3: f64) -> f64 {
    if n_total_cm3 <= 0.0 {
        return 0.0;
    }
    let ln_r = (n_total_cm3 / n_ref_cm3).ln();
    v1_ev * (ln_r + (ln_r * ln_r + 0.5_f64).sqrt())
}

/// Slotboom-de Graaff 1976 bandgap narrowing (eV).
///
/// V₁=9 meV, N_ref=1e17 cm⁻³.
///
/// Returns the bandgap narrowing ΔEg in electron-volts for the given total
/// impurity concentration `n_total_cm3` (cm⁻³).
pub fn slotboom_delta_eg_ev(n_total_cm3: f64) -> f64 {
    delta_eg_ev_impl(9.0e-3, 1.0e17, n_total_cm3)
}

/// Klaassen-Slotboom-de Graaff 1992 bandgap narrowing (eV).
///
/// V₁=6.92 meV, N_ref=1.3e17 cm⁻³.
///
/// Returns the bandgap narrowing ΔEg in electron-volts for the given total
/// impurity concentration `n_total_cm3` (cm⁻³).
pub fn klaassen_delta_eg_ev(n_total_cm3: f64) -> f64 {
    delta_eg_ev_impl(6.92e-3, 1.3e17, n_total_cm3)
}

/// Effective bandgap narrowing for GaAs using the measured single-term model.
///
/// Source: Harmon, Melloch & Lundstrom (1994), "Effective band-gap shrinkage in GaAs,"
/// Appl. Phys. Lett. 64(4):502. DOI: 10.1063/1.111075. (Purdue ePubs open access.)
///
/// Formula: ΔEg_eff = A · N^(1/3)   [eV; N in cm⁻³; A in eV·cm]
/// p-GaAs: A = 2.55e-8 eV·cm  → 25.5 meV at N = 1e18 cm⁻³
/// n-GaAs: A = 3.23e-8 eV·cm  → 32.3 meV at N = 1e18 cm⁻³
///
/// Uses effective BGN (not optical BGN), appropriate for minority-carrier transport
/// and open-circuit voltage calculations in solar cells.
///
/// Returns zero below N = 1e14 cm⁻³ (intrinsic regime where BGN is negligible).
pub fn harmon1994_gaas_delta_eg_ev(n_total_cm3: f64, is_n_type: bool) -> f64 {
    if n_total_cm3 < 1e14 {
        return 0.0;
    }
    let a = if is_n_type { 3.23e-8_f64 } else { 2.55e-8_f64 };
    a * n_total_cm3.powf(1.0 / 3.0)
}

/// Effective intrinsic carrier concentration squared (cm⁻⁶), accounting for BGN.
///
/// n_ie² = n_i² · exp(ΔEg / V_T)
///
/// # Arguments
/// * `ni_cm3`     — intrinsic carrier concentration (cm⁻³)
/// * `delta_eg_ev` — bandgap narrowing energy (eV), must be ≥ 0
/// * `vt_v`       — thermal voltage k_B T / q (V)
pub fn ni_eff_squared_cm6(ni_cm3: f64, delta_eg_ev: f64, vt_v: f64) -> f64 {
    ni_cm3 * ni_cm3 * (delta_eg_ev / vt_v).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slotboom_at_n_ref_is_positive() {
        let result = slotboom_delta_eg_ev(1e17);
        assert!(result > 0.0, "Expected ΔEg > 0 at N=N_ref, got {result}");
    }

    #[test]
    fn bernoulli_like_continuity() {
        // Check smoothness of ΔEg near N_ref using a small relative step.
        // eps = 1e-6 (1 ppm): expected finite-difference ≈ V1/sqrt(0.5) * 2*eps ≈ 2.5e-8 eV.
        // Threshold set at 1e-6 eV (1 μeV) — far below any physically relevant variation.
        let eps = 1e-6_f64;
        let d1 = slotboom_delta_eg_ev(1e17 * (1.0 - eps));
        let d2 = slotboom_delta_eg_ev(1e17 * (1.0 + eps));
        let diff_ev = (d1 - d2).abs();
        assert!(
            diff_ev < 1e-6,
            "Expected ΔEg smooth at N_ref (eps=1e-6): |d1-d2|={diff_ev:.3e} eV"
        );
    }
}
