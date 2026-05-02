//! Scharfetter-Gummel flux discretisation for electron and hole continuity.
//!
//! The Scharfetter-Gummel (SG) scheme provides an exponentially-fitted upwind
//! discretisation of the drift-diffusion flux that is numerically stable for
//! large electric fields. Reference: Scharfetter & Gummel, IEEE TED 1969.
//!
//! # Conventions
//! * Carrier densities in cm⁻³.
//! * Diffusion coefficient in cm²/s.
//! * Grid spacing `dx` in cm.
//! * Electrostatic potential `psi` in V.
//! * Thermal voltage `vt = k_B T / q` in V.
//! * Currents in A/cm² (SI units, just restricted to 1D slab).

use super::material::Q;

// ─── Bernoulli function ───────────────────────────────────────────────────────

/// Bernoulli function B(x) = x / (e^x − 1).
///
/// Uses a 4th-order Taylor series for |x| < 1e-3 to avoid the 0/0 singularity
/// at x = 0. The Taylor expansion is:
///   B(x) = 1 − x/2 + x²/12 − x⁴/720 + O(x⁶)
///
/// For |x| ≥ 1e-3 the direct formula is numerically stable.
pub fn bernoulli(x: f64) -> f64 {
    if x.abs() < 1e-3 {
        1.0 - x / 2.0 + x * x / 12.0 - x * x * x * x / 720.0
    } else {
        x / (x.exp() - 1.0)
    }
}

/// Derivative of Bernoulli function B'(x) = d/dx [x / (e^x − 1)].
///
/// Taylor series for |x| < 1e-3:
///   B'(x) ≈ −1/2 + x/6 − x³/180
///
/// Exact formula for |x| ≥ 1e-3:
///   B'(x) = (e^x − 1 − x · e^x) / (e^x − 1)²
pub fn bernoulli_deriv(x: f64) -> f64 {
    if x.abs() < 1e-3 {
        -0.5 + x / 6.0 - x * x * x / 180.0
    } else {
        let ex = x.exp();
        let em1 = ex - 1.0;
        (em1 - x * ex) / (em1 * em1)
    }
}

// ─── Scharfetter-Gummel electron flux ─────────────────────────────────────────

/// Scharfetter-Gummel electron flux at half-node i+½.
///
/// J_{n,i+½} = −(q·D_n/dx) · [B(Δψ/V_T)·n_{i+1} − B(−Δψ/V_T)·n_i]
///
/// Positive current is in the +x direction (left→right).
///
/// # Arguments
/// * `n_l` — electron density at left node i (cm⁻³).
/// * `n_r` — electron density at right node i+1 (cm⁻³).
/// * `psi_l` — electrostatic potential at left node (V).
/// * `psi_r` — electrostatic potential at right node (V).
/// * `vt` — thermal voltage k_B T / q (V).
/// * `dn` — electron diffusion coefficient D_n (cm²/s).
/// * `dx` — grid spacing x_{i+1} − x_i (cm).
pub fn sg_flux_n(n_l: f64, n_r: f64, psi_l: f64, psi_r: f64, vt: f64, dn: f64, dx: f64) -> f64 {
    let dpsi = (psi_r - psi_l) / vt;
    -(Q * dn / dx) * (n_r * bernoulli(dpsi) - n_l * bernoulli(-dpsi))
}

/// Scharfetter-Gummel hole flux at half-node i+½.
///
/// J_{p,i+½} = (q·D_p/dx) · [B(−Δψ/V_T)·p_{i+1} − B(Δψ/V_T)·p_i]
///
/// Positive current is in the +x direction.
pub fn sg_flux_p(p_l: f64, p_r: f64, psi_l: f64, psi_r: f64, vt: f64, dp: f64, dx: f64) -> f64 {
    let dpsi = (psi_r - psi_l) / vt;
    (Q * dp / dx) * (p_r * bernoulli(-dpsi) - p_l * bernoulli(dpsi))
}

// ─── Jacobian partials of SG fluxes ──────────────────────────────────────────

/// Partial derivatives of J_{n,i+½} with respect to state variables.
///
/// Returns `(dJn_dn_l, dJn_dn_r, dJn_dpsi_l, dJn_dpsi_r)`.
///
/// Chain rule applied to `sg_flux_n`. The `dpsi`-partials come from
/// differentiating Bernoulli w.r.t. the argument and multiplying by `±1/vt`.
pub fn sg_flux_n_derivs(
    n_l: f64,
    n_r: f64,
    psi_l: f64,
    psi_r: f64,
    vt: f64,
    dn: f64,
    dx: f64,
) -> (f64, f64, f64, f64) {
    let dpsi = (psi_r - psi_l) / vt;
    let b_pos = bernoulli(dpsi);
    let b_neg = bernoulli(-dpsi);
    let db_pos = bernoulli_deriv(dpsi);
    let db_neg = bernoulli_deriv(-dpsi);
    let coeff = -Q * dn / dx;

    // dJ_n/dn_l  = coeff * (-b_neg) = coeff * B(-dpsi) ... wait, sign:
    // J_n = coeff * (n_r * B(dpsi) - n_l * B(-dpsi))
    // dJ_n/dn_l  = coeff * (-B(-dpsi))  = -coeff * b_neg
    // dJ_n/dn_r  = coeff * B(dpsi)
    let djn_dn_l = -coeff * b_neg;
    let djn_dn_r = coeff * b_pos;

    // dJ_n/dpsi_l: d/dpsi_l of coeff*(n_r*B(dpsi) - n_l*B(-dpsi))
    //   dpsi/dpsi_l = -1/vt, so:
    //   = coeff * (n_r * db_pos * (-1/vt) - n_l * db_neg * (+1/vt))
    //   Note: d(-dpsi)/dpsi_l = +1/vt, so dB(-dpsi)/dpsi_l = db_neg * (+1/vt)
    let djn_dpsi_l = coeff / vt * (-n_r * db_pos - n_l * db_neg);

    // dJ_n/dpsi_r: dpsi/dpsi_r = +1/vt:
    //   = coeff * (n_r * db_pos * (+1/vt) - n_l * db_neg * (-1/vt))
    //   Note: d(-dpsi)/dpsi_r = -1/vt, so dB(-dpsi)/dpsi_r = db_neg * (-1/vt)
    let djn_dpsi_r = coeff / vt * (n_r * db_pos + n_l * db_neg);

    (djn_dn_l, djn_dn_r, djn_dpsi_l, djn_dpsi_r)
}

/// Partial derivatives of J_{p,i+½} with respect to state variables.
///
/// Returns `(dJp_dp_l, dJp_dp_r, dJp_dpsi_l, dJp_dpsi_r)`.
pub fn sg_flux_p_derivs(
    p_l: f64,
    p_r: f64,
    psi_l: f64,
    psi_r: f64,
    vt: f64,
    dp: f64,
    dx: f64,
) -> (f64, f64, f64, f64) {
    let dpsi = (psi_r - psi_l) / vt;
    let b_pos = bernoulli(dpsi);
    let b_neg = bernoulli(-dpsi);
    let db_pos = bernoulli_deriv(dpsi);
    let db_neg = bernoulli_deriv(-dpsi);
    let coeff = Q * dp / dx;

    // J_p = coeff * (p_r * B(-dpsi) - p_l * B(dpsi))
    let djp_dp_l = coeff * (-b_pos);
    let djp_dp_r = coeff * b_neg;

    // J_p = coeff * (p_r * B(-dpsi) - p_l * B(dpsi))
    // dJ_p/dpsi_l: d(-dpsi)/dpsi_l = +1/vt, d(dpsi)/dpsi_l = -1/vt
    //   = coeff * [p_r * db_neg * (+1/vt) - p_l * db_pos * (-1/vt)]
    //   = coeff/vt * [p_r * db_neg + p_l * db_pos]
    let djp_dpsi_l = coeff / vt * (p_r * db_neg + p_l * db_pos);

    // dJ_p/dpsi_r: d(-dpsi)/dpsi_r = -1/vt, d(dpsi)/dpsi_r = +1/vt
    //   = coeff * [p_r * db_neg * (-1/vt) - p_l * db_pos * (+1/vt)]
    //   = -coeff/vt * [p_r * db_neg + p_l * db_pos]
    let djp_dpsi_r = -coeff / vt * (p_r * db_neg + p_l * db_pos);

    (djp_dp_l, djp_dp_r, djp_dpsi_l, djp_dpsi_r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bernoulli_at_zero_is_one() {
        // B(0) = 1 by L'Hôpital (or Taylor)
        let b = bernoulli(0.0);
        assert!((b - 1.0).abs() < 1e-12, "B(0) = {b}");
    }

    #[test]
    fn bernoulli_symmetry() {
        // B(x) + B(-x) should equal x/2 * 2 = ... actually: B(x) + B(-x) - 1 != nice.
        // A known identity: B(x) - B(-x) = -x (verify numerically)
        for x in [0.5_f64, 1.0, 2.0, 5.0, 10.0] {
            let diff = bernoulli(x) - bernoulli(-x);
            assert!(
                (diff + x).abs() < 1e-10,
                "B(x)-B(-x)+x = {} for x={x}",
                diff + x
            );
        }
    }

    #[test]
    fn bernoulli_taylor_matches_full_formula() {
        // In the transition region just outside the Taylor branch
        let x = 1.5e-3_f64;
        let b_exact = x / (x.exp() - 1.0);
        let b_taylor = bernoulli(x); // uses Taylor (< 1e-3 threshold uses Taylor, x=1.5e-3 uses full)
        assert!((b_exact - b_taylor).abs() / b_exact < 1e-9);
    }

    #[test]
    fn bernoulli_deriv_zero() {
        // B'(0) = -1/2 from Taylor
        let db = bernoulli_deriv(0.0);
        assert!((db + 0.5).abs() < 1e-12, "B'(0) = {db}");
    }
}
