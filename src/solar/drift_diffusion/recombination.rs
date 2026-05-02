//! Carrier recombination models: SRH, radiative, and Auger.
//!
//! All models assume Boltzmann statistics (non-degenerate). The trap level for
//! SRH is assumed at mid-gap (n₁ = p₁ = nᵢ). References:
//! * Shockley-Read-Hall: Hall (1952), Shockley & Read (1952).
//! * Radiative: van Roosbroeck & Shockley (1954).
//! * Auger: Landsberg (1970), Dziewior & Schmid (1977) for Si.

use super::material::SemiconductorMaterial;

// ─── SRH recombination ────────────────────────────────────────────────────────

/// Shockley-Read-Hall recombination rate R_SRH (cm⁻³ s⁻¹).
///
/// Mid-gap trap: n₁ = p₁ = nᵢ.
///
/// R = (np − nᵢ²) / [τ_p(n + nᵢ) + τ_n(p + nᵢ)]
///
/// Sign convention: positive R means net recombination (np > nᵢ²).
pub fn srh_rate(n: f64, p: f64, ni: f64, tau_n: f64, tau_p: f64) -> f64 {
    let denom = tau_p * (n + ni) + tau_n * (p + ni);
    if denom < 1e-300 {
        return 0.0;
    }
    (n * p - ni * ni) / denom
}

/// Partial derivative ∂R_SRH / ∂n.
pub fn srh_dn(n: f64, p: f64, ni: f64, tau_n: f64, tau_p: f64) -> f64 {
    let denom = tau_p * (n + ni) + tau_n * (p + ni);
    if denom < 1e-300 {
        return 0.0;
    }
    let np_ni2 = n * p - ni * ni;
    // d/dn [(np-ni²)/D] = (p·D - (np-ni²)·tau_p) / D²
    (p * denom - np_ni2 * tau_p) / (denom * denom)
}

/// Partial derivative ∂R_SRH / ∂p.
pub fn srh_dp(n: f64, p: f64, ni: f64, tau_n: f64, tau_p: f64) -> f64 {
    let denom = tau_p * (n + ni) + tau_n * (p + ni);
    if denom < 1e-300 {
        return 0.0;
    }
    let np_ni2 = n * p - ni * ni;
    // d/dp [(np-ni²)/D] = (n·D - (np-ni²)·tau_n) / D²
    (n * denom - np_ni2 * tau_n) / (denom * denom)
}

// ─── Radiative recombination ──────────────────────────────────────────────────

/// Bimolecular (radiative) recombination rate (cm⁻³ s⁻¹).
///
/// R_rad = B · (np − nᵢ²)
pub fn radiative_rate(n: f64, p: f64, ni: f64, b_rad: f64) -> f64 {
    b_rad * (n * p - ni * ni)
}

/// ∂R_rad / ∂n = B · p
pub fn radiative_dn(p: f64, b_rad: f64) -> f64 {
    b_rad * p
}

/// ∂R_rad / ∂p = B · n
pub fn radiative_dp(n: f64, b_rad: f64) -> f64 {
    b_rad * n
}

// ─── Auger recombination ──────────────────────────────────────────────────────

/// Auger recombination rate (cm⁻³ s⁻¹).
///
/// R_A = (Cₙ·n + Cₚ·p) · (np − nᵢ²)
pub fn auger_rate(n: f64, p: f64, ni: f64, cn: f64, cp: f64) -> f64 {
    (cn * n + cp * p) * (n * p - ni * ni)
}

/// ∂R_A / ∂n.
///
/// d/dn [(Cn·n + Cp·p)(np - ni²)] = Cn·(np - ni²) + (Cn·n + Cp·p)·p
pub fn auger_dn(n: f64, p: f64, ni: f64, cn: f64, cp: f64) -> f64 {
    let np_ni2 = n * p - ni * ni;
    cn * np_ni2 + (cn * n + cp * p) * p
}

/// ∂R_A / ∂p.
///
/// d/dp [(Cn·n + Cp·p)(np - ni²)] = Cp·(np - ni²) + (Cn·n + Cp·p)·n
pub fn auger_dp(n: f64, p: f64, ni: f64, cn: f64, cp: f64) -> f64 {
    let np_ni2 = n * p - ni * ni;
    cp * np_ni2 + (cn * n + cp * p) * n
}

// ─── Total recombination ──────────────────────────────────────────────────────

/// Total net recombination rate R = R_SRH + R_rad + R_Auger (cm⁻³ s⁻¹).
pub fn total_recombination(n: f64, p: f64, ni: f64, mat: &SemiconductorMaterial) -> f64 {
    srh_rate(n, p, ni, mat.tau_n_s, mat.tau_p_s)
        + radiative_rate(n, p, ni, mat.b_rad_cm3_s)
        + auger_rate(n, p, ni, mat.cn_auger_cm6_s, mat.cp_auger_cm6_s)
}

/// ∂R_total / ∂n.
pub fn total_recombination_dn(n: f64, p: f64, ni: f64, mat: &SemiconductorMaterial) -> f64 {
    srh_dn(n, p, ni, mat.tau_n_s, mat.tau_p_s)
        + radiative_dn(p, mat.b_rad_cm3_s)
        + auger_dn(n, p, ni, mat.cn_auger_cm6_s, mat.cp_auger_cm6_s)
}

/// ∂R_total / ∂p.
pub fn total_recombination_dp(n: f64, p: f64, ni: f64, mat: &SemiconductorMaterial) -> f64 {
    srh_dp(n, p, ni, mat.tau_n_s, mat.tau_p_s)
        + radiative_dp(n, mat.b_rad_cm3_s)
        + auger_dp(n, p, ni, mat.cn_auger_cm6_s, mat.cp_auger_cm6_s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solar::drift_diffusion::material::SemiconductorMaterial;

    #[test]
    fn srh_rate_at_equilibrium_is_zero() {
        let ni = 1e10_f64;
        let tau = 1e-6;
        let n = ni;
        let p = ni;
        let r = srh_rate(n, p, ni, tau, tau);
        assert!(r.abs() < 1e-6, "R_SRH at equilibrium = {r}");
    }

    #[test]
    fn srh_minority_carrier_limit() {
        // n-type: n ≈ nd >> ni, excess holes Δp injected so p >> p_eq = ni²/nd.
        // R_SRH ≈ Δp / tau_p in the minority-carrier low-injection limit.
        // For the approximation to hold with < 5% error we need p >> ni and n >> ni.
        let ni = 1e10_f64;
        let nd = 1e16_f64;
        let n = nd; // majority electrons ≈ nd
                    // Inject Δp = 1e12 >> p_eq = 1e4 (so np - ni² ≈ n*Δp >> 0)
        let delta_p = 1e12_f64;
        let p = ni * ni / nd + delta_p;
        let tau_n = 1e-6;
        let tau_p = 1e-6;
        let r = srh_rate(n, p, ni, tau_n, tau_p);
        // Analytical: (np - ni²) / [tau_p*(n+ni) + tau_n*(p+ni)]
        // With n≈nd >> ni >> p, denominator ≈ tau_p * n.
        // Numerator ≈ n * Δp.  So R ≈ Δp / tau_p.
        let r_approx = delta_p / tau_p;
        let rel = (r - r_approx).abs() / r_approx;
        assert!(rel < 0.05, "SRH minority approx rel err = {rel}");
    }

    #[test]
    fn total_recombination_matches_sum() {
        let mat = SemiconductorMaterial::silicon();
        let ni = mat.ni_cm3;
        let n = 1e15_f64;
        let p = 1e15_f64;
        let r_total = total_recombination(n, p, ni, &mat);
        let r_srh = srh_rate(n, p, ni, mat.tau_n_s, mat.tau_p_s);
        let r_rad = radiative_rate(n, p, ni, mat.b_rad_cm3_s);
        let r_aug = auger_rate(n, p, ni, mat.cn_auger_cm6_s, mat.cp_auger_cm6_s);
        let diff = (r_total - (r_srh + r_rad + r_aug)).abs();
        assert!(
            diff < 1e-10 * r_total.abs().max(1e-20),
            "total != sum, diff={diff}"
        );
    }
}
