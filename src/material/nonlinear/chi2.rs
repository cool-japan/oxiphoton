use std::f64::consts::PI;

/// Second-order nonlinear optical interactions (χ²).
///
/// Covers Second Harmonic Generation (SHG), Sum/Difference Frequency Generation,
/// and Optical Parametric Amplification (OPA).
///
/// Reference: Boyd, "Nonlinear Optics", 3rd ed., Ch. 2.
///
/// Effective nonlinear susceptibility d_eff for common χ² materials.
#[derive(Debug, Clone, Copy)]
pub struct Chi2Material {
    /// Linear refractive index at fundamental
    pub n1: f64,
    /// Linear refractive index at second harmonic
    pub n2: f64,
    /// Effective d-coefficient (m/V), also d_eff = χ²/2
    pub d_eff: f64,
}

impl Chi2Material {
    pub fn new(n1: f64, n2: f64, d_eff: f64) -> Self {
        Self { n1, n2, d_eff }
    }

    /// KTP (KTiOPO₄) at 1064nm/532nm (type-II phase matching)
    pub fn ktp() -> Self {
        Self {
            n1: 1.7396,
            n2: 1.7468,
            d_eff: 3.64e-12,
        } // d_eff in m/V
    }

    /// LiNbO₃ (Lithium Niobate) at 1064nm/532nm
    pub fn lithium_niobate() -> Self {
        Self {
            n1: 2.156,
            n2: 2.232,
            d_eff: 27e-12,
        } // d_33 ≈ 27 pm/V
    }

    /// BBO (β-Barium Borate) at 800nm/400nm
    pub fn bbo() -> Self {
        Self {
            n1: 1.661,
            n2: 1.672,
            d_eff: 2.0e-12,
        }
    }

    /// Phase mismatch Δk = k(2ω) - 2k(ω) for SHG.
    ///
    /// Δk = 2ω/c · (n₂ω - n_ω) where n₂ω is index at second harmonic.
    pub fn phase_mismatch_shg(&self, omega: f64) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        2.0 * omega / SPEED_OF_LIGHT * (self.n2 - self.n1)
    }

    /// SHG conversion efficiency in the undepleted pump approximation.
    ///
    /// η = (d_eff²·ω²·I_ω·L²)/(n_ω²·n₂ω·ε₀·c³) · sinc²(ΔkL/2)
    ///
    /// This returns the normalized efficiency (η · c³ · ε₀ · n₁² · n₂).
    pub fn shg_efficiency_normalized(&self, omega: f64, length: f64, delta_k: f64) -> f64 {
        let sinc_val = if (delta_k * length / 2.0).abs() < 1e-10 {
            1.0
        } else {
            (delta_k * length / 2.0).sin() / (delta_k * length / 2.0)
        };
        let d2 = self.d_eff * self.d_eff;
        let omega2 = omega * omega;
        d2 * omega2 * length * length * sinc_val * sinc_val
    }

    /// Coherence length: L_c = π/Δk (distance for maximum SHG build-up).
    pub fn coherence_length(&self, omega: f64) -> f64 {
        let dk = self.phase_mismatch_shg(omega);
        if dk.abs() < 1e-30 {
            f64::INFINITY
        } else {
            PI / dk.abs()
        }
    }

    /// Quasi-phase-matched (QPM) poling period for SHG.
    ///
    /// Λ = 2π/Δk (required periodicity for periodic poling)
    pub fn qpm_period(&self, omega: f64) -> f64 {
        let dk = self.phase_mismatch_shg(omega);
        if dk.abs() < 1e-30 {
            f64::INFINITY
        } else {
            2.0 * PI / dk.abs()
        }
    }
}

/// Optical Parametric Amplification (OPA) gain.
///
/// Signal gain in the undepleted pump approximation:
///   G_s = cosh²(g·L)  where  g = sqrt(Γ² - (Δk/2)²)
///   Γ = sqrt(2·ω_s·ω_i·d_eff²·I_p / (n_s·n_i·n_p·ε₀·c³))
pub fn opa_gain(gamma: f64, delta_k: f64, length: f64) -> f64 {
    let g_sq = gamma * gamma - (delta_k / 2.0) * (delta_k / 2.0);
    if g_sq > 0.0 {
        let g = g_sq.sqrt();
        (g * length).cosh().powi(2)
    } else {
        // Phase-mismatch dominated: oscillatory
        let g = (-g_sq).sqrt();
        1.0 + (gamma / g).powi(2) * (g * length).sin().powi(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::conversion::SPEED_OF_LIGHT;

    #[test]
    fn ktp_d_eff_physical() {
        let ktp = Chi2Material::ktp();
        // d_eff for KTP is ~2-4 pm/V = 2-4×10⁻¹² m/V
        assert!(ktp.d_eff > 1e-12 && ktp.d_eff < 1e-11);
    }

    #[test]
    fn linbo3_d_eff_physical() {
        let lnbo = Chi2Material::lithium_niobate();
        // d_33 for LiNbO3 is ~20-30 pm/V
        assert!(lnbo.d_eff > 1e-11 && lnbo.d_eff < 1e-10);
    }

    #[test]
    fn shg_phase_mismatch_nonzero_for_dispersive() {
        let ktp = Chi2Material::ktp();
        let omega = 2.0 * PI * SPEED_OF_LIGHT / 1064e-9;
        let dk = ktp.phase_mismatch_shg(omega);
        // Dispersion causes nonzero phase mismatch unless phase-matched
        assert!(dk.abs() > 0.0 || ktp.n1 == ktp.n2);
    }

    #[test]
    fn coherence_length_positive() {
        let ktp = Chi2Material::ktp();
        let omega = 2.0 * PI * SPEED_OF_LIGHT / 1064e-9;
        let lc = ktp.coherence_length(omega);
        assert!(lc > 0.0);
    }

    #[test]
    fn shg_efficiency_max_at_zero_mismatch() {
        let mat = Chi2Material::lithium_niobate();
        let omega = 2.0 * PI * SPEED_OF_LIGHT / 1064e-9;
        let length = 10e-3; // 10mm crystal
        let eta_pm = mat.shg_efficiency_normalized(omega, length, 0.0);
        let eta_mismatch = mat.shg_efficiency_normalized(omega, length, 1000.0);
        assert!(
            eta_pm > eta_mismatch,
            "Phase-matched SHG should be more efficient"
        );
    }

    #[test]
    fn qpm_period_positive() {
        let ktp = Chi2Material::ktp();
        let omega = 2.0 * PI * SPEED_OF_LIGHT / 1064e-9;
        let period = ktp.qpm_period(omega);
        assert!(period > 0.0);
    }

    #[test]
    fn opa_gain_no_phase_mismatch() {
        // At phase matching: G = cosh²(g*L) ≥ 1
        let g = 100.0; // gain coefficient (m⁻¹)
        let gain = opa_gain(g, 0.0, 1e-3);
        assert!(gain >= 1.0);
    }

    #[test]
    fn opa_gain_decreases_with_mismatch() {
        let g = 100.0;
        let l = 1e-3;
        let gain_pm = opa_gain(g, 0.0, l);
        let gain_mis = opa_gain(g, 1e5, l); // large mismatch
        assert!(gain_pm >= gain_mis);
    }
}
