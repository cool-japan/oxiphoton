/// Kerr (χ³) nonlinear optical effect.
///
/// The Kerr effect causes an intensity-dependent refractive index:
///   n(I) = n₀ + n₂·I
///
/// where n₂ is the nonlinear refractive index coefficient (m²/W) and
/// I is the optical intensity (W/m²).
///
/// This leads to self-phase modulation (SPM) in waveguides:
///   φ_NL = γ·P·L
///
/// where γ = n₂·ω/(c·A_eff) is the nonlinear coefficient (rad/(W·m)).
///
/// Kerr nonlinear coefficient for a material.
#[derive(Debug, Clone, Copy)]
pub struct KerrMaterial {
    /// Linear refractive index
    pub n0: f64,
    /// Nonlinear refractive index n₂ (m²/W)
    pub n2: f64,
}

impl KerrMaterial {
    pub fn new(n0: f64, n2: f64) -> Self {
        Self { n0, n2 }
    }

    /// Silica glass (standard SMF28)
    pub fn silica() -> Self {
        Self {
            n0: 1.45,
            n2: 2.6e-20,
        } // m²/W at 1550nm
    }

    /// Silicon (Si waveguide at 1550nm)
    pub fn silicon() -> Self {
        Self {
            n0: 3.476,
            n2: 6e-18,
        } // m²/W at 1550nm (two-photon absorption region)
    }

    /// Silicon nitride (Si₃N₄)
    pub fn silicon_nitride() -> Self {
        Self {
            n0: 2.0,
            n2: 2.4e-19,
        } // m²/W at 1550nm
    }

    /// Intensity-dependent refractive index n(I) = n₀ + n₂·I
    pub fn refractive_index(&self, intensity: f64) -> f64 {
        self.n0 + self.n2 * intensity
    }

    /// Self-phase modulation (SPM) nonlinear phase shift.
    ///
    /// φ_NL = n₂·k₀·I·L
    pub fn spm_phase(&self, intensity: f64, k0: f64, length: f64) -> f64 {
        self.n2 * k0 * intensity * length
    }

    /// Nonlinear coefficient γ = n₂·ω/(c·A_eff) [rad/(W·m)].
    pub fn gamma(&self, omega: f64, a_eff: f64) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        self.n2 * omega / (SPEED_OF_LIGHT * a_eff)
    }

    /// Nonlinear length: L_NL = 1/(γ·P₀) — characteristic length for SPM.
    pub fn nonlinear_length(&self, gamma: f64, peak_power: f64) -> f64 {
        if gamma * peak_power < 1e-30 {
            f64::INFINITY
        } else {
            1.0 / (gamma * peak_power)
        }
    }
}

/// B-integral: accumulated nonlinear phase over a path.
///
/// B = (2π/λ) · ∫ n₂ · I(z) dz
///
/// A B-integral < 1 rad indicates negligible nonlinear effects.
pub fn b_integral(n2: f64, wavelength: f64, intensity_profile: &[f64], dz: f64) -> f64 {
    use std::f64::consts::PI;
    let k0 = 2.0 * PI / wavelength;
    k0 * n2 * intensity_profile.iter().sum::<f64>() * dz
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn kerr_silica_n2_physical() {
        let mat = KerrMaterial::silica();
        // n2 for silica is ~2-3×10⁻²⁰ m²/W
        assert!(mat.n2 > 1e-21 && mat.n2 < 1e-19);
    }

    #[test]
    fn kerr_refractive_index_increases_with_intensity() {
        let mat = KerrMaterial::silica();
        let i1 = 1e12; // 1 TW/m²
        let i2 = 2e12;
        assert!(mat.refractive_index(i2) > mat.refractive_index(i1));
    }

    #[test]
    fn kerr_spm_phase_linear_in_length() {
        let mat = KerrMaterial::silica();
        let k0 = 2.0 * PI / 1550e-9;
        let i = 1e12;
        let phi1 = mat.spm_phase(i, k0, 1.0);
        let phi2 = mat.spm_phase(i, k0, 2.0);
        assert!((phi2 / phi1 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn kerr_gamma_si_larger_than_silica() {
        use crate::units::conversion::SPEED_OF_LIGHT;
        let si = KerrMaterial::silicon();
        let sio2 = KerrMaterial::silica();
        let omega = 2.0 * PI * SPEED_OF_LIGHT / 1550e-9;
        let a_eff = 0.5e-12; // 0.5 μm² for Si waveguide
        let gamma_si = si.gamma(omega, a_eff);
        let gamma_sio2 = sio2.gamma(omega, a_eff);
        assert!(gamma_si > gamma_sio2, "Si has larger γ than silica");
    }

    #[test]
    fn b_integral_zero_for_zero_intensity() {
        let profile = vec![0.0f64; 100];
        let b = b_integral(2.6e-20, 1550e-9, &profile, 1e-3);
        assert_eq!(b, 0.0);
    }

    #[test]
    fn nonlinear_length_decreases_with_power() {
        let mat = KerrMaterial::silicon();
        let omega = 2.0 * std::f64::consts::PI * crate::units::conversion::SPEED_OF_LIGHT / 1550e-9;
        let a_eff = 0.5e-12;
        let gamma = mat.gamma(omega, a_eff);
        let l1 = mat.nonlinear_length(gamma, 1e-3); // 1mW
        let l2 = mat.nonlinear_length(gamma, 10e-3); // 10mW
        assert!(l1 > l2, "Higher power → shorter nonlinear length");
    }
}
