/// Raman scattering model for nonlinear fiber/waveguide optics.
///
/// The Raman effect produces a frequency-shifted copy of the pump wave
/// (Stokes shift ΩR downward, Anti-Stokes shift upward).
///
/// In fiber optics the Raman contribution to the nonlinear response is
/// characterised by the fractional Raman response fR ≈ 0.18 for silica, and
/// the Raman gain coefficient gR(Δω) peaked at the Stokes shift.
///
/// Stimulated Raman Scattering (SRS) gain coefficient (m/W):
///   gR(Δω) = gR_peak · h(Δω)
///
/// where h(Δω) is the normalised Raman response spectrum.
use std::f64::consts::PI;

/// Raman response parameters for a nonlinear material.
#[derive(Debug, Clone, Copy)]
pub struct RamanMaterial {
    /// Fractional Raman contribution to χ³ (0 < fR < 1)
    pub f_r: f64,
    /// Peak Raman gain coefficient (m/W)
    pub g_r_peak: f64,
    /// Stokes frequency shift (rad/s) — Ω_R = 2π × ν_shift
    pub omega_r: f64,
    /// Raman linewidth (rad/s) — Γ_R (FWHM / 2)
    pub gamma_r: f64,
    /// Damping timescale τ₁ (s), resonance timescale τ₂ (s) for delayed response
    pub tau1: f64,
    pub tau2: f64,
}

impl RamanMaterial {
    /// Silica (SiO₂) — standard single-mode fiber.
    ///
    /// Parameters from Agrawal, "Nonlinear Fiber Optics", 5th ed., Table 2.1.
    pub fn silica() -> Self {
        Self {
            f_r: 0.18,
            g_r_peak: 1e-13,             // ~1e-13 m/W at peak shift
            omega_r: 2.0 * PI * 13.2e12, // 13.2 THz Stokes shift
            gamma_r: 2.0 * PI * 3.0e12,  // ~3 THz linewidth
            tau1: 12.2e-15,              // 12.2 fs
            tau2: 32.0e-15,              // 32 fs
        }
    }

    /// Silicon — dominant phonon at 15.6 THz, large peak gain.
    pub fn silicon() -> Self {
        Self {
            f_r: 0.043,
            g_r_peak: 76e-11,             // ~76 cm/GW = 7.6e-9 m/W (at 1550nm)
            omega_r: 2.0 * PI * 15.6e12,  // 15.6 THz optical phonon
            gamma_r: 2.0 * PI * 0.105e12, // ~105 GHz linewidth (narrow)
            tau1: 10.2e-15,
            tau2: 3.03e-12, // narrow line → long tau2
        }
    }

    /// Diamond — highest Raman gain per unit frequency, ultra-wide shift.
    pub fn diamond() -> Self {
        Self {
            f_r: 0.10,
            g_r_peak: 75e-12,            // ~75 pm/W
            omega_r: 2.0 * PI * 40.0e12, // 40 THz (1332 cm⁻¹)
            gamma_r: 2.0 * PI * 0.5e12,  // ~500 GHz
            tau1: 8.0e-15,
            tau2: 0.64e-12,
        }
    }

    /// Raman gain spectral shape h(Δω) using the two-oscillator model.
    ///
    /// h(Δω) = (τ₁² + τ₂²) / (τ₁·τ₂²) · exp(-Δω·τ₂) · sin(Δω·τ₁)
    ///
    /// Normalised so that its integral = 1.
    pub fn raman_response(&self, delta_omega: f64) -> f64 {
        if delta_omega <= 0.0 {
            return 0.0;
        }
        let tau1 = self.tau1;
        let tau2 = self.tau2;
        let prefactor = (tau1 * tau1 + tau2 * tau2) / (tau1 * tau2 * tau2);
        prefactor * (-delta_omega * tau2).exp() * (delta_omega * tau1).sin()
    }

    /// Lorentzian approximation to Raman gain spectrum (m/W).
    ///
    ///   gR(Δω) = gR_peak · ΓR² / ((Δω - ΩR)² + ΓR²)
    pub fn gain_spectrum(&self, delta_omega: f64) -> f64 {
        let dw = delta_omega - self.omega_r;
        let gr = self.gamma_r;
        self.g_r_peak * gr * gr / (dw * dw + gr * gr)
    }

    /// Peak Raman gain (m/W) — at Δω = ΩR.
    pub fn peak_gain(&self) -> f64 {
        self.g_r_peak
    }

    /// Stokes wavelength shift (m) at pump wavelength λ_p.
    ///
    ///   Δλ ≈ λ_p² · ΩR / (2π·c)
    pub fn stokes_shift_wavelength(&self, lambda_pump: f64) -> f64 {
        let c = 2.998e8;
        lambda_pump * lambda_pump * self.omega_r / (2.0 * PI * c)
    }

    /// SRS threshold power (W) for a fiber of effective length L_eff and mode area A_eff.
    ///
    ///   P_th ≈ 16 · A_eff / (gR_peak · L_eff)
    pub fn srs_threshold(&self, a_eff: f64, l_eff: f64) -> f64 {
        16.0 * a_eff / (self.g_r_peak * l_eff)
    }

    /// Raman gain (m⁻¹) for pump power P (W) and mode area A_eff (m²).
    ///
    ///   g = gR · P / A_eff
    pub fn gain_coefficient(&self, pump_power: f64, a_eff: f64) -> f64 {
        self.g_r_peak * pump_power / a_eff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silica_raman_params_physical() {
        let m = RamanMaterial::silica();
        assert!(m.f_r > 0.0 && m.f_r < 1.0);
        assert!(m.omega_r > 2.0 * PI * 10e12); // > 10 THz
        assert!(m.g_r_peak > 0.0);
    }

    #[test]
    fn raman_response_positive_at_stokes() {
        let m = RamanMaterial::silica();
        let h = m.raman_response(m.omega_r);
        assert!(h > 0.0);
    }

    #[test]
    fn raman_response_zero_for_negative_shift() {
        let m = RamanMaterial::silica();
        assert!(m.raman_response(-1e12) == 0.0);
    }

    #[test]
    fn gain_spectrum_peaks_at_omega_r() {
        let m = RamanMaterial::silica();
        let at_peak = m.gain_spectrum(m.omega_r);
        let off_peak = m.gain_spectrum(m.omega_r * 1.5);
        assert!(at_peak > off_peak);
    }

    #[test]
    fn stokes_shift_wavelength_positive() {
        let m = RamanMaterial::silica();
        let dl = m.stokes_shift_wavelength(1550e-9);
        assert!(dl > 0.0 && dl < 200e-9); // < 200nm shift
    }

    #[test]
    fn srs_threshold_physical() {
        let m = RamanMaterial::silica();
        // SMF: A_eff ≈ 80μm², L_eff ≈ 20km
        let p_th = m.srs_threshold(80e-12, 20e3);
        // Should be in watts range (typically ~1W)
        assert!(p_th > 0.1 && p_th < 100.0);
    }

    #[test]
    fn silicon_raman_higher_stokes_than_silica() {
        let sio2 = RamanMaterial::silica();
        let si = RamanMaterial::silicon();
        // Silicon: 15.6 THz vs Silica: 13.2 THz
        assert!(si.omega_r > sio2.omega_r);
    }
}
