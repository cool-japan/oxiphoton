//! Optoacoustic and acousto-optic interaction physics
//!
//! Implements:
//! - Stimulated Brillouin scattering (SBS) in optical fibres
//! - Photothermal effect and thermal lensing
//! - Acousto-optic modulation (AOM / acousto-optic deflector)

use std::f64::consts::PI;

/// Speed of light in vacuum (m/s)
const C_LIGHT: f64 = 2.99792458e8;

/// Stimulated Brillouin Scattering (SBS) in optical waveguides.
///
/// SBS arises from the parametric coupling between the pump photon, a Stokes
/// photon, and an acoustic phonon via electrostriction. It is the dominant
/// nonlinear impairment in coherent optical fibre communication.
///
/// # Key relations
/// - Brillouin shift:  ν_B = 2 n v_a / λ
/// - Stokes wavelength: λ_S = λ / (1 + 2 n v_a / c)
/// - SBS threshold:    P_th ≈ 21 A_eff / (g_B L_eff)
/// - Gain spectrum:    g(ν) = g_B / [1 + (ν−ν_B)²/(Δν_B/2)²]
#[derive(Debug, Clone)]
pub struct StimulatedBrillouinScattering {
    /// Refractive index of waveguide core
    pub n: f64,
    /// Acoustic velocity in waveguide (m/s)
    pub c_sound: f64,
    /// Pump wavelength (m)
    pub wavelength_m: f64,
    /// Brillouin linewidth Δν_B (Hz, FWHM)
    pub linewidth_hz: f64,
    /// Peak Brillouin gain coefficient g_B (m/W)
    pub g_b_peak: f64,
}

impl StimulatedBrillouinScattering {
    /// Brillouin frequency shift (Hz).
    ///
    /// ν_B = 2 n v_a / λ
    ///
    /// For silica fibre at 1550 nm: n = 1.45, v_a = 5960 m/s → ν_B ≈ 11.16 GHz
    pub fn brillouin_shift_hz(&self) -> f64 {
        2.0 * self.n * self.c_sound / self.wavelength_m
    }

    /// Stokes wavelength (m).
    ///
    /// The Stokes frequency is ν_S = ν_pump − ν_B, so:
    ///   λ_S = c / ν_S = c / (c/λ − 2nv_a/λ) = λ × c / (c − 2 n v_a)
    ///
    /// The Stokes photon is red-shifted (longer wavelength) relative to the pump.
    /// Typical shift: ~89 pm at 1550 nm in silica (ν_B ≈ 11.16 GHz).
    pub fn stokes_wavelength_m(&self) -> f64 {
        let denom = C_LIGHT - 2.0 * self.n * self.c_sound;
        if denom.abs() < 1.0e-10 {
            return f64::INFINITY;
        }
        self.wavelength_m * C_LIGHT / denom
    }

    /// Anti-Stokes wavelength (m).
    ///
    /// λ_AS = λ_pump / (1 − 2 n v_a / c)
    ///
    /// Anti-Stokes is blue-shifted; typically much weaker (thermally weighted).
    pub fn anti_stokes_wavelength_m(&self) -> f64 {
        let shift_factor = 2.0 * self.n * self.c_sound / C_LIGHT;
        let denom = 1.0 - shift_factor;
        if denom.abs() < 1.0e-15 {
            return f64::INFINITY;
        }
        self.wavelength_m / denom
    }

    /// SBS threshold power in a fibre span.
    ///
    /// P_th ≈ 21 A_eff / (g_B L_eff)
    ///
    /// # Arguments
    /// * `effective_area_m2`   — mode field area A_eff (m²)
    /// * `effective_length_m`  — effective interaction length L_eff (m)
    ///   For passive fibre of length L and attenuation α: L_eff = (1 − e^{−αL}) / α
    pub fn threshold_power_w(&self, effective_area_m2: f64, effective_length_m: f64) -> f64 {
        if self.g_b_peak <= 0.0 || effective_length_m <= 0.0 {
            return f64::INFINITY;
        }
        21.0 * effective_area_m2 / (self.g_b_peak * effective_length_m)
    }

    /// Brillouin gain spectrum (Lorentzian lineshape).
    ///
    /// g(ν) = g_B / [1 + ((ν − ν_B) / (Δν_B / 2))²]
    ///
    /// # Arguments
    /// * `freq_offset_hz` — frequency offset from pump (Hz); peak at ν = ν_B
    pub fn gain_at_frequency(&self, freq_offset_hz: f64) -> f64 {
        let delta_nu = freq_offset_hz - self.brillouin_shift_hz();
        let half_lw = self.linewidth_hz / 2.0;
        if half_lw <= 0.0 {
            return if delta_nu.abs() < 1.0e-9 {
                self.g_b_peak
            } else {
                0.0
            };
        }
        self.g_b_peak / (1.0 + (delta_nu / half_lw).powi(2))
    }

    /// Stokes power growth along propagation direction z.
    ///
    /// P_S(z) = P_S0 × exp(g_B P_pump z / A_eff)
    ///
    /// Valid in the undepleted pump approximation (P_S ≪ P_pump).
    pub fn stokes_power_gain(&self, pump_power_w: f64, z_m: f64, area_m2: f64) -> f64 {
        if area_m2 <= 0.0 {
            return 1.0;
        }
        let exponent = self.g_b_peak * pump_power_w * z_m / area_m2;
        exponent.exp()
    }

    /// Effective Brillouin length (m): length over which Stokes grows by e.
    ///
    /// L_B = A_eff / (g_B P_pump)
    pub fn effective_length_m(&self, pump_power_w: f64, area_m2: f64) -> f64 {
        if self.g_b_peak <= 0.0 || pump_power_w <= 0.0 {
            return f64::INFINITY;
        }
        area_m2 / (self.g_b_peak * pump_power_w)
    }
}

/// Photothermal effect and thermal lensing in absorbing optical media.
///
/// When a laser beam propagates through a medium with non-zero absorption
/// (or residual impurity absorption), the deposited heat changes the refractive
/// index through the thermo-optic coefficient dn/dT. The resulting radial
/// index gradient forms a thermal lens that defocuses (or focuses) the beam.
#[derive(Debug, Clone)]
pub struct ThermalLensing {
    /// Thermo-optic coefficient dn/dT (K⁻¹)
    pub dn_dt: f64,
    /// Optical absorption coefficient μ_a (m⁻¹)
    pub absorption_coeff: f64,
    /// Thermal conductivity κ (W m⁻¹ K⁻¹)
    pub thermal_conductivity: f64,
    /// Beam waist (1/e² radius) w₀ (m)
    pub beam_waist_m: f64,
}

impl ThermalLensing {
    /// On-axis temperature rise (K) for CW steady-state irradiation.
    ///
    /// For a Gaussian beam in a medium of path length L, with absorbed power
    /// P_abs = P_in × (1 − exp(−μ_a L)) ≈ P_in × μ_a × L for μ_a L ≪ 1:
    ///
    ///   ΔT(0) ≈ P_abs × μ_a / (4π κ)   [per unit length, integrated over beam]
    ///
    /// This uses the infinite-medium Green's function for a cylindrical heat source.
    pub fn temperature_rise_k(&self, absorbed_power_w: f64, path_length_m: f64) -> f64 {
        if self.thermal_conductivity <= 0.0 || path_length_m <= 0.0 {
            return 0.0;
        }
        // Integrated temperature rise for Gaussian heat deposition
        let q_lin =
            absorbed_power_w * self.absorption_coeff / (4.0 * PI * self.thermal_conductivity);
        q_lin * path_length_m
    }

    /// Thermal lens focal length (m).
    ///
    /// The radial index gradient n(r) = n₀ + (dn/dT) ΔT(r) acts as a thin lens.
    /// For Gaussian absorbed power profile:
    ///
    ///   1/f_th = (π κ w₀²)⁻¹ × P_abs × |dn/dT| × μ_a × L
    ///
    /// Sign convention: negative f (diverging lens) when dn/dT < 0.
    pub fn thermal_lens_focal_length_m(&self, power_w: f64, path_length_m: f64) -> f64 {
        let w0 = self.beam_waist_m;
        if w0 <= 0.0 || path_length_m <= 0.0 || power_w <= 0.0 {
            return f64::INFINITY;
        }
        let denom = power_w * self.dn_dt.abs() * self.absorption_coeff * path_length_m;
        if denom.abs() < 1.0e-30 {
            return f64::INFINITY;
        }
        let f_th = PI * self.thermal_conductivity * w0 * w0 / denom;
        if self.dn_dt < 0.0 {
            -f_th
        } else {
            f_th
        }
    }

    /// Critical power for thermal self-focusing (m).
    ///
    /// Above P_crit, the thermal lens focuses the beam so strongly that
    /// diffraction is overcome and the beam narrows runaway:
    ///
    ///   P_crit = π κ w₀ / (L × |dn/dT| × μ_a)
    pub fn critical_power_w(&self, path_length_m: f64) -> f64 {
        let w0 = self.beam_waist_m;
        if path_length_m <= 0.0 || self.dn_dt.abs() < 1.0e-30 || self.absorption_coeff <= 0.0 {
            return f64::INFINITY;
        }
        PI * self.thermal_conductivity * w0
            / (path_length_m * self.dn_dt.abs() * self.absorption_coeff)
    }

    /// Phase accumulated across the beam cross-section due to thermal lensing.
    ///
    /// Δφ_th = (2π / λ) × |dn/dT| × ΔT × L
    pub fn thermal_phase_rad(
        &self,
        wavelength_m: f64,
        absorbed_power_w: f64,
        path_length_m: f64,
    ) -> f64 {
        if wavelength_m <= 0.0 {
            return 0.0;
        }
        let delta_t = self.temperature_rise_k(absorbed_power_w, path_length_m);
        (2.0 * PI / wavelength_m) * self.dn_dt.abs() * delta_t * path_length_m
    }
}

/// Acousto-optic modulator (AOM) — Bragg diffraction by sound.
///
/// An AOM uses a piezoelectric transducer to launch a travelling acoustic
/// wave (angular frequency Ω, wavelength Λ = v_a / f_a) into an optical
/// medium. Light can be diffracted into the ±1 orders by Bragg scattering,
/// with a frequency shift equal to the acoustic frequency (Doppler shift).
#[derive(Debug, Clone)]
pub struct AcoustoOpticModulator {
    /// Refractive index of the AOM medium
    pub n: f64,
    /// Acoustic frequency f_a (Hz)
    pub acoustic_freq_hz: f64,
    /// Acoustic phase velocity v_a (m/s)
    pub acoustic_velocity_m_s: f64,
    /// Optical interaction length L (m)
    pub l_interaction_m: f64,
    /// Optical wavelength in vacuum (m)
    pub wavelength_m: f64,
}

impl AcoustoOpticModulator {
    /// Acoustic wavelength Λ = v_a / f_a (m)
    pub fn acoustic_wavelength_m(&self) -> f64 {
        self.acoustic_velocity_m_s / self.acoustic_freq_hz
    }

    /// Bragg angle inside the medium: sin(θ_B) = λ / (2 n Λ)
    ///
    /// This is the angle of incidence (measured from the acoustic wavefront)
    /// at which the Bragg condition is exactly satisfied.
    pub fn bragg_angle_rad(&self) -> f64 {
        let arg = self.wavelength_m / (2.0 * self.n * self.acoustic_wavelength_m());
        arg.clamp(-1.0, 1.0).asin()
    }

    /// Q parameter (Klein-Cook): Q = 2π λ L / (n Λ²)
    ///
    /// Bragg regime: Q ≫ 1 (Q > 10). Raman-Nath regime: Q ≪ 1.
    pub fn klein_cook_q(&self) -> f64 {
        let lambda_a = self.acoustic_wavelength_m();
        2.0 * PI * self.wavelength_m * self.l_interaction_m / (self.n * lambda_a * lambda_a)
    }

    /// Raman-Nath parameter ν = π Δn L / λ
    ///
    /// Δn is the index modulation depth induced by the acoustic wave.
    pub fn raman_nath_parameter(&self, delta_n: f64) -> f64 {
        PI * delta_n * self.l_interaction_m / self.wavelength_m
    }

    /// Diffraction efficiency in the Bragg regime: η = sin²(ν)
    ///
    /// Maximum efficiency η = 100 % when ν = π/2 (i.e. Δn L = λ/2).
    pub fn diffraction_efficiency(&self, delta_n: f64) -> f64 {
        let nu = self.raman_nath_parameter(delta_n);
        nu.sin().powi(2)
    }

    /// Frequency of the diffracted beam: f_out = f_in ± n_order × f_acoustic
    ///
    /// Positive order (+1): beam propagating with acoustic wave (up-shifted).
    /// Negative order (−1): beam propagating against acoustic wave (down-shifted).
    pub fn diffracted_frequency_hz(&self, input_freq_hz: f64, order: i32) -> f64 {
        input_freq_hz + order as f64 * self.acoustic_freq_hz
    }

    /// Required acoustic power density (W/m²) for a target diffraction efficiency.
    ///
    /// From the AOM figure of merit M₂ = n⁶ p² / (ρ v_a³):
    ///   Δn² = M₂ I_ac / 2  →  η = sin²(π √(M₂ I_ac) L / (2 λ))
    ///
    /// This returns I_ac for η → η_target:
    ///   I_ac = (2/M₂) × [arcsin(√η) × λ / (π L)]²
    pub fn required_acoustic_intensity(
        &self,
        target_efficiency: f64,
        m2_figure_of_merit: f64,
    ) -> f64 {
        let eta = target_efficiency.clamp(0.0, 1.0);
        if m2_figure_of_merit <= 0.0 || self.l_interaction_m <= 0.0 {
            return f64::INFINITY;
        }
        let arcsin_sqrt_eta = eta.sqrt().asin();
        let delta_n = arcsin_sqrt_eta * self.wavelength_m / (PI * self.l_interaction_m);
        2.0 * delta_n * delta_n / m2_figure_of_merit
    }

    /// Deflection angle of the first diffracted order (rad).
    ///
    /// θ_d ≈ λ / Λ = λ f_a / v_a  (paraxial, outside medium)
    pub fn deflection_angle_rad(&self) -> f64 {
        self.wavelength_m / self.acoustic_wavelength_m()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sbs_brillouin_shift_silica() {
        let sbs = StimulatedBrillouinScattering {
            n: 1.45,
            c_sound: 5960.0,
            wavelength_m: 1550.0e-9,
            linewidth_hz: 30.0e6,
            g_b_peak: 5.0e-11,
        };
        let nu = sbs.brillouin_shift_hz();
        // ν_B = 2 × 1.45 × 5960 / 1550e-9 ≈ 11.16 GHz
        let expected = 2.0 * 1.45 * 5960.0 / 1550.0e-9;
        assert!(
            (nu - expected).abs() / expected < 1.0e-10,
            "ν_B={}GHz expected {}GHz",
            nu / 1.0e9,
            expected / 1.0e9
        );
        // Within 2% of 11.16 GHz
        assert!(
            (nu - 11.16e9).abs() / 11.16e9 < 0.02,
            "ν_B={}GHz",
            nu / 1.0e9
        );
    }

    #[test]
    fn sbs_stokes_red_shifted() {
        let sbs = StimulatedBrillouinScattering {
            n: 1.45,
            c_sound: 5960.0,
            wavelength_m: 1550.0e-9,
            linewidth_hz: 30.0e6,
            g_b_peak: 5.0e-11,
        };
        let lambda_s = sbs.stokes_wavelength_m();
        // Stokes should be red-shifted (longer wavelength than pump)
        assert!(lambda_s > sbs.wavelength_m, "Stokes must be red-shifted");
        // Shift ≈ 89 pm for 11.16 GHz shift at 1550 nm
        let shift_nm = (lambda_s - sbs.wavelength_m) * 1.0e9;
        assert!(
            shift_nm > 0.05 && shift_nm < 0.5,
            "Δλ={}pm",
            shift_nm * 1000.0
        );
    }

    #[test]
    fn sbs_gain_at_peak() {
        let sbs = StimulatedBrillouinScattering {
            n: 1.45,
            c_sound: 5960.0,
            wavelength_m: 1550.0e-9,
            linewidth_hz: 30.0e6,
            g_b_peak: 5.0e-11,
        };
        let nu_b = sbs.brillouin_shift_hz();
        let g_peak = sbs.gain_at_frequency(nu_b);
        assert!(
            (g_peak - sbs.g_b_peak).abs() / sbs.g_b_peak < 1.0e-10,
            "g_peak={}",
            g_peak
        );
    }

    #[test]
    fn sbs_gain_half_at_halfwidth() {
        let sbs = StimulatedBrillouinScattering {
            n: 1.45,
            c_sound: 5960.0,
            wavelength_m: 1550.0e-9,
            linewidth_hz: 30.0e6,
            g_b_peak: 5.0e-11,
        };
        let nu_b = sbs.brillouin_shift_hz();
        // At ν = ν_B + ΔνB/2, Lorentzian → g_B / 2
        let g_half = sbs.gain_at_frequency(nu_b + sbs.linewidth_hz / 2.0);
        assert!((g_half - sbs.g_b_peak / 2.0).abs() / (sbs.g_b_peak / 2.0) < 1.0e-10);
    }

    #[test]
    fn aom_bragg_angle_silica() {
        let aom = AcoustoOpticModulator {
            n: 1.5,
            acoustic_freq_hz: 80.0e6,
            acoustic_velocity_m_s: 3630.0, // TeO₂
            l_interaction_m: 5.0e-3,
            wavelength_m: 532.0e-9,
        };
        let theta = aom.bragg_angle_rad();
        // sin(θ_B) = λ/(2nΛ) = 532e-9 / (2 × 1.5 × 3630/80e6)
        let lambda_a = aom.acoustic_wavelength_m();
        let expected_sin = 532.0e-9 / (2.0 * 1.5 * lambda_a);
        let expected = expected_sin.clamp(-1.0, 1.0).asin();
        assert!(
            (theta - expected).abs() < 1.0e-12,
            "θ_B={}mrad",
            theta * 1000.0
        );
    }

    #[test]
    fn aom_diffraction_efficiency_max() {
        let aom = AcoustoOpticModulator {
            n: 1.5,
            acoustic_freq_hz: 80.0e6,
            acoustic_velocity_m_s: 3630.0,
            l_interaction_m: 5.0e-3,
            wavelength_m: 532.0e-9,
        };
        // At ν = π/2 → sin²(π/2) = 1 → 100 % efficiency
        // Δn = ν × λ / (π L) = (π/2) × 532e-9 / (π × 5e-3) = 532e-9 / (2 × 5e-3) = 53.2 µ
        let delta_n_max = aom.wavelength_m / (2.0 * aom.l_interaction_m);
        let eta = aom.diffraction_efficiency(delta_n_max);
        assert!((eta - 1.0).abs() < 1.0e-10, "η_max={}", eta);
    }

    #[test]
    fn aom_frequency_shift() {
        let aom = AcoustoOpticModulator {
            n: 1.5,
            acoustic_freq_hz: 80.0e6,
            acoustic_velocity_m_s: 3630.0,
            l_interaction_m: 5.0e-3,
            wavelength_m: 532.0e-9,
        };
        let f_in = 563.0e12; // ≈ 532 nm
        let f_out_plus = aom.diffracted_frequency_hz(f_in, 1);
        let f_out_minus = aom.diffracted_frequency_hz(f_in, -1);
        assert!((f_out_plus - f_in - 80.0e6).abs() < 1.0, "f_out+");
        assert!((f_out_minus - f_in + 80.0e6).abs() < 1.0, "f_out−");
    }

    #[test]
    fn thermal_lens_focal_length_sign() {
        // Negative dn/dT → diverging lens (negative f)
        let lens = ThermalLensing {
            dn_dt: -1.0e-5,
            absorption_coeff: 1.0,
            thermal_conductivity: 1.4,
            beam_waist_m: 1.0e-3,
        };
        let f = lens.thermal_lens_focal_length_m(1.0, 0.05);
        assert!(
            f < 0.0,
            "Negative dn/dT should give diverging (negative) focal length, got {}",
            f
        );
    }
}
