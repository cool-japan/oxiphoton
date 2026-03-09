//! Distributed and Lumped Raman Fiber Amplifier models.
//!
//! Implements on-off gain, net gain, noise figure, effective length, and
//! gain spectrum for stimulated Raman scattering (SRS) amplifiers.
//!
//! References:
//! - Headley & Agrawal, "Raman Amplification in Fiber Optical Communication Systems",
//!   Academic Press 2005.
//! - Islam, "Raman Amplifiers for Telecommunications", IEEE J. Sel. Top. QE 8(3) 2002.

use super::edfa::PumpDirection;

/// Planck constant (J·s)
const H_PLANCK: f64 = 6.626_070_15e-34;
/// Speed of light (m/s)
const C_LIGHT: f64 = 2.997_924_58e8;

/// Raman peak shift for silica fiber (cm⁻¹) — dominant Stokes peak.
const RAMAN_PEAK_SHIFT_CM_INV: f64 = 440.0;
/// Raman gain linewidth (cm⁻¹) for Lorentzian spectrum.
const RAMAN_LINEWIDTH_CM_INV: f64 = 60.0;

/// Fiber type determining Raman gain coefficient and effective area.
#[derive(Debug, Clone, PartialEq)]
pub enum RamanFiberType {
    /// Standard single-mode fiber (SMF-28). g_R ≈ 0.4 (W·km)⁻¹ at 1550 nm.
    Smf28,
    /// Dispersion-compensating fiber. g_R ≈ 3 (W·km)⁻¹ at 1550 nm.
    Dcf,
    /// Highly nonlinear fiber. g_R ≈ 2 (W·km)⁻¹ at 1550 nm.
    Hnlf,
    /// User-defined fiber parameters.
    Custom {
        /// Raman gain coefficient g_R [(W·km)⁻¹] at peak shift.
        raman_gain_coeff: f64,
        /// Effective area (μm²).
        effective_area_um2: f64,
        /// Fiber loss at 1550 nm (dB/km).
        loss_db_per_km: f64,
    },
}

impl RamanFiberType {
    /// Raman gain coefficient g_R (W·m)⁻¹ — converted from (W·km)⁻¹.
    pub fn raman_gain_coefficient(&self) -> f64 {
        let g_per_w_km = match self {
            RamanFiberType::Smf28 => 0.4,
            RamanFiberType::Dcf => 3.0,
            RamanFiberType::Hnlf => 2.0,
            RamanFiberType::Custom {
                raman_gain_coeff, ..
            } => *raman_gain_coeff,
        };
        g_per_w_km * 1e-3 // (W·km)⁻¹ → (W·m)⁻¹
    }

    /// Effective mode area A_eff (μm²).
    pub fn effective_area_um2(&self) -> f64 {
        match self {
            RamanFiberType::Smf28 => 80.0,
            RamanFiberType::Dcf => 20.0,
            RamanFiberType::Hnlf => 12.0,
            RamanFiberType::Custom {
                effective_area_um2, ..
            } => *effective_area_um2,
        }
    }

    /// Fiber loss at 1550 nm (dB/km).
    pub fn loss_db_per_km_at_1550(&self) -> f64 {
        match self {
            RamanFiberType::Smf28 => 0.2,
            RamanFiberType::Dcf => 0.5,
            RamanFiberType::Hnlf => 0.9,
            RamanFiberType::Custom { loss_db_per_km, .. } => *loss_db_per_km,
        }
    }

    /// Loss coefficient α (1/m) at 1550 nm.
    fn loss_per_m(&self) -> f64 {
        self.loss_db_per_km_at_1550() * 1e-3 * f64::ln(10.0) / 10.0
    }
}

/// Distributed Raman Amplifier (DRA) model.
///
/// Models on-off gain, net gain (accounting for signal loss), effective length,
/// noise figure and gain spectrum for single or dual-pump DRA.
#[derive(Debug, Clone)]
pub struct RamanAmplifier {
    /// Fiber length (km).
    pub fiber_length_km: f64,
    /// Pump wavelengths (m).
    pub pump_wavelengths: Vec<f64>,
    /// Pump powers (mW).
    pub pump_powers: Vec<f64>,
    /// Signal wavelength (m).
    pub signal_wavelength: f64,
    /// Pump propagation direction.
    pub pump_direction: PumpDirection,
    /// Fiber type.
    pub fiber_type: RamanFiberType,
}

impl RamanAmplifier {
    /// Single-pump DRA constructor.
    pub fn new_single_pump(pump_wl: f64, pump_mw: f64, signal_wl: f64, length_km: f64) -> Self {
        Self {
            fiber_length_km: length_km,
            pump_wavelengths: vec![pump_wl],
            pump_powers: vec![pump_mw],
            signal_wavelength: signal_wl,
            pump_direction: PumpDirection::Counterpropagating,
            fiber_type: RamanFiberType::Smf28,
        }
    }

    /// Dual-pump DRA constructor (for flatter gain spectrum).
    pub fn new_dual_pump(
        pump_wls: [f64; 2],
        pump_mws: [f64; 2],
        signal_wl: f64,
        length_km: f64,
    ) -> Self {
        Self {
            fiber_length_km: length_km,
            pump_wavelengths: pump_wls.to_vec(),
            pump_powers: pump_mws.to_vec(),
            signal_wavelength: signal_wl,
            pump_direction: PumpDirection::Counterpropagating,
            fiber_type: RamanFiberType::Smf28,
        }
    }

    /// Raman frequency shift (cm⁻¹) between pump `pump_idx` and signal.
    pub fn raman_shift_cm_inv(&self, pump_idx: usize) -> f64 {
        if pump_idx >= self.pump_wavelengths.len() {
            return 0.0;
        }
        let pump_wl = self.pump_wavelengths[pump_idx];
        // Wavenumber difference: ν̃ = (1/λ_pump - 1/λ_signal) in cm⁻¹
        let nu_pump_cm = 1.0 / (pump_wl * 100.0); // 1/m → 1/cm
        let nu_sig_cm = 1.0 / (self.signal_wavelength * 100.0);
        nu_pump_cm - nu_sig_cm
    }

    /// Lorentzian Raman gain shape factor at shift Ω (cm⁻¹) relative to peak.
    ///
    /// g_shape(Ω) = (Γ/2)² / ((Ω - Ω_peak)² + (Γ/2)²)
    fn raman_shape_at_shift(&self, shift_cm_inv: f64) -> f64 {
        let gamma_half = RAMAN_LINEWIDTH_CM_INV / 2.0;
        let delta = shift_cm_inv - RAMAN_PEAK_SHIFT_CM_INV;
        (gamma_half * gamma_half) / (delta * delta + gamma_half * gamma_half)
    }

    /// Effective Raman gain exponent coefficient g_eff (m⁻¹) at the signal.
    ///
    /// `raman_gain_coefficient()` returns the bulk specific gain g_R/A_eff in (W·m)⁻¹.
    /// The total gain exponent per unit length is Σ_pumps g_bulk · P_pump · shape(Ω).
    pub fn raman_gain_coefficient_at_signal(&self) -> f64 {
        if self.pump_wavelengths.is_empty() {
            return 0.0;
        }
        // g_bulk = g_R / A_eff in (W·m)⁻¹ — no need to divide by A_eff again
        let g_bulk = self.fiber_type.raman_gain_coefficient();

        // Total effective gain coefficient summing all pumps (units: m⁻¹)
        let total: f64 = self
            .pump_wavelengths
            .iter()
            .zip(self.pump_powers.iter())
            .enumerate()
            .map(|(i, (&_pwl, &p_mw))| {
                let shift = self.raman_shift_cm_inv(i);
                let shape = self.raman_shape_at_shift(shift);
                let p_w = p_mw * 1e-3;
                g_bulk * p_w * shape
            })
            .sum();
        total
    }

    /// Effective fiber length L_eff = (1 - exp(-α·L)) / α (m).
    pub fn effective_length_m(&self) -> f64 {
        let alpha = self.fiber_type.loss_per_m();
        let l_m = self.fiber_length_km * 1e3;
        if alpha < 1e-12 {
            l_m
        } else {
            (1.0 - (-alpha * l_m).exp()) / alpha
        }
    }

    /// On-off Raman gain (dB), ignoring fiber loss.
    ///
    /// G_on_off = g_R(Ω) · P_pump · L_eff / A_eff
    pub fn on_off_gain_db(&self) -> f64 {
        let g_coeff = self.raman_gain_coefficient_at_signal(); // 1/(m·W) * W = 1/m
        let l_eff = self.effective_length_m();
        // g_coeff already includes P_pump, so: G = exp(g_coeff * L_eff)
        let gain_linear = (g_coeff * l_eff).exp();
        10.0 * gain_linear.log10()
    }

    /// Net Raman gain (dB): on-off gain minus signal attenuation.
    ///
    /// G_net = G_on_off - α_signal · L
    pub fn net_gain_db(&self) -> f64 {
        let alpha_db_km = self.fiber_type.loss_db_per_km_at_1550();
        let signal_loss_db = alpha_db_km * self.fiber_length_km;
        self.on_off_gain_db() - signal_loss_db
    }

    /// Noise figure of the Raman amplifier (dB).
    ///
    /// For counter-pumped DRA:
    ///   NF ≈ -G_net_linear_dB + NF_phonon
    /// A simplified model: NF ≈ α·L - G_on_off + phonon contribution (~3 dB).
    pub fn noise_figure_db(&self) -> f64 {
        let alpha_db_km = self.fiber_type.loss_db_per_km_at_1550();
        let loss_db = alpha_db_km * self.fiber_length_km;
        // Effective NF for backward-pumped DRA (Bromage model):
        // NF_eff ≈ 10*log10(2*n_sp * (1 - 1/G)) + loss_db
        // Simplified: NF ≈ -G_on_off_dB + 2*loss_db + 3 (phonon noise)
        let g_on_off = self.on_off_gain_db();
        let nf = 2.0 * loss_db - g_on_off + 3.0;
        // Clamp to physically meaningful range
        nf.max(0.1)
    }

    /// Raman gain spectrum at arbitrary wavelength (dB/km).
    ///
    /// Evaluates the Lorentzian gain profile from each pump.
    /// `raman_gain_coefficient()` is already g_R/A_eff in (W·m)⁻¹.
    pub fn gain_at_wavelength(&self, wavelength: f64) -> f64 {
        if self.pump_wavelengths.is_empty() {
            return 0.0;
        }
        // g_bulk in (W·m)⁻¹, convert to (W·km)⁻¹ for dB/km output
        let g_bulk_per_w_km = self.fiber_type.raman_gain_coefficient() * 1e3;

        let total_gain_coeff_nep_per_km: f64 = self
            .pump_wavelengths
            .iter()
            .zip(self.pump_powers.iter())
            .map(|(&pump_wl, &p_mw)| {
                let nu_pump_cm = 1.0 / (pump_wl * 100.0);
                let nu_sig_cm = 1.0 / (wavelength * 100.0);
                let shift = nu_pump_cm - nu_sig_cm;
                let shape = self.raman_shape_at_shift(shift);
                let p_w = p_mw * 1e-3;
                // gain coefficient [km⁻¹] = g_bulk [(W·km)⁻¹] * P [W] * shape
                g_bulk_per_w_km * p_w * shape
            })
            .sum();
        // convert neper/km → dB/km
        total_gain_coeff_nep_per_km * 10.0 / f64::ln(10.0)
    }

    /// Pump power required for transparency (net gain = 0 dB) in mW.
    ///
    /// Solves G_on_off = α·L, i.e. g_bulk · P_pump · L_eff = α·L.
    pub fn transparency_pump_power_mw(&self) -> f64 {
        let alpha = self.fiber_type.loss_per_m();
        let l_m = self.fiber_length_km * 1e3;
        let l_eff = self.effective_length_m();
        // g_bulk = g_R / A_eff in (W·m)⁻¹
        let g_bulk = self.fiber_type.raman_gain_coefficient();

        // Gain shape at signal (use first pump or default to peak)
        let shape = if self.pump_wavelengths.is_empty() {
            1.0
        } else {
            self.raman_shape_at_shift(self.raman_shift_cm_inv(0))
        };

        if g_bulk * shape * l_eff < 1e-20 {
            return f64::INFINITY;
        }
        // Required: g_bulk * P * L_eff = alpha * L
        let p_w = alpha * l_m / (g_bulk * shape * l_eff);
        p_w * 1e3 // W → mW
    }

    /// Double Rayleigh backscattering (DRB) penalty (dB).
    ///
    /// DRB ≈ 2 * α² * L² * G_on_off (very rough estimate).
    /// More accurate models require integration of backscattered fields.
    pub fn drb_penalty_db(&self) -> f64 {
        let rayleigh_coeff = 1.5e-4_f64; // typical Rayleigh loss fraction for SMF-28
        let _l_m = self.fiber_length_km * 1e3;
        let g_linear = 10.0_f64.powf(self.on_off_gain_db() / 10.0);
        // DRB power relative to signal ≈ (α_R * L_eff)² * G
        let l_eff = self.effective_length_m();
        let drb_ratio = (rayleigh_coeff * l_eff).powi(2) * g_linear;
        if drb_ratio <= 0.0 {
            return 0.0;
        }
        -10.0 * drb_ratio.log10() // convert to dB penalty
    }
}

// ─── Lumped Raman Amplifier ───────────────────────────────────────────────────

/// Lumped (discrete) Raman amplifier using short, high-confinement fiber.
///
/// Uses HNLF or DCF to achieve Raman gain over a short length (~100 m–2 km),
/// suitable for on-chip or rack-scale applications.
#[derive(Debug, Clone)]
pub struct LumpedRamanAmplifier {
    /// Active fiber length (m).
    pub fiber_length_m: f64,
    /// Pump power (W).
    pub pump_power_w: f64,
    /// Signal wavelength (m).
    pub signal_wavelength: f64,
    /// Fiber type (should be HNLF or DCF for lumped operation).
    pub fiber_type: RamanFiberType,
}

impl LumpedRamanAmplifier {
    /// Construct a lumped Raman amplifier with HNLF.
    pub fn new_hnlf(length_m: f64, pump_w: f64, signal_wl: f64) -> Self {
        Self {
            fiber_length_m: length_m,
            pump_power_w: pump_w,
            signal_wavelength: signal_wl,
            fiber_type: RamanFiberType::Hnlf,
        }
    }

    /// Effective length L_eff = (1 - exp(-α·L)) / α (m).
    pub fn effective_length_m(&self) -> f64 {
        let alpha = self.fiber_type.loss_per_m();
        if alpha < 1e-12 {
            self.fiber_length_m
        } else {
            (1.0 - (-alpha * self.fiber_length_m).exp()) / alpha
        }
    }

    /// On-off Raman gain (dB).
    ///
    /// `raman_gain_coefficient()` returns the bulk specific gain g_R/A_eff in (W·m)⁻¹,
    /// so the gain exponent is simply (g_R/A_eff) · P_pump · L_eff.
    pub fn gain_db(&self) -> f64 {
        // g_bulk = g_R / A_eff in (W·m)⁻¹
        let g_bulk = self.fiber_type.raman_gain_coefficient();
        let l_eff = self.effective_length_m();
        // Assume pump wavelength is at the Raman peak shift relative to signal
        let gain_linear = (g_bulk * self.pump_power_w * l_eff).exp();
        10.0 * gain_linear.log10()
    }

    /// Noise figure (dB) — simplified phonon noise model for forward-pumped case.
    pub fn noise_figure_db(&self) -> f64 {
        // Forward-pumped lumped Raman has higher NF than backward-pumped DRA
        // NF ≈ loss_before_gain + phonon noise ≈ 2*(α_dB*L) + 3 dB
        let loss_db = self.fiber_type.loss_db_per_km_at_1550() * self.fiber_length_m * 1e-3;
        (2.0 * loss_db + 3.0).max(3.0)
    }

    /// Stimulated Raman scattering threshold pump power (W).
    ///
    /// Threshold criterion: g_R · P_th · L_eff / A_eff = 16 (Agrawal criterion).
    pub fn threshold_power_w(&self) -> f64 {
        let g_r = self.fiber_type.raman_gain_coefficient();
        let a_eff_m2 = self.fiber_type.effective_area_um2() * 1e-12;
        let l_eff = self.effective_length_m();
        if g_r * l_eff < 1e-30 {
            return f64::INFINITY;
        }
        16.0 * a_eff_m2 / (g_r * l_eff)
    }
}

// ─── Utility ─────────────────────────────────────────────────────────────────

/// Convert wavelength (m) to Raman shift (cm⁻¹) relative to a pump wavelength.
pub fn wavelength_to_raman_shift(signal_wl: f64, pump_wl: f64) -> f64 {
    let nu_pump = 1.0 / (pump_wl * 100.0); // cm⁻¹
    let nu_sig = 1.0 / (signal_wl * 100.0); // cm⁻¹
    nu_pump - nu_sig
}

/// Stokes wavelength for a given pump wavelength and Raman shift (cm⁻¹).
pub fn stokes_wavelength(pump_wl: f64, shift_cm_inv: f64) -> f64 {
    let nu_pump_cm = 1.0 / (pump_wl * 100.0);
    let nu_stokes_cm = nu_pump_cm - shift_cm_inv;
    if nu_stokes_cm <= 0.0 {
        return f64::INFINITY;
    }
    1.0 / (nu_stokes_cm * 100.0)
}

/// Photon energy at wavelength λ (m) in Joules: E = h·c/λ.
pub fn photon_energy_j(wavelength: f64) -> f64 {
    H_PLANCK * C_LIGHT / wavelength
}

// ─── Raman gain spectrum (standalone) ────────────────────────────────────────

/// Evaluate normalised Raman gain profile g(Ω)/g_max for silica fiber.
///
/// Uses a Lorentzian profile centred at the dominant 440 cm⁻¹ peak.
pub fn raman_gain_profile(shift_cm_inv: f64) -> f64 {
    let gamma_half = RAMAN_LINEWIDTH_CM_INV / 2.0;
    let delta = shift_cm_inv - RAMAN_PEAK_SHIFT_CM_INV;
    (gamma_half * gamma_half) / (delta * delta + gamma_half * gamma_half)
}

/// Compute Raman gain bandwidth at half-maximum (cm⁻¹).
pub fn raman_bandwidth_fwhm_cm_inv() -> f64 {
    RAMAN_LINEWIDTH_CM_INV
}

// ─── Validate Raman shift conversion ─────────────────────────────────────────
fn _stokes_wavelength_nm(pump_nm: f64, shift_cm_inv: f64) -> f64 {
    stokes_wavelength(pump_nm * 1e-9, shift_cm_inv) * 1e9
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_raman_gain_positive_for_adequate_pump() {
        // 1 W pump at 1455 nm → signal at 1550 nm, 80 km SMF-28
        let dra = RamanAmplifier::new_single_pump(1455e-9, 1000.0, 1550e-9, 80.0);
        let g = dra.on_off_gain_db();
        assert!(g > 0.0, "On-off gain should be positive; got {g}");
    }

    #[test]
    fn test_effective_length_less_than_physical_length() {
        let dra = RamanAmplifier::new_single_pump(1455e-9, 500.0, 1550e-9, 80.0);
        let l_eff = dra.effective_length_m();
        let l_phys = 80.0 * 1e3;
        assert!(
            l_eff < l_phys,
            "Effective length {l_eff} must be less than physical length {l_phys}"
        );
    }

    #[test]
    fn test_stokes_wavelength_relation() {
        // 1455 nm pump + 440 cm⁻¹ shift → ~1550 nm Stokes
        let stokes = stokes_wavelength(1455e-9, 440.0);
        // Should be in range 1540–1570 nm
        assert!(
            stokes > 1540e-9 && stokes < 1580e-9,
            "Stokes wavelength {:.1} nm out of expected range",
            stokes * 1e9
        );
    }

    #[test]
    fn test_raman_profile_peak_at_440() {
        let peak_val = raman_gain_profile(440.0);
        let off_val = raman_gain_profile(0.0);
        assert_abs_diff_eq!(peak_val, 1.0, epsilon = 1e-12);
        assert!(
            off_val < peak_val,
            "Profile must be less than 1 away from peak"
        );
    }

    #[test]
    fn test_transparency_pump_power_finite() {
        let dra = RamanAmplifier::new_single_pump(1455e-9, 100.0, 1550e-9, 80.0);
        let p_trans = dra.transparency_pump_power_mw();
        assert!(
            p_trans.is_finite() && p_trans > 0.0,
            "Transparency power must be finite and positive; got {p_trans}"
        );
    }

    #[test]
    fn test_lumped_raman_gain_increases_with_pump() {
        let amp_low = LumpedRamanAmplifier::new_hnlf(500.0, 0.5, 1550e-9);
        let amp_high = LumpedRamanAmplifier::new_hnlf(500.0, 2.0, 1550e-9);
        assert!(
            amp_high.gain_db() > amp_low.gain_db(),
            "Higher pump power must produce more gain"
        );
    }

    #[test]
    fn test_dual_pump_raman_shift() {
        let dra = RamanAmplifier::new_dual_pump([1430e-9, 1455e-9], [500.0, 500.0], 1550e-9, 80.0);
        let shift0 = dra.raman_shift_cm_inv(0);
        let shift1 = dra.raman_shift_cm_inv(1);
        // Both pumps should have positive Raman shift (pump at shorter wavelength)
        assert!(
            shift0 > 0.0,
            "Raman shift for pump 0 must be positive; got {shift0}"
        );
        assert!(
            shift1 > 0.0,
            "Raman shift for pump 1 must be positive; got {shift1}"
        );
    }

    #[test]
    fn test_fiber_type_loss_coefficients() {
        let smf = RamanFiberType::Smf28;
        let dcf = RamanFiberType::Dcf;
        assert!(
            dcf.loss_db_per_km_at_1550() > smf.loss_db_per_km_at_1550(),
            "DCF must have higher loss than SMF-28"
        );
    }

    #[test]
    fn test_raman_gain_coefficient_units() {
        // SMF-28 g_R = 0.4 (W·km)⁻¹ = 4e-4 (W·m)⁻¹
        let smf = RamanFiberType::Smf28;
        assert_abs_diff_eq!(smf.raman_gain_coefficient(), 4e-4, epsilon = 1e-10);
    }

    #[test]
    fn test_net_gain_less_than_on_off_gain() {
        let dra = RamanAmplifier::new_single_pump(1455e-9, 1000.0, 1550e-9, 80.0);
        let on_off = dra.on_off_gain_db();
        let net = dra.net_gain_db();
        assert!(
            net < on_off,
            "Net gain {net} must be less than on-off gain {on_off}"
        );
    }
}
