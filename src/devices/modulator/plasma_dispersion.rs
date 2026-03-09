//! Silicon plasma dispersion modulator model.
//!
//! Implements Soref-Bennett equations for free-carrier-induced refractive
//! index change and absorption in silicon at telecom wavelengths.
//! Also models PIN diode carrier dynamics.

use crate::error::OxiPhotonError;
use std::f64::consts::PI;

// Physical constants
const Q_E: f64 = 1.602_176_634e-19; // Elementary charge (C)
const K_B: f64 = 1.380_649e-23; // Boltzmann constant (J/K)
const EPS0: f64 = 8.854_187_817e-12; // Permittivity of free space (F/m)
const EPS_SI: f64 = 11.7; // Silicon relative permittivity (RF)
const NI_SI: f64 = 1.0e10; // Silicon intrinsic carrier density (cm⁻³)

/// Silicon plasma dispersion model for electro-optic modulation.
///
/// Based on Soref-Bennett (1987) empirical equations at λ = 1.55 μm:
///   Δn = -[8.8×10⁻²² · ΔNe + 8.5×10⁻¹⁸ · ΔNh^0.8]
///   Δα = [8.5×10⁻¹⁸ · ΔNe + 6.0×10⁻¹⁸ · ΔNh]  (cm⁻¹)
#[derive(Debug, Clone)]
pub struct SiPlasmaDispersion {
    /// Operating wavelength (m)
    pub wavelength: f64,
    /// Excess electron concentration (cm⁻³)
    pub delta_ne: f64,
    /// Excess hole concentration (cm⁻³)
    pub delta_nh: f64,
    /// Optical confinement factor Γ (0 to 1)
    pub confinement: f64,
    /// Modulator length (m)
    pub length: f64,
}

impl SiPlasmaDispersion {
    /// Create a new Si plasma dispersion modulator (zero carriers initially).
    pub fn new(wavelength: f64, confinement: f64, length: f64) -> Self {
        Self {
            wavelength,
            delta_ne: 0.0,
            delta_nh: 0.0,
            confinement: confinement.clamp(0.0, 1.0),
            length,
        }
    }

    /// Set carrier concentrations (builder pattern).
    pub fn with_carriers(mut self, delta_ne: f64, delta_nh: f64) -> Self {
        self.delta_ne = delta_ne;
        self.delta_nh = delta_nh;
        self
    }

    /// Index change due to free carriers (Soref-Bennett at λ=1.55 μm).
    ///
    /// Δn = -[8.8×10⁻²² · ΔNe + 8.5×10⁻¹⁸ · ΔNh^0.8]
    pub fn delta_n(&self) -> f64 {
        -(8.8e-22 * self.delta_ne + 8.5e-18 * self.delta_nh.powf(0.8))
    }

    /// Loss change due to free-carrier absorption (cm⁻¹).
    ///
    /// Δα = 8.5×10⁻¹⁸ · ΔNe + 6.0×10⁻¹⁸ · ΔNh
    pub fn delta_alpha_per_cm(&self) -> f64 {
        8.5e-18 * self.delta_ne + 6.0e-18 * self.delta_nh
    }

    /// Effective mode index change: Δn_eff = Γ · Δn.
    pub fn delta_n_eff(&self) -> f64 {
        self.confinement * self.delta_n()
    }

    /// Phase shift: Δφ = 2π/λ · Δn_eff · L (rad).
    pub fn phase_shift_rad(&self) -> f64 {
        2.0 * PI / self.wavelength * self.delta_n_eff() * self.length
    }

    /// Insertion loss from free-carrier absorption (dB).
    ///
    /// IL = 4.343 · Γ · Δα[m⁻¹] · L
    /// where Δα[m⁻¹] = Δα[cm⁻¹] · 100
    pub fn fca_loss_db(&self) -> f64 {
        let alpha_m = self.delta_alpha_per_cm() * 100.0; // cm⁻¹ → m⁻¹
        4.343 * self.confinement * alpha_m * self.length
    }

    /// Voltage-phase efficiency V_π · L (V·m) for a PIN-diode modulator.
    ///
    /// Given the external V_pi (at which one π phase shift occurs), returns V_pi·L.
    pub fn v_pi_l(&self, v_pi: f64) -> f64 {
        v_pi * self.length
    }

    /// Modulation bandwidth from carrier lifetime τ (s): f_3dB = 1/(2πτ).
    pub fn bandwidth_3db(tau_s: f64) -> f64 {
        if tau_s <= 0.0 {
            return f64::INFINITY;
        }
        1.0 / (2.0 * PI * tau_s)
    }

    /// Energy per bit (fJ/bit) for a capacitive modulator: E = C·Vpp²/4.
    ///
    /// `capacitance_ff` is in femtofarads, `v_pp` is the peak-to-peak voltage swing.
    pub fn energy_per_bit_fj(capacitance_ff: f64, v_pp: f64) -> f64 {
        let c_f = capacitance_ff * 1e-15; // fF → F
        c_f * v_pp * v_pp / 4.0 * 1e15 // J → fJ
    }

    /// Full Soref-Bennett model at arbitrary wavelength (empirical scaling).
    ///
    /// Scales the 1550 nm coefficients approximately with wavelength:
    ///   Δn ∝ λ² (from Drude model)
    ///   Δα ∝ λ² (free-carrier absorption)
    ///
    /// Returns (Δn, Δα_cm) at the given wavelength (μm).
    pub fn soref_bennett(wavelength_um: f64, delta_ne: f64, delta_nh: f64) -> (f64, f64) {
        // Scale factor relative to 1.55 μm
        let scale = (wavelength_um / 1.55).powi(2);
        let dn = -(8.8e-22 * delta_ne + 8.5e-18 * delta_nh.powf(0.8)) * scale;
        let dalpha = (8.5e-18 * delta_ne + 6.0e-18 * delta_nh) * scale;
        (dn, dalpha)
    }

    /// Silicon refractive index vs wavelength (Sellmeier approximation).
    ///
    /// Valid approximately 1.2–5 μm for crystalline Si.
    pub fn si_refractive_index(wavelength_um: f64) -> f64 {
        // Simplified Sellmeier for Si (Li 1993)
        let lsq = wavelength_um * wavelength_um;
        let n_sq = 1.0
            + 10.6684 * lsq / (lsq - 0.091302)
            + 0.003043 * lsq / (lsq - 1.13475)
            + 1.5413 * lsq / (lsq - 1104.0);
        n_sq.max(1.0).sqrt()
    }

    /// Extinction ratio in dB for MZI modulator with this phase shift.
    ///
    /// For an ideal MZI: ER = 10·log10(|1 + exp(i·Δφ)|²) → ∞ at Δφ = π
    pub fn mzi_extinction_ratio_db(&self) -> f64 {
        let dphi = self.phase_shift_rad().abs();
        // Transmission through MZI: T = cos²(Δφ/2)
        let t_on = (dphi / 2.0).cos().powi(2); // at 0 phase (max)
        let t_off = (dphi / 2.0 + PI / 2.0).cos().powi(2); // at Δφ phase
        if t_off < 1e-30 {
            return 60.0; // very high ER
        }
        let er = t_on / t_off;
        10.0 * er.log10()
    }
}

/// Carrier dynamics model for PIN diode modulator.
#[derive(Debug, Clone)]
pub struct PinDiodeModel {
    /// Intrinsic region width (μm)
    pub i_region_width: f64,
    /// Modulator length (μm)
    pub length: f64,
    /// N-region doping (cm⁻³)
    pub n_doping: f64,
    /// P-region doping (cm⁻³)
    pub p_doping: f64,
    /// Minority carrier lifetime (ns)
    pub carrier_lifetime: f64,
    /// Electron mobility (cm²/V·s)
    pub mobility_e: f64,
    /// Hole mobility (cm²/V·s)
    pub mobility_h: f64,
}

impl PinDiodeModel {
    /// Typical silicon PIN diode parameters for photonic modulators.
    pub fn silicon() -> Self {
        Self {
            i_region_width: 0.5,   // 500 nm i-region
            length: 1000.0,        // 1 mm modulator
            n_doping: 1e18,        // N+ region
            p_doping: 1e18,        // P+ region
            carrier_lifetime: 1.0, // 1 ns lifetime
            mobility_e: 1400.0,    // Si electron mobility
            mobility_h: 450.0,     // Si hole mobility
        }
    }

    /// Built-in voltage V_bi ≈ (kT/q) · ln(Na·Nd/ni²) at 300 K.
    pub fn built_in_voltage(&self) -> f64 {
        let vt = K_B * 300.0 / Q_E; // thermal voltage ≈ 0.026 V
        vt * (self.n_doping * self.p_doping / (NI_SI * NI_SI)).ln()
    }

    /// Depletion width at bias voltage V (m).
    ///
    /// W_dep = W_i + sqrt(2·ε_Si·(V_bi - V)/(q·N_eff))
    /// where N_eff = Na·Nd/(Na+Nd) is the effective doping.
    pub fn depletion_width(&self, v_bias: f64) -> f64 {
        let vbi = self.built_in_voltage();
        let v_total = vbi - v_bias; // reverse bias increases depletion
        if v_total <= 0.0 {
            return self.i_region_width * 1e-6;
        }
        let n_eff = self.n_doping * self.p_doping / (self.n_doping + self.p_doping);
        let w_extra = (2.0 * EPS0 * EPS_SI * v_total / (Q_E * n_eff * 1e6)).sqrt(); // in m
        self.i_region_width * 1e-6 + w_extra
    }

    /// Carrier concentration in the i-region at current I (mA).
    ///
    /// Approximates injected carrier density for forward-biased PIN:
    ///   ΔN ≈ τ · I / (q · V_i)
    /// where V_i = i_region_width × length (active volume).
    ///
    /// Returns (ΔNe, ΔNh) in cm⁻³.
    pub fn carrier_concentration(&self, current_ma: f64) -> (f64, f64) {
        let i_a = current_ma * 1e-3; // mA → A
        let tau_s = self.carrier_lifetime * 1e-9; // ns → s
                                                  // Active volume in cm³
        let volume_cm3 = self.i_region_width * 1e-4 // μm → cm
            * self.length * 1e-4               // μm → cm
            * self.i_region_width * 1e-4; // thickness ≈ i_width (square cross-section)
        if volume_cm3 <= 0.0 {
            return (0.0, 0.0);
        }
        let delta_n = tau_s * i_a / (Q_E * volume_cm3);
        (delta_n, delta_n) // equal electron and hole injection in i-region
    }

    /// Junction capacitance per unit length (fF/μm) at bias V.
    ///
    /// C = ε_Si · ε₀ · A / W_dep(V)
    /// normalized per μm of modulator length.
    pub fn junction_capacitance_ff_per_um(&self, v_bias: f64) -> f64 {
        let w_dep = self.depletion_width(v_bias); // m
        let area_per_um = self.i_region_width * 1e-6 * 1e-6; // i_width (m) × 1 μm length (m)
        let c_f_per_um = EPS0 * EPS_SI * area_per_um / w_dep;
        c_f_per_um * 1e15 // F/m → fF/μm
    }

    /// RC bandwidth (GHz) given series resistance Rs (Ω).
    ///
    /// f_3dB = 1 / (2π · Rs · C_total)
    /// where C_total = capacitance_per_um × length.
    pub fn rc_bandwidth_ghz(&self, v_bias: f64, rs_ohm: f64) -> f64 {
        let c_per_um = self.junction_capacitance_ff_per_um(v_bias); // fF/μm
        let c_total_f = c_per_um * 1e-15 * self.length; // fF/μm × μm × 1e-15 = F
        if c_total_f <= 0.0 || rs_ohm <= 0.0 {
            return f64::INFINITY;
        }
        1.0 / (2.0 * PI * rs_ohm * c_total_f) * 1e-9 // Hz → GHz
    }
}

/// Error type for modulator calculations (thin wrapper).
#[allow(dead_code)]
fn check_positive(val: f64, name: &str) -> Result<f64, OxiPhotonError> {
    if val > 0.0 {
        Ok(val)
    } else {
        Err(OxiPhotonError::NumericalError(format!(
            "{name} must be positive, got {val}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn modulator_1e17() -> SiPlasmaDispersion {
        SiPlasmaDispersion::new(1.55e-6, 0.8, 1e-3).with_carriers(1e17, 1e17)
    }

    #[test]
    fn delta_n_negative_for_free_carriers() {
        let m = modulator_1e17();
        assert!(m.delta_n() < 0.0, "delta_n = {}", m.delta_n());
    }

    #[test]
    fn delta_alpha_positive_for_free_carriers() {
        let m = modulator_1e17();
        assert!(m.delta_alpha_per_cm() > 0.0);
    }

    #[test]
    fn delta_n_eff_scales_with_confinement() {
        let m1 = SiPlasmaDispersion::new(1.55e-6, 1.0, 1e-3).with_carriers(1e17, 1e17);
        let m2 = SiPlasmaDispersion::new(1.55e-6, 0.5, 1e-3).with_carriers(1e17, 1e17);
        assert_relative_eq!(m1.delta_n_eff(), 2.0 * m2.delta_n_eff(), epsilon = 1e-15);
    }

    #[test]
    fn phase_shift_proportional_to_length() {
        let m1 = SiPlasmaDispersion::new(1.55e-6, 0.8, 1e-3).with_carriers(1e17, 1e17);
        let m2 = SiPlasmaDispersion::new(1.55e-6, 0.8, 2e-3).with_carriers(1e17, 1e17);
        assert_relative_eq!(
            m2.phase_shift_rad(),
            2.0 * m1.phase_shift_rad(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn fca_loss_positive() {
        let m = modulator_1e17();
        assert!(m.fca_loss_db() > 0.0);
    }

    #[test]
    fn bandwidth_from_lifetime() {
        let tau = 1e-9; // 1 ns
        let bw = SiPlasmaDispersion::bandwidth_3db(tau);
        // f_3dB ≈ 159 MHz
        assert!((bw - 159_154_943.0).abs() / bw < 1e-5, "bw = {bw}");
    }

    #[test]
    fn energy_per_bit_formula() {
        // C = 10 fF, Vpp = 2 V → E = 10e-15 * 4 / 4 = 10 fJ
        let e = SiPlasmaDispersion::energy_per_bit_fj(10.0, 2.0);
        assert_relative_eq!(e, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn soref_bennett_scales_with_wavelength() {
        let (dn1, _) = SiPlasmaDispersion::soref_bennett(1.55, 1e17, 1e17);
        let (dn2, _) = SiPlasmaDispersion::soref_bennett(1.3, 1e17, 1e17);
        // At longer wavelength, |Δn| is larger (λ² scaling)
        assert!(dn1.abs() > dn2.abs());
    }

    #[test]
    fn si_refractive_index_at_1550nm() {
        let n = SiPlasmaDispersion::si_refractive_index(1.55);
        assert!(n > 3.4 && n < 3.6, "n_Si = {n}");
    }

    #[test]
    fn mzi_extinction_ratio_positive() {
        let m = modulator_1e17();
        let er = m.mzi_extinction_ratio_db();
        assert!(er.is_finite(), "ER = {er}");
    }

    #[test]
    fn pin_builtin_voltage_reasonable() {
        let pin = PinDiodeModel::silicon();
        let vbi = pin.built_in_voltage();
        // Si P+N+ junction: Vbi ≈ 0.85-0.95 V
        assert!(vbi > 0.7 && vbi < 1.2, "Vbi = {vbi}");
    }

    #[test]
    fn pin_depletion_width_decreases_with_forward_bias() {
        let pin = PinDiodeModel::silicon();
        let w0 = pin.depletion_width(0.0);
        let w_fwd = pin.depletion_width(0.5);
        assert!(w_fwd < w0, "w(0)={w0:.2e}, w(0.5V)={w_fwd:.2e}");
    }

    #[test]
    fn pin_carrier_concentration_scales_with_current() {
        let pin = PinDiodeModel::silicon();
        let (n1, _) = pin.carrier_concentration(1.0);
        let (n2, _) = pin.carrier_concentration(2.0);
        assert_relative_eq!(n2, 2.0 * n1, epsilon = 1e-10);
    }

    #[test]
    fn pin_rc_bandwidth_positive() {
        let pin = PinDiodeModel::silicon();
        let bw = pin.rc_bandwidth_ghz(-2.0, 50.0);
        assert!(bw > 0.0 && bw.is_finite(), "BW = {bw} GHz");
    }
}
