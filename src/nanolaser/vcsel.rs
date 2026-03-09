//! VCSEL (Vertical-Cavity Surface-Emitting Laser) physics model.
//!
//! Models oxide-aperture VCSELs for 850 nm (GaAs/AlGaAs DBR), 1310 nm (InP-based),
//! and 1550 nm (GaAsSb/AlGaAsSb) wavelength bands. Includes:
//!
//! - DBR reflectivity and effective penetration depth
//! - Photon lifetime from cavity Q
//! - Threshold current and slope efficiency
//! - Thermal resistance and self-heating
//! - Far-field divergence, single-mode condition
//! - VCSEL array models with thermal/optical crosstalk
//!
//! # References
//!
//! - K. Iga, "Surface-emitting laser — its birth and generation of new optoelectronics field",
//!   IEEE J. Sel. Topics Quantum Electron. 6, 1201 (2000).
//! - R. Michalzik (Ed.), "VCSELs", Springer 2013.

use super::rate_equations::GeneralizedRateEquations;
use std::f64::consts::PI;

/// Speed of light in vacuum (m/s)
const C_LIGHT: f64 = 2.997_924_58e8;

// ─── DbrMaterial ─────────────────────────────────────────────────────────────

/// DBR mirror material system with associated refractive index pair.
#[derive(Debug, Clone)]
pub enum DbrMaterial {
    /// 850 nm GaAs/Al₀.₉Ga₀.₁As VCSEL DBR.
    AlGaAs850nm,
    /// 1310 nm InAlGaAs/InAlAs VCSEL DBR (InP substrate).
    InAlGaAs1310,
    /// 1550 nm GaAsSb/AlAsSb VCSEL DBR.
    GaAsSb1550,
    /// Custom DBR with user-specified refractive indices and bulk reflectivity.
    Custom {
        /// High-index layer refractive index.
        n_high: f64,
        /// Low-index layer refractive index.
        n_low: f64,
        /// Specified mirror reflectivity (overrides formula if > 0).
        reflectivity: f64,
    },
}

impl DbrMaterial {
    /// Returns (n_high, n_low) for the material system.
    pub fn refractive_indices(&self) -> (f64, f64) {
        match self {
            DbrMaterial::AlGaAs850nm => (3.64, 2.97), // GaAs / Al₀.₉Ga₀.₁As at 850 nm
            DbrMaterial::InAlGaAs1310 => (3.52, 3.09), // InGaAlAs / InAlAs
            DbrMaterial::GaAsSb1550 => (3.51, 2.95),  // GaAsSb / AlAsSb
            DbrMaterial::Custom { n_high, n_low, .. } => (*n_high, *n_low),
        }
    }

    /// DBR power reflectivity for `n_pairs` quarter-wave layer pairs.
    ///
    /// R = \[(n_h/n_l)^{2N} − 1\]² / \[(n_h/n_l)^{2N} + 1\]²
    pub fn dbr_reflectivity(&self, n_pairs: usize) -> f64 {
        if let DbrMaterial::Custom { reflectivity, .. } = self {
            if *reflectivity > 0.0 {
                return reflectivity.clamp(0.0, 1.0);
            }
        }
        let (n_h, n_l) = self.refractive_indices();
        let ratio = (n_h / n_l).powi(2 * n_pairs as i32);
        let r = ((ratio - 1.0) / (ratio + 1.0)).powi(2);
        r.clamp(0.0, 1.0)
    }

    /// DBR stopband width Δλ (nm).
    ///
    /// Δλ/λ = (4/π) · arcsin((n_h − n_l)/(n_h + n_l))
    pub fn dbr_bandwidth_nm(&self) -> f64 {
        let (n_h, n_l) = self.refractive_indices();
        let wavelength_nm = match self {
            DbrMaterial::AlGaAs850nm => 850.0,
            DbrMaterial::InAlGaAs1310 => 1310.0,
            DbrMaterial::GaAsSb1550 => 1550.0,
            DbrMaterial::Custom { .. } => 1000.0,
        };
        let delta_lambda_over_lambda = (4.0 / PI) * ((n_h - n_l) / (n_h + n_l)).asin();
        wavelength_nm * delta_lambda_over_lambda
    }

    /// Effective extra cavity length from DBR penetration (in units of pairs).
    ///
    /// L_pen = λ/(4·(n_h − n_l))  (quarter-wave stack penetration depth)
    pub fn penetration_depth_pairs(&self) -> f64 {
        let (n_h, n_l) = self.refractive_indices();
        1.0 / (4.0 * (n_h - n_l) / (n_h + n_l))
    }
}

// ─── Vcsel ────────────────────────────────────────────────────────────────────

/// VCSEL device model with oxide aperture confinement.
#[derive(Debug, Clone)]
pub struct Vcsel {
    /// Oxide aperture (active region) diameter (μm).
    pub active_diameter_um: f64,
    /// Physical cavity length (nm), typically one optical wavelength.
    pub cavity_length_nm: f64,
    /// Number of top (output) DBR mirror pairs.
    pub n_top_dbr_pairs: usize,
    /// Number of bottom (high-reflectivity) DBR mirror pairs.
    pub n_bottom_dbr_pairs: usize,
    /// DBR material system.
    pub dbr_material: DbrMaterial,
    /// Emission wavelength (m).
    pub wavelength: f64,
    /// Underlying laser rate equation model.
    pub rate_eqs: GeneralizedRateEquations,
}

impl Vcsel {
    /// Standard 850 nm GaAs VCSEL (oxide-aperture, 980 nm pump not needed).
    pub fn new_850nm_standard(diameter_um: f64, injection_current_ma: f64) -> Self {
        let wavelength = 850e-9;
        let active_volume_m3 = PI * (diameter_um * 0.5e-6).powi(2) * 7e-9; // 7 nm active thickness
        let q_factor = 1000.0;
        let tau_ph_ps = q_factor * wavelength / (2.0 * PI * C_LIGHT) * 1e12;
        let mut gre = GeneralizedRateEquations::new_conventional_laser(injection_current_ma);
        gre.active_volume = active_volume_m3;
        gre.photon_lifetime_ps = tau_ph_ps;
        gre.injection_current_ua = injection_current_ma * 1e3;
        gre.beta_factor = 1e-3; // slightly higher β than edge emitter
        gre.group_velocity = C_LIGHT / 3.64;
        Self {
            active_diameter_um: diameter_um,
            cavity_length_nm: 255.0, // ~λ/n for 850 nm in GaAs
            n_top_dbr_pairs: 22,
            n_bottom_dbr_pairs: 35,
            dbr_material: DbrMaterial::AlGaAs850nm,
            wavelength,
            rate_eqs: gre,
        }
    }

    /// 1550 nm InGaAs VCSEL on GaAs substrate with metamorphic DBR.
    pub fn new_1550nm_ingaas(diameter_um: f64, injection_current_ma: f64) -> Self {
        let wavelength = 1550e-9;
        let active_volume_m3 = PI * (diameter_um * 0.5e-6).powi(2) * 10e-9;
        let q_factor = 2000.0;
        let tau_ph_ps = q_factor * wavelength / (2.0 * PI * C_LIGHT) * 1e12;
        let mut gre = GeneralizedRateEquations::new_conventional_laser(injection_current_ma);
        gre.active_volume = active_volume_m3;
        gre.photon_lifetime_ps = tau_ph_ps;
        gre.injection_current_ua = injection_current_ma * 1e3;
        gre.beta_factor = 1e-3;
        gre.group_velocity = C_LIGHT / 3.51;
        Self {
            active_diameter_um: diameter_um,
            cavity_length_nm: 455.0, // ~λ/n for 1550 nm in GaAsSb
            n_top_dbr_pairs: 18,
            n_bottom_dbr_pairs: 30,
            dbr_material: DbrMaterial::GaAsSb1550,
            wavelength,
            rate_eqs: gre,
        }
    }

    /// Active cross-sectional area A = π·(d/2)² (μm²).
    pub fn active_area_um2(&self) -> f64 {
        PI * (self.active_diameter_um * 0.5).powi(2)
    }

    /// Effective cavity length including DBR penetration (nm).
    ///
    /// L_eff = L_cav + 2·L_pen where L_pen = (λ/4)·(n_h+n_l)/(n_h−n_l).
    pub fn effective_cavity_length_nm(&self) -> f64 {
        let (n_h, n_l) = self.dbr_material.refractive_indices();
        let lambda_nm = self.wavelength * 1e9;
        let l_pen_each = (lambda_nm / 4.0) * (n_h + n_l) / (n_h - n_l);
        self.cavity_length_nm + 2.0 * l_pen_each
    }

    /// Photon lifetime τ_ph (ps) from round-trip loss.
    ///
    /// 1/τ_ph = c/n_g · (1/L_eff) · ln(1/(R_top·R_bot)) / 2
    pub fn photon_lifetime_ps(&self) -> f64 {
        let r_top = self.dbr_material.dbr_reflectivity(self.n_top_dbr_pairs);
        let r_bot = self.dbr_material.dbr_reflectivity(self.n_bottom_dbr_pairs);
        let (n_h, _) = self.dbr_material.refractive_indices();
        let l_eff_m = self.effective_cavity_length_nm() * 1e-9;
        let n_g = n_h;
        let mirror_loss = -0.5 * (r_top * r_bot).max(1e-20).ln() / l_eff_m;
        let tau_ph_s = n_g / (C_LIGHT * mirror_loss);
        tau_ph_s * 1e12
    }

    /// Threshold current I_th (mA).
    pub fn threshold_current_ma(&self) -> f64 {
        let mut gre = self.rate_eqs.clone();
        gre.photon_lifetime_ps = self.photon_lifetime_ps();
        gre.threshold_current_ua() * 1e-3
    }

    /// Slope efficiency dP/dI (mW/mA) above threshold.
    ///
    /// η_s = (hν/q) · Γ · (1 − R_top) / (1 − R_top·R_bot)
    pub fn slope_efficiency_mw_per_ma(&self) -> f64 {
        let r_top = self.dbr_material.dbr_reflectivity(self.n_top_dbr_pairs);
        let r_bot = self.dbr_material.dbr_reflectivity(self.n_bottom_dbr_pairs);
        let hnu_over_q = (6.626_070_15e-34 * C_LIGHT / self.wavelength) / 1.602_176_634e-19;
        let coupling = (1.0 - r_top) / (1.0 - r_top * r_bot);
        hnu_over_q * self.rate_eqs.confinement_factor * coupling * 1e3 // mW/mA = W/A
    }

    /// Output power P_out (mW) at injection current I (mA).
    pub fn output_power_mw(&self, current_ma: f64) -> f64 {
        let i_th = self.threshold_current_ma();
        if current_ma <= i_th {
            return 0.0;
        }
        self.slope_efficiency_mw_per_ma() * (current_ma - i_th)
    }

    /// Wall-plug efficiency η_WP = P_opt / (I·V).
    pub fn wall_plug_efficiency(&self, current_ma: f64, voltage: f64) -> f64 {
        let p_elec = current_ma * 1e-3 * voltage; // Watts
        let p_opt = self.output_power_mw(current_ma) * 1e-3;
        if p_elec < 1e-30 {
            0.0
        } else {
            p_opt / p_elec
        }
    }

    /// Far-field divergence half-angle θ (degrees).
    ///
    /// For a Gaussian beam: θ ≈ λ/(π·w₀) where w₀ = active radius.
    pub fn far_field_divergence_deg(&self) -> f64 {
        let w0_m = self.active_diameter_um * 0.5e-6;
        let theta_rad = self.wavelength / (PI * w0_m);
        theta_rad.to_degrees()
    }

    /// Returns `true` if the VCSEL is expected to operate in a single transverse mode.
    ///
    /// Single-mode condition: oxide aperture diameter < ~4 μm.
    pub fn is_single_mode(&self) -> bool {
        self.active_diameter_um < 4.0
    }

    /// Estimated polarisation switching current (mA).
    ///
    /// Typical rule-of-thumb: switching occurs at ~3–5× threshold.
    pub fn polarization_switching_current_ma(&self) -> f64 {
        self.threshold_current_ma() * 4.0
    }

    /// Thermal resistance R_th (K/mW).
    ///
    /// Approximate formula for oxide-confined VCSEL:
    /// R_th ≈ 1 / (4·k_th·d) where k_th = 46 W/(m·K) for GaAs.
    pub fn thermal_resistance_k_per_mw(&self) -> f64 {
        let k_th = 46.0; // W/(m·K) — GaAs
        let d_m = self.active_diameter_um * 1e-6;
        1.0 / (4.0 * k_th * d_m) * 1e-3 // K/mW
    }

    /// Junction temperature rise ΔT (K) from self-heating.
    ///
    /// ΔT = P_dissipated · R_th  where P_diss = I·V − P_opt
    pub fn junction_temperature_rise_k(&self, current_ma: f64, voltage: f64) -> f64 {
        let p_elec_mw = current_ma * voltage; // mW (I in mA, V in V)
        let p_opt_mw = self.output_power_mw(current_ma);
        let p_diss_mw = (p_elec_mw - p_opt_mw).max(0.0);
        p_diss_mw * self.thermal_resistance_k_per_mw()
    }
}

// ─── VcselArray ──────────────────────────────────────────────────────────────

/// 1D or 2D VCSEL array of identical elements on a regular pitch.
#[derive(Debug, Clone)]
pub struct VcselArray {
    /// Number of VCSEL elements in the array.
    pub n_elements: usize,
    /// Centre-to-centre pitch (μm).
    pub pitch_um: f64,
    /// Reference VCSEL element (all elements assumed identical).
    pub vcsel: Vcsel,
}

impl VcselArray {
    /// Construct a uniform VCSEL array.
    pub fn new(n: usize, pitch_um: f64, vcsel: Vcsel) -> Self {
        Self {
            n_elements: n.max(1),
            pitch_um,
            vcsel,
        }
    }

    /// Total optical output power (mW) when all elements are driven at `current_ma`.
    pub fn total_power_mw(&self, current_ma: f64) -> f64 {
        self.n_elements as f64 * self.vcsel.output_power_mw(current_ma)
    }

    /// Estimated optical crosstalk between adjacent elements (dB).
    ///
    /// Crosstalk decays exponentially with pitch/beam-width ratio.
    /// XT ≈ −20·log₁₀(exp(−(pitch/w₀)²)) = 8.686·(pitch/w₀)²
    pub fn crosstalk_db(&self) -> f64 {
        let w0_um = self.vcsel.active_diameter_um * 0.5;
        if w0_um < 1e-10 {
            return -100.0;
        }
        let ratio = self.pitch_um / w0_um;
        -8.686 * ratio * ratio
    }

    /// Total array footprint (μm) measured centre-to-centre of end elements.
    pub fn array_size_um(&self) -> f64 {
        if self.n_elements <= 1 {
            self.vcsel.active_diameter_um
        } else {
            (self.n_elements - 1) as f64 * self.pitch_um + self.vcsel.active_diameter_um
        }
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_dbr_reflectivity_increases_with_pairs() {
        let mat = DbrMaterial::AlGaAs850nm;
        let r10 = mat.dbr_reflectivity(10);
        let r25 = mat.dbr_reflectivity(25);
        assert!(
            r25 > r10,
            "Reflectivity should increase with pairs: r10={} r25={}",
            r10,
            r25
        );
        assert!(r25 < 1.0, "Reflectivity must be < 1");
    }

    #[test]
    fn test_dbr_bandwidth_reasonable() {
        let mat = DbrMaterial::AlGaAs850nm;
        let bw = mat.dbr_bandwidth_nm();
        // AlGaAs stopband is typically 50–100 nm
        assert!(
            bw > 20.0 && bw < 150.0,
            "DBR bandwidth out of range: {} nm",
            bw
        );
    }

    #[test]
    fn test_vcsel_850_threshold_ma_range() {
        let vcsel = Vcsel::new_850nm_standard(8.0, 5.0);
        let i_th = vcsel.threshold_current_ma();
        // VCSEL threshold should be positive and sub-100 mA
        assert!(
            i_th > 0.0 && i_th < 100.0,
            "Threshold out of range: {} mA",
            i_th
        );
    }

    #[test]
    fn test_vcsel_output_power_zero_below_threshold() {
        let vcsel = Vcsel::new_850nm_standard(8.0, 5.0);
        let i_th = vcsel.threshold_current_ma();
        let p = vcsel.output_power_mw(i_th * 0.5);
        assert_abs_diff_eq!(p, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vcsel_single_mode_condition() {
        let sm = Vcsel::new_850nm_standard(3.0, 2.0);
        assert!(sm.is_single_mode(), "3 μm aperture should be single-mode");
        let mm = Vcsel::new_850nm_standard(10.0, 5.0);
        assert!(!mm.is_single_mode(), "10 μm aperture should be multi-mode");
    }

    #[test]
    fn test_array_total_power_scales_with_elements() {
        let vcsel = Vcsel::new_850nm_standard(8.0, 10.0);
        let i_th = vcsel.threshold_current_ma();
        let arr1 = VcselArray::new(1, 25.0, vcsel.clone());
        let arr4 = VcselArray::new(4, 25.0, vcsel.clone());
        let i_bias = i_th * 3.0;
        let p1 = arr1.total_power_mw(i_bias);
        let p4 = arr4.total_power_mw(i_bias);
        assert_abs_diff_eq!(p4, 4.0 * p1, epsilon = 1e-6);
    }

    #[test]
    fn test_thermal_resistance_reasonable() {
        let vcsel = Vcsel::new_850nm_standard(8.0, 5.0);
        let r_th = vcsel.thermal_resistance_k_per_mw();
        // For 8 μm GaAs VCSEL: ~1–5 K/mW
        assert!(
            r_th > 0.1 && r_th < 20.0,
            "Thermal resistance out of range: {} K/mW",
            r_th
        );
    }

    #[test]
    fn test_effective_cavity_length_larger_than_physical() {
        let vcsel = Vcsel::new_850nm_standard(8.0, 5.0);
        let l_eff = vcsel.effective_cavity_length_nm();
        assert!(
            l_eff > vcsel.cavity_length_nm,
            "Effective cavity must be longer than physical"
        );
    }
}
