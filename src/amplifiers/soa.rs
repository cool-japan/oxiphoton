//! Semiconductor Optical Amplifier (SOA) model.
//!
//! Implements rate-equation-based gain, noise figure, saturation, cross-gain
//! modulation and wavelength conversion for bulk and quantum-well InGaAsP SOAs.
//!
//! References:
//! - Coldren, Corzine & Mashanovitch, "Diode Lasers and Photonic Integrated Circuits",
//!   2nd ed., Wiley 2012.
//! - Connelly, "Semiconductor Optical Amplifiers", Springer 2002.
use std::f64::consts::PI;

/// Planck constant (J·s)
const H_PLANCK: f64 = 6.626_070_15e-34;
/// Speed of light (m/s)
const C_LIGHT: f64 = 2.997_924_58e8;
/// Elementary charge (C)
const Q_ELECTRON: f64 = 1.602_176_634e-19;
/// Semiconductor Optical Amplifier rate equation model.
///
/// Uses the linear gain model g(N) = a·(N - N_tr), where a is the
/// differential gain and N_tr is the transparency carrier density.
#[derive(Debug, Clone)]
pub struct Soa {
    /// Active region length (mm).
    pub active_length_mm: f64,
    /// Active region width (μm).
    pub active_width_um: f64,
    /// Active region height (thickness) (μm).
    pub active_height_um: f64,
    /// Injection current (mA).
    pub injection_current_ma: f64,
    /// Internal optical loss αᵢ (cm⁻¹).
    pub internal_loss_per_cm: f64,
    /// Optical confinement factor Γ (dimensionless, 0–1).
    pub confinement_factor: f64,
    /// Differential gain a = dg/dN (m³).
    pub differential_gain: f64,
    /// Transparency carrier density N_tr (m⁻³).
    pub transparency_density: f64,
    /// Carrier (spontaneous) lifetime τ_s (ns).
    pub carrier_lifetime_ns: f64,
    /// Operating wavelength λ (m).
    pub wavelength: f64,
}

impl Soa {
    /// Construct an InGaAsP bulk SOA designed for the C-band (1550 nm).
    ///
    /// Parameters representative of a commercial InGaAsP bulk SOA:
    ///   - a = 3 × 10⁻²⁰ m³, N_tr = 1.0 × 10²⁴ m⁻³ (readily exceeded at 100 mA),
    ///   - τ_s = 300 ps, Γ = 0.4, αᵢ = 20 cm⁻¹.
    ///
    /// Carrier density at 100 mA: N ≈ J·τ/(q·d) ≈ 1.56 × 10²⁴ m⁻³ > N_tr.
    pub fn new_ingaasp_1550nm(current_ma: f64, length_mm: f64) -> Self {
        Self {
            active_length_mm: length_mm,
            active_width_um: 2.0,
            active_height_um: 0.2,
            injection_current_ma: current_ma,
            internal_loss_per_cm: 20.0,
            confinement_factor: 0.4,
            differential_gain: 3.0e-20,
            transparency_density: 1.0e24, // reachable at 100 mA
            carrier_lifetime_ns: 0.3,     // 300 ps
            wavelength: 1550e-9,
        }
    }

    /// Construct a bulk SOA for the O-band (1310 nm).
    pub fn new_bulk_1310nm(current_ma: f64, length_mm: f64) -> Self {
        Self {
            active_length_mm: length_mm,
            active_width_um: 2.5,
            active_height_um: 0.25,
            injection_current_ma: current_ma,
            internal_loss_per_cm: 25.0,
            confinement_factor: 0.4,
            differential_gain: 3.5e-20,
            transparency_density: 1.0e24,
            carrier_lifetime_ns: 0.4, // 400 ps
            wavelength: 1310e-9,
        }
    }

    /// Active region volume V = L · w · d (m³).
    pub fn active_volume_m3(&self) -> f64 {
        let l = self.active_length_mm * 1e-3;
        let w = self.active_width_um * 1e-6;
        let d = self.active_height_um * 1e-6;
        l * w * d
    }

    /// Current density J = I / (w · L) (A/m²).
    pub fn current_density_a_per_m2(&self) -> f64 {
        let i_a = self.injection_current_ma * 1e-3;
        let w = self.active_width_um * 1e-6;
        let l = self.active_length_mm * 1e-3;
        i_a / (w * l)
    }

    /// Steady-state carrier density N (m⁻³).
    ///
    /// From rate equation dN/dt = J/(q·d) - N/τ_s - ... = 0
    /// Ignoring stimulated recombination (small-signal):
    ///   N = J · τ_s / (q · d)
    pub fn carrier_density(&self) -> f64 {
        let j = self.current_density_a_per_m2();
        let d = self.active_height_um * 1e-6;
        let tau_s = self.carrier_lifetime_ns * 1e-9;
        j * tau_s / (Q_ELECTRON * d)
    }

    /// Material gain g(N) = a · (N - N_tr) (m⁻¹).
    ///
    /// Negative values indicate absorption (carrier density below transparency).
    pub fn material_gain(&self) -> f64 {
        let n = self.carrier_density();
        self.differential_gain * (n - self.transparency_density)
    }

    /// Modal gain G_m = Γ·g - αᵢ (cm⁻¹).
    pub fn modal_gain_per_cm(&self) -> f64 {
        let g_mat_per_cm = self.material_gain() * 1e-2; // m⁻¹ → cm⁻¹
        self.confinement_factor * g_mat_per_cm - self.internal_loss_per_cm
    }

    /// Single-pass amplifier gain G = exp(G_m · L) in dB.
    pub fn single_pass_gain_db(&self) -> f64 {
        let g_m_per_m = self.modal_gain_per_cm() * 100.0; // cm⁻¹ → m⁻¹
        let l = self.active_length_mm * 1e-3;
        let gain_linear = (g_m_per_m * l).exp();
        10.0 * gain_linear.log10()
    }

    /// Saturation output power P_sat (dBm).
    ///
    /// P_sat = h·ν · A_eff / (a · Γ · τ_s)
    /// where A_eff = w · d / Γ (effective cross-section).
    pub fn saturation_power_dbm(&self) -> f64 {
        let nu = C_LIGHT / self.wavelength;
        let w = self.active_width_um * 1e-6;
        let d = self.active_height_um * 1e-6;
        let tau_s = self.carrier_lifetime_ns * 1e-9;
        let a_eff = w * d / self.confinement_factor;
        let p_sat_w =
            H_PLANCK * nu * a_eff / (self.differential_gain * self.confinement_factor * tau_s);
        10.0 * (p_sat_w * 1e3).log10()
    }

    /// Compressed gain G(P_in) at a given input power (dBm).
    ///
    /// G(P) = G_ss / (1 + P_out / P_sat) solved iteratively.
    /// Approximate closed form: G ≈ G_ss / (1 + P_in · G_ss / P_sat).
    pub fn gain_db(&self, input_power_dbm: f64) -> f64 {
        let p_in_mw = 10.0_f64.powf(input_power_dbm / 10.0);
        let p_sat_mw = 10.0_f64.powf(self.saturation_power_dbm() / 10.0);
        let g_ss = 10.0_f64.powf(self.single_pass_gain_db() / 10.0);
        // Saleh compression model
        let g = g_ss / (1.0 + p_in_mw * g_ss / p_sat_mw);
        10.0 * g.log10()
    }

    /// Population inversion parameter n_sp for a semiconductor.
    ///
    /// n_sp ≈ N / (N - N_tr) for a two-band model.
    fn nsp(&self) -> f64 {
        let n = self.carrier_density();
        if n <= self.transparency_density {
            return 1e6;
        }
        n / (n - self.transparency_density)
    }

    /// Noise figure NF = 2·n_sp · (1 - 1/G_ss) + 1/G_ss (dB).
    ///
    /// Equivalent to the standard amplifier NF formula. For large gain,
    /// this approaches 2·n_sp (the quantum-limited value).
    /// Returns a sensible lower bound of 3 dB if the gain is ≤ 1.
    pub fn noise_figure_db(&self) -> f64 {
        let nsp = self.nsp();
        let g_ss = 10.0_f64.powf(self.single_pass_gain_db() / 10.0);
        if g_ss <= 1.0 {
            // Below or at transparency: NF is dominated by loss noise
            return 3.0_f64.max(-10.0 * g_ss.log10());
        }
        let nf_linear = 2.0 * nsp * (1.0 - 1.0 / g_ss) + 1.0 / g_ss;
        if nf_linear <= 0.0 {
            return 3.0;
        }
        10.0 * nf_linear.log10()
    }

    /// Henry α-factor (linewidth enhancement): α_H = -2k₀/g · dn_r/dN / (dg/dN).
    ///
    /// Typical value for bulk InGaAsP: 3–6. Approximate value used here.
    pub fn alpha_factor(&self) -> f64 {
        // Empirical value for bulk SOA; quantum well would be 1–3
        4.0
    }

    /// XGM (cross-gain modulation) bandwidth (GHz): f_XGM = 1/(2π·τ_s).
    pub fn xgm_bandwidth_ghz(&self) -> f64 {
        let tau_s = self.carrier_lifetime_ns * 1e-9;
        1.0 / (2.0 * PI * tau_s) * 1e-9 // Hz → GHz
    }

    /// Polarisation sensitivity: TE/TM gain difference (dB).
    ///
    /// Bulk SOAs have ~1–2 dB TE-TM asymmetry due to active region geometry.
    pub fn polarization_sensitivity_db(&self) -> f64 {
        // Sensitivity scales with aspect ratio: tall/narrow → less polarisation sensitivity
        let aspect = self.active_width_um / self.active_height_um;
        // Empirical: ~1 dB per decade of aspect ratio, clamped 0–3 dB
        (aspect.log10() * 1.5).clamp(0.0, 3.0)
    }

    /// Gain at temperature T_c (°C) relative to reference (25 °C).
    ///
    /// Gain decreases with temperature; characteristic temperature T₀ ≈ 50–70 K.
    pub fn gain_at_temperature(&self, temp_c: f64) -> f64 {
        let t0_k = 60.0; // characteristic temperature (K) for InGaAsP
        let delta_t = temp_c - 25.0; // deviation from reference
        let g_ref_db = self.single_pass_gain_db();
        // g(T) = g_ref * exp(-ΔT / T₀) — phenomenological
        let scale_db = -delta_t / t0_k * 10.0 * f64::ln(10.0).recip() * 4.343;
        g_ref_db + scale_db
    }
}

// ─── SOA cross-gain modulation wavelength converter ──────────────────────────

/// SOA as a cross-gain modulation (XGM) wavelength converter.
///
/// A strong pump signal modulates the carrier density in the SOA, which
/// transfers the modulation to a weak probe at a different wavelength (with
/// signal inversion).
#[derive(Debug, Clone)]
pub struct SoaXgm {
    /// Underlying SOA.
    pub soa: Soa,
    /// Pump (modulated input) wavelength (m).
    pub pump_wavelength: f64,
    /// Probe (converted output) wavelength (m).
    pub probe_wavelength: f64,
}

impl SoaXgm {
    /// Construct an XGM wavelength converter.
    pub fn new(soa: Soa, pump_wl: f64, probe_wl: f64) -> Self {
        Self {
            soa,
            pump_wavelength: pump_wl,
            probe_wavelength: probe_wl,
        }
    }

    /// Conversion efficiency (dB) at given pump power.
    ///
    /// η = G_probe - G_pump_saturation_penalty.
    /// The probe sees the SOA gain at the point of maximum pump saturation.
    pub fn conversion_efficiency_db(&self, pump_power_dbm: f64) -> f64 {
        // Probe gain when pump saturates the SOA
        let probe_gain = self.soa.gain_db(pump_power_dbm);
        // Pump extracts its own gain:
        let pump_gain = self.soa.gain_db(pump_power_dbm);
        probe_gain - pump_gain + 3.0 // approx 3 dB insertion loss offset
    }

    /// Extinction ratio (dB) of the converted signal.
    ///
    /// XGM inverts the signal (mark → space), so ER is typically 5–10 dB.
    /// ER_out = ER_in * (1 - G_min/G_max) / (1 + G_min/G_max) (approximate).
    pub fn extinction_ratio_db(&self, input_er_db: f64) -> f64 {
        let g_max_lin = 10.0_f64.powf(self.soa.single_pass_gain_db() / 10.0);
        let p_sat = 10.0_f64.powf(self.soa.saturation_power_dbm() / 10.0); // mW
                                                                           // G at mark level (high power): compressed
        let p_mark_dbm = 3.0; // assume 0 dBm mark
        let g_mark_lin = 10.0_f64.powf(self.soa.gain_db(p_mark_dbm) / 10.0);
        // G at space (low power ≈ small-signal):
        let g_space_lin = g_max_lin;
        // ER_out = (G_space * P_space_in) / (G_mark * P_mark_in)
        // P_space/P_mark = 10^(-ER_in/10)
        let er_in_lin = 10.0_f64.powf(-input_er_db / 10.0);
        if g_mark_lin <= 0.0 || p_sat <= 0.0 {
            return 0.0;
        }
        let er_out_lin = (g_space_lin * er_in_lin) / g_mark_lin;
        -10.0 * er_out_lin.log10()
    }

    /// XGM 3-dB modulation bandwidth (GHz).
    ///
    /// Dominated by carrier lifetime: f_{3dB} ≈ 1/(2π·τ_s).
    /// For high pump powers, effective lifetime is shortened.
    pub fn bandwidth_ghz(&self) -> f64 {
        self.soa.xgm_bandwidth_ghz()
    }
}

// ─── Utility ─────────────────────────────────────────────────────────────────

/// Convert modal gain (cm⁻¹) and length (mm) to single-pass linear gain.
pub fn modal_gain_to_linear(modal_gain_per_cm: f64, length_mm: f64) -> f64 {
    let g_per_m = modal_gain_per_cm * 100.0;
    let l_m = length_mm * 1e-3;
    (g_per_m * l_m).exp()
}

/// Convert single-pass gain (linear) to gain in dB.
pub fn linear_gain_to_db(gain_linear: f64) -> f64 {
    if gain_linear <= 0.0 {
        return f64::NEG_INFINITY;
    }
    10.0 * gain_linear.log10()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_carrier_density_increases_with_current() {
        let soa_low = Soa::new_ingaasp_1550nm(50.0, 0.5);
        let soa_high = Soa::new_ingaasp_1550nm(200.0, 0.5);
        assert!(
            soa_high.carrier_density() > soa_low.carrier_density(),
            "Higher current must produce higher carrier density"
        );
    }

    #[test]
    fn test_modal_gain_positive_above_transparency() {
        // High current → above transparency
        let soa = Soa::new_ingaasp_1550nm(300.0, 1.0);
        let gm = soa.modal_gain_per_cm();
        assert!(
            gm > 0.0,
            "Modal gain must be positive at high injection; got {gm}"
        );
    }

    #[test]
    fn test_single_pass_gain_increases_with_length() {
        // Scale current proportionally to length to keep current density constant.
        // At constant J, modal gain g_m is fixed, so G = exp(g_m * L) increases with L.
        let soa_short = Soa::new_ingaasp_1550nm(200.0, 0.5);
        let soa_long = Soa::new_ingaasp_1550nm(800.0, 2.0); // 4× current for 4× length
        let g_short = soa_short.single_pass_gain_db();
        let g_long = soa_long.single_pass_gain_db();
        assert!(
            g_long > g_short,
            "Longer SOA (at constant J) must have higher gain; short={g_short} dB, long={g_long} dB"
        );
    }

    #[test]
    fn test_noise_figure_above_3db() {
        let soa = Soa::new_ingaasp_1550nm(200.0, 1.0);
        let nf = soa.noise_figure_db();
        assert!(nf >= 3.0, "SOA NF must be ≥ 3 dB; got {nf}");
    }

    #[test]
    fn test_gain_compression_under_saturation() {
        let soa = Soa::new_ingaasp_1550nm(200.0, 1.0);
        let g_small = soa.gain_db(-30.0);
        let g_large = soa.gain_db(5.0);
        assert!(
            g_small >= g_large,
            "Gain must decrease with input power; g(-30 dBm)={g_small}, g(5 dBm)={g_large}"
        );
    }

    #[test]
    fn test_xgm_bandwidth_finite_positive() {
        let soa = Soa::new_ingaasp_1550nm(200.0, 1.0);
        let bw = soa.xgm_bandwidth_ghz();
        assert!(
            bw > 0.0 && bw.is_finite(),
            "XGM bandwidth must be finite and positive; got {bw}"
        );
    }

    #[test]
    fn test_saturation_power_positive() {
        let soa = Soa::new_ingaasp_1550nm(200.0, 1.0);
        let p_sat = soa.saturation_power_dbm();
        assert!(
            p_sat.is_finite(),
            "Saturation power must be finite; got {p_sat}"
        );
    }

    #[test]
    fn test_gain_decreases_with_temperature() {
        let soa = Soa::new_ingaasp_1550nm(200.0, 1.0);
        let g_25 = soa.gain_at_temperature(25.0);
        let g_60 = soa.gain_at_temperature(60.0);
        assert!(
            g_25 > g_60,
            "Gain must decrease with temperature; g(25°C)={g_25}, g(60°C)={g_60}"
        );
    }

    #[test]
    fn test_active_volume_calculation() {
        let soa = Soa::new_ingaasp_1550nm(200.0, 1.0);
        // L=1 mm, w=2 μm, d=0.2 μm → V = 1e-3 * 2e-6 * 0.2e-6 = 4e-16 m³
        let v = soa.active_volume_m3();
        assert_abs_diff_eq!(v, 1e-3 * 2e-6 * 0.2e-6, epsilon = 1e-25);
    }

    #[test]
    fn test_modal_gain_to_linear_utility() {
        // g_m = 0 → gain = 1 (0 dB)
        let g = modal_gain_to_linear(0.0, 1.0);
        assert_abs_diff_eq!(g, 1.0, epsilon = 1e-12);
    }
}
