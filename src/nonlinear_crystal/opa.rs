//! Optical Parametric Amplifier (OPA) and Oscillator (OPO) simulation.
//!
//! Provides:
//! - Single-pass OPA gain (no depletion: G = cosh²(gL))
//! - Parametric gain coefficient Γ from pump intensity
//! - Phase mismatch, group velocity mismatch (GVM), walk-off length
//! - OPO threshold pump intensity and power
//! - Signal output power above threshold
//! - Quasi-phase matched SHG (QPM-SHG) efficiency for PPLN

use crate::error::OxiPhotonError;
use crate::nonlinear_crystal::crystals::NloCrystal;

/// Speed of light in vacuum (m/s).
const C0: f64 = 2.99792458e8;
/// Permittivity of free space (F/m).
const EPS0: f64 = 8.854187817e-12;

// ─── Optical Parametric Amplifier ──────────────────────────────────────────

/// Optical Parametric Amplifier (OPA).
///
/// Models single-pass parametric amplification in the undepleted-pump approximation.
/// The signal gain is G_s = cosh²(g·L) where g is the parametric gain coefficient.
pub struct OpticalParametricAmplifier {
    /// NLO crystal.
    pub crystal: NloCrystal,
    /// Pump wavelength λ_p (nm).
    pub pump_wavelength_nm: f64,
    /// Signal wavelength λ_s (nm).
    pub signal_wavelength_nm: f64,
    /// Crystal length L (mm).
    pub crystal_length_mm: f64,
    /// Effective nonlinear coefficient d_eff (pm/V).
    pub d_eff_pm_per_v: f64,
    /// Pump intensity I_p (GW/cm²).
    pub pump_intensity_gw_per_cm2: f64,
}

impl OpticalParametricAmplifier {
    /// Construct a new OPA.
    pub fn new(
        crystal: NloCrystal,
        pump_nm: f64,
        signal_nm: f64,
        length_mm: f64,
        d_eff: f64,
        pump_intensity: f64,
    ) -> Self {
        Self {
            crystal,
            pump_wavelength_nm: pump_nm,
            signal_wavelength_nm: signal_nm,
            crystal_length_mm: length_mm,
            d_eff_pm_per_v: d_eff,
            pump_intensity_gw_per_cm2: pump_intensity,
        }
    }

    /// Idler wavelength λ_i (nm): 1/λ_i = 1/λ_p − 1/λ_s.
    pub fn idler_wavelength_nm(&self) -> f64 {
        let inv = 1.0 / self.pump_wavelength_nm - 1.0 / self.signal_wavelength_nm;
        if inv <= 0.0 {
            return f64::INFINITY;
        }
        1.0 / inv
    }

    /// Parametric gain coefficient g (1/mm) in the undepleted-pump approximation.
    ///
    /// Γ² = (ω_s · ω_i · d_eff² · I_p) / (n_s · n_i · n_p · ε₀ · c³)
    ///
    /// g = sqrt(Γ² − (Δk/2)²); if Γ² < (Δk/2)² the process is non-oscillatory.
    pub fn gain_coefficient_per_mm(&self) -> f64 {
        let gamma2 = self.gamma_squared_per_m2();
        let dk_half = self.phase_mismatch(0.0) / 2.0; // rough estimate at θ=0
        let g2 = gamma2 - dk_half * dk_half;
        if g2 <= 0.0 {
            // Non-oscillatory regime: use Γ as a lower bound
            return gamma2.sqrt() * 1e-3; // 1/m → 1/mm
        }
        g2.sqrt() * 1e-3
    }

    /// Γ² (1/m²) — coupling coefficient squared from pump intensity.
    fn gamma_squared_per_m2(&self) -> f64 {
        let omega_s = 2.0 * std::f64::consts::PI * C0 / (self.signal_wavelength_nm * 1e-9);
        let lambda_i_nm = self.idler_wavelength_nm();
        let lambda_i_nm = if lambda_i_nm.is_finite() {
            lambda_i_nm
        } else {
            2000.0
        };
        let omega_i = 2.0 * std::f64::consts::PI * C0 / (lambda_i_nm * 1e-9);
        let d_eff = self.d_eff_pm_per_v * 1e-12; // m/V
        let i_p = self.pump_intensity_gw_per_cm2 * 1e13; // GW/cm² → W/m²
        let n_s = self.crystal.n_ordinary(self.signal_wavelength_nm);
        let n_i = self.crystal.n_ordinary(lambda_i_nm);
        let n_p = self.crystal.n_ordinary(self.pump_wavelength_nm);

        // Γ² = ω_s · ω_i · d_eff² · I_p / (n_s · n_i · n_p · ε₀ · c³)
        (omega_s * omega_i * d_eff * d_eff * i_p) / (n_s * n_i * n_p * EPS0 * C0 * C0 * C0)
    }

    /// Phase mismatch Δk (rad/m) at crystal cut angle θ (rad).
    ///
    /// Δk = k_s + k_i − k_p (collinear, ordinary polarizations).
    pub fn phase_mismatch(&self, theta_rad: f64) -> f64 {
        let lambda_i_nm = self.idler_wavelength_nm();
        let lambda_i_nm = if lambda_i_nm.is_finite() {
            lambda_i_nm
        } else {
            2000.0
        };
        let n_s = self
            .crystal
            .n_extraordinary(self.signal_wavelength_nm, theta_rad);
        let n_i = self.crystal.n_extraordinary(lambda_i_nm, theta_rad);
        let n_p = self.crystal.n_ordinary(self.pump_wavelength_nm);
        let k_s = 2.0 * std::f64::consts::PI * n_s / (self.signal_wavelength_nm * 1e-9);
        let k_i = 2.0 * std::f64::consts::PI * n_i / (lambda_i_nm * 1e-9);
        let k_p = 2.0 * std::f64::consts::PI * n_p / (self.pump_wavelength_nm * 1e-9);
        k_s + k_i - k_p
    }

    /// Single-pass signal gain G_s (linear) in the undepleted-pump approximation.
    ///
    /// G_s = cosh²(g·L)
    pub fn signal_gain_linear(&self) -> f64 {
        let g = self.gain_coefficient_per_mm();
        let l = self.crystal_length_mm;
        let gl = g * l;
        gl.cosh().powi(2)
    }

    /// Signal gain in dB: 10·log₁₀(G_s).
    pub fn signal_gain_db(&self) -> f64 {
        10.0 * self.signal_gain_linear().log10()
    }

    /// OPA signal bandwidth δλ_s (nm) from phase-matching curve curvature.
    ///
    /// Approximated from group velocity mismatch:
    /// δλ ≈ (π/(GVM · L)) · (λ_s²/c)
    pub fn signal_bandwidth_nm(&self) -> f64 {
        let gvm = self.gvm_ps_per_mm().abs();
        let l_mm = self.crystal_length_mm;
        if gvm < 1e-10 || l_mm < 1e-10 {
            return f64::INFINITY;
        }
        // δt = 1/(GVM·L) in THz → δλ ≈ λ²/(c·π) · δf
        // GVM in ps/mm: 1/gvm/l_mm = THz
        let delta_f_thz = 1.0 / (gvm * l_mm); // THz bandwidth
        let lambda_m = self.signal_wavelength_nm * 1e-9;
        // δλ = λ²/(c) · δf
        let delta_lambda_m = lambda_m * lambda_m / C0 * delta_f_thz * 1e12;
        delta_lambda_m * 1e9 // → nm
    }

    /// Group velocity mismatch (GVM) between signal and idler (ps/mm).
    ///
    /// GVM = 1/v_{g,s} − 1/v_{g,i} = n_{g,s}/c − n_{g,i}/c
    pub fn gvm_ps_per_mm(&self) -> f64 {
        let lambda_i_nm = self.idler_wavelength_nm();
        let lambda_i_nm = if lambda_i_nm.is_finite() {
            lambda_i_nm
        } else {
            2000.0
        };
        // Numerical group index: n_g = n - λ dn/dλ
        let dl_nm = 0.01;
        let n_s = self.crystal.n_ordinary(self.signal_wavelength_nm);
        let n_s_p = self.crystal.n_ordinary(self.signal_wavelength_nm + dl_nm);
        let dn_s = (n_s_p - n_s) / (dl_nm * 1e-9);
        let ng_s = n_s - self.signal_wavelength_nm * 1e-9 * dn_s;

        let n_i = self.crystal.n_ordinary(lambda_i_nm);
        let n_i_p = self.crystal.n_ordinary(lambda_i_nm + dl_nm);
        let dn_i = (n_i_p - n_i) / (dl_nm * 1e-9);
        let ng_i = n_i - lambda_i_nm * 1e-9 * dn_i;

        // GVM = (1/v_gs - 1/v_gi) = (n_gs - n_gi)/c in s/m; convert to ps/mm
        let gvm_s_per_m = (ng_s - ng_i) / C0;
        gvm_s_per_m * 1e12 * 1e-3 // s/m → ps/mm
    }

    /// Pulse walk-off length L_walk (mm): length over which signal and idler pulses
    /// of width τ (ps) separate by one pulse width.
    ///
    /// L_walk = τ / |GVM|
    pub fn walkoff_length_mm(&self, pulse_width_ps: f64) -> f64 {
        let gvm = self.gvm_ps_per_mm().abs();
        if gvm < 1e-12 {
            return f64::INFINITY;
        }
        pulse_width_ps / gvm
    }

    /// Output signal energy E_out (μJ) for given input E_in (μJ) in the no-depletion limit.
    pub fn output_signal_energy_uj(&self, input_signal_energy_uj: f64) -> f64 {
        input_signal_energy_uj * self.signal_gain_linear()
    }

    /// Pump depletion fraction at full conversion (theoretical maximum).
    ///
    /// By the Manley-Rowe relations, at 100% signal + idler conversion:
    /// fraction = (ω_s + ω_i) / ω_p = λ_p / λ_s + λ_p / λ_i (energy ratio)
    /// This equals 1 by energy conservation (ω_p = ω_s + ω_i).
    pub fn pump_depletion_at_conversion(&self) -> f64 {
        // Full depletion: every pump photon → 1 signal + 1 idler photon
        // energy conservation guarantees 100% energy transfer at perfect PM
        1.0
    }
}

// ─── Optical Parametric Oscillator ─────────────────────────────────────────

/// Optical Parametric Oscillator (OPO).
///
/// Models a singly-resonant OPO cavity with signal feedback.
/// Threshold condition: the round-trip gain must overcome the round-trip loss.
pub struct OpticalParametricOscillator {
    /// OPA gain medium.
    pub opa: OpticalParametricAmplifier,
    /// Total cavity round-trip loss (fraction, 0–1, combining all losses).
    pub cavity_round_trip_loss: f64,
    /// Output coupler transmission (fraction, 0–1).
    pub output_coupler_transmission: f64,
    /// Optical cavity length L_cav (mm).
    pub cavity_length_mm: f64,
}

impl OpticalParametricOscillator {
    /// Construct a new OPO.
    pub fn new(
        opa: OpticalParametricAmplifier,
        round_trip_loss: f64,
        oc_transmission: f64,
        cavity_length_mm: f64,
    ) -> Self {
        Self {
            opa,
            cavity_round_trip_loss: round_trip_loss,
            output_coupler_transmission: oc_transmission,
            cavity_length_mm,
        }
    }

    /// Total round-trip loss factor R_eff = (1 − L_total) where L_total includes
    /// OC transmission and internal losses.
    fn effective_reflectivity(&self) -> f64 {
        let total_loss = (self.cavity_round_trip_loss + self.output_coupler_transmission).min(1.0);
        (1.0 - total_loss).max(0.0)
    }

    /// Threshold pump intensity I_th (GW/cm²).
    ///
    /// At threshold: cosh²(g_th · L) = 1/sqrt(R_eff).
    /// Solving for g_th and then for I_th from Γ².
    pub fn threshold_intensity_gw_per_cm2(&self) -> f64 {
        let r = self.effective_reflectivity();
        if r <= 0.0 {
            return f64::INFINITY; // cannot oscillate
        }
        // cosh(g·L) = R^(-1/4) for a singly-resonant OPO with one round-trip gain pass
        // Threshold: cosh²(g_th·L) = 1/sqrt(R)
        let cosh_val = (1.0 / r.sqrt()).max(1.0);
        let g_th_l = cosh_val.acosh();
        let g_th_per_mm = g_th_l / self.opa.crystal_length_mm;
        let g_th_per_m = g_th_per_mm * 1e3;

        // Γ² = g² (at Δk=0), so I_th = g_th² · (n_s·n_i·n_p·ε₀·c³)/(ω_s·ω_i·d_eff²)
        let lambda_s = self.opa.signal_wavelength_nm * 1e-9;
        let lambda_i = self.opa.idler_wavelength_nm();
        let lambda_i = if lambda_i.is_finite() {
            lambda_i * 1e-9
        } else {
            2000e-9
        };
        let omega_s = 2.0 * std::f64::consts::PI * C0 / lambda_s;
        let omega_i = 2.0 * std::f64::consts::PI * C0 / lambda_i;
        let d_eff = self.opa.d_eff_pm_per_v * 1e-12;
        let n_s = self.opa.crystal.n_ordinary(self.opa.signal_wavelength_nm);
        let n_i = self
            .opa
            .crystal
            .n_ordinary(self.opa.idler_wavelength_nm().min(5000.0));
        let n_p = self.opa.crystal.n_ordinary(self.opa.pump_wavelength_nm);

        let i_th_w_per_m2 = g_th_per_m * g_th_per_m * (n_s * n_i * n_p * EPS0 * C0 * C0 * C0)
            / (omega_s * omega_i * d_eff * d_eff);
        i_th_w_per_m2 * 1e-13 // W/m² → GW/cm²
    }

    /// Returns true if the current pump intensity exceeds the OPO threshold.
    pub fn is_above_threshold(&self) -> bool {
        self.opa.pump_intensity_gw_per_cm2 > self.threshold_intensity_gw_per_cm2()
    }

    /// Threshold pump power P_th (W) for a given beam area A (μm²).
    pub fn threshold_power_w(&self, beam_area_um2: f64) -> f64 {
        let i_th = self.threshold_intensity_gw_per_cm2() * 1e13; // W/m²
        let area_m2 = beam_area_um2 * 1e-12;
        i_th * area_m2
    }

    /// Signal output power P_out (mW) above threshold, using slope efficiency.
    ///
    /// Approximation: P_out ∝ T_oc · (P_pump/P_th − 1) with slope efficiency ~ ω_s/ω_p.
    pub fn output_power_mw(&self, pump_power_w: f64, pump_beam_area_um2: f64) -> f64 {
        let p_th = self.threshold_power_w(pump_beam_area_um2);
        if pump_power_w <= p_th {
            return 0.0;
        }
        // Quantum slope efficiency: η_slope = (ω_s/ω_p) * T_oc / (T_oc + L_int)
        let omega_s = C0 / (self.opa.signal_wavelength_nm * 1e-9);
        let omega_p = C0 / (self.opa.pump_wavelength_nm * 1e-9);
        let quantum_efficiency = omega_s / omega_p;
        let slope_eff = quantum_efficiency * self.output_coupler_transmission
            / (self.output_coupler_transmission + self.cavity_round_trip_loss).max(1e-10);
        let p_out_w = slope_eff * (pump_power_w - p_th);
        p_out_w * 1e3 // W → mW
    }

    /// Approximate tuning range (nm) achievable by varying temperature ±50°C.
    pub fn tuning_range_nm(&self) -> f64 {
        let rate = self.opa.crystal.temperature_tuning_rate_nm_per_c();
        rate.abs() * 100.0 // ±50°C range
    }

    /// Free spectral range (GHz) of the OPO cavity.
    ///
    /// FSR = c / (2 · n_g · L_cav)
    pub fn fsr_ghz(&self) -> f64 {
        let n_g = self.opa.crystal.n_ordinary(self.opa.signal_wavelength_nm);
        let l_m = self.cavity_length_mm * 1e-3;
        let fsr_hz = C0 / (2.0 * n_g * l_m);
        fsr_hz * 1e-9 // Hz → GHz
    }
}

// ─── Quasi-phase matched SHG (QPM-SHG) ────────────────────────────────────

/// Quasi-phase matched SHG in periodically-poled crystals (e.g., PPLN).
///
/// Uses the first-order QPM effective coefficient d_eff = (2/π) · d₃₃.
pub struct QpmShg {
    /// PPLN or other periodically-poled crystal.
    pub ppln: NloCrystal,
    /// Fundamental wavelength λ₁ (nm).
    pub fundamental_wavelength_nm: f64,
    /// Crystal length L (mm).
    pub crystal_length_mm: f64,
    /// d₃₃ coefficient of the crystal (pm/V). For LiNbO₃, d₃₃ ≈ 27 pm/V.
    pub d33_pm_per_v: f64,
}

impl QpmShg {
    /// Construct a new QPM-SHG object.
    pub fn new(ppln: NloCrystal, lambda_nm: f64, length_mm: f64, d33: f64) -> Self {
        Self {
            ppln,
            fundamental_wavelength_nm: lambda_nm,
            crystal_length_mm: length_mm,
            d33_pm_per_v: d33,
        }
    }

    /// Effective QPM nonlinear coefficient for first-order QPM.
    ///
    /// d_eff = (2/π) · d₃₃
    pub fn d_eff_qpm(&self) -> f64 {
        2.0 / std::f64::consts::PI * self.d33_pm_per_v
    }

    /// Normalized SHG conversion efficiency η_norm (%/W/cm²).
    ///
    /// η_norm = (8π²·d_eff²·L²) / (n₁²·n₂·ε₀·c·λ₁²·A) · A
    /// Simplifies to: η_norm = (8π²·d_eff²·L²) / (n₁²·n₂·ε₀·c·λ₁²) in %/W
    /// Per unit area → %/W/cm²
    pub fn normalized_efficiency(&self) -> f64 {
        let lambda1_m = self.fundamental_wavelength_nm * 1e-9;
        let lambda2_m = lambda1_m / 2.0;
        let l_m = self.crystal_length_mm * 1e-3;
        let d_eff = self.d_eff_qpm() * 1e-12; // m/V
        let n1 = self.ppln.n_ordinary(self.fundamental_wavelength_nm);
        let n2 = self.ppln.n_ordinary(self.fundamental_wavelength_nm / 2.0);

        // η_norm [%/W/cm²] = η_pw_factor [1/(W/m²)] * 1e2 * 1e-4
        let eta_factor = (8.0 * std::f64::consts::PI.powi(2) * d_eff * d_eff * l_m * l_m)
            / (n1 * n1 * n2 * EPS0 * C0 * lambda2_m * lambda1_m * lambda1_m);
        // Convert W⁻¹·m⁻² → %/W/cm²: multiply by 1e-4 (m² → cm²) × 100 (%):
        eta_factor * 1e-4 * 100.0
    }

    /// SHG conversion efficiency (fraction 0–1) for given peak power P (W) and beam area A (μm²).
    ///
    /// η = η_norm \[%/W/cm²\] · P \[W\] · A \[cm²\] / 100
    pub fn conversion_efficiency(&self, peak_power_w: f64, beam_area_um2: f64) -> f64 {
        let area_cm2 = beam_area_um2 * 1e-8; // μm² → cm²
        let eta_norm = self.normalized_efficiency(); // %/W/cm²
        let eta = eta_norm * peak_power_w * area_cm2 / 100.0;
        eta.min(1.0)
    }

    /// Required crystal length L (mm) to achieve target efficiency η_target (0–1)
    /// for given peak power P (W) and beam area A (μm²).
    ///
    /// From η = η_norm·P·A · L² (∝ L²), solve L = sqrt(η / (const · P · A)).
    pub fn required_length_mm(
        &self,
        target_efficiency: f64,
        power_w: f64,
        area_um2: f64,
    ) -> Result<f64, OxiPhotonError> {
        if target_efficiency <= 0.0 || target_efficiency > 1.0 {
            return Err(OxiPhotonError::NumericalError(
                "Target efficiency must be in (0, 1]".to_string(),
            ));
        }
        // η_norm ∝ L², so η(L) = η_norm(L=1mm) * (L/1mm)²
        // Compute η per mm² first
        let lambda1_m = self.fundamental_wavelength_nm * 1e-9;
        let lambda2_m = lambda1_m / 2.0;
        let d_eff = self.d_eff_qpm() * 1e-12;
        let n1 = self.ppln.n_ordinary(self.fundamental_wavelength_nm);
        let n2 = self.ppln.n_ordinary(self.fundamental_wavelength_nm / 2.0);
        let area_m2 = area_um2 * 1e-12;

        let eta_per_l2 = (8.0 * std::f64::consts::PI.powi(2) * d_eff * d_eff * power_w)
            / (n1 * n1 * n2 * EPS0 * C0 * lambda2_m * lambda1_m * lambda1_m * area_m2);
        // eta_per_l2 is in 1/m²
        if eta_per_l2 <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "Cannot achieve target efficiency — check input parameters".to_string(),
            ));
        }
        let l_m = (target_efficiency / eta_per_l2).sqrt();
        Ok(l_m * 1e3) // m → mm
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nonlinear_crystal::crystals::NloCrystal;

    fn make_opa() -> OpticalParametricAmplifier {
        OpticalParametricAmplifier::new(
            NloCrystal::bbo(),
            532.0, // pump nm
            800.0, // signal nm
            5.0,   // length mm
            2.3,   // d_eff pm/V
            0.5,   // pump intensity GW/cm²
        )
    }

    #[test]
    fn test_opa_idler_wavelength() {
        let opa = make_opa();
        let idler = opa.idler_wavelength_nm();
        // 1/532 - 1/800 = 1/λ_i
        let expected = 1.0 / (1.0 / 532.0 - 1.0 / 800.0);
        assert!(
            (idler - expected).abs() < 0.1,
            "OPA idler {:.2} nm, expected {:.2} nm",
            idler,
            expected
        );
    }

    #[test]
    fn test_opa_gain_positive() {
        let opa = make_opa();
        let gain = opa.signal_gain_linear();
        assert!(
            gain >= 1.0,
            "OPA gain {:.4} should be >= 1 (amplification)",
            gain
        );
    }

    #[test]
    fn test_opa_gain_increases_with_length() {
        let opa_short =
            OpticalParametricAmplifier::new(NloCrystal::bbo(), 532.0, 800.0, 2.0, 2.3, 0.5);
        let opa_long =
            OpticalParametricAmplifier::new(NloCrystal::bbo(), 532.0, 800.0, 10.0, 2.3, 0.5);
        let gain_short = opa_short.signal_gain_linear();
        let gain_long = opa_long.signal_gain_linear();
        assert!(
            gain_long >= gain_short,
            "Gain with longer crystal ({:.4}) should be >= shorter ({:.4})",
            gain_long,
            gain_short
        );
    }

    #[test]
    fn test_opo_threshold_positive() {
        let opa = make_opa();
        let opo = OpticalParametricOscillator::new(opa, 0.05, 0.10, 50.0);
        let i_th = opo.threshold_intensity_gw_per_cm2();
        assert!(
            i_th > 0.0 && i_th.is_finite(),
            "OPO threshold intensity {:.4} GW/cm² should be positive and finite",
            i_th
        );
    }

    #[test]
    fn test_opo_above_threshold() {
        // Use very high pump intensity to ensure above threshold
        let opa_high =
            OpticalParametricAmplifier::new(NloCrystal::bbo(), 532.0, 800.0, 10.0, 2.3, 10.0);
        let opo = OpticalParametricOscillator::new(opa_high, 0.05, 0.10, 50.0);
        assert!(
            opo.is_above_threshold(),
            "OPO with 10 GW/cm² pump should be above threshold"
        );
    }

    #[test]
    fn test_qpm_d_eff() {
        let ppln = NloCrystal::ppln(19.0);
        let qpm = QpmShg::new(ppln, 1064.0, 20.0, 27.0);
        let d_eff = qpm.d_eff_qpm();
        let expected = 2.0 / std::f64::consts::PI * 27.0;
        assert!(
            (d_eff - expected).abs() < 1e-6,
            "QPM d_eff {:.4} != 2/π*d33 = {:.4}",
            d_eff,
            expected
        );
    }

    #[test]
    fn test_conversion_efficiency_low_power() {
        // At low power, efficiency ∝ P (and ∝ P² for L fixed, but η << 1)
        let ppln = NloCrystal::ppln(19.0);
        let qpm1 = QpmShg::new(ppln.clone(), 1064.0, 20.0, 27.0);
        let qpm2 = QpmShg::new(ppln, 1064.0, 20.0, 27.0);
        let eta1 = qpm1.conversion_efficiency(1.0, 1e4); // 1 W, 100 μm² beam
        let eta2 = qpm2.conversion_efficiency(4.0, 1e4); // 4 W
                                                         // η ∝ P (for fixed area, low-depletion), so η(4W)/η(1W) ≈ 4
        if eta1 > 0.0 && eta2 < 0.9 {
            let ratio = eta2 / eta1;
            assert!(
                (ratio - 4.0).abs() < 0.5,
                "Efficiency ratio at 4x power: {:.3} (expected ≈ 4)",
                ratio
            );
        }
    }
}
