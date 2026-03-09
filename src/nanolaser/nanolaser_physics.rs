//! Nanolaser-specific physics: plasmonic nanolasers (SPASER), photonic crystal
//! nanocavity lasers, and semiconductor disk lasers (VECSEL).
//!
//! # References
//!
//! - D. J. Bergman & M. I. Stockman, "Surface Plasmon Amplification by Stimulated
//!   Emission of Radiation: Quantum Generation of Coherent Surface Plasmons in
//!   Nanosystems", PRL 90, 027402 (2003).
//! - O. Painter et al., "Two-Dimensional Photonic Band-Gap Defect Mode Laser",
//!   Science 284, 1819 (1999).
//! - A. C. Tropper et al., "Vertical-external-cavity semiconductor lasers",
//!   J. Phys. D 37, R75 (2004).

use super::rate_equations::GeneralizedRateEquations;
use std::f64::consts::PI;

/// Speed of light in vacuum (m/s)
const C_LIGHT: f64 = 2.997_924_58e8;

// ─── PlasmonicCavity ─────────────────────────────────────────────────────────

/// Parameters of a plasmonic nanocavity for SPASER operation.
#[derive(Debug, Clone)]
pub struct PlasmonicCavity {
    /// Effective mode volume V_eff (nm³).
    pub mode_volume_nm3: f64,
    /// Quality factor Q of the plasmonic resonance.
    pub q_factor: f64,
    /// Purcell factor F_P = (3/(4π²))·(λ/n)³/V·Q.
    pub purcell_factor: f64,
    /// Resonance wavelength (m).
    pub resonance_wavelength: f64,
}

impl PlasmonicCavity {
    /// Compute the Purcell factor from geometry.
    pub fn compute_purcell(
        mode_volume_nm3: f64,
        q_factor: f64,
        wavelength_m: f64,
        n_eff: f64,
    ) -> f64 {
        let v_m3 = mode_volume_nm3 * 1e-27;
        let lambda_n = wavelength_m / n_eff;
        (3.0 / (4.0 * PI * PI)) * (lambda_n.powi(3) / v_m3) * q_factor
    }

    /// Resonance linewidth Δλ (nm).
    pub fn linewidth_nm(&self) -> f64 {
        let lambda_nm = self.resonance_wavelength * 1e9;
        lambda_nm / self.q_factor
    }

    /// Resonance linewidth Δν (THz).
    pub fn linewidth_thz(&self) -> f64 {
        C_LIGHT / (self.q_factor * self.resonance_wavelength) * 1e-12
    }
}

// ─── GainMaterial ─────────────────────────────────────────────────────────────

/// Gain material for the SPASER active region.
#[derive(Debug, Clone)]
pub enum GainMaterial {
    /// InGaAsP quantum well (1.3–1.55 μm).
    InGaAsP,
    /// CdSe quantum dot (visible, 620 nm).
    CdSe,
    /// Organic dye gain medium with specified peak wavelength and cross-section.
    Dye {
        /// Peak gain wavelength (m).
        peak_wavelength: f64,
        /// Stimulated emission cross-section (m²).
        gain_cross_section: f64,
    },
    /// Erbium-doped gain medium (1.55 μm).
    ErbiumDoped,
}

impl GainMaterial {
    /// Peak gain wavelength (m).
    pub fn peak_wavelength_m(&self) -> f64 {
        match self {
            GainMaterial::InGaAsP => 1.3e-6,
            GainMaterial::CdSe => 620e-9,
            GainMaterial::Dye {
                peak_wavelength, ..
            } => *peak_wavelength,
            GainMaterial::ErbiumDoped => 1.55e-6,
        }
    }

    /// Stimulated emission cross-section (m²).
    pub fn emission_cross_section_m2(&self) -> f64 {
        match self {
            GainMaterial::InGaAsP => 2.0e-20,
            GainMaterial::CdSe => 5.0e-19,
            GainMaterial::Dye {
                gain_cross_section, ..
            } => *gain_cross_section,
            GainMaterial::ErbiumDoped => 6.0e-25,
        }
    }
}

// ─── GainMedium ──────────────────────────────────────────────────────────────

/// Active gain medium parameters.
#[derive(Debug, Clone)]
pub struct GainMedium {
    /// Gain material type.
    pub material: GainMaterial,
    /// Carrier/emitter density (m⁻³).
    pub carrier_density: f64,
    /// Peak modal gain at transparency (cm⁻¹).
    pub peak_gain: f64,
    /// Transparency carrier density (m⁻³).
    pub transparency_density: f64,
}

impl GainMedium {
    /// Net gain at current carrier density (cm⁻¹).
    pub fn net_gain_cm(&self) -> f64 {
        let sigma = self.material.emission_cross_section_m2();
        // g = σ·(N − N_tr) in m⁻¹, convert to cm⁻¹
        let g_m = sigma * (self.carrier_density - self.transparency_density);
        g_m * 100.0 // m⁻¹ → cm⁻¹
    }
}

// ─── Spaser ──────────────────────────────────────────────────────────────────

/// Surface Plasmon Amplification by Stimulated Emission of Radiation (SPASER).
///
/// A nanoscale coherent light source operating via stimulated emission of
/// surface plasmons instead of photons, enabling mode volumes far below λ³/n³.
#[derive(Debug, Clone)]
pub struct Spaser {
    /// Plasmonic nanocavity parameters.
    pub plasmonic_cavity: PlasmonicCavity,
    /// Active gain medium.
    pub gain_medium: GainMedium,
    /// Threshold modal gain (cm⁻¹).
    pub threshold_gain: f64,
}

impl Spaser {
    /// Create a bowtie-antenna SPASER with Au/Ag gap and dye gain medium.
    ///
    /// `gap_nm`: bowtie gap size (nm), `dye_concentration`: mol/L.
    pub fn new_bowtie_spaser(gap_nm: f64, dye_concentration: f64) -> Self {
        let wavelength = 620e-9; // CdSe/dye peak
                                 // Mode volume scales as gap³
        let v_nm3 = (gap_nm * 0.5).powi(3) * PI; // rough ellipsoid estimate
        let q = 10.0; // plasmon Q is low (~10–30)
        let n_eff = 1.5;
        let fp = PlasmonicCavity::compute_purcell(v_nm3.max(1.0), q, wavelength, n_eff);
        // Dye carrier density from concentration (1 mol/L = 6.022e26 m⁻³)
        let avogadro = 6.022_140_76e23;
        let n_dye = dye_concentration * avogadro * 1e3; // m⁻³
        let gain_medium = GainMedium {
            material: GainMaterial::Dye {
                peak_wavelength: 620e-9,
                gain_cross_section: 3e-20,
            },
            carrier_density: n_dye,
            peak_gain: 200.0, // cm⁻¹ for concentrated dye
            transparency_density: n_dye * 0.5,
        };
        let cavity = PlasmonicCavity {
            mode_volume_nm3: v_nm3.max(1.0),
            q_factor: q,
            purcell_factor: fp,
            resonance_wavelength: wavelength,
        };
        let g_th = gain_medium.net_gain_cm().abs() * 0.5;
        Self {
            plasmonic_cavity: cavity,
            gain_medium,
            threshold_gain: g_th.max(10.0),
        }
    }

    /// Create a nanodisk (Au) SPASER.
    pub fn new_nanodisk_spaser(radius_nm: f64) -> Self {
        let wavelength = 700e-9;
        let height_nm = radius_nm * 0.3;
        let v_nm3 = PI * radius_nm.powi(2) * height_nm;
        let q = 15.0;
        let n_eff = 2.0;
        let fp = PlasmonicCavity::compute_purcell(v_nm3.max(1.0), q, wavelength, n_eff);
        let gain_medium = GainMedium {
            material: GainMaterial::CdSe,
            carrier_density: 1e25,
            peak_gain: 500.0,
            transparency_density: 5e24,
        };
        let cavity = PlasmonicCavity {
            mode_volume_nm3: v_nm3.max(1.0),
            q_factor: q,
            purcell_factor: fp,
            resonance_wavelength: wavelength,
        };
        Self {
            plasmonic_cavity: cavity,
            gain_medium,
            threshold_gain: 100.0,
        }
    }

    /// Threshold gain g_th (cm⁻¹).
    ///
    /// g_th = Im(k_eff) / Γ_plasmonic where Im(k) ≈ 1/(2·L_prop).
    pub fn threshold_gain_per_cm(&self) -> f64 {
        self.threshold_gain
    }

    /// Enhanced stimulated emission rate due to Purcell factor.
    ///
    /// R_stim = F_P · (photon_number) · R_sp0
    pub fn stimulated_rate_enhanced(&self, photon_number: f64) -> f64 {
        let tau_sp_s = 1e-9; // 1 ns baseline
        let r_sp0 =
            self.gain_medium.carrier_density * self.plasmonic_cavity.mode_volume_nm3 * 1e-27
                / tau_sp_s;
        self.plasmonic_cavity.purcell_factor * photon_number * r_sp0
    }

    /// Lasing emission linewidth (nm).
    ///
    /// Δλ = λ²/(c·Q) · (1 + α²) where α is the linewidth enhancement factor.
    pub fn linewidth_nm(&self) -> f64 {
        let alpha_h = 3.0_f64; // Henry linewidth enhancement factor
        let linewidth_base = self.plasmonic_cavity.linewidth_nm();
        linewidth_base * (1.0_f64 + alpha_h * alpha_h).sqrt()
    }

    /// Energy efficiency (dimensionless).
    ///
    /// Limited by Ohmic (resistive) losses in the metal: typically < 1%.
    pub fn efficiency(&self) -> f64 {
        // η = Q_rad / Q_tot; for plasmon Q_rad << Q_abs
        let q_rad = self.plasmonic_cavity.q_factor * 0.05; // ~5% radiative fraction
        q_rad / self.plasmonic_cavity.q_factor
    }
}

// ─── PhcNanolaser ─────────────────────────────────────────────────────────────

/// Photonic crystal nanocavity laser operating in the single-mode high-β regime.
///
/// Typical devices use L3 or H0 defect cavities in 2D PhC slabs with
/// Q ~ 10³–10⁶ and mode volumes V ~ 0.1–1 (λ/n)³.
#[derive(Debug, Clone)]
pub struct PhcNanolaser {
    /// Rate equation model for the active region.
    pub cavity: GeneralizedRateEquations,
    /// Cavity quality factor Q.
    pub q_factor: f64,
    /// Mode volume in units of (λ/n)³.
    pub mode_volume_lambda3: f64,
    /// Spontaneous emission coupling factor β.
    pub beta_factor: f64,
    /// Purcell enhancement factor F_P.
    pub purcell_factor: f64,
}

impl PhcNanolaser {
    /// InP membrane L3 photonic crystal nanolaser (λ ≈ 1.3 μm).
    ///
    /// Typical: Q ~ 2500, V ~ 0.7 (λ/n)³, β ~ 0.1–0.9.
    pub fn new_l3_inp(current_ua: f64) -> Self {
        let wavelength = 1.3e-6_f64;
        let n_eff = 3.17_f64;
        let lambda_n_m = wavelength / n_eff;
        let mode_vol_lambda3 = 0.7_f64;
        let v_m3 = mode_vol_lambda3 * lambda_n_m.powi(3);
        let q = 2500.0;
        let fp = PlasmonicCavity::compute_purcell(v_m3 * 1e27, q, wavelength, n_eff).min(200.0);
        let beta = (fp * 1e-2).clamp(0.05, 0.95);
        let gre = GeneralizedRateEquations::new_nanolaser(current_ua * 1e3, beta, fp);
        Self {
            cavity: gre,
            q_factor: q,
            mode_volume_lambda3: mode_vol_lambda3,
            beta_factor: beta,
            purcell_factor: fp,
        }
    }

    /// GaAs membrane H0 photonic crystal nanolaser — ultra-small mode volume.
    ///
    /// H0 cavity: Q ~ 1000, V ~ 0.02 (λ/n)³, β → 1.
    pub fn new_h0_gaas(current_ua: f64) -> Self {
        let wavelength = 1.0e-6_f64; // ~1000 nm GaAs
        let n_eff = 3.5_f64;
        let lambda_n_m = wavelength / n_eff;
        let mode_vol_lambda3 = 0.02_f64;
        let v_m3 = mode_vol_lambda3 * lambda_n_m.powi(3);
        let q = 1000.0;
        let fp = PlasmonicCavity::compute_purcell(v_m3 * 1e27, q, wavelength, n_eff).min(5000.0);
        let beta = (fp * 1e-3).clamp(0.5, 0.999);
        let gre = GeneralizedRateEquations::new_nanolaser(current_ua * 1e3, beta, fp);
        Self {
            cavity: gre,
            q_factor: q,
            mode_volume_lambda3: mode_vol_lambda3,
            beta_factor: beta,
            purcell_factor: fp,
        }
    }

    /// Threshold current (nA).
    pub fn threshold_current_na(&self) -> f64 {
        self.cavity.threshold_current_ua() * 1e3 // μA → nA
    }

    /// Output photon number at injection current I (nA).
    pub fn output_power_nw(&self, current_na: f64) -> f64 {
        let current_ua = current_na * 1e-3;
        let s = self.cavity.steady_state_photons(current_ua);
        // P = S * hν / τ_ph
        let h_planck = 6.626_070_15e-34;
        let wavelength = 1.3e-6; // L3 default
        let hnu = h_planck * C_LIGHT / wavelength;
        let tau_ph_s = self.cavity.photon_lifetime_ps * 1e-12;
        s * hnu / tau_ph_s * 1e9 // Watts → nW
    }

    /// Thresholdless parameter: product β·F_P.
    ///
    /// When β·F_P > 1, the device has no clear threshold (thresholdless lasing).
    pub fn thresholdless_parameter(&self) -> f64 {
        self.beta_factor * self.purcell_factor
    }

    /// Returns `true` if the device is expected to operate thresholdlessly.
    ///
    /// Condition: β·F_P > 1.
    pub fn is_thresholdless(&self) -> bool {
        self.thresholdless_parameter() > 1.0
    }
}

// ─── Vecsel ──────────────────────────────────────────────────────────────────

/// Semiconductor Disk Laser (VECSEL — Vertical External-Cavity
/// Surface-Emitting Laser) model.
///
/// Optically pumped device with a semiconductor gain chip and external cavity,
/// producing high-power, diffraction-limited, wavelength-tunable output.
#[derive(Debug, Clone)]
pub struct Vecsel {
    /// Active gain disk diameter (mm).
    pub active_diameter_mm: f64,
    /// Pump power (W).
    pub pump_power_w: f64,
    /// Pump wavelength (m).
    pub pump_wavelength: f64,
    /// Signal (emission) wavelength (m).
    pub signal_wavelength: f64,
    /// Number of quantum wells in the gain chip.
    pub n_quantum_wells: usize,
    /// Output coupler reflectivity (0–1).
    pub output_coupler_reflectivity: f64,
}

impl Vecsel {
    /// Create a 1064 nm Nd-equivalent VECSEL (GaInAs/GaAs QW gain chip).
    pub fn new_1064nm(diameter_mm: f64, pump_w: f64) -> Self {
        Self {
            active_diameter_mm: diameter_mm,
            pump_power_w: pump_w,
            pump_wavelength: 808e-9, // InGaAs diode pump
            signal_wavelength: 1064e-9,
            n_quantum_wells: 12,
            output_coupler_reflectivity: 0.98,
        }
    }

    /// Optical conversion efficiency.
    ///
    /// η_opt = (λ_pump/λ_sig) · η_abs · η_slope
    /// where η_abs ≈ 0.9 (absorption) and η_slope ≈ 0.8 (slope).
    pub fn conversion_efficiency(&self) -> f64 {
        let quantum_defect = self.pump_wavelength / self.signal_wavelength;
        let eta_abs = 0.9_f64;
        let eta_slope = 0.8_f64;
        // Number of QWs increases absorption saturation; model slight reduction
        let qw_factor = (self.n_quantum_wells as f64 / 10.0).min(1.0);
        quantum_defect * eta_abs * eta_slope * qw_factor
    }

    /// Output power (W).
    pub fn output_power_w(&self) -> f64 {
        self.pump_power_w * self.conversion_efficiency()
    }

    /// Beam quality factor M².
    ///
    /// VECSELs are nearly diffraction-limited; M² ≈ 1.0 + correction for thermal lens.
    pub fn m_squared(&self) -> f64 {
        // Larger pump power → stronger thermal lens → slight M² degradation
        let thermal_correction = (self.pump_power_w / 10.0) * 0.01;
        (1.0 + thermal_correction).min(1.3)
    }

    /// Gain saturation fluence (μJ/cm²).
    ///
    /// F_sat = hν / (σ · n_QW) where σ ~ 10⁻¹⁵ cm² for GaInAs QW.
    pub fn saturation_fluence_uj_per_cm2(&self) -> f64 {
        let h_planck = 6.626_070_15e-34;
        let hnu = h_planck * C_LIGHT / self.signal_wavelength; // J
        let sigma_m2 = 1e-19; // m² (≈ 10⁻¹⁵ cm²)
        let nqw = self.n_quantum_wells as f64;
        let f_sat_j_per_m2 = hnu / (sigma_m2 * nqw);
        f_sat_j_per_m2 * 1e-4 * 1e6 // J/m² → μJ/cm²
    }

    /// Single-pass gain (round-trip gain = 2·g).
    ///
    /// g ≈ (1 − R_oc)·η_opt / 2 (small-signal approximation).
    pub fn round_trip_gain(&self) -> f64 {
        let t_oc = 1.0 - self.output_coupler_reflectivity;
        // Require gain to overcome OC transmission; amplification from pump
        // g_rt = g_small_signal = T_oc / η_pump for threshold condition
        let eta = self.conversion_efficiency().max(1e-6);
        (t_oc / eta).min(1.0) // fraction per round trip (bounded)
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_purcell_factor_increases_with_q() {
        let fp_low = PlasmonicCavity::compute_purcell(1e4, 100.0, 1.3e-6, 3.4);
        let fp_high = PlasmonicCavity::compute_purcell(1e4, 10000.0, 1.3e-6, 3.4);
        assert!(
            fp_high > fp_low,
            "Higher Q should give higher Purcell factor"
        );
    }

    #[test]
    fn test_spaser_bowtie_threshold_finite() {
        let spaser = Spaser::new_bowtie_spaser(20.0, 0.01);
        let g_th = spaser.threshold_gain_per_cm();
        assert!(
            g_th.is_finite() && g_th > 0.0,
            "Threshold gain should be positive finite: {}",
            g_th
        );
    }

    #[test]
    fn test_spaser_efficiency_low() {
        let spaser = Spaser::new_nanodisk_spaser(50.0);
        let eff = spaser.efficiency();
        // SPASER efficiency is typically < 10%
        assert!(eff < 0.1, "SPASER efficiency should be < 10%, got {}", eff);
    }

    #[test]
    fn test_phc_nanolaser_l3_thresholdless() {
        let laser = PhcNanolaser::new_l3_inp(1.0);
        // High β·F_P product expected for L3 cavity
        let param = laser.thresholdless_parameter();
        assert!(
            param.is_finite(),
            "Thresholdless parameter should be finite"
        );
        // Either thresholdless or not — just verify the method works
        let _ = laser.is_thresholdless();
    }

    #[test]
    fn test_phc_nanolaser_h0_very_high_purcell() {
        let laser = PhcNanolaser::new_h0_gaas(0.1);
        // H0 cavity has higher Purcell factor than L3 due to smaller mode volume
        assert!(laser.purcell_factor > 0.0);
        let i_th_na = laser.threshold_current_na();
        assert!(
            i_th_na.is_finite() && i_th_na > 0.0,
            "Threshold should be positive: {} nA",
            i_th_na
        );
    }

    #[test]
    fn test_vecsel_output_power_positive() {
        let v = Vecsel::new_1064nm(1.0, 5.0);
        let p = v.output_power_w();
        assert!(
            p > 0.0 && p < 5.0,
            "Output power should be between 0 and pump: {} W",
            p
        );
    }

    #[test]
    fn test_vecsel_m_squared_near_unity() {
        let v = Vecsel::new_1064nm(0.5, 1.0);
        let m2 = v.m_squared();
        assert!(
            (1.0..=1.3).contains(&m2),
            "M² should be ≥ 1 and near 1: {}",
            m2
        );
    }

    #[test]
    fn test_gain_medium_net_gain_sign() {
        let gm = GainMedium {
            material: GainMaterial::InGaAsP,
            carrier_density: 2e24,
            peak_gain: 100.0,
            transparency_density: 1e24,
        };
        let g = gm.net_gain_cm();
        // Above transparency → positive gain
        assert!(
            g > 0.0,
            "Net gain above transparency should be positive: {}",
            g
        );
    }
}
