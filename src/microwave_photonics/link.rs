/// Analog photonic link model for RF-over-fiber systems.
///
/// Models the RF gain, noise figure, spurious-free dynamic range (SFDR),
/// and intercept points for Mach-Zehnder modulator (MZM), phase modulator,
/// and electro-absorption modulator (EAM) based analog photonic links.
use std::f64::consts::PI;

/// Physical constants
const BOLTZMANN: f64 = 1.380_649e-23; // J/K
const SPEED_OF_LIGHT: f64 = 2.997_924_58e8; // m/s
const ELECTRON_CHARGE: f64 = 1.602_176_634e-19; // C
const TEMPERATURE_K: f64 = 290.0; // K (standard noise temperature)

/// Electro-optic modulator type for an analog photonic link.
#[derive(Debug, Clone)]
pub enum EoModulatorType {
    /// Mach-Zehnder intensity modulator.
    Mzm {
        /// Half-wave voltage \[V\].
        vpi: f64,
        /// Insertion loss \[dB\].
        insertion_loss_db: f64,
        /// DC bias operating point.
        bias_point: MzmBias,
    },
    /// Phase modulator.
    PhaseModulator {
        /// Half-wave voltage \[V\].
        vpi: f64,
    },
    /// Electro-absorption modulator.
    Eam {
        /// Extinction ratio \[dB\].
        extinction_ratio_db: f64,
        /// Insertion loss \[dB\].
        insertion_loss_db: f64,
    },
}

/// DC bias operating point for a Mach-Zehnder modulator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MzmBias {
    /// Quadrature push-pull (π/2) — maximum linearity for intensity links.
    QuadraturePush,
    /// Quadrature pull (−π/2).
    QuadraturePull,
    /// Minimum transmission (π).
    MinimumTransmission,
    /// Maximum transmission (0).
    MaximumTransmission,
    /// Custom bias point \[radians\].
    Custom(f64),
}

impl MzmBias {
    /// Return the bias angle in radians.
    pub fn radians(self) -> f64 {
        match self {
            MzmBias::QuadraturePush => PI / 2.0,
            MzmBias::QuadraturePull => -PI / 2.0,
            MzmBias::MinimumTransmission => PI,
            MzmBias::MaximumTransmission => 0.0,
            MzmBias::Custom(phi) => phi,
        }
    }

    /// Optical transmission at the bias point (normalized 0–1).
    pub fn transmission(self) -> f64 {
        let phi = self.radians();
        0.5 * (1.0 + phi.cos())
    }
}

/// Photodetector parameters.
#[derive(Debug, Clone)]
pub struct PhotodetectorParams {
    /// Responsivity \[A/W\].
    pub responsivity: f64,
    /// 3-dB electrical bandwidth \[Hz\].
    pub bandwidth_hz: f64,
    /// Dark current \[A\].
    pub dark_current_a: f64,
    /// Load resistance \[Ω\] (typically 50 Ω).
    pub load_resistance: f64,
}

impl PhotodetectorParams {
    /// Create a typical InGaAs PIN photodetector for C-band links.
    pub fn typical_ingaas() -> Self {
        PhotodetectorParams {
            responsivity: 0.85,
            bandwidth_hz: 20.0e9,
            dark_current_a: 5.0e-9,
            load_resistance: 50.0,
        }
    }
}

/// Analog photonic link: RF → EO modulator → fiber → photodetector → RF.
///
/// The link is characterized by RF gain, noise figure, SFDR, and linearity
/// metrics (OIP2, OIP3, P1dB).
#[derive(Debug, Clone)]
pub struct AnalogPhotonicLink {
    /// Optical carrier wavelength \[m\].
    pub wavelength: f64,
    /// Average optical power at the photodetector input \[dBm\].
    pub optical_power_dbm: f64,
    /// Electro-optic modulator type.
    pub modulator: EoModulatorType,
    /// Fiber span length \[km\].
    pub fiber_length_km: f64,
    /// Fiber attenuation coefficient \[dB/km\].
    pub fiber_loss_db_per_km: f64,
    /// Photodetector parameters.
    pub pd: PhotodetectorParams,
}

impl AnalogPhotonicLink {
    /// Create a standard intensity-modulated direct-detection (IMDD) link
    /// with an MZM biased at quadrature.
    ///
    /// # Arguments
    /// * `wl` – optical wavelength \[m\]
    /// * `power_dbm` – optical power at PD \[dBm\]
    /// * `vpi` – MZM half-wave voltage \[V\]
    /// * `length_km` – fiber length \[km\]
    pub fn new_intensity_modulated(wl: f64, power_dbm: f64, vpi: f64, length_km: f64) -> Self {
        AnalogPhotonicLink {
            wavelength: wl,
            optical_power_dbm: power_dbm,
            modulator: EoModulatorType::Mzm {
                vpi,
                insertion_loss_db: 3.0,
                bias_point: MzmBias::QuadraturePush,
            },
            fiber_length_km: length_km,
            fiber_loss_db_per_km: 0.2,
            pd: PhotodetectorParams::typical_ingaas(),
        }
    }

    /// Average photocurrent at the photodetector \[A\].
    pub fn dc_photocurrent_a(&self) -> f64 {
        let power_w = dbm_to_watts(self.optical_power_dbm);
        let transmission = self.modulator_bias_transmission();
        power_w * transmission * self.pd.responsivity
    }

    /// Optical transmission factor at the modulator bias point (dimensionless, 0–1).
    fn modulator_bias_transmission(&self) -> f64 {
        match &self.modulator {
            EoModulatorType::Mzm { bias_point, .. } => bias_point.transmission(),
            EoModulatorType::PhaseModulator { .. } => 1.0,
            EoModulatorType::Eam {
                extinction_ratio_db,
                ..
            } => {
                // Bias at midpoint of the transfer curve
                let er_linear = db_to_linear(*extinction_ratio_db);
                (1.0 + 1.0 / er_linear) / 2.0
            }
        }
    }

    /// Effective Vπ (half-wave voltage) of the modulator \[V\].
    fn vpi(&self) -> f64 {
        match &self.modulator {
            EoModulatorType::Mzm { vpi, .. } => *vpi,
            EoModulatorType::PhaseModulator { vpi } => *vpi,
            EoModulatorType::Eam { .. } => {
                // EAM doesn't have a Vπ; return a nominal value based on ER
                1.0
            }
        }
    }

    /// RF gain of the link \[dB\].
    ///
    /// For an MZM link biased at quadrature:
    ///   G_RF = (π·R_f·R_L·I_dc / (2·Vπ))²
    /// where R_f = PD responsivity \[A/W\], R_L = load resistance \[Ω\].
    pub fn rf_gain_db(&self) -> f64 {
        let i_dc = self.dc_photocurrent_a();
        let gain_linear = match &self.modulator {
            EoModulatorType::Mzm {
                vpi, bias_point, ..
            } => {
                let phi0 = bias_point.radians();
                // slope of MZM transfer function at bias point
                let slope = (-phi0.sin()) * PI / (2.0 * vpi);
                // RF gain = (slope * R_L * I_dc_per_unit_optical)^2 ...
                // More precisely: G = (π R_pd R_L / 2 Vpi)^2 * I_dc^2 / 1 at quadrature
                let g = (PI * self.pd.responsivity * self.pd.load_resistance / (2.0 * vpi)).powi(2)
                    * (phi0.cos().powi(2))
                    + (slope * self.pd.load_resistance * i_dc / self.pd.responsivity).powi(2) * 0.0;
                // Simplified quadrature gain
                let _ = g;
                let numerator = PI * self.pd.load_resistance * i_dc;
                let denominator = 2.0 * vpi;
                (numerator / denominator).powi(2)
            }
            EoModulatorType::PhaseModulator { vpi } => {
                // Phase modulation → intensity via discriminator; approx same form
                let numerator = PI * self.pd.load_resistance * i_dc;
                let denominator = 2.0 * vpi;
                (numerator / denominator).powi(2)
            }
            EoModulatorType::Eam {
                extinction_ratio_db,
                ..
            } => {
                let er = db_to_linear(*extinction_ratio_db);
                // EAM slope efficiency ≈ (er - 1)/(er + 1) * I_dc / V_swing
                // Use 1V swing approximation
                let v_swing = 1.0;
                let slope = (er - 1.0) / (er + 1.0) * i_dc / v_swing;
                (slope * self.pd.load_resistance).powi(2)
            }
        };
        10.0 * gain_linear.log10()
    }

    /// Noise figure of the link \[dB\].
    ///
    /// For a shot-noise limited IMDD MZM link:
    ///   NF = 1/G * (shot noise PSD) / (kT) + thermal contributions
    pub fn noise_figure_db(&self) -> f64 {
        let i_dc = self.dc_photocurrent_a();
        let g_linear = db_to_linear(self.rf_gain_db());
        let rl = self.pd.load_resistance;

        // Shot noise current PSD: S_shot = 2 * q * I_dc [A²/Hz]
        let s_shot = 2.0 * ELECTRON_CHARGE * i_dc;
        // Thermal noise PSD: S_thermal = 4 * k * T / R_L [A²/Hz]
        let s_thermal = 4.0 * BOLTZMANN * TEMPERATURE_K / rl;
        // Dark current noise
        let s_dark = 2.0 * ELECTRON_CHARGE * self.pd.dark_current_a;

        // Total output noise referenced to 1 Hz bandwidth
        let s_out = (s_shot + s_thermal + s_dark) * rl.powi(2); // V²/Hz

        // Noise at input (standard kT = -174 dBm/Hz)
        let s_in = BOLTZMANN * TEMPERATURE_K; // W/Hz = kT

        // NF = (output noise / G) / (kT) expressed in dB
        let nf_linear = s_out / (g_linear * s_in * rl);
        if nf_linear > 0.0 {
            10.0 * nf_linear.log10()
        } else {
            f64::INFINITY
        }
    }

    /// Spurious-free dynamic range \[dB·Hz^(2/3)\].
    ///
    /// SFDR = (2/3) * (OIP3 \[W\] / noise_floor \[W/Hz\])^(2/3) expressed in dBHz^(2/3).
    pub fn sfdr_db_hz(&self) -> f64 {
        let oip3_dbm = self.oip3_dbm();
        // Noise floor = kT * NF [dBm/Hz]
        let nf_db = self.noise_figure_db();
        let noise_floor_dbm_per_hz = -174.0 + nf_db; // dBm/Hz
                                                     // SFDR = (2/3)(OIP3 - noise_floor) in dB·Hz^(2/3)
        (2.0 / 3.0) * (oip3_dbm - noise_floor_dbm_per_hz)
    }

    /// Second-order output intercept point OIP2 \[dBm\].
    ///
    /// For an MZM at quadrature the second-order nonlinearity is suppressed;
    /// the dominant term gives OIP2 = 2·Vπ·I_dc·R_L / (π·some_factor).
    pub fn oip2_dbm(&self) -> f64 {
        let i_dc = self.dc_photocurrent_a();
        let rl = self.pd.load_resistance;
        let vpi = self.vpi();
        // OIP2 ~ (4 Vpi I_dc R_L) / π  for MZM at quadrature (2nd order)
        let oip2_w = (4.0 * vpi * i_dc * rl) / PI;
        watts_to_dbm(oip2_w)
    }

    /// Third-order output intercept point OIP3 \[dBm\].
    ///
    /// For an MZM at quadrature:
    ///   OIP3 = (2/3) * (2·Vπ/π)^2 * I_dc * R_L
    ///
    /// Reference: Cox, "Analog Optical Links", Cambridge 2004, Ch. 4.
    pub fn oip3_dbm(&self) -> f64 {
        let i_dc = self.dc_photocurrent_a();
        let rl = self.pd.load_resistance;
        let vpi = self.vpi();
        match &self.modulator {
            EoModulatorType::Mzm { bias_point, .. } => {
                let phi0 = bias_point.radians();
                // General MZM OIP3: proportional to sin(phi0) for odd-order terms
                // At quadrature sin(π/2) = 1, so max OIP3
                let factor = phi0.sin().abs();
                let oip3_w = (2.0 / 3.0) * (2.0 * vpi / PI).powi(2) * i_dc * rl * factor;
                watts_to_dbm(oip3_w)
            }
            EoModulatorType::PhaseModulator { vpi } => {
                // Phase modulator converted to intensity: reduced OIP3
                let oip3_w = (2.0 / 3.0) * (2.0 * vpi / PI).powi(2) * i_dc * rl * 0.5;
                watts_to_dbm(oip3_w)
            }
            EoModulatorType::Eam {
                extinction_ratio_db,
                ..
            } => {
                let er = db_to_linear(*extinction_ratio_db);
                // EAM OIP3 is generally lower due to nonlinear transfer curve
                let oip3_w = (2.0 / 3.0) * i_dc.powi(2) * rl * er / (er + 1.0);
                watts_to_dbm(oip3_w)
            }
        }
    }

    /// 1-dB compression point \[dBm\] (output referred).
    ///
    /// P1dB ≈ OIP3 − 9.6 dB (for a memoryless third-order system).
    pub fn p1db_dbm(&self) -> f64 {
        self.oip3_dbm() - 9.6
    }

    /// Link bandwidth \[Hz\], limited by the photodetector and fiber dispersion.
    pub fn bandwidth_hz(&self) -> f64 {
        // PD bandwidth
        let bw_pd = self.pd.bandwidth_hz;

        // Chromatic dispersion bandwidth for fiber (SMF-28 at 1550 nm):
        // BW_dispersion ≈ 1 / (π * D * L * Δλ) — simplified to:
        // For a typical DML with Δλ ≈ 0.1 nm at 1550 nm:
        let d_ps_per_nm_km: f64 = match self.wavelength {
            wl if wl < 1.35e-9 => 0.0_f64,
            wl if wl < 1.40e-9 => -50.0_f64, // normal dispersion region
            wl if wl < 1.60e-9 => 17.0_f64,  // anomalous dispersion (SMF-28)
            _ => 20.0_f64,
        };
        let delta_lambda_nm: f64 = 0.1; // typical laser linewidth contribution
        let bw_dispersion = if d_ps_per_nm_km.abs() > 1e-10 {
            // BW (GHz) ≈ 0.44 / (|D| * L * Δλ) in appropriate units
            let d_s_per_m_km = d_ps_per_nm_km * 1e-12 / 1e-9; // s/(m·km)
            1.0 / (PI * d_s_per_m_km.abs() * self.fiber_length_km * delta_lambda_nm * 1e-9)
        } else {
            f64::INFINITY
        };

        // Modulator bandwidth (simplified: assume modulator BW >> PD BW)
        // Use minimum of PD and dispersion limits
        bw_pd.min(bw_dispersion)
    }

    /// Carrier-to-noise ratio \[dB\] for a given RF input power and carrier frequency.
    ///
    /// # Arguments
    /// * `rf_power_dbm` – RF input power \[dBm\]
    /// * `rf_freq_hz` – RF carrier frequency \[Hz\]
    pub fn cnr_db(&self, rf_power_dbm: f64, rf_freq_hz: f64) -> f64 {
        let rf_power_w = dbm_to_watts(rf_power_dbm);
        let g_linear = db_to_linear(self.rf_gain_db());
        // Signal power at output
        let signal_out = rf_power_w * g_linear;
        // Optical carrier frequency
        let nu = SPEED_OF_LIGHT / self.wavelength;
        let i_dc = self.dc_photocurrent_a();
        let rl = self.pd.load_resistance;

        // Shot noise (dominant at high optical power)
        let shot_psd = 2.0 * ELECTRON_CHARGE * i_dc * rl.powi(2); // W/Hz
                                                                  // Thermal noise
        let thermal_psd = 4.0 * BOLTZMANN * TEMPERATURE_K * rl; // W/Hz
                                                                // RIN noise: typical laser RIN ~ -155 dB/Hz
        let rin_per_hz = 1e-15_f64; // -150 dB/Hz
        let rin_psd = rin_per_hz * i_dc.powi(2) * rl.powi(2); // W/Hz

        // Total noise in a 1-Hz bandwidth
        let _ = (nu, rf_freq_hz); // used implicitly via link parameters
        let noise_1hz = shot_psd + thermal_psd + rin_psd;
        let cnr = signal_out / noise_1hz;
        10.0 * cnr.log10()
    }

    /// Compute a complete link budget summary.
    pub fn link_budget(&self) -> LinkBudget {
        LinkBudget {
            rf_gain_db: self.rf_gain_db(),
            noise_figure_db: self.noise_figure_db(),
            sfdr_db_hz: self.sfdr_db_hz(),
            bandwidth_hz: self.bandwidth_hz(),
            oip3_dbm: self.oip3_dbm(),
        }
    }
}

/// Summary of the key link performance metrics.
#[derive(Debug, Clone)]
pub struct LinkBudget {
    /// RF link gain \[dB\].
    pub rf_gain_db: f64,
    /// Link noise figure \[dB\].
    pub noise_figure_db: f64,
    /// Spurious-free dynamic range \[dB·Hz^(2/3)\].
    pub sfdr_db_hz: f64,
    /// Usable link bandwidth \[Hz\].
    pub bandwidth_hz: f64,
    /// Output third-order intercept point \[dBm\].
    pub oip3_dbm: f64,
}

impl std::fmt::Display for LinkBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Analog Photonic Link Budget ===")?;
        writeln!(f, "  RF Gain        : {:+.2} dB", self.rf_gain_db)?;
        writeln!(f, "  Noise Figure   : {:.2} dB", self.noise_figure_db)?;
        writeln!(f, "  SFDR           : {:.2} dB·Hz^(2/3)", self.sfdr_db_hz)?;
        writeln!(f, "  Bandwidth      : {:.3} GHz", self.bandwidth_hz * 1e-9)?;
        write!(f, "  OIP3           : {:+.2} dBm", self.oip3_dbm)
    }
}

// ─── Helper functions ─────────────────────────────────────────────────────────

/// Convert dBm to watts.
fn dbm_to_watts(dbm: f64) -> f64 {
    1e-3 * 10.0_f64.powf(dbm / 10.0)
}

/// Convert watts to dBm.
fn watts_to_dbm(w: f64) -> f64 {
    if w > 0.0 {
        10.0 * (w * 1e3).log10()
    } else {
        f64::NEG_INFINITY
    }
}

/// Convert dB ratio to linear scale.
fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 10.0)
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn default_link() -> AnalogPhotonicLink {
        // Optical power: 0 dBm (1 mW) at PD, Vpi = 5V, 10 km SMF-28
        AnalogPhotonicLink::new_intensity_modulated(1550e-9, 0.0, 5.0, 10.0)
    }

    #[test]
    fn test_dc_photocurrent() {
        let link = default_link();
        // At quadrature, T = 0.5; I_dc = 1e-3 * 0.5 * 0.85 = 0.425 mA
        let i_dc = link.dc_photocurrent_a();
        assert_abs_diff_eq!(i_dc, 0.425e-3, epsilon = 1e-6);
    }

    #[test]
    fn test_rf_gain_negative() {
        // Standard IMDD MZM links have negative RF gain for low optical power
        let link = default_link();
        let gain = link.rf_gain_db();
        // At 0 dBm optical power the gain is typically in range −30 to −10 dB
        assert!(gain < 0.0, "RF gain should be negative at 0 dBm: {}", gain);
    }

    #[test]
    fn test_rf_gain_increases_with_optical_power() {
        let link_low = AnalogPhotonicLink::new_intensity_modulated(1550e-9, 0.0, 5.0, 10.0);
        let link_high = AnalogPhotonicLink::new_intensity_modulated(1550e-9, 10.0, 5.0, 10.0);
        assert!(
            link_high.rf_gain_db() > link_low.rf_gain_db(),
            "Higher optical power should give higher RF gain"
        );
    }

    #[test]
    fn test_oip3_increases_with_optical_power() {
        let link_low = AnalogPhotonicLink::new_intensity_modulated(1550e-9, 0.0, 5.0, 10.0);
        let link_high = AnalogPhotonicLink::new_intensity_modulated(1550e-9, 10.0, 5.0, 10.0);
        assert!(link_high.oip3_dbm() > link_low.oip3_dbm());
    }

    #[test]
    fn test_p1db_below_oip3() {
        let link = default_link();
        let p1db = link.p1db_dbm();
        let oip3 = link.oip3_dbm();
        // P1dB should be ~9.6 dB below OIP3
        assert_abs_diff_eq!(oip3 - p1db, 9.6, epsilon = 0.01);
    }

    #[test]
    fn test_link_budget_fields() {
        let link = default_link();
        let budget = link.link_budget();
        // All fields should be finite
        assert!(budget.rf_gain_db.is_finite());
        assert!(budget.noise_figure_db.is_finite());
        assert!(budget.sfdr_db_hz.is_finite());
        assert!(budget.bandwidth_hz > 0.0);
        assert!(budget.oip3_dbm.is_finite());
    }

    #[test]
    fn test_mzm_bias_quadrature_transmission() {
        let bias = MzmBias::QuadraturePush;
        // T = 0.5 * (1 + cos(π/2)) = 0.5
        assert_abs_diff_eq!(bias.transmission(), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_mzm_bias_max_transmission() {
        let bias = MzmBias::MaximumTransmission;
        // T = 0.5 * (1 + cos(0)) = 1.0
        assert_abs_diff_eq!(bias.transmission(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mzm_bias_min_transmission() {
        let bias = MzmBias::MinimumTransmission;
        // T = 0.5 * (1 + cos(π)) = 0.0
        assert_abs_diff_eq!(bias.transmission(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bandwidth_limited_by_pd() {
        let link = default_link();
        // PD bandwidth is 20 GHz; link bandwidth must not exceed PD bandwidth
        assert!(link.bandwidth_hz() <= link.pd.bandwidth_hz + 1.0);
    }

    #[test]
    fn test_sfdr_reasonable_range() {
        let link = default_link();
        let sfdr = link.sfdr_db_hz();
        // Typical IMDD links: SFDR 80–130 dB·Hz^(2/3)
        assert!(sfdr > 50.0 && sfdr < 180.0, "SFDR={:.1} dB·Hz^(2/3)", sfdr);
    }

    #[test]
    fn test_dbm_watts_roundtrip() {
        for dbm in [-30.0_f64, -10.0, 0.0, 10.0, 20.0] {
            let w = dbm_to_watts(dbm);
            let back = watts_to_dbm(w);
            assert_abs_diff_eq!(back, dbm, epsilon = 1e-9);
        }
    }
}
