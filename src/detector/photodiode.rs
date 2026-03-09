//! PIN photodiode and avalanche photodiode (APD) models.
//!
//! Implements first-principles detector physics: responsivity, quantum
//! efficiency, frequency response, shot/thermal noise, NEP, D*, and SNR.

use crate::error::OxiPhotonError;

// ── Physical constants ────────────────────────────────────────────────────────
const H_PLANCK: f64 = 6.626_070_15e-34; // J·s
const E_CHARGE: f64 = 1.602_176_634e-19; // C
const C0: f64 = 2.997_924_58e8; // m/s
const KB: f64 = 1.380_649e-23; // J/K

// ── PIN Photodiode ────────────────────────────────────────────────────────────

/// First-principles model of a PIN photodiode.
///
/// # Reference quantities
/// - Photocurrent: `I = R * P`
/// - Frequency response (1st-order RC): `H(f) = 1 / sqrt(1 + (f/f_3dB)²)`
/// - Shot noise density: `i_shot = sqrt(2·e·(I_sig + I_dark))`  (A/√Hz)
/// - Johnson noise density: `i_J = sqrt(4·k_B·T / R_L)`  (A/√Hz)
#[derive(Debug, Clone)]
pub struct PinPhotodiode {
    /// Responsivity at peak wavelength (A/W).
    pub responsivity_a_per_w: f64,
    /// Peak (design) wavelength (nm).
    pub peak_wavelength_nm: f64,
    /// 3-dB electrical bandwidth (GHz).
    pub bandwidth_ghz: f64,
    /// Dark current (nA).
    pub dark_current_na: f64,
    /// Junction capacitance (fF).
    pub capacitance_ff: f64,
    /// Series resistance (Ω).
    pub series_resistance_ohm: f64,
    /// Active area (μm²).
    pub active_area_um2: f64,
}

impl PinPhotodiode {
    /// Create a PIN photodiode with explicit parameters.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] when any parameter is
    /// non-positive or not finite.
    pub fn new(
        responsivity_a_per_w: f64,
        peak_wavelength_nm: f64,
        bandwidth_ghz: f64,
        dark_current_na: f64,
        capacitance_ff: f64,
        series_resistance_ohm: f64,
        active_area_um2: f64,
    ) -> Result<Self, OxiPhotonError> {
        if !responsivity_a_per_w.is_finite() || responsivity_a_per_w <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "responsivity must be positive and finite".into(),
            ));
        }
        if !peak_wavelength_nm.is_finite() || peak_wavelength_nm <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "peak_wavelength_nm must be positive and finite".into(),
            ));
        }
        if !bandwidth_ghz.is_finite() || bandwidth_ghz <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "bandwidth_ghz must be positive and finite".into(),
            ));
        }
        Ok(Self {
            responsivity_a_per_w,
            peak_wavelength_nm,
            bandwidth_ghz,
            dark_current_na,
            capacitance_ff,
            series_resistance_ohm,
            active_area_um2,
        })
    }

    /// Typical InGaAs PIN photodiode for 1550 nm telecom.
    ///
    /// R = 0.9 A/W, BW = 10 GHz, I_dark = 1 nA.
    pub fn ingaas_1550() -> Self {
        Self {
            responsivity_a_per_w: 0.9,
            peak_wavelength_nm: 1550.0,
            bandwidth_ghz: 10.0,
            dark_current_na: 1.0,
            capacitance_ff: 200.0,
            series_resistance_ohm: 10.0,
            active_area_um2: 50.0,
        }
    }

    /// Typical Si PIN photodiode for visible (900 nm).
    ///
    /// R = 0.55 A/W, BW = 1 GHz, I_dark = 5 nA.
    pub fn si_900nm() -> Self {
        Self {
            responsivity_a_per_w: 0.55,
            peak_wavelength_nm: 900.0,
            bandwidth_ghz: 1.0,
            dark_current_na: 5.0,
            capacitance_ff: 500.0,
            series_resistance_ohm: 20.0,
            active_area_um2: 200.0,
        }
    }

    // ── Quantum efficiency ────────────────────────────────────────────────

    /// Internal quantum efficiency at the design wavelength.
    ///
    /// η = R · hν / e = R · hc / (λ · e)
    pub fn quantum_efficiency(&self) -> f64 {
        let lambda_m = self.peak_wavelength_nm * 1e-9;
        self.responsivity_a_per_w * H_PLANCK * C0 / (lambda_m * E_CHARGE)
    }

    /// Quantum efficiency at an arbitrary wavelength `lambda_nm`.
    ///
    /// Scales linearly with wavelength relative to peak (constant photon
    /// absorption coefficient assumption, η ∝ λ for fixed R):
    /// η(λ) = η_peak · (λ / λ_peak)
    pub fn quantum_efficiency_at(&self, lambda_nm: f64) -> f64 {
        let eta_peak = self.quantum_efficiency();
        eta_peak * (lambda_nm / self.peak_wavelength_nm)
    }

    // ── Signal ────────────────────────────────────────────────────────────

    /// Photocurrent for incident optical power.
    ///
    /// I_ph = R · P
    pub fn photocurrent_a(&self, power_w: f64) -> f64 {
        self.responsivity_a_per_w * power_w
    }

    // ── Frequency response ────────────────────────────────────────────────

    /// Normalised electrical frequency response (magnitude).
    ///
    /// First-order RC roll-off: H(f) = 1 / √(1 + (f / f_3dB)²)
    pub fn frequency_response(&self, freq_ghz: f64) -> f64 {
        let ratio = freq_ghz / self.bandwidth_ghz;
        1.0 / (1.0 + ratio * ratio).sqrt()
    }

    // ── Noise ─────────────────────────────────────────────────────────────

    /// Shot noise current spectral density (A/√Hz).
    ///
    /// i_shot = √(2 · e · (I_sig + I_dark))
    pub fn shot_noise_a_per_sqrt_hz(&self, signal_power_w: f64) -> f64 {
        let i_sig = self.photocurrent_a(signal_power_w);
        let i_dark = self.dark_current_na * 1e-9;
        (2.0 * E_CHARGE * (i_sig + i_dark)).sqrt()
    }

    /// Thermal (Johnson) noise current spectral density (A/√Hz).
    ///
    /// i_J = √(4 · k_B · T / R_L)
    pub fn thermal_noise_a_per_sqrt_hz(&self, temperature_k: f64, load_resistance_ohm: f64) -> f64 {
        (4.0 * KB * temperature_k / load_resistance_ohm).sqrt()
    }

    /// Total noise current spectral density (A/√Hz) — quadrature sum.
    pub fn total_noise_a_per_sqrt_hz(&self, power_w: f64, temp_k: f64, r_load: f64) -> f64 {
        let i_shot = self.shot_noise_a_per_sqrt_hz(power_w);
        let i_th = self.thermal_noise_a_per_sqrt_hz(temp_k, r_load);
        (i_shot * i_shot + i_th * i_th).sqrt()
    }

    // ── Figures of merit ─────────────────────────────────────────────────

    /// Noise-equivalent power (W/√Hz).
    ///
    /// NEP = i_noise / R  (at zero signal, dark-current dominated)
    pub fn nep_w_per_sqrt_hz(&self, temp_k: f64, r_load: f64) -> f64 {
        let i_noise = self.total_noise_a_per_sqrt_hz(0.0, temp_k, r_load);
        i_noise / self.responsivity_a_per_w
    }

    /// Specific detectivity D* (cm · √Hz / W).
    ///
    /// D* = √(A_det) / NEP, with area converted from μm² to cm².
    pub fn d_star_cm_sqrt_hz_per_w(&self, temp_k: f64, r_load: f64) -> f64 {
        let nep = self.nep_w_per_sqrt_hz(temp_k, r_load);
        // μm² → cm²: 1 μm² = 1e-8 cm²
        let area_cm2 = self.active_area_um2 * 1e-8;
        area_cm2.sqrt() / nep
    }

    /// Approximate saturation output power (dBm).
    ///
    /// Assumes linear operation up to I_max = 10 mA (device-limited);
    /// P_sat = I_max / R.
    pub fn saturation_power_dbm(&self) -> f64 {
        let i_max_a = 10e-3; // 10 mA typical linear limit
        let p_sat_w = i_max_a / self.responsivity_a_per_w;
        10.0 * (p_sat_w / 1e-3).log10()
    }

    /// Rise time (10%–90%) from bandwidth: t_r = 0.35 / BW (ps).
    pub fn rise_time_ps(&self) -> f64 {
        0.35 / (self.bandwidth_ghz * 1e9) * 1e12 // convert to ps
    }

    /// Electrical signal-to-noise ratio (dB) for given optical power and
    /// detection bandwidth.
    pub fn snr_db(&self, power_w: f64, bandwidth_ghz: f64, temp_k: f64, r_load: f64) -> f64 {
        let i_sig = self.photocurrent_a(power_w);
        let noise_density = self.total_noise_a_per_sqrt_hz(power_w, temp_k, r_load);
        let bandwidth_hz = bandwidth_ghz * 1e9;
        let i_noise = noise_density * bandwidth_hz.sqrt();
        if i_noise == 0.0 {
            return f64::INFINITY;
        }
        20.0 * (i_sig / i_noise).log10()
    }

    /// Minimum detectable power (dBm) at SNR = 1.
    ///
    /// MDP = NEP · √(B)
    pub fn mdp_dbm(&self, bandwidth_ghz: f64, temp_k: f64, r_load: f64) -> f64 {
        let nep = self.nep_w_per_sqrt_hz(temp_k, r_load);
        let bandwidth_hz = bandwidth_ghz * 1e9;
        let mdp_w = nep * bandwidth_hz.sqrt();
        10.0 * (mdp_w / 1e-3).log10()
    }
}

// ── APD ───────────────────────────────────────────────────────────────────────

/// Avalanche photodiode (APD) model.
///
/// Wraps a [`PinPhotodiode`] and adds impact-ionisation gain physics.
///
/// # Key relationships
/// - Effective responsivity: R_eff = R · M
/// - Excess noise factor: F(M) = M^x  (McIntyre model, simplified)
/// - APD shot noise: i² = 2·e·(R·P·M + I_dark)·F(M)·M²  per Hz
/// - Gain-bandwidth product: M · BW = const
#[derive(Debug, Clone)]
pub struct AvalanchePhotodiode {
    /// Underlying PIN detector (unity-gain parameters).
    pub pin: PinPhotodiode,
    /// Avalanche multiplication gain M (dimensionless, ≥ 1).
    pub gain_m: f64,
    /// McIntyre excess-noise exponent x: F(M) = M^x.
    /// Typical values: 0.3 (InGaAs), 0.5–1.0 (Si).
    pub excess_noise_factor_x: f64,
    /// Gain-bandwidth product (GHz).
    pub gain_bandwidth_product_ghz: f64,
}

impl AvalanchePhotodiode {
    /// Create an APD from a PIN model plus gain parameters.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] when gain < 1 or
    /// excess-noise exponent is not in (0, 2].
    pub fn new(
        pin: PinPhotodiode,
        gain_m: f64,
        excess_noise_factor_x: f64,
        gain_bandwidth_product_ghz: f64,
    ) -> Result<Self, OxiPhotonError> {
        if gain_m < 1.0 {
            return Err(OxiPhotonError::NumericalError(
                "APD gain must be >= 1".into(),
            ));
        }
        if excess_noise_factor_x <= 0.0 || excess_noise_factor_x > 2.0 {
            return Err(OxiPhotonError::NumericalError(
                "excess_noise_factor_x must be in (0, 2]".into(),
            ));
        }
        Ok(Self {
            pin,
            gain_m,
            excess_noise_factor_x,
            gain_bandwidth_product_ghz,
        })
    }

    /// Typical InGaAs APD for 1550 nm: M = 10, x = 0.7, GBP = 150 GHz.
    pub fn ingaas_apd() -> Self {
        Self {
            pin: PinPhotodiode::ingaas_1550(),
            gain_m: 10.0,
            excess_noise_factor_x: 0.7,
            gain_bandwidth_product_ghz: 150.0,
        }
    }

    // ── Performance metrics ───────────────────────────────────────────────

    /// Effective responsivity including avalanche gain (A/W).
    pub fn effective_responsivity(&self) -> f64 {
        self.pin.responsivity_a_per_w * self.gain_m
    }

    /// McIntyre excess noise factor F(M) = M^x.
    pub fn excess_noise_factor(&self) -> f64 {
        self.gain_m.powf(self.excess_noise_factor_x)
    }

    /// Effective 3-dB bandwidth at current gain: BW = GBP / M.
    pub fn effective_bandwidth_ghz(&self) -> f64 {
        self.gain_bandwidth_product_ghz / self.gain_m
    }

    /// APD shot noise current spectral density (A/√Hz).
    ///
    /// i²_noise = 2·e·(R·P·M + I_dark)·F(M)·M²
    pub fn apd_noise_a_per_sqrt_hz(&self, power_w: f64) -> f64 {
        let i_primary = self.pin.photocurrent_a(power_w); // before gain
        let i_dark = self.pin.dark_current_na * 1e-9;
        let f_m = self.excess_noise_factor();
        let m2 = self.gain_m * self.gain_m;
        // total primary current (signal + dark), then multiplied by M²·F(M)
        let noise_sq = 2.0 * E_CHARGE * (i_primary + i_dark) * f_m * m2;
        noise_sq.sqrt()
    }

    /// Optimum gain that maximises SNR for a given signal power and load.
    ///
    /// Analytically: M_opt = (I_th² / (e·R·P·F_slope))^(1/3)
    /// where I_th is thermal-noise current squared per Hz and
    /// F_slope ≈ x·M^(x-1) (linearised).  Solved iteratively here.
    pub fn optimum_gain(&self, signal_power_w: f64, temp_k: f64, r_load: f64) -> f64 {
        // Thermal noise spectral density (A²/Hz)
        let i_th_sq = 4.0 * KB * temp_k / r_load;
        let r = self.pin.responsivity_a_per_w;
        let i_sig = r * signal_power_w;
        let x = self.excess_noise_factor_x;

        // Sweep M from 1 to 1000 and find maximum SNR
        let mut best_m = 1.0_f64;
        let mut best_snr = f64::NEG_INFINITY;

        let steps = 500usize;
        for k in 1..=steps {
            let m = 1.0 + (k as f64 / steps as f64) * 999.0;
            let f_m = m.powf(x);
            let i_dark = self.pin.dark_current_na * 1e-9;
            // shot noise A²/Hz
            let i_shot_sq = 2.0 * E_CHARGE * (i_sig + i_dark) * f_m * m * m;
            // signal power ∝ (R·P·M)²
            let i_signal_sq = (r * signal_power_w * m).powi(2);
            let snr = i_signal_sq / (i_shot_sq + i_th_sq);
            if snr > best_snr {
                best_snr = snr;
                best_m = m;
            }
        }
        best_m
    }

    /// APD signal-to-noise ratio (dB).
    pub fn snr_db(&self, power_w: f64, bw_ghz: f64, temp_k: f64, r_load: f64) -> f64 {
        let i_sig = self.effective_responsivity() * power_w;
        let i_shot_density = self.apd_noise_a_per_sqrt_hz(power_w);
        let i_th_density = self.pin.thermal_noise_a_per_sqrt_hz(temp_k, r_load);
        let bw_hz = bw_ghz * 1e9;
        let i_noise =
            ((i_shot_density * i_shot_density + i_th_density * i_th_density) * bw_hz).sqrt();
        if i_noise == 0.0 {
            return f64::INFINITY;
        }
        20.0 * (i_sig / i_noise).log10()
    }

    /// Noise-equivalent power (W/√Hz) for the APD.
    pub fn nep_w_per_sqrt_hz(&self, temp_k: f64, r_load: f64) -> f64 {
        let i_noise = {
            let i_shot = self.apd_noise_a_per_sqrt_hz(0.0);
            let i_th = self.pin.thermal_noise_a_per_sqrt_hz(temp_k, r_load);
            (i_shot * i_shot + i_th * i_th).sqrt()
        };
        i_noise / self.effective_responsivity()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn ingaas() -> PinPhotodiode {
        PinPhotodiode::ingaas_1550()
    }

    #[test]
    fn test_ingaas_qe_reasonable() {
        let det = ingaas();
        let qe = det.quantum_efficiency();
        // InGaAs at 1550 nm with R=0.9 A/W → η ≈ 72 %
        assert!(qe > 0.60 && qe < 0.95, "QE = {qe:.3} outside [0.60, 0.95]");
    }

    #[test]
    fn test_photocurrent_linear() {
        let det = ingaas();
        let p = 1e-3; // 1 mW
        let i = det.photocurrent_a(p);
        assert_relative_eq!(i, det.responsivity_a_per_w * p, epsilon = 1e-15);
        // linearity: double power → double current
        let i2 = det.photocurrent_a(2.0 * p);
        assert_relative_eq!(i2, 2.0 * i, epsilon = 1e-15);
    }

    #[test]
    fn test_shot_noise_scaling() {
        let det = ingaas();
        let i1 = det.shot_noise_a_per_sqrt_hz(1e-3);
        let i4 = det.shot_noise_a_per_sqrt_hz(4e-3);
        // shot noise ∝ sqrt(P): 4× power → 2× noise (approximately, when
        // signal >> dark)
        let ratio = i4 / i1;
        assert!(
            (ratio - 2.0).abs() < 0.05,
            "ratio = {ratio:.4}, expected ≈ 2.0"
        );
    }

    #[test]
    fn test_nep_positive() {
        let det = ingaas();
        let nep = det.nep_w_per_sqrt_hz(300.0, 50.0);
        assert!(nep > 0.0, "NEP must be positive, got {nep}");
        assert!(nep.is_finite(), "NEP must be finite");
    }

    #[test]
    fn test_rise_time_formula() {
        let det = ingaas(); // BW = 10 GHz
        let tr = det.rise_time_ps();
        // t_r = 0.35 / 10e9 * 1e12 = 35 ps
        assert_relative_eq!(tr, 35.0, epsilon = 1e-6);
    }

    #[test]
    fn test_apd_gain_multiplies_current() {
        let apd = AvalanchePhotodiode::ingaas_apd();
        let r_eff = apd.effective_responsivity();
        assert_relative_eq!(
            r_eff,
            apd.pin.responsivity_a_per_w * apd.gain_m,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_apd_excess_noise() {
        let apd = AvalanchePhotodiode::ingaas_apd();
        let f_m = apd.excess_noise_factor();
        // F(M) = M^x = 10^0.7 ≈ 5.01 — always > 1 for M > 1
        assert!(f_m > 1.0, "F(M) = {f_m:.3} should exceed 1");
    }

    #[test]
    fn test_d_star_positive() {
        let det = ingaas();
        let d_star = det.d_star_cm_sqrt_hz_per_w(300.0, 50.0);
        assert!(d_star > 0.0, "D* must be positive");
        assert!(d_star.is_finite(), "D* must be finite");
    }
}
