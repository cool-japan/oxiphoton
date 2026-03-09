//! Avalanche Photodetector (APD) model.
//!
//! Implements McIntyre's model for excess noise, shot noise, and SNR
//! calculations for avalanche photodiodes.

// Physical constants
const Q_E: f64 = 1.602_176_634e-19; // Elementary charge (C)
const K_B: f64 = 1.380_649e-23; // Boltzmann constant (J/K)

/// Avalanche Photodetector (APD) model.
///
/// Combines primary responsivity, avalanche gain, and McIntyre noise model
/// to compute SNR, NEP, and optimal gain.
#[derive(Debug, Clone)]
pub struct AvalanchePhotodetector {
    /// Primary responsivity (A/W) before avalanche gain
    pub responsivity: f64,
    /// Ionization ratio k = α_h/α_e (0 for pure electron injection, 1 for equal)
    pub ionization_ratio: f64,
    /// Avalanche gain M
    pub gain: f64,
    /// Gain-bandwidth product (Hz) — bandwidth at unit gain
    pub bandwidth_hz: f64,
    /// Dark current (nA)
    pub dark_current_na: f64,
    /// Junction capacitance (fF)
    pub capacitance_ff: f64,
    /// Load resistance (Ω)
    pub load_resistance: f64,
}

impl AvalanchePhotodetector {
    /// Typical InGaAs APD at 1.55 μm.
    ///
    /// Parameters based on commercial InGaAs APDs (e.g., Hamamatsu G8931).
    pub fn ingaas_1550() -> Self {
        Self {
            responsivity: 0.85,     // A/W at unit gain
            ionization_ratio: 0.45, // InGaAs/InP APD typical k
            gain: 10.0,             // typical operating gain
            bandwidth_hz: 10e9,     // 10 GHz gain-bandwidth product
            dark_current_na: 10.0,  // 10 nA dark current
            capacitance_ff: 150.0,  // 150 fF junction capacitance
            load_resistance: 50.0,  // 50 Ω load
        }
    }

    /// Typical Si APD at 800 nm.
    ///
    /// Parameters based on commercial Si APDs (e.g., Hamamatsu S2381).
    pub fn si_800() -> Self {
        Self {
            responsivity: 0.5,      // A/W at unit gain (~80% QE at 800nm)
            ionization_ratio: 0.02, // Si APD: k very small (electron-initiated)
            gain: 100.0,            // Si APDs can have high gain
            bandwidth_hz: 50e9,     // larger GBP for Si
            dark_current_na: 0.5,   // lower dark current
            capacitance_ff: 80.0,
            load_resistance: 50.0,
        }
    }

    /// Total responsivity with avalanche gain: R_total = M · R_primary.
    pub fn effective_responsivity(&self) -> f64 {
        self.gain * self.responsivity
    }

    /// Excess noise factor F(M) by McIntyre's model.
    ///
    /// F(M) = k·M + (1 - k)·(2 - 1/M)
    /// This reduces to F = 2 - 1/M for k=0 (pure electron injection)
    /// and F = M for k=1 (equal ionization rates).
    pub fn excess_noise_factor(&self) -> f64 {
        let k = self.ionization_ratio;
        let m = self.gain;
        if m <= 0.0 {
            return 1.0;
        }
        k * m + (1.0 - k) * (2.0 - 1.0 / m)
    }

    /// Signal-to-noise ratio (dB) for given optical power and electrical bandwidth.
    ///
    /// SNR = (M·R·P)² / (i_shot² + i_thermal² + i_dark²)
    /// where the noise currents are spectral densities integrated over bandwidth B.
    pub fn snr_db(&self, power_w: f64, bandwidth_hz: f64) -> f64 {
        let t = 300.0; // room temperature
        let i_signal = self.effective_responsivity() * power_w;
        let i_ph = self.responsivity * power_w; // primary photocurrent

        // Shot noise (including excess noise): i_shot² = 2·q·M²·F·I_ph·B
        let f_excess = self.excess_noise_factor();
        let i_shot_sq = 2.0 * Q_E * self.gain * self.gain * f_excess * i_ph * bandwidth_hz;

        // Dark current shot noise: i_dark² = 2·q·M²·F·I_dark·B
        let i_dark_a = self.dark_current_na * 1e-9;
        let i_dark_sq = 2.0 * Q_E * self.gain * self.gain * f_excess * i_dark_a * bandwidth_hz;

        // Thermal (Johnson) noise: i_th² = 4·k·T·B/R
        let i_thermal_sq = 4.0 * K_B * t * bandwidth_hz / self.load_resistance;

        let noise_total = i_shot_sq + i_dark_sq + i_thermal_sq;
        if noise_total <= 0.0 {
            return f64::INFINITY;
        }
        10.0 * (i_signal * i_signal / noise_total).log10()
    }

    /// Noise-equivalent power NEP = noise_current_density / R_eff (W/√Hz).
    ///
    /// NEP = sqrt(2·q·M²·F·I_dark + 4kT/R) / (M·R_primary)
    pub fn nep_w_per_sqrthz(&self, bandwidth_hz: f64) -> f64 {
        let t = 300.0;
        let f_excess = self.excess_noise_factor();
        let i_dark_a = self.dark_current_na * 1e-9;

        // Noise current spectral density (A/√Hz)
        let shot_density = (2.0 * Q_E * self.gain * self.gain * f_excess * i_dark_a).sqrt();
        let thermal_density = (4.0 * K_B * t / self.load_resistance).sqrt();

        // Total noise density in dark condition
        let total_noise_density = (shot_density * shot_density
            + thermal_density * thermal_density
            + 2.0
                * Q_E
                * self.gain
                * self.gain
                * f_excess
                * self.responsivity
                * bandwidth_hz
                * 1e-3)
            .sqrt();
        // Approximate: just use dark + thermal noise density for NEP definition
        let noise_density =
            (shot_density * shot_density + thermal_density * thermal_density).sqrt();

        let r_eff = self.effective_responsivity();
        if r_eff <= 0.0 {
            return f64::INFINITY;
        }
        let _ = total_noise_density; // silence unused variable
        noise_density / r_eff
    }

    /// Shot noise current spectral density (A/√Hz).
    ///
    /// i_shot = sqrt(2·q·I·M²·F(M))
    pub fn shot_noise_density(&self, current_a: f64) -> f64 {
        let f_excess = self.excess_noise_factor();
        (2.0 * Q_E * current_a * self.gain * self.gain * f_excess).sqrt()
    }

    /// Thermal (Johnson) noise current spectral density (A/√Hz).
    ///
    /// i_thermal = sqrt(4·k·T/R_load)
    pub fn thermal_noise_density(&self, temperature_k: f64) -> f64 {
        (4.0 * K_B * temperature_k / self.load_resistance).sqrt()
    }

    /// Optimum gain M that minimizes total noise figure.
    ///
    /// Minimized by dNEP/dM = 0. For k << 1, M_opt ≈ (i_thermal / (q·R·P))^(1/3).
    /// Uses numerical search over a range.
    pub fn optimal_gain(&self, optical_power_w: f64, temperature_k: f64) -> f64 {
        let i_thermal_sq = 4.0 * K_B * temperature_k / self.load_resistance;
        let i_ph = self.responsivity * optical_power_w;
        let i_dark_a = self.dark_current_na * 1e-9;
        let k = self.ionization_ratio;

        // For k ≈ 0, optimal gain: M_opt ≈ (i_thermal_sq / (2·q·(2-k)·(I_ph+I_dark)))^{1/3}
        let base_noise_current = 2.0 * Q_E * (2.0 - k) * (i_ph + i_dark_a);
        if base_noise_current <= 0.0 {
            return 1.0;
        }
        let m_opt = (i_thermal_sq / base_noise_current).powf(1.0 / 3.0);
        m_opt.clamp(1.0, 1000.0)
    }

    /// 3 dB bandwidth at a given gain level.
    ///
    /// For gain-bandwidth limited APDs: BW(M) = GBP / M.
    pub fn bandwidth_at_gain(&self, gain: f64) -> f64 {
        if gain <= 0.0 {
            return self.bandwidth_hz;
        }
        self.bandwidth_hz / gain
    }

    /// Photocurrent output for incident optical power (A).
    ///
    /// I_ph = M · R_primary · P
    pub fn photocurrent(&self, power_w: f64) -> f64 {
        self.effective_responsivity() * power_w
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ingaas_responsivity_reasonable() {
        let apd = AvalanchePhotodetector::ingaas_1550();
        // With gain=10, effective R = 8.5 A/W
        assert_relative_eq!(apd.effective_responsivity(), 8.5, epsilon = 0.1);
    }

    #[test]
    fn si_apd_high_gain() {
        let apd = AvalanchePhotodetector::si_800();
        assert!(apd.gain > 50.0);
    }

    #[test]
    fn excess_noise_factor_bounded_below() {
        let apd = AvalanchePhotodetector::ingaas_1550();
        let f = apd.excess_noise_factor();
        // F(M) ≥ 1 always
        assert!(f >= 1.0, "F = {f}");
    }

    #[test]
    fn excess_noise_factor_k0_approaches_two() {
        // For k=0, M large: F → 2
        let mut apd = AvalanchePhotodetector::si_800();
        apd.gain = 100.0;
        apd.ionization_ratio = 0.0;
        let f = apd.excess_noise_factor();
        // F = 2 - 1/M → 2 for large M
        assert!((f - 2.0).abs() < 0.02, "F = {f}");
    }

    #[test]
    fn excess_noise_factor_k1_equals_m() {
        // For k=1, F(M) = M
        let mut apd = AvalanchePhotodetector::ingaas_1550();
        apd.ionization_ratio = 1.0;
        apd.gain = 10.0;
        assert_relative_eq!(apd.excess_noise_factor(), apd.gain, epsilon = 1e-10);
    }

    #[test]
    fn snr_increases_with_power() {
        let apd = AvalanchePhotodetector::ingaas_1550();
        let snr_low = apd.snr_db(1e-6, 1e9); // 1 μW
        let snr_high = apd.snr_db(1e-3, 1e9); // 1 mW
        assert!(
            snr_high > snr_low,
            "SNR(low)={snr_low:.1}, SNR(high)={snr_high:.1}"
        );
    }

    #[test]
    fn nep_positive() {
        let apd = AvalanchePhotodetector::ingaas_1550();
        let nep = apd.nep_w_per_sqrthz(1e9);
        assert!(nep > 0.0 && nep.is_finite(), "NEP = {nep}");
    }

    #[test]
    fn shot_noise_scales_sqrt_current() {
        let apd = AvalanchePhotodetector::ingaas_1550();
        let s1 = apd.shot_noise_density(1e-3);
        let s4 = apd.shot_noise_density(4e-3);
        // shot noise ∝ sqrt(I) so s4/s1 = 2
        assert_relative_eq!(s4 / s1, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn thermal_noise_decreases_with_lower_temperature() {
        let apd = AvalanchePhotodetector::ingaas_1550();
        let th_room = apd.thermal_noise_density(300.0);
        let th_cold = apd.thermal_noise_density(77.0);
        assert!(th_cold < th_room);
    }

    #[test]
    fn optimal_gain_in_reasonable_range() {
        let apd = AvalanchePhotodetector::ingaas_1550();
        let m_opt = apd.optimal_gain(1e-6, 300.0);
        assert!((1.0..=1000.0).contains(&m_opt), "M_opt = {m_opt}");
    }

    #[test]
    fn bandwidth_decreases_with_gain() {
        let apd = AvalanchePhotodetector::ingaas_1550();
        let bw10 = apd.bandwidth_at_gain(10.0);
        let bw100 = apd.bandwidth_at_gain(100.0);
        assert!(bw10 > bw100, "BW(10)={bw10:.1e}, BW(100)={bw100:.1e}");
    }

    #[test]
    fn photocurrent_proportional_to_power() {
        let apd = AvalanchePhotodetector::ingaas_1550();
        let i1 = apd.photocurrent(1e-3);
        let i2 = apd.photocurrent(2e-3);
        assert_relative_eq!(i2, 2.0 * i1, epsilon = 1e-10);
    }
}
