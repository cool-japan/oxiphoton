/// Photodiode model for optical receivers.
///
/// Models the key figures of merit for a photodetector:
/// - Responsivity R (A/W): current generated per optical power
/// - Quantum efficiency η: fraction of photons converted to electron-hole pairs
/// - Bandwidth BW (Hz): 3dB electrical bandwidth
/// - Dark current I_dark (A): current without illumination
/// - Noise-equivalent power (NEP) (W/√Hz)
///
/// Photodiode device model.
#[derive(Debug, Clone, Copy)]
pub struct Photodiode {
    /// Responsivity at operating wavelength (A/W)
    pub responsivity: f64,
    /// Dark current (A)
    pub dark_current: f64,
    /// 3dB bandwidth (Hz)
    pub bandwidth_hz: f64,
    /// Operating wavelength (m)
    pub wavelength: f64,
    /// Junction capacitance (F)
    pub capacitance: f64,
    /// Load resistance (Ω)
    pub load_resistance: f64,
}

impl Photodiode {
    /// Create a photodiode with given parameters.
    pub fn new(responsivity: f64, dark_current: f64, bandwidth_hz: f64, wavelength: f64) -> Self {
        Self {
            responsivity,
            dark_current,
            bandwidth_hz,
            wavelength,
            capacitance: 0.0,
            load_resistance: 50.0,
        }
    }

    /// InGaAs PIN photodiode for 1550nm (typical high-speed datacom)
    pub fn ingaas_pin_1550() -> Self {
        Self {
            responsivity: 0.95,   // A/W (≈95% at 1550nm)
            dark_current: 0.5e-9, // 0.5 nA
            bandwidth_hz: 10e9,   // 10 GHz
            wavelength: 1550e-9,
            capacitance: 0.1e-12,  // 100 fF
            load_resistance: 50.0, // Ω
        }
    }

    /// Silicon PIN photodiode for 850nm (datacom OM3/OM4 fiber)
    pub fn si_pin_850() -> Self {
        Self {
            responsivity: 0.6,  // A/W
            dark_current: 1e-9, // 1 nA
            bandwidth_hz: 1e9,  // 1 GHz
            wavelength: 850e-9,
            capacitance: 0.5e-12,
            load_resistance: 50.0,
        }
    }

    /// External quantum efficiency: η = R·hf/e = R·hc/(e·λ)
    pub fn quantum_efficiency(&self) -> f64 {
        let h = 6.626e-34;
        let c = 2.998e8;
        let e = 1.602e-19;
        self.responsivity * h * c / (e * self.wavelength)
    }

    /// Photocurrent for incident power P (W): I_ph = R·P
    pub fn photocurrent(&self, power: f64) -> f64 {
        self.responsivity * power
    }

    /// Total current: I = I_ph + I_dark
    pub fn total_current(&self, power: f64) -> f64 {
        self.photocurrent(power) + self.dark_current
    }

    /// Shot noise current spectral density (A/√Hz):
    ///   i_shot = sqrt(2·e·I_total·B) / sqrt(B) = sqrt(2·e·I_total)
    pub fn shot_noise_density(&self, power: f64) -> f64 {
        let e = 1.602e-19;
        let i_total = self.total_current(power);
        (2.0 * e * i_total).sqrt()
    }

    /// Thermal (Johnson) noise density from load resistance (A/√Hz):
    ///   i_thermal = sqrt(4·k_B·T / R_L)
    pub fn thermal_noise_density(&self, temperature_k: f64) -> f64 {
        let k_b = 1.381e-23;
        (4.0 * k_b * temperature_k / self.load_resistance).sqrt()
    }

    /// Total noise current density (A/√Hz):
    ///   i_n = sqrt(i_shot² + i_thermal²)
    pub fn total_noise_density(&self, power: f64, temperature_k: f64) -> f64 {
        let i_s = self.shot_noise_density(power);
        let i_t = self.thermal_noise_density(temperature_k);
        (i_s * i_s + i_t * i_t).sqrt()
    }

    /// Signal-to-noise ratio (electrical) at given optical power and bandwidth:
    ///   SNR = (R·P)² / (noise² · B)
    pub fn snr(&self, power: f64, bandwidth_hz: f64, temperature_k: f64) -> f64 {
        let i_ph = self.photocurrent(power);
        let i_n = self.total_noise_density(power, temperature_k);
        (i_ph * i_ph) / (i_n * i_n * bandwidth_hz)
    }

    /// Noise-equivalent power (NEP) [W/√Hz]:
    ///   NEP = i_n / R  (minimum detectable signal per √Hz)
    pub fn nep(&self, temperature_k: f64) -> f64 {
        // At minimum signal (dark current dominated):
        let i_n = self.total_noise_density(0.0, temperature_k);
        i_n / self.responsivity
    }

    /// Minimum detectable power for SNR = 1 over bandwidth B:
    ///   P_min = NEP · sqrt(B)
    pub fn sensitivity(&self, bandwidth_hz: f64, temperature_k: f64) -> f64 {
        self.nep(temperature_k) * bandwidth_hz.sqrt()
    }

    /// RC-limited bandwidth from capacitance and load resistance:
    ///   f_3dB = 1/(2π·R·C)
    pub fn rc_bandwidth(&self) -> f64 {
        if self.capacitance < 1e-30 {
            f64::INFINITY
        } else {
            1.0 / (2.0 * std::f64::consts::PI * self.load_resistance * self.capacitance)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingaas_qe_near_unity() {
        let pd = Photodiode::ingaas_pin_1550();
        let qe = pd.quantum_efficiency();
        // InGaAs at 1550nm: R=0.95A/W → QE = R·hc/(eλ) ≈ 0.76 (photon energy 0.8eV)
        assert!(
            qe > 0.7 && qe <= 1.0,
            "QE={qe:.3} out of expected range [0.7, 1.0]"
        );
    }

    #[test]
    fn photocurrent_proportional_to_power() {
        let pd = Photodiode::ingaas_pin_1550();
        let i1 = pd.photocurrent(1e-3);
        let i2 = pd.photocurrent(2e-3);
        assert!((i2 / i1 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn shot_noise_increases_with_power() {
        let pd = Photodiode::ingaas_pin_1550();
        let n1 = pd.shot_noise_density(0.1e-3);
        let n2 = pd.shot_noise_density(1e-3);
        assert!(n2 > n1);
    }

    #[test]
    fn snr_increases_with_power() {
        let pd = Photodiode::ingaas_pin_1550();
        let snr1 = pd.snr(0.1e-3, 1e9, 300.0);
        let snr2 = pd.snr(1e-3, 1e9, 300.0);
        assert!(snr2 > snr1, "Higher power → better SNR");
    }

    #[test]
    fn nep_positive() {
        let pd = Photodiode::ingaas_pin_1550();
        let nep = pd.nep(300.0);
        assert!(nep > 0.0 && nep.is_finite());
    }

    #[test]
    fn sensitivity_increases_with_bandwidth() {
        let pd = Photodiode::ingaas_pin_1550();
        let s1 = pd.sensitivity(1e9, 300.0);
        let s2 = pd.sensitivity(10e9, 300.0);
        assert!(s2 > s1, "Wider bandwidth → higher minimum detectable power");
    }

    #[test]
    fn rc_bandwidth_finite_with_capacitance() {
        let pd = Photodiode::ingaas_pin_1550();
        let bw = pd.rc_bandwidth();
        assert!(bw > 0.0 && bw.is_finite());
        // 100fF, 50Ω → f = 1/(2π·50·100e-15) ≈ 31.8 GHz
        assert!(
            bw > 10e9,
            "RC bandwidth should be > 10GHz for small capacitance"
        );
    }

    #[test]
    fn dark_current_nonzero() {
        let pd = Photodiode::ingaas_pin_1550();
        assert!(pd.dark_current > 0.0);
    }

    #[test]
    fn total_current_includes_dark() {
        let pd = Photodiode::ingaas_pin_1550();
        let i_total = pd.total_current(0.0);
        assert!((i_total - pd.dark_current).abs() < 1e-20);
    }
}
