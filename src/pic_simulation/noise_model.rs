/// Noise models for photonic integrated circuits.
///
/// Covers shot noise, laser relative intensity noise (RIN),
/// polarisation-dependent loss (PDL), and thermal (Johnson-Nyquist) noise.
use std::f64::consts::PI;

/// Planck constant (J·s).
pub const H_PLANCK: f64 = 6.62607015e-34;
/// Speed of light in vacuum (m/s).
pub const C_LIGHT: f64 = 2.99792458e8;
/// Boltzmann constant (J/K).
pub const KB: f64 = 1.380649e-23;
/// Elementary charge (C).
const Q_ELECTRON: f64 = 1.602176634e-19;

// ---------------------------------------------------------------------------
// Shot Noise
// ---------------------------------------------------------------------------

/// Shot-noise model for a photodetector.
///
/// Models quantum-limited optical detection including dark current.
/// All noise quantities are RMS values at the specified electrical bandwidth.
#[derive(Clone, Debug)]
pub struct ShotNoise {
    /// Detector responsivity R (A/W).
    pub responsivity_a_per_w: f64,
    /// Dark current I_d (A).
    pub dark_current_a: f64,
    /// Electrical detection bandwidth BW (Hz).
    pub bandwidth_hz: f64,
}

impl ShotNoise {
    /// Create a shot noise model for a typical silicon photodetector.
    pub fn typical_si() -> Self {
        Self {
            responsivity_a_per_w: 0.8,
            dark_current_a: 1e-9,
            bandwidth_hz: 10e9,
        }
    }

    /// RMS shot noise current: i_shot = √(2q (I_ph + I_d) BW).
    pub fn noise_current_rms_a(&self, optical_power_w: f64) -> f64 {
        let i_ph = self.responsivity_a_per_w * optical_power_w;
        (2.0 * Q_ELECTRON * (i_ph + self.dark_current_a) * self.bandwidth_hz).sqrt()
    }

    /// Signal-to-noise ratio (power SNR) in shot-noise-limited regime.
    pub fn snr_shot_limited(&self, signal_power_w: f64) -> f64 {
        let i_signal = self.responsivity_a_per_w * signal_power_w;
        let i_noise = self.noise_current_rms_a(signal_power_w);
        if i_noise <= 0.0 {
            return f64::INFINITY;
        }
        (i_signal / i_noise).powi(2)
    }

    /// Noise-equivalent power: NEP = √(2q I_d BW) / R.
    pub fn noise_equivalent_power_w(&self) -> f64 {
        let i_dark_noise = (2.0 * Q_ELECTRON * self.dark_current_a * self.bandwidth_hz).sqrt();
        if self.responsivity_a_per_w <= 0.0 {
            return f64::INFINITY;
        }
        i_dark_noise / self.responsivity_a_per_w
    }

    /// Minimum detectable power for a given required SNR.
    ///
    /// Derived from: SNR = R²P² / (2q R P BW) → P_min = 2q SNR BW / R
    pub fn minimum_detectable_power_w(&self, snr_required: f64) -> f64 {
        if self.responsivity_a_per_w <= 0.0 {
            return f64::INFINITY;
        }
        2.0 * Q_ELECTRON * snr_required * self.bandwidth_hz / self.responsivity_a_per_w
    }

    /// SNR in dB.
    pub fn snr_db(&self, signal_power_w: f64) -> f64 {
        10.0 * self.snr_shot_limited(signal_power_w).log10()
    }

    /// Photon flux for a given optical power and wavelength.
    pub fn photon_flux(&self, power_w: f64, wavelength_m: f64) -> f64 {
        let photon_energy = H_PLANCK * C_LIGHT / wavelength_m;
        power_w / photon_energy
    }
}

// ---------------------------------------------------------------------------
// Relative Intensity Noise (RIN)
// ---------------------------------------------------------------------------

/// Relative intensity noise model for a laser source.
///
/// RIN characterises the fractional power fluctuations of the laser,
/// expressed in dBc/Hz at a given offset frequency.
#[derive(Clone, Debug)]
pub struct RinNoise {
    /// RIN spectral density in dBc/Hz (e.g., −150 dBc/Hz for a DFB laser).
    pub rin_db_per_hz: f64,
    /// Electrical bandwidth over which RIN is integrated (Hz).
    pub bandwidth_hz: f64,
    /// Mean optical power at the detector (W).
    pub power_w: f64,
}

impl RinNoise {
    /// Create a model for a typical DFB laser.
    pub fn typical_dfb(power_w: f64, bandwidth_hz: f64) -> Self {
        Self {
            rin_db_per_hz: -150.0,
            bandwidth_hz,
            power_w,
        }
    }

    /// RIN spectral density in linear units (Hz^{-1}).
    pub fn rin_linear(&self) -> f64 {
        10.0_f64.powf(self.rin_db_per_hz / 10.0)
    }

    /// RMS optical power noise: σ_P = P × √(RIN × BW).
    pub fn power_noise_rms_w(&self) -> f64 {
        self.power_w * (self.rin_linear() * self.bandwidth_hz).sqrt()
    }

    /// Signal-to-noise ratio limited by RIN for a given signal fraction.
    ///
    /// `signal_power_fraction` is the modulation depth (0–1).
    pub fn snr_rin_limited(&self, signal_power_fraction: f64) -> f64 {
        let p_noise = self.power_noise_rms_w();
        if p_noise <= 0.0 {
            return f64::INFINITY;
        }
        let p_signal = self.power_w * signal_power_fraction;
        (p_signal / p_noise).powi(2)
    }

    /// Dynamic range limited by RIN (dB).
    pub fn dynamic_range_db(&self) -> f64 {
        10.0 * (1.0 / (self.rin_linear() * self.bandwidth_hz)).log10()
    }
}

// ---------------------------------------------------------------------------
// Polarisation-Dependent Loss (PDL)
// ---------------------------------------------------------------------------

/// Polarisation-dependent loss model.
///
/// PDL = 10 log₁₀(T_max / T_min), where T_max and T_min are the maximum
/// and minimum power transmissions over all input polarisation states.
#[derive(Clone, Debug)]
pub struct PolarizationDependentLoss {
    /// PDL magnitude (dB), always ≥ 0.
    pub pdl_db: f64,
}

impl PolarizationDependentLoss {
    /// Construct from a PDL value in dB.
    pub fn new(pdl_db: f64) -> Self {
        Self {
            pdl_db: pdl_db.abs(),
        }
    }

    /// Ratio T_max / T_min (linear).
    pub fn transmission_ratio(&self) -> f64 {
        10.0_f64.powf(self.pdl_db / 10.0)
    }

    /// Worst-case additional insertion loss from PDL: ΔIL = PDL / 2 (dB).
    pub fn worst_case_insertion_loss_db(&self) -> f64 {
        self.pdl_db / 2.0
    }

    /// SNR penalty in a coherent receiver due to PDL (dB).
    pub fn snr_penalty_db(&self) -> f64 {
        0.5 * self.pdl_db
    }

    /// Jones matrix amplitude eigenvalues (√T_max, √T_min).
    pub fn jones_eigenvalues(&self) -> (f64, f64) {
        let t_max = self.transmission_ratio().sqrt();
        let t_min = 1.0 / t_max;
        (t_max, t_min)
    }

    /// Average insertion loss including PDL (dB).
    pub fn average_insertion_loss_db(&self) -> f64 {
        // Average T = (T_max + T_min) / 2 → IL_avg in dB
        let ratio = self.transmission_ratio();
        let t_avg = (ratio + 1.0 / ratio) / 2.0;
        if t_avg <= 0.0 {
            return f64::INFINITY;
        }
        -10.0 * t_avg.log10()
    }

    /// Degree of polarisation-dependent loss (normalised 0–1).
    pub fn normalised_pdl(&self) -> f64 {
        let r = self.transmission_ratio();
        (r - 1.0) / (r + 1.0)
    }
}

// ---------------------------------------------------------------------------
// Thermal Noise
// ---------------------------------------------------------------------------

/// Johnson-Nyquist (thermal / resistor) noise model.
///
/// Applicable to transimpedance amplifiers, electrical circuits following
/// the photodetector, and electrical inter-connects in PICs.
#[derive(Clone, Debug)]
pub struct ThermalNoise {
    /// Physical temperature T (K).
    pub temperature_k: f64,
    /// Load / source resistance R (Ω).
    pub resistance_ohm: f64,
    /// Electrical bandwidth BW (Hz).
    pub bandwidth_hz: f64,
}

impl ThermalNoise {
    /// Create model at room temperature (300 K).
    pub fn room_temperature(resistance_ohm: f64, bandwidth_hz: f64) -> Self {
        Self {
            temperature_k: 300.0,
            resistance_ohm,
            bandwidth_hz,
        }
    }

    /// RMS thermal voltage noise: V_rms = √(4 k_B T R BW).
    pub fn voltage_noise_rms_v(&self) -> f64 {
        (4.0 * KB * self.temperature_k * self.resistance_ohm * self.bandwidth_hz).sqrt()
    }

    /// RMS thermal current noise: I_rms = V_rms / R.
    pub fn current_noise_rms_a(&self) -> f64 {
        if self.resistance_ohm <= 0.0 {
            return 0.0;
        }
        self.voltage_noise_rms_v() / self.resistance_ohm
    }

    /// Noise power spectral density S_V = 4 k_B T R (V²/Hz).
    pub fn voltage_psd_v2_per_hz(&self) -> f64 {
        4.0 * KB * self.temperature_k * self.resistance_ohm
    }

    /// Noise power delivered to a matched load: P_n = k_B T BW (W).
    pub fn available_noise_power_w(&self) -> f64 {
        KB * self.temperature_k * self.bandwidth_hz
    }

    /// Noise figure contribution of this resistor to a receiver chain (dB).
    pub fn noise_figure_db(&self, reference_temperature_k: f64) -> f64 {
        let t_ratio = self.temperature_k / reference_temperature_k;
        10.0 * (1.0 + t_ratio).log10()
    }
}

// ---------------------------------------------------------------------------
// Optical SNR (OSNR)
// ---------------------------------------------------------------------------

/// Optical signal-to-noise ratio utility functions.
pub struct OsnrModel;

impl OsnrModel {
    /// Convert OSNR (dB, in 0.1 nm reference bandwidth) to electrical SNR.
    ///
    /// SNR_el = OSNR / (2 × BW_optical / BW_electrical)
    pub fn osnr_to_electrical_snr(
        osnr_db: f64,
        bandwidth_optical_hz: f64,
        bandwidth_electrical_hz: f64,
    ) -> f64 {
        let osnr_linear = 10.0_f64.powf(osnr_db / 10.0);
        osnr_linear * bandwidth_electrical_hz / (2.0 * bandwidth_optical_hz)
    }

    /// Required OSNR for a given BER using the Gaussian approximation.
    pub fn required_osnr_bpsk_db(ber_target: f64, bandwidth_ratio: f64) -> f64 {
        // Q² = OSNR / bandwidth_ratio for BPSK; BER = erfc(sqrt(Q²/2))/2
        // Approximate inverse erfc via simple iteration (no external crate)
        let q2 = Self::q_squared_from_ber(ber_target);
        10.0 * (q2 * bandwidth_ratio).log10()
    }

    /// Q² factor from BER using erfc⁻¹ approximation.
    fn q_squared_from_ber(ber: f64) -> f64 {
        // BER ≈ 0.5 erfc(Q/√2) → Q ≈ √(-2 ln(2 BER)) for BER ≪ 0.5
        let ber_clamped = ber.clamp(1e-20, 0.5);
        let q = (-2.0 * (2.0 * ber_clamped).ln()).sqrt();
        q * q
    }

    /// Photon number per bit for a given OSNR and bit rate.
    pub fn photons_per_bit(osnr_linear: f64, symbol_rate_hz: f64, noise_bandwidth_hz: f64) -> f64 {
        osnr_linear * noise_bandwidth_hz / symbol_rate_hz
    }
}

// ---------------------------------------------------------------------------
// Phase Noise
// ---------------------------------------------------------------------------

/// Laser phase noise / linewidth model.
///
/// A Lorentzian lineshape with FWHM = Δν (Hz).
#[derive(Clone, Debug)]
pub struct PhaseNoise {
    /// Laser linewidth Δν (Hz).
    pub linewidth_hz: f64,
    /// Integration time or symbol duration (s).
    pub integration_time_s: f64,
}

impl PhaseNoise {
    /// RMS phase noise accumulated over the integration time.
    ///
    /// σ_φ = √(2π Δν τ) for a Lorentzian lineshape.
    pub fn rms_phase_noise_rad(&self) -> f64 {
        (2.0 * PI * self.linewidth_hz * self.integration_time_s).sqrt()
    }

    /// Phase noise variance (rad²).
    pub fn phase_variance_rad2(&self) -> f64 {
        2.0 * PI * self.linewidth_hz * self.integration_time_s
    }

    /// Coherence time τ_c = 1 / (π Δν).
    pub fn coherence_time_s(&self) -> f64 {
        1.0 / (PI * self.linewidth_hz)
    }

    /// Coherence length L_c = c / (π Δν) in metres.
    pub fn coherence_length_m(&self) -> f64 {
        C_LIGHT / (PI * self.linewidth_hz)
    }

    /// SNR penalty due to phase noise in a DPSK system (dB).
    pub fn dpsk_snr_penalty_db(&self) -> f64 {
        let sigma2 = self.phase_variance_rad2();
        // Penalty ≈ 10 log10(1 + π² σ_φ²/ 3) for small σ_φ
        10.0 * (1.0 + std::f64::consts::PI * std::f64::consts::PI * sigma2 / 3.0).log10()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shot_noise_increases_with_power() {
        let sn = ShotNoise::typical_si();
        let n1 = sn.noise_current_rms_a(1e-3);
        let n2 = sn.noise_current_rms_a(10e-3);
        assert!(n2 > n1, "noise should grow with power");
    }

    #[test]
    fn shot_noise_nep_positive() {
        let sn = ShotNoise::typical_si();
        let nep = sn.noise_equivalent_power_w();
        assert!(nep > 0.0 && nep < 1e-6, "NEP={}", nep);
    }

    #[test]
    fn rin_noise_dfb_typical_range() {
        let rin = RinNoise::typical_dfb(1e-3, 10e9);
        let noise = rin.power_noise_rms_w();
        // For RIN = -150 dBc/Hz and BW = 10 GHz: σ ≈ 1e-3 × √(1e-15 × 1e10) = 31.6 nW
        assert!(noise > 0.0 && noise < 1e-3, "RIN noise={:.3e}", noise);
    }

    #[test]
    fn pdl_eigenvalues_product_unity() {
        let pdl = PolarizationDependentLoss::new(3.0);
        let (t_max, t_min) = pdl.jones_eigenvalues();
        assert!(
            (t_max * t_min - 1.0).abs() < 1e-10,
            "product={}",
            t_max * t_min
        );
    }

    #[test]
    fn pdl_worst_case_il() {
        let pdl = PolarizationDependentLoss::new(2.0);
        assert!((pdl.worst_case_insertion_loss_db() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn thermal_noise_room_temperature() {
        let tn = ThermalNoise::room_temperature(50.0, 1e9);
        let v = tn.voltage_noise_rms_v();
        // Expected: √(4 × 1.38e-23 × 300 × 50 × 1e9) ≈ 28.8 μV
        assert!(v > 1e-6 && v < 1e-3, "V_rms={:.3e}", v);
    }

    #[test]
    fn thermal_noise_current_ohms_law() {
        let tn = ThermalNoise::room_temperature(1000.0, 1e6);
        let v = tn.voltage_noise_rms_v();
        let i = tn.current_noise_rms_a();
        assert!((v / 1000.0 - i).abs() < 1e-20, "Ohm's law violation");
    }

    #[test]
    fn phase_noise_coherence_time() {
        let pn = PhaseNoise {
            linewidth_hz: 100e3,
            integration_time_s: 1e-9,
        };
        let tc = pn.coherence_time_s();
        // τ_c = 1/(π × 100e3) ≈ 3.18 μs
        assert!((tc - 1.0 / (PI * 100e3)).abs() < 1e-12, "τ_c={:.3e}", tc);
    }

    #[test]
    fn shot_noise_min_detectable_power_consistent() {
        let sn = ShotNoise::typical_si();
        let snr_req = 10.0; // 10:1 power SNR
        let p_min = sn.minimum_detectable_power_w(snr_req);
        let snr_actual = sn.snr_shot_limited(p_min);
        // Should be approximately equal (within 50% due to dark current contribution)
        assert!(
            (snr_actual / snr_req - 1.0).abs() < 0.5,
            "SNR={}",
            snr_actual
        );
    }
}
