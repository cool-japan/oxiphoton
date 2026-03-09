//! Photodetector responsivity and bandwidth models.
//!
//! Responsivity R (A/W) and quantum efficiency QE relate as:
//!   R = QE · e / (hν) = QE · e·λ / (h·c)
//!
//! Bandwidth is limited by:
//!   - RC time constant: f_RC = 1/(2π·R·C)
//!   - Transit time: f_tr = 0.44/τ_tr  where τ_tr = W/v_sat
//!   - Total 3dB BW: 1/f_3dB² = 1/f_RC² + 1/f_tr²
//!
//! Noise current:
//!   Shot noise: i²_shot = 2·e·I_dc·B
//!   Thermal noise: i²_th = 4·k_B·T·B/R_L
//!   NEP = √(i_noise²)/R  (W/√Hz)

use std::f64::consts::PI;

const PLANCK: f64 = 6.626e-34; // J·s
const E_CHARGE: f64 = 1.602e-19; // C
const K_BOLTZMANN: f64 = 1.381e-23; // J/K
const SPEED_OF_LIGHT_M: f64 = 2.998e8; // m/s

/// Spectral responsivity model.
#[derive(Debug, Clone)]
pub struct SpectralResponsivity {
    /// Material bandgap wavelength λ_g (m) — cutoff
    pub lambda_cutoff: f64,
    /// Peak responsivity R_peak (A/W)
    pub r_peak: f64,
    /// Peak wavelength (m)
    pub lambda_peak: f64,
    /// Spectral width (m, Gaussian σ)
    pub bandwidth_nm: f64,
}

impl SpectralResponsivity {
    /// InGaAs PIN photodiode (1000–1700 nm).
    pub fn ingaas_pin() -> Self {
        Self {
            lambda_cutoff: 1700e-9,
            r_peak: 0.95,
            lambda_peak: 1550e-9,
            bandwidth_nm: 300e-9,
        }
    }

    /// Si PIN photodiode (400–1100 nm).
    pub fn si_pin() -> Self {
        Self {
            lambda_cutoff: 1100e-9,
            r_peak: 0.6,
            lambda_peak: 850e-9,
            bandwidth_nm: 200e-9,
        }
    }

    /// Ge photodiode (800–1550 nm).
    pub fn ge_pin() -> Self {
        Self {
            lambda_cutoff: 1600e-9,
            r_peak: 0.85,
            lambda_peak: 1300e-9,
            bandwidth_nm: 300e-9,
        }
    }

    /// Responsivity at wavelength λ (m) in A/W.
    pub fn responsivity_at(&self, wavelength: f64) -> f64 {
        if wavelength > self.lambda_cutoff {
            return 0.0;
        }
        let dl = wavelength - self.lambda_peak;
        let sigma = self.bandwidth_nm / 2.0;
        let r = self.r_peak * (-dl * dl / (2.0 * sigma * sigma)).exp();
        r.max(0.0)
    }

    /// Quantum efficiency QE at wavelength λ.
    pub fn quantum_efficiency_at(&self, wavelength: f64) -> f64 {
        let r = self.responsivity_at(wavelength);
        // QE = R × hν / e = R × hc / (e × λ)
        r * PLANCK * SPEED_OF_LIGHT_M / (E_CHARGE * wavelength)
    }

    /// Maximum QE (at peak wavelength).
    pub fn peak_qe(&self) -> f64 {
        self.quantum_efficiency_at(self.lambda_peak)
    }
}

/// Photodetector bandwidth model.
#[derive(Debug, Clone, Copy)]
pub struct DetectorBandwidth {
    /// Load resistance Ω
    pub r_load: f64,
    /// Junction capacitance F
    pub c_junction: f64,
    /// Depletion width W (m)
    pub depletion_width: f64,
    /// Carrier saturation velocity m/s
    pub v_sat: f64,
    /// Parasitic inductance H (for package effects)
    pub l_parasitic: f64,
}

impl DetectorBandwidth {
    /// Create bandwidth model.
    pub fn new(r_load: f64, c_junction: f64, depletion_width: f64, v_sat: f64) -> Self {
        Self {
            r_load,
            c_junction,
            depletion_width,
            v_sat,
            l_parasitic: 0.0,
        }
    }

    /// InGaAs PIN at 25 GHz (50Ω, 30fF, W=2µm).
    pub fn ingaas_25ghz() -> Self {
        Self::new(50.0, 30e-15, 2e-6, 1e5)
    }

    /// High-speed InGaAs APD (100 GHz class).
    pub fn ingaas_apd_100ghz() -> Self {
        Self::new(50.0, 10e-15, 0.5e-6, 1.2e5)
    }

    /// RC-limited bandwidth: f_RC = 1/(2π·R·C).
    pub fn f_rc_hz(&self) -> f64 {
        1.0 / (2.0 * PI * self.r_load * self.c_junction)
    }

    /// Transit-time-limited bandwidth: f_tr = 0.44/τ_tr = 0.44·v_sat/W.
    pub fn f_transit_hz(&self) -> f64 {
        0.44 * self.v_sat / self.depletion_width
    }

    /// 3 dB bandwidth (combined RC and transit): 1/f_3dB² = 1/f_RC² + 1/f_tr².
    pub fn f_3db_hz(&self) -> f64 {
        let f_rc = self.f_rc_hz();
        let f_tr = self.f_transit_hz();
        1.0 / (1.0 / (f_rc * f_rc) + 1.0 / (f_tr * f_tr)).sqrt()
    }

    /// 3 dB bandwidth in GHz.
    pub fn f_3db_ghz(&self) -> f64 {
        self.f_3db_hz() / 1e9
    }

    /// Gain-bandwidth product (for APDs): GBP = M × f_3dB.
    pub fn gain_bandwidth_product(&self, avalanche_gain: f64) -> f64 {
        avalanche_gain * self.f_3db_hz()
    }
}

/// Noise equivalent power (NEP) model.
#[derive(Debug, Clone, Copy)]
pub struct DetectorNoise {
    /// Responsivity R (A/W)
    pub responsivity: f64,
    /// DC photocurrent I_dc (A)
    pub i_dc: f64,
    /// Load resistance R_L (Ω)
    pub r_load: f64,
    /// Temperature T (K)
    pub temperature: f64,
    /// Dark current I_dark (A)
    pub i_dark: f64,
}

impl DetectorNoise {
    /// Create noise model.
    pub fn new(responsivity: f64, i_dc: f64, r_load: f64) -> Self {
        Self {
            responsivity,
            i_dc,
            r_load,
            temperature: 300.0,
            i_dark: 1e-9,
        }
    }

    /// Shot noise current spectral density (A/√Hz).
    pub fn shot_noise_psd(&self) -> f64 {
        (2.0 * E_CHARGE * (self.i_dc + self.i_dark)).sqrt()
    }

    /// Thermal (Johnson) noise current spectral density (A/√Hz).
    pub fn thermal_noise_psd(&self) -> f64 {
        (4.0 * K_BOLTZMANN * self.temperature / self.r_load).sqrt()
    }

    /// Total noise current spectral density (A/√Hz).
    pub fn total_noise_psd(&self) -> f64 {
        let s = self.shot_noise_psd();
        let t = self.thermal_noise_psd();
        (s * s + t * t).sqrt()
    }

    /// Noise equivalent power NEP (W/√Hz).
    pub fn nep(&self) -> f64 {
        if self.responsivity < 1e-30 {
            return f64::INFINITY;
        }
        self.total_noise_psd() / self.responsivity
    }

    /// Detectivity D* (Jones = cm·√Hz/W, area-normalized NEP).
    pub fn detectivity_jones(&self, area_cm2: f64) -> f64 {
        let nep = self.nep();
        if nep < 1e-30 {
            return f64::INFINITY;
        }
        area_cm2.sqrt() / nep
    }

    /// Minimum detectable power in bandwidth B (Hz) at SNR=1.
    pub fn min_detectable_power(&self, bandwidth_hz: f64) -> f64 {
        self.nep() * bandwidth_hz.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responsivity_at_peak_is_r_peak() {
        let r = SpectralResponsivity::ingaas_pin();
        let r_at_peak = r.responsivity_at(r.lambda_peak);
        assert!((r_at_peak - r.r_peak).abs() < 1e-6);
    }

    #[test]
    fn responsivity_zero_beyond_cutoff() {
        let r = SpectralResponsivity::ingaas_pin();
        assert!(r.responsivity_at(2000e-9) == 0.0);
    }

    #[test]
    fn quantum_efficiency_le_1() {
        let r = SpectralResponsivity::si_pin();
        let qe = r.quantum_efficiency_at(r.lambda_peak);
        assert!(qe > 0.0 && qe <= 1.0, "QE={qe:.3}");
    }

    #[test]
    fn detector_bandwidth_f3db_positive() {
        let bw = DetectorBandwidth::ingaas_25ghz();
        let f = bw.f_3db_ghz();
        assert!(f > 0.0 && f < 500.0, "f_3dB={f:.1}GHz");
    }

    #[test]
    fn detector_bandwidth_f3db_less_than_both_limits() {
        let bw = DetectorBandwidth::ingaas_25ghz();
        let f3 = bw.f_3db_hz();
        assert!(f3 < bw.f_rc_hz(), "f_3dB should be < f_RC");
        assert!(f3 < bw.f_transit_hz(), "f_3dB should be < f_tr");
    }

    #[test]
    fn noise_nep_positive() {
        let n = DetectorNoise::new(0.9, 1e-3, 50.0);
        assert!(n.nep() > 0.0);
    }

    #[test]
    fn shot_noise_increases_with_current() {
        let n1 = DetectorNoise::new(0.9, 1e-3, 50.0);
        let n2 = DetectorNoise::new(0.9, 10e-3, 50.0);
        assert!(n2.shot_noise_psd() > n1.shot_noise_psd());
    }

    #[test]
    fn ingaas_25ghz_bandwidth_near_25() {
        let bw = DetectorBandwidth::ingaas_25ghz();
        let f = bw.f_3db_ghz();
        assert!(f > 1.0 && f < 200.0, "f_3dB={f:.1}GHz");
    }
}
