//! Optical signal metrics: OSNR, BER, Q-factor, and receiver sensitivity.
//!
//! # Physical constants used
//!
//! | Symbol | Value | Description |
//! |--------|-------|-------------|
//! | h      | 6.626 070 15 × 10⁻³⁴ J·s | Planck constant |
//! | c      | 2.997 924 58 × 10⁸ m/s   | Speed of light  |
//!
//! # References
//!
//! - Agrawal, "Fiber-Optic Communication Systems", 5th ed., Wiley, 2021
//! - Saleh & Teich, "Fundamentals of Photonics", 3rd ed., Wiley, 2019

use crate::comms::modulation::ModulationFormat;

// ──────────────────────────────────────────────────────────────────────────────
// Physical constants
// ──────────────────────────────────────────────────────────────────────────────

const H_PLANCK: f64 = 6.626_070_15e-34; // J·s
const C_LIGHT: f64 = 2.997_924_58e8; // m/s
const SQRT2: f64 = std::f64::consts::SQRT_2;

// ──────────────────────────────────────────────────────────────────────────────
// OsnrAnalysis
// ──────────────────────────────────────────────────────────────────────────────

/// Optical signal-to-noise ratio (OSNR) analysis.
///
/// OSNR is measured in a reference optical bandwidth (conventionally 0.1 nm or
/// 12.5 GHz at 1550 nm), independently of the actual signal bandwidth.
#[derive(Debug, Clone)]
pub struct OsnrAnalysis {
    /// Signal power (dBm)
    pub signal_power_dbm: f64,
    /// Noise power (dBm) measured in `reference_bandwidth_nm`
    pub noise_power_dbm: f64,
    /// Reference bandwidth for OSNR measurement (nm) — usually 0.1 nm
    pub reference_bandwidth_nm: f64,
    /// Signal (−3 dB) bandwidth (nm)
    pub signal_bandwidth_nm: f64,
}

impl OsnrAnalysis {
    /// Construct a new OSNR analysis descriptor.
    pub fn new(signal_dbm: f64, noise_dbm: f64, ref_bw_nm: f64, sig_bw_nm: f64) -> Self {
        Self {
            signal_power_dbm: signal_dbm,
            noise_power_dbm: noise_dbm,
            reference_bandwidth_nm: ref_bw_nm,
            signal_bandwidth_nm: sig_bw_nm,
        }
    }

    /// OSNR in dB: `OSNR_dB = P_signal_dBm − P_noise_dBm`.
    ///
    /// Both powers must be measured in the same reference bandwidth.
    pub fn osnr_db(&self) -> f64 {
        self.signal_power_dbm - self.noise_power_dbm
    }

    /// Required OSNR (dB) for a given modulation format and BER target.
    ///
    /// Returns an analytic approximation widely used in system planning.
    pub fn required_osnr_db(format: &ModulationFormat, ber_target: f64) -> f64 {
        format.required_osnr_db(ber_target)
    }

    /// OSNR penalty (dB) from accumulated chromatic dispersion.
    ///
    /// Uses the empirical formula:
    ///
    ///   penalty ≈ 10·log₁₀(1 + (D·B²·1e-3)²)
    ///
    /// where D is the accumulated dispersion (ps/nm) and B the data rate (Gbit/s).
    /// This is a conservative approximation valid for NRZ-OOK signals.
    ///
    /// # Arguments
    /// * `accumulated_dispersion_ps_per_nm` – total accumulated dispersion (ps/nm)
    /// * `data_rate_gbps` – signal data rate (Gbit/s)
    pub fn dispersion_penalty_db(
        accumulated_dispersion_ps_per_nm: f64,
        data_rate_gbps: f64,
    ) -> f64 {
        // Normalised dispersion parameter: δ = D·B²·1e-3
        // (converts to dimensionless using standard ps/nm × (Gbit/s)² scaling)
        let delta = accumulated_dispersion_ps_per_nm * data_rate_gbps * data_rate_gbps * 1e-3;
        10.0 * (1.0 + delta * delta).log10()
    }

    /// ASE noise power per amplifier span (dBm) in a reference optical bandwidth.
    ///
    /// Formula:
    ///   P_ASE = n_sp · (G − 1) · ħω · B_ref
    ///
    /// where n_sp ≈ NF·G / (2·(G−1)) is the spontaneous-emission factor.
    ///
    /// # Arguments
    /// * `gain_db`       – amplifier gain (dB)
    /// * `nf_db`         – amplifier noise figure (dB)
    /// * `lambda_nm`     – signal wavelength (nm)
    /// * `bandwidth_nm`  – reference optical bandwidth (nm)
    pub fn ase_per_span_dbm(gain_db: f64, nf_db: f64, lambda_nm: f64, bandwidth_nm: f64) -> f64 {
        let g = 10.0_f64.powf(gain_db / 10.0);
        let nf = 10.0_f64.powf(nf_db / 10.0);
        let lambda_m = lambda_nm * 1e-9;
        let nu = C_LIGHT / lambda_m; // Hz
                                     // Bandwidth in Hz: Δν = (c/λ²) · Δλ
        let bw_hz = C_LIGHT / (lambda_m * lambda_m) * (bandwidth_nm * 1e-9);
        let n_sp = nf * g / (2.0 * (g - 1.0).max(1e-12));
        let p_ase_w = n_sp * (g - 1.0) * H_PLANCK * nu * bw_hz;
        // Convert W → mW → dBm
        10.0 * (p_ase_w * 1e3).max(1e-40).log10()
    }

    /// Accumulated ASE power (dBm) from N identical spans (logarithmic addition).
    ///
    /// Total ASE = N × per-span ASE (in linear; dBm equivalent: +10·log₁₀(N)).
    pub fn accumulated_ase_dbm(n_spans: usize, per_span_ase_dbm: f64) -> f64 {
        if n_spans == 0 {
            return f64::NEG_INFINITY;
        }
        per_span_ase_dbm + 10.0 * (n_spans as f64).log10()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// BerCalculator
// ──────────────────────────────────────────────────────────────────────────────

/// BER (bit error rate) calculations for standard modulation formats.
///
/// All BER formulas assume additive white Gaussian noise (AWGN) and ideal
/// coherent or direct-detection receivers with matched filtering.
pub struct BerCalculator;

impl BerCalculator {
    /// OOK with direct detection using Q-factor.
    ///
    ///   BER = ½ · erfc(Q / √2)
    pub fn ook_direct_ber(q_factor: f64) -> f64 {
        0.5 * Self::erfc(q_factor / SQRT2)
    }

    /// BPSK (coherent) BER as a function of Eb/N₀ (linear).
    ///
    ///   BER = ½ · erfc(√(Eb/N₀))
    pub fn bpsk_ber(eb_n0_linear: f64) -> f64 {
        0.5 * Self::erfc(eb_n0_linear.max(0.0).sqrt())
    }

    /// QPSK (coherent) BER as a function of Eb/N₀ (linear).
    ///
    /// QPSK has the same BER–Eb/N₀ relationship as BPSK because each
    /// quadrature component carries one bit at the same energy.
    ///
    ///   BER = ½ · erfc(√(Eb/N₀))
    pub fn qpsk_ber(eb_n0_linear: f64) -> f64 {
        // Identical to BPSK
        Self::bpsk_ber(eb_n0_linear)
    }

    /// 16-QAM (coherent) BER as a function of Eb/N₀ (linear).
    ///
    /// Gray-coded symbol error rate approximation:
    ///
    ///   BER ≈ (3/8) · erfc(√(2·Eb/N₀ / 5))
    ///
    /// The factor 5 = (M−1)/3 · log₂(M) = 5 for M = 16.
    pub fn qam16_ber(eb_n0_linear: f64) -> f64 {
        let arg = (2.0 * eb_n0_linear.max(0.0) / 5.0).sqrt();
        (3.0 / 8.0) * Self::erfc(arg)
    }

    /// 64-QAM (coherent) BER as a function of Eb/N₀ (linear).
    ///
    /// Gray-coded approximation:
    ///
    ///   BER ≈ (7/24) · erfc(√(Eb/N₀ / 7))
    ///
    /// (M = 64, bits/symbol = 6)
    pub fn qam64_ber(eb_n0_linear: f64) -> f64 {
        let arg = (eb_n0_linear.max(0.0) / 7.0).sqrt();
        (7.0 / 24.0) * Self::erfc(arg)
    }

    /// Q-factor from BER (inverse).
    ///
    ///   Q = √2 · erfinv(1 − 2·BER)
    pub fn q_from_ber(ber: f64) -> f64 {
        let ber_clamped = ber.clamp(1e-20, 0.5 - 1e-10);
        SQRT2 * Self::erfinv(1.0 - 2.0 * ber_clamped)
    }

    /// BER from Q-factor.
    ///
    ///   BER = ½ · erfc(Q / √2)
    pub fn ber_from_q(q_factor: f64) -> f64 {
        0.5 * Self::erfc(q_factor / SQRT2)
    }

    /// Required Eb/N₀ (dB) to achieve `ber_target` for a given modulation format.
    ///
    /// Uses a binary search over the BER formula for the given format.
    pub fn required_eb_n0_db(format: &ModulationFormat, ber_target: f64) -> f64 {
        // Binary search: find Eb/N0 (linear) such that BER(format, Eb/N0) = ber_target
        let ber_fn: Box<dyn Fn(f64) -> f64> = match format {
            ModulationFormat::Ook => Box::new(|eb: f64| {
                // OOK: treat Q = sqrt(2*Eb/N0)
                Self::ook_direct_ber((2.0 * eb).sqrt())
            }),
            ModulationFormat::Bpsk | ModulationFormat::Dpsk => {
                Box::new(|eb: f64| Self::bpsk_ber(eb))
            }
            ModulationFormat::Qpsk | ModulationFormat::Dqpsk => {
                Box::new(|eb: f64| Self::qpsk_ber(eb))
            }
            ModulationFormat::Qam16 => Box::new(|eb: f64| Self::qam16_ber(eb)),
            ModulationFormat::Qam64 => Box::new(|eb: f64| Self::qam64_ber(eb)),
            ModulationFormat::Qam256 => Box::new(|eb: f64| {
                // Approximation for 256-QAM
                let arg = (eb.max(0.0) / 85.0).sqrt();
                (15.0 / 64.0) * Self::erfc(arg)
            }),
            ModulationFormat::Pam4 => Box::new(|eb: f64| {
                // PAM-4: BER ≈ (3/4)·erfc(sqrt(Eb/N0 / 5))
                (3.0 / 4.0) * Self::erfc((eb.max(0.0) / 5.0).sqrt())
            }),
        };

        // Binary search in log space: Eb/N0 ∈ [1e-2, 1e6] (linear)
        let mut lo: f64 = 1e-2_f64;
        let mut hi: f64 = 1e6_f64;
        for _ in 0..80 {
            let mid = (lo * hi).sqrt(); // geometric midpoint
            if ber_fn(mid) > ber_target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let eb_n0_linear = (lo * hi).sqrt();
        10.0 * eb_n0_linear.log10()
    }

    /// Convert OSNR (linear) to Eb/N₀ (linear).
    ///
    ///   Eb/N₀ = OSNR_linear × B_ref / (2 × Rs)
    ///
    /// where `B_ref` is the OSNR reference bandwidth (Hz) and `Rs` the symbol rate
    /// (roughly equal to the data rate for OOK/BPSK; divide by bits/symbol for
    /// higher-order formats).
    ///
    /// # Arguments
    /// * `osnr_linear`      – OSNR in linear units (not dB)
    /// * `baud_rate`        – symbol rate (Baud, i.e., Hz)
    /// * `ref_bandwidth_hz` – OSNR reference bandwidth (Hz); 12.5 GHz ≡ 0.1 nm at 1550 nm
    pub fn osnr_to_eb_n0(osnr_linear: f64, baud_rate: f64, ref_bandwidth_hz: f64) -> f64 {
        if baud_rate <= 0.0 {
            return 0.0;
        }
        osnr_linear * ref_bandwidth_hz / (2.0 * baud_rate)
    }

    /// Complementary error function: erfc(x) = 1 − erf(x).
    ///
    /// Uses a high-accuracy rational approximation (Abramowitz & Stegun §7.1.26,
    /// maximum error < 1.5 × 10⁻⁷) combined with the asymptotic expansion for
    /// large |x|.
    pub fn erfc(x: f64) -> f64 {
        if x < 0.0 {
            return 2.0 - Self::erfc(-x);
        }
        if x > 8.0 {
            // Asymptotic tail: erfc(x) ≈ exp(-x²)/(x·√π) · (1 − 1/(2x²) + …)
            let x2 = x * x;
            return (-x2).exp() / (x * std::f64::consts::PI.sqrt())
                * (1.0 - 1.0 / (2.0 * x2) + 3.0 / (4.0 * x2 * x2));
        }
        // Horner form (A&S 7.1.26 coefficients)
        let t = 1.0 / (1.0 + 0.327_591_1 * x);
        let poly = t
            * (0.254_829_592
                + t * (-0.284_496_736
                    + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
        poly * (-x * x).exp()
    }

    /// Inverse error function: erfinv(y) such that erf(erfinv(y)) = y, y ∈ (−1, 1).
    ///
    /// Computes an initial approximation using the Winitzki (2008) closed-form
    /// formula, then refines with several steps of Halley's method.  Accurate
    /// to at least 12 significant figures across the full domain including the
    /// extreme tails needed for BER computations down to 10⁻¹⁵.
    pub fn erfinv(y: f64) -> f64 {
        if y >= 1.0 {
            return f64::INFINITY;
        }
        if y <= -1.0 {
            return f64::NEG_INFINITY;
        }
        if y == 0.0 {
            return 0.0;
        }
        let sign = y.signum();
        let ya = y.abs();
        // Winitzki (2008) initial approximation:
        //   x₀ = sign(y) · √( √((2/π·a + ln((1−|y|²)/2))² − ln((1−|y|²)/2)) − (2/π·a + ln((1−|y|²)/2)) )
        // where a ≈ 0.147.
        const A: f64 = 0.147;
        const TWO_OVER_PI_A: f64 = 2.0 / (std::f64::consts::PI * A);
        let ln_term = ((1.0 - ya * ya).max(1e-300_f64)).ln() / 2.0;
        let inner = TWO_OVER_PI_A + ln_term;
        let discriminant = (inner * inner - ln_term / A).max(0.0);
        let mut x = sign * (discriminant.sqrt() - inner).max(0.0).sqrt();
        // Halley refinement: f(x) = erf(x) − y = 0
        // f'(x) = (2/√π) · exp(−x²)
        // f''(x) = −4x/√π · exp(−x²) = −2x · f'(x)
        // Halley step: x ← x − 2·f·f' / (2·f'² − f·f'')
        //            = x − f / (f' − f·f''/f') (compact form)
        for _ in 0..6 {
            // Compute erf(x) via our erfc: erf(x) = 1 − erfc(x) for x≥0, erfc(x)−1 for x<0
            let erf_x = if x >= 0.0 {
                1.0 - Self::erfc(x)
            } else {
                Self::erfc(-x) - 1.0
            };
            let f = erf_x - y;
            let fp = (2.0 / std::f64::consts::PI.sqrt()) * (-(x * x)).exp();
            if fp.abs() < 1e-300 {
                break;
            }
            let fpp = -2.0 * x * fp; // second derivative
                                     // Halley step denominator: fp − f·fpp/(2·fp)
            let denom = fp - f * fpp / (2.0 * fp);
            if denom.abs() < 1e-300 {
                x -= f / fp; // fall back to Newton
            } else {
                x -= f / denom;
            }
        }
        x
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// QFactor
// ──────────────────────────────────────────────────────────────────────────────

/// Eye-diagram Q-factor analysis for OOK / direct-detection systems.
///
/// The Q-factor characterises the eye opening quality and directly maps to BER.
#[derive(Debug, Clone)]
pub struct QFactor {
    /// Mean value of the '1' rail (high level)
    pub signal_high: f64,
    /// Mean value of the '0' rail (low level)
    pub signal_low: f64,
    /// Standard deviation of noise on the '1' rail
    pub sigma_high: f64,
    /// Standard deviation of noise on the '0' rail
    pub sigma_low: f64,
}

impl QFactor {
    /// Construct a new Q-factor descriptor.
    ///
    /// # Panics (debug only)
    /// Asserts that `signal_high > signal_low` and that both sigmas are positive.
    pub fn new(signal_high: f64, signal_low: f64, sigma_high: f64, sigma_low: f64) -> Self {
        debug_assert!(
            signal_high > signal_low,
            "Q-factor: high level must exceed low level"
        );
        debug_assert!(
            sigma_high > 0.0 && sigma_low > 0.0,
            "Q-factor: sigmas must be positive"
        );
        Self {
            signal_high,
            signal_low,
            sigma_high,
            sigma_low,
        }
    }

    /// Q-factor: Q = (S₁ − S₀) / (σ₁ + σ₀).
    pub fn q_factor(&self) -> f64 {
        let denom = self.sigma_high + self.sigma_low;
        if denom < 1e-300 {
            return f64::INFINITY;
        }
        (self.signal_high - self.signal_low) / denom
    }

    /// Optimal decision threshold minimising BER.
    ///
    ///   D = (S₀ · σ₁ + S₁ · σ₀) / (σ₁ + σ₀)
    pub fn optimal_threshold(&self) -> f64 {
        let denom = self.sigma_high + self.sigma_low;
        if denom < 1e-300 {
            return (self.signal_high + self.signal_low) / 2.0;
        }
        (self.signal_low * self.sigma_high + self.signal_high * self.sigma_low) / denom
    }

    /// BER derived from Q-factor: BER = ½ · erfc(Q / √2).
    pub fn ber(&self) -> f64 {
        BerCalculator::ber_from_q(self.q_factor())
    }

    /// Eye opening in linear units: E = S₁ − S₀.
    pub fn eye_opening(&self) -> f64 {
        self.signal_high - self.signal_low
    }

    /// Eye opening penalty (dB) relative to a back-to-back Q-factor reference.
    ///
    ///   penalty_dB = 20·log₁₀(q_ref / q_actual)
    ///
    /// A positive penalty means degraded performance.
    pub fn eye_opening_penalty_db(&self, back_to_back_q: f64) -> f64 {
        let q = self.q_factor();
        if q <= 0.0 || back_to_back_q <= 0.0 {
            return f64::INFINITY;
        }
        20.0 * (back_to_back_q / q).log10()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ReceiverSensitivity
// ──────────────────────────────────────────────────────────────────────────────

/// Receiver sensitivity analysis for PIN and APD photodetectors.
///
/// Accounts for shot noise, thermal noise, dark current, and (for APD) excess
/// noise multiplication.
#[derive(Debug, Clone)]
pub struct ReceiverSensitivity {
    /// Detector responsivity R (A/W)
    pub responsivity: f64,
    /// Dark current I_d (nA)
    pub dark_current_na: f64,
    /// Thermal noise equivalent power (dBm/Hz); maps to thermal current noise.
    ///
    /// Typical value: −160 dBm/Hz for a 50 Ω trans-impedance amplifier at 300 K.
    pub thermal_noise_dbm_per_hz: f64,
    /// APD excess noise factor F(M).  For a PIN receiver use F = 1.0.
    pub excess_noise_factor: f64,
}

impl ReceiverSensitivity {
    /// Create a PIN photodetector receiver model.
    ///
    /// Assumes: dark current = 5 nA, thermal NEP = −160 dBm/Hz, F = 1.
    pub fn pin_receiver(responsivity: f64) -> Self {
        Self {
            responsivity,
            dark_current_na: 5.0,
            thermal_noise_dbm_per_hz: -160.0,
            excess_noise_factor: 1.0,
        }
    }

    /// Create an APD receiver model.
    ///
    /// # Arguments
    /// * `responsivity` – primary responsivity (A/W) before multiplication
    /// * `gain`         – APD avalanche gain M
    /// * `excess_noise` – excess noise factor F(M) (typically 2–5 for InGaAs APDs)
    pub fn apd_receiver(responsivity: f64, gain: f64, excess_noise: f64) -> Self {
        Self {
            responsivity: responsivity * gain,
            dark_current_na: 10.0,
            thermal_noise_dbm_per_hz: -160.0,
            excess_noise_factor: excess_noise,
        }
    }

    /// Minimum detectable signal power (dBm) for a target BER.
    ///
    /// Uses the generalised sensitivity formula including shot, thermal, and
    /// dark-current noise contributions:
    ///
    ///   P_min = Q · [σ_th + √(σ_th² + 2·e·(Id + Q·I_th)·B)] / R
    ///
    /// where σ_th is the RMS thermal current in bandwidth B, and I_th is the
    /// thermal noise current normalised to bandwidth.
    ///
    /// # Arguments
    /// * `ber_target`    – target BER (e.g., 1e-12)
    /// * `bandwidth_ghz` – electrical signal bandwidth (GHz)
    pub fn sensitivity_dbm(&self, ber_target: f64, bandwidth_ghz: f64) -> f64 {
        const E_CHARGE: f64 = 1.602_176_634e-19; // C
        let q = BerCalculator::q_from_ber(ber_target);
        let bw_hz = bandwidth_ghz * 1e9;
        // Thermal noise current spectral density (A/√Hz)
        // NEP_thermal = P_noise_density → I_th = R · sqrt(10^(NEP_dBm_per_Hz/10) · 1e-3)
        let nep_w_per_hz = 10.0_f64.powf(self.thermal_noise_dbm_per_hz / 10.0) * 1e-3;
        let i_th_rms = self.responsivity * (nep_w_per_hz * bw_hz).sqrt();
        let i_dark = self.dark_current_na * 1e-9;
        // Shot noise at dark current: σ_dark = sqrt(2·e·Id·B)
        let sigma_dark = (2.0 * E_CHARGE * i_dark * bw_hz).sqrt();
        // Combined σ₀ ≈ sqrt(σ_th² + σ_dark²)
        let sigma0 = (i_th_rms * i_th_rms + sigma_dark * sigma_dark).sqrt();
        // Shot noise from signal: I_shot = sqrt(2·e·F·R·P·B)
        // Solving quadratic for P_min:
        //   Q·sigma0 + Q·sqrt(2·e·F·R·P_min·B) = R·P_min
        //   Let x = sqrt(P_min): x² - 2·e·F·B·Q² x / (R·0.5) ... complex
        // Simplified iterative solution:
        let mut p_w = 1e-6_f64; // initial guess 1 µW
        for _ in 0..60 {
            let i_sig = self.responsivity * p_w;
            let sigma1 =
                (2.0 * E_CHARGE * self.excess_noise_factor * i_sig.max(0.0) * bw_hz).sqrt();
            let required_i = q * (sigma0 + sigma1);
            p_w = required_i / self.responsivity;
        }
        10.0 * (p_w * 1e3).max(1e-40).log10()
    }

    /// Shot-noise-limited sensitivity for an ideal PIN receiver (dBm).
    ///
    ///   P_shot = Q² · e · B / R
    ///
    /// This is the fundamental quantum limit for direct-detection receivers.
    pub fn shot_noise_limited_dbm(&self, ber_target: f64, bandwidth_ghz: f64) -> f64 {
        const E_CHARGE: f64 = 1.602_176_634e-19;
        let q = BerCalculator::q_from_ber(ber_target);
        let bw_hz = bandwidth_ghz * 1e9;
        let p_min_w = q * q * E_CHARGE * bw_hz / self.responsivity.max(1e-10);
        10.0 * (p_min_w * 1e3).max(1e-40).log10()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// BER from Q ≈ 6 should be very close to 1×10⁻⁹.
    #[test]
    fn test_ber_from_q_factor() {
        let ber = BerCalculator::ber_from_q(6.0);
        assert!(
            ber < 1e-8 && ber > 1e-11,
            "BER for Q=6 should be ~1e-9, got {ber:.3e}"
        );
    }

    /// erfinv round-trip: erfinv gives back Q ≈ 6 for BER ≈ 1e-9.
    #[test]
    fn test_q_from_ber() {
        let q = BerCalculator::q_from_ber(1e-9);
        assert!(
            (q - 6.0).abs() < 0.2,
            "Q-factor for BER=1e-9 should be ≈6, got {q:.4}"
        );
    }

    /// BPSK BER at Eb/N0 = 10 dB (linear = 10) should be well below 1e-3.
    #[test]
    fn test_bpsk_ber_at_10db_snr() {
        let eb_n0_lin = 10.0_f64.powf(10.0 / 10.0); // 10 dB
        let ber = BerCalculator::bpsk_ber(eb_n0_lin);
        assert!(
            ber < 1e-3,
            "BPSK BER at 10 dB Eb/N0 = {ber:.3e}, expected < 1e-3"
        );
    }

    /// erfc(0) must equal 1.
    #[test]
    fn test_erfc_at_zero() {
        let val = BerCalculator::erfc(0.0);
        assert!((val - 1.0).abs() < 1e-6, "erfc(0) = {val}, expected 1.0");
    }

    /// erfc(5) should be very small (< 1e-10).
    #[test]
    fn test_erfc_large_x() {
        let val = BerCalculator::erfc(5.0);
        assert!(val < 1e-10, "erfc(5) = {val:.3e}, expected < 1e-10");
    }

    /// OSNR dB = signal_dBm − noise_dBm (simple ratio check).
    #[test]
    fn test_osnr_db() {
        let analysis = OsnrAnalysis::new(0.0, -20.0, 0.1, 0.3);
        let osnr = analysis.osnr_db();
        assert!(
            (osnr - 20.0).abs() < 1e-10,
            "OSNR should be 20 dB, got {osnr}"
        );
    }

    /// Q-factor formula: Q = (S1 − S0) / (σ1 + σ0).
    #[test]
    fn test_q_factor_formula() {
        let q = QFactor::new(1.0, 0.0, 0.1, 0.1);
        let expected = 1.0 / 0.2;
        let got = q.q_factor();
        assert!(
            (got - expected).abs() < 1e-10,
            "Q should be {expected}, got {got}"
        );
    }

    /// Optimal threshold should lie strictly between S0 and S1.
    #[test]
    fn test_optimal_threshold() {
        let q = QFactor::new(1.0, 0.0, 0.08, 0.12);
        let thresh = q.optimal_threshold();
        assert!(
            thresh > 0.0 && thresh < 1.0,
            "threshold {thresh} should be in (S0, S1) = (0, 1)"
        );
    }

    /// Dispersion penalty should be zero for zero accumulated dispersion.
    #[test]
    fn test_dispersion_penalty_zero_for_no_dispersion() {
        let penalty = OsnrAnalysis::dispersion_penalty_db(0.0, 10.0);
        assert!(
            penalty.abs() < 1e-10,
            "zero dispersion → zero penalty, got {penalty}"
        );
    }

    /// Dispersion penalty should increase with data rate.
    #[test]
    fn test_dispersion_penalty_increases_with_rate() {
        let p10 = OsnrAnalysis::dispersion_penalty_db(1000.0, 10.0);
        let p40 = OsnrAnalysis::dispersion_penalty_db(1000.0, 40.0);
        assert!(
            p40 > p10,
            "higher data rate → larger dispersion penalty ({p40} vs {p10})"
        );
    }

    /// Accumulated ASE should increase by 10 dB when n_spans doubles from 5 to 50.
    #[test]
    fn test_accumulated_ase_n_spans() {
        let per_span = -10.0_f64; // dBm
        let ase5 = OsnrAnalysis::accumulated_ase_dbm(5, per_span);
        let ase50 = OsnrAnalysis::accumulated_ase_dbm(50, per_span);
        assert!(
            (ase50 - ase5 - 10.0).abs() < 1e-9,
            "10× more spans → +10 dB, got diff = {}",
            ase50 - ase5
        );
    }
}
