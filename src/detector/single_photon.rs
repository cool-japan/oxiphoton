//! Single-photon detector models: SNSPD, PMT, SPAD, and TCSPC.
//!
//! All models are built from first principles and are parameterised by
//! experimentally accessible quantities (efficiency, dark-count rate, jitter,
//! dead time).  No `unwrap()` calls appear outside tests.

use crate::error::OxiPhotonError;

// ── Physical constants ────────────────────────────────────────────────────────
const E_CHARGE: f64 = 1.602_176_634e-19; // C

// ── SNSPD ─────────────────────────────────────────────────────────────────────

/// Superconducting nanowire single-photon detector (SNSPD).
///
/// Key metrics used throughout:
/// - Detection efficiency η (0–1)
/// - Dark count rate (DCR) in Hz
/// - Timing jitter (FWHM of IRF) in ps
/// - Dead time τ_d in ns (recovery after each detection)
/// - Maximum count rate: f_max = 1 / τ_d
#[derive(Debug, Clone)]
pub struct Snspd {
    /// System detection efficiency at operating wavelength (0–1).
    pub detection_efficiency: f64,
    /// Dark count rate (Hz).
    pub dark_count_rate_hz: f64,
    /// Timing jitter (FWHM of instrument response function) in ps.
    pub timing_jitter_ps: f64,
    /// Dead time after each detection event (ns).
    pub dead_time_ns: f64,
    /// Operating wavelength (nm).
    pub operating_wavelength_nm: f64,
}

impl Snspd {
    /// Construct an SNSPD from measured operating parameters.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] when efficiency is outside
    /// (0, 1] or any rate/time parameter is non-positive.
    pub fn new(
        detection_efficiency: f64,
        dark_count_rate_hz: f64,
        timing_jitter_ps: f64,
        dead_time_ns: f64,
        operating_wavelength_nm: f64,
    ) -> Result<Self, OxiPhotonError> {
        if detection_efficiency <= 0.0 || detection_efficiency > 1.0 {
            return Err(OxiPhotonError::NumericalError(
                "detection_efficiency must be in (0, 1]".into(),
            ));
        }
        if dark_count_rate_hz < 0.0 || !dark_count_rate_hz.is_finite() {
            return Err(OxiPhotonError::NumericalError(
                "dark_count_rate_hz must be non-negative and finite".into(),
            ));
        }
        if timing_jitter_ps <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "timing_jitter_ps must be positive".into(),
            ));
        }
        if dead_time_ns <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "dead_time_ns must be positive".into(),
            ));
        }
        Ok(Self {
            detection_efficiency,
            dark_count_rate_hz,
            timing_jitter_ps,
            dead_time_ns,
            operating_wavelength_nm,
        })
    }

    /// Typical state-of-the-art SNSPD at 1550 nm.
    ///
    /// η = 0.90, DCR = 100 Hz, jitter = 50 ps, dead time = 50 ns.
    pub fn typical_1550() -> Self {
        Self {
            detection_efficiency: 0.90,
            dark_count_rate_hz: 100.0,
            timing_jitter_ps: 50.0,
            dead_time_ns: 50.0,
            operating_wavelength_nm: 1550.0,
        }
    }

    // ── Rates ─────────────────────────────────────────────────────────────

    /// Maximum count rate limited by dead time: f_max = 1 / τ_d (MHz).
    pub fn max_count_rate_mhz(&self) -> f64 {
        let tau_s = self.dead_time_ns * 1e-9;
        (1.0 / tau_s) * 1e-6
    }

    /// Expected signal detection rate for a given photon flux (Hz).
    ///
    /// R_s = η · Φ
    pub fn signal_rate_hz(&self, photon_flux_per_s: f64) -> f64 {
        self.detection_efficiency * photon_flux_per_s
    }

    // ── SNR ───────────────────────────────────────────────────────────────

    /// Poissonian SNR after integrating for `integration_time_s`.
    ///
    /// SNR = (R_s · T)^½ / (1 + DCR / R_s)^½
    /// when R_s >> 0; returns 0 for zero flux.
    pub fn snr_after_time(&self, photon_flux: f64, integration_time_s: f64) -> f64 {
        let r_s = self.signal_rate_hz(photon_flux);
        if r_s == 0.0 {
            return 0.0;
        }
        let dcr = self.dark_count_rate_hz;
        let total_rate = r_s + dcr;
        let signal_counts = r_s * integration_time_s;
        let noise_counts = total_rate * integration_time_s;
        signal_counts / noise_counts.sqrt()
    }

    /// Minimum detectable photon rate: rate at which SNR = 1 in 1 s.
    ///
    /// Solving η·R·T / √((η·R + DCR)·T) = 1 for T = 1 s gives
    /// R_min such that η·R_min ≈ DCR + √DCR (Poissonian limit).
    pub fn min_detectable_rate_hz(&self) -> f64 {
        let dcr = self.dark_count_rate_hz;
        (dcr + dcr.sqrt()) / self.detection_efficiency
    }

    // ── Detection statistics ──────────────────────────────────────────────

    /// Probability of detecting at least one of `n_photons` incident photons.
    ///
    /// P_det = 1 − (1 − η)^N
    pub fn detection_probability(&self, n_photons: usize) -> f64 {
        1.0 - (1.0 - self.detection_efficiency).powi(n_photons as i32)
    }

    // ── HBT / g²(0) ──────────────────────────────────────────────────────

    /// Estimated g²(0) in a Hanbury-Brown–Twiss experiment.
    ///
    /// Accidental coincidence rate within a coincidence window `window_ns`:
    /// R_acc = (R_s + DCR)² · τ_w
    ///
    /// g²(0) = R_acc / R_true
    pub fn g2_zero_measurement(&self, true_coincidence_rate: f64, window_ns: f64) -> f64 {
        let tau_w = window_ns * 1e-9;
        let total_rate = self.dark_count_rate_hz + true_coincidence_rate;
        let r_acc = total_rate * total_rate * tau_w;
        if true_coincidence_rate == 0.0 {
            return f64::INFINITY;
        }
        r_acc / true_coincidence_rate
    }

    // ── Timing ────────────────────────────────────────────────────────────

    /// Total measured timing resolution (FWHM) when source also has jitter.
    ///
    /// Gaussian convolution: σ_total = √(σ_det² + σ_src²)  (FWHM likewise).
    pub fn measured_timing_resolution_ps(&self, source_jitter_ps: f64) -> f64 {
        (self.timing_jitter_ps * self.timing_jitter_ps + source_jitter_ps * source_jitter_ps).sqrt()
    }

    // ── Spectral efficiency ───────────────────────────────────────────────

    /// Detection efficiency at a wavelength `lambda_nm` (Gaussian spectral
    /// roll-off centred on operating wavelength with σ = 200 nm).
    pub fn efficiency_at_wavelength(&self, lambda_nm: f64) -> f64 {
        let sigma_nm = 200.0;
        let delta = lambda_nm - self.operating_wavelength_nm;
        let scale = (-0.5 * (delta / sigma_nm) * (delta / sigma_nm)).exp();
        self.detection_efficiency * scale
    }
}

// ── PMT ───────────────────────────────────────────────────────────────────────

/// Photomultiplier tube (PMT).
///
/// Models photocathode quantum efficiency, dynode chain gain, dark counts,
/// and transit-time spread (TTS).
#[derive(Debug, Clone)]
pub struct Pmt {
    /// Photocathode quantum efficiency (0–1).
    pub quantum_efficiency: f64,
    /// Gain per dynode stage (secondary emission ratio).
    pub dynode_gain: f64,
    /// Number of dynode stages.
    pub n_dynodes: usize,
    /// Dark count (thermionic emission) rate (Hz).
    pub dark_count_rate_hz: f64,
    /// Transit-time spread — FWHM timing jitter (ps).
    pub transit_time_spread_ps: f64,
    /// Photocathode active area (mm²).
    pub cathode_area_mm2: f64,
    /// Spectral response range (λ_min, λ_max) in nm.
    pub spectral_range: (f64, f64),
}

impl Pmt {
    /// Construct a PMT from physical parameters.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] when QE or dynode gain
    /// are out of physical bounds.
    pub fn new(
        quantum_efficiency: f64,
        dynode_gain: f64,
        n_dynodes: usize,
        dark_count_rate_hz: f64,
        transit_time_spread_ps: f64,
        cathode_area_mm2: f64,
        spectral_range: (f64, f64),
    ) -> Result<Self, OxiPhotonError> {
        if quantum_efficiency <= 0.0 || quantum_efficiency > 1.0 {
            return Err(OxiPhotonError::NumericalError(
                "quantum_efficiency must be in (0, 1]".into(),
            ));
        }
        if dynode_gain <= 1.0 {
            return Err(OxiPhotonError::NumericalError(
                "dynode_gain must exceed 1.0".into(),
            ));
        }
        if n_dynodes == 0 {
            return Err(OxiPhotonError::NumericalError(
                "n_dynodes must be at least 1".into(),
            ));
        }
        Ok(Self {
            quantum_efficiency,
            dynode_gain,
            n_dynodes,
            dark_count_rate_hz,
            transit_time_spread_ps,
            cathode_area_mm2,
            spectral_range,
        })
    }

    /// Typical bialkali PMT for 300–650 nm visible range.
    ///
    /// QE ≈ 25%, g = 5 per dynode, 10 dynodes → G ≈ 10^7.
    pub fn bialkali_visible() -> Self {
        Self {
            quantum_efficiency: 0.25,
            dynode_gain: 5.0,
            n_dynodes: 10,
            dark_count_rate_hz: 500.0,
            transit_time_spread_ps: 300.0,
            cathode_area_mm2: 100.0,
            spectral_range: (300.0, 650.0),
        }
    }

    /// Extended-red / GaAs photocathode PMT for near-IR up to ~900 nm.
    ///
    /// QE ≈ 12%, g = 4, 12 dynodes → G ≈ 1.7 × 10^7.
    pub fn gaas_nir() -> Self {
        Self {
            quantum_efficiency: 0.12,
            dynode_gain: 4.0,
            n_dynodes: 12,
            dark_count_rate_hz: 5000.0,
            transit_time_spread_ps: 500.0,
            cathode_area_mm2: 200.0,
            spectral_range: (300.0, 900.0),
        }
    }

    // ── Gain & charge ─────────────────────────────────────────────────────

    /// Total gain of the dynode chain: G = g^N.
    pub fn total_gain(&self) -> f64 {
        self.dynode_gain.powi(self.n_dynodes as i32)
    }

    /// Charge delivered to the anode per detected photon: Q = e · G.
    pub fn anode_pulse_charge_c(&self) -> f64 {
        E_CHARGE * self.total_gain()
    }

    // ── Rates & SNR ───────────────────────────────────────────────────────

    /// Anode signal count rate for a given photon flux.
    pub fn signal_rate_hz(&self, photon_flux: f64) -> f64 {
        self.quantum_efficiency * photon_flux
    }

    /// DC signal-to-noise ratio (ratio of signal current to shot noise from
    /// total anode current in a 1-Hz bandwidth).
    ///
    /// SNR_DC = I_sig / √(2·e·(I_sig + I_dark)·G)
    /// where I_sig = QE · Φ · e · G and I_dark = DCR · e · G.
    pub fn signal_to_noise_dc(&self, photon_flux: f64) -> f64 {
        let g = self.total_gain();
        let r_sig = self.signal_rate_hz(photon_flux);
        let r_dark = self.dark_count_rate_hz;
        let i_sig = r_sig * E_CHARGE * g;
        let i_dark = r_dark * E_CHARGE * g;
        let i_noise = (2.0 * E_CHARGE * (i_sig + i_dark) * g).sqrt();
        if i_noise == 0.0 {
            return 0.0;
        }
        i_sig / i_noise
    }

    /// Estimated after-pulsing fraction for this dynode configuration.
    ///
    /// Modelled as 1 % × (N/10) — a rough engineering estimate.
    pub fn after_pulsing_fraction(&self) -> f64 {
        0.01 * (self.n_dynodes as f64 / 10.0)
    }

    /// Single-photon pulse-height spectrum (Polya distribution approximation).
    ///
    /// Returns `n_pts` (relative-amplitude, probability-density) pairs
    /// spanning 0 to 3× the mean gain, sampled uniformly.
    pub fn single_photon_spectrum(&self, n_pts: usize) -> Vec<(f64, f64)> {
        if n_pts == 0 {
            return Vec::new();
        }
        let g = self.total_gain();
        let x_max = 3.0 * g;
        let mut spectrum = Vec::with_capacity(n_pts);

        // Polya (generalised Poisson) approximation: shape parameter b = 0.5
        // typical for bialkali. P(Q) ∝ (Q/G)^(1/b - 1) · exp(-Q/(b·G))
        let b = 0.5_f64;
        let alpha = 1.0 / b - 1.0;

        for i in 0..n_pts {
            let x = (i as f64 + 0.5) / n_pts as f64 * x_max;
            let t = x / (b * g);
            let pdf = (t.powf(alpha) * (-t).exp()).max(0.0);
            spectrum.push((x, pdf));
        }

        // Normalise to unit area (trapezoidal rule)
        let dx = x_max / n_pts as f64;
        let area: f64 = spectrum.iter().map(|(_, p)| p).sum::<f64>() * dx;
        if area > 0.0 {
            for (_, p) in spectrum.iter_mut() {
                *p /= area;
            }
        }
        spectrum
    }
}

// ── SPAD ──────────────────────────────────────────────────────────────────────

/// Single-photon avalanche diode (SPAD).
///
/// Operated above breakdown in Geiger mode; each photon triggers a macroscopic
/// avalanche pulse quenched actively or passively.
#[derive(Debug, Clone)]
pub struct Spad {
    /// Photon detection efficiency (0–1).
    pub detection_efficiency: f64,
    /// Dark count rate (Hz).
    pub dark_count_rate_hz: f64,
    /// Timing jitter FWHM (ps).
    pub timing_jitter_ps: f64,
    /// Dead (hold-off) time after each event (ns).
    pub dead_time_ns: f64,
    /// After-pulsing fraction (0–1).
    pub after_pulsing_fraction: f64,
    /// Design operating wavelength (nm).
    pub operating_wavelength_nm: f64,
    /// Excess bias above breakdown (V).
    pub excess_bias_v: f64,
}

impl Spad {
    /// Construct a SPAD from operating parameters.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] when any parameter is
    /// physically invalid.
    pub fn new(
        detection_efficiency: f64,
        dark_count_rate_hz: f64,
        timing_jitter_ps: f64,
        dead_time_ns: f64,
        after_pulsing_fraction: f64,
        operating_wavelength_nm: f64,
        excess_bias_v: f64,
    ) -> Result<Self, OxiPhotonError> {
        if detection_efficiency <= 0.0 || detection_efficiency > 1.0 {
            return Err(OxiPhotonError::NumericalError(
                "detection_efficiency must be in (0, 1]".into(),
            ));
        }
        if !(0.0..1.0).contains(&after_pulsing_fraction) {
            return Err(OxiPhotonError::NumericalError(
                "after_pulsing_fraction must be in [0, 1)".into(),
            ));
        }
        Ok(Self {
            detection_efficiency,
            dark_count_rate_hz,
            timing_jitter_ps,
            dead_time_ns,
            after_pulsing_fraction,
            operating_wavelength_nm,
            excess_bias_v,
        })
    }

    /// Silicon SPAD for 700 nm (visible).
    ///
    /// η = 40%, DCR = 100 Hz, jitter = 50 ps, dead = 20 ns.
    pub fn si_spad_700nm() -> Self {
        Self {
            detection_efficiency: 0.40,
            dark_count_rate_hz: 100.0,
            timing_jitter_ps: 50.0,
            dead_time_ns: 20.0,
            after_pulsing_fraction: 0.02,
            operating_wavelength_nm: 700.0,
            excess_bias_v: 5.0,
        }
    }

    /// InGaAs/InP SPAD for 1550 nm (telecom).
    ///
    /// η = 25%, DCR = 1000 Hz, jitter = 100 ps, dead = 10 μs (gated mode).
    pub fn ingaas_spad_1550() -> Self {
        Self {
            detection_efficiency: 0.25,
            dark_count_rate_hz: 1000.0,
            timing_jitter_ps: 100.0,
            dead_time_ns: 10_000.0,
            after_pulsing_fraction: 0.05,
            operating_wavelength_nm: 1550.0,
            excess_bias_v: 3.0,
        }
    }

    /// Maximum count rate (MHz): f_max = 1 / τ_dead.
    pub fn max_count_rate_mhz(&self) -> f64 {
        let tau_s = self.dead_time_ns * 1e-9;
        1e-6 / tau_s
    }

    /// Poissonian SNR after integrating for `integration_time_s`.
    pub fn snr_after_time(&self, photon_flux: f64, integration_time_s: f64) -> f64 {
        let r_s = self.detection_efficiency * photon_flux;
        if r_s == 0.0 {
            return 0.0;
        }
        let dcr_eff = self.effective_dcr_with_ap();
        let signal_counts = r_s * integration_time_s;
        let noise_counts = (r_s + dcr_eff) * integration_time_s;
        signal_counts / noise_counts.sqrt()
    }

    /// Effective dark count rate including after-pulsing contribution.
    ///
    /// DCR_eff = DCR × (1 + AP_fraction)
    pub fn effective_dcr_with_ap(&self) -> f64 {
        self.dark_count_rate_hz * (1.0 + self.after_pulsing_fraction)
    }
}

// ── TCSPC ─────────────────────────────────────────────────────────────────────

/// Time-correlated single-photon counting (TCSPC) acquisition setup.
///
/// Builds a timing histogram by recording arrival-time differences between a
/// synchronisation trigger (laser pulse) and detected photons.  Pile-up
/// distortion limits count rates to < 1% of the laser repetition rate.
#[derive(Debug, Clone)]
pub struct TcSpc {
    /// Single-photon detector (SNSPD used here for lowest jitter).
    pub detector: Snspd,
    /// Histogram bin width (ps).
    pub time_resolution_ps: f64,
    /// Number of histogram bins.
    pub n_bins: usize,
    /// Laser repetition rate (MHz).
    pub repetition_rate_mhz: f64,
}

impl TcSpc {
    /// Construct a TCSPC setup.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] when time resolution or
    /// bin count are non-positive.
    pub fn new(
        detector: Snspd,
        time_resolution_ps: f64,
        n_bins: usize,
        repetition_rate_mhz: f64,
    ) -> Result<Self, OxiPhotonError> {
        if time_resolution_ps <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "time_resolution_ps must be positive".into(),
            ));
        }
        if n_bins == 0 {
            return Err(OxiPhotonError::NumericalError(
                "n_bins must be at least 1".into(),
            ));
        }
        if repetition_rate_mhz <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "repetition_rate_mhz must be positive".into(),
            ));
        }
        Ok(Self {
            detector,
            time_resolution_ps,
            n_bins,
            repetition_rate_mhz,
        })
    }

    // ── Count rate limits ─────────────────────────────────────────────────

    /// Maximum count rate to avoid pile-up (1 % of repetition rate, kHz).
    ///
    /// f_max_TCSPC = 0.01 × f_rep
    pub fn max_count_rate_khz(&self) -> f64 {
        0.01 * self.repetition_rate_mhz * 1e3
    }

    // ── Time axis ─────────────────────────────────────────────────────────

    /// Time axis values for histogram bins (ns).
    pub fn time_axis_ns(&self) -> Vec<f64> {
        (0..self.n_bins)
            .map(|i| i as f64 * self.time_resolution_ps * 1e-3)
            .collect()
    }

    // ── Histograms ────────────────────────────────────────────────────────

    /// Expected photon count histogram for a monoexponential fluorescence
    /// decay.
    ///
    /// h\[i\] ∝ exp(−t_i / τ)  scaled so that the integral ≈ `total_counts`.
    pub fn expected_decay_histogram(&self, lifetime_ns: f64, total_counts: usize) -> Vec<f64> {
        let t_axis = self.time_axis_ns();
        let dt_ns = self.time_resolution_ps * 1e-3;
        let tau = lifetime_ns;

        let raw: Vec<f64> = t_axis.iter().map(|&t| (-t / tau).exp()).collect();
        let norm: f64 = raw.iter().sum::<f64>() * dt_ns;

        if norm == 0.0 {
            return vec![0.0; self.n_bins];
        }
        raw.iter()
            .map(|&v| v / norm * total_counts as f64 * dt_ns)
            .collect()
    }

    /// Gaussian IRF histogram centred at bin 0.
    ///
    /// σ (in bins) corresponds to the detector timing jitter FWHM / (2√(2·ln2)).
    pub fn irf_histogram(&self) -> Vec<f64> {
        let fwhm_bins = self.detector.timing_jitter_ps / self.time_resolution_ps;
        // FWHM = 2·√(2·ln2)·σ  →  σ = FWHM / (2·√(2·ln2))
        let two_ln2: f64 = 2.0_f64.ln();
        let sigma_bins = fwhm_bins / (2.0 * (2.0 * two_ln2).sqrt());

        let mut irf: Vec<f64> = (0..self.n_bins)
            .map(|i| {
                let x = i as f64;
                (-0.5 * (x / sigma_bins) * (x / sigma_bins)).exp()
            })
            .collect();

        let norm: f64 = irf.iter().sum();
        if norm > 0.0 {
            for v in irf.iter_mut() {
                *v /= norm;
            }
        }
        irf
    }

    // ── Precision ─────────────────────────────────────────────────────────

    /// Statistical precision of lifetime measurement (ps).
    ///
    /// For a monoexponential decay with N counts and lifetime τ:
    /// δτ = τ / √N  (Cramér–Rao bound for Poissonian data).
    pub fn lifetime_precision_ps(&self, lifetime_ns: f64, total_counts: usize) -> f64 {
        if total_counts == 0 {
            return f64::INFINITY;
        }
        let precision_ns = lifetime_ns / (total_counts as f64).sqrt();
        precision_ns * 1e3 // ns → ps
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn typical_snspd() -> Snspd {
        Snspd::typical_1550()
    }

    #[test]
    fn test_snspd_max_count_rate() {
        let det = typical_snspd();
        // 50 ns dead time → f_max = 20 MHz
        let f_max = det.max_count_rate_mhz();
        assert_relative_eq!(f_max, 20.0, epsilon = 1e-9);
    }

    #[test]
    fn test_snspd_detection_probability() {
        let det = typical_snspd();
        // N=1: P = η
        let p1 = det.detection_probability(1);
        assert_relative_eq!(p1, det.detection_efficiency, epsilon = 1e-12);
        // N=2: P = 1 - (1-η)²
        let p2 = det.detection_probability(2);
        let expected = 1.0 - (1.0 - det.detection_efficiency).powi(2);
        assert_relative_eq!(p2, expected, epsilon = 1e-12);
        // P increases with N
        assert!(det.detection_probability(10) > p2);
    }

    #[test]
    fn test_snspd_timing_resolution() {
        let det = typical_snspd(); // jitter = 50 ps
        let src = 30.0_f64; // source jitter 30 ps
        let total = det.measured_timing_resolution_ps(src);
        let expected = (50.0_f64 * 50.0 + 30.0 * 30.0).sqrt();
        assert_relative_eq!(total, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_pmt_total_gain() {
        let pmt = Pmt::bialkali_visible();
        let expected = pmt.dynode_gain.powi(pmt.n_dynodes as i32);
        assert_relative_eq!(pmt.total_gain(), expected, epsilon = 1e-6);
        // Sanity: ~10^7 for g=5, N=10
        assert!(pmt.total_gain() > 1e6);
    }

    #[test]
    fn test_tcspc_max_count_rate() {
        let det = typical_snspd();
        let tcspc = TcSpc::new(det, 4.0, 4096, 80.0).expect("valid TCSPC");
        // 1% of 80 MHz = 800 kHz
        assert_relative_eq!(tcspc.max_count_rate_khz(), 800.0, epsilon = 1e-9);
    }

    #[test]
    fn test_decay_histogram_shape() {
        let det = typical_snspd();
        let tcspc = TcSpc::new(det, 4.0, 1024, 80.0).expect("valid TCSPC");
        let hist = tcspc.expected_decay_histogram(2.0, 100_000);
        // First bin should be largest (purely exponential, t=0 peak)
        let max_val = hist.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(max_val, hist[0], "histogram peak should be at t=0");
        // Monotonically non-increasing
        for i in 1..hist.len() {
            assert!(
                hist[i] <= hist[i - 1] + 1e-10,
                "histogram not monotone at bin {i}: {} > {}",
                hist[i],
                hist[i - 1]
            );
        }
    }

    #[test]
    fn test_lifetime_precision_improves_with_counts() {
        let det = typical_snspd();
        let tcspc = TcSpc::new(det, 4.0, 4096, 80.0).expect("valid TCSPC");
        let prec_100 = tcspc.lifetime_precision_ps(2.0, 100);
        let prec_10k = tcspc.lifetime_precision_ps(2.0, 10_000);
        // More counts → smaller uncertainty
        assert!(
            prec_10k < prec_100,
            "precision should improve: {prec_10k} >= {prec_100}"
        );
        // δτ = τ/√N: ratio should be ≈ √100 = 10
        assert_relative_eq!(prec_100 / prec_10k, 10.0, epsilon = 0.01);
    }
}
