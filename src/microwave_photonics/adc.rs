/// Photonic analog-to-digital conversion (ADC) and channelized receiver models.
///
/// Covers:
/// - Photonic-assisted ADC using optical pulse sampling/stretching
/// - SNR, timing-jitter limits, and WDM interleaving analysis
/// - Photonic channelized receivers for instantaneous frequency measurement
use std::f64::consts::PI;

// ─── PhotonicAdc ──────────────────────────────────────────────────────────────

/// Photonic-assisted analog-to-digital converter.
///
/// Uses mode-locked laser pulses as a low-jitter sampling clock, enabling
/// ADC performance far beyond what electronic clocks can achieve at multi-GHz
/// sample rates.
#[derive(Debug, Clone)]
pub struct PhotonicAdc {
    /// Sampling rate \[Gsamples/s\].
    pub sampling_rate_gsps: f64,
    /// Effective number of bits (ENOB).
    pub bits: u32,
    /// Optical bandwidth \[GHz\] (limits the analog input bandwidth).
    pub optical_bandwidth_ghz: f64,
    /// Modulator half-wave voltage Vπ \[V\].
    pub modulator_vpi: f64,
}

impl PhotonicAdc {
    /// Create a photonic ADC with default modulator Vπ = 5 V.
    ///
    /// # Arguments
    /// * `rate_gsps` – sampling rate \[Gsamples/s\]
    /// * `enob` – effective number of bits
    /// * `bandwidth_ghz` – optical/RF input bandwidth \[GHz\]
    pub fn new(rate_gsps: f64, enob: u32, bandwidth_ghz: f64) -> Self {
        PhotonicAdc {
            sampling_rate_gsps: rate_gsps,
            bits: enob,
            optical_bandwidth_ghz: bandwidth_ghz,
            modulator_vpi: 5.0,
        }
    }

    /// Create a photonic ADC with an explicit modulator Vπ.
    pub fn new_with_vpi(rate_gsps: f64, enob: u32, bandwidth_ghz: f64, vpi: f64) -> Self {
        PhotonicAdc {
            sampling_rate_gsps: rate_gsps,
            bits: enob,
            optical_bandwidth_ghz: bandwidth_ghz,
            modulator_vpi: vpi,
        }
    }

    /// Theoretical SFDR from ENOB: SFDR = 6.02 · ENOB + 1.76 \[dB\].
    pub fn sfdr_db(&self) -> f64 {
        6.02 * self.bits as f64 + 1.76
    }

    /// Theoretical peak SNR from ENOB: SNR = 6.02 · ENOB + 1.76 \[dB\].
    ///
    /// This is the standard formula for an ideal sinusoidal input at −3 dBFS.
    pub fn snr_theoretical_db(&self) -> f64 {
        6.02 * self.bits as f64 + 1.76
    }

    /// SNR limited by timing jitter at a given RF input frequency \[dB\].
    ///
    /// Aperture jitter σt degrades SNR as:
    ///   SNR_jitter = −20 · log₁₀(2π · f_RF · σ_t)
    ///
    /// # Arguments
    /// * `rf_freq_ghz` – RF carrier frequency \[GHz\]
    /// * `jitter_fs` – RMS timing jitter of the optical clock \[femtoseconds\]
    pub fn timing_jitter_snr_db(&self, rf_freq_ghz: f64, jitter_fs: f64) -> f64 {
        let f_hz = rf_freq_ghz * 1.0e9;
        let sigma_t = jitter_fs * 1.0e-15;
        let arg = 2.0 * PI * f_hz * sigma_t;
        if arg <= 0.0 {
            return f64::INFINITY;
        }
        -20.0 * arg.log10()
    }

    /// Effective maximum RF input frequency given timing jitter and required SNR \[GHz\].
    ///
    ///   f_max = 1 / (2π · σ_t · 10^(SNR_req/20))
    ///
    /// # Arguments
    /// * `jitter_fs` – RMS timing jitter \[femtoseconds\]
    /// * `snr_req_db` – required SNR \[dB\]
    pub fn max_frequency_ghz(&self, jitter_fs: f64, snr_req_db: f64) -> f64 {
        let sigma_t = jitter_fs * 1.0e-15;
        let snr_linear = 10.0_f64.powf(snr_req_db / 20.0);
        if sigma_t <= 0.0 || snr_linear <= 0.0 {
            return f64::INFINITY;
        }
        let f_hz = 1.0 / (2.0 * PI * sigma_t * snr_linear);
        f_hz * 1.0e-9 // convert to GHz
    }

    /// Effective sampling rate with WDM-parallel channel interleaving \[Gsamples/s\].
    ///
    /// Using N WDM channels, each running at the single-channel rate, the
    /// effective Nyquist bandwidth is multiplied by N.
    ///
    /// # Arguments
    /// * `n_channels` – number of WDM interleaved channels
    pub fn interleaved_rate(&self, n_channels: usize) -> f64 {
        self.sampling_rate_gsps * n_channels as f64
    }

    /// Nyquist bandwidth \[GHz\] = sampling_rate / 2.
    pub fn nyquist_bandwidth_ghz(&self) -> f64 {
        self.sampling_rate_gsps / 2.0
    }

    /// Spurious free dynamic range limited by the modulator nonlinearity \[dB\].
    ///
    /// For an MZM, the nonlinear transfer function introduces harmonic
    /// distortion. The harmonic-limited SFDR is:
    ///   SFDR_HD3 = (2/3) · (OIP3 / noise_floor)
    /// Approximated here from the modulator Vπ and optical power.
    ///
    /// # Arguments
    /// * `optical_power_dbm` – optical power at the modulator input \[dBm\]
    pub fn modulator_sfdr_db(&self, optical_power_dbm: f64) -> f64 {
        // Simplified: for MZM at quadrature, OIP3 scales with (2Vpi/pi)^2
        // Noise floor set at -174 dBm/Hz + NF
        let oip3_proxy = 10.0 * (self.modulator_vpi * 2.0 / PI).log10() + optical_power_dbm;
        let noise_floor_dbm_hz = -174.0 + 5.0; // assume 5 dB NF
        (2.0 / 3.0) * (oip3_proxy - noise_floor_dbm_hz)
    }

    /// Effective ENOB achievable at a given RF frequency and timing jitter.
    ///
    ///   ENOB_jitter = (SNR_jitter − 1.76) / 6.02
    ///
    /// # Arguments
    /// * `rf_freq_ghz` – RF carrier frequency \[GHz\]
    /// * `jitter_fs` – timing jitter \[femtoseconds\]
    pub fn enob_at_frequency(&self, rf_freq_ghz: f64, jitter_fs: f64) -> f64 {
        let snr = self.timing_jitter_snr_db(rf_freq_ghz, jitter_fs);
        // Limit to the architectural ENOB
        let enob_jitter = (snr - 1.76) / 6.02;
        enob_jitter.min(self.bits as f64)
    }
}

// ─── PhotonicChannelizer ──────────────────────────────────────────────────────

/// Photonic channelized receiver that splits a wide RF spectrum into N sub-bands.
///
/// Uses WDM optical filtering and multiple photodetectors to achieve
/// channelization of wide-bandwidth RF signals with fine frequency resolution.
#[derive(Debug, Clone)]
pub struct PhotonicChannelizer {
    /// Number of frequency channels.
    pub n_channels: usize,
    /// Per-channel bandwidth \[GHz\].
    pub channel_bandwidth_ghz: f64,
    /// Total instantaneous bandwidth \[GHz\].
    pub total_bandwidth_ghz: f64,
    /// Adjacent-channel isolation \[dB\].
    pub channel_isolation_db: f64,
}

impl PhotonicChannelizer {
    /// Create a uniform photonic channelizer.
    ///
    /// The per-channel bandwidth is automatically set to
    /// `total_bw_ghz / n_channels` with 20 dB channel isolation.
    ///
    /// # Arguments
    /// * `n_channels` – number of channels
    /// * `total_bw_ghz` – total RF bandwidth to cover \[GHz\]
    pub fn new(n_channels: usize, total_bw_ghz: f64) -> Self {
        let channel_bw = if n_channels > 0 {
            total_bw_ghz / n_channels as f64
        } else {
            0.0
        };
        PhotonicChannelizer {
            n_channels,
            channel_bandwidth_ghz: channel_bw,
            total_bandwidth_ghz: total_bw_ghz,
            channel_isolation_db: 20.0,
        }
    }

    /// Create a channelizer with specified per-channel bandwidth and overlap factor.
    ///
    /// An overlap factor > 1 means adjacent channels share some bandwidth
    /// (useful for reconstructing signals near channel edges).
    pub fn new_with_overlap(n_channels: usize, channel_bw_ghz: f64, overlap: f64) -> Self {
        let effective_bw_per_ch = channel_bw_ghz / overlap.max(1.0);
        let total_bw = n_channels as f64 * effective_bw_per_ch;
        PhotonicChannelizer {
            n_channels,
            channel_bandwidth_ghz: channel_bw_ghz,
            total_bandwidth_ghz: total_bw,
            channel_isolation_db: 20.0,
        }
    }

    /// Center frequency of channel `ch` (0-indexed) \[GHz\].
    ///
    /// Channels are uniformly spaced starting from `channel_bw / 2`.
    pub fn channel_center_freq(&self, channel: usize) -> f64 {
        let spacing = if self.n_channels > 0 {
            self.total_bandwidth_ghz / self.n_channels as f64
        } else {
            0.0
        };
        (channel as f64 + 0.5) * spacing
    }

    /// Per-channel 3-dB bandwidth \[GHz\].
    pub fn channel_bandwidth_ghz(&self) -> f64 {
        self.channel_bandwidth_ghz
    }

    /// Instantaneous bandwidth (IBW): total RF bandwidth covered \[GHz\].
    pub fn instantaneous_bandwidth_ghz(&self) -> f64 {
        self.total_bandwidth_ghz
    }

    /// Frequency resolution (minimum detectable frequency difference) \[GHz\].
    ///
    /// For a channelizer with N channels over BW_total:
    ///   Δf = BW_total / N = channel_bw
    pub fn frequency_resolution_ghz(&self) -> f64 {
        if self.n_channels > 0 {
            self.total_bandwidth_ghz / self.n_channels as f64
        } else {
            0.0
        }
    }

    /// Number of channels covering a given RF frequency \[GHz\].
    ///
    /// Returns the channel index (0-indexed), or None if out of range.
    pub fn channel_for_frequency(&self, freq_ghz: f64) -> Option<usize> {
        if freq_ghz < 0.0 || freq_ghz > self.total_bandwidth_ghz {
            return None;
        }
        let spacing = self.total_bandwidth_ghz / self.n_channels as f64;
        let ch = (freq_ghz / spacing).floor() as usize;
        if ch < self.n_channels {
            Some(ch)
        } else {
            Some(self.n_channels - 1)
        }
    }

    /// Dynamic range of the channelizer \[dB\].
    ///
    /// Dominated by the channel isolation and spurious-free range of the
    /// sub-channel ADC. This returns the isolation-limited estimate.
    pub fn dynamic_range_db(&self) -> f64 {
        self.channel_isolation_db
    }

    /// Minimum pulse width detectable given the channel bandwidth \[ns\].
    ///
    ///   τ_min ≈ 1 / BW_channel
    pub fn min_pulse_width_ns(&self) -> f64 {
        if self.channel_bandwidth_ghz > 0.0 {
            1.0 / self.channel_bandwidth_ghz
        } else {
            f64::INFINITY
        }
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn adc_8bit() -> PhotonicAdc {
        PhotonicAdc::new(40.0, 8, 20.0)
    }

    // ── PhotonicAdc ─────────────────────────────────────────────────────────

    #[test]
    fn test_sfdr_8bit() {
        let adc = adc_8bit();
        // 6.02 * 8 + 1.76 = 49.92 dB
        assert_abs_diff_eq!(adc.sfdr_db(), 49.92, epsilon = 0.01);
    }

    #[test]
    fn test_snr_theoretical_8bit() {
        let adc = adc_8bit();
        assert_abs_diff_eq!(adc.snr_theoretical_db(), 49.92, epsilon = 0.01);
    }

    #[test]
    fn test_timing_jitter_snr_decreases_with_frequency() {
        let adc = adc_8bit();
        let snr_1ghz = adc.timing_jitter_snr_db(1.0, 100.0);
        let snr_10ghz = adc.timing_jitter_snr_db(10.0, 100.0);
        assert!(
            snr_10ghz < snr_1ghz,
            "SNR should decrease at higher frequency"
        );
    }

    #[test]
    fn test_timing_jitter_snr_formula() {
        let adc = adc_8bit();
        // SNR = -20 log10(2π * 1e9 * 100e-15) = -20 log10(2π * 1e-4)
        let expected = -20.0 * (2.0 * PI * 1.0e9 * 100.0e-15).log10();
        let computed = adc.timing_jitter_snr_db(1.0, 100.0);
        assert_abs_diff_eq!(computed, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_max_frequency_ghz() {
        let adc = adc_8bit();
        // Higher SNR requirement → lower max frequency
        let f_low_snr = adc.max_frequency_ghz(100.0, 30.0);
        let f_high_snr = adc.max_frequency_ghz(100.0, 50.0);
        assert!(
            f_high_snr < f_low_snr,
            "Higher SNR req → lower max frequency"
        );
    }

    #[test]
    fn test_interleaved_rate() {
        let adc = adc_8bit();
        let rate = adc.interleaved_rate(4);
        assert_abs_diff_eq!(rate, 160.0, epsilon = 1e-6);
    }

    #[test]
    fn test_nyquist_bandwidth() {
        let adc = PhotonicAdc::new(20.0, 10, 10.0);
        assert_abs_diff_eq!(adc.nyquist_bandwidth_ghz(), 10.0, epsilon = 1e-9);
    }

    #[test]
    fn test_enob_at_frequency_capped_by_bits() {
        let adc = PhotonicAdc::new(40.0, 6, 20.0);
        // Very low jitter → jitter-limited ENOB would exceed architectural ENOB
        let enob = adc.enob_at_frequency(1.0, 0.1); // 0.1 fs jitter
        assert!(
            enob <= adc.bits as f64,
            "ENOB must not exceed architectural bits"
        );
    }

    #[test]
    fn test_enob_decreases_with_jitter() {
        let adc = adc_8bit();
        let enob_low = adc.enob_at_frequency(5.0, 10.0);
        let enob_high = adc.enob_at_frequency(5.0, 1000.0);
        assert!(
            enob_low > enob_high,
            "ENOB should be higher with less jitter"
        );
    }

    // ── PhotonicChannelizer ──────────────────────────────────────────────────

    #[test]
    fn test_channelizer_channel_bw() {
        let ch = PhotonicChannelizer::new(16, 40.0);
        assert_abs_diff_eq!(ch.channel_bandwidth_ghz(), 2.5, epsilon = 1e-9);
    }

    #[test]
    fn test_channelizer_center_freqs() {
        let ch = PhotonicChannelizer::new(4, 40.0);
        // Channels: [0–10], [10–20], [20–30], [30–40] GHz
        // Centers: 5, 15, 25, 35 GHz
        assert_abs_diff_eq!(ch.channel_center_freq(0), 5.0, epsilon = 1e-9);
        assert_abs_diff_eq!(ch.channel_center_freq(1), 15.0, epsilon = 1e-9);
        assert_abs_diff_eq!(ch.channel_center_freq(3), 35.0, epsilon = 1e-9);
    }

    #[test]
    fn test_channelizer_frequency_resolution() {
        let ch = PhotonicChannelizer::new(8, 16.0);
        assert_abs_diff_eq!(ch.frequency_resolution_ghz(), 2.0, epsilon = 1e-9);
    }

    #[test]
    fn test_channelizer_channel_lookup() {
        let ch = PhotonicChannelizer::new(4, 40.0);
        assert_eq!(ch.channel_for_frequency(5.0), Some(0));
        assert_eq!(ch.channel_for_frequency(25.0), Some(2));
        assert_eq!(ch.channel_for_frequency(50.0), None);
    }

    #[test]
    fn test_channelizer_min_pulse_width() {
        let ch = PhotonicChannelizer::new(10, 10.0);
        // BW per channel = 1 GHz → min pulse width = 1 ns
        assert_abs_diff_eq!(ch.min_pulse_width_ns(), 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_channelizer_instantaneous_bandwidth() {
        let ch = PhotonicChannelizer::new(8, 40.0);
        assert_abs_diff_eq!(ch.instantaneous_bandwidth_ghz(), 40.0, epsilon = 1e-9);
    }
}
