//! Amplifier noise analysis: ASE, NF, Friis cascade, RIN, and phase noise.
//!
//! References:
//! - Saleh & Teich, "Fundamentals of Photonics", 3rd ed., §18.
//! - Agrawal, "Fiber-Optic Communication Systems", 6th ed., §7.
//! - Henry, "Theory of the linewidth of semiconductor lasers", IEEE JQE 18(2) 1982.
use std::f64::consts::PI;

/// Planck constant (J·s)
const H_PLANCK: f64 = 6.626_070_15e-34;
/// Speed of light (m/s)
const C_LIGHT: f64 = 2.997_924_58e8;

// ─── AmplifierNoiseAnalysis ───────────────────────────────────────────────────

/// Single-stage optical amplifier noise characterisation.
#[derive(Debug, Clone)]
pub struct AmplifierNoiseAnalysis {
    /// Amplifier gain (dB).
    pub gain_db: f64,
    /// Noise figure F (dB).
    pub noise_figure_db: f64,
    /// Optical noise bandwidth B_o (Hz).
    pub bandwidth_hz: f64,
}

impl AmplifierNoiseAnalysis {
    /// Construct a noise analysis for a single amplifier stage.
    pub fn new(gain_db: f64, nf_db: f64, bandwidth_hz: f64) -> Self {
        Self {
            gain_db,
            noise_figure_db: nf_db,
            bandwidth_hz,
        }
    }

    fn gain_linear(&self) -> f64 {
        10.0_f64.powf(self.gain_db / 10.0)
    }

    fn noise_figure_linear(&self) -> f64 {
        10.0_f64.powf(self.noise_figure_db / 10.0)
    }

    /// ASE power (dBm) added by this amplifier stage.
    ///
    /// P_ASE = (F·G - 1) · h·ν · B_o   (Watt)
    ///
    /// Factor (F·G - 1) comes from the noise figure definition:
    ///   F = (SNR_in / SNR_out) = 1/G + 2·n_sp·(G-1)/G ≈ 2·n_sp for large G.
    pub fn ase_power_dbm(&self, frequency_hz: f64) -> f64 {
        let g = self.gain_linear();
        let f = self.noise_figure_linear();
        let p_ase_w = (f * g - 1.0) * H_PLANCK * frequency_hz * self.bandwidth_hz;
        if p_ase_w <= 0.0 {
            return f64::NEG_INFINITY;
        }
        10.0 * (p_ase_w * 1e3).log10()
    }

    /// OSNR contribution of this stage for a given input signal power.
    ///
    /// OSNR = G · P_in / P_ASE.
    pub fn osnr_contribution_db(&self, input_power_dbm: f64) -> f64 {
        let p_in_mw = 10.0_f64.powf(input_power_dbm / 10.0);
        let p_out_mw = p_in_mw * self.gain_linear();
        let frequency_hz = C_LIGHT / 1550e-9; // assume C-band
        let p_ase_dbm = self.ase_power_dbm(frequency_hz);
        if p_ase_dbm == f64::NEG_INFINITY {
            return f64::INFINITY;
        }
        let p_ase_mw = 10.0_f64.powf(p_ase_dbm / 10.0);
        if p_ase_mw <= 0.0 {
            return f64::INFINITY;
        }
        10.0 * (p_out_mw / p_ase_mw).log10()
    }

    /// Electrical SNR (dB) after direct-detection photodiode, limited by
    /// signal-ASE beat noise.
    ///
    /// SNR_elec = P_sig² / (4 · P_sig · P_ASE · B_e/B_o + 2 · P_ASE² · B_e/B_o)
    /// Simplified dominant term (signal-ASE): SNR ≈ P_sig / (2 · P_ASE · B_e/B_o).
    pub fn electrical_snr_db(&self, signal_power_dbm: f64, detector_bandwidth_hz: f64) -> f64 {
        let frequency_hz = C_LIGHT / 1550e-9;
        let p_ase_dbm = self.ase_power_dbm(frequency_hz);
        if p_ase_dbm == f64::NEG_INFINITY {
            return f64::INFINITY;
        }
        let p_sig_mw = 10.0_f64.powf(signal_power_dbm / 10.0);
        let p_ase_mw = 10.0_f64.powf(p_ase_dbm / 10.0);
        let bw_ratio = detector_bandwidth_hz / self.bandwidth_hz;
        let snr = p_sig_mw / (2.0 * p_ase_mw * bw_ratio);
        if snr <= 0.0 {
            return f64::NEG_INFINITY;
        }
        10.0 * snr.log10()
    }
}

// ─── CascadedNoiseAnalysis ────────────────────────────────────────────────────

/// Cascaded amplifier chain noise analysis using the Friis formula.
///
/// Each stage is specified as (gain_db, noise_figure_db).
#[derive(Debug, Clone)]
pub struct CascadedNoiseAnalysis {
    /// Stages: (gain_db, noise_figure_db) per stage.
    pub stages: Vec<(f64, f64)>,
}

impl CascadedNoiseAnalysis {
    /// Construct a cascaded noise analysis from a list of (gain_db, NF_db) pairs.
    pub fn new(stages: Vec<(f64, f64)>) -> Self {
        Self { stages }
    }

    /// Total noise figure via Friis formula (linear):
    ///
    /// F_total = F₁ + (F₂-1)/G₁ + (F₃-1)/(G₁·G₂) + …
    pub fn total_noise_figure_db(&self) -> f64 {
        if self.stages.is_empty() {
            return 0.0;
        }
        let mut f_total = 0.0_f64;
        let mut cumulative_gain = 1.0_f64;
        for (i, &(g_db, nf_db)) in self.stages.iter().enumerate() {
            let f = 10.0_f64.powf(nf_db / 10.0);
            let g = 10.0_f64.powf(g_db / 10.0);
            if i == 0 {
                f_total = f;
            } else {
                f_total += (f - 1.0) / cumulative_gain;
            }
            cumulative_gain *= g;
        }
        10.0 * f_total.log10()
    }

    /// Total gain (dB): G_total = Σ G_i (sum in dB = product in linear).
    pub fn total_gain_db(&self) -> f64 {
        self.stages.iter().map(|&(g, _)| g).sum()
    }

    /// Index of the stage that contributes most to the total noise figure.
    ///
    /// Computed by evaluating the Friis contribution of each stage.
    pub fn dominant_stage_index(&self) -> usize {
        if self.stages.is_empty() {
            return 0;
        }
        let mut contributions = Vec::with_capacity(self.stages.len());
        let mut cumulative_gain = 1.0_f64;
        for (i, &(g_db, nf_db)) in self.stages.iter().enumerate() {
            let f = 10.0_f64.powf(nf_db / 10.0);
            let g = 10.0_f64.powf(g_db / 10.0);
            let contrib = if i == 0 {
                f
            } else {
                (f - 1.0) / cumulative_gain
            };
            contributions.push(contrib);
            cumulative_gain *= g;
        }
        contributions
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// OSNR (dB) at each stage output for a given launch power and reference bandwidth.
    ///
    /// Uses: OSNR_n = G_1·…·G_n · P_launch / Σ_{k=1}^{n} P_ASE_k_propagated
    /// where P_ASE_k_propagated = P_ASE_k · G_{k+1} · … · G_n.
    pub fn osnr_profile_db(&self, launch_power_dbm: f64, bandwidth_nm: f64) -> Vec<f64> {
        if self.stages.is_empty() {
            return Vec::new();
        }
        let frequency_hz = C_LIGHT / 1550e-9;
        let bw_hz = C_LIGHT / (1550e-9 * 1550e-9) * (bandwidth_nm * 1e-9);

        // Compute per-stage ASE power (mW) at stage output
        let stage_ase_mw: Vec<f64> = self
            .stages
            .iter()
            .map(|&(g_db, nf_db)| {
                let g = 10.0_f64.powf(g_db / 10.0);
                let f = 10.0_f64.powf(nf_db / 10.0);
                let p_ase_w = (f * g - 1.0) * H_PLANCK * frequency_hz * bw_hz;
                p_ase_w.max(0.0) * 1e3
            })
            .collect();

        // Gains from stage k onwards
        let n = self.stages.len();
        let mut result = Vec::with_capacity(n);

        // Cumulative gain up to stage n (linear)
        let mut cum_gain: Vec<f64> = vec![1.0; n + 1];
        for i in 0..n {
            let g = 10.0_f64.powf(self.stages[i].0 / 10.0);
            cum_gain[i + 1] = cum_gain[i] * g;
        }

        let p_launch_mw = 10.0_f64.powf(launch_power_dbm / 10.0);

        for stage_idx in 0..n {
            // Signal power at output of stage `stage_idx`
            let p_sig_mw = p_launch_mw * cum_gain[stage_idx + 1];
            // Accumulated ASE at output of stage `stage_idx`:
            // ASE from stage k is amplified by G_{k+1} * ... * G_{stage_idx}
            let p_ase_total_mw: f64 = (0..=stage_idx)
                .map(|k| {
                    let prop_gain = cum_gain[stage_idx + 1] / cum_gain[k + 1];
                    stage_ase_mw[k] * prop_gain
                })
                .sum();
            let osnr = if p_ase_total_mw > 0.0 {
                10.0 * (p_sig_mw / p_ase_total_mw).log10()
            } else {
                f64::INFINITY
            };
            result.push(osnr);
        }
        result
    }
}

// ─── RinAnalysis ─────────────────────────────────────────────────────────────

/// Relative Intensity Noise (RIN) analysis for laser sources and amplifiers.
///
/// RIN(f) = S_P(f) / P² where S_P(f) is the one-sided power spectral density
/// of power fluctuations.
#[derive(Debug, Clone)]
pub struct RinAnalysis {
    /// Source RIN (dBc/Hz); typical -150 to -160 dBc/Hz for DFB lasers.
    pub source_rin_db_per_hz: f64,
    /// Amplifier gain (dB).
    pub amplifier_gain_db: f64,
}

impl RinAnalysis {
    /// Construct a RIN analysis instance.
    pub fn new(rin_db_hz: f64, gain_db: f64) -> Self {
        Self {
            source_rin_db_per_hz: rin_db_hz,
            amplifier_gain_db: gain_db,
        }
    }

    /// RIN of the amplified signal (dBc/Hz).
    ///
    /// For small-signal amplification, RIN is preserved: RIN_out ≈ RIN_in.
    /// ASE from the amplifier adds to intensity noise, but for high OSNR
    /// the dominant contribution is the source RIN.
    pub fn amplified_rin_db_per_hz(&self) -> f64 {
        // RIN is preserved through linear amplification
        self.source_rin_db_per_hz
    }

    /// SNR limited by RIN over detection bandwidth B (Hz).
    ///
    /// SNR_RIN = 1 / (RIN_linear · B)
    pub fn snr_from_rin_db(&self, bandwidth_hz: f64) -> f64 {
        let rin_linear = 10.0_f64.powf(self.source_rin_db_per_hz / 10.0);
        let snr = 1.0 / (rin_linear * bandwidth_hz);
        if snr <= 0.0 {
            return f64::NEG_INFINITY;
        }
        10.0 * snr.log10()
    }

    /// RIN-limited OSNR (dBc) in bandwidth B (Hz).
    ///
    /// Defined as the equivalent OSNR at which shot-noise SNR = RIN SNR.
    pub fn rin_limited_osnr_db(&self, bandwidth_hz: f64) -> f64 {
        self.snr_from_rin_db(bandwidth_hz)
    }
}

// ─── LaserLinewidth ───────────────────────────────────────────────────────────

/// Laser phase noise and linewidth model based on the Schawlow-Townes
/// (modified by Henry's α-factor) formalism.
#[derive(Debug, Clone)]
pub struct LaserLinewidth {
    /// Intrinsic (Schawlow-Townes) linewidth Δν_ST (Hz).
    pub intrinsic_linewidth_hz: f64,
    /// Linewidth enhancement (Henry α) factor.
    pub alpha_factor: f64,
    /// Output power (mW).
    pub output_power_mw: f64,
}

impl LaserLinewidth {
    /// Construct a laser linewidth model.
    pub fn new(st_linewidth_hz: f64, alpha: f64, power_mw: f64) -> Self {
        Self {
            intrinsic_linewidth_hz: st_linewidth_hz,
            alpha_factor: alpha,
            output_power_mw: power_mw,
        }
    }

    /// Full (modified Schawlow-Townes) linewidth (Hz):
    ///
    /// Δν = Δν_ST · (1 + α²)
    pub fn full_linewidth_hz(&self) -> f64 {
        self.intrinsic_linewidth_hz * (1.0 + self.alpha_factor * self.alpha_factor)
    }

    /// Single-sided phase noise PSD S_φ(f) (dBc/Hz) at frequency offset f.
    ///
    /// For a Lorentzian lineshape:
    ///   S_φ(f) = Δν_full / (π · f²)   [rad²/Hz]
    /// Converted to dBc/Hz.
    pub fn phase_noise_psd_dbc_hz(&self, offset_freq_hz: f64) -> f64 {
        if offset_freq_hz <= 0.0 {
            return f64::INFINITY;
        }
        let delta_nu = self.full_linewidth_hz();
        let s_phi = delta_nu / (PI * offset_freq_hz * offset_freq_hz);
        10.0 * s_phi.log10()
    }

    /// Coherence length L_c = c / (π · Δν_full) (m).
    pub fn coherence_length_m(&self) -> f64 {
        let delta_nu = self.full_linewidth_hz();
        if delta_nu <= 0.0 {
            return f64::INFINITY;
        }
        C_LIGHT / (PI * delta_nu)
    }

    /// Coherence time τ_c = 1 / (π · Δν_full) (s).
    pub fn coherence_time_s(&self) -> f64 {
        let delta_nu = self.full_linewidth_hz();
        if delta_nu <= 0.0 {
            return f64::INFINITY;
        }
        1.0 / (PI * delta_nu)
    }
}

// ─── Utility free functions ───────────────────────────────────────────────────

/// Convert optical bandwidth from nm to Hz at centre wavelength λ₀ (nm).
pub fn bandwidth_nm_to_hz(bandwidth_nm: f64, center_wavelength_nm: f64) -> f64 {
    let lambda = center_wavelength_nm * 1e-9;
    C_LIGHT / (lambda * lambda) * (bandwidth_nm * 1e-9)
}

/// Convert noise figure (dB) and gain (dB) to ASE spectral density n_sp.
///
/// n_sp = F_linear / 2 for large G limit.
pub fn nf_to_nsp(nf_db: f64) -> f64 {
    let f_lin = 10.0_f64.powf(nf_db / 10.0);
    f_lin / 2.0
}

/// Effective noise figure of an attenuator (loss L_dB > 0) at temperature T.
///
/// For a passive attenuator: F_att = L (linear) — purely thermal noise.
pub fn attenuator_noise_figure_db(loss_db: f64) -> f64 {
    // NF of attenuator = loss (in linear = 1/transmission)
    loss_db
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_ase_power_positive() {
        let ana = AmplifierNoiseAnalysis::new(20.0, 5.0, 12.5e9);
        let p_ase = ana.ase_power_dbm(C_LIGHT / 1550e-9);
        assert!(p_ase.is_finite(), "ASE power must be finite; got {p_ase}");
        // Should be a negative dBm value (very small power)
        assert!(p_ase < 0.0, "ASE power must be below 0 dBm; got {p_ase}");
    }

    #[test]
    fn test_friis_first_stage_dominates() {
        // First stage: 20 dB gain, 5 dB NF → dominates
        // Second stage: 20 dB gain, 8 dB NF → divided by G1
        let cascade = CascadedNoiseAnalysis::new(vec![(20.0, 5.0), (20.0, 8.0)]);
        let dominant = cascade.dominant_stage_index();
        assert_eq!(dominant, 0, "First stage should dominate for equal gains");
    }

    #[test]
    fn test_friis_total_nf_greater_than_first_stage() {
        let cascade = CascadedNoiseAnalysis::new(vec![(20.0, 4.0), (20.0, 6.0)]);
        let nf_total = cascade.total_noise_figure_db();
        // Must be ≥ 4 dB (first stage NF)
        assert!(
            nf_total >= 4.0,
            "Total NF must be ≥ first stage NF; got {nf_total}"
        );
    }

    #[test]
    fn test_total_gain_is_sum() {
        let cascade = CascadedNoiseAnalysis::new(vec![(10.0, 4.0), (15.0, 5.0), (8.0, 4.5)]);
        let total = cascade.total_gain_db();
        assert_abs_diff_eq!(total, 33.0, epsilon = 1e-10);
    }

    #[test]
    fn test_osnr_profile_decreasing() {
        // OSNR should decrease through a lossy cascade
        let cascade = CascadedNoiseAnalysis::new(vec![(20.0, 5.0), (20.0, 5.0), (20.0, 5.0)]);
        let profile = cascade.osnr_profile_db(0.0, 0.1);
        assert_eq!(
            profile.len(),
            3,
            "Profile length must match number of stages"
        );
        // Each successive stage should reduce OSNR
        assert!(
            profile[0] > profile[2],
            "OSNR must decrease through cascade; stage 0={:.1}, stage 2={:.1}",
            profile[0],
            profile[2]
        );
    }

    #[test]
    fn test_rin_snr_decreases_with_bandwidth() {
        let rin = RinAnalysis::new(-155.0, 20.0);
        let snr_narrow = rin.snr_from_rin_db(1e9);
        let snr_wide = rin.snr_from_rin_db(10e9);
        assert!(
            snr_narrow > snr_wide,
            "RIN SNR must decrease with bandwidth"
        );
    }

    #[test]
    fn test_laser_linewidth_modified_schawlow_townes() {
        // α = 0 → full linewidth = intrinsic linewidth
        let laser = LaserLinewidth::new(100e3, 0.0, 1.0);
        assert_abs_diff_eq!(laser.full_linewidth_hz(), 100e3, epsilon = 1.0);
        // α = 1 → full linewidth = 2 * intrinsic
        let laser2 = LaserLinewidth::new(100e3, 1.0, 1.0);
        assert_abs_diff_eq!(laser2.full_linewidth_hz(), 200e3, epsilon = 1.0);
    }

    #[test]
    fn test_coherence_length_inversely_proportional_to_linewidth() {
        let laser_narrow = LaserLinewidth::new(10e3, 3.0, 1.0);
        let laser_wide = LaserLinewidth::new(100e3, 3.0, 1.0);
        assert!(
            laser_narrow.coherence_length_m() > laser_wide.coherence_length_m(),
            "Narrower linewidth must give longer coherence length"
        );
    }

    #[test]
    fn test_bandwidth_nm_to_hz_c_band() {
        // 0.1 nm at 1550 nm ≈ 12.5 GHz
        let bw_hz = bandwidth_nm_to_hz(0.1, 1550.0);
        assert!(
            bw_hz > 10e9 && bw_hz < 15e9,
            "0.1 nm BW at 1550 nm should be ~12.5 GHz; got {:.3} GHz",
            bw_hz * 1e-9
        );
    }

    #[test]
    fn test_phase_noise_decreases_with_offset_frequency() {
        let laser = LaserLinewidth::new(100e3, 5.0, 10.0);
        let s1 = laser.phase_noise_psd_dbc_hz(1e6);
        let s10 = laser.phase_noise_psd_dbc_hz(10e6);
        assert!(
            s1 > s10,
            "Phase noise PSD must decrease with offset frequency (1/f² slope)"
        );
    }
}
