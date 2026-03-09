//! Mode Division Multiplexing (MDM) System-Level Analysis.
//!
//! This module provides system-level capacity analysis, spectral efficiency
//! calculations, MIMO complexity estimation, and SDM gain comparisons for
//! MDM-based fiber-optic transmission systems.
//!
//! # Key formulas
//! - Total capacity: C = N_modes × N_wl × B × SE
//! - Shannon capacity: C = N_modes × B × log₂(1 + SNR / N_modes)
//! - MIMO complexity: O(N_modes² × N_taps × B_sym)
//!
//! # References
//! - Essiambre et al., "Capacity Limits of Optical Fiber Networks", JLT 2010
//! - Li et al., "Space-Division Multiplexing: The Next Frontier", OFC 2014

// ── Modulation Format ─────────────────────────────────────────────────────────

/// Optical modulation format used in each MDM channel.
#[derive(Debug, Clone, PartialEq)]
pub enum MdmModFormat {
    /// Dual-polarisation QPSK (DP-QPSK): 4 bits/symbol, SE ≈ 4 b/s/Hz.
    DpQpsk,
    /// Dual-polarisation 16-QAM (DP-16QAM): 8 bits/symbol, SE ≈ 8 b/s/Hz.
    Dp16Qam,
    /// Dual-polarisation 64-QAM (DP-64QAM): 12 bits/symbol, SE ≈ 12 b/s/Hz.
    Dp64Qam,
}

impl MdmModFormat {
    /// Bits per symbol (both polarisations combined).
    pub fn bits_per_symbol(&self) -> u32 {
        match self {
            MdmModFormat::DpQpsk => 4,
            MdmModFormat::Dp16Qam => 8,
            MdmModFormat::Dp64Qam => 12,
        }
    }

    /// Theoretical spectral efficiency \[b/s/Hz\] (with Nyquist shaping, no FEC overhead).
    pub fn spectral_efficiency(&self) -> f64 {
        match self {
            MdmModFormat::DpQpsk => 4.0,
            MdmModFormat::Dp16Qam => 8.0,
            MdmModFormat::Dp64Qam => 12.0,
        }
    }

    /// Required OSNR \[dB\] for BER = 10⁻³ (pre-FEC threshold).
    pub fn required_osnr_db(&self) -> f64 {
        match self {
            MdmModFormat::DpQpsk => 12.0,
            MdmModFormat::Dp16Qam => 18.5,
            MdmModFormat::Dp64Qam => 25.0,
        }
    }

    /// FEC overhead fraction (ITU-T G.975.1 standard codes).
    pub fn fec_overhead(&self) -> f64 {
        match self {
            MdmModFormat::DpQpsk => 0.07,  // BFEC 7%
            MdmModFormat::Dp16Qam => 0.20, // SD-FEC 20%
            MdmModFormat::Dp64Qam => 0.25, // SD-FEC 25%
        }
    }

    /// Net spectral efficiency after FEC overhead \[b/s/Hz\].
    pub fn net_spectral_efficiency(&self) -> f64 {
        self.spectral_efficiency() / (1.0 + self.fec_overhead())
    }
}

// ── MDM Transmission System ───────────────────────────────────────────────────

/// Complete MDM transmission system model.
///
/// Combines a few-mode fiber with WDM channel plan, spatial multiplexing,
/// and digital coherent detection with MIMO equalization.
pub struct MdmSystem {
    /// Few-mode fiber link.
    pub fiber: super::few_mode_fiber::FewModeFiber,
    /// Number of spatial modes used (may be ≤ fiber.n_modes).
    pub n_spatial_modes: usize,
    /// Number of WDM wavelength channels.
    pub n_wavelength_channels: usize,
    /// Symbol rate per channel \[GBaud\].
    pub baud_rate_gbaud: f64,
    /// Modulation format.
    pub modulation_format: MdmModFormat,
}

impl MdmSystem {
    /// Construct a new MDM system.
    pub fn new(
        fiber: super::few_mode_fiber::FewModeFiber,
        n_modes: usize,
        n_wl: usize,
        baud: f64,
        format: MdmModFormat,
    ) -> Self {
        Self {
            fiber,
            n_spatial_modes: n_modes,
            n_wavelength_channels: n_wl,
            baud_rate_gbaud: baud,
            modulation_format: format,
        }
    }

    /// Total raw capacity \[Tb/s\]:
    ///   C = n_modes × n_wl × baud × SE
    pub fn total_capacity_tbps(&self) -> f64 {
        let se = self.modulation_format.spectral_efficiency();
        self.n_spatial_modes as f64
            * self.n_wavelength_channels as f64
            * self.baud_rate_gbaud
            * se
            * 1.0e-3 // GBaud × b/s/Hz → Gb/s per mode per WL; ×1e9/1e12 = 1e-3
    }

    /// Spectral efficiency per spatial mode \[b/s/Hz\].
    pub fn spectral_efficiency_per_mode(&self) -> f64 {
        self.modulation_format.spectral_efficiency()
    }

    /// Aggregate spectral efficiency across all spatial modes \[b/s/Hz\].
    pub fn aggregate_se_bps_per_hz(&self) -> f64 {
        self.n_spatial_modes as f64 * self.modulation_format.spectral_efficiency()
    }

    /// MIMO computational complexity \[TOPS = 10¹² operations/s\].
    ///
    ///   TOPS = n_modes² × n_taps × baud \[GBaud\] × 10⁹ / 10¹²
    pub fn mimo_complexity_tops(&self, n_taps: usize) -> f64 {
        let n = self.n_spatial_modes as f64;
        let ops_per_sym = n * n * n_taps as f64;
        ops_per_sym * self.baud_rate_gbaud * 1.0e9 / 1.0e12
    }

    /// Shannon-limit capacity \[Tb/s\] with MIMO equalization.
    ///
    ///   C = n_modes × B × log₂(1 + SNR / n_modes)
    ///
    /// The SNR/n_modes factor accounts for noise loading across spatial modes.
    pub fn shannon_capacity_tbps(&self, snr_db: f64, bandwidth_thz: f64) -> f64 {
        let snr = 10.0_f64.powf(snr_db / 10.0);
        let n = self.n_spatial_modes as f64;
        let bw_hz = bandwidth_thz * 1.0e12;
        let capacity_bps = n * bw_hz * (1.0 + snr / n).log2();
        capacity_bps * 1.0e-12 // → Tb/s
    }

    /// Mode-dependent loss (MDL) penalty on effective SNR \[dB\].
    ///
    /// Estimated from the variance of per-mode loss across the fiber.
    /// MDL_penalty ≈ σ_loss² / (2 · ln2)  (Gaussian approximation)
    pub fn mdl_penalty_db(&self) -> f64 {
        let losses = &self.fiber.loss_db_per_km;
        let n = losses.len() as f64;
        if n < 2.0 {
            return 0.0;
        }
        let mean = losses.iter().sum::<f64>() / n;
        let variance = losses.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / n;
        let total_loss_spread = (variance.sqrt() * self.fiber.length_km).powi(2);
        total_loss_spread / (2.0 * 2.0_f64.ln())
    }

    /// Estimated system reach \[km\] at a given target capacity and SNR.
    ///
    /// Uses a simplified EDFA-amplified link model:
    ///   SNR_system = P_launch − NF − 10·log10(n_spans·h·f·Δf) − span_loss
    ///
    /// Reach is iterated until capacity drops below target.
    pub fn reach_km(&self, target_tbps: f64, snr_db: f64) -> f64 {
        // Binary search over span count
        let span_length_km = 80.0_f64;
        let max_spans = 100usize;
        let bandwidth_thz = self.n_wavelength_channels as f64 * self.baud_rate_gbaud * 1.0e-3; // GHz → THz
        let mut lo = 0.0_f64;
        let mut hi = max_spans as f64 * span_length_km;
        // Simple model: SNR degrades by 0.3 dB per 100 km
        for _ in 0..50 {
            let mid = (lo + hi) / 2.0;
            let snr_degraded = snr_db - 0.3 * mid / 100.0;
            let cap = if snr_degraded > 0.0 {
                self.shannon_capacity_tbps(snr_degraded, bandwidth_thz)
            } else {
                0.0
            };
            if cap >= target_tbps {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (lo + hi) / 2.0
    }
}

// ── Mode Group Demultiplexing ─────────────────────────────────────────────────

/// Mode Group Demultiplexing (MGDM) — operates on mode groups rather than
/// individual LP modes to reduce MIMO complexity.
///
/// Mode groups in a step-index fiber:
///   Group 1: LP01 (1 spatial mode × 2 pols = 2 modes)
///   Group 2: LP11 (2 spatial × 2 pols = 4 modes)
///   Group 3: LP21 + LP02 (3 spatial × 2 pols = 6 modes)
pub struct ModegroupDemux {
    /// Mode indices belonging to each group: `[[0], [1,2], [3,4,5], ...]`
    pub mode_groups: Vec<Vec<usize>>,
    /// Intra-group crosstalk (within the same mode group) \[dB\].
    pub crosstalk_within_group_db: f64,
    /// Inter-group crosstalk (between different mode groups) \[dB\].
    pub crosstalk_between_groups_db: f64,
}

impl ModegroupDemux {
    /// Build a 3-group MGDM demultiplexer for the given FMF.
    ///
    /// Groups are assigned by mode order: LP01, LP11(×2), LP21+LP02(×3).
    pub fn new_3group(fiber: &super::few_mode_fiber::FewModeFiber) -> Self {
        let n = fiber.n_modes;
        // Group 0: LP01 (modes 0)
        // Group 1: LP11a, LP11b (modes 1, 2)
        // Group 2: LP21a, LP21b, LP02 (modes 3, 4, 5) – if present
        let group0: Vec<usize> = (0..1).filter(|&i| i < n).collect();
        let group1: Vec<usize> = (1..3).filter(|&i| i < n).collect();
        let group2: Vec<usize> = (3..6).filter(|&i| i < n).collect();
        let mut groups = vec![group0, group1];
        if !group2.is_empty() {
            groups.push(group2);
        }
        Self {
            mode_groups: groups,
            crosstalk_within_group_db: -15.0, // intra-group XT harder to suppress
            crosstalk_between_groups_db: -30.0, // inter-group XT suppressed by DGD
        }
    }

    /// Number of mode groups.
    pub fn n_groups(&self) -> usize {
        self.mode_groups.len()
    }

    /// Number of modes in each group.
    pub fn modes_per_group(&self) -> Vec<usize> {
        self.mode_groups.iter().map(|g| g.len()).collect()
    }

    /// Effective capacity factor accounting for intra-group cross-talk.
    ///
    ///   factor = Π_{groups} (1 − XT_within_group_linear)^{1/n_modes}
    pub fn effective_capacity_factor(&self) -> f64 {
        let xt_linear = 10.0_f64.powf(self.crosstalk_within_group_db / 10.0);
        // Each group has some cross-talk penalty; larger groups suffer more
        let penalty_per_group: Vec<f64> = self
            .mode_groups
            .iter()
            .map(|g| {
                let n_in_group = g.len() as f64;
                (1.0 - xt_linear).max(0.0).powf(1.0 / n_in_group.max(1.0))
            })
            .collect();
        penalty_per_group.iter().product::<f64>() / self.mode_groups.len() as f64
            * self.mode_groups.len() as f64 // normalise back
    }

    /// MIMO matrix size required: maximum number of modes in any single group.
    pub fn required_mimo_size(&self) -> usize {
        self.mode_groups.iter().map(|g| g.len()).max().unwrap_or(0)
    }
}

// ── SDM Capacity Comparison ───────────────────────────────────────────────────

/// Comparison table of capacity across different SDM fiber types.
///
/// All capacities are computed at the same WDM bandwidth and SNR.
pub struct SdmCapacityComparison {
    /// Capacity of conventional SMF + WDM \[Tb/s\].
    pub smf_capacity_tbps: f64,
    /// Capacity of few-mode fiber (6 modes) + WDM + MIMO \[Tb/s\].
    pub fmf_capacity_tbps: f64,
    /// Capacity of multicore fiber (7 cores) + WDM \[Tb/s\].
    pub mcf_capacity_tbps: f64,
    /// Capacity of few-mode multicore fiber (7 cores × 6 modes) + WDM + MIMO \[Tb/s\].
    pub fmf_mcf_capacity_tbps: f64,
}

impl SdmCapacityComparison {
    /// Compute capacity comparison for the given WDM bandwidth and per-channel SNR.
    ///
    /// Uses the Shannon capacity formula: C = N · B · log₂(1 + SNR/N).
    pub fn compute(bandwidth_thz: f64, snr_db: f64) -> Self {
        let snr = 10.0_f64.powf(snr_db / 10.0);
        let bw_hz = bandwidth_thz * 1.0e12;

        // Capacity helper: C [Tb/s]
        let capacity_tbps =
            |n_sdm: f64| -> f64 { n_sdm * bw_hz * (1.0 + snr / n_sdm).log2() * 1.0e-12 };

        let smf_cap = capacity_tbps(1.0);
        let fmf_cap = capacity_tbps(6.0); // 6 spatial modes
        let mcf_cap = capacity_tbps(7.0); // 7 cores, single-mode each
        let fm_mcf_cap = capacity_tbps(42.0); // 7 cores × 6 modes

        Self {
            smf_capacity_tbps: smf_cap,
            fmf_capacity_tbps: fmf_cap,
            mcf_capacity_tbps: mcf_cap,
            fmf_mcf_capacity_tbps: fm_mcf_cap,
        }
    }

    /// SDM capacity gain over conventional SMF (linear ratio).
    pub fn sdm_gain_over_smf(&self) -> f64 {
        if self.smf_capacity_tbps <= 0.0 {
            return 0.0;
        }
        self.fmf_mcf_capacity_tbps / self.smf_capacity_tbps
    }

    /// Format a capacity summary table as a `String`.
    pub fn print_summary(&self) -> String {
        let gain = self.sdm_gain_over_smf();
        format!(
            "SDM Capacity Comparison\n\
             ========================\n\
             SMF (reference):        {:>8.1} Tb/s\n\
             FMF (6 modes):          {:>8.1} Tb/s  ({:.1}× SMF)\n\
             MCF (7 cores):          {:>8.1} Tb/s  ({:.1}× SMF)\n\
             FM-MCF (7c×6m):         {:>8.1} Tb/s  ({:.1}× SMF)\n\
             ========================\n\
             Total SDM gain: {:.0}×",
            self.smf_capacity_tbps,
            self.fmf_capacity_tbps,
            self.fmf_capacity_tbps / self.smf_capacity_tbps.max(1.0e-12),
            self.mcf_capacity_tbps,
            self.mcf_capacity_tbps / self.smf_capacity_tbps.max(1.0e-12),
            self.fmf_mcf_capacity_tbps,
            self.fmf_mcf_capacity_tbps / self.smf_capacity_tbps.max(1.0e-12),
            gain
        )
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::few_mode_fiber::FewModeFiber;
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_system(n_modes: usize, n_wl: usize) -> MdmSystem {
        let fiber = FewModeFiber::new_6mode(100.0, 1.55e-6);
        MdmSystem::new(fiber, n_modes, n_wl, 32.0, MdmModFormat::DpQpsk)
    }

    #[test]
    fn test_modformat_bits_per_symbol() {
        assert_eq!(MdmModFormat::DpQpsk.bits_per_symbol(), 4);
        assert_eq!(MdmModFormat::Dp16Qam.bits_per_symbol(), 8);
        assert_eq!(MdmModFormat::Dp64Qam.bits_per_symbol(), 12);
    }

    #[test]
    fn test_total_capacity_scales_with_modes() {
        let sys1 = make_system(2, 80);
        let sys2 = make_system(4, 80);
        assert!(sys2.total_capacity_tbps() > sys1.total_capacity_tbps());
    }

    #[test]
    fn test_shannon_capacity_positive() {
        let sys = make_system(6, 80);
        let cap = sys.shannon_capacity_tbps(20.0, 4.0);
        assert!(cap > 0.0, "Shannon capacity should be positive");
    }

    #[test]
    fn test_aggregate_se_multiple_of_per_mode() {
        let sys = make_system(6, 80);
        assert_abs_diff_eq!(
            sys.aggregate_se_bps_per_hz(),
            6.0 * sys.spectral_efficiency_per_mode(),
            epsilon = 1.0e-9
        );
    }

    #[test]
    fn test_mimo_complexity_grows_with_taps() {
        let sys = make_system(6, 80);
        let c1 = sys.mimo_complexity_tops(32);
        let c2 = sys.mimo_complexity_tops(64);
        assert!(c2 > c1, "Complexity should grow with tap count");
    }

    #[test]
    fn test_sdm_gain_greater_than_1() {
        let comp = SdmCapacityComparison::compute(4.0, 20.0);
        assert!(comp.sdm_gain_over_smf() > 1.0, "SDM should outperform SMF");
    }

    #[test]
    fn test_fmf_mcf_capacity_greater_than_fmf() {
        let comp = SdmCapacityComparison::compute(4.0, 20.0);
        assert!(
            comp.fmf_mcf_capacity_tbps > comp.fmf_capacity_tbps,
            "FM-MCF should exceed FMF capacity"
        );
    }

    #[test]
    fn test_modegroup_demux_3groups() {
        let fiber = FewModeFiber::new_6mode(100.0, 1.55e-6);
        let demux = ModegroupDemux::new_3group(&fiber);
        assert_eq!(demux.n_groups(), 3, "Should have 3 mode groups");
        let total_modes: usize = demux.modes_per_group().iter().sum();
        assert!(total_modes > 0, "Should have modes assigned to groups");
    }

    #[test]
    fn test_print_summary_non_empty() {
        let comp = SdmCapacityComparison::compute(4.0, 20.0);
        let summary = comp.print_summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("SMF"), "Summary should mention SMF");
    }

    #[test]
    fn test_net_se_less_than_gross() {
        for fmt in &[
            MdmModFormat::DpQpsk,
            MdmModFormat::Dp16Qam,
            MdmModFormat::Dp64Qam,
        ] {
            assert!(
                fmt.net_spectral_efficiency() < fmt.spectral_efficiency(),
                "Net SE must be less than gross SE for {:?}",
                fmt
            );
        }
    }
}
