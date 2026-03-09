//! WDM optical system design following ITU-T G.694.1.
//!
//! Implements channel planning for C-band and C+L band DWDM systems,
//! modulation format characterization, and line system performance metrics.
//!
//! # Reference
//! - ITU-T G.694.1: Spectral grids for WDM applications — DWDM frequency grid
//! - ITU-T G.977: Characteristics of optically amplified optical fibre submarine cable systems

/// Speed of light in vacuum \[m/s\]
const C_M_PER_S: f64 = 2.997_924_58e8;

// ─────────────────────────────────────────────────────────────────────────────
// ItuGrid
// ─────────────────────────────────────────────────────────────────────────────

/// ITU-T G.694.1 frequency grid type.
#[derive(Debug, Clone, PartialEq)]
pub enum ItuGrid {
    /// Fixed 100 GHz spacing — classic C-band DWDM (up to 40 channels).
    Fixed100Ghz,
    /// Fixed 50 GHz spacing — dense C-band (up to 80 channels).
    Fixed50Ghz,
    /// Fixed 25 GHz spacing — ultra-dense C-band (up to 160 channels).
    Fixed25Ghz,
    /// Flex-grid: variable slot width, minimum slot granularity 12.5 GHz.
    Flex {
        /// Minimum channel spacing granularity \[GHz\].
        min_spacing_ghz: f64,
    },
}

impl ItuGrid {
    /// Nominal channel spacing \[GHz\].
    pub fn spacing_ghz(&self) -> f64 {
        match self {
            ItuGrid::Fixed100Ghz => 100.0,
            ItuGrid::Fixed50Ghz => 50.0,
            ItuGrid::Fixed25Ghz => 25.0,
            ItuGrid::Flex { min_spacing_ghz } => *min_spacing_ghz,
        }
    }

    /// Whether this is a flex-grid specification.
    pub fn is_flex(&self) -> bool {
        matches!(self, ItuGrid::Flex { .. })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ItuChannelPlan
// ─────────────────────────────────────────────────────────────────────────────

/// ITU-T G.694.1 channel plan.
///
/// Channel frequencies follow the formula:
/// ```text
///   f_k = f_ref + k × Δf   \[THz\]
/// ```
/// where `f_ref = 193.1 THz` (ITU reference), `k` is the channel index
/// (0-based from the lowest frequency channel), and `Δf` is the grid spacing.
#[derive(Debug, Clone)]
pub struct ItuChannelPlan {
    /// Grid type (fixed or flex).
    pub grid_type: ItuGrid,
    /// Reference (anchor) frequency for channel 0 \[THz\].
    pub center_frequency_thz: f64,
    /// Total number of channels.
    pub n_channels: usize,
}

impl ItuChannelPlan {
    /// Standard C-band 100 GHz grid — 32 channels from 191.3 to 194.4 THz.
    ///
    /// Covers the conventional C-band (1530–1565 nm).
    pub fn new_c_band_100ghz() -> Self {
        Self {
            grid_type: ItuGrid::Fixed100Ghz,
            // Start at 191.3 THz so channels span 191.3–194.4 THz (32 channels)
            center_frequency_thz: 191.3,
            n_channels: 32,
        }
    }

    /// Standard C-band 50 GHz grid — 64 channels from 191.35 to 194.45 THz.
    pub fn new_c_band_50ghz() -> Self {
        Self {
            grid_type: ItuGrid::Fixed50Ghz,
            center_frequency_thz: 191.35,
            n_channels: 64,
        }
    }

    /// C+L band 100 GHz grid — 80 channels spanning both C and L band.
    ///
    /// C-band: 191.3–196.2 THz, L-band: 184.5–191.2 THz (approx).
    pub fn new_c_plus_l_100ghz() -> Self {
        Self {
            grid_type: ItuGrid::Fixed100Ghz,
            center_frequency_thz: 184.5,
            n_channels: 80,
        }
    }

    /// Create a flex-grid channel plan with custom parameters.
    pub fn new_flex(start_freq_thz: f64, min_spacing_ghz: f64, n_channels: usize) -> Self {
        Self {
            grid_type: ItuGrid::Flex { min_spacing_ghz },
            center_frequency_thz: start_freq_thz,
            n_channels,
        }
    }

    /// Grid spacing \[GHz\].
    pub fn spacing_ghz(&self) -> f64 {
        self.grid_type.spacing_ghz()
    }

    /// Channel frequency at index `channel` \[THz\].
    ///
    /// `f_k = f_ref + k × Δf`
    pub fn channel_frequency_thz(&self, channel: usize) -> f64 {
        let df_thz = self.spacing_ghz() * 1e-3; // GHz → THz
        self.center_frequency_thz + channel as f64 * df_thz
    }

    /// Channel wavelength at index `channel` \[nm\].
    pub fn channel_wavelength_nm(&self, channel: usize) -> f64 {
        let f_hz = self.channel_frequency_thz(channel) * 1e12;
        C_M_PER_S / f_hz * 1e9
    }

    /// Check whether `freq_thz` falls on the ITU grid (within ±1 MHz tolerance).
    pub fn is_on_grid(&self, freq_thz: f64) -> bool {
        let df_thz = self.spacing_ghz() * 1e-3;
        let offset = freq_thz - self.center_frequency_thz;
        // Remainder when divided by spacing
        let remainder = offset.rem_euclid(df_thz);
        let tol_thz = 1e-6; // 1 MHz
        remainder < tol_thz || (df_thz - remainder) < tol_thz
    }

    /// Return the index of the channel nearest to `freq_thz`.
    ///
    /// Clamps to `[0, n_channels - 1]`.
    pub fn nearest_channel(&self, freq_thz: f64) -> usize {
        if self.n_channels == 0 {
            return 0;
        }
        let df_thz = self.spacing_ghz() * 1e-3;
        let raw_idx = ((freq_thz - self.center_frequency_thz) / df_thz).round();

        raw_idx.max(0.0).min((self.n_channels - 1) as f64) as usize
    }

    /// Total capacity \[Tb/s\] assuming `bits_per_symbol` polarization-multiplexed
    /// symbols at `baud_gbaud` Gbaud per channel.
    ///
    /// Capacity = n_channels × bits_per_symbol × baud_gbaud \[Gb/s\] / 1000
    pub fn total_capacity_tbps(&self, bits_per_symbol: u32, baud_gbaud: f64) -> f64 {
        let per_channel_gbps = bits_per_symbol as f64 * baud_gbaud;
        self.n_channels as f64 * per_channel_gbps / 1e3
    }

    /// All channel frequencies as a vector \[THz\].
    pub fn all_frequencies_thz(&self) -> Vec<f64> {
        (0..self.n_channels)
            .map(|k| self.channel_frequency_thz(k))
            .collect()
    }

    /// All channel wavelengths as a vector \[nm\].
    pub fn all_wavelengths_nm(&self) -> Vec<f64> {
        (0..self.n_channels)
            .map(|k| self.channel_wavelength_nm(k))
            .collect()
    }

    /// Total optical bandwidth occupied \[THz\].
    pub fn occupied_bandwidth_thz(&self) -> f64 {
        if self.n_channels == 0 {
            return 0.0;
        }
        self.spacing_ghz() * 1e-3 * self.n_channels as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WdmModFormat
// ─────────────────────────────────────────────────────────────────────────────

/// WDM modulation format with associated performance parameters.
#[derive(Debug, Clone, PartialEq)]
pub enum WdmModFormat {
    /// On-off keying, non-return-to-zero (OOK-NRZ).
    OokNrz,
    /// Dual-polarization QPSK (DP-QPSK): 2 bits/symbol × 2 pol = 4 b/sym.
    DpQpsk,
    /// Dual-polarization 16-QAM (DP-16QAM): 4 b/sym × 2 pol = 8 b/sym.
    Dp16Qam,
    /// Dual-polarization 64-QAM (DP-64QAM): 6 b/sym × 2 pol = 12 b/sym.
    Dp64Qam,
    /// Dual-polarization 256-QAM (DP-256QAM): 8 b/sym × 2 pol = 16 b/sym.
    Dp256Qam,
    /// User-defined format.
    Custom {
        /// Bits per symbol (including polarization multiplexing).
        bits_per_symbol: u32,
        /// Required OSNR at BER = 1×10⁻² (pre-FEC) \[dB\].
        required_osnr_db: f64,
    },
}

impl WdmModFormat {
    /// Bits per (polarization-multiplexed) symbol.
    pub fn bits_per_symbol(&self) -> u32 {
        match self {
            WdmModFormat::OokNrz => 1,
            WdmModFormat::DpQpsk => 4,
            WdmModFormat::Dp16Qam => 8,
            WdmModFormat::Dp64Qam => 12,
            WdmModFormat::Dp256Qam => 16,
            WdmModFormat::Custom {
                bits_per_symbol, ..
            } => *bits_per_symbol,
        }
    }

    /// Required OSNR \[dB\] at BER = 1×10⁻² (pre-FEC threshold).
    ///
    /// Values are typical implementation penalties from literature:
    /// - OOK-NRZ: ~14 dB
    /// - DP-QPSK: ~10.5 dB  (Shannon: ~8 dB, +2.5 dB penalty)
    /// - DP-16QAM: ~16 dB
    /// - DP-64QAM: ~22 dB
    /// - DP-256QAM: ~28 dB
    pub fn required_osnr_db(&self) -> f64 {
        match self {
            WdmModFormat::OokNrz => 14.0,
            WdmModFormat::DpQpsk => 10.5,
            WdmModFormat::Dp16Qam => 16.0,
            WdmModFormat::Dp64Qam => 22.0,
            WdmModFormat::Dp256Qam => 28.0,
            WdmModFormat::Custom {
                required_osnr_db, ..
            } => *required_osnr_db,
        }
    }

    /// Spectral efficiency \[bits/s/Hz\].
    ///
    /// For polarization-multiplexed formats the spectral efficiency equals
    /// `bits_per_symbol` (since one symbol carries information in both polarizations).
    pub fn spectral_efficiency_bps_per_hz(&self) -> f64 {
        self.bits_per_symbol() as f64
    }

    /// Typical modulation gain relative to OOK \[dB\].
    pub fn modulation_gain_db(&self) -> f64 {
        self.required_osnr_db() - WdmModFormat::OokNrz.required_osnr_db()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WdmLineSystem
// ─────────────────────────────────────────────────────────────────────────────

/// WDM optical line system.
///
/// Encapsulates the channel plan, modulation format, baud rate, launch power,
/// and FEC overhead to compute system-level performance metrics.
#[derive(Debug, Clone)]
pub struct WdmLineSystem {
    /// ITU channel plan.
    pub channel_plan: ItuChannelPlan,
    /// Per-channel launch power \[dBm\].
    pub launch_power_dbm_per_channel: f64,
    /// Modulation format.
    pub modulation_format: WdmModFormat,
    /// Symbol rate \[Gbaud\].
    pub baud_rate_gbaud: f64,
    /// FEC overhead fraction (e.g., 0.20 for 20% overhead).
    pub fec_overhead: f64,
}

impl WdmLineSystem {
    /// Construct a new WDM line system.
    ///
    /// # Arguments
    /// - `plan` — ITU channel plan
    /// - `launch_dbm` — per-channel launch power \[dBm\]
    /// - `format` — modulation format
    /// - `baud` — symbol rate \[Gbaud\]
    pub fn new(plan: ItuChannelPlan, launch_dbm: f64, format: WdmModFormat, baud: f64) -> Self {
        Self {
            channel_plan: plan,
            launch_power_dbm_per_channel: launch_dbm,
            modulation_format: format,
            baud_rate_gbaud: baud,
            fec_overhead: 0.20,
        }
    }

    /// Construct with explicit FEC overhead.
    pub fn with_fec_overhead(mut self, overhead: f64) -> Self {
        self.fec_overhead = overhead;
        self
    }

    /// Gross channel bit rate (before FEC removal) \[Gb/s\].
    ///
    /// `R_gross = bits_per_symbol × baud_rate`
    pub fn gross_bit_rate_gbps(&self) -> f64 {
        self.modulation_format.bits_per_symbol() as f64 * self.baud_rate_gbaud
    }

    /// Net channel bit rate after FEC overhead removal \[Gb/s\].
    ///
    /// `R_net = R_gross / (1 + fec_overhead)`
    pub fn net_bit_rate_gbps(&self) -> f64 {
        self.gross_bit_rate_gbps() / (1.0 + self.fec_overhead)
    }

    /// Total system capacity \[Tb/s\].
    pub fn total_capacity_tbps(&self) -> f64 {
        self.net_bit_rate_gbps() * self.channel_plan.n_channels as f64 / 1e3
    }

    /// Channel bandwidth using Nyquist criterion with RRC roll-off \[GHz\].
    ///
    /// `B = 1.1 × baud_rate`  (10% excess for root-raised-cosine filter guard band)
    pub fn channel_bandwidth_ghz(&self) -> f64 {
        1.1 * self.baud_rate_gbaud
    }

    /// Spectral efficiency \[bits/s/Hz\].
    ///
    /// `SE = net_bit_rate / channel_spacing`
    pub fn spectral_efficiency(&self) -> f64 {
        let channel_spacing_hz = self.channel_plan.spacing_ghz() * 1e9;
        self.net_bit_rate_gbps() * 1e9 / channel_spacing_hz
    }

    /// OSNR margin \[dB\] for the given actual OSNR.
    ///
    /// `margin = OSNR_actual - OSNR_required`
    pub fn osnr_margin_db(&self, actual_osnr_db: f64) -> f64 {
        actual_osnr_db - self.modulation_format.required_osnr_db()
    }

    /// Return `true` when `actual_osnr_db` exceeds the required OSNR.
    pub fn is_viable(&self, actual_osnr_db: f64) -> bool {
        self.osnr_margin_db(actual_osnr_db) >= 0.0
    }

    /// Number of channels.
    pub fn n_channels(&self) -> usize {
        self.channel_plan.n_channels
    }

    /// Launch power per channel \[mW\].
    pub fn launch_power_mw(&self) -> f64 {
        10.0_f64.powf(self.launch_power_dbm_per_channel / 10.0)
    }

    /// Total launch power into the fiber \[dBm\].
    ///
    /// `P_total = P_ch + 10·log10(N_ch)`
    pub fn total_launch_power_dbm(&self) -> f64 {
        self.launch_power_dbm_per_channel + 10.0 * (self.channel_plan.n_channels as f64).log10()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn itu_c_band_100ghz_channel_count() {
        let plan = ItuChannelPlan::new_c_band_100ghz();
        assert_eq!(plan.n_channels, 32);
    }

    #[test]
    fn itu_channel_frequency_spacing() {
        let plan = ItuChannelPlan::new_c_band_100ghz();
        let f0 = plan.channel_frequency_thz(0);
        let f1 = plan.channel_frequency_thz(1);
        assert_abs_diff_eq!(f1 - f0, 0.1, epsilon = 1e-9); // 100 GHz = 0.1 THz
    }

    #[test]
    fn itu_channel_wavelength_in_c_band() {
        let plan = ItuChannelPlan::new_c_band_100ghz();
        // C-band: 1530–1565 nm
        for k in 0..plan.n_channels {
            let wl = plan.channel_wavelength_nm(k);
            assert!(wl > 1520.0 && wl < 1580.0, "ch{k}: λ={wl:.2} nm");
        }
    }

    #[test]
    fn itu_nearest_channel_roundtrip() {
        let plan = ItuChannelPlan::new_c_band_50ghz();
        for k in [0, 10, 30, 63] {
            let f = plan.channel_frequency_thz(k);
            assert_eq!(plan.nearest_channel(f), k);
        }
    }

    #[test]
    fn itu_is_on_grid() {
        let plan = ItuChannelPlan::new_c_band_100ghz();
        let f_on = plan.channel_frequency_thz(5);
        assert!(plan.is_on_grid(f_on));
        // Off-grid: add 37 MHz
        assert!(!plan.is_on_grid(f_on + 37e-6));
    }

    #[test]
    fn wdm_mod_format_bits_per_symbol() {
        assert_eq!(WdmModFormat::DpQpsk.bits_per_symbol(), 4);
        assert_eq!(WdmModFormat::Dp16Qam.bits_per_symbol(), 8);
        assert_eq!(WdmModFormat::Dp64Qam.bits_per_symbol(), 12);
        assert_eq!(WdmModFormat::Dp256Qam.bits_per_symbol(), 16);
    }

    #[test]
    fn wdm_mod_format_required_osnr_increases_with_order() {
        assert!(WdmModFormat::DpQpsk.required_osnr_db() < WdmModFormat::Dp16Qam.required_osnr_db());
        assert!(
            WdmModFormat::Dp16Qam.required_osnr_db() < WdmModFormat::Dp64Qam.required_osnr_db()
        );
    }

    #[test]
    fn wdm_line_system_net_bit_rate() {
        let plan = ItuChannelPlan::new_c_band_100ghz();
        let sys = WdmLineSystem::new(plan, 0.0, WdmModFormat::DpQpsk, 32.0);
        // gross = 4 * 32 = 128 Gb/s, net = 128 / 1.2 ≈ 106.67 Gb/s
        let expected = 128.0 / 1.20;
        assert_abs_diff_eq!(sys.net_bit_rate_gbps(), expected, epsilon = 0.01);
    }

    #[test]
    fn wdm_line_system_capacity_scales() {
        let plan = ItuChannelPlan::new_c_band_100ghz();
        let sys = WdmLineSystem::new(plan, 0.0, WdmModFormat::Dp16Qam, 32.0);
        let capacity = sys.total_capacity_tbps();
        assert!(capacity > 0.0);
        // 32 channels × (8 × 32 / 1.2) Gb/s / 1000 ≈ 6.83 Tb/s
        assert!(capacity > 5.0 && capacity < 10.0);
    }

    #[test]
    fn wdm_line_system_osnr_margin() {
        let plan = ItuChannelPlan::new_c_band_100ghz();
        let sys = WdmLineSystem::new(plan, 0.0, WdmModFormat::DpQpsk, 32.0);
        // Required OSNR = 10.5 dB; actual = 15 dB → margin = 4.5 dB
        assert_abs_diff_eq!(sys.osnr_margin_db(15.0), 4.5, epsilon = 1e-9);
        assert!(sys.is_viable(15.0));
        assert!(!sys.is_viable(9.0));
    }

    #[test]
    fn wdm_line_system_spectral_efficiency_positive() {
        let plan = ItuChannelPlan::new_c_band_50ghz();
        let sys = WdmLineSystem::new(plan, 0.0, WdmModFormat::Dp16Qam, 32.0);
        let se = sys.spectral_efficiency();
        assert!(se > 0.0 && se < 20.0, "SE = {se}");
    }

    #[test]
    fn itu_total_capacity_formula() {
        let plan = ItuChannelPlan::new_c_band_100ghz();
        // 32 channels × 4 b/sym × 32 Gbaud / 1000 = 4.096 Tb/s
        let cap = plan.total_capacity_tbps(4, 32.0);
        assert_abs_diff_eq!(cap, 32.0 * 4.0 * 32.0 / 1000.0, epsilon = 1e-9);
    }
}
