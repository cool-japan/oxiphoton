//! Wavelength Division Multiplexing (WDM) system model.
//!
//! WDM allows N channels to share a single fiber, each at a different wavelength.
//! Standard ITU grids:
//!   - DWDM C-band: 193.1 THz ± N×12.5 GHz (0.1 nm spacing, up to 80+ channels)
//!   - CWDM: 1270–1610 nm, 20 nm spacing (18 channels)
//!   - LAN-WDM: 1295–1310 nm, 4 channels (100 GHz spacing)
//!
//! Key metrics:
//!   - Aggregate capacity: N × B (total Gb/s)
//!   - Spectral efficiency: C / (N × Δf) (bits/s/Hz)
//!   - Channel crosstalk from MUX/DEMUX filter non-idealities

const SPEED_OF_LIGHT_M: f64 = 2.998e8;

/// ITU-T WDM grid specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WdmGrid {
    /// Dense WDM, 100 GHz spacing (0.8 nm at 1550 nm)
    Dwdm100Ghz,
    /// Dense WDM, 50 GHz spacing (0.4 nm at 1550 nm)
    Dwdm50Ghz,
    /// Dense WDM, 25 GHz spacing (0.2 nm at 1550 nm)
    Dwdm25Ghz,
    /// Coarse WDM, 20 nm spacing
    Cwdm,
    /// LAN WDM (4 channels, 100 GHz)
    LanWdm,
}

impl WdmGrid {
    /// Channel spacing in Hz.
    pub fn spacing_hz(&self) -> f64 {
        match self {
            WdmGrid::Dwdm100Ghz => 100e9,
            WdmGrid::Dwdm50Ghz => 50e9,
            WdmGrid::Dwdm25Ghz => 25e9,
            WdmGrid::Cwdm => SPEED_OF_LIGHT_M / (1550e-9 * 1550e-9) * 20e-9, // ~2.5 THz
            WdmGrid::LanWdm => 100e9,
        }
    }

    /// Channel spacing in nm at 1550 nm.
    pub fn spacing_nm(&self) -> f64 {
        let df = self.spacing_hz();
        1550e-9 * 1550e-9 * df / SPEED_OF_LIGHT_M * 1e9
    }
}

/// WDM channel specification.
#[derive(Debug, Clone, Copy)]
pub struct WdmChannel {
    /// Channel index (0-based)
    pub index: usize,
    /// Center frequency (Hz)
    pub frequency_hz: f64,
    /// Channel bandwidth (Hz)
    pub bandwidth_hz: f64,
    /// Launch power (dBm)
    pub launch_power_dbm: f64,
}

impl WdmChannel {
    /// Center wavelength (m).
    pub fn wavelength_m(&self) -> f64 {
        SPEED_OF_LIGHT_M / self.frequency_hz
    }

    /// Frequency offset from another channel (Hz).
    pub fn frequency_offset(&self, other: &WdmChannel) -> f64 {
        (self.frequency_hz - other.frequency_hz).abs()
    }
}

/// WDM system model.
#[derive(Debug, Clone)]
pub struct WdmSystem {
    /// Channels
    pub channels: Vec<WdmChannel>,
    /// Grid type
    pub grid: WdmGrid,
    /// MUX/DEMUX insertion loss per channel (dB)
    pub mux_loss_db: f64,
    /// Adjacent-channel crosstalk suppression (dB)
    pub crosstalk_suppression_db: f64,
}

impl WdmSystem {
    /// Create DWDM system with N channels, starting at 193.1 THz.
    pub fn dwdm(n_channels: usize, grid: WdmGrid) -> Self {
        let f0 = 193.1e12; // ITU reference frequency (Hz)
        let df = grid.spacing_hz();
        let channels = (0..n_channels)
            .map(|i| WdmChannel {
                index: i,
                frequency_hz: f0 + i as f64 * df,
                bandwidth_hz: df * 0.5, // 50% fill factor
                launch_power_dbm: 0.0,
            })
            .collect();
        Self {
            channels,
            grid,
            mux_loss_db: 3.0,
            crosstalk_suppression_db: 30.0,
        }
    }

    /// Standard 8-channel DWDM C-band at 100 GHz.
    pub fn c_band_8ch() -> Self {
        Self::dwdm(8, WdmGrid::Dwdm100Ghz)
    }

    /// Standard 4-channel LAN-WDM (IEEE 802.3ba).
    pub fn lan_wdm_4ch() -> Self {
        Self::dwdm(4, WdmGrid::LanWdm)
    }

    /// Total number of channels.
    pub fn n_channels(&self) -> usize {
        self.channels.len()
    }

    /// Total aggregate capacity at given per-channel data rate (Gb/s).
    pub fn aggregate_capacity_gbps(&self, per_channel_gbps: f64) -> f64 {
        self.channels.len() as f64 * per_channel_gbps
    }

    /// Spectral efficiency (bits/s/Hz) for given per-channel rate and format.
    pub fn spectral_efficiency(&self, per_channel_gbps: f64) -> f64 {
        let df = self.grid.spacing_hz();
        per_channel_gbps * 1e9 / df
    }

    /// C-band occupied bandwidth (THz).
    pub fn occupied_bandwidth_thz(&self) -> f64 {
        let df = self.grid.spacing_hz();
        self.channels.len() as f64 * df / 1e12
    }

    /// Wavelength range (nm): (λ_min, λ_max).
    pub fn wavelength_range_nm(&self) -> (f64, f64) {
        let f_min = self
            .channels
            .iter()
            .map(|c| c.frequency_hz)
            .fold(f64::INFINITY, f64::min);
        let f_max = self
            .channels
            .iter()
            .map(|c| c.frequency_hz)
            .fold(f64::NEG_INFINITY, f64::max);
        (
            SPEED_OF_LIGHT_M / f_max * 1e9,
            SPEED_OF_LIGHT_M / f_min * 1e9,
        )
    }

    /// Crosstalk power ratio from nearest-neighbor channel (linear).
    ///
    ///   XT = 10^(-crosstalk_suppression_db / 10)
    pub fn nearest_neighbor_xt(&self) -> f64 {
        10.0_f64.powf(-self.crosstalk_suppression_db / 10.0)
    }

    /// OSNR penalty from accumulated crosstalk (dB), assuming all N-1 channels interfere.
    ///
    ///   OSNR_penalty ≈ -10·log10(1 - (N-1)·XT_linear)
    pub fn xt_osnr_penalty_db(&self) -> f64 {
        let n = self.n_channels() as f64;
        let xt = self.nearest_neighbor_xt();
        let total_xt = (n - 1.0) * xt;
        if total_xt >= 1.0 {
            return 30.0;
        }
        -10.0 * (1.0 - total_xt).log10()
    }

    /// Channel frequencies as vector (Hz).
    pub fn frequencies_hz(&self) -> Vec<f64> {
        self.channels.iter().map(|c| c.frequency_hz).collect()
    }

    /// Add a Gaussian MUX filter to calculate isolation at offset Δf (Hz).
    ///
    /// MUX filter: H(f) = exp(-2·ln2·(Δf/BW_3dB)²)
    /// Suppression at adjacent channel offset.
    pub fn mux_isolation_db(&self, delta_f_hz: f64, bw_3db_hz: f64) -> f64 {
        let x = delta_f_hz / bw_3db_hz;
        let h_sq = (-4.0 * std::f64::consts::LN_2 * x * x).exp();
        -10.0 * h_sq.max(1e-30).log10()
    }
}

/// Ring resonator add-drop WDM filter.
///
/// Each channel is dropped by a resonant ring tuned to that channel's frequency.
#[derive(Debug, Clone, Copy)]
pub struct RingWdmFilter {
    /// Ring resonance FSR (Hz)
    pub fsr_hz: f64,
    /// Ring quality factor Q
    pub q_factor: f64,
    /// Insertion loss (dB)
    pub insertion_loss_db: f64,
    /// Through/drop extinction ratio (dB)
    pub extinction_ratio_db: f64,
}

impl RingWdmFilter {
    /// Microring filter for DWDM 100 GHz channels.
    pub fn dwdm_100ghz() -> Self {
        Self {
            fsr_hz: 1.5e12, // ~15 channels of 100 GHz
            q_factor: 1e4,
            insertion_loss_db: 0.5,
            extinction_ratio_db: 20.0,
        }
    }

    /// 3 dB filter bandwidth (Hz).
    pub fn bandwidth_hz(&self, center_frequency_hz: f64) -> f64 {
        center_frequency_hz / self.q_factor
    }

    /// Adjacent channel suppression (dB) at channel spacing Δf.
    pub fn adjacent_suppression_db(&self, delta_f_hz: f64, center_hz: f64) -> f64 {
        let bw = self.bandwidth_hz(center_hz);
        // Lorentzian filter: |H(f)|² = 1 / (1 + (Δf/(BW/2))²)
        let x = 2.0 * delta_f_hz / bw;
        10.0 * (1.0 + x * x).log10()
    }

    /// Number of channels supportable within FSR.
    pub fn max_channels(&self, channel_spacing_hz: f64) -> usize {
        (self.fsr_hz / channel_spacing_hz).floor() as usize
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DwdmChannelPlan
// ─────────────────────────────────────────────────────────────────────────────

/// ITU-T DWDM channel plan centered at a given frequency/wavelength.
#[derive(Debug, Clone)]
pub struct DwdmChannelPlan {
    /// Center wavelength of the plan (nm)
    pub center_wavelength_nm: f64,
    /// Channel spacing (GHz)
    pub channel_spacing_ghz: f64,
    /// Number of channels
    pub n_channels: usize,
}

impl DwdmChannelPlan {
    /// Create a new DWDM channel plan.
    ///
    /// Channels are numbered 0 … n_channels-1.  Channel index 0 is the lowest
    /// frequency (longest wavelength) channel.
    pub fn new(center_wavelength_nm: f64, channel_spacing_ghz: f64, n_channels: usize) -> Self {
        Self {
            center_wavelength_nm,
            channel_spacing_ghz,
            n_channels,
        }
    }

    /// Center frequency of the plan (THz).
    fn center_freq_thz(&self) -> f64 {
        SPEED_OF_LIGHT_M / (self.center_wavelength_nm * 1e-9) / 1e12
    }

    /// Frequency of channel `ch_idx` in THz.
    ///
    /// Channels are symmetric around the center; for an even number of channels
    /// the center lies between two channels.
    fn channel_freq_thz(&self, ch_idx: usize) -> f64 {
        let df = self.channel_spacing_ghz * 1e-3; // GHz → THz
        let half = (self.n_channels as f64 - 1.0) / 2.0;
        self.center_freq_thz() + (ch_idx as f64 - half) * df
    }

    /// All channel center wavelengths (nm), index 0 = lowest frequency.
    pub fn channel_wavelengths_nm(&self) -> Vec<f64> {
        (0..self.n_channels)
            .map(|i| {
                let f_thz = self.channel_freq_thz(i);
                SPEED_OF_LIGHT_M / (f_thz * 1e12) * 1e9
            })
            .collect()
    }

    /// All channel center frequencies (THz), index 0 = lowest frequency.
    pub fn channel_frequencies_thz(&self) -> Vec<f64> {
        (0..self.n_channels)
            .map(|i| self.channel_freq_thz(i))
            .collect()
    }

    /// Find the index of the channel whose centre wavelength is nearest to
    /// `wl_nm`.  Returns `None` if the plan has no channels.
    pub fn channel_index_for_wavelength(&self, wl_nm: f64) -> Option<usize> {
        if self.n_channels == 0 {
            return None;
        }
        let wls = self.channel_wavelengths_nm();
        let idx = wls
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (*a - wl_nm)
                    .abs()
                    .partial_cmp(&(*b - wl_nm).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);
        idx
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CrosstalkMatrix
// ─────────────────────────────────────────────────────────────────────────────

/// N × N crosstalk matrix for a WDM demux/mux.
///
/// Element `[from, to]` holds the crosstalk power (dB) leaked from channel
/// `from` into channel `to`.  Diagonal entries are 0 dB by convention (direct
/// path, not actually cross-talk in the isolation sense).
#[derive(Debug, Clone)]
pub struct CrosstalkMatrix {
    n: usize,
    /// Row-major, size n × n.  Entry \[from * n + to\].
    data: Vec<f64>,
}

impl CrosstalkMatrix {
    /// Create an n×n crosstalk matrix.  All off-diagonal entries default to
    /// −60 dB (negligible), diagonal = 0 dB.
    pub fn new(n_channels: usize) -> Self {
        let n = n_channels;
        let mut data = vec![-60.0_f64; n * n];
        for i in 0..n {
            data[i * n + i] = 0.0;
        }
        Self { n, data }
    }

    /// Set the crosstalk from channel `from` to channel `to` (dB, ≤ 0 typically).
    ///
    /// A value of −30 dB means 1/1000 of the power leaks through.
    pub fn set_crosstalk(&mut self, from: usize, to: usize, xtalk_db: f64) {
        assert!(from < self.n && to < self.n, "channel index out of range");
        self.data[from * self.n + to] = xtalk_db;
    }

    /// Total interference power received by `channel` from all other channels
    /// (dBm-style logarithmic addition).
    ///
    /// Returns the power sum of all interfering channels (assuming each
    /// transmits 0 dBm), expressed in dB relative to that reference.
    ///
    /// Formula: 10·log10( Σ_{k≠channel} 10^(XT\[k,channel\]/10) )
    pub fn total_interference_db(&self, channel: usize) -> f64 {
        assert!(channel < self.n, "channel index out of range");
        let sum: f64 = (0..self.n)
            .filter(|&k| k != channel)
            .map(|k| 10.0_f64.powf(self.data[k * self.n + channel] / 10.0))
            .sum();
        if sum <= 0.0 {
            return f64::NEG_INFINITY;
        }
        10.0 * sum.log10()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OpticalAddDropMux (OADM)
// ─────────────────────────────────────────────────────────────────────────────

/// Simple optical add-drop multiplexer (OADM) model.
///
/// Channels can be added to or dropped from the bus.  All channels not
/// explicitly dropped are "express" channels that pass through with a fixed
/// insertion loss.
#[derive(Debug, Clone)]
pub struct OpticalAddDropMux {
    channel_plan: DwdmChannelPlan,
    /// Power (dBm) of each channel on the bus; `None` = channel absent.
    bus: Vec<Option<f64>>,
    /// Fixed insertion loss for express (pass-through) channels (dB).
    insertion_loss_db: f64,
}

impl OpticalAddDropMux {
    /// Create a new OADM with the given channel plan.
    pub fn new(channel_plan: DwdmChannelPlan) -> Self {
        let n = channel_plan.n_channels;
        Self {
            channel_plan,
            bus: vec![None; n],
            insertion_loss_db: 1.0,
        }
    }

    /// Add (or replace) a channel on the bus.
    pub fn add_channel(&mut self, ch_idx: usize, power_dbm: f64) {
        assert!(
            ch_idx < self.channel_plan.n_channels,
            "channel index out of range"
        );
        self.bus[ch_idx] = Some(power_dbm);
    }

    /// Drop channel `ch_idx` from the bus, returning its power (dBm), or
    /// `None` if the channel was absent.
    pub fn drop_channel(&mut self, ch_idx: usize) -> Option<f64> {
        assert!(
            ch_idx < self.channel_plan.n_channels,
            "channel index out of range"
        );
        self.bus[ch_idx].take()
    }

    /// Return (index, power_dBm) pairs for all express channels currently on
    /// the bus, with insertion loss applied.
    pub fn express_channels(&self) -> Vec<(usize, f64)> {
        self.bus
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.map(|pw| (i, pw - self.insertion_loss_db)))
            .collect()
    }

    /// Fixed insertion loss for express channels (dB).
    pub fn insertion_loss_db(&self) -> f64 {
        self.insertion_loss_db
    }

    /// Immutable reference to the underlying channel plan.
    pub fn channel_plan(&self) -> &DwdmChannelPlan {
        &self.channel_plan
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PowerEqualizer
// ─────────────────────────────────────────────────────────────────────────────

/// WDM power equalizer — adjusts per-channel gains to reach a target power.
///
/// After calling `equalize`, `channel_gains[i]` holds the gain (dB) that was
/// applied to channel `i`.  Positive gain means amplification, negative means
/// attenuation.
#[derive(Debug, Clone)]
pub struct PowerEqualizer {
    /// Number of channels
    pub n_channels: usize,
    /// Target power per channel (dBm)
    pub target_power_dbm: f64,
    /// Per-channel gain applied by the equalizer (dB)
    pub channel_gains: Vec<f64>,
}

impl PowerEqualizer {
    /// Create a new equalizer for `n_channels` channels with the given target.
    pub fn new(n_channels: usize, target_power_dbm: f64) -> Self {
        Self {
            n_channels,
            target_power_dbm,
            channel_gains: vec![0.0; n_channels],
        }
    }

    /// Compute the per-channel gains required to bring `input_powers_dbm` to the target.
    ///
    /// `input_powers_dbm` must have length ≥ `n_channels`.  Extra entries are ignored.
    pub fn equalize(&mut self, input_powers_dbm: &[f64]) {
        for (i, &p_in) in input_powers_dbm.iter().take(self.n_channels).enumerate() {
            self.channel_gains[i] = self.target_power_dbm - p_in;
        }
    }

    /// Peak-to-peak variation of output powers (dB).
    ///
    /// Returns 0 if fewer than 2 channels are present.
    pub fn max_variation_db(&self) -> f64 {
        if self.n_channels < 2 {
            return 0.0;
        }
        let min_g = self
            .channel_gains
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_g = self
            .channel_gains
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        max_g - min_g
    }

    /// Sum of all channel gains (dB).
    pub fn total_gain_db(&self) -> f64 {
        self.channel_gains.iter().sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Roadm
// ─────────────────────────────────────────────────────────────────────────────

/// ROADM (Reconfigurable Optical Add-Drop Multiplexer).
///
/// Models a degree-`n_degree`, `n_channels`-channel ROADM as a switching matrix.
/// `switching_matrix[from_port][channel]` = `Some(to_port)` if that channel is
/// routed from `from_port` to `to_port`; `None` means the channel is dropped.
#[derive(Debug, Clone)]
pub struct Roadm {
    /// Number of directions (degree)
    pub n_degree: usize,
    /// Number of WDM channels
    pub n_channels: usize,
    /// Insertion loss for pass-through channels (dB)
    pub insertion_loss_db: f64,
    /// Inter-port isolation (dB)
    pub isolation_db: f64,
    /// Routing table: \[from_port\]\[channel\] → Some(to_port) | None (drop)
    pub switching_matrix: Vec<Vec<Option<usize>>>,
}

impl Roadm {
    /// Create a new ROADM with all channels in the drop state.
    pub fn new(n_degree: usize, n_channels: usize) -> Self {
        Self {
            n_degree,
            n_channels,
            insertion_loss_db: 5.0,
            isolation_db: 40.0,
            switching_matrix: vec![vec![None; n_channels]; n_degree],
        }
    }

    /// Route `channel` arriving on `from_port` to `to_port`.
    ///
    /// Returns an error if either port index is out of range.
    pub fn add_connection(
        &mut self,
        from_port: usize,
        channel: usize,
        to_port: usize,
    ) -> Result<(), crate::error::OxiPhotonError> {
        if from_port >= self.n_degree || to_port >= self.n_degree {
            return Err(crate::error::OxiPhotonError::InvalidLayer(format!(
                "port index out of range: from={from_port}, to={to_port}, degree={}",
                self.n_degree
            )));
        }
        if channel >= self.n_channels {
            return Err(crate::error::OxiPhotonError::InvalidLayer(format!(
                "channel index {channel} out of range (n_channels={})",
                self.n_channels
            )));
        }
        self.switching_matrix[from_port][channel] = Some(to_port);
        Ok(())
    }

    /// Drop `channel` on `from_port` (set routing to None).
    pub fn drop_connection(&mut self, from_port: usize, channel: usize) {
        if from_port < self.n_degree && channel < self.n_channels {
            self.switching_matrix[from_port][channel] = None;
        }
    }

    /// Return the output port for `channel` arriving on `input_port`, or `None` if dropped.
    pub fn route_channel(&self, channel: usize, input_port: usize) -> Option<usize> {
        self.switching_matrix
            .get(input_port)?
            .get(channel)?
            .as_ref()
            .copied()
    }

    /// Insertion loss for a through-path (dB).
    pub fn through_loss_db(&self) -> f64 {
        self.insertion_loss_db
    }

    /// Add/drop port additional loss relative to through path (dB).
    ///
    /// Typically ~2 dB additional for add/drop vs express.
    pub fn add_drop_loss_db(&self) -> f64 {
        self.insertion_loss_db + 2.0
    }

    /// Count the number of currently active (non-dropped) connections.
    pub fn active_connections(&self) -> usize {
        self.switching_matrix
            .iter()
            .flat_map(|row| row.iter())
            .filter(|x| x.is_some())
            .count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OpticalAmplifier
// ─────────────────────────────────────────────────────────────────────────────

/// Type of optical amplifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmplifierType {
    /// Erbium-doped fiber amplifier (C-band)
    Edfa,
    /// Semiconductor optical amplifier (O-band)
    Soa,
    /// Distributed Raman amplifier
    Raman,
}

/// Optical amplifier model (EDFA / SOA / Raman).
///
/// Models small-signal gain, noise figure, saturation, and ASE noise.
#[derive(Debug, Clone)]
pub struct OpticalAmplifier {
    /// Small-signal gain (dB)
    pub gain_db: f64,
    /// Noise figure (dB)
    pub noise_figure_db: f64,
    /// Saturation output power (dBm)
    pub saturation_power_dbm: f64,
    /// Amplifier type
    pub amplifier_type: AmplifierType,
}

impl OpticalAmplifier {
    /// Standard C-band EDFA: 20 dB gain, 5 dB NF, +17 dBm P_sat.
    pub fn edfa_c_band() -> Self {
        Self {
            gain_db: 20.0,
            noise_figure_db: 5.0,
            saturation_power_dbm: 17.0,
            amplifier_type: AmplifierType::Edfa,
        }
    }

    /// O-band SOA: 15 dB gain, 8 dB NF, +10 dBm P_sat.
    pub fn soa_o_band() -> Self {
        Self {
            gain_db: 15.0,
            noise_figure_db: 8.0,
            saturation_power_dbm: 10.0,
            amplifier_type: AmplifierType::Soa,
        }
    }

    /// Output power (dBm) for a given input power, accounting for gain saturation.
    ///
    /// Uses the Saleh model approximation:
    ///   G_sat ≈ G_0 / (1 + P_out / P_sat)
    /// Solved iteratively as: P_out = G_0·P_in / (1 + P_out/P_sat).
    pub fn output_power_dbm(&self, input_power_dbm: f64) -> f64 {
        let g0 = 10.0_f64.powf(self.gain_db / 10.0);
        let p_in_mw = 10.0_f64.powf(input_power_dbm / 10.0);
        let p_sat_mw = 10.0_f64.powf(self.saturation_power_dbm / 10.0);
        // Iterate to solve: p_out = g0 * p_in / (1 + p_out / p_sat)
        let mut p_out = g0 * p_in_mw;
        for _ in 0..50 {
            p_out = g0 * p_in_mw / (1.0 + p_out / p_sat_mw);
        }
        10.0 * p_out.max(1e-30).log10()
    }

    /// Output OSNR (dB) given input OSNR and noise bandwidth.
    ///
    /// Noise accumulation from ASE: OSNR_out = 1 / (1/OSNR_in + NF·hν·bw / P_sig).
    /// Simplified formula: OSNR_out ≈ OSNR_in − NF_db (first approximation).
    pub fn output_osnr_db(&self, input_osnr_db: f64, _bw_nm: f64) -> f64 {
        // Conservative estimate: amplifier degrades OSNR by (NF - 3) dB
        input_osnr_db - (self.noise_figure_db - 3.0).max(0.0)
    }

    /// Total ASE noise power (dBm) in a given optical bandwidth.
    ///
    /// P_ase = n_sp · (G-1) · h·ν · Δν
    /// where n_sp = F·G / (2·(G-1)) for large G ≈ F/2.
    pub fn noise_power_dbm(&self, bw_hz: f64) -> f64 {
        const H_PLANCK: f64 = 6.626e-34; // J·s
        const NU_C_BAND: f64 = 193.1e12; // Hz (reference)
        let g = 10.0_f64.powf(self.gain_db / 10.0);
        let nf = 10.0_f64.powf(self.noise_figure_db / 10.0);
        // n_sp ≈ NF * G / (2*(G-1))
        let n_sp = nf * g / (2.0 * (g - 1.0).max(1e-10));
        let p_ase_w = n_sp * (g - 1.0) * H_PLANCK * NU_C_BAND * bw_hz;
        10.0 * (p_ase_w * 1e3).max(1e-30).log10() // convert W → mW → dBm
    }

    /// Gain at the operating point including saturation.
    ///
    /// Returns the actual gain (dB) when input power is `input_power_dbm`.
    pub fn gain_at_saturation(&self, input_power_dbm: f64) -> f64 {
        let p_out_dbm = self.output_power_dbm(input_power_dbm);
        p_out_dbm - input_power_dbm
    }

    /// ASE power spectral density (W/Hz) at the output.
    ///
    ///   S_ase = n_sp · (G−1) · h·ν
    pub fn ase_psd(&self, _input_signal_power_dbm: f64) -> f64 {
        const H_PLANCK: f64 = 6.626e-34;
        const NU_C_BAND: f64 = 193.1e12;
        let g = 10.0_f64.powf(self.gain_db / 10.0);
        let nf = 10.0_f64.powf(self.noise_figure_db / 10.0);
        let n_sp = nf * g / (2.0 * (g - 1.0).max(1e-10));
        n_sp * (g - 1.0) * H_PLANCK * NU_C_BAND
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LinkBudget
// ─────────────────────────────────────────────────────────────────────────────

/// A single component in a link budget (positive = gain, negative = loss in dB).
#[derive(Debug, Clone)]
pub struct LinkComponent {
    /// Human-readable component name
    pub name: String,
    /// Net gain/loss contribution (dB): positive = gain, negative = loss
    pub gain_or_loss_db: f64,
}

/// Optical link budget calculator.
///
/// Accumulates gains and losses along a link.  Components are added in
/// signal-propagation order; the net margin at the receiver is computed
/// as TX_launch − accumulated_loss + accumulated_gain − Rx_sensitivity.
#[derive(Debug, Clone)]
pub struct LinkBudget {
    /// Components in propagation order
    pub components: Vec<LinkComponent>,
}

impl LinkBudget {
    /// Create an empty link budget.
    pub fn new() -> Self {
        Self {
            components: Vec::new(),
        }
    }

    /// Add an arbitrary component (amplifier, coupler, connector, …).
    pub fn add_component(&mut self, name: &str, gain_loss_db: f64) {
        self.components.push(LinkComponent {
            name: name.to_owned(),
            gain_or_loss_db: gain_loss_db,
        });
    }

    /// Add a fiber span.
    pub fn add_fiber(&mut self, length_km: f64, loss_db_per_km: f64) {
        self.add_component(
            &format!("Fiber {length_km:.1} km"),
            -(length_km * loss_db_per_km),
        );
    }

    /// Add an optical amplifier (gain > 0).
    pub fn add_amplifier(&mut self, gain_db: f64) {
        self.add_component("Amplifier", gain_db);
    }

    /// Sum of all positive contributions (dB).
    pub fn total_gain_db(&self) -> f64 {
        self.components
            .iter()
            .map(|c| c.gain_or_loss_db)
            .filter(|&g| g > 0.0)
            .sum()
    }

    /// Absolute value of the sum of all negative contributions (dB).
    pub fn total_loss_db(&self) -> f64 {
        -self
            .components
            .iter()
            .map(|c| c.gain_or_loss_db)
            .filter(|&g| g < 0.0)
            .sum::<f64>()
    }

    /// Net gain/loss: positive means gain exceeds loss.
    pub fn net_db(&self) -> f64 {
        self.components.iter().map(|c| c.gain_or_loss_db).sum()
    }

    /// Link margin (dB) = launch_power + net_db − receiver_sensitivity.
    ///
    /// A positive margin means the link is feasible.
    pub fn margin_db(&self, receiver_sensitivity_dbm: f64, launch_power_dbm: f64) -> f64 {
        launch_power_dbm + self.net_db() - receiver_sensitivity_dbm
    }

    /// Maximum reach (km) for a given fiber loss, launch power and sensitivity.
    ///
    /// Ignores any previously added fiber spans; uses only the non-fiber net.
    pub fn max_reach_km(
        &self,
        fiber_loss_db_per_km: f64,
        launch_power_dbm: f64,
        sensitivity_dbm: f64,
    ) -> f64 {
        if fiber_loss_db_per_km <= 0.0 {
            return f64::INFINITY;
        }
        let available_loss_db = launch_power_dbm - sensitivity_dbm + self.total_gain_db();
        (available_loss_db / fiber_loss_db_per_km).max(0.0)
    }
}

impl Default for LinkBudget {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wdm_c_band_8ch_n_channels() {
        let wdm = WdmSystem::c_band_8ch();
        assert_eq!(wdm.n_channels(), 8);
    }

    #[test]
    fn wdm_capacity_scales_with_rate() {
        let wdm = WdmSystem::c_band_8ch();
        let c1 = wdm.aggregate_capacity_gbps(100.0);
        let c2 = wdm.aggregate_capacity_gbps(200.0);
        assert!((c2 - 2.0 * c1).abs() < 1e-6);
    }

    #[test]
    fn wdm_spectral_efficiency_positive() {
        let wdm = WdmSystem::c_band_8ch();
        assert!(wdm.spectral_efficiency(100.0) > 0.0);
    }

    #[test]
    fn wdm_wavelength_range_in_c_band() {
        let wdm = WdmSystem::c_band_8ch();
        let (lmin, lmax) = wdm.wavelength_range_nm();
        assert!(lmin > 1530.0 && lmax < 1570.0, "λ=[{lmin:.1},{lmax:.1}]nm");
    }

    #[test]
    fn wdm_grid_spacing_100ghz() {
        assert!((WdmGrid::Dwdm100Ghz.spacing_hz() - 100e9).abs() < 1e6);
    }

    #[test]
    fn ring_filter_max_channels_positive() {
        let f = RingWdmFilter::dwdm_100ghz();
        let n = f.max_channels(100e9);
        assert!(n > 0);
    }

    #[test]
    fn ring_filter_adjacent_suppression_positive() {
        let f = RingWdmFilter::dwdm_100ghz();
        let supp = f.adjacent_suppression_db(100e9, 193.1e12);
        assert!(supp > 0.0);
    }

    #[test]
    fn wdm_lan_4ch_count() {
        let wdm = WdmSystem::lan_wdm_4ch();
        assert_eq!(wdm.n_channels(), 4);
    }

    #[test]
    fn wdm_occupied_bw_positive() {
        let wdm = WdmSystem::c_band_8ch();
        assert!(wdm.occupied_bandwidth_thz() > 0.0);
    }

    // ── DwdmChannelPlan ──────────────────────────────────────────────────────

    #[test]
    fn dwdm_plan_channel_count() {
        let plan = DwdmChannelPlan::new(1550.0, 100.0, 8);
        assert_eq!(plan.channel_wavelengths_nm().len(), 8);
        assert_eq!(plan.channel_frequencies_thz().len(), 8);
    }

    #[test]
    fn dwdm_plan_center_wavelength_in_range() {
        let plan = DwdmChannelPlan::new(1550.0, 100.0, 8);
        let wls = plan.channel_wavelengths_nm();
        let wl_min = wls.iter().cloned().fold(f64::INFINITY, f64::min);
        let wl_max = wls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        // Center should be roughly 1550 nm
        let center = (wl_min + wl_max) / 2.0;
        assert!((center - 1550.0).abs() < 1.0, "center={center:.3} nm");
    }

    #[test]
    fn dwdm_plan_frequencies_increasing() {
        let plan = DwdmChannelPlan::new(1550.0, 100.0, 4);
        let freqs = plan.channel_frequencies_thz();
        for w in freqs.windows(2) {
            assert!(w[1] > w[0], "frequencies should increase with index");
        }
    }

    #[test]
    fn dwdm_plan_channel_spacing_correct() {
        let spacing_ghz = 100.0_f64;
        let plan = DwdmChannelPlan::new(1550.0, spacing_ghz, 4);
        let freqs = plan.channel_frequencies_thz();
        for w in freqs.windows(2) {
            let df_ghz = (w[1] - w[0]) * 1e3; // THz → GHz
            assert!((df_ghz - spacing_ghz).abs() < 0.01, "df={df_ghz} GHz");
        }
    }

    #[test]
    fn dwdm_plan_channel_index_for_wavelength() {
        let plan = DwdmChannelPlan::new(1550.0, 100.0, 8);
        let wls = plan.channel_wavelengths_nm();
        // Ask for channel nearest to the 3rd channel's wavelength
        let idx = plan.channel_index_for_wavelength(wls[3]).unwrap();
        assert_eq!(idx, 3);
    }

    #[test]
    fn dwdm_plan_channel_index_none_for_empty() {
        let plan = DwdmChannelPlan::new(1550.0, 100.0, 0);
        assert!(plan.channel_index_for_wavelength(1550.0).is_none());
    }

    // ── CrosstalkMatrix ──────────────────────────────────────────────────────

    #[test]
    fn crosstalk_matrix_diagonal_zero() {
        let m = CrosstalkMatrix::new(4);
        for i in 0..4 {
            assert!((m.data[i * 4 + i] - 0.0).abs() < 1e-12);
        }
    }

    #[test]
    fn crosstalk_matrix_set_and_interference() {
        let mut m = CrosstalkMatrix::new(3);
        m.set_crosstalk(0, 1, -20.0); // −20 dB from ch0 → ch1
        m.set_crosstalk(2, 1, -20.0); // −20 dB from ch2 → ch1
        let ti = m.total_interference_db(1);
        // Two equal −20 dB interferers → sum = 2×10^(-2) = 10^(-1.7)
        // dB = 10·log10(0.02) ≈ −17.0 dB
        let expected = 10.0 * (2.0 * 10.0_f64.powf(-2.0)).log10();
        assert!((ti - expected).abs() < 1e-6, "ti={ti}, expected={expected}");
    }

    #[test]
    fn crosstalk_matrix_default_interference_very_low() {
        let m = CrosstalkMatrix::new(8);
        let ti = m.total_interference_db(0);
        assert!(
            ti < -50.0,
            "default off-diagonal −60 dB → low interference, got {ti}"
        );
    }

    // ── OpticalAddDropMux ────────────────────────────────────────────────────

    #[test]
    fn oadm_add_then_express() {
        let plan = DwdmChannelPlan::new(1550.0, 100.0, 4);
        let mut oadm = OpticalAddDropMux::new(plan);
        oadm.add_channel(0, 0.0);
        oadm.add_channel(1, 3.0);
        let express = oadm.express_channels();
        // Both channels present, power reduced by insertion_loss_db
        assert_eq!(express.len(), 2);
        let il = oadm.insertion_loss_db();
        assert!((il - 1.0).abs() < 1e-12);
        for (idx, pwr) in &express {
            let original = if *idx == 0 { 0.0 } else { 3.0 };
            assert!((pwr - (original - il)).abs() < 1e-10);
        }
    }

    #[test]
    fn oadm_drop_removes_from_bus() {
        let plan = DwdmChannelPlan::new(1550.0, 100.0, 4);
        let mut oadm = OpticalAddDropMux::new(plan);
        oadm.add_channel(2, -3.0);
        let dropped = oadm.drop_channel(2);
        assert!((dropped.unwrap() - (-3.0)).abs() < 1e-12);
        // Now channel 2 should be absent
        assert!(oadm.express_channels().iter().all(|(i, _)| *i != 2));
    }

    #[test]
    fn oadm_drop_absent_returns_none() {
        let plan = DwdmChannelPlan::new(1550.0, 100.0, 4);
        let mut oadm = OpticalAddDropMux::new(plan);
        assert!(oadm.drop_channel(0).is_none());
    }

    #[test]
    fn oadm_express_empty_when_no_channels() {
        let plan = DwdmChannelPlan::new(1550.0, 100.0, 4);
        let oadm = OpticalAddDropMux::new(plan);
        assert!(oadm.express_channels().is_empty());
    }

    #[test]
    fn oadm_insertion_loss_default_one_db() {
        let plan = DwdmChannelPlan::new(1550.0, 100.0, 4);
        let oadm = OpticalAddDropMux::new(plan);
        assert!((oadm.insertion_loss_db() - 1.0).abs() < 1e-12);
    }

    // ── PowerEqualizer tests ─────────────────────────────────────────────────

    #[test]
    fn power_equalizer_equalizes_to_target() {
        let mut eq = PowerEqualizer::new(4, 0.0);
        let input_powers = [-3.0_f64, -5.0, -1.0, -8.0];
        eq.equalize(&input_powers);
        for (i, &p_in) in input_powers.iter().enumerate() {
            let p_out = p_in + eq.channel_gains[i];
            assert!((p_out - 0.0).abs() < 1e-12, "ch{i}: p_out={p_out}");
        }
    }

    #[test]
    fn power_equalizer_max_variation_correct() {
        let mut eq = PowerEqualizer::new(3, 0.0);
        eq.equalize(&[-10.0, -5.0, -3.0]);
        // Gains: +10, +5, +3 → variation = 10 - 3 = 7 dB
        let var = eq.max_variation_db();
        assert!((var - 7.0).abs() < 1e-10, "variation={var}");
    }

    #[test]
    fn power_equalizer_total_gain_sum() {
        let mut eq = PowerEqualizer::new(3, 0.0);
        eq.equalize(&[-1.0, -2.0, -3.0]);
        // gains = 1, 2, 3 → total = 6
        let total = eq.total_gain_db();
        assert!((total - 6.0).abs() < 1e-10, "total_gain={total}");
    }

    // ── ROADM tests ──────────────────────────────────────────────────────────

    #[test]
    fn roadm_add_and_route_connection() {
        let mut roadm = Roadm::new(4, 8);
        roadm.add_connection(0, 3, 2).expect("valid connection");
        let out = roadm.route_channel(3, 0);
        assert_eq!(out, Some(2));
    }

    #[test]
    fn roadm_drop_connection_removes_route() {
        let mut roadm = Roadm::new(4, 8);
        roadm.add_connection(1, 5, 3).expect("valid connection");
        roadm.drop_connection(1, 5);
        assert!(roadm.route_channel(5, 1).is_none());
    }

    #[test]
    fn roadm_out_of_range_port_returns_error() {
        let mut roadm = Roadm::new(4, 8);
        let result = roadm.add_connection(10, 0, 0); // port 10 out of range
        assert!(result.is_err());
    }

    #[test]
    fn roadm_through_loss_positive() {
        let roadm = Roadm::new(4, 8);
        assert!(roadm.through_loss_db() > 0.0);
    }

    #[test]
    fn roadm_add_drop_loss_greater_than_through() {
        let roadm = Roadm::new(4, 8);
        assert!(roadm.add_drop_loss_db() > roadm.through_loss_db());
    }

    #[test]
    fn roadm_active_connections_count() {
        let mut roadm = Roadm::new(4, 8);
        roadm.add_connection(0, 0, 1).expect("ok");
        roadm.add_connection(0, 1, 2).expect("ok");
        roadm.add_connection(1, 0, 3).expect("ok");
        assert_eq!(roadm.active_connections(), 3);
    }

    // ── OpticalAmplifier tests ───────────────────────────────────────────────

    #[test]
    fn edfa_output_power_increases_with_gain() {
        let amp = OpticalAmplifier::edfa_c_band();
        let p_in = -20.0_f64; // dBm
        let p_out = amp.output_power_dbm(p_in);
        assert!(p_out > p_in, "amplifier should increase power");
    }

    #[test]
    fn edfa_gain_at_saturation_less_than_small_signal() {
        let amp = OpticalAmplifier::edfa_c_band();
        // High input power → saturated gain < small-signal gain
        let p_in_high = 10.0_f64; // dBm (near saturation)
        let gain_sat = amp.gain_at_saturation(p_in_high);
        assert!(
            gain_sat < amp.gain_db,
            "saturated gain={gain_sat} vs small-signal={}",
            amp.gain_db
        );
    }

    #[test]
    fn soa_noise_figure_higher_than_edfa() {
        let edfa = OpticalAmplifier::edfa_c_band();
        let soa = OpticalAmplifier::soa_o_band();
        assert!(soa.noise_figure_db > edfa.noise_figure_db);
    }

    #[test]
    fn amplifier_ase_psd_positive() {
        let amp = OpticalAmplifier::edfa_c_band();
        let psd = amp.ase_psd(-30.0);
        assert!(psd > 0.0, "ASE PSD should be positive, got {psd}");
    }

    #[test]
    fn amplifier_noise_power_increases_with_bandwidth() {
        let amp = OpticalAmplifier::edfa_c_band();
        let p1 = amp.noise_power_dbm(10e9); // 10 GHz
        let p2 = amp.noise_power_dbm(100e9); // 100 GHz
        assert!(p2 > p1, "wider bandwidth → more noise: p1={p1}, p2={p2}");
    }

    // ── LinkBudget tests ─────────────────────────────────────────────────────

    #[test]
    fn link_budget_net_fiber_only() {
        let mut budget = LinkBudget::new();
        budget.add_fiber(80.0, 0.2); // 80 km × 0.2 dB/km = 16 dB loss
        let net = budget.net_db();
        assert!((net + 16.0).abs() < 1e-10, "net={net}");
    }

    #[test]
    fn link_budget_net_with_amplifier() {
        let mut budget = LinkBudget::new();
        budget.add_fiber(80.0, 0.2); // -16 dB
        budget.add_amplifier(16.0); // +16 dB
        let net = budget.net_db();
        assert!(net.abs() < 1e-10, "net={net}");
    }

    #[test]
    fn link_budget_margin_positive_when_feasible() {
        let mut budget = LinkBudget::new();
        budget.add_fiber(10.0, 0.2); // -2 dB
        let margin = budget.margin_db(-28.0, 0.0); // 0 - (-2) - (-28) = 26 dB
        assert!(margin > 0.0, "margin={margin}");
    }

    #[test]
    fn link_budget_max_reach_positive() {
        let budget = LinkBudget::new();
        let reach = budget.max_reach_km(0.2, 0.0, -28.0);
        // available_loss = 0 - (-28) + 0 = 28 dB → 28/0.2 = 140 km
        assert!((reach - 140.0).abs() < 1e-6, "reach={reach}");
    }

    #[test]
    fn link_budget_total_loss_positive() {
        let mut budget = LinkBudget::new();
        budget.add_fiber(100.0, 0.2);
        budget.add_component("Connector", -0.5);
        assert!(budget.total_loss_db() > 0.0);
    }

    #[test]
    fn link_budget_total_gain_zero_for_loss_only() {
        let mut budget = LinkBudget::new();
        budget.add_fiber(50.0, 0.2);
        assert!((budget.total_gain_db() - 0.0).abs() < 1e-12);
    }
}
