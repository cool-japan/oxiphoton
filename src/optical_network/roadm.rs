//! ROADM (Reconfigurable Optical Add-Drop Multiplexer) modeling.
//!
//! Implements wavelength-selective switches (WSS), ROADM nodes, and
//! optical cross-connects (OXC) for flexible optical networking.
//!
//! # Architecture
//! A modern ROADM node consists of:
//! - Ingress/egress EDFAs per direction
//! - WSS for express switching and add/drop
//! - Optionally: colorless, directionless, contentionless (CDC) architecture
//!
//! # References
//! - P. Winzer, "Optical Networking Beyond WDM," IEEE Photonics J., 2012
//! - OIF 400ZR implementation agreement

use super::wdm_system::ItuChannelPlan;
use crate::error::{OxiPhotonError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// WavelengthSelectiveSwitch
// ─────────────────────────────────────────────────────────────────────────────

/// Wavelength-selective switch (WSS) — the key component in modern ROADMs.
///
/// A 1×N WSS can route any wavelength from its single input to any of its N
/// output ports independently. An M×N WSS generalises this to M inputs.
///
/// The switching matrix is indexed as `[output_port][channel_index]` and holds
/// `Some(input_port)` when that channel is routed, or `None` for unconnected.
#[derive(Debug, Clone)]
pub struct WavelengthSelectiveSwitch {
    /// Number of output ports (1×N: N ports; M×N: N ports).
    pub n_ports: usize,
    /// Number of input ports (1 for 1×N, M for M×N).
    pub n_input_ports: usize,
    /// Channel plan.
    pub channel_plan: ItuChannelPlan,
    /// Insertion loss \[dB\].
    pub insertion_loss_db: f64,
    /// Adjacent-channel isolation \[dB\] (typ. 35–45 dB).
    pub channel_isolation_db: f64,
    /// 3 dB passband width per channel \[GHz\] (typ. 75–80% of channel spacing).
    pub passband_width_ghz: f64,
    /// Switching matrix: `[output_port][channel]` → `Some(input_port)` or `None`.
    pub switching_matrix: Vec<Vec<Option<usize>>>,
}

impl WavelengthSelectiveSwitch {
    /// Create a 1×N WSS with default parameters for the given channel plan.
    pub fn new_1xn(n: usize, plan: ItuChannelPlan) -> Self {
        let n_channels = plan.n_channels;
        Self {
            n_ports: n,
            n_input_ports: 1,
            insertion_loss_db: 5.5,
            channel_isolation_db: 40.0,
            passband_width_ghz: plan.spacing_ghz() * 0.75,
            channel_plan: plan,
            switching_matrix: vec![vec![None; n_channels]; n],
        }
    }

    /// Create an M×N WSS.
    pub fn new_mxn(m: usize, n: usize, plan: ItuChannelPlan) -> Self {
        let n_channels = plan.n_channels;
        Self {
            n_ports: n,
            n_input_ports: m,
            insertion_loss_db: 7.0,
            channel_isolation_db: 40.0,
            passband_width_ghz: plan.spacing_ghz() * 0.75,
            channel_plan: plan,
            switching_matrix: vec![vec![None; n_channels]; n],
        }
    }

    /// Route `channel` from `input` port to `output` port.
    pub fn route(&mut self, channel: usize, input: usize, output: usize) -> Result<()> {
        self.validate_channel(channel)?;
        self.validate_input(input)?;
        self.validate_output(output)?;
        self.switching_matrix[output][channel] = Some(input);
        Ok(())
    }

    /// Drop a channel on the specified output (disconnect it).
    pub fn drop_channel(&mut self, channel: usize, output: usize) -> Result<()> {
        self.validate_channel(channel)?;
        self.validate_output(output)?;
        self.switching_matrix[output][channel] = None;
        Ok(())
    }

    /// Add a channel from `input` to `output` (alias for `route`).
    pub fn add_channel(&mut self, channel: usize, input: usize, output: usize) -> Result<()> {
        self.route(channel, input, output)
    }

    /// Passband penalty due to cascaded WSS filtering \[dB\].
    ///
    /// Each WSS narrows the effective passband. A simplified model:
    /// `ΔL_passband ≈ 0.5 × n_cascades` dB (linear approximation).
    ///
    /// A more accurate model uses: `passband_n = passband_1 × exp(-0.5 × n)`.
    pub fn passband_penalty_db(&self, n_cascades: usize) -> f64 {
        // Each cascade adds ~0.5 dB passband narrowing penalty
        0.5 * n_cascades as f64
    }

    /// Maximum number of cascaded WSS nodes before passband penalty exceeds the limit.
    ///
    /// Returns the integer number of cascades that keeps the penalty below `max_penalty_db`.
    pub fn max_cascades_before_penalty_exceeds(&self, max_penalty_db: f64) -> usize {
        if max_penalty_db <= 0.0 {
            return 0;
        }
        (max_penalty_db / 0.5).floor() as usize
    }

    /// Number of active (connected) channel-port assignments.
    pub fn active_connections(&self) -> usize {
        self.switching_matrix
            .iter()
            .flat_map(|row| row.iter())
            .filter(|c| c.is_some())
            .count()
    }

    /// Return the input port routing for `channel` on `output`, if connected.
    pub fn get_route(&self, channel: usize, output: usize) -> Option<usize> {
        self.switching_matrix
            .get(output)
            .and_then(|row| row.get(channel))
            .copied()
            .flatten()
    }

    // ── Private helpers ─────────────────────────────────────────────────────

    fn validate_channel(&self, channel: usize) -> Result<()> {
        if channel >= self.channel_plan.n_channels {
            Err(OxiPhotonError::InvalidLayer(format!(
                "channel {channel} out of range (n_channels={})",
                self.channel_plan.n_channels
            )))
        } else {
            Ok(())
        }
    }

    fn validate_input(&self, input: usize) -> Result<()> {
        if input >= self.n_input_ports {
            Err(OxiPhotonError::InvalidLayer(format!(
                "input port {input} out of range (n_input_ports={})",
                self.n_input_ports
            )))
        } else {
            Ok(())
        }
    }

    fn validate_output(&self, output: usize) -> Result<()> {
        if output >= self.n_ports {
            Err(OxiPhotonError::InvalidLayer(format!(
                "output port {output} out of range (n_ports={})",
                self.n_ports
            )))
        } else {
            Ok(())
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RoadmNode
// ─────────────────────────────────────────────────────────────────────────────

/// ROADM node with configurable degree and CDC capabilities.
///
/// A degree-N ROADM has N fiber directions (pairs of ingress/egress fiber).
/// Express channels pass through without going to the local add/drop ports.
/// Add channels are injected into the network from the local client.
/// Drop channels are extracted from the network to the local client.
#[derive(Debug, Clone)]
pub struct RoadmNode {
    /// Number of fiber directions (degree).
    pub degree: usize,
    /// Express WSS per direction pair (routes express channels).
    pub express_wss: Vec<WavelengthSelectiveSwitch>,
    /// Add WSS (local transponders → network).
    pub add_wss: WavelengthSelectiveSwitch,
    /// Drop WSS (network → local transponders).
    pub drop_wss: WavelengthSelectiveSwitch,
    /// Additional node loss budget \[dB\] (connectors, patch panels, etc.).
    pub node_loss_db: f64,
    /// Colorless add/drop architecture.
    colorless: bool,
    /// Directionless switching.
    directionless: bool,
    /// Contentionless (multiple same-wavelength channels from different directions).
    contentionless: bool,
    /// Channels currently being added at this node.
    add_channels: Vec<usize>,
    /// Channels currently being dropped at this node.
    drop_channels: Vec<usize>,
}

impl RoadmNode {
    /// Create a degree-N ROADM node with standard (non-CDC) architecture.
    pub fn new(degree: usize, plan: ItuChannelPlan) -> Self {
        let n_ch = plan.n_channels;
        // One express WSS per direction
        let express_wss = (0..degree)
            .map(|_| WavelengthSelectiveSwitch::new_1xn(degree, plan.clone()))
            .collect();

        Self {
            degree,
            express_wss,
            add_wss: WavelengthSelectiveSwitch::new_1xn(degree, plan.clone()),
            drop_wss: WavelengthSelectiveSwitch::new_1xn(degree, plan),
            node_loss_db: 3.0,
            colorless: false,
            directionless: false,
            contentionless: false,
            add_channels: Vec::with_capacity(n_ch),
            drop_channels: Vec::with_capacity(n_ch),
        }
    }

    /// Enable colorless/directionless/contentionless (CDC) capabilities.
    pub fn with_cdc(mut self) -> Self {
        self.colorless = true;
        self.directionless = true;
        self.contentionless = true;
        self
    }

    /// Configure which channels are locally added and dropped at this node.
    ///
    /// `add_channels` and `drop_channels` are channel indices into the plan.
    pub fn add_drop_channels(&mut self, add_channels: &[usize], drop_channels: &[usize]) {
        self.add_channels.clear();
        self.add_channels.extend_from_slice(add_channels);
        self.drop_channels.clear();
        self.drop_channels.extend_from_slice(drop_channels);
    }

    /// Route express channels between two directions.
    ///
    /// Routes all specified `channels` from `in_port` to `out_port`
    /// on the in_port's express WSS.
    pub fn express_channels(
        &mut self,
        channels: &[usize],
        in_port: usize,
        out_port: usize,
    ) -> Result<()> {
        if in_port >= self.degree {
            return Err(OxiPhotonError::InvalidLayer(format!(
                "in_port {in_port} >= degree {}",
                self.degree
            )));
        }
        for &ch in channels {
            self.express_wss[in_port].route(ch, 0, out_port)?;
        }
        Ok(())
    }

    /// Total node insertion loss \[dB\] (WSS loss + node overhead).
    ///
    /// Express path loss = WSS insertion loss + node_loss_db
    pub fn total_loss_db(&self) -> f64 {
        self.express_wss.first().map_or(self.node_loss_db, |w| {
            w.insertion_loss_db + self.node_loss_db
        })
    }

    /// Add/drop path loss \[dB\] (two WSS stages + node overhead).
    pub fn add_drop_loss_db(&self) -> f64 {
        self.add_wss.insertion_loss_db + self.drop_wss.insertion_loss_db + self.node_loss_db
    }

    /// Colorless: any wavelength can be added/dropped from any add/drop port.
    pub fn is_colorless(&self) -> bool {
        self.colorless
    }

    /// Directionless: traffic can be steered to any degree without
    /// wavelength blocking.
    pub fn is_directionless(&self) -> bool {
        self.directionless
    }

    /// Contentionless: simultaneous channels at the same wavelength from
    /// different directions can be handled without contention.
    pub fn is_contentionless(&self) -> bool {
        self.contentionless
    }

    /// Number of channels being locally added.
    pub fn n_add_channels(&self) -> usize {
        self.add_channels.len()
    }

    /// Number of channels being locally dropped.
    pub fn n_drop_channels(&self) -> usize {
        self.drop_channels.len()
    }

    /// Slice of locally added channel indices.
    pub fn added_channels(&self) -> &[usize] {
        &self.add_channels
    }

    /// Slice of locally dropped channel indices.
    pub fn dropped_channels(&self) -> &[usize] {
        &self.drop_channels
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OxcGranularity / OpticalCrossConnect
// ─────────────────────────────────────────────────────────────────────────────

/// Switching granularity of an optical cross-connect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OxcGranularity {
    /// Fiber-level switching: the entire fiber (all wavelengths together).
    Fiber,
    /// Waveband switching: groups of adjacent wavelengths.
    Waveband,
    /// Wavelength switching: individual wavelength granularity.
    Wavelength,
}

/// Optical cross-connect (OXC) model.
///
/// An OXC provides transparent, all-optical path switching.
/// This model computes traffic-theoretic blocking probability using an
/// Erlang-B approximation and provides loss budget estimates.
#[derive(Debug, Clone)]
pub struct OpticalCrossConnect {
    /// Number of fiber ports.
    pub n_ports: usize,
    /// Number of wavelengths (per fiber).
    pub n_wavelengths: usize,
    /// Switching granularity.
    pub switching_granularity: OxcGranularity,
}

impl OpticalCrossConnect {
    /// Create a new OXC.
    pub fn new(n_ports: usize, n_wl: usize, gran: OxcGranularity) -> Self {
        Self {
            n_ports,
            n_wavelengths: n_wl,
            switching_granularity: gran,
        }
    }

    /// Blocking probability using the Erlang-B formula (simplified, iterative).
    ///
    /// For a single wavelength / circuit-switched resource with `C` circuits
    /// and offered load `A` Erlangs:
    /// ```text
    ///   B = (A^C / C!) / Σ_{k=0}^{C} (A^k / k!)
    /// ```
    /// Here `C = n_wavelengths` for wavelength-granular OXC.
    pub fn blocking_probability(&self, load_erlangs: f64) -> f64 {
        let c = match self.switching_granularity {
            OxcGranularity::Wavelength => self.n_wavelengths,
            OxcGranularity::Waveband => (self.n_wavelengths / 4).max(1),
            OxcGranularity::Fiber => 1,
        };
        erlang_b(load_erlangs, c)
    }

    /// Total switch port count (ingress + egress).
    pub fn port_count(&self) -> usize {
        self.n_ports * 2
    }

    /// Estimated insertion loss \[dB\] based on port count.
    ///
    /// Waveguide-based OXC loss scales approximately as `3 + log2(N)` dB.
    pub fn insertion_loss_db(&self) -> f64 {
        3.0 + (self.n_ports as f64).log2()
    }

    /// Maximum throughput capacity \[Tb/s\] assuming `bits_per_channel` \[Gb/s\] per wavelength.
    pub fn max_throughput_tbps(&self, bits_per_channel_gbps: f64) -> f64 {
        self.n_ports as f64 * self.n_wavelengths as f64 * bits_per_channel_gbps / 1e3
    }
}

/// Erlang-B blocking probability (iterative method, numerically stable).
fn erlang_b(a: f64, c: usize) -> f64 {
    if c == 0 {
        return 1.0;
    }
    if a <= 0.0 {
        return 0.0;
    }
    // Recursive formula: B(A, C) = (A·B(A,C-1)) / (C + A·B(A,C-1))
    let mut b = 1.0_f64;
    for k in 1..=(c as u64) {
        b = a * b / (k as f64 + a * b);
    }
    b
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optical_network::wdm_system::ItuChannelPlan;
    use approx::assert_abs_diff_eq;

    fn test_plan() -> ItuChannelPlan {
        ItuChannelPlan::new_c_band_100ghz()
    }

    #[test]
    fn wss_1xn_creation() {
        let wss = WavelengthSelectiveSwitch::new_1xn(8, test_plan());
        assert_eq!(wss.n_ports, 8);
        assert_eq!(wss.n_input_ports, 1);
        assert_eq!(wss.active_connections(), 0);
    }

    #[test]
    fn wss_route_and_get() {
        let mut wss = WavelengthSelectiveSwitch::new_1xn(4, test_plan());
        wss.route(5, 0, 2).expect("valid route");
        assert_eq!(wss.get_route(5, 2), Some(0));
        assert_eq!(wss.get_route(5, 1), None);
    }

    #[test]
    fn wss_drop_channel() {
        let mut wss = WavelengthSelectiveSwitch::new_1xn(4, test_plan());
        wss.route(3, 0, 1).expect("ok");
        wss.drop_channel(3, 1).expect("ok");
        assert_eq!(wss.get_route(3, 1), None);
    }

    #[test]
    fn wss_invalid_channel_returns_error() {
        let mut wss = WavelengthSelectiveSwitch::new_1xn(4, test_plan());
        let result = wss.route(999, 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn wss_passband_penalty_linear() {
        let wss = WavelengthSelectiveSwitch::new_1xn(4, test_plan());
        assert_abs_diff_eq!(wss.passband_penalty_db(10), 5.0, epsilon = 1e-9);
    }

    #[test]
    fn wss_max_cascades() {
        let wss = WavelengthSelectiveSwitch::new_1xn(4, test_plan());
        // 3 dB max penalty / 0.5 dB per cascade = 6 cascades
        assert_eq!(wss.max_cascades_before_penalty_exceeds(3.0), 6);
    }

    #[test]
    fn roadm_node_creation() {
        let node = RoadmNode::new(4, test_plan());
        assert_eq!(node.degree, 4);
        assert_eq!(node.express_wss.len(), 4);
        assert!(!node.is_colorless());
    }

    #[test]
    fn roadm_node_cdc() {
        let node = RoadmNode::new(4, test_plan()).with_cdc();
        assert!(node.is_colorless());
        assert!(node.is_directionless());
        assert!(node.is_contentionless());
    }

    #[test]
    fn roadm_add_drop_channels() {
        let mut node = RoadmNode::new(4, test_plan());
        node.add_drop_channels(&[0, 1, 2], &[5, 6]);
        assert_eq!(node.n_add_channels(), 3);
        assert_eq!(node.n_drop_channels(), 2);
    }

    #[test]
    fn roadm_express_valid_route() {
        let mut node = RoadmNode::new(4, test_plan());
        let result = node.express_channels(&[0, 1], 0, 2);
        assert!(result.is_ok());
    }

    #[test]
    fn roadm_total_loss_positive() {
        let node = RoadmNode::new(4, test_plan());
        assert!(node.total_loss_db() > 0.0);
    }

    #[test]
    fn oxc_blocking_erlang_b() {
        let oxc = OpticalCrossConnect::new(8, 40, OxcGranularity::Wavelength);
        let bp = oxc.blocking_probability(10.0);
        // With 40 circuits and 10 Erlangs, blocking should be very low
        assert!((0.0..=1.0).contains(&bp));
        assert!(bp < 0.01, "blocking={bp:.6}");
    }

    #[test]
    fn oxc_blocking_increases_with_load() {
        let oxc = OpticalCrossConnect::new(4, 4, OxcGranularity::Wavelength);
        let bp_low = oxc.blocking_probability(1.0);
        let bp_high = oxc.blocking_probability(10.0);
        assert!(bp_high > bp_low);
    }

    #[test]
    fn oxc_insertion_loss_positive() {
        let oxc = OpticalCrossConnect::new(32, 80, OxcGranularity::Wavelength);
        assert!(oxc.insertion_loss_db() > 0.0);
    }

    #[test]
    fn erlang_b_zero_load() {
        // Zero load → zero blocking
        assert_abs_diff_eq!(erlang_b(0.0, 10), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn erlang_b_zero_circuits() {
        // Zero circuits → always blocked
        assert_abs_diff_eq!(erlang_b(5.0, 0), 1.0, epsilon = 1e-9);
    }
}
