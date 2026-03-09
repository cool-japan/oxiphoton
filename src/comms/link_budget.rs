//! Optical link budget and DWDM system analysis.
//!
//! Provides:
//! - Simple point-to-point link budget with power margin calculations
//! - DWDM link budget with nonlinear threshold and Shannon capacity
//! - FEC analysis (G.709 hard-decision and soft-decision FEC)
//!
//! # References
//!
//! - ITU-T G.977 / G.978 (transoceanic DWDM)
//! - Winzer & Essiambre, "Advanced Modulation Formats for High-Capacity
//!   Optical Transport Networks", J. Lightw. Technol., 24(12):4711–4728, 2006

// ──────────────────────────────────────────────────────────────────────────────
// LinkBudget (simple point-to-point)
// ──────────────────────────────────────────────────────────────────────────────

/// Simple point-to-point optical link budget.
///
/// Models a single fiber span with connector/splice loss and computes the
/// received power, power margin, and maximum reach.
#[derive(Debug, Clone)]
pub struct LinkBudget {
    /// Transmitter launch power (dBm)
    pub tx_power_dbm: f64,
    /// Fiber attenuation coefficient (dB/km)
    pub fiber_loss_db_per_km: f64,
    /// Link length (km)
    pub link_length_km: f64,
    /// Total connector and splice loss (dB)
    pub connector_loss_db: f64,
    /// Minimum acceptable receiver input power (dBm) for target BER
    pub rx_sensitivity_dbm: f64,
}

impl LinkBudget {
    /// Construct a new link budget.
    ///
    /// # Arguments
    /// * `tx_dbm`        – transmitter launch power (dBm)
    /// * `loss_per_km`   – fiber attenuation (dB/km)
    /// * `length_km`     – link length (km)
    /// * `connector_db`  – total connector and splice loss (dB)
    /// * `rx_sens_dbm`   – receiver sensitivity (dBm)
    pub fn new(
        tx_dbm: f64,
        loss_per_km: f64,
        length_km: f64,
        connector_db: f64,
        rx_sens_dbm: f64,
    ) -> Self {
        Self {
            tx_power_dbm: tx_dbm,
            fiber_loss_db_per_km: loss_per_km,
            link_length_km: length_km,
            connector_loss_db: connector_db,
            rx_sensitivity_dbm: rx_sens_dbm,
        }
    }

    /// Total fiber loss over the link (dB).
    ///
    ///   L_fiber = α × L
    pub fn fiber_loss_total_db(&self) -> f64 {
        self.fiber_loss_db_per_km * self.link_length_km
    }

    /// Received signal power (dBm).
    ///
    ///   P_rx = P_tx − α·L − L_connector
    pub fn received_power_dbm(&self) -> f64 {
        self.tx_power_dbm - self.fiber_loss_total_db() - self.connector_loss_db
    }

    /// Power margin (dB) above the receiver sensitivity threshold.
    ///
    ///   margin = P_rx − P_rx_min
    ///
    /// A positive margin indicates a feasible link.
    pub fn power_margin_db(&self) -> f64 {
        self.received_power_dbm() - self.rx_sensitivity_dbm
    }

    /// Maximum feasible link length for zero power margin (km).
    ///
    ///   L_max = (P_tx − P_rx_min − L_connector) / α
    ///
    /// Returns 0 if the link is already infeasible (P_tx < P_rx_min + connector).
    pub fn max_length_km(&self) -> f64 {
        if self.fiber_loss_db_per_km <= 0.0 {
            return f64::INFINITY;
        }
        let available_loss = self.tx_power_dbm - self.rx_sensitivity_dbm - self.connector_loss_db;
        (available_loss / self.fiber_loss_db_per_km).max(0.0)
    }

    /// Returns `true` if the link has a positive power margin.
    pub fn is_feasible(&self) -> bool {
        self.power_margin_db() > 0.0
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// FiberLink / AmplifierSpec (supporting structs for DwdmLinkBudget)
// ──────────────────────────────────────────────────────────────────────────────

/// Fiber transmission parameters for a DWDM link.
#[derive(Debug, Clone)]
pub struct FiberLink {
    /// Fiber length per span (km)
    pub length_km: f64,
    /// Attenuation coefficient (dB/km); G.652 SMF ≈ 0.2 dB/km at 1550 nm
    pub loss_db_per_km: f64,
    /// Chromatic dispersion coefficient (ps/(nm·km)); G.652 SMF ≈ 17 ps/(nm·km)
    pub dispersion_ps_per_nm_km: f64,
    /// Effective area A_eff (µm²); G.652 SMF ≈ 80 µm²
    pub effective_area_um2: f64,
    /// Nonlinear refractive index n₂ (m²/W); fused silica ≈ 2.6 × 10⁻²⁰ m²/W
    pub nonlinear_index_n2_m2_per_w: f64,
}

impl FiberLink {
    /// Standard G.652 single-mode fiber at 1550 nm.
    pub fn smf28() -> Self {
        Self {
            length_km: 80.0,
            loss_db_per_km: 0.2,
            dispersion_ps_per_nm_km: 17.0,
            effective_area_um2: 80.0,
            nonlinear_index_n2_m2_per_w: 2.6e-20,
        }
    }

    /// Nonlinear coefficient γ (1/(W·km)).
    ///
    ///   γ = 2π·n₂ / (λ·A_eff)
    pub fn nonlinear_coefficient_per_w_km(&self, lambda_nm: f64) -> f64 {
        let lambda_m = lambda_nm * 1e-9;
        let a_eff_m2 = self.effective_area_um2 * 1e-12;
        2.0 * std::f64::consts::PI * self.nonlinear_index_n2_m2_per_w / (lambda_m * a_eff_m2)
        // Result in 1/(W·m) — convert to 1/(W·km)
        * 1e-3
    }

    /// Effective nonlinear length per span (km).
    ///
    ///   L_eff = (1 − e^{−α·L}) / α
    ///
    /// where α is in 1/km (converted from dB/km).
    pub fn effective_length_km(&self) -> f64 {
        let alpha_per_km = self.loss_db_per_km / (10.0 * std::f64::consts::LOG10_E);
        let al = alpha_per_km * self.length_km;
        (1.0 - (-al).exp()) / alpha_per_km.max(1e-15)
    }
}

/// Amplifier specification for a DWDM link.
#[derive(Debug, Clone)]
pub struct AmplifierSpec {
    /// Per-amplifier gain (dB)
    pub gain_db: f64,
    /// Per-amplifier noise figure (dB)
    pub noise_figure_db: f64,
    /// Number of amplifier spans
    pub n_spans: usize,
    /// Span length (km)
    pub span_length_km: f64,
}

impl AmplifierSpec {
    /// Standard EDFA with 20 dB gain, 5 dB NF, for 80 km spans.
    pub fn edfa_standard() -> Self {
        Self {
            gain_db: 20.0,
            noise_figure_db: 5.0,
            n_spans: 10,
            span_length_km: 80.0,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DwdmLinkBudget
// ──────────────────────────────────────────────────────────────────────────────

/// DWDM system link budget with nonlinear impairment analysis.
///
/// Computes OSNR, nonlinear threshold power, Shannon capacity, and maximum
/// transmission reach for a multi-channel DWDM system.
#[derive(Debug, Clone)]
pub struct DwdmLinkBudget {
    /// Number of WDM channels
    pub n_channels: usize,
    /// Channel spacing (GHz)
    pub channel_spacing_ghz: f64,
    /// Per-channel launch power (dBm)
    pub per_channel_power_dbm: f64,
    /// Fiber transmission parameters
    pub fiber: FiberLink,
    /// Amplifier specification
    pub amplifiers: AmplifierSpec,
}

impl DwdmLinkBudget {
    /// Construct a DWDM link budget.
    pub fn new(
        n_channels: usize,
        channel_spacing_ghz: f64,
        per_channel_power_dbm: f64,
        fiber: FiberLink,
        amplifiers: AmplifierSpec,
    ) -> Self {
        Self {
            n_channels,
            channel_spacing_ghz,
            per_channel_power_dbm,
            fiber,
            amplifiers,
        }
    }

    /// Total fiber launch power summed over all channels (dBm).
    ///
    ///   P_total = P_ch + 10·log₁₀(N)
    pub fn total_launch_power_dbm(&self) -> f64 {
        if self.n_channels == 0 {
            return f64::NEG_INFINITY;
        }
        self.per_channel_power_dbm + 10.0 * (self.n_channels as f64).log10()
    }

    /// Nonlinear threshold (NLT) power per channel (dBm).
    ///
    /// The NLT is the per-channel power at which the nonlinear penalty equals
    /// the linear penalty, often approximated as:
    ///
    ///   P_NLT = √(α·A_eff / (γ·N_spans·L_eff)) \[W\]
    ///
    /// converted to dBm.
    ///
    /// # Arguments uses fiber wavelength via field; caller supplies λ via the
    /// system OSNR call.  Here we use 1550 nm as the canonical C-band wavelength.
    pub fn nonlinear_threshold_dbm(&self) -> f64 {
        let lambda_nm = 1550.0_f64;
        let gamma = self.fiber.nonlinear_coefficient_per_w_km(lambda_nm); // 1/(W·km)
        let l_eff = self.fiber.effective_length_km(); // km
        let alpha_per_km = self.fiber.loss_db_per_km / (10.0 * std::f64::consts::LOG10_E);
        // NLT formula (Poggiolini et al., simplified):
        //   P_NLT [W] = sqrt(alpha [1/km] * A_eff [km²]) / sqrt(gamma [1/(W·km)] * N * L_eff [km])
        // We use the simplified GN model single-span approximation here.
        let a_eff_km2 = self.fiber.effective_area_um2 * 1e-18; // µm² → km²
        let n = self.amplifiers.n_spans as f64;
        let num = alpha_per_km * a_eff_km2;
        let den = gamma * n * l_eff;
        if den <= 0.0 {
            return f64::INFINITY;
        }
        let p_nlt_w_km = (num / den).sqrt(); // in W·km — convert to W
        let p_nlt_w = p_nlt_w_km / l_eff.max(1e-10);
        10.0 * (p_nlt_w * 1e3).max(1e-40).log10() // W → mW → dBm
    }

    /// Returns `true` if the per-channel power is below the nonlinear threshold.
    pub fn is_linear_regime(&self) -> bool {
        self.per_channel_power_dbm < self.nonlinear_threshold_dbm()
    }

    /// Minimum number of amplifier spans needed to achieve `target_osnr_db`.
    ///
    /// Solves the OSNR formula for N, rounding up to the nearest integer.
    ///
    /// # Arguments
    /// * `target_osnr_db` – minimum acceptable system OSNR (dB)
    /// * `lambda_nm`      – signal centre wavelength (nm)
    pub fn required_spans(&self, target_osnr_db: f64, lambda_nm: f64) -> usize {
        // Binary search over span count
        for n in 1..=10_000_usize {
            let chain = crate::comms::modulation::AmplifierChain::new(
                n,
                self.amplifiers.gain_db,
                self.amplifiers.noise_figure_db,
                self.amplifiers.span_length_km * self.fiber.loss_db_per_km,
            );
            let osnr = chain.output_osnr_db(self.per_channel_power_dbm, lambda_nm, 0.1);
            if osnr < target_osnr_db {
                return n;
            }
        }
        10_000
    }

    /// System OSNR at the end of the amplifier chain (dB).
    ///
    /// # Arguments
    /// * `lambda_nm` – signal centre wavelength (nm)
    pub fn system_osnr_db(&self, lambda_nm: f64) -> f64 {
        let chain = crate::comms::modulation::AmplifierChain::new(
            self.amplifiers.n_spans,
            self.amplifiers.gain_db,
            self.amplifiers.noise_figure_db,
            self.amplifiers.span_length_km * self.fiber.loss_db_per_km,
        );
        chain.output_osnr_db(self.per_channel_power_dbm, lambda_nm, 0.1)
    }

    /// Maximum aggregate Shannon capacity (Tbit/s).
    ///
    /// Applies the Shannon–Hartley theorem to each channel using the system OSNR
    /// as the SNR per channel:
    ///
    ///   C = N · Δf · log₂(1 + OSNR_linear)
    ///
    /// where Δf is the channel spacing (Hz) and N the number of channels.
    /// This is the idealized upper bound; real capacity is lower.
    pub fn shannon_capacity_tbps(&self) -> f64 {
        let osnr_db = self.system_osnr_db(1550.0);
        let osnr_lin = 10.0_f64.powf(osnr_db / 10.0);
        let bw_hz = self.channel_spacing_ghz * 1e9;
        let capacity_per_channel = bw_hz * (1.0 + osnr_lin).log2();
        let total_bps = self.n_channels as f64 * capacity_per_channel;
        total_bps / 1e12 // bps → Tbit/s
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// FecAnalysis
// ──────────────────────────────────────────────────────────────────────────────

/// Forward error correction (FEC) overhead and performance analysis.
///
/// Encapsulates the key parameters of an FEC scheme: overhead, net coding gain,
/// input BER threshold, and output (post-FEC) BER.
#[derive(Debug, Clone)]
pub struct FecAnalysis {
    /// FEC overhead as a percentage of the line rate (%)
    pub overhead_percent: f64,
    /// Net coding gain (dB) at the target output BER
    pub coding_gain_db: f64,
    /// Maximum pre-FEC BER the codec can correct (input threshold)
    pub input_ber_threshold: f64,
    /// Post-FEC BER achieved at the input threshold
    pub output_ber: f64,
}

impl FecAnalysis {
    /// G.709 / OTU hard-decision FEC (7% overhead, ≈8.6 dB NCG).
    ///
    /// Standardised in ITU-T G.709 (OTU framing).  Commonly used in metro and
    /// long-haul DWDM systems up to 100 Gbit/s.
    pub fn g709_hard_fec() -> Self {
        Self {
            overhead_percent: 7.0,
            coding_gain_db: 8.6,
            input_ber_threshold: 3.8e-3,
            output_ber: 1e-15,
        }
    }

    /// Soft-decision FEC (20% overhead, ≈11 dB NCG).
    ///
    /// Typical of turbo-product or LDPC codes used in coherent 100G/400G systems.
    pub fn soft_decision_fec() -> Self {
        Self {
            overhead_percent: 20.0,
            coding_gain_db: 11.0,
            input_ber_threshold: 2.0e-2,
            output_ber: 1e-15,
        }
    }

    /// Construct a custom FEC scheme.
    pub fn new(overhead_pct: f64, coding_gain_db: f64, input_ber: f64, output_ber: f64) -> Self {
        Self {
            overhead_percent: overhead_pct,
            coding_gain_db,
            input_ber_threshold: input_ber,
            output_ber,
        }
    }

    /// Effective information (payload) data rate (Gbit/s) from the line rate.
    ///
    ///   R_data = R_line / (1 + overhead / 100)
    pub fn effective_data_rate_gbps(&self, line_rate_gbps: f64) -> f64 {
        line_rate_gbps / (1.0 + self.overhead_percent / 100.0)
    }

    /// Required pre-FEC OSNR (dB).
    ///
    /// The FEC coding gain allows operating at a lower (pre-FEC) OSNR than
    /// the FEC-free system would require for the same post-FEC BER:
    ///
    ///   OSNR_pre_FEC = OSNR_fec_free − coding_gain_dB
    ///
    /// where `base_osnr_db` is the OSNR needed without FEC (at the output BER).
    pub fn required_pre_fec_osnr_db(&self, base_osnr_db: f64) -> f64 {
        base_osnr_db - self.coding_gain_db
    }

    /// Returns `true` if the pre-FEC BER is within the decodable range.
    ///
    /// The FEC can correct errors only up to `input_ber_threshold`.
    pub fn is_decodable(&self, pre_fec_ber: f64) -> bool {
        pre_fec_ber <= self.input_ber_threshold
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Received power decreases as link length increases.
    #[test]
    fn test_received_power_decreases_with_distance() {
        let short = LinkBudget::new(0.0, 0.2, 50.0, 1.0, -28.0);
        let long = LinkBudget::new(0.0, 0.2, 100.0, 1.0, -28.0);
        let p_short = short.received_power_dbm();
        let p_long = long.received_power_dbm();
        assert!(
            p_long < p_short,
            "longer link → lower received power: {p_long} vs {p_short}"
        );
    }

    /// A short link should have a positive power margin.
    #[test]
    fn test_power_margin_positive() {
        // 0 dBm tx, 0.2 dB/km × 10 km = 2 dB, 0.5 dB connector → P_rx = -2.5 dBm
        // sensitivity = -28 dBm → margin = 25.5 dB
        let lb = LinkBudget::new(0.0, 0.2, 10.0, 0.5, -28.0);
        let margin = lb.power_margin_db();
        assert!(
            margin > 0.0,
            "power margin should be positive, got {margin}"
        );
    }

    /// max_length_km formula: L = (P_tx − P_rx_min − connector) / α.
    #[test]
    fn test_max_length_formula() {
        let tx = 0.0_f64;
        let alpha = 0.2_f64;
        let connector = 1.0_f64;
        let sens = -28.0_f64;
        let lb = LinkBudget::new(tx, alpha, 0.0, connector, sens);
        let l_max = lb.max_length_km();
        let expected = (tx - sens - connector) / alpha;
        assert!(
            (l_max - expected).abs() < 1e-10,
            "L_max should be {expected:.2} km, got {l_max:.2}"
        );
    }

    /// Total launch power increases with the number of channels.
    #[test]
    fn test_total_launch_power_increases_with_channels() {
        let fiber = FiberLink::smf28();
        let amps = AmplifierSpec::edfa_standard();
        let sys8 = DwdmLinkBudget::new(8, 100.0, 0.0, fiber.clone(), amps.clone());
        let sys32 = DwdmLinkBudget::new(32, 100.0, 0.0, fiber, amps);
        let p8 = sys8.total_launch_power_dbm();
        let p32 = sys32.total_launch_power_dbm();
        assert!(
            p32 > p8,
            "more channels → higher total launch power: {p32} vs {p8}"
        );
    }

    /// Shannon capacity should be a positive, finite number.
    #[test]
    fn test_shannon_capacity_positive() {
        let fiber = FiberLink::smf28();
        let amps = AmplifierSpec::edfa_standard();
        let sys = DwdmLinkBudget::new(40, 100.0, 0.0, fiber, amps);
        let cap = sys.shannon_capacity_tbps();
        assert!(
            cap > 0.0 && cap.is_finite(),
            "Shannon capacity = {cap} Tbit/s"
        );
    }

    /// G.709 effective data rate = line_rate / 1.07.
    #[test]
    fn test_fec_g709_effective_rate() {
        let fec = FecAnalysis::g709_hard_fec();
        let line_rate = 107.0_f64; // Gbit/s (standard OTU2 line rate)
        let data_rate = fec.effective_data_rate_gbps(line_rate);
        let expected = line_rate / 1.07;
        assert!(
            (data_rate - expected).abs() < 1e-6,
            "G.709 effective rate should be {expected:.4}, got {data_rate:.4}"
        );
    }

    /// is_decodable: BER below threshold → decodable; above → not.
    #[test]
    fn test_fec_is_decodable() {
        let fec = FecAnalysis::g709_hard_fec();
        assert!(fec.is_decodable(1e-4), "BER=1e-4 should be decodable");
        assert!(
            !fec.is_decodable(1e-2),
            "BER=1e-2 should NOT be decodable by G.709"
        );
    }

    /// Soft-decision FEC has higher overhead than G.709.
    #[test]
    fn test_soft_fec_higher_overhead() {
        let hard = FecAnalysis::g709_hard_fec();
        let soft = FecAnalysis::soft_decision_fec();
        assert!(
            soft.overhead_percent > hard.overhead_percent,
            "soft-decision FEC overhead ({}) > hard ({}):",
            soft.overhead_percent,
            hard.overhead_percent
        );
    }

    /// Fiber nonlinear threshold should be a finite value in a physically
    /// meaningful dBm range (−100 … +30 dBm).
    #[test]
    fn test_nonlinear_threshold_finite() {
        let fiber = FiberLink::smf28();
        let amps = AmplifierSpec::edfa_standard();
        let sys = DwdmLinkBudget::new(80, 100.0, 0.0, fiber, amps);
        let nlt = sys.nonlinear_threshold_dbm();
        assert!(
            nlt.is_finite() && nlt > -100.0 && nlt < 30.0,
            "NLT should be in physical range −100..+30 dBm, got {nlt} dBm"
        );
    }

    /// FEC coding gain must improve the required pre-FEC OSNR (lower it).
    #[test]
    fn test_fec_reduces_required_osnr() {
        let fec = FecAnalysis::soft_decision_fec();
        let base_osnr = 15.0_f64; // dB
        let pre_fec = fec.required_pre_fec_osnr_db(base_osnr);
        assert!(
            pre_fec < base_osnr,
            "pre-FEC OSNR {pre_fec} should be less than base {base_osnr}"
        );
    }

    /// link_budget feasibility check.
    #[test]
    fn test_link_budget_feasibility() {
        let feasible = LinkBudget::new(0.0, 0.2, 10.0, 0.5, -28.0);
        let infeasible = LinkBudget::new(0.0, 0.2, 200.0, 0.5, -28.0);
        assert!(feasible.is_feasible(), "short link should be feasible");
        assert!(
            !infeasible.is_feasible(),
            "200 km link should be infeasible at 0.2 dB/km"
        );
    }
}
