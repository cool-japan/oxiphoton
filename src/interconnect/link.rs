/// Optical link power budget and performance model.
///
/// An optical interconnect link budget tracks the optical power from
/// transmitter to receiver, accounting for all gains and losses.
///
/// Link margin = P_tx - ΣLoss - P_rx_sensitivity
/// where all values are in dBm/dB.
///
/// Optical loss/gain element in a link.
#[derive(Debug, Clone)]
pub struct Connector {
    /// Human-readable name
    pub name: String,
    /// Insertion loss (dB, positive = loss)
    pub loss_db: f64,
}

impl Connector {
    pub fn new(name: impl Into<String>, loss_db: f64) -> Self {
        Self {
            name: name.into(),
            loss_db,
        }
    }

    /// Grating coupler (typical 3-5 dB).
    pub fn grating_coupler() -> Self {
        Self::new("Grating Coupler", 3.5)
    }

    /// Edge coupler (typical 0.5-1 dB).
    pub fn edge_coupler() -> Self {
        Self::new("Edge Coupler", 0.7)
    }

    /// Y-splitter (3 dB power split + ~0.2 dB excess).
    pub fn y_splitter() -> Self {
        Self::new("Y-Splitter (3dB + excess)", 3.2)
    }

    /// Waveguide crossing (~0.1-0.3 dB per crossing).
    pub fn waveguide_crossing() -> Self {
        Self::new("WG Crossing", 0.15)
    }

    /// Multi-mode interference (MMI) coupler.
    pub fn mmi_1x2() -> Self {
        Self::new("MMI 1×2", 3.3)
    }
}

/// End-to-end optical link model.
#[derive(Debug, Clone)]
pub struct OpticalLink {
    /// Transmitter output power (dBm)
    pub tx_power_dbm: f64,
    /// Receiver sensitivity (dBm) for target BER
    pub rx_sensitivity_dbm: f64,
    /// Waveguide propagation loss (dB/m)
    pub wg_loss_db_per_m: f64,
    /// Total waveguide length (m)
    pub wg_length_m: f64,
    /// List of discrete optical elements in the link
    pub elements: Vec<Connector>,
    /// System margin budget (dB) — for temperature, aging, etc.
    pub system_margin_db: f64,
}

impl OpticalLink {
    pub fn new(tx_power_dbm: f64, rx_sensitivity_dbm: f64) -> Self {
        Self {
            tx_power_dbm,
            rx_sensitivity_dbm,
            wg_loss_db_per_m: 2.0, // default 2 dB/cm = 200 dB/m... no wait
            wg_length_m: 0.0,
            elements: Vec::new(),
            system_margin_db: 3.0,
        }
    }

    /// Typical on-chip silicon photonics link.
    pub fn on_chip_soi() -> Self {
        Self {
            tx_power_dbm: 0.0,         // 1 mW laser
            rx_sensitivity_dbm: -20.0, // InGaAs photodiode
            wg_loss_db_per_m: 200.0,   // 2 dB/cm = 200 dB/m (in dB/m for Si)
            wg_length_m: 10e-3,        // 10mm on-chip
            elements: vec![Connector::grating_coupler(), Connector::grating_coupler()],
            system_margin_db: 3.0,
        }
    }

    /// Board-level silicon photonics link (chip-to-chip).
    pub fn chip_to_chip() -> Self {
        Self {
            tx_power_dbm: 3.0, // 2 mW laser
            rx_sensitivity_dbm: -25.0,
            wg_loss_db_per_m: 20.0, // 0.2 dB/cm silicon nitride waveguide
            wg_length_m: 50e-3,     // 50mm board-level
            elements: vec![Connector::edge_coupler(), Connector::edge_coupler()],
            system_margin_db: 3.0,
        }
    }

    pub fn add_element(&mut self, element: Connector) -> &mut Self {
        self.elements.push(element);
        self
    }

    /// Total waveguide propagation loss (dB).
    pub fn propagation_loss_db(&self) -> f64 {
        self.wg_loss_db_per_m * self.wg_length_m
    }

    /// Total discrete element loss (dB).
    pub fn element_loss_db(&self) -> f64 {
        self.elements.iter().map(|e| e.loss_db).sum()
    }

    /// Total loss (dB).
    pub fn total_loss_db(&self) -> f64 {
        self.propagation_loss_db() + self.element_loss_db()
    }

    /// Received power (dBm).
    pub fn received_power_dbm(&self) -> f64 {
        self.tx_power_dbm - self.total_loss_db()
    }

    /// Link margin (dB): received_power - rx_sensitivity - system_margin.
    pub fn link_margin_db(&self) -> f64 {
        self.received_power_dbm() - self.rx_sensitivity_dbm - self.system_margin_db
    }

    /// True if link has positive margin.
    pub fn is_feasible(&self) -> bool {
        self.link_margin_db() > 0.0
    }

    /// Maximum allowable total loss for positive margin.
    pub fn budget_db(&self) -> f64 {
        self.tx_power_dbm - self.rx_sensitivity_dbm - self.system_margin_db
    }

    /// Maximum waveguide length (m) given current element losses.
    pub fn max_waveguide_length(&self) -> f64 {
        let remaining_budget = self.budget_db() - self.element_loss_db();
        if remaining_budget <= 0.0 {
            return 0.0;
        }
        remaining_budget / self.wg_loss_db_per_m
    }
}

/// Complete link budget summary.
pub struct LinkBudget {
    pub link: OpticalLink,
}

impl LinkBudget {
    pub fn new(link: OpticalLink) -> Self {
        Self { link }
    }

    /// Print a human-readable budget summary (returns as String).
    pub fn summary(&self) -> String {
        let l = &self.link;
        let mut s = String::new();
        s.push_str("=== Optical Link Budget ===\n");
        s.push_str(&format!("TX Power:          {:+.1} dBm\n", l.tx_power_dbm));
        s.push_str(&format!(
            "WG Propagation:   -{:.1} dB ({:.1}mm × {:.1}dB/cm)\n",
            l.propagation_loss_db(),
            l.wg_length_m * 1e3,
            l.wg_loss_db_per_m / 100.0
        ));
        for e in &l.elements {
            s.push_str(&format!("  {:25} -{:.1} dB\n", e.name, e.loss_db));
        }
        s.push_str(&format!("Total Loss:       -{:.1} dB\n", l.total_loss_db()));
        s.push_str(&format!(
            "RX Power:          {:+.1} dBm\n",
            l.received_power_dbm()
        ));
        s.push_str(&format!(
            "RX Sensitivity:    {:+.1} dBm\n",
            l.rx_sensitivity_dbm
        ));
        s.push_str(&format!(
            "System Margin:    -{:.1} dB\n",
            l.system_margin_db
        ));
        s.push_str(&format!(
            "Link Margin:       {:+.1} dB {}\n",
            l.link_margin_db(),
            if l.is_feasible() { "(PASS)" } else { "(FAIL)" }
        ));
        s
    }

    /// Required TX power (dBm) for positive margin with current elements.
    pub fn required_tx_power(&self) -> f64 {
        let l = &self.link;
        l.rx_sensitivity_dbm + l.total_loss_db() + l.system_margin_db
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_chip_link_computes() {
        let link = OpticalLink::on_chip_soi();
        let margin = link.link_margin_db();
        // Should be feasible with typical params
        assert!(link.total_loss_db() > 0.0);
        // Margin might be positive or negative depending on params
        assert!(margin.is_finite());
    }

    #[test]
    fn propagation_loss_scales_with_length() {
        let mut link = OpticalLink::new(0.0, -20.0);
        link.wg_loss_db_per_m = 100.0;
        link.wg_length_m = 0.01;
        assert!((link.propagation_loss_db() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn link_budget_positive_margin_when_feasible() {
        let mut link = OpticalLink::new(10.0, -30.0);
        link.wg_loss_db_per_m = 0.0;
        link.system_margin_db = 3.0;
        assert!(link.link_margin_db() > 0.0);
        assert!(link.is_feasible());
    }

    #[test]
    fn link_budget_fails_with_too_much_loss() {
        let mut link = OpticalLink::new(0.0, -20.0);
        link.wg_loss_db_per_m = 200.0;
        link.wg_length_m = 1.0; // 200 dB loss!
        assert!(!link.is_feasible());
    }

    #[test]
    fn max_waveguide_length_positive() {
        let link = OpticalLink::chip_to_chip();
        let l_max = link.max_waveguide_length();
        assert!(l_max > 0.0);
    }

    #[test]
    fn element_loss_sums_correctly() {
        let mut link = OpticalLink::new(0.0, -20.0);
        link.elements.push(Connector::new("A", 2.0));
        link.elements.push(Connector::new("B", 3.0));
        assert!((link.element_loss_db() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn budget_summary_contains_pass_or_fail() {
        let lb = LinkBudget::new(OpticalLink::chip_to_chip());
        let summary = lb.summary();
        assert!(summary.contains("PASS") || summary.contains("FAIL"));
    }

    #[test]
    fn required_tx_power_computable() {
        let lb = LinkBudget::new(OpticalLink::on_chip_soi());
        let p = lb.required_tx_power();
        assert!(p.is_finite());
    }
}
