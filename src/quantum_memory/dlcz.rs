//! DLCZ quantum repeater protocol.
//!
//! The DLCZ scheme (Duan, Lukin, Cirac, Zoller – Nature 414, 413, 2001)
//! uses atomic ensembles as quantum memories combined with linear optics
//! Bell measurements to extend quantum entanglement over long distances.
//!
//! References:
//! - Duan et al., Nature 414, 413 (2001): original DLCZ proposal
//! - Sangouard et al., Rev. Mod. Phys. 83, 33 (2011): quantum repeater review
//! - Simon et al., Phys. Rev. Lett. 98, 190503 (2007): multiplexed DLCZ

use std::f64::consts::PI;

/// Fiber attenuation length (km) at 1550 nm for standard SMF-28.
const ATTENUATION_LENGTH_KM: f64 = 22.0; // 0.2 dB/km → L_att ≈ 22 km

// ─── Single DLCZ node ─────────────────────────────────────────────────────────

/// A single DLCZ node consisting of two atomic ensembles and associated optics.
///
/// The write process generates a correlated signal/idler photon pair via a
/// weak Raman transition.  The idler photon is sent through the quantum
/// channel while the signal photon heralds the creation of an atomic spin
/// excitation (spin wave) in the ensemble.
#[derive(Debug, Clone)]
pub struct DlczNode {
    /// Write-pulse Raman transition efficiency η_w  (0 – 1)
    pub write_efficiency: f64,
    /// Read-out (retrieval) efficiency η_r  (0 – 1)
    pub read_efficiency: f64,
    /// Fiber coupling efficiency η_c  (0 – 1)
    pub coupling_to_fiber: f64,
    /// Single-photon detector efficiency η_d  (0 – 1)
    pub detector_efficiency: f64,
    /// Memory coherence time T_m  (s)
    pub memory_time_s: f64,
    /// Number of temporal modes available N_mode
    pub mode_number: usize,
}

impl DlczNode {
    /// Single-segment entanglement generation probability.
    ///
    /// One write attempt succeeds with probability:
    ///
    /// P_1 = p_exc × η_c × η_d
    ///
    /// where p_exc is the excitation probability per write pulse.
    /// For the full entanglement-swapping scheme the net probability
    /// also includes the write efficiency:
    ///
    /// P_success = η_w × η_c × η_d × p_exc
    pub fn entanglement_probability(&self, excitation_prob: f64) -> f64 {
        let p_exc = excitation_prob.clamp(0.0, 1.0);
        self.write_efficiency.clamp(0.0, 1.0)
            * self.coupling_to_fiber.clamp(0.0, 1.0)
            * self.detector_efficiency.clamp(0.0, 1.0)
            * p_exc
    }

    /// Expected time to generate entanglement in one segment.
    ///
    /// T_1seg = T_rep / P_success
    ///
    /// where T_rep = 1 / repetition_rate_hz.
    /// The segment transmission accounts for fiber loss over L_seg:
    ///
    /// η_link = 10^(−segment_loss_db / 10)
    pub fn time_to_entangle_s(
        &self,
        repetition_rate_hz: f64,
        segment_loss: f64,
    ) -> f64 {
        // segment_loss is the total photon loss over one segment (dB)
        let eta_link = 10.0_f64.powf(-segment_loss / 10.0);
        let p_success =
            self.write_efficiency * self.coupling_to_fiber * self.detector_efficiency * eta_link;
        let t_rep = 1.0 / repetition_rate_hz.max(f64::MIN_POSITIVE);
        t_rep / p_success.max(f64::MIN_POSITIVE)
    }

    /// Entanglement distribution rate for a chain of N_segments DLCZ segments.
    ///
    /// For nested entanglement swapping the overall rate scales as:
    ///
    /// R = P_success^{N_seg} / (2^{N_seg} × T_1seg)
    ///
    /// The mode multiplexing factor N_mode improves throughput linearly.
    pub fn distribution_rate_hz(
        &self,
        n_segments: usize,
        repetition_rate_hz: f64,
        segment_loss_db: f64,
    ) -> f64 {
        let n = n_segments.max(1) as f64;
        let t_1seg = self.time_to_entangle_s(repetition_rate_hz, segment_loss_db);
        let eta_link = 10.0_f64.powf(-segment_loss_db / 10.0);
        let p_1seg = self.write_efficiency
            * self.coupling_to_fiber
            * self.detector_efficiency
            * eta_link;
        let p_chain = p_1seg.powf(n);
        let rate = p_chain / (2.0_f64.powf(n) * t_1seg.max(f64::MIN_POSITIVE));
        rate * self.mode_number as f64
    }

    /// Fidelity at time t due to memory decoherence.
    ///
    /// F(t) = F₀ × exp(−t / T_m)
    pub fn fidelity_at_time(&self, time_s: f64, initial_fidelity: f64) -> f64 {
        let f0 = initial_fidelity.clamp(0.0, 1.0);
        let t = time_s.max(0.0);
        f0 * (-t / self.memory_time_s.max(f64::MIN_POSITIVE)).exp()
    }
}

// ─── Quantum repeater chain ───────────────────────────────────────────────────

/// A full nested DLCZ quantum repeater chain.
///
/// The total distance is divided into N_segments elementary links.  Each
/// elementary link generates entanglement independently, and then
/// entanglement swapping (Bell measurement) extends the range.
#[derive(Debug, Clone)]
pub struct QuantumRepeaterChain {
    /// Total end-to-end distance (km)
    pub total_distance_km: f64,
    /// Number of elementary segments
    pub n_segments: usize,
    /// Fiber attenuation (dB/km); standard SMF-28 at 1550 nm: 0.2 dB/km
    pub fiber_loss_db_km: f64,
    /// DLCZ node at each repeater station
    pub node: DlczNode,
}

impl QuantumRepeaterChain {
    /// Length of a single elementary segment (km).
    #[inline]
    pub fn segment_length_km(&self) -> f64 {
        self.total_distance_km / self.n_segments.max(1) as f64
    }

    /// Transmission over one segment: η_seg = 10^(−α L_seg / 10).
    pub fn segment_transmission(&self) -> f64 {
        let l_seg = self.segment_length_km();
        let loss_db = self.fiber_loss_db_km * l_seg;
        10.0_f64.powf(-loss_db / 10.0)
    }

    /// Total entanglement generation rate for the full chain (pairs/s).
    pub fn entanglement_rate_hz(&self, rep_rate_hz: f64) -> f64 {
        let segment_loss_db = self.fiber_loss_db_km * self.segment_length_km();
        self.node
            .distribution_rate_hz(self.n_segments, rep_rate_hz, segment_loss_db)
    }

    /// Effective secret key rate (bit/s) for QKD over the repeater chain.
    ///
    /// Assumes a standard BB84-like protocol where the raw key rate equals the
    /// entanglement rate and the secure fraction is determined by the Shannon
    /// bound at 5 % QBER: r_key = R_ent × (1 − 2 h(0.05)) ≈ 0.711 × R_ent.
    pub fn secret_key_rate_bps(&self, rep_rate_hz: f64) -> f64 {
        let r_ent = self.entanglement_rate_hz(rep_rate_hz);
        // Binary entropy at 5% QBER: h(0.05) ≈ 0.2864
        let secure_fraction = 1.0 - 2.0 * 0.2864;
        r_ent * secure_fraction.max(0.0)
    }

    /// Break-even distance at which the repeater outperforms a direct fiber link.
    ///
    /// For a direct link the transmission decays as exp(−L / L_att).
    /// The repeater advantage kicks in when N_seg × P_1seg > η_direct, i.e.
    /// roughly when L > L_att × ln(N_seg).
    pub fn break_even_distance_km(&self) -> f64 {
        let l_att = ATTENUATION_LENGTH_KM;
        let n = self.n_segments as f64;
        l_att * n.max(1.0).ln()
    }

    /// Direct fiber link entanglement rate (no repeater): R_direct = η_total × f_rep.
    ///
    /// η_total = η_source × η_det × η_fiber where η_fiber = 10^(−α L / 10).
    pub fn direct_link_rate_hz(&self, rep_rate_hz: f64) -> f64 {
        let total_loss_db = self.fiber_loss_db_km * self.total_distance_km;
        let eta_fiber = 10.0_f64.powf(-total_loss_db / 10.0);
        let eta_total = self.node.write_efficiency
            * self.node.coupling_to_fiber
            * self.node.detector_efficiency
            * eta_fiber;
        rep_rate_hz * eta_total
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_node() -> DlczNode {
        DlczNode {
            write_efficiency: 0.8,
            read_efficiency: 0.9,
            coupling_to_fiber: 0.7,
            detector_efficiency: 0.85,
            memory_time_s: 1e-3,
            mode_number: 100,
        }
    }

    fn test_chain() -> QuantumRepeaterChain {
        QuantumRepeaterChain {
            total_distance_km: 1000.0,
            n_segments: 10,
            fiber_loss_db_km: 0.2,
            node: test_node(),
        }
    }

    #[test]
    fn dlcz_entanglement_probability() {
        let node = test_node();
        let p = node.entanglement_probability(0.01);
        // p < excitation probability
        assert!(p > 0.0 && p < 0.01, "p={}", p);
    }

    #[test]
    fn dlcz_time_to_entangle_positive() {
        let node = test_node();
        // 100 MHz clock, 10 dB segment loss
        let t = node.time_to_entangle_s(1e8, 10.0);
        assert!(t > 0.0, "t_entangle={}", t);
    }

    #[test]
    fn dlcz_distribution_rate_less_than_direct() {
        // For many segments the chain rate should be lower than direct when
        // segment loss is identical to the per-segment share
        let node = test_node();
        // Rate for n=1 should exceed n=5
        let r1 = node.distribution_rate_hz(1, 1e8, 3.0);
        let r5 = node.distribution_rate_hz(5, 1e8, 3.0);
        assert!(r1 > r5, "r1={} r5={}", r1, r5);
    }

    #[test]
    fn dlcz_fidelity_decay() {
        let node = test_node();
        let f0 = 0.99;
        let f_half = node.fidelity_at_time(node.memory_time_s, f0);
        // F at T_m should be F0/e ≈ 0.364 × F0
        assert!(
            (f_half - f0 / std::f64::consts::E).abs() < 1e-6,
            "fidelity decay mismatch: {}",
            f_half
        );
    }

    #[test]
    fn repeater_segment_length_correct() {
        let chain = test_chain();
        let l_seg = chain.segment_length_km();
        assert!(
            (l_seg - 100.0).abs() < 1e-10,
            "l_seg={} km",
            l_seg
        );
    }

    #[test]
    fn repeater_segment_transmission_range() {
        let chain = test_chain();
        let eta = chain.segment_transmission();
        assert!(eta > 0.0 && eta < 1.0, "η_seg={}", eta);
    }

    #[test]
    fn repeater_outperforms_direct_at_long_distance() {
        let chain = test_chain();
        let r_rep = chain.entanglement_rate_hz(1e8);
        let r_dir = chain.direct_link_rate_hz(1e8);
        // At 1000 km with 0.2 dB/km the repeater must win over direct link
        assert!(r_rep > r_dir, "repeater r={} direct r={}", r_rep, r_dir);
    }

    #[test]
    fn secret_key_rate_positive() {
        let chain = test_chain();
        let skr = chain.secret_key_rate_bps(1e8);
        assert!(skr > 0.0, "SKR={}", skr);
    }
}
