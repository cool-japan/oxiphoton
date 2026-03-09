//! Topological edge state analysis for photonic systems.
//!
//! This module provides:
//! - `TopologicalEdgeState` — a localised midgap mode with chirality
//! - `PhotonicTopologicalInsulator` — interface between two topological domains
//! - `AnomalousQhpc` — anomalous quantum Hall photonic crystal
//!
//! ## Physical background
//!
//! In a Chern insulator the bulk-edge correspondence guarantees that the number
//! of chiral (unidirectional) edge modes equals the Chern number difference
//! |C_right − C_left| between the two domains.  Chiral modes cannot backscatter
//! because there is no counter-propagating mode at the same energy within the gap.
//!
//! For valley-Hall systems the protection is partial (valley-momentum locking)
//! but still yields high transmission around sharp bends provided the perturbation
//! does not mix the two valleys.

use std::f64::consts::PI;

// ─── Topological edge state ────────────────────────────────────────────────────

/// A localised topological edge state.
///
/// Edge states arise at the interface between two topological phases and are
/// characterised by their energy (within the bulk gap), their exponential
/// localisation length, their chirality (propagation direction), and the
/// gap centre they live in.
#[derive(Debug, Clone)]
pub struct TopologicalEdgeState {
    /// Eigenenergy of the edge state (same units as the Hamiltonian).
    pub energy: f64,
    /// Exponential localisation length ξ (in units of lattice sites).
    ///
    /// |ψ(n)| ∝ exp(−n/ξ) for large n.
    pub localization_xi: f64,
    /// Chirality: +1 (right-moving) or −1 (left-moving).
    ///
    /// Relevant for Chern-insulator edge modes; 0 means non-chiral (e.g. SSH).
    pub chirality: i8,
    /// Mid-gap reference energy.
    pub gap_center: f64,
}

impl TopologicalEdgeState {
    /// Construct a topological edge state.
    ///
    /// # Arguments
    /// * `energy`          – eigenenergy of the state
    /// * `localization_xi` – localisation length (lattice sites)
    /// * `chirality`       – +1, 0, or −1
    /// * `gap_center`      – centre of the bulk gap
    pub fn new(energy: f64, localization_xi: f64, chirality: i8, gap_center: f64) -> Self {
        Self {
            energy,
            localization_xi,
            chirality,
            gap_center,
        }
    }

    /// Spatial probability density profile |ψ(n)|² along the edge.
    ///
    /// Uses the normalised exponential decay:
    ///   |ψ(n)|² = A exp(−2n/ξ)
    ///
    /// where A is chosen so that Σ_n |ψ(n)|² = 1 over `n_sites` sites:
    ///   A = (1 − exp(−2/ξ))  [geometric series normalisation]
    ///
    /// Returns a vector of length `n_sites`.
    pub fn localization_profile(&self, n_sites: usize) -> Vec<f64> {
        if n_sites == 0 || self.localization_xi <= 0.0 {
            return vec![0.0; n_sites];
        }
        // Normalisation factor from geometric series
        let decay_per_site = (-2.0 / self.localization_xi).exp();
        let norm = if decay_per_site < 1.0 - 1e-15 {
            1.0 - decay_per_site
        } else {
            // Very large ξ → uniform distribution
            1.0 / n_sites as f64
        };
        (0..n_sites)
            .map(|n| norm * ((-2.0 * n as f64) / self.localization_xi).exp())
            .collect()
    }

    /// Physical penetration depth (metres) given the lattice constant.
    ///
    /// L_pen = ξ × a
    pub fn penetration_depth_m(&self, lattice_constant_m: f64) -> f64 {
        self.localization_xi * lattice_constant_m
    }

    /// Returns `true` if the state energy lies within a quarter-gap-width of
    /// the gap centre (i.e. is genuinely mid-gap).
    ///
    /// The criterion is |E − gap_centre| < gap_width / 4.
    pub fn is_midgap(&self, gap_width: f64) -> bool {
        (self.energy - self.gap_center).abs() < gap_width / 4.0
    }

    /// Returns `true` if the edge state is chiral (chirality ≠ 0).
    pub fn is_chiral(&self) -> bool {
        self.chirality != 0
    }

    /// Returns `true` if backscattering is suppressed (chiral mode).
    ///
    /// Chiral modes have no counter-propagating partner within the gap and
    /// therefore cannot backscatter off any perturbation that preserves the gap.
    pub fn backscattering_suppressed(&self) -> bool {
        self.is_chiral()
    }
}

// ─── Photonic topological insulator ───────────────────────────────────────────

/// Photonic topological insulator (PTI) formed at a domain wall between two
/// topologically distinct photonic crystals.
///
/// The number of interfacial modes is given by the bulk-edge correspondence:
///   N_interface = |C_right − C_left|
///
/// The interface modes are chiral and lie within the shared photonic band gap.
#[derive(Debug, Clone)]
pub struct PhotonicTopologicalInsulator {
    /// Chern number of the left domain.
    pub n_left: i32,
    /// Chern number of the right domain.
    pub n_right: i32,
    /// Shared photonic band gap: (f_low, f_high) in Hz.
    pub frequency_gap: (f64, f64),
}

impl PhotonicTopologicalInsulator {
    /// Construct a PTI interface.
    ///
    /// # Arguments
    /// * `n_left`        – Chern number of the left domain
    /// * `n_right`       – Chern number of the right domain
    /// * `frequency_gap` – shared band gap (f_low, f_high) in Hz
    pub fn new(n_left: i32, n_right: i32, frequency_gap: (f64, f64)) -> Self {
        Self {
            n_left,
            n_right,
            frequency_gap,
        }
    }

    /// Number of topological interface modes (bulk-edge correspondence).
    ///
    /// N = |C_right − C_left|
    pub fn n_interface_states(&self) -> i32 {
        (self.n_right - self.n_left).abs()
    }

    /// Mid-gap frequency (arithmetic mean of the band-gap edges) in Hz.
    pub fn midgap_frequency(&self) -> f64 {
        (self.frequency_gap.0 + self.frequency_gap.1) / 2.0
    }

    /// Band-gap bandwidth in Hz.
    pub fn bandwidth_hz(&self) -> f64 {
        self.frequency_gap.1 - self.frequency_gap.0
    }

    /// Returns `true` if the interface states are topologically protected.
    ///
    /// Protected means N_interface > 0 (the two sides have different Chern numbers).
    pub fn is_protected(&self) -> bool {
        self.n_interface_states() > 0
    }

    /// Approximate group velocity of the chiral edge mode (m/s).
    ///
    /// Estimated from a linear dispersion across the full Brillouin zone:
    ///   v_g ≈ (Δω × a) / (2π)
    ///
    /// where Δω = 2π × bandwidth_hz and a is the lattice constant.
    /// We take a = 1 m (normalised) and return the dimensional velocity as
    /// v_g = Δω / (2π / a) = bandwidth_hz × a.
    ///
    /// For a physical system, multiply the result by the actual lattice constant.
    ///
    /// # Arguments
    /// * `freq_hz` – operating frequency (unused in this linear approximation
    ///   but retained for API consistency with nonlinear models)
    pub fn edge_mode_group_velocity(&self, freq_hz: f64) -> f64 {
        // Linear dispersion: v_g = Δω / (BZ size) × a
        // Normalised to lattice constant a = 1 µm (representative)
        let _ = freq_hz; // linear model: v_g independent of frequency
        let a = 1e-6; // 1 µm lattice constant
        let delta_omega = 2.0 * PI * self.bandwidth_hz();
        let bz_size = 2.0 * PI / a; // |BZ| = 2π/a
        delta_omega / bz_size // = bandwidth_hz × a
    }

    /// Returns `true` if backscattering is suppressed for the interface modes.
    ///
    /// Equivalent to `is_protected()`.
    pub fn backscattering_suppressed(&self) -> bool {
        self.is_protected()
    }

    /// Fractional bandwidth: Δf / f_centre.
    pub fn fractional_bandwidth(&self) -> f64 {
        let centre = self.midgap_frequency();
        if centre.abs() < 1e-30 {
            return 0.0;
        }
        self.bandwidth_hz() / centre
    }
}

// ─── Anomalous quantum Hall photonic crystal ───────────────────────────────────

/// Anomalous quantum Hall effect photonic crystal (AQHPC).
///
/// A gyrotropic photonic crystal with time-reversal symmetry breaking (e.g. via
/// a magneto-optical effect or synthetic gauge field) can realise a non-zero
/// Chern number without an external magnetic field.  The unidirectional edge
/// modes are completely backscattering-immune.
#[derive(Debug, Clone)]
pub struct AnomalousQhpc {
    /// Lattice type: "honeycomb", "kagome", or "square".
    pub lattice: &'static str,
    /// Chern number of the occupied band(s).
    pub chern_number: i32,
    /// Fractional band gap Δω/ω₀.
    pub band_gap_fraction: f64,
}

impl AnomalousQhpc {
    /// Construct an AQHPC with the given lattice, Chern number, and gap.
    pub fn new(lattice: &'static str, chern_number: i32, band_gap_fraction: f64) -> Self {
        Self {
            lattice,
            chern_number,
            band_gap_fraction,
        }
    }

    /// Canonical honeycomb gyrotropic photonic crystal.
    ///
    /// Parameters are representative of magneto-optical honeycomb PhCs with
    /// a Chern number C = 1 and a gap fraction ≈ 0.1.
    pub fn honeycomb_gyrotropic() -> Self {
        Self {
            lattice: "honeycomb",
            chern_number: 1,
            band_gap_fraction: 0.10,
        }
    }

    /// Kagome AQHPC with flat band and C = 1.
    pub fn kagome_flat_band() -> Self {
        Self {
            lattice: "kagome",
            chern_number: 1,
            band_gap_fraction: 0.05,
        }
    }

    /// Square-lattice AQHPC with C = 2 (two occupied bands).
    pub fn square_double_chern() -> Self {
        Self {
            lattice: "square",
            chern_number: 2,
            band_gap_fraction: 0.08,
        }
    }

    /// Number of unidirectional (chiral) edge modes per edge = |C|.
    pub fn unidirectional_edge_mode_count(&self) -> i32 {
        self.chern_number.abs()
    }

    /// Returns `true` — all transmission is immune to backscattering in a
    /// Chern insulator when the perturbation preserves the gap.
    pub fn transmission_backscattering_immune(&self) -> bool {
        self.chern_number != 0
    }

    /// Topological protection of the gap: `true` if Chern number ≠ 0.
    pub fn is_topologically_non_trivial(&self) -> bool {
        self.chern_number != 0
    }

    /// Estimated quality factor of the edge-mode transmission resonance.
    ///
    /// Q ≈ f / Δf where Δf ≈ band_gap_fraction × f.
    /// Returns a dimensionless ratio.
    pub fn edge_mode_q_factor(&self) -> f64 {
        if self.band_gap_fraction < 1e-15 {
            return 0.0;
        }
        1.0 / self.band_gap_fraction
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── TopologicalEdgeState ────────────────────────────────────────────────

    #[test]
    fn edge_state_localization_profile_normalised() {
        let state = TopologicalEdgeState::new(0.0, 3.0, 1, 0.0);
        let profile = state.localization_profile(50);
        let total: f64 = profile.iter().sum();
        // Normalisation should be close to 1 for large n_sites relative to ξ
        assert!((total - 1.0).abs() < 0.05, "Total probability = {total}");
    }

    #[test]
    fn edge_state_profile_decays_exponentially() {
        let state = TopologicalEdgeState::new(0.0, 2.0, 1, 0.0);
        let profile = state.localization_profile(20);
        // Profile should be monotonically decreasing
        for i in 1..profile.len() {
            assert!(
                profile[i] <= profile[i - 1] + 1e-15,
                "Profile not monotone at index {i}: {} > {}",
                profile[i],
                profile[i - 1]
            );
        }
    }

    #[test]
    fn edge_state_penetration_depth() {
        let state = TopologicalEdgeState::new(0.0, 4.0, 1, 0.0);
        let a = 500e-9; // 500 nm lattice constant
        let depth = state.penetration_depth_m(a);
        assert!((depth - 4.0 * a).abs() < 1e-20, "depth = {depth}");
    }

    #[test]
    fn edge_state_is_midgap() {
        let state = TopologicalEdgeState::new(0.05, 3.0, 1, 0.0);
        assert!(state.is_midgap(1.0)); // |0.05 - 0| = 0.05 < 1/4 = 0.25
        assert!(!state.is_midgap(0.1)); // 0.05 > 0.1/4 = 0.025
    }

    #[test]
    fn edge_state_chirality() {
        let chiral = TopologicalEdgeState::new(0.0, 3.0, 1, 0.0);
        let non_chiral = TopologicalEdgeState::new(0.0, 3.0, 0, 0.0);
        assert!(chiral.is_chiral());
        assert!(chiral.backscattering_suppressed());
        assert!(!non_chiral.is_chiral());
        assert!(!non_chiral.backscattering_suppressed());
    }

    // ── PhotonicTopologicalInsulator ────────────────────────────────────────

    #[test]
    fn pti_n_interface_states() {
        let pti = PhotonicTopologicalInsulator::new(0, 1, (190e12, 210e12));
        assert_eq!(pti.n_interface_states(), 1);
    }

    #[test]
    fn pti_n_interface_states_chern_difference() {
        let pti = PhotonicTopologicalInsulator::new(-1, 2, (190e12, 210e12));
        assert_eq!(pti.n_interface_states(), 3);
    }

    #[test]
    fn pti_midgap_frequency() {
        let pti = PhotonicTopologicalInsulator::new(0, 1, (190e12, 210e12));
        assert!((pti.midgap_frequency() - 200e12).abs() < 1e6);
    }

    #[test]
    fn pti_is_protected() {
        let protected = PhotonicTopologicalInsulator::new(0, 1, (190e12, 210e12));
        let trivial = PhotonicTopologicalInsulator::new(1, 1, (190e12, 210e12));
        assert!(protected.is_protected());
        assert!(!trivial.is_protected());
    }

    #[test]
    fn pti_bandwidth() {
        let pti = PhotonicTopologicalInsulator::new(0, 1, (190e12, 210e12));
        assert!((pti.bandwidth_hz() - 20e12).abs() < 1.0);
    }

    #[test]
    fn pti_group_velocity_positive() {
        let pti = PhotonicTopologicalInsulator::new(0, 1, (190e12, 210e12));
        let vg = pti.edge_mode_group_velocity(200e12);
        assert!(vg > 0.0, "Group velocity should be positive, got {vg}");
    }

    // ── AnomalousQhpc ───────────────────────────────────────────────────────

    #[test]
    fn aqhpc_honeycomb_gyrotropic() {
        let phc = AnomalousQhpc::honeycomb_gyrotropic();
        assert_eq!(phc.lattice, "honeycomb");
        assert_eq!(phc.chern_number, 1);
        assert!(phc.transmission_backscattering_immune());
    }

    #[test]
    fn aqhpc_unidirectional_edge_modes() {
        let phc = AnomalousQhpc::square_double_chern();
        assert_eq!(phc.unidirectional_edge_mode_count(), 2);
    }

    #[test]
    fn aqhpc_q_factor_positive() {
        let phc = AnomalousQhpc::honeycomb_gyrotropic();
        let q = phc.edge_mode_q_factor();
        assert!(q > 0.0, "Q factor should be positive, got {q}");
    }
}
