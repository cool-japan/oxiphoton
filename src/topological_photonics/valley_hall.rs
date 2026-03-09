//! Valley Hall effect in photonic crystals.
//!
//! When inversion symmetry is broken in a honeycomb photonic crystal (e.g. by
//! making the A and B sublattice sites different), a band gap opens at the Dirac
//! points K and K′.  Each valley acquires an opposite half-integer Berry curvature
//! and a valley Chern number C_v = ±½.
//!
//! At a domain wall between two crystals with opposite symmetry breaking (Δε > 0
//! on one side, Δε < 0 on the other), the valley Chern number difference is
//! |C_v(K) − C_v′(K)| = 1, which guarantees a topological interface (kink) state
//! in the gap.
//!
//! Valley kink states propagate without backscattering as long as intervalley
//! scattering (K ↔ K′) is suppressed, which holds for smooth perturbations whose
//! spatial scale is large compared with a/4π.

use std::f64::consts::PI;

// Speed of light in vacuum (m/s)
const C_LIGHT: f64 = 2.997_924_58e8;

// ─── Valley Hall photonic crystal ─────────────────────────────────────────────

/// Honeycomb photonic crystal with broken inversion symmetry.
///
/// The asymmetry between the A and B sublattices (parameterised by Δε) opens a
/// photonic band gap at the K and K′ points and endows each valley with a Berry
/// curvature Ω(K) = −Ω(K′) = sign(Δε)/2.
#[derive(Debug, Clone)]
pub struct ValleyHallPhC {
    /// Lattice constant a (m).
    pub lattice_constant_m: f64,
    /// Permittivity contrast between A and B sublattice sites:
    ///   ε_A = ε̄ + Δε/2,  ε_B = ε̄ − Δε/2.
    pub delta_epsilon: f64,
    /// Average permittivity ε̄.
    pub base_epsilon: f64,
}

impl ValleyHallPhC {
    /// Construct a valley Hall photonic crystal.
    ///
    /// # Arguments
    /// * `a_m`       – lattice constant (m)
    /// * `delta_eps` – permittivity contrast Δε (dimensionless)
    /// * `eps`       – average permittivity ε̄
    pub fn new(a_m: f64, delta_eps: f64, eps: f64) -> Self {
        Self {
            lattice_constant_m: a_m,
            delta_epsilon: delta_eps,
            base_epsilon: eps,
        }
    }

    /// Valley Chern number of the lower band at the K valley.
    ///
    /// C_v(K) = sign(Δε) / 2  (half-integer valley Chern number)
    pub fn valley_chern_number_k(&self) -> f64 {
        self.delta_epsilon.signum() * 0.5
    }

    /// Valley Chern number of the lower band at the K′ valley.
    ///
    /// C_v(K′) = −C_v(K) by time-reversal symmetry.
    pub fn valley_chern_number_kprime(&self) -> f64 {
        -self.valley_chern_number_k()
    }

    /// Returns `true` if topological valley interface states exist.
    ///
    /// A non-zero Δε breaks inversion symmetry and opens a gap, guaranteeing
    /// valley kink states at an interface with the opposite symmetry breaking.
    pub fn interface_states_exist(&self) -> bool {
        self.delta_epsilon.abs() > 0.0
    }

    /// Fractional band gap Δω/ω opened by the symmetry breaking.
    ///
    /// In the long-wavelength limit (small Δε relative to ε̄):
    ///   Δω/ω ≈ |Δε| / (2 ε̄)
    pub fn band_gap_fraction(&self) -> f64 {
        self.delta_epsilon.abs() / (2.0 * self.base_epsilon)
    }

    /// Valley polarisation of an excited mode.
    ///
    /// For a circularly polarised excitation at angle θ (measured from the K-K′
    /// axis), the intensity ratio between K and K′ valleys follows:
    ///   P_V = cos(θ)
    ///
    /// This is a simplified model; the actual polarisation depends on the
    /// photonic band structure and the polarisation of the source.
    ///
    /// P_V = (I_K − I_{K′}) / (I_K + I_{K′})
    pub fn valley_polarization(&self, excitation_angle_rad: f64) -> f64 {
        // Linear model: maximum contrast along K–K′ direction (θ = 0)
        excitation_angle_rad.cos() * self.delta_epsilon.signum()
    }

    /// Refraction angle at a valley Hall interface.
    ///
    /// Valley refraction (also called "valley beam splitting") bends the beam
    /// depending on its valley index.  The refraction angle is related to the
    /// incident angle via momentum conservation at the interface:
    ///
    ///   sin(θ_r) = sin(θ_i) × (n_eff_left / n_eff_right)
    ///
    /// For opposite sign of Δε on the two sides the effective indices are
    /// different, leading to anomalous refraction.  Here we use the simple
    /// symmetric model n_eff ∝ √ε̄(1 ± band_gap_fraction/2).
    pub fn valley_refraction_angle(&self, incident_angle_rad: f64) -> f64 {
        let fg = self.band_gap_fraction();
        // Effective index ratio approximation for small fg
        let n_ratio = (1.0 + fg / 2.0) / (1.0 - fg / 2.0);
        let sin_r = incident_angle_rad.sin() * n_ratio;
        // Clamp to valid range for asin
        sin_r.clamp(-1.0, 1.0).asin()
    }

    /// Frequency of the K-point Dirac cone in Hz.
    ///
    /// For a honeycomb lattice the Dirac point lies at:
    ///   ω_K = c × |K| / √ε̄  with  |K| = 4π / (3a)
    ///
    /// so:
    ///   f_K = c / (√ε̄ × a × 3√3/(4π)) ≈ c / (√ε̄ × a × √3/π × (π/2))
    ///
    /// Simplified closed form:
    ///   f_K = c × 4π / (3√3 × a × 2π × √ε̄) = 2c / (3√3 × a × √ε̄)
    pub fn k_point_frequency_hz(&self) -> f64 {
        // |K| = 4π/(3a) for honeycomb lattice
        let k_mag = 4.0 * PI / (3.0 * self.lattice_constant_m);
        let n_eff = self.base_epsilon.sqrt();
        // f_K = c |K| / (2π n_eff)
        C_LIGHT * k_mag / (2.0 * PI * n_eff)
    }

    /// Sign of the Berry curvature at the K valley.
    ///
    /// Equals +1 when Δε > 0 and −1 when Δε < 0.
    pub fn berry_curvature_sign_at_k(&self) -> f64 {
        self.delta_epsilon.signum()
    }
}

// ─── Valley kink state ─────────────────────────────────────────────────────────

/// Topological valley kink state at a zigzag interface.
///
/// Valley kink states are localised at the domain wall between two valley Hall
/// photonic crystals with opposite Δε signs.  They propagate in one direction
/// at the K valley and in the opposite direction at K′, which is the photonic
/// analogue of the quantum spin Hall effect.
///
/// Backscattering at smooth obstacles is suppressed because intervalley coupling
/// is weak; sharp corners with preserved valley polarisation transmit with
/// near-unity efficiency.
#[derive(Debug, Clone)]
pub struct ValleyKinkState {
    /// Group velocity as a fraction of the Dirac cone velocity v_Dirac.
    ///
    /// Typically 0.3–0.9 depending on the gap fraction and position in the gap.
    pub group_velocity_fraction: f64,
    /// Valley index: +1 = K valley, −1 = K′ valley.
    pub valley: i8,
    /// Propagation direction (angle in radians, 0 = +x direction).
    pub propagation_direction: f64,
}

impl ValleyKinkState {
    /// Construct a valley kink state.
    ///
    /// # Arguments
    /// * `group_velocity_fraction` – v_g / v_Dirac (0 to 1)
    /// * `valley`                   – +1 for K, −1 for K′
    /// * `propagation_direction`    – angle in radians
    pub fn new(group_velocity_fraction: f64, valley: i8, propagation_direction: f64) -> Self {
        Self {
            group_velocity_fraction,
            valley,
            propagation_direction,
        }
    }

    /// Returns `true` if `self` and `other` are counter-propagating (opposite valleys).
    ///
    /// Two kink states are counter-propagating when they belong to opposite valleys
    /// (K ↔ K′).  This is the analogue of spin-momentum locking: the K-valley mode
    /// propagates right and the K′-valley mode propagates left at the same interface.
    pub fn is_counter_propagating(&self, other: &ValleyKinkState) -> bool {
        self.valley != other.valley
    }

    /// Returns `true` — valley kink states are immune to backscattering at
    /// sharp corners provided intervalley scattering is negligible.
    ///
    /// This is a fundamental consequence of valley-momentum locking: a rightward
    /// K-valley mode cannot scatter into a leftward K-valley mode because the
    /// leftward mode has K′ character.
    pub fn immune_to_sharp_corners(&self) -> bool {
        true
    }

    /// Returns `true` if `self` is a right-propagating kink state.
    ///
    /// Defined as |angle mod 2π| < π/2 or > 3π/2.
    pub fn is_right_propagating(&self) -> bool {
        let angle = self.propagation_direction.rem_euclid(2.0 * PI);
        !(PI / 2.0..=3.0 * PI / 2.0).contains(&angle)
    }

    /// Estimated transmission through a sharp 60° bend in a valley waveguide.
    ///
    /// Literature values for valley Hall systems with moderate gap fraction:
    ///   T ≈ 0.85–0.95 for a 60° bend.
    ///
    /// We model this as T = 1 − 0.1 × (1 − v_g/v_Dirac), which interpolates
    /// between T → 0.9 for flat-band modes and T → 1.0 for Dirac-cone modes.
    pub fn transmission_at_60_bend(&self) -> f64 {
        (1.0 - 0.1 * (1.0 - self.group_velocity_fraction.clamp(0.0, 1.0))).max(0.0)
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valley_hall_phc_interface_states_with_delta_eps() {
        let phc = ValleyHallPhC::new(500e-9, 0.5, 12.0);
        assert!(phc.interface_states_exist());
    }

    #[test]
    fn valley_hall_phc_no_interface_states_symmetric() {
        let phc = ValleyHallPhC::new(500e-9, 0.0, 12.0);
        assert!(!phc.interface_states_exist());
    }

    #[test]
    fn valley_hall_phc_band_gap_fraction() {
        let phc = ValleyHallPhC::new(500e-9, 1.0, 10.0);
        let fg = phc.band_gap_fraction();
        assert!((fg - 0.05).abs() < 1e-12, "fg = {fg}");
    }

    #[test]
    fn valley_chern_numbers_opposite() {
        let phc = ValleyHallPhC::new(500e-9, 0.5, 12.0);
        let ck = phc.valley_chern_number_k();
        let ck_prime = phc.valley_chern_number_kprime();
        assert!(
            (ck + ck_prime).abs() < 1e-12,
            "C_v(K) + C_v(K') should be 0"
        );
        assert!((ck.abs() - 0.5).abs() < 1e-12, "|C_v(K)| should be 0.5");
    }

    #[test]
    fn valley_chern_sign_flips_with_delta_eps() {
        let phc_pos = ValleyHallPhC::new(500e-9, 0.5, 12.0);
        let phc_neg = ValleyHallPhC::new(500e-9, -0.5, 12.0);
        assert!(phc_pos.valley_chern_number_k() > 0.0);
        assert!(phc_neg.valley_chern_number_k() < 0.0);
    }

    #[test]
    fn valley_polarization_maximum_at_zero_angle() {
        let phc = ValleyHallPhC::new(500e-9, 1.0, 12.0);
        let p_zero = phc.valley_polarization(0.0).abs();
        let p_ortho = phc.valley_polarization(PI / 2.0).abs();
        assert!(p_zero > p_ortho, "Polarisation should be maximal at θ=0");
    }

    #[test]
    fn valley_refraction_angle_positive_for_positive_incidence() {
        let phc = ValleyHallPhC::new(500e-9, 0.5, 12.0);
        let theta_r = phc.valley_refraction_angle(0.3);
        assert!(
            theta_r > 0.0,
            "Refraction angle should be positive for positive incidence"
        );
    }

    #[test]
    fn k_point_frequency_positive() {
        let phc = ValleyHallPhC::new(500e-9, 0.5, 12.0);
        let fk = phc.k_point_frequency_hz();
        assert!(fk > 0.0, "K-point frequency must be positive, got {fk}");
    }

    #[test]
    fn berry_curvature_sign_at_k_matches_delta_eps() {
        let phc_pos = ValleyHallPhC::new(500e-9, 0.5, 12.0);
        let phc_neg = ValleyHallPhC::new(500e-9, -0.5, 12.0);
        assert!(phc_pos.berry_curvature_sign_at_k() > 0.0);
        assert!(phc_neg.berry_curvature_sign_at_k() < 0.0);
    }

    // ── ValleyKinkState ─────────────────────────────────────────────────────

    #[test]
    fn valley_kink_counter_propagating() {
        let k_mode = ValleyKinkState::new(0.7, 1, 0.0);
        let kprime_mode = ValleyKinkState::new(0.7, -1, PI);
        assert!(k_mode.is_counter_propagating(&kprime_mode));
        assert!(!k_mode.is_counter_propagating(&ValleyKinkState::new(0.7, 1, PI)));
    }

    #[test]
    fn valley_kink_immune_to_sharp_corners() {
        let state = ValleyKinkState::new(0.7, 1, 0.0);
        assert!(state.immune_to_sharp_corners());
    }

    #[test]
    fn valley_kink_transmission_at_bend_near_unity() {
        let state = ValleyKinkState::new(0.9, 1, 0.0);
        let t = state.transmission_at_60_bend();
        assert!(t > 0.8 && t <= 1.0, "T = {t}");
    }
}
