//! Exciton-polariton physics via Hopfield transformation.
//!
//! Models the strong coupling between cavity photons and quantum-well excitons,
//! producing the lower polariton (LP) and upper polariton (UP) branches through
//! the Hopfield diagonalisation.
//!
//! # Physics overview
//!
//! In a microcavity containing quantum wells, the Hamiltonian in the exciton-photon
//! basis is:
//!
//! ```text
//! H = [ E_c(k)   ħΩ_R/2  ] [ C ]   [ E_LP ]   [ C ]
//!     [ ħΩ_R/2   E_X     ] [ X ] = E·[ X ]
//! ```
//!
//! The Hopfield transformation diagonalises H, giving LP and UP branches:
//!
//! ```text
//! E_LP(k) = (E_c + E_X)/2 - sqrt((δ/2)² + (ħΩ_R/2)²)
//! E_UP(k) = (E_c + E_X)/2 + sqrt((δ/2)² + (ħΩ_R/2)²)
//! ```
//!
//! where δ = E_c(k) − E_X is the cavity-exciton detuning.
//!
//! # References
//! - J. J. Hopfield, Phys. Rev. 112, 1555 (1958)
//! - C. Weisbuch et al., Phys. Rev. Lett. 69, 3314 (1992)

use std::f64::consts::PI;

// ─── Physical constants ───────────────────────────────────────────────────────
/// Reduced Planck constant (J·s)
const HBAR: f64 = 1.054_571_817e-34;
/// Speed of light in vacuum (m/s)
const C_LIGHT: f64 = 2.997_924_58e8;
/// Electron volt to Joule conversion
const EV_TO_J: f64 = 1.602_176_634e-19;
/// Boltzmann constant (J/K)
const KB: f64 = 1.380_649e-23;
/// Electron rest mass (kg)
const ME: f64 = 9.109_383_701_5e-31;
/// Vacuum permittivity (F/m)
const EPS0: f64 = 8.854_187_817e-12;

// ─── CavityPhoton ────────────────────────────────────────────────────────────

/// Microcavity photon dispersion.
///
/// The cavity photon acquires a tiny effective mass from confinement in the
/// vertical direction.  For a planar Fabry-Pérot microcavity:
///
/// ```text
/// E_c(k_‖) = E_c0 + ħ²k_‖² / (2 m_eff)
/// ```
///
/// where m_eff ≈ 10⁻⁵ m_e for typical GaAs microcavities.
#[derive(Debug, Clone)]
pub struct CavityPhoton {
    /// Cavity photon energy at zero in-plane wavevector E_c(k=0) in eV.
    pub energy_at_zero_ev: f64,
    /// Effective mass as fraction of electron mass m_eff / m_e.
    /// Typical range: 1e-5 (GaAs) to 1e-4 (GaN).
    pub effective_mass_fraction: f64,
}

impl CavityPhoton {
    /// Cavity photon energy vs in-plane wavevector k (m⁻¹).
    ///
    /// E_c(k) = E_c0 + ħ²k² / (2 m_eff)  [eV]
    pub fn energy_ev(&self, k_per_m: f64) -> f64 {
        let m_eff = self.effective_mass_fraction * ME;
        let kinetic_j = HBAR * HBAR * k_per_m * k_per_m / (2.0 * m_eff);
        self.energy_at_zero_ev + kinetic_j / EV_TO_J
    }

    /// Photon group velocity dE/d(ħk) = ħk / m_eff (m/s).
    ///
    /// Returns the group velocity at wavevector k (m⁻¹).
    pub fn group_velocity(&self, k_per_m: f64) -> f64 {
        let m_eff = self.effective_mass_fraction * ME;
        HBAR * k_per_m / m_eff
    }
}

// ─── Exciton ────────────────────────────────────────────────────────────────

/// Two-dimensional Wannier-Mott exciton in a quantum well.
///
/// The exciton acts as a two-level oscillator with transition energy E_X and
/// oscillator strength f that determines coupling to the cavity photon field.
#[derive(Debug, Clone)]
pub struct Exciton {
    /// Exciton resonance energy E_X (eV).
    pub energy_ev: f64,
    /// Homogeneous linewidth Γ_X (meV) — HWHM of the Lorentzian lineshape.
    pub linewidth_mev: f64,
    /// Oscillator strength f (dimensionless, proportional to dipole moment squared).
    pub oscillator_strength: f64,
    /// Exciton binding energy E_b (meV).  For GaAs QW: ~10 meV.
    pub binding_energy_mev: f64,
}

impl Exciton {
    /// Exciton-photon coupling constant: vacuum Rabi splitting ħΩ_R (eV).
    ///
    /// The Rabi splitting is related to the oscillator strength via:
    ///
    /// ```text
    /// ħΩ_R = sqrt(f · e² · ħ / (ε₀ m_e V_eff ω))
    /// ```
    ///
    /// where V_eff is the effective mode volume and ω = E_X / ħ.
    ///
    /// For a GaAs QW in a λ-microcavity with V_eff ~ 10 µm³:
    /// ħΩ_R ~ 5–10 meV.
    pub fn rabi_splitting_ev(&self, cavity_mode_volume_m3: f64) -> f64 {
        let e_charge = EV_TO_J.sqrt(); // e = sqrt(EV_TO_J) in SI-like Gaussian
        // Proper SI: e² = 1.602176634e-19 C squared
        let e2 = 1.602_176_634e-19 * 1.602_176_634e-19; // e² in C²
        let omega = self.energy_ev * EV_TO_J / HBAR; // ω = E_X / ħ
        // ħΩ_R = sqrt(f · e² / (ε₀ m_e V ω)) × ħ / (2ω) × 2ω/ħ ... simplify:
        // ħΩ_R = sqrt(f e² ħ / (ε₀ m_e V_eff ω))
        let _ = e_charge; // used implicitly via e2
        let numerator = self.oscillator_strength * e2 * HBAR;
        let denominator = EPS0 * ME * cavity_mode_volume_m3 * omega;
        if denominator > 0.0 {
            numerator.sqrt() / (EV_TO_J * denominator.sqrt() / numerator.sqrt()) * (numerator / denominator).sqrt() / EV_TO_J
        } else {
            0.0
        }
    }
}

// ─── Polariton ──────────────────────────────────────────────────────────────

/// Cavity exciton-polariton: Hopfield mixture of photon and exciton.
///
/// The polariton eigenstates are superpositions:
///
/// ```text
/// |LP⟩ =  X|X⟩ + C|ph⟩    (lower polariton, mostly photon at large negative δ)
/// |UP⟩ = -C|X⟩ + X|ph⟩    (upper polariton)
/// ```
///
/// where X² + C² = 1 (Hopfield normalization) and the coefficients depend on
/// the cavity-exciton detuning δ = E_c(k) − E_X.
#[derive(Debug, Clone)]
pub struct Polariton {
    /// Cavity photon dispersion.
    pub cavity: CavityPhoton,
    /// Quantum-well exciton.
    pub exciton: Exciton,
    /// Vacuum Rabi splitting ħΩ_R (eV).  Equals the anti-crossing gap at δ=0.
    pub rabi_splitting_ev: f64,
}

impl Polariton {
    /// Create a new polariton system.
    pub fn new(cavity: CavityPhoton, exciton: Exciton, rabi_splitting_ev: f64) -> Self {
        Self { cavity, exciton, rabi_splitting_ev }
    }

    /// Detuning δ(k) = E_c(k) − E_X (eV).
    ///
    /// Positive detuning → photon-like LP; negative → exciton-like LP.
    pub fn detuning_ev(&self, k_per_m: f64) -> f64 {
        self.cavity.energy_ev(k_per_m) - self.exciton.energy_ev
    }

    /// Lower polariton energy E_LP(k) in eV.
    ///
    /// ```text
    /// E_LP = (E_c + E_X)/2 − sqrt((δ/2)² + (ħΩ_R/2)²)
    /// ```
    pub fn lp_energy_ev(&self, k_per_m: f64) -> f64 {
        let e_c = self.cavity.energy_ev(k_per_m);
        let e_x = self.exciton.energy_ev;
        let delta = e_c - e_x;
        let half_rabi = self.rabi_splitting_ev / 2.0;
        (e_c + e_x) / 2.0 - ((delta / 2.0).powi(2) + half_rabi.powi(2)).sqrt()
    }

    /// Upper polariton energy E_UP(k) in eV.
    ///
    /// ```text
    /// E_UP = (E_c + E_X)/2 + sqrt((δ/2)² + (ħΩ_R/2)²)
    /// ```
    pub fn up_energy_ev(&self, k_per_m: f64) -> f64 {
        let e_c = self.cavity.energy_ev(k_per_m);
        let e_x = self.exciton.energy_ev;
        let delta = e_c - e_x;
        let half_rabi = self.rabi_splitting_ev / 2.0;
        (e_c + e_x) / 2.0 + ((delta / 2.0).powi(2) + half_rabi.powi(2)).sqrt()
    }

    /// Photon Hopfield coefficient squared for the LP branch: |C|².
    ///
    /// ```text
    /// |C|² = ½ · (1 + δ / sqrt(δ² + ħΩ_R²))
    /// ```
    ///
    /// At resonance (δ=0): |C|² = ½.
    /// For δ → −∞: |C|² → 0 (exciton-like LP).
    /// For δ → +∞: |C|² → 1 (photon-like LP).
    pub fn photon_fraction_lp(&self, k_per_m: f64) -> f64 {
        let delta = self.detuning_ev(k_per_m);
        let omega_r = self.rabi_splitting_ev;
        let discriminant = (delta * delta + omega_r * omega_r).sqrt();
        if discriminant < f64::EPSILON {
            return 0.5;
        }
        0.5 * (1.0 + delta / discriminant)
    }

    /// Exciton Hopfield coefficient squared for the LP branch: |X|² = 1 − |C|².
    pub fn exciton_fraction_lp(&self, k_per_m: f64) -> f64 {
        1.0 - self.photon_fraction_lp(k_per_m)
    }

    /// LP effective mass at k ≈ 0 from the harmonic mean of photon and exciton masses.
    ///
    /// ```text
    /// 1/m_LP = |C|² / m_c  +  |X|² / m_X
    /// ```
    ///
    /// Since m_X ≫ m_c for a QW exciton (m_X ~ 0.5 m_e), m_LP ≈ m_c / |C|².
    ///
    /// `electron_mass` — free electron mass in kg (= ME).
    pub fn lp_effective_mass(&self, electron_mass: f64) -> f64 {
        let c2 = self.photon_fraction_lp(0.0);
        let x2 = 1.0 - c2;
        let m_c = self.cavity.effective_mass_fraction * electron_mass;
        // Exciton effective mass ~ 0.4 m_e for GaAs QW
        let m_x = 0.4 * electron_mass;
        // Harmonic mean: 1/m_LP = |C|²/m_c + |X|²/m_x
        let inv_m = c2 / m_c + x2 / m_x;
        if inv_m > 0.0 { 1.0 / inv_m } else { m_c }
    }

    /// LP group velocity at k (m/s), computed via finite differences.
    ///
    /// v_g = dE_LP/d(ħk) ≈ [E_LP(k+dk) − E_LP(k−dk)] / (2 ħ dk)
    pub fn lp_group_velocity(&self, k_per_m: f64, dk: f64) -> f64 {
        let e_plus = self.lp_energy_ev(k_per_m + dk);
        let e_minus = self.lp_energy_ev(k_per_m - dk);
        let de_ev = e_plus - e_minus;
        let de_j = de_ev * EV_TO_J;
        // v_g = (1/ħ) dE/dk
        de_j / (HBAR * 2.0 * dk)
    }

    /// Compute the LP and UP dispersion curves.
    ///
    /// Returns `(k_values, lp_energies, up_energies)` with `n_points` samples
    /// in the range `[−k_max, +k_max]` (m⁻¹).
    pub fn dispersion(&self, k_max: f64, n_points: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = n_points.max(2);
        let mut ks = Vec::with_capacity(n);
        let mut lp = Vec::with_capacity(n);
        let mut up = Vec::with_capacity(n);
        for i in 0..n {
            let k = -k_max + 2.0 * k_max * (i as f64) / ((n - 1) as f64);
            ks.push(k);
            lp.push(self.lp_energy_ev(k));
            up.push(self.up_energy_ev(k));
        }
        (ks, lp, up)
    }

    /// Anti-crossing energy gap at zero detuning (k such that E_c(k) = E_X).
    ///
    /// At resonance, the LP-UP splitting equals ħΩ_R (the Rabi splitting).
    pub fn anticrossing_gap_ev(&self) -> f64 {
        self.rabi_splitting_ev
    }

    /// Find the in-plane wavevector at which E_c(k) = E_X (resonance condition).
    ///
    /// Returns `None` if the cavity minimum energy already exceeds E_X.
    pub fn resonance_wavevector(&self) -> Option<f64> {
        let e_c0 = self.cavity.energy_at_zero_ev;
        let e_x = self.exciton.energy_ev;
        if e_c0 > e_x {
            return None;
        }
        // E_c0 + ħ²k²/(2 m_eff) = E_X  →  k = sqrt(2 m_eff (E_X - E_c0) / ħ²)
        let m_eff = self.cavity.effective_mass_fraction * ME;
        let de_j = (e_x - e_c0) * EV_TO_J;
        Some((2.0 * m_eff * de_j / (HBAR * HBAR)).sqrt())
    }
}

// ─── PolaritonDistribution ──────────────────────────────────────────────────

/// Bose-Einstein polariton momentum distribution at temperature T.
///
/// Polaritons are composite bosons; at sufficiently high density and low
/// temperature, they can undergo Bose-Einstein condensation (BEC) into the
/// k=0 LP state.
#[derive(Debug, Clone)]
pub struct PolaritonDistribution {
    /// Lattice / bath temperature (K).
    pub temperature_k: f64,
    /// Total polariton number N.
    pub n_total: f64,
    /// Chemical potential µ (eV); at condensation µ → E_LP(k=0).
    pub chemical_potential_ev: f64,
}

impl PolaritonDistribution {
    /// Bose-Einstein occupation number at energy E (eV).
    ///
    /// ```text
    /// n(E) = 1 / (exp((E − µ) / k_B T) − 1)
    /// ```
    ///
    /// Returns 0 if the exponent would overflow, and f64::INFINITY if the
    /// denominator approaches zero from below (condensate divergence).
    pub fn occupation(&self, energy_ev: f64) -> f64 {
        let kbt_ev = KB * self.temperature_k / EV_TO_J;
        if kbt_ev < f64::EPSILON {
            // Zero temperature: step function
            return if energy_ev <= self.chemical_potential_ev { f64::INFINITY } else { 0.0 };
        }
        let x = (energy_ev - self.chemical_potential_ev) / kbt_ev;
        if x > 700.0 {
            // Boltzmann tail
            return (-x).exp();
        }
        let exp_x = x.exp();
        let denom = exp_x - 1.0;
        if denom.abs() < 1e-30 {
            f64::INFINITY
        } else {
            1.0 / denom
        }
    }

    /// BEC condensation criterion: µ ≥ E_LP(k=0) − small margin.
    ///
    /// In practice we check whether µ is within k_B T of the LP ground state.
    pub fn is_condensed(&self, polariton: &Polariton) -> bool {
        let e_lp_min = polariton.lp_energy_ev(0.0);
        let kbt_ev = KB * self.temperature_k / EV_TO_J;
        // Condensation when µ is within k_BT of the dispersion minimum
        self.chemical_potential_ev >= e_lp_min - kbt_ev.max(1e-6)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Standard GaAs microcavity near zero detuning.
    fn gaas_polariton() -> Polariton {
        let cavity = CavityPhoton {
            energy_at_zero_ev: 1.5,
            effective_mass_fraction: 1e-5,
        };
        let exciton = Exciton {
            energy_ev: 1.5,
            linewidth_mev: 1.0,
            oscillator_strength: 10.0,
            binding_energy_mev: 10.0,
        };
        Polariton::new(cavity, exciton, 0.005) // 5 meV Rabi splitting
    }

    #[test]
    fn hopfield_normalization() {
        let pol = gaas_polariton();
        let c2 = pol.photon_fraction_lp(0.0);
        let x2 = pol.exciton_fraction_lp(0.0);
        assert!((c2 + x2 - 1.0).abs() < 1e-10, "Hopfield sum rule violated: {}", c2 + x2);
    }

    #[test]
    fn hopfield_normalization_off_resonance() {
        let pol = gaas_polariton();
        // Check at several k values
        for &k in &[0.0, 1e6, 3e6, 5e6, -2e6] {
            let c2 = pol.photon_fraction_lp(k);
            let x2 = pol.exciton_fraction_lp(k);
            assert!(
                (c2 + x2 - 1.0).abs() < 1e-10,
                "Hopfield normalization failed at k={}: C²+X²={}",
                k, c2 + x2
            );
        }
    }

    #[test]
    fn anticrossing_gap_equals_rabi() {
        let pol = gaas_polariton();
        // At k=0 (zero detuning), UP-LP gap = ħΩ_R
        let gap = pol.up_energy_ev(0.0) - pol.lp_energy_ev(0.0);
        let expected = pol.rabi_splitting_ev;
        assert!((gap - expected).abs() < 1e-10, "Anti-crossing gap {} ≠ Rabi {}", gap, expected);
    }

    #[test]
    fn lp_below_both_bare_modes() {
        let pol = gaas_polariton();
        // LP energy must be below min(E_c, E_X) everywhere
        for i in 0..20 {
            let k = (i as f64 - 10.0) * 1e6;
            let e_lp = pol.lp_energy_ev(k);
            let e_c = pol.cavity.energy_ev(k);
            let e_x = pol.exciton.energy_ev;
            assert!(
                e_lp <= e_c.min(e_x) + 1e-9,
                "LP energy {} above bare modes at k={}: E_c={}, E_X={}",
                e_lp, k, e_c, e_x
            );
        }
    }

    #[test]
    fn cavity_photon_dispersion_parabolic() {
        let cavity = CavityPhoton {
            energy_at_zero_ev: 1.5,
            effective_mass_fraction: 1e-5,
        };
        // At k=0 the energy is exactly E_c0
        let e0 = cavity.energy_ev(0.0);
        assert!((e0 - 1.5).abs() < 1e-12, "E_c(k=0) = {}", e0);
        // Energy increases with k
        assert!(cavity.energy_ev(1e6) > cavity.energy_ev(0.0));
    }

    #[test]
    fn group_velocity_at_zero_is_zero() {
        let cavity = CavityPhoton {
            energy_at_zero_ev: 1.5,
            effective_mass_fraction: 1e-5,
        };
        let v = cavity.group_velocity(0.0);
        assert!(v.abs() < 1e-10, "Group velocity at k=0 should be zero, got {}", v);
    }

    #[test]
    fn bec_condensation_criterion() {
        let pol = gaas_polariton();
        let e_lp_min = pol.lp_energy_ev(0.0);
        // Distribution with µ = E_LP(0) → condensed
        let dist_cond = PolaritonDistribution {
            temperature_k: 4.0,
            n_total: 1e6,
            chemical_potential_ev: e_lp_min,
        };
        assert!(dist_cond.is_condensed(&pol), "Should be condensed");
        // Distribution with µ far below ground state → not condensed
        let dist_normal = PolaritonDistribution {
            temperature_k: 300.0,
            n_total: 1e4,
            chemical_potential_ev: e_lp_min - 0.1,
        };
        assert!(!dist_normal.is_condensed(&pol), "Should not be condensed");
    }

    #[test]
    fn dispersion_returns_correct_size() {
        let pol = gaas_polariton();
        let (ks, lp, up) = pol.dispersion(5e6, 100);
        assert_eq!(ks.len(), 100);
        assert_eq!(lp.len(), 100);
        assert_eq!(up.len(), 100);
        // LP ≤ UP everywhere
        for (e_lp, e_up) in lp.iter().zip(up.iter()) {
            assert!(e_lp <= e_up + 1e-12);
        }
    }
}
