//! Photonic time crystals (PTCs) and spatiotemporal photonic crystals.
//!
//! A photonic time crystal is spatially uniform but periodically modulated
//! in time: ε(t) = ε₀ + Δε cos(Ω_m t).  Unlike a spatial photonic crystal
//! (which opens energy/frequency band gaps), a PTC opens **momentum** band
//! gaps — ranges of k that cannot propagate for a given ω.
//!
//! Hallmarks of PTCs:
//!   - Amplification of electromagnetic waves above a threshold.
//!   - Vacuum squeezing of quantum fluctuations.
//!   - Topological winding number in the (ω, Ω_m) parameter space.
//!
//! References:
//!   - Sharabi et al., Science Advances 2021
//!   - Carminati & Vu, Phys. Rev. Lett. 2022
//!   - Hayran & Monticone, Optica 2022

use std::f64::consts::PI;

/// Speed of light in vacuum (m/s).
const C_LIGHT: f64 = 2.997_924_58e8;

// ─── Photonic time crystal ────────────────────────────────────────────────────

/// Spatially homogeneous medium whose permittivity is periodically modulated:
///
/// ```text
/// ε(t) = ε₀ + Δε cos(Ω_m t)
/// ```
///
/// The magnetic permeability is μ = 1 (non-magnetic).
#[derive(Debug, Clone, Copy)]
pub struct PhotonicTimeCrystal {
    /// Average (DC) permittivity ε₀.
    pub eps0: f64,
    /// Modulation amplitude Δε.
    pub delta_eps: f64,
    /// Modulation angular frequency Ω_m (rad/s).
    pub omega_mod: f64,
    /// Linear loss rate γ (rad/s) — represents cavity or material absorption.
    pub loss_rate: f64,
}

impl PhotonicTimeCrystal {
    /// Base refractive index n₀ = √ε₀.
    pub fn base_index(&self) -> f64 {
        self.eps0.sqrt()
    }

    /// Relative modulation depth δ = Δε / (2 ε₀).
    pub fn modulation_depth(&self) -> f64 {
        self.delta_eps / (2.0 * self.eps0)
    }

    /// Unperturbed resonance frequency for a given carrier frequency ω₀.
    ///
    /// At the primary PTC resonance: Ω_m = 2 ω₀.  This helper extracts
    /// ω₀ from the stored Ω_m.
    pub fn resonant_omega0(&self) -> f64 {
        self.omega_mod / 2.0
    }

    /// Width of the primary momentum band gap at Ω_m = 2ω₀ (rad/s).
    ///
    /// ```text
    /// Δω_gap = (Δε / 4ε₀) × ω₀ = δ ω₀ / 2
    /// ```
    pub fn band_gap_width_rad(&self, omega0: f64) -> f64 {
        self.modulation_depth() / 2.0 * omega0
    }

    /// Momentum gap width Δk = Δω_gap / c (m⁻¹).
    pub fn momentum_gap_per_m(&self, omega0: f64) -> f64 {
        self.band_gap_width_rad(omega0) / C_LIGHT
    }

    /// Net amplification rate Γ when the operating point is inside the
    /// momentum band gap (rad/s).
    ///
    /// ```text
    /// Γ = √[ (δ ω₀ / 4)² – (γ/2)² ]
    /// ```
    ///
    /// Returns 0.0 if the loss rate is too high or the system is outside the
    /// gap (below threshold).
    pub fn amplification_rate(&self, omega0: f64) -> f64 {
        let coupling_sq = (self.modulation_depth() * omega0 / 4.0).powi(2);
        let loss_half_sq = (self.loss_rate / 2.0).powi(2);
        let discriminant = coupling_sq - loss_half_sq;
        if discriminant <= 0.0 {
            0.0
        } else {
            discriminant.sqrt()
        }
    }

    /// Return `true` when the PTC is above its amplification threshold at ω₀.
    pub fn is_above_threshold(&self, omega0: f64) -> bool {
        self.amplification_rate(omega0) > 0.0
    }

    /// Squeezing factor (dB) of one vacuum-fluctuation quadrature for a PTC
    /// operated **below** threshold.
    ///
    /// Below threshold the system acts as a degenerate parametric oscillator.
    /// The squeezed quadrature variance evolves as:
    ///
    /// ```text
    /// V_sq(t) = exp(–2 Γ_sub t)
    /// Squeezing [dB] = –10 log₁₀(V_sq) = 20 Γ_sub t / ln(10)
    /// ```
    ///
    /// where Γ_sub = √[ (γ/2)² – (δ ω₀/4)² ] is the sub-threshold damping
    /// enhancement.  Returns 0.0 if above threshold.
    pub fn squeezing_factor_db(&self, omega0: f64, time_s: f64) -> f64 {
        if self.is_above_threshold(omega0) {
            return 0.0;
        }
        let loss_half_sq = (self.loss_rate / 2.0).powi(2);
        let coupling_sq = (self.modulation_depth() * omega0 / 4.0).powi(2);
        let gamma_sub = (loss_half_sq - coupling_sq).max(0.0).sqrt();
        20.0 * gamma_sub * time_s / 10_f64.ln()
    }

    /// Topological winding number of the time-modulated dispersion.
    ///
    /// A simple criterion:
    ///   - **0** (trivial)  when the frequency ω₀ lies *outside* the band gap,
    ///     i.e. |ω₀ – Ω_m/2| > Δω_gap/2.
    ///   - **1** (topological)  when ω₀ lies *inside* the band gap.
    pub fn temporal_winding_number(&self, omega0: f64) -> i32 {
        let gap_half = self.band_gap_width_rad(omega0) / 2.0;
        let detuning = (omega0 - self.omega_mod / 2.0).abs();
        if detuning < gap_half {
            1
        } else {
            0
        }
    }

    /// Floquet quasi-energy splitting at the band edge (rad/s).
    ///
    /// At the primary resonance Ω_m = 2ω₀ the two quasi-energy branches
    /// split by Δε/(2ε₀) × ω₀ / 2.
    pub fn quasi_energy_splitting(&self, omega0: f64) -> f64 {
        self.modulation_depth() * omega0 / 2.0
    }

    /// Group velocity inside the PTC medium (m/s).
    ///
    /// v_g = c / n₀.  Modulation does not change the group velocity to
    /// zeroth order; first-order corrections are O(δ²).
    pub fn group_velocity(&self) -> f64 {
        let n0 = self.base_index();
        if n0 == 0.0 {
            return C_LIGHT;
        }
        C_LIGHT / n0
    }
}

// ─── Spatiotemporal crystal ───────────────────────────────────────────────────

/// One-dimensional spatiotemporal photonic crystal: ε periodic in both x and t.
///
/// The permittivity is expanded to first order:
///
/// ```text
/// ε(x, t) = ε₀₀ + ε₁₀ cos(G x) + ε₀₁ cos(Ω t) + ε₁₁ cos(G x – Ω t)
/// ```
///
/// The spatiotemporal coupling term ε₁₁ breaks spatial reciprocity, enabling
/// non-reciprocal propagation (different group velocities for +k and –k).
#[derive(Debug, Clone, Copy)]
pub struct SpatiotemporalCrystal {
    /// Spatial period a (m).
    pub spatial_period_m: f64,
    /// Temporal period T (s).
    pub temporal_period_s: f64,
    /// DC (spatially uniform, time-constant) permittivity component ε₀₀.
    pub eps_00: f64,
    /// Spatial modulation amplitude ε₁₀.
    pub eps_10: f64,
    /// Temporal modulation amplitude ε₀₁.
    pub eps_01: f64,
    /// Spatiotemporal coupling amplitude ε₁₁ (drives non-reciprocity).
    pub eps_11: f64,
}

impl SpatiotemporalCrystal {
    /// Spatial Bragg vector G = 2π / a (rad/m).
    fn bragg_vector(&self) -> f64 {
        if self.spatial_period_m == 0.0 {
            return 0.0;
        }
        2.0 * PI / self.spatial_period_m
    }

    /// Temporal Bragg frequency Ω = 2π / T (rad/s).
    fn temporal_freq(&self) -> f64 {
        if self.temporal_period_s == 0.0 {
            return 0.0;
        }
        2.0 * PI / self.temporal_period_s
    }

    /// Light-cone tilt: the ratio v_st = Ω / G describes the effective
    /// velocity of the travelling modulation pattern.
    ///
    /// When v_st ≠ 0 the dispersion cone tilts asymmetrically → non-reciprocal.
    pub fn light_cone_tilt(&self) -> f64 {
        let g = self.bragg_vector();
        if g == 0.0 {
            return 0.0;
        }
        self.temporal_freq() / g
    }

    /// Non-reciprocal bandwidth Δν (Hz) — approximate first-order estimate.
    ///
    /// The splitting between forward and backward group velocities is
    /// proportional to ε₁₁ / ε₀₀:
    ///
    /// ```text
    /// Δν_NR ≈ (ε₁₁ / ε₀₀) × v_g / (2 a)
    /// ```
    pub fn nonreciprocal_bandwidth_hz(&self) -> f64 {
        if self.eps_00 == 0.0 || self.spatial_period_m == 0.0 {
            return 0.0;
        }
        let n0 = self.eps_00.sqrt();
        let v_g = C_LIGHT / n0;
        let nr_ratio = self.eps_11.abs() / self.eps_00;
        nr_ratio * v_g / (2.0 * self.spatial_period_m)
    }

    /// Isolation ratio I (dB) between forward (+k) and backward (–k) waves.
    ///
    /// First-order perturbative estimate:
    ///
    /// ```text
    /// I ≈ 20 log₁₀(1 + |ε₁₁| / ε₀₀)
    /// ```
    pub fn isolation_ratio_db(&self) -> f64 {
        if self.eps_00 == 0.0 {
            return 0.0;
        }
        20.0 * (1.0 + self.eps_11.abs() / self.eps_00).log10()
    }

    /// Total modulation depth (ratio of ac to dc component).
    pub fn total_modulation_depth(&self) -> f64 {
        if self.eps_00 == 0.0 {
            return 0.0;
        }
        (self.eps_10.powi(2) + self.eps_01.powi(2) + self.eps_11.powi(2)).sqrt() / self.eps_00
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_ptc() -> PhotonicTimeCrystal {
        PhotonicTimeCrystal {
            eps0: 2.25, // n₀ = 1.5
            delta_eps: 0.5,
            omega_mod: 4e14, // Ω_m = 2ω₀ with ω₀ = 2×10¹⁴
            loss_rate: 1e10,
        }
    }

    #[test]
    fn ptc_above_threshold() {
        let ptc = default_ptc();
        let omega0 = ptc.omega_mod / 2.0;
        assert!(
            ptc.is_above_threshold(omega0),
            "PTC should be above threshold at resonance"
        );
    }

    #[test]
    fn amplification_rate_positive_above_threshold() {
        let ptc = default_ptc();
        let omega0 = ptc.omega_mod / 2.0;
        let gamma = ptc.amplification_rate(omega0);
        assert!(
            gamma > 0.0,
            "amplification rate should be positive: {gamma}"
        );
    }

    #[test]
    fn squeezing_zero_above_threshold() {
        let ptc = default_ptc();
        let omega0 = ptc.omega_mod / 2.0;
        let sq = ptc.squeezing_factor_db(omega0, 1e-9);
        assert_eq!(sq, 0.0, "squeezing should be zero above threshold");
    }

    #[test]
    fn squeezing_positive_below_threshold() {
        let ptc = PhotonicTimeCrystal {
            eps0: 2.25,
            delta_eps: 1e-3, // very small modulation → below threshold
            omega_mod: 4e14,
            loss_rate: 1e12, // large loss
        };
        let omega0 = ptc.omega_mod / 2.0;
        let sq = ptc.squeezing_factor_db(omega0, 1e-9);
        assert!(sq >= 0.0, "squeezing should be non-negative: {sq}");
    }

    #[test]
    fn winding_number_inside_gap() {
        let ptc = default_ptc();
        let omega0 = ptc.omega_mod / 2.0; // exactly at resonance centre
        assert_eq!(ptc.temporal_winding_number(omega0), 1);
    }

    #[test]
    fn winding_number_outside_gap() {
        let ptc = default_ptc();
        let omega0_far = ptc.omega_mod * 3.0; // far from resonance
        assert_eq!(ptc.temporal_winding_number(omega0_far), 0);
    }

    #[test]
    fn base_index_correct() {
        let ptc = default_ptc();
        let n = ptc.base_index();
        assert!(
            (n - 1.5).abs() < 1e-10,
            "base index should be √2.25 = 1.5: {n}"
        );
    }

    #[test]
    fn spatiotemporal_isolation_positive() {
        let stc = SpatiotemporalCrystal {
            spatial_period_m: 500e-9,
            temporal_period_s: 1e-12,
            eps_00: 2.25,
            eps_10: 0.1,
            eps_01: 0.1,
            eps_11: 0.2,
        };
        let iso = stc.isolation_ratio_db();
        assert!(iso > 0.0, "isolation should be positive: {iso}");
    }
}
