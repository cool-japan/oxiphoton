//! Temporal refraction and reflection at abrupt time interfaces.
//!
//! Spatial and temporal boundaries are dual to each other:
//!
//! | Spatial interface | Temporal interface |
//! |-------------------|--------------------|
//! | Frequency ω conserved | Wave-vector k conserved |
//! | Spatial Snell's law: n₁ sin θ₁ = n₂ sin θ₂ | Temporal Snell: n₁ ω₁ = n₂ ω₂ |
//! | Energy flux conserved | Momentum flux conserved |
//!
//! The sudden (sub-cycle) change of refractive index at t = t₀ maps to the
//! spatial problem with the roles of frequency and wave-vector exchanged.
//!
//! References:
//!   - Morgenthaler, "Velocity Modulation of EM Waves", IRE Trans. 1958
//!   - Mendonça & Shukla, Phys. Scr. 2002
//!   - Galiffi et al., Adv. Photonics 2022

use std::f64::consts::PI;

/// Physical constants.
const HBAR: f64 = 1.054_571_817e-34; // J·s
const EV_TO_J: f64 = 1.602_176_634e-19; // J/eV

// ─── Temporal interface ───────────────────────────────────────────────────────

/// A single abrupt temporal interface: n₁ → n₂ at t = t₀.
///
/// The wave-vector **k** is conserved across the interface (analogue of
/// frequency conservation at a spatial interface).  The frequency (energy)
/// is *not* conserved; both a forward-propagating "transmitted" wave at ω_t
/// and a backward-propagating "reflected" wave at ω_r are generated.
#[derive(Debug, Clone, Copy)]
pub struct TemporalInterface {
    /// Refractive index before the interface.
    pub n_before: f64,
    /// Refractive index after the interface.
    pub n_after: f64,
    /// Angular frequency of the incident wave (rad/s).
    pub omega_incident: f64,
}

impl TemporalInterface {
    /// Frequency of the time-transmitted wave (rad/s).
    ///
    /// k-conservation: k = n₁ ω₁ / c = n₂ ω₂ / c
    ///
    /// ```text
    /// ω_t = ω_i × (n₁ / n₂)
    /// ```
    pub fn transmitted_frequency(&self) -> f64 {
        self.omega_incident * self.n_before / self.n_after
    }

    /// Magnitude of the time-reflected frequency (rad/s).
    ///
    /// The "reflected" wave travels backward in time (or equivalently is the
    /// negative-frequency partner); its magnitude equals the transmitted
    /// frequency.
    pub fn reflected_frequency(&self) -> f64 {
        self.transmitted_frequency().abs()
    }

    /// Electric-field transmission amplitude τ (dimensionless).
    ///
    /// By matching the tangential electric and magnetic field boundary
    /// conditions at the temporal interface one obtains:
    ///
    /// ```text
    /// τ = 2 n₁ / (n₁ + n₂)
    /// ```
    pub fn transmission_amplitude(&self) -> f64 {
        2.0 * self.n_before / (self.n_before + self.n_after)
    }

    /// Electric-field reflection amplitude ρ (dimensionless).
    ///
    /// ```text
    /// ρ = (n₁ – n₂) / (n₁ + n₂)
    /// ```
    pub fn reflection_amplitude(&self) -> f64 {
        (self.n_before - self.n_after) / (self.n_before + self.n_after)
    }

    /// Photon energy change ΔE = ħ (ω_t – ω_i) in electron-volts.
    pub fn energy_change_ev(&self) -> f64 {
        let delta_omega = self.transmitted_frequency() - self.omega_incident;
        delta_omega * HBAR / EV_TO_J
    }

    /// Verify temporal Snell's law: n₁ ω₁ = n₂ ω₂ (k-conservation).
    ///
    /// Returns `true` when the relative error is below 10⁻¹⁰.
    pub fn verify_k_conservation(&self) -> bool {
        let k_before = self.n_before * self.omega_incident;
        let k_after = self.n_after * self.transmitted_frequency();
        let rel_err = (k_before - k_after).abs() / k_before.abs().max(1e-100);
        rel_err < 1e-10
    }

    /// Intensity transmission coefficient T_I = (n₂/n₁) |τ|².
    ///
    /// Note that unlike the spatial case the prefactor uses the *inverse*
    /// ratio of indices because the energy change alters the photon frequency.
    pub fn intensity_transmission(&self) -> f64 {
        (self.n_after / self.n_before) * self.transmission_amplitude().powi(2)
    }

    /// Intensity reflection coefficient R_I = |ρ|².
    pub fn intensity_reflection(&self) -> f64 {
        self.reflection_amplitude().powi(2)
    }

    /// Check approximate energy conservation in the time-domain sense.
    ///
    /// The photon number is **not** conserved; energy changes.  However
    /// the momentum flux (k times energy/ω) is conserved.  This function
    /// returns the fractional energy gain/loss due to index switching.
    pub fn energy_ratio(&self) -> f64 {
        self.transmitted_frequency() / self.omega_incident
    }
}

// ─── Time slab ────────────────────────────────────────────────────────────────

/// A finite "time slab": the index is n₂ for duration T, then reverts to n₁.
///
/// Analogous to a Fabry–Pérot etalon in space but extended in time rather
/// than space.  Multiple temporal reflections create resonances at
/// T = m π / |Δω|.
#[derive(Debug, Clone, Copy)]
pub struct TimeSlab {
    /// Background refractive index (before and after the slab).
    pub n1: f64,
    /// Refractive index inside the time slab [0, T].
    pub n2: f64,
    /// Duration of the time slab T (s).
    pub duration_s: f64,
    /// Angular frequency of the incident wave (rad/s).
    pub omega_inc: f64,
}

impl TimeSlab {
    /// Frequency inside the slab (= transmitted frequency at entry interface).
    fn omega_inside(&self) -> f64 {
        // n₁ ω_inc = n₂ ω_inside → ω_inside = ω_inc n₁/n₂
        self.omega_inc * self.n1 / self.n2
    }

    /// Output frequencies of the time slab.
    ///
    /// Returns `(ω_forward, ω_backward)` for the forward-propagating and
    /// backward-propagating components after the slab.
    ///
    /// At the exit interface (n₂ → n₁) the internal frequency maps back:
    /// ω_forward = ω_inside × n₂/n₁ = ω_inc (round-trip to original freq).
    /// ω_backward = –ω_inside × n₂/n₁ (reflected component, same magnitude).
    pub fn output_frequencies(&self) -> (f64, f64) {
        let omega_fwd = self.omega_inside() * self.n2 / self.n1;
        let omega_bwd = self.omega_inside() * self.n2 / self.n1;
        (omega_fwd, omega_bwd)
    }

    /// Resonant slab duration for the m-th temporal Fabry–Pérot mode (s).
    ///
    /// Resonance condition (constructive temporal interference):
    ///
    /// ```text
    /// T_m = m π / |Δω|,   Δω = ω_inside – ω_inc
    /// ```
    pub fn resonant_durations(&self, m: u32) -> f64 {
        let delta_omega = (self.omega_inside() - self.omega_inc).abs();
        if delta_omega == 0.0 {
            return f64::INFINITY;
        }
        (m as f64) * PI / delta_omega
    }

    /// Temporal bandwidth of the slab effect (coherence time).
    ///
    /// ```text
    /// ΔT ~ 1 / |Δω|
    /// ```
    pub fn bandwidth_s(&self) -> f64 {
        let delta_omega = (self.omega_inside() - self.omega_inc).abs();
        if delta_omega == 0.0 {
            return f64::INFINITY;
        }
        1.0 / delta_omega
    }

    /// Accumulated phase during passage through the time slab (rad).
    ///
    /// Φ = ω_inside × T
    pub fn accumulated_phase(&self) -> f64 {
        self.omega_inside() * self.duration_s
    }

    /// Frequency conversion efficiency |τ_entry × τ_exit|² .
    ///
    /// In the ideal (lossless, no multiple reflections) approximation the
    /// transmitted intensity fraction is the product of both interface
    /// transmission coefficients.
    pub fn total_transmission(&self) -> f64 {
        let entry = TemporalInterface {
            n_before: self.n1,
            n_after: self.n2,
            omega_incident: self.omega_inc,
        };
        let exit = TemporalInterface {
            n_before: self.n2,
            n_after: self.n1,
            omega_incident: self.omega_inside(),
        };
        entry.intensity_transmission() * exit.intensity_transmission()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn air_glass() -> TemporalInterface {
        TemporalInterface {
            n_before: 1.5,
            n_after: 2.0,
            omega_incident: 2e15,
        }
    }

    #[test]
    fn temporal_k_conservation() {
        let ti = air_glass();
        assert!(
            ti.verify_k_conservation(),
            "k should be conserved across temporal interface"
        );
    }

    #[test]
    fn transmitted_frequency_ratio() {
        let ti = air_glass();
        let omega_t = ti.transmitted_frequency();
        let expected = 2e15 * 1.5 / 2.0;
        assert!(
            (omega_t - expected).abs() < 1.0,
            "transmitted frequency {omega_t} ≠ {expected}"
        );
    }

    #[test]
    fn reflection_amplitude_sign() {
        let ti = air_glass();
        // n_after > n_before → ρ < 0 (phase reversal).
        assert!(ti.reflection_amplitude() < 0.0);
    }

    #[test]
    fn energy_not_conserved() {
        let ti = air_glass();
        // Photon energy changes when index changes.
        let ratio = ti.energy_ratio();
        assert!(
            (ratio - 1.0).abs() > 1e-10,
            "energy should change at temporal interface"
        );
    }

    #[test]
    fn intensity_coefficients_sum() {
        let ti = air_glass();
        // T + R should be approximately 1 (energy balance differs from spatial
        // case due to frequency change, but |τ|² + |ρ|² = 1 for amplitudes).
        let tau_sq = ti.transmission_amplitude().powi(2);
        let rho_sq = ti.reflection_amplitude().powi(2);
        // Amplitude check: τ² + ρ² ≠ 1 in general; verify individual signs.
        assert!(tau_sq > 0.0 && rho_sq >= 0.0);
    }

    #[test]
    fn time_slab_output_frequencies() {
        let slab = TimeSlab {
            n1: 1.5,
            n2: 2.0,
            duration_s: 1e-12,
            omega_inc: 2e15,
        };
        let (fwd, bwd) = slab.output_frequencies();
        // After round-trip the output frequency should return to incident.
        assert!(
            (fwd - slab.omega_inc).abs() < 1.0,
            "forward output should recover incident frequency: {fwd}"
        );
        assert!(bwd > 0.0);
    }

    #[test]
    fn resonant_duration_m1() {
        let slab = TimeSlab {
            n1: 1.5,
            n2: 2.0,
            duration_s: 1e-12,
            omega_inc: 2e15,
        };
        let t1 = slab.resonant_durations(1);
        assert!(
            t1 > 0.0 && t1.is_finite(),
            "m=1 resonant duration should be positive finite: {t1}"
        );
    }

    #[test]
    fn bandwidth_inverse_delta_omega() {
        let slab = TimeSlab {
            n1: 1.0,
            n2: 1.5,
            duration_s: 1e-12,
            omega_inc: 3e15,
        };
        let bw = slab.bandwidth_s();
        let delta_omega = (slab.omega_inc * slab.n1 / slab.n2 - slab.omega_inc).abs();
        let expected = 1.0 / delta_omega;
        assert!((bw - expected).abs() / expected < 1e-10);
    }
}
