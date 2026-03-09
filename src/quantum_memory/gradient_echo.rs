//! Gradient Echo Memory (GEM) quantum memory.
//!
//! GEM applies an external magnetic (or electric) field gradient along the
//! storage medium to create a controlled reversible inhomogeneous broadening.
//! Reversing the gradient at time τ causes a photon echo at t = 2τ.  Unlike
//! the AFC protocol, GEM allows on-demand recall at arbitrary times ≤ T_2*.
//!
//! References:
//! - Alexander et al., PRL 96, 043602 (2006): original GEM experiment
//! - Hetet et al., PRL 100, 023601 (2008): GEM with warm vapour
//! - Buchler et al., Opt. Lett. 35, 1091 (2010): efficiency analysis
//! - Sparkes et al., PRA 82, 043847 (2010): Raman GEM theory

use std::f64::consts::PI;

/// Gyromagnetic ratio of the electron (rad / (s · T))
const GAMMA_E_RAD_S_T: f64 = 1.760859e11;

// ─── Gradient echo memory ─────────────────────────────────────────────────────

/// Gradient Echo Memory (GEM).
///
/// A longitudinal field gradient G creates a linear frequency chirp across the
/// medium, broadening the absorption by B ≈ γ G L / (2π).  Reversing G causes
/// all dipoles to rephase and re-emit the stored field as an echo.
#[derive(Debug, Clone)]
pub struct GradientEchoMemory {
    /// Resonant optical depth of the ensemble
    pub optical_depth: f64,
    /// Medium length (m)
    pub length_m: f64,
    /// Memory bandwidth B  (Hz); set by the gradient strength
    pub bandwidth_hz: f64,
    /// Spin-wave decoherence rate γ_c  (rad/s)
    pub decoherence_rate: f64,
}

impl GradientEchoMemory {
    /// Storage efficiency.
    ///
    /// In the high-OD limit η → 1.  General expression:
    ///
    /// η_store = 1 − exp(−OD)
    pub fn storage_efficiency(&self) -> f64 {
        1.0 - (-self.optical_depth).exp()
    }

    /// Retrieval efficiency.
    ///
    /// For ideal gradient reversal the retrieval mirrors the storage step.
    /// η_ret ≈ 1 − exp(−OD)  (same as storage in the lossless limit).
    pub fn retrieval_efficiency(&self) -> f64 {
        self.storage_efficiency()
    }

    /// Total round-trip efficiency: η_total = η_store × η_ret.
    pub fn total_efficiency(&self) -> f64 {
        let eta = self.storage_efficiency();
        eta * eta
    }

    /// Echo time: t_echo = 2τ where τ is the gradient application duration.
    #[inline]
    pub fn echo_time_s(&self, gradient_time_s: f64) -> f64 {
        2.0 * gradient_time_s.max(0.0)
    }

    /// Spin-wave lifetime T_2*  (s).
    ///
    /// Limited by residual inhomogeneous broadening after the gradient reversal:
    ///
    /// T_2* = 1 / (π × Δν_inhom) = 1 / γ_c  (treating γ_c = π Δν_inhom)
    pub fn spin_wave_lifetime_s(&self) -> f64 {
        1.0 / self.decoherence_rate.max(f64::MIN_POSITIVE)
    }

    /// Multimode storage capacity N = OD × B × τ.
    ///
    /// Each temporal mode occupies a bandwidth B/N_modes, and the gradient
    /// application time τ sets the time window.
    pub fn multimode_capacity(&self, gradient_time_s: f64) -> f64 {
        self.optical_depth * self.bandwidth_hz * gradient_time_s.max(0.0)
    }

    /// GEM allows on-demand recall (unlike AFC which has fixed echo time).
    #[inline]
    pub fn is_on_demand(&self) -> bool {
        true
    }

    /// Required magnetic field gradient  G = 2π B / (γ_e L)  (T/m).
    ///
    /// `g_factor` accounts for the Landé g-factor of the specific transition.
    pub fn required_gradient_t_m(&self, g_factor: f64) -> f64 {
        let gyro = g_factor * GAMMA_E_RAD_S_T;
        2.0 * PI * self.bandwidth_hz / (gyro.max(f64::MIN_POSITIVE) * self.length_m.max(f64::MIN_POSITIVE))
    }

    /// Efficiency vs optical-depth curve: returns Vec<(OD, η_total)>.
    ///
    /// Useful for design trade-off studies: increasing OD always improves
    /// single-pass efficiency but may worsen noise via spontaneous emission.
    pub fn efficiency_vs_od(od_max: f64, n_points: usize) -> Vec<(f64, f64)> {
        let n = n_points.max(2);
        (1..=n)
            .map(|i| {
                let od = od_max * i as f64 / n as f64;
                let eta_single = 1.0 - (-od).exp();
                (od, eta_single * eta_single)
            })
            .collect()
    }
}

// ─── Off-resonant Raman GEM ───────────────────────────────────────────────────

/// Off-resonant Raman GEM for telecom-bandwidth quantum memory.
///
/// A strong off-resonant control field with Rabi frequency Ω_c and detuning Δ
/// (|Δ| ≫ γ_e) creates an effective two-photon coupling between the signal field
/// and a long-lived spin wave.  This removes single-photon absorption noise.
#[derive(Debug, Clone)]
pub struct RamanGem {
    /// Control Rabi frequency Ω_c  (rad/s)
    pub control_rabi_freq: f64,
    /// Single-photon detuning Δ from the excited state  (rad/s); Δ ≫ γ_e
    pub raman_detuning: f64,
    /// Resonant optical depth of the medium
    pub optical_depth: f64,
    /// Memory bandwidth (Hz)
    pub bandwidth_hz: f64,
}

impl RamanGem {
    /// Effective two-photon coupling strength.
    ///
    /// g_eff = Ω_c² / (4 Δ)  (rad/s)
    pub fn effective_coupling(&self) -> f64 {
        self.control_rabi_freq * self.control_rabi_freq
            / (4.0 * self.raman_detuning.abs().max(f64::MIN_POSITIVE))
    }

    /// Adiabatic storage condition: Ω_c ≫ |dΩ_c/dt| / Ω_c.
    ///
    /// Approximated as Ω_c × τ_pulse ≫ 1 (pulse area ≫ 1 radian).
    pub fn is_adiabatic(&self, pulse_duration_s: f64) -> bool {
        self.control_rabi_freq * pulse_duration_s.max(0.0) > 1.0
    }

    /// Spontaneous-emission noise per stored photon.
    ///
    /// The probability that the stored photon causes a spontaneous decay is
    /// suppressed by the large detuning:
    ///
    /// P_se ≈ γ_e / (4 Δ) × OD
    ///
    /// Here γ_e is approximated from the Rabi frequency ratio.
    pub fn spontaneous_emission_noise(&self) -> f64 {
        // Natural linewidth γ_e estimated from the context: for Rb-87 D2 ~ 2π × 6 MHz
        // We express in terms of the available parameters:
        //   P_se = (γ_e / (4 |Δ|)) × OD
        // Use effective coupling: g_eff = Ω_c² / (4|Δ|) → |Δ| = Ω_c²/(4 g_eff)
        // For a generic estimate we set γ_e ≈ Ω_c / 10 (weak saturation regime)
        let gamma_e_est = self.control_rabi_freq / 10.0;
        gamma_e_est / (4.0 * self.raman_detuning.abs().max(f64::MIN_POSITIVE))
            * self.optical_depth
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_gem() -> GradientEchoMemory {
        GradientEchoMemory {
            optical_depth: 5.0,
            length_m: 0.05,   // 5 cm cell
            bandwidth_hz: 1e6, // 1 MHz
            decoherence_rate: 2.0 * PI * 1e3, // 1 kHz decay
        }
    }

    #[test]
    fn gem_storage_efficiency_high_od() {
        let gem = test_gem();
        let eta = gem.storage_efficiency();
        // OD=5 → η ≈ 1 − e^{−5} ≈ 0.993
        assert!(
            (eta - (1.0 - (-5.0_f64).exp())).abs() < 1e-10,
            "η={}",
            eta
        );
    }

    #[test]
    fn gem_total_efficiency_less_than_storage() {
        let gem = test_gem();
        assert!(gem.total_efficiency() <= gem.storage_efficiency());
    }

    #[test]
    fn gem_echo_time_twice_gradient_time() {
        let gem = test_gem();
        let tau = 1e-6_f64; // 1 µs
        let t_echo = gem.echo_time_s(tau);
        assert!(
            (t_echo - 2e-6).abs() < 1e-15,
            "t_echo={} s",
            t_echo
        );
    }

    #[test]
    fn gem_spin_wave_lifetime_milliseconds() {
        let gem = test_gem();
        let t2 = gem.spin_wave_lifetime_s();
        // γ_c = 2π × 1 kHz → T_2* = 1/(2π×1e3) ≈ 159 µs
        assert!(t2 > 1e-6 && t2 < 1.0, "T_2*={}", t2);
    }

    #[test]
    fn gem_multimode_capacity_positive() {
        let gem = test_gem();
        let n = gem.multimode_capacity(1e-4); // 100 µs gradient
        assert!(n > 0.0, "N_modes={}", n);
    }

    #[test]
    fn gem_is_on_demand() {
        let gem = test_gem();
        assert!(gem.is_on_demand());
    }

    #[test]
    fn gem_required_gradient_reasonable() {
        let gem = test_gem();
        // g_factor = 1 (electron)
        let g = gem.required_gradient_t_m(1.0);
        // Expect < 1 T/m for MHz bandwidth over cm cell
        assert!(g > 0.0 && g < 10.0, "G={} T/m", g);
    }

    #[test]
    fn gem_efficiency_vs_od_monotone() {
        let curve = GradientEchoMemory::efficiency_vs_od(10.0, 50);
        assert_eq!(curve.len(), 50);
        for w in curve.windows(2) {
            assert!(
                w[1].1 >= w[0].1,
                "efficiency should increase with OD: {:?}",
                w
            );
        }
    }

    #[test]
    fn raman_gem_effective_coupling() {
        let raman = RamanGem {
            control_rabi_freq: 2.0 * PI * 1e9, // 1 GHz Rabi
            raman_detuning: 2.0 * PI * 10e9,   // 10 GHz detuning
            optical_depth: 100.0,
            bandwidth_hz: 1e9,
        };
        let g_eff = raman.effective_coupling();
        // g_eff = Ω_c²/(4Δ) = (2π×1e9)²/(4×2π×10e9)
        assert!(g_eff > 0.0, "g_eff={}", g_eff);
    }

    #[test]
    fn raman_gem_adiabaticity() {
        let raman = RamanGem {
            control_rabi_freq: 2.0 * PI * 1e9,
            raman_detuning: 2.0 * PI * 10e9,
            optical_depth: 100.0,
            bandwidth_hz: 1e9,
        };
        // Pulse duration 1 µs: Ω_c × τ = 2π×1e9 × 1e-6 ≫ 1 → adiabatic
        assert!(raman.is_adiabatic(1e-6));
        // Very short pulse: not adiabatic
        assert!(!raman.is_adiabatic(1e-12));
    }
}
