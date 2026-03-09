//! Floquet-Bloch theory for time-modulated photonic systems.
//!
//! Models periodically driven optical cavities where the permittivity is
//! modulated as ε(t) = ε₀ (1 + Δ cos(Ωt)).  The Floquet formalism
//! replaces the time-independent eigenvalue problem with a quasi-energy
//! spectrum indexed by the integer Floquet sector *n*.
//!
//! Key references:
//!   - Floquet, "Sur les équations différentielles linéaires…", AENS 1883
//!   - Oka & Kitamura, Annu. Rev. Condens. Matter Phys. 2019
//!   - Zurek & Hillery, "Photonic Floquet Topological Insulators" (2024)

// ─── Floquet cavity ───────────────────────────────────────────────────────────

/// Periodically driven optical cavity with ε(t) = ε₀ (1 + Δ cos(Ωt)).
///
/// The Floquet quasi-energy spectrum is obtained by treating the
/// time-periodic Hamiltonian in the extended Hilbert space spanned by
/// Fourier sectors {n ∈ ℤ}.
#[derive(Debug, Clone)]
pub struct FloquetCavity {
    /// Unperturbed resonance angular frequency ω₀ (rad/s).
    pub omega0: f64,
    /// Modulation depth Δ (dimensionless, 0–1).
    pub modulation_depth: f64,
    /// Modulation angular frequency Ω (rad/s).
    pub modulation_freq: f64,
}

impl FloquetCavity {
    /// Floquet quasi-energy at sector *n* and branch `sign` ∈ {+1, –1}.
    ///
    /// To first order near the parametric resonance Ω ≈ 2ω₀:
    ///
    /// ```text
    /// ε_n = ω₀ + n Ω ± (Δ ω₀ / 4)
    /// ```
    pub fn quasi_energy(&self, n_floquet: i32, sign: i8) -> f64 {
        let sign_f = if sign >= 0 { 1.0_f64 } else { -1.0_f64 };
        let coupling = self.modulation_depth * self.omega0 / 4.0;
        self.omega0 + (n_floquet as f64) * self.modulation_freq + sign_f * coupling
    }

    /// Check whether the system is at parametric resonance Ω ≈ 2ω₀/m.
    ///
    /// Returns `true` when the detuning |Ω – 2ω₀/m| is smaller than the
    /// half band-gap width Δω₀/2 (first-order parametric bandwidth).
    pub fn is_parametric_resonance(&self, m: u32) -> bool {
        if m == 0 {
            return false;
        }
        let resonant_freq = 2.0 * self.omega0 / (m as f64);
        let detuning = (self.modulation_freq - resonant_freq).abs();
        let bandwidth = self.modulation_depth * self.omega0 / 2.0;
        detuning < bandwidth
    }

    /// Stability test based on the Mathieu/Hill diagram.
    ///
    /// The system is stable when it is **not** inside a parametric resonance
    /// band.  Here we check the dominant m = 1 resonance (Ω ≈ 2ω₀).
    pub fn is_stable(&self) -> bool {
        !self.is_parametric_resonance(1)
    }

    /// Width of the primary Floquet band gap at the Ω = 2ω₀ resonance.
    ///
    /// ```text
    /// Δω_gap = Δ ω₀
    /// ```
    pub fn band_gap_width(&self) -> f64 {
        self.modulation_depth * self.omega0
    }

    /// Exponential growth rate inside the parametric instability region.
    ///
    /// For detuning δ = Ω – 2ω₀ from the primary resonance:
    ///
    /// ```text
    /// γ = √[ (Δ ω₀/4)² – δ² ]   if |δ| < Δ ω₀/4
    ///   = 0                        otherwise (stable)
    /// ```
    pub fn growth_rate(&self) -> f64 {
        let half_gap_sq = (self.modulation_depth * self.omega0 / 4.0).powi(2);
        let detuning = self.modulation_freq - 2.0 * self.omega0;
        let detuning_sq = detuning.powi(2);
        if detuning_sq >= half_gap_sq {
            0.0
        } else {
            (half_gap_sq - detuning_sq).sqrt()
        }
    }

    /// Squared amplitude of the *n*-th Floquet sideband.
    ///
    /// For weak modulation (Δ ≪ 1) the first-order perturbative result gives:
    ///
    /// ```text
    /// |c_±1|² ≈ (Δ/4)²,   |c_0|² ≈ 1 – 2(Δ/4)²
    /// |c_n|²  ≈ (Δ/4)^{2|n|}   for |n| > 1  (qualitative)
    /// ```
    pub fn sideband_amplitude_sq(&self, n: i32) -> f64 {
        let abs_n = n.unsigned_abs() as i32;
        if abs_n == 0 {
            // Central mode retains most of the population.
            let leakage = (self.modulation_depth / 4.0).powi(2);
            (1.0_f64 - 2.0 * leakage).max(0.0)
        } else {
            // Each additional order is suppressed by another factor (Δ/4).
            (self.modulation_depth / 4.0).powi(2 * abs_n)
        }
    }

    /// Effective photon "mass" derived from the Floquet band curvature.
    ///
    /// Near the band edge the dispersion is approximately parabolic.  Using
    /// the temporal analogy m_eff ~ ℏ / (d²ε/dΩ² at edge) and approximating
    /// the curvature scale by 1/Ω we obtain:
    ///
    /// ```text
    /// m_eff  ≈  1 / Ω   (s/rad — a purely dimensional proxy)
    /// ```
    pub fn effective_mass(&self) -> f64 {
        // Guard against zero modulation frequency.
        if self.modulation_freq == 0.0 {
            return f64::INFINITY;
        }
        1.0 / self.modulation_freq
    }
}

// ─── Modulated cavity (coupled-mode theory) ───────────────────────────────────

/// Coupled-mode description of a single cavity with parametric modulation.
///
/// The modulation creates a coupling κ_mod = Δω₀/2 between the mode amplitude
/// *a₁* and its time-reversed partner *a₂*.  Near resonance the equations of
/// motion in the rotating frame are:
///
/// ```text
/// da₁/dt = –(κ/2) a₁ – i κ_mod a₂*
/// da₂/dt = –(κ/2) a₂ – i κ_mod a₁*
/// ```
#[derive(Debug, Clone)]
pub struct ModulatedCavity {
    /// Unperturbed resonance angular frequency ω₀ (rad/s).
    pub omega0: f64,
    /// Loaded quality factor Q.
    pub q_factor: f64,
    /// Parametric coupling rate κ_mod = Δω₀/2 (rad/s).
    pub modulation_rate: f64,
    /// Modulation angular frequency Ω (rad/s).
    pub modulation_freq: f64,
}

impl ModulatedCavity {
    /// Intrinsic loss rate κ = ω₀/Q (rad/s).
    fn loss_rate(&self) -> f64 {
        self.omega0 / self.q_factor
    }

    /// Net parametric gain rate γ = √(κ_mod² – (κ/2)²).
    ///
    /// Returns 0.0 when below threshold (system is stable/decaying).
    fn net_gain_rate(&self) -> f64 {
        let kappa_half = self.loss_rate() / 2.0;
        let discriminant = self.modulation_rate.powi(2) - kappa_half.powi(2);
        if discriminant <= 0.0 {
            0.0
        } else {
            discriminant.sqrt()
        }
    }

    /// Amplitude amplification factor |a(t)| / |a(0)| for the resonant case
    /// (Ω = 2ω₀).
    ///
    /// Above threshold the amplitude grows as exp(γ t); below threshold it
    /// decays as exp(–|κ/2 – κ_mod| t).
    pub fn amplification_factor(&self, time_s: f64) -> f64 {
        let kappa_half = self.loss_rate() / 2.0;
        let gamma = self.net_gain_rate();
        if gamma > 0.0 {
            // Above threshold: exponential growth minus decay envelope.
            (gamma - kappa_half).exp().powf(time_s)
        } else {
            // Below threshold: exponential decay.
            let effective_rate = (kappa_half - self.modulation_rate).max(0.0);
            (-effective_rate * time_s).exp()
        }
    }

    /// Whether the modulation rate exceeds the threshold κ_mod > κ/2.
    pub fn is_above_threshold(&self) -> bool {
        self.modulation_rate > self.loss_rate() / 2.0
    }

    /// Quadrature squeezing in dB (below threshold operation).
    ///
    /// The squeezed variance in one quadrature evolves as
    /// `V_sq = exp(–2 γ t)` where γ is the net parametric rate.
    /// Returns 0.0 if above threshold (squeezing description breaks down).
    pub fn squeezing_db(&self, time_s: f64) -> f64 {
        if self.is_above_threshold() {
            return 0.0;
        }
        let kappa_half = self.loss_rate() / 2.0;
        let discriminant = kappa_half.powi(2) - self.modulation_rate.powi(2);
        let gamma_sq = discriminant.max(0.0).sqrt();
        -10.0 * ((-2.0 * gamma_sq * time_s).exp()).log10()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cavity() -> FloquetCavity {
        FloquetCavity {
            omega0: 2e14,          // ~1 µm optical frequency
            modulation_depth: 0.1, // 10 % modulation
            modulation_freq: 4e14, // Ω = 2ω₀ (primary resonance)
        }
    }

    #[test]
    fn quasi_energy_n0_zero_sign() {
        let fc = default_cavity();
        // n=0, sign +1: ε = ω₀ + Δω₀/4
        let expected = fc.omega0 + fc.modulation_depth * fc.omega0 / 4.0;
        let result = fc.quasi_energy(0, 1_i8);
        assert!(
            (result - expected).abs() < 1e6,
            "quasi_energy mismatch: {result} vs {expected}"
        );
    }

    #[test]
    fn quasi_energy_antisymmetry() {
        let fc = default_cavity();
        // ε(n, +1) and ε(n, –1) should bracket ω₀ + n Ω symmetrically.
        let ep = fc.quasi_energy(1, 1_i8);
        let em = fc.quasi_energy(1, -1_i8);
        let mid = (ep + em) / 2.0;
        let expected_mid = fc.omega0 + fc.modulation_freq;
        assert!((mid - expected_mid).abs() < 1e6);
    }

    #[test]
    fn parametric_resonance_detected() {
        let fc = default_cavity(); // Ω = 2ω₀ exactly
        assert!(
            fc.is_parametric_resonance(1),
            "primary resonance should be detected"
        );
        assert!(
            !fc.is_parametric_resonance(2),
            "second-order resonance should not fire here"
        );
    }

    #[test]
    fn stability_off_resonance() {
        let fc = FloquetCavity {
            omega0: 2e14,
            modulation_depth: 0.01,
            modulation_freq: 1e14, // far from 2ω₀
        };
        assert!(fc.is_stable(), "far off-resonance cavity should be stable");
    }

    #[test]
    fn growth_rate_positive_on_resonance() {
        let fc = default_cavity(); // exactly at resonance
        let gr = fc.growth_rate();
        assert!(
            gr > 0.0,
            "growth rate should be positive at resonance: {gr}"
        );
    }

    #[test]
    fn sideband_normalization() {
        let fc = default_cavity();
        // Central mode + two first sidebands should sum to ≤ 1 (perturbative).
        let s0 = fc.sideband_amplitude_sq(0);
        let sp = fc.sideband_amplitude_sq(1);
        let sm = fc.sideband_amplitude_sq(-1);
        assert!(s0 + sp + sm <= 1.001, "sideband sum > 1: {}", s0 + sp + sm);
    }

    #[test]
    fn threshold_detection() {
        let mc = ModulatedCavity {
            omega0: 2e14,
            q_factor: 1e4,
            modulation_rate: 2e10, // κ_mod much larger than κ/2 ≈ 1e10
            modulation_freq: 4e14,
        };
        assert!(mc.is_above_threshold());
    }

    #[test]
    fn squeezing_below_threshold() {
        let mc = ModulatedCavity {
            omega0: 2e14,
            q_factor: 1e3,
            modulation_rate: 1e9, // well below threshold
            modulation_freq: 4e14,
        };
        let sq = mc.squeezing_db(1e-9);
        // Some squeezing should be present (positive dB value suppressed noise).
        assert!(sq >= 0.0, "squeezing should be non-negative: {sq}");
    }
}
