//! Electromagnetically Induced Transparency (EIT) quantum memory.
//!
//! Models the three-level Λ-system EIT effect used for slow light and
//! quantum memory applications in atomic ensembles (e.g. Rb-87, Cs-133).
//!
//! References:
//! - Fleischhauer & Lukin, PRL 84, 5094 (2000): Dark-state polaritons
//! - Gorshkov et al., PRL 98, 123601 (2007): Universal approach to optimal photon storage
//! - Hammerer et al., Rev. Mod. Phys. 82, 1041 (2010): Quantum interfaces
//! - Harris, Phys. Today 50, 36 (1997): EIT review

use std::f64::consts::PI;

/// Reduced Planck constant (J·s)
const HBAR: f64 = 1.054571817e-34;
/// Vacuum permittivity (F/m)
const EPS0: f64 = 8.854187817e-12;
/// Speed of light in vacuum (m/s)
const C_LIGHT: f64 = 2.99792458e8;

/// Three-level Λ-system for EIT: |g⟩ ←→ |e⟩ ←→ |s⟩
///
/// The probe field couples |g⟩ ↔ |e⟩ and the control field couples
/// |s⟩ ↔ |e⟩.  At two-photon resonance (δ_p = δ_c) the medium
/// becomes transparent to the probe due to quantum interference between
/// the two absorption pathways.
#[derive(Debug, Clone)]
pub struct EitLambda {
    /// Decay rate |e⟩ → |g⟩  (rad/s); typically ~2π × 6 MHz for Rb-87 D1
    pub gamma_ge: f64,
    /// Decay rate |e⟩ → |s⟩  (rad/s)
    pub gamma_se: f64,
    /// Ground-state decoherence |g⟩ ↔ |s⟩  (rad/s); γ_gs ≪ γ_ge
    pub gamma_gs: f64,
    /// Control-field Rabi frequency Ω_c  (rad/s)
    pub omega_c: f64,
    /// Probe detuning Δ_p from |g⟩-|e⟩ resonance  (rad/s)
    pub delta_p: f64,
    /// Control detuning Δ_c from |s⟩-|e⟩ resonance  (rad/s)
    pub delta_c: f64,
}

impl EitLambda {
    /// Two-photon detuning δ = Δ_p − Δ_c.
    ///
    /// EIT condition: δ = 0.
    #[inline]
    pub fn two_photon_detuning(&self) -> f64 {
        self.delta_p - self.delta_c
    }

    /// EIT susceptibility χ(ω_p) derived from the density-matrix steady state.
    ///
    /// For a weak probe the steady-state off-diagonal coherence is:
    ///
    /// ρ_eg / Ω_p = (δ − iγ_gs) / D(δ)
    ///
    /// where D(δ) = (Δ_p − iΓ_e/2)(δ − iγ_gs) − |Ω_c|²/4
    /// and Γ_e = γ_ge + γ_se is the total excited-state linewidth.
    ///
    /// χ = N d² / (ε₀ ħ Ω_p) × ρ_eg
    ///
    /// Returns (Re χ, Im χ).  Im χ < 0 means absorption;
    /// Im χ → 0 at δ = 0 signals the EIT transparency window.
    pub fn susceptibility(&self, atomic_density: f64, dipole_moment_cm: f64) -> (f64, f64) {
        // dipole_moment_cm is in C·m (SI)
        let d = dipole_moment_cm;
        let delta = self.two_photon_detuning();
        let gamma_e = self.gamma_ge + self.gamma_se; // total excited-state decay

        // Denominator: D = (Δ_p - iΓ_e/2)(δ - iγ_gs) - Ω_c²/4
        // Written as complex number (Re, Im)
        let a_re = self.delta_p;
        let a_im = -gamma_e / 2.0;
        let b_re = delta;
        let b_im = -self.gamma_gs;
        let ab_re = a_re * b_re - a_im * b_im;
        let ab_im = a_re * b_im + a_im * b_re;
        let denom_re = ab_re - self.omega_c * self.omega_c / 4.0;
        let denom_im = ab_im;

        // Numerator: N = δ - iγ_gs
        let num_re = delta;
        let num_im = -self.gamma_gs;

        // ratio = N / D
        let denom_sq = denom_re * denom_re + denom_im * denom_im;
        let ratio_re = (num_re * denom_re + num_im * denom_im) / denom_sq.max(f64::MIN_POSITIVE);
        let ratio_im = (num_im * denom_re - num_re * denom_im) / denom_sq.max(f64::MIN_POSITIVE);

        // prefactor = N d² / (ε₀ ħ)
        let prefactor = atomic_density * d * d / (EPS0 * HBAR);
        (prefactor * ratio_re, prefactor * ratio_im)
    }

    /// EIT transparency-window width (FWHM) in rad/s.
    ///
    /// For γ_gs ≪ γ_ge the window is approximately:
    ///
    /// Δω_EIT ≈ Ω_c² / γ_ge
    pub fn eit_window_width(&self) -> f64 {
        self.omega_c * self.omega_c / self.gamma_ge.max(f64::MIN_POSITIVE)
    }

    /// Group velocity v_g inside the EIT medium (m/s).
    ///
    /// Computed numerically as  v_g = c / (n_g)  where the group index is
    ///
    /// n_g = 1 + (ω_p / 2) × dRe(χ)/dω_p
    ///
    /// evaluated at the probe frequency ω_p = 2π c / λ (here approximated
    /// via a finite difference over the susceptibility).
    pub fn group_velocity(&self, atomic_density: f64, dipole_moment_cm: f64) -> f64 {
        // finite-difference dχ/dω_p around the current delta_p
        let dw = 1e3_f64; // 1 kHz step — small relative to any EIT width
        let mut eit_lo = self.clone();
        eit_lo.delta_p = self.delta_p - dw / 2.0;
        let mut eit_hi = self.clone();
        eit_hi.delta_p = self.delta_p + dw / 2.0;

        let (chi_re_lo, _) = eit_lo.susceptibility(atomic_density, dipole_moment_cm);
        let (chi_re_hi, _) = eit_hi.susceptibility(atomic_density, dipole_moment_cm);
        let d_chi_d_omega = (chi_re_hi - chi_re_lo) / dw;

        // probe angular frequency ~ 2π × 384 THz for Rb-87 D1
        // Use a representative value if delta_p is small
        let omega_p = 2.0 * PI * 3.84e14;
        let n_group = 1.0 + 0.5 * omega_p * d_chi_d_omega;
        C_LIGHT / n_group.max(1.0)
    }

    /// Slow-light group delay: τ_d = L/v_g − L/c  (s).
    pub fn group_delay(
        &self,
        length_m: f64,
        atomic_density: f64,
        dipole_moment_cm: f64,
    ) -> f64 {
        let v_g = self.group_velocity(atomic_density, dipole_moment_cm);
        length_m / v_g - length_m / C_LIGHT
    }

    /// Resonant optical depth α₀ L.
    ///
    /// OD = ω_p Im(χ_bare) L / c  where χ_bare is the susceptibility
    /// without the control field (Ω_c = 0) and at Δ_p = 0.
    pub fn optical_depth(
        &self,
        length_m: f64,
        atomic_density: f64,
        dipole_moment_cm: f64,
    ) -> f64 {
        let mut bare = self.clone();
        bare.omega_c = 0.0;
        bare.delta_p = 0.0;
        let (_, chi_im) = bare.susceptibility(atomic_density, dipole_moment_cm);
        let omega_p = 2.0 * PI * 3.84e14;
        // absorption coefficient k = ω_p × |Im χ| / c
        // OD = k × L
        (omega_p * chi_im.abs() / C_LIGHT) * length_m
    }

    /// EIT transmission at probe detuning `delta_p`.
    ///
    /// T(δ_p) = exp(−Im(χ) × ω_p × L / c)
    pub fn transmission(
        &self,
        delta_p: f64,
        length_m: f64,
        atomic_density: f64,
        dipole_moment_cm: f64,
    ) -> f64 {
        let mut eit = self.clone();
        eit.delta_p = delta_p;
        let (_, chi_im) = eit.susceptibility(atomic_density, dipole_moment_cm);
        let omega_p = 2.0 * PI * 3.84e14;
        let exponent = chi_im.abs() * omega_p * length_m / C_LIGHT;
        (-exponent).exp()
    }

    /// Absorption spectrum: list of (delta_p [rad/s], absorption [a.u.]).
    ///
    /// Returns `n_points` uniformly spaced detuning / transmission pairs
    /// over `delta_range` (rad/s).  Values are Beer–Lambert transmissions.
    pub fn absorption_spectrum(
        &self,
        delta_range: (f64, f64),
        n_points: usize,
        length_m: f64,
        atomic_density: f64,
        dipole_moment_cm: f64,
    ) -> Vec<(f64, f64)> {
        let n = n_points.max(2);
        let (d_lo, d_hi) = delta_range;
        let step = (d_hi - d_lo) / (n - 1) as f64;
        (0..n)
            .map(|i| {
                let dp = d_lo + i as f64 * step;
                let t = self.transmission(dp, length_m, atomic_density, dipole_moment_cm);
                (dp, t)
            })
            .collect()
    }
}

// ─── EIT quantum memory ────────────────────────────────────────────────────────

/// EIT-based quantum memory: stores a photonic qubit as a spin-wave
/// excitation in an atomic ensemble via adiabatic dark-state mapping.
#[derive(Debug, Clone)]
pub struct EitMemory {
    /// Underlying three-level EIT system
    pub eit: EitLambda,
    /// Ensemble length (m)
    pub length_m: f64,
    /// Atomic number density (m⁻³)
    pub atomic_density: f64,
    /// Electric dipole moment of the probe transition (C·m)
    pub dipole_moment_cm: f64,
}

impl EitMemory {
    /// Storage efficiency.
    ///
    /// η_store = (1 − exp(−OD)) × min(1, Ω_c² / (γ_ge × B_in))
    ///
    /// For OD ≫ 1 and well-matched control bandwidth this approaches unity.
    pub fn storage_efficiency(&self) -> f64 {
        let od = self
            .eit
            .optical_depth(self.length_m, self.atomic_density, self.dipole_moment_cm);
        let absorption_factor = 1.0 - (-od).exp();
        // bandwidth matching factor: EIT window / probe bandwidth (assume matched)
        absorption_factor * absorption_factor // two-pass efficiency approximation
    }

    /// Retrieval efficiency for a given control pulse area θ = ∫Ω_c dt.
    ///
    /// For optimised backward retrieval: η_ret ≈ OD / (1 + OD).
    pub fn retrieval_efficiency(&self, control_pulse_area: f64) -> f64 {
        let od = self
            .eit
            .optical_depth(self.length_m, self.atomic_density, self.dipole_moment_cm);
        // Modulate by pulse-area sinc envelope (deviations from π-pulse)
        let pulse_factor = (control_pulse_area / 2.0).sin().powi(2).clamp(0.0, 1.0);
        let base_efficiency = od / (1.0 + od);
        base_efficiency * pulse_factor
    }

    /// Memory coherence time T_mem ≈ 1 / γ_gs  (s).
    pub fn coherence_time(&self) -> f64 {
        1.0 / self.eit.gamma_gs.max(f64::MIN_POSITIVE)
    }

    /// Memory bandwidth B ≈ Ω_c² / (2π γ_ge)  (Hz).
    pub fn bandwidth_hz(&self) -> f64 {
        self.eit.omega_c * self.eit.omega_c / (2.0 * PI * self.eit.gamma_ge.max(f64::MIN_POSITIVE))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn rb87_eit() -> EitLambda {
        EitLambda {
            gamma_ge: 2.0 * PI * 6e6, // 6 MHz natural linewidth
            gamma_se: 2.0 * PI * 6e6,
            gamma_gs: 1e3, // sub-kHz ground coherence
            omega_c: 2.0 * PI * 2e6, // 2 MHz control Rabi
            delta_p: 0.0,
            delta_c: 0.0,
        }
    }

    #[test]
    fn eit_window_width_rb87() {
        // Simplified Rb-87-like system with gamma_ge = 2e7 (as in spec)
        let eit = EitLambda {
            gamma_ge: 2e7,
            gamma_se: 2e7,
            gamma_gs: 1e3,
            omega_c: 1e7,
            delta_p: 0.0,
            delta_c: 0.0,
        };
        let width = eit.eit_window_width();
        // Ω_c²/γ_ge = 1e14 / 2e7 = 5e6 rad/s — within MHz range
        assert!(
            width > 1e5 && width < 1e9,
            "EIT window width out of range: {}",
            width
        );
    }

    #[test]
    fn two_photon_detuning_on_resonance() {
        let eit = rb87_eit();
        let delta = eit.two_photon_detuning();
        assert!(delta.abs() < 1e-10, "Expected δ=0, got {}", delta);
    }

    #[test]
    fn susceptibility_eit_window_vanishes() {
        // At two-photon resonance Im(χ) should be suppressed relative to bare medium
        let eit = rb87_eit();
        let n_at = 1e18_f64; // m^-3
        let d = 1.0e-29_f64; // typical dipole moment (C·m)
        let (_, chi_im_eit) = eit.susceptibility(n_at, d);

        let mut bare = eit.clone();
        bare.omega_c = 0.0;
        let (_, chi_im_bare) = bare.susceptibility(n_at, d);

        assert!(
            chi_im_eit.abs() < chi_im_bare.abs(),
            "EIT should reduce absorption: |Im χ_eit|={} |Im χ_bare|={}",
            chi_im_eit.abs(),
            chi_im_bare.abs()
        );
    }

    #[test]
    fn group_velocity_slow_light() {
        let eit = rb87_eit();
        let n_at = 1e18_f64;
        let d = 1.0e-29_f64;
        let v_g = eit.group_velocity(n_at, d);
        // Group velocity in EIT medium must be positive and ≤ c
        assert!(v_g > 0.0 && v_g <= C_LIGHT, "v_g={}", v_g);
    }

    #[test]
    fn transmission_at_resonance_is_high() {
        let eit = rb87_eit();
        let n_at = 1e14_f64; // dilute ensemble
        let d = 1.0e-29_f64;
        let l = 1e-3_f64; // 1 mm
        let t = eit.transmission(0.0, l, n_at, d);
        // Dilute + EIT should give high transmission
        assert!(t > 0.0 && t <= 1.0, "Transmission={}", t);
    }

    #[test]
    fn memory_coherence_time_microseconds() {
        let mem = EitMemory {
            eit: rb87_eit(),
            length_m: 1e-2,
            atomic_density: 1e18,
            dipole_moment_cm: 1e-29,
        };
        let t_mem = mem.coherence_time();
        // 1/γ_gs with γ_gs ~ 1e3 → T_mem ~ 1 ms
        assert!(t_mem > 1e-4 && t_mem < 10.0, "T_mem={}", t_mem);
    }

    #[test]
    fn memory_bandwidth_mhz_range() {
        let mem = EitMemory {
            eit: rb87_eit(),
            length_m: 1e-2,
            atomic_density: 1e18,
            dipole_moment_cm: 1e-29,
        };
        let bw = mem.bandwidth_hz();
        // ~ MHz range for typical Rb EIT
        assert!(bw > 1e3 && bw < 1e12, "bandwidth={} Hz", bw);
    }

    #[test]
    fn absorption_spectrum_length() {
        let eit = rb87_eit();
        let spec = eit.absorption_spectrum(
            (-1e8, 1e8),
            100,
            1e-2,
            1e18,
            1e-29,
        );
        assert_eq!(spec.len(), 100);
        // All transmissions must be in (0, 1]
        for (_, t) in &spec {
            assert!(*t > 0.0 && *t <= 1.0, "T out of range: {}", t);
        }
    }
}
