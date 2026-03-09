//! Jaynes-Cummings model for polaritonics: single emitter coupled to a cavity mode.
//!
//! This implementation is tailored for the polaritonics context — quantum-dot or
//! molecular emitters inside a photonic-crystal or Fabry-Pérot microcavity, where
//! the vacuum Rabi splitting, Purcell factor, and single-photon blockade are the
//! figures of merit of interest.
//!
//! The Jaynes-Cummings Hamiltonian (rotating-wave approximation):
//!
//! ```text
//! H = ħωc a†a + ħωa σ†σ + ħg (aσ† + a†σ)
//! ```
//!
//! # Dressed states
//!
//! For the n-excitation manifold the dressed-state (polariton) energies are:
//!
//! ```text
//! E±,n = ħ[(n + ½)ωc + ωa/2] ± ħ sqrt(g²(n+1) + Δ²/4)
//! ```
//!
//! where Δ = ωa − ωc is the emitter-cavity detuning.
//!
//! # References
//! - E. T. Jaynes & F. W. Cummings, Proc. IEEE 51, 89 (1963)
//! - H. J. Kimble, "Strong interactions of single atoms and photons in cavity QED",
//!   Phys. Scr. T76, 127 (1998)

use std::f64::consts::PI;

// ─── Physical constants ───────────────────────────────────────────────────────
/// Speed of light in vacuum (m/s)
const C_LIGHT: f64 = 2.997_924_58e8;
/// Vacuum permittivity (F/m)
const _EPS0: f64 = 8.854_187_817e-12;

// ─── JaynesCummings ──────────────────────────────────────────────────────────

/// Jaynes-Cummings model for a single two-level emitter coupled to a cavity mode.
///
/// All frequencies are in rad/s (angular frequencies).  Physical units are
/// converted at the boundary (see helper methods).
#[derive(Debug, Clone)]
pub struct JaynesCummings {
    /// Cavity resonance frequency ωc (rad/s).
    pub omega_cavity: f64,
    /// Two-level emitter transition frequency ωa (rad/s).
    pub omega_atom: f64,
    /// Vacuum Rabi coupling rate g = d·E_vac / ħ (rad/s).
    ///
    /// For a quantum dot in a photonic-crystal nanocavity: g/2π ~ 10–100 GHz.
    pub coupling_g: f64,
    /// Cavity field decay rate κ = ωc / (2Q) (rad/s).
    pub kappa: f64,
    /// Emitter spontaneous emission rate γ = 1/τ_sp (rad/s).
    pub gamma: f64,
}

impl JaynesCummings {
    /// Vacuum Rabi splitting Ω_R = 2g (rad/s).
    ///
    /// This is the frequency separation of the two polariton peaks at resonance (Δ=0).
    #[inline]
    pub fn vacuum_rabi_splitting(&self) -> f64 {
        2.0 * self.coupling_g
    }

    /// Emitter-cavity detuning Δ = ωa − ωc (rad/s).
    #[inline]
    pub fn detuning(&self) -> f64 {
        self.omega_atom - self.omega_cavity
    }

    /// Strong-coupling criterion: g > max(κ, γ) / 2.
    ///
    /// In the strong-coupling regime the vacuum Rabi splitting is resolved and
    /// coherent exchange of energy between emitter and cavity field occurs faster
    /// than any decay process.
    pub fn is_strong_coupling(&self) -> bool {
        let max_rate = self.kappa.max(self.gamma);
        self.coupling_g > max_rate / 2.0
    }

    /// Dressed-state (polariton) energies for the n-excitation manifold.
    ///
    /// ```text
    /// E±,n / ħ = (n + ½)ωc + ωa/2 ± sqrt(g²(n+1) + Δ²/4)
    /// ```
    ///
    /// Returns `(E_minus / ħ, E_plus / ħ)` in rad/s units.
    pub fn dressed_state_energies(&self, n: usize) -> (f64, f64) {
        let delta = self.detuning();
        let generalized_rabi = (self.coupling_g * self.coupling_g * (n + 1) as f64
            + delta * delta / 4.0)
            .sqrt();
        let center = (n as f64 + 0.5) * self.omega_cavity + self.omega_atom / 2.0;
        (center - generalized_rabi, center + generalized_rabi)
    }

    /// Vacuum Rabi splitting for the n=1 manifold (first polariton doublet).
    ///
    /// ```text
    /// ΔE_n=1 / ħ = 2 sqrt(g² + Δ²/4)
    /// ```
    pub fn n1_splitting(&self) -> f64 {
        let delta = self.detuning();
        2.0 * (self.coupling_g * self.coupling_g + delta * delta / 4.0).sqrt()
    }

    /// Purcell factor via geometry (mode volume and Q-factor definition).
    ///
    /// ```text
    /// F_P = (3 / 4π²) × (λ/n)³ / V × Q
    /// ```
    ///
    /// This is the standard formula for the Purcell enhancement factor at the
    /// antinode of the cavity mode for an ideal dipole aligned with the field.
    ///
    /// # Parameters
    /// - `q_factor` — cavity quality factor
    /// - `mode_volume_m3` — effective mode volume V (m³)
    /// - `n_medium` — refractive index at the emitter location
    /// - `wavelength_m` — free-space emission wavelength (m)
    pub fn purcell_factor(
        &self,
        q_factor: f64,
        mode_volume_m3: f64,
        n_medium: f64,
        wavelength_m: f64,
    ) -> f64 {
        if mode_volume_m3 <= 0.0 || n_medium <= 0.0 || wavelength_m <= 0.0 {
            return 0.0;
        }
        let lambda_n = wavelength_m / n_medium;
        let cubic_wavelength = lambda_n.powi(3);
        (3.0 / (4.0 * PI * PI)) * (cubic_wavelength / mode_volume_m3) * q_factor
    }

    /// Purcell-enhanced total emitter decay rate Γ_eff = F_P × Γ_0 (rad/s).
    ///
    /// The spontaneous emission rate is enhanced by the Purcell factor into the
    /// cavity mode.  Additional non-radiative channels add to the total.
    pub fn enhanced_decay_rate(&self, purcell_factor: f64, gamma0: f64) -> f64 {
        purcell_factor * gamma0
    }

    /// Cooperativity C = g² / (κγ).
    ///
    /// C > 1 means the coherent coupling dominates over losses.
    /// C > 1 is a necessary (but not sufficient) condition for strong coupling.
    pub fn cooperativity(&self) -> f64 {
        if self.kappa * self.gamma < f64::EPSILON {
            return f64::INFINITY;
        }
        self.coupling_g * self.coupling_g / (self.kappa * self.gamma)
    }

    /// Cavity transmission spectrum T(ω) showing vacuum Rabi splitting.
    ///
    /// Uses the input-output theory expression for the cavity transmission.
    /// In the strong-coupling regime, two Lorentzian peaks appear at ω± = ωc ± g.
    ///
    /// Returns `Vec<(omega_rad_s, T_normalised)>`.
    pub fn transmission_spectrum(
        &self,
        omega_range: (f64, f64),
        n_points: usize,
    ) -> Vec<(f64, f64)> {
        let n = n_points.max(2);
        let (e_minus, e_plus) = self.dressed_state_energies(0);
        let half_kappa = self.kappa / 2.0;
        let decay_sum = half_kappa + self.gamma / 4.0;

        let mut result = Vec::with_capacity(n);
        let mut t_max = 0.0_f64;

        for i in 0..n {
            let omega = omega_range.0
                + (omega_range.1 - omega_range.0) * (i as f64) / ((n - 1) as f64);
            let lorentz = |omega_res: f64| -> f64 {
                let den = (omega - omega_res).powi(2) + decay_sum.powi(2);
                half_kappa * half_kappa / den
            };
            let t = lorentz(e_minus) + lorentz(e_plus);
            result.push((omega, t));
            if t > t_max {
                t_max = t;
            }
        }
        // Normalise to maximum
        if t_max > 0.0 {
            for entry in &mut result {
                entry.1 /= t_max;
            }
        }
        result
    }

    /// Single-photon blockade criterion: g > κ.
    ///
    /// When g > κ, the second photon cannot enter the cavity at the one-photon
    /// resonance because the n=2 manifold is shifted by the anharmonicity
    /// Δ_anh = g(√2 − 1).
    pub fn has_photon_blockade(&self) -> bool {
        self.coupling_g > self.kappa
    }

    /// Jaynes-Cummings energy ladder: |n,±⟩ energies for n = 0 to n_max.
    ///
    /// Returns `Vec<(E_minus / ħ, E_plus / ħ)>` in rad/s for each manifold.
    pub fn energy_ladder(&self, n_max: usize) -> Vec<(f64, f64)> {
        (0..=n_max)
            .map(|n| self.dressed_state_energies(n))
            .collect()
    }

    /// Vacuum Rabi oscillation: excited-state population P_e(t) at resonance.
    ///
    /// Starting from |e, 0⟩ (excited emitter, vacuum field), the excitation
    /// undergoes coherent oscillation:
    ///
    /// ```text
    /// P_e(t) = cos²(g · t)    (Δ = 0 limit)
    /// ```
    ///
    /// Returns the excited-state probability at time t_s (seconds).
    pub fn vacuum_rabi_oscillation(&self, t_s: f64) -> f64 {
        let delta = self.detuning();
        if delta.abs() < f64::EPSILON * self.omega_cavity {
            // Resonant case: simple cosine
            (self.coupling_g * t_s).cos().powi(2)
        } else {
            // General case with detuning
            let omega_gen = (self.coupling_g * self.coupling_g + delta * delta / 4.0).sqrt();
            (omega_gen * t_s).cos().powi(2)
        }
    }
}

// ─── CavityType ─────────────────────────────────────────────────────────────

/// Classification of common cavity QED system architectures.
#[derive(Debug, Clone, PartialEq)]
pub enum CavityType {
    /// Macroscopic optical Fabry-Pérot cavity (mirror spacing ~ cm).
    FabryPerot,
    /// Silica microsphere whispering-gallery mode resonator.
    Microsphere,
    /// Photonic-crystal nanocavity (L3, H1, double heterostructure).
    PhotonicCrystal,
    /// Silica microtoroid on a chip.
    Microtoroid,
    /// Superconducting circuit with transmon qubit (circuit QED).
    SuperconductingCircuit,
}

/// Cavity QED system with complete physical parameters.
///
/// Encapsulates the key figures of merit for a cavity-emitter system:
/// the coupling rate g, cavity decay κ, emitter decay γ, and derived
/// quantities (Q, cooperativity, Purcell factor).
#[derive(Debug, Clone)]
pub struct CavityQedSystem {
    /// System architecture.
    pub system_type: CavityType,
    /// Cavity resonance angular frequency ω₀ (rad/s).
    pub omega_0: f64,
    /// Vacuum Rabi coupling rate g (rad/s).
    pub g_coupling: f64,
    /// Cavity decay rate κ (rad/s).
    pub kappa: f64,
    /// Emitter spontaneous emission rate γ (rad/s).
    pub gamma: f64,
    /// Cavity quality factor Q = ω₀ / κ.
    pub q_factor: f64,
    /// Mode volume in units of (λ/n)³.
    pub mode_volume_lambda3: f64,
}

impl CavityType {
    /// Return representative literature parameters for each system type.
    ///
    /// Parameter ranges are taken from state-of-the-art experiments:
    /// - Fabry-Pérot: Kimble group Cs atom in F-P cavity
    /// - Microsphere: WGM resonator with single emitter
    /// - PhC: InAs QD in GaAs PhC L3 nanocavity
    /// - Microtoroid: Silica toroid with Rb atom
    /// - SC circuit: Transmon qubit in transmission line resonator
    pub fn typical_parameters(&self) -> CavityQedSystem {
        let two_pi = 2.0 * PI;
        match self {
            CavityType::FabryPerot => {
                // Optical F-P: g/2π ~ 16 MHz, κ/2π ~ 4 MHz, γ/2π ~ 2.6 MHz
                // (Ye, Kimble, Katori, Science 2008)
                let omega_0 = two_pi * 3.5e14; // ~850 nm Cs D2 line
                CavityQedSystem {
                    system_type: CavityType::FabryPerot,
                    omega_0,
                    g_coupling: two_pi * 16e6,
                    kappa: two_pi * 4e6,
                    gamma: two_pi * 2.6e6,
                    q_factor: omega_0 / (two_pi * 4e6),
                    mode_volume_lambda3: 1e6, // macroscopic
                }
            }
            CavityType::Microsphere => {
                // WGM microsphere: Q ~ 1e8, V ~ 1e3 (λ/n)³
                let omega_0 = two_pi * C_LIGHT / 1.3e-6;
                let kappa = omega_0 / 1e8;
                CavityQedSystem {
                    system_type: CavityType::Microsphere,
                    omega_0,
                    g_coupling: two_pi * 1e6,
                    kappa,
                    gamma: two_pi * 1e6,
                    q_factor: 1e8,
                    mode_volume_lambda3: 1e3,
                }
            }
            CavityType::PhotonicCrystal => {
                // InAs QD in GaAs PhC L3: g/2π ~ 40 GHz, Q ~ 1e4
                // (Yoshie et al., Nature 2004; Reithmaier et al., Nature 2004)
                let omega_0 = two_pi * C_LIGHT / 920e-9; // 920 nm InAs QD
                let kappa = omega_0 / 1e4;
                CavityQedSystem {
                    system_type: CavityType::PhotonicCrystal,
                    omega_0,
                    g_coupling: two_pi * 40e9,
                    kappa,
                    gamma: two_pi * 1e9,
                    q_factor: 1e4,
                    mode_volume_lambda3: 0.4, // sub-wavelength
                }
            }
            CavityType::Microtoroid => {
                // Silica microtoroid: Q ~ 1e8, g/2π ~ 70 MHz
                let omega_0 = two_pi * C_LIGHT / 780e-9; // Rb D1 line
                let kappa = omega_0 / 1e8;
                CavityQedSystem {
                    system_type: CavityType::Microtoroid,
                    omega_0,
                    g_coupling: two_pi * 70e6,
                    kappa,
                    gamma: two_pi * 3e6,
                    q_factor: 1e8,
                    mode_volume_lambda3: 2e3,
                }
            }
            CavityType::SuperconductingCircuit => {
                // Transmon in λ/4 CPW resonator: g/2π ~ 200 MHz, Q ~ 1e4
                // (Blais et al., PRA 2004; Wallraff et al., Nature 2004)
                let omega_0 = two_pi * 6e9; // 6 GHz microwave
                let kappa = two_pi * 1e6;   // κ/2π ~ 1 MHz
                CavityQedSystem {
                    system_type: CavityType::SuperconductingCircuit,
                    omega_0,
                    g_coupling: two_pi * 200e6,
                    kappa,
                    gamma: two_pi * 1e4, // very small in SC circuits
                    q_factor: omega_0 / kappa,
                    mode_volume_lambda3: f64::NAN, // not applicable at microwave
                }
            }
        }
    }

    /// Create a `JaynesCummings` model from this system's typical parameters.
    pub fn as_jaynes_cummings(&self) -> JaynesCummings {
        let p = self.typical_parameters();
        JaynesCummings {
            omega_cavity: p.omega_0,
            omega_atom: p.omega_0, // resonant by default
            coupling_g: p.g_coupling,
            kappa: p.kappa,
            gamma: p.gamma,
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// PhC nanocavity JC model: g/2π ~ 40 GHz, κ ~ ω/Q, γ/2π ~ 1 GHz
    fn phc_jc() -> JaynesCummings {
        let two_pi = 2.0 * PI;
        let omega_c = two_pi * C_LIGHT / 920e-9;
        let q = 1e4_f64;
        JaynesCummings {
            omega_cavity: omega_c,
            omega_atom: omega_c, // resonant
            coupling_g: two_pi * 40e9,
            kappa: omega_c / q,
            gamma: two_pi * 1e9,
        }
    }

    #[test]
    fn jaynes_cummings_strong_coupling() {
        // PhC nanocavity: g/2π ~ 40 GHz > max(κ,γ)/2
        let jc = phc_jc();
        assert!(jc.is_strong_coupling(), "PhC nanocavity should be in strong coupling");
        assert!(
            jc.cooperativity() > 10.0,
            "Cooperativity should be >10, got {}",
            jc.cooperativity()
        );
    }

    #[test]
    fn vacuum_rabi_splitting_is_2g() {
        let jc = phc_jc();
        let vrs = jc.vacuum_rabi_splitting();
        assert!((vrs - 2.0 * jc.coupling_g).abs() < 1e-6);
    }

    #[test]
    fn dressed_state_resonant_n0_splitting() {
        let jc = phc_jc();
        let (e_minus, e_plus) = jc.dressed_state_energies(0);
        // At resonance (Δ=0): splitting = 2g
        let splitting = e_plus - e_minus;
        assert!(
            (splitting - 2.0 * jc.coupling_g).abs() / jc.coupling_g < 1e-10,
            "n=0 splitting should be 2g, got {}",
            splitting
        );
    }

    #[test]
    fn cooperativity_superconducting_circuit() {
        // SC circuit typically has very high cooperativity
        let jc = CavityType::SuperconductingCircuit.as_jaynes_cummings();
        let c = jc.cooperativity();
        assert!(c > 100.0, "SC circuit cooperativity should be >100, got {}", c);
    }

    #[test]
    fn photon_blockade_strong_coupling() {
        let jc = phc_jc();
        // For PhC: g/2π ~ 40 GHz >> κ
        assert!(jc.has_photon_blockade(), "PhC should exhibit photon blockade");
    }

    #[test]
    fn vacuum_rabi_oscillation_resonant() {
        let two_pi = 2.0 * PI;
        let omega_c = 2.0e15;
        let jc = JaynesCummings {
            omega_cavity: omega_c,
            omega_atom: omega_c, // resonant
            coupling_g: two_pi * 1e9, // 1 GHz coupling
            kappa: two_pi * 1e6,
            gamma: two_pi * 1e6,
        };
        // At t=0: P_e = cos²(0) = 1
        let p0 = jc.vacuum_rabi_oscillation(0.0);
        assert!((p0 - 1.0).abs() < 1e-10, "P_e(0) should be 1, got {}", p0);
        // At t = π/(2g): P_e = cos²(π/2) = 0
        let t_pi_half = PI / (2.0 * two_pi * 1e9);
        let p_half = jc.vacuum_rabi_oscillation(t_pi_half);
        assert!(p_half.abs() < 1e-6, "P_e(π/2g) should be ~0, got {}", p_half);
    }

    #[test]
    fn energy_ladder_anharmonicity() {
        let jc = phc_jc();
        let ladder = jc.energy_ladder(3);
        assert_eq!(ladder.len(), 4); // n = 0,1,2,3
        // For each manifold, E+ > E-
        for (n, (e_minus, e_plus)) in ladder.iter().enumerate() {
            assert!(e_plus > e_minus, "E+ <= E- at n={}", n);
        }
        // Anharmonicity: splitting for n=1 is sqrt(2) × splitting for n=0
        let split_0 = ladder[0].1 - ladder[0].0;
        let split_1 = ladder[1].1 - ladder[1].0;
        let ratio = split_1 / split_0;
        assert!(
            (ratio - 2.0_f64.sqrt()).abs() < 1e-6,
            "JC anharmonicity ratio should be √2 at resonance, got {}",
            ratio
        );
    }

    #[test]
    fn transmission_spectrum_two_peaks() {
        let jc = phc_jc();
        let omega_c = jc.omega_cavity;
        let span = 10.0 * jc.vacuum_rabi_splitting();
        let spectrum = jc.transmission_spectrum((omega_c - span, omega_c + span), 2000);
        // Find peaks
        let mut peaks = 0usize;
        for i in 1..spectrum.len() - 1 {
            let (_, t_prev) = spectrum[i - 1];
            let (_, t_curr) = spectrum[i];
            let (_, t_next) = spectrum[i + 1];
            if t_curr > t_prev && t_curr > t_next && t_curr > 0.3 {
                peaks += 1;
            }
        }
        assert_eq!(peaks, 2, "Expected 2 transmission peaks (vacuum Rabi doublet), found {}", peaks);
    }
}
