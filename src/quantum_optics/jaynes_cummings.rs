//! Jaynes-Cummings model: two-level atom coupled to a single electromagnetic cavity mode.
//!
//! Hamiltonian (rotating wave approximation):
//!   H = ℏωc a†a + (ℏωa/2) σz + ℏg (a†σ⁻ + a σ⁺)
//!
//! where:
//!  - ωc  = cavity angular frequency
//!  - ωa  = atomic transition angular frequency
//!  - g   = vacuum Rabi coupling strength
//!  - a, a† = photon annihilation / creation operators
//!  - σ±, σz = Pauli spin-½ operators for the two-level atom

use num_complex::Complex64;

use crate::error::{OxiPhotonError, Result};

/// Reduced Planck constant (J·s) — available for callers that compute energies in joules.
#[allow(dead_code)]
const HBAR: f64 = 1.054_571_817e-34;

// ─── Main structure ──────────────────────────────────────────────────────────

/// Jaynes-Cummings model parameters.
///
/// The Hilbert space is spanned by product states |atom, photon⟩ where atom ∈ {g, e} and
/// photon ∈ {0, 1, …, n_photons_max}.  The total dimension is 2·(n_photons_max + 1).
#[derive(Debug, Clone)]
pub struct JaynesCummings {
    /// Cavity angular frequency ωc (rad s⁻¹)
    pub omega_cavity: f64,
    /// Atomic transition angular frequency ωa (rad s⁻¹)
    pub omega_atom: f64,
    /// Vacuum Rabi coupling g (rad s⁻¹)
    pub coupling_g: f64,
    /// Cavity energy decay rate κ (s⁻¹)
    pub kappa: f64,
    /// Atomic spontaneous emission rate γ (s⁻¹)
    pub gamma: f64,
    /// Maximum Fock-state index used for matrix representation
    pub n_photons_max: usize,
}

impl JaynesCummings {
    /// Create a new Jaynes-Cummings model.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] if any frequency / rate is non-positive or if
    /// `n_max` is zero.
    pub fn new(
        omega_cavity: f64,
        omega_atom: f64,
        coupling_g: f64,
        kappa: f64,
        gamma: f64,
        n_max: usize,
    ) -> Result<Self> {
        if omega_cavity <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "omega_cavity must be positive, got {omega_cavity}"
            )));
        }
        if omega_atom <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "omega_atom must be positive, got {omega_atom}"
            )));
        }
        if coupling_g <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "coupling_g must be positive, got {coupling_g}"
            )));
        }
        if kappa <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "kappa must be positive, got {kappa}"
            )));
        }
        if gamma <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "gamma must be positive, got {gamma}"
            )));
        }
        if n_max == 0 {
            return Err(OxiPhotonError::NumericalError(
                "n_photons_max must be at least 1".to_string(),
            ));
        }
        Ok(Self {
            omega_cavity,
            omega_atom,
            coupling_g,
            kappa,
            gamma,
            n_photons_max: n_max,
        })
    }

    // ── Spectroscopic quantities ─────────────────────────────────────────────

    /// Atom-cavity detuning Δ = ωa − ωc (rad s⁻¹).
    ///
    /// Δ = 0 means the atom is resonant with the cavity.
    #[inline]
    pub fn detuning(&self) -> f64 {
        self.omega_atom - self.omega_cavity
    }

    /// On-resonance vacuum Rabi splitting = 2g (rad s⁻¹).
    ///
    /// This is the frequency separation of the two dressed states |±, 0⟩ at Δ = 0.
    #[inline]
    pub fn vacuum_rabi_splitting(&self) -> f64 {
        2.0 * self.coupling_g
    }

    /// Photon-number-dependent Rabi frequency Ω_n = 2g√(n+1) (rad s⁻¹).
    #[inline]
    pub fn rabi_frequency(&self, n_photons: usize) -> f64 {
        2.0 * self.coupling_g * ((n_photons + 1) as f64).sqrt()
    }

    /// Eigenenergies of the dressed states |±, n⟩ divided by ℏ (angular frequency units).
    ///
    /// E±,n / ℏ = (n + ½) ωc ± ½ √(Δ² + 4g²(n+1))
    ///
    /// Returns `(E_minus / ℏ, E_plus / ℏ)`.
    pub fn dressed_state_energies(&self, n: usize) -> (f64, f64) {
        let delta = self.detuning();
        let generalized =
            (delta * delta + 4.0 * self.coupling_g * self.coupling_g * (n + 1) as f64).sqrt();
        let center = (n as f64 + 0.5) * self.omega_cavity;
        (center - 0.5 * generalized, center + 0.5 * generalized)
    }

    // ── Hamiltonian matrix ────────────────────────────────────────────────────

    /// Build the Jaynes-Cummings Hamiltonian matrix (divided by ℏ) in the basis
    ///
    ///   |g, 0⟩, |e, 0⟩, |g, 1⟩, |e, 1⟩, …, |g, N⟩, |e, N⟩
    ///
    /// where N = `n_photons_max`.  The full dimension is `dim = 2*(N+1)`.
    ///
    /// Non-zero block structure (for each excitation-number manifold {n, n+1}):
    ///   H_diag(|e,n⟩) = n·ωc + ωa/2
    ///   H_diag(|g,n⟩) = n·ωc − ωa/2
    ///   H_off(|e,n⟩↔|g,n+1⟩) = g·√(n+1)
    pub fn hamiltonian_matrix(&self) -> Vec<Vec<Complex64>> {
        let n = self.n_photons_max;
        let dim = 2 * (n + 1);
        let mut h = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];

        for fock in 0..=n {
            // Column / row indices: ground = 2*fock, excited = 2*fock + 1
            let ig = 2 * fock;
            let ie = 2 * fock + 1;

            // Diagonal: cavity energy + atom energy (σz/2 terms)
            let e_cav = fock as f64 * self.omega_cavity;
            h[ig][ig] = Complex64::new(e_cav - self.omega_atom * 0.5, 0.0);
            h[ie][ie] = Complex64::new(e_cav + self.omega_atom * 0.5, 0.0);

            // Off-diagonal coupling: |e, fock⟩ ↔ |g, fock+1⟩  (if fock+1 ≤ n)
            if fock < n {
                let ig_next = 2 * (fock + 1); // |g, fock+1⟩
                let coupling = self.coupling_g * ((fock + 1) as f64).sqrt();
                h[ie][ig_next] = Complex64::new(coupling, 0.0);
                h[ig_next][ie] = Complex64::new(coupling, 0.0);
            }
        }
        h
    }

    // ── Dynamics ─────────────────────────────────────────────────────────────

    /// Compute population inversion W(t) = ⟨σz⟩(t) for an initial state |g, n⟩.
    ///
    /// In the resonant case (Δ = 0) and ignoring dissipation:
    ///   W(t) = −cos(Ω_n · t)
    ///
    /// where Ω_n = 2g√(n+1).  The atom starts in the ground state (W = −1), so:
    ///   W(0) = −1,   W(π/Ω_n) = +1.
    ///
    /// For the off-resonant general case the formula generalises to
    ///   W(t) = −\[Δ² + 4g²(n+1)·cos(Ω̃_n t)\] / Ω̃_n²
    /// with Ω̃_n = √(Δ² + 4g²(n+1)).
    ///
    /// Returns a `Vec` of `(time_ns, W)` pairs.
    pub fn rabi_oscillation(
        &self,
        n_photons: usize,
        t_max_ns: f64,
        n_steps: usize,
    ) -> Result<Vec<(f64, f64)>> {
        if n_steps < 2 {
            return Err(OxiPhotonError::NumericalError(
                "n_steps must be ≥ 2".to_string(),
            ));
        }
        let delta = self.detuning();
        let omega_tilde_sq =
            delta * delta + 4.0 * self.coupling_g * self.coupling_g * (n_photons + 1) as f64;
        let omega_tilde = omega_tilde_sq.sqrt();

        let t_max_s = t_max_ns * 1e-9;
        let mut result = Vec::with_capacity(n_steps);

        for i in 0..n_steps {
            let t = t_max_s * (i as f64) / ((n_steps - 1) as f64);
            let w = if omega_tilde_sq < f64::EPSILON {
                // No coupling — atom stays in ground state
                -1.0
            } else {
                -(delta * delta
                    + 4.0
                        * self.coupling_g
                        * self.coupling_g
                        * (n_photons + 1) as f64
                        * (omega_tilde * t).cos())
                    / omega_tilde_sq
            };
            result.push((t * 1e9, w));
        }
        Ok(result)
    }

    // ── Collapse and revival ─────────────────────────────────────────────────

    /// Collapse time for a coherent-state field with mean photon number n̄.
    ///
    /// t_collapse ≈ √n̄ / g  (in seconds)
    ///
    /// This is the time over which the Rabi oscillations dephase due to the
    /// spread in photon numbers of a coherent state.
    pub fn collapse_time(&self, mean_photon_number: f64) -> f64 {
        mean_photon_number.sqrt() / self.coupling_g
    }

    /// Revival time for a coherent-state field with mean photon number n̄.
    ///
    /// t_revival ≈ 2π√n̄ / g  (in seconds)
    ///
    /// This is the time at which all Rabi oscillations re-phase coherently.
    pub fn revival_time(&self, mean_photon_number: f64) -> f64 {
        2.0 * std::f64::consts::PI * mean_photon_number.sqrt() / self.coupling_g
    }

    // ── Strong coupling ──────────────────────────────────────────────────────

    /// Strong-coupling condition: g > max(κ, γ) / 2.
    pub fn is_strong_coupling(&self) -> bool {
        let max_rate = self.kappa.max(self.gamma);
        self.coupling_g > max_rate / 2.0
    }

    /// Purcell factor F_P = 4g² / (κγ).
    ///
    /// When F_P ≫ 1 the atom emission is strongly enhanced by the cavity.
    pub fn purcell_factor(&self) -> f64 {
        4.0 * self.coupling_g * self.coupling_g / (self.kappa * self.gamma)
    }

    /// Cooperativity C = g² / (κγ).
    ///
    /// C > 1 is required for efficient quantum information applications.
    pub fn cooperativity(&self) -> f64 {
        self.coupling_g * self.coupling_g / (self.kappa * self.gamma)
    }

    // ── Spectra ──────────────────────────────────────────────────────────────

    /// Cavity transmission spectrum T(ω) including vacuum Rabi splitting.
    ///
    /// In the strong-coupling regime the spectrum shows two Lorentzian peaks
    /// (dressed-state resonances) separated by 2g.  The transfer function is
    ///
    ///   T(ω) ∝ |χ_+(ω)|² + |χ_-(ω)|²
    ///
    /// where
    ///   χ±(ω) = κ/2 / \[i(ω − ω± ) + κ/2 + γ/4\]
    ///
    /// and ω± are the dressed-state frequencies for n=0.
    ///
    /// Returns `Vec<(omega, T)>` with T normalised to its maximum.
    pub fn transmission_spectrum(
        &self,
        omega_range: (f64, f64),
        n_pts: usize,
    ) -> Result<Vec<(f64, f64)>> {
        if n_pts < 2 {
            return Err(OxiPhotonError::NumericalError(
                "n_pts must be ≥ 2".to_string(),
            ));
        }
        let (e_minus, e_plus) = self.dressed_state_energies(0);
        let half_kappa = self.kappa / 2.0;
        let half_gamma_4 = self.gamma / 4.0;
        let denom_extra = half_kappa + half_gamma_4;

        let mut result: Vec<(f64, f64)> = Vec::with_capacity(n_pts);
        let mut t_max = 0.0_f64;

        for i in 0..n_pts {
            let omega =
                omega_range.0 + (omega_range.1 - omega_range.0) * (i as f64) / ((n_pts - 1) as f64);

            // Contribution from each dressed state
            let lorentz = |omega_res: f64| -> f64 {
                let denum = (omega - omega_res).powi(2) + denom_extra.powi(2);
                (half_kappa * half_kappa) / denum
            };

            let t = lorentz(e_minus) + lorentz(e_plus);
            result.push((omega, t));
            if t > t_max {
                t_max = t;
            }
        }

        // Normalise
        if t_max > 0.0 {
            for entry in &mut result {
                entry.1 /= t_max;
            }
        }
        Ok(result)
    }

    // ── Dispersive regime ────────────────────────────────────────────────────

    /// Dispersive (off-resonant) shift of the cavity frequency per photon:
    ///   χ = g² / Δ   (rad s⁻¹)
    ///
    /// Valid when |Δ| ≫ g.  The cavity frequency becomes ωc ± χ depending on
    /// the qubit state, enabling QND readout.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] if Δ = 0 (resonant case).
    pub fn dispersive_shift(&self) -> Result<f64> {
        let delta = self.detuning();
        if delta.abs() < f64::EPSILON {
            return Err(OxiPhotonError::NumericalError(
                "Dispersive shift is undefined at resonance (Δ = 0)".to_string(),
            ));
        }
        Ok(self.coupling_g * self.coupling_g / delta)
    }

    /// Signal-to-noise ratio for a QND qubit readout through the cavity.
    ///
    /// The SNR for homodyne detection of the dispersive shift over time `T` is
    ///
    ///   SNR = 2|χ| · √(κ_ex · T) / (κ/2)
    ///
    /// where κ_ex is the external (coupling) decay rate and the factor 2|χ|/κ
    /// represents the phase angle per photon in units of the cavity linewidth.
    ///
    /// # Errors
    /// Propagates errors from \[`dispersive_shift`\].
    pub fn qnd_readout_snr(&self, kappa_ex: f64, measurement_time_s: f64) -> Result<f64> {
        let chi = self.dispersive_shift()?;
        // Phase response: 2χ quanta of rotation per photon, bandwidth κ/2
        let phase_per_photon = (2.0 * chi.abs()) / (self.kappa / 2.0);
        let snr = phase_per_photon * (kappa_ex * measurement_time_s).sqrt();
        Ok(snr)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Build a standard test model: resonant, strong coupling.
    fn test_model() -> JaynesCummings {
        // g/(2π) = 10 MHz, κ/(2π) = 1 MHz, γ/(2π) = 0.1 MHz
        let two_pi = 2.0 * std::f64::consts::PI;
        JaynesCummings::new(
            2.0 * two_pi * 1e9, // ωc: 2π × 1 GHz
            2.0 * two_pi * 1e9, // ωa = ωc (resonant)
            two_pi * 10e6,      // g: 2π × 10 MHz
            two_pi * 1e6,       // κ: 2π × 1 MHz
            two_pi * 0.1e6,     // γ: 2π × 0.1 MHz
            20,
        )
        .expect("valid model")
    }

    #[test]
    fn test_vacuum_rabi_splitting() {
        let jc = test_model();
        let two_pi = 2.0 * std::f64::consts::PI;
        // 2g = 2 × 2π × 10 MHz
        assert_relative_eq!(
            jc.vacuum_rabi_splitting(),
            2.0 * two_pi * 10e6,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_rabi_frequency_n_photons() {
        let jc = test_model();
        let two_pi = 2.0 * std::f64::consts::PI;
        // n=0: Ω₀ = 2g√1 = 2g
        assert_relative_eq!(jc.rabi_frequency(0), 2.0 * two_pi * 10e6, epsilon = 1e-6);
        // n=3: Ω₃ = 2g√4 = 4g
        assert_relative_eq!(jc.rabi_frequency(3), 4.0 * two_pi * 10e6, epsilon = 1e-6);
    }

    #[test]
    fn test_strong_coupling_condition() {
        let jc = test_model();
        // g = 2π·10 MHz, max(κ,γ)/2 = 2π·0.5 MHz — strong coupling
        assert!(jc.is_strong_coupling());

        // Build a weak-coupling model
        let two_pi = 2.0 * std::f64::consts::PI;
        let weak = JaynesCummings::new(
            two_pi * 1e9,
            two_pi * 1e9,
            two_pi * 0.1e6, // g = 0.1 MHz
            two_pi * 10e6,  // κ = 10 MHz >> g
            two_pi * 0.1e6,
            5,
        )
        .expect("valid model");
        assert!(!weak.is_strong_coupling());
    }

    #[test]
    fn test_purcell_factor() {
        let jc = test_model();
        // F_P = 4g²/(κγ)
        let two_pi = 2.0 * std::f64::consts::PI;
        let g = two_pi * 10e6;
        let kappa = two_pi * 1e6;
        let gamma = two_pi * 0.1e6;
        let expected = 4.0 * g * g / (kappa * gamma);
        assert_relative_eq!(jc.purcell_factor(), expected, epsilon = 1e-6);
    }

    #[test]
    fn test_rabi_oscillation_cosine() {
        let jc = test_model();
        // For n=0, Ω₀ = 2g.  Starting in |g,1⟩ with the formula W = -cos(Ω₀ t):
        //   W(0) = -1,  W(π/Ω₀) = +1
        let omega0 = jc.rabi_frequency(0);
        let t_pi_ns = (std::f64::consts::PI / omega0) * 1e9;
        let steps = 1000;
        let oscillation = jc.rabi_oscillation(0, t_pi_ns, steps).expect("ok");

        let w_at_0 = oscillation[0].1;
        let w_at_pi = oscillation[steps - 1].1;

        assert_relative_eq!(w_at_0, -1.0, epsilon = 1e-10);
        assert_relative_eq!(w_at_pi, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_dispersive_shift() {
        // Off-resonant model: Δ = ωa − ωc = 2π × 100 MHz
        let two_pi = 2.0 * std::f64::consts::PI;
        let g = two_pi * 10e6;
        let delta = two_pi * 100e6;
        let jc = JaynesCummings::new(
            two_pi * 1e9,
            two_pi * 1e9 + delta,
            g,
            two_pi * 1e6,
            two_pi * 0.1e6,
            10,
        )
        .expect("valid model");

        let chi = jc.dispersive_shift().expect("off-resonant");
        let expected = g * g / delta;
        // Use relative epsilon for large frequency values (~6 MHz scale)
        assert_relative_eq!(chi, expected, max_relative = 1e-10);
    }

    #[test]
    fn test_dressed_state_splitting() {
        let jc = test_model(); // resonant: Δ = 0
        let (e_minus, e_plus) = jc.dressed_state_energies(0);
        // At resonance: E± = ωc/2 ± g
        // splitting = E+ - E- = 2g
        // Use relative epsilon for large angular frequency values (~2π × 10 MHz)
        assert_relative_eq!(e_plus - e_minus, 2.0 * jc.coupling_g, max_relative = 1e-10);
    }

    #[test]
    fn test_hamiltonian_hermitian() {
        let jc = test_model();
        let h = jc.hamiltonian_matrix();
        let dim = h.len();
        for (i, _) in h.iter().enumerate().take(dim) {
            for (j, _) in h.iter().enumerate().take(dim) {
                assert_relative_eq!(h[i][j].re, h[j][i].re, epsilon = 1e-10);
                assert_relative_eq!(h[i][j].im, -h[j][i].im, epsilon = 1e-10);
            }
        }
    }
}
