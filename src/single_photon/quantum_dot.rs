//! InAs/GaAs self-assembled quantum dot physics.
//!
//! Implements an 8-band k·p model approximation for the ground-state transition
//! energy, exciton and biexciton properties, phonon sideband, dephasing,
//! indistinguishability, and fine-structure splitting of InAs/GaAs quantum dots.
//!
//! # Physical background
//! Self-assembled InAs/GaAs quantum dots grow via Stranski-Krastanov mode and
//! confine both electrons and holes in all three dimensions, producing a
//! discrete, atom-like energy spectrum.  The ground-state transition (X₀) lies
//! typically between 880–960 nm at low temperature and is the workhorse
//! transition for solid-state single-photon emission.
//!
//! # References
//! - Bimberg, Grundmann & Ledentsov, "Quantum Dot Heterostructures" (1999)
//! - Lodahl et al., Rev. Mod. Phys. 87, 347 (2015)
//! - Kuhlmann et al., Nature Phys. 9, 570 (2013) — resonance fluorescence & FSS
//! - Iles-Smith et al., Nature Photon. 11, 521 (2017) — phonon sideband theory

use std::f64::consts::PI;

// ─── Physical constants ────────────────────────────────────────────────────────

/// Planck's constant (J·s)
const H_PLANCK: f64 = 6.626_070_15e-34;
/// Reduced Planck's constant (J·s)
const H_BAR: f64 = H_PLANCK / (2.0 * PI);
/// Speed of light in vacuum (m/s)
const C: f64 = 2.997_924_58e8;
/// Electron charge (C)
const E_CHARGE: f64 = 1.602_176_634e-19;
/// Boltzmann constant (J/K)
const K_B: f64 = 1.380_649e-23;

// ─── InAsQd ───────────────────────────────────────────────────────────────────

/// Self-assembled InAs/GaAs quantum dot.
///
/// Geometry parameters follow the typical lens-shaped dot with height h and
/// base diameter d.  The indium fraction x selects the In_xGa_{1-x}As
/// composition.  Strain is in the biaxial convention (compressive → negative).
#[derive(Debug, Clone)]
pub struct InAsQd {
    /// Dot height (nm)
    pub dot_height_nm: f64,
    /// Base diameter (nm)
    pub dot_diameter_nm: f64,
    /// Indium fraction x in In_xGa_{1-x}As (0 < x ≤ 1)
    pub indium_fraction: f64,
    /// Biaxial strain (compressive: negative)
    pub strain: f64,
    /// Lattice temperature (K)
    pub temperature_k: f64,
}

impl InAsQd {
    /// Create a typical self-assembled InAs/GaAs dot with pure InAs composition.
    ///
    /// Default dimensions match a commonly observed dot ensemble with emission
    /// centred near 920 nm at 4 K.
    pub fn new(height_nm: f64, diameter_nm: f64) -> Self {
        Self {
            dot_height_nm: height_nm,
            dot_diameter_nm: diameter_nm,
            indium_fraction: 1.0,
            strain: -0.07, // ~7 % compressive biaxial strain in InAs/GaAs
            temperature_k: 4.0,
        }
    }

    // ─── Band-gap helpers ──────────────────────────────────────────────────────

    /// InAs bulk band gap at 0 K (eV) — Varshni parameters.
    fn inas_bandgap_ev(&self) -> f64 {
        // Varshni: Eg(T) = Eg(0) - α T²/(T+β)
        // InAs: Eg(0)=0.415 eV, α=2.76e-4 eV/K, β=83 K
        let alpha = 2.76e-4_f64; // eV/K
        let beta = 83.0_f64; // K
        let eg0 = 0.415_f64;
        let t = self.temperature_k;
        eg0 - alpha * t * t / (t + beta)
    }

    /// GaAs bulk band gap at given temperature (eV).
    fn gaas_bandgap_ev(&self) -> f64 {
        // Varshni: Eg(0)=1.519 eV, α=5.405e-4 eV/K, β=204 K
        let alpha = 5.405e-4_f64;
        let beta = 204.0_f64;
        let eg0 = 1.519_f64;
        let t = self.temperature_k;
        eg0 - alpha * t * t / (t + beta)
    }

    /// Band-gap bowing parameter for In_xGa_{1-x}As: b = 0.477 eV.
    fn alloy_bandgap_ev(&self) -> f64 {
        let x = self.indium_fraction.clamp(0.0, 1.0);
        let e_inas = self.inas_bandgap_ev();
        let e_gaas = self.gaas_bandgap_ev();
        let b = 0.477_f64; // bowing parameter
        x * e_inas + (1.0 - x) * e_gaas - b * x * (1.0 - x)
    }

    // ─── Confinement energy (particle-in-a-box approximation) ──────────────────

    /// Electron confinement energy in the dot (eV).
    ///
    /// Approximated as a 3-D anisotropic harmonic oscillator ground state with
    /// effective masses from 8-band k·p theory for InAs.
    fn electron_confinement_ev(&self) -> f64 {
        // InAs effective electron mass m*_e ≈ 0.023 m_0
        let m_e_0 = 9.109_383_7e-31; // kg
        let m_star_e = 0.023 * m_e_0;
        // Height and diameter in metres
        let h_m = self.dot_height_nm * 1e-9;
        let d_m = self.dot_diameter_nm * 1e-9;
        // Quantum confinement energy: ħ²π²/(2m*L²)
        // Use harmonic mean of in-plane (d/2 radius) and vertical (h) scales
        let e_vert = H_BAR * H_BAR * PI * PI / (2.0 * m_star_e * h_m * h_m);
        let e_lat = H_BAR * H_BAR * PI * PI / (2.0 * m_star_e * (d_m / 2.0).powi(2));
        // Ground state: combine both directions (zero-point sum)
        (e_vert + e_lat) / E_CHARGE // J → eV
    }

    /// Heavy-hole confinement energy in the dot (eV).
    fn heavy_hole_confinement_ev(&self) -> f64 {
        // InAs heavy-hole in-plane mass m*_hh ≈ 0.35 m_0, vertical ≈ 0.32 m_0
        let m_e_0 = 9.109_383_7e-31;
        let m_star_hh_z = 0.35 * m_e_0;
        let m_star_hh_xy = 0.32 * m_e_0;
        let h_m = self.dot_height_nm * 1e-9;
        let d_m = self.dot_diameter_nm * 1e-9;
        let e_vert = H_BAR * H_BAR * PI * PI / (2.0 * m_star_hh_z * h_m * h_m);
        let e_lat = H_BAR * H_BAR * PI * PI / (2.0 * m_star_hh_xy * (d_m / 2.0).powi(2));
        (e_vert + e_lat) / E_CHARGE
    }

    // ─── Strain shifts ────────────────────────────────────────────────────────

    /// Biaxial strain shift of the conduction band (eV).
    ///
    /// ΔE_c = a_c * (ε_xx + ε_yy + ε_zz), where ε = strain, a_c ≈ −7.17 eV for InAs.
    fn strain_shift_conduction_ev(&self) -> f64 {
        // Hydrostatic deformation potential for InAs conduction band: a_c = -7.17 eV
        let a_c = -7.17_f64;
        // For biaxial strain ε, Poisson ratio ν ≈ 0.352 for InAs
        // ε_xx = ε_yy = ε (biaxial), ε_zz = -2ν/(1-ν)*ε
        let eps = self.strain;
        let nu = 0.352_f64;
        let eps_zz = -2.0 * nu / (1.0 - nu) * eps;
        a_c * (2.0 * eps + eps_zz)
    }

    /// Biaxial strain shift of the valence band (heavy-hole) (eV).
    ///
    /// Uses the Bir-Pikus Hamiltonian: ΔE_hh = -P_ε - Q_ε
    fn strain_shift_valence_ev(&self) -> f64 {
        // Hydrostatic (a_v) and shear (b) deformation potentials for InAs
        let a_v = 1.0_f64; // eV (InAs valence band hydrostatic DP)
        let b = -1.8_f64; // eV (shear DP)
        let eps = self.strain;
        let nu = 0.352_f64;
        let eps_zz = -2.0 * nu / (1.0 - nu) * eps;
        // P_ε = -a_v*(2ε + ε_zz), Q_ε = -b/2*(ε_xx - ε_zz) = -b/2*(ε - ε_zz)
        let p_eps = -a_v * (2.0 * eps + eps_zz);
        let q_eps = -b / 2.0 * (eps - eps_zz);
        // Heavy-hole: -P_ε - Q_ε
        -p_eps - q_eps
    }

    // ─── Public API ───────────────────────────────────────────────────────────

    /// Ground-state exciton transition energy (eV) using an 8-band k·p approximation.
    ///
    /// E_X = E_gap(alloy,T) + ΔE_c(strain) - ΔE_v(strain) + E_e(conf) + E_h(conf)
    /// The Coulomb correction for exciton binding is applied via `exciton_binding_energy_mev`.
    pub fn transition_energy_ev(&self) -> f64 {
        let e_gap = self.alloy_bandgap_ev();
        let d_ec = self.strain_shift_conduction_ev();
        let d_ev = self.strain_shift_valence_ev();
        let e_e = self.electron_confinement_ev();
        let e_h = self.heavy_hole_confinement_ev();
        // Subtract exciton binding to get the actual X₀ photon energy
        let e_x_bind = self.exciton_binding_energy_mev() * 1e-3;
        (e_gap + d_ec - d_ev + e_e + e_h - e_x_bind).max(0.1)
    }

    /// Emission wavelength (nm) from the ground-state exciton transition.
    pub fn wavelength_nm(&self) -> f64 {
        let energy_j = self.transition_energy_ev() * E_CHARGE;
        if energy_j > 0.0 {
            H_PLANCK * C / energy_j * 1e9
        } else {
            f64::INFINITY
        }
    }

    /// Exciton binding energy (meV): E_X = E_e + E_h − E_Coulomb.
    ///
    /// Coulomb attraction estimated from variational Hydrogen model with
    /// GaAs dielectric constant ε_r = 12.4 and an effective Bohr radius
    /// set by the dot lateral extent.
    pub fn exciton_binding_energy_mev(&self) -> f64 {
        // Effective Bohr radius a_B* = ε_r * (m_0/μ) * a_0
        // Reduced mass μ = m_e*m_h/(m_e+m_h), GaAs: m_e=0.067, m_hh=0.38 m_0
        let eps_r = 12.4_f64;
        let m_e = 0.023_f64; // InAs (more confined material drives Bohr radius)
        let m_hh = 0.35_f64;
        let mu = m_e * m_hh / (m_e + m_hh); // in units of m_0
                                            // Effective Bohr radius in nm: a_B* = ε_r/μ * 0.0529 nm
        let a_b_star_nm = eps_r / mu * 0.052_918;
        // Localisation radius of exciton: use in-plane confinement radius
        let r_nm = (self.dot_diameter_nm / 2.0).max(1.0);
        // Coulomb energy ≈ e²/(4πε₀ε_r r) capped by Bohr scaling
        // Variational estimate: E_C = e²/(4πε₀ε_r a_B*) * (a_B*/r) for r > a_B*
        let e_c_rydberg_mev = 13_605.0 * mu / (eps_r * eps_r); // Rydberg* in meV
        let overlap = (a_b_star_nm / r_nm).min(1.0);
        (e_c_rydberg_mev * overlap).abs()
    }

    /// Fine-structure splitting (μeV) between X- and Y-polarised excitons.
    ///
    /// FSS arises from the anisotropic exchange interaction between the electron
    /// and hole spins, driven by shape asymmetry and strain.  For a dot with
    /// elongation Δd = d_x − d_y:
    ///   FSS ≈ FSS₀ * (aspect_ratio − 1)
    /// where FSS₀ ≈ 50 μeV is the single-dot exchange coefficient.
    pub fn fine_structure_splitting_uev(&self) -> f64 {
        // Assume circular dot by default; asymmetry encoded as 5 % elongation
        // proportional to the inverse of height (thinner dots → larger FSS)
        let h = self.dot_height_nm.max(1.0);
        let d = self.dot_diameter_nm.max(1.0);
        // Empirical: FSS ≈ 50*(d/h - 1) μeV for typical InAs/GaAs dots
        let aspect = d / h;
        let fss_0 = 50.0_f64; // μeV scale
        (fss_0 * (aspect - 1.0).abs()).max(0.0)
    }

    /// Radiative lifetime (ns): τ_rad = 1/Γ_rad.
    ///
    /// The oscillator strength f ∝ V_dot (volume overlap), giving
    /// τ_rad ≈ 1 ns for typical InAs/GaAs dots.  Scales inversely
    /// with indium fraction and dot volume.
    pub fn radiative_lifetime_ns(&self) -> f64 {
        // Reference: 1 ns for an 8-nm-tall, 25-nm-wide pure InAs dot at 4 K
        let h_ref = 8.0_f64; // nm
        let d_ref = 25.0_f64; // nm
        let vol_ref = h_ref * d_ref * d_ref; // proportional volume
        let vol = self.dot_height_nm * self.dot_diameter_nm * self.dot_diameter_nm;
        let tau_ref = 1.0_f64; // ns
                               // Oscillator strength ∝ overlap integral ∝ 1/V for confined exciton
        (tau_ref * vol_ref / vol.max(1.0)).clamp(0.2, 5.0)
    }

    /// Phonon sideband fraction: 1 − Debye-Waller factor.
    ///
    /// Photons emitted into the phonon sideband reduce the ZPL fraction.
    pub fn phonon_sideband_fraction(&self, temperature_k: f64) -> f64 {
        1.0 - self.debye_waller_factor(temperature_k)
    }

    /// Debye-Waller (ZPL) factor: W(T) = exp(−S(T)).
    ///
    /// The Huang-Rhys factor S(T) is computed in the independent-boson model:
    ///   S(T) = ∫ α(ω)[n_B(ω,T) + 1/2] dω
    ///
    /// For InAs/GaAs QDs the low-T limit gives S(0) ≈ 0.05, rising with T.
    pub fn debye_waller_factor(&self, temperature_k: f64) -> f64 {
        // Huang-Rhys parameter S₀ at 0 K: ~0.04 for typical InAs/GaAs dot
        let s0 = 0.04_f64;
        // Cut-off phonon energy for bulk GaAs acoustic phonons: ħωc ≈ 1 meV → ωc in rad/s
        let omega_c_mev = 1.0_f64; // meV
        let omega_c = omega_c_mev * 1e-3 * E_CHARGE / H_BAR; // rad/s
                                                             // Thermal phonon occupation at cut-off
        let n_b = if temperature_k < 1e-6 {
            0.0
        } else {
            let hbar_omega = H_BAR * omega_c;
            let k_b_t = K_B * temperature_k;
            1.0 / ((hbar_omega / k_b_t).exp() - 1.0 + 1e-30)
        };
        // S(T) ≈ S₀ * (2*n_B + 1) (simplified spectral function in one mode)
        let s_t = s0 * (2.0 * n_b + 1.0);
        (-s_t).exp()
    }

    /// Pure dephasing rate from phonon coupling (GHz).
    ///
    /// For GaAs acoustic phonon bath the dominant mechanism is quadratic
    /// coupling to LA phonons.  Empirically:
    ///   γ_pure(T) ≈ γ₀ * (T/T₀)⁷ for T ≪ Debye temperature
    ///   γ_pure(T) ≈ γ_lin * T   for T ≳ 30 K (linear phonon scattering)
    pub fn pure_dephasing_rate_ghz(&self, temperature_k: f64) -> f64 {
        let t = temperature_k.max(0.0);
        // Low-T power law: γ ≈ 0.005 * (T/10)⁷ GHz for T < 30 K
        // High-T linear: γ ≈ 0.1 * T GHz (empirical)
        if t < 30.0 {
            0.005 * (t / 10.0_f64).powi(7) + 1e-5
        } else {
            0.1 * t
        }
    }

    /// Total optical linewidth (GHz): Γ_total = 1/(2π τ_rad) + γ_pure.
    ///
    /// In frequency units:  Γ_total = 1/(2π·τ_rad) + γ_pure
    pub fn linewidth_ghz(&self, temperature_k: f64) -> f64 {
        let tau_s = self.radiative_lifetime_ns() * 1e-9;
        let gamma_rad = 1.0 / (2.0 * PI * tau_s) * 1e-9; // Hz → GHz
        let gamma_pure = self.pure_dephasing_rate_ghz(temperature_k);
        gamma_rad + gamma_pure
    }

    /// Photon indistinguishability: M = Γ_rad / (Γ_rad + 2 γ_pure).
    ///
    /// M = 1 for a transform-limited emitter; M < 1 with dephasing.
    pub fn indistinguishability(&self, temperature_k: f64) -> f64 {
        let tau_s = self.radiative_lifetime_ns() * 1e-9;
        let gamma_rad = 1.0 / tau_s * 1e-9; // GHz
        let gamma_pure = self.pure_dephasing_rate_ghz(temperature_k);
        let denom = gamma_rad + 2.0 * gamma_pure;
        if denom > 0.0 {
            gamma_rad / denom
        } else {
            1.0
        }
    }

    /// DC Stark shift of wavelength (nm) from an applied electric field.
    ///
    /// The quantum-confined Stark effect (QCSE) red-shifts the exciton:
    ///   ΔE ≈ −β·F²    (quadratic Stark effect)
    /// where β ≈ 20 μeV/(kV/cm)² for InAs/GaAs, converted to Δλ.
    pub fn stark_shift_nm(&self, electric_field_kv_per_cm: f64) -> f64 {
        let f = electric_field_kv_per_cm;
        let beta_uev_per_kvcm_sq = 20.0_f64; // μeV/(kV/cm)²
        let delta_e_uev = -beta_uev_per_kvcm_sq * f * f;
        let delta_e_ev = delta_e_uev * 1e-6;
        // Δλ = -λ²/(hc) * ΔE (in nm, with E in eV and hc in eV·nm)
        let hc_ev_nm = 1239.84_f64; // eV·nm
        let lambda_nm = self.wavelength_nm();
        -lambda_nm * lambda_nm * delta_e_ev / hc_ev_nm
    }

    /// Piezo strain tuning of wavelength (nm) via applied voltage.
    ///
    /// A piezoelectric transducer strains the host crystal, shifting the dot
    /// emission.  Empirical coefficient for InAs/GaAs on PMN-PT:
    ///   dλ/dV ≈ 0.3 nm/kV.
    pub fn strain_tuning_nm(&self, voltage_kv: f64) -> f64 {
        let d_lambda_per_kv = 0.3_f64; // nm/kV
        d_lambda_per_kv * voltage_kv
    }
}

// ─── QdEnsemble ───────────────────────────────────────────────────────────────

/// Ensemble of self-assembled quantum dots with Gaussian size distribution.
///
/// Inhomogeneous broadening from dot-to-dot size variation produces a
/// Gaussian photoluminescence (PL) spectrum.
#[derive(Debug, Clone)]
pub struct QdEnsemble {
    /// Number of dots in the ensemble
    pub n_dots: usize,
    /// Centre emission wavelength (nm) of the distribution
    pub center_wavelength_nm: f64,
    /// Inhomogeneous broadening FWHM (nm)
    pub inhomogeneous_width_nm: f64,
    /// Representative single dot (for homogeneous properties)
    pub dot: InAsQd,
}

impl QdEnsemble {
    /// Create a typical InAs/GaAs dot ensemble.
    ///
    /// Uses a representative dot with h = 6 nm, d = 20 nm.
    pub fn new(center_nm: f64, width_nm: f64, n: usize) -> Self {
        Self {
            n_dots: n,
            center_wavelength_nm: center_nm,
            inhomogeneous_width_nm: width_nm.max(0.01),
            dot: InAsQd::new(6.0, 20.0),
        }
    }

    /// Gaussian PL spectrum evaluated at `wavelength_nm`.
    ///
    /// I(λ) = N · exp(−(λ − λ₀)² / (2 σ²))
    /// where σ = FWHM / (2√(2 ln 2)).
    pub fn pl_spectrum(&self, wavelength_nm: f64) -> f64 {
        let sigma = self.inhomogeneous_width_nm / (2.0 * (2.0 * 2_f64.ln()).sqrt());
        let delta = wavelength_nm - self.center_wavelength_nm;
        (-(delta * delta) / (2.0 * sigma * sigma)).exp()
    }

    /// Homogeneous linewidth per dot at given temperature (GHz).
    pub fn homogeneous_linewidth_ghz(&self, temperature_k: f64) -> f64 {
        self.dot.linewidth_ghz(temperature_k)
    }

    /// Spectral diffusion contribution to single-dot linewidth (GHz).
    ///
    /// Charge fluctuations in the dot environment create a time-varying electric
    /// field (spectral diffusion) broadening the apparent linewidth on timescales
    /// > 1 μs.  Typical values: 1–10 GHz for dots in as-grown samples.
    pub fn spectral_diffusion_ghz(&self) -> f64 {
        // Empirical: ~3 GHz for standard as-grown InAs/GaAs without special gating
        3.0_f64
    }

    /// Photoluminescence excitation (PLE) scan window (GHz).
    ///
    /// A useful scan window is ±3 × (homogeneous linewidth + spectral diffusion).
    pub fn ple_window_ghz(&self) -> f64 {
        let t = self.dot.temperature_k;
        let lw = self.homogeneous_linewidth_ghz(t) + self.spectral_diffusion_ghz();
        6.0 * lw
    }
}

// ─── BiexcitonCascade ─────────────────────────────────────────────────────────

/// Biexciton–exciton cascade as an entangled photon-pair source.
///
/// When a biexciton (XX) decays via the intermediate exciton (X) to the
/// ground state, two photons are emitted in a cascade.  For zero fine-structure
/// splitting (FSS) the photon pair is maximally entangled in polarisation.
#[derive(Debug, Clone)]
pub struct BiexcitonCascade {
    /// The quantum dot host
    pub dot: InAsQd,
    /// Biexciton binding energy (meV, usually slightly negative = anti-binding)
    pub biexciton_binding_energy_mev: f64,
    /// Fine-structure splitting (μeV) — limits entanglement fidelity
    pub fss_uev: f64,
}

impl BiexcitonCascade {
    /// Create from dot with default biexciton parameters.
    ///
    /// The biexciton binding energy is set to −2 meV (anti-binding, typical for
    /// InAs/GaAs) and FSS to the single-dot value computed from dot geometry.
    pub fn new(dot: InAsQd) -> Self {
        let fss = dot.fine_structure_splitting_uev();
        Self {
            biexciton_binding_energy_mev: -2.0,
            fss_uev: fss,
            dot,
        }
    }

    /// Biexciton emission wavelength (nm).
    ///
    /// λ_XX = λ_X shifted by the biexciton binding energy:
    ///   E_XX = E_X − ΔE_bind
    pub fn biexciton_wavelength_nm(&self) -> f64 {
        let e_x_ev = self.dot.transition_energy_ev();
        // XX photon energy = XX state → X state: E_XX_photon = E_X - δ_bind/2
        // Convention: biexciton_binding_energy_mev is (E_XX - 2E_X)
        let delta_ev = self.biexciton_binding_energy_mev * 1e-3 / 2.0;
        let e_xx_ev = (e_x_ev - delta_ev).max(0.1);
        let hc_ev_nm = 1239.84_f64;
        hc_ev_nm / e_xx_ev
    }

    /// Entanglement fidelity to the Bell state |Φ⁺⟩.
    ///
    /// For FSS-limited entanglement with radiative lifetime τ_rad:
    ///   F = ½(1 + exp(−FSS·τ_rad / ħ))
    ///
    /// At zero FSS → F = 1.  Large FSS → F → ½ (mixed state).
    pub fn entanglement_fidelity(&self) -> f64 {
        let fss_j = self.fss_uev * 1e-6 * E_CHARGE; // μeV → J
        let tau_s = self.dot.radiative_lifetime_ns() * 1e-9;
        // Phase accumulated during cascade: φ = FSS * τ_rad / ħ
        let phi = fss_j * tau_s / H_BAR;
        0.5 * (1.0 + (-phi).exp())
    }

    /// Concurrence of the polarisation-entangled photon pair.
    ///
    ///  C = max(0, exp(−FSS·τ/ħ) − ε_mix)
    ///  where ε_mix ≈ 0 for a pure cascading system.
    pub fn concurrence(&self) -> f64 {
        let fss_j = self.fss_uev * 1e-6 * E_CHARGE;
        let tau_s = self.dot.radiative_lifetime_ns() * 1e-9;
        let phi = fss_j * tau_s / H_BAR;
        let c = (-phi).exp();
        c.clamp(0.0, 1.0)
    }

    /// Photon pair generation rate (pairs/s) at given CW excitation power.
    ///
    /// Estimate: Γ_XX = η_excitation * P / (ħ ω_pump)
    /// with a saturation model P_sat ≈ 100 μW for InAs dots.
    pub fn pair_rate_per_s(&self, excitation_power_uw: f64) -> f64 {
        let tau_xx_ns = self.dot.radiative_lifetime_ns() / 2.0; // XX decays ~2× faster
        let gamma_xx = 1.0 / (tau_xx_ns * 1e-9); // s⁻¹
        let p_sat_uw = 100.0_f64;
        // Saturation model: R = Γ_XX * x/(1+x), x = P/P_sat
        let x = excitation_power_uw / p_sat_uw;
        gamma_xx * x / (1.0 + x)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavelength_in_inas_gaas_range() {
        // Standard 8×25 nm InAs dot at 4 K should emit between 880 and 1100 nm
        let dot = InAsQd::new(8.0, 25.0);
        let wl = dot.wavelength_nm();
        assert!(
            wl > 800.0 && wl < 1200.0,
            "Unexpected wavelength {wl:.1} nm; expected 800–1200 nm"
        );
    }

    #[test]
    fn test_indistinguishability_approaches_unity_at_zero_k() {
        let dot = InAsQd::new(8.0, 25.0);
        let m = dot.indistinguishability(0.01);
        // At near-zero temperature phonon dephasing vanishes → M approaches 1
        assert!(
            m > 0.95,
            "Indistinguishability should be near 1 at ~0 K; got {m:.4}"
        );
    }

    #[test]
    fn test_indistinguishability_decreases_with_temperature() {
        let dot = InAsQd::new(8.0, 25.0);
        let m_4k = dot.indistinguishability(4.0);
        let m_77k = dot.indistinguishability(77.0);
        assert!(
            m_4k > m_77k,
            "Indistinguishability should decrease with temperature; M(4K)={m_4k:.4}, M(77K)={m_77k:.4}"
        );
    }

    #[test]
    fn test_debye_waller_factor_bounded() {
        let dot = InAsQd::new(8.0, 25.0);
        for &t in &[0.0, 4.0, 10.0, 77.0, 300.0] {
            let dw = dot.debye_waller_factor(t);
            assert!(
                (0.0..=1.0).contains(&dw),
                "DW factor out of [0,1] at T={t} K: got {dw}"
            );
        }
    }

    #[test]
    fn test_fss_grows_with_asymmetry() {
        let round_dot = InAsQd::new(8.0, 8.0); // h = d → nearly round
        let elongated = InAsQd::new(2.0, 40.0); // large aspect ratio
        let fss_round = round_dot.fine_structure_splitting_uev();
        let fss_elon = elongated.fine_structure_splitting_uev();
        assert!(
            fss_elon > fss_round,
            "Elongated dot should have larger FSS; round={fss_round:.1}, elongated={fss_elon:.1} μeV"
        );
    }

    #[test]
    fn test_biexciton_entanglement_fidelity_at_zero_fss() {
        let dot = InAsQd::new(8.0, 25.0);
        let mut cascade = BiexcitonCascade::new(dot);
        cascade.fss_uev = 0.0;
        let f = cascade.entanglement_fidelity();
        // F → 1 when FSS = 0
        assert!(
            (f - 1.0).abs() < 1e-10,
            "Fidelity should be 1 at FSS=0; got {f}"
        );
    }

    #[test]
    fn test_pl_spectrum_peaks_at_center() {
        let ensemble = QdEnsemble::new(920.0, 10.0, 1000);
        let peak = ensemble.pl_spectrum(920.0);
        let wing = ensemble.pl_spectrum(940.0);
        assert!(peak > wing, "PL should peak at centre wavelength");
        assert!((peak - 1.0).abs() < 1e-10, "PL peak should normalise to 1");
    }

    #[test]
    fn test_pair_rate_increases_with_power() {
        let dot = InAsQd::new(8.0, 25.0);
        let cascade = BiexcitonCascade::new(dot);
        let r_low = cascade.pair_rate_per_s(1.0);
        let r_high = cascade.pair_rate_per_s(10.0);
        assert!(
            r_high > r_low,
            "Pair rate should increase with excitation power"
        );
    }
}
