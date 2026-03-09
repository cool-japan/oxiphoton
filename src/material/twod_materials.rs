/// Two-dimensional material models: Graphene (Kubo formula), hexagonal boron nitride (hBN),
/// MoS₂, and black phosphorus.
///
/// Physical constants used throughout this module are defined at module level so that
/// all implementations share the same values without risk of accidental drift.
use num_complex::Complex64;

// ─── Physical constants ───────────────────────────────────────────────────────
const HBAR: f64 = 1.054_571_817e-34; // J·s (reduced Planck constant)
const KB: f64 = 1.380_649e-23; // J/K  (Boltzmann constant)
const E_CHARGE: f64 = 1.602_176_634e-19; // C   (elementary charge)
const EPS0: f64 = 8.854_187_817e-12; // F/m  (vacuum permittivity)
const SPEED_OF_LIGHT: f64 = 2.997_924_58e8; // m/s
                                            // Fine structure constant α = e²/(4πε₀ℏc)
const FINE_STRUCTURE: f64 = 7.297_352_569_3e-3;

// ─── GrapheneSheet ────────────────────────────────────────────────────────────

/// Surface conductivity of a graphene monolayer computed via the Kubo formula.
///
/// The total conductivity is split into intraband (Drude-like) and interband
/// (quantum transition) contributions:
///   σ(ω) = σ_intra(ω) + σ_inter(ω)
///
/// References
/// ----------
/// * Falkovsky & Varlamov, Eur. Phys. J. B 56, 281 (2007)
/// * Hanson, IEEE Trans. Antennas Propag. 56, 747 (2008)
#[derive(Debug, Clone)]
pub struct GrapheneSheet {
    /// Fermi energy in eV (≥ 0; doping level).
    pub fermi_energy_ev: f64,
    /// Carrier scattering time in picoseconds (> 0).
    pub relaxation_time_ps: f64,
    /// Lattice / carrier temperature in Kelvin.
    pub temperature_k: f64,
}

impl GrapheneSheet {
    /// Create a new graphene sheet with the given material parameters.
    ///
    /// # Arguments
    /// * `fermi_energy_ev`   – Fermi energy (eV).  Typical gated graphene: 0.1 – 0.6 eV.
    /// * `relaxation_time_ps` – Momentum-relaxation time (ps). High-quality graphene: ~1 ps.
    /// * `temperature_k`     – Temperature (K).
    pub fn new(fermi_energy_ev: f64, relaxation_time_ps: f64, temperature_k: f64) -> Self {
        Self {
            fermi_energy_ev,
            relaxation_time_ps,
            temperature_k,
        }
    }

    /// Fermi energy in Joules (internal helper).
    #[inline]
    fn fermi_energy_j(&self) -> f64 {
        self.fermi_energy_ev * E_CHARGE
    }

    /// Scattering rate Γ = 1/τ in rad/s.
    #[inline]
    fn gamma(&self) -> f64 {
        1.0 / (self.relaxation_time_ps * 1e-12)
    }

    /// Intraband (Drude-like) surface conductivity.
    ///
    /// Exact finite-temperature expression:
    ///   σ_intra = (i e² / π ℏ²) · (E_F / (ω + iΓ))
    ///
    /// At finite T the Fermi energy is effectively replaced by
    ///   ∫₀^∞ (f(E) + f(−E)) E dE  → E_F  for k_B T ≪ E_F
    ///
    /// For the general case the finite-T correction adds a ln(2cosh) term; we
    /// include it here for correctness at moderate temperatures.
    pub fn intraband(&self, omega: f64) -> Complex64 {
        let ef = self.fermi_energy_j();
        let gamma = self.gamma();
        let kt = KB * self.temperature_k;

        // Finite-temperature prefactor: 2 k_B T ln(2 cosh(E_F / 2k_BT))
        let x = ef / (2.0 * kt);
        let _thermal_factor = if x > 300.0 {
            // Numerical precision: cosh argument overflows – use asymptotic ef
            2.0 * ef
        } else {
            2.0 * kt * x.cosh().ln().mul_add(1.0, (2.0_f64).ln())
            // = 2kT * ln(2 cosh(x))
        };
        // 2kT * ln(2*cosh(x)) = 2kT*(ln(2) + ln(cosh(x)))
        // Re-derive more carefully:
        let thermal_factor = 2.0 * kt * ((2.0_f64 * x.cosh()).ln());

        let prefactor = E_CHARGE * E_CHARGE / (std::f64::consts::PI * HBAR * HBAR);
        let denom = Complex64::new(omega, gamma);
        Complex64::new(0.0, 1.0) * prefactor * thermal_factor / denom
    }

    /// Interband surface conductivity (quantum transition contribution).
    ///
    /// Simplified form (valid near room temperature):
    ///   σ_inter = (e²/4ℏ) · [tanh((ℏω/2 + E_F)/(2k_BT)) + tanh((ℏω/2 − E_F)/(2k_BT))]
    ///
    /// At T→0 this reduces to the step function σ_inter = e²/4ℏ for ℏω > 2E_F.
    pub fn interband(&self, omega: f64) -> Complex64 {
        let ef = self.fermi_energy_j();
        let kt = KB * self.temperature_k;
        let hbar_omega_half = HBAR * omega / 2.0;

        let t1 = ((hbar_omega_half + ef) / (2.0 * kt)).tanh();
        let t2 = ((hbar_omega_half - ef) / (2.0 * kt)).tanh();

        // Real part of interband conductivity
        let real_part = (E_CHARGE * E_CHARGE / (4.0 * HBAR)) * (t1 + t2);

        // Imaginary part via Kramers-Kronig (simplified principal-value integral):
        //   Im[σ_inter] ≈ (e²/4πℏ) · ln(|ℏω − 2E_F| / |ℏω + 2E_F|)
        let hbar_omega = HBAR * omega;
        let two_ef = 2.0 * ef;
        let im_part = if (hbar_omega - two_ef).abs() < 1e-30 {
            0.0
        } else {
            (E_CHARGE * E_CHARGE / (4.0 * std::f64::consts::PI * HBAR))
                * ((hbar_omega - two_ef).abs() / (hbar_omega + two_ef).abs()).ln()
        };

        Complex64::new(real_part, im_part)
    }

    /// Total complex surface conductivity σ(ω) = σ_intra(ω) + σ_inter(ω).
    pub fn surface_conductivity(&self, omega: f64) -> Complex64 {
        self.intraband(omega) + self.interband(omega)
    }

    /// Effective sheet permittivity for a thin-film model of graphene.
    ///
    ///   ε_eff = 1 + σ / (i ω ε₀ d)
    ///
    /// where `d` is the effective graphene thickness (typically 0.335 nm for a monolayer).
    pub fn sheet_permittivity(&self, omega: f64, thickness_m: f64) -> Complex64 {
        let sigma = self.surface_conductivity(omega);
        let i_omega_eps0_d = Complex64::new(0.0, omega * EPS0 * thickness_m);
        Complex64::new(1.0, 0.0) + sigma / i_omega_eps0_d
    }

    /// Graphene surface-plasmon wavevector from the dispersion relation:
    ///
    ///   k_sp = i ω ε₀ (ε_above + ε_below) / (2 σ)
    ///
    /// Highly confined when k_sp ≫ ω/c.
    pub fn plasmon_wavevector(&self, omega: f64, eps_above: f64, eps_below: f64) -> Complex64 {
        let sigma = self.surface_conductivity(omega);
        let numerator = Complex64::new(0.0, omega * EPS0 * (eps_above + eps_below));
        numerator / (2.0 * sigma)
    }

    /// Group velocity of graphene plasmons at the given angular frequency.
    ///
    /// Computed from the numerical derivative of the plasmon dispersion.
    pub fn plasmon_group_velocity(&self, omega: f64) -> f64 {
        let deps = 1e-3 * omega.abs().max(1.0);
        let k1 = self.plasmon_wavevector(omega - deps, 1.0, 1.0);
        let k2 = self.plasmon_wavevector(omega + deps, 1.0, 1.0);
        // dω/dk ≈ 2δω / (k2 - k1)
        let dk = (k2 - k1).re;
        if dk.abs() < 1e-30 {
            0.0
        } else {
            2.0 * deps / dk
        }
    }

    /// Universal optical absorption of graphene: π α ≈ 2.3 %.
    ///
    /// This is a fundamental constant independent of frequency.
    pub fn universal_absorption() -> f64 {
        std::f64::consts::PI * FINE_STRUCTURE
    }

    /// DC surface conductivity in the zero-frequency limit.
    ///
    ///   σ_DC = e² E_F τ / (π ℏ²)
    pub fn dc_conductivity(&self) -> f64 {
        let ef = self.fermi_energy_j();
        let tau = self.relaxation_time_ps * 1e-12;
        E_CHARGE * E_CHARGE * ef * tau / (std::f64::consts::PI * HBAR * HBAR)
    }
}

// ─── HexagonalBN ─────────────────────────────────────────────────────────────

/// Hexagonal boron nitride (hBN) – a natural hyperbolic phonon polariton material.
///
/// hBN has two distinct Reststrahlen bands where in-plane (ε_∥) and out-of-plane
/// (ε_⊥) permittivities take opposite signs, enabling hyperbolic dispersion of
/// phonon polaritons.
///
/// Lorentz oscillator parameters are taken from:
///   Caldwell et al., Nature Commun. 5, 5221 (2014)
///   Dai et al., Science 343, 1125 (2014)
#[derive(Debug, Clone)]
pub struct HexagonalBN {
    /// Number of atomic layers.  Affects effective screening (dielectric environment).
    pub n_layers: u32,
}

// hBN phonon parameters (angular frequencies in rad/s, converted from cm⁻¹)
// 1 cm⁻¹ = 2π × c × 100 rad/s
const CM1_TO_RADS: f64 = 2.0 * std::f64::consts::PI * SPEED_OF_LIGHT * 100.0;

// Upper Reststrahlen band (in-plane, ~1370 cm⁻¹ TO and ~1610 cm⁻¹ LO)
const HBN_TO_UPPER: f64 = 1370.0 * CM1_TO_RADS; // TO phonon frequency (in-plane)
const HBN_LO_UPPER: f64 = 1610.0 * CM1_TO_RADS; // LO phonon frequency (in-plane)
const HBN_GAMMA_UPPER: f64 = 5.0 * CM1_TO_RADS; // linewidth (in-plane upper)

// Lower Reststrahlen band (out-of-plane, ~760 cm⁻¹ TO and ~825 cm⁻¹ LO)
const HBN_TO_LOWER: f64 = 760.0 * CM1_TO_RADS;
const HBN_LO_LOWER: f64 = 825.0 * CM1_TO_RADS;
const HBN_GAMMA_LOWER: f64 = 4.0 * CM1_TO_RADS;

// High-frequency dielectric constants
const HBN_EPS_INF_INPLANE: f64 = 4.87;
const HBN_EPS_INF_OUTOFPLANE: f64 = 2.95;

impl HexagonalBN {
    /// Create a new hBN model with the given number of atomic layers.
    pub fn new(n_layers: u32) -> Self {
        Self { n_layers }
    }

    /// In-plane (ordinary) complex permittivity using a Lorentz oscillator model.
    ///
    ///   ε_∥(ω) = ε_∞ · [1 + (ω_LO² − ω_TO²) / (ω_TO² − ω² − iΓω)]
    pub fn permittivity_in_plane(&self, omega: f64) -> Complex64 {
        let to = HBN_TO_UPPER;
        let lo = HBN_LO_UPPER;
        let gamma = HBN_GAMMA_UPPER;
        let eps_inf = HBN_EPS_INF_INPLANE;

        // Screening correction for thin layers (simple dielectric confinement factor)
        let screening = if self.n_layers == 1 { 0.95 } else { 1.0 };

        let omega2 = omega * omega;
        let denom = Complex64::new(to * to - omega2, -gamma * omega);
        let osc = Complex64::new(lo * lo - to * to, 0.0) / denom;
        Complex64::new(eps_inf * screening, 0.0) * (Complex64::new(1.0, 0.0) + osc)
    }

    /// Out-of-plane (extraordinary) complex permittivity.
    ///
    ///   ε_⊥(ω) = ε_∞_⊥ · [1 + (ω_LO_z² − ω_TO_z²) / (ω_TO_z² − ω² − iΓ_zω)]
    pub fn permittivity_out_of_plane(&self, omega: f64) -> Complex64 {
        let to = HBN_TO_LOWER;
        let lo = HBN_LO_LOWER;
        let gamma = HBN_GAMMA_LOWER;
        let eps_inf = HBN_EPS_INF_OUTOFPLANE;

        let omega2 = omega * omega;
        let denom = Complex64::new(to * to - omega2, -gamma * omega);
        let osc = Complex64::new(lo * lo - to * to, 0.0) / denom;
        Complex64::new(eps_inf, 0.0) * (Complex64::new(1.0, 0.0) + osc)
    }

    /// Frequency ranges (in rad/s) where hBN is hyperbolic.
    ///
    /// Hyperbolic regime: ε_∥ and ε_⊥ have opposite signs.
    /// hBN has two such bands (Type I and Type II):
    ///   * Type II (upper): ω ∈ [ω_TO_∥, ω_LO_∥]  – ε_∥ < 0, ε_⊥ > 0
    ///   * Type I  (lower): ω ∈ [ω_TO_⊥, ω_LO_⊥]  – ε_⊥ < 0, ε_∥ > 0
    pub fn hyperbolic_frequency_range(&self) -> Vec<(f64, f64)> {
        vec![
            (HBN_TO_LOWER, HBN_LO_LOWER), // Type I: lower Reststrahlen band
            (HBN_TO_UPPER, HBN_LO_UPPER), // Type II: upper Reststrahlen band
        ]
    }

    /// Returns `true` when `omega` falls inside a hyperbolic band.
    pub fn is_hyperbolic(&self, omega: f64) -> bool {
        self.hyperbolic_frequency_range()
            .iter()
            .any(|&(lo, hi)| omega >= lo && omega <= hi)
    }

    /// Hyperbolic phonon polariton in-plane wavevector (simplified TM dispersion).
    ///
    /// For a hyperbolic medium the dispersion relation for a TM polariton mode is:
    ///   k_∥² / ε_⊥ + k_z² / ε_∥ = (ω/c)²
    ///
    /// Solving for k_∥ given k_z:
    ///   k_∥ = sqrt[ε_⊥ · ((ω/c)² − k_z²/ε_∥)]
    pub fn hpp_wavevector(&self, omega: f64, kz: f64) -> Complex64 {
        let eps_par = self.permittivity_in_plane(omega);
        let eps_perp = self.permittivity_out_of_plane(omega);
        let k0sq = (omega / SPEED_OF_LIGHT).powi(2);
        let kz_sq = Complex64::new(kz * kz, 0.0);
        let arg = eps_perp * (Complex64::new(k0sq, 0.0) - kz_sq / eps_par);
        arg.sqrt()
    }
}

// ─── MoS₂ ────────────────────────────────────────────────────────────────────

/// MoS₂ monolayer / few-layer model with valley degree of freedom.
///
/// MoS₂ transitions from an indirect bandgap semiconductor (bulk, ~1.2 eV)
/// to a direct-gap semiconductor (monolayer, ~1.8 eV) due to quantum confinement.
///
/// Permittivity uses a two-oscillator model capturing the A and B excitons.
///
/// References
/// ----------
/// * Mak et al., PRL 105, 136805 (2010)
/// * Splendiani et al., Nano Lett. 10, 1271 (2010)
#[derive(Debug, Clone)]
pub struct MoS2 {
    /// Number of layers.  1 = monolayer (direct gap); ≥ 2 transitions toward bulk.
    pub n_layers: u32,
    /// Biaxial strain ε (positive = tensile, negative = compressive).
    /// Strain tunes the bandgap at ~−50 meV/% for tensile strain.
    pub strain: f64,
}

impl MoS2 {
    /// Create a new MoS₂ model.
    pub fn new(n_layers: u32, strain: f64) -> Self {
        Self { n_layers, strain }
    }

    /// Direct bandgap energy in eV.
    ///
    /// * Monolayer: ~1.80 eV direct gap (K-point)
    /// * Bilayer: ~1.65 eV (still somewhat direct)
    /// * ≥3 layers: ~1.20 eV indirect gap (Γ–K)
    ///
    /// Strain shifts the gap by ~−50 meV/% (tensile decreases gap).
    pub fn bandgap_ev(&self) -> f64 {
        let base = match self.n_layers {
            1 => 1.80,
            2 => 1.65,
            _ => 1.20,
        };
        // Gauge factor: −50 meV per 1% biaxial tensile strain
        base - 0.050 * self.strain * 100.0
    }

    /// A-exciton transition energy in eV (spin-orbit split valence band, upper).
    pub fn a_exciton_energy_ev(&self) -> f64 {
        // A exciton is slightly below bandgap due to exciton binding energy (~0.5 eV)
        // Net optical gap (A exciton) ≈ 1.88 eV for monolayer
        let base = match self.n_layers {
            1 => 1.88,
            2 => 1.72,
            _ => 1.85, // bulk A exciton (less well-defined)
        };
        base - 0.050 * self.strain * 100.0
    }

    /// B-exciton transition energy in eV (spin-orbit split valence band, lower).
    ///
    /// Spin-orbit splitting in MoS₂ valence band ≈ 150 meV.
    pub fn b_exciton_energy_ev(&self) -> f64 {
        self.a_exciton_energy_ev() + 0.150 // ~150 meV spin-orbit splitting
    }

    /// Complex permittivity using a two-oscillator (A + B exciton) Lorentz model.
    ///
    ///   ε(ω) = ε_∞ + Σ_j  f_j · ω_j² / (ω_j² − ω² − i γ_j ω)
    pub fn permittivity(&self, omega: f64) -> Complex64 {
        let eps_inf = Complex64::new(15.0, 0.0); // high-frequency background

        // Convert exciton energies from eV to rad/s
        let ea = self.a_exciton_energy_ev() * E_CHARGE / HBAR;
        let eb = self.b_exciton_energy_ev() * E_CHARGE / HBAR;

        // Linewidths (FWHM in energy):  ~25 meV for A, ~40 meV for B
        let gamma_a = 0.025 * E_CHARGE / HBAR;
        let gamma_b = 0.040 * E_CHARGE / HBAR;

        // Oscillator strengths (arbitrary units calibrated to match experiments)
        let fa = 2.5;
        let fb = 1.5;

        let lorentz = |omega_0: f64, gamma: f64, f: f64| -> Complex64 {
            let denom = Complex64::new(omega_0 * omega_0 - omega * omega, -gamma * omega);
            Complex64::new(f * omega_0 * omega_0, 0.0) / denom
        };

        eps_inf + lorentz(ea, gamma_a, fa) + lorentz(eb, gamma_b, fb)
    }

    /// Returns `true` for monolayer (direct bandgap at K point).
    pub fn is_direct_bandgap(&self) -> bool {
        self.n_layers == 1
    }

    /// Valley polarization under circularly polarized excitation.
    ///
    /// In monolayer MoS₂ K and K′ valleys are optically active for σ+ and σ−
    /// light respectively due to broken inversion symmetry.
    ///
    /// `helicity` ∈ [−1, +1]: −1 = fully σ−, +1 = fully σ+.
    ///
    /// Returns the degree of valley polarization (0 = no polarization, 1 = complete).
    pub fn valley_polarization(&self, helicity: f64) -> f64 {
        if !self.is_direct_bandgap() {
            // Valley contrast is strongly reduced in indirect-gap (bulk) materials
            return 0.0;
        }
        // Simplified: linear in helicity, reduced by intervalley scattering
        // (intervalley scattering time ~1 ps, valley lifetime ~10 ps → η ≈ 0.3)
        let intrinsic_efficiency = 0.30;
        helicity.abs().min(1.0) * intrinsic_efficiency
    }
}

// ─── Black Phosphorus ─────────────────────────────────────────────────────────

/// Crystallographic direction in black phosphorus (anisotropic crystal).
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BPDirection {
    /// Armchair direction (x-axis): smaller effective mass, larger bandgap optical transition.
    Armchair,
    /// Zigzag direction (y-axis): larger effective mass, lower optical transition energy.
    Zigzag,
}

/// Black phosphorus – highly anisotropic 2D semiconductor (puckered lattice).
///
/// BP features strong in-plane optical anisotropy between armchair and zigzag
/// directions and a layer-dependent bandgap ranging from ~2.0 eV (monolayer) to
/// ~0.3 eV (bulk).
///
/// References
/// ----------
/// * Tran et al., PRB 89, 235319 (2014)
/// * Liu et al., Nature Chem. 6, 1023 (2014)
#[derive(Debug, Clone)]
pub struct BlackPhosphorus {
    /// Number of layers (1 = monolayer, ≥6 ≈ bulk).
    pub n_layers: u32,
    /// In-plane crystallographic direction for optical measurements.
    pub direction: BPDirection,
}

impl BlackPhosphorus {
    /// Create a new BP model.
    pub fn new(n_layers: u32, direction: BPDirection) -> Self {
        Self {
            n_layers,
            direction,
        }
    }

    /// Quasiparticle bandgap in eV.
    ///
    /// Strong layer-dependence (quantum confinement + interlayer coupling):
    ///   * Monolayer:  ~2.0 eV (direct, Γ)
    ///   * 2 layers:   ~1.3 eV
    ///   * 3 layers:   ~0.9 eV
    ///   * 5+ layers:  ~0.4 eV
    ///   * Bulk:       ~0.3 eV (direct at Γ)
    pub fn bandgap_ev(&self) -> f64 {
        match self.n_layers {
            1 => 2.00,
            2 => 1.30,
            3 => 0.90,
            4 => 0.60,
            5 => 0.45,
            _ => 0.30,
        }
    }

    /// Complex permittivity via a single-oscillator Lorentz model for the
    /// fundamental exciton along the chosen crystallographic direction.
    pub fn permittivity(&self, omega: f64) -> Complex64 {
        let eg_j = self.bandgap_ev() * E_CHARGE;
        let omega_0 = eg_j / HBAR; // fundamental transition frequency

        // Oscillator strength and linewidth depend on direction (anisotropy)
        let (f, gamma_ev) = match self.direction {
            BPDirection::Armchair => (8.0, 0.080), // stronger, narrower (AC)
            BPDirection::Zigzag => (3.0, 0.120),   // weaker, broader (ZZ)
        };
        let gamma = gamma_ev * E_CHARGE / HBAR;

        let eps_inf = Complex64::new(8.3, 0.0);
        let denom = Complex64::new(omega_0 * omega_0 - omega * omega, -gamma * omega);
        eps_inf + Complex64::new(f * omega_0 * omega_0, 0.0) / denom
    }

    /// Effective mass ratio m*/m_e along the chosen direction.
    ///
    /// BP has one of the most anisotropic band structures of any 2D material:
    ///   * Armchair: m* ≈ 0.15 m_e  (light mass, high mobility)
    ///   * Zigzag:   m* ≈ 0.70 m_e  (heavy mass)
    pub fn effective_mass_ratio(&self) -> f64 {
        match self.direction {
            BPDirection::Armchair => 0.15,
            BPDirection::Zigzag => 0.70,
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // Typical mid-IR frequency for graphene plasmonic tests (~10 THz)
    const OMEGA_TEST: f64 = 2.0 * PI * 10.0e12; // rad/s

    #[test]
    fn test_graphene_dc_conductivity() {
        // σ_DC = e² E_F τ / (π ℏ²)
        // Check linear scaling with E_F and τ
        let g1 = GrapheneSheet::new(0.2, 1.0, 300.0);
        let g2 = GrapheneSheet::new(0.4, 1.0, 300.0);
        let g3 = GrapheneSheet::new(0.2, 2.0, 300.0);

        let dc1 = g1.dc_conductivity();
        let dc2 = g2.dc_conductivity();
        let dc3 = g3.dc_conductivity();

        assert!(dc1 > 0.0, "DC conductivity must be positive");
        // Doubling E_F should double σ_DC
        let ratio_ef = dc2 / dc1;
        assert!(
            (ratio_ef - 2.0).abs() < 1e-10,
            "σ_DC should scale linearly with E_F, got ratio {ratio_ef}"
        );
        // Doubling τ should double σ_DC
        let ratio_tau = dc3 / dc1;
        assert!(
            (ratio_tau - 2.0).abs() < 1e-10,
            "σ_DC should scale linearly with τ, got ratio {ratio_tau}"
        );
    }

    #[test]
    fn test_graphene_universal_absorption() {
        let abs = GrapheneSheet::universal_absorption();
        // Should be ≈ πα ≈ 0.02293
        assert!(
            (abs - 0.02293).abs() < 5e-4,
            "Universal absorption should be ≈ 2.3%, got {abs:.5}"
        );
    }

    #[test]
    fn test_graphene_intraband_drude() {
        // At low frequencies (intraband dominates) Im(σ_intra) > 0 for positive E_F
        // This is the Drude-like tail: large imaginary part at ω ≪ E_F/ℏ
        let g = GrapheneSheet::new(0.3, 1.0, 300.0);
        let low_omega = 2.0 * PI * 1.0e12; // 1 THz — well below E_F/ℏ
        let sigma = g.intraband(low_omega);
        assert!(
            sigma.im > 0.0,
            "Im(σ_intra) should be positive for positive E_F at low ω, got {:.4e}",
            sigma.im
        );
    }

    #[test]
    fn test_graphene_plasmon_wavevector() {
        // Graphene plasmon wavevector should be larger than free-space k₀ (confined plasmon).
        // At mid-IR frequencies (~30 THz) with high-quality graphene (τ = 10 ps),
        // the confinement factor k_sp / k₀ exceeds 10.
        let g = GrapheneSheet::new(0.3, 10.0, 300.0); // τ = 10 ps (high quality)
        let omega = 2.0 * PI * 30.0e12; // 30 THz (mid-IR range, ~10 μm)
        let k_sp = g.plasmon_wavevector(omega, 1.0, 1.0);
        let k0 = omega / SPEED_OF_LIGHT;

        // |k_sp| ≫ k₀ confirms strong spatial confinement of the graphene plasmon
        let confinement = k_sp.norm() / k0;
        assert!(
            confinement > 10.0,
            "Graphene plasmon should be highly confined at 30 THz with τ=10ps: \
             |k_sp|/k₀ = {confinement:.2}, expected > 10"
        );
    }

    #[test]
    fn test_hbn_hyperbolic_range() {
        let hbn = HexagonalBN::new(10);
        let ranges = hbn.hyperbolic_frequency_range();
        assert!(
            !ranges.is_empty(),
            "hBN should have at least one hyperbolic frequency range"
        );
        for (lo, hi) in &ranges {
            assert!(
                hi > lo,
                "Each range must have hi > lo, got ({lo:.3e}, {hi:.3e})"
            );
        }
    }

    #[test]
    fn test_hbn_is_hyperbolic() {
        let hbn = HexagonalBN::new(10);
        // Test in the middle of the upper Reststrahlen band (~1490 cm⁻¹)
        let omega_mid_upper = 1490.0 * CM1_TO_RADS;
        assert!(
            hbn.is_hyperbolic(omega_mid_upper),
            "hBN should be hyperbolic at ~1490 cm⁻¹ (upper band)"
        );
        // Test in the middle of the lower Reststrahlen band (~793 cm⁻¹)
        let omega_mid_lower = 793.0 * CM1_TO_RADS;
        assert!(
            hbn.is_hyperbolic(omega_mid_lower),
            "hBN should be hyperbolic at ~793 cm⁻¹ (lower band)"
        );
        // Test outside both bands (e.g., 500 cm⁻¹)
        let omega_outside = 500.0 * CM1_TO_RADS;
        assert!(
            !hbn.is_hyperbolic(omega_outside),
            "hBN should NOT be hyperbolic at 500 cm⁻¹"
        );
    }

    #[test]
    fn test_mos2_monolayer_direct_gap() {
        let mos2 = MoS2::new(1, 0.0);
        assert!(
            mos2.is_direct_bandgap(),
            "MoS₂ monolayer must have a direct bandgap"
        );
        let eg = mos2.bandgap_ev();
        // Monolayer gap ≈ 1.8 eV
        assert!(
            (eg - 1.80).abs() < 0.05,
            "Monolayer MoS₂ gap ≈ 1.80 eV, got {eg:.3}"
        );
    }

    #[test]
    fn test_mos2_bulk_indirect() {
        let mos2 = MoS2::new(3, 0.0);
        assert!(
            !mos2.is_direct_bandgap(),
            "MoS₂ with 3+ layers must be indirect bandgap"
        );
    }

    #[test]
    fn test_mos2_bandgap_decreases_with_layers() {
        let eg1 = MoS2::new(1, 0.0).bandgap_ev();
        let eg2 = MoS2::new(2, 0.0).bandgap_ev();
        let eg3 = MoS2::new(3, 0.0).bandgap_ev();
        assert!(
            eg1 > eg2 && eg2 > eg3,
            "Bandgap must decrease with layer count: {eg1:.3} > {eg2:.3} > {eg3:.3}"
        );
    }

    #[test]
    fn test_graphene_sheet_permittivity() {
        // ε_eff should differ significantly from 1.0 (graphene modifies local dielectric)
        let g = GrapheneSheet::new(0.3, 1.0, 300.0);
        let d_graphene = 0.335e-9; // 0.335 nm monolayer thickness
        let eps = g.sheet_permittivity(OMEGA_TEST, d_graphene);
        let diff = (eps - Complex64::new(1.0, 0.0)).norm();
        assert!(
            diff > 0.01,
            "Sheet permittivity should deviate from vacuum, diff = {diff:.4}"
        );
    }

    #[test]
    fn test_bp_anisotropy() {
        let bp_ac = BlackPhosphorus::new(1, BPDirection::Armchair);
        let bp_zz = BlackPhosphorus::new(1, BPDirection::Zigzag);
        // Armchair effective mass is lighter
        assert!(
            bp_ac.effective_mass_ratio() < bp_zz.effective_mass_ratio(),
            "Armchair m* should be lighter than zigzag m*"
        );
    }

    #[test]
    fn test_bp_bandgap_layer_dependence() {
        let eg1 = BlackPhosphorus::new(1, BPDirection::Armchair).bandgap_ev();
        let eg2 = BlackPhosphorus::new(2, BPDirection::Armchair).bandgap_ev();
        let eg_bulk = BlackPhosphorus::new(10, BPDirection::Armchair).bandgap_ev();
        assert!(
            eg1 > eg2 && eg2 > eg_bulk,
            "BP bandgap must decrease with layers: {eg1:.2} > {eg2:.2} > {eg_bulk:.2}"
        );
    }
}
