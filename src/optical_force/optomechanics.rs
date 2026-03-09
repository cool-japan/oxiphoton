/// Cavity Optomechanics
///
/// Implements the physics of cavity optomechanics where radiation pressure couples
/// the intracavity photon field to a mechanical oscillator.
///
/// # Hamiltonian
///
/// The optomechanical Hamiltonian (in the rotating frame) is:
///   H = -ħΔ a†a + ħΩ_m b†b + ħg0(b† + b) a†a + input/output terms
///
/// where:
/// - a (a†) are photon annihilation (creation) operators
/// - b (b†) are phonon annihilation (creation) operators
/// - Δ = ω_L - ω_c is the laser-cavity detuning
/// - g0 is the vacuum optomechanical coupling rate
/// - Ω_m is the mechanical oscillation frequency
///
/// # Key Phenomena
///
/// - **Dynamical backaction**: The cavity field modifies mechanical damping and frequency
/// - **Sideband cooling**: Red-detuned driving (Δ = -Ω_m) cools the mechanical mode
/// - **Optomechanical entanglement**: Strong coupling enables quantum correlations
/// - **Optical spring**: Radiation pressure shifts the mechanical resonance frequency
use crate::error::OxiPhotonError;

/// Reduced Planck constant (J·s)
const HBAR: f64 = 1.054_571_817e-34;
/// Boltzmann constant (J/K)
const KB: f64 = 1.380_649e-23;

/// Cavity optomechanical system.
///
/// Describes a Fabry-Pérot cavity where radiation pressure couples the intracavity
/// electromagnetic field to a mechanical degree of freedom (e.g., a movable mirror).
///
/// The cavity resonance frequency depends on mirror position x:
///   ω_c(x) = ω_c0 - G*x  (G = dω_c/dx = -ω_c0/L for Fabry-Pérot)
#[derive(Debug, Clone)]
pub struct OptomechanicalCavity {
    /// Cavity resonance frequency (rad/s)
    pub omega_c: f64,
    /// Total cavity decay rate κ = κ_ex + κ_0 (rad/s)
    pub kappa: f64,
    /// External (coupling) decay rate κ_ex (rad/s)
    pub kappa_ex: f64,
    /// Mechanical oscillation frequency Ω_m (rad/s)
    pub omega_m: f64,
    /// Intrinsic mechanical damping rate γ_m (rad/s)
    pub gamma_m: f64,
    /// Effective mass of mechanical mode (kg)
    pub mass_kg: f64,
    /// Vacuum optomechanical coupling: g0 = G * x_zpf (rad/s)
    pub g0: f64,
    /// Zero-point fluctuation amplitude (m), computed from mass and omega_m
    pub zero_point_motion: f64,
}

impl OptomechanicalCavity {
    /// Create a new optomechanical cavity configuration.
    ///
    /// # Arguments
    /// * `omega_c` - Cavity resonance frequency (rad/s)
    /// * `kappa` - Total cavity linewidth (rad/s)
    /// * `kappa_ex` - External coupling rate (rad/s), must be ≤ kappa
    /// * `omega_m` - Mechanical frequency (rad/s)
    /// * `gamma_m` - Mechanical damping rate (rad/s)
    /// * `mass_kg` - Effective mechanical mass (kg)
    /// * `g0` - Vacuum optomechanical coupling (rad/s)
    pub fn new(
        omega_c: f64,
        kappa: f64,
        kappa_ex: f64,
        omega_m: f64,
        gamma_m: f64,
        mass_kg: f64,
        g0: f64,
    ) -> Self {
        let x_zpf = (HBAR / (2.0 * mass_kg * omega_m)).sqrt();
        Self {
            omega_c,
            kappa,
            kappa_ex,
            omega_m,
            gamma_m,
            mass_kg,
            g0,
            zero_point_motion: x_zpf,
        }
    }

    /// Zero-point fluctuation amplitude: x_zpf = sqrt(ħ/(2mΩ_m)) (m).
    ///
    /// This is the quantum ground-state position uncertainty of the mechanical oscillator.
    pub fn zero_point_motion(&self) -> f64 {
        (HBAR / (2.0 * self.mass_kg * self.omega_m)).sqrt()
    }

    /// Optomechanical coupling rate per unit displacement: G = g0 / x_zpf (rad/s/m).
    ///
    /// Represents how strongly each meter of mirror displacement shifts the cavity frequency.
    /// For a Fabry-Pérot cavity: G = ω_c / L.
    pub fn coupling_rate_per_m(&self) -> f64 {
        let x_zpf = self.zero_point_motion();
        self.g0 / x_zpf
    }

    /// Enhanced many-photon optomechanical coupling: g = g0 * sqrt(n̄_c) (rad/s).
    ///
    /// With n̄_c intracavity photons, the linearized coupling is amplified by sqrt(n̄_c).
    ///
    /// # Arguments
    /// * `n_bar_photons` - Mean intracavity photon number
    pub fn enhanced_coupling(&self, n_bar_photons: f64) -> f64 {
        self.g0 * n_bar_photons.sqrt()
    }

    /// Optomechanical cooperativity: C = 4g²/(κ γ_m) = 4g0² n̄/(κ γ_m).
    ///
    /// The cooperativity quantifies whether the optomechanical interaction
    /// dominates over dissipation. C > 1 is the strong coupling regime.
    ///
    /// # Arguments
    /// * `n_bar_photons` - Mean intracavity photon number
    pub fn cooperativity(&self, n_bar_photons: f64) -> f64 {
        let g = self.enhanced_coupling(n_bar_photons);
        4.0 * g * g / (self.kappa * self.gamma_m)
    }

    /// Optomechanical (optical) damping rate from dynamical backaction.
    ///
    /// In the rotating-wave approximation, the optical damping is:
    ///   Γ_opt = Γ_+ - Γ_-
    ///
    /// where:
    ///   Γ_± = g² κ / [κ²/4 + (Δ ∓ Ω_m)²]
    ///
    /// Red-detuned driving (Δ < 0, Δ ≈ -Ω_m) gives cooling: Γ_opt > 0.
    /// Blue-detuned driving (Δ > 0, Δ ≈ +Ω_m) gives amplification: Γ_opt < 0.
    ///
    /// # Arguments
    /// * `detuning` - Laser-cavity detuning Δ = ω_L - ω_c (rad/s), negative for red
    /// * `n_bar` - Mean intracavity photon number
    ///
    /// # Returns
    /// Optical damping rate Γ_opt (rad/s)
    pub fn optical_damping(&self, detuning: f64, n_bar: f64) -> f64 {
        let g2 = self.enhanced_coupling(n_bar).powi(2);
        let kappa_half = self.kappa / 2.0;
        let kh2 = kappa_half * kappa_half;

        // Anti-Stokes (cooling) sideband at cavity frequency ω_c + Ω_m:
        // Laser must satisfy ω_L + Ω_m ≈ ω_c → Δ = ω_L - ω_c ≈ -Ω_m
        // The relevant detuning pole is at Δ + Ω_m = 0 when Δ = -Ω_m
        let delta_anti_stokes = detuning + self.omega_m;
        let gamma_cooling = g2 * self.kappa / (kh2 + delta_anti_stokes * delta_anti_stokes);

        // Stokes (heating) sideband at cavity frequency ω_c - Ω_m:
        // The relevant detuning pole is at Δ - Ω_m = 0 when Δ = +Ω_m
        let delta_stokes = detuning - self.omega_m;
        let gamma_heating = g2 * self.kappa / (kh2 + delta_stokes * delta_stokes);

        // Γ_opt = A_- - A_+ where A_- (A_+) is the anti-Stokes (Stokes) scattering rate
        gamma_cooling - gamma_heating
    }

    /// Optical spring frequency shift from radiation-pressure backaction.
    ///
    /// The spring constant shift is:
    ///   δΩ_m = g² * [-(Δ - Ω_m)/(κ²/4 + (Δ-Ω_m)²) + (Δ + Ω_m)/(κ²/4 + (Δ+Ω_m)²)]
    ///
    /// # Arguments
    /// * `detuning` - Laser-cavity detuning Δ (rad/s)
    /// * `n_bar` - Mean intracavity photon number
    ///
    /// # Returns
    /// Mechanical frequency shift δΩ_m (rad/s)
    pub fn optical_spring_shift(&self, detuning: f64, n_bar: f64) -> f64 {
        let g2 = self.enhanced_coupling(n_bar).powi(2);
        let kh2 = (self.kappa / 2.0).powi(2);

        let delta_minus = detuning - self.omega_m;
        let delta_plus = detuning + self.omega_m;

        g2 * (-delta_minus / (kh2 + delta_minus * delta_minus)
            + delta_plus / (kh2 + delta_plus * delta_plus))
    }

    /// Sideband resolution parameter: Ω_m / (κ/2).
    ///
    /// Resolved sideband: Ω_m >> κ/2 → individual sidebands can be addressed.
    /// Unresolved sideband: Ω_m << κ/2 → sidebands overlap.
    pub fn sideband_resolution(&self) -> f64 {
        self.omega_m / (self.kappa / 2.0)
    }

    /// Returns true if the system is in the resolved sideband regime (Ω_m > κ/2).
    pub fn is_resolved_sideband(&self) -> bool {
        self.omega_m > self.kappa / 2.0
    }

    /// Mean thermal phonon number at temperature T.
    ///
    /// From Bose-Einstein statistics:
    ///   n̄_th = 1 / [exp(ħΩ_m / kT) - 1]
    ///
    /// # Arguments
    /// * `temperature_k` - Temperature (K)
    pub fn thermal_phonons(&self, temperature_k: f64) -> f64 {
        if temperature_k < 1.0e-10 {
            return 0.0;
        }
        let x = HBAR * self.omega_m / (KB * temperature_k);
        if x > 700.0 {
            // Avoid overflow: high frequency / low temperature limit
            (-x).exp()
        } else {
            1.0 / (x.exp() - 1.0)
        }
    }

    /// Standard quantum limit (SQL) for position measurement sensitivity.
    ///
    /// x_SQL = sqrt(ħ / (m Ω_m))
    ///
    /// At the SQL, measurement imprecision equals the quantum backaction noise.
    /// This is a factor of sqrt(2) above the zero-point motion.
    pub fn sql_position_m(&self) -> f64 {
        (HBAR / (self.mass_kg * self.omega_m)).sqrt()
    }

    /// Minimum achievable mean phonon number in ground-state cooling.
    ///
    /// In the resolved sideband limit with red-detuned driving (Δ = -Ω_m):
    ///   n̄_min ≈ (κ / 4Ω_m)²
    ///
    /// This is the quantum backaction cooling limit.
    ///
    /// # Arguments
    /// * `n_bar_photons` - Mean intracavity photon number (for verification of cooperativity)
    ///
    /// # Returns
    /// Minimum achievable mean phonon number
    pub fn minimum_phonon_number(&self, n_bar_photons: f64) -> f64 {
        // Quantum backaction limit (resolved sideband):
        let n_ba = (self.kappa / (4.0 * self.omega_m)).powi(2);

        // Classical noise limit from finite cooperativity:
        let cooperativity = self.cooperativity(n_bar_photons);
        if cooperativity < 1.0e-30 {
            return f64::INFINITY;
        }
        let n_classical = 1.0 / cooperativity;

        // The achievable minimum is max of these two floors
        n_ba.max(n_classical)
    }

    /// Critical intracavity photon number for unit cooperativity.
    ///
    /// C = 1 when n̄ = κ γ_m / (4 g0²)
    pub fn critical_photon_number(&self) -> f64 {
        self.kappa * self.gamma_m / (4.0 * self.g0 * self.g0)
    }

    /// Steady-state intracavity photon number from input power and detuning.
    ///
    /// From input-output theory, the intracavity amplitude:
    ///   |a|² = (κ_ex / ħω_c) * P_in / [κ²/4 + Δ²]
    ///
    /// Note: we use the approximation ħω_c ≈ photon energy at cavity resonance.
    ///
    /// # Arguments
    /// * `input_power_w` - Input laser power (W)
    /// * `detuning` - Laser-cavity detuning Δ (rad/s)
    ///
    /// # Returns
    /// Mean intracavity photon number n̄_c
    pub fn intracavity_photons(&self, input_power_w: f64, detuning: f64) -> f64 {
        let photon_energy = HBAR * self.omega_c;
        let input_photon_rate = input_power_w / photon_energy;
        let kh2 = (self.kappa / 2.0).powi(2);
        self.kappa_ex * input_photon_rate / (kh2 + detuning * detuning)
    }

    /// Check whether quantum entanglement between photons and phonons is possible.
    ///
    /// Entanglement requires strong cooperativity and low thermal occupation:
    ///   C_q = C / n̄_th >> 1
    ///
    /// A necessary condition is C > n̄_th (quantum cooperativity > 1).
    ///
    /// # Arguments
    /// * `n_bar_photons` - Intracavity photon number
    /// * `temperature_k` - Bath temperature (K)
    ///
    /// # Returns
    /// `true` if entanglement criterion C > n̄_th is met
    pub fn entanglement_possible(&self, n_bar_photons: f64, temperature_k: f64) -> bool {
        let c = self.cooperativity(n_bar_photons);
        let n_th = self.thermal_phonons(temperature_k);
        c > n_th
    }
}

/// Membrane-in-the-middle optomechanical system.
///
/// A partially reflective membrane placed inside a Fabry-Pérot cavity
/// enables strong optomechanical coupling while preserving high mechanical
/// quality factor (the membrane can be made very thin and rigid).
///
/// The coupling depends on membrane position as:
///   g0_eff = g0_bare * |cos(2 k_c x_m)|
///
/// where k_c = ω_c / c is the cavity wave number.
#[derive(Debug, Clone)]
pub struct MembraneOptomechanics {
    /// Underlying optomechanical cavity description
    pub cavity: OptomechanicalCavity,
    /// Membrane power reflectivity (0-1)
    pub membrane_reflectivity: f64,
    /// Membrane equilibrium position as fraction of cavity length (0-1)
    pub membrane_position_fraction: f64,
}

impl MembraneOptomechanics {
    /// Create a membrane-in-the-middle system.
    ///
    /// # Arguments
    /// * `cavity` - Base optomechanical cavity parameters
    /// * `reflectivity` - Membrane reflectivity (0-1)
    /// * `position_fraction` - Membrane position as fraction of cavity length (0-1)
    ///
    /// # Errors
    /// Returns error if reflectivity is outside [0, 1] or position is outside [0, 1]
    pub fn new(
        cavity: OptomechanicalCavity,
        reflectivity: f64,
        position_fraction: f64,
    ) -> Result<Self, OxiPhotonError> {
        if !(0.0..=1.0).contains(&reflectivity) {
            return Err(OxiPhotonError::NumericalError(format!(
                "Membrane reflectivity {:.4} must be in [0, 1]",
                reflectivity
            )));
        }
        if !(0.0..=1.0).contains(&position_fraction) {
            return Err(OxiPhotonError::NumericalError(format!(
                "Membrane position fraction {:.4} must be in [0, 1]",
                position_fraction
            )));
        }
        Ok(Self {
            cavity,
            membrane_reflectivity: reflectivity,
            membrane_position_fraction: position_fraction,
        })
    }

    /// Position-dependent coupling enhancement factor |cos(2k_c x_m)|.
    ///
    /// The coupling is maximum at the intensity anti-node (cos² = 1)
    /// and zero at the intensity node (cos² = 0).
    pub fn coupling_enhancement(&self) -> f64 {
        // Phase accumulated at membrane position: φ = 2 * k_c * x_m
        // where x_m = position_fraction * L_cavity
        // k_c = ω_c / c, and for a round trip: 2 * k_c * L = 2π * FSR/FSR = 2π * m
        // The phase is 2π * position_fraction * (cavity mode number)
        // Simplified: use position_fraction directly as the fractional round-trip phase
        let phase = 2.0 * std::f64::consts::PI * self.membrane_position_fraction;
        phase.cos().abs()
    }

    /// Effective vacuum coupling including membrane position and reflectivity.
    ///
    /// g0_eff = g0_bare * sqrt(R) * |cos(2k_c x_m)|
    ///
    /// The sqrt(R) factor accounts for partial reflectivity of the membrane.
    pub fn effective_g0(&self) -> f64 {
        self.cavity.g0 * self.membrane_reflectivity.sqrt() * self.coupling_enhancement()
    }
}

/// Photon recoil in atomic physics.
///
/// When an atom absorbs or emits a photon, it receives a momentum kick
/// of ħk. This limits laser cooling to the recoil temperature.
///
/// # Physical Context
///
/// The recoil energy and temperature set fundamental limits in:
/// - Laser cooling (Doppler cooling limit > recoil limit for many atoms)
/// - Atom interferometry (recoil frequency as metrological reference)
/// - Atom optics (matter wave diffraction)
#[derive(Debug, Clone)]
pub struct PhotonRecoil {
    /// Atomic mass (atomic mass units, 1 amu = 1.66054e-27 kg)
    pub atom_mass_amu: f64,
    /// Photon wavelength (nm)
    pub wavelength_nm: f64,
}

/// Conversion: 1 atomic mass unit in kg
const AMU_KG: f64 = 1.660_539_066_6e-27;

impl PhotonRecoil {
    /// Create a photon recoil calculator for a given atom and wavelength.
    ///
    /// # Arguments
    /// * `atom_mass_amu` - Atomic mass (amu)
    /// * `lambda_nm` - Wavelength of the photon (nm)
    pub fn new(atom_mass_amu: f64, lambda_nm: f64) -> Self {
        Self {
            atom_mass_amu,
            wavelength_nm: lambda_nm,
        }
    }

    /// Atom mass in kilograms.
    fn mass_kg(&self) -> f64 {
        self.atom_mass_amu * AMU_KG
    }

    /// Photon wave vector magnitude k = 2π/λ (m⁻¹).
    fn k_vector(&self) -> f64 {
        2.0 * std::f64::consts::PI / (self.wavelength_nm * 1.0e-9)
    }

    /// Single-photon recoil velocity: v_r = ħk/m (m/s).
    ///
    /// The velocity kick an atom receives from absorbing one photon.
    pub fn recoil_velocity_m_per_s(&self) -> f64 {
        HBAR * self.k_vector() / self.mass_kg()
    }

    /// Single-photon recoil energy: E_r = ħ²k²/(2m) (J).
    ///
    /// Equivalently: E_r = ½ m v_r²
    pub fn recoil_energy_j(&self) -> f64 {
        let vr = self.recoil_velocity_m_per_s();
        0.5 * self.mass_kg() * vr * vr
    }

    /// Recoil temperature: T_r = ħ²k²/(2m kB) (nanokelvin).
    ///
    /// The temperature equivalent of the recoil energy. This sets
    /// the fundamental lower limit for Doppler cooling.
    ///
    /// # Returns
    /// Recoil temperature (nK)
    pub fn recoil_temperature_nk(&self) -> f64 {
        let e_r = self.recoil_energy_j();
        let t_r_k = e_r / KB;
        t_r_k * 1.0e9 // convert K to nK
    }

    /// Doppler cooling limit: T_D = ħΓ/(2kB) (microkelvin).
    ///
    /// The minimum temperature achievable by Doppler cooling, set by
    /// the balance between momentum diffusion from photon absorption
    /// and the damping force.
    ///
    /// # Arguments
    /// * `linewidth_hz` - Natural linewidth of the cooling transition Γ (Hz)
    ///
    /// # Returns
    /// Doppler cooling limit temperature (μK)
    pub fn doppler_cooling_limit_uk(&self, linewidth_hz: f64) -> f64 {
        let gamma = 2.0 * std::f64::consts::PI * linewidth_hz;
        let t_d_k = HBAR * gamma / (2.0 * KB);
        t_d_k * 1.0e6 // convert K to μK
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Standard optomechanical system: silica microsphere cavity
    /// ω_c ≈ 2π × 200 THz, κ ≈ 2π × 15 MHz (Q ~ 10^7)
    /// Ω_m ≈ 2π × 50 MHz (resolved sideband), γ_m ≈ 2π × 100 Hz
    /// mass ~ 10 ng, g0 ≈ 2π × 100 Hz
    fn standard_cavity() -> OptomechanicalCavity {
        let two_pi = 2.0 * std::f64::consts::PI;
        OptomechanicalCavity::new(
            two_pi * 200.0e12, // ω_c: 200 THz
            two_pi * 15.0e6,   // κ: 15 MHz
            two_pi * 7.5e6,    // κ_ex: 7.5 MHz (critical coupling)
            two_pi * 50.0e6,   // Ω_m: 50 MHz
            two_pi * 100.0,    // γ_m: 100 Hz
            10.0e-12,          // mass: 10 ng
            two_pi * 100.0,    // g0: 100 Hz
        )
    }

    #[test]
    fn test_zero_point_motion() {
        let cav = standard_cavity();
        let x_zpf_calc = cav.zero_point_motion();
        let x_zpf_expected = (HBAR / (2.0 * cav.mass_kg * cav.omega_m)).sqrt();
        assert_abs_diff_eq!(x_zpf_calc, x_zpf_expected, epsilon = 1.0e-25);
        // For 10 ng at 50 MHz: ~4.6 fm
        assert!(x_zpf_calc > 1.0e-16, "x_zpf should be in femtometer range");
        assert!(x_zpf_calc < 1.0e-12, "x_zpf should be in femtometer range");
    }

    #[test]
    fn test_cooperativity_formula() {
        let cav = standard_cavity();
        let n_bar = 1000.0;
        let c = cav.cooperativity(n_bar);
        // C = 4 * g0² * n_bar / (κ * γ_m)
        let expected = 4.0 * cav.g0 * cav.g0 * n_bar / (cav.kappa * cav.gamma_m);
        assert_abs_diff_eq!(c, expected, epsilon = 1.0e-10);
    }

    #[test]
    fn test_resolved_sideband() {
        let cav = standard_cavity();
        // Ω_m = 50 MHz, κ = 15 MHz → Ω_m > κ/2 = 7.5 MHz ✓
        assert!(
            cav.is_resolved_sideband(),
            "50 MHz mechanical, 15 MHz linewidth should be resolved sideband"
        );
        let resolution = cav.sideband_resolution();
        // Should be ~50/7.5 ≈ 6.67
        assert!(
            (resolution - 6.67).abs() < 0.1,
            "Sideband resolution = Ω_m/(κ/2)"
        );
    }

    #[test]
    fn test_thermal_phonons_at_zero() {
        let cav = standard_cavity();
        // At T → 0, n_th → 0
        let n_th_cold = cav.thermal_phonons(1.0e-6); // 1 μK
        let n_th_zero = cav.thermal_phonons(0.0);
        assert!(n_th_cold < 1.0e-10, "At 1 μK, n_th should be negligible");
        assert_abs_diff_eq!(n_th_zero, 0.0, epsilon = 1.0e-30);
    }

    #[test]
    fn test_thermal_phonons_high_temperature() {
        let cav = standard_cavity();
        // At room temperature (300 K) for 50 MHz: n_th = kT/(ħΩ_m) ≈ 6300
        let n_th = cav.thermal_phonons(300.0);
        let n_th_classical = KB * 300.0 / (HBAR * cav.omega_m);
        // Should agree within ~1% in classical limit
        assert!(
            (n_th / n_th_classical - 1.0).abs() < 0.01,
            "High-T limit: n_th ≈ kT/ħΩ_m, got {} vs {}",
            n_th,
            n_th_classical
        );
    }

    #[test]
    fn test_minimum_phonon_number() {
        let cav = standard_cavity();
        // In the resolved sideband limit, ground state cooling is possible
        // With sufficient photons, n_min < 1
        let n_bar_high = 1.0e8; // many photons
        let n_min = cav.minimum_phonon_number(n_bar_high);
        // In resolved sideband: n_min ≈ (κ/4Ω_m)² ≈ (15/200)² ≈ 0.0056
        assert!(
            n_min < 1.0,
            "Ground state cooling should be achievable, n_min = {}",
            n_min
        );
    }

    #[test]
    fn test_photon_recoil_velocity() {
        // Rb-87 at 780 nm: v_r = ħk/m
        let recoil = PhotonRecoil::new(87.0, 780.0);
        let vr = recoil.recoil_velocity_m_per_s();
        let k = 2.0 * std::f64::consts::PI / (780.0e-9);
        let m = 87.0 * AMU_KG;
        let expected = HBAR * k / m;
        assert_abs_diff_eq!(vr, expected, epsilon = 1.0e-10);
        // Rb-87 recoil velocity ~5.9 mm/s
        assert!(
            (vr - 5.9e-3).abs() < 0.1e-3,
            "Rb recoil ~5.9 mm/s, got {} mm/s",
            vr * 1000.0
        );
    }

    #[test]
    fn test_recoil_temperature() {
        // Rb-87 at 780 nm: T_r = ħ²k²/(2m*kB) ≈ 181 nK
        // (Some references define T_r = ħ²k²/(m*kB) = 2*E_r/kB ≈ 362 nK,
        //  but we use the thermodynamic definition T_r = E_r/kB = ħ²k²/(2m*kB).)
        let recoil = PhotonRecoil::new(87.0, 780.0);
        let t_r_nk = recoil.recoil_temperature_nk();
        // Check formula consistency: E_r = ħ²k²/(2m), T_r = E_r/kB
        let k = recoil.k_vector();
        let m = recoil.mass_kg();
        let e_r = HBAR * HBAR * k * k / (2.0 * m);
        let t_r_expected_nk = e_r / KB * 1.0e9;
        assert_abs_diff_eq!(t_r_nk, t_r_expected_nk, epsilon = 1.0e-6);
        // Rb-87 at 780 nm: ~180.9 nK (thermodynamic definition T_r = E_r/kB)
        assert!(
            (t_r_nk - 180.9).abs() < 1.0,
            "Rb recoil T ~180.9 nK, got {} nK",
            t_r_nk
        );
    }

    #[test]
    fn test_optical_damping_red_detuned() {
        let cav = standard_cavity();
        let n_bar = 1000.0;
        // Red detuning Δ = -Ω_m → maximum cooling
        let gamma_opt = cav.optical_damping(-cav.omega_m, n_bar);
        assert!(
            gamma_opt > 0.0,
            "Red-detuned driving should give positive (cooling) optical damping"
        );
    }

    #[test]
    fn test_optical_damping_blue_detuned() {
        let cav = standard_cavity();
        let n_bar = 1000.0;
        // Blue detuning Δ = +Ω_m → amplification
        let gamma_opt = cav.optical_damping(cav.omega_m, n_bar);
        assert!(
            gamma_opt < 0.0,
            "Blue-detuned driving should give negative (amplifying) optical damping"
        );
    }

    #[test]
    fn test_membrane_system_valid() {
        let cav = standard_cavity();
        let mem = MembraneOptomechanics::new(cav, 0.5, 0.25);
        assert!(mem.is_ok(), "Valid membrane parameters should succeed");
        let mem = mem.expect("valid parameters");
        assert!(
            mem.effective_g0() >= 0.0,
            "Effective g0 must be non-negative"
        );
    }

    #[test]
    fn test_membrane_invalid_reflectivity() {
        let cav = standard_cavity();
        let result = MembraneOptomechanics::new(cav, 1.5, 0.5);
        assert!(result.is_err(), "Reflectivity > 1 should fail");
    }

    #[test]
    fn test_intracavity_photons_resonance() {
        let cav = standard_cavity();
        let power = 1.0e-3; // 1 mW input
                            // On resonance (Δ=0), n̄ should be maximum
        let n_on = cav.intracavity_photons(power, 0.0);
        let n_off = cav.intracavity_photons(power, cav.omega_m);
        assert!(
            n_on > n_off,
            "On resonance should give more intracavity photons"
        );
        assert!(n_on > 0.0, "Must have positive photon number");
    }

    #[test]
    fn test_sql_vs_zpf() {
        let cav = standard_cavity();
        let x_sql = cav.sql_position_m();
        let x_zpf = cav.zero_point_motion();
        // SQL = sqrt(2) * x_zpf
        assert_abs_diff_eq!(x_sql, 2.0_f64.sqrt() * x_zpf, epsilon = 1.0e-25);
    }
}
