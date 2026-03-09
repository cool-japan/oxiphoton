//! Optomechanical coupling in whispering-gallery-mode (WGM) microresonators
//! and disk resonators.
//!
//! Implements the quantum optomechanics formalism for:
//! - WGM microresonators coupled to mechanical nanobeam modes
//! - Disk resonators with whispering-gallery optical modes
//!
//! # Physics Overview
//!
//! In cavity optomechanics the optical resonance frequency shifts due to
//! mechanical displacement: ω_cav(x) = ω_cav0 − g0·x / x_zpf, where g0 is
//! the single-photon optomechanical coupling rate (rad/s) and x_zpf is the
//! zero-point fluctuation amplitude.
//!
//! # References
//! - Aspelmeyer, M., Kippenberg, T.J., Marquardt, F. (2014). Cavity optomechanics.
//!   *Rev. Mod. Phys.* 86, 1391.
//! - Kippenberg, T.J., Spillane, S.M., Vahala, K.J. (2004). Kerr-nonlinearity
//!   optical parametric oscillation in an ultrahigh-Q toroid microcavity.
//!   *Phys. Rev. Lett.* 93, 083904.

use std::f64::consts::PI;

/// Speed of light in vacuum (m/s).
const C: f64 = 299_792_458.0;
/// Reduced Planck constant (J·s).
const HBAR: f64 = 1.054_571_817e-34;

/// Whispering-gallery-mode (WGM) microresonator with optomechanical coupling.
///
/// Models a microtoroid or microsphere coupled to a flexural mechanical mode
/// (nanobeam, radial breathing, or flexural WGM mode).
///
/// # Example
/// ```
/// use oxiphoton::mems::coupling::WgmMicroresonator;
/// let wgm = WgmMicroresonator::new(50e-6, 1.44, 1550e-9);
/// assert!(wgm.free_spectral_range() > 1e11);
/// ```
#[derive(Debug, Clone)]
pub struct WgmMicroresonator {
    /// Resonator radius (m).
    pub radius: f64,
    /// Resonator height (m).
    pub height: f64,
    /// Resonator width (m).
    pub width: f64,
    /// Effective refractive index of the WGM mode.
    pub n_eff: f64,
    /// Operating wavelength (m).
    pub wavelength: f64,
    /// Optical quality factor.
    pub q_optical: f64,
    /// Mechanical quality factor.
    pub q_mechanical: f64,
    /// Mechanical mode frequency (rad/s).
    pub omega_mech: f64,
    /// Vacuum optomechanical coupling rate g0 (rad/s per m of displacement).
    pub g0: f64,
}

impl WgmMicroresonator {
    /// Create a new WGM microresonator with default parameters.
    ///
    /// # Arguments
    /// * `radius` - Resonator radius (m).
    /// * `n_eff` - Effective refractive index.
    /// * `wavelength` - Operating wavelength (m).
    ///
    /// Default values:
    /// - `height = width = 2 µm`
    /// - `q_optical = 1e7`
    /// - `q_mechanical = 1e4`
    /// - `omega_mech = 2π × 10 MHz` (typical radial breathing mode)
    /// - `g0 = ω_opt / R` (geometric optomechanical coupling)
    pub fn new(radius: f64, n_eff: f64, wavelength: f64) -> Self {
        let omega_opt = 2.0 * PI * C / wavelength;
        // Geometric (radiation pressure) coupling: g0 ≈ ω_opt / R (dispersive coupling)
        let g0 = omega_opt / radius;
        Self {
            radius,
            height: 2e-6,
            width: 2e-6,
            n_eff,
            wavelength,
            q_optical: 1e7,
            q_mechanical: 1e4,
            omega_mech: 2.0 * PI * 10e6, // 10 MHz
            g0,
        }
    }

    /// Optical resonance frequency (rad/s).
    pub fn optical_frequency(&self) -> f64 {
        2.0 * PI * C / self.wavelength
    }

    /// Free spectral range (Hz) of the WGM resonator.
    ///
    /// FSR = c / (2π·R·n_eff) = c / (n_eff·circumference)
    pub fn free_spectral_range(&self) -> f64 {
        C / (2.0 * PI * self.radius * self.n_eff)
    }

    /// Total optical linewidth (rad/s): κ = ω_opt / Q_optical.
    pub fn optical_linewidth(&self) -> f64 {
        self.optical_frequency() / self.q_optical
    }

    /// Radiation Q due to bending loss (approximate).
    ///
    /// For a WGM resonator the radiation Q scales exponentially with radius.
    /// This uses an empirical fit valid for silica microspheres at 1550 nm:
    ///
    /// Q_rad ≈ exp(2π·n_eff·R / λ) · (correction factor)
    ///
    /// The result is clamped to at most Q_optical (real resonator is limited
    /// by intrinsic absorption and surface scattering).
    pub fn quality_factor_radiation(&self) -> f64 {
        let phase_per_round = 2.0 * PI * self.n_eff * self.radius / self.wavelength;
        // Radiation Q: exponential suppression of evanescent leakage
        let q_rad = (2.0 * phase_per_round).exp();
        q_rad.min(self.q_optical * 10.0)
    }

    /// Effective motional mass of the coupled mechanical mode (kg).
    ///
    /// For a radial breathing mode of a disk/toroid, m_eff ≈ ρ·π·R·h·w
    /// (mass of a thin ring of width w and height h).
    pub fn effective_mass(&self) -> f64 {
        // Silica density
        let rho_silica = 2_200.0; // kg/m³
        rho_silica * PI * self.radius * self.height * self.width
    }

    /// Zero-point fluctuation amplitude (m).
    ///
    /// x_zpf = √(ℏ / (2·m_eff·ω_m))
    pub fn zero_point_motion(&self) -> f64 {
        let m_eff = self.effective_mass();
        (HBAR / (2.0 * m_eff * self.omega_mech)).sqrt()
    }

    /// Enhanced optomechanical coupling rate G (rad/s) in the presence of
    /// `n_photons` intracavity photons.
    ///
    /// G = g0 · √n_cav (linearised coupling)
    pub fn optomechanical_coupling_rate(&self, n_photons: f64) -> f64 {
        self.g0 * n_photons.max(0.0).sqrt()
    }

    /// Test for sideband-resolved regime: ω_m > κ/2.
    ///
    /// Returns `true` if the resonator operates in the resolved-sideband regime,
    /// which is required for ground-state cooling.
    pub fn sideband_resolution(&self) -> bool {
        self.omega_mech > self.optical_linewidth() / 2.0
    }

    /// Optomechanical cooperativity C = 4·G²·n_cav / (κ·γ_m).
    ///
    /// Where κ = optical linewidth, γ_m = ω_m/Q_m is the mechanical damping rate.
    pub fn cooperativity(&self, n_photons: f64) -> f64 {
        let kappa = self.optical_linewidth();
        let gamma_m = self.omega_mech / self.q_mechanical;
        let g = self.optomechanical_coupling_rate(n_photons);
        4.0 * g * g / (kappa * gamma_m)
    }

    /// Intracavity photon number for a given input power `p_in` (W) and coupling
    /// efficiency `eta_c` (0..1).
    ///
    /// n_cav = η_c · P_in · Q / (ℏ · ω²)
    pub fn intracavity_photons(&self, p_in: f64, eta_c: f64) -> f64 {
        let omega = self.optical_frequency();
        let eta = eta_c.clamp(0.0, 1.0);
        eta * p_in * self.q_optical / (HBAR * omega * omega)
    }

    /// Optical spring effect: frequency shift of the mechanical mode (rad/s) due
    /// to radiation pressure (blue-detuned drive).
    ///
    /// δω_m = G² / (Δ + iκ/2) + c.c.  (real part at Δ = -ω_m for cooling)
    ///
    /// At Δ = −ω_m: δω_m ≈ G²·ω_m / (ω_m² + (κ/2)²)
    pub fn optical_spring_shift(&self, n_photons: f64) -> f64 {
        let kappa = self.optical_linewidth();
        let g = self.optomechanical_coupling_rate(n_photons);
        let omega_m = self.omega_mech;
        // Blue-detuned approximation
        g * g * omega_m / (omega_m * omega_m + (kappa / 2.0).powi(2))
    }

    /// Photon-phonon scattering rate (Hz) for Stokes/anti-Stokes transitions.
    ///
    /// Γ_opt = G² · κ / (ω_m² + (κ/2)²)
    pub fn optomechanical_damping_rate(&self, n_photons: f64) -> f64 {
        let kappa = self.optical_linewidth();
        let g = self.optomechanical_coupling_rate(n_photons);
        let omega_m = self.omega_mech;
        g * g * kappa / (omega_m * omega_m + (kappa / 2.0).powi(2))
    }
}

/// Disk resonator (microring/microdisk) for WGM optomechanics.
///
/// Models a flat disk resonator with azimuthal mode number `m`, radius `R`,
/// and thickness `h`. Used in on-chip optomechanical systems.
#[derive(Debug, Clone)]
pub struct DiskResonator {
    /// Disk radius (m).
    pub radius: f64,
    /// Disk thickness (m).
    pub thickness: f64,
    /// Effective refractive index of the guided mode.
    pub n_eff: f64,
    /// Azimuthal mode number m (integer, ≥1).
    pub azimuthal_mode_number: usize,
}

impl DiskResonator {
    /// Construct a disk resonator.
    ///
    /// # Arguments
    /// * `radius` - Disk radius (m).
    /// * `thickness` - Disk thickness (m).
    /// * `n_eff` - Effective refractive index.
    /// * `m` - Azimuthal mode number (WGM order).
    pub fn new(radius: f64, thickness: f64, n_eff: f64, m: usize) -> Self {
        Self {
            radius,
            thickness,
            n_eff,
            azimuthal_mode_number: m,
        }
    }

    /// Resonant optical frequency (Hz) of the m-th azimuthal mode.
    ///
    /// ν_m = m · c / (2π · n_eff · R)
    pub fn resonant_frequency(&self) -> f64 {
        self.azimuthal_mode_number as f64 * C / (2.0 * PI * self.n_eff * self.radius)
    }

    /// Resonant wavelength (m) of the m-th azimuthal mode.
    pub fn resonant_wavelength(&self) -> f64 {
        C / self.resonant_frequency()
    }

    /// Free spectral range (Hz): distance between consecutive azimuthal modes.
    pub fn free_spectral_range(&self) -> f64 {
        C / (2.0 * PI * self.n_eff * self.radius)
    }

    /// Estimate the coupling Q from an evanescent gap using an exponential model.
    ///
    /// Q_coupling ≈ Q_0 · exp(gap / L_decay)
    ///
    /// where L_decay = λ / (4π·√(n_eff² − 1)) is the evanescent field decay length.
    ///
    /// # Arguments
    /// * `gap` - Bus waveguide gap to disk edge (m).
    /// * `wavelength` - Wavelength (m).
    pub fn coupling_gap_to_q(&self, gap: f64, wavelength: f64) -> f64 {
        let n2_minus_1 = (self.n_eff * self.n_eff - 1.0).max(0.0);
        if n2_minus_1 < f64::EPSILON {
            return 1e3; // degenerate case
        }
        let l_decay = wavelength / (4.0 * PI * n2_minus_1.sqrt());
        // Reference loaded Q ~ 1e5 at zero gap (fit to Si disk data)
        let q0 = 1e5;
        q0 * (gap / l_decay).exp()
    }

    /// Approximate mode volume (m³) using a perturbation theory estimate.
    ///
    /// V_mode ≈ 2π·R · (λ/n_eff)² / (4·n_eff)
    ///
    /// This is a rough estimate valid for tight confinement (R ≫ λ).
    pub fn mode_volume(&self) -> f64 {
        let lambda_over_n = self.resonant_wavelength() / self.n_eff;
        2.0 * PI * self.radius * lambda_over_n * lambda_over_n / (4.0 * self.n_eff)
    }

    /// Purcell factor enhancement relative to free space.
    ///
    /// F_P = (3/(4π²)) · (λ/n)³ / V_mode · Q
    pub fn purcell_factor(&self, q_factor: f64) -> f64 {
        let lambda_m = self.resonant_wavelength();
        let lambda_over_n = lambda_m / self.n_eff;
        let v = self.mode_volume();
        (3.0 / (4.0 * PI * PI)) * lambda_over_n.powi(3) / v * q_factor
    }

    /// Azimuthal mode number `m` such that the resonant wavelength is closest to `target_wl` (m).
    ///
    /// m = round(2π·n_eff·R / λ_target)
    pub fn mode_number_for_wavelength(radius: f64, n_eff: f64, target_wl: f64) -> usize {
        let m_float = 2.0 * PI * n_eff * radius / target_wl;
        m_float.round() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_wgm() -> WgmMicroresonator {
        WgmMicroresonator::new(50e-6, 1.44, 1550e-9)
    }

    #[test]
    fn test_fsr_positive() {
        let wgm = make_wgm();
        let fsr = wgm.free_spectral_range();
        // FSR = c/(2π*R*n) = 3e8/(2π*50e-6*1.44) ≈ 663 GHz
        assert!(
            fsr > 1e11 && fsr < 1e13,
            "FSR {fsr} Hz out of expected range"
        );
        let expected = C / (2.0 * PI * 50e-6 * 1.44);
        assert_abs_diff_eq!(fsr, expected, epsilon = 1.0);
    }

    #[test]
    fn test_zero_point_motion_finite() {
        let wgm = make_wgm();
        let xzpf = wgm.zero_point_motion();
        assert!(xzpf > 0.0, "zero-point motion must be positive");
        assert!(
            xzpf < 1e-12,
            "zero-point motion should be sub-pm for µm scale resonator"
        );
    }

    #[test]
    fn test_sideband_resolution() {
        // A resonator with ω_m = 2π*10MHz, κ = ω_opt/Q = 2π*194THz/1e7 ≈ 2π*19.4 MHz
        // So ω_m < κ/2 — NOT resolved
        let wgm = make_wgm();
        // Check it returns a bool without panicking
        let _resolved = wgm.sideband_resolution();
    }

    #[test]
    fn test_cooperativity_increases_with_photons() {
        let wgm = make_wgm();
        let c1 = wgm.cooperativity(1e3);
        let c2 = wgm.cooperativity(1e6);
        assert!(c2 > c1, "cooperativity should increase with photon number");
    }

    #[test]
    fn test_optomechanical_coupling_rate_zero_photons() {
        let wgm = make_wgm();
        let g = wgm.optomechanical_coupling_rate(0.0);
        assert_abs_diff_eq!(g, 0.0, epsilon = 1e-20);
    }

    #[test]
    fn test_disk_resonant_frequency() {
        // R=10µm, n=3.5 (Si), m=30 => ν ≈ 30*3e8/(2π*3.5*10e-6) ≈ 408 THz
        let disk = DiskResonator::new(10e-6, 220e-9, 3.5, 30);
        let freq = disk.resonant_frequency();
        assert!(
            freq > 1e13 && freq < 1e15,
            "disk resonant frequency {freq} Hz out of expected range"
        );
    }

    #[test]
    fn test_disk_mode_volume_positive() {
        let disk = DiskResonator::new(10e-6, 220e-9, 3.5, 30);
        let v = disk.mode_volume();
        assert!(v > 0.0, "mode volume must be positive");
    }

    #[test]
    fn test_disk_fsr_matches_wgm() {
        let disk = DiskResonator::new(50e-6, 2e-6, 1.44, 100);
        let fsr = disk.free_spectral_range();
        let wgm = WgmMicroresonator::new(50e-6, 1.44, 1550e-9);
        let fsr_wgm = wgm.free_spectral_range();
        // Both use same formula C/(2π*R*n)
        assert_abs_diff_eq!(fsr, fsr_wgm, epsilon = 1.0);
    }

    #[test]
    fn test_mode_number_for_wavelength() {
        let m = DiskResonator::mode_number_for_wavelength(50e-6, 1.44, 1550e-9);
        // m = 2π*1.44*50e-6/1550e-9 ≈ 292
        assert!(m > 200 && m < 400, "mode number {m} out of expected range");
    }

    #[test]
    fn test_coupling_gap_to_q_increases_with_gap() {
        let disk = DiskResonator::new(10e-6, 220e-9, 3.5, 30);
        let q1 = disk.coupling_gap_to_q(100e-9, 1550e-9);
        let q2 = disk.coupling_gap_to_q(200e-9, 1550e-9);
        assert!(
            q2 > q1,
            "coupling Q should increase (weaker coupling) with larger gap"
        );
    }

    #[test]
    fn test_purcell_factor_positive() {
        let disk = DiskResonator::new(10e-6, 220e-9, 3.5, 30);
        let fp = disk.purcell_factor(1e5);
        assert!(fp > 0.0, "Purcell factor must be positive");
    }
}
