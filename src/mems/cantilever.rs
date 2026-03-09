//! Euler-Bernoulli cantilever beam model for MEMS optical sensors.
//!
//! Implements mechanical and optical-force physics for micro-cantilever beams
//! used in atomic force microscopy (AFM), optical pressure sensing, and
//! photothermal spectroscopy.
//!
//! # Reference
//! Sader, J.E. (1998). Frequency response of cantilever beams immersed in
//! viscous fluids with applications to the atomic force microscope.
//! *Journal of Applied Physics*, 84(1), 64–76.

use std::f64::consts::PI;

use crate::units::conversion::BOLTZMANN;

/// First eigenvalue of the Euler-Bernoulli beam equation: (β₁L) = 1.875104069.
const BETA_1_L: f64 = 1.875_104_069;

/// Speed of light in vacuum (m/s).
const C: f64 = 299_792_458.0;

/// Material properties for common MEMS cantilever materials.
#[derive(Debug, Clone, PartialEq)]
pub enum CantileverMaterial {
    /// Single-crystal silicon (most common MEMS material).
    Silicon,
    /// Silicon nitride (low-stress, high Q factor).
    SiliconNitride,
    /// Gold (for plasmonic / conductive cantilevers).
    Gold,
    /// User-defined material with explicit elastic/density parameters.
    Custom {
        /// Young's modulus in Pa.
        youngs_modulus: f64,
        /// Mass density in kg/m³.
        density: f64,
    },
}

impl CantileverMaterial {
    /// Young's modulus in Pa.
    pub fn youngs_modulus(&self) -> f64 {
        match self {
            CantileverMaterial::Silicon => 130.0e9,
            CantileverMaterial::SiliconNitride => 250.0e9,
            CantileverMaterial::Gold => 79.0e9,
            CantileverMaterial::Custom { youngs_modulus, .. } => *youngs_modulus,
        }
    }

    /// Mass density in kg/m³.
    pub fn density(&self) -> f64 {
        match self {
            CantileverMaterial::Silicon => 2_330.0,
            CantileverMaterial::SiliconNitride => 3_100.0,
            CantileverMaterial::Gold => 19_300.0,
            CantileverMaterial::Custom { density, .. } => *density,
        }
    }
}

/// Euler-Bernoulli cantilever beam for MEMS optical sensors.
///
/// Models a rectangular cross-section beam clamped at one end. Provides
/// resonant frequency, spring constant, deflection under various forces,
/// and thermal noise floor.
///
/// # Example
/// ```
/// use oxiphoton::mems::cantilever::{OpticalCantilever, CantileverMaterial};
/// let c = OpticalCantilever::new(100e-6, 10e-6, 1e-6, CantileverMaterial::Silicon);
/// let f0 = c.resonant_frequency();
/// assert!(f0 > 1e3 && f0 < 1e7, "resonant frequency should be kHz-MHz range");
/// ```
#[derive(Debug, Clone)]
pub struct OpticalCantilever {
    /// Beam length (m).
    pub length: f64,
    /// Beam width (m).
    pub width: f64,
    /// Beam thickness (m).
    pub thickness: f64,
    /// Young's modulus (Pa).
    pub youngs_modulus: f64,
    /// Mass density (kg/m³).
    pub density: f64,
    /// Mechanical quality factor (dimensionless).
    pub q_factor: f64,
}

impl OpticalCantilever {
    /// Create a new cantilever from geometry and material.
    ///
    /// The quality factor defaults to 1000 (typical MEMS in vacuum).
    pub fn new(length: f64, width: f64, thickness: f64, material: CantileverMaterial) -> Self {
        Self {
            length,
            width,
            thickness,
            youngs_modulus: material.youngs_modulus(),
            density: material.density(),
            q_factor: 1_000.0,
        }
    }

    /// Second moment of area (m⁴) for a rectangular cross section.
    ///
    /// I = w·t³/12
    pub fn moment_of_inertia(&self) -> f64 {
        self.width * self.thickness.powi(3) / 12.0
    }

    /// Cross-sectional area (m²).
    fn cross_section_area(&self) -> f64 {
        self.width * self.thickness
    }

    /// Effective mass of the fundamental mode (m_eff = 0.2427·ρ·A·L).
    ///
    /// The prefactor 0.2427 comes from the mode-shape integral of the
    /// Euler-Bernoulli beam equation.
    pub fn effective_mass(&self) -> f64 {
        0.2427 * self.density * self.cross_section_area() * self.length
    }

    /// Fundamental resonant frequency (Hz) in vacuum.
    ///
    /// Uses the exact first eigenvalue of the clamped-free beam:
    ///
    /// f₁ = (β₁L)² / (2π·L²) · √(E·I / (ρ·A))
    ///
    /// where β₁L = 1.875104069 for the fundamental mode.
    pub fn resonant_frequency(&self) -> f64 {
        let ei = self.youngs_modulus * self.moment_of_inertia();
        let rho_a = self.density * self.cross_section_area();
        let l2 = self.length * self.length;
        BETA_1_L * BETA_1_L / (2.0 * PI * l2) * (ei / rho_a).sqrt()
    }

    /// Spring constant (N/m) of the cantilever tip.
    ///
    /// k = 3·E·I / L³
    pub fn spring_constant(&self) -> f64 {
        3.0 * self.youngs_modulus * self.moment_of_inertia() / self.length.powi(3)
    }

    /// Tip deflection (m) under a static point force at the tip.
    ///
    /// δ = F·L³ / (3·E·I)
    pub fn deflection_from_force(&self, force: f64) -> f64 {
        force / self.spring_constant()
    }

    /// Tip deflection (m) due to radiation pressure from a reflected optical beam.
    ///
    /// The radiation pressure force is F = 2·P·(1+R)/c for a partially-reflecting
    /// surface (the factor of 2 for reflection). For simplicity, force is evaluated
    /// at the beam tip.
    ///
    /// # Arguments
    /// * `power_w` - Incident optical power in Watts.
    /// * `reflectivity` - Power reflectivity (0..1).
    pub fn optical_force_deflection(&self, power_w: f64, reflectivity: f64) -> f64 {
        let r = reflectivity.clamp(0.0, 1.0);
        let force = power_w * (1.0 + r) / C;
        self.deflection_from_force(force)
    }

    /// Thermo-mechanical displacement noise spectral density at resonance (m/√Hz).
    ///
    /// Sx(f₀) = √(4·k_B·T·k / (ω₀·Q))
    ///
    /// This is the Langevin force noise divided by (k·Q·ω₀), evaluated at ω₀.
    ///
    /// # Arguments
    /// * `temperature_k` - Temperature in Kelvin.
    pub fn thermo_mechanical_noise(&self, temperature_k: f64) -> f64 {
        let omega0 = 2.0 * PI * self.resonant_frequency();
        let k = self.spring_constant();
        (4.0 * BOLTZMANN * temperature_k * k / (omega0 * self.q_factor)).sqrt()
    }

    /// Mass sensitivity of the resonant frequency (Hz/kg).
    ///
    /// dω/dm = −ω₀ / (2·m_eff)
    ///
    /// A positive return value indicates the magnitude; actual shift is negative
    /// (added mass decreases frequency).
    pub fn mass_sensitivity(&self) -> f64 {
        let omega0 = 2.0 * PI * self.resonant_frequency();
        omega0 / (2.0 * self.effective_mass())
    }

    /// Minimum detectable mass (kg) given a frequency resolution `df_hz` (Hz).
    ///
    /// δm = δω / |dω/dm| = 2π·δf / mass_sensitivity
    pub fn minimum_detectable_mass(&self, df_hz: f64) -> f64 {
        2.0 * PI * df_hz / self.mass_sensitivity()
    }

    /// Deflection sensitivity at the photodetector (m/m) for an optical lever
    /// readout with arm length `lever_arm_m`.
    ///
    /// For small angles: Δx_det = 2·(L_lever/L_beam)·δ_tip
    pub fn optical_lever_sensitivity(&self, lever_arm_m: f64) -> f64 {
        2.0 * lever_arm_m / self.length
    }

    /// Frequency response magnitude at drive frequency `f_drive` (Hz).
    ///
    /// Uses the standard driven harmonic oscillator transfer function:
    /// H = 1 / √((1 - (f/f0)²)² + (f/(f0·Q))²)
    pub fn frequency_response(&self, f_drive: f64) -> f64 {
        let f0 = self.resonant_frequency();
        let ratio = f_drive / f0;
        let denom = ((1.0 - ratio * ratio).powi(2) + (ratio / self.q_factor).powi(2)).sqrt();
        if denom < f64::EPSILON {
            self.q_factor
        } else {
            1.0 / denom
        }
    }

    /// 3-dB bandwidth of the resonance peak (Hz).
    ///
    /// Δf = f₀ / Q
    pub fn bandwidth_3db(&self) -> f64 {
        self.resonant_frequency() / self.q_factor
    }

    /// Static deflection profile along the beam length.
    ///
    /// Returns deflection (m) at position `x` (0 ≤ x ≤ L) from the clamp
    /// under a tip point load F.
    ///
    /// w(x) = F·x²·(3L - x) / (6·E·I)
    pub fn deflection_profile(&self, x: f64, force: f64) -> f64 {
        let x_clamped = x.clamp(0.0, self.length);
        let ei = self.youngs_modulus * self.moment_of_inertia();
        force * x_clamped * x_clamped * (3.0 * self.length - x_clamped) / (6.0 * ei)
    }

    /// Bending stress at position x along the neutral axis surface (Pa).
    ///
    /// σ(x) = E·y_max·d²w/dx² where y_max = t/2 and d²w/dx² = F(L-x)/(EI)
    pub fn bending_stress(&self, x: f64, force: f64) -> f64 {
        let x_clamped = x.clamp(0.0, self.length);
        let ei = self.youngs_modulus * self.moment_of_inertia();
        let y_max = self.thickness / 2.0;
        // Curvature = F(L-x)/(EI), stress = E*y*curvature
        self.youngs_modulus * y_max * force * (self.length - x_clamped) / ei
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn silicon_cantilever() -> OpticalCantilever {
        // Typical AFM cantilever: 100 µm × 10 µm × 1 µm, silicon
        OpticalCantilever::new(100e-6, 10e-6, 1e-6, CantileverMaterial::Silicon)
    }

    #[test]
    fn test_spring_constant_silicon() {
        let c = silicon_cantilever();
        // k = 3*E*I/L³
        // E=130GPa, I=10e-6*(1e-6)^3/12 = 8.333e-25 m^4
        // L^3 = (100e-6)^3 = 1e-12 m^3
        // k = 3*130e9*8.333e-25/1e-12 = 0.3250 N/m
        let k = c.spring_constant();
        assert!(
            k > 0.1 && k < 1.0,
            "spring constant should be ~0.325 N/m, got {k}"
        );
        let i = c.moment_of_inertia();
        let k_manual = 3.0 * 130e9 * i / (100e-6f64).powi(3);
        assert_abs_diff_eq!(k, k_manual, epsilon = 1e-10);
    }

    #[test]
    fn test_resonant_frequency_range() {
        let c = silicon_cantilever();
        let f0 = c.resonant_frequency();
        // A 100 µm silicon cantilever should resonate in tens-of-kHz range
        assert!(
            f0 > 10_000.0 && f0 < 1_000_000.0,
            "resonant frequency {f0} Hz out of expected range"
        );
    }

    #[test]
    fn test_deflection_from_force() {
        let c = silicon_cantilever();
        let force = 1e-9; // 1 nN
        let delta = c.deflection_from_force(force);
        // δ = F/k; k ~ 0.325 N/m => δ ~ 3.08 nm
        assert!(
            delta > 1e-10 && delta < 1e-7,
            "deflection {delta} m out of expected range"
        );
        // Round-trip: force from deflection
        let k = c.spring_constant();
        assert_abs_diff_eq!(delta * k, force, epsilon = 1e-20);
    }

    #[test]
    fn test_thermo_mechanical_noise_positive() {
        let c = silicon_cantilever();
        let noise = c.thermo_mechanical_noise(300.0); // 300 K
        assert!(noise > 0.0, "noise should be positive");
        assert!(noise < 1e-9, "noise should be sub-nm/√Hz at room temp");
    }

    #[test]
    fn test_optical_force_deflection() {
        let c = silicon_cantilever();
        // 1 mW at 100% reflectivity
        let delta = c.optical_force_deflection(1e-3, 1.0);
        assert!(delta > 0.0, "optical force deflection must be positive");
        // Force = 2P/c = 2*1e-3/3e8 ~ 6.67e-12 N
        // deflection ~ 6.67e-12 / 0.325 ~ 2e-11 m (sub-pm range)
        assert!(delta < 1e-9, "optical deflection should be sub-nm for 1 mW");
    }

    #[test]
    fn test_mass_sensitivity() {
        let c = silicon_cantilever();
        let sens = c.mass_sensitivity(); // Hz/kg
        assert!(sens > 0.0, "mass sensitivity should be positive");
        // For a ~50 kHz resonator with m_eff ~ 1e-12 kg, sens ~ 1e17 Hz/kg
        assert!(
            sens > 1e10,
            "mass sensitivity should be large for microscale cantilever"
        );
    }

    #[test]
    fn test_gold_vs_silicon_q_density() {
        // Gold is denser than silicon, so effective mass should be higher
        let si = OpticalCantilever::new(100e-6, 10e-6, 1e-6, CantileverMaterial::Silicon);
        let au = OpticalCantilever::new(100e-6, 10e-6, 1e-6, CantileverMaterial::Gold);
        assert!(
            au.effective_mass() > si.effective_mass(),
            "gold cantilever should have greater effective mass"
        );
        // And lower resonant frequency (softer + denser)
        assert!(
            au.resonant_frequency() < si.resonant_frequency(),
            "gold cantilever should have lower resonant frequency"
        );
    }

    #[test]
    fn test_deflection_profile_at_tip() {
        let c = silicon_cantilever();
        let force = 1e-9;
        let tip_profile = c.deflection_profile(c.length, force);
        let tip_direct = c.deflection_from_force(force);
        // At x=L: w(L) = F*L^2*(3L-L)/(6EI) = F*L^2*2L/(6EI) = F*L^3/(3EI)
        assert_abs_diff_eq!(tip_profile, tip_direct, epsilon = 1e-18);
    }
}
