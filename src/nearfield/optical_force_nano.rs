//! Optical forces on nanoparticles and near-field trapping
//!
//! Covers:
//! - Gradient and scattering forces on Rayleigh particles (r ≪ λ)
//! - Near-field enhanced optical forces near plasmonic nanostructures
//! - Nanoparticle tracking analysis (NTA) — Stokes-Einstein diffusion
//!
//! Physical basis:
//!   Gradient force:   F_grad = (1/4) Re(α) ∇|E|²
//!   Scattering force: F_scat = n σ_ext I / c
//!   Clausius-Mossotti polarizability: α = 4πε₀ a³ (n²−1)/(n²+2)
//!
//! SI units throughout.  Energies in J, forces in N, lengths in m.

use num_complex::Complex64;
use std::f64::consts::PI;

// ─── Physical constants ───────────────────────────────────────────────────────

const C0: f64 = 2.997_924_58e8; // m/s
const EPS0: f64 = 8.854_187_817e-12; // F/m
const K_B: f64 = 1.380_649e-23; // J/K

// ─── NanoparticleForce ───────────────────────────────────────────────────────

/// Optical force analysis on a spherical nanoparticle in the Rayleigh regime.
///
/// Valid when the particle radius a satisfies a ≪ λ (typically a < λ/20).
/// The particle is characterised by its complex refractive index n_p in a
/// medium of real index n_m.
///
/// The Clausius-Mossotti polarizability (including radiation reaction):
///
///   α = α₀ / (1 − i k³ α₀ / 6π)
///
/// where α₀ = 4πε₀ a³ (m²−1)/(m²+2) and m = n_p/n_m.
#[derive(Debug, Clone)]
pub struct NanoparticleForce {
    /// Particle radius in metres
    pub radius: f64,
    /// Complex refractive index of the particle
    pub n_particle: Complex64,
    /// Real refractive index of the surrounding medium
    pub n_medium: f64,
    /// Wavelength in metres (free-space)
    pub wavelength: f64,
    /// Beam power in watts (used for force calculations)
    pub power: f64,
}

impl NanoparticleForce {
    /// Construct a new nanoparticle force analyser.
    ///
    /// # Arguments
    /// * `radius`     - particle radius in metres
    /// * `n_particle` - complex refractive index of particle
    /// * `n_medium`   - real refractive index of surrounding medium
    /// * `wavelength` - free-space wavelength in metres
    pub fn new(radius: f64, n_particle: Complex64, n_medium: f64, wavelength: f64) -> Self {
        Self {
            radius,
            n_particle,
            n_medium,
            wavelength,
            power: 1.0e-3, // 1 mW default
        }
    }

    /// Relative refractive index m = n_p / n_m (complex)
    fn relative_index(&self) -> Complex64 {
        self.n_particle / Complex64::new(self.n_medium, 0.0)
    }

    /// Static Clausius-Mossotti polarizability α₀ in C·m / (V/m) = F·m².
    ///
    /// α₀ = 4π ε₀ a³ · (m²−1) / (m²+2)
    fn polarizability_static(&self) -> Complex64 {
        let a = self.radius;
        let m = self.relative_index();
        let m2 = m * m;
        let cm = (m2 - Complex64::new(1.0, 0.0)) / (m2 + Complex64::new(2.0, 0.0));
        Complex64::new(4.0 * PI * EPS0 * a * a * a, 0.0) * cm
    }

    /// Full polarizability with radiation-reaction correction (Draine 1988).
    ///
    /// α = α₀ (1 − i k³ α₀ / (6π ε₀))⁻¹
    ///
    /// This ensures optical theorem consistency: σ_ext = k Im(α) / ε₀.
    pub fn polarizability(&self) -> Complex64 {
        let k = 2.0 * PI * self.n_medium / self.wavelength;
        let alpha0 = self.polarizability_static();
        let k3 = k * k * k;
        let rr = Complex64::new(0.0, -k3 / (6.0 * PI * EPS0));
        let correction = Complex64::new(1.0, 0.0) - rr * alpha0;
        if correction.norm() < f64::EPSILON {
            return alpha0;
        }
        alpha0 / correction
    }

    /// Gradient force per unit intensity gradient in N/(W/m³).
    ///
    /// F_grad = (1/2) Re(α) ∇|E|² / |E|²  · (|E|² / (2n c ε₀))
    ///
    /// Returns the coefficient such that F_grad = coefficient × ∇I.
    pub fn gradient_force_per_intensity_gradient(&self) -> f64 {
        let alpha = self.polarizability();
        // F_grad / (∇I) = Re(α) / (2 n c ε₀)  [N·m²/W]
        alpha.re / (2.0 * self.n_medium * C0 * EPS0)
    }

    /// Scattering cross-section in m² (Rayleigh limit).
    ///
    /// σ_scat = (128π⁵/3) · (a/λ)⁴ · |α_cm|² / (4πε₀)²
    ///
    /// where α_cm is the Clausius-Mossotti factor.
    pub fn scattering_cross_section(&self) -> f64 {
        let k = 2.0 * PI * self.n_medium / self.wavelength;
        let alpha = self.polarizability();
        let k4 = k * k * k * k;
        // σ_scat = k⁴ |α|² / (6π ε₀²)
        k4 * alpha.norm_sqr() / (6.0 * PI * EPS0 * EPS0)
    }

    /// Absorption cross-section in m².
    ///
    /// σ_abs = k Im(α) / ε₀  (from optical theorem)
    pub fn absorption_cross_section(&self) -> f64 {
        let k = 2.0 * PI * self.n_medium / self.wavelength;
        let alpha = self.polarizability();
        (k * alpha.im / EPS0).max(0.0)
    }

    /// Extinction cross-section: σ_ext = σ_abs + σ_scat (m²).
    pub fn extinction_cross_section(&self) -> f64 {
        self.absorption_cross_section() + self.scattering_cross_section()
    }

    /// Scattering (radiation pressure) force in N for a given intensity.
    ///
    /// F_scat = n_m σ_ext I / c
    pub fn scattering_force(&self, intensity_w_per_m2: f64) -> f64 {
        self.n_medium * self.extinction_cross_section() * intensity_w_per_m2 / C0
    }

    /// Trapping stiffness of a Gaussian beam optical tweezer (N/m).
    ///
    /// For a particle displaced laterally by δ from the beam axis:
    ///   F_trap ≈ −k_trap δ
    ///
    /// k_trap ≈ (1/2) Re(α) I₀ / (n_m c ε₀ w₀²)
    ///
    /// # Arguments
    /// * `beam_waist`      - 1/e² radius of the Gaussian beam (m)
    /// * `peak_intensity`  - on-axis peak intensity (W/m²)
    pub fn trap_stiffness_n_per_m(&self, beam_waist: f64, peak_intensity: f64) -> f64 {
        let alpha = self.polarizability();
        if beam_waist < f64::EPSILON {
            return 0.0;
        }
        // k_trap = Re(α) I₀ / (2 n_m c ε₀ w₀²)
        alpha.re * peak_intensity / (2.0 * self.n_medium * C0 * EPS0 * beam_waist * beam_waist)
    }

    /// Escape force from the optical trap in picoNewtons.
    ///
    /// The maximum restoring force before the particle escapes the trap:
    ///   F_esc ≈ 0.43 k_trap w₀  (Gaussian beam analytic result)
    pub fn escape_force_pn(&self, beam_waist: f64, peak_intensity: f64) -> f64 {
        let k_trap = self.trap_stiffness_n_per_m(beam_waist, peak_intensity);
        let f_esc_n = 0.43 * k_trap * beam_waist;
        f_esc_n * 1.0e12 // N → pN
    }
}

// ─── NearFieldForce ──────────────────────────────────────────────────────────

/// Optical force on a nanoparticle near a plasmonic nanostructure.
///
/// The local field enhancement dramatically amplifies both gradient and
/// scattering forces.  The effective intensity gradient scales as:
///
///   ∇I_local ≈ g² I_inc / d_char
///
/// where g = |E_local/E_inc| is the field enhancement and d_char is the
/// characteristic gradient length of the near-field.
#[derive(Debug, Clone)]
pub struct NearFieldForce {
    /// Near-field enhancement factor |E_local| / |E_inc|
    pub field_enhancement: f64,
    /// Characteristic near-field gradient length in metres
    pub gradient_length: f64,
    /// Free-space wavelength in metres
    pub wavelength: f64,
}

impl NearFieldForce {
    /// Construct a near-field force calculator.
    ///
    /// # Arguments
    /// * `enhancement`          - |E_local/E_inc| at the hotspot
    /// * `gradient_length_nm`   - characteristic field decay length in nm
    /// * `wavelength`           - free-space wavelength in metres
    pub fn new(enhancement: f64, gradient_length_nm: f64, wavelength: f64) -> Self {
        Self {
            field_enhancement: enhancement,
            gradient_length: gradient_length_nm * 1.0e-9,
            wavelength,
        }
    }

    /// Enhanced gradient force on a Rayleigh nanoparticle (N).
    ///
    /// The local intensity gradient is amplified by g² relative to the
    /// far-field, and the gradient length is the hot-spot decay length:
    ///
    ///   ∇I_local ≈ g² I_inc / gradient_length
    ///
    /// where I_inc = power / (π w₀²) for a focused Gaussian beam.
    ///
    /// # Arguments
    /// * `particle` - nanoparticle force properties
    /// * `power`    - incident laser power in W
    pub fn enhanced_gradient_force(&self, particle: &NanoparticleForce, power: f64) -> f64 {
        // Far-field intensity estimate: I_inc ≈ power / λ² (diffraction limited focus)
        let i_inc = power / (self.wavelength * self.wavelength);
        // Local gradient: ∇I_local ≈ g² I_inc / d_char
        let grad_i_local =
            self.field_enhancement * self.field_enhancement * i_inc / self.gradient_length;
        // Gradient force coefficient
        let coeff = particle.gradient_force_per_intensity_gradient();
        coeff * grad_i_local
    }

    /// Optical binding force between two particles trapped near the hotspot (N).
    ///
    /// Two trapped nanoparticles interact via the scattered fields.  The
    /// binding force oscillates with particle separation as the standing wave:
    ///
    ///   F_bind(d) ≈ F₀ · sin(2π n_eff d / λ) / (n_eff d / λ)
    ///
    /// where F₀ ~ Re(α)² ω / (c² r⁶) is the coupling strength.
    ///
    /// Returns approximate binding force magnitude in N.
    pub fn optical_binding_force(&self, separation: f64, particle_radius: f64) -> f64 {
        if separation < f64::EPSILON {
            return 0.0;
        }
        let k = 2.0 * PI / self.wavelength;
        let kd = k * separation;
        // Dipole-dipole interaction strength (rough proportionality)
        let r6 = (particle_radius * 2.0).powi(6);
        let alpha0 = EPS0 * particle_radius.powi(3); // rough scale
        let f0 = alpha0 * alpha0 * (2.0 * PI * C0 / self.wavelength) / (EPS0 * EPS0 * r6 * C0 * C0);
        // Oscillatory component
        f0 * kd.sin() / kd
    }
}

// ─── NtaSimulator ────────────────────────────────────────────────────────────

/// Nanoparticle tracking analysis (NTA) simulator.
///
/// NTA measures Brownian motion of nanoparticles in suspension by tracking
/// individual particle trajectories under laser illumination.  The diffusion
/// coefficient is extracted from the mean-squared displacement (MSD), and the
/// particle size is determined via the Stokes-Einstein relation.
///
/// Key relations:
///   D = k_B T / (6π η r)    [Stokes-Einstein]
///   ⟨r²⟩ = 4 D t            [2D MSD from particle tracking]
///   v_s = 2r²(ρ_p−ρ_m)g/(9η) [Stokes sedimentation velocity]
#[derive(Debug, Clone)]
pub struct NtaSimulator {
    /// Particle radius in nm
    pub particle_radius_nm: f64,
    /// Dynamic viscosity of the medium in Pa·s (water at 25°C: 0.000_890)
    pub medium_viscosity: f64,
    /// Temperature in Kelvin
    pub temperature_k: f64,
}

impl NtaSimulator {
    /// Construct a simulator for particles in water at 25°C.
    ///
    /// # Arguments
    /// * `radius_nm` - particle radius in nm
    pub fn new(radius_nm: f64) -> Self {
        Self {
            particle_radius_nm: radius_nm,
            medium_viscosity: 8.90e-4, // water at 25°C [Pa·s]
            temperature_k: 298.15,     // 25°C
        }
    }

    /// Stokes-Einstein diffusion coefficient in m²/s.
    ///
    /// D = k_B T / (6π η r)
    pub fn diffusion_coefficient_m2_per_s(&self) -> f64 {
        let r = self.particle_radius_nm * 1.0e-9;
        let eta = self.medium_viscosity;
        let t = self.temperature_k;
        K_B * t / (6.0 * PI * eta * r)
    }

    /// Mean-squared displacement in m² at time t (2D projection).
    ///
    /// ⟨r²⟩ = 4 D t  (2D random walk)
    pub fn msd_at_time(&self, time_s: f64) -> f64 {
        4.0 * self.diffusion_coefficient_m2_per_s() * time_s
    }

    /// Infer particle radius from measured diffusion coefficient.
    ///
    /// r = k_B T / (6π η D)
    ///
    /// # Arguments
    /// * `d_m2_s`    - measured diffusion coefficient in m²/s
    /// * `viscosity` - medium dynamic viscosity in Pa·s
    /// * `temp_k`    - medium temperature in K
    pub fn radius_from_diffusion(d_m2_s: f64, viscosity: f64, temp_k: f64) -> f64 {
        if d_m2_s < f64::EPSILON {
            return 0.0;
        }
        K_B * temp_k / (6.0 * PI * viscosity * d_m2_s)
    }

    /// Stokes sedimentation velocity in nm/s.
    ///
    /// v_s = 2 r² (ρ_p − ρ_m) g / (9 η)
    ///
    /// # Arguments
    /// * `particle_density` - particle mass density in kg/m³
    /// * `medium_density`   - medium mass density in kg/m³
    pub fn sedimentation_velocity_nm_per_s(
        &self,
        particle_density: f64,
        medium_density: f64,
    ) -> f64 {
        const G: f64 = 9.80665; // gravitational acceleration m/s²
        let r = self.particle_radius_nm * 1.0e-9;
        let eta = self.medium_viscosity;
        let delta_rho = particle_density - medium_density;
        let v_m_per_s = 2.0 * r * r * delta_rho * G / (9.0 * eta);
        v_m_per_s * 1.0e9 // m/s → nm/s
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn gold_particle_532nm() -> NanoparticleForce {
        // 20 nm gold particle at 532 nm in water (n=1.33)
        // Gold optical constants at 532 nm: n ≈ 0.47 + 2.40i (Palik)
        NanoparticleForce::new(20.0e-9, Complex64::new(0.47, 2.40), 1.33, 532.0e-9)
    }

    fn silica_particle_1064nm() -> NanoparticleForce {
        // 100 nm silica particle at 1064 nm in water
        // Silica: n ≈ 1.45 (real, low absorption in NIR)
        NanoparticleForce::new(100.0e-9, Complex64::new(1.45, 0.0), 1.33, 1064.0e-9)
    }

    // ── Polarizability ────────────────────────────────────────────────────────

    #[test]
    fn test_polarizability_nonzero() {
        let p = gold_particle_532nm();
        let alpha = p.polarizability();
        assert!(
            alpha.norm() > 0.0,
            "Polarizability must be nonzero: {:?}",
            alpha
        );
    }

    #[test]
    fn test_polarizability_static_scales_as_r3() {
        // Particle 1: r = 10 nm; Particle 2: r = 20 nm → ratio = 8
        let p1 = NanoparticleForce::new(10.0e-9, Complex64::new(1.45, 0.0), 1.0, 1064.0e-9);
        let p2 = NanoparticleForce::new(20.0e-9, Complex64::new(1.45, 0.0), 1.0, 1064.0e-9);
        let a1 = p1.polarizability_static().norm();
        let a2 = p2.polarizability_static().norm();
        assert_abs_diff_eq!(a2 / a1, 8.0, epsilon = 0.01);
    }

    // ── Cross-sections ────────────────────────────────────────────────────────

    #[test]
    fn test_extinction_ge_absorption() {
        let p = gold_particle_532nm();
        let sigma_ext = p.extinction_cross_section();
        let sigma_abs = p.absorption_cross_section();
        assert!(
            sigma_ext >= sigma_abs,
            "σ_ext ({sigma_ext:.3e}) must be ≥ σ_abs ({sigma_abs:.3e})"
        );
    }

    #[test]
    fn test_scattering_cross_section_positive() {
        let p = gold_particle_532nm();
        let sigma_scat = p.scattering_cross_section();
        assert!(sigma_scat > 0.0, "σ_scat must be positive: {sigma_scat}");
    }

    #[test]
    fn test_absorption_cross_section_positive_for_lossy() {
        let p = gold_particle_532nm();
        let sigma_abs = p.absorption_cross_section();
        assert!(
            sigma_abs > 0.0,
            "σ_abs must be positive for gold: {sigma_abs}"
        );
    }

    // ── Forces ────────────────────────────────────────────────────────────────

    #[test]
    fn test_gradient_force_positive_for_dielectric() {
        // For a dielectric (n_p > n_m), Re(α) > 0 → positive gradient force coefficient
        let p = silica_particle_1064nm();
        let coeff = p.gradient_force_per_intensity_gradient();
        assert!(
            coeff > 0.0,
            "Gradient force coeff should be positive for dielectric: {coeff}"
        );
    }

    #[test]
    fn test_scattering_force_positive() {
        let p = gold_particle_532nm();
        let f = p.scattering_force(1.0e9); // 1 GW/m² (pulsed)
        assert!(f > 0.0, "Scattering force must be positive: {f}");
    }

    #[test]
    fn test_trap_stiffness_positive_dielectric() {
        let p = silica_particle_1064nm();
        let k_trap = p.trap_stiffness_n_per_m(1.0e-6, 1.0e10);
        assert!(k_trap > 0.0, "Trap stiffness must be positive: {k_trap}");
    }

    #[test]
    fn test_escape_force_pn_positive() {
        let p = silica_particle_1064nm();
        let f_esc = p.escape_force_pn(1.0e-6, 1.0e10);
        assert!(f_esc > 0.0, "Escape force must be positive: {f_esc}");
    }

    // ── NearFieldForce ────────────────────────────────────────────────────────

    #[test]
    fn test_near_field_force_enhanced_over_far_field() {
        let near = NearFieldForce::new(100.0, 5.0, 800.0e-9);
        let far = NearFieldForce::new(1.0, 5.0, 800.0e-9);
        let p = silica_particle_1064nm();
        let f_near = near.enhanced_gradient_force(&p, 1.0e-3);
        let f_far = far.enhanced_gradient_force(&p, 1.0e-3);
        // Enhancement should be proportional to g² = 100² = 10000
        assert!(
            f_near > f_far * 100.0,
            "Near-field force ({f_near:.3e}) should vastly exceed far-field ({f_far:.3e})"
        );
    }

    // ── NtaSimulator ─────────────────────────────────────────────────────────

    #[test]
    fn test_diffusion_coefficient_100nm() {
        // 100 nm particle in water at 25°C → D ≈ 4.37e-12 m²/s (literature)
        let nta = NtaSimulator::new(100.0);
        let d = nta.diffusion_coefficient_m2_per_s();
        assert!(
            d > 1.0e-12 && d < 1.0e-10,
            "D for 100 nm particle should be ~4e-12 m²/s: {d:.3e}"
        );
    }

    #[test]
    fn test_diffusion_scales_inversely_with_radius() {
        let nta1 = NtaSimulator::new(50.0);
        let nta2 = NtaSimulator::new(100.0);
        let d1 = nta1.diffusion_coefficient_m2_per_s();
        let d2 = nta2.diffusion_coefficient_m2_per_s();
        // D ∝ 1/r → d1 / d2 = r2 / r1 = 2.0
        assert_abs_diff_eq!(d1 / d2, 2.0, epsilon = 1.0e-10);
    }

    #[test]
    fn test_msd_linear_in_time() {
        let nta = NtaSimulator::new(100.0);
        let d = nta.diffusion_coefficient_m2_per_s();
        let msd_1s = nta.msd_at_time(1.0);
        let msd_2s = nta.msd_at_time(2.0);
        assert_abs_diff_eq!(msd_1s, 4.0 * d, epsilon = 1.0e-20);
        assert_abs_diff_eq!(msd_2s, 8.0 * d, epsilon = 1.0e-20);
    }

    #[test]
    fn test_radius_from_diffusion_roundtrip() {
        let nta = NtaSimulator::new(75.0);
        let d = nta.diffusion_coefficient_m2_per_s();
        let r_recovered =
            NtaSimulator::radius_from_diffusion(d, nta.medium_viscosity, nta.temperature_k);
        // Should recover 75 nm (within floating-point precision)
        assert_abs_diff_eq!(r_recovered * 1.0e9, 75.0, epsilon = 1.0e-6);
    }

    #[test]
    fn test_sedimentation_positive_for_denser_particle() {
        let nta = NtaSimulator::new(100.0);
        // Gold: ρ = 19300 kg/m³, water: ρ = 1000 kg/m³
        let v = nta.sedimentation_velocity_nm_per_s(19300.0, 1000.0);
        assert!(
            v > 0.0,
            "Sedimentation velocity should be positive for sinking particle: {v}"
        );
    }

    #[test]
    fn test_sedimentation_zero_for_matched_density() {
        let nta = NtaSimulator::new(100.0);
        let v = nta.sedimentation_velocity_nm_per_s(1000.0, 1000.0);
        assert_abs_diff_eq!(v, 0.0, epsilon = 1.0e-30);
    }
}
