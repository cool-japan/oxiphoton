//! Optical Tweezers Simulation
//!
//! Implements force calculations for optical tweezers using the Rayleigh approximation
//! (particle radius r << wavelength λ). This regime allows analytical expressions for
//! gradient and scattering forces.
//!
//! # Physical Background
//!
//! The gradient force arises from the interaction of the induced dipole with the
//! field gradient. For a sphere in the Rayleigh limit:
//!   F_grad = (2π n_m r³ / c) * ((m²-1)/(m²+2)) * ∇I
//!
//! where m = n_p/n_m is the relative refractive index.
//!
//! The scattering force acts along the beam propagation direction:
//!   F_scat = n_m σ_ext I / c
//!
//! Stable trapping requires |F_grad| > |F_scat| along the axial direction.

use std::f64::consts::PI;

/// Speed of light in vacuum (m/s)
const C0: f64 = 2.997_924_58e8;
/// Boltzmann constant (J/K)
const KB: f64 = 1.380_649e-23;
/// Permittivity of free space (F/m)
const EPS0: f64 = 8.854_187_817e-12;

/// Axis specification for trap stiffness calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrapAxis {
    /// Lateral (transverse, radial) direction perpendicular to beam
    Lateral,
    /// Axial direction along the beam propagation axis
    Axial,
}

/// Optical tweezers simulation using the Rayleigh approximation (r << λ).
///
/// Models a focused Gaussian beam trap for dielectric microspheres.
/// Valid when the particle radius is much smaller than the wavelength,
/// typically r < λ/10.
///
/// The trap is centered at the coordinate origin, with the beam propagating
/// along the +z axis.
#[derive(Debug, Clone)]
pub struct OpticalTweezers {
    /// Beam power at the focus (W)
    pub beam_power_w: f64,
    /// Wavelength in vacuum (nm)
    pub wavelength_nm: f64,
    /// Numerical aperture of the focusing objective
    pub numerical_aperture: f64,
    /// Refractive index of the surrounding medium
    pub medium_index: f64,
}

impl OpticalTweezers {
    /// Create a new optical tweezers configuration.
    ///
    /// # Arguments
    /// * `power_w` - Beam power (W)
    /// * `lambda_nm` - Wavelength in vacuum (nm)
    /// * `na` - Numerical aperture of the objective
    /// * `n_medium` - Refractive index of the surrounding medium
    pub fn new(power_w: f64, lambda_nm: f64, na: f64, n_medium: f64) -> Self {
        Self {
            beam_power_w: power_w,
            wavelength_nm: lambda_nm,
            numerical_aperture: na,
            medium_index: n_medium,
        }
    }

    /// Gaussian beam waist (1/e² radius) at focus: w0 ≈ λ/(π*NA) (nm).
    ///
    /// This is the Abbe diffraction limit approximation for the focused spot size.
    pub fn beam_waist_nm(&self) -> f64 {
        self.wavelength_nm / (PI * self.numerical_aperture)
    }

    /// Peak intensity at focus center: I0 = 2P/(π*w0²) (W/m²).
    ///
    /// For a Gaussian beam with total power P and waist w0.
    pub fn peak_intensity_w_per_m2(&self) -> f64 {
        let w0_m = self.beam_waist_nm() * 1.0e-9;
        2.0 * self.beam_power_w / (PI * w0_m * w0_m)
    }

    /// Rayleigh range: z_R = π*w0²*n_medium/λ (nm).
    ///
    /// The axial distance over which the beam waist increases by √2.
    pub fn rayleigh_range_nm(&self) -> f64 {
        let w0_nm = self.beam_waist_nm();
        PI * w0_nm * w0_nm * self.medium_index / self.wavelength_nm
    }

    /// Clausius-Mossotti polarizability factor K(m) = (m²-1)/(m²+2).
    ///
    /// This factor governs the strength of dielectric response.
    /// Positive for n_p > n_m (attracted to high intensity regions).
    fn clausius_mossotti(m_relative: f64) -> f64 {
        let m2 = m_relative * m_relative;
        (m2 - 1.0) / (m2 + 2.0)
    }

    /// Rayleigh scattering cross-section σ_scat (m²).
    ///
    /// σ_scat = (128π⁵r⁶)/(3λ⁴) * ((m²-1)/(m²+2))²
    fn scattering_cross_section_m2(radius_m: f64, lambda_m: f64, m_relative: f64) -> f64 {
        let k_cm = Self::clausius_mossotti(m_relative);
        let r6 = radius_m.powi(6);
        let lambda4 = lambda_m.powi(4);
        128.0 * PI.powi(5) * r6 / (3.0 * lambda4) * k_cm * k_cm
    }

    /// Intensity at position (x, y, z) relative to focus for a Gaussian beam.
    ///
    /// I(r,z) = I0 * (w0/w(z))² * exp(-2r²/w(z)²)
    /// where w(z) = w0 * sqrt(1 + (z/zR)²)
    fn intensity_at_position(&self, pos_nm: [f64; 3]) -> f64 {
        let w0_nm = self.beam_waist_nm();
        let z_r = self.rayleigh_range_nm();
        let x = pos_nm[0];
        let y = pos_nm[1];
        let z = pos_nm[2];

        let r_sq = x * x + y * y;
        let wz_sq = w0_nm * w0_nm * (1.0 + (z / z_r).powi(2));

        let i0 = self.peak_intensity_w_per_m2();
        i0 * (w0_nm * w0_nm / wz_sq) * (-2.0 * r_sq / wz_sq).exp()
    }

    /// Gradient of intensity at position (nm⁻¹ × W/m²), returned in SI (W/m³).
    ///
    /// Uses numerical differentiation with a small step size.
    fn intensity_gradient_si(&self, pos_nm: [f64; 3]) -> [f64; 3] {
        let delta_nm = self.beam_waist_nm() * 1.0e-4; // 0.01% of waist
        let delta_m = delta_nm * 1.0e-9;

        let mut grad = [0.0f64; 3];
        for axis in 0..3 {
            let mut pos_plus = pos_nm;
            let mut pos_minus = pos_nm;
            pos_plus[axis] += delta_nm;
            pos_minus[axis] -= delta_nm;
            let i_plus = self.intensity_at_position(pos_plus);
            let i_minus = self.intensity_at_position(pos_minus);
            grad[axis] = (i_plus - i_minus) / (2.0 * delta_m);
        }
        grad
    }

    /// Gradient force on a Rayleigh particle (r << λ) in units of Newtons.
    ///
    /// F_grad = (2π n_m r³ / c) * K(m) * ∇I
    ///
    /// where K(m) = (m²-1)/(m²+2) is the Clausius-Mossotti factor
    /// and m = n_p/n_m is the relative refractive index.
    ///
    /// # Arguments
    /// * `particle_radius_nm` - Particle radius (nm), must be << λ
    /// * `particle_index` - Refractive index of particle
    /// * `position_nm` - Position relative to focus [x, y, z] (nm)
    ///
    /// # Returns
    /// Gradient force vector [Fx, Fy, Fz] (N)
    pub fn gradient_force_n(
        &self,
        particle_radius_nm: f64,
        particle_index: f64,
        position_nm: [f64; 3],
    ) -> [f64; 3] {
        let r_m = particle_radius_nm * 1.0e-9;
        let m = particle_index / self.medium_index;
        let k_cm = Self::clausius_mossotti(m);

        // Prefactor: α = 4π*ε0*n_m²*r³*K(m)
        // F_grad = (α/(4ε0*n_m*c)) * ∇I = (π*n_m*r³*K(m)/c) * ∇I
        // More precisely: F = (n_m²*r³/(2c*ε0)) * α' * ∇(|E|²/2)
        // In terms of intensity: F_grad = (2π*n_m*r³/c) * K(m) * ∇I
        let prefactor = 2.0 * PI * self.medium_index * r_m.powi(3) * k_cm / C0;

        let grad_i = self.intensity_gradient_si(position_nm);
        [
            prefactor * grad_i[0],
            prefactor * grad_i[1],
            prefactor * grad_i[2],
        ]
    }

    /// Scattering force (along beam propagation +z) on a Rayleigh particle.
    ///
    /// F_scat = n_m * σ_scat * I / c
    ///
    /// # Arguments
    /// * `particle_radius_nm` - Particle radius (nm)
    /// * `particle_index` - Refractive index of particle
    /// * `intensity` - Local intensity (W/m²)
    ///
    /// # Returns
    /// Scattering force in +z direction (N)
    pub fn scattering_force_n(
        &self,
        particle_radius_nm: f64,
        particle_index: f64,
        intensity: f64,
    ) -> f64 {
        let r_m = particle_radius_nm * 1.0e-9;
        let lambda_m = self.wavelength_nm * 1.0e-9 / self.medium_index;
        let m = particle_index / self.medium_index;

        let sigma = Self::scattering_cross_section_m2(r_m, lambda_m, m);
        self.medium_index * sigma * intensity / C0
    }

    /// Total force on a Rayleigh particle at a given position.
    ///
    /// Combines gradient force (3D restoring) and scattering force (axial pushing).
    ///
    /// # Arguments
    /// * `particle_radius_nm` - Particle radius (nm)
    /// * `particle_index` - Refractive index of particle
    /// * `position_nm` - Position relative to focus [x, y, z] (nm)
    ///
    /// # Returns
    /// Total force vector [Fx, Fy, Fz] (N)
    pub fn total_force_n(
        &self,
        particle_radius_nm: f64,
        particle_index: f64,
        position_nm: [f64; 3],
    ) -> [f64; 3] {
        let f_grad = self.gradient_force_n(particle_radius_nm, particle_index, position_nm);
        let intensity = self.intensity_at_position(position_nm);
        let f_scat = self.scattering_force_n(particle_radius_nm, particle_index, intensity);

        [f_grad[0], f_grad[1], f_grad[2] + f_scat]
    }

    /// Trap stiffness along a specified axis near the trap center.
    ///
    /// k = -dF/dx ≈ F(Δx)/Δx for small displacements.
    /// Uses numerical differentiation about the center.
    ///
    /// # Arguments
    /// * `particle_radius_nm` - Particle radius (nm)
    /// * `particle_index` - Refractive index of particle
    /// * `axis` - Lateral (radial) or Axial (along beam)
    ///
    /// # Returns
    /// Trap stiffness (N/m), positive means restoring
    pub fn trap_stiffness_n_per_m(
        &self,
        particle_radius_nm: f64,
        particle_index: f64,
        axis: TrapAxis,
    ) -> f64 {
        let delta_nm = self.beam_waist_nm() * 0.001; // 0.1% of waist
        let delta_m = delta_nm * 1.0e-9;

        match axis {
            TrapAxis::Lateral => {
                // Differentiate lateral force (x-component) along x
                let pos_plus = [delta_nm, 0.0, 0.0];
                let pos_minus = [-delta_nm, 0.0, 0.0];
                let f_plus = self.gradient_force_n(particle_radius_nm, particle_index, pos_plus);
                let f_minus = self.gradient_force_n(particle_radius_nm, particle_index, pos_minus);
                // k = -dFx/dx: force is restoring so F_plus < 0, F_minus > 0
                -(f_plus[0] - f_minus[0]) / (2.0 * delta_m)
            }
            TrapAxis::Axial => {
                // Differentiate axial force along z (gradient + scattering)
                let pos_plus = [0.0, 0.0, delta_nm];
                let pos_minus = [0.0, 0.0, -delta_nm];
                let f_plus = self.total_force_n(particle_radius_nm, particle_index, pos_plus);
                let f_minus = self.total_force_n(particle_radius_nm, particle_index, pos_minus);
                -(f_plus[2] - f_minus[2]) / (2.0 * delta_m)
            }
        }
    }

    /// Optical potential depth at a given position (units of kT at 300 K).
    ///
    /// U(r) = -(2π n_m r³ / c) * K(m) * I(r)
    ///
    /// The trap depth is U(0) - U(∞) = U(0) since I(∞) = 0.
    ///
    /// # Arguments
    /// * `particle_radius_nm` - Particle radius (nm)
    /// * `particle_index` - Refractive index of particle
    ///
    /// # Returns
    /// Trap depth in units of kT (dimensionless)
    pub fn trap_depth_kt(&self, particle_radius_nm: f64, particle_index: f64) -> f64 {
        let r_m = particle_radius_nm * 1.0e-9;
        let m = particle_index / self.medium_index;
        let k_cm = Self::clausius_mossotti(m);
        let i0 = self.peak_intensity_w_per_m2();

        // U_max = prefactor * I0 (trap depth in Joules)
        let prefactor = 2.0 * PI * self.medium_index * r_m.powi(3) * k_cm / C0;
        let u_joules = prefactor.abs() * i0;

        let kt = KB * 300.0;
        u_joules / kt
    }

    /// Escape force: maximum axial force the trap can exert before the particle escapes.
    ///
    /// Estimated as the maximum axial gradient force (positive direction),
    /// which occurs at z ≈ z_R / √2.
    ///
    /// # Arguments
    /// * `particle_radius_nm` - Particle radius (nm)
    /// * `particle_index` - Refractive index of particle
    ///
    /// # Returns
    /// Escape force magnitude (N)
    pub fn escape_force_n(&self, particle_radius_nm: f64, particle_index: f64) -> f64 {
        let z_r = self.rayleigh_range_nm();
        // Maximum restoring force occurs near z ≈ z_R/sqrt(2) behind focus
        let z_test = -z_r / 2.0_f64.sqrt();
        let f = self.gradient_force_n(particle_radius_nm, particle_index, [0.0, 0.0, z_test]);
        f[2].abs()
    }

    /// Brownian motion (thermal fluctuation) amplitude: x_rms = sqrt(kT/k) (nm).
    ///
    /// From the equipartition theorem: ½k*x² = ½kT
    ///
    /// # Arguments
    /// * `trap_stiffness` - Spring constant (N/m)
    /// * `temperature_k` - Temperature (K)
    ///
    /// # Returns
    /// RMS displacement (nm)
    pub fn thermal_fluctuation_nm(&self, trap_stiffness: f64, temperature_k: f64) -> f64 {
        let x_rms_m = (KB * temperature_k / trap_stiffness.abs()).sqrt();
        x_rms_m * 1.0e9
    }

    /// Equilibrium axial position where the net axial force is zero.
    ///
    /// The scattering force pushes particles downstream (+z), while the
    /// gradient force pulls them toward focus. The equilibrium is slightly
    /// beyond the focus (positive z).
    ///
    /// Uses bisection search in range [0, 2*z_R].
    ///
    /// # Arguments
    /// * `particle_radius_nm` - Particle radius (nm)
    /// * `particle_index` - Refractive index of particle
    ///
    /// # Returns
    /// Equilibrium axial position (nm), positive = downstream from focus
    pub fn equilibrium_position_nm(&self, particle_radius_nm: f64, particle_index: f64) -> f64 {
        let z_r = self.rayleigh_range_nm();
        // The equilibrium is where gradient force (restoring) balances scattering (pushing)
        // Search in range [0, 2*z_R]
        let mut z_low = 0.0_f64;
        let mut z_high = 2.0 * z_r;

        // Force at focus: scattering > 0, gradient_z = 0 → net positive → search above
        // Force far away: both forces ~0, but scattering decays faster
        // Find zero crossing
        for _ in 0..64 {
            let z_mid = (z_low + z_high) / 2.0;
            let f = self.total_force_n(particle_radius_nm, particle_index, [0.0, 0.0, z_mid]);
            let fz = f[2];
            if fz > 0.0 {
                z_low = z_mid;
            } else {
                z_high = z_mid;
            }
        }
        (z_low + z_high) / 2.0
    }
}

/// Optical binding force between two particles in a laser field.
///
/// When multiple particles are placed in a laser field, they interact
/// through the scattered light field, creating an effective optical binding
/// potential that leads to preferred particle separations.
#[derive(Debug, Clone)]
pub struct OpticalBinding {
    pub tweezers: OpticalTweezers,
}

impl OpticalBinding {
    /// Create a new optical binding configuration.
    pub fn new(tweezers: OpticalTweezers) -> Self {
        Self { tweezers }
    }

    /// Optical binding force between two particles at separation d.
    ///
    /// In the far-field dipole approximation, the binding potential oscillates as
    /// a decaying sinusoidal function of the optical path length:
    ///   F_bind ≈ F_scale * sin(2π n_m d/λ) / (n_m d/λ)²
    ///
    /// This simplified model captures the oscillatory nature and correct period.
    ///
    /// # Arguments
    /// * `particle_radius_nm` - Particle radius (nm)
    /// * `particle_index` - Refractive index of particle
    /// * `separation_nm` - Center-to-center separation (nm)
    ///
    /// # Returns
    /// Optical binding force (N), positive = repulsive
    pub fn binding_force_n(
        &self,
        particle_radius_nm: f64,
        particle_index: f64,
        separation_nm: f64,
    ) -> f64 {
        let r_m = particle_radius_nm * 1.0e-9;
        let m = particle_index / self.tweezers.medium_index;
        let k_cm = OpticalTweezers::clausius_mossotti(m);

        // Polarizability magnitude
        let alpha = 4.0 * PI * EPS0 * self.tweezers.medium_index.powi(2) * r_m.powi(3) * k_cm;

        // Optical path length in medium
        let lambda_m = self.tweezers.wavelength_nm * 1.0e-9;
        let k_med = 2.0 * PI * self.tweezers.medium_index / lambda_m;
        let d_m = separation_nm * 1.0e-9;

        // Scale force from laser intensity
        let i0 = self.tweezers.peak_intensity_w_per_m2();
        let e0_sq = 2.0 * i0 / (C0 * EPS0 * self.tweezers.medium_index);
        let f_scale =
            alpha.powi(2) * e0_sq * k_med.powi(3) / (4.0 * PI * EPS0 * self.tweezers.medium_index);

        // Oscillatory factor (simplified dipole interaction)
        let kd = k_med * d_m;
        if kd < 1.0e-10 {
            return 0.0;
        }
        f_scale * (kd).sin() / (kd * kd)
    }

    /// Preferred binding separation: first minimum of the oscillatory binding potential.
    ///
    /// Minimum (maximum attractive force) occurs near d ≈ λ/(2 n_m).
    ///
    /// # Returns
    /// Preferred separation (nm)
    pub fn preferred_separation_nm(&self) -> f64 {
        // First potential minimum is near half wavelength in medium
        self.tweezers.wavelength_nm / (2.0 * self.tweezers.medium_index)
    }
}

/// Dual-beam (counter-propagating) optical trap.
///
/// Two counter-propagating beams produce a more stable axial confinement
/// than a single-beam trap, as scattering forces from the two beams
/// partially cancel near the midplane.
#[derive(Debug, Clone)]
pub struct DualBeamTrap {
    pub beam1: OpticalTweezers,
    pub beam2: OpticalTweezers,
}

impl DualBeamTrap {
    /// Create a symmetric dual-beam trap with equal, counter-propagating beams.
    ///
    /// The two foci coincide at z=0, with beam1 propagating in +z
    /// and beam2 propagating in -z direction.
    pub fn symmetric(power_w: f64, lambda_nm: f64, na: f64, n_medium: f64) -> Self {
        let beam1 = OpticalTweezers::new(power_w, lambda_nm, na, n_medium);
        let beam2 = OpticalTweezers::new(power_w, lambda_nm, na, n_medium);
        Self { beam1, beam2 }
    }

    /// Total axial force on a particle at axial position z.
    ///
    /// Beam1 (+z propagating): scattering force in +z, gradient in -z direction
    /// Beam2 (-z propagating): scattering force in -z, gradient in +z direction
    ///
    /// # Arguments
    /// * `particle_radius_nm` - Particle radius (nm)
    /// * `particle_index` - Refractive index of particle
    /// * `z_nm` - Axial position (nm), positive is toward beam1 focus direction
    ///
    /// # Returns
    /// Net axial force (N), positive in +z direction
    pub fn axial_force_n(&self, particle_radius_nm: f64, particle_index: f64, z_nm: f64) -> f64 {
        // Beam 1: propagates in +z, focus at origin
        let pos1 = [0.0, 0.0, z_nm];
        let f1 = self
            .beam1
            .total_force_n(particle_radius_nm, particle_index, pos1);

        // Beam 2: propagates in -z, focus at origin
        // We mirror the position for beam2's frame (it sees -z as forward)
        let pos2 = [0.0, 0.0, -z_nm];
        let f2 = self
            .beam2
            .total_force_n(particle_radius_nm, particle_index, pos2);

        // Beam1 contributes +z components directly
        // Beam2 force in its frame: f2[2] is in its -z direction, so multiply by -1
        f1[2] - f2[2]
    }

    /// Find the axial equilibrium position by bisection.
    ///
    /// For a symmetric trap, this should be z = 0.
    ///
    /// # Returns
    /// Equilibrium axial position (nm)
    pub fn equilibrium_z_nm(&self, particle_radius_nm: f64, particle_index: f64) -> f64 {
        let z_r = self.beam1.rayleigh_range_nm();
        let mut z_low = -z_r;
        let mut z_high = z_r;

        for _ in 0..64 {
            let z_mid = (z_low + z_high) / 2.0;
            let fz = self.axial_force_n(particle_radius_nm, particle_index, z_mid);
            if fz > 0.0 {
                z_low = z_mid;
            } else {
                z_high = z_mid;
            }
        }
        (z_low + z_high) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn standard_tweezers() -> OpticalTweezers {
        // 1064 nm Nd:YAG, 100 mW, NA=1.2, water (n=1.33)
        OpticalTweezers::new(0.1, 1064.0, 1.2, 1.33)
    }

    #[test]
    fn test_beam_waist() {
        let tw = standard_tweezers();
        let w0 = tw.beam_waist_nm();
        let expected = 1064.0 / (PI * 1.2);
        assert_abs_diff_eq!(w0, expected, epsilon = 1.0e-10);
        // Should be ~282 nm
        assert!((w0 - 282.0).abs() < 1.0);
    }

    #[test]
    fn test_peak_intensity() {
        let tw = standard_tweezers();
        let w0_m = tw.beam_waist_nm() * 1.0e-9;
        let i0 = tw.peak_intensity_w_per_m2();
        let expected = 2.0 * tw.beam_power_w / (PI * w0_m * w0_m);
        assert_abs_diff_eq!(i0, expected, epsilon = 1.0);
    }

    #[test]
    fn test_gradient_force_at_center_zero() {
        let tw = standard_tweezers();
        // At the focus center, ∇I = 0 by symmetry, so gradient force = 0
        let f = tw.gradient_force_n(100.0, 1.5, [0.0, 0.0, 0.0]);
        // Due to symmetry, all components should be ~ 0
        assert!(f[0].abs() < 1.0e-25, "Fx at center should be zero");
        assert!(f[1].abs() < 1.0e-25, "Fy at center should be zero");
        assert!(f[2].abs() < 1.0e-25, "Fz at center should be zero");
    }

    #[test]
    fn test_trap_stiffness_positive() {
        let tw = standard_tweezers();
        // Lateral trap stiffness should be positive (restoring force)
        let k_lat = tw.trap_stiffness_n_per_m(100.0, 1.5, TrapAxis::Lateral);
        assert!(
            k_lat > 0.0,
            "Lateral stiffness must be positive, got {}",
            k_lat
        );
    }

    #[test]
    fn test_trap_depth_in_kt() {
        // Standard polystyrene bead (r=100nm, n=1.59) in 1064nm trap
        // Should give > 1 kT trap depth for reasonable parameters
        let tw = standard_tweezers();
        let depth = tw.trap_depth_kt(100.0, 1.59);
        assert!(
            depth > 1.0,
            "Trap depth should exceed 1 kT, got {} kT",
            depth
        );
    }

    #[test]
    fn test_thermal_fluctuation_decreases_with_stiffness() {
        let tw = standard_tweezers();
        // Larger stiffness → smaller thermal fluctuations
        let x1 = tw.thermal_fluctuation_nm(1.0e-5, 300.0);
        let x2 = tw.thermal_fluctuation_nm(1.0e-4, 300.0);
        let x3 = tw.thermal_fluctuation_nm(1.0e-3, 300.0);
        assert!(x1 > x2, "Higher stiffness should reduce fluctuations");
        assert!(x2 > x3, "Higher stiffness should reduce fluctuations");
    }

    #[test]
    fn test_dual_beam_equilibrium_at_center() {
        // Symmetric dual-beam trap should have equilibrium at z=0
        let trap = DualBeamTrap::symmetric(0.1, 1064.0, 1.2, 1.33);
        let z_eq = trap.equilibrium_z_nm(100.0, 1.59);
        // Should be very close to 0 for symmetric trap
        assert!(
            z_eq.abs() < 5.0,
            "Symmetric dual-beam equilibrium should be near z=0 nm, got {} nm",
            z_eq
        );
    }

    #[test]
    fn test_rayleigh_range_positive() {
        let tw = standard_tweezers();
        let z_r = tw.rayleigh_range_nm();
        assert!(z_r > 0.0, "Rayleigh range must be positive");
    }

    #[test]
    fn test_clausius_mossotti_sign() {
        // For n_p > n_m: K > 0 (attracted to high intensity)
        // For n_p < n_m: K < 0 (repelled from high intensity)
        let k_positive = OpticalTweezers::clausius_mossotti(1.2); // m > 1
        let k_negative = OpticalTweezers::clausius_mossotti(0.8); // m < 1
        assert!(k_positive > 0.0, "K(m>1) should be positive");
        assert!(k_negative < 0.0, "K(m<1) should be negative");
    }
}
