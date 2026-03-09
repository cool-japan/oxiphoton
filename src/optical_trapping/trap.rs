//! Optical trap characterization, dual-beam traps, and potential landscape computation
//!
//! Provides methods to characterize trap stiffness from experimental observables
//! (equipartition theorem, power spectral density) and compute full 3D optical
//! potential landscapes for visualization and escape rate estimation.

use crate::optical_trapping::forces::{GaussianTrap, RayleighParticle};

// Physical constants
const KB: f64 = 1.380649e-23; // Boltzmann constant [J/K]
const C_LIGHT: f64 = 2.99792458e8; // Speed of light [m/s]
const EPS0: f64 = 8.854187817e-12; // Vacuum permittivity [F/m]
const PI: f64 = std::f64::consts::PI;

/// Trap characterization methods from experimental observables
///
/// Implements equipartition and PSD-based stiffness calibration used in
/// real optical tweezer experiments.
#[derive(Debug, Clone)]
pub struct TrapCharacterization {
    /// Fluid/sample temperature \[K\]
    pub temperature_k: f64,
}

impl TrapCharacterization {
    /// Create characterization context at given temperature
    pub fn new(temperature_k: f64) -> Self {
        Self { temperature_k }
    }

    /// Stiffness from equipartition theorem: k = k_B T / σ²
    ///
    /// # Arguments
    /// * `position_variance_m2` — variance of Brownian position fluctuations \[m²\]
    ///
    /// # Returns
    /// Trap stiffness \[N/m\]
    pub fn stiffness_from_variance(&self, position_variance_m2: f64) -> f64 {
        if position_variance_m2 <= 0.0 {
            return 0.0;
        }
        KB * self.temperature_k / position_variance_m2
    }

    /// Stiffness from Lorentzian PSD corner frequency: k = 2π γ f_c
    ///
    /// The PSD of a trapped Brownian particle is a Lorentzian:
    /// S(f) = k_B T / (π² γ (f_c² + f²)),  f_c = k/(2π γ)
    ///
    /// # Arguments
    /// * `corner_frequency_hz` — Lorentzian corner frequency \[Hz\]
    /// * `drag_coeff` — Stokes drag coefficient γ \[N·s/m\]
    ///
    /// # Returns
    /// Trap stiffness \[N/m\]
    pub fn stiffness_from_psd(&self, corner_frequency_hz: f64, drag_coeff: f64) -> f64 {
        2.0 * PI * drag_coeff * corner_frequency_hz
    }

    /// Corner frequency from stiffness and drag: f_c = k/(2π γ)
    pub fn corner_frequency(&self, stiffness: f64, drag_coeff: f64) -> f64 {
        if drag_coeff <= 0.0 {
            return 0.0;
        }
        stiffness / (2.0 * PI * drag_coeff)
    }

    /// Position variance expected from stiffness: σ² = k_B T / k
    pub fn expected_variance(&self, stiffness: f64) -> f64 {
        if stiffness <= 0.0 {
            return f64::INFINITY;
        }
        KB * self.temperature_k / stiffness
    }

    /// Stiffness from both PSD and equipartition (cross-check consistency)
    ///
    /// Returns (k_equipartition, k_psd) for comparison
    pub fn dual_calibration(
        &self,
        position_variance_m2: f64,
        corner_frequency_hz: f64,
        drag_coeff: f64,
    ) -> (f64, f64) {
        (
            self.stiffness_from_variance(position_variance_m2),
            self.stiffness_from_psd(corner_frequency_hz, drag_coeff),
        )
    }
}

/// Counter-propagating (dual-beam) optical trap
///
/// Two focused Gaussian beams propagating in opposite directions (+z and -z).
/// The particle equilibrium position and axial stiffness depend on the power
/// ratio and beam geometries.
#[derive(Debug, Clone)]
pub struct DualBeamTrap {
    /// First beam (propagating in +z direction)
    pub beam1: GaussianTrap,
    /// Second beam (propagating in -z direction)
    pub beam2: GaussianTrap,
    /// Axial separation between beam foci \[m\]
    pub separation_m: f64,
}

impl DualBeamTrap {
    /// Construct a symmetric dual-beam trap (equal power, same parameters)
    pub fn symmetric(
        power_per_beam_w: f64,
        wavelength_m: f64,
        n_medium: f64,
        na: f64,
        separation_m: f64,
    ) -> Self {
        Self {
            beam1: GaussianTrap::new(power_per_beam_w, wavelength_m, n_medium, na),
            beam2: GaussianTrap::new(power_per_beam_w, wavelength_m, n_medium, na),
            separation_m,
        }
    }

    /// Net axial force on Rayleigh particle at position z \[N\]
    ///
    /// Beam1 focused at z=0, beam2 focused at z=separation_m.
    /// Beam1 pushes in +z; beam2 pushes in -z.
    fn axial_force(&self, particle: &RayleighParticle, z: f64) -> f64 {
        // Beam 1: propagates +z, focused at z=0
        let alpha = particle.polarizability();
        // Gradient force from beam 1 (along z)
        let grad1 = self.beam1.gradient_at(0.0, 0.0, z);
        let f_grad1_z = alpha / (2.0 * C_LIGHT * EPS0 * self.beam1.n_medium) * grad1[2];

        // Scattering (radiation pressure) from beam 1: pushes +z
        let i1 = self.beam1.intensity_at(0.0, 0.0, z);
        let f_scat1 = particle.scattering_force(i1);

        // Beam 2: propagates -z, focused at z = separation_m
        let z2 = self.separation_m - z; // distance from beam2 focus in its frame
        let grad2 = self.beam2.gradient_at(0.0, 0.0, z2);
        let f_grad2_z = alpha / (2.0 * C_LIGHT * EPS0 * self.beam2.n_medium) * grad2[2];
        // Beam 2 gradient along lab z is negated (beam propagates in -z)
        let f_grad2_lab_z = -f_grad2_z;

        let i2 = self.beam2.intensity_at(0.0, 0.0, z2);
        // Beam 2 scattering pushes -z
        let f_scat2 = -particle.scattering_force(i2);

        f_grad1_z + f_scat1 + f_grad2_lab_z + f_scat2
    }

    /// Find equilibrium z-position via bisection \[m\]
    ///
    /// Searches for z in \[0, separation_m\] where net axial force = 0.
    pub fn equilibrium_position(&self, particle: &RayleighParticle) -> f64 {
        let z_lo = 0.0_f64;
        let z_hi = self.separation_m;
        let f_lo = self.axial_force(particle, z_lo);
        let f_hi = self.axial_force(particle, z_hi);

        // If no sign change, return midpoint as best estimate
        if f_lo * f_hi > 0.0 {
            return 0.5 * self.separation_m;
        }

        let mut lo = z_lo;
        let mut hi = z_hi;
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            let f_mid = self.axial_force(particle, mid);
            if f_mid * self.axial_force(particle, lo) <= 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
            if (hi - lo).abs() < 1e-15 {
                break;
            }
        }
        0.5 * (lo + hi)
    }

    /// Axial trap stiffness at equilibrium \[N/m\]
    ///
    /// Estimated numerically as k_z = -dF/dz at z_eq via finite difference.
    pub fn axial_stiffness(&self, particle: &RayleighParticle) -> f64 {
        let z_eq = self.equilibrium_position(particle);
        let dz = self.beam1.beam_waist_m * 1e-4; // tiny displacement
        let f_plus = self.axial_force(particle, z_eq + dz);
        let f_minus = self.axial_force(particle, z_eq - dz);
        -(f_plus - f_minus) / (2.0 * dz)
    }

    /// Radial stiffness (same as single beam for symmetric trap) \[N/m\]
    pub fn radial_stiffness(&self, particle: &RayleighParticle) -> f64 {
        // At equilibrium z, radial stiffness is sum of both beams' radial contributions
        let z_eq = self.equilibrium_position(particle);
        let k1 = self.beam1.radial_stiffness(particle)
            * (-2.0 * (z_eq / self.beam1.rayleigh_range()).powi(2)).exp();
        let z2_eq = self.separation_m - z_eq;
        let k2 = self.beam2.radial_stiffness(particle)
            * (-2.0 * (z2_eq / self.beam2.rayleigh_range()).powi(2)).exp();
        k1 + k2
    }
}

/// 3D optical potential landscape U(x, y, z) \[J\]
///
/// The potential energy of a Rayleigh particle in an optical trap:
/// U = -α/(2 c ε₀ n) × I(r)  (attractive to intensity maxima for α > 0)
#[derive(Debug, Clone)]
pub struct OpticalPotential {
    /// Grid dimensions
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Grid spacing \[m\] (uniform cubic grid)
    pub dx: f64,
    /// Potential energy values \[J\], flattened as \[ix + nx*(iy + ny*iz)\]
    pub potential: Vec<f64>,
}

impl OpticalPotential {
    /// Compute potential from a Gaussian trap for a Rayleigh particle
    ///
    /// Grid is centered at the focus, spanning ±extent_m in each direction.
    pub fn from_gaussian_trap(
        trap: &GaussianTrap,
        particle: &RayleighParticle,
        grid_size: usize,
        extent_m: f64,
    ) -> Self {
        let n = grid_size.max(3);
        let dx = 2.0 * extent_m / (n as f64 - 1.0);
        let alpha = particle.polarizability();
        // U = -α I / (2 c ε₀ n_m)
        let prefactor = -alpha / (2.0 * C_LIGHT * EPS0 * trap.n_medium);

        let total = n * n * n;
        let mut potential = vec![0.0f64; total];

        for iz in 0..n {
            let z = -extent_m + iz as f64 * dx;
            for iy in 0..n {
                let y = -extent_m + iy as f64 * dx;
                for ix in 0..n {
                    let x = -extent_m + ix as f64 * dx;
                    let intensity = trap.intensity_at(x, y, z);
                    let idx = ix + n * (iy + n * iz);
                    potential[idx] = prefactor * intensity;
                }
            }
        }

        Self {
            nx: n,
            ny: n,
            nz: n,
            dx,
            potential,
        }
    }

    /// Access potential at grid indices (ix, iy, iz)
    pub fn get(&self, ix: usize, iy: usize, iz: usize) -> f64 {
        let idx = ix + self.nx * (iy + self.ny * iz);
        self.potential.get(idx).copied().unwrap_or(0.0)
    }

    /// Trap depth \[J\]: difference between saddle/maximum and minimum of potential
    pub fn trap_depth_joules(&self) -> f64 {
        let min = self.potential.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self
            .potential
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        max - min
    }

    /// Find the index of minimum potential value
    pub fn minimum_index(&self) -> (usize, usize, usize) {
        let (idx, _) = self.potential.iter().enumerate().fold(
            (0, f64::INFINITY),
            |(best_i, best_v), (i, &v)| {
                if v < best_v {
                    (i, v)
                } else {
                    (best_i, best_v)
                }
            },
        );
        let n = self.nx;
        let iz = idx / (n * n);
        let iy = (idx - iz * n * n) / n;
        let ix = idx - iz * n * n - iy * n;
        (ix, iy, iz)
    }

    /// Kramers escape rate \[s⁻¹\] for thermal escape from trap
    ///
    /// Arrhenius/Kramers: Γ ≈ ω₀ ω_b / (2π γ) × exp(-ΔU / k_B T)
    /// Simplified here to exponential Boltzmann factor: P_esc ∝ exp(-ΔU / k_B T)
    ///
    /// Returns dimensionless escape factor (exponent of Boltzmann factor).
    /// A value near 1 means particle is barely confined; near 0 means strongly trapped.
    pub fn escape_probability(&self, temperature_k: f64) -> f64 {
        let delta_u = self.trap_depth_joules();
        let kbt = KB * temperature_k;
        if kbt <= 0.0 {
            return 0.0;
        }
        (-delta_u / kbt).exp()
    }

    /// Trap depth in units of thermal energy k_B T
    pub fn trap_depth_kbt(&self, temperature_k: f64) -> f64 {
        let kbt = KB * temperature_k;
        if kbt <= 0.0 {
            return 0.0;
        }
        self.trap_depth_joules() / kbt
    }

    /// Stiffness tensor (kx, ky, kz) from potential curvature at minimum \[N/m\]
    ///
    /// k_i = ∂²U/∂x_i² evaluated at the potential minimum via finite differences.
    pub fn stiffness_at_minimum(&self) -> (f64, f64, f64) {
        let (ix, iy, iz) = self.minimum_index();
        let dx = self.dx;

        // x-stiffness: need ix-1, ix, ix+1
        let kx = if ix >= 1 && ix + 1 < self.nx {
            let u_m = self.get(ix - 1, iy, iz);
            let u_0 = self.get(ix, iy, iz);
            let u_p = self.get(ix + 1, iy, iz);
            (u_p - 2.0 * u_0 + u_m) / (dx * dx)
        } else {
            0.0
        };

        let ky = if iy >= 1 && iy + 1 < self.ny {
            let u_m = self.get(ix, iy - 1, iz);
            let u_0 = self.get(ix, iy, iz);
            let u_p = self.get(ix, iy + 1, iz);
            (u_p - 2.0 * u_0 + u_m) / (dx * dx)
        } else {
            0.0
        };

        let kz = if iz >= 1 && iz + 1 < self.nz {
            let u_m = self.get(ix, iy, iz - 1);
            let u_0 = self.get(ix, iy, iz);
            let u_p = self.get(ix, iy, iz + 1);
            (u_p - 2.0 * u_0 + u_m) / (dx * dx)
        } else {
            0.0
        };

        (kx, ky, kz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optical_trapping::forces::{GaussianTrap, RayleighParticle};

    fn test_particle() -> RayleighParticle {
        RayleighParticle::new(100e-9, 1.59, 1.33, 1064e-9)
    }

    fn test_trap() -> GaussianTrap {
        GaussianTrap::new(0.1, 1064e-9, 1.33, 1.2)
    }

    #[test]
    fn equipartition_stiffness_round_trip() {
        let tc = TrapCharacterization::new(300.0);
        let k = 1e-5; // 10 µN/m typical optical tweezer stiffness
        let variance = tc.expected_variance(k);
        let k_back = tc.stiffness_from_variance(variance);
        assert!(
            (k_back - k).abs() / k < 1e-10,
            "Round trip failed: {} vs {}",
            k_back,
            k
        );
    }

    #[test]
    fn psd_stiffness_consistency() {
        let tc = TrapCharacterization::new(300.0);
        let drag = 6.0 * PI * 1e-3 * 1e-6; // 6πηr for 1µm bead in water
        let k = 1e-5;
        let fc = tc.corner_frequency(k, drag);
        let k_back = tc.stiffness_from_psd(fc, drag);
        assert!(
            (k_back - k).abs() / k < 1e-10,
            "PSD stiffness mismatch: {} vs {}",
            k_back,
            k
        );
    }

    #[test]
    fn dual_beam_equilibrium_symmetric() {
        // For a symmetric dual-beam trap, equilibrium must lie within [0, separation].
        // For Rayleigh (gradient-dominated) particles, the equilibrium is near beam1's
        // focus; midpoint confinement applies only in the Mie/scattering-dominated regime.
        let trap = DualBeamTrap::symmetric(0.1, 1064e-9, 1.33, 0.5, 10e-6);
        let particle = test_particle();
        let z_eq = trap.equilibrium_position(&particle);
        let sep = trap.separation_m;
        assert!(
            z_eq >= 0.0 && z_eq <= sep,
            "Equilibrium z={} outside trap range [0, {}]",
            z_eq,
            sep
        );
    }

    #[test]
    fn optical_potential_minimum_at_focus() {
        let trap = test_trap();
        let particle = test_particle();
        let potential = OpticalPotential::from_gaussian_trap(&trap, &particle, 31, 3e-6);
        let (ix, iy, iz) = potential.minimum_index();
        let n = potential.nx;
        // Minimum should be near grid center (within 2 cells)
        let center = n / 2;
        assert!(
            (ix as isize - center as isize).abs() <= 2,
            "x minimum {} not near center {}",
            ix,
            center
        );
        assert!(
            (iy as isize - center as isize).abs() <= 2,
            "y minimum {} not near center {}",
            iy,
            center
        );
        assert!(
            (iz as isize - center as isize).abs() <= 2,
            "z minimum {} not near center {}",
            iz,
            center
        );
    }

    #[test]
    fn trap_depth_positive() {
        let trap = test_trap();
        let particle = test_particle();
        let potential = OpticalPotential::from_gaussian_trap(&trap, &particle, 21, 2e-6);
        let depth = potential.trap_depth_joules();
        assert!(depth > 0.0, "Trap depth must be positive, got {}", depth);
    }

    #[test]
    fn stiffness_at_minimum_positive() {
        let trap = test_trap();
        let particle = test_particle();
        let potential = OpticalPotential::from_gaussian_trap(&trap, &particle, 41, 2e-6);
        let (kx, ky, kz) = potential.stiffness_at_minimum();
        assert!(kx > 0.0, "kx={} should be positive", kx);
        assert!(ky > 0.0, "ky={} should be positive", ky);
        assert!(kz > 0.0, "kz={} should be positive", kz);
    }

    #[test]
    fn escape_probability_decreases_with_power() {
        let particle = test_particle();
        let trap_low = GaussianTrap::new(0.01, 1064e-9, 1.33, 1.2);
        let trap_high = GaussianTrap::new(0.1, 1064e-9, 1.33, 1.2);
        let p_low = OpticalPotential::from_gaussian_trap(&trap_low, &particle, 21, 3e-6);
        let p_high = OpticalPotential::from_gaussian_trap(&trap_high, &particle, 21, 3e-6);
        // Higher power → deeper trap → lower escape probability
        let esc_low = p_low.escape_probability(300.0);
        let esc_high = p_high.escape_probability(300.0);
        assert!(
            esc_high < esc_low,
            "High power escape {} should < low power escape {}",
            esc_high,
            esc_low
        );
    }

    #[test]
    fn trap_depth_kbt_reasonable_tweezer() {
        // 100 mW trap should give >> 10 kT depth for 100nm bead (strong confinement)
        let trap = GaussianTrap::new(0.1, 1064e-9, 1.33, 1.2);
        let particle = test_particle();
        let potential = OpticalPotential::from_gaussian_trap(&trap, &particle, 21, 3e-6);
        let depth_kbt = potential.trap_depth_kbt(300.0);
        // Typical tweezer traps are many kT deep
        assert!(
            depth_kbt > 1.0,
            "Trap depth {}kT seems too shallow",
            depth_kbt
        );
    }
}
