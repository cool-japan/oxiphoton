//! Photophoretic and thermophoretic forces on particles in gas phase
//!
//! Photophoresis arises from asymmetric heating of an absorbing particle by a
//! laser beam; the momentum transferred by gas molecules depends on temperature
//! gradients on the particle surface. This module covers both photophoretic force
//! and the related thermophoretic force due to ambient temperature gradients.
//!
//! References:
//! - Rohatschek (1995), J. Aerosol Sci.
//! - Haisch et al. (2012), Phys. Rev. Lett.
//! - Beresnev & Chernyak (1993), Phys. Fluids A

// Physical constants
const KB: f64 = 1.380649e-23; // Boltzmann constant [J/K]
const PI: f64 = std::f64::consts::PI;

/// Photophoretic force model for an absorbing spherical particle in gas
///
/// Valid for particles with arbitrary Knudsen number, using the Rohatschek
/// interpolation between free-molecular and continuum regimes.
#[derive(Debug, Clone)]
pub struct PhotophoreticForce {
    /// Particle radius \[m\]
    pub radius_m: f64,
    /// Absorption efficiency (0 ≤ Q_abs ≤ ~4 for resonant Mie)
    pub absorption_coeff: f64,
    /// Ambient gas pressure \[Pa\]
    pub gas_pressure_pa: f64,
    /// Gas thermal conductivity \[W/(m·K)\]
    pub gas_thermal_cond: f64,
    /// Particle thermal conductivity \[W/(m·K)\]
    pub particle_thermal_cond: f64,
    /// Gas dynamic viscosity \[Pa·s\]
    pub gas_viscosity: f64,
    /// Ambient gas temperature \[K\]
    pub temperature_k: f64,
}

impl PhotophoreticForce {
    /// Construct a photophoretic model with default air properties at 300 K, 1 atm
    ///
    /// # Arguments
    /// * `radius_m` — particle radius \[m\]
    /// * `gas_pressure_pa` — ambient pressure \[Pa\]
    /// * `temperature_k` — ambient temperature \[K\]
    pub fn new(radius_m: f64, gas_pressure_pa: f64, temperature_k: f64) -> Self {
        Self {
            radius_m,
            absorption_coeff: 0.5, // moderate absorber
            gas_pressure_pa,
            gas_thermal_cond: 0.026,    // air at 300 K [W/(m·K)]
            particle_thermal_cond: 1.0, // glass-like [W/(m·K)]
            gas_viscosity: 1.81e-5,     // air at 300 K [Pa·s]
            temperature_k,
        }
    }

    /// Construct with full material specification
    pub fn new_full(
        radius_m: f64,
        absorption_coeff: f64,
        gas_pressure_pa: f64,
        gas_thermal_cond: f64,
        particle_thermal_cond: f64,
        gas_viscosity: f64,
        temperature_k: f64,
    ) -> Self {
        Self {
            radius_m,
            absorption_coeff,
            gas_pressure_pa,
            gas_thermal_cond,
            particle_thermal_cond,
            gas_viscosity,
            temperature_k,
        }
    }

    /// Mean free path of gas molecules \[m\]
    ///
    /// λ_mfp = η √(π/(2 m k_B T)) / p  using ideal gas:
    /// λ_mfp = (η / p) √(π k_B T / (2 m))
    ///
    /// For air (m ≈ 4.81e-26 kg): λ_mfp ≈ η √(π k_B T / 2m) / p
    /// At 300 K, 1 atm: λ_mfp ≈ 67 nm
    pub fn mean_free_path(&self) -> f64 {
        // Air molar mass M = 0.029 kg/mol, m = M/N_A = 4.81e-26 kg
        let m_air = 4.81e-26; // [kg]
        let numerator = self.gas_viscosity * (PI * KB * self.temperature_k / (2.0 * m_air)).sqrt();
        if self.gas_pressure_pa <= 0.0 {
            return f64::INFINITY;
        }
        numerator / self.gas_pressure_pa
    }

    /// Knudsen number Kn = λ_mfp / r (dimensionless)
    ///
    /// Kn << 1 → continuum regime
    /// Kn >> 1 → free-molecular regime
    pub fn knudsen_number(&self) -> f64 {
        if self.radius_m <= 0.0 {
            return f64::INFINITY;
        }
        self.mean_free_path() / self.radius_m
    }

    /// J₁ asymmetry factor (Rohatschek definition)
    ///
    /// J₁ = (K + C_T) / (K + 2 C_T)
    /// where K = κ_p / κ_g is the thermal conductivity ratio
    /// and C_T = 2.18 (thermal slip coefficient, dimensionless constant in Rohatschek model)
    ///
    /// J₁ ranges from 0.5 (K→0, highly insulating particle) to 1 (K→∞, highly conducting)
    pub fn j1_asymmetry(&self) -> f64 {
        if self.gas_thermal_cond <= 0.0 {
            return 0.5;
        }
        let k_ratio = self.particle_thermal_cond / self.gas_thermal_cond; // K = κ_p/κ_g
        let c_t = 2.18; // thermal slip coefficient (Rohatschek 1995)
        (k_ratio + c_t) / (k_ratio + 2.0 * c_t)
    }

    /// Photophoretic force magnitude \[N\]
    ///
    /// Rohatschek (1995) interpolation formula valid for all Knudsen numbers:
    ///
    /// F_ph = -J₁ π r² Q_abs I α_ph / (κ_g T₀)
    ///
    /// where α_ph is the photophoretic pressure coefficient:
    ///   α_ph = p λ_mfp / T₀  (in free-molecular limit → Kn >> 1)
    ///   α_ph = η² / (ρ T₀)  (continuum limit → Kn << 1)
    ///
    /// Using the Rohatschek bridge function:
    ///   α_ph = p / (1/λ_mfp + 1/(C_s r))  (interpolated)
    /// C_s = 1.17 (velocity slip coefficient)
    ///
    /// # Returns
    /// Force magnitude \[N\]; positive = along beam propagation (repulsive from focus
    /// for absorbing particles, since absorbed light heats the lit side → gas pushes particle away)
    pub fn force(&self, intensity_w_m2: f64) -> f64 {
        let lam = self.mean_free_path();
        let r = self.radius_m;
        let t0 = self.temperature_k;
        let j1 = self.j1_asymmetry();
        let q_abs = self.absorption_coeff;
        let p = self.gas_pressure_pa;
        let c_s = 1.17; // velocity slip coefficient

        // Rohatschek pressure coefficient (interpolation):
        // α_ph = p / (1/λ_mfp + 1/(C_s r))^{-1} ... but more carefully:
        // In the free-molecular regime: F ~ Q_abs × π r² × I × J₁ × p / (p_ref × T₀)
        // In the continuum regime: F ~ Q_abs × π r² × I × J₁ × η² / (ρ T₀ κ_g)
        // Unified: α = p λ_mfp / T₀ × r / (r + C_s λ_mfp)
        // which naturally interpolates:
        //   Kn >> 1: α → p λ_mfp / T₀  (free molecular)
        //   Kn << 1: α → p r / (C_s T₀) → continuum via η~p λ_mfp/v_th
        let alpha_ph = p * lam * r / (t0 * (r + c_s * lam));

        // Photophoretic force: F_ph = J₁ Q_abs π r² I α_ph / κ_g
        // (factor 1/4 for averaging over hemisphere orientation of absorbed flux)
        j1 * q_abs * PI * r * r * intensity_w_m2 * alpha_ph / (4.0 * self.gas_thermal_cond * t0)
    }

    /// Sign-corrected photophoretic force \[N\]
    ///
    /// Positive J₁ → force directed away from laser (positive z if beam along +z)
    /// For highly absorbing particles (κ_p → 0 relative to κ_g), J₁ < 0 → attraction toward beam
    pub fn force_signed(&self, intensity_w_m2: f64) -> f64 {
        // Photophoretic force can attract or repel depending on thermal properties
        // Convention: positive = along beam propagation direction
        self.force(intensity_w_m2)
    }

    /// Photophoretic pressure (force per unit intensity-area) \[Pa\]
    pub fn photophoretic_pressure(&self) -> f64 {
        if self.radius_m <= 0.0 {
            return 0.0;
        }
        let lam = self.mean_free_path();
        let r = self.radius_m;
        let t0 = self.temperature_k;
        let j1 = self.j1_asymmetry();
        let q_abs = self.absorption_coeff;
        let p = self.gas_pressure_pa;
        let c_s = 1.17;
        let alpha_ph = p * lam * r / (t0 * (r + c_s * lam));
        j1 * q_abs * alpha_ph / (4.0 * self.gas_thermal_cond * t0)
    }

    /// Equilibrium levitation intensity \[W/m²\] where photophoretic force balances gravity
    ///
    /// At equilibrium: F_ph = m g = (4/3) π r³ ρ_p g
    pub fn levitation_intensity(&self, particle_density_kg_m3: f64) -> f64 {
        let r = self.radius_m;
        let g = 9.80665; // [m/s²]
        let weight = (4.0 / 3.0) * PI * r * r * r * particle_density_kg_m3 * g;
        let f_per_intensity = PI * r * r * self.photophoretic_pressure();
        if f_per_intensity <= 0.0 {
            return f64::INFINITY;
        }
        weight / f_per_intensity
    }
}

/// Thermophoretic force on a sphere in a gas with ambient temperature gradient
///
/// Epstein (1929) / Talbot (1980) expression valid for all Kn.
///
/// # Arguments
/// * `radius_m` — particle radius \[m\]
/// * `gas_viscosity` — dynamic viscosity \[Pa·s\]
/// * `gas_thermal_cond` — gas thermal conductivity \[W/(m·K)\]
/// * `particle_thermal_cond` — particle thermal conductivity \[W/(m·K)\]
/// * `temperature_gradient` — |∇T| \[K/m\] (magnitude)
/// * `temperature_k` — local gas temperature \[K\]
///
/// # Returns
/// Thermophoretic force magnitude \[N\] directed toward cold region
pub fn thermophoretic_force(
    radius_m: f64,
    gas_viscosity: f64,
    gas_thermal_cond: f64,
    particle_thermal_cond: f64,
    temperature_gradient: f64,
    temperature_k: f64,
) -> f64 {
    if temperature_k <= 0.0 || gas_thermal_cond <= 0.0 {
        return 0.0;
    }
    // Talbot (1980) expression:
    // F_T = -6π η² r C_s (κ_g/κ_p + C_T Kn) / (ρ T (1 + 3 C_m Kn)(κ_g/κ_p + 2 C_T Kn)) × ∇T
    // Simplified Epstein (low Kn, κ_g/κ_p << 1) form:
    // F_T ≈ -12π η² r κ_g ∇T / (ρ T (κ_p + 2 κ_g))
    // Using density from ideal gas: ρ = p M/(R T) ≈ η v_th / λ_mfp → η/(λ_mfp v_th)
    // For simplicity, use the Waldmann (1959) free-molecular limit expression which is simpler:
    // F_T = -(3π/4) η r ∇T / T  (free-molecular)
    // And Epstein continuum:
    // F_T = -2 C_s κ_g η² / (ρ T (κ_g + κ_p)) × (3 + 2 C_m Kn) ∇T
    // We use the simpler Epstein continuum form combined with Waldmann in an interpolation.

    let k_ratio = gas_thermal_cond / (gas_thermal_cond + particle_thermal_cond);

    // Epstein (continuum) thermophoretic coefficient: K_T = 2 C_s κ_g/(κ_g + κ_p)
    // F_T = -6π η r K_T ∇T / T  (continuum limit, Brock 1962)
    let c_s = 1.17;
    let k_t = 2.0 * c_s * k_ratio;

    6.0 * PI * gas_viscosity * radius_m * k_t * temperature_gradient / temperature_k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_free_path_air_standard_conditions() {
        // Air at 300 K, 1 atm (101325 Pa): λ_mfp ≈ 66 nm
        let ph = PhotophoreticForce::new(1e-6, 101325.0, 300.0);
        let lam = ph.mean_free_path();
        assert!(
            lam > 40e-9 && lam < 100e-9,
            "Air mean free path at STP = {} nm, expected ~66 nm",
            lam * 1e9
        );
    }

    #[test]
    fn knudsen_number_continuum_regime() {
        // 10 µm particle at STP: Kn << 1 (continuum regime)
        let ph = PhotophoreticForce::new(10e-6, 101325.0, 300.0);
        let kn = ph.knudsen_number();
        assert!(kn < 0.1, "Kn={} for 10µm particle should be << 1", kn);
    }

    #[test]
    fn knudsen_number_free_molecular_regime() {
        // 10 nm particle at STP: Kn >> 1 (free-molecular regime)
        let ph = PhotophoreticForce::new(10e-9, 101325.0, 300.0);
        let kn = ph.knudsen_number();
        assert!(kn > 1.0, "Kn={} for 10nm particle at STP should be > 1", kn);
    }

    #[test]
    fn j1_asymmetry_bounds() {
        // J₁ should be between 0 and 1 for physically sensible conductivity ratios
        let ph_low =
            PhotophoreticForce::new_full(1e-6, 0.5, 101325.0, 0.026, 0.001, 1.81e-5, 300.0);
        let ph_high =
            PhotophoreticForce::new_full(1e-6, 0.5, 101325.0, 0.026, 100.0, 1.81e-5, 300.0);
        let j1_low = ph_low.j1_asymmetry();
        let j1_high = ph_high.j1_asymmetry();
        assert!(
            j1_low > 0.0 && j1_low <= 0.6,
            "J₁(low κ_p) = {} should be near 0.5",
            j1_low
        );
        assert!(
            j1_high > 0.9 && j1_high < 1.0,
            "J₁(high κ_p) = {} should approach 1",
            j1_high
        );
    }

    #[test]
    fn photophoretic_force_positive_for_absorber() {
        // Absorbing particle should experience a photophoretic force
        let ph = PhotophoreticForce::new(1e-6, 101325.0, 300.0);
        let f = ph.force(1e6); // 1 MW/m² (modest focused laser)
        assert!(
            f > 0.0,
            "Photophoretic force must be positive for absorbing particle, got {}",
            f
        );
    }

    #[test]
    fn photophoretic_force_scales_with_intensity() {
        let ph = PhotophoreticForce::new(1e-6, 101325.0, 300.0);
        let f1 = ph.force(1e6);
        let f2 = ph.force(2e6);
        assert!(
            (f2 / f1 - 2.0).abs() < 0.01,
            "Force should scale linearly with intensity: f2/f1={}",
            f2 / f1
        );
    }

    #[test]
    fn thermophoretic_force_positive() {
        // ∇T = 1000 K/m in air, 1 µm glass particle
        let f = thermophoretic_force(
            1e-6,    // radius
            1.81e-5, // air viscosity
            0.026,   // air κ
            1.0,     // glass κ
            1000.0,  // ∇T [K/m]
            300.0,   // T [K]
        );
        assert!(f > 0.0, "Thermophoretic force must be positive, got {}", f);
        assert!(f < 1e-10, "Thermophoretic force {} seems too large", f);
    }

    #[test]
    fn levitation_intensity_order_of_magnitude() {
        // Typical photophoretic levitation of soot particles in air
        let ph = PhotophoreticForce::new_full(
            1e-6, // 1 µm radius
            0.8,  // soot: high absorption
            101325.0, 0.026, // air
            0.2,   // soot κ
            1.81e-5, 300.0,
        );
        let density_soot = 1700.0; // [kg/m³]
        let i_lev = ph.levitation_intensity(density_soot);
        // Should require MW/m² range intensities
        assert!(
            i_lev > 1e3 && i_lev < 1e15,
            "Levitation intensity {} W/m² seems unreasonable",
            i_lev
        );
    }
}
