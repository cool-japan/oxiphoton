//! Tissue optical properties and light transport models
//!
//! Implements:
//! - Optical properties of biological tissue (μa, μs, g, n)
//! - Diffusion approximation for fluence rate computation
//! - Hemoglobin oxygenation spectroscopy
//! - Photodynamic therapy (PDT) dosimetry

use crate::error::OxiPhotonError;

/// Speed of light in vacuum (m/s)
const C0: f64 = 2.99792458e8;
/// Planck constant (J·s)
const H_PLANCK: f64 = 6.62607015e-34;
/// Elementary charge (C) — used for eV conversions
#[allow(dead_code)]
const E_CHARGE: f64 = 1.602176634e-19;
/// Universal gas constant (J/(mol·K))
const R_GAS: f64 = 8.314462618;
/// Avogadro's number (mol⁻¹)
const AVOGADRO: f64 = 6.02214076e23;

/// Optical properties of biological tissue at a given wavelength.
///
/// All coefficients follow standard tissue optics conventions:
/// - μa: absorption coefficient (cm⁻¹)
/// - μs: scattering coefficient (cm⁻¹)
/// - g:  anisotropy factor (dimensionless, 0 = isotropic, ~0.9 for tissue)
/// - n:  refractive index (~1.37 for soft tissue)
#[derive(Debug, Clone)]
pub struct TissueOpticalProperties {
    /// Human-readable tissue name
    pub name: String,
    /// Wavelength in nanometers
    pub wavelength_nm: f64,
    /// Absorption coefficient μa (cm⁻¹)
    pub absorption_coefficient_cm: f64,
    /// Scattering coefficient μs (cm⁻¹)
    pub scattering_coefficient_cm: f64,
    /// Anisotropy factor g ∈ \[-1, 1\]
    pub anisotropy_factor: f64,
    /// Refractive index n
    pub refractive_index: f64,
}

impl TissueOpticalProperties {
    /// Create a new tissue optical property set.
    pub fn new(
        name: impl Into<String>,
        lambda_nm: f64,
        mu_a: f64,
        mu_s: f64,
        g: f64,
        n: f64,
    ) -> Self {
        Self {
            name: name.into(),
            wavelength_nm: lambda_nm,
            absorption_coefficient_cm: mu_a,
            scattering_coefficient_cm: mu_s,
            anisotropy_factor: g,
            refractive_index: n,
        }
    }

    /// Skin dermis optical properties at 630 nm (PDT wavelength).
    ///
    /// Representative values: μa=0.2 cm⁻¹, μs=200 cm⁻¹, g=0.9
    pub fn skin_dermis_630nm() -> Self {
        Self::new("Skin Dermis @ 630 nm", 630.0, 0.2, 200.0, 0.9, 1.37)
    }

    /// Muscle tissue optical properties at 630 nm.
    ///
    /// Representative values: μa=0.5 cm⁻¹, μs=100 cm⁻¹, g=0.9
    pub fn muscle_630nm() -> Self {
        Self::new("Muscle @ 630 nm", 630.0, 0.5, 100.0, 0.9, 1.37)
    }

    /// Brain gray matter optical properties at 630 nm.
    ///
    /// Representative values: μa=0.1 cm⁻¹, μs=100 cm⁻¹, g=0.9
    pub fn brain_gray_matter() -> Self {
        Self::new("Brain Gray Matter @ 630 nm", 630.0, 0.1, 100.0, 0.9, 1.36)
    }

    /// Liver optical properties at 630 nm.
    ///
    /// Liver has higher absorption due to hemoglobin and bilirubin content.
    pub fn liver() -> Self {
        Self::new("Liver @ 630 nm", 630.0, 1.5, 150.0, 0.9, 1.38)
    }

    /// Fully oxygenated whole blood optical properties at 630 nm.
    ///
    /// Oxygenated hemoglobin (HbO2) dominates absorption in the red window.
    pub fn blood_oxygenated() -> Self {
        Self::new(
            "Blood (oxygenated) @ 630 nm",
            630.0,
            2.3,
            400.0,
            0.994,
            1.40,
        )
    }

    /// Adipose (fat) tissue optical properties at 630 nm.
    ///
    /// Fat has low absorption and moderate scattering.
    pub fn fat() -> Self {
        Self::new("Fat @ 630 nm", 630.0, 0.06, 120.0, 0.9, 1.44)
    }

    /// Reduced scattering coefficient: μs' = μs · (1 − g)
    ///
    /// This accounts for the forward-scattering bias and is used in
    /// the diffusion approximation.
    pub fn reduced_scattering_coefficient(&self) -> f64 {
        self.scattering_coefficient_cm * (1.0 - self.anisotropy_factor)
    }

    /// Transport mean free path: l_tr = 1 / (μa + μs')  (cm)
    pub fn transport_mfp_cm(&self) -> f64 {
        let mu_s_prime = self.reduced_scattering_coefficient();
        1.0 / (self.absorption_coefficient_cm + mu_s_prime)
    }

    /// Scattering mean free path: l_s = 1 / μs  (μm)
    ///
    /// Converted from cm to μm (×10 000).
    pub fn scattering_mfp_um(&self) -> f64 {
        1.0 / self.scattering_coefficient_cm * 1.0e4
    }

    /// Effective attenuation coefficient: μeff = √(3 · μa · (μa + μs'))  (cm⁻¹)
    pub fn effective_attenuation_cm(&self) -> f64 {
        let mu_s_prime = self.reduced_scattering_coefficient();
        (3.0 * self.absorption_coefficient_cm * (self.absorption_coefficient_cm + mu_s_prime))
            .sqrt()
    }

    /// Penetration depth: δ = 1 / μeff  (cm)
    pub fn penetration_depth_cm(&self) -> f64 {
        1.0 / self.effective_attenuation_cm()
    }

    /// Single-scattering albedo: a = μs / (μa + μs)  (dimensionless)
    pub fn albedo(&self) -> f64 {
        self.scattering_coefficient_cm
            / (self.absorption_coefficient_cm + self.scattering_coefficient_cm)
    }

    /// Diffusion coefficient: D = 1 / (3 · (μa + μs'))  (cm)
    pub fn diffusion_coefficient_cm(&self) -> f64 {
        let mu_s_prime = self.reduced_scattering_coefficient();
        1.0 / (3.0 * (self.absorption_coefficient_cm + mu_s_prime))
    }

    /// Henyey-Greenstein phase function.
    ///
    /// p(θ) = (1 − g²) / (4π · (1 + g² − 2g·cosθ)^{3/2})
    ///
    /// Gives the probability density (per steradian) for scattering
    /// through polar angle θ.
    pub fn phase_function(&self, theta_rad: f64) -> f64 {
        let g = self.anisotropy_factor;
        let cos_theta = theta_rad.cos();
        let g2 = g * g;
        let denom = (1.0 + g2 - 2.0 * g * cos_theta).powf(1.5);
        (1.0 - g2) / (4.0 * std::f64::consts::PI * denom)
    }
}

/// Diffusion approximation solver for light transport in turbid media.
///
/// Solves the steady-state diffusion equation:
///   ∇²φ − μeff² · φ = −3D · S
/// where φ is the fluence rate (W/cm²) and S is the isotropic source term (W/cm³).
pub struct DiffusionModel {
    /// Tissue optical properties
    pub tissue: TissueOpticalProperties,
    /// Physical domain size in cm: (x, y, z)
    pub domain_size_cm: (f64, f64, f64),
}

impl DiffusionModel {
    /// Create a new diffusion model with given tissue and domain.
    pub fn new(tissue: TissueOpticalProperties, domain_cm: (f64, f64, f64)) -> Self {
        Self {
            tissue,
            domain_size_cm: domain_cm,
        }
    }

    /// Fluence rate from an isotropic point source at depth z₀.
    ///
    /// Uses the Green's function solution for an infinite homogeneous medium:
    ///   φ(r) = P / (4π D r) · exp(−μeff · r)
    /// where r is the 3-D distance from the source point.
    ///
    /// # Arguments
    /// * `source_depth_cm` — z-coordinate of the source (cm)
    /// * `source_power_mw` — source power in milliwatts
    /// * `observation_point` — (x, y, z) coordinates in cm
    ///
    /// # Returns
    /// Fluence rate in mW/cm².
    pub fn point_source_fluence(
        &self,
        source_depth_cm: f64,
        source_power_mw: f64,
        observation_point: (f64, f64, f64),
    ) -> f64 {
        let (ox, oy, oz) = observation_point;
        let r = (ox * ox + oy * oy + (oz - source_depth_cm) * (oz - source_depth_cm)).sqrt();
        if r < 1.0e-12 {
            // Avoid singularity at source location
            return f64::INFINITY;
        }
        let mu_eff = self.tissue.effective_attenuation_cm();
        let d = self.tissue.diffusion_coefficient_cm();
        source_power_mw / (4.0 * std::f64::consts::PI * d * r) * (-mu_eff * r).exp()
    }

    /// Semi-infinite slab solution for surface illumination.
    ///
    /// Uses extrapolated boundary condition (EBC): the fluence vanishes at
    /// z = −z_b where z_b = 2 · A · D, with A ≈ 3.0 for tissue-air interface.
    ///
    /// For a collimated beam converted to isotropic source at depth z₀ = 1/μt:
    ///   φ(z) ≈ 3D·E₀ · \[exp(−μeff·|z−z₀|) + exp(−μeff·(z+z₀+2z_b))\] / (2D·μeff)
    ///
    /// # Arguments
    /// * `z_cm` — observation depth (cm, measured from surface)
    /// * `source_power_per_cm2` — incident irradiance (mW/cm²)
    ///
    /// # Returns
    /// Fluence rate in mW/cm².
    pub fn slab_fluence(&self, z_cm: f64, source_power_per_cm2: f64) -> f64 {
        let mu_a = self.tissue.absorption_coefficient_cm;
        let mu_s_prime = self.tissue.reduced_scattering_coefficient();
        let mu_t = mu_a + mu_s_prime;
        let mu_eff = self.tissue.effective_attenuation_cm();
        let d = self.tissue.diffusion_coefficient_cm();

        // Virtual source depth from collimated-to-diffuse conversion
        let z0 = 1.0 / mu_t;

        // Extrapolated boundary depth
        let a_factor = (1.0 + self.tissue.refractive_index)
            / (1.0 - self.tissue.refractive_index + 1.0e-12).abs();
        // A ≈ 3.0 for n=1.37 tissue (standard Fresnel reflection factor)
        let a_boundary = if a_factor.is_finite() && a_factor > 0.0 {
            a_factor.min(10.0)
        } else {
            3.0
        };
        let z_b = 2.0 * a_boundary * d;

        // Point source and its image source (EBC method)
        let term1 = (-(mu_eff * (z_cm - z0).abs())).exp();
        let term2 = (-(mu_eff * (z_cm + z0 + 2.0 * z_b))).exp();

        // Diffusion source is 3D * mu_t * E0 (diffusion source strength)
        3.0 * d * mu_t * source_power_per_cm2 * (term1 + term2) / (2.0 * d * mu_eff)
    }

    /// Compute the 1-D steady-state depth profile of fluence rate.
    ///
    /// Uses the simplified plane-wave diffusion solution:
    ///   φ(z) = 3 D S exp(−μeff z) / μeff
    ///
    /// # Arguments
    /// * `source_irradiance` — incident irradiance at surface (mW/cm²)
    /// * `z_max_cm` — maximum depth (cm)
    /// * `n_pts` — number of depth points
    ///
    /// # Returns
    /// Vector of (depth_cm, fluence_rate_mW_per_cm2) pairs.
    pub fn depth_profile(
        &self,
        source_irradiance: f64,
        z_max_cm: f64,
        n_pts: usize,
    ) -> Vec<(f64, f64)> {
        let mu_eff = self.tissue.effective_attenuation_cm();
        let d = self.tissue.diffusion_coefficient_cm();
        let mu_s_prime = self.tissue.reduced_scattering_coefficient();
        let mu_a = self.tissue.absorption_coefficient_cm;
        // Isotropic source strength from collimated beam
        let s = (mu_a + mu_s_prime) * source_irradiance;

        (0..n_pts)
            .map(|i| {
                let z = z_max_cm * (i as f64) / ((n_pts - 1).max(1) as f64);
                let phi = 3.0 * d * s * (-mu_eff * z).exp() / mu_eff;
                (z, phi)
            })
            .collect()
    }

    /// Effective penetration depth: δ = 1/μeff (cm)
    pub fn effective_depth_cm(&self) -> f64 {
        self.tissue.penetration_depth_cm()
    }

    /// Absorbed power density at depth z: Q(z) = μa · φ(z)  (mW/cm³)
    pub fn absorbed_power_density(&self, z_cm: f64, source_irradiance: f64) -> f64 {
        let profile = self.depth_profile(source_irradiance, z_cm, 2);
        // Last point is at z_cm
        let phi_z = profile.last().map(|&(_, p)| p).unwrap_or(0.0);
        self.tissue.absorption_coefficient_cm * phi_z
    }

    /// Thermal damage integral (Arrhenius model).
    ///
    /// Ω = A · ∫ exp(−E_a / (R T(t))) dt
    ///
    /// Standard parameters for tissue coagulation:
    /// - A = 3.1e98 s⁻¹ (frequency factor)
    /// - E_a = 6.28e5 J/mol (activation energy)
    ///
    /// Ω > 1 indicates irreversible thermal damage.
    ///
    /// # Arguments
    /// * `temperature_history_k` — slice of (time_s, temperature_K) pairs
    ///
    /// # Returns
    /// Dimensionless damage integral Ω.
    pub fn thermal_damage_integral(&self, temperature_history_k: &[(f64, f64)]) -> f64 {
        // Standard Arrhenius parameters for tissue (Henriques & Moritz model)
        const FREQ_FACTOR: f64 = 3.1e98; // s⁻¹
        const ACTIVATION_ENERGY: f64 = 6.28e5; // J/mol

        if temperature_history_k.len() < 2 {
            return 0.0;
        }

        let mut omega = 0.0;
        for window in temperature_history_k.windows(2) {
            let (t0, temp0) = window[0];
            let (t1, temp1) = window[1];
            let dt = t1 - t0;
            if dt <= 0.0 {
                continue;
            }
            // Trapezoid rule for integration
            let rate0 = FREQ_FACTOR * (-ACTIVATION_ENERGY / (R_GAS * temp0)).exp();
            let rate1 = FREQ_FACTOR * (-ACTIVATION_ENERGY / (R_GAS * temp1)).exp();
            omega += 0.5 * (rate0 + rate1) * dt;
        }
        omega
    }

    /// Simplified 1-D Monte Carlo estimate of fluence rate.
    ///
    /// Launches `n_photons` unit-weight photons from z=0, tracks them
    /// through the tissue using exponential step-length sampling with
    /// Russian-roulette termination. Absorption is tallied as fluence.
    ///
    /// This is an approximate 1-D slab geometry (no lateral spread).
    ///
    /// # Arguments
    /// * `n_photons` — number of photon packets
    /// * `z_max_cm` — slab thickness (cm)
    /// * `n_bins` — number of depth bins for fluence accumulation
    ///
    /// # Returns
    /// Vector of (bin_center_cm, normalized_fluence) pairs.
    pub fn monte_carlo_fluence_1d(
        &self,
        n_photons: usize,
        z_max_cm: f64,
        n_bins: usize,
    ) -> Vec<(f64, f64)> {
        let mu_a = self.tissue.absorption_coefficient_cm;
        let mu_s = self.tissue.scattering_coefficient_cm;
        let mu_t = mu_a + mu_s;
        let bin_width = z_max_cm / n_bins as f64;
        let albedo = mu_s / mu_t;

        let mut fluence = vec![0.0_f64; n_bins];

        // Simple linear congruential generator (no rand crate dependency)
        let mut rng_state: u64 = 0x123456789ABCDEF0;
        let lcg_next = |state: &mut u64| -> f64 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*state >> 33) as f64) / (u32::MAX as f64)
        };

        for _ in 0..n_photons {
            let mut z = 0.0_f64;
            let mut weight = 1.0_f64;
            let mut direction = 1.0_f64; // +z direction (downward)

            loop {
                // Sample step length from exponential distribution
                let u = lcg_next(&mut rng_state).max(1.0e-15);
                let step = -u.ln() / mu_t;
                let z_new = z + direction * step;

                // Check boundary: z < 0 (surface, specular reflection) or z > z_max (transmitted)
                if z_new < 0.0 {
                    // Reflect at surface (simplified: kill photon)
                    break;
                }
                if z_new > z_max_cm {
                    // Photon transmitted through slab
                    break;
                }

                z = z_new;

                // Deposit absorbed weight into bin
                let bin_idx = ((z / z_max_cm) * n_bins as f64) as usize;
                let bin_idx = bin_idx.min(n_bins - 1);
                let absorbed_weight = weight * (1.0 - albedo);
                fluence[bin_idx] += absorbed_weight;
                weight *= albedo;

                // Russian roulette
                if weight < 0.01 {
                    let u_rr = lcg_next(&mut rng_state);
                    if u_rr < 0.1 {
                        weight /= 0.1;
                    } else {
                        break;
                    }
                }

                // Scatter: Henyey-Greenstein cosine sampling
                let u_cos = lcg_next(&mut rng_state);
                let g = self.tissue.anisotropy_factor;
                let cos_theta = if g.abs() < 1.0e-6 {
                    1.0 - 2.0 * u_cos
                } else {
                    let s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u_cos);
                    (1.0 + g * g - s * s) / (2.0 * g)
                };
                direction *= cos_theta;
                // Clamp to [-1, 1]
                direction = direction.clamp(-1.0, 1.0);
            }
        }

        // Normalize by number of photons and bin volume
        let norm = n_photons as f64 * bin_width;
        (0..n_bins)
            .map(|i| {
                let z_center = (i as f64 + 0.5) * bin_width;
                (z_center, fluence[i] / norm)
            })
            .collect()
    }
}

/// Hemoglobin oxygenation model for tissue spectroscopy.
///
/// Models the combined absorption of oxygenated (HbO2) and deoxygenated (Hb)
/// hemoglobin, enabling calculation of tissue absorption and SpO2 estimation.
pub struct HemoglobinModel {
    /// Total hemoglobin concentration (g/dL)
    pub hemoglobin_concentration_g_per_dl: f64,
    /// Oxygen saturation (SpO2), dimensionless in \[0, 1\]
    pub oxygen_saturation: f64,
}

impl HemoglobinModel {
    /// Create a hemoglobin model with the given concentration and SpO2.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] if concentration ≤ 0 or SpO2 ∉ \[0, 1\].
    pub fn new(hb_conc_g_dl: f64, spo2: f64) -> Result<Self, OxiPhotonError> {
        if hb_conc_g_dl <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "Hemoglobin concentration must be positive, got {}",
                hb_conc_g_dl
            )));
        }
        if !(0.0..=1.0).contains(&spo2) {
            return Err(OxiPhotonError::NumericalError(format!(
                "SpO2 must be in [0, 1], got {}",
                spo2
            )));
        }
        Ok(Self {
            hemoglobin_concentration_g_per_dl: hb_conc_g_dl,
            oxygen_saturation: spo2,
        })
    }

    /// Construct a model representing normal healthy adult blood.
    ///
    /// Hb = 15 g/dL, SpO2 = 0.98
    pub fn normal_blood() -> Self {
        Self {
            hemoglobin_concentration_g_per_dl: 15.0,
            oxygen_saturation: 0.98,
        }
    }

    /// Compute tissue absorption coefficient due to hemoglobin.
    ///
    /// μa = (ε_HbO2 · SpO2 + ε_Hb · (1 − SpO2)) · c_M
    ///
    /// where c_M is the molar concentration (mol/L).
    /// Molecular weight of hemoglobin tetramer ≈ 64 458 g/mol.
    pub fn absorption_coefficient_cm(&self, lambda_nm: f64) -> f64 {
        let mw_hb = 64_458.0; // g/mol (hemoglobin tetramer)
                              // Convert g/dL → g/L → mol/L
        let conc_mol_per_l = (self.hemoglobin_concentration_g_per_dl * 10.0) / mw_hb;
        let eps_hbo2 = Self::epsilon_hbo2_cm_per_m(lambda_nm);
        let eps_hb = Self::epsilon_hb_cm_per_m(lambda_nm);
        let eps_mixed = eps_hbo2 * self.oxygen_saturation + eps_hb * (1.0 - self.oxygen_saturation);
        // Beer-Lambert: μa (cm⁻¹) = ε (M⁻¹cm⁻¹) × c (M)
        eps_mixed * conc_mol_per_l
    }

    /// Molar extinction coefficient of oxygenated hemoglobin (HbO2) in M⁻¹cm⁻¹.
    ///
    /// Values are tabulated from Prahl's website (Oregon Medical Laser Center)
    /// and interpolated for the given wavelength using a piecewise linear model
    /// of key spectral features.
    pub fn epsilon_hbo2_cm_per_m(lambda_nm: f64) -> f64 {
        // Tabulated HbO2 extinction coefficients at selected wavelengths (nm → M⁻¹cm⁻¹)
        // Source: Prahl (1999), Oregon Medical Laser Center
        let table: &[(f64, f64)] = &[
            (600.0, 1214.0),
            (620.0, 808.0),
            (630.0, 742.0),
            (640.0, 830.0),
            (660.0, 1214.0),
            (680.0, 692.0),
            (700.0, 476.0),
            (720.0, 392.0),
            (740.0, 392.0),
            (760.0, 1060.0),
            (780.0, 1528.0),
            (800.0, 1068.0),
            (820.0, 756.0),
            (840.0, 542.0),
            (860.0, 428.0),
            (880.0, 380.0),
            (900.0, 372.0),
            (920.0, 420.0),
            (940.0, 476.0),
            (960.0, 536.0),
        ];
        piecewise_linear_interp(table, lambda_nm)
    }

    /// Molar extinction coefficient of deoxygenated hemoglobin (Hb) in M⁻¹cm⁻¹.
    pub fn epsilon_hb_cm_per_m(lambda_nm: f64) -> f64 {
        // Tabulated Hb extinction coefficients at selected wavelengths (nm → M⁻¹cm⁻¹)
        let table: &[(f64, f64)] = &[
            (600.0, 7140.0),
            (620.0, 3276.0),
            (630.0, 2288.0),
            (640.0, 1680.0),
            (660.0, 906.0),
            (680.0, 536.0),
            (700.0, 364.0),
            (720.0, 282.0),
            (740.0, 274.0),
            (760.0, 392.0),
            (780.0, 584.0),
            (800.0, 1068.0), // Isosbestic point
            (820.0, 1350.0),
            (840.0, 1304.0),
            (860.0, 1148.0),
            (880.0, 1028.0),
            (900.0, 922.0),
            (920.0, 836.0),
            (940.0, 756.0),
            (960.0, 680.0),
        ];
        piecewise_linear_interp(table, lambda_nm)
    }

    /// Isosbestic wavelength where HbO2 and Hb have equal extinction (≈ 800 nm).
    ///
    /// At this wavelength, absorption is independent of oxygenation state,
    /// enabling absolute concentration measurements.
    pub fn isosbestic_wavelength_nm() -> f64 {
        800.0
    }

    /// Estimate SpO2 from the ratio of optical densities at 660 and 940 nm.
    ///
    /// This is the principle behind pulse oximetry:
    ///   R = (OD at 660) / (OD at 940)
    ///
    /// Empirical calibration: SpO2 ≈ 110 − 25R  (simplified linear model)
    pub fn spo2_from_ratio(ratio_660_940: f64) -> f64 {
        // Simplified empirical model (Moyle 2002)
        let spo2 = 1.10 - 0.25 * ratio_660_940;
        spo2.clamp(0.0, 1.0)
    }
}

/// Photodynamic therapy (PDT) dosimetry model.
///
/// Computes light dosimetry, singlet oxygen generation rate, and
/// effective treatment depth for PDT using a photosensitizer-tissue system.
pub struct PdtDosimetry {
    /// Tissue optical properties at the treatment wavelength
    pub tissue: TissueOpticalProperties,
    /// Photosensitizer concentration (μM = micromolar)
    pub photosensitizer_concentration_um: f64,
    /// Molar absorption coefficient of photosensitizer at treatment wavelength (M⁻¹cm⁻¹)
    pub photosensitizer_epsilon_cm: f64,
    /// Singlet oxygen quantum yield φ_Δ (photons absorbed → ¹O₂ generated)
    pub quantum_yield_singlet_oxygen: f64,
    /// Incident irradiance at tissue surface (mW/cm²)
    pub irradiance_mw_per_cm2: f64,
}

impl PdtDosimetry {
    /// Create a PDT dosimetry model.
    pub fn new(
        tissue: TissueOpticalProperties,
        conc_um: f64,
        epsilon: f64,
        phi_delta: f64,
        irradiance: f64,
    ) -> Self {
        Self {
            tissue,
            photosensitizer_concentration_um: conc_um,
            photosensitizer_epsilon_cm: epsilon,
            quantum_yield_singlet_oxygen: phi_delta,
            irradiance_mw_per_cm2: irradiance,
        }
    }

    /// Standard Photofrin PDT at 630 nm with given irradiance.
    ///
    /// Photofrin parameters: ε ≈ 1170 M⁻¹cm⁻¹ at 630 nm, φ_Δ ≈ 0.89,
    /// typical clinical concentration ≈ 2 μM in tissue.
    pub fn photofrin_630nm(irradiance_mw: f64) -> Self {
        Self::new(
            TissueOpticalProperties::skin_dermis_630nm(),
            2.0,    // 2 μM photosensitizer concentration
            1170.0, // M⁻¹cm⁻¹
            0.89,   // singlet oxygen quantum yield
            irradiance_mw,
        )
    }

    /// Treatment dose delivered at the surface after time t.
    ///
    /// D = E₀ · t  (J/cm²) where E₀ is the irradiance (mW/cm² → W/cm² conversion).
    pub fn treatment_dose_j_per_cm2(&self, time_s: f64) -> f64 {
        self.irradiance_mw_per_cm2 * 1.0e-3 * time_s
    }

    /// Singlet oxygen generation rate at the tissue surface.
    ///
    /// Rate = φ_Δ · \[PS\]_M · ε · I / (h·ν)  (μM/s)
    ///
    /// where \[PS\]_M is molar concentration, I is photon flux (photons·cm⁻²·s⁻¹).
    pub fn singlet_oxygen_rate_um_per_s(&self) -> f64 {
        let lambda_m = self.tissue.wavelength_nm * 1.0e-9;
        let photon_energy_j = H_PLANCK * C0 / lambda_m;
        // Convert mW/cm² → W/cm², then to photon flux
        let photon_flux = self.irradiance_mw_per_cm2 * 1.0e-3 / photon_energy_j;
        // Concentration in mol/L (μM → M: ÷ 1e6)
        let conc_m = self.photosensitizer_concentration_um * 1.0e-6;
        // Absorption rate = ε · c_M (cm⁻¹) × photon_flux (photons/cm²/s)
        // But ε is in M⁻¹cm⁻¹, conc in M → ε·c in cm⁻¹ is absorption per path length
        // Rate per unit volume: ε_cm · c_M · photon_flux × φ_Δ × N_A (in μM/s)
        let abs_rate_per_vol = self.photosensitizer_epsilon_cm * conc_m * photon_flux;
        // Convert to μM/s: abs_rate (mol/L/s) × 1e6
        self.quantum_yield_singlet_oxygen * abs_rate_per_vol * 1.0e6 / AVOGADRO
    }

    /// Fluence rate at depth z below the tissue surface.
    ///
    /// Uses Beer-Lambert attenuation with effective attenuation coefficient:
    ///   I(z) = I₀ · exp(−μeff · z)
    pub fn fluence_rate_at_depth(&self, z_cm: f64) -> f64 {
        let mu_eff = self.tissue.effective_attenuation_cm();
        self.irradiance_mw_per_cm2 * (-mu_eff * z_cm).exp()
    }

    /// Effective treatment depth where accumulated fluence equals threshold.
    ///
    /// Solves: threshold = I₀ · exp(−μeff · z_eff) · t_treat
    ///   z_eff = −ln(threshold / (I₀ · t)) / μeff
    ///
    /// Returns depth at which the fluence rate equals the threshold level,
    /// normalized by treatment time = 1 s (for threshold in J/cm²).
    pub fn effective_treatment_depth_cm(&self, threshold_j_per_cm2: f64) -> f64 {
        let mu_eff = self.tissue.effective_attenuation_cm();
        // threshold = I0 (W/cm²) × exp(-μeff·z)
        let i0_w = self.irradiance_mw_per_cm2 * 1.0e-3;
        if threshold_j_per_cm2 <= 0.0 || i0_w <= 0.0 {
            return 0.0;
        }
        let ratio = threshold_j_per_cm2 / i0_w;
        if ratio >= 1.0 {
            return 0.0;
        }
        -ratio.ln() / mu_eff
    }
}

/// Piecewise linear interpolation over a sorted (x, y) table.
///
/// Clamps to boundary values for out-of-range inputs.
fn piecewise_linear_interp(table: &[(f64, f64)], x: f64) -> f64 {
    if table.is_empty() {
        return 0.0;
    }
    if x <= table[0].0 {
        return table[0].1;
    }
    if x >= table[table.len() - 1].0 {
        return table[table.len() - 1].1;
    }
    for i in 0..table.len() - 1 {
        let (x0, y0) = table[i];
        let (x1, y1) = table[i + 1];
        if x >= x0 && x <= x1 {
            let t = (x - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }
    table[table.len() - 1].1
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1.0e-10;

    #[test]
    fn test_reduced_scattering() {
        let tissue = TissueOpticalProperties::skin_dermis_630nm();
        // μs' = μs * (1 - g) = 200 * (1 - 0.9) = 20
        let mu_s_prime = tissue.reduced_scattering_coefficient();
        let expected = tissue.scattering_coefficient_cm * (1.0 - tissue.anisotropy_factor);
        assert!(
            (mu_s_prime - expected).abs() < TOL,
            "Expected {expected}, got {mu_s_prime}"
        );
        assert!((mu_s_prime - 20.0).abs() < 1.0e-10);
    }

    #[test]
    fn test_effective_attenuation() {
        let tissue = TissueOpticalProperties::skin_dermis_630nm();
        // μa=0.2, μs'=20 → μeff = sqrt(3*0.2*(0.2+20)) = sqrt(12.12) ≈ 3.481
        let mu_eff = tissue.effective_attenuation_cm();
        let mu_s_prime = tissue.reduced_scattering_coefficient();
        let expected = (3.0
            * tissue.absorption_coefficient_cm
            * (tissue.absorption_coefficient_cm + mu_s_prime))
            .sqrt();
        assert!((mu_eff - expected).abs() < TOL);
        assert!(mu_eff > 0.0);
    }

    #[test]
    fn test_penetration_depth() {
        let tissue = TissueOpticalProperties::skin_dermis_630nm();
        let delta = tissue.penetration_depth_cm();
        let mu_eff = tissue.effective_attenuation_cm();
        assert!((delta - 1.0 / mu_eff).abs() < TOL);
        assert!(delta > 0.0);
    }

    #[test]
    fn test_phase_function_normalized() {
        // ∫₀^π p(θ) sin(θ) dθ should equal 1/(2π) so that
        // ∫₀^π p(θ) sin(θ) 2π dθ = 1 (normalization over full sphere)
        let tissue = TissueOpticalProperties::skin_dermis_630nm();
        let n_pts = 10_000;
        let mut integral = 0.0;
        let d_theta = std::f64::consts::PI / n_pts as f64;
        for i in 0..n_pts {
            let theta = (i as f64 + 0.5) * d_theta;
            integral += tissue.phase_function(theta) * theta.sin() * d_theta;
        }
        // The integral should equal 1/(2π) since the full integral is 2π × this
        let full_sphere_integral = 2.0 * std::f64::consts::PI * integral;
        assert!(
            (full_sphere_integral - 1.0).abs() < 1.0e-5,
            "Phase function normalization failed: got {full_sphere_integral}"
        );
    }

    #[test]
    fn test_diffusion_depth_profile_decreasing() {
        let tissue = TissueOpticalProperties::skin_dermis_630nm();
        let model = DiffusionModel::new(tissue, (1.0, 1.0, 2.0));
        let profile = model.depth_profile(100.0, 2.0, 20);
        // Fluence rate must be strictly decreasing with depth
        for i in 1..profile.len() {
            assert!(
                profile[i].1 <= profile[i - 1].1,
                "Profile not decreasing at index {i}: {} > {}",
                profile[i].1,
                profile[i - 1].1
            );
        }
        // Also check that it decreases significantly overall
        let phi_0 = profile[0].1;
        let phi_last = profile[profile.len() - 1].1;
        assert!(
            phi_last < phi_0 * 0.5,
            "Fluence should decrease substantially with depth"
        );
    }

    #[test]
    fn test_hemoglobin_normal_blood_spo2() {
        let hb = HemoglobinModel::normal_blood();
        assert!((hb.oxygen_saturation - 0.98).abs() < TOL);
        assert!((hb.hemoglobin_concentration_g_per_dl - 15.0).abs() < TOL);
    }

    #[test]
    fn test_isosbestic_wavelength() {
        let lambda = HemoglobinModel::isosbestic_wavelength_nm();
        // Isosbestic wavelength should be approximately 800 nm
        assert!(
            (lambda - 800.0).abs() < 10.0,
            "Isosbestic wavelength should be ~800 nm, got {lambda}"
        );
        // At isosbestic point, epsilon_HbO2 ≈ epsilon_Hb
        let eps_hbo2 = HemoglobinModel::epsilon_hbo2_cm_per_m(lambda);
        let eps_hb = HemoglobinModel::epsilon_hb_cm_per_m(lambda);
        let rel_diff = (eps_hbo2 - eps_hb).abs() / eps_hbo2;
        assert!(
            rel_diff < 0.05,
            "At isosbestic wavelength, ε_HbO2 ({eps_hbo2}) ≈ ε_Hb ({eps_hb})"
        );
    }

    #[test]
    fn test_pdt_dose_increases_with_time() {
        let pdt = PdtDosimetry::photofrin_630nm(100.0);
        let dose_10s = pdt.treatment_dose_j_per_cm2(10.0);
        let dose_20s = pdt.treatment_dose_j_per_cm2(20.0);
        let dose_100s = pdt.treatment_dose_j_per_cm2(100.0);
        assert!(dose_20s > dose_10s, "Dose should increase with time");
        assert!(dose_100s > dose_20s, "Dose should increase with time");
        // At 100 mW/cm², 10 s → 1 J/cm²
        assert!((dose_10s - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn test_monte_carlo_total_absorbed() {
        // Run a small MC simulation and verify that total absorbed energy is bounded
        let tissue = TissueOpticalProperties::muscle_630nm();
        let model = DiffusionModel::new(tissue, (1.0, 1.0, 1.0));
        let profile = model.monte_carlo_fluence_1d(1000, 1.0, 10);
        // Check that all fluence values are non-negative
        for (_, phi) in &profile {
            assert!(*phi >= 0.0, "Fluence must be non-negative");
        }
        // The profile should have non-zero total absorbed energy
        let total: f64 = profile.iter().map(|(_, p)| p).sum();
        assert!(total > 0.0, "Monte Carlo must deposit some energy");
    }
}
