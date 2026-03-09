//! Photonic chemical and bio-sensing models.
//!
//! Covers three principal transduction mechanisms:
//! - Evanescent-field absorption in rib/strip waveguides.
//! - Whispering-gallery-mode (WGM) microresonator biosensors.
//! - Surface plasmon resonance (SPR) sensors (Kretschmann configuration).

use num_complex::Complex64;
use std::f64::consts::PI;

/// Speed of light in vacuum (m/s).
const C: f64 = 2.997_924_58e8;

// ---------------------------------------------------------------------------
// Evanescent Field Absorption Sensor
// ---------------------------------------------------------------------------

/// Evanescent-field absorption sensor based on a single-mode waveguide.
///
/// A fraction Γ of the guided mode power travels in the cladding as an
/// evanescent field.  When the cladding is replaced by an absorbing analyte
/// the effective absorption coefficient is `α_eff = Γ · α_sample`.
#[derive(Debug, Clone)]
pub struct EvanescentSensor {
    /// Waveguide length (m).
    pub waveguide_length_m: f64,
    /// Power confinement factor in the evanescent region (Γ, 0–1).
    pub evanescent_fraction: f64,
    /// Operating wavelength (m).
    pub wavelength: f64,
    /// Waveguide core refractive index.
    pub n_waveguide: f64,
    /// Cladding (analyte) refractive index.
    pub n_cladding: f64,
}

impl EvanescentSensor {
    /// Construct an evanescent sensor, estimating Γ from the V-number.
    ///
    /// The evanescent confinement fraction is approximated from the normalised
    /// frequency as `Γ ≈ 1 – (V – 1)⁻¹` (clamped to [0.01, 0.99]).
    ///
    /// # Arguments
    /// * `length_m` – Sensing interaction length (m).
    /// * `wavelength` – Operating wavelength (m).
    /// * `n_wg` – Waveguide core index.
    /// * `n_clad` – Cladding index.
    pub fn new(length_m: f64, wavelength: f64, n_wg: f64, n_clad: f64) -> Self {
        // Estimate V-number for a 400 nm wide waveguide (representative)
        let width = 400e-9_f64;
        let v_number = PI * width * (n_wg * n_wg - n_clad * n_clad).sqrt() / wavelength;
        let gamma = if v_number > 1.0 {
            (1.0 - 1.0 / v_number).clamp(0.01, 0.99)
        } else {
            0.30
        };
        Self {
            waveguide_length_m: length_m,
            evanescent_fraction: gamma,
            wavelength,
            n_waveguide: n_wg,
            n_cladding: n_clad,
        }
    }

    /// Effective absorption coefficient including the evanescent overlap.
    ///
    /// `α_eff = Γ · α_sample`
    ///
    /// # Arguments
    /// * `sample_absorption` – Bulk absorption coefficient of the analyte (m⁻¹).
    ///
    /// # Returns
    /// Effective propagation loss (m⁻¹).
    pub fn effective_absorption(&self, sample_absorption: f64) -> f64 {
        self.evanescent_fraction * sample_absorption
    }

    /// Absorbance (dimensionless, in absorbance units = –log₁₀ transmission).
    ///
    /// `A = α_eff · L / ln(10)`
    ///
    /// # Arguments
    /// * `sample_absorption_per_m` – Bulk analyte absorption coefficient (m⁻¹).
    ///
    /// # Returns
    /// Absorbance (AU).
    pub fn absorbance(&self, sample_absorption_per_m: f64) -> f64 {
        self.effective_absorption(sample_absorption_per_m) * self.waveguide_length_m / 10.0_f64.ln()
    }

    /// Minimum detectable concentration using the Lambert–Beer law.
    ///
    /// At the detection limit the absorbance equals `min_detectable_abs`:
    /// `c_min = min_detectable_abs / (ε · α_eff · L / ln(10))`
    ///
    /// # Arguments
    /// * `molar_absorption` – Molar absorption coefficient ε (L·mol⁻¹·cm⁻¹).
    /// * `min_detectable_abs` – Minimum detectable absorbance change (AU).
    ///
    /// # Returns
    /// Detection limit (mol/L).
    pub fn detection_limit_mol_per_l(&self, molar_absorption: f64, min_detectable_abs: f64) -> f64 {
        // Convert ε from L/(mol·cm) to m⁻¹/(mol/m³)
        // ε [L/mol/cm] → α [m⁻¹] per 1 mol/L: α = ε * ln(10) * 1000 / 100
        let epsilon_si = molar_absorption * 10.0_f64.ln() * 10.0; // m⁻¹ per mol/L
        let effective_path = self.evanescent_fraction * self.waveguide_length_m * 100.0; // cm
        if effective_path <= 0.0 || molar_absorption <= 0.0 {
            return f64::INFINITY;
        }
        min_detectable_abs
            / (epsilon_si * effective_path / (10.0_f64.ln() * 100.0 * self.waveguide_length_m)
                * self.waveguide_length_m
                * self.evanescent_fraction
                / 10.0_f64.ln())
    }

    /// Enhancement factor relative to a free-space absorption cell of the
    /// specified path length.
    ///
    /// `E = Γ · L_wg / L_free`
    ///
    /// # Arguments
    /// * `free_space_path_m` – Equivalent free-space path length (m).
    ///
    /// # Returns
    /// Dimensionless enhancement factor.
    pub fn enhancement_factor(&self, free_space_path_m: f64) -> f64 {
        if free_space_path_m <= 0.0 {
            return 0.0;
        }
        self.evanescent_fraction * self.waveguide_length_m / free_space_path_m
    }
}

// ---------------------------------------------------------------------------
// Whispering Gallery Mode Biosensor
// ---------------------------------------------------------------------------

/// Whispering-gallery-mode (WGM) microresonator biosensor.
///
/// The resonance wavelength shifts as the local refractive index changes due
/// to adsorbed analyte molecules.
#[derive(Debug, Clone)]
pub struct WgmBiosensor {
    /// Resonator radius (μm).
    pub resonator_radius_um: f64,
    /// Quality factor.
    pub q_factor: f64,
    /// Operating wavelength (m).
    pub wavelength: f64,
    /// Resonator material refractive index.
    pub n_resonator: f64,
    /// Background medium refractive index.
    pub n_medium: f64,
}

impl WgmBiosensor {
    /// Construct a WGM biosensor with default silica indices.
    ///
    /// # Arguments
    /// * `radius_um` – Resonator radius (μm).
    /// * `q` – Quality factor.
    /// * `wavelength` – Operating wavelength (m).
    pub fn new(radius_um: f64, q: f64, wavelength: f64) -> Self {
        Self {
            resonator_radius_um: radius_um,
            q_factor: q,
            wavelength,
            n_resonator: 1.44,
            n_medium: 1.33,
        }
    }

    /// Bulk refractive-index sensitivity.
    ///
    /// `S = dλ/dn = λ · (n_medium) / n_resonator` [nm/RIU]  (first-order approx.)
    ///
    /// # Returns
    /// Sensitivity (nm/RIU).
    pub fn bulk_sensitivity_nm_per_riu(&self) -> f64 {
        self.wavelength * 1e9 * self.n_medium / self.n_resonator
    }

    /// Limit of detection in refractive index units.
    ///
    /// `LOD = σ_noise / S`
    ///
    /// # Arguments
    /// * `noise_pm` – Resonance wavelength tracking noise (pm RMS).
    ///
    /// # Returns
    /// Detection limit (RIU).
    pub fn limit_of_detection_riu(&self, noise_pm: f64) -> f64 {
        let s_nm = self.bulk_sensitivity_nm_per_riu();
        if s_nm <= 0.0 {
            return f64::INFINITY;
        }
        (noise_pm * 1e-3) / s_nm
    }

    /// Estimated wavelength shift for a single adsorbed molecule.
    ///
    /// Approximated as:
    /// `Δλ/λ ≈ α_mol / (n_r² · V_mode)` (normalised)
    ///
    /// with `V_mode ≈ (λ/n_r)³`.
    ///
    /// # Arguments
    /// * `molecule_polarizability_nm3` – Excess polarizability of the molecule
    ///   (nm³; volume × (n_mol² – n_bg²)).
    ///
    /// # Returns
    /// Resonance wavelength shift (pm).
    pub fn single_molecule_shift_pm(&self, molecule_polarizability_nm3: f64) -> f64 {
        let lambda_nm = self.wavelength * 1e9;
        let v_mode_nm3 = (lambda_nm / self.n_resonator).powi(3);
        if v_mode_nm3 <= 0.0 {
            return 0.0;
        }
        // Δλ/λ = α / (n² · V_mode)
        let delta_lambda_over_lambda =
            molecule_polarizability_nm3 / (self.n_resonator * self.n_resonator * v_mode_nm3);
        delta_lambda_over_lambda * lambda_nm * 1e3 // convert nm → pm
    }

    /// Resonance linewidth (FWHM).
    ///
    /// `Δλ = λ / Q`
    ///
    /// # Returns
    /// Linewidth (pm).
    pub fn linewidth_pm(&self) -> f64 {
        self.wavelength * 1e12 / self.q_factor
    }

    /// Resonance tracking noise limited by photon shot noise.
    ///
    /// `σ = Δλ / (2.355 · √N_photons)` where `N_photons = Φ · τ`.
    ///
    /// # Arguments
    /// * `photon_flux` – Photon flux impinging on the detector (photons/s).
    /// * `integration_time_s` – Integration time (s).
    ///
    /// # Returns
    /// Resonance position uncertainty (pm RMS).
    pub fn resonance_tracking_noise_pm(&self, photon_flux: f64, integration_time_s: f64) -> f64 {
        let n_photons = photon_flux * integration_time_s;
        if n_photons <= 0.0 {
            return f64::INFINITY;
        }
        self.linewidth_pm() / (2.355 * n_photons.sqrt())
    }
}

// ---------------------------------------------------------------------------
// Surface Plasmon Resonance Sensor
// ---------------------------------------------------------------------------

/// Metal type for SPR sensor.
#[derive(Debug, Clone, PartialEq)]
pub enum SprMetal {
    Gold,
    Silver,
    Copper,
    Aluminum,
}

impl SprMetal {
    /// Drude-model permittivity at a given wavelength.
    ///
    /// Uses tabulated Drude parameters for each metal.  The Drude model is:
    /// `ε(ω) = ε_∞ - ωp²/(ω² + iγω)`.
    ///
    /// # Arguments
    /// * `wavelength_nm` – Free-space wavelength (nm).
    ///
    /// # Returns
    /// Complex permittivity ε = ε₁ + iε₂.
    pub fn permittivity(&self, wavelength_nm: f64) -> Complex64 {
        // Angular frequency of light
        let omega = 2.0 * PI * C / (wavelength_nm * 1e-9);
        let (eps_inf, omega_p, gamma) = match self {
            SprMetal::Gold => (9.5, 1.37e16, 1.45e14),
            SprMetal::Silver => (5.0, 1.38e16, 2.73e13),
            SprMetal::Copper => (9.0, 1.36e16, 1.45e14),
            SprMetal::Aluminum => (1.0, 2.24e16, 1.22e14),
        };
        let omega2 = omega * omega;
        let real_part = eps_inf - omega_p * omega_p / (omega2 + gamma * gamma);
        let imag_part = omega_p * omega_p * gamma / (omega * (omega2 + gamma * gamma));
        Complex64::new(real_part, imag_part)
    }
}

/// Surface plasmon resonance (SPR) sensor in Kretschmann configuration.
///
/// A thin metal film on a prism couples evanescent light into a surface
/// plasmon at the metal–sample interface.  The resonance angle is extremely
/// sensitive to changes in the near-surface refractive index.
#[derive(Debug, Clone)]
pub struct SprSensor {
    /// Metal film thickness (nm).
    pub metal_thickness_nm: f64,
    /// Prism refractive index.
    pub prism_index: f64,
    /// Operating wavelength (m).
    pub wavelength: f64,
    /// Metal type.
    pub metal_type: SprMetal,
}

impl SprSensor {
    /// Construct an SPR sensor.
    ///
    /// # Arguments
    /// * `metal_type` – Metal choice.
    /// * `thickness_nm` – Metal film thickness (nm).
    /// * `wavelength` – Operating wavelength (m).
    pub fn new(metal_type: SprMetal, thickness_nm: f64, wavelength: f64) -> Self {
        Self {
            metal_thickness_nm: thickness_nm,
            prism_index: 1.515,
            wavelength,
            metal_type,
        }
    }

    /// Surface plasmon wave vector at the metal–sample interface.
    ///
    /// `k_sp = k₀ · √(ε_m · n_s² / (ε_m + n_s²))`
    ///
    /// For a propagating SPR we need `ε_m · n_s² / (ε_m + n_s²) > 0`.
    /// With a Drude metal `ε_r < 0` and `|ε_r| > n_s²`, giving a negative
    /// denominator and a negative numerator, so the ratio is positive.
    fn k_sp(&self, n_sample: f64) -> f64 {
        let eps_m = self.metal_type.permittivity(self.wavelength * 1e9);
        let ns2 = n_sample * n_sample;
        let eps_r = eps_m.re;
        let denom = eps_r + ns2;
        // SPR requires the ratio to be positive (propagating plasmon)
        let product = if denom.abs() < f64::EPSILON {
            return 0.0;
        } else {
            eps_r * ns2 / denom
        };
        if product <= 0.0 {
            return 0.0;
        }
        let k0 = 2.0 * PI / self.wavelength;
        k0 * product.sqrt()
    }

    /// SPR resonance angle (degrees) in the prism.
    ///
    /// Phase-matching condition: `k_sp = k_0 · n_prism · sin(θ_SPR)`
    ///
    /// # Arguments
    /// * `n_sample` – Refractive index of the sample (dielectric).
    ///
    /// # Returns
    /// Resonance angle (degrees) inside the prism; `NaN` if no SPR possible.
    pub fn resonance_angle_deg(&self, n_sample: f64) -> f64 {
        let k0 = 2.0 * PI / self.wavelength;
        let k_sp_val = self.k_sp(n_sample);
        if k_sp_val <= 0.0 {
            return f64::NAN;
        }
        let sin_theta = k_sp_val / (k0 * self.prism_index);
        if sin_theta.abs() > 1.0 {
            return f64::NAN;
        }
        sin_theta.asin().to_degrees()
    }

    /// Bulk sensitivity `dθ/dn` at the nominal sample index `n = 1.333`.
    ///
    /// Computed numerically with Δn = 1e-4.
    ///
    /// # Returns
    /// Sensitivity (degrees/RIU).
    pub fn bulk_sensitivity_deg_per_riu(&self) -> f64 {
        let n0 = 1.333_f64;
        let dn = 1e-4_f64;
        let theta_p = self.resonance_angle_deg(n0 + dn);
        let theta_m = self.resonance_angle_deg(n0 - dn);
        if theta_p.is_nan() || theta_m.is_nan() {
            return 0.0;
        }
        (theta_p - theta_m) / (2.0 * dn)
    }

    /// Minimum detectable refractive index change.
    ///
    /// `Δn_min = angular_resolution / sensitivity`
    ///
    /// # Arguments
    /// * `angular_resolution_mdeg` – Angular measurement resolution (mdeg).
    ///
    /// # Returns
    /// Minimum detectable Δn (RIU).
    pub fn minimum_detectable_delta_n(&self, angular_resolution_mdeg: f64) -> f64 {
        let s = self.bulk_sensitivity_deg_per_riu();
        if s.abs() < f64::EPSILON {
            return f64::INFINITY;
        }
        (angular_resolution_mdeg * 1e-3) / s
    }

    /// Approximate sensing range in RIU.
    ///
    /// Defined as the range of *n* values for which the SPR resonance angle
    /// remains within ±10° of the resonance angle at `n = 1.333`.
    ///
    /// # Returns
    /// Sensing range (RIU).
    pub fn sensing_range_riu(&self) -> f64 {
        let s = self.bulk_sensitivity_deg_per_riu();
        if s.abs() < f64::EPSILON {
            return 0.0;
        }
        20.0 / s
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn evanescent_effective_absorption() {
        let sensor = EvanescentSensor::new(0.01, 1550e-9, 2.0, 1.33);
        let alpha_eff = sensor.effective_absorption(100.0);
        assert!(alpha_eff > 0.0 && alpha_eff <= 100.0);
        assert_abs_diff_eq!(
            alpha_eff,
            sensor.evanescent_fraction * 100.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn evanescent_enhancement_factor() {
        let sensor = EvanescentSensor::new(0.10, 1310e-9, 3.5, 1.44);
        let e = sensor.enhancement_factor(0.001);
        // 10 cm waveguide vs 1 mm free space → at least 10× with Γ=1 upper bound
        assert!(e > 1.0, "Enhancement should exceed 1: {}", e);
    }

    #[test]
    fn wgm_linewidth_scaling() {
        // Higher Q → narrower linewidth
        let wgm_lo = WgmBiosensor::new(50.0, 1e6, 1550e-9);
        let wgm_hi = WgmBiosensor::new(50.0, 1e8, 1550e-9);
        assert!(wgm_hi.linewidth_pm() < wgm_lo.linewidth_pm());
        assert_abs_diff_eq!(
            wgm_lo.linewidth_pm() / wgm_hi.linewidth_pm(),
            100.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn wgm_sensitivity_positive() {
        let wgm = WgmBiosensor::new(50.0, 1e7, 1550e-9);
        assert!(wgm.bulk_sensitivity_nm_per_riu() > 0.0);
    }

    #[test]
    fn wgm_lod_decreases_with_less_noise() {
        let wgm = WgmBiosensor::new(50.0, 1e7, 1550e-9);
        let lod1 = wgm.limit_of_detection_riu(1.0);
        let lod2 = wgm.limit_of_detection_riu(10.0);
        assert!(lod1 < lod2, "LOD should improve with lower noise");
    }

    #[test]
    fn spr_gold_permittivity_sign() {
        let eps = SprMetal::Gold.permittivity(633.0);
        // Gold at 633 nm: ε₁ < 0 (metallic), ε₂ > 0
        assert!(
            eps.re < 0.0,
            "Gold ε₁ should be negative at 633 nm: {}",
            eps.re
        );
        assert!(eps.im > 0.0, "Gold ε₂ should be positive: {}", eps.im);
    }

    #[test]
    fn spr_resonance_angle_physical() {
        let spr = SprSensor::new(SprMetal::Gold, 50.0, 633e-9);
        let theta = spr.resonance_angle_deg(1.333);
        assert!(!theta.is_nan(), "Resonance angle should be valid");
        assert!(
            theta > 40.0 && theta < 75.0,
            "SPR angle out of typical range: {}°",
            theta
        );
    }

    #[test]
    fn spr_sensitivity_gold_reasonable() {
        let spr = SprSensor::new(SprMetal::Gold, 50.0, 633e-9);
        let s = spr.bulk_sensitivity_deg_per_riu();
        // Gold at 633 nm: typically 40–100 deg/RIU
        assert!(s > 10.0 && s < 300.0, "Sensitivity out of range: {}", s);
    }
}
