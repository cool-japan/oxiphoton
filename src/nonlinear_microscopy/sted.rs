//! Stimulated Emission Depletion (STED) Nanoscopy
//!
//! STED breaks the diffraction limit by using a donut-shaped depletion beam to
//! inhibit fluorescence outside a central subdiffraction region. The effective
//! PSF width scales as:
//!
//! `Δr_eff = Δr_conf / sqrt(1 + I_STED / I_sat)`
//!
//! where `I_sat` is the saturation intensity at which the depletion probability
//! is 50%. With I_STED/I_sat ≫ 1, resolution well below 50 nm is achievable.
//!
//! # References
//! - Hell & Wichmann, Opt. Lett. 19, 780 (1994)
//! - Klar et al., PNAS 97, 8206 (2000)
//! - Willig et al., Nat. Methods 3, 721 (2006)

use std::f64::consts::PI;

/// Speed of light in vacuum \[m/s\]
pub const C_LIGHT: f64 = 2.99792458e8;
/// Planck constant \[J·s\]
pub const H_PLANCK: f64 = 6.62607015e-34;
/// Reduced Planck constant \[J·s\]
pub const HBAR: f64 = 1.054571817e-34;
/// Boltzmann constant \[J/K\]
pub const KB: f64 = 1.380649e-23;

/// STED depletion beam (donut / helical phase vortex).
///
/// The donut profile is created by applying a spiral phase plate (e^{iφ})
/// to a Gaussian beam. The resulting intensity pattern has a zero on the
/// optical axis and a ring-shaped maximum.
#[derive(Debug, Clone)]
pub struct StedBeam {
    /// Total depletion beam power \[W\]
    pub power_w: f64,
    /// Depletion beam wavelength \[m\]
    pub wavelength_m: f64,
    /// Numerical aperture of the focusing objective
    pub na: f64,
    /// Refractive index of the immersion medium
    pub n_medium: f64,
}

impl StedBeam {
    /// Construct a STED depletion beam.
    pub fn new(power_w: f64, wavelength_m: f64, na: f64, n_medium: f64) -> Self {
        Self {
            power_w,
            wavelength_m,
            na,
            n_medium,
        }
    }

    /// Beam waist w₀ ≈ λ/(π NA/n) for the focused Gaussian (before helical phase) \[m\].
    pub fn beam_waist_m(&self) -> f64 {
        self.wavelength_m / (PI * self.na / self.n_medium)
    }

    /// Peak irradiance of the STED donut ring \[W/m²\].
    ///
    /// For a helical-phase donut, the ring maximum occurs at r ≈ w₀/√2 and the
    /// peak intensity is approximately:
    /// `I_peak ≈ 2 P / (π w₀²) × f_donut`
    ///
    /// where `f_donut ≈ 0.54` is a form factor accounting for the donut shape.
    pub fn peak_intensity_w_m2(&self) -> f64 {
        let w0 = self.beam_waist_m();
        let area = PI * w0 * w0;
        // Donut peak ~ 2P/πw₀² × 0.54 (helical beam form factor)
        2.0 * self.power_w / area * 0.54
    }

    /// STED donut intensity at radial distance r \[m\] and axial position z \[m\].
    ///
    /// Approximation for a helical-phase beam:
    /// `I(r, z) = I_peak · sin²(π r / r₀)² · exp(−2z²/w_z²)`
    ///
    /// where r₀ = beam_waist and w_z = Rayleigh length.
    pub fn donut_intensity(&self, r: f64, z: f64) -> f64 {
        let w0 = self.beam_waist_m();
        let z_r = PI * self.n_medium * w0 * w0 / self.wavelength_m;
        let i_peak = self.peak_intensity_w_m2();
        // Radial donut profile: sin²(π r / w₀)
        let radial = (PI * r / w0).sin().powi(2);
        // Axial Gaussian envelope (depletion beam also has axial extent)
        let axial = (-2.0 * z * z / (z_r * z_r)).exp();
        i_peak * radial * radial * axial
    }

    /// Effective lateral resolution after STED depletion \[m\].
    ///
    /// `Δr_eff = Δr_conf / sqrt(1 + I_STED_peak / I_sat)`
    ///
    /// # Arguments
    /// * `confocal_res_m`        — Confocal diffraction-limited FWHM \[m\]
    /// * `saturation_intensity`  — Fluorophore saturation intensity \[W/m²\]
    pub fn effective_resolution(&self, confocal_res_m: f64, saturation_intensity: f64) -> f64 {
        let i_peak = self.peak_intensity_w_m2();
        let zeta = i_peak / saturation_intensity.max(f64::EPSILON);
        confocal_res_m / (1.0 + zeta).sqrt()
    }

    /// Saturation factor ζ = I_peak / I_sat (dimensionless).
    ///
    /// Resolution improvement scales as 1/√(1+ζ). Typically ζ ~ 10–100 in
    /// practical STED instruments.
    pub fn saturation_factor(&self, saturation_intensity: f64) -> f64 {
        self.peak_intensity_w_m2() / saturation_intensity.max(f64::EPSILON)
    }

    /// Effective PSF FWHM taking both the depletion profile and fluorophore
    /// photophysics into account \[m\].
    ///
    /// `Δr_eff = Δr_conf / sqrt(1 + I_peak/I_sat(fluorophore))`
    pub fn effective_psf_fwhm(&self, fluorophore: &Fluorophore, confocal_fwhm: f64) -> f64 {
        let i_sat = fluorophore.saturation_intensity(self.wavelength_m);
        self.effective_resolution(confocal_fwhm, i_sat)
    }
}

/// Fluorophore photophysical parameters relevant to STED.
///
/// The key parameter controlling STED efficiency is the saturation intensity
/// I_sat = hν / (σ_sted · τ_fl), at which the depletion probability is 1/e.
#[derive(Debug, Clone)]
pub struct Fluorophore {
    /// One-photon absorption cross section \[m²\] (e.g., 3×10⁻²⁰ m² for ATTO647N)
    pub absorption_cross_section_m2: f64,
    /// Fluorescence emission wavelength \[m\]
    pub emission_wavelength_m: f64,
    /// Stimulated emission cross section at the STED wavelength \[m²\]
    pub stimulated_emission_cross_section_m2: f64,
    /// Fluorescence lifetime τ_fl \[ns\]
    pub lifetime_ns: f64,
    /// Fluorescence quantum yield (0–1)
    pub quantum_yield: f64,
}

impl Fluorophore {
    /// Construct a fluorophore model.
    pub fn new(
        absorption_cross_section_m2: f64,
        emission_wavelength_m: f64,
        stimulated_emission_cross_section_m2: f64,
        lifetime_ns: f64,
        quantum_yield: f64,
    ) -> Self {
        Self {
            absorption_cross_section_m2,
            emission_wavelength_m,
            stimulated_emission_cross_section_m2,
            lifetime_ns,
            quantum_yield,
        }
    }

    /// Saturation intensity I_sat = hν_STED / (σ_sted · τ_fl) \[W/m²\].
    ///
    /// At I = I_sat the population remaining in S₁ at the end of the STED pulse
    /// is reduced to 1/e of its initial value.
    ///
    /// # Arguments
    /// * `sted_wavelength_m` — Wavelength of the depletion beam \[m\]
    pub fn saturation_intensity(&self, sted_wavelength_m: f64) -> f64 {
        let h_nu = H_PLANCK * C_LIGHT / sted_wavelength_m;
        let tau_s = self.lifetime_ns * 1e-9;
        let sigma_sted = self.stimulated_emission_cross_section_m2.max(f64::EPSILON);
        h_nu / (sigma_sted * tau_s)
    }

    /// STED depletion efficiency η \[0–1\].
    ///
    /// `η = exp(−σ_sted · I_sted · τ_fl / hν_sted)`
    ///
    /// The residual fluorescence fraction is `1 − η`.
    ///
    /// # Arguments
    /// * `sted_intensity` — Local STED irradiance \[W/m²\]
    pub fn sted_depletion(&self, sted_intensity: f64) -> f64 {
        // Photon flux Φ = I / (hν) — use emission wavelength as a proxy for STED
        // In practice one would pass sted_wavelength separately; here we use the
        // stored emission wavelength as an approximation (STED ≈ red-shifted emission).
        let h_nu = H_PLANCK * C_LIGHT / self.emission_wavelength_m;
        let tau_s = self.lifetime_ns * 1e-9;
        let exponent = -self.stimulated_emission_cross_section_m2 * sted_intensity * tau_s / h_nu;
        exponent.exp()
    }

    /// Effective brightness per molecule: B = Q · σ_abs · I_exc · (1 − depletion) \[photons/s\].
    ///
    /// # Arguments
    /// * `excitation_intensity` — Excitation irradiance \[W/m²\]
    /// * `sted_intensity`       — Local STED irradiance \[W/m²\]
    pub fn effective_brightness(&self, excitation_intensity: f64, sted_intensity: f64) -> f64 {
        let h_nu_exc = H_PLANCK * C_LIGHT / self.emission_wavelength_m; // approx
        let rate_abs = self.absorption_cross_section_m2 * excitation_intensity / h_nu_exc;
        let depletion_factor = 1.0 - self.sted_depletion(sted_intensity);
        // Residual emission ∝ 1 − η
        self.quantum_yield * rate_abs * (1.0 - depletion_factor)
    }
}

/// 3D STED configuration with separate XY (lateral donut) and Z (axial donut) beams.
///
/// The XY donut provides lateral super-resolution via helical phase.
/// The Z donut ("bottle beam") provides axial super-resolution via a
/// top-hat axial phase (0–π shift).
#[derive(Debug, Clone)]
pub struct Sted3d {
    /// Lateral-resolution STED beam (helical phase, donut)
    pub xy_sted: StedBeam,
    /// Axial-resolution STED beam (π-step phase, bottle beam)
    pub z_sted: StedBeam,
}

impl Sted3d {
    /// Construct a 3D STED system.
    pub fn new(xy_sted: StedBeam, z_sted: StedBeam) -> Self {
        Self { xy_sted, z_sted }
    }

    /// Effective 3D STED resolution (Δx, Δy, Δz) \[m\].
    ///
    /// Lateral: Δr_eff = Δr_conf / sqrt(1 + I_xy/I_sat)
    /// Axial:   Δz_eff = Δz_conf / sqrt(1 + I_z/I_sat)
    ///
    /// Confocal references:
    /// - Δr_conf = 0.51 λ_exc / NA  (λ_exc ≈ 650 nm for standard STED)
    /// - Δz_conf = 1.77 λ_exc n / NA²
    pub fn effective_3d_resolution(&self, fluorophore: &Fluorophore) -> (f64, f64, f64) {
        // Use STED wavelength as a proxy for the excitation wavelength:
        // excitation ≈ 0.85 × depletion wavelength (rough guide)
        let lambda_exc = self.xy_sted.wavelength_m * 0.85;
        let na = self.xy_sted.na;
        let n = self.xy_sted.n_medium;

        let dr_conf = 0.51 * lambda_exc / na;
        let dz_conf = 1.77 * lambda_exc * n / (na * na);

        let i_sat_xy = fluorophore.saturation_intensity(self.xy_sted.wavelength_m);
        let i_sat_z = fluorophore.saturation_intensity(self.z_sted.wavelength_m);

        let dr_eff = self.xy_sted.effective_resolution(dr_conf, i_sat_xy);
        let dz_eff = self.z_sted.effective_resolution(dz_conf, i_sat_z);

        (dr_eff, dr_eff, dz_eff)
    }

    /// Required STED beam power to achieve a target lateral resolution \[W\].
    ///
    /// From the resolution formula:
    /// `I_STED = I_sat · ((Δr_conf/Δr_target)² − 1)`
    ///
    /// Power is then back-calculated from I_peak.
    ///
    /// # Arguments
    /// * `target_resolution_m` — Desired effective FWHM \[m\]
    /// * `fluorophore`         — Fluorophore being imaged
    pub fn required_sted_power(&self, target_resolution_m: f64, fluorophore: &Fluorophore) -> f64 {
        let lambda_exc = self.xy_sted.wavelength_m * 0.85;
        let dr_conf = 0.51 * lambda_exc / self.xy_sted.na;
        let i_sat = fluorophore.saturation_intensity(self.xy_sted.wavelength_m);

        let ratio_sq = (dr_conf / target_resolution_m.max(f64::EPSILON)).powi(2);
        let i_required = i_sat * (ratio_sq - 1.0).max(0.0);

        // Back-calculate power: I_peak ≈ 2P/(π w₀²) × 0.54
        let w0 = self.xy_sted.beam_waist_m();
        let area = PI * w0 * w0;
        i_required * area / (2.0 * 0.54)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atto647n() -> Fluorophore {
        // ATTO647N: σ_abs ≈ 3.4×10⁻²⁰ m², σ_sted ≈ 1×10⁻²⁰ m², τ_fl = 3.5 ns, QY = 0.65
        Fluorophore::new(3.4e-20, 660e-9, 1.0e-20, 3.5, 0.65)
    }

    fn standard_sted() -> StedBeam {
        // 775 nm CW STED beam, NA 1.4 oil, 200 mW
        StedBeam::new(0.200, 775e-9, 1.4, 1.515)
    }

    #[test]
    fn test_saturation_intensity_physical_range() {
        let fl = atto647n();
        let i_sat = fl.saturation_intensity(775e-9);
        // Expected: hν / (σ_sted × τ_fl) ≈ 2.56×10⁻¹⁹ / (1e-20 × 3.5e-9) ≈ 7.3 MW/m²
        assert!(
            i_sat > 1e5 && i_sat < 1e11,
            "I_sat = {} W/m² outside physical range",
            i_sat
        );
    }

    #[test]
    fn test_sted_resolution_improves_with_power() {
        let sted_low = StedBeam::new(0.010, 775e-9, 1.4, 1.515);
        let sted_high = StedBeam::new(0.500, 775e-9, 1.4, 1.515);
        let fl = atto647n();
        let confocal_fwhm = 200e-9;
        let res_low = sted_low.effective_psf_fwhm(&fl, confocal_fwhm);
        let res_high = sted_high.effective_psf_fwhm(&fl, confocal_fwhm);
        assert!(
            res_high < res_low,
            "Higher STED power should give finer resolution"
        );
        assert!(
            res_low <= confocal_fwhm,
            "STED resolution must be ≤ confocal"
        );
    }

    #[test]
    fn test_donut_zero_on_axis() {
        let beam = standard_sted();
        let i_axis = beam.donut_intensity(0.0, 0.0);
        assert!(i_axis.abs() < 1e-30, "Donut must have zero on-axis (r=0)");
    }

    #[test]
    fn test_donut_intensity_positive_off_axis() {
        let beam = standard_sted();
        let w0 = beam.beam_waist_m();
        let i_ring = beam.donut_intensity(w0 * 0.5, 0.0);
        assert!(i_ring > 0.0, "Donut intensity should be positive off-axis");
    }

    #[test]
    fn test_depletion_increases_with_intensity() {
        let fl = atto647n();
        let eta_low = fl.sted_depletion(1e9);
        let eta_high = fl.sted_depletion(1e11);
        // Higher intensity → larger depletion → smaller η (residual fraction)
        assert!(
            eta_high < eta_low,
            "Depletion should increase with STED intensity"
        );
    }

    #[test]
    fn test_3d_sted_axial_finer_than_confocal() {
        let xy = StedBeam::new(0.100, 775e-9, 1.4, 1.515);
        let z = StedBeam::new(0.150, 775e-9, 1.4, 1.515);
        let scope = Sted3d::new(xy, z);
        let fl = atto647n();
        let (dx, _dy, dz) = scope.effective_3d_resolution(&fl);
        let lambda_exc = 775e-9 * 0.85;
        let dz_conf = 1.77 * lambda_exc * 1.515 / (1.4_f64.powi(2));
        let dr_conf = 0.51 * lambda_exc / 1.4;
        assert!(dx < dr_conf, "STED Δx should be < confocal");
        assert!(dz < dz_conf, "STED Δz should be < confocal");
    }

    #[test]
    fn test_required_power_monotone_with_resolution() {
        let xy = StedBeam::new(0.1, 775e-9, 1.4, 1.515);
        let z = StedBeam::new(0.1, 775e-9, 1.4, 1.515);
        let scope = Sted3d::new(xy, z);
        let fl = atto647n();
        let p_50nm = scope.required_sted_power(50e-9, &fl);
        let p_30nm = scope.required_sted_power(30e-9, &fl);
        assert!(p_30nm > p_50nm, "Finer resolution requires more STED power");
    }

    #[test]
    fn test_saturation_factor_dimensionless() {
        let beam = standard_sted();
        let fl = atto647n();
        let zeta = beam.saturation_factor(fl.saturation_intensity(beam.wavelength_m));
        assert!(
            zeta.is_finite() && zeta >= 0.0,
            "ζ must be finite and non-negative"
        );
    }
}
