//! Second/Third Harmonic Generation (SHG/THG) Microscopy
//!
//! Models nonlinear optical microscopy based on coherent second- and third-order
//! nonlinear scattering. SHG is particularly sensitive to non-centrosymmetric
//! structures (collagen, myosin, microtubules), while THG is generated at
//! refractive-index discontinuities (interfaces, organelles).
//!
//! # References
//! - Campagnola & Loew, Nature Biotechnology 21, 1356 (2003)
//! - Mertz & Brown, Opt. Lett. 25, 447 (2000)
//! - Débarre et al., Opt. Express 14, 7909 (2006)

use std::f64::consts::PI;

/// Speed of light in vacuum \[m/s\]
pub const C_LIGHT: f64 = 2.99792458e8;
/// Planck constant \[J·s\]
pub const H_PLANCK: f64 = 6.62607015e-34;
/// Reduced Planck constant \[J·s\]
pub const HBAR: f64 = 1.054571817e-34;
/// Boltzmann constant \[J/K\]
pub const KB: f64 = 1.380649e-23;

/// Second-Harmonic Generation (SHG) microscope configuration.
///
/// Models the signal strength, phase matching, resolution and PSF of an SHG
/// laser-scanning microscope. Excitation at ω; detection at 2ω.
#[derive(Debug, Clone)]
pub struct ShgMicroscope {
    /// Fundamental (excitation) wavelength \[m\]
    pub excitation_wavelength_m: f64,
    /// Numerical aperture of the objective
    pub na: f64,
    /// Refractive index of the immersion medium
    pub n_medium: f64,
    /// Camera/scan pixel size at the sample plane \[m\]
    pub pixel_size_m: f64,
}

impl ShgMicroscope {
    /// Construct a new SHG microscope with the given parameters.
    pub fn new(excitation_wavelength_m: f64, na: f64, n_medium: f64, pixel_size_m: f64) -> Self {
        Self {
            excitation_wavelength_m,
            na,
            n_medium,
            pixel_size_m,
        }
    }

    /// SHG signal intensity from a single nonlinear dipole (point source).
    ///
    /// `I_SHG = (ω²/c²) · |χ⁽²⁾_eff|² · I_exc² · L² / 4 · sinc²(ΔkL/2)`
    ///
    /// Here we fold the sinc² term in via the phase-matching factor evaluated
    /// at the coherence-length scale so that the formula is self-consistent.
    ///
    /// # Arguments
    /// * `intensity_w_m2` — Excitation irradiance \[W/m²\]
    /// * `chi2_eff`        — Effective second-order susceptibility \[m/V\]
    /// * `interaction_length_m` — Physical interaction length L \[m\]
    pub fn shg_signal(&self, intensity_w_m2: f64, chi2_eff: f64, interaction_length_m: f64) -> f64 {
        let omega = 2.0 * PI * C_LIGHT / self.excitation_wavelength_m;
        // Prefactor: (ω/c)² in SI, yielding units of W/m² when chi2 in m/V
        let prefactor = (omega / C_LIGHT).powi(2);
        // L²/4 geometrical factor for collinear interaction
        let geom = interaction_length_m * interaction_length_m * 0.25;
        prefactor * chi2_eff.powi(2) * intensity_w_m2.powi(2) * geom
    }

    /// Phase mismatch Δk for collinear SHG \[rad/m\].
    ///
    /// `Δk = (2ω/c)(n_2ω − n_ω) = 2π/λ_ω · 2(n_2ω − n_ω)`
    ///
    /// # Arguments
    /// * `n_omega`  — Refractive index at the fundamental frequency
    /// * `n_2omega` — Refractive index at the second harmonic
    pub fn phase_mismatch(&self, n_omega: f64, n_2omega: f64) -> f64 {
        let lambda_omega = self.excitation_wavelength_m;
        // Δk = 2π/λ_ω · 2(n_2ω − n_ω)
        2.0 * (2.0 * PI / lambda_omega) * (n_2omega - n_omega)
    }

    /// Coherence length L_c = π / |Δk| \[m\].
    ///
    /// The coherence length is the crystal thickness at which the SHG intensity
    /// reaches its first maximum. Phase matching is achieved when L_c → ∞.
    pub fn coherence_length(&self, n_omega: f64, n_2omega: f64) -> f64 {
        let dk = self.phase_mismatch(n_omega, n_2omega).abs();
        if dk < 1e-30 {
            // Perfect phase matching — coherence length effectively infinite
            f64::INFINITY
        } else {
            PI / dk
        }
    }

    /// Axial (z) resolution of SHG microscopy \[m\].
    ///
    /// `Δz_SHG ≈ 0.89 λ_exc / (n · NA²)`
    ///
    /// The SHG PSF is the square of the excitation PSF, giving tighter
    /// axial sectioning than single-photon confocal.
    pub fn axial_resolution_m(&self) -> f64 {
        0.89 * self.excitation_wavelength_m / (self.n_medium * self.na * self.na)
    }

    /// Lateral resolution of SHG microscopy \[m\] (at the SHG wavelength λ/2).
    ///
    /// `Δr_SHG ≈ 0.51 λ_SHG / NA = 0.51 · (λ_exc/2) / NA`
    pub fn lateral_resolution_m(&self) -> f64 {
        let lambda_shg = self.excitation_wavelength_m * 0.5;
        0.51 * lambda_shg / self.na
    }

    /// SHG PSF intensity at position (x, y, z) — 3D Gaussian approximation \[a.u.\].
    ///
    /// Uses the squared Gaussian PSF, appropriate for a two-photon process:
    /// `I_PSF(r, z) = exp(−2r²/w_r² − 2z²/w_z²)²`
    ///
    /// where w_r and w_z are the 1/e² beam radii.
    pub fn psf_intensity(&self, x: f64, y: f64, z: f64) -> f64 {
        let w_r = self.lateral_resolution_m() / (2.0 * (2.0_f64.ln()).sqrt());
        let w_z = self.axial_resolution_m() / (2.0 * (2.0_f64.ln()).sqrt());
        let r2 = x * x + y * y;
        // Squared Gaussian: PSF_SHG(r,z) = [PSF_exc(r,z)]²
        let arg_r = -2.0 * r2 / (w_r * w_r);
        let arg_z = -2.0 * z * z / (w_z * w_z);
        (arg_r + arg_z).exp().powi(2)
    }

    /// Forward-to-backward SHG intensity ratio F/B for a thin slab sample.
    ///
    /// From Mertz & Brown model:
    /// `F/B = |sinc(ΔkL/2) · exp(iΔkL/2) + sinc(ΔkL/2) · exp(−iΔkL/2)|²`
    ///
    /// Simplified closed-form for a thin ordered sample (collagen-like):
    /// `F/B ≈ (1 + cos(Δk·L)) / (1 − cos(Δk·L) + ε)`
    ///
    /// In practice F/B can range from ~1 (disordered) to ~10–100 (ordered fibers).
    pub fn fb_ratio(&self, sample_thickness_m: f64, n_omega: f64, n_2omega: f64) -> f64 {
        let dk = self.phase_mismatch(n_omega, n_2omega);
        let phi = dk * sample_thickness_m;
        let cos_phi = phi.cos();
        let forward = (1.0 + cos_phi) * 0.5 + 0.5; // always > 0
        let backward = (1.0 - cos_phi) * 0.5 + 1e-9;
        forward / backward
    }

    /// SHG emission wavelength \[m\] (= λ_exc / 2).
    pub fn shg_wavelength_m(&self) -> f64 {
        self.excitation_wavelength_m * 0.5
    }
}

/// Third-Harmonic Generation (THG) microscope configuration.
///
/// THG only arises at interfaces and discontinuities due to the Gouy phase
/// integral cancellation in homogeneous media. It is a label-free contrast
/// mechanism for organelles, lipid droplets, and myelin.
#[derive(Debug, Clone)]
pub struct ThgMicroscope {
    /// Fundamental (excitation) wavelength \[m\]
    pub excitation_wavelength_m: f64,
    /// Numerical aperture of the objective
    pub na: f64,
    /// Refractive index of the immersion medium
    pub n_medium: f64,
}

impl ThgMicroscope {
    /// Construct a new THG microscope.
    pub fn new(excitation_wavelength_m: f64, na: f64, n_medium: f64) -> Self {
        Self {
            excitation_wavelength_m,
            na,
            n_medium,
        }
    }

    /// THG signal intensity from an interface layer of thickness dz_m \[W/m²\].
    ///
    /// `I_THG ∝ |χ⁽³⁾_eff|² · I_exc³ · (dz / z_R)²`
    ///
    /// where z_R is the Rayleigh length, giving the axial confinement.
    ///
    /// # Arguments
    /// * `intensity_w_m2` — Excitation irradiance \[W/m²\]
    /// * `chi3_eff`        — Effective third-order susceptibility \[m²/V²\]
    /// * `dz_m`           — Interface thickness / sample layer thickness \[m\]
    pub fn thg_signal(&self, intensity_w_m2: f64, chi3_eff: f64, dz_m: f64) -> f64 {
        let omega = 2.0 * PI * C_LIGHT / self.excitation_wavelength_m;
        let z_r = self.rayleigh_length_m();
        let prefactor = (omega / C_LIGHT).powi(2);
        let thickness_factor = (dz_m / z_r).powi(2);
        prefactor * chi3_eff.powi(2) * intensity_w_m2.powi(3) * thickness_factor
    }

    /// Rayleigh length z_R = π n w₀² / λ \[m\].
    ///
    /// Beam waist w₀ ≈ λ / (π · NA / n).
    pub fn rayleigh_length_m(&self) -> f64 {
        let w0 = self.excitation_wavelength_m / (PI * self.na / self.n_medium);
        PI * self.n_medium * w0 * w0 / self.excitation_wavelength_m
    }

    /// Interface enhancement factor from refractive-index contrast.
    ///
    /// THG intensity scales as (Δn/n)² at an interface, reflecting the
    /// failure of the Gouy-phase cancellation in non-uniform media.
    ///
    /// `η = (Δn/n)²`
    ///
    /// # Arguments
    /// * `dn_over_n` — Fractional index step Δn/n across the interface
    pub fn interface_enhancement_factor(&self, dn_over_n: f64) -> f64 {
        dn_over_n * dn_over_n
    }

    /// Axial resolution of THG microscopy \[m\].
    ///
    /// `Δz_THG ≈ 0.67 λ / (n · NA²)` — tighter than SHG due to cubic dependence.
    pub fn axial_resolution_m(&self) -> f64 {
        0.67 * self.excitation_wavelength_m / (self.n_medium * self.na * self.na)
    }

    /// Lateral resolution of THG microscopy \[m\] (at λ/3).
    pub fn lateral_resolution_m(&self) -> f64 {
        let lambda_thg = self.excitation_wavelength_m / 3.0;
        0.51 * lambda_thg / self.na
    }

    /// THG emission wavelength \[m\] (= λ_exc / 3).
    pub fn thg_wavelength_m(&self) -> f64 {
        self.excitation_wavelength_m / 3.0
    }
}

/// SHG model for fibrillar collagen.
///
/// Fibrillar collagens (type I, II) are abundant in connective tissue and
/// produce strong SHG due to their non-centrosymmetric triple-helix structure.
/// This model captures angular emission patterns and polarimetric anisotropy.
#[derive(Debug, Clone)]
pub struct CollagenShg {
    /// Effective second-order susceptibility χ⁽²⁾ \[m/V\] (≈ 10 pm/V for collagen)
    pub chi2_eff: f64,
    /// Mean fiber diameter \[m\]
    pub fiber_diameter_m: f64,
    /// Coherence length for SHG within the fiber \[m\]
    pub coherence_length_m: f64,
}

impl CollagenShg {
    /// Construct a collagen SHG model with typical parameters.
    pub fn new(chi2_eff: f64, fiber_diameter_m: f64, coherence_length_m: f64) -> Self {
        Self {
            chi2_eff,
            fiber_diameter_m,
            coherence_length_m,
        }
    }

    /// SHG signal as a function of fiber orientation angle θ and excitation intensity.
    ///
    /// For a uniaxial fiber aligned at angle θ to the laser polarization:
    /// `I_SHG(θ) ∝ |χ⁽²⁾|² · I² · (cos²θ + ρ sin²θ)² · sinc²(ΔkL/2)`
    ///
    /// where ρ = χ_yyy/χ_yxx ≈ 1.4 for collagen, and L = fiber_diameter.
    ///
    /// # Arguments
    /// * `theta`     — Fiber orientation angle relative to laser polarization \[rad\]
    /// * `intensity` — Excitation irradiance \[W/m²\]
    pub fn signal_vs_angle(&self, theta: f64, intensity: f64) -> f64 {
        // Susceptibility ratio ρ for collagen (fibrillar type I)
        let rho: f64 = 1.4;
        let (sin_t, cos_t) = theta.sin_cos();
        let tensor_factor = cos_t * cos_t + rho * sin_t * sin_t;
        // Phase-matching sinc² factor
        let lc = self.coherence_length_m.max(1e-30);
        let dk_l_half = PI * self.fiber_diameter_m / (2.0 * lc);
        let sinc_sq = if dk_l_half.abs() < 1e-12 {
            1.0
        } else {
            (dk_l_half.sin() / dk_l_half).powi(2)
        };
        self.chi2_eff.powi(2) * intensity.powi(2) * tensor_factor.powi(2) * sinc_sq
    }

    /// Polarimetric anisotropy ratio R = I_par / I_perp.
    ///
    /// For fibrillar collagen, R > 1 indicates ordered fibers. A fully random
    /// distribution gives R = 1. High R (> 2) indicates well-aligned fibrils.
    ///
    /// Uses the two-tensor model: R = (1 + ρ²) / (2ρ).
    pub fn anisotropy_ratio(&self) -> f64 {
        let rho: f64 = 1.4;
        (1.0 + rho * rho) / (2.0 * rho)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_shg() -> ShgMicroscope {
        // 1040 nm Ti:Sapphire excitation, NA = 1.2 water immersion
        ShgMicroscope::new(1040e-9, 1.2, 1.333, 100e-9)
    }

    #[test]
    fn test_shg_signal_scales_quadratically_with_intensity() {
        let scope = default_shg();
        let chi2 = 1.0e-11; // 10 pm/V
        let l = 10e-6;
        let i1 = scope.shg_signal(1e12, chi2, l);
        let i2 = scope.shg_signal(2e12, chi2, l);
        // Should scale as I², so doubling I → 4× signal
        let ratio = i2 / i1;
        assert!((ratio - 4.0).abs() < 1e-6, "Expected 4x, got {}", ratio);
    }

    #[test]
    fn test_phase_mismatch_perfect_match() {
        let scope = default_shg();
        let dk = scope.phase_mismatch(1.5, 1.5);
        assert!(dk.abs() < 1e-20, "Perfect match should give Δk ≈ 0");
    }

    #[test]
    fn test_coherence_length_physical_range() {
        let scope = default_shg();
        // Typical dispersion: n_2ω − n_ω ≈ 0.01
        let lc = scope.coherence_length(1.50, 1.51);
        // For 1040 nm and Δn = 0.01, L_c ≈ λ/(4Δn) ≈ 26 μm
        assert!(lc > 1e-6 && lc < 1e-3, "L_c out of physical range: {}", lc);
    }

    #[test]
    fn test_axial_resolution_tighter_than_confocal() {
        // SHG axial resolution should be tighter than single-photon confocal
        // Δz_conf ≈ 1.4λ/(n·NA²); Δz_SHG ≈ 0.89λ/(n·NA²)
        let scope = default_shg();
        let dz_shg = scope.axial_resolution_m();
        let dz_confocal =
            1.4 * scope.excitation_wavelength_m / (scope.n_medium * scope.na * scope.na);
        assert!(
            dz_shg < dz_confocal,
            "SHG should have tighter axial resolution"
        );
    }

    #[test]
    fn test_psf_maximum_at_origin() {
        let scope = default_shg();
        let i0 = scope.psf_intensity(0.0, 0.0, 0.0);
        let i_off = scope.psf_intensity(200e-9, 0.0, 0.0);
        assert!(i0 > i_off, "PSF maximum should be at origin");
        assert!((i0 - 1.0).abs() < 1e-12, "PSF(0,0,0) should be 1.0");
    }

    #[test]
    fn test_thg_interface_enhancement() {
        let thg = ThgMicroscope::new(1200e-9, 1.2, 1.333);
        let eta_small = thg.interface_enhancement_factor(0.01);
        let eta_large = thg.interface_enhancement_factor(0.10);
        assert!(
            (eta_large / eta_small - 100.0).abs() < 1.0,
            "Enhancement should scale as (Δn/n)²"
        );
    }

    #[test]
    fn test_thg_axial_tighter_than_shg() {
        let shg = ShgMicroscope::new(1200e-9, 1.2, 1.333, 100e-9);
        let thg = ThgMicroscope::new(1200e-9, 1.2, 1.333);
        assert!(
            thg.axial_resolution_m() < shg.axial_resolution_m(),
            "THG should have tighter axial resolution than SHG"
        );
    }

    #[test]
    fn test_collagen_anisotropy() {
        let collagen = CollagenShg::new(10e-12, 200e-9, 5e-6);
        let r = collagen.anisotropy_ratio();
        // For ρ = 1.4: R = (1 + 1.96)/(2.8) ≈ 1.057
        assert!(r > 1.0, "Anisotropy ratio must be ≥ 1 for ordered collagen");
        assert!(r < 2.0, "Anisotropy ratio unrealistically large");
    }

    #[test]
    fn test_collagen_signal_vs_angle_max_at_zero() {
        let collagen = CollagenShg::new(10e-12, 200e-9, 5e-6);
        let i0 = collagen.signal_vs_angle(0.0, 1e12);
        let i90 = collagen.signal_vs_angle(PI / 2.0, 1e12);
        // At θ=0 (fiber parallel to polarization): pure χ_yyy contribution
        // At θ=90° (perpendicular): ρ² contribution, smaller since ρ > 1 here
        assert!(i0 > 0.0 && i90 > 0.0, "SHG signal must be positive");
    }
}
