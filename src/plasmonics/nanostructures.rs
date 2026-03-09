//! Plasmonic nanostructures: LSPR, Mie theory, nanorods, gaps, antennas
//!
//! Implements:
//! - PlasmonicNanoparticle: spherical particle LSPR via quasi-static Mie theory
//! - PlasmonicNanorod:      anisotropic prolate-spheroid resonances
//! - PlasmonicGap:          coupled-particle gap enhancement
//! - DipoleAntenna:         gap-fed plasmonic dipole antenna

use num_complex::Complex64;
use std::f64::consts::PI;

use crate::plasmonics::spp::DrudeMetal;

// Speed of light (m/s) — local constant to keep this module self-contained
const C0: f64 = 2.997_924_58e8;

// ──────────────────────────────────────────────────────────────────────────────
// PlasmonicNanoparticle
// ──────────────────────────────────────────────────────────────────────────────

/// Spherical metallic nanoparticle described by Mie theory in the
/// quasi-static (dipole) limit.
///
/// Valid when the particle radius r ≪ λ (typically r < 30 nm for visible light).
/// The key quantity is the Clausius-Mossotti polarizability:
///
///   α(ω) = 4π r³ · (ε_m − ε_d) / (ε_m + 2 ε_d)
///
/// The Fröhlich condition Re(ε_m) = −2 ε_d defines the LSPR.
#[derive(Debug, Clone)]
pub struct PlasmonicNanoparticle {
    /// Particle radius in nm
    pub radius_nm: f64,
    pub metal: DrudeMetal,
    /// Real permittivity of the surrounding medium (ε_d > 0)
    pub eps_medium: f64,
}

impl PlasmonicNanoparticle {
    pub fn new(radius_nm: f64, metal: DrudeMetal, eps_medium: f64) -> Self {
        Self {
            radius_nm,
            metal,
            eps_medium,
        }
    }

    /// Complex Clausius-Mossotti polarizability in m³.
    ///
    /// α(ω) = 4π r³ · (ε_m − ε_d) / (ε_m + 2 ε_d)
    pub fn polarizability(&self, omega: f64) -> Complex64 {
        let r_m = self.radius_nm * 1.0e-9;
        let eps_m = self.metal.permittivity(omega);
        let eps_d = Complex64::new(self.eps_medium, 0.0);
        let vol = 4.0 * PI * r_m * r_m * r_m;
        vol * (eps_m - eps_d) / (eps_m + 2.0 * eps_d)
    }

    /// LSPR angular frequency (rad/s) found by solving Re(ε_m(ω)) = −2 ε_d.
    ///
    /// For the Drude model this gives:
    ///
    ///   ω_lspr² = ωp² / (ε_∞ + 2 ε_d) − γ²
    fn lspr_omega(&self) -> f64 {
        let eps_inf = self.metal.eps_inf;
        let wp = self.metal.omega_p;
        let gamma = self.metal.gamma;
        let eps_d = self.eps_medium;
        let arg = wp * wp / (eps_inf + 2.0 * eps_d) - gamma * gamma;
        if arg > 0.0 {
            arg.sqrt()
        } else {
            wp / (eps_inf + 2.0 * eps_d).sqrt()
        }
    }

    /// LSPR peak wavelength in nm.
    pub fn lspr_wavelength_nm(&self) -> f64 {
        let omega_lspr = self.lspr_omega();
        2.0 * PI * C0 / omega_lspr * 1.0e9
    }

    /// LSPR quality factor: Q ≈ ω_sp / γ (Drude limit).
    pub fn lspr_quality_factor(&self) -> f64 {
        self.lspr_omega() / self.metal.gamma
    }

    /// Extinction cross-section in nm² (quasi-static limit).
    ///
    /// σ_ext = k · Im(α) where k = ω√ε_d / c
    pub fn extinction_cross_section_nm2(&self, omega: f64) -> f64 {
        let k = omega * self.eps_medium.sqrt() / C0; // wavenumber in medium
        let alpha = self.polarizability(omega);
        let sigma_m2 = k * alpha.im;
        sigma_m2 * 1.0e18 // m² → nm²
    }

    /// Scattering cross-section in nm² (dipole radiation):
    ///
    /// σ_sca = k⁴ |α|² / (6π)
    pub fn scattering_cross_section_nm2(&self, omega: f64) -> f64 {
        let k = omega * self.eps_medium.sqrt() / C0;
        let alpha = self.polarizability(omega);
        let k4 = k * k * k * k;
        let alpha2 = alpha.norm_sqr();
        let sigma_m2 = k4 * alpha2 / (6.0 * PI);
        sigma_m2 * 1.0e18
    }

    /// Absorption cross-section in nm²: σ_abs = σ_ext − σ_sca.
    pub fn absorption_cross_section_nm2(&self, omega: f64) -> f64 {
        let ext = self.extinction_cross_section_nm2(omega);
        let sca = self.scattering_cross_section_nm2(omega);
        (ext - sca).max(0.0)
    }

    /// Near-field enhancement |E/E₀|² at the particle surface.
    ///
    /// In the quasi-static dipole approximation the tangential field at the
    /// equator of the sphere gives:
    ///
    ///   |E/E₀|² = |(ε_m − ε_d) / (ε_m + 2 ε_d)|²  + 1  (rough estimate)
    ///
    /// Here we use the standard approximation:
    ///
    ///   |f_LF|² = |(ε_m − ε_d) / (ε_m + 2 ε_d)|²
    pub fn near_field_enhancement(&self, omega: f64) -> f64 {
        let eps_m = self.metal.permittivity(omega);
        let eps_d = Complex64::new(self.eps_medium, 0.0);
        let f = (eps_m - eps_d) / (eps_m + 2.0 * eps_d);
        f.norm_sqr()
    }

    /// SERS enhancement factor: EF = |E/E₀|⁴ ≈ (near-field enhancement)².
    pub fn sers_enhancement(&self, omega: f64) -> f64 {
        let fe2 = self.near_field_enhancement(omega);
        fe2 * fe2
    }

    /// Sensitivity of the LSPR peak wavelength to the medium refractive index
    /// in nm/RIU (refractive index unit).
    ///
    /// Estimated via finite difference: Δλ_lspr / Δn_d for Δn_d = 0.01.
    pub fn sensitivity_nm_per_riu(&self) -> f64 {
        let dn = 0.01_f64;
        let n_d = self.eps_medium.sqrt();
        let eps_d2 = (n_d + dn) * (n_d + dn);

        let np2 = PlasmonicNanoparticle::new(self.radius_nm, self.metal.clone(), eps_d2);
        let lam1 = self.lspr_wavelength_nm();
        let lam2 = np2.lspr_wavelength_nm();
        (lam2 - lam1) / dn
    }

    /// Figure of merit: FOM = sensitivity \[nm/RIU\] / FWHM \[nm\].
    ///
    /// FWHM is approximated from the Lorentzian linewidth at the LSPR:
    ///   FWHM ≈ 2π c γ / ω_lspr²  (in nm)
    pub fn figure_of_merit_per_riu(&self) -> f64 {
        let sensitivity = self.sensitivity_nm_per_riu();
        let omega_lspr = self.lspr_omega();
        // Lorentzian FWHM in frequency: Δω ≈ γ  (Drude)
        // Convert to wavelength: Δλ ≈ (λ²/c) * Δω/(2π) = (2πc/ω²) * γ
        let gamma = self.metal.gamma;
        let fwhm_nm = 2.0 * PI * C0 * gamma / (omega_lspr * omega_lspr) * 1.0e9;
        if fwhm_nm < f64::EPSILON {
            return 0.0;
        }
        sensitivity / fwhm_nm
    }

    /// Extinction spectrum over a wavelength range \[λ_min, λ_max\] in nm.
    /// Returns Vec<(wavelength_nm, extinction_cross_section_nm²)>.
    pub fn extinction_spectrum(
        &self,
        lambda_min_nm: f64,
        lambda_max_nm: f64,
        n_pts: usize,
    ) -> Vec<(f64, f64)> {
        (0..n_pts)
            .map(|i| {
                let lam_nm = lambda_min_nm
                    + (lambda_max_nm - lambda_min_nm) * i as f64 / (n_pts - 1).max(1) as f64;
                let omega = 2.0 * PI * C0 / (lam_nm * 1.0e-9);
                let sigma = self.extinction_cross_section_nm2(omega);
                (lam_nm, sigma)
            })
            .collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// PlasmonicNanorod
// ──────────────────────────────────────────────────────────────────────────────

/// Metallic nanorod modelled as a prolate spheroid.
///
/// Semi-axes: b (short, x/y) and a (long, z).  The longitudinal LSPR
/// (along z) is red-shifted relative to the spherical case.
///
/// Depolarization factor for a prolate spheroid (a > b):
///
///   L_z = (1 − e²)/e² · \[−1 + (1/(2e)) · ln((1+e)/(1−e))\]
///
/// where e² = 1 − (b/a)².
#[derive(Debug, Clone)]
pub struct PlasmonicNanorod {
    /// Short semi-axis (radius) in nm
    pub short_axis_nm: f64,
    /// Long semi-axis (half-length) in nm
    pub long_axis_nm: f64,
    pub metal: DrudeMetal,
    pub eps_medium: f64,
}

impl PlasmonicNanorod {
    pub fn new(short_nm: f64, long_nm: f64, metal: DrudeMetal, eps_medium: f64) -> Self {
        Self {
            short_axis_nm: short_nm,
            long_axis_nm: long_nm,
            metal,
            eps_medium,
        }
    }

    /// Aspect ratio AR = long_axis / short_axis.
    pub fn aspect_ratio(&self) -> f64 {
        self.long_axis_nm / self.short_axis_nm
    }

    /// Depolarization factor along the long (z) axis of a prolate spheroid.
    pub fn depolarization_factor_long(&self) -> f64 {
        let ar = self.aspect_ratio();
        if (ar - 1.0).abs() < 1.0e-6 {
            return 1.0 / 3.0; // sphere limit
        }
        let e2 = 1.0 - 1.0 / (ar * ar);
        let e = e2.sqrt();
        let lz = (1.0 - e2) / (e2) * (-1.0 + (1.0 / (2.0 * e)) * ((1.0 + e) / (1.0 - e)).ln());
        lz.abs()
    }

    /// Depolarization factor along the transverse (x or y) axis.
    /// Constraint: L_x + L_y + L_z = 1, and by symmetry L_x = L_y.
    pub fn depolarization_factor_trans(&self) -> f64 {
        (1.0 - self.depolarization_factor_long()) / 2.0
    }

    /// LSPR angular frequency for an axis with depolarization factor L.
    ///
    /// From the quasi-static condition:
    ///   Re(ε_m) = −ε_d · (1 − L) / L
    ///
    /// Solved analytically for the Drude model.
    fn lspr_omega_for_depol(&self, dep_l: f64) -> f64 {
        // Re(ε_m(ω)) = ε_∞ − ωp²/(ω² + γ²) ≈ ε_∞ − ωp²/ω²  (low loss)
        // Setting equal to target:
        //   ε_∞ − ωp²/(ω² + γ²) = −ε_d(1−L)/L
        //   ωp²/(ω² + γ²) = ε_∞ + ε_d(1−L)/L
        let eps_inf = self.metal.eps_inf;
        let wp = self.metal.omega_p;
        let gamma = self.metal.gamma;
        let eps_d = self.eps_medium;
        let target = eps_inf + eps_d * (1.0 - dep_l) / dep_l;
        if target <= 0.0 {
            return wp / eps_inf.sqrt();
        }
        let omega2 = wp * wp / target - gamma * gamma;
        if omega2 > 0.0 {
            omega2.sqrt()
        } else {
            wp / target.sqrt()
        }
    }

    /// Longitudinal LSPR wavelength in nm (long axis).
    pub fn longitudinal_lspr_nm(&self) -> f64 {
        let omega = self.lspr_omega_for_depol(self.depolarization_factor_long());
        2.0 * PI * C0 / omega * 1.0e9
    }

    /// Transverse LSPR wavelength in nm (short axis, ≈ spherical resonance).
    pub fn transverse_lspr_nm(&self) -> f64 {
        let omega = self.lspr_omega_for_depol(self.depolarization_factor_trans());
        2.0 * PI * C0 / omega * 1.0e9
    }

    /// Longitudinal complex polarizability (Clausius-Mossotti for a spheroid).
    ///
    /// α_z = V ε_d (ε_m − ε_d) / (ε_d + L_z (ε_m − ε_d))
    pub fn polarizability_longitudinal(&self, omega: f64) -> Complex64 {
        let a_m = self.long_axis_nm * 1.0e-9;
        let b_m = self.short_axis_nm * 1.0e-9;
        let vol = 4.0 / 3.0 * PI * a_m * b_m * b_m;
        let eps_m = self.metal.permittivity(omega);
        let eps_d = Complex64::new(self.eps_medium, 0.0);
        let l = Complex64::new(self.depolarization_factor_long(), 0.0);
        let numerator = eps_d * (eps_m - eps_d);
        let denominator = eps_d + l * (eps_m - eps_d);
        vol * numerator / denominator
    }

    /// Transverse complex polarizability.
    pub fn polarizability_transverse(&self, omega: f64) -> Complex64 {
        let a_m = self.long_axis_nm * 1.0e-9;
        let b_m = self.short_axis_nm * 1.0e-9;
        let vol = 4.0 / 3.0 * PI * a_m * b_m * b_m;
        let eps_m = self.metal.permittivity(omega);
        let eps_d = Complex64::new(self.eps_medium, 0.0);
        let l = Complex64::new(self.depolarization_factor_trans(), 0.0);
        let numerator = eps_d * (eps_m - eps_d);
        let denominator = eps_d + l * (eps_m - eps_d);
        vol * numerator / denominator
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// PlasmonicGap
// ──────────────────────────────────────────────────────────────────────────────

/// Two coupled spherical nanoparticles with a nanogap.
///
/// The gap field enhancement is estimated using the coupled-dipole model
/// in the quasi-static limit.
#[derive(Debug, Clone)]
pub struct PlasmonicGap {
    pub particle1: PlasmonicNanoparticle,
    pub particle2: PlasmonicNanoparticle,
    /// Edge-to-edge gap distance in nm
    pub gap_nm: f64,
}

impl PlasmonicGap {
    pub fn new(p1: PlasmonicNanoparticle, p2: PlasmonicNanoparticle, gap_nm: f64) -> Self {
        Self {
            particle1: p1,
            particle2: p2,
            gap_nm,
        }
    }

    /// Approximate gap electric-field enhancement |E_gap / E₀|².
    ///
    /// Uses the lightning-rod / coupled dipole scaling:
    ///
    ///   EF_gap ≈ EF_single × (r_eff / gap)^n
    ///
    /// where n ≈ 1 (empirical from coupled-dipole theory) and r_eff is the
    /// effective radius of the smaller particle.
    pub fn gap_enhancement(&self, omega: f64) -> f64 {
        let ef1 = self.particle1.near_field_enhancement(omega);
        let ef2 = self.particle2.near_field_enhancement(omega);
        // Geometric mean of individual near-field enhancements as baseline
        let ef_single = (ef1 * ef2).sqrt();
        let r_eff = self.particle1.radius_nm.min(self.particle2.radius_nm);
        // Coupling factor: (r / gap)^1  — gap enhancement is multiplicative
        let coupling = (r_eff / self.gap_nm).max(1.0);
        ef_single * coupling
    }

    /// Coupling-induced red-shift of the LSPR in nm.
    ///
    /// Classical coupling model: Δλ ≈ λ_sp · A · exp(−g / (κ · r))
    /// with A ≈ 0.18, κ ≈ 0.23 (empirical for Au spheres).
    pub fn lspr_redshift_nm(&self) -> f64 {
        let lam_sp = self.particle1.lspr_wavelength_nm();
        let r = self.particle1.radius_nm;
        let g = self.gap_nm;
        let a = 0.18_f64;
        let kappa = 0.23_f64;
        lam_sp * a * (-g / (kappa * r)).exp()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DipoleAntenna
// ──────────────────────────────────────────────────────────────────────────────

/// Gap-fed plasmonic dipole antenna.
///
/// Models a symmetric pair of metal arms with a nanogap at the feed point.
/// The resonance condition is approximately half-wavelength with an effective
/// wavelength scaling λ_eff = n₁ + n₂ · (λ/λ_p).
#[derive(Debug, Clone)]
pub struct DipoleAntenna {
    /// Single arm length in nm
    pub arm_length_nm: f64,
    /// Arm width in nm
    pub arm_width_nm: f64,
    /// Feed gap in nm
    pub gap_nm: f64,
    pub metal: DrudeMetal,
    /// Permittivity of the substrate
    pub eps_substrate: f64,
}

impl DipoleAntenna {
    pub fn new(
        length_nm: f64,
        width_nm: f64,
        gap_nm: f64,
        metal: DrudeMetal,
        eps_sub: f64,
    ) -> Self {
        Self {
            arm_length_nm: length_nm,
            arm_width_nm: width_nm,
            gap_nm,
            metal,
            eps_substrate: eps_sub,
        }
    }

    /// Resonance wavelength in nm using the effective wavelength model:
    ///
    ///   λ_res ≈ 2 · n_eff · arm_length
    ///
    /// where n_eff is the effective index from the real part of the metal
    /// refractive index blended with the substrate (average dielectric).
    pub fn resonance_wavelength_nm(&self) -> f64 {
        // Effective medium: average of substrate and air above
        let eps_eff = (self.eps_substrate + 1.0) / 2.0;
        let n_eff = eps_eff.sqrt();
        // Full antenna length = 2 * arm_length (plus gap, which is small)
        let total_length_nm = 2.0 * self.arm_length_nm + self.gap_nm;
        // Half-wave resonance: λ_res = 2 n_eff L
        2.0 * n_eff * total_length_nm
    }

    /// Radiation efficiency η = P_rad / P_total.
    ///
    /// For a plasmonic antenna η < 1 due to ohmic losses.  Estimated via the
    /// ratio of radiation resistance to total resistance (radiation + loss).
    ///
    /// Approximate formula:
    ///   η ≈ 1 / (1 + P_ohm / P_rad) ≈ 1 / (1 + (L_sp / l_antenna))
    ///
    /// where L_sp is the SPP propagation length on the metal surface.
    pub fn radiation_efficiency(&self) -> f64 {
        let lambda_res_m = self.resonance_wavelength_nm() * 1.0e-9;
        let omega = 2.0 * PI * C0 / lambda_res_m;
        // SPP propagation length on the metal (air cladding)
        let spp = crate::plasmonics::spp::SurfacePlasmonPolariton::new(self.metal.clone(), 1.0);
        let l_sp_nm = spp.propagation_length_um(omega) * 1.0e3; // µm → nm
        let l_arm = self.arm_length_nm;
        // efficiency ≈ 1/(1 + l_arm/L_sp)
        if l_sp_nm < f64::EPSILON {
            return 0.0;
        }
        1.0 / (1.0 + l_arm / l_sp_nm)
    }

    /// Directivity D of a short dipole antenna.  For a half-wave dipole D ≈ 1.64.
    /// For a subwavelength antenna D ≈ 1.5.
    pub fn directivity(&self) -> f64 {
        1.5_f64
    }

    /// Gap electric-field enhancement |E_gap / E₀|² at the feed point.
    ///
    /// Scales roughly as (L/gap)^α with α ≈ 1 for a plasmonic antenna, modulated
    /// by the LSPR polarizability.
    pub fn gap_enhancement(&self, omega: f64) -> f64 {
        let eps_m = self.metal.permittivity(omega);
        let eps_d = Complex64::new(1.0, 0.0); // air in gap
        let f_lf = (eps_m - eps_d) / (eps_m + 2.0 * eps_d);
        let local_fe = f_lf.norm_sqr();
        // Geometric focusing: (arm_length / gap)
        let geometric = self.arm_length_nm / self.gap_nm.max(0.1);
        local_fe * geometric
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn omega_from_nm(lam_nm: f64) -> f64 {
        2.0 * PI * C0 / (lam_nm * 1.0e-9)
    }

    // ── PlasmonicNanoparticle ───────────────────────────────────────────────

    #[test]
    fn test_lspr_wavelength_gold_in_water() {
        // 20 nm Au sphere in water (n=1.33 → ε_d ≈ 1.77)
        let gold = DrudeMetal::gold();
        let np = PlasmonicNanoparticle::new(10.0, gold, 1.77);
        let lam = np.lspr_wavelength_nm();
        // Expected ~510–560 nm for gold in water (Drude model approx)
        assert!(
            lam > 400.0 && lam < 700.0,
            "Gold LSPR in water should be ~500-600 nm; got {lam:.1} nm"
        );
    }

    #[test]
    fn test_extinction_cross_section_at_resonance_peaks() {
        let gold = DrudeMetal::gold();
        let np = PlasmonicNanoparticle::new(10.0, gold, 1.77);
        let spectrum = np.extinction_spectrum(400.0, 800.0, 200);

        // Find peak
        let (lam_peak, sigma_peak) =
            spectrum
                .iter()
                .copied()
                .fold(
                    (0.0_f64, 0.0_f64),
                    |(bl, bs), (l, s)| {
                        if s > bs {
                            (l, s)
                        } else {
                            (bl, bs)
                        }
                    },
                );
        // Peak should exist and be positive
        assert!(sigma_peak > 0.0, "Extinction peak must be positive");
        // Peak should be in visible range
        assert!(
            lam_peak > 400.0 && lam_peak < 800.0,
            "Peak must be in visible range; got {lam_peak:.1} nm"
        );
    }

    #[test]
    fn test_sers_enhancement_gt_scattering() {
        // SERS EF = |E/E0|^4  should exceed the scattering CS normalized cross section
        let gold = DrudeMetal::gold();
        let np = PlasmonicNanoparticle::new(10.0, gold, 1.77);
        let lspr_nm = np.lspr_wavelength_nm();
        let omega = omega_from_nm(lspr_nm);
        let fe2 = np.near_field_enhancement(omega);
        let sers = np.sers_enhancement(omega);
        // SERS EF = fe2^2 must be >= fe2 (since fe2 >= 1 at resonance)
        assert!(
            sers >= fe2,
            "SERS EF ({sers:.2}) should be >= near-field enhancement squared ({fe2:.2})"
        );
    }

    #[test]
    fn test_near_field_enhancement_at_resonance() {
        let gold = DrudeMetal::gold();
        let np = PlasmonicNanoparticle::new(10.0, gold, 1.77);
        let lspr_nm = np.lspr_wavelength_nm();
        let omega = omega_from_nm(lspr_nm);
        let fe2 = np.near_field_enhancement(omega);
        assert!(
            fe2 > 1.0,
            "Near-field enhancement at LSPR should exceed 1; got {fe2:.3}"
        );
    }

    #[test]
    fn test_sensitivity_nm_per_riu_positive() {
        let gold = DrudeMetal::gold();
        let np = PlasmonicNanoparticle::new(10.0, gold, 1.77);
        let s = np.sensitivity_nm_per_riu();
        assert!(
            s > 0.0,
            "LSPR sensitivity must be positive (red-shift with higher n); got {s}"
        );
    }

    // ── PlasmonicNanorod ────────────────────────────────────────────────────

    #[test]
    fn test_nanorod_aspect_ratio() {
        let gold = DrudeMetal::gold();
        let rod = PlasmonicNanorod::new(10.0, 40.0, gold, 1.77);
        let ar = rod.aspect_ratio();
        let expected = 40.0 / 10.0;
        assert!(
            (ar - expected).abs() < 1.0e-10,
            "Aspect ratio mismatch: got {ar}, expected {expected}"
        );
    }

    #[test]
    fn test_nanorod_longitudinal_redshifted() {
        let gold = DrudeMetal::gold();
        let rod = PlasmonicNanorod::new(10.0, 40.0, gold, 1.77);
        let lam_long = rod.longitudinal_lspr_nm();
        let lam_trans = rod.transverse_lspr_nm();
        assert!(
            lam_long > lam_trans,
            "Longitudinal LSPR ({lam_long:.1} nm) must be red-shifted vs transverse ({lam_trans:.1} nm)"
        );
    }

    // ── PlasmonicGap ────────────────────────────────────────────────────────

    #[test]
    fn test_gap_enhancement_larger_than_single() {
        let gold1 = DrudeMetal::gold();
        let gold2 = DrudeMetal::gold();
        let np1 = PlasmonicNanoparticle::new(10.0, gold1.clone(), 1.0);
        let np2 = PlasmonicNanoparticle::new(10.0, gold2.clone(), 1.0);
        let gap = PlasmonicGap::new(np1.clone(), np2.clone(), 2.0);
        let omega = omega_from_nm(np1.lspr_wavelength_nm());
        let fe_single = np1.near_field_enhancement(omega);
        let fe_gap = gap.gap_enhancement(omega);
        assert!(
            fe_gap >= fe_single,
            "Gap enhancement ({fe_gap:.2}) should exceed single-particle enhancement ({fe_single:.2})"
        );
    }
}
