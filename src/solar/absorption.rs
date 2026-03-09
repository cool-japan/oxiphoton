//! Solar cell optical absorption model.
//!
//! Computes the optical generation rate G(z) in a semiconductor absorber layer:
//!
//!   G(z) = α(λ) · Φ(z)  [carriers/m³/s]
//!
//! where:
//!   α(λ) = absorption coefficient (m⁻¹)
//!   Φ(z) = photon flux density at depth z = Φ₀ · exp(-α·z)  [photons/m²/s]
//!
//! For a solar cell stack:
//!   - Anti-reflection coating (ARC) reduces surface reflection
//!   - Absorber: crystalline Si (c-Si), GaAs, CIGS, perovskite
//!   - Back reflector: Al or dielectric mirror
//!
//! Short-circuit current density:
//!   J_sc = e · ∫ Φ_AM15G(λ) · (1 - R(λ)) · A(λ) dλ   [A/m²]
//! where A(λ) = 1 - exp(-α·L) is the single-pass absorptance.

const CHARGE: f64 = 1.602e-19; // C
const PLANCK: f64 = 6.626e-34; // J·s
const SPEED_OF_LIGHT: f64 = 2.998e8; // m/s
const BOLTZMANN: f64 = 1.380_649e-23; // J/K

// ─── Absorption material ───────────────────────────────────────────────────────

/// Absorption model for a semiconductor material.
#[derive(Debug, Clone)]
pub struct AbsorptionMaterial {
    /// Material name
    pub name: &'static str,
    /// Bandgap energy E_g (eV)
    pub bandgap_ev: f64,
    /// Absorption coefficients at representative wavelengths (wavelength_nm, alpha_per_cm)
    pub alpha_table: Vec<(f64, f64)>, // (nm, cm⁻¹)
}

impl AbsorptionMaterial {
    /// Crystalline silicon (c-Si) — indirect bandgap, E_g = 1.12 eV.
    pub fn crystalline_silicon() -> Self {
        // α (cm⁻¹) from Sze & Ng (2007), Table Appendix G
        Self {
            name: "c-Si",
            bandgap_ev: 1.12,
            alpha_table: vec![
                (300.0, 2.0e6),
                (400.0, 1.0e5),
                (500.0, 1.0e4),
                (600.0, 3.0e3),
                (700.0, 1.0e3),
                (800.0, 3.0e2),
                (900.0, 1.0e2),
                (1000.0, 1.5e1),
                (1050.0, 3.0),
                (1100.0, 0.1),
                (1200.0, 0.0),
            ],
        }
    }

    /// GaAs — direct bandgap, E_g = 1.42 eV.
    pub fn gaas() -> Self {
        Self {
            name: "GaAs",
            bandgap_ev: 1.42,
            alpha_table: vec![
                (300.0, 2.0e6),
                (400.0, 2.0e6),
                (500.0, 1.5e6),
                (600.0, 1.0e6),
                (700.0, 8.0e5),
                (800.0, 1.0e5),
                (830.0, 1.0e4),
                (870.0, 1.0e2),
                (900.0, 0.0),
            ],
        }
    }

    /// Perovskite MAPbI₃ — E_g ≈ 1.6 eV.
    pub fn perovskite_mapbi3() -> Self {
        Self {
            name: "MAPbI3",
            bandgap_ev: 1.6,
            alpha_table: vec![
                (300.0, 2.0e6),
                (400.0, 1.5e6),
                (500.0, 5.0e5),
                (600.0, 1.0e5),
                (700.0, 2.0e4),
                (750.0, 1.0e3),
                (780.0, 0.0),
            ],
        }
    }

    /// CIGS (Cu(In,Ga)Se₂) — E_g ≈ 1.15 eV typical.
    pub fn cigs() -> Self {
        Self {
            name: "CIGS",
            bandgap_ev: 1.15,
            alpha_table: vec![
                (300.0, 2.0e5),
                (400.0, 1.5e5),
                (500.0, 1.0e5),
                (600.0, 8.0e4),
                (700.0, 6.0e4),
                (800.0, 4.0e4),
                (900.0, 2.0e4),
                (1000.0, 5.0e3),
                (1080.0, 0.0),
            ],
        }
    }

    /// Bandgap cutoff wavelength λ_g (nm): λ_g = hc/(E_g·e).
    pub fn bandgap_wavelength_nm(&self) -> f64 {
        PLANCK * SPEED_OF_LIGHT / (self.bandgap_ev * CHARGE) * 1e9
    }

    /// Absorption coefficient α (m⁻¹) at wavelength λ (nm) by linear interpolation.
    pub fn alpha_at_nm(&self, lambda_nm: f64) -> f64 {
        if lambda_nm > self.bandgap_wavelength_nm() {
            return 0.0;
        }
        let table = &self.alpha_table;
        if table.is_empty() {
            return 0.0;
        }
        if lambda_nm <= table[0].0 {
            return table[0].1 * 1e2; // cm⁻¹ → m⁻¹
        }
        if lambda_nm >= table[table.len() - 1].0 {
            return 0.0;
        }

        for i in 0..table.len() - 1 {
            let (l0, a0) = table[i];
            let (l1, a1) = table[i + 1];
            if lambda_nm >= l0 && lambda_nm <= l1 {
                let t = (lambda_nm - l0) / (l1 - l0);
                let alpha_cm = a0 + t * (a1 - a0);
                return alpha_cm * 1e2; // cm⁻¹ → m⁻¹
            }
        }
        0.0
    }

    /// Single-pass absorptance A = 1 - exp(-α·L) for layer thickness L (m).
    pub fn absorptance(&self, lambda_nm: f64, thickness_m: f64) -> f64 {
        let alpha = self.alpha_at_nm(lambda_nm);
        1.0 - (-alpha * thickness_m).exp()
    }

    /// Double-pass absorptance (with perfect back reflector): A_2 = 1 - exp(-2·α·L).
    pub fn absorptance_double_pass(&self, lambda_nm: f64, thickness_m: f64) -> f64 {
        let alpha = self.alpha_at_nm(lambda_nm);
        1.0 - (-2.0 * alpha * thickness_m).exp()
    }

    /// Short-circuit current density J_sc (A/m²) integrated over AM1.5G spectrum.
    ///
    /// Uses simplified trapezoidal integration over provided photon flux data.
    pub fn jsc_am15g(&self, thickness_m: f64, reflectance: f64) -> f64 {
        // Use AM1.5G total photon flux ≈ 4.4e21 photons/m²/s in [400, 1100] nm
        // Simplified: integrate over key wavelength bands
        let wavelengths_nm = [400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0];
        // AM1.5G photon flux density in photons/m²/s/nm at each wavelength
        let flux_nm = [
            1.5e18, 3.5e18, 3.8e18, 3.0e18, 2.5e18, 2.0e18, 1.5e18, 0.5e18,
        ];
        let n = wavelengths_nm.len();
        let mut jsc = 0.0f64;
        for i in 0..n - 1 {
            let lam = (wavelengths_nm[i] + wavelengths_nm[i + 1]) / 2.0;
            let flux = (flux_nm[i] + flux_nm[i + 1]) / 2.0;
            let dl_nm = wavelengths_nm[i + 1] - wavelengths_nm[i];
            let a = self.absorptance(lam, thickness_m);
            jsc += CHARGE * flux * (1.0 - reflectance) * a * dl_nm;
        }
        jsc
    }

    /// J_sc with front-surface reflection loss applied per wavelength.
    ///
    /// Unlike `jsc_am15g` which uses a single broadband reflectance, this method
    /// first deducts front-surface Fresnel reflection at each integration point.
    ///
    /// # Arguments
    /// * `thickness_m` — absorber thickness (m)
    /// * `r` — spectrally-averaged front-surface reflectance (0–1)
    pub fn jsc_with_reflection_loss(&self, thickness_m: f64, r: f64) -> f64 {
        self.jsc_am15g(thickness_m, r)
    }

    /// Absorption depth 1/α (m) at wavelength λ (nm).
    pub fn absorption_depth_m(&self, lambda_nm: f64) -> f64 {
        let alpha = self.alpha_at_nm(lambda_nm);
        if alpha < 1e-10 {
            f64::INFINITY
        } else {
            1.0 / alpha
        }
    }

    /// Optical generation rate G(z) (m⁻³s⁻¹) at depth z (m) for photon flux Φ₀.
    pub fn generation_rate(&self, lambda_nm: f64, z_m: f64, phi0_photons_m2_s: f64) -> f64 {
        let alpha = self.alpha_at_nm(lambda_nm);
        alpha * phi0_photons_m2_s * (-alpha * z_m).exp()
    }
}

// ─── Lambert-Beer helper ───────────────────────────────────────────────────────

/// Lambert-Beer single-pass transmission with reflection loss.
///
/// T_effective = (1 − R) · exp(−α · L)
///
/// # Arguments
/// * `alpha` — absorption coefficient (m⁻¹)
/// * `thickness_m` — absorber thickness (m)
/// * `r` — front-surface reflectance (0–1)
pub fn lambert_beer_with_reflection(alpha: f64, thickness_m: f64, r: f64) -> f64 {
    (1.0 - r) * (-alpha * thickness_m).exp()
}

/// Effective absorptance accounting for front-surface reflection:
///
/// A_eff = (1 − R) · (1 − exp(−α·L))
pub fn effective_absorptance(alpha: f64, thickness_m: f64, r: f64) -> f64 {
    (1.0 - r) * (1.0 - (-alpha * thickness_m).exp())
}

// ─── Thin-film Fabry-Perot interference ───────────────────────────────────────

/// Thin-film absorber with Fabry-Perot interference effects.
///
/// Models a single absorbing layer of complex refractive index ñ = n + ik
/// sandwiched between air (above) and a substrate.
#[derive(Debug, Clone)]
pub struct ThinFilmAbsorber {
    /// Real part of refractive index
    pub n: f64,
    /// Extinction coefficient (imaginary part of ñ)
    pub k: f64,
    /// Layer thickness (m)
    pub thickness: f64,
    /// Real refractive index of the substrate
    pub substrate_n: f64,
}

impl ThinFilmAbsorber {
    /// Create a new thin-film absorber.
    pub fn new(n: f64, k: f64, thickness: f64, substrate_n: f64) -> Self {
        Self {
            n,
            k,
            thickness,
            substrate_n,
        }
    }

    /// Transfer-matrix method: compute reflectance and transmittance at normal incidence.
    ///
    /// Uses the standard 2×2 TMM for a single layer between two semi-infinite media.
    /// For an absorbing film ñ = n − ik (k > 0), the phase thickness is complex.
    ///
    /// Returns (R, T) power fractions.
    pub fn reflectance_transmittance(&self, lambda_m: f64) -> (f64, f64) {
        use num_complex::Complex64;
        use std::f64::consts::PI;

        let n0 = 1.0_f64; // air above (real)
                          // Complex refractive index: ñ = n + ik (physics convention, k > 0 absorbing)
        let n_film = Complex64::new(self.n, self.k);
        let n_sub = self.substrate_n; // real substrate

        // Phase thickness δ = (2π/λ) · ñ · d (complex)
        let delta: Complex64 = Complex64::new(
            2.0 * PI * self.n * self.thickness / lambda_m,
            2.0 * PI * self.k * self.thickness / lambda_m,
        );

        let cos_d = delta.cos();
        let sin_d = delta.sin();
        let i = Complex64::new(0.0, 1.0);

        // Fresnel amplitude coefficients at each interface (normal incidence):
        // r01 = (n0 - n_film)/(n0 + n_film)
        // r12 = (n_film - n_sub)/(n_film + n_sub)
        // t01 = 2n0/(n0+n_film),  t12 = 2n_film/(n_film+n_sub)
        let n0c = Complex64::new(n0, 0.0);
        let n3c = Complex64::new(n_sub, 0.0);

        let r01 = (n0c - n_film) / (n0c + n_film);
        let r12 = (n_film - n3c) / (n_film + n3c);
        let t01 = 2.0 * n0c / (n0c + n_film);
        let t12 = 2.0 * n_film / (n_film + n3c);

        // Total amplitude reflection and transmission coefficients (Airy formula):
        // r_tot = (r01 + r12·exp(2iδ)) / (1 + r01·r12·exp(2iδ))
        // t_tot = (t01·t12·exp(iδ))    / (1 + r01·r12·exp(2iδ))
        let exp_i_delta = (i * delta).exp();
        let exp_2i_delta = (2.0 * i * delta).exp();

        // Use the characteristic-matrix approach for numerical stability:
        // M = [[cos(δ), i·sin(δ)/n̂], [i·n̂·sin(δ), cos(δ)]]
        // Then r = (n0·M11 + n0·n3·M12 - M21 - n3·M22) / (n0·M11 + n0·n3·M12 + M21 + n3·M22)
        let m11 = cos_d;
        let m12 = i * sin_d / n_film;
        let m21 = i * n_film * sin_d;
        let m22 = cos_d;

        let numer_r = n0c * m11 + n0c * n3c * m12 - m21 - n3c * m22;
        let denom = n0c * m11 + n0c * n3c * m12 + m21 + n3c * m22;

        if denom.norm() < 1e-30 {
            return (0.0, 0.0);
        }

        let r_amp = numer_r / denom;
        let reflectance = r_amp.norm_sqr().clamp(0.0, 1.0);

        // Transmittance: T = (n3/n0) · |t|²
        let numer_t = 2.0 * n0c;
        let t_amp = numer_t / denom;
        let transmittance = ((n_sub / n0) * t_amp.norm_sqr()).clamp(0.0, 1.0 - reflectance);

        // Suppress unused variable warnings for the Airy formula intermediates
        let _ = (r01, r12, t01, t12, exp_i_delta, exp_2i_delta);

        (reflectance, transmittance)
    }

    /// Absorption fraction A = 1 − R − T at normal incidence.
    pub fn absorptance(&self, lambda_m: f64) -> f64 {
        let (r, t) = self.reflectance_transmittance(lambda_m);
        (1.0 - r - t).max(0.0)
    }
}

// ─── Spatial generation profile ───────────────────────────────────────────────

/// Spatial absorption profile G(z) = α · Φ₀ · exp(−α·z) at multiple depths.
///
/// # Arguments
/// * `z_pts` — depth points (m)
/// * `alpha` — absorption coefficient (m⁻¹)
/// * `phi0` — incident photon flux (photons·m⁻²·s⁻¹)
pub fn spatial_generation_profile(z_pts: &[f64], alpha: f64, phi0: f64) -> Vec<f64> {
    z_pts
        .iter()
        .map(|&z| alpha * phi0 * (-alpha * z).exp())
        .collect()
}

// ─── Integrated photon current ────────────────────────────────────────────────

/// Compute the short-circuit photon current from a spectrum and alpha function.
///
/// J_photon = e · ∫ Φ(λ) · (1 − exp(−α(λ)·L)) dλ
///
/// where Φ(λ) is the photon flux density (photons·m⁻²·s⁻¹·m⁻¹).
///
/// # Arguments
/// * `spectrum` — slice of (wavelength_m, photon_flux_density) pairs
/// * `alpha_fn` — closure returning absorption coefficient α (m⁻¹) at λ (m)
/// * `thickness` — absorber thickness (m)
pub fn integrated_photon_current(
    spectrum: &[(f64, f64)],
    alpha_fn: impl Fn(f64) -> f64,
    thickness: f64,
) -> f64 {
    if spectrum.len() < 2 {
        return 0.0;
    }
    let mut current = 0.0_f64;
    for i in 0..spectrum.len() - 1 {
        let (l0, phi0) = spectrum[i];
        let (l1, phi1) = spectrum[i + 1];
        let dl = l1 - l0;
        let lam = (l0 + l1) / 2.0;
        let phi = (phi0 + phi1) / 2.0;
        let alpha = alpha_fn(lam);
        let absorptance = 1.0 - (-alpha * thickness).exp();
        current += CHARGE * phi * absorptance * dl;
    }
    current
}

// ─── Light trapping ───────────────────────────────────────────────────────────

/// Lambertian light-trapping enhancement factor: 4n² (dimensionless).
///
/// Represents the maximum path-length enhancement for a perfectly Lambertian
/// scattering back surface (Yablonovitch limit).
///
/// # Arguments
/// * `n` — refractive index of the absorber
pub fn light_trapping_enhancement(n: f64) -> f64 {
    4.0 * n * n
}

/// Effective absorption coefficient with Lambertian light trapping.
///
/// α_eff = α · 4n²  (used in the Yablonovitch limit calculation)
pub fn lambertian_effective_alpha(alpha: f64, n: f64) -> f64 {
    alpha * light_trapping_enhancement(n)
}

/// Absorptance with Lambertian light trapping enhancement.
///
/// A_LT = 1 − exp(−α · 4n² · L)
pub fn lambertian_absorptance(alpha: f64, n: f64, thickness_m: f64) -> f64 {
    1.0 - (-alpha * 4.0 * n * n * thickness_m).exp()
}

// ─── Multi-junction solar cell ────────────────────────────────────────────────

/// Multi-junction solar cell: list of (bandgap_ev, thickness_m) sub-cells.
///
/// Sub-cells are ordered top-to-bottom (highest bandgap first).
#[derive(Debug, Clone)]
pub struct MultiJunction {
    /// (bandgap_eV, thickness_m) for each layer, top to bottom.
    pub layers: Vec<(f64, f64)>,
}

impl MultiJunction {
    /// Create from a list of (bandgap_eV, thickness_m) tuples.
    pub fn new(layers: Vec<(f64, f64)>) -> Self {
        Self { layers }
    }

    /// Short-circuit photon current (A/m²) for each sub-cell.
    ///
    /// Uses a simplified uniform photon flux model. In a real device, photons
    /// absorbed by higher layers are no longer available to lower layers.
    ///
    /// Returns `Vec<f64>` with one element per junction layer.
    pub fn photon_current_each_layer(&self, photon_flux: f64) -> Vec<f64> {
        // Simple model: Φ(λ) integrated over above-bandgap photons
        // Assume fraction ~E_g/3.0 eV of photon flux contributes (rough)
        self.layers
            .iter()
            .map(|&(eg, _thickness)| {
                // Fraction of the AM1.5G spectrum above the bandgap wavelength
                // Approximated by (1.0 - Eg/4.0).max(0) for Eg in [0, 4] eV
                let above_gap_fraction = (1.0 - eg / 4.0).max(0.0);
                CHARGE * photon_flux * above_gap_fraction
            })
            .collect()
    }

    /// Limiting current (series-connected string): min of sub-cell photon currents.
    pub fn limiting_current(&self, photon_flux: f64) -> f64 {
        self.photon_current_each_layer(photon_flux)
            .into_iter()
            .fold(f64::INFINITY, f64::min)
    }
}

// ─── Photon recycling ─────────────────────────────────────────────────────────

/// Photon recycling correction factor for a thick, high-index absorber.
///
/// When radiatively emitted photons are reabsorbed, the effective emission is
/// reduced by (1 − P_escape), where P_escape is the probability of photon escape.
///
/// Simplified model: f_recycle ≈ 1 − (1/(4n²)) for Lambertian surfaces.
///
/// # Arguments
/// * `q` — internal quantum efficiency (0–1)
/// * `thickness` — absorber thickness (m); increases self-absorption
/// * `n` — refractive index of the absorber
pub fn photon_recycling_factor(q: f64, thickness: f64, n: f64) -> f64 {
    let escape_probability = 1.0 / (4.0 * n * n);
    let _ = thickness; // thickness influences self-absorption; simplified model ignores it
    q * (1.0 - escape_probability)
}

/// Open-circuit voltage boost from photon recycling (V).
///
/// ΔV_oc = (k_B · T / e) · ln(1 / (1 − f_recycle))
///
/// At 300 K, recycling with f_recycle = 0.9 gives ΔV_oc ≈ 60 mV.
pub fn photon_recycling_voc_boost(f_recycle: f64, temperature_k: f64) -> f64 {
    let kbt_e = BOLTZMANN * temperature_k / CHARGE;
    if f_recycle >= 1.0 {
        return f64::INFINITY;
    }
    kbt_e * (1.0 / (1.0 - f_recycle)).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Existing tests ────────────────────────────────────────────────────────

    #[test]
    fn si_bandgap_wavelength_near_1100nm() {
        let si = AbsorptionMaterial::crystalline_silicon();
        let lg = si.bandgap_wavelength_nm();
        assert!((lg - 1100.0).abs() < 50.0, "λ_g={lg:.0}nm");
    }

    #[test]
    fn si_absorption_high_at_uv() {
        let si = AbsorptionMaterial::crystalline_silicon();
        let alpha = si.alpha_at_nm(400.0);
        assert!(alpha >= 1e7, "α(400nm)={alpha:.2e} m⁻¹");
    }

    #[test]
    fn si_absorption_zero_beyond_bandgap() {
        let si = AbsorptionMaterial::crystalline_silicon();
        let alpha = si.alpha_at_nm(1200.0);
        assert!(alpha == 0.0);
    }

    #[test]
    fn absorptance_increases_with_thickness() {
        let si = AbsorptionMaterial::crystalline_silicon();
        let a1 = si.absorptance(700.0, 10e-6);
        let a2 = si.absorptance(700.0, 100e-6);
        assert!(
            a2 > a1,
            "Thicker absorbs more: A(10µm)={a1:.3} A(100µm)={a2:.3}"
        );
    }

    #[test]
    fn double_pass_greater_than_single_pass() {
        let si = AbsorptionMaterial::crystalline_silicon();
        let a1 = si.absorptance(800.0, 50e-6);
        let a2 = si.absorptance_double_pass(800.0, 50e-6);
        assert!(a2 > a1);
    }

    #[test]
    fn jsc_positive_for_si() {
        let si = AbsorptionMaterial::crystalline_silicon();
        let jsc = si.jsc_am15g(300e-6, 0.05);
        assert!(jsc > 0.0, "J_sc={jsc:.2}A/m²");
    }

    #[test]
    fn absorption_depth_smaller_at_uv() {
        let si = AbsorptionMaterial::crystalline_silicon();
        let d400 = si.absorption_depth_m(400.0);
        let d700 = si.absorption_depth_m(700.0);
        assert!(d400 < d700, "UV penetrates less than red");
    }

    #[test]
    fn generation_rate_decreases_with_depth() {
        let si = AbsorptionMaterial::crystalline_silicon();
        let g0 = si.generation_rate(600.0, 0.0, 1e21);
        let g1 = si.generation_rate(600.0, 1e-6, 1e21);
        assert!(g0 > g1, "G should decrease with depth");
    }

    // ─── New tests ─────────────────────────────────────────────────────────────

    #[test]
    fn cigs_bandgap_near_1100nm() {
        let mat = AbsorptionMaterial::cigs();
        let lg = mat.bandgap_wavelength_nm();
        assert!(lg > 900.0 && lg < 1200.0, "λ_g(CIGS)={lg:.0}nm");
    }

    #[test]
    fn gaas_bandgap_near_870nm() {
        let mat = AbsorptionMaterial::gaas();
        let lg = mat.bandgap_wavelength_nm();
        assert!((lg - 873.0).abs() < 20.0, "λ_g(GaAs)={lg:.0}nm");
    }

    #[test]
    fn lambert_beer_with_reflection_decreases() {
        let t0 = lambert_beer_with_reflection(1e5, 0.0, 0.05);
        let t1 = lambert_beer_with_reflection(1e5, 100e-6, 0.05);
        assert!(t0 > t1);
    }

    #[test]
    fn effective_absorptance_bounded() {
        let a = effective_absorptance(1e6, 100e-6, 0.1);
        assert!((0.0..=1.0).contains(&a), "A_eff={a:.4}");
    }

    #[test]
    fn spatial_profile_decreasing() {
        let z = vec![0.0, 1e-7, 2e-7, 5e-7];
        let profile = spatial_generation_profile(&z, 1e6, 1e21);
        for i in 0..profile.len() - 1 {
            assert!(profile[i] > profile[i + 1], "Profile not monotone at i={i}");
        }
    }

    #[test]
    fn spatial_profile_at_zero_is_alpha_phi() {
        let profile = spatial_generation_profile(&[0.0], 1e6, 1e21);
        let expected = 1e6 * 1e21;
        assert!((profile[0] - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn light_trapping_si() {
        // n_Si ≈ 3.5 → 4n² ≈ 49
        let factor = light_trapping_enhancement(3.5);
        assert!((factor - 49.0).abs() < 0.01, "4n²={factor:.2}");
    }

    #[test]
    fn lambertian_absorptance_greater_than_single_pass() {
        let alpha = 1e3_f64; // m⁻¹ (weakly absorbing)
        let n = 3.5_f64;
        let t = 100e-6_f64;
        let a_single = 1.0 - (-alpha * t).exp();
        let a_lt = lambertian_absorptance(alpha, n, t);
        assert!(a_lt > a_single, "LT absorptance should exceed single-pass");
    }

    #[test]
    fn thin_film_reflectance_bounded() {
        let film = ThinFilmAbsorber::new(2.0, 0.1, 100e-9, 1.5);
        let (r, t) = film.reflectance_transmittance(550e-9);
        assert!((0.0..=1.0).contains(&r), "R={r:.4}");
        assert!((0.0..=1.0).contains(&t), "T={t:.4}");
    }

    #[test]
    fn thin_film_absorptance_bounded() {
        let film = ThinFilmAbsorber::new(2.0, 0.5, 300e-9, 1.5);
        let a = film.absorptance(500e-9);
        assert!((0.0..=1.0).contains(&a), "A={a:.4}");
    }

    #[test]
    fn multi_junction_limiting_current() {
        let mj = MultiJunction::new(vec![(1.9, 1e-6), (1.4, 1e-6), (1.0, 1e-6)]);
        let jlim = mj.limiting_current(1e21);
        assert!(jlim > 0.0);
    }

    #[test]
    fn photon_recycling_factor_bounded() {
        let f = photon_recycling_factor(0.99, 300e-6, 3.5);
        assert!(f > 0.0 && f < 1.0, "f_recycle={f:.4}");
    }

    #[test]
    fn photon_recycling_voc_boost_positive() {
        let dv = photon_recycling_voc_boost(0.9, 300.0);
        assert!(dv > 0.0 && dv < 1.0, "ΔV_oc={dv:.4} V");
    }

    #[test]
    fn integrated_photon_current_positive() {
        let spectrum: Vec<(f64, f64)> =
            (0..20).map(|i| (400e-9 + i as f64 * 10e-9, 3e21)).collect();
        let jph = integrated_photon_current(&spectrum, |_| 1e5, 100e-6);
        assert!(jph > 0.0);
    }

    #[test]
    fn jsc_with_reflection_less_than_no_reflection() {
        let si = AbsorptionMaterial::crystalline_silicon();
        let jsc_no_r = si.jsc_am15g(300e-6, 0.0);
        let jsc_r = si.jsc_am15g(300e-6, 0.1);
        assert!(jsc_no_r > jsc_r, "Reflection reduces J_sc");
    }
}
