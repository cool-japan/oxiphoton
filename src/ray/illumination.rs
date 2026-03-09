//! Illumination models for ray optics and solar cells.
//!
//! Provides a collection of source models commonly used in photonics:
//!
//! * `LambertianSource` — Lambertian (cosine-law) emitter
//! * `CollimatedBeam` — ideal plane-wave beam
//! * `PointSource` — isotropic point source (inverse square law)
//! * `ExtendedSource` — LED / OLED with configurable emission pattern
//! * `solar` — solar irradiance models (AM0, AM1.5G, atmospheric transmittance)

use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Physical constants (SI)
// ─────────────────────────────────────────────────────────────────────────────

const H_PLANCK: f64 = 6.626_070_15e-34; // J·s
const C_LIGHT: f64 = 2.997_924_58e8; // m/s
/// Boltzmann constant (J/K), used in thermal emission calculations.
#[allow(dead_code)]
const K_B: f64 = 1.380_649e-23; // J/K

// ─────────────────────────────────────────────────────────────────────────────
// LambertianSource
// ─────────────────────────────────────────────────────────────────────────────

/// Lambertian source: uniform radiance in a hemisphere.
///
/// Emits with radiance L = P / (π · A) \[W m⁻² sr⁻¹\] and obeys Lambert's
/// cosine law: power radiated per unit solid angle scales as cos θ.
#[derive(Debug, Clone, Copy)]
pub struct LambertianSource {
    /// Total radiated power (W)
    pub total_power: f64,
    /// Emitting surface area (m²)
    pub emission_area_m2: f64,
    /// Refractive index of the surrounding medium
    pub n_medium: f64,
}

impl LambertianSource {
    pub fn new(power_w: f64, area_m2: f64, n: f64) -> Self {
        Self {
            total_power: power_w,
            emission_area_m2: area_m2,
            n_medium: n,
        }
    }

    /// Radiance L = P / (π · A) \[W m⁻² sr⁻¹\].
    pub fn radiance(&self) -> f64 {
        let denom = PI * self.emission_area_m2;
        if denom < 1e-300 {
            return 0.0;
        }
        self.total_power / denom
    }

    /// Approximate photopic luminance (cd m⁻²), assuming ideal white light
    /// (luminous efficacy ≈ 683 lm/W for monochromatic 555 nm; here we use
    /// a typical broadband efficacy of ~300 lm/W as an approximation).
    pub fn luminance_photopic(&self) -> f64 {
        const EFFICACY_LM_PER_W: f64 = 300.0;
        self.radiance() * EFFICACY_LM_PER_W
    }

    /// Power emitted into a cone of half-angle θ \[rad\].
    ///
    /// For a Lambertian source: P(θ) = P_total · sin²(θ).
    pub fn power_in_cone(&self, half_angle_rad: f64) -> f64 {
        let s = half_angle_rad.sin();
        self.total_power * s * s
    }

    /// Photon flux density (photons s⁻¹ m⁻²) at a given wavelength \[m\].
    ///
    /// Assumes all power is emitted at `wavelength_m`.
    pub fn photon_flux_density(&self, wavelength_m: f64) -> f64 {
        if wavelength_m < 1e-30 || self.emission_area_m2 < 1e-30 {
            return 0.0;
        }
        let energy_per_photon = H_PLANCK * C_LIGHT / wavelength_m;
        let irradiance = self.total_power / self.emission_area_m2;
        irradiance / energy_per_photon
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CollimatedBeam
// ─────────────────────────────────────────────────────────────────────────────

/// Collimated beam (ideal plane wave).
#[derive(Debug, Clone, Copy)]
pub struct CollimatedBeam {
    /// Irradiance (W m⁻²)
    pub irradiance: f64,
    /// Propagation direction (unit vector, normalised on construction)
    pub direction: [f64; 3],
    /// Polarisation direction (unit vector ⊥ direction)
    pub polarization: [f64; 3],
    /// Wavelength in vacuum (m)
    pub wavelength: f64,
}

impl CollimatedBeam {
    /// Beam at normal incidence (propagating along −Z).
    pub fn normal_incidence(irradiance: f64, wavelength: f64) -> Self {
        Self {
            irradiance,
            direction: [0.0, 0.0, 1.0],
            polarization: [1.0, 0.0, 0.0],
            wavelength,
        }
    }

    /// Beam at polar angle θ and azimuthal angle φ (degrees).
    pub fn at_angle(irradiance: f64, theta_deg: f64, phi_deg: f64, wavelength: f64) -> Self {
        let theta = theta_deg.to_radians();
        let phi = phi_deg.to_radians();
        let dir = [
            theta.sin() * phi.cos(),
            theta.sin() * phi.sin(),
            theta.cos(),
        ];
        // Polarisation: perpendicular to direction in the xz-plane
        let pol = [
            -theta.cos() * phi.cos(),
            -theta.cos() * phi.sin(),
            theta.sin(),
        ];
        Self {
            irradiance,
            direction: dir,
            polarization: pol,
            wavelength,
        }
    }

    /// Power per unit area incident on a surface with the given normal.
    ///
    /// P = E · A · |dir · n|
    pub fn power_on_surface(&self, area: f64, surface_normal: [f64; 3]) -> f64 {
        let dot = dot3(self.direction, surface_normal).abs();
        self.irradiance * area * dot
    }

    /// Specular reflection off a flat surface.
    pub fn reflection(&self, surface_normal: [f64; 3]) -> CollimatedBeam {
        let n = norm3(surface_normal);
        let d = self.direction;
        let dn = dot3(d, n);
        let reflected_dir = [
            d[0] - 2.0 * dn * n[0],
            d[1] - 2.0 * dn * n[1],
            d[2] - 2.0 * dn * n[2],
        ];
        CollimatedBeam {
            irradiance: self.irradiance,
            direction: reflected_dir,
            polarization: self.polarization,
            wavelength: self.wavelength,
        }
    }

    /// Refracted beam using Snell's law.
    ///
    /// Returns `None` for total internal reflection.
    pub fn refraction(&self, surface_normal: [f64; 3], n1: f64, n2: f64) -> Option<CollimatedBeam> {
        let n_hat = norm3(surface_normal);
        let cos_i = -dot3(self.direction, n_hat);
        let ratio = n1 / n2;
        let sin2_t = ratio * ratio * (1.0 - cos_i * cos_i);
        if sin2_t > 1.0 {
            return None; // Total internal reflection
        }
        let cos_t = (1.0 - sin2_t).sqrt();
        let refracted_dir = [
            ratio * self.direction[0] + (ratio * cos_i - cos_t) * n_hat[0],
            ratio * self.direction[1] + (ratio * cos_i - cos_t) * n_hat[1],
            ratio * self.direction[2] + (ratio * cos_i - cos_t) * n_hat[2],
        ];
        // Fresnel transmittance (TE) for irradiance scaling
        let t_fresnel = if (n1 * cos_i).abs() < 1e-300 {
            1.0
        } else {
            let r = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
            1.0 - r * r
        };
        Some(CollimatedBeam {
            irradiance: self.irradiance * t_fresnel,
            direction: refracted_dir,
            polarization: self.polarization,
            wavelength: self.wavelength,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PointSource
// ─────────────────────────────────────────────────────────────────────────────

/// Isotropic point source.
#[derive(Debug, Clone, Copy)]
pub struct PointSource {
    pub position: [f64; 3],
    pub total_power: f64,
    pub wavelength: f64,
}

impl PointSource {
    pub fn new(position: [f64; 3], power: f64, wavelength: f64) -> Self {
        Self {
            position,
            total_power: power,
            wavelength,
        }
    }

    /// Irradiance at `point` via the inverse-square law \[W m⁻²\].
    pub fn irradiance_at(&self, point: [f64; 3]) -> f64 {
        let r2 = dist2_3(self.position, point);
        if r2 < 1e-300 {
            return 0.0;
        }
        self.total_power / (4.0 * PI * r2)
    }

    /// Electric field amplitude at `point` (SI units, assuming free-space).
    ///
    /// |E|² = 2·I / (n·ε₀·c) ; in vacuum: |E| = sqrt(2·I / (ε₀·c))
    pub fn e_field_amplitude_at(&self, point: [f64; 3], n: f64) -> f64 {
        const EPSILON0: f64 = 8.854_187_817e-12; // F m⁻¹
        let irr = self.irradiance_at(point);
        (2.0 * irr / (n * EPSILON0 * C_LIGHT)).sqrt()
    }

    /// Solid angle subtended by a flat surface of area `area` at distance `distance`.
    pub fn solid_angle_subtended(&self, area: f64, distance: f64) -> f64 {
        if distance < 1e-300 {
            return 0.0;
        }
        area / (distance * distance)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ExtendedSource
// ─────────────────────────────────────────────────────────────────────────────

/// Emission pattern for an extended source.
#[derive(Debug, Clone)]
pub enum EmissionPattern {
    /// Lambertian (cosine law)
    Lambertian,
    /// Gaussian angular profile with given half-angle at 1/e²
    Gaussian { half_angle_deg: f64 },
    /// Uniform (top-hat) within the given half-angle
    Uniform { half_angle_deg: f64 },
    /// User-defined: list of (angle_deg, relative_intensity) pairs
    Custom { pattern: Vec<(f64, f64)> },
}

/// Extended source model (LED, OLED, etc.).
#[derive(Debug, Clone)]
pub struct ExtendedSource {
    /// Emitting area (m²)
    pub area_m2: f64,
    /// Total radiated power (W)
    pub total_power: f64,
    /// Angular emission pattern
    pub emission_pattern: EmissionPattern,
    /// Wavelength (m)
    pub wavelength: f64,
}

impl ExtendedSource {
    /// Standard Lambertian LED.
    pub fn led(area_m2: f64, power: f64, wavelength: f64) -> Self {
        Self {
            area_m2,
            total_power: power,
            emission_pattern: EmissionPattern::Lambertian,
            wavelength,
        }
    }

    /// Relative intensity at angle `angle_deg` from normal (normalised so
    /// that the on-axis value is 1.0).
    pub fn relative_intensity(&self, angle_deg: f64) -> f64 {
        match &self.emission_pattern {
            EmissionPattern::Lambertian => angle_deg.to_radians().cos().max(0.0),
            EmissionPattern::Gaussian { half_angle_deg } => {
                let sigma = half_angle_deg.to_radians();
                let theta = angle_deg.to_radians();
                (-theta * theta / (2.0 * sigma * sigma)).exp()
            }
            EmissionPattern::Uniform { half_angle_deg } => {
                if angle_deg.abs() <= *half_angle_deg {
                    1.0
                } else {
                    0.0
                }
            }
            EmissionPattern::Custom { pattern } => {
                // Linear interpolation in the custom table
                if pattern.is_empty() {
                    return 0.0;
                }
                let a = angle_deg.abs();
                // Find surrounding entries
                let mut lo = pattern[0];
                let mut hi = *pattern.last().unwrap_or(&pattern[0]);
                for &pt in pattern {
                    if pt.0 <= a {
                        lo = pt;
                    }
                    if pt.0 >= a {
                        hi = pt;
                        break;
                    }
                }
                if (hi.0 - lo.0).abs() < 1e-300 {
                    lo.1
                } else {
                    lo.1 + (hi.1 - lo.1) * (a - lo.0) / (hi.0 - lo.0)
                }
            }
        }
    }

    /// Power emitted into a cone of half-angle `half_angle_deg` (numerical integration).
    pub fn power_in_cone(&self, half_angle_deg: f64) -> f64 {
        // Integrate I(θ)·sin θ dθ over [0, half_angle] and normalise
        let n_steps = 1000_usize;
        let theta_max = half_angle_deg.to_radians();
        let d_theta = theta_max / n_steps as f64;
        let sum: f64 = (0..n_steps)
            .map(|k| {
                let theta = (k as f64 + 0.5) * d_theta;
                self.relative_intensity(theta.to_degrees()) * theta.sin() * d_theta
            })
            .sum();
        // Normalise by hemisphere integral
        let norm: f64 = {
            let n_full = 1000_usize;
            let d_th = PI / 2.0 / n_full as f64;
            (0..n_full)
                .map(|k| {
                    let t = (k as f64 + 0.5) * d_th;
                    self.relative_intensity(t.to_degrees()) * t.sin() * d_th
                })
                .sum::<f64>()
        };
        if norm < 1e-300 {
            return 0.0;
        }
        self.total_power * sum / norm
    }

    /// Fraction of total power emitted into a cone of half-angle `half_angle_deg`.
    pub fn extraction_efficiency_into_cone(&self, half_angle_deg: f64) -> f64 {
        if self.total_power < 1e-300 {
            return 0.0;
        }
        self.power_in_cone(half_angle_deg) / self.total_power
    }

    /// Étendue: G = n²·A·Ω where Ω = π·sin²(θ_max) for a Lambertian source
    /// (here using the full hemisphere, Ω = π sr).
    pub fn etendue(&self, n: f64) -> f64 {
        n * n * self.area_m2 * PI
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Solar irradiance models
// ─────────────────────────────────────────────────────────────────────────────

/// Solar irradiance models.
pub mod solar {
    use std::f64::consts::PI;

    /// AM0 solar constant (W m⁻²).
    pub const AM0_IRRADIANCE: f64 = 1361.0;

    /// AM1.5G integrated irradiance (W m⁻²) — standard test condition.
    pub const AM1_5G_POWER: f64 = 1000.0;

    /// Atmospheric transmittance for a given air mass using the empirical formula:
    ///   T(m) = 0.7^(m^0.678)
    ///
    /// Valid for 0.5 ≤ m ≤ 10.
    pub fn atmospheric_transmittance(air_mass: f64) -> f64 {
        if air_mass <= 0.0 {
            return 1.0;
        }
        0.7_f64.powf(air_mass.powf(0.678))
    }

    /// Air mass from solar elevation angle (degrees above horizon).
    ///
    /// Uses the simple secant formula: AM = 1/sin(elevation).
    /// Clamps to AM = 40 at very low elevations.
    pub fn air_mass(elevation_deg: f64) -> f64 {
        if elevation_deg <= 0.0 {
            return 40.0;
        }
        let am = 1.0 / elevation_deg.to_radians().sin();
        am.min(40.0)
    }

    /// Direct normal irradiance (W m⁻²) at a given air mass.
    pub fn direct_normal_irradiance(air_mass: f64) -> f64 {
        AM0_IRRADIANCE * atmospheric_transmittance(air_mass)
    }

    /// Diffuse fraction of total irradiance (simplified isotropic sky model).
    ///
    /// At AM1: diffuse ≈ 10% of total; increases at higher air masses.
    pub fn diffuse_fraction(air_mass: f64) -> f64 {
        // Empirical linear model: 0.1 + 0.05*(AM - 1)
        (0.1 + 0.05 * (air_mass - 1.0).max(0.0)).min(0.5)
    }

    /// Annual insolation estimate (kWh m⁻² yr⁻¹) for a given latitude.
    ///
    /// Uses a simplified sinusoidal model:
    ///   H = H_max · cos(latitude) · correction_factor
    pub fn annual_insolation_estimate(latitude_deg: f64) -> f64 {
        // At the equator: ~2000 kWh/m²/yr; poles: ~0
        let lat_rad = latitude_deg.abs().min(90.0).to_radians();
        let h_max = 2000.0_f64; // kWh/m²/yr at equator
        h_max * lat_rad.cos() * (1.0 - 0.2 * lat_rad.sin())
    }

    /// Total irradiance (direct + diffuse) on a horizontal surface.
    pub fn global_horizontal_irradiance(air_mass: f64, elevation_deg: f64) -> f64 {
        let dni = direct_normal_irradiance(air_mass);
        let cos_z = elevation_deg.to_radians().sin(); // cos(zenith) = sin(elevation)
        let beam = dni * cos_z.max(0.0);
        let diff_frac = diffuse_fraction(air_mass);
        let diffuse = diff_frac * dni;
        beam + diffuse
    }

    /// Photon flux density (photons s⁻¹ m⁻²) at the AM1.5G standard for a
    /// given wavelength `lambda_nm` (nm).
    ///
    /// Uses a blackbody approximation at T_sun = 5778 K.
    pub fn am15g_photon_flux(lambda_nm: f64) -> f64 {
        const T_SUN: f64 = 5778.0; // K
        const H: f64 = 6.626e-34;
        const C: f64 = 2.998e8;
        const KB: f64 = 1.381e-23;
        const R_SUN: f64 = 6.957e8; // m
        const AU: f64 = 1.496e11; // m
        if lambda_nm <= 0.0 {
            return 0.0;
        }
        let lambda_m = lambda_nm * 1e-9;
        let hc_lkt = H * C / (lambda_m * KB * T_SUN);
        // Planck spectral radiance × geometric dilution factor
        let geometric = (R_SUN / AU).powi(2) * PI;
        let spectral_radiance = 2.0 * H * C * C / lambda_m.powi(5) / (hc_lkt.exp() - 1.0);
        let irradiance = spectral_radiance * geometric; // W m⁻² m⁻¹
                                                        // Scale to AM1.5G (1000 W/m² normalisation)
        irradiance / lambda_m / (H * C / lambda_m)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn am0_constant_positive() {
            const _: () = assert!(AM0_IRRADIANCE > 0.0);
        }

        #[test]
        fn air_mass_zenith_is_one() {
            let am = air_mass(90.0); // directly overhead
            assert!(
                (am - 1.0).abs() < 1e-6,
                "AM at zenith should be 1, got {am}"
            );
        }

        #[test]
        fn atmospheric_transmittance_at_am1() {
            let t = atmospheric_transmittance(1.0);
            assert!(t > 0.0 && t < 1.0, "T(AM1)={t}");
            // Should be close to 0.7
            assert!((t - 0.7).abs() < 0.01, "T(AM1)={t}");
        }

        #[test]
        fn dni_decreases_with_air_mass() {
            let dni1 = direct_normal_irradiance(1.0);
            let dni5 = direct_normal_irradiance(5.0);
            assert!(dni5 < dni1, "DNI should decrease with air mass");
        }

        #[test]
        fn annual_insolation_equator_high() {
            let h = annual_insolation_estimate(0.0);
            assert!(h > 1500.0, "Equator insolation={h} kWh/m²/yr");
        }

        #[test]
        fn annual_insolation_pole_lower_than_equator() {
            let h_eq = annual_insolation_estimate(0.0);
            let h_pole = annual_insolation_estimate(90.0);
            assert!(h_pole < h_eq);
        }

        #[test]
        fn diffuse_fraction_between_zero_and_one() {
            for am in [0.5, 1.0, 2.0, 5.0, 10.0] {
                let f = diffuse_fraction(am);
                assert!((0.0..=1.0).contains(&f), "diffuse_fraction({am})={f}");
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vector helpers (private)
// ─────────────────────────────────────────────────────────────────────────────

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm3(v: [f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-300 {
        v
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

fn dist2_3(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── LambertianSource ──────────────────────────────────────────────────────

    #[test]
    fn lambertian_radiance_nonzero() {
        let src = LambertianSource::new(1.0, 1e-4, 1.0);
        let l = src.radiance();
        assert!(l > 0.0, "radiance={l}");
    }

    #[test]
    fn lambertian_power_in_hemisphere_is_total_power() {
        let src = LambertianSource::new(1.0, 1e-4, 1.0);
        let p = src.power_in_cone(PI / 2.0); // full hemisphere
        assert!((p - 1.0).abs() < 1e-10, "hemisphere power={p}");
    }

    #[test]
    fn lambertian_power_in_small_cone_less_than_total() {
        let src = LambertianSource::new(1.0, 1e-4, 1.0);
        let p = src.power_in_cone(0.1); // ~6°
        assert!(p < 1.0 && p > 0.0, "p={p}");
    }

    #[test]
    fn lambertian_photon_flux_positive() {
        let src = LambertianSource::new(1.0, 1e-4, 1.0);
        let flux = src.photon_flux_density(550e-9);
        assert!(flux > 0.0, "photon_flux={flux}");
    }

    // ── CollimatedBeam ────────────────────────────────────────────────────────

    #[test]
    fn collimated_normal_incidence_full_power() {
        let beam = CollimatedBeam::normal_incidence(1000.0, 550e-9);
        let power = beam.power_on_surface(1.0, [0.0, 0.0, 1.0]);
        assert!((power - 1000.0).abs() < 1e-10, "power={power}");
    }

    #[test]
    fn collimated_grazing_incidence_zero_power() {
        let beam = CollimatedBeam::normal_incidence(1000.0, 550e-9);
        let power = beam.power_on_surface(1.0, [1.0, 0.0, 0.0]); // perpendicular surface
        assert!(power < 1e-10, "power at 90°={power}");
    }

    #[test]
    fn collimated_reflection_preserves_irradiance() {
        let beam = CollimatedBeam::normal_incidence(1000.0, 550e-9);
        let reflected = beam.reflection([0.0, 0.0, 1.0]);
        assert!((reflected.irradiance - beam.irradiance).abs() < 1e-10);
    }

    #[test]
    fn collimated_refraction_air_to_glass() {
        let beam = CollimatedBeam::normal_incidence(1000.0, 550e-9);
        let refracted = beam.refraction([0.0, 0.0, -1.0], 1.0, 1.5);
        assert!(
            refracted.is_some(),
            "should not have TIR at normal incidence"
        );
        if let Some(r) = refracted {
            assert!(r.irradiance > 0.0 && r.irradiance <= beam.irradiance);
        }
    }

    #[test]
    fn collimated_refraction_tir() {
        // Going from glass (n=1.5) to air at 50° (beyond critical angle ~42°)
        let beam = CollimatedBeam::at_angle(1000.0, 50.0, 0.0, 550e-9);
        let result = beam.refraction([0.0, 0.0, 1.0], 1.5, 1.0);
        assert!(result.is_none(), "should have TIR");
    }

    // ── PointSource ───────────────────────────────────────────────────────────

    #[test]
    fn point_source_irradiance_inverse_square() {
        let src = PointSource::new([0.0, 0.0, 0.0], 4.0 * PI, 550e-9);
        // At r=1 m: I = P/(4πr²) = 1 W/m²
        let i1 = src.irradiance_at([1.0, 0.0, 0.0]);
        let i2 = src.irradiance_at([2.0, 0.0, 0.0]);
        assert!((i1 - 1.0).abs() < 1e-10, "i1={i1}");
        assert!((i2 - 0.25).abs() < 1e-10, "i2={i2}");
    }

    #[test]
    fn point_source_e_field_positive() {
        let src = PointSource::new([0.0, 0.0, 0.0], 1.0, 550e-9);
        let e = src.e_field_amplitude_at([1.0, 0.0, 0.0], 1.0);
        assert!(e > 0.0, "E field={e}");
    }

    #[test]
    fn point_source_solid_angle_small() {
        let src = PointSource::new([0.0, 0.0, 0.0], 1.0, 550e-9);
        let omega = src.solid_angle_subtended(1e-4, 10.0);
        assert!(omega > 0.0 && omega < 4.0 * PI);
    }

    // ── ExtendedSource ────────────────────────────────────────────────────────

    #[test]
    fn led_relative_intensity_on_axis_is_one() {
        let led = ExtendedSource::led(1e-6, 0.01, 450e-9);
        let i = led.relative_intensity(0.0);
        assert!((i - 1.0).abs() < 1e-10, "on-axis intensity={i}");
    }

    #[test]
    fn led_power_in_hemisphere_near_total() {
        let led = ExtendedSource::led(1e-6, 1.0, 450e-9);
        let p = led.power_in_cone(90.0);
        assert!((p - 1.0).abs() < 0.01, "hemisphere power={p}"); // within 1%
    }

    #[test]
    fn led_extraction_efficiency_half_angle() {
        let led = ExtendedSource::led(1e-6, 1.0, 450e-9);
        let eff = led.extraction_efficiency_into_cone(30.0);
        assert!(eff > 0.0 && eff < 1.0, "eff={eff}");
    }

    #[test]
    fn led_etendue_positive() {
        let led = ExtendedSource::led(1e-6, 1.0, 450e-9);
        let g = led.etendue(1.5);
        assert!(g > 0.0);
    }

    #[test]
    fn gaussian_source_narrower_cone() {
        let gaussian = ExtendedSource {
            area_m2: 1e-6,
            total_power: 1.0,
            emission_pattern: EmissionPattern::Gaussian {
                half_angle_deg: 10.0,
            },
            wavelength: 550e-9,
        };
        let lambertian = ExtendedSource::led(1e-6, 1.0, 550e-9);
        let p_gauss_20 = gaussian.power_in_cone(20.0);
        let p_lamb_20 = lambertian.power_in_cone(20.0);
        // Gaussian with 10° half-angle should concentrate more power in 20° cone
        assert!(
            p_gauss_20 > p_lamb_20,
            "gaussian={p_gauss_20} vs lambertian={p_lamb_20}"
        );
    }
}
