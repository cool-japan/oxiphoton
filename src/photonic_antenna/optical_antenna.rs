//! Optical nanoantenna theory — Hertzian dipole, metallic nanorod, and
//! optical Yagi-Uda antenna.
//!
//! All physical quantities use SI units throughout.  Wavelengths in metres,
//! frequencies in rad/s, powers in watts, field enhancements dimensionless.
use std::f64::consts::PI;

// ─── Physical constants ───────────────────────────────────────────────────────

const C_LIGHT: f64 = 2.997_924_58e8; // m/s
const EPS0: f64 = 8.854_187_817e-12; // F/m
const HBAR: f64 = 1.054_571_817e-34; // J·s
const ETA0: f64 = 376.730_313_668; // Ω (free-space impedance)

// ─── HertzianDipole ──────────────────────────────────────────────────────────

/// Hertzian (infinitesimal) dipole antenna operating at optical frequency.
///
/// The Hertzian dipole is the fundamental radiating element.  Its analytical
/// solution (exact for k·L → 0) provides exact radiation formulas that
/// serve as reference for nanoantenna design.
///
/// Reference: Novotny & Hecht, *Principles of Nano-Optics*, Ch. 2.
#[derive(Debug, Clone)]
pub struct HertzianDipole {
    /// Free-space wavelength of the excitation field (m)
    pub wavelength_m: f64,
    /// Magnitude of the electric dipole moment |p| (C·m)
    pub dipole_moment_cm: f64,
    /// Real refractive index of the surrounding medium
    pub n_medium: f64,
}

impl HertzianDipole {
    /// Angular frequency from wavelength: ω = 2π c / λ
    pub fn omega(&self) -> f64 {
        2.0 * PI * C_LIGHT / self.wavelength_m
    }

    /// Total radiated power in watts.
    ///
    /// P = |p|² ω⁴ n / (12π ε₀ c³)
    ///
    /// Derived from Larmor formula extended to a medium with index n.
    pub fn radiated_power_w(&self, omega: f64) -> f64 {
        let p = self.dipole_moment_cm;
        let n = self.n_medium;
        // P = p² ω⁴ n / (12π ε₀ c³)
        p * p * omega.powi(4) * n / (12.0 * PI * EPS0 * C_LIGHT.powi(3))
    }

    /// Directivity of a Hertzian dipole: D = 1.5 (independent of frequency)
    pub fn directivity(&self) -> f64 {
        1.5
    }

    /// Radiation resistance in Ohms.
    ///
    /// R_rad = (π/3) (λ_eff/L)⁻² × (η₀/n)
    ///
    /// where λ_eff = λ/n is the wavelength in the medium.
    ///
    /// * `length_m` — physical length of the equivalent short dipole (L ≪ λ)
    pub fn radiation_resistance_ohm(&self, length_m: f64) -> f64 {
        let lambda_eff = self.wavelength_m / self.n_medium;
        let eta = ETA0 / self.n_medium;
        // R_rad = (2π/3) η (L/λ_eff)²
        (2.0 * PI / 3.0) * eta * (length_m / lambda_eff).powi(2)
    }

    /// Near-field intensity (proportional) at distance r from the dipole.
    ///
    /// In the near zone (r ≪ λ) the dominant term scales as 1/r⁶:
    ///
    ///   I_NF ∝ |p|² / r⁶
    ///
    /// Returns units of (C²·m²·m⁻⁶) = C²/m⁴ — multiply by (ω⁴/(ε₀ c)) for
    /// absolute power density, but here only the r-dependence is needed.
    pub fn near_field_intensity(&self, r_m: f64) -> f64 {
        if r_m < 1.0e-30 {
            return f64::MAX;
        }
        let p = self.dipole_moment_cm;
        p * p / r_m.powi(6)
    }

    /// Far-field radiation pattern (intensity angular dependence).
    ///
    /// f(θ) = sin²θ  where θ is the polar angle from the dipole axis.
    ///
    /// Normalised so that the maximum (θ = π/2) equals 1.
    pub fn far_field_pattern(&self, theta_rad: f64) -> f64 {
        theta_rad.sin().powi(2)
    }

    /// Zero-point fluctuation power from quantum vacuum via Fermi's golden rule.
    ///
    /// Γ_vac = |p|² ω³ n / (3π ε₀ ħ c³)
    ///
    /// Returns the spontaneous emission rate in s⁻¹ (natural linewidth).
    pub fn spontaneous_emission_rate(&self) -> f64 {
        let omega = self.omega();
        let p = self.dipole_moment_cm;
        let n = self.n_medium;
        p * p * omega.powi(3) * n / (3.0 * PI * EPS0 * HBAR * C_LIGHT.powi(3))
    }
}

// ─── NanorodAntenna ──────────────────────────────────────────────────────────

/// Metallic nanorod antenna operating near its λ/2 plasmonic resonance.
///
/// The effective wavelength approximation (Novotny 2007) accounts for the
/// strong field penetration into the metal, giving λ_eff ≪ λ₀.
///
/// Physical quantities:
/// - Field enhancement: |E|/|E₀| at the tip (lightning-rod effect)
/// - Cross sections: absorption and scattering
/// - SERS enhancement: EF ≈ |E/E₀|⁴
/// - Radiation efficiency: η = Γ_rad / (Γ_rad + Γ_abs)
#[derive(Debug, Clone)]
pub struct NanorodAntenna {
    /// Rod length in metres (L ≈ λ_eff/2 at resonance)
    pub length_m: f64,
    /// Rod radius in metres
    pub radius_m: f64,
    /// Complex permittivity of the metal: (Re ε, Im ε)  — Drude model
    pub eps_metal: (f64, f64),
    /// Free-space wavelength of the excitation (m)
    pub wavelength_m: f64,
    /// Real refractive index of the surrounding medium
    pub n_medium: f64,
}

impl NanorodAntenna {
    /// Effective wavelength inside the nanorod using Novotny's approximation.
    ///
    /// The effective wavelength λ_eff is shorter than the free-space wavelength
    /// due to field penetration into the metal (plasmonic confinement).
    ///
    /// Following Novotny (2007), the antenna resonance condition L = λ_eff/2
    /// involves an effective wavelength determined by the complex SPP dispersion:
    ///
    ///   λ_eff = λ_0 / n_eff
    ///
    /// where n_eff is derived from the SPP propagation constant on a metallic
    /// wire.  For a thin rod with radius a and metal permittivity ε_m in a
    /// medium of index n, the quasi-static limit gives:
    ///
    ///   k_spp ≈ k_0 × sqrt( −ε_m × n² / (ε_m + n²) )  \[complex\]
    ///
    /// The real part of k_spp / k_0 defines n_eff.  For typical metals
    /// (Re ε_m ≪ −1 at optical frequencies) this yields n_eff > n_medium.
    pub fn effective_wavelength_m(&self) -> f64 {
        let (eps_re, eps_im) = self.eps_metal;
        let n2 = self.n_medium * self.n_medium;

        // SPP k-vector squared: k_spp² = k₀² × ε_m n² / (ε_m + n²)
        // Re and Im parts of (ε_m × n²) / (ε_m + n²)
        let denom_re = eps_re + n2;
        let denom_im = eps_im;
        let denom_sq = denom_re * denom_re + denom_im * denom_im;

        if denom_sq < f64::MIN_POSITIVE {
            return self.wavelength_m / self.n_medium;
        }

        // Numerator: ε_m × n² = (eps_re + i eps_im) × n²
        let num_re = eps_re * n2;
        let num_im = eps_im * n2;

        // Complex ratio: (num_re + i num_im) / (denom_re + i denom_im)
        let ratio_re = (num_re * denom_re + num_im * denom_im) / denom_sq;
        let ratio_im = (num_im * denom_re - num_re * denom_im) / denom_sq;

        // n_eff = Re[ sqrt(ratio) ], using sqrt of a complex number
        let r_mag = (ratio_re * ratio_re + ratio_im * ratio_im).sqrt();
        let theta = ratio_im.atan2(ratio_re) / 2.0;
        let n_eff_re = r_mag.sqrt() * theta.cos();

        // For metallic media (Re ε_m < 0, large |ε_m|), n_eff_re > n_medium.
        // Guard: if n_eff ≤ n_medium (non-metallic or unphysical input), use fallback.
        let n_eff = if n_eff_re > self.n_medium {
            n_eff_re
        } else {
            self.n_medium * 1.5
        };
        self.wavelength_m / n_eff
    }

    /// Check whether the rod length satisfies the half-wave resonance condition.
    ///
    /// Resonant if |L − λ_eff/2| / λ_eff < 10 %.
    pub fn is_resonant(&self) -> bool {
        let lam_eff = self.effective_wavelength_m();
        (self.length_m - lam_eff / 2.0).abs() / lam_eff < 0.1
    }

    /// Near-field enhancement |E|/|E₀| at the antenna tip.
    ///
    /// Lightning-rod + resonance estimate:
    ///
    ///   |E/E₀|² ≈ (λ_eff / a)² / (3 × Im ε)²
    ///
    /// Physical cap: enhancement saturates around 200 for a single rod.
    pub fn field_enhancement(&self) -> f64 {
        let (_eps_re, eps_im) = self.eps_metal;
        let lam_eff = self.effective_wavelength_m();
        if eps_im.abs() < 1.0e-10 || self.radius_m < 1.0e-12 {
            return 1.0;
        }
        let ratio = lam_eff / self.radius_m;
        let denom = 3.0 * eps_im.abs();
        // |E/E0|² estimate
        let enhancement_sq = ratio * ratio / (denom * denom);
        enhancement_sq.sqrt().clamp(1.0, 200.0)
    }

    /// Absorption cross section of the nanorod (m²).
    ///
    /// Using the quasi-static rod approximation:
    ///
    ///   σ_abs = k Im(α) / ε₀  with α = polarisability tensor diagonal
    ///
    /// Simplified estimate for a prolate spheroid along its long axis:
    ///   σ_abs ≈ V × k × Im(ε_m) / n_medium²  (geometric factor omitted)
    pub fn absorption_cross_section_m2(&self) -> f64 {
        let (_eps_re, eps_im) = self.eps_metal;
        let k = 2.0 * PI * self.n_medium / self.wavelength_m;
        // Rod volume (cylinder)
        let vol = PI * self.radius_m * self.radius_m * self.length_m;
        // Quasi-static absorption: σ_abs ≈ 2 V k Im(ε)/n²
        (2.0 * vol * k * eps_im.abs() / (self.n_medium * self.n_medium)).max(0.0)
    }

    /// Scattering cross section of the nanorod (m²).
    ///
    /// Rayleigh–Mie limit for a small rod:
    ///
    ///   σ_sca ≈ k⁴ |α|² / (6π ε₀²)
    ///
    /// Simplified: σ_sca ≈ (k⁴ V² / (6π)) × |ε_m − n²|²
    pub fn scattering_cross_section_m2(&self) -> f64 {
        let (eps_re, eps_im) = self.eps_metal;
        let k = 2.0 * PI * self.n_medium / self.wavelength_m;
        let vol = PI * self.radius_m * self.radius_m * self.length_m;
        let n2 = self.n_medium * self.n_medium;
        // |ε_m - n²|²
        let delta_eps2 = (eps_re - n2) * (eps_re - n2) + eps_im * eps_im;
        // σ_sca ≈ k⁴ V² |Δε|² / (6π)
        (k.powi(4) * vol * vol * delta_eps2 / (6.0 * PI)).max(0.0)
    }

    /// SERS electromagnetic enhancement factor.
    ///
    /// EF ≈ |E/E₀|⁴  (electromagnetic contribution to SERS)
    pub fn sers_enhancement(&self) -> f64 {
        self.field_enhancement().powi(4)
    }

    /// Radiation efficiency η = Γ_rad / (Γ_rad + Γ_abs) ∈ \[0, 1\].
    ///
    /// Computed from the ratio of scattering to extinction cross sections.
    pub fn radiation_efficiency(&self) -> f64 {
        let sigma_sca = self.scattering_cross_section_m2();
        let sigma_abs = self.absorption_cross_section_m2();
        let sigma_ext = sigma_sca + sigma_abs;
        if sigma_ext < 1.0e-60 {
            return 0.0;
        }
        (sigma_sca / sigma_ext).clamp(0.0, 1.0)
    }
}

// ─── YagiUdaAntenna ──────────────────────────────────────────────────────────

/// Optical Yagi-Uda antenna: feed nanorod + directors + reflector.
///
/// The optical Yagi-Uda (Kühn et al. 2008; Taminiau et al. 2008) is a
/// multi-element nanoantenna that achieves directional emission into a
/// specific half-space.  Director nanorods are slightly shorter than the
/// feed (capacitive coupling) and placed in front; the reflector is slightly
/// longer (inductive) and placed behind.
#[derive(Debug, Clone)]
pub struct YagiUdaAntenna {
    /// Feed nanorod (driven element)
    pub feed: NanorodAntenna,
    /// Number of director elements
    pub n_directors: usize,
    /// Inter-element spacing between directors (m) — typically 0.3 λ_eff
    pub director_spacing_m: f64,
    /// Scale factor for reflector length relative to feed (> 1, typically 1.1)
    pub reflector_length_scale: f64,
    /// Free-space wavelength (m)
    pub wavelength_m: f64,
    /// Refractive index of surrounding medium
    pub n_medium: f64,
}

impl YagiUdaAntenna {
    /// Front-to-back ratio in dB.
    ///
    /// Empirical model based on number of directors:
    ///
    ///   FBR ≈ 5 + 2 × n_directors  (dB, capped at 15 dB for typical OYU)
    pub fn front_to_back_ratio_db(&self) -> f64 {
        (5.0 + 2.0 * self.n_directors as f64).min(15.0)
    }

    /// Directivity estimate: D ≈ 1.5 × (1 + n_directors)
    pub fn directivity(&self) -> f64 {
        1.5 * (1.0 + self.n_directors as f64)
    }

    /// Half-power beamwidth (HPBW) in degrees.
    ///
    /// Approximate via: HPBW ≈ 60° / sqrt(D)
    pub fn hpbw_deg(&self) -> f64 {
        let d = self.directivity();
        if d < 1.0e-10 {
            return 180.0;
        }
        60.0 / d.sqrt()
    }

    /// Array factor magnitude along the director axis at polar angle θ.
    ///
    /// Models the N-element linear array (feed + directors) with uniform
    /// spacing d and progressive phase shift Δφ derived from the director
    /// coupling (approximately k·d for broadside-to-endfire pattern):
    ///
    ///   AF(θ) = |Σ_{n=0}^{N-1} exp(i n (k d cos θ + Δφ))| / N
    ///
    /// where Δφ = −k d (endfire condition toward θ = 0).
    pub fn array_factor_magnitude(&self, theta_rad: f64) -> f64 {
        let n_total = self.n_directors + 1; // feed + directors
        let k = 2.0 * PI * self.n_medium / self.wavelength_m;
        let d = self.director_spacing_m;
        // Endfire progressive phase
        let delta_phi = -k * d;
        let psi = k * d * theta_rad.cos() + delta_phi;

        // Uniform array factor
        if psi.abs() < 1.0e-12 {
            return 1.0;
        }
        let n = n_total as f64;
        let af = (n * psi / 2.0).sin() / (n * (psi / 2.0).sin());
        af.abs()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn gold_nanorod(length_nm: f64, radius_nm: f64, wavelength_nm: f64) -> NanorodAntenna {
        NanorodAntenna {
            length_m: length_nm * 1.0e-9,
            radius_m: radius_nm * 1.0e-9,
            // Gold at ~800 nm: Re ε ≈ -25, Im ε ≈ 1.5
            eps_metal: (-25.0, 1.5),
            wavelength_m: wavelength_nm * 1.0e-9,
            n_medium: 1.0,
        }
    }

    #[test]
    fn hertzian_dipole_directivity() {
        let d = HertzianDipole {
            wavelength_m: 1064e-9,
            dipole_moment_cm: 1e-29,
            n_medium: 1.0,
        };
        assert!(
            (d.directivity() - 1.5).abs() < 1.0e-10,
            "Dipole directivity must be exactly 1.5, got {}",
            d.directivity()
        );
    }

    #[test]
    fn hertzian_dipole_radiated_power_positive() {
        let d = HertzianDipole {
            wavelength_m: 800e-9,
            dipole_moment_cm: 1e-29,
            n_medium: 1.5,
        };
        let omega = d.omega();
        let p = d.radiated_power_w(omega);
        assert!(p > 0.0, "Radiated power must be positive: {p}");
    }

    #[test]
    fn hertzian_dipole_far_field_sin_squared() {
        let d = HertzianDipole {
            wavelength_m: 1e-6,
            dipole_moment_cm: 1e-29,
            n_medium: 1.0,
        };
        // At θ = 90° pattern should be 1.0
        assert!((d.far_field_pattern(PI / 2.0) - 1.0).abs() < 1.0e-12);
        // At θ = 0° pattern should be 0.0
        assert!(d.far_field_pattern(0.0).abs() < 1.0e-12);
    }

    #[test]
    fn hertzian_dipole_near_field_r6_scaling() {
        let d = HertzianDipole {
            wavelength_m: 1e-6,
            dipole_moment_cm: 1e-29,
            n_medium: 1.0,
        };
        let i1 = d.near_field_intensity(10.0e-9);
        let i2 = d.near_field_intensity(20.0e-9);
        // Ratio should be 2^6 = 64
        let ratio = i1 / i2;
        assert!(
            (ratio - 64.0).abs() < 1.0e-6,
            "Near-field r^-6 scaling: ratio={ratio}"
        );
    }

    #[test]
    fn nanorod_effective_wavelength_shorter_than_freespace() {
        let rod = gold_nanorod(100.0, 10.0, 800.0);
        let lam_eff = rod.effective_wavelength_m();
        assert!(
            lam_eff < rod.wavelength_m,
            "λ_eff={:.1}nm must be shorter than λ={:.1}nm",
            lam_eff * 1e9,
            rod.wavelength_m * 1e9
        );
    }

    #[test]
    fn nanorod_cross_sections_positive() {
        let rod = gold_nanorod(120.0, 12.0, 800.0);
        assert!(rod.absorption_cross_section_m2() > 0.0);
        assert!(rod.scattering_cross_section_m2() > 0.0);
    }

    #[test]
    fn nanorod_radiation_efficiency_in_range() {
        let rod = gold_nanorod(120.0, 12.0, 800.0);
        let eta = rod.radiation_efficiency();
        assert!(
            (0.0..=1.0).contains(&eta),
            "Efficiency must be in [0,1]: {eta}"
        );
    }

    #[test]
    fn yagi_uda_directivity_increases_with_directors() {
        let feed = gold_nanorod(100.0, 10.0, 800.0);
        let yagi3 = YagiUdaAntenna {
            feed: feed.clone(),
            n_directors: 3,
            director_spacing_m: 240.0e-9,
            reflector_length_scale: 1.1,
            wavelength_m: 800.0e-9,
            n_medium: 1.0,
        };
        let yagi1 = YagiUdaAntenna {
            feed,
            n_directors: 1,
            director_spacing_m: 240.0e-9,
            reflector_length_scale: 1.1,
            wavelength_m: 800.0e-9,
            n_medium: 1.0,
        };
        assert!(
            yagi3.directivity() > yagi1.directivity(),
            "More directors → higher directivity"
        );
    }
}
