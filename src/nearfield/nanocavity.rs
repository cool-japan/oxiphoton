//! Optical nanocavities and nanoresonators
//!
//! Covers plasmonic gap cavities (bowtie, NPoM, MIM) and dielectric PhC
//! nanocavities.  All structures provide:
//!   - mode volume V_eff in m³
//!   - quality factor Q
//!   - Purcell factor F_P = (3/4π²) · (λ/n)³ · Q / V_eff
//!   - resonance wavelength (analytical / semi-analytical estimate)
//!
//! Reference wavelengths are in metres throughout; nm used only for
//! constructor convenience parameters.

use num_complex::Complex64;
use std::f64::consts::PI;

// ─── Physical constants ───────────────────────────────────────────────────────

const C0: f64 = 2.997_924_58e8; // m/s

// ─── BowtieMaterial ──────────────────────────────────────────────────────────

/// Metallic material for plasmonic structures (Drude parameters)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BowtieMaterial {
    /// Gold: ωp ≈ 1.37e16 rad/s, γ ≈ 1.22e14 rad/s, ε∞ ≈ 9.5
    Gold,
    /// Silver: ωp ≈ 1.37e16 rad/s, γ ≈ 2.73e13 rad/s, ε∞ ≈ 3.7
    Silver,
    /// Aluminum: ωp ≈ 2.24e16 rad/s, γ ≈ 1.22e14 rad/s, ε∞ ≈ 1.0
    Aluminum,
}

impl BowtieMaterial {
    /// Plasma frequency ωp in rad/s
    fn omega_p(&self) -> f64 {
        match self {
            BowtieMaterial::Gold => 1.37e16,
            BowtieMaterial::Silver => 1.37e16,
            BowtieMaterial::Aluminum => 2.24e16,
        }
    }

    /// Drude damping rate γ in rad/s
    fn gamma(&self) -> f64 {
        match self {
            BowtieMaterial::Gold => 1.22e14,
            BowtieMaterial::Silver => 2.73e13,
            BowtieMaterial::Aluminum => 1.22e14,
        }
    }

    /// Background permittivity ε∞
    fn eps_inf(&self) -> f64 {
        match self {
            BowtieMaterial::Gold => 9.5,
            BowtieMaterial::Silver => 3.7,
            BowtieMaterial::Aluminum => 1.0,
        }
    }

    /// Drude permittivity ε(ω) = ε∞ − ωp² / (ω² + iγω)
    pub fn permittivity(&self, omega: f64) -> Complex64 {
        let wp = self.omega_p();
        let gam = self.gamma();
        let ei = self.eps_inf();
        let denom = Complex64::new(omega * omega, gam * omega);
        Complex64::new(ei, 0.0) - Complex64::new(wp * wp, 0.0) / denom
    }

    /// Intrinsic quality factor from Drude model: Q_i ≈ ω / γ
    fn intrinsic_q(&self, omega: f64) -> f64 {
        omega / self.gamma()
    }
}

// ─── Helper: Purcell factor ───────────────────────────────────────────────────

/// Compute Purcell factor from quality factor, mode volume, and wavelength.
///
/// F_P = (3 / 4π²) · (λ/n)³ / V_eff · Q
fn purcell_from_q_v(q: f64, mode_volume_m3: f64, wavelength_m: f64, n: f64) -> f64 {
    if mode_volume_m3 < 1.0e-50 {
        return 0.0;
    }
    let lambda_n = wavelength_m / n;
    let prefactor = 3.0 / (4.0 * PI * PI);
    prefactor * (lambda_n * lambda_n * lambda_n) * q / mode_volume_m3
}

// ─── BowtiAntenna ────────────────────────────────────────────────────────────

/// Bowtie nanoantenna: two coupled triangular metal nanoparticles.
///
/// The bowtie geometry concentrates optical fields in the nanogap between
/// two triangular arms.  Due to the lightning-rod effect and coupled-dipole
/// resonance, field enhancements of 100–1000× are achievable.
///
/// Semi-analytical model based on:
/// - Effective half-wave antenna resonance with plasmonic wavelength contraction
/// - Gap mode volume V ≈ gap³
/// - Q limited by radiation resistance + Ohmic losses
#[derive(Debug, Clone)]
pub struct BowtiAntenna {
    /// Arm length of each triangular arm in metres
    pub arm_length: f64,
    /// Gap between the two triangles in nm
    pub gap_nm: f64,
    /// Metal material
    pub material: BowtieMaterial,
    /// Design wavelength in metres
    pub wavelength: f64,
    /// Substrate refractive index
    pub substrate_index: f64,
}

impl BowtiAntenna {
    /// Construct a gold bowtie antenna with specified arm length and gap.
    ///
    /// # Arguments
    /// * `arm_length_nm` - single arm length in nm
    /// * `gap_nm`        - gap between tips in nm
    /// * `wavelength`    - design wavelength in metres
    pub fn new_gold(arm_length_nm: f64, gap_nm: f64, wavelength: f64) -> Self {
        Self {
            arm_length: arm_length_nm * 1.0e-9,
            gap_nm,
            material: BowtieMaterial::Gold,
            wavelength,
            substrate_index: 1.5, // glass substrate
        }
    }

    /// Near-field enhancement factor |E_gap| / |E_inc| at the gap centre.
    ///
    /// Estimated from the analytic coupled-dipole + lightning-rod model:
    ///   EF ≈ Re[ε_m / ε_d] · (L / gap)^(1/2)
    /// capped at physically observed maximum for plasmonic systems.
    pub fn field_enhancement(&self) -> f64 {
        let omega = 2.0 * PI * C0 / self.wavelength;
        let eps_m = self.material.permittivity(omega);
        let eps_d = self.substrate_index * self.substrate_index;

        // Re-part of Clausius-Mossotti local field factor
        let lf = (eps_m.re.abs() / eps_d).sqrt();
        // Geometric lightning-rod: sqrt(L/gap)
        let gap_m = self.gap_nm * 1.0e-9;
        let geom = (self.arm_length / gap_m).sqrt();
        // Empirical cap for plasmonic bowtie (order of magnitude)
        (lf * geom).min(2000.0)
    }

    /// Mode volume: V ≈ gap³ for a gap-dominated bowtie cavity (m³).
    pub fn mode_volume(&self) -> f64 {
        let gap_m = self.gap_nm * 1.0e-9;
        gap_m * gap_m * gap_m
    }

    /// Quality factor limited by both radiation and Ohmic losses.
    ///
    /// Q_total⁻¹ = Q_rad⁻¹ + Q_ohm⁻¹
    ///
    /// Q_rad ≈ 5–20 for plasmonic antennas (radiation dominated at small gaps)
    /// Q_ohm ≈ ω / (2γ)  from Drude model
    pub fn q_factor(&self) -> f64 {
        let omega = 2.0 * PI * C0 / self.wavelength;
        let q_ohm = self.material.intrinsic_q(omega) / 2.0;
        let q_rad = 10.0_f64; // typical radiation Q for bowtie (empirical)
        1.0 / (1.0 / q_rad + 1.0 / q_ohm)
    }

    /// Purcell factor at the bowtie gap centre.
    pub fn purcell_factor(&self) -> f64 {
        let n = self.substrate_index;
        purcell_from_q_v(self.q_factor(), self.mode_volume(), self.wavelength, n)
    }

    /// SERS electromagnetic enhancement factor: g_SERS = |E/E0|⁴
    pub fn sers_enhancement(&self) -> f64 {
        let fe = self.field_enhancement();
        fe * fe * fe * fe
    }

    /// Resonance wavelength using half-wave antenna model with plasmonic
    /// wavelength contraction and effective medium correction.
    ///
    /// λ_res ≈ 2 n_eff · (2L + gap)  with contraction factor η_c
    pub fn resonance_wavelength(&self) -> f64 {
        // Effective index: average of metal surface and substrate
        let omega_est = 2.0 * PI * C0 / self.wavelength;
        let eps_m = self.material.permittivity(omega_est);
        let eps_d = self.substrate_index * self.substrate_index;
        // SPP effective index along the arm (MIM-like)
        let n_eff = ((eps_m.re.abs() * eps_d) / (eps_m.re.abs() + eps_d))
            .sqrt()
            .max(1.0);
        let total_length = 2.0 * self.arm_length + self.gap_nm * 1.0e-9;
        2.0 * n_eff * total_length
    }
}

// ─── NanoparticleOnMirror ────────────────────────────────────────────────────

/// Nanoparticle-on-mirror (NPoM) cavity.
///
/// A metallic nanoparticle separated from a flat metallic mirror by a thin
/// dielectric spacer forms an ultracompact plasmonic cavity.  The gap mode
/// is strongly confined (V_eff ~ gap³).
#[derive(Debug, Clone)]
pub struct NanoparticleOnMirror {
    /// Nanoparticle radius in nm
    pub particle_radius_nm: f64,
    /// Dielectric spacer thickness in nm
    pub gap_nm: f64,
    /// Particle metal material
    pub particle_material: BowtieMaterial,
    /// Mirror metal material
    pub mirror_material: BowtieMaterial,
    /// Refractive index of spacer dielectric
    pub spacer_index: f64,
}

impl NanoparticleOnMirror {
    /// Construct a gold NPoM with specified particle radius and gap.
    pub fn new_gold(radius_nm: f64, gap_nm: f64) -> Self {
        Self {
            particle_radius_nm: radius_nm,
            gap_nm,
            particle_material: BowtieMaterial::Gold,
            mirror_material: BowtieMaterial::Gold,
            spacer_index: 1.46, // SiO2 spacer
        }
    }

    /// Mode volume of the gap plasmon mode (m³).
    ///
    /// The near-field gap mode is confined to a disk of effective radius
    /// √(R·gap) and depth ≈ gap (Baumberg circular disk model):
    ///
    ///   V_eff ≈ π R gap²
    ///
    /// This gives deeply sub-diffraction volumes for nm gaps.
    pub fn mode_volume(&self) -> f64 {
        let gap = self.gap_nm * 1.0e-9;
        let r = self.particle_radius_nm * 1.0e-9;
        PI * r * gap * gap
    }

    /// Quality factor of the fundamental gap mode.
    ///
    /// Q is limited by radiation (Q_rad ~ 10–30) and Ohmic losses.
    pub fn q_factor(&self) -> f64 {
        let lambda = self.resonance_wavelength();
        let omega = 2.0 * PI * C0 / lambda;
        let q_ohm = self.particle_material.intrinsic_q(omega) / 2.0;
        let q_rad = 15.0_f64;
        1.0 / (1.0 / q_rad + 1.0 / q_ohm)
    }

    /// Gap field enhancement |E_gap| / |E_inc| (NPoM typically 200–2000×).
    ///
    /// Enhancement ∝ (R / gap)^(1/3) × Q  (Baumberg scaling)
    pub fn field_enhancement(&self) -> f64 {
        let r = self.particle_radius_nm;
        let gap = self.gap_nm;
        let scale = (r / gap).powf(1.0 / 3.0);
        let q = self.q_factor();
        (scale * q).min(3000.0)
    }

    /// Purcell factor at the NPoM gap centre.
    pub fn purcell_factor(&self) -> f64 {
        let n = self.spacer_index;
        let lambda = self.resonance_wavelength();
        purcell_from_q_v(self.q_factor(), self.mode_volume(), lambda, n)
    }

    /// Approximate resonance wavelength of the fundamental gap mode.
    ///
    /// Uses the coupled-sphere-mirror image model:
    ///   ω_res ≈ ω_lspr · (1 − gap / R · correction)
    pub fn resonance_wavelength(&self) -> f64 {
        // Gold LSPR in the spacer medium: quasi-static Fröhlich condition
        let eps_d = self.spacer_index * self.spacer_index;
        let omega_p = self.particle_material.omega_p();
        let eps_inf = self.particle_material.eps_inf();
        let omega_lspr = (omega_p * omega_p / (eps_inf + 2.0 * eps_d)).sqrt();
        // Red-shift due to proximity to mirror (image coupling)
        let gap = self.gap_nm;
        let r = self.particle_radius_nm;
        let coupling_shift = 1.0 - 0.2 * (gap / r).sqrt();
        let omega_res = omega_lspr * coupling_shift.max(0.5);
        2.0 * PI * C0 / omega_res
    }

    /// Multiple facet-mode resonances of the NPoM (return wavelengths in m).
    ///
    /// NPoM supports a series of gap modes with azimuthal quantum numbers
    /// l = 1, 2, 3, ...  Each mode is blue-shifted relative to l=1.
    pub fn facet_modes(&self) -> Vec<f64> {
        let lambda0 = self.resonance_wavelength();
        // Higher modes blue-shift approximately as l·Δω where Δω ≈ 5% per mode
        (1..=5_usize)
            .map(|l| lambda0 / (1.0 + 0.05 * (l as f64 - 1.0)))
            .collect()
    }
}

// ─── PhCNanocavity ───────────────────────────────────────────────────────────

/// Photonic crystal (PhC) slab nanocavity cavity type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhCCavityType {
    /// L3 cavity: 3 missing holes along Γ-K.  Q ~ 10⁴–10⁶
    L3,
    /// H1 cavity: 1 missing hole (point defect).  Q ~ 10³–10⁴
    H1,
    /// H3 cavity: 3 missing holes forming a triangle.  Q ~ 10⁴–10⁵
    H3,
    /// Heterostructure: modulated lattice constant.  Q > 10⁶ theoretically
    Heterostructure,
}

/// Photonic crystal nanocavity in a triangular-lattice slab.
///
/// The fundamental resonance wavelength is approximately at the band-edge of
/// the photonic crystal, and the mode volume approaches the diffraction limit
/// V_eff ~ (λ/n)³.
#[derive(Debug, Clone)]
pub struct PhCNanocavity {
    /// Triangular lattice constant in nm
    pub lattice_constant_nm: f64,
    /// Hole radius in nm
    pub hole_radius_nm: f64,
    /// Slab thickness in nm
    pub slab_thickness_nm: f64,
    /// Effective slab refractive index
    pub n_slab: f64,
    /// Cavity type
    pub cavity_type: PhCCavityType,
}

impl PhCNanocavity {
    /// L3 cavity in GaAs (n ≈ 3.4).
    pub fn new_l3_gaas(a: f64, r: f64) -> Self {
        Self {
            lattice_constant_nm: a,
            hole_radius_nm: r,
            slab_thickness_nm: a * 0.6,
            n_slab: 3.4,
            cavity_type: PhCCavityType::L3,
        }
    }

    /// L3 cavity in silicon nitride (n ≈ 2.0).
    pub fn new_l3_sin(a: f64, r: f64) -> Self {
        Self {
            lattice_constant_nm: a,
            hole_radius_nm: r,
            slab_thickness_nm: a * 0.5,
            n_slab: 2.0,
            cavity_type: PhCCavityType::L3,
        }
    }

    /// Theoretical quality factor for each cavity type.
    ///
    /// Based on published simulations:
    /// - L3 optimised: Q ~ 10⁶
    /// - H1: Q ~ 10⁴
    /// - H3: Q ~ 10⁵
    /// - Heterostructure: Q ~ 10⁸
    pub fn quality_factor(&self) -> f64 {
        match self.cavity_type {
            PhCCavityType::L3 => 1.0e6,
            PhCCavityType::H1 => 1.0e4,
            PhCCavityType::H3 => 1.0e5,
            PhCCavityType::Heterostructure => 1.0e8,
        }
    }

    /// Mode volume in units of (λ/n)³.
    ///
    /// Typical values from FDTD simulations:
    /// L3 ~ 0.7, H1 ~ 0.3, H3 ~ 0.4, Heterostructure ~ 0.8
    pub fn mode_volume_cubic_lambda(&self) -> f64 {
        match self.cavity_type {
            PhCCavityType::L3 => 0.7,
            PhCCavityType::H1 => 0.3,
            PhCCavityType::H3 => 0.4,
            PhCCavityType::Heterostructure => 0.8,
        }
    }

    /// Mode volume in m³.
    pub fn mode_volume_m3(&self) -> f64 {
        let lambda_n = self.resonance_wavelength() / self.n_slab;
        let v_norm = self.mode_volume_cubic_lambda();
        v_norm * lambda_n * lambda_n * lambda_n
    }

    /// Purcell factor for an emitter optimally positioned and polarised.
    pub fn purcell_factor(&self) -> f64 {
        let lambda = self.resonance_wavelength();
        let n = self.n_slab;
        purcell_from_q_v(self.quality_factor(), self.mode_volume_m3(), lambda, n)
    }

    /// Approximate resonance wavelength in metres.
    ///
    /// For a triangular PhC slab the TE bandgap spans roughly:
    ///   0.25 < a/λ < 0.35  (for r/a ~ 0.3, n ~ 3.4)
    /// The L3 cavity resonance sits near the band-edge: a/λ ≈ 0.26
    pub fn resonance_wavelength(&self) -> f64 {
        let a_m = self.lattice_constant_nm * 1.0e-9;
        let r_over_a = self.hole_radius_nm / self.lattice_constant_nm;
        // Empirical dispersion correction for r/a ratio
        let a_over_lambda = 0.26 + 0.15 * (r_over_a - 0.30).clamp(-0.1, 0.1);
        a_m / a_over_lambda
    }

    /// External Q factor when coupled to a waveguide with a given number of
    /// barrier holes (smaller gap_holes → stronger coupling → lower Q_ext).
    ///
    /// Empirical: Q_ext ≈ Q_total · exp(n_holes · 0.8)
    pub fn coupling_to_waveguide(&self, waveguide_gap_holes: usize) -> f64 {
        let q_total = self.quality_factor();
        // Coupling decays exponentially with number of barrier holes
        let barrier = waveguide_gap_holes as f64;
        q_total * (barrier * 0.8_f64).exp()
    }

    /// Zero-point electric field in V/m for a given wavelength.
    ///
    /// E_zpf = sqrt(ħω / (2 ε₀ n² V_eff))
    pub fn zero_point_field_v_per_m(&self, wavelength: f64) -> f64 {
        const HBAR: f64 = 1.054_571_817e-34;
        const EPS0: f64 = 8.854_187_817e-12;
        let omega = 2.0 * PI * C0 / wavelength;
        let n = self.n_slab;
        let v = self.mode_volume_m3();
        if v < 1.0e-50 {
            return 0.0;
        }
        (HBAR * omega / (2.0 * EPS0 * n * n * v)).sqrt()
    }
}

// ─── MimNanocavity ───────────────────────────────────────────────────────────

/// Metal-insulator-metal (MIM) gap plasmon nanocavity.
///
/// Two metallic slabs separated by a thin dielectric gap support gap plasmon
/// polaritons (GPPs) with extremely short wavelengths (λ_gpp ≪ λ).  The
/// cavity is formed by a finite-length MIM structure with open ends.
#[derive(Debug, Clone)]
pub struct MimNanocavity {
    /// Cavity length in nm
    pub length_nm: f64,
    /// Cavity width in nm
    pub width_nm: f64,
    /// Dielectric gap thickness in nm
    pub gap_nm: f64,
    /// Metal material
    pub metal: BowtieMaterial,
    /// Refractive index of dielectric filling
    pub dielectric_index: f64,
}

impl MimNanocavity {
    /// Construct a gold/air MIM cavity.
    pub fn new_gold_air(length_nm: f64, width_nm: f64, gap_nm: f64) -> Self {
        Self {
            length_nm,
            width_nm,
            gap_nm,
            metal: BowtieMaterial::Gold,
            dielectric_index: 1.0,
        }
    }

    /// Effective complex refractive index of the fundamental MIM gap mode.
    ///
    /// In the thin-gap quasi-static limit (k₀ d ≪ 1) the TM₀ gap plasmon
    /// dispersion reduces to:
    ///
    ///   n_eff ≈ sqrt(−ε_m)  · [1 + ε_d/(2 ε_m) · k₀ d · correction]
    ///
    /// Leading-order result (dominant when |ε_m| ≫ ε_d):
    ///
    ///   n_eff ≈ sqrt(−ε_m)
    ///
    /// This gives Re(n_eff) ~ 5 and Im(n_eff) ~ 0.18 for gold at 800 nm.
    pub fn effective_index(&self, wavelength: f64) -> Complex64 {
        let omega = 2.0 * PI * C0 / wavelength;
        let eps_m = self.metal.permittivity(omega);
        let eps_d = Complex64::new(self.dielectric_index * self.dielectric_index, 0.0);
        let d = self.gap_nm * 1.0e-9;
        let k0 = omega / C0;

        // Leading-order thin-gap GPP effective index
        let neff_sq_0 = -eps_m;
        let n_eff_0 = neff_sq_0.sqrt();
        let n0 = if n_eff_0.re < 0.0 { -n_eff_0 } else { n_eff_0 };

        // First-order correction in k0*d: Δn ≈ eps_d/(2*n0) * k0*d
        let k0d = k0 * d;
        let delta_n = eps_d * Complex64::new(k0d, 0.0) / (Complex64::new(2.0, 0.0) * n0);

        let n_eff = n0 + delta_n;
        if n_eff.re < 0.0 {
            -n_eff
        } else {
            n_eff
        }
    }

    /// Resonance wavelength from the Fabry-Pérot condition: 2 n_eff L = m λ (m=1).
    ///
    /// The end reflection phase shift is approximately π for metallic termination.
    pub fn resonance_wavelength(&self) -> f64 {
        // Iterative self-consistent solution since n_eff depends on λ
        let l = self.length_nm * 1.0e-9;
        // Initial guess: free-space half-wave
        let mut lambda = 2.0 * l * self.dielectric_index;
        for _ in 0..20 {
            let n_eff = self.effective_index(lambda);
            lambda = 2.0 * n_eff.re * l;
            if lambda < 200.0e-9 {
                lambda = 200.0e-9;
                break;
            }
        }
        lambda
    }

    /// Mode volume of the MIM cavity (m³).
    ///
    /// V_eff ≈ d · w · L_eff  where L_eff accounts for field penetration.
    pub fn mode_volume(&self) -> f64 {
        let d = self.gap_nm * 1.0e-9;
        let w = self.width_nm * 1.0e-9;
        let l = self.length_nm * 1.0e-9;
        // Effective length includes penetration depth into metal ends (~10 nm each)
        let l_eff = l + 20.0e-9;
        d * w * l_eff
    }

    /// Quality factor of the MIM cavity.
    ///
    /// Q is limited by propagation loss of the gap plasmon:
    ///   Q ≈ π n_eff_re / (n_eff_im) · (L / λ_eff)
    pub fn q_factor(&self) -> f64 {
        let lambda = self.resonance_wavelength();
        let n_eff = self.effective_index(lambda);
        if n_eff.im.abs() < 1.0e-10 || n_eff.re < 1.0e-10 {
            return 10.0; // fallback for degenerate cases
        }
        // Q = π Re(n_eff) / Im(n_eff)  (propagation Q for one round trip)
        PI * n_eff.re / n_eff.im.abs()
    }

    /// Purcell factor for an emitter at the centre of the MIM gap.
    pub fn purcell_factor(&self) -> f64 {
        let n = self.dielectric_index;
        let lambda = self.resonance_wavelength();
        purcell_from_q_v(self.q_factor(), self.mode_volume(), lambda, n)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── BowtiMaterial ─────────────────────────────────────────────────────────

    #[test]
    fn test_bowtie_material_gold_permittivity_negative_re() {
        // Gold at 532 nm: Re(ε) should be strongly negative
        let omega = 2.0 * PI * C0 / 532.0e-9;
        let eps = BowtieMaterial::Gold.permittivity(omega);
        assert!(
            eps.re < 0.0,
            "Gold Re(ε) at 532 nm should be negative: {}",
            eps.re
        );
    }

    #[test]
    fn test_bowtie_material_silver_lower_loss_than_gold() {
        let omega = 2.0 * PI * C0 / 632.0e-9;
        let q_ag = BowtieMaterial::Silver.intrinsic_q(omega);
        let q_au = BowtieMaterial::Gold.intrinsic_q(omega);
        assert!(
            q_ag > q_au,
            "Silver should have higher intrinsic Q (lower loss) than gold: {q_ag} vs {q_au}"
        );
    }

    // ── BowtiAntenna ──────────────────────────────────────────────────────────

    #[test]
    fn test_bowtie_field_enhancement_positive() {
        let bowtie = BowtiAntenna::new_gold(75.0, 5.0, 800.0e-9);
        let fe = bowtie.field_enhancement();
        assert!(fe > 0.0, "Field enhancement must be positive: {fe}");
    }

    #[test]
    fn test_bowtie_sers_enhancement_is_fe4() {
        let bowtie = BowtiAntenna::new_gold(75.0, 5.0, 800.0e-9);
        let fe = bowtie.field_enhancement();
        let sers = bowtie.sers_enhancement();
        assert_abs_diff_eq!(sers, fe * fe * fe * fe, epsilon = 1.0);
    }

    #[test]
    fn test_bowtie_mode_volume_scales_with_gap() {
        let bowtie1 = BowtiAntenna::new_gold(75.0, 5.0, 800.0e-9);
        let bowtie2 = BowtiAntenna::new_gold(75.0, 10.0, 800.0e-9);
        let v1 = bowtie1.mode_volume();
        let v2 = bowtie2.mode_volume();
        // V ∝ gap³ → ratio = 8
        assert_abs_diff_eq!(v2 / v1, 8.0, epsilon = 1.0e-10);
    }

    #[test]
    fn test_bowtie_purcell_factor_positive() {
        let bowtie = BowtiAntenna::new_gold(75.0, 5.0, 800.0e-9);
        let fp = bowtie.purcell_factor();
        assert!(fp > 0.0, "Purcell factor must be positive: {fp}");
    }

    #[test]
    fn test_bowtie_resonance_wavelength_reasonable() {
        let bowtie = BowtiAntenna::new_gold(75.0, 5.0, 800.0e-9);
        let lam = bowtie.resonance_wavelength();
        assert!(
            lam > 300.0e-9 && lam < 3.0e-6,
            "Resonance wavelength should be 300 nm–3 µm: {} nm",
            lam * 1.0e9
        );
    }

    // ── NanoparticleOnMirror ──────────────────────────────────────────────────

    #[test]
    fn test_npom_mode_volume_sub_lambda3() {
        let npom = NanoparticleOnMirror::new_gold(40.0, 1.0);
        let v = npom.mode_volume();
        let lambda = npom.resonance_wavelength();
        let lambda3 = lambda * lambda * lambda;
        assert!(
            v < lambda3,
            "NPoM mode volume should be sub-diffraction: V={v:.3e} λ³={lambda3:.3e}"
        );
    }

    #[test]
    fn test_npom_facet_modes_monotone_decreasing() {
        let npom = NanoparticleOnMirror::new_gold(40.0, 1.0);
        let modes = npom.facet_modes();
        assert_eq!(modes.len(), 5);
        for i in 0..modes.len() - 1 {
            assert!(
                modes[i] > modes[i + 1],
                "Facet modes should be monotone decreasing in wavelength"
            );
        }
    }

    #[test]
    fn test_npom_field_enhancement_increases_with_smaller_gap() {
        let npom1 = NanoparticleOnMirror::new_gold(40.0, 2.0);
        let npom2 = NanoparticleOnMirror::new_gold(40.0, 1.0);
        let fe1 = npom1.field_enhancement();
        let fe2 = npom2.field_enhancement();
        assert!(
            fe2 > fe1,
            "Smaller gap should give higher enhancement: {fe2} vs {fe1}"
        );
    }

    // ── PhCNanocavity ─────────────────────────────────────────────────────────

    #[test]
    fn test_phc_l3_quality_factor() {
        let cav = PhCNanocavity::new_l3_gaas(260.0, 78.0);
        let q = cav.quality_factor();
        assert_abs_diff_eq!(q, 1.0e6, epsilon = 1.0);
    }

    #[test]
    fn test_phc_resonance_wavelength_in_nir() {
        // GaAs PhC with a=260 nm: should resonate around 1 µm
        let cav = PhCNanocavity::new_l3_gaas(260.0, 78.0);
        let lam = cav.resonance_wavelength();
        assert!(
            lam > 600.0e-9 && lam < 2.0e-6,
            "PhC resonance should be in NIR: {} nm",
            lam * 1.0e9
        );
    }

    #[test]
    fn test_phc_purcell_factor_large() {
        let cav = PhCNanocavity::new_l3_gaas(260.0, 78.0);
        let fp = cav.purcell_factor();
        // L3 with Q~1e6 and V~0.7 (λ/n)³ → very large Purcell
        assert!(fp > 1000.0, "L3 Purcell factor should be > 1000: {fp}");
    }

    #[test]
    fn test_phc_zero_point_field_positive() {
        let cav = PhCNanocavity::new_l3_gaas(260.0, 78.0);
        let lambda = cav.resonance_wavelength();
        let ezpf = cav.zero_point_field_v_per_m(lambda);
        assert!(ezpf > 0.0, "Zero-point field must be positive: {ezpf}");
    }

    #[test]
    fn test_phc_heterostructure_q_exceeds_l3() {
        let l3 = PhCNanocavity::new_l3_gaas(260.0, 78.0);
        let hs = PhCNanocavity {
            lattice_constant_nm: 260.0,
            hole_radius_nm: 78.0,
            slab_thickness_nm: 156.0,
            n_slab: 3.4,
            cavity_type: PhCCavityType::Heterostructure,
        };
        assert!(hs.quality_factor() > l3.quality_factor());
    }

    // ── MimNanocavity ─────────────────────────────────────────────────────────

    #[test]
    fn test_mim_effective_index_large_real_part() {
        // MIM gap plasmon should have n_eff >> 1 (strong confinement)
        let mim = MimNanocavity::new_gold_air(200.0, 50.0, 5.0);
        let lambda = 800.0e-9;
        let n_eff = mim.effective_index(lambda);
        assert!(
            n_eff.re > 1.0,
            "MIM n_eff should exceed free-space: {}",
            n_eff.re
        );
    }

    #[test]
    fn test_mim_mode_volume_positive() {
        let mim = MimNanocavity::new_gold_air(200.0, 50.0, 5.0);
        let v = mim.mode_volume();
        assert!(v > 0.0, "Mode volume must be positive: {v}");
    }

    #[test]
    fn test_mim_purcell_factor_positive() {
        let mim = MimNanocavity::new_gold_air(200.0, 50.0, 5.0);
        let fp = mim.purcell_factor();
        assert!(fp > 0.0, "MIM Purcell factor must be positive: {fp}");
    }
}
