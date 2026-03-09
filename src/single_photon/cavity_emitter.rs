//! Cavity-enhanced single-photon sources.
//!
//! Models Purcell-enhanced single-photon emission, photon extraction efficiency
//! from high-index semiconductors, and benchmarking figures of merit.
//!
//! # Physical background
//! Placing a quantum emitter in a cavity with quality factor Q and mode volume V
//! enhances its spontaneous emission rate by the Purcell factor:
//!   F_P = (3/4π²)(λ/n)³(Q/V)
//! This shortens the radiative lifetime, suppresses dephasing relative to
//! radiative decay, and funnels photons preferentially into a single mode.
//!
//! # References
//! - Purcell, Phys. Rev. 69, 681 (1946) — original Purcell effect
//! - Senellart, Solomon & White, Nat. Nano 12, 1026 (2017) — practical devices
//! - Tomm et al., Nat. Nano 16, 399 (2021) — bright single-photon source
//! - Uppu et al., Sci. Adv. 6, eabc8268 (2020) — photonic-crystal single-photon source

use std::f64::consts::PI;

// ─── Physical constants ────────────────────────────────────────────────────────

// ─── CavityEnhancedSource ─────────────────────────────────────────────────────

/// Single-photon source with Purcell-enhanced spontaneous emission.
///
/// Captures the key device parameters determining brightness,
/// indistinguishability, and photon purity.
#[derive(Debug, Clone)]
pub struct CavityEnhancedSource {
    /// Free-space radiative lifetime of the bare emitter (ns)
    pub emitter_lifetime_ns: f64,
    /// Purcell factor F_P (cavity-enhanced emission rate / free-space rate)
    pub purcell_factor: f64,
    /// Free-space quantum efficiency η₀ (fraction of excitations that emit photons)
    pub quantum_efficiency: f64,
    /// β factor: fraction of emission into the cavity mode without Purcell
    pub coupling_efficiency: f64,
    /// Cavity-to-fibre/objective collection efficiency η_coll
    pub cavity_collection_efficiency: f64,
}

impl CavityEnhancedSource {
    /// Create a cavity-enhanced SPE.
    ///
    /// - `lifetime_ns`:  free-space radiative lifetime (ns)
    /// - `purcell`:      Purcell factor F_P
    /// - `eta`:          bare quantum efficiency η₀ (0–1)
    /// - `beta`:         β factor into cavity mode (0–1)
    /// - `coll`:         collection efficiency from cavity (0–1)
    pub fn new(lifetime_ns: f64, purcell: f64, eta: f64, beta: f64, coll: f64) -> Self {
        Self {
            emitter_lifetime_ns: lifetime_ns.max(0.001),
            purcell_factor: purcell.max(1.0),
            quantum_efficiency: eta.clamp(0.0, 1.0),
            coupling_efficiency: beta.clamp(0.0, 1.0),
            cavity_collection_efficiency: coll.clamp(0.0, 1.0),
        }
    }

    /// Enhanced spontaneous emission rate Γ_cav = F_P · Γ₀ (GHz).
    ///
    /// Γ₀ = 1/τ_rad; Γ_cav = F_P/τ_rad.
    pub fn enhanced_rate_ghz(&self) -> f64 {
        let gamma_0_ghz = 1.0 / (self.emitter_lifetime_ns * 1e-9) * 1e-9; // Hz → GHz
        gamma_0_ghz * self.purcell_factor
    }

    /// Cavity-enhanced radiative lifetime τ_cav = 1/Γ_cav (ns).
    pub fn cavity_lifetime_ns(&self) -> f64 {
        self.emitter_lifetime_ns / self.purcell_factor
    }

    /// Effective β factor (emission fraction into cavity mode) with Purcell.
    ///
    /// β_eff = F_P · η / (F_P · η + (1 − η))
    ///
    /// As F_P → ∞, β_eff → 1 (all emission into cavity mode).
    pub fn beta_factor_enhanced(&self) -> f64 {
        let fp = self.purcell_factor;
        let eta = self.quantum_efficiency;
        let numerator = fp * eta;
        let denominator = fp * eta + (1.0 - eta).max(0.0);
        if denominator > 0.0 {
            numerator / denominator
        } else {
            1.0
        }
    }

    /// Brightness (photons into first lens per excitation pulse).
    ///
    /// B = β_eff · η_coll
    pub fn brightness_per_pulse(&self) -> f64 {
        self.beta_factor_enhanced() * self.cavity_collection_efficiency
    }

    /// Indistinguishability with Purcell enhancement.
    ///
    /// M = Γ_rad_cav / (Γ_rad_cav + 2 γ_pure)
    ///   = (F_P / τ₀) / (F_P/τ₀ + 2 γ_pure)
    ///
    /// `pure_dephasing_ghz`: pure dephasing rate γ_pure (GHz)
    pub fn indistinguishability(&self, pure_dephasing_ghz: f64) -> f64 {
        let gamma_cav_ghz = self.enhanced_rate_ghz();
        let denom = gamma_cav_ghz + 2.0 * pure_dephasing_ghz.max(0.0);
        if denom > 0.0 {
            gamma_cav_ghz / denom
        } else {
            1.0
        }
    }

    /// Second-order coherence g²(0).
    ///
    /// For a true single emitter in a cavity g²(0) is suppressed by the
    /// single-emitter blockade.  Residual multi-photon probability comes from
    /// re-excitation and background fluorescence.
    ///
    /// `multiphoton_prob`: probability of a multi-photon pulse (0–1)
    pub fn g2_zero(&self, multiphoton_prob: f64) -> f64 {
        (2.0 * multiphoton_prob.clamp(0.0, 0.5)).min(1.0)
    }

    /// Heralded single-photon efficiency.
    ///
    /// η_her = β_eff · η_coll / (β_eff · η_coll + background)
    /// Simplified: η_her ≈ β_eff · η_coll² (loss in second optical element)
    pub fn heralded_efficiency(&self) -> f64 {
        let b = self.beta_factor_enhanced();
        let eta_c = self.cavity_collection_efficiency;
        // Heralded path: must detect in both channels
        (b * eta_c * eta_c).clamp(0.0, 1.0)
    }
}

// ─── ExtractionEfficiency ─────────────────────────────────────────────────────

/// Photon extraction efficiency from a high-index semiconductor.
///
/// Without special structures, the critical angle for total internal reflection
/// severely limits the fraction of photons escaping the semiconductor.
#[derive(Debug, Clone)]
pub struct ExtractionEfficiency {
    /// Refractive index of the semiconductor (e.g., 3.5 for GaAs)
    pub n_semiconductor: f64,
    /// Numerical aperture of the collection objective
    pub collection_na: f64,
}

impl ExtractionEfficiency {
    /// Create an extraction efficiency model.
    ///
    /// `n`: semiconductor refractive index, `na`: objective NA.
    pub fn new(n: f64, na: f64) -> Self {
        Self {
            n_semiconductor: n.max(1.0),
            collection_na: na.clamp(0.0, 1.0),
        }
    }

    /// Critical angle for total internal reflection (degrees).
    ///
    /// θ_c = arcsin(n_air / n_semi) = arcsin(1/n) for n_air = 1.
    pub fn critical_angle_deg(&self) -> f64 {
        let theta_c = (1.0 / self.n_semiconductor).asin();
        theta_c.to_degrees()
    }

    /// Fraction of emitted photons within the TIR escape cone.
    ///
    /// η_TIR = (1 − cos θ_c) / 2  (fraction of 4π solid angle below TIR cone,
    /// one side only; total 2 × this for top + bottom).
    pub fn tir_limited_fraction(&self) -> f64 {
        let theta_c = (1.0_f64 / self.n_semiconductor).asin();
        (1.0 - theta_c.cos()) / 2.0
    }

    /// Fraction of emitted photons collected by the objective lens.
    ///
    /// Within the escape cone the objective collects a solid-angle fraction:
    ///   η_NA = (1 − cos(arcsin(NA/n))) inside the TIR cone.
    pub fn fraction_into_na(&self) -> f64 {
        let sin_theta_na = self.collection_na / self.n_semiconductor;
        if sin_theta_na >= 1.0 {
            // NA larger than escape cone → collect all photons in escape cone
            return self.tir_limited_fraction();
        }
        let theta_na = sin_theta_na.asin();
        (1.0 - theta_na.cos()) / 2.0
    }

    /// Extraction with a single-layer anti-reflection coating.
    ///
    /// An ideal AR coating eliminates Fresnel reflection (R ≈ 0.3 for GaAs/air)
    /// and roughly doubles the fraction of photons extracted per facet.
    /// Practical AR coatings achieve R ≈ 0.5 % → multiply by ~1/(1-R_bare).
    pub fn with_ar_coating(&self) -> f64 {
        let r_bare = ((self.n_semiconductor - 1.0) / (self.n_semiconductor + 1.0)).powi(2);
        let eta_bare = self.fraction_into_na();
        // AR coating reduces reflective loss: η_ar = η_bare / (1 - r_bare)
        (eta_bare / (1.0 - r_bare + 1e-30)).min(1.0)
    }

    /// Extraction with a photonic crystal structure etched above the QD layer.
    ///
    /// PhC extraction antennas (e.g., circular Bragg gratings) can redirect
    /// 40–60 % of photons into a single-mode fibre.  Return a typical
    /// enhancement factor (fraction of 4π solid angle redirected).
    pub fn with_photonic_crystal(&self) -> f64 {
        // Typical PhC-enhanced extraction: 40–55 % into a high-NA lens
        // Here we scale from the bare extraction by a 3× enhancement cap
        let eta_bare = self.fraction_into_na();
        (eta_bare * 3.0).min(0.55)
    }

    /// Extraction with a bullseye (circular Bragg) grating.
    ///
    /// The Yao-Smirnov bullseye grating focuses emission into a Gaussian-like
    /// far-field.  Demonstrated extraction > 85 % into NA = 0.65 objectives.
    ///
    /// `n_rings`: number of concentric grating rings (efficiency saturates ~8 rings)
    pub fn with_bullseye_grating(&self, n_rings: usize) -> f64 {
        // Saturating efficiency model: η(N) = η_max * (1 - exp(-N/N_sat))
        // Physical saturation: the diffraction efficiency of each additional ring
        // diminishes as near-field coupling decays exponentially.  N_sat ≈ 3 rings
        // in the coupled-mode picture (Yao-Smirnov design analysis).
        let eta_max = 0.88_f64; // 88 % peak extraction at saturation
        let n_sat = 2.5_f64; // characteristic saturation ring count
        let n = n_rings as f64;
        eta_max * (1.0 - (-n / n_sat).exp())
    }
}

// ─── SinglePhotonBenchmark ────────────────────────────────────────────────────

/// Figures of merit for benchmarking a single-photon source.
///
/// Captures the three key properties: brightness (B), indistinguishability (M),
/// and purity (1 − g²(0)), as well as the repetition rate.
#[derive(Debug, Clone)]
pub struct SinglePhotonBenchmark {
    /// Brightness — photons per excitation pulse into first lens (0–1)
    pub brightness: f64,
    /// Indistinguishability M (0–1)
    pub indistinguishability: f64,
    /// g²(0) — lower is better (0 = ideal SPE)
    pub g2_zero: f64,
    /// Repetition rate (MHz)
    pub repetition_rate_mhz: f64,
}

impl SinglePhotonBenchmark {
    /// Create a benchmark record.
    pub fn new(b: f64, m: f64, g2: f64, rep_mhz: f64) -> Self {
        Self {
            brightness: b.clamp(0.0, 1.0),
            indistinguishability: m.clamp(0.0, 1.0),
            g2_zero: g2.clamp(0.0, 1.0),
            repetition_rate_mhz: rep_mhz.max(0.0),
        }
    }

    /// Raw key rate for quantum key distribution (kbits/s).
    ///
    /// R_QKD ≈ B · f_rep · η_channel · M · (1 − 2*h(e))
    /// Simplified as: R_QKD = B · f_rep · η_channel  (kbps, ignoring QBER)
    ///
    /// `channel_efficiency`: total channel transmittance (0–1)
    pub fn qkd_key_rate_kbps(&self, channel_efficiency: f64) -> f64 {
        let f_hz = self.repetition_rate_mhz * 1e6;
        let purity = (1.0 - self.g2_zero).max(0.0);
        // Simplistic sifted key estimate: multiply by 0.5 for BB84 basis reconciliation
        let raw_rate_bps = self.brightness * f_hz * channel_efficiency * purity * 0.5;
        raw_rate_bps / 1000.0 // bps → kbps
    }

    /// Boson sampling complexity metric for n-photon sampling.
    ///
    /// Quantum computational advantage requires (B·M)^n to scale favourably
    /// over classical simulation.  This metric computes (B·M)^n.
    pub fn boson_sampling_merit(&self, n_photons: usize) -> f64 {
        let bm = (self.brightness * self.indistinguishability).clamp(0.0, 1.0);
        bm.powi(n_photons as i32)
    }

    /// Figure of merit for entanglement distribution.
    ///
    /// FOM_entangle = B · M · (1 − g²(0))
    /// Optimised to 1 only for B=1, M=1, g²(0)=0.
    pub fn entanglement_merit(&self) -> f64 {
        let purity = (1.0 - self.g2_zero).max(0.0);
        self.brightness * self.indistinguishability * purity
    }

    /// Overall benchmark score ∈ [0, 1].
    ///
    /// Geometric mean of brightness, indistinguishability, and purity,
    /// weighted equally:
    ///   S = (B · M · (1 − g²(0)))^{1/3}
    pub fn overall_score(&self) -> f64 {
        let purity = (1.0 - self.g2_zero).max(0.0);
        (self.brightness * self.indistinguishability * purity)
            .max(0.0)
            .powf(1.0 / 3.0)
    }
}

// ─── Purcell factor calculator ────────────────────────────────────────────────

/// Compute the Purcell factor F_P for a cavity with given parameters.
///
/// F_P = (3/4π²) · (λ/n)³ · Q/V
///
/// - `wavelength_nm`: emitter wavelength in vacuum (nm)
/// - `refractive_index`: cavity medium refractive index
/// - `q_factor`: quality factor Q of the cavity mode
/// - `mode_volume_lambda_cubed`: mode volume V in units of (λ/n)³
///
/// Returns the on-resonance Purcell factor for a perfectly positioned emitter
/// with aligned dipole moment.
pub fn purcell_factor(
    wavelength_nm: f64,
    refractive_index: f64,
    q_factor: f64,
    mode_volume_lambda_cubed: f64,
) -> f64 {
    // F_P = (3/4π²) · Q/V  (in units where V is normalised to (λ/n)³)
    if mode_volume_lambda_cubed <= 0.0 || q_factor <= 0.0 {
        return 1.0;
    }
    let _ = wavelength_nm;
    let _ = refractive_index;
    (3.0 / (4.0 * PI * PI)) * q_factor / mode_volume_lambda_cubed
}

/// Compute the effective mode volume (in units of (λ/n)³) for common cavities.
///
/// Returns V_mode / (λ/n)³ for different photonic cavity types.
#[derive(Debug, Clone, PartialEq)]
pub enum CavityType {
    /// Micropillar (distributed Bragg reflector cavity): V ~ 5–30 (λ/n)³
    Micropillar {
        diameter_nm: f64,
        wavelength_nm: f64,
        refractive_index: f64,
    },
    /// Photonic crystal L3 cavity: V ~ 0.5–1.0 (λ/n)³
    PhCL3 { lattice_constant_nm: f64 },
    /// Open Fabry-Pérot microcavity: V ~ 10–1000 (λ/n)³
    FabryPerot {
        length_um: f64,
        beam_waist_um: f64,
        wavelength_nm: f64,
        refractive_index: f64,
    },
    /// Microring resonator: V ~ 100–10000 (λ/n)³
    MicroRing {
        radius_um: f64,
        cross_section_um2: f64,
        wavelength_nm: f64,
        refractive_index: f64,
    },
}

impl CavityType {
    /// Approximate mode volume in units of (λ/n)³.
    pub fn mode_volume_lambda_cubed(&self) -> f64 {
        match self {
            CavityType::Micropillar {
                diameter_nm,
                wavelength_nm,
                refractive_index,
            } => {
                // Approximate: V ≈ π*(d/2)² * λ_cav / (λ/n)³
                let lambda_n_nm = wavelength_nm / refractive_index; // nm
                let r = diameter_nm / 2.0;
                let area = PI * r * r;
                // Effective mode height ≈ λ_cav / 2n (half-wavelength cavity)
                let height = lambda_n_nm / 2.0;
                let v_nm3 = area * height;
                let lambda_n3 = lambda_n_nm.powi(3);
                v_nm3 / lambda_n3
            }
            CavityType::PhCL3 {
                lattice_constant_nm,
            } => {
                // L3 cavity: V ≈ 0.7 (λ/n)³ for optimised holes
                // Weak dependence on lattice constant (affects Q not V strongly)
                let _a = lattice_constant_nm;
                0.7
            }
            CavityType::FabryPerot {
                length_um,
                beam_waist_um,
                wavelength_nm,
                refractive_index,
            } => {
                let lambda_n_nm = wavelength_nm / refractive_index;
                let waist_nm = beam_waist_um * 1000.0;
                let len_nm = length_um * 1000.0;
                let v_nm3 = PI * waist_nm * waist_nm * len_nm;
                v_nm3 / lambda_n_nm.powi(3)
            }
            CavityType::MicroRing {
                radius_um,
                cross_section_um2,
                wavelength_nm,
                refractive_index,
            } => {
                let lambda_n_nm = wavelength_nm / refractive_index;
                // Mode volume ≈ circumference × cross-section area
                let circumf_nm = 2.0 * PI * radius_um * 1000.0;
                let xs_nm2 = cross_section_um2 * 1e6; // μm² → nm²
                let v_nm3 = circumf_nm * xs_nm2;
                v_nm3 / lambda_n_nm.powi(3)
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─ CavityEnhancedSource ───────────────────────────────────────────────────

    #[test]
    fn test_cavity_lifetime_shorter_with_purcell() {
        let src = CavityEnhancedSource::new(1.0, 10.0, 0.95, 0.95, 0.8);
        let tau_cav = src.cavity_lifetime_ns();
        assert!(
            tau_cav < 1.0,
            "Cavity lifetime should be shorter than free-space lifetime; got {tau_cav:.3} ns"
        );
        assert!(
            (tau_cav - 0.1).abs() < 1e-9,
            "τ_cav = τ₀/F_P = 0.1 ns; got {tau_cav:.6}"
        );
    }

    #[test]
    fn test_beta_enhanced_approaches_unity_at_high_purcell() {
        // F_P = 100 → β_eff → 1
        let src = CavityEnhancedSource::new(1.0, 100.0, 0.90, 0.95, 0.8);
        let beta = src.beta_factor_enhanced();
        assert!(
            beta > 0.98,
            "β_eff should approach 1 at large Purcell factor; got {beta:.4}"
        );
    }

    #[test]
    fn test_indistinguishability_increases_with_purcell() {
        let gamma_pure = 0.1_f64; // GHz — fixed dephasing
        let src_low = CavityEnhancedSource::new(1.0, 1.0, 0.95, 0.95, 0.8);
        let src_high = CavityEnhancedSource::new(1.0, 50.0, 0.95, 0.95, 0.8);
        let m_low = src_low.indistinguishability(gamma_pure);
        let m_high = src_high.indistinguishability(gamma_pure);
        assert!(
            m_high > m_low,
            "Higher Purcell factor should give higher indistinguishability; low={m_low:.4}, high={m_high:.4}"
        );
    }

    #[test]
    fn test_g2_zero_bounded() {
        let src = CavityEnhancedSource::new(1.0, 10.0, 0.95, 0.95, 0.8);
        let g2 = src.g2_zero(0.01);
        assert!(
            (0.0..=1.0).contains(&g2),
            "g²(0) must be in [0,1]; got {g2}"
        );
    }

    // ─ ExtractionEfficiency ───────────────────────────────────────────────────

    #[test]
    fn test_critical_angle_gaas() {
        // GaAs n=3.5 → θ_c = arcsin(1/3.5) ≈ 16.6°
        let ee = ExtractionEfficiency::new(3.5, 0.65);
        let theta_c = ee.critical_angle_deg();
        assert!(
            (theta_c - 16.6).abs() < 0.5,
            "Critical angle for GaAs should be ~16.6°; got {theta_c:.2}°"
        );
    }

    #[test]
    fn test_tir_limited_fraction_small_for_high_index() {
        let ee = ExtractionEfficiency::new(3.5, 0.65);
        let eta_tir = ee.tir_limited_fraction();
        // For n=3.5, escape fraction ≈ 2 % per facet
        assert!(
            eta_tir < 0.05,
            "TIR-limited extraction should be < 5 % for n=3.5; got {eta_tir:.4}"
        );
    }

    #[test]
    fn test_bullseye_saturates() {
        let ee = ExtractionEfficiency::new(3.5, 0.65);
        let eta_1 = ee.with_bullseye_grating(1);
        let eta_8 = ee.with_bullseye_grating(8);
        let eta_20 = ee.with_bullseye_grating(20);
        assert!(eta_8 > eta_1, "More rings should improve extraction");
        // Should saturate: η(20) ≈ η(8)
        assert!(
            (eta_20 - eta_8).abs() < 0.05,
            "Bullseye extraction should saturate beyond ~8 rings; Δη = {:.4}",
            (eta_20 - eta_8).abs()
        );
    }

    #[test]
    fn test_ar_coating_improves_extraction() {
        let ee = ExtractionEfficiency::new(3.5, 0.65);
        let eta_bare = ee.fraction_into_na();
        let eta_ar = ee.with_ar_coating();
        assert!(
            eta_ar >= eta_bare,
            "AR coating should improve extraction; bare={eta_bare:.4}, AR={eta_ar:.4}"
        );
    }

    // ─ SinglePhotonBenchmark ──────────────────────────────────────────────────

    #[test]
    fn test_benchmark_overall_score_ideal() {
        let bench = SinglePhotonBenchmark::new(1.0, 1.0, 0.0, 76.0);
        let score = bench.overall_score();
        assert!(
            (score - 1.0).abs() < 1e-10,
            "Ideal SPE (B=M=1, g²=0) should score 1.0; got {score}"
        );
    }

    #[test]
    fn test_benchmark_overall_score_zero_brightness() {
        let bench = SinglePhotonBenchmark::new(0.0, 1.0, 0.0, 76.0);
        assert_eq!(bench.overall_score(), 0.0, "Zero brightness → score = 0");
    }

    #[test]
    fn test_boson_sampling_merit_decreases_with_n() {
        let bench = SinglePhotonBenchmark::new(0.9, 0.95, 0.02, 76.0);
        let m2 = bench.boson_sampling_merit(2);
        let m10 = bench.boson_sampling_merit(10);
        assert!(
            m2 > m10,
            "Boson sampling merit should decrease with photon number; m2={m2:.4}, m10={m10:.6}"
        );
    }

    #[test]
    fn test_qkd_key_rate_positive() {
        let bench = SinglePhotonBenchmark::new(0.8, 0.95, 0.01, 80.0);
        let rate = bench.qkd_key_rate_kbps(0.1);
        assert!(rate > 0.0, "QKD key rate must be positive; got {rate}");
    }

    // ─ Purcell factor ─────────────────────────────────────────────────────────

    #[test]
    fn test_purcell_factor_positive() {
        let fp = purcell_factor(920.0, 3.5, 10_000.0, 0.7);
        assert!(
            fp > 0.0 && fp.is_finite(),
            "Purcell factor must be positive and finite; got {fp}"
        );
    }

    #[test]
    fn test_phc_l3_mode_volume() {
        let cav = CavityType::PhCL3 {
            lattice_constant_nm: 250.0,
        };
        let v = cav.mode_volume_lambda_cubed();
        // L3 cavity: V ≈ 0.7 (λ/n)³
        assert!(
            (v - 0.7).abs() < 0.01,
            "PhC L3 mode volume should be ~0.7 (λ/n)³; got {v:.4}"
        );
    }
}
