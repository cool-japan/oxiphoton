use std::f64::consts::PI;

// ─── Physical constants ───────────────────────────────────────────────────────

const C0: f64 = 2.997_924_58e8; // m/s

// ─── SersSubstrate ───────────────────────────────────────────────────────────

/// Characterisation of a SERS substrate (e.g., bowtie arrays, roughened metal,
/// nanoparticle aggregates).
///
/// The key figure of merit is the hotspot enhancement factor at both the pump
/// and Stokes wavelengths.
#[derive(Debug, Clone)]
pub struct SersSubstrate {
    /// Field enhancement |E/E₀| at the hotspot (at laser wavelength)
    pub hotspot_field_enhancement: f64,
    /// Hotspot surface density in m⁻² (number of hotspots per m²)
    pub hotspot_density: f64,
    /// Normal Raman cross-section per molecule in m²/sr
    pub molecule_cross_section: f64,
    /// Laser excitation wavelength in metres
    pub laser_wavelength: f64,
    /// Raman Stokes wavelength in metres
    pub raman_wavelength: f64,
}

impl SersSubstrate {
    /// Construct a bowtie-based SERS substrate.
    ///
    /// # Arguments
    /// * `enhancement`  - |E/E₀| at hotspot
    /// * `density`      - hotspot density in m⁻² (e.g., 1e12 for dense array)
    /// * `laser_wl`     - laser wavelength in metres
    pub fn new_bowtie(enhancement: f64, density: f64, laser_wl: f64) -> Self {
        // Raman wavelength for a typical Stokes shift (~1000 cm⁻¹ from 532 nm → ~560 nm)
        let shift_cm1 = 1000.0_f64;
        let k_laser = 1.0 / laser_wl * 1.0e-2; // wavenumber in cm⁻¹
        let k_raman = k_laser - shift_cm1;
        let raman_wl = if k_raman > 0.0 {
            1.0 / (k_raman * 1.0e2)
        } else {
            laser_wl * 1.05
        };

        Self {
            hotspot_field_enhancement: enhancement,
            hotspot_density: density,
            molecule_cross_section: 1.0e-30, // typical for aromatic molecule [m²/sr]
            laser_wavelength: laser_wl,
            raman_wavelength: raman_wl,
        }
    }

    /// Electromagnetic SERS enhancement factor.
    ///
    /// EF_EM = |E(ω_L)|² |E(ω_R)|² / |E₀|⁴
    ///
    /// In the equal-wavelength approximation (small Stokes shift):
    ///   EF_EM ≈ |E/E₀|⁴
    pub fn electromagnetic_ef(&self) -> f64 {
        let g = self.hotspot_field_enhancement;
        g * g * g * g
    }

    /// SERS Raman cross-section per molecule in m²/sr.
    ///
    /// σ_SERS = EF_EM × σ_Raman
    pub fn sers_cross_section(&self) -> f64 {
        self.electromagnetic_ef() * self.molecule_cross_section
    }

    /// Single-molecule SERS signal (detected photons per second).
    ///
    /// S = σ_SERS · (I_laser / ħω_laser) · η_collection
    ///
    /// # Arguments
    /// * `laser_intensity_w_per_m2` - laser irradiance at the hotspot (W/m²)
    /// * `collection_efficiency`    - fraction of scattered photons collected (0–1)
    pub fn single_molecule_signal(
        &self,
        laser_intensity_w_per_m2: f64,
        collection_efficiency: f64,
    ) -> f64 {
        const HBAR: f64 = 1.054_571_817e-34;
        let omega_laser = 2.0 * PI * C0 / self.laser_wavelength;
        // Photon flux: N_photon/s/m² = I / (ħω)
        let photon_flux = laser_intensity_w_per_m2 / (HBAR * omega_laser);
        // Signal = σ_SERS [m²/sr] × flux [photons/s/m²] × Ω_coll [sr] × η_coll
        // For a 4π solid angle collection: Ω_coll ≈ 4π × collection_efficiency
        // We return signal in photons/s (integrated over all solid angles):
        self.sers_cross_section() * photon_flux * 4.0 * PI * collection_efficiency
    }

    /// Minimum detectable surface density of molecules for a given SNR target.
    ///
    /// For a shot-noise limited detector with integration time T:
    ///   SNR = S · T / sqrt(S · T)  →  N_min = SNR² / S
    ///
    /// Returns number of molecules per m² for the given conditions.
    ///
    /// # Arguments
    /// * `laser_intensity`    - W/m²
    /// * `collection_eff`     - 0–1
    /// * `snr`               - required signal-to-noise ratio (e.g., 3 for LOD)
    /// * `integration_time`   - seconds
    pub fn detection_limit(
        &self,
        laser_intensity: f64,
        collection_eff: f64,
        snr: f64,
        integration_time: f64,
    ) -> f64 {
        let signal_per_molecule = self.single_molecule_signal(laser_intensity, collection_eff);
        if signal_per_molecule < f64::EPSILON || integration_time < f64::EPSILON {
            return f64::INFINITY;
        }
        // SNR for N molecules: snr = sqrt(N · S · T)  → N_min = (snr²) / (S · T)
        let n_min_molecules = snr * snr / (signal_per_molecule * integration_time);
        // Convert to molecules / m² using hotspot density
        n_min_molecules / (self.hotspot_density.max(1.0))
    }

    /// Chemical enhancement factor from charge-transfer interactions.
    ///
    /// The chemical EF arises from molecule-metal charge transfer and is
    /// approximately 10–100 (independent of the structure geometry).
    pub fn chemical_ef() -> f64 {
        // Canonical value from SERS literature (Moskovits 1985, Lombardi 2008)
        30.0
    }

    /// Total SERS enhancement factor: EF_total = EF_EM × EF_chem
    pub fn total_ef(&self) -> f64 {
        self.electromagnetic_ef() * Self::chemical_ef()
    }
}

// ─── TersSetup ───────────────────────────────────────────────────────────────

/// Tip-Enhanced Raman Spectroscopy (TERS) configuration.
///
/// TERS uses a metallic scanning probe tip as a nanoscale antenna to
/// concentrate light at the apex.  The tip radius determines both the
/// spatial resolution and the field enhancement.
///
/// Tip enhancement model:
///   |E_tip / E_far|² ≈ Q · (λ / (2π r_tip))²  (lightning-rod + resonance)
///
/// where Q is the quality factor of the tip plasmon resonance.
#[derive(Debug, Clone)]
pub struct TersSetup {
    /// Tip apex radius in nm
    pub tip_radius_nm: f64,
    /// Tip material: "gold" or "silver"
    pub tip_material: String,
    /// Gap between tip and sample in nm
    pub gap_nm: f64,
    /// Laser excitation wavelength in metres
    pub laser_wavelength: f64,
    /// Excitation power in µW
    pub excitation_power_uw: f64,
}

impl TersSetup {
    /// Construct a gold TERS tip configuration.
    ///
    /// # Arguments
    /// * `radius_nm`   - tip apex radius in nm
    /// * `gap_nm`      - tip-sample gap in nm
    /// * `wavelength`  - excitation wavelength in metres
    pub fn new_gold_tip(radius_nm: f64, gap_nm: f64, wavelength: f64) -> Self {
        Self {
            tip_radius_nm: radius_nm,
            tip_material: String::from("gold"),
            gap_nm,
            laser_wavelength: wavelength,
            excitation_power_uw: 100.0,
        }
    }

    /// Quality factor of the tip plasmon resonance (Drude estimate).
    fn tip_q(&self) -> f64 {
        // Gold Q factor at visible: Q ≈ Re(ε_m) / Im(ε_m) / 2
        // Drude gold: ωp=1.37e16, γ=1.22e14
        let omega_p = 1.37e16_f64;
        let gamma = 1.22e14_f64;
        let omega = 2.0 * PI * C0 / self.laser_wavelength;
        let eps_re = -(omega_p * omega_p) / (omega * omega + gamma * gamma);
        let eps_im = gamma * omega_p * omega_p / (omega * (omega * omega + gamma * gamma));
        if eps_im.abs() < f64::EPSILON {
            return 10.0;
        }
        (eps_re.abs() / eps_im).clamp(1.0, 50.0)
    }

    /// Electric field enhancement |E_tip / E_far| at the tip apex.
    ///
    /// Combines lightning-rod effect and plasmon resonance:
    ///   g_tip ≈ Q · (λ / (4π r))^(1/2)
    pub fn field_enhancement(&self) -> f64 {
        let q = self.tip_q();
        let lambda = self.laser_wavelength;
        let r = self.tip_radius_nm * 1.0e-9;
        // Gap enhancement factor: additional ~(r_tip/gap)^(1/3) from mirror interaction
        let gap_factor = (self.tip_radius_nm / self.gap_nm).powf(1.0 / 3.0);
        let base_fe = q * (lambda / (4.0 * PI * r)).sqrt();
        (base_fe * gap_factor).min(1.0e4) // physical cap
    }

    /// Spatial resolution of TERS in nm.
    ///
    /// Resolution ≈ tip radius (Abbe-beating by near-field confinement).
    pub fn spatial_resolution_nm(&self) -> f64 {
        self.tip_radius_nm
    }

    /// TERS electromagnetic enhancement factor.
    ///
    /// EF_TERS = |E_tip/E_far|⁴
    pub fn ters_enhancement_factor(&self) -> f64 {
        let fe = self.field_enhancement();
        fe * fe * fe * fe
    }

    /// Near-field to far-field signal ratio in dB.
    ///
    /// Ratio = 10 log₁₀(EF_TERS)
    pub fn near_field_to_far_field_ratio_db(&self) -> f64 {
        10.0 * self.ters_enhancement_factor().log10()
    }

    /// TERS contrast: ratio of near-field (tip-in) to far-field (tip-retracted) signal.
    ///
    /// In practice, contrast = (S_tip_in − S_tip_out) / S_tip_out ≈ EF_TERS × (A_tip/A_spot)
    /// where A_tip/A_spot is the ratio of tip aperture to diffraction-limited spot area.
    pub fn signal_contrast(&self) -> f64 {
        let a_tip = PI * (self.tip_radius_nm * 1.0e-9).powi(2);
        let a_spot = PI * (self.laser_wavelength / 2.0).powi(2);
        if a_spot < f64::EPSILON {
            return 1.0;
        }
        self.ters_enhancement_factor() * (a_tip / a_spot)
    }
}

// ─── NanotagType ─────────────────────────────────────────────────────────────

/// Type of nanoparticle used as SERS nanotag.
#[derive(Debug, Clone)]
pub enum NanotagType {
    /// Gold nanosphere with given diameter in nm
    AuNanosphere { diameter_nm: f64 },
    /// Gold nanostar: core + protruding arms for multiple hotspots
    AuNanostar { core_nm: f64, arm_length_nm: f64 },
    /// Aggregated gold nanoparticle clusters
    AuAggregates { n_particles: usize },
}

// ─── SersNanotag ─────────────────────────────────────────────────────────────

/// SERS nanotag: nanoparticle coated with Raman reporter molecules.
///
/// Nanotags are used as molecular barcodes in biosensing, imaging, and
/// multiplexed detection assays.  Multiple tags can be distinguished by their
/// unique Raman fingerprint spectra.
#[derive(Debug, Clone)]
pub struct SersNanotag {
    /// Nanoparticle type and geometry
    pub particle_type: NanotagType,
    /// Raman reporter molecule name
    pub molecule: String,
    /// SERS enhancement factor (EF_EM)
    pub ef: f64,
    /// Effective SERS cross-section per molecule in pm² (= 1e-24 m²)
    pub cross_section_pm2: f64,
}

impl SersNanotag {
    /// Construct a gold nanosphere SERS nanotag with a Raman reporter.
    ///
    /// Enhancement factor is estimated from sphere surface plasmon enhancement.
    ///
    /// # Arguments
    /// * `diameter_nm` - sphere diameter in nm
    /// * `molecule`    - Raman reporter name (e.g., "4-MBA", "DTNB")
    pub fn new_gold_nanosphere(diameter_nm: f64, molecule: &str) -> Self {
        // Quasi-static Mie enhancement for a sphere at LSPR:
        // EF ≈ (9 ε_d / |ε_m + 2ε_d|)⁴
        // For gold in water (ε_d=1.77) at LSPR: EF ~ 1e4
        let ef_base = 1.0e6_f64; // single sphere; improved by molecule-surface proximity
                                 // Larger spheres have lower curvature → slightly less enhancement per volume
        let size_factor = (30.0_f64 / diameter_nm).powi(2).clamp(0.1, 10.0);
        let ef = ef_base * size_factor;
        // Cross-section: σ_SERS ≈ EF × σ_normal where σ_normal ~ 1e-30 m²/sr × 4π
        let sigma_pm2 = ef * 1.0e-30 * 4.0 * PI * 1.0e24; // in pm²

        Self {
            particle_type: NanotagType::AuNanosphere { diameter_nm },
            molecule: molecule.to_string(),
            ef,
            cross_section_pm2: sigma_pm2,
        }
    }

    /// Construct a gold nanostar SERS nanotag.
    ///
    /// Nanostars have sharp tips that act as lightning rods, providing very
    /// high field enhancement (EF ~ 10⁸–10¹⁰).
    ///
    /// # Arguments
    /// * `core_nm`      - nanostar core diameter in nm
    /// * `arm_nm`       - arm length in nm
    /// * `molecule`     - Raman reporter molecule
    pub fn new_gold_nanostar(core_nm: f64, arm_nm: f64, molecule: &str) -> Self {
        // Each arm tip provides ~10× higher enhancement than a sphere
        // Nanostar EF ~ 1e8–1e10
        let n_arms = 6_usize; // typical nanostar has 6–10 arms
        let ef_per_arm = 1.0e8_f64 * (arm_nm / 40.0).powi(2).clamp(0.1, 100.0);
        let ef = ef_per_arm * n_arms as f64;
        let sigma_pm2 = ef * 1.0e-30 * 4.0 * PI * 1.0e24;

        Self {
            particle_type: NanotagType::AuNanostar {
                core_nm,
                arm_length_nm: arm_nm,
            },
            molecule: molecule.to_string(),
            ef,
            cross_section_pm2: sigma_pm2,
        }
    }

    /// Theoretical detection sensitivity in picomolar (pM) concentration.
    ///
    /// Estimated from: C_min ~ 1 / (N_A × V_spot × σ_SERS / σ_threshold)
    /// where σ_threshold is the minimum detectable cross-section and V_spot
    /// is the focal volume (~1 fL = 1e-15 L).
    ///
    /// Simplified empirical scaling: C_pM ≈ 1 / (EF / 1e6)^0.5
    pub fn detection_sensitivity_pm(&self) -> f64 {
        if self.ef < f64::EPSILON {
            return f64::INFINITY;
        }
        // Empirical: better enhancement → lower detection limit
        let ef_norm = self.ef / 1.0e6;
        1.0 / ef_norm.sqrt()
    }

    /// Number of distinct Raman reporters that can be multiplexed.
    ///
    /// The multiplexing capacity is limited by the spectral bandwidth and the
    /// Raman peak density in the fingerprint region (500–1800 cm⁻¹):
    ///   - fingerprint range: 1300 cm⁻¹
    ///   - spectral resolution: ~2 cm⁻¹ (spectrometer)
    ///   - typical peak width: 15 cm⁻¹
    ///   - capacity ≈ 1300 / 15 ≈ 87  (but limited to ~30 in practice)
    ///
    /// Returns realistic multiplexing capacity based on nanotag type.
    pub fn multiplexing_capacity(&self) -> usize {
        match &self.particle_type {
            NanotagType::AuNanosphere { .. } => 30,
            NanotagType::AuNanostar { .. } => 25, // broader peaks
            NanotagType::AuAggregates { n_particles } => {
                // More particles → more hotspots but less reproducible spectra
                (*n_particles).min(10)
            }
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── SersSubstrate ─────────────────────────────────────────────────────────

    #[test]
    fn test_electromagnetic_ef_is_g4() {
        let sub = SersSubstrate::new_bowtie(100.0, 1.0e12, 532.0e-9);
        let ef = sub.electromagnetic_ef();
        assert_abs_diff_eq!(ef, 100.0_f64.powi(4), epsilon = 1.0);
    }

    #[test]
    fn test_sers_cross_section_gt_normal() {
        let sub = SersSubstrate::new_bowtie(100.0, 1.0e12, 532.0e-9);
        let sigma_sers = sub.sers_cross_section();
        assert!(
            sigma_sers > sub.molecule_cross_section,
            "SERS cross-section must exceed normal Raman: {sigma_sers:.3e} vs {:.3e}",
            sub.molecule_cross_section
        );
    }

    #[test]
    fn test_total_ef_includes_chemical() {
        let sub = SersSubstrate::new_bowtie(100.0, 1.0e12, 532.0e-9);
        let ef_em = sub.electromagnetic_ef();
        let ef_total = sub.total_ef();
        let ef_chem = SersSubstrate::chemical_ef();
        assert_abs_diff_eq!(ef_total, ef_em * ef_chem, epsilon = 1.0);
    }

    #[test]
    fn test_single_molecule_signal_positive() {
        let sub = SersSubstrate::new_bowtie(200.0, 1.0e12, 532.0e-9);
        let signal = sub.single_molecule_signal(1.0e9, 0.05);
        assert!(
            signal > 0.0,
            "Single-molecule signal must be positive: {signal}"
        );
    }

    #[test]
    fn test_detection_limit_decreases_with_higher_ef() {
        let sub_low = SersSubstrate::new_bowtie(100.0, 1.0e12, 532.0e-9);
        let sub_high = SersSubstrate::new_bowtie(1000.0, 1.0e12, 532.0e-9);
        let lod_low = sub_low.detection_limit(1.0e9, 0.05, 3.0, 1.0);
        let lod_high = sub_high.detection_limit(1.0e9, 0.05, 3.0, 1.0);
        assert!(
            lod_high < lod_low,
            "Higher EF should give lower detection limit: {lod_high:.3e} vs {lod_low:.3e}"
        );
    }

    #[test]
    fn test_chemical_ef_constant() {
        let ef_chem = SersSubstrate::chemical_ef();
        assert!(
            (10.0..=1000.0).contains(&ef_chem),
            "Chemical EF should be 10–1000: {ef_chem}"
        );
    }

    // ── TersSetup ─────────────────────────────────────────────────────────────

    #[test]
    fn test_ters_field_enhancement_positive() {
        let ters = TersSetup::new_gold_tip(20.0, 1.0, 633.0e-9);
        let fe = ters.field_enhancement();
        assert!(fe > 1.0, "TERS field enhancement must exceed 1: {fe}");
    }

    #[test]
    fn test_ters_enhancement_factor_is_fe4() {
        let ters = TersSetup::new_gold_tip(20.0, 1.0, 633.0e-9);
        let fe = ters.field_enhancement();
        let ef = ters.ters_enhancement_factor();
        assert_abs_diff_eq!(ef, fe.powi(4), epsilon = 1.0);
    }

    #[test]
    fn test_ters_spatial_resolution_equals_tip_radius() {
        let ters = TersSetup::new_gold_tip(15.0, 1.0, 633.0e-9);
        assert_abs_diff_eq!(ters.spatial_resolution_nm(), 15.0, epsilon = 1.0e-10);
    }

    #[test]
    fn test_ters_near_field_db_positive_for_enhancement() {
        let ters = TersSetup::new_gold_tip(20.0, 1.0, 633.0e-9);
        let db = ters.near_field_to_far_field_ratio_db();
        assert!(
            db > 0.0,
            "Near-field to far-field ratio in dB should be positive: {db}"
        );
    }

    #[test]
    fn test_ters_smaller_tip_gives_better_resolution() {
        let ters1 = TersSetup::new_gold_tip(10.0, 1.0, 633.0e-9);
        let ters2 = TersSetup::new_gold_tip(30.0, 1.0, 633.0e-9);
        assert!(
            ters1.spatial_resolution_nm() < ters2.spatial_resolution_nm(),
            "Smaller tip → better resolution"
        );
    }

    // ── SersNanotag ──────────────────────────────────────────────────────────

    #[test]
    fn test_gold_nanosphere_nanotag_ef_positive() {
        let tag = SersNanotag::new_gold_nanosphere(40.0, "4-MBA");
        assert!(tag.ef > 0.0, "Nanotag EF must be positive: {}", tag.ef);
    }

    #[test]
    fn test_gold_nanostar_nanotag_ef_gt_sphere() {
        let sphere = SersNanotag::new_gold_nanosphere(40.0, "4-MBA");
        let star = SersNanotag::new_gold_nanostar(40.0, 50.0, "4-MBA");
        assert!(
            star.ef > sphere.ef,
            "Nanostar EF ({:.3e}) should exceed sphere EF ({:.3e})",
            star.ef,
            sphere.ef
        );
    }

    #[test]
    fn test_detection_sensitivity_decreases_with_ef() {
        // Higher EF → lower detection limit (in pM)
        let sphere = SersNanotag::new_gold_nanosphere(40.0, "4-MBA");
        let star = SersNanotag::new_gold_nanostar(40.0, 50.0, "4-MBA");
        let lod_sphere = sphere.detection_sensitivity_pm();
        let lod_star = star.detection_sensitivity_pm();
        assert!(
            lod_star < lod_sphere,
            "Nanostar should have lower LOD: {lod_star:.3} vs {lod_sphere:.3} pM"
        );
    }

    #[test]
    fn test_multiplexing_capacity_nanosphere() {
        let tag = SersNanotag::new_gold_nanosphere(40.0, "4-MBA");
        let cap = tag.multiplexing_capacity();
        assert!(cap > 0 && cap <= 100, "Multiplexing capacity: {cap}");
    }

    #[test]
    fn test_multiplexing_capacity_aggregate_limited_by_n() {
        let agg = SersNanotag {
            particle_type: NanotagType::AuAggregates { n_particles: 5 },
            molecule: String::from("DTNB"),
            ef: 1.0e10,
            cross_section_pm2: 1.0e4,
        };
        assert_eq!(agg.multiplexing_capacity(), 5);
    }
}
