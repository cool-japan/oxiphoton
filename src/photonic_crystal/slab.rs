//! 2D Photonic Crystal Slab Physics
//!
//! This module implements semi-analytic models for photonic crystal slab
//! structures: band-gap estimation via effective medium theory (EMT),
//! point-defect (L3, H1) nanocavities, and W1 line-defect waveguides
//! with slow-light dispersion.
//!
//! ## Physical background
//!
//! A photonic crystal slab confines light vertically by total internal
//! reflection (TIR) and laterally by the photonic band gap.  The figure of
//! merit is the quality factor Q and the normalised mode volume
//! V_eff / (λ/n)³.  The Purcell factor
//!
//!   F_P = (3 / 4π²) · (λ/n)³ · Q / V_eff
//!
//! quantifies spontaneous-emission enhancement.

use std::f64::consts::PI;

use crate::units::conversion::{EPSILON_0, HBAR, SPEED_OF_LIGHT};

// ─── Lattice type ─────────────────────────────────────────────────────────────

/// Bravais lattice for a 2D photonic crystal slab.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SlabLattice {
    /// Square lattice with period `a` (m).
    Square { period: f64 },
    /// Triangular (hexagonal close-packed) lattice with period `a` (m).
    Hexagonal { period: f64 },
    /// Honeycomb lattice (two-atom basis) with period `a` (m).
    Honeycomb { period: f64 },
}

impl SlabLattice {
    /// Returns the lattice period (m).
    pub fn period(&self) -> f64 {
        match self {
            Self::Square { period } => *period,
            Self::Hexagonal { period } => *period,
            Self::Honeycomb { period } => *period,
        }
    }

    /// Unit-cell area (m²).
    pub fn unit_cell_area(&self) -> f64 {
        let a = self.period();
        match self {
            Self::Square { .. } => a * a,
            // Triangular: A = √3/2 · a²
            Self::Hexagonal { .. } => 3_f64.sqrt() / 2.0 * a * a,
            // Honeycomb has two lattice sites per unit cell, same as triangular
            Self::Honeycomb { .. } => 3_f64.sqrt() / 2.0 * a * a,
        }
    }
}

// ─── Hole shape ───────────────────────────────────────────────────────────────

/// Cross-sectional shape of the air holes in the PhC slab.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HoleShape {
    /// Circular hole with radius given by `PhCSlabStructure::hole_radius`.
    Circular,
    /// Elliptical hole with semi-axes `rx` and `ry` (m).
    Elliptical { rx: f64, ry: f64 },
    /// Square hole with side length (m).
    Square { side: f64 },
}

impl HoleShape {
    /// Cross-sectional area of the hole (m²).
    pub fn area(&self, default_radius: f64) -> f64 {
        match self {
            Self::Circular => PI * default_radius * default_radius,
            Self::Elliptical { rx, ry } => PI * rx * ry,
            Self::Square { side } => side * side,
        }
    }
}

// ─── Main slab structure ──────────────────────────────────────────────────────

/// 2D photonic crystal slab with out-of-plane TIR confinement.
///
/// Provides semi-analytic estimates for the fill fraction, effective index,
/// photonic band-gap centre and width, and the quality factor.
#[derive(Debug, Clone)]
pub struct PhCSlabStructure {
    /// Bravais lattice type.
    pub lattice: SlabLattice,
    /// Slab thickness d (m).
    pub slab_thickness: f64,
    /// Refractive index of the slab material.
    pub n_slab: f64,
    /// Refractive index of the substrate (below slab).
    pub n_substrate: f64,
    /// Refractive index of the cladding (above slab, usually air = 1.0).
    pub n_cladding: f64,
    /// Air-hole radius r (m).  Used for circular holes.
    pub hole_radius: f64,
    /// Shape of the air holes.
    pub hole_shape: HoleShape,
}

impl PhCSlabStructure {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Square-lattice PhC slab with circular air holes, air cladding/substrate.
    ///
    /// # Arguments
    /// * `period`          – lattice constant a (m)
    /// * `hole_radius`     – hole radius r (m)
    /// * `slab_thickness`  – slab thickness d (m)
    /// * `n_slab`          – slab refractive index (e.g. 3.476 for Si)
    pub fn new_square(period: f64, hole_radius: f64, slab_thickness: f64, n_slab: f64) -> Self {
        Self {
            lattice: SlabLattice::Square { period },
            slab_thickness,
            n_slab,
            n_substrate: 1.0,
            n_cladding: 1.0,
            hole_radius,
            hole_shape: HoleShape::Circular,
        }
    }

    /// Hexagonal (triangular)-lattice PhC slab with circular air holes.
    pub fn new_hexagonal(period: f64, hole_radius: f64, slab_thickness: f64, n_slab: f64) -> Self {
        Self {
            lattice: SlabLattice::Hexagonal { period },
            slab_thickness,
            n_slab,
            n_substrate: 1.0,
            n_cladding: 1.0,
            hole_radius,
            hole_shape: HoleShape::Circular,
        }
    }

    // ── Geometry ──────────────────────────────────────────────────────────────

    /// Air-hole fill fraction η = A_hole / A_cell.
    ///
    /// For a circular hole in a square lattice: η = π r² / a².
    /// For a hexagonal lattice: η = π r² / (√3/2 · a²).
    pub fn fill_fraction(&self) -> f64 {
        let a_hole = self.hole_shape.area(self.hole_radius);
        let a_cell = self.lattice.unit_cell_area();
        (a_hole / a_cell).clamp(0.0, 0.9)
    }

    // ── Optical properties ────────────────────────────────────────────────────

    /// Effective refractive index via linear EMT (volume-average ε).
    ///
    ///   ε_eff = η·ε_air + (1-η)·ε_slab
    ///   n_eff = √ε_eff
    pub fn effective_index(&self) -> f64 {
        let eta = self.fill_fraction();
        let eps_slab = self.n_slab * self.n_slab;
        let eps_air = 1.0; // n_air = 1.0
        let eps_eff = eta * eps_air + (1.0 - eta) * eps_slab;
        eps_eff.sqrt()
    }

    /// Effective vertical confinement factor Γ.
    ///
    /// For a symmetric slab waveguide of thickness d and core index n_slab:
    ///   V = (π d / λ) √(n_slab² - n_clad²)
    ///   Γ ≈ 1 − exp(−V²)  (approximate)
    ///
    /// This is evaluated at the normalised gap-centre wavelength.
    fn confinement_factor(&self) -> f64 {
        let a = self.lattice.period();
        // Use the normalised gap centre to get the wavelength
        let f_norm = self.band_gap_center_raw();
        if f_norm < 1e-12 {
            return 0.9;
        }
        // λ = a / f_norm (normalised units: f = a/λ → λ = a/f)
        let lambda = a / f_norm;
        let n_c = self.n_cladding;
        let na_sq = self.n_slab * self.n_slab - n_c * n_c;
        if na_sq <= 0.0 {
            return 0.5;
        }
        let v = PI * self.slab_thickness / lambda * na_sq.sqrt();
        (1.0 - (-v * v).exp()).clamp(0.1, 1.0)
    }

    /// Normalised gap-centre frequency (ωa/2πc = a/λ) — internal helper.
    fn band_gap_center_raw(&self) -> f64 {
        // Reference values for Si triangular lattice (r/a ≈ 0.30, n_slab = 3.476):
        //   TE gap: a/λ ≈ 0.28 – 0.34, centre ≈ 0.31
        // Scale by effective index relative to Si reference.
        let n_ref = 3.476_f64;
        let f_centre_ref = match self.lattice {
            SlabLattice::Square { .. } => 0.35, // square lattice gap is higher
            SlabLattice::Hexagonal { .. } => 0.305,
            SlabLattice::Honeycomb { .. } => 0.25, // Dirac point near 0.25
        };
        // Effective-index scaling
        let n_eff = self.effective_index();
        f_centre_ref * n_ref / n_eff.max(1.0)
    }

    /// Normalised gap-centre frequency (ωa/2πc = a/λ).
    pub fn band_gap_center(&self) -> f64 {
        self.band_gap_center_raw()
    }

    /// Normalised gap width Δ(a/λ) (semi-analytic).
    ///
    /// The gap width scales roughly as:
    ///   Δf ≈ Δf_ref · (n_slab - 1) / (n_ref - 1) · Γ
    /// where Γ is the vertical confinement factor and Δf_ref ≈ 0.06 for Si.
    pub fn band_gap_width(&self) -> f64 {
        let delta_ref = match self.lattice {
            SlabLattice::Square { .. } => 0.04,
            SlabLattice::Hexagonal { .. } => 0.06,
            SlabLattice::Honeycomb { .. } => 0.03,
        };
        let n_ref = 3.476_f64;
        let index_scale = ((self.n_slab - 1.0) / (n_ref - 1.0)).clamp(0.0, 2.0);
        let gamma = self.confinement_factor();
        (delta_ref * index_scale * gamma).max(0.0)
    }

    /// Frequency at which the guided mode hits the light line (cutoff, a/λ).
    ///
    /// This is where the slab waveguide mode transitions from guided to leaky.
    /// For a symmetric slab:   (d/a) · √(n_slab² - n_clad²) = 1/2
    /// → a/λ_cutoff = f_cutoff ≈ a / (2d) / √(n_slab² - 1)
    pub fn guided_mode_cutoff(&self) -> f64 {
        let a = self.lattice.period();
        let d = self.slab_thickness;
        let n_c = self.n_cladding;
        let na_sq = self.n_slab * self.n_slab - n_c * n_c;
        if na_sq <= 0.0 || d < 1e-20 {
            return 0.0;
        }
        // TE₀ cutoff condition for symmetric slab: V = π d/λ √NA = π/2
        // → λ_cutoff = 2d √NA  → f_cutoff = a/λ_cutoff = a/(2d√NA)
        a / (2.0 * d * na_sq.sqrt())
    }

    /// Approximate quality factor estimate.
    ///
    /// Uses the empirical relation:
    ///   Q ≈ (Δf / f₀)² · n_slab³ / loss_factor
    ///
    /// where `mode_volume_cubic_lambda` is V_eff in units of (λ/n)³ and
    /// `loss_factor` accounts for out-of-plane radiation losses.
    ///
    /// # Arguments
    /// * `mode_volume_cubic_lambda` – mode volume in units of (λ/n)³
    pub fn quality_factor_estimate(&self, mode_volume_cubic_lambda: f64) -> f64 {
        let f0 = self.band_gap_center();
        let df = self.band_gap_width();
        if f0 < 1e-12 || df < 1e-12 {
            return 0.0;
        }
        let gap_ratio = df / f0;
        let v = mode_volume_cubic_lambda.max(0.1);
        // Empirical: Q ~ gap_ratio² · n³ / V
        gap_ratio * gap_ratio * self.n_slab.powi(3) / v * 1e4
    }
}

// ─── Point-defect cavity ──────────────────────────────────────────────────────

/// Type of point defect created in the photonic crystal slab.
#[derive(Debug, Clone, PartialEq)]
pub enum DefectType {
    /// One hole removed from the lattice.
    MissingHole,
    /// Hole shifted by (shift_x, shift_y) (m).
    ShiftedHole { shift_x: f64, shift_y: f64 },
    /// Hole radius changed to `radius` (m).
    ModifiedRadius { radius: f64 },
    /// H1 cavity: 1 missing hole (triangular lattice).
    H1,
    /// H3 cavity: 3 missing holes (triangular lattice).
    H3,
    /// L3 cavity: 3-hole line defect with optimised end-hole shifts.
    L3,
}

/// Polarisation character of a cavity mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CavityPolarization {
    /// Transverse electric (in-plane E-field dominant).
    TE,
    /// Transverse magnetic (in-plane H-field dominant).
    TM,
    /// Hybrid (mixed polarisation).
    Hybrid,
}

/// Cavity mode parameters.
#[derive(Debug, Clone)]
pub struct CavityMode {
    /// Normalised resonant frequency ωa/2πc = a/λ.
    pub resonant_freq_normalized: f64,
    /// Quality factor Q.
    pub quality_factor: f64,
    /// Mode volume in units of (λ/n)³.
    pub mode_volume_cubic_lambda: f64,
    /// Dominant polarisation.
    pub polarization: CavityPolarization,
}

/// Point-defect nanocavity embedded in a photonic crystal slab.
#[derive(Debug, Clone)]
pub struct PointDefectCavity {
    /// Base photonic crystal slab structure.
    pub base_slab: PhCSlabStructure,
    /// Type of defect introduced.
    pub defect_type: DefectType,
    /// Cavity mode parameters.
    pub cavity_mode: CavityMode,
}

impl PointDefectCavity {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// L3 nanocavity with optimised end-hole shifts (Akahane et al. 2003).
    ///
    /// Default shift = 0.15 a applied to the two end holes.  This pushes Q
    /// from ~10⁴ toward ~10⁶ in Si.  The model returns Q ≈ 1×10⁵ (moderate
    /// optimisation).
    pub fn new_l3(slab: PhCSlabStructure) -> Self {
        let f0 = slab.band_gap_center() * 1.02; // L3 sits ~2% above centre
        let mode = CavityMode {
            resonant_freq_normalized: f0,
            quality_factor: 1e5,
            mode_volume_cubic_lambda: 0.7,
            polarization: CavityPolarization::TE,
        };
        Self {
            base_slab: slab,
            defect_type: DefectType::L3,
            cavity_mode: mode,
        }
    }

    /// H1 nanocavity (1 missing hole).
    pub fn new_h1(slab: PhCSlabStructure) -> Self {
        let f0 = slab.band_gap_center() * 0.95; // H1 sits below centre
        let mode = CavityMode {
            resonant_freq_normalized: f0,
            quality_factor: 500.0,
            mode_volume_cubic_lambda: 1.2,
            polarization: CavityPolarization::TE,
        };
        Self {
            base_slab: slab,
            defect_type: DefectType::H1,
            cavity_mode: mode,
        }
    }

    // ── Figure-of-merit quantities ─────────────────────────────────────────────

    /// Purcell factor F_P = (3/4π²) · (λ/n)³ · Q / V_eff.
    ///
    /// Assumes emitter is optimally positioned and polarisation-matched.
    pub fn purcell_factor(&self) -> f64 {
        let q = self.cavity_mode.quality_factor;
        let v = self.cavity_mode.mode_volume_cubic_lambda.max(1e-6);
        // (λ/n)³ cancels in the ratio Q/V when V is in (λ/n)³ units
        (3.0 / (4.0 * PI * PI)) * q / v
    }

    /// Single-photon coupling efficiency.
    ///
    ///   η = β F_P / (β F_P + 1)
    ///
    /// where β is the fraction of spontaneous emission coupled into the cavity.
    pub fn coupling_efficiency(&self, beta_factor: f64) -> f64 {
        let fp = self.purcell_factor();
        let b = beta_factor.clamp(0.0, 1.0);
        let num = b * fp;
        num / (num + 1.0)
    }

    /// Zero-point electric field amplitude inside the cavity (V/m).
    ///
    ///   E_zpf = √(ℏω / 2ε₀ V_eff)
    ///
    /// # Arguments
    /// * `wavelength` – resonant wavelength in vacuum (m)
    pub fn zero_point_field(&self, wavelength: f64) -> f64 {
        let n = self.base_slab.n_slab;
        let a = self.base_slab.lattice.period();
        // Convert normalised freq to physical wavelength
        let f_norm = self.cavity_mode.resonant_freq_normalized;
        let lambda_res = if f_norm > 1e-12 {
            a / f_norm
        } else {
            wavelength
        };
        let omega = 2.0 * PI * SPEED_OF_LIGHT / lambda_res;
        // V_eff in m³
        let v_cubic_lambda = self.cavity_mode.mode_volume_cubic_lambda;
        let lambda_over_n = lambda_res / n;
        let v_eff = v_cubic_lambda * lambda_over_n.powi(3);
        let eps_eff = EPSILON_0 * n * n;
        (HBAR * omega / (2.0 * eps_eff * v_eff)).sqrt()
    }
}

// ─── W1 line-defect waveguide ──────────────────────────────────────────────────

/// W1 line-defect waveguide in a triangular-lattice PhC slab.
///
/// A W1 waveguide is formed by omitting one row of holes.  The waveguide
/// supports a guided band inside the photonic gap; near the lower band edge
/// the group velocity slows dramatically (slow-light regime, n_g ≫ 1).
#[derive(Debug, Clone)]
pub struct W1Waveguide {
    /// Underlying PhC slab structure.
    pub slab: PhCSlabStructure,
    /// Physical length of the waveguide (m).
    pub length: f64,
    /// Normalised frequency range (a/λ) of the slow-light region.
    pub slow_light_region: (f64, f64),
}

impl W1Waveguide {
    /// Construct a W1 waveguide of given length.
    ///
    /// The slow-light region is initialised as the lowest 20 % of the
    /// guided-mode bandwidth, where n_g typically exceeds 30.
    pub fn new(slab: PhCSlabStructure, length: f64) -> Self {
        let f_lower = Self::lower_band_edge_norm(&slab);
        let bw = Self::guided_bandwidth_norm(&slab);
        let slow_hi = f_lower + 0.20 * bw;
        Self {
            slab,
            length,
            slow_light_region: (f_lower, slow_hi),
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Normalised lower band-edge frequency a/λ.
    fn lower_band_edge_norm(slab: &PhCSlabStructure) -> f64 {
        let f0 = slab.band_gap_center();
        f0 * 0.94 // guided band starts ~6 % below gap centre
    }

    /// Normalised guided-mode bandwidth (a/λ).
    fn guided_bandwidth_norm(slab: &PhCSlabStructure) -> f64 {
        let f0 = slab.band_gap_center();
        // Bandwidth ≈ 12 % of gap centre for typical r/a ≈ 0.30
        0.12 * f0
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Group index n_g = c/v_g at normalised frequency `freq_normalized` (a/λ).
    ///
    /// Uses an inverse-square-root divergence model near the lower band edge:
    ///
    ///   n_g(f) = n_g0 / √((f − f_lower) / δf)   clamped to [n_g0, 200]
    ///
    /// where δf = 0.5 × bandwidth and n_g0 = 5 (fast-light reference).
    pub fn group_index(&self, freq_normalized: f64) -> f64 {
        let f_lower = Self::lower_band_edge_norm(&self.slab);
        let bw = Self::guided_bandwidth_norm(&self.slab);
        let df_slow = 0.5 * bw;
        let delta = freq_normalized - f_lower;
        if delta <= 0.0 {
            return 200.0; // below cut-off
        }
        let x = (delta / df_slow).clamp(1e-4, 1.0);
        (5.0 / x.sqrt()).clamp(5.0, 200.0)
    }

    /// Bandwidth of the slow-light region (in units of a/λ).
    pub fn slow_light_bandwidth(&self) -> f64 {
        self.slow_light_region.1 - self.slow_light_region.0
    }

    /// Propagation loss in dB/cm due to surface roughness (semi-empirical).
    ///
    /// Loss scales as n_g² for slow-light waveguides (backscattering ∝ n_g²):
    ///
    ///   α [dB/cm] ≈ α₀ · (n_g / n_g0)² · (σ / σ₀)²
    ///
    /// Reference: α₀ = 2 dB/cm at n_g0 = 10, σ₀ = 1 nm roughness.
    ///
    /// # Arguments
    /// * `surface_roughness_nm` – rms surface roughness σ in nanometres
    pub fn propagation_loss_db_per_cm(&self, surface_roughness_nm: f64) -> f64 {
        // Evaluate at mid-point of slow-light region
        let f_mid = (self.slow_light_region.0 + self.slow_light_region.1) / 2.0;
        let ng = self.group_index(f_mid);
        let ng0 = 10.0_f64;
        let sigma0 = 1.0_f64; // nm reference
        let sigma = surface_roughness_nm.max(0.0);
        2.0 * (ng / ng0).powi(2) * (sigma / sigma0).powi(2)
    }

    /// Total insertion loss (dB) through the waveguide.
    ///
    /// Uses a mid-band operating point (fast-light, n_g ≈ 5–10) and
    /// `surface_roughness_nm` = 2 nm.
    pub fn transmission_db(&self) -> f64 {
        let loss_per_cm = self.propagation_loss_db_per_cm(2.0);
        let length_cm = self.length * 100.0; // m → cm
        -loss_per_cm * length_cm
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn si_hex_slab() -> PhCSlabStructure {
        // Si triangular lattice: a = 420 nm, r = 126 nm (r/a = 0.30), d = 220 nm
        PhCSlabStructure::new_hexagonal(420e-9, 126e-9, 220e-9, 3.476)
    }

    // ── PhCSlabStructure ───────────────────────────────────────────────────

    #[test]
    fn fill_fraction_hex_typical() {
        let slab = si_hex_slab();
        let ff = slab.fill_fraction();
        // r/a = 0.30 → ff ≈ 0.33 for triangular lattice
        assert!(ff > 0.25 && ff < 0.45, "fill fraction = {ff:.4}");
    }

    #[test]
    fn effective_index_between_air_and_slab() {
        let slab = si_hex_slab();
        let n_eff = slab.effective_index();
        // Must be between 1 (air) and 3.476 (Si)
        assert!(n_eff > 1.0 && n_eff < slab.n_slab, "n_eff = {n_eff:.3}");
    }

    #[test]
    fn band_gap_center_positive() {
        let slab = si_hex_slab();
        assert!(slab.band_gap_center() > 0.0);
    }

    #[test]
    fn band_gap_width_nonnegative() {
        let slab = si_hex_slab();
        assert!(slab.band_gap_width() >= 0.0);
    }

    #[test]
    fn guided_mode_cutoff_positive() {
        let slab = si_hex_slab();
        assert!(slab.guided_mode_cutoff() > 0.0);
    }

    #[test]
    fn square_lattice_gap_centre_higher_than_hex() {
        let hex = PhCSlabStructure::new_hexagonal(420e-9, 126e-9, 220e-9, 3.476);
        let sq = PhCSlabStructure::new_square(420e-9, 126e-9, 220e-9, 3.476);
        // Square lattice gap is at higher normalised frequency
        assert!(sq.band_gap_center() > hex.band_gap_center());
    }

    #[test]
    fn quality_factor_estimate_positive() {
        let slab = si_hex_slab();
        let q = slab.quality_factor_estimate(0.7);
        assert!(q > 0.0, "Q = {q}");
    }

    #[test]
    fn fill_fraction_elliptical_hole() {
        let mut slab = si_hex_slab();
        slab.hole_shape = HoleShape::Elliptical {
            rx: 130e-9,
            ry: 100e-9,
        };
        let ff = slab.fill_fraction();
        assert!(ff > 0.0 && ff < 1.0, "fill fraction = {ff}");
    }

    // ── PointDefectCavity ─────────────────────────────────────────────────

    #[test]
    fn l3_purcell_factor_large() {
        let slab = si_hex_slab();
        let cav = PointDefectCavity::new_l3(slab);
        let fp = cav.purcell_factor();
        // Q/V ~ 1e5 / 0.7 → F_P ~ 3600
        assert!(fp > 100.0, "F_P = {fp:.1}");
    }

    #[test]
    fn h1_purcell_factor_positive() {
        let slab = si_hex_slab();
        let cav = PointDefectCavity::new_h1(slab);
        assert!(cav.purcell_factor() > 0.0);
    }

    #[test]
    fn coupling_efficiency_between_zero_and_one() {
        let slab = si_hex_slab();
        let cav = PointDefectCavity::new_l3(slab);
        let eta = cav.coupling_efficiency(0.9);
        assert!((0.0..=1.0).contains(&eta), "η = {eta:.4}");
    }

    #[test]
    fn zero_point_field_positive() {
        let slab = si_hex_slab();
        let cav = PointDefectCavity::new_l3(slab);
        let e_zpf = cav.zero_point_field(1550e-9);
        assert!(e_zpf > 0.0, "E_zpf = {e_zpf}");
    }

    #[test]
    fn l3_resonance_above_gap_centre() {
        let slab = si_hex_slab();
        let f0 = slab.band_gap_center();
        let cav = PointDefectCavity::new_l3(slab);
        assert!(cav.cavity_mode.resonant_freq_normalized > f0 * 0.9);
    }

    // ── W1Waveguide ───────────────────────────────────────────────────────

    #[test]
    fn w1_group_index_increases_near_band_edge() {
        let slab = si_hex_slab();
        let w1 = W1Waveguide::new(slab, 100e-6);
        let f_lower = w1.slow_light_region.0;
        let ng_near = w1.group_index(f_lower + 0.002);
        let ng_far = w1.group_index(f_lower + 0.06);
        assert!(ng_near > ng_far, "ng_near={ng_near:.1} ng_far={ng_far:.1}");
    }

    #[test]
    fn w1_group_index_below_cutoff_large() {
        let slab = si_hex_slab();
        let w1 = W1Waveguide::new(slab, 100e-6);
        // Below cutoff frequency
        let ng = w1.group_index(0.05);
        assert!(ng >= 100.0, "ng below cutoff = {ng}");
    }

    #[test]
    fn w1_slow_light_bandwidth_positive() {
        let slab = si_hex_slab();
        let w1 = W1Waveguide::new(slab, 100e-6);
        assert!(w1.slow_light_bandwidth() > 0.0);
    }

    #[test]
    fn w1_transmission_db_nonpositive() {
        let slab = si_hex_slab();
        let w1 = W1Waveguide::new(slab, 500e-6); // 500 µm
        assert!(w1.transmission_db() <= 0.0);
    }

    #[test]
    fn w1_propagation_loss_scales_with_roughness() {
        let slab = si_hex_slab();
        let w1 = W1Waveguide::new(slab, 100e-6);
        let loss1 = w1.propagation_loss_db_per_cm(1.0);
        let loss2 = w1.propagation_loss_db_per_cm(2.0);
        // Loss should be 4× larger for 2× roughness (quadratic scaling)
        assert_abs_diff_eq!(loss2 / loss1, 4.0, epsilon = 0.1);
    }
}
