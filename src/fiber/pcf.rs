//! Photonic crystal fiber (PCF) model.
//!
//! PCFs have a microstructured cladding with a regular array of air holes
//! surrounding a solid silica core. Key properties:
//!
//! - **Index-guiding PCF**: solid core, air-hole cladding — higher NA, endlessly single-mode
//! - **Hollow-core PCF**: photonic bandgap guiding — low nonlinearity, low loss
//!
//! For index-guiding PCF with triangular hole lattice:
//!   - Pitch Λ: hole-to-hole spacing
//!   - Hole diameter d
//!   - Air-filling fraction: f ≈ (π/2√3)·(d/Λ)²
//!   - Effective cladding index via effective-medium theory (EMT)
//!
//! Endlessly single-mode condition: d/Λ < 0.45
//!
//! References:
//!   Knight et al., OL 21, 1547 (1996)
//!   Mortensen et al., OL 28, 393 (2003)
//!   Birks et al., OL 22, 961 (1997)

use crate::error::OxiPhotonError;
use std::f64::consts::PI;

// Speed of light in m/s
const C0: f64 = 2.99792458e8;

// ─────────────────────────────────────────────────────────────────────────────
// Legacy simple struct (kept for backward compatibility)
// ─────────────────────────────────────────────────────────────────────────────

/// Photonic crystal fiber with triangular hole lattice.
#[derive(Debug, Clone, Copy)]
pub struct PhotonicCrystalFiber {
    /// Background (silica) refractive index
    pub n_silica: f64,
    /// Hole refractive index (1.0 for air)
    pub n_hole: f64,
    /// Lattice pitch Λ (m): distance between adjacent holes
    pub pitch: f64,
    /// Hole diameter d (m)
    pub hole_diameter: f64,
    /// Core radius (set to Λ for single missing hole)
    pub core_radius: f64,
}

impl PhotonicCrystalFiber {
    /// Create a PCF from core parameters.
    pub fn new(n_silica: f64, n_hole: f64, pitch: f64, hole_diameter: f64) -> Self {
        Self {
            n_silica,
            n_hole,
            pitch,
            hole_diameter,
            core_radius: pitch, // defect = 1 missing hole
        }
    }

    /// Endlessly single-mode (ESM) PCF at 1550 nm.
    ///
    /// d/Λ = 0.3 < 0.45 → ESM condition satisfied.
    /// Λ = 5µm, d = 1.5µm.
    pub fn esm_1550() -> Self {
        Self::new(1.444, 1.0, 5e-6, 1.5e-6)
    }

    /// Highly nonlinear PCF (HN-PCF) — small core, large air fraction.
    ///
    /// Λ = 1.5µm, d = 1.2µm, d/Λ = 0.8.
    pub fn highly_nonlinear() -> Self {
        Self::new(1.444, 1.0, 1.5e-6, 1.2e-6)
    }

    /// Large-mode-area (LMA) PCF — large core, low NA.
    ///
    /// Λ = 15µm, d = 3µm, d/Λ = 0.2.
    pub fn large_mode_area() -> Self {
        Self::new(1.444, 1.0, 15e-6, 3.0e-6)
    }

    /// Air-filling fraction of the cladding.
    ///
    ///   f = π/(2√3) · (d/Λ)²  [triangular lattice]
    pub fn air_fill_fraction(&self) -> f64 {
        let ratio = self.hole_diameter / self.pitch;
        PI / (2.0 * 3.0_f64.sqrt()) * ratio * ratio
    }

    /// Effective cladding index via effective-medium approximation.
    ///
    ///   n_eff_clad = √(f·n_hole² + (1-f)·n_silica²)
    pub fn effective_cladding_index(&self) -> f64 {
        let f = self.air_fill_fraction();
        (f * self.n_hole * self.n_hole + (1.0 - f) * self.n_silica * self.n_silica).sqrt()
    }

    /// Numerical aperture (approximate, using effective cladding index).
    pub fn numerical_aperture(&self) -> f64 {
        let n_clad = self.effective_cladding_index();
        if self.n_silica <= n_clad {
            return 0.0;
        }
        (self.n_silica * self.n_silica - n_clad * n_clad).sqrt()
    }

    /// V-number (using effective cladding index).
    pub fn v_number(&self, wavelength: f64) -> f64 {
        2.0 * PI * self.core_radius * self.numerical_aperture() / wavelength
    }

    /// d/Λ ratio (structural parameter).
    pub fn d_over_lambda(&self) -> f64 {
        self.hole_diameter / self.pitch
    }

    /// True if endlessly single-mode condition d/Λ < 0.45 is met.
    pub fn is_endlessly_single_mode(&self) -> bool {
        self.d_over_lambda() < 0.45
    }

    /// Mode field diameter (approximate, MFD ≈ 2·core_radius for small d/Λ).
    ///
    /// More accurate MFD requires full mode solver. Here we use:
    ///   MFD ≈ 2·Λ·(0.65 + 1.619/V^1.5 + 2.879/V^6)  [Marcuse formula]
    pub fn mode_field_diameter(&self, wavelength: f64) -> f64 {
        let v = self.v_number(wavelength).max(0.5);
        2.0 * self.pitch * (0.65 + 1.619 / v.powf(1.5) + 2.879 / v.powi(6))
    }

    /// Effective mode area A_eff ≈ π·(MFD/2)² (m²).
    pub fn effective_mode_area(&self, wavelength: f64) -> f64 {
        let mfd = self.mode_field_diameter(wavelength);
        PI * (mfd / 2.0).powi(2)
    }

    /// Nonlinear coefficient γ = n₂·ω/(c·A_eff) (W⁻¹m⁻¹).
    ///
    /// For fused silica: n₂ = 2.6e-20 m²/W.
    pub fn nonlinear_coefficient(&self, wavelength: f64) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        let n2_silica = 2.6e-20; // m²/W
        let omega = 2.0 * PI * SPEED_OF_LIGHT / wavelength;
        let a_eff = self.effective_mode_area(wavelength);
        n2_silica * omega / (SPEED_OF_LIGHT * a_eff)
    }

    /// Zero-dispersion wavelength (ZDW) — qualitative estimate.
    ///
    /// For standard silica fibers, ZDW ≈ 1270 nm. PCF can shift ZDW
    /// by waveguide dispersion. For HN-PCF, ZDW can be near 800 nm.
    /// This is a rough empirical estimate based on core size.
    pub fn zero_dispersion_wavelength_nm(&self) -> f64 {
        // Approximate: smaller core → shorter ZDW
        let core_um = self.core_radius * 1e6;
        // Empirical fit: ZDW_nm ≈ 1270 - 90*(2.5 - core_um) for core_um in [0.5, 5.0]
        (1270.0 - 90.0 * (2.5 - core_um)).clamp(600.0, 1550.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core defect type
// ─────────────────────────────────────────────────────────────────────────────

/// Type of core defect in the PCF lattice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoreDefect {
    /// One missing hole — standard solid-core PCF.
    SolidCore,
    /// Air core (hollow-core PCF, bandgap-guided).
    HollowCore,
    /// Multiple missing holes — large-mode-area PCF.
    LargeMode,
}

// ─────────────────────────────────────────────────────────────────────────────
// PcfGeometry
// ─────────────────────────────────────────────────────────────────────────────

/// Photonic crystal fiber geometry (triangular lattice).
///
/// Models air holes in a silica background arranged on a triangular lattice.
/// The core is formed by one or more missing holes.
///
/// # References
/// - Knight et al., Opt. Lett. 21, 1547 (1996) — first PCF
/// - Birks et al., Opt. Lett. 22, 961 (1997) — endlessly single-mode
/// - Mortensen et al., Opt. Lett. 28, 393 (2003) — effective index formulas
#[derive(Debug, Clone)]
pub struct PcfGeometry {
    /// Λ — hole-to-hole spacing (µm)
    pub pitch_um: f64,
    /// d — air hole diameter (µm)
    pub hole_diameter_um: f64,
    /// Number of rings of holes surrounding the core
    pub n_rings: u32,
    /// Type of core defect
    pub core_defect: CoreDefect,
    /// Background refractive index (e.g. silica at the operating wavelength)
    pub background_index: f64,
}

impl PcfGeometry {
    /// Create a new PCF geometry with validation.
    ///
    /// # Errors
    /// Returns `OxiPhotonError::NumericalError` if parameters are unphysical.
    pub fn new(
        pitch_um: f64,
        hole_diameter_um: f64,
        n_rings: u32,
        core_defect: CoreDefect,
        background_index: f64,
    ) -> Result<Self, OxiPhotonError> {
        if pitch_um <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "pitch_um must be positive, got {pitch_um}"
            )));
        }
        if hole_diameter_um <= 0.0 || hole_diameter_um >= pitch_um {
            return Err(OxiPhotonError::NumericalError(format!(
                "hole_diameter_um must be in (0, pitch_um), got {hole_diameter_um} vs pitch {pitch_um}"
            )));
        }
        if n_rings == 0 {
            return Err(OxiPhotonError::NumericalError(
                "n_rings must be at least 1".to_string(),
            ));
        }
        if !(1.0..=4.0).contains(&background_index) {
            return Err(OxiPhotonError::NumericalError(format!(
                "background_index must be in [1, 4], got {background_index}"
            )));
        }
        Ok(Self {
            pitch_um,
            hole_diameter_um,
            n_rings,
            core_defect,
            background_index,
        })
    }

    /// Filling fraction of air: f = π·d² / (2√3·Λ²)
    ///
    /// For a triangular lattice unit cell of area √3·Λ²/2,
    /// the air fraction from a single hole of area π(d/2)² is:
    ///   f = π·(d/2)² / (√3·Λ²/2) = π·d² / (2√3·Λ²)
    pub fn fill_fraction(&self) -> f64 {
        let ratio = self.hole_diameter_um / self.pitch_um;
        PI * ratio * ratio / (2.0 * 3.0_f64.sqrt())
    }

    /// Relative hole size d/Λ.
    pub fn d_over_lambda(&self) -> f64 {
        self.hole_diameter_um / self.pitch_um
    }

    /// Effective cladding area (µm²): area spanned by cladding holes.
    ///
    /// Approximated as the area of an annulus from first ring to outer edge.
    pub fn cladding_area_um2(&self) -> f64 {
        let n = self.n_rings as f64;
        // Outer radius ≈ (n_rings + 0.5) * pitch
        let r_outer = (n + 0.5) * self.pitch_um;
        // Inner radius ≈ pitch (first ring)
        let r_inner = self.pitch_um;
        PI * (r_outer * r_outer - r_inner * r_inner)
    }

    /// Core area (µm²): approximately one unit cell of the lattice.
    ///
    /// For a triangular lattice with one missing hole: A_core ≈ √3·Λ²/2.
    /// For a large-mode defect (7 missing holes): A_core ≈ 7·√3·Λ²/2.
    pub fn core_area_um2(&self) -> f64 {
        let unit_cell = 3.0_f64.sqrt() / 2.0 * self.pitch_um * self.pitch_um;
        match self.core_defect {
            CoreDefect::SolidCore => unit_cell,
            CoreDefect::HollowCore => unit_cell,
            CoreDefect::LargeMode => 7.0 * unit_cell,
        }
    }

    /// Approximate mode field diameter (µm).
    ///
    /// For d/Λ → 0 (weak guiding): MFD ≈ 2·Λ.
    /// For large d/Λ (strong guiding): MFD → smaller, approaches core size.
    /// Using a smooth empirical interpolation.
    pub fn approximate_mfd_um(&self) -> f64 {
        let d_lam = self.d_over_lambda();
        // Empirical: MFD ≈ 2*Λ*(1 - 0.6*d/Λ + 0.2*(d/Λ)²)
        // This gives MFD ≈ 2Λ at d/Λ→0 and MFD ≈ 1.2Λ at d/Λ=0.8
        let scale = 1.0 - 0.6 * d_lam + 0.2 * d_lam * d_lam;
        2.0 * self.pitch_um * scale.max(0.3)
    }

    /// Number of air holes in n_rings rings of a triangular lattice.
    ///
    /// Formula: N = 3·n·(n-1)/2·4 + 6·n for ring n (hexagonal shell).
    /// Total for n rings: Σ_{k=1}^{n} 6k = 3·n·(n+1).
    /// For `SolidCore`, subtract 0 (no hole at center by definition).
    pub fn n_holes(&self) -> usize {
        let n = self.n_rings as usize;
        3 * n * (n + 1)
    }

    /// Hole positions in the triangular lattice (µm), returned as (x, y) pairs.
    ///
    /// Triangular lattice basis vectors:
    ///   a₁ = Λ·(1, 0)
    ///   a₂ = Λ·(1/2, √3/2)
    ///
    /// Iterates over rings 1..=n_rings, generating shell by shell.
    pub fn hole_positions(&self) -> Vec<(f64, f64)> {
        let lambda = self.pitch_um;
        let mut positions = Vec::with_capacity(self.n_holes());

        // Triangular lattice: generate integer coordinates then filter by ring
        // Use axial coordinate system: (q, r) with q+r+s=0
        let n = self.n_rings as i32;
        for q in -n..=n {
            for r in (-n).max(-q - n)..=(n).min(-q + n) {
                // Skip origin (core)
                if q == 0 && r == 0 {
                    continue;
                }
                // Convert axial to Cartesian
                let x = lambda * (q as f64 + 0.5 * r as f64);
                let y = lambda * (3.0_f64.sqrt() / 2.0 * r as f64);
                positions.push((x, y));
            }
        }
        positions
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PcfMode — effective index method (Mortensen et al. 2003)
// ─────────────────────────────────────────────────────────────────────────────

/// PCF mode properties using the empirical effective index method.
///
/// Implements the Mortensen et al. (2003) fitting formulas for the effective
/// cladding index and resulting mode parameters.
///
/// # Key formulas
/// - Effective V-number: V_eff = (2π·Λ/λ)·NA_eff
/// - Effective index shift: Δn ∝ (d/Λ)² · F(V_eff)
/// - Endlessly single-mode: d/Λ < 0.45 (Birks criterion)
#[derive(Debug, Clone)]
pub struct PcfMode {
    /// PCF geometry
    pub geometry: PcfGeometry,
    /// Operating wavelength (nm)
    pub wavelength_nm: f64,
}

impl PcfMode {
    /// Create a new PCF mode calculator.
    pub fn new(geometry: PcfGeometry, wavelength_nm: f64) -> Self {
        Self {
            geometry,
            wavelength_nm,
        }
    }

    /// Wavelength in metres.
    fn wavelength_m(&self) -> f64 {
        self.wavelength_nm * 1e-9
    }

    /// Background (core) index — silica at the operating wavelength.
    fn n_core(&self) -> f64 {
        self.geometry.background_index
    }

    /// Effective cladding index via the Mortensen empirical formula.
    ///
    /// Based on the normalized frequency V_pcf, the effective index is:
    ///   n_clad_eff = n_bg · √(1 − (1.5·d/Λ)² · tanh(A · V_pcf^B))
    ///
    /// where A = 0.29 and B = -0.97 are fit coefficients (Mortensen 2003).
    pub fn effective_cladding_index(&self) -> f64 {
        let d_lam = self.geometry.d_over_lambda();
        let n_bg = self.n_core();
        let v_pcf = self.normalized_frequency();

        // Mortensen fitting: correction factor decays with V_pcf
        // For low V_pcf (large wavelength relative to Λ): strong mode expansion
        // For high V_pcf: approaches EMT result
        let a = 0.29_f64;
        let b = -0.97_f64;
        let f_v = if v_pcf > 1e-6 {
            (a * v_pcf.powf(b)).tanh().clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Effective index reduction
        let correction = (1.5 * d_lam) * (1.5 * d_lam) * f_v;
        let n_clad_sq = n_bg * n_bg * (1.0 - correction);
        n_clad_sq.max(1.0).sqrt()
    }

    /// PCF V-number: V = (2π/λ) · Λ · √(n_core² − n_clad²)
    pub fn v_number(&self) -> f64 {
        let lambda_m = self.wavelength_m();
        let pitch_m = self.geometry.pitch_um * 1e-6;
        let n_c = self.n_core();
        let n_cl = self.effective_cladding_index();
        if n_c <= n_cl {
            return 0.0;
        }
        2.0 * PI / lambda_m * pitch_m * (n_c * n_c - n_cl * n_cl).sqrt()
    }

    /// Normalized PCF frequency.
    ///
    ///   V_pcf = (2π·Λ/λ) · √(n_bg² − 1) · (d/Λ)
    ///
    /// This is the structural parameter relating hole size to wavelength.
    pub fn normalized_frequency(&self) -> f64 {
        let lambda_m = self.wavelength_m();
        let pitch_m = self.geometry.pitch_um * 1e-6;
        let n_bg = self.n_core();
        let d_lam = self.geometry.d_over_lambda();
        // n_air = 1.0
        let dn = (n_bg * n_bg - 1.0_f64).max(0.0).sqrt();
        2.0 * PI * pitch_m / lambda_m * dn * d_lam
    }

    /// Effective mode index n_eff (between n_clad_eff and n_core).
    ///
    /// Uses a Gaussian mode approximation:
    ///   n_eff ≈ n_clad + (n_core − n_clad) · (1 − exp(−V²/2))
    ///
    /// This approaches n_clad for V→0 (cut-off) and n_core for V→∞ (ray limit).
    pub fn effective_index(&self) -> f64 {
        let n_c = self.n_core();
        let n_cl = self.effective_cladding_index();
        let v = self.v_number();
        // Fraction of field in core increases with V
        let w = 1.0 - (-v * v / 2.0).exp();
        n_cl + (n_c - n_cl) * w
    }

    /// Whether the PCF is endlessly single-mode.
    ///
    /// Birks criterion: d/Λ < 0.45 ensures single-mode guidance at all wavelengths.
    pub fn is_endlessly_single_mode(&self) -> bool {
        self.geometry.d_over_lambda() < 0.45
    }

    /// Single-mode cutoff d/Λ (Birks criterion, constant ≈ 0.45).
    pub fn single_mode_cutoff_d_over_lambda() -> f64 {
        0.45
    }

    /// Group index n_g = n_eff − λ · dn_eff/dλ.
    ///
    /// Computed via two-point finite difference:
    ///   n_g ≈ n_eff(λ) − λ · \[n_eff(λ+δ) − n_eff(λ−δ)\] / (2δ)
    pub fn group_index(&self) -> f64 {
        let delta_nm = 0.1; // 0.1 nm step
        let lambda = self.wavelength_nm;
        let n_eff_c = self.effective_index();

        let mode_p = PcfMode::new(self.geometry.clone(), lambda + delta_nm);
        let mode_m = PcfMode::new(self.geometry.clone(), lambda - delta_nm);
        let dn_dlam = (mode_p.effective_index() - mode_m.effective_index()) / (2.0 * delta_nm);

        n_eff_c - lambda * dn_dlam
    }

    /// Chromatic dispersion D (ps/nm/km).
    ///
    /// D = −(λ/c) · d²n_eff/dλ²
    ///
    /// Computed via second-order finite difference of n_eff(λ).
    /// Units: ps·nm⁻¹·km⁻¹ (standard telecom convention).
    pub fn chromatic_dispersion_ps_per_nm_km(&self) -> f64 {
        let delta_nm = 1.0; // 1 nm step for stable FD
        let lambda = self.wavelength_nm;

        let mode_p = PcfMode::new(self.geometry.clone(), lambda + delta_nm);
        let mode_m = PcfMode::new(self.geometry.clone(), lambda - delta_nm);
        let n_eff_c = self.effective_index();
        let n_eff_p = mode_p.effective_index();
        let n_eff_m = mode_m.effective_index();

        // d²n/dλ² via central difference (λ in nm, then convert)
        let d2n_dlam2 = (n_eff_p - 2.0 * n_eff_c + n_eff_m) / (delta_nm * delta_nm);

        // D = -(λ/c) * d²n/dλ²,  with λ in m and converting to ps/nm/km
        // λ in nm → λ_m = λ * 1e-9
        // d²n/dλ² in nm⁻² → multiply by 1e18 for m⁻²
        // Result in s/m² → multiply by 1e12 (s→ps) * 1e3 (m→km) / 1e9 (m⁻¹→nm⁻¹)
        //   = 1e6 overall
        // So: D [ps/nm/km] = -(λ_nm * 1e-9 / C0) * d2n_dlam2_nm^{-2} * 1e18 * 1e6
        //                   = -(λ_nm / C0) * d2n_dlam2_nm^{-2} * 1e15
        let c_nm_per_s = C0 * 1e9; // nm/s
        -(lambda / c_nm_per_s) * d2n_dlam2 * 1e15
    }

    /// Effective mode area A_eff (µm²).
    ///
    /// Approximated from a Gaussian field profile:
    ///   w ≈ MFD/2 = Λ·(0.65 + 1.619·V^{−1.5} + 2.879·V^{−6})
    ///   A_eff = π·w²
    pub fn effective_area_um2(&self) -> f64 {
        let pitch_um = self.geometry.pitch_um;
        let v = self.v_number().max(0.5);
        let w_um = pitch_um * (0.65 + 1.619 / v.powf(1.5) + 2.879 / v.powi(6));
        PI * w_um * w_um
    }

    /// Nonlinear coefficient γ = 2π·n₂/(λ·A_eff) (W⁻¹·km⁻¹).
    ///
    /// # Arguments
    /// - `n2_m2_per_w`: nonlinear index n₂ in m²/W (silica: 2.6×10⁻²⁰ m²/W)
    pub fn nonlinear_coefficient_per_w_km(&self, n2_m2_per_w: f64) -> f64 {
        let lambda_m = self.wavelength_m();
        let a_eff_m2 = self.effective_area_um2() * 1e-12; // µm² → m²
                                                          // γ [1/(W·m)] = 2π·n₂ / (λ·A_eff)
        let gamma_per_w_m = 2.0 * PI * n2_m2_per_w / (lambda_m * a_eff_m2);
        gamma_per_w_m * 1e3 // /m → /km
    }

    /// Numerical aperture: NA = √(n_core² − n_clad²).
    pub fn numerical_aperture(&self) -> f64 {
        let n_c = self.n_core();
        let n_cl = self.effective_cladding_index();
        if n_c <= n_cl {
            return 0.0;
        }
        (n_c * n_c - n_cl * n_cl).sqrt()
    }

    /// Maximum acceptance half-angle (rad).
    ///
    ///   θ_max = arcsin(NA)
    pub fn acceptance_angle_rad(&self) -> f64 {
        self.numerical_aperture().min(1.0).asin()
    }

    /// Confinement loss (dB/m).
    ///
    /// For a finite-cladding PCF with N_rings rings, the confinement loss
    /// scales exponentially with the number of rings:
    ///   CL ≈ α₀ · exp(−β · N_rings · d/Λ)
    ///
    /// where α₀ ≈ 10 dB/m and β ≈ 3.0 are empirical constants derived
    /// from full-vectorial computations (Kuhlmey et al. 2002).
    ///
    /// The loss also depends on the V-number: higher V → lower leakage.
    pub fn confinement_loss_db_per_m(&self) -> f64 {
        let n_rings = self.geometry.n_rings as f64;
        let d_lam = self.geometry.d_over_lambda();
        let v = self.v_number().max(0.5);

        // Base loss with exponential decay per ring
        let alpha0 = 10.0_f64; // dB/m at 1 ring
        let beta = 3.0_f64;
        let v_factor = (-0.3 * (v - 1.0).max(0.0)).exp(); // V-number penalty for low V
        alpha0 * (-beta * n_rings * d_lam).exp() * v_factor
    }

    /// Bend loss (dB/m) at bend radius R (mm).
    ///
    /// Uses the Petermann formula adapted for PCF:
    ///   CL_bend ≈ (C / R_mm) · exp(−D · R_mm · NA² / λ)
    ///
    /// where C and D are structure-dependent constants.
    /// Reference: Petermann, Electron. Lett. 1977; adapted for PCF by
    /// Lægsgaard & Bjarklev 2003.
    pub fn bend_loss_db_per_m(&self, bend_radius_mm: f64) -> f64 {
        if bend_radius_mm <= 0.0 {
            return f64::INFINITY;
        }
        let na = self.numerical_aperture();
        let lambda_um = self.wavelength_nm * 1e-3; // nm → µm
        let pitch_um = self.geometry.pitch_um;

        // Petermann-type formula:
        // α_bend [dB/m] = (C / R) * exp(-D * R * NA² / λ_µm)
        // C ≈ 4.343 (dB conversion factor from e-folding)
        // D: effective reciprocal penetration depth
        let d_coeff = PI * na * na * pitch_um / lambda_um;
        4.343 / bend_radius_mm * (-d_coeff * bend_radius_mm).exp()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BirefringentPcf
// ─────────────────────────────────────────────────────────────────────────────

/// Birefringent PCF — elliptical holes or asymmetric lattice.
///
/// Birefringence arises from:
/// 1. **Geometric**: elliptical holes break cylindrical symmetry → form birefringence.
/// 2. **Stress**: stress-applying parts (SAPs) with different thermal expansion → stress birefringence.
///
/// The total birefringence B = |n_slow − n_fast|.
///
/// # Reference
/// Ortigosa-Blanch et al., Opt. Lett. 25, 1325 (2000).
#[derive(Debug, Clone)]
pub struct BirefringentPcf {
    /// Base PCF geometry (circular holes in absence of ellipticity)
    pub base_geometry: PcfGeometry,
    /// b/a for elliptical holes (1.0 = circular, >1 = elongated along y)
    pub hole_aspect_ratio: f64,
    /// Index difference due to stress rods (0 if no SAPs)
    pub stress_rod_dn: f64,
}

impl BirefringentPcf {
    /// Create a birefringent PCF.
    ///
    /// # Errors
    /// Returns error if aspect ratio ≤ 0 or base geometry is invalid.
    pub fn new(
        base_geometry: PcfGeometry,
        hole_aspect_ratio: f64,
        stress_rod_dn: f64,
    ) -> Result<Self, OxiPhotonError> {
        if hole_aspect_ratio <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "hole_aspect_ratio must be positive, got {hole_aspect_ratio}"
            )));
        }
        if stress_rod_dn < 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "stress_rod_dn must be non-negative, got {stress_rod_dn}"
            )));
        }
        Ok(Self {
            base_geometry,
            hole_aspect_ratio,
            stress_rod_dn,
        })
    }

    /// Modal birefringence B = |n_slow − n_fast|.
    ///
    /// Form birefringence from elliptical holes (semi-axes a, b = a·aspect_ratio):
    ///   B_form ≈ C_form · (b/a − 1) · (d/Λ)² · n_bg · (NA/n_bg)²
    ///
    /// Stress birefringence from SAPs:
    ///   B_stress = C_stress · stress_rod_dn
    ///
    /// where the coefficients are empirically calibrated from vectorial FEM.
    pub fn birefringence(&self, lambda_nm: f64) -> f64 {
        let d_lam = self.base_geometry.d_over_lambda();
        let n_bg = self.base_geometry.background_index;
        // Mode using PcfMode approximation
        let mode = PcfMode::new(self.base_geometry.clone(), lambda_nm);
        let na = mode.numerical_aperture();

        // Form birefringence coefficient (empirical, based on Ortigosa-Blanch 2000)
        let c_form = 0.15_f64;
        let b_form = c_form
            * (self.hole_aspect_ratio - 1.0).abs()
            * d_lam
            * d_lam
            * n_bg
            * (na / n_bg).powi(2);

        // Stress birefringence
        let c_stress = 0.5_f64;
        let b_stress = c_stress * self.stress_rod_dn;

        b_form + b_stress
    }

    /// Beat length L_B = λ/B (mm).
    ///
    /// Over this distance the relative phase between x- and y-polarized modes
    /// accumulates 2π.
    pub fn beat_length_mm(&self, lambda_nm: f64) -> f64 {
        let b = self.birefringence(lambda_nm);
        if b < 1e-15 {
            return f64::INFINITY;
        }
        let lambda_mm = lambda_nm * 1e-6; // nm → mm
        lambda_mm / b
    }

    /// Is the fiber polarization-maintaining?
    ///
    /// Strong PM condition: B > 5×10⁻⁴ (typical panda/bowtie PM fiber threshold).
    pub fn is_polarization_maintaining(&self) -> bool {
        // Use a representative telecom wavelength if no specific one given
        self.birefringence(1550.0) > 5e-4
    }

    /// Polarization crosstalk after length L (dB).
    ///
    /// In the presence of perturbations with correlation length h,
    /// crosstalk power grows as:
    ///   X(L) = 10·log10(h·L / L_B²)  \[simplified linear-coupling model\]
    ///
    /// Here we use the H-parameter formulation:
    ///   X \[dB\] = 10·log10(H·L)
    pub fn polarization_crosstalk_db(&self, length_m: f64, lambda_nm: f64) -> f64 {
        let h = self.h_parameter(lambda_nm);
        if h <= 0.0 || length_m <= 0.0 {
            return f64::NEG_INFINITY;
        }
        10.0 * (h * length_m).log10()
    }

    /// H-parameter (polarization crosstalk coefficient per unit length, m⁻¹).
    ///
    /// Models random coupling due to fiber imperfections:
    ///   H ≈ (λ/L_B)² · σ_coupling
    ///
    /// where σ_coupling ≈ 1e-4 m⁻¹ is a typical perturbation strength
    /// (curvature, stress non-uniformity). L_B enters squared so that
    /// higher birefringence → lower H → better polarization extinction.
    pub fn h_parameter(&self, lambda_nm: f64) -> f64 {
        let l_b_m = self.beat_length_mm(lambda_nm) * 1e-3; // mm → m
        if l_b_m.is_infinite() || l_b_m <= 0.0 {
            return 0.0;
        }
        // σ_coupling: typical value for a well-made PM fiber ~ 1e-4 m⁻¹·(m/L_B)²
        let sigma = 1e-4_f64; // m⁻¹ at reference L_B = 1 m
        let lambda_m = lambda_nm * 1e-9;
        sigma * (lambda_m / l_b_m).powi(2)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HollowCorePcf
// ─────────────────────────────────────────────────────────────────────────────

/// Hollow-core photonic crystal fiber (HC-PCF).
///
/// Guides light by photonic bandgap effect rather than total internal reflection.
/// Key properties:
/// - Ultra-low nonlinearity (air core: n₂ ≈ 3×10⁻²³ m²/W)
/// - Group velocity close to c (latency advantage over SMF)
/// - Narrow transmission window (bandgap width ≈ 15–20%)
///
/// # Reference
/// Cregan et al., Science 285, 1537 (1999).
/// Roberts et al., Opt. Express 13, 236 (2005).
#[derive(Debug, Clone)]
pub struct HollowCorePcf {
    /// Lattice pitch Λ (µm)
    pub pitch_um: f64,
    /// Hollow core radius r_c (µm)
    pub core_radius_um: f64,
    /// Cladding hole diameter d (µm)
    pub cladding_hole_diameter_um: f64,
    /// Silica refractive index at operating wavelength
    pub silica_index: f64,
    /// Number of rings of cladding holes
    pub n_rings: u32,
}

impl HollowCorePcf {
    /// Create a hollow-core PCF.
    ///
    /// # Errors
    /// Returns error if parameters are unphysical.
    pub fn new(
        pitch_um: f64,
        core_radius_um: f64,
        cladding_hole_diameter_um: f64,
        silica_index: f64,
        n_rings: u32,
    ) -> Result<Self, OxiPhotonError> {
        if pitch_um <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "pitch_um must be positive, got {pitch_um}"
            )));
        }
        if core_radius_um <= 0.0 || core_radius_um > 5.0 * pitch_um {
            return Err(OxiPhotonError::NumericalError(format!(
                "core_radius_um must be in (0, 5*pitch), got {core_radius_um}"
            )));
        }
        if cladding_hole_diameter_um <= 0.0 || cladding_hole_diameter_um >= pitch_um {
            return Err(OxiPhotonError::NumericalError(format!(
                "cladding_hole_diameter_um must be in (0, pitch), got {cladding_hole_diameter_um}"
            )));
        }
        if silica_index < 1.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "silica_index must be ≥ 1.0, got {silica_index}"
            )));
        }
        if n_rings == 0 {
            return Err(OxiPhotonError::NumericalError(
                "n_rings must be ≥ 1".to_string(),
            ));
        }
        Ok(Self {
            pitch_um,
            core_radius_um,
            cladding_hole_diameter_um,
            silica_index,
            n_rings,
        })
    }

    /// Effective silica web thickness (µm): t = Λ − d.
    fn web_thickness_um(&self) -> f64 {
        self.pitch_um - self.cladding_hole_diameter_um
    }

    /// Photonic bandgap center wavelength (nm).
    ///
    /// The bandgap arises from anti-resonance of the silica struts.
    /// Modified anti-resonant reflecting optical waveguide (ARROW) model:
    ///   λ_c = 2·t·√(n_silica² − 1) / m   for m = 1 (first bandgap)
    ///
    /// where t is the silica web thickness.
    ///
    /// Reference: Litchinitser et al., Opt. Lett. 27, 1592 (2002).
    pub fn bandgap_center_wavelength_nm(&self) -> f64 {
        let t_um = self.web_thickness_um();
        let dn = (self.silica_index * self.silica_index - 1.0_f64).sqrt();
        // First anti-resonance order m=1: λ = 2*t*dn
        // Convert µm → nm
        2.0 * t_um * dn * 1e3
    }

    /// Bandgap width (nm).
    ///
    /// Approximate: Δλ/λ_c ≈ 0.18 for typical HC-PCF with d/Λ ≈ 0.9.
    /// Slightly narrower for tighter lattice.
    pub fn bandgap_width_nm(&self) -> f64 {
        let d_lam = self.cladding_hole_diameter_um / self.pitch_um;
        // Empirical: width fraction ≈ 0.12 + 0.08*(d/Λ - 0.8) for d/Λ near 0.9
        let width_fraction = (0.12 + 0.08 * (d_lam - 0.8)).clamp(0.10, 0.25);
        self.bandgap_center_wavelength_nm() * width_fraction
    }

    /// Transmission window \[λ_min, λ_max\] (nm).
    pub fn transmission_window(&self) -> (f64, f64) {
        let center = self.bandgap_center_wavelength_nm();
        let half_width = self.bandgap_width_nm() / 2.0;
        (center - half_width, center + half_width)
    }

    /// Air fill fraction in the core region.
    ///
    /// For a hollow core, the air fraction is very high (> 0.99 ideally).
    /// Approximated as ratio of core area to total cell area.
    pub fn air_fill_fraction(&self) -> f64 {
        let r_c = self.core_radius_um;
        let pitch = self.pitch_um;
        // Core cell area ≈ 7 unit cells for standard HC-PCF
        let core_area = PI * r_c * r_c;
        let cell_area = 7.0 * (3.0_f64.sqrt() / 2.0 * pitch * pitch);
        (core_area / cell_area).min(0.9999)
    }

    /// Group velocity as fraction of c: v_g/c.
    ///
    /// For an air-guided mode in HC-PCF, v_g/c ≈ 1 − δ,
    /// where δ accounts for the small field overlap with silica.
    ///
    /// Using the perturbation-theory result:
    ///   v_g/c ≈ 1 − η_silica · (n_silica² − 1) / 2
    ///
    /// where η_silica is the fractional power in silica (≈ 0.005–0.01).
    pub fn group_velocity_fraction(&self) -> f64 {
        let eta_silica = 1.0 - self.air_fill_fraction(); // fraction in glass
        let n_sil = self.silica_index;
        let delta = eta_silica * (n_sil * n_sil - 1.0) / 2.0;
        (1.0 - delta).max(0.99) // v_g/c very close to 1
    }

    /// Latency advantage over standard SMF (ns/km).
    ///
    ///   Δτ = (1/v_g − 1/c) · L
    ///
    /// For SMF: v_g/c ≈ 1/n_g_smf ≈ 1/1.4677 at 1550 nm.
    /// For HC-PCF: v_g/c ≈ 0.9995.
    ///
    /// Δτ \[ns/km\] = (n_g_smf/c − 1/v_g) · 1e12 · 1e3 / 1e9
    ///            = (n_g_smf − c/v_g) · 1e3/c \[ns/km\]
    pub fn latency_advantage_ns_per_km(&self) -> f64 {
        let n_g_smf = 1.4677_f64; // SMF group index at 1550 nm
        let v_g_frac = self.group_velocity_fraction(); // v_g/c
                                                       // SMF propagation delay per km: n_g_smf / c * 1e3 m = n_g_smf * 1e3 / c s
                                                       // HC-PCF propagation delay per km: 1 / (v_g_frac * c) * 1e3 m
                                                       // Advantage = (1/v_g_hcpcf - 1/v_g_smf) ... wait sign convention:
                                                       // HC-PCF is FASTER, so advantage = SMF delay - HC delay
                                                       // Delay HC [ns/km] = (1e3 / (v_g_frac * C0)) * 1e9
                                                       // Delay SMF [ns/km] = (1e3 * n_g_smf / C0) * 1e9
        let delay_smf_ns_per_km = n_g_smf / C0 * 1e3 * 1e9;
        let delay_hcpcf_ns_per_km = 1.0 / (v_g_frac * C0) * 1e3 * 1e9;
        (delay_smf_ns_per_km - delay_hcpcf_ns_per_km).max(0.0)
    }

    /// Nonlinear coefficient γ (W⁻¹·km⁻¹).
    ///
    /// For air-guided mode: n₂_eff ≈ η_silica · n₂_silica + η_air · n₂_air
    /// n₂_air ≈ 3×10⁻²³ m²/W (negligible).
    /// Effective mode area for HC-PCF: A_eff ≈ π·r_core².
    /// Result: γ << 1 W⁻¹km⁻¹ (typically 0.001–0.01 for HC-PCF).
    pub fn nonlinear_coefficient_per_w_km(&self) -> f64 {
        let n2_silica = 2.6e-20_f64; // m²/W
        let n2_air = 3e-23_f64; // m²/W
        let eta_sil = 1.0 - self.air_fill_fraction();
        let eta_air = self.air_fill_fraction();
        let n2_eff = eta_sil * n2_silica + eta_air * n2_air;

        // Mode area ≈ π*r_core² in m²
        let a_eff_m2 = PI * (self.core_radius_um * 1e-6).powi(2);
        // Use center wavelength for ω
        let lambda_m = self.bandgap_center_wavelength_nm() * 1e-9;
        let omega = 2.0 * PI * C0 / lambda_m;

        // γ [1/(W·m)] = n₂·ω/(c·A_eff)
        let gamma_per_w_m = n2_eff * omega / (C0 * a_eff_m2);
        gamma_per_w_m * 1e3 // /m → /km
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PcfOptimizer
// ─────────────────────────────────────────────────────────────────────────────

/// PCF parameter optimizer: find pitch and d/Λ for target mode properties.
///
/// Uses a grid search over (Λ, d/Λ) parameter space with a weighted
/// figure-of-merit combining deviations from target MFD and/or dispersion.
///
/// # Example targets
/// - Large-mode-area: target MFD ≈ 20 µm at 1064 nm
/// - Dispersion-flattened: target D ≈ 0 ps/nm/km at 1310 nm
/// - Highly nonlinear: target MFD ≈ 2 µm at 800 nm
#[derive(Debug, Clone)]
pub struct PcfOptimizer {
    /// Target mode field diameter (µm), if any
    pub target_mfd_um: Option<f64>,
    /// Target chromatic dispersion (ps/nm/km), if any
    pub target_dispersion_ps_per_nm_km: Option<f64>,
    /// Operating wavelength (nm)
    pub target_wavelength_nm: f64,
    /// Maximum allowed d/Λ (0.45 for SM constraint, 1.0 for no constraint)
    pub max_d_over_lambda: f64,
    /// Background (silica) refractive index
    pub background_index: f64,
}

impl PcfOptimizer {
    /// Create a new optimizer for the given wavelength and material.
    pub fn new(target_wavelength_nm: f64, background_index: f64) -> Self {
        Self {
            target_mfd_um: None,
            target_dispersion_ps_per_nm_km: None,
            target_wavelength_nm,
            max_d_over_lambda: 1.0,
            background_index,
        }
    }

    /// Set a target mode field diameter (µm).
    pub fn with_target_mfd(mut self, mfd_um: f64) -> Self {
        self.target_mfd_um = Some(mfd_um);
        self
    }

    /// Set a target chromatic dispersion (ps/nm/km).
    pub fn with_target_dispersion(mut self, d_ps_per_nm_km: f64) -> Self {
        self.target_dispersion_ps_per_nm_km = Some(d_ps_per_nm_km);
        self
    }

    /// Apply the endlessly-single-mode constraint (d/Λ ≤ 0.45).
    pub fn with_single_mode_constraint(mut self) -> Self {
        self.max_d_over_lambda = 0.45;
        self
    }

    /// Figure of merit (lower = better).
    ///
    /// Weighted sum of squared relative deviations from targets.
    pub fn figure_of_merit(&self, geom: &PcfGeometry) -> f64 {
        let mode = PcfMode::new(geom.clone(), self.target_wavelength_nm);
        let mut fom = 0.0_f64;
        let mut n_terms = 0_u32;

        if let Some(target_mfd) = self.target_mfd_um {
            let mfd = mode.effective_area_um2().sqrt() * (PI / 4.0).sqrt() * 2.0;
            let rel_err = (mfd - target_mfd) / target_mfd;
            fom += rel_err * rel_err;
            n_terms += 1;
        }

        if let Some(target_disp) = self.target_dispersion_ps_per_nm_km {
            let d = mode.chromatic_dispersion_ps_per_nm_km();
            // Normalise by a reference dispersion of 20 ps/nm/km
            let ref_disp = 20.0_f64;
            let abs_err = (d - target_disp) / ref_disp;
            fom += abs_err * abs_err;
            n_terms += 1;
        }

        if n_terms == 0 {
            return 0.0;
        }
        fom / n_terms as f64
    }

    /// Grid search over (pitch_um, d/Λ) parameter space.
    ///
    /// Searches:
    /// - pitch: 1.0 µm … 20.0 µm in 40 steps
    /// - d/Λ: 0.05 … max_d_over_lambda in 40 steps
    ///
    /// # Errors
    /// Returns error if no valid geometry is found.
    pub fn optimize(&self) -> Result<PcfGeometry, OxiPhotonError> {
        if self.target_mfd_um.is_none() && self.target_dispersion_ps_per_nm_km.is_none() {
            return Err(OxiPhotonError::NumericalError(
                "At least one optimization target must be set".to_string(),
            ));
        }

        let n_pitch = 40_usize;
        let n_d = 40_usize;
        let pitch_min = 1.0_f64;
        let pitch_max = 20.0_f64;
        let d_min = 0.05_f64;
        let d_max = self.max_d_over_lambda;

        let mut best_fom = f64::INFINITY;
        let mut best_geom: Option<PcfGeometry> = None;

        for i_pitch in 0..n_pitch {
            let pitch = pitch_min + (pitch_max - pitch_min) * i_pitch as f64 / (n_pitch - 1) as f64;

            for i_d in 0..n_d {
                let d_lam = d_min + (d_max - d_min) * i_d as f64 / (n_d - 1) as f64;

                let hole_d = pitch * d_lam;
                let geom = match PcfGeometry::new(
                    pitch,
                    hole_d,
                    3,
                    CoreDefect::SolidCore,
                    self.background_index,
                ) {
                    Ok(g) => g,
                    Err(_) => continue,
                };

                let fom = self.figure_of_merit(&geom);
                if fom < best_fom {
                    best_fom = fom;
                    best_geom = Some(geom);
                }
            }
        }

        best_geom.ok_or_else(|| {
            OxiPhotonError::NumericalError("Optimizer found no valid geometry".to_string())
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Legacy PhotonicCrystalFiber tests (preserved) ──────────────────────

    #[test]
    fn pcf_esm_is_single_mode() {
        let pcf = PhotonicCrystalFiber::esm_1550();
        assert!(
            pcf.is_endlessly_single_mode(),
            "d/Λ={:.2}",
            pcf.d_over_lambda()
        );
    }

    #[test]
    fn pcf_hn_not_single_mode() {
        let pcf = PhotonicCrystalFiber::highly_nonlinear();
        assert!(!pcf.is_endlessly_single_mode());
    }

    #[test]
    fn pcf_air_fill_fraction_range() {
        let pcf = PhotonicCrystalFiber::esm_1550();
        let f = pcf.air_fill_fraction();
        assert!(f > 0.0 && f < 1.0, "f={f:.3}");
    }

    #[test]
    fn pcf_effective_cladding_below_silica() {
        let pcf = PhotonicCrystalFiber::esm_1550();
        assert!(pcf.effective_cladding_index() < pcf.n_silica);
    }

    #[test]
    fn pcf_na_positive() {
        let pcf = PhotonicCrystalFiber::esm_1550();
        assert!(pcf.numerical_aperture() > 0.0);
    }

    #[test]
    fn pcf_mfd_reasonable() {
        let pcf = PhotonicCrystalFiber::esm_1550();
        let mfd = pcf.mode_field_diameter(1550e-9) * 1e6;
        assert!(mfd > 1.0 && mfd < 30.0, "MFD={mfd:.1}µm");
    }

    #[test]
    fn pcf_hn_large_gamma() {
        let pcf_hn = PhotonicCrystalFiber::highly_nonlinear();
        let pcf_lma = PhotonicCrystalFiber::large_mode_area();
        let g_hn = pcf_hn.nonlinear_coefficient(1550e-9);
        let g_lma = pcf_lma.nonlinear_coefficient(1550e-9);
        assert!(g_hn > g_lma, "HN-PCF should have larger γ");
    }

    #[test]
    fn pcf_a_eff_positive() {
        let pcf = PhotonicCrystalFiber::esm_1550();
        let a_eff = pcf.effective_mode_area(1550e-9);
        assert!(a_eff > 0.0);
    }

    // ── New PcfGeometry tests ──────────────────────────────────────────────

    #[test]
    fn test_pcf_fill_fraction() {
        // f = π*d²/(2√3*Λ²); for d=1.5, Λ=5: f = π*2.25/(2*1.732*25)
        let geom = PcfGeometry::new(5.0, 1.5, 3, CoreDefect::SolidCore, 1.444).unwrap();
        let f_expected = PI * 1.5 * 1.5 / (2.0 * 3.0_f64.sqrt() * 25.0);
        let f_actual = geom.fill_fraction();
        assert!(
            (f_actual - f_expected).abs() < 1e-10,
            "f={f_actual:.6} expected {f_expected:.6}"
        );
        assert!(
            f_actual > 0.0 && f_actual < 1.0,
            "fill fraction out of range: {f_actual}"
        );
    }

    #[test]
    fn test_pcf_d_over_lambda_constraint() {
        // d/Λ must be in (0, 1) for valid PCF
        let geom = PcfGeometry::new(5.0, 1.5, 3, CoreDefect::SolidCore, 1.444).unwrap();
        let d_lam = geom.d_over_lambda();
        assert!(d_lam > 0.0 && d_lam < 1.0, "d/Λ={d_lam:.3}");
    }

    #[test]
    fn test_endlessly_single_mode() {
        // d/Λ = 0.3 < 0.45 → ESM = true
        let geom = PcfGeometry::new(5.0, 1.5, 3, CoreDefect::SolidCore, 1.444).unwrap();
        assert!((geom.d_over_lambda() - 0.3).abs() < 1e-10);
        let mode = PcfMode::new(geom, 1550.0);
        assert!(mode.is_endlessly_single_mode());
    }

    #[test]
    fn test_not_single_mode() {
        // d/Λ = 0.6 > 0.45 → ESM = false
        let geom = PcfGeometry::new(5.0, 3.0, 3, CoreDefect::SolidCore, 1.444).unwrap();
        assert!((geom.d_over_lambda() - 0.6).abs() < 1e-10);
        let mode = PcfMode::new(geom, 1550.0);
        assert!(!mode.is_endlessly_single_mode());
    }

    #[test]
    fn test_v_number_positive() {
        let geom = PcfGeometry::new(5.0, 1.5, 3, CoreDefect::SolidCore, 1.444).unwrap();
        let mode = PcfMode::new(geom, 1550.0);
        let v = mode.v_number();
        assert!(v > 0.0, "V-number should be positive, got {v}");
    }

    #[test]
    fn test_effective_index_between_air_and_silica() {
        // n_air (=1.0) < n_eff < n_silica (≈1.444)
        let geom = PcfGeometry::new(5.0, 1.5, 3, CoreDefect::SolidCore, 1.444).unwrap();
        let mode = PcfMode::new(geom, 1550.0);
        let n_eff = mode.effective_index();
        assert!(n_eff > 1.0, "n_eff={n_eff:.5} should be above air index");
        assert!(
            n_eff < 1.444,
            "n_eff={n_eff:.5} should be below silica index"
        );
    }

    #[test]
    fn test_nonlinear_coeff() {
        // Large MFD PCF → small γ
        let geom = PcfGeometry::new(15.0, 3.0, 3, CoreDefect::LargeMode, 1.444).unwrap();
        let mode = PcfMode::new(geom, 1550.0);
        let n2_silica = 2.6e-20_f64; // m²/W
        let gamma = mode.nonlinear_coefficient_per_w_km(n2_silica);
        // For large-mode-area PCF, γ << 1 W⁻¹km⁻¹
        assert!(gamma > 0.0, "γ must be positive");
        assert!(gamma < 10.0, "LMA-PCF γ={gamma:.3} W⁻¹km⁻¹ should be small");
    }

    #[test]
    fn test_confinement_loss_decreases_with_rings() {
        let geom3 = PcfGeometry::new(5.0, 1.5, 3, CoreDefect::SolidCore, 1.444).unwrap();
        let geom6 = PcfGeometry::new(5.0, 1.5, 6, CoreDefect::SolidCore, 1.444).unwrap();
        let mode3 = PcfMode::new(geom3, 1550.0);
        let mode6 = PcfMode::new(geom6, 1550.0);
        let cl3 = mode3.confinement_loss_db_per_m();
        let cl6 = mode6.confinement_loss_db_per_m();
        assert!(
            cl6 < cl3,
            "More rings → lower CL: CL(3)={cl3:.4} CL(6)={cl6:.4}"
        );
    }

    #[test]
    fn test_birefringent_beat_length() {
        // L_B = λ/B; check consistent with birefringence formula
        let geom = PcfGeometry::new(5.0, 1.5, 3, CoreDefect::SolidCore, 1.444).unwrap();
        let bpcf = BirefringentPcf::new(geom, 2.0, 0.0).unwrap();
        let lambda_nm = 1550.0;
        let b = bpcf.birefringence(lambda_nm);
        let l_b = bpcf.beat_length_mm(lambda_nm);
        if b > 1e-15 {
            let expected_l_b_mm = lambda_nm * 1e-6 / b;
            assert!(
                (l_b - expected_l_b_mm).abs() / expected_l_b_mm < 1e-6,
                "L_B mismatch: got {l_b:.4} mm, expected {expected_l_b_mm:.4} mm"
            );
        }
    }

    #[test]
    fn test_birefringent_pm_condition() {
        // Strong aspect ratio → high B → PM fiber
        let geom = PcfGeometry::new(5.0, 1.5, 3, CoreDefect::SolidCore, 1.444).unwrap();
        // Very high aspect ratio AND large stress rod Δn to ensure B > 5e-4
        let bpcf = BirefringentPcf::new(geom.clone(), 5.0, 1e-3).unwrap();
        assert!(
            bpcf.is_polarization_maintaining(),
            "B={:.2e} should be > 5e-4",
            bpcf.birefringence(1550.0)
        );
        // Near-circular holes, no stress → not PM
        let bpcf_low = BirefringentPcf::new(geom, 1.0001, 0.0).unwrap();
        assert!(!bpcf_low.is_polarization_maintaining());
    }

    #[test]
    fn test_hollow_core_bandgap_center() {
        // HC-PCF: pitch=3.8, core_r=7.5 um, hole_d=3.3, silica=1.444
        // ARROW: λ_c = 2*t*√(n²-1)*1e3 nm where t = pitch-d in µm
        let hc = HollowCorePcf::new(3.8, 7.5, 3.3, 1.444, 5).unwrap();
        let lc = hc.bandgap_center_wavelength_nm();
        // t = 0.5 µm, dn=√(1.444²-1)=1.0507 → λ_c ≈ 2*0.5*1.0507*1000 = 1050 nm
        assert!(
            lc > 200.0 && lc < 5000.0,
            "bandgap center λ={lc:.1} nm out of range"
        );
    }

    #[test]
    fn test_hollow_core_low_nonlinearity() {
        let hc = HollowCorePcf::new(3.8, 7.5, 3.3, 1.444, 5).unwrap();
        let gamma = hc.nonlinear_coefficient_per_w_km();
        // HC-PCF: γ << 1 W⁻¹km⁻¹ (typically 0.001 – 0.01)
        assert!(gamma < 0.1, "HC-PCF γ={gamma:.4} W⁻¹km⁻¹ should be << 1");
        assert!(gamma > 0.0, "γ must be positive");
    }

    #[test]
    fn test_hole_positions_count() {
        // 1 ring: 6 holes; 2 rings: 18 holes; 3 rings: 36 holes
        for n_rings in 1..=5_u32 {
            let geom = PcfGeometry::new(5.0, 1.5, n_rings, CoreDefect::SolidCore, 1.444).unwrap();
            let positions = geom.hole_positions();
            let expected = 3 * n_rings as usize * (n_rings as usize + 1);
            assert_eq!(
                positions.len(),
                expected,
                "n_rings={n_rings}: got {} holes, expected {}",
                positions.len(),
                expected
            );
        }
    }

    #[test]
    fn test_pcf_numerical_aperture_positive() {
        let geom = PcfGeometry::new(5.0, 1.5, 3, CoreDefect::SolidCore, 1.444).unwrap();
        let mode = PcfMode::new(geom, 1550.0);
        let na = mode.numerical_aperture();
        assert!(na > 0.0, "NA={na:.4} should be positive");
    }

    #[test]
    fn test_pcf_chromatic_dispersion() {
        // Just check it returns a finite f64
        let geom = PcfGeometry::new(5.0, 1.5, 3, CoreDefect::SolidCore, 1.444).unwrap();
        let mode = PcfMode::new(geom, 1550.0);
        let d = mode.chromatic_dispersion_ps_per_nm_km();
        assert!(
            d.is_finite(),
            "chromatic dispersion should be finite, got {d}"
        );
    }

    // ── Additional optimizer test ──────────────────────────────────────────

    #[test]
    fn test_pcf_optimizer_mfd_target() {
        let opt = PcfOptimizer::new(1550.0, 1.444)
            .with_target_mfd(10.0)
            .with_single_mode_constraint();
        let result = opt.optimize();
        assert!(result.is_ok(), "Optimizer failed: {:?}", result.err());
        let geom = result.unwrap();
        assert!(
            geom.d_over_lambda() <= 0.45 + 1e-9,
            "SM constraint violated"
        );
    }
}
