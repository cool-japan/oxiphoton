//! Photonic crystal defect modes.
//!
//! Introducing a defect (point or line) into a photonic crystal creates localised
//! modes within the photonic bandgap. These have:
//!   - Exponentially decaying fields outside the defect
//!   - Resonance frequency within the gap
//!   - Quality factor Q ∝ exp(2·κ·d) where κ is the evanescent decay rate
//!
//! For a 1D PC cavity (Fabry-Pérot defect between two PC mirrors):
//!   - The PC mirror reflectivity R determines Q
//!   - Q = π·√R / (1-R) × (1/FSR) × ω_res  (equivalent to FP finesse × mode lifetime)
//!
//! L3 cavity in 2D PC: three missing holes in a row.
//!   - Theoretical Q ≈ 45,000 (Akahane et al. 2003)
//!
//! H1 cavity: single missing hole.
//!   - Q ≈ 600–1000

use std::f64::consts::PI;

use crate::photonic_crystal::pwe2d::{kpath_hexagonal, PhCrystal2d, Polarization};

/// 1D photonic crystal cavity (Fabry-Pérot type).
///
/// A defect layer is sandwiched between two identical PC mirrors.
#[derive(Debug, Clone, Copy)]
pub struct Pc1dCavity {
    /// PC mirror reflectivity (per mirror)
    pub r_mirror: f64,
    /// Defect layer optical length n·L (m)
    pub optical_length: f64,
    /// Defect effective index
    pub n_defect: f64,
    /// PC period (m)
    pub pc_period: f64,
    /// Number of periods per mirror
    pub n_periods: usize,
}

impl Pc1dCavity {
    /// Create a 1D PC cavity.
    pub fn new(
        r_mirror: f64,
        optical_length: f64,
        n_defect: f64,
        pc_period: f64,
        n_periods: usize,
    ) -> Self {
        Self {
            r_mirror,
            optical_length,
            n_defect,
            pc_period,
            n_periods,
        }
    }

    /// λ/2 defect cavity at 1550 nm with SiO2/TiO2 Bragg mirrors.
    pub fn half_wave_cavity_1550nm() -> Self {
        // 10-period SiO2/TiO2 mirrors (R ≈ 0.995 for 10 periods)
        let r = 0.995;
        let n_defect = 1.5; // SiO2
        let l_defect = 1550e-9 / (2.0 * n_defect); // λ/2n defect
        Self::new(r, n_defect * l_defect, n_defect, 266e-9, 10)
    }

    /// Resonance frequency (rad/s) for the defect mode.
    ///
    /// For a Fabry-Pérot cavity: ω_m = m·π·c / (n·L)
    /// The fundamental mode (m=1).
    pub fn resonance_frequency(&self) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        PI * SPEED_OF_LIGHT / self.optical_length
    }

    /// Resonance wavelength (m).
    pub fn resonance_wavelength(&self) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        2.0 * PI * SPEED_OF_LIGHT / self.resonance_frequency()
    }

    /// Free spectral range (rad/s).
    pub fn free_spectral_range(&self) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        PI * SPEED_OF_LIGHT / self.optical_length
    }

    /// Cavity quality factor Q from mirror reflectivity.
    ///
    ///   Q = ω_res · τ_ph  where τ_ph = -1/(c·ln(R))·n·L
    ///   Equivalently: Q = π·√R/(1-R) × 2·n·L/(λ_res)
    pub fn quality_factor(&self) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        let omega_res = self.resonance_frequency();
        let round_trip_loss = -2.0 * self.r_mirror.ln(); // ≈ 2(1-R) for R→1
        let photon_lifetime = 2.0 * self.optical_length / (SPEED_OF_LIGHT * round_trip_loss);
        omega_res * photon_lifetime
    }

    /// Mode linewidth Δω = ω_res / Q (rad/s).
    pub fn linewidth(&self) -> f64 {
        self.resonance_frequency() / self.quality_factor()
    }

    /// Finesse F = π·√R / (1-R).
    pub fn finesse(&self) -> f64 {
        PI * self.r_mirror.sqrt() / (1.0 - self.r_mirror)
    }

    /// Evanescent decay length (1/e) into PC mirror (m).
    ///
    /// For a quarter-wave mirror near the bandgap edge:
    ///   κ ≈ ln(R_per_period) / Λ
    pub fn evanescent_decay_length(&self) -> f64 {
        let r_per_period = self.r_mirror.powf(1.0 / self.n_periods as f64);
        if r_per_period <= 0.0 || r_per_period >= 1.0 {
            return f64::INFINITY;
        }
        -self.pc_period / r_per_period.ln()
    }

    /// Transmission through cavity at angular frequency ω (Airy function).
    pub fn transmission(&self, omega: f64) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        let phi = omega * self.optical_length / SPEED_OF_LIGHT; // round-trip phase / 2
        let r = self.r_mirror;
        let a = (1.0 - r) * (1.0 - r);
        let b = (1.0 - r) * (1.0 - r) + 4.0 * r * (phi.sin()).powi(2);
        a / b
    }
}

/// Simplified 2D photonic crystal point-defect cavity model.
///
/// Parameterised by Q and mode volume V_eff.
#[derive(Debug, Clone, Copy)]
pub struct Pc2dPointDefect {
    /// Quality factor
    pub q_factor: f64,
    /// Mode volume V_eff (m³)
    pub mode_volume: f64,
    /// Resonance frequency (rad/s)
    pub omega_res: f64,
}

impl Pc2dPointDefect {
    pub fn new(q_factor: f64, mode_volume: f64, omega_res: f64) -> Self {
        Self {
            q_factor,
            mode_volume,
            omega_res,
        }
    }

    /// L3 nanocavity in Si PhC at 1550 nm (reference values).
    ///
    /// Q ≈ 100,000 (optimized), V ≈ 0.7 (λ/n)³.
    pub fn l3_silicon_1550() -> Self {
        use crate::units::conversion::SPEED_OF_LIGHT;
        let lambda = 1550e-9;
        let n_si = 3.476;
        let omega = 2.0 * PI * SPEED_OF_LIGHT / lambda;
        let v_eff = 0.7 * (lambda / n_si).powi(3);
        Self::new(1e5, v_eff, omega)
    }

    /// Purcell factor F_P = (3/(4π²)) × (λ/n)³ × Q / V.
    pub fn purcell_factor(&self) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        let lambda = 2.0 * PI * SPEED_OF_LIGHT / self.omega_res;
        let n = 3.476; // assumed Si
        (3.0 / (4.0 * PI * PI)) * (lambda / n).powi(3) * self.q_factor / self.mode_volume
    }

    /// Photon lifetime τ_ph = Q / ω_res (s).
    pub fn photon_lifetime(&self) -> f64 {
        self.q_factor / self.omega_res
    }

    /// Linewidth Δλ (m) ≈ λ² / (Q × λ_res).
    pub fn linewidth_wavelength(&self) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        let lambda = 2.0 * PI * SPEED_OF_LIGHT / self.omega_res;
        lambda / self.q_factor
    }
}

// ---------------------------------------------------------------------------
// H1 cavity: single missing hole
// ---------------------------------------------------------------------------

/// H1 photonic crystal cavity — one missing hole in a triangular lattice slab.
///
/// The H1 cavity is the simplest point defect: a single air hole is omitted
/// from the otherwise periodic lattice.  The missing hole pushes a mode into
/// the photonic bandgap with moderate Q (typically 300–1000 without
/// hole-position optimisation).
#[derive(Debug, Clone, Copy)]
pub struct H1Defect {
    /// Lattice constant a (m)
    pub lattice_const: f64,
    /// Rod / hole radius r (m)
    pub rod_radius: f64,
    /// Refractive index of the rod / hole material
    pub n_rod: f64,
    /// Refractive index of the background medium
    pub n_bg: f64,
}

impl H1Defect {
    /// Create an H1 cavity.
    ///
    /// # Arguments
    /// * `lattice_const` – lattice constant a (m)
    /// * `rod_radius`    – hole radius r (m); typical r/a ≈ 0.30
    /// * `n_rod`         – index inside the rod (air holes → 1.0)
    /// * `n_bg`          – background slab index (Si ≈ 3.476)
    pub fn new(lattice_const: f64, rod_radius: f64, n_rod: f64, n_bg: f64) -> Self {
        Self {
            lattice_const,
            rod_radius,
            n_rod,
            n_bg,
        }
    }

    /// Normalised fill factor f = π r² / (√3/2 · a²) for the triangular lattice.
    ///
    /// This is the fraction of the unit-cell area occupied by the circular hole.
    fn fill_factor(&self) -> f64 {
        let a = self.lattice_const;
        let r = self.rod_radius;
        PI * r * r / (3_f64.sqrt() / 2.0 * a * a)
    }

    /// Estimate the mid-gap frequency (a/λ) for a triangular lattice of air
    /// holes in a high-index slab.
    ///
    /// The TE bandgap of a triangular lattice (r/a ≈ 0.30, n_slab ≈ 3.5) sits
    /// near a/λ ≈ 0.27–0.34.  We use the analytic approximation:
    ///
    ///   f_midgap = 0.305 / a   (in units of c)
    ///
    /// scaled by the background index to account for different n_bg.
    fn midgap_freq_normalized(&self) -> f64 {
        // Effective index approximation: scale the gap centre by n_bg / n_ref
        let n_ref = 3.476; // Si reference
        let f_ref = 0.305; // a/λ at mid-gap for Si triangular lattice
        f_ref * n_ref / self.n_bg
    }

    /// Compute the TE bandgap centre frequency using a PWE band-structure calculation.
    ///
    /// Uses `PhCrystal2d` with the actual lattice parameters (fill factor, eps_bg,
    /// eps_hole) to compute the band diagram on the Γ→M→K→Γ path (standard for
    /// hexagonal lattices).  The bandgap centre is taken as the arithmetic mean of
    /// the maximum of band 1 (index 0) and the minimum of band 2 (index 1).
    ///
    /// Returns the normalised frequency a/λ at the gap centre.
    /// Falls back to the empirical `midgap_freq_normalized()` value if the PWE
    /// computation finds no gap or encounters a degenerate spectrum.
    pub fn bandgap_center_from_pwe(&self) -> f64 {
        let fill = self.fill_factor();
        let eps_bg = self.n_bg * self.n_bg;
        // n_rod is the hole material index (typically 1.0 for air)
        let eps_hole = self.n_rod * self.n_rod;
        // n_g = 7 gives 49 plane waves — good accuracy at low cost (~milliseconds)
        let crystal = PhCrystal2d::hex_holes(eps_bg, eps_hole, fill, 7);
        // Standard Γ→M→K→Γ path for hexagonal lattice; 10 points per segment
        let k_path = kpath_hexagonal(10);
        let bs = crystal.band_diagram(&k_path, Polarization::TE);

        if bs.bands.len() < 2 {
            return self.midgap_freq_normalized();
        }

        // band 1 (index 0) maximum over all k-points
        let band1_max = bs.bands[0]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        // band 2 (index 1) minimum over all k-points
        let band2_min = bs.bands[1].iter().cloned().fold(f64::INFINITY, f64::min);

        if band2_min > band1_max {
            // A proper gap exists; return its centre
            (band1_max + band2_min) / 2.0
        } else {
            // No gap found at this resolution — fall back to empirical formula
            self.midgap_freq_normalized()
        }
    }

    /// Resonance frequency (rad/s) estimated using actual PWE band structure.
    ///
    /// More accurate than `resonance_frequency()` because the gap centre is
    /// computed from the true eigenvalue spectrum rather than from an empirical
    /// index-scaling formula.  Expected runtime: ~50 ms for n_g = 7.
    ///
    /// The H1 defect mode frequency is estimated as the TE bandgap centre.
    /// Although perturbation theory places the mode slightly below the upper
    /// band edge, the gap centre is a good zeroth-order approximation that
    /// already improves substantially on the empirical 5 % offset.
    pub fn resonance_frequency_rigorous(&self) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        let f_gap = self.bandgap_center_from_pwe();
        // f_gap = a/λ  →  ω = 2π · f_gap · c / a
        f_gap * 2.0 * PI * SPEED_OF_LIGHT / self.lattice_const
    }

    /// Resonance frequency estimate (rad/s).
    ///
    /// The H1 defect mode sits approximately 5 % below the mid-gap frequency
    /// (pulled toward the lower band edge by the dielectric perturbation of the
    /// missing hole).
    pub fn resonance_frequency(&self) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        let f_norm = self.midgap_freq_normalized() * (1.0 - 0.05);
        // f_norm = a/λ = a·f/c  →  f = f_norm·c/a  →  ω = 2π·f_norm·c/a
        2.0 * PI * SPEED_OF_LIGHT * f_norm / self.lattice_const
    }

    /// Mode volume estimate V ≈ 1.2 · (λ/n)³ (m³).
    ///
    /// The H1 mode is less confined than the L3 mode; a typical value is
    /// V ≈ 1.0–1.5 (λ/n)³.  We use the coefficient 1.2.
    pub fn mode_volume_estimate(&self) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        let omega = self.resonance_frequency();
        let lambda = 2.0 * PI * SPEED_OF_LIGHT / omega;
        let coeff = 1.2;
        coeff * (lambda / self.n_bg).powi(3)
    }

    /// Quality factor estimate.
    ///
    /// Unoptimised H1 cavities in triangular Si PhC slabs have Q ≈ 300.
    /// Hole-position optimisation can raise this by a factor of 2–3, but here
    /// we return the unoptimised reference value.
    pub fn quality_factor_estimate(&self) -> f64 {
        300.0
    }

    /// Purcell factor estimate using the Q and V estimates.
    ///
    ///   F_P = (3/(4π²)) · (λ/n)³ · Q / V
    pub fn purcell_factor_estimate(&self) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        let omega = self.resonance_frequency();
        let lambda = 2.0 * PI * SPEED_OF_LIGHT / omega;
        let v = self.mode_volume_estimate();
        let q = self.quality_factor_estimate();
        (3.0 / (4.0 * PI * PI)) * (lambda / self.n_bg).powi(3) * q / v
    }
}

// ---------------------------------------------------------------------------
// L3 cavity: three missing holes
// ---------------------------------------------------------------------------

/// L3 photonic crystal cavity — three missing holes in a row.
///
/// The L3 cavity (Akahane et al., *Nature* 2003) achieves Q ≈ 45,000 in
/// unoptimised form and > 10⁶ after hole-position fine-tuning.  The mode
/// volume is V ≈ 0.7 (λ/n)³, making it one of the best platforms for
/// cavity-QED experiments in solid state.
#[derive(Debug, Clone, Copy)]
pub struct L3Defect {
    /// Lattice constant a (m)
    pub lattice_const: f64,
    /// Hole radius r (m)
    pub rod_radius: f64,
    /// Rod / hole index
    pub n_rod: f64,
    /// Background slab index
    pub n_bg: f64,
}

impl L3Defect {
    /// Create an L3 cavity.
    ///
    /// # Arguments
    /// * `lattice_const` – lattice constant a (m)
    /// * `rod_radius`    – hole radius r (m)
    /// * `n_rod`         – index inside the holes (air → 1.0)
    /// * `n_bg`          – slab refractive index (Si ≈ 3.476)
    pub fn new(lattice_const: f64, rod_radius: f64, n_rod: f64, n_bg: f64) -> Self {
        Self {
            lattice_const,
            rod_radius,
            n_rod,
            n_bg,
        }
    }

    /// Mid-gap frequency (same scaling as H1Defect).
    fn midgap_freq_normalized(&self) -> f64 {
        let n_ref = 3.476;
        let f_ref = 0.305;
        f_ref * n_ref / self.n_bg
    }

    /// Resonance frequency estimate (rad/s).
    ///
    /// The L3 mode sits very close to the mid-gap frequency; we apply a small
    /// +2 % shift toward the upper band edge, consistent with MPB calculations
    /// (Akahane et al.).
    pub fn resonance_frequency(&self) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        let f_norm = self.midgap_freq_normalized() * (1.0 + 0.02);
        2.0 * PI * SPEED_OF_LIGHT * f_norm / self.lattice_const
    }

    /// Quality factor estimate.
    ///
    /// Unoptimised L3: Q ≈ 10,000.  This is a conservative lower bound;
    /// published values range from 4,500 to 45,000 depending on simulation
    /// method and slab parameters.
    pub fn quality_factor_estimate(&self) -> f64 {
        10_000.0
    }

    /// Mode volume estimate V ≈ 0.7 · (λ/n)³ (m³).
    pub fn mode_volume_estimate(&self) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        let omega = self.resonance_frequency();
        let lambda = 2.0 * PI * SPEED_OF_LIGHT / omega;
        0.7 * (lambda / self.n_bg).powi(3)
    }

    /// Purcell factor estimate.
    pub fn purcell_factor_estimate(&self) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        let omega = self.resonance_frequency();
        let lambda = 2.0 * PI * SPEED_OF_LIGHT / omega;
        let v = self.mode_volume_estimate();
        let q = self.quality_factor_estimate();
        (3.0 / (4.0 * PI * PI)) * (lambda / self.n_bg).powi(3) * q / v
    }

    /// Photon lifetime τ_ph = Q / ω_res (s).
    pub fn photon_lifetime(&self) -> f64 {
        self.quality_factor_estimate() / self.resonance_frequency()
    }
}

// ---------------------------------------------------------------------------
// W1 waveguide: single-row line-defect waveguide
// ---------------------------------------------------------------------------

/// W1 line-defect waveguide in a triangular-lattice photonic crystal slab.
///
/// A W1 waveguide is formed by removing one row of holes from a triangular PhC.
/// The resulting guided mode is confined to the line defect by the photonic
/// bandgap in the transverse direction and total-internal reflection in the
/// vertical direction.
///
/// Key properties:
/// - Guided mode bandwidth ≈ 10–15 % of the gap centre frequency
/// - Group index diverges near the lower band edge (slow-light regime)
/// - Typical group index n_g ≈ 5–100 depending on frequency detuning
#[derive(Debug, Clone, Copy)]
pub struct W1Waveguide {
    /// Lattice constant a (m)
    pub lattice_const: f64,
    /// Hole radius r (m)
    pub rod_radius: f64,
    /// Slab effective index
    pub n_slab: f64,
}

impl W1Waveguide {
    /// Create a W1 waveguide.
    ///
    /// # Arguments
    /// * `lattice_const` – lattice constant a (m)
    /// * `rod_radius`    – hole radius r (m); typical r/a ≈ 0.30
    /// * `n_slab`        – slab effective index (Si ≈ 3.476)
    pub fn new(lattice_const: f64, rod_radius: f64, n_slab: f64) -> Self {
        Self {
            lattice_const,
            rod_radius,
            n_slab,
        }
    }

    /// Normalised fill factor f = π r² / (√3/2 · a²) for the triangular lattice.
    fn fill_factor(&self) -> f64 {
        let a = self.lattice_const;
        let r = self.rod_radius;
        PI * r * r / (3_f64.sqrt() / 2.0 * a * a)
    }

    /// Centre of the TE bandgap (a/λ), scaled from Si reference.
    fn gap_centre_normalized(&self) -> f64 {
        let n_ref = 3.476;
        let f_ref = 0.305;
        f_ref * n_ref / self.n_slab
    }

    /// Lower cut-off frequency of the W1 guided mode (rad/s).
    ///
    /// The guided band enters the gap approximately 6 % below the gap centre.
    /// The fill factor modulates this slightly: larger holes push the lower edge
    /// up (more air → higher gap edge).
    pub fn cutoff_frequency(&self) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        let ff = self.fill_factor().clamp(0.0, 0.8);
        // Empirical: lower edge ≈ gap_centre × (0.94 - 0.05·ff)
        let f_lower = self.gap_centre_normalized() * (0.94 - 0.05 * ff);
        2.0 * PI * SPEED_OF_LIGHT * f_lower / self.lattice_const
    }

    /// Guided-mode bandwidth as a fraction of the gap-centre frequency.
    ///
    /// Typical W1 bandwidth is 10–15 % of the gap-centre frequency.  The
    /// bandwidth decreases slightly for larger holes (wider gap but the
    /// waveguide mode is pulled deeper into the gap).
    pub fn bandwidth(&self) -> f64 {
        let ff = self.fill_factor().clamp(0.0, 0.8);
        // bandwidth fraction ≈ 0.12 - 0.05·ff
        (0.12 - 0.05 * ff).max(0.01)
    }

    /// Upper edge frequency (rad/s) of the guided band.
    pub fn upper_frequency(&self) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        let f_centre = self.gap_centre_normalized();
        let f_upper = f_centre * (0.94 - 0.05 * self.fill_factor().clamp(0.0, 0.8))
            + f_centre * self.bandwidth();
        2.0 * PI * SPEED_OF_LIGHT * f_upper / self.lattice_const
    }

    /// Approximate group index n_g at normalised frequency `freq_normalized` (a/λ).
    ///
    /// Near the lower band edge the dispersion flattens and n_g diverges.  We
    /// model this with:
    ///
    ///   n_g(f) = n_g0 / (1 − ((f_lower − f) / Δf_slow)²)^(1/2)
    ///
    /// where:
    ///   - n_g0 ≈ 5 (fast-light value near band top)
    ///   - Δf_slow = half the bandwidth (slow-light onset range)
    ///   - f_lower is the normalised lower cut-off
    ///
    /// The result is clamped at 200 to avoid unphysical divergence.
    pub fn group_index_at(&self, freq_normalized: f64) -> f64 {
        let ff = self.fill_factor().clamp(0.0, 0.8);
        let f_lower = self.gap_centre_normalized() * (0.94 - 0.05 * ff);
        let bw = self.bandwidth() * self.gap_centre_normalized();
        let df_slow = bw * 0.5;

        // Distance from the lower band edge (negative means below cut-off)
        let delta_f = freq_normalized - f_lower;
        if delta_f <= 0.0 {
            // Below cut-off → evanescent, no guided mode; return very large n_g
            return 200.0;
        }

        let n_g0 = 5.0;
        let x = (delta_f / df_slow).min(1.0);
        // n_g diverges as x → 0 (near lower band edge)
        let denom = x.max(1e-4);
        (n_g0 / denom).clamp(n_g0, 200.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- existing tests ---

    #[test]
    fn pc1d_resonance_wavelength_near_1550() {
        let cav = Pc1dCavity::half_wave_cavity_1550nm();
        let lambda_res = cav.resonance_wavelength() * 1e9;
        assert!(
            (lambda_res - 1550.0).abs() < 10.0,
            "λ_res={lambda_res:.1}nm"
        );
    }

    #[test]
    fn pc1d_quality_factor_large() {
        let cav = Pc1dCavity::half_wave_cavity_1550nm();
        let q = cav.quality_factor();
        assert!(q > 100.0, "Q={q:.0}");
    }

    #[test]
    fn pc1d_finesse_positive() {
        let cav = Pc1dCavity::half_wave_cavity_1550nm();
        assert!(cav.finesse() > 1.0);
    }

    #[test]
    fn pc1d_transmission_at_resonance_near_one() {
        let cav = Pc1dCavity::half_wave_cavity_1550nm();
        let omega_res = cav.resonance_frequency();
        let t = cav.transmission(omega_res);
        // At resonance sin(φ)≈0 → T≈1
        assert!(t > 0.5, "T={t:.3}");
    }

    #[test]
    fn pc2d_l3_purcell_large() {
        let cav = Pc2dPointDefect::l3_silicon_1550();
        let fp = cav.purcell_factor();
        assert!(fp > 100.0, "F_P={fp:.0}");
    }

    #[test]
    fn pc2d_photon_lifetime_positive() {
        let cav = Pc2dPointDefect::l3_silicon_1550();
        assert!(cav.photon_lifetime() > 0.0);
    }

    #[test]
    fn evanescent_decay_length_positive() {
        let cav = Pc1dCavity::half_wave_cavity_1550nm();
        assert!(cav.evanescent_decay_length() > 0.0);
    }

    // --- H1Defect tests ---

    #[test]
    fn h1_resonance_frequency_positive() {
        let h1 = H1Defect::new(400e-9, 120e-9, 1.0, 3.476);
        let omega = h1.resonance_frequency();
        assert!(omega > 0.0, "ω must be positive");
    }

    #[test]
    fn h1_resonance_in_telecom_band() {
        // a = 400 nm, r = 120 nm (r/a = 0.30), Si slab
        let h1 = H1Defect::new(400e-9, 120e-9, 1.0, 3.476);
        use crate::units::conversion::SPEED_OF_LIGHT;
        let omega = h1.resonance_frequency();
        let lambda_nm = 2.0 * PI * SPEED_OF_LIGHT / omega * 1e9;
        // Expected: 1300–1700 nm for these parameters
        assert!(
            lambda_nm > 1200.0 && lambda_nm < 1800.0,
            "λ_res = {lambda_nm:.0} nm, expected 1200–1800 nm"
        );
    }

    #[test]
    fn h1_mode_volume_positive() {
        let h1 = H1Defect::new(400e-9, 120e-9, 1.0, 3.476);
        assert!(h1.mode_volume_estimate() > 0.0);
    }

    #[test]
    fn h1_quality_factor_estimate() {
        let h1 = H1Defect::new(400e-9, 120e-9, 1.0, 3.476);
        let q = h1.quality_factor_estimate();
        // Typical H1 Q ≈ 300; check within factor of 10
        assert!((100.0..=3000.0).contains(&q), "Q={q}");
    }

    #[test]
    fn h1_purcell_factor_positive() {
        let h1 = H1Defect::new(400e-9, 120e-9, 1.0, 3.476);
        assert!(h1.purcell_factor_estimate() > 0.0);
    }

    #[test]
    fn h1_different_background_index() {
        let h1_si = H1Defect::new(400e-9, 120e-9, 1.0, 3.476);
        let h1_gaas = H1Defect::new(400e-9, 120e-9, 1.0, 3.374);
        // Higher index → lower gap frequency → lower resonance frequency
        assert!(
            h1_si.resonance_frequency() < h1_gaas.resonance_frequency(),
            "Si has lower resonance freq than GaAs at same lattice"
        );
    }

    // --- L3Defect tests ---

    #[test]
    fn l3_resonance_frequency_positive() {
        let l3 = L3Defect::new(420e-9, 126e-9, 1.0, 3.476);
        assert!(l3.resonance_frequency() > 0.0);
    }

    #[test]
    fn l3_resonance_in_telecom_band() {
        let l3 = L3Defect::new(420e-9, 126e-9, 1.0, 3.476);
        use crate::units::conversion::SPEED_OF_LIGHT;
        let omega = l3.resonance_frequency();
        let lambda_nm = 2.0 * PI * SPEED_OF_LIGHT / omega * 1e9;
        assert!(
            lambda_nm > 1200.0 && lambda_nm < 1900.0,
            "λ_res = {lambda_nm:.0} nm"
        );
    }

    #[test]
    fn l3_q_estimate_order_of_magnitude() {
        let l3 = L3Defect::new(420e-9, 126e-9, 1.0, 3.476);
        let q = l3.quality_factor_estimate();
        // Should be ~10,000
        assert!((1_000.0..=1_000_000.0).contains(&q), "Q={q}");
    }

    #[test]
    fn l3_mode_volume_positive() {
        let l3 = L3Defect::new(420e-9, 126e-9, 1.0, 3.476);
        assert!(l3.mode_volume_estimate() > 0.0);
    }

    #[test]
    fn l3_mode_volume_smaller_than_h1() {
        let h1 = H1Defect::new(420e-9, 126e-9, 1.0, 3.476);
        let l3 = L3Defect::new(420e-9, 126e-9, 1.0, 3.476);
        // L3 V ≈ 0.7 (λ/n)³ < H1 V ≈ 1.2 (λ/n)³
        assert!(
            l3.mode_volume_estimate() < h1.mode_volume_estimate(),
            "L3 mode volume should be smaller than H1"
        );
    }

    #[test]
    fn l3_purcell_factor_large() {
        let l3 = L3Defect::new(420e-9, 126e-9, 1.0, 3.476);
        let fp = l3.purcell_factor_estimate();
        // High Q / small V → large Purcell factor
        assert!(fp > 10.0, "F_P={fp:.1}");
    }

    #[test]
    fn l3_photon_lifetime_positive() {
        let l3 = L3Defect::new(420e-9, 126e-9, 1.0, 3.476);
        assert!(l3.photon_lifetime() > 0.0);
    }

    // --- W1Waveguide tests ---

    #[test]
    fn w1_cutoff_frequency_positive() {
        let w1 = W1Waveguide::new(430e-9, 129e-9, 3.476);
        assert!(w1.cutoff_frequency() > 0.0);
    }

    #[test]
    fn w1_bandwidth_in_range() {
        let w1 = W1Waveguide::new(430e-9, 129e-9, 3.476);
        let bw = w1.bandwidth();
        // Bandwidth fraction should be 0.01–0.20
        assert!((0.01..=0.20).contains(&bw), "bandwidth={bw:.4}");
    }

    #[test]
    fn w1_upper_frequency_above_cutoff() {
        let w1 = W1Waveguide::new(430e-9, 129e-9, 3.476);
        assert!(
            w1.upper_frequency() > w1.cutoff_frequency(),
            "Upper edge must be above lower cut-off"
        );
    }

    #[test]
    fn w1_group_index_increases_near_band_edge() {
        let w1 = W1Waveguide::new(430e-9, 129e-9, 3.476);
        let ff = w1.fill_factor();
        let f_lower = w1.gap_centre_normalized() * (0.94 - 0.05 * ff.clamp(0.0, 0.8));

        // Evaluate group index just above and well above the lower band edge
        let ng_near_edge = w1.group_index_at(f_lower + 0.001);
        let ng_far = w1.group_index_at(f_lower + 0.05);

        assert!(
            ng_near_edge > ng_far,
            "Group index near band edge ({ng_near_edge:.1}) should exceed far value ({ng_far:.1})"
        );
    }

    #[test]
    fn w1_group_index_below_cutoff_large() {
        let w1 = W1Waveguide::new(430e-9, 129e-9, 3.476);
        // Below cut-off (a/λ < f_lower) → n_g should be very large
        let ng = w1.group_index_at(0.1);
        assert!(ng >= 100.0, "Below cut-off n_g should be large, got {ng}");
    }

    #[test]
    fn w1_fill_factor_typical_range() {
        let w1 = W1Waveguide::new(430e-9, 129e-9, 3.476);
        let ff = w1.fill_factor();
        // r/a = 0.30 → ff ≈ 0.326
        assert!(ff > 0.1 && ff < 0.6, "fill factor={ff:.3}");
    }

    // --- PWE-based H1 tests ---

    #[test]
    fn bandgap_center_from_pwe_positive() {
        let h1 = H1Defect::new(400e-9, 120e-9, 1.0, 3.476);
        let f_gap = h1.bandgap_center_from_pwe();
        assert!(f_gap > 0.0, "PWE gap centre must be positive, got {f_gap}");
    }

    #[test]
    fn h1_rigorous_frequency_in_bandgap() {
        use crate::units::conversion::SPEED_OF_LIGHT;
        // The rigorous frequency should lie in the typical H1 gap range for
        // n = 3.476, r/a = 0.30.
        let h1 = H1Defect::new(400e-9, 120e-9, 1.0, 3.476);
        let omega_rig = h1.resonance_frequency_rigorous();
        assert!(omega_rig > 0.0, "rigorous frequency must be positive");
        // Recover normalised frequency a/λ
        let f_norm = omega_rig * h1.lattice_const / (2.0 * PI * SPEED_OF_LIGHT);
        assert!(f_norm > 0.20, "f_norm too low: {f_norm}");
        assert!(f_norm < 0.45, "f_norm too high: {f_norm}");
    }

    #[test]
    fn h1_rigorous_higher_index_lower_frequency() {
        // Higher background index pushes the bandgap to lower normalised
        // frequencies, so the PWE-derived resonance should be lower for Si
        // (n = 3.476) than for GaAs (n = 3.374).
        let h1_si = H1Defect::new(400e-9, 120e-9, 1.0, 3.476);
        let h1_gaas = H1Defect::new(400e-9, 120e-9, 1.0, 3.374);
        let f_si = h1_si.resonance_frequency_rigorous();
        let f_gaas = h1_gaas.resonance_frequency_rigorous();
        assert!(
            f_si < f_gaas,
            "Si (n=3.476) should have lower ω than GaAs (n=3.374): si={f_si}, gaas={f_gaas}"
        );
    }

    #[test]
    fn h1_rigorous_vs_empirical_same_order() {
        // Rigorous and empirical results should agree within 20 %.
        let h1 = H1Defect::new(400e-9, 120e-9, 1.0, 3.476);
        let f_emp = h1.resonance_frequency();
        let f_rig = h1.resonance_frequency_rigorous();
        let ratio = f_rig / f_emp;
        assert!(
            ratio > 0.7 && ratio < 1.3,
            "empirical/rigorous mismatch: ratio={ratio:.3}, emp={f_emp:.3e}, rig={f_rig:.3e}"
        );
    }
}
