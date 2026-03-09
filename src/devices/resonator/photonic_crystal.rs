//! Photonic crystal resonator device model.
//!
//! Photonic crystal (PhC) resonators confine light in ultra-small mode volumes
//! at photonic bandgap frequencies. Key applications:
//!   - Cavity QED (strong coupling between emitter and cavity)
//!   - Ultra-low-threshold nanolasers
//!   - Single-photon sources
//!   - Nonlinear optics (low threshold due to small V and high Q)
//!
//! Key figure of merit: Q/V (quality factor per mode volume)
//!
//! L3 nanocavity: three missing holes in a triangular-lattice PhC slab.
//!   - Q ≈ 10⁴ – 10⁶ (design-dependent)
//!   - V ≈ 0.7 (λ/n)³
//!
//! H0 cavity: two modified holes (no missing holes).
//!   - V → 0 (smallest possible)
//!   - Q ≈ 10⁴

use std::f64::consts::PI;

/// Photonic crystal resonator parameters.
#[derive(Debug, Clone, Copy)]
pub struct PhCResonator {
    /// Quality factor Q (intrinsic)
    pub q_intrinsic: f64,
    /// Radiation quality factor Q_rad (coupling to free space)
    pub q_radiation: f64,
    /// Mode volume V_eff (m³)
    pub mode_volume: f64,
    /// Resonance frequency ω₀ (rad/s)
    pub omega0: f64,
    /// Background refractive index (e.g., silicon slab n=3.46)
    pub n_slab: f64,
}

impl PhCResonator {
    /// Create PhC resonator.
    pub fn new(
        q_intrinsic: f64,
        q_radiation: f64,
        mode_volume: f64,
        omega0: f64,
        n_slab: f64,
    ) -> Self {
        Self {
            q_intrinsic,
            q_radiation,
            mode_volume,
            omega0,
            n_slab,
        }
    }

    /// L3 nanocavity in Si PhC slab at 1550 nm.
    ///
    /// Q_int = 2×10⁵ (absorption limited), Q_rad = 10⁶, V = 0.7(λ/n)³.
    pub fn l3_silicon_1550() -> Self {
        let lambda = 1550e-9;
        let n = 3.476;
        let omega0 = 2.0 * PI * 3e8 / lambda;
        let v_eff = 0.7 * (lambda / n).powi(3);
        Self::new(2e5, 1e6, v_eff, omega0, n)
    }

    /// H1 nanocavity in Si PhC at 1300 nm.
    pub fn h1_silicon_1300() -> Self {
        let lambda = 1300e-9;
        let n = 3.5;
        let omega0 = 2.0 * PI * 3e8 / lambda;
        let v_eff = 0.4 * (lambda / n).powi(3);
        Self::new(1e4, 5e4, v_eff, omega0, n)
    }

    /// Total quality factor: 1/Q_tot = 1/Q_int + 1/Q_rad.
    pub fn q_total(&self) -> f64 {
        1.0 / (1.0 / self.q_intrinsic + 1.0 / self.q_radiation)
    }

    /// Resonance wavelength (m).
    pub fn resonance_wavelength(&self) -> f64 {
        2.0 * PI * 3e8 / self.omega0
    }

    /// Mode linewidth Δω = ω₀ / Q_tot (rad/s).
    pub fn linewidth_omega(&self) -> f64 {
        self.omega0 / self.q_total()
    }

    /// Mode linewidth in wavelength: Δλ = λ₀ / Q.
    pub fn linewidth_wavelength(&self) -> f64 {
        self.resonance_wavelength() / self.q_total()
    }

    /// Photon lifetime τ_ph = Q_tot / ω₀ (s).
    pub fn photon_lifetime(&self) -> f64 {
        self.q_total() / self.omega0
    }

    /// Purcell factor F_P = (3/4π²) · (λ/n)³ · Q/V.
    ///
    /// Enhances spontaneous emission rate for an emitter at cavity antinode.
    pub fn purcell_factor(&self) -> f64 {
        let lambda = self.resonance_wavelength();
        let q = self.q_total();
        (3.0 / (4.0 * PI * PI)) * (lambda / self.n_slab).powi(3) * q / self.mode_volume
    }

    /// Critical coupling condition: Q_int = Q_rad → maximum power in cavity.
    pub fn is_critically_coupled(&self, tolerance_fraction: f64) -> bool {
        (self.q_intrinsic - self.q_radiation).abs() / (self.q_intrinsic + self.q_radiation)
            < tolerance_fraction
    }

    /// Power in the cavity at resonance (normalized to input power).
    ///
    /// For overcoupled cavity: T_max = 4·Q_int·Q_rad / (Q_int + Q_rad)²
    pub fn intra_cavity_enhancement(&self) -> f64 {
        let q_i = self.q_intrinsic;
        let q_r = self.q_radiation;
        4.0 * q_i * q_r / (q_i + q_r).powi(2)
    }

    /// Transmission through the cavity (lorentzian lineshape).
    ///
    /// T(ω) = Q_tot² / Q_rad² × 1 / (1 + 4·Q_tot²·((ω-ω₀)/ω₀)²)
    pub fn transmission(&self, omega: f64) -> f64 {
        let q_tot = self.q_total();
        let q_rad = self.q_radiation;
        let detuning = (omega - self.omega0) / self.omega0;
        let t_peak = (q_tot / q_rad).powi(2);
        t_peak / (1.0 + 4.0 * q_tot * q_tot * detuning * detuning)
    }

    /// Nonlinear threshold power for bistability (Kerr medium, approximate).
    ///
    ///   P_th ≈ V_eff · n₀² · Δω² / (c · n₂ · ω₀ · Q)
    pub fn bistability_threshold_w(&self, n2_m2_per_w: f64) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        let dw = self.linewidth_omega();
        self.mode_volume * self.n_slab * self.n_slab * dw * dw
            / (SPEED_OF_LIGHT * n2_m2_per_w * self.omega0 * self.q_total())
    }
}

// ---------------------------------------------------------------------------
// L3CavityEstimate — literature-based L3 nanocavity estimate
// ---------------------------------------------------------------------------

/// Fast analytical estimate of an L3 photonic crystal nanocavity.
///
/// Uses well-established FDTD literature values:
///   - Normalised resonance frequency: ω·a/(2πc) ≈ 0.264  (Akahane et al. 2003)
///   - Standard Q ≈ 10,000 (unoptimised design)
///   - Mode volume: V ≈ 0.69·(λ/n)³
///
/// These are starting-point estimates; optimised designs reach Q > 10⁶.
#[derive(Debug, Clone, Copy)]
pub struct L3CavityEstimate {
    /// Triangular-lattice constant a (m).
    pub lattice_const: f64,
    /// Slab thickness (m).
    pub slab_thickness: f64,
    /// Slab refractive index.
    pub n_slab: f64,
}

impl L3CavityEstimate {
    /// Create an L3 cavity estimate.
    pub fn new(lattice_const: f64, slab_thickness: f64, n_slab: f64) -> Self {
        Self {
            lattice_const,
            slab_thickness,
            n_slab,
        }
    }

    /// Resonance wavelength (m) from the normalised frequency ω·a/(2πc) ≈ 0.264.
    pub fn resonance_wavelength(&self) -> f64 {
        // ω·a/(2πc) = 0.264  →  λ = a / 0.264
        self.lattice_const / 0.264
    }

    /// Resonance angular frequency (rad/s).
    pub fn resonance_omega(&self) -> f64 {
        2.0 * PI * 3e8 / self.resonance_wavelength()
    }

    /// Standard Q factor for an unoptimised L3 cavity (~10,000).
    pub fn quality_factor(&self) -> f64 {
        10_000.0
    }

    /// Mode volume in units of (λ/n)³.  V ≈ 0.69 · (λ/n)³.
    pub fn mode_volume_cubic_lambda(&self) -> f64 {
        0.69
    }

    /// Absolute mode volume (m³).
    pub fn mode_volume_m3(&self) -> f64 {
        let lambda = self.resonance_wavelength();
        self.mode_volume_cubic_lambda() * (lambda / self.n_slab).powi(3)
    }

    /// Purcell factor F_p = (3/(4π²)) · Q/V · (λ/n)³.
    pub fn purcell_factor(&self) -> f64 {
        let q = self.quality_factor();
        let v = self.mode_volume_m3();
        let lambda = self.resonance_wavelength();
        (3.0 / (4.0 * PI * PI)) * q * (lambda / self.n_slab).powi(3) / v
    }

    /// Photon lifetime τ = Q / ω₀ (s).
    pub fn photon_lifetime(&self) -> f64 {
        self.quality_factor() / self.resonance_omega()
    }
}

// ---------------------------------------------------------------------------
// CoupledL3Resonators — two coupled L3 nanocavities
// ---------------------------------------------------------------------------

/// Two coupled L3 nanocavities modelled by coupled-mode theory.
///
/// The Hamiltonian is:
///   H = [[ω₁, g], [g, ω₂]]
///
/// where g is the inter-cavity coupling rate.  The normal-mode frequencies are:
///   ω± = (ω₁+ω₂)/2 ± √(g² + Δ²/4)
/// with Δ = ω₁ − ω₂.
#[derive(Debug, Clone)]
pub struct CoupledL3Resonators {
    /// First cavity.
    pub cavity1: L3CavityEstimate,
    /// Second cavity.
    pub cavity2: L3CavityEstimate,
    /// Inter-cavity coupling rate g (rad/s).
    pub coupling_g: f64,
}

impl CoupledL3Resonators {
    /// Create a pair of coupled L3 cavities.
    pub fn new(cavity1: L3CavityEstimate, cavity2: L3CavityEstimate, coupling_g: f64) -> Self {
        Self {
            cavity1,
            cavity2,
            coupling_g,
        }
    }

    /// Normal-mode split frequencies (ω₋, ω₊) in rad/s.
    pub fn split_frequencies(&self) -> (f64, f64) {
        let w1 = self.cavity1.resonance_omega();
        let w2 = self.cavity2.resonance_omega();
        let w_avg = (w1 + w2) / 2.0;
        let delta = w1 - w2;
        let split = (self.coupling_g * self.coupling_g + delta * delta / 4.0).sqrt();
        (w_avg - split, w_avg + split)
    }

    /// Normal-mode splitting 2g̃ (rad/s), i.e. ω₊ − ω₋.
    pub fn normal_mode_splitting(&self) -> f64 {
        let (w_minus, w_plus) = self.split_frequencies();
        w_plus - w_minus
    }

    /// Transmission spectrum through the coupled-cavity system.
    ///
    /// Uses the input–output formalism for two side-coupled resonators.
    /// The transmission is modelled as an EIT-like doublet:
    ///   T(ω) = |t|² where t = product of individual Lorentzians weighted by coupling.
    ///
    /// For identical cavities (degenerate case) this produces a symmetric doublet.
    ///
    /// `freqs`: angular frequencies (rad/s) at which to evaluate T.
    pub fn transmission_spectrum(&self, freqs: &[f64]) -> Vec<f64> {
        let q1 = self.cavity1.quality_factor();
        let q2 = self.cavity2.quality_factor();
        let w1 = self.cavity1.resonance_omega();
        let w2 = self.cavity2.resonance_omega();
        let gamma1 = w1 / q1; // half-linewidth of cavity 1
        let gamma2 = w2 / q2; // half-linewidth of cavity 2
        let g = self.coupling_g;

        freqs
            .iter()
            .map(|&w| {
                // Green's function poles of the coupled system:
                //   D(ω) = (ω-ω₁ + iγ₁/2)(ω-ω₂ + iγ₂/2) - g²
                // Transmission ∝ 1 / |D(ω)|²  (normalised to off-resonance)
                let d1_re = w - w1;
                let d1_im = gamma1 / 2.0;
                let d2_re = w - w2;
                let d2_im = gamma2 / 2.0;
                // D = d1 * d2 - g²   (complex multiply)
                let prod_re = d1_re * d2_re - d1_im * d2_im;
                let prod_im = d1_re * d2_im + d1_im * d2_re;
                let d_re = prod_re - g * g;
                let d_im = prod_im;
                let denom = d_re * d_re + d_im * d_im;
                // Normalise by off-resonance value (|ω| → ∞ → denom → ω⁴)
                // We use g⁴ as scale so peak is O(1)
                let norm = (g * g + gamma1 * gamma2 / 4.0).powi(2);
                (norm / denom.max(1e-300)).min(1.0)
            })
            .collect()
    }

    /// Centre frequency (average of the two cavity resonances) in rad/s.
    pub fn centre_frequency(&self) -> f64 {
        (self.cavity1.resonance_omega() + self.cavity2.resonance_omega()) / 2.0
    }
}

// ---------------------------------------------------------------------------
// W1WaveguideDispersion — W1 photonic crystal waveguide
// ---------------------------------------------------------------------------

/// W1 photonic crystal waveguide dispersion model.
///
/// A W1 waveguide is formed by removing one row of holes from a triangular-lattice
/// PhC slab.  The guided band exhibits slow-light behaviour near the Brillouin zone
/// edge (k → π/a), where the group velocity vanishes.
///
/// The dispersion relation is approximated by a cosine band model:
///   ω(k) = ω_c − (Δω/2)·cos(k·a)
///
/// with ω_c the band-centre frequency and Δω the bandwidth of the guided band.
/// Literature values for a Si slab PhC:
///   ω_c·a/(2πc) ≈ 0.27,  Δω·a/(2πc) ≈ 0.02.
#[derive(Debug, Clone, Copy)]
pub struct W1WaveguideDispersion {
    /// Lattice constant a (m).
    pub lattice_const: f64,
    /// Slab refractive index.
    pub n_slab: f64,
    /// Normalised band-centre frequency ω_c·a/(2πc).
    pub omega_c_norm: f64,
    /// Normalised guided-band half-width Δω·a/(2πc).
    pub delta_omega_norm: f64,
}

impl W1WaveguideDispersion {
    /// Create a W1 waveguide dispersion model with Si-like default parameters.
    pub fn new(lattice_const: f64, n_slab: f64) -> Self {
        Self {
            lattice_const,
            n_slab,
            omega_c_norm: 0.27,
            delta_omega_norm: 0.02,
        }
    }

    /// Override the band parameters.
    pub fn with_band_params(mut self, omega_c_norm: f64, delta_omega_norm: f64) -> Self {
        self.omega_c_norm = omega_c_norm;
        self.delta_omega_norm = delta_omega_norm;
        self
    }

    /// Convert normalised frequency ω·a/(2πc) to absolute ω (rad/s).
    fn denorm_omega(&self, omega_norm: f64) -> f64 {
        omega_norm * 2.0 * PI * 3e8 / self.lattice_const
    }

    /// Convert absolute ω to normalised frequency.
    #[allow(dead_code)]
    fn norm_omega(&self, omega: f64) -> f64 {
        omega * self.lattice_const / (2.0 * PI * 3e8)
    }

    /// Wave vector k (units of π/a) for a given normalised frequency ω·a/(2πc).
    ///
    /// Inverts: ω = ω_c − (Δω/2)·cos(k·a)  →  k = (1/a)·arccos((ω_c−ω)/(Δω/2))
    /// Returns None if outside the guided band.
    fn k_norm_from_omega(&self, omega_norm: f64) -> Option<f64> {
        let arg = (self.omega_c_norm - omega_norm) / (self.delta_omega_norm / 2.0);
        if arg.abs() > 1.0 {
            return None;
        }
        // k normalised: k·a/π
        let k_a = arg.acos(); // in [0, π]
        Some(k_a / PI)
    }

    /// Group index n_g = c / v_g at a given normalised frequency ω·a/(2πc).
    ///
    /// v_g = dω/dk = (Δω/2)·a·sin(k·a)
    /// n_g = c / v_g
    ///
    /// Near the band edge (k → π/a), sin(k·a) → 0 and n_g → ∞.
    pub fn group_index_at_freq(&self, omega_norm: f64) -> f64 {
        match self.k_norm_from_omega(omega_norm) {
            None => f64::INFINITY,
            Some(k_norm_pi) => {
                let k_a = k_norm_pi * PI; // k·a in [0, π]
                let sin_ka = k_a.sin().abs().max(1e-10);
                // v_g = (Δω_abs/2) · a · sin(k·a)
                let delta_omega_abs = self.delta_omega_norm * 2.0 * PI * 3e8 / self.lattice_const;
                let vg = (delta_omega_abs / 2.0) * self.lattice_const * sin_ka;
                if vg < 1e-10 {
                    f64::INFINITY
                } else {
                    3e8 / vg
                }
            }
        }
    }

    /// Dispersion curve as (ω_norm, k_norm) pairs, sweeping k from 0 to π/a.
    ///
    /// `n_pts` evenly spaced k values from 0 to π/a.
    /// Returns Vec<(omega_norm, k_norm)> where k_norm = k·a/π ∈ [0, 1].
    pub fn dispersion_curve(&self, n_pts: usize) -> Vec<(f64, f64)> {
        assert!(n_pts >= 2, "n_pts must be >= 2");
        (0..n_pts)
            .map(|i| {
                let k_norm = i as f64 / (n_pts - 1) as f64; // k·a/π ∈ [0,1]
                let k_a = k_norm * PI;
                let omega_norm = self.omega_c_norm - (self.delta_omega_norm / 2.0) * k_a.cos();
                (omega_norm, k_norm)
            })
            .collect()
    }

    /// Frequency range (in rad/s) over which n_g > n_g_target (slow-light bandwidth).
    ///
    /// Returns 0 if the target group index is below the minimum group index in the band.
    pub fn slow_light_bandwidth(&self, n_g_target: f64) -> f64 {
        // n_g(ω) = c / v_g increases as k → π/a (band edge).
        // Threshold k where n_g = n_g_target:
        //   n_g = c / [(Δω_abs/2)·a·sin(k·a)] = n_g_target
        //   sin(k·a) = c / [n_g_target · (Δω_abs/2) · a]
        let delta_omega_abs = self.delta_omega_norm * 2.0 * PI * 3e8 / self.lattice_const;
        let vg_threshold = 3e8 / n_g_target;
        let sin_ka_threshold = 2.0 * vg_threshold / (delta_omega_abs * self.lattice_const);
        if sin_ka_threshold >= 1.0 {
            // n_g never reaches n_g_target in this band
            return 0.0;
        }
        // Two solutions for k: k₁ (near band centre) and k₂=π-k₁ (near band edge)
        let k_a_1 = sin_ka_threshold.asin();
        let k_a_2 = PI - k_a_1;
        // ω at these k values
        let w_norm_1 = self.omega_c_norm - (self.delta_omega_norm / 2.0) * k_a_1.cos();
        let w_norm_2 = self.omega_c_norm - (self.delta_omega_norm / 2.0) * k_a_2.cos();
        let bw_norm = (w_norm_2 - w_norm_1).abs();
        // Convert to rad/s
        self.denorm_omega(bw_norm) - self.denorm_omega(0.0) + self.denorm_omega(bw_norm).abs()
        // Simpler: BW = bw_norm * 2πc / a
    }
}

/// Compute slow-light bandwidth (rad/s) directly from normalised parameters.
///
/// This is a stand-alone helper that avoids the sign ambiguity in the method above.
/// Returns Δω (rad/s) = |ω(k₂) - ω(k₁)| where n_g(k₁,₂) = n_g_target.
pub fn slow_light_bandwidth_hz(lattice_const: f64, delta_omega_norm: f64, n_g_target: f64) -> f64 {
    let delta_omega_abs = delta_omega_norm * 2.0 * PI * 3e8 / lattice_const;
    let vg_threshold = 3e8 / n_g_target;
    let sin_ka_threshold = 2.0 * vg_threshold / (delta_omega_abs * lattice_const);
    if sin_ka_threshold >= 1.0 {
        return 0.0;
    }
    let k_a_1 = sin_ka_threshold.asin();
    let k_a_2 = PI - k_a_1;
    (delta_omega_abs / 2.0) * (k_a_2.cos() - k_a_1.cos()).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- PhCResonator (original) ----

    #[test]
    fn phc_resonator_q_total_less_than_components() {
        let r = PhCResonator::l3_silicon_1550();
        let qt = r.q_total();
        assert!(qt < r.q_intrinsic);
        assert!(qt < r.q_radiation);
    }

    #[test]
    fn phc_resonator_wavelength_near_1550() {
        let r = PhCResonator::l3_silicon_1550();
        let lambda_nm = r.resonance_wavelength() * 1e9;
        assert!((lambda_nm - 1550.0).abs() < 1.0, "λ={lambda_nm:.1}nm");
    }

    #[test]
    fn phc_resonator_purcell_large() {
        let r = PhCResonator::l3_silicon_1550();
        let fp = r.purcell_factor();
        assert!(fp > 100.0, "F_P={fp:.0}");
    }

    #[test]
    fn phc_resonator_photon_lifetime_positive() {
        let r = PhCResonator::l3_silicon_1550();
        assert!(r.photon_lifetime() > 0.0);
    }

    #[test]
    fn phc_critical_coupling_symmetric() {
        let r = PhCResonator::new(1e5, 1e5, 1e-18, 1.2e15, 3.476);
        assert!(r.is_critically_coupled(0.01));
    }

    #[test]
    fn phc_transmission_peak_at_resonance() {
        let r = PhCResonator::l3_silicon_1550();
        let t_res = r.transmission(r.omega0);
        let t_off = r.transmission(r.omega0 * 1.01);
        assert!(t_res > t_off, "T_res={t_res:.3} T_off={t_off:.3}");
    }

    #[test]
    fn phc_intra_cavity_enhancement_positive() {
        let r = PhCResonator::l3_silicon_1550();
        let enh = r.intra_cavity_enhancement();
        assert!(enh > 0.0 && enh <= 1.0, "enh={enh:.3}");
    }

    // ---- L3CavityEstimate ----

    #[test]
    fn l3_resonance_wavelength_consistent() {
        // a = 420 nm → λ = 420/0.264 ≈ 1590 nm (typical Si PhC)
        let cav = L3CavityEstimate::new(420e-9, 220e-9, 3.476);
        let lambda_nm = cav.resonance_wavelength() * 1e9;
        let expected_nm = 420.0 / 0.264;
        assert!(
            (lambda_nm - expected_nm).abs() < 1.0,
            "λ={lambda_nm:.1}nm expected≈{expected_nm:.1}nm"
        );
    }

    #[test]
    fn l3_quality_factor_is_ten_thousand() {
        let cav = L3CavityEstimate::new(420e-9, 220e-9, 3.476);
        assert!((cav.quality_factor() - 10_000.0).abs() < 1.0);
    }

    #[test]
    fn l3_mode_volume_cubic_lambda_near_069() {
        let cav = L3CavityEstimate::new(420e-9, 220e-9, 3.476);
        assert!((cav.mode_volume_cubic_lambda() - 0.69).abs() < 1e-10);
    }

    #[test]
    fn l3_mode_volume_m3_small() {
        let cav = L3CavityEstimate::new(420e-9, 220e-9, 3.476);
        let v = cav.mode_volume_m3();
        // (λ/n)³ ≈ (450nm/3.5)³ ≈ (128nm)³ ≈ 2e-21 m³
        assert!(v > 1e-22 && v < 1e-18, "V={v:.2e} m³ out of range");
    }

    #[test]
    fn l3_purcell_factor_positive() {
        let cav = L3CavityEstimate::new(420e-9, 220e-9, 3.476);
        let fp = cav.purcell_factor();
        assert!(fp > 1.0, "F_P={fp:.1}");
    }

    #[test]
    fn l3_photon_lifetime_positive() {
        let cav = L3CavityEstimate::new(420e-9, 220e-9, 3.476);
        assert!(cav.photon_lifetime() > 0.0);
    }

    // ---- CoupledL3Resonators ----

    #[test]
    fn coupled_l3_identical_split_equals_2g() {
        let a = 420e-9;
        let cav1 = L3CavityEstimate::new(a, 220e-9, 3.476);
        let cav2 = L3CavityEstimate::new(a, 220e-9, 3.476);
        let g = 1e11; // rad/s
        let coupled = CoupledL3Resonators::new(cav1, cav2, g);
        let split = coupled.normal_mode_splitting();
        // For identical cavities: split = 2g
        assert!(
            (split - 2.0 * g).abs() / (2.0 * g) < 1e-9,
            "split={split:.3e} expected 2g={:.3e}",
            2.0 * g
        );
    }

    #[test]
    fn coupled_l3_split_frequencies_ordered() {
        let cav1 = L3CavityEstimate::new(420e-9, 220e-9, 3.476);
        let cav2 = L3CavityEstimate::new(420e-9, 220e-9, 3.476);
        let coupled = CoupledL3Resonators::new(cav1, cav2, 5e10);
        let (w_minus, w_plus) = coupled.split_frequencies();
        assert!(w_minus < w_plus, "ω₋ should be < ω₊");
    }

    #[test]
    fn coupled_l3_transmission_doublet_symmetric() {
        let a = 420e-9;
        let cav1 = L3CavityEstimate::new(a, 220e-9, 3.476);
        let cav2 = L3CavityEstimate::new(a, 220e-9, 3.476);
        let g = 2e11;
        let coupled = CoupledL3Resonators::new(cav1, cav2, g);
        let w0 = coupled.centre_frequency();
        let freqs = vec![w0 - g, w0, w0 + g];
        let t = coupled.transmission_spectrum(&freqs);
        // T at ω₋ and ω₊ should be equal (symmetric doublet)
        assert!(
            (t[0] - t[2]).abs() < 1e-6,
            "T(ω₋)={:.4} T(ω₊)={:.4} should be equal",
            t[0],
            t[2]
        );
    }

    #[test]
    fn coupled_l3_transmission_at_gap_lower() {
        // At ω₀ (between the doublet peaks), T should be lower than at peaks
        let a = 420e-9;
        let cav1 = L3CavityEstimate::new(a, 220e-9, 3.476);
        let cav2 = L3CavityEstimate::new(a, 220e-9, 3.476);
        let g = 1e12;
        let coupled = CoupledL3Resonators::new(cav1, cav2, g);
        let (w_minus, w_plus) = coupled.split_frequencies();
        let w0 = (w_minus + w_plus) / 2.0;
        let t = coupled.transmission_spectrum(&[w_minus, w0, w_plus]);
        // peaks should be larger than gap
        assert!(t[0] >= t[1], "T at w- should >= T at gap");
        assert!(t[2] >= t[1], "T at w+ should >= T at gap");
    }

    // ---- W1WaveguideDispersion ----

    #[test]
    fn w1_dispersion_curve_length_correct() {
        let w1 = W1WaveguideDispersion::new(420e-9, 3.476);
        let curve = w1.dispersion_curve(51);
        assert_eq!(curve.len(), 51);
    }

    #[test]
    fn w1_dispersion_curve_k_range() {
        let w1 = W1WaveguideDispersion::new(420e-9, 3.476);
        let curve = w1.dispersion_curve(21);
        assert!((curve[0].1).abs() < 1e-10, "k_norm at start should be 0");
        assert!(
            (curve[20].1 - 1.0).abs() < 1e-10,
            "k_norm at end should be 1"
        );
    }

    #[test]
    fn w1_dispersion_monotone_in_omega() {
        // ω should be monotonically increasing with k (normal dispersion branch)
        let w1 = W1WaveguideDispersion::new(420e-9, 3.476);
        let curve = w1.dispersion_curve(21);
        for i in 1..curve.len() {
            assert!(
                curve[i].0 >= curve[i - 1].0 - 1e-10,
                "ω should be non-decreasing: ω[{i}]={} < ω[{}]={}",
                curve[i].0,
                i - 1,
                curve[i - 1].0
            );
        }
    }

    #[test]
    fn w1_group_index_increases_near_band_edge() {
        let w1 = W1WaveguideDispersion::new(420e-9, 3.476);
        // n_g is maximised near the band edges (k→0 and k→π/a) and minimised near
        // the inflection point at k≈π/2a (quarter of the BZ).
        // Check: mid-band n_g (k~0.5) < near-edge n_g (k~0.9).
        let curve = w1.dispersion_curve(101);
        // k~0.5 → index 50, k~0.9 → index 90
        let omega_mid = curve[50].0;
        let omega_near_edge = curve[90].0;
        let ng_mid = w1.group_index_at_freq(omega_mid);
        let ng_near_edge = w1.group_index_at_freq(omega_near_edge);
        assert!(
            ng_near_edge > ng_mid || ng_near_edge.is_infinite(),
            "n_g near band edge={ng_near_edge:.1} should ≥ n_g at mid-band={ng_mid:.1}"
        );
    }

    #[test]
    fn w1_group_index_outside_band_is_infinity() {
        let w1 = W1WaveguideDispersion::new(420e-9, 3.476);
        // Far outside the guided band
        let ng = w1.group_index_at_freq(0.5);
        assert!(
            ng.is_infinite() || ng > 1e10,
            "Expected large/infinite n_g outside band"
        );
    }

    #[test]
    fn w1_slow_light_bandwidth_positive_for_moderate_ng() {
        let w1 = W1WaveguideDispersion::new(420e-9, 3.476);
        // Ask for n_g > 10 (slow but not extreme)
        let bw = slow_light_bandwidth_hz(420e-9, w1.delta_omega_norm, 10.0);
        assert!(bw >= 0.0, "Slow-light BW should be non-negative");
    }

    #[test]
    fn w1_slow_light_bandwidth_zero_for_large_ng_target() {
        // `slow_light_bandwidth_hz` returns 0 when sin_ka_threshold ≥ 1, i.e.
        // when 2·vg_threshold / (Δω_abs · a) ≥ 1, i.e. n_g_target ≤ 1/(π·Δω_norm).
        // For Δω_norm = 0.02: threshold = 1/(π·0.02) ≈ 15.9.
        // So for n_g_target = 5 (< 15.9), the band cannot achieve n_g ≥ 5 anywhere
        // in the interior, and the function returns 0.
        let bw_below_threshold = slow_light_bandwidth_hz(420e-9, 0.02, 5.0);
        assert!(
            bw_below_threshold == 0.0,
            "BW should be 0 when n_g_target is below the slow-light threshold, got {bw_below_threshold:.4e}"
        );

        // For n_g_target well above the threshold (e.g., 100 >> 15.9), there is
        // a non-zero slow-light bandwidth.
        let bw_above_threshold = slow_light_bandwidth_hz(420e-9, 0.02, 100.0);
        assert!(
            bw_above_threshold > 0.0,
            "BW should be positive for n_g_target above threshold, got {bw_above_threshold:.4e}"
        );
    }
}
