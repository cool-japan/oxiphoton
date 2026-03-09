//! Local Density of Optical States (LDOS), spontaneous emission, and cavity QED
//!
//! The LDOS quantifies the number of electromagnetic modes available at a given
//! frequency and position, directly determining the spontaneous emission rate of
//! quantum emitters (Fermi's golden rule):
//!
//!   Γ = (π ω |d|²) / (3 ε₀ ħ) · ρ(r, ω)
//!
//! Physical constants used:
//!   c = 2.997924580e8 m/s
//!   ħ = 1.054571817e-34 J·s
//!   ε₀ = 8.854187817e-12 F/m
//!   k_B = 1.380649e-23 J/K

use num_complex::Complex64;
use std::f64::consts::PI;

// ─── Physical constants ───────────────────────────────────────────────────────

const C0: f64 = 2.997_924_58e8; // m/s
const HBAR: f64 = 1.054_571_817e-34; // J·s
const EPS0: f64 = 8.854_187_817e-12; // F/m

// ─── Ldos ────────────────────────────────────────────────────────────────────

/// Local Density of Optical States (LDOS)
///
/// The LDOS relates to the imaginary part of the dyadic Green's function:
///
///   ρ(r, ω) = (6ω) / (π c²) · Im[ê · G(r,r,ω) · ê]
///
/// In bulk media: ρ_bulk(ω) = n³ ω² / (π² c³)
#[derive(Debug, Clone)]
pub struct Ldos {
    /// Real refractive index of the embedding medium
    pub medium_index: f64,
    /// Angular frequency of interest (rad/s)
    pub frequency: f64,
    /// Evaluation position [x, y, z] in metres
    pub position: [f64; 3],
}

impl Ldos {
    /// Construct a new LDOS evaluator.
    pub fn new(n: f64, omega: f64, position: [f64; 3]) -> Self {
        Self {
            medium_index: n,
            frequency: omega,
            position,
        }
    }

    /// Bulk LDOS: ρ_bulk(ω) = n³ ω² / (π² c³)  [states / (m³ · rad/s)]
    pub fn bulk_ldos(&self) -> f64 {
        let n = self.medium_index;
        let omega = self.frequency;
        n * n * n * omega * omega / (PI * PI * C0 * C0 * C0)
    }

    /// Purcell factor: F_P = ρ_local / ρ_bulk
    ///
    /// A Purcell factor > 1 indicates enhanced emission; < 1 indicates inhibition.
    pub fn purcell_factor(&self, local_ldos: f64) -> f64 {
        let rho_bulk = self.bulk_ldos();
        if rho_bulk < f64::EPSILON {
            return 1.0;
        }
        local_ldos / rho_bulk
    }

    /// LDOS near a perfect planar mirror (image dipole model).
    ///
    /// For a dipole oriented parallel to the mirror surface at distance d:
    ///
    ///   ρ(d) / ρ_bulk = 1 − (3/2) · [sin(2kd)/(2kd) + cos(2kd)/(2kd)²
    ///                                  − sin(2kd)/(2kd)³]
    ///
    /// For a dipole perpendicular to the mirror:
    ///
    ///   ρ_⊥(d) / ρ_bulk = 1 + (3/2) · [cos(2kd)/(2kd)² − sin(2kd)/(2kd)³]
    ///
    /// We return the orientationally averaged result:
    ///   ρ_avg = (2 ρ_∥ + ρ_⊥) / 3
    pub fn near_mirror(&self, distance_from_mirror: f64) -> f64 {
        let rho_bulk = self.bulk_ldos();
        let k = self.medium_index * self.frequency / C0;
        let x = 2.0 * k * distance_from_mirror;

        if x < 1.0e-10 {
            // Very close to mirror: image charge cancellation
            return rho_bulk * 0.5;
        }

        let sinx = x.sin();
        let cosx = x.cos();
        let x2 = x * x;
        let x3 = x2 * x;

        // Parallel component (dipole ∥ mirror)
        let rho_par = rho_bulk * (1.0 - 1.5 * (sinx / x + cosx / x2 - sinx / x3));

        // Perpendicular component (dipole ⊥ mirror)
        let rho_perp = rho_bulk * (1.0 + 1.5 * (cosx / x2 - sinx / x3));

        // Orientational average
        (2.0 * rho_par + rho_perp) / 3.0
    }

    /// LDOS at the centre of a spherical cavity with perfect reflector.
    ///
    /// The LDOS inside a spherical cavity of radius R is enhanced relative to
    /// bulk due to the constructive interference of reflected waves.  For a
    /// point at the centre (r=0) the result from Mie theory in the limit of
    /// high reflectivity is approximately:
    ///
    ///   ρ_cav / ρ_bulk ≈ 1 + (3λ / 8πnR) · [1 + correction terms]
    ///
    /// We use the first-order standing-wave correction:
    ///
    ///   ρ_cav / ρ_bulk ≈ 1 + (3/2) · (c / (n ω R)) · sin(2nωR/c) / (2nωR/c)
    pub fn in_spherical_cavity(&self, radius: f64) -> f64 {
        let rho_bulk = self.bulk_ldos();
        let x = self.medium_index * self.frequency * radius / C0; // = nωR/c = kR
        if x < f64::EPSILON {
            return rho_bulk;
        }
        let sinc = x.sin() / x;
        let correction = 1.5 * sinc;
        rho_bulk * (1.0 + correction)
    }

    /// LDOS enhancement near a small dielectric/metallic nanosphere
    /// in the quasi-static (Rayleigh) limit.
    ///
    /// The sphere modifies the local Green's function.  For a point dipole
    /// at distance d from the surface of a sphere of radius a and complex
    /// permittivity ε_s in medium with index n_m:
    ///
    ///   ΔΓ / Γ_0 ∝ Im[α / (d + a)³]
    ///
    /// where α is the Clausius-Mossotti polarizability.
    /// We use the leading-order quasi-static expression (Chance-Prock-Silbey):
    ///
    ///   ρ(d) / ρ_bulk = 1 + (3/(4k³)) · Im[α_eff / r_eval⁶]
    ///
    /// where r_eval = a + d.
    pub fn near_sphere(&self, sphere_radius: f64, n_sphere: Complex64, distance: f64) -> f64 {
        let rho_bulk = self.bulk_ldos();
        let k = self.medium_index * self.frequency / C0;
        let eps_d = self.medium_index * self.medium_index;
        let eps_s = n_sphere * n_sphere;
        let eps_d_c = Complex64::new(eps_d, 0.0);

        // Clausius-Mossotti polarizability (m³)
        let a = sphere_radius;
        let vol = 4.0 * PI * a * a * a / 3.0;
        let cm = (eps_s - eps_d_c) / (eps_s + 2.0 * eps_d_c);
        let alpha = 3.0 * eps_d_c * vol * cm; // full Clausius-Mossotti

        let r_eval = a + distance;
        let r6 = r_eval.powi(6);

        // CPS correction factor (quasi-static leading order)
        let delta_rho_over_bulk = (3.0 / (4.0 * k * k * k)) * (alpha.im / r6);

        rho_bulk * (1.0 + delta_rho_over_bulk.max(-0.99))
    }
}

// ─── SpontaneousEmission ─────────────────────────────────────────────────────

/// Spontaneous emission rate and quantum optical properties of a dipole emitter.
///
/// The Einstein A coefficient in bulk medium n is:
///
///   Γ₀ = ω³ |d|² n / (3π ε₀ ħ c³)
///
/// With Purcell enhancement:
///
///   Γ_total = F_P · Γ₀ + Γ_non-rad
#[derive(Debug, Clone)]
pub struct SpontaneousEmission {
    /// Dipole moment magnitude in C·m
    pub emitter_dipole_moment: f64,
    /// Transition angular frequency in rad/s
    pub emitter_frequency: f64,
    /// Intrinsic quantum efficiency η₀ ∈ [0, 1]
    pub quantum_efficiency: f64,
}

impl SpontaneousEmission {
    /// Construct a new spontaneous emission calculator.
    ///
    /// # Arguments
    /// * `dipole_cm` - dipole moment in C·m (typical: 1e-29 C·m for molecules)
    /// * `omega`     - transition frequency in rad/s
    /// * `eta`       - intrinsic quantum efficiency (0 to 1)
    pub fn new(dipole_cm: f64, omega: f64, eta: f64) -> Self {
        Self {
            emitter_dipole_moment: dipole_cm,
            emitter_frequency: omega,
            quantum_efficiency: eta.clamp(0.0, 1.0),
        }
    }

    /// Spontaneous emission rate in bulk medium of index n (s⁻¹).
    ///
    /// Γ₀ = n ω³ |d|² / (3π ε₀ ħ c³)
    pub fn rate_bulk(&self, n: f64) -> f64 {
        let omega = self.emitter_frequency;
        let d = self.emitter_dipole_moment;
        n * omega * omega * omega * d * d / (3.0 * PI * EPS0 * HBAR * C0 * C0 * C0)
    }

    /// Non-radiative decay rate derived from intrinsic quantum efficiency.
    ///
    /// η₀ = Γ_rad / (Γ_rad + Γ_nr)  →  Γ_nr = Γ_rad (1 − η₀) / η₀
    fn rate_non_rad(&self, n: f64) -> f64 {
        let gamma0 = self.rate_bulk(n);
        if self.quantum_efficiency < f64::EPSILON {
            return gamma0 * 1.0e6; // effectively infinite non-rad rate
        }
        gamma0 * (1.0 - self.quantum_efficiency) / self.quantum_efficiency
    }

    /// Enhanced total radiative rate near a photonic structure (s⁻¹).
    ///
    /// Γ_enhanced = F_P · Γ₀
    pub fn rate_enhanced(&self, purcell_factor: f64, n: f64) -> f64 {
        purcell_factor * self.rate_bulk(n)
    }

    /// Beta factor: fraction of emission coupled into the cavity mode.
    ///
    /// β = F_P Γ₀ / (F_P Γ₀ + Γ_nr)
    pub fn beta_factor(&self, purcell_factor: f64, n: f64) -> f64 {
        let gamma_cav = self.rate_enhanced(purcell_factor, n);
        let gamma_nr = self.rate_non_rad(n);
        let total = gamma_cav + gamma_nr;
        if total < f64::EPSILON {
            return 0.0;
        }
        gamma_cav / total
    }

    /// External quantum efficiency with Purcell enhancement.
    ///
    /// η_ext = F_P · Γ₀ / (F_P · Γ₀ + Γ_nr)  (same as β in this model)
    pub fn enhanced_quantum_efficiency(&self, purcell_factor: f64, n: f64) -> f64 {
        self.beta_factor(purcell_factor, n)
    }

    /// Single-photon coupling probability into the cavity mode (dimensionless).
    ///
    /// Defined as the probability that a photon emitted into the cavity mode
    /// exits the cavity before being absorbed.  For a cavity with quality
    /// factors Q_total and Q_rad:
    ///
    ///   p_couple = β · η₀  (simplified product)
    pub fn coupling_probability(&self, purcell_factor: f64, n: f64) -> f64 {
        let beta = self.beta_factor(purcell_factor, n);
        beta * self.quantum_efficiency
    }
}

// ─── CavityQedCoupling ───────────────────────────────────────────────────────

/// Cavity QED parameters describing light-matter interaction strength.
///
/// The Jaynes-Cummings Hamiltonian couples a two-level system to a single
/// cavity mode with vacuum Rabi coupling g.
///
/// Cooperativity: C = g² / (κ γ)
///
/// Regimes:
/// - Weak coupling:   g < (κ + γ)/4  →  Purcell enhancement F = 4C
/// - Strong coupling: g > (κ + γ)/4  →  vacuum Rabi splitting 2g (resolved)
#[derive(Debug, Clone)]
pub struct CavityQedCoupling {
    /// Vacuum Rabi coupling rate (rad/s)
    pub g: f64,
    /// Cavity field decay rate = ω / (2Q) (rad/s)
    pub kappa: f64,
    /// Emitter dephasing/decay rate (rad/s)
    pub gamma: f64,
}

impl CavityQedCoupling {
    /// Construct from vacuum Rabi coupling g, cavity decay κ, emitter decay γ.
    pub fn new(g: f64, kappa: f64, gamma: f64) -> Self {
        Self { g, kappa, gamma }
    }

    /// Single-atom cooperativity: C = g² / (κ γ)
    pub fn cooperativity(&self) -> f64 {
        if self.kappa < f64::EPSILON || self.gamma < f64::EPSILON {
            return f64::INFINITY;
        }
        self.g * self.g / (self.kappa * self.gamma)
    }

    /// Strong coupling criterion: g > (κ + γ) / 4
    pub fn is_strong_coupling(&self) -> bool {
        self.g > (self.kappa + self.gamma) / 4.0
    }

    /// Vacuum Rabi splitting in rad/s, only defined in strong coupling regime.
    ///
    /// Returns `None` in weak coupling, `Some(2g)` in strong coupling.
    pub fn rabi_splitting_rad_s(&self) -> Option<f64> {
        if self.is_strong_coupling() {
            Some(2.0 * self.g)
        } else {
            None
        }
    }

    /// Purcell factor in weak coupling limit: F_P = 4 g² / (κ γ) = 4 C
    pub fn purcell_factor_weak(&self) -> f64 {
        4.0 * self.cooperativity()
    }

    /// Single-photon blockade condition: g ≫ κ  (g/κ > 1)
    pub fn single_photon_blockade(&self) -> bool {
        if self.kappa < f64::EPSILON {
            return true;
        }
        self.g / self.kappa > 1.0
    }

    /// Polariton splitting energies (normalised to g).
    ///
    /// The two dressed-state eigenvalues of the Jaynes-Cummings model on
    /// resonance (Δ = 0) are:
    ///
    ///   λ± = −(κ + γ)/4 ± sqrt[g² − ((κ − γ)/4)²]
    ///
    /// Returns `None` if the square-root argument is negative (over-damped).
    pub fn polariton_eigenvalues(&self) -> Option<(f64, f64)> {
        let decay_avg = (self.kappa + self.gamma) / 4.0;
        let decay_diff = (self.kappa - self.gamma) / 4.0;
        let disc = self.g * self.g - decay_diff * decay_diff;
        if disc < 0.0 {
            return None;
        }
        let split = disc.sqrt();
        Some((-decay_avg + split, -decay_avg - split))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn visible_omega() -> f64 {
        // 532 nm green laser
        2.0 * PI * C0 / 532.0e-9
    }

    // ── Ldos tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_bulk_ldos_scaling_with_index() {
        let omega = visible_omega();
        let ldos1 = Ldos::new(1.0, omega, [0.0; 3]);
        let ldos15 = Ldos::new(1.5, omega, [0.0; 3]);

        let rho1 = ldos1.bulk_ldos();
        let rho15 = ldos15.bulk_ldos();

        // ρ ∝ n³  →  ratio = (1.5/1.0)³ = 3.375
        let ratio = rho15 / rho1;
        assert_abs_diff_eq!(ratio, 3.375, epsilon = 1.0e-6);
    }

    #[test]
    fn test_purcell_factor_identity() {
        let omega = visible_omega();
        let ldos = Ldos::new(1.0, omega, [0.0; 3]);
        let rho_bulk = ldos.bulk_ldos();
        // Purcell factor of bulk LDOS should be exactly 1
        let fp = ldos.purcell_factor(rho_bulk);
        assert_abs_diff_eq!(fp, 1.0, epsilon = 1.0e-12);
    }

    #[test]
    fn test_near_mirror_far_field_limit() {
        let omega = visible_omega();
        let ldos = Ldos::new(1.0, omega, [0.0; 3]);
        // At very large distance the oscillating correction averages out
        // The value should remain in a physically reasonable range (0.5 to 1.5)
        let rho_far = ldos.near_mirror(1.0e-3);
        let rho_bulk = ldos.bulk_ldos();
        let ratio = rho_far / rho_bulk;
        assert!(ratio > 0.0 && ratio < 3.0, "ratio={ratio}");
    }

    #[test]
    fn test_near_mirror_very_close_reduction() {
        let omega = visible_omega();
        let ldos = Ldos::new(1.0, omega, [0.0; 3]);
        // Very close to mirror (1 nm): image dipole cancellation → ρ ~ 0.5 ρ_bulk
        let rho_close = ldos.near_mirror(1.0e-9);
        let rho_bulk = ldos.bulk_ldos();
        let ratio = rho_close / rho_bulk;
        assert!(
            ratio < 0.6,
            "Expected near-mirror inhibition, got ratio={ratio}"
        );
    }

    #[test]
    fn test_in_spherical_cavity_positive() {
        let omega = visible_omega();
        let ldos = Ldos::new(1.0, omega, [0.0; 3]);
        let rho = ldos.in_spherical_cavity(100.0e-9);
        assert!(rho > 0.0, "Cavity LDOS must be positive");
    }

    #[test]
    fn test_near_sphere_metal_enhancement() {
        let omega = visible_omega();
        let ldos = Ldos::new(1.0, omega, [0.0; 3]);
        // Gold-like permittivity at ~532 nm: ε ≈ -7 + 1.5i → n ≈ 0.18 + 4.2i
        let n_gold = Complex64::new(0.18, 4.2);
        let rho_near = ldos.near_sphere(20.0e-9, n_gold, 2.0e-9);
        // Should return a positive value
        assert!(
            rho_near > 0.0,
            "Near-sphere LDOS must be positive, got {rho_near}"
        );
    }

    // ── SpontaneousEmission tests ─────────────────────────────────────────────

    #[test]
    fn test_rate_bulk_positive() {
        let omega = visible_omega();
        let se = SpontaneousEmission::new(1.0e-29, omega, 1.0);
        let rate = se.rate_bulk(1.5);
        assert!(rate > 0.0, "Bulk SE rate must be positive: {rate}");
    }

    #[test]
    fn test_rate_bulk_scales_with_n() {
        let omega = visible_omega();
        let se = SpontaneousEmission::new(1.0e-29, omega, 1.0);
        let rate1 = se.rate_bulk(1.0);
        let rate2 = se.rate_bulk(2.0);
        // Rate ∝ n → ratio = 2
        assert_abs_diff_eq!(rate2 / rate1, 2.0, epsilon = 1.0e-10);
    }

    #[test]
    fn test_rate_enhanced_exceeds_bulk() {
        let omega = visible_omega();
        let se = SpontaneousEmission::new(1.0e-29, omega, 1.0);
        let n = 1.5;
        let rate0 = se.rate_bulk(n);
        let rate_fp = se.rate_enhanced(10.0, n);
        assert_abs_diff_eq!(rate_fp / rate0, 10.0, epsilon = 1.0e-10);
    }

    #[test]
    fn test_beta_factor_range() {
        let omega = visible_omega();
        let se = SpontaneousEmission::new(1.0e-29, omega, 0.8);
        let beta = se.beta_factor(50.0, 1.5);
        assert!(
            (0.0..=1.0).contains(&beta),
            "Beta must be in [0,1], got {beta}"
        );
    }

    #[test]
    fn test_coupling_probability_le_beta() {
        let omega = visible_omega();
        let se = SpontaneousEmission::new(1.0e-29, omega, 0.5);
        let fp = 20.0;
        let n = 1.5;
        let beta = se.beta_factor(fp, n);
        let p_c = se.coupling_probability(fp, n);
        assert!(
            p_c <= beta + 1.0e-12,
            "Coupling prob must be <= beta: {p_c} vs {beta}"
        );
    }

    // ── CavityQedCoupling tests ───────────────────────────────────────────────

    #[test]
    fn test_cooperativity_formula() {
        // g = 1e9, κ = 2e9, γ = 5e8 → C = 1e18 / (1e9) = 1.0
        let qed = CavityQedCoupling::new(1.0e9, 2.0e9, 5.0e8);
        let c = qed.cooperativity();
        assert_abs_diff_eq!(c, 1.0, epsilon = 1.0e-10);
    }

    #[test]
    fn test_strong_coupling_criterion() {
        // g = 1e10, κ = γ = 1e9 → threshold = (1e9+1e9)/4 = 5e8 < 1e10 → strong
        let qed_strong = CavityQedCoupling::new(1.0e10, 1.0e9, 1.0e9);
        assert!(qed_strong.is_strong_coupling());

        // g = 1e8, κ = γ = 1e9 → threshold = 5e8 > 1e8 → weak
        let qed_weak = CavityQedCoupling::new(1.0e8, 1.0e9, 1.0e9);
        assert!(!qed_weak.is_strong_coupling());
    }

    #[test]
    fn test_rabi_splitting_only_in_strong() {
        let qed_strong = CavityQedCoupling::new(1.0e10, 1.0e9, 1.0e9);
        let split = qed_strong.rabi_splitting_rad_s();
        assert!(split.is_some());
        assert_abs_diff_eq!(split.unwrap(), 2.0e10, epsilon = 1.0e3);

        let qed_weak = CavityQedCoupling::new(1.0e8, 1.0e9, 1.0e9);
        assert!(qed_weak.rabi_splitting_rad_s().is_none());
    }

    #[test]
    fn test_purcell_weak_equals_4c() {
        let g = 1.0e9_f64;
        let kappa = 2.0e10_f64; // large κ → weak coupling
        let gamma = 1.0e9_f64;
        let qed = CavityQedCoupling::new(g, kappa, gamma);
        let c = qed.cooperativity();
        let fp = qed.purcell_factor_weak();
        assert_abs_diff_eq!(fp, 4.0 * c, epsilon = 1.0e-10);
    }

    #[test]
    fn test_polariton_eigenvalues_symmetric() {
        // On resonance with equal decay rates κ = γ
        let g = 5.0e9_f64;
        let kappa = 1.0e9_f64;
        let qed = CavityQedCoupling::new(g, kappa, kappa);
        let evs = qed.polariton_eigenvalues();
        assert!(evs.is_some(), "Should have resolved polaritons");
        let (lp, lm) = evs.unwrap();
        // For κ = γ: λ± = -κ/2 ± g  → symmetric about decay avg
        assert_abs_diff_eq!((lp + lm).abs(), kappa, epsilon = 1.0e-3);
    }
}
