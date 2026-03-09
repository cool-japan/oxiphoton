/// Soliton dynamics in optical fibers.
///
/// Covers fundamental solitons, higher-order solitons, soliton trapping via XPM,
/// and the Peregrine (rogue-wave) soliton.  All formulas follow Agrawal,
/// "Nonlinear Fiber Optics" (6th ed.) unless otherwise noted.
///
/// Sign convention: anomalous GVD → β₂ < 0.
use num_complex::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// FundamentalSoliton
// ---------------------------------------------------------------------------

/// Analytical fundamental soliton of the NLSE.
///
/// The field envelope (in the retarded frame) is:
/// ```text
///   ψ(z, t) = √P₀ · sech(t/T₀) · exp(i·z/(2·z₀))
/// ```
/// where P₀ = |β₂| / (γ T₀²) is the soliton peak power and
/// z₀ = π T₀² / (2|β₂|) is the soliton period.
#[derive(Debug, Clone)]
pub struct FundamentalSoliton {
    /// Peak power P₀ (W).
    pub peak_power: f64,
    /// 1/e half-width T₀ (s).  Related to FWHM by T_FWHM = 2·T₀·ln(1+√2).
    pub pulse_width: f64,
    /// Group-velocity dispersion β₂ (s²/m).  Must be negative for solitons.
    pub beta2: f64,
    /// Nonlinear coefficient γ (1/(W·m)).
    pub gamma: f64,
    /// Centre wavelength λ₀ (m).
    pub wavelength: f64,
}

impl FundamentalSoliton {
    /// Construct a fundamental soliton and compute the required peak power
    /// from the soliton condition N=1: P₀ = |β₂| / (γ T₀²).
    ///
    /// # Panics
    /// Panics if `gamma` ≤ 0 or `t0` ≤ 0 (non-physical parameters).
    pub fn new(t0: f64, beta2: f64, gamma: f64, wavelength: f64) -> Self {
        assert!(gamma > 0.0, "FundamentalSoliton: gamma must be positive");
        assert!(
            t0 > 0.0,
            "FundamentalSoliton: pulse_width (T0) must be positive"
        );
        let peak_power = beta2.abs() / (gamma * t0 * t0);
        Self {
            peak_power,
            pulse_width: t0,
            beta2,
            gamma,
            wavelength,
        }
    }

    /// Soliton peak power P₀ = |β₂| / (γ T₀²) (W).
    #[inline]
    pub fn soliton_power(&self) -> f64 {
        self.beta2.abs() / (self.gamma * self.pulse_width * self.pulse_width)
    }

    /// Soliton period z₀ = π T₀² / (2|β₂|) (m).
    ///
    /// The field returns to its initial profile at z = 2·z₀.
    #[inline]
    pub fn soliton_period(&self) -> f64 {
        PI * self.pulse_width * self.pulse_width / (2.0 * self.beta2.abs())
    }

    /// Complex field envelope ψ(z, t) = √P₀ · sech(t/T₀) · exp(i·z/(2·z₀)).
    ///
    /// Returns the slowly-varying envelope in units of √W.
    pub fn field(&self, z: f64, t: f64) -> Complex64 {
        let z0 = self.soliton_period();
        let amplitude = self.peak_power.sqrt();
        let sech = 1.0 / (t / self.pulse_width).cosh();
        let phase = z / (2.0 * z0);
        amplitude * sech * Complex64::new(0.0, phase).exp()
    }

    /// Soliton order N = √(γ P₀ T₀² / |β₂|).
    ///
    /// N = 1 for a fundamental soliton by construction.  Non-unity values
    /// indicate the stored `peak_power` was set externally.
    pub fn soliton_number(&self) -> f64 {
        let n_sq =
            self.gamma * self.peak_power * self.pulse_width * self.pulse_width / self.beta2.abs();
        n_sq.sqrt()
    }

    /// Raman self-frequency shift (SSFS) rate dΩ/dz (rad/s/m).
    ///
    /// The intrapulse Raman scattering continuously red-shifts the soliton.
    /// The rate is (Agrawal §5.5):
    /// ```text
    ///   dΩ/dz = −8 T_R |β₂| / (15 T₀⁴)
    /// ```
    /// where T_R ≈ 3 fs is the first moment of the Raman response.
    ///
    /// Returns a negative value (red shift).
    pub fn raman_frequency_shift_rate(&self, raman_time: f64) -> f64 {
        -8.0 * raman_time * self.beta2.abs() / (15.0 * self.pulse_width.powi(4))
    }

    /// Phase-matched dispersive-wave (Cherenkov radiation) detuning Δω (rad/s).
    ///
    /// The dispersive wave is emitted at the frequency offset that satisfies the
    /// phase-matching condition β(ω₀ + Δω) = γ P₀ / 2, expanded to third order:
    /// ```text
    ///   β₂ Δω²/2 + β₃ Δω³/6 = γ P₀ / 2
    /// ```
    /// Solving for the dominant β₃ root (Agrawal §12.1):
    /// ```text
    ///   Δω ≈ −3 β₂ / β₃   (leading-order approximation)
    /// ```
    ///
    /// Returns 0 if `beta3` is too small to yield a meaningful root.
    pub fn dispersive_wave_frequency(&self, beta3: f64) -> f64 {
        if beta3.abs() < 1e-60 {
            return 0.0;
        }
        // Leading-order solution of β₂ Δω/2 + β₃ Δω²/6 = γP₀/2
        // Δω ≈ -3β₂/β₃ − correction from nonlinear term
        let delta_omega_linear = -3.0 * self.beta2 / beta3;
        // Nonlinear correction: δ = 3 γ P₀ / (β₃ Δω²)
        let correction = if delta_omega_linear.abs() > 1e-10 {
            3.0 * self.gamma * self.peak_power / (beta3 * delta_omega_linear * delta_omega_linear)
        } else {
            0.0
        };
        delta_omega_linear - correction
    }
}

// ---------------------------------------------------------------------------
// HigherOrderSoliton
// ---------------------------------------------------------------------------

/// N-soliton (higher-order soliton) that breathes with period z₀ and undergoes
/// fission into N fundamental solitons after the fission distance L_fiss.
#[derive(Debug, Clone)]
pub struct HigherOrderSoliton {
    /// Soliton order N (≥ 2 for higher-order).
    pub soliton_number: u32,
    /// 1/e half-width of the initial pulse T₀ (s).
    pub t0: f64,
    /// Group-velocity dispersion β₂ (s²/m).  Must be negative (anomalous).
    pub beta2: f64,
    /// Nonlinear coefficient γ (1/(W·m)).
    pub gamma: f64,
}

impl HigherOrderSoliton {
    /// Create a higher-order soliton with order `n`, pulse width `t0`, GVD `beta2`,
    /// and nonlinear coefficient `gamma`.
    ///
    /// # Panics
    /// Panics if `n` = 0, `gamma` ≤ 0, or `t0` ≤ 0.
    pub fn new(n: u32, t0: f64, beta2: f64, gamma: f64) -> Self {
        assert!(n > 0, "HigherOrderSoliton: soliton_number must be ≥ 1");
        assert!(gamma > 0.0, "HigherOrderSoliton: gamma must be positive");
        assert!(t0 > 0.0, "HigherOrderSoliton: t0 must be positive");
        Self {
            soliton_number: n,
            t0,
            beta2,
            gamma,
        }
    }

    /// Required peak power P₀ = N² |β₂| / (γ T₀²) (W).
    pub fn peak_power(&self) -> f64 {
        let n = self.soliton_number as f64;
        n * n * self.beta2.abs() / (self.gamma * self.t0 * self.t0)
    }

    /// Soliton period z₀ = π T₀² / (2|β₂|) (m).
    ///
    /// All N constituent solitons share the same z₀; the envelope repeats at z₀.
    pub fn soliton_period(&self) -> f64 {
        PI * self.t0 * self.t0 / (2.0 * self.beta2.abs())
    }

    /// Fission distance L_fiss ≈ z₀ / N (m).
    ///
    /// At this distance higher-order effects (Raman, TOD) cause the soliton to
    /// break apart into N fundamental solitons (Agrawal §13.4).
    pub fn fission_distance(&self) -> f64 {
        self.soliton_period() / self.soliton_number as f64
    }

    /// The N fundamental solitons produced after fission.
    ///
    /// After break-up the k-th soliton (k = 1 … N) has pulse width
    /// T_k = T₀ / (2N − 2k + 1).  The shortest, most energetic soliton
    /// is k = N; it carries the most Raman shift (Agrawal §13.4).
    ///
    /// Each product soliton inherits the same β₂, γ, and wavelength
    /// (1550 nm as a placeholder — the caller should refine this).
    pub fn fission_products(&self) -> Vec<FundamentalSoliton> {
        let n = self.soliton_number;
        (1..=n)
            .map(|k| {
                let width_factor = 2 * n - 2 * k + 1;
                let tk = self.t0 / width_factor as f64;
                FundamentalSoliton::new(tk, self.beta2, self.gamma, 1550e-9)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// SolitonTrap
// ---------------------------------------------------------------------------

/// Cross-phase modulation (XPM) trapping between two co-propagating solitons.
///
/// When two fundamental solitons at different wavelengths propagate together,
/// XPM can lock their group velocities, creating a "soliton molecule".
/// The trapping is most effective when both solitons have the same order (N=1)
/// and the temporal separation is smaller than a few T₀.
#[derive(Debug, Clone)]
pub struct SolitonTrap {
    /// First soliton.
    pub soliton1: FundamentalSoliton,
    /// Second soliton.
    pub soliton2: FundamentalSoliton,
    /// Temporal separation Δt between the two soliton centres (s).
    pub temporal_separation: f64,
}

impl SolitonTrap {
    /// Construct a soliton trap from two fundamental solitons and their
    /// temporal separation.
    pub fn new(s1: FundamentalSoliton, s2: FundamentalSoliton, dt: f64) -> Self {
        Self {
            soliton1: s1,
            soliton2: s2,
            temporal_separation: dt,
        }
    }

    /// XPM attraction force magnitude (approximate, W/m units).
    ///
    /// The restoring force on soliton 2 due to the XPM potential of soliton 1
    /// is estimated as (Agrawal §7.4):
    /// ```text
    ///   F_xpm ≈ 2γ P₁ · sech²(Δt/T₀) · tanh(Δt/T₀) / T₀
    /// ```
    /// This is a first-moment approximation of the XPM gradient force.
    pub fn xpm_attraction_force(&self) -> f64 {
        let s1 = &self.soliton1;
        let s2 = &self.soliton2;
        // Average γ and T₀
        let gamma_eff = 0.5 * (s1.gamma + s2.gamma);
        let t0_eff = 0.5 * (s1.pulse_width + s2.pulse_width);
        let xi = self.temporal_separation / t0_eff;
        let sech2 = 1.0 / xi.cosh().powi(2);
        let tanh_val = xi.tanh();
        2.0 * gamma_eff * s1.peak_power * sech2 * tanh_val.abs() / t0_eff
    }

    /// Returns `true` when trapping conditions are met.
    ///
    /// Conditions (Agrawal §7.4):
    /// 1. Both solitons are fundamental (N ≈ 1).
    /// 2. Both have anomalous GVD (β₂ < 0).
    /// 3. Temporal separation |Δt| < 5 T₀ (overlap threshold).
    pub fn trapping_condition(&self) -> bool {
        let n1 = self.soliton1.soliton_number();
        let n2 = self.soliton2.soliton_number();
        let anomalous1 = self.soliton1.beta2 < 0.0;
        let anomalous2 = self.soliton2.beta2 < 0.0;
        let t0_avg = 0.5 * (self.soliton1.pulse_width + self.soliton2.pulse_width);
        let overlap = self.temporal_separation.abs() < 5.0 * t0_avg;
        (n1 - 1.0).abs() < 0.2 && (n2 - 1.0).abs() < 0.2 && anomalous1 && anomalous2 && overlap
    }
}

// ---------------------------------------------------------------------------
// PeregineSoliton (Peregrine / rogue-wave soliton)
// ---------------------------------------------------------------------------

/// Peregrine soliton — rational solution of the focusing NLSE on a continuous
/// wave background.  It is a model for optical rogue waves (freak waves on a
/// finite-power background).
///
/// The solution is localised in both space and time, and reaches a peak
/// amplitude of 3 × the background, making it the most dramatic amplification
/// possible in the scalar NLSE.
///
/// Reference: D. H. Peregrine, J. Aust. Math. Soc. Ser. B 25, 16 (1983).
#[derive(Debug, Clone)]
pub struct PeregineSoliton {
    /// Background field amplitude a₀ (√W).  The CW background power is a₀².
    pub background_amplitude: f64,
    /// Group-velocity dispersion β₂ (s²/m).  Must be negative (focusing NLSE).
    pub beta2: f64,
    /// Nonlinear coefficient γ (1/(W·m)).
    pub gamma: f64,
}

impl PeregineSoliton {
    /// Construct a Peregrine soliton on a CW background with amplitude `a0`.
    ///
    /// # Panics
    /// Panics if `gamma` ≤ 0 or `a0` ≤ 0.
    pub fn new(a0: f64, beta2: f64, gamma: f64) -> Self {
        assert!(
            a0 > 0.0,
            "PeregineSoliton: background_amplitude must be positive"
        );
        assert!(gamma > 0.0, "PeregineSoliton: gamma must be positive");
        Self {
            background_amplitude: a0,
            beta2,
            gamma,
        }
    }

    /// Complex field envelope ψ(z, t) of the Peregrine soliton.
    ///
    /// In normalised coordinates (Agrawal §A.3 / Peregrine 1983):
    /// ```text
    ///   ψ(z, t) = a₀ · [1 − 4(1 + 2iζ)/(1 + 4τ² + 4ζ²)] · exp(i γ a₀² z)
    /// ```
    /// where ζ = γ a₀² z and τ = t √(2γ a₀² / |β₂|) are dimensionless.
    pub fn field(&self, z: f64, t: f64) -> Complex64 {
        let a0 = self.background_amplitude;
        let zeta = self.gamma * a0 * a0 * z;
        let tau_scale = (2.0 * self.gamma * a0 * a0 / self.beta2.abs()).sqrt();
        let tau = t * tau_scale;
        let denom = 1.0 + 4.0 * tau * tau + 4.0 * zeta * zeta;
        let numerator = Complex64::new(4.0, -8.0 * zeta); // 4(1 + 2iζ) with conjugated sign
        let rational_part = Complex64::new(1.0, 0.0) - Complex64::new(4.0, 8.0 * zeta) / denom;
        let _ = numerator; // computed inline above
        let background_phase = Complex64::new(0.0, self.gamma * a0 * a0 * z).exp();
        a0 * rational_part * background_phase
    }

    /// Peak amplitude of the Peregrine soliton: 3 a₀ (at z = 0, t = 0).
    pub fn peak_amplitude(&self) -> f64 {
        3.0 * self.background_amplitude
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── FundamentalSoliton ───────────────────────────────────────────────────

    #[test]
    fn fundamental_soliton_power_matches_condition() {
        // N=1 ⟹ P₀ = |β₂| / (γ T₀²)
        let t0 = 1e-12;
        let beta2 = -20e-27; // -20 ps²/km in SI
        let gamma = 1.3e-3;
        let sol = FundamentalSoliton::new(t0, beta2, gamma, 1550e-9);
        let expected_power = beta2.abs() / (gamma * t0 * t0);
        assert_abs_diff_eq!(sol.soliton_power(), expected_power, epsilon = 1e-10);
    }

    #[test]
    fn fundamental_soliton_number_is_one() {
        let sol = FundamentalSoliton::new(1e-12, -20e-27, 1.3e-3, 1550e-9);
        assert_abs_diff_eq!(sol.soliton_number(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn fundamental_soliton_period_positive() {
        let sol = FundamentalSoliton::new(1e-12, -20e-27, 1.3e-3, 1550e-9);
        assert!(sol.soliton_period() > 0.0);
    }

    #[test]
    fn fundamental_soliton_field_at_z0_t0() {
        // At z=0, t=0: field = √P₀ · sech(0) · exp(0) = √P₀
        let sol = FundamentalSoliton::new(1e-12, -20e-27, 1.3e-3, 1550e-9);
        let f = sol.field(0.0, 0.0);
        assert_abs_diff_eq!(f.norm(), sol.peak_power.sqrt(), epsilon = 1e-14);
    }

    #[test]
    fn fundamental_soliton_raman_shift_is_negative() {
        let sol = FundamentalSoliton::new(1e-12, -20e-27, 1.3e-3, 1550e-9);
        let rate = sol.raman_frequency_shift_rate(3e-15); // T_R ≈ 3 fs
        assert!(
            rate < 0.0,
            "SSFS rate must be negative (red shift), got {rate}"
        );
    }

    #[test]
    fn fundamental_soliton_dispersive_wave_zero_beta3() {
        let sol = FundamentalSoliton::new(1e-12, -20e-27, 1.3e-3, 1550e-9);
        assert_abs_diff_eq!(sol.dispersive_wave_frequency(0.0), 0.0, epsilon = 1e-30);
    }

    #[test]
    fn fundamental_soliton_dispersive_wave_nonzero_beta3() {
        let sol = FundamentalSoliton::new(1e-12, -20e-27, 1.3e-3, 1550e-9);
        let beta3 = 0.1e-39; // typical silica TOD
        let dw = sol.dispersive_wave_frequency(beta3);
        // Leading-order: Δω ≈ -3β₂/β₃ > 0 (β₂ < 0 ⟹ positive Δω)
        assert!(dw.is_finite(), "dispersive_wave_frequency must be finite");
    }

    // ── HigherOrderSoliton ───────────────────────────────────────────────────

    #[test]
    fn higher_order_soliton_peak_power() {
        let n = 3u32;
        let t0 = 1e-12;
        let beta2 = -20e-27;
        let gamma = 1.3e-3;
        let hos = HigherOrderSoliton::new(n, t0, beta2, gamma);
        let expected = (n as f64).powi(2) * beta2.abs() / (gamma * t0 * t0);
        assert_abs_diff_eq!(hos.peak_power(), expected, epsilon = 1e-10);
    }

    #[test]
    fn higher_order_soliton_fission_distance_less_than_period() {
        let hos = HigherOrderSoliton::new(3, 1e-12, -20e-27, 1.3e-3);
        let z0 = hos.soliton_period();
        let lfiss = hos.fission_distance();
        // L_fiss = z₀/N < z₀ for N > 1
        assert!(
            lfiss < z0,
            "fission distance {lfiss:.3e} must be < soliton period {z0:.3e}"
        );
    }

    #[test]
    fn higher_order_soliton_fission_products_count() {
        let n = 4u32;
        let hos = HigherOrderSoliton::new(n, 1e-12, -20e-27, 1.3e-3);
        let products = hos.fission_products();
        assert_eq!(
            products.len(),
            n as usize,
            "should produce N fundamental solitons"
        );
    }

    #[test]
    fn higher_order_soliton_products_ordered_by_width() {
        // Product solitons: T_k = T₀/(2N-2k+1).  Widths increase with k.
        let hos = HigherOrderSoliton::new(3, 1e-12, -20e-27, 1.3e-3);
        let products = hos.fission_products();
        // k=1 → T₀/5; k=2 → T₀/3; k=3 → T₀/1
        assert!(products[0].pulse_width < products[1].pulse_width);
        assert!(products[1].pulse_width < products[2].pulse_width);
    }

    // ── SolitonTrap ──────────────────────────────────────────────────────────

    #[test]
    fn soliton_trap_trapping_condition_fundamental_pair() {
        let s1 = FundamentalSoliton::new(1e-12, -20e-27, 1.3e-3, 1550e-9);
        let s2 = FundamentalSoliton::new(1.2e-12, -18e-27, 1.3e-3, 1530e-9);
        let trap = SolitonTrap::new(s1, s2, 2e-12); // 2 ps separation
                                                    // Both are fundamental (N≈1), anomalous, and close → should trap
        assert!(trap.trapping_condition());
    }

    #[test]
    fn soliton_trap_no_trap_large_separation() {
        let s1 = FundamentalSoliton::new(1e-12, -20e-27, 1.3e-3, 1550e-9);
        let s2 = FundamentalSoliton::new(1e-12, -20e-27, 1.3e-3, 1530e-9);
        let trap = SolitonTrap::new(s1, s2, 100e-12); // 100 ps >> 5 T₀
        assert!(!trap.trapping_condition(), "Large separation: no trapping");
    }

    #[test]
    fn soliton_trap_xpm_force_positive() {
        let s1 = FundamentalSoliton::new(1e-12, -20e-27, 1.3e-3, 1550e-9);
        let s2 = FundamentalSoliton::new(1e-12, -20e-27, 1.3e-3, 1530e-9);
        let trap = SolitonTrap::new(s1, s2, 2e-12);
        assert!(
            trap.xpm_attraction_force() >= 0.0,
            "XPM force must be non-negative"
        );
    }

    // ── PeregineSoliton ──────────────────────────────────────────────────────

    #[test]
    fn peregine_soliton_peak_amplitude_triple_background() {
        let sol = PeregineSoliton::new(1.0, -20e-27, 1.3e-3);
        assert_abs_diff_eq!(sol.peak_amplitude(), 3.0, epsilon = 1e-12);
    }

    #[test]
    fn peregine_soliton_field_at_origin() {
        // At (z=0, t=0): ψ = a₀ [1 - 4/(1)] · exp(0) = a₀(-3) → |ψ| = 3 a₀
        let a0 = 2.0;
        let sol = PeregineSoliton::new(a0, -20e-27, 1.3e-3);
        let f = sol.field(0.0, 0.0);
        assert_abs_diff_eq!(f.norm(), 3.0 * a0, epsilon = 1e-10);
    }

    #[test]
    fn peregine_soliton_background_at_large_t() {
        // For |τ| → ∞ the rational factor → 1, so |ψ| → a₀
        let a0 = 1.5;
        let sol = PeregineSoliton::new(a0, -20e-27, 1.3e-3);
        // Use large t (many T₀ equivalents)
        let t_scale = (sol.beta2.abs() / (2.0 * sol.gamma * a0 * a0)).sqrt();
        let f = sol.field(0.0, 100.0 * t_scale); // τ = 100 ≫ 1
                                                 // |ψ| ≈ a₀ within ~0.5 %
        assert!(
            (f.norm() / a0 - 1.0).abs() < 0.005,
            "|ψ(t→∞)| = {:.4} should ≈ a₀ = {a0}",
            f.norm()
        );
    }
}
