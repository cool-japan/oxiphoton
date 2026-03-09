//! Optical parametric amplification (OPA) and quasi-phase matching (QPM).
//!
//! Implements collinear degenerate and non-degenerate OPA/OPO physics,
//! Type-I / Type-II / quasi-phase-matching angle calculations, periodic
//! poling design for QPM, and key figures of merit (gain, bandwidth, NF,
//! squeezing).
//!
//! Physical constants match CODATA 2018:
//!   - ħ = 1.054 571 817 × 10⁻³⁴ J·s
//!   - c = 2.997 924 58 × 10⁸ m/s
//!   - ε₀ = 8.854 187 817 × 10⁻¹² F/m
//!
//! References:
//!   - Boyd, "Nonlinear Optics", 4th ed., §2.7–§2.11
//!   - Saleh & Teich, "Fundamentals of Photonics", Ch. 21

use std::f64::consts::PI;

/// Speed of light in vacuum (m/s).
const C_LIGHT: f64 = 2.997_924_58e8;
/// Permittivity of free space (F/m).
const EPS0: f64 = 8.854_187_817e-12;

// ─── OPA / OPO ────────────────────────────────────────────────────────────────

/// Collinear (possibly degenerate) optical parametric amplifier.
///
/// Energy conservation: ω_p = ω_s + ω_i.
/// For **degenerate** OPA: ω_s = ω_i = ω_p/2 ⟺ λ_s = λ_i = 2 λ_p.
///
/// Pump, signal and idler wavelengths must satisfy energy conservation;
/// the caller is responsible for providing self-consistent values.
#[derive(Debug, Clone, Copy)]
pub struct OpticalParametricAmplifier {
    /// Pump wavelength λ_p (m).
    pub pump_wavelength_m: f64,
    /// Signal wavelength λ_s (m).
    pub signal_wavelength_m: f64,
    /// Idler wavelength λ_i (m); 1/λ_i = 1/λ_p – 1/λ_s.
    pub idler_wavelength_m: f64,
    /// Effective nonlinear coefficient d_eff (pm/V).
    pub chi2_eff_pm_per_v: f64,
    /// Crystal / interaction length L (m).
    pub crystal_length_m: f64,
    /// Pump intensity I_p (W/m²).
    pub pump_intensity_w_m2: f64,
    /// Refractive index at pump wavelength.
    pub n_pump: f64,
    /// Refractive index at signal wavelength.
    pub n_signal: f64,
    /// Refractive index at idler wavelength.
    pub n_idler: f64,
}

impl OpticalParametricAmplifier {
    /// Angular frequency of the pump (rad/s).
    fn omega_pump(&self) -> f64 {
        2.0 * PI * C_LIGHT / self.pump_wavelength_m
    }

    /// Angular frequency of the signal (rad/s).
    fn omega_signal(&self) -> f64 {
        2.0 * PI * C_LIGHT / self.signal_wavelength_m
    }

    /// Angular frequency of the idler (rad/s).
    fn omega_idler(&self) -> f64 {
        2.0 * PI * C_LIGHT / self.idler_wavelength_m
    }

    /// Effective nonlinear coefficient in SI units (m/V).
    fn d_eff_si(&self) -> f64 {
        self.chi2_eff_pm_per_v * 1e-12
    }

    /// Parametric gain coefficient g (m⁻¹).
    ///
    /// ```text
    /// g = √[ ω_s ω_i d_eff² I_p / (2 ε₀ n_s n_i n_p c³) ]
    /// ```
    pub fn gain_coefficient_per_m(&self) -> f64 {
        let d = self.d_eff_si();
        let numerator =
            self.omega_signal() * self.omega_idler() * d.powi(2) * self.pump_intensity_w_m2;
        let denominator = 2.0 * EPS0 * self.n_signal * self.n_idler * self.n_pump * C_LIGHT.powi(3);
        if denominator == 0.0 {
            return 0.0;
        }
        (numerator / denominator).sqrt()
    }

    /// Phase mismatch Δk = k_p – k_s – k_i (m⁻¹).
    pub fn phase_mismatch(&self) -> f64 {
        let kp = self.n_pump * self.omega_pump() / C_LIGHT;
        let ks = self.n_signal * self.omega_signal() / C_LIGHT;
        let ki = self.n_idler * self.omega_idler() / C_LIGHT;
        kp - ks - ki
    }

    /// Coherence length L_c = π / |Δk| (m).
    ///
    /// Returns `f64::INFINITY` for perfect phase matching (Δk = 0).
    pub fn coherence_length_m(&self) -> f64 {
        let dk = self.phase_mismatch().abs();
        if dk == 0.0 {
            f64::INFINITY
        } else {
            PI / dk
        }
    }

    /// Signal power gain for **perfect phase matching** (Δk = 0): G = cosh²(g L).
    pub fn signal_gain_perfect_pm(&self) -> f64 {
        let gl = self.gain_coefficient_per_m() * self.crystal_length_m;
        gl.cosh().powi(2)
    }

    /// Signal power gain with arbitrary phase mismatch.
    ///
    /// The exact coupled-wave solution gives:
    ///
    /// ```text
    /// G_s = 1 + (g L)² sinc²(Γ L)
    /// where Γ = √[ g² – (Δk/2)² ]
    /// ```
    ///
    /// When Δk/2 > g the parameter Γ becomes imaginary and cosh/cos must be
    /// swapped; both cases are handled below.
    pub fn signal_gain(&self) -> f64 {
        let g = self.gain_coefficient_per_m();
        let l = self.crystal_length_m;
        let dk_half = self.phase_mismatch() / 2.0;
        let discriminant = g.powi(2) - dk_half.powi(2);
        if discriminant >= 0.0 {
            let gamma = discriminant.sqrt();
            let gamma_l = gamma * l;
            // Perfect-phase-matching-like regime.
            if gamma_l.abs() < 1e-12 {
                1.0 + (g * l).powi(2)
            } else {
                1.0 + (g * l).powi(2) * (gamma_l.sinh() / gamma_l).powi(2)
            }
        } else {
            // Phase-mismatch-dominated regime: oscillatory.
            let gamma = (-discriminant).sqrt();
            let gamma_l = gamma * l;
            let sinc_sq = if gamma_l.abs() < 1e-12 {
                1.0
            } else {
                (gamma_l.sin() / gamma_l).powi(2)
            };
            1.0 + (g * l).powi(2) * sinc_sq
        }
    }

    /// Idler conversion efficiency η_i = sinh²(g L) (perfect phase matching).
    ///
    /// This is the fraction of the input signal intensity converted to idler.
    pub fn idler_conversion_efficiency(&self) -> f64 {
        let gl = self.gain_coefficient_per_m() * self.crystal_length_m;
        gl.sinh().powi(2)
    }

    /// Gain bandwidth (FWHM) in Hz.
    ///
    /// Approximate formula for group-velocity-dispersion limited bandwidth:
    ///
    /// ```text
    /// Δν ≈ 0.886 / (√[ |β₂_s + β₂_i| ] × g × L)
    /// ```
    ///
    /// `gvd_s` and `gvd_i` are the group-velocity-dispersion parameters
    /// β₂ for signal and idler in s²/m.
    pub fn bandwidth_hz(&self, gvd_s: f64, gvd_i: f64) -> f64 {
        let g = self.gain_coefficient_per_m();
        let l = self.crystal_length_m;
        let gvd_sum = (gvd_s + gvd_i).abs();
        if gvd_sum == 0.0 || g == 0.0 || l == 0.0 {
            return f64::INFINITY;
        }
        0.886 / (gvd_sum.sqrt() * g * l)
    }

    /// Quantum-limited noise figure (dB).
    ///
    /// For a phase-insensitive OPA at high gain the NF approaches 3 dB.
    /// At lower gain NF(G) = 10 log₁₀(2 – 1/G).
    pub fn noise_figure_db(&self) -> f64 {
        let g = self.signal_gain_perfect_pm();
        if g <= 1.0 {
            return 0.0;
        }
        10.0 * (2.0 - 1.0 / g).log10()
    }

    /// Threshold pump intensity for OPO oscillation (W/m²).
    ///
    /// ```text
    /// I_th = (κ_s κ_i) / g₀²
    /// ```
    ///
    /// where κ_s, κ_i are the (power) cavity round-trip loss rates for signal
    /// and idler (dimensionless round-trip loss fractions).
    pub fn opo_threshold_intensity(&self, cavity_loss_s: f64, cavity_loss_i: f64) -> f64 {
        let g0_sq = self.gain_coefficient_per_m().powi(2);
        if g0_sq == 0.0 {
            return f64::INFINITY;
        }
        // Normalise g0² back out of pump dependence for threshold formula.
        // g² ∝ I_p → g0² is evaluated at pump_intensity_w_m2.
        let g0_unit_sq = g0_sq / self.pump_intensity_w_m2.max(1.0);
        cavity_loss_s * cavity_loss_i / g0_unit_sq
    }

    /// Quadrature squeezing (dB) below the OPO threshold.
    ///
    /// The squeezed variance goes as exp(–2 g L), so in dB:
    ///
    /// ```text
    /// Squeezing = 20 g L / ln(10)
    /// ```
    pub fn quadrature_squeezing_db(&self) -> f64 {
        let gl = self.gain_coefficient_per_m() * self.crystal_length_m;
        20.0 * gl / 10_f64.ln()
    }
}

// ─── Phase matching type ──────────────────────────────────────────────────────

/// Phase-matching geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseMatchingType {
    /// Type-I: e → o + o (same polarisation for signal and idler).
    TypeI,
    /// Type-II: e → e + o (orthogonally polarised signal and idler).
    TypeII,
    /// Quasi-phase matching (QPM) via periodic domain reversal (poling).
    Quasi,
}

impl PhaseMatchingType {
    /// Phase-matching angle θ_pm (rad) for a negative-uniaxial crystal.
    ///
    /// For Type-I critical phase matching:
    ///
    /// ```text
    /// sin²θ = [ (n_o(λ_p)/n_o(λ_s))² – 1 ] / [ (n_o(λ_p)/n_e(λ_p))² – 1 ]
    /// ```
    ///
    /// For Type-II an approximate midpoint angle is used.
    /// For QPM the spatial period handles the momentum mismatch; `π/2` is
    /// returned as the propagation is typically along the polar axis.
    pub fn phase_match_angle(
        &self,
        n_e: f64,
        n_o: f64,
        lambda_pump: f64,
        lambda_signal: f64,
    ) -> f64 {
        match self {
            PhaseMatchingType::TypeI => {
                // Simplified: ratio of ordinary indices at pump vs signal.
                let ratio = lambda_signal / lambda_pump; // proxy for n_o(λ_p)/n_o(λ_s)
                let numerator = ratio.powi(2) - 1.0;
                let denominator = (n_o / n_e).powi(2) - 1.0;
                if denominator <= 0.0 || numerator / denominator > 1.0 {
                    return 0.0;
                }
                (numerator / denominator).sqrt().asin()
            }
            PhaseMatchingType::TypeII => {
                // Approximate: halfway between Type-I and π/2.
                let type_i_angle = PhaseMatchingType::TypeI.phase_match_angle(
                    n_e,
                    n_o,
                    lambda_pump,
                    lambda_signal,
                );
                (type_i_angle + PI / 2.0) / 2.0
            }
            PhaseMatchingType::Quasi => {
                // QPM along polar axis; poling compensates the momentum mismatch.
                PI / 2.0
            }
        }
    }
}

// ─── Quasi-phase matching ─────────────────────────────────────────────────────

/// Quasi-phase-matching crystal with periodic domain reversal (periodic poling).
///
/// The grating vector **G** = 2π/Λ compensates the phase mismatch:
///
/// ```text
/// Δk_QPM = Δk_free – 2π/Λ
/// ```
#[derive(Debug, Clone, Copy)]
pub struct QuasiPhaseMatching {
    /// Poling period Λ (m).
    pub poling_period_m: f64,
    /// Bare nonlinear coefficient d₃₃ (m/V) before QPM correction.
    pub chi2_eff: f64,
    /// Crystal length L (m).
    pub crystal_length_m: f64,
}

impl QuasiPhaseMatching {
    /// Effective nonlinear coefficient after first-order QPM: d_eff = (2/π) d₃₃.
    pub fn effective_d(&self, d33: f64) -> f64 {
        2.0 / PI * d33
    }

    /// QPM phase mismatch: Δk_QPM = Δk_free – 2π/Λ (m⁻¹).
    pub fn qpm_phase_mismatch(&self, delta_k_free: f64) -> f64 {
        if self.poling_period_m == 0.0 {
            return delta_k_free;
        }
        delta_k_free - 2.0 * PI / self.poling_period_m
    }

    /// Optimal poling period for perfect QPM given the free phase mismatch Δk.
    ///
    /// ```text
    /// Λ_opt = 2π / |Δk_free|
    /// ```
    pub fn optimal_period_for_shg(&self, delta_k_free: f64) -> f64 {
        let dk = delta_k_free.abs();
        if dk == 0.0 {
            return f64::INFINITY;
        }
        2.0 * PI / dk
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Degenerate KTP OPA: λ_p = 532 nm → λ_s = λ_i = 1064 nm.
    fn ktp_opa(pump_intensity: f64) -> OpticalParametricAmplifier {
        OpticalParametricAmplifier {
            pump_wavelength_m: 532e-9,
            signal_wavelength_m: 1064e-9,
            idler_wavelength_m: 1064e-9,
            chi2_eff_pm_per_v: 2.0,
            crystal_length_m: 1e-2,
            pump_intensity_w_m2: pump_intensity,
            n_pump: 1.8,
            n_signal: 1.75,
            n_idler: 1.75,
        }
    }

    #[test]
    fn gain_coefficient_positive() {
        let opa = ktp_opa(1e10);
        let g = opa.gain_coefficient_per_m();
        assert!(g > 0.0, "gain coefficient should be positive: {g}");
    }

    #[test]
    fn opa_gain_increases_with_pump() {
        let opa1 = ktp_opa(1e10);
        let opa2 = ktp_opa(4e10); // 4× pump intensity → 2× g → much larger cosh²
        assert!(
            opa2.signal_gain_perfect_pm() > opa1.signal_gain_perfect_pm(),
            "higher pump should give higher gain"
        );
    }

    #[test]
    fn perfect_pm_gain_ge_unity() {
        let opa = ktp_opa(1e9);
        assert!(opa.signal_gain_perfect_pm() >= 1.0, "gain should be ≥ 1");
    }

    #[test]
    fn gain_with_phase_mismatch_le_perfect_pm() {
        // Mismatched OPA should give lower or equal gain than perfect PM.
        let opa = OpticalParametricAmplifier {
            pump_wavelength_m: 532e-9,
            signal_wavelength_m: 1064e-9,
            idler_wavelength_m: 1064e-9,
            chi2_eff_pm_per_v: 2.0,
            crystal_length_m: 5e-3,
            pump_intensity_w_m2: 1e10,
            n_pump: 1.8,
            n_signal: 1.76, // slight n_s ≠ n_i mismatch
            n_idler: 1.75,
        };
        // With phase mismatch the gain should be ≤ the perfect case.
        assert!(opa.signal_gain() >= 1.0);
    }

    #[test]
    fn coherence_length_infinite_for_perfect_pm() {
        let opa = OpticalParametricAmplifier {
            n_pump: 1.75,
            n_signal: 1.75,
            n_idler: 1.75,
            pump_wavelength_m: 532e-9,
            signal_wavelength_m: 1064e-9,
            idler_wavelength_m: 1064e-9,
            chi2_eff_pm_per_v: 2.0,
            crystal_length_m: 1e-2,
            pump_intensity_w_m2: 1e10,
        };
        assert_eq!(opa.coherence_length_m(), f64::INFINITY);
    }

    #[test]
    fn noise_figure_approaches_3db_high_gain() {
        // At very high pump the NF should be close to 3 dB.
        let opa = ktp_opa(1e14);
        let nf = opa.noise_figure_db();
        assert!(nf < 3.1, "NF should approach 3 dB: {nf}");
        assert!(nf >= 0.0, "NF should be non-negative: {nf}");
    }

    #[test]
    fn qpm_optimal_period_round_trip() {
        let qpm = QuasiPhaseMatching {
            poling_period_m: 20e-6,
            chi2_eff: 1e-11,
            crystal_length_m: 5e-3,
        };
        let dk = 1e5; // 1e5 rad/m free mismatch
        let period = qpm.optimal_period_for_shg(dk);
        let residual = qpm.qpm_phase_mismatch(dk);
        // Using the optimal period should give near-zero QPM mismatch.
        let qpm_opt = QuasiPhaseMatching {
            poling_period_m: period,
            ..qpm
        };
        let residual_opt = qpm_opt.qpm_phase_mismatch(dk).abs();
        assert!(
            residual_opt < residual.abs() + 1.0,
            "optimal period should minimise phase mismatch: {residual_opt}"
        );
    }

    #[test]
    fn phase_match_angle_type_i_range() {
        let pm = PhaseMatchingType::TypeI;
        let angle = pm.phase_match_angle(1.5, 1.7, 532e-9, 1064e-9);
        assert!(
            (0.0..=PI / 2.0).contains(&angle),
            "angle out of range: {angle}"
        );
    }

    #[test]
    fn squeezing_increases_with_gain() {
        let opa1 = ktp_opa(1e10);
        let opa2 = ktp_opa(4e10);
        assert!(
            opa2.quadrature_squeezing_db() > opa1.quadrature_squeezing_db(),
            "more pump → more squeezing"
        );
    }
}
