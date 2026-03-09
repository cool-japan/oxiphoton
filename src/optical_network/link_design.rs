//! WDM link margin analysis and multi-span design.
//!
//! Implements the Gaussian Noise (GN) model for coherent WDM systems,
//! span OSNR accumulation, nonlinear noise estimation, and optimal launch
//! power computation.
//!
//! # Model Overview
//! The end-to-end SNR is computed as:
//! ```text
//!   1/SNR_total = 1/SNR_ASE + 1/SNR_NL
//! ```
//! where `SNR_ASE` accumulates amplified spontaneous emission noise and
//! `SNR_NL` is the nonlinear SNR from the GN model.
//!
//! # References
//! - P. Poggiolini, "The GN Model of Non-Linear Propagation in Uncompensated
//!   Coherent Optical Systems," JLT 30(24), 2012.
//! - ITU-T G.977 — Optically amplified submarine cable systems

use super::wdm_system::WdmLineSystem;

/// Planck's constant \[J·s\]
const H_PLANCK: f64 = 6.626_070_15e-34;
/// Reference frequency for C-band \[Hz\] (193.1 THz)
const F_REF_HZ: f64 = 193.1e12;
/// Boltzmann constant \[J/K\] — not used here but kept for documentation
#[allow(dead_code)]
const K_B: f64 = 1.380_649e-23;

// ─────────────────────────────────────────────────────────────────────────────
// FiberType
// ─────────────────────────────────────────────────────────────────────────────

/// Fiber type with characteristic parameters.
#[derive(Debug, Clone, PartialEq)]
pub enum FiberType {
    /// G.652 standard single-mode fiber (SMF-28).
    Smf28 {
        /// Effective mode area \[µm²\] (typ. 80–85 µm²).
        effective_area_um2: f64,
        /// Nonlinear refractive index \[m²/W\] (typ. 2.6×10⁻²⁰).
        n2: f64,
    },
    /// G.655 non-zero dispersion-shifted fiber (TrueWave-RS).
    TrueWave {
        /// Effective mode area \[µm²\] (typ. 50–65 µm²).
        effective_area_um2: f64,
    },
    /// G.654 ultra-low-loss fiber (e.g., Corning SMF-28 ULL).
    UltraLowLoss {
        /// Fiber loss \[dB/km\] (typ. 0.155–0.17 dB/km).
        loss_db_per_km: f64,
    },
    /// G.653 dispersion-shifted fiber (legacy, rarely used).
    DispersionShifted,
}

impl FiberType {
    /// Nonlinear coefficient `γ = 2π·n₂ / (λ·A_eff)` \[1/(W·km)\].
    ///
    /// Using λ = 1550 nm as reference wavelength.
    pub fn nonlinear_coefficient_per_w_per_km(&self) -> f64 {
        let lambda_m = 1.55e-6_f64;
        match self {
            FiberType::Smf28 {
                effective_area_um2,
                n2,
            } => {
                let a_eff_m2 = effective_area_um2 * 1e-12; // µm² → m²
                                                           // 2πn₂/(λ A_eff) gives 1/(m·W); ×1000 converts to 1/(km·W)
                2.0 * std::f64::consts::PI * n2 / (lambda_m * a_eff_m2) * 1e3
            }
            FiberType::TrueWave { effective_area_um2 } => {
                let a_eff_m2 = effective_area_um2 * 1e-12;
                let n2 = 2.6e-20_f64;
                2.0 * std::f64::consts::PI * n2 / (lambda_m * a_eff_m2) * 1e3
            }
            FiberType::UltraLowLoss { .. } => {
                // Larger effective area ≈ 130 µm²
                let a_eff_m2 = 130e-12_f64;
                let n2 = 2.2e-20_f64;
                2.0 * std::f64::consts::PI * n2 / (lambda_m * a_eff_m2) * 1e3
            }
            FiberType::DispersionShifted => {
                // A_eff ≈ 50 µm²
                let a_eff_m2 = 50e-12_f64;
                let n2 = 2.6e-20_f64;
                2.0 * std::f64::consts::PI * n2 / (lambda_m * a_eff_m2) * 1e3
            }
        }
    }

    /// Effective mode area \[µm²\].
    pub fn effective_area_um2(&self) -> f64 {
        match self {
            FiberType::Smf28 {
                effective_area_um2, ..
            } => *effective_area_um2,
            FiberType::TrueWave { effective_area_um2 } => *effective_area_um2,
            FiberType::UltraLowLoss { .. } => 130.0,
            FiberType::DispersionShifted => 50.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OpticalSpan
// ─────────────────────────────────────────────────────────────────────────────

/// A single optical span (fiber section + booster/pre-amplifier).
///
/// Each span is characterized by its fiber length, fiber parameters,
/// and the amplifier placed at the output to compensate the span loss.
#[derive(Debug, Clone)]
pub struct OpticalSpan {
    /// Fiber length \[km\].
    pub fiber_length_km: f64,
    /// Fiber attenuation \[dB/km\].
    pub fiber_loss_db_per_km: f64,
    /// Chromatic dispersion \[ps/(nm·km)\].
    pub fiber_dispersion_ps_per_nm_km: f64,
    /// Fiber type (determines nonlinear coefficient).
    pub fiber_type: FiberType,
    /// Amplifier gain \[dB\] (should equal span loss for transparent operation).
    pub amplifier_gain_db: f64,
    /// Amplifier noise figure \[dB\] (typ. 4.5–6 dB for EDFA).
    pub amplifier_nf_db: f64,
}

impl OpticalSpan {
    /// Create a standard SMF-28 span with EDFA compensation.
    ///
    /// Amplifier gain is set equal to span loss (transparent).
    pub fn new_smf28(length_km: f64, amp_gain_db: f64, amp_nf_db: f64) -> Self {
        Self {
            fiber_length_km: length_km,
            fiber_loss_db_per_km: 0.2,
            fiber_dispersion_ps_per_nm_km: 17.0,
            fiber_type: FiberType::Smf28 {
                effective_area_um2: 80.0,
                n2: 2.6e-20,
            },
            amplifier_gain_db: amp_gain_db,
            amplifier_nf_db: amp_nf_db,
        }
    }

    /// Create an ultra-low-loss (G.654) span.
    pub fn new_ull(length_km: f64, amp_gain_db: f64, amp_nf_db: f64) -> Self {
        Self {
            fiber_length_km: length_km,
            fiber_loss_db_per_km: 0.157,
            fiber_dispersion_ps_per_nm_km: 20.0,
            fiber_type: FiberType::UltraLowLoss {
                loss_db_per_km: 0.157,
            },
            amplifier_gain_db: amp_gain_db,
            amplifier_nf_db: amp_nf_db,
        }
    }

    /// Span loss: `L_span = α × L` \[dB\].
    pub fn span_loss_db(&self) -> f64 {
        self.fiber_loss_db_per_km * self.fiber_length_km
    }

    /// Linear loss coefficient `α` \[1/km\] (Naperian).
    pub fn alpha_per_km(&self) -> f64 {
        self.fiber_loss_db_per_km / (10.0 * std::f64::consts::LOG10_E)
    }

    /// Effective length \[km\]: `L_eff = (1 - exp(-α·L)) / α`.
    pub fn effective_length_km(&self) -> f64 {
        let alpha = self.alpha_per_km();
        if alpha < 1e-12 {
            return self.fiber_length_km;
        }
        (1.0 - (-alpha * self.fiber_length_km).exp()) / alpha
    }

    /// OSNR contribution from this span \[dB\].
    ///
    /// The OSNR added by one span (signal + ASE in reference bandwidth B_ref = 12.5 GHz):
    /// ```text
    ///   OSNR_span = P_launch / (h·ν·NF·(G-1)·B_ref)
    /// ```
    /// Approximate for large gain: `OSNR_span ≈ P_launch / (h·ν·NF·G·B_ref)`.
    ///
    /// `bandwidth_nm` is the OSNR reference bandwidth (0.1 nm = 12.5 GHz at 1550 nm).
    pub fn osnr_contribution_db(&self, launch_power_dbm: f64, bandwidth_nm: f64) -> f64 {
        let p_launch_w = 1e-3 * 10.0_f64.powf(launch_power_dbm / 10.0);
        let nf_linear = 10.0_f64.powf(self.amplifier_nf_db / 10.0);
        let g_linear = 10.0_f64.powf(self.amplifier_gain_db / 10.0);
        // Convert bandwidth_nm to Hz: Δν = c·Δλ/λ²
        let lambda_m = 1.55e-6_f64;
        let bw_hz = 2.998e8 * bandwidth_nm * 1e-9 / (lambda_m * lambda_m);
        let p_ase = nf_linear * (g_linear - 1.0) * H_PLANCK * F_REF_HZ * bw_hz;
        let osnr = p_launch_w / p_ase.max(1e-30);
        10.0 * osnr.log10()
    }

    /// Accumulated dispersion over this span \[ps/nm\].
    pub fn dispersion_ps_per_nm(&self) -> f64 {
        self.fiber_dispersion_ps_per_nm_km * self.fiber_length_km
    }

    /// Nonlinear phase shift (SPM): `φ_NL = γ·P₀·L_eff` \[rad\].
    pub fn nonlinear_phase_shift_rad(&self, launch_power_dbm: f64) -> f64 {
        let p0_w = 1e-3 * 10.0_f64.powf(launch_power_dbm / 10.0);
        let gamma = self.fiber_type.nonlinear_coefficient_per_w_per_km();
        gamma * p0_w * self.effective_length_km()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WdmLink
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-span WDM optical link.
///
/// Models a chain of optical spans and computes end-to-end OSNR,
/// nonlinear SNR, total SNR, optimal launch power, and link margin.
#[derive(Debug, Clone)]
pub struct WdmLink {
    /// Ordered list of optical spans.
    pub spans: Vec<OpticalSpan>,
    /// WDM line system (modulation format, baud rate, etc.).
    pub system: WdmLineSystem,
}

impl WdmLink {
    /// Create a new WDM link.
    pub fn new(spans: Vec<OpticalSpan>, system: WdmLineSystem) -> Self {
        Self { spans, system }
    }

    /// Number of spans.
    pub fn n_spans(&self) -> usize {
        self.spans.len()
    }

    /// Total link length \[km\].
    pub fn total_length_km(&self) -> f64 {
        self.spans.iter().map(|s| s.fiber_length_km).sum()
    }

    /// End-of-link OSNR \[dB\] using the Gaussian noise accumulation model.
    ///
    /// For N spans with identical parameters:
    /// ```text
    ///   OSNR_total = OSNR_span - 10·log10(N_spans)
    /// ```
    /// For heterogeneous spans, noise contributions are summed linearly:
    /// ```text
    ///   1/OSNR_total = Σ_i 1/OSNR_i
    /// ```
    pub fn end_of_link_osnr_db(&self) -> f64 {
        if self.spans.is_empty() {
            return f64::INFINITY;
        }
        let p_dbm = self.system.launch_power_dbm_per_channel;
        let bw_nm = 0.1_f64; // ITU reference bandwidth: 0.1 nm = 12.5 GHz
                             // Sum OSNR noise contributions (linear)
        let noise_sum: f64 = self
            .spans
            .iter()
            .map(|span| {
                let osnr_db = span.osnr_contribution_db(p_dbm, bw_nm);
                10.0_f64.powf(-osnr_db / 10.0)
            })
            .sum();
        if noise_sum <= 0.0 {
            return f64::INFINITY;
        }
        -10.0 * noise_sum.log10()
    }

    /// Nonlinear SNR \[dB\] from the simplified GN model.
    ///
    /// The GN model gives the nonlinear noise power scaling as:
    /// ```text
    ///   P_NL ∝ γ² · P³ · L_eff² · N_spans
    /// ```
    /// Hence:
    /// ```text
    ///   SNR_NL = 1 / (κ · γ² · P² · L_eff² · N_spans)
    /// ```
    /// where κ is a constant that depends on the modulation format and
    /// channel count (we use a simplified scalar model).
    pub fn nonlinear_snr_db(&self) -> f64 {
        if self.spans.is_empty() {
            return f64::INFINITY;
        }
        let p_w = 1e-3 * 10.0_f64.powf(self.system.launch_power_dbm_per_channel / 10.0);
        // Compute span-averaged parameters
        let n_sp = self.spans.len() as f64;
        let gamma_avg: f64 = self
            .spans
            .iter()
            .map(|s| s.fiber_type.nonlinear_coefficient_per_w_per_km())
            .sum::<f64>()
            / n_sp;
        let l_eff_avg_km: f64 = self
            .spans
            .iter()
            .map(|s| s.effective_length_km())
            .sum::<f64>()
            / n_sp;
        let d_avg: f64 = self
            .spans
            .iter()
            .map(|s| s.fiber_dispersion_ps_per_nm_km)
            .sum::<f64>()
            / n_sp;

        // GN model scaling constant (simplified; uses WDM bandwidth).
        // All length quantities in km, γ in W⁻¹km⁻¹.
        //
        // β₂ in s²/m: β₂ ≈ -D·λ²/(2πc) ≈ D × 1.28e-26 s²/m
        // where D is in ps/(nm·km).
        let beta2_s2_per_m = d_avg.abs() * 1.28e-26;
        let bw_hz = self.system.channel_plan.n_channels as f64
            * self.system.channel_plan.spacing_ghz()
            * 1e9;
        // α in /m for the logarithm argument (physical units match)
        let alpha_per_m = self
            .spans
            .first()
            .map(|s| s.alpha_per_km() / 1000.0)
            .unwrap_or(4.6e-5_f64);
        // κ_gn ≈ (8/27) * π * ln(π² |β₂| B_WDM² / α)
        let arg =
            std::f64::consts::PI * std::f64::consts::PI * beta2_s2_per_m.abs() * bw_hz * bw_hz
                / alpha_per_m.max(1e-30);
        let kappa = (8.0 / 27.0) * std::f64::consts::PI * arg.max(1.0).ln();

        // γ in W⁻¹km⁻¹, L_eff in km → γ²·L_eff² in W⁻²km⁻² km² = W⁻²
        // This is dimensionally consistent since P² in W² cancels.
        let snr_nl_inv =
            kappa * gamma_avg * gamma_avg * p_w * p_w * l_eff_avg_km * l_eff_avg_km * n_sp;
        if snr_nl_inv <= 0.0 {
            return f64::INFINITY;
        }
        -10.0 * snr_nl_inv.log10()
    }

    /// Total SNR \[dB\] combining ASE and nonlinear noise.
    ///
    /// ```text
    ///   1/SNR_total = 1/SNR_ASE + 1/SNR_NL
    /// ```
    pub fn total_snr_db(&self) -> f64 {
        let snr_ase = 10.0_f64.powf(self.end_of_link_osnr_db() / 10.0);
        let snr_nl = 10.0_f64.powf(self.nonlinear_snr_db() / 10.0);
        let inv_total = 1.0 / snr_ase.max(1e-30) + 1.0 / snr_nl.max(1e-30);
        -10.0 * inv_total.log10()
    }

    /// Optimal launch power \[dBm\] that maximises total SNR.
    ///
    /// From ∂SNR_total/∂P = 0:
    /// ```text
    ///   P_opt = (SNR_ASE_coeff / (2 · SNR_NL_coeff))^{1/3}
    /// ```
    /// which simplifies to an iterative golden-section search over
    /// launch power in \[-6, +6\] dBm.
    pub fn optimal_launch_power_dbm(&self) -> f64 {
        // Golden-section search for maximum total SNR
        let mut lo = -10.0_f64;
        let mut hi = 10.0_f64;
        let gr = (5.0_f64.sqrt() - 1.0) / 2.0; // golden ratio conjugate
        let tol = 1e-4_f64;

        // Clone link for testing different launch powers
        let mut test_link = self.clone();

        for _ in 0..100 {
            let x1 = hi - gr * (hi - lo);
            let x2 = lo + gr * (hi - lo);

            test_link.system.launch_power_dbm_per_channel = x1;
            let snr1 = test_link.total_snr_db();
            test_link.system.launch_power_dbm_per_channel = x2;
            let snr2 = test_link.total_snr_db();

            if snr1 < snr2 {
                lo = x1;
            } else {
                hi = x2;
            }
            if (hi - lo).abs() < tol {
                break;
            }
        }
        (lo + hi) / 2.0
    }

    /// Link margin \[dB\]: `total_SNR - required_SNR`.
    pub fn link_margin_db(&self) -> f64 {
        let total_snr = self.total_snr_db();
        let required = self.system.modulation_format.required_osnr_db();
        total_snr - required
    }

    /// Maximum reach \[km\] for a target OSNR.
    ///
    /// Scales from the current link: reach scales linearly with 1/N_spans
    /// when noise is ASE-dominated:
    /// ```text
    ///   OSNR ∝ 1/N_spans  ⟹  N_max = N_current × OSNR_current/OSNR_target
    /// ```
    /// Then `L_max = L_span × N_max`.
    pub fn max_reach_km(&self, target_osnr_db: f64) -> f64 {
        if self.spans.is_empty() {
            return 0.0;
        }
        let current_osnr_db = self.end_of_link_osnr_db();
        let osnr_excess = current_osnr_db - target_osnr_db;
        if osnr_excess <= 0.0 {
            // Current link already fails — reach is zero
            return 0.0;
        }
        // Number of additional spans before hitting target:
        // OSNR_n = OSNR_1 - 10·log10(n) ≥ target
        // 10·log10(n) ≤ OSNR_1 - target
        let n_max = 10.0_f64.powf(osnr_excess / 10.0) * self.n_spans() as f64;
        let avg_span_km = self.total_length_km() / self.n_spans() as f64;
        n_max * avg_span_km
    }

    /// Q-factor \[dB\] converted from total SNR.
    ///
    /// For AWGN: `Q [dB] ≈ SNR_total [dB] - 3 dB` (common approximation).
    pub fn q_factor_db(&self) -> f64 {
        self.total_snr_db() - 3.0
    }

    /// BER estimate from Q-factor (AWGN model).
    ///
    /// `BER ≈ (1/2) · erfc(Q / √2)`
    pub fn ber_estimate(&self) -> f64 {
        let q_linear = 10.0_f64.powf(self.q_factor_db() / 20.0);
        0.5 * erfc_approx(q_linear / std::f64::consts::SQRT_2)
    }

    /// Total accumulated dispersion \[ps/nm\].
    pub fn total_dispersion_ps_per_nm(&self) -> f64 {
        self.spans.iter().map(|s| s.dispersion_ps_per_nm()).sum()
    }
}

/// Approximation of erfc(x) using Horner's method (Abramowitz & Stegun 7.1.26).
fn erfc_approx(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc_approx(-x);
    }
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    (-x * x).exp() * poly
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optical_network::wdm_system::{ItuChannelPlan, WdmLineSystem, WdmModFormat};
    use approx::assert_abs_diff_eq;

    fn make_system() -> WdmLineSystem {
        WdmLineSystem::new(
            ItuChannelPlan::new_c_band_100ghz(),
            0.0,
            WdmModFormat::DpQpsk,
            32.0,
        )
    }

    fn make_link(n_spans: usize) -> WdmLink {
        let spans: Vec<OpticalSpan> = (0..n_spans)
            .map(|_| OpticalSpan::new_smf28(80.0, 16.0, 5.5))
            .collect();
        WdmLink::new(spans, make_system())
    }

    #[test]
    fn fiber_type_nonlinear_coeff_smf28() {
        let fiber = FiberType::Smf28 {
            effective_area_um2: 80.0,
            n2: 2.6e-20,
        };
        let gamma = fiber.nonlinear_coefficient_per_w_per_km();
        // SMF-28: γ ≈ 1.3 W⁻¹km⁻¹
        assert!(gamma > 1.0 && gamma < 2.0, "γ = {gamma:.3}");
    }

    #[test]
    fn optical_span_span_loss() {
        let span = OpticalSpan::new_smf28(80.0, 16.0, 5.5);
        assert_abs_diff_eq!(span.span_loss_db(), 16.0, epsilon = 1e-9);
    }

    #[test]
    fn optical_span_effective_length() {
        let span = OpticalSpan::new_smf28(80.0, 16.0, 5.5);
        let l_eff = span.effective_length_km();
        // For α ≈ 0.046/km: L_eff = (1 - exp(-0.046×80))/0.046 ≈ 20 km
        assert!(l_eff > 15.0 && l_eff < 25.0, "L_eff = {l_eff:.2} km");
    }

    #[test]
    fn optical_span_nonlinear_phase_shift_positive() {
        let span = OpticalSpan::new_smf28(80.0, 16.0, 5.5);
        let phi = span.nonlinear_phase_shift_rad(0.0); // 1 mW launch
        assert!(phi > 0.0);
    }

    #[test]
    fn optical_span_dispersion() {
        let span = OpticalSpan::new_smf28(80.0, 16.0, 5.5);
        // D = 17 ps/(nm·km) × 80 km = 1360 ps/nm
        assert_abs_diff_eq!(span.dispersion_ps_per_nm(), 1360.0, epsilon = 1e-6);
    }

    #[test]
    fn wdm_link_total_length() {
        let link = make_link(10);
        assert_abs_diff_eq!(link.total_length_km(), 800.0, epsilon = 1e-9);
    }

    #[test]
    fn wdm_link_osnr_decreases_with_more_spans() {
        let link5 = make_link(5);
        let link20 = make_link(20);
        assert!(link5.end_of_link_osnr_db() > link20.end_of_link_osnr_db());
    }

    #[test]
    fn wdm_link_total_snr_positive() {
        let link = make_link(10);
        let snr = link.total_snr_db();
        assert!(snr > 0.0 && snr < 50.0, "SNR = {snr:.2} dB");
    }

    #[test]
    fn wdm_link_optimal_launch_power_in_range() {
        let link = make_link(8);
        let p_opt = link.optimal_launch_power_dbm();
        // Typical optimal launch power: -3 to +3 dBm
        assert!(p_opt > -10.0 && p_opt < 10.0, "P_opt = {p_opt:.2} dBm");
    }

    #[test]
    fn wdm_link_max_reach_positive() {
        let link = make_link(5);
        let reach = link.max_reach_km(20.0); // target 20 dB OSNR
        assert!(reach > 0.0);
    }

    #[test]
    fn wdm_link_ber_between_zero_and_one() {
        let link = make_link(8);
        let ber = link.ber_estimate();
        assert!((0.0..=1.0).contains(&ber), "BER = {ber:.2e}");
    }

    #[test]
    fn erfc_approx_at_zero_is_one() {
        assert_abs_diff_eq!(erfc_approx(0.0), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn erfc_approx_large_x_near_zero() {
        // erfc(5) ≈ 1.54e-12
        assert!(erfc_approx(5.0) < 1e-5);
    }

    #[test]
    fn optical_span_ull_lower_loss() {
        let smf = OpticalSpan::new_smf28(80.0, 16.0, 5.0);
        let ull = OpticalSpan::new_ull(80.0, 16.0, 5.0);
        assert!(ull.fiber_loss_db_per_km < smf.fiber_loss_db_per_km);
    }
}
