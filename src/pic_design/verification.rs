//! PIC Design Rule Checking (DRC) and Circuit Verification.
//!
//! Provides:
//! - Design rule checking for photonic layout constraints
//! - Optical circuit performance budget verification
//! - Thermal analysis for thermo-optic phase shifters
//! - Yield estimation via deterministic Monte Carlo (LCG PRNG)

use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic LCG (local copy to keep this module independent)
// ─────────────────────────────────────────────────────────────────────────────

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Box-Muller transform: returns a standard-normal variate.
    fn next_normal(&mut self) -> f64 {
        let u1 = (self.next_f64() + 1.0e-15).min(1.0 - 1.0e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DRC Severity / Violation
// ─────────────────────────────────────────────────────────────────────────────

/// Severity level of a DRC violation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DrcSeverity {
    /// Must be fixed before tape-out.
    Error,
    /// Should be fixed; may affect yield or performance.
    Warning,
    /// Informational note; within spec but close to limits.
    Info,
}

impl std::fmt::Display for DrcSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Error => write!(f, "ERROR"),
            Self::Warning => write!(f, "WARNING"),
            Self::Info => write!(f, "INFO"),
        }
    }
}

/// A single design rule violation.
#[derive(Debug, Clone)]
pub struct DrcViolation {
    /// Human-readable rule identifier (e.g. "MIN_WIDTH")
    pub rule: String,
    /// (x, y) location of the violation in the layout (µm)
    pub location: [f64; 2],
    /// Severity classification
    pub severity: DrcSeverity,
    /// Actual measured value
    pub actual_value: f64,
    /// Minimum (or maximum) allowed value
    pub min_allowed: f64,
}

impl DrcViolation {
    fn new_error(rule: &str, location: [f64; 2], actual: f64, min: f64) -> Self {
        Self {
            rule: rule.to_owned(),
            location,
            severity: DrcSeverity::Error,
            actual_value: actual,
            min_allowed: min,
        }
    }

    fn new_warning(rule: &str, location: [f64; 2], actual: f64, min: f64) -> Self {
        Self {
            rule: rule.to_owned(),
            location,
            severity: DrcSeverity::Warning,
            actual_value: actual,
            min_allowed: min,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DesignRuleChecker
// ─────────────────────────────────────────────────────────────────────────────

/// Design Rule Checker for photonic IC layouts.
///
/// Rules are process-specific and are loaded from preset process configurations.
/// Each `check_*` method returns `None` if the check passes, or `Some(violation)`
/// if the value violates the design rule.
#[derive(Debug, Clone)]
pub struct DesignRuleChecker {
    /// Minimum allowed waveguide width (nm)
    pub min_width_nm: f64,
    /// Minimum waveguide-to-waveguide spacing (nm)
    pub min_spacing_nm: f64,
    /// Minimum bend radius (µm)
    pub min_bend_radius_um: f64,
    /// Minimum coupling gap (nm)
    pub min_coupling_gap_nm: f64,
    /// Maximum allowed waveguide segment length (mm)
    pub max_waveguide_length_mm: f64,
}

impl DesignRuleChecker {
    /// Rules for the standard 220 nm SOI process (IMEC/IME A*STAR style).
    pub fn new_soi_220nm() -> Self {
        Self {
            min_width_nm: 300.0,
            min_spacing_nm: 200.0,
            min_bend_radius_um: 5.0,
            min_coupling_gap_nm: 100.0,
            max_waveguide_length_mm: 50.0,
        }
    }

    /// Rules for the standard 400 nm SiN process.
    pub fn new_sin_400nm() -> Self {
        Self {
            min_width_nm: 500.0,
            min_spacing_nm: 400.0,
            min_bend_radius_um: 50.0,
            min_coupling_gap_nm: 200.0,
            max_waveguide_length_mm: 100.0,
        }
    }

    /// Check waveguide width.
    ///
    /// Returns `Some(DrcViolation)` if `width_nm < min_width_nm`.
    ///
    /// # Arguments
    /// * `width_nm`  – Measured width (nm)
    /// * `location`  – (x, y) position in µm
    pub fn check_width(&self, width_nm: f64, location: [f64; 2]) -> Option<DrcViolation> {
        if width_nm < self.min_width_nm {
            Some(DrcViolation::new_error(
                "MIN_WIDTH",
                location,
                width_nm,
                self.min_width_nm,
            ))
        } else if width_nm < self.min_width_nm * 1.1 {
            Some(DrcViolation::new_warning(
                "MIN_WIDTH_MARGIN",
                location,
                width_nm,
                self.min_width_nm * 1.1,
            ))
        } else {
            None
        }
    }

    /// Check waveguide-to-waveguide spacing.
    ///
    /// Returns `Some(DrcViolation)` if `spacing_nm < min_spacing_nm`.
    ///
    /// # Arguments
    /// * `spacing_nm` – Measured edge-to-edge spacing (nm)
    /// * `location`   – (x, y) position in µm
    pub fn check_spacing(&self, spacing_nm: f64, location: [f64; 2]) -> Option<DrcViolation> {
        if spacing_nm < self.min_spacing_nm {
            Some(DrcViolation::new_error(
                "MIN_SPACING",
                location,
                spacing_nm,
                self.min_spacing_nm,
            ))
        } else if spacing_nm < self.min_spacing_nm * 1.2 {
            Some(DrcViolation::new_warning(
                "MIN_SPACING_MARGIN",
                location,
                spacing_nm,
                self.min_spacing_nm * 1.2,
            ))
        } else {
            None
        }
    }

    /// Check bend radius.
    ///
    /// Returns `Some(DrcViolation)` if `radius_um < min_bend_radius_um`.
    ///
    /// # Arguments
    /// * `radius_um` – Bend radius (µm)
    /// * `location`  – (x, y) position in µm
    pub fn check_bend_radius(&self, radius_um: f64, location: [f64; 2]) -> Option<DrcViolation> {
        if radius_um < self.min_bend_radius_um {
            Some(DrcViolation::new_error(
                "MIN_BEND_RADIUS",
                location,
                radius_um,
                self.min_bend_radius_um,
            ))
        } else if radius_um < self.min_bend_radius_um * 1.5 {
            Some(DrcViolation::new_warning(
                "MIN_BEND_RADIUS_MARGIN",
                location,
                radius_um,
                self.min_bend_radius_um * 1.5,
            ))
        } else {
            None
        }
    }

    /// Check coupling gap.
    ///
    /// Returns `Some(DrcViolation)` if `gap_nm < min_coupling_gap_nm`.
    ///
    /// # Arguments
    /// * `gap_nm`   – Coupling gap (nm)
    /// * `location` – (x, y) position in µm
    pub fn check_coupling_gap(&self, gap_nm: f64, location: [f64; 2]) -> Option<DrcViolation> {
        if gap_nm < self.min_coupling_gap_nm {
            Some(DrcViolation::new_error(
                "MIN_COUPLING_GAP",
                location,
                gap_nm,
                self.min_coupling_gap_nm,
            ))
        } else if gap_nm < self.min_coupling_gap_nm * 1.3 {
            Some(DrcViolation::new_warning(
                "MIN_COUPLING_GAP_MARGIN",
                location,
                gap_nm,
                self.min_coupling_gap_nm * 1.3,
            ))
        } else {
            None
        }
    }

    /// Run all checks on a waveguide segment and collect violations.
    ///
    /// # Arguments
    /// * `width_nm`     – Waveguide width (nm)
    /// * `bend_radius`  – Bend radius if applicable (µm); use `f64::INFINITY` for straight
    /// * `gap_nm`       – Coupling gap (nm); use `f64::INFINITY` if no coupling
    /// * `location`     – (x, y) position (µm)
    pub fn check_all(
        &self,
        width_nm: f64,
        bend_radius: f64,
        gap_nm: f64,
        location: [f64; 2],
    ) -> Vec<DrcViolation> {
        let mut violations = Vec::new();
        if let Some(v) = self.check_width(width_nm, location) {
            violations.push(v);
        }
        if bend_radius.is_finite() {
            if let Some(v) = self.check_bend_radius(bend_radius, location) {
                violations.push(v);
            }
        }
        if gap_nm.is_finite() {
            if let Some(v) = self.check_coupling_gap(gap_nm, location) {
                violations.push(v);
            }
        }
        violations
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CircuitVerifier
// ─────────────────────────────────────────────────────────────────────────────

/// Optical circuit performance verifier.
///
/// Checks system-level constraints such as loss budget, bandwidth, power
/// handling, and yields estimates from process variation Monte Carlo.
#[derive(Debug, Clone)]
pub struct CircuitVerifier {
    /// Verification wavelength sweep (m)
    pub wavelengths: Vec<f64>,
}

impl CircuitVerifier {
    /// Create a verifier with a uniform wavelength sweep.
    ///
    /// # Arguments
    /// * `start_nm` – Start wavelength (nm)
    /// * `stop_nm`  – Stop wavelength (nm)
    /// * `n_points` – Number of wavelength points
    pub fn new(start_nm: f64, stop_nm: f64, n_points: usize) -> Self {
        let n = n_points.max(2);
        let wavelengths: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                (start_nm + (stop_nm - start_nm) * t) * 1.0e-9
            })
            .collect();
        Self { wavelengths }
    }

    /// Verify the total insertion loss stays within budget.
    ///
    /// # Arguments
    /// * `component_losses` – Per-component insertion losses (dB, positive = lossy)
    /// * `budget_db`        – Maximum allowed total loss (dB)
    ///
    /// Returns `true` if total loss ≤ budget.
    pub fn check_loss_budget(&self, component_losses: &[f64], budget_db: f64) -> bool {
        let total: f64 = component_losses.iter().sum();
        total <= budget_db
    }

    /// Verify that every component has sufficient optical bandwidth.
    ///
    /// # Arguments
    /// * `component_bw_nm` – Per-component 3-dB bandwidth (nm)
    /// * `required_nm`     – Minimum acceptable bandwidth (nm)
    ///
    /// Returns `true` if all components satisfy the bandwidth constraint.
    pub fn check_bandwidth(&self, component_bw_nm: &[f64], required_nm: f64) -> bool {
        component_bw_nm.iter().all(|&bw| bw >= required_nm)
    }

    /// Verify that no component exceeds the power handling threshold.
    ///
    /// # Arguments
    /// * `powers_mw`     – Per-component optical power (mW)
    /// * `threshold_mw`  – Maximum allowed power per component (mW)
    ///
    /// Returns `true` if all powers are below the threshold.
    pub fn check_power_handling(&self, powers_mw: &[f64], threshold_mw: f64) -> bool {
        powers_mw.iter().all(|&p| p <= threshold_mw)
    }

    /// Estimate the manufacturing yield from Gaussian performance distributions.
    ///
    /// Uses the standard normal CDF to compute the probability that each
    /// parameter falls within spec, then multiplies yields (independent components).
    ///
    /// # Arguments
    /// * `mean_performances` – Nominal (mean) performance values
    /// * `std_devs`          – Standard deviations (process variation)
    /// * `specs`             – Minimum acceptable performance values
    ///
    /// Returns yield fraction in [0, 1].
    pub fn yield_estimate(
        &self,
        mean_performances: &[f64],
        std_devs: &[f64],
        specs: &[f64],
    ) -> f64 {
        let n = mean_performances.len().min(std_devs.len()).min(specs.len());
        if n == 0 {
            return 1.0;
        }
        (0..n)
            .map(|i| {
                let sigma = std_devs[i].max(1.0e-15);
                let z = (mean_performances[i] - specs[i]) / sigma;
                // P(X ≥ spec) = Φ(z) using erfc approximation
                normal_cdf(z)
            })
            .product()
    }

    /// Monte Carlo yield simulation using a deterministic LCG.
    ///
    /// For each sample, draws normally-distributed performance values for every
    /// component and counts the fraction of samples that pass all specs.
    ///
    /// # Arguments
    /// * `n_samples` – Number of Monte Carlo trials
    /// * `means`     – Mean performance for each component
    /// * `stds`      – Standard deviation for each component
    /// * `specs`     – Minimum spec for each component (performance ≥ spec to pass)
    ///
    /// Returns yield fraction in [0, 1].
    pub fn monte_carlo_yield(
        &self,
        n_samples: usize,
        means: &[f64],
        stds: &[f64],
        specs: &[f64],
    ) -> f64 {
        if n_samples == 0 || means.is_empty() {
            return 1.0;
        }
        let n_comp = means.len().min(stds.len()).min(specs.len());
        let mut rng = Lcg::new(0xDEAD_BEEF_CAFE_1234);
        let mut pass = 0usize;
        for _ in 0..n_samples {
            let all_pass = (0..n_comp).all(|i| {
                let z = rng.next_normal();
                let val = means[i] + stds[i] * z;
                val >= specs[i]
            });
            if all_pass {
                pass += 1;
            }
        }
        pass as f64 / n_samples as f64
    }
}

/// Standard normal CDF Φ(x) approximated via the error function.
///
/// Φ(x) = (1 + erf(x / √2)) / 2
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

/// Error function approximation (Abramowitz & Stegun 7.1.26, max error < 1.5e-7).
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    sign * (1.0 - poly * (-x * x).exp())
}

// ─────────────────────────────────────────────────────────────────────────────
// ThermalAnalyzer
// ─────────────────────────────────────────────────────────────────────────────

/// Thermal analysis for thermo-optic phase shifters.
///
/// Models resistive heaters on a photonic waveguide and computes the resulting
/// optical phase shift via the thermo-optic effect (dn/dT).
#[derive(Debug, Clone)]
pub struct ThermalAnalyzer {
    /// Substrate (Si) thermal conductivity (W / m·K)
    pub substrate_thermal_conductivity: f64,
    /// Heater resistance (Ω)
    pub heater_resistance: f64,
    /// Heater length (µm)
    pub heater_length_um: f64,
}

impl ThermalAnalyzer {
    /// Build a thermal model for the standard SOI platform.
    ///
    /// Uses Si thermal conductivity of 148 W/(m·K) and a 10 kΩ doped-Si heater.
    pub fn new_soi() -> Self {
        Self {
            substrate_thermal_conductivity: 148.0,
            heater_resistance: 10_000.0,
            heater_length_um: 100.0,
        }
    }

    /// Temperature rise (K) in the waveguide core for a given heater power.
    ///
    /// Simplified thermal resistance model:
    /// R_th ≈ 1 / (2π · k · L)  (cylindrical heat flow from a line source)
    ///
    /// # Arguments
    /// * `power_mw` – Electrical power dissipated in the heater (mW)
    pub fn temperature_rise_k(&self, power_mw: f64) -> f64 {
        let p_w = power_mw * 1.0e-3;
        let l_m = self.heater_length_um * 1.0e-6;
        // Thermal resistance (K/W) for a heater of length L on Si substrate
        let r_th = 1.0 / (2.0 * PI * self.substrate_thermal_conductivity * l_m);
        p_w * r_th
    }

    /// Optical phase shift (rad) induced by the heater power.
    ///
    /// Δφ = (2π / λ) · (dn/dT) · ΔT · L_wg
    ///
    /// # Arguments
    /// * `power_mw`   – Heater power (mW)
    /// * `dn_dt`      – Thermo-optic coefficient (1/K); Si ≈ 1.86e-4 K⁻¹
    /// * `length_um`  – Waveguide overlap length with heater (µm)
    /// * `wavelength` – Design wavelength (m)
    pub fn phase_shift_rad(
        &self,
        power_mw: f64,
        dn_dt: f64,
        length_um: f64,
        wavelength: f64,
    ) -> f64 {
        let delta_t = self.temperature_rise_k(power_mw);
        let l_m = length_um * 1.0e-6;
        2.0 * PI / wavelength * dn_dt * delta_t * l_m
    }

    /// Power (mW) required for a π phase shift.
    ///
    /// P_pi = π / (dφ/dP)
    ///
    /// # Arguments
    /// * `dn_dt`      – Thermo-optic coefficient (1/K)
    /// * `length_um`  – Waveguide overlap length (µm)
    /// * `wavelength` – Design wavelength (m)
    pub fn vpi_equivalent_power_mw(&self, dn_dt: f64, length_um: f64, wavelength: f64) -> f64 {
        let dphi_dp = self.phase_shift_rad(1.0, dn_dt, length_um, wavelength);
        if dphi_dp.abs() < 1.0e-30 {
            return f64::INFINITY;
        }
        PI / dphi_dp
    }

    /// Thermo-optic bandwidth (Hz) limited by thermal diffusion.
    ///
    /// f_th ≈ k / (ρ · c_p · d²)
    ///
    /// For Si: k = 148 W/(m·K), ρ·c_p ≈ 1.63e6 J/(m³·K), d ~ 2 µm (BOX thickness).
    /// Typical value: ~ 100 kHz – 1 MHz.
    pub fn thermo_optic_bandwidth_hz(&self) -> f64 {
        let rho_cp = 1.63e6_f64; // J/(m³·K) for Si
        let d_m = 2.0e-6_f64; // characteristic diffusion length (BOX thickness)
        self.substrate_thermal_conductivity / (rho_cp * d_m * d_m)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── DRC tests ─────────────────────────────────────────────────────────────

    #[test]
    fn test_drc_width_ok() {
        let drc = DesignRuleChecker::new_soi_220nm();
        let v = drc.check_width(500.0, [0.0, 0.0]);
        assert!(v.is_none(), "500 nm should pass DRC for SOI 220 nm");
    }

    #[test]
    fn test_drc_width_violation() {
        let drc = DesignRuleChecker::new_soi_220nm();
        let v = drc.check_width(200.0, [10.0, 5.0]);
        assert!(v.is_some());
        let viol = v.unwrap();
        assert_eq!(viol.severity, DrcSeverity::Error);
    }

    #[test]
    fn test_drc_bend_radius_warning() {
        let drc = DesignRuleChecker::new_soi_220nm();
        // Just above minimum (5 µm) but inside warning band (< 7.5 µm)
        let v = drc.check_bend_radius(6.0, [20.0, 20.0]);
        assert!(v.is_some());
        assert_eq!(v.unwrap().severity, DrcSeverity::Warning);
    }

    #[test]
    fn test_drc_coupling_gap_error() {
        let drc = DesignRuleChecker::new_soi_220nm();
        let v = drc.check_coupling_gap(50.0, [100.0, 50.0]);
        assert!(v.is_some());
        assert_eq!(v.unwrap().severity, DrcSeverity::Error);
    }

    #[test]
    fn test_drc_check_all_no_violations() {
        let drc = DesignRuleChecker::new_soi_220nm();
        let violations = drc.check_all(450.0, 10.0, 200.0, [0.0, 0.0]);
        assert!(
            violations.is_empty(),
            "No violations expected for valid parameters"
        );
    }

    // ── CircuitVerifier tests ─────────────────────────────────────────────────

    #[test]
    fn test_loss_budget_pass() {
        let cv = CircuitVerifier::new(1530.0, 1570.0, 41);
        let losses = vec![1.0, 0.5, 0.3, 0.2];
        assert!(cv.check_loss_budget(&losses, 3.0));
    }

    #[test]
    fn test_loss_budget_fail() {
        let cv = CircuitVerifier::new(1530.0, 1570.0, 41);
        let losses = vec![2.0, 1.5, 0.5];
        assert!(!cv.check_loss_budget(&losses, 3.0));
    }

    #[test]
    fn test_bandwidth_check() {
        let cv = CircuitVerifier::new(1530.0, 1570.0, 41);
        assert!(cv.check_bandwidth(&[50.0, 60.0, 80.0], 40.0));
        assert!(!cv.check_bandwidth(&[50.0, 30.0, 80.0], 40.0));
    }

    #[test]
    fn test_yield_estimate_high_margin() {
        let cv = CircuitVerifier::new(1530.0, 1570.0, 41);
        // Mean >> spec → yield ≈ 1
        let yield_val = cv.yield_estimate(&[10.0, 20.0], &[0.1, 0.1], &[1.0, 2.0]);
        assert!(
            yield_val > 0.99,
            "Yield should be near 1 with high margin: {yield_val}"
        );
    }

    #[test]
    fn test_monte_carlo_yield_deterministic() {
        let cv = CircuitVerifier::new(1530.0, 1570.0, 41);
        let y1 = cv.monte_carlo_yield(1000, &[5.0], &[1.0], &[3.0]);
        let y2 = cv.monte_carlo_yield(1000, &[5.0], &[1.0], &[3.0]);
        // Deterministic: same seed → same result
        assert_abs_diff_eq!(y1, y2, epsilon = 1.0e-10);
    }

    #[test]
    fn test_monte_carlo_yield_ordering() {
        let cv = CircuitVerifier::new(1530.0, 1570.0, 41);
        // Tight spec (near mean) → lower yield than relaxed spec
        let y_tight = cv.monte_carlo_yield(2000, &[5.0], &[1.0], &[5.5]);
        let y_relaxed = cv.monte_carlo_yield(2000, &[5.0], &[1.0], &[3.0]);
        assert!(y_tight < y_relaxed, "Tighter spec should give lower yield");
    }

    // ── ThermalAnalyzer tests ────────────────────────────────────────────────

    #[test]
    fn test_temperature_rise_positive() {
        let ta = ThermalAnalyzer::new_soi();
        let dt = ta.temperature_rise_k(10.0);
        assert!(
            dt > 0.0 && dt.is_finite(),
            "ΔT should be positive and finite: {dt}"
        );
    }

    #[test]
    fn test_phase_shift_scales_with_power() {
        let ta = ThermalAnalyzer::new_soi();
        let phi_1 = ta.phase_shift_rad(5.0, 1.86e-4, 100.0, 1.55e-6);
        let phi_2 = ta.phase_shift_rad(10.0, 1.86e-4, 100.0, 1.55e-6);
        assert_abs_diff_eq!(phi_2, 2.0 * phi_1, epsilon = 1.0e-6);
    }

    #[test]
    fn test_vpi_power_finite() {
        let ta = ThermalAnalyzer::new_soi();
        let p_pi = ta.vpi_equivalent_power_mw(1.86e-4, 100.0, 1.55e-6);
        assert!(
            p_pi > 0.0 && p_pi.is_finite(),
            "Pπ should be finite positive: {p_pi}"
        );
    }

    #[test]
    fn test_thermo_optic_bandwidth() {
        let ta = ThermalAnalyzer::new_soi();
        let f = ta.thermo_optic_bandwidth_hz();
        // Expect roughly 10 kHz – 100 MHz range for Si heaters
        assert!(
            f > 1.0e3 && f < 1.0e9,
            "Thermal bandwidth out of range: {f} Hz"
        );
    }

    #[test]
    fn test_erf_at_zero() {
        assert_abs_diff_eq!(erf(0.0), 0.0, epsilon = 1.0e-7);
    }

    #[test]
    fn test_erf_large_positive() {
        assert_abs_diff_eq!(erf(4.0), 1.0, epsilon = 1.0e-5);
    }

    #[test]
    fn test_normal_cdf_symmetry() {
        assert_abs_diff_eq!(normal_cdf(0.0), 0.5, epsilon = 1.0e-7);
        let lo = normal_cdf(-2.0);
        let hi = normal_cdf(2.0);
        assert_abs_diff_eq!(lo + hi, 1.0, epsilon = 1.0e-6);
    }
}
