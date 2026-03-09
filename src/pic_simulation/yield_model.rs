/// Manufacturing yield and process-variability models for PICs.
///
/// Provides statistical models for:
/// - Width/height process variations and their effect on optical performance
/// - Die yield using Murphy's and Poisson models
/// - Monte Carlo yield simulation (LCG-based, no external RNG crate)
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Process Variation
// ---------------------------------------------------------------------------

/// Statistical process-variation model for silicon-photonics waveguides.
///
/// Captures the dominant fabrication imperfections (CD variation) and their
/// influence on effective index, coupling ratio, and MZI performance.
#[derive(Clone, Debug)]
pub struct ProcessVariation {
    /// 1σ waveguide width variation σ_w (nm).
    pub waveguide_width_sigma_nm: f64,
    /// 1σ waveguide height / thickness variation σ_h (nm).
    pub waveguide_height_sigma_nm: f64,
    /// Sensitivity of n_eff to width: dn_eff/dw (m^{-1}).
    pub n_eff_sensitivity: f64,
    /// Sensitivity of power coupling ratio to width: dκ/dw (m^{-1}).
    pub coupling_sensitivity: f64,
}

impl ProcessVariation {
    /// Typical 180-nm silicon photonics process (DUV lithography).
    pub fn typical_soi_180nm() -> Self {
        Self {
            waveguide_width_sigma_nm: 3.0,
            waveguide_height_sigma_nm: 1.5,
            n_eff_sensitivity: 3.0e6,    // dn_eff/dw ~ 3/μm
            coupling_sensitivity: 2.0e6, // dκ/dw ~ 2/μm
        }
    }

    /// Phase error per unit length due to width variation (rad/m).
    ///
    /// σ_φ/L = (2π/λ) × (dn_eff/dw) × σ_w
    pub fn phase_error_per_m(&self, wavelength_m: f64) -> f64 {
        2.0 * PI / wavelength_m * self.n_eff_sensitivity * self.waveguide_width_sigma_nm * 1e-9
    }

    /// 1σ coupling ratio variation due to width variation.
    ///
    /// σ_κ = (dκ/dw) × σ_w
    pub fn coupling_variation(&self) -> f64 {
        self.coupling_sensitivity * self.waveguide_width_sigma_nm * 1e-9
    }

    /// Expected MZI extinction ratio limited by phase error (dB).
    ///
    /// For a phase error σ_φ = σ_φ/L × L, the through-port null becomes
    /// non-ideal. A simple estimate: ER ≈ 20 log₁₀(1/σ_φ).
    pub fn mzi_extinction_ratio_db(&self, arm_length_m: f64, wavelength_m: f64) -> f64 {
        let sigma_phi = self.phase_error_per_m(wavelength_m) * arm_length_m;
        if sigma_phi <= 0.0 {
            return 60.0;
        }
        // ER_dB = -20 log10(sigma_phi / 2)  [factor 2 from MZI transfer]
        let er = -20.0 * (sigma_phi / 2.0).log10();
        er.clamp(0.0, 60.0)
    }

    /// Height-contribution to phase error per unit length (rad/m).
    ///
    /// Uses an approximated height sensitivity of 0.5 × n_eff_sensitivity
    /// (height variation is typically less impactful for rib waveguides).
    pub fn height_phase_error_per_m(&self, wavelength_m: f64) -> f64 {
        2.0 * PI / wavelength_m
            * (self.n_eff_sensitivity * 0.5)
            * self.waveguide_height_sigma_nm
            * 1e-9
    }

    /// Combined (RSS) phase error per unit length including both width and height.
    pub fn combined_phase_error_per_m(&self, wavelength_m: f64) -> f64 {
        let w = self.phase_error_per_m(wavelength_m);
        let h = self.height_phase_error_per_m(wavelength_m);
        (w * w + h * h).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Yield Model
// ---------------------------------------------------------------------------

/// Statistical die yield model.
///
/// Implements Murphy's and Poisson defect-density yield models,
/// which are standard in semiconductor and PIC manufacturing.
#[derive(Clone, Debug)]
pub struct YieldModel {
    /// Defect density D₀ (defects/cm²).
    pub defect_density_per_cm2: f64,
    /// Die area A (cm²).
    pub die_area_cm2: f64,
    /// Critical area fraction — fraction of die area sensitive to point defects.
    pub critical_area_fraction: f64,
}

impl YieldModel {
    /// Construct a yield model.
    pub fn new(
        defect_density_per_cm2: f64,
        die_area_cm2: f64,
        critical_area_fraction: f64,
    ) -> Self {
        Self {
            defect_density_per_cm2,
            die_area_cm2,
            critical_area_fraction: critical_area_fraction.clamp(0.0, 1.0),
        }
    }

    /// Effective defect-sensitive area.
    fn effective_area(&self) -> f64 {
        self.defect_density_per_cm2 * self.die_area_cm2 * self.critical_area_fraction
    }

    /// Murphy's yield: Y = (1 − exp(−D₀A)) / (D₀A).
    ///
    /// More accurate than Poisson for large defect products because it accounts
    /// for defect clustering.
    pub fn murphy_yield(&self) -> f64 {
        let da = self.effective_area();
        if da < 1e-10 {
            return 1.0;
        }
        (1.0 - (-da).exp()) / da
    }

    /// Poisson yield: Y = exp(−D₀A).
    ///
    /// Simplest model; accurate for small D₀A products.
    pub fn poisson_yield(&self) -> f64 {
        (-self.effective_area()).exp()
    }

    /// Negative-binomial yield with clustering parameter α.
    ///
    /// Y = (1 + D₀A/α)^{−α}.  α→∞ → Poisson; α→0 → clustered.
    pub fn negative_binomial_yield(&self, alpha: f64) -> f64 {
        let da = self.effective_area();
        (1.0 + da / alpha).powf(-alpha)
    }

    /// Break-even defect density for a target yield: D₀ = −ln(Y) / A.
    ///
    /// Derived from the Poisson model as a conservative upper bound.
    pub fn defect_density_for_yield(&self, target_yield: f64) -> f64 {
        let target = target_yield.clamp(1e-10, 1.0 - 1e-10);
        let a = self.die_area_cm2 * self.critical_area_fraction;
        if a <= 0.0 {
            return f64::INFINITY;
        }
        -target.ln() / a
    }

    /// Number of good dies expected on a full wafer (Murphy model).
    pub fn good_dies_on_wafer(&self, wafer_area_cm2: f64) -> f64 {
        self.murphy_yield() * (wafer_area_cm2 / self.die_area_cm2).floor()
    }

    /// Cost of good die ratio (good dies / total dies on wafer).
    pub fn cost_of_good_die(&self) -> f64 {
        self.murphy_yield()
    }

    /// Yield as a function of defect density (Poisson) for a sweep.
    pub fn yield_vs_defect_density(
        die_area_cm2: f64,
        critical_area_fraction: f64,
        defect_range: &[f64],
    ) -> Vec<(f64, f64)> {
        defect_range
            .iter()
            .map(|&d0| {
                let da = d0 * die_area_cm2 * critical_area_fraction;
                (d0, (-da).exp())
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Monte Carlo Yield Analysis
// ---------------------------------------------------------------------------

/// Monte Carlo yield analysis using a linear congruential generator.
///
/// The LCG avoids the `rand` crate while providing adequate statistical
/// quality for engineering-level yield estimation.
#[derive(Clone, Debug)]
pub struct MonteCarloYield {
    /// Number of simulated dies.
    pub n_samples: usize,
    /// Process variation model.
    pub variation: ProcessVariation,
    /// Maximum acceptable 1σ phase error (rad) — the spec limit.
    pub spec_phase_error_rad: f64,
}

/// LCG state update (Knuth / Steele MMIX parameters).
#[inline]
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

/// Draw a U[0,1) sample from LCG state.
#[inline]
fn lcg_uniform(state: u64) -> f64 {
    (state >> 33) as f64 / u32::MAX as f64
}

/// Box-Muller transform: two U[0,1) → N(0,1) sample (returns first of pair).
fn box_muller(u1: f64, u2: f64) -> f64 {
    let u1_safe = u1.max(1e-300); // guard against log(0)
    (-2.0 * u1_safe.ln()).sqrt() * (2.0 * PI * u2).cos()
}

impl MonteCarloYield {
    /// Construct a Monte Carlo yield analyser.
    pub fn new(n_samples: usize, variation: ProcessVariation, spec_phase_error_rad: f64) -> Self {
        Self {
            n_samples,
            variation,
            spec_phase_error_rad,
        }
    }

    /// Run LCG-based Monte Carlo simulation.
    ///
    /// Draws `n_samples` Gaussian phase errors and returns the fraction
    /// whose magnitude falls within ±spec_phase_error_rad.
    pub fn simulate(&self, arm_length_m: f64, wavelength_m: f64) -> f64 {
        if self.n_samples == 0 {
            return 0.0;
        }
        let sigma_phi = self.variation.phase_error_per_m(wavelength_m) * arm_length_m;
        let mut passes = 0usize;
        // Seed chosen to avoid trivial zero state
        let mut state: u64 = 0x_DEAD_BEEF_CAFE_1234;
        let limit = self.spec_phase_error_rad;

        let mut i = 0;
        while i < self.n_samples {
            state = lcg_next(state);
            let u1 = lcg_uniform(state);
            state = lcg_next(state);
            let u2 = lcg_uniform(state);
            let z = box_muller(u1, u2);
            let phi_err = sigma_phi * z;
            if phi_err.abs() <= limit {
                passes += 1;
            }
            i += 1;
        }
        passes as f64 / self.n_samples as f64
    }

    /// Simulate yield over a range of arm lengths.
    ///
    /// Returns a vector of `(arm_length_m, yield_fraction)` pairs.
    pub fn yield_vs_arm_length(
        &self,
        max_length_m: f64,
        wavelength_m: f64,
        n_points: usize,
    ) -> Vec<(f64, f64)> {
        if n_points == 0 {
            return Vec::new();
        }
        let step = max_length_m / n_points.max(1) as f64;
        (0..n_points)
            .map(|k| {
                let l = step * (k + 1) as f64;
                let y = self.simulate(l, wavelength_m);
                (l, y)
            })
            .collect()
    }

    /// Yield as a function of spec tightness (spec_sigma multiples).
    ///
    /// Returns `(sigma_multiple, yield_fraction)` pairs.
    pub fn yield_vs_spec_sigma(
        &self,
        arm_length_m: f64,
        wavelength_m: f64,
        sigma_multiples: &[f64],
    ) -> Vec<(f64, f64)> {
        sigma_multiples
            .iter()
            .map(|&s| {
                let sigma_phi = self.variation.phase_error_per_m(wavelength_m) * arm_length_m;
                let spec = s * sigma_phi;
                let mc = MonteCarloYield {
                    n_samples: self.n_samples,
                    variation: self.variation.clone(),
                    spec_phase_error_rad: spec,
                };
                let y = mc.simulate(arm_length_m, wavelength_m);
                (s, y)
            })
            .collect()
    }

    /// Estimate the minimum arm length that achieves a target yield.
    ///
    /// Uses bisection over [0, max_length_m].
    pub fn min_length_for_yield(
        &self,
        target_yield: f64,
        max_length_m: f64,
        wavelength_m: f64,
    ) -> Option<f64> {
        let y_at_zero = self.simulate(1e-9, wavelength_m); // near-zero length
        if y_at_zero < target_yield {
            return None;
        } // even zero length fails
        let mut lo = 0.0_f64;
        let mut hi = max_length_m;
        for _ in 0..50 {
            let mid = (lo + hi) / 2.0;
            let y = self.simulate(mid, wavelength_m);
            if y >= target_yield {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(lo)
    }
}

// ---------------------------------------------------------------------------
// Trim Correction
// ---------------------------------------------------------------------------

/// Post-fabrication trimming model (thermal or UV trimming of n_eff).
///
/// Estimates the phase correction range required to compensate for
/// process variation and the associated heating power.
#[derive(Clone, Debug)]
pub struct TrimCorrection {
    /// Thermo-optic coefficient dn_eff/dT (K^{-1}).
    pub dn_dt: f64,
    /// Heater efficiency (K/mW).
    pub heater_efficiency_k_per_mw: f64,
    /// Waveguide arm length (m).
    pub arm_length_m: f64,
    /// Operating wavelength (m).
    pub wavelength_m: f64,
}

impl TrimCorrection {
    /// Typical silicon thermo-optic trim parameters.
    pub fn typical_si(arm_length_m: f64, wavelength_m: f64) -> Self {
        Self {
            dn_dt: 1.86e-4,
            heater_efficiency_k_per_mw: 10.0,
            arm_length_m,
            wavelength_m,
        }
    }

    /// Phase tuning range for a given heater power (rad).
    pub fn phase_range_rad(&self, heater_power_mw: f64) -> f64 {
        let delta_t = heater_power_mw * self.heater_efficiency_k_per_mw;
        let delta_n = self.dn_dt * delta_t;
        2.0 * PI * delta_n * self.arm_length_m / self.wavelength_m
    }

    /// Required heater power to correct a given phase error (mW).
    pub fn required_power_mw(&self, phase_error_rad: f64) -> f64 {
        let delta_n = phase_error_rad * self.wavelength_m / (2.0 * PI * self.arm_length_m);
        let delta_t = delta_n / self.dn_dt;
        delta_t / self.heater_efficiency_k_per_mw
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn murphy_yield_low_defect() {
        let model = YieldModel {
            defect_density_per_cm2: 0.1,
            die_area_cm2: 1.0,
            critical_area_fraction: 1.0,
        };
        let y = model.murphy_yield();
        // Murphy: (1 - exp(-0.1)) / 0.1 ≈ 0.9516
        assert!((y - 0.9516).abs() < 0.01, "Y={}", y);
    }

    #[test]
    fn poisson_yield_high_defect_low() {
        let model = YieldModel {
            defect_density_per_cm2: 5.0,
            die_area_cm2: 1.0,
            critical_area_fraction: 1.0,
        };
        let y = model.poisson_yield();
        assert!(y < 0.01, "Y={}", y);
    }

    #[test]
    fn murphy_yield_zero_defects_is_one() {
        let model = YieldModel {
            defect_density_per_cm2: 0.0,
            die_area_cm2: 1.0,
            critical_area_fraction: 1.0,
        };
        assert!((model.murphy_yield() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn yield_monotone_decreasing_with_defects() {
        let y1 = YieldModel::new(0.1, 1.0, 1.0).murphy_yield();
        let y2 = YieldModel::new(1.0, 1.0, 1.0).murphy_yield();
        let y3 = YieldModel::new(10.0, 1.0, 1.0).murphy_yield();
        assert!(y1 > y2 && y2 > y3);
    }

    #[test]
    fn process_variation_phase_error_positive() {
        let pv = ProcessVariation::typical_soi_180nm();
        let err = pv.phase_error_per_m(1550e-9);
        assert!(err > 0.0, "phase_error={}", err);
    }

    #[test]
    fn monte_carlo_high_spec_near_unity() {
        let pv = ProcessVariation::typical_soi_180nm();
        let mc = MonteCarloYield::new(1000, pv, 1000.0); // very loose spec
        let y = mc.simulate(100e-6, 1550e-9);
        assert!(y > 0.95, "yield={}", y);
    }

    #[test]
    fn monte_carlo_tight_spec_lower_yield() {
        let pv = ProcessVariation::typical_soi_180nm();
        let mc_loose = MonteCarloYield::new(500, pv.clone(), 1.0);
        let mc_tight = MonteCarloYield::new(500, pv, 0.001);
        let y_loose = mc_loose.simulate(1e-3, 1550e-9);
        let y_tight = mc_tight.simulate(1e-3, 1550e-9);
        assert!(y_loose >= y_tight, "loose={}, tight={}", y_loose, y_tight);
    }

    #[test]
    fn yield_vs_arm_length_returns_correct_count() {
        let pv = ProcessVariation::typical_soi_180nm();
        let mc = MonteCarloYield::new(200, pv, 0.5);
        let sweep = mc.yield_vs_arm_length(1e-3, 1550e-9, 5);
        assert_eq!(sweep.len(), 5);
    }

    #[test]
    fn trim_correction_round_trip() {
        let trim = TrimCorrection::typical_si(500e-6, 1550e-9);
        let target_phase = 0.5;
        let power_mw = trim.required_power_mw(target_phase);
        let phase_back = trim.phase_range_rad(power_mw);
        assert!(
            (phase_back - target_phase).abs() < 1e-10,
            "Δφ={}",
            phase_back
        );
    }
}
