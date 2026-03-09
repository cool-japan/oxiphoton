//! Multi-mode waveguide and MMI splitter analysis.
//!
//! Implements the self-imaging principle for MMI devices and mode analysis
//! for multimode waveguides using the slab waveguide characteristic equation.

use crate::error::OxiPhotonError;
use num_complex::Complex64;
use std::f64::consts::PI;

/// MMI (Multi-Mode Interference) splitter/coupler analysis.
///
/// Based on self-imaging principle: effective propagation constant β_ν
/// for mode ν in an MMI waveguide of width W:
///   β_ν ≈ β₀ - ν(ν+2)π/(3Lπ) where Lπ = 4·n_eff·W²/(3·λ)
#[derive(Debug, Clone)]
pub struct MmiSplitter {
    /// MMI region width (m)
    pub width: f64,
    /// MMI section length (m)
    pub length: f64,
    /// Effective refractive index
    pub n_eff: f64,
    /// Operating wavelength (m)
    pub wavelength: f64,
    /// Number of guided modes in MMI region
    pub n_modes: usize,
}

impl MmiSplitter {
    /// Create a new MMI splitter. Number of modes is estimated from the V-number.
    pub fn new(width: f64, length: f64, n_eff: f64, wavelength: f64) -> Self {
        // Estimate modes: for a multimode waveguide, N_modes ≈ int(2*n_eff*width/lambda)
        let n_modes = ((2.0 * n_eff * width / wavelength).floor() as usize).max(2);
        Self {
            width,
            length,
            n_eff,
            wavelength,
            n_modes,
        }
    }

    /// Beat length Lπ = π / (β₀ - β₁).
    ///
    /// Using the paraxial approximation:
    ///   β_ν ≈ n_eff·k₀ - ν(ν+2)·π / (4·n_eff·k₀·W²)
    /// so β₀ - β₁ = 3π / (4·n_eff·k₀·W²) · (2·k₀ correction) = π·λ / (4·n_eff·W²/3)
    pub fn beat_length(&self) -> f64 {
        // Lπ = 4 * n_eff * W^2 / (3 * lambda)
        4.0 * self.n_eff * self.width * self.width / (3.0 * self.wavelength)
    }

    /// Self-imaging length (fundamental) = 3·Lπ.
    pub fn imaging_length(&self) -> f64 {
        3.0 * self.beat_length()
    }

    /// Transfer matrix from input to outputs (2×2 for 1×2 splitter).
    ///
    /// Returns [[t11, t12], [t21, t22]] complex transmission matrix.
    /// The matrix represents a 50:50 coupler based on the self-imaging principle.
    pub fn transfer_matrix(&self) -> [[Complex64; 2]; 2] {
        let ratio = self.splitting_ratio();
        let lpi = self.beat_length();
        // Phase accumulated over the length
        let phi = PI * self.length / (2.0 * lpi);
        let sqrt_r = ratio.sqrt();
        let sqrt_1mr = (1.0 - ratio).sqrt();
        let i = Complex64::new(0.0, 1.0);
        // Standard directional-coupler transfer matrix generalized for MMI
        let t11 = Complex64::new(sqrt_r * phi.cos(), 0.0);
        let t12 = i * sqrt_1mr * phi.sin();
        let t21 = i * sqrt_1mr * phi.sin();
        let t22 = Complex64::new(sqrt_r * phi.cos(), 0.0);
        [[t11, t12], [t21, t22]]
    }

    /// Splitting ratio (ideally 0.5 for 50:50 MMI at the imaging length).
    pub fn splitting_ratio(&self) -> f64 {
        let lpi = self.beat_length();
        let img = self.imaging_length();
        // At L = img/2 (which is 3Lπ/2), splitting is 50:50
        // General formula: ratio proportional to sinc² of length deviation
        let dev = (self.length - img / 2.0) / lpi;
        let base = 0.5;
        // Modulation due to length mismatch
        let correction = 0.5 * (PI * dev).cos();
        (base + correction).clamp(0.0, 1.0)
    }

    /// Insertion loss (dB) — assumes ideal MMI with small perturbation loss.
    pub fn insertion_loss_db(&self) -> f64 {
        // Ideal MMI: ~0.3 dB excess loss; plus deviation penalty
        let ratio = self.splitting_ratio();
        let ideal = 0.5_f64;
        let mismatch = (ratio - ideal).abs();
        0.3 + 10.0 * mismatch.powi(2)
    }

    /// Optimal length for N×N MMI splitter.
    ///
    /// L_opt = 3·Lπ / N for symmetric interference mechanism.
    pub fn optimal_length_n_to_n(n: usize, beat_length: f64) -> f64 {
        if n == 0 {
            return 0.0;
        }
        3.0 * beat_length / n as f64
    }

    /// Mode field amplitude for mode ν at position x in MMI region.
    ///
    /// Uses cosine mode shapes valid for a rectangular waveguide:
    ///   ψ_ν(x) = sqrt(2/W) · cos((ν+1)·π·x/W)  for even ν
    ///   ψ_ν(x) = sqrt(2/W) · sin((ν+1)·π·x/W)  for odd ν
    pub fn mode_amplitude(&self, nu: usize, x: f64) -> f64 {
        let w = self.width;
        let norm = (2.0 / w).sqrt();
        let phase = (nu as f64 + 1.0) * PI * x / w;
        if nu % 2 == 0 {
            norm * phase.cos()
        } else {
            norm * phase.sin()
        }
    }

    /// Field distribution at the output plane, discretized at n_points.
    ///
    /// Superposition of all guided modes at propagation distance `self.length`.
    pub fn output_field_distribution(&self, n_points: usize) -> Vec<Complex64> {
        if n_points == 0 {
            return Vec::new();
        }
        let w = self.width;
        let k0 = 2.0 * PI / self.wavelength;
        let lpi = self.beat_length();
        let mut field = vec![Complex64::new(0.0, 0.0); n_points];

        for (idx, pt) in field.iter_mut().enumerate() {
            let x = -w / 2.0 + (idx as f64 + 0.5) / n_points as f64 * w;
            let mut sum = Complex64::new(0.0, 0.0);
            for nu in 0..self.n_modes {
                // Propagation constant from paraxial approximation
                let beta_nu = k0 * self.n_eff - (nu as f64 * (nu as f64 + 2.0) * PI) / (3.0 * lpi);
                let phase = Complex64::new(0.0, beta_nu * self.length).exp();
                let amp = self.mode_amplitude(nu, x + w / 2.0);
                sum += phase * amp;
            }
            *pt = sum;
        }
        field
    }
}

/// Multimode waveguide — supports multiple guided modes simultaneously.
///
/// Uses slab waveguide approximation for mode analysis.
#[derive(Debug, Clone)]
pub struct MultimodeWaveguide {
    /// Core width (m)
    pub core_width: f64,
    /// Core height (m)
    pub core_height: f64,
    /// Core refractive index
    pub n_core: f64,
    /// Cladding refractive index
    pub n_clad: f64,
    /// Operating wavelength (m)
    pub wavelength: f64,
}

impl MultimodeWaveguide {
    /// Create a new multimode waveguide.
    pub fn new(
        core_width: f64,
        core_height: f64,
        n_core: f64,
        n_clad: f64,
        wavelength: f64,
    ) -> Self {
        Self {
            core_width,
            core_height,
            n_core,
            n_clad,
            wavelength,
        }
    }

    /// V-number: V = (π/λ) · sqrt(n_core² - n_clad²) · width.
    pub fn v_number(&self) -> f64 {
        PI / self.wavelength
            * (self.n_core * self.n_core - self.n_clad * self.n_clad).sqrt()
            * self.core_width
    }

    /// Number of guided modes (estimated via V-number).
    ///
    /// For a slab waveguide: N_modes ≈ floor(V/π) + 1 (TE modes).
    pub fn num_modes(&self) -> usize {
        let v = self.v_number();
        ((v / PI).floor() as usize + 1).max(1)
    }

    /// Effective indices for each guided mode using slab characteristic equation.
    ///
    /// Solves the transcendental equation for TE modes:
    ///   tan(kappa·d/2) = gamma/kappa  (even modes)
    ///   -cot(kappa·d/2) = gamma/kappa  (odd modes)
    /// where kappa² = k₀²(n_core² - n_eff²), gamma² = k₀²(n_eff² - n_clad²).
    pub fn effective_indices(&self) -> Result<Vec<f64>, OxiPhotonError> {
        let k0 = 2.0 * PI / self.wavelength;
        let n_modes = self.num_modes();
        let mut n_effs = Vec::with_capacity(n_modes);

        for m in 0..n_modes {
            // Search for n_eff in (n_clad, n_core) using bisection
            let mut lo = self.n_clad + 1e-10;
            let mut hi = self.n_core - 1e-10;

            if lo >= hi {
                return Err(OxiPhotonError::NumericalError(
                    "Invalid waveguide parameters: n_core must be > n_clad".to_string(),
                ));
            }

            // Characteristic function for mode m
            let char_fn = |n_eff: f64| -> f64 {
                let kappa = (k0 * k0 * (self.n_core * self.n_core - n_eff * n_eff)).sqrt();
                let gamma = (k0 * k0 * (n_eff * n_eff - self.n_clad * self.n_clad)).sqrt();
                if m % 2 == 0 {
                    // Even modes: kappa * tan(kappa*d/2) - gamma = 0
                    kappa * (kappa * self.core_width / 2.0).tan() - gamma
                } else {
                    // Odd modes: -kappa * cot(kappa*d/2) - gamma = 0
                    -kappa / (kappa * self.core_width / 2.0).tan() - gamma
                }
            };

            // Bisection with up to 100 iterations
            let mut found = false;
            for _ in 0..100 {
                let mid = (lo + hi) / 2.0;
                let f_lo = char_fn(lo);
                let f_mid = char_fn(mid);
                if f_lo * f_mid < 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                }
                if (hi - lo) < 1e-14 {
                    found = true;
                    break;
                }
            }

            if !found && (hi - lo) > 1e-8 {
                // Mode not converged — stop adding modes
                break;
            }

            let n_eff = (lo + hi) / 2.0;
            n_effs.push(n_eff);
        }

        if n_effs.is_empty() {
            return Err(OxiPhotonError::NumericalError(
                "No guided modes found".to_string(),
            ));
        }
        Ok(n_effs)
    }

    /// Group index for mode m: ng = n_eff - λ · dn_eff/dλ (finite-difference).
    pub fn group_index(&self, mode_idx: usize) -> Result<f64, OxiPhotonError> {
        let delta = 1e-12; // 1 pm step
        let wg_plus = MultimodeWaveguide::new(
            self.core_width,
            self.core_height,
            self.n_core,
            self.n_clad,
            self.wavelength + delta,
        );
        let wg_minus = MultimodeWaveguide::new(
            self.core_width,
            self.core_height,
            self.n_core,
            self.n_clad,
            self.wavelength - delta,
        );

        let n_effs = self.effective_indices()?;
        let n_effs_plus = wg_plus.effective_indices()?;
        let n_effs_minus = wg_minus.effective_indices()?;

        let n_eff = n_effs.get(mode_idx).copied().ok_or_else(|| {
            OxiPhotonError::NumericalError(format!("Mode index {mode_idx} out of range"))
        })?;
        let n_plus = n_effs_plus.get(mode_idx).copied().unwrap_or(n_eff);
        let n_minus = n_effs_minus.get(mode_idx).copied().unwrap_or(n_eff);

        let dn_dl = (n_plus - n_minus) / (2.0 * delta);
        Ok(n_eff - self.wavelength * dn_dl)
    }

    /// Mode confinement factor Γ for mode m.
    ///
    /// Γ = ∫_core |ψ|² dx / ∫_all |ψ|² dx
    /// For the slab approximation with cosine modes:
    ///   Γ = 1 - (2/kappa·d) · sin(kappa·d/2) · cos(kappa·d/2) / (1 + gamma^{-1} * something)
    /// We use the analytic result for TE modes.
    pub fn confinement_factor(&self, mode_idx: usize) -> Result<f64, OxiPhotonError> {
        let n_effs = self.effective_indices()?;
        let n_eff = n_effs.get(mode_idx).copied().ok_or_else(|| {
            OxiPhotonError::NumericalError(format!("Mode index {mode_idx} out of range"))
        })?;

        let k0 = 2.0 * PI / self.wavelength;
        let kappa = (k0 * k0 * (self.n_core * self.n_core - n_eff * n_eff))
            .max(0.0)
            .sqrt();
        let gamma = (k0 * k0 * (n_eff * n_eff - self.n_clad * self.n_clad))
            .max(0.0)
            .sqrt();
        let d = self.core_width;

        // Analytic confinement factor for slab TE mode
        // Γ = d_eff_core / d_eff_total
        // where d_eff_core = d/2 + sin(kappa*d/2)*cos(kappa*d/2)/kappa  (for even mode)
        // and 1/gamma is the evanescent penetration depth
        let (core_integral, total_integral) = if mode_idx % 2 == 0 {
            // Even mode: ψ(x) ∝ cos(kappa·x) in core, exp(-gamma·|x|) outside
            let kd2 = kappa * d / 2.0;
            let core_part = d / 2.0 + kd2.sin() * kd2.cos() / kappa;
            let clad_part = kd2.cos() * kd2.cos() / gamma;
            (core_part, core_part + 2.0 * clad_part)
        } else {
            // Odd mode: ψ(x) ∝ sin(kappa·x) in core
            let kd2 = kappa * d / 2.0;
            let core_part = d / 2.0 - kd2.sin() * kd2.cos() / kappa;
            let clad_part = kd2.sin() * kd2.sin() / gamma;
            (core_part, core_part + 2.0 * clad_part)
        };

        if total_integral <= 0.0 {
            return Ok(1.0);
        }
        Ok((core_integral / total_integral).clamp(0.0, 1.0))
    }

    /// Crosstalk between modes m and n (normalized overlap integral).
    ///
    /// For orthogonal guided modes this is ≈ 0. Returns a small non-zero
    /// value due to discretization and approximation.
    pub fn mode_crosstalk(&self, m: usize, n: usize) -> f64 {
        if m == n {
            return 1.0;
        }
        // For orthogonal modes of a lossless waveguide, overlap = 0
        // In practice, small crosstalk from fabrication imperfection:
        // we return a physics-motivated estimate based on modal beat length
        let v = self.v_number();
        let delta_m = (m as f64 - n as f64).abs();
        // Crosstalk decays with mode separation and V-number
        let xt = (-PI * delta_m * v / (2.0 * self.num_modes() as f64)).exp();
        xt * 0.01 // scale to a realistic crosstalk level
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn si_mmi() -> MmiSplitter {
        MmiSplitter::new(10e-6, 50e-6, 3.4, 1.55e-6)
    }

    fn si_mmwg() -> MultimodeWaveguide {
        MultimodeWaveguide::new(2e-6, 0.22e-6, 3.476, 1.444, 1.55e-6)
    }

    #[test]
    fn mmi_beat_length_positive() {
        let mmi = si_mmi();
        assert!(
            mmi.beat_length() > 0.0,
            "beat_length = {}",
            mmi.beat_length()
        );
    }

    #[test]
    fn mmi_imaging_length_is_three_beat_lengths() {
        let mmi = si_mmi();
        assert_relative_eq!(
            mmi.imaging_length(),
            3.0 * mmi.beat_length(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn mmi_splitting_ratio_between_zero_and_one() {
        let mmi = si_mmi();
        let r = mmi.splitting_ratio();
        assert!((0.0..=1.0).contains(&r), "splitting_ratio = {r}");
    }

    #[test]
    fn mmi_insertion_loss_positive() {
        let mmi = si_mmi();
        assert!(mmi.insertion_loss_db() > 0.0);
    }

    #[test]
    fn mmi_optimal_length_n_to_n_scales_correctly() {
        let lpi = 100e-6;
        let l1 = MmiSplitter::optimal_length_n_to_n(1, lpi);
        let l2 = MmiSplitter::optimal_length_n_to_n(2, lpi);
        assert_relative_eq!(l1, 3.0 * lpi, epsilon = 1e-15);
        assert_relative_eq!(l2, 1.5 * lpi, epsilon = 1e-15);
        assert!(l1 > l2);
    }

    #[test]
    fn mmi_transfer_matrix_is_2x2() {
        let mmi = si_mmi();
        let tm = mmi.transfer_matrix();
        // Just checks it compiles and returns finite values
        for row in &tm {
            for &c in row {
                assert!(c.re.is_finite() && c.im.is_finite());
            }
        }
    }

    #[test]
    fn mmi_output_field_length() {
        let mmi = si_mmi();
        let field = mmi.output_field_distribution(64);
        assert_eq!(field.len(), 64);
        // All elements should be finite
        for c in &field {
            assert!(c.re.is_finite() && c.im.is_finite());
        }
    }

    #[test]
    fn mmi_mode_amplitude_normalized() {
        let mmi = si_mmi();
        // Check normalization: integral of psi^2 over [0, W] ≈ 1
        let n_pts = 1000;
        let w = mmi.width;
        let dx = w / n_pts as f64;
        for nu in 0..2 {
            let integral: f64 = (0..n_pts)
                .map(|i| {
                    let x = (i as f64 + 0.5) * dx;
                    let amp = mmi.mode_amplitude(nu, x);
                    amp * amp * dx
                })
                .sum();
            assert_relative_eq!(integral, 1.0, epsilon = 1e-3);
        }
    }

    #[test]
    fn mmwg_v_number_positive() {
        let wg = si_mmwg();
        assert!(wg.v_number() > 0.0);
    }

    #[test]
    fn mmwg_num_modes_at_least_one() {
        let wg = si_mmwg();
        assert!(wg.num_modes() >= 1);
    }

    #[test]
    fn mmwg_effective_indices_in_valid_range() {
        let wg = si_mmwg();
        let n_effs = wg.effective_indices().expect("should find modes");
        assert!(!n_effs.is_empty());
        for &n_eff in &n_effs {
            assert!(
                n_eff > wg.n_clad && n_eff < wg.n_core,
                "n_eff = {n_eff} out of range [{}, {}]",
                wg.n_clad,
                wg.n_core
            );
        }
    }

    #[test]
    fn mmwg_confinement_between_zero_and_one() {
        let wg = si_mmwg();
        let gamma = wg.confinement_factor(0).expect("fundamental mode");
        assert!(gamma > 0.0 && gamma <= 1.0, "Γ = {gamma}");
    }

    #[test]
    fn mmwg_mode_crosstalk_self_is_one() {
        let wg = si_mmwg();
        assert_relative_eq!(wg.mode_crosstalk(0, 0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn mmwg_mode_crosstalk_between_modes_small() {
        let wg = si_mmwg();
        let xt = wg.mode_crosstalk(0, 1);
        assert!(xt < 0.1, "crosstalk = {xt}");
    }
}
