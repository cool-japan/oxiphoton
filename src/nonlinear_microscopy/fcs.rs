//! Fluorescence Correlation Spectroscopy (FCS) and FCCS
//!
//! FCS analyses fluctuations of fluorescence intensity in a small (<1 fL) confocal
//! volume to extract diffusion coefficients, concentrations, hydrodynamic radii,
//! and molecular interaction parameters.
//!
//! The normalised autocorrelation function for 3D diffusion in a Gaussian PSF is:
//!
//! `G(τ) = (1/N) · (1 + τ/τ_D)⁻¹ · (1 + τ/κ²τ_D)^{−1/2}`
//!
//! where `N` = mean number of molecules in V_eff, `τ_D = ω_xy²/(4D)` is the
//! lateral diffusion time, and `κ = ω_z/ω_xy` is the axial-to-lateral ratio.
//!
//! # References
//! - Elson & Magde, Biopolymers 13, 1 (1974)
//! - Rigler et al., Eur. Biophys. J. 22, 169 (1993)
//! - Schwille et al., Biophys. J. 72, 1878 (1997)

use std::f64::consts::PI;

/// Speed of light \[m/s\]
pub const C_LIGHT: f64 = 2.99792458e8;
/// Planck constant \[J·s\]
pub const H_PLANCK: f64 = 6.62607015e-34;
/// Reduced Planck constant \[J·s\]
pub const HBAR: f64 = 1.054571817e-34;
/// Boltzmann constant \[J/K\]
pub const KB: f64 = 1.380649e-23;
/// Avogadro number [mol⁻¹]
const AVOGADRO: f64 = 6.02214076e23;

/// FCS confocal setup parameters.
///
/// The 3D Gaussian PSF is parameterised by lateral beam waist ω_xy and
/// axial beam waist ω_z. The ratio κ = ω_z/ω_xy is typically 3–10 for
/// water-immersion objectives.
#[derive(Debug, Clone)]
pub struct FcsSetup {
    /// Lateral beam waist ω_xy (1/e² intensity radius) \[m\]
    pub beam_waist_xy_m: f64,
    /// Axial beam waist ω_z (1/e² intensity radius) \[m\]
    pub beam_waist_z_m: f64,
    /// Excitation wavelength \[m\]
    pub wavelength_m: f64,
}

impl FcsSetup {
    /// Construct an FCS setup.
    pub fn new(beam_waist_xy_m: f64, beam_waist_z_m: f64, wavelength_m: f64) -> Self {
        Self {
            beam_waist_xy_m,
            beam_waist_z_m,
            wavelength_m,
        }
    }

    /// Axial-to-lateral ratio κ = ω_z / ω_xy.
    pub fn kappa(&self) -> f64 {
        self.beam_waist_z_m / self.beam_waist_xy_m.max(f64::EPSILON)
    }

    /// Lateral diffusion time τ_D = ω_xy² / (4D) \[s\].
    ///
    /// # Arguments
    /// * `diffusion_coeff_m2_s` — Translational diffusion coefficient D \[m²/s\]
    pub fn diffusion_time(&self, diffusion_coeff_m2_s: f64) -> f64 {
        self.beam_waist_xy_m * self.beam_waist_xy_m / (4.0 * diffusion_coeff_m2_s.max(f64::EPSILON))
    }

    /// Normalised autocorrelation function G(τ) for a single diffusing species.
    ///
    /// `G(τ) = (1/N) · (1 + τ/τ_D)⁻¹ · (1 + τ/(κ² τ_D))^{−1/2}`
    ///
    /// # Arguments
    /// * `tau_s`          — Lag time τ \[s\]
    /// * `n_molecules`    — Mean number of molecules in the focal volume ‹N›
    /// * `diffusion_coeff`— Diffusion coefficient D \[m²/s\]
    pub fn autocorrelation(&self, tau_s: f64, n_molecules: f64, diffusion_coeff: f64) -> f64 {
        let tau_d = self.diffusion_time(diffusion_coeff);
        let kappa = self.kappa();
        let lat = 1.0 + tau_s / tau_d;
        let ax = 1.0 + tau_s / (kappa * kappa * tau_d);
        (1.0 / n_molecules.max(f64::EPSILON)) * (1.0 / lat) * (1.0 / ax.sqrt())
    }

    /// G(0) = 1/‹N› — zero-lag autocorrelation amplitude.
    ///
    /// From G(0) one directly reads the mean occupancy N of the focal volume.
    pub fn g_zero(&self, n_molecules: f64) -> f64 {
        1.0 / n_molecules.max(f64::EPSILON)
    }

    /// Effective focal volume V_eff = π^{3/2} ω_xy² ω_z \[m³\].
    ///
    /// This is the "observation volume" from which fluorescence fluctuations
    /// originate. Typical values: 0.1–1 fL for water-immersion objectives.
    pub fn focal_volume_m3(&self) -> f64 {
        PI.powf(1.5) * self.beam_waist_xy_m * self.beam_waist_xy_m * self.beam_waist_z_m
    }

    /// Molecular concentration from the zero-lag amplitude G(0) = 1/N \[mol/L\].
    ///
    /// `C = N / (V_eff · N_A)` in mol/L
    ///
    /// # Arguments
    /// * `g0` — Measured G(0) amplitude
    pub fn concentration_from_g0(&self, g0: f64) -> f64 {
        let n = 1.0 / g0.max(f64::EPSILON);
        let v_eff = self.focal_volume_m3();
        // mol/m³ → mol/L: divide by 1000
        n / (v_eff * AVOGADRO * 1000.0)
    }

    /// Autocorrelation with triplet-state dark fraction.
    ///
    /// Rapid triplet blinking (τ_T ≪ τ_D) adds a fast component:
    ///
    /// `G_T(τ) = G(τ) × [(1 − T + T·exp(−τ/τ_T)) / (1−T)]`
    ///
    /// # Arguments
    /// * `tau_s`           — Lag time \[s\]
    /// * `n_molecules`     — Mean occupancy ‹N›
    /// * `diffusion_coeff` — D \[m²/s\]
    /// * `triplet_fraction`— T: fraction of molecules in dark triplet state
    /// * `triplet_time_s`  — τ_T: triplet-state lifetime \[s\]
    pub fn autocorrelation_with_triplet(
        &self,
        tau_s: f64,
        n_molecules: f64,
        diffusion_coeff: f64,
        triplet_fraction: f64,
        triplet_time_s: f64,
    ) -> f64 {
        let g_diff = self.autocorrelation(tau_s, n_molecules, diffusion_coeff);
        let t = triplet_fraction.clamp(0.0, 0.9999);
        let triplet_factor =
            (1.0 - t + t * (-tau_s / triplet_time_s.max(f64::EPSILON)).exp()) / (1.0 - t);
        g_diff * triplet_factor
    }

    /// Molecular brightness B = ‹I› / ‹N› \[counts/s/molecule\].
    ///
    /// # Arguments
    /// * `avg_intensity_counts_s` — Mean count rate \[counts/s\]
    /// * `n_molecules`            — Mean occupancy ‹N›
    pub fn molecular_brightness(&self, avg_intensity_counts_s: f64, n_molecules: f64) -> f64 {
        avg_intensity_counts_s / n_molecules.max(f64::EPSILON)
    }
}

/// Fluorescence Cross-Correlation Spectroscopy (FCCS) — two-channel detection.
///
/// FCCS measures the cross-correlation between two spectrally distinct channels.
/// A non-zero cross-correlation amplitude indicates co-diffusion of doubly-labelled
/// molecules, enabling quantification of binding events.
#[derive(Debug, Clone)]
pub struct FccsMeasurement {
    /// FCS setup for detection channel 1 (e.g., green)
    pub fcs_ch1: FcsSetup,
    /// FCS setup for detection channel 2 (e.g., red)
    pub fcs_ch2: FcsSetup,
    /// Fractional overlap of the two focal volumes (0 = no overlap, 1 = perfect)
    pub overlap: f64,
}

impl FccsMeasurement {
    /// Construct an FCCS measurement.
    pub fn new(fcs_ch1: FcsSetup, fcs_ch2: FcsSetup, overlap: f64) -> Self {
        Self {
            fcs_ch1,
            fcs_ch2,
            overlap: overlap.clamp(0.0, 1.0),
        }
    }

    /// Cross-correlation amplitude G_cross(0).
    ///
    /// `G_cross(0) = N_double / (N_ch1 · N_ch2) · overlap`
    ///
    /// where N_double is the mean number of doubly-labelled (bound) molecules.
    ///
    /// # Arguments
    /// * `n_ch1`   — Mean occupancy in channel 1
    /// * `n_ch2`   — Mean occupancy in channel 2
    /// * `n_double`— Mean occupancy of doubly-labelled species
    pub fn cross_correlation_amplitude(&self, n_ch1: f64, n_ch2: f64, n_double: f64) -> f64 {
        let denom = n_ch1 * n_ch2;
        if denom < f64::EPSILON {
            return 0.0;
        }
        n_double / denom * self.overlap
    }

    /// Bound fraction from cross-correlation amplitude ratio.
    ///
    /// `f_bound = G_cross(0) · N_total`
    ///
    /// This gives the average fraction of molecules that are in a bound/complex state.
    ///
    /// # Arguments
    /// * `g_cross`  — Measured G_cross(0) amplitude
    /// * `n_total`  — Total mean occupancy (both species)
    pub fn bound_fraction(&self, g_cross: f64, n_total: f64) -> f64 {
        (g_cross * n_total).clamp(0.0, 1.0)
    }
}

/// FCS curve fitting — extract diffusion coefficient D and mean occupancy N.
///
/// Uses a simple gradient-descent (Levenberg–Marquardt inspired) approach to
/// minimise chi-squared between measured G(τ) data and the 3D Gaussian model.
/// In production one would use a proper LM implementation; here we provide a
/// physically transparent iterative refinement.
#[derive(Debug, Clone)]
pub struct FcsFitter {
    /// Initial guess for mean molecule number ‹N›
    pub n_molecules_init: f64,
    /// Initial guess for diffusion coefficient D \[m²/s\]
    pub diffusion_coeff_init: f64,
}

impl FcsFitter {
    /// Construct an FCS fitter with initial parameter guesses.
    pub fn new(n_molecules_init: f64, diffusion_coeff_init: f64) -> Self {
        Self {
            n_molecules_init,
            diffusion_coeff_init,
        }
    }

    /// Fit G(τ) data to the 3D Gaussian FCS model.
    ///
    /// Uses a simple gradient-descent with relative step size adaptation.
    /// Converges in ~50–200 iterations for well-behaved FCS curves.
    ///
    /// # Arguments
    /// * `setup`   — FCS optical geometry
    /// * `tau_data`— Array of lag times \[s\]
    /// * `g_data`  — Corresponding measured G(τ) values
    ///
    /// # Returns
    /// `(N_fit, D_fit, chi2)` where chi2 is the final weighted residual.
    pub fn fit(&self, setup: &FcsSetup, tau_data: &[f64], g_data: &[f64]) -> (f64, f64, f64) {
        if tau_data.is_empty() || g_data.is_empty() || tau_data.len() != g_data.len() {
            return (self.n_molecules_init, self.diffusion_coeff_init, f64::NAN);
        }

        let mut n_fit = self.n_molecules_init.max(f64::EPSILON);
        let mut d_fit = self.diffusion_coeff_init.max(f64::EPSILON);

        let n_iter = 200_usize;
        let step_init = 0.1_f64; // 10% relative step
        let mut step = step_init;

        let chi2 = |n: f64, d: f64| -> f64 {
            tau_data
                .iter()
                .zip(g_data.iter())
                .map(|(&tau, &g_meas)| {
                    let g_model = setup.autocorrelation(tau, n, d);
                    let residual = g_meas - g_model;
                    residual * residual
                })
                .sum()
        };

        let mut best_chi2 = chi2(n_fit, d_fit);

        for _iter in 0..n_iter {
            // Try steps in N and D directions
            let candidates = [
                (n_fit * (1.0 + step), d_fit),
                (n_fit * (1.0 - step), d_fit),
                (n_fit, d_fit * (1.0 + step)),
                (n_fit, d_fit * (1.0 - step)),
                (n_fit * (1.0 + step), d_fit * (1.0 + step)),
                (n_fit * (1.0 - step), d_fit * (1.0 - step)),
            ];

            let mut improved = false;
            for &(n_try, d_try) in &candidates {
                if n_try <= 0.0 || d_try <= 0.0 {
                    continue;
                }
                let c = chi2(n_try, d_try);
                if c < best_chi2 {
                    best_chi2 = c;
                    n_fit = n_try;
                    d_fit = d_try;
                    improved = true;
                }
            }
            if !improved {
                step *= 0.5;
            }
            if step < 1e-10 {
                break;
            }
        }

        (n_fit, d_fit, best_chi2)
    }

    /// Hydrodynamic radius from Stokes–Einstein relation \[m\].
    ///
    /// `R_H = k_B T / (6π η D)`
    ///
    /// # Arguments
    /// * `diffusion_coeff` — Measured D \[m²/s\]
    /// * `viscosity_pa_s`  — Dynamic viscosity η \[Pa·s\] (water at 25°C ≈ 8.9×10⁻⁴)
    /// * `temperature_k`   — Temperature T \[K\]
    pub fn hydrodynamic_radius(
        &self,
        diffusion_coeff: f64,
        viscosity_pa_s: f64,
        temperature_k: f64,
    ) -> f64 {
        KB * temperature_k / (6.0 * PI * viscosity_pa_s * diffusion_coeff.max(f64::EPSILON))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gfp_setup() -> FcsSetup {
        // Typical confocal FCS: ω_xy = 200 nm, ω_z = 1 μm, λ = 488 nm
        FcsSetup::new(200e-9, 1000e-9, 488e-9)
    }

    #[test]
    fn test_focal_volume_femtolitre_range() {
        let setup = gfp_setup();
        let v = setup.focal_volume_m3();
        // V_eff = π^{3/2} × (200e-9)² × 1000e-9 ≈ 0.22 fL
        // 1 fL = 1e-15 L = 1e-18 m³ → range [0.01, 10] fL = [1e-20, 1e-17] m³
        assert!(
            v > 1e-20 && v < 1e-17,
            "Focal volume {} m³ outside 0.01–10 fL",
            v
        );
    }

    #[test]
    fn test_g_zero_equals_one_over_n() {
        let setup = gfp_setup();
        let n = 5.0;
        let g0 = setup.g_zero(n);
        assert!(
            (g0 - 1.0 / n).abs() < 1e-15,
            "G(0) should equal 1/N, got {}",
            g0
        );
    }

    #[test]
    fn test_autocorrelation_decreases_with_lag() {
        let setup = gfp_setup();
        // D_GFP ≈ 87 μm²/s
        let d = 87e-12;
        let n = 3.0;
        let g0 = setup.autocorrelation(0.0, n, d);
        let g_long = setup.autocorrelation(1.0, n, d); // 1 s >> τ_D
        assert!(
            g0 > g_long,
            "G(τ) should decrease from G(0) to 0 at long lags"
        );
    }

    #[test]
    fn test_concentration_from_g0_n_m_range() {
        let setup = gfp_setup();
        // 3 molecules in ~0.22 fL → C ≈ 3 / (0.22e-15 × 6.02e23 × 1000) ≈ 22 nM
        let g0 = setup.g_zero(3.0);
        let c_mol_per_l = setup.concentration_from_g0(g0);
        assert!(
            c_mol_per_l > 1e-9 && c_mol_per_l < 1e-5,
            "Concentration {} mol/L outside nM–μM range",
            c_mol_per_l
        );
    }

    #[test]
    fn test_triplet_correction_raises_g() {
        let setup = gfp_setup();
        let d = 87e-12;
        let n = 3.0;
        let tau = 1e-6_f64; // 1 μs (within triplet lifetime range)
        let g_no_triplet = setup.autocorrelation(tau, n, d);
        let g_triplet = setup.autocorrelation_with_triplet(tau, n, d, 0.2, 5e-6);
        // Triplet correction inflates G at short lags
        assert!(
            g_triplet > g_no_triplet,
            "Triplet correction should increase G(τ) at short lags"
        );
    }

    #[test]
    fn test_fitter_recovers_diffusion_coefficient() {
        let setup = gfp_setup();
        let d_true = 87e-12;
        let n_true = 4.0;
        // Generate synthetic G(τ) data
        let tau_data: Vec<f64> = (0..20)
            .map(|i| 1e-6 * 10.0_f64.powf(i as f64 * 0.2))
            .collect();
        let g_data: Vec<f64> = tau_data
            .iter()
            .map(|&tau| setup.autocorrelation(tau, n_true, d_true))
            .collect();

        let fitter = FcsFitter::new(2.0, 50e-12);
        let (n_fit, d_fit, _chi2) = fitter.fit(&setup, &tau_data, &g_data);
        assert!(
            (n_fit - n_true).abs() / n_true < 0.05,
            "N_fit = {} vs N_true = {}",
            n_fit,
            n_true
        );
        assert!(
            (d_fit - d_true).abs() / d_true < 0.10,
            "D_fit = {} vs D_true = {}",
            d_fit,
            d_true
        );
    }

    #[test]
    fn test_hydrodynamic_radius_gfp() {
        let fitter = FcsFitter::new(1.0, 87e-12);
        // GFP: D ≈ 87 μm²/s, η_water(25°C) = 8.9×10⁻⁴ Pa·s → R_H ≈ 2.8 nm
        let r_h = fitter.hydrodynamic_radius(87e-12, 8.9e-4, 298.0);
        assert!(
            r_h > 1e-9 && r_h < 10e-9,
            "R_H = {} m outside 1–10 nm range for GFP",
            r_h
        );
    }

    #[test]
    fn test_fccs_cross_correlation_zero_without_double() {
        let ch1 = gfp_setup();
        let ch2 = FcsSetup::new(200e-9, 1000e-9, 561e-9);
        let fccs = FccsMeasurement::new(ch1, ch2, 0.9);
        let g_cross = fccs.cross_correlation_amplitude(5.0, 5.0, 0.0);
        assert!(g_cross.abs() < 1e-15, "No double-labelled → G_cross = 0");
    }

    #[test]
    fn test_fccs_bound_fraction_clamped() {
        let ch1 = gfp_setup();
        let ch2 = FcsSetup::new(200e-9, 1000e-9, 561e-9);
        let fccs = FccsMeasurement::new(ch1, ch2, 0.9);
        let g_cross = fccs.cross_correlation_amplitude(5.0, 5.0, 3.0);
        let f_bound = fccs.bound_fraction(g_cross, 10.0);
        assert!(
            (0.0..=1.0).contains(&f_bound),
            "Bound fraction {} out of [0,1]",
            f_bound
        );
    }
}
