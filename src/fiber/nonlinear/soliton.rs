//! Optical soliton propagation in fibers.
//!
//! A fundamental soliton (N=1) balances SPM and anomalous GVD such that
//! its pulse shape is preserved over propagation:
//!
//!   A(z, t) = √P₀ · sech(t/T₀) · exp(iγP₀z/2)
//!
//! Soliton parameters:
//!   N² = γ·P₀·T₀² / |β₂|
//!   L_D = T₀² / |β₂|   (dispersion length)
//!   L_NL = 1/(γ·P₀)    (nonlinear length)
//!   z_0 = π·L_D/2      (soliton period)
//!
//! For N=1: L_D = L_NL, P₀ = |β₂|/(γ·T₀²)
//!
//! Higher-order solitons (N>1) undergo periodic breathing (fission for N>>1).
//!
//! Soliton self-frequency shift (SSFS) due to Raman:
//!   dΩ/dz = -8·T_R·γ²·P₀²/(15·|β₂|)

use std::f64::consts::PI;

/// Optical soliton model for fiber propagation.
#[derive(Debug, Clone, Copy)]
pub struct SolitonFiber {
    /// Nonlinear coefficient γ (W⁻¹m⁻¹)
    pub gamma: f64,
    /// Group velocity dispersion |β₂| (s²/m), should be > 0 (anomalous dispersion)
    pub beta2_abs: f64,
    /// Raman response time T_R (s) for SSFS
    pub t_raman: f64,
}

impl SolitonFiber {
    /// Create soliton fiber model.
    ///
    /// `beta2_abs`: |β₂| in s²/m (positive value, anomalous dispersion implied).
    pub fn new(gamma: f64, beta2_abs: f64) -> Self {
        Self {
            gamma,
            beta2_abs,
            t_raman: 3e-15,
        } // silica T_R ≈ 3 fs
    }

    /// Standard SMF-28 at 1550 nm (anomalous dispersion regime).
    ///
    /// γ = 1.3 W⁻¹km⁻¹, D = 17 ps/(nm·km) → β₂ ≈ -21.7 ps²/km
    pub fn smf28_1550() -> Self {
        Self::new(1.3e-3, 21.7e-27)
    }

    /// Soliton order N for peak power P₀ and pulse duration T₀ (s, half-width at 1/e).
    pub fn soliton_order(&self, p0_w: f64, t0_s: f64) -> f64 {
        (self.gamma * p0_w * t0_s * t0_s / self.beta2_abs).sqrt()
    }

    /// Peak power P₀ (W) for fundamental soliton (N=1) with duration T₀ (s).
    pub fn fundamental_peak_power(&self, t0_s: f64) -> f64 {
        self.beta2_abs / (self.gamma * t0_s * t0_s)
    }

    /// Dispersion length L_D = T₀² / |β₂| (m).
    pub fn dispersion_length(&self, t0_s: f64) -> f64 {
        t0_s * t0_s / self.beta2_abs
    }

    /// Nonlinear length L_NL = 1/(γP₀) (m).
    pub fn nonlinear_length(&self, p0_w: f64) -> f64 {
        1.0 / (self.gamma * p0_w)
    }

    /// Soliton period z₀ = π·L_D/2 (m).
    pub fn soliton_period(&self, t0_s: f64) -> f64 {
        PI * self.dispersion_length(t0_s) / 2.0
    }

    /// Soliton FWHM pulse duration T_FWHM = 2·ln(1+√2)·T₀ ≈ 1.763·T₀.
    pub fn fwhm_from_t0(t0_s: f64) -> f64 {
        2.0 * (1.0 + 2.0_f64.sqrt()).ln() * t0_s
    }

    /// T₀ from FWHM: T₀ = FWHM / (2·ln(1+√2)).
    pub fn t0_from_fwhm(fwhm_s: f64) -> f64 {
        fwhm_s / (2.0 * (1.0 + 2.0_f64.sqrt()).ln())
    }

    /// Soliton self-frequency shift (SSFS) rate dΩ/dz (rad/s per meter).
    ///
    ///   dΩ/dz ≈ -8·T_R·γ²·P₀² / (15·|β₂|)
    pub fn ssfs_rate(&self, p0_w: f64) -> f64 {
        -8.0 * self.t_raman * self.gamma * self.gamma * p0_w * p0_w / (15.0 * self.beta2_abs)
    }

    /// Frequency shift after propagation length L (m): ΔΩ = dΩ/dz · L.
    pub fn ssfs_frequency_shift(&self, p0_w: f64, length_m: f64) -> f64 {
        self.ssfs_rate(p0_w) * length_m
    }

    /// Wavelength shift due to SSFS: Δλ ≈ -λ²/c · ΔΩ/(2π) (m).
    pub fn ssfs_wavelength_shift(&self, p0_w: f64, length_m: f64, wavelength_m: f64) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        let d_omega = self.ssfs_frequency_shift(p0_w, length_m);
        -wavelength_m * wavelength_m * d_omega / (2.0 * PI * SPEED_OF_LIGHT)
    }

    /// Sech² pulse intensity profile: I(t) = P₀ · sech²(t/T₀).
    ///
    /// Returns vector of (time_s, intensity_W) pairs.
    pub fn pulse_profile(&self, p0_w: f64, t0_s: f64, n_pts: usize) -> Vec<(f64, f64)> {
        let t_max = 4.0 * t0_s;
        (0..n_pts)
            .map(|i| {
                let t = -t_max + 2.0 * t_max * i as f64 / (n_pts - 1) as f64;
                let sech = 1.0 / (t / t0_s).cosh();
                (t, p0_w * sech * sech)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soliton_order_one_for_fundamental() {
        let f = SolitonFiber::smf28_1550();
        let t0 = 1e-12; // 1 ps
        let p0 = f.fundamental_peak_power(t0);
        let n = f.soliton_order(p0, t0);
        assert!((n - 1.0).abs() < 1e-6, "N={n:.6}");
    }

    #[test]
    fn soliton_dispersion_length_scales_t0_squared() {
        let f = SolitonFiber::smf28_1550();
        let ld1 = f.dispersion_length(1e-12);
        let ld2 = f.dispersion_length(2e-12);
        assert!((ld2 / ld1 - 4.0).abs() < 1e-6);
    }

    #[test]
    fn soliton_period_positive() {
        let f = SolitonFiber::smf28_1550();
        let z0 = f.soliton_period(1e-12);
        assert!(z0 > 0.0);
    }

    #[test]
    fn soliton_fwhm_t0_roundtrip() {
        let t0 = 1e-12;
        let fwhm = SolitonFiber::fwhm_from_t0(t0);
        let t0_back = SolitonFiber::t0_from_fwhm(fwhm);
        assert!((t0_back - t0).abs() / t0 < 1e-10);
    }

    #[test]
    fn soliton_ssfs_red_shifts() {
        let f = SolitonFiber::smf28_1550();
        let p0 = f.fundamental_peak_power(1e-12);
        let d_lambda = f.ssfs_wavelength_shift(p0, 1e3, 1550e-9);
        assert!(d_lambda > 0.0, "SSFS should red-shift the soliton");
    }

    #[test]
    fn soliton_pulse_profile_peak_at_center() {
        let f = SolitonFiber::smf28_1550();
        let p0 = 0.1;
        let t0 = 1e-12;
        let profile = f.pulse_profile(p0, t0, 101);
        let center_idx = 50;
        let peak = profile[center_idx].1;
        // All other points should be ≤ peak
        for &(_, intensity) in &profile {
            assert!(intensity <= peak + 1e-10);
        }
    }
}
