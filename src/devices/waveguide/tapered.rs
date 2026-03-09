//! Adiabatic taper analysis for photonic waveguides.
//!
//! Models gradual width transitions between two waveguide widths with various
//! taper profiles. Uses coupled-mode theory to estimate insertion loss.

use std::f64::consts::PI;

/// Taper profile shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaperProfile {
    /// Linear width change: w(z) = w1 + (w2-w1)·z/L
    Linear,
    /// Exponential: w(z) = w1 · (w2/w1)^(z/L)
    Exponential,
    /// Gaussian: w(z) = w1 + (w2-w1)·erf(2z/L - 1) normalized
    Gaussian,
    /// Chebyshev polynomial taper of given order (more gradual at endpoints)
    Chebyshev { order: usize },
}

/// Adiabatic taper — gradually changes waveguide width from w1 to w2.
///
/// Uses the adiabatic criterion: |dβ/dz| · (coupling_length) << Δβ.
#[derive(Debug, Clone)]
pub struct AdiabaticTaper {
    /// Input width (m)
    pub width_in: f64,
    /// Output width (m)
    pub width_out: f64,
    /// Taper length (m)
    pub length: f64,
    /// Taper profile shape
    pub taper_profile: TaperProfile,
    /// Core refractive index
    pub n_core: f64,
    /// Cladding refractive index
    pub n_clad: f64,
    /// Operating wavelength (m)
    pub wavelength: f64,
}

impl AdiabaticTaper {
    /// Create a new adiabatic taper.
    pub fn new(
        width_in: f64,
        width_out: f64,
        length: f64,
        profile: TaperProfile,
        n_core: f64,
        n_clad: f64,
        wavelength: f64,
    ) -> Self {
        Self {
            width_in,
            width_out,
            length,
            taper_profile: profile,
            n_core,
            n_clad,
            wavelength,
        }
    }

    /// Width at position z along taper (0 ≤ z ≤ length).
    pub fn width_at(&self, z: f64) -> f64 {
        let z_clamped = z.clamp(0.0, self.length);
        let t = if self.length > 0.0 {
            z_clamped / self.length
        } else {
            0.0
        };
        match self.taper_profile {
            TaperProfile::Linear => self.width_in + (self.width_out - self.width_in) * t,
            TaperProfile::Exponential => {
                if self.width_in > 0.0 && self.width_out > 0.0 {
                    self.width_in * (self.width_out / self.width_in).powf(t)
                } else {
                    self.width_in + (self.width_out - self.width_in) * t
                }
            }
            TaperProfile::Gaussian => {
                // erf-based smooth taper
                let u = 2.0 * t - 1.0; // maps [0,1] to [-1,1]
                let erf_val = erf_approx(3.0 * u); // stretch for steeper transition
                let norm = erf_approx(3.0);
                let frac = (erf_val + norm) / (2.0 * norm);
                self.width_in + (self.width_out - self.width_in) * frac
            }
            TaperProfile::Chebyshev { order } => {
                // Chebyshev polynomial evaluated at t in [0,1]
                // Maps to cos argument and uses T_n profile for minimum ripple
                let x = 2.0 * t - 1.0; // [-1, 1]
                let cheb = chebyshev_t(order, x);
                // Normalize so that T_n(-1) → 0 and T_n(1) → 1
                let t_min = chebyshev_t(order, -1.0);
                let t_max = chebyshev_t(order, 1.0);
                let frac = if (t_max - t_min).abs() > 1e-15 {
                    (cheb - t_min) / (t_max - t_min)
                } else {
                    t
                };
                self.width_in + (self.width_out - self.width_in) * frac.clamp(0.0, 1.0)
            }
        }
    }

    /// Local effective index at position z using slab waveguide approximation.
    ///
    /// Solves for the fundamental TE mode effective index via the transcendental
    /// characteristic equation.
    pub fn local_n_eff(&self, z: f64) -> f64 {
        let w = self.width_at(z);
        let k0 = 2.0 * PI / self.wavelength;
        // Bisection on slab TE characteristic equation for the fundamental mode
        let lo = self.n_clad + 1e-10;
        let hi = self.n_core - 1e-10;
        if lo >= hi {
            return (self.n_core + self.n_clad) / 2.0;
        }
        let char_fn = |n_eff: f64| -> f64 {
            let kappa = (k0 * k0 * (self.n_core * self.n_core - n_eff * n_eff))
                .max(0.0)
                .sqrt();
            let gamma = (k0 * k0 * (n_eff * n_eff - self.n_clad * self.n_clad))
                .max(0.0)
                .sqrt();
            kappa * (kappa * w / 2.0).tan() - gamma
        };
        bisect(char_fn, lo, hi, 80).unwrap_or((lo + hi) / 2.0)
    }

    /// Adiabaticity parameter at position z.
    ///
    /// Defined as |dw/dz| / (w · Δβ / k₀), should be << 1 for adiabatic propagation.
    /// Δβ = β₀ - β₁ ≈ π / Lπ where Lπ is the local beat length.
    pub fn adiabaticity_param(&self, z: f64) -> f64 {
        let dz = self.length * 1e-5;
        let z1 = (z - dz).max(0.0);
        let z2 = (z + dz).min(self.length);
        let dw_dz = (self.width_at(z2) - self.width_at(z1)) / (z2 - z1);

        let w = self.width_at(z);
        let n_eff = self.local_n_eff(z);
        let k0 = 2.0 * PI / self.wavelength;
        let beta0 = k0 * n_eff;

        // Estimate beat length: Lπ ≈ 4 n_eff w² / (3 λ)
        let lpi = 4.0 * n_eff * w * w / (3.0 * self.wavelength);
        let delta_beta = if lpi > 0.0 { PI / lpi } else { beta0 * 0.1 };

        if delta_beta <= 0.0 || w <= 0.0 {
            return 0.0;
        }
        dw_dz.abs() / (w * delta_beta)
    }

    /// Minimum adiabatic taper length to achieve low loss.
    ///
    /// Derived from the requirement that max(adiabaticity_param) < 0.1:
    ///   L_min = 10 · |Δw| / (w_avg · Δβ_avg)
    pub fn min_adiabatic_length(&self) -> f64 {
        let w_avg = (self.width_in + self.width_out) / 2.0;
        let n_eff_avg = self.local_n_eff(self.length / 2.0);
        let lpi = 4.0 * n_eff_avg * w_avg * w_avg / (3.0 * self.wavelength);
        let delta_beta = if lpi > 0.0 { PI / lpi } else { 1.0 };
        let delta_w = (self.width_out - self.width_in).abs();
        if delta_beta <= 0.0 {
            return self.length;
        }
        10.0 * delta_w / (w_avg * delta_beta)
    }

    /// Estimated insertion loss (dB) via coupled-mode theory.
    ///
    /// Integrates the squared local coupling coefficient along the taper:
    ///   IL ≈ 10·log10(1 - ∫|κ(z)|² dz / (Lπ_avg)²)
    pub fn insertion_loss_db(&self) -> f64 {
        let n_seg = 200usize;
        let dz = self.length / n_seg as f64;
        let mut loss_integral = 0.0;

        for i in 0..n_seg {
            let z = (i as f64 + 0.5) * dz;
            let ap = self.adiabaticity_param(z);
            // Loss per unit length ∝ ap²
            loss_integral += ap * ap * dz;
        }

        // Convert coupling integral to dB loss
        let loss_fraction = (loss_integral * 0.5).min(0.999);
        if loss_fraction <= 0.0 {
            return 0.0;
        }
        -10.0 * (1.0 - loss_fraction).log10()
    }

    /// Profile sampled at n_points along z.
    ///
    /// Returns Vec of (z, width) tuples.
    pub fn sample_profile(&self, n_points: usize) -> Vec<(f64, f64)> {
        if n_points == 0 {
            return Vec::new();
        }
        (0..n_points)
            .map(|i| {
                let z = i as f64 / (n_points - 1).max(1) as f64 * self.length;
                (z, self.width_at(z))
            })
            .collect()
    }

    /// Transfer efficiency (0 to 1).
    ///
    /// Estimated from adiabaticity: η = exp(-π · IL_linear).
    pub fn transfer_efficiency(&self) -> f64 {
        let il_db = self.insertion_loss_db();
        let il_linear = 10.0_f64.powf(-il_db / 10.0);
        il_linear.clamp(0.0, 1.0)
    }
}

/// Bisection root finder for a continuous function on [lo, hi].
fn bisect<F: Fn(f64) -> f64>(f: F, mut lo: f64, mut hi: f64, max_iter: usize) -> Option<f64> {
    let f_lo = f(lo);
    let f_hi = f(hi);
    if f_lo * f_hi > 0.0 {
        return None;
    }
    for _ in 0..max_iter {
        let mid = (lo + hi) / 2.0;
        let f_mid = f(mid);
        if f_lo * f_mid < 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
        if (hi - lo) < 1e-14 {
            break;
        }
    }
    Some((lo + hi) / 2.0)
}

/// Polynomial approximation to the error function erf(x) (Abramowitz & Stegun).
fn erf_approx(x: f64) -> f64 {
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0
        - (A1 * t + A2 * t * t + A3 * t.powi(3) + A4 * t.powi(4) + A5 * t.powi(5)) * (-x * x).exp();
    sign * y
}

/// Chebyshev polynomial T_n(x) evaluated at x ∈ [-1, 1].
fn chebyshev_t(n: usize, x: f64) -> f64 {
    match n {
        0 => 1.0,
        1 => x,
        _ => {
            let mut t_prev = 1.0_f64;
            let mut t_curr = x;
            for _ in 2..=n {
                let t_next = 2.0 * x * t_curr - t_prev;
                t_prev = t_curr;
                t_curr = t_next;
            }
            t_curr
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn si_taper() -> AdiabaticTaper {
        AdiabaticTaper::new(
            0.5e-6,
            3.0e-6,
            100e-6,
            TaperProfile::Linear,
            3.476,
            1.444,
            1.55e-6,
        )
    }

    #[test]
    fn taper_width_at_endpoints() {
        let t = si_taper();
        assert_relative_eq!(t.width_at(0.0), t.width_in, epsilon = 1e-15);
        assert_relative_eq!(t.width_at(t.length), t.width_out, epsilon = 1e-15);
    }

    #[test]
    fn taper_width_at_midpoint_linear() {
        let t = si_taper();
        let w_mid = t.width_at(t.length / 2.0);
        let expected = (t.width_in + t.width_out) / 2.0;
        assert_relative_eq!(w_mid, expected, epsilon = 1e-12);
    }

    #[test]
    fn taper_exponential_endpoints() {
        let t = AdiabaticTaper::new(
            0.5e-6,
            3.0e-6,
            100e-6,
            TaperProfile::Exponential,
            3.476,
            1.444,
            1.55e-6,
        );
        assert_relative_eq!(t.width_at(0.0), t.width_in, epsilon = 1e-15);
        assert_relative_eq!(t.width_at(t.length), t.width_out, epsilon = 1e-12);
    }

    #[test]
    fn taper_gaussian_endpoints() {
        let t = AdiabaticTaper::new(
            0.5e-6,
            3.0e-6,
            100e-6,
            TaperProfile::Gaussian,
            3.476,
            1.444,
            1.55e-6,
        );
        assert_relative_eq!(t.width_at(0.0), t.width_in, epsilon = 1e-10);
        assert_relative_eq!(t.width_at(t.length), t.width_out, epsilon = 1e-10);
    }

    #[test]
    fn taper_chebyshev_endpoints() {
        let t = AdiabaticTaper::new(
            0.5e-6,
            3.0e-6,
            100e-6,
            TaperProfile::Chebyshev { order: 3 },
            3.476,
            1.444,
            1.55e-6,
        );
        assert_relative_eq!(t.width_at(0.0), t.width_in, epsilon = 1e-10);
        assert_relative_eq!(t.width_at(t.length), t.width_out, epsilon = 1e-10);
    }

    #[test]
    fn local_n_eff_in_range() {
        let t = si_taper();
        let n_eff = t.local_n_eff(t.length / 2.0);
        assert!(
            n_eff > t.n_clad && n_eff < t.n_core,
            "n_eff = {n_eff} not in ({}, {})",
            t.n_clad,
            t.n_core
        );
    }

    #[test]
    fn adiabaticity_param_finite() {
        let t = si_taper();
        let ap = t.adiabaticity_param(t.length / 2.0);
        assert!(ap.is_finite(), "adiabaticity param = {ap}");
    }

    #[test]
    fn min_adiabatic_length_positive() {
        let t = si_taper();
        let l_min = t.min_adiabatic_length();
        assert!(l_min > 0.0, "min length = {l_min}");
    }

    #[test]
    fn insertion_loss_non_negative() {
        let t = si_taper();
        let il = t.insertion_loss_db();
        assert!(il >= 0.0, "IL = {il}");
    }

    #[test]
    fn sample_profile_correct_length() {
        let t = si_taper();
        let profile = t.sample_profile(50);
        assert_eq!(profile.len(), 50);
        assert_relative_eq!(profile[0].0, 0.0, epsilon = 1e-15);
        assert_relative_eq!(profile[49].0, t.length, epsilon = 1e-12);
    }

    #[test]
    fn transfer_efficiency_between_zero_and_one() {
        let t = si_taper();
        let eta = t.transfer_efficiency();
        assert!((0.0..=1.0).contains(&eta), "eta = {eta}");
    }
}
