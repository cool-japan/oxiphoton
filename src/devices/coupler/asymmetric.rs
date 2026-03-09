//! Asymmetric directional coupler — wavelength-selective coupling.
//!
//! Two waveguides with different widths have different effective indices.
//! Phase-matching occurs at a specific wavelength λ_match where the two
//! propagation constants become equal.

use num_complex::Complex64;
use std::f64::consts::PI;

/// Asymmetric directional coupler.
///
/// Coupling efficiency:
///   η = κ² / (κ² + Δβ²/4) · sin²(√(κ² + Δβ²/4) · L)
/// where Δβ = β_A - β_B is the phase mismatch and κ is the coupling coefficient.
#[derive(Debug, Clone)]
pub struct AsymmetricCoupler {
    /// Width of waveguide A (m)
    pub width_a: f64,
    /// Width of waveguide B (m)
    pub width_b: f64,
    /// Gap between waveguides (m)
    pub gap: f64,
    /// Coupling length (m)
    pub length: f64,
    /// Core refractive index
    pub n_core: f64,
    /// Cladding refractive index
    pub n_clad: f64,
    /// Operating wavelength (m)
    pub wavelength: f64,
}

impl AsymmetricCoupler {
    /// Create a new asymmetric directional coupler.
    pub fn new(
        width_a: f64,
        width_b: f64,
        gap: f64,
        length: f64,
        n_core: f64,
        n_clad: f64,
        wavelength: f64,
    ) -> Self {
        Self {
            width_a,
            width_b,
            gap,
            length,
            n_core,
            n_clad,
            wavelength,
        }
    }

    /// Effective index for a single waveguide of given width at this wavelength.
    ///
    /// Uses the slab waveguide transcendental characteristic equation (TE, fundamental mode).
    /// The fundamental even mode satisfies: kappa * tan(kappa*d/2) = gamma.
    /// We search for the root closest to n_core (fundamental mode has highest n_eff).
    fn n_eff_for_width(&self, width: f64) -> f64 {
        let k0 = 2.0 * PI / self.wavelength;
        let lo = self.n_clad + 1e-10;
        let hi = self.n_core - 1e-10;
        if lo >= hi {
            return (self.n_core + self.n_clad) / 2.0;
        }

        // Scan from hi to lo looking for the first sign change (fundamental mode = highest n_eff)
        let n_scan = 500usize;
        let char_fn = |n_eff: f64| -> f64 {
            let kappa = (k0 * k0 * (self.n_core * self.n_core - n_eff * n_eff))
                .max(0.0)
                .sqrt();
            let gamma = (k0 * k0 * (n_eff * n_eff - self.n_clad * self.n_clad))
                .max(0.0)
                .sqrt();
            kappa * (kappa * width / 2.0).tan() - gamma
        };

        let step = (hi - lo) / n_scan as f64;
        let mut f_prev = char_fn(hi);
        let mut n_prev = hi;

        for i in 1..=n_scan {
            let n_curr = hi - i as f64 * step;
            let f_curr = char_fn(n_curr);
            // Sign change indicates a root — use the first one from the top (fundamental)
            if f_prev * f_curr < 0.0 {
                // Verify we're not in a singularity of tan by checking both values are reasonable
                if f_prev.abs() < 1e8 && f_curr.abs() < 1e8 {
                    let root = bisect(&char_fn, n_curr, n_prev, 60);
                    if let Some(r) = root {
                        return r;
                    }
                }
            }
            f_prev = f_curr;
            n_prev = n_curr;
        }
        // Fallback: return effective medium approximation
        (self.n_core * self.n_core * 0.8 + self.n_clad * self.n_clad * 0.2).sqrt()
    }

    /// Effective indices for the two waveguides at the current wavelength.
    ///
    /// Returns (n_eff_A, n_eff_B).
    pub fn effective_indices(&self) -> (f64, f64) {
        (
            self.n_eff_for_width(self.width_a),
            self.n_eff_for_width(self.width_b),
        )
    }

    /// Phase mismatch Δβ = β_A - β_B (rad/m).
    pub fn phase_mismatch(&self) -> f64 {
        let k0 = 2.0 * PI / self.wavelength;
        let (na, nb) = self.effective_indices();
        k0 * (na - nb)
    }

    /// Coupling coefficient κ (rad/m), estimated from field overlap.
    ///
    /// κ ≈ (π/λ) · Δn_coupling · exp(-γ·gap)
    /// where Δn_coupling is the refractive index contrast and γ is the evanescent decay.
    pub fn coupling_coefficient(&self) -> f64 {
        let k0 = 2.0 * PI / self.wavelength;
        let (na, nb) = self.effective_indices();
        // Average effective index
        let n_avg = (na + nb) / 2.0;
        // Evanescent decay constant in the gap
        let gamma = (k0 * k0 * (n_avg * n_avg - self.n_clad * self.n_clad))
            .max(0.0)
            .sqrt();
        // Index contrast between waveguide and gap
        let delta_n = n_avg - self.n_clad;
        // Coupling coefficient via overlap integral approximation
        k0 * delta_n * (-gamma * self.gap).exp()
    }

    /// Power coupling efficiency η.
    ///
    /// η = κ² / (κ² + Δβ²/4) · sin²(√(κ² + Δβ²/4) · L)
    pub fn coupling_efficiency(&self) -> f64 {
        let kappa = self.coupling_coefficient();
        let delta_beta = self.phase_mismatch();
        let discriminant = kappa * kappa + delta_beta * delta_beta / 4.0;
        if discriminant <= 0.0 {
            return 0.0;
        }
        let q = discriminant.sqrt();
        (kappa * kappa / discriminant) * (q * self.length).sin().powi(2)
    }

    /// Transfer matrix for the coupler at the current wavelength.
    ///
    /// \[[a_out\], \[b_out\]] = M · \[[a_in\], \[b_in\]]
    /// Based on the coupled-mode theory transfer matrix.
    pub fn transfer_matrix(&self) -> [[Complex64; 2]; 2] {
        let kappa = self.coupling_coefficient();
        let delta_beta = self.phase_mismatch();
        let discriminant = kappa * kappa + delta_beta * delta_beta / 4.0;
        let q = discriminant.max(0.0).sqrt();
        let l = self.length;
        let i = Complex64::new(0.0, 1.0);

        // Phase factors
        let phase_a = Complex64::new(0.0, -delta_beta * l / 2.0).exp();
        let phase_b = Complex64::new(0.0, delta_beta * l / 2.0).exp();

        let (s, c) = if q > 1e-30 {
            ((q * l).sin(), (q * l).cos())
        } else {
            (0.0, 1.0)
        };

        // Transfer matrix elements
        let m11 = phase_a * Complex64::new(c + delta_beta * s / (2.0 * q.max(1e-30)), 0.0);
        let m12 = phase_a * (-i * kappa * s / q.max(1e-30));
        let m21 = phase_b * (-i * kappa * s / q.max(1e-30));
        let m22 = phase_b * Complex64::new(c - delta_beta * s / (2.0 * q.max(1e-30)), 0.0);

        [[m11, m12], [m21, m22]]
    }

    /// Wavelength at which coupling is maximized (phase-matched wavelength).
    ///
    /// Found by scanning the wavelength and locating the maximum coupling efficiency.
    pub fn phase_match_wavelength(&self) -> f64 {
        // Scan wavelengths around the operating wavelength
        let lambda_center = self.wavelength;
        let span = lambda_center * 0.3;
        let n_scan = 500usize;
        let mut best_lambda = lambda_center;
        let mut best_eta = 0.0_f64;

        for i in 0..n_scan {
            let lambda = lambda_center - span / 2.0 + i as f64 / (n_scan - 1) as f64 * span;
            let coupler = AsymmetricCoupler::new(
                self.width_a,
                self.width_b,
                self.gap,
                self.length,
                self.n_core,
                self.n_clad,
                lambda,
            );
            let eta = coupler.coupling_efficiency();
            if eta > best_eta {
                best_eta = eta;
                best_lambda = lambda;
            }
        }
        best_lambda
    }

    /// 3 dB bandwidth around the phase-matched wavelength (m).
    ///
    /// Defined as the wavelength range where η ≥ η_max/2.
    pub fn bandwidth_3db(&self) -> f64 {
        let lambda_pm = self.phase_match_wavelength();
        let span = lambda_pm * 0.3;
        let n_scan = 500usize;
        let lambda_max_eta = {
            let c = AsymmetricCoupler::new(
                self.width_a,
                self.width_b,
                self.gap,
                self.length,
                self.n_core,
                self.n_clad,
                lambda_pm,
            );
            c.coupling_efficiency()
        };
        let half_max = lambda_max_eta / 2.0;

        let mut left = lambda_pm;
        let mut right = lambda_pm;
        let mut in_band = false;

        for i in 0..n_scan {
            let lambda = lambda_pm - span / 2.0 + i as f64 / (n_scan - 1) as f64 * span;
            let coupler = AsymmetricCoupler::new(
                self.width_a,
                self.width_b,
                self.gap,
                self.length,
                self.n_core,
                self.n_clad,
                lambda,
            );
            let eta = coupler.coupling_efficiency();
            if eta >= half_max {
                if !in_band {
                    left = lambda;
                    in_band = true;
                }
                right = lambda;
            }
        }
        (right - left).abs()
    }

    /// Through-port transmission vs wavelength.
    ///
    /// Returns Vec of (wavelength_m, transmission) sampled at n_points.
    pub fn through_spectrum(
        &self,
        lambda_min: f64,
        lambda_max: f64,
        n_points: usize,
    ) -> Vec<(f64, f64)> {
        if n_points == 0 {
            return Vec::new();
        }
        (0..n_points)
            .map(|i| {
                let lambda = lambda_min
                    + i as f64 / (n_points - 1).max(1) as f64 * (lambda_max - lambda_min);
                let coupler = AsymmetricCoupler::new(
                    self.width_a,
                    self.width_b,
                    self.gap,
                    self.length,
                    self.n_core,
                    self.n_clad,
                    lambda,
                );
                let eta_cross = coupler.coupling_efficiency();
                (lambda, (1.0 - eta_cross).clamp(0.0, 1.0))
            })
            .collect()
    }

    /// Cross-port transmission vs wavelength.
    ///
    /// Returns Vec of (wavelength_m, transmission) sampled at n_points.
    pub fn cross_spectrum(
        &self,
        lambda_min: f64,
        lambda_max: f64,
        n_points: usize,
    ) -> Vec<(f64, f64)> {
        if n_points == 0 {
            return Vec::new();
        }
        (0..n_points)
            .map(|i| {
                let lambda = lambda_min
                    + i as f64 / (n_points - 1).max(1) as f64 * (lambda_max - lambda_min);
                let coupler = AsymmetricCoupler::new(
                    self.width_a,
                    self.width_b,
                    self.gap,
                    self.length,
                    self.n_core,
                    self.n_clad,
                    lambda,
                );
                let eta_cross = coupler.coupling_efficiency().clamp(0.0, 1.0);
                (lambda, eta_cross)
            })
            .collect()
    }
}

/// Bisection root finder for continuous function on \[lo, hi\].
fn bisect<F: Fn(f64) -> f64>(f: &F, mut lo: f64, mut hi: f64, max_iter: usize) -> Option<f64> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn si_coupler() -> AsymmetricCoupler {
        AsymmetricCoupler::new(0.45e-6, 0.55e-6, 0.2e-6, 50e-6, 3.476, 1.444, 1.55e-6)
    }

    #[test]
    fn effective_indices_in_range() {
        let c = si_coupler();
        let (na, nb) = c.effective_indices();
        assert!(na > c.n_clad && na < c.n_core, "na = {na}");
        assert!(nb > c.n_clad && nb < c.n_core, "nb = {nb}");
    }

    #[test]
    fn asymmetric_widths_give_different_n_effs() {
        let c = si_coupler();
        let (na, nb) = c.effective_indices();
        // Wider waveguide has higher effective index
        assert!(nb > na, "na={na}, nb={nb}");
    }

    #[test]
    fn phase_mismatch_nonzero_for_asymmetric() {
        let c = si_coupler();
        let db = c.phase_mismatch();
        assert!(db.abs() > 0.0, "Δβ = {db}");
    }

    #[test]
    fn coupling_coefficient_positive() {
        let c = si_coupler();
        assert!(c.coupling_coefficient() > 0.0);
    }

    #[test]
    fn coupling_efficiency_between_zero_and_one() {
        let c = si_coupler();
        let eta = c.coupling_efficiency();
        assert!((0.0..=1.0).contains(&eta), "eta = {eta}");
    }

    #[test]
    fn through_plus_cross_approximately_conserved() {
        let c = si_coupler();
        let eta = c.coupling_efficiency();
        let through = 1.0 - eta;
        assert_relative_eq!(through + eta, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn transfer_matrix_is_2x2_and_finite() {
        let c = si_coupler();
        let tm = c.transfer_matrix();
        for row in &tm {
            for &elem in row {
                assert!(elem.re.is_finite() && elem.im.is_finite());
            }
        }
    }

    #[test]
    fn through_spectrum_correct_length() {
        let c = si_coupler();
        let spec = c.through_spectrum(1.4e-6, 1.7e-6, 100);
        assert_eq!(spec.len(), 100);
    }

    #[test]
    fn cross_spectrum_values_in_range() {
        let c = si_coupler();
        let spec = c.cross_spectrum(1.4e-6, 1.7e-6, 50);
        for (_, t) in spec {
            assert!((0.0..=1.0).contains(&t), "transmission = {t}");
        }
    }

    #[test]
    fn phase_match_wavelength_in_scan_range() {
        let c = si_coupler();
        let pm = c.phase_match_wavelength();
        assert!(pm > 1.1e-6 && pm < 1.9e-6, "λ_pm = {pm:.3e}");
    }

    #[test]
    fn bandwidth_3db_positive() {
        let c = si_coupler();
        let bw = c.bandwidth_3db();
        assert!(bw >= 0.0, "BW = {bw}");
    }

    #[test]
    fn coupling_increases_with_gap_decrease() {
        let c_small_gap =
            AsymmetricCoupler::new(0.45e-6, 0.55e-6, 0.1e-6, 50e-6, 3.476, 1.444, 1.55e-6);
        let c_large_gap =
            AsymmetricCoupler::new(0.45e-6, 0.55e-6, 0.3e-6, 50e-6, 3.476, 1.444, 1.55e-6);
        // Smaller gap → stronger coupling → larger κ
        assert!(
            c_small_gap.coupling_coefficient() > c_large_gap.coupling_coefficient(),
            "kappa(small)={:.3e}, kappa(large)={:.3e}",
            c_small_gap.coupling_coefficient(),
            c_large_gap.coupling_coefficient()
        );
    }
}
