//! Brendel-Bormann (BB) model for metal permittivity.
//!
//! Combines a Drude free-electron term with Gaussian-broadened Lorentz oscillators.
//! Provides better accuracy than the simple Drude-Lorentz model for noble metals,
//! particularly near interband transition edges.
//!
//! ε(ω) = ε_∞ - ωp²/(ω(ω+iΓ₀)) + Σ_j χ_j(ω)
//!
//! where χ_j is the Gaussian-broadened Lorentz oscillator integral
//! (see Rakic et al., Applied Optics 37, 5271, 1998).

use num_complex::Complex64;
use std::f64::consts::PI;

/// Single oscillator parameters for the Brendel-Bormann model.
#[derive(Debug, Clone, Copy)]
pub struct BbOscillator {
    /// Oscillator amplitude (rad/s)
    pub sigma: f64,
    /// Center frequency ω_j (rad/s)
    pub omega_j: f64,
    /// Lorentzian width Γ_j (rad/s)
    pub gamma_j: f64,
}

/// Brendel-Bormann (BB) model for metal permittivity.
///
/// ε(ω) = ε_∞ - ωp²/(ω(ω+iΓ₀)) + Σ_j χ_BB_j(ω)
///
/// The oscillator integral is evaluated by Gaussian quadrature (N=20 points).
#[derive(Debug, Clone)]
pub struct BrendelBormannModel {
    /// High-frequency permittivity
    pub eps_inf: f64,
    /// Plasma frequency (rad/s)
    pub omega_p: f64,
    /// Free-electron damping (rad/s)
    pub gamma_0: f64,
    /// Oscillator parameters
    pub oscillators: Vec<BbOscillator>,
}

impl BrendelBormannModel {
    /// Gold parameters (Rakic et al. 1998, Table 1).
    ///
    /// ε_∞ = 1.0, ω_p = 1.37×10¹⁶ rad/s, Γ₀ = 4.06×10¹³ rad/s
    pub fn gold() -> Self {
        let c = 2.997_924_58e8;
        let ev_to_rad = Q_E / HBAR; // converts eV to rad/s
                                    // Parameters in eV from Rakic et al. 1998:
                                    // f0=0.770, Gamma0=0.05 eV, omega_p=9.03 eV
                                    // Oscillators: (f, Gamma, omega, sigma)
        let _ = c;
        let wp = 9.03 * ev_to_rad;
        let g0 = 0.053 * ev_to_rad;

        Self {
            eps_inf: 1.0,
            omega_p: wp,
            gamma_0: g0,
            oscillators: vec![
                BbOscillator {
                    sigma: 0.94 * ev_to_rad,
                    omega_j: 0.415 * ev_to_rad,
                    gamma_j: 0.241 * ev_to_rad,
                },
                BbOscillator {
                    sigma: 1.36 * ev_to_rad,
                    omega_j: 0.830 * ev_to_rad,
                    gamma_j: 0.345 * ev_to_rad,
                },
                BbOscillator {
                    sigma: 0.54 * ev_to_rad,
                    omega_j: 2.969 * ev_to_rad,
                    gamma_j: 0.870 * ev_to_rad,
                },
                BbOscillator {
                    sigma: 0.77 * ev_to_rad,
                    omega_j: 4.304 * ev_to_rad,
                    gamma_j: 2.494 * ev_to_rad,
                },
                BbOscillator {
                    sigma: 0.36 * ev_to_rad,
                    omega_j: 13.32 * ev_to_rad,
                    gamma_j: 2.214 * ev_to_rad,
                },
            ],
        }
    }

    /// Silver parameters (Rakic et al. 1998).
    pub fn silver() -> Self {
        let ev_to_rad = Q_E / HBAR;
        let wp = 9.01 * ev_to_rad;
        let g0 = 0.018 * ev_to_rad;

        Self {
            eps_inf: 1.0,
            omega_p: wp,
            gamma_0: g0,
            oscillators: vec![
                BbOscillator {
                    sigma: 0.845 * ev_to_rad,
                    omega_j: 0.816 * ev_to_rad,
                    gamma_j: 0.452 * ev_to_rad,
                },
                BbOscillator {
                    sigma: 0.065 * ev_to_rad,
                    omega_j: 4.481 * ev_to_rad,
                    gamma_j: 0.065 * ev_to_rad,
                },
                BbOscillator {
                    sigma: 0.124 * ev_to_rad,
                    omega_j: 8.185 * ev_to_rad,
                    gamma_j: 0.916 * ev_to_rad,
                },
                BbOscillator {
                    sigma: 0.011 * ev_to_rad,
                    omega_j: 9.083 * ev_to_rad,
                    gamma_j: 0.290 * ev_to_rad,
                },
                BbOscillator {
                    sigma: 0.840 * ev_to_rad,
                    omega_j: 20.29 * ev_to_rad,
                    gamma_j: 2.418 * ev_to_rad,
                },
            ],
        }
    }

    /// Aluminum parameters (Rakic et al. 1998).
    pub fn aluminum() -> Self {
        let ev_to_rad = Q_E / HBAR;
        let wp = 14.98 * ev_to_rad;
        let g0 = 0.047 * ev_to_rad;

        Self {
            eps_inf: 1.0,
            omega_p: wp,
            gamma_0: g0,
            oscillators: vec![
                BbOscillator {
                    sigma: 2.62 * ev_to_rad,
                    omega_j: 1.644 * ev_to_rad,
                    gamma_j: 2.565 * ev_to_rad,
                },
                BbOscillator {
                    sigma: 2.76 * ev_to_rad,
                    omega_j: 0.124 * ev_to_rad,
                    gamma_j: 0.179 * ev_to_rad,
                },
                BbOscillator {
                    sigma: 2.45 * ev_to_rad,
                    omega_j: 1.646 * ev_to_rad,
                    gamma_j: 1.178 * ev_to_rad,
                },
            ],
        }
    }

    /// Copper parameters (Rakic et al. 1998).
    pub fn copper() -> Self {
        let ev_to_rad = Q_E / HBAR;
        let wp = 10.83 * ev_to_rad;
        let g0 = 0.030 * ev_to_rad;

        Self {
            eps_inf: 1.0,
            omega_p: wp,
            gamma_0: g0,
            oscillators: vec![
                BbOscillator {
                    sigma: 4.0 * ev_to_rad,
                    omega_j: 1.03 * ev_to_rad,
                    gamma_j: 0.42 * ev_to_rad,
                },
                BbOscillator {
                    sigma: 2.0 * ev_to_rad,
                    omega_j: 2.12 * ev_to_rad,
                    gamma_j: 0.50 * ev_to_rad,
                },
                BbOscillator {
                    sigma: 1.5 * ev_to_rad,
                    omega_j: 4.50 * ev_to_rad,
                    gamma_j: 1.24 * ev_to_rad,
                },
                BbOscillator {
                    sigma: 0.8 * ev_to_rad,
                    omega_j: 8.97 * ev_to_rad,
                    gamma_j: 3.50 * ev_to_rad,
                },
            ],
        }
    }

    /// Compute the Drude contribution to permittivity.
    fn drude_term(&self, omega: f64) -> Complex64 {
        let i = Complex64::new(0.0, 1.0);
        let wp_sq = self.omega_p * self.omega_p;
        let denom = omega * (omega + i * self.gamma_0);
        if denom.norm() < 1e-30 {
            return Complex64::new(0.0, 0.0);
        }
        -wp_sq / denom
    }

    /// Evaluate a single BB oscillator contribution via Gaussian quadrature.
    ///
    /// χ_j(ω) = σ_j/√(2π) · ∫ exp(-u²/2)/(ω_j - ω - iΓ_j + √2·σ_j·u) du
    ///
    /// The integral is evaluated using Gauss-Hermite quadrature (20 points).
    fn oscillator_term(&self, osc: &BbOscillator, omega: f64) -> Complex64 {
        let i = Complex64::new(0.0, 1.0);
        let sigma = osc.sigma;
        let omega_j = osc.omega_j;
        let gamma_j = osc.gamma_j;

        // Gauss-Hermite quadrature nodes and weights (20 points)
        // Using physics convention: ∫ f(x) exp(-x²) dx ≈ Σ w_i f(x_i)
        // Transform: u = √2 · x → exp(-u²/2) du = √2·exp(-x²) dx
        let (nodes, weights) = gauss_hermite_20();

        let prefactor = sigma / (2.0 * PI).sqrt();
        let sum: Complex64 = nodes
            .iter()
            .zip(weights.iter())
            .map(|(&x, &w)| {
                let u = 2.0_f64.sqrt() * x;
                let denom = omega_j - omega - i * gamma_j + 2.0_f64.sqrt() * sigma * u;
                if denom.norm() < 1e-30 {
                    Complex64::new(0.0, 0.0)
                } else {
                    Complex64::new(w * 2.0_f64.sqrt(), 0.0) / denom
                }
            })
            .sum();

        prefactor * sum
    }

    /// Compute complex permittivity at angular frequency ω (rad/s).
    pub fn permittivity(&self, omega: f64) -> Complex64 {
        let eps = Complex64::new(self.eps_inf, 0.0)
            + self.drude_term(omega)
            + self
                .oscillators
                .iter()
                .map(|osc| self.oscillator_term(osc, omega))
                .sum::<Complex64>();
        eps
    }

    /// Refractive index n + ik at angular frequency ω (rad/s).
    pub fn refractive_index(&self, omega: f64) -> (f64, f64) {
        let eps = self.permittivity(omega);
        // n + ik = sqrt(eps) — principal branch with n ≥ 0
        let z = eps.sqrt();
        let (n, k) = if z.re >= 0.0 {
            (z.re, z.im)
        } else {
            (-z.re, -z.im)
        };
        (n, k.abs())
    }

    /// Permittivity spectrum over wavelength range [lambda_min, lambda_max] (μm).
    ///
    /// Returns Vec of (wavelength_um, permittivity).
    pub fn spectrum(
        &self,
        lambda_min_um: f64,
        lambda_max_um: f64,
        n_points: usize,
    ) -> Vec<(f64, Complex64)> {
        if n_points == 0 {
            return Vec::new();
        }
        (0..n_points)
            .map(|i| {
                let lambda_um = lambda_min_um
                    + i as f64 / (n_points - 1).max(1) as f64 * (lambda_max_um - lambda_min_um);
                let lambda_m = lambda_um * 1e-6;
                let omega = 2.0 * PI * 2.997_924_58e8 / lambda_m;
                (lambda_um, self.permittivity(omega))
            })
            .collect()
    }
}

// Reduced Planck constant (J·s)
const HBAR: f64 = 1.054_571_817e-34;
// Elementary charge (C)
const Q_E: f64 = 1.602_176_634e-19;

/// Gauss-Hermite quadrature nodes and weights for 20-point integration.
///
/// Computes ∫ f(x) exp(-x²) dx ≈ Σ w_i f(x_i)
fn gauss_hermite_20() -> ([f64; 20], [f64; 20]) {
    // Nodes (positive only; negative nodes are symmetric)
    let nodes: [f64; 20] = [
        -5.387_480_890_011_233,
        -4.603_682_449_550_744,
        -3.944_764_040_402_426,
        -3.347_854_567_383_216,
        -2.788_806_058_428_706,
        -2.254_974_002_089_678,
        -1.738_537_712_934_97,
        -1.234_076_215_395_323,
        -0.737_473_728_510_801,
        -0.245_340_708_300_901,
        0.245_340_708_300_901,
        0.737_473_728_510_801,
        1.234_076_215_395_323,
        1.738_537_712_934_97,
        2.254_974_002_089_678,
        2.788_806_058_428_706,
        3.347_854_567_383_216,
        3.944_764_040_402_426,
        4.603_682_449_550_744,
        5.387_480_890_011_233,
    ];
    let weights: [f64; 20] = [
        2.229_060_746_473_142e-13,
        4.399_340_992_273_24e-10,
        1.086_069_370_769_133e-7,
        7.802_556_478_532_034e-6,
        2.283_386_360_163_538e-4,
        3.243_773_342_238_137e-3,
        2.481_052_088_685_932e-2,
        1.090_172_060_200_233e-1,
        2.866_755_053_628_034e-1,
        4.622_436_696_006_101e-1,
        4.622_436_696_006_101e-1,
        2.866_755_053_628_034e-1,
        1.090_172_060_200_233e-1,
        2.481_052_088_685_932e-2,
        3.243_773_342_238_137e-3,
        2.283_386_360_163_538e-4,
        7.802_556_478_532_034e-6,
        1.086_069_370_769_133e-7,
        4.399_340_992_273_24e-10,
        2.229_060_746_473_142e-13,
    ];
    (nodes, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn omega_from_lambda_nm(lambda_nm: f64) -> f64 {
        2.0 * PI * 2.997_924_58e8 / (lambda_nm * 1e-9)
    }

    #[test]
    fn gold_permittivity_at_visible_has_negative_real() {
        let au = BrendelBormannModel::gold();
        let omega = omega_from_lambda_nm(600.0);
        let eps = au.permittivity(omega);
        // Gold at visible: Re(ε) < 0 (metallic behavior)
        assert!(eps.re < 0.0, "Re(eps_Au)={:.2} at 600nm", eps.re);
    }

    #[test]
    fn silver_permittivity_at_nir_has_negative_real() {
        let ag = BrendelBormannModel::silver();
        let omega = omega_from_lambda_nm(800.0);
        let eps = ag.permittivity(omega);
        assert!(eps.re < 0.0, "Re(eps_Ag)={:.2} at 800nm", eps.re);
    }

    #[test]
    fn aluminum_permittivity_uv_metallic() {
        let al = BrendelBormannModel::aluminum();
        let omega = omega_from_lambda_nm(400.0);
        let eps = al.permittivity(omega);
        assert!(eps.re < 0.0, "Re(eps_Al)={:.2} at 400nm", eps.re);
    }

    #[test]
    fn copper_permittivity_finite() {
        let cu = BrendelBormannModel::copper();
        let omega = omega_from_lambda_nm(700.0);
        let eps = cu.permittivity(omega);
        assert!(eps.re.is_finite() && eps.im.is_finite());
    }

    #[test]
    fn gold_n_positive() {
        let au = BrendelBormannModel::gold();
        let omega = omega_from_lambda_nm(1550.0);
        let (n, k) = au.refractive_index(omega);
        assert!(n >= 0.0, "n = {n}");
        assert!(k >= 0.0, "k = {k}");
    }

    #[test]
    fn silver_extinction_coefficient_nonzero() {
        let ag = BrendelBormannModel::silver();
        let omega = omega_from_lambda_nm(500.0);
        let (_, k) = ag.refractive_index(omega);
        assert!(k > 0.0, "k_Ag = {k}");
    }

    #[test]
    fn spectrum_correct_length() {
        let au = BrendelBormannModel::gold();
        let spec = au.spectrum(0.4, 1.0, 20);
        assert_eq!(spec.len(), 20);
    }

    #[test]
    fn spectrum_wavelengths_monotone() {
        let au = BrendelBormannModel::gold();
        let spec = au.spectrum(0.4, 1.6, 50);
        for i in 1..spec.len() {
            assert!(spec[i].0 > spec[i - 1].0);
        }
    }

    #[test]
    fn gauss_hermite_weights_sum_to_sqrt_pi() {
        let (_, w) = gauss_hermite_20();
        let sum: f64 = w.iter().sum();
        // For Gauss-Hermite: Σ w_i = sqrt(π)
        assert_relative_eq!(sum, PI.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn gold_eps_inf_positive() {
        let au = BrendelBormannModel::gold();
        assert!(au.eps_inf > 0.0);
    }
}
