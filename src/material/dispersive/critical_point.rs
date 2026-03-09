//! Critical Point (CP) model for metal permittivity.
//!
//! Extends the Drude model with critical point oscillators that model
//! interband transitions near band edges with a phase factor φ_p.
//!
//! Reference: Etchegoin et al., J. Chem. Phys. 125, 164705 (2006).
//!
//! ε(ω) = ε_∞ - Ω_p²/(ω² + iΓ_p·ω) + Σ_p A_p·Ω_p·[exp(-iφ_p)/(Ω_p-ω-iΓ_p) + exp(iφ_p)/(Ω_p+ω+iΓ_p)]

use num_complex::Complex64;
use std::f64::consts::PI;

/// Single critical point oscillator parameters.
#[derive(Debug, Clone, Copy)]
pub struct CriticalPoint {
    /// Amplitude A_p
    pub amplitude: f64,
    /// Resonance frequency Ω_p (rad/s)
    pub omega_p: f64,
    /// Broadening Γ_p (rad/s)
    pub gamma_p: f64,
    /// Phase angle φ_p (rad)
    pub phi_p: f64,
}

/// Critical Point model for metal permittivity.
///
/// Combines a Drude free-electron term with critical-point oscillators
/// for improved accuracy near interband transitions.
#[derive(Debug, Clone)]
pub struct CriticalPointModel {
    /// High-frequency permittivity
    pub eps_inf: f64,
    /// Plasma frequency (rad/s)
    pub omega_plasma: f64,
    /// Plasma damping (rad/s)
    pub gamma_plasma: f64,
    /// Critical point oscillators
    pub critical_points: Vec<CriticalPoint>,
}

// Physical constants
const HBAR: f64 = 1.054_571_817e-34;
const Q_E: f64 = 1.602_176_634e-19;

impl CriticalPointModel {
    /// Gold parameters from Etchegoin et al. 2006 (Table I).
    ///
    /// Provides accurate optical constants for Au from 0.2 to 2.0 μm.
    pub fn gold_etchegoin() -> Self {
        let ev = Q_E / HBAR; // eV to rad/s conversion
        Self {
            eps_inf: 1.54,
            omega_plasma: 9.0265 * ev,
            gamma_plasma: 0.069 * ev,
            critical_points: vec![
                CriticalPoint {
                    amplitude: 1.1071,
                    omega_p: 2.3530 * ev,
                    gamma_p: 0.4313 * ev,
                    phi_p: -PI / 4.0,
                },
                CriticalPoint {
                    amplitude: 0.5081,
                    omega_p: 3.0995 * ev,
                    gamma_p: 0.7927 * ev,
                    phi_p: PI / 4.0,
                },
            ],
        }
    }

    /// Silver parameters from Etchegoin et al. 2006 (Table I).
    pub fn silver_etchegoin() -> Self {
        let ev = Q_E / HBAR;
        Self {
            eps_inf: 1.17,
            omega_plasma: 8.4480 * ev,
            gamma_plasma: 0.055 * ev,
            critical_points: vec![
                CriticalPoint {
                    amplitude: 1.7461,
                    omega_p: 4.0789 * ev,
                    gamma_p: 0.9875 * ev,
                    phi_p: -PI / 4.0,
                },
                CriticalPoint {
                    amplitude: 1.1012,
                    omega_p: 5.3126 * ev,
                    gamma_p: 4.4798 * ev,
                    phi_p: PI / 4.0,
                },
            ],
        }
    }

    /// Compute the Drude free-electron contribution.
    fn drude_contribution(&self, omega: f64) -> Complex64 {
        let i = Complex64::new(0.0, 1.0);
        let op2 = self.omega_plasma * self.omega_plasma;
        let denom = omega * omega + i * self.gamma_plasma * omega;
        if denom.norm() < 1e-30 {
            return Complex64::new(0.0, 0.0);
        }
        -op2 / denom
    }

    /// Compute a single critical point contribution.
    ///
    /// χ_p(ω) = A_p · Ω_p · [exp(-iφ_p)/(Ω_p - ω - iΓ_p) + exp(iφ_p)/(Ω_p + ω + iΓ_p)]
    fn cp_contribution(&self, cp: &CriticalPoint, omega: f64) -> Complex64 {
        let i = Complex64::new(0.0, 1.0);
        let omega_p = cp.omega_p;
        let gamma_p = cp.gamma_p;
        let phi = cp.phi_p;
        let amp = cp.amplitude;

        let exp_neg = (-i * phi).exp();
        let exp_pos = (i * phi).exp();

        let denom1 = Complex64::new(omega_p - omega, -gamma_p);
        let denom2 = Complex64::new(omega_p + omega, gamma_p);

        let term1 = if denom1.norm() > 1e-30 {
            exp_neg / denom1
        } else {
            Complex64::new(0.0, 0.0)
        };
        let term2 = if denom2.norm() > 1e-30 {
            exp_pos / denom2
        } else {
            Complex64::new(0.0, 0.0)
        };

        Complex64::new(amp * omega_p, 0.0) * (term1 + term2)
    }

    /// Compute complex permittivity at angular frequency ω (rad/s).
    pub fn permittivity(&self, omega: f64) -> Complex64 {
        let drude = self.drude_contribution(omega);
        let cp_sum: Complex64 = self
            .critical_points
            .iter()
            .map(|cp| self.cp_contribution(cp, omega))
            .sum();
        Complex64::new(self.eps_inf, 0.0) + drude + cp_sum
    }

    /// Complex refractive index (n, k) at angular frequency ω (rad/s).
    ///
    /// Uses the principal square root with n ≥ 0.
    pub fn refractive_index(&self, omega: f64) -> (f64, f64) {
        let eps = self.permittivity(omega);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn omega_from_nm(lambda_nm: f64) -> f64 {
        2.0 * PI * 2.997_924_58e8 / (lambda_nm * 1e-9)
    }

    #[test]
    fn gold_permittivity_at_633nm_negative_real() {
        let au = CriticalPointModel::gold_etchegoin();
        let omega = omega_from_nm(633.0);
        let eps = au.permittivity(omega);
        // Gold is metallic: Re(ε) < 0 at visible
        assert!(eps.re < 0.0, "Re(ε)={:.2}", eps.re);
    }

    #[test]
    fn silver_permittivity_at_400nm_metallic() {
        let ag = CriticalPointModel::silver_etchegoin();
        let omega = omega_from_nm(400.0);
        let eps = ag.permittivity(omega);
        assert!(eps.re < 0.0, "Re(ε_Ag)={:.2}", eps.re);
    }

    #[test]
    fn gold_n_k_non_negative() {
        let au = CriticalPointModel::gold_etchegoin();
        for lambda_nm in [400.0, 600.0, 800.0, 1000.0, 1550.0] {
            let omega = omega_from_nm(lambda_nm);
            let (n, k) = au.refractive_index(omega);
            assert!(n >= 0.0, "n < 0 at {lambda_nm}nm: n={n}");
            assert!(k >= 0.0, "k < 0 at {lambda_nm}nm: k={k}");
        }
    }

    #[test]
    fn silver_n_k_non_negative() {
        let ag = CriticalPointModel::silver_etchegoin();
        for lambda_nm in [400.0, 800.0, 1550.0] {
            let omega = omega_from_nm(lambda_nm);
            let (n, k) = ag.refractive_index(omega);
            assert!(n >= 0.0 && k >= 0.0, "n={n}, k={k} at {lambda_nm}nm");
        }
    }

    #[test]
    fn gold_eps_large_imaginary_at_ir() {
        // At 1550 nm, Au has large Im(ε) (high absorption)
        let au = CriticalPointModel::gold_etchegoin();
        let omega = omega_from_nm(1550.0);
        let eps = au.permittivity(omega);
        // Gold at IR: Im(ε) should be large and positive (lossy metal)
        assert!(eps.im > 1.0, "Im(ε_Au@1550nm)={:.2}", eps.im);
    }

    #[test]
    fn spectrum_returns_correct_length() {
        let au = CriticalPointModel::gold_etchegoin();
        let spec = au.spectrum(0.4, 1.6, 30);
        assert_eq!(spec.len(), 30);
    }

    #[test]
    fn spectrum_wavelengths_increasing() {
        let ag = CriticalPointModel::silver_etchegoin();
        let spec = ag.spectrum(0.3, 1.0, 20);
        for i in 1..spec.len() {
            assert!(spec[i].0 > spec[i - 1].0);
        }
    }

    #[test]
    fn drude_term_zero_omega_diverges_handled() {
        // At omega = 0, Drude term diverges — the function should return a large (finite) value
        // or handle gracefully via the denominator check
        let au = CriticalPointModel::gold_etchegoin();
        let eps = au.permittivity(1e6); // very low frequency
                                        // Should be finite (possibly very large negative real)
        assert!(!eps.re.is_nan() && !eps.im.is_nan());
    }
}
