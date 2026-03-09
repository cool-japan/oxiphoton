use crate::material::{DispersiveMaterial, DrudeLorentzPole};
use crate::units::{RefractiveIndex, Wavelength};
use num_complex::Complex64;
use std::f64::consts::PI;

use crate::units::conversion::SPEED_OF_LIGHT;

/// Drude-Lorentz model: eps(omega) = eps_inf + sum_j A_j / (omega_j^2 - omega^2 - i*gamma_j*omega)
///
/// Used for metals (Au, Ag, Al) with multiple Lorentz oscillators.
#[derive(Debug, Clone)]
pub struct DrudeLorentz {
    pub name: String,
    pub eps_inf: f64,
    pub poles: Vec<DrudeLorentzPole>,
}

impl DrudeLorentz {
    pub fn new(name: impl Into<String>, eps_inf: f64, poles: Vec<DrudeLorentzPole>) -> Self {
        Self {
            name: name.into(),
            eps_inf,
            poles,
        }
    }

    /// Gold (Au) — Rakic et al. 1998 (Brendel-Bormann model simplified to D-L)
    pub fn au() -> Self {
        // Drude + 2 Lorentz oscillators fitted to experimental data
        Self::new(
            "Au",
            1.0,
            vec![
                // Drude term (omega_j = 0)
                DrudeLorentzPole {
                    amplitude: (1.3202e16_f64).powi(2), // omega_p^2
                    frequency: 0.0,
                    damping: 1.0805e14,
                },
                // Lorentz oscillator 1
                DrudeLorentzPole {
                    amplitude: 0.26698 * (1.3202e16_f64).powi(2),
                    frequency: 3.8711e15,
                    damping: 4.4642e14,
                },
                // Lorentz oscillator 2
                DrudeLorentzPole {
                    amplitude: 3.0834 * (1.3202e16_f64).powi(2),
                    frequency: 4.1684e15,
                    damping: 2.3555e15,
                },
            ],
        )
    }

    /// Silver (Ag) — Rakic et al. 1998
    pub fn ag() -> Self {
        Self::new(
            "Ag",
            1.0,
            vec![
                DrudeLorentzPole {
                    amplitude: (1.3899e16_f64).powi(2),
                    frequency: 0.0,
                    damping: 3.2258e13,
                },
                DrudeLorentzPole {
                    amplitude: 0.7603 * (1.3899e16_f64).powi(2),
                    frequency: 6.3956e15,
                    damping: 1.6693e15,
                },
                DrudeLorentzPole {
                    amplitude: 0.2246 * (1.3899e16_f64).powi(2),
                    frequency: 1.0640e16,
                    damping: 5.7180e15,
                },
            ],
        )
    }

    /// Aluminum (Al) — Rakic et al. 1998
    pub fn al() -> Self {
        Self::new(
            "Al",
            1.0,
            vec![
                DrudeLorentzPole {
                    amplitude: (2.2562e16_f64).powi(2),
                    frequency: 0.0,
                    damping: 1.2541e14,
                },
                DrudeLorentzPole {
                    amplitude: 0.1274 * (2.2562e16_f64).powi(2),
                    frequency: 2.3955e15,
                    damping: 5.1404e14,
                },
                DrudeLorentzPole {
                    amplitude: 0.05 * (2.2562e16_f64).powi(2),
                    frequency: 2.8025e15,
                    damping: 9.5975e14,
                },
            ],
        )
    }

    fn permittivity_at_omega(&self, omega: f64) -> Complex64 {
        let mut eps = Complex64::new(self.eps_inf, 0.0);
        for pole in &self.poles {
            let numer = Complex64::new(pole.amplitude, 0.0);
            let denom = Complex64::new(
                pole.frequency * pole.frequency - omega * omega,
                -pole.damping * omega,
            );
            eps += numer / denom;
        }
        eps
    }
}

impl DispersiveMaterial for DrudeLorentz {
    fn refractive_index(&self, wavelength: Wavelength) -> RefractiveIndex {
        let omega = 2.0 * PI * SPEED_OF_LIGHT / wavelength.0;
        let eps = self.permittivity_at_omega(omega);
        let n_complex = eps.sqrt();
        RefractiveIndex {
            n: n_complex.re.abs(),
            k: n_complex.im.abs(),
        }
    }

    fn permittivity(&self, wavelength: Wavelength) -> Complex64 {
        let omega = 2.0 * PI * SPEED_OF_LIGHT / wavelength.0;
        self.permittivity_at_omega(omega)
    }

    fn fdtd_poles(&self) -> Vec<DrudeLorentzPole> {
        self.poles.clone()
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn au_at_1550nm() {
        let au = DrudeLorentz::au();
        let ri = au.refractive_index(Wavelength::from_nm(1550.0));
        // Au @ 1550nm: n ~ 0.5, k ~ 10.7 (approximate, model-dependent)
        assert!(ri.n < 2.0, "Au n={} should be small at 1550nm", ri.n);
        assert!(ri.k > 5.0, "Au k={} should be large at 1550nm", ri.k);
    }

    #[test]
    fn ag_at_1550nm() {
        let ag = DrudeLorentz::ag();
        let ri = ag.refractive_index(Wavelength::from_nm(1550.0));
        assert!(ri.n < 2.0, "Ag n={} should be small at 1550nm", ri.n);
        assert!(ri.k > 5.0, "Ag k={} should be large at 1550nm", ri.k);
    }

    #[test]
    fn al_at_1550nm() {
        let al = DrudeLorentz::al();
        let ri = al.refractive_index(Wavelength::from_nm(1550.0));
        assert!(ri.k > 5.0, "Al k={} should be large at 1550nm", ri.k);
    }

    #[test]
    fn metal_has_negative_real_permittivity() {
        let au = DrudeLorentz::au();
        let eps = au.permittivity(Wavelength::from_nm(1550.0));
        assert!(
            eps.re < 0.0,
            "Metal should have negative real permittivity, got {}",
            eps.re
        );
    }

    #[test]
    fn au_at_visible() {
        let au = DrudeLorentz::au();
        let ri = au.refractive_index(Wavelength::from_nm(550.0));
        // Gold has interband transitions in visible range
        assert!(ri.n > 0.0);
        assert!(ri.k > 0.0);
    }

    #[test]
    fn drude_lorentz_permittivity_consistency() {
        let au = DrudeLorentz::au();
        let wl = Wavelength::from_nm(800.0);
        let ri = au.refractive_index(wl);
        let eps = au.permittivity(wl);
        let eps_from_ri = ri.to_permittivity_scalar();
        assert_relative_eq!(eps.re, eps_from_ri.re, epsilon = 0.1);
        assert_relative_eq!(eps.im, eps_from_ri.im, epsilon = 0.1);
    }
}
