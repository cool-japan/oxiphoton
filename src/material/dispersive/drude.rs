use crate::material::{DispersiveMaterial, DrudeLorentzPole};
use crate::units::{RefractiveIndex, Wavelength};
use num_complex::Complex64;
use std::f64::consts::PI;

use crate::units::conversion::SPEED_OF_LIGHT;

/// Drude model for metals: eps(omega) = eps_inf - omega_p^2 / (omega^2 + i*gamma*omega)
#[derive(Debug, Clone)]
pub struct Drude {
    pub name: String,
    /// High-frequency permittivity
    pub eps_inf: f64,
    /// Plasma frequency (rad/s)
    pub omega_p: f64,
    /// Damping rate (rad/s)
    pub gamma: f64,
}

impl Drude {
    pub fn new(name: impl Into<String>, eps_inf: f64, omega_p: f64, gamma: f64) -> Self {
        Self {
            name: name.into(),
            eps_inf,
            omega_p,
            gamma,
        }
    }

    fn permittivity_at_omega(&self, omega: f64) -> Complex64 {
        let eps_inf = Complex64::new(self.eps_inf, 0.0);
        let wp2 = self.omega_p * self.omega_p;
        let denom = Complex64::new(omega * omega, self.gamma * omega);
        eps_inf - wp2 / denom
    }
}

impl DispersiveMaterial for Drude {
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
        vec![DrudeLorentzPole {
            amplitude: self.omega_p * self.omega_p,
            frequency: 0.0, // Drude: resonance at 0
            damping: self.gamma,
        }]
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drude_metal_has_complex_index() {
        let metal = Drude::new("TestMetal", 1.0, 1.37e16, 4.08e13);
        let ri = metal.refractive_index(Wavelength::from_nm(800.0));
        assert!(
            ri.k > 0.0,
            "Metal should have nonzero extinction coefficient"
        );
    }
}
