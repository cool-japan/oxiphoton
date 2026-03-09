use crate::mode::effective_index::{SlabMode, SlabWaveguide};

/// Slab (planar) waveguide device.
///
/// Encapsulates the geometry and provides mode analysis.
#[derive(Debug, Clone)]
pub struct SlabWaveguideDevice {
    pub n_core: f64,
    pub n_clad: f64,
    /// Core thickness (m).
    pub thickness: f64,
}

impl SlabWaveguideDevice {
    pub fn new(n_core: f64, n_clad: f64, thickness: f64) -> Self {
        Self {
            n_core,
            n_clad,
            thickness,
        }
    }

    /// Solve all guided TE modes at wavelength (m).
    pub fn te_modes(&self, wavelength: f64) -> Vec<SlabMode> {
        SlabWaveguide::new(self.n_core, self.n_clad, self.thickness).solve_te(wavelength)
    }

    /// Solve all guided TM modes at wavelength (m).
    pub fn tm_modes(&self, wavelength: f64) -> Vec<SlabMode> {
        SlabWaveguide::new(self.n_core, self.n_clad, self.thickness).solve_tm(wavelength)
    }

    /// Normalized frequency V = k0 · h/2 · sqrt(n_core² - n_clad²).
    pub fn v_number(&self, wavelength: f64) -> f64 {
        use std::f64::consts::PI;
        let k0 = 2.0 * PI / wavelength;
        k0 * self.thickness / 2.0 * (self.n_core * self.n_core - self.n_clad * self.n_clad).sqrt()
    }

    /// Number of guided TE modes (estimated from V).
    pub fn n_te_modes(&self, wavelength: f64) -> usize {
        use std::f64::consts::PI;
        let v = self.v_number(wavelength);
        (v / (PI / 2.0)).ceil() as usize
    }

    /// Confinement factor Γ for TE0 mode at wavelength (m).
    ///
    /// Γ = P_core / P_total (fraction of power in the core).
    pub fn confinement_factor_te0(&self, wavelength: f64) -> Option<f64> {
        let modes = self.te_modes(wavelength);
        let mode = modes.first()?;
        let n_eff = mode.n_eff;

        use std::f64::consts::PI;
        let k0 = 2.0 * PI / wavelength;
        let kappa = (self.n_core * self.n_core * k0 * k0 - n_eff * n_eff * k0 * k0).sqrt();
        let gamma = (n_eff * n_eff * k0 * k0 - self.n_clad * self.n_clad * k0 * k0).sqrt();
        let h = self.thickness;

        // For symmetric TE0 even mode (E ∝ cos(κx) inside core):
        // ∫_{-h/2}^{h/2} cos²(κx) dx = h/2 + sin(κh)/(2κ)
        // Each cladding: ∫_{h/2}^{∞} cos²(κh/2) e^{-2γ(x-h/2)} dx = cos²(κh/2)/(2γ)
        // Total cladding (both sides): cos²(κh/2)/γ
        let p_core = h / 2.0 + (kappa * h / 2.0).sin() * (kappa * h / 2.0).cos() / kappa;
        let cos_kh2 = (kappa * h / 2.0).cos();
        let p_clad = cos_kh2 * cos_kh2 / gamma;
        let gamma_factor = p_core / (p_core + p_clad);
        Some(gamma_factor.clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slab_device_modes() {
        let slab = SlabWaveguideDevice::new(3.476, 1.444, 500e-9);
        let modes = slab.te_modes(1550e-9);
        assert!(!modes.is_empty());
        assert!(modes[0].n_eff > 1.444 && modes[0].n_eff < 3.476);
    }

    #[test]
    fn slab_device_confinement_factor() {
        let slab = SlabWaveguideDevice::new(3.476, 1.444, 500e-9);
        let gamma = slab.confinement_factor_te0(1550e-9).unwrap();
        assert!(gamma > 0.0 && gamma < 1.0, "Gamma={gamma}");
        assert!(
            gamma > 0.5,
            "Si slab should have high confinement, Γ={gamma}"
        );
    }
}
