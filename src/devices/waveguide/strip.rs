use crate::mode::effective_index::{strip_waveguide_eim, Polarization};

/// Rectangular (strip) waveguide device.
///
/// Uses the Effective Index Method (EIM) for mode analysis.
/// For high-accuracy calculations, use `FdModeSolver2d` directly.
#[derive(Debug, Clone)]
pub struct StripWaveguide {
    pub n_core: f64,
    pub n_clad: f64,
    /// Waveguide width (m).
    pub width: f64,
    /// Waveguide height (m).
    pub height: f64,
}

impl StripWaveguide {
    pub fn new(n_core: f64, n_clad: f64, width: f64, height: f64) -> Self {
        Self {
            n_core,
            n_clad,
            width,
            height,
        }
    }

    /// Effective index of the quasi-TE fundamental mode using EIM.
    pub fn n_eff_te(&self, wavelength: f64) -> Option<f64> {
        strip_waveguide_eim(
            self.n_core,
            self.n_clad,
            self.width,
            self.height,
            wavelength,
            Polarization::TE,
        )
    }

    /// Effective index of the quasi-TM fundamental mode using EIM.
    pub fn n_eff_tm(&self, wavelength: f64) -> Option<f64> {
        strip_waveguide_eim(
            self.n_core,
            self.n_clad,
            self.width,
            self.height,
            wavelength,
            Polarization::TM,
        )
    }

    /// Group index from finite-difference approximation.
    ///
    /// n_g = n_eff - λ · dn_eff/dλ
    /// Computed numerically with wavelength step δλ.
    pub fn group_index_te(&self, wavelength: f64, delta_lambda: f64) -> Option<f64> {
        let n1 = self.n_eff_te(wavelength - delta_lambda)?;
        let n2 = self.n_eff_te(wavelength + delta_lambda)?;
        let dn_dlambda = (n2 - n1) / (2.0 * delta_lambda);
        Some(n1 - wavelength * dn_dlambda + delta_lambda * dn_dlambda)
        // More precisely: n_g = n_eff(λ) - λ * dn_eff/dλ
        // Using central n_eff:
    }

    /// Group index (proper formula).
    pub fn group_index(&self, wavelength: f64) -> Option<f64> {
        let dl = wavelength * 1e-4; // small wavelength step
        let n_lo = self.n_eff_te(wavelength - dl)?;
        let n_hi = self.n_eff_te(wavelength + dl)?;
        let n_eff = self.n_eff_te(wavelength)?;
        let dn_dlambda = (n_hi - n_lo) / (2.0 * dl);
        Some(n_eff - wavelength * dn_dlambda)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_waveguide_si_guided() {
        let wg = StripWaveguide::new(3.476, 1.444, 500e-9, 220e-9);
        let n_eff = wg.n_eff_te(1550e-9).expect("Should find TE mode");
        assert!(
            n_eff > 1.444 && n_eff < 3.476,
            "n_eff={n_eff:.4} out of range"
        );
    }

    #[test]
    fn strip_waveguide_group_index() {
        let wg = StripWaveguide::new(3.476, 1.444, 500e-9, 220e-9);
        let ng = wg.group_index(1550e-9).expect("Group index computation");
        // Group index for Si strip is typically 3.5–5.0
        assert!(ng > 2.0 && ng < 6.0, "Group index ng={ng:.4}");
    }
}
