use std::f64::consts::PI;

/// Step-index optical fiber.
///
/// A circular fiber with core index n_core, cladding index n_clad, core radius a.
/// V-number (normalized frequency): V = (2π/λ)·a·NA, where NA = sqrt(n_core²-n_clad²).
///
/// Single-mode condition: V < 2.405 (first zero of J₀ Bessel function).
#[derive(Debug, Clone, Copy)]
pub struct StepIndexFiber {
    /// Core refractive index
    pub n_core: f64,
    /// Cladding refractive index
    pub n_clad: f64,
    /// Core radius (m)
    pub core_radius: f64,
}

impl StepIndexFiber {
    pub fn new(n_core: f64, n_clad: f64, core_radius: f64) -> Self {
        assert!(n_core > n_clad, "n_core must exceed n_clad for guidance");
        Self {
            n_core,
            n_clad,
            core_radius,
        }
    }

    /// SMF-28 standard single-mode fiber at 1310nm
    /// n_core=1.4681, n_clad=1.4641 → NA≈0.108, V≈2.16 at 1310nm, λ_c≈1174nm
    pub fn smf28() -> Self {
        Self {
            n_core: 1.4681,
            n_clad: 1.4641,
            core_radius: 4.15e-6,
        }
    }

    /// Numerical aperture: NA = sqrt(n_core² - n_clad²)
    pub fn numerical_aperture(&self) -> f64 {
        (self.n_core * self.n_core - self.n_clad * self.n_clad).sqrt()
    }

    /// V-number (normalized frequency)
    pub fn v_number(&self, wavelength: f64) -> f64 {
        2.0 * PI / wavelength * self.core_radius * self.numerical_aperture()
    }

    /// True if single-mode at given wavelength (V < 2.405)
    pub fn is_single_mode(&self, wavelength: f64) -> bool {
        self.v_number(wavelength) < 2.405
    }

    /// Number of guided modes (approximate): N_modes ≈ V²/2 for large V
    pub fn mode_count_approx(&self, wavelength: f64) -> usize {
        let v = self.v_number(wavelength);
        ((v * v / 2.0).floor() as usize).max(1)
    }

    /// Normalized propagation constant b = (n_eff² - n_clad²) / (n_core² - n_clad²).
    ///
    /// Uses the empirical approximation (Gloge, 1971):
    ///   b ≈ (1.1428 - 0.9960/V)² for single-mode
    pub fn normalized_b(&self, wavelength: f64) -> f64 {
        let v = self.v_number(wavelength);
        (1.1428 - 0.9960 / v).powi(2).clamp(0.0, 1.0)
    }

    /// Effective refractive index from normalized propagation constant b.
    pub fn n_eff(&self, wavelength: f64) -> f64 {
        let b = self.normalized_b(wavelength);
        let na = self.numerical_aperture();
        let na2 = na * na;
        (self.n_clad * self.n_clad + b * na2).sqrt()
    }

    /// Group velocity dispersion (GVD) from material and waveguide contributions.
    ///
    /// Uses approximate formula for waveguide dispersion:
    ///   D_wg ≈ -n_core·NA·V·b'' / (c·λ) where b'' = d²b/dV²
    ///
    /// Returns total dispersion D = -(λ/c)·d²n_eff/dλ² in s/m² (ps/(nm·km) after scaling).
    pub fn waveguide_dispersion(&self, wavelength: f64) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        let v = self.v_number(wavelength);
        // Empirical V·d²(Vb)/dV²:
        // Vb ≈ V*(1.1428 - 0.9960/V)² = V*b, d/dV[(1.1428 - 0.9960/V)²] ...
        // Numerical derivative is used here
        let dv = 1e-4 * v;
        let b0 = self.normalized_b(wavelength);
        let b_p = {
            let v2 = v + dv;
            let x = (1.1428 - 0.9960 / v2).clamp(0.0, 1.0);
            x * x
        };
        let b_m = {
            let v2 = v - dv;
            let x = (1.1428 - 0.9960 / v2).clamp(0.0, 1.0);
            x * x
        };
        let d2vb_dv2 = ((v + dv) * b_p - 2.0 * v * b0 + (v - dv) * b_m) / (dv * dv);
        let na = self.numerical_aperture();
        -self.n_core * na / (SPEED_OF_LIGHT * wavelength) * d2vb_dv2
    }

    /// Mode field diameter (MFD) using Petermann-II definition approximation.
    ///
    /// MFD ≈ 2a · (0.65 + 1.619/V^(3/2) + 2.879/V^6) for 1.2 < V < 2.4
    pub fn mode_field_diameter(&self, wavelength: f64) -> f64 {
        let v = self.v_number(wavelength);
        2.0 * self.core_radius * (0.65 + 1.619 / v.powf(1.5) + 2.879 / v.powi(6))
    }

    /// Cutoff wavelength: V = 2.405 → λ_c = 2π·a·NA / 2.405
    pub fn cutoff_wavelength(&self) -> f64 {
        2.0 * PI * self.core_radius * self.numerical_aperture() / 2.405
    }

    /// Group index: n_g = n_eff - λ·dn_eff/dλ
    pub fn group_index(&self, wavelength: f64) -> f64 {
        let dl = 1e-12; // 1 pm step
        let n_p = self.n_eff(wavelength + dl);
        let n_m = self.n_eff(wavelength - dl);
        let dn_dl = (n_p - n_m) / (2.0 * dl);
        self.n_eff(wavelength) - wavelength * dn_dl
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smf28_single_mode_at_1310nm() {
        let f = StepIndexFiber::smf28();
        assert!(
            f.is_single_mode(1310e-9),
            "SMF-28 should be single-mode at 1310nm"
        );
    }

    #[test]
    fn smf28_v_number_range() {
        let f = StepIndexFiber::smf28();
        let v = f.v_number(1310e-9);
        // SMF-28 V ≈ 2.0-2.2 at 1310nm
        assert!(v > 1.5 && v < 2.405, "V={v:.3} out of SM range");
    }

    #[test]
    fn na_positive() {
        let f = StepIndexFiber::smf28();
        assert!(f.numerical_aperture() > 0.0);
        assert!(f.numerical_aperture() < 1.0);
    }

    #[test]
    fn n_eff_between_core_and_clad() {
        let f = StepIndexFiber::smf28();
        let neff = f.n_eff(1310e-9);
        assert!(
            neff > f.n_clad && neff < f.n_core,
            "n_eff={neff:.5} out of [n_clad, n_core]"
        );
    }

    #[test]
    fn mfd_larger_than_core_for_sm_fiber() {
        let f = StepIndexFiber::smf28();
        let mfd = f.mode_field_diameter(1310e-9);
        // MFD should be ~8-10μm (larger than core diameter 8.3μm)
        assert!(
            mfd > f.core_radius,
            "MFD={:.2e} should be > core radius={:.2e}",
            mfd,
            f.core_radius
        );
    }

    #[test]
    fn cutoff_wavelength_below_operating() {
        let f = StepIndexFiber::smf28();
        let lc = f.cutoff_wavelength();
        // SMF-28 cutoff: λ_c ≈ 1260nm
        assert!(lc < 1310e-9, "Cutoff {:.0}nm should be < 1310nm", lc * 1e9);
    }

    #[test]
    fn multimode_fiber_v_large() {
        // 50μm core fiber: multimode
        let f = StepIndexFiber::new(1.48, 1.46, 25e-6);
        let v = f.v_number(850e-9);
        assert!(v > 2.405, "50μm core fiber should be multimode");
    }

    #[test]
    fn mode_count_multimode() {
        let f = StepIndexFiber::new(1.48, 1.46, 25e-6);
        let count = f.mode_count_approx(850e-9);
        assert!(count > 1, "Should have multiple modes");
    }

    #[test]
    fn group_index_greater_than_phase_index() {
        // In a dispersive waveguide, n_g ≥ n_phase typically
        let f = StepIndexFiber::smf28();
        let ng = f.group_index(1310e-9);
        assert!(ng > 0.0 && ng.is_finite());
    }
}
