/// Anisotropic material model — dielectric tensor ε for birefringent crystals.
///
/// Optically anisotropic media are characterised by a permittivity tensor:
///   ε = diag(εx, εy, εz)    (for principal-axis-aligned crystals)
///
/// Classification:
///   - Isotropic: εx = εy = εz  (cubic symmetry)
///   - Uniaxial:  εx = εy ≠ εz  (tetragonal, hexagonal, trigonal)
///   - Biaxial:   εx ≠ εy ≠ εz  (orthorhombic, monoclinic, triclinic)
///
/// For uniaxial crystals the two indices are:
///   no = ordinary ray (perpendicular to optic axis)
///   ne = extraordinary ray (parallel to optic axis)
use num_complex::Complex64;

/// Permittivity tensor for an optically anisotropic material.
///
/// Stored as principal-axis diagonal components (εx, εy, εz).
#[derive(Debug, Clone, Copy)]
pub struct DielectricTensor {
    pub eps_x: Complex64,
    pub eps_y: Complex64,
    pub eps_z: Complex64,
}

impl DielectricTensor {
    pub fn new(eps_x: Complex64, eps_y: Complex64, eps_z: Complex64) -> Self {
        Self {
            eps_x,
            eps_y,
            eps_z,
        }
    }

    /// Isotropic (all components equal).
    pub fn isotropic(eps: Complex64) -> Self {
        Self {
            eps_x: eps,
            eps_y: eps,
            eps_z: eps,
        }
    }

    /// Uniaxial: ordinary eps_o (x,y) and extraordinary eps_e (z).
    pub fn uniaxial(eps_o: Complex64, eps_e: Complex64) -> Self {
        Self {
            eps_x: eps_o,
            eps_y: eps_o,
            eps_z: eps_e,
        }
    }

    /// Refractive index nx = sqrt(εx), ny, nz (real part, for lossless materials).
    pub fn refractive_indices(&self) -> (f64, f64, f64) {
        (
            self.eps_x.re.sqrt(),
            self.eps_y.re.sqrt(),
            self.eps_z.re.sqrt(),
        )
    }
}

/// Anisotropic (birefringent) material.
#[derive(Debug, Clone, Copy)]
pub struct AnisotropicMaterial {
    /// Ordinary refractive index (x, y)
    pub n_o: f64,
    /// Extraordinary refractive index (z = optic axis)
    pub n_e: f64,
    /// Birefringence Δn = ne - no
    pub delta_n: f64,
}

impl AnisotropicMaterial {
    pub fn new(n_o: f64, n_e: f64) -> Self {
        Self {
            n_o,
            n_e,
            delta_n: n_e - n_o,
        }
    }

    /// Lithium niobate (LiNbO₃) — negative uniaxial, at 1550nm.
    pub fn lithium_niobate() -> Self {
        Self::new(2.138, 2.211) // no < ne → positive uniaxial
    }

    /// Calcite (CaCO₃) — strong negative uniaxial, at 589nm.
    pub fn calcite() -> Self {
        Self::new(1.6584, 1.4864) // no > ne → negative uniaxial
    }

    /// Beta-barium borate (β-BBO) — negative uniaxial, at 1064nm.
    pub fn bbo() -> Self {
        Self::new(1.6551, 1.5425)
    }

    /// KTP (KTiOPO₄) — biaxial, at 1064nm (nx, ny approximated as n_o, nz as n_e).
    pub fn ktp() -> Self {
        Self::new(1.7377, 1.8297)
    }

    /// Quartz (SiO₂ crystalline) — positive uniaxial, at 589nm.
    pub fn quartz() -> Self {
        Self::new(1.5442, 1.5533)
    }

    /// Dielectric tensor (wavelength-independent approximation).
    pub fn tensor(&self) -> DielectricTensor {
        DielectricTensor::uniaxial(
            Complex64::new(self.n_o * self.n_o, 0.0),
            Complex64::new(self.n_e * self.n_e, 0.0),
        )
    }

    /// Walk-off angle ρ (rad) for extraordinary ray at internal angle θ (rad) from optic axis.
    ///
    ///   tan(ρ) = -(ne² - no²)·sin(θ)·cos(θ) / (ne²·sin²θ + no²·cos²θ)
    pub fn walkoff_angle(&self, theta: f64) -> f64 {
        let no2 = self.n_o * self.n_o;
        let ne2 = self.n_e * self.n_e;
        let num = -(ne2 - no2) * theta.sin() * theta.cos();
        let den = ne2 * theta.sin() * theta.sin() + no2 * theta.cos() * theta.cos();
        (num / den).atan()
    }

    /// Extraordinary refractive index at angle θ from optic axis.
    ///
    ///   1/ne(θ)² = cos²θ/no² + sin²θ/ne²
    pub fn n_extraordinary(&self, theta: f64) -> f64 {
        let no = self.n_o;
        let ne = self.n_e;
        let cos2 = theta.cos() * theta.cos();
        let sin2 = theta.sin() * theta.sin();
        1.0 / (cos2 / (no * no) + sin2 / (ne * ne)).sqrt()
    }

    /// Phase-matching angle for Type-I SHG: ne(2ω, θ) = no(ω).
    ///
    /// Returns θ_pm (rad) from optic axis, or None if not achievable.
    pub fn phase_match_angle_shg_type1(
        &self,
        n_o_fundamental: f64,
        n_e_shg: &AnisotropicMaterial,
    ) -> Option<f64> {
        // Need ne(2ω, θ) = no(ω)
        // 1/ne(2ω,θ)² = cos²θ/n_o(2ω)² + sin²θ/n_e(2ω)²
        // Solving: sin²θ = (1/n_target² - 1/n_o²) / (1/n_e² - 1/n_o²)
        let n_target = n_o_fundamental;
        let n_o2 = n_e_shg.n_o;
        let n_e2 = n_e_shg.n_e;
        let inv_nt2 = 1.0 / (n_target * n_target);
        let inv_no2 = 1.0 / (n_o2 * n_o2);
        let inv_ne2 = 1.0 / (n_e2 * n_e2);
        let sin2_theta = (inv_nt2 - inv_no2) / (inv_ne2 - inv_no2);
        if !(0.0..=1.0).contains(&sin2_theta) {
            None
        } else {
            Some(sin2_theta.sqrt().asin())
        }
    }

    /// True if positive uniaxial (ne > no).
    pub fn is_positive_uniaxial(&self) -> bool {
        self.n_e > self.n_o
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lnbo3_positive_uniaxial() {
        let m = AnisotropicMaterial::lithium_niobate();
        assert!(m.is_positive_uniaxial());
        assert!(m.delta_n > 0.0);
    }

    #[test]
    fn calcite_negative_uniaxial() {
        let m = AnisotropicMaterial::calcite();
        assert!(!m.is_positive_uniaxial());
        assert!(m.delta_n < 0.0);
    }

    #[test]
    fn extraordinary_index_at_zero_angle_is_ordinary() {
        let m = AnisotropicMaterial::calcite();
        let ne_at_0 = m.n_extraordinary(0.0);
        assert!((ne_at_0 - m.n_o).abs() < 1e-8);
    }

    #[test]
    fn extraordinary_index_at_90deg_is_ne() {
        let m = AnisotropicMaterial::calcite();
        let ne_at_90 = m.n_extraordinary(std::f64::consts::FRAC_PI_2);
        assert!((ne_at_90 - m.n_e).abs() < 1e-8);
    }

    #[test]
    fn walkoff_zero_on_axis() {
        let m = AnisotropicMaterial::bbo();
        let rho = m.walkoff_angle(0.0);
        assert!(rho.abs() < 1e-12);
    }

    #[test]
    fn tensor_uniaxial_xy_equal() {
        let m = AnisotropicMaterial::calcite();
        let t = m.tensor();
        assert!((t.eps_x - t.eps_y).norm() < 1e-12);
        assert!((t.eps_x - t.eps_z).norm() > 0.01);
    }

    #[test]
    fn phase_match_angle_bbo_shg() {
        // BBO type-I SHG at 1064nm → 532nm
        let bbo_fund = AnisotropicMaterial::new(1.6551, 1.5425); // at 1064nm
        let bbo_shg = AnisotropicMaterial::new(1.6749, 1.5555); // at 532nm
        let angle = bbo_fund.phase_match_angle_shg_type1(bbo_fund.n_o, &bbo_shg);
        assert!(angle.is_some());
        let theta = angle.unwrap();
        // Typical BBO type-I SHG phase match ≈ 22.8° for 1064→532nm
        assert!(theta > 0.1 && theta < 1.0, "theta={theta:.3} rad");
    }
}
