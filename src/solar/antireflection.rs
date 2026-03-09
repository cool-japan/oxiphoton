//! Anti-reflection coating (ARC) design for solar cells.
//!
//! A single-layer ARC minimizes reflection when:
//!   1. Optical thickness condition: n_ARC · d = λ₀/4 (quarter-wave)
//!   2. Index matching: n_ARC = √(n_air · n_substrate)
//!
//! For silicon solar cells (n_Si ≈ 3.5 at λ = 600 nm):
//!   n_ARC = √(1.0 × 3.5) ≈ 1.87  → SiN_x (n ≈ 1.9–2.0) is standard
//!
//! Double-layer ARC (DLARC) further reduces reflection:
//!   n₁ = n_air^(1/3) · n_sub^(2/3)  (modified condition)
//!   n₂ = n_air^(2/3) · n_sub^(1/3)
//!
//! Weighted reflectance (solar-weighted average):
//!   R_w = ∫ R(λ) · I_AM15(λ) dλ / ∫ I_AM15(λ) dλ

/// Single-layer anti-reflection coating model.
#[derive(Debug, Clone, Copy)]
pub struct SingleLayerArc {
    /// ARC refractive index n_ARC
    pub n_arc: f64,
    /// ARC physical thickness d (m)
    pub thickness: f64,
    /// Substrate refractive index
    pub n_substrate: f64,
    /// Incident medium index (usually 1.0 for air)
    pub n_incident: f64,
}

impl SingleLayerArc {
    /// Create a single-layer ARC.
    pub fn new(n_arc: f64, thickness: f64, n_substrate: f64) -> Self {
        Self {
            n_arc,
            thickness,
            n_substrate,
            n_incident: 1.0,
        }
    }

    /// Standard SiNₓ ARC on silicon solar cell (optimized for 600 nm).
    ///
    /// n_SiNx ≈ 2.0, d = 75 nm (λ/4 at 600 nm).
    pub fn sinx_on_silicon() -> Self {
        Self::new(2.0, 75e-9, 3.5)
    }

    /// MgF₂ ARC on glass (n_MgF2 ≈ 1.38).
    pub fn mgf2_on_glass() -> Self {
        Self::new(1.38, 96e-9, 1.52) // 96 nm for λ/4 at 530 nm
    }

    /// Optimal ARC index (geometric mean): n_opt = √(n_inc · n_sub).
    pub fn optimal_index(&self) -> f64 {
        (self.n_incident * self.n_substrate).sqrt()
    }

    /// Optimal ARC thickness for quarter-wave at wavelength λ (m).
    pub fn optimal_thickness(&self, wavelength: f64) -> f64 {
        wavelength / (4.0 * self.n_arc)
    }

    /// Reflectance R(λ) using TMM for normal incidence.
    ///
    /// Exact formula for single-layer at normal incidence:
    ///   r = (r01 + r12·exp(2iδ)) / (1 + r01·r12·exp(2iδ))
    ///   where r01 = (n0-n1)/(n0+n1), r12 = (n1-n2)/(n1+n2)
    ///   δ = 2π·n1·d/λ
    pub fn reflectance(&self, wavelength: f64) -> f64 {
        use std::f64::consts::PI;
        let n0 = self.n_incident;
        let n1 = self.n_arc;
        let n2 = self.n_substrate;
        let r01 = (n0 - n1) / (n0 + n1);
        let r12 = (n1 - n2) / (n1 + n2);
        let delta = 2.0 * PI * n1 * self.thickness / wavelength;
        let cos2d = (2.0 * delta).cos();
        let sin2d = (2.0 * delta).sin();

        // R = |r|² where r = (r01 + r12*exp(2iδ)) / (1 + r01*r12*exp(2iδ))
        let num_re = r01 + r12 * cos2d;
        let num_im = r12 * sin2d;
        let den_re = 1.0 + r01 * r12 * cos2d;
        let den_im = r01 * r12 * sin2d;
        let num_sq = num_re * num_re + num_im * num_im;
        let den_sq = den_re * den_re + den_im * den_im;
        if den_sq < 1e-30 {
            return 1.0;
        }
        num_sq / den_sq
    }

    /// Reflectance spectrum as Vec<(λ_nm, R)>.
    pub fn reflectance_spectrum(
        &self,
        lambda_min_nm: f64,
        lambda_max_nm: f64,
        n_pts: usize,
    ) -> Vec<(f64, f64)> {
        (0..n_pts)
            .map(|i| {
                let lambda_nm =
                    lambda_min_nm + (lambda_max_nm - lambda_min_nm) * i as f64 / (n_pts - 1) as f64;
                let r = self.reflectance(lambda_nm * 1e-9);
                (lambda_nm, r)
            })
            .collect()
    }

    /// Solar-weighted reflectance (approximate).
    ///
    /// Uses simple weighting over 400–1100 nm range with flat solar spectrum approximation.
    pub fn solar_weighted_reflectance(&self) -> f64 {
        let spectrum = self.reflectance_spectrum(400.0, 1100.0, 50);
        spectrum.iter().map(|(_, r)| r).sum::<f64>() / spectrum.len() as f64
    }

    /// Bare surface reflectance (without ARC).
    pub fn bare_reflectance(&self) -> f64 {
        let r = (self.n_incident - self.n_substrate) / (self.n_incident + self.n_substrate);
        r * r
    }

    /// ARC efficiency: relative reduction in solar-weighted reflectance vs bare surface.
    pub fn arc_efficiency(&self) -> f64 {
        let r_bare = self.bare_reflectance();
        let r_arc = self.solar_weighted_reflectance();
        if r_bare < 1e-10 {
            return 0.0;
        }
        (r_bare - r_arc) / r_bare
    }
}

/// Double-layer anti-reflection coating model.
#[derive(Debug, Clone, Copy)]
pub struct DoubleLayerArc {
    /// First layer (top, adjacent to air)
    pub layer1: SingleLayerArc,
    /// Second layer (bottom, adjacent to substrate)
    pub n2: f64,
    pub d2: f64,
}

impl DoubleLayerArc {
    /// MgF₂ / ZnS double-layer ARC on Si (standard solar cell DLARC).
    ///
    /// n₁ = 1.38 (MgF₂), n₂ = 2.35 (ZnS).
    pub fn mgf2_zns_on_si() -> Self {
        let l1 = SingleLayerArc::new(1.38, 103e-9, 2.35); // d₁ = λ/4 at 570 nm
        Self {
            layer1: l1,
            n2: 2.35,
            d2: 60e-9,
        } // d₂ = λ/4 at 560 nm
    }

    /// Reflectance at wavelength λ (m): simplified formula.
    pub fn reflectance(&self, wavelength: f64) -> f64 {
        // Approximate: propagate through layer 1 using effective substrate = layer 2 + Si
        let mut l1 = self.layer1;
        l1.n_substrate = self.n2;
        let r1_entry = l1.reflectance(wavelength);
        // Layer 2 as ARC on Si
        let l2 = SingleLayerArc::new(self.n2, self.d2, l1.n_substrate);
        let r2 = l2.reflectance(wavelength);
        // Approximate total: R ≈ r1_entry * (1-r2)² + r2 * (1-r1_entry)²  (rough)
        let t1_sq = (1.0 - r1_entry).powi(2);
        let t2_sq = (1.0 - r2).powi(2);
        r1_entry + r2 * t1_sq * t2_sq // very rough approximation
    }

    /// Solar-weighted reflectance.
    pub fn solar_weighted_reflectance(&self) -> f64 {
        let n = 50;
        let spec: Vec<f64> = (0..n)
            .map(|i| {
                let lam = (400.0 + 700.0 * i as f64 / (n - 1) as f64) * 1e-9;
                self.reflectance(lam)
            })
            .collect();
        spec.iter().sum::<f64>() / n as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arc_reflectance_minimum_near_target() {
        let arc = SingleLayerArc::sinx_on_silicon();
        // Minimum should be near 600 nm
        let r_600 = arc.reflectance(600e-9);
        let r_400 = arc.reflectance(400e-9);
        let r_1000 = arc.reflectance(1000e-9);
        // R at 600nm should be less than at extremes
        assert!(
            r_600 < r_400 || r_600 < r_1000,
            "R(600nm)={r_600:.3}, R(400nm)={r_400:.3}, R(1000nm)={r_1000:.3}"
        );
    }

    #[test]
    fn arc_reflectance_less_than_bare() {
        let arc = SingleLayerArc::sinx_on_silicon();
        let r_arc = arc.solar_weighted_reflectance();
        let r_bare = arc.bare_reflectance();
        assert!(
            r_arc < r_bare,
            "ARC R={r_arc:.3} should be < bare R={r_bare:.3}"
        );
    }

    #[test]
    fn arc_efficiency_positive() {
        let arc = SingleLayerArc::sinx_on_silicon();
        let eff = arc.arc_efficiency();
        assert!(eff > 0.0 && eff <= 1.0, "ARC efficiency={eff:.3}");
    }

    #[test]
    fn arc_optimal_index() {
        let arc = SingleLayerArc::sinx_on_silicon();
        let n_opt = arc.optimal_index();
        // n_opt = sqrt(1.0 * 3.5) ≈ 1.87
        assert!((n_opt - 1.87).abs() < 0.1, "n_opt={n_opt:.2}");
    }

    #[test]
    fn arc_spectrum_length() {
        let arc = SingleLayerArc::mgf2_on_glass();
        let spec = arc.reflectance_spectrum(400.0, 700.0, 30);
        assert_eq!(spec.len(), 30);
    }

    #[test]
    fn dlarc_solar_reflectance_low() {
        let dlarc = DoubleLayerArc::mgf2_zns_on_si();
        let r = dlarc.solar_weighted_reflectance();
        assert!((0.0..0.5).contains(&r), "R={r:.3}");
    }
}
