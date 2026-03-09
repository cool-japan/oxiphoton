//! Ridge (rib) waveguide analysis.
//!
//! A ridge waveguide has a raised ridge on a slab:
//!
//! ```text
//!        |<-- w -->|
//!   _____|         |_____   ← upper cladding n_clad
//!   |       ridge       |
//!   |  n_core  h_ridge  |
//!   |___________________|   ← slab (n_core, h_slab thick)
//!   |    substrate          ← lower cladding n_sub
//! ```
//!
//! Uses the Effective Index Method (EIM):
//!   1. Solve 2 vertical slab problems: inner (ridge) and outer (slab only).
//!   2. Use effective indices as horizontal waveguide indices.
//!
//! Reference: Tamir, "Integrated Optics" (1982), Chapter 3.

use crate::mode::effective_index::{AsymmetricSlab, Polarization, SlabWaveguide};

/// Ridge (rib) waveguide geometry and mode analysis.
#[derive(Debug, Clone)]
pub struct RidgeWaveguide {
    /// Core refractive index (ridge and slab material).
    pub n_core: f64,
    /// Substrate / lower cladding refractive index.
    pub n_sub: f64,
    /// Upper cladding refractive index.
    pub n_clad: f64,
    /// Ridge width (m).
    pub width: f64,
    /// Ridge height above slab (m).
    pub h_ridge: f64,
    /// Slab thickness below ridge (m).
    pub h_slab: f64,
}

impl RidgeWaveguide {
    /// Create a ridge waveguide.
    ///
    /// # Arguments
    /// - `n_core`: core (ridge + slab) refractive index
    /// - `n_sub`: substrate refractive index
    /// - `n_clad`: upper cladding index
    /// - `width`: ridge width (m)
    /// - `h_ridge`: extra ridge height above slab (m)
    /// - `h_slab`: slab thickness (m)
    pub fn new(
        n_core: f64,
        n_sub: f64,
        n_clad: f64,
        width: f64,
        h_ridge: f64,
        h_slab: f64,
    ) -> Self {
        Self {
            n_core,
            n_sub,
            n_clad,
            width,
            h_ridge,
            h_slab,
        }
    }

    /// Standard SOI ridge waveguide: Si core on SiO2.
    ///
    /// Ridge: 500nm wide, 220nm total (50nm slab + 170nm ridge).
    pub fn soi_standard() -> Self {
        Self::new(3.476, 1.444, 1.0, 500e-9, 170e-9, 50e-9)
    }

    fn solve_vertical(&self, h_total: f64, wavelength: f64, pol: Polarization) -> Option<f64> {
        // Asymmetric: substrate | core | cladding
        let n_max_clad = self.n_sub.max(self.n_clad);
        if self.n_core <= n_max_clad {
            return None;
        }
        let slab = AsymmetricSlab::new(self.n_sub, self.n_core, self.n_clad, h_total);
        let modes = match pol {
            Polarization::TE => slab.solve_te(wavelength),
            Polarization::TM => slab.solve_tm(wavelength),
        };
        modes.into_iter().next().map(|m| m.n_eff)
    }

    /// Effective index of the ridge region (vertical slab with full height).
    fn n_eff_inner(&self, wavelength: f64, pol: Polarization) -> Option<f64> {
        self.solve_vertical(self.h_ridge + self.h_slab, wavelength, pol)
    }

    /// Effective index of the outer slab region (only slab thickness).
    fn n_eff_outer(&self, wavelength: f64, pol: Polarization) -> Option<f64> {
        if self.h_slab < 1e-12 {
            return None;
        }
        self.solve_vertical(self.h_slab, wavelength, pol)
    }

    /// Effective index of the ridge waveguide using EIM.
    ///
    /// Two-step EIM:
    ///   1. Solve vertical slab → n_eff_inner (ridge region) and n_eff_outer (slab region).
    ///   2. Solve horizontal slab with n_eff_inner as core and n_eff_outer as cladding.
    pub fn n_eff(&self, wavelength: f64, pol: Polarization) -> Option<f64> {
        let n_inner = self.n_eff_inner(wavelength, pol)?;
        let n_outer = self.n_eff_outer(wavelength, pol)?;
        if n_inner <= n_outer {
            return None;
        }
        let horiz = SlabWaveguide::new(n_inner, n_outer, self.width);
        let modes = match pol {
            Polarization::TE => horiz.solve_te(wavelength),
            Polarization::TM => horiz.solve_tm(wavelength),
        };
        modes.into_iter().next().map(|m| m.n_eff)
    }

    /// TE effective index.
    pub fn n_eff_te(&self, wavelength: f64) -> Option<f64> {
        self.n_eff(wavelength, Polarization::TE)
    }

    /// TM effective index.
    pub fn n_eff_tm(&self, wavelength: f64) -> Option<f64> {
        self.n_eff(wavelength, Polarization::TM)
    }

    /// Group index of TE mode (n_g = n_eff - λ·dn_eff/dλ).
    pub fn group_index_te(&self, wavelength: f64) -> Option<f64> {
        let dl = wavelength * 1e-4;
        let n_lo = self.n_eff_te(wavelength - dl)?;
        let n_hi = self.n_eff_te(wavelength + dl)?;
        let n_eff = self.n_eff_te(wavelength)?;
        let dn_dl = (n_hi - n_lo) / (2.0 * dl);
        Some(n_eff - wavelength * dn_dl)
    }

    /// Birefringence: n_eff_TE - n_eff_TM.
    pub fn birefringence(&self, wavelength: f64) -> Option<f64> {
        let nte = self.n_eff_te(wavelength)?;
        let ntm = self.n_eff_tm(wavelength)?;
        Some(nte - ntm)
    }

    /// V-number of the ridge core region.
    ///
    /// V = (2π/λ) · h_total · sqrt(n_core² - n_clad²)
    pub fn v_number(&self, wavelength: f64) -> f64 {
        use std::f64::consts::PI;
        let h = self.h_ridge + self.h_slab;
        let n_max_clad = self.n_sub.max(self.n_clad);
        let na_sq = self.n_core * self.n_core - n_max_clad * n_max_clad;
        if na_sq <= 0.0 {
            return 0.0;
        }
        2.0 * PI / wavelength * h * na_sq.sqrt()
    }

    /// Confinement factor estimate (fraction of power in ridge region).
    ///
    /// Approximated from EIM: Γ ≈ (n_eff² - n_outer²) / (n_inner² - n_outer²)
    pub fn confinement_factor(&self, wavelength: f64) -> Option<f64> {
        let n_eff = self.n_eff_te(wavelength)?;
        let n_inner = self.n_eff_inner(wavelength, Polarization::TE)?;
        let n_outer = self.n_eff_outer(wavelength, Polarization::TE)?;
        let denom = n_inner * n_inner - n_outer * n_outer;
        if denom < 1e-20 {
            return Some(0.0);
        }
        let gamma = (n_eff * n_eff - n_outer * n_outer) / denom;
        Some(gamma.clamp(0.0, 1.0))
    }

    /// Single-mode condition: V < π/2 for symmetric ridge.
    pub fn is_single_mode(&self, wavelength: f64) -> bool {
        self.v_number(wavelength) < std::f64::consts::FRAC_PI_2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn soi_ridge() -> RidgeWaveguide {
        RidgeWaveguide::soi_standard()
    }

    #[test]
    fn ridge_te_mode_exists() {
        let wg = soi_ridge();
        let n = wg.n_eff_te(1550e-9);
        assert!(n.is_some(), "Should find TE mode");
        let n_eff = n.unwrap();
        assert!(n_eff > 1.0 && n_eff < 3.476, "n_eff={n_eff:.4}");
    }

    #[test]
    fn ridge_v_number_positive() {
        let wg = soi_ridge();
        let v = wg.v_number(1550e-9);
        assert!(v > 0.0, "V={v:.4}");
    }

    #[test]
    fn ridge_group_index_in_range() {
        let wg = soi_ridge();
        let ng = wg.group_index_te(1550e-9);
        assert!(ng.is_some(), "Group index should exist");
        let ng = ng.unwrap();
        assert!(ng > 1.5 && ng < 8.0, "ng={ng:.4}");
    }

    #[test]
    fn ridge_birefringence_finite() {
        let wg = soi_ridge();
        if let Some(b) = wg.birefringence(1550e-9) {
            assert!(b.is_finite(), "birefringence={b}");
        }
    }

    #[test]
    fn ridge_confinement_in_range() {
        let wg = soi_ridge();
        if let Some(g) = wg.confinement_factor(1550e-9) {
            assert!((0.0..=1.0).contains(&g), "gamma={g:.4}");
        }
    }

    #[test]
    fn soi_standard_constructor() {
        let wg = RidgeWaveguide::soi_standard();
        assert!((wg.n_core - 3.476).abs() < 1e-6);
        assert!((wg.width - 500e-9).abs() < 1e-15);
    }
}
