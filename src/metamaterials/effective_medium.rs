//! Effective medium theory (EMT) — Maxwell Garnett, Bruggeman, and multilayer EMT.

// ---------------------------------------------------------------------------
// Maxwell Garnett (MG) effective medium
// ---------------------------------------------------------------------------

/// Maxwell Garnett effective medium theory for spherical inclusions in a host matrix.
///
/// Valid in the dilute limit (fill fraction f < 0.3).
#[derive(Debug, Clone)]
pub struct MaxwellGarnett {
    /// Permittivity of the host (background) medium.
    pub eps_host: f64,
    /// Permittivity of the spherical inclusions.
    pub eps_inclusion: f64,
    /// Volume fill fraction of the inclusions (0 < f ≤ 1).
    pub fill_fraction: f64,
}

impl MaxwellGarnett {
    /// Effective permittivity:
    ///
    /// ε_eff = ε_h  [1 + 3f (ε_i − ε_h) / (ε_i + 2ε_h − f(ε_i − ε_h))]
    pub fn effective_permittivity(&self) -> f64 {
        let eh = self.eps_host;
        let ei = self.eps_inclusion;
        let f = self.fill_fraction.clamp(0.0, 1.0);
        let delta = ei - eh;
        let denom = ei + 2.0 * eh - f * delta;
        if denom.abs() < 1e-30 {
            // Degenerate (metallic resonance): return large value
            return eh * (1.0 + 3.0 * f * delta.signum() * 1e15);
        }
        eh * (1.0 + 3.0 * f * delta / denom)
    }

    /// Effective refractive index: n_eff = √(max(ε_eff, 0)).
    pub fn effective_index(&self) -> f64 {
        self.effective_permittivity().max(0.0).sqrt()
    }

    /// Returns `true` for the dilute limit where MG theory is reliable (f < 0.3).
    pub fn is_valid(&self) -> bool {
        self.fill_fraction < 0.3
    }

    /// Clausius–Mossotti polarisability factor (β) for a single spherical inclusion:
    ///
    /// β = (ε_i − ε_h) / (ε_i + 2 ε_h)
    ///
    /// The MG mixing formula uses `3 f β`, i.e. this factor × 3 × fill_fraction.
    pub fn clausius_mossotti_factor(&self) -> f64 {
        let delta = self.eps_inclusion - self.eps_host;
        let denom = self.eps_inclusion + 2.0 * self.eps_host;
        if denom.abs() < 1e-30 {
            return f64::INFINITY;
        }
        delta / denom
    }
}

// ---------------------------------------------------------------------------
// Bruggeman symmetric EMT
// ---------------------------------------------------------------------------

/// Bruggeman effective medium theory — symmetric formulation valid for all fill fractions,
/// including near the percolation threshold.
///
/// Solves the implicit equation:
///
/// f₁ (ε₁ − ε_eff)/(ε₁ + 2 ε_eff) + f₂ (ε₂ − ε_eff)/(ε₂ + 2 ε_eff) = 0
#[derive(Debug, Clone)]
pub struct BruggemanEmt {
    /// Permittivity of material 1.
    pub eps1: f64,
    /// Permittivity of material 2.
    pub eps2: f64,
    /// Volume fraction of material 1 (f₂ = 1 − f₁).
    pub f1: f64,
}

impl BruggemanEmt {
    fn f2(&self) -> f64 {
        (1.0 - self.f1).clamp(0.0, 1.0)
    }

    /// Residual of the Bruggeman equation at a trial ε_eff.
    fn residual(&self, eps_eff: f64) -> f64 {
        let t1 = {
            let d = self.eps1 + 2.0 * eps_eff;
            if d.abs() < 1e-60 {
                0.0
            } else {
                self.f1 * (self.eps1 - eps_eff) / d
            }
        };
        let t2 = {
            let d = self.eps2 + 2.0 * eps_eff;
            if d.abs() < 1e-60 {
                0.0
            } else {
                self.f2() * (self.eps2 - eps_eff) / d
            }
        };
        t1 + t2
    }

    /// Solve the Bruggeman equation via bisection.
    ///
    /// The analytic solution of the Bruggeman cubic is:
    ///
    /// ε_eff = (1/4){ (3f₁−1)ε₁ + (3f₂−1)ε₂
    ///         ± √[(3f₁−1)²ε₁² + (3f₂−1)²ε₂² + 2ε₁ε₂(2 + 9f₁f₂ − 3(f₁+f₂))] }
    ///
    /// We use the analytic form for speed and correctness.
    pub fn effective_permittivity(&self) -> f64 {
        let f1 = self.f1.clamp(0.0, 1.0);
        let f2 = 1.0 - f1;
        let e1 = self.eps1;
        let e2 = self.eps2;

        // Coefficients of the Bruggeman cubic reduced to quadratic for the
        // symmetric 3-D sphere model:
        // (3f1-1)ε1 + (3f2-1)ε2  ± sqrt(...)
        let b = (3.0 * f1 - 1.0) * e1 + (3.0 * f2 - 1.0) * e2;
        let disc = b * b + 8.0 * e1 * e2;

        if disc < 0.0 {
            // Complex solution — fall back to iterative refinement
            return self.effective_permittivity_iterative();
        }

        let eps_plus = (b + disc.sqrt()) / 4.0;
        let eps_minus = (b - disc.sqrt()) / 4.0;

        // Choose the physically meaningful root (one that lies between ε₁ and ε₂,
        // or satisfies the original equation with smallest |residual|).
        let r_plus = self.residual(eps_plus).abs();
        let r_minus = self.residual(eps_minus).abs();

        if r_plus <= r_minus {
            eps_plus
        } else {
            eps_minus
        }
    }

    /// Iterative (bisection) fallback for the Bruggeman equation.
    fn effective_permittivity_iterative(&self) -> f64 {
        let lo = self.eps1.min(self.eps2);
        let hi = self.eps1.max(self.eps2);
        let mut a = lo;
        let mut b = hi;
        let mut mid = (a + b) / 2.0;
        for _ in 0..200 {
            mid = (a + b) / 2.0;
            let r_mid = self.residual(mid);
            let r_a = self.residual(a);
            if r_mid.abs() < 1e-12 {
                break;
            }
            if r_a * r_mid < 0.0 {
                b = mid;
            } else {
                a = mid;
            }
        }
        mid
    }

    /// Effective refractive index n_eff = √(max(ε_eff, 0)).
    pub fn effective_index(&self) -> f64 {
        self.effective_permittivity().max(0.0).sqrt()
    }

    /// Percolation threshold for 3-D spheres: f_c = 1/3.
    pub fn percolation_threshold(&self) -> f64 {
        1.0 / 3.0
    }

    /// Returns `true` when material 1 forms a percolating network (f1 > f_c).
    pub fn is_percolated(&self) -> bool {
        self.f1 > self.percolation_threshold()
    }
}

// ---------------------------------------------------------------------------
// 1-D Multilayer EMT (anisotropic)
// ---------------------------------------------------------------------------

/// 1-D multilayer effective medium theory for a periodic bilayer structure.
///
/// The in-plane and out-of-plane components are:
///
/// ε_‖ = f_a ε_a + f_b ε_b         (parallel / TM / ordinary ray)
/// 1/ε_⊥ = f_a/ε_a + f_b/ε_b      (series / TE / extraordinary ray)
#[derive(Debug, Clone)]
pub struct MultilayerEmt {
    /// Permittivity of layer A.
    pub eps_a: f64,
    /// Permittivity of layer B.
    pub eps_b: f64,
    /// Volume fill fraction of layer A (f_b = 1 − f_a).
    pub f_a: f64,
}

impl MultilayerEmt {
    /// In-plane (TM / ordinary) effective permittivity — parallel capacitor model.
    pub fn eps_parallel(&self) -> f64 {
        let f_b = 1.0 - self.f_a;
        self.f_a * self.eps_a + f_b * self.eps_b
    }

    /// Out-of-plane (TE / extraordinary) effective permittivity — series capacitor model.
    pub fn eps_perpendicular(&self) -> f64 {
        let f_b = 1.0 - self.f_a;
        // Guard against division by zero while preserving sign.
        let inv_a = if self.eps_a.abs() < 1e-30 {
            self.eps_a.signum() * 1e30
        } else {
            1.0 / self.eps_a
        };
        let inv_b = if self.eps_b.abs() < 1e-30 {
            self.eps_b.signum() * 1e30
        } else {
            1.0 / self.eps_b
        };
        let inv = self.f_a * inv_a + f_b * inv_b;
        if inv.abs() < 1e-30 {
            return f64::INFINITY;
        }
        1.0 / inv
    }

    /// Returns `true` when the medium is hyperbolic: ε_‖ × ε_⊥ < 0.
    pub fn is_hyperbolic(&self) -> bool {
        self.eps_parallel() * self.eps_perpendicular() < 0.0
    }

    /// Type-I hyperbolic: ε_⊥ < 0 and ε_‖ > 0.
    pub fn is_type_i_hyperbolic(&self) -> bool {
        self.eps_perpendicular() < 0.0 && self.eps_parallel() > 0.0
    }

    /// Type-II hyperbolic: ε_‖ < 0 and ε_⊥ > 0.
    pub fn is_type_ii_hyperbolic(&self) -> bool {
        self.eps_parallel() < 0.0 && self.eps_perpendicular() > 0.0
    }

    /// Optical axis direction for extraordinary-ray propagation.
    ///
    /// Returns `true` for z-axis anisotropy (normal to layers).
    pub fn is_uniaxial_normal(&self) -> bool {
        (self.eps_a - self.eps_b).abs() > 1e-10
    }

    /// Birefringence: Δn = |n_‖ − n_⊥|.
    pub fn birefringence(&self) -> f64 {
        let n_par = self.eps_parallel().max(0.0).sqrt();
        let n_perp = self.eps_perpendicular().max(0.0).sqrt();
        (n_par - n_perp).abs()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maxwell_garnett_dilute_limit() {
        // At f → 0, ε_eff → ε_host
        let mg = MaxwellGarnett {
            eps_host: 2.25,
            eps_inclusion: 12.0,
            fill_fraction: 0.001,
        };
        let eps = mg.effective_permittivity();
        assert!(
            (eps - 2.25).abs() < 0.05,
            "ε_eff should approach ε_host in dilute limit, got {}",
            eps
        );
    }

    #[test]
    fn maxwell_garnett_full_inclusion() {
        // At f = 1, ε_eff → ε_inclusion
        let mg = MaxwellGarnett {
            eps_host: 1.0,
            eps_inclusion: 5.0,
            fill_fraction: 0.999,
        };
        let eps = mg.effective_permittivity();
        // Should be close to ε_inclusion
        assert!(
            (eps - 5.0).abs() < 1.0,
            "ε_eff should approach ε_inclusion at high f, got {}",
            eps
        );
    }

    #[test]
    fn bruggeman_symmetric() {
        // With f1 = f2 = 0.5, ε_eff must satisfy Bruggeman equation
        let bg = BruggemanEmt {
            eps1: 1.0,
            eps2: 9.0,
            f1: 0.5,
        };
        let eps = bg.effective_permittivity();
        let residual = bg.residual(eps);
        assert!(
            residual.abs() < 1e-8,
            "Bruggeman residual should be ~0, got {}",
            residual
        );
    }

    #[test]
    fn multilayer_hyperbolic_condition() {
        // Ag/TiO₂: ε_Ag = -10, ε_TiO₂ = 7, f_Ag = 0.5
        let ml = MultilayerEmt {
            eps_a: -10.0,
            eps_b: 7.0,
            f_a: 0.5,
        };
        assert!(ml.is_hyperbolic(), "Should be hyperbolic");
        // ε_‖ = 0.5(-10) + 0.5(7) = -1.5 < 0 → type II
        assert!(ml.is_type_ii_hyperbolic(), "Should be Type-II hyperbolic");
    }

    #[test]
    fn multilayer_isotropic_same_material() {
        // Same material in both layers → isotropic
        let ml = MultilayerEmt {
            eps_a: 4.0,
            eps_b: 4.0,
            f_a: 0.3,
        };
        assert!((ml.eps_parallel() - 4.0).abs() < 1e-10);
        assert!((ml.eps_perpendicular() - 4.0).abs() < 1e-10);
        assert!(!ml.is_hyperbolic());
    }

    #[test]
    fn multilayer_type_i_hyperbolic() {
        // ε_a > 0, ε_b < 0, f_a large so ε_‖ > 0 but ε_⊥ < 0
        let ml = MultilayerEmt {
            eps_a: 10.0,
            eps_b: -1.0,
            f_a: 0.9,
        };
        // ε_‖ = 0.9(10) + 0.1(-1) = 8.9 > 0
        // 1/ε_⊥ = 0.9/10 + 0.1/(-1) = 0.09 - 0.1 = -0.01 → ε_⊥ = -100 < 0
        assert!(ml.is_type_i_hyperbolic(), "Should be Type-I hyperbolic");
    }

    #[test]
    fn bruggeman_pure_material_limit() {
        // f1 = 1 → ε_eff = ε1
        let bg = BruggemanEmt {
            eps1: 4.0,
            eps2: 9.0,
            f1: 1.0,
        };
        let eps = bg.effective_permittivity();
        assert!(
            (eps - 4.0).abs() < 0.1,
            "f1=1 should give ε_eff ≈ ε1, got {}",
            eps
        );
    }

    #[test]
    fn clausius_mossotti_factor() {
        let mg = MaxwellGarnett {
            eps_host: 1.0,
            eps_inclusion: 4.0,
            fill_fraction: 0.1,
        };
        // (4-1)/(4+2) = 0.5
        let cm = mg.clausius_mossotti_factor();
        assert!(
            (cm - 0.5).abs() < 1e-10,
            "CM factor should be 0.5, got {}",
            cm
        );
    }
}
