//! Graded-index (GRIN) fiber model.
//!
//! GRIN fibers have a radially varying refractive index profile:
//!
//!   n(r) = n₁ · √(1 - 2Δ·(r/a)^α)   for r ≤ a
//!   n(r) = n₂ = n₁·√(1-2Δ)           for r > a
//!
//! where:
//!   n₁  = core peak index
//!   a   = core radius
//!   Δ   = (n₁² - n₂²)/(2n₁²) ≈ (n₁-n₂)/n₁  (relative index difference)
//!   α   = profile exponent (α=2: parabolic, α→∞: step-index)
//!
//! The parabolic (α=2) profile minimises intermodal dispersion and is the
//! standard for multimode GRIN fiber (e.g., OM3/OM4).
//!
//! Key quantities:
//!   - V-number: same as step-index with NA = n₁·√(2Δ)
//!   - Number of modes: N ≈ (α/(α+2)) · V²/2
//!   - Optimal α for minimum modal dispersion: α_opt ≈ 2 - 2·Δ (profile dispersion corrected)

use std::f64::consts::PI;

/// Graded-index fiber with alpha-power profile.
#[derive(Debug, Clone, Copy)]
pub struct GrinFiber {
    /// Peak core refractive index n₁
    pub n_core: f64,
    /// Cladding refractive index n₂
    pub n_clad: f64,
    /// Core radius a (m)
    pub core_radius: f64,
    /// Profile exponent α (2 = parabolic)
    pub alpha: f64,
    /// Relative index difference Δ
    pub delta: f64,
}

impl GrinFiber {
    /// Create a GRIN fiber from parameters.
    pub fn new(n_core: f64, n_clad: f64, core_radius: f64, alpha: f64) -> Self {
        let delta = (n_core * n_core - n_clad * n_clad) / (2.0 * n_core * n_core);
        Self {
            n_core,
            n_clad,
            core_radius,
            alpha,
            delta,
        }
    }

    /// Standard 50/125 µm OM3 multimode GRIN fiber at 850 nm.
    ///
    /// n₁=1.4804, n₂=1.4585 (Δ≈1.5%), α=2 (parabolic), a=25µm.
    pub fn om3_50_125() -> Self {
        Self::new(1.4804, 1.4585, 25e-6, 2.0)
    }

    /// Standard 62.5/125 µm OM1 multimode GRIN fiber.
    ///
    /// n₁=1.4964, n₂=1.4725, a=31.25µm.
    pub fn om1_62_125() -> Self {
        Self::new(1.4964, 1.4725, 31.25e-6, 2.0)
    }

    /// Numerical aperture: NA = n₁·√(2Δ).
    pub fn numerical_aperture(&self) -> f64 {
        self.n_core * (2.0 * self.delta).sqrt()
    }

    /// V-number at given wavelength (m).
    pub fn v_number(&self, wavelength: f64) -> f64 {
        2.0 * PI * self.core_radius * self.numerical_aperture() / wavelength
    }

    /// Number of guided modes (approximate, for large V).
    ///
    ///   N ≈ (α/(α+2)) · V²/2
    pub fn number_of_modes(&self, wavelength: f64) -> f64 {
        let v = self.v_number(wavelength);
        (self.alpha / (self.alpha + 2.0)) * v * v / 2.0
    }

    /// Refractive index at radius r (m) from fiber axis.
    pub fn index_at_radius(&self, r: f64) -> f64 {
        let a = self.core_radius;
        if r <= a {
            self.n_core * (1.0 - 2.0 * self.delta * (r / a).powf(self.alpha)).sqrt()
        } else {
            self.n_clad
        }
    }

    /// Acceptance angle θ_max (rad) in air, from NA.
    pub fn acceptance_angle(&self) -> f64 {
        self.numerical_aperture().asin()
    }

    /// Intermodal dispersion (ps/km) for parabolic profile (α=2).
    ///
    /// For optimal α=2 profile: δτ/L ≈ n₁·Δ²/(2c) × ... (reduced by ~1000× vs step-index)
    /// Approximate formula: δτ/L ≈ n₁·Δ²/(20c) [ps/km for α=2]
    pub fn intermodal_dispersion_ps_per_km(&self) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        // For α=2: δτ/L = n₁·Δ²/(20·c) in s/m → ps/km
        let dt_per_m = self.n_core * self.delta * self.delta / (20.0 * SPEED_OF_LIGHT);
        dt_per_m * 1e12 * 1e3 // convert s/m → ps/km
    }

    /// Bandwidth-distance product B·L (GHz·km) for parabolic GRIN.
    ///
    ///   B·L ≈ 0.44 / δτ
    pub fn bandwidth_distance_product_ghz_km(&self) -> f64 {
        let dt_ps_km = self.intermodal_dispersion_ps_per_km();
        if dt_ps_km < 1e-30 {
            return f64::INFINITY;
        }
        // B·L = 0.44/δτ where δτ in ns/km and B in GHz
        0.44 / (dt_ps_km * 1e-3) // δτ in ns/km
    }

    /// Optimal profile exponent α_opt that minimises intermodal dispersion.
    ///
    /// Including profile dispersion correction:
    ///   α_opt ≈ 2 + ε - Δ·(4+ε)(3+ε)/(3+2ε)
    /// For simplicity, we use α_opt ≈ 2 - 2·Δ (Olshansky approximation).
    pub fn optimal_alpha(&self) -> f64 {
        2.0 - 2.0 * self.delta
    }

    /// Radial index profile as a vector over n_pts points from 0 to 2a.
    pub fn index_profile(&self, n_pts: usize) -> Vec<(f64, f64)> {
        let r_max = 2.0 * self.core_radius;
        (0..n_pts)
            .map(|i| {
                let r = i as f64 * r_max / (n_pts - 1) as f64;
                (r, self.index_at_radius(r))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grin_na_positive() {
        let f = GrinFiber::om3_50_125();
        assert!(f.numerical_aperture() > 0.0);
        assert!(f.numerical_aperture() < 0.5);
    }

    #[test]
    fn grin_index_peak_at_center() {
        let f = GrinFiber::om3_50_125();
        assert!((f.index_at_radius(0.0) - f.n_core).abs() < 1e-10);
    }

    #[test]
    fn grin_index_at_boundary_close_to_clad() {
        let f = GrinFiber::om3_50_125();
        let n_at_a = f.index_at_radius(f.core_radius);
        assert!((n_at_a - f.n_clad).abs() < 0.01);
    }

    #[test]
    fn grin_index_outside_core_is_clad() {
        let f = GrinFiber::om3_50_125();
        assert!((f.index_at_radius(2.0 * f.core_radius) - f.n_clad).abs() < 1e-10);
    }

    #[test]
    fn grin_v_number_large_multimode() {
        let f = GrinFiber::om3_50_125();
        let v = f.v_number(850e-9);
        assert!(v > 20.0, "V={v:.1} — should be multimode");
    }

    #[test]
    fn grin_modes_many() {
        let f = GrinFiber::om3_50_125();
        let n = f.number_of_modes(850e-9);
        assert!(n > 100.0, "N≈{n:.0}");
    }

    #[test]
    fn grin_intermodal_dispersion_positive() {
        let f = GrinFiber::om3_50_125();
        let dt = f.intermodal_dispersion_ps_per_km();
        assert!(dt > 0.0 && dt < 1000.0, "δτ={dt:.1} ps/km");
    }

    #[test]
    fn grin_bandwidth_product_reasonable() {
        let f = GrinFiber::om3_50_125();
        let bw = f.bandwidth_distance_product_ghz_km();
        // OM3 spec: ≥2000 MHz·km at 850nm
        assert!(bw > 0.1, "B·L={bw:.1} GHz·km");
    }

    #[test]
    fn grin_optimal_alpha_near_2() {
        let f = GrinFiber::om3_50_125();
        let a = f.optimal_alpha();
        assert!(a > 1.9 && a < 2.1, "α_opt={a:.3}");
    }

    #[test]
    fn grin_index_profile_length() {
        let f = GrinFiber::om3_50_125();
        let prof = f.index_profile(100);
        assert_eq!(prof.len(), 100);
    }
}
