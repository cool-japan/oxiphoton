/// Waveguide bend model — bend loss and effective index shift.
///
/// A waveguide bent to radius R experiences two effects:
///   1. Radiation loss: field leaks outward at the bend
///   2. Effective index shift: modes shift toward the outer edge
///
/// The bend loss coefficient α_bend (m⁻¹) for a slab waveguide is given by
/// the Marcatili-Schmeltzer formula (approximate):
///
///   α_bend ≈ C₁ · exp(-C₂ · R)
///
/// where C₁, C₂ are functions of the waveguide cross-section and wavelength.
///
/// For practical silicon photonics strip waveguides, empirical formulas
/// calibrated to FDTD/FEM simulations are used.
use std::f64::consts::PI;

/// Waveguide bend model.
#[derive(Debug, Clone, Copy)]
pub struct WaveguideBend {
    /// Effective index of the straight waveguide mode
    pub n_eff: f64,
    /// Group index (for dispersion)
    pub n_g: f64,
    /// Core material refractive index (e.g. 3.48 for Si, 2.0 for SiN)
    pub n_core: f64,
    /// Cladding refractive index
    pub n_clad: f64,
    /// Equivalent slab core width (m) for loss estimate
    pub core_width: f64,
    /// Operating wavelength (m)
    pub wavelength: f64,
}

impl WaveguideBend {
    pub fn new(
        n_eff: f64,
        n_g: f64,
        n_core: f64,
        n_clad: f64,
        core_width: f64,
        wavelength: f64,
    ) -> Self {
        Self {
            n_eff,
            n_g,
            n_core,
            n_clad,
            core_width,
            wavelength,
        }
    }

    /// Standard SOI strip waveguide at 1550nm (220nm × 500nm, SiO₂ clad).
    pub fn soi_strip_1550() -> Self {
        Self {
            n_eff: 2.44,
            n_g: 4.18,
            n_core: 3.48,
            n_clad: 1.444,
            core_width: 500e-9,
            wavelength: 1550e-9,
        }
    }

    /// Effective index shift due to bending (conformal mapping approximation).
    ///
    /// The equivalent straight waveguide has a modified index profile where
    /// the effective index increases toward the outer edge by:
    ///   Δn_eff ≈ n_eff · w / (2R)
    ///
    /// The dominant mode sees a centroid-shifted effective index.
    pub fn n_eff_bend(&self, radius: f64) -> f64 {
        // First-order perturbation: n_eff_bend ≈ n_eff · (1 + w/(2R))
        self.n_eff * (1.0 + self.core_width / (2.0 * radius))
    }

    /// Bend-induced phase shift (rad) for arc length = 2π·R (full circle).
    pub fn phase_shift_full_ring(&self, radius: f64) -> f64 {
        let k0 = 2.0 * PI / self.wavelength;
        let dn = self.n_eff_bend(radius) - self.n_eff;
        k0 * dn * 2.0 * PI * radius
    }

    /// Returns `(kappa_x, beta, gamma, a)` for the Marcuse 1971 formula.
    ///
    /// - `kappa_x`: transverse wavenumber inside core [rad/m]
    /// - `beta`: longitudinal propagation constant [rad/m]
    /// - `gamma`: transverse decay constant in cladding [rad/m]
    /// - `a`: half core width [m]
    ///
    /// Stability: `gamma` is clamped to `1 / (100 · core_width)` near cutoff to
    /// avoid divide-by-zero. When `kappa_x = 0` (n_eff ≥ n_core, unphysical),
    /// the caller returns `f64::INFINITY`.
    fn mode_parameters(&self) -> (f64, f64, f64, f64) {
        let k0 = 2.0 * PI / self.wavelength;
        let beta = k0 * self.n_eff;

        let kappa_x_sq = k0 * k0 * self.n_core * self.n_core - beta * beta;
        let kappa_x = if kappa_x_sq > 0.0 {
            kappa_x_sq.sqrt()
        } else {
            0.0
        };

        let gamma_sq = beta * beta - k0 * k0 * self.n_clad * self.n_clad;
        let gamma_min = 1.0 / (100.0 * self.core_width);
        let gamma = if gamma_sq > 0.0 {
            gamma_sq.sqrt().max(gamma_min)
        } else {
            gamma_min
        };

        let a = self.core_width / 2.0;
        (kappa_x, beta, gamma, a)
    }

    /// Bend loss for a 90° arc using the Marcuse 1971 conformal-transformation formula.
    ///
    /// Reference: D. Marcuse, "Bending Losses of the Asymmetric Slab Waveguide,"
    /// Bell Syst. Tech. J. 50(8), 2551–2563 (1971), Eq. 26.
    ///
    /// This is a first-order slab (2D) approximation. For full 3D ridge/strip
    /// accuracy a complete mode-solver is needed.
    ///
    /// Stability: near cutoff (n_eff ≈ n_clad), γ is clamped to `1/(100·core_width)`
    /// to avoid divergence. When κ_x = 0 (n_eff ≥ n_core, unphysical for bound mode),
    /// returns `f64::INFINITY`.
    pub fn bend_loss_db_per_90deg(&self, bend_radius_m: f64) -> f64 {
        let (kappa_x, beta, gamma, a) = self.mode_parameters();

        // Guard: kappa_x = 0 means n_eff >= n_core (unphysical for guided mode)
        if kappa_x == 0.0 {
            return f64::INFINITY;
        }

        // Marcuse 1971 bend loss coefficient [Np/m]
        let prefactor = (kappa_x * kappa_x) / (beta * gamma * (1.0 + gamma * a));
        let exp_evanescent = (2.0 * gamma * a).exp();
        let cubic_exp_arg = -(2.0 / 3.0) * (gamma.powi(3) / beta.powi(2)) * bend_radius_m;
        // Guard against underflow: exp(-x) ≈ 0 for x > 745 (f64 min_positive)
        let cubic_exp = if cubic_exp_arg < -745.0 {
            0.0
        } else {
            cubic_exp_arg.exp()
        };

        let alpha_bend_npm = prefactor * exp_evanescent * cubic_exp;

        // Convert to dB/90°: multiply by arc length (π·R/2), convert Np→dB
        alpha_bend_npm * (PI * bend_radius_m / 2.0) * 10.0 / std::f64::consts::LN_10
    }

    /// Bend loss in dB/turn (full 360°) for a ring of radius R.
    pub fn bend_loss_db_per_turn(&self, radius: f64) -> f64 {
        4.0 * self.bend_loss_db_per_90deg(radius)
    }

    /// Minimum bend radius (m) for bend loss < target_db per 90° turn.
    ///
    /// Solves α_bend(R) = target_db by binary search.
    pub fn minimum_bend_radius(&self, target_db: f64) -> f64 {
        let mut r_lo = 1e-6;
        let mut r_hi = 1e-3;
        for _ in 0..50 {
            let r_mid = (r_lo + r_hi) / 2.0;
            if self.bend_loss_db_per_90deg(r_mid) > target_db {
                r_lo = r_mid;
            } else {
                r_hi = r_mid;
            }
        }
        (r_lo + r_hi) / 2.0
    }

    /// Mode mismatch loss (dB) between straight and bent waveguide at junction.
    ///
    /// Approximated as: L_mm ≈ (Δn_eff · L_beat)² / (8 · ln2)
    /// where L_beat = λ / Δn_eff.
    /// Simplified to:
    ///   L_mm ≈ (n_eff · w / (2R))² · (some coupling integral)
    pub fn mode_mismatch_loss_db(&self, radius: f64) -> f64 {
        let dn_relative = self.core_width / (2.0 * radius);
        // Mode overlap loss ~ (1 - exp(-dn²/σ²)) where σ characterises the mode
        let sigma = 0.05; // typical for SOI modes
        let loss = dn_relative * dn_relative / (sigma * sigma);
        4.343 * loss // convert neper to dB
    }

    /// Total bend loss (mode mismatch + radiation) in dB for a 90° arc.
    pub fn total_loss_db_per_90deg(&self, radius: f64) -> f64 {
        self.bend_loss_db_per_90deg(radius) + self.mode_mismatch_loss_db(radius)
    }
}

/// S-bend (offset coupler bend) model.
#[derive(Debug, Clone, Copy)]
pub struct SBend {
    pub waveguide: WaveguideBend,
    /// Lateral offset (m)
    pub offset: f64,
    /// Total horizontal length (m)
    pub length: f64,
}

impl SBend {
    pub fn new(waveguide: WaveguideBend, offset: f64, length: f64) -> Self {
        Self {
            waveguide,
            offset,
            length,
        }
    }

    /// Minimum bend radius in an S-bend: R_min ≈ L²/(8·offset).
    pub fn min_radius(&self) -> f64 {
        self.length * self.length / (8.0 * self.offset)
    }

    /// Estimated total loss (dB) for the complete S-bend.
    pub fn total_loss_db(&self) -> f64 {
        let r_min = self.min_radius();
        // S-bend consists of two 90° arcs
        2.0 * self.waveguide.total_loss_db_per_90deg(r_min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_eff_bend_larger_than_straight() {
        let wg = WaveguideBend::soi_strip_1550();
        let n_bend = wg.n_eff_bend(5e-6);
        assert!(n_bend > wg.n_eff);
    }

    #[test]
    fn larger_radius_less_loss() {
        let wg = WaveguideBend::soi_strip_1550();
        let loss_small = wg.bend_loss_db_per_90deg(2e-6);
        let loss_large = wg.bend_loss_db_per_90deg(50e-6);
        assert!(loss_large < loss_small, "Large R should have less loss");
    }

    #[test]
    fn bend_loss_positive() {
        let wg = WaveguideBend::soi_strip_1550();
        let loss = wg.bend_loss_db_per_90deg(5e-6);
        assert!(loss >= 0.0);
    }

    #[test]
    fn minimum_radius_within_range() {
        let wg = WaveguideBend::soi_strip_1550();
        let r_min = wg.minimum_bend_radius(0.1);
        assert!(r_min > 0.1e-6 && r_min < 500e-6, "r_min={r_min:.2e}");
    }

    #[test]
    fn sbend_radius_formula() {
        let wg = WaveguideBend::soi_strip_1550();
        let sb = SBend::new(wg, 10e-6, 100e-6);
        let r = sb.min_radius();
        // R = 100²/(8×10) = 125μm
        assert!((r - 125e-6).abs() < 1e-9, "r={r:.2e}");
    }

    #[test]
    fn n_eff_bend_approaches_straight_at_large_r() {
        let wg = WaveguideBend::soi_strip_1550();
        let n_bend = wg.n_eff_bend(1.0); // 1m radius → essentially straight
        let rel_diff = (n_bend - wg.n_eff).abs() / wg.n_eff;
        assert!(rel_diff < 1e-6);
    }
}
