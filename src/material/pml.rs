use num_complex::Complex64;
use std::f64::consts::PI;

use crate::material::DispersiveMaterial;
use crate::units::conversion::SPEED_OF_LIGHT;
use crate::units::{RefractiveIndex, Wavelength};

/// Permittivity of free space [F/m]
const EPSILON_0: f64 = 8.854_187_817e-12;

// ─── Per-cell profile output ──────────────────────────────────────────────────

/// Per-cell CFS-PML profile arrays for FDTD absorbing layers.
///
/// Produced by [`Pml::cell_profiles`]. Each vector has length `n_cells`;
/// index 0 corresponds to the innermost (simulation-adjacent) cell, index
/// `n_cells - 1` to the outermost (PEC-backed) cell.
#[derive(Debug, Clone)]
pub struct PmlCellProfiles {
    /// Electric conductivity σ [S/m] — zero at inner face, maximum at outer.
    pub sigma: Vec<f64>,
    /// Coordinate-stretching real part κ (dimensionless) — 1 at inner face.
    pub kappa: Vec<f64>,
    /// CFS frequency shift α [S/m] — maximum at inner face, zero at outer.
    pub alpha: Vec<f64>,
}

// ─── Main PML struct ──────────────────────────────────────────────────────────

/// Perfectly Matched Layer absorbing boundary material.
///
/// Implements the Complex Frequency Shifted PML (CFS-PML) per Roden & Gedney
/// (2000). Provides graded σ(s), κ(s), α(s) profiles and per-cell arrays for
/// FDTD integration.
///
/// ## Profile equations
///
/// Let `s/d ∈ [0, 1]` be the normalised depth into the PML (0 = inner
/// simulation boundary, 1 = outer PEC wall):
///
/// ```text
/// σ(s/d) = σ_max · (s/d)^m
/// κ(s/d) = 1 + (κ_max − 1) · (s/d)^m
/// α(s/d) = α_max · (1 − s/d)^m_a
/// ```
///
/// The complex stretching variable in the frequency domain is:
///
/// ```text
/// s̃(ω) = κ(x) + σ(x) / (ε₀ · (α(x) + jω))
/// ```
///
/// ## Reference
/// J. A. Roden and S. D. Gedney, "Convolution PML (CPML): An efficient FDTD
/// implementation of the CFS-PML for arbitrary media," *Microw. Opt. Technol.
/// Lett.*, vol. 27, no. 5, pp. 334–339, 2000.
#[derive(Debug, Clone)]
pub struct Pml {
    /// Total PML thickness in metres.
    pub thickness: f64,
    /// Polynomial grading exponent `m` (typically 3–4).
    pub polynomial_order: u32,
    /// κ_max ≥ 1.0: real-axis stretch magnitude at the outer PEC wall.
    pub kappa_max: f64,
    /// α_max ≥ 0.0: CFS frequency shift at the inner face [S/m].
    pub alpha_max: f64,
    /// `m_a`: polynomial order for the α grading (typically 1).
    pub alpha_grading_order: u32,
    /// Override for σ_max [S/m]. `None` → compute via Bérenger–Roden formula.
    pub sigma_max: Option<f64>,
    /// Relative permittivity ε_r of the host medium adjacent to the PML.
    pub host_eps_r: f64,
}

// ─── Constructors ─────────────────────────────────────────────────────────────

impl Pml {
    /// Classical Bérenger PML (no CFS stretch, κ = 1, α = 0).
    ///
    /// This preserves the original API so existing code continues to compile.
    pub fn new(thickness: f64, max_conductivity: f64) -> Self {
        Self {
            thickness,
            polynomial_order: 3,
            kappa_max: 1.0,
            alpha_max: 0.0,
            alpha_grading_order: 1,
            sigma_max: Some(max_conductivity),
            host_eps_r: 1.0,
        }
    }

    /// CFS-PML with optimal σ_max for host medium with relative permittivity
    /// `host_eps_r`, κ_max = 15, and a small CFS frequency shift.
    ///
    /// σ_max is resolved lazily from the Bérenger–Roden formula when `dx` is
    /// supplied to [`Pml::sigma_max_resolved`] or [`Pml::cell_profiles`].
    pub fn new_optimal(thickness: f64, host_eps_r: f64) -> Self {
        Self {
            thickness,
            polynomial_order: 3,
            kappa_max: 15.0,
            alpha_max: 0.05 * EPSILON_0,
            alpha_grading_order: 1,
            sigma_max: None,
            host_eps_r,
        }
    }
}

// ─── Profile methods ──────────────────────────────────────────────────────────

impl Pml {
    /// Optimal σ_max [S/m] via the Bérenger–Roden formula.
    ///
    /// ```text
    /// σ_opt = (m + 1) / (150 · π · √ε_r · dx)
    /// ```
    ///
    /// # Arguments
    /// * `dx` — FDTD cell size along the PML direction, in metres.
    pub fn sigma_max_optimal(&self, dx: f64) -> f64 {
        (self.polynomial_order as f64 + 1.0) / (150.0 * PI * self.host_eps_r.sqrt() * dx)
    }

    /// Resolve σ_max: use the stored override if present, otherwise compute the
    /// optimal value.
    ///
    /// # Arguments
    /// * `dx` — FDTD cell size in metres, used only when `sigma_max` is `None`.
    pub fn sigma_max_resolved(&self, dx: f64) -> f64 {
        self.sigma_max.unwrap_or_else(|| self.sigma_max_optimal(dx))
    }

    /// Electric conductivity at normalised depth `s_over_d ∈ [0, 1]`.
    ///
    /// ```text
    /// σ(s/d) = σ_max · (s/d)^m
    /// ```
    ///
    /// # Arguments
    /// * `s_over_d` — normalised depth (0 = inner face, 1 = outer PEC wall).
    /// * `sigma_max` — maximum conductivity [S/m]; obtain via
    ///   [`Pml::sigma_max_resolved`].
    pub fn sigma(&self, s_over_d: f64, sigma_max: f64) -> f64 {
        sigma_max * s_over_d.powf(self.polynomial_order as f64)
    }

    /// Coordinate-stretching real part κ at normalised depth `s_over_d`.
    ///
    /// ```text
    /// κ(s/d) = 1 + (κ_max − 1) · (s/d)^m
    /// ```
    pub fn kappa(&self, s_over_d: f64) -> f64 {
        1.0 + (self.kappa_max - 1.0) * s_over_d.powf(self.polynomial_order as f64)
    }

    /// CFS frequency shift α at normalised depth `s_over_d`.
    ///
    /// ```text
    /// α(s/d) = α_max · (1 − s/d)^m_a
    /// ```
    pub fn alpha(&self, s_over_d: f64) -> f64 {
        self.alpha_max * (1.0 - s_over_d).powf(self.alpha_grading_order as f64)
    }

    /// Compute per-cell σ, κ, and α arrays for FDTD integration.
    ///
    /// Cell centres are placed at `s/d = (i + 0.5) / n_cells` for
    /// `i ∈ 0..n_cells`.
    ///
    /// # Arguments
    /// * `n_cells` — number of PML cells.
    /// * `dx`      — FDTD cell size in metres.
    pub fn cell_profiles(&self, n_cells: usize, dx: f64) -> PmlCellProfiles {
        let sig_max = self.sigma_max_resolved(dx);
        let n = n_cells as f64;

        let sigma: Vec<f64> = (0..n_cells)
            .map(|i| {
                let s = (i as f64 + 0.5) / n;
                self.sigma(s, sig_max)
            })
            .collect();

        let kappa: Vec<f64> = (0..n_cells)
            .map(|i| {
                let s = (i as f64 + 0.5) / n;
                self.kappa(s)
            })
            .collect();

        let alpha: Vec<f64> = (0..n_cells)
            .map(|i| {
                let s = (i as f64 + 0.5) / n;
                self.alpha(s)
            })
            .collect();

        PmlCellProfiles {
            sigma,
            kappa,
            alpha,
        }
    }

    /// Complex effective permittivity at a given point inside the PML.
    ///
    /// The CFS coordinate stretch in the frequency domain is:
    ///
    /// ```text
    /// s̃(ω) = κ + σ / (ε₀ · (α + jω))
    /// ε_eff = ε_r · s̃(ω)
    /// ```
    ///
    /// # Arguments
    /// * `s_over_d` — normalised depth (0 = inner, 1 = outer).
    /// * `omega`    — angular frequency in rad/s.
    /// * `dx`       — FDTD cell size in metres (used to resolve σ_max when not set).
    pub fn complex_eps_eff(&self, s_over_d: f64, omega: f64, dx: f64) -> Complex64 {
        let sig = self.sigma(s_over_d, self.sigma_max_resolved(dx));
        let kap = self.kappa(s_over_d);
        let alp = self.alpha(s_over_d);

        // CFS stretch: s̃(ω) = κ + σ / (ε₀·(α + jω))
        let denom = Complex64::new(alp, omega); // α + jω
        let stretch = Complex64::new(kap, 0.0) + Complex64::new(sig / EPSILON_0, 0.0) / denom;

        Complex64::new(self.host_eps_r, 0.0) * stretch
    }
}

// ─── DispersiveMaterial impl ──────────────────────────────────────────────────

impl DispersiveMaterial for Pml {
    /// Return the complex refractive index n + ik evaluated at the midpoint of
    /// the PML (s/d = 0.5) at the given wavelength.
    ///
    /// A reference FDTD cell size of 1 nm (1e-9 m) is used to resolve σ_max
    /// for the midpoint sample.  The returned index is approximate and intended
    /// for material-database queries; real FDTD simulations use
    /// [`Pml::cell_profiles`] directly.
    fn refractive_index(&self, wavelength: Wavelength) -> RefractiveIndex {
        // Reference cell size for σ_max resolution when used as a database entry.
        const REF_DX: f64 = 1e-9; // 1 nm
        let omega = 2.0 * PI * SPEED_OF_LIGHT / wavelength.0;
        let eps_eff = self.complex_eps_eff(0.5, omega, REF_DX);

        // Principal square root; ensure Im(n) ≥ 0 (absorbing convention).
        let n_complex = eps_eff.sqrt();
        let n_absorbing = if n_complex.im >= 0.0 {
            n_complex
        } else {
            -n_complex
        };

        RefractiveIndex {
            n: n_absorbing.re.abs(),
            k: n_absorbing.im.abs(),
        }
    }

    fn name(&self) -> &str {
        "CFS-PML"
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn sigma_profile_is_polynomial() {
        let pml = Pml::new_optimal(1e-6, 1.0);
        let sig_max = pml.sigma_max_optimal(1e-8);
        let s = 0.5_f64;
        let expected = sig_max * s.powi(3);
        assert_relative_eq!(pml.sigma(s, sig_max), expected, epsilon = 1e-10);
    }

    #[test]
    fn kappa_profile_is_one_at_inner_face() {
        let pml = Pml::new_optimal(1e-6, 1.0);
        assert_relative_eq!(pml.kappa(0.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn kappa_profile_is_kappa_max_at_outer_face() {
        let pml = Pml::new_optimal(1e-6, 1.0);
        assert_relative_eq!(pml.kappa(1.0), pml.kappa_max, epsilon = 1e-12);
    }

    #[test]
    fn alpha_profile_is_alpha_max_at_inner_face() {
        let pml = Pml::new_optimal(1e-6, 1.0);
        assert_relative_eq!(pml.alpha(0.0), pml.alpha_max, epsilon = 1e-30);
    }

    #[test]
    fn alpha_profile_is_zero_at_outer_face() {
        let pml = Pml::new_optimal(1e-6, 1.0);
        assert!(pml.alpha(1.0).abs() < 1e-30);
    }

    #[test]
    fn eps_eff_yields_absorbing_refractive_index() {
        // For e^{+jωt} convention, Im(ε_eff) is negative (lossy medium with σ > 0).
        // Verify by taking the square root and checking that Im(n) is non-zero,
        // confirming the PML is absorbing regardless of sign convention.
        let pml = Pml::new_optimal(1e-6, 1.0);
        let omega = 2.0 * PI * SPEED_OF_LIGHT / 1550e-9;
        let eps = pml.complex_eps_eff(0.5, omega, 1e-8);
        // eps must be non-real (has a significant imaginary part)
        assert!(
            eps.im.abs() > 0.0,
            "PML ε_eff must have nonzero imaginary part (lossy medium)"
        );
        // The complex square root must give Im(n) ≠ 0
        let n_sq = eps.sqrt();
        assert!(
            n_sq.im.abs() > 0.0,
            "Im(n_eff) must be nonzero for PML to absorb; got {:.3e}",
            n_sq.im
        );
    }

    #[test]
    fn optimal_sigma_formula() {
        let pml = Pml::new_optimal(1e-6, 1.0);
        let sig = pml.sigma_max_optimal(1e-8);
        // For m=3, eps_r=1, dx=1e-8: σ_opt = 4 / (150π·1e-8)
        let expected = 4.0 / (150.0 * PI * 1e-8);
        assert_relative_eq!(sig, expected, epsilon = 1e-6 * expected);
    }
}
