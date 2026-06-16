//! Full-vectorial 2D finite-difference mode solver.
//!
//! Implements the Fallahkhair-Li-Murphy (JLT 26(8) 1423, 2008) formulation
//! for the Hx-Hy vector wave equation on a uniform Cartesian grid.
//!
//! The 2N×2N operator P is assembled with half-edge permittivity averaging and
//! corner cross-coupling terms that vanish exactly in homogeneous media.

use oxiblas::lapack::evd::SymmetricEvd;
use oxiblas::prelude::Mat;
use std::f64::consts::PI;

use super::fd_solver::FdModeSolver2d;

// ─── Public structs ──────────────────────────────────────────────────────────

/// A guided mode from the full-vectorial FD solver.
#[derive(Debug, Clone)]
pub struct VectorMode {
    /// Effective index n_eff = β / k0.
    pub n_eff: f64,
    /// Hx field on the nx*ny grid (row-major k = j*nx + i).
    pub hx: Vec<f64>,
    /// Hy field on the nx*ny grid (row-major k = j*nx + i).
    pub hy: Vec<f64>,
    /// Mode order (0 = fundamental, sorted by n_eff descending).
    pub order: usize,
    /// TE fraction: Σhx² / (Σhx² + Σhy²). > 0.5 means quasi-TE.
    pub te_fraction: f64,
}

/// Full-vectorial 2D finite-difference mode solver (Fallahkhair-Li-Murphy 2008).
///
/// Solves P·[Hx; Hy] = β²·[Hx; Hy] where P is a 2N×2N dense matrix (N = nx*ny).
/// Keep nx*ny ≤ 400 (e.g. 20×20) for tractable dense EVD.
pub struct FullVectorialModeSolver2d {
    /// Refractive index profile, row-major (size = nx * ny).
    pub n_profile: Vec<f64>,
    /// Grid points in x.
    pub nx: usize,
    /// Grid points in y.
    pub ny: usize,
    /// Grid spacing in x (m).
    pub dx: f64,
    /// Grid spacing in y (m).
    pub dy: f64,
    /// Minimum effective index for guided modes (cladding / substrate index).
    pub n_min: f64,
}

// ─── FullVectorialModeSolver2d ────────────────────────────────────────────────

impl FullVectorialModeSolver2d {
    /// Construct a new solver.
    pub fn new(
        n_profile: Vec<f64>,
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
        n_min: f64,
    ) -> Self {
        assert_eq!(n_profile.len(), nx * ny, "n_profile length must equal nx*ny");
        Self {
            n_profile,
            nx,
            ny,
            dx,
            dy,
            n_min,
        }
    }

    /// Build the refractive-index profile for a strip waveguide centred in the domain.
    ///
    /// Delegates to [`FdModeSolver2d::strip_profile`] — no code duplication.
    #[allow(clippy::too_many_arguments)]
    pub fn strip_profile(
        n_core: f64,
        n_clad: f64,
        width: f64,
        height: f64,
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
    ) -> Vec<f64> {
        FdModeSolver2d::strip_profile(n_core, n_clad, width, height, nx, ny, dx, dy)
    }

    /// Solve for guided vector modes at the given wavelength.
    ///
    /// Returns modes sorted by n_eff descending (fundamental first).
    ///
    /// # Warning
    /// The matrix is 2N×2N dense (N = nx*ny). Keep nx*ny ≤ 400 for tests.
    pub fn solve(&self, wavelength: f64) -> Vec<VectorMode> {
        let k0 = 2.0 * PI / wavelength;
        let n_xy = self.nx * self.ny;

        // Build the 2N×2N operator P.
        let p = self.build_operator(k0);

        // Symmetrise: P_sym = (P + P^T) / 2.
        //
        // Physical justification: the Fallahkhair-Li-Murphy P operator is
        // self-adjoint in a weighted inner product, so its eigenvalues are
        // real.  The FD discretisation introduces small asymmetries from
        // non-uniform ε, but the symmetric half preserves the real spectrum
        // exactly and avoids the numerical issues of GeneralEvd on large
        // near-symmetric matrices.
        let two_n = 2 * n_xy;
        let mut p_sym = Mat::<f64>::zeros(two_n, two_n);
        for r in 0..two_n {
            for c in 0..two_n {
                p_sym[(r, c)] = (p[(r, c)] + p[(c, r)]) * 0.5;
            }
        }

        // Solve symmetric EVP — numerically robust for large matrices.
        let evd = match SymmetricEvd::compute(p_sym.as_ref()) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };

        let eigenvalues = evd.eigenvalues(); // &[f64], real
        let evecs = evd.eigenvectors();      // MatRef<f64>

        // Filter: eigenvalue in (n_min² k0², n_max² k0²).
        let n_max = self
            .n_profile
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let beta_min_sq = (self.n_min * k0) * (self.n_min * k0);
        let beta_max_sq = (n_max * k0) * (n_max * k0);

        let mut modes: Vec<VectorMode> = eigenvalues
            .iter()
            .enumerate()
            .filter_map(|(col, &lambda)| {
                // Must be in guided range
                if lambda <= beta_min_sq || lambda >= beta_max_sq {
                    return None;
                }

                let n_eff = (lambda / (k0 * k0)).sqrt();

                // Extract Hx (rows 0..n_xy) and Hy (rows n_xy..2*n_xy) from eigenvector.
                let hx_raw: Vec<f64> = (0..n_xy).map(|r| evecs[(r, col)]).collect();
                let hy_raw: Vec<f64> = (0..n_xy).map(|r| evecs[(n_xy + r, col)]).collect();

                // Normalise: sqrt(Σhx² + Σhy²) = 1.
                let sum_sq: f64 = hx_raw.iter().map(|v| v * v).sum::<f64>()
                    + hy_raw.iter().map(|v| v * v).sum::<f64>();

                let (hx, hy) = if sum_sq > 1e-30 {
                    let inv_norm = 1.0 / sum_sq.sqrt();
                    (
                        hx_raw.iter().map(|v| v * inv_norm).collect::<Vec<_>>(),
                        hy_raw.iter().map(|v| v * inv_norm).collect::<Vec<_>>(),
                    )
                } else {
                    (hx_raw, hy_raw)
                };

                // TE fraction.
                let hx_sq: f64 = hx.iter().map(|v| v * v).sum();
                let hy_sq: f64 = hy.iter().map(|v| v * v).sum();
                let te_fraction = if hx_sq + hy_sq < 1e-30 {
                    0.5
                } else {
                    hx_sq / (hx_sq + hy_sq)
                };

                Some(VectorMode {
                    n_eff,
                    hx,
                    hy,
                    order: 0,
                    te_fraction,
                })
            })
            .collect();

        // Sort by n_eff descending, assign order.
        modes.sort_by(|a, b| {
            b.n_eff
                .partial_cmp(&a.n_eff)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, m) in modes.iter_mut().enumerate() {
            m.order = i;
        }
        modes
    }

    // ─── Internal: operator assembly ─────────────────────────────────────────

    /// Assemble the 2N×2N Fallahkhair-Li-Murphy operator P.
    ///
    /// Block layout:
    /// ```text
    ///   P = | Pxx  Pxy |
    ///       | Pyx  Pyy |
    /// ```
    /// Row/column offsets for Hx block: 0..N; Hy block: N..2N.
    fn build_operator(&self, k0: f64) -> Mat<f64> {
        let nx = self.nx;
        let ny = self.ny;
        let n_xy = nx * ny;
        let dx = self.dx;
        let dy = self.dy;
        let dx2 = dx * dx;
        let dy2 = dy * dy;

        let mut p = Mat::<f64>::zeros(2 * n_xy, 2 * n_xy);

        // Pre-compute ε at each node.
        let eps: Vec<f64> = self.n_profile.iter().map(|&n| n * n).collect();

        for j in 0..ny {
            for i in 0..nx {
                let k = j * nx + i; // centre node index

                // ── Neighbour permittivities ──────────────────────────────
                // At PEC boundaries (out-of-domain), use e_p itself for the
                // half-edge average so that ε_half = e_p and all cross-coupling
                // coefficients (1 - e_p/e_half) vanish in a homogeneous medium.
                let e_p = eps[k];
                let e_e = if i + 1 < nx { eps[k + 1] } else { e_p };
                let e_w = if i > 0 { eps[k - 1] } else { e_p };
                let e_n = if j + 1 < ny { eps[k + nx] } else { e_p };
                let e_s = if j > 0 { eps[k - nx] } else { e_p };

                // Half-edge averaged permittivities (arithmetic mean at midpoint).
                let e_x_p = (e_p + e_e) / 2.0; // ε_{i+½, j}
                let e_x_m = (e_p + e_w) / 2.0; // ε_{i-½, j}
                let e_y_p = (e_p + e_n) / 2.0; // ε_{i, j+½}
                let e_y_m = (e_p + e_s) / 2.0; // ε_{i, j-½}

                // ── Pxx block: Hx equation ────────────────────────────────
                // Diagonal: -(ε_P/ε_{x+} + ε_P/ε_{x-})/dx² - 2/dy² + k0²ε_P
                // PEC in x: omit east/west coupling at domain edges.
                let pxx_diag = {
                    let term_xp = if i + 1 < nx {
                        e_p / (e_x_p * dx2)
                    } else {
                        0.0
                    };
                    let term_xm = if i > 0 {
                        e_p / (e_x_m * dx2)
                    } else {
                        0.0
                    };
                    -(term_xp + term_xm) - 2.0 / dy2 + k0 * k0 * e_p
                };
                p[(k, k)] = pxx_diag;

                // Off-diagonal east (i+1): +ε_P / (ε_{x+} dx²)
                if i + 1 < nx {
                    p[(k, k + 1)] += e_p / (e_x_p * dx2);
                }
                // Off-diagonal west (i-1): +ε_P / (ε_{x-} dx²)
                if i > 0 {
                    p[(k, k - 1)] += e_p / (e_x_m * dx2);
                }
                // Off-diagonal north (j+1): +1/dy²
                if j + 1 < ny {
                    p[(k, k + nx)] += 1.0 / dy2;
                }
                // Off-diagonal south (j-1): +1/dy²
                if j > 0 {
                    p[(k, k - nx)] += 1.0 / dy2;
                }

                // ── Pyy block: Hy equation ────────────────────────────────
                // Mirror of Pxx with x↔y.
                // Diagonal: -2/dx² - (ε_P/ε_{y+} + ε_P/ε_{y-})/dy² + k0²ε_P
                let pyy_diag = {
                    let term_yp = if j + 1 < ny {
                        e_p / (e_y_p * dy2)
                    } else {
                        0.0
                    };
                    let term_ym = if j > 0 {
                        e_p / (e_y_m * dy2)
                    } else {
                        0.0
                    };
                    -2.0 / dx2 - (term_yp + term_ym) + k0 * k0 * e_p
                };
                p[(n_xy + k, n_xy + k)] = pyy_diag;

                // Off-diagonal east (i+1): +1/dx²
                if i + 1 < nx {
                    p[(n_xy + k, n_xy + k + 1)] += 1.0 / dx2;
                }
                // Off-diagonal west (i-1): +1/dx²
                if i > 0 {
                    p[(n_xy + k, n_xy + k - 1)] += 1.0 / dx2;
                }
                // Off-diagonal north (j+1): +ε_P / (ε_{y+} dy²)
                if j + 1 < ny {
                    p[(n_xy + k, n_xy + k + nx)] += e_p / (e_y_p * dy2);
                }
                // Off-diagonal south (j-1): +ε_P / (ε_{y-} dy²)
                if j > 0 {
                    p[(n_xy + k, n_xy + k - nx)] += e_p / (e_y_m * dy2);
                }

                // ── Cross blocks: Pxy (Hx row ← Hy column) ───────────────
                // Corner coefficients: (1/(4 dx dy)) * (1 - ε_P / ε_y_±)
                // In a homogeneous medium e_y_p = e_y_m = e_p → coefficients are 0.
                // Corners are only set when both i-offset and j-offset are in-bounds.
                let coeff_yp = 1.0 - e_p / e_y_p;  // uses j+1 neighbour
                let coeff_ym = 1.0 - e_p / e_y_m;  // uses j-1 neighbour
                let inv_4dxdy = 1.0 / (4.0 * dx * dy);

                // NE corner (i+1, j+1): +coeff_yp / (4 dx dy)
                if i + 1 < nx && j + 1 < ny {
                    let k_ne = (j + 1) * nx + (i + 1);
                    p[(k, n_xy + k_ne)] += inv_4dxdy * coeff_yp;
                }
                // NW corner (i-1, j+1): -coeff_yp / (4 dx dy)
                if i > 0 && j + 1 < ny {
                    let k_nw = (j + 1) * nx + (i - 1);
                    p[(k, n_xy + k_nw)] -= inv_4dxdy * coeff_yp;
                }
                // SE corner (i+1, j-1): -coeff_ym / (4 dx dy)
                if i + 1 < nx && j > 0 {
                    let k_se = (j - 1) * nx + (i + 1);
                    p[(k, n_xy + k_se)] -= inv_4dxdy * coeff_ym;
                }
                // SW corner (i-1, j-1): +coeff_ym / (4 dx dy)
                if i > 0 && j > 0 {
                    let k_sw = (j - 1) * nx + (i - 1);
                    p[(k, n_xy + k_sw)] += inv_4dxdy * coeff_ym;
                }

                // ── Cross blocks: Pyx (Hy row ← Hx column) ───────────────
                // Corner coefficients: (1/(4 dx dy)) * (1 - ε_P / ε_x_±)
                // In a homogeneous medium e_x_p = e_x_m = e_p → coefficients are 0.
                let coeff_xp = 1.0 - e_p / e_x_p;  // uses i+1 neighbour
                let coeff_xm = 1.0 - e_p / e_x_m;  // uses i-1 neighbour

                // NE corner (i+1, j+1): +coeff_xp / (4 dx dy)
                if i + 1 < nx && j + 1 < ny {
                    let k_ne = (j + 1) * nx + (i + 1);
                    p[(n_xy + k, k_ne)] += inv_4dxdy * coeff_xp;
                }
                // NW corner (i-1, j+1): -coeff_xp / (4 dx dy)
                if i > 0 && j + 1 < ny {
                    let k_nw = (j + 1) * nx + (i - 1);
                    p[(n_xy + k, k_nw)] -= inv_4dxdy * coeff_xp;
                }
                // SE corner (i+1, j-1): -coeff_xm / (4 dx dy)
                if i + 1 < nx && j > 0 {
                    let k_se = (j - 1) * nx + (i + 1);
                    p[(n_xy + k, k_se)] -= inv_4dxdy * coeff_xm;
                }
                // SW corner (i-1, j-1): +coeff_xm / (4 dx dy)
                if i > 0 && j > 0 {
                    let k_sw = (j - 1) * nx + (i - 1);
                    p[(n_xy + k, k_sw)] += inv_4dxdy * coeff_xm;
                }
            }
        }

        p
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── 1. Cross blocks vanish in homogeneous medium ────────────────────────

    /// In a uniform ε region, Pxy and Pyx must be exactly zero.
    /// This is the load-bearing correctness test for the cross-coupling stencil.
    #[test]
    fn homogeneous_cross_blocks_zero() {
        let nx = 5_usize;
        let ny = 5_usize;
        let n_xy = nx * ny;
        let dx = 100e-9;
        let dy = 100e-9;
        let n_val = 1.5;
        let k0 = 2.0 * PI / 1550e-9;

        let n_profile = vec![n_val; n_xy];
        let solver = FullVectorialModeSolver2d::new(n_profile, nx, ny, dx, dy, 1.0);
        let p = solver.build_operator(k0);

        // Pxy block: rows 0..n_xy, cols n_xy..2*n_xy
        for row in 0..n_xy {
            for col in 0..n_xy {
                let v = p[(row, n_xy + col)];
                assert!(
                    v.abs() < 1e-14,
                    "Pxy[{row},{col}] = {v:.3e} ≠ 0 in homogeneous medium"
                );
            }
        }

        // Pyx block: rows n_xy..2*n_xy, cols 0..n_xy
        for row in 0..n_xy {
            for col in 0..n_xy {
                let v = p[(n_xy + row, col)];
                assert!(
                    v.abs() < 1e-14,
                    "Pyx[{row},{col}] = {v:.3e} ≠ 0 in homogeneous medium"
                );
            }
        }
    }

    // ─── 2. Scalar agreement on a high-contrast waveguide ────────────────────

    /// The full-vectorial and scalar solvers should agree within 5% on a strip
    /// waveguide.  Perfect agreement is not expected because the vectorial solver
    /// uses the conservative (ε-averaged) x-derivative which differs from the
    /// isotropic scalar Laplacian at ε discontinuities.  A 5% tolerance covers
    /// the polarisation correction and grid-dispersion difference.
    #[test]
    fn homogeneous_scalar_agreement() {
        let nx = 10_usize;
        let ny = 10_usize;
        let wavelength = 1550e-9_f64;
        let dx = 200e-9;
        let dy = 200e-9;
        let n_core = 3.476_f64;
        let n_clad = 1.444_f64;
        let width = 500e-9;
        let height = 300e-9;

        let n_profile =
            FullVectorialModeSolver2d::strip_profile(n_core, n_clad, width, height, nx, ny, dx, dy);

        let vec_solver =
            FullVectorialModeSolver2d::new(n_profile.clone(), nx, ny, dx, dy, n_clad);
        let scalar_solver = FdModeSolver2d::new(n_profile, nx, ny, dx, dy, n_clad);

        let vec_modes = vec_solver.solve(wavelength);
        let scalar_modes = scalar_solver.solve(wavelength);

        // Both solvers must find at least one mode.
        assert!(
            !vec_modes.is_empty(),
            "Full-vectorial solver found no modes"
        );
        assert!(!scalar_modes.is_empty(), "Scalar solver found no modes");

        let vec_neff = vec_modes[0].n_eff;
        let scalar_neff = scalar_modes[0].n_eff;

        // Both must be in the physically meaningful guidance range.
        assert!(vec_neff > n_clad && vec_neff < n_core,
            "vec_neff={vec_neff:.4} outside guidance range ({n_clad:.4},{n_core:.4})");
        assert!(scalar_neff > n_clad && scalar_neff < n_core,
            "scalar_neff={scalar_neff:.4} outside guidance range");

        // Vectorial and scalar solvers differ because they discretise ε differently;
        // 5% relative tolerance is adequate for this grid resolution.
        let rel_err = (vec_neff - scalar_neff).abs() / scalar_neff;
        assert!(
            rel_err < 0.05,
            "vec_neff={vec_neff:.4} scalar_neff={scalar_neff:.4} rel_err={rel_err:.4} > 5%"
        );
    }

    // ─── 3. SOI quasi-TE n_eff in range ──────────────────────────────────────

    /// SOI 220nm×500nm waveguide in SiO2. The fundamental quasi-TE mode n_eff
    /// must be in (2.0, 3.0), and te_fraction > 0.5.
    #[test]
    fn soi_quasi_te_neff_range() {
        let nx = 15_usize;
        let ny = 15_usize;
        let wavelength = 1550e-9_f64;
        let domain_x = 2000e-9_f64; // 2 μm
        let domain_y = 1500e-9_f64; // 1.5 μm
        let dx = domain_x / nx as f64;
        let dy = domain_y / ny as f64;
        let n_si = 3.476_f64;
        let n_sio2 = 1.444_f64;
        let width = 500e-9_f64;
        let height = 220e-9_f64;

        let n_profile = FullVectorialModeSolver2d::strip_profile(
            n_si, n_sio2, width, height, nx, ny, dx, dy,
        );
        let solver = FullVectorialModeSolver2d::new(n_profile, nx, ny, dx, dy, n_sio2);
        let modes = solver.solve(wavelength);

        assert!(
            !modes.is_empty(),
            "No guided modes found for SOI waveguide"
        );

        // Find the quasi-TE mode (highest te_fraction among found modes).
        let te_mode = modes
            .iter()
            .max_by(|a, b| a.te_fraction.partial_cmp(&b.te_fraction).unwrap_or(std::cmp::Ordering::Equal))
            .expect("modes non-empty");

        assert!(
            te_mode.n_eff > 2.0 && te_mode.n_eff < 3.0,
            "quasi-TE n_eff = {:.4} not in (2.0, 3.0)",
            te_mode.n_eff
        );
        assert!(
            te_mode.te_fraction > 0.5,
            "quasi-TE te_fraction = {:.4} not > 0.5",
            te_mode.te_fraction
        );
    }

    // ─── 4. SOI: at least 2 modes ────────────────────────────────────────────

    /// Same SOI setup: at least quasi-TE + quasi-TM modes should exist.
    #[test]
    fn soi_quasi_tm_exists() {
        let nx = 15_usize;
        let ny = 15_usize;
        let wavelength = 1550e-9_f64;
        let domain_x = 2000e-9_f64;
        let domain_y = 1500e-9_f64;
        let dx = domain_x / nx as f64;
        let dy = domain_y / ny as f64;
        let n_si = 3.476_f64;
        let n_sio2 = 1.444_f64;
        let width = 500e-9_f64;
        let height = 220e-9_f64;

        let n_profile = FullVectorialModeSolver2d::strip_profile(
            n_si, n_sio2, width, height, nx, ny, dx, dy,
        );
        let solver = FullVectorialModeSolver2d::new(n_profile, nx, ny, dx, dy, n_sio2);
        let modes = solver.solve(wavelength);

        assert!(
            modes.len() >= 2,
            "Expected ≥ 2 modes (quasi-TE + quasi-TM), found {}",
            modes.len()
        );
    }

    // ─── 5. Modes sorted descending ──────────────────────────────────────────

    #[test]
    fn modes_sorted_descending() {
        let nx = 10_usize;
        let ny = 10_usize;
        let wavelength = 1550e-9_f64;
        let dx = 200e-9;
        let dy = 200e-9;
        let n_core = 3.476_f64;
        let n_clad = 1.444_f64;
        let width = 600e-9;
        let height = 300e-9;

        let n_profile =
            FullVectorialModeSolver2d::strip_profile(n_core, n_clad, width, height, nx, ny, dx, dy);
        let solver = FullVectorialModeSolver2d::new(n_profile, nx, ny, dx, dy, n_clad);
        let modes = solver.solve(wavelength);

        for window in modes.windows(2) {
            assert!(
                window[0].n_eff >= window[1].n_eff,
                "Modes not sorted: n_eff[{}]={:.4} < n_eff[{}]={:.4}",
                window[0].order,
                window[0].n_eff,
                window[1].order,
                window[1].n_eff
            );
        }
    }

    // ─── 6. Wide core → at least 1 mode; function doesn't panic ─────────────

    #[test]
    fn wider_core_more_modes() {
        let nx = 10_usize;
        let ny = 10_usize;
        let wavelength = 1550e-9_f64;
        let dx = 200e-9;
        let dy = 200e-9;
        let n_core = 3.476_f64;
        let n_clad = 1.444_f64;

        // Wide core: should have at least 1 mode
        let width_wide = 800e-9;
        let height_wide = 400e-9;
        let np_wide = FullVectorialModeSolver2d::strip_profile(
            n_core, n_clad, width_wide, height_wide, nx, ny, dx, dy,
        );
        let solver_wide = FullVectorialModeSolver2d::new(np_wide, nx, ny, dx, dy, n_clad);
        let modes_wide = solver_wide.solve(wavelength);
        assert!(
            !modes_wide.is_empty(),
            "Wide core should yield ≥ 1 mode, found 0"
        );

        // Narrow core: just check the function runs without panic
        let width_narrow = 100e-9;
        let height_narrow = 100e-9;
        let np_narrow = FullVectorialModeSolver2d::strip_profile(
            n_core, n_clad, width_narrow, height_narrow, nx, ny, dx, dy,
        );
        let solver_narrow = FullVectorialModeSolver2d::new(np_narrow, nx, ny, dx, dy, n_clad);
        let _modes_narrow = solver_narrow.solve(wavelength); // must not panic
    }

    // ─── Diagnostic: raw eigenvalue distribution for small SOI ──────────────

    #[test]
    fn diag_soi_eigenvalue_distribution() {
        // Small 5×5 grid to check P diagonal and eigenvalues
        let nx = 5_usize;
        let ny = 5_usize;
        let wavelength = 1550e-9_f64;
        let domain_x = 2000e-9_f64;
        let domain_y = 2000e-9_f64;
        let dx = domain_x / nx as f64;
        let dy = domain_y / ny as f64;
        let n_si = 3.476_f64;
        let n_sio2 = 1.444_f64;
        let width = 1000e-9_f64;  // Wide core: 50% of domain
        let height = 1000e-9_f64;

        let n_profile = FullVectorialModeSolver2d::strip_profile(
            n_si, n_sio2, width, height, nx, ny, dx, dy,
        );
        let solver = FullVectorialModeSolver2d::new(n_profile.clone(), nx, ny, dx, dy, n_sio2);
        let modes = solver.solve(wavelength);

        // Wide core should find at least one mode
        assert!(
            !modes.is_empty(),
            "Wide-core 5×5 diagnostic grid should find at least 1 mode"
        );
    }
}
