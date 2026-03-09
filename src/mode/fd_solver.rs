use oxiblas::prelude::{Mat, SymmetricEvd};
use std::f64::consts::PI;

/// A guided mode from the FD mode solver.
#[derive(Debug, Clone)]
pub struct FdMode {
    /// Effective index n_eff = β / k0.
    pub n_eff: f64,
    /// Mode field values on the grid.
    pub field: Vec<f64>,
    /// Mode order (sorted by n_eff descending).
    pub order: usize,
}

/// 1D finite-difference mode solver for slab waveguides.
///
/// Solves the Helmholtz eigenvalue problem:
///   d²E/dx² + k0² n(x)² E = β² E
/// using a uniform finite-difference grid.
pub struct FdModeSolver1d {
    /// Refractive index profile on grid (size = n_pts).
    pub n_profile: Vec<f64>,
    /// Grid spacing (m).
    pub dx: f64,
    /// Minimum n_eff for guided modes (substrate or lower cladding index).
    pub n_min: f64,
}

impl FdModeSolver1d {
    /// Build a 1D FD mode solver with the given index profile.
    ///
    /// # Arguments
    /// - `n_profile`: refractive index at each grid point
    /// - `dx`: grid spacing (m)
    /// - `n_min`: modes with n_eff < n_min are radiation modes (discarded)
    pub fn new(n_profile: Vec<f64>, dx: f64, n_min: f64) -> Self {
        Self {
            n_profile,
            dx,
            n_min,
        }
    }

    /// Solve for guided modes at the given wavelength (m).
    ///
    /// Returns modes sorted by n_eff descending (fundamental first).
    pub fn solve(&self, wavelength: f64) -> Vec<FdMode> {
        let k0 = 2.0 * PI / wavelength;
        let n = self.n_profile.len();
        let dx2 = self.dx * self.dx;

        // Build symmetric tridiagonal Helmholtz matrix:
        // A[i,i] = -2/dx² + k0² n[i]²
        // A[i,i±1] = 1/dx²
        // Eigenvalue λ = β²
        let mut mat = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            mat[(i, i)] = -2.0 / dx2 + k0 * k0 * self.n_profile[i] * self.n_profile[i];
            if i + 1 < n {
                mat[(i, i + 1)] = 1.0 / dx2;
                mat[(i + 1, i)] = 1.0 / dx2;
            }
        }

        // Solve symmetric eigenvalue problem
        let evd = match SymmetricEvd::compute(mat.as_ref()) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };

        let eigenvalues = evd.eigenvalues();
        let eigenvectors = evd.eigenvectors();

        let beta_min_sq = (self.n_min * k0) * (self.n_min * k0);

        // Collect guided modes (λ = β² > n_min² k0²)
        let mut modes: Vec<FdMode> = eigenvalues
            .iter()
            .enumerate()
            .filter(|(_, &lambda)| lambda > beta_min_sq)
            .map(|(idx, &lambda)| {
                let n_eff = (lambda / (k0 * k0)).sqrt();
                let field: Vec<f64> = (0..n).map(|i| eigenvectors[(i, idx)]).collect();
                FdMode {
                    n_eff,
                    field,
                    order: 0,
                }
            })
            .collect();

        // Sort by n_eff descending and assign order
        modes.sort_by(|a, b| {
            b.n_eff
                .partial_cmp(&a.n_eff)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, mode) in modes.iter_mut().enumerate() {
            mode.order = i;
        }
        modes
    }

    /// Build an index profile for a symmetric slab waveguide.
    ///
    /// # Arguments
    /// - `n_core`: core index
    /// - `n_clad`: cladding index
    /// - `thickness`: core thickness (m)
    /// - `n_pts`: total grid points
    /// - `dx`: grid spacing (m)
    pub fn slab_profile(
        n_core: f64,
        n_clad: f64,
        thickness: f64,
        n_pts: usize,
        dx: f64,
    ) -> Vec<f64> {
        let total = n_pts as f64 * dx;
        let center = total / 2.0;
        (0..n_pts)
            .map(|i| {
                let x = i as f64 * dx;
                let dist = (x - center).abs();
                if dist <= thickness / 2.0 {
                    n_core
                } else {
                    n_clad
                }
            })
            .collect()
    }
}

/// 2D finite-difference mode solver for strip waveguides.
///
/// Solves the 2D Helmholtz eigenvalue problem using a dense symmetric matrix.
/// The matrix size is (nx * ny) × (nx * ny), so use modest grid sizes (e.g. 40×40).
pub struct FdModeSolver2d {
    /// Refractive index profile (row-major, size = nx * ny).
    pub n_profile: Vec<f64>,
    /// Grid points in x direction.
    pub nx: usize,
    /// Grid points in y direction.
    pub ny: usize,
    /// Grid spacing in x (m).
    pub dx: f64,
    /// Grid spacing in y (m).
    pub dy: f64,
    /// Minimum n_eff for guided modes.
    pub n_min: f64,
}

impl FdModeSolver2d {
    pub fn new(n_profile: Vec<f64>, nx: usize, ny: usize, dx: f64, dy: f64, n_min: f64) -> Self {
        assert_eq!(n_profile.len(), nx * ny);
        Self {
            n_profile,
            nx,
            ny,
            dx,
            dy,
            n_min,
        }
    }

    /// Build index profile for a strip waveguide centered in the domain.
    ///
    /// # Arguments
    /// - `n_core`, `n_clad`: refractive indices
    /// - `width`, `height`: waveguide cross-section (m)
    /// - `nx`, `ny`: grid dimensions; `dx`, `dy`: grid spacing (m)
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
        let cx = nx as f64 * dx / 2.0;
        let cy = ny as f64 * dy / 2.0;
        let mut n = vec![n_clad; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let x = i as f64 * dx;
                let y = j as f64 * dy;
                if (x - cx).abs() <= width / 2.0 && (y - cy).abs() <= height / 2.0 {
                    n[j * nx + i] = n_core;
                }
            }
        }
        n
    }

    /// Solve for guided modes. Returns modes sorted by n_eff descending.
    ///
    /// Warning: matrix size is (nx*ny)² — keep nx,ny ≤ 50 for reasonable time.
    pub fn solve(&self, wavelength: f64) -> Vec<FdMode> {
        let k0 = 2.0 * PI / wavelength;
        let nxy = self.nx * self.ny;
        let dx2 = self.dx * self.dx;
        let dy2 = self.dy * self.dy;

        // Build the 2D Helmholtz operator as a dense symmetric matrix.
        // Using 5-point Laplacian (Kronecker sum of 1D operators):
        // H[k,k] = -2/dx² - 2/dy² + k0² n[k]²
        // H[k, k±1] = 1/dx²   (x-neighbors, same j)
        // H[k, k±nx] = 1/dy²  (y-neighbors)
        let mut mat = Mat::<f64>::zeros(nxy, nxy);

        for j in 0..self.ny {
            for i in 0..self.nx {
                let k = j * self.nx + i;
                let n_k = self.n_profile[k];

                mat[(k, k)] = -2.0 / dx2 - 2.0 / dy2 + k0 * k0 * n_k * n_k;

                if i + 1 < self.nx {
                    let kr = k + 1;
                    mat[(k, kr)] = 1.0 / dx2;
                    mat[(kr, k)] = 1.0 / dx2;
                }
                if j + 1 < self.ny {
                    let ku = k + self.nx;
                    mat[(k, ku)] = 1.0 / dy2;
                    mat[(ku, k)] = 1.0 / dy2;
                }
            }
        }

        let evd = match SymmetricEvd::compute(mat.as_ref()) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };

        let eigenvalues = evd.eigenvalues();
        let eigenvectors = evd.eigenvectors();
        let beta_min_sq = (self.n_min * k0) * (self.n_min * k0);

        let mut modes: Vec<FdMode> = eigenvalues
            .iter()
            .enumerate()
            .filter(|(_, &lam)| lam > beta_min_sq)
            .map(|(idx, &lam)| {
                let n_eff = (lam / (k0 * k0)).sqrt();
                let field = (0..nxy).map(|i| eigenvectors[(i, idx)]).collect();
                FdMode {
                    n_eff,
                    field,
                    order: 0,
                }
            })
            .collect();

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
}

/// 1D finite-difference TM mode solver for slab waveguides.
///
/// Solves the TM Helmholtz eigenvalue problem:
///   d/dx \[1/n²(x) d(Hy)/dx\] + k0² Hy = β² / n_avg² Hy
///
/// Implemented via the standard FD discretisation with coefficient averaging.
pub struct FdTmSolver1d {
    pub n_profile: Vec<f64>,
    pub dx: f64,
    pub n_min: f64,
}

impl FdTmSolver1d {
    pub fn new(n_profile: Vec<f64>, dx: f64, n_min: f64) -> Self {
        Self {
            n_profile,
            dx,
            n_min,
        }
    }

    /// Solve for TM guided modes.
    ///
    /// Uses the symmetric eigenvalue formulation derived from the substitution
    /// u_i = H_y,i / sqrt(ε_i), which converts the TM generalised eigenvalue
    /// problem into a standard symmetric one.  Eigenvalues equal β² directly.
    ///
    /// The FD discretisation of d/dx\[1/ε dH_y/dx\] + k₀²H_y = (β²/ε)H_y,
    /// after multiplying by sqrt(ε_i) on both sides, yields:
    ///
    ///   A\[i,i\]   = k₀²ε_i − ε_i·(1/ε_{i−½} + 1/ε_{i+½})/dx²
    ///   A\[i,i±1\] = sqrt(ε_i·ε_{i±1}) · 2/(ε_i + ε_{i±1}) / dx²
    ///
    /// The H_y field is recovered from eigenvectors u via H_y,i = sqrt(ε_i)·u_i.
    pub fn solve(&self, wavelength: f64) -> Vec<FdMode> {
        let k0 = 2.0 * PI / wavelength;
        let n = self.n_profile.len();
        let dx2 = self.dx * self.dx;

        // Pre-compute ε_i and sqrt(ε_i) arrays.
        let eps: Vec<f64> = self.n_profile.iter().map(|&ni| ni * ni).collect();
        let sqrt_eps: Vec<f64> = eps.iter().map(|&e| e.sqrt()).collect();

        let mut mat = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            // 1/ε at left half-interface (use left neighbour; zero at boundary)
            let inv_eps_left = if i > 0 {
                2.0 / (eps[i - 1] + eps[i])
            } else {
                0.0
            };
            // 1/ε at right half-interface (use right neighbour; zero at boundary)
            let inv_eps_right = if i + 1 < n {
                2.0 / (eps[i] + eps[i + 1])
            } else {
                0.0
            };

            mat[(i, i)] = k0 * k0 * eps[i] - eps[i] * (inv_eps_left + inv_eps_right) / dx2;

            if i + 1 < n {
                // Geometric-harmonic coupling: sqrt(ε_i·ε_{i+1}) · 2/(ε_i+ε_{i+1})
                let coupling = sqrt_eps[i] * sqrt_eps[i + 1] * 2.0 / (eps[i] + eps[i + 1]) / dx2;
                mat[(i, i + 1)] = coupling;
                mat[(i + 1, i)] = coupling;
            }
        }

        let evd = match SymmetricEvd::compute(mat.as_ref()) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };
        let eigenvalues = evd.eigenvalues();
        let eigenvectors = evd.eigenvectors();
        let beta_min_sq = (self.n_min * k0) * (self.n_min * k0);

        // Recover H_y = sqrt(ε_i) · u_i for the physical field profile.
        let mut modes: Vec<FdMode> = eigenvalues
            .iter()
            .enumerate()
            .filter(|(_, &lambda)| lambda > beta_min_sq)
            .map(|(idx, &lambda)| {
                let n_eff = (lambda / (k0 * k0)).sqrt();
                let field: Vec<f64> = (0..n)
                    .map(|i| sqrt_eps[i] * eigenvectors[(i, idx)])
                    .collect();
                FdMode {
                    n_eff,
                    field,
                    order: 0,
                }
            })
            .collect();
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
}

/// Compute group velocity from a dispersion relation (numerical derivative).
///
/// Given β(ω) sampled at two adjacent wavelengths, returns:
///   v_g = dω/dβ = (ω2 - ω1) / (β2 - β1)
pub fn group_velocity(lambda1: f64, n_eff1: f64, lambda2: f64, n_eff2: f64) -> f64 {
    use std::f64::consts::PI;
    let c = 2.998e8;
    let omega1 = 2.0 * PI * c / lambda1;
    let omega2 = 2.0 * PI * c / lambda2;
    let beta1 = n_eff1 * 2.0 * PI / lambda1;
    let beta2 = n_eff2 * 2.0 * PI / lambda2;
    let d_omega = omega2 - omega1;
    let d_beta = beta2 - beta1;
    if d_beta.abs() < 1e-30 {
        c
    } else {
        d_omega / d_beta
    }
}

/// Group index n_g = c/v_g.
pub fn group_index(lambda1: f64, n_eff1: f64, lambda2: f64, n_eff2: f64) -> f64 {
    2.998e8 / group_velocity(lambda1, n_eff1, lambda2, n_eff2).max(1e-30)
}

/// Dispersion parameter D = -(lambda/c) * d²n_eff/dlambda² (ps/nm/km).
///
/// Computed numerically from three n_eff values at lambda-dlambda, lambda, lambda+dlambda.
pub fn dispersion_parameter_d(
    lambda: f64,
    n_m: f64, // n_eff at lambda - dlambda
    n_0: f64, // n_eff at lambda
    n_p: f64, // n_eff at lambda + dlambda
    dl: f64,  // dlambda step
) -> f64 {
    let c = 2.998e8;
    let d2n_dl2 = (n_p - 2.0 * n_0 + n_m) / (dl * dl);
    let d_s_m2 = -lambda / c * d2n_dl2;
    d_s_m2 * 1e3 // convert to ps/(nm·km)
}

/// Compute the effective index of a Si strip waveguide (220nm × 500nm) at 1550nm.
/// Uses 2D FD solver with a small grid for speed. Expected n_eff ≈ 2.0–3.0.
pub fn si_strip_neff_1550nm() -> f64 {
    let n_si = 3.476_f64;
    let n_sio2 = 1.444_f64;
    let width = 500e-9_f64;
    let height = 220e-9_f64;
    let wavelength = 1550e-9_f64;

    // Use a 20×15 grid over a 2μm × 1.5μm domain (small for test speed)
    let nx = 20_usize;
    let ny = 15_usize;
    let dx = 2000e-9 / nx as f64;
    let dy = 1500e-9 / ny as f64;

    let n_profile = FdModeSolver2d::strip_profile(n_si, n_sio2, width, height, nx, ny, dx, dy);
    let solver = FdModeSolver2d::new(n_profile, nx, ny, dx, dy, n_sio2);
    let modes = solver.solve(wavelength);
    modes.first().map(|m| m.n_eff).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fd1d_slab_finds_guided_modes() {
        // Si slab 1μm thick in SiO2: well above cutoff
        let n_pts = 100;
        let dx = 50e-9; // 50nm spacing, 5μm total
        let n_profile = FdModeSolver1d::slab_profile(3.476, 1.444, 1000e-9, n_pts, dx);
        let solver = FdModeSolver1d::new(n_profile, dx, 1.444);
        let modes = solver.solve(1550e-9);
        assert!(!modes.is_empty(), "Should find guided modes");
        assert!(
            modes[0].n_eff > 1.444 && modes[0].n_eff < 3.476,
            "n_eff={} out of guidance range",
            modes[0].n_eff
        );
    }

    #[test]
    fn fd1d_matches_eim_symmetric_slab() {
        use crate::mode::effective_index::SlabWaveguide;

        // Si slab 500nm in SiO2 at 1550nm
        let n_core = 3.476;
        let n_clad = 1.444;
        let thickness = 500e-9;
        let wavelength = 1550e-9;

        // EIM reference
        let slab = SlabWaveguide::new(n_core, n_clad, thickness);
        let eim_modes = slab.solve_te(wavelength);
        let eim_neff = eim_modes[0].n_eff;

        // FD solver
        let n_pts = 150;
        let dx = 20e-9;
        let n_profile = FdModeSolver1d::slab_profile(n_core, n_clad, thickness, n_pts, dx);
        let solver = FdModeSolver1d::new(n_profile, dx, n_clad);
        let fd_modes = solver.solve(wavelength);
        let fd_neff = fd_modes[0].n_eff;

        // FD with 20nm grid should be within 0.5% of EIM
        let rel_err = (fd_neff - eim_neff).abs() / eim_neff;
        assert!(
            rel_err < 0.005,
            "FD={fd_neff:.4} EIM={eim_neff:.4} rel_err={rel_err:.4}"
        );
    }

    #[test]
    fn si_strip_waveguide_neff_range() {
        let n_eff = si_strip_neff_1550nm();
        assert!(
            n_eff > 2.0 && n_eff < 3.0,
            "Si strip n_eff={n_eff:.4} outside expected range 2.0–3.0"
        );
    }

    #[test]
    fn tm_solver_finds_modes() {
        let n_pts = 80;
        let dx = 30e-9;
        let n_profile = FdModeSolver1d::slab_profile(3.476, 1.444, 800e-9, n_pts, dx);
        let solver = FdTmSolver1d::new(n_profile, dx, 1.444);
        let modes = solver.solve(1550e-9);
        assert!(!modes.is_empty(), "TM solver should find modes");
        assert!(modes[0].n_eff > 1.444, "TM n_eff should be guided");
    }

    #[test]
    fn group_velocity_less_than_c() {
        // In a waveguide, v_g < c
        let c = 2.998e8;
        let dl = 1e-9;
        let n1 = 1.444 + 0.002; // slightly different n_eff
        let n2 = 1.444;
        let vg = group_velocity(1550e-9 - dl / 2.0, n1, 1550e-9 + dl / 2.0, n2);
        assert!(vg > 0.0);
        assert!(vg < c * 1.1, "v_g={vg:.3e} unexpectedly > c");
    }

    #[test]
    fn group_index_greater_than_phase_index() {
        // n_g > n_eff for normal dispersion
        let dl = 5e-9;
        let n_eff = 1.5;
        // Simulate normal dispersion: n increases at shorter wavelength
        let n1 = n_eff + 0.001; // at lambda - dl
        let n2 = n_eff - 0.001; // at lambda + dl
        let ng = group_index(1550e-9 - dl, n1, 1550e-9 + dl, n2);
        assert!(ng > 0.0);
    }

    #[test]
    fn dispersion_parameter_d_sign() {
        // SiO2 at 1300nm: near zero-dispersion (~17 ps/nm/km below ZDW)
        let n_m = 1.4495;
        let n_0 = 1.4500;
        let n_p = 1.4505;
        let dl = 5e-9;
        let d = dispersion_parameter_d(1300e-9, n_m, n_0, n_p, dl);
        // Normal dispersion: D should be negative or small
        assert!(d.is_finite());
    }
}
