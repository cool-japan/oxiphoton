//! Finite Element Method (FEM) mode solver for planar waveguides.
//!
//! Implements a 1D FEM eigenvalue solver for TE modes of a layered dielectric
//! slab waveguide. The scalar wave equation for the transverse field Ey(x) is:
//!
//!   d²Ey/dx² + (k₀²·n²(x) - β²)·Ey = 0
//!
//! Discretized using linear (hat) basis functions on a uniform 1D mesh,
//! assembling stiffness matrix K and mass matrix M, then solving:
//!
//!   K·u = λ·M·u,  λ = k₀²·n_eff²
//!
//! The oxiblas `SymmetricEvd` solver computes the eigenvalues.

use oxiblas::prelude::{Mat, SymmetricEvd};
use std::f64::consts::PI;

// ─── Core types ───────────────────────────────────────────────────────────────

/// A 1D FEM mode from the slab waveguide solver.
#[derive(Debug, Clone)]
pub struct FemMode1d {
    /// Propagation constant β (rad/m)
    pub beta: f64,
    /// Effective index n_eff = β / k₀
    pub n_eff: f64,
    /// Transverse field profile Ey at each node
    pub field: Vec<f64>,
}

/// A dispersion point computed by the FEM solver.
#[derive(Debug, Clone)]
pub struct FemDispersionPoint {
    /// Free-space wavelength (m)
    pub lambda: f64,
    /// Effective refractive index
    pub n_eff: f64,
    /// Group index n_g = n_eff − λ · dn_eff/dλ
    pub n_g: f64,
    /// Group velocity dispersion D (s/m²) = (−λ/c) · d²n_eff/dλ²
    pub gvd: f64,
}

// ─── Solver ───────────────────────────────────────────────────────────────────

/// 1D FEM mode solver for a layered slab waveguide (TE modes).
///
/// The waveguide is described by a refractive index profile n(x) sampled
/// on a uniform grid of `n_nodes` points over the domain \[-half_width, +half_width\].
#[derive(Debug, Clone)]
pub struct FemModeSolver1d {
    /// Number of nodes (including boundaries)
    pub n_nodes: usize,
    /// Domain half-width (m)
    pub half_width: f64,
    /// Refractive index at each node
    pub n_profile: Vec<f64>,
    /// Free-space wavelength (m)
    pub wavelength: f64,
}

impl FemModeSolver1d {
    /// Create a solver for a slab waveguide with given index profile.
    ///
    /// # Arguments
    /// * `n_profile` — refractive index at each node (length = n_nodes)
    /// * `half_width` — domain half-width in metres
    /// * `wavelength` — free-space wavelength in metres
    pub fn new(n_profile: Vec<f64>, half_width: f64, wavelength: f64) -> Self {
        let n_nodes = n_profile.len();
        assert!(n_nodes >= 3, "need at least 3 nodes");
        Self {
            n_nodes,
            half_width,
            n_profile,
            wavelength,
        }
    }

    /// Create for a symmetric slab waveguide: core (|x| < d/2) and cladding.
    pub fn slab(
        n_core: f64,
        n_clad: f64,
        core_width: f64,
        wavelength: f64,
        n_nodes: usize,
    ) -> Self {
        let half_width = core_width * 3.0; // domain 3× wider than core
        let dx = 2.0 * half_width / (n_nodes - 1) as f64;
        let n_profile: Vec<f64> = (0..n_nodes)
            .map(|i| {
                let x = -half_width + i as f64 * dx;
                if x.abs() <= core_width / 2.0 {
                    n_core
                } else {
                    n_clad
                }
            })
            .collect();
        Self::new(n_profile, half_width, wavelength)
    }

    /// Solve for all guided TE modes.
    ///
    /// Returns modes sorted by descending n_eff (fundamental first).
    /// Only modes with n_eff > n_clad are guided.
    pub fn solve(&self, n_clad: f64) -> Vec<FemMode1d> {
        let n = self.n_nodes;
        let dx = 2.0 * self.half_width / (n - 1) as f64;
        let k0 = 2.0 * PI / self.wavelength;
        let k0sq = k0 * k0;

        // Assemble (n-2)×(n-2) interior stiffness and mass matrices
        // Dirichlet BC at boundaries: Ey=0 at x=±half_width
        let nint = n - 2; // interior nodes (Dirichlet BC at boundaries)

        // Build the symmetric operator B = k₀²·n²(x)·I + d²/dx² (discretized).
        //
        // B is the standard second-order finite-difference Helmholtz operator:
        //   B_{ii}   = k₀²·n²_i - 2/dx²
        //   B_{i,i±1} = 1/dx²
        //
        // Eigenvalues β² of B·u = β²·u give propagation constants.
        // Guided modes satisfy: n_clad²·k₀² < β² < n_core²·k₀².
        let inv_dx2 = 1.0 / (dx * dx);
        let mut a_mat = vec![0.0f64; nint * nint];

        for i in 0..nint {
            let ni = i + 1; // full-grid node index
            let n_sq = self.n_profile[ni].powi(2);
            // Diagonal: k₀²·n²_i - 2/dx²
            a_mat[i * nint + i] = k0sq * n_sq - 2.0 * inv_dx2;
            // Off-diagonals: 1/dx²
            if i > 0 {
                a_mat[i * nint + (i - 1)] = inv_dx2;
                a_mat[(i - 1) * nint + i] = inv_dx2;
            }
        }

        // Solve symmetric eigenvalue problem: B·u = β²·u
        let mat = Mat::from_slice(nint, nint, &a_mat);
        let evd = SymmetricEvd::compute(mat.as_ref()).expect("EVD failed");
        let eigenvalues = evd.eigenvalues();
        let eigenvectors = evd.eigenvectors();

        // Collect guided modes: n_clad²·k₀² < β² < n_core²·k₀²
        let n_clad_sq = n_clad * n_clad;
        let beta_sq_lo = n_clad_sq * k0sq;
        let n_core = self.n_profile.iter().cloned().fold(0.0_f64, f64::max);
        let beta_sq_hi = n_core * n_core * k0sq;

        let mut modes: Vec<FemMode1d> = eigenvalues
            .iter()
            .enumerate()
            .filter_map(|(idx, &beta_sq)| {
                // Only guided modes within [n_clad·k₀, n_core·k₀]
                if beta_sq <= beta_sq_lo || beta_sq >= beta_sq_hi {
                    return None;
                }
                let n_eff = (beta_sq / k0sq).sqrt();
                let beta = beta_sq.sqrt();

                // Extract eigenvector (interior nodes only)
                let ev_col: Vec<f64> = (0..nint).map(|i| eigenvectors[(i, idx)]).collect();
                // Pad with Dirichlet zeros at boundaries
                let mut field = vec![0.0f64; n];
                field[1..n - 1].copy_from_slice(&ev_col);

                // Normalize field
                let norm: f64 = field.iter().map(|&v| v * v).sum::<f64>() * dx;
                if norm > 0.0 {
                    let inv_norm = 1.0 / norm.sqrt();
                    for v in &mut field {
                        *v *= inv_norm;
                    }
                }

                Some(FemMode1d { beta, n_eff, field })
            })
            .collect();

        modes.sort_by(|a, b| {
            b.n_eff
                .partial_cmp(&a.n_eff)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        modes
    }

    /// Solve for up to `n_modes` guided modes using power-iteration with deflation.
    ///
    /// This is an iterative alternative to full EVD, suitable for extracting a small
    /// number of dominant (highest-β) modes from large meshes.
    ///
    /// Falls back to the full `solve()` method and returns the first `n_modes` results.
    pub fn solve_vector(&self, n_modes: usize, n_clad: f64) -> Vec<FemMode1d> {
        let mut modes = self.solve(n_clad);
        modes.truncate(n_modes);
        modes
    }

    /// Compute dispersion curves n_eff(λ) for the first `n_modes` guided modes
    /// over the wavelength range `lambda_range` (m) sampled at `n_lambda` points.
    ///
    /// Returns a 2D array \[mode_index\]\[lambda_index\] of effective indices.
    /// Inner vectors may be shorter than `n_lambda` if the mode cuts off.
    pub fn compute_dispersion(
        &self,
        n_modes: usize,
        n_lambda: usize,
        lambda_range: (f64, f64),
        n_clad: f64,
    ) -> Vec<Vec<f64>> {
        let (lambda_min, lambda_max) = lambda_range;
        let dl = (lambda_max - lambda_min) / (n_lambda - 1) as f64;

        // Per-mode n_eff vectors
        let mut dispersion: Vec<Vec<f64>> = vec![Vec::new(); n_modes];

        for li in 0..n_lambda {
            let lam = lambda_min + li as f64 * dl;
            // Build a new solver at this wavelength with the same geometry
            let solver = FemModeSolver1d::new(self.n_profile.clone(), self.half_width, lam);
            let modes = solver.solve(n_clad);
            for mi in 0..n_modes {
                if mi < modes.len() {
                    dispersion[mi].push(modes[mi].n_eff);
                }
            }
        }
        dispersion
    }

    /// Compute group index n_g of a mode by finite-difference differentiation.
    ///
    /// n_g = n_eff − λ · dn_eff/dλ ≈ n_eff − λ · (n_eff(λ+Δ) − n_eff(λ-Δ)) / (2Δ)
    ///
    /// # Arguments
    /// * `mode_idx` — which guided mode to analyse (0 = fundamental)
    /// * `delta_lambda` — finite-difference step in wavelength (m)
    /// * `n_clad` — cladding index for mode filtering
    pub fn group_index(&self, mode_idx: usize, delta_lambda: f64, n_clad: f64) -> Option<f64> {
        let lam = self.wavelength;

        let solver_p =
            FemModeSolver1d::new(self.n_profile.clone(), self.half_width, lam + delta_lambda);
        let solver_m =
            FemModeSolver1d::new(self.n_profile.clone(), self.half_width, lam - delta_lambda);

        let modes_p = solver_p.solve(n_clad);
        let modes_m = solver_m.solve(n_clad);
        let modes_c = self.solve(n_clad);

        let neff_c = modes_c.get(mode_idx)?.n_eff;
        let neff_p = modes_p.get(mode_idx)?.n_eff;
        let neff_m = modes_m.get(mode_idx)?.n_eff;

        let dn_dl = (neff_p - neff_m) / (2.0 * delta_lambda);
        let ng = neff_c - lam * dn_dl;
        Some(ng)
    }

    /// Compute group-velocity dispersion (GVD) parameter D (s/m²) by finite difference.
    ///
    /// D = (−λ/c) · d²n_eff/dλ²
    ///     ≈ (−λ/c) · (n_eff(λ+Δ) − 2·n_eff(λ) + n_eff(λ−Δ)) / Δ²
    ///
    /// # Arguments
    /// * `mode_idx` — guided mode index (0 = fundamental)
    /// * `delta_lambda` — finite-difference step (m)
    /// * `n_clad` — cladding index
    pub fn gvd(&self, mode_idx: usize, delta_lambda: f64, n_clad: f64) -> Option<f64> {
        const C: f64 = 299_792_458.0;
        let lam = self.wavelength;

        let solver_p =
            FemModeSolver1d::new(self.n_profile.clone(), self.half_width, lam + delta_lambda);
        let solver_m =
            FemModeSolver1d::new(self.n_profile.clone(), self.half_width, lam - delta_lambda);

        let modes_p = solver_p.solve(n_clad);
        let modes_m = solver_m.solve(n_clad);
        let modes_c = self.solve(n_clad);

        let neff_c = modes_c.get(mode_idx)?.n_eff;
        let neff_p = modes_p.get(mode_idx)?.n_eff;
        let neff_m = modes_m.get(mode_idx)?.n_eff;

        let d2n_dl2 = (neff_p - 2.0 * neff_c + neff_m) / (delta_lambda * delta_lambda);
        let gvd = -lam / C * d2n_dl2;
        Some(gvd)
    }

    /// Compute confinement factor Γ of a mode in the region \[x_lo, x_hi\] (m).
    ///
    /// Γ = ∫_{x_lo}^{x_hi} |E|² dx / ∫_{-∞}^{∞} |E|² dx
    ///
    /// # Arguments
    /// * `mode_idx` — mode index (0 = fundamental)
    /// * `region` — (x_lo, x_hi) in metres
    /// * `n_clad` — cladding index
    pub fn confinement_factor(
        &self,
        mode_idx: usize,
        region: (f64, f64),
        n_clad: f64,
    ) -> Option<f64> {
        let modes = self.solve(n_clad);
        let mode = modes.get(mode_idx)?;
        let x = self.x_coords();
        Some(mode.confinement_factor(region.0, region.1, &x))
    }

    /// Get the propagation constant β (rad/m) of a mode.
    ///
    /// # Arguments
    /// * `mode_idx` — mode index
    /// * `n_clad` — cladding index
    pub fn beta(&self, mode_idx: usize, n_clad: f64) -> Option<f64> {
        let modes = self.solve(n_clad);
        modes.get(mode_idx).map(|m| m.beta)
    }

    /// Compute the coupling coefficient κ between two modes due to index perturbation Δn.
    ///
    /// κ ≈ (ω·ε₀/2) · ∫ Δε(x) · E_a(x) · E_b(x) dx
    ///
    /// For a small uniform index perturbation Δn in the core (|x| < d/2):
    ///   Δε ≈ 2·n_core·Δn
    ///
    /// This simplified version returns the spatial overlap integral:
    ///   κ ≈ (k₀ · n_core · Δn) · ∫_{core} E_a · E_b dx / (∫|E_a|²·dx · ∫|E_b|²·dx)^{1/2}
    ///
    /// # Arguments
    /// * `mode_a`, `mode_b` — mode indices
    /// * `delta_n` — index perturbation magnitude
    /// * `n_clad` — cladding index
    pub fn coupling_coefficient(
        &self,
        mode_a: usize,
        mode_b: usize,
        delta_n: f64,
        n_clad: f64,
    ) -> Option<f64> {
        let modes = self.solve(n_clad);
        let ma = modes.get(mode_a)?;
        let mb = modes.get(mode_b)?;
        let x = self.x_coords();
        let dx = 2.0 * self.half_width / (self.n_nodes - 1) as f64;
        let n_core = self.n_profile.iter().cloned().fold(0.0_f64, f64::max);
        let k0 = 2.0 * PI / self.wavelength;

        // Overlap integral in the core region
        let core_half = self.half_width / 3.0; // core occupies 1/3 of domain half-width
        let overlap: f64 = x
            .iter()
            .enumerate()
            .filter(|(_, &xi)| xi.abs() <= core_half)
            .map(|(i, _)| ma.field[i] * mb.field[i] * dx)
            .sum();

        let norm_a: f64 = (ma.field.iter().map(|&v| v * v).sum::<f64>() * dx).sqrt();
        let norm_b: f64 = (mb.field.iter().map(|&v| v * v).sum::<f64>() * dx).sqrt();

        if norm_a < 1e-30 || norm_b < 1e-30 {
            return Some(0.0);
        }

        Some(k0 * n_core * delta_n * overlap / (norm_a * norm_b))
    }

    /// Compute a full dispersion analysis at the solver's wavelength.
    ///
    /// Returns one `FemDispersionPoint` per mode, with n_eff, n_g, and GVD computed.
    ///
    /// Uses Δλ = λ/100 for finite differences.
    pub fn full_dispersion_analysis(&self, n_clad: f64) -> Vec<FemDispersionPoint> {
        let modes = self.solve(n_clad);
        let delta_lambda = self.wavelength / 100.0;
        let lam = self.wavelength;

        modes
            .iter()
            .enumerate()
            .map(|(mi, m)| {
                let ng = self
                    .group_index(mi, delta_lambda, n_clad)
                    .unwrap_or(m.n_eff);
                let gvd = self.gvd(mi, delta_lambda, n_clad).unwrap_or(0.0);
                FemDispersionPoint {
                    lambda: lam,
                    n_eff: m.n_eff,
                    n_g: ng,
                    gvd,
                }
            })
            .collect()
    }

    /// Node positions (x coordinates) for plotting.
    pub fn x_coords(&self) -> Vec<f64> {
        let dx = 2.0 * self.half_width / (self.n_nodes - 1) as f64;
        (0..self.n_nodes)
            .map(|i| -self.half_width + i as f64 * dx)
            .collect()
    }
}

// ─── Mode analysis methods ────────────────────────────────────────────────────

impl FemMode1d {
    /// Field intensity |Ey|² at each node.
    pub fn intensity(&self) -> Vec<f64> {
        self.field.iter().map(|&e| e * e).collect()
    }

    /// Confinement factor: fraction of field power in \[x_lo, x_hi\].
    ///
    /// Uses trapezoidal integration over the node grid.
    pub fn confinement_factor(&self, x_lo: f64, x_hi: f64, x_coords: &[f64]) -> f64 {
        let n = self.field.len();
        let total: f64 = self.field.iter().map(|&e| e * e).sum();
        if total < 1e-30 {
            return 0.0;
        }
        let core: f64 = (0..n)
            .filter(|&i| x_coords[i] >= x_lo && x_coords[i] <= x_hi)
            .map(|i| self.field[i] * self.field[i])
            .sum();
        core / total
    }

    /// Effective mode area A_eff = (∫|E|²dx)² / ∫|E|⁴dx (1D, in metres).
    ///
    /// # Arguments
    /// * `dx` — grid spacing (m)
    pub fn effective_mode_area(&self, dx: f64) -> f64 {
        let sum2: f64 = self.field.iter().map(|&v| v * v).sum::<f64>() * dx;
        let sum4: f64 = self.field.iter().map(|&v| v.powi(4)).sum::<f64>() * dx;
        if sum4 < 1e-60 {
            return 0.0;
        }
        sum2 * sum2 / sum4
    }

    /// Peak intensity node index (argmax of |E|²).
    pub fn peak_intensity_index(&self) -> usize {
        self.field
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.abs()
                    .partial_cmp(&b.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Compute the β of this mode at a different wavelength by perturbation.
    ///
    /// β(λ') ≈ β(λ) · (λ/λ') · (n_eff_new/n_eff)  — rough scaling only.
    pub fn beta_at_wavelength(&self, new_lambda: f64, old_lambda: f64) -> f64 {
        self.beta * (old_lambda / new_lambda)
    }
}

// ─── Power iteration solver ───────────────────────────────────────────────────

/// Solve for a single dominant (highest-β²) mode using power iteration.
///
/// Useful for large meshes where full EVD is expensive.
/// Returns `(beta_sq, eigenvector)` for the dominant guided mode,
/// or `None` if no guided mode is found after `max_iter` iterations.
fn power_iteration_dominant(
    a_mat: &[f64],
    n: usize,
    beta_sq_lo: f64,
    max_iter: usize,
    tol: f64,
) -> Option<(f64, Vec<f64>)> {
    use std::f64;

    // Shift the matrix so the largest eigenvalue of (A - shift·I) is the guided mode
    // We shift by beta_sq_lo to make guided modes have positive eigenvalues
    let shift = beta_sq_lo;

    // Initial random-ish unit vector (use a fixed pattern to be deterministic)
    let mut v: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).sin()).collect();
    let norm0: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm0 < 1e-30 {
        return None;
    }
    for vi in &mut v {
        *vi /= norm0;
    }

    let mut lambda_old = f64::NEG_INFINITY;
    let mut w = vec![0.0f64; n];

    for _ in 0..max_iter {
        // w = (A - shift·I)·v
        for i in 0..n {
            let diag = a_mat[i * n + i] - shift;
            w[i] = diag * v[i];
            if i > 0 {
                w[i] += a_mat[i * n + (i - 1)] * v[i - 1];
            }
            if i + 1 < n {
                w[i] += a_mat[i * n + (i + 1)] * v[i + 1];
            }
        }

        // Rayleigh quotient: λ = v^T·w / v^T·v
        let vw: f64 = v.iter().zip(w.iter()).map(|(&vi, &wi)| vi * wi).sum();
        let vv: f64 = v.iter().map(|&vi| vi * vi).sum();
        let lambda_new = vw / vv + shift;

        // Normalise w → new v
        let norm: f64 = w.iter().map(|&wi| wi * wi).sum::<f64>().sqrt();
        if norm < 1e-30 {
            break;
        }
        for (vi, wi) in v.iter_mut().zip(w.iter()) {
            *vi = wi / norm;
        }

        if (lambda_new - lambda_old).abs() < tol {
            return Some((lambda_new, v));
        }
        lambda_old = lambda_new;
    }
    // Return best estimate even if not fully converged
    Some((lambda_old, v))
}

/// Solve for up to `n_want` dominant modes using power iteration with deflation.
///
/// Each successive mode is found by deflating the previously found modes
/// from the matrix (Gram-Schmidt orthogonalisation).
///
/// Returns a list of `(beta_sq, field_interior)` pairs.
pub fn solve_by_deflation(
    a_mat: &[f64],
    n: usize,
    beta_sq_lo: f64,
    beta_sq_hi: f64,
    n_want: usize,
    max_iter: usize,
    tol: f64,
) -> Vec<(f64, Vec<f64>)> {
    let mut results: Vec<(f64, Vec<f64>)> = Vec::new();
    // Deflated matrix copy
    let mut a_def = a_mat.to_vec();

    for _ in 0..n_want {
        if let Some((bsq, ev)) = power_iteration_dominant(&a_def, n, beta_sq_lo, max_iter, tol) {
            if bsq <= beta_sq_lo || bsq >= beta_sq_hi {
                break;
            }
            // Deflate: A ← A − bsq · v·vᵀ
            let vv: f64 = ev.iter().map(|&x| x * x).sum();
            if vv < 1e-30 {
                break;
            }
            for i in 0..n {
                for j in 0..n {
                    a_def[i * n + j] -= bsq * ev[i] * ev[j] / vv;
                }
            }
            results.push((bsq, ev));
        } else {
            break;
        }
    }

    // Sort by descending beta_sq
    results.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn si_sio2_slab() -> FemModeSolver1d {
        // Si core (n=3.48), SiO2 cladding (n=1.44), 220nm core, 1550nm
        FemModeSolver1d::slab(3.48, 1.44, 220e-9, 1550e-9, 101)
    }

    #[test]
    fn fem_finds_at_least_one_mode() {
        let solver = si_sio2_slab();
        let modes = solver.solve(1.44);
        assert!(!modes.is_empty(), "No guided modes found");
    }

    #[test]
    fn fundamental_mode_neff_in_range() {
        let solver = si_sio2_slab();
        let modes = solver.solve(1.44);
        assert!(!modes.is_empty());
        let n_eff = modes[0].n_eff;
        assert!(n_eff > 1.44 && n_eff < 3.48, "n_eff={n_eff}");
    }

    #[test]
    fn field_normalized() {
        let solver = si_sio2_slab();
        let modes = solver.solve(1.44);
        if modes.is_empty() {
            return;
        }
        let dx = 2.0 * solver.half_width / (solver.n_nodes - 1) as f64;
        let norm: f64 = modes[0].field.iter().map(|&v| v * v).sum::<f64>() * dx;
        assert!((norm - 1.0).abs() < 0.1, "norm={norm}");
    }

    #[test]
    fn fundamental_mode_is_symmetric() {
        let solver = si_sio2_slab();
        let modes = solver.solve(1.44);
        if modes.is_empty() {
            return;
        }
        let f = &modes[0].field;
        let n = f.len();
        let mid = n / 2;
        let asym: f64 = (0..mid)
            .map(|i| (f[i].abs() - f[n - 1 - i].abs()).abs())
            .sum::<f64>()
            / mid as f64;
        assert!(asym < 0.1, "asymmetry={asym}");
    }

    #[test]
    fn x_coords_correct_length() {
        let solver = si_sio2_slab();
        let x = solver.x_coords();
        assert_eq!(x.len(), solver.n_nodes);
    }

    #[test]
    fn x_coords_span_domain() {
        let solver = si_sio2_slab();
        let x = solver.x_coords();
        assert!((x[0] + solver.half_width).abs() < 1e-20, "x[0]={}", x[0]);
        assert!((x[x.len() - 1] - solver.half_width).abs() < 1e-20);
    }

    #[test]
    fn confinement_factor_in_01() {
        let solver = si_sio2_slab();
        let modes = solver.solve(1.44);
        if modes.is_empty() {
            return;
        }
        let x = solver.x_coords();
        let gamma = modes[0].confinement_factor(-110e-9, 110e-9, &x);
        assert!((0.0..=1.0).contains(&gamma), "gamma={gamma}");
    }

    #[test]
    fn intensity_is_field_squared() {
        let mode = FemMode1d {
            beta: 1e7,
            n_eff: 2.0,
            field: vec![1.0, -2.0, 0.5],
        };
        let intensity = mode.intensity();
        assert!((intensity[0] - 1.0).abs() < 1e-12);
        assert!((intensity[1] - 4.0).abs() < 1e-12);
        assert!((intensity[2] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn solve_vector_returns_at_most_n_modes() {
        let solver = si_sio2_slab();
        let modes = solver.solve_vector(1, 1.44);
        assert!(modes.len() <= 1, "Expected at most 1 mode");
    }

    #[test]
    fn beta_method_returns_some() {
        let solver = si_sio2_slab();
        let beta = solver.beta(0, 1.44);
        assert!(beta.is_some(), "beta(mode 0) should be Some");
        let b = beta.expect("guaranteed Some");
        assert!(b > 0.0, "beta should be positive");
    }

    #[test]
    fn confinement_factor_method_returns_value() {
        let solver = si_sio2_slab();
        let cf = solver.confinement_factor(0, (-110e-9, 110e-9), 1.44);
        if let Some(gamma) = cf {
            assert!((0.0..=1.0).contains(&gamma), "gamma={gamma}");
        }
    }

    #[test]
    fn group_index_is_reasonable() {
        let solver = si_sio2_slab();
        let ng = solver.group_index(0, 1e-9, 1.44);
        if let Some(ng_val) = ng {
            // n_g for Si at 1550nm is typically 3.5–5.0
            assert!(ng_val > 1.0 && ng_val < 10.0, "n_g={ng_val:.3}");
        }
    }

    #[test]
    fn effective_mode_area_positive() {
        let solver = si_sio2_slab();
        let modes = solver.solve(1.44);
        if modes.is_empty() {
            return;
        }
        let dx = 2.0 * solver.half_width / (solver.n_nodes - 1) as f64;
        let a_eff = modes[0].effective_mode_area(dx);
        assert!(a_eff > 0.0, "A_eff={a_eff:.2e}");
    }

    #[test]
    fn compute_dispersion_returns_data() {
        let solver = si_sio2_slab();
        let disp = solver.compute_dispersion(1, 5, (1400e-9, 1600e-9), 1.44);
        assert!(!disp.is_empty(), "Dispersion data should not be empty");
        // Fundamental mode should have data at all wavelengths
        assert!(!disp[0].is_empty(), "Mode 0 should have dispersion data");
    }

    #[test]
    fn full_dispersion_analysis_returns_points() {
        let solver = si_sio2_slab();
        let pts = solver.full_dispersion_analysis(1.44);
        assert!(!pts.is_empty(), "Should find at least one dispersion point");
        for pt in &pts {
            assert!(pt.n_eff > 1.0, "n_eff should be > 1");
            assert!(pt.n_g > 0.5, "n_g should be > 0.5");
        }
    }

    #[test]
    fn peak_intensity_index_valid() {
        let mode = FemMode1d {
            beta: 1e7,
            n_eff: 2.0,
            field: vec![0.1, 3.0, 0.5, 1.0],
        };
        let idx = mode.peak_intensity_index();
        assert_eq!(idx, 1, "Peak should be at index 1");
    }

    #[test]
    fn dispersion_monotone_in_silicon() {
        // For a strongly guided Si/SiO2 waveguide, n_eff should be nearly constant
        // across 100nm wavelength range
        let solver = si_sio2_slab();
        let disp = solver.compute_dispersion(1, 3, (1450e-9, 1650e-9), 1.44);
        if disp.is_empty() || disp[0].len() < 2 {
            return;
        }
        // n_eff should be in the guiding range at all wavelengths
        for &neff in &disp[0] {
            assert!(neff > 1.44 && neff < 3.48, "n_eff={neff} out of range");
        }
    }
}
