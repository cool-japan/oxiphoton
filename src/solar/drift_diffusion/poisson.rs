//! Poisson equation discretisation for electrostatic potential.
//!
//! Uses a Voronoi (finite-volume) discretisation on a 1D non-uniform mesh.
//! The residual at each interior node i is:
//!
//!   `F_ψ[i]` = ε_{i+½}(ψ_{i+1} − ψ_i)/dx_r
//!           − ε_{i-½}(ψ_i − ψ_{i-1})/dx_l
//!           + q·(p_i − n_i + N_d^+ − N_a^−) · dx_avg
//!
//! where dx_avg = (dx_l + dx_r) / 2 is the Voronoi cell width.
//!
//! # Units
//! * ε in F/cm (EPSILON_0 is F/m; multiply by 1e-2 for F/cm).
//! * ψ in V.
//! * q in C.
//! * dx in cm.
//! * Carrier densities in cm⁻³.
//! * Residual in C/cm² (net charge × Voronoi width / area).

use crate::units::conversion::EPSILON_0;

use super::material::Q;

/// Permittivity in F/cm for a relative permittivity `eps_r`.
///
/// `EPSILON_0` is in SI (F/m). Multiplying by 1e-2 converts F/m → F/cm.
#[inline]
pub fn eps_fcm(eps_r: f64) -> f64 {
    eps_r * EPSILON_0 * 1e-2
}

/// Poisson equation residual at interior node `i` (in C/cm²).
///
/// Only valid for `1 <= i <= n - 2` (interior nodes). Boundary nodes are
/// handled by Dirichlet boundary conditions in the Newton solver.
///
/// # Arguments
/// * `i` — node index (must be interior: 1 ≤ i ≤ N-2).
/// * `psi` — electrostatic potential at all nodes (V), length N.
/// * `n` — electron density (cm⁻³), length N.
/// * `p` — hole density (cm⁻³), length N.
/// * `nd` — ionised donor density (cm⁻³), length N.
/// * `na` — ionised acceptor density (cm⁻³), length N.
/// * `dx` — grid spacing `dx[k] = x[k+1] - x[k]` (cm), length N-1.
/// * `eps_r` — relative permittivity (uniform across device).
#[allow(clippy::too_many_arguments)]
pub fn poisson_residual(
    i: usize,
    psi: &[f64],
    n: &[f64],
    p: &[f64],
    nd: &[f64],
    na: &[f64],
    dx: &[f64],
    eps_r: f64,
) -> f64 {
    let eps = eps_fcm(eps_r);
    let dx_l = dx[i - 1];
    let dx_r = dx[i];
    let dx_avg = 0.5 * (dx_l + dx_r);
    let flux_r = eps * (psi[i + 1] - psi[i]) / dx_r;
    let flux_l = eps * (psi[i] - psi[i - 1]) / dx_l;
    flux_r - flux_l + Q * (p[i] - n[i] + nd[i] - na[i]) * dx_avg
}

/// Jacobian entries of the Poisson residual at node `i` with respect to:
/// (ψ_{i-1}, ψ_i, ψ_{i+1}, n_i, p_i).
///
/// Returns `(dF_psi_l, dF_psi_c, dF_psi_r, dF_n, dF_p)`.
///
/// Used in the Newton Jacobian assembly.
pub fn poisson_jacobian(i: usize, dx: &[f64], eps_r: f64) -> (f64, f64, f64, f64, f64) {
    let eps = eps_fcm(eps_r);
    let dx_l = dx[i - 1];
    let dx_r = dx[i];
    let dx_avg = 0.5 * (dx_l + dx_r);

    // F = eps*(psi[i+1]-psi[i])/dx_r - eps*(psi[i]-psi[i-1])/dx_l + q*(p-n+Nd-Na)*dx_avg
    // ∂F/∂ψ_{i-1} = +ε/dx_l   (from second term: -eps*(-1/dx_l) = +eps/dx_l)
    let df_psi_l = eps / dx_l;
    // ∂F/∂ψ_{i+1} = +ε/dx_r   (from first term: eps*(+1/dx_r))
    let df_psi_r = eps / dx_r;
    // ∂F/∂ψ_i = −ε/dx_r − ε/dx_l = −ε*(1/dx_l + 1/dx_r)
    let df_psi_c = -eps * (1.0 / dx_l + 1.0 / dx_r);
    // ∂F/∂n_i = −q * dx_avg
    let df_n = -Q * dx_avg;
    // ∂F/∂p_i = +q * dx_avg
    let df_p = Q * dx_avg;

    (df_psi_l, df_psi_c, df_psi_r, df_n, df_p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisson_residual_flat_potential_zero_charge() {
        // If psi is uniform and charge-neutral, residual should be zero.
        let n = 10;
        let psi = vec![0.5_f64; n];
        let carriers = vec![1e16_f64; n];
        // Charge neutral: p - n + nd - na = 0 => nd = na + n - p
        // For simplicity: n=p, nd=na
        let nd = vec![1e16_f64; n];
        let na = vec![1e16_f64; n];
        let dx = vec![1e-5_f64; n - 1];

        for i in 1..n - 1 {
            let res = poisson_residual(i, &psi, &carriers, &carriers, &nd, &na, &dx, 11.7);
            assert!(res.abs() < 1e-30, "Residual at node {i}: {res}");
        }
    }
}
