//! Gummel decoupled iteration for the coupled Poisson + drift-diffusion system.
//!
//! Each outer iteration solves three tridiagonal (Thomas) sub-problems:
//! 1. Poisson (linear, n/p fixed)
//! 2. Electron continuity (linear in n, SG flux, ψ/p fixed)
//! 3. Hole continuity (linear in p, SG flux, ψ/n fixed)
//!
//! For equilibrium: Newton on the nonlinear Poisson equation with Boltzmann
//! carrier update (n = nᵢ·exp(ψ/VT), p = nᵢ·exp(−ψ/VT)).
//!
//! All sub-solves use the Thomas algorithm (O(N), M-matrix structure).
//!
//! References:
//! Gummel (1964), IEEE TED 11(10), 455-465.
//! Selberherr (1984), "Analysis and Simulation of Semiconductor Devices."

use crate::error::OxiPhotonError;

use super::{
    continuity::bernoulli,
    material::{SemiconductorMaterial, StatisticsModel, Q},
    poisson::eps_fcm,
    recombination::total_recombination,
};

// ─── Thomas algorithm ─────────────────────────────────────────────────────────

/// Solve a tridiagonal system A·x = rhs using the Thomas algorithm.
///
/// * `lower[i]` — sub-diagonal for row i+1 (length n-1).
/// * `diag[i]`  — main diagonal for row i (length n, modified in-place).
/// * `upper[i]` — super-diagonal for row i (length n-1).
/// * `rhs[i]`   — right-hand side (length n, modified in-place).
fn thomas_solve(
    lower: &[f64],
    diag: &mut [f64],
    upper: &[f64],
    rhs: &mut [f64],
) -> Result<Vec<f64>, OxiPhotonError> {
    let n = diag.len();
    for i in 1..n {
        let pivot = diag[i - 1];
        if pivot.abs() < 1e-300 {
            return Err(OxiPhotonError::NumericalError(format!(
                "Thomas: near-zero pivot at row {}",
                i - 1
            )));
        }
        let w = lower[i - 1] / pivot;
        diag[i] -= w * upper[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }
    let mut x = vec![0.0_f64; n];
    let last = n - 1;
    if diag[last].abs() < 1e-300 {
        return Err(OxiPhotonError::NumericalError(
            "Thomas: zero pivot at last row".to_string(),
        ));
    }
    x[last] = rhs[last] / diag[last];
    for i in (0..last).rev() {
        x[i] = (rhs[i] - upper[i] * x[i + 1]) / diag[i];
    }
    Ok(x)
}

// ─── Equilibrium Poisson Newton ───────────────────────────────────────────────

/// Newton solve for the nonlinear Poisson equation at equilibrium.
///
/// Carrier concentrations are updated via Boltzmann statistics at each step:
///   n(ψ) = nᵢ_eff[i]·exp(ψ/VT),  p(ψ) = nᵢ_eff[i]·exp(−ψ/VT)
///
/// The tridiagonal Jacobian at interior node i:
///   lower = ε/dx_l,  diag = −ε(1/dx_l+1/dx_r) − q·dx_avg·nᵢ_eff[i]/VT·(eψ+e−ψ),  upper = ε/dx_r
///
/// `ni_eff` is a per-node effective intrinsic carrier concentration array,
/// pre-computed from the material's BGN model.
#[allow(clippy::too_many_arguments)]
fn solve_poisson_equilibrium(
    psi: &mut [f64],
    nd: &[f64],
    na: &[f64],
    dx: &[f64],
    eps_r: f64,
    ni_eff: &[f64],
    vt: f64,
    psi_left: f64,
    psi_right: f64,
    max_iter: usize,
) -> Result<usize, OxiPhotonError> {
    let nn = psi.len();
    let eps = eps_fcm(eps_r);

    for iter in 0..max_iter {
        let mut lower = vec![0.0_f64; nn - 1];
        let mut diag = vec![0.0_f64; nn];
        let mut upper = vec![0.0_f64; nn - 1];
        let mut f = vec![0.0_f64; nn];

        // Boundary rows
        diag[0] = 1.0;
        f[0] = psi[0] - psi_left;
        diag[nn - 1] = 1.0;
        f[nn - 1] = psi[nn - 1] - psi_right;

        for i in 1..nn - 1 {
            let dx_l = dx[i - 1];
            let dx_r = dx[i];
            let dx_avg = 0.5 * (dx_l + dx_r);
            let ep = (psi[i] / vt).exp();
            let em = (-psi[i] / vt).exp();
            let ni_i = ni_eff[i];
            let n_i = ni_i * ep;
            let p_i = ni_i * em;

            f[i] = eps * (psi[i + 1] - psi[i]) / dx_r - eps * (psi[i] - psi[i - 1]) / dx_l
                + Q * (p_i - n_i + nd[i] - na[i]) * dx_avg;

            lower[i - 1] = eps / dx_l;
            diag[i] = -eps * (1.0 / dx_l + 1.0 / dx_r) - Q * dx_avg * ni_i / vt * (ep + em);
            upper[i] = eps / dx_r;
        }

        // Convergence check on interior residuals
        let f_norm = f[1..nn - 1]
            .iter()
            .map(|&x| x.abs())
            .fold(0.0_f64, f64::max);
        if f_norm < 1e-8 {
            return Ok(iter);
        }

        // Solve J·Δψ = −F
        let mut neg_f: Vec<f64> = f.iter().map(|&x| -x).collect();
        let delta = thomas_solve(&lower, &mut diag, &upper, &mut neg_f)?;

        // Apply update with potential damping (cap at 5·VT per step)
        let max_dpsi = delta[1..nn - 1]
            .iter()
            .map(|&d| d.abs())
            .fold(0.0_f64, f64::max);
        let damp = if max_dpsi > 5.0 * vt {
            5.0 * vt / max_dpsi
        } else {
            1.0
        };
        for i in 1..nn - 1 {
            psi[i] += damp * delta[i];
        }
    }

    Err(OxiPhotonError::NumericalError(format!(
        "Poisson equilibrium Newton did not converge in {max_iter} iterations"
    )))
}

// ─── Continuity tridiagonal sub-solvers ──────────────────────────────────────

/// Solve the electron continuity equation as a linear tridiagonal system.
///
/// With ψ fixed, the SG flux is linear in n. Recombination is treated explicitly
/// (Picard/Gummel): R is evaluated at old n, old p and placed on the RHS.
/// This preserves the M-matrix structure of the SG discretisation, which guarantees
/// Thomas-algorithm stability regardless of the electric-field magnitude.
///
/// Tridiagonal coefficients at interior node i (Δψ_r = (ψ[i+1]-ψ[i])/VT):
///   lower = −(q·dn_edge[i-1]/dx_l)·B(−Δψ_l)
///   diag  = (q·dn_edge[i]/dx_r)·B(−Δψ_r) + (q·dn_edge[i-1]/dx_l)·B(Δψ_l)
///   upper = −(q·dn_edge[i]/dx_r)·B(Δψ_r)
///   rhs   = q·(R_old − G)·dx_avg
///
/// `dn_per_edge` is a per-half-node electron diffusivity array (harmonic-mean
/// of adjacent node diffusivities), length `nn-1`. Using per-edge values allows
/// Fermi-Dirac statistics to vary spatially with carrier density.
///
/// `ni_eff_node` is a per-node effective intrinsic carrier concentration array.
#[allow(clippy::too_many_arguments)]
fn solve_n_tridiag(
    n: &mut [f64],
    psi: &[f64],
    p_old: &[f64],
    gen: &[f64],
    dx: &[f64],
    mat: &SemiconductorMaterial,
    temp_k: f64,
    n_left: f64,
    n_right: f64,
    ni_eff_node: &[f64],
    dn_per_edge: &[f64],
) -> Result<(), OxiPhotonError> {
    let nn = n.len();
    let ni = mat.ni_cm3;
    let vt = mat.vt_at(temp_k);

    let mut lower = vec![0.0_f64; nn - 1];
    let mut diag = vec![0.0_f64; nn];
    let mut upper = vec![0.0_f64; nn - 1];
    let mut rhs = vec![0.0_f64; nn];

    diag[0] = 1.0;
    rhs[0] = n_left;
    diag[nn - 1] = 1.0;
    rhs[nn - 1] = n_right;

    for i in 1..nn - 1 {
        let dx_l = dx[i - 1];
        let dx_r = dx[i];
        let dx_avg = 0.5 * (dx_l + dx_r);
        let dpsi_r = (psi[i + 1] - psi[i]) / vt;
        let dpsi_l = (psi[i] - psi[i - 1]) / vt;
        let br_pos = bernoulli(dpsi_r);
        let br_neg = bernoulli(-dpsi_r);
        let bl_pos = bernoulli(dpsi_l);
        let bl_neg = bernoulli(-dpsi_l);
        // edge i-1 is the left half-node, edge i is the right half-node
        let cr = Q * dn_per_edge[i] / dx_r;
        let cl = Q * dn_per_edge[i - 1] / dx_l;

        lower[i - 1] = -cl * bl_neg;
        diag[i] = cr * br_neg + cl * bl_pos;
        upper[i] = -cr * br_pos;

        let r_old = total_recombination(n[i], p_old[i], ni_eff_node[i], mat);
        rhs[i] = Q * (r_old - gen[i]) * dx_avg;
    }

    let n_new = thomas_solve(&lower, &mut diag, &upper, &mut rhs)?;
    for i in 0..nn {
        n[i] = n_new[i].max(1e-10 * ni);
    }
    Ok(())
}

/// Solve the hole continuity equation as a linear tridiagonal system.
///
/// With ψ, n_cur fixed, the SG flux is linear in p. Recombination is treated explicitly
/// (Picard/Gummel): R is evaluated at n_cur, old p and placed on the RHS.
/// This preserves the M-matrix structure of the SG discretisation.
///
/// Tridiagonal coefficients at interior node i:
///   lower = −(q·dp_edge[i-1]/dx_l)·B(Δψ_l)
///   diag  = (q·dp_edge[i-1]/dx_l)·B(−Δψ_l) + (q·dp_edge[i]/dx_r)·B(Δψ_r)
///   upper = −(q·dp_edge[i]/dx_r)·B(−Δψ_r)
///   rhs   = −q·(R_old − G)·dx_avg
///
/// `dp_per_edge` is a per-half-node hole diffusivity array (harmonic-mean of
/// adjacent node diffusivities), length `nn-1`.
///
/// `ni_eff_node` is a per-node effective intrinsic carrier concentration array.
#[allow(clippy::too_many_arguments)]
fn solve_p_tridiag(
    p: &mut [f64],
    psi: &[f64],
    n_cur: &[f64],
    gen: &[f64],
    dx: &[f64],
    mat: &SemiconductorMaterial,
    temp_k: f64,
    p_left: f64,
    p_right: f64,
    ni_eff_node: &[f64],
    dp_per_edge: &[f64],
) -> Result<(), OxiPhotonError> {
    let nn = p.len();
    let ni = mat.ni_cm3;
    let vt = mat.vt_at(temp_k);

    let mut lower = vec![0.0_f64; nn - 1];
    let mut diag = vec![0.0_f64; nn];
    let mut upper = vec![0.0_f64; nn - 1];
    let mut rhs = vec![0.0_f64; nn];

    diag[0] = 1.0;
    rhs[0] = p_left;
    diag[nn - 1] = 1.0;
    rhs[nn - 1] = p_right;

    for i in 1..nn - 1 {
        let dx_l = dx[i - 1];
        let dx_r = dx[i];
        let dx_avg = 0.5 * (dx_l + dx_r);
        let dpsi_r = (psi[i + 1] - psi[i]) / vt;
        let dpsi_l = (psi[i] - psi[i - 1]) / vt;
        let br_pos = bernoulli(dpsi_r);
        let br_neg = bernoulli(-dpsi_r);
        let bl_pos = bernoulli(dpsi_l);
        let bl_neg = bernoulli(-dpsi_l);
        // edge i-1 is the left half-node, edge i is the right half-node
        let cr = Q * dp_per_edge[i] / dx_r;
        let cl = Q * dp_per_edge[i - 1] / dx_l;

        // F_p = J_{p,i-½} - J_{p,i+½} + q(R-G)dx_avg = 0
        // J_{p,i-½} = (q·dp_l/dx_l)(p[i]*B(-Δψ_l) - p[i-1]*B(Δψ_l))
        // J_{p,i+½} = (q·dp_r/dx_r)(p[i+1]*B(-Δψ_r) - p[i]*B(Δψ_r))
        // → lower: -cl*bl_pos, diag: cl*bl_neg + cr*br_pos, upper: -cr*br_neg
        lower[i - 1] = -cl * bl_pos;
        diag[i] = cl * bl_neg + cr * br_pos;
        upper[i] = -cr * br_neg;

        let r_old = total_recombination(n_cur[i], p[i], ni_eff_node[i], mat);
        rhs[i] = -Q * (r_old - gen[i]) * dx_avg;
    }

    let p_new = thomas_solve(&lower, &mut diag, &upper, &mut rhs)?;
    for i in 0..nn {
        p[i] = p_new[i].max(1e-10 * ni);
    }
    Ok(())
}

// ─── Gummel non-equilibrium outer loop ───────────────────────────────────────

/// Solve Poisson with quasi-Fermi potentials fixed (non-equilibrium inner Newton).
///
/// In the Gummel framework, quasi-Fermi potentials φ_n, φ_p are held fixed while
/// updating ψ. Carrier densities are related to ψ via:
///   n = n_ref · exp((ψ − ψ_ref) / VT)
///   p = p_ref · exp(−(ψ − ψ_ref) / VT)
///
/// This makes Poisson exponentially nonlinear in ψ, but the linearized Jacobian
/// is still tridiagonal. Uses the same Newton+Thomas approach as equilibrium,
/// providing fast convergence within the Gummel outer loop.
#[allow(clippy::too_many_arguments)]
fn solve_poisson_gummel_inner(
    psi: &mut [f64],
    n_ref: &[f64],
    p_ref: &[f64],
    psi_ref: &[f64],
    nd: &[f64],
    na: &[f64],
    dx: &[f64],
    eps_r: f64,
    vt: f64,
    psi_left: f64,
    psi_right: f64,
    max_inner: usize,
) -> Result<(), OxiPhotonError> {
    let nn = psi.len();
    let eps = eps_fcm(eps_r);

    for _iter in 0..max_inner {
        let mut lower = vec![0.0_f64; nn - 1];
        let mut diag = vec![0.0_f64; nn];
        let mut upper = vec![0.0_f64; nn - 1];
        let mut f = vec![0.0_f64; nn];

        diag[0] = 1.0;
        f[0] = psi[0] - psi_left;
        diag[nn - 1] = 1.0;
        f[nn - 1] = psi[nn - 1] - psi_right;

        for i in 1..nn - 1 {
            let dx_l = dx[i - 1];
            let dx_r = dx[i];
            let dx_avg = 0.5 * (dx_l + dx_r);

            // Carrier densities from quasi-Fermi level approach:
            // n = n_ref[i] * exp((ψ[i] - ψ_ref[i]) / VT)
            // p = p_ref[i] * exp(-(ψ[i] - ψ_ref[i]) / VT)
            let dv = (psi[i] - psi_ref[i]) / vt;
            let n_i = n_ref[i] * dv.exp();
            let p_i = p_ref[i] * (-dv).exp();

            f[i] = eps * (psi[i + 1] - psi[i]) / dx_r - eps * (psi[i] - psi[i - 1]) / dx_l
                + Q * (p_i - n_i + nd[i] - na[i]) * dx_avg;

            // Jacobian: d(p_i)/dψ[i] = -p_i/VT, d(n_i)/dψ[i] = n_i/VT
            lower[i - 1] = eps / dx_l;
            diag[i] = -eps * (1.0 / dx_l + 1.0 / dx_r) - Q * dx_avg * (n_i + p_i) / vt;
            upper[i] = eps / dx_r;
        }

        let f_norm = f[1..nn - 1]
            .iter()
            .map(|&x| x.abs())
            .fold(0.0_f64, f64::max);
        if f_norm < 1e-8 {
            return Ok(());
        }

        let mut neg_f: Vec<f64> = f.iter().map(|&x| -x).collect();
        let delta = thomas_solve(&lower, &mut diag, &upper, &mut neg_f)?;

        // Damped update: cap at 5·VT per step
        let max_dpsi = delta[1..nn - 1]
            .iter()
            .map(|&d| d.abs())
            .fold(0.0_f64, f64::max);
        let damp = if max_dpsi > 5.0 * vt {
            5.0 * vt / max_dpsi
        } else {
            1.0
        };
        for i in 1..nn - 1 {
            psi[i] += damp * delta[i];
        }
    }
    // Unconverged inner Newton is acceptable — Gummel outer will compensate
    Ok(())
}

/// Gummel decoupled iteration for the non-equilibrium drift-diffusion system.
///
/// Each outer iteration:
/// 1. Solve Poisson (nonlinear Newton on tridiagonal, quasi-Fermi fixed) → update ψ
/// 2. Update n, p from new ψ and quasi-Fermi levels
/// 3. Solve electron continuity (linear tridiagonal in n, ψ/p fixed) → update n
/// 4. Solve hole continuity (linear tridiagonal in p, ψ/n fixed) → update p
///
/// Repeat until scaled changes in ψ, n, p all fall below `tol`.
#[allow(clippy::too_many_arguments)]
fn gummel_nonequil_solve(
    psi: &mut [f64],
    n: &mut [f64],
    p: &mut [f64],
    nd: &[f64],
    na: &[f64],
    gen: &[f64],
    dx: &[f64],
    mat: &SemiconductorMaterial,
    temp_k: f64,
    psi_left: f64,
    psi_right: f64,
    n_left: f64,
    p_left: f64,
    n_right: f64,
    p_right: f64,
    max_iter: usize,
    tol: f64,
) -> Result<usize, OxiPhotonError> {
    let vt = mat.vt_at(temp_k);
    let eps_r = mat.eps_r;
    let nn = psi.len();
    let n_edges = nn - 1;

    // For Fermi-Dirac statistics, allow more iterations because the spatially-varying
    // diffusivity introduces additional nonlinearity into the Gummel sweep.
    let effective_max_iters = if matches!(mat.statistics, StatisticsModel::FermiDirac) {
        max_iter.max(150)
    } else {
        max_iter
    };

    // Pre-compute per-node effective intrinsic carrier concentration.
    // BGN raises ni_eff in heavily-doped regions (N > ~1e18 cm⁻³).
    let ni_eff_node: Vec<f64> = (0..nn)
        .map(|i| mat.n_ie_squared(temp_k, nd[i], na[i]).sqrt())
        .collect();

    for iter in 0..effective_max_iters {
        let psi_old: Vec<f64> = psi.to_vec();
        let n_old: Vec<f64> = n.to_vec();
        let p_old: Vec<f64> = p.to_vec();

        // Pre-compute per-node diffusivities, frozen at the start of this Gummel iteration.
        // For Fermi-Dirac, D varies with local carrier density (modified Einstein relation).
        // For Boltzmann, D = μ·V_T is uniform; map to the same array for consistency.
        let dn_node: Vec<f64> = (0..nn)
            .map(|i| match mat.statistics {
                StatisticsModel::Boltzmann => mat.dn_cm2_s(temp_k),
                StatisticsModel::FermiDirac => mat.dn_cm2_s_fd(temp_k, n_old[i]),
            })
            .collect();
        let dp_node: Vec<f64> = (0..nn)
            .map(|i| match mat.statistics {
                StatisticsModel::Boltzmann => mat.dp_cm2_s(temp_k),
                StatisticsModel::FermiDirac => mat.dp_cm2_s_fd(temp_k, p_old[i]),
            })
            .collect();

        // Derive per-edge diffusivities via harmonic mean of adjacent node values.
        // Harmonic mean preserves positivity and is the correct interpolation for
        // series-connected resistors (flux = D·∇n, each half-interval a resistor).
        let dn_edge: Vec<f64> = (0..n_edges)
            .map(|i| {
                let a = dn_node[i];
                let b = dn_node[i + 1];
                if a == 0.0 || b == 0.0 {
                    0.0
                } else {
                    2.0 * a * b / (a + b)
                }
            })
            .collect();
        let dp_edge: Vec<f64> = (0..n_edges)
            .map(|i| {
                let a = dp_node[i];
                let b = dp_node[i + 1];
                if a == 0.0 || b == 0.0 {
                    0.0
                } else {
                    2.0 * a * b / (a + b)
                }
            })
            .collect();

        // Step 1: solve Poisson with quasi-Fermi potentials fixed (10 inner Newton steps).
        // Carries n, p via n_ref*exp((ψ-ψ_ref)/VT), p_ref*exp(-(ψ-ψ_ref)/VT) inside.
        solve_poisson_gummel_inner(
            psi, &n_old, &p_old, &psi_old, nd, na, dx, eps_r, vt, psi_left, psi_right, 10,
        )?;

        // Step 2: solve electron continuity for n (new ψ, old p as background).
        // The continuity solve re-linearizes the SG flux with the updated ψ.
        solve_n_tridiag(
            n,
            psi,
            &p_old,
            gen,
            dx,
            mat,
            temp_k,
            n_left,
            n_right,
            &ni_eff_node,
            &dn_edge,
        )?;

        // Step 3: solve hole continuity for p (new ψ, new n from step 2).
        let n_cur: Vec<f64> = n.to_vec();
        solve_p_tridiag(
            p,
            psi,
            &n_cur,
            gen,
            dx,
            mat,
            temp_k,
            p_left,
            p_right,
            &ni_eff_node,
            &dp_edge,
        )?;

        // Convergence: scaled changes in ψ, n, p.
        // Use per-node ni_eff for the carrier scaling so that minority carriers in
        // heavily-doped regions (where n or p << ni) are correctly normalised.
        let dpsi = psi
            .iter()
            .zip(psi_old.iter())
            .map(|(a, b)| ((a - b) / vt).abs())
            .fold(0.0_f64, f64::max);
        let dn_s = n
            .iter()
            .zip(n_old.iter())
            .enumerate()
            .map(|(i, (a, b))| {
                let scale = b.max(ni_eff_node[i]);
                ((a - b) / scale).abs()
            })
            .fold(0.0_f64, f64::max);
        let dp_s = p
            .iter()
            .zip(p_old.iter())
            .enumerate()
            .map(|(i, (a, b))| {
                let scale = b.max(ni_eff_node[i]);
                ((a - b) / scale).abs()
            })
            .fold(0.0_f64, f64::max);

        if dpsi < tol && dn_s < tol && dp_s < tol {
            return Ok(iter + 1);
        }
    }

    Err(OxiPhotonError::NumericalError(format!(
        "Gummel iteration did not converge in {effective_max_iters} iterations"
    )))
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Solve the equilibrium (zero bias, zero generation) system.
///
/// Uses Newton on the nonlinear Poisson equation (tridiagonal), then updates
/// n, p from Boltzmann statistics (n = nᵢ·exp(ψ/VT), p = nᵢ²/n).
///
/// Returns the number of Poisson Newton iterations.
#[allow(clippy::too_many_arguments)]
pub fn solve_equilibrium_gummel(
    psi: &mut [f64],
    n: &mut [f64],
    p: &mut [f64],
    nd: &[f64],
    na: &[f64],
    dx: &[f64],
    mat: &SemiconductorMaterial,
    temp_k: f64,
    psi_left: f64,
    psi_right: f64,
    n_left: f64,
    p_left: f64,
    n_right: f64,
    p_right: f64,
) -> Result<usize, OxiPhotonError> {
    let nn = psi.len();
    let ni = mat.ni_cm3;
    let vt = mat.vt_at(temp_k);

    // Pre-compute per-node effective intrinsic carrier concentration.
    // BGN raises ni_eff in heavily-doped regions; for None model this is just ni everywhere.
    let ni_eff_node: Vec<f64> = (0..nn)
        .map(|i| mat.n_ie_squared(temp_k, nd[i], na[i]).sqrt())
        .collect();

    // Set boundary nodes
    psi[0] = psi_left;
    n[0] = n_left;
    p[0] = p_left;
    psi[nn - 1] = psi_right;
    n[nn - 1] = n_right;
    p[nn - 1] = p_right;

    // Initialise interior n, p from current ψ via Boltzmann (using per-node ni_eff)
    for i in 1..nn - 1 {
        let ni_i = ni_eff_node[i];
        n[i] = (ni_i * (psi[i] / vt).exp()).max(1e-10 * ni);
        p[i] = (ni_i * (-psi[i] / vt).exp()).max(1e-10 * ni);
    }

    // Solve nonlinear Poisson (tridiagonal Newton with Boltzmann n,p)
    let n_iters = solve_poisson_equilibrium(
        psi,
        nd,
        na,
        dx,
        mat.eps_r,
        &ni_eff_node,
        vt,
        psi_left,
        psi_right,
        200,
    )?;

    // Update n, p from converged ψ (using per-node ni_eff)
    for i in 1..nn - 1 {
        let ni_i = ni_eff_node[i];
        n[i] = (ni_i * (psi[i] / vt).exp()).max(1e-10 * ni);
        p[i] = (ni_i * (-psi[i] / vt).exp()).max(1e-10 * ni);
    }

    Ok(n_iters)
}

/// Non-equilibrium Gummel solve (dark IV and illuminated IV).
///
/// Keeps the same signature as the original `newton_solve` for backward
/// compatibility with all callers in `mod.rs`.
#[allow(clippy::too_many_arguments)]
pub fn newton_solve(
    psi: &mut [f64],
    n: &mut [f64],
    p: &mut [f64],
    nd: &[f64],
    na: &[f64],
    gen: &[f64],
    dx: &[f64],
    mat: &SemiconductorMaterial,
    temp_k: f64,
    v_left: f64,
    v_right: f64,
    max_iter: usize,
    tol: f64,
) -> Result<usize, OxiPhotonError> {
    let nn = psi.len();
    let n_left = n[0];
    let p_left = p[0];
    let n_right = n[nn - 1];
    let p_right = p[nn - 1];

    gummel_nonequil_solve(
        psi, n, p, nd, na, gen, dx, mat, temp_k, v_left, v_right, n_left, p_left, n_right, p_right,
        max_iter, tol,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solar::drift_diffusion::material::SemiconductorMaterial;

    #[test]
    fn residual_flat_potential_charge_neutral_zero() {
        let mat = SemiconductorMaterial::silicon();
        let ni = mat.ni_cm3;
        let n_nodes = 10;
        let dx = vec![1e-5_f64; n_nodes - 1];
        let mut psi = vec![0.0_f64; n_nodes];
        let mut n = vec![ni; n_nodes];
        let mut p = vec![ni; n_nodes];
        let nd = vec![0.0_f64; n_nodes];
        let na = vec![0.0_f64; n_nodes];
        let vt = mat.vt_at(300.0);

        // For the intrinsic case with BgnModel::None, ni_eff = ni everywhere
        let ni_eff_node = vec![ni; n_nodes];

        // A flat intrinsic device should already be at equilibrium
        let result = solve_poisson_equilibrium(
            &mut psi,
            &nd,
            &na,
            &dx,
            mat.eps_r,
            &ni_eff_node,
            vt,
            0.0,
            0.0,
            10,
        );
        assert!(result.is_ok(), "Should converge: {:?}", result);
        // Check n, p are still ni after Boltzmann update
        for i in 0..n_nodes {
            n[i] = (ni * (psi[i] / vt).exp()).max(1e-10 * ni);
            p[i] = (ni * (-psi[i] / vt).exp()).max(1e-10 * ni);
            let np = n[i] * p[i];
            assert!(
                (np - ni * ni).abs() / (ni * ni) < 1e-6,
                "Mass action at {i}: n*p={np:.3e}, ni²={:.3e}",
                ni * ni
            );
        }
    }

    #[test]
    fn thomas_solve_small_system() {
        // Solve: [2 -1 0; -1 2 -1; 0 -1 2] * x = [1; 0; 1]
        // Solution: x = [1; 1; 1]
        let lower = vec![-1.0_f64, -1.0];
        let mut diag = vec![2.0_f64, 2.0, 2.0];
        let upper = vec![-1.0_f64, -1.0];
        let mut rhs = vec![1.0_f64, 0.0, 1.0];
        let x = thomas_solve(&lower, &mut diag, &upper, &mut rhs).expect("solve");
        assert!((x[0] - 1.0).abs() < 1e-12, "x[0] = {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-12, "x[1] = {}", x[1]);
        assert!((x[2] - 1.0).abs() < 1e-12, "x[2] = {}", x[2]);
    }
}
