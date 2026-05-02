/// Topology optimization for photonic devices.
///
/// Combines the adjoint gradient with:
///   1. Density filtering (spatial smoothing to impose minimum feature size)
///   2. Projection (sigmoid/Heaviside to push design toward binary)
///   3. Manufacturing constraints (minimum feature size, connectivity)
///
/// The standard pipeline (Sigmund 2007, Jensen & Sigmund 2011):
///   ρ̃ = filter(ρ)               — smoothed density
///   ρ̄ = project(ρ̃)              — projected (binarised) density
///   ε(r) = eps_min + ρ̄·(eps_max - eps_min)   — permittivity
use super::adjoint::DesignRegion;

/// Compute the continuation schedule for β-continuation.
///
/// Returns a `Vec` of `(iteration_start, beta)` pairs. After `n_steps_per_beta`
/// OC iterations at each β level, β doubles (or advances to the next entry).
///
/// # Arguments
/// * `n_steps_per_beta` – number of OC iterations at each β level
/// * `betas`            – ordered sequence of β values
pub fn continuation_schedule(n_steps_per_beta: usize, betas: &[f64]) -> Vec<(usize, f64)> {
    betas
        .iter()
        .enumerate()
        .map(|(i, &b)| (i * n_steps_per_beta, b))
        .collect()
}

/// Simple "concentrate density in upper half" figure of merit for integration testing.
///
/// FOM = mean(ρ̄\[i\] for i in upper half of grid).
/// This exercises the full OC pipeline without requiring a real BPM/FDTD solve.
pub struct Pseudo2dFom {
    /// Number of pixels in x-direction
    pub nx: usize,
    /// Number of pixels in z-direction
    pub nz: usize,
}

impl Pseudo2dFom {
    /// Create a new `Pseudo2dFom` for an `nx × nz` design grid.
    pub fn new(nx: usize, nz: usize) -> Self {
        Self { nx, nz }
    }

    /// Evaluate FOM and compute gradient w.r.t. projected density.
    ///
    /// Returns `(fom, grad_projected)` where `grad_projected[i] = dFOM/dρ̄[i]`.
    pub fn evaluate(&self, projected: &[f64]) -> (f64, Vec<f64>) {
        let n = self.nx * self.nz;
        let upper_half_start = n / 2;
        let n_upper = n - upper_half_start;
        let fom = projected[upper_half_start..].iter().sum::<f64>() / n_upper as f64;
        let mut grad = vec![0.0_f64; n];
        for g in grad.iter_mut().take(n).skip(upper_half_start) {
            *g = 1.0 / n_upper as f64;
        }
        (fom, grad)
    }
}

/// Topology optimizer combining filtering, projection, and gradient updates.
pub struct TopologyOptimizer {
    /// Current design region
    pub region: DesignRegion,
    /// Filter radius (in pixels) for density filtering
    pub filter_radius: f64,
    /// Projection steepness β (increases over iterations)
    pub beta: f64,
    /// Projection threshold η
    pub eta: f64,
    /// Current iteration
    pub iteration: usize,
    /// History of FOM values
    pub fom_history: Vec<f64>,
}

/// Compute the mean projected volume fraction for a raw density array.
///
/// Applies the Gaussian density filter followed by the Heaviside projection
/// and returns the mean of the projected values.  This is a pure function that
/// does NOT require a `TopologyOptimizer` instance, so it can be used in
/// bisection helpers without cloning the full optimizer.
fn projected_volume(
    rho: &[f64],
    beta: f64,
    eta: f64,
    filter_radius: f64,
    nx: usize,
    nz: usize,
) -> f64 {
    let n = nx * nz;
    let r = filter_radius;
    let r_int = r.ceil() as i64;
    let tanh_b_eta = (beta * eta).tanh();
    let denom = tanh_b_eta + (beta * (1.0 - eta)).tanh();

    let mut total = 0.0_f64;
    for j in 0..nz {
        for i in 0..nx {
            let mut weight_sum = 0.0_f64;
            let mut val_sum = 0.0_f64;
            for dj in -r_int..=r_int {
                for di in -r_int..=r_int {
                    let dist2 = (di * di + dj * dj) as f64;
                    if dist2 > r * r {
                        continue;
                    }
                    let ni = i as i64 + di;
                    let nj = j as i64 + dj;
                    if ni < 0 || ni >= nx as i64 || nj < 0 || nj >= nz as i64 {
                        continue;
                    }
                    let w = (-(dist2) / (r * r)).exp();
                    weight_sum += w;
                    val_sum += w * rho[nj as usize * nx + ni as usize];
                }
            }
            let filtered_ij = val_sum / weight_sum.max(1e-30);
            let projected_ij = (tanh_b_eta + (beta * (filtered_ij - eta)).tanh()) / denom;
            total += projected_ij;
        }
    }
    total / n as f64
}

impl TopologyOptimizer {
    pub fn new(region: DesignRegion, filter_radius: f64) -> Self {
        Self {
            region,
            filter_radius,
            beta: 1.0,
            eta: 0.5,
            iteration: 0,
            fom_history: Vec::new(),
        }
    }

    /// Apply Gaussian density filter to smooth the design variables.
    ///
    /// Each pixel ρ̃(i,j) = Σ_{k,l} w(i-k, j-l) · ρ(k,l)
    /// where w is a Gaussian kernel with radius r_filter.
    pub fn filter_density(&self) -> Vec<f64> {
        let nx = self.region.nx;
        let nz = self.region.nz;
        let r = self.filter_radius;
        let mut filtered = vec![0.0; nx * nz];

        for j in 0..nz {
            for i in 0..nx {
                let mut weight_sum = 0.0;
                let mut val_sum = 0.0;
                // Gaussian kernel
                let r_int = r.ceil() as i64;
                for dj in -r_int..=r_int {
                    for di in -r_int..=r_int {
                        let dist2 = (di * di + dj * dj) as f64;
                        if dist2 > r * r {
                            continue;
                        }
                        let ni = i as i64 + di;
                        let nj = j as i64 + dj;
                        if ni < 0 || ni >= nx as i64 || nj < 0 || nj >= nz as i64 {
                            continue;
                        }
                        let w = (-(dist2) / (r * r)).exp();
                        weight_sum += w;
                        val_sum += w * self.region.rho[nj as usize * nx + ni as usize];
                    }
                }
                filtered[j * nx + i] = val_sum / weight_sum.max(1e-30);
            }
        }
        filtered
    }

    /// Adjoint of the density filter: `(F^T u)[i] = Σ_j W_{ij} u[j] / w[j]`
    ///
    /// Because the kernel W is symmetric (`W_{ij} = W_{ji}`), the adjoint filter
    /// is computed by swapping the roles of source and target in the convolution
    /// while dividing by the *source* pixel's weight sum `w[j]` rather than the
    /// current pixel's weight sum.
    ///
    /// This is the correct transpose of `filter_density` and is needed for the
    /// full chain-rule when converting a gradient from filtered-density space
    /// back to raw-density space.
    pub fn filter_adjoint(&self, du_dfiltered: &[f64]) -> Vec<f64> {
        let nx = self.region.nx;
        let nz = self.region.nz;
        let r = self.filter_radius;
        let r_int = r.ceil() as i64;

        // Pre-compute weight sums for every pixel (same kernel as filter_density)
        let mut weight_sums = vec![0.0_f64; nx * nz];
        for j in 0..nz {
            for i in 0..nx {
                let mut ws = 0.0_f64;
                for dj in -r_int..=r_int {
                    for di in -r_int..=r_int {
                        let dist2 = (di * di + dj * dj) as f64;
                        if dist2 > r * r {
                            continue;
                        }
                        let ni = i as i64 + di;
                        let nj = j as i64 + dj;
                        if ni < 0 || ni >= nx as i64 || nj < 0 || nj >= nz as i64 {
                            continue;
                        }
                        ws += (-(dist2) / (r * r)).exp();
                    }
                }
                weight_sums[j * nx + i] = ws.max(1e-30);
            }
        }

        // Adjoint: for each raw pixel i, accumulate contributions from all
        // filtered pixels j that used pixel i in their convolution.
        // (F^T u)[i] = Σ_j  W_{ij} * u[j] / w[j]
        // Because W is symmetric W_{ij} = W_{ji}, we iterate over j's neighbourhood
        // and contribute to i.  Equivalently, loop over i and its neighbourhood j:
        //   for each neighbour j of i:  result[i] += W_{ij} * u[j] / w[j]
        let mut result = vec![0.0_f64; nx * nz];
        for j in 0..nz {
            for i in 0..nx {
                let mut acc = 0.0_f64;
                for dj in -r_int..=r_int {
                    for di in -r_int..=r_int {
                        let dist2 = (di * di + dj * dj) as f64;
                        if dist2 > r * r {
                            continue;
                        }
                        // neighbour pixel (ni, nj) — this is the "j" in (F^T u)[i]
                        let ni = i as i64 + di;
                        let nj = j as i64 + dj;
                        if ni < 0 || ni >= nx as i64 || nj < 0 || nj >= nz as i64 {
                            continue;
                        }
                        let w = (-(dist2) / (r * r)).exp();
                        let src_idx = nj as usize * nx + ni as usize;
                        acc += w * du_dfiltered[src_idx] / weight_sums[src_idx];
                    }
                }
                result[j * nx + i] = acc;
            }
        }
        result
    }

    /// Jacobian of the Heaviside projection `dρ̄/dρ̃` at each pixel.
    ///
    ///   `dρ̄/dρ̃ = β · sech²(β·(ρ̃ - η)) / (tanh(β·η) + tanh(β·(1-η)))`
    ///
    /// where `sech²(x) = 1 / cosh²(x)`.
    pub fn projection_jacobian(&self, filtered: &[f64]) -> Vec<f64> {
        let beta = self.beta;
        let eta = self.eta;
        let denom = (beta * eta).tanh() + (beta * (1.0 - eta)).tanh();
        filtered
            .iter()
            .map(|&rho_tilde| {
                let cx = (beta * (rho_tilde - eta)).cosh();
                // sech²(x) = 1/cosh²(x)
                beta / (cx * cx * denom)
            })
            .collect()
    }

    /// Convert gradient w.r.t. projected density (`dFOM/dρ̄`) to gradient
    /// w.r.t. raw density (`dFOM/dρ_raw`) via the full chain rule:
    ///
    ///   `dFOM/dρ_raw = F^T(dρ̄/dρ̃ ⊙ dFOM/dρ̄)`
    ///
    /// where `⊙` is element-wise multiplication, `F^T` is the adjoint filter,
    /// and `dρ̄/dρ̃` is the projection Jacobian.
    pub fn raw_gradient(&self, grad_projected: &[f64]) -> Vec<f64> {
        let filtered = self.filter_density();
        let proj_jac = self.projection_jacobian(&filtered);
        // dFOM/dρ̃[i] = proj_jac[i] * grad_projected[i]
        let d_filtered: Vec<f64> = proj_jac
            .iter()
            .zip(grad_projected.iter())
            .map(|(j, g)| j * g)
            .collect();
        self.filter_adjoint(&d_filtered)
    }

    /// Apply one Optimality Criteria update for a given Lagrange multiplier λ.
    ///
    /// Helper for `oc_step` bisection — does NOT mutate `self`.
    fn apply_oc_update(&self, gradient: &[f64], lambda: f64, move_limit: f64) -> Vec<f64> {
        const RHO_MIN: f64 = 1e-3;
        const RHO_MAX: f64 = 1.0;
        self.region
            .rho
            .iter()
            .zip(gradient.iter())
            .map(|(&rho_i, &g_i)| {
                let be = (g_i / lambda).max(0.0).sqrt();
                let rho_candidate = rho_i * be;
                let rho_clamped_move = rho_candidate.clamp(rho_i - move_limit, rho_i + move_limit);
                rho_clamped_move.clamp(RHO_MIN, RHO_MAX)
            })
            .collect()
    }

    /// Compute the projected volume fraction that would result from applying the
    /// OC update with Lagrange multiplier `lambda`.  Does NOT mutate `self`.
    fn volume_fraction_for_lambda(&self, gradient: &[f64], lambda: f64, move_limit: f64) -> f64 {
        let rho_new = self.apply_oc_update(gradient, lambda, move_limit);
        projected_volume(
            &rho_new,
            self.beta,
            self.eta,
            self.filter_radius,
            self.region.nx,
            self.region.nz,
        )
    }

    /// Perform one Optimality Criteria step with volume constraint via bisection.
    ///
    /// `gradient` should be `dFOM/dρ_raw` (positive = increasing FOM with increasing ρ).
    /// `target_volume` is the desired mean projected density V* ∈ (0, 1).
    /// `move_limit` is the maximum allowed change per pixel per step (e.g., 0.2).
    ///
    /// Returns `Ok(new_mean_projected_volume)` on convergence, or an error if the
    /// target volume lies outside the range achievable by the update rule.
    ///
    /// Algorithm (Bendsøe & Sigmund 2003):
    ///   `ρ_new[i] = clamp(clamp(ρ[i] * sqrt(max(0, gradient[i]/λ)), ρ[i]-m, ρ[i]+m), ρ_min, ρ_max)`
    ///   Bisect λ until |mean(project(filter(ρ_new))) - V*| < 1e-4
    pub fn oc_step(
        &mut self,
        gradient: &[f64],
        target_volume: f64,
        move_limit: f64,
    ) -> Result<f64, crate::error::OxiPhotonError> {
        const BISECT_TOL: f64 = 1e-4;
        const MAX_ITER: usize = 50;

        let mut lambda_lo = 1e-9_f64;
        let mut lambda_hi = 1e9_f64;

        let vol_lo = self.volume_fraction_for_lambda(gradient, lambda_lo, move_limit);
        let vol_hi = self.volume_fraction_for_lambda(gradient, lambda_hi, move_limit);

        // vol decreases as lambda increases (higher λ penalises expansion more)
        // so vol_lo >= vol_hi should hold; target must lie in [vol_hi, vol_lo].
        if !(vol_hi <= target_volume + BISECT_TOL && vol_lo >= target_volume - BISECT_TOL) {
            // Clamp to the nearest achievable volume rather than hard-fail
            let rho_new = if (target_volume - vol_hi).abs() < (target_volume - vol_lo).abs() {
                self.apply_oc_update(gradient, lambda_hi, move_limit)
            } else {
                self.apply_oc_update(gradient, lambda_lo, move_limit)
            };
            let actual_vol = projected_volume(
                &rho_new,
                self.beta,
                self.eta,
                self.filter_radius,
                self.region.nx,
                self.region.nz,
            );
            self.region.rho = rho_new;
            self.iteration += 1;
            return Err(crate::error::OxiPhotonError::Convergence(format!(
                "OC bisection failed to bracket target volume {target_volume:.4}; \
                 achievable range [{vol_hi:.4}, {vol_lo:.4}], applied nearest ({actual_vol:.4})"
            )));
        }

        // Bisect
        let mut lambda_mid = lambda_lo;
        for _ in 0..MAX_ITER {
            lambda_mid = (lambda_lo + lambda_hi) / 2.0;
            let vol_mid = self.volume_fraction_for_lambda(gradient, lambda_mid, move_limit);
            if (vol_mid - target_volume).abs() < BISECT_TOL {
                break;
            }
            // Higher lambda → lower volume fraction
            if vol_mid > target_volume {
                lambda_lo = lambda_mid;
            } else {
                lambda_hi = lambda_mid;
            }
        }

        let rho_new = self.apply_oc_update(gradient, lambda_mid, move_limit);
        let final_vol = projected_volume(
            &rho_new,
            self.beta,
            self.eta,
            self.filter_radius,
            self.region.nx,
            self.region.nz,
        );
        self.region.rho = rho_new;
        self.iteration += 1;
        Ok(final_vol)
    }

    /// Update beta from the continuation schedule based on the current iteration.
    ///
    /// Scans the schedule (a list of `(iteration_start, beta)` pairs) in reverse
    /// order and applies the first entry whose `iteration_start ≤ self.iteration`.
    pub fn apply_continuation(&mut self, schedule: &[(usize, f64)]) {
        for &(start, beta) in schedule.iter().rev() {
            if self.iteration >= start {
                self.beta = beta;
                break;
            }
        }
    }

    /// Apply Heaviside projection to binarise filtered density.
    ///
    ///   ρ̄ = tanh(β·η) + tanh(β·(ρ̃ - η)) / tanh(β·η) + tanh(β·(1-η))  (normalised)
    pub fn project_density(&self, filtered: &[f64]) -> Vec<f64> {
        let beta = self.beta;
        let eta = self.eta;
        let tanh_b_eta = (beta * eta).tanh();
        let denom = tanh_b_eta + (beta * (1.0 - eta)).tanh();
        filtered
            .iter()
            .map(|&rho| (tanh_b_eta + (beta * (rho - eta)).tanh()) / denom)
            .collect()
    }

    /// Minimum length scale (m) enforced by the filter.
    ///
    ///   l_min ≈ 2 · r_filter · dx
    pub fn min_feature_size(&self) -> f64 {
        2.0 * self.filter_radius * self.region.dx
    }

    /// Gradually increase projection steepness β (continuation method).
    ///
    /// Called every n_increase iterations to sharpen projection.
    pub fn increase_beta(&mut self, factor: f64) {
        self.beta *= factor;
    }

    /// Fraction of pixels that are "binarised" (|ρ - 0.5| > threshold).
    pub fn binarisation_fraction(&self, threshold: f64) -> f64 {
        let n_binary = self
            .region
            .rho
            .iter()
            .filter(|&&r| (r - 0.5).abs() > threshold)
            .count();
        n_binary as f64 / self.region.n_params() as f64
    }

    /// Check if design is sufficiently binarised for fabrication.
    pub fn is_binarised(&self) -> bool {
        self.binarisation_fraction(0.45) > 0.95
    }

    /// Update design variables using the full chain rule through filter and projection.
    ///
    /// `dfom_drbar`: gradient of the figure-of-merit w.r.t. the *projected* density ρ̄.
    /// The chain rule through filter and projection is applied internally:
    ///   dFOM/dρ_raw = F^T( dρ̄/dρ̃ ⊙ dFOM/dρ̄ )
    pub fn step(&mut self, dfom_drbar: &[f64], step_size: f64) {
        let filtered = self.filter_density();
        let dproj = self.projection_jacobian(&filtered);
        let dfom_drtilde: Vec<f64> = dfom_drbar
            .iter()
            .zip(dproj.iter())
            .map(|(g, j)| g * j)
            .collect();
        let dfom_drho_raw = self.filter_adjoint(&dfom_drtilde);
        for (rho, &g) in self.region.rho.iter_mut().zip(dfom_drho_raw.iter()) {
            *rho = (*rho + step_size * g).clamp(0.0, 1.0);
        }
        self.iteration += 1;
    }

    /// Update design variables using a pre-computed raw-density gradient
    /// (caller has already applied the chain rule through filter + projection).
    ///
    /// This is the legacy "simplified" update — use `step()` for the correct
    /// chain-rule version.
    pub fn step_with_raw_gradient(&mut self, raw_gradient: &[f64], step_size: f64) {
        for (rho, &g) in self.region.rho.iter_mut().zip(raw_gradient.iter()) {
            *rho = (*rho + step_size * g).clamp(0.0, 1.0);
        }
        self.iteration += 1;
    }

    /// Log current FOM value.
    pub fn record_fom(&mut self, fom: f64) {
        self.fom_history.push(fom);
    }

    /// Best FOM achieved so far.
    pub fn best_fom(&self) -> f64 {
        self.fom_history
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Check convergence: FOM change over last n_window iterations < tolerance.
    pub fn is_converged(&self, n_window: usize, tolerance: f64) -> bool {
        if self.fom_history.len() < n_window {
            return false;
        }
        let n = self.fom_history.len();
        let recent: &[f64] = &self.fom_history[n - n_window..];
        let max = recent.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = recent.iter().cloned().fold(f64::INFINITY, f64::min);
        (max - min).abs() < tolerance
    }
}

/// Binary projection utility for post-processing continuous designs.
pub struct BinaryProjection;

impl BinaryProjection {
    /// Hard threshold: ρ > 0.5 → 1.0, else → 0.0.
    pub fn hard_threshold(rho: &[f64]) -> Vec<f64> {
        rho.iter()
            .map(|&r| if r > 0.5 { 1.0 } else { 0.0 })
            .collect()
    }

    /// Erode a binary design (shrink material regions by 1 pixel).
    pub fn erode(binary: &[f64], nx: usize, nz: usize) -> Vec<f64> {
        let mut result = binary.to_vec();
        for j in 0..nz {
            for i in 0..nx {
                if binary[j * nx + i] < 0.5 {
                    continue; // already void
                }
                // Check 4-connected neighbours
                let is_edge = (i == 0)
                    || (i == nx - 1)
                    || (j == 0)
                    || (j == nz - 1)
                    || (i > 0 && binary[j * nx + (i - 1)] < 0.5)
                    || (i < nx - 1 && binary[j * nx + (i + 1)] < 0.5)
                    || (j > 0 && binary[(j - 1) * nx + i] < 0.5)
                    || (j < nz - 1 && binary[(j + 1) * nx + i] < 0.5);
                if is_edge {
                    result[j * nx + i] = 0.0;
                }
            }
        }
        result
    }

    /// Dilate a binary design (grow material regions by 1 pixel).
    pub fn dilate(binary: &[f64], nx: usize, nz: usize) -> Vec<f64> {
        let mut result = binary.to_vec();
        for j in 0..nz {
            for i in 0..nx {
                if binary[j * nx + i] > 0.5 {
                    continue; // already material
                }
                let has_neighbor = (i > 0 && binary[j * nx + (i - 1)] > 0.5)
                    || (i < nx - 1 && binary[j * nx + (i + 1)] > 0.5)
                    || (j > 0 && binary[(j - 1) * nx + i] > 0.5)
                    || (j < nz - 1 && binary[(j + 1) * nx + i] > 0.5);
                if has_neighbor {
                    result[j * nx + i] = 1.0;
                }
            }
        }
        result
    }

    /// Opening = erosion then dilation (removes small features).
    pub fn opening(binary: &[f64], nx: usize, nz: usize) -> Vec<f64> {
        let eroded = Self::erode(binary, nx, nz);
        Self::dilate(&eroded, nx, nz)
    }

    /// Closing = dilation then erosion (fills small gaps).
    pub fn closing(binary: &[f64], nx: usize, nz: usize) -> Vec<f64> {
        let dilated = Self::dilate(binary, nx, nz);
        Self::erode(&dilated, nx, nz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_preserves_uniform_design() {
        let region = DesignRegion::new(8, 8, 20e-9, 1.0, 4.0);
        // Uniform ρ=0.5 should remain 0.5 after filtering
        let opt = TopologyOptimizer::new(region, 2.0);
        let filtered = opt.filter_density();
        for &f in &filtered {
            assert!((f - 0.5).abs() < 1e-6, "f={f:.4}");
        }
    }

    #[test]
    fn projection_sharpens_near_boundary() {
        let region = DesignRegion::new(4, 1, 20e-9, 1.0, 4.0);
        let mut opt = TopologyOptimizer::new(region, 1.0);
        opt.beta = 10.0;
        let filtered = vec![0.1, 0.4, 0.6, 0.9];
        let projected = opt.project_density(&filtered);
        // After projection with β=10, values near 0 and 1
        assert!(projected[0] < 0.2);
        assert!(projected[3] > 0.8);
    }

    #[test]
    fn min_feature_size_physical() {
        let region = DesignRegion::new(10, 10, 20e-9, 1.0, 4.0);
        let opt = TopologyOptimizer::new(region, 2.0);
        let mfs = opt.min_feature_size();
        // 2 * 2.0 * 20nm = 80nm
        assert!((mfs - 80e-9).abs() < 1e-12);
    }

    #[test]
    fn binary_hard_threshold() {
        let rho = vec![0.2, 0.5, 0.7, 0.1, 0.9];
        let binary = BinaryProjection::hard_threshold(&rho);
        assert_eq!(binary[0], 0.0);
        assert_eq!(binary[2], 1.0);
        assert_eq!(binary[4], 1.0);
    }

    #[test]
    fn erode_reduces_fill() {
        let nx = 5;
        let nz = 5;
        let binary = vec![1.0; nx * nz]; // all material
        let eroded = BinaryProjection::erode(&binary, nx, nz);
        let fill_before = binary.iter().sum::<f64>() / (nx * nz) as f64;
        let fill_after = eroded.iter().sum::<f64>() / (nx * nz) as f64;
        assert!(fill_after < fill_before);
    }

    #[test]
    fn dilate_increases_fill() {
        let nx = 5;
        let nz = 5;
        let mut binary = vec![0.0; nx * nz];
        binary[2 * nx + 2] = 1.0; // single pixel
        let dilated = BinaryProjection::dilate(&binary, nx, nz);
        let fill = dilated.iter().sum::<f64>();
        assert!(fill > 1.0); // more than one pixel
    }

    #[test]
    fn convergence_detection() {
        let region = DesignRegion::new(4, 4, 20e-9, 1.0, 4.0);
        let mut opt = TopologyOptimizer::new(region, 1.0);
        for _ in 0..10 {
            opt.record_fom(1.0 + 1e-10); // effectively constant
        }
        assert!(opt.is_converged(5, 1e-6));
    }

    #[test]
    fn binarisation_fraction_initial() {
        let region = DesignRegion::new(10, 10, 20e-9, 1.0, 4.0);
        let opt = TopologyOptimizer::new(region, 1.0);
        // All rho = 0.5, none exceed threshold 0.45
        assert_eq!(opt.binarisation_fraction(0.45), 0.0);
    }
}
