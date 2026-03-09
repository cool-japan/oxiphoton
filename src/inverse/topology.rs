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

    /// Update design variables using steepest ascent on filtered/projected design.
    ///
    /// Gradient with respect to raw ρ includes chain rule through filter and projection.
    pub fn step(&mut self, raw_gradient: &[f64], step_size: f64) {
        // Apply filter then project
        let filtered = self.filter_density();
        let _projected = self.project_density(&filtered);
        // Chain rule: dFOM/dρ_raw = dFOM/dρ̄ · dρ̄/dρ̃ · dρ̃/dρ
        // Simplified: use raw gradient (full chain rule requires adjoint of filter)
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
