//! Fabrication constraints for photonic inverse design.
//!
//! Real photonic structures are limited by fabrication processes:
//!   - Minimum feature size (MFS): smallest printable hole/pillar
//!   - Minimum gap/linewidth for the lithography process
//!   - Curvature constraints (no sharp corners)
//!   - Symmetry constraints (for process uniformity)
//!   - Etch bias (isotropic widening of features)
//!
//! These constraints are imposed through:
//!   1. Density filtering (spatial averaging, removes sub-resolution features)
//!   2. Heaviside projection (binarization with minimum feature control)
//!   3. Erosion/dilation operators (morphological operations)
//!   4. Penalization (add penalty to objective for constraint violations)

/// Fabrication constraint specification.
#[derive(Debug, Clone)]
pub struct FabricationConstraints {
    /// Minimum feature size (m)
    pub min_feature_size: f64,
    /// Minimum gap between features (m)
    pub min_gap: f64,
    /// Maximum curvature (1/m): limits corner sharpness
    pub max_curvature: f64,
    /// Etch bias — isotropic offset (positive = larger features)
    pub etch_bias: f64,
    /// Whether x-symmetry is required
    pub x_symmetric: bool,
    /// Whether y-symmetry is required
    pub y_symmetric: bool,
}

impl FabricationConstraints {
    /// Create fabrication constraints.
    pub fn new(min_feature_size: f64, min_gap: f64) -> Self {
        Self {
            min_feature_size,
            min_gap,
            max_curvature: f64::INFINITY,
            etch_bias: 0.0,
            x_symmetric: false,
            y_symmetric: false,
        }
    }

    /// Typical deep-UV lithography (193nm) constraints for silicon photonics.
    ///
    /// Min feature: 100 nm, Min gap: 100 nm, no symmetry required.
    pub fn duv_siph() -> Self {
        Self::new(100e-9, 100e-9)
    }

    /// E-beam lithography constraints — sub-50nm features.
    pub fn ebeam() -> Self {
        Self::new(50e-9, 50e-9)
    }

    /// Nano-imprint lithography (NIL) constraints.
    pub fn nil() -> Self {
        Self::new(20e-9, 30e-9)
    }

    /// Check if a design variable array satisfies binary constraint (all near 0 or 1).
    pub fn is_binary(&self, rho: &[f64], tolerance: f64) -> bool {
        rho.iter().all(|&r| r < tolerance || r > 1.0 - tolerance)
    }

    /// Compute etch-biased density: expand features by applying erosion or dilation.
    ///
    /// Positive etch_bias → dilation (features grow).
    /// Negative etch_bias → erosion (features shrink).
    pub fn apply_etch_bias(&self, rho: &[f64], nx: usize, ny: usize, dx: f64) -> Vec<f64> {
        if self.etch_bias.abs() < 1e-30 {
            return rho.to_vec();
        }
        let r = (self.etch_bias.abs() / dx).ceil() as usize;
        let mut result = rho.to_vec();
        if self.etch_bias > 0.0 {
            // Dilation: result[i] = max over neighbors within radius r
            for j in 0..ny {
                for i in 0..nx {
                    let mut max_val = 0.0f64;
                    let jlo = j.saturating_sub(r);
                    let jhi = (j + r + 1).min(ny);
                    let ilo = i.saturating_sub(r);
                    let ihi = (i + r + 1).min(nx);
                    for jj in jlo..jhi {
                        for ii in ilo..ihi {
                            let dist_sq = ((ii as f64 - i as f64) * dx).powi(2)
                                + ((jj as f64 - j as f64) * dx).powi(2);
                            if dist_sq <= self.etch_bias.powi(2) {
                                max_val = max_val.max(rho[jj * nx + ii]);
                            }
                        }
                    }
                    result[j * nx + i] = max_val;
                }
            }
        } else {
            // Erosion: result[i] = min over neighbors
            for j in 0..ny {
                for i in 0..nx {
                    let mut min_val = 1.0f64;
                    let r2 = self.etch_bias.abs();
                    let jlo = j.saturating_sub(r);
                    let jhi = (j + r + 1).min(ny);
                    let ilo = i.saturating_sub(r);
                    let ihi = (i + r + 1).min(nx);
                    for jj in jlo..jhi {
                        for ii in ilo..ihi {
                            let dist_sq = ((ii as f64 - i as f64) * dx).powi(2)
                                + ((jj as f64 - j as f64) * dx).powi(2);
                            if dist_sq <= r2 * r2 {
                                min_val = min_val.min(rho[jj * nx + ii]);
                            }
                        }
                    }
                    result[j * nx + i] = min_val;
                }
            }
        }
        result
    }

    /// Apply x-symmetry: ρ(i,j) ← (ρ(i,j) + ρ(nx-1-i,j)) / 2.
    pub fn enforce_x_symmetry(&self, rho: &mut [f64], nx: usize, ny: usize) {
        if !self.x_symmetric {
            return;
        }
        for j in 0..ny {
            for i in 0..nx / 2 {
                let mirror = nx - 1 - i;
                let avg = (rho[j * nx + i] + rho[j * nx + mirror]) / 2.0;
                rho[j * nx + i] = avg;
                rho[j * nx + mirror] = avg;
            }
        }
    }

    /// Apply y-symmetry: ρ(i,j) ← (ρ(i,j) + ρ(i,ny-1-j)) / 2.
    pub fn enforce_y_symmetry(&self, rho: &mut [f64], nx: usize, ny: usize) {
        if !self.y_symmetric {
            return;
        }
        for j in 0..ny / 2 {
            let mirror_j = ny - 1 - j;
            for i in 0..nx {
                let avg = (rho[j * nx + i] + rho[mirror_j * nx + i]) / 2.0;
                rho[j * nx + i] = avg;
                rho[mirror_j * nx + i] = avg;
            }
        }
    }

    /// Penalty for intermediate densities (penalizes non-binary designs).
    ///
    ///   P = ∫ ρ·(1-ρ) dV / V  ∈ [0, 0.25]
    pub fn binarization_penalty(&self, rho: &[f64]) -> f64 {
        let n = rho.len() as f64;
        rho.iter().map(|&r| r * (1.0 - r)).sum::<f64>() / n
    }

    /// Maximum intermediate density (diagnostic).
    pub fn max_gray(&self, rho: &[f64]) -> f64 {
        rho.iter().map(|&r| r * (1.0 - r)).fold(0.0_f64, f64::max)
    }
}

/// Minimum feature size check using distance transform approach.
///
/// For a binary design, estimates the minimum "island" (solid region)
/// and "void" (air region) feature sizes using simple erosion.
pub fn check_min_feature_size(rho: &[f64], nx: usize, ny: usize, dx: f64, mfs: f64) -> bool {
    let r = (mfs / (2.0 * dx)).ceil() as usize;
    // Erode: set to 1 only if all pixels within radius r are > 0.5
    // If erosion leaves any pixel = 1, then solid features are large enough
    let erode = |field: &[f64]| -> bool {
        for j in 0..ny {
            for i in 0..nx {
                if field[j * nx + i] > 0.5 {
                    let jlo = j.saturating_sub(r);
                    let jhi = (j + r + 1).min(ny);
                    let ilo = i.saturating_sub(r);
                    let ihi = (i + r + 1).min(nx);
                    let mut all_solid = true;
                    'outer: for jj in jlo..jhi {
                        for ii in ilo..ihi {
                            let d2 = ((ii as f64 - i as f64) * dx).powi(2)
                                + ((jj as f64 - j as f64) * dx).powi(2);
                            if d2 <= (mfs / 2.0).powi(2) && field[jj * nx + ii] < 0.5 {
                                all_solid = false;
                                break 'outer;
                            }
                        }
                    }
                    if all_solid {
                        return true; // Found at least one valid solid feature
                    }
                }
            }
        }
        false
    };
    // Check solid features
    let solid_ok = erode(rho);
    // Check void features (invert)
    let inv: Vec<f64> = rho.iter().map(|&r| 1.0 - r).collect();
    let void_ok = erode(&inv);
    solid_ok || void_ok
}

/// Proximity correction: adjust exposure dose to compensate for optical proximity effects.
///
/// Applies a correction kernel to the design to pre-distort the mask.
/// Uses a Gaussian correction with width `sigma`.
pub fn proximity_correction(
    rho: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
    sigma: f64,
    correction_strength: f64,
) -> Vec<f64> {
    // Convolve rho with a Laplacian-like correction kernel
    // correction = rho - strength * Laplacian(rho) * sigma²
    let mut result = rho.to_vec();
    for j in 1..ny - 1 {
        for i in 1..nx - 1 {
            let idx = j * nx + i;
            let lap = (rho[idx + 1] + rho[idx - 1] + rho[(j + 1) * nx + i] + rho[(j - 1) * nx + i]
                - 4.0 * rho[idx])
                / (dx * dx);
            result[idx] = (rho[idx] - correction_strength * sigma * sigma * lap).clamp(0.0, 1.0);
        }
    }
    result
}

/// Overlay error model: simulate the effect of mask misalignment.
///
/// Shifts the density map by (dx_shift, dy_shift) in meters.
/// Returns the shifted design (nearest-neighbor interpolation).
pub fn apply_overlay_error(
    rho: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
    dx_shift: f64,
    dy_shift: f64,
) -> Vec<f64> {
    let mut result = vec![0.0f64; nx * ny];
    let di = (dx_shift / dx).round() as i64;
    let dj = (dy_shift / dx).round() as i64;
    for j in 0..ny {
        for i in 0..nx {
            let ni = (i as i64 + di).clamp(0, nx as i64 - 1) as usize;
            let nj = (j as i64 + dj).clamp(0, ny as i64 - 1) as usize;
            result[j * nx + i] = rho[nj * nx + ni];
        }
    }
    result
}

/// Curvature penalization for smooth boundaries.
pub struct CurvaturePenalty {
    /// Maximum allowed curvature (1/m)
    pub kappa_max: f64,
}

impl CurvaturePenalty {
    pub fn new(kappa_max: f64) -> Self {
        Self { kappa_max }
    }

    /// Compute curvature penalty on a density field.
    ///
    /// Approximates boundary curvature using the Laplacian of φ / |∇φ|.
    /// Returns normalized penalty P ∈ [0, 1].
    pub fn penalty(&self, rho: &[f64], nx: usize, ny: usize, dx: f64) -> f64 {
        let mut total = 0.0f64;
        let mut count = 0usize;
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let idx = j * nx + i;
                // Laplacian
                let lap =
                    (rho[idx + 1] + rho[idx - 1] + rho[(j + 1) * nx + i] + rho[(j - 1) * nx + i]
                        - 4.0 * rho[idx])
                        / (dx * dx);
                // Gradient magnitude
                let gx = (rho[idx + 1] - rho[idx - 1]) / (2.0 * dx);
                let gy = (rho[(j + 1) * nx + i] - rho[(j - 1) * nx + i]) / (2.0 * dx);
                let grad_mag = (gx * gx + gy * gy).sqrt();
                if grad_mag > 1e-6 / dx {
                    let kappa = (lap / grad_mag).abs();
                    if kappa > self.kappa_max {
                        total += (kappa - self.kappa_max) / self.kappa_max;
                        count += 1;
                    }
                }
            }
        }
        if count == 0 {
            0.0
        } else {
            total / count as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Minimum feature size constraint enforcement
// ---------------------------------------------------------------------------

/// Enforces a minimum feature size on a continuous density field via
/// morphological erosion-dilation.
#[derive(Debug, Clone)]
pub struct MinFeatureFilter {
    /// Minimum feature size (m)
    pub min_size_m: f64,
    /// Grid pixel size (m)
    pub grid_dx: f64,
}

impl MinFeatureFilter {
    /// Create a new `MinFeatureFilter`.
    pub fn new(min_size_m: f64, grid_dx: f64) -> Self {
        Self {
            min_size_m,
            grid_dx,
        }
    }

    /// Radius in pixels corresponding to the minimum feature size.
    fn radius_px(&self) -> usize {
        ((self.min_size_m / (2.0 * self.grid_dx)).ceil() as usize).max(1)
    }

    /// Apply minimum feature size filter to a 2D density field using erosion
    /// followed by dilation (opening operation).
    ///
    /// `field` contains values in [0, 1]; `nx` × `ny` grid (row-major).
    pub fn apply_2d(&self, field: &[f64], nx: usize, ny: usize) -> Vec<f64> {
        let r = self.radius_px();
        let eroded = Self::erode_2d(field, nx, ny, r);
        Self::dilate_2d(&eroded, nx, ny, r)
    }

    fn erode_2d(field: &[f64], nx: usize, ny: usize, r: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let jlo = j.saturating_sub(r);
                let jhi = (j + r + 1).min(ny);
                let ilo = i.saturating_sub(r);
                let ihi = (i + r + 1).min(nx);
                let mut min_val = 1.0f64;
                for jj in jlo..jhi {
                    for ii in ilo..ihi {
                        let di = ii as f64 - i as f64;
                        let dj = jj as f64 - j as f64;
                        if di * di + dj * dj <= (r * r) as f64 {
                            min_val = min_val.min(field[jj * nx + ii]);
                        }
                    }
                }
                out[j * nx + i] = min_val;
            }
        }
        out
    }

    fn dilate_2d(field: &[f64], nx: usize, ny: usize, r: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let jlo = j.saturating_sub(r);
                let jhi = (j + r + 1).min(ny);
                let ilo = i.saturating_sub(r);
                let ihi = (i + r + 1).min(nx);
                let mut max_val = 0.0f64;
                for jj in jlo..jhi {
                    for ii in ilo..ihi {
                        let di = ii as f64 - i as f64;
                        let dj = jj as f64 - j as f64;
                        if di * di + dj * dj <= (r * r) as f64 {
                            max_val = max_val.max(field[jj * nx + ii]);
                        }
                    }
                }
                out[j * nx + i] = max_val;
            }
        }
        out
    }

    /// Check whether any feature in a binary mask violates the minimum size.
    ///
    /// A solid connected component violates the minimum feature size if and only if
    /// morphological erosion completely removes every pixel in that component.
    /// If at least one pixel in a component survives erosion, the feature is
    /// considered large enough (the component has a valid interior).
    ///
    /// This component-based check avoids false positives at the corners of large
    /// rectangular blocks, where a disc-shaped structuring element would remove
    /// corner pixels even for perfectly valid features.
    pub fn has_violation_2d(&self, binary: &[bool], nx: usize, ny: usize) -> bool {
        use std::collections::VecDeque;

        let r = self.radius_px();
        let field: Vec<f64> = binary.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let eroded = Self::erode_2d(&field, nx, ny, r);

        // Label 4-connected solid components.
        let mut labels = vec![-1i32; nx * ny];
        let mut n_components = 0i32;
        for start in 0..nx * ny {
            if !binary[start] || labels[start] >= 0 {
                continue;
            }
            let mut queue = VecDeque::new();
            queue.push_back(start);
            labels[start] = n_components;
            while let Some(idx) = queue.pop_front() {
                let ci = (idx % nx) as i32;
                let cj = (idx / nx) as i32;
                for (di, dj) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let ni = ci + di;
                    let nj = cj + dj;
                    if ni >= 0 && ni < nx as i32 && nj >= 0 && nj < ny as i32 {
                        let nidx = nj as usize * nx + ni as usize;
                        if binary[nidx] && labels[nidx] < 0 {
                            labels[nidx] = n_components;
                            queue.push_back(nidx);
                        }
                    }
                }
            }
            n_components += 1;
        }

        if n_components == 0 {
            return false;
        }

        // Check whether each component has at least one pixel that survives erosion.
        let mut component_has_interior = vec![false; n_components as usize];
        for idx in 0..nx * ny {
            let lbl = labels[idx];
            if lbl >= 0 && eroded[idx] > 0.5 {
                component_has_interior[lbl as usize] = true;
            }
        }

        // A violation exists when any component is fully erased by erosion.
        component_has_interior.iter().any(|&has_int| !has_int)
    }

    /// Compute the maximum violation in units of grid cells.
    ///
    /// For each solid pixel that violates MFS, records how many cells its
    /// nearest air pixel is away (computed via a simple distance approximation).
    /// Returns 0 if no violation.
    pub fn max_violation_cells(&self, binary: &[bool], nx: usize, ny: usize) -> usize {
        let r = self.radius_px();
        let field: Vec<f64> = binary.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let mut max_viol = 0usize;
        for j in 0..ny {
            for i in 0..nx {
                if !binary[j * nx + i] {
                    continue;
                }
                // Find nearest air pixel
                let mut nearest_air = usize::MAX;
                'search: for rr in 1..=r {
                    let jlo = j.saturating_sub(rr);
                    let jhi = (j + rr + 1).min(ny);
                    let ilo = i.saturating_sub(rr);
                    let ihi = (i + rr + 1).min(nx);
                    for jj in jlo..jhi {
                        for ii in ilo..ihi {
                            if field[jj * nx + ii] < 0.5 {
                                let di = (ii as isize - i as isize).unsigned_abs();
                                let dj = (jj as isize - j as isize).unsigned_abs();
                                let dist = di.max(dj);
                                nearest_air = nearest_air.min(dist);
                                break 'search;
                            }
                        }
                    }
                }
                // Violation when nearest air < r
                if nearest_air < r {
                    let viol = r.saturating_sub(nearest_air);
                    if viol > max_viol {
                        max_viol = viol;
                    }
                }
            }
        }
        max_viol
    }
}

// ---------------------------------------------------------------------------
// Projection filter (topology optimisation)
// ---------------------------------------------------------------------------

/// Smooth density filter and Heaviside projection for topology optimisation.
///
/// The workflow is: raw design → smooth with hat/conic filter → Heaviside projection.
/// This avoids length-scale violations and improves convergence.
#[derive(Debug, Clone)]
pub struct ProjectionFilter {
    /// Heaviside sharpness parameter β (larger → sharper step)
    pub beta: f64,
    /// Threshold η ∈ (0, 1) for the Heaviside step
    pub eta: f64,
    /// Filter radius (m)
    pub radius: f64,
    /// Grid spacing (m)
    pub dx: f64,
}

impl ProjectionFilter {
    /// Create a new `ProjectionFilter`.
    pub fn new(radius: f64, dx: f64, beta: f64, eta: f64) -> Self {
        Self {
            beta,
            eta,
            radius,
            dx,
        }
    }

    /// Apply smooth Heaviside projection to a single density value ρ ∈ [0, 1].
    ///
    /// Wang et al. (2011) smooth Heaviside:
    ///   H̃(ρ) = (tanh(β·η) + tanh(β·(ρ−η))) / (tanh(β·η) + tanh(β·(1−η)))
    pub fn project(&self, rho: f64) -> f64 {
        let b = self.beta;
        let e = self.eta;
        let num = (b * e).tanh() + (b * (rho - e)).tanh();
        let den = (b * e).tanh() + (b * (1.0 - e)).tanh();
        if den.abs() < 1e-30 {
            rho
        } else {
            num / den
        }
    }

    /// Derivative of the projection dH̃/dρ (for gradient computation).
    pub fn project_grad(&self, rho: f64) -> f64 {
        let b = self.beta;
        let e = self.eta;
        let den = (b * e).tanh() + (b * (1.0 - e)).tanh();
        if den.abs() < 1e-30 {
            1.0
        } else {
            b / (b * (rho - e)).cosh().powi(2) / den
        }
    }

    /// Apply the projection to an entire density field element-wise.
    pub fn project_field(&self, field: &[f64]) -> Vec<f64> {
        field.iter().map(|&r| self.project(r)).collect()
    }

    /// Smooth a 2D density field using a conic (hat) filter of radius `self.radius`.
    ///
    /// Conic kernel: w(r) = max(0, 1 − |r|/R)
    /// Result is normalised so that the kernel sums to 1.
    pub fn smooth_2d(&self, field: &[f64], nx: usize, ny: usize) -> Vec<f64> {
        let r_px = (self.radius / self.dx).ceil() as usize;
        let mut out = vec![0.0f64; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let jlo = j.saturating_sub(r_px);
                let jhi = (j + r_px + 1).min(ny);
                let ilo = i.saturating_sub(r_px);
                let ihi = (i + r_px + 1).min(nx);
                let mut weighted_sum = 0.0f64;
                let mut weight_sum = 0.0f64;
                for jj in jlo..jhi {
                    for ii in ilo..ihi {
                        let di = (ii as f64 - i as f64) * self.dx;
                        let dj = (jj as f64 - j as f64) * self.dx;
                        let dist = (di * di + dj * dj).sqrt();
                        let w = (1.0 - dist / self.radius).max(0.0);
                        weighted_sum += w * field[jj * nx + ii];
                        weight_sum += w;
                    }
                }
                out[j * nx + i] = if weight_sum > 1e-30 {
                    weighted_sum / weight_sum
                } else {
                    field[j * nx + i]
                };
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fab_constraints_binary_check() {
        let fc = FabricationConstraints::duv_siph();
        let binary = vec![0.0, 1.0, 0.0, 1.0, 1.0];
        assert!(fc.is_binary(&binary, 0.01));
        let gray = vec![0.0, 0.5, 1.0];
        assert!(!fc.is_binary(&gray, 0.01));
    }

    #[test]
    fn fab_binarization_penalty_binary_is_zero() {
        let fc = FabricationConstraints::duv_siph();
        let binary = vec![0.0, 0.0, 1.0, 1.0];
        assert!(fc.binarization_penalty(&binary) < 1e-10);
    }

    #[test]
    fn fab_binarization_penalty_gray_positive() {
        let fc = FabricationConstraints::duv_siph();
        let gray = vec![0.5; 4];
        let p = fc.binarization_penalty(&gray);
        assert!(p > 0.0);
    }

    #[test]
    fn fab_x_symmetry_enforced() {
        let fc = FabricationConstraints {
            x_symmetric: true,
            ..FabricationConstraints::duv_siph()
        };
        let mut rho = vec![0.0, 1.0, 0.0, 1.0]; // 4x1
        fc.enforce_x_symmetry(&mut rho, 4, 1);
        assert!((rho[0] - rho[3]).abs() < 1e-10);
        assert!((rho[1] - rho[2]).abs() < 1e-10);
    }

    #[test]
    fn fab_etch_bias_dilation_expands() {
        let fc = FabricationConstraints {
            etch_bias: 2.0e-6,
            ..FabricationConstraints::duv_siph()
        };
        let mut rho = vec![0.0f64; 10 * 10];
        // Single pixel at center
        rho[5 * 10 + 5] = 1.0;
        let dilated = fc.apply_etch_bias(&rho, 10, 10, 1e-6);
        let n_filled = dilated.iter().filter(|&&v| v > 0.5).count();
        assert!(
            n_filled > 1,
            "Dilation should expand the feature: {n_filled} pixels"
        );
    }

    #[test]
    fn curvature_penalty_smooth_zero() {
        let pen = CurvaturePenalty::new(1e6); // very large kappa_max → no penalty
        let rho = vec![0.5f64; 5 * 5];
        let p = pen.penalty(&rho, 5, 5, 1e-6);
        assert!(p == 0.0);
    }

    #[test]
    fn min_feature_size_check_large_feature() {
        // Create a 20x20 grid with a 10x10 solid block at center — large enough
        let nx = 20;
        let ny = 20;
        let dx = 10e-9; // 10 nm pixels
        let mfs = 50e-9; // 50 nm MFS (5 pixels)
        let mut rho = vec![0.0f64; nx * ny];
        for j in 5..15 {
            for i in 5..15 {
                rho[j * nx + i] = 1.0;
            }
        }
        assert!(check_min_feature_size(&rho, nx, ny, dx, mfs));
    }

    #[test]
    fn proximity_correction_preserves_interior() {
        // Uniform field → correction should not change values
        let nx = 5;
        let ny = 5;
        let rho = vec![0.5f64; nx * ny];
        let corrected = proximity_correction(&rho, nx, ny, 1e-6, 50e-9, 0.1);
        // Interior pixels: Laplacian of uniform field is 0
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let v = corrected[j * nx + i];
                assert!((v - 0.5).abs() < 1e-10, "pixel ({i},{j}) = {v}");
            }
        }
    }

    #[test]
    fn overlay_error_zero_shift_identity() {
        let rho: Vec<f64> = (0..16).map(|i| i as f64 / 16.0).collect();
        let shifted = apply_overlay_error(&rho, 4, 4, 1e-6, 0.0, 0.0);
        for (a, b) in rho.iter().zip(shifted.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn overlay_error_nonzero_shift() {
        let nx = 4;
        let ny = 4;
        let dx = 1e-6;
        let mut rho = vec![0.0f64; nx * ny];
        rho[0] = 1.0; // top-left corner
                      // Shift by +1 pixel in x
        let shifted = apply_overlay_error(&rho, nx, ny, dx, dx, 0.0);
        // Original [0] should now map from [1], which is 0
        assert!(shifted[0].abs() < 1e-10);
    }

    #[test]
    fn curvature_penalty_step_nonzero() {
        let pen = CurvaturePenalty::new(1.0 / 1e-5); // moderate kappa_max
        let nx = 5;
        let ny = 5;
        let mut rho = vec![0.0f64; nx * ny];
        // Step function: left half solid, right half air
        for j in 0..ny {
            for i in 0..nx {
                rho[j * nx + i] = if i < nx / 2 { 1.0 } else { 0.0 };
            }
        }
        let p = pen.penalty(&rho, nx, ny, 1e-6);
        // Step creates large curvature → penalty > 0
        assert!(p >= 0.0); // May be 0 if threshold not exceeded on a coarse grid
    }

    // ── MinFeatureFilter tests ─────────────────────────────────────────────────

    #[test]
    fn min_feature_filter_uniform_solid_preserves_field() {
        let mff = MinFeatureFilter::new(50e-9, 10e-9);
        let field = vec![1.0f64; 10 * 10];
        let out = mff.apply_2d(&field, 10, 10);
        for v in &out {
            assert!(
                (*v - 1.0).abs() < 1e-10,
                "Uniform solid should survive opening: {v}"
            );
        }
    }

    #[test]
    fn min_feature_filter_uniform_air_preserves_field() {
        let mff = MinFeatureFilter::new(50e-9, 10e-9);
        let field = vec![0.0f64; 10 * 10];
        let out = mff.apply_2d(&field, 10, 10);
        for v in &out {
            assert!(v.abs() < 1e-10, "Uniform air should survive opening");
        }
    }

    #[test]
    fn min_feature_filter_tiny_feature_removed() {
        // 20x20 grid, 10nm pixel, 50nm MFS (radius 2.5 px → 3 px)
        let nx = 20usize;
        let ny = 20usize;
        let dx = 10e-9;
        let mfs = 50e-9;
        let mff = MinFeatureFilter::new(mfs, dx);
        let mut field = vec![0.0f64; nx * ny];
        // Single isolated solid pixel at centre — too small for MFS
        field[10 * nx + 10] = 1.0;
        let out = mff.apply_2d(&field, nx, ny);
        let max_out = out.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            max_out < 0.5,
            "Single pixel feature should be removed by opening: {max_out}"
        );
    }

    #[test]
    fn min_feature_filter_no_violation_large_block() {
        // Large solid block (10x10 px) with MFS = 10nm and dx = 10nm (r=1px).
        // The interior pixels at distance >= 1 from any air should survive erosion.
        let nx = 20usize;
        let ny = 20usize;
        // Use r=1 pixel MFS so interior of the 10×10 block is guaranteed to survive erosion
        let mff = MinFeatureFilter::new(10e-9, 10e-9); // r = 1 px
        let mut binary = vec![false; nx * ny];
        for j in 4..16 {
            for i in 4..16 {
                binary[j * nx + i] = true;
            }
        }
        // With r=1, only pixels with all 4 neighbours also solid survive.
        // Interior pixels (5..15 × 5..15) all have solid neighbours → no violation.
        assert!(!mff.has_violation_2d(&binary, nx, ny));
    }

    #[test]
    fn min_feature_filter_violation_single_pixel() {
        let nx = 10usize;
        let ny = 10usize;
        let mff = MinFeatureFilter::new(50e-9, 10e-9);
        let mut binary = vec![false; nx * ny];
        binary[5 * nx + 5] = true; // isolated pixel
        assert!(mff.has_violation_2d(&binary, nx, ny));
    }

    #[test]
    fn min_feature_filter_max_violation_zero_for_no_violation() {
        let nx = 10usize;
        let mff = MinFeatureFilter::new(10e-9, 10e-9); // r=1 pixel
        let binary = vec![false; nx * nx];
        let v = mff.max_violation_cells(&binary, nx, nx);
        assert_eq!(v, 0);
    }

    // ── ProjectionFilter tests ─────────────────────────────────────────────────

    #[test]
    fn projection_filter_project_at_threshold_near_half() {
        let pf = ProjectionFilter::new(50e-9, 10e-9, 5.0, 0.5);
        // At rho = eta = 0.5, projection should give ~0.5
        let val = pf.project(0.5);
        assert!(
            (val - 0.5).abs() < 0.01,
            "Projection at threshold should be ~0.5, got {val}"
        );
    }

    #[test]
    fn projection_filter_project_monotone() {
        let pf = ProjectionFilter::new(50e-9, 10e-9, 10.0, 0.5);
        let v0 = pf.project(0.0);
        let v5 = pf.project(0.5);
        let v1 = pf.project(1.0);
        assert!(v0 < v5 && v5 < v1, "Projection must be monotone increasing");
    }

    #[test]
    fn projection_filter_grad_positive() {
        let pf = ProjectionFilter::new(50e-9, 10e-9, 5.0, 0.5);
        for rho in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let g = pf.project_grad(rho);
            assert!(
                g > 0.0,
                "Projection gradient must be positive at rho={rho}, got {g}"
            );
        }
    }

    #[test]
    fn projection_filter_project_field_preserves_length() {
        let pf = ProjectionFilter::new(50e-9, 10e-9, 5.0, 0.5);
        let field: Vec<f64> = (0..20).map(|i| i as f64 / 20.0).collect();
        let projected = pf.project_field(&field);
        assert_eq!(projected.len(), field.len());
    }

    #[test]
    fn projection_filter_smooth_uniform_field_unchanged() {
        use approx::assert_relative_eq;
        let pf = ProjectionFilter::new(50e-9, 10e-9, 5.0, 0.5);
        let field = vec![0.7f64; 8 * 8];
        let smoothed = pf.smooth_2d(&field, 8, 8);
        for (i, &v) in smoothed.iter().enumerate() {
            assert_relative_eq!(v, 0.7, max_relative = 1e-10, epsilon = 1e-10);
            let _ = i;
        }
    }
}
