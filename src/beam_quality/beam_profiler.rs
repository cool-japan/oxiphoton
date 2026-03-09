//! Beam profiling and statistical moment analysis.
//!
//! Implements first- and second-order statistical moments of 1D and 2D intensity
//! profiles, including the **D4σ** beam width defined by ISO 11146.
//!
//! # D4σ definition
//! For a 1D profile I(x):
//! ```text
//!   x̄   = ∫ x I(x) dx / ∫ I(x) dx          (centroid)
//!   σ²  = ∫ (x − x̄)² I(x) dx / ∫ I(x) dx   (second moment)
//!   D4σ = 4σ                                  (ISO 11146 diameter)
//! ```
//!
//! For an ideal Gaussian w(x) = exp(−2x²/w²):
//!   σ = w/2  →  D4σ = 2w  (= 1/e² diameter)

/// Small positive floor used to avoid division by zero in normalised profiles.
const INTENSITY_FLOOR: f64 = 1e-300;

// ─── 1D beam profile ─────────────────────────────────────────────────────────

/// One-dimensional beam profile: intensity as a function of transverse position.
///
/// All positions are in metres; intensities are dimensionless (should be normalised
/// to peak = 1 or total power, but the methods work for any non-negative scale).
#[derive(Debug, Clone)]
pub struct BeamProfile1d {
    /// Transverse sample positions (m), must be monotonically increasing.
    pub positions_m: Vec<f64>,
    /// Intensity samples (non-negative).
    pub intensities: Vec<f64>,
}

impl BeamProfile1d {
    /// Generate a Gaussian intensity profile.
    ///
    /// `I(x) = exp(−2 (x − center)² / waist²)`
    ///
    /// # Arguments
    /// * `center_m` – Beam centroid position (m).
    /// * `waist_m` – 1/e² beam radius w (m).
    /// * `n_points` – Number of sample points.
    /// * `extent_m` – Half-width of the sampling range (m).
    pub fn from_gaussian(center_m: f64, waist_m: f64, n_points: usize, extent_m: f64) -> Self {
        let n = n_points.max(2);
        let mut positions = Vec::with_capacity(n);
        let mut intensities = Vec::with_capacity(n);
        for i in 0..n {
            let x = center_m - extent_m + 2.0 * extent_m * i as f64 / (n - 1) as f64;
            let exponent = -2.0 * (x - center_m) * (x - center_m) / (waist_m * waist_m);
            positions.push(x);
            intensities.push(exponent.exp());
        }
        Self {
            positions_m: positions,
            intensities,
        }
    }

    /// Total integrated power (trapezoidal rule).
    fn total_power(&self) -> f64 {
        trapz(&self.positions_m, &self.intensities)
    }

    /// Intensity-weighted centroid (first moment).
    ///
    /// `x̄ = ∫ x I(x) dx / ∫ I(x) dx`
    pub fn centroid_m(&self) -> f64 {
        let power = self.total_power();
        if power.abs() < INTENSITY_FLOOR {
            return self.positions_m.first().copied().unwrap_or(0.0);
        }
        let wx: Vec<f64> = self
            .positions_m
            .iter()
            .zip(self.intensities.iter())
            .map(|(&x, &i)| x * i)
            .collect();
        trapz(&self.positions_m, &wx) / power
    }

    /// Second-moment radius σ (not diameter).
    ///
    /// `σ = √( ∫ (x − x̄)² I(x) dx / ∫ I(x) dx )`
    pub fn second_moment_width_m(&self) -> f64 {
        let xbar = self.centroid_m();
        let power = self.total_power();
        if power.abs() < INTENSITY_FLOOR {
            return 0.0;
        }
        let wx2: Vec<f64> = self
            .positions_m
            .iter()
            .zip(self.intensities.iter())
            .map(|(&x, &i)| {
                let dx = x - xbar;
                dx * dx * i
            })
            .collect();
        let var = trapz(&self.positions_m, &wx2) / power;
        var.max(0.0).sqrt()
    }

    /// D4σ diameter (ISO 11146) = 4 × second-moment radius.
    ///
    /// For an ideal Gaussian, D4σ = 2w (the 1/e² diameter).
    pub fn d4sigma_m(&self) -> f64 {
        4.0 * self.second_moment_width_m()
    }

    /// Position of the intensity peak (m).
    ///
    /// Returns the position of the sample with the highest intensity.
    pub fn peak_position_m(&self) -> f64 {
        self.positions_m
            .iter()
            .zip(self.intensities.iter())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&x, _)| x)
            .unwrap_or(0.0)
    }

    /// Peak intensity value.
    pub fn peak_intensity(&self) -> f64 {
        self.intensities
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(0.0)
    }

    /// Full-width at half-maximum (FWHM) by linear interpolation.
    ///
    /// Finds the left and right half-maximum crossings and returns their separation.
    /// Returns 0 if the profile does not cross the half-maximum on both sides.
    pub fn fwhm_m(&self) -> f64 {
        let peak = self.peak_intensity();
        if peak < INTENSITY_FLOOR {
            return 0.0;
        }
        let half = 0.5 * peak;
        let n = self.positions_m.len();
        if n < 2 {
            return 0.0;
        }

        // Find left crossing (intensity rising through half)
        let left = {
            let mut pos = None;
            for i in 0..n - 1 {
                let i0 = self.intensities[i];
                let i1 = self.intensities[i + 1];
                if i0 <= half && i1 > half {
                    let t = (half - i0) / (i1 - i0);
                    pos = Some(
                        self.positions_m[i] + t * (self.positions_m[i + 1] - self.positions_m[i]),
                    );
                    break;
                }
            }
            pos
        };

        // Find right crossing (intensity falling through half)
        let right = {
            let mut pos = None;
            for i in (0..n - 1).rev() {
                let i0 = self.intensities[i];
                let i1 = self.intensities[i + 1];
                if i0 > half && i1 <= half {
                    let t = (half - i0) / (i1 - i0);
                    pos = Some(
                        self.positions_m[i] + t * (self.positions_m[i + 1] - self.positions_m[i]),
                    );
                    break;
                }
            }
            pos
        };

        match (left, right) {
            (Some(l), Some(r)) => (r - l).abs(),
            _ => 0.0,
        }
    }

    /// Encircled energy fraction within `r_pixels` (array-index units) of the centroid.
    ///
    /// This is a simplified 1D version treating `r_pixels` as a symmetric half-width
    /// in sample-index space around the centroid index.
    pub fn encircled_energy(intensities: &[f64], r_pixels: f64) -> f64 {
        if intensities.is_empty() {
            return 0.0;
        }
        let total: f64 = intensities.iter().sum();
        if total < INTENSITY_FLOOR {
            return 0.0;
        }
        // Centroid index
        let cx: f64 = intensities
            .iter()
            .enumerate()
            .map(|(i, &v)| i as f64 * v)
            .sum::<f64>()
            / total;

        let inside: f64 = intensities
            .iter()
            .enumerate()
            .filter(|(i, _)| (*i as f64 - cx).abs() <= r_pixels)
            .map(|(_, &v)| v)
            .sum();
        inside / total
    }
}

// ─── 2D beam profile ─────────────────────────────────────────────────────────

/// Two-dimensional beam profile: a camera-acquired (or simulated) intensity image.
///
/// The image is stored in row-major order.  Pixel (ix, iy) maps to physical position
/// `(ix × dx_m, iy × dy_m)`.
#[derive(Debug, Clone)]
pub struct BeamProfile2d {
    /// Number of pixels in x.
    pub nx: usize,
    /// Number of pixels in y.
    pub ny: usize,
    /// Physical pixel pitch in x (m).
    pub dx_m: f64,
    /// Physical pixel pitch in y (m).
    pub dy_m: f64,
    /// Row-major intensity data (background-subtracted, non-negative).
    pub data: Vec<f64>,
}

impl BeamProfile2d {
    /// Generate a 2D Gaussian intensity profile.
    ///
    /// `I(x,y) = exp(−2(x−cx)²/wx² − 2(y−cy)²/wy²)`
    ///
    /// # Arguments
    /// * `nx`, `ny` – Image dimensions (pixels).
    /// * `dx_m` – Pixel pitch (same in x and y, m).
    /// * `cx_m`, `cy_m` – Centroid position in physical coordinates (m).
    /// * `wx_m`, `wy_m` – 1/e² beam radii (m).
    pub fn from_gaussian(
        nx: usize,
        ny: usize,
        dx_m: f64,
        cx_m: f64,
        cy_m: f64,
        wx_m: f64,
        wy_m: f64,
    ) -> Self {
        let mut data = Vec::with_capacity(nx * ny);
        for iy in 0..ny {
            let y = iy as f64 * dx_m - cy_m;
            for ix in 0..nx {
                let x = ix as f64 * dx_m - cx_m;
                let val = (-2.0 * x * x / (wx_m * wx_m) - 2.0 * y * y / (wy_m * wy_m)).exp();
                data.push(val);
            }
        }
        Self {
            nx,
            ny,
            dx_m,
            dy_m: dx_m,
            data,
        }
    }

    /// Inline index for pixel (ix, iy) in row-major data.
    #[inline]
    fn idx(&self, ix: usize, iy: usize) -> usize {
        iy * self.nx + ix
    }

    /// Total integrated power (sum × pixel area).
    pub fn total_power(&self) -> f64 {
        self.data.iter().sum::<f64>() * self.dx_m * self.dy_m
    }

    /// Peak intensity (maximum pixel value).
    pub fn peak_intensity(&self) -> f64 {
        self.data
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(0.0)
    }

    /// Intensity-weighted centroid in physical coordinates (m, m).
    ///
    /// Returns `(x̄, ȳ)`.
    pub fn centroid(&self) -> (f64, f64) {
        let total: f64 = self.data.iter().sum();
        if total.abs() < INTENSITY_FLOOR {
            return (0.0, 0.0);
        }
        let mut sx = 0.0_f64;
        let mut sy = 0.0_f64;
        for iy in 0..self.ny {
            let y = iy as f64 * self.dy_m;
            for ix in 0..self.nx {
                let x = ix as f64 * self.dx_m;
                let v = self.data[self.idx(ix, iy)];
                sx += x * v;
                sy += y * v;
            }
        }
        (sx / total, sy / total)
    }

    /// Second-order central moments: (σ_xx, σ_yy, σ_xy) in m².
    ///
    /// ```text
    ///   σ_xx = ∫∫ (x − x̄)² I dA / ∫∫ I dA
    ///   σ_yy = ∫∫ (y − ȳ)² I dA / ∫∫ I dA
    ///   σ_xy = ∫∫ (x − x̄)(y − ȳ) I dA / ∫∫ I dA
    /// ```
    pub fn second_moments(&self) -> (f64, f64, f64) {
        let total: f64 = self.data.iter().sum();
        if total.abs() < INTENSITY_FLOOR {
            return (0.0, 0.0, 0.0);
        }
        let (xbar, ybar) = self.centroid();
        let mut sxx = 0.0_f64;
        let mut syy = 0.0_f64;
        let mut sxy = 0.0_f64;
        for iy in 0..self.ny {
            let dy = iy as f64 * self.dy_m - ybar;
            for ix in 0..self.nx {
                let dx = ix as f64 * self.dx_m - xbar;
                let v = self.data[self.idx(ix, iy)];
                sxx += dx * dx * v;
                syy += dy * dy * v;
                sxy += dx * dy * v;
            }
        }
        (sxx / total, syy / total, sxy / total)
    }

    /// D4σ beam diameters in x and y (m).
    ///
    /// `D4σ_x = 4 √σ_xx`,  `D4σ_y = 4 √σ_yy`
    pub fn d4sigma(&self) -> (f64, f64) {
        let (sxx, syy, _) = self.second_moments();
        (4.0 * sxx.max(0.0).sqrt(), 4.0 * syy.max(0.0).sqrt())
    }

    /// Subtract a constant background level from all pixels (floor at zero).
    pub fn subtract_background(&mut self, background_level: f64) {
        for v in self.data.iter_mut() {
            *v = (*v - background_level).max(0.0);
        }
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Trapezoidal integration: ∫ y dx over the given sample points.
fn trapz(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for i in 0..n - 1 {
        sum += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i]);
    }
    sum
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// D4σ of a Gaussian profile should equal 2w (the 1/e² diameter).
    #[test]
    fn d4sigma_gaussian_1d() {
        let w = 0.5e-3;
        let profile = BeamProfile1d::from_gaussian(0.0, w, 1001, 5.0 * w);
        let d4s = profile.d4sigma_m();
        let expected = 2.0 * w;
        assert!(
            (d4s - expected).abs() / expected < 0.02,
            "D4σ = {d4s}, expected 2w = {expected}"
        );
    }

    /// Centroid of a symmetric profile should be at the specified center.
    #[test]
    fn centroid_symmetric_gaussian() {
        let center = 1.0e-3;
        let profile = BeamProfile1d::from_gaussian(center, 0.5e-3, 1001, 5.0e-3);
        let xbar = profile.centroid_m();
        assert!(
            (xbar - center).abs() < 1e-7,
            "Centroid: got {xbar}, expected {center}"
        );
    }

    /// FWHM of Gaussian I(x) = exp(-2x²/w²).
    ///
    /// Half-maximum at -2x²/w² = ln(0.5), so x_hm = w √(ln2/2).
    /// FWHM = 2 x_hm = w √(2 ln 2) ≈ 1.1774 w.
    #[test]
    fn fwhm_gaussian_1d() {
        let w = 1.0e-3;
        let profile = BeamProfile1d::from_gaussian(0.0, w, 2001, 4.0 * w);
        let fwhm = profile.fwhm_m();
        // I(x) = exp(-2x²/w²): FWHM = w √(2 ln 2) ≈ 1.1774 w
        let expected = w * (2.0 * 2.0_f64.ln()).sqrt();
        assert!(
            (fwhm - expected).abs() / expected < 0.01,
            "FWHM: got {fwhm}, expected {expected}"
        );
    }

    /// 2D centroid of a Gaussian centred at (cx, cy).
    #[test]
    fn centroid_2d_gaussian() {
        let nx = 200;
        let ny = 200;
        let dx = 10e-6;
        let cx = 0.5 * nx as f64 * dx;
        let cy = 0.5 * ny as f64 * dx;
        let prof = BeamProfile2d::from_gaussian(nx, ny, dx, cx, cy, 50e-6, 50e-6);
        let (xc, yc) = prof.centroid();
        assert!((xc - cx).abs() < dx, "x centroid: got {xc}, expected {cx}");
        assert!((yc - cy).abs() < dx, "y centroid: got {yc}, expected {cy}");
    }

    /// D4σ of 2D Gaussian should equal 2w in both axes.
    #[test]
    fn d4sigma_2d_gaussian() {
        let nx = 300;
        let ny = 300;
        let dx = 5e-6;
        let wx = 80e-6;
        let wy = 60e-6;
        let cx = 0.5 * nx as f64 * dx;
        let cy = 0.5 * ny as f64 * dx;
        let prof = BeamProfile2d::from_gaussian(nx, ny, dx, cx, cy, wx, wy);
        let (d4sx, d4sy) = prof.d4sigma();
        assert!(
            (d4sx - 2.0 * wx).abs() / (2.0 * wx) < 0.03,
            "D4σ_x: got {d4sx}, expected {}",
            2.0 * wx
        );
        assert!(
            (d4sy - 2.0 * wy).abs() / (2.0 * wy) < 0.03,
            "D4σ_y: got {d4sy}, expected {}",
            2.0 * wy
        );
    }

    /// Background subtraction clamps at zero.
    #[test]
    fn background_subtraction_clamps_zero() {
        let mut prof = BeamProfile2d::from_gaussian(50, 50, 5e-6, 0.0, 0.0, 20e-6, 20e-6);
        prof.subtract_background(0.5);
        assert!(
            prof.data.iter().all(|&v| v >= 0.0),
            "All values must be non-negative after background subtraction"
        );
    }

    /// Peak intensity of unit Gaussian is 1.
    #[test]
    fn peak_intensity_unit_gaussian() {
        let profile = BeamProfile1d::from_gaussian(0.0, 1e-3, 101, 5e-3);
        let peak = profile.peak_intensity();
        assert!(
            (peak - 1.0).abs() < 1e-6,
            "Peak intensity of unit Gaussian: {peak}"
        );
    }

    /// Encircled energy at infinite radius should be 1.
    #[test]
    fn encircled_energy_full_range() {
        let intensities = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let ee = BeamProfile1d::encircled_energy(&intensities, 1000.0);
        assert!(
            (ee - 1.0).abs() < 1e-10,
            "Encircled energy at huge radius: {ee}"
        );
    }
}
