//! Deformable mirror models for adaptive optics.
//!
//! Provides:
//! - [`DeformableMirror`]: Continuous facesheet DM with Gaussian influence functions
//! - [`SegmentedMirror`]: Piston/tip/tilt segmented mirror (hexagonal layout)
//! - [`ZernikeCorrector`]: Modal wavefront corrector using Zernike polynomials
//!
//! # Conventions
//! - All spatial coordinates in metres unless noted otherwise
//! - Actuator commands in metres (stroke)
//! - Zernike indices follow ANSI/OSA single-index ordering (j=0 = piston)

use crate::error::OxiPhotonError;

const PI: f64 = std::f64::consts::PI;
const TWO_PI: f64 = 2.0 * PI;

// ─────────────────────────────────────────────────────────────────────────────
// Zernike polynomial basis (ANSI/OSA single-index, j = 0..20)
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate ANSI/OSA Zernike polynomial Z_j at (rho, theta).
///
/// j = 0 : piston, 1 : tip-x, 2 : tip-y, 3 : defocus, …
/// Uses the orthonormal form with normalisation factor sqrt(n+1) (or
/// sqrt(2(n+1)) for non-rotationally-symmetric terms).
pub fn zernike_ansi(j: usize, rho: f64, theta: f64) -> f64 {
    if rho > 1.0 {
        return 0.0;
    }
    let r2 = rho * rho;
    let r3 = r2 * rho;
    let r4 = r2 * r2;
    let r5 = r4 * rho;
    let r6 = r4 * r2;
    match j {
        0 => 1.0,
        1 => 2.0 * rho * theta.cos(),
        2 => 2.0 * rho * theta.sin(),
        3 => (3.0_f64).sqrt() * (2.0 * r2 - 1.0),
        4 => (6.0_f64).sqrt() * r2 * (2.0 * theta).cos(),
        5 => (6.0_f64).sqrt() * r2 * (2.0 * theta).sin(),
        6 => (8.0_f64).sqrt() * (3.0 * r3 - 2.0 * rho) * theta.cos(),
        7 => (8.0_f64).sqrt() * (3.0 * r3 - 2.0 * rho) * theta.sin(),
        8 => (5.0_f64).sqrt() * (6.0 * r4 - 6.0 * r2 + 1.0),
        9 => (8.0_f64).sqrt() * r3 * (3.0 * theta).cos(),
        10 => (8.0_f64).sqrt() * r3 * (3.0 * theta).sin(),
        11 => (10.0_f64).sqrt() * (4.0 * r4 - 3.0 * r2) * (2.0 * theta).cos(),
        12 => (10.0_f64).sqrt() * (4.0 * r4 - 3.0 * r2) * (2.0 * theta).sin(),
        13 => (12.0_f64).sqrt() * (10.0 * r5 - 12.0 * r3 + 3.0 * rho) * theta.cos(),
        14 => (12.0_f64).sqrt() * (10.0 * r5 - 12.0 * r3 + 3.0 * rho) * theta.sin(),
        15 => (7.0_f64).sqrt() * (20.0 * r6 - 30.0 * r4 + 12.0 * r2 - 1.0),
        16 => (12.0_f64).sqrt() * r4 * (4.0 * theta).cos(),
        17 => (12.0_f64).sqrt() * r4 * (4.0 * theta).sin(),
        18 => (14.0_f64).sqrt() * (5.0 * r5 - 4.0 * r3) * (3.0 * theta).cos(),
        19 => (14.0_f64).sqrt() * (5.0 * r5 - 4.0 * r3) * (3.0 * theta).sin(),
        20 => (16.0_f64).sqrt() * (15.0 * r6 - 20.0 * r4 + 6.0 * r2) * (2.0 * theta).cos(),
        _ => 0.0,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DeformableMirror
// ─────────────────────────────────────────────────────────────────────────────

/// Continuous facesheet deformable mirror with Gaussian influence functions.
///
/// Each actuator produces a Gaussian displacement on the mirror surface.
/// The total surface is the superposition of all actuator contributions
/// weighted by their commands.
///
/// # Model
/// The influence function for actuator k centred at (x_k, y_k) with
/// influence radius σ_k is:
///
/// IF_k(x,y) = exp(−((x−x_k)² + (y−y_k)²) / (2σ_k²))
#[derive(Debug, Clone)]
pub struct DeformableMirror {
    /// Number of actuators.
    pub n_actuators: usize,
    /// (x, y) position of each actuator in metres.
    pub actuator_positions: Vec<[f64; 2]>,
    /// Commanded stroke for each actuator in metres.
    pub actuator_commands: Vec<f64>,
    /// Pre-computed influence functions on a pixel grid: `[n_actuators][n_pixels²]`.
    /// Stored row-major; n_pixels is stored separately.
    pub influence_functions: Vec<Vec<f64>>,
    /// Maximum stroke magnitude (±) in metres.
    pub stroke_limit: f64,
    /// Size of the pixel grid used for influence functions.
    pub n_pixels: usize,
    /// Physical size of the mirror aperture in metres.
    pub aperture_size: f64,
    /// Gaussian σ of each influence function in metres.
    pub influence_radius: f64,
}

impl DeformableMirror {
    /// Create a square-grid DM with `n_side × n_side` actuators.
    ///
    /// # Arguments
    /// * `n_side` — number of actuators per side
    /// * `pitch` — actuator pitch in metres
    /// * `influence_radius` — Gaussian σ of the influence function in metres
    ///
    /// The DM aperture is inferred as `n_side * pitch`.
    /// A 64×64 pixel grid is used for the influence functions.
    pub fn new_square_grid(n_side: usize, pitch: f64, influence_radius: f64) -> Self {
        let n_actuators = n_side * n_side;
        let aperture = n_side as f64 * pitch;
        let n_pixels: usize = 64;

        let mut positions = Vec::with_capacity(n_actuators);
        for row in 0..n_side {
            for col in 0..n_side {
                let x = (col as f64 + 0.5) * pitch - aperture * 0.5;
                let y = (row as f64 + 0.5) * pitch - aperture * 0.5;
                positions.push([x, y]);
            }
        }

        // Pre-compute Gaussian influence functions on the pixel grid.
        let pixel_scale = aperture / n_pixels as f64;
        let two_sigma2 = 2.0 * influence_radius * influence_radius;
        let mut influence_functions = Vec::with_capacity(n_actuators);

        for &[ax, ay] in &positions {
            let mut if_k = Vec::with_capacity(n_pixels * n_pixels);
            for py in 0..n_pixels {
                for px in 0..n_pixels {
                    let x = (px as f64 + 0.5) * pixel_scale - aperture * 0.5;
                    let y = (py as f64 + 0.5) * pixel_scale - aperture * 0.5;
                    let dx = x - ax;
                    let dy = y - ay;
                    let val = (-(dx * dx + dy * dy) / two_sigma2).exp();
                    if_k.push(val);
                }
            }
            influence_functions.push(if_k);
        }

        Self {
            n_actuators,
            actuator_positions: positions,
            actuator_commands: vec![0.0; n_actuators],
            influence_functions,
            stroke_limit: 10e-6, // 10 µm default
            n_pixels,
            aperture_size: aperture,
            influence_radius,
        }
    }

    /// Compute the mirror surface height (in metres) at point (x, y).
    ///
    /// Uses bilinear interpolation of the precomputed influence function grid.
    pub fn surface_shape(&self, x: f64, y: f64) -> f64 {
        let pixel_scale = self.aperture_size / self.n_pixels as f64;
        let half = self.aperture_size * 0.5;

        // Map (x,y) to fractional pixel index.
        let px_f = (x + half) / pixel_scale - 0.5;
        let py_f = (y + half) / pixel_scale - 0.5;

        let px0 = px_f.floor() as isize;
        let py0 = py_f.floor() as isize;

        let n = self.n_pixels as isize;
        // Check bounds.
        if px0 < 0 || py0 < 0 || px0 + 1 >= n || py0 + 1 >= n {
            return 0.0;
        }
        let fx = px_f - px0 as f64;
        let fy = py_f - py0 as f64;

        let idx = |py: isize, px: isize| -> usize { (py * n + px) as usize };

        let mut height = 0.0_f64;
        for (k, cmd) in self.actuator_commands.iter().enumerate() {
            if cmd.abs() < 1e-30 {
                continue;
            }
            let if_k = &self.influence_functions[k];
            let v00 = if_k[idx(py0, px0)];
            let v01 = if_k[idx(py0, px0 + 1)];
            let v10 = if_k[idx(py0 + 1, px0)];
            let v11 = if_k[idx(py0 + 1, px0 + 1)];
            let interp = v00 * (1.0 - fx) * (1.0 - fy)
                + v01 * fx * (1.0 - fy)
                + v10 * (1.0 - fx) * fy
                + v11 * fx * fy;
            height += cmd * interp;
        }
        height
    }

    /// Set the command (stroke) for actuator `actuator` in metres.
    ///
    /// Returns an error if the actuator index is out of range or the stroke
    /// exceeds `stroke_limit`.
    pub fn set_command(&mut self, actuator: usize, value: f64) -> Result<(), OxiPhotonError> {
        if actuator >= self.n_actuators {
            return Err(OxiPhotonError::NumericalError(format!(
                "Actuator index {} out of range (n_actuators = {})",
                actuator, self.n_actuators
            )));
        }
        if value.abs() > self.stroke_limit {
            return Err(OxiPhotonError::NumericalError(format!(
                "Actuator {} command {:.3e} m exceeds stroke limit ±{:.3e} m",
                actuator, value, self.stroke_limit
            )));
        }
        self.actuator_commands[actuator] = value;
        Ok(())
    }

    /// Apply the DM surface to a flat wavefront array (in-place).
    ///
    /// The wavefront is sampled on an `n_pixels × n_pixels` grid with spacing
    /// `pixel_size` metres. The DM surface is added to the wavefront (OPD in
    /// metres, with factor of 2 for reflection).
    pub fn apply_wavefront(&self, wavefront: &mut Vec<f64>, pixel_size: f64, n_pixels: usize) {
        let n2 = n_pixels * n_pixels;
        if wavefront.len() < n2 {
            wavefront.resize(n2, 0.0);
        }

        // Build DM surface on the requested grid.
        let half = (n_pixels as f64 * pixel_size) * 0.5;
        for py in 0..n_pixels {
            for px in 0..n_pixels {
                let x = (px as f64 + 0.5) * pixel_size - half;
                let y = (py as f64 + 0.5) * pixel_size - half;
                // Factor 2 for reflection (double-pass).
                wavefront[py * n_pixels + px] += 2.0 * self.surface_shape(x, y);
            }
        }
    }

    /// Set all actuator commands to zero (flatten the mirror).
    pub fn flatten(&mut self) {
        for cmd in self.actuator_commands.iter_mut() {
            *cmd = 0.0;
        }
    }

    /// Return the stroke range as `(−stroke_limit, +stroke_limit)`.
    pub fn stroke_range(&self) -> (f64, f64) {
        (-self.stroke_limit, self.stroke_limit)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SegmentedMirror
// ─────────────────────────────────────────────────────────────────────────────

/// Hexagonal segmented mirror with piston, tip, and tilt per segment.
///
/// Segments are arranged in concentric hexagonal rings. The segment
/// centres follow the flat-top hexagonal packing geometry.
///
/// # Degrees of freedom
/// For each segment: piston (z), tip-x (tilt around y-axis), tip-y
/// (tilt around x-axis). Total actuators = 3 × n_segments.
#[derive(Debug, Clone)]
pub struct SegmentedMirror {
    /// Number of segments.
    pub n_segments: usize,
    /// Circumscribed radius of each segment (centre-to-vertex) in metres.
    pub segment_radius: f64,
    /// Piston (z-offset) per segment in metres.
    pub pistons: Vec<f64>,
    /// Tip around x-axis per segment in radians.
    pub tip_x: Vec<f64>,
    /// Tip around y-axis per segment in radians.
    pub tip_y: Vec<f64>,
    /// (x, y) centre of each segment in metres.
    pub segment_centres: Vec<[f64; 2]>,
}

impl SegmentedMirror {
    /// Create a hexagonal segmented mirror with `rings` concentric rings.
    ///
    /// - `rings = 0` → 1 central segment
    /// - `rings = 1` → 7 segments
    /// - `rings = k` → 1 + 6*(1+2+…+k) = 3k²+3k+1 segments
    pub fn new_hexagonal(rings: usize, segment_radius: f64) -> Self {
        let mut centres = Vec::new();
        // Hexagonal lattice vectors (flat-top orientation).
        // For flat-top hexagons the pitch between centres is sqrt(3)*R.
        let hex_pitch = (3.0_f64).sqrt() * segment_radius;

        // Axial coordinate enumeration of hex grid up to radius `rings`.
        for q in -(rings as i64)..=(rings as i64) {
            let r_min = (-(rings as i64)).max(-q - (rings as i64));
            let r_max = (rings as i64).min(-q + rings as i64);
            for r in r_min..=r_max {
                // Convert axial (q, r) → Cartesian.
                let x = hex_pitch * (q as f64 + r as f64 * 0.5);
                let y = hex_pitch * (r as f64 * (3.0_f64).sqrt() * 0.5);
                centres.push([x, y]);
            }
        }

        let n = centres.len();
        Self {
            n_segments: n,
            segment_radius,
            pistons: vec![0.0; n],
            tip_x: vec![0.0; n],
            tip_y: vec![0.0; n],
            segment_centres: centres,
        }
    }

    /// Set the piston for segment `seg` in metres.
    pub fn set_piston(&mut self, seg: usize, piston: f64) {
        if seg < self.n_segments {
            self.pistons[seg] = piston;
        }
    }

    /// Set tip around y-axis (x-tilt) for segment `seg` in radians.
    pub fn set_tip_x(&mut self, seg: usize, tip: f64) {
        if seg < self.n_segments {
            self.tip_x[seg] = tip;
        }
    }

    /// Set tip around x-axis (y-tilt) for segment `seg` in radians.
    pub fn set_tip_y(&mut self, seg: usize, tip: f64) {
        if seg < self.n_segments {
            self.tip_y[seg] = tip;
        }
    }

    /// Total number of actuator degrees of freedom: 3 × n_segments.
    pub fn total_actuators(&self) -> usize {
        3 * self.n_segments
    }

    /// Compute the surface height at (x, y) by finding the nearest segment.
    ///
    /// Returns `None` if (x, y) falls in a gap between segments.
    pub fn surface_height(&self, x: f64, y: f64) -> Option<f64> {
        // Find the nearest segment centre.
        let mut best_idx = 0;
        let mut best_dist2 = f64::INFINITY;
        for (i, &[cx, cy]) in self.segment_centres.iter().enumerate() {
            let d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            if d2 < best_dist2 {
                best_dist2 = d2;
                best_idx = i;
            }
        }
        // Check inside segment hexagon (inscribed circle radius = R*sqrt(3)/2).
        let inradius = self.segment_radius * (3.0_f64).sqrt() * 0.5;
        if best_dist2.sqrt() > inradius {
            return None;
        }
        let [cx, cy] = self.segment_centres[best_idx];
        let dx = x - cx;
        let dy = y - cy;
        let height = self.pistons[best_idx] + self.tip_x[best_idx] * dx + self.tip_y[best_idx] * dy;
        Some(height)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ZernikeCorrector
// ─────────────────────────────────────────────────────────────────────────────

/// Wavefront corrector operating in the Zernike modal basis.
///
/// Fits the measured wavefront error to a set of Zernike modes via
/// least-squares projection and reconstructs the correction surface.
///
/// # Notes
/// The Zernike modes use the ANSI/OSA single-index ordering implemented
/// in [`zernike_ansi`]. Modes beyond j=20 are silently set to zero.
#[derive(Debug, Clone)]
pub struct ZernikeCorrector {
    /// Number of Zernike modes.
    pub n_modes: usize,
    /// Fitted Zernike coefficients (metres, OPD convention).
    pub coefficients: Vec<f64>,
    /// Radius of the pupil aperture in metres.
    pub aperture_radius: f64,
}

impl ZernikeCorrector {
    /// Create a new ZernikeCorrector with `n_modes` modes and given aperture.
    pub fn new(n_modes: usize, aperture_radius: f64) -> Self {
        Self {
            n_modes,
            coefficients: vec![0.0; n_modes],
            aperture_radius,
        }
    }

    /// Fit the wavefront to Zernike modes via direct projection.
    ///
    /// # Arguments
    /// * `wavefront` — flat slice of wavefront OPD values in metres
    /// * `x_coords` — x-coordinate for each wavefront sample in metres
    /// * `y_coords` — y-coordinate for each wavefront sample in metres
    ///
    /// Points outside the aperture (r > aperture_radius) are ignored.
    pub fn fit_wavefront(&mut self, wavefront: &[f64], x_coords: &[f64], y_coords: &[f64]) {
        let n_pts = wavefront.len().min(x_coords.len()).min(y_coords.len());
        let mut numerators = vec![0.0_f64; self.n_modes];
        let mut denominators = vec![0.0_f64; self.n_modes];

        for i in 0..n_pts {
            let x = x_coords[i];
            let y = y_coords[i];
            let rho = (x * x + y * y).sqrt() / self.aperture_radius;
            if rho > 1.0 {
                continue;
            }
            let theta = y.atan2(x);
            let w = wavefront[i];
            for j in 0..self.n_modes {
                let z = zernike_ansi(j, rho, theta);
                numerators[j] += w * z;
                denominators[j] += z * z;
            }
        }

        for j in 0..self.n_modes {
            self.coefficients[j] = if denominators[j].abs() > 1e-30 {
                numerators[j] / denominators[j]
            } else {
                0.0
            };
        }
    }

    /// Reconstruct the wavefront correction at position (x, y) in metres.
    ///
    /// Returns the OPD correction value in metres.
    pub fn reconstruct(&self, x: f64, y: f64) -> f64 {
        let rho = (x * x + y * y).sqrt() / self.aperture_radius;
        if rho > 1.0 {
            return 0.0;
        }
        let theta = y.atan2(x);
        self.coefficients
            .iter()
            .enumerate()
            .map(|(j, &c)| c * zernike_ansi(j, rho, theta))
            .sum()
    }

    /// Compute the RMS residual after subtracting the Zernike reconstruction.
    ///
    /// Returns RMS in the same units as the wavefront (metres).
    pub fn residual_rms(&self, wavefront: &[f64], x_coords: &[f64], y_coords: &[f64]) -> f64 {
        let n_pts = wavefront.len().min(x_coords.len()).min(y_coords.len());
        let mut sum2 = 0.0_f64;
        let mut count = 0usize;
        for i in 0..n_pts {
            let x = x_coords[i];
            let y = y_coords[i];
            let rho = (x * x + y * y).sqrt() / self.aperture_radius;
            if rho > 1.0 {
                continue;
            }
            let correction = self.reconstruct(x, y);
            let residual = wavefront[i] - correction;
            sum2 += residual * residual;
            count += 1;
        }
        if count == 0 {
            return 0.0;
        }
        (sum2 / count as f64).sqrt()
    }

    /// Approximate Strehl ratio via the Maréchal approximation.
    ///
    /// Uses the RMS of the fitted Zernike coefficients as a proxy for the
    /// residual phase variance:
    ///
    /// S ≈ exp(−(2π σ_rms / λ)²)
    ///
    /// Here the coefficients are in metres, and λ = 0.5 µm is assumed.
    /// For a corrector with zero coefficients this returns 1.0.
    pub fn strehl_ratio(&self) -> f64 {
        // Use remaining power in coefficients j>=1 (exclude piston j=0).
        let sigma2: f64 = self.coefficients.iter().skip(1).map(|c| c * c).sum();
        let sigma_rms = sigma2.sqrt();
        // Assume λ = 500 nm = 5e-7 m for the Maréchal formula.
        let lambda = 500e-9_f64;
        let phase_rms = TWO_PI * sigma_rms / lambda;
        (-phase_rms * phase_rms).exp()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dm_square_grid_actuator_count() {
        let dm = DeformableMirror::new_square_grid(4, 1e-3, 0.6e-3);
        assert_eq!(dm.n_actuators, 16);
        assert_eq!(dm.actuator_commands.len(), 16);
        assert_eq!(dm.influence_functions.len(), 16);
    }

    #[test]
    fn test_dm_set_command_within_stroke() {
        let mut dm = DeformableMirror::new_square_grid(3, 1e-3, 0.6e-3);
        dm.stroke_limit = 5e-6;
        let result = dm.set_command(0, 3e-6);
        assert!(result.is_ok());
        assert!((dm.actuator_commands[0] - 3e-6).abs() < 1e-15);
    }

    #[test]
    fn test_dm_set_command_exceeds_stroke() {
        let mut dm = DeformableMirror::new_square_grid(3, 1e-3, 0.6e-3);
        dm.stroke_limit = 5e-6;
        let result = dm.set_command(0, 10e-6);
        assert!(result.is_err());
    }

    #[test]
    fn test_dm_set_command_out_of_range() {
        let mut dm = DeformableMirror::new_square_grid(3, 1e-3, 0.6e-3);
        let result = dm.set_command(99, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_dm_flatten_zeros_all_commands() {
        let mut dm = DeformableMirror::new_square_grid(3, 1e-3, 0.6e-3);
        for i in 0..dm.n_actuators {
            let _ = dm.set_command(i, 0.0);
        }
        dm.flatten();
        for &c in &dm.actuator_commands {
            assert_eq!(c, 0.0);
        }
    }

    #[test]
    fn test_dm_stroke_range() {
        let dm = DeformableMirror::new_square_grid(3, 1e-3, 0.6e-3);
        let (lo, hi) = dm.stroke_range();
        assert!(lo < 0.0);
        assert!(hi > 0.0);
        assert!((lo + hi).abs() < 1e-30);
    }

    #[test]
    fn test_dm_surface_shape_flat() {
        // With all commands zero, surface should be zero everywhere.
        let dm = DeformableMirror::new_square_grid(4, 0.5e-3, 0.3e-3);
        let h = dm.surface_shape(0.0, 0.0);
        assert!(h.abs() < 1e-30, "Flat DM surface should be zero, got {}", h);
    }

    #[test]
    fn test_dm_surface_shape_nonzero_command() {
        let mut dm = DeformableMirror::new_square_grid(4, 0.5e-3, 0.4e-3);
        dm.stroke_limit = 10e-6;
        // Push the central-most actuator.
        let _ = dm.set_command(5, 1e-6);
        let h = dm.surface_shape(0.0, 0.0);
        // Should be non-zero near origin.
        assert!(h.abs() > 0.0, "Surface should deflect with nonzero command");
    }

    #[test]
    fn test_segmented_mirror_hexagonal_ring0() {
        let sm = SegmentedMirror::new_hexagonal(0, 1e-3);
        assert_eq!(sm.n_segments, 1);
        assert_eq!(sm.total_actuators(), 3);
    }

    #[test]
    fn test_segmented_mirror_hexagonal_ring1() {
        let sm = SegmentedMirror::new_hexagonal(1, 1e-3);
        assert_eq!(sm.n_segments, 7);
        assert_eq!(sm.total_actuators(), 21);
    }

    #[test]
    fn test_segmented_mirror_hexagonal_ring2() {
        let sm = SegmentedMirror::new_hexagonal(2, 1e-3);
        assert_eq!(sm.n_segments, 19);
    }

    #[test]
    fn test_segmented_mirror_set_piston() {
        let mut sm = SegmentedMirror::new_hexagonal(1, 1e-3);
        sm.set_piston(0, 500e-9);
        assert!((sm.pistons[0] - 500e-9).abs() < 1e-20);
    }

    #[test]
    fn test_segmented_mirror_surface_height_centre() {
        // rings=0 gives a single segment at the origin.
        let mut sm = SegmentedMirror::new_hexagonal(0, 2e-3);
        sm.set_piston(0, 1e-6);
        let h = sm.surface_height(0.0, 0.0);
        assert!(h.is_some(), "Centre should be inside the single segment");
        // At origin with zero tip, height = piston = 1e-6 m.
        assert!(
            (h.unwrap() - 1e-6).abs() < 1e-12,
            "Surface height at centre = {}, expected 1e-6",
            h.unwrap()
        );
    }

    #[test]
    fn test_zernike_corrector_piston_fit() {
        let mut zc = ZernikeCorrector::new(4, 1.0);
        // Flat wavefront + piston = 1e-6 m.
        let n_pts = 100;
        let mut wf = Vec::with_capacity(n_pts);
        let mut xs = Vec::with_capacity(n_pts);
        let mut ys = Vec::with_capacity(n_pts);
        for i in 0..n_pts {
            let theta = TWO_PI * i as f64 / n_pts as f64;
            let r = 0.5;
            xs.push(r * theta.cos());
            ys.push(r * theta.sin());
            wf.push(1e-6);
        }
        zc.fit_wavefront(&wf, &xs, &ys);
        // j=0 (piston) coefficient should be close to 1e-6.
        assert!(
            (zc.coefficients[0] - 1e-6).abs() < 5e-7,
            "Piston coefficient mismatch: {}",
            zc.coefficients[0]
        );
    }

    #[test]
    fn test_zernike_corrector_flat_wavefront_strehl() {
        let zc = ZernikeCorrector::new(10, 1.0);
        // All coefficients zero → Strehl should be 1.0.
        let s = zc.strehl_ratio();
        assert!(
            (s - 1.0).abs() < 1e-12,
            "Flat corrector Strehl should be 1.0, got {}",
            s
        );
    }

    #[test]
    fn test_zernike_corrector_residual_rms_decreases() {
        let mut zc = ZernikeCorrector::new(6, 1.0);
        // Build a simple radially-sampled grid so the wavefront has a well-defined RMS.
        // Mix several Zernike modes to guarantee non-trivial RMS.
        let n_pts = 500;
        let mut wf = Vec::with_capacity(n_pts);
        let mut xs = Vec::with_capacity(n_pts);
        let mut ys = Vec::with_capacity(n_pts);
        for k in 0..n_pts {
            let theta = TWO_PI * k as f64 / n_pts as f64;
            let r = 0.5 + 0.4 * (k as f64 / n_pts as f64);
            let x = r * theta.cos();
            let y = r * theta.sin();
            xs.push(x);
            ys.push(y);
            let rho = r;
            // Deliberately large coefficient so the RMS is unambiguously > 0.
            let w = 500e-9 * zernike_ansi(3, rho, theta) + 200e-9 * zernike_ansi(4, rho, theta);
            wf.push(w);
        }
        // Compute uncorrected residual (no fitting done yet).
        let rms_before = {
            let n = wf.len() as f64;
            let mean = wf.iter().sum::<f64>() / n;
            (wf.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt()
        };
        // Sanity: the wavefront should have non-zero RMS before correction.
        assert!(
            rms_before > 1e-12,
            "RMS before correction should be > 0, got {}",
            rms_before
        );

        // Fit and compute residual.
        zc.fit_wavefront(&wf, &xs, &ys);
        let rms_after = zc.residual_rms(&wf, &xs, &ys);
        assert!(
            rms_after < rms_before,
            "Residual RMS ({:.3e}) should be smaller than original RMS ({:.3e}) after correction",
            rms_after,
            rms_before
        );
    }

    #[test]
    fn test_zernike_ansi_piston() {
        // Z_0 = 1.0 everywhere inside aperture.
        let val = zernike_ansi(0, 0.5, 0.0);
        assert!((val - 1.0).abs() < 1e-12, "Z_0 should be 1.0, got {}", val);
    }

    #[test]
    fn test_zernike_ansi_outside_aperture() {
        // Outside aperture (rho > 1) should return 0.
        let val = zernike_ansi(3, 1.5, 0.0);
        assert_eq!(val, 0.0, "Z outside aperture should be 0");
    }

    #[test]
    fn test_dm_apply_wavefront_length() {
        let dm = DeformableMirror::new_square_grid(4, 0.5e-3, 0.3e-3);
        let mut wf = vec![0.0_f64; 0];
        dm.apply_wavefront(&mut wf, 1e-4, 16);
        assert_eq!(wf.len(), 16 * 16);
    }
}
