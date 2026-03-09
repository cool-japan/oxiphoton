//! Wavefront sensor models for adaptive optics.
//!
//! Provides:
//! - [`ShackHartmannSensor`]: Lenslet-array wavefront sensor with Hudgin geometry
//!   reconstruction and noise modelling
//! - [`PyramidSensor`]: Four-sided pyramid wavefront sensor with modulation
//! - [`CurvatureSensor`]: Defocused-intensity curvature sensor (Roddier 1988)
//!
//! # References
//! - Shack & Platt (1971) — lenslet-array WFS
//! - Ragazzoni (1996) — pyramid WFS
//! - Roddier (1988) — curvature sensing
//! - Hudgin (1977) — wavefront reconstruction geometry

use crate::error::OxiPhotonError;

const PI: f64 = std::f64::consts::PI;
const TWO_PI: f64 = 2.0 * PI;

// ─────────────────────────────────────────────────────────────────────────────
// ShackHartmannSensor
// ─────────────────────────────────────────────────────────────────────────────

/// Shack-Hartmann wavefront sensor.
///
/// A lenslet array divides the pupil into sub-apertures. Each lenslet
/// focuses its portion of the wavefront onto a detector; the centroid shift
/// of each focal spot yields the local wavefront slope.
///
/// # Reconstruction
/// Uses the Hudgin geometry: slopes are measured at the centres of
/// sub-apertures and the wavefront is estimated at the corners using
/// a least-squares integration (zonal reconstructor).
#[derive(Debug, Clone)]
pub struct ShackHartmannSensor {
    /// Number of lenslets in x direction.
    pub n_lenslets_x: usize,
    /// Number of lenslets in y direction.
    pub n_lenslets_y: usize,
    /// Lenslet pitch (centre-to-centre spacing) in metres.
    pub lenslet_pitch: f64,
    /// Lenslet focal length in metres.
    pub focal_length: f64,
    /// Detector pixel size in metres.
    pub pixel_size: f64,
    /// Sensing wavelength in metres.
    pub wavelength: f64,
    /// Measured centroid for each sub-aperture `[n_lenslets_x*n_lenslets_y][x,y]` in metres.
    pub centroids: Vec<[f64; 2]>,
    /// Reference (flat wavefront) centroids `[n_sub][x,y]` in metres.
    pub reference_centroids: Vec<[f64; 2]>,
}

impl ShackHartmannSensor {
    /// Create a new Shack-Hartmann sensor.
    ///
    /// # Arguments
    /// * `n_x`, `n_y` — number of lenslets in each direction
    /// * `pitch` — lenslet pitch in metres
    /// * `focal_length` — lenslet focal length in metres
    /// * `wavelength` — sensing wavelength in metres
    pub fn new(n_x: usize, n_y: usize, pitch: f64, focal_length: f64, wavelength: f64) -> Self {
        let n_sub = n_x * n_y;
        let pixel_size = wavelength * focal_length / pitch; // Nyquist-sampled
        Self {
            n_lenslets_x: n_x,
            n_lenslets_y: n_y,
            lenslet_pitch: pitch,
            focal_length,
            pixel_size,
            wavelength,
            centroids: vec![[0.0, 0.0]; n_sub],
            reference_centroids: vec![[0.0, 0.0]; n_sub],
        }
    }

    /// Total number of sub-apertures.
    pub fn n_subapertures(&self) -> usize {
        self.n_lenslets_x * self.n_lenslets_y
    }

    /// Compute wavefront slopes from the centroid offsets.
    ///
    /// Returns `Vec<[sx, sy]>` where each slope is in radians (angle of
    /// incidence on the lenslet = centroid_offset / focal_length).
    pub fn measure_slopes(&self) -> Vec<[f64; 2]> {
        self.centroids
            .iter()
            .zip(self.reference_centroids.iter())
            .map(|(&c, &r)| {
                let dx = c[0] - r[0];
                let dy = c[1] - r[1];
                [dx / self.focal_length, dy / self.focal_length]
            })
            .collect()
    }

    /// Reconstruct the wavefront phase using Hudgin geometry least-squares.
    ///
    /// Returns a flat Vec of wavefront OPD values (in radians) on a grid
    /// of size `(n_lenslets_x + 1) × (n_lenslets_y + 1)` (corner nodes).
    ///
    /// The simple zonal integration uses cumulative trapezoidal sums along
    /// rows and columns, then averages the two paths.
    pub fn reconstruct_wavefront(&self) -> Vec<f64> {
        let nx = self.n_lenslets_x;
        let ny = self.n_lenslets_y;
        let slopes = self.measure_slopes();

        // Grid of wavefront nodes: (nx+1)×(ny+1).
        let node_nx = nx + 1;
        let node_ny = ny + 1;
        let mut w_row = vec![0.0_f64; node_nx * node_ny];
        let mut w_col = vec![0.0_f64; node_nx * node_ny];

        let slope_at = |iy: usize, ix: usize| -> [f64; 2] {
            if iy < ny && ix < nx {
                slopes[iy * nx + ix]
            } else {
                [0.0, 0.0]
            }
        };

        // Integrate slopes along x-rows first.
        // w_row[iy*(nx+1) + ix+1] = w_row[iy*(nx+1)+ix] + sx * pitch
        for iy in 0..node_ny {
            for ix in 0..nx {
                let s = slope_at(iy.min(ny - 1), ix);
                let cur = w_row[iy * node_nx + ix];
                w_row[iy * node_nx + ix + 1] = cur + s[0] * self.lenslet_pitch;
            }
        }

        // Integrate slopes along y-columns.
        for ix in 0..node_nx {
            for iy in 0..ny {
                let s = slope_at(iy, ix.min(nx - 1));
                let cur = w_col[iy * node_nx + ix];
                w_col[(iy + 1) * node_nx + ix] = cur + s[1] * self.lenslet_pitch;
            }
        }

        // Average the two integration paths.
        (0..node_nx * node_ny)
            .map(|i| (w_row[i] + w_col[i]) * 0.5)
            .collect()
    }

    /// Set the current centroids as the reference (null the sensor).
    ///
    /// After this call, `measure_slopes` will return zero for the current
    /// wavefront. Used to calibrate out static aberrations.
    pub fn null_reference(&mut self) {
        self.reference_centroids.clone_from(&self.centroids);
    }

    /// Photon-noise limited slope RMS in radians.
    ///
    /// The Shack-Hartmann centroiding precision due to photon noise is:
    ///
    /// σ_θ = λ / (2π · d · √N)
    ///
    /// where d = lenslet pitch and N = number of photons per sub-aperture.
    ///
    /// # Arguments
    /// * `n_photons` — detected photons per sub-aperture per frame
    pub fn photon_noise_rms(&self, n_photons: f64) -> f64 {
        if n_photons <= 0.0 {
            return f64::INFINITY;
        }
        self.wavelength / (TWO_PI * self.lenslet_pitch * n_photons.sqrt())
    }

    /// Set centroid measurements from external slope data (e.g., simulation).
    ///
    /// `slopes_x` and `slopes_y` are 2D slope arrays (angles in radians),
    /// indexed `[iy][ix]`. The sensor converts to focal-plane centroid positions.
    pub fn set_slopes_from_arrays(
        &mut self,
        slopes_x: &[Vec<f64>],
        slopes_y: &[Vec<f64>],
    ) -> Result<(), OxiPhotonError> {
        let ny = slopes_x.len();
        let nx = if ny > 0 { slopes_x[0].len() } else { 0 };
        if ny != self.n_lenslets_y || nx != self.n_lenslets_x {
            return Err(OxiPhotonError::NumericalError(format!(
                "Slope array size ({nx}×{ny}) does not match sensor ({nx2}×{ny2})",
                nx2 = self.n_lenslets_x,
                ny2 = self.n_lenslets_y
            )));
        }
        for iy in 0..ny {
            for ix in 0..nx {
                let sx = if ix < slopes_x[iy].len() {
                    slopes_x[iy][ix]
                } else {
                    0.0
                };
                let sy = if ix < slopes_y[iy].len() {
                    slopes_y[iy][ix]
                } else {
                    0.0
                };
                // Convert angle (rad) → centroid position: c = θ * f.
                self.centroids[iy * nx + ix] = [sx * self.focal_length, sy * self.focal_length];
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyramidSensor
// ─────────────────────────────────────────────────────────────────────────────

/// Four-sided pyramid wavefront sensor.
///
/// A four-faceted glass pyramid splits the beam into four pupil images.
/// The difference signals between these images give the wavefront slopes.
///
/// # Model
/// With circular modulation of radius θ_mod, the sensitivity function
/// is approximately linear for |θ| < θ_mod (the modulated regime).
/// Without modulation the sensor is nonlinear.
///
/// The four quadrant intensity images are labelled A (top-left),
/// B (top-right), C (bottom-left), D (bottom-right).
#[derive(Debug, Clone)]
pub struct PyramidSensor {
    /// Pupil image size in pixels (each quadrant is n_pixels × n_pixels).
    pub n_pixels: usize,
    /// Modulation radius in radians.
    pub modulation_radius: f64,
    /// Sensing wavelength in metres.
    pub wavelength: f64,
}

impl PyramidSensor {
    /// Create a new pyramid sensor.
    ///
    /// # Arguments
    /// * `n_pixels` — size of each pupil quadrant image
    /// * `mod_radius` — modulation radius in radians (0 = unmodulated)
    /// * `wavelength` — sensing wavelength in metres
    pub fn new(n_pixels: usize, mod_radius: f64, wavelength: f64) -> Self {
        Self {
            n_pixels,
            modulation_radius: mod_radius,
            wavelength,
        }
    }

    /// Compute the four quadrant intensity images from a slope map.
    ///
    /// Uses the linear approximation valid in the modulated regime.
    /// Each quadrant image is a flat Vec of length `n_pixels²`.
    ///
    /// # Arguments
    /// * `slopes_x` — x-slope for each pixel in radians
    /// * `slopes_y` — y-slope for each pixel in radians
    ///
    /// # Returns
    /// `[A, B, C, D]` where each is a Vec of length `n_pixels * n_pixels`.
    pub fn intensity_signals(&self, slopes_x: &[f64], slopes_y: &[f64]) -> [Vec<f64>; 4] {
        let n2 = self.n_pixels * self.n_pixels;
        let n_pts = slopes_x.len().min(slopes_y.len()).min(n2);

        // In the modulated, linear regime:
        //   Sx(i) = (A+B - C-D) / (A+B+C+D)  ∝ sx / θ_mod
        //   Sy(i) = (A+C - B-D) / (A+B+C+D)  ∝ sy / θ_mod
        //
        // Invert to get quadrant intensities assuming uniform amplitude I0 = 1:
        //   A = 0.25*(1 + sx/θ_mod + sy/θ_mod)
        //   B = 0.25*(1 + sx/θ_mod - sy/θ_mod)
        //   C = 0.25*(1 - sx/θ_mod + sy/θ_mod)
        //   D = 0.25*(1 - sx/θ_mod - sy/θ_mod)
        let theta_mod = self.modulation_radius.max(1e-10);

        let mut a_img = vec![0.25_f64; n2];
        let mut b_img = vec![0.25_f64; n2];
        let mut c_img = vec![0.25_f64; n2];
        let mut d_img = vec![0.25_f64; n2];

        for i in 0..n_pts {
            let sx = slopes_x[i] / theta_mod;
            let sy = slopes_y[i] / theta_mod;
            // Clip to valid range [-1, 1] for linear regime.
            let sx = sx.clamp(-0.9, 0.9);
            let sy = sy.clamp(-0.9, 0.9);
            a_img[i] = 0.25 * (1.0 + sx + sy);
            b_img[i] = 0.25 * (1.0 + sx - sy);
            c_img[i] = 0.25 * (1.0 - sx + sy);
            d_img[i] = 0.25 * (1.0 - sx - sy);
        }

        [a_img, b_img, c_img, d_img]
    }

    /// Reconstruct slopes from four quadrant pupil images.
    ///
    /// # Arguments
    /// * `quadrant_images` — slice of `[A, B, C, D]` intensity tuples,
    ///   one per pixel
    ///
    /// # Returns
    /// `(slopes_x, slopes_y)` both in radians.
    pub fn reconstruct_slopes(&self, quadrant_images: &[[f64; 4]]) -> (Vec<f64>, Vec<f64>) {
        let theta_mod = self.modulation_radius.max(1e-10);
        let mut sx_out = Vec::with_capacity(quadrant_images.len());
        let mut sy_out = Vec::with_capacity(quadrant_images.len());

        for &[a, b, c, d] in quadrant_images {
            let total = a + b + c + d;
            if total < 1e-30 {
                sx_out.push(0.0);
                sy_out.push(0.0);
            } else {
                // Sx = (A+B - C-D) / total, Sy = (A+C - B-D) / total.
                let sx_norm = (a + b - c - d) / total;
                let sy_norm = (a + c - b - d) / total;
                sx_out.push(sx_norm * theta_mod);
                sy_out.push(sy_norm * theta_mod);
            }
        }
        (sx_out, sy_out)
    }

    /// Sensitivity factor in radians of slope per radian of wavefront tilt.
    ///
    /// For the modulated pyramid, sensitivity = 1/θ_mod (linear regime).
    pub fn sensitivity(&self) -> f64 {
        1.0 / self.modulation_radius.max(1e-10)
    }

    /// Minimum detectable slope due to photon noise (radians).
    ///
    /// σ_θ ≈ θ_mod / √(N/4) where N is photons per pixel.
    pub fn slope_noise_rms(&self, n_photons_per_pixel: f64) -> f64 {
        if n_photons_per_pixel <= 0.0 {
            return f64::INFINITY;
        }
        self.modulation_radius / (n_photons_per_pixel * 0.25).sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CurvatureSensor
// ─────────────────────────────────────────────────────────────────────────────

/// Curvature wavefront sensor (Roddier 1988).
///
/// Two defocused pupil images at conjugate planes ±z from focus are recorded.
/// Their normalised intensity difference is proportional to the wavefront
/// Laplacian (via the irradiance transport equation).
///
/// # Model
/// Let I₊ and I₋ be the two defocused images. Then:
///
///   C(x,y) = (I₊ − I₋) / (I₊ + I₋) ∝ ∇²φ / (defocus_distance)
///
/// Reconstruction is performed via iterative Fourier integration.
#[derive(Debug, Clone)]
pub struct CurvatureSensor {
    /// Size of pupil image in pixels.
    pub n_pixels: usize,
    /// Physical pupil diameter in metres.
    pub pupil_diameter: f64,
    /// Defocus distance from focus (±z) in metres.
    pub defocus_distance: f64,
    /// Sensing wavelength in metres.
    pub wavelength: f64,
    /// Defocused intensity above focus (I₊).
    pub intensity_plus: Vec<f64>,
    /// Defocused intensity below focus (I₋).
    pub intensity_minus: Vec<f64>,
}

impl CurvatureSensor {
    /// Create a new curvature sensor.
    pub fn new(
        n_pixels: usize,
        pupil_diameter: f64,
        defocus_distance: f64,
        wavelength: f64,
    ) -> Self {
        let n2 = n_pixels * n_pixels;
        Self {
            n_pixels,
            pupil_diameter,
            defocus_distance,
            wavelength,
            intensity_plus: vec![1.0; n2],
            intensity_minus: vec![1.0; n2],
        }
    }

    /// Compute the curvature signal C(x,y) = (I₊ − I₋) / (I₊ + I₋).
    ///
    /// Returns a flat Vec of length `n_pixels²`.
    pub fn curvature_signal(&self) -> Vec<f64> {
        self.intensity_plus
            .iter()
            .zip(self.intensity_minus.iter())
            .map(|(&ip, &im)| {
                let total = ip + im;
                if total < 1e-30 {
                    0.0
                } else {
                    (ip - im) / total
                }
            })
            .collect()
    }

    /// Reconstruct the wavefront Laplacian from the curvature signal.
    ///
    /// Returns the proportional OPD map (in metres) using the simple
    /// relation: ∇²φ ≈ C / (f²/R) where f is the focal ratio and R is
    /// the defocus distance.
    ///
    /// This implementation uses a simple iterative Jacobi solver on the
    /// discrete Laplacian for the Poisson equation ∇²φ = S.
    pub fn reconstruct_wavefront(&self) -> Vec<f64> {
        let n = self.n_pixels;
        let n2 = n * n;
        let curvature = self.curvature_signal();

        // Scale factor: source term S = C * (λ * f²) / (2π * d)
        // where d = defocus_distance, f = focal_length (use pupil as proxy).
        let pixel_size = self.pupil_diameter / n as f64;
        let scale = pixel_size * pixel_size / (TWO_PI * self.defocus_distance.max(1e-10));

        let mut phi = vec![0.0_f64; n2];
        let source: Vec<f64> = curvature.iter().map(|&c| c * scale).collect();

        // Jacobi iteration for ∇²φ = S (Poisson equation).
        let mut phi_new = phi.clone();
        for _iter in 0..200 {
            for iy in 1..n - 1 {
                for ix in 1..n - 1 {
                    let idx = iy * n + ix;
                    let lap_neighbors = phi[(iy - 1) * n + ix]
                        + phi[(iy + 1) * n + ix]
                        + phi[iy * n + ix - 1]
                        + phi[iy * n + ix + 1];
                    phi_new[idx] = (lap_neighbors - source[idx]) * 0.25;
                }
            }
            phi.clone_from(&phi_new);
        }
        phi
    }

    /// RMS of the curvature signal (dimensionless).
    pub fn signal_rms(&self) -> f64 {
        let signal = self.curvature_signal();
        let n = signal.len() as f64;
        if n < 1.0 {
            return 0.0;
        }
        let mean = signal.iter().sum::<f64>() / n;
        let var = signal.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n;
        var.sqrt()
    }

    /// Load intensity images from flat arrays.
    pub fn set_intensities(
        &mut self,
        i_plus: Vec<f64>,
        i_minus: Vec<f64>,
    ) -> Result<(), OxiPhotonError> {
        let n2 = self.n_pixels * self.n_pixels;
        if i_plus.len() != n2 || i_minus.len() != n2 {
            return Err(OxiPhotonError::NumericalError(format!(
                "Intensity arrays must have length n_pixels² = {}, got {} and {}",
                n2,
                i_plus.len(),
                i_minus.len()
            )));
        }
        self.intensity_plus = i_plus;
        self.intensity_minus = i_minus;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shwfs_new_dimensions() {
        let wfs = ShackHartmannSensor::new(8, 8, 0.5e-3, 10e-3, 633e-9);
        assert_eq!(wfs.n_lenslets_x, 8);
        assert_eq!(wfs.n_lenslets_y, 8);
        assert_eq!(wfs.centroids.len(), 64);
        assert_eq!(wfs.reference_centroids.len(), 64);
    }

    #[test]
    fn test_shwfs_null_reference() {
        let mut wfs = ShackHartmannSensor::new(4, 4, 0.5e-3, 10e-3, 633e-9);
        // Set non-zero centroids.
        wfs.centroids[0] = [1e-5, 2e-5];
        wfs.null_reference();
        let slopes = wfs.measure_slopes();
        assert!(
            slopes[0][0].abs() < 1e-20,
            "After null, slope should be zero"
        );
        assert!(
            slopes[0][1].abs() < 1e-20,
            "After null, slope should be zero"
        );
    }

    #[test]
    fn test_shwfs_measure_slopes_offset() {
        let mut wfs = ShackHartmannSensor::new(4, 4, 0.5e-3, 10e-3, 633e-9);
        // Set a known centroid offset on sub-aperture 0.
        let offset_x = 5e-6; // 5 µm
        wfs.centroids[0] = [offset_x, 0.0];
        let slopes = wfs.measure_slopes();
        let expected_sx = offset_x / wfs.focal_length;
        assert!(
            (slopes[0][0] - expected_sx).abs() < 1e-18,
            "Slope x mismatch: {} vs {}",
            slopes[0][0],
            expected_sx
        );
    }

    #[test]
    fn test_shwfs_photon_noise_rms_increases_with_fewer_photons() {
        let wfs = ShackHartmannSensor::new(8, 8, 0.5e-3, 10e-3, 633e-9);
        let rms_100 = wfs.photon_noise_rms(100.0);
        let rms_1000 = wfs.photon_noise_rms(1000.0);
        assert!(rms_100 > rms_1000, "Fewer photons should give larger noise");
    }

    #[test]
    fn test_shwfs_photon_noise_zero_photons() {
        let wfs = ShackHartmannSensor::new(8, 8, 0.5e-3, 10e-3, 633e-9);
        let rms = wfs.photon_noise_rms(0.0);
        assert!(rms.is_infinite(), "Zero photons should give infinite noise");
    }

    #[test]
    fn test_shwfs_reconstruct_flat() {
        // Flat wavefront → all slopes zero → reconstructed wavefront all zero.
        let wfs = ShackHartmannSensor::new(4, 4, 0.5e-3, 10e-3, 633e-9);
        let wf = wfs.reconstruct_wavefront();
        for v in &wf {
            assert!(v.abs() < 1e-30, "Flat reconstruction should be zero");
        }
    }

    #[test]
    fn test_shwfs_set_slopes_wrong_size() {
        let mut wfs = ShackHartmannSensor::new(4, 4, 0.5e-3, 10e-3, 633e-9);
        let bad_slopes = vec![vec![0.0_f64; 3]; 3]; // wrong size
        let result = wfs.set_slopes_from_arrays(&bad_slopes, &bad_slopes);
        assert!(result.is_err());
    }

    #[test]
    fn test_pyramid_intensity_signals_sum_to_one() {
        let ps = PyramidSensor::new(16, 1e-3, 633e-9);
        let slopes_x = vec![0.0_f64; 256];
        let slopes_y = vec![0.0_f64; 256];
        let imgs = ps.intensity_signals(&slopes_x, &slopes_y);
        // For zero slopes, each quadrant should be 0.25.
        for (i, _) in imgs[0].iter().enumerate() {
            let total = imgs[0][i] + imgs[1][i] + imgs[2][i] + imgs[3][i];
            assert!(
                (total - 1.0).abs() < 1e-12,
                "Quadrant intensities should sum to 1, got {}",
                total
            );
        }
    }

    #[test]
    fn test_pyramid_round_trip_slopes() {
        let ps = PyramidSensor::new(16, 1e-3, 633e-9);
        let sx_in = vec![5e-4_f64; 256];
        let sy_in = vec![-3e-4_f64; 256];
        let imgs = ps.intensity_signals(&sx_in, &sy_in);
        // Convert to quadrant image format.
        let quad_images: Vec<[f64; 4]> = (0..256)
            .map(|i| [imgs[0][i], imgs[1][i], imgs[2][i], imgs[3][i]])
            .collect();
        let (sx_out, sy_out) = ps.reconstruct_slopes(&quad_images);
        assert!(
            (sx_out[0] - sx_in[0]).abs() < 1e-10,
            "Round-trip sx mismatch: {} vs {}",
            sx_out[0],
            sx_in[0]
        );
        assert!(
            (sy_out[0] - sy_in[0]).abs() < 1e-10,
            "Round-trip sy mismatch: {} vs {}",
            sy_out[0],
            sy_in[0]
        );
    }

    #[test]
    fn test_pyramid_sensitivity() {
        let mod_r = 2e-3_f64;
        let ps = PyramidSensor::new(16, mod_r, 633e-9);
        let expected = 1.0 / mod_r;
        assert!(
            (ps.sensitivity() - expected).abs() < 1e-10,
            "Sensitivity mismatch"
        );
    }

    #[test]
    fn test_curvature_sensor_flat_signal() {
        // Equal intensities → curvature signal = 0 everywhere.
        let cs = CurvatureSensor::new(16, 4e-3, 1e-3, 633e-9);
        let signal = cs.curvature_signal();
        for &v in &signal {
            assert!(v.abs() < 1e-12, "Flat curvature signal should be 0");
        }
    }

    #[test]
    fn test_curvature_sensor_set_intensities_wrong_size() {
        let mut cs = CurvatureSensor::new(8, 4e-3, 1e-3, 633e-9);
        let bad = vec![1.0_f64; 10]; // should be 64
        let result = cs.set_intensities(bad.clone(), bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_curvature_sensor_signal_rms_nonzero() {
        let mut cs = CurvatureSensor::new(8, 4e-3, 1e-3, 633e-9);
        let n2 = 64_usize;
        // Create an asymmetric intensity distribution.
        let i_plus: Vec<f64> = (0..n2)
            .map(|i| 1.0 + 0.1 * (i as f64 / 8.0).sin())
            .collect();
        let i_minus: Vec<f64> = (0..n2)
            .map(|i| 1.0 - 0.1 * (i as f64 / 8.0).sin())
            .collect();
        let _ = cs.set_intensities(i_plus, i_minus);
        let rms = cs.signal_rms();
        assert!(rms > 0.0, "RMS of non-flat curvature signal should be > 0");
    }
}
