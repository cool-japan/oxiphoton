//! Photoacoustic tomography (PAT / PACT) reconstruction algorithms
//!
//! Implements:
//! - Delay-and-sum (DAS) beamforming for 2-D image reconstruction
//! - Circular detector array geometry for full-view PACT
//! - PAT resolution metrics (axial, lateral, penetration depth)

use std::f64::consts::PI;

/// Delay-and-sum beamformer for 2-D photoacoustic tomography.
///
/// Each detector records a time-resolved pressure signal p_i(t).
/// The absorber at position (x, y) is reconstructed by summing signal
/// values delayed by the appropriate time of flight:
///
///   I(x, y) = Σ_i p_i( t_i(x, y) )
///
/// where t_i = √\[(x−x_i)² + (y−y_i)²\] / c_s.
#[derive(Debug, Clone)]
pub struct DelayAndSumBeamformer {
    /// Speed of sound in tissue (m/s)
    pub c_sound: f64,
    /// Detector positions \[(x_i, y_i)\] in metres
    pub detector_positions: Vec<[f64; 2]>,
    /// Time-sampling interval Δt (s)
    pub dt_s: f64,
}

impl DelayAndSumBeamformer {
    /// Time of arrival from detector `det_idx` to pixel (x, y).
    ///
    /// t_i = √\[(x−x_i)² + (y−y_i)²\] / c_s
    pub fn time_of_arrival(&self, det_idx: usize, x: f64, y: f64) -> f64 {
        if det_idx >= self.detector_positions.len() {
            return 0.0;
        }
        let [xi, yi] = self.detector_positions[det_idx];
        let dx = x - xi;
        let dy = y - yi;
        (dx * dx + dy * dy).sqrt() / self.c_sound
    }

    /// Reconstruct the initial pressure at a single pixel (x, y).
    ///
    /// Linearly interpolates between adjacent time samples for sub-sample accuracy.
    pub fn reconstruct_point(&self, signals: &[Vec<f64>], x: f64, y: f64) -> f64 {
        let n_det = self.detector_positions.len().min(signals.len());
        if n_det == 0 || self.dt_s <= 0.0 {
            return 0.0;
        }

        let mut sum = 0.0_f64;
        for (i, sig) in signals.iter().enumerate().take(n_det) {
            let toa = self.time_of_arrival(i, x, y);
            let sample_frac = toa / self.dt_s;
            let sample_idx = sample_frac as usize;
            let frac = sample_frac - sample_idx as f64;

            if sig.is_empty() {
                continue;
            }

            let s0 = if sample_idx < sig.len() {
                sig[sample_idx]
            } else {
                0.0
            };
            let s1 = if sample_idx + 1 < sig.len() {
                sig[sample_idx + 1]
            } else {
                0.0
            };
            sum += s0 * (1.0 - frac) + s1 * frac;
        }
        sum
    }

    /// Reconstruct a full 2-D image on a regular grid.
    ///
    /// Returns a `ny × nx` matrix (row = y index, col = x index)
    /// of reconstructed initial pressure values.
    pub fn reconstruct_image(
        &self,
        signals: &[Vec<f64>],
        x_range: (f64, f64),
        y_range: (f64, f64),
        nx: usize,
        ny: usize,
    ) -> Vec<Vec<f64>> {
        let mut image = vec![vec![0.0_f64; nx]; ny];

        if nx == 0 || ny == 0 {
            return image;
        }

        let dx = if nx > 1 {
            (x_range.1 - x_range.0) / (nx - 1) as f64
        } else {
            0.0
        };
        let dy = if ny > 1 {
            (y_range.1 - y_range.0) / (ny - 1) as f64
        } else {
            0.0
        };

        for (iy, image_row) in image.iter_mut().enumerate().take(ny) {
            let y = y_range.0 + iy as f64 * dy;
            for (ix, pixel) in image_row.iter_mut().enumerate().take(nx) {
                let x = x_range.0 + ix as f64 * dx;
                *pixel = self.reconstruct_point(signals, x, y);
            }
        }
        image
    }
}

/// Circular piezoelectric detector (PET) array for full-view PACT.
///
/// Detectors are uniformly distributed around a circle of radius R.
/// Full angular coverage (360°) provides equal spatial resolution in
/// all directions (isotropic PSF).
#[derive(Debug, Clone)]
pub struct CircularPetArray {
    /// Array radius (m)
    pub radius_m: f64,
    /// Number of detector elements
    pub n_detectors: usize,
    /// Speed of sound in coupling medium (m/s)
    pub c_sound: f64,
    /// Detector bandwidth (Hz, single-sided −6 dB)
    pub bandwidth_hz: f64,
}

impl CircularPetArray {
    /// Cartesian positions of all detector elements.
    ///
    /// θ_i = 2π i / N, position = (R cos θ_i, R sin θ_i)
    pub fn detector_positions(&self) -> Vec<[f64; 2]> {
        (0..self.n_detectors)
            .map(|i| {
                let theta = 2.0 * PI * (i as f64) / (self.n_detectors as f64).max(1.0);
                [self.radius_m * theta.cos(), self.radius_m * theta.sin()]
            })
            .collect()
    }

    /// Spatial resolution (m) limited by detector bandwidth.
    ///
    /// For a full-view circular array with bandwidth BW, both axial and
    /// lateral resolution are approximately equal:
    ///   Δr ≈ 0.88 c_s / (2 BW)
    pub fn spatial_resolution_m(&self) -> f64 {
        0.88 * self.c_sound / (2.0 * self.bandwidth_hz)
    }

    /// Field of view: diameter of the region that all detectors can image
    /// without aliasing ≈ array radius (limited by angular coverage).
    pub fn field_of_view_m(&self) -> f64 {
        self.radius_m
    }

    /// Nyquist sampling frequency derived from the round-trip travel time.
    ///
    /// The maximum acoustic frequency that can be detected without temporal
    /// aliasing at the Nyquist criterion.
    pub fn nyquist_frequency(&self) -> f64 {
        // Maximum time of flight ≈ 2 R / c_s (diameter traversal); Nyquist ≈ N_t / (2 × T_max)
        // Simplified: based on 1000 time samples over 2R/c_s
        0.5 / (2.0 * self.radius_m / self.c_sound / 1000.0)
    }

    /// Angular sampling interval (degrees) between adjacent detectors.
    pub fn angular_step_deg(&self) -> f64 {
        360.0 / self.n_detectors as f64
    }

    /// Total acquisition time (s) for a single-shot full-view scan.
    ///
    /// Equals the maximum time of flight: t_max = 2R / c_s
    pub fn acquisition_time_s(&self) -> f64 {
        2.0 * self.radius_m / self.c_sound
    }

    /// Build a DAS beamformer from this array geometry.
    pub fn to_das_beamformer(&self, dt_s: f64) -> DelayAndSumBeamformer {
        DelayAndSumBeamformer {
            c_sound: self.c_sound,
            detector_positions: self.detector_positions(),
            dt_s,
        }
    }
}

/// PAT spatial resolution and penetration depth metrics.
///
/// Quantifies the trade-off between resolution (higher frequency = finer)
/// and penetration depth (higher frequency = more ultrasound attenuation).
#[derive(Debug, Clone)]
pub struct PatResolution {
    /// Speed of sound (m/s)
    pub c_sound: f64,
    /// Centre frequency of the detector (Hz)
    pub center_frequency_hz: f64,
    /// Fractional bandwidth BW/f₀ (typically 0.6–0.8 for wideband PZT)
    pub bandwidth_fraction: f64,
    /// Numerical aperture = sin(half-angle of detector aperture)
    pub numerical_aperture: f64,
}

impl PatResolution {
    /// Axial (depth) resolution limited by detector bandwidth.
    ///
    /// Δz ≈ 0.88 c_s / BW = 0.88 c_s / (f₀ × BW_fraction)
    pub fn axial_resolution_m(&self) -> f64 {
        let bw = self.center_frequency_hz * self.bandwidth_fraction;
        0.88 * self.c_sound / bw
    }

    /// Lateral (transverse) resolution at the focal plane.
    ///
    /// Δx ≈ c_s / (2 NA f₀)  (diffraction-limited acoustic focus)
    pub fn lateral_resolution_m(&self) -> f64 {
        self.c_sound / (self.center_frequency_hz * 2.0 * self.numerical_aperture)
    }

    /// Acoustic wavelength at the centre frequency.
    ///
    /// λ_ac = c_s / f₀
    pub fn acoustic_wavelength_m(&self) -> f64 {
        self.c_sound / self.center_frequency_hz
    }

    /// Penetration depth limited by ultrasound attenuation in tissue.
    ///
    /// Soft tissue attenuation: α ≈ 0.5 dB cm⁻¹ MHz⁻¹ → depth ≈ 10/(α × f₀).
    /// The `us_attenuation_db_cm_mhz` parameter allows specifying the coefficient
    /// for different media.
    ///
    /// d_pen ≈ SNR_threshold / (α × f₀)   \[cm\] → converted to \[m\]
    pub fn penetration_depth_m(&self, us_attenuation_db_cm_mhz: f64) -> f64 {
        let f_mhz = self.center_frequency_hz / 1.0e6;
        // 10 dB SNR at threshold
        let depth_cm = 10.0 / (us_attenuation_db_cm_mhz * f_mhz).max(1.0e-12);
        depth_cm * 1.0e-2
    }

    /// Ratio of penetration depth to axial resolution (depth-resolution product).
    ///
    /// Higher is better; typical values 50–500 for PAT systems.
    pub fn depth_resolution_product(&self, us_attenuation_db_cm_mhz: f64) -> f64 {
        self.penetration_depth_m(us_attenuation_db_cm_mhz)
            / self.axial_resolution_m().max(f64::EPSILON)
    }
}

/// Filtered back-projection weight function for PAT reconstruction.
///
/// Implements the weighting function from the universal back-projection
/// formula (Xu & Wang 2005):
///   w(r, θ) = cos(angle between detector normal and direction to pixel)
pub fn back_projection_weight(
    detector_pos: [f64; 2],
    pixel_pos: [f64; 2],
    detector_normal: [f64; 2],
) -> f64 {
    let dx = pixel_pos[0] - detector_pos[0];
    let dy = pixel_pos[1] - detector_pos[1];
    let dist = (dx * dx + dy * dy).sqrt();
    if dist < 1.0e-15 {
        return 1.0;
    }
    // Dot product of unit direction with detector normal
    let ux = dx / dist;
    let uy = dy / dist;
    (ux * detector_normal[0] + uy * detector_normal[1]).abs()
}

/// Universal back-projection reconstructor for PAT.
///
/// Implements the continuous limit back-projection:
///   p₀(r) ∝ Σ_i ∫ \[p(r_i, t) − t ∂p/∂t\] t=|r−r_i|/c_s  × cos(θ_i) dt
///
/// This produces artefact-reduced reconstructions compared to simple DAS.
#[derive(Debug, Clone)]
pub struct UniversalBackProjection {
    /// DAS beamformer providing geometry and time-delay logic
    pub das: DelayAndSumBeamformer,
}

impl UniversalBackProjection {
    /// Reconstruct a pixel using the universal back-projection formula.
    ///
    /// Applies the t × ∂p/∂t correction (finite difference approximation)
    /// and cosine weighting before summation.
    pub fn reconstruct_point(&self, signals: &[Vec<f64>], x: f64, y: f64) -> f64 {
        let n_det = self.das.detector_positions.len().min(signals.len());
        if n_det == 0 || self.das.dt_s <= 0.0 {
            return 0.0;
        }

        let mut sum = 0.0_f64;
        for (i, &[xi, yi]) in self.das.detector_positions.iter().enumerate().take(n_det) {
            let toa = self.das.time_of_arrival(i, x, y);
            let sample_frac = toa / self.das.dt_s;
            let sample_idx = sample_frac as usize;

            let sig = &signals[i];
            if sig.len() < 2 {
                continue;
            }

            // Interpolated pressure at t = toa
            let frac = sample_frac - sample_idx as f64;
            let p_val = if sample_idx < sig.len() {
                let s0 = sig[sample_idx];
                let s1 = if sample_idx + 1 < sig.len() {
                    sig[sample_idx + 1]
                } else {
                    0.0
                };
                s0 * (1.0 - frac) + s1 * frac
            } else {
                0.0
            };

            // Time derivative dp/dt (finite difference)
            let dp_dt = if sample_idx + 1 < sig.len() && sample_idx > 0 {
                (sig[sample_idx + 1] - sig[sample_idx - 1]) / (2.0 * self.das.dt_s)
            } else {
                0.0
            };

            // UBP correction: [p - t dp/dt]
            let ubp_val = p_val - toa * dp_dt;

            // Cosine weight: normal pointing toward centre (for circular array, n_i = -pos_i/|pos_i|)
            let dist_det = (xi * xi + yi * yi).sqrt().max(f64::EPSILON);
            let nx = -xi / dist_det;
            let ny = -yi / dist_det;
            // Direction from detector to pixel
            let dx = x - xi;
            let dy = y - yi;
            let dist_pix = (dx * dx + dy * dy).sqrt().max(f64::EPSILON);
            let cos_w = ((dx / dist_pix) * nx + (dy / dist_pix) * ny).abs();

            sum += ubp_val * cos_w;
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pat_axial_resolution() {
        let res = PatResolution {
            c_sound: 1540.0,
            center_frequency_hz: 5.0e6,
            bandwidth_fraction: 0.8,
            numerical_aperture: 0.5,
        };
        let dz = res.axial_resolution_m();
        // Δz = 0.88 × 1540 / (5e6 × 0.8) = 0.88 × 1540 / 4e6 ≈ 339 µm
        assert!(dz > 100.0e-6 && dz < 1000.0e-6, "Δz={}µm", dz * 1.0e6);
    }

    #[test]
    fn pat_lateral_resolution() {
        let res = PatResolution {
            c_sound: 1540.0,
            center_frequency_hz: 5.0e6,
            bandwidth_fraction: 0.8,
            numerical_aperture: 0.5,
        };
        let dx = res.lateral_resolution_m();
        // Δx = 1540 / (5e6 × 2 × 0.5) = 1540/5e6 = 308 µm
        assert!(dx > 50.0e-6 && dx < 1000.0e-6, "Δx={}µm", dx * 1.0e6);
    }

    #[test]
    fn penetration_depth_5mhz() {
        let res = PatResolution {
            c_sound: 1540.0,
            center_frequency_hz: 5.0e6,
            bandwidth_fraction: 0.8,
            numerical_aperture: 0.5,
        };
        // α = 0.5 dB/cm/MHz → depth = 10/(0.5×5) = 4 cm = 0.04 m
        let depth = res.penetration_depth_m(0.5);
        assert!(depth > 0.01 && depth < 0.20, "depth={}cm", depth * 100.0);
    }

    #[test]
    fn circular_array_positions_count() {
        let arr = CircularPetArray {
            radius_m: 0.05,
            n_detectors: 128,
            c_sound: 1540.0,
            bandwidth_hz: 5.0e6,
        };
        let pos = arr.detector_positions();
        assert_eq!(pos.len(), 128);
    }

    #[test]
    fn circular_array_positions_on_circle() {
        let r = 0.05_f64;
        let arr = CircularPetArray {
            radius_m: r,
            n_detectors: 64,
            c_sound: 1540.0,
            bandwidth_hz: 5.0e6,
        };
        for [x, y] in arr.detector_positions() {
            let dist = (x * x + y * y).sqrt();
            assert!(
                (dist - r).abs() < 1.0e-12,
                "detector not on circle: dist={}",
                dist
            );
        }
    }

    #[test]
    fn das_time_of_arrival() {
        let das = DelayAndSumBeamformer {
            c_sound: 1540.0,
            detector_positions: vec![[0.0, 0.05]],
            dt_s: 1.0e-8,
        };
        // Pixel at origin, detector at (0, 0.05 m) → t = 0.05/1540 ≈ 32.47 µs
        let toa = das.time_of_arrival(0, 0.0, 0.0);
        let expected = 0.05 / 1540.0;
        assert!((toa - expected).abs() < 1.0e-12, "toa={}", toa);
    }

    #[test]
    fn das_reconstruct_small_grid() {
        let das = DelayAndSumBeamformer {
            c_sound: 1540.0,
            detector_positions: vec![[0.0, 0.05], [-0.05, 0.0], [0.05, 0.0]],
            dt_s: 1.0e-8,
        };
        // Constant-signal detectors → reconstruction should be non-zero everywhere
        let n_samples = 5000;
        let signals: Vec<Vec<f64>> = (0..3).map(|_| vec![1.0_f64; n_samples]).collect();
        let img = das.reconstruct_image(&signals, (-0.02, 0.02), (-0.02, 0.02), 4, 4);
        assert_eq!(img.len(), 4);
        assert_eq!(img[0].len(), 4);
        // All pixels should have same non-zero value (constant signals)
        for row in &img {
            for &v in row {
                assert!(v > 0.0, "Expected positive reconstruction, got {}", v);
            }
        }
    }

    #[test]
    fn back_projection_weight_orthogonal() {
        // Detector at (0, 1), normal pointing down (0, -1), pixel at (0, 0)
        // Direction from det to pixel = (0, -1), cos_angle = |(0,-1)·(0,-1)| = 1
        let w = back_projection_weight([0.0, 1.0], [0.0, 0.0], [0.0, -1.0]);
        assert!((w - 1.0).abs() < 1.0e-10, "w={}", w);
    }
}
