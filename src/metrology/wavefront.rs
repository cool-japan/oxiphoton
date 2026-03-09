//! Wavefront sensing and analysis.
//!
//! Provides:
//! - [`WavefrontMap`]: 2D pupil-plane wavefront error map (in waves)
//! - [`HartmannShack`]: Hartmann-Shack wavefront sensor simulation and reconstruction
//! - [`PsfAnalysis`]: Point spread function analysis (Airy disk, MTF, Strehl)
//!
//! Conventions follow Born & Wolf "Principles of Optics" and Mahajan
//! "Optical Imaging and Aberrations".

use num_complex::Complex64;
use oxifft::fft2d;

use crate::error::OxiPhotonError;

const PI: f64 = std::f64::consts::PI;
const TWO_PI: f64 = 2.0 * PI;

// ────────────────────────────────────────────────────────────────────────────
// Zernike basis (ANSI/OSA single-index ordering, up to j = 14)
// ────────────────────────────────────────────────────────────────────────────

/// Evaluate ANSI/OSA single-index Zernike polynomial at (ρ, θ).
///
/// j = 0 : piston, 1 : tip, 2 : tilt, 3 : defocus, …
/// Polynomials are NOT normalised (raw basis for projection).
fn zernike_basis(j: usize, rho: f64, theta: f64) -> f64 {
    let r2 = rho * rho;
    let r3 = r2 * rho;
    let r4 = r2 * r2;
    match j {
        0 => 1.0,
        1 => rho * theta.cos(),
        2 => rho * theta.sin(),
        3 => 2.0 * r2 - 1.0,
        4 => r2 * (2.0 * theta).cos(),
        5 => r2 * (2.0 * theta).sin(),
        6 => (3.0 * r2 - 2.0) * rho * theta.cos(),
        7 => (3.0 * r2 - 2.0) * rho * theta.sin(),
        8 => 6.0 * r4 - 6.0 * r2 + 1.0,
        9 => r3 * (3.0 * theta).cos(),
        10 => r3 * (3.0 * theta).sin(),
        11 => (4.0 * r2 - 3.0) * r2 * (2.0 * theta).cos(),
        12 => (4.0 * r2 - 3.0) * r2 * (2.0 * theta).sin(),
        13 => (10.0 * r4 - 12.0 * r2 + 3.0) * rho * theta.cos(),
        14 => (10.0 * r4 - 12.0 * r2 + 3.0) * rho * theta.sin(),
        _ => 0.0,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// WavefrontMap
// ────────────────────────────────────────────────────────────────────────────

/// 2D wavefront error map W(x,y) in units of wavelengths.
///
/// The pupil is a circle of diameter `pupil_diameter_mm` centred on the grid.
/// Points outside the pupil are set to 0.
#[derive(Debug, Clone)]
pub struct WavefrontMap {
    /// Wavefront error, row-major (row = y, col = x), in waves.
    pub data: Vec<Vec<f64>>,
    /// Number of columns (x-samples).
    pub nx: usize,
    /// Number of rows (y-samples).
    pub ny: usize,
    /// Pupil diameter in mm.
    pub pupil_diameter_mm: f64,
    /// Reference wavelength in nm.
    pub wavelength_nm: f64,
}

impl WavefrontMap {
    /// Create a flat (zero) wavefront map.
    pub fn new(nx: usize, ny: usize, pupil_diameter_mm: f64, wavelength_nm: f64) -> Self {
        Self {
            data: vec![vec![0.0; nx]; ny],
            nx,
            ny,
            pupil_diameter_mm,
            wavelength_nm,
        }
    }

    /// Build a wavefront from a list of `(zernike_index_j, amplitude_waves)` pairs.
    ///
    /// Uses ANSI/OSA single-index ordering (j = 0 = piston).
    /// Points outside the unit-radius pupil are forced to 0.
    pub fn from_zernike_coefficients(
        coeffs: &[(usize, f64)],
        nx: usize,
        ny: usize,
        pupil_diameter_mm: f64,
        wavelength_nm: f64,
    ) -> Self {
        let mut data = vec![vec![0.0_f64; nx]; ny];
        let cx = (nx as f64 - 1.0) * 0.5;
        let cy = (ny as f64 - 1.0) * 0.5;
        let r_pix = (nx.min(ny) as f64 - 1.0) * 0.5;

        for (iy, data_row) in data.iter_mut().enumerate().take(ny) {
            for (ix, data_cell) in data_row.iter_mut().enumerate().take(nx) {
                let xn = (ix as f64 - cx) / r_pix;
                let yn = (iy as f64 - cy) / r_pix;
                let rho2 = xn * xn + yn * yn;
                if rho2 > 1.0 {
                    continue;
                }
                let rho = rho2.sqrt();
                let theta = yn.atan2(xn);
                let w: f64 = coeffs
                    .iter()
                    .map(|&(j, amp)| amp * zernike_basis(j, rho, theta))
                    .sum();
                *data_cell = w;
            }
        }
        Self {
            data,
            nx,
            ny,
            pupil_diameter_mm,
            wavelength_nm,
        }
    }

    /// Collect all in-pupil samples.
    fn pupil_samples(&self) -> Vec<f64> {
        let cx = (self.nx as f64 - 1.0) * 0.5;
        let cy = (self.ny as f64 - 1.0) * 0.5;
        let r_pix = (self.nx.min(self.ny) as f64 - 1.0) * 0.5;
        let mut vals = Vec::with_capacity(self.nx * self.ny);
        for iy in 0..self.ny {
            for ix in 0..self.nx {
                let xn = (ix as f64 - cx) / r_pix;
                let yn = (iy as f64 - cy) / r_pix;
                if xn * xn + yn * yn <= 1.0 {
                    vals.push(self.data[iy][ix]);
                }
            }
        }
        vals
    }

    /// RMS wavefront error in waves (variance over in-pupil samples).
    pub fn rms_waves(&self) -> f64 {
        let samples = self.pupil_samples();
        if samples.is_empty() {
            return 0.0;
        }
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let var = samples
            .iter()
            .map(|&w| (w - mean) * (w - mean))
            .sum::<f64>()
            / n;
        var.sqrt()
    }

    /// Peak-to-valley wavefront error in waves.
    pub fn pv_waves(&self) -> f64 {
        let samples = self.pupil_samples();
        if samples.is_empty() {
            return 0.0;
        }
        let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        max - min
    }

    /// RMS wavefront error in nanometres.
    pub fn rms_nm(&self) -> f64 {
        self.rms_waves() * self.wavelength_nm
    }

    /// Strehl ratio via the Maréchal approximation: S = exp(−(2π·σ)²).
    ///
    /// Valid for σ < λ/14 (about 0.07 waves).
    pub fn strehl_marechal(&self) -> f64 {
        let sigma = self.rms_waves();
        let phase = TWO_PI * sigma;
        (-phase * phase).exp()
    }

    /// Strehl ratio computed from PSF peak relative to the diffraction-limited peak.
    ///
    /// Uses an 128×128 PSF grid. For a flat wavefront this equals 1.
    pub fn strehl_fourier(&self) -> f64 {
        let n_psf = 128;
        let abberated = self.psf(n_psf);
        let flat = Self::new(self.nx, self.ny, self.pupil_diameter_mm, self.wavelength_nm);
        let dl_psf = flat.psf(n_psf);

        let abberated_peak = abberated
            .iter()
            .flat_map(|row| row.iter().cloned())
            .fold(0.0_f64, f64::max);
        let dl_peak = dl_psf
            .iter()
            .flat_map(|row| row.iter().cloned())
            .fold(0.0_f64, f64::max);

        if dl_peak < 1e-30 {
            return 1.0;
        }
        (abberated_peak / dl_peak).min(1.0)
    }

    /// Least-squares Zernike fit to the wavefront data.
    ///
    /// Returns Zernike coefficients (in waves) for j = 0..n_terms.
    pub fn fit_zernike(&self, n_terms: usize) -> Vec<f64> {
        let n_terms = n_terms.min(15);
        let cx = (self.nx as f64 - 1.0) * 0.5;
        let cy = (self.ny as f64 - 1.0) * 0.5;
        let r_pix = (self.nx.min(self.ny) as f64 - 1.0) * 0.5;

        let mut num = vec![0.0_f64; n_terms];
        let mut den = vec![0.0_f64; n_terms];

        for iy in 0..self.ny {
            for ix in 0..self.nx {
                let xn = (ix as f64 - cx) / r_pix;
                let yn = (iy as f64 - cy) / r_pix;
                let rho2 = xn * xn + yn * yn;
                if rho2 > 1.0 {
                    continue;
                }
                let rho = rho2.sqrt();
                let theta = yn.atan2(xn);
                let w = self.data[iy][ix];
                for j in 0..n_terms {
                    let z = zernike_basis(j, rho, theta);
                    num[j] += w * z;
                    den[j] += z * z;
                }
            }
        }
        (0..n_terms)
            .map(|j| {
                if den[j].abs() < 1e-30 {
                    0.0
                } else {
                    num[j] / den[j]
                }
            })
            .collect()
    }

    /// Residual wavefront after subtracting the best n-term Zernike fit.
    pub fn zernike_residual(&self, n_terms: usize) -> WavefrontMap {
        let coeffs = self.fit_zernike(n_terms);
        let cx = (self.nx as f64 - 1.0) * 0.5;
        let cy = (self.ny as f64 - 1.0) * 0.5;
        let r_pix = (self.nx.min(self.ny) as f64 - 1.0) * 0.5;

        let mut data = vec![vec![0.0_f64; self.nx]; self.ny];
        for (iy, data_row) in data.iter_mut().enumerate().take(self.ny) {
            for (ix, data_cell) in data_row.iter_mut().enumerate().take(self.nx) {
                let xn = (ix as f64 - cx) / r_pix;
                let yn = (iy as f64 - cy) / r_pix;
                let rho2 = xn * xn + yn * yn;
                if rho2 > 1.0 {
                    continue;
                }
                let rho = rho2.sqrt();
                let theta = yn.atan2(xn);
                let fit: f64 = coeffs
                    .iter()
                    .enumerate()
                    .map(|(j, &c)| c * zernike_basis(j, rho, theta))
                    .sum();
                *data_cell = self.data[iy][ix] - fit;
            }
        }
        WavefrontMap {
            data,
            nx: self.nx,
            ny: self.ny,
            pupil_diameter_mm: self.pupil_diameter_mm,
            wavelength_nm: self.wavelength_nm,
        }
    }

    /// Complex pupil function P(x,y) = A(x,y)·exp(i·2π·W(x,y)).
    ///
    /// Amplitude A = 1 inside the pupil, 0 outside.
    pub fn pupil_function(&self) -> Vec<Vec<Complex64>> {
        let cx = (self.nx as f64 - 1.0) * 0.5;
        let cy = (self.ny as f64 - 1.0) * 0.5;
        let r_pix = (self.nx.min(self.ny) as f64 - 1.0) * 0.5;

        let mut pf = vec![vec![Complex64::new(0.0, 0.0); self.nx]; self.ny];
        for (iy, pf_row) in pf.iter_mut().enumerate().take(self.ny) {
            for (ix, pf_cell) in pf_row.iter_mut().enumerate().take(self.nx) {
                let xn = (ix as f64 - cx) / r_pix;
                let yn = (iy as f64 - cy) / r_pix;
                if xn * xn + yn * yn <= 1.0 {
                    let phase = TWO_PI * self.data[iy][ix];
                    *pf_cell = Complex64::from_polar(1.0, phase);
                }
            }
        }
        pf
    }

    /// PSF as |FT\[P\]|² normalised so that the diffraction-limited peak equals 1.
    ///
    /// The pupil function is zero-padded to `n_psf × n_psf` before the FFT.
    /// The result is centred (DC-shifted).
    pub fn psf(&self, n_psf: usize) -> Vec<Vec<f64>> {
        // Build zero-padded pupil in a flat buffer (n_psf × n_psf).
        let cx_p = (self.nx as f64 - 1.0) * 0.5;
        let cy_p = (self.ny as f64 - 1.0) * 0.5;
        let r_pix = (self.nx.min(self.ny) as f64 - 1.0) * 0.5;

        let off_x = (n_psf.saturating_sub(self.nx)) / 2;
        let off_y = (n_psf.saturating_sub(self.ny)) / 2;

        let mut flat = vec![oxifft::kernel::Complex::new(0.0_f64, 0.0_f64); n_psf * n_psf];

        for iy in 0..self.ny.min(n_psf) {
            for ix in 0..self.nx.min(n_psf) {
                let xn = (ix as f64 - cx_p) / r_pix;
                let yn = (iy as f64 - cy_p) / r_pix;
                let amplitude = if xn * xn + yn * yn <= 1.0 { 1.0 } else { 0.0 };
                let phase = TWO_PI * self.data[iy][ix];
                let oy = iy + off_y;
                let ox = ix + off_x;
                if oy < n_psf && ox < n_psf {
                    flat[oy * n_psf + ox] = oxifft::kernel::Complex::new(
                        amplitude * phase.cos(),
                        amplitude * phase.sin(),
                    );
                }
            }
        }

        // 2D FFT via oxifft.
        let spectrum = fft2d(&flat, n_psf, n_psf);

        // |FFT|² with fft-shift: swap quadrants.
        let mut psf_raw = vec![vec![0.0_f64; n_psf]; n_psf];
        let half = n_psf / 2;
        for ky in 0..n_psf {
            for kx in 0..n_psf {
                let v = spectrum[ky * n_psf + kx];
                let i2 = v.re * v.re + v.im * v.im;
                // fft-shift indices.
                let sy = (ky + half) % n_psf;
                let sx = (kx + half) % n_psf;
                psf_raw[sy][sx] = i2;
            }
        }

        // Normalise so that the peak = 1 for a flat wavefront is n_pupils²
        // (actual normalisation: divide by sum so it represents fractional intensity).
        let total: f64 = psf_raw.iter().flat_map(|r| r.iter().cloned()).sum();
        if total > 1e-30 {
            for row in &mut psf_raw {
                for v in row.iter_mut() {
                    *v /= total;
                }
            }
        }
        psf_raw
    }

    /// Encircled energy fraction within `radius_pixels` of the PSF centroid.
    pub fn encircled_energy(&self, psf: &[Vec<f64>], radius_pixels: f64) -> f64 {
        let nrows = psf.len();
        if nrows == 0 {
            return 0.0;
        }
        let ncols = psf[0].len();

        // Find centroid.
        let mut sum_total = 0.0_f64;
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        for (iy, row) in psf.iter().enumerate() {
            for (ix, &v) in row.iter().enumerate() {
                sum_total += v;
                sum_x += v * ix as f64;
                sum_y += v * iy as f64;
            }
        }
        if sum_total < 1e-30 {
            return 0.0;
        }
        let cx = sum_x / sum_total;
        let cy = sum_y / sum_total;

        // Encircled energy.
        let r2 = radius_pixels * radius_pixels;
        let mut ee = 0.0_f64;
        for (iy, psf_row) in psf.iter().enumerate().take(nrows) {
            for (ix, &psf_cell) in psf_row.iter().enumerate().take(ncols) {
                let dx = ix as f64 - cx;
                let dy = iy as f64 - cy;
                if dx * dx + dy * dy <= r2 {
                    ee += psf_cell;
                }
            }
        }
        ee / sum_total
    }

    /// Extract tip (j=1) and tilt (j=2) Zernike coefficients in waves.
    pub fn tip_tilt_waves(&self) -> (f64, f64) {
        let c = self.fit_zernike(3);
        (c[1], c[2])
    }

    /// Extract defocus (j=3) Zernike coefficient in waves.
    pub fn defocus_waves(&self) -> f64 {
        let c = self.fit_zernike(4);
        c[3]
    }

    /// RMS of astigmatism terms (j=4,5) in waves.
    pub fn astigmatism_waves(&self) -> f64 {
        let c = self.fit_zernike(6);
        (c[4] * c[4] + c[5] * c[5]).sqrt()
    }

    /// Add two wavefront maps (must have identical nx, ny).
    pub fn add(&self, other: &WavefrontMap) -> Result<WavefrontMap, OxiPhotonError> {
        if self.nx != other.nx || self.ny != other.ny {
            return Err(OxiPhotonError::NumericalError(format!(
                "WavefrontMap size mismatch: ({},{}) vs ({},{})",
                self.nx, self.ny, other.nx, other.ny
            )));
        }
        let mut data = vec![vec![0.0_f64; self.nx]; self.ny];
        for (iy, data_row) in data.iter_mut().enumerate().take(self.ny) {
            for (ix, data_cell) in data_row.iter_mut().enumerate().take(self.nx) {
                *data_cell = self.data[iy][ix] + other.data[iy][ix];
            }
        }
        Ok(WavefrontMap {
            data,
            nx: self.nx,
            ny: self.ny,
            pupil_diameter_mm: self.pupil_diameter_mm,
            wavelength_nm: self.wavelength_nm,
        })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// HartmannShack
// ────────────────────────────────────────────────────────────────────────────

/// Hartmann-Shack wavefront sensor.
///
/// The sensor consists of a microlens array (lenslets) placed at the pupil plane.
/// Each lenslet focuses a sub-aperture of the incoming wavefront onto a detector.
/// The centroid displacement of each spot gives the local wavefront slope.
#[derive(Debug, Clone)]
pub struct HartmannShack {
    /// Lenslet pitch (centre-to-centre spacing) in mm.
    pub lenslet_pitch_mm: f64,
    /// Lenslet focal length in mm.
    pub lenslet_focal_length_mm: f64,
    /// Number of lenslets in x-direction.
    pub n_lenslets_x: usize,
    /// Number of lenslets in y-direction.
    pub n_lenslets_y: usize,
    /// Detector pixel size in µm.
    pub pixel_size_um: f64,
    /// Sensing wavelength in nm.
    pub wavelength_nm: f64,
    /// Pupil diameter in mm.
    pub pupil_diameter_mm: f64,
}

impl HartmannShack {
    /// Construct a new Hartmann-Shack sensor.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pitch_mm: f64,
        fl_mm: f64,
        nx: usize,
        ny: usize,
        pixel_um: f64,
        lambda_nm: f64,
        pupil_mm: f64,
    ) -> Self {
        Self {
            lenslet_pitch_mm: pitch_mm,
            lenslet_focal_length_mm: fl_mm,
            n_lenslets_x: nx,
            n_lenslets_y: ny,
            pixel_size_um: pixel_um,
            wavelength_nm: lambda_nm,
            pupil_diameter_mm: pupil_mm,
        }
    }

    /// Wavefront slope sensitivity in waves/mm (minimum detectable slope).
    ///
    /// Limited by sub-aperture diffraction: δ_slope ≈ λ / (d · f) · (pixel_size/2)
    /// where d = lenslet pitch, f = focal length.
    pub fn slope_sensitivity_waves_per_mm(&self) -> f64 {
        let lambda_mm = self.wavelength_nm * 1e-6;
        // Minimum detectable displacement ≈ 0.1 pixel (centroiding precision).
        let min_disp_mm = self.pixel_size_um * 1e-3 * 0.1;
        // slope = disp / fl → in rad/mm → convert to waves/mm
        let slope_rad_per_mm = min_disp_mm / self.lenslet_focal_length_mm;
        slope_rad_per_mm / (TWO_PI * lambda_mm / self.lenslet_pitch_mm)
    }

    /// Maximum measurable wavefront slope in waves/mm (dynamic range).
    ///
    /// Spot must stay within its sub-aperture: max_disp = pitch/2 → max_slope = pitch/(2·f)
    pub fn dynamic_range_waves_per_mm(&self) -> f64 {
        let lambda_mm = self.wavelength_nm * 1e-6;
        let max_disp_mm = self.lenslet_pitch_mm / 2.0;
        let slope_rad_per_mm = max_disp_mm / self.lenslet_focal_length_mm;
        slope_rad_per_mm / (TWO_PI * lambda_mm / self.lenslet_pitch_mm)
    }

    /// Number of independent wavefront modes reconstructable ≈ n_lenslets / 2.
    pub fn reconstructable_modes(&self) -> usize {
        let n_total = self.n_lenslets_x * self.n_lenslets_y;
        n_total / 2
    }

    /// Spot displacement in µm from local wavefront slopes (rad/mm).
    ///
    /// Δx = f_mm · (∂W/∂x)_rad_per_mm × 1000  \[µm\]
    pub fn spot_displacement_um(
        &self,
        slope_x_rad_per_mm: f64,
        slope_y_rad_per_mm: f64,
    ) -> (f64, f64) {
        let dx = self.lenslet_focal_length_mm * slope_x_rad_per_mm * 1000.0;
        let dy = self.lenslet_focal_length_mm * slope_y_rad_per_mm * 1000.0;
        (dx, dy)
    }

    /// Local wavefront slope in rad/mm from spot displacement in µm.
    pub fn slope_from_displacement_mm(&self, dx_um: f64, dy_um: f64) -> (f64, f64) {
        let dx_mm = dx_um * 1e-3;
        let dy_mm = dy_um * 1e-3;
        let sx = dx_mm / self.lenslet_focal_length_mm;
        let sy = dy_mm / self.lenslet_focal_length_mm;
        (sx, sy)
    }

    /// Reconstruct wavefront from measured lenslet slopes (Southwell geometry zonal).
    ///
    /// Uses iterative integration: W\[i+1,j\] = W\[i,j\] + sx\[i,j\]·Δx and
    /// symmetrically for y, then averages both estimates.
    pub fn reconstruct_wavefront(
        &self,
        slopes_x: &[Vec<f64>],
        slopes_y: &[Vec<f64>],
    ) -> WavefrontMap {
        let ny = slopes_x.len();
        let nx = if ny > 0 { slopes_x[0].len() } else { 0 };
        let dx = self.lenslet_pitch_mm;
        let dy = self.lenslet_pitch_mm;

        // Convert slopes (rad/mm) to waves/lenslet step: Δφ = slope × d / (2π × λ_mm)
        let lambda_mm = self.wavelength_nm * 1e-6;
        let scale_x = dx / (TWO_PI * lambda_mm);
        let scale_y = dy / (TWO_PI * lambda_mm);

        // Southwell zonal: integrate row-by-row then column-by-column, average.
        let mut w = vec![vec![0.0_f64; nx]; ny];

        // Integrate along x-rows.
        for iy in 0..ny {
            for ix in 1..nx {
                let sx = if iy < slopes_x.len() && ix - 1 < slopes_x[iy].len() {
                    slopes_x[iy][ix - 1]
                } else {
                    0.0
                };
                w[iy][ix] = w[iy][ix - 1] + sx * scale_x;
            }
        }

        // Correct columns by integrating along y.
        let mut wc = w.clone();
        for iy in 1..ny {
            for ix in 0..nx {
                let sy = if iy - 1 < slopes_y.len() && ix < slopes_y[iy - 1].len() {
                    slopes_y[iy - 1][ix]
                } else {
                    0.0
                };
                wc[iy][ix] = wc[iy - 1][ix] + sy * scale_y;
            }
        }

        // Average row-integration and column-integration results.
        let mut data = vec![vec![0.0_f64; nx]; ny];
        for iy in 0..ny {
            for ix in 0..nx {
                data[iy][ix] = (w[iy][ix] + wc[iy][ix]) * 0.5;
            }
        }

        let pupil_d = self.pupil_diameter_mm;
        let wl = self.wavelength_nm;

        WavefrontMap {
            data,
            nx,
            ny,
            pupil_diameter_mm: pupil_d,
            wavelength_nm: wl,
        }
    }

    /// Simulate sensor output (slopes in rad/mm) given a known wavefront.
    ///
    /// Computes finite-difference slopes from the wavefront, then adds
    /// Gaussian noise (σ = `noise_level_um / f_mm` converted to rad/mm).
    pub fn simulate_measurement(
        &self,
        wavefront: &WavefrontMap,
        noise_level_um: f64,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let nx = self.n_lenslets_x;
        let ny = self.n_lenslets_y;
        let lambda_mm = self.wavelength_nm * 1e-6;

        // Sample wavefront on lenslet grid by bilinear interpolation.
        let cx = (wavefront.nx as f64 - 1.0) * 0.5;
        let cy = (wavefront.ny as f64 - 1.0) * 0.5;
        let r_pix = (wavefront.nx.min(wavefront.ny) as f64 - 1.0) * 0.5;

        // Pixels per lenslet step.
        let px_per_lens_x = (wavefront.nx as f64 - 1.0) / (nx as f64 - 1.0).max(1.0);
        let px_per_lens_y = (wavefront.ny as f64 - 1.0) / (ny as f64 - 1.0).max(1.0);

        let noise_rad_per_mm = if self.lenslet_focal_length_mm > 0.0 {
            noise_level_um * 1e-3 / self.lenslet_focal_length_mm
        } else {
            0.0
        };

        // Use a simple deterministic noise (fixed-seed pseudo random for reproducibility).
        let mut sx = vec![vec![0.0_f64; nx]; ny];
        let mut sy = vec![vec![0.0_f64; nx]; ny];

        let sample_wf = |ix_f: f64, iy_f: f64| -> f64 {
            let ix0 = ix_f.floor() as usize;
            let iy0 = iy_f.floor() as usize;
            let ix0 = ix0.min(wavefront.nx.saturating_sub(2));
            let iy0 = iy0.min(wavefront.ny.saturating_sub(2));
            let fx = ix_f - ix0 as f64;
            let fy = iy_f - iy0 as f64;
            let v00 = wavefront.data[iy0][ix0];
            let v01 = wavefront.data[iy0][ix0 + 1];
            let v10 = wavefront.data[iy0 + 1][ix0];
            let v11 = wavefront.data[iy0 + 1][ix0 + 1];
            v00 * (1.0 - fx) * (1.0 - fy)
                + v01 * fx * (1.0 - fy)
                + v10 * (1.0 - fx) * fy
                + v11 * fx * fy
        };

        for iy in 0..ny {
            for ix in 0..nx {
                // Pixel coordinates of this lenslet centre.
                let px = cx + (ix as f64 - (nx as f64 - 1.0) * 0.5) * px_per_lens_x;
                let py = cy + (iy as f64 - (ny as f64 - 1.0) * 0.5) * px_per_lens_y;

                // Check inside pupil.
                let xn = (px - cx) / r_pix;
                let yn = (py - cy) / r_pix;
                if xn * xn + yn * yn > 1.0 {
                    continue;
                }

                // Finite difference gradient (in waves/pixel).
                let dp = px_per_lens_x.max(1.0);
                let px_l = (px - dp * 0.5).max(0.0);
                let px_r = (px + dp * 0.5).min(wavefront.nx as f64 - 1.0);
                let py_b = (py - dp * 0.5).max(0.0);
                let py_t = (py + dp * 0.5).min(wavefront.ny as f64 - 1.0);

                let dw_dx_waves_per_pix =
                    (sample_wf(px_r, py) - sample_wf(px_l, py)) / (px_r - px_l).max(1e-30);
                let dw_dy_waves_per_pix =
                    (sample_wf(px, py_t) - sample_wf(px, py_b)) / (py_t - py_b).max(1e-30);

                // Convert waves/pixel → rad/mm.
                let pix_per_mm = wavefront.nx as f64 / self.pupil_diameter_mm;
                let dw_dx_rad_mm = dw_dx_waves_per_pix * pix_per_mm * TWO_PI * lambda_mm;
                let dw_dy_rad_mm = dw_dy_waves_per_pix * pix_per_mm * TWO_PI * lambda_mm;

                // Deterministic noise (based on index, avoids rand dependency).
                let noise_factor = ((ix + iy * nx) as f64 * 1.618_f64).sin() * noise_rad_per_mm;

                sx[iy][ix] = dw_dx_rad_mm + noise_factor;
                sy[iy][ix] = dw_dy_rad_mm + noise_factor * 0.7;
            }
        }
        (sx, sy)
    }

    /// Zernike reconstruction matrix Z (n_lenslets_active × n_modes).
    ///
    /// Row l corresponds to lenslet l (x then y slopes interleaved),
    /// column j is Zernike mode j.
    pub fn zernike_reconstruction_matrix(&self, n_modes: usize) -> Vec<Vec<f64>> {
        let n_modes = n_modes.min(15);
        let nx = self.n_lenslets_x;
        let ny = self.n_lenslets_y;
        let n_lenslets = nx * ny;
        let n_rows = 2 * n_lenslets; // x and y slopes

        let cx = (nx as f64 - 1.0) * 0.5;
        let cy = (ny as f64 - 1.0) * 0.5;
        let r = (nx.min(ny) as f64 - 1.0) * 0.5;
        let dr = 0.5 / r.max(1.0); // differentiation step

        let mut mat = vec![vec![0.0_f64; n_modes]; n_rows];

        for iy in 0..ny {
            for ix in 0..nx {
                let xn = (ix as f64 - cx) / r.max(1.0);
                let yn = (iy as f64 - cy) / r.max(1.0);
                let rho = (xn * xn + yn * yn).sqrt();
                let theta = yn.atan2(xn);
                let l = iy * nx + ix;

                let (mat_x_part, mat_y_part) = mat.split_at_mut(l + n_lenslets);
                for (j, (mat_lj, mat_lyj)) in mat_x_part[l]
                    .iter_mut()
                    .zip(mat_y_part[0].iter_mut())
                    .enumerate()
                    .take(n_modes)
                {
                    // Numerical gradient of Z_j at (ρ,θ).
                    let rho_p = (rho + dr).min(1.0);
                    let rho_m = (rho - dr).max(0.0);
                    let z_pr = zernike_basis(j, rho_p, theta);
                    let z_mr = zernike_basis(j, rho_m, theta);
                    let dz_drho = (z_pr - z_mr) / (rho_p - rho_m).max(1e-30);

                    // ∂Z/∂x = (∂Z/∂ρ)·(∂ρ/∂x) = dz_drho · cos(θ) / R_pix
                    *mat_lj = dz_drho * theta.cos(); // x-slope row
                    *mat_lyj = dz_drho * theta.sin(); // y-slope row
                }
            }
        }
        mat
    }

    /// Compute Strehl ratio from measured slopes by reconstructing the wavefront.
    pub fn measured_strehl(&self, slopes_x: &[Vec<f64>], slopes_y: &[Vec<f64>]) -> f64 {
        let wf = self.reconstruct_wavefront(slopes_x, slopes_y);
        wf.strehl_marechal()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// PsfAnalysis
// ────────────────────────────────────────────────────────────────────────────

/// Point spread function analysis utilities.
pub struct PsfAnalysis;

impl PsfAnalysis {
    /// Airy disk intensity at normalised radius r_norm = r / (λ·f/D).
    ///
    /// I(r) = \[2·J₁(π·r_norm) / (π·r_norm)\]²
    ///
    /// Returns 1 at r_norm = 0 (diffraction-limited maximum).
    pub fn airy_disk(r_norm: f64) -> f64 {
        if r_norm.abs() < 1e-10 {
            return 1.0;
        }
        let x = PI * r_norm;
        let j1 = Self::bessel_j1(x);
        (2.0 * j1 / x).powi(2)
    }

    /// Radius of first Airy zero in µm: r = 1.22·λ·f/D = 1.22·λ/NA.
    ///
    /// `lambda_nm`: wavelength in nm, `numerical_aperture`: NA of the optic.
    pub fn airy_zero_radius_um(lambda_nm: f64, numerical_aperture: f64) -> f64 {
        1.22 * lambda_nm * 1e-3 / numerical_aperture
    }

    /// FWHM of Airy disk ≈ 1.028·λ/NA in µm.
    pub fn airy_fwhm_um(lambda_nm: f64, numerical_aperture: f64) -> f64 {
        1.028 * lambda_nm * 1e-3 / numerical_aperture
    }

    /// Bessel J₁(x) approximation via Maclaurin + asymptotic series.
    ///
    /// Uses the 6-term polynomial approximation from Abramowitz & Stegun §9.4.
    pub fn bessel_j1(x: f64) -> f64 {
        if x == 0.0 {
            return 0.0;
        }
        let ax = x.abs();
        let sign = if x < 0.0 { -1.0 } else { 1.0 };

        if ax < 8.0 {
            // Polynomial approximation (A&S 9.4.6).
            let y = x * x;
            let p1 = 72_362_614_232.0_f64;
            let p2 = -7_895_059_235.0_f64;
            let p3 = 242_396_853.1_f64;
            let p4 = -2_972_611.439_f64;
            let p5 = 15_704.482_60_f64;
            let p6 = -30.16322_f64;
            let q1 = 144_725_228_442.0_f64;
            let q2 = 2_300_535_178.0_f64;
            let q3 = 18_583_304.74_f64;
            let q4 = 99_447.433_94_f64;
            let q5 = 376.9991397_f64;
            let q6 = 1.0_f64;
            let num = x * (p1 + y * (p2 + y * (p3 + y * (p4 + y * (p5 + y * p6)))));
            let den = q1 + y * (q2 + y * (q3 + y * (q4 + y * (q5 + y * q6))));
            num / den
        } else {
            // Asymptotic expansion (A&S 9.2.9).
            let z = 8.0 / ax;
            let y = z * z;
            let xx = ax - 2.356_194_490_2; // ax - 3π/4
            let p = 1.0
                + y * (-0.183_105e-2
                    + y * (-0.3516396_496e-4 + y * (0.245_752_020_6e-5 + y * (-0.240_337_019e-6))));
            let q_poly = 0.04687499995
                + y * (-0.2002690873e-3
                    + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
            (2.0 / (PI * ax)).sqrt() * (p * xx.cos() - z * q_poly * xx.sin()) * sign
        }
    }

    /// PSF from a WavefrontMap (convenience wrapper).
    pub fn psf_from_wavefront(wavefront: &WavefrontMap, n_pts: usize) -> Vec<Vec<f64>> {
        wavefront.psf(n_pts)
    }

    /// 1D MTF from PSF (radial average, normalised to MTF(0) = 1).
    ///
    /// Returns `(spatial_freq_cycles_per_pixel, mtf)` pairs.
    pub fn mtf_from_psf(psf: &[Vec<f64>]) -> Vec<(f64, f64)> {
        let nrows = psf.len();
        if nrows == 0 {
            return Vec::new();
        }
        let ncols = psf[0].len();
        let n = nrows.min(ncols);

        // OTF via 2D FFT of PSF (use oxifft).
        let flat: Vec<oxifft::kernel::Complex<f64>> = psf
            .iter()
            .flat_map(|row| row.iter().map(|&v| oxifft::kernel::Complex::new(v, 0.0)))
            .collect();
        let spectrum = fft2d(&flat, nrows, ncols);

        // Radial MTF: average |OTF| at each frequency ring.
        let half_n = n / 2;
        let mut mtf_sum = vec![0.0_f64; half_n + 1];
        let mut mtf_cnt = vec![0_usize; half_n + 1];

        let dc_val = spectrum[0].re.abs().max(1e-30);

        for ky in 0..nrows {
            for kx in 0..ncols {
                let fky = if ky > nrows / 2 {
                    ky as f64 - nrows as f64
                } else {
                    ky as f64
                };
                let fkx = if kx > ncols / 2 {
                    kx as f64 - ncols as f64
                } else {
                    kx as f64
                };
                let r = (fky * fky + fkx * fkx).sqrt() as usize;
                if r <= half_n {
                    let v = spectrum[ky * ncols + kx];
                    mtf_sum[r] += (v.re * v.re + v.im * v.im).sqrt();
                    mtf_cnt[r] += 1;
                }
            }
        }

        (0..=half_n)
            .map(|r| {
                let freq = r as f64 / n as f64;
                let mtf = if mtf_cnt[r] > 0 {
                    (mtf_sum[r] / mtf_cnt[r] as f64) / dc_val
                } else {
                    0.0
                };
                (freq, mtf)
            })
            .collect()
    }

    /// Strehl ratio from PSF: S = PSF_peak / diffraction_limited_peak.
    pub fn strehl_from_psf(psf: &[Vec<f64>], diffraction_limited_peak: f64) -> f64 {
        if diffraction_limited_peak < 1e-30 {
            return 1.0;
        }
        let peak = psf
            .iter()
            .flat_map(|row| row.iter().cloned())
            .fold(0.0_f64, f64::max);
        (peak / diffraction_limited_peak).min(1.0)
    }

    /// Optical transfer function (complex): OTF = FT\[PSF\] normalised.
    pub fn otf_from_psf(psf: &[Vec<f64>]) -> Vec<Vec<Complex64>> {
        let nrows = psf.len();
        if nrows == 0 {
            return Vec::new();
        }
        let ncols = psf[0].len();
        let flat: Vec<oxifft::kernel::Complex<f64>> = psf
            .iter()
            .flat_map(|row| row.iter().map(|&v| oxifft::kernel::Complex::new(v, 0.0)))
            .collect();
        let spectrum = fft2d(&flat, nrows, ncols);
        let dc = spectrum[0].re.abs().max(1e-30);
        let mut otf = vec![vec![Complex64::new(0.0, 0.0); ncols]; nrows];
        for iy in 0..nrows {
            for ix in 0..ncols {
                let v = spectrum[iy * ncols + ix];
                otf[iy][ix] = Complex64::new(v.re / dc, v.im / dc);
            }
        }
        otf
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const NX: usize = 32;
    const NY: usize = 32;
    const PUPIL_MM: f64 = 10.0;
    const LAMBDA_NM: f64 = 633.0;

    fn flat_wf() -> WavefrontMap {
        WavefrontMap::new(NX, NY, PUPIL_MM, LAMBDA_NM)
    }

    fn defocus_wf(amp: f64) -> WavefrontMap {
        // j=3 = defocus
        WavefrontMap::from_zernike_coefficients(&[(3, amp)], NX, NY, PUPIL_MM, LAMBDA_NM)
    }

    #[test]
    fn test_flat_wavefront_zero_rms() {
        let wf = flat_wf();
        assert!(
            wf.rms_waves() < 1e-12,
            "Flat wavefront must have zero RMS, got {}",
            wf.rms_waves()
        );
        assert!(
            (wf.strehl_marechal() - 1.0).abs() < 1e-10,
            "Flat wavefront Strehl must be 1"
        );
    }

    #[test]
    fn test_strehl_marechal_formula() {
        let sigma = 0.05; // waves — within Maréchal criterion
        let wf =
            WavefrontMap::from_zernike_coefficients(&[(3, sigma)], NX, NY, PUPIL_MM, LAMBDA_NM);
        let s_computed = wf.strehl_marechal();
        let rms = wf.rms_waves();
        let s_expected = (-(TWO_PI * rms).powi(2)).exp();
        assert!(
            (s_computed - s_expected).abs() < 1e-10,
            "Strehl Maréchal mismatch: {} vs {}",
            s_computed,
            s_expected
        );
    }

    #[test]
    fn test_rms_increases_with_aberration() {
        let wf1 = defocus_wf(0.05);
        let wf2 = defocus_wf(0.15);
        assert!(
            wf2.rms_waves() > wf1.rms_waves(),
            "Larger defocus amplitude must give larger RMS"
        );
    }

    #[test]
    fn test_pv_greater_than_rms() {
        let wf = defocus_wf(0.2);
        let pv = wf.pv_waves();
        let rms = wf.rms_waves();
        // PV >= RMS always; for smooth aberrations PV >> RMS.
        assert!(pv >= rms, "PV ({}) must be >= RMS ({})", pv, rms);
    }

    #[test]
    fn test_pupil_function_unit_modulus() {
        let wf = defocus_wf(0.3);
        let pf = wf.pupil_function();
        let cx = (NX as f64 - 1.0) * 0.5;
        let cy = (NY as f64 - 1.0) * 0.5;
        let r_pix = (NX.min(NY) as f64 - 1.0) * 0.5;

        for (iy, pf_row) in pf.iter().enumerate().take(NY) {
            for (ix, pf_cell) in pf_row.iter().enumerate().take(NX) {
                let xn = (ix as f64 - cx) / r_pix;
                let yn = (iy as f64 - cy) / r_pix;
                if xn * xn + yn * yn <= 1.0 {
                    let modulus = pf_cell.norm();
                    assert!(
                        (modulus - 1.0).abs() < 1e-10,
                        "Pupil function modulus must be 1, got {}",
                        modulus
                    );
                }
            }
        }
    }

    #[test]
    fn test_hs_spot_displacement() {
        let hs = HartmannShack::new(0.5, 10.0, 8, 8, 5.6, 633.0, 10.0);
        let slope_x = 0.001; // rad/mm
        let slope_y = 0.002;
        let (dx, dy) = hs.spot_displacement_um(slope_x, slope_y);
        let expected_dx = 10.0 * slope_x * 1000.0; // f_mm × slope × 1000 µm/mm
        let expected_dy = 10.0 * slope_y * 1000.0;
        assert!(
            (dx - expected_dx).abs() < 1e-8,
            "dx mismatch: {} vs {}",
            dx,
            expected_dx
        );
        assert!(
            (dy - expected_dy).abs() < 1e-8,
            "dy mismatch: {} vs {}",
            dy,
            expected_dy
        );
    }

    #[test]
    fn test_hs_reconstructable_modes() {
        let hs = HartmannShack::new(0.5, 10.0, 8, 8, 5.6, 633.0, 10.0);
        let modes = hs.reconstructable_modes();
        assert_eq!(
            modes,
            8 * 8 / 2,
            "reconstructable_modes should be n_total/2"
        );
    }

    #[test]
    fn test_airy_disk_at_zero() {
        let val = PsfAnalysis::airy_disk(0.0);
        assert!(
            (val - 1.0).abs() < 1e-10,
            "Airy disk at zero must be 1, got {}",
            val
        );
    }

    #[test]
    fn test_airy_fwhm_formula() {
        // FWHM ≈ 1.028 λ/NA
        let lambda_nm = 633.0;
        let na = 0.5;
        let fwhm = PsfAnalysis::airy_fwhm_um(lambda_nm, na);
        let expected = 1.028 * 633.0 * 1e-3 / 0.5;
        assert!(
            (fwhm - expected).abs() < 1e-10,
            "FWHM mismatch: {} vs {}",
            fwhm,
            expected
        );
    }

    #[test]
    fn test_bessel_j1_zero() {
        let val = PsfAnalysis::bessel_j1(0.0);
        assert!(val.abs() < 1e-10, "J1(0) must be 0, got {}", val);
    }

    #[test]
    fn test_wavefront_add() {
        let wf1 = WavefrontMap::from_zernike_coefficients(&[(3, 0.1)], NX, NY, PUPIL_MM, LAMBDA_NM);
        let wf2 =
            WavefrontMap::from_zernike_coefficients(&[(3, 0.05)], NX, NY, PUPIL_MM, LAMBDA_NM);
        let sum = wf1.add(&wf2).expect("add should succeed");
        assert_eq!(sum.nx, NX);
        assert_eq!(sum.ny, NY);
        // Centre pixel check.
        let cy = NY / 2;
        let cx_idx = NX / 2;
        let expected = wf1.data[cy][cx_idx] + wf2.data[cy][cx_idx];
        assert!((sum.data[cy][cx_idx] - expected).abs() < 1e-12);
    }
}
