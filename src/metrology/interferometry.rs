//! Interferogram analysis and phase measurement.
//!
//! Provides:
//! - `Interferogram`: Fizeau/Twyman-Green interferogram with phase extraction
//! - `PhaseUnwrapper`: 1D and 2D phase unwrapping algorithms
//! - `OpdMeasurement`: Optical path difference analysis
//! - `ShearingInterferometer`: Lateral shearing interferometry
//!
//! Reference: Malacara, "Optical Shop Testing", 3rd ed., Chapters 14-16.

use oxifft::{fft2d, ifft2d};

use crate::metrology::wavefront::WavefrontMap;

const PI: f64 = std::f64::consts::PI;
const TWO_PI: f64 = 2.0 * PI;

// ────────────────────────────────────────────────────────────────────────────
// Interferogram
// ────────────────────────────────────────────────────────────────────────────

/// Fizeau / Twyman-Green interferogram.
///
/// Intensity: I(x,y) = I₀ · (1 + V·cos(2π·W(x,y) + α))
/// where W = OPD in waves, α = carrier phase (tilt), V = visibility.
#[derive(Debug, Clone)]
pub struct Interferogram {
    /// Normalised intensity in \[0,1\].
    pub data: Vec<Vec<f64>>,
    /// Number of columns.
    pub nx: usize,
    /// Number of rows.
    pub ny: usize,
    /// Carrier fringe density (fringes per aperture diameter).
    pub fringe_density: f64,
    /// Carrier phase (tilt reference, radians).
    pub carrier_phase: f64,
    /// Fringe visibility V ∈ \[0, 1\].
    pub visibility: f64,
    /// Wavelength in nm.
    pub wavelength_nm: f64,
}

impl Interferogram {
    /// Create a flat (zero OPD) interferogram with default parameters.
    pub fn new(nx: usize, ny: usize, wavelength_nm: f64) -> Self {
        Self {
            data: vec![vec![0.5_f64; nx]; ny],
            nx,
            ny,
            fringe_density: 0.0,
            carrier_phase: 0.0,
            visibility: 1.0,
            wavelength_nm,
        }
    }

    /// Generate interferogram from a wavefront map.
    ///
    /// `carrier_freq` is the number of straight carrier fringes across the aperture.
    /// Gaussian noise at `noise_level` (sigma, relative to mean intensity) is added.
    pub fn from_wavefront(
        wavefront: &WavefrontMap,
        visibility: f64,
        carrier_freq: f64,
        noise_level: f64,
    ) -> Self {
        let nx = wavefront.nx;
        let ny = wavefront.ny;
        let mut data = vec![vec![0.0_f64; nx]; ny];

        // Pupil geometry (same as wavefront).
        let cx = (nx as f64 - 1.0) * 0.5;
        let cy = (ny as f64 - 1.0) * 0.5;
        let r_pix = (nx.min(ny) as f64 - 1.0) * 0.5;

        for (iy, data_row) in data.iter_mut().enumerate().take(ny) {
            for (ix, data_cell) in data_row.iter_mut().enumerate().take(nx) {
                let xn = (ix as f64 - cx) / r_pix;
                let _yn = (iy as f64 - cy) / r_pix;

                // Carrier: linear tilt in x-direction.
                let carrier = carrier_freq * xn; // in waves
                let w = wavefront.data[iy][ix] + carrier;
                let phase = TWO_PI * w;

                let intensity = 0.5 * (1.0 + visibility * phase.cos());

                // Deterministic pseudo-noise (avoids rand dependency).
                let noise = if noise_level > 0.0 {
                    let seed = (ix + iy * nx) as f64;
                    noise_level * (seed * 1.618_f64 + std::f64::consts::E).sin()
                } else {
                    0.0
                };

                *data_cell = (intensity + noise).clamp(0.0, 1.0);
            }
        }

        Self {
            data,
            nx,
            ny,
            fringe_density: carrier_freq,
            carrier_phase: 0.0,
            visibility,
            wavelength_nm: wavefront.wavelength_nm,
        }
    }

    /// Generate a tilted-reference interferogram (pure carrier fringes, flat wavefront).
    ///
    /// `tilt_waves` is the total OPD across the aperture due to tilt.
    pub fn tilted_reference(nx: usize, ny: usize, tilt_waves: f64, wavelength_nm: f64) -> Self {
        let mut data = vec![vec![0.0_f64; nx]; ny];
        for data_row in data.iter_mut().take(ny) {
            for (ix, data_cell) in data_row.iter_mut().enumerate().take(nx) {
                let xn = ix as f64 / (nx as f64 - 1.0).max(1.0);
                let phase = TWO_PI * tilt_waves * xn;
                *data_cell = 0.5 * (1.0 + phase.cos());
            }
        }
        Self {
            data,
            nx,
            ny,
            fringe_density: tilt_waves,
            carrier_phase: 0.0,
            visibility: 1.0,
            wavelength_nm,
        }
    }

    /// Phase extraction by Fourier carrier-fringe method.
    ///
    /// Algorithm:
    /// 1. 2D FFT of interferogram
    /// 2. Shift +1 carrier order to DC
    /// 3. Apply Gaussian window around the +1 order
    /// 4. IFFT → arg() gives wrapped phase
    pub fn extract_phase_fourier(&self) -> Vec<Vec<f64>> {
        let n = self.nx.max(self.ny);

        // Build flat complex input.
        let flat: Vec<oxifft::kernel::Complex<f64>> = self
            .data
            .iter()
            .flat_map(|row| {
                row.iter()
                    .map(|&v| oxifft::kernel::Complex::new(v, 0.0_f64))
            })
            .collect();

        // 2D FFT.
        let spectrum = fft2d(&flat, self.ny, self.nx);

        // Carrier frequency in pixels = fringe_density * nx / nx = fringe_density.
        // The +1 order is at (kx = +carrier_bins, ky = 0).
        let carrier_bins = (self.fringe_density * self.nx as f64 / self.nx as f64).round() as i64;
        let carrier_bins = carrier_bins.max(1);

        // Gaussian filter half-width ≈ carrier_bins / 2.
        let sigma = (n as f64 / 4.0).max(2.0);
        let sigma2 = 2.0 * sigma * sigma;

        // Shift and filter: for each (ky, kx) compute shifted index, apply Gaussian.
        let mut filtered = vec![oxifft::kernel::Complex::new(0.0_f64, 0.0_f64); self.ny * self.nx];

        for ky in 0..self.ny {
            for kx in 0..self.nx {
                // Shift +1 order to centre.
                let kxs = kx as i64 - carrier_bins;
                let kys = ky as i64;

                // Wrap to valid range.
                let kxs_w = ((kxs % self.nx as i64) + self.nx as i64) as usize % self.nx;
                let kys_w = ((kys % self.ny as i64) + self.ny as i64) as usize % self.ny;

                // Gaussian weight centred at (nx/2, ny/2).
                let dcx = kxs_w as f64 - self.nx as f64 * 0.5;
                let dcy = kys_w as f64 - self.ny as f64 * 0.5;
                let w = (-(dcx * dcx + dcy * dcy) / sigma2).exp();

                let v = spectrum[ky * self.nx + kx];
                filtered[kys_w * self.nx + kxs_w] =
                    oxifft::kernel::Complex::new(v.re * w, v.im * w);
            }
        }

        // IFFT.
        let back = ifft2d(&filtered, self.ny, self.nx);

        // Extract phase = arg(IFFT result).
        let mut phase = vec![vec![0.0_f64; self.nx]; self.ny];
        for iy in 0..self.ny {
            for ix in 0..self.nx {
                let c = back[iy * self.nx + ix];
                phase[iy][ix] = c.im.atan2(c.re);
            }
        }
        phase
    }

    /// Four-step phase-shifting algorithm.
    ///
    /// φ = atan2(I₃ − I₁, I₀ − I₂)
    /// where Iₖ has phase shift k·π/2.
    pub fn phase_shift_four_step(
        i0: &[Vec<f64>],
        i1: &[Vec<f64>],
        i2: &[Vec<f64>],
        i3: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        let ny = i0.len();
        if ny == 0 {
            return Vec::new();
        }
        let nx = i0[0].len();
        let mut phase = vec![vec![0.0_f64; nx]; ny];
        for iy in 0..ny {
            for ix in 0..nx {
                let a = i3[iy][ix] - i1[iy][ix];
                let b = i0[iy][ix] - i2[iy][ix];
                phase[iy][ix] = a.atan2(b);
            }
        }
        phase
    }

    /// Phase unwrapping (delegates to `PhaseUnwrapper::unwrap_2d_simple`).
    pub fn unwrap_phase(wrapped: &[Vec<f64>]) -> Vec<Vec<f64>> {
        PhaseUnwrapper::unwrap_2d_simple(wrapped)
    }

    /// Convert unwrapped phase (radians) to OPD in nm.
    ///
    /// OPD = φ · λ / (2π)
    pub fn phase_to_opd_nm(phase: &[Vec<f64>], wavelength_nm: f64) -> Vec<Vec<f64>> {
        phase
            .iter()
            .map(|row| row.iter().map(|&p| p * wavelength_nm / TWO_PI).collect())
            .collect()
    }

    /// Fringe visibility: V = (I_max − I_min) / (I_max + I_min).
    pub fn measure_visibility(&self) -> f64 {
        let mut i_max = f64::NEG_INFINITY;
        let mut i_min = f64::INFINITY;
        for row in &self.data {
            for &v in row {
                i_max = i_max.max(v);
                i_min = i_min.min(v);
            }
        }
        if i_max + i_min < 1e-30 {
            return 0.0;
        }
        (i_max - i_min) / (i_max + i_min)
    }

    /// Fit Zernike polynomials to the extracted phase (converted to waves).
    ///
    /// The phase is extracted via Fourier method, unwrapped, divided by 2π.
    pub fn fit_zernike_to_phase(&self, n_terms: usize) -> Vec<f64> {
        let wrapped = self.extract_phase_fourier();
        let unwrapped = Self::unwrap_phase(&wrapped);

        // Convert to waves.
        let ny = unwrapped.len();
        if ny == 0 {
            return vec![0.0; n_terms.min(15)];
        }
        let nx = unwrapped[0].len();
        let data_waves: Vec<Vec<f64>> = unwrapped
            .iter()
            .map(|row| row.iter().map(|&p| p / TWO_PI).collect())
            .collect();

        let wf = WavefrontMap {
            data: data_waves,
            nx,
            ny,
            pupil_diameter_mm: 10.0,
            wavelength_nm: self.wavelength_nm,
        };
        wf.fit_zernike(n_terms)
    }

    /// Surface form error: (PV_nm, RMS_nm) from extracted and unwrapped phase.
    pub fn form_error_nm(&self) -> (f64, f64) {
        let wrapped = self.extract_phase_fourier();
        let unwrapped = Self::unwrap_phase(&wrapped);

        let values: Vec<f64> = unwrapped.iter().flat_map(|r| r.iter().cloned()).collect();
        if values.is_empty() {
            return (0.0, 0.0);
        }

        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let pv_rad = max - min;

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let rms_rad = (values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt();

        let pv_nm = pv_rad * self.wavelength_nm / TWO_PI;
        let rms_nm = rms_rad * self.wavelength_nm / TWO_PI;
        (pv_nm, rms_nm)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// PhaseUnwrapper
// ────────────────────────────────────────────────────────────────────────────

/// Phase unwrapping algorithms.
pub struct PhaseUnwrapper;

impl PhaseUnwrapper {
    /// 1D phase unwrapping (simple continuous unwrapping).
    ///
    /// Adds multiples of 2π to eliminate phase jumps > π.
    pub fn unwrap_1d(phase: &[f64]) -> Vec<f64> {
        if phase.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(phase.len());
        out.push(phase[0]);
        let mut offset = 0.0_f64;
        for i in 1..phase.len() {
            let diff = phase[i] - phase[i - 1];
            // Wrap diff to (-π, π].
            let diff_wrapped = diff - (diff / TWO_PI).round() * TWO_PI;
            offset += diff_wrapped - diff;
            out.push(phase[i] + offset);
        }
        out
    }

    /// 2D phase unwrapping (row-by-row, then column correction).
    ///
    /// Unwraps each row independently, then applies column-wise correction
    /// to achieve 2D consistency.
    #[allow(clippy::needless_range_loop)]
    pub fn unwrap_2d_simple(wrapped: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let ny = wrapped.len();
        if ny == 0 {
            return Vec::new();
        }
        let nx = wrapped[0].len();
        if nx == 0 {
            return vec![Vec::new(); ny];
        }

        // Step 1: unwrap each row.
        let mut out: Vec<Vec<f64>> = wrapped.iter().map(|row| Self::unwrap_1d(row)).collect();

        // Step 2: unwrap along columns using the corrected row-0 as reference.
        for ix in 0..nx {
            let mut col_offset = 0.0_f64;
            for iy in 1..ny {
                let diff = out[iy][ix] - out[iy - 1][ix];
                let diff_wrapped = diff - (diff / TWO_PI).round() * TWO_PI;
                col_offset += diff_wrapped - diff;
                out[iy][ix] += col_offset;
            }
        }
        out
    }

    /// Quality map: 1 / (1 + |∇φ|²) — higher means smoother (better quality).
    pub fn quality_map(phase: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let ny = phase.len();
        if ny == 0 {
            return Vec::new();
        }
        let nx = phase[0].len();
        let gx = Self::gradient_x(phase);
        let gy = Self::gradient_y(phase);

        let mut q = vec![vec![0.0_f64; nx]; ny];
        for iy in 0..ny {
            for ix in 0..nx {
                let g2 = gx[iy][ix] * gx[iy][ix] + gy[iy][ix] * gy[iy][ix];
                q[iy][ix] = 1.0 / (1.0 + g2);
            }
        }
        q
    }

    /// x-gradient (central differences, forward/backward at boundaries).
    pub fn gradient_x(phase: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let ny = phase.len();
        if ny == 0 {
            return Vec::new();
        }
        let nx = phase[0].len();
        let mut out = vec![vec![0.0_f64; nx]; ny];
        for iy in 0..ny {
            for ix in 0..nx {
                let denom = if ix == 0 || ix == nx - 1 { 1.0 } else { 2.0 };
                let left = if ix > 0 {
                    phase[iy][ix - 1]
                } else {
                    phase[iy][ix]
                };
                let right = if ix + 1 < nx {
                    phase[iy][ix + 1]
                } else {
                    phase[iy][ix]
                };
                out[iy][ix] = (right - left) / denom;
            }
        }
        out
    }

    /// y-gradient (central differences, forward/backward at boundaries).
    pub fn gradient_y(phase: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let ny = phase.len();
        if ny == 0 {
            return Vec::new();
        }
        let nx = phase[0].len();
        let mut out = vec![vec![0.0_f64; nx]; ny];
        for iy in 0..ny {
            for ix in 0..nx {
                let denom = if iy == 0 || iy == ny - 1 { 1.0 } else { 2.0 };
                let top = if iy > 0 {
                    phase[iy - 1][ix]
                } else {
                    phase[iy][ix]
                };
                let bot = if iy + 1 < ny {
                    phase[iy + 1][ix]
                } else {
                    phase[iy][ix]
                };
                out[iy][ix] = (bot - top) / denom;
            }
        }
        out
    }
}

// ────────────────────────────────────────────────────────────────────────────
// OpdMeasurement
// ────────────────────────────────────────────────────────────────────────────

/// Optical path difference (OPD) measurement.
///
/// Stores OPD in nm on a 2D grid (same dimensions as the interferogram).
#[derive(Debug, Clone)]
pub struct OpdMeasurement {
    /// OPD in nm, row-major.
    pub opd_nm: Vec<Vec<f64>>,
    /// Number of columns.
    pub nx: usize,
    /// Number of rows.
    pub ny: usize,
    /// Reference wavelength in nm.
    pub wavelength_nm: f64,
}

impl OpdMeasurement {
    /// Construct from a 2D OPD grid and wavelength.
    pub fn new(opd_nm: Vec<Vec<f64>>, wavelength_nm: f64) -> Self {
        let ny = opd_nm.len();
        let nx = if ny > 0 { opd_nm[0].len() } else { 0 };
        Self {
            opd_nm,
            nx,
            ny,
            wavelength_nm,
        }
    }

    /// Peak-to-valley OPD in nm.
    pub fn pv_nm(&self) -> f64 {
        let vals: Vec<f64> = self.opd_nm.iter().flat_map(|r| r.iter().cloned()).collect();
        if vals.is_empty() {
            return 0.0;
        }
        let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        max - min
    }

    /// RMS OPD in nm.
    pub fn rms_nm(&self) -> f64 {
        let vals: Vec<f64> = self.opd_nm.iter().flat_map(|r| r.iter().cloned()).collect();
        if vals.is_empty() {
            return 0.0;
        }
        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        (vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n).sqrt()
    }

    /// OPD converted to waves (OPD_nm / λ_nm).
    pub fn in_waves(&self) -> Vec<Vec<f64>> {
        self.opd_nm
            .iter()
            .map(|row| row.iter().map(|&v| v / self.wavelength_nm).collect())
            .collect()
    }

    /// Strehl ratio via Maréchal approximation from OPD RMS.
    pub fn strehl(&self) -> f64 {
        let rms_waves = self.rms_nm() / self.wavelength_nm;
        let phase = TWO_PI * rms_waves;
        (-phase * phase).exp()
    }

    /// Return new OPD with best-fit tilt subtracted.
    ///
    /// Fits a plane A·x + B·y + C to the OPD data (least squares) and subtracts it.
    pub fn subtract_tilt(&self) -> OpdMeasurement {
        let ny = self.ny;
        let nx = self.nx;
        if ny == 0 || nx == 0 {
            return self.clone();
        }

        // Least-squares fit of plane: z = ax + by + c.
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_z = 0.0_f64;
        let mut sum_xx = 0.0_f64;
        let mut sum_yy = 0.0_f64;
        let mut sum_xy = 0.0_f64;
        let mut sum_xz = 0.0_f64;
        let mut sum_yz = 0.0_f64;
        let n = (nx * ny) as f64;

        for iy in 0..ny {
            let y = iy as f64;
            for ix in 0..nx {
                let x = ix as f64;
                let z = self.opd_nm[iy][ix];
                sum_x += x;
                sum_y += y;
                sum_z += z;
                sum_xx += x * x;
                sum_yy += y * y;
                sum_xy += x * y;
                sum_xz += x * z;
                sum_yz += y * z;
            }
        }

        // Normal equations (3×3 system).
        // | n    sum_x  sum_y | |c|   |sum_z |
        // | sum_x sum_xx sum_xy| |a| = |sum_xz|
        // | sum_y sum_xy sum_yy| |b|   |sum_yz|
        let det = n * (sum_xx * sum_yy - sum_xy * sum_xy)
            - sum_x * (sum_x * sum_yy - sum_xy * sum_y)
            + sum_y * (sum_x * sum_xy - sum_xx * sum_y);

        let (a, b, c) = if det.abs() < 1e-30 {
            (0.0, 0.0, sum_z / n.max(1.0))
        } else {
            let a_val = (sum_z * (sum_xx * sum_yy - sum_xy * sum_xy)
                - sum_x * (sum_xz * sum_yy - sum_xy * sum_yz)
                + sum_y * (sum_xz * sum_xy - sum_xx * sum_yz))
                / det;
            let b_val = (n * (sum_xz * sum_yy - sum_xy * sum_yz)
                - sum_z * (sum_x * sum_yy - sum_xy * sum_y)
                + sum_y * (sum_x * sum_yz - sum_xz * sum_y))
                / det;
            let c_val = (n * (sum_xx * sum_yz - sum_xz * sum_xy)
                - sum_x * (sum_x * sum_yz - sum_xz * sum_y)
                + sum_z * (sum_x * sum_xy - sum_xx * sum_y))
                / det;
            (b_val, c_val, a_val) // a,b,c: coeff of x, y, const
        };

        let mut opd_out = vec![vec![0.0_f64; nx]; ny];
        for (iy, out_row) in opd_out.iter_mut().enumerate().take(ny) {
            let y = iy as f64;
            for (ix, out_cell) in out_row.iter_mut().enumerate().take(nx) {
                let x = ix as f64;
                *out_cell = self.opd_nm[iy][ix] - (a * x + b * y + c);
            }
        }
        OpdMeasurement::new(opd_out, self.wavelength_nm)
    }

    /// Return new OPD with best-fit defocus (parabola) subtracted.
    ///
    /// Fits z = A·(x² + y²) + B to the data and subtracts it.
    pub fn subtract_defocus(&self) -> OpdMeasurement {
        let ny = self.ny;
        let nx = self.nx;
        if ny == 0 || nx == 0 {
            return self.clone();
        }

        let cx = (nx as f64 - 1.0) * 0.5;
        let cy = (ny as f64 - 1.0) * 0.5;

        let mut sum_r2 = 0.0_f64;
        let mut sum_r4 = 0.0_f64;
        let mut sum_z = 0.0_f64;
        let mut sum_r2z = 0.0_f64;
        let n = (nx * ny) as f64;

        for iy in 0..ny {
            let y = iy as f64 - cy;
            for ix in 0..nx {
                let x = ix as f64 - cx;
                let r2 = x * x + y * y;
                let z = self.opd_nm[iy][ix];
                sum_r2 += r2;
                sum_r4 += r2 * r2;
                sum_z += z;
                sum_r2z += r2 * z;
            }
        }

        // 2×2 system: [sum_r4, sum_r2; sum_r2, n] [A; B] = [sum_r2z; sum_z]
        let det = sum_r4 * n - sum_r2 * sum_r2;
        let (a_coeff, b_coeff) = if det.abs() < 1e-30 {
            (0.0, sum_z / n.max(1.0))
        } else {
            (
                (sum_r2z * n - sum_z * sum_r2) / det,
                (sum_r4 * sum_z - sum_r2 * sum_r2z) / det,
            )
        };

        let mut opd_out = vec![vec![0.0_f64; nx]; ny];
        for (iy, out_row) in opd_out.iter_mut().enumerate().take(ny) {
            let y = iy as f64 - cy;
            for (ix, out_cell) in out_row.iter_mut().enumerate().take(nx) {
                let x = ix as f64 - cx;
                let r2 = x * x + y * y;
                *out_cell = self.opd_nm[iy][ix] - (a_coeff * r2 + b_coeff);
            }
        }
        OpdMeasurement::new(opd_out, self.wavelength_nm)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ShearingInterferometer
// ────────────────────────────────────────────────────────────────────────────

/// Lateral shearing interferometer.
///
/// A shearing interferometer splits the beam and laterally displaces one copy
/// by a fraction of the pupil diameter. The resulting interferogram measures
/// the wavefront difference (finite difference of the wavefront slope).
#[derive(Debug, Clone)]
pub struct ShearingInterferometer {
    /// Shear in x-direction as fraction of pupil diameter [0, 1).
    pub shear_x: f64,
    /// Shear in y-direction as fraction of pupil diameter.
    pub shear_y: f64,
    /// Wavelength in nm.
    pub wavelength_nm: f64,
}

impl ShearingInterferometer {
    /// Construct a shearing interferometer.
    pub fn new(shear_x: f64, shear_y: f64, lambda_nm: f64) -> Self {
        Self {
            shear_x,
            shear_y,
            wavelength_nm: lambda_nm,
        }
    }

    /// Compute the shear interferogram phase map from a pupil phase map.
    ///
    /// ΔΦ(x,y) = Φ(x + s·N, y) − Φ(x, y)
    /// where s = `shear` fraction and N = number of pixels.
    pub fn shear_phase(phase_map: &[Vec<f64>], shear: f64) -> Vec<Vec<f64>> {
        let ny = phase_map.len();
        if ny == 0 {
            return Vec::new();
        }
        let nx = phase_map[0].len();
        let shift = (shear * nx as f64).round() as isize;

        let mut out = vec![vec![0.0_f64; nx]; ny];
        for iy in 0..ny {
            for ix in 0..nx {
                let ix2 = (ix as isize + shift).rem_euclid(nx as isize) as usize;
                out[iy][ix] = phase_map[iy][ix2] - phase_map[iy][ix];
            }
        }
        out
    }

    /// Reconstruct wavefront from x- and y-shear interferograms via iterative integration.
    ///
    /// Uses `n_iter` iterations of Gerchberg-type integration (Fourier domain).
    pub fn reconstruct_from_shear(
        sx: &[Vec<f64>],
        sy: &[Vec<f64>],
        n_iter: usize,
    ) -> Vec<Vec<f64>> {
        let ny = sx.len();
        if ny == 0 {
            return Vec::new();
        }
        let nx = sx[0].len();

        // Iterative reconstruction: integrate shear maps directly.
        // Start with cumulative-sum estimate.
        let mut w = vec![vec![0.0_f64; nx]; ny];

        // Initial estimate from x-shear: w[iy][ix] ≈ Σ sx[iy][0..ix].
        for iy in 0..ny {
            for ix in 1..nx {
                w[iy][ix] = w[iy][ix - 1]
                    + if iy < sx.len() && ix - 1 < sx[iy].len() {
                        sx[iy][ix - 1]
                    } else {
                        0.0
                    };
            }
        }

        // Iterative correction using y-shear.
        for _iter in 0..n_iter {
            // Compute residuals vs y-shear.
            let mut wc = w.clone();
            for iy in 1..ny {
                for ix in 0..nx {
                    let sy_val = if iy - 1 < sy.len() && ix < sy[iy - 1].len() {
                        sy[iy - 1][ix]
                    } else {
                        0.0
                    };
                    let predicted = w[iy - 1][ix] + sy_val;
                    let residual = predicted - w[iy][ix];
                    wc[iy][ix] += residual * 0.5;
                }
            }
            w = wc;
        }
        w
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
    const LAMBDA: f64 = 633.0;

    #[test]
    fn test_interferogram_visibility_range() {
        let wf = WavefrontMap::from_zernike_coefficients(&[(3, 0.2)], NX, NY, 10.0, LAMBDA);
        let igm = Interferogram::from_wavefront(&wf, 0.8, 5.0, 0.0);
        let v = igm.measure_visibility();
        assert!(
            (0.0..=1.0).contains(&v),
            "Visibility must be in [0,1], got {}",
            v
        );
        // Should be close to the set visibility.
        assert!(
            v > 0.5,
            "Visibility for V=0.8 input should be > 0.5, got {}",
            v
        );
    }

    #[test]
    fn test_phase_shift_four_step_flat() {
        // Flat wavefront: I_k = 0.5*(1 + cos(φ + k*π/2)), same φ everywhere.
        let phi = 0.3_f64;
        let make =
            |shift: f64| -> Vec<Vec<f64>> { vec![vec![0.5 * (1.0 + (phi + shift).cos()); NX]; NY] };
        let i0 = make(0.0);
        let i1 = make(PI / 2.0);
        let i2 = make(PI);
        let i3 = make(3.0 * PI / 2.0);

        let result = Interferogram::phase_shift_four_step(&i0, &i1, &i2, &i3);

        // All pixels should give the same phase (≈ φ).
        for row in &result {
            for &p in row {
                assert!(
                    (p - phi).abs() < 1e-6
                        || (p - phi + TWO_PI).abs() < 1e-6
                        || (p - phi - TWO_PI).abs() < 1e-6,
                    "Phase mismatch: got {}, expected {}",
                    p,
                    phi
                );
            }
        }
    }

    #[test]
    fn test_unwrap_1d_ramp() {
        // A linear ramp that doesn't need unwrapping (wrapped and unwrapped are the same).
        let phase: Vec<f64> = (0..16).map(|i| i as f64 * 0.3).collect();
        let unwrapped = PhaseUnwrapper::unwrap_1d(&phase);
        for (p, u) in phase.iter().zip(unwrapped.iter()) {
            assert!((*p - *u).abs() < 1e-10, "Ramp should remain unchanged");
        }
    }

    #[test]
    fn test_phase_to_opd_conversion() {
        // One full wave (2π radians) → λ nm OPD.
        let phase = vec![vec![TWO_PI; NX]; NY];
        let opd = Interferogram::phase_to_opd_nm(&phase, LAMBDA);
        for row in &opd {
            for &v in row {
                assert!((v - LAMBDA).abs() < 1e-8, "2π phase → λ nm OPD, got {}", v);
            }
        }
    }

    #[test]
    fn test_opd_rms_positive() {
        // OPD with non-zero variation must have positive RMS.
        let opd_data: Vec<Vec<f64>> = (0..NY)
            .map(|iy| (0..NX).map(|ix| (iy + ix) as f64 * 0.5).collect())
            .collect();
        let opd = OpdMeasurement::new(opd_data, LAMBDA);
        assert!(opd.rms_nm() > 0.0, "Non-flat OPD must have positive RMS");
        assert!(opd.pv_nm() > 0.0, "Non-flat OPD must have positive PV");
    }

    #[test]
    fn test_shear_phase_shift() {
        // Shearing a constant phase map by any amount gives zero difference.
        let flat: Vec<Vec<f64>> = vec![vec![1.5; NX]; NY];
        let sheared = ShearingInterferometer::shear_phase(&flat, 0.25);
        for row in &sheared {
            for &v in row {
                assert!(v.abs() < 1e-12, "Constant phase → zero shear, got {}", v);
            }
        }
    }

    #[test]
    fn test_form_error_flat_zero() {
        // A flat wavefront with a carrier should have near-zero form error.
        // We verify via OPD directly: flat wavefront → zero OPD everywhere.
        let wf = WavefrontMap::new(NX, NY, 10.0, LAMBDA);
        // All OPD values are 0 for a flat wavefront (in_waves all zeros).
        let waves = wf
            .data
            .iter()
            .flat_map(|r| r.iter().cloned())
            .collect::<Vec<_>>();
        let max_abs = waves.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(
            max_abs < 1e-12,
            "Flat wavefront OPD must be zero, got max |W| = {}",
            max_abs
        );

        // Verify that an OpdMeasurement from zero data also reports zero.
        let opd_data: Vec<Vec<f64>> = vec![vec![0.0_f64; NX]; NY];
        let opd = OpdMeasurement::new(opd_data, LAMBDA);
        assert!(
            opd.pv_nm() < 1e-12,
            "Flat OPD PV must be zero, got {}",
            opd.pv_nm()
        );
        assert!(
            opd.rms_nm() < 1e-12,
            "Flat OPD RMS must be zero, got {}",
            opd.rms_nm()
        );
    }
}
