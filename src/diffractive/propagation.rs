//! Scalar diffraction propagation: angular spectrum, Fresnel, Fraunhofer,
//! diffractive lenses (Fresnel zone plates, kinoforms), and SLM hologram synthesis.
//!
//! Physical conventions:
//! - Field coordinates in μm
//! - Wavelength in nm (converted internally)
//! - Phase in radians
//! - Complex field amplitude: U(x,y) ∈ ℂ
//!
//! References:
//! - Goodman, "Introduction to Fourier Optics", 3rd ed. (2005)
//! - Saldin et al., "Optics of Diffractive Elements" (Springer, 2017)

use num_complex::Complex64;
use std::f64::consts::PI;

#[allow(dead_code)]
const C0: f64 = 2.99792458e8;

// ---------------------------------------------------------------------------
// 2D FFT helpers (radix-2 Cooley-Tukey, same pattern as fiber/propagation.rs)
// ---------------------------------------------------------------------------

/// Radix-2 Cooley-Tukey FFT (in-place, row vector).
fn fft_1d(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    if n <= 1 {
        return x.to_vec();
    }
    // Bit-reversal permutation + butterfly (recursive)
    let half = n / 2;
    let even: Vec<Complex64> = (0..half).map(|k| x[2 * k]).collect();
    let odd: Vec<Complex64> = (0..half).map(|k| x[2 * k + 1]).collect();
    let fe = fft_1d(&even);
    let fo = fft_1d(&odd);
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    for k in 0..half {
        let angle = -2.0 * PI * k as f64 / n as f64;
        let twiddle = Complex64::new(angle.cos(), angle.sin());
        out[k] = fe[k] + twiddle * fo[k];
        out[k + half] = fe[k] - twiddle * fo[k];
    }
    out
}

/// Inverse FFT via conjugate trick.
fn ifft_1d(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    let conj_x: Vec<Complex64> = x.iter().map(|v| v.conj()).collect();
    let fft_conj = fft_1d(&conj_x);
    fft_conj.iter().map(|v| v.conj() / n as f64).collect()
}

/// 2D FFT: row-wise then column-wise 1D FFTs.
///
/// Input: `field[row][col]`, rows = Ny, cols = Nx.
fn fft_2d(field: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let ny = field.len();
    if ny == 0 {
        return Vec::new();
    }
    let nx = field[0].len();

    // Row-wise FFT
    let mut row_fft: Vec<Vec<Complex64>> = field.iter().map(|row| fft_1d(row)).collect();

    // Column-wise FFT
    for (col, _) in (0..nx).zip(std::iter::repeat(())) {
        let col_data: Vec<Complex64> = (0..ny).map(|r| row_fft[r][col]).collect();
        let col_out = fft_1d(&col_data);
        for r in 0..ny {
            row_fft[r][col] = col_out[r];
        }
    }
    row_fft
}

/// 2D inverse FFT.
fn ifft_2d(field: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let ny = field.len();
    if ny == 0 {
        return Vec::new();
    }
    let nx = field[0].len();

    // Row-wise IFFT
    let mut row_ifft: Vec<Vec<Complex64>> = field.iter().map(|row| ifft_1d(row)).collect();

    // Column-wise IFFT
    for (col, _) in (0..nx).zip(std::iter::repeat(())) {
        let col_data: Vec<Complex64> = (0..ny).map(|r| row_ifft[r][col]).collect();
        let col_out = ifft_1d(&col_data);
        for r in 0..ny {
            row_ifft[r][col] = col_out[r];
        }
    }
    row_ifft
}

/// FFT-shift: moves zero-frequency component to center of spectrum.
fn fftshift_2d(field: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let ny = field.len();
    if ny == 0 {
        return Vec::new();
    }
    let nx = field[0].len();
    let shift_y = ny / 2;
    let shift_x = nx / 2;
    let mut out = field.to_vec();
    for r in 0..ny {
        for c in 0..nx {
            out[(r + shift_y) % ny][(c + shift_x) % nx] = field[r][c];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// ScalarDiffraction
// ---------------------------------------------------------------------------

/// Scalar diffraction propagation methods.
///
/// All methods operate on 2D complex field arrays `field[row][col]`.
pub struct ScalarDiffraction;

impl ScalarDiffraction {
    /// Angular spectrum propagation — exact scalar diffraction.
    ///
    /// Algorithm:
    /// 1. Compute 2D FFT of input field: Ã(fx, fy)
    /// 2. Multiply by transfer function H(fx,fy) = exp(i·kz·z)
    ///    where kz = sqrt(k² - (2π·fx)² - (2π·fy)²)
    ///    (evanescent components: kz imaginary → set to 0)
    /// 3. Inverse 2D FFT
    ///
    /// # Arguments
    /// - `field` — input complex field `U(x,y,0)` \[row\]\[col\]
    /// - `dx_um` — pixel pitch in x (μm)
    /// - `dy_um` — pixel pitch in y (μm)
    /// - `z_um`  — propagation distance (μm)
    /// - `lambda_nm` — wavelength (nm)
    pub fn angular_spectrum(
        field: &[Vec<Complex64>],
        dx_um: f64,
        dy_um: f64,
        z_um: f64,
        lambda_nm: f64,
    ) -> Vec<Vec<Complex64>> {
        let ny = field.len();
        if ny == 0 {
            return Vec::new();
        }
        let nx = field[0].len();
        let lambda_um = lambda_nm * 1e-3;
        let k = 2.0 * PI / lambda_um; // wave number (1/μm)

        // 2D FFT of input field
        let spectrum = fft_2d(field);

        // Frequency axes (1/μm): centered around DC after fftshift convention
        // The FFT output has DC at [0][0]; we apply fftshift in the transfer function
        let spectrum_shifted = fftshift_2d(&spectrum);

        // Apply angular spectrum transfer function
        let output_spectrum: Vec<Vec<Complex64>> = spectrum_shifted
            .iter()
            .enumerate()
            .map(|(r, row)| {
                // fy centered: row index → shifted freq
                let fy_idx = r as f64 - ny as f64 / 2.0;
                let fy = fy_idx / (ny as f64 * dy_um);
                row.iter()
                    .enumerate()
                    .map(|(c, &val)| {
                        let fx_idx = c as f64 - nx as f64 / 2.0;
                        let fx = fx_idx / (nx as f64 * dx_um);
                        let kxy_sq = (2.0 * PI * fx).powi(2) + (2.0 * PI * fy).powi(2);
                        let k_sq = k * k;
                        if kxy_sq > k_sq {
                            // Evanescent: suppress
                            Complex64::new(0.0, 0.0)
                        } else {
                            let kz = (k_sq - kxy_sq).sqrt();
                            val * Complex64::new(0.0, kz * z_um).exp()
                        }
                    })
                    .collect()
            })
            .collect();

        // Shift back and IFFT
        let output_unshifted = fftshift_2d(&output_spectrum);
        ifft_2d(&output_unshifted)
    }

    /// Fresnel (paraxial) propagation via convolution with Fresnel kernel.
    ///
    /// Transfer function in frequency domain:
    ///   H(fx,fy) = exp(ikz) · exp(-iπλz(fx²+fy²))
    ///
    /// This is the far-field Fresnel approximation for kz ≈ k.
    pub fn fresnel(
        field: &[Vec<Complex64>],
        dx_um: f64,
        z_um: f64,
        lambda_nm: f64,
    ) -> Vec<Vec<Complex64>> {
        let ny = field.len();
        if ny == 0 {
            return Vec::new();
        }
        let nx = field[0].len();
        let lambda_um = lambda_nm * 1e-3;
        let k = 2.0 * PI / lambda_um;

        // 2D FFT
        let spectrum = fft_2d(field);
        let spectrum_shifted = fftshift_2d(&spectrum);

        // Apply Fresnel transfer function H = exp(ikz)·exp(-iπλz(fx²+fy²))
        let propagation_phase = Complex64::new(0.0, k * z_um).exp(); // global phase

        let output_spectrum: Vec<Vec<Complex64>> = spectrum_shifted
            .iter()
            .enumerate()
            .map(|(r, row)| {
                let fy_idx = r as f64 - ny as f64 / 2.0;
                let fy = fy_idx / (ny as f64 * dx_um);
                row.iter()
                    .enumerate()
                    .map(|(c, &val)| {
                        let fx_idx = c as f64 - nx as f64 / 2.0;
                        let fx = fx_idx / (nx as f64 * dx_um);
                        let quadratic_phase = -PI * lambda_um * z_um * (fx * fx + fy * fy);
                        let h = propagation_phase * Complex64::new(0.0, quadratic_phase).exp();
                        val * h
                    })
                    .collect()
            })
            .collect();

        let output_unshifted = fftshift_2d(&output_spectrum);
        ifft_2d(&output_unshifted)
    }

    /// Fraunhofer (far-field) diffraction.
    ///
    /// In the far field: U(x,y,z) ∝ FT\[U₀\](x/(λz), y/(λz))
    ///
    /// This implementation returns the 2D FFT of the input field, scaled
    /// to physical coordinates at distance z.
    pub fn fraunhofer(
        field: &[Vec<Complex64>],
        dx_um: f64,
        z_um: f64,
        lambda_nm: f64,
    ) -> Vec<Vec<Complex64>> {
        let ny = field.len();
        if ny == 0 {
            return Vec::new();
        }
        let nx = field[0].len();
        let lambda_um = lambda_nm * 1e-3;

        // Scale factor: Δx_out = λz / (N·Δx_in)
        let scale_x = lambda_um * z_um / (nx as f64 * dx_um);
        let _scale_y = lambda_um * z_um / (ny as f64 * dx_um);

        // Normalization constant: 1/(iλz)
        let norm = 1.0 / (lambda_um * z_um);

        let spectrum = fft_2d(field);
        let shifted = fftshift_2d(&spectrum);

        // Apply scale normalization and quadratic phase (dropped in far-field approx)
        shifted
            .iter()
            .map(|row| row.iter().map(|&v| v * norm * (dx_um * scale_x)).collect())
            .collect()
    }

    /// Public 2D FFT (for external use / testing).
    pub fn fft2d(field: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
        fft_2d(field)
    }

    /// Fresnel number N_F = a² / (λ · z).
    ///
    /// Determines propagation regime:
    /// - N_F >> 1: Fresnel (paraxial / near-field)
    /// - N_F << 1: Fraunhofer (far-field)
    pub fn fresnel_number(aperture_um: f64, z_um: f64, lambda_nm: f64) -> f64 {
        let lambda_um = lambda_nm * 1e-3;
        aperture_um * aperture_um / (lambda_um * z_um)
    }

    /// Returns true if propagation is in the Fraunhofer (far-field) regime.
    ///
    /// Criterion: N_F < 0.1
    pub fn is_fraunhofer(aperture_um: f64, z_um: f64, lambda_nm: f64) -> bool {
        Self::fresnel_number(aperture_um, z_um, lambda_nm) < 0.1
    }

    /// Returns true if propagation is in the Fresnel regime.
    ///
    /// Criterion: 1 < N_F < a/λ (paraxial but not far-field)
    pub fn is_fresnel(aperture_um: f64, z_um: f64, lambda_nm: f64) -> bool {
        let nf = Self::fresnel_number(aperture_um, z_um, lambda_nm);
        nf > 1.0
    }
}

// ---------------------------------------------------------------------------
// DiffractiveLens
// ---------------------------------------------------------------------------

/// Type of diffractive lens / zone plate.
#[derive(Debug, Clone, PartialEq)]
pub enum DiffractiveLensType {
    /// Binary amplitude zone plate (alternating opaque/transparent rings): η ≈ 10.1%
    BinaryAmplitude,
    /// Binary phase zone plate (0/π phase alternation): η ≈ 40.5%
    BinaryPhase,
    /// Multi-level phase lens (staircase approximation to continuous): η increases with levels
    MultiLevel {
        /// Number of discrete phase levels
        n_levels: usize,
    },
    /// Continuous phase kinoform (ideal): η = 100%
    Continuous,
}

/// Diffractive lens: Fresnel zone plate or kinoform.
///
/// A diffractive lens focuses by diffraction rather than refraction.
/// It consists of concentric phase/amplitude rings (Fresnel zones).
///
/// The zone radii satisfy:
///   r_m = sqrt(m · λ · f)  for m = 1, 2, ..., N_zones
///
/// and the phase profile of a kinoform is:
///   φ(r) = mod(π · r² / (λ · f), 2π)
#[derive(Debug, Clone)]
pub struct DiffractiveLens {
    /// Focal length (mm)
    pub focal_length_mm: f64,
    /// Lens diameter (mm)
    pub diameter_mm: f64,
    /// Design wavelength (nm)
    pub wavelength_nm: f64,
    /// Number of Fresnel zones (computed from parameters)
    pub n_zones: usize,
    /// Lens type
    pub lens_type: DiffractiveLensType,
}

impl DiffractiveLens {
    /// Create a new diffractive lens.
    ///
    /// N_zones is computed automatically as floor(D² / (4·λ·f)).
    pub fn new(
        focal_mm: f64,
        diameter_mm: f64,
        lambda_nm: f64,
        lens_type: DiffractiveLensType,
    ) -> Self {
        let lambda_mm = lambda_nm * 1e-6; // nm → mm
        let n_zones = ((diameter_mm * diameter_mm) / (4.0 * lambda_mm * focal_mm))
            .floor()
            .max(1.0) as usize;
        Self {
            focal_length_mm: focal_mm,
            diameter_mm,
            wavelength_nm: lambda_nm,
            n_zones,
            lens_type,
        }
    }

    /// Number of Fresnel zones N = D² / (4·λ·f).
    pub fn n_zones(&self) -> usize {
        self.n_zones
    }

    /// Radii of Fresnel zone boundaries: r_m = sqrt(m · λ · f) (mm).
    ///
    /// Returns N_zones radii for m = 1 .. N_zones.
    pub fn zone_radii_mm(&self) -> Vec<f64> {
        let lambda_mm = self.wavelength_nm * 1e-6;
        (1..=self.n_zones)
            .map(|m| (m as f64 * lambda_mm * self.focal_length_mm).sqrt())
            .collect()
    }

    /// Width of the outermost Fresnel zone δr = r_N − r_{N-1} ≈ λf/D (μm).
    ///
    /// This is the finest feature size and sets the NA.
    pub fn outermost_zone_width_um(&self) -> f64 {
        let radii = self.zone_radii_mm();
        if radii.is_empty() {
            return 0.0;
        }
        let r_n = radii[radii.len() - 1];
        let r_nm1 = if radii.len() > 1 {
            radii[radii.len() - 2]
        } else {
            0.0
        };
        (r_n - r_nm1) * 1e3 // mm → μm
    }

    /// Numerical aperture NA = D / (2·f).
    ///
    /// Also equal to λ / (2·δr) where δr is the outermost zone width.
    pub fn numerical_aperture(&self) -> f64 {
        self.diameter_mm / (2.0 * self.focal_length_mm)
    }

    /// First-order diffraction efficiency for the chosen lens type:
    /// - BinaryAmplitude: η = 1/π² ≈ 10.13%
    /// - BinaryPhase:     η = 4/π² ≈ 40.53%
    /// - MultiLevel(L):   η = sinc²(1/L) → approaches 1 for large L
    /// - Continuous:      η = 1 (100%)
    pub fn diffraction_efficiency(&self) -> f64 {
        match &self.lens_type {
            DiffractiveLensType::BinaryAmplitude => 1.0 / (PI * PI),
            DiffractiveLensType::BinaryPhase => 4.0 / (PI * PI),
            DiffractiveLensType::MultiLevel { n_levels } => {
                let l = *n_levels as f64;
                let x = 1.0 / l;
                if x.abs() < 1e-12 {
                    1.0
                } else {
                    let pix = PI * x;
                    (pix.sin() / pix).powi(2)
                }
            }
            DiffractiveLensType::Continuous => 1.0,
        }
    }

    /// Focal depth (depth of focus) δz = 2·λ / NA² (μm).
    pub fn depth_of_focus_um(&self) -> f64 {
        let lambda_um = self.wavelength_nm * 1e-3;
        let na = self.numerical_aperture();
        if na < 1e-12 {
            return f64::INFINITY;
        }
        2.0 * lambda_um / (na * na)
    }

    /// Kinoform phase profile φ(r) = mod(π·r²/(λ·f), 2π) in radians.
    ///
    /// For r in mm.
    pub fn phase_profile(&self, r_mm: f64) -> f64 {
        let lambda_mm = self.wavelength_nm * 1e-6;
        let phi = PI * r_mm * r_mm / (lambda_mm * self.focal_length_mm);
        phi % (2.0 * PI)
    }

    /// 1D axial diffraction pattern (on-axis intensity vs radial position in focal plane).
    ///
    /// Returns (x_mm, intensity) for n_pts samples across the lens diameter.
    pub fn diffraction_pattern(&self, n_pts: usize) -> Vec<(f64, f64)> {
        if n_pts == 0 {
            return Vec::new();
        }
        let lambda_mm = self.wavelength_nm * 1e-6;
        let r_max = self.diameter_mm / 2.0;
        let dr = 2.0 * r_max / n_pts as f64;

        // Compute 1D Fourier transform of aperture function in focal plane
        // For a zone plate: u(x) ∝ ∫ A(r) exp(iφ(r)) exp(-i·2π·r·x/(λ·f)) · r dr
        // Discrete sum approximation
        let n_r = n_pts * 4; // oversample for integration
        let dr_fine = 2.0 * r_max / n_r as f64;

        (0..n_pts)
            .map(|i| {
                let x_mm = -r_max + (i as f64 + 0.5) * dr;
                // Spatial frequency for this output position: fx = x/(λf)
                let fx = x_mm / (lambda_mm * self.focal_length_mm);

                let mut re = 0.0_f64;
                let mut im = 0.0_f64;
                for j in 0..n_r {
                    let r = -r_max + (j as f64 + 0.5) * dr_fine;
                    let phi = self.phase_profile(r.abs());
                    let transmission = Complex64::new(0.0, phi).exp();
                    // Aperture function: 1 inside lens, 0 outside
                    let aperture = if r.abs() <= r_max { 1.0 } else { 0.0 };
                    let phase_out = -2.0 * PI * r * fx;
                    let integrand = transmission * aperture * Complex64::new(0.0, phase_out).exp();
                    re += integrand.re * dr_fine;
                    im += integrand.im * dr_fine;
                }
                let intensity = re * re + im * im;
                (x_mm, intensity)
            })
            .collect()
    }

    /// Chromatic aberration sensitivity: δf/δλ = -f/λ (mm/nm).
    ///
    /// A diffractive lens has the opposite sign to a refractive lens,
    /// and much stronger chromatic aberration: shorter wavelengths focus farther.
    pub fn chromatic_sensitivity_mm_per_nm(&self) -> f64 {
        -self.focal_length_mm / self.wavelength_nm
    }
}

// ---------------------------------------------------------------------------
// SlmHologram
// ---------------------------------------------------------------------------

/// Spatial Light Modulator (SLM) hologram for beam shaping and wavefront control.
///
/// Models a phase-only SLM with pixel pitch `pixel_pitch_um` and `n_pixels_x × n_pixels_y` pixels.
/// Phase range: \[0, max_phase_rad\] (typically 2π).
#[derive(Debug, Clone)]
pub struct SlmHologram {
    /// Pixel pitch (μm)
    pub pixel_pitch_um: f64,
    /// Number of pixels in x
    pub n_pixels_x: usize,
    /// Number of pixels in y
    pub n_pixels_y: usize,
    /// Maximum phase (rad); typically 2π
    pub max_phase_rad: f64,
    /// Design wavelength (nm)
    pub wavelength_nm: f64,
}

impl SlmHologram {
    /// Create a new SLM hologram model.
    pub fn new(pitch_um: f64, nx: usize, ny: usize, lambda_nm: f64) -> Self {
        Self {
            pixel_pitch_um: pitch_um,
            n_pixels_x: nx,
            n_pixels_y: ny,
            max_phase_rad: 2.0 * PI,
            wavelength_nm: lambda_nm,
        }
    }

    /// Gerchberg-Saxton (GS) algorithm for computer-generated hologram synthesis.
    ///
    /// Finds a phase-only hologram such that the far-field intensity matches
    /// `target_intensity` (normalized to sum = 1).
    ///
    /// Algorithm (iterative):
    /// 1. Start with random phase, unit amplitude in SLM plane
    /// 2. Forward FFT → target plane
    /// 3. Replace amplitude with sqrt(target), keep computed phase
    /// 4. Inverse FFT → SLM plane
    /// 5. Replace amplitude with 1 (phase only), keep computed phase
    /// 6. Repeat for `n_iterations` steps
    ///
    /// Returns the phase map φ(x,y) ∈ \[0, 2π\].
    pub fn gerchberg_saxton(
        &self,
        target_intensity: &[Vec<f64>],
        n_iterations: usize,
    ) -> Vec<Vec<f64>> {
        let ny = self.n_pixels_y;
        let nx = self.n_pixels_x;

        if ny == 0 || nx == 0 || target_intensity.is_empty() {
            return vec![vec![0.0; nx]; ny];
        }

        // Normalize target to total power = 1
        let total: f64 = target_intensity
            .iter()
            .flat_map(|row| row.iter())
            .sum::<f64>();
        let norm = if total < 1e-30 { 1.0 } else { 1.0 / total };

        let target_amp: Vec<Vec<f64>> = target_intensity
            .iter()
            .map(|row| row.iter().map(|&v| (v * norm).max(0.0).sqrt()).collect())
            .collect();

        // Initialize with uniform amplitude, deterministic phase seed
        let mut slm_field: Vec<Vec<Complex64>> = (0..ny)
            .map(|r| {
                (0..nx)
                    .map(|c| {
                        // Deterministic pseudo-random initial phase (avoids rand dependency)
                        let phase = ((r * nx + c) as f64 * 2.654_123_7) % (2.0 * PI);
                        Complex64::new(phase.cos(), phase.sin())
                    })
                    .collect()
            })
            .collect();

        for _ in 0..n_iterations {
            // 1. Forward FFT
            let spectrum = fft_2d(&slm_field);
            let spectrum_shifted = fftshift_2d(&spectrum);

            // 2. Replace amplitude with target, keep phase
            let constrained_spectrum: Vec<Vec<Complex64>> = spectrum_shifted
                .iter()
                .enumerate()
                .map(|(r, row)| {
                    let t_row = target_amp.get(r).map(|tr| tr.as_slice()).unwrap_or(&[]);
                    row.iter()
                        .enumerate()
                        .map(|(c, &v)| {
                            let target_a = if c < t_row.len() { t_row[c] } else { 0.0 };
                            let phase = v.arg();
                            Complex64::new(target_a * phase.cos(), target_a * phase.sin())
                        })
                        .collect()
                })
                .collect();

            // 3. Inverse FFT → SLM plane
            let unshifted = fftshift_2d(&constrained_spectrum);
            let slm_back = ifft_2d(&unshifted);

            // 4. Replace amplitude with 1, keep phase (phase-only constraint)
            slm_field = slm_back
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|&v| {
                            let phase = v.arg();
                            Complex64::new(phase.cos(), phase.sin())
                        })
                        .collect()
                })
                .collect();
        }

        // Extract phase map in [0, 2π]
        slm_field
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&v| {
                        let phase = v.arg(); // in [-π, π]
                        if phase < 0.0 {
                            phase + 2.0 * PI
                        } else {
                            phase
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Grating+lens combined hologram for beam steering and focusing.
    ///
    /// Phase: φ(x,y) = 2π·sin(θ)·x/λ + π·(x²+y²)/(λ·f)
    ///
    /// - `steering_angle_mrad` — beam steering angle (mrad)
    /// - `focal_length_mm` — focal length for focusing term (mm)
    pub fn grating_lens(&self, steering_angle_mrad: f64, focal_length_mm: f64) -> Vec<Vec<f64>> {
        let pitch_mm = self.pixel_pitch_um * 1e-3;
        let lambda_mm = self.wavelength_nm * 1e-6;
        let sin_theta = (steering_angle_mrad * 1e-3).sin();

        (0..self.n_pixels_y)
            .map(|r| {
                let y_mm = (r as f64 - self.n_pixels_y as f64 / 2.0) * pitch_mm;
                (0..self.n_pixels_x)
                    .map(|c| {
                        let x_mm = (c as f64 - self.n_pixels_x as f64 / 2.0) * pitch_mm;
                        let grating_phase = 2.0 * PI * sin_theta * x_mm / lambda_mm;
                        let lens_phase = if focal_length_mm.abs() > 1e-30 {
                            PI * (x_mm * x_mm + y_mm * y_mm) / (lambda_mm * focal_length_mm)
                        } else {
                            0.0
                        };
                        let total = (grating_phase + lens_phase) % (2.0 * PI);
                        if total < 0.0 {
                            total + 2.0 * PI
                        } else {
                            total
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Spot array hologram: generates N_x × N_y spots in the far field.
    ///
    /// Implements a simple Dammann-like phase pattern for uniform N_x × N_y spots.
    /// Phase = sum of two 1D Dammann gratings (separable product).
    pub fn spot_array(&self, n_spots_x: usize, n_spots_y: usize) -> Vec<Vec<f64>> {
        let nx = self.n_pixels_x.max(1);
        let ny = self.n_pixels_y.max(1);
        let ns_x = n_spots_x.max(1);
        let ns_y = n_spots_y.max(1);

        (0..ny)
            .map(|r| {
                (0..nx)
                    .map(|c| {
                        // Fractional position within one period
                        let xf = (c % (nx / ns_x + 1)) as f64 / (nx / ns_x + 1) as f64;
                        let yf = (r % (ny / ns_y + 1)) as f64 / (ny / ns_y + 1) as f64;
                        // Binary phase: 0 or π based on Dammann transition
                        let px = if xf < 0.5 { 0.0 } else { PI };
                        let py = if yf < 0.5 { 0.0 } else { PI };
                        (px + py) % (2.0 * PI)
                    })
                    .collect()
            })
            .collect()
    }

    /// Estimate hologram efficiency: fraction of power diffracted into target region.
    ///
    /// Computes ∑_target |U|² / ∑_all |U|² where U is the far-field field
    /// obtained by FFT of the phase hologram.
    pub fn efficiency_estimate(&self, phase_map: &[Vec<f64>], target: &[Vec<f64>]) -> f64 {
        let ny = phase_map.len();
        if ny == 0 {
            return 0.0;
        }
        let nx = phase_map[0].len();

        // Build complex field from phase map
        let slm_field: Vec<Vec<Complex64>> = phase_map
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&phi| Complex64::new(phi.cos(), phi.sin()))
                    .collect()
            })
            .collect();

        // Far-field via FFT
        let far_field = fft_2d(&slm_field);

        // Total power
        let total_power: f64 = far_field
            .iter()
            .flat_map(|row| row.iter().map(|v| v.norm_sqr()))
            .sum();

        if total_power < 1e-60 {
            return 0.0;
        }

        // Target region power (non-zero entries of target mask)
        let target_power: f64 = far_field
            .iter()
            .enumerate()
            .take(ny)
            .flat_map(|(r, row)| {
                row.iter().enumerate().take(nx).map(move |(c, v)| {
                    let t = target
                        .get(r)
                        .and_then(|tr| tr.get(c))
                        .copied()
                        .unwrap_or(0.0);
                    if t > 1e-12 {
                        v.norm_sqr()
                    } else {
                        0.0
                    }
                })
            })
            .sum();

        (target_power / total_power).clamp(0.0, 1.0)
    }

    /// Add Zernike polynomial wavefront correction to a base phase map.
    ///
    /// `zernike_coeffs`: list of (n, m, coefficient_rad) where (n,m) are
    /// Zernike polynomial indices (OSA convention) and coefficient is in radians.
    ///
    /// Z(n,m)(r,θ) for r = normalized radial coordinate \[0,1\], θ = azimuth.
    pub fn add_zernike_correction(
        &self,
        base: &[Vec<f64>],
        zernike_coeffs: &[(usize, usize, f64)],
    ) -> Vec<Vec<f64>> {
        let ny = self.n_pixels_y;
        let nx = self.n_pixels_x;
        let r_max = (nx.min(ny) as f64) / 2.0;

        (0..ny)
            .map(|r| {
                (0..nx)
                    .map(|c| {
                        // Normalize coordinates to unit circle
                        let dx = c as f64 - nx as f64 / 2.0;
                        let dy = r as f64 - ny as f64 / 2.0;
                        let rho = (dx * dx + dy * dy).sqrt() / r_max;
                        let theta = dy.atan2(dx);

                        // Sum Zernike contributions
                        let correction: f64 = zernike_coeffs
                            .iter()
                            .map(|&(n, m, coeff)| coeff * zernike_polynomial(n, m, rho, theta))
                            .sum();

                        let base_val = base
                            .get(r)
                            .and_then(|row| row.get(c))
                            .copied()
                            .unwrap_or(0.0);
                        let total = base_val + correction;
                        let modded = total % (2.0 * PI);
                        if modded < 0.0 {
                            modded + 2.0 * PI
                        } else {
                            modded
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Zernike polynomial evaluation
// ---------------------------------------------------------------------------

/// Evaluate the Zernike polynomial Z_n^m(ρ, θ) using the OSA standard.
///
/// For |m| ≤ n and (n - |m|) even:
///   Z_n^m(ρ,θ) = R_n^|m|(ρ) · cos(m·θ)  if m ≥ 0
///   Z_n^m(ρ,θ) = R_n^|m|(ρ) · sin(|m|·θ) if m < 0
///
/// Radial polynomial R_n^m evaluated via the Jacobi-polynomial formula.
fn zernike_polynomial(n: usize, m: usize, rho: f64, theta: f64) -> f64 {
    let m_signed = m as i64;
    let abs_m = m_signed.unsigned_abs() as usize;

    if rho > 1.0 {
        return 0.0; // outside unit circle
    }
    if (n as i64 - m_signed).rem_euclid(2) != 0 {
        return 0.0; // undefined Zernike index
    }

    let r = zernike_radial(n, abs_m, rho);
    if m_signed >= 0 {
        r * (abs_m as f64 * theta).cos()
    } else {
        r * (abs_m as f64 * theta).sin()
    }
}

/// Radial Zernike polynomial R_n^m(ρ).
fn zernike_radial(n: usize, m: usize, rho: f64) -> f64 {
    if n < m {
        return 0.0;
    }
    if (n - m) % 2 != 0 {
        return 0.0;
    }
    let n_max_k = (n - m) / 2;
    let mut result = 0.0;
    for k in 0..=n_max_k {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let num = factorial(n - k) as f64;
        let denom = (factorial(k) * factorial((n + m) / 2 - k) * factorial((n - m) / 2 - k)) as f64;
        let power = rho.powi((n - 2 * k) as i32);
        result += sign * (num / denom) * power;
    }
    result
}

/// Integer factorial (saturates at u64::MAX for large n).
fn factorial(n: usize) -> u64 {
    if n <= 1 {
        return 1;
    }
    (2..=n as u64).fold(1u64, |acc, x| acc.saturating_mul(x))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diffractive::grating::HolographicGrating;

    // 1. Fresnel number formula: N_F = a²/(λ·z)
    #[test]
    fn test_fresnel_number_formula() {
        let a = 100.0; // 100 μm aperture
        let z = 10_000.0; // 10 mm = 10000 μm
        let lambda = 0.5; // 500 nm = 0.5 μm → lambda_nm = 500
        let nf = ScalarDiffraction::fresnel_number(a, z, lambda * 1e3);
        let expected = a * a / (lambda * z);
        assert!(
            (nf - expected).abs() < 1e-10,
            "Fresnel number mismatch: {nf} vs {expected}"
        );
    }

    // 2. Fraunhofer regime: large z → N_F << 1
    #[test]
    fn test_fraunhofer_regime() {
        // 10 μm aperture, 10 cm propagation, 500 nm → N_F = 100²/(0.5·100000) = 0.002
        let is_ff = ScalarDiffraction::is_fraunhofer(10.0, 100_000.0, 500.0);
        assert!(
            is_ff,
            "Should be in Fraunhofer regime for small aperture / large z"
        );
    }

    // 3. Fresnel regime: small z → N_F > 1
    #[test]
    fn test_fresnel_regime() {
        // 1 mm aperture, 1 mm propagation, 500 nm → N_F = 1000²/(0.5·1000) = 2000
        let is_fr = ScalarDiffraction::is_fresnel(1000.0, 1000.0, 500.0);
        assert!(
            is_fr,
            "Should be in Fresnel regime for large aperture / small z"
        );
    }

    // 4. Zone plate radii: r_m = sqrt(m·λ·f)
    #[test]
    fn test_zone_plate_radii() {
        let lens = DiffractiveLens::new(100.0, 2.0, 500.0, DiffractiveLensType::Continuous);
        let radii = lens.zone_radii_mm();
        let lambda_mm = 500e-6;
        for (idx, &r) in radii.iter().enumerate() {
            let m = (idx + 1) as f64;
            let expected = (m * lambda_mm * 100.0_f64).sqrt();
            assert!(
                (r - expected).abs() < 1e-9,
                "Zone {m} radius: {r:.6} vs expected {expected:.6}"
            );
        }
    }

    // 5. NA = D/(2f)
    #[test]
    fn test_diffractive_lens_na() {
        let lens = DiffractiveLens::new(100.0, 2.0, 500.0, DiffractiveLensType::Continuous);
        let na = lens.numerical_aperture();
        let expected = 2.0 / (2.0 * 100.0);
        assert!(
            (na - expected).abs() < 1e-10,
            "NA={na}, expected {expected}"
        );
    }

    // 6. Kinoform efficiency = 1.0
    #[test]
    fn test_diffractive_lens_efficiency_kinoform() {
        let lens = DiffractiveLens::new(50.0, 1.0, 1064.0, DiffractiveLensType::Continuous);
        let eta = lens.diffraction_efficiency();
        assert!(
            (eta - 1.0).abs() < 1e-10,
            "Kinoform efficiency should be 1.0, got {eta}"
        );
    }

    // 7. Chromatic sensitivity δf/δλ = -f/λ < 0
    #[test]
    fn test_chromatic_sensitivity() {
        let lens = DiffractiveLens::new(100.0, 5.0, 550.0, DiffractiveLensType::Continuous);
        let sens = lens.chromatic_sensitivity_mm_per_nm();
        let expected = -100.0 / 550.0;
        assert!(
            (sens - expected).abs() < 1e-10,
            "Chromatic sensitivity: {sens}, expected {expected}"
        );
        assert!(
            sens < 0.0,
            "Chromatic sensitivity must be negative for diffractive lens"
        );
    }

    // 8. SLM grating+lens output matches SLM dimensions
    #[test]
    fn test_slm_grating_lens_size() {
        let slm = SlmHologram::new(8.0, 64, 64, 532.0);
        let phase_map = slm.grating_lens(5.0, 200.0);
        assert_eq!(phase_map.len(), 64, "Row count should match n_pixels_y");
        assert_eq!(phase_map[0].len(), 64, "Col count should match n_pixels_x");
        // All phases in [0, 2π]
        for row in &phase_map {
            for &phi in row {
                assert!(
                    (0.0..=2.0 * PI + 1e-10).contains(&phi),
                    "Phase out of range: {phi}"
                );
            }
        }
    }

    // 9. Holographic grating: Raman-Nath parameter Q
    #[test]
    fn test_holographic_grating_thin() {
        // Thin grating: Q << 1
        let g = HolographicGrating::new(10.0, 0.01, 1.5);
        let q = g.raman_nath_parameter(500.0, 1.0); // 1 μm thick
        assert!(q < 1.0, "Thin grating should have Q < 1, got Q = {q:.4}");
    }

    // 10. Angular spectrum: propagate z=0 → field unchanged
    #[test]
    fn test_angular_spectrum_identity() {
        // Small 4×4 field for efficiency
        let nx = 4;
        let ny = 4;
        let field: Vec<Vec<Complex64>> = (0..ny)
            .map(|r| {
                (0..nx)
                    .map(|c| Complex64::new((r + c) as f64, 0.0))
                    .collect()
            })
            .collect();

        // Propagate zero distance: output ≈ input
        let out = ScalarDiffraction::angular_spectrum(&field, 1.0, 1.0, 0.0, 500.0);

        let max_err = field
            .iter()
            .zip(out.iter())
            .flat_map(|(row_in, row_out)| {
                row_in
                    .iter()
                    .zip(row_out.iter())
                    .map(|(a, b)| (a - b).norm())
            })
            .fold(0.0_f64, f64::max);

        assert!(
            max_err < 1e-8,
            "Angular spectrum at z=0 should return input field, max_err={max_err:.2e}"
        );
    }
}
