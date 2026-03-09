use std::f64::consts::PI;

use oxifft::{fft, ifft, Complex};

/// 1D FFT-BPM (split-step beam propagation method).
///
/// Solves the paraxial Helmholtz equation:
///   ∂E/∂z = (i/2k) ∂²E/∂x² + ik₀ Δn E
///
/// where Δn = n(x) - n_ref (refractive index contrast from reference).
///
/// Split-step algorithm per z-step dz:
///   1. Half-step free-space phase in k-space: Ê *= exp(-i kx²/(2k₀n_ref) · dz/2)
///   2. Full-step material phase in real space: E *= exp(i k₀ Δn dz)
///   3. Half-step free-space phase in k-space: Ê *= exp(-i kx²/(2k₀n_ref) · dz/2)
pub struct FftBpm1d {
    /// Number of transverse grid points.
    pub nx: usize,
    /// Transverse grid spacing (m).
    pub dx: f64,
    /// Reference refractive index (background medium).
    pub n_ref: f64,
    /// Free-space wavelength (m).
    pub wavelength: f64,
    /// Refractive index profile n(x). If None, use n_ref (free space).
    pub n_profile: Option<Vec<f64>>,
    /// Current field E(x) — complex amplitude.
    pub field: Vec<Complex<f64>>,
}

impl FftBpm1d {
    /// Create a new 1D FFT-BPM simulation.
    ///
    /// # Arguments
    /// - `nx`: number of transverse grid points (power-of-2 recommended)
    /// - `dx`: transverse grid spacing (m)
    /// - `n_ref`: reference (background) refractive index
    /// - `wavelength`: free-space wavelength (m)
    pub fn new(nx: usize, dx: f64, n_ref: f64, wavelength: f64) -> Self {
        Self {
            nx,
            dx,
            n_ref,
            wavelength,
            n_profile: None,
            field: vec![Complex::zero(); nx],
        }
    }

    /// Set the refractive index profile.
    pub fn set_index_profile(&mut self, n_profile: Vec<f64>) {
        assert_eq!(n_profile.len(), self.nx);
        self.n_profile = Some(n_profile);
    }

    /// Set a Gaussian input beam at position x_center with waist w0.
    pub fn set_gaussian_input(&mut self, amplitude: f64, x_center: f64, w0: f64) {
        for i in 0..self.nx {
            let x = i as f64 * self.dx - x_center;
            let envelope = (-x * x / (w0 * w0)).exp();
            self.field[i] = Complex::new(amplitude * envelope, 0.0);
        }
    }

    /// Set field from complex amplitude array.
    pub fn set_field(&mut self, field: Vec<Complex<f64>>) {
        assert_eq!(field.len(), self.nx);
        self.field = field;
    }

    /// Propagate by one z-step dz (m) using the split-step method.
    pub fn step(&mut self, dz: f64) {
        let k0 = 2.0 * PI / self.wavelength;
        let kn = k0 * self.n_ref; // k0 n_ref (propagation constant of reference medium)

        // Precompute spatial frequencies (centered FFT convention)
        // kx[i] = 2π · i / (nx · dx) for i < nx/2
        //       = 2π · (i - nx) / (nx · dx) for i >= nx/2
        let kx: Vec<f64> = (0..self.nx)
            .map(|i| {
                let ki = if i < self.nx / 2 {
                    i as f64
                } else {
                    i as f64 - self.nx as f64
                };
                2.0 * PI * ki / (self.nx as f64 * self.dx)
            })
            .collect();

        // Half-step free-space phase factor: exp(-i kx²/(2k₀n₀) · dz/2)
        // Each application propagates half a step; two applications = full step.
        let half_free_phase: Vec<Complex<f64>> = kx
            .iter()
            .map(|&kxi| {
                let phase = -kxi * kxi * dz / (4.0 * kn);
                Complex::new(0.0, phase).exp_im()
            })
            .collect();

        // Step 1: forward FFT
        let mut spec = fft(&self.field);

        // Step 2: half free-space phase
        for (s, &h) in spec.iter_mut().zip(&half_free_phase) {
            *s *= h;
        }

        // Step 3: inverse FFT → real space
        let mut e = ifft(&spec);

        // Step 4: material phase exp(i k0 Δn dz)
        if let Some(ref np) = self.n_profile {
            for (ei, &ni) in e.iter_mut().zip(np) {
                let dn = ni - self.n_ref;
                let phase = k0 * dn * dz;
                *ei *= Complex::new(0.0, phase).exp_im();
            }
        }
        // (if no profile, only free-space propagation — no material phase)

        // Step 5: forward FFT again
        let mut spec2 = fft(&e);

        // Step 6: second half free-space phase
        for (s, &h) in spec2.iter_mut().zip(&half_free_phase) {
            *s *= h;
        }

        // Step 7: inverse FFT → final field
        self.field = ifft(&spec2);
    }

    /// Propagate by n_steps × dz.
    pub fn propagate(&mut self, dz: f64, n_steps: usize) {
        for _ in 0..n_steps {
            self.step(dz);
        }
    }

    /// Get the field intensity |E(x)|².
    pub fn intensity(&self) -> Vec<f64> {
        self.field
            .iter()
            .map(|e| e.re * e.re + e.im * e.im)
            .collect()
    }

    /// Get the peak intensity (max of |E|²).
    pub fn peak_intensity(&self) -> f64 {
        self.intensity().iter().cloned().fold(0.0_f64, f64::max)
    }

    /// Get the 1/e² beam width (second moment or FWHM-based).
    ///
    /// Returns the RMS width (σ) of the intensity distribution.
    pub fn rms_width(&self) -> f64 {
        let intensity = self.intensity();
        let total: f64 = intensity.iter().sum();
        if total == 0.0 {
            return 0.0;
        }
        let mean = intensity
            .iter()
            .enumerate()
            .map(|(i, &v)| i as f64 * self.dx * v)
            .sum::<f64>()
            / total;
        let var = intensity
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let x = i as f64 * self.dx - mean;
                x * x * v
            })
            .sum::<f64>()
            / total;
        var.sqrt()
    }
}

/// Extension trait for complex exponential.
trait ComplexExpIm {
    /// Compute exp(i * self.im) where self.re = 0 (pure imaginary exponent).
    fn exp_im(self) -> Self;
}

impl ComplexExpIm for Complex<f64> {
    fn exp_im(self) -> Self {
        // exp(i·θ) = cos(θ) + i·sin(θ)
        let (s, c) = self.im.sin_cos();
        Complex::new(c, s)
    }
}

/// 2D FFT-BPM: propagation in z, transverse plane (x, y).
///
/// Uses a 2D split-step approach. Field is stored as row-major E[j*nx + i].
pub struct FftBpm2d {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    pub n_ref: f64,
    pub wavelength: f64,
    pub n_profile: Option<Vec<f64>>,
    pub field: Vec<Complex<f64>>,
}

impl FftBpm2d {
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64, n_ref: f64, wavelength: f64) -> Self {
        Self {
            nx,
            ny,
            dx,
            dy,
            n_ref,
            wavelength,
            n_profile: None,
            field: vec![Complex::zero(); nx * ny],
        }
    }

    pub fn set_index_profile(&mut self, n_profile: Vec<f64>) {
        assert_eq!(n_profile.len(), self.nx * self.ny);
        self.n_profile = Some(n_profile);
    }

    /// Set a 2D Gaussian input beam centered at (xc, yc) with waist w0.
    pub fn set_gaussian_input(&mut self, amplitude: f64, xc: f64, yc: f64, w0: f64) {
        for j in 0..self.ny {
            for i in 0..self.nx {
                let x = i as f64 * self.dx - xc;
                let y = j as f64 * self.dy - yc;
                let r2 = x * x + y * y;
                let env = (-r2 / (w0 * w0)).exp();
                self.field[j * self.nx + i] = Complex::new(amplitude * env, 0.0);
            }
        }
    }

    pub fn step(&mut self, dz: f64) {
        let k0 = 2.0 * PI / self.wavelength;
        let kn = k0 * self.n_ref;
        let nxy = self.nx * self.ny;

        // Compute 2D spatial frequencies
        let kx_vec: Vec<f64> = (0..self.nx)
            .map(|i| {
                let ki = if i < self.nx / 2 {
                    i as f64
                } else {
                    i as f64 - self.nx as f64
                };
                2.0 * PI * ki / (self.nx as f64 * self.dx)
            })
            .collect();
        let ky_vec: Vec<f64> = (0..self.ny)
            .map(|j| {
                let kj = if j < self.ny / 2 {
                    j as f64
                } else {
                    j as f64 - self.ny as f64
                };
                2.0 * PI * kj / (self.ny as f64 * self.dy)
            })
            .collect();

        // Half-step free-space phase factor: exp(-i (kx²+ky²)/(2kn) · dz/2)
        let free_phase: Vec<Complex<f64>> = (0..nxy)
            .map(|k| {
                let i = k % self.nx;
                let j = k / self.nx;
                let phase = -(kx_vec[i] * kx_vec[i] + ky_vec[j] * ky_vec[j]) * dz / (4.0 * kn);
                Complex::new(0.0, phase).exp_im()
            })
            .collect();

        // Forward 2D FFT
        let mut spec = oxifft::fft2d(&self.field, self.ny, self.nx);

        // Half free-space phase
        for (s, &h) in spec.iter_mut().zip(&free_phase) {
            *s *= h;
        }

        // Inverse 2D FFT
        let mut e = oxifft::ifft2d(&spec, self.ny, self.nx);

        // Material phase
        if let Some(ref np) = self.n_profile {
            for (ei, &ni) in e.iter_mut().zip(np) {
                let dn = ni - self.n_ref;
                let phase = k0 * dn * dz;
                *ei *= Complex::new(0.0, phase).exp_im();
            }
        }

        // Forward 2D FFT
        let mut spec2 = oxifft::fft2d(&e, self.ny, self.nx);

        // Second half free-space phase
        for (s, &h) in spec2.iter_mut().zip(&free_phase) {
            *s *= h;
        }

        // Inverse 2D FFT
        self.field = oxifft::ifft2d(&spec2, self.ny, self.nx);
    }

    pub fn propagate(&mut self, dz: f64, n_steps: usize) {
        for _ in 0..n_steps {
            self.step(dz);
        }
    }

    /// Get 2D intensity |E(x,y)|².
    pub fn intensity(&self) -> Vec<f64> {
        self.field
            .iter()
            .map(|e| e.re * e.re + e.im * e.im)
            .collect()
    }

    /// Peak intensity.
    pub fn peak_intensity(&self) -> f64 {
        self.intensity().iter().cloned().fold(0.0_f64, f64::max)
    }

    /// Total power (integral of |E|² × dx × dy).
    pub fn total_power(&self) -> f64 {
        self.intensity().iter().sum::<f64>() * self.dx * self.dy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_bpm_free_space_power_conservation() {
        // Gaussian beam in free space: total power should be conserved
        let nx = 256;
        let dx = 100e-9; // 100nm, total 25.6μm
        let mut bpm = FftBpm1d::new(nx, dx, 1.0, 1550e-9);
        let xc = nx as f64 * dx / 2.0;
        bpm.set_gaussian_input(1.0, xc, 1e-6); // w0 = 1μm

        let p0: f64 = bpm.intensity().iter().sum::<f64>() * dx;
        bpm.propagate(1e-7, 5); // 5 steps of 100nm
        let p1: f64 = bpm.intensity().iter().sum::<f64>() * dx;

        // Power conserved within 0.1%
        let rel_err = (p1 - p0).abs() / p0;
        assert!(rel_err < 1e-3, "Power not conserved: rel_err={rel_err:.2e}");
    }

    #[test]
    fn fft_bpm_gaussian_beam_spreads() {
        // After propagation, a Gaussian beam should spread (width increases)
        let nx = 512;
        let dx = 50e-9;
        let mut bpm = FftBpm1d::new(nx, dx, 1.0, 1550e-9);
        let xc = nx as f64 * dx / 2.0;
        let w0 = 1e-6; // 1μm
        bpm.set_gaussian_input(1.0, xc, w0);
        let w_init = bpm.rms_width();

        bpm.propagate(1e-6, 10); // 10μm propagation
        let w_final = bpm.rms_width();
        assert!(
            w_final > w_init,
            "Beam should spread: w_init={w_init:.3e} w_final={w_final:.3e}"
        );
    }

    #[test]
    fn fft_bpm_gaussian_analytical_match() {
        // Gaussian beam propagation: compare peak amplitude to analytical solution
        // Analytical: |E(0,z)| = 1 / sqrt(1 + (z/z_R)²)^(1/2)
        // For 1D BPM, peak amplitude ~ 1 / (1 + (z/z_R)²)^(1/4)
        // Actually for BPM (paraxial) the intensity peak: I_peak(z) = I_peak(0) / sqrt(1 + (z/z_R)²)
        let nx = 512;
        let dx = 30e-9; // 30nm spacing, total ~15μm domain
        let wavelength = 1550e-9;
        let n_ref = 1.0;
        let w0 = 1.5e-6; // 1.5μm waist
        let z_r = PI * w0 * w0 / wavelength; // Rayleigh range

        let mut bpm = FftBpm1d::new(nx, dx, n_ref, wavelength);
        let xc = nx as f64 * dx / 2.0;
        bpm.set_gaussian_input(1.0, xc, w0);

        let i0 = bpm.peak_intensity();

        // Propagate small fraction of Rayleigh range: 2% (high accuracy)
        let dz = 0.02 * z_r;
        let n_steps = 10;
        let z_total = dz * n_steps as f64;

        bpm.propagate(dz, n_steps);
        let i_final = bpm.peak_intensity();

        // Analytical 1D peak intensity: I_peak(z) = I0 / sqrt(1 + (z/z_R)²)
        let expected_ratio = 1.0 / (1.0 + (z_total / z_r).powi(2)).sqrt();
        let computed_ratio = i_final / i0;

        let rel_err = (computed_ratio - expected_ratio).abs() / expected_ratio;
        assert!(
            rel_err < 0.001,
            "Gaussian beam propagation error: rel_err={rel_err:.4e} \
             (computed={computed_ratio:.6}, expected={expected_ratio:.6})"
        );
    }
}
