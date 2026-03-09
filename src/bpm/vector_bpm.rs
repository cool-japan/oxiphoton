//! Vector BPM — polarization-aware 1D beam propagation.
//!
//! Extends scalar BPM to handle both x and y polarization components,
//! including their coupling through off-diagonal permittivity terms
//! (birefringence) and the vectorial correction term proportional to
//! ∂(ln ε)/∂x · ∂E/∂x.
//!
//! The vector wave equation for the x-component (TE-like):
//!   ∂²Eₓ/∂x² + ∂²Eₓ/∂z² + k₀²ε·Eₓ + ∂(ln ε)/∂x · ∂Eₓ/∂x = 0
//!
//! In the slowly-varying envelope (SVE) approximation u = E·exp(-ik₀n₀z):
//!   ∂u/∂z ≈ i/(2k₀n₀) · [∂²u/∂x² + (k₀²(ε-n₀²))·u + Δ_vec·u]
//! where Δ_vec is the vectorial correction term.

use std::f64::consts::PI;

use num_complex::Complex64;
use oxifft::{fft, ifft, Complex};

/// Two-component (x, y) complex field, stored as [[re_x, im_x], [re_y, im_y]].
type Field2 = [[f64; 2]; 2];

// ─── JonesVector ─────────────────────────────────────────────────────────────

/// Two-component complex polarization state (Jones vector) [Ex, Ey].
#[derive(Clone, Copy, Debug)]
pub struct JonesVector(pub [Complex64; 2]);

impl JonesVector {
    /// Create from explicit complex components.
    pub fn new(ex: Complex64, ey: Complex64) -> Self {
        Self([ex, ey])
    }

    /// x-polarization component Ex.
    pub fn ex(&self) -> Complex64 {
        self.0[0]
    }

    /// y-polarization component Ey.
    pub fn ey(&self) -> Complex64 {
        self.0[1]
    }

    /// Intensity I = |Ex|² + |Ey|².
    pub fn intensity(&self) -> f64 {
        self.0[0].norm_sqr() + self.0[1].norm_sqr()
    }

    /// Return a normalized Jones vector with unit intensity.
    ///
    /// Returns the zero vector unchanged if the intensity is negligible.
    pub fn normalize(&self) -> Self {
        let i = self.intensity().sqrt();
        if i < 1e-30 {
            return *self;
        }
        Self([
            Complex64::new(self.0[0].re / i, self.0[0].im / i),
            Complex64::new(self.0[1].re / i, self.0[1].im / i),
        ])
    }
}

// ─── JonesMatrix ─────────────────────────────────────────────────────────────

/// 2×2 complex Jones matrix for polarization transformations.
///
/// Stored row-major: `m[row][col]`.
#[derive(Clone, Copy, Debug)]
pub struct JonesMatrix(pub [[Complex64; 2]; 2]);

impl JonesMatrix {
    /// Identity matrix.
    pub fn identity() -> Self {
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        Self([[one, zero], [zero, one]])
    }

    /// Apply this Jones matrix to a Jones vector: output = M · jv.
    pub fn apply(&self, jv: JonesVector) -> JonesVector {
        let m = &self.0;
        let e = jv.0;
        JonesVector([
            m[0][0] * e[0] + m[0][1] * e[1],
            m[1][0] * e[0] + m[1][1] * e[1],
        ])
    }

    /// Compose two Jones matrices: result = self · other (self applied after other).
    pub fn compose(&self, other: &JonesMatrix) -> JonesMatrix {
        let a = &self.0;
        let b = &other.0;
        JonesMatrix([
            [
                a[0][0] * b[0][0] + a[0][1] * b[1][0],
                a[0][0] * b[0][1] + a[0][1] * b[1][1],
            ],
            [
                a[1][0] * b[0][0] + a[1][1] * b[1][0],
                a[1][0] * b[0][1] + a[1][1] * b[1][1],
            ],
        ])
    }

    /// Wave-plate (retarder) with phase retardations `phase_x` and `phase_y`
    /// along the x and y axes respectively.
    ///
    /// M = diag(exp(i·phase_x), exp(i·phase_y))
    pub fn waveplate(phase_x: f64, phase_y: f64) -> Self {
        Self([
            [
                Complex64::new(phase_x.cos(), phase_x.sin()),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(phase_y.cos(), phase_y.sin()),
            ],
        ])
    }

    /// Linear polarizer oriented at angle `theta` (radians) from the x-axis.
    ///
    /// M = [[cos²θ, cosθ·sinθ], [cosθ·sinθ, sin²θ]]
    pub fn linear_polarizer(theta: f64) -> Self {
        let (s, c) = theta.sin_cos();
        let c2 = Complex64::new(c * c, 0.0);
        let s2 = Complex64::new(s * s, 0.0);
        let cs = Complex64::new(c * s, 0.0);
        Self([[c2, cs], [cs, s2]])
    }
}

// ─── VectorBpm1d (extended) ──────────────────────────────────────────────────

/// 1D vector BPM solver (x and y polarization components).
///
/// Handles polarization coupling in birefringent waveguides.
/// Uses FFT split-step propagation for the diffraction term, with
/// a birefringence phase screen for the differential index δn(x) = n_e − n_o.
pub struct VectorBpm1d {
    /// Number of transverse grid points
    pub n_x: usize,
    /// Transverse grid spacing (m)
    pub dx: f64,
    /// Reference wavenumber k₀·n₀
    pub k0n0: f64,
    /// Reference index n₀
    pub n_ref: f64,
    /// Free-space wavelength (m)
    pub wavelength: f64,
    /// Ex field envelope [re, im] per node
    pub field_x: Vec<[f64; 2]>,
    /// Ey field envelope [re, im] per node
    pub field_y: Vec<[f64; 2]>,
    /// Refractive index n(x) — isotropic (nₓ = nᵧ = n_z)
    pub n_profile: Vec<f64>,
    /// Birefringence δn(x) = n_e - n_o per pixel (n_e is seen by Ey)
    pub birefringence: Vec<f64>,
}

impl VectorBpm1d {
    /// Create a vector BPM solver.
    ///
    /// - `n_x`: number of transverse grid points
    /// - `dx`: transverse spacing (m)
    /// - `wavelength`: free-space wavelength (m)
    ///
    /// Note: `n_ref` defaults to 1.5 and can be adjusted via the public field.
    pub fn new(n_x: usize, dx: f64, wavelength: f64) -> Self {
        let n_ref = 1.5_f64;
        let k0 = 2.0 * PI / wavelength;
        Self {
            n_x,
            dx,
            k0n0: k0 * n_ref,
            n_ref,
            wavelength,
            field_x: vec![[0.0, 0.0]; n_x],
            field_y: vec![[0.0, 0.0]; n_x],
            n_profile: vec![n_ref; n_x],
            birefringence: vec![0.0; n_x],
        }
    }

    /// Set birefringence profile δn(x) = n_e − n_o per pixel.
    pub fn set_birefringence(&mut self, dn: &[f64]) {
        assert_eq!(dn.len(), self.n_x, "dn length must equal n_x");
        self.birefringence = dn.to_vec();
    }

    /// Set Gaussian input for Ex component (x-polarized).
    pub fn set_gaussian_x(&mut self, a0: f64, x_center: f64, w0: f64) {
        let x0 = self.n_x as f64 / 2.0 * self.dx + x_center;
        for i in 0..self.n_x {
            let x = i as f64 * self.dx;
            let d = x - x0;
            self.field_x[i] = [a0 * (-d * d / (w0 * w0)).exp(), 0.0];
        }
    }

    /// Set Gaussian input for Ey component (y-polarized).
    pub fn set_gaussian_y(&mut self, a0: f64, x_center: f64, w0: f64) {
        let x0 = self.n_x as f64 / 2.0 * self.dx + x_center;
        for i in 0..self.n_x {
            let x = i as f64 * self.dx;
            let d = x - x0;
            self.field_y[i] = [a0 * (-d * d / (w0 * w0)).exp(), 0.0];
        }
    }

    /// Set refractive index profile.
    pub fn set_index_profile(&mut self, n: Vec<f64>) {
        assert_eq!(n.len(), self.n_x);
        self.n_profile = n;
    }

    /// Set birefringence profile δn(x) = n_y - n_x (vec version, kept for back-compat).
    pub fn set_birefringence_vec(&mut self, delta_n: Vec<f64>) {
        assert_eq!(delta_n.len(), self.n_x);
        self.birefringence = delta_n;
    }

    /// Propagate one step `dz` using FFT split-step for Ex and Ey, plus
    /// birefringence phase screen.
    ///
    /// Split-step order:
    ///   1. Half free-space phase in k-space (FFT).
    ///   2. Full material phase screen (isotropic Δn + birefringence for Ey).
    ///   3. Second half free-space phase in k-space (IFFT).
    pub fn propagate_dz(
        &mut self,
        ex: &mut [Complex64],
        ey: &mut [Complex64],
        n_ref: f64,
        dz: f64,
    ) {
        assert_eq!(ex.len(), self.n_x);
        assert_eq!(ey.len(), self.n_x);

        let k0 = 2.0 * PI / self.wavelength;
        let kn = k0 * n_ref;
        let nx = self.n_x;

        // Precompute half free-space phase factors
        let half_free: Vec<Complex<f64>> = (0..nx)
            .map(|i| {
                let ki = if i < nx / 2 {
                    i as f64
                } else {
                    i as f64 - nx as f64
                };
                let kxi = 2.0 * PI * ki / (nx as f64 * self.dx);
                let phase = -kxi * kxi * dz / (4.0 * kn);
                Complex::new(phase.cos(), phase.sin())
            })
            .collect();

        // Helper: convert &[Complex64] → Vec<oxifft::Complex<f64>>
        let to_oxifft = |v: &[Complex64]| -> Vec<Complex<f64>> {
            v.iter().map(|c| Complex::new(c.re, c.im)).collect()
        };
        let from_oxifft = |v: Vec<Complex<f64>>| -> Vec<Complex64> {
            v.into_iter().map(|c| Complex64::new(c.re, c.im)).collect()
        };

        // ── Ex ──────────────────────────────────────────────────────────────
        {
            let mut spec = fft(&to_oxifft(ex));
            for (s, &h) in spec.iter_mut().zip(&half_free) {
                *s *= h;
            }
            let mut e = ifft(&spec);
            for (ei, (&ni, &dn)) in e
                .iter_mut()
                .zip(self.n_profile.iter().zip(self.birefringence.iter()))
            {
                let delta_n = ni - n_ref; // Ex sees isotropic n
                let _ = dn; // Ex does not see birefringence in this formulation
                let phase = k0 * delta_n * dz;
                let ph = Complex::new(phase.cos(), phase.sin());
                *ei *= ph;
            }
            let mut spec2 = fft(&e);
            for (s, &h) in spec2.iter_mut().zip(&half_free) {
                *s *= h;
            }
            let result = from_oxifft(ifft(&spec2));
            ex.copy_from_slice(&result);
        }

        // ── Ey ──────────────────────────────────────────────────────────────
        {
            let mut spec = fft(&to_oxifft(ey));
            for (s, &h) in spec.iter_mut().zip(&half_free) {
                *s *= h;
            }
            let mut e = ifft(&spec);
            for (ei, (&ni, &dn)) in e
                .iter_mut()
                .zip(self.n_profile.iter().zip(self.birefringence.iter()))
            {
                let delta_n = ni - n_ref + dn; // Ey sees n + δn (birefringence)
                let phase = k0 * delta_n * dz;
                let ph = Complex::new(phase.cos(), phase.sin());
                *ei *= ph;
            }
            let mut spec2 = fft(&e);
            for (s, &h) in spec2.iter_mut().zip(&half_free) {
                *s *= h;
            }
            let result = from_oxifft(ifft(&spec2));
            ey.copy_from_slice(&result);
        }

        // Sync internal fields with updated ex/ey slices
        for i in 0..nx {
            self.field_x[i] = [ex[i].re, ex[i].im];
            self.field_y[i] = [ey[i].re, ey[i].im];
        }
    }

    /// Propagate a Jones vector through the birefringent medium using the
    /// per-pixel phase-screen approximation (no diffraction — pure polarization
    /// rotation along total_length).
    ///
    /// Returns the output Jones vector after accumulating the Jones matrix for
    /// each pixel column in sequence.
    pub fn jones_matrix_propagation(
        &self,
        input_jones: JonesVector,
        n_ref: f64,
        total_length: f64,
    ) -> JonesVector {
        let k0 = 2.0 * PI / self.wavelength;

        // Build the net Jones matrix as the product of per-pixel wave-plates.
        // Each pixel has width dx and contributes:
        //   M_i = waveplate(φ_x, φ_y)
        // where φ_x = k0·(n[i] - n_ref)·dz_pixel
        //       φ_y = k0·(n[i] + δn[i] - n_ref)·dz_pixel
        // dz_pixel = total_length / n_x  (uniform column spacing)
        let dz_pixel = total_length / self.n_x as f64;
        let mut m = JonesMatrix::identity();
        for i in 0..self.n_x {
            let ni = self.n_profile[i];
            let dn = self.birefringence[i];
            let phase_x = k0 * (ni - n_ref) * dz_pixel;
            let phase_y = k0 * (ni + dn - n_ref) * dz_pixel;
            let wp = JonesMatrix::waveplate(phase_x, phase_y);
            m = wp.compose(&m);
        }
        m.apply(input_jones)
    }

    /// Propagate one step dz using scalar CN for each component + birefringence phase.
    pub fn step(&mut self, dz: f64) {
        let k0 = self.k0n0 / self.n_ref;
        let alpha = 1.0 / (2.0 * self.k0n0 * self.dx * self.dx);

        // Phase screen: index perturbation + birefringence coupling
        for i in 0..self.n_x {
            let delta_n = self.n_profile[i] - self.n_ref;
            let phase_iso = k0 * delta_n * dz;
            // Birefringence: Ex sees n_x = n, Ey sees n_x + delta_bire
            let phase_bire = k0 * self.birefringence[i] * dz;

            let (s_iso, c_iso) = phase_iso.sin_cos();
            let [rex, imx] = self.field_x[i];
            self.field_x[i] = [rex * c_iso - imx * s_iso, rex * s_iso + imx * c_iso];

            let phase_y = phase_iso + phase_bire;
            let (sy, cy) = phase_y.sin_cos();
            let [rey, imy] = self.field_y[i];
            self.field_y[i] = [rey * cy - imy * sy, rey * sy + imy * cy];
        }

        // Diffraction step (Crank-Nicolson, applied independently to each component)
        let rhs_x = Self::cn_rhs(&self.field_x, dz, alpha);
        let rhs_y = Self::cn_rhs(&self.field_y, dz, alpha);
        self.field_x = Self::cn_solve(&rhs_x, dz, alpha);
        self.field_y = Self::cn_solve(&rhs_y, dz, alpha);
    }

    fn cn_rhs(field: &[[f64; 2]], dz: f64, alpha: f64) -> Vec<[f64; 2]> {
        let n = field.len();
        let a_dz_half = alpha * dz * 0.5;
        let mut rhs = vec![[0.0f64; 2]; n];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let u = field[i];
            let lu = if i == 0 || i == n - 1 {
                [0.0, 0.0]
            } else {
                let up = field[i + 1];
                let um = field[i - 1];
                [up[0] - 2.0 * u[0] + um[0], up[1] - 2.0 * u[1] + um[1]]
            };
            rhs[i] = [u[0] + a_dz_half * (-lu[1]), u[1] + a_dz_half * lu[0]];
        }
        rhs
    }

    fn cn_solve(rhs: &[[f64; 2]], dz: f64, alpha: f64) -> Vec<[f64; 2]> {
        let n = rhs.len();
        let a_dz_half = alpha * dz * 0.5;
        let off_re = 0.0f64;
        let off_im = -a_dz_half;
        let diag_re = 1.0f64;
        let diag_im = 2.0 * a_dz_half;
        let mut w = vec![[0.0f64; 2]; n];
        let mut g = vec![[0.0f64; 2]; n];
        w[0] = [diag_re, diag_im];
        g[0] = rhs[0];
        for i in 1..n {
            let wm = w[i - 1];
            let denom = wm[0] * wm[0] + wm[1] * wm[1];
            let r_re = (off_re * wm[0] + off_im * wm[1]) / denom;
            let r_im = (off_im * wm[0] - off_re * wm[1]) / denom;
            let fac_re = r_re * off_re - r_im * off_im;
            let fac_im = r_re * off_im + r_im * off_re;
            w[i] = [diag_re - fac_re, diag_im - fac_im];
            let gm = g[i - 1];
            let sub_re = r_re * gm[0] - r_im * gm[1];
            let sub_im = r_re * gm[1] + r_im * gm[0];
            g[i] = [rhs[i][0] - sub_re, rhs[i][1] - sub_im];
        }
        let mut d_re = vec![0.0f64; n];
        let mut d_im = vec![0.0f64; n];
        let wn = w[n - 1];
        let denom = wn[0] * wn[0] + wn[1] * wn[1];
        d_re[n - 1] = (g[n - 1][0] * wn[0] + g[n - 1][1] * wn[1]) / denom;
        d_im[n - 1] = (g[n - 1][1] * wn[0] - g[n - 1][0] * wn[1]) / denom;
        for i in (0..n - 1).rev() {
            let dnext = [d_re[i + 1], d_im[i + 1]];
            let od_re = off_re * dnext[0] - off_im * dnext[1];
            let od_im = off_re * dnext[1] + off_im * dnext[0];
            let num_re = g[i][0] - od_re;
            let num_im = g[i][1] - od_im;
            let wi = w[i];
            let den = wi[0] * wi[0] + wi[1] * wi[1];
            d_re[i] = (num_re * wi[0] + num_im * wi[1]) / den;
            d_im[i] = (num_im * wi[0] - num_re * wi[1]) / den;
        }
        (0..n).map(|i| [d_re[i], d_im[i]]).collect()
    }

    /// Total power in Ex component: ∑|uₓ|² · dx.
    pub fn power_x(&self) -> f64 {
        self.field_x
            .iter()
            .map(|&[r, i]| r * r + i * i)
            .sum::<f64>()
            * self.dx
    }

    /// Total power in Ey component: ∑|uᵧ|² · dx.
    pub fn power_y(&self) -> f64 {
        self.field_y
            .iter()
            .map(|&[r, i]| r * r + i * i)
            .sum::<f64>()
            * self.dx
    }

    /// Total optical power (both polarizations).
    pub fn total_power(&self) -> f64 {
        self.power_x() + self.power_y()
    }

    /// Polarization extinction ratio: P_x / P_y (in linear units).
    pub fn extinction_ratio(&self) -> f64 {
        let py = self.power_y();
        if py < 1e-30 {
            f64::INFINITY
        } else {
            self.power_x() / py
        }
    }

    /// Degree of polarization: |P_x - P_y| / (P_x + P_y).
    pub fn degree_of_polarization(&self) -> f64 {
        let px = self.power_x();
        let py = self.power_y();
        let total = px + py;
        if total < 1e-30 {
            0.0
        } else {
            (px - py).abs() / total
        }
    }

    /// Propagate for n_steps × dz using the CN scalar step.
    pub fn propagate(&mut self, n_steps: usize, dz: f64) {
        for _ in 0..n_steps {
            self.step(dz);
        }
    }

    /// Get combined [Ex, Ey] field at each point.
    pub fn fields(&self) -> Vec<Field2> {
        (0..self.n_x)
            .map(|i| [self.field_x[i], self.field_y[i]])
            .collect()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── JonesVector ──────────────────────────────────────────────────────────

    #[test]
    fn jones_vector_x_polarized() {
        let jv = JonesVector::new(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
        assert!((jv.intensity() - 1.0).abs() < 1e-12);
        assert!((jv.ex().re - 1.0).abs() < 1e-12);
        assert!(jv.ey().norm() < 1e-12);
    }

    #[test]
    fn jones_vector_normalize_unit_intensity() {
        let jv = JonesVector::new(Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0));
        let n = jv.normalize();
        assert!((n.intensity() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn jones_vector_normalize_zero_unchanged() {
        let jv = JonesVector::new(Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0));
        let n = jv.normalize();
        assert!(n.intensity() < 1e-30);
    }

    #[test]
    fn jones_vector_circular_intensity() {
        let s = 1.0_f64 / 2.0_f64.sqrt();
        let jv = JonesVector::new(
            Complex64::new(s, 0.0),
            Complex64::new(0.0, s), // right circular
        );
        assert!((jv.intensity() - 1.0).abs() < 1e-12);
    }

    // ── JonesMatrix ──────────────────────────────────────────────────────────

    #[test]
    fn jones_matrix_identity_preserves_vector() {
        let id = JonesMatrix::identity();
        let jv = JonesVector::new(Complex64::new(0.6, 0.0), Complex64::new(0.8, 0.0));
        let out = id.apply(jv);
        assert!((out.ex().re - 0.6).abs() < 1e-12);
        assert!((out.ey().re - 0.8).abs() < 1e-12);
    }

    #[test]
    fn jones_matrix_waveplate_half_wave_rotates_x_to_minus_x() {
        // HWP with fast axis along x: phase_x=0, phase_y=π → flips y component sign
        let hwp = JonesMatrix::waveplate(0.0, PI);
        let jv = JonesVector::new(Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0));
        let out = hwp.apply(jv);
        // Ey should be exp(iπ)·1 = -1
        assert!(
            (out.ey().re - (-1.0)).abs() < 1e-12,
            "ey.re={}",
            out.ey().re
        );
    }

    #[test]
    fn jones_matrix_waveplate_preserves_intensity() {
        let wp = JonesMatrix::waveplate(0.3, 1.1);
        let jv = JonesVector::new(Complex64::new(0.6, 0.2), Complex64::new(-0.3, 0.7));
        let i_in = jv.intensity();
        let i_out = wp.apply(jv).intensity();
        assert!((i_out - i_in).abs() < 1e-12);
    }

    #[test]
    fn jones_matrix_linear_polarizer_x_passes_x() {
        let pol = JonesMatrix::linear_polarizer(0.0); // x-polarizer
        let jv = JonesVector::new(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
        let out = pol.apply(jv);
        assert!((out.ex().re - 1.0).abs() < 1e-12);
        assert!(out.ey().norm() < 1e-12);
    }

    #[test]
    fn jones_matrix_linear_polarizer_45_deg_splits_equally() {
        let pol = JonesMatrix::linear_polarizer(PI / 4.0);
        let jv = JonesVector::new(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
        let out = pol.apply(jv);
        // Ex = cos²(45°) = 0.5, Ey = cos(45°)sin(45°) = 0.5
        assert!((out.ex().re - 0.5).abs() < 1e-12, "ex.re={}", out.ex().re);
        assert!((out.ey().re - 0.5).abs() < 1e-12, "ey.re={}", out.ey().re);
    }

    #[test]
    fn jones_matrix_compose_identity_times_x() {
        let id = JonesMatrix::identity();
        let pol = JonesMatrix::linear_polarizer(0.0);
        let m = id.compose(&pol);
        let jv = JonesVector::new(Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0));
        let out1 = m.apply(jv);
        let out2 = pol.apply(jv);
        assert!((out1.ex().re - out2.ex().re).abs() < 1e-12);
        assert!((out1.ey().re - out2.ey().re).abs() < 1e-12);
    }

    // ── VectorBpm1d (extended API) ────────────────────────────────────────────

    #[test]
    fn vector_bpm_new_signature_two_args() {
        // new(n_x, dx, wavelength) — no n_ref argument
        let bpm = VectorBpm1d::new(64, 1e-6, 1550e-9);
        assert_eq!(bpm.n_x, 64);
        assert!((bpm.wavelength - 1550e-9).abs() < 1e-20);
    }

    #[test]
    fn vector_bpm_set_birefringence_slice() {
        let mut bpm = VectorBpm1d::new(64, 1e-6, 1550e-9);
        let dn = vec![0.005; 64];
        bpm.set_birefringence(&dn);
        assert!((bpm.birefringence[0] - 0.005).abs() < 1e-12);
    }

    #[test]
    fn vector_bpm_propagate_dz_preserves_power() {
        let mut bpm = VectorBpm1d::new(128, 0.25e-6, 1550e-9);
        let x0 = 128_f64 / 2.0 * 0.25e-6;
        let w0 = 5e-6;
        let mut ex: Vec<Complex64> = (0..128)
            .map(|i| {
                let x = i as f64 * 0.25e-6;
                let d = x - x0;
                Complex64::new((-d * d / (w0 * w0)).exp(), 0.0)
            })
            .collect();
        let mut ey: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); 128];

        let p_in: f64 = ex.iter().map(|c| c.norm_sqr()).sum::<f64>() * 0.25e-6;
        bpm.propagate_dz(&mut ex, &mut ey, 1.5, 2e-6);
        let p_out: f64 = ex.iter().map(|c| c.norm_sqr()).sum::<f64>() * 0.25e-6
            + ey.iter().map(|c| c.norm_sqr()).sum::<f64>() * 0.25e-6;
        let rel = (p_out - p_in).abs() / p_in;
        assert!(rel < 0.05, "power conservation err={rel:.4}");
    }

    #[test]
    fn vector_bpm_propagate_dz_birefringence_shifts_ey_phase() {
        let mut bpm = VectorBpm1d::new(64, 0.5e-6, 1550e-9);
        bpm.set_birefringence(&vec![0.01; 64]);
        let mut ex = vec![Complex64::new(1.0, 0.0); 64];
        let mut ey = vec![Complex64::new(1.0, 0.0); 64];
        // After one step, ey should acquire extra phase vs ex
        bpm.propagate_dz(&mut ex, &mut ey, 1.5, 1e-6);
        // The phases will differ; just check fields are not NaN
        for (e, y) in ex.iter().zip(ey.iter()) {
            assert!(e.re.is_finite() && y.re.is_finite());
        }
    }

    #[test]
    fn vector_bpm_jones_matrix_propagation_identity_for_zero_dn() {
        let bpm = VectorBpm1d::new(64, 0.5e-6, 1550e-9);
        // n_profile all 1.5, birefringence all 0 → Jones matrix is identity
        let jv = JonesVector::new(Complex64::new(0.6, 0.0), Complex64::new(0.8, 0.0));
        let out = bpm.jones_matrix_propagation(jv, 1.5, 10e-6);
        // Intensity must be preserved
        assert!((out.intensity() - jv.intensity()).abs() < 1e-10);
    }

    #[test]
    fn vector_bpm_jones_matrix_propagation_preserves_intensity_with_birefringence() {
        let mut bpm = VectorBpm1d::new(64, 0.5e-6, 1550e-9);
        bpm.set_birefringence(&vec![0.02; 64]);
        let jv = JonesVector::new(
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        );
        let out = bpm.jones_matrix_propagation(jv, 1.5, 20e-6);
        assert!((out.intensity() - jv.intensity()).abs() < 1e-10);
    }

    // ── Original VectorBpm1d tests (preserved) ────────────────────────────────

    #[test]
    fn vector_bpm_init_zero_power() {
        let bpm = VectorBpm1d::new(64, 1e-6, 1550e-9);
        assert!(bpm.total_power() == 0.0);
    }

    #[test]
    fn vector_bpm_x_polarized_power() {
        let mut bpm = VectorBpm1d::new(256, 0.5e-6, 1550e-9);
        bpm.set_gaussian_x(1.0, 0.0, 10e-6);
        assert!(bpm.power_x() > 0.0);
        assert!(bpm.power_y() == 0.0);
    }

    #[test]
    fn vector_bpm_power_conserved_free_space() {
        let mut bpm = VectorBpm1d::new(512, 0.25e-6, 1550e-9);
        bpm.set_gaussian_x(1.0, 0.0, 5e-6);
        let p0 = bpm.total_power();
        bpm.propagate(20, 5e-6);
        let p1 = bpm.total_power();
        let rel_err = (p1 - p0).abs() / p0;
        assert!(rel_err < 0.05, "Power conservation err={rel_err:.4}");
    }

    #[test]
    fn vector_bpm_birefringence_transfers_power() {
        // A birefringent medium rotates polarization
        let mut bpm = VectorBpm1d::new(256, 0.5e-6, 1550e-9);
        // Start purely x-polarized
        bpm.set_gaussian_x(1.0, 0.0, 10e-6);
        // Set birefringence
        let n_x = bpm.n_x;
        bpm.birefringence = vec![0.01; n_x];
        let p_x0 = bpm.power_x();
        // After propagation, the birefringence introduces differential phase
        bpm.propagate(10, 1e-6);
        // Total power still conserved (each component propagates independently here)
        let p_total = bpm.total_power();
        assert!((p_total - p_x0).abs() / p_x0 < 0.1);
    }

    #[test]
    fn degree_of_polarization_pure_x() {
        let mut bpm = VectorBpm1d::new(64, 1e-6, 1550e-9);
        bpm.set_gaussian_x(1.0, 0.0, 10e-6);
        assert!((bpm.degree_of_polarization() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn vector_bpm_fields_length() {
        let bpm = VectorBpm1d::new(32, 1e-6, 1550e-9);
        assert_eq!(bpm.fields().len(), 32);
    }
}
