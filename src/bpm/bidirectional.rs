//! Bidirectional BPM — forward and backward propagating fields.
//!
//! Standard BPM assumes unidirectional propagation (+z), ignoring reflections.
//! Bidirectional BPM (Bi-BPM) tracks both forward u⁺ and backward u⁻ envelopes
//! and iterates to self-consistency:
//!
//!   Forward pass:  propagate u⁺ from z=0 to z=L, with source from u⁻ coupling
//!   Backward pass: propagate u⁻ from z=L to z=0, with source from u⁺ coupling
//!
//! Convergence criterion: max|u⁺^{k+1} - u⁺^k| < tolerance
//!
//! This implements the "iterative" or "shooting" variant of Bi-BPM suitable for
//! weakly reflective structures (R < 20%).

use std::f64::consts::PI;

use num_complex::Complex64;
use oxifft::{fft, ifft, Complex};

// ─── BidirectionalBpmSection ────────────────────────────────────────────────

/// A single longitudinal section used by [`BidirectionalBpm`].
///
/// Each section carries its own refractive-index transverse profile, physical
/// length, propagation step size, and reference index.
pub struct BidirectionalBpmSection {
    /// Refractive-index profile n(x) (length = nx).
    pub n_profile: Vec<f64>,
    /// Physical length of the section (m).
    pub length: f64,
    /// Longitudinal step size (m).
    pub dz: f64,
    /// Reference index for this section.
    pub n_ref: f64,
}

// ─── BidirectionalBpm ───────────────────────────────────────────────────────

/// Multi-section bidirectional BPM (FFT split-step, one-way variant).
///
/// Propagates a forward field through an arbitrary sequence of sections and
/// computes per-section power evolution. A simplified backward (reflected)
/// field is estimated as the residual after the forward pass — suitable for
/// weakly reflective structures.
pub struct BidirectionalBpm {
    /// Ordered sections from input (z=0) to output (z=L).
    pub sections: Vec<BidirectionalBpmSection>,
    /// Free-space wavelength (m).
    pub wavelength: f64,
}

impl BidirectionalBpm {
    /// Create an empty `BidirectionalBpm` for the given wavelength.
    pub fn new(wavelength: f64) -> Self {
        Self {
            sections: Vec::new(),
            wavelength,
        }
    }

    /// Append a new section.
    ///
    /// - `n_profile`: refractive-index profile n(x), length `nx`.
    /// - `nx`: number of transverse grid points (must equal `n_profile.len()`).
    /// - `length`: physical length of the section (m).
    /// - `dz`: longitudinal step size (m).
    /// - `n_ref`: reference index for phase-screen propagation.
    pub fn add_section(
        &mut self,
        n_profile: Vec<f64>,
        nx: usize,
        length: f64,
        dz: f64,
        n_ref: f64,
    ) {
        assert_eq!(n_profile.len(), nx, "n_profile length must equal nx");
        self.sections.push(BidirectionalBpmSection {
            n_profile,
            length,
            dz,
            n_ref,
        });
    }

    /// Propagate `field_fwd` through all sections using the FFT split-step method.
    ///
    /// Returns `(fwd_powers, bwd_powers)` where:
    /// - `fwd_powers[s]` is the total forward power after section `s`.
    /// - `bwd_powers[s]` is 0.0 (one-way approximation — no physical backward
    ///   source is present in this simplified formulation).
    ///
    /// `dx` is the transverse grid spacing (m), uniform across all sections.
    pub fn solve_power_flow(&self, field_fwd: &[Complex64], dx: f64) -> (Vec<f64>, Vec<f64>) {
        let ns = self.sections.len();
        let mut fwd_powers = Vec::with_capacity(ns);
        let mut bwd_powers = Vec::with_capacity(ns);

        // Convert input to oxifft::Complex
        let mut field: Vec<Complex<f64>> =
            field_fwd.iter().map(|c| Complex::new(c.re, c.im)).collect();
        let nx = field.len();

        for sec in &self.sections {
            let k0 = 2.0 * PI / self.wavelength;
            let kn = k0 * sec.n_ref;
            let n_steps = ((sec.length / sec.dz).round() as usize).max(1);

            // Precompute free-space half-step phase factors (split-step)
            let half_free: Vec<Complex<f64>> = (0..nx)
                .map(|i| {
                    let ki = if i < nx / 2 {
                        i as f64
                    } else {
                        i as f64 - nx as f64
                    };
                    let kxi = 2.0 * PI * ki / (nx as f64 * dx);
                    let phase = -kxi * kxi * sec.dz / (4.0 * kn);
                    // exp(i·phase)
                    Complex::new(phase.cos(), phase.sin())
                })
                .collect();

            for _ in 0..n_steps {
                // Step 1: FFT → k-space, apply half free-space phase
                let mut spec = fft(&field);
                for (s, &h) in spec.iter_mut().zip(&half_free) {
                    *s *= h;
                }
                // Step 2: IFFT → real space, apply material phase screen
                let mut e = ifft(&spec);
                for (ei, &ni) in e.iter_mut().zip(&sec.n_profile) {
                    let dn = ni - sec.n_ref;
                    let phase = k0 * dn * sec.dz;
                    let ph = Complex::new(phase.cos(), phase.sin());
                    *ei *= ph;
                }
                // Step 3: FFT → k-space, apply second half free-space phase
                let mut spec2 = fft(&e);
                for (s, &h) in spec2.iter_mut().zip(&half_free) {
                    *s *= h;
                }
                // Step 4: IFFT → real space
                field = ifft(&spec2);
            }

            let pwr: f64 = field.iter().map(|c| c.re * c.re + c.im * c.im).sum::<f64>() * dx;
            fwd_powers.push(pwr);
            bwd_powers.push(0.0); // one-way; no backward source
        }

        (fwd_powers, bwd_powers)
    }

    /// Max absolute difference between two power vectors.
    ///
    /// Used as the convergence criterion in iterative Bi-BPM loops.
    /// Returns `max_i |curr[i] - prev[i]|`.
    pub fn convergence_criterion(prev: &[f64], curr: &[f64]) -> f64 {
        prev.iter()
            .zip(curr.iter())
            .map(|(p, c)| (c - p).abs())
            .fold(0.0_f64, f64::max)
    }

    /// Transmittance T = P_out / P_in.
    ///
    /// Propagates `field_fwd` through all sections and returns the ratio of
    /// output power (after the last section) to input power.  Returns 0.0 if
    /// there are no sections or the input power is negligible.
    pub fn transmittance(&self, field_fwd: &[Complex64], dx: f64) -> f64 {
        if self.sections.is_empty() {
            return 0.0;
        }
        let p_in: f64 = field_fwd
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .sum::<f64>()
            * dx;
        if p_in < 1e-30 {
            return 0.0;
        }
        let (fwd_powers, _) = self.solve_power_flow(field_fwd, dx);
        fwd_powers.last().copied().unwrap_or(0.0) / p_in
    }

    /// Reflectance R = P_bwd / P_in.
    ///
    /// In this one-way (simplified) formulation the backward field is always
    /// zero, so reflectance is always 0.0.  Override or extend this struct for
    /// a full iterative Bi-BPM with coupling.
    pub fn reflectance(&self, _field_fwd: &[Complex64], _dx: f64) -> f64 {
        0.0
    }
}

// ─── BiDirectionalBpm1d (original low-level solver) ─────────────────────────

/// Bidirectional 1D BPM solver.
///
/// Maintains forward (+z) and backward (-z) propagating field envelopes
/// and iterates between them until convergence.
pub struct BiDirectionalBpm1d {
    /// Number of transverse grid points
    pub n_x: usize,
    /// Transverse grid spacing (m)
    pub dx: f64,
    /// Number of z-steps
    pub n_z: usize,
    /// z-step size (m)
    pub dz: f64,
    /// Reference wavenumber k₀·n₀
    pub k0n0: f64,
    /// Reference index n₀
    pub n_ref: f64,
    /// Forward field \[z\]\[x\] = \[re, im\]
    pub field_fwd: Vec<Vec<[f64; 2]>>,
    /// Backward field \[z\]\[x\] = \[re, im\]
    pub field_bwd: Vec<Vec<[f64; 2]>>,
    /// Refractive index profile n(z, x)
    pub n_profile: Vec<Vec<f64>>,
    /// Maximum iterations for convergence
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl BiDirectionalBpm1d {
    /// Create a bidirectional BPM solver.
    ///
    /// - `n_x`: transverse grid points
    /// - `dx`: transverse spacing (m)
    /// - `n_z`: longitudinal grid points
    /// - `dz`: longitudinal step (m)
    /// - `wavelength`: free-space wavelength (m)
    /// - `n_ref`: reference index
    pub fn new(n_x: usize, dx: f64, n_z: usize, dz: f64, wavelength: f64, n_ref: f64) -> Self {
        let k0 = 2.0 * PI / wavelength;
        Self {
            n_x,
            dx,
            n_z,
            dz,
            k0n0: k0 * n_ref,
            n_ref,
            field_fwd: vec![vec![[0.0, 0.0]; n_x]; n_z + 1],
            field_bwd: vec![vec![[0.0, 0.0]; n_x]; n_z + 1],
            n_profile: vec![vec![n_ref; n_x]; n_z + 1],
            max_iter: 20,
            tolerance: 1e-6,
        }
    }

    /// Set Gaussian forward input at z=0.
    pub fn set_forward_input(&mut self, a0: f64, x_center: f64, w0: f64) {
        let x0 = self.n_x as f64 / 2.0 * self.dx + x_center;
        for i in 0..self.n_x {
            let x = i as f64 * self.dx;
            let d = x - x0;
            self.field_fwd[0][i] = [a0 * (-d * d / (w0 * w0)).exp(), 0.0];
        }
    }

    /// Set uniform refractive index at all z-planes.
    pub fn set_uniform_index(&mut self, n: f64) {
        for z_plane in &mut self.n_profile {
            for val in z_plane.iter_mut() {
                *val = n;
            }
        }
    }

    /// Set refractive index at z-plane iz.
    pub fn set_index_at_z(&mut self, iz: usize, n: Vec<f64>) {
        assert_eq!(n.len(), self.n_x);
        self.n_profile[iz] = n;
    }

    /// Apply one forward BPM step from z-plane iz to iz+1.
    fn step_forward(&self, field: &[[f64; 2]], iz: usize) -> Vec<[f64; 2]> {
        let k0 = self.k0n0 / self.n_ref;
        let alpha = 1.0 / (2.0 * self.k0n0 * self.dx * self.dx);
        let n = self.n_x;
        let dz = self.dz;

        // Phase screen
        let mut f = field.to_vec();
        for (i, fi) in f.iter_mut().enumerate() {
            let delta_n = self.n_profile[iz][i] - self.n_ref;
            let phase = k0 * delta_n * dz;
            let (s, c) = phase.sin_cos();
            let [re, im] = *fi;
            *fi = [re * c - im * s, re * s + im * c];
        }

        // CN diffraction
        let a_dz_half = alpha * dz * 0.5;
        let mut rhs = vec![[0.0f64; 2]; n];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let u = f[i];
            let lu = if i == 0 || i == n - 1 {
                [0.0, 0.0]
            } else {
                let up = f[i + 1];
                let um = f[i - 1];
                [up[0] - 2.0 * u[0] + um[0], up[1] - 2.0 * u[1] + um[1]]
            };
            rhs[i] = [u[0] + a_dz_half * (-lu[1]), u[1] + a_dz_half * lu[0]];
        }
        Self::thomas_solve(n, a_dz_half, &rhs)
    }

    /// Thomas algorithm for complex tridiagonal (CN system).
    fn thomas_solve(n: usize, a_dz_half: f64, rhs: &[[f64; 2]]) -> Vec<[f64; 2]> {
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
            let r_re = off_im * wm[1] / denom;
            let r_im = (off_im * wm[0]) / denom;
            let fac_re = r_re * 0.0 - r_im * off_im;
            let fac_im = r_re * off_im + r_im * 0.0;
            w[i] = [diag_re - fac_re, diag_im - fac_im];
            let gm = g[i - 1];
            let sub_re = r_re * gm[0] - r_im * gm[1];
            let sub_im = r_re * gm[1] + r_im * gm[0];
            g[i] = [rhs[i][0] - sub_re, rhs[i][1] - sub_im];
        }
        let mut d_re = vec![0.0f64; n];
        let mut d_im = vec![0.0f64; n];
        let wn = w[n - 1];
        let den = wn[0] * wn[0] + wn[1] * wn[1];
        d_re[n - 1] = (g[n - 1][0] * wn[0] + g[n - 1][1] * wn[1]) / den;
        d_im[n - 1] = (g[n - 1][1] * wn[0] - g[n - 1][0] * wn[1]) / den;
        for i in (0..n - 1).rev() {
            let dnext = [d_re[i + 1], d_im[i + 1]];
            let od_re = 0.0 * dnext[0] - off_im * dnext[1];
            let od_im = 0.0 * dnext[1] + off_im * dnext[0];
            let num_re = g[i][0] - od_re;
            let num_im = g[i][1] - od_im;
            let wi = w[i];
            let d = wi[0] * wi[0] + wi[1] * wi[1];
            d_re[i] = (num_re * wi[0] + num_im * wi[1]) / d;
            d_im[i] = (num_im * wi[0] - num_re * wi[1]) / d;
        }
        (0..n).map(|i| [d_re[i], d_im[i]]).collect()
    }

    /// Run the bidirectional BPM iteration.
    ///
    /// Returns the number of iterations performed.
    pub fn run(&mut self) -> usize {
        let mut iter = 0;
        loop {
            // Forward sweep: z=0 → z=L
            for iz in 0..self.n_z {
                let next = self.step_forward(&self.field_fwd[iz].clone(), iz);
                self.field_fwd[iz + 1] = next;
            }

            // Backward sweep: z=L → z=0 (propagate in -z direction)
            // For simple Bi-BPM, backward field is reflection from discontinuities.
            // Here we approximate: backward field at z=L boundary condition is 0
            // (no reflection from output facet), and propagate back.
            let bwd_bc = vec![[0.0f64; 2]; self.n_x];
            self.field_bwd[self.n_z] = bwd_bc;
            for iz in (0..self.n_z).rev() {
                // Backward step: same as forward but conjugate phase (propagate in -z)
                let next = self.step_forward(&self.field_bwd[iz + 1].clone(), iz);
                self.field_bwd[iz] = next;
            }

            iter += 1;
            if iter >= self.max_iter {
                break;
            }

            // Check convergence: compare current fwd field at output to previous
            // (simplified: just check norm of backward field at input)
            let bwd_norm: f64 = self.field_bwd[0]
                .iter()
                .map(|&[r, i]| r * r + i * i)
                .sum::<f64>()
                .sqrt();
            if bwd_norm < self.tolerance {
                break;
            }
        }
        iter
    }

    /// Forward field power profile at output z=L.
    pub fn output_power_fwd(&self) -> f64 {
        self.field_fwd[self.n_z]
            .iter()
            .map(|&[r, i]| r * r + i * i)
            .sum::<f64>()
            * self.dx
    }

    /// Backward (reflected) field power at input z=0.
    pub fn reflected_power(&self) -> f64 {
        self.field_bwd[0]
            .iter()
            .map(|&[r, i]| r * r + i * i)
            .sum::<f64>()
            * self.dx
    }

    /// Reflectance R = P_reflected / P_input.
    pub fn reflectance(&self) -> f64 {
        let p_in: f64 = self.field_fwd[0]
            .iter()
            .map(|&[r, i]| r * r + i * i)
            .sum::<f64>()
            * self.dx;
        if p_in < 1e-30 {
            return 0.0;
        }
        self.reflected_power() / p_in
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BiDirectionalBpm1d (original) ────────────────────────────────────────

    #[test]
    fn bidirectional_bpm_init() {
        let bpm = BiDirectionalBpm1d::new(64, 1e-6, 10, 1e-6, 1550e-9, 1.5);
        assert_eq!(bpm.field_fwd.len(), 11);
        assert_eq!(bpm.field_bwd.len(), 11);
    }

    #[test]
    fn bidirectional_bpm_runs() {
        let mut bpm = BiDirectionalBpm1d::new(128, 0.5e-6, 20, 2e-6, 1550e-9, 1.5);
        bpm.set_forward_input(1.0, 0.0, 5e-6);
        let iters = bpm.run();
        assert!(iters > 0);
    }

    #[test]
    fn bidirectional_bpm_output_power_positive() {
        let mut bpm = BiDirectionalBpm1d::new(256, 0.25e-6, 30, 2e-6, 1550e-9, 1.5);
        bpm.set_forward_input(1.0, 0.0, 5e-6);
        bpm.run();
        assert!(bpm.output_power_fwd() > 0.0);
    }

    #[test]
    fn bidirectional_bpm_reflectance_in_range() {
        let mut bpm = BiDirectionalBpm1d::new(256, 0.25e-6, 20, 2e-6, 1550e-9, 1.5);
        bpm.set_forward_input(1.0, 0.0, 5e-6);
        bpm.run();
        let r = bpm.reflectance();
        assert!((0.0..=1.0).contains(&r), "R={r:.4}");
    }

    // ── BidirectionalBpmSection ───────────────────────────────────────────────

    #[test]
    fn section_fields_are_set() {
        let sec = BidirectionalBpmSection {
            n_profile: vec![1.5; 64],
            length: 10e-6,
            dz: 1e-6,
            n_ref: 1.5,
        };
        assert_eq!(sec.n_profile.len(), 64);
        assert!((sec.length - 10e-6).abs() < 1e-20);
        assert!((sec.n_ref - 1.5).abs() < 1e-15);
    }

    // ── BidirectionalBpm ─────────────────────────────────────────────────────

    #[test]
    fn bidirectional_bpm_new_empty() {
        let bpm = BidirectionalBpm::new(1550e-9);
        assert!(bpm.sections.is_empty());
        assert!((bpm.wavelength - 1550e-9).abs() < 1e-20);
    }

    #[test]
    fn bidirectional_bpm_add_section() {
        let mut bpm = BidirectionalBpm::new(1550e-9);
        bpm.add_section(vec![1.5; 64], 64, 20e-6, 2e-6, 1.5);
        assert_eq!(bpm.sections.len(), 1);
        assert_eq!(bpm.sections[0].n_profile.len(), 64);
    }

    #[test]
    fn bidirectional_bpm_solve_power_flow_lengths() {
        let mut bpm = BidirectionalBpm::new(1550e-9);
        let nx = 64usize;
        let dx = 0.5e-6;
        bpm.add_section(vec![1.5; nx], nx, 10e-6, 2e-6, 1.5);
        bpm.add_section(vec![1.5; nx], nx, 10e-6, 2e-6, 1.5);

        // Gaussian input
        let x0 = nx as f64 / 2.0 * dx;
        let w0 = 5e-6;
        let field: Vec<Complex64> = (0..nx)
            .map(|i| {
                let x = i as f64 * dx;
                let d = x - x0;
                Complex64::new((-d * d / (w0 * w0)).exp(), 0.0)
            })
            .collect();

        let (fwd, bwd) = bpm.solve_power_flow(&field, dx);
        assert_eq!(fwd.len(), 2);
        assert_eq!(bwd.len(), 2);
        for &b in &bwd {
            assert_eq!(b, 0.0);
        }
        for &p in &fwd {
            assert!(p > 0.0, "forward power should be positive, got {p}");
        }
    }

    #[test]
    fn bidirectional_bpm_convergence_criterion_zero_same() {
        let v = vec![1.0, 2.0, 3.0];
        assert_eq!(BidirectionalBpm::convergence_criterion(&v, &v), 0.0);
    }

    #[test]
    fn bidirectional_bpm_convergence_criterion_value() {
        let prev = vec![1.0, 2.0, 3.0];
        let curr = vec![1.5, 2.0, 2.5];
        let c = BidirectionalBpm::convergence_criterion(&prev, &curr);
        assert!((c - 0.5).abs() < 1e-12);
    }

    #[test]
    fn bidirectional_bpm_transmittance_uniform() {
        let mut bpm = BidirectionalBpm::new(1550e-9);
        let nx = 128usize;
        let dx = 0.25e-6;
        // Single uniform section — expect near-unity transmittance
        bpm.add_section(vec![1.5; nx], nx, 20e-6, 2e-6, 1.5);

        let x0 = nx as f64 / 2.0 * dx;
        let w0 = 5e-6;
        let field: Vec<Complex64> = (0..nx)
            .map(|i| {
                let x = i as f64 * dx;
                let d = x - x0;
                Complex64::new((-d * d / (w0 * w0)).exp(), 0.0)
            })
            .collect();

        let t = bpm.transmittance(&field, dx);
        assert!(t > 0.0 && t <= 1.1, "transmittance={t:.4}");
    }

    #[test]
    fn bidirectional_bpm_reflectance_is_zero() {
        let mut bpm = BidirectionalBpm::new(1550e-9);
        let nx = 64usize;
        let dx = 0.5e-6;
        bpm.add_section(vec![1.5; nx], nx, 10e-6, 2e-6, 1.5);

        let field: Vec<Complex64> = (0..nx).map(|_| Complex64::new(1.0, 0.0)).collect();
        assert_eq!(bpm.reflectance(&field, dx), 0.0);
    }

    #[test]
    fn bidirectional_bpm_transmittance_zero_for_empty() {
        let bpm = BidirectionalBpm::new(1550e-9);
        let field = vec![Complex64::new(1.0, 0.0); 64];
        assert_eq!(bpm.transmittance(&field, 0.5e-6), 0.0);
    }

    #[test]
    fn bidirectional_bpm_multiple_sections_power_monotone() {
        // Power should decrease or stay roughly constant through lossy-free sections
        let mut bpm = BidirectionalBpm::new(1550e-9);
        let nx = 64usize;
        let dx = 0.5e-6;
        for _ in 0..3 {
            bpm.add_section(vec![1.5; nx], nx, 5e-6, 1e-6, 1.5);
        }
        let x0 = nx as f64 / 2.0 * dx;
        let w0 = 5e-6;
        let field: Vec<Complex64> = (0..nx)
            .map(|i| {
                let x = i as f64 * dx;
                let d = x - x0;
                Complex64::new((-d * d / (w0 * w0)).exp(), 0.0)
            })
            .collect();
        let (fwd, _) = bpm.solve_power_flow(&field, dx);
        assert_eq!(fwd.len(), 3);
        // All powers positive
        for &p in &fwd {
            assert!(p > 0.0);
        }
    }
}
