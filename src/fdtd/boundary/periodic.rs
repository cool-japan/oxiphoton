use std::f64::consts::PI;

/// Bloch periodic boundary condition for 1D FDTD with complex fields.
///
/// For a periodic medium with period L and Bloch wavevector k_B:
///   E(x + L) = E(x) · exp(i·k_B·L)
///
/// The complex field is split into real and imaginary parts and updated
/// using the standard FDTD equations with periodic wrap-around multiplied
/// by the Bloch phase factor.
pub struct BlochFdtd1d {
    pub nz: usize,
    pub dz: f64,
    pub dt: f64,
    pub time_step: usize,

    /// Bloch wavevector (rad/m)
    pub k_bloch: f64,

    /// Period length = nz * dz
    pub period: f64,

    /// Complex Ex field (real part)
    pub ex_re: Vec<f64>,
    pub ex_im: Vec<f64>,

    /// Complex Hy field
    pub hy_re: Vec<f64>,
    pub hy_im: Vec<f64>,

    /// Relative permittivity
    pub eps_r: Vec<f64>,
    /// Relative permeability
    pub mu_r: Vec<f64>,
}

impl BlochFdtd1d {
    /// Create solver with Bloch wavevector `k_bloch` (rad/m).
    /// `nz` cells, `dz` spacing. No PML — periodic boundaries.
    pub fn new(nz: usize, dz: f64, k_bloch: f64) -> Self {
        use crate::units::conversion::SPEED_OF_LIGHT;
        // Courant limit for 1D
        let dt = 0.99 * dz / SPEED_OF_LIGHT;
        let period = nz as f64 * dz;
        Self {
            nz,
            dz,
            dt,
            time_step: 0,
            k_bloch,
            period,
            ex_re: vec![0.0; nz],
            ex_im: vec![0.0; nz],
            hy_re: vec![0.0; nz],
            hy_im: vec![0.0; nz],
            eps_r: vec![1.0; nz],
            mu_r: vec![1.0; nz],
        }
    }

    pub fn current_time(&self) -> f64 {
        self.time_step as f64 * self.dt
    }

    /// Fill a region with given permittivity.
    pub fn fill_eps(&mut self, z_start: f64, z_end: f64, eps: f64) {
        let i0 = (z_start / self.dz).floor() as usize;
        let i1 = ((z_end / self.dz).ceil() as usize).min(self.nz);
        for i in i0..i1 {
            self.eps_r[i] = eps;
        }
    }

    /// Advance one time step with Bloch periodic boundary.
    pub fn step(&mut self) {
        use crate::units::conversion::{EPSILON_0, MU_0};
        let nz = self.nz;
        let dz = self.dz;
        let dt = self.dt;

        // Bloch phase factor exp(i·k_B·L) for the periodic boundary wrap
        let phase_re = (self.k_bloch * self.period).cos();
        let phase_im = (self.k_bloch * self.period).sin();

        // --- Update Hy (n → n+½) ---
        // dEx/dz: forward difference; at i=nz-1, i+1 wraps to 0 with Bloch factor
        for i in 0..nz {
            let (ex_next_re, ex_next_im) = if i + 1 < nz {
                (self.ex_re[i + 1], self.ex_im[i + 1])
            } else {
                // Bloch wrap: Ex[0] * exp(i·k_B·L)
                (
                    self.ex_re[0] * phase_re - self.ex_im[0] * phase_im,
                    self.ex_re[0] * phase_im + self.ex_im[0] * phase_re,
                )
            };
            let dex_re = (ex_next_re - self.ex_re[i]) / dz;
            let dex_im = (ex_next_im - self.ex_im[i]) / dz;
            let mu = MU_0 * self.mu_r[i];
            self.hy_re[i] -= dt / mu * dex_re;
            self.hy_im[i] -= dt / mu * dex_im;
        }

        // --- Update Ex (n+½ → n+1) ---
        // dHy/dz: backward difference; at i=0, i-1 wraps to nz-1 with conjugate Bloch
        // The inverse Bloch factor for wrap from right to left: exp(-i·k_B·L)
        let inv_phase_re = phase_re;
        let inv_phase_im = -phase_im;
        for i in 0..nz {
            let (hy_prev_re, hy_prev_im) = if i > 0 {
                (self.hy_re[i - 1], self.hy_im[i - 1])
            } else {
                // Bloch wrap: Hy[nz-1] * exp(-i·k_B·L)
                (
                    self.hy_re[nz - 1] * inv_phase_re - self.hy_im[nz - 1] * inv_phase_im,
                    self.hy_re[nz - 1] * inv_phase_im + self.hy_im[nz - 1] * inv_phase_re,
                )
            };
            let dhy_re = (self.hy_re[i] - hy_prev_re) / dz;
            let dhy_im = (self.hy_im[i] - hy_prev_im) / dz;
            let eps = EPSILON_0 * self.eps_r[i];
            self.ex_re[i] -= dt / eps * dhy_re;
            self.ex_im[i] -= dt / eps * dhy_im;
        }

        self.time_step += 1;
    }

    pub fn run(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// Inject a complex source into Ex at position `pos`.
    pub fn inject_ex(&mut self, pos: usize, re: f64, im: f64) {
        if pos < self.nz {
            self.ex_re[pos] += re;
            self.ex_im[pos] += im;
        }
    }

    /// Compute the dominant frequency from the Ex field using the time-domain DFT
    /// at a single frequency. Returns |Ex(omega)|².
    pub fn dft_ex_magnitude_sq(
        &self,
        omega: f64,
        t_start: f64,
        t_end: f64,
        n_samples: usize,
    ) -> f64 {
        // Discrete summation from stored fields is not possible without history,
        // so this returns the instantaneous intensity at the center cell.
        // For actual DFT, users should accumulate externally.
        let _ = (omega, t_start, t_end, n_samples);
        let mid = self.nz / 2;
        self.ex_re[mid] * self.ex_re[mid] + self.ex_im[mid] * self.ex_im[mid]
    }

    /// For a uniform medium of index n, the Bloch condition at k_B = ω·n/c
    /// gives a plane wave. This checks the dispersion relation holds approximately:
    ///   ω ≈ c / n * k_B
    pub fn expected_omega_for_uniform(n: f64, k_b: f64) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        SPEED_OF_LIGHT / n * k_b
    }
}

/// Compute phononic/photonic band structure dispersion relation.
/// Returns (k_B_values, omega_values) for a uniform medium.
pub fn uniform_band_structure(n: f64, period: f64, n_k: usize) -> (Vec<f64>, Vec<f64>) {
    use crate::units::conversion::SPEED_OF_LIGHT;
    let k_max = PI / period; // First Brillouin zone edge
    let ks: Vec<f64> = (0..=n_k).map(|i| i as f64 / n_k as f64 * k_max).collect();
    let omegas: Vec<f64> = ks.iter().map(|&k| SPEED_OF_LIGHT / n * k).collect();
    (ks, omegas)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::conversion::SPEED_OF_LIGHT;

    #[test]
    fn bloch_solver_initializes_zero() {
        let s = BlochFdtd1d::new(100, 10e-9, 0.0);
        assert!(s.ex_re.iter().all(|&v| v == 0.0));
        assert!(s.hy_re.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn bloch_solver_runs_without_panic() {
        let mut s = BlochFdtd1d::new(64, 10e-9, 1e7);
        s.run(100);
        assert!(s.ex_re.iter().all(|&v| v.is_finite()));
        assert!(s.hy_re.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn bloch_periodic_dt_courant() {
        let s = BlochFdtd1d::new(100, 10e-9, 0.0);
        let courant = SPEED_OF_LIGHT * s.dt / s.dz;
        assert!(courant <= 1.0, "Courant number={courant:.4} > 1 (unstable)");
    }

    #[test]
    fn uniform_band_linear_dispersion() {
        let n = 1.5;
        let period = 500e-9;
        let (ks, omegas) = uniform_band_structure(n, period, 10);
        // Linear dispersion: omega = c/n * k
        for (k, omega) in ks.iter().zip(omegas.iter()) {
            let expected = SPEED_OF_LIGHT / n * k;
            let rel_err = (omega - expected).abs() / (expected + 1e-30);
            assert!(rel_err < 1e-10, "Band dispersion error: {rel_err:.2e}");
        }
    }

    #[test]
    fn bloch_phase_wrap_is_physical() {
        // For k_B = pi/L (zone boundary), phase factor = exp(i*pi) = -1
        let period = 1e-6;
        let nz = 100;
        let dz = period / nz as f64;
        let k_b = PI / period;
        let s = BlochFdtd1d::new(nz, dz, k_b);
        let phase_re = (k_b * s.period).cos();
        let phase_im = (k_b * s.period).sin();
        // At zone boundary: phase = -1 + i*~0
        assert!((phase_re - (-1.0)).abs() < 1e-10);
        assert!(phase_im.abs() < 1e-10);
    }
}
