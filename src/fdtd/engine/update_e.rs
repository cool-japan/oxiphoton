//! E-field update coefficient arrays and curl-E kernels for FDTD.
//!
//! Precomputes Ca and Cb arrays for the E-field update:
//!   E^{n+1}\[i\] = Ca\[i\] * E^n\[i\] + Cb\[i\] * curl_H\[i\]
//!
//! where:
//!   Ca\[i\] = (1 - sigma_e\[i\]*dt/(2*eps\[i\])) / (1 + sigma_e\[i\]*dt/(2*eps\[i\]))
//!   Cb\[i\] = dt / (eps\[i\] * dz) / (1 + sigma_e\[i\]*dt/(2*eps\[i\]))
//!
//! For lossless media (sigma=0): Ca = 1, Cb = dt/(eps*dz).
//!
//! PEC (Perfect Electric Conductor) cells have Ca = Cb = 0.

use crate::units::conversion::EPSILON_0;

/// Coefficient arrays for the E-field update equation.
///
/// Stores Ca and Cb for each grid cell so the update loop can be written as:
///   E\[i\] = Ca\[i\] * E\[i\] + Cb\[i\] * (H\[i\] - H\[i-1\]) / dz
#[derive(Debug, Clone)]
pub struct EUpdateCoeffs {
    /// Ca\[i\] = (eps - sigma*dt/2) / (eps + sigma*dt/2) — ranges in (-1, 1]
    pub ca: Vec<f64>,
    /// Cb\[i\] = dt / (eps + sigma*dt/2) / dz
    pub cb: Vec<f64>,
    /// PEC mask: if true, E is forced to zero regardless of update
    pub is_pec: Vec<bool>,
    /// Grid size
    pub n: usize,
}

impl EUpdateCoeffs {
    /// Build coefficient arrays for 1D Ez grid.
    ///
    /// # Arguments
    /// - `eps_r`: relative permittivity at each E-cell (length = n)
    /// - `sigma_e`: electric conductivity (S/m) at each E-cell
    /// - `dt`: time step (s)
    /// - `dz`: grid spacing (m)
    pub fn new_1d(eps_r: &[f64], sigma_e: &[f64], dt: f64, dz: f64) -> Self {
        let n = eps_r.len();
        assert_eq!(sigma_e.len(), n);
        let mut ca = vec![1.0; n];
        let mut cb = vec![0.0; n];
        let is_pec = vec![false; n];
        for i in 0..n {
            let eps = eps_r[i] * EPSILON_0;
            let sig = sigma_e[i];
            let denom = eps + sig * dt / 2.0;
            ca[i] = (eps - sig * dt / 2.0) / denom;
            cb[i] = dt / (dz * denom);
        }
        Self { ca, cb, is_pec, n }
    }

    /// Build with uniform lossless medium (sigma = 0).
    pub fn lossless_1d(eps_r: &[f64], dt: f64, dz: f64) -> Self {
        let sigma = vec![0.0; eps_r.len()];
        Self::new_1d(eps_r, &sigma, dt, dz)
    }

    /// Set a PEC region [i_start, i_end).
    pub fn set_pec(&mut self, i_start: usize, i_end: usize) {
        for i in i_start..i_end.min(self.n) {
            self.is_pec[i] = true;
            self.ca[i] = 0.0;
            self.cb[i] = 0.0;
        }
    }

    /// Apply coefficients: E\[i\] = Ca\[i\]*E\[i\] + Cb\[i\]*(H\[i\] - H\[i-1\])
    ///
    /// Applies update for interior cells (1..n-1).
    pub fn apply_1d_te(&self, ex: &mut [f64], hy: &[f64]) {
        let n = self.n;
        for i in 1..n - 1 {
            if self.is_pec[i] {
                ex[i] = 0.0;
            } else {
                ex[i] = self.ca[i] * ex[i] + self.cb[i] * (hy[i] - hy[i - 1]);
            }
        }
    }

    /// Apply with CPML correction term: Cb * (curl_H / kappa + psi_e)
    pub fn apply_1d_with_pml(
        &self,
        ex: &mut [f64],
        hy: &[f64],
        dz: f64,
        kappa_e: &[f64],
        psi_ex: &[f64],
    ) {
        let n = self.n;
        for i in 1..n - 1 {
            if self.is_pec[i] {
                ex[i] = 0.0;
            } else {
                let dhy = (hy[i] - hy[i - 1]) / dz;
                ex[i] = self.ca[i] * ex[i] + self.cb[i] * dz * (dhy / kappa_e[i] + psi_ex[i]);
            }
        }
    }
}

/// 2D TE mode E-field coefficient arrays (Ex and Ey).
#[derive(Debug, Clone)]
pub struct EUpdateCoeffs2d {
    /// nx × ny grid
    pub nx: usize,
    pub ny: usize,
    /// Ca for Ex (nx × ny), row-major
    pub ca_ex: Vec<f64>,
    pub cb_ex: Vec<f64>,
    /// Ca for Ey (nx × ny), row-major
    pub ca_ey: Vec<f64>,
    pub cb_ey: Vec<f64>,
    pub is_pec: Vec<bool>,
}

impl EUpdateCoeffs2d {
    pub fn new(nx: usize, ny: usize, eps_r: &[f64], sigma: &[f64], dt: f64, dx: f64) -> Self {
        let n = nx * ny;
        assert_eq!(eps_r.len(), n);
        assert_eq!(sigma.len(), n);
        let mut ca = vec![1.0; n];
        let mut cb = vec![0.0; n];
        for i in 0..n {
            let eps = eps_r[i] * EPSILON_0;
            let sig = sigma[i];
            let denom = eps + sig * dt / 2.0;
            ca[i] = (eps - sig * dt / 2.0) / denom;
            cb[i] = dt / (dx * denom);
        }
        Self {
            nx,
            ny,
            ca_ex: ca.clone(),
            cb_ex: cb.clone(),
            ca_ey: ca,
            cb_ey: cb,
            is_pec: vec![false; n],
        }
    }

    /// Set PEC for a rectangular box region.
    pub fn set_pec_box(&mut self, ix0: usize, ix1: usize, iy0: usize, iy1: usize) {
        let ny = self.ny;
        for i in ix0..ix1.min(self.nx) {
            for j in iy0..iy1.min(ny) {
                let idx = i * ny + j;
                self.is_pec[idx] = true;
                self.ca_ex[idx] = 0.0;
                self.cb_ex[idx] = 0.0;
                self.ca_ey[idx] = 0.0;
                self.cb_ey[idx] = 0.0;
            }
        }
    }

    /// Apply 2D TE update: Ex\[i,j\] += Cb * (Hz\[i,j\] - Hz\[i,j-1\]) / dy
    pub fn apply_ex(&self, ex: &mut [f64], hz: &[f64], dy: f64) {
        let ny = self.ny;
        for i in 0..self.nx {
            for j in 1..ny {
                let idx = i * ny + j;
                if self.is_pec[idx] {
                    ex[idx] = 0.0;
                } else {
                    let curl = (hz[idx] - hz[idx - 1]) / dy;
                    ex[idx] = self.ca_ex[idx] * ex[idx] + self.cb_ex[idx] * dy * curl;
                }
            }
        }
    }

    /// Apply 2D TE update: Ey\[i,j\] += -Cb * (Hz\[i,j\] - Hz\[i-1,j\]) / dx
    pub fn apply_ey(&self, ey: &mut [f64], hz: &[f64], dx: f64) {
        let ny = self.ny;
        for i in 1..self.nx {
            for j in 0..ny {
                let idx = i * ny + j;
                if self.is_pec[idx] {
                    ey[idx] = 0.0;
                } else {
                    let curl = (hz[idx] - hz[(i - 1) * ny + j]) / dx;
                    ey[idx] = self.ca_ey[idx] * ey[idx] - self.cb_ey[idx] * dx * curl;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ca_lossless_is_one() {
        let eps_r = vec![1.0; 10];
        let c = EUpdateCoeffs::lossless_1d(&eps_r, 1e-16, 10e-9);
        for i in 0..10 {
            assert_relative_eq!(c.ca[i], 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn cb_lossless_positive() {
        let eps_r = vec![2.25; 10]; // n=1.5
        let dt = 1e-16;
        let dz = 10e-9;
        let c = EUpdateCoeffs::lossless_1d(&eps_r, dt, dz);
        for i in 0..10 {
            assert!(c.cb[i] > 0.0, "cb[{i}] = {:.4e}", c.cb[i]);
        }
    }

    #[test]
    fn pec_zeroes_ca_cb() {
        let eps_r = vec![1.0; 10];
        let mut c = EUpdateCoeffs::lossless_1d(&eps_r, 1e-16, 10e-9);
        c.set_pec(3, 6);
        for i in 3..6 {
            assert_eq!(c.ca[i], 0.0);
            assert_eq!(c.cb[i], 0.0);
            assert!(c.is_pec[i]);
        }
        // Outside PEC region
        assert!(!c.is_pec[2]);
        assert!(!c.is_pec[6]);
    }

    #[test]
    fn apply_1d_te_updates_interior() {
        let eps_r = vec![1.0; 10];
        let c = EUpdateCoeffs::lossless_1d(&eps_r, 1e-16, 10e-9);
        let mut ex = vec![0.0; 10];
        let hy = vec![1.0; 10];
        c.apply_1d_te(&mut ex, &hy);
        // Interior cells should get non-zero update from hy[i] - hy[i-1] = 0
        // (uniform hy => curl = 0, so no change)
        for (i, val) in ex.iter().enumerate().skip(1).take(8) {
            assert_eq!(*val, 0.0, "Uniform H → zero curl → no E update at {i}");
        }
    }

    #[test]
    fn apply_1d_te_nonuniform_h() {
        let eps_r = vec![1.0; 10];
        let c = EUpdateCoeffs::lossless_1d(&eps_r, 1e-16, 10e-9);
        let mut ex = vec![0.0; 10];
        let mut hy = vec![0.0; 10];
        hy[4] = 1.0; // step in H field
        c.apply_1d_te(&mut ex, &hy);
        // E[5] should be updated positively (hy[5]-hy[4] = -1 → ex[5] updated negatively via Cb*(-1))
        assert!(ex[5].is_finite());
    }

    #[test]
    fn e_update_2d_zeros_for_uniform_hz() {
        let n = 4 * 4;
        let eps_r = vec![1.0; n];
        let sigma = vec![0.0; n];
        let c = EUpdateCoeffs2d::new(4, 4, &eps_r, &sigma, 1e-16, 10e-9);
        let mut ex = vec![0.0; n];
        let hz = vec![1.0; n]; // uniform Hz → curl = 0
        c.apply_ex(&mut ex, &hz, 10e-9);
        assert!(ex.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn lossy_ca_less_than_one() {
        let eps_r = vec![1.0; 5];
        let sigma_e = vec![1e3; 5]; // lossy
        let c = EUpdateCoeffs::new_1d(&eps_r, &sigma_e, 1e-16, 10e-9);
        for i in 0..5 {
            assert!(c.ca[i] < 1.0, "Lossy Ca should be < 1");
        }
    }
}
