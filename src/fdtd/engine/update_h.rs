//! H-field update coefficient arrays and curl-H kernels for FDTD.
//!
//! Precomputes Da and Db arrays for the H-field update:
//!   H^{n+1/2}\[i\] = Da\[i\] * H^{n-1/2}\[i\] - Db\[i\] * curl_E\[i\]
//!
//! where:
//!   Da\[i\] = (1 - sigma_m\[i\]*dt/(2*mu\[i\])) / (1 + sigma_m\[i\]*dt/(2*mu\[i\]))
//!   Db\[i\] = dt / (mu\[i\] * dz) / (1 + sigma_m\[i\]*dt/(2*mu\[i\]))
//!
//! For lossless non-magnetic media: Da = 1, Db = dt/(mu_0*dz).
//!
//! PMC (Perfect Magnetic Conductor) cells have Da = Db = 0 and H = 0.

use crate::units::conversion::MU_0;

/// Coefficient arrays for the H-field update equation.
#[derive(Debug, Clone)]
pub struct HUpdateCoeffs {
    /// Da\[i\] = (mu - sigma_m*dt/2) / (mu + sigma_m*dt/2)
    pub da: Vec<f64>,
    /// Db\[i\] = dt / (mu + sigma_m*dt/2) / dz
    pub db: Vec<f64>,
    /// PMC mask: if true, H is forced to zero
    pub is_pmc: Vec<bool>,
    pub n: usize,
}

impl HUpdateCoeffs {
    /// Build H-update coefficients for 1D Hy update.
    ///
    /// # Arguments
    /// - `mu_r`: relative permeability at each H-cell
    /// - `sigma_m`: magnetic loss (Ω/m) at each H-cell
    /// - `dt`: time step (s)
    /// - `dz`: grid spacing (m)
    pub fn new_1d(mu_r: &[f64], sigma_m: &[f64], dt: f64, dz: f64) -> Self {
        let n = mu_r.len();
        assert_eq!(sigma_m.len(), n);
        let mut da = vec![1.0; n];
        let mut db = vec![0.0; n];
        let is_pmc = vec![false; n];
        for i in 0..n {
            let mu = mu_r[i] * MU_0;
            let sig = sigma_m[i];
            let denom = mu + sig * dt / 2.0;
            da[i] = (mu - sig * dt / 2.0) / denom;
            db[i] = dt / (dz * denom);
        }
        Self { da, db, is_pmc, n }
    }

    /// Lossless, non-magnetic (mu_r = 1, sigma_m = 0) coefficients.
    pub fn lossless_1d(n: usize, dt: f64, dz: f64) -> Self {
        let mu_r = vec![1.0; n];
        let sigma_m = vec![0.0; n];
        Self::new_1d(&mu_r, &sigma_m, dt, dz)
    }

    /// Mark a region as PMC (force H = 0).
    pub fn set_pmc(&mut self, i_start: usize, i_end: usize) {
        for i in i_start..i_end.min(self.n) {
            self.is_pmc[i] = true;
            self.da[i] = 0.0;
            self.db[i] = 0.0;
        }
    }

    /// Apply 1D H update: Hy\[i\] = Da\[i\]*Hy\[i\] - Db\[i\]*(Ex\[i+1\] - Ex\[i\])
    ///
    /// Applies to cells 0..n-1 (forward difference in E).
    pub fn apply_1d_tem(&self, hy: &mut [f64], ex: &[f64]) {
        let n = self.n;
        for i in 0..n - 1 {
            if self.is_pmc[i] {
                hy[i] = 0.0;
            } else {
                hy[i] = self.da[i] * hy[i] - self.db[i] * (ex[i + 1] - ex[i]);
            }
        }
    }

    /// Apply 1D H update with CPML correction.
    pub fn apply_1d_with_pml(
        &self,
        hy: &mut [f64],
        ex: &[f64],
        dz: f64,
        kappa_h: &[f64],
        psi_hy: &[f64],
    ) {
        let n = self.n;
        for i in 0..n - 1 {
            if self.is_pmc[i] {
                hy[i] = 0.0;
            } else {
                let dex = (ex[i + 1] - ex[i]) / dz;
                hy[i] = self.da[i] * hy[i] - self.db[i] * dz * (dex / kappa_h[i] + psi_hy[i]);
            }
        }
    }

    /// Subpixel-averaged Db for a material interface within cell i.
    ///
    /// `frac` = fraction of cell filled with material having mu_r2.
    /// Uses harmonic average: mu_avg = 1 / (frac/mu2 + (1-frac)/mu1).
    pub fn subpixel_db(
        &self,
        _i: usize,
        mu_r1: f64,
        mu_r2: f64,
        frac: f64,
        dt: f64,
        dz: f64,
    ) -> f64 {
        let mu_avg = 1.0 / (frac / (mu_r2 * MU_0) + (1.0 - frac) / (mu_r1 * MU_0));
        dt / (dz * mu_avg)
    }
}

/// 2D TE mode H-field coefficients (Hz only).
#[derive(Debug, Clone)]
pub struct HUpdateCoeffs2d {
    pub nx: usize,
    pub ny: usize,
    pub da_hz: Vec<f64>,
    pub db_hz: Vec<f64>,
    pub is_pmc: Vec<bool>,
}

impl HUpdateCoeffs2d {
    pub fn new(nx: usize, ny: usize, mu_r: &[f64], sigma_m: &[f64], dt: f64, dx: f64) -> Self {
        let n = nx * ny;
        assert_eq!(mu_r.len(), n);
        assert_eq!(sigma_m.len(), n);
        let mut da = vec![1.0; n];
        let mut db = vec![0.0; n];
        for i in 0..n {
            let mu = mu_r[i] * MU_0;
            let sig = sigma_m[i];
            let denom = mu + sig * dt / 2.0;
            da[i] = (mu - sig * dt / 2.0) / denom;
            db[i] = dt / (dx * denom);
        }
        Self {
            nx,
            ny,
            da_hz: da,
            db_hz: db,
            is_pmc: vec![false; n],
        }
    }

    /// Lossless non-magnetic.
    pub fn lossless(nx: usize, ny: usize, dt: f64, dx: f64) -> Self {
        let n = nx * ny;
        Self::new(nx, ny, &vec![1.0; n], &vec![0.0; n], dt, dx)
    }

    /// Apply 2D TE Hz update:
    ///   Hz\[i,j\] = Da*Hz\[i,j\] - Db*(Ey\[i+1,j\] - Ey\[i,j\])/dx + Db*(Ex\[i,j+1\] - Ex\[i,j\])/dy
    pub fn apply_hz(&self, hz: &mut [f64], ex: &[f64], ey: &[f64], dx: f64, dy: f64) {
        let ny = self.ny;
        for i in 0..self.nx - 1 {
            for j in 0..ny - 1 {
                let idx = i * ny + j;
                if self.is_pmc[idx] {
                    hz[idx] = 0.0;
                } else {
                    let dey = (ey[(i + 1) * ny + j] - ey[idx]) / dx;
                    let dex = (ex[i * ny + j + 1] - ex[idx]) / dy;
                    hz[idx] = self.da_hz[idx] * hz[idx] - self.db_hz[idx] * (dx * dey - dy * dex);
                }
            }
        }
    }
}

/// 3D H-field coefficient arrays for all three H components.
#[derive(Debug, Clone)]
pub struct HUpdateCoeffs3d {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub da: Vec<f64>,
    pub db_x: Vec<f64>,
    pub db_y: Vec<f64>,
    pub db_z: Vec<f64>,
}

impl HUpdateCoeffs3d {
    /// Lossless vacuum coefficients.
    pub fn lossless(nx: usize, ny: usize, nz: usize, dt: f64, dx: f64, dy: f64, dz: f64) -> Self {
        let n = nx * ny * nz;
        let db = dt / MU_0;
        Self {
            nx,
            ny,
            nz,
            da: vec![1.0; n],
            db_x: vec![db / dx; n],
            db_y: vec![db / dy; n],
            db_z: vec![db / dz; n],
        }
    }

    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.ny + j) * self.nz + k
    }

    /// Apply Hx update: Hx\[i,j,k\] -= Da*(dEz/dy - dEy/dz)
    pub fn apply_hx(&self, hx: &mut [f64], ey: &[f64], ez: &[f64]) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        for i in 0..nx {
            for j in 0..ny - 1 {
                for k in 0..nz - 1 {
                    let idx = self.idx(i, j, k);
                    let dez_dy = ez[self.idx(i, j + 1, k)] - ez[idx];
                    let dey_dz = ey[self.idx(i, j, k + 1)] - ey[idx];
                    hx[idx] =
                        self.da[idx] * hx[idx] - self.db_y[idx] * dez_dy + self.db_z[idx] * dey_dz;
                }
            }
        }
    }

    /// Apply Hy update: Hy\[i,j,k\] -= Da*(dEx/dz - dEz/dx)
    pub fn apply_hy(&self, hy: &mut [f64], ex: &[f64], ez: &[f64]) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        for i in 0..nx - 1 {
            for j in 0..ny {
                for k in 0..nz - 1 {
                    let idx = self.idx(i, j, k);
                    let dex_dz = ex[self.idx(i, j, k + 1)] - ex[idx];
                    let dez_dx = ez[self.idx(i + 1, j, k)] - ez[idx];
                    hy[idx] =
                        self.da[idx] * hy[idx] - self.db_z[idx] * dex_dz + self.db_x[idx] * dez_dx;
                }
            }
        }
    }

    /// Apply Hz update: Hz\[i,j,k\] -= Da*(dEy/dx - dEx/dy)
    pub fn apply_hz(&self, hz: &mut [f64], ex: &[f64], ey: &[f64]) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        for i in 0..nx - 1 {
            for j in 0..ny - 1 {
                for k in 0..nz {
                    let idx = self.idx(i, j, k);
                    let dey_dx = ey[self.idx(i + 1, j, k)] - ey[idx];
                    let dex_dy = ex[self.idx(i, j + 1, k)] - ex[idx];
                    hz[idx] =
                        self.da[idx] * hz[idx] - self.db_x[idx] * dey_dx + self.db_y[idx] * dex_dy;
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
    fn da_lossless_is_one() {
        let c = HUpdateCoeffs::lossless_1d(10, 1e-16, 10e-9);
        for i in 0..10 {
            assert_relative_eq!(c.da[i], 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn db_positive() {
        let c = HUpdateCoeffs::lossless_1d(10, 1e-16, 10e-9);
        for i in 0..10 {
            assert!(c.db[i] > 0.0);
        }
    }

    #[test]
    fn pmc_zeroes_h() {
        let mut c = HUpdateCoeffs::lossless_1d(10, 1e-16, 10e-9);
        c.set_pmc(3, 7);
        let mut hy = vec![1.0_f64; 10];
        let ex = vec![0.0_f64; 10];
        c.apply_1d_tem(&mut hy, &ex);
        for h in hy.iter().take(7).skip(3) {
            assert_eq!(*h, 0.0);
        }
    }

    #[test]
    fn apply_1d_tem_uniform_e_no_change() {
        let c = HUpdateCoeffs::lossless_1d(10, 1e-16, 10e-9);
        let mut hy = vec![1.0_f64; 10];
        let ex = vec![1.0_f64; 10]; // uniform → curl = 0
        c.apply_1d_tem(&mut hy, &ex);
        for h in hy.iter().take(9) {
            assert_relative_eq!(*h, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn apply_1d_tem_gradient_updates_h() {
        let c = HUpdateCoeffs::lossless_1d(10, 1e-16, 10e-9);
        let mut hy = vec![0.0_f64; 10];
        let mut ex = vec![0.0_f64; 10];
        ex[5] = 1.0; // step in E
        c.apply_1d_tem(&mut hy, &ex);
        // hy[4] gets -Db*(ex[5]-ex[4]) = -Db < 0
        assert!(hy[4] < 0.0);
    }

    #[test]
    fn h2d_lossless_da_one() {
        let c = HUpdateCoeffs2d::lossless(4, 4, 1e-16, 10e-9);
        assert!(c.da_hz.iter().all(|&v| (v - 1.0).abs() < 1e-12));
    }

    #[test]
    fn h3d_da_all_one() {
        let c = HUpdateCoeffs3d::lossless(4, 4, 4, 1e-16, 10e-9, 10e-9, 10e-9);
        assert!(c.da.iter().all(|&v| (v - 1.0).abs() < 1e-12));
    }

    #[test]
    fn subpixel_db_between_extremes() {
        let c = HUpdateCoeffs::lossless_1d(5, 1e-16, 10e-9);
        let db_1 = c.subpixel_db(0, 1.0, 1.0, 0.5, 1e-16, 10e-9); // same material
        let db_c = c.db[0];
        assert_relative_eq!(db_1, db_c, epsilon = 1e-10);
    }
}
