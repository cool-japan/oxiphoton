//! Near-to-far-field (NTFF) transforms for FDTD.
//!
//! Provides 2D and 3D near-to-far-field transforms using the field equivalence principle.
//! Records tangential E and H fields on a closed Huygens surface and computes
//! far-field radiation patterns via DFT integration.
//!
//! References:
//!   - Taflove & Hagness, "Computational Electrodynamics", 3rd Ed., Ch. 8
//!   - Balanis, "Antenna Theory", 3rd Ed.

use num_complex::Complex64;
use std::f64::consts::PI;

// Speed of light (m/s)
const C: f64 = 2.997_924_58e8;

// ─────────────────────────────────────────────────────────────────
// 2D NTFF (original)
// ─────────────────────────────────────────────────────────────────

/// Near-to-far-field (NTFF) transform for 2D TE FDTD.
///
/// Uses the field equivalence principle (Huygens surface):
/// Records tangential E and H fields on a closed rectangular contour
/// and computes the far-field radiation pattern via the DFT.
///
/// For 2D TE (Hz, Ex, Ey), the far-field is computed as a function of
/// angle θ ∈ [0, 2π) from the DFT of the surface fields.
///
/// Reference: Taflove & Hagness, "Computational Electrodynamics", Ch. 8.
pub struct NearToFarField2d {
    /// Number of x cells in the main grid
    pub nx: usize,
    /// Number of y cells
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,

    /// NTFF box boundaries (in cell indices)
    pub i_min: usize,
    pub i_max: usize,
    pub j_min: usize,
    pub j_max: usize,

    /// Angular frequency of interest (rad/s)
    pub omega: f64,

    // DFT accumulators on the four sides (complex)
    /// Bottom (j=j_min): Hz and Ex DFTs along i direction
    hz_bot: Vec<Complex64>,
    ex_bot: Vec<Complex64>,

    /// Top (j=j_max): Hz and Ex DFTs along i direction
    hz_top: Vec<Complex64>,
    ex_top: Vec<Complex64>,

    /// Left (i=i_min): Hz and Ey DFTs along j direction
    hz_left: Vec<Complex64>,
    ey_left: Vec<Complex64>,

    /// Right (i=i_max): Hz and Ey DFTs along j direction
    hz_right: Vec<Complex64>,
    ey_right: Vec<Complex64>,
}

impl NearToFarField2d {
    /// Create NTFF monitor recording at a single frequency `omega` (rad/s).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
        i_min: usize,
        i_max: usize,
        j_min: usize,
        j_max: usize,
        omega: f64,
    ) -> Self {
        let ni = i_max - i_min + 1;
        let nj = j_max - j_min + 1;
        Self {
            nx,
            ny,
            dx,
            dy,
            i_min,
            i_max,
            j_min,
            j_max,
            omega,
            hz_bot: vec![Complex64::new(0.0, 0.0); ni],
            ex_bot: vec![Complex64::new(0.0, 0.0); ni],
            hz_top: vec![Complex64::new(0.0, 0.0); ni],
            ex_top: vec![Complex64::new(0.0, 0.0); ni],
            hz_left: vec![Complex64::new(0.0, 0.0); nj],
            ey_left: vec![Complex64::new(0.0, 0.0); nj],
            hz_right: vec![Complex64::new(0.0, 0.0); nj],
            ey_right: vec![Complex64::new(0.0, 0.0); nj],
        }
    }

    /// Accumulate field values at time `t` with time step `dt`.
    pub fn accumulate(&mut self, hz: &[f64], ex: &[f64], ey: &[f64], t: f64, dt: f64) {
        let phase = Complex64::new(0.0, self.omega * t).exp() * dt;

        // Bottom (j = j_min)
        let j = self.j_min;
        for (k, i) in (self.i_min..=self.i_max).enumerate() {
            if j < self.ny && i < self.nx {
                self.hz_bot[k] += hz[j * self.nx + i] * phase;
            }
            if j < self.ny && i <= self.nx {
                self.ex_bot[k] += ex[j * (self.nx + 1) + i] * phase;
            }
        }

        // Top (j = j_max)
        let j = self.j_max;
        for (k, i) in (self.i_min..=self.i_max).enumerate() {
            if j < self.ny && i < self.nx {
                self.hz_top[k] += hz[j * self.nx + i] * phase;
            }
            if j < self.ny && i <= self.nx {
                self.ex_top[k] += ex[j * (self.nx + 1) + i] * phase;
            }
        }

        // Left (i = i_min)
        let i = self.i_min;
        for (k, j) in (self.j_min..=self.j_max).enumerate() {
            if j < self.ny && i < self.nx {
                self.hz_left[k] += hz[j * self.nx + i] * phase;
            }
            if j <= self.ny && i < self.nx {
                self.ey_left[k] += ey[j * self.nx + i] * phase;
            }
        }

        // Right (i = i_max)
        let i = self.i_max;
        for (k, j) in (self.j_min..=self.j_max).enumerate() {
            if j < self.ny && i < self.nx {
                self.hz_right[k] += hz[j * self.nx + i] * phase;
            }
            if j <= self.ny && i < self.nx {
                self.ey_right[k] += ey[j * self.nx + i] * phase;
            }
        }
    }

    /// Compute far-field radiation pattern as a function of angle θ.
    pub fn radiation_pattern(&self, theta_deg_values: &[f64]) -> Vec<f64> {
        let k0 = self.omega / C;
        let ni = self.i_max - self.i_min + 1;
        let nj = self.j_max - self.j_min + 1;

        let cx = (self.i_min + self.i_max) as f64 / 2.0 * self.dx;
        let cy = (self.j_min + self.j_max) as f64 / 2.0 * self.dy;

        theta_deg_values
            .iter()
            .map(|&theta_deg| {
                let theta = theta_deg * PI / 180.0;
                let cos_t = theta.cos();
                let sin_t = theta.sin();
                let mut f = Complex64::new(0.0, 0.0);

                // Bottom side
                let y_bot = self.j_min as f64 * self.dy - cy;
                for k in 0..ni {
                    let x = (self.i_min + k) as f64 * self.dx - cx;
                    let r_dot = x * cos_t + y_bot * sin_t;
                    let ph = Complex64::new(0.0, k0 * r_dot).exp();
                    f += (self.ex_bot[k] - self.hz_bot[k]) * ph * self.dx;
                }

                // Top side
                let y_top = self.j_max as f64 * self.dy - cy;
                for k in 0..ni {
                    let x = (self.i_min + k) as f64 * self.dx - cx;
                    let r_dot = x * cos_t + y_top * sin_t;
                    let ph = Complex64::new(0.0, k0 * r_dot).exp();
                    f += (-self.ex_top[k] + self.hz_top[k]) * ph * self.dx;
                }

                // Left side
                let x_left = self.i_min as f64 * self.dx - cx;
                for k in 0..nj {
                    let y = (self.j_min + k) as f64 * self.dy - cy;
                    let r_dot = x_left * cos_t + y * sin_t;
                    let ph = Complex64::new(0.0, k0 * r_dot).exp();
                    f += (-self.ey_left[k] - self.hz_left[k]) * ph * self.dy;
                }

                // Right side
                let x_right = self.i_max as f64 * self.dx - cx;
                for k in 0..nj {
                    let y = (self.j_min + k) as f64 * self.dy - cy;
                    let r_dot = x_right * cos_t + y * sin_t;
                    let ph = Complex64::new(0.0, k0 * r_dot).exp();
                    f += (self.ey_right[k] + self.hz_right[k]) * ph * self.dy;
                }

                f.norm_sqr()
            })
            .collect()
    }

    /// Directivity index: ratio of max to mean power in radiation pattern.
    pub fn directivity(&self, n_angles: usize) -> f64 {
        let thetas: Vec<f64> = (0..n_angles)
            .map(|i| i as f64 / n_angles as f64 * 360.0)
            .collect();
        let pattern = self.radiation_pattern(&thetas);
        let max_p = pattern.iter().cloned().fold(0.0_f64, f64::max);
        let mean_p = pattern.iter().sum::<f64>() / n_angles as f64;
        if mean_p > 0.0 {
            max_p / mean_p
        } else {
            0.0
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// 3D NTFF
// ─────────────────────────────────────────────────────────────────

/// 3D near-to-far-field transform.
///
/// Integrates tangential E and H on a closed rectangular surface around the source
/// and computes far-field radiation patterns in spherical coordinates (θ, φ).
///
/// The Huygens surface has 6 faces. On each face, we accumulate DFT of the
/// equivalent surface currents:
///   J  = n̂ × H  (electric surface current, A/m)
///   M  = -n̂ × E  (magnetic surface current, V/m)
///
/// The far-field electric potential functions (N, L) are computed by
/// integrating J and M over the surface with appropriate phase factors.
///
/// # Grid Layout
/// Fields: `field[i * ny * nz + j * nz + k]`
pub struct NearToFarField3d {
    /// Free-space wavelength (m) (computed from omega)
    pub wavelength: f64,
    /// Surface extent
    pub i0: usize,
    pub i1: usize,
    pub j0: usize,
    pub j1: usize,
    pub k0: usize,
    pub k1: usize,
    /// Angular frequency (rad/s)
    pub omega: f64,
    /// DFT accumulators for equivalent surface currents on 6 faces
    /// Each face stores complex J and M current components
    /// Face 0: k=k0 (bottom, normal=-Z), Face 1: k=k1 (top, normal=+Z)
    /// Face 2: j=j0 (front, normal=-Y), Face 3: j=j1 (back, normal=+Y)
    /// Face 4: i=i0 (left, normal=-X), Face 5: i=i1 (right, normal=+X)
    // Bottom face (k=k0): Jx, Jy (tangential J) and Mx, My
    jx_bot: Vec<Complex64>,
    jy_bot: Vec<Complex64>,
    mx_bot: Vec<Complex64>,
    my_bot: Vec<Complex64>,
    // Top face (k=k1)
    jx_top: Vec<Complex64>,
    jy_top: Vec<Complex64>,
    mx_top: Vec<Complex64>,
    my_top: Vec<Complex64>,
    // Front face (j=j0): Jx, Jz and Mx, Mz
    jx_fnt: Vec<Complex64>,
    jz_fnt: Vec<Complex64>,
    mx_fnt: Vec<Complex64>,
    mz_fnt: Vec<Complex64>,
    // Back face (j=j1)
    jx_bck: Vec<Complex64>,
    jz_bck: Vec<Complex64>,
    mx_bck: Vec<Complex64>,
    mz_bck: Vec<Complex64>,
    // Left face (i=i0): Jy, Jz and My, Mz
    jy_lft: Vec<Complex64>,
    jz_lft: Vec<Complex64>,
    my_lft: Vec<Complex64>,
    mz_lft: Vec<Complex64>,
    // Right face (i=i1)
    jy_rgt: Vec<Complex64>,
    jz_rgt: Vec<Complex64>,
    my_rgt: Vec<Complex64>,
    mz_rgt: Vec<Complex64>,
    /// Number of accumulation samples
    pub n_samples: usize,
    /// Simulation time step (s)
    pub dt: f64,
    /// Grid spacings (m)
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    /// Grid dimensions
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    // Face sizes (precomputed)
    ni: usize,
    nj: usize,
    nk: usize,
}

impl NearToFarField3d {
    /// Create a new 3D NTFF monitor.
    ///
    /// # Arguments
    /// * `i0..i1`, `j0..j1`, `k0..k1` — Huygens surface boundaries (inclusive)
    /// * `omega` — angular frequency to analyze (rad/s)
    /// * `dt` — simulation time step (s)
    /// * `dx`, `dy`, `dz` — grid spacings (m)
    /// * `nx`, `ny`, `nz` — grid dimensions
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        k0: usize,
        k1: usize,
        omega: f64,
        dt: f64,
        dx: f64,
        dy: f64,
        dz: f64,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Self {
        let wavelength = 2.0 * PI * C / omega;
        let ni = i1.saturating_sub(i0) + 1;
        let nj = j1.saturating_sub(j0) + 1;
        let nk = k1.saturating_sub(k0) + 1;

        // Bottom/top face sizes: ni × nj cells
        let n_bot_top = ni * nj;
        // Front/back face sizes: ni × nk cells
        let n_fnt_bck = ni * nk;
        // Left/right face sizes: nj × nk cells
        let n_lft_rgt = nj * nk;

        let zero_vec = |n: usize| vec![Complex64::new(0.0, 0.0); n];

        Self {
            wavelength,
            i0,
            i1,
            j0,
            j1,
            k0,
            k1,
            omega,
            jx_bot: zero_vec(n_bot_top),
            jy_bot: zero_vec(n_bot_top),
            mx_bot: zero_vec(n_bot_top),
            my_bot: zero_vec(n_bot_top),
            jx_top: zero_vec(n_bot_top),
            jy_top: zero_vec(n_bot_top),
            mx_top: zero_vec(n_bot_top),
            my_top: zero_vec(n_bot_top),
            jx_fnt: zero_vec(n_fnt_bck),
            jz_fnt: zero_vec(n_fnt_bck),
            mx_fnt: zero_vec(n_fnt_bck),
            mz_fnt: zero_vec(n_fnt_bck),
            jx_bck: zero_vec(n_fnt_bck),
            jz_bck: zero_vec(n_fnt_bck),
            mx_bck: zero_vec(n_fnt_bck),
            mz_bck: zero_vec(n_fnt_bck),
            jy_lft: zero_vec(n_lft_rgt),
            jz_lft: zero_vec(n_lft_rgt),
            my_lft: zero_vec(n_lft_rgt),
            mz_lft: zero_vec(n_lft_rgt),
            jy_rgt: zero_vec(n_lft_rgt),
            jz_rgt: zero_vec(n_lft_rgt),
            my_rgt: zero_vec(n_lft_rgt),
            mz_rgt: zero_vec(n_lft_rgt),
            n_samples: 0,
            dt,
            dx,
            dy,
            dz,
            nx,
            ny,
            nz,
            ni,
            nj,
            nk,
        }
    }

    /// Flat index for 3D arrays: `field[i * ny * nz + j * nz + k]`
    #[inline]
    fn idx3(&self, i: usize, j: usize, k: usize) -> usize {
        i * self.ny * self.nz + j * self.nz + k
    }

    /// Accumulate surface current DFT from the current field state.
    ///
    /// On each face, computes J = n̂ × H and M = -n̂ × E and accumulates
    /// their running DFT at the monitor frequency.
    #[allow(clippy::too_many_arguments)]
    pub fn accumulate(
        &mut self,
        time_step: usize,
        ex: &[f64],
        ey: &[f64],
        ez: &[f64],
        hx: &[f64],
        hy: &[f64],
        hz: &[f64],
    ) {
        let t = time_step as f64 * self.dt;
        let phase = Complex64::new(0.0, self.omega * t).exp() * self.dt;

        // Helper: safe array access
        let get = |arr: &[f64], idx: usize| arr.get(idx).copied().unwrap_or(0.0);

        // ── Bottom face: k = k0, normal n̂ = -ẑ ───────────────────
        // J = n̂ × H = (-ẑ) × H = (-ẑ) × (Hx,Hy,Hz) = (Hy, -Hx, 0)
        // M = -n̂ × E = (ẑ) × E = (ẑ) × (Ex,Ey,Ez) = (-Ey, Ex, 0)
        let k = self.k0;
        if k < self.nz {
            for (ai, i) in (self.i0..=self.i1.min(self.nx.saturating_sub(1))).enumerate() {
                for (aj, j) in (self.j0..=self.j1.min(self.ny.saturating_sub(1))).enumerate() {
                    let idx = self.idx3(i, j, k);
                    let cell = ai * self.nj + aj;
                    if cell < self.jx_bot.len() {
                        self.jx_bot[cell] += get(hy, idx) * phase;
                        self.jy_bot[cell] += -get(hx, idx) * phase;
                        self.mx_bot[cell] += -get(ey, idx) * phase;
                        self.my_bot[cell] += get(ex, idx) * phase;
                    }
                }
            }
        }

        // ── Top face: k = k1, normal n̂ = +ẑ ─────────────────────
        // J = n̂ × H = ẑ × H = (-Hy, Hx, 0)
        // M = -n̂ × E = -ẑ × E = (Ey, -Ex, 0)
        let k = self.k1.min(self.nz.saturating_sub(1));
        for (ai, i) in (self.i0..=self.i1.min(self.nx.saturating_sub(1))).enumerate() {
            for (aj, j) in (self.j0..=self.j1.min(self.ny.saturating_sub(1))).enumerate() {
                let idx = self.idx3(i, j, k);
                let cell = ai * self.nj + aj;
                if cell < self.jx_top.len() {
                    self.jx_top[cell] += -get(hy, idx) * phase;
                    self.jy_top[cell] += get(hx, idx) * phase;
                    self.mx_top[cell] += get(ey, idx) * phase;
                    self.my_top[cell] += -get(ex, idx) * phase;
                }
            }
        }

        // ── Front face: j = j0, normal n̂ = -ŷ ───────────────────
        // J = (-ŷ) × H = (-ŷ) × (Hx,Hy,Hz) = (-Hz, 0, Hx)
        // M = -(−ŷ) × E = ŷ × E = ŷ × (Ex,Ey,Ez) = (Ez, 0, -Ex)
        let j = self.j0;
        if j < self.ny {
            for (ai, i) in (self.i0..=self.i1.min(self.nx.saturating_sub(1))).enumerate() {
                for (ak, k) in (self.k0..=self.k1.min(self.nz.saturating_sub(1))).enumerate() {
                    let idx = self.idx3(i, j, k);
                    let cell = ai * self.nk + ak;
                    if cell < self.jx_fnt.len() {
                        self.jx_fnt[cell] += -get(hz, idx) * phase;
                        self.jz_fnt[cell] += get(hx, idx) * phase;
                        self.mx_fnt[cell] += get(ez, idx) * phase;
                        self.mz_fnt[cell] += -get(ex, idx) * phase;
                    }
                }
            }
        }

        // ── Back face: j = j1, normal n̂ = +ŷ ────────────────────
        // J = ŷ × H = (Hz, 0, -Hx)
        // M = -(ŷ) × E = -ŷ × E = (-Ez, 0, Ex)
        let j = self.j1.min(self.ny.saturating_sub(1));
        for (ai, i) in (self.i0..=self.i1.min(self.nx.saturating_sub(1))).enumerate() {
            for (ak, k) in (self.k0..=self.k1.min(self.nz.saturating_sub(1))).enumerate() {
                let idx = self.idx3(i, j, k);
                let cell = ai * self.nk + ak;
                if cell < self.jx_bck.len() {
                    self.jx_bck[cell] += get(hz, idx) * phase;
                    self.jz_bck[cell] += -get(hx, idx) * phase;
                    self.mx_bck[cell] += -get(ez, idx) * phase;
                    self.mz_bck[cell] += get(ex, idx) * phase;
                }
            }
        }

        // ── Left face: i = i0, normal n̂ = -x̂ ────────────────────
        // J = (-x̂) × H = (0, Hz, -Hy)
        // M = -(-x̂) × E = x̂ × E = (0, -Ez, Ey)
        let i = self.i0;
        if i < self.nx {
            for (aj, j) in (self.j0..=self.j1.min(self.ny.saturating_sub(1))).enumerate() {
                for (ak, k) in (self.k0..=self.k1.min(self.nz.saturating_sub(1))).enumerate() {
                    let idx = self.idx3(i, j, k);
                    let cell = aj * self.nk + ak;
                    if cell < self.jy_lft.len() {
                        self.jy_lft[cell] += get(hz, idx) * phase;
                        self.jz_lft[cell] += -get(hy, idx) * phase;
                        self.my_lft[cell] += -get(ez, idx) * phase;
                        self.mz_lft[cell] += get(ey, idx) * phase;
                    }
                }
            }
        }

        // ── Right face: i = i1, normal n̂ = +x̂ ───────────────────
        // J = x̂ × H = (0, -Hz, Hy)
        // M = -(x̂) × E = (-x̂) × E = (0, Ez, -Ey)
        let i = self.i1.min(self.nx.saturating_sub(1));
        for (aj, j) in (self.j0..=self.j1.min(self.ny.saturating_sub(1))).enumerate() {
            for (ak, k) in (self.k0..=self.k1.min(self.nz.saturating_sub(1))).enumerate() {
                let idx = self.idx3(i, j, k);
                let cell = aj * self.nk + ak;
                if cell < self.jy_rgt.len() {
                    self.jy_rgt[cell] += -get(hz, idx) * phase;
                    self.jz_rgt[cell] += get(hy, idx) * phase;
                    self.my_rgt[cell] += get(ez, idx) * phase;
                    self.mz_rgt[cell] += -get(ey, idx) * phase;
                }
            }
        }

        self.n_samples += 1;
    }

    /// Compute the radiation integral vectors N and L at angle (theta, phi).
    ///
    /// N = ∫∫ J · exp(j·k·r'·r̂) dS'  (electric potential vector)
    /// L = ∫∫ M · exp(j·k·r'·r̂) dS'  (magnetic potential vector)
    ///
    /// Returns (N_theta, N_phi, L_theta, L_phi) complex amplitudes.
    fn radiation_integrals(&self, theta: f64, phi: f64) -> ([Complex64; 2], [Complex64; 2]) {
        let k0 = self.omega / C;
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let sin_p = phi.sin();
        let cos_p = phi.cos();

        // Observer direction unit vector: r̂ = (sin_t*cos_p, sin_t*sin_p, cos_t)
        let rx = sin_t * cos_p;
        let ry = sin_t * sin_p;
        let rz = cos_t;

        // Center of the NTFF box (physical coords)
        let cx = 0.5 * (self.i0 + self.i1) as f64 * self.dx;
        let cy = 0.5 * (self.j0 + self.j1) as f64 * self.dy;
        let cz = 0.5 * (self.k0 + self.k1) as f64 * self.dz;

        // Accumulate N (from J) and L (from M) vectors in Cartesian coords
        let mut nx_c = Complex64::new(0.0, 0.0);
        let mut ny_c = Complex64::new(0.0, 0.0);
        let mut nz_c = Complex64::new(0.0, 0.0);
        let mut lx_c = Complex64::new(0.0, 0.0);
        let mut ly_c = Complex64::new(0.0, 0.0);
        let mut lz_c = Complex64::new(0.0, 0.0);

        // ── Bottom face (k=k0, n=-z, dS=dx*dy) ──────────────────
        let z = self.k0 as f64 * self.dz - cz;
        let ds = self.dx * self.dy;
        for (ai, i) in (self.i0..=self.i1.min(self.nx.saturating_sub(1))).enumerate() {
            for (aj, j) in (self.j0..=self.j1.min(self.ny.saturating_sub(1))).enumerate() {
                let x = i as f64 * self.dx - cx;
                let y = j as f64 * self.dy - cy;
                let r_dot = rx * x + ry * y + rz * z;
                let ph = Complex64::new(0.0, k0 * r_dot).exp() * ds;
                let cell = ai * self.nj + aj;
                if cell < self.jx_bot.len() {
                    nx_c += self.jx_bot[cell] * ph;
                    ny_c += self.jy_bot[cell] * ph;
                    lx_c += self.mx_bot[cell] * ph;
                    ly_c += self.my_bot[cell] * ph;
                }
            }
        }

        // ── Top face (k=k1, n=+z, dS=dx*dy) ─────────────────────
        let z = self.k1 as f64 * self.dz - cz;
        for (ai, i) in (self.i0..=self.i1.min(self.nx.saturating_sub(1))).enumerate() {
            for (aj, j) in (self.j0..=self.j1.min(self.ny.saturating_sub(1))).enumerate() {
                let x = i as f64 * self.dx - cx;
                let y = j as f64 * self.dy - cy;
                let r_dot = rx * x + ry * y + rz * z;
                let ph = Complex64::new(0.0, k0 * r_dot).exp() * ds;
                let cell = ai * self.nj + aj;
                if cell < self.jx_top.len() {
                    nx_c += self.jx_top[cell] * ph;
                    ny_c += self.jy_top[cell] * ph;
                    lx_c += self.mx_top[cell] * ph;
                    ly_c += self.my_top[cell] * ph;
                }
            }
        }

        // ── Front face (j=j0, n=-y, dS=dx*dz) ───────────────────
        let y = self.j0 as f64 * self.dy - cy;
        let ds_fnt = self.dx * self.dz;
        for (ai, i) in (self.i0..=self.i1.min(self.nx.saturating_sub(1))).enumerate() {
            for (ak, k) in (self.k0..=self.k1.min(self.nz.saturating_sub(1))).enumerate() {
                let x = i as f64 * self.dx - cx;
                let z = k as f64 * self.dz - cz;
                let r_dot = rx * x + ry * y + rz * z;
                let ph = Complex64::new(0.0, k0 * r_dot).exp() * ds_fnt;
                let cell = ai * self.nk + ak;
                if cell < self.jx_fnt.len() {
                    nx_c += self.jx_fnt[cell] * ph;
                    nz_c += self.jz_fnt[cell] * ph;
                    lx_c += self.mx_fnt[cell] * ph;
                    lz_c += self.mz_fnt[cell] * ph;
                }
            }
        }

        // ── Back face (j=j1, n=+y, dS=dx*dz) ────────────────────
        let y = self.j1 as f64 * self.dy - cy;
        for (ai, i) in (self.i0..=self.i1.min(self.nx.saturating_sub(1))).enumerate() {
            for (ak, k) in (self.k0..=self.k1.min(self.nz.saturating_sub(1))).enumerate() {
                let x = i as f64 * self.dx - cx;
                let z = k as f64 * self.dz - cz;
                let r_dot = rx * x + ry * y + rz * z;
                let ph = Complex64::new(0.0, k0 * r_dot).exp() * ds_fnt;
                let cell = ai * self.nk + ak;
                if cell < self.jx_bck.len() {
                    nx_c += self.jx_bck[cell] * ph;
                    nz_c += self.jz_bck[cell] * ph;
                    lx_c += self.mx_bck[cell] * ph;
                    lz_c += self.mz_bck[cell] * ph;
                }
            }
        }

        // ── Left face (i=i0, n=-x, dS=dy*dz) ────────────────────
        let x = self.i0 as f64 * self.dx - cx;
        let ds_lft = self.dy * self.dz;
        for (aj, j) in (self.j0..=self.j1.min(self.ny.saturating_sub(1))).enumerate() {
            for (ak, k) in (self.k0..=self.k1.min(self.nz.saturating_sub(1))).enumerate() {
                let y = j as f64 * self.dy - cy;
                let z = k as f64 * self.dz - cz;
                let r_dot = rx * x + ry * y + rz * z;
                let ph = Complex64::new(0.0, k0 * r_dot).exp() * ds_lft;
                let cell = aj * self.nk + ak;
                if cell < self.jy_lft.len() {
                    ny_c += self.jy_lft[cell] * ph;
                    nz_c += self.jz_lft[cell] * ph;
                    ly_c += self.my_lft[cell] * ph;
                    lz_c += self.mz_lft[cell] * ph;
                }
            }
        }

        // ── Right face (i=i1, n=+x, dS=dy*dz) ───────────────────
        let x = self.i1 as f64 * self.dx - cx;
        for (aj, j) in (self.j0..=self.j1.min(self.ny.saturating_sub(1))).enumerate() {
            for (ak, k) in (self.k0..=self.k1.min(self.nz.saturating_sub(1))).enumerate() {
                let y = j as f64 * self.dy - cy;
                let z = k as f64 * self.dz - cz;
                let r_dot = rx * x + ry * y + rz * z;
                let ph = Complex64::new(0.0, k0 * r_dot).exp() * ds_lft;
                let cell = aj * self.nk + ak;
                if cell < self.jy_rgt.len() {
                    ny_c += self.jy_rgt[cell] * ph;
                    nz_c += self.jz_rgt[cell] * ph;
                    ly_c += self.my_rgt[cell] * ph;
                    lz_c += self.mz_rgt[cell] * ph;
                }
            }
        }

        // Project N and L onto (theta_hat, phi_hat) components
        // theta_hat = (cos_t*cos_p, cos_t*sin_p, -sin_t)
        // phi_hat   = (-sin_p, cos_p, 0)
        let n_theta = nx_c * (cos_t * cos_p) + ny_c * (cos_t * sin_p) + nz_c * (-sin_t);
        let n_phi = nx_c * (-sin_p) + ny_c * cos_p;
        let l_theta = lx_c * (cos_t * cos_p) + ly_c * (cos_t * sin_p) + lz_c * (-sin_t);
        let l_phi = lx_c * (-sin_p) + ly_c * cos_p;

        ([n_theta, n_phi], [l_theta, l_phi])
    }

    /// Compute far-field radiation pattern at angle (theta, phi) in spherical coordinates.
    ///
    /// Returns (E_theta, E_phi) complex amplitudes as (\[re, im\], \[re, im\]).
    ///
    /// Far-field approximation:
    ///   E_theta ≈ -j·k₀/(4π) · (L_phi + η₀·N_theta)
    ///   E_phi   ≈  j·k₀/(4π) · (L_theta - η₀·N_phi)
    ///
    /// where η₀ = 377 Ω is the free-space impedance.
    pub fn far_field(&self, theta: f64, phi: f64, r: f64) -> ([f64; 2], [f64; 2]) {
        let k0 = self.omega / C;
        let eta0 = 376.730_313_461_77_f64;
        let ([n_theta, n_phi], [l_theta, l_phi]) = self.radiation_integrals(theta, phi);

        // Far-field electric field (Balanis 6-122a,b)
        let j = Complex64::new(0.0, 1.0);
        let factor = -j * k0 / (4.0 * PI * r);
        let e_theta = factor * (l_phi + eta0 * n_theta);
        let e_phi = factor * (-l_theta + eta0 * n_phi);

        ([e_theta.re, e_theta.im], [e_phi.re, e_phi.im])
    }

    /// Compute the directivity pattern over a spherical grid.
    ///
    /// Returns a 2D grid of directivity (dBi) values:
    ///   result\[i_theta\]\[i_phi\]  where theta in \[0, π\], phi in \[0, 2π\]
    ///
    /// # Arguments
    /// * `n_theta` — number of theta samples (polar angle, 0=north pole)
    /// * `n_phi` — number of phi samples (azimuthal angle)
    pub fn directivity_pattern(&self, n_theta: usize, n_phi: usize) -> Vec<Vec<f64>> {
        let r = 1.0_f64; // Far-field distance (normalized)

        // Compute |E|² at all angles
        let mut power_grid = vec![vec![0.0_f64; n_phi]; n_theta];
        let mut total_power = 0.0_f64;

        for (it, theta_row) in power_grid.iter_mut().enumerate().take(n_theta) {
            let theta = it as f64 / (n_theta - 1).max(1) as f64 * PI;
            let sin_t = theta.sin();
            for (ip, cell) in theta_row.iter_mut().enumerate().take(n_phi) {
                let phi = ip as f64 / (n_phi - 1).max(1) as f64 * 2.0 * PI;
                let ([et_re, et_im], [ep_re, ep_im]) = self.far_field(theta, phi, r);
                let power = et_re * et_re + et_im * et_im + ep_re * ep_re + ep_im * ep_im;
                *cell = power;
                // Weight by sin(theta) for spherical integration
                total_power += power * sin_t;
            }
        }

        // Normalize to compute directivity (ratio to isotropic)
        // Isotropic power = total_power / (4π) if integrating over full sphere
        let d_theta = if n_theta > 1 {
            PI / (n_theta - 1) as f64
        } else {
            1.0
        };
        let d_phi = if n_phi > 1 {
            2.0 * PI / (n_phi - 1) as f64
        } else {
            1.0
        };
        let p_total = total_power * d_theta * d_phi;
        let p_isotropic = p_total / (4.0 * PI);

        // Convert to dBi
        power_grid
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&p| {
                        if p_isotropic > 1e-60 && p > 1e-60 {
                            10.0 * (p / p_isotropic).log10()
                        } else {
                            -999.0
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Estimate total radiated power (W) from near-field surface integration.
    ///
    /// Uses the Poynting theorem:
    ///   P_rad = 0.5 · Re ∫∫ (E × H*) · n̂ dS
    ///
    /// Approximated from the DFT accumulators as the norm of the total current.
    pub fn total_radiated_power(&self) -> f64 {
        let eta0 = 376.730_313_461_77_f64;
        // Simple estimate: ||N||² / (2*eta0) integrated over all directions
        // For a single angle, this gives a rough estimate
        // Full integration would require summing over the sphere
        let mut total = 0.0_f64;

        // Sum |J|² over all faces as a proxy for radiated power
        let sum_j_sq = |v: &[Complex64]| v.iter().map(|c| c.norm_sqr()).sum::<f64>();
        total += sum_j_sq(&self.jx_bot) + sum_j_sq(&self.jy_bot);
        total += sum_j_sq(&self.jx_top) + sum_j_sq(&self.jy_top);
        total += sum_j_sq(&self.jx_fnt) + sum_j_sq(&self.jz_fnt);
        total += sum_j_sq(&self.jx_bck) + sum_j_sq(&self.jz_bck);
        total += sum_j_sq(&self.jy_lft) + sum_j_sq(&self.jz_lft);
        total += sum_j_sq(&self.jy_rgt) + sum_j_sq(&self.jz_rgt);

        // Scale by eta0/(2) and face areas for dimensional consistency
        total * eta0 / 2.0 * self.dx * self.dy * self.dz
    }

    /// Returns the number of cells on each face: (n_bot_top, n_fnt_bck, n_lft_rgt).
    pub fn face_sizes(&self) -> (usize, usize, usize) {
        (self.ni * self.nj, self.ni * self.nk, self.nj * self.nk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ntff() -> NearToFarField2d {
        let nx = 80;
        let ny = 80;
        let dx = 20e-9;
        let dy = 20e-9;
        NearToFarField2d::new(nx, ny, dx, dy, 15, 65, 15, 65, 2.0 * PI * 200e12)
    }

    #[test]
    fn ntff_initializes_zero() {
        let m = make_ntff();
        assert!(m.hz_bot.iter().all(|v| v.norm() == 0.0));
        assert!(m.ex_bot.iter().all(|v| v.norm() == 0.0));
    }

    #[test]
    fn ntff_accumulate_does_not_panic() {
        let mut m = make_ntff();
        let nx = m.nx;
        let ny = m.ny;
        let hz = vec![0.0f64; nx * ny];
        let ex = vec![0.0f64; (nx + 1) * ny];
        let ey = vec![0.0f64; nx * (ny + 1)];
        m.accumulate(&hz, &ex, &ey, 1e-15, 1e-17);
        // Accumulators stay zero when fields are zero
        assert!(m.hz_bot.iter().all(|v| v.norm() == 0.0));
    }

    #[test]
    fn ntff_radiation_pattern_returns_correct_size() {
        let m = make_ntff();
        let angles: Vec<f64> = (0..36).map(|i| i as f64 * 10.0).collect();
        let pattern = m.radiation_pattern(&angles);
        assert_eq!(pattern.len(), 36);
        assert!(pattern.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn ntff_zero_fields_give_zero_pattern() {
        let m = make_ntff();
        let angles = vec![0.0, 90.0, 180.0, 270.0];
        let pattern = m.radiation_pattern(&angles);
        for &p in &pattern {
            assert!(
                p.abs() < 1e-60,
                "pattern should be ~0 with no field accumulated"
            );
        }
    }

    // 3D NTFF tests
    #[test]
    fn ntff_3d_new_correct_dimensions() {
        let ntff = NearToFarField3d::new(
            5,
            15,
            5,
            15,
            5,
            15,
            2.0 * PI * 200e12,
            1e-17,
            20e-9,
            20e-9,
            20e-9,
            20,
            20,
            20,
        );
        let (n_bt, n_fb, n_lr) = ntff.face_sizes();
        assert_eq!(n_bt, 11 * 11, "Bottom/top face should be ni*nj");
        assert_eq!(n_fb, 11 * 11, "Front/back face should be ni*nk");
        assert_eq!(n_lr, 11 * 11, "Left/right face should be nj*nk");
        assert_eq!(ntff.n_samples, 0);
    }

    #[test]
    fn ntff_3d_accumulate_with_zero_fields() {
        let mut ntff = NearToFarField3d::new(
            5,
            15,
            5,
            15,
            5,
            15,
            2.0 * PI * 200e12,
            1e-17,
            20e-9,
            20e-9,
            20e-9,
            20,
            20,
            20,
        );
        let n = 20 * 20 * 20;
        let ex = vec![0.0f64; n];
        let ey = vec![0.0f64; n];
        let ez = vec![0.0f64; n];
        let hx = vec![0.0f64; n];
        let hy = vec![0.0f64; n];
        let hz = vec![0.0f64; n];
        ntff.accumulate(0, &ex, &ey, &ez, &hx, &hy, &hz);
        assert_eq!(ntff.n_samples, 1);
        // With zero fields, all currents should be zero
        assert!(ntff.jx_bot.iter().all(|c| c.norm() == 0.0));
    }

    #[test]
    fn ntff_3d_accumulate_with_nonzero_fields() {
        let mut ntff = NearToFarField3d::new(
            3,
            8,
            3,
            8,
            3,
            8,
            2.0 * PI * 200e12,
            1e-17,
            20e-9,
            20e-9,
            20e-9,
            12,
            12,
            12,
        );
        let n = 12 * 12 * 12;
        let ex = vec![1.0f64; n];
        let ey = vec![0.5f64; n];
        let ez = vec![0.0f64; n];
        let hx = vec![0.0f64; n];
        let hy = vec![1.0f64; n];
        let hz = vec![0.0f64; n];
        ntff.accumulate(0, &ex, &ey, &ez, &hx, &hy, &hz);
        // With nonzero Hy, jx_bot should accumulate (J = (-z) × H → Jx = Hy)
        let any_nonzero = ntff.jx_bot.iter().any(|c| c.norm() > 0.0)
            || ntff.jy_bot.iter().any(|c| c.norm() > 0.0)
            || ntff.mx_bot.iter().any(|c| c.norm() > 0.0);
        assert!(
            any_nonzero,
            "Nonzero fields should produce nonzero surface currents"
        );
    }

    #[test]
    fn ntff_3d_far_field_returns_finite() {
        let ntff = NearToFarField3d::new(
            5,
            15,
            5,
            15,
            5,
            15,
            2.0 * PI * 200e12,
            1e-17,
            20e-9,
            20e-9,
            20e-9,
            20,
            20,
            20,
        );
        let ([et_re, et_im], [ep_re, ep_im]) = ntff.far_field(PI / 2.0, 0.0, 1.0);
        assert!(et_re.is_finite() && et_im.is_finite());
        assert!(ep_re.is_finite() && ep_im.is_finite());
    }

    #[test]
    fn ntff_3d_directivity_pattern_shape() {
        let ntff = NearToFarField3d::new(
            5,
            10,
            5,
            10,
            5,
            10,
            2.0 * PI * 200e12,
            1e-17,
            20e-9,
            20e-9,
            20e-9,
            15,
            15,
            15,
        );
        let n_theta = 10;
        let n_phi = 12;
        let pattern = ntff.directivity_pattern(n_theta, n_phi);
        assert_eq!(pattern.len(), n_theta, "Pattern should have n_theta rows");
        assert_eq!(pattern[0].len(), n_phi, "Pattern should have n_phi columns");
    }

    #[test]
    fn ntff_3d_total_radiated_power_nonnegative() {
        let ntff = NearToFarField3d::new(
            5,
            10,
            5,
            10,
            5,
            10,
            2.0 * PI * 200e12,
            1e-17,
            20e-9,
            20e-9,
            20e-9,
            15,
            15,
            15,
        );
        let p = ntff.total_radiated_power();
        assert!(p >= 0.0, "Total radiated power should be non-negative: {p}");
    }

    #[test]
    fn ntff_3d_wavelength_computed_correctly() {
        let omega = 2.0 * PI * 200e12;
        let ntff = NearToFarField3d::new(
            5, 10, 5, 10, 5, 10, omega, 1e-17, 20e-9, 20e-9, 20e-9, 15, 15, 15,
        );
        let expected_lambda = C / 200e12;
        assert!(
            (ntff.wavelength - expected_lambda).abs() < 1e-12,
            "Wavelength mismatch: {} vs {}",
            ntff.wavelength,
            expected_lambda
        );
    }
}
