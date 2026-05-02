use num_complex::Complex64;
use std::f64::consts::PI;

/// 3D Bloch/quasi-periodic boundary condition for photonic crystal simulation.
///
/// Applies phase-shift: E(r + R) = E(r) · exp(i k·R)
///
/// For real-valued FDTD fields, the Bloch phase correction is applied at the
/// periodic boundaries by multiplying the wrapping field value by the real part
/// of the phase factor (cosine), which is valid for standing-wave modes.
/// Full complex-field Bloch FDTD requires storing both real and imaginary parts
/// of the field (see [`BlochFdtd1d`](super::periodic::BlochFdtd1d) for 1D).
///
/// Index convention: field is stored in row-major order with stride ny*nz,
/// i.e. field[i*ny*nz + j*nz + k] corresponds to grid point (i, j, k).
pub struct BlochBc3d {
    /// Bloch wavevector x-component (rad/m)
    pub kx: f64,
    /// Bloch wavevector y-component (rad/m)
    pub ky: f64,
    /// Bloch wavevector z-component (rad/m)
    pub kz: f64,
    /// Number of grid cells in x
    pub nx: usize,
    /// Number of grid cells in y
    pub ny: usize,
    /// Number of grid cells in z
    pub nz: usize,
    /// Grid spacing in x (m)
    pub dx: f64,
    /// Grid spacing in y (m)
    pub dy: f64,
    /// Grid spacing in z (m)
    pub dz: f64,
}

impl BlochBc3d {
    /// Create a new 3D Bloch boundary condition.
    ///
    /// # Arguments
    /// * `kx`, `ky`, `kz` — Bloch wavevector components in rad/m
    /// * `nx`, `ny`, `nz` — grid dimensions
    /// * `dx`, `dy`, `dz` — grid spacings in metres
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kx: f64,
        ky: f64,
        kz: f64,
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> Self {
        Self {
            kx,
            ky,
            kz,
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
        }
    }

    // -----------------------------------------------------------------------
    // Phase factors
    // -----------------------------------------------------------------------

    /// Phase factor for x-direction periodicity: exp(i · kx · Lx)
    pub fn phase_factor_x(&self) -> Complex64 {
        let phase = self.kx * self.nx as f64 * self.dx;
        Complex64::new(phase.cos(), phase.sin())
    }

    /// Phase factor for y-direction periodicity: exp(i · ky · Ly)
    pub fn phase_factor_y(&self) -> Complex64 {
        let phase = self.ky * self.ny as f64 * self.dy;
        Complex64::new(phase.cos(), phase.sin())
    }

    /// Phase factor for z-direction periodicity: exp(i · kz · Lz)
    pub fn phase_factor_z(&self) -> Complex64 {
        let phase = self.kz * self.nz as f64 * self.dz;
        Complex64::new(phase.cos(), phase.sin())
    }

    // -----------------------------------------------------------------------
    // Boundary application helpers
    // -----------------------------------------------------------------------

    /// Inline index helper: (i, j, k) → flat index in nx × ny × nz field.
    #[inline]
    fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        i * self.ny * self.nz + j * self.nz + k
    }

    /// Apply Bloch BC in the x-direction to a single real-valued field.
    ///
    /// The correction applies the cosine of the Bloch phase as a real
    /// amplitude factor (appropriate for real FDTD when the mode is
    /// decomposed into its Bloch envelope):
    ///
    ///   field[0, j, k]    ← field[nx-1, j, k] · cos(-kx·Lx)
    ///   field[nx-1, j, k] ← field[0, j, k]    · cos( kx·Lx)
    pub fn apply_x_phase(&self, field: &mut [f64]) {
        let pf = self.phase_factor_x();
        let cos_pos = pf.re; // cos(+kx·Lx)
        let cos_neg = pf.re; // cos is even: cos(-kx·Lx) = cos(+kx·Lx)
        for j in 0..self.ny {
            for k in 0..self.nz {
                let idx_0 = self.idx(0, j, k);
                let idx_nm1 = self.idx(self.nx - 1, j, k);
                // Save originals before overwriting
                let v0 = field[idx_0];
                let vnm1 = field[idx_nm1];
                // Apply Bloch phase wrap
                field[idx_nm1] = v0 * cos_pos;
                field[idx_0] = vnm1 * cos_neg;
            }
        }
    }

    /// Apply Bloch BC in the y-direction to a single real-valued field.
    pub fn apply_y_phase(&self, field: &mut [f64]) {
        let pf = self.phase_factor_y();
        let cos_val = pf.re;
        for i in 0..self.nx {
            for k in 0..self.nz {
                let idx_0 = self.idx(i, 0, k);
                let idx_nm1 = self.idx(i, self.ny - 1, k);
                let v0 = field[idx_0];
                let vnm1 = field[idx_nm1];
                field[idx_nm1] = v0 * cos_val;
                field[idx_0] = vnm1 * cos_val;
            }
        }
    }

    /// Apply Bloch BC in the z-direction to a single real-valued field.
    pub fn apply_z_phase(&self, field: &mut [f64]) {
        let pf = self.phase_factor_z();
        let cos_val = pf.re;
        for i in 0..self.nx {
            for j in 0..self.ny {
                let idx_0 = self.idx(i, j, 0);
                let idx_nm1 = self.idx(i, j, self.nz - 1);
                let v0 = field[idx_0];
                let vnm1 = field[idx_nm1];
                field[idx_nm1] = v0 * cos_val;
                field[idx_0] = vnm1 * cos_val;
            }
        }
    }

    /// Apply Bloch BCs in all three directions to all six field components.
    ///
    /// # Arguments
    /// * `ex`, `ey`, `ez` — electric field components (flat nx·ny·nz slices)
    /// * `hx`, `hy`, `hz` — magnetic field components (flat nx·ny·nz slices)
    pub fn apply_all(
        &self,
        ex: &mut [f64],
        ey: &mut [f64],
        ez: &mut [f64],
        hx: &mut [f64],
        hy: &mut [f64],
        hz: &mut [f64],
    ) {
        for field in [ex, ey, ez, hx, hy, hz] {
            self.apply_x_phase(field);
            self.apply_y_phase(field);
            self.apply_z_phase(field);
        }
    }

    // -----------------------------------------------------------------------
    // High-symmetry k-points — 2D square lattice
    // -----------------------------------------------------------------------

    /// Γ point: k = (0, 0, 0) for a square lattice with period `a`.
    pub fn square_lattice_gamma(a: f64, nx: usize, ny: usize, nz: usize) -> Self {
        let dx = a / nx as f64;
        let dy = a / ny as f64;
        let dz = dx; // arbitrary for 2D
        Self::new(0.0, 0.0, 0.0, nx, ny, nz, dx, dy, dz)
    }

    /// X point: k = (π/a, 0, 0) for a 2D square lattice.
    pub fn square_lattice_x(a: f64, nx: usize, ny: usize, nz: usize) -> Self {
        let dx = a / nx as f64;
        let dy = a / ny as f64;
        let dz = dx;
        Self::new(PI / a, 0.0, 0.0, nx, ny, nz, dx, dy, dz)
    }

    /// M point: k = (π/a, π/a, 0) for a 2D square lattice.
    pub fn square_lattice_m(a: f64, nx: usize, ny: usize, nz: usize) -> Self {
        let dx = a / nx as f64;
        let dy = a / ny as f64;
        let dz = dx;
        Self::new(PI / a, PI / a, 0.0, nx, ny, nz, dx, dy, dz)
    }

    // -----------------------------------------------------------------------
    // High-symmetry k-points — 3D FCC lattice
    // -----------------------------------------------------------------------
    // FCC reciprocal lattice vectors: b1 = 2π/a(-1,1,1), b2 = 2π/a(1,-1,1),
    //   b3 = 2π/a(1,1,-1).  High-symmetry points in Cartesian (kx,ky,kz):
    //   Γ = (0,0,0)
    //   X = (2π/a)(0,1,0)        = (0, 2π/a, 0)
    //   L = (2π/a)(1/2,1/2,1/2) = (π/a, π/a, π/a)
    //   W = (2π/a)(1/2,1,0)      = (π/a, 2π/a, 0)

    /// Γ point for a 3D FCC lattice with conventional lattice constant `a`.
    pub fn fcc_lattice_gamma(a: f64, nx: usize, ny: usize, nz: usize) -> Self {
        let d = a / nx as f64;
        Self::new(0.0, 0.0, 0.0, nx, ny, nz, d, d, d)
    }

    /// X point for a 3D FCC lattice: k = (0, 2π/a, 0).
    pub fn fcc_lattice_x(a: f64, nx: usize, ny: usize, nz: usize) -> Self {
        let d = a / nx as f64;
        Self::new(0.0, 2.0 * PI / a, 0.0, nx, ny, nz, d, d, d)
    }

    /// L point for a 3D FCC lattice: k = (π/a, π/a, π/a).
    pub fn fcc_lattice_l(a: f64, nx: usize, ny: usize, nz: usize) -> Self {
        let d = a / nx as f64;
        Self::new(PI / a, PI / a, PI / a, nx, ny, nz, d, d, d)
    }

    /// W point for a 3D FCC lattice: k = (π/a, 2π/a, 0).
    pub fn fcc_lattice_w(a: f64, nx: usize, ny: usize, nz: usize) -> Self {
        let d = a / nx as f64;
        Self::new(PI / a, 2.0 * PI / a, 0.0, nx, ny, nz, d, d, d)
    }
}

// ---------------------------------------------------------------------------
// Band structure calculator
// ---------------------------------------------------------------------------

/// Band structure calculator using Bloch FDTD.
///
/// Sweeps a sequence of k-vectors along a high-symmetry path, excites the
/// photonic crystal with a broad-band pulse, collects the time-domain signal,
/// and identifies resonant frequencies via DFT peak detection.
///
/// # Example
/// ```ignore
/// let path = BandStructureCalc::square_lattice_path(20, 500e-9);
/// let calc = BandStructureCalc::new(path, 512, (1e13, 1e15), 2000);
/// let bands = calc.compute(500e-9, |_x, _y, _z| 1.0);
/// ```
pub struct BandStructureCalc {
    /// Sequence of k-points to sweep, each as [kx, ky, kz] in rad/m.
    pub k_path: Vec<[f64; 3]>,
    /// Frequency resolution (number of DFT bins).
    pub n_freqs: usize,
    /// Frequency range (f_min, f_max) in Hz.
    pub freq_range: (f64, f64),
    /// Number of FDTD time steps per k-point.
    pub n_timesteps: usize,
}

impl BandStructureCalc {
    /// Create a new band structure calculator.
    pub fn new(
        k_path: Vec<[f64; 3]>,
        n_freqs: usize,
        freq_range: (f64, f64),
        n_timesteps: usize,
    ) -> Self {
        Self {
            k_path,
            n_freqs,
            freq_range,
            n_timesteps,
        }
    }

    /// Compute the photonic band structure.
    ///
    /// For each k-point in `self.k_path`, a minimal 3D FDTD simulation is run
    /// using the Bloch boundary condition.  A broadband Gaussian pulse excites
    /// the crystal; the time-domain signal at a probe point is DFT'd to find
    /// resonances.
    ///
    /// # Arguments
    /// * `lattice_const` — real-space lattice constant in metres
    /// * `eps_fn`        — dielectric function ε(x, y, z) in SI units
    ///
    /// # Returns
    /// `Vec<Vec<f64>>` — outer index = k-point index, inner = resonant frequencies (Hz).
    pub fn compute(
        &self,
        lattice_const: f64,
        eps_fn: impl Fn(f64, f64, f64) -> f64,
    ) -> Vec<Vec<f64>> {
        use crate::units::conversion::SPEED_OF_LIGHT;

        // Use a coarse grid that covers one unit cell
        let n_cell: usize = 16;
        let dx = lattice_const / n_cell as f64;
        let dy = dx;
        let dz = dx;
        let nx = n_cell;
        let ny = n_cell;
        let nz = n_cell;
        let n_total = nx * ny * nz;

        // Courant time step (3D stability: dt ≤ dx / (c √3))
        let dt = 0.5 * dx / (SPEED_OF_LIGHT * 3.0_f64.sqrt());

        // Build permittivity map
        let eps: Vec<f64> = (0..n_total)
            .map(|idx| {
                let i = idx / (ny * nz);
                let j = (idx / nz) % ny;
                let k = idx % nz;
                let x = (i as f64 + 0.5) * dx;
                let y = (j as f64 + 0.5) * dy;
                let z = (k as f64 + 0.5) * dz;
                eps_fn(x, y, z)
            })
            .collect();

        let mut results: Vec<Vec<f64>> = Vec::with_capacity(self.k_path.len());

        for &[kx, ky, kz] in &self.k_path {
            let bloch = BlochBc3d::new(kx, ky, kz, nx, ny, nz, dx, dy, dz);

            // Initialise fields
            let mut ex = vec![0.0_f64; n_total];
            let mut ey = vec![0.0_f64; n_total];
            let mut ez = vec![0.0_f64; n_total];
            let mut hx = vec![0.0_f64; n_total];
            let mut hy = vec![0.0_f64; n_total];
            let mut hz = vec![0.0_f64; n_total];

            // Gaussian pulse parameters (centred at t0, width tw)
            let f_centre = 0.5 * (self.freq_range.0 + self.freq_range.1);
            let tw = 3.0 / (2.0 * PI * f_centre); // ~3 cycles
            let t0 = 5.0 * tw;

            // Probe at grid centre
            let probe_idx = bloch.idx(nx / 2, ny / 2, nz / 2);

            let mut signal: Vec<f64> = Vec::with_capacity(self.n_timesteps);

            for step in 0..self.n_timesteps {
                let t = step as f64 * dt;

                // Gaussian source excitation (additive into Ez)
                let src =
                    (-0.5 * ((t - t0) / tw).powi(2)).exp() * (2.0 * PI * f_centre * (t - t0)).cos();
                let src_idx = bloch.idx(nx / 4, ny / 4, nz / 4);
                if src_idx < ez.len() {
                    ez[src_idx] += src;
                }

                // Simplified Yee update (isotropic, uniform mu=mu0)
                update_h_3d(
                    &mut hx, &mut hy, &mut hz, &ex, &ey, &ez, nx, ny, nz, dt, dx, dy, dz,
                );
                bloch.apply_all(&mut ex, &mut ey, &mut ez, &mut hx, &mut hy, &mut hz);

                update_e_3d(
                    &mut ex, &mut ey, &mut ez, &hx, &hy, &hz, &eps, nx, ny, nz, dt, dx, dy, dz,
                );
                bloch.apply_all(&mut ex, &mut ey, &mut ez, &mut hx, &mut hy, &mut hz);

                // Sample probe
                if probe_idx < ez.len() {
                    signal.push(ez[probe_idx]);
                }
            }

            let resonances = Self::find_resonances(&signal, dt, self.freq_range, self.n_freqs);
            results.push(resonances);
        }

        results
    }

    /// Find resonant frequencies from a time-domain signal via DFT peak finding.
    ///
    /// Computes the power spectrum via DFT over the given frequency range and
    /// returns all local-maximum frequencies (simple peak detection).
    ///
    /// # Arguments
    /// * `signal`     — time-domain samples
    /// * `dt`         — time step in seconds
    /// * `freq_range` — (f_min, f_max) in Hz
    /// * `n_freqs`    — number of frequency bins
    ///
    /// # Returns
    /// Frequencies (Hz) at which the power spectrum has local maxima.
    pub fn find_resonances(
        signal: &[f64],
        dt: f64,
        freq_range: (f64, f64),
        n_freqs: usize,
    ) -> Vec<f64> {
        if signal.is_empty() || n_freqs < 3 {
            return Vec::new();
        }

        let (f_min, f_max) = freq_range;
        let df = (f_max - f_min) / (n_freqs.saturating_sub(1).max(1)) as f64;

        // Compute power spectrum by direct DFT summation
        let power: Vec<f64> = (0..n_freqs)
            .map(|fi| {
                let freq = f_min + fi as f64 * df;
                let omega = 2.0 * PI * freq;
                let mut re = 0.0_f64;
                let mut im = 0.0_f64;
                for (n, &s) in signal.iter().enumerate() {
                    let phase = omega * n as f64 * dt;
                    re += s * phase.cos();
                    im -= s * phase.sin();
                }
                re * re + im * im
            })
            .collect();

        // Peak detection: local maxima (simple 1-neighbour comparison)
        let mut peaks: Vec<f64> = Vec::new();
        for i in 1..n_freqs.saturating_sub(1) {
            if power[i] > power[i - 1] && power[i] > power[i + 1] {
                let freq = f_min + i as f64 * df;
                peaks.push(freq);
            }
        }

        peaks
    }

    /// Standard Γ–X–M–Γ path for a 2D square lattice.
    ///
    /// # Arguments
    /// * `n_k` — number of k-points along the full path
    /// * `a`   — lattice constant in metres
    ///
    /// # Returns
    /// Vec of [kx, ky, kz] points along Γ→X→M→Γ.
    pub fn square_lattice_path(n_k: usize, a: f64) -> Vec<[f64; 3]> {
        // Divide n_k evenly across the three segments Γ→X, X→M, M→Γ
        let seg = (n_k / 3).max(1);
        let mut path: Vec<[f64; 3]> = Vec::with_capacity(3 * seg + 1);

        let gamma = [0.0_f64, 0.0, 0.0];
        let x_pt = [PI / a, 0.0, 0.0];
        let m_pt = [PI / a, PI / a, 0.0];

        // Γ → X
        for i in 0..seg {
            let t = i as f64 / seg as f64;
            path.push(lerp_kpt(gamma, x_pt, t));
        }
        // X → M
        for i in 0..seg {
            let t = i as f64 / seg as f64;
            path.push(lerp_kpt(x_pt, m_pt, t));
        }
        // M → Γ
        for i in 0..=seg {
            let t = i as f64 / seg as f64;
            path.push(lerp_kpt(m_pt, gamma, t));
        }

        path
    }

    /// Standard Γ–X–U|K–Γ–L–W path for the FCC Brillouin zone.
    ///
    /// Cartesian k-coordinates for conventional lattice constant `a`.
    pub fn fcc_bz_path(n_k: usize, a: f64) -> Vec<[f64; 3]> {
        let seg = (n_k / 5).max(1);
        let mut path: Vec<[f64; 3]> = Vec::with_capacity(5 * seg + 1);

        let gamma = [0.0_f64, 0.0, 0.0];
        let x_pt = [0.0, 2.0 * PI / a, 0.0];
        let k_pt = [3.0 * PI / (2.0 * a), 3.0 * PI / (2.0 * a), 0.0];
        let l_pt = [PI / a, PI / a, PI / a];
        let w_pt = [PI / a, 2.0 * PI / a, 0.0];

        for (start, end) in [
            (gamma, x_pt),
            (x_pt, k_pt),
            (k_pt, gamma),
            (gamma, l_pt),
            (l_pt, w_pt),
        ] {
            for i in 0..seg {
                let t = i as f64 / seg as f64;
                path.push(lerp_kpt(start, end, t));
            }
        }
        // Close path at W
        path.push(w_pt);

        path
    }

    /// Convert frequencies to normalised units: f·a/c.
    ///
    /// # Arguments
    /// * `freqs`         — band frequencies per k-point (Hz)
    /// * `lattice_const` — lattice constant in metres
    ///
    /// # Returns
    /// Same structure with frequencies replaced by dimensionless fa/c.
    pub fn normalized_frequencies(freqs: &[Vec<f64>], lattice_const: f64) -> Vec<Vec<f64>> {
        use crate::units::conversion::SPEED_OF_LIGHT;
        freqs
            .iter()
            .map(|band| {
                band.iter()
                    .map(|&f| f * lattice_const / SPEED_OF_LIGHT)
                    .collect()
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Internal Yee update helpers (simplified isotropic, no PML)
// ---------------------------------------------------------------------------

/// Update H-field components using the Yee scheme (central differences).
#[allow(clippy::too_many_arguments)]
fn update_h_3d(
    hx: &mut [f64],
    hy: &mut [f64],
    hz: &mut [f64],
    ex: &[f64],
    ey: &[f64],
    ez: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    dt: f64,
    dx: f64,
    dy: f64,
    dz: f64,
) {
    use crate::units::conversion::MU_0;
    let coeff = dt / MU_0;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let idx = i * ny * nz + j * nz + k;
                let jp1 = if j + 1 < ny { j + 1 } else { 0 };
                let kp1 = if k + 1 < nz { k + 1 } else { 0 };
                let ip1 = if i + 1 < nx { i + 1 } else { 0 };

                let dez_dy = (ez[i * ny * nz + jp1 * nz + k] - ez[idx]) / dy;
                let dey_dz = (ey[i * ny * nz + j * nz + kp1] - ey[idx]) / dz;
                let dex_dz = (ex[i * ny * nz + j * nz + kp1] - ex[idx]) / dz;
                let dez_dx = (ez[ip1 * ny * nz + j * nz + k] - ez[idx]) / dx;
                let dey_dx = (ey[ip1 * ny * nz + j * nz + k] - ey[idx]) / dx;
                let dex_dy = (ex[i * ny * nz + jp1 * nz + k] - ex[idx]) / dy;

                hx[idx] -= (dez_dy - dey_dz) * coeff;
                hy[idx] -= (dex_dz - dez_dx) * coeff;
                hz[idx] -= (dey_dx - dex_dy) * coeff;
            }
        }
    }
}

/// Update E-field components using the Yee scheme (central differences).
#[allow(clippy::too_many_arguments)]
fn update_e_3d(
    ex: &mut [f64],
    ey: &mut [f64],
    ez: &mut [f64],
    hx: &[f64],
    hy: &[f64],
    hz: &[f64],
    eps: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    dt: f64,
    dx: f64,
    dy: f64,
    dz: f64,
) {
    use crate::units::conversion::EPSILON_0;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let idx = i * ny * nz + j * nz + k;
                let jm1 = if j > 0 { j - 1 } else { ny - 1 };
                let km1 = if k > 0 { k - 1 } else { nz - 1 };
                let im1 = if i > 0 { i - 1 } else { nx - 1 };

                let eps_val = eps[idx].max(1.0);
                let eps_eff = EPSILON_0 * eps_val;

                let dhz_dy = (hz[idx] - hz[i * ny * nz + jm1 * nz + k]) / dy;
                let dhy_dz = (hy[idx] - hy[i * ny * nz + j * nz + km1]) / dz;
                let dhx_dz = (hx[idx] - hx[i * ny * nz + j * nz + km1]) / dz;
                let dhz_dx = (hz[idx] - hz[im1 * ny * nz + j * nz + k]) / dx;
                let dhy_dx = (hy[idx] - hy[im1 * ny * nz + j * nz + k]) / dx;
                let dhx_dy = (hx[idx] - hx[i * ny * nz + jm1 * nz + k]) / dy;

                ex[idx] += dt / eps_eff * (dhz_dy - dhy_dz);
                ey[idx] += dt / eps_eff * (dhx_dz - dhz_dx);
                ez[idx] += dt / eps_eff * (dhy_dx - dhx_dy);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Linear interpolation between two k-points.
fn lerp_kpt(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_bloch_bc3d_creation() {
        // k = (0,0,0): all phase factors should equal 1+0i
        let bc = BlochBc3d::new(0.0, 0.0, 0.0, 32, 32, 32, 10e-9, 10e-9, 10e-9);
        let pfx = bc.phase_factor_x();
        let pfy = bc.phase_factor_y();
        let pfz = bc.phase_factor_z();

        assert!((pfx.re - 1.0).abs() < TOL, "pfx.re={}", pfx.re);
        assert!(pfx.im.abs() < TOL, "pfx.im={}", pfx.im);
        assert!((pfy.re - 1.0).abs() < TOL, "pfy.re={}", pfy.re);
        assert!(pfy.im.abs() < TOL, "pfy.im={}", pfy.im);
        assert!((pfz.re - 1.0).abs() < TOL, "pfz.re={}", pfz.re);
        assert!(pfz.im.abs() < TOL, "pfz.im={}", pfz.im);
    }

    #[test]
    fn test_phase_factor_pi() {
        // kx · nx · dx = kx · Lx = π → phase_factor_x = exp(iπ) = -1+0i
        let nx = 50;
        let dx = 10e-9;
        let lx = nx as f64 * dx;
        let kx = PI / lx;
        let bc = BlochBc3d::new(kx, 0.0, 0.0, nx, 32, 32, dx, dx, dx);
        let pf = bc.phase_factor_x();
        assert!((pf.re - (-1.0)).abs() < TOL, "pf.re={}", pf.re);
        assert!(pf.im.abs() < TOL, "pf.im={}", pf.im);
    }

    #[test]
    fn test_square_lattice_path() {
        let a = 500e-9;
        let n_k = 30;
        let path = BandStructureCalc::square_lattice_path(n_k, a);

        // Path must be non-empty
        assert!(!path.is_empty());

        // First point should be Γ = (0, 0, 0)
        let gamma = path[0];
        assert!(gamma[0].abs() < TOL, "First k-point should be Γ (kx=0)");
        assert!(gamma[1].abs() < TOL, "First k-point should be Γ (ky=0)");
        assert!(gamma[2].abs() < TOL, "First k-point should be Γ (kz=0)");

        // Last point should be Γ again (M → Γ)
        let last = path[path.len() - 1];
        assert!(last[0].abs() < TOL, "Last k-point should be Γ (kx=0)");
        assert!(last[1].abs() < TOL, "Last k-point should be Γ (ky=0)");

        // Check that the X point (π/a, 0, 0) appears somewhere in path
        let has_x = path
            .iter()
            .any(|&pt| (pt[0] - PI / a).abs() < 1e-6 * PI / a && pt[1].abs() < 1e-6 * PI / a);
        assert!(has_x, "X point (π/a, 0, 0) not found in path");

        // Check that the M point (π/a, π/a, 0) appears somewhere
        let has_m = path.iter().any(|&pt| {
            (pt[0] - PI / a).abs() < 1e-6 * PI / a && (pt[1] - PI / a).abs() < 1e-6 * PI / a
        });
        assert!(has_m, "M point (π/a, π/a, 0) not found in path");
    }

    #[test]
    fn test_band_structure_find_resonances() {
        // Generate a pure sinusoidal signal at a known frequency
        let dt = 1e-15; // 1 fs
        let f_known = 3.0e14; // 300 THz (≈ 1 µm)
        let n_samples = 4096;
        let signal: Vec<f64> = (0..n_samples)
            .map(|n| (2.0 * PI * f_known * n as f64 * dt).sin())
            .collect();

        let freq_range = (1.0e14, 6.0e14);
        let n_freqs = 256;
        let peaks = BandStructureCalc::find_resonances(&signal, dt, freq_range, n_freqs);

        // At least one peak should be found
        assert!(
            !peaks.is_empty(),
            "No resonances found in sinusoidal signal"
        );

        // The closest peak to f_known should be within the frequency resolution
        let df = (freq_range.1 - freq_range.0) / (n_freqs - 1) as f64;
        let closest = peaks
            .iter()
            .map(|&f| (f - f_known).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(
            closest < 5.0 * df,
            "Closest resonance {:.3e} Hz is more than 5 bins from known freq {:.3e} Hz",
            f_known - closest,
            f_known
        );
    }

    #[test]
    fn test_normalized_frequencies() {
        use crate::units::conversion::SPEED_OF_LIGHT;
        let a = 500e-9_f64;
        let f = 1.0e14_f64; // Hz
        let expected = f * a / SPEED_OF_LIGHT;

        let bands = vec![vec![f, 2.0 * f], vec![3.0 * f]];
        let norm = BandStructureCalc::normalized_frequencies(&bands, a);

        assert_eq!(norm.len(), 2);
        assert_eq!(norm[0].len(), 2);
        assert!((norm[0][0] - expected).abs() < 1e-12);
        assert!((norm[0][1] - 2.0 * expected).abs() < 1e-12);
        assert!((norm[1][0] - 3.0 * expected).abs() < 1e-12);
    }
}
