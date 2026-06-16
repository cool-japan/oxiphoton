//! Total-Field/Scattered-Field (TF/SF) formulation for FDTD.
//!
//! TF/SF separates the computational domain into:
//!   - Total-Field (TF) region: contains incident + scattered fields
//!   - Scattered-Field (SF) region: contains only scattered fields
//!
//! The TF/SF interface corrects field updates at the boundary:
//!   H at TF/SF boundary += ±E_inc / (μ₀·c·dz)
//!   E at TF/SF boundary += ±H_inc · dt / (ε₀·dz)
//!
//! This allows clean plane-wave injection without PML reflections.
//!
//! For 1D TE:
//!   At i = i_left (SF→TF):  H\[i-1\] -= E_inc / (μ₀·c·dz)·dt/dx
//!   At i = i_right (TF→SF): H\[i\]   += E_inc / (μ₀·c·dz)·dt/dx

use crate::fdtd::source::plane_wave::{GaussianEnvelope3d, Polarization3d, PropagationAxis};
use std::f64::consts::PI;

const EPS0: f64 = 8.854_187_817e-12;
const MU0: f64 = 1.256_637_061_4e-6;
const C: f64 = 2.997_924_58e8;

/// TF/SF configuration for 1D FDTD.
#[derive(Debug, Clone, Copy)]
pub struct TfsfConfig1d {
    /// Left boundary of TF region (grid index)
    pub i_left: usize,
    /// Right boundary of TF region (grid index)
    pub i_right: usize,
    /// Incident wave center frequency (Hz)
    pub f0: f64,
    /// Incident wave Gaussian pulse width (s); 0 = CW
    pub tau: f64,
    /// Time offset (s)
    pub t0: f64,
    /// Incident amplitude (V/m)
    pub amplitude: f64,
}

impl TfsfConfig1d {
    /// Create TF/SF with Gaussian pulse.
    pub fn gaussian(i_left: usize, i_right: usize, f0: f64, tau: f64, amplitude: f64) -> Self {
        let t0 = 3.0 * tau;
        Self {
            i_left,
            i_right,
            f0,
            tau,
            t0,
            amplitude,
        }
    }

    /// Create TF/SF with CW source.
    pub fn cw(i_left: usize, i_right: usize, f0: f64, amplitude: f64) -> Self {
        Self {
            i_left,
            i_right,
            f0,
            tau: 0.0,
            t0: 0.0,
            amplitude,
        }
    }

    /// Incident E-field at time t (in the TF region).
    pub fn e_inc(&self, t: f64) -> f64 {
        let envelope = if self.tau > 0.0 {
            let dt = t - self.t0;
            (-(dt / self.tau).powi(2)).exp()
        } else {
            1.0
        };
        let phase = 2.0 * PI * self.f0 * (t - self.t0);
        self.amplitude * envelope * phase.sin()
    }

    /// Incident H-field at time t.
    ///
    /// In free space: H_inc = E_inc / Z₀  where Z₀ = 377 Ω
    pub fn h_inc(&self, t: f64) -> f64 {
        self.e_inc(t) / 377.0
    }

    /// Apply TF/SF correction to E-field array (at H→E update step).
    ///
    /// Must be called AFTER the regular E update.
    pub fn apply_e_correction(&self, ex: &mut [f64], t: f64, dt: f64, dz: f64) {
        let h_inc = self.h_inc(t);
        let coeff = dt / (EPS0 * dz);
        // At i_left (SF→TF boundary): E[i_left] needs +H_inc correction
        if self.i_left < ex.len() {
            ex[self.i_left] += coeff * h_inc;
        }
        // At i_right+1 (TF→SF boundary): E[i_right+1] needs -H_inc correction
        if self.i_right + 1 < ex.len() {
            ex[self.i_right + 1] -= coeff * h_inc;
        }
    }

    /// Apply TF/SF correction to H-field array (at E→H update step).
    ///
    /// Must be called AFTER the regular H update.
    pub fn apply_h_correction(&self, hy: &mut [f64], t: f64, dt: f64, dz: f64) {
        let e_inc = self.e_inc(t);
        let coeff = dt / (MU0 * dz);
        // At i_left-1 (SF→TF): H[i_left-1] needs -E_inc correction
        if self.i_left > 0 {
            hy[self.i_left - 1] -= coeff * e_inc;
        }
        // At i_right (TF→SF): H[i_right] needs +E_inc correction
        if self.i_right < hy.len() {
            hy[self.i_right] += coeff * e_inc;
        }
    }
}

/// 2D TF/SF box configuration.
///
/// The TF region is a rectangular box; the SF surrounds it.
#[derive(Debug, Clone, Copy)]
pub struct TfsfBox2d {
    pub ix_left: usize,
    pub ix_right: usize,
    pub iy_bottom: usize,
    pub iy_top: usize,
    /// Incidence angle from normal (rad, 0 = normal incidence)
    pub theta_inc: f64,
    pub f0: f64,
    pub tau: f64,
    pub t0: f64,
    pub amplitude: f64,
}

impl TfsfBox2d {
    /// Create a 2D TF/SF box with normal incidence.
    pub fn normal_incidence(
        ix_left: usize,
        ix_right: usize,
        iy_bottom: usize,
        iy_top: usize,
        f0: f64,
        tau: f64,
        amplitude: f64,
    ) -> Self {
        let t0 = if tau > 0.0 { 3.0 * tau } else { 0.0 };
        Self {
            ix_left,
            ix_right,
            iy_bottom,
            iy_top,
            theta_inc: 0.0,
            f0,
            tau,
            t0,
            amplitude,
        }
    }

    /// Incident E-field at position (x, y) and time t (normal incidence, propagating in +x).
    pub fn e_inc_at(&self, x: f64, t: f64) -> f64 {
        let t_ret = t - x / C; // retarded time
        let envelope = if self.tau > 0.0 {
            let dt = t_ret - self.t0;
            (-(dt / self.tau).powi(2)).exp()
        } else {
            1.0
        };
        let phase = 2.0 * PI * self.f0 * (t_ret - self.t0);
        self.amplitude * envelope * phase.sin()
    }

    /// Width of TF region (in grid cells).
    pub fn width(&self) -> usize {
        self.ix_right.saturating_sub(self.ix_left)
    }

    /// Height of TF region (in grid cells).
    pub fn height(&self) -> usize {
        self.iy_top.saturating_sub(self.iy_bottom)
    }
}

/// TF/SF source tracker: maintains 1D auxiliary grid for incident field.
///
/// The auxiliary grid propagates only the incident wave in free space,
/// providing exact values for TF/SF corrections.
pub struct TfsfAux1d {
    /// Auxiliary E-field (incident only)
    pub e_aux: Vec<f64>,
    /// Auxiliary H-field (incident only)
    pub h_aux: Vec<f64>,
    /// Grid spacing (m)
    pub dz: f64,
    /// Time step (s)
    pub dt: f64,
    /// Source index in auxiliary grid
    pub src_i: usize,
    pub f0: f64,
    pub tau: f64,
    pub t0: f64,
    pub amplitude: f64,
    pub step: usize,
}

impl TfsfAux1d {
    /// Create auxiliary grid for TF/SF.
    pub fn new(n: usize, dz: f64, f0: f64, tau: f64, amplitude: f64) -> Self {
        let dt = 0.99 * dz / C;
        let t0 = if tau > 0.0 { 3.0 * tau } else { 0.0 };
        Self {
            e_aux: vec![0.0; n],
            h_aux: vec![0.0; n],
            dz,
            dt,
            src_i: n / 4,
            f0,
            tau,
            t0,
            amplitude,
            step: 0,
        }
    }

    /// Advance auxiliary grid one step.
    pub fn advance(&mut self) {
        let n = self.e_aux.len();
        // H update
        for i in 0..n - 1 {
            self.h_aux[i] -= self.dt / (MU0 * self.dz) * (self.e_aux[i + 1] - self.e_aux[i]);
        }
        // E update
        for i in 1..n {
            self.e_aux[i] += self.dt / (EPS0 * self.dz) * (self.h_aux[i] - self.h_aux[i - 1]);
        }
        // Inject source
        let t = self.step as f64 * self.dt;
        let envelope = if self.tau > 0.0 {
            let dt = t - self.t0;
            (-(dt / self.tau).powi(2)).exp()
        } else {
            1.0
        };
        let phase = 2.0 * PI * self.f0 * (t - self.t0);
        let src = self.amplitude * envelope * phase.sin();
        if self.src_i < n {
            self.e_aux[self.src_i] += src;
        }
        self.step += 1;
    }

    /// E-field at source position in auxiliary grid.
    pub fn e_at_src(&self) -> f64 {
        self.e_aux[self.src_i]
    }
}

// ─────────────────────────────────────────────────────────────────
// 3D TFSF Source
// ─────────────────────────────────────────────────────────────────

/// 3D TF/SF source.
///
/// Separates the computational domain into a Total-Field inner box and
/// Scattered-Field outer region. Uses a 1D auxiliary FDTD grid to track
/// the reference (incident) wave, and applies correction currents to the
/// six faces of the TFSF box at each time step.
///
/// The correction at each face adds or subtracts the auxiliary-grid
/// incident field to "peel off" the incident wave, leaving only
/// scattered fields in the SF region.
///
/// Indexing convention: all 3D field arrays use row-major order
///   `field[i * ny * nz + j * nz + k]`
pub struct TfsfSource3d {
    /// Inner boundary of TFSF region
    pub i_min: usize,
    pub j_min: usize,
    pub k_min: usize,
    /// Outer boundary of TFSF region
    pub i_max: usize,
    pub j_max: usize,
    pub k_max: usize,
    /// Propagation direction of incident wave
    pub axis: PropagationAxis,
    /// Polarization of incident E-field
    pub polarization: Polarization3d,
    /// Peak amplitude (V/m)
    pub amplitude: f64,
    /// Angular frequency (rad/s)
    pub omega: f64,
    /// Optional Gaussian pulse envelope
    pub envelope: Option<GaussianEnvelope3d>,

    // 1D auxiliary FDTD grid (Ez/Hy polarization by default)
    aux_ez: Vec<f64>,
    aux_hy: Vec<f64>,
    /// Auxiliary grid spacing (m)
    pub aux_dz: f64,
    /// Auxiliary grid time step (s)
    pub aux_dt: f64,
    /// Number of auxiliary grid cells
    pub naux: usize,
    /// Source position in auxiliary grid
    pub aux_source_pos: usize,
    /// Current simulation time step
    time_step: usize,
}

impl TfsfSource3d {
    /// Create a new 3D TF/SF source.
    ///
    /// # Arguments
    /// * `i_min..i_max`, `j_min..j_max`, `k_min..k_max` — TFSF box boundaries (inclusive)
    /// * `axis` — incident wave propagation direction
    /// * `polarization` — E-field polarization
    /// * `amplitude` — peak amplitude (V/m)
    /// * `omega` — angular frequency (rad/s)
    /// * `dz` — grid spacing along the propagation axis (m)
    /// * `dt` — simulation time step (s)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        i_min: usize,
        j_min: usize,
        k_min: usize,
        i_max: usize,
        j_max: usize,
        k_max: usize,
        axis: PropagationAxis,
        polarization: Polarization3d,
        amplitude: f64,
        omega: f64,
        dz: f64,
        _dt: f64,
    ) -> Self {
        // Size the auxiliary grid to span the TFSF region with margin
        let span = match axis.axis_index() {
            0 => i_max.saturating_sub(i_min) + 20,
            1 => j_max.saturating_sub(j_min) + 20,
            _ => k_max.saturating_sub(k_min) + 20,
        };
        let naux = (span + 2).max(64);
        let aux_dt = 0.99 * dz / C;
        let aux_source_pos = naux / 4;

        Self {
            i_min,
            j_min,
            k_min,
            i_max,
            j_max,
            k_max,
            axis,
            polarization,
            amplitude,
            omega,
            envelope: None,
            aux_ez: vec![0.0; naux],
            aux_hy: vec![0.0; naux],
            aux_dz: dz,
            aux_dt,
            naux,
            aux_source_pos,
            time_step: 0,
        }
    }

    /// Add a Gaussian pulse envelope to the TFSF source.
    pub fn with_gaussian_pulse(mut self, t0: f64, sigma: f64) -> Self {
        self.envelope = Some(GaussianEnvelope3d::new(t0, sigma));
        self
    }

    /// Compute incident E-field from auxiliary grid at a given index offset from source.
    fn e_inc_at_offset(&self, offset_cells: isize) -> f64 {
        let idx = (self.aux_source_pos as isize + offset_cells) as usize;
        if idx < self.naux {
            self.aux_ez[idx]
        } else {
            0.0
        }
    }

    /// Compute incident H-field from auxiliary grid at a given index offset from source.
    fn h_inc_at_offset(&self, offset_cells: isize) -> f64 {
        let idx = (self.aux_source_pos as isize + offset_cells) as usize;
        if idx < self.naux {
            self.aux_hy[idx]
        } else {
            0.0
        }
    }

    /// Step the 1D auxiliary FDTD grid one time step and inject the source.
    pub fn step_aux(&mut self) {
        let n = self.naux;
        // H update (Leapfrog)
        for i in 0..n - 1 {
            self.aux_hy[i] -=
                self.aux_dt / (MU0 * self.aux_dz) * (self.aux_ez[i + 1] - self.aux_ez[i]);
        }
        // E update
        for i in 1..n {
            self.aux_ez[i] +=
                self.aux_dt / (EPS0 * self.aux_dz) * (self.aux_hy[i] - self.aux_hy[i - 1]);
        }
        // Inject source at aux_source_pos
        let t = self.time_step as f64 * self.aux_dt;
        let env = match &self.envelope {
            Some(g) => g.evaluate(t),
            None => 1.0,
        };
        let src = self.amplitude * env * (self.omega * t).sin();
        if self.aux_source_pos < n {
            self.aux_ez[self.aux_source_pos] += src;
        }
        self.time_step += 1;
    }

    /// Flat index for 3D array with layout [i][j]\[k\] → i*ny*nz + j*nz + k.
    #[inline]
    fn idx(i: usize, j: usize, k: usize, ny: usize, nz: usize) -> usize {
        i * ny * nz + j * nz + k
    }

    /// Apply E-field corrections at the six TFSF boundary faces.
    ///
    /// Must be called AFTER the regular 3D E-field update.
    /// For each face, the incident field from the 1D aux grid is used to
    /// subtract the incident contribution from the SF-side cells.
    ///
    /// The correction sign and field component depends on the incident polarization
    /// and the face orientation (inward/outward normal vs. propagation).
    #[allow(clippy::too_many_arguments)]
    pub fn apply_e_correction(
        &self,
        ex: &mut [f64],
        ey: &mut [f64],
        ez: &mut [f64],
        nx: usize,
        ny: usize,
        nz: usize,
        _dt: f64,
        dx: f64,
        dy: f64,
        dz: f64,
    ) {
        // The correction coefficient converts the auxiliary H-field to an E-field correction
        // at the TFSF face. Coefficient is dt/(eps0 * grid_spacing_perpendicular).
        // We use the aux_dt (which should match main dt) and the perpendicular cell spacing.

        let coeff_x = self.aux_dt / (EPS0 * dx);
        let coeff_y = self.aux_dt / (EPS0 * dy);
        let coeff_z = self.aux_dt / (EPS0 * dz);

        // Offset in the aux grid: the offset between aux source and the face
        // We use a simple approximation: the incident field at face = aux_ez[src_pos]
        // (or appropriately offset for propagation distance).
        // For correctness in a simple implementation, we use e_inc_at_offset(0) as
        // the "current" incident field, and h_inc_at_offset for H complement.

        let e_inc = self.e_inc_at_offset(0);
        let h_inc = self.h_inc_at_offset(0);

        // E-field corrections: at each face, apply ±h_inc * coeff to the tangential E
        // The sign convention follows Taflove & Hagness Ch.5 TF/SF.

        // Face k = k_min (bottom XY face, normal = -Z, TF side is above)
        if self.k_min < nz {
            for i in self.i_min..=self.i_max.min(nx.saturating_sub(1)) {
                for j in self.j_min..=self.j_max.min(ny.saturating_sub(1)) {
                    let idx = Self::idx(i, j, self.k_min, ny, nz);
                    // Correction for tangential E components at bottom face
                    match self.polarization {
                        Polarization3d::Ex => {
                            if idx < ex.len() {
                                ex[idx] -= coeff_z * h_inc;
                            }
                        }
                        Polarization3d::Ey => {
                            if idx < ey.len() {
                                ey[idx] += coeff_z * h_inc;
                            }
                        }
                        Polarization3d::Ez => {
                            if idx < ez.len() {
                                ez[idx] += coeff_x * e_inc;
                            }
                        }
                    }
                }
            }
        }

        // Face k = k_max+1 (top XY face, normal = +Z, SF side is above)
        let k_top = self.k_max + 1;
        if k_top < nz {
            for i in self.i_min..=self.i_max.min(nx.saturating_sub(1)) {
                for j in self.j_min..=self.j_max.min(ny.saturating_sub(1)) {
                    let idx = Self::idx(i, j, k_top, ny, nz);
                    match self.polarization {
                        Polarization3d::Ex => {
                            if idx < ex.len() {
                                ex[idx] += coeff_z * h_inc;
                            }
                        }
                        Polarization3d::Ey => {
                            if idx < ey.len() {
                                ey[idx] -= coeff_z * h_inc;
                            }
                        }
                        Polarization3d::Ez => {
                            if idx < ez.len() {
                                ez[idx] -= coeff_x * e_inc;
                            }
                        }
                    }
                }
            }
        }

        // Face j = j_min (front XZ face, normal = -Y)
        if self.j_min < ny {
            for i in self.i_min..=self.i_max.min(nx.saturating_sub(1)) {
                for k in self.k_min..=self.k_max.min(nz.saturating_sub(1)) {
                    let idx = Self::idx(i, self.j_min, k, ny, nz);
                    match self.polarization {
                        Polarization3d::Ex => {
                            if idx < ex.len() {
                                ex[idx] += coeff_y * e_inc;
                            }
                        }
                        Polarization3d::Ey => {}
                        Polarization3d::Ez => {
                            if idx < ez.len() {
                                ez[idx] -= coeff_y * h_inc;
                            }
                        }
                    }
                }
            }
        }

        // Face j = j_max+1 (back XZ face, normal = +Y)
        let j_back = self.j_max + 1;
        if j_back < ny {
            for i in self.i_min..=self.i_max.min(nx.saturating_sub(1)) {
                for k in self.k_min..=self.k_max.min(nz.saturating_sub(1)) {
                    let idx = Self::idx(i, j_back, k, ny, nz);
                    match self.polarization {
                        Polarization3d::Ex => {
                            if idx < ex.len() {
                                ex[idx] -= coeff_y * e_inc;
                            }
                        }
                        Polarization3d::Ey => {}
                        Polarization3d::Ez => {
                            if idx < ez.len() {
                                ez[idx] += coeff_y * h_inc;
                            }
                        }
                    }
                }
            }
        }

        // Face i = i_min (left YZ face, normal = -X)
        if self.i_min < nx {
            for j in self.j_min..=self.j_max.min(ny.saturating_sub(1)) {
                for k in self.k_min..=self.k_max.min(nz.saturating_sub(1)) {
                    let idx = Self::idx(self.i_min, j, k, ny, nz);
                    match self.polarization {
                        Polarization3d::Ex => {}
                        Polarization3d::Ey => {
                            if idx < ey.len() {
                                ey[idx] -= coeff_x * e_inc;
                            }
                        }
                        Polarization3d::Ez => {
                            if idx < ez.len() {
                                ez[idx] += coeff_x * h_inc;
                            }
                        }
                    }
                }
            }
        }

        // Face i = i_max+1 (right YZ face, normal = +X)
        let i_right = self.i_max + 1;
        if i_right < nx {
            for j in self.j_min..=self.j_max.min(ny.saturating_sub(1)) {
                for k in self.k_min..=self.k_max.min(nz.saturating_sub(1)) {
                    let idx = Self::idx(i_right, j, k, ny, nz);
                    match self.polarization {
                        Polarization3d::Ex => {}
                        Polarization3d::Ey => {
                            if idx < ey.len() {
                                ey[idx] += coeff_x * e_inc;
                            }
                        }
                        Polarization3d::Ez => {
                            if idx < ez.len() {
                                ez[idx] -= coeff_x * h_inc;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Apply H-field corrections at the six TFSF boundary faces.
    ///
    /// Must be called AFTER the regular 3D H-field update.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_h_correction(
        &self,
        hx: &mut [f64],
        hy: &mut [f64],
        hz: &mut [f64],
        nx: usize,
        ny: usize,
        nz: usize,
        _dt: f64,
        dx: f64,
        dy: f64,
        dz: f64,
    ) {
        let coeff_x = self.aux_dt / (MU0 * dx);
        let coeff_y = self.aux_dt / (MU0 * dy);
        let coeff_z = self.aux_dt / (MU0 * dz);

        let e_inc = self.e_inc_at_offset(0);
        let h_inc = self.h_inc_at_offset(0);

        // Face k = k_min (bottom XY face)
        if self.k_min > 0 {
            let k = self.k_min - 1;
            for i in self.i_min..=self.i_max.min(nx.saturating_sub(1)) {
                for j in self.j_min..=self.j_max.min(ny.saturating_sub(1)) {
                    let idx = Self::idx(i, j, k, ny, nz);
                    match self.polarization {
                        Polarization3d::Ex => {
                            if idx < hx.len() {
                                hx[idx] -= coeff_z * e_inc;
                            }
                        }
                        Polarization3d::Ey => {
                            if idx < hy.len() {
                                hy[idx] += coeff_z * e_inc;
                            }
                        }
                        Polarization3d::Ez => {
                            if idx < hz.len() {
                                hz[idx] -= coeff_x * h_inc;
                            }
                        }
                    }
                }
            }
        }

        // Face k = k_max (top XY face)
        let k = self.k_max.min(nz.saturating_sub(1));
        for i in self.i_min..=self.i_max.min(nx.saturating_sub(1)) {
            for j in self.j_min..=self.j_max.min(ny.saturating_sub(1)) {
                let idx = Self::idx(i, j, k, ny, nz);
                match self.polarization {
                    Polarization3d::Ex => {
                        if idx < hx.len() {
                            hx[idx] += coeff_z * e_inc;
                        }
                    }
                    Polarization3d::Ey => {
                        if idx < hy.len() {
                            hy[idx] -= coeff_z * e_inc;
                        }
                    }
                    Polarization3d::Ez => {
                        if idx < hz.len() {
                            hz[idx] += coeff_x * h_inc;
                        }
                    }
                }
            }
        }

        // Face j = j_min (front XZ face)
        if self.j_min > 0 {
            let j = self.j_min - 1;
            for i in self.i_min..=self.i_max.min(nx.saturating_sub(1)) {
                for k in self.k_min..=self.k_max.min(nz.saturating_sub(1)) {
                    let idx = Self::idx(i, j, k, ny, nz);
                    match self.polarization {
                        Polarization3d::Ex => {
                            if idx < hx.len() {
                                hx[idx] -= coeff_y * h_inc;
                            }
                        }
                        Polarization3d::Ey => {}
                        Polarization3d::Ez => {
                            if idx < hz.len() {
                                hz[idx] += coeff_y * e_inc;
                            }
                        }
                    }
                }
            }
        }

        // Face j = j_max
        let j = self.j_max.min(ny.saturating_sub(1));
        for i in self.i_min..=self.i_max.min(nx.saturating_sub(1)) {
            for k in self.k_min..=self.k_max.min(nz.saturating_sub(1)) {
                let idx = Self::idx(i, j, k, ny, nz);
                match self.polarization {
                    Polarization3d::Ex => {
                        if idx < hx.len() {
                            hx[idx] += coeff_y * h_inc;
                        }
                    }
                    Polarization3d::Ey => {}
                    Polarization3d::Ez => {
                        if idx < hz.len() {
                            hz[idx] -= coeff_y * e_inc;
                        }
                    }
                }
            }
        }

        // Face i = i_min (left YZ face)
        if self.i_min > 0 {
            let i = self.i_min - 1;
            for j in self.j_min..=self.j_max.min(ny.saturating_sub(1)) {
                for k in self.k_min..=self.k_max.min(nz.saturating_sub(1)) {
                    let idx = Self::idx(i, j, k, ny, nz);
                    match self.polarization {
                        Polarization3d::Ex => {}
                        Polarization3d::Ey => {
                            if idx < hy.len() {
                                hy[idx] -= coeff_x * h_inc;
                            }
                        }
                        Polarization3d::Ez => {
                            if idx < hz.len() {
                                hz[idx] += coeff_x * e_inc;
                            }
                        }
                    }
                }
            }
        }

        // Face i = i_max
        let i = self.i_max.min(nx.saturating_sub(1));
        for j in self.j_min..=self.j_max.min(ny.saturating_sub(1)) {
            for k in self.k_min..=self.k_max.min(nz.saturating_sub(1)) {
                let idx = Self::idx(i, j, k, ny, nz);
                match self.polarization {
                    Polarization3d::Ex => {}
                    Polarization3d::Ey => {
                        if idx < hy.len() {
                            hy[idx] += coeff_x * h_inc;
                        }
                    }
                    Polarization3d::Ez => {
                        if idx < hz.len() {
                            hz[idx] -= coeff_x * e_inc;
                        }
                    }
                }
            }
        }
    }

    /// Flat index for 3D array with k-major layout: k*(nx*ny) + j*nx + i.
    ///
    /// Used by [`apply_e_correction_kfirst`] and [`apply_h_correction_kfirst`]
    /// to match the indexing convention of [`crate::fdtd::Fdtd3d`].
    #[inline]
    fn idx_kfirst(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
        k * (nx * ny) + j * nx + i
    }

    /// Apply E-field corrections using k-major array layout (k*(nx*ny)+j*nx+i).
    ///
    /// Equivalent to [`TfsfSource3d::apply_e_correction`] but uses the k-major indexing
    /// convention of [`crate::fdtd::Fdtd3d`] rather than the default i-major
    /// convention.  Must be called AFTER the regular 3D E-field update.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_e_correction_kfirst(
        &self,
        ex: &mut [f64],
        ey: &mut [f64],
        ez: &mut [f64],
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
    ) {
        let coeff_x = self.aux_dt / (EPS0 * dx);
        let coeff_y = self.aux_dt / (EPS0 * dy);
        let coeff_z = self.aux_dt / (EPS0 * dz);
        let e_inc = self.e_inc_at_offset(0);
        let h_inc = self.h_inc_at_offset(0);

        // Face k = k_min (bottom XY face, normal = -Z, TF side is above)
        if self.k_min < nz {
            for i in self.i_min..=self.i_max.min(nx.saturating_sub(1)) {
                for j in self.j_min..=self.j_max.min(ny.saturating_sub(1)) {
                    let idx = Self::idx_kfirst(i, j, self.k_min, nx, ny);
                    match self.polarization {
                        Polarization3d::Ex => {
                            if idx < ex.len() {
                                ex[idx] -= coeff_z * h_inc;
                            }
                        }
                        Polarization3d::Ey => {
                            if idx < ey.len() {
                                ey[idx] += coeff_z * h_inc;
                            }
                        }
                        Polarization3d::Ez => {
                            if idx < ez.len() {
                                ez[idx] += coeff_x * e_inc;
                            }
                        }
                    }
                }
            }
        }

        // Face k = k_max+1 (top XY face, normal = +Z, SF side is above)
        let k_top = self.k_max + 1;
        if k_top < nz {
            for i in self.i_min..=self.i_max.min(nx.saturating_sub(1)) {
                for j in self.j_min..=self.j_max.min(ny.saturating_sub(1)) {
                    let idx = Self::idx_kfirst(i, j, k_top, nx, ny);
                    match self.polarization {
                        Polarization3d::Ex => {
                            if idx < ex.len() {
                                ex[idx] += coeff_z * h_inc;
                            }
                        }
                        Polarization3d::Ey => {
                            if idx < ey.len() {
                                ey[idx] -= coeff_z * h_inc;
                            }
                        }
                        Polarization3d::Ez => {
                            if idx < ez.len() {
                                ez[idx] -= coeff_x * e_inc;
                            }
                        }
                    }
                }
            }
        }

        // Face j = j_min (front XZ face, normal = -Y)
        if self.j_min < ny {
            for i in self.i_min..=self.i_max.min(nx.saturating_sub(1)) {
                for k in self.k_min..=self.k_max.min(nz.saturating_sub(1)) {
                    let idx = Self::idx_kfirst(i, self.j_min, k, nx, ny);
                    match self.polarization {
                        Polarization3d::Ex => {
                            if idx < ex.len() {
                                ex[idx] += coeff_y * e_inc;
                            }
                        }
                        Polarization3d::Ey => {}
                        Polarization3d::Ez => {
                            if idx < ez.len() {
                                ez[idx] -= coeff_y * h_inc;
                            }
                        }
                    }
                }
            }
        }

        // Face j = j_max+1 (back XZ face, normal = +Y)
        let j_back = self.j_max + 1;
        if j_back < ny {
            for i in self.i_min..=self.i_max.min(nx.saturating_sub(1)) {
                for k in self.k_min..=self.k_max.min(nz.saturating_sub(1)) {
                    let idx = Self::idx_kfirst(i, j_back, k, nx, ny);
                    match self.polarization {
                        Polarization3d::Ex => {
                            if idx < ex.len() {
                                ex[idx] -= coeff_y * e_inc;
                            }
                        }
                        Polarization3d::Ey => {}
                        Polarization3d::Ez => {
                            if idx < ez.len() {
                                ez[idx] += coeff_y * h_inc;
                            }
                        }
                    }
                }
            }
        }

        // Face i = i_min (left YZ face, normal = -X)
        if self.i_min < nx {
            for j in self.j_min..=self.j_max.min(ny.saturating_sub(1)) {
                for k in self.k_min..=self.k_max.min(nz.saturating_sub(1)) {
                    let idx = Self::idx_kfirst(self.i_min, j, k, nx, ny);
                    match self.polarization {
                        Polarization3d::Ex => {}
                        Polarization3d::Ey => {
                            if idx < ey.len() {
                                ey[idx] -= coeff_x * e_inc;
                            }
                        }
                        Polarization3d::Ez => {
                            if idx < ez.len() {
                                ez[idx] += coeff_x * h_inc;
                            }
                        }
                    }
                }
            }
        }

        // Face i = i_max+1 (right YZ face, normal = +X)
        let i_right = self.i_max + 1;
        if i_right < nx {
            for j in self.j_min..=self.j_max.min(ny.saturating_sub(1)) {
                for k in self.k_min..=self.k_max.min(nz.saturating_sub(1)) {
                    let idx = Self::idx_kfirst(i_right, j, k, nx, ny);
                    match self.polarization {
                        Polarization3d::Ex => {}
                        Polarization3d::Ey => {
                            if idx < ey.len() {
                                ey[idx] += coeff_x * e_inc;
                            }
                        }
                        Polarization3d::Ez => {
                            if idx < ez.len() {
                                ez[idx] -= coeff_x * h_inc;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Apply H-field corrections using k-major array layout (k*(nx*ny)+j*nx+i).
    ///
    /// Equivalent to [`TfsfSource3d::apply_h_correction`] but uses the k-major indexing
    /// convention of [`crate::fdtd::Fdtd3d`].  Must be called AFTER the regular
    /// 3D H-field update.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_h_correction_kfirst(
        &self,
        hx: &mut [f64],
        hy: &mut [f64],
        hz: &mut [f64],
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
    ) {
        let coeff_x = self.aux_dt / (MU0 * dx);
        let coeff_y = self.aux_dt / (MU0 * dy);
        let coeff_z = self.aux_dt / (MU0 * dz);
        let e_inc = self.e_inc_at_offset(0);
        let h_inc = self.h_inc_at_offset(0);

        // Face k = k_min (bottom XY face)
        if self.k_min > 0 {
            let k = self.k_min - 1;
            for i in self.i_min..=self.i_max.min(nx.saturating_sub(1)) {
                for j in self.j_min..=self.j_max.min(ny.saturating_sub(1)) {
                    let idx = Self::idx_kfirst(i, j, k, nx, ny);
                    match self.polarization {
                        Polarization3d::Ex => {
                            if idx < hx.len() {
                                hx[idx] -= coeff_z * e_inc;
                            }
                        }
                        Polarization3d::Ey => {
                            if idx < hy.len() {
                                hy[idx] += coeff_z * e_inc;
                            }
                        }
                        Polarization3d::Ez => {
                            if idx < hz.len() {
                                hz[idx] -= coeff_x * h_inc;
                            }
                        }
                    }
                }
            }
        }

        // Face k = k_max (top XY face)
        let k = self.k_max.min(nz.saturating_sub(1));
        for i in self.i_min..=self.i_max.min(nx.saturating_sub(1)) {
            for j in self.j_min..=self.j_max.min(ny.saturating_sub(1)) {
                let idx = Self::idx_kfirst(i, j, k, nx, ny);
                match self.polarization {
                    Polarization3d::Ex => {
                        if idx < hx.len() {
                            hx[idx] += coeff_z * e_inc;
                        }
                    }
                    Polarization3d::Ey => {
                        if idx < hy.len() {
                            hy[idx] -= coeff_z * e_inc;
                        }
                    }
                    Polarization3d::Ez => {
                        if idx < hz.len() {
                            hz[idx] += coeff_x * h_inc;
                        }
                    }
                }
            }
        }

        // Face j = j_min (front XZ face)
        if self.j_min > 0 {
            let j = self.j_min - 1;
            for i in self.i_min..=self.i_max.min(nx.saturating_sub(1)) {
                for k in self.k_min..=self.k_max.min(nz.saturating_sub(1)) {
                    let idx = Self::idx_kfirst(i, j, k, nx, ny);
                    match self.polarization {
                        Polarization3d::Ex => {
                            if idx < hx.len() {
                                hx[idx] -= coeff_y * h_inc;
                            }
                        }
                        Polarization3d::Ey => {}
                        Polarization3d::Ez => {
                            if idx < hz.len() {
                                hz[idx] += coeff_y * e_inc;
                            }
                        }
                    }
                }
            }
        }

        // Face j = j_max (back XZ face)
        let j = self.j_max.min(ny.saturating_sub(1));
        for i in self.i_min..=self.i_max.min(nx.saturating_sub(1)) {
            for k in self.k_min..=self.k_max.min(nz.saturating_sub(1)) {
                let idx = Self::idx_kfirst(i, j, k, nx, ny);
                match self.polarization {
                    Polarization3d::Ex => {
                        if idx < hx.len() {
                            hx[idx] += coeff_y * h_inc;
                        }
                    }
                    Polarization3d::Ey => {}
                    Polarization3d::Ez => {
                        if idx < hz.len() {
                            hz[idx] -= coeff_y * e_inc;
                        }
                    }
                }
            }
        }

        // Face i = i_min (left YZ face)
        if self.i_min > 0 {
            let i = self.i_min - 1;
            for j in self.j_min..=self.j_max.min(ny.saturating_sub(1)) {
                for k in self.k_min..=self.k_max.min(nz.saturating_sub(1)) {
                    let idx = Self::idx_kfirst(i, j, k, nx, ny);
                    match self.polarization {
                        Polarization3d::Ex => {}
                        Polarization3d::Ey => {
                            if idx < hy.len() {
                                hy[idx] -= coeff_x * h_inc;
                            }
                        }
                        Polarization3d::Ez => {
                            if idx < hz.len() {
                                hz[idx] += coeff_x * e_inc;
                            }
                        }
                    }
                }
            }
        }

        // Face i = i_max (right YZ face)
        let i = self.i_max.min(nx.saturating_sub(1));
        for j in self.j_min..=self.j_max.min(ny.saturating_sub(1)) {
            for k in self.k_min..=self.k_max.min(nz.saturating_sub(1)) {
                let idx = Self::idx_kfirst(i, j, k, nx, ny);
                match self.polarization {
                    Polarization3d::Ex => {}
                    Polarization3d::Ey => {
                        if idx < hy.len() {
                            hy[idx] += coeff_x * h_inc;
                        }
                    }
                    Polarization3d::Ez => {
                        if idx < hz.len() {
                            hz[idx] -= coeff_x * e_inc;
                        }
                    }
                }
            }
        }
    }

    /// Returns the current incident E-field value from the auxiliary grid.
    pub fn current_e_inc(&self) -> f64 {
        self.e_inc_at_offset(0)
    }

    /// Returns the current incident H-field value from the auxiliary grid.
    pub fn current_h_inc(&self) -> f64 {
        self.h_inc_at_offset(0)
    }

    /// Returns the current simulation time step index.
    pub fn time_step(&self) -> usize {
        self.time_step
    }

    /// Returns the TFSF box dimensions as (nx, ny, nz).
    pub fn box_dimensions(&self) -> (usize, usize, usize) {
        (
            self.i_max.saturating_sub(self.i_min),
            self.j_max.saturating_sub(self.j_min),
            self.k_max.saturating_sub(self.k_min),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tfsf_e_inc_oscillates() {
        let tfsf = TfsfConfig1d::cw(10, 90, 1.94e14, 1.0);
        let e1 = tfsf.e_inc(1.0 / (4.0 * 1.94e14));
        let e2 = tfsf.e_inc(3.0 / (4.0 * 1.94e14));
        assert!(e1 * e2 < 0.0);
    }

    #[test]
    fn tfsf_h_inc_proportional_to_e() {
        let tfsf = TfsfConfig1d::cw(10, 90, 1.94e14, 1.0);
        let t = 1.0 / (4.0 * 1.94e14);
        let ratio = tfsf.e_inc(t) / tfsf.h_inc(t);
        assert!((ratio - 377.0).abs() < 1.0, "ratio={ratio:.1}");
    }

    #[test]
    fn tfsf_e_correction_modifies_field() {
        let tfsf = TfsfConfig1d::cw(10, 90, 1.94e14, 1.0);
        let mut ex = vec![0.0f64; 100];
        tfsf.apply_e_correction(&mut ex, 1.0 / (4.0 * 1.94e14), 1e-18, 10e-9);
        let any_nonzero = ex.iter().any(|&v| v.abs() > 0.0);
        assert!(any_nonzero);
    }

    #[test]
    fn tfsf_h_correction_modifies_field() {
        let tfsf = TfsfConfig1d::cw(10, 90, 1.94e14, 1.0);
        let mut hy = vec![0.0f64; 100];
        tfsf.apply_h_correction(&mut hy, 1.0 / (4.0 * 1.94e14), 1e-18, 10e-9);
        let any_nonzero = hy.iter().any(|&v| v.abs() > 0.0);
        assert!(any_nonzero);
    }

    #[test]
    fn tfsf_box_dimensions() {
        let box2d = TfsfBox2d::normal_incidence(10, 90, 10, 70, 1.94e14, 0.0, 1.0);
        assert_eq!(box2d.width(), 80);
        assert_eq!(box2d.height(), 60);
    }

    #[test]
    fn tfsf_aux_grid_advances() {
        let mut aux = TfsfAux1d::new(200, 5e-9, 1.94e14, 10e-15, 1.0);
        for _ in 0..100 {
            aux.advance();
        }
        assert_eq!(aux.step, 100);
        // After injection, some field should be present
        let max_e = aux
            .e_aux
            .iter()
            .cloned()
            .fold(0.0_f64, |a, b| a.abs().max(b.abs()));
        assert!(max_e > 0.0);
    }

    #[test]
    fn tfsf_e_inc_at_position() {
        let box2d = TfsfBox2d::normal_incidence(10, 90, 10, 70, 1e14, 0.0, 1.0);
        let e = box2d.e_inc_at(0.0, 1.0 / (4.0 * 1e14));
        assert!(e.abs() > 0.0);
    }

    #[test]
    fn tfsf_3d_new_and_step() {
        let omega = 2.0 * PI * 1.94e14;
        let dz = 10e-9;
        let dt = 0.99 * dz / C;
        let mut src = TfsfSource3d::new(
            5,
            5,
            5,
            15,
            15,
            15,
            PropagationAxis::PlusZ,
            Polarization3d::Ex,
            1.0,
            omega,
            dz,
            dt,
        );
        // Step the aux grid several times
        for _ in 0..50 {
            src.step_aux();
        }
        assert_eq!(src.time_step, 50);
        // Aux grid should have nonzero fields after injection
        let max_aux = src.aux_ez.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            max_aux.abs() > 0.0,
            "Aux grid should have nonzero fields after stepping"
        );
    }

    #[test]
    fn tfsf_3d_apply_e_correction_does_not_panic() {
        let omega = 2.0 * PI * 1.94e14;
        let dz = 10e-9;
        let dt = 0.99 * dz / C;
        let mut src = TfsfSource3d::new(
            3,
            3,
            3,
            8,
            8,
            8,
            PropagationAxis::PlusZ,
            Polarization3d::Ex,
            1.0,
            omega,
            dz,
            dt,
        )
        .with_gaussian_pulse(30e-15, 10e-15);
        for _ in 0..100 {
            src.step_aux();
        }
        let nx = 12;
        let ny = 12;
        let nz = 12;
        let n = nx * ny * nz;
        let mut ex = vec![0.0f64; n];
        let mut ey = vec![0.0f64; n];
        let mut ez = vec![0.0f64; n];
        src.apply_e_correction(&mut ex, &mut ey, &mut ez, nx, ny, nz, dt, dz, dz, dz);
        // Should not panic and some correction should have been applied
        let any_nonzero = ex
            .iter()
            .chain(ey.iter())
            .chain(ez.iter())
            .any(|&v| v.abs() > 0.0);
        assert!(
            any_nonzero,
            "E correction should modify some field component"
        );
    }

    #[test]
    fn tfsf_3d_box_dimensions() {
        let omega = 2.0 * PI * 1.94e14;
        let dz = 10e-9;
        let dt = 0.99 * dz / C;
        let src = TfsfSource3d::new(
            5,
            5,
            5,
            20,
            20,
            20,
            PropagationAxis::PlusZ,
            Polarization3d::Ex,
            1.0,
            omega,
            dz,
            dt,
        );
        let (bx, by, bz) = src.box_dimensions();
        assert_eq!(bx, 15);
        assert_eq!(by, 15);
        assert_eq!(bz, 15);
    }

    #[test]
    fn tfsf_3d_apply_h_correction_does_not_panic() {
        let omega = 2.0 * PI * 1.94e14;
        let dz = 10e-9;
        let dt = 0.99 * dz / C;
        let mut src = TfsfSource3d::new(
            3,
            3,
            3,
            8,
            8,
            8,
            PropagationAxis::PlusZ,
            Polarization3d::Ex,
            1.0,
            omega,
            dz,
            dt,
        );
        for _ in 0..100 {
            src.step_aux();
        }
        let nx = 12;
        let ny = 12;
        let nz = 12;
        let n = nx * ny * nz;
        let mut hx = vec![0.0f64; n];
        let mut hy = vec![0.0f64; n];
        let mut hz = vec![0.0f64; n];
        src.apply_h_correction(&mut hx, &mut hy, &mut hz, nx, ny, nz, dt, dz, dz, dz);
        let any_nonzero = hx
            .iter()
            .chain(hy.iter())
            .chain(hz.iter())
            .any(|&v| v.abs() > 0.0);
        assert!(
            any_nonzero,
            "H correction should modify some field component"
        );
    }
}
