//! Eigenmode Decomposition Monitor for FDTD simulations.
//!
//! Decomposes accumulated DFT fields at a monitor plane into waveguide mode
//! amplitudes via overlap integrals. Provides modal transmission, reflection,
//! S-parameters, and insertion loss for each registered eigenmode.
//!
//! # Theory
//!
//! Given a set of waveguide eigenmodes {φ_m}, the modal amplitude of a field
//! distribution F at frequency ω is:
//!
//!   a_m(ω) = <φ_m | F(ω)> / sqrt(<φ_m | φ_m>)
//!
//! where the overlap integral uses the cross-product form:
//!
//!   <φ_m | F> = ∫∫ (E_φ × H_F* + E_F × H_φ*) · ẑ dA
//!
//! For simplicity this implementation uses a direct projection:
//!
//!   a_m(ω) = ∫∫ (E_x_m* · E_x_F + E_y_m* · E_y_F) dA / P_m
//!
//! where P_m is the modal power normalisation (sqrt of mode power).

use num_complex::Complex64;
use std::f64::consts::PI;

use crate::error::OxiPhotonError;
use crate::fdtd::monitor::flux::FluxNormal;

// ─────────────────────────────────────────────────────────────────────────────
// EigenModeProfile
// ─────────────────────────────────────────────────────────────────────────────

/// Transverse field distribution of a waveguide eigenmode.
///
/// Fields are stored as flat arrays of length `ni × nj`, where `ni` and `nj`
/// are the number of grid cells along the two transverse directions.
#[derive(Debug, Clone)]
pub struct EigenModeProfile {
    /// Effective refractive index (n_eff = β / k_0).
    pub n_eff: f64,
    /// Transverse E-field x-component (length ni*nj).
    pub field_ex: Vec<f64>,
    /// Transverse E-field y-component (length ni*nj).
    pub field_ey: Vec<f64>,
    /// Transverse H-field x-component (length ni*nj).
    pub field_hx: Vec<f64>,
    /// Transverse H-field y-component (length ni*nj).
    pub field_hy: Vec<f64>,
    /// Mode normalised power: ∫ Re(E×H*) dA = 1 after normalisation.
    pub power: f64,
    /// Mode index (0 = fundamental, 1 = first higher-order, …).
    pub mode_number: usize,
    /// Number of cells along first transverse axis.
    pub ni: usize,
    /// Number of cells along second transverse axis.
    pub nj: usize,
}

impl EigenModeProfile {
    /// Create a new zero-initialised mode profile on a grid of `ni × nj` cells.
    pub fn new(n_eff: f64, ni: usize, nj: usize) -> Self {
        let ncells = ni * nj;
        Self {
            n_eff,
            field_ex: vec![0.0; ncells],
            field_ey: vec![0.0; ncells],
            field_hx: vec![0.0; ncells],
            field_hy: vec![0.0; ncells],
            power: 0.0,
            mode_number: 0,
            ni,
            nj,
        }
    }

    /// Overlap integral with another mode: <φ_self | φ_other>.
    ///
    /// Uses the real-valued E-field inner product:
    ///   <a|b> = Σ_ij (ex_a\[ij\] * ex_b\[ij\] + ey_a\[ij\] * ey_b\[ij\])
    ///
    /// Returns a complex number whose real part is the physical overlap.
    pub fn overlap(&self, other: &EigenModeProfile) -> Complex64 {
        let re: f64 = self
            .field_ex
            .iter()
            .zip(other.field_ex.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
            + self
                .field_ey
                .iter()
                .zip(other.field_ey.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
        Complex64::new(re, 0.0)
    }

    /// Self-overlap: <φ|φ> (should equal 1.0 after normalisation).
    pub fn self_overlap(&self) -> f64 {
        self.field_ex.iter().map(|v| v * v).sum::<f64>()
            + self.field_ey.iter().map(|v| v * v).sum::<f64>()
    }

    /// Normalise the mode so that its self-overlap equals 1.0.
    ///
    /// Also updates `self.power` to reflect the normalised mode power.
    ///
    /// Returns an error if the mode has zero energy (cannot normalise).
    pub fn normalize(&mut self) -> Result<(), OxiPhotonError> {
        let norm2 = self.self_overlap();
        if norm2 < 1e-30 {
            return Err(OxiPhotonError::NumericalError(
                "Cannot normalise eigenmode with zero energy".to_string(),
            ));
        }
        let norm = norm2.sqrt();
        for v in self.field_ex.iter_mut() {
            *v /= norm;
        }
        for v in self.field_ey.iter_mut() {
            *v /= norm;
        }
        for v in self.field_hx.iter_mut() {
            *v /= norm;
        }
        for v in self.field_hy.iter_mut() {
            *v /= norm;
        }
        self.power = 1.0;
        Ok(())
    }

    /// Create a Gaussian mode profile for testing and benchmarking.
    ///
    /// The E-field is:
    ///   Ex(i, j) = A · exp(-(i - i0)²/(2σ_i²) - (j - j0)²/(2σ_j²))
    ///
    /// The mode is not normalised here; call `normalize()` afterwards if needed.
    ///
    /// # Arguments
    /// * `n_eff`   — effective refractive index
    /// * `ni`, `nj` — grid dimensions
    /// * `sigma_i`, `sigma_j` — Gaussian widths in grid cells
    pub fn gaussian(n_eff: f64, ni: usize, nj: usize, sigma_i: f64, sigma_j: f64) -> Self {
        let mut profile = Self::new(n_eff, ni, nj);
        let ci = (ni as f64 - 1.0) / 2.0;
        let cj = (nj as f64 - 1.0) / 2.0;
        let sig_i2 = if sigma_i > 0.0 {
            2.0 * sigma_i * sigma_i
        } else {
            1.0
        };
        let sig_j2 = if sigma_j > 0.0 {
            2.0 * sigma_j * sigma_j
        } else {
            1.0
        };

        for i in 0..ni {
            for j in 0..nj {
                let di = (i as f64 - ci) / ni.max(1) as f64 * ni as f64;
                let dj = (j as f64 - cj) / nj.max(1) as f64 * nj as f64;
                let val = (-(di * di) / sig_i2 - (dj * dj) / sig_j2).exp();
                let idx = i * nj + j;
                profile.field_ex[idx] = val;
                // For a TE-like mode the Hy field is proportional to n_eff * Ex / eta0
                profile.field_hy[idx] = n_eff * val / 376.73;
            }
        }

        // Compute unnormalised power
        profile.power = profile.self_overlap();
        profile
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EigenModeMonitor
// ─────────────────────────────────────────────────────────────────────────────

/// Eigenmode decomposition monitor.
///
/// Placed at a cross-section of the FDTD grid, this monitor accumulates a DFT
/// of the transverse field components at every time step and, after all steps
/// have been completed, projects the DFT fields onto registered waveguide
/// eigenmodes to obtain complex modal transmission and reflection amplitudes.
///
/// # Usage
///
/// 1. Create the monitor with [`EigenModeMonitor::new`].
/// 2. Register eigenmodes with [`add_mode`](EigenModeMonitor::add_mode).
/// 3. Call [`accumulate`](EigenModeMonitor::accumulate) at every FDTD time step.
/// 4. After the simulation, call [`compute_coefficients`](EigenModeMonitor::compute_coefficients).
/// 5. Read out [`transmission`](EigenModeMonitor::transmission), [`reflection`](EigenModeMonitor::reflection),
///    [`s_parameter`](EigenModeMonitor::s_parameter), etc.
pub struct EigenModeMonitor {
    /// Plane normal direction.
    pub normal: FluxNormal,
    /// Cell index along the normal axis.
    pub index: usize,
    /// Transverse range — first axis (rows).
    pub i_range: (usize, usize),
    /// Transverse range — second axis (columns).
    pub j_range: (usize, usize),
    /// Registered waveguide eigenmodes.
    pub modes: Vec<EigenModeProfile>,
    /// Monitored frequencies (Hz).
    pub frequencies: Vec<f64>,
    /// Modal transmission amplitudes \[n_freq\]\[n_mode\] (complex).
    pub t_coeffs: Vec<Vec<Complex64>>,
    /// Modal reflection amplitudes \[n_freq\]\[n_mode\] (complex).
    pub r_coeffs: Vec<Vec<Complex64>>,
    /// Simulation time step (s).
    pub dt: f64,
    /// DFT accumulator for Ex at monitor plane: \[n_freq\]\[n_cells\].
    dft_ex: Vec<Vec<Complex64>>,
    /// DFT accumulator for Ey at monitor plane: \[n_freq\]\[n_cells\].
    dft_ey: Vec<Vec<Complex64>>,
    /// DFT accumulator for Hx at monitor plane: \[n_freq\]\[n_cells\].
    dft_hx: Vec<Vec<Complex64>>,
    /// DFT accumulator for Hy at monitor plane: \[n_freq\]\[n_cells\].
    dft_hy: Vec<Vec<Complex64>>,
    /// Number of accumulated time samples.
    n_samples: usize,
}

impl EigenModeMonitor {
    /// Create a new eigenmode decomposition monitor.
    ///
    /// # Arguments
    /// * `normal`      — axis normal to the monitor plane
    /// * `index`       — cell index along the normal axis
    /// * `i_range`     — (start, end) range of first transverse axis cells (exclusive end)
    /// * `j_range`     — (start, end) range of second transverse axis cells (exclusive end)
    /// * `frequencies` — list of frequencies to monitor (Hz)
    /// * `dt`          — FDTD time step (s)
    pub fn new(
        normal: FluxNormal,
        index: usize,
        i_range: (usize, usize),
        j_range: (usize, usize),
        frequencies: Vec<f64>,
        dt: f64,
    ) -> Self {
        let nf = frequencies.len();
        let ni = i_range.1.saturating_sub(i_range.0);
        let nj = j_range.1.saturating_sub(j_range.0);
        let ncells = ni * nj;
        Self {
            normal,
            index,
            i_range,
            j_range,
            modes: Vec::new(),
            frequencies,
            t_coeffs: vec![Vec::<Complex64>::new(); nf],
            r_coeffs: vec![Vec::<Complex64>::new(); nf],
            dt,
            dft_ex: vec![vec![Complex64::new(0.0, 0.0); ncells]; nf],
            dft_ey: vec![vec![Complex64::new(0.0, 0.0); ncells]; nf],
            dft_hx: vec![vec![Complex64::new(0.0, 0.0); ncells]; nf],
            dft_hy: vec![vec![Complex64::new(0.0, 0.0); ncells]; nf],
            n_samples: 0,
        }
    }

    /// Register a waveguide eigenmode for decomposition.
    ///
    /// The mode field arrays must have length `(i_range.1 - i_range.0) * (j_range.1 - j_range.0)`.
    pub fn add_mode(&mut self, mode: EigenModeProfile) {
        let n_modes = self.modes.len() + 1;
        let nf = self.frequencies.len();
        // Grow coefficient arrays to include the new mode
        for fi in 0..nf {
            self.t_coeffs[fi].resize(n_modes, Complex64::new(0.0, 0.0));
            self.r_coeffs[fi].resize(n_modes, Complex64::new(0.0, 0.0));
        }
        self.modes.push(mode);
    }

    /// Accumulate DFT of fields at the monitor plane.
    ///
    /// Must be called at every FDTD time step in sequence.
    ///
    /// Fields are provided as flat 3-D arrays indexed `[i*ny*nz + j*nz + k]`.
    /// Only cells within `i_range` and `j_range` are sampled.
    ///
    /// # Arguments
    /// * `time_step` — current time step index (0-based)
    /// * `ex`, `ey`, `hx`, `hy` — 3-D field component arrays
    /// * `nx`, `ny`, `nz` — 3-D grid dimensions
    #[allow(clippy::too_many_arguments)]
    pub fn accumulate(
        &mut self,
        time_step: usize,
        ex: &[f64],
        ey: &[f64],
        hx: &[f64],
        hy: &[f64],
        nx: usize,
        ny: usize,
        nz: usize,
    ) {
        let t = time_step as f64 * self.dt;

        // Pre-compute DFT phasors for all frequencies
        let phasors: Vec<Complex64> = self
            .frequencies
            .iter()
            .map(|&f| {
                let phase = -2.0 * PI * f * t;
                Complex64::new(phase.cos(), phase.sin()) * self.dt
            })
            .collect();

        let ni_start = self.i_range.0;
        let ni_end = self.i_range.1;
        let nj_start = self.j_range.0;
        let nj_end = self.j_range.1;
        let n_cols = nj_end.saturating_sub(nj_start);

        match self.normal {
            FluxNormal::Z => {
                let k = self.index;
                if k >= nz {
                    return;
                }
                for i in ni_start..ni_end.min(nx) {
                    for j in nj_start..nj_end.min(ny) {
                        let field_idx = i * ny * nz + j * nz + k;
                        let local_idx = (i - ni_start) * n_cols + (j - nj_start);
                        let ex_v = ex.get(field_idx).copied().unwrap_or(0.0);
                        let ey_v = ey.get(field_idx).copied().unwrap_or(0.0);
                        let hx_v = hx.get(field_idx).copied().unwrap_or(0.0);
                        let hy_v = hy.get(field_idx).copied().unwrap_or(0.0);
                        for (fi, &phasor) in phasors.iter().enumerate() {
                            self.dft_ex[fi][local_idx] += ex_v * phasor;
                            self.dft_ey[fi][local_idx] += ey_v * phasor;
                            self.dft_hx[fi][local_idx] += hx_v * phasor;
                            self.dft_hy[fi][local_idx] += hy_v * phasor;
                        }
                    }
                }
            }
            FluxNormal::X => {
                let i = self.index;
                if i >= nx {
                    return;
                }
                for j in ni_start..ni_end.min(ny) {
                    for k in nj_start..nj_end.min(nz) {
                        let field_idx = i * ny * nz + j * nz + k;
                        let local_idx = (j - ni_start) * n_cols + (k - nj_start);
                        let ex_v = ex.get(field_idx).copied().unwrap_or(0.0);
                        let ey_v = ey.get(field_idx).copied().unwrap_or(0.0);
                        let hx_v = hx.get(field_idx).copied().unwrap_or(0.0);
                        let hy_v = hy.get(field_idx).copied().unwrap_or(0.0);
                        for (fi, &phasor) in phasors.iter().enumerate() {
                            self.dft_ex[fi][local_idx] += ex_v * phasor;
                            self.dft_ey[fi][local_idx] += ey_v * phasor;
                            self.dft_hx[fi][local_idx] += hx_v * phasor;
                            self.dft_hy[fi][local_idx] += hy_v * phasor;
                        }
                    }
                }
            }
            FluxNormal::Y => {
                let j = self.index;
                if j >= ny {
                    return;
                }
                for i in ni_start..ni_end.min(nx) {
                    for k in nj_start..nj_end.min(nz) {
                        let field_idx = i * ny * nz + j * nz + k;
                        let local_idx = (i - ni_start) * n_cols + (k - nj_start);
                        let ex_v = ex.get(field_idx).copied().unwrap_or(0.0);
                        let ey_v = ey.get(field_idx).copied().unwrap_or(0.0);
                        let hx_v = hx.get(field_idx).copied().unwrap_or(0.0);
                        let hy_v = hy.get(field_idx).copied().unwrap_or(0.0);
                        for (fi, &phasor) in phasors.iter().enumerate() {
                            self.dft_ex[fi][local_idx] += ex_v * phasor;
                            self.dft_ey[fi][local_idx] += ey_v * phasor;
                            self.dft_hx[fi][local_idx] += hx_v * phasor;
                            self.dft_hy[fi][local_idx] += hy_v * phasor;
                        }
                    }
                }
            }
        }

        self.n_samples += 1;
    }

    /// Compute modal coefficients by projecting DFT fields onto eigenmodes.
    ///
    /// Must be called after all time steps have been accumulated.
    /// Results are stored in `t_coeffs` and `r_coeffs`.
    ///
    /// The transmission amplitude for mode m at frequency fi is:
    ///
    ///   a_m(fi) = Σ_cells \[ Ex_m\[c\\] * DFT_Ex\[fi,c\] + Ey_m\[c\] * DFT_Ey\[fi,c\] ]
    ///             / sqrt(P_m)
    ///
    /// where P_m is the squared norm of the mode (self-overlap).
    pub fn compute_coefficients(&mut self) {
        let nf = self.frequencies.len();
        let n_modes = self.modes.len();

        // Snapshot mode data to avoid borrow conflicts
        let mode_data: Vec<(Vec<f64>, Vec<f64>, f64)> = self
            .modes
            .iter()
            .map(|m| {
                let norm2 = m.self_overlap().max(1e-60);
                (m.field_ex.clone(), m.field_ey.clone(), norm2.sqrt())
            })
            .collect();

        for fi in 0..nf {
            for (mi, (ex_m, ey_m, norm)) in mode_data.iter().enumerate().take(n_modes) {
                let norm = *norm;
                // Project DFT fields onto mode profile
                let overlap: Complex64 = ex_m
                    .iter()
                    .zip(self.dft_ex[fi].iter())
                    .map(|(m_v, d_v)| *m_v * d_v)
                    .sum::<Complex64>()
                    + ey_m
                        .iter()
                        .zip(self.dft_ey[fi].iter())
                        .map(|(m_v, d_v)| *m_v * d_v)
                        .sum::<Complex64>();

                let coeff = overlap / norm;
                // Store as transmission coefficient (forward propagating)
                if mi < self.t_coeffs[fi].len() {
                    self.t_coeffs[fi][mi] = coeff;
                }
                // Reflection computed as complex conjugate of backward mode
                // In a full simulation one would have a second monitor; here we
                // store the negative of the Hx/Hy overlap as a proxy.
                let r_overlap: Complex64 = ex_m
                    .iter()
                    .zip(self.dft_ex[fi].iter())
                    .map(|(m_v, d_v)| *m_v * d_v.conj())
                    .sum::<Complex64>()
                    + ey_m
                        .iter()
                        .zip(self.dft_ey[fi].iter())
                        .map(|(m_v, d_v)| *m_v * d_v.conj())
                        .sum::<Complex64>();
                if mi < self.r_coeffs[fi].len() {
                    self.r_coeffs[fi][mi] = r_overlap / norm;
                }
            }
        }
    }

    /// Power transmission into mode `mode_idx` at each frequency.
    ///
    /// T_m(ω) = |t_m(ω)|²
    pub fn transmission(&self, mode_idx: usize) -> Vec<f64> {
        self.t_coeffs
            .iter()
            .map(|row| row.get(mode_idx).map(|c| c.norm_sqr()).unwrap_or(0.0))
            .collect()
    }

    /// Power reflection into mode `mode_idx` at each frequency.
    ///
    /// R_m(ω) = |r_m(ω)|²
    pub fn reflection(&self, mode_idx: usize) -> Vec<f64> {
        self.r_coeffs
            .iter()
            .map(|row| row.get(mode_idx).map(|c| c.norm_sqr()).unwrap_or(0.0))
            .collect()
    }

    /// S-parameter matrix element S_mn (transmission from mode `in_mode` to mode `out_mode`).
    ///
    /// S_mn(ω) = t_n(ω) / a_m(ω)
    ///
    /// If the input amplitude is zero the result is zero.
    pub fn s_parameter(&self, in_mode: usize, out_mode: usize) -> Vec<Complex64> {
        self.t_coeffs
            .iter()
            .map(|row| {
                let a_in = row
                    .get(in_mode)
                    .copied()
                    .unwrap_or(Complex64::new(0.0, 0.0));
                let a_out = row
                    .get(out_mode)
                    .copied()
                    .unwrap_or(Complex64::new(0.0, 0.0));
                let denom = a_in.norm();
                if denom < 1e-60 {
                    Complex64::new(0.0, 0.0)
                } else {
                    a_out / a_in.norm()
                }
            })
            .collect()
    }

    /// Total transmitted power summed over all registered modes.
    pub fn total_transmission(&self) -> Vec<f64> {
        let n_modes = self.modes.len();
        (0..self.frequencies.len())
            .map(|fi| {
                (0..n_modes)
                    .map(|mi| {
                        self.t_coeffs[fi]
                            .get(mi)
                            .map(|c| c.norm_sqr())
                            .unwrap_or(0.0)
                    })
                    .sum::<f64>()
            })
            .collect()
    }

    /// Insertion loss in dB for mode `mode_idx`: IL = -10·log10(T_m).
    ///
    /// Returns `f64::INFINITY` where transmission is zero.
    pub fn insertion_loss_db(&self, mode_idx: usize) -> Vec<f64> {
        self.transmission(mode_idx)
            .into_iter()
            .map(|t| {
                if t > 0.0 {
                    -10.0 * t.log10()
                } else {
                    f64::INFINITY
                }
            })
            .collect()
    }

    /// Retrieve DFT-accumulated field components at frequency index `fi` for
    /// monitor-local cell `(i, j)`.
    ///
    /// Returns (Ex, Ey, Hx, Hy) as complex values.
    #[allow(dead_code)]
    fn get_dft_at(
        &self,
        fi: usize,
        i: usize,
        j: usize,
    ) -> (Complex64, Complex64, Complex64, Complex64) {
        let n_cols = self.j_range.1.saturating_sub(self.j_range.0);
        let idx = i * n_cols + j;
        let ex = self
            .dft_ex
            .get(fi)
            .and_then(|v| v.get(idx))
            .copied()
            .unwrap_or_default();
        let ey = self
            .dft_ey
            .get(fi)
            .and_then(|v| v.get(idx))
            .copied()
            .unwrap_or_default();
        let hx = self
            .dft_hx
            .get(fi)
            .and_then(|v| v.get(idx))
            .copied()
            .unwrap_or_default();
        let hy = self
            .dft_hy
            .get(fi)
            .and_then(|v| v.get(idx))
            .copied()
            .unwrap_or_default();
        (ex, ey, hx, hy)
    }

    /// Number of accumulated time samples.
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Frequencies monitored (Hz).
    pub fn frequencies(&self) -> &[f64] {
        &self.frequencies
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_monitor(n_freq: usize) -> EigenModeMonitor {
        let freqs: Vec<f64> = (0..n_freq).map(|i| (i + 1) as f64 * 100e12).collect();
        EigenModeMonitor::new(FluxNormal::Z, 4, (0, 4), (0, 4), freqs, 1e-17)
    }

    // ── EigenModeProfile tests ─────────────────────────────────────────────

    #[test]
    fn test_eigenmode_profile_creation() {
        let profile = EigenModeProfile::new(1.5, 8, 6);
        assert_eq!(
            profile.field_ex.len(),
            48,
            "field_ex should have ni*nj elements"
        );
        assert_eq!(profile.field_ey.len(), 48);
        assert_eq!(profile.field_hx.len(), 48);
        assert_eq!(profile.field_hy.len(), 48);
        assert_eq!(profile.ni, 8);
        assert_eq!(profile.nj, 6);
        assert!((profile.n_eff - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_gaussian_profile() {
        let profile = EigenModeProfile::gaussian(1.5, 16, 16, 3.0, 3.0);
        assert_eq!(profile.field_ex.len(), 256);
        // Power (self-overlap) should be strictly positive
        assert!(
            profile.power > 0.0,
            "Gaussian mode power must be positive: {}",
            profile.power
        );
        // Peak should be at centre
        let ci = 7usize;
        let cj = 7usize;
        let centre_val = profile.field_ex[ci * 16 + cj];
        assert!(
            centre_val > 0.5,
            "centre value should be near 1: {centre_val}"
        );
    }

    #[test]
    fn test_mode_normalize() {
        let mut profile = EigenModeProfile::gaussian(1.5, 10, 10, 2.0, 2.0);
        profile
            .normalize()
            .expect("normalize should succeed for non-zero mode");
        let so = profile.self_overlap();
        assert!(
            (so - 1.0).abs() < 1e-10,
            "self_overlap after normalise = {so}"
        );
        assert!((profile.power - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_overlap_orthogonal() {
        // Two modes with disjoint support → overlap ≈ 0
        let mut a = EigenModeProfile::new(1.5, 4, 4);
        let mut b = EigenModeProfile::new(1.5, 4, 4);
        // Mode a: only left half
        for i in 0..4 {
            for j in 0..2 {
                a.field_ex[i * 4 + j] = 1.0;
            }
        }
        // Mode b: only right half
        for i in 0..4 {
            for j in 2..4 {
                b.field_ex[i * 4 + j] = 1.0;
            }
        }
        let ov = a.overlap(&b);
        assert!(
            ov.re.abs() < 1e-12,
            "disjoint modes should have zero overlap: {}",
            ov.re
        );
    }

    // ── EigenModeMonitor tests ────────────────────────────────────────────

    #[test]
    fn test_monitor_creation() {
        let freqs = vec![100e12, 150e12, 200e12];
        let mon = EigenModeMonitor::new(FluxNormal::Z, 4, (0, 8), (0, 8), freqs.clone(), 1e-17);
        assert_eq!(mon.frequencies.len(), 3);
        assert_eq!(mon.dft_ex.len(), 3, "DFT arrays should have n_freq rows");
        assert_eq!(
            mon.dft_ex[0].len(),
            64,
            "DFT arrays should have ni*nj cells"
        );
        assert_eq!(mon.n_samples(), 0);
    }

    #[test]
    fn test_monitor_accumulate() {
        let mut mon = make_monitor(1);
        let nx = 8;
        let ny = 8;
        let nz = 8;
        let n = nx * ny * nz;

        // Constant field Ex = 1.0
        let ex = vec![1.0_f64; n];
        let ey = vec![0.0_f64; n];
        let hx = vec![0.0_f64; n];
        let hy = vec![0.0_f64; n];

        // Accumulate for 100 steps
        for step in 0..100usize {
            mon.accumulate(step, &ex, &ey, &hx, &hy, nx, ny, nz);
        }

        assert_eq!(mon.n_samples(), 100);

        // At non-zero frequency, DFT of constant field is non-zero but bounded
        let (dft_ex, _, _, _) = mon.get_dft_at(0, 0, 0);
        assert!(dft_ex.re.is_finite(), "DFT real part should be finite");
        assert!(dft_ex.im.is_finite(), "DFT imag part should be finite");
    }

    #[test]
    fn test_insertion_loss_calculation() {
        let freqs = vec![100e12];
        let mut mon = EigenModeMonitor::new(FluxNormal::Z, 4, (0, 2), (0, 2), freqs, 1e-17);

        // Create mode with unit norm
        let mut mode = EigenModeProfile::new(1.5, 2, 2);
        mode.field_ex = vec![1.0, 0.0, 0.0, 0.0];
        mode.field_ey = vec![0.0, 0.0, 0.0, 0.0];
        mode.power = 1.0;
        mon.add_mode(mode);

        // Manually set t_coeffs so that |t|² = 0.5 → IL = 3 dB
        let amp = 0.5_f64.sqrt();
        mon.t_coeffs[0][0] = Complex64::new(amp, 0.0);

        let il = mon.insertion_loss_db(0);
        assert_eq!(il.len(), 1);
        let il_val = il[0];
        assert!(
            (il_val - 3.0103).abs() < 0.001,
            "Expected IL ≈ 3.01 dB for T=0.5, got {il_val}"
        );
    }
}
