//! Poynting flux monitors for 3D FDTD.
//!
//! Computes time-domain and frequency-domain Poynting vector flux through 2D planes.
//!
//! The Poynting vector is:
//!   S = E × H
//!
//! The flux through a plane with outward normal n̂ is:
//!   Φ = ∫∫ (E × H) · n̂ dA
//!
//! For the Z-normal plane:
//!   Sz = Ex·Hy - Ey·Hx
//!
//! For the X-normal plane:
//!   Sx = Ey·Hz - Ez·Hy
//!
//! For the Y-normal plane:
//!   Sy = Ez·Hx - Ex·Hz

use num_complex::Complex64;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────
// 1D Flux Monitors (original)
// ─────────────────────────────────────────────────────────────────

/// Flux monitor: computes time-averaged Poynting flux at a cell
///
/// For 1D: S_z = (1/2) * Re(Ex * Hy*)
#[derive(Debug, Clone)]
pub struct FluxMonitor1d {
    /// Cell index of the monitor
    pub position: usize,
    /// Accumulated flux (integral of Ex * Hy over time)
    pub accumulated_flux: f64,
    /// Number of accumulation steps
    pub steps: usize,
}

impl FluxMonitor1d {
    pub fn new(position: usize) -> Self {
        Self {
            position,
            accumulated_flux: 0.0,
            steps: 0,
        }
    }

    /// Record fields at this time step
    pub fn record(&mut self, ex: f64, hy: f64) {
        self.accumulated_flux += ex * hy;
        self.steps += 1;
    }

    /// Time-averaged flux (W/m^2) — approximate
    pub fn average_flux(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.accumulated_flux / self.steps as f64
        }
    }
}

/// Flux monitor using DFT fields for frequency-resolved flux
#[derive(Debug, Clone)]
pub struct FluxMonitorDft {
    pub position: usize,
    pub omegas: Vec<f64>,
    pub e_dft: Vec<Complex64>,
    pub h_dft: Vec<Complex64>,
}

impl FluxMonitorDft {
    pub fn new(position: usize, frequencies_hz: &[f64]) -> Self {
        let omegas: Vec<f64> = frequencies_hz.iter().map(|&f| 2.0 * PI * f).collect();
        let nf = omegas.len();
        Self {
            position,
            omegas,
            e_dft: vec![Complex64::new(0.0, 0.0); nf],
            h_dft: vec![Complex64::new(0.0, 0.0); nf],
        }
    }

    pub fn accumulate(&mut self, ex: f64, hy: f64, t: f64, dt: f64) {
        for (k, &omega) in self.omegas.iter().enumerate() {
            let phase = Complex64::new(0.0, -omega * t).exp() * dt;
            self.e_dft[k] += ex * phase;
            self.h_dft[k] += hy * phase;
        }
    }

    /// Frequency-resolved Poynting flux: S(omega) = (1/2) * Re(Ex(omega) * Hy*(omega))
    pub fn flux_spectrum(&self) -> Vec<f64> {
        self.e_dft
            .iter()
            .zip(&self.h_dft)
            .map(|(e, h)| 0.5 * (e * h.conj()).re)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────
// 3D Flux Monitor
// ─────────────────────────────────────────────────────────────────

/// Normal direction of the flux monitor plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluxNormal {
    /// X-normal plane (YZ plane)
    X,
    /// Y-normal plane (XZ plane)
    Y,
    /// Z-normal plane (XY plane)
    Z,
}

/// 3D power flux monitor — computes Poynting vector flux through a 2D plane.
///
/// Records the instantaneous flux Φ(t) = ∫∫ S·n̂ dA at each time step, and
/// simultaneously accumulates a DFT of the flux at specified frequencies for
/// spectral analysis.
///
/// # Index convention
/// For a Z-normal plane at k=`index`:
///   `i_range` = (i_start, i_end) — row indices (X direction)
///   `j_range` = (j_start, j_end) — column indices (Y direction)
pub struct FluxMonitor3d {
    /// Plane normal direction
    pub normal: FluxNormal,
    /// Index along the normal axis (cell index)
    pub index: usize,
    /// Range along first transverse axis
    pub i_range: (usize, usize),
    /// Range along second transverse axis
    pub j_range: (usize, usize),
    /// Accumulated flux time series: (time, flux) pairs
    pub flux_time_series: Vec<(f64, f64)>,
    /// Frequencies for DFT (Hz)
    pub frequencies: Vec<f64>,
    /// Real part of DFT flux at each frequency
    pub flux_dft_re: Vec<f64>,
    /// Imaginary part of DFT flux at each frequency
    pub flux_dft_im: Vec<f64>,
    /// Simulation time step (s)
    pub dt: f64,
    /// Number of DFT accumulations
    n_dft_samples: usize,
}

impl FluxMonitor3d {
    /// Create a new 3D flux monitor.
    ///
    /// # Arguments
    /// * `normal` — plane normal direction (X, Y, or Z)
    /// * `index` — plane index along the normal axis
    /// * `range1` — range of first transverse axis indices (inclusive start, exclusive end)
    /// * `range2` — range of second transverse axis indices (inclusive start, exclusive end)
    /// * `frequencies` — frequencies for DFT accumulation (Hz)
    /// * `dt` — simulation time step (s)
    pub fn new(
        normal: FluxNormal,
        index: usize,
        range1: (usize, usize),
        range2: (usize, usize),
        frequencies: Vec<f64>,
        dt: f64,
    ) -> Self {
        let nf = frequencies.len();
        Self {
            normal,
            index,
            i_range: range1,
            j_range: range2,
            flux_time_series: Vec::new(),
            frequencies,
            flux_dft_re: vec![0.0; nf],
            flux_dft_im: vec![0.0; nf],
            dt,
            n_dft_samples: 0,
        }
    }

    /// Compute the Poynting flux index at cell (a, b, c) with grid (ny, nz).
    #[inline]
    fn field_idx(i: usize, j: usize, k: usize, ny: usize, nz: usize) -> usize {
        i * ny * nz + j * nz + k
    }

    /// Record Poynting flux through the monitoring plane at the current time step.
    ///
    /// Integrates the normal component of S = E × H over the monitoring plane.
    /// Also accumulates the DFT of the flux for frequency-domain analysis.
    ///
    /// # Arguments
    /// * `time` — current simulation time (s)
    /// * `nx`, `ny`, `nz` — 3D grid dimensions
    /// * `ex..hz` — 3D field component arrays (flat, row-major)
    /// * `dx`, `dy`, `dz` — grid spacings (m)
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        time: f64,
        nx: usize,
        ny: usize,
        _nz: usize,
        ex: &[f64],
        ey: &[f64],
        ez: &[f64],
        hx: &[f64],
        hy: &[f64],
        hz: &[f64],
        dx: f64,
        dy: f64,
        dz: f64,
    ) {
        let flux = match self.normal {
            FluxNormal::Z => {
                // Sz = Ex·Hy - Ey·Hx, integrated over XY plane at k=index
                let k = self.index;
                let nz_eff = _nz;
                if k >= nz_eff {
                    return;
                }
                let mut total = 0.0_f64;
                for i in self.i_range.0..self.i_range.1.min(nx) {
                    for j in self.j_range.0..self.j_range.1.min(ny) {
                        let idx = Self::field_idx(i, j, k, ny, nz_eff);
                        let ex_v = ex.get(idx).copied().unwrap_or(0.0);
                        let ey_v = ey.get(idx).copied().unwrap_or(0.0);
                        let hx_v = hx.get(idx).copied().unwrap_or(0.0);
                        let hy_v = hy.get(idx).copied().unwrap_or(0.0);
                        total += (ex_v * hy_v - ey_v * hx_v) * dx * dy;
                    }
                }
                total
            }
            FluxNormal::X => {
                // Sx = Ey·Hz - Ez·Hy, integrated over YZ plane at i=index
                let i = self.index;
                if i >= nx {
                    return;
                }
                let nz_eff = _nz;
                let mut total = 0.0_f64;
                for j in self.i_range.0..self.i_range.1.min(ny) {
                    for k in self.j_range.0..self.j_range.1.min(nz_eff) {
                        let idx = Self::field_idx(i, j, k, ny, nz_eff);
                        let ey_v = ey.get(idx).copied().unwrap_or(0.0);
                        let ez_v = ez.get(idx).copied().unwrap_or(0.0);
                        let hy_v = hy.get(idx).copied().unwrap_or(0.0);
                        let hz_v = hz.get(idx).copied().unwrap_or(0.0);
                        total += (ey_v * hz_v - ez_v * hy_v) * dy * dz;
                    }
                }
                total
            }
            FluxNormal::Y => {
                // Sy = Ez·Hx - Ex·Hz, integrated over XZ plane at j=index
                let j = self.index;
                if j >= ny {
                    return;
                }
                let nz_eff = _nz;
                let mut total = 0.0_f64;
                for i in self.i_range.0..self.i_range.1.min(nx) {
                    for k in self.j_range.0..self.j_range.1.min(nz_eff) {
                        let idx = Self::field_idx(i, j, k, ny, nz_eff);
                        let ex_v = ex.get(idx).copied().unwrap_or(0.0);
                        let ez_v = ez.get(idx).copied().unwrap_or(0.0);
                        let hx_v = hx.get(idx).copied().unwrap_or(0.0);
                        let hz_v = hz.get(idx).copied().unwrap_or(0.0);
                        total += (ez_v * hx_v - ex_v * hz_v) * dx * dz;
                    }
                }
                total
            }
        };

        // Store time series
        self.flux_time_series.push((time, flux));

        // Accumulate DFT
        for (fi, &freq) in self.frequencies.iter().enumerate() {
            let phase = 2.0 * PI * freq * time;
            self.flux_dft_re[fi] += flux * phase.cos() * self.dt;
            self.flux_dft_im[fi] -= flux * phase.sin() * self.dt;
        }
        self.n_dft_samples += 1;
    }

    /// Total time-averaged flux (W), computed as the mean of the time series.
    pub fn time_averaged_flux(&self) -> f64 {
        if self.flux_time_series.is_empty() {
            return 0.0;
        }
        let total: f64 = self.flux_time_series.iter().map(|(_, f)| f).sum();
        total / self.flux_time_series.len() as f64
    }

    /// Power spectral density at each monitored frequency.
    ///
    /// Returns |DFT(flux)|² for each frequency.
    pub fn power_spectrum(&self) -> Vec<f64> {
        self.flux_dft_re
            .iter()
            .zip(self.flux_dft_im.iter())
            .map(|(&r, &im)| r * r + im * im)
            .collect()
    }

    /// Complex DFT flux at each monitored frequency.
    ///
    /// Returns (re, im) pairs.
    pub fn complex_spectrum(&self) -> Vec<(f64, f64)> {
        self.flux_dft_re
            .iter()
            .zip(self.flux_dft_im.iter())
            .map(|(&r, &im)| (r, im))
            .collect()
    }

    /// Frequency-resolved flux: Re[DFT(flux)] / dt at each frequency.
    ///
    /// This gives the time-averaged Poynting flux at each frequency (approximately).
    pub fn frequency_resolved_flux(&self) -> Vec<f64> {
        if self.dt < 1e-50 {
            return vec![0.0; self.frequencies.len()];
        }
        self.flux_dft_re
            .iter()
            .map(|&r| r / self.dt.max(1e-50))
            .collect()
    }

    /// Peak instantaneous flux over the time series.
    pub fn peak_flux(&self) -> f64 {
        self.flux_time_series
            .iter()
            .map(|(_, f)| f.abs())
            .fold(0.0_f64, f64::max)
    }

    /// Number of recorded time steps.
    pub fn n_steps(&self) -> usize {
        self.flux_time_series.len()
    }

    /// Transmittance relative to a reference flux monitor.
    ///
    /// T(f) = |S_out(f)| / |S_ref(f)|
    pub fn transmittance(&self, reference: &FluxMonitor3d) -> Vec<f64> {
        let ps_self = self.power_spectrum();
        let ps_ref = reference.power_spectrum();
        ps_self
            .iter()
            .zip(ps_ref.iter())
            .map(|(&p_out, &p_ref)| {
                if p_ref > 1e-60 {
                    (p_out / p_ref).sqrt()
                } else {
                    0.0
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_3d_fields(nx: usize, ny: usize, nz: usize) -> [Vec<f64>; 6] {
        let n = nx * ny * nz;
        let ex: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let ey = vec![0.0f64; n];
        let ez = vec![0.0f64; n];
        let hx = vec![0.0f64; n];
        let hy: Vec<f64> = (0..n).map(|i| 0.5 * (i as f64 / n as f64)).collect();
        let hz = vec![0.0f64; n];
        [ex, ey, ez, hx, hy, hz]
    }

    #[test]
    fn flux_monitor_1d_average() {
        let mut mon = FluxMonitor1d::new(10);
        // record(ex, hy): accumulated_flux += ex*hy
        // 1.0*0.5 + 2.0*1.0 = 0.5 + 2.0 = 2.5, /2 steps = 1.25
        mon.record(1.0, 0.5);
        mon.record(2.0, 1.0);
        let avg = mon.average_flux();
        assert!((avg - 1.25).abs() < 1e-10, "avg={avg}");
    }

    #[test]
    fn flux_monitor_dft_spectrum() {
        let mut mon = FluxMonitorDft::new(10, &[100e12]);
        let omega = 2.0 * PI * 100e12;
        let dt = 1e-17;
        for n in 0..1000 {
            let t = n as f64 * dt;
            mon.accumulate((omega * t).cos(), (omega * t).cos(), t, dt);
        }
        let spec = mon.flux_spectrum();
        assert_eq!(spec.len(), 1);
        // Should be positive for CW signal
        assert!(spec[0].is_finite());
    }

    #[test]
    fn flux_monitor_3d_z_normal_records() {
        let nx = 8;
        let ny = 8;
        let nz = 8;
        let dx = 10e-9;
        let dy = 10e-9;
        let dz = 10e-9;
        let mut mon = FluxMonitor3d::new(FluxNormal::Z, 4, (0, nx), (0, ny), vec![100e12], 1e-17);
        let [ex, ey, ez, hx, hy, hz] = make_3d_fields(nx, ny, nz);
        mon.record(1e-17, nx, ny, nz, &ex, &ey, &ez, &hx, &hy, &hz, dx, dy, dz);
        assert_eq!(mon.n_steps(), 1, "Should record one time step");
    }

    #[test]
    fn flux_monitor_3d_x_normal_records() {
        let nx = 8;
        let ny = 8;
        let nz = 8;
        let dx = 10e-9;
        let dy = 10e-9;
        let dz = 10e-9;
        let mut mon = FluxMonitor3d::new(FluxNormal::X, 4, (0, ny), (0, nz), vec![100e12], 1e-17);
        let [ex, ey, ez, hx, hy, hz] = make_3d_fields(nx, ny, nz);
        mon.record(1e-17, nx, ny, nz, &ex, &ey, &ez, &hx, &hy, &hz, dx, dy, dz);
        assert_eq!(mon.n_steps(), 1);
    }

    #[test]
    fn flux_monitor_3d_y_normal_records() {
        let nx = 8;
        let ny = 8;
        let nz = 8;
        let dx = 10e-9;
        let dy = 10e-9;
        let dz = 10e-9;
        let mut mon = FluxMonitor3d::new(FluxNormal::Y, 4, (0, nx), (0, nz), vec![100e12], 1e-17);
        let [ex, ey, ez, hx, hy, hz] = make_3d_fields(nx, ny, nz);
        mon.record(1e-17, nx, ny, nz, &ex, &ey, &ez, &hx, &hy, &hz, dx, dy, dz);
        assert_eq!(mon.n_steps(), 1);
    }

    #[test]
    fn flux_monitor_3d_time_averaged_flux() {
        let nx = 6;
        let ny = 6;
        let nz = 6;
        let dx = 10e-9;
        let dy = 10e-9;
        let dz = 10e-9;
        let dt = 1e-17;
        let mut mon = FluxMonitor3d::new(FluxNormal::Z, 3, (1, 5), (1, 5), vec![], dt);
        let [ex, ey, ez, hx, hy, hz] = make_3d_fields(nx, ny, nz);
        for step in 0..10 {
            let t = step as f64 * dt;
            mon.record(t, nx, ny, nz, &ex, &ey, &ez, &hx, &hy, &hz, dx, dy, dz);
        }
        let avg = mon.time_averaged_flux();
        assert!(
            avg.is_finite(),
            "Time-averaged flux should be finite: {avg}"
        );
    }

    #[test]
    fn flux_monitor_3d_power_spectrum_length() {
        let mon = FluxMonitor3d::new(
            FluxNormal::Z,
            4,
            (0, 4),
            (0, 4),
            vec![100e12, 200e12, 300e12],
            1e-17,
        );
        assert_eq!(mon.power_spectrum().len(), 3);
    }

    #[test]
    fn flux_monitor_3d_peak_flux() {
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let dx = 10e-9;
        let dy = 10e-9;
        let dz = 10e-9;
        let dt = 1e-17;
        let mut mon = FluxMonitor3d::new(FluxNormal::Z, 2, (0, nx), (0, ny), vec![], dt);
        // Create fields with known Sz
        let n = nx * ny * nz;
        let ex: Vec<f64> = vec![1.0; n];
        let ey = vec![0.0f64; n];
        let ez = vec![0.0f64; n];
        let hx = vec![0.0f64; n];
        let hy: Vec<f64> = vec![1.0; n];
        let hz = vec![0.0f64; n];
        mon.record(dt, nx, ny, nz, &ex, &ey, &ez, &hx, &hy, &hz, dx, dy, dz);
        let peak = mon.peak_flux();
        assert!(peak > 0.0, "Peak flux should be positive: {peak}");
    }

    #[test]
    fn flux_monitor_3d_out_of_bounds_index() {
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let dx = 10e-9;
        let dy = 10e-9;
        let dz = 10e-9;
        let dt = 1e-17;
        // index > nz should not panic or record
        let mut mon = FluxMonitor3d::new(FluxNormal::Z, 100, (0, nx), (0, ny), vec![], dt);
        let [ex, ey, ez, hx, hy, hz] = make_3d_fields(nx, ny, nz);
        mon.record(dt, nx, ny, nz, &ex, &ey, &ez, &hx, &hy, &hz, dx, dy, dz);
        assert_eq!(mon.n_steps(), 0, "Out-of-bounds index should not record");
    }
}
