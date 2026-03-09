//! DFT monitors for FDTD — frequency-domain field analysis via running DFT.
//!
//! The running DFT accumulates at each time step:
//!   F_re(f) += field(t) * cos(2π·f·t) * dt
//!   F_im(f) -= field(t) * sin(2π·f·t) * dt
//!
//! This gives the complex spectrum F(f) = F_re(f) - i·F_im(f).

use crate::fdtd::monitor::field::{FieldComp3d, MonitorRegion3d};
use num_complex::Complex64;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────
// 1D DFT Monitor (original)
// ─────────────────────────────────────────────────────────────────

/// DFT (Discrete Fourier Transform) monitor for 1D FDTD.
///
/// Accumulates E and H fields at specified frequencies using:
/// F(omega) += F(t) * exp(-j*omega*t) * dt
#[derive(Debug, Clone)]
pub struct DftMonitor1d {
    /// Cell index where the monitor is located
    pub position: usize,
    /// Angular frequencies to monitor (rad/s)
    pub omegas: Vec<f64>,
    /// Accumulated E-field DFT (one complex value per frequency)
    pub e_dft: Vec<Complex64>,
    /// Accumulated H-field DFT
    pub h_dft: Vec<Complex64>,
}

impl DftMonitor1d {
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

    /// Accumulate field values at time t with time step dt
    pub fn accumulate(&mut self, ex: f64, hy: f64, t: f64, dt: f64) {
        for (k, &omega) in self.omegas.iter().enumerate() {
            let phase = Complex64::new(0.0, -omega * t).exp() * dt;
            self.e_dft[k] += ex * phase;
            self.h_dft[k] += hy * phase;
        }
    }

    /// Get complex reflectance relative to incident spectrum
    /// reflectance\[k\] = |E_reflected\[k\] / E_incident\[k\]|^2
    pub fn reflectance(&self, incident: &DftMonitor1d) -> Vec<f64> {
        self.e_dft
            .iter()
            .zip(&incident.e_dft)
            .map(|(er, ei)| (er / ei).norm_sqr())
            .collect()
    }

    /// Get transmittance relative to incident spectrum
    pub fn transmittance(&self, incident: &DftMonitor1d) -> Vec<f64> {
        self.e_dft
            .iter()
            .zip(&incident.e_dft)
            .map(|(et, ei)| (et / ei).norm_sqr())
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────
// 3D DFT Monitor
// ─────────────────────────────────────────────────────────────────

/// 3D DFT monitor — computes frequency-domain field components via running DFT.
///
/// For each monitored frequency f and each cell in the region, accumulates:
///   dft_re\[fi\]\[cell\] += field(t) * cos(2π·f·t) * dt
///   dft_im\[fi\]\[cell\] -= field(t) * sin(2π·f·t) * dt
///
/// After simulation, `get_dft(fi)` returns the complex Fourier amplitude at frequency `fi`.
pub struct DftMonitor3d {
    /// The spatial region to monitor
    pub region: MonitorRegion3d,
    /// Which field component to monitor
    pub component: FieldComp3d,
    /// Frequencies to monitor (Hz)
    pub frequencies: Vec<f64>,
    /// Real parts of DFT: shape \[n_freqs\]\[n_cells\]
    pub dft_re: Vec<Vec<f64>>,
    /// Imaginary parts of DFT: shape \[n_freqs\]\[n_cells\]
    pub dft_im: Vec<Vec<f64>>,
    /// Number of samples accumulated so far
    pub n_samples: usize,
    /// Simulation time step (s) — used for normalization
    pub dt: f64,
    /// Number of cells in the monitored region
    pub n_cells: usize,
    /// Grid dimensions (stored for region extraction)
    nx: usize,
    ny: usize,
    nz: usize,
}

impl DftMonitor3d {
    /// Create a new 3D DFT monitor.
    ///
    /// # Arguments
    /// * `region` — spatial region to monitor
    /// * `component` — field component to accumulate
    /// * `frequencies` — list of frequencies to monitor (Hz)
    /// * `dt` — simulation time step (s)
    /// * `nx`, `ny`, `nz` — grid dimensions
    pub fn new(
        region: MonitorRegion3d,
        component: FieldComp3d,
        frequencies: Vec<f64>,
        dt: f64,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Self {
        let n_cells = region.n_cells(nx, ny, nz);
        let nf = frequencies.len();
        Self {
            region,
            component,
            frequencies,
            dft_re: vec![vec![0.0; n_cells]; nf],
            dft_im: vec![vec![0.0; n_cells]; nf],
            n_samples: 0,
            dt,
            n_cells,
            nx,
            ny,
            nz,
        }
    }

    /// Extract field values for the monitored region and component.
    fn extract_fields(
        &self,
        ex: &[f64],
        ey: &[f64],
        ez: &[f64],
        hx: &[f64],
        hy: &[f64],
        hz: &[f64],
    ) -> Vec<f64> {
        use crate::fdtd::monitor::field::FieldComp3d as Fc;
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let comp = self.component;

        let get = |idx: usize,
                   ex: &[f64],
                   ey: &[f64],
                   ez: &[f64],
                   hx: &[f64],
                   hy: &[f64],
                   hz: &[f64]|
         -> f64 {
            match comp {
                Fc::Ex => ex.get(idx).copied().unwrap_or(0.0),
                Fc::Ey => ey.get(idx).copied().unwrap_or(0.0),
                Fc::Ez => ez.get(idx).copied().unwrap_or(0.0),
                Fc::Hx => hx.get(idx).copied().unwrap_or(0.0),
                Fc::Hy => hy.get(idx).copied().unwrap_or(0.0),
                Fc::Hz => hz.get(idx).copied().unwrap_or(0.0),
                Fc::AbsE => {
                    let a = ex.get(idx).copied().unwrap_or(0.0);
                    let b = ey.get(idx).copied().unwrap_or(0.0);
                    let c = ez.get(idx).copied().unwrap_or(0.0);
                    (a * a + b * b + c * c).sqrt()
                }
                Fc::AbsH => {
                    let a = hx.get(idx).copied().unwrap_or(0.0);
                    let b = hy.get(idx).copied().unwrap_or(0.0);
                    let c = hz.get(idx).copied().unwrap_or(0.0);
                    (a * a + b * b + c * c).sqrt()
                }
            }
        };

        match &self.region {
            MonitorRegion3d::SliceXY { k } => {
                let k = *k;
                if k >= nz {
                    return vec![0.0; self.n_cells];
                }
                let mut out = Vec::with_capacity(nx * ny);
                for i in 0..nx {
                    for j in 0..ny {
                        out.push(get(i * ny * nz + j * nz + k, ex, ey, ez, hx, hy, hz));
                    }
                }
                out
            }
            MonitorRegion3d::SliceXZ { j } => {
                let j = *j;
                if j >= ny {
                    return vec![0.0; self.n_cells];
                }
                let mut out = Vec::with_capacity(nx * nz);
                for i in 0..nx {
                    for k in 0..nz {
                        out.push(get(i * ny * nz + j * nz + k, ex, ey, ez, hx, hy, hz));
                    }
                }
                out
            }
            MonitorRegion3d::SliceYZ { i } => {
                let i = *i;
                if i >= nx {
                    return vec![0.0; self.n_cells];
                }
                let mut out = Vec::with_capacity(ny * nz);
                for j in 0..ny {
                    for k in 0..nz {
                        out.push(get(i * ny * nz + j * nz + k, ex, ey, ez, hx, hy, hz));
                    }
                }
                out
            }
            MonitorRegion3d::FullVolume => (0..nx * ny * nz)
                .map(|idx| get(idx, ex, ey, ez, hx, hy, hz))
                .collect(),
            MonitorRegion3d::SubVolume {
                i0,
                i1,
                j0,
                j1,
                k0,
                k1,
            } => {
                let i0 = *i0;
                let i1 = (*i1).min(nx);
                let j0 = *j0;
                let j1 = (*j1).min(ny);
                let k0 = *k0;
                let k1 = (*k1).min(nz);
                let mut out = Vec::new();
                for i in i0..i1 {
                    for j in j0..j1 {
                        for k in k0..k1 {
                            out.push(get(i * ny * nz + j * nz + k, ex, ey, ez, hx, hy, hz));
                        }
                    }
                }
                out
            }
        }
    }

    /// Update the DFT accumulators with the current field state at the given time step.
    ///
    /// Must be called every time step (or at regular intervals) during the FDTD loop.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        time_step: usize,
        nx: usize,
        ny: usize,
        nz: usize,
        ex: &[f64],
        ey: &[f64],
        ez: &[f64],
        hx: &[f64],
        hy: &[f64],
        hz: &[f64],
    ) {
        let t = time_step as f64 * self.dt;
        let fields = self.extract_fields(ex, ey, ez, hx, hy, hz);
        let _ = nx;
        let _ = ny;
        let _ = nz; // grid dims stored at construction

        for (fi, &freq) in self.frequencies.iter().enumerate() {
            let two_pi_f_t = 2.0 * PI * freq * t;
            let cos_val = two_pi_f_t.cos() * self.dt;
            let sin_val = two_pi_f_t.sin() * self.dt;
            for (ci, &field_val) in fields.iter().enumerate() {
                if ci < self.dft_re[fi].len() {
                    self.dft_re[fi][ci] += field_val * cos_val;
                    self.dft_im[fi][ci] -= field_val * sin_val;
                }
            }
        }
        self.n_samples += 1;
    }

    /// Get the complex DFT result for frequency index `fi`.
    ///
    /// Returns `Vec<(re, im)>` for all cells in the monitored region.
    pub fn get_dft(&self, fi: usize) -> Option<Vec<(f64, f64)>> {
        if fi >= self.frequencies.len() {
            return None;
        }
        let re = &self.dft_re[fi];
        let im = &self.dft_im[fi];
        Some(re.iter().zip(im.iter()).map(|(&r, &i)| (r, i)).collect())
    }

    /// Get the magnitude spectrum at a single cell index within the monitored region.
    ///
    /// Returns a Vec of magnitudes, one per frequency.
    pub fn cell_spectrum(&self, cell_idx: usize) -> Vec<f64> {
        self.frequencies
            .iter()
            .enumerate()
            .map(|(fi, _)| {
                let re = self.dft_re[fi].get(cell_idx).copied().unwrap_or(0.0);
                let im = self.dft_im[fi].get(cell_idx).copied().unwrap_or(0.0);
                (re * re + im * im).sqrt()
            })
            .collect()
    }

    /// Power spectrum (|DFT|²) summed over all cells in the monitored region.
    ///
    /// Returns a Vec of total power, one per frequency.
    pub fn power_spectrum(&self) -> Vec<f64> {
        self.frequencies
            .iter()
            .enumerate()
            .map(|(fi, _)| {
                self.dft_re[fi]
                    .iter()
                    .zip(self.dft_im[fi].iter())
                    .map(|(&r, &im)| r * r + im * im)
                    .sum::<f64>()
            })
            .collect()
    }

    /// Peak magnitude across all frequencies and cells.
    pub fn peak_magnitude(&self) -> f64 {
        let mut peak = 0.0_f64;
        for fi in 0..self.frequencies.len() {
            for (&r, &im) in self.dft_re[fi].iter().zip(self.dft_im[fi].iter()) {
                peak = peak.max((r * r + im * im).sqrt());
            }
        }
        peak
    }

    /// Number of monitored frequencies.
    pub fn n_frequencies(&self) -> usize {
        self.frequencies.len()
    }

    /// Transmittance spectrum relative to a reference DFT monitor.
    ///
    /// Returns T(f) = P_out(f) / P_ref(f) for each frequency.
    pub fn transmittance(&self, reference: &DftMonitor3d) -> Vec<f64> {
        self.power_spectrum()
            .iter()
            .zip(reference.power_spectrum().iter())
            .map(|(&p_out, &p_ref)| if p_ref > 1e-60 { p_out / p_ref } else { 0.0 })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dft_monitor_accumulates() {
        let mut mon = DftMonitor1d::new(50, &[300e12]);
        let omega = 2.0 * PI * 300e12;
        let dt = 1e-17;
        // Inject a CW signal exactly at the monitor frequency
        for n in 0..10000 {
            let t = n as f64 * dt;
            let ex = (omega * t).cos();
            let hy = (omega * t).cos();
            mon.accumulate(ex, hy, t, dt);
        }
        // DFT of cos(omega*t) at omega should be large
        assert!(mon.e_dft[0].norm() > 0.0);
    }

    #[test]
    fn dft_monitor_off_frequency_small() {
        let mut mon = DftMonitor1d::new(50, &[300e12]);
        let dt = 1e-17;
        // Inject at a very different frequency
        let f_inject = 100e12;
        let omega_in = 2.0 * PI * f_inject;
        for n in 0..10000 {
            let t = n as f64 * dt;
            let ex = (omega_in * t).cos();
            mon.accumulate(ex, 0.0, t, dt);
        }
        // Power at 300 THz should be much smaller than at 100 THz
        let mut mon_match = DftMonitor1d::new(50, &[100e12]);
        for n in 0..10000 {
            let t = n as f64 * dt;
            let ex = (omega_in * t).cos();
            mon_match.accumulate(ex, 0.0, t, dt);
        }
        assert!(mon.e_dft[0].norm() < mon_match.e_dft[0].norm());
    }

    // 3D DFT monitor tests
    #[test]
    fn dft_monitor_3d_new() {
        let mon = DftMonitor3d::new(
            MonitorRegion3d::SliceXY { k: 4 },
            FieldComp3d::Ex,
            vec![100e12, 200e12],
            1e-17,
            8,
            8,
            8,
        );
        assert_eq!(mon.n_frequencies(), 2);
        assert_eq!(mon.n_cells, 8 * 8);
        assert_eq!(mon.n_samples, 0);
    }

    #[test]
    fn dft_monitor_3d_update_accumulates() {
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let dt = 1e-17;
        let freq = 100e12_f64;
        let omega = 2.0 * PI * freq;
        let mut mon = DftMonitor3d::new(
            MonitorRegion3d::SliceXY { k: 2 },
            FieldComp3d::Ex,
            vec![freq],
            dt,
            nx,
            ny,
            nz,
        );
        let n = nx * ny * nz;
        let ey = vec![0.0f64; n];
        let ez = vec![0.0f64; n];
        let hx = vec![0.0f64; n];
        let hy = vec![0.0f64; n];
        let hz = vec![0.0f64; n];
        for step in 0..1000usize {
            let t = step as f64 * dt;
            let ex: Vec<f64> = (0..n).map(|_| (omega * t).cos()).collect();
            mon.update(step, nx, ny, nz, &ex, &ey, &ez, &hx, &hy, &hz);
        }
        assert_eq!(mon.n_samples, 1000);
        let ps = mon.power_spectrum();
        assert!(ps[0] > 0.0, "Power spectrum should be nonzero: {}", ps[0]);
    }

    #[test]
    fn dft_monitor_3d_get_dft() {
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let dt = 1e-17;
        let mut mon = DftMonitor3d::new(
            MonitorRegion3d::SliceXY { k: 2 },
            FieldComp3d::Ex,
            vec![100e12],
            dt,
            nx,
            ny,
            nz,
        );
        let n = nx * ny * nz;
        let ex = vec![1.0f64; n];
        let ey = vec![0.0f64; n];
        let ez = vec![0.0f64; n];
        let hx = vec![0.0f64; n];
        let hy = vec![0.0f64; n];
        let hz = vec![0.0f64; n];
        mon.update(0, nx, ny, nz, &ex, &ey, &ez, &hx, &hy, &hz);
        let dft = mon.get_dft(0);
        assert!(dft.is_some(), "get_dft should return Some for valid index");
        let dft_vec = dft.unwrap();
        assert_eq!(dft_vec.len(), nx * ny, "DFT should have n_cells entries");
    }

    #[test]
    fn dft_monitor_3d_cell_spectrum() {
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let dt = 1e-17;
        let mut mon = DftMonitor3d::new(
            MonitorRegion3d::SliceXY { k: 2 },
            FieldComp3d::Ex,
            vec![100e12, 200e12],
            dt,
            nx,
            ny,
            nz,
        );
        let n = nx * ny * nz;
        let ex = vec![1.0f64; n];
        let ey = vec![0.0f64; n];
        let ez = vec![0.0f64; n];
        let hx = vec![0.0f64; n];
        let hy = vec![0.0f64; n];
        let hz = vec![0.0f64; n];
        for step in 0..100 {
            mon.update(step, nx, ny, nz, &ex, &ey, &ez, &hx, &hy, &hz);
        }
        let spec = mon.cell_spectrum(0);
        assert_eq!(
            spec.len(),
            2,
            "Spectrum should have one value per frequency"
        );
        assert!(
            spec.iter().all(|&v| v >= 0.0),
            "All magnitudes should be non-negative"
        );
    }

    #[test]
    fn dft_monitor_3d_get_dft_out_of_range() {
        let mon = DftMonitor3d::new(
            MonitorRegion3d::SliceXY { k: 2 },
            FieldComp3d::Ex,
            vec![100e12],
            1e-17,
            4,
            4,
            4,
        );
        assert!(
            mon.get_dft(99).is_none(),
            "Out-of-range index should return None"
        );
    }
}
