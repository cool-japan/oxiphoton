//! Mode decomposition monitor for FDTD.
//!
//! Decomposes recorded fields at a cross-section into guided mode amplitudes
//! using overlap integrals with the mode profiles.
//!
//! Forward mode amplitude:
//!   a⁺ = ∫ (E × H_mode* + E_mode* × H) · ẑ dy / (2 · P_mode)
//!
//! where P_mode = (1/2) Re ∫ (E_mode × H_mode*) · ẑ dy is the mode power.
//!
//! Simplification for TE mode in 2D:
//!   a⁺ ∝ ∫ E_z · E_z_mode dy / ∫ |E_z_mode|² dy

/// Mode decomposition monitor for a 1D cross-section.
///
/// Records E and H fields at a fixed x-index over time,
/// then computes modal amplitudes via overlap integrals.
#[derive(Debug, Clone)]
pub struct ModeMonitor {
    /// x-index of monitoring plane
    pub ix: usize,
    /// y-indices included in the monitor
    pub iy_range: std::ops::Range<usize>,
    /// Mode profile E-field (normalized)
    pub mode_e: Vec<f64>,
    /// Mode profile H-field (normalized)
    pub mode_h: Vec<f64>,
    /// Mode power normalization (∫|E_mode|² dy)
    pub mode_power: f64,
    /// Time-domain recorded E-field at monitor plane: `[time_step][iy]`
    pub recorded_e: Vec<Vec<f64>>,
    /// Time-domain recorded H-field at monitor plane
    pub recorded_h: Vec<Vec<f64>>,
    /// Time step for sampling
    pub dt: f64,
}

impl ModeMonitor {
    /// Create a new mode monitor.
    pub fn new(ix: usize, iy_start: usize, mode_e: Vec<f64>, mode_h: Vec<f64>, dt: f64) -> Self {
        let n = mode_e.len();
        let mode_power = mode_e.iter().map(|e| e * e).sum::<f64>();
        Self {
            ix,
            iy_range: iy_start..iy_start + n,
            mode_e,
            mode_h,
            mode_power,
            recorded_e: Vec::new(),
            recorded_h: Vec::new(),
            dt,
        }
    }

    /// Record E and H fields from the FDTD grid at current time step.
    ///
    /// `ez_grid`: flat array ez[ix * ny + iy], ny is grid height
    pub fn record(&mut self, ez_grid: &[f64], hy_grid: &[f64], ny: usize) {
        let e_slice: Vec<f64> = self
            .iy_range
            .clone()
            .map(|iy| {
                let idx = self.ix * ny + iy;
                if idx < ez_grid.len() {
                    ez_grid[idx]
                } else {
                    0.0
                }
            })
            .collect();
        let h_slice: Vec<f64> = self
            .iy_range
            .clone()
            .map(|iy| {
                let idx = self.ix * ny + iy;
                if idx < hy_grid.len() {
                    hy_grid[idx]
                } else {
                    0.0
                }
            })
            .collect();
        self.recorded_e.push(e_slice);
        self.recorded_h.push(h_slice);
    }

    /// Compute mode amplitude overlap at time step `t_idx`.
    ///
    /// Returns the forward-propagating mode amplitude a⁺ (normalized).
    pub fn mode_amplitude(&self, t_idx: usize) -> f64 {
        if t_idx >= self.recorded_e.len() || self.mode_power < 1e-30 {
            return 0.0;
        }
        let e = &self.recorded_e[t_idx];
        let overlap: f64 = e
            .iter()
            .zip(self.mode_e.iter())
            .map(|(ef, em)| ef * em)
            .sum();
        overlap / self.mode_power
    }

    /// Compute DFT of mode amplitude at frequency f (Hz).
    ///
    /// Returns complex amplitude (real, imag).
    pub fn dft_amplitude(&self, f_hz: f64) -> (f64, f64) {
        use std::f64::consts::PI;
        let n_t = self.recorded_e.len();
        if n_t == 0 {
            return (0.0, 0.0);
        }
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for t_idx in 0..n_t {
            let t = t_idx as f64 * self.dt;
            let amp = self.mode_amplitude(t_idx);
            let phase = 2.0 * PI * f_hz * t;
            re += amp * phase.cos();
            im += amp * phase.sin();
        }
        (re / n_t as f64, im / n_t as f64)
    }

    /// Spectral power in mode at frequency f (Hz).
    pub fn spectral_power(&self, f_hz: f64) -> f64 {
        let (re, im) = self.dft_amplitude(f_hz);
        re * re + im * im
    }

    /// Peak mode amplitude across all recorded time steps.
    pub fn peak_amplitude(&self) -> f64 {
        self.recorded_e
            .iter()
            .enumerate()
            .map(|(t, _)| self.mode_amplitude(t).abs())
            .fold(0.0_f64, f64::max)
    }

    /// Number of recorded time steps.
    pub fn n_steps(&self) -> usize {
        self.recorded_e.len()
    }

    /// Time-domain mode amplitude vector.
    pub fn amplitude_series(&self) -> Vec<f64> {
        (0..self.n_steps())
            .map(|t| self.mode_amplitude(t))
            .collect()
    }
}

/// Flux monitor using mode overlap: computes net power in a mode.
///
/// P_mode = (1/T) ∫₀ᵀ Re[a⁺(t) · E_mode(y₀) · H_mode(y₀)] dt
pub struct ModeFluxMonitor {
    pub monitor: ModeMonitor,
}

impl ModeFluxMonitor {
    pub fn new(ix: usize, iy_start: usize, mode_e: Vec<f64>, mode_h: Vec<f64>, dt: f64) -> Self {
        Self {
            monitor: ModeMonitor::new(ix, iy_start, mode_e, mode_h, dt),
        }
    }

    /// Time-averaged power flux in the mode (a.u.).
    pub fn average_power(&self) -> f64 {
        let series = self.monitor.amplitude_series();
        if series.is_empty() {
            return 0.0;
        }
        series.iter().map(|a| a * a).sum::<f64>() / series.len() as f64
    }

    /// Transmission coefficient T = P_out / P_in (pass in reference monitor's power).
    pub fn transmission(&self, p_in: f64) -> f64 {
        if p_in < 1e-30 {
            return 0.0;
        }
        self.average_power() / p_in
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian_mode(n: usize) -> Vec<f64> {
        let w = n as f64 / 4.0;
        (0..n)
            .map(|i| {
                let y = i as f64 - n as f64 / 2.0;
                (-(y / w).powi(2)).exp()
            })
            .collect()
    }

    #[test]
    fn mode_monitor_record_length() {
        let mode_e = gaussian_mode(20);
        let mode_h = mode_e.clone();
        let mut mon = ModeMonitor::new(5, 0, mode_e, mode_h, 1e-18);
        let ez = vec![1.0_f64; 10 * 20];
        let hy = vec![0.001_f64; 10 * 20];
        mon.record(&ez, &hy, 20);
        mon.record(&ez, &hy, 20);
        assert_eq!(mon.n_steps(), 2);
    }

    #[test]
    fn mode_monitor_amplitude_nonzero() {
        let mode_e = gaussian_mode(20);
        let mode_h = mode_e.clone();
        let mut mon = ModeMonitor::new(5, 0, mode_e, mode_h, 1e-18);
        // Inject field matching mode profile
        let ez = vec![1.0_f64; 10 * 20];
        let hy = vec![0.0_f64; 10 * 20];
        mon.record(&ez, &hy, 20);
        let amp = mon.mode_amplitude(0);
        assert!(amp.abs() > 0.0, "amplitude should be nonzero");
    }

    #[test]
    fn mode_monitor_peak_nonnegative() {
        let mode_e = gaussian_mode(20);
        let mode_h = mode_e.clone();
        let mut mon = ModeMonitor::new(5, 0, mode_e, mode_h, 1e-18);
        let ez = vec![0.5_f64; 10 * 20];
        let hy = vec![0.0_f64; 10 * 20];
        for _ in 0..5 {
            mon.record(&ez, &hy, 20);
        }
        assert!(mon.peak_amplitude() >= 0.0);
    }

    #[test]
    fn mode_monitor_dft_amplitude() {
        let mode_e = gaussian_mode(20);
        let mode_h = mode_e.clone();
        let dt = 1e-16;
        let mut mon = ModeMonitor::new(5, 0, mode_e, mode_h, dt);
        let ez = vec![1.0_f64; 10 * 20];
        let hy = vec![0.0_f64; 10 * 20];
        for _ in 0..100 {
            mon.record(&ez, &hy, 20);
        }
        let (re, im) = mon.dft_amplitude(1.94e14);
        assert!((re * re + im * im).sqrt().is_finite());
    }

    #[test]
    fn mode_flux_monitor_average_power() {
        let mode_e = gaussian_mode(20);
        let mode_h = mode_e.clone();
        let mut mfm = ModeFluxMonitor::new(5, 0, mode_e, mode_h, 1e-18);
        let ez = vec![1.0_f64; 10 * 20];
        let hy = vec![0.0_f64; 10 * 20];
        for _ in 0..10 {
            mfm.monitor.record(&ez, &hy, 20);
        }
        let p = mfm.average_power();
        assert!(p >= 0.0, "power={p:.4}");
    }

    #[test]
    fn mode_flux_transmission_ratio() {
        let mode_e = gaussian_mode(20);
        let mode_h = mode_e.clone();
        let mut mfm = ModeFluxMonitor::new(5, 0, mode_e, mode_h, 1e-18);
        let ez = vec![1.0_f64; 10 * 20];
        let hy = vec![0.0_f64; 10 * 20];
        for _ in 0..5 {
            mfm.monitor.record(&ez, &hy, 20);
        }
        let p = mfm.average_power();
        let t = mfm.transmission(p * 2.0);
        assert!((t - 0.5).abs() < 0.01, "T={t:.3}");
    }
}
