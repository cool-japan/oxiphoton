use crate::fdtd::boundary::absorbing::MurAbc1d;
use crate::fdtd::boundary::pml::Cpml;
use crate::fdtd::config::BoundaryConfig;
use crate::fdtd::config::{Dimensions, GridSpacing};
use crate::fdtd::courant::courant_dt;
use crate::fdtd::engine::yee::Yee1d;
use crate::fdtd::monitor::dft::DftMonitor1d;
use crate::fdtd::monitor::field::FieldMonitor1d;
use crate::fdtd::monitor::flux::{FluxMonitor1d, FluxMonitorDft};
use crate::fdtd::source::plane_wave::PlaneWaveSource;
use crate::units::conversion::{EPSILON_0, MU_0};

/// Time-domain probe recording Ex at a fixed grid position.
pub struct TimeProbe1d {
    /// Grid position index
    pub position: usize,
    /// Time values (s)
    pub times: Vec<f64>,
    /// Ex values
    pub ex_values: Vec<f64>,
}

impl TimeProbe1d {
    pub fn new(position: usize) -> Self {
        Self {
            position,
            times: Vec::new(),
            ex_values: Vec::new(),
        }
    }

    fn record(&mut self, time: f64, ex: f64) {
        self.times.push(time);
        self.ex_values.push(ex);
    }

    /// Peak |Ex| recorded over time.
    pub fn peak(&self) -> f64 {
        self.ex_values
            .iter()
            .cloned()
            .fold(0.0_f64, |a, v| a.max(v.abs()))
    }

    /// RMS Ex over time.
    pub fn rms(&self) -> f64 {
        if self.ex_values.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = self.ex_values.iter().map(|e| e * e).sum();
        (sum_sq / self.ex_values.len() as f64).sqrt()
    }
}

/// Subcell material averaging: compute eps_r as harmonic average at an interface.
///
/// - `eps_left`, `eps_right`: permittivities on either side
/// - `frac`: fraction of cell filled with right material (0=all left, 1=all right)
///
/// Uses harmonic average (appropriate for normal E component).
pub fn subcell_eps_harmonic(eps_left: f64, eps_right: f64, frac: f64) -> f64 {
    1.0 / ((1.0 - frac) / eps_left + frac / eps_right)
}

/// Subcell material averaging using arithmetic average (tangential E component).
pub fn subcell_eps_arithmetic(eps_left: f64, eps_right: f64, frac: f64) -> f64 {
    (1.0 - frac) * eps_left + frac * eps_right
}

/// 1D FDTD solver (TEM wave, Ex/Hy)
///
/// Implements the standard Yee algorithm with CPML absorbing boundaries.
///
/// Update equations:
///   Hy\[i\]^{n+1/2} = Hy\[i\]^{n-1/2} - (dt/(mu*dz)) * (Ex\[i+1\]^n - Ex\[i\]^n)
///   Ex\[i\]^{n+1} = Ex\[i\]^n - (dt/(eps*dz)) * (Hy\[i\]^{n+1/2} - Hy\[i-1\]^{n+1/2})
pub struct Fdtd1d {
    pub grid: Yee1d,
    pub dt: f64,
    pub time_step: usize,
    pml: Cpml,
    /// CPML auxiliary fields for H update (at H positions)
    psi_hy: Vec<f64>,
    /// CPML auxiliary fields for E update (at E positions)
    psi_ex: Vec<f64>,
    pub sources: Vec<PlaneWaveSource>,
    pub dft_monitors: Vec<DftMonitor1d>,
    pub flux_monitors: Vec<FluxMonitor1d>,
    pub flux_dft_monitors: Vec<FluxMonitorDft>,
    pub field_monitors: Vec<FieldMonitor1d>,
    /// Time history of Ex at a probe position
    pub time_probes: Vec<TimeProbe1d>,
}

impl Fdtd1d {
    /// Create a new 1D FDTD solver
    pub fn new(nz: usize, dz: f64, boundary: &BoundaryConfig) -> Self {
        let dt = 0.99
            * courant_dt(
                Dimensions::OneD { nz },
                GridSpacing { dx: dz, dy: dz, dz },
                1.0,
            );

        let pml = Cpml::new(
            nz,
            boundary.pml_cells,
            dz,
            dt,
            boundary.pml_m,
            boundary.pml_r0,
        );

        Self {
            grid: Yee1d::new(nz, dz),
            dt,
            time_step: 0,
            pml,
            psi_hy: vec![0.0; nz],
            psi_ex: vec![0.0; nz],
            sources: Vec::new(),
            dft_monitors: Vec::new(),
            flux_monitors: Vec::new(),
            flux_dft_monitors: Vec::new(),
            field_monitors: Vec::new(),
            time_probes: Vec::new(),
        }
    }

    /// Add a field monitor (snapshot at intervals).
    pub fn add_field_monitor(&mut self, monitor: FieldMonitor1d) {
        self.field_monitors.push(monitor);
    }

    /// Add a time-domain probe at grid position `pos`.
    pub fn add_time_probe(&mut self, pos: usize) {
        self.time_probes.push(TimeProbe1d::new(pos));
    }

    /// Fill a material slab [z_start, z_end) with given eps_r.
    pub fn fill_eps(&mut self, z_start: f64, z_end: f64, eps_r: f64) {
        let nz = self.grid.nz;
        let dz = self.grid.dz;
        let i0 = (z_start / dz).floor() as usize;
        let i1 = ((z_end / dz).ceil() as usize).min(nz);
        for i in i0..i1 {
            self.grid.eps_r[i] = eps_r;
        }
    }

    /// Set subcell-averaged eps at cell `i` for interface at fractional position `frac`.
    pub fn set_subcell_eps(&mut self, i: usize, eps_left: f64, eps_right: f64, frac: f64) {
        if i < self.grid.nz {
            self.grid.eps_r[i] = subcell_eps_harmonic(eps_left, eps_right, frac);
        }
    }

    /// Export field to a Vec<(z, ex, hy)> tuple array.
    pub fn export_fields(&self) -> Vec<(f64, f64, f64)> {
        let dz = self.grid.dz;
        (0..self.grid.nz)
            .map(|i| (i as f64 * dz, self.grid.ex[i], self.grid.hy[i]))
            .collect()
    }

    /// Total electromagnetic energy in the domain.
    ///
    ///   U = 0.5 * (eps0 * sum(eps_r * Ex²) + mu0 * sum(mu_r * Hy²)) * dz
    pub fn total_energy(&self) -> f64 {
        let dz = self.grid.dz;
        let e_energy: f64 = self
            .grid
            .ex
            .iter()
            .zip(self.grid.eps_r.iter())
            .map(|(e, &eps)| eps * e * e)
            .sum::<f64>()
            * 0.5
            * EPSILON_0
            * dz;
        let h_energy: f64 = self
            .grid
            .hy
            .iter()
            .zip(self.grid.mu_r.iter())
            .map(|(h, &mu)| mu * h * h)
            .sum::<f64>()
            * 0.5
            * MU_0
            * dz;
        e_energy + h_energy
    }

    /// Create an alternate solver with Mur ABC (first-order) instead of PML.
    pub fn with_mur_abc(nz: usize, dz: f64) -> Fdtd1dMur {
        Fdtd1dMur::new(nz, dz)
    }

    pub fn current_time(&self) -> f64 {
        self.time_step as f64 * self.dt
    }

    /// Advance one time step
    pub fn step(&mut self) {
        let nz = self.grid.nz;
        let dz = self.grid.dz;
        let dt = self.dt;
        let t = self.current_time();

        // --- Update H field (at n+1/2) ---
        for i in 0..nz - 1 {
            let dex = self.grid.ex[i + 1] - self.grid.ex[i];
            // CPML auxiliary field update
            self.psi_hy[i] = self.pml.b_h[i] * self.psi_hy[i] + self.pml.c_h[i] * dex / dz;
            let kappa = self.pml.kappa_h[i];
            self.grid.hy[i] -=
                dt / (MU_0 * self.grid.mu_r[i]) * (dex / (kappa * dz) + self.psi_hy[i]);
        }

        // --- Inject sources (hard source, soft source) ---
        for src in &self.sources {
            let pos = src.position;
            if pos < nz {
                // Soft source: add to existing field
                self.grid.ex[pos] += src.amplitude(t + 0.5 * dt);
            }
        }

        // --- Update E field (at n+1) ---
        for i in 1..nz - 1 {
            let dhy = self.grid.hy[i] - self.grid.hy[i - 1];
            // CPML auxiliary field update
            self.psi_ex[i] = self.pml.b_e[i] * self.psi_ex[i] + self.pml.c_e[i] * dhy / dz;
            let kappa = self.pml.kappa_e[i];
            self.grid.ex[i] -=
                dt / (EPSILON_0 * self.grid.eps_r[i]) * (dhy / (kappa * dz) + self.psi_ex[i]);
        }
        // Boundary: PEC (perfect electric conductor) at ends
        self.grid.ex[0] = 0.0;
        self.grid.ex[nz - 1] = 0.0;

        self.time_step += 1;

        // --- Record monitors ---
        let t_new = self.current_time();
        for mon in &mut self.dft_monitors {
            let pos = mon.position.min(nz - 1);
            let ex = self.grid.ex[pos];
            let hy = self.grid.hy[pos];
            mon.accumulate(ex, hy, t_new, dt);
        }
        for mon in &mut self.flux_monitors {
            let pos = mon.position.min(nz - 1);
            mon.record(self.grid.ex[pos], self.grid.hy[pos]);
        }
        for mon in &mut self.flux_dft_monitors {
            let pos = mon.position.min(nz - 1);
            mon.accumulate(self.grid.ex[pos], self.grid.hy[pos], t_new, dt);
        }
        for mon in &mut self.field_monitors {
            mon.record(self.time_step, t_new, &self.grid.ex, &self.grid.hy);
        }
        for probe in &mut self.time_probes {
            let pos = probe.position.min(nz - 1);
            probe.record(t_new, self.grid.ex[pos]);
        }
    }

    /// Run for the given number of steps
    pub fn run(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// Add a point source
    pub fn add_source(&mut self, source: PlaneWaveSource) {
        self.sources.push(source);
    }

    /// Add a DFT monitor
    pub fn add_dft_monitor(&mut self, monitor: DftMonitor1d) {
        self.dft_monitors.push(monitor);
    }

    /// Add a flux monitor
    pub fn add_flux_monitor(&mut self, monitor: FluxMonitor1d) {
        self.flux_monitors.push(monitor);
    }

    /// Add a DFT flux monitor
    pub fn add_flux_dft_monitor(&mut self, monitor: FluxMonitorDft) {
        self.flux_dft_monitors.push(monitor);
    }
}

/// 1D FDTD with Mur first-order ABC instead of PML.
///
/// Lighter-weight for simple reflectance calculations.
pub struct Fdtd1dMur {
    pub ex: Vec<f64>,
    pub hy: Vec<f64>,
    pub eps_r: Vec<f64>,
    pub mu_r: Vec<f64>,
    pub dz: f64,
    pub dt: f64,
    mur: MurAbc1d,
    pub time_step: usize,
    pub sources: Vec<(usize, f64)>, // (position, amplitude)
    pub time_probes: Vec<TimeProbe1d>,
}

impl Fdtd1dMur {
    const C: f64 = 2.998e8;

    pub fn new(nz: usize, dz: f64) -> Self {
        let dt = 0.99 * dz / Self::C;
        let mur = MurAbc1d::new(dz, dt);
        Self {
            ex: vec![0.0; nz],
            hy: vec![0.0; nz],
            eps_r: vec![1.0; nz],
            mu_r: vec![1.0; nz],
            dz,
            dt,
            mur,
            time_step: 0,
            sources: Vec::new(),
            time_probes: Vec::new(),
        }
    }

    pub fn fill_eps(&mut self, z_start: f64, z_end: f64, eps_r: f64) {
        let n = self.ex.len();
        let i0 = (z_start / self.dz).floor() as usize;
        let i1 = ((z_end / self.dz).ceil() as usize).min(n);
        for i in i0..i1 {
            self.eps_r[i] = eps_r;
        }
    }

    pub fn add_hard_source(&mut self, pos: usize, amp: f64) {
        self.sources.push((pos, amp));
    }

    pub fn add_time_probe(&mut self, pos: usize) {
        self.time_probes.push(TimeProbe1d::new(pos));
    }

    pub fn step(&mut self, src_waveform: f64) {
        let n = self.ex.len();
        let t = self.time_step as f64 * self.dt;
        self.mur.save(&self.ex);

        // H update
        for i in 0..n - 1 {
            let dex = self.ex[i + 1] - self.ex[i];
            self.hy[i] -= self.dt / (MU_0 * self.mu_r[i] * self.dz) * dex;
        }
        // E update
        for i in 1..n - 1 {
            let dhy = self.hy[i] - self.hy[i - 1];
            self.ex[i] -= self.dt / (EPSILON_0 * self.eps_r[i] * self.dz) * dhy;
        }
        // Sources
        for &(pos, amp) in &self.sources {
            if pos < n {
                self.ex[pos] += amp * src_waveform;
            }
        }
        self.mur.apply(&mut self.ex);
        self.time_step += 1;

        let t_new = t + self.dt;
        for probe in &mut self.time_probes {
            let pos = probe.position.min(n - 1);
            probe.record(t_new, self.ex[pos]);
        }
    }

    pub fn run(&mut self, steps: usize, waveform: impl Fn(usize) -> f64) {
        for step in 0..steps {
            let amp = waveform(step);
            self.step(amp);
        }
    }

    pub fn current_time(&self) -> f64 {
        self.time_step as f64 * self.dt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fdtd::source::plane_wave::PlaneWaveSource;
    use crate::fdtd::source::GaussianEnvelope;

    fn basic_solver(nz: usize) -> Fdtd1d {
        let dz = 10e-9;
        Fdtd1d::new(nz, dz, &BoundaryConfig::pml(20))
    }

    #[test]
    fn fdtd1d_runs_without_panic() {
        let mut solver = basic_solver(200);
        let pulse = GaussianEnvelope::new(50.0 * solver.dt, 10.0 * solver.dt);
        solver.add_source(PlaneWaveSource::new(50, Box::new(pulse)));
        solver.run(500);
        // Just check it doesn't panic and fields are finite
        assert!(solver.grid.ex.iter().all(|&v| v.is_finite()));
        assert!(solver.grid.hy.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn fdtd1d_fields_start_zero() {
        let solver = basic_solver(100);
        assert!(solver.grid.ex.iter().all(|&v| v == 0.0));
        assert!(solver.grid.hy.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn fdtd1d_pulse_propagates() {
        let mut solver = basic_solver(500);
        let pulse = GaussianEnvelope::new(30.0 * solver.dt, 8.0 * solver.dt);
        solver.add_source(PlaneWaveSource::new(100, Box::new(pulse)));

        let max_before: f64 = solver
            .grid
            .ex
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        assert_eq!(max_before, 0.0);

        let travel_cells = 200;
        let steps = (travel_cells as f64 * 1.5) as usize;
        solver.run(steps);

        let max_after: f64 = solver.grid.ex.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(max_after > 0.0, "Pulse should have propagated");
    }

    #[test]
    fn fdtd1d_field_monitor_records() {
        let mut solver = basic_solver(200);
        let pulse = GaussianEnvelope::new(50.0 * solver.dt, 10.0 * solver.dt);
        solver.add_source(PlaneWaveSource::new(50, Box::new(pulse)));
        let mon = FieldMonitor1d::new(200, 10);
        solver.add_field_monitor(mon);
        solver.run(100);
        assert!(solver.field_monitors[0].n_snapshots() > 0);
    }

    #[test]
    fn fdtd1d_time_probe_records() {
        let mut solver = basic_solver(200);
        let pulse = GaussianEnvelope::new(50.0 * solver.dt, 10.0 * solver.dt);
        solver.add_source(PlaneWaveSource::new(50, Box::new(pulse)));
        solver.add_time_probe(100);
        solver.run(200);
        let probe = &solver.time_probes[0];
        assert_eq!(probe.times.len(), 200);
        assert!(probe.peak() >= 0.0);
    }

    #[test]
    fn fdtd1d_fill_eps_changes_material() {
        let mut solver = basic_solver(200);
        solver.fill_eps(50e-9, 100e-9, 2.25);
        // 5 cells filled with eps=2.25 (n=1.5)
        let start_cell = (50e-9_f64 / (10e-9_f64)).floor() as usize;
        assert!((solver.grid.eps_r[start_cell] - 2.25).abs() < 1e-10);
    }

    #[test]
    fn fdtd1d_total_energy_zero_initially() {
        let solver = basic_solver(100);
        let e = solver.total_energy();
        assert_eq!(e, 0.0);
    }

    #[test]
    fn fdtd1d_total_energy_positive_after_source() {
        let mut solver = basic_solver(200);
        let pulse = GaussianEnvelope::new(20.0 * solver.dt, 5.0 * solver.dt);
        solver.add_source(PlaneWaveSource::new(50, Box::new(pulse)));
        solver.run(30);
        let e = solver.total_energy();
        assert!(e > 0.0, "Energy should be positive after source injection");
    }

    #[test]
    fn fdtd1d_export_fields_length() {
        let solver = basic_solver(100);
        let fields = solver.export_fields();
        assert_eq!(fields.len(), 100);
        assert_eq!(fields[0].0, 0.0);
        assert!((fields[5].0 - 50e-9).abs() < 1e-20);
    }

    #[test]
    fn subcell_eps_harmonic_midpoint() {
        let eps = subcell_eps_harmonic(1.0, 4.0, 0.5);
        // Harmonic mean of 1 and 4 at frac=0.5: 1/(0.5/1 + 0.5/4) = 1/0.625 = 1.6
        assert!((eps - 1.6).abs() < 1e-10);
    }

    #[test]
    fn subcell_eps_arithmetic_midpoint() {
        let eps = subcell_eps_arithmetic(1.0, 4.0, 0.5);
        // Arithmetic mean: 0.5*1 + 0.5*4 = 2.5
        assert!((eps - 2.5).abs() < 1e-10);
    }

    #[test]
    fn fdtd1d_mur_runs_without_panic() {
        let mut sim = Fdtd1dMur::new(200, 10e-9);
        sim.add_hard_source(50, 1.0);
        let dt = sim.dt;
        let t0 = 30.0 * dt;
        let sigma = 8.0 * dt;
        sim.run(300, |step| {
            let t = step as f64 * dt;
            (-(t - t0).powi(2) / (2.0 * sigma * sigma)).exp()
        });
        let max_e = sim.ex.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
        assert!(max_e.is_finite() && max_e < 1e10);
    }

    #[test]
    fn fdtd1d_mur_probe_records() {
        let mut sim = Fdtd1dMur::new(200, 10e-9);
        sim.add_hard_source(50, 1.0);
        sim.add_time_probe(150);
        sim.run(100, |_| 1.0);
        assert_eq!(sim.time_probes[0].times.len(), 100);
    }
}
