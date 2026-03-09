/// Extended types for the 3D FDTD engine: sources, monitors, materials, checkpoints.
///
/// This companion module to `fdtd_3d` keeps the main struct under 2000 lines.
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Field components and axes
// ─────────────────────────────────────────────────────────────────────────────

/// Which field component a probe or source targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldComponent3d {
    Ex,
    Ey,
    Ez,
    Hx,
    Hy,
    Hz,
}

/// Cartesian axis label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis3d {
    X,
    Y,
    Z,
}

// ─────────────────────────────────────────────────────────────────────────────
// Material
// ─────────────────────────────────────────────────────────────────────────────

/// Material parameters for a 3D FDTD cell region.
///
/// Electric conductivity `sigma_e` introduces resistive loss in E-field updates.
/// Magnetic conductivity `sigma_m` introduces loss in H-field updates.
#[derive(Debug, Clone, Copy)]
pub struct Fdtd3dMaterial {
    /// Relative permittivity (dimensionless, ≥ 1.0)
    pub eps_r: f64,
    /// Relative permeability (dimensionless, ≥ 1.0)
    pub mu_r: f64,
    /// Electric conductivity (S/m)
    pub sigma_e: f64,
    /// Magnetic conductivity (Ω/m)
    pub sigma_m: f64,
}

impl Fdtd3dMaterial {
    /// Perfect electric conductor approximation (very high sigma_e)
    pub fn pec() -> Self {
        Self {
            eps_r: 1.0,
            mu_r: 1.0,
            sigma_e: 1e10,
            sigma_m: 0.0,
        }
    }

    /// Lossless dielectric with given relative permittivity
    pub fn dielectric(eps_r: f64) -> Self {
        Self {
            eps_r,
            mu_r: 1.0,
            sigma_e: 0.0,
            sigma_m: 0.0,
        }
    }

    /// Vacuum (free space)
    pub fn vacuum() -> Self {
        Self {
            eps_r: 1.0,
            mu_r: 1.0,
            sigma_e: 0.0,
            sigma_m: 0.0,
        }
    }
}

impl Default for Fdtd3dMaterial {
    fn default() -> Self {
        Self::vacuum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Source waveforms
// ─────────────────────────────────────────────────────────────────────────────

/// Trait for a time-domain source waveform.
/// Must be `Send + Sync` so it can be stored in the solver struct.
pub trait SourceWaveform3d: Send + Sync {
    /// Return the waveform value at time `t` (seconds).
    fn value(&self, t: f64) -> f64;
}

/// Gaussian-modulated sinusoidal waveform:  exp(-(t-t0)²/(2σ²)) · cos(ω₀·t)
#[derive(Debug, Clone)]
pub struct GaussianWaveform3d {
    /// Centre time (s)
    pub t0: f64,
    /// Gaussian width (s)
    pub sigma: f64,
    /// Angular frequency (rad/s)
    pub omega0: f64,
}

impl GaussianWaveform3d {
    /// Convenience constructor
    pub fn new(t0: f64, sigma: f64, omega0: f64) -> Self {
        Self { t0, sigma, omega0 }
    }
}

impl SourceWaveform3d for GaussianWaveform3d {
    fn value(&self, t: f64) -> f64 {
        let env = (-(t - self.t0).powi(2) / (2.0 * self.sigma.powi(2))).exp();
        env * (self.omega0 * t).cos()
    }
}

/// Continuous-wave sinusoidal waveform: cos(ω₀·t + φ)
#[derive(Debug, Clone)]
pub struct CwWaveform3d {
    /// Angular frequency (rad/s)
    pub omega0: f64,
    /// Phase offset (rad)
    pub phase: f64,
}

impl CwWaveform3d {
    pub fn new(omega0: f64, phase: f64) -> Self {
        Self { omega0, phase }
    }
}

impl SourceWaveform3d for CwWaveform3d {
    fn value(&self, t: f64) -> f64 {
        (self.omega0 * t + self.phase).cos()
    }
}

/// Pure Gaussian pulse (no carrier): exp(-(t-t0)²/(2σ²))
#[derive(Debug, Clone)]
pub struct GaussianPulse3d {
    pub t0: f64,
    pub sigma: f64,
}

impl GaussianPulse3d {
    pub fn new(t0: f64, sigma: f64) -> Self {
        Self { t0, sigma }
    }
}

impl SourceWaveform3d for GaussianPulse3d {
    fn value(&self, t: f64) -> f64 {
        (-(t - self.t0).powi(2) / (2.0 * self.sigma.powi(2))).exp()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Source types
// ─────────────────────────────────────────────────────────────────────────────

/// A 3D FDTD source: either a single point dipole or an entire plane.
pub enum SourceType3d {
    /// Hard-source point dipole at cell (i,j,k)
    PointDipole {
        i: usize,
        j: usize,
        k: usize,
        component: FieldComponent3d,
        amplitude: f64,
        waveform: Box<dyn SourceWaveform3d>,
    },
    /// Plane-wave injection over an entire cross-section
    PlaneWave {
        axis: Axis3d,
        position: usize,
        component: FieldComponent3d,
        amplitude: f64,
        waveform: Box<dyn SourceWaveform3d>,
    },
}

impl SourceType3d {
    /// Evaluate the source amplitude at time `t`.
    pub fn amplitude_at(&self, t: f64) -> f64 {
        match self {
            SourceType3d::PointDipole {
                amplitude,
                waveform,
                ..
            } => amplitude * waveform.value(t),
            SourceType3d::PlaneWave {
                amplitude,
                waveform,
                ..
            } => amplitude * waveform.value(t),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Monitors
// ─────────────────────────────────────────────────────────────────────────────

/// Point field probe: records (time, value) pairs at a single cell.
#[derive(Debug, Clone)]
pub struct FieldProbe3d {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub component: FieldComponent3d,
    /// Recorded (time_s, field_value) pairs
    pub time_series: Vec<(f64, f64)>,
}

impl FieldProbe3d {
    pub fn new(i: usize, j: usize, k: usize, component: FieldComponent3d) -> Self {
        Self {
            i,
            j,
            k,
            component,
            time_series: Vec::new(),
        }
    }

    /// Append a single observation.
    pub fn record(&mut self, t: f64, value: f64) {
        self.time_series.push((t, value));
    }
}

/// Plane snapshot monitor: periodically records a 2D slice of one field component.
#[derive(Debug, Clone)]
pub struct PlaneMonitor3d {
    /// Axis normal to the slice
    pub normal: Axis3d,
    /// Index along the normal axis
    pub index: usize,
    /// Which field component to record
    pub component: FieldComponent3d,
    /// Recorded snapshots (each is a flat 2-D array)
    pub snapshots: Vec<Vec<f64>>,
    /// Record one snapshot every N steps
    pub record_every: usize,
}

impl PlaneMonitor3d {
    pub fn new(
        normal: Axis3d,
        index: usize,
        component: FieldComponent3d,
        record_every: usize,
    ) -> Self {
        Self {
            normal,
            index,
            component,
            snapshots: Vec::new(),
            record_every,
        }
    }
}

/// DFT (spectral) probe at a single cell over multiple frequencies.
///
/// Accumulates  Σ f(t)·cos(2π·ν·t)  and  Σ f(t)·sin(2π·ν·t)  running sums
/// so the final spectrum can be obtained without storing the full time series.
#[derive(Debug, Clone)]
pub struct DftProbe3d {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub component: FieldComponent3d,
    /// Frequencies to monitor (Hz)
    pub frequencies: Vec<f64>,
    /// Running cosine accumulator (one per frequency)
    pub accum_re: Vec<f64>,
    /// Running sine accumulator (one per frequency)
    pub accum_im: Vec<f64>,
}

impl DftProbe3d {
    pub fn new(
        i: usize,
        j: usize,
        k: usize,
        component: FieldComponent3d,
        frequencies: Vec<f64>,
    ) -> Self {
        let nf = frequencies.len();
        Self {
            i,
            j,
            k,
            component,
            frequencies,
            accum_re: vec![0.0; nf],
            accum_im: vec![0.0; nf],
        }
    }

    /// Update running DFT sums with field value `field_val` at time `t`.
    pub fn update(&mut self, t: f64, field_val: f64) {
        for (fi, &freq) in self.frequencies.iter().enumerate() {
            let phase = 2.0 * PI * freq * t;
            self.accum_re[fi] += field_val * phase.cos();
            self.accum_im[fi] += field_val * phase.sin();
        }
    }

    /// Return `(freq, re, im)` triples for all monitored frequencies.
    pub fn spectrum(&self) -> Vec<(f64, f64, f64)> {
        self.frequencies
            .iter()
            .enumerate()
            .map(|(fi, &freq)| (freq, self.accum_re[fi], self.accum_im[fi]))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Checkpoint
// ─────────────────────────────────────────────────────────────────────────────

/// Snapshot of all six field arrays, enabling simulation restart.
#[derive(Debug, Clone)]
pub struct Checkpoint3d {
    /// Time step index when the checkpoint was taken
    pub time_step: usize,
    pub ex: Vec<f64>,
    pub ey: Vec<f64>,
    pub ez: Vec<f64>,
    pub hx: Vec<f64>,
    pub hy: Vec<f64>,
    pub hz: Vec<f64>,
}

impl Checkpoint3d {
    /// Number of cells stored (should equal nx*ny*nz)
    pub fn num_cells(&self) -> usize {
        self.ex.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ── Waveform tests ──────────────────────────────────────────────────────

    #[test]
    fn gaussian_waveform_peak_at_t0() {
        let w = GaussianWaveform3d::new(10e-15, 3e-15, 2.0 * PI * 300e12);
        // At t=t0, envelope is 1 and cos(ω*t0) is the carrier
        let peak_env = (-(0.0_f64).powi(2) / (2.0 * (3e-15_f64).powi(2))).exp();
        let expected = peak_env * (w.omega0 * w.t0).cos();
        assert_relative_eq!(w.value(w.t0), expected, epsilon = 1e-15);
    }

    #[test]
    fn gaussian_waveform_decays_far_from_t0() {
        let w = GaussianWaveform3d::new(10e-15, 1e-15, 0.0);
        // far from centre the envelope should be essentially zero
        let v = w.value(100e-15);
        assert!(v.abs() < 1e-100, "Expected near-zero but got {v}");
    }

    #[test]
    fn gaussian_pulse_peaks_at_t0() {
        let w = GaussianPulse3d::new(5e-15, 1e-15);
        assert_relative_eq!(w.value(5e-15), 1.0, epsilon = 1e-15);
    }

    #[test]
    fn cw_waveform_oscillates_correctly() {
        let omega = 2.0 * PI * 200e12;
        let w = CwWaveform3d::new(omega, 0.0);
        let t = 1.0 / 200e12; // one full period
        assert_relative_eq!(w.value(t), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn cw_waveform_phase_offset() {
        let omega = 2.0 * PI * 200e12;
        let w = CwWaveform3d::new(omega, PI / 2.0);
        // cos(ω*0 + π/2) = cos(π/2) ≈ 0
        assert!(w.value(0.0).abs() < 1e-14);
    }

    // ── Material tests ──────────────────────────────────────────────────────

    #[test]
    fn material_vacuum_has_unit_values() {
        let m = Fdtd3dMaterial::vacuum();
        assert_relative_eq!(m.eps_r, 1.0);
        assert_relative_eq!(m.mu_r, 1.0);
        assert_eq!(m.sigma_e, 0.0);
        assert_eq!(m.sigma_m, 0.0);
    }

    #[test]
    fn material_dielectric_sets_eps_r() {
        let m = Fdtd3dMaterial::dielectric(2.25);
        assert_relative_eq!(m.eps_r, 2.25);
        assert_eq!(m.sigma_e, 0.0);
    }

    #[test]
    fn material_pec_has_high_conductivity() {
        let m = Fdtd3dMaterial::pec();
        assert!(m.sigma_e > 1e9, "PEC sigma_e should be very large");
    }

    // ── Probe tests ─────────────────────────────────────────────────────────

    #[test]
    fn field_probe_records_time_series() {
        let mut p = FieldProbe3d::new(5, 5, 5, FieldComponent3d::Ez);
        p.record(0.0, 0.1);
        p.record(1e-15, 0.2);
        assert_eq!(p.time_series.len(), 2);
        assert_relative_eq!(p.time_series[1].1, 0.2);
    }

    #[test]
    fn dft_probe_accumulates_spectrum() {
        let freqs = vec![200e12, 300e12];
        let mut probe = DftProbe3d::new(0, 0, 0, FieldComponent3d::Ex, freqs.clone());
        // Inject a DC field (value = 1) at t = 0
        probe.update(0.0, 1.0);
        // cos(0) = 1 for all frequencies, sin(0) = 0
        assert_relative_eq!(probe.accum_re[0], 1.0, epsilon = 1e-14);
        assert_relative_eq!(probe.accum_im[0], 0.0, epsilon = 1e-14);
    }

    #[test]
    fn dft_probe_spectrum_length_matches_frequencies() {
        let freqs = vec![100e12, 200e12, 300e12];
        let probe = DftProbe3d::new(0, 0, 0, FieldComponent3d::Ez, freqs.clone());
        let spec = probe.spectrum();
        assert_eq!(spec.len(), 3);
    }

    // ── Checkpoint tests ────────────────────────────────────────────────────

    #[test]
    fn checkpoint_num_cells_correct() {
        let n = 27; // 3×3×3
        let cp = Checkpoint3d {
            time_step: 10,
            ex: vec![0.0; n],
            ey: vec![0.0; n],
            ez: vec![0.0; n],
            hx: vec![0.0; n],
            hy: vec![0.0; n],
            hz: vec![0.0; n],
        };
        assert_eq!(cp.num_cells(), n);
    }

    #[test]
    fn checkpoint_clone_is_independent() {
        let mut cp = Checkpoint3d {
            time_step: 5,
            ex: vec![1.0; 8],
            ey: vec![0.0; 8],
            ez: vec![0.0; 8],
            hx: vec![0.0; 8],
            hy: vec![0.0; 8],
            hz: vec![0.0; 8],
        };
        let cp2 = cp.clone();
        cp.ex[0] = 99.0;
        assert_relative_eq!(cp2.ex[0], 1.0); // clone is independent
    }

    // ── Axis / component enum coverage ─────────────────────────────────────

    #[test]
    fn field_component_eq() {
        assert_eq!(FieldComponent3d::Ex, FieldComponent3d::Ex);
        assert_ne!(FieldComponent3d::Ex, FieldComponent3d::Ey);
    }

    #[test]
    fn axis3d_eq() {
        assert_eq!(Axis3d::X, Axis3d::X);
        assert_ne!(Axis3d::X, Axis3d::Z);
    }
}
