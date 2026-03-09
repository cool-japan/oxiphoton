//! Field snapshot and time-averaged intensity monitors for FDTD.
//!
//! Provides 1D, 2D, and 3D field monitors for recording field snapshots
//! at user-specified time intervals and computing time-averaged intensity maps.

// ─────────────────────────────────────────────────────────────────
// 1D Monitors
// ─────────────────────────────────────────────────────────────────

/// A single 1D field snapshot (Ex and Hy at all grid points).
#[derive(Debug, Clone)]
pub struct FieldSnapshot1d {
    /// Time step index at which snapshot was taken.
    pub step: usize,
    /// Simulation time (s).
    pub time: f64,
    /// Ex field values.
    pub ex: Vec<f64>,
    /// Hy field values.
    pub hy: Vec<f64>,
}

impl FieldSnapshot1d {
    pub fn new(step: usize, time: f64, ex: Vec<f64>, hy: Vec<f64>) -> Self {
        Self { step, time, ex, hy }
    }

    /// Instantaneous Poynting vector Sz = Ex × Hy at each cell.
    pub fn poynting_z(&self) -> Vec<f64> {
        self.ex
            .iter()
            .zip(self.hy.iter())
            .map(|(e, h)| e * h)
            .collect()
    }

    /// Peak |Ex| over all grid points.
    pub fn peak_ex(&self) -> f64 {
        self.ex.iter().cloned().fold(0.0_f64, |a, v| a.max(v.abs()))
    }
}

/// 1D field monitor: records snapshots every `interval` steps.
pub struct FieldMonitor1d {
    /// Record every this many steps (0 = record every step).
    pub interval: usize,
    /// Stored snapshots.
    pub snapshots: Vec<FieldSnapshot1d>,
    /// Running sum of Ex² for time-averaged intensity.
    intensity_sum: Vec<f64>,
    /// Count of accumulations.
    n_accum: usize,
    /// Grid size.
    n: usize,
}

impl FieldMonitor1d {
    /// Create a field monitor for a grid of size `n`.
    ///
    /// Set `interval = 1` to record every step, `interval = 10` for every 10th step.
    pub fn new(n: usize, interval: usize) -> Self {
        let interval = interval.max(1);
        Self {
            interval,
            snapshots: Vec::new(),
            intensity_sum: vec![0.0; n],
            n_accum: 0,
            n,
        }
    }

    /// Offer the current fields to the monitor.
    ///
    /// The monitor decides internally whether to record based on `step`.
    pub fn record(&mut self, step: usize, time: f64, ex: &[f64], hy: &[f64]) {
        // Always accumulate for time-average
        for (i, (&e, &h)) in ex.iter().zip(hy.iter()).enumerate().take(self.n) {
            self.intensity_sum[i] += e * e + h * h;
        }
        self.n_accum += 1;

        // Snapshot only at intervals
        if step % self.interval == 0 {
            self.snapshots.push(FieldSnapshot1d::new(
                step,
                time,
                ex[..self.n].to_vec(),
                hy[..self.n].to_vec(),
            ));
        }
    }

    /// Time-averaged intensity (ex² + hy² averaged over all accumulated steps).
    pub fn time_averaged_intensity(&self) -> Vec<f64> {
        if self.n_accum == 0 {
            return vec![0.0; self.n];
        }
        self.intensity_sum
            .iter()
            .map(|s| s / self.n_accum as f64)
            .collect()
    }

    /// Peak time-averaged intensity.
    pub fn peak_averaged_intensity(&self) -> f64 {
        self.time_averaged_intensity()
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max)
    }

    /// Number of recorded snapshots.
    pub fn n_snapshots(&self) -> usize {
        self.snapshots.len()
    }

    /// Get the last snapshot, if any.
    pub fn last_snapshot(&self) -> Option<&FieldSnapshot1d> {
        self.snapshots.last()
    }
}

// ─────────────────────────────────────────────────────────────────
// 2D Monitors
// ─────────────────────────────────────────────────────────────────

/// A single 2D field snapshot (Hz, Ex, Ey for TE mode).
#[derive(Debug, Clone)]
pub struct FieldSnapshot2d {
    /// Time step index.
    pub step: usize,
    /// Simulation time (s).
    pub time: f64,
    /// Grid dimensions.
    pub nx: usize,
    pub ny: usize,
    /// Hz field (row-major: hz[i*ny + j]).
    pub hz: Vec<f64>,
    /// Ex field.
    pub ex: Vec<f64>,
    /// Ey field.
    pub ey: Vec<f64>,
}

impl FieldSnapshot2d {
    pub fn new(
        step: usize,
        time: f64,
        nx: usize,
        ny: usize,
        hz: Vec<f64>,
        ex: Vec<f64>,
        ey: Vec<f64>,
    ) -> Self {
        Self {
            step,
            time,
            nx,
            ny,
            hz,
            ex,
            ey,
        }
    }

    /// Index helper: row-major (i*ny + j).
    pub fn idx(&self, i: usize, j: usize) -> usize {
        i * self.ny + j
    }

    /// Instantaneous intensity |E|² = Ex² + Ey² at each cell.
    pub fn e_intensity(&self) -> Vec<f64> {
        self.ex
            .iter()
            .zip(self.ey.iter())
            .map(|(&ex, &ey)| ex * ex + ey * ey)
            .collect()
    }

    /// Peak |Hz|.
    pub fn peak_hz(&self) -> f64 {
        self.hz.iter().cloned().fold(0.0_f64, |a, v| a.max(v.abs()))
    }

    /// Extract a horizontal slice (fixed row `i`) of Hz.
    pub fn hz_row(&self, i: usize) -> Vec<f64> {
        (0..self.ny).map(|j| self.hz[self.idx(i, j)]).collect()
    }

    /// Extract a vertical slice (fixed column `j`) of Hz.
    pub fn hz_col(&self, j: usize) -> Vec<f64> {
        (0..self.nx).map(|i| self.hz[self.idx(i, j)]).collect()
    }
}

/// 2D field monitor: records snapshots every `interval` steps.
pub struct FieldMonitor2d {
    pub interval: usize,
    pub snapshots: Vec<FieldSnapshot2d>,
    intensity_sum: Vec<f64>,
    n_accum: usize,
    nx: usize,
    ny: usize,
}

impl FieldMonitor2d {
    pub fn new(nx: usize, ny: usize, interval: usize) -> Self {
        let interval = interval.max(1);
        Self {
            interval,
            snapshots: Vec::new(),
            intensity_sum: vec![0.0; nx * ny],
            n_accum: 0,
            nx,
            ny,
        }
    }

    pub fn record(&mut self, step: usize, time: f64, hz: &[f64], ex: &[f64], ey: &[f64]) {
        let n = self.nx * self.ny;
        for i in 0..n {
            self.intensity_sum[i] += ex[i] * ex[i] + ey[i] * ey[i];
        }
        self.n_accum += 1;

        if step % self.interval == 0 {
            self.snapshots.push(FieldSnapshot2d::new(
                step,
                time,
                self.nx,
                self.ny,
                hz[..n].to_vec(),
                ex[..n].to_vec(),
                ey[..n].to_vec(),
            ));
        }
    }

    /// Time-averaged |E|² intensity map.
    pub fn time_averaged_intensity(&self) -> Vec<f64> {
        if self.n_accum == 0 {
            return vec![0.0; self.nx * self.ny];
        }
        self.intensity_sum
            .iter()
            .map(|s| s / self.n_accum as f64)
            .collect()
    }

    pub fn n_snapshots(&self) -> usize {
        self.snapshots.len()
    }
}

// ─────────────────────────────────────────────────────────────────
// 3D Monitors
// ─────────────────────────────────────────────────────────────────

/// Specification of which region the 3D field monitor covers.
#[derive(Debug, Clone, Copy)]
pub enum MonitorRegion3d {
    /// Record the entire 3D simulation volume
    FullVolume,
    /// Record a 2D XY cross-section at fixed k
    SliceXY { k: usize },
    /// Record a 2D XZ cross-section at fixed j
    SliceXZ { j: usize },
    /// Record a 2D YZ cross-section at fixed i
    SliceYZ { i: usize },
    /// Record a sub-volume [i0..i1] × [j0..j1] × [k0..k1]
    SubVolume {
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        k0: usize,
        k1: usize,
    },
}

impl MonitorRegion3d {
    /// Return the shape [ni, nj, nk] of the recorded data for a given grid.
    pub fn data_shape(&self, nx: usize, ny: usize, nz: usize) -> [usize; 3] {
        match self {
            MonitorRegion3d::FullVolume => [nx, ny, nz],
            MonitorRegion3d::SliceXY { .. } => [nx, ny, 1],
            MonitorRegion3d::SliceXZ { .. } => [nx, 1, nz],
            MonitorRegion3d::SliceYZ { .. } => [1, ny, nz],
            MonitorRegion3d::SubVolume {
                i0,
                i1,
                j0,
                j1,
                k0,
                k1,
            } => [
                i1.saturating_sub(*i0),
                j1.saturating_sub(*j0),
                k1.saturating_sub(*k0),
            ],
        }
    }

    /// Return the number of cells in this region.
    pub fn n_cells(&self, nx: usize, ny: usize, nz: usize) -> usize {
        let [a, b, c] = self.data_shape(nx, ny, nz);
        a * b * c
    }
}

/// Which field component the monitor records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldComp3d {
    /// Electric field X component
    Ex,
    /// Electric field Y component
    Ey,
    /// Electric field Z component
    Ez,
    /// Magnetic field X component
    Hx,
    /// Magnetic field Y component
    Hy,
    /// Magnetic field Z component
    Hz,
    /// |E| = sqrt(Ex² + Ey² + Ez²)
    AbsE,
    /// |H| = sqrt(Hx² + Hy² + Hz²)
    AbsH,
}

/// A single 3D field snapshot.
#[derive(Debug, Clone)]
pub struct FieldSnapshot3d {
    /// Time step index at which snapshot was taken
    pub time_step: usize,
    /// Flattened field data
    pub data: Vec<f64>,
    /// Shape [ni, nj, nk] of the recorded data region
    pub shape: [usize; 3],
}

impl FieldSnapshot3d {
    pub fn new(time_step: usize, data: Vec<f64>, shape: [usize; 3]) -> Self {
        Self {
            time_step,
            data,
            shape,
        }
    }

    /// Element at (i, j, k) within the recorded region.
    pub fn at(&self, i: usize, j: usize, k: usize) -> Option<f64> {
        let [ni, nj, nk] = self.shape;
        if i < ni && j < nj && k < nk {
            Some(self.data[i * nj * nk + j * nk + k])
        } else {
            None
        }
    }

    /// Maximum absolute value in the snapshot.
    pub fn peak(&self) -> f64 {
        self.data
            .iter()
            .cloned()
            .fold(0.0_f64, |a, v| a.max(v.abs()))
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if data is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Extract a field component from the 6 field arrays at a single flat index.
#[inline]
#[allow(clippy::too_many_arguments)]
fn extract_component(
    comp: FieldComp3d,
    idx: usize,
    ex: &[f64],
    ey: &[f64],
    ez: &[f64],
    hx: &[f64],
    hy: &[f64],
    hz: &[f64],
) -> f64 {
    match comp {
        FieldComp3d::Ex => ex.get(idx).copied().unwrap_or(0.0),
        FieldComp3d::Ey => ey.get(idx).copied().unwrap_or(0.0),
        FieldComp3d::Ez => ez.get(idx).copied().unwrap_or(0.0),
        FieldComp3d::Hx => hx.get(idx).copied().unwrap_or(0.0),
        FieldComp3d::Hy => hy.get(idx).copied().unwrap_or(0.0),
        FieldComp3d::Hz => hz.get(idx).copied().unwrap_or(0.0),
        FieldComp3d::AbsE => {
            let ex_v = ex.get(idx).copied().unwrap_or(0.0);
            let ey_v = ey.get(idx).copied().unwrap_or(0.0);
            let ez_v = ez.get(idx).copied().unwrap_or(0.0);
            (ex_v * ex_v + ey_v * ey_v + ez_v * ez_v).sqrt()
        }
        FieldComp3d::AbsH => {
            let hx_v = hx.get(idx).copied().unwrap_or(0.0);
            let hy_v = hy.get(idx).copied().unwrap_or(0.0);
            let hz_v = hz.get(idx).copied().unwrap_or(0.0);
            (hx_v * hx_v + hy_v * hy_v + hz_v * hz_v).sqrt()
        }
    }
}

/// 3D field monitor — records field snapshots on a 2D cross-section or full 3D volume.
///
/// Supports time-averaged intensity computation across all stored snapshots.
pub struct FieldMonitor3d {
    /// The region to record
    pub region: MonitorRegion3d,
    /// Which field component to record
    pub component: FieldComp3d,
    /// Stored snapshots
    pub snapshots: Vec<FieldSnapshot3d>,
    /// Record every N steps
    pub record_every: usize,
    /// Internal step counter
    step_count: usize,
    /// Running sum for time-averaged intensity
    intensity_sum: Vec<f64>,
    /// Number of intensity accumulations
    n_accum: usize,
}

impl FieldMonitor3d {
    /// Create a monitor for a Z-constant (XY) slice.
    pub fn new_slice_xy(k: usize, component: FieldComp3d, record_every: usize) -> Self {
        Self::new(MonitorRegion3d::SliceXY { k }, component, record_every)
    }

    /// Create a monitor for a Y-constant (XZ) slice.
    pub fn new_slice_xz(j: usize, component: FieldComp3d, record_every: usize) -> Self {
        Self::new(MonitorRegion3d::SliceXZ { j }, component, record_every)
    }

    /// Create a monitor for an X-constant (YZ) slice.
    pub fn new_slice_yz(i: usize, component: FieldComp3d, record_every: usize) -> Self {
        Self::new(MonitorRegion3d::SliceYZ { i }, component, record_every)
    }

    /// Create a monitor for the full volume.
    pub fn new_full_volume(component: FieldComp3d, record_every: usize) -> Self {
        Self::new(MonitorRegion3d::FullVolume, component, record_every)
    }

    /// Create a monitor with an explicit region specification.
    pub fn new(region: MonitorRegion3d, component: FieldComp3d, record_every: usize) -> Self {
        let record_every = record_every.max(1);
        Self {
            region,
            component,
            snapshots: Vec::new(),
            record_every,
            step_count: 0,
            intensity_sum: Vec::new(),
            n_accum: 0,
        }
    }

    /// Record fields at the current time step.
    ///
    /// Accumulates intensity every step and stores a snapshot every `record_every` steps.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
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
        let data = self.extract_data(nx, ny, nz, ex, ey, ez, hx, hy, hz);

        // Initialize intensity_sum on first call
        if self.intensity_sum.is_empty() {
            self.intensity_sum = vec![0.0; data.len()];
        }

        // Accumulate squared field for time average
        for (acc, &v) in self.intensity_sum.iter_mut().zip(data.iter()) {
            *acc += v * v;
        }
        self.n_accum += 1;

        // Store snapshot if at the record interval
        if self.step_count % self.record_every == 0 {
            let shape = self.region.data_shape(nx, ny, nz);
            self.snapshots
                .push(FieldSnapshot3d::new(time_step, data, shape));
        }
        self.step_count += 1;
    }

    /// Extract data for the monitored region and component.
    #[allow(clippy::too_many_arguments)]
    fn extract_data(
        &self,
        nx: usize,
        ny: usize,
        nz: usize,
        ex: &[f64],
        ey: &[f64],
        ez: &[f64],
        hx: &[f64],
        hy: &[f64],
        hz: &[f64],
    ) -> Vec<f64> {
        let comp = self.component;
        match &self.region {
            MonitorRegion3d::SliceXY { k } => {
                let k = *k;
                if k >= nz {
                    return vec![];
                }
                let mut out = Vec::with_capacity(nx * ny);
                for i in 0..nx {
                    for j in 0..ny {
                        let idx = i * ny * nz + j * nz + k;
                        out.push(extract_component(comp, idx, ex, ey, ez, hx, hy, hz));
                    }
                }
                out
            }
            MonitorRegion3d::SliceXZ { j } => {
                let j = *j;
                if j >= ny {
                    return vec![];
                }
                let mut out = Vec::with_capacity(nx * nz);
                for i in 0..nx {
                    for k in 0..nz {
                        let idx = i * ny * nz + j * nz + k;
                        out.push(extract_component(comp, idx, ex, ey, ez, hx, hy, hz));
                    }
                }
                out
            }
            MonitorRegion3d::SliceYZ { i } => {
                let i = *i;
                if i >= nx {
                    return vec![];
                }
                let mut out = Vec::with_capacity(ny * nz);
                for j in 0..ny {
                    for k in 0..nz {
                        let idx = i * ny * nz + j * nz + k;
                        out.push(extract_component(comp, idx, ex, ey, ez, hx, hy, hz));
                    }
                }
                out
            }
            MonitorRegion3d::FullVolume => {
                let n = nx * ny * nz;
                (0..n)
                    .map(|idx| extract_component(comp, idx, ex, ey, ez, hx, hy, hz))
                    .collect()
            }
            MonitorRegion3d::SubVolume {
                i0,
                i1,
                j0,
                j1,
                k0,
                k1,
            } => {
                let i0 = *i0;
                let i1 = i1.min(&nx);
                let j0 = *j0;
                let j1 = j1.min(&ny);
                let k0 = *k0;
                let k1 = k1.min(&nz);
                let mut out = Vec::with_capacity((i1 - i0) * (j1 - j0) * (k1 - k0));
                for i in i0..*i1 {
                    for j in j0..*j1 {
                        for k in k0..*k1 {
                            let idx = i * ny * nz + j * nz + k;
                            out.push(extract_component(comp, idx, ex, ey, ez, hx, hy, hz));
                        }
                    }
                }
                out
            }
        }
    }

    /// Number of stored snapshots.
    pub fn num_snapshots(&self) -> usize {
        self.snapshots.len()
    }

    /// Get a snapshot by index.
    pub fn get_snapshot(&self, idx: usize) -> Option<&FieldSnapshot3d> {
        self.snapshots.get(idx)
    }

    /// Maximum field value (absolute) across all snapshots.
    pub fn peak_field(&self) -> f64 {
        self.snapshots
            .iter()
            .map(|s| s.peak())
            .fold(0.0_f64, f64::max)
    }

    /// Time-averaged field intensity (field² / n_accum) over the monitored region.
    ///
    /// Returns a flat Vec of averaged squared field values.
    pub fn time_averaged_intensity(&self) -> Vec<f64> {
        if self.n_accum == 0 {
            return vec![];
        }
        self.intensity_sum
            .iter()
            .map(|&s| s / self.n_accum as f64)
            .collect()
    }

    /// Peak time-averaged intensity over the monitored region.
    pub fn peak_averaged_intensity(&self) -> f64 {
        self.time_averaged_intensity()
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max)
    }

    /// Get the last recorded snapshot.
    pub fn last_snapshot(&self) -> Option<&FieldSnapshot3d> {
        self.snapshots.last()
    }

    /// Compute the RMS field value across all snapshots and cells.
    pub fn rms_field(&self) -> f64 {
        let avg_int = self.time_averaged_intensity();
        if avg_int.is_empty() {
            return 0.0;
        }
        let mean_sq = avg_int.iter().sum::<f64>() / avg_int.len() as f64;
        mean_sq.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fields_1d(n: usize, amp: f64) -> (Vec<f64>, Vec<f64>) {
        let ex: Vec<f64> = (0..n).map(|i| amp * (i as f64 / n as f64)).collect();
        let hy: Vec<f64> = (0..n).map(|i| amp * 0.5 * (i as f64 / n as f64)).collect();
        (ex, hy)
    }

    fn make_fields_3d(nx: usize, ny: usize, nz: usize, amp: f64) -> [Vec<f64>; 6] {
        let n = nx * ny * nz;
        let ex: Vec<f64> = (0..n).map(|i| amp * (i as f64 / n as f64)).collect();
        let ey = vec![0.0f64; n];
        let ez = vec![0.0f64; n];
        let hx = vec![0.0f64; n];
        let hy: Vec<f64> = (0..n).map(|i| amp * 0.5 * (i as f64 / n as f64)).collect();
        let hz = vec![0.0f64; n];
        [ex, ey, ez, hx, hy, hz]
    }

    #[test]
    fn field_monitor_1d_records_snapshots() {
        let mut mon = FieldMonitor1d::new(100, 5);
        for step in 0..20usize {
            let (ex, hy) = make_fields_1d(100, 1.0);
            mon.record(step, step as f64 * 1e-15, &ex, &hy);
        }
        // Steps 0, 5, 10, 15 → 4 snapshots
        assert_eq!(mon.n_snapshots(), 4);
    }

    #[test]
    fn field_monitor_1d_intensity_positive() {
        let mut mon = FieldMonitor1d::new(50, 1);
        let (ex, hy) = make_fields_1d(50, 2.0);
        mon.record(0, 0.0, &ex, &hy);
        let avg = mon.time_averaged_intensity();
        assert!(avg.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn field_snapshot_poynting() {
        let snap = FieldSnapshot1d::new(0, 0.0, vec![1.0, 2.0, 3.0], vec![0.5, 1.0, 1.5]);
        let sz = snap.poynting_z();
        assert_eq!(sz.len(), 3);
        assert!((sz[0] - 0.5).abs() < 1e-12);
        assert!((sz[1] - 2.0).abs() < 1e-12);
        assert!((sz[2] - 4.5).abs() < 1e-12);
    }

    #[test]
    fn field_snapshot_2d_slice() {
        let snap = FieldSnapshot2d::new(
            0,
            0.0,
            3,
            4,
            (0..12).map(|i| i as f64).collect(),
            vec![0.0; 12],
            vec![0.0; 12],
        );
        let row = snap.hz_row(1);
        assert_eq!(row.len(), 4);
        assert_eq!(row[0], 4.0);
        assert_eq!(row[3], 7.0);
    }

    #[test]
    fn field_monitor_2d_records_at_interval() {
        let mut mon = FieldMonitor2d::new(10, 10, 3);
        let n = 100;
        let hz = vec![1.0_f64; n];
        let ex = vec![0.5_f64; n];
        let ey = vec![0.5_f64; n];
        for step in 0..9usize {
            mon.record(step, step as f64 * 1e-15, &hz, &ex, &ey);
        }
        // Steps 0, 3, 6 → 3 snapshots
        assert_eq!(mon.n_snapshots(), 3);
    }

    #[test]
    fn field_monitor_1d_no_record_before_any_step() {
        let mon = FieldMonitor1d::new(50, 1);
        assert_eq!(mon.n_snapshots(), 0);
        assert!(mon.last_snapshot().is_none());
        let avg = mon.time_averaged_intensity();
        assert!(avg.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn field_snapshot_peak_ex() {
        let snap = FieldSnapshot1d::new(0, 0.0, vec![-3.0, 1.0, 2.0], vec![0.0; 3]);
        assert!((snap.peak_ex() - 3.0).abs() < 1e-12);
    }

    // 3D monitor tests
    #[test]
    fn field_monitor_3d_slice_xy_records() {
        let nx = 8;
        let ny = 8;
        let nz = 8;
        let mut mon = FieldMonitor3d::new_slice_xy(4, FieldComp3d::Ex, 1);
        let [ex, ey, ez, hx, hy, hz] = make_fields_3d(nx, ny, nz, 1.0);
        mon.record(0, nx, ny, nz, &ex, &ey, &ez, &hx, &hy, &hz);
        assert_eq!(mon.num_snapshots(), 1);
        let snap = mon.get_snapshot(0).expect("should have snapshot");
        assert_eq!(snap.shape, [nx, ny, 1]);
        assert_eq!(snap.data.len(), nx * ny);
    }

    #[test]
    fn field_monitor_3d_slice_xz_records() {
        let nx = 6;
        let ny = 6;
        let nz = 6;
        let mut mon = FieldMonitor3d::new_slice_xz(3, FieldComp3d::Hy, 1);
        let [ex, ey, ez, hx, hy, hz] = make_fields_3d(nx, ny, nz, 1.0);
        mon.record(0, nx, ny, nz, &ex, &ey, &ez, &hx, &hy, &hz);
        assert_eq!(mon.num_snapshots(), 1);
        let snap = mon.get_snapshot(0).expect("should have snapshot");
        assert_eq!(snap.shape, [nx, 1, nz]);
    }

    #[test]
    fn field_monitor_3d_slice_yz_records() {
        let nx = 6;
        let ny = 6;
        let nz = 6;
        let mut mon = FieldMonitor3d::new_slice_yz(2, FieldComp3d::Ez, 1);
        let [ex, ey, ez, hx, hy, hz] = make_fields_3d(nx, ny, nz, 1.0);
        mon.record(0, nx, ny, nz, &ex, &ey, &ez, &hx, &hy, &hz);
        assert_eq!(mon.num_snapshots(), 1);
        let snap = mon.get_snapshot(0).expect("should have snapshot");
        assert_eq!(snap.shape, [1, ny, nz]);
    }

    #[test]
    fn field_monitor_3d_full_volume_records() {
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let mut mon = FieldMonitor3d::new_full_volume(FieldComp3d::AbsE, 1);
        let [ex, ey, ez, hx, hy, hz] = make_fields_3d(nx, ny, nz, 1.0);
        mon.record(0, nx, ny, nz, &ex, &ey, &ez, &hx, &hy, &hz);
        assert_eq!(mon.num_snapshots(), 1);
        let snap = mon.get_snapshot(0).expect("should have snapshot");
        assert_eq!(snap.data.len(), nx * ny * nz);
    }

    #[test]
    fn field_monitor_3d_interval_filtering() {
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let mut mon = FieldMonitor3d::new_slice_xy(2, FieldComp3d::Ex, 5);
        let [ex, ey, ez, hx, hy, hz] = make_fields_3d(nx, ny, nz, 1.0);
        for step in 0..15usize {
            mon.record(step, nx, ny, nz, &ex, &ey, &ez, &hx, &hy, &hz);
        }
        // Steps 0, 5, 10 → 3 snapshots
        assert_eq!(mon.num_snapshots(), 3);
    }

    #[test]
    fn field_monitor_3d_time_averaged_intensity() {
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let mut mon = FieldMonitor3d::new_slice_xy(2, FieldComp3d::Ex, 1);
        let [ex, ey, ez, hx, hy, hz] = make_fields_3d(nx, ny, nz, 2.0);
        for step in 0..10usize {
            mon.record(step, nx, ny, nz, &ex, &ey, &ez, &hx, &hy, &hz);
        }
        let avg = mon.time_averaged_intensity();
        assert!(
            !avg.is_empty(),
            "Time-averaged intensity should not be empty"
        );
        assert!(
            avg.iter().any(|&v| v > 0.0),
            "Should have nonzero intensities"
        );
    }

    #[test]
    fn field_monitor_3d_peak_field() {
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let mut mon = FieldMonitor3d::new_full_volume(FieldComp3d::Ex, 1);
        let [ex, ey, ez, hx, hy, hz] = make_fields_3d(nx, ny, nz, 3.0);
        mon.record(0, nx, ny, nz, &ex, &ey, &ez, &hx, &hy, &hz);
        let peak = mon.peak_field();
        assert!(peak > 0.0, "Peak field should be positive: {peak}");
        assert!(
            peak <= 3.0 + 1e-10,
            "Peak should not exceed amplitude: {peak}"
        );
    }

    #[test]
    fn field_snapshot_3d_at_index() {
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let snap = FieldSnapshot3d::new(0, data, [2, 3, 4]);
        assert_eq!(snap.at(0, 0, 0), Some(0.0));
        assert_eq!(snap.at(1, 2, 3), Some(23.0));
        assert_eq!(snap.at(2, 0, 0), None); // out of bounds
    }

    #[test]
    fn field_monitor_3d_abs_e_nonnegative() {
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let mut mon = FieldMonitor3d::new_slice_xy(2, FieldComp3d::AbsE, 1);
        let [ex, ey, ez, hx, hy, hz] = make_fields_3d(nx, ny, nz, 1.0);
        mon.record(0, nx, ny, nz, &ex, &ey, &ez, &hx, &hy, &hz);
        let snap = mon.get_snapshot(0).expect("should have snapshot");
        assert!(
            snap.data.iter().all(|&v| v >= 0.0),
            "AbsE values should be non-negative"
        );
    }
}
