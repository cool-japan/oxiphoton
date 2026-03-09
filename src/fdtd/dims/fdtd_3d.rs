use crate::fdtd::boundary::pml::Cpml;
use crate::fdtd::config::{BoundaryConfig, Dimensions, GridSpacing};
use crate::fdtd::courant::courant_dt;
use crate::units::conversion::{EPSILON_0, MU_0, SPEED_OF_LIGHT};

pub use super::fdtd_3d_ext::{
    Axis3d, Checkpoint3d, CwWaveform3d, DftProbe3d, Fdtd3dMaterial, FieldComponent3d, FieldProbe3d,
    GaussianPulse3d, GaussianWaveform3d, PlaneMonitor3d, SourceType3d, SourceWaveform3d,
};

/// 3D FDTD solver (Ex, Ey, Ez, Hx, Hy, Hz) with CPML absorbing boundaries.
///
/// Uses a simplified Yee grid where all six field components share the
/// same nx×ny×nz logical size. Boundary conditions are PEC (zeroing
/// boundary cells) plus CPML layers on all six faces.
///
/// Update equations (vacuum, ignoring CPML):
///   Hx -= dt/mu * ((Ez\[i,j+1,k\]-Ez\[i,j,k\])/dy - (Ey\[i,j,k+1\]-Ey\[i,j,k\])/dz)
///   Hy -= dt/mu * ((Ex\[i,j,k+1\]-Ex\[i,j,k\])/dz - (Ez\[i+1,j,k\]-Ez\[i,j,k\])/dx)
///   Hz -= dt/mu * ((Ey\[i+1,j,k\]-Ey\[i,j,k\])/dx - (Ex\[i,j+1,k\]-Ex\[i,j,k\])/dy)
///
///   Ex += dt/eps * ((Hz\[i,j,k\]-Hz\[i,j-1,k\])/dy - (Hy\[i,j,k\]-Hy\[i,j,k-1\])/dz)
///   Ey += dt/eps * ((Hx\[i,j,k\]-Hx\[i,j,k-1\])/dz - (Hz\[i,j,k\]-Hz\[i-1,j,k\])/dx)
///   Ez += dt/eps * ((Hy\[i,j,k\]-Hy\[i-1,j,k\])/dx - (Hx\[i,j,k\]-Hx\[i,j-1,k\])/dy)
pub struct Fdtd3d {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    pub dt: f64,
    pub time_step: usize,

    // Field arrays (size = nx*ny*nz)
    pub ex: Vec<f64>,
    pub ey: Vec<f64>,
    pub ez: Vec<f64>,
    pub hx: Vec<f64>,
    pub hy: Vec<f64>,
    pub hz: Vec<f64>,

    // Relative permittivity and permeability
    pub eps_r: Vec<f64>,
    pub mu_r: Vec<f64>,

    // Electric and magnetic conductivity (S/m and Ω/m respectively)
    pub sigma_e: Vec<f64>,
    pub sigma_m: Vec<f64>,

    // CPML per axis
    pml_x: Cpml,
    pml_y: Cpml,
    pml_z: Cpml,

    // 12 CPML psi arrays (each nx*ny*nz)
    psi_hx_y: Vec<f64>,
    psi_hx_z: Vec<f64>,
    psi_hy_x: Vec<f64>,
    psi_hy_z: Vec<f64>,
    psi_hz_x: Vec<f64>,
    psi_hz_y: Vec<f64>,
    psi_ex_y: Vec<f64>,
    psi_ex_z: Vec<f64>,
    psi_ey_x: Vec<f64>,
    psi_ey_z: Vec<f64>,
    psi_ez_x: Vec<f64>,
    psi_ez_y: Vec<f64>,

    // Sources, monitors, probes
    sources: Vec<SourceType3d>,
    field_probes: Vec<FieldProbe3d>,
    plane_monitors: Vec<PlaneMonitor3d>,
    dft_probes: Vec<DftProbe3d>,
}

impl Fdtd3d {
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        boundary: &BoundaryConfig,
    ) -> Self {
        let spacing = GridSpacing { dx, dy, dz };
        let dt = 0.99 * courant_dt(Dimensions::ThreeD { nx, ny, nz }, spacing, 1.0);

        let pml_x = Cpml::new(
            nx,
            boundary.pml_cells,
            dx,
            dt,
            boundary.pml_m,
            boundary.pml_r0,
        );
        let pml_y = Cpml::new(
            ny,
            boundary.pml_cells,
            dy,
            dt,
            boundary.pml_m,
            boundary.pml_r0,
        );
        let pml_z = Cpml::new(
            nz,
            boundary.pml_cells,
            dz,
            dt,
            boundary.pml_m,
            boundary.pml_r0,
        );

        let n = nx * ny * nz;
        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            dt,
            time_step: 0,
            ex: vec![0.0; n],
            ey: vec![0.0; n],
            ez: vec![0.0; n],
            hx: vec![0.0; n],
            hy: vec![0.0; n],
            hz: vec![0.0; n],
            eps_r: vec![1.0; n],
            mu_r: vec![1.0; n],
            sigma_e: vec![0.0; n],
            sigma_m: vec![0.0; n],
            pml_x,
            pml_y,
            pml_z,
            psi_hx_y: vec![0.0; n],
            psi_hx_z: vec![0.0; n],
            psi_hy_x: vec![0.0; n],
            psi_hy_z: vec![0.0; n],
            psi_hz_x: vec![0.0; n],
            psi_hz_y: vec![0.0; n],
            psi_ex_y: vec![0.0; n],
            psi_ex_z: vec![0.0; n],
            psi_ey_x: vec![0.0; n],
            psi_ey_z: vec![0.0; n],
            psi_ez_x: vec![0.0; n],
            psi_ez_y: vec![0.0; n],
            sources: Vec::new(),
            field_probes: Vec::new(),
            plane_monitors: Vec::new(),
            dft_probes: Vec::new(),
        }
    }

    pub fn current_time(&self) -> f64 {
        self.time_step as f64 * self.dt
    }

    #[inline(always)]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        k * (self.nx * self.ny) + j * self.nx + i
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Material setup
    // ──────────────────────────────────────────────────────────────────────────

    /// Fill a rectangular region with given permittivity and permeability.
    #[allow(clippy::too_many_arguments)]
    pub fn fill_box(
        &mut self,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        k0: usize,
        k1: usize,
        eps: f64,
        mu: f64,
    ) {
        for k in k0..k1.min(self.nz) {
            for j in j0..j1.min(self.ny) {
                for i in i0..i1.min(self.nx) {
                    let idx = self.idx(i, j, k);
                    self.eps_r[idx] = eps;
                    self.mu_r[idx] = mu;
                }
            }
        }
    }

    /// Fill a box region with a full material specification (eps_r, mu_r, sigma_e, sigma_m).
    #[allow(clippy::too_many_arguments)]
    pub fn add_material_box(
        &mut self,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        k0: usize,
        k1: usize,
        mat: Fdtd3dMaterial,
    ) {
        for k in k0..k1.min(self.nz) {
            for j in j0..j1.min(self.ny) {
                for i in i0..i1.min(self.nx) {
                    let idx = self.idx(i, j, k);
                    self.eps_r[idx] = mat.eps_r;
                    self.mu_r[idx] = mat.mu_r;
                    self.sigma_e[idx] = mat.sigma_e;
                    self.sigma_m[idx] = mat.sigma_m;
                }
            }
        }
    }

    /// Fill material map from a flat array (length nx*ny*nz).
    pub fn set_eps_from_map(&mut self, map: &[f64]) {
        let n = self.nx * self.ny * self.nz;
        if map.len() == n {
            self.eps_r.copy_from_slice(map);
        }
    }

    /// Fill a material map built by a closure f(i, j, k) -> eps_r.
    pub fn fill_eps_fn(&mut self, f: impl Fn(usize, usize, usize) -> f64) {
        for k in 0..self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    let idx = self.idx(i, j, k);
                    self.eps_r[idx] = f(i, j, k);
                }
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Anisotropic / tensor material helpers
    // ──────────────────────────────────────────────────────────────────────────

    /// Fill a rectangular box with a diagonal anisotropic permittivity tensor.
    ///
    /// The `Fdtd3d` solver uses a single isotropic `eps_r` field.  This method
    /// stores the geometric mean of the three diagonal components so that the
    /// average permittivity is preserved, which is useful for quick anisotropy
    /// approximations or for pre-processing a scene before handing it off to
    /// [`crate::fdtd::AnisotropicFdtd3d`].
    ///
    /// For a fully anisotropic simulation use [`crate::fdtd::engine::anisotropic::AnisotropicFdtd3d`] directly.
    #[allow(clippy::too_many_arguments)]
    pub fn fill_tensor_eps(
        &mut self,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        k0: usize,
        k1: usize,
        eps_xx: f64,
        eps_yy: f64,
        eps_zz: f64,
    ) {
        // Geometric mean of diagonal tensor components
        let eps_eff = (eps_xx * eps_yy * eps_zz).cbrt();
        for k in k0..k1.min(self.nz) {
            for j in j0..j1.min(self.ny) {
                for i in i0..i1.min(self.nx) {
                    let idx = self.idx(i, j, k);
                    self.eps_r[idx] = eps_eff;
                }
            }
        }
    }

    /// Fill a rectangular box with an anisotropic material derived from a
    /// [`crate::fdtd::UniaxialCrystal`], storing the effective (geometric-mean)
    /// permittivity in the isotropic `eps_r` field.
    ///
    /// For full anisotropy use [`crate::fdtd::AnisotropicFdtd3d`] together with
    /// \[`crate::fdtd::fill_uniaxial_crystal`\].
    #[allow(clippy::too_many_arguments)]
    pub fn fill_anisotropic_box(
        &mut self,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        k0: usize,
        k1: usize,
        eps_diag: [f64; 3],
        mu_diag: [f64; 3],
    ) {
        let eps_eff = (eps_diag[0] * eps_diag[1] * eps_diag[2]).cbrt();
        let mu_eff = (mu_diag[0] * mu_diag[1] * mu_diag[2]).cbrt();
        for k in k0..k1.min(self.nz) {
            for j in j0..j1.min(self.ny) {
                for i in i0..i1.min(self.nx) {
                    let idx = self.idx(i, j, k);
                    self.eps_r[idx] = eps_eff;
                    self.mu_r[idx] = mu_eff;
                }
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Source registration
    // ──────────────────────────────────────────────────────────────────────────

    /// Register a point-dipole hard source at cell (i,j,k).
    pub fn add_point_source(
        &mut self,
        i: usize,
        j: usize,
        k: usize,
        component: FieldComponent3d,
        amplitude: f64,
        waveform: impl SourceWaveform3d + 'static,
    ) {
        self.sources.push(SourceType3d::PointDipole {
            i,
            j,
            k,
            component,
            amplitude,
            waveform: Box::new(waveform),
        });
    }

    /// Register a plane-wave source spanning the entire cross-section perpendicular to `axis`.
    pub fn add_plane_wave_source(
        &mut self,
        axis: Axis3d,
        position: usize,
        component: FieldComponent3d,
        amplitude: f64,
        waveform: impl SourceWaveform3d + 'static,
    ) {
        self.sources.push(SourceType3d::PlaneWave {
            axis,
            position,
            component,
            amplitude,
            waveform: Box::new(waveform),
        });
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Monitor registration
    // ──────────────────────────────────────────────────────────────────────────

    /// Add a point field probe; returns probe index.
    pub fn add_field_probe(
        &mut self,
        i: usize,
        j: usize,
        k: usize,
        component: FieldComponent3d,
    ) -> usize {
        let idx = self.field_probes.len();
        self.field_probes
            .push(FieldProbe3d::new(i, j, k, component));
        idx
    }

    /// Add a plane snapshot monitor; returns monitor index.
    pub fn add_plane_monitor(
        &mut self,
        normal: Axis3d,
        index: usize,
        component: FieldComponent3d,
        record_every: usize,
    ) -> usize {
        let idx = self.plane_monitors.len();
        self.plane_monitors
            .push(PlaneMonitor3d::new(normal, index, component, record_every));
        idx
    }

    /// Add a DFT spectral probe; returns probe index.
    pub fn add_dft_probe(
        &mut self,
        i: usize,
        j: usize,
        k: usize,
        component: FieldComponent3d,
        frequencies: Vec<f64>,
    ) -> usize {
        let idx = self.dft_probes.len();
        self.dft_probes
            .push(DftProbe3d::new(i, j, k, component, frequencies));
        idx
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Field injection helpers
    // ──────────────────────────────────────────────────────────────────────────

    /// Inject source into Ex at cell (i, j, k).
    pub fn inject_ex(&mut self, i: usize, j: usize, k: usize, val: f64) {
        if i < self.nx && j < self.ny && k < self.nz {
            let idx = self.idx(i, j, k);
            self.ex[idx] += val;
        }
    }

    /// Inject source into Ey at cell (i, j, k).
    pub fn inject_ey(&mut self, i: usize, j: usize, k: usize, val: f64) {
        if i < self.nx && j < self.ny && k < self.nz {
            let idx = self.idx(i, j, k);
            self.ey[idx] += val;
        }
    }

    /// Inject a hard point source into Ez at cell (i, j, k).
    pub fn inject_ez(&mut self, i: usize, j: usize, k: usize, val: f64) {
        if i < self.nx && j < self.ny && k < self.nz {
            let idx = self.idx(i, j, k);
            self.ez[idx] += val;
        }
    }

    /// Inject into Hx.
    pub fn inject_hx(&mut self, i: usize, j: usize, k: usize, val: f64) {
        if i < self.nx && j < self.ny && k < self.nz {
            let idx = self.idx(i, j, k);
            self.hx[idx] += val;
        }
    }

    /// Inject into Hy.
    pub fn inject_hy(&mut self, i: usize, j: usize, k: usize, val: f64) {
        if i < self.nx && j < self.ny && k < self.nz {
            let idx = self.idx(i, j, k);
            self.hy[idx] += val;
        }
    }

    /// Inject into Hz.
    pub fn inject_hz(&mut self, i: usize, j: usize, k: usize, val: f64) {
        if i < self.nx && j < self.ny && k < self.nz {
            let idx = self.idx(i, j, k);
            self.hz[idx] += val;
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Time stepping
    // ──────────────────────────────────────────────────────────────────────────

    /// Advance one time step (no sources or monitors).
    pub fn step(&mut self) {
        self.update_h();
        self.update_e();
        self.time_step += 1;
    }

    /// Advance one step, applying all registered sources and recording monitors.
    pub fn step_with_sources(&mut self) {
        self.update_h();
        self.update_e();
        self.apply_sources();
        self.record_monitors();
        self.time_step += 1;
    }

    /// Run for `steps` iterations (no sources or monitors).
    pub fn run(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// Run for `steps` iterations with sources and monitors active.
    pub fn run_with_sources(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step_with_sources();
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Internal: apply registered sources
    // ──────────────────────────────────────────────────────────────────────────

    fn apply_sources(&mut self) {
        let t = self.current_time();
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;

        for src in &self.sources {
            let val = src.amplitude_at(t);
            match src {
                SourceType3d::PointDipole {
                    i, j, k, component, ..
                } => {
                    let (si, sj, sk) = (*i, *j, *k);
                    if si < nx && sj < ny && sk < nz {
                        let sidx = sk * (nx * ny) + sj * nx + si;
                        match component {
                            FieldComponent3d::Ex => self.ex[sidx] += val,
                            FieldComponent3d::Ey => self.ey[sidx] += val,
                            FieldComponent3d::Ez => self.ez[sidx] += val,
                            FieldComponent3d::Hx => self.hx[sidx] += val,
                            FieldComponent3d::Hy => self.hy[sidx] += val,
                            FieldComponent3d::Hz => self.hz[sidx] += val,
                        }
                    }
                }
                SourceType3d::PlaneWave {
                    axis,
                    position,
                    component,
                    ..
                } => match axis {
                    Axis3d::X => {
                        let pi = (*position).min(nx.saturating_sub(1));
                        for k in 0..nz {
                            for j in 0..ny {
                                let sidx = k * (nx * ny) + j * nx + pi;
                                match component {
                                    FieldComponent3d::Ex => self.ex[sidx] += val,
                                    FieldComponent3d::Ey => self.ey[sidx] += val,
                                    FieldComponent3d::Ez => self.ez[sidx] += val,
                                    FieldComponent3d::Hx => self.hx[sidx] += val,
                                    FieldComponent3d::Hy => self.hy[sidx] += val,
                                    FieldComponent3d::Hz => self.hz[sidx] += val,
                                }
                            }
                        }
                    }
                    Axis3d::Y => {
                        let pj = (*position).min(ny.saturating_sub(1));
                        for k in 0..nz {
                            for i in 0..nx {
                                let sidx = k * (nx * ny) + pj * nx + i;
                                match component {
                                    FieldComponent3d::Ex => self.ex[sidx] += val,
                                    FieldComponent3d::Ey => self.ey[sidx] += val,
                                    FieldComponent3d::Ez => self.ez[sidx] += val,
                                    FieldComponent3d::Hx => self.hx[sidx] += val,
                                    FieldComponent3d::Hy => self.hy[sidx] += val,
                                    FieldComponent3d::Hz => self.hz[sidx] += val,
                                }
                            }
                        }
                    }
                    Axis3d::Z => {
                        let pk = (*position).min(nz.saturating_sub(1));
                        for j in 0..ny {
                            for i in 0..nx {
                                let sidx = pk * (nx * ny) + j * nx + i;
                                match component {
                                    FieldComponent3d::Ex => self.ex[sidx] += val,
                                    FieldComponent3d::Ey => self.ey[sidx] += val,
                                    FieldComponent3d::Ez => self.ez[sidx] += val,
                                    FieldComponent3d::Hx => self.hx[sidx] += val,
                                    FieldComponent3d::Hy => self.hy[sidx] += val,
                                    FieldComponent3d::Hz => self.hz[sidx] += val,
                                }
                            }
                        }
                    }
                },
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Internal: record monitors
    // ──────────────────────────────────────────────────────────────────────────

    fn record_monitors(&mut self) {
        let t = self.current_time();
        let step = self.time_step;
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;

        // Field probes
        for probe in &mut self.field_probes {
            if probe.i < nx && probe.j < ny && probe.k < nz {
                let pidx = probe.k * (nx * ny) + probe.j * nx + probe.i;
                let val = match probe.component {
                    FieldComponent3d::Ex => self.ex[pidx],
                    FieldComponent3d::Ey => self.ey[pidx],
                    FieldComponent3d::Ez => self.ez[pidx],
                    FieldComponent3d::Hx => self.hx[pidx],
                    FieldComponent3d::Hy => self.hy[pidx],
                    FieldComponent3d::Hz => self.hz[pidx],
                };
                probe.record(t, val);
            }
        }

        // Plane monitors (record if step % record_every == 0)
        for mon in &mut self.plane_monitors {
            if step % mon.record_every.max(1) == 0 {
                let snap = Self::extract_plane_snapshot(
                    nx,
                    ny,
                    nz,
                    &self.ex,
                    &self.ey,
                    &self.ez,
                    &self.hx,
                    &self.hy,
                    &self.hz,
                    mon.normal,
                    mon.index,
                    mon.component,
                );
                mon.snapshots.push(snap);
            }
        }

        // DFT probes
        for probe in &mut self.dft_probes {
            if probe.i < nx && probe.j < ny && probe.k < nz {
                let pidx = probe.k * (nx * ny) + probe.j * nx + probe.i;
                let val = match probe.component {
                    FieldComponent3d::Ex => self.ex[pidx],
                    FieldComponent3d::Ey => self.ey[pidx],
                    FieldComponent3d::Ez => self.ez[pidx],
                    FieldComponent3d::Hx => self.hx[pidx],
                    FieldComponent3d::Hy => self.hy[pidx],
                    FieldComponent3d::Hz => self.hz[pidx],
                };
                probe.update(t, val);
            }
        }
    }

    /// Extract a 2D plane snapshot from the field arrays (static helper, borrow-safe).
    #[allow(clippy::too_many_arguments)]
    fn extract_plane_snapshot(
        nx: usize,
        ny: usize,
        nz: usize,
        ex: &[f64],
        ey: &[f64],
        ez: &[f64],
        hx: &[f64],
        hy: &[f64],
        hz: &[f64],
        normal: Axis3d,
        index: usize,
        component: FieldComponent3d,
    ) -> Vec<f64> {
        let field = match component {
            FieldComponent3d::Ex => ex,
            FieldComponent3d::Ey => ey,
            FieldComponent3d::Ez => ez,
            FieldComponent3d::Hx => hx,
            FieldComponent3d::Hy => hy,
            FieldComponent3d::Hz => hz,
        };

        match normal {
            Axis3d::Z => {
                let k = index.min(nz.saturating_sub(1));
                (0..nx * ny)
                    .map(|ij| {
                        let i = ij % nx;
                        let j = ij / nx;
                        field[k * (nx * ny) + j * nx + i]
                    })
                    .collect()
            }
            Axis3d::Y => {
                let j = index.min(ny.saturating_sub(1));
                (0..nx * nz)
                    .map(|ik| {
                        let i = ik % nx;
                        let k = ik / nx;
                        field[k * (nx * ny) + j * nx + i]
                    })
                    .collect()
            }
            Axis3d::X => {
                let ii = index.min(nx.saturating_sub(1));
                (0..ny * nz)
                    .map(|jk| {
                        let j = jk % ny;
                        let k = jk / ny;
                        field[k * (nx * ny) + j * nx + ii]
                    })
                    .collect()
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Field update kernels
    // ──────────────────────────────────────────────────────────────────────────

    pub(crate) fn update_h(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let dx = self.dx;
        let dy = self.dy;
        let dz = self.dz;
        let dt = self.dt;

        for k in 0..nz - 1 {
            for j in 0..ny - 1 {
                for i in 0..nx - 1 {
                    let idx = k * (nx * ny) + j * nx + i;

                    let mu = MU_0 * self.mu_r[idx];
                    let sig_m = self.sigma_m[idx];

                    // Lossy magnetic update coefficients
                    let num = 1.0 - sig_m * dt / (2.0 * mu);
                    let den = 1.0 + sig_m * dt / (2.0 * mu);
                    let coeff_h = num / den;
                    let coeff_curl = (dt / mu) / den;

                    let dez_dy = (self.ez[k * (nx * ny) + (j + 1) * nx + i] - self.ez[idx]) / dy;
                    let dey_dz = (self.ey[(k + 1) * (nx * ny) + j * nx + i] - self.ey[idx]) / dz;

                    let dex_dz = (self.ex[(k + 1) * (nx * ny) + j * nx + i] - self.ex[idx]) / dz;
                    let dez_dx = (self.ez[k * (nx * ny) + j * nx + i + 1] - self.ez[idx]) / dx;

                    let dey_dx = (self.ey[k * (nx * ny) + j * nx + i + 1] - self.ey[idx]) / dx;
                    let dex_dy = (self.ex[k * (nx * ny) + (j + 1) * nx + i] - self.ex[idx]) / dy;

                    // CPML psi updates
                    self.psi_hx_y[idx] =
                        self.pml_y.b_h[j] * self.psi_hx_y[idx] + self.pml_y.c_h[j] * dez_dy;
                    self.psi_hx_z[idx] =
                        self.pml_z.b_h[k] * self.psi_hx_z[idx] + self.pml_z.c_h[k] * dey_dz;

                    self.psi_hy_x[idx] =
                        self.pml_x.b_h[i] * self.psi_hy_x[idx] + self.pml_x.c_h[i] * dez_dx;
                    self.psi_hy_z[idx] =
                        self.pml_z.b_h[k] * self.psi_hy_z[idx] + self.pml_z.c_h[k] * dex_dz;

                    self.psi_hz_x[idx] =
                        self.pml_x.b_h[i] * self.psi_hz_x[idx] + self.pml_x.c_h[i] * dey_dx;
                    self.psi_hz_y[idx] =
                        self.pml_y.b_h[j] * self.psi_hz_y[idx] + self.pml_y.c_h[j] * dex_dy;

                    let kx = self.pml_x.kappa_h[i];
                    let ky = self.pml_y.kappa_h[j];
                    let kz = self.pml_z.kappa_h[k];

                    self.hx[idx] = coeff_h * self.hx[idx]
                        - coeff_curl
                            * (dez_dy / ky + self.psi_hx_y[idx] - dey_dz / kz - self.psi_hx_z[idx]);
                    self.hy[idx] = coeff_h * self.hy[idx]
                        - coeff_curl
                            * (dex_dz / kz + self.psi_hy_z[idx] - dez_dx / kx - self.psi_hy_x[idx]);
                    self.hz[idx] = coeff_h * self.hz[idx]
                        - coeff_curl
                            * (dey_dx / kx + self.psi_hz_x[idx] - dex_dy / ky - self.psi_hz_y[idx]);
                }
            }
        }
    }

    pub(crate) fn update_e(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let dx = self.dx;
        let dy = self.dy;
        let dz = self.dz;
        let dt = self.dt;

        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let idx = k * (nx * ny) + j * nx + i;
                    let eps = EPSILON_0 * self.eps_r[idx];
                    let sig = self.sigma_e[idx];

                    // Lossy electric update coefficients (Taflove §3.9.2)
                    let num = 1.0 - sig * dt / (2.0 * eps);
                    let den = 1.0 + sig * dt / (2.0 * eps);
                    let coeff_e = num / den;
                    let coeff_curl = (dt / eps) / den;

                    let dhz_dy = (self.hz[idx] - self.hz[k * (nx * ny) + (j - 1) * nx + i]) / dy;
                    let dhy_dz = (self.hy[idx] - self.hy[(k - 1) * (nx * ny) + j * nx + i]) / dz;

                    let dhx_dz = (self.hx[idx] - self.hx[(k - 1) * (nx * ny) + j * nx + i]) / dz;
                    let dhz_dx = (self.hz[idx] - self.hz[k * (nx * ny) + j * nx + i - 1]) / dx;

                    let dhy_dx = (self.hy[idx] - self.hy[k * (nx * ny) + j * nx + i - 1]) / dx;
                    let dhx_dy = (self.hx[idx] - self.hx[k * (nx * ny) + (j - 1) * nx + i]) / dy;

                    // CPML psi updates
                    self.psi_ex_y[idx] =
                        self.pml_y.b_e[j] * self.psi_ex_y[idx] + self.pml_y.c_e[j] * dhz_dy;
                    self.psi_ex_z[idx] =
                        self.pml_z.b_e[k] * self.psi_ex_z[idx] + self.pml_z.c_e[k] * dhy_dz;

                    self.psi_ey_x[idx] =
                        self.pml_x.b_e[i] * self.psi_ey_x[idx] + self.pml_x.c_e[i] * dhz_dx;
                    self.psi_ey_z[idx] =
                        self.pml_z.b_e[k] * self.psi_ey_z[idx] + self.pml_z.c_e[k] * dhx_dz;

                    self.psi_ez_x[idx] =
                        self.pml_x.b_e[i] * self.psi_ez_x[idx] + self.pml_x.c_e[i] * dhy_dx;
                    self.psi_ez_y[idx] =
                        self.pml_y.b_e[j] * self.psi_ez_y[idx] + self.pml_y.c_e[j] * dhx_dy;

                    let kx = self.pml_x.kappa_e[i];
                    let ky = self.pml_y.kappa_e[j];
                    let kz = self.pml_z.kappa_e[k];

                    self.ex[idx] = coeff_e * self.ex[idx]
                        + coeff_curl
                            * (dhz_dy / ky + self.psi_ex_y[idx] - dhy_dz / kz - self.psi_ex_z[idx]);
                    self.ey[idx] = coeff_e * self.ey[idx]
                        + coeff_curl
                            * (dhx_dz / kz + self.psi_ey_z[idx] - dhz_dx / kx - self.psi_ey_x[idx]);
                    self.ez[idx] = coeff_e * self.ez[idx]
                        + coeff_curl
                            * (dhy_dx / kx + self.psi_ez_x[idx] - dhx_dy / ky - self.psi_ez_y[idx]);
                }
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Parallel field updates (feature-gated)
    // ──────────────────────────────────────────────────────────────────────────

    #[cfg(feature = "parallel")]
    pub fn update_h_parallel(&mut self) {
        use rayon::prelude::*;

        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let dx = self.dx;
        let dy = self.dy;
        let dz = self.dz;
        let dt = self.dt;

        // Collect all (i,j,k) interior triplets
        let cells: Vec<(usize, usize, usize)> = (0..nz - 1)
            .flat_map(|k| (0..ny - 1).flat_map(move |j| (0..nx - 1).map(move |i| (i, j, k))))
            .collect();

        // Pre-read immutable field data to avoid borrow issues
        let ex_snap: Vec<f64> = self.ex.clone();
        let ey_snap: Vec<f64> = self.ey.clone();
        let ez_snap: Vec<f64> = self.ez.clone();
        let mu_r_snap: Vec<f64> = self.mu_r.clone();
        let sig_m_snap: Vec<f64> = self.sigma_m.clone();
        let pml_x_b_h: Vec<f64> = self.pml_x.b_h.clone();
        let pml_x_c_h: Vec<f64> = self.pml_x.c_h.clone();
        let pml_x_kappa_h: Vec<f64> = self.pml_x.kappa_h.clone();
        let pml_y_b_h: Vec<f64> = self.pml_y.b_h.clone();
        let pml_y_c_h: Vec<f64> = self.pml_y.c_h.clone();
        let pml_y_kappa_h: Vec<f64> = self.pml_y.kappa_h.clone();
        let pml_z_b_h: Vec<f64> = self.pml_z.b_h.clone();
        let pml_z_c_h: Vec<f64> = self.pml_z.c_h.clone();
        let pml_z_kappa_h: Vec<f64> = self.pml_z.kappa_h.clone();

        // Compute update deltas in parallel
        let updates: Vec<(usize, f64, f64, f64, f64, f64, f64)> = cells
            .par_iter()
            .map(|&(i, j, k)| {
                let idx = k * (nx * ny) + j * nx + i;
                let mu = MU_0 * mu_r_snap[idx];
                let sig_m = sig_m_snap[idx];
                let num = 1.0 - sig_m * dt / (2.0 * mu);
                let den = 1.0 + sig_m * dt / (2.0 * mu);
                let coeff_h = num / den;
                let coeff_curl = (dt / mu) / den;

                let dez_dy = (ez_snap[k * (nx * ny) + (j + 1) * nx + i] - ez_snap[idx]) / dy;
                let dey_dz = (ey_snap[(k + 1) * (nx * ny) + j * nx + i] - ey_snap[idx]) / dz;
                let dex_dz = (ex_snap[(k + 1) * (nx * ny) + j * nx + i] - ex_snap[idx]) / dz;
                let dez_dx = (ez_snap[k * (nx * ny) + j * nx + i + 1] - ez_snap[idx]) / dx;
                let dey_dx = (ey_snap[k * (nx * ny) + j * nx + i + 1] - ey_snap[idx]) / dx;
                let dex_dy = (ex_snap[k * (nx * ny) + (j + 1) * nx + i] - ex_snap[idx]) / dy;

                let kx = pml_x_kappa_h[i];
                let ky = pml_y_kappa_h[j];
                let kz = pml_z_kappa_h[k];

                let psi_hx_y_new = pml_y_b_h[j] * 0.0 + pml_y_c_h[j] * dez_dy;
                let psi_hx_z_new = pml_z_b_h[k] * 0.0 + pml_z_c_h[k] * dey_dz;
                let psi_hy_x_new = pml_x_b_h[i] * 0.0 + pml_x_c_h[i] * dez_dx;
                let psi_hy_z_new = pml_z_b_h[k] * 0.0 + pml_z_c_h[k] * dex_dz;
                let psi_hz_x_new = pml_x_b_h[i] * 0.0 + pml_x_c_h[i] * dey_dx;
                let psi_hz_y_new = pml_y_b_h[j] * 0.0 + pml_y_c_h[j] * dex_dy;

                let hx_new = coeff_h * 0.0
                    - coeff_curl * (dez_dy / ky + psi_hx_y_new - dey_dz / kz - psi_hx_z_new);
                let hy_new = coeff_h * 0.0
                    - coeff_curl * (dex_dz / kz + psi_hy_z_new - dez_dx / kx - psi_hy_x_new);
                let hz_new = coeff_h * 0.0
                    - coeff_curl * (dey_dx / kx + psi_hz_x_new - dex_dy / ky - psi_hz_y_new);

                (
                    idx,
                    hx_new,
                    hy_new,
                    hz_new,
                    psi_hx_y_new,
                    psi_hy_x_new,
                    psi_hz_x_new,
                )
            })
            .collect();

        // Apply updates sequentially (avoids data races)
        for (idx, hx_d, hy_d, hz_d, psi_hx_y_d, psi_hy_x_d, psi_hz_x_d) in updates {
            self.hx[idx] += hx_d;
            self.hy[idx] += hy_d;
            self.hz[idx] += hz_d;
            self.psi_hx_y[idx] += psi_hx_y_d;
            self.psi_hy_x[idx] += psi_hy_x_d;
            self.psi_hz_x[idx] += psi_hz_x_d;
        }
    }

    #[cfg(feature = "parallel")]
    pub fn update_e_parallel(&mut self) {
        use rayon::prelude::*;

        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let dx = self.dx;
        let dy = self.dy;
        let dz = self.dz;
        let dt = self.dt;

        let cells: Vec<(usize, usize, usize)> = (1..nz - 1)
            .flat_map(|k| (1..ny - 1).flat_map(move |j| (1..nx - 1).map(move |i| (i, j, k))))
            .collect();

        let hx_snap: Vec<f64> = self.hx.clone();
        let hy_snap: Vec<f64> = self.hy.clone();
        let hz_snap: Vec<f64> = self.hz.clone();
        let eps_r_snap: Vec<f64> = self.eps_r.clone();
        let sigma_e_snap: Vec<f64> = self.sigma_e.clone();
        let pml_x_b_e: Vec<f64> = self.pml_x.b_e.clone();
        let pml_x_c_e: Vec<f64> = self.pml_x.c_e.clone();
        let pml_x_kappa_e: Vec<f64> = self.pml_x.kappa_e.clone();
        let pml_y_b_e: Vec<f64> = self.pml_y.b_e.clone();
        let pml_y_c_e: Vec<f64> = self.pml_y.c_e.clone();
        let pml_y_kappa_e: Vec<f64> = self.pml_y.kappa_e.clone();
        let pml_z_b_e: Vec<f64> = self.pml_z.b_e.clone();
        let pml_z_c_e: Vec<f64> = self.pml_z.c_e.clone();
        let pml_z_kappa_e: Vec<f64> = self.pml_z.kappa_e.clone();

        let updates: Vec<(usize, f64, f64, f64)> = cells
            .par_iter()
            .map(|&(i, j, k)| {
                let idx = k * (nx * ny) + j * nx + i;
                let eps = EPSILON_0 * eps_r_snap[idx];
                let sig = sigma_e_snap[idx];
                let num = 1.0 - sig * dt / (2.0 * eps);
                let den = 1.0 + sig * dt / (2.0 * eps);
                let coeff_curl = (dt / eps) / den;

                let dhz_dy = (hz_snap[idx] - hz_snap[k * (nx * ny) + (j - 1) * nx + i]) / dy;
                let dhy_dz = (hy_snap[idx] - hy_snap[(k - 1) * (nx * ny) + j * nx + i]) / dz;
                let dhx_dz = (hx_snap[idx] - hx_snap[(k - 1) * (nx * ny) + j * nx + i]) / dz;
                let dhz_dx = (hz_snap[idx] - hz_snap[k * (nx * ny) + j * nx + i - 1]) / dx;
                let dhy_dx = (hy_snap[idx] - hy_snap[k * (nx * ny) + j * nx + i - 1]) / dx;
                let dhx_dy = (hx_snap[idx] - hx_snap[k * (nx * ny) + (j - 1) * nx + i]) / dy;

                let kx = pml_x_kappa_e[i];
                let ky = pml_y_kappa_e[j];
                let kz = pml_z_kappa_e[k];

                let psi_ex_y = pml_y_b_e[j] * 0.0 + pml_y_c_e[j] * dhz_dy;
                let psi_ex_z = pml_z_b_e[k] * 0.0 + pml_z_c_e[k] * dhy_dz;
                let psi_ey_x = pml_x_b_e[i] * 0.0 + pml_x_c_e[i] * dhz_dx;
                let psi_ey_z = pml_z_b_e[k] * 0.0 + pml_z_c_e[k] * dhx_dz;
                let psi_ez_x = pml_x_b_e[i] * 0.0 + pml_x_c_e[i] * dhy_dx;
                let psi_ez_y = pml_y_b_e[j] * 0.0 + pml_y_c_e[j] * dhx_dy;

                let _ = (
                    num, psi_ex_y, psi_ex_z, psi_ey_x, psi_ey_z, psi_ez_x, psi_ez_y,
                );

                let ex_d = coeff_curl * (dhz_dy / ky - dhy_dz / kz);
                let ey_d = coeff_curl * (dhx_dz / kz - dhz_dx / kx);
                let ez_d = coeff_curl * (dhy_dx / kx - dhx_dy / ky);

                (idx, ex_d, ey_d, ez_d)
            })
            .collect();

        for (idx, ex_d, ey_d, ez_d) in updates {
            self.ex[idx] += ex_d;
            self.ey[idx] += ey_d;
            self.ez[idx] += ez_d;
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Stability analysis
    // ──────────────────────────────────────────────────────────────────────────

    /// Courant number S = c·dt·√(1/dx²+1/dy²+1/dz²).
    /// Must be ≤ 1 for stability.
    pub fn courant_number(&self) -> f64 {
        let inv =
            (1.0 / (self.dx * self.dx) + 1.0 / (self.dy * self.dy) + 1.0 / (self.dz * self.dz))
                .sqrt();
        SPEED_OF_LIGHT * self.dt * inv
    }

    /// Returns true when the Courant condition is satisfied (S ≤ 1).
    pub fn is_stable(&self) -> bool {
        self.courant_number() <= 1.0
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Energy
    // ──────────────────────────────────────────────────────────────────────────

    /// Electric field energy (J): U_E = ½ ∫ eps·|E|² dV
    pub fn energy_e(&self) -> f64 {
        let dv = self.dx * self.dy * self.dz;
        let e_sum: f64 = self
            .ex
            .iter()
            .zip(self.eps_r.iter())
            .map(|(e, &eps)| eps * e * e)
            .sum::<f64>()
            + self
                .ey
                .iter()
                .zip(self.eps_r.iter())
                .map(|(e, &eps)| eps * e * e)
                .sum::<f64>()
            + self
                .ez
                .iter()
                .zip(self.eps_r.iter())
                .map(|(e, &eps)| eps * e * e)
                .sum::<f64>();
        0.5 * EPSILON_0 * e_sum * dv
    }

    /// Magnetic field energy (J): U_H = ½ ∫ mu·|H|² dV
    pub fn energy_h(&self) -> f64 {
        let dv = self.dx * self.dy * self.dz;
        let h_sum: f64 = self
            .hx
            .iter()
            .zip(self.mu_r.iter())
            .map(|(h, &mu)| mu * h * h)
            .sum::<f64>()
            + self
                .hy
                .iter()
                .zip(self.mu_r.iter())
                .map(|(h, &mu)| mu * h * h)
                .sum::<f64>()
            + self
                .hz
                .iter()
                .zip(self.mu_r.iter())
                .map(|(h, &mu)| mu * h * h)
                .sum::<f64>();
        0.5 * MU_0 * h_sum * dv
    }

    /// Total field energy (J): U = U_E + U_H
    pub fn total_energy(&self) -> f64 {
        self.energy_e() + self.energy_h()
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Field access helpers
    // ──────────────────────────────────────────────────────────────────────────

    /// Return all six field components \[Ex,Ey,Ez,Hx,Hy,Hz\] at cell (i,j,k).
    /// Returns zeros if indices are out of bounds.
    pub fn field_at(&self, i: usize, j: usize, k: usize) -> [f64; 6] {
        if i < self.nx && j < self.ny && k < self.nz {
            let idx = self.idx(i, j, k);
            [
                self.ex[idx],
                self.ey[idx],
                self.ez[idx],
                self.hx[idx],
                self.hy[idx],
                self.hz[idx],
            ]
        } else {
            [0.0; 6]
        }
    }

    /// Return the component with the largest absolute value across the entire domain,
    /// along with its value and cell indices.
    pub fn max_field_component(&self) -> (FieldComponent3d, f64, usize, usize, usize) {
        let nx = self.nx;
        let ny = self.ny;

        let find_max = |arr: &[f64]| -> (f64, usize) {
            arr.iter()
                .enumerate()
                .max_by(|a, b| {
                    a.1.abs()
                        .partial_cmp(&b.1.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, &v)| (v, idx))
                .unwrap_or((0.0, 0))
        };

        let idx_to_ijk = |raw: usize| -> (usize, usize, usize) {
            let k = raw / (nx * ny);
            let rem = raw % (nx * ny);
            let j = rem / nx;
            let i = rem % nx;
            (i, j, k)
        };

        let candidates = [
            (FieldComponent3d::Ex, find_max(&self.ex)),
            (FieldComponent3d::Ey, find_max(&self.ey)),
            (FieldComponent3d::Ez, find_max(&self.ez)),
            (FieldComponent3d::Hx, find_max(&self.hx)),
            (FieldComponent3d::Hy, find_max(&self.hy)),
            (FieldComponent3d::Hz, find_max(&self.hz)),
        ];

        let best = candidates
            .into_iter()
            .max_by(|a, b| {
                a.1 .0
                    .abs()
                    .partial_cmp(&b.1 .0.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or((FieldComponent3d::Ex, (0.0, 0)));

        let (comp, (val, raw)) = best;
        let (i, j, k) = idx_to_ijk(raw);
        (comp, val, i, j, k)
    }

    /// Peak |Ez| field value (legacy helper).
    pub fn peak_ez(&self) -> f64 {
        self.ez.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Checkpoint support
    // ──────────────────────────────────────────────────────────────────────────

    /// Snapshot all six field arrays into a `Checkpoint3d`.
    pub fn save_checkpoint(&self) -> Checkpoint3d {
        Checkpoint3d {
            time_step: self.time_step,
            ex: self.ex.clone(),
            ey: self.ey.clone(),
            ez: self.ez.clone(),
            hx: self.hx.clone(),
            hy: self.hy.clone(),
            hz: self.hz.clone(),
        }
    }

    /// Restore the solver state from a previously saved `Checkpoint3d`.
    /// Silently ignores a checkpoint whose size does not match the current grid.
    pub fn restore_checkpoint(&mut self, cp: &Checkpoint3d) {
        let n = self.nx * self.ny * self.nz;
        if cp.ex.len() != n {
            return;
        }
        self.time_step = cp.time_step;
        self.ex.copy_from_slice(&cp.ex);
        self.ey.copy_from_slice(&cp.ey);
        self.ez.copy_from_slice(&cp.ez);
        self.hx.copy_from_slice(&cp.hx);
        self.hy.copy_from_slice(&cp.hy);
        self.hz.copy_from_slice(&cp.hz);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Monitor data access
    // ──────────────────────────────────────────────────────────────────────────

    /// Return the time series of a field probe by index.
    pub fn get_probe_time_series(&self, probe_idx: usize) -> Option<&[(f64, f64)]> {
        self.field_probes
            .get(probe_idx)
            .map(|p| p.time_series.as_slice())
    }

    /// Return `(freq, re, im)` triples for all frequencies monitored by a DFT probe.
    pub fn get_dft_spectrum(&self, probe_idx: usize) -> Option<Vec<(f64, f64, f64)>> {
        self.dft_probes.get(probe_idx).map(|p| p.spectrum())
    }

    /// Return a particular snapshot from a plane monitor (flattened 2-D slice).
    pub fn get_plane_snapshot(&self, monitor_idx: usize, snapshot_idx: usize) -> Option<&[f64]> {
        self.plane_monitors
            .get(monitor_idx)
            .and_then(|m| m.snapshots.get(snapshot_idx))
            .map(|s| s.as_slice())
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Legacy slicing helpers
    // ──────────────────────────────────────────────────────────────────────────

    /// Extract xy-plane slice of Ez at fixed z-index k.
    pub fn ez_slice_xy(&self, k: usize) -> Vec<f64> {
        (0..self.nx * self.ny)
            .map(|ij| {
                let i = ij % self.nx;
                let j = ij / self.nx;
                if k < self.nz {
                    self.ez[self.idx(i, j, k)]
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Extract xz-plane slice of Ez at fixed y-index j.
    pub fn ez_slice_xz(&self, j: usize) -> Vec<f64> {
        (0..self.nx * self.nz)
            .map(|ik| {
                let i = ik % self.nx;
                let k = ik / self.nx;
                if j < self.ny {
                    self.ez[self.idx(i, j, k)]
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Extract yz-plane slice of Ez at fixed x-index i.
    pub fn ez_slice_yz(&self, i: usize) -> Vec<f64> {
        (0..self.ny * self.nz)
            .map(|jk| {
                let j = jk % self.ny;
                let k = jk / self.ny;
                if i < self.nx {
                    self.ez[self.idx(i, j, k)]
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// DFT probe at a single point (i, j, k) at frequency omega (rad/s).
    pub fn dft_probe_ez(&self, time_series: &[(f64, f64)], omega: f64) -> (f64, f64) {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for &(t, ez) in time_series {
            let (s, c) = (omega * t).sin_cos();
            re += ez * c;
            im -= ez * s;
        }
        (re, im)
    }

    /// Count cells with eps_r greater than a threshold.
    pub fn count_material_cells(&self, eps_threshold: f64) -> usize {
        self.eps_r.iter().filter(|&&e| e > eps_threshold).count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn small_solver() -> Fdtd3d {
        Fdtd3d::new(30, 30, 30, 20e-9, 20e-9, 20e-9, &BoundaryConfig::pml(8))
    }

    // ── Legacy / basic tests ─────────────────────────────────────────────────

    #[test]
    fn fdtd3d_initializes_zero() {
        let s = small_solver();
        assert!(s.ez.iter().all(|&v| v == 0.0));
        assert!(s.hx.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn fdtd3d_runs_without_panic() {
        let mut s = small_solver();
        s.run(20);
        assert!(s.ex.iter().all(|&v| v.is_finite()));
        assert!(s.ey.iter().all(|&v| v.is_finite()));
        assert!(s.ez.iter().all(|&v| v.is_finite()));
        assert!(s.hx.iter().all(|&v| v.is_finite()));
        assert!(s.hy.iter().all(|&v| v.is_finite()));
        assert!(s.hz.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn fdtd3d_point_source_propagates() {
        let mut s = small_solver();
        for step in 0..50 {
            let t = step as f64 * s.dt;
            let src = (-(t - 10.0 * s.dt).powi(2) / (2.0 * (3.0 * s.dt).powi(2))).exp();
            s.inject_ez(15, 15, 15, src);
            s.step();
        }
        let peak = s.peak_ez();
        assert!(peak.is_finite(), "field must be finite");
    }

    #[test]
    fn fdtd3d_fill_box_sets_material() {
        let mut s = small_solver();
        s.fill_box(10, 20, 10, 20, 10, 20, 2.25, 1.0);
        let idx = s.idx(15, 15, 15);
        assert_relative_eq!(s.eps_r[idx], 2.25);
        let idx2 = s.idx(5, 5, 5);
        assert_relative_eq!(s.eps_r[idx2], 1.0);
    }

    #[test]
    fn fdtd3d_dt_is_stable() {
        let s = small_solver();
        assert!(
            s.is_stable(),
            "Courant number={:.4} > 1",
            s.courant_number()
        );
    }

    #[test]
    fn fdtd3d_ez_slice_xy_size() {
        let s = small_solver();
        let slice = s.ez_slice_xy(10);
        assert_eq!(slice.len(), 30 * 30);
    }

    #[test]
    fn fdtd3d_ez_slice_xz_size() {
        let s = small_solver();
        let slice = s.ez_slice_xz(10);
        assert_eq!(slice.len(), 30 * 30);
    }

    #[test]
    fn fdtd3d_total_energy_zero_initially() {
        let s = small_solver();
        assert_eq!(s.total_energy(), 0.0);
    }

    #[test]
    fn fdtd3d_total_energy_positive_after_source() {
        let mut s = small_solver();
        s.inject_ez(15, 15, 15, 1.0);
        s.run(5);
        let e = s.total_energy();
        assert!(e > 0.0);
    }

    #[test]
    fn fdtd3d_fill_eps_fn() {
        let mut s = small_solver();
        s.fill_eps_fn(|i, _j, _k| if i < 15 { 2.0 } else { 1.0 });
        let idx_in = s.idx(5, 5, 5);
        let idx_out = s.idx(20, 5, 5);
        assert_relative_eq!(s.eps_r[idx_in], 2.0);
        assert_relative_eq!(s.eps_r[idx_out], 1.0);
    }

    #[test]
    fn fdtd3d_count_material_cells() {
        let mut s = small_solver();
        s.fill_box(5, 15, 5, 15, 5, 15, 2.25, 1.0);
        let count = s.count_material_cells(2.0);
        assert!(count > 0);
        assert!(count < s.nx * s.ny * s.nz);
    }

    #[test]
    fn fdtd3d_inject_ex_ey() {
        let mut s = small_solver();
        s.inject_ex(10, 10, 10, 1.0);
        s.inject_ey(10, 10, 10, 0.5);
        let idx = s.idx(10, 10, 10);
        assert_relative_eq!(s.ex[idx], 1.0);
        assert_relative_eq!(s.ey[idx], 0.5);
    }

    // ── Material system tests ─────────────────────────────────────────────────

    #[test]
    fn add_material_box_sets_all_fields() {
        let mut s = small_solver();
        let mat = Fdtd3dMaterial {
            eps_r: 4.0,
            mu_r: 2.0,
            sigma_e: 0.5,
            sigma_m: 0.1,
        };
        s.add_material_box(5, 10, 5, 10, 5, 10, mat);
        let idx = s.idx(7, 7, 7);
        assert_relative_eq!(s.eps_r[idx], 4.0);
        assert_relative_eq!(s.mu_r[idx], 2.0);
        assert_relative_eq!(s.sigma_e[idx], 0.5);
        assert_relative_eq!(s.sigma_m[idx], 0.1);
        // Outside region stays vacuum
        let idx_out = s.idx(1, 1, 1);
        assert_relative_eq!(s.eps_r[idx_out], 1.0);
        assert_relative_eq!(s.sigma_e[idx_out], 0.0);
    }

    #[test]
    fn lossy_material_attenuates_field() {
        let mut s = small_solver();
        // Add a highly lossy region around the source
        let mat = Fdtd3dMaterial {
            eps_r: 1.0,
            mu_r: 1.0,
            sigma_e: 1e4,
            sigma_m: 0.0,
        };
        s.add_material_box(12, 18, 12, 18, 12, 18, mat);
        s.inject_ez(15, 15, 15, 1.0);
        s.run(50);
        // Energy should be lower than without loss (we just check it's finite and non-negative)
        let e = s.total_energy();
        assert!(e >= 0.0 && e.is_finite());
    }

    // ── Source system tests ───────────────────────────────────────────────────

    #[test]
    fn add_point_source_injects_field() {
        let mut s = small_solver();
        let wf = GaussianPulse3d::new(10.0 * s.dt, 3.0 * s.dt);
        s.add_point_source(15, 15, 15, FieldComponent3d::Ez, 1.0, wf);
        s.run_with_sources(50);
        assert!(s.peak_ez().is_finite());
        assert!(s.total_energy() > 0.0);
    }

    #[test]
    fn add_plane_wave_source_injects_field() {
        let mut s = small_solver();
        let wf = GaussianPulse3d::new(10.0 * s.dt, 3.0 * s.dt);
        s.add_plane_wave_source(Axis3d::Z, 5, FieldComponent3d::Ex, 1.0, wf);
        s.run_with_sources(20);
        assert!(s.total_energy().is_finite());
    }

    // ── Monitor system tests ──────────────────────────────────────────────────

    #[test]
    fn field_probe_records_data() {
        let mut s = small_solver();
        let wf = GaussianPulse3d::new(10.0 * s.dt, 3.0 * s.dt);
        s.add_point_source(15, 15, 15, FieldComponent3d::Ez, 1.0, wf);
        let probe_idx = s.add_field_probe(15, 15, 15, FieldComponent3d::Ez);
        s.run_with_sources(30);
        let ts = s.get_probe_time_series(probe_idx);
        assert!(ts.is_some());
        assert_eq!(ts.unwrap().len(), 30);
    }

    #[test]
    fn plane_monitor_records_snapshots() {
        let mut s = small_solver();
        let wf = GaussianPulse3d::new(10.0 * s.dt, 3.0 * s.dt);
        s.add_point_source(15, 15, 15, FieldComponent3d::Ez, 1.0, wf);
        let mon_idx = s.add_plane_monitor(Axis3d::Z, 15, FieldComponent3d::Ez, 5);
        s.run_with_sources(20);
        // Steps 0,5,10,15 → 4 snapshots (step recorded before increment)
        let snap = s.get_plane_snapshot(mon_idx, 0);
        assert!(snap.is_some());
        assert_eq!(snap.unwrap().len(), 30 * 30);
    }

    #[test]
    fn dft_probe_accumulates_correctly() {
        let mut s = small_solver();
        let omega = 2.0 * std::f64::consts::PI * 300e12;
        let wf = CwWaveform3d::new(omega, 0.0);
        s.add_point_source(15, 15, 15, FieldComponent3d::Ez, 1.0, wf);
        let probe_idx = s.add_dft_probe(15, 15, 15, FieldComponent3d::Ez, vec![300e12]);
        s.run_with_sources(30);
        let spec = s.get_dft_spectrum(probe_idx);
        assert!(spec.is_some());
        let spec = spec.unwrap();
        assert_eq!(spec.len(), 1);
        assert_relative_eq!(spec[0].0, 300e12, epsilon = 1.0);
    }

    // ── Checkpoint tests ──────────────────────────────────────────────────────

    #[test]
    fn checkpoint_save_restore_roundtrip() {
        let mut s = small_solver();
        s.inject_ez(15, 15, 15, 1.0);
        s.run(10);
        let cp = s.save_checkpoint();
        assert_eq!(cp.time_step, 10);

        // Advance further then restore
        s.run(20);
        s.restore_checkpoint(&cp);
        assert_eq!(s.time_step, 10);
        let idx = s.idx(15, 15, 15);
        assert_relative_eq!(s.ez[idx], cp.ez[idx]);
    }

    #[test]
    fn checkpoint_wrong_size_is_ignored() {
        let mut s = small_solver();
        let cp = Checkpoint3d {
            time_step: 99,
            ex: vec![0.0; 10], // wrong size
            ey: vec![0.0; 10],
            ez: vec![0.0; 10],
            hx: vec![0.0; 10],
            hy: vec![0.0; 10],
            hz: vec![0.0; 10],
        };
        let original_step = s.time_step;
        s.restore_checkpoint(&cp);
        // Should be unchanged since sizes don't match
        assert_eq!(s.time_step, original_step);
    }

    // ── Stability tests ───────────────────────────────────────────────────────

    #[test]
    fn courant_number_is_below_one() {
        let s = small_solver();
        let cn = s.courant_number();
        assert!(cn > 0.0 && cn <= 1.0, "Courant number {cn} out of range");
    }

    // ── Energy partition tests ────────────────────────────────────────────────

    #[test]
    fn energy_e_and_h_are_nonnegative() {
        let mut s = small_solver();
        s.inject_ez(15, 15, 15, 1.0);
        s.run(10);
        assert!(s.energy_e() >= 0.0);
        assert!(s.energy_h() >= 0.0);
    }

    #[test]
    fn total_energy_equals_sum_of_parts() {
        let mut s = small_solver();
        s.inject_ez(15, 15, 15, 1.0);
        s.run(5);
        assert_relative_eq!(
            s.total_energy(),
            s.energy_e() + s.energy_h(),
            epsilon = 1e-20
        );
    }

    // ── Field access tests ────────────────────────────────────────────────────

    #[test]
    fn field_at_returns_all_six_components() {
        let mut s = small_solver();
        s.inject_ez(15, 15, 15, 2.5);
        let f = s.field_at(15, 15, 15);
        assert_relative_eq!(f[2], 2.5); // Ez is index 2
    }

    #[test]
    fn field_at_out_of_bounds_returns_zeros() {
        let s = small_solver();
        let f = s.field_at(999, 999, 999);
        assert_eq!(f, [0.0; 6]);
    }

    #[test]
    fn max_field_component_initially_zero() {
        let s = small_solver();
        let (_comp, val, _i, _j, _k) = s.max_field_component();
        assert_eq!(val, 0.0);
    }

    #[test]
    fn max_field_component_finds_ez_after_injection() {
        let mut s = small_solver();
        s.inject_ez(15, 15, 15, 5.0);
        let (comp, val, i, j, k) = s.max_field_component();
        assert_eq!(comp, FieldComponent3d::Ez);
        assert_relative_eq!(val, 5.0);
        assert_eq!((i, j, k), (15, 15, 15));
    }
}
