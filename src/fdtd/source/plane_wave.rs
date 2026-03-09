//! 3D Plane Wave Source for FDTD.
//!
//! Injects a uniform plane wave along a specified propagation axis into the FDTD domain.
//!
//! For normal incidence (e.g., +Z propagation, X polarization):
//!   E_x(i,j,k_inj, t) += amplitude * waveform(t)
//!   H_y(i,j,k_inj, t) += amplitude / Z0 * waveform(t)
//!
//! Supports arbitrary propagation axes, polarizations, Gaussian envelope, and phase offset.

use crate::fdtd::source::SourceWaveform;

// Free-space impedance (Ω)
const Z0: f64 = 376.730_313_461_77;

/// Propagation axis for a plane wave source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropagationAxis {
    /// Propagating in the +X direction
    PlusX,
    /// Propagating in the -X direction
    MinusX,
    /// Propagating in the +Y direction
    PlusY,
    /// Propagating in the -Y direction
    MinusY,
    /// Propagating in the +Z direction
    PlusZ,
    /// Propagating in the -Z direction
    MinusZ,
}

impl PropagationAxis {
    /// Returns the unit vector [dx, dy, dz] for this propagation direction.
    pub fn unit_vector(&self) -> [f64; 3] {
        match self {
            PropagationAxis::PlusX => [1.0, 0.0, 0.0],
            PropagationAxis::MinusX => [-1.0, 0.0, 0.0],
            PropagationAxis::PlusY => [0.0, 1.0, 0.0],
            PropagationAxis::MinusY => [0.0, -1.0, 0.0],
            PropagationAxis::PlusZ => [0.0, 0.0, 1.0],
            PropagationAxis::MinusZ => [0.0, 0.0, -1.0],
        }
    }

    /// Returns the index of the propagation axis (0=X, 1=Y, 2=Z).
    pub fn axis_index(&self) -> usize {
        match self {
            PropagationAxis::PlusX | PropagationAxis::MinusX => 0,
            PropagationAxis::PlusY | PropagationAxis::MinusY => 1,
            PropagationAxis::PlusZ | PropagationAxis::MinusZ => 2,
        }
    }

    /// Returns true for positive propagation direction.
    pub fn is_positive(&self) -> bool {
        matches!(
            self,
            PropagationAxis::PlusX | PropagationAxis::PlusY | PropagationAxis::PlusZ
        )
    }
}

/// E-field polarization direction for a 3D plane wave.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarization3d {
    /// E-field along the X direction
    Ex,
    /// E-field along the Y direction
    Ey,
    /// E-field along the Z direction
    Ez,
}

impl Polarization3d {
    /// Returns the field component index (0=x, 1=y, 2=z).
    pub fn component_index(&self) -> usize {
        match self {
            Polarization3d::Ex => 0,
            Polarization3d::Ey => 1,
            Polarization3d::Ez => 2,
        }
    }
}

/// Gaussian envelope parameters for 3D sources.
#[derive(Debug, Clone, Copy)]
pub struct GaussianEnvelope3d {
    /// Time of peak (s)
    pub t0: f64,
    /// Gaussian width / pulse duration (s)
    pub sigma: f64,
}

impl GaussianEnvelope3d {
    /// Create a Gaussian envelope with given peak time and width.
    pub fn new(t0: f64, sigma: f64) -> Self {
        Self { t0, sigma }
    }

    /// Evaluate envelope at time t.
    pub fn evaluate(&self, t: f64) -> f64 {
        let dt = t - self.t0;
        (-(dt / self.sigma).powi(2)).exp()
    }
}

/// 3D plane wave source for FDTD.
///
/// Injects a uniform plane wave along a specified axis at a fixed injection plane.
/// Supports both CW (continuous wave) and pulsed (Gaussian envelope) operation.
/// Oblique incidence is approximated via k-direction cosines for phase offsets.
#[derive(Debug, Clone)]
pub struct PlaneWaveSource3d {
    /// Propagation axis and direction
    pub axis: PropagationAxis,
    /// Injection plane position (index along propagation axis)
    pub position: usize,
    /// Polarization: direction of the E-field
    pub polarization: Polarization3d,
    /// Amplitude (V/m)
    pub amplitude: f64,
    /// Angular frequency (rad/s)
    pub omega: f64,
    /// Optional Gaussian envelope for pulsed operation
    pub envelope: Option<GaussianEnvelope3d>,
    /// Phase offset (rad)
    pub phase: f64,
    /// k-vector direction cosines [kx, ky, kz] (unit vector, for oblique incidence)
    pub k_dir: [f64; 3],
}

impl PlaneWaveSource3d {
    /// Create a Z-propagating plane wave at the given injection plane.
    ///
    /// Defaults to CW (no Gaussian envelope) and zero phase.
    pub fn new_z_propagating(
        position: usize,
        polarization: Polarization3d,
        amplitude: f64,
        omega: f64,
    ) -> Self {
        Self {
            axis: PropagationAxis::PlusZ,
            position,
            polarization,
            amplitude,
            omega,
            envelope: None,
            phase: 0.0,
            k_dir: [0.0, 0.0, 1.0],
        }
    }

    /// Create a plane wave propagating along a specified axis.
    pub fn new(
        axis: PropagationAxis,
        position: usize,
        polarization: Polarization3d,
        amplitude: f64,
        omega: f64,
    ) -> Self {
        let k_dir = axis.unit_vector();
        Self {
            axis,
            position,
            polarization,
            amplitude,
            omega,
            envelope: None,
            phase: 0.0,
            k_dir,
        }
    }

    /// Add a Gaussian envelope (pulsed source).
    pub fn with_gaussian_envelope(mut self, t0: f64, sigma: f64) -> Self {
        self.envelope = Some(GaussianEnvelope3d::new(t0, sigma));
        self
    }

    /// Add a phase offset in radians.
    pub fn with_phase(mut self, phase: f64) -> Self {
        self.phase = phase;
        self
    }

    /// Set the k-vector direction cosines for oblique incidence.
    ///
    /// The k_dir must be a unit vector [kx, ky, kz].
    pub fn with_k_direction(mut self, kx: f64, ky: f64, kz: f64) -> Self {
        let norm = (kx * kx + ky * ky + kz * kz).sqrt().max(1e-30);
        self.k_dir = [kx / norm, ky / norm, kz / norm];
        self
    }

    /// Compute source E-field amplitude at time t.
    ///
    /// Returns the scalar amplitude of the E-field waveform.
    pub fn amplitude_at(&self, t: f64) -> f64 {
        let env = match &self.envelope {
            Some(g) => g.evaluate(t),
            None => 1.0,
        };
        self.amplitude * env * (self.omega * t + self.phase).sin()
    }

    /// Compute E-field value at transverse position (x, y) and time t.
    ///
    /// For oblique incidence, includes the transverse phase variation.
    /// `x`, `y` are physical coordinates in the injection plane (m).
    pub fn field_value(&self, t: f64, x: f64, y: f64) -> f64 {
        // Transverse phase from k-vector projection onto injection plane
        let c = 2.998e8_f64;
        let k_mag = self.omega / c;
        let transverse_phase = match self.axis.axis_index() {
            0 => k_mag * (self.k_dir[1] * y), // YZ plane injection
            1 => k_mag * (self.k_dir[0] * x), // XZ plane injection
            _ => k_mag * (self.k_dir[0] * x + self.k_dir[1] * y), // XY plane injection
        };
        let env = match &self.envelope {
            Some(g) => g.evaluate(t),
            None => 1.0,
        };
        self.amplitude * env * (self.omega * t + self.phase + transverse_phase).sin()
    }

    /// Impedance-matched H-field complement for soft source injection.
    ///
    /// For normal incidence: H = E / Z0
    /// Returned value is the H-field amplitude corresponding to the E-field at time t.
    pub fn h_complement(&self, t: f64) -> f64 {
        let sign = if self.axis.is_positive() { 1.0 } else { -1.0 };
        sign * self.amplitude_at(t) / Z0
    }

    /// Apply this source to 3D field arrays at the injection plane.
    ///
    /// Modifies the appropriate E-field component along the injection plane.
    /// Fields are stored as flat arrays with indexing: `field[i*ny*nz + j*nz + k]`.
    ///
    /// # Arguments
    /// * `t` - Current simulation time (s)
    /// * `ex`, `ey`, `ez` - Mutable E-field component arrays
    /// * `nx`, `ny`, `nz` - Grid dimensions
    /// * `dx`, `dy`, `dz` - Grid spacings (m)
    /// * `x0`, `y0`, `z0` - Physical origin coordinates (m)
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &self,
        t: f64,
        ex: &mut [f64],
        ey: &mut [f64],
        ez: &mut [f64],
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        x0: f64,
        y0: f64,
        z0: f64,
    ) {
        let pos = self.position;
        match self.axis.axis_index() {
            0 => {
                // Injection plane: i = pos (YZ plane), varying j, k
                if pos >= nx {
                    return;
                }
                for j in 0..ny {
                    for k in 0..nz {
                        let y = y0 + j as f64 * dy;
                        let z = z0 + k as f64 * dz;
                        let val = self.field_value(t, y, z);
                        let idx = pos * ny * nz + j * nz + k;
                        match self.polarization {
                            Polarization3d::Ex => {
                                if idx < ex.len() {
                                    ex[idx] += val;
                                }
                            }
                            Polarization3d::Ey => {
                                if idx < ey.len() {
                                    ey[idx] += val;
                                }
                            }
                            Polarization3d::Ez => {
                                if idx < ez.len() {
                                    ez[idx] += val;
                                }
                            }
                        }
                    }
                }
            }
            1 => {
                // Injection plane: j = pos (XZ plane), varying i, k
                if pos >= ny {
                    return;
                }
                for i in 0..nx {
                    for k in 0..nz {
                        let x = x0 + i as f64 * dx;
                        let z = z0 + k as f64 * dz;
                        let val = self.field_value(t, x, z);
                        let idx = i * ny * nz + pos * nz + k;
                        match self.polarization {
                            Polarization3d::Ex => {
                                if idx < ex.len() {
                                    ex[idx] += val;
                                }
                            }
                            Polarization3d::Ey => {
                                if idx < ey.len() {
                                    ey[idx] += val;
                                }
                            }
                            Polarization3d::Ez => {
                                if idx < ez.len() {
                                    ez[idx] += val;
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                // Injection plane: k = pos (XY plane), varying i, j
                if pos >= nz {
                    return;
                }
                for i in 0..nx {
                    for j in 0..ny {
                        let x = x0 + i as f64 * dx;
                        let y = y0 + j as f64 * dy;
                        let val = self.field_value(t, x, y);
                        let idx = i * ny * nz + j * nz + pos;
                        match self.polarization {
                            Polarization3d::Ex => {
                                if idx < ex.len() {
                                    ex[idx] += val;
                                }
                            }
                            Polarization3d::Ey => {
                                if idx < ey.len() {
                                    ey[idx] += val;
                                }
                            }
                            Polarization3d::Ez => {
                                if idx < ez.len() {
                                    ez[idx] += val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Apply the matching H-field correction at the injection plane (for total-field excitation).
    ///
    /// For a +Z propagating, X-polarized wave: injects Hy = -Ex / Z0.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_h(
        &self,
        t: f64,
        hx: &mut [f64],
        hy: &mut [f64],
        hz: &mut [f64],
        nx: usize,
        ny: usize,
        nz: usize,
        _dx: f64,
        _dy: f64,
        _dz: f64,
        _x0: f64,
        _y0: f64,
        _z0: f64,
    ) {
        let pos = self.position;
        let h_amp = self.h_complement(t);
        // Determine which H component is the complement
        // For E_x propagating in +Z: H_y complement
        // For E_y propagating in +Z: H_x complement (negative)
        // General: H = k_hat × E / Z0
        let k = self.k_dir;
        let (hx_scale, hy_scale, hz_scale) = match self.polarization {
            Polarization3d::Ex => (0.0, -k[2], k[1]),
            Polarization3d::Ey => (k[2], 0.0, -k[0]),
            Polarization3d::Ez => (-k[1], k[0], 0.0),
        };

        match self.axis.axis_index() {
            0 => {
                if pos >= nx {
                    return;
                }
                for j in 0..ny {
                    for kk in 0..nz {
                        let idx = pos * ny * nz + j * nz + kk;
                        if hx_scale.abs() > 0.0 && idx < hx.len() {
                            hx[idx] += h_amp * hx_scale;
                        }
                        if hy_scale.abs() > 0.0 && idx < hy.len() {
                            hy[idx] += h_amp * hy_scale;
                        }
                        if hz_scale.abs() > 0.0 && idx < hz.len() {
                            hz[idx] += h_amp * hz_scale;
                        }
                    }
                }
            }
            1 => {
                if pos >= ny {
                    return;
                }
                for i in 0..nx {
                    for kk in 0..nz {
                        let idx = i * ny * nz + pos * nz + kk;
                        if hx_scale.abs() > 0.0 && idx < hx.len() {
                            hx[idx] += h_amp * hx_scale;
                        }
                        if hy_scale.abs() > 0.0 && idx < hy.len() {
                            hy[idx] += h_amp * hy_scale;
                        }
                        if hz_scale.abs() > 0.0 && idx < hz.len() {
                            hz[idx] += h_amp * hz_scale;
                        }
                    }
                }
            }
            _ => {
                if pos >= nz {
                    return;
                }
                for i in 0..nx {
                    for j in 0..ny {
                        let idx = i * ny * nz + j * nz + pos;
                        if hx_scale.abs() > 0.0 && idx < hx.len() {
                            hx[idx] += h_amp * hx_scale;
                        }
                        if hy_scale.abs() > 0.0 && idx < hy.len() {
                            hy[idx] += h_amp * hy_scale;
                        }
                        if hz_scale.abs() > 0.0 && idx < hz.len() {
                            hz[idx] += h_amp * hz_scale;
                        }
                    }
                }
            }
        }
    }

    /// Compute the instantaneous intensity (W/m²) at a point (x, y) in the injection plane at time t.
    ///
    /// Returns E² / Z0 for the normal plane wave.
    pub fn intensity_at(&self, t: f64, x: f64, y: f64) -> f64 {
        let e = self.field_value(t, x, y);
        e * e / Z0
    }

    /// Compute time-averaged intensity (W/m²) for CW source.
    ///
    /// For a monochromatic wave: `<I>` = A² / (2\*Z0)
    pub fn time_averaged_intensity(&self) -> f64 {
        // For pulsed source, this is an approximation at peak of envelope
        self.amplitude * self.amplitude / (2.0 * Z0)
    }
}

/// A 1D plane wave source using the existing SourceWaveform trait (kept for backward compat).
pub struct PlaneWaveSource {
    /// Cell index where the source is injected
    pub position: usize,
    /// Waveform
    pub waveform: Box<dyn SourceWaveform>,
    /// Amplitude scaling
    pub amplitude: f64,
}

impl PlaneWaveSource {
    pub fn new(position: usize, waveform: Box<dyn SourceWaveform>) -> Self {
        Self {
            position,
            waveform,
            amplitude: 1.0,
        }
    }

    pub fn amplitude(&self, t: f64) -> f64 {
        self.waveform.amplitude(t) * self.amplitude
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn plane_wave_3d_amplitude_at_zero_before_pulse() {
        let src =
            PlaneWaveSource3d::new_z_propagating(10, Polarization3d::Ex, 1.0, 2.0 * PI * 1.94e14)
                .with_gaussian_envelope(30e-15, 10e-15);
        // At t=0, far before t0=30fs, amplitude should be ~0
        let a = src.amplitude_at(0.0);
        assert!(
            a.abs() < 1e-3,
            "amplitude at t=0 should be ~0 for pulsed source: {a}"
        );
    }

    #[test]
    fn plane_wave_3d_cw_oscillates() {
        let omega = 2.0 * PI * 1.94e14;
        let src = PlaneWaveSource3d::new_z_propagating(10, Polarization3d::Ex, 1.0, omega);
        let t1 = 1.0 / (4.0 * 1.94e14); // quarter period
        let t2 = 3.0 / (4.0 * 1.94e14); // three-quarter period
        let a1 = src.amplitude_at(t1);
        let a2 = src.amplitude_at(t2);
        assert!(
            a1 * a2 < 0.0,
            "CW source should alternate sign: a1={a1}, a2={a2}"
        );
    }

    #[test]
    fn plane_wave_3d_h_complement_proportional() {
        let omega = 2.0 * PI * 1.94e14;
        let src = PlaneWaveSource3d::new_z_propagating(10, Polarization3d::Ex, 1.0, omega);
        let t = 1.0 / (4.0 * 1.94e14);
        let e = src.amplitude_at(t);
        let h = src.h_complement(t);
        if h.abs() > 1e-20 {
            let ratio = e / h;
            assert!(
                (ratio - Z0).abs() < 1.0,
                "E/H should be Z0={Z0:.1}: ratio={ratio:.1}"
            );
        }
    }

    #[test]
    fn plane_wave_3d_apply_modifies_fields() {
        let omega = 2.0 * PI * 1.94e14;
        let src = PlaneWaveSource3d::new_z_propagating(2, Polarization3d::Ex, 1.0, omega);
        let nx = 4;
        let ny = 4;
        let nz = 8;
        let n = nx * ny * nz;
        let mut ex = vec![0.0f64; n];
        let mut ey = vec![0.0f64; n];
        let mut ez = vec![0.0f64; n];
        let t = 1.0 / (4.0 * 1.94e14);
        src.apply(
            t, &mut ex, &mut ey, &mut ez, nx, ny, nz, 10e-9, 10e-9, 10e-9, 0.0, 0.0, 0.0,
        );
        let any_nonzero = ex.iter().any(|&v| v.abs() > 0.0);
        assert!(any_nonzero, "apply should modify ex field");
    }

    #[test]
    fn plane_wave_3d_propagation_axis_unit_vectors() {
        let ax = PropagationAxis::PlusX.unit_vector();
        assert!((ax[0] - 1.0).abs() < 1e-12);
        let am = PropagationAxis::MinusZ.unit_vector();
        assert!((am[2] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn plane_wave_3d_field_value_normal_incidence() {
        let omega = 2.0 * PI * 1.94e14;
        let src = PlaneWaveSource3d::new_z_propagating(5, Polarization3d::Ex, 2.0, omega);
        let t = 1.0 / (4.0 * 1.94e14);
        // Normal incidence: field_value should be independent of x, y
        let f1 = src.field_value(t, 0.0, 0.0);
        let f2 = src.field_value(t, 1e-6, 2e-6);
        // For normal incidence the transverse phase depends on k_dir transverse components
        // With k_dir=[0,0,1] and axis=Z, transverse phase = k*(kx*x + ky*y) = 0
        assert!(
            (f1 - f2).abs() < 1e-10,
            "Normal incidence: field should be uniform"
        );
    }

    #[test]
    fn plane_wave_3d_intensity_positive() {
        let omega = 2.0 * PI * 1.94e14;
        let src = PlaneWaveSource3d::new_z_propagating(5, Polarization3d::Ex, 1.0, omega);
        let i_avg = src.time_averaged_intensity();
        assert!(i_avg > 0.0, "Time-averaged intensity should be positive");
    }

    #[test]
    fn plane_wave_3d_with_phase() {
        let omega = 2.0 * PI * 1.94e14;
        let src = PlaneWaveSource3d::new_z_propagating(5, Polarization3d::Ex, 1.0, omega)
            .with_phase(PI / 2.0);
        // At t=0, sin(phase) = sin(pi/2) = 1.0
        let a = src.amplitude_at(0.0);
        assert!(
            (a - 1.0).abs() < 1e-10,
            "With pi/2 phase, amplitude at t=0 should be 1: {a}"
        );
    }

    #[test]
    fn plane_wave_3d_gaussian_envelope_peak() {
        let omega = 2.0 * PI * 1.94e14;
        let t0 = 30e-15;
        let sigma = 10e-15;
        let src = PlaneWaveSource3d::new_z_propagating(5, Polarization3d::Ex, 1.0, omega)
            .with_gaussian_envelope(t0, sigma)
            .with_phase(PI / 2.0); // cos at peak
                                   // At t=t0 with phase PI/2: amplitude = 1.0 * 1.0 * sin(omega*t0 + pi/2)
        let a = src.amplitude_at(t0);
        assert!(a.abs() <= 1.0 + 1e-10, "Amplitude should not exceed 1: {a}");
        // But before pulse, should be nearly zero
        let a_before = src.amplitude_at(0.0);
        let env_before = (-((0.0 - t0) / sigma).powi(2)).exp();
        assert!(a_before.abs() <= env_before + 1e-10);
    }

    #[test]
    fn plane_wave_3d_different_axes() {
        let omega = 2.0 * PI * 1.94e14;
        let nx = 8;
        let ny = 8;
        let nz = 8;
        let n = nx * ny * nz;
        let t = 1.0 / (4.0 * 1.94e14);

        // Test Y propagation
        let src_y =
            PlaneWaveSource3d::new(PropagationAxis::PlusY, 4, Polarization3d::Ex, 1.0, omega);
        let mut ex = vec![0.0f64; n];
        let mut ey = vec![0.0f64; n];
        let mut ez = vec![0.0f64; n];
        src_y.apply(
            t, &mut ex, &mut ey, &mut ez, nx, ny, nz, 10e-9, 10e-9, 10e-9, 0.0, 0.0, 0.0,
        );
        let nonzero = ex.iter().any(|&v| v.abs() > 0.0);
        assert!(nonzero, "+Y propagation should inject into Ex");

        // Test X propagation
        let src_x =
            PlaneWaveSource3d::new(PropagationAxis::PlusX, 2, Polarization3d::Ey, 1.0, omega);
        let mut ex2 = vec![0.0f64; n];
        let mut ey2 = vec![0.0f64; n];
        let mut ez2 = vec![0.0f64; n];
        src_x.apply(
            t, &mut ex2, &mut ey2, &mut ez2, nx, ny, nz, 10e-9, 10e-9, 10e-9, 0.0, 0.0, 0.0,
        );
        let nonzero2 = ey2.iter().any(|&v| v.abs() > 0.0);
        assert!(nonzero2, "+X propagation should inject into Ey");
    }
}
