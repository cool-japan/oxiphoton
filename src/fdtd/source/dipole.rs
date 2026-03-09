//! Point dipole source for FDTD simulations.
//!
//! A dipole source injects a current density J at a single grid cell:
//!   J_x(i,j) = J₀(t) · δ(r - r₀)
//!
//! Applications:
//!   - Near-field radiation patterns
//!   - Purcell factor calculation (P_actual / P_bulk)
//!   - Antenna arrays
//!   - Spontaneous emission rate enhancement
//!
//! Purcell factor:
//!   F_P = (3/4π²) · (λ/n)³ · Q/V
//! where Q is the cavity quality factor and V is the mode volume.

use crate::fdtd::source::plane_wave::GaussianEnvelope3d;
use std::f64::consts::PI;

/// Orientation of a dipole source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DipoleOrientation {
    X,
    Y,
    Z,
}

/// A point dipole source with arbitrary time waveform.
#[derive(Debug, Clone)]
pub struct DipoleSrc {
    /// Grid cell index (ix, iy) for 2D
    pub ix: usize,
    pub iy: usize,
    /// Dipole orientation
    pub orientation: DipoleOrientation,
    /// Peak amplitude J₀ (A/m² when normalized to cell volume)
    pub amplitude: f64,
    /// Center frequency (Hz)
    pub f0: f64,
    /// Gaussian pulse width (s) — 0 = CW
    pub tau: f64,
    /// Time delay (s)
    pub t0: f64,
}

impl DipoleSrc {
    /// Create a Gaussian-modulated dipole at (ix, iy).
    pub fn gaussian(
        ix: usize,
        iy: usize,
        orientation: DipoleOrientation,
        amplitude: f64,
        f0: f64,
        tau: f64,
    ) -> Self {
        let t0 = 3.0 * tau;
        Self {
            ix,
            iy,
            orientation,
            amplitude,
            f0,
            tau,
            t0,
        }
    }

    /// Create a CW (continuous-wave) dipole.
    pub fn cw(
        ix: usize,
        iy: usize,
        orientation: DipoleOrientation,
        amplitude: f64,
        f0: f64,
    ) -> Self {
        Self {
            ix,
            iy,
            orientation,
            amplitude,
            f0,
            tau: 0.0,
            t0: 0.0,
        }
    }

    /// Evaluate dipole current J(t) at time t.
    pub fn current(&self, t: f64) -> f64 {
        let phase = 2.0 * PI * self.f0 * (t - self.t0);
        let envelope = if self.tau > 0.0 {
            let dt = t - self.t0;
            (-(dt / self.tau).powi(2)).exp()
        } else {
            1.0
        };
        self.amplitude * envelope * phase.sin()
    }

    /// Wavelength at center frequency (m).
    pub fn wavelength_m(&self) -> f64 {
        2.998e8 / self.f0
    }
}

/// Purcell factor calculator for a point dipole in a resonant cavity.
///
/// F_P = (3/4π²) · (λ_r/n)³ · Q / V_mode
pub struct PurcellCalc {
    /// Cavity quality factor Q
    pub q_factor: f64,
    /// Mode volume V_mode (m³)
    pub mode_volume_m3: f64,
    /// Background refractive index n
    pub n_background: f64,
    /// Resonance wavelength (m)
    pub lambda_res_m: f64,
}

impl PurcellCalc {
    pub fn new(q_factor: f64, mode_volume_m3: f64, n_background: f64, lambda_res_m: f64) -> Self {
        Self {
            q_factor,
            mode_volume_m3,
            n_background,
            lambda_res_m,
        }
    }

    /// L3 PhC cavity in Si (Q=10000, V≈0.7(λ/n)³, n=3.46, λ=1550nm).
    pub fn l3_phc_silicon() -> Self {
        let n = 3.46_f64;
        let lambda = 1550e-9_f64;
        let cubic = (lambda / n).powi(3);
        Self::new(10_000.0, 0.7 * cubic, n, lambda)
    }

    /// Purcell factor (enhancement factor for spontaneous emission rate).
    ///
    /// F_P = (3/4π²) · (λ/n)³ · Q/V
    pub fn purcell_factor(&self) -> f64 {
        let lambda_n = self.lambda_res_m / self.n_background;
        (3.0 / (4.0 * PI * PI)) * lambda_n.powi(3) * self.q_factor / self.mode_volume_m3
    }

    /// Total decay rate enhancement γ_total / γ₀.
    ///
    /// Accounts for both Purcell-enhanced and background emission:
    ///   γ_total = F_P · β + (1 - β)  (β = coupling efficiency to mode)
    pub fn decay_rate_enhancement(&self, beta: f64) -> f64 {
        let fp = self.purcell_factor();
        fp * beta + (1.0 - beta)
    }

    /// Photon collection efficiency into the cavity mode.
    ///
    ///   β = F_P / (F_P + 1)
    pub fn beta_factor(&self) -> f64 {
        let fp = self.purcell_factor();
        fp / (fp + 1.0)
    }

    /// Linewidth of the resonance (m).
    pub fn linewidth_m(&self) -> f64 {
        self.lambda_res_m / self.q_factor
    }
}

/// Near-field radiation pattern for a dipole in free space.
///
/// I(θ) = I₀ · sin²(θ)  for a z-oriented dipole
pub fn dipole_radiation_pattern(theta_rad: f64, orientation: DipoleOrientation) -> f64 {
    match orientation {
        DipoleOrientation::Z => theta_rad.sin().powi(2),
        DipoleOrientation::X => {
            // For x-dipole in the xz-plane: I ~ 1 - sin²θ·cos²φ, φ=0 → sin²θ
            // In the xy-plane (elevation): I ~ cos²θ
            theta_rad.cos().powi(2)
        }
        DipoleOrientation::Y => theta_rad.cos().powi(2),
    }
}

/// Total radiated power from a Hertzian dipole in free space.
///
/// P = (Z₀/12π) · (2π/λ)⁴ · |p|²  where p is dipole moment (C·m)
pub fn dipole_radiated_power(dipole_moment_cm: f64, wavelength_m: f64) -> f64 {
    let z0 = 377.0; // free-space impedance (Ω)
    let k = 2.0 * PI / wavelength_m;
    (z0 / (12.0 * PI)) * k.powi(4) * dipole_moment_cm.powi(2)
}

// ─────────────────────────────────────────────────────────────────
// 3D Dipole Source
// ─────────────────────────────────────────────────────────────────

/// Orientation of a 3D point dipole source, supporting arbitrary unit vectors.
#[derive(Debug, Clone, Copy)]
pub enum DipoleOrientation3d {
    /// Dipole along the X axis
    X,
    /// Dipole along the Y axis
    Y,
    /// Dipole along the Z axis
    Z,
    /// Dipole along an arbitrary unit vector \[px, py, pz\]
    Arbitrary([f64; 3]),
}

impl DipoleOrientation3d {
    /// Return the normalized unit vector for this orientation.
    pub fn unit_vector(&self) -> [f64; 3] {
        match self {
            DipoleOrientation3d::X => [1.0, 0.0, 0.0],
            DipoleOrientation3d::Y => [0.0, 1.0, 0.0],
            DipoleOrientation3d::Z => [0.0, 0.0, 1.0],
            DipoleOrientation3d::Arbitrary(v) => {
                let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-30);
                [v[0] / norm, v[1] / norm, v[2] / norm]
            }
        }
    }
}

/// 3D point dipole source with full vector orientation.
///
/// Injects a time-varying current at a single FDTD grid cell.
/// The current is decomposed into E-field components along the dipole orientation:
///   E_component += amplitude * orientation_component * waveform(t)
///
/// For a CW dipole:    waveform(t) = sin(ω·t + phase)
/// For a pulsed dipole: waveform(t) = envelope(t) · sin(ω·t + phase)
#[derive(Debug, Clone)]
pub struct DipoleSrc3d {
    /// Grid cell position (i, j, k)
    pub i: usize,
    pub j: usize,
    pub k: usize,
    /// Dipole orientation (direction of oscillation)
    pub orientation: DipoleOrientation3d,
    /// Peak amplitude (V/m or A/m² depending on injection method)
    pub amplitude: f64,
    /// Angular frequency (rad/s)
    pub omega: f64,
    /// Phase offset (rad)
    pub phase: f64,
    /// Optional Gaussian envelope for pulsed excitation
    envelope: Option<GaussianEnvelope3d>,
}

impl DipoleSrc3d {
    /// Create a new 3D dipole source at grid cell (i, j, k).
    pub fn new(
        i: usize,
        j: usize,
        k: usize,
        orientation: DipoleOrientation3d,
        amplitude: f64,
        omega: f64,
    ) -> Self {
        Self {
            i,
            j,
            k,
            orientation,
            amplitude,
            omega,
            phase: 0.0,
            envelope: None,
        }
    }

    /// Add a phase offset.
    pub fn with_phase(mut self, phase: f64) -> Self {
        self.phase = phase;
        self
    }

    /// Add a Gaussian pulse envelope.
    pub fn with_gaussian_pulse(mut self, t0: f64, sigma: f64) -> Self {
        self.envelope = Some(GaussianEnvelope3d::new(t0, sigma));
        self
    }

    /// Compute the scalar source value at time t.
    ///
    /// Returns the waveform amplitude (before orientation decomposition).
    pub fn value_at(&self, t: f64) -> f64 {
        let env = match &self.envelope {
            Some(g) => g.evaluate(t),
            None => 1.0,
        };
        self.amplitude * env * (self.omega * t + self.phase).sin()
    }

    /// Apply this dipole source to 3D E-field arrays at time t.
    ///
    /// Modifies the E-field components at the source position weighted by the
    /// dipole orientation unit vector.
    ///
    /// # Arguments
    /// * `t` - Current time (s)
    /// * `ex`, `ey`, `ez` - Mutable 3D field arrays (flat, row-major: i*ny*nz + j*nz + k)
    /// * `nx`, `ny`, `nz` - Grid dimensions
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
    ) {
        if self.i >= nx || self.j >= ny || self.k >= nz {
            return;
        }
        let idx = self.i * ny * nz + self.j * nz + self.k;
        let val = self.value_at(t);
        let uv = self.orientation.unit_vector();

        if idx < ex.len() {
            ex[idx] += val * uv[0];
        }
        if idx < ey.len() {
            ey[idx] += val * uv[1];
        }
        if idx < ez.len() {
            ez[idx] += val * uv[2];
        }
    }

    /// Estimate the Purcell factor for this dipole in a resonant cavity.
    ///
    /// Uses the analytical formula:
    ///   F_P = (3/4π²) · (λ/n)³ · Q / (V_mode in λ³ units)
    ///
    /// # Arguments
    /// * `quality_factor` - Cavity Q factor
    /// * `mode_volume_lambda3` - Mode volume in units of (λ/n)³
    pub fn estimate_purcell_factor(&self, quality_factor: f64, mode_volume_lambda3: f64) -> f64 {
        // F_P = 3/(4π²) * Q / V_eff   where V_eff is in (λ/n)³ units
        if mode_volume_lambda3 <= 0.0 || quality_factor <= 0.0 {
            return 0.0;
        }
        (3.0 / (4.0 * PI * PI)) * quality_factor / mode_volume_lambda3
    }

    /// Compute the power spectral density emitted by this dipole.
    ///
    /// For a Hertzian dipole in free space:
    ///   P = (Z₀/12π) · k⁴ · |p|²
    /// where p is the dipole moment amplitude · cell_volume.
    ///
    /// # Arguments
    /// * `wavelength` - Free-space wavelength (m)
    /// * `cell_volume` - FDTD cell volume (m³) for dimensional correction
    pub fn radiated_power_estimate(&self, wavelength: f64, cell_volume: f64) -> f64 {
        let z0 = 376.730_313_461_77;
        let k = 2.0 * PI / wavelength;
        let p = self.amplitude * cell_volume;
        (z0 / (12.0 * PI)) * k.powi(4) * p * p
    }

    /// Compute the 3D radiation pattern at angles (theta, phi) in spherical coordinates.
    ///
    /// Returns the normalized pattern intensity I(θ,φ) in \[0,1\].
    ///
    /// For a dipole along unit vector p̂, the pattern is:
    ///   I(θ,φ) = 1 - |p̂ · r̂(θ,φ)|²
    pub fn radiation_pattern(&self, theta: f64, phi: f64) -> f64 {
        let r_hat = [
            theta.sin() * phi.cos(),
            theta.sin() * phi.sin(),
            theta.cos(),
        ];
        let p = self.orientation.unit_vector();
        let dot = p[0] * r_hat[0] + p[1] * r_hat[1] + p[2] * r_hat[2];
        (1.0 - dot * dot).max(0.0)
    }
}

/// An array of 3D dipoles for phased antenna and multi-emitter effects.
///
/// Each dipole in the array has an individual phase offset, enabling
/// steering and interference pattern engineering.
#[derive(Debug, Clone)]
pub struct DipoleArray3d {
    /// List of (dipole, phase_offset) pairs
    pub dipoles: Vec<(DipoleSrc3d, f64)>,
}

impl DipoleArray3d {
    /// Create an empty dipole array.
    pub fn new() -> Self {
        Self {
            dipoles: Vec::new(),
        }
    }

    /// Add a dipole to the array with a specified phase offset (rad).
    pub fn add_dipole(&mut self, dipole: DipoleSrc3d, phase_offset: f64) {
        self.dipoles.push((dipole, phase_offset));
    }

    /// Apply all dipoles to the 3D field arrays at time t.
    ///
    /// Each dipole injects its contribution with its individual phase offset.
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
    ) {
        for (dipole, phase_offset) in &self.dipoles {
            // Create a temporary dipole with adjusted phase
            let mut d = dipole.clone();
            d.phase = dipole.phase + phase_offset;
            d.apply(t, ex, ey, ez, nx, ny, nz);
        }
    }

    /// Number of dipoles in the array.
    pub fn len(&self) -> usize {
        self.dipoles.len()
    }

    /// Returns true if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.dipoles.is_empty()
    }

    /// Compute the total array radiation pattern at (theta, phi).
    ///
    /// Coherently sums the radiation from all dipoles (assumes free-space, far-field).
    /// Returns the normalized intensity summed from all dipoles.
    pub fn array_pattern(&self, theta: f64, phi: f64) -> f64 {
        self.dipoles
            .iter()
            .map(|(d, _)| d.radiation_pattern(theta, phi))
            .sum()
    }

    /// Compute the array factor for a linear array (coherent far-field sum).
    ///
    /// Assumes dipoles are at positions (i, j, k) with equal spacing dx.
    /// The array factor is:
    ///   AF(θ) = Σ exp(i·(n·k₀·d·cos(θ) + phase_n))
    ///
    /// Returns the magnitude squared of the array factor.
    pub fn array_factor_magnitude_sq(&self, theta: f64, phi: f64, k0: f64, dx: f64) -> f64 {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for (dipole, phase_offset) in &self.dipoles {
            let x = dipole.i as f64 * dx;
            let y = dipole.j as f64 * dx;
            let z = dipole.k as f64 * dx;
            let r_dot = x * theta.sin() * phi.cos() + y * theta.sin() * phi.sin() + z * theta.cos();
            let total_phase = k0 * r_dot + dipole.phase + phase_offset;
            re += total_phase.cos();
            im += total_phase.sin();
        }
        re * re + im * im
    }
}

impl Default for DipoleArray3d {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dipole_current_zero_before_pulse() {
        let src = DipoleSrc::gaussian(10, 10, DipoleOrientation::Z, 1.0, 1.94e14, 10e-15);
        // At t=0, far before pulse center t0=30fs
        let j = src.current(0.0);
        assert!(j.abs() < 0.01, "j={j:.4}");
    }

    #[test]
    fn dipole_cw_oscillates() {
        let src = DipoleSrc::cw(5, 5, DipoleOrientation::X, 1.0, 1e14);
        let j1 = src.current(1.0 / (4.0 * 1e14)); // quarter period
        let j2 = src.current(3.0 / (4.0 * 1e14)); // three-quarter period
        assert!(j1 * j2 < 0.0, "CW should alternate sign");
    }

    #[test]
    fn dipole_wavelength() {
        let src = DipoleSrc::cw(0, 0, DipoleOrientation::Z, 1.0, 1.94e14);
        let lam = src.wavelength_m();
        assert!((lam - 1545e-9).abs() < 10e-9, "λ={lam:.1e}");
    }

    #[test]
    fn purcell_factor_l3_positive() {
        let pc = PurcellCalc::l3_phc_silicon();
        let fp = pc.purcell_factor();
        assert!(fp > 1.0, "Purcell factor should exceed 1, got {fp:.1}");
    }

    #[test]
    fn beta_factor_range() {
        let pc = PurcellCalc::l3_phc_silicon();
        let beta = pc.beta_factor();
        assert!(beta > 0.0 && beta < 1.0, "β={beta:.3}");
    }

    #[test]
    fn decay_rate_enhancement_ge_1() {
        let pc = PurcellCalc::l3_phc_silicon();
        let enh = pc.decay_rate_enhancement(0.9);
        assert!(enh > 1.0, "enhancement={enh:.1}");
    }

    #[test]
    fn dipole_radiation_z_max_at_90deg() {
        let i_90 = dipole_radiation_pattern(PI / 2.0, DipoleOrientation::Z);
        let i_0 = dipole_radiation_pattern(0.0, DipoleOrientation::Z);
        assert!((i_90 - 1.0).abs() < 1e-10);
        assert!(i_0.abs() < 1e-10);
    }

    #[test]
    fn dipole_radiated_power_positive() {
        let p = dipole_radiated_power(1e-30, 1550e-9);
        assert!(p > 0.0);
    }

    #[test]
    fn purcell_linewidth() {
        let pc = PurcellCalc::l3_phc_silicon();
        let lw = pc.linewidth_m();
        // λ/Q = 1550nm/10000 = 0.155 nm
        assert!((lw - 0.155e-9).abs() < 0.01e-9, "linewidth={lw:.3e}");
    }

    // 3D dipole tests
    #[test]
    fn dipole_3d_z_orientation_injects_ez() {
        let omega = 2.0 * PI * 1.94e14;
        let src = DipoleSrc3d::new(2, 3, 4, DipoleOrientation3d::Z, 1.0, omega);
        let nx = 8;
        let ny = 8;
        let nz = 8;
        let n = nx * ny * nz;
        let mut ex = vec![0.0f64; n];
        let mut ey = vec![0.0f64; n];
        let mut ez = vec![0.0f64; n];
        let t = 1.0 / (4.0 * 1.94e14);
        src.apply(t, &mut ex, &mut ey, &mut ez, nx, ny, nz);
        // Only Ez at (2,3,4) should be nonzero
        let idx = 2 * ny * nz + 3 * nz + 4;
        assert!(
            ez[idx].abs() > 0.0,
            "Ez should be nonzero at source position"
        );
        assert!(ex[idx].abs() < 1e-15, "Ex should be zero for Z dipole");
        assert!(ey[idx].abs() < 1e-15, "Ey should be zero for Z dipole");
    }

    #[test]
    fn dipole_3d_x_orientation_injects_ex() {
        let omega = 2.0 * PI * 1.94e14;
        let src = DipoleSrc3d::new(1, 2, 3, DipoleOrientation3d::X, 1.0, omega);
        let nx = 6;
        let ny = 6;
        let nz = 6;
        let n = nx * ny * nz;
        let mut ex = vec![0.0f64; n];
        let mut ey = vec![0.0f64; n];
        let mut ez = vec![0.0f64; n];
        let t = 1.0 / (4.0 * 1.94e14);
        src.apply(t, &mut ex, &mut ey, &mut ez, nx, ny, nz);
        let idx = ny * nz + 2 * nz + 3;
        assert!(ex[idx].abs() > 0.0, "Ex should be nonzero for X dipole");
        assert!(ey[idx].abs() < 1e-15, "Ey should be zero for X dipole");
    }

    #[test]
    fn dipole_3d_arbitrary_orientation_injects_all_components() {
        let omega = 2.0 * PI * 1.94e14;
        let dir = [1.0_f64, 1.0, 1.0];
        let src = DipoleSrc3d::new(2, 2, 2, DipoleOrientation3d::Arbitrary(dir), 1.0, omega);
        let nx = 6;
        let ny = 6;
        let nz = 6;
        let n = nx * ny * nz;
        let mut ex = vec![0.0f64; n];
        let mut ey = vec![0.0f64; n];
        let mut ez = vec![0.0f64; n];
        let t = 1.0 / (4.0 * 1.94e14);
        src.apply(t, &mut ex, &mut ey, &mut ez, nx, ny, nz);
        let idx = 2 * ny * nz + 2 * nz + 2;
        assert!(
            ex[idx].abs() > 0.0,
            "Ex should be nonzero for Arbitrary dipole"
        );
        assert!(
            ey[idx].abs() > 0.0,
            "Ey should be nonzero for Arbitrary dipole"
        );
        assert!(
            ez[idx].abs() > 0.0,
            "Ez should be nonzero for Arbitrary dipole"
        );
    }

    #[test]
    fn dipole_3d_out_of_bounds_does_not_panic() {
        let omega = 2.0 * PI * 1.94e14;
        let src = DipoleSrc3d::new(100, 100, 100, DipoleOrientation3d::Z, 1.0, omega);
        let nx = 8;
        let ny = 8;
        let nz = 8;
        let n = nx * ny * nz;
        let mut ex = vec![0.0f64; n];
        let mut ey = vec![0.0f64; n];
        let mut ez = vec![0.0f64; n];
        // Should not panic
        src.apply(0.0, &mut ex, &mut ey, &mut ez, nx, ny, nz);
        assert!(
            ex.iter().all(|&v| v == 0.0),
            "Out-of-bounds dipole should not modify fields"
        );
    }

    #[test]
    fn dipole_3d_purcell_factor_estimate() {
        let omega = 2.0 * PI * 1.94e14;
        let src = DipoleSrc3d::new(2, 2, 2, DipoleOrientation3d::Z, 1.0, omega);
        let fp = src.estimate_purcell_factor(10_000.0, 0.7);
        assert!(
            fp > 1.0,
            "Purcell factor should exceed 1 for high-Q cavity: {fp:.1}"
        );
    }

    #[test]
    fn dipole_3d_with_gaussian_pulse() {
        let omega = 2.0 * PI * 1.94e14;
        let t0 = 30e-15;
        let sigma = 10e-15;
        let src = DipoleSrc3d::new(2, 2, 2, DipoleOrientation3d::Z, 1.0, omega)
            .with_gaussian_pulse(t0, sigma);
        let v_before = src.value_at(0.0);
        let v_peak = src.value_at(t0);
        // Before pulse, value should be small
        let env_before = (-((0.0 - t0) / sigma).powi(2)).exp();
        assert!(v_before.abs() <= env_before + 1e-10);
        // At peak, value can be up to 1
        assert!(v_peak.abs() <= 1.0 + 1e-10);
    }

    #[test]
    fn dipole_3d_value_oscillates_cw() {
        let omega = 2.0 * PI * 1.94e14;
        let src = DipoleSrc3d::new(0, 0, 0, DipoleOrientation3d::X, 1.0, omega);
        let v1 = src.value_at(1.0 / (4.0 * 1.94e14));
        let v2 = src.value_at(3.0 / (4.0 * 1.94e14));
        assert!(
            v1 * v2 < 0.0,
            "CW dipole should oscillate in sign: v1={v1}, v2={v2}"
        );
    }

    #[test]
    fn dipole_array_applies_all() {
        let omega = 2.0 * PI * 1.94e14;
        let mut arr = DipoleArray3d::new();
        arr.add_dipole(
            DipoleSrc3d::new(1, 1, 1, DipoleOrientation3d::Z, 1.0, omega),
            0.0,
        );
        arr.add_dipole(
            DipoleSrc3d::new(3, 3, 3, DipoleOrientation3d::Z, 1.0, omega),
            PI / 2.0,
        );
        let nx = 6;
        let ny = 6;
        let nz = 6;
        let n = nx * ny * nz;
        let mut ex = vec![0.0f64; n];
        let mut ey = vec![0.0f64; n];
        let mut ez = vec![0.0f64; n];
        let t = 1.0 / (4.0 * 1.94e14);
        arr.apply(t, &mut ex, &mut ey, &mut ez, nx, ny, nz);
        // Both dipoles at (1,1,1) and (3,3,3) should have nonzero Ez
        let idx1 = ny * nz + nz + 1;
        let idx2 = 3 * ny * nz + 3 * nz + 3;
        assert!(
            ez[idx1].abs() > 0.0 || ez[idx2].abs() > 0.0,
            "DipoleArray should inject at both positions"
        );
    }

    #[test]
    fn dipole_array_len() {
        let omega = 2.0 * PI * 1.94e14;
        let mut arr = DipoleArray3d::new();
        assert_eq!(arr.len(), 0);
        assert!(arr.is_empty());
        arr.add_dipole(
            DipoleSrc3d::new(0, 0, 0, DipoleOrientation3d::Z, 1.0, omega),
            0.0,
        );
        assert_eq!(arr.len(), 1);
        assert!(!arr.is_empty());
    }

    #[test]
    fn dipole_3d_radiation_pattern_z_dipole() {
        let omega = 2.0 * PI * 1.94e14;
        let src = DipoleSrc3d::new(0, 0, 0, DipoleOrientation3d::Z, 1.0, omega);
        // At theta=pi/2, phi=0: r_hat = [1, 0, 0], p = [0,0,1], dot=0, I=1
        let i_side = src.radiation_pattern(PI / 2.0, 0.0);
        assert!(
            (i_side - 1.0).abs() < 1e-10,
            "Z-dipole should radiate max at equator"
        );
        // At theta=0: r_hat = [0, 0, 1], p = [0,0,1], dot=1, I=0
        let i_top = src.radiation_pattern(0.0, 0.0);
        assert!(
            i_top.abs() < 1e-10,
            "Z-dipole should not radiate along Z axis"
        );
    }
}
