//! MEMS scanning mirrors: tilt mirrors, gimbal mirrors, and variable optical attenuators.
//!
//! Models resonant and quasi-static MEMS mirrors used in LIDAR, OCT beam scanning,
//! optical coherence tomography, and fiber-optic switching. Includes:
//!
//! - [`MemosTiltMirror`]: 2-axis tilt mirror with Lissajous and raster scan patterns.
//! - [`GimbalMirror`]: Gimbal-mounted mirror with decoupled inner/outer axes.
//! - [`MemosVoa`]: MEMS variable optical attenuator.
//!
//! # References
//! - Urey, H. (2002). Torsional MEMS scanner design for high-resolution displays.
//!   *SPIE Photonics West*.
//! - Rochus, V. et al. (2006). Nonlinear modelling of electro-mechanical coupling.
//!   *Sensors and Actuators A*.

use std::f64::consts::PI;

/// MEMS two-axis tilt mirror.
///
/// Models a mirror plate suspended on torsional springs, driven resonantly
/// or quasi-statically by electrostatic comb drives or vertical electrodes.
///
/// # Example
/// ```
/// use oxiphoton::mems::mems_mirror::MemosTiltMirror;
/// let m = MemosTiltMirror::new(1e-3, 1000.0, 500.0, 200.0, 150.0);
/// let [tx, ty] = m.scan_lissajous(0.0, 1.5);
/// assert!((tx).abs() < m.max_angle_x + 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct MemosTiltMirror {
    /// Mirror plate side length (m), assumed square.
    pub mirror_size: f64,
    /// Resonant frequency of the fast (x) axis (Hz).
    pub resonant_freq_x: f64,
    /// Resonant frequency of the slow (y) axis (Hz).
    pub resonant_freq_y: f64,
    /// Mechanical quality factor of the fast axis.
    pub q_factor_x: f64,
    /// Mechanical quality factor of the slow axis.
    pub q_factor_y: f64,
    /// Maximum optical scan angle of the fast axis (rad, half-angle).
    pub max_angle_x: f64,
    /// Maximum optical scan angle of the slow axis (rad, half-angle).
    pub max_angle_y: f64,
    /// Nominal drive voltage (V).
    pub drive_voltage: f64,
}

impl MemosTiltMirror {
    /// Construct a new tilt mirror with default scan angles and drive voltage.
    ///
    /// Defaults:
    /// - `max_angle_x = max_angle_y = 0.2 rad` (~11.5° optical half-angle)
    /// - `drive_voltage = 100 V` (typical electrostatic MEMS)
    pub fn new(size: f64, freq_x: f64, freq_y: f64, q_x: f64, q_y: f64) -> Self {
        Self {
            mirror_size: size,
            resonant_freq_x: freq_x,
            resonant_freq_y: freq_y,
            q_factor_x: q_x,
            q_factor_y: q_y,
            max_angle_x: 0.2,
            max_angle_y: 0.2,
            drive_voltage: 100.0,
        }
    }

    /// Instantaneous scan angles for a Lissajous pattern.
    ///
    /// `[θx, θy]` at time `t` (s):
    /// - θx = θ_max_x · sin(2π·fx·t)
    /// - θy = θ_max_y · sin(2π·fy·freq_ratio·t + π/2)
    ///
    /// `freq_ratio` sets fy/fx; for a 1:1 ratio with π/2 phase offset the
    /// pattern is circular.
    pub fn scan_lissajous(&self, t: f64, freq_ratio: f64) -> [f64; 2] {
        let theta_x = self.max_angle_x * (2.0 * PI * self.resonant_freq_x * t).sin();
        let theta_y =
            self.max_angle_y * (2.0 * PI * self.resonant_freq_y * freq_ratio * t + PI / 2.0).sin();
        [theta_x, theta_y]
    }

    /// Instantaneous scan angles for a raster pattern.
    ///
    /// `[θx, θy]` at time `t` (s):
    /// - Fast axis (x): sinusoidal at `resonant_freq_x`.
    /// - Slow axis (y): staircase (step per line), advancing each `line_time` seconds.
    ///
    /// # Arguments
    /// * `t` - Current time (s).
    /// * `line_time` - Duration of one horizontal line (s).
    /// * `n_lines` - Total number of scan lines.
    pub fn scan_raster(&self, t: f64, line_time: f64, n_lines: usize) -> [f64; 2] {
        if n_lines == 0 || line_time <= 0.0 {
            return [0.0, 0.0];
        }
        let theta_x = self.max_angle_x * (2.0 * PI * self.resonant_freq_x * t).sin();
        // Current line index (wraps around)
        let frame_time = line_time * n_lines as f64;
        let t_in_frame = t.rem_euclid(frame_time);
        let line_idx = (t_in_frame / line_time) as usize;
        // Staircase from -max to +max over n_lines steps
        let n = n_lines as f64;
        let theta_y = self.max_angle_y * (2.0 * (line_idx as f64 / (n - 1.0).max(1.0)) - 1.0);
        [theta_x, theta_y]
    }

    /// Transverse beam displacement (m) at a distance `distance` (m) from the mirror.
    ///
    /// For a tilt mirror, the beam deflects by 2·θ·distance (factor-of-2 from reflection).
    pub fn beam_deflection(&self, angle: f64, distance: f64) -> f64 {
        2.0 * angle * distance
    }

    /// 3-dB mechanical bandwidth of the fast axis (Hz).
    ///
    /// Δf = f₀_x / Q_x
    pub fn bandwidth_3db(&self) -> f64 {
        self.resonant_freq_x / self.q_factor_x
    }

    /// Actuation sensitivity (rad/V) at the resonant frequency.
    ///
    /// Approximated as: θ_max / V_drive (small-signal linear model).
    pub fn actuation_sensitivity(&self) -> f64 {
        self.max_angle_x / self.drive_voltage
    }

    /// Frequency response amplitude of the fast axis at drive frequency `f` (Hz).
    ///
    /// Uses the driven harmonic oscillator model.
    pub fn frequency_response_x(&self, f: f64) -> f64 {
        let ratio = f / self.resonant_freq_x;
        let denom = ((1.0 - ratio * ratio).powi(2) + (ratio / self.q_factor_x).powi(2)).sqrt();
        if denom < f64::EPSILON {
            self.q_factor_x
        } else {
            1.0 / denom
        }
    }

    /// Frequency response amplitude of the slow axis at drive frequency `f` (Hz).
    pub fn frequency_response_y(&self, f: f64) -> f64 {
        let ratio = f / self.resonant_freq_y;
        let denom = ((1.0 - ratio * ratio).powi(2) + (ratio / self.q_factor_y).powi(2)).sqrt();
        if denom < f64::EPSILON {
            self.q_factor_y
        } else {
            1.0 / denom
        }
    }

    /// Optical field of view (FOV) in radians (full angle) for each axis.
    ///
    /// Returns `[fov_x, fov_y]` — full optical scan angles.
    pub fn field_of_view(&self) -> [f64; 2] {
        [2.0 * self.max_angle_x, 2.0 * self.max_angle_y]
    }

    /// Mirror moment of inertia about the tilt axis (kg·m²).
    ///
    /// Approximated as a thin square plate rotating about a central axis:
    /// I = m·a²/6 where a = mirror_size and m is plate mass with 2.3e-6 kg/m² aerial density.
    pub fn moment_of_inertia(&self) -> f64 {
        // Polysilicon areal density ~2.3e-6 kg/m² for a 1 µm thick Si plate
        let areal_density = 2_330.0 * 1e-6; // Si density * 1 µm thickness
        let m = areal_density * self.mirror_size * self.mirror_size;
        m * self.mirror_size * self.mirror_size / 6.0
    }
}

/// Gimbal-mounted MEMS mirror with decoupled inner and outer tilt axes.
///
/// The inner mirror performs fast scanning; the outer frame provides the
/// orthogonal slow axis. The two axes are mechanically decoupled by the gimbal.
#[derive(Debug, Clone)]
pub struct GimbalMirror {
    /// Inner mirror (fast axis).
    pub inner_mirror: MemosTiltMirror,
    /// Resonant frequency of the outer gimbal frame (Hz).
    pub outer_frame_freq: f64,
    /// Quality factor of the outer gimbal frame.
    pub outer_frame_q: f64,
}

impl GimbalMirror {
    /// Construct a gimbal mirror from an inner mirror and outer frame parameters.
    pub fn new(inner: MemosTiltMirror, outer_freq: f64, outer_q: f64) -> Self {
        Self {
            inner_mirror: inner,
            outer_frame_freq: outer_freq,
            outer_frame_q: outer_q,
        }
    }

    /// Frequency response of both axes at drive frequency `drive_freq` (Hz)
    /// with drive amplitude `drive_amp` (V).
    ///
    /// Returns `[θx, θy]` — peak optical angles of inner and outer axes.
    pub fn coupled_response(&self, drive_freq: f64, drive_amp: f64) -> [f64; 2] {
        // Inner axis response (fast)
        let h_inner = self.inner_mirror.frequency_response_x(drive_freq);
        // Outer axis response
        let ratio_outer = drive_freq / self.outer_frame_freq;
        let h_outer = {
            let denom = ((1.0 - ratio_outer * ratio_outer).powi(2)
                + (ratio_outer / self.outer_frame_q).powi(2))
            .sqrt();
            if denom < f64::EPSILON {
                self.outer_frame_q
            } else {
                1.0 / denom
            }
        };
        let theta_x = self.inner_mirror.actuation_sensitivity() * drive_amp * h_inner;
        let theta_y = self.inner_mirror.actuation_sensitivity() * drive_amp * h_outer;
        [theta_x, theta_y]
    }

    /// Scan Lissajous with both axes driven at their natural frequencies.
    pub fn scan_lissajous(&self, t: f64) -> [f64; 2] {
        let theta_x = self.inner_mirror.max_angle_x
            * (2.0 * PI * self.inner_mirror.resonant_freq_x * t).sin();
        let theta_y =
            self.inner_mirror.max_angle_y * (2.0 * PI * self.outer_frame_freq * t + PI / 2.0).sin();
        [theta_x, theta_y]
    }
}

/// MEMS Variable Optical Attenuator (VOA).
///
/// A laterally-displaced mirror or shutter element that partially obscures a
/// guided mode, producing attenuation proportional to displacement.
///
/// # Model
/// The attenuation is modelled using a Gaussian beam overlap integral:
///
/// Loss(Δx) = −10·log₁₀(exp(−(Δx/w₀)²))
///
/// where w₀ is the 1/e² beam waist at the VOA gap.
#[derive(Debug, Clone)]
pub struct MemosVoa {
    /// Air gap length between input and output fibers/waveguides (m).
    pub gap: f64,
    /// Operating wavelength (m).
    pub wavelength: f64,
    /// Fixed insertion loss at zero displacement (dB).
    pub coupling_loss_db: f64,
    /// Beam waist at the gap (m), derived from `wavelength` and `gap`.
    beam_waist: f64,
}

impl MemosVoa {
    /// Construct a MEMS VOA for the given wavelength.
    ///
    /// Uses a standard single-mode fiber MFD of 10 µm and computes the
    /// Gaussian beam waist at the midpoint of a 200 µm air gap.
    pub fn new(wavelength: f64) -> Self {
        // Typical SMF MFD = 10 µm, half at each side
        let w0_fiber = 5e-6; // 1/e² radius at fiber facet
        let gap = 200e-6;
        // Diffraction-broadened waist at gap midpoint: w(z) = w0*sqrt(1+(z/zR)^2)
        // zR = π*w0^2/λ
        let z_r = PI * w0_fiber * w0_fiber / wavelength;
        let z = gap / 2.0;
        let w_gap = w0_fiber * (1.0 + (z / z_r).powi(2)).sqrt();
        Self {
            gap,
            wavelength,
            coupling_loss_db: 0.5, // typical SMF-SMF coupling loss
            beam_waist: w_gap,
        }
    }

    /// Attenuation (dB) for a lateral displacement `displacement` (m) of the shutter.
    ///
    /// Uses Gaussian beam transverse overlap:
    /// T(Δx) = exp(−(Δx/w₀)²)
    /// Loss = −10·log₁₀(T)
    pub fn attenuation_db(&self, displacement: f64) -> f64 {
        let t = (-(displacement / self.beam_waist).powi(2)).exp();
        if t <= 0.0 {
            f64::INFINITY
        } else {
            -10.0 * t.log10()
        }
    }

    /// Fixed insertion loss at zero displacement (dB).
    pub fn insertion_loss_db(&self) -> f64 {
        self.coupling_loss_db
    }

    /// Total loss (dB) including insertion loss and displacement-induced attenuation.
    pub fn total_loss_db(&self, displacement: f64) -> f64 {
        self.insertion_loss_db() + self.attenuation_db(displacement)
    }

    /// Return the displacement (m) required to achieve a target attenuation (dB).
    ///
    /// Inverts the Gaussian model: Δx = w₀ · √(ln(10) · att_db / 10)
    ///
    /// Returns `None` if the target is zero (no attenuation needed).
    pub fn displacement_for_attenuation(&self, att_db: f64) -> Option<f64> {
        if att_db <= 0.0 {
            return None;
        }
        let dx = self.beam_waist * (att_db * 10_f64.ln() / 10.0).sqrt();
        Some(dx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_tilt_mirror() -> MemosTiltMirror {
        MemosTiltMirror::new(1e-3, 1000.0, 500.0, 200.0, 150.0)
    }

    #[test]
    fn test_lissajous_within_fov() {
        let m = make_tilt_mirror();
        for i in 0..100 {
            let t = i as f64 * 1e-5;
            let [tx, ty] = m.scan_lissajous(t, 1.5);
            assert!(tx.abs() <= m.max_angle_x + 1e-12, "tx out of range: {tx}");
            assert!(ty.abs() <= m.max_angle_y + 1e-12, "ty out of range: {ty}");
        }
    }

    #[test]
    fn test_raster_x_sinusoidal() {
        let m = make_tilt_mirror();
        let line_time = 1e-3;
        let n_lines = 100;
        // At t=0 the fast axis should be at 0
        let [tx0, _] = m.scan_raster(0.0, line_time, n_lines);
        assert_abs_diff_eq!(tx0, 0.0, epsilon = 1e-12);
        // At t = 1/(4*fx) the fast axis is at max
        let t_max = 1.0 / (4.0 * m.resonant_freq_x);
        let [tx_max, _] = m.scan_raster(t_max, line_time, n_lines);
        assert_abs_diff_eq!(tx_max, m.max_angle_x, epsilon = 1e-12);
    }

    #[test]
    fn test_beam_deflection() {
        let m = make_tilt_mirror();
        let angle = 0.1; // rad
        let dist = 0.5; // 50 cm
        let deflection = m.beam_deflection(angle, dist);
        assert_abs_diff_eq!(deflection, 2.0 * angle * dist, epsilon = 1e-15);
    }

    #[test]
    fn test_bandwidth_3db() {
        let m = make_tilt_mirror();
        let bw = m.bandwidth_3db();
        // bw = f0/Q = 1000/200 = 5 Hz
        assert_abs_diff_eq!(bw, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_actuation_sensitivity() {
        let m = make_tilt_mirror();
        let sens = m.actuation_sensitivity();
        // 0.2 rad / 100 V = 0.002 rad/V
        assert_abs_diff_eq!(sens, 0.002, epsilon = 1e-10);
    }

    #[test]
    fn test_gimbal_coupled_response() {
        let inner = make_tilt_mirror();
        let gimbal = GimbalMirror::new(inner, 200.0, 100.0);
        let [tx, _ty] = gimbal.coupled_response(1000.0, 50.0);
        // At resonance of inner axis, response is amplified by Q
        assert!(tx > 0.0, "inner axis response should be positive");
    }

    #[test]
    fn test_voa_zero_displacement() {
        let voa = MemosVoa::new(1550e-9);
        let att = voa.attenuation_db(0.0);
        assert_abs_diff_eq!(att, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_voa_increasing_attenuation() {
        let voa = MemosVoa::new(1550e-9);
        let att1 = voa.attenuation_db(1e-6);
        let att2 = voa.attenuation_db(5e-6);
        assert!(
            att2 > att1,
            "larger displacement should give more attenuation"
        );
    }

    #[test]
    fn test_voa_displacement_for_attenuation() {
        let voa = MemosVoa::new(1550e-9);
        let target_db = 3.0;
        let dx = voa
            .displacement_for_attenuation(target_db)
            .expect("should return a displacement for positive attenuation");
        let att = voa.attenuation_db(dx);
        assert_abs_diff_eq!(att, target_db, epsilon = 1e-6);
    }

    #[test]
    fn test_frequency_response_at_resonance() {
        let m = make_tilt_mirror();
        // At f = f0_x, response = Q
        let h = m.frequency_response_x(m.resonant_freq_x);
        assert_abs_diff_eq!(h, m.q_factor_x, epsilon = 0.1);
    }
}
