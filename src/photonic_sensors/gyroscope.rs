//! Optical gyroscope models based on the Sagnac effect.
//!
//! Includes:
//! - Fiber optic gyroscope (FOG)
//! - Ring laser gyroscope (RLG)
//! - Integrated photonic gyroscope (on-chip)

use std::f64::consts::PI;

/// Speed of light in vacuum (m/s).
const C: f64 = 2.997_924_58e8;
/// Reduced Planck constant × 2π = Planck constant (J·s).
const H_PLANCK: f64 = 6.626_070_15e-34;
/// Earth rotation rate (rad/s).
pub const EARTH_ROTATION_RATE_RAD_S: f64 = 7.292_115e-5;

// ---------------------------------------------------------------------------
// Fiber Optic Gyroscope (FOG)
// ---------------------------------------------------------------------------

/// Fiber optic gyroscope (FOG) model.
///
/// The FOG exploits the Sagnac effect: counter-propagating beams in a
/// multi-turn fibre coil accumulate a phase difference proportional to the
/// rotation rate about the coil axis.
#[derive(Debug, Clone)]
pub struct FiberOpticGyroscope {
    /// Coil mean radius (m).
    pub coil_radius_m: f64,
    /// Total fibre length in the coil (m).
    pub fiber_length_m: f64,
    /// Operating wavelength in vacuum (m).
    pub wavelength: f64,
    /// Number of turns in the coil.
    pub n_turns: usize,
    /// Photodetector current noise density (A/√Hz).
    pub photodetector_noise: f64,
}

impl FiberOpticGyroscope {
    /// Construct a FOG.
    ///
    /// # Arguments
    /// * `coil_radius` – Mean coil radius (m).
    /// * `n_turns` – Number of fibre turns.
    /// * `fiber_length` – Total fibre length; if zero it is computed from
    ///   `2π · r · n_turns`.
    /// * `wavelength` – Operating wavelength (m).
    pub fn new(coil_radius: f64, n_turns: usize, fiber_length: f64, wavelength: f64) -> Self {
        let computed_length = if fiber_length <= 0.0 {
            2.0 * PI * coil_radius * n_turns as f64
        } else {
            fiber_length
        };
        Self {
            coil_radius_m: coil_radius,
            fiber_length_m: computed_length,
            wavelength,
            n_turns,
            photodetector_noise: 1e-12,
        }
    }

    /// Enclosed area of the coil (m²).
    fn coil_area_m2(&self) -> f64 {
        PI * self.coil_radius_m * self.coil_radius_m * self.n_turns as f64
    }

    /// Sagnac phase shift for a given rotation rate.
    ///
    /// `Δφ = 4π · A · Ω / (λ · c)` where `A = N · π · r²`.
    ///
    /// # Arguments
    /// * `rotation_rate_rad_s` – Rotation rate about the coil axis (rad/s).
    ///
    /// # Returns
    /// Phase difference (rad).
    pub fn sagnac_phase(&self, rotation_rate_rad_s: f64) -> f64 {
        4.0 * PI * self.coil_area_m2() * rotation_rate_rad_s / (self.wavelength * C)
    }

    /// Scale factor of the FOG (rad / (rad/s)).
    ///
    /// `S = 4π · A / (λ · c)`
    pub fn scale_factor(&self) -> f64 {
        4.0 * PI * self.coil_area_m2() / (self.wavelength * C)
    }

    /// Recover the rotation rate from a measured Sagnac phase.
    ///
    /// `Ω = Δφ / S`
    ///
    /// # Arguments
    /// * `delta_phase_rad` – Measured Sagnac phase (rad).
    ///
    /// # Returns
    /// Rotation rate (rad/s).
    pub fn rotation_rate(&self, delta_phase_rad: f64) -> f64 {
        let s = self.scale_factor();
        if s.abs() < f64::EPSILON {
            return 0.0;
        }
        delta_phase_rad / s
    }

    /// Noise-equivalent rotation rate (shot-noise limited).
    ///
    /// The shot-noise limited phase sensitivity is:
    /// `δφ_min = √(2·h·f / (η·P))` [rad/√Hz]
    ///
    /// from which `NEΩ = δφ_min / S`.
    ///
    /// # Arguments
    /// * `optical_power_w` – Optical power entering the coil (W).
    ///
    /// # Returns
    /// Noise-equivalent rotation rate (rad/s/√Hz).
    pub fn noise_equivalent_rotation(&self, optical_power_w: f64) -> f64 {
        if optical_power_w <= 0.0 {
            return f64::INFINITY;
        }
        let freq = C / self.wavelength;
        let h_nu = H_PLANCK * freq;
        // Shot-noise phase: δφ = √(2·hν/P) rad/√Hz
        let delta_phi = (2.0 * h_nu / optical_power_w).sqrt();
        let s = self.scale_factor();
        if s.abs() < f64::EPSILON {
            return f64::INFINITY;
        }
        delta_phi / s
    }

    /// Approximate bias stability dominated by 1/f noise.
    ///
    /// A typical high-quality FOG achieves ~0.001 deg/hr at 1 km of fibre.
    /// We scale empirically as `0.001 · (1000/L)^0.5` deg/hr.
    ///
    /// # Returns
    /// Bias stability (deg/hr).
    pub fn bias_stability_deg_per_hr(&self) -> f64 {
        0.001_f64 * (1000.0 / self.fiber_length_m).sqrt()
    }

    /// Test whether Earth's rotation rate is measurable above the noise floor.
    ///
    /// # Arguments
    /// * `power_w` – Input optical power (W).
    ///
    /// # Returns
    /// `true` if `NEΩ < EARTH_ROTATION_RATE_RAD_S`.
    pub fn earth_rotation_measurable(&self, power_w: f64) -> bool {
        self.noise_equivalent_rotation(power_w) < EARTH_ROTATION_RATE_RAD_S
    }

    /// Lock-in threshold (dead band) for a basic open-loop FOG.
    ///
    /// Mechanically dithered FOGs overcome this; here we estimate the
    /// threshold as `NEΩ · √(Δf_dither)` with `Δf_dither ≈ 100 Hz`.
    ///
    /// # Returns
    /// Lock-in threshold rotation rate (rad/s) — approximate.
    pub fn lock_in_threshold_rad_s(&self) -> f64 {
        // Open-loop FOG lock-in is dominated by back-scatter; typical value ~1e-6 rad/s
        // We model it as a fixed fraction of the scale factor expressed as a rate.
        let typical_backscatter_phase = 1e-7_f64; // rad RMS
        let s = self.scale_factor();
        if s.abs() < f64::EPSILON {
            return 0.0;
        }
        typical_backscatter_phase / s
    }
}

// ---------------------------------------------------------------------------
// Ring Laser Gyroscope (RLG)
// ---------------------------------------------------------------------------

/// Ring laser gyroscope (RLG) model.
///
/// The RLG uses a laser cavity formed in a closed ring path. Rotation splits
/// the CW and CCW resonant frequencies by the Sagnac effect, producing a
/// measurable beat note.
#[derive(Debug, Clone)]
pub struct RingLaserGyroscope {
    /// Enclosed area of the ring cavity (m²).
    pub cavity_area_m2: f64,
    /// Optical path length (perimeter) of the cavity (m).
    pub perimeter_m: f64,
    /// Operating wavelength (m).
    pub wavelength: f64,
    /// Quality factor of the ring cavity.
    pub q_factor: f64,
}

impl RingLaserGyroscope {
    /// Construct a square ring laser gyroscope.
    ///
    /// # Arguments
    /// * `side_length_m` – Side length of the square ring (m).
    /// * `wavelength` – Operating wavelength (m).
    pub fn new(side_length_m: f64, wavelength: f64) -> Self {
        Self {
            cavity_area_m2: side_length_m * side_length_m,
            perimeter_m: 4.0 * side_length_m,
            wavelength,
            q_factor: 1e12,
        }
    }

    /// Sagnac beat frequency for a given rotation rate.
    ///
    /// `Δf = 4 · A · Ω / (λ · P)`
    ///
    /// # Arguments
    /// * `rotation_rate_rad_s` – Rotation rate (rad/s).
    ///
    /// # Returns
    /// Beat frequency (Hz).
    pub fn beat_frequency_hz(&self, rotation_rate_rad_s: f64) -> f64 {
        4.0 * self.cavity_area_m2 * rotation_rate_rad_s / (self.wavelength * self.perimeter_m)
    }

    /// Scale factor: Hz per (rad/s).
    ///
    /// `K = 4A / (λP)`
    pub fn scale_factor(&self) -> f64 {
        4.0 * self.cavity_area_m2 / (self.wavelength * self.perimeter_m)
    }

    /// Lock-in rate: below this rotation rate the CW and CCW modes
    /// frequency-lock and the gyroscope output is zero.
    ///
    /// `Ω_L ≈ r · P / (4A · τ_rt · Q)` where `τ_rt = P/c`.
    ///
    /// # Returns
    /// Lock-in rotation rate (rad/s).
    pub fn lock_in_rate_rad_s(&self) -> f64 {
        let tau_rt = self.perimeter_m / C; // round-trip time (s)
        self.perimeter_m / (4.0 * self.cavity_area_m2 * tau_rt * self.q_factor * 2.0 * PI)
    }

    /// Minimum detectable rotation rate (shot-noise limited).
    ///
    /// `δΩ = 1/K · √(h·f·BW / P)` [rad/s/√Hz]
    ///
    /// # Arguments
    /// * `power_w` – Intracavity optical power (W).
    ///
    /// # Returns
    /// Sensitivity (rad/s/√Hz).
    pub fn sensitivity_rad_s_per_sqrthz(&self, power_w: f64) -> f64 {
        if power_w <= 0.0 {
            return f64::INFINITY;
        }
        let freq = C / self.wavelength;
        let h_nu = H_PLANCK * freq;
        let k = self.scale_factor();
        if k.abs() < f64::EPSILON {
            return f64::INFINITY;
        }
        (h_nu / power_w).sqrt() / k
    }
}

// ---------------------------------------------------------------------------
// Integrated Photonic Gyroscope (on-chip)
// ---------------------------------------------------------------------------

/// Integrated photonic gyroscope implemented as a waveguide coil on a chip.
///
/// Compared with a FOG the enclosed area is much smaller but propagation
/// loss and compact form factor are the key trade-offs.
#[derive(Debug, Clone)]
pub struct IntegratedGyroscope {
    /// Total waveguide length (m).
    pub waveguide_length_m: f64,
    /// Coil radius (mm).
    pub coil_radius_mm: f64,
    /// Waveguide effective refractive index.
    pub n_eff: f64,
    /// Waveguide propagation loss (dB/cm).
    pub propagation_loss_db_per_cm: f64,
    /// Operating wavelength (m).
    pub wavelength: f64,
}

impl IntegratedGyroscope {
    /// Construct an integrated waveguide gyroscope.
    ///
    /// # Arguments
    /// * `length_m` – Total waveguide path length (m).
    /// * `radius_mm` – Coil radius (mm).
    /// * `n_eff` – Effective refractive index of the guided mode.
    /// * `loss_db_cm` – Propagation loss (dB/cm).
    /// * `wavelength` – Operating wavelength (m).
    pub fn new(
        length_m: f64,
        radius_mm: f64,
        n_eff: f64,
        loss_db_cm: f64,
        wavelength: f64,
    ) -> Self {
        Self {
            waveguide_length_m: length_m,
            coil_radius_mm: radius_mm,
            n_eff,
            propagation_loss_db_per_cm: loss_db_cm,
            wavelength,
        }
    }

    /// Number of turns.
    fn n_turns(&self) -> f64 {
        let radius_m = self.coil_radius_mm * 1e-3;
        if radius_m <= 0.0 {
            return 0.0;
        }
        self.waveguide_length_m / (2.0 * PI * radius_m)
    }

    /// Enclosed area of the integrated coil (m²).
    fn area_m2(&self) -> f64 {
        let radius_m = self.coil_radius_mm * 1e-3;
        PI * radius_m * radius_m * self.n_turns()
    }

    /// Sagnac phase shift (rad) for a rotation rate (rad/s).
    ///
    /// `Δφ = 4π · A · n_eff · Ω / (λ · c)`
    ///
    /// The `n_eff` factor accounts for the phase velocity in the waveguide.
    pub fn sagnac_phase(&self, omega: f64) -> f64 {
        4.0 * PI * self.area_m2() * self.n_eff * omega / (self.wavelength * C)
    }

    /// Scale factor (rad / (rad/s)).
    pub fn scale_factor(&self) -> f64 {
        4.0 * PI * self.area_m2() * self.n_eff / (self.wavelength * C)
    }

    /// Minimum detectable rotation rate (shot-noise limited), accounting for
    /// propagation loss reducing the available optical power.
    ///
    /// # Arguments
    /// * `source_power_w` – Input optical power from the source (W).
    ///
    /// # Returns
    /// Minimum detectable rotation rate (rad/s/√Hz).
    pub fn minimum_detectable_rotation(&self, source_power_w: f64) -> f64 {
        if source_power_w <= 0.0 {
            return f64::INFINITY;
        }
        let loss_db = self.propagation_loss_penalty_db();
        let power_after_loss = source_power_w * 10.0_f64.powf(-loss_db / 10.0);
        let freq = C / self.wavelength;
        let h_nu = H_PLANCK * freq;
        let s = self.scale_factor();
        if s.abs() < f64::EPSILON || power_after_loss <= 0.0 {
            return f64::INFINITY;
        }
        (2.0 * h_nu / power_after_loss).sqrt() / s
    }

    /// Total propagation loss penalty over the full waveguide length (dB).
    pub fn propagation_loss_penalty_db(&self) -> f64 {
        self.propagation_loss_db_per_cm * (self.waveguide_length_m * 100.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn fog_sagnac_phase_earth_rotation() {
        // 1 km fibre, 10 cm radius coil at 1550 nm
        let fog = FiberOpticGyroscope::new(0.10, 1592, 1000.0, 1550e-9);
        let phi = fog.sagnac_phase(EARTH_ROTATION_RATE_RAD_S);
        // Should be on the order of micro-radians to milli-radians
        assert!(
            phi > 1e-6 && phi < 1.0,
            "Sagnac phase out of expected range: {}",
            phi
        );
    }

    #[test]
    fn fog_scale_factor_roundtrip() {
        let fog = FiberOpticGyroscope::new(0.15, 1000, 942.5, 1310e-9);
        let omega = 0.01_f64; // rad/s
        let phi = fog.sagnac_phase(omega);
        let omega_recovered = fog.rotation_rate(phi);
        assert_abs_diff_eq!(omega_recovered, omega, epsilon = 1e-12);
    }

    #[test]
    fn fog_earth_rotation_detectable_high_power() {
        let fog = FiberOpticGyroscope::new(0.15, 2000, 1884.0, 1550e-9);
        assert!(
            fog.earth_rotation_measurable(1e-3),
            "Should detect Earth rotation at 1 mW"
        );
    }

    #[test]
    fn rlg_beat_frequency_scaling() {
        // 10 cm square ring at 633 nm
        let rlg = RingLaserGyroscope::new(0.10, 633e-9);
        let f = rlg.beat_frequency_hz(EARTH_ROTATION_RATE_RAD_S);
        // Typical RLG gives ~1–100 Hz for Earth rate
        assert!(
            f > 0.0 && f < 1000.0,
            "Beat frequency out of range: {} Hz",
            f
        );
    }

    #[test]
    fn rlg_scale_factor_beat_consistency() {
        let rlg = RingLaserGyroscope::new(0.05, 1550e-9);
        let k = rlg.scale_factor();
        let omega = 1.0_f64;
        assert_abs_diff_eq!(rlg.beat_frequency_hz(omega), k * omega, epsilon = 1e-10);
    }

    #[test]
    fn integrated_gyro_loss_penalty() {
        // 1 m waveguide with 1 dB/cm → 100 dB total loss
        let ig = IntegratedGyroscope::new(1.0, 5.0, 1.5, 1.0, 1550e-9);
        assert_abs_diff_eq!(ig.propagation_loss_penalty_db(), 100.0, epsilon = 1e-6);
    }

    #[test]
    fn integrated_gyro_sagnac_sign() {
        let ig = IntegratedGyroscope::new(0.1, 2.0, 1.5, 0.5, 1550e-9);
        let phi_pos = ig.sagnac_phase(1.0);
        let phi_neg = ig.sagnac_phase(-1.0);
        assert!(phi_pos > 0.0 && phi_neg < 0.0);
        assert_abs_diff_eq!(phi_pos, -phi_neg, epsilon = 1e-20);
    }
}
