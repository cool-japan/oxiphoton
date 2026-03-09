//! LiDAR-specific OPA design and link-budget analysis.
//!
//! Covers:
//! - OPA-based solid-state LiDAR system parameters
//! - Maximum range, range resolution, angular resolution
//! - Phase noise to pointing error mapping
//! - Silicon photonics chip-scale OPA specification
//!
//! All quantities in SI units unless noted.

use std::f64::consts::PI;

// ─── Physical constants ───────────────────────────────────────────────────────

const C_LIGHT: f64 = 2.997_924_58e8; // m/s
const H_PLANCK: f64 = 6.626_070_15e-34; // J·s

// ─── OpaLidar ────────────────────────────────────────────────────────────────

/// OPA-based pulsed direct-detection LiDAR system.
///
/// The receive sensitivity is modelled using a simplified optical link budget:
///
///   P_r = P_t × A_r × ρ / (4π R²) × Ω_beam / (4π)
///
/// where:
/// - P_t  = peak pulse power (W)
/// - A_r  = receiver aperture area (m²)
/// - ρ    = target surface reflectivity (dimensionless, Lambertian)
/// - R    = range (m)
/// - Ω_beam = beam solid angle (sr) ≈ (HPBW)²
///
/// SNR-limited maximum range uses the simplified threshold:
///
///   P_r(R_max) = N_ph_min × h ν × bandwidth
///
/// where N_ph_min ≈ 10 (photons for shot-noise limited detection at SNR = 5).
#[derive(Debug, Clone)]
pub struct OpaLidar {
    /// OPA aperture (1D phased array)
    pub opa: super::phased_array::OpticalPhasedArray1d,
    /// Pulse energy (J)
    pub pulse_energy_j: f64,
    /// Pulse duration (s)
    pub pulse_duration_s: f64,
    /// Receiver aperture area (m²)
    pub receiver_aperture_m2: f64,
    /// Operating wavelength (m)
    pub wavelength_m: f64,
    /// Phase DAC resolution (bits)
    pub n_bits_dac: u32,
}

impl OpaLidar {
    /// Angular resolution (HPBW) in radians.
    pub fn angular_resolution_rad(&self) -> f64 {
        self.opa.hpbw_rad()
    }

    /// Angular resolution in degrees.
    pub fn angular_resolution_deg(&self) -> f64 {
        self.angular_resolution_rad().to_degrees()
    }

    /// Number of resolvable points across a given field-of-view.
    ///
    /// N_pts = FOV / HPBW
    ///
    /// * `fov_rad` — total scan field-of-view (radians)
    pub fn n_resolvable_points(&self, fov_rad: f64) -> usize {
        let hpbw = self.angular_resolution_rad();
        if hpbw < f64::MIN_POSITIVE {
            return 0;
        }
        (fov_rad / hpbw).floor() as usize
    }

    /// Peak pulse power derived from energy and duration.
    ///
    ///   P_peak = E_pulse / τ_pulse
    pub fn peak_power_w(&self) -> f64 {
        if self.pulse_duration_s < f64::MIN_POSITIVE {
            return 0.0;
        }
        self.pulse_energy_j / self.pulse_duration_s
    }

    /// Photon energy: E_ph = h ν = h c / λ
    pub fn photon_energy_j(&self) -> f64 {
        H_PLANCK * C_LIGHT / self.wavelength_m
    }

    /// Beam solid angle (sr) ≈ (HPBW)² for a 1D→2D approximation.
    ///
    /// For a 1D array, the divergence in the transverse plane is assumed to
    /// be limited by diffraction from a single emitter (isotropic in that
    /// plane), giving Ω_beam ≈ π × (HPBW/2)² / 4 ≈ HPBW².
    pub fn beam_solid_angle_sr(&self) -> f64 {
        let hpbw = self.angular_resolution_rad();
        hpbw * hpbw
    }

    /// Received power at range R for a Lambertian target with reflectivity ρ.
    ///
    ///   P_r = P_peak × A_r × ρ / (π R²)  [Lambertian, full hemisphere]
    ///       × (Ω_beam / (4π))              [fraction intercepted by beam]
    ///
    /// Simplified far-field form:
    ///
    ///   P_r ≈ P_peak × A_r × ρ × Ω_beam / (4π² R²)
    pub fn received_power_w(&self, range_m: f64, target_reflectivity: f64) -> f64 {
        if range_m < 1.0e-6 {
            return 0.0;
        }
        let p_peak = self.peak_power_w();
        let omega = self.beam_solid_angle_sr();
        // Simplified LIDAR equation (Lambertian target):
        // P_r = P_t A_r ρ / (4π R²)  ← already includes beam solid angle in numerator
        p_peak * self.receiver_aperture_m2 * target_reflectivity * omega
            / (4.0 * PI * PI * range_m * range_m)
    }

    /// Maximum range limited by shot-noise SNR.
    ///
    /// Solves P_r(R_max) = N_min × h ν × noise_bandwidth_hz, with N_min = 10.
    ///
    ///   R_max = sqrt( P_peak × A_r × ρ × Ω_beam / (4π² × N_min × hν × BW) )
    ///
    /// * `target_reflectivity` — Lambertian surface reflectivity (0–1)
    /// * `noise_bandwidth_hz` — receiver electrical noise bandwidth (Hz)
    pub fn max_range_m(&self, target_reflectivity: f64, noise_bandwidth_hz: f64) -> f64 {
        const N_MIN_PHOTONS: f64 = 10.0; // minimum detectable photon count
        let p_noise = N_MIN_PHOTONS * self.photon_energy_j() * noise_bandwidth_hz;
        if p_noise < f64::MIN_POSITIVE {
            return f64::MAX;
        }
        let p_peak = self.peak_power_w();
        let omega = self.beam_solid_angle_sr();
        let numerator = p_peak * self.receiver_aperture_m2 * target_reflectivity * omega;
        let denominator = 4.0 * PI * PI * p_noise;
        if denominator < f64::MIN_POSITIVE {
            return 0.0;
        }
        (numerator / denominator).sqrt()
    }

    /// Range resolution: ΔR = c τ_pulse / 2.
    pub fn range_resolution_m(&self) -> f64 {
        C_LIGHT * self.pulse_duration_s / 2.0
    }

    /// Pointing error (radians) arising from phase noise on the OPA.
    ///
    ///   Δθ = Δφ_rms × λ / (2π d)
    ///
    /// * `phase_noise_rms_rad` — RMS phase noise per element (rad)
    pub fn pointing_error_from_phase_noise_rad(&self, phase_noise_rms_rad: f64) -> f64 {
        let d = self.opa.pitch_m;
        phase_noise_rms_rad * self.wavelength_m / (2.0 * PI * d)
    }

    /// Required DAC resolution (bits) for a given pointing accuracy.
    ///
    /// B ≥ ceil( log₂( 2π / (δθ × 2π d / λ) ) )
    ///   = ceil( log₂( λ / (δθ d) ) )
    ///
    /// * `required_pointing_accuracy_rad` — maximum allowable 1-sigma pointing error
    pub fn required_dac_bits(&self, required_pointing_accuracy_rad: f64) -> u32 {
        if required_pointing_accuracy_rad < f64::MIN_POSITIVE {
            return 32;
        }
        let d = self.opa.pitch_m;
        let ratio = self.wavelength_m / (required_pointing_accuracy_rad * d);
        if ratio <= 1.0 {
            return 1;
        }
        ratio.log2().ceil() as u32
    }

    /// Maximum frame rate (fps) achievable given the phase update rate.
    ///
    ///   fps = phase_update_rate_hz / n_resolvable_points(fov_rad)
    ///
    /// * `phase_update_rate_hz` — how many phase settings the driver can apply per second
    /// * `fov_rad`              — total scan FOV (radians)
    pub fn scan_rate_fps(&self, phase_update_rate_hz: f64, fov_rad: f64) -> f64 {
        let n_pts = self.n_resolvable_points(fov_rad);
        if n_pts == 0 {
            return 0.0;
        }
        phase_update_rate_hz / n_pts as f64
    }

    /// Nyquist-limited 3-dB detection bandwidth for a pulse of duration τ.
    ///
    ///   BW ≈ 0.44 / τ_pulse  (Gaussian pulse bandwidth)
    pub fn receiver_bandwidth_hz(&self) -> f64 {
        if self.pulse_duration_s < f64::MIN_POSITIVE {
            return 0.0;
        }
        0.44 / self.pulse_duration_s
    }
}

// ─── SiliconOpa ──────────────────────────────────────────────────────────────

/// Silicon photonics chip-scale OPA specification.
///
/// Silicon-photonics OPAs (Sun et al. 2013; Poulton et al. 2019) achieve
/// 2D beam steering by combining:
/// - Phase-array control (x-axis steering via thermo-optic or EO phase shifters)
/// - Wavelength tuning (y-axis steering via grating emission dispersion)
#[derive(Debug, Clone)]
pub struct SiliconOpa {
    /// Number of emitter elements
    pub n_elements: usize,
    /// Total aperture width (m): W = N × d
    pub aperture_width_m: f64,
    /// Center wavelength (m) — typically 1310 nm or 1550 nm
    pub wavelength_m: f64,
    /// Wavelength tuning range (nm) for grating-assisted 2D scanning
    pub tuning_range_nm: f64,
    /// Half-wave voltage for electro-optic or thermo-optic phase shifter (V)
    pub phase_shifter_vpi: f64,
}

impl SiliconOpa {
    /// Element pitch: d = W / N
    pub fn element_pitch_m(&self) -> f64 {
        if self.n_elements == 0 {
            return 0.0;
        }
        self.aperture_width_m / self.n_elements as f64
    }

    /// HPBW in x (phase-array direction, radians):
    ///   Δθ_x ≈ 0.886 λ / W
    pub fn hpbw_x_rad(&self) -> f64 {
        if self.aperture_width_m < f64::MIN_POSITIVE {
            return PI;
        }
        0.886 * self.wavelength_m / self.aperture_width_m
    }

    /// Steering angle (radians) produced by wavelength shift Δλ.
    ///
    /// For a grating-coupled OPA with grating period Λ and fill angle θ_g:
    ///
    ///   dθ/dλ ≈ −1/Λ × 1/cos(θ_g)
    ///
    /// In the small-angle paraxial approximation (θ_g ≈ 0):
    ///
    ///   Δθ_y ≈ −Δλ / Λ ≈ −Δλ × n_g / λ²
    ///
    /// Here we use the simplified dispersive steering: Δθ ≈ Δλ/λ (1 rad per octave).
    ///
    /// * `delta_lambda_m` — wavelength change from the centre (m)
    pub fn wavelength_steering_angle_rad(&self, delta_lambda_m: f64) -> f64 {
        // Grating dispersion: dθ/dλ ≈ n_g / λ_center (group index n_g ≈ 4 for Si at 1550 nm)
        let n_group: f64 = 4.0; // Si photonic wire group index at 1550 nm
        (n_group / self.wavelength_m) * delta_lambda_m
    }

    /// Total 2D field of view: (FOV_x from phase steering, FOV_y from wavelength tuning).
    ///
    /// FOV_x ≈ 2 × arcsin(λ/(2d) − 1/N)   [phase-array scan, grating-lobe free]
    /// FOV_y ≈ Δθ from tuning_range_nm      [wavelength scan]
    ///
    /// Returns (fov_x_rad, fov_y_rad).
    pub fn total_fov_2d_rad2(&self) -> (f64, f64) {
        let d = self.element_pitch_m();
        // Phase-steering FOV (grating-lobe free)
        let fov_x = if d > f64::MIN_POSITIVE {
            let arg = self.wavelength_m / (2.0 * d) - 1.0 / self.n_elements.max(1) as f64;
            let half = if arg >= 1.0 {
                PI / 2.0
            } else if arg <= -1.0 {
                0.0
            } else {
                arg.asin()
            };
            2.0 * half
        } else {
            0.0
        };
        // Wavelength-steering FOV
        let delta_lambda = self.tuning_range_nm * 1.0e-9 / 2.0; // ±half range
        let fov_y = 2.0 * self.wavelength_steering_angle_rad(delta_lambda).abs();
        (fov_x, fov_y)
    }

    /// Power per element given a total optical power budget.
    pub fn power_per_element_w(&self, total_power_w: f64) -> f64 {
        if self.n_elements == 0 {
            return 0.0;
        }
        total_power_w / self.n_elements as f64
    }

    /// Number of resolvable points in 2D:
    ///
    ///   N_x × N_y = (FOV_x / HPBW_x) × (FOV_y / HPBW_y_approx)
    ///
    /// HPBW_y in the wavelength dimension ≈ dθ/dλ × δλ_min,
    /// where δλ_min ≈ wavelength_m² / (L_grating × n_g).
    ///
    /// For simplicity we use HPBW_y ≈ HPBW_x (square aperture approximation).
    pub fn n_resolvable_spots_2d(&self) -> usize {
        let (fov_x, fov_y) = self.total_fov_2d_rad2();
        let hpbw = self.hpbw_x_rad();
        if hpbw < f64::MIN_POSITIVE {
            return 0;
        }
        let nx = (fov_x / hpbw).round() as usize;
        let ny = (fov_y / hpbw).round() as usize;
        nx.max(1) * ny.max(1)
    }

    /// Drive voltage for a π phase shift (Vπ): this is simply `phase_shifter_vpi`.
    ///
    /// The required drive voltage for arbitrary phase φ:
    ///
    ///   V(φ) = Vπ × φ / π
    pub fn drive_voltage_for_phase(&self, phase_rad: f64) -> f64 {
        self.phase_shifter_vpi * phase_rad / PI
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::photonic_antenna::phased_array::OpticalPhasedArray1d;

    fn default_lidar() -> OpaLidar {
        OpaLidar {
            opa: OpticalPhasedArray1d::new(64, 775.0e-9, 1550.0e-9),
            pulse_energy_j: 1.0e-9,       // 1 nJ
            pulse_duration_s: 10.0e-9,    // 10 ns → 100 MHz BW
            receiver_aperture_m2: 1.0e-4, // 1 cm²
            wavelength_m: 1550.0e-9,
            n_bits_dac: 8,
        }
    }

    #[test]
    fn lidar_range_resolution_10ns() {
        let lidar = default_lidar();
        let dr = lidar.range_resolution_m();
        // c × 10 ns / 2 = 1.499 m
        assert!((dr - 1.499).abs() < 0.01, "Range resolution: {dr}m");
    }

    #[test]
    fn lidar_angular_resolution_positive() {
        let lidar = default_lidar();
        let ar = lidar.angular_resolution_rad();
        assert!(ar > 0.0, "Angular resolution must be positive: {ar}");
    }

    #[test]
    fn lidar_max_range_positive_for_reflective_target() {
        let lidar = default_lidar();
        let r_max = lidar.max_range_m(0.1, 100.0e6);
        assert!(r_max > 0.0, "Max range must be positive: {r_max}");
    }

    #[test]
    fn lidar_received_power_decreases_with_range() {
        let lidar = default_lidar();
        let p1 = lidar.received_power_w(10.0, 0.1);
        let p2 = lidar.received_power_w(100.0, 0.1);
        assert!(
            p1 > p2,
            "Power must decrease with range: P(10m)={p1}, P(100m)={p2}"
        );
    }

    #[test]
    fn lidar_pointing_error_proportional_to_phase_noise() {
        let lidar = default_lidar();
        let err1 = lidar.pointing_error_from_phase_noise_rad(0.1);
        let err2 = lidar.pointing_error_from_phase_noise_rad(0.2);
        assert!(
            (err2 / err1 - 2.0).abs() < 1.0e-10,
            "Pointing error must scale linearly with phase noise"
        );
    }

    #[test]
    fn lidar_required_dac_bits_reasonable() {
        let lidar = default_lidar();
        // 0.1 mrad pointing accuracy
        let bits = lidar.required_dac_bits(1.0e-4);
        assert!(
            (4..=16).contains(&bits),
            "DAC bits should be in [4, 16]: {bits}"
        );
    }

    #[test]
    fn silicon_opa_fov_positive() {
        let opa = SiliconOpa {
            n_elements: 512,
            aperture_width_m: 400.0e-6, // 400 µm
            wavelength_m: 1550.0e-9,
            tuning_range_nm: 100.0,
            phase_shifter_vpi: 3.0,
        };
        let (fov_x, fov_y) = opa.total_fov_2d_rad2();
        assert!(fov_x > 0.0, "FOV_x must be positive: {fov_x}");
        assert!(fov_y > 0.0, "FOV_y must be positive: {fov_y}");
    }

    #[test]
    fn silicon_opa_power_per_element_divides_correctly() {
        let opa = SiliconOpa {
            n_elements: 100,
            aperture_width_m: 100.0e-6,
            wavelength_m: 1550.0e-9,
            tuning_range_nm: 50.0,
            phase_shifter_vpi: 5.0,
        };
        let p = opa.power_per_element_w(10.0e-3); // 10 mW total
        assert!(
            (p - 0.1e-3).abs() < 1.0e-10,
            "Power per element must be 0.1 mW: {p}"
        );
    }
}
