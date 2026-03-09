//! LiDAR (Light Detection and Ranging) models.
//!
//! Covers direct time-of-flight (ToF), frequency-modulated continuous-wave (FMCW),
//! scanning 3-D point cloud, and photon-counting (SPAD / Geiger-mode) architectures.

use std::f64::consts::PI;

/// Speed of light in vacuum (m/s).
const C: f64 = 2.997_924_58e8;

// ---------------------------------------------------------------------------
// Direct Time-of-Flight LiDAR
// ---------------------------------------------------------------------------

/// Direct Time-of-Flight (ToF) LiDAR sensor model.
///
/// Models the link budget and performance metrics for a pulsed monostatic LiDAR
/// system operating in the near-infrared.
#[derive(Debug, Clone)]
pub struct TofLidar {
    /// Operating wavelength (m).
    pub wavelength: f64,
    /// Peak optical pulse power (W).
    pub peak_power_w: f64,
    /// Laser pulse width (ns).
    pub pulse_width_ns: f64,
    /// Pulse repetition rate (kHz).
    pub repetition_rate_khz: f64,
    /// Effective receiver aperture area (m²).
    pub receiver_aperture_m2: f64,
    /// Overall receiver quantum / optical efficiency (dimensionless, 0–1).
    pub detector_efficiency: f64,
    /// Noise-equivalent power of the detector (W/√Hz).
    pub noise_equivalent_power: f64,
}

impl TofLidar {
    /// Construct a 905 nm (silicon APD) ToF LiDAR with typical parameters.
    ///
    /// # Arguments
    /// * `peak_power_w` – Peak pulse power (W).
    /// * `aperture_m2` – Receiver aperture area (m²).
    pub fn new_905nm(peak_power_w: f64, aperture_m2: f64) -> Self {
        Self {
            wavelength: 905e-9,
            peak_power_w,
            pulse_width_ns: 5.0,
            repetition_rate_khz: 100.0,
            receiver_aperture_m2: aperture_m2,
            detector_efficiency: 0.70,
            noise_equivalent_power: 1e-12,
        }
    }

    /// Construct a 1550 nm (InGaAs APD) ToF LiDAR with typical parameters.
    ///
    /// # Arguments
    /// * `peak_power_w` – Peak pulse power (W).
    /// * `aperture_m2` – Receiver aperture area (m²).
    pub fn new_1550nm(peak_power_w: f64, aperture_m2: f64) -> Self {
        Self {
            wavelength: 1550e-9,
            peak_power_w,
            pulse_width_ns: 3.0,
            repetition_rate_khz: 75.0,
            receiver_aperture_m2: aperture_m2,
            detector_efficiency: 0.60,
            noise_equivalent_power: 5e-13,
        }
    }

    /// Compute target range from measured time-of-flight.
    ///
    /// `R = c * t / 2` (round-trip factor of 2).
    ///
    /// # Arguments
    /// * `tof_s` – Measured round-trip time (s).
    ///
    /// # Returns
    /// Target range in metres.
    pub fn range_from_tof(&self, tof_s: f64) -> f64 {
        C * tof_s / 2.0
    }

    /// Estimate the SNR-limited maximum range.
    ///
    /// Derived from the monostatic LiDAR range equation solved for SNR = 1:
    ///
    /// `R_max ≈ (P_t · η_r · A_r · ρ / (π · NEP²))^(1/4)`
    ///
    /// where the bandwidth is matched to the pulse width `BW = 1/(2·τ_pulse)`.
    ///
    /// # Arguments
    /// * `target_reflectivity` – Lambertian surface reflectivity (0–1).
    ///
    /// # Returns
    /// Maximum detectable range (m) for SNR = 1.
    pub fn max_range_m(&self, target_reflectivity: f64) -> f64 {
        let bw_hz = 1.0 / (2.0 * self.pulse_width_ns * 1e-9);
        let noise_power = self.noise_equivalent_power * bw_hz.sqrt();
        let numerator = self.peak_power_w
            * self.detector_efficiency
            * self.receiver_aperture_m2
            * target_reflectivity;
        let denominator = PI * noise_power * noise_power;
        if denominator <= 0.0 {
            return 0.0;
        }
        (numerator / denominator).powf(0.25)
    }

    /// Range resolution determined by the pulse width.
    ///
    /// `δR = c · τ_pulse / 2`
    ///
    /// # Returns
    /// Range resolution (m).
    pub fn range_resolution_m(&self) -> f64 {
        C * self.pulse_width_ns * 1e-9 / 2.0
    }

    /// Signal-to-noise ratio at a given range.
    ///
    /// `SNR = P_received / (NEP · √BW)`
    ///
    /// where `BW = 1/(2·τ_pulse)` (matched filter bandwidth).
    ///
    /// # Arguments
    /// * `range_m` – Target range (m).
    /// * `target_reflectivity` – Surface reflectivity (0–1).
    ///
    /// # Returns
    /// Dimensionless SNR (linear, not dB).
    pub fn snr_at_range(&self, range_m: f64, target_reflectivity: f64) -> f64 {
        if range_m <= 0.0 {
            return 0.0;
        }
        let p_rx = self.received_power_w(range_m, target_reflectivity, 1.0);
        let bw_hz = 1.0 / (2.0 * self.pulse_width_ns * 1e-9);
        let noise = self.noise_equivalent_power * bw_hz.sqrt();
        if noise <= 0.0 {
            return f64::INFINITY;
        }
        p_rx / noise
    }

    /// Received power using the monostatic LiDAR equation (Lambertian target).
    ///
    /// `P_r = P_t · η_t · A_r · ρ · T_atm² / (π · R²)`
    ///
    /// # Arguments
    /// * `range_m` – Target range (m).
    /// * `reflectivity` – Target surface reflectivity (0–1).
    /// * `transmittance` – One-way atmospheric transmittance (0–1).
    ///
    /// # Returns
    /// Received optical power (W).
    pub fn received_power_w(&self, range_m: f64, reflectivity: f64, transmittance: f64) -> f64 {
        if range_m <= 0.0 {
            return 0.0;
        }
        self.peak_power_w
            * self.detector_efficiency
            * self.receiver_aperture_m2
            * reflectivity
            * transmittance
            * transmittance
            / (PI * range_m * range_m)
    }

    /// Maximum unambiguous range limited by the pulse repetition frequency.
    ///
    /// `R_unamb = c / (2 · f_rep)`
    ///
    /// # Returns
    /// Maximum unambiguous range (m).
    pub fn unambiguous_range_m(&self) -> f64 {
        C / (2.0 * self.repetition_rate_khz * 1e3)
    }
}

// ---------------------------------------------------------------------------
// FMCW LiDAR
// ---------------------------------------------------------------------------

/// Frequency-Modulated Continuous-Wave (FMCW) LiDAR model.
///
/// Coherent ranging technique that recovers both range and velocity from a
/// linearly chirped continuous-wave optical signal.
#[derive(Debug, Clone)]
pub struct FmcwLidar {
    /// Centre optical wavelength (m).
    pub center_wavelength: f64,
    /// Linear chirp bandwidth (GHz).
    pub bandwidth_ghz: f64,
    /// Single chirp duration (μs).
    pub chirp_duration_us: f64,
    /// Transmit optical power (mW).
    pub transmit_power_mw: f64,
}

impl FmcwLidar {
    /// Construct an FMCW LiDAR system.
    ///
    /// # Arguments
    /// * `wavelength` – Centre wavelength (m).
    /// * `bandwidth_ghz` – Chirp bandwidth (GHz).
    /// * `chirp_us` – Chirp duration (μs).
    pub fn new(wavelength: f64, bandwidth_ghz: f64, chirp_us: f64) -> Self {
        Self {
            center_wavelength: wavelength,
            bandwidth_ghz,
            chirp_duration_us: chirp_us,
            transmit_power_mw: 10.0,
        }
    }

    /// Beat frequency produced by a stationary target at range *R*.
    ///
    /// `f_beat = 2 · R · B / (c · T_chirp)`
    ///
    /// # Arguments
    /// * `range_m` – Target range (m).
    ///
    /// # Returns
    /// Beat frequency (Hz).
    pub fn beat_frequency_hz(&self, range_m: f64) -> f64 {
        let bw_hz = self.bandwidth_ghz * 1e9;
        let t_chirp_s = self.chirp_duration_us * 1e-6;
        2.0 * range_m * bw_hz / (C * t_chirp_s)
    }

    /// Recover target range from a measured beat frequency.
    ///
    /// `R = f_beat · c · T / (2 · B)`
    ///
    /// # Arguments
    /// * `beat_freq_hz` – Measured beat frequency (Hz).
    ///
    /// # Returns
    /// Target range (m).
    pub fn range_from_beat(&self, beat_freq_hz: f64) -> f64 {
        let bw_hz = self.bandwidth_ghz * 1e9;
        let t_chirp_s = self.chirp_duration_us * 1e-6;
        beat_freq_hz * C * t_chirp_s / (2.0 * bw_hz)
    }

    /// Range resolution set by the chirp bandwidth.
    ///
    /// `δR = c / (2 · B)`
    ///
    /// # Returns
    /// Range resolution (m).
    pub fn range_resolution_m(&self) -> f64 {
        C / (2.0 * self.bandwidth_ghz * 1e9)
    }

    /// Target velocity from the Doppler frequency shift.
    ///
    /// `v = f_Doppler · λ / 2`
    ///
    /// # Arguments
    /// * `doppler_freq_hz` – Measured Doppler frequency (Hz).
    ///
    /// # Returns
    /// Radial velocity (m/s); positive = approaching.
    pub fn velocity_from_doppler(&self, doppler_freq_hz: f64) -> f64 {
        doppler_freq_hz * self.center_wavelength / 2.0
    }

    /// Velocity resolution within a single chirp interval.
    ///
    /// `δv = λ / (2 · T_chirp)`
    ///
    /// # Returns
    /// Velocity resolution (m/s).
    pub fn velocity_resolution_ms(&self) -> f64 {
        self.center_wavelength / (2.0 * self.chirp_duration_us * 1e-6)
    }

    /// Nyquist-limited maximum unambiguous range.
    ///
    /// `R_max = f_sample · c · T / (2 · B)`
    ///
    /// # Arguments
    /// * `sample_rate_gsps` – ADC sample rate (Gsps).
    ///
    /// # Returns
    /// Maximum unambiguous range (m).
    pub fn max_range_m(&self, sample_rate_gsps: f64) -> f64 {
        let f_nyq = sample_rate_gsps * 1e9 / 2.0;
        self.range_from_beat(f_nyq)
    }
}

// ---------------------------------------------------------------------------
// 3-D LiDAR Scanner
// ---------------------------------------------------------------------------

/// 3-D scanning LiDAR system model.
///
/// Wraps a [`TofLidar`] and models a mechanical or MEMS raster scan over a
/// specified field of view at a given angular resolution.
#[derive(Debug, Clone)]
pub struct LidarScanner {
    /// Underlying ToF sensor.
    pub sensor: TofLidar,
    /// Horizontal field of view (degrees).
    pub horizontal_fov_deg: f64,
    /// Vertical field of view (degrees).
    pub vertical_fov_deg: f64,
    /// Angular resolution in both axes (degrees per step).
    pub angular_resolution_deg: f64,
}

impl LidarScanner {
    /// Construct a 3-D LiDAR scanner.
    ///
    /// # Arguments
    /// * `sensor` – Underlying ToF LiDAR sensor.
    /// * `h_fov` – Horizontal field of view (degrees).
    /// * `v_fov` – Vertical field of view (degrees).
    /// * `resolution` – Angular step size (degrees).
    pub fn new(sensor: TofLidar, h_fov: f64, v_fov: f64, resolution: f64) -> Self {
        Self {
            sensor,
            horizontal_fov_deg: h_fov,
            vertical_fov_deg: v_fov,
            angular_resolution_deg: resolution,
        }
    }

    /// Number of vertical scan lines.
    pub fn n_scan_lines(&self) -> usize {
        if self.angular_resolution_deg <= 0.0 {
            return 0;
        }
        (self.vertical_fov_deg / self.angular_resolution_deg).ceil() as usize + 1
    }

    /// Number of points per horizontal scan line.
    pub fn points_per_line(&self) -> usize {
        if self.angular_resolution_deg <= 0.0 {
            return 0;
        }
        (self.horizontal_fov_deg / self.angular_resolution_deg).ceil() as usize + 1
    }

    /// Total number of 3-D points in one frame.
    pub fn total_points(&self) -> usize {
        self.n_scan_lines() * self.points_per_line()
    }

    /// Achievable frame rate limited by the pulse repetition rate.
    ///
    /// `f_frame = f_rep / N_total_points`
    ///
    /// # Returns
    /// Frame rate (Hz); returns 0 if no points.
    pub fn frame_rate_hz(&self) -> f64 {
        let n = self.total_points();
        if n == 0 {
            return 0.0;
        }
        self.sensor.repetition_rate_khz * 1e3 / n as f64
    }

    /// Convert spherical polar coordinates to Cartesian.
    ///
    /// Convention: azimuth measured from +x towards +y; elevation from the
    /// horizontal plane towards +z.
    ///
    /// # Arguments
    /// * `range` – Radial distance (m).
    /// * `azimuth_deg` – Azimuth angle (degrees).
    /// * `elevation_deg` – Elevation angle (degrees).
    ///
    /// # Returns
    /// `[x, y, z]` position (m).
    pub fn to_cartesian(&self, range: f64, azimuth_deg: f64, elevation_deg: f64) -> [f64; 3] {
        let az = azimuth_deg.to_radians();
        let el = elevation_deg.to_radians();
        let r_horiz = range * el.cos();
        [r_horiz * az.cos(), r_horiz * az.sin(), range * el.sin()]
    }
}

// ---------------------------------------------------------------------------
// Photon-Counting LiDAR (SPAD / Geiger-mode)
// ---------------------------------------------------------------------------

/// Photon-counting LiDAR based on single-photon avalanche diodes (SPADs).
///
/// Geiger-mode operation enables single-photon sensitivity but introduces
/// dead-time and dark count rate limitations.
#[derive(Debug, Clone)]
pub struct PhotonCountingLidar {
    /// Operating wavelength (m).
    pub wavelength: f64,
    /// Photon detection efficiency (PDE) — probability of detecting an
    /// incident photon (0–1).
    pub spad_efficiency: f64,
    /// Dark count rate (counts per second).
    pub dark_count_rate_cps: f64,
    /// Single-photon timing jitter (ps FWHM).
    pub timing_jitter_ps: f64,
    /// SPAD dead time after a detection event (ns).
    pub dead_time_ns: f64,
}

impl PhotonCountingLidar {
    /// Construct a photon-counting LiDAR with default SPAD parameters.
    ///
    /// # Arguments
    /// * `wavelength` – Operating wavelength (m).
    /// * `pde` – Photon detection efficiency (0–1).
    pub fn new(wavelength: f64, pde: f64) -> Self {
        Self {
            wavelength,
            spad_efficiency: pde,
            dark_count_rate_cps: 500.0,
            timing_jitter_ps: 50.0,
            dead_time_ns: 25.0,
        }
    }

    /// Signal-to-noise ratio for accumulated photon counts.
    ///
    /// Uses the Poisson shot-noise model:
    /// `SNR = S · √N_frames / √(S + B_dark)`
    ///
    /// where `S` = signal photons per frame, `B_dark` = background + dark
    /// counts per frame.
    ///
    /// # Arguments
    /// * `signal_photons` – Mean signal photons detected per frame.
    /// * `background_photons` – Mean background + dark photons per frame.
    /// * `n_frames` – Number of accumulated frames.
    ///
    /// # Returns
    /// Dimensionless SNR (linear).
    pub fn snr(&self, signal_photons: f64, background_photons: f64, n_frames: usize) -> f64 {
        let nf = n_frames as f64;
        let denominator = (signal_photons + background_photons).sqrt();
        if denominator <= 0.0 {
            return 0.0;
        }
        signal_photons * nf.sqrt() / denominator
    }

    /// Range precision limited by timing jitter and photon statistics.
    ///
    /// `σ_R = (c · σ_t) / 2 = c · FWHM_jitter / (2 · 2.355 · √N_photons)`
    ///
    /// # Arguments
    /// * `n_photons` – Number of detected photons accumulated.
    /// * `signal_rate` – Signal photon count rate (counts/s) — reserved for
    ///   saturation checks (currently unused in the formula).
    ///
    /// # Returns
    /// Range precision one-sigma (mm).
    pub fn range_precision_mm(&self, n_photons: usize, _signal_rate: f64) -> f64 {
        if n_photons == 0 {
            return f64::INFINITY;
        }
        let sigma_t_s = self.timing_jitter_ps * 1e-12 / (2.355 * (n_photons as f64).sqrt());
        C * sigma_t_s / 2.0 * 1e3
    }

    /// Estimate the maximum detectable range (SNR = 5 criterion).
    ///
    /// Derived from the single-shot received photon count using the LiDAR
    /// range equation then finding range where signal photons per shot = 5.
    ///
    /// # Arguments
    /// * `peak_power_w` – Source peak pulse power (W).
    /// * `aperture_m2` – Receiver aperture area (m²).
    /// * `reflectivity` – Target reflectivity (0–1).
    ///
    /// # Returns
    /// Maximum range (m) for 5 detected photons per pulse.
    pub fn max_range_m(&self, peak_power_w: f64, aperture_m2: f64, reflectivity: f64) -> f64 {
        // Planck constant (J·s)
        const H: f64 = 6.626_070_15e-34;
        // Photon energy
        let h_nu = H * C / self.wavelength;
        // Assume 5 ns pulse — generic default
        let pulse_width_s = 5e-9_f64;
        let pulse_energy_j = peak_power_w * pulse_width_s;
        let n_photons_emitted = pulse_energy_j / h_nu;
        // Solve P_rx*PDE/(hν) = 5  =>  R² = n_emitted*PDE*Ar*ρ/(5π)
        let target_detections = 5.0_f64;
        let r2 = n_photons_emitted * self.spad_efficiency * aperture_m2 * reflectivity
            / (target_detections * PI);
        if r2 <= 0.0 {
            return 0.0;
        }
        r2.sqrt()
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
    fn tof_range_from_tof_roundtrip() {
        let lidar = TofLidar::new_905nm(100.0, 1e-4);
        let range = 150.0_f64; // m
        let tof = 2.0 * range / C;
        assert_abs_diff_eq!(lidar.range_from_tof(tof), range, epsilon = 1e-9);
    }

    #[test]
    fn tof_range_resolution() {
        // 5 ns pulse → δR = 3e8 * 5e-9 / 2 = 0.75 m
        let lidar = TofLidar::new_905nm(100.0, 1e-4);
        assert_abs_diff_eq!(lidar.range_resolution_m(), 0.7494814500_f64, epsilon = 1e-3);
    }

    #[test]
    fn tof_unambiguous_range() {
        // 100 kHz → R_unamb = 3e8 / (2e5) = 1500 m
        let lidar = TofLidar::new_905nm(100.0, 1e-4);
        assert_abs_diff_eq!(lidar.unambiguous_range_m(), 1498.962_f64, epsilon = 1.0);
    }

    #[test]
    fn tof_received_power_decreases_with_range() {
        let lidar = TofLidar::new_1550nm(50.0, 5e-4);
        let p1 = lidar.received_power_w(100.0, 0.5, 0.9);
        let p2 = lidar.received_power_w(200.0, 0.5, 0.9);
        assert!(p1 > p2, "Power must decrease with range");
        // Inverse-square: p1/p2 should be ~4
        assert_abs_diff_eq!(p1 / p2, 4.0, epsilon = 0.05);
    }

    #[test]
    fn fmcw_range_beat_roundtrip() {
        let fmcw = FmcwLidar::new(1550e-9, 100.0, 100.0);
        let range = 50.0_f64;
        let f_beat = fmcw.beat_frequency_hz(range);
        assert_abs_diff_eq!(fmcw.range_from_beat(f_beat), range, epsilon = 1e-9);
    }

    #[test]
    fn fmcw_range_resolution() {
        // 100 GHz → δR = 3e8/(2e11) = 1.5 mm
        let fmcw = FmcwLidar::new(1550e-9, 100.0, 100.0);
        assert_abs_diff_eq!(fmcw.range_resolution_m(), 1.4987e-3, epsilon = 1e-5);
    }

    #[test]
    fn fmcw_velocity_from_doppler() {
        let fmcw = FmcwLidar::new(1550e-9, 100.0, 100.0);
        // 10 m/s → f_D = 2*10/1550e-9 = 12.9 MHz
        let v_in = 10.0_f64;
        let f_d = 2.0 * v_in / 1550e-9;
        assert_abs_diff_eq!(fmcw.velocity_from_doppler(f_d), v_in, epsilon = 1e-6);
    }

    #[test]
    fn scanner_total_points() {
        let sensor = TofLidar::new_905nm(100.0, 1e-4);
        let scanner = LidarScanner::new(sensor, 120.0, 30.0, 0.1);
        let pts = scanner.total_points();
        // Expect (1200+1)*(300+1) = 361 501 approx
        assert!(
            pts > 300_000 && pts < 400_000,
            "Unexpected point count: {}",
            pts
        );
    }

    #[test]
    fn scanner_cartesian_conversion() {
        let sensor = TofLidar::new_905nm(100.0, 1e-4);
        let scanner = LidarScanner::new(sensor, 90.0, 30.0, 0.5);
        let [x, y, z] = scanner.to_cartesian(10.0, 0.0, 0.0);
        // Straight ahead at 0° azimuth, 0° elevation
        assert_abs_diff_eq!(x, 10.0, epsilon = 1e-9);
        assert_abs_diff_eq!(y, 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(z, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn photon_counting_range_precision_decreases_with_photons() {
        let pc = PhotonCountingLidar::new(1550e-9, 0.3);
        let p1 = pc.range_precision_mm(10, 1e6);
        let p4 = pc.range_precision_mm(40, 1e6);
        // Precision improves as √N: p4 ≈ p1/2
        assert!(p4 < p1);
        assert_abs_diff_eq!(p1 / p4, 2.0_f64, epsilon = 0.1);
    }
}
