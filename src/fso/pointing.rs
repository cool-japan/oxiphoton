//! Beam pointing, acquisition, and satellite optical link geometry.
//!
//! Models the pointing loss due to platform jitter, mean pointing error, and the
//! closed-loop tracking performance of an optical terminal.  Also provides
//! satellite-link helpers (slant range, Doppler shift, availability).
//!
//! # Physical Background
//!
//! For a Gaussian beam with 1/e² half-angle θ_div, the on-axis intensity falls as:
//!   I(θ_e) / I(0) = exp(−2 θ_e² / θ_div²)
//!
//! Pointing loss: L_pt (dB) = −10 log₁₀[exp(−2 θ_e² / θ_div²)]
//!                          = 8.686 θ_e² / θ_div²
//!
//! where θ_e is the effective RMS pointing error angle.
//!
//! # References
//! - Arnon, "Pointing, acquisition and tracking" in Optical Wireless Comm., 2012
//! - Toyoshima et al., "Ground-to-LEO optical link experiments", IEEE JLT 2005

use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// PointingSystem
// ─────────────────────────────────────────────────────────────────────────────

/// Optical antenna pointing and tracking system.
#[derive(Debug, Clone)]
pub struct PointingSystem {
    /// Platform mechanical jitter, RMS per-axis (µrad).
    pub platform_jitter_urad_rms: f64,
    /// Closed-loop tracking bandwidth (Hz).
    pub tracking_bandwidth_hz: f64,
    /// Residual tracking error after closed-loop (µrad RMS).
    pub tracking_residual_urad: f64,
    /// Control loop latency (ms).
    pub tracking_delay_ms: f64,
    /// Link distance (km).
    pub link_distance_km: f64,
}

impl PointingSystem {
    /// Construct a new pointing system.
    ///
    /// # Arguments
    /// * `jitter_urad` — platform jitter RMS (µrad)
    /// * `bw_hz` — tracking loop bandwidth (Hz)
    /// * `delay_ms` — loop latency (ms)
    /// * `dist_km` — link distance (km)
    pub fn new(jitter_urad: f64, bw_hz: f64, delay_ms: f64, dist_km: f64) -> Self {
        // Residual tracking error approximation using Wiener-Hopf (open-loop jitter,
        // closed-loop bandwidth reduction):  σ_res ≈ σ_jitter * √(f_n / BW) * BW correction
        // For a simple first-order integrator: σ_res ≈ σ_jitter * (f_c / BW)^{0.5}
        // Here we use a simple conservative estimate: residual ≈ jitter / √(BW/10)
        let bw = bw_hz.max(0.1);
        let residual = jitter_urad / (bw / 10.0).sqrt();
        Self {
            platform_jitter_urad_rms: jitter_urad,
            tracking_bandwidth_hz: bw,
            tracking_residual_urad: residual,
            tracking_delay_ms: delay_ms.max(0.0),
            link_distance_km: dist_km.max(0.001),
        }
    }

    /// Effective RMS pointing error angle (µrad), combining jitter and residual.
    ///
    /// σ_eff = √(σ_jitter² − σ_corrected²) where σ_corrected² = σ_jitter² − σ_res²
    /// simplifies to σ_eff = σ_res (residual after tracking).
    pub fn effective_jitter_urad(&self) -> f64 {
        // Total pointing error = quadrature sum of residual + delay-induced error
        let delay_error_urad = self.platform_jitter_urad_rms
            * (2.0 * PI * self.tracking_bandwidth_hz * self.tracking_delay_ms * 1e-3).min(1.0);
        (self.tracking_residual_urad.powi(2) + delay_error_urad.powi(2)).sqrt()
    }

    /// Pointing loss in dB for a Gaussian beam with given full-angle divergence.
    ///
    /// L_pt = 8.686 (θ_e / θ_half)² dB
    /// where θ_half = θ_div / 2 is the half-angle (1/e² intensity radius).
    pub fn pointing_loss_db(&self, beam_divergence_mrad: f64) -> f64 {
        let theta_e_urad = self.effective_jitter_urad();
        let theta_half_urad = beam_divergence_mrad * 1e3 / 2.0; // half angle in µrad
        if theta_half_urad <= 0.0 {
            return f64::INFINITY;
        }
        8.686 * (theta_e_urad / theta_half_urad).powi(2)
    }

    /// Power penalty from pointing error (same as pointing_loss_db, alias).
    pub fn power_penalty_db(&self, beam_divergence_mrad: f64) -> f64 {
        self.pointing_loss_db(beam_divergence_mrad)
    }

    /// Probability of initial acquisition within a given field of regard in `scan_time_s`.
    ///
    /// Uses a simplified model: the beam scans the uncertainty cone at `update_rate_hz`,
    /// and the probability that at least one dwell hits the target follows a geometric
    /// distribution.  The per-dwell hit probability is estimated from the beam solid
    /// angle vs. the field-of-regard solid angle.
    pub fn acquisition_probability(&self, field_of_regard_mrad: f64, scan_time_s: f64) -> f64 {
        // Number of positions scanned in scan_time_s at update rate
        let n_dwells = (scan_time_s * self.tracking_bandwidth_hz) as usize;
        if n_dwells == 0 {
            return 0.0;
        }
        // Beam solid angle / FoR solid angle: probability of hitting target per dwell
        let theta_beam_urad = self.effective_jitter_urad().max(1.0);
        let theta_for_urad = field_of_regard_mrad * 1e3;
        if theta_for_urad <= 0.0 {
            return 0.0;
        }
        let p_hit = (theta_beam_urad / theta_for_urad).powi(2).min(1.0);
        // P(acquire) = 1 - (1 - p_hit)^n_dwells
        1.0 - (1.0 - p_hit).powi(n_dwells as i32)
    }

    /// Mean acquisition time (seconds) given pointing uncertainty (mrad).
    ///
    /// Mean time = 1 / (update_rate * p_hit) seconds.
    pub fn mean_acquisition_time_s(&self, uncertainty_mrad: f64) -> f64 {
        let theta_beam_urad = self.effective_jitter_urad().max(1.0);
        let theta_unc_urad = uncertainty_mrad * 1e3;
        if theta_unc_urad <= 0.0 || theta_beam_urad <= 0.0 {
            return f64::INFINITY;
        }
        let p_hit = (theta_beam_urad / theta_unc_urad).powi(2).min(1.0);
        let bw = self.tracking_bandwidth_hz.max(1.0);
        1.0 / (bw * p_hit)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SatelliteOpticalLink
// ─────────────────────────────────────────────────────────────────────────────

/// Satellite-to-ground (or inter-satellite) optical link geometry and performance.
#[derive(Debug, Clone)]
pub struct SatelliteOpticalLink {
    /// Orbit altitude above sea level (km).
    pub orbit_altitude_km: f64,
    /// Satellite orbital speed (km/s).
    pub satellite_speed_km_s: f64,
    /// Pointing and tracking system at the ground terminal.
    pub pointing_system: PointingSystem,
    /// Optical wavelength (m).
    pub wavelength: f64,
    /// Transmitter aperture diameter (m).
    pub tx_aperture_m: f64,
}

impl SatelliteOpticalLink {
    /// Create a standard LEO satellite optical link (ISS-like, ~550 km, ~7.6 km/s).
    pub fn new_leo(altitude_km: f64) -> Self {
        // Orbital speed: v = √(GM_earth / r);  r = (R_earth + h)
        let r_earth_km = 6371.0;
        let gm: f64 = 3.986_004_418e5; // km³/s²
        let r = r_earth_km + altitude_km;
        let v_km_s = (gm / r).sqrt();
        let pointing = PointingSystem::new(5.0, 1000.0, 1.0, altitude_km);
        Self {
            orbit_altitude_km: altitude_km,
            satellite_speed_km_s: v_km_s,
            pointing_system: pointing,
            wavelength: 1550e-9,
            tx_aperture_m: 0.1,
        }
    }

    /// Create a standard GEO satellite optical link (~35 786 km, 3.07 km/s).
    pub fn new_geo() -> Self {
        let altitude_km = 35_786.0_f64;
        let r_earth_km = 6371.0_f64;
        let gm: f64 = 3.986_004_418e5;
        let r = r_earth_km + altitude_km;
        let v_km_s = (gm / r).sqrt();
        let pointing = PointingSystem::new(1.0, 100.0, 5.0, altitude_km);
        Self {
            orbit_altitude_km: altitude_km,
            satellite_speed_km_s: v_km_s,
            pointing_system: pointing,
            wavelength: 1550e-9,
            tx_aperture_m: 0.25,
        }
    }

    /// Maximum Doppler frequency shift Δf = v_rel * f / c (MHz).
    ///
    /// Maximum relative velocity is the satellite speed projected along the
    /// line of sight (maximum at horizon pass: ≈ satellite speed).
    pub fn max_doppler_shift_mhz(&self) -> f64 {
        let c = 2.997_924_58e8; // m/s
        let v_m_s = self.satellite_speed_km_s * 1e3;
        let freq = c / self.wavelength;
        v_m_s / c * freq * 1e-6 // MHz
    }

    /// Round-trip light time (ms) at zenith (altitude = link distance).
    pub fn round_trip_time_ms(&self) -> f64 {
        let c_km_s = 2.997_924_58e5; // km/s
        2.0 * self.orbit_altitude_km / c_km_s * 1e3 // ms
    }

    /// Instantaneous ground-track speed at the ground (km/s).
    ///
    /// v_ground = v_orbit * R_earth / (R_earth + h)
    pub fn ground_track_velocity_km_s(&self) -> f64 {
        let r_earth = 6371.0;
        let h = self.orbit_altitude_km;
        self.satellite_speed_km_s * r_earth / (r_earth + h)
    }

    /// Slant-range (km) from ground to satellite at elevation angle θ_el (degrees).
    ///
    /// Using spherical Earth geometry:
    /// R = −R_earth sin(θ_el) + √[(R_earth + h)² − R_earth² cos²(θ_el)]
    pub fn path_length_km(&self, elevation_deg: f64) -> f64 {
        let theta = elevation_deg.to_radians();
        let r_e = 6371.0;
        let h = self.orbit_altitude_km;
        let r_sat = r_e + h;
        let cos_theta = theta.cos();
        let discriminant = r_sat * r_sat - r_e * r_e * cos_theta * cos_theta;
        if discriminant < 0.0 {
            return 0.0; // below horizon
        }
        -r_e * theta.sin() + discriminant.sqrt()
    }

    /// Required pointing accuracy: σ < θ_div / 10 (µrad).
    ///
    /// θ_div = 2.44 λ / D_tx (diffraction-limited half-angle).
    pub fn required_pointing_accuracy_urad(&self) -> f64 {
        let theta_div_rad = 2.44 * self.wavelength / self.tx_aperture_m;
        theta_div_rad * 1e6 / 10.0 // µrad; /10 for 3-sigma budget
    }

    /// Link availability (fraction of time link operates) considering cloud cover.
    ///
    /// Simplified: availability = (1 − cloud_cover)^k where k accounts for
    /// atmospheric correlation over the beam; k = 1.2 for typical ground station.
    pub fn availability_percent(&self, cloud_cover_fraction: f64) -> f64 {
        let cc = cloud_cover_fraction.clamp(0.0, 1.0);
        // Exposure benefit from spatial diversity / site selection: exponent < 1
        100.0 * (1.0 - cc).powf(1.2)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_pointing() -> PointingSystem {
        PointingSystem::new(10.0, 500.0, 2.0, 5.0)
    }

    /// Pointing loss must be non-negative.
    #[test]
    fn test_pointing_loss_non_negative() {
        let ps = default_pointing();
        assert!(ps.pointing_loss_db(1.0) >= 0.0);
    }

    /// Larger jitter → larger pointing loss.
    #[test]
    fn test_pointing_loss_increases_with_jitter() {
        let ps_small = PointingSystem::new(1.0, 500.0, 2.0, 5.0);
        let ps_large = PointingSystem::new(20.0, 500.0, 2.0, 5.0);
        assert!(
            ps_large.pointing_loss_db(0.5) > ps_small.pointing_loss_db(0.5),
            "Larger jitter must give more pointing loss"
        );
    }

    /// GEO slant range at zenith (90°) should equal orbital altitude.
    #[test]
    fn test_geo_slant_range_zenith() {
        let geo = SatelliteOpticalLink::new_geo();
        let r = geo.path_length_km(90.0);
        let expected = geo.orbit_altitude_km;
        assert!(
            (r - expected).abs() < 50.0,
            "GEO zenith slant range = {r:.0} km (expected ≈ {expected:.0} km)"
        );
    }

    /// LEO orbital speed should be ~7.5–8.0 km/s at 550 km.
    #[test]
    fn test_leo_orbital_speed() {
        let leo = SatelliteOpticalLink::new_leo(550.0);
        let v = leo.satellite_speed_km_s;
        assert!(v > 7.0 && v < 8.5, "LEO v = {v:.3} km/s (expected 7.0–8.5)");
    }

    /// Doppler shift at 1550 nm for LEO: should be in GHz range.
    #[test]
    fn test_doppler_shift_leo() {
        let leo = SatelliteOpticalLink::new_leo(550.0);
        let df = leo.max_doppler_shift_mhz();
        // 7.6 km/s / 3e8 m/s * (3e8/1550e-9) = ~4900 MHz ≈ 4.9 GHz
        assert!(df > 1000.0, "Doppler = {df:.0} MHz (expected > 1000 MHz)");
    }

    /// Acquisition probability increases with scan time.
    #[test]
    fn test_acquisition_prob_increases_with_time() {
        let ps = default_pointing();
        let p1 = ps.acquisition_probability(5.0, 0.1);
        let p10 = ps.acquisition_probability(5.0, 10.0);
        assert!(p10 >= p1, "p10={p10:.4} p1={p1:.4}");
    }

    /// Link availability decreases with cloud cover.
    #[test]
    fn test_availability_decreases_with_clouds() {
        let leo = SatelliteOpticalLink::new_leo(550.0);
        let avail_clear = leo.availability_percent(0.0);
        let avail_overcast = leo.availability_percent(0.9);
        assert!(avail_clear > avail_overcast);
        assert!((avail_clear - 100.0).abs() < 1e-6);
    }

    /// Slant range is longer at low elevation angles than at zenith.
    #[test]
    fn test_slant_range_elevation_dependence() {
        let leo = SatelliteOpticalLink::new_leo(550.0);
        let r_zenith = leo.path_length_km(90.0);
        let r_10deg = leo.path_length_km(10.0);
        assert!(
            r_10deg > r_zenith,
            "Low elevation must give longer slant range"
        );
    }

    /// Required pointing accuracy for 10 cm aperture at 1550 nm: ~3.8 µrad.
    #[test]
    fn test_required_pointing_accuracy() {
        let leo = SatelliteOpticalLink::new_leo(550.0);
        // θ_div = 2.44 * 1550e-9 / 0.1 = 37.78 µrad; /10 = 3.78 µrad
        let acc = leo.required_pointing_accuracy_urad();
        assert!(acc > 1.0 && acc < 10.0, "Accuracy = {acc:.2} µrad");
    }
}
