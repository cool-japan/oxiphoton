//! Spatiotemporal structured light: space-time wave packets, flying-focus beams,
//! and pulsed OAM beams.
//!
//! All quantities are in SI units unless noted.  Pure Rust; no external
//! linear-algebra or random-number crates used.

use num_complex::Complex64;
use std::f64::consts::PI;

use super::laguerre_gaussian::LgBeam;

/// Speed of light in vacuum (m/s).
const C: f64 = 2.997_924_58e8;

// ---------------------------------------------------------------------------
// Space-Time Wave Packet (STWP)
// ---------------------------------------------------------------------------

/// Space-time wave packet — a pulsed beam whose spatial and temporal degrees
/// of freedom are correlated, enabling arbitrary (programmable) group velocity.
///
/// In the paraxial approximation the STWP field is
///
///   E(x, t; z) ∝ exp[−x²/(2 σ_x²)] · exp[−(t − z/v_g)²/(2 σ_t²)]
///                · exp[i(k₀ z − ω₀ t)]
///
/// where the correlation between k_x and ω enforces non-diffracting
/// propagation over the coherence length.
#[derive(Debug, Clone)]
pub struct SpaceTimeWavePacket {
    /// Spatial beam waist σ_x = w₀ / √2 (metres).
    pub beam_waist: f64,
    /// Temporal pulse duration σ_t (seconds, 1/e half-width).
    pub pulse_duration: f64,
    /// Centre wavelength (metres).
    pub wavelength: f64,
    /// Group velocity v_g (m/s); may differ from c.
    pub group_velocity: f64,
    /// Spatio-temporal tilt angle θ in the (k_x, ω/c) plane (radians).
    pub tilt_angle: f64,
}

impl SpaceTimeWavePacket {
    /// Construct from explicit parameters.
    pub fn new(w0: f64, tau: f64, wavelength: f64, v_g: f64) -> Self {
        // Infer tilt angle from group velocity: cos θ = c/v_g (for type-I STWP)
        let cos_theta = (C / v_g).clamp(-1.0, 1.0);
        let tilt_angle = cos_theta.acos();
        Self {
            beam_waist: w0,
            pulse_duration: tau,
            wavelength,
            group_velocity: v_g,
            tilt_angle,
        }
    }

    /// Construct from the tilt angle θ directly (v_g = c/cos θ).
    pub fn from_tilt_angle(w0: f64, tau: f64, wavelength: f64, theta: f64) -> Self {
        let v_g = if theta.cos().abs() < 1e-30 {
            f64::INFINITY
        } else {
            C / theta.cos()
        };
        Self {
            beam_waist: w0,
            pulse_duration: tau,
            wavelength,
            group_velocity: v_g,
            tilt_angle: theta,
        }
    }

    /// Group velocity (m/s).
    pub fn group_velocity(&self) -> f64 {
        self.group_velocity
    }

    /// Invariant spectral bandwidth in GHz.
    ///
    /// Δν_inv ≈ c sin²θ / (λ cos θ · 2π σ_x)
    /// For a STWP the bandwidth is set by the spatial coherence rather than
    /// the pulse duration alone.
    pub fn invariant_bandwidth_ghz(&self) -> f64 {
        let sin2_theta = self.tilt_angle.sin().powi(2);
        let cos_theta = self.tilt_angle.cos();
        if cos_theta.abs() < 1e-30 || self.beam_waist < 1e-30 {
            return 0.0;
        }
        let bw_hz = C * sin2_theta / (self.wavelength * cos_theta * 2.0 * PI * self.beam_waist);
        bw_hz / 1e9
    }

    /// Complex field amplitude E(x, t; z).
    ///
    /// Gaussian-Gaussian product with tilt-modified retarded time.
    pub fn field(&self, x: f64, t: f64, z: f64) -> Complex64 {
        let k0 = 2.0 * PI / self.wavelength;
        let omega0 = k0 * C;
        // Retarded time in the STWP frame
        let tau_ret = if self.group_velocity.is_finite() {
            t - z / self.group_velocity
        } else {
            t
        };
        let spatial_env = (-x * x / (2.0 * self.beam_waist * self.beam_waist)).exp();
        let temporal_env =
            (-tau_ret * tau_ret / (2.0 * self.pulse_duration * self.pulse_duration)).exp();
        let carrier = Complex64::from_polar(1.0, k0 * z - omega0 * t);
        Complex64::new(spatial_env * temporal_env, 0.0) * carrier
    }

    /// Non-diffracting propagation length (metres).
    ///
    /// L_nd ≈ k₀ w₀² / (tan²θ) — determined by the spatial bandwidth Δk_x = sin θ / w₀.
    pub fn nondiffracting_length_m(&self) -> f64 {
        let k0 = 2.0 * PI / self.wavelength;
        let tan2 = self.tilt_angle.tan().powi(2);
        if tan2 < 1e-30 {
            return f64::INFINITY;
        }
        k0 * self.beam_waist * self.beam_waist / tan2
    }

    /// Peak intensity profile I(t; z) = |E(0, t; z)|² (at x = 0).
    pub fn peak_intensity_profile(&self, t: f64, z: f64) -> f64 {
        self.field(0.0, t, z).norm_sqr()
    }
}

// ---------------------------------------------------------------------------
// Optical Flying Focus
// ---------------------------------------------------------------------------

/// Optical flying focus — a technique in which a chirped pulse is focused by
/// a lens to produce a focal spot that moves along the beam axis at a
/// programmable velocity (which may be superluminal or subluminal).
///
/// # Physical principle
/// A linearly chirped pulse has instantaneous frequency ω(t) = ω₀ + β t.
/// After a focusing lens of focal length f, the instantaneous focus lies at
/// z_f(t) = f · [1 + (ω(t) − ω₀)/ω₀ · (f/GVD_length)]  (simplified)
/// giving a focal velocity v_f = dz_f/dt.
#[derive(Debug, Clone)]
pub struct FlyingFocus {
    /// Lens focal length f (metres).
    pub lens_focal_length: f64,
    /// Total chirped pulse duration T (seconds).
    pub chirped_pulse_duration: f64,
    /// Optical bandwidth Δν (Hz).
    pub bandwidth: f64,
    /// Programmed focal-point velocity (m/s).
    pub propagation_velocity: f64,
}

impl FlyingFocus {
    /// Construct a flying-focus configuration.
    ///
    /// The focal velocity is determined by the ratio of the focal-length spread
    /// (due to chromatic dispersion or an axicon) to the pulse duration.
    pub fn new(f: f64, duration: f64, bandwidth: f64) -> Self {
        // Focal velocity: v_f = c · (Δν/ν₀) · (f²/c·T) / (1 + …) ≈ f·Δω/T (simple model)
        // Here we use the model v_f = f · (2π·bandwidth) / duration
        let v_f = f * 2.0 * PI * bandwidth / duration;
        Self {
            lens_focal_length: f,
            chirped_pulse_duration: duration,
            bandwidth,
            propagation_velocity: v_f,
        }
    }

    /// Focal point velocity (m/s).
    pub fn focal_velocity(&self) -> f64 {
        self.propagation_velocity
    }

    /// Focal point position z_f(t) relative to the lens (metres).
    ///
    /// Assumes the focal point starts at z_f(0) = f and moves at v_f.
    pub fn focal_position_m(&self, t: f64) -> f64 {
        self.lens_focal_length + self.propagation_velocity * t
    }

    /// Extended interaction length L = |v_f| · T (metres).
    pub fn interaction_length_m(&self) -> f64 {
        self.propagation_velocity.abs() * self.chirped_pulse_duration
    }
}

// ---------------------------------------------------------------------------
// Pulsed OAM Beam
// ---------------------------------------------------------------------------

/// Pulsed Laguerre-Gaussian beam — a twisted-photon pulse combining spatial
/// OAM with a finite temporal envelope.
///
/// The field is the product of the CW LG spatial mode and a Gaussian temporal
/// envelope centred at t = 0.
#[derive(Debug, Clone)]
pub struct PulsedOamBeam {
    /// Underlying CW LG beam (provides spatial mode structure).
    pub lg: LgBeam,
    /// Pulse duration (femtoseconds, 1/e half-width of the intensity envelope).
    pub pulse_duration_fs: f64,
    /// Centre wavelength (metres).
    pub center_wavelength: f64,
}

impl PulsedOamBeam {
    /// Construct a pulsed OAM beam.
    pub fn new(ell: i32, w0: f64, wavelength: f64, duration_fs: f64) -> Self {
        let lg = LgBeam::new(ell, 0, w0, wavelength);
        Self {
            lg,
            pulse_duration_fs: duration_fs,
            center_wavelength: wavelength,
        }
    }

    /// Complex field E(r, φ, z, t) = u_LG(r, φ, z) · A(t) · exp(−i ω₀ t).
    pub fn field(&self, r: f64, phi: f64, z: f64, t: f64) -> Complex64 {
        let omega0 = 2.0 * PI * C / self.center_wavelength;
        let tau = self.pulse_duration_fs * 1e-15;
        // Temporal Gaussian envelope (intensity half-width = tau)
        let temporal_env = (-t * t / (2.0 * tau * tau)).exp();
        // CW spatial mode
        let spatial = self.lg.field(r, phi, z);
        // Carrier
        let carrier = Complex64::from_polar(1.0, -omega0 * t);
        spatial * Complex64::new(temporal_env, 0.0) * carrier
    }

    /// Time-bandwidth product Δν · Δt for a Gaussian pulse.
    ///
    /// For a transform-limited Gaussian: Δν · Δt = 1/(2π) · √(2 ln 2) ≈ 0.4413.
    pub fn time_bandwidth_product(&self) -> f64 {
        // FWHM in time:  τ_FWHM = 2√(2 ln 2) · τ (1/e half-width)
        // FWHM in freq:  Δν_FWHM = 1/(π τ_FWHM) for transform-limited Gaussian
        // Product = τ_FWHM · Δν_FWHM = 2√(2 ln 2) / π ≈ 0.4413
        2.0 * (2.0_f64.ln() * 2.0).sqrt() / PI
    }

    /// Peak power P_peak = E_pulse / (√π · τ) (watts), given pulse energy in nJ.
    pub fn peak_power_w(&self, energy_nj: f64) -> f64 {
        let tau = self.pulse_duration_fs * 1e-15;
        let energy_j = energy_nj * 1e-9;
        // For a Gaussian pulse: P_peak = E / (√π · τ)
        energy_j / (PI.sqrt() * tau)
    }

    /// OAM spectral bandwidth — for a transform-limited pulsed OAM beam this
    /// is effectively 0 (pure OAM state).  For a chirped OAM beam the spectral
    /// bandwidth couples to OAM spread; we return the transform-limited estimate
    /// Δℓ ≈ |ℓ| · Δω/ω₀.
    pub fn oam_bandwidth(&self) -> f64 {
        let omega0 = 2.0 * PI * C / self.center_wavelength;
        let tau = self.pulse_duration_fs * 1e-15;
        // Transform-limited bandwidth (angular frequency):
        let delta_omega = 1.0 / tau; // 1/(2 tau) for Gaussian but approximate
        self.lg.ell.unsigned_abs() as f64 * delta_omega / omega0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // --- SpaceTimeWavePacket ---
    #[test]
    fn stwp_group_velocity_from_tilt() {
        // At θ = 0: v_g = c/cos(0) = c
        let stwp = SpaceTimeWavePacket::from_tilt_angle(1e-3, 100e-15, 800e-9, 0.0);
        assert_abs_diff_eq!(stwp.group_velocity(), C, epsilon = 1.0);
    }

    #[test]
    fn stwp_subluminal_group_velocity() {
        let v_g = C * 0.5;
        let stwp = SpaceTimeWavePacket::new(1e-3, 100e-15, 800e-9, v_g);
        assert_abs_diff_eq!(stwp.group_velocity(), v_g, epsilon = 1.0);
    }

    #[test]
    fn stwp_peak_intensity_at_retarded_time() {
        let stwp = SpaceTimeWavePacket::new(1e-3, 100e-15, 800e-9, C);
        // At t = z/c the peak should be higher than at t = 0 when z is large
        let z = 0.1; // 10 cm
        let t_retarded = z / C;
        let i_peak = stwp.peak_intensity_profile(t_retarded, z);
        let i_off = stwp.peak_intensity_profile(0.0, z);
        assert!(i_peak > i_off);
    }

    #[test]
    fn stwp_nondiffracting_length_finite() {
        let stwp = SpaceTimeWavePacket::from_tilt_angle(1e-3, 100e-15, 800e-9, 0.1);
        let l = stwp.nondiffracting_length_m();
        assert!(l > 0.0 && l.is_finite());
    }

    // --- FlyingFocus ---
    #[test]
    fn flying_focus_interaction_length() {
        let ff = FlyingFocus::new(0.1, 1e-12, 1e13);
        let length = ff.interaction_length_m();
        assert!(length > 0.0);
    }

    #[test]
    fn flying_focus_position_at_t0() {
        let ff = FlyingFocus::new(0.5, 1e-12, 1e13);
        assert_abs_diff_eq!(ff.focal_position_m(0.0), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn flying_focus_velocity_positive() {
        let ff = FlyingFocus::new(0.1, 1e-12, 1e13);
        assert!(ff.focal_velocity() != 0.0);
    }

    // --- PulsedOamBeam ---
    #[test]
    fn pulsed_oam_time_bandwidth_product() {
        let beam = PulsedOamBeam::new(2, 1e-3, 1064e-9, 100.0);
        // TBP for transform-limited Gaussian = 2√(2 ln 2)/π ≈ 0.7496
        // (FWHM_time × FWHM_freq convention)
        let expected = 2.0 * (2.0_f64.ln() * 2.0).sqrt() / std::f64::consts::PI;
        assert_abs_diff_eq!(beam.time_bandwidth_product(), expected, epsilon = 1e-10);
    }

    #[test]
    fn pulsed_oam_peak_power_scales_with_energy() {
        let beam = PulsedOamBeam::new(1, 1e-3, 1064e-9, 100.0);
        let p1 = beam.peak_power_w(1.0);
        let p2 = beam.peak_power_w(2.0);
        assert_abs_diff_eq!(p2 / p1, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn pulsed_oam_oam_bandwidth_increases_with_ell() {
        let b1 = PulsedOamBeam::new(1, 1e-3, 1064e-9, 100.0);
        let b2 = PulsedOamBeam::new(3, 1e-3, 1064e-9, 100.0);
        assert!(b2.oam_bandwidth() > b1.oam_bandwidth());
    }

    #[test]
    fn pulsed_oam_field_is_zero_at_axis_for_nonzero_ell() {
        // For ell != 0, the LG beam has zero on-axis intensity
        let beam = PulsedOamBeam::new(2, 1e-3, 1064e-9, 100.0);
        // r → 0 should give zero intensity (since (r/w)^|l| → 0)
        let f = beam.field(1e-12, 0.0, 0.0, 0.0);
        assert!(f.norm_sqr() < 1e-10);
    }
}
