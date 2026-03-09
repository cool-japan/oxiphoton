pub mod dipole;
pub mod gaussian;
pub mod mode_source;
pub mod plane_wave;
pub mod tfsf;

pub use dipole::{
    dipole_radiated_power, dipole_radiation_pattern, DipoleArray3d, DipoleOrientation,
    DipoleOrientation3d, DipoleSrc, DipoleSrc3d, PurcellCalc,
};
pub use gaussian::{GaussianBeam3d, GaussianPulse};
pub use mode_source::{ModeProfile, ModeSource};
pub use plane_wave::{
    GaussianEnvelope3d, PlaneWaveSource, PlaneWaveSource3d, Polarization3d, PropagationAxis,
};
pub use tfsf::{TfsfAux1d, TfsfBox2d, TfsfConfig1d, TfsfSource3d};

use crate::units::Wavelength;

/// A time-domain source waveform
pub trait SourceWaveform: Send + Sync {
    /// Return field amplitude at time t (seconds)
    fn amplitude(&self, t: f64) -> f64;
}

/// Gaussian modulated pulse: envelope * cos(2*pi*f0*t)
///
/// E(t) = exp(-((t-t0)/tau)^2) * cos(2*pi*f0*(t-t0))
pub struct GaussianModulated {
    /// Center frequency (Hz)
    pub f0: f64,
    /// Time of peak (s)
    pub t0: f64,
    /// Gaussian width (s): sigma = tau / sqrt(2)
    pub tau: f64,
}

impl GaussianModulated {
    /// Create from center wavelength and desired bandwidth
    pub fn from_wavelength(center: Wavelength, bandwidth_factor: f64) -> Self {
        use crate::units::conversion::SPEED_OF_LIGHT;
        let f0 = SPEED_OF_LIGHT / center.0;
        let tau = bandwidth_factor / f0;
        let t0 = 3.0 * tau;
        Self { f0, t0, tau }
    }
}

impl SourceWaveform for GaussianModulated {
    fn amplitude(&self, t: f64) -> f64 {
        use std::f64::consts::PI;
        let dt = t - self.t0;
        let env = (-(dt / self.tau).powi(2)).exp();
        env * (2.0 * PI * self.f0 * dt).cos()
    }
}

/// Pure Gaussian envelope (for broadband excitation)
pub struct GaussianEnvelope {
    pub t0: f64,
    pub tau: f64,
}

impl GaussianEnvelope {
    pub fn new(t0: f64, tau: f64) -> Self {
        Self { t0, tau }
    }
}

impl SourceWaveform for GaussianEnvelope {
    fn amplitude(&self, t: f64) -> f64 {
        let dt = t - self.t0;
        (-(dt / self.tau).powi(2)).exp()
    }
}

/// Continuous wave (CW) source
pub struct ContinuousWave {
    pub f0: f64,
    pub phase: f64,
}

impl ContinuousWave {
    pub fn new(wavelength: Wavelength) -> Self {
        use crate::units::conversion::SPEED_OF_LIGHT;
        Self {
            f0: SPEED_OF_LIGHT / wavelength.0,
            phase: 0.0,
        }
    }
}

impl SourceWaveform for ContinuousWave {
    fn amplitude(&self, t: f64) -> f64 {
        use std::f64::consts::PI;
        (2.0 * PI * self.f0 * t + self.phase).sin()
    }
}
