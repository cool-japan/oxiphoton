//! X-ray & EUV Optics Module
//!
//! Provides simulation tools for X-ray and extreme ultraviolet (EUV) optics:
//!
//! - **Fresnel Zone Plates** (FZP), Compound Refractive Lenses (CRL), and
//!   Kirkpatrick–Baez (KB) mirror systems for X-ray focusing.
//! - **Multilayer mirrors** for EUV (13.5 nm) and hard X-ray reflective optics.
//! - **Bragg diffraction** from crystal analysers, Darwin widths, and Johann
//!   spectrometer geometry.
//! - **Synchrotron radiation** characteristics, undulator/wiggler insertion
//!   devices, and free-electron laser (FEL) fundamentals.
//!
//! # Wavelength / energy conventions
//! All lengths are in metres (SI).  Helper constants convert between photon
//! energy in keV and wavelength:
//! ```text
//! λ [m] = hc / E = 1.23984193e-9 / E[keV]
//! ```

pub mod bragg;
pub mod fresnel_zone_plate;
pub mod multilayer_mirror;
pub mod synchrotron;

pub use bragg::*;
pub use fresnel_zone_plate::*;
pub use multilayer_mirror::*;
pub use synchrotron::*;

// ─── Shared physical constants ─────────────────────────────────────────────

/// Speed of light in vacuum (m s⁻¹).
pub const C0: f64 = 2.997_924_58e8;

/// Planck constant (J s).
pub const H_PLANCK: f64 = 6.626_070_15e-34;

/// Electron rest energy (J): m_e c².
pub const ME_C2_J: f64 = 8.187_105_77e-14;

/// Classical electron radius (m).
pub const R_ELECTRON: f64 = 2.817_940_3e-15;

/// hc product (m·J) used for λ–E conversion.
pub const HC_J_M: f64 = H_PLANCK * C0; // ≈ 1.98644568e-25

/// Convert photon energy in keV to wavelength in metres.
#[inline]
pub fn kev_to_wavelength_m(energy_kev: f64) -> f64 {
    // E [J] = energy_kev * 1e3 * 1.602176634e-19
    HC_J_M / (energy_kev * 1e3 * 1.602_176_634e-19)
}

/// Convert wavelength in metres to photon energy in keV.
#[inline]
pub fn wavelength_m_to_kev(wavelength_m: f64) -> f64 {
    HC_J_M / (wavelength_m * 1e3 * 1.602_176_634e-19)
}
