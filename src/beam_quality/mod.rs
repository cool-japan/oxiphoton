//! Beam quality characterisation for laser beams.
//!
//! This module implements the M² beam quality parameter and associated tools
//! following the ISO 11146 standard (*Test methods for laser beam widths,
//! divergence angles and beam propagation ratios*).
//!
//! # Sub-modules
//! | Module | Contents |
//! |--------|----------|
//! | [`m2_factor`] | M² factor, ABCD propagation, mode-specific formulae, brightness |
//! | [`caustic`]   | Beam caustic measurement and parabolic fitting |
//! | [`iso_standard`] | ISO 11146 measurement protocol and sampling checks |
//! | [`beam_profiler`] | 1D/2D intensity profile moments (D4σ, FWHM, centroid) |
//!
//! # Quick start
//! ```rust,ignore
//! use oxiphoton::beam_quality::m2_factor::BeamQuality;
//!
//! // 1064 nm beam with M² = 1.3, 0.5 mm waist at z = 0
//! let bq = BeamQuality::new(1064e-9, 1.3, 0.5e-3, 0.0);
//! println!("Rayleigh range: {:.3} mm", bq.rayleigh_range_m() * 1e3);
//! println!("BPP:           {:.3} mm·mrad", bq.beam_parameter_product() * 1e6);
//! ```

pub mod beam_profiler;
pub mod caustic;
pub mod iso_standard;
pub mod m2_factor;

// Re-export the most commonly used items at module level.
pub use beam_profiler::{BeamProfile1d, BeamProfile2d};
pub use caustic::{BeamCaustic, BeamMeasurement};
pub use iso_standard::{synthetic_gaussian_caustic, Iso11146Measurement, Iso11146Result};
pub use m2_factor::{
    brightness_w_per_m2_sr, hermite_gaussian_m2, laguerre_gaussian_m2, BeamQuality,
};
