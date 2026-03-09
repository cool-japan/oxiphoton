//! Free-Space Optical (FSO) Communications & Atmospheric Propagation Module
//!
//! Provides comprehensive models for atmospheric turbulence effects on laser
//! propagation, FSO link budget analysis, beam wander statistics, and
//! pointing/acquisition systems for ground-to-ground and satellite optical links.
//!
//! # Modules
//! - [`turbulence`]: Atmospheric turbulence models (Kolmogorov, Hufnagel-Valley, etc.)
//! - [`fso_link`]: Free-space optical link budget and fading statistics
//! - [`beam_wander`]: Beam wander variance and tip-tilt correction
//! - [`pointing`]: Pointing, acquisition, and satellite link geometry

pub mod beam_wander;
pub mod fso_link;
pub mod pointing;
pub mod turbulence;

pub use beam_wander::*;
pub use fso_link::*;
pub use pointing::*;
pub use turbulence::*;
