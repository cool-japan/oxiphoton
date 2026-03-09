//! Adaptive optics module for OxiPhoton.
//!
//! Provides complete adaptive optics simulation including:
//! - Deformable mirrors (continuous facesheet, segmented, Zernike correctors)
//! - Wavefront sensors (Shack-Hartmann, pyramid, curvature)
//! - Control algorithms (integrator, modal, predictive)
//! - Atmospheric turbulence models (Kolmogorov, layered, phase screens)
//!
//! # References
//! - Hardy, "Adaptive Optics for Astronomical Telescopes" (1998)
//! - Tyson, "Principles of Adaptive Optics" (4th ed., 2015)
//! - Roddier (ed.), "Adaptive Optics in Astronomy" (1999)

pub mod atmosphere;
pub mod control;
pub mod deformable_mirror;
pub mod wavefront_sensor;

pub use atmosphere::*;
pub use control::*;
pub use deformable_mirror::*;
pub use wavefront_sensor::*;
