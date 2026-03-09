//! Nonlinear Optical Microscopy Module
//!
//! This module provides simulation models for advanced nonlinear optical
//! microscopy and nanoscopy techniques used in biological and materials imaging.
//!
//! # Submodules
//!
//! | Module | Technique | Key quantity |
//! |--------|-----------|-------------|
//! | [`shg_microscopy`] | SHG / THG microscopy | χ⁽²⁾/χ⁽³⁾ signal, phase matching, PSF |
//! | [`cars`] | CARS & SRS microscopy | Raman susceptibility, spectral lineshape |
//! | [`sted`] | STED nanoscopy | Depletion PSF, saturation intensity |
//! | [`fcs`] | Fluorescence correlation spectroscopy | G(τ), diffusion time, N, D |
//!
//! # Physical constants (re-exported for convenience)
//! Physical constants (`KB`, `H_PLANCK`, `C_LIGHT`, `HBAR`) are defined in
//! each submodule rather than centrally to keep modules self-contained.
//!
//! # Example — SHG resolution
//! ```rust,no_run
//! use oxiphoton::nonlinear_microscopy::shg_microscopy::ShgMicroscope;
//!
//! let scope = ShgMicroscope::new(1040e-9, 1.2, 1.333, 100e-9);
//! let dz = scope.axial_resolution_m();
//! let signal = scope.shg_signal(1e12, 10e-12, 5e-6);
//! println!("Axial resolution: {:.1} nm", dz * 1e9);
//! println!("SHG signal: {:.3e} W/m²", signal);
//! ```

pub mod cars;
pub mod fcs;
pub mod shg_microscopy;
pub mod sted;

// Convenience re-exports for the most commonly used types
pub use cars::{CarsSetup, CarsSignal, RamanSusceptibility, SrsDetector};
pub use fcs::{FccsMeasurement, FcsFitter, FcsSetup};
pub use shg_microscopy::{CollagenShg, ShgMicroscope, ThgMicroscope};
pub use sted::{Fluorophore, Sted3d, StedBeam};
