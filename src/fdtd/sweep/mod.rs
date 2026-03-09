//! Parametric sweep framework for photonic simulations.
//!
//! Provides utilities for sweeping simulation parameters across ranges,
//! running convergence studies, and performing wavelength scans.
//!
//! # Examples
//! ```
//! use oxiphoton::fdtd::sweep::parameter::{ParamSweep, WavelengthSweep};
//!
//! // Linear sweep
//! let sweep = ParamSweep::linspace("thickness_nm", 100.0, 500.0, 5);
//! let results = sweep.run(|t| t * t);
//!
//! // Wavelength sweep from 1000 nm to 1600 nm
//! let ws = WavelengthSweep::new(1000.0, 1600.0, 61);
//! let _lambdas = ws.wavelengths_nm();
//! ```

pub mod parameter;

pub use parameter::{ConvergenceSweep, ParamGrid, ParamSweep, WavelengthSweep};
