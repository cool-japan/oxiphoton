//! Ultrafast pulse characterization module.
//!
//! Provides simulation and retrieval algorithms for characterizing ultrashort
//! optical pulses via:
//! - FROG (Frequency-Resolved Optical Gating)
//! - SPIDER (Spectral Phase Interferometry for Direct Electric-field Reconstruction)
//! - Autocorrelation (intensity, interferometric, cross-correlation)
//! - Pulse shaping (4-f SLM, DAZZLER acousto-optic)
//!
//! # Physical background
//!
//! Ultrafast pulses (femtosecond to picosecond regime) carry complex temporal
//! phase structure — chirp, higher-order dispersion — that cannot be measured
//! by slow detectors. These techniques encode temporal phase information into
//! measurable spectral or intensity signals.

pub mod autocorrelation;
pub mod frog;
pub mod pulse_shaping;
pub mod spider;

pub use autocorrelation::*;
pub use frog::*;
pub use pulse_shaping::*;
pub use spider::*;
