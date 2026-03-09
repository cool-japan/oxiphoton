//! Photonic antenna module — optical nanoantenna theory, phased arrays,
//! LiDAR-specific OPA design, and radiation pattern analysis.
//!
//! # Sub-modules
//!
//! - [`optical_antenna`] — Hertzian dipole, nanorod, optical Yagi-Uda antenna
//! - [`phased_array`]    — 1D and 2D optical phased array beam steering
//! - [`lidar_opa`]       — LiDAR-optimised OPA system design and link budget
//! - [`pattern`]         — Antenna pattern metrics, directivity, Friis equation
//!
//! # Physical conventions
//!
//! All spatial quantities in SI metres, frequencies in rad/s, angles in radians
//! unless explicitly suffixed `_deg`.  Gains are linear (not dB) unless the
//! identifier contains `_db` or `_dbi`.

pub mod lidar_opa;
pub mod optical_antenna;
pub mod pattern;
pub mod phased_array;

// ─── Re-exports ───────────────────────────────────────────────────────────────

pub use optical_antenna::{HertzianDipole, NanorodAntenna, YagiUdaAntenna};

pub use phased_array::{OpticalPhasedArray1d, OpticalPhasedArray2d};

pub use lidar_opa::{OpaLidar, SiliconOpa};

pub use pattern::{
    directivity_dbi_from_pattern, directivity_from_pattern, effective_aperture_m2,
    free_space_path_loss_db, friis_equation, gain_dbi_to_linear, gain_linear_to_dbi,
    AntennaPatternMetrics,
};
