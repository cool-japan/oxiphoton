//! Integrated Photonic Sensors & LiDAR Module
//!
//! Provides simulation and modelling of photonic sensing systems including:
//! - LiDAR (direct ToF, FMCW, photon-counting, scanning)
//! - Optical gyroscopes (FOG, RLG, integrated)
//! - Photonic chemical sensors (evanescent, WGM, SPR)
//! - Photonic inertial sensors (accelerometer, pressure, strain)

pub mod chemical;
pub mod gyroscope;
pub mod inertial;
pub mod lidar;

pub use chemical::*;
pub use gyroscope::*;
pub use inertial::*;
pub use lidar::*;
