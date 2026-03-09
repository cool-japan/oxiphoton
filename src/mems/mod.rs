//! Optical MEMS & Microresonator Physics
//!
//! This module provides simulation and modeling tools for Micro-Electro-Mechanical
//! Systems (MEMS) used in photonic applications:
//!
//! - [`cantilever`]: Euler-Bernoulli cantilever beam models for MEMS sensors
//! - [`fabry_perot_mems`]: Tunable MEMS Fabry-Pérot cavities with electrostatic actuation
//! - [`mems_mirror`]: MEMS scanning mirrors (tilt, gimbal, VOA)
//! - [`coupling`]: Optomechanical coupling in WGM microresonators and disk resonators

pub mod cantilever;
pub mod coupling;
pub mod fabry_perot_mems;
pub mod mems_mirror;

pub use cantilever::*;
pub use coupling::*;
pub use fabry_perot_mems::*;
pub use mems_mirror::*;
