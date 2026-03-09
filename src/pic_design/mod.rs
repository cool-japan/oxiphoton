//! Photonic Integrated Circuit (PIC) Design Tools
//!
//! This module provides a complete toolchain for PIC design including:
//! - Process Design Kits (PDK) for SOI and SiN platforms
//! - Waveguide routing utilities (bends, tapers, crossings, S-bends)
//! - Parameter optimization (PSO, ring optimizer, MZI optimizer)
//! - Design Rule Checking (DRC) and circuit verification

pub mod optimization;
pub mod pdk;
pub mod routing;
pub mod verification;

pub use optimization::*;
pub use pdk::*;
pub use routing::*;
pub use verification::*;
