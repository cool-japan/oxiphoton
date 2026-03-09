//! Spatial Division Multiplexing (SDM) Module
//!
//! Implements LP mode solving, few-mode fiber propagation, mode coupling,
//! and Mode Division Multiplexing (MDM) system analysis for SDM fiber communications.
//!
//! # Overview
//! - `lp_modes`: Linearly Polarized (LP) mode solver for step-index fiber
//! - `few_mode_fiber`: Few-mode fiber and multicore fiber propagation models
//! - `mode_coupling`: Random mode coupling, MIMO equalizers, and mode converters
//! - `mdm_system`: System-level MDM capacity and impairment analysis

pub mod few_mode_fiber;
pub mod lp_modes;
pub mod mdm_system;
pub mod mode_coupling;

pub use few_mode_fiber::*;
pub use lp_modes::*;
pub use mdm_system::*;
pub use mode_coupling::*;
