//! Optical Network Design & WDM Systems.
//!
//! This module provides comprehensive tools for designing and analyzing
//! wavelength-division multiplexed (WDM) optical networks, including:
//!
//! - ITU-T G.694.1 channel plans (fixed and flex-grid)
//! - WDM line system capacity and OSNR analysis
//! - ROADM node modeling with wavelength-selective switches
//! - Multi-span link margin analysis (Gaussian noise model)
//! - Physical-layer impairments: PMD, SRS, XPM, FWM

pub mod impairments;
pub mod link_design;
pub mod roadm;
pub mod wdm_system;

pub use impairments::*;
pub use link_design::*;
pub use roadm::*;
pub use wdm_system::*;
