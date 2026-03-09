//! Optical Forces and Radiation Pressure Simulation
//!
//! This module provides tools for simulating optical forces including:
//! - Maxwell stress tensor calculations for electromagnetic force computation
//! - Optical tweezers simulation using gradient and scattering forces
//! - Optomechanical cavity coupling and dynamical backaction
//!
//! # Physical Background
//! Optical forces arise from the transfer of momentum from photons to matter.
//! The Maxwell stress tensor provides the rigorous electromagnetic framework,
//! while the Rayleigh approximation enables efficient particle trap simulations.

pub mod maxwell_stress;
pub mod optomechanics;
pub mod tweezers;

pub use maxwell_stress::{MaxwellStressTensor, RadiationPressure};
pub use optomechanics::{MembraneOptomechanics, OptomechanicalCavity, PhotonRecoil};
pub use tweezers::{DualBeamTrap, OpticalBinding, OpticalTweezers, TrapAxis};
