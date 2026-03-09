//! Plasmonics module for OxiPhoton
//!
//! Provides surface plasmon polariton (SPP) dispersion, LSPR for nanostructures,
//! metal-insulator-metal (MIM) waveguides, Kretschmann ATR configuration,
//! and plasmonic nanostructure calculations.
//!
//! Physical basis: Drude model for metal permittivity, transfer matrix method
//! for multilayer configurations, Mie theory for nanoparticles.

pub mod nanostructures;
pub mod spp;

pub use nanostructures::{DipoleAntenna, PlasmonicGap, PlasmonicNanoparticle, PlasmonicNanorod};
pub use spp::{DrudeMetal, KretschmannConfig, MimWaveguide, SurfacePlasmonPolariton};
