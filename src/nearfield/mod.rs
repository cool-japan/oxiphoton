//! Near-field Optics & Nanophotonics module for OxiPhoton
//!
//! Provides:
//! - Local Density of Optical States (LDOS) and Purcell enhancement
//! - Optical nanocavities: bowtie, NPoM, PhC, MIM
//! - Optical forces on nanoparticles (Rayleigh regime + near-field)
//! - Surface-Enhanced Raman Scattering (SERS) and TERS
//!
//! Physical foundations:
//! - Dyadic Green's function formalism for LDOS
//! - Clausius-Mossotti polarizability for nanoparticle forces
//! - Electromagnetic enhancement model for SERS
//! - Cavity QED for strong/weak coupling regimes

pub mod ldos;
pub mod nanocavity;
pub mod optical_force_nano;
pub mod sers;

pub use ldos::*;
pub use nanocavity::*;
pub use optical_force_nano::*;
pub use sers::*;
