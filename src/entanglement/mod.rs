//! Photon pair sources and quantum entanglement module.
//!
//! Provides:
//! - SPDC (Spontaneous Parametric Down-Conversion) photon pair sources
//! - Quantum entanglement measures (concurrence, negativity, von Neumann entropy)
//! - Bell inequality tests (CHSH, CH, Mermin)
//! - Quantum Key Distribution protocols (BB84, E91, CV-QKD)

pub mod bell_inequality;
pub mod entanglement_measures;
pub mod qkd;
pub mod spdc;

pub use bell_inequality::*;
pub use entanglement_measures::*;
pub use qkd::*;
pub use spdc::*;
