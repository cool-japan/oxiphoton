//! Quantum Photonic Computing module for OxiPhoton.
//!
//! Provides:
//! - Multi-mode Fock state algebra and superpositions
//! - Linear optical quantum gates (beam splitters, MZI meshes, Reck/Clements decompositions)
//! - Hong-Ou-Mandel (HOM) two-photon interference and indistinguishability measurement
//! - Gaussian boson sampling (GBS) with squeezed states and PNR detection
//!
//! # References
//! - Knill, Laflamme, Milburn (2001) — linear optical quantum computing
//! - Hong, Ou, Mandel (1987) — two-photon interference
//! - Arrazola et al. (2021) — Gaussian boson sampling
//! - Reck et al. (1994); Clements et al. (2016) — MZI mesh decompositions

pub mod boson_sampling;
pub mod fock_state;
pub mod hom_effect;
pub mod linear_optical;

pub use boson_sampling::*;
pub use fock_state::*;
pub use hom_effect::*;
pub use linear_optical::*;
