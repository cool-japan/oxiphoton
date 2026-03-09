//! Quantum Dot & Solid-State Single-Photon Emitters module for OxiPhoton.
//!
//! Provides physics models for:
//! - InAs/GaAs self-assembled quantum dots (8-band k·p approximation)
//! - Solid-state colour centres (NV, SiV, SnV in diamond; hBN defects)
//! - Photon statistics: g²(τ), HBT interferometry, photon-number distributions
//! - Cavity-enhanced single-photon sources (Purcell effect, extraction efficiency)
//!
//! # References
//! - Lodahl et al., Rev. Mod. Phys. 87, 347 (2015)  — solid-state SPE review
//! - Senellart, Solomon & White, Nat. Nano 12, 1026 (2017) — engineered SPSs
//! - Doherty et al., Phys. Rep. 528, 1 (2013) — NV centre review
//! - Bradac et al., Nat. Comm. 10, 5625 (2019) — colour-centre comparison

pub mod cavity_emitter;
pub mod color_center;
pub mod photon_statistics;
pub mod quantum_dot;

pub use cavity_emitter::*;
pub use color_center::*;
pub use photon_statistics::*;
pub use quantum_dot::*;
