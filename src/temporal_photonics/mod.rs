//! Temporal photonics: time-modulated and time-varying optical systems.
//!
//! This module covers physics where the optical medium properties vary
//! periodically (or abruptly) in **time** rather than (or in addition to)
//! space.  Key phenomena include:
//!
//! - **Floquet-Bloch theory** (`floquet_theory`): quasi-energy band structure
//!   of periodically driven cavities, parametric resonances, Floquet sidebands.
//! - **Optical parametric amplification** (`parametric_amplification`): OPA /
//!   OPO gain, phase matching (Type-I, II, QPM), bandwidth, noise figure.
//! - **Temporal refraction/reflection** (`time_refraction`): sudden index
//!   change in time — k-conserving frequency conversion, temporal Snell's law,
//!   time slabs.
//! - **Photonic time crystals** (`photonic_time_crystal`): momentum band gaps,
//!   amplification threshold, vacuum squeezing, topological winding numbers,
//!   non-reciprocal spatiotemporal crystals.
//!
//! # Quick example
//!
//! ```rust
//! use oxiphoton::temporal_photonics::time_refraction::TemporalInterface;
//!
//! let ti = TemporalInterface { n_before: 1.5, n_after: 2.0, omega_incident: 2e15 };
//! assert!(ti.verify_k_conservation());
//! ```

pub mod floquet_theory;
pub mod parametric_amplification;
pub mod photonic_time_crystal;
pub mod time_refraction;

// ─── Re-exports ───────────────────────────────────────────────────────────────

pub use floquet_theory::{FloquetCavity, ModulatedCavity};
pub use parametric_amplification::{
    OpticalParametricAmplifier, PhaseMatchingType, QuasiPhaseMatching,
};
pub use photonic_time_crystal::{PhotonicTimeCrystal, SpatiotemporalCrystal};
pub use time_refraction::{TemporalInterface, TimeSlab};
