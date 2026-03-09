//! Topological photonics simulation module.
//!
//! This module provides first-principles models for topological photonic systems,
//! covering 1D and 2D topological invariants, edge state analysis, and valley
//! Hall photonics.
//!
//! ## Submodules
//!
//! | Submodule                | Content |
//! |--------------------------|---------|
//! | [`ssh_chain`]            | SSH model, winding number, Zak phase, finite-chain edge states |
//! | [`chern_insulator`]      | QWZ 2D Chern insulator, Berry curvature, Chern number |
//! | [`topological_edge_states`] | Edge state analysis, PTI interfaces, AQHPC |
//! | [`valley_hall`]          | Valley Hall photonic crystals, kink states |
//!
//! ## Key physics
//!
//! - **SSH model**: 1D chain with alternating hopping t₁, t₂.  The winding number
//!   W = 1 (and Zak phase γ = π) for t₂ > t₁, guaranteeing two zero-energy edge
//!   modes in a finite chain (bulk-edge correspondence).
//!
//! - **QWZ Chern insulator**: 2D two-band model H(k) = d(k)·σ.  The Chern number
//!   is an integer topological invariant computed from the Berry curvature integral
//!   over the Brillouin zone: C = (1/2π)∬ Ω_z d²k.
//!
//! - **Photonic topological insulator**: Interface between two PhC domains with
//!   different Chern numbers hosts |ΔC| chiral edge modes immune to backscattering.
//!
//! - **Valley Hall effect**: Broken inversion symmetry in a honeycomb PhC opens a
//!   gap at the K/K′ points and assigns half-integer valley Chern numbers ±½.
//!   Domain walls host topologically protected valley kink states.

pub mod chern_insulator;
pub mod ssh_chain;
pub mod topological_edge_states;
pub mod valley_hall;

// ─── Convenience re-exports ───────────────────────────────────────────────────

pub use chern_insulator::{berry_curvature_map, chern_from_curvature_map, QwzModel};
pub use ssh_chain::{PhotonicSshResonator, SshChain};
pub use topological_edge_states::{
    AnomalousQhpc, PhotonicTopologicalInsulator, TopologicalEdgeState as TopoEdgeState,
};
pub use valley_hall::{ValleyHallPhC, ValleyKinkState};
