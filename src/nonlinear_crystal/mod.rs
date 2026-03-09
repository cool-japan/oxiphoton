//! Nonlinear Crystal Optics Module
//!
//! Provides simulation of nonlinear optical processes in crystals including:
//! - Phase matching analysis (Type I, II, III, quasi-phase matching)
//! - Second and third harmonic generation (SHG/THG) efficiency
//! - Optical parametric amplification/oscillation (OPA/OPO)
//! - Crystal properties for common NLO materials (BBO, KTP, LiNbO3, KDP)
//! - Walk-off, acceptance bandwidths, coherence lengths
//! - QPM (quasi-phase matching) with periodic poling

pub mod crystals;
pub mod opa;
pub mod phase_matching;

pub use crystals::{CrystalClass, NloCrystal, SellmeierCoeff};
pub use opa::{OpticalParametricAmplifier, OpticalParametricOscillator, QpmShg};
pub use phase_matching::{
    ConversionProcess, FrequencyConversion, PhaseMatchingType, SHGPhaseMatching,
};
