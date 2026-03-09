//! Quantum Optics simulation module for OxiPhoton.
//!
//! Provides:
//! - Jaynes-Cummings model (cavity QED)
//! - Photon number statistics (Fock, coherent, thermal, squeezed states)
//! - First- and second-order coherence theory (g¹, g²)
//! - Spatial coherence via Van Cittert-Zernike theorem
//! - Hanbury-Brown–Twiss experiment modelling

pub mod coherence;
pub mod jaynes_cummings;
pub mod photon_statistics;

pub use coherence::{
    FirstOrderCoherence, HBTExperiment, LightSource, SecondOrderCoherence, SpatialCoherence,
};
pub use jaynes_cummings::JaynesCummings;
pub use photon_statistics::{
    is_sub_poissonian, is_super_poissonian, mandel_q_parameter, mean_photon_number,
    second_order_coherence_zero_delay, variance_from_distribution, CoherentState, FockState,
    SqueezedState, ThermalState,
};
