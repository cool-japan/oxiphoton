/// `pic_simulation` — Circuit-level simulation of photonic integrated circuits.
///
/// This module provides transfer-matrix and noise models for PIC elements
/// without relying on physical layout (see `pic_design` for that). It is
/// entirely pure-Rust with no external numeric-library dependencies.
///
/// # Sub-modules
///
/// | Module | Contents |
/// |--------|----------|
/// | [`circuit_elements`] | Directional coupler, microring, MZI, complex arithmetic |
/// | [`transfer_matrix`] | Waveguide sections, grating couplers, Y-junctions, cascade |
/// | [`noise_model`] | Shot noise, RIN, PDL, thermal noise, phase noise |
/// | [`yield_model`] | Process variation, Murphy/Poisson yield, Monte Carlo |
///
/// # Quick Example
///
/// ```rust,no_run
/// use oxiphoton::pic_simulation::circuit_elements::{DirectionalCoupler, MachZehnderInterferometer};
///
/// let mzi = MachZehnderInterferometer::symmetric();
/// let t_through = mzi.through_transmission();
/// let t_cross   = mzi.cross_transmission();
/// assert!((t_through + t_cross - 1.0).abs() < 1e-10);
/// ```
pub mod circuit_elements;
pub mod noise_model;
pub mod transfer_matrix;
pub mod yield_model;

// ---------------------------------------------------------------------------
// Flat re-exports
// ---------------------------------------------------------------------------

// Complex number and 2×2 transfer matrix — used throughout the module
pub use circuit_elements::{Complex, TransferMatrix2x2};

// PIC building blocks
pub use circuit_elements::{DirectionalCoupler, MachZehnderInterferometer, MicroringResonator};

// Cascade infrastructure
pub use transfer_matrix::{GratingCoupler, PicCascade, WaveguideSection, YJunction};

// Noise models
pub use noise_model::{
    OsnrModel, PhaseNoise, PolarizationDependentLoss, RinNoise, ShotNoise, ThermalNoise, C_LIGHT,
    H_PLANCK, KB,
};

// Yield and variability models
pub use yield_model::{MonteCarloYield, ProcessVariation, TrimCorrection, YieldModel};
