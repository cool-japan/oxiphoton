//! Photoacoustics — laser-induced ultrasound generation, imaging and spectroscopy
//!
//! This module provides a comprehensive set of physical models for photoacoustic
//! (PA) and optoacoustic phenomena relevant to medical imaging, gas sensing,
//! and opto-acoustic device physics.
//!
//! # Sub-modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`pa_generation`] | Grueneisen parameter, initial pressure, spectral unmixing |
//! | [`pa_imaging`]    | Delay-and-sum beamforming, circular PAT arrays, resolution metrics |
//! | [`optoacoustic`]  | SBS, thermal lensing, acousto-optic modulators |
//! | [`pa_spectroscopy`] | Resonant PA cells, gas sensing, NNEA figure of merit |
//!
//! # Key physical relations
//!
//! * **Initial pressure rise** (stress confinement): p₀ = Γ μ_a F
//! * **Grueneisen parameter**: Γ = β c_s² / C_p
//! * **PAT axial resolution**: Δz ≈ 0.88 c_s / BW
//! * **Brillouin shift**: ν_B = 2 n v_a / λ
//! * **Bragg angle** (AOM): sin θ_B = λ / (2 n Λ)

pub mod optoacoustic;
pub mod pa_generation;
pub mod pa_imaging;
pub mod pa_spectroscopy;

// Re-export the most commonly used types at the module boundary
pub use optoacoustic::{AcoustoOpticModulator, StimulatedBrillouinScattering, ThermalLensing};
pub use pa_generation::{GrueneisenParameter, PhotoacousticSource, SpectralUnmixing};
pub use pa_imaging::{
    back_projection_weight, CircularPetArray, DelayAndSumBeamformer, PatResolution,
    UniversalBackProjection,
};
pub use pa_spectroscopy::{
    beer_lambert_absorptance, ideal_gas_number_density, PaCell, PaGasSensor,
};
