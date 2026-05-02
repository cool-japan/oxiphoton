pub mod absorption;
pub(crate) mod am0_e490_data;
pub mod antireflection;
pub mod back_reflector;
pub mod drift_diffusion;
pub mod light_trapping;
pub mod spectral_response;
pub mod spectrum;
pub mod stack;
pub mod textured_grating;
pub mod texturing;

pub use absorption::AbsorptionMaterial;
pub use antireflection::{DoubleLayerArc, SingleLayerArc};
pub use back_reflector::{
    evaluate_design, optimize_ar_and_back_reflector, ArBackReflectorDesign, DesignParams,
};
pub use light_trapping::{
    lambertian_jsc_si, single_pass_jsc_si, LightTrappingAnalysis, LightTrappingComparison,
};
pub use spectral_response::{
    compute_spectral_response, compute_spectral_response_dd, material_for_absorber,
    single_wavelength_absorption_z, AbsorberLayer, ArCoatingSpec, BackReflectorSpec,
    DriftDiffusionDeviceConfig, SolarCellDesign, SpectralResponse, TexturingSpec,
};
pub use spectrum::{SolarSpectrum, AM15G_DATA};
pub use stack::{SolarCellStack, StackLayer};
pub use textured_grating::{evaluate_textured_absorption, TexturedAbsorptionResult};
pub use texturing::{InvertedPyramidArray, LightTrappingModel, TextureType};
