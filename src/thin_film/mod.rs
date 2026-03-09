/// Multilayer thin film optics using the transfer matrix method.
///
/// Provides simulation of anti-reflection coatings, high-reflectance mirrors,
/// bandpass filters, edge filters, and Fabry-Perot etalons.
///
/// # References
/// - Born & Wolf, "Principles of Optics", Ch. 1
/// - Macleod, "Thin-Film Optical Filters", 4th ed.
/// - Heavens, "Optical Properties of Thin Solid Films"
pub mod coatings;
pub mod transfer_matrix;

pub use coatings::{
    AntiReflectionCoating, BandpassFilter, EdgeFilter, FabryPerotEtalon, HighReflectanceMirror,
};
pub use transfer_matrix::{
    fresnel_r_p, fresnel_r_s, fresnel_t_s, snell_law, Layer, MultilayerStack, Polarization,
};
