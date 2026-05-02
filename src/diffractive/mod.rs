/// Diffractive optics: gratings, holographic DOEs, Fresnel/diffractive lenses,
/// scalar diffraction propagation (Rayleigh-Sommerfeld, Fresnel, Fraunhofer),
/// and grating spectrometer design.
///
/// Physical conventions used throughout:
/// - Wavelengths in nm
/// - Distances in μm unless noted mm
/// - Angles in radians unless noted deg
/// - Refractive indices dimensionless
pub mod grating;
pub mod propagation;

pub use grating::{
    DammannGrating, DiffractionGrating, GratingSpectrometer, GratingType, HolographicGrating,
    VolumeBraggGrating, VolumeGrating,
};
pub use propagation::{DiffractiveLens, DiffractiveLensType, ScalarDiffraction, SlmHologram};
