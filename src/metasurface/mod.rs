pub mod geometric_phase;
pub mod metalens;
pub mod reflectarray;
/// Optical Metasurfaces & Flat Optics Module
///
/// Provides physics-based models for:
/// - Dielectric and plasmonic unit cells (Mie resonances, Drude model)
/// - Metalenses (hyperbolic phase, chromatic aberration, PSF)
/// - Pancharatnam-Berry / geometric-phase elements
/// - Reflectarrays and Reconfigurable Intelligent Surfaces (RIS)
///
/// All lengths in SI units (metres) unless noted otherwise.
/// Angles in radians internally; public helpers accept/return degrees where noted.
pub mod unit_cell;

pub use geometric_phase::{
    CircPolarization, MetasurfaceFunction, PbBeamSplitter, PbPhaseElement,
    SpinMultiplexedMetasurface,
};
pub use metalens::{
    Metalens, MetalensDoublet, MetalensUnitCellType, TuningMechanism, VarifocalMetalens,
};
pub use reflectarray::{HolographicMetasurface, ReflectArray, RisElement};
pub use unit_cell::{
    DielectricPillar, HuygensMetasurface, PlasmonicAntenna, PlasmonicMetal, VAntenna,
};
