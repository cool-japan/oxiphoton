/// THz (Terahertz) optics simulation module.
///
/// Covers:
/// - THz generation: optical rectification and photoconductive antennas
/// - THz time-domain spectroscopy (THz-TDS)
/// - THz waveguide modes (parallel-plate, rectangular, circular)
/// - Drude model for metals and doped semiconductors at THz frequencies
/// - Atmospheric THz absorption from water-vapor resonances
pub mod generation;
pub mod materials;
pub mod tds;

pub use generation::{FelThzSource, OpticalRectification, PcaSubstrate, PhotoconductiveAntenna};
pub use materials::{AtmosphericAbsorption, DrudeTHz, ThzMaterial};
pub use tds::{ThzGuideType, ThzTds, ThzWaveguide};
