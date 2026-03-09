/// Statistical and Coherence Optics module for OxiPhoton.
///
/// Provides rigorous implementations of:
/// - Mutual coherence functions and cross-spectral density
/// - Temporal coherence and power spectral density
/// - Spatial coherence and Van Cittert-Zernike theorem
/// - Speckle statistics and reduction methods
pub mod mutual_coherence;
pub mod spatial;
pub mod speckle;
pub mod temporal;

pub use mutual_coherence::*;
pub use spatial::*;
pub use speckle::*;
pub use temporal::*;
