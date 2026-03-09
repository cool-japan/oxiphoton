pub mod adc;
pub mod beamforming;
/// Microwave Photonics module for OxiPhoton.
///
/// Covers:
/// - Analog photonic links (RF-over-fiber)
/// - Photonic RF signal processing and filtering
/// - Photonic beamforming for phased array antennas
/// - Photonic analog-to-digital conversion (ADC)
pub mod link;
pub mod rf_filter;

pub use adc::{PhotonicAdc, PhotonicChannelizer};
pub use beamforming::{BfnArchitecture, OpticalBfn, PhotonicBeamformer};
pub use link::{AnalogPhotonicLink, EoModulatorType, LinkBudget, MzmBias, PhotodetectorParams};
pub use rf_filter::{PhotonicHilbertTransformer, PhotonicRfFilter, RingResonatorRfFilter};
