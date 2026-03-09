/// Top-level detector physics module.
///
/// Provides rigorous photodetector models independent of the device-level
/// `devices::detector` sub-system:
///
/// - [`photodiode`]: PIN photodiode and APD models with noise analysis
/// - [`single_photon`]: SNSPD, PMT, SPAD, and TCSPC models
/// - [`noise`]: Detector noise toolkit and photon-counting statistics
pub mod noise;
pub mod photodiode;
pub mod single_photon;

pub use noise::{DetectorNoiseModel, NoiseAnalysis, PhotonCounting};
pub use photodiode::{AvalanchePhotodiode, PinPhotodiode};
pub use single_photon::{Pmt, Snspd, Spad, TcSpc};
