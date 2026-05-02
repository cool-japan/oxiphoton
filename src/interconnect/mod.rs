//! Optical interconnect modeling for silicon photonics.
//!
//! Models end-to-end optical link performance including:
//! - Transmitter (laser/modulator) power and extinction ratio
//! - Waveguide propagation loss and bend loss
//! - Splitter/coupler insertion losses
//! - Receiver sensitivity and bandwidth
//! - Link power budget and margin
//! - End-to-end S-parameter cascade and link metrics (feature `interconnect`)
//! - BER vs OSNR characterisation (feature `interconnect`)
//! - WDM crosstalk matrix (feature `interconnect`)
pub mod channel;
pub mod link;
pub mod network;
pub mod transceiver;
pub mod wdm;

#[cfg(feature = "interconnect")]
pub mod ber_analysis;
#[cfg(feature = "interconnect")]
pub mod eye_diagram;
#[cfg(feature = "interconnect")]
pub mod sparam_link;
#[cfg(feature = "interconnect")]
pub mod wdm_crosstalk;

pub use channel::OpticalChannel;
pub use link::{Connector, LinkBudget, OpticalLink};
pub use network::{NetworkTopology, PhotonicNetwork};
pub use transceiver::{ModulationFormat, OpticalReceiver, OpticalTransmitter, Transceiver};
pub use wdm::{
    CrosstalkMatrix, DwdmChannelPlan, OpticalAddDropMux, RingWdmFilter, WdmChannel, WdmGrid,
    WdmSystem,
};

#[cfg(feature = "interconnect")]
pub use ber_analysis::{ber_vs_osnr_sweep_for_link, BerOsnrCurve, LinkPerformanceAnalysis};
#[cfg(feature = "interconnect")]
pub use eye_diagram::{prbs, simulate_eye, EyeDiagramConfig, EyeDiagramResult};
#[cfg(feature = "interconnect")]
pub use sparam_link::{
    chip_to_chip_link_response, DirectionalCoupler, SiPhElement, SiPhLink, Splitter50_50,
    WaveguideSection,
};
#[cfg(feature = "interconnect")]
pub use wdm_crosstalk::{WdmCh, WdmCrosstalkMatrix};
