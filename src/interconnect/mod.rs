//! Optical interconnect modeling for silicon photonics.
//!
//! Models end-to-end optical link performance including:
//! - Transmitter (laser/modulator) power and extinction ratio
//! - Waveguide propagation loss and bend loss
//! - Splitter/coupler insertion losses
//! - Receiver sensitivity and bandwidth
//! - Link power budget and margin
pub mod channel;
pub mod link;
pub mod network;
pub mod transceiver;
pub mod wdm;

pub use channel::OpticalChannel;
pub use link::{Connector, LinkBudget, OpticalLink};
pub use network::{NetworkTopology, PhotonicNetwork};
pub use transceiver::{ModulationFormat, OpticalReceiver, OpticalTransmitter, Transceiver};
pub use wdm::{
    CrosstalkMatrix, DwdmChannelPlan, OpticalAddDropMux, RingWdmFilter, WdmChannel, WdmGrid,
    WdmSystem,
};
