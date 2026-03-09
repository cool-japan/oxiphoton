//! Silicon Photonics Integrated Circuit (PIC) components.
//!
//! Provides transfer-matrix and coupled-mode theory models for key silicon
//! photonics building blocks: ring resonators, Mach-Zehnder interferometers,
//! arrayed waveguide gratings (AWG), and directional couplers.
//!
//! # Physical models
//!
//! - **Ring resonators**: All-pass and add-drop configurations using transfer
//!   matrix formalism with round-trip field equations.
//! - **MZI**: 2×2 unitary coupler matrices cascaded with arm phase delays.
//! - **AWG**: Phased-array demultiplexer model with Gaussian channel profiles.

pub mod awg;
pub mod mzi;
pub mod ring_resonator;

pub use awg::{ArrayedWaveguideGrating, ItuGrid};
pub use mzi::{MachZehnderInterferometer, MziSwitch, SwitchState};
pub use ring_resonator::{AddDropRing, AllPassRing, RingModulator};
