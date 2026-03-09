//! Nanolaser & Micro-Laser Physics Module for OxiPhoton.
//!
//! Provides models for nano- and micro-scale laser devices including:
//!
//! - Generalized laser rate equations with Purcell factor and spontaneous
//!   emission coupling (β→1 regime, thresholdless nanolasers)
//! - VCSEL (Vertical-Cavity Surface-Emitting Laser) physics and array models
//! - Nanolaser-specific physics: plasmonic nanolasers (SPASER), photonic
//!   crystal nanocavity lasers, and semiconductor disk lasers (VECSEL)
//! - Laser noise: RIN spectrum, frequency noise PSD, and partition noise
//!
//! # References
//!
//! - S. Noda, "Seeking the Ultimate Nanolaser", Science 314, 260 (2006)
//! - T. Baba, "Photonic crystals and microdisk cavities based on GaInAsP–InP",
//!   IEEE J. Sel. Topics Quantum Electron. 3, 808 (1997)
//! - K. J. Vahala, "Optical microcavities", Nature 424, 839 (2003)
//! - C. Z. Ning, "Semiconductor nanolasers and the size-energy-efficiency
//!   challenge", Adv. Photon. 1, 014002 (2019)

pub mod laser_noise;
pub mod nanolaser_physics;
pub mod rate_equations;
pub mod vcsel;

pub use laser_noise::*;
pub use nanolaser_physics::*;
pub use rate_equations::*;
pub use vcsel::*;
