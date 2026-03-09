//! Optical Communications signal processing and analysis.
//!
//! This module provides:
//! - BER (bit error rate) calculations for OOK, BPSK, QPSK, QAM
//! - Q-factor and OSNR analysis
//! - Modulation format definitions and constellation generation
//! - Coherent receiver modelling (phase noise, CD compensation)
//! - Optical amplifier chain and link-budget analysis
//! - FEC (forward error correction) overhead and gain
//!
//! # Quick-start
//!
//! ```rust
//! use oxiphoton::comms::metrics::BerCalculator;
//! let ber = BerCalculator::ber_from_q(6.0);
//! assert!(ber < 1e-8);
//! ```

pub mod link_budget;
pub mod metrics;
pub mod modulation;
