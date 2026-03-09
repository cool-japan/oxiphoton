//! Optical Code Division Multiple Access (OCDMA) simulation module.
//!
//! This module provides a comprehensive pure-Rust toolkit for modelling and
//! analysing optical CDMA systems at the physical layer.  It covers:
//!
//! * **Spreading codes** (`spreading_codes`) — Optical Orthogonal Codes (OOC),
//!   Orthogonal Variable Spreading Factor (OVSF) trees, and Gold codes.
//! * **OCDMA transceivers** (`ocdma_system`) — Incoherent (OOK) and coherent
//!   (bipolar) transceiver models with multiple-access interference analysis.
//! * **Spectral encoding** (`spectral_encoding`) — Spectral Amplitude Coding
//!   (SAC-OCDMA) with the MDW code family, and Spectral Phase Coding
//!   (SPC-OCDMA) with Walsh–Hadamard codes.
//! * **Performance analysis** (`performance`) — Q-function, BER models
//!   (Gaussian approximation and exact binomial sum), capacity estimation,
//!   and multi-access scheme comparison.
//!
//! # Design principles
//! * No `unwrap()` calls — all fallible paths use `unwrap_or`/`unwrap_or_default`.
//! * No external numeric dependencies — pure-Rust arithmetic throughout.
//! * No warnings — all items are either exported or `#[allow(dead_code)]`
//!   annotated where appropriate.
//!
//! # Quick start
//! ```rust,ignore
//! use oxiphoton::optical_cdma::spreading_codes::OpticalOrthogonalCode;
//!
//! let ooc = OpticalOrthogonalCode::new(13, 3, 1, 1);
//! println!("Max codewords: {}", ooc.max_codewords()); // 2
//! let codes = ooc.generate_codes();
//! assert_eq!(codes.len(), 2);
//! ```

pub mod ocdma_system;
pub mod performance;
pub mod spectral_encoding;
pub mod spreading_codes;

// ---------------------------------------------------------------------------
// Convenience re-exports
// ---------------------------------------------------------------------------

pub use ocdma_system::{CoherentOcdma, IncoherentOcdma, MaiAnalyzer};
pub use performance::{erfc_approx, q_function, MultipleAccessComparison, OokOcdmaBer};
pub use spectral_encoding::{SacOcdma, SpcOcdma};
pub use spreading_codes::{GoldCode, OpticalOrthogonalCode, OvsfTree};
