//! Polarimetry and polarization optics module.
//!
//! Provides Stokes parameters, Jones calculus, and Mueller matrix formalism
//! for analysis and synthesis of polarized light states and optical systems.
//!
//! # Formalisms
//! - **Jones calculus**: 2-component complex vector/2×2 complex matrix for fully polarized light
//! - **Stokes/Mueller calculus**: 4-component real vector/4×4 real matrix for partially polarized
//!   or unpolarized light
//!
//! # References
//! - Chipman, R. A. "Polarimetry." Handbook of Optics, Vol. 2 (1995)
//! - Collett, E. "Field Guide to Polarization." SPIE (2005)

pub mod jones;
pub mod mueller;
pub mod stokes;

pub use jones::{JonesMatrix, JonesVector};
pub use mueller::{MuellerMatrix, PolarDecomposition, StokesPolarimeter};
pub use stokes::{PolarizationState, StokesVector};
