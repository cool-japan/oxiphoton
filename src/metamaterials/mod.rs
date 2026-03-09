//! # Metamaterials
//!
//! Bulk metamaterial physics for OxiPhoton, covering:
//!
//! - **Negative-index media** (`negative_index`): Veselago double-negative media,
//!   split-ring resonators (SRR), and Drude wire arrays.
//! - **Transformation optics** (`transformation_optics`): Coordinate-transformation
//!   cloaks (cylindrical, carpet), Luneburg lens, and Maxwell fish-eye lens.
//! - **Effective medium theory** (`effective_medium`): Maxwell Garnett, Bruggeman,
//!   and 1-D multilayer EMT including hyperbolic metamaterials.
//! - **Hyperlens & superlens** (`hyperlens`): Pendry flat superlens, optical
//!   cylindrical hyperlens, and spherical near-field superlens.
//!
//! > **Note:** This module covers *bulk* metamaterials.  For *flat* metasurfaces
//! > (phase-gradient arrays, Huygens' surfaces, geometric-phase elements) see
//! > [`crate::metasurface`].

pub mod effective_medium;
pub mod hyperlens;
pub mod negative_index;
pub mod transformation_optics;

// Re-export the most commonly used types at the module level.
pub use effective_medium::{BruggemanEmt, MaxwellGarnett, MultilayerEmt};
pub use hyperlens::{OpticalHyperlens, PendrySuperLens, SphericalSuperlens};
pub use negative_index::{DoubleNegativeMedium, DrudeWireArray, SplitRingResonator};
pub use transformation_optics::{CarpetCloak, CylindricalCloak, LuneburgLens, MaxwellFishEye};
