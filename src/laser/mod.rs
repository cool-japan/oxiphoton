pub mod eom;
/// Laser physics module for OxiPhoton.
///
/// Provides mode-locked laser simulation using the Haus master equation,
/// electro-optic modulators (Pockels effect, IQ), acousto-optic modulator
/// models for ultrafast and coherent photonics, as well as solid-state and
/// semiconductor laser rate equation models including:
///
/// - Four-level and three-level laser rate equations (Nd:YAG, Er:fiber, etc.)
/// - Q-switched pulse dynamics
/// - Semiconductor diode laser (InGaAsP, VCSEL, DFB)
/// - Gain saturation, threshold analysis, relaxation oscillations
pub mod mode_locked;
pub mod rate_equations;
pub mod semiconductor;

pub use eom::{AcoustomOpticModulator, EomType, IqModulator, PockelsEom};
pub use mode_locked::{HausMasterEquation, KerrLensModelocking, ModeLockLaser, Sesam};
pub use rate_equations::{FourLevelLaser, QSwitchedLaser, ThreeLevelLaser};
pub use semiconductor::{DfbLaser, SemiconductorLaser, Vcsel};
