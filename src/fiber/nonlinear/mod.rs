//! Nonlinear fiber optics effects.
//!
//! Covers the principal third-order nonlinear processes in silica optical fibers:
//! SPM (self-phase modulation), XPM (cross-phase modulation), FWM (four-wave mixing),
//! and soliton propagation.
pub mod fwm;
pub mod soliton;
pub mod spm;
pub mod xpm;

pub use fwm::{FwmFiber, FwmPhaseMatching, ParametricAmplifier};
pub use soliton::SolitonFiber;
pub use spm::{SplitStepNls, SpmFiber};
pub use xpm::{TwoChannelPropagation, XpmChannel, XpmCoeff};
