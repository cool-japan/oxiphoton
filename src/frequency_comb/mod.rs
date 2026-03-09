/// Optical Frequency Combs and Precision Metrology.
///
/// Provides a comprehensive framework for simulating and analyzing optical
/// frequency comb sources, stabilization techniques, comb-based spectroscopy,
/// and precision timing/metrology applications.
///
/// # Modules
/// - `comb`: Core frequency comb physics — Ti:Sa, erbium fiber, and Kerr microcombs
/// - `stabilization`: f-2f interferometry, PLL locking, and optical atomic clocks
/// - `spectroscopy`: Dual-comb spectroscopy, direct comb spectroscopy, and HHG attosecond sources
/// - `timing`: Allan deviation analysis, fiber frequency transfer, and relativistic geodesy
pub mod comb;
pub mod spectroscopy;
pub mod stabilization;
pub mod timing;

pub use comb::*;
pub use spectroscopy::*;
pub use stabilization::*;
pub use timing::*;
