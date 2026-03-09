//! Optical Coherence Tomography (OCT) simulation module.
//!
//! Provides tools for spectral-domain OCT (SD-OCT), time-domain OCT (TD-OCT),
//! and swept-source OCT (SS-OCT), along with A-scan/B-scan processing,
//! coherence gating, Doppler flow measurement, and signal quality analysis.
//!
//! # References
//! - Drexler & Fujimoto, "Optical Coherence Tomography", 2nd ed. (2015)
//! - Fercher et al., "Optical coherence tomography — principles and applications",
//!   Rep. Prog. Phys. 66 (2003) 239–303
//! - Leitgeb et al., "Performance of Fourier domain vs. time domain optical coherence
//!   tomography", Opt. Express 11 (2003) 889–894

pub mod analysis;
pub mod spectral_domain;

pub use analysis::{AScanProcessor, DopplerOct, OctMetrics, WindowFunction};
pub use spectral_domain::{SdOct, SsOct, TdOct};
