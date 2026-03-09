//! FDTD simulation analysis tools.
//!
//! Provides post-processing utilities for FDTD simulations:
//! - Courant stability checking
//! - Memory usage estimation
//! - Grid convergence analysis
//! - Field norms and RMS diagnostics
//! - Power spectral density and autocorrelation

pub mod convergence;

pub use convergence::{
    check_courant_stability, check_courant_stability_1d, check_courant_stability_2d,
    compute_autocorrelation, compute_psd, convergence_test, estimate_memory_mb,
    estimate_memory_usage, field_max_norm, field_norm_l2, field_rms, fit_convergence_order,
    psd_peak_frequency, ConvergenceResult, CourantResult,
};
