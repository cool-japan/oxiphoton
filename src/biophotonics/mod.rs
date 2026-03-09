//! Biophotonics and Tissue Optics module
//!
//! Provides models for light-tissue interaction including:
//! - Tissue optical properties (scattering/absorption coefficients)
//! - Diffusion approximation for light transport in turbid media
//! - Hemoglobin absorption spectroscopy
//! - Photodynamic therapy (PDT) dosimetry
//! - Fluorescence lifetime and FRET models
//! - Two-photon excitation (2PE) microscopy
//! - Fluorescence lifetime imaging (FLIM) analysis

pub mod fluorescence;
pub mod tissue;

pub use fluorescence::{FlimAnalysis, Fluorophore, FretPair, TwoPhotonExcitation};
pub use tissue::{DiffusionModel, HemoglobinModel, PdtDosimetry, TissueOpticalProperties};
