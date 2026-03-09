pub mod database;
pub mod dispersive;
pub mod effective_medium;
pub mod nonlinear;
pub mod pml;
pub mod twod_materials;
pub use twod_materials::*;

use crate::units::{RefractiveIndex, Wavelength};
use num_complex::Complex64;

/// Drude-Lorentz pole parameters for FDTD dispersive material modeling
#[derive(Debug, Clone, Copy)]
pub struct DrudeLorentzPole {
    pub amplitude: f64,
    pub frequency: f64,
    pub damping: f64,
}

/// Trait for wavelength-dependent (dispersive) materials
pub trait DispersiveMaterial: Send + Sync {
    /// Compute complex refractive index at given wavelength
    fn refractive_index(&self, wavelength: Wavelength) -> RefractiveIndex;

    /// Compute complex permittivity at given wavelength
    fn permittivity(&self, wavelength: Wavelength) -> Complex64 {
        self.refractive_index(wavelength).to_permittivity_scalar()
    }

    /// Return FDTD pole parameters (for ADE method)
    fn fdtd_poles(&self) -> Vec<DrudeLorentzPole> {
        Vec::new()
    }

    /// Human-readable name of the material
    fn name(&self) -> &str;
}

pub use database::MaterialDatabase;
pub use dispersive::cauchy::Cauchy;
pub use dispersive::drude::Drude;
pub use dispersive::drude_lorentz::DrudeLorentz;
pub use dispersive::sellmeier::Sellmeier;
pub use dispersive::tabulated::Tabulated;
pub use nonlinear::chi2::Chi2Material;
pub use nonlinear::kerr::KerrMaterial;
pub mod anisotropic;
pub mod gain;
pub use anisotropic::{AnisotropicMaterial, DielectricTensor};
pub use gain::{GainMedium, LaserModel, TwoLevelMedium};
pub use nonlinear::raman::RamanMaterial;
