use crate::material::DispersiveMaterial;
use crate::units::{RefractiveIndex, Wavelength};

/// Perfectly Matched Layer (PML) absorbing boundary material
///
/// Stub for Phase 2 — will be used with FDTD CPML implementation.
#[derive(Debug, Clone)]
pub struct Pml {
    pub thickness: f64,
    pub max_conductivity: f64,
    pub polynomial_order: u32,
}

impl Pml {
    pub fn new(thickness: f64, max_conductivity: f64) -> Self {
        Self {
            thickness,
            max_conductivity,
            polynomial_order: 3,
        }
    }
}

impl DispersiveMaterial for Pml {
    fn refractive_index(&self, _wavelength: Wavelength) -> RefractiveIndex {
        RefractiveIndex::real(1.0)
    }

    fn name(&self) -> &str {
        "PML"
    }
}
