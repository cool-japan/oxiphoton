use thiserror::Error;

#[derive(Debug, Error)]
pub enum OxiPhotonError {
    #[error("Invalid wavelength: {0} m (must be positive and finite)")]
    InvalidWavelength(f64),

    #[error("Invalid refractive index: n={n}, k={k} (n must be positive, k must be non-negative)")]
    InvalidRefractiveIndex { n: f64, k: f64 },

    #[error("Invalid layer: {0}")]
    InvalidLayer(String),

    #[error("Material not found: {0}")]
    MaterialNotFound(String),

    #[error("Numerical error: {0}")]
    NumericalError(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("GDSII parse error: {0}")]
    Gds(String),

    #[error("Convergence error: {0}")]
    Convergence(String),
}

pub type Result<T> = std::result::Result<T, OxiPhotonError>;
