pub mod eigenmode;
pub mod rcwa;
#[allow(clippy::module_inception)]
pub mod smatrix;
pub mod transfer_matrix;

use serde::{Deserialize, Serialize};

/// Polarization state for electromagnetic wave
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Polarization {
    /// Transverse Electric (s-polarization): E-field perpendicular to plane of incidence
    TE,
    /// Transverse Magnetic (p-polarization): H-field perpendicular to plane of incidence
    TM,
}

pub use eigenmode::{
    confinement_loss, coupling_efficiency, effective_loss_db_per_cm, mode_loss_db_per_cm,
    overlap_integral, overlap_matrix, propagation_loss_db, EigenMode, EigenmodePropagator, EmeMode,
    EmeSegment, EmeSolver, SMatrix2x2, SMatrixNd,
};
pub use rcwa::{GratingLayer, RcwaResult, RcwaSolver};
pub use smatrix::SMatrixN;
pub use transfer_matrix::{Layer, TransferMatrix, TransferMatrixResult};
