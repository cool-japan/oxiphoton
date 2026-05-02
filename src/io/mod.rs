//! Input/Output module for simulation data.
//!
//! Supports multiple file formats for photonic simulation data:
//!   - `gds`: GDSII layout (text representation)
//!   - `stl`: STL triangular mesh for 3D geometry
//!   - `vtk`: VTK legacy format for field visualization
//!   - `hdf5`: HDF5-style structured data (text fallback, no C library required)
//!
//! All writers produce self-contained ASCII files suitable for post-processing.

#[cfg(feature = "io-gds")]
pub mod gds_io;
pub mod hdf5_text;
pub mod lumerical;
pub mod oxirs_bridge;
pub mod stl;
pub mod touchstone;
pub mod vtk;

#[cfg(feature = "io-gds")]
pub use gds_io::{GdsBinaryReader, GdsBinaryWriter, GdsTextExporter};
pub use hdf5_text::{Hdf5TextDataset, Hdf5TextFile};
pub use lumerical::{LumericalDomain, LumericalParser, LumericalSimulation};
#[cfg(feature = "io-oxirs")]
pub use oxirs_bridge::OxirsConnection;
pub use oxirs_bridge::{KnowledgeGraph, PhotonicSimExporter, RdfObject, Triple};
pub use stl::{StlMesh, StlTriangle, StlWriter};
pub use touchstone::{
    cascade_two_port, s_to_t_matrix, t_to_s_matrix, TouchstoneFormat, TouchstoneReader,
    TouchstoneWriter,
};
pub use vtk::{VtkField2d, VtkWriter};
