//! Optical Computing & Photonic Neural Networks
//!
//! Implements MZI mesh architectures, photonic neural networks, optical
//! matrix-vector multiplication, and optical reservoir computing.

pub mod mzi_mesh;
pub mod optical_matrix;
pub mod photonic_nn;
pub mod reservoir;

pub use mzi_mesh::*;
pub use optical_matrix::*;
pub use photonic_nn::*;
pub use reservoir::*;
