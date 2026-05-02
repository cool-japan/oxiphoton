//! Inverse design and adjoint sensitivity analysis for photonic structures.
//!
//! The adjoint method computes the gradient of a figure of merit (FOM)
//! with respect to all design parameters in two FDTD simulations:
//!   1. Forward simulation: compute fields E_fwd from source
//!   2. Adjoint simulation: compute E_adj from adjoint source at monitor
//!
//! The gradient is: dFOM/dε(r) = -2ω/c² · Re[E_fwd(r) · E_adj(r)]
//!
//! This enables gradient-based optimization of complex photonic structures
//! using algorithms like L-BFGS or Adam.
pub mod adjoint;
pub mod adjoint_3d;
pub mod fabrication;
pub mod parametric;
pub mod shape;
pub mod topology;

pub use adjoint::{AdjointOptimizer, AdjointSolver, AdjointSolver2d, DesignRegion, FomGradient};
pub use adjoint_3d::{
    AdjointSolver3d, DesignRegion3d, DesignVariable, FdtdSourceConfig, PortPlane, VectorField3d,
    VectorSourcePattern,
};
pub use fabrication::{CurvaturePenalty, FabricationConstraints};
pub use parametric::{
    ConvergenceHistory, MomentumGD, MultiStart, NelderMead, ParametricProblem, Pso,
};
pub use shape::{LevelSet, ParametricShape};
pub use topology::{continuation_schedule, BinaryProjection, Pseudo2dFom, TopologyOptimizer};
