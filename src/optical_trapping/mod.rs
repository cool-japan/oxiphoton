//! Optical Trapping and Manipulation Module
//!
//! Provides comprehensive simulation tools for optical trapping physics:
//!
//! - **forces**: Optical gradient and scattering forces in Rayleigh (dipole) and
//!   Mie regimes. Includes polarizability, scattering cross-sections, and
//!   analytical Gaussian beam trap fields.
//!
//! - **trap**: Trap stiffness characterization (equipartition, PSD methods),
//!   dual-beam counter-propagating traps, and 3D optical potential landscapes
//!   with Kramers escape rate estimation.
//!
//! - **brownian**: Brownian dynamics via the overdamped Langevin equation
//!   (Euler-Maruyama integrator), Stokes drag, Einstein diffusion relation,
//!   MSD computation, power spectral density analysis, and Faxén wall corrections.
//!
//! - **photophoresis**: Photophoretic force model for gas-phase absorbing particles
//!   (Rohatschek interpolation), thermophoretic force (Epstein/Brock), and
//!   levitation intensity estimation.
//!
//! # Physical Regime Overview
//!
//! | Module        | Regime                         | Key Methods                        |
//! |---------------|--------------------------------|-------------------------------------|
//! | forces        | Rayleigh: a ≪ λ                | `polarizability`, `gradient_force`  |
//! | forces        | Mie: arbitrary a/λ             | `q_ext`, `radiation_pressure_force` |
//! | trap          | Harmonic well characterization | `stiffness_from_variance`           |
//! | brownian      | Stochastic Langevin dynamics   | `LangevinSimulator::run`            |
//! | photophoresis | Gas-phase photophoresis        | `PhotophoreticForce::force`         |
//!
//! # Example
//! ```rust,ignore
//! use oxiphoton::optical_trapping::forces::{GaussianTrap, RayleighParticle};
//! use oxiphoton::optical_trapping::brownian::{stokes_drag, LangevinSimulator};
//!
//! let trap = GaussianTrap::new(0.1, 1064e-9, 1.33, 1.2); // 100 mW, NA=1.2
//! let particle = RayleighParticle::new(100e-9, 1.59, 1.33, 1064e-9);
//! let k_r = trap.radial_stiffness(&particle);
//!
//! let gamma = stokes_drag(100e-9, 1e-3);
//! let mut sim = LangevinSimulator::with_drag(300.0, gamma, [k_r, k_r, k_r * 0.1], 1e-5);
//! let trajectory = sim.run(10000);
//! let msd = LangevinSimulator::mean_square_displacement(&trajectory, 500);
//! ```

pub mod brownian;
pub mod forces;
pub mod photophoresis;
pub mod trap;

// Core force types and Gaussian trap
pub use forces::{GaussianTrap, MieParticle, RayleighParticle};

// Trap characterization and potential landscape
pub use trap::{DualBeamTrap, OpticalPotential, TrapCharacterization};

// Brownian dynamics
pub use brownian::{diffusion_coefficient, faxen_drag_correction, stokes_drag, LangevinSimulator};

// Photophoresis and thermophoresis
pub use photophoresis::{thermophoretic_force, PhotophoreticForce};
