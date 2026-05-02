# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-05-03

### Added
- Comprehensive tests for semiconductor models and FDTD simulation validation
- Integration tests for solar drift-diffusion models and spectral response calculations
- Integration tests for adjoint field computation and solar design optimizations
- Integration tests for PML material model, solar optics, thermal convection, and topology optimization
- `GdsReader::parse()` text-format reader with round-trip support for all GDS element types (`GdsBoundary`, `GdsPath`, `GdsSref`, `GdsAref`, `GdsText`)
- Enhanced `GdsWriter` now emits full point geometry (was lossy summary-only)
- `MetasurfaceFunction::Hologram` variant now carries a real phase map with bilinear interpolation; `hologram_from_target()` constructor runs Gerchberg-Saxton algorithm using OxiFFT
- `MetasurfaceFunction::hologram_from_phase_map()` constructor for precomputed phase grids
- `PhotonicNetwork::power_efficiency()` now correctly computes ring-topology power efficiency (previously returned 0.0)
- `HigherOrderSoliton::wavelength_m` field; `fission_products()` now propagates center wavelength to product solitons (was hardcoded 1550 nm)
- `SemiconductorLaser::photon_density_ss()` public method with correct steady-state formula
- `OxirsConnection` (feature `io-oxirs`): real HTTP POST/GET SPARQL implementation using `ureq`; replaces no-op stubs

## [0.1.0] - 2026-03-09

### Added
- Initial release of OxiPhoton, a comprehensive photonics simulation library in pure Rust
- FDTD (Finite-Difference Time-Domain) engine with 2D/3D support, CPML absorbing boundaries, dispersive media (Drude, Lorentz, Kerr, Raman, SHG)
- BPM (Beam Propagation Method) solver
- Mode solvers: finite-difference (FD) and finite-element (FEM) in 1D/2D
- S-matrix and Transfer Matrix Method (TMM) for thin-film optics
- RCWA (Rigorous Coupled-Wave Analysis) for gratings
- Waveguide and fiber optics models (SMF-28, HNLF, DSF presets)
- Nonlinear fiber optics: NLSE solver with split-step Fourier method
- Photonic crystal band structure calculations
- Dispersive material models: Sellmeier (SiO2, Si, Si3N4, GaAs, InP, and more)
- Touchstone (.s2p/.sNp) file I/O
- Inverse design: adjoint method and fabrication constraint handling
- Interconnect models: WDM channel planning, link budget
- Quantum photonics: boson sampling, Fock states, entanglement measures
- Ray optics: aberrations, illumination models
- Solar cell and absorption models
- Polarimetry: Jones and Mueller calculus
- Adaptive optics: Shack-Hartmann wavefront sensor, DM control
- Over 3900 unit tests covering all major components

[0.1.1]: https://github.com/cool-japan/oxiphoton/releases/tag/v0.1.1
[0.1.0]: https://github.com/cool-japan/oxiphoton/releases/tag/v0.1.0
