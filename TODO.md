# OxiPhoton Development Roadmap

## Phase 1: Foundation -- Transfer Matrix + Materials + Units (Current)

- [x] Project scaffold and Cargo.toml
- [x] Error types (`src/error.rs`)
- [x] Units module (`src/units/`)
  - [x] Wavelength, Frequency, WaveNumber newtypes
  - [x] RefractiveIndex, Permittivity, Permeability
  - [x] ElectricField, MagneticField, Intensity, Poynting
  - [x] Angle, NumericalAperture, FocalLength
  - [x] Physical constants and conversion traits
- [x] Material module (`src/material/`)
  - [x] DispersiveMaterial trait
  - [x] Sellmeier model (SiO2, Si, Si3N4, TiO2, GaAs, InP, MgF2)
  - [x] Drude metal model
  - [x] Drude-Lorentz model (Au, Ag, Al)
  - [x] Cauchy model
  - [x] Tabulated (n,k) interpolation
  - [x] MaterialDatabase with built-in materials
  - [x] PML stub
- [x] Transfer Matrix Method (`src/smatrix/`)
  - [x] Layer and ConstantMaterial types
  - [x] TMM solve (single wavelength/angle)
  - [x] TMM spectrum (wavelength sweep)
  - [x] TE/TM polarization
  - [x] Angle-dependent (Snell's law)
  - [x] Dispersive material support
- [x] Validation tests
  - [x] Fresnel equations (R=0.04, Brewster, TIR)
  - [x] Bragg mirror (R > 0.99)
  - [x] AR coating (R ~ 0.012)
  - [x] Energy conservation (R+T+A=1)
- [x] Benchmarks (smatrix_bench)
- [x] README.md and TODO.md

## Phase 2: 1D/2D FDTD ✓

- [x] Geometry primitives
- [x] Yee grid data structure
- [x] 1D FDTD engine
- [x] 2D FDTD (TE/TM)
- [x] CPML boundary
- [x] Sources (plane wave, Gaussian, TFSF)
- [x] DFT monitor (oxifft integration)
- [x] Flux monitor
- [x] Courant condition
- [x] Mie scattering validation

## Phase 3: BPM + Mode Solver + Devices ✓

- [x] Finite-difference mode solver
- [x] Effective index method
- [x] FFT-BPM
- [x] FD-BPM
- [x] Waveguide devices (slab, ridge, strip)
- [x] Couplers (directional, MMI)
- [x] Ring resonator
- [x] Grating coupler

## Phase 4: 3D FDTD + RCWA

- [ ] 3D FDTD engine
- [ ] Dispersive FDTD (ADE)
- [ ] Bloch periodic boundary
- [ ] Far-field transform
- [ ] RCWA
- [ ] EME (Eigenmode Expansion)

## Phase 5: Inverse Design + Advanced

- [ ] Adjoint method
- [ ] Topology optimization
- [ ] Fabrication constraints
- [ ] Photonic crystal band structure
- [ ] Fiber modes and nonlinear propagation
- [ ] Optical interconnect modeling
- [ ] Solar cell optics
- [ ] Metalens design
- [ ] Ray tracing
- [ ] I/O (GDSII, VTK, HDF5)
