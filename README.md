# oxiphoton — Pure Rust Computational Photonics Framework

[![crates.io](https://img.shields.io/crates/v/oxiphoton.svg)](https://crates.io/crates/oxiphoton)
[![docs.rs](https://docs.rs/oxiphoton/badge.svg)](https://docs.rs/oxiphoton)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![MSRV: 1.75+](https://img.shields.io/badge/MSRV-1.75+-orange.svg)]()
[![CI](https://img.shields.io/badge/CI-passing-brightgreen.svg)]()

> A comprehensive, pure-Rust computational photonics and optical simulation framework covering FDTD, BPM, mode solving, silicon photonics, fiber optics, quantum photonics, and more — with zero C/Fortran dependencies.

## Overview

**oxiphoton** is a full-stack photonics simulation library written entirely in Rust. It provides production-grade implementations of the core algorithms used in modern photonic design and research: Finite-Difference Time-Domain (FDTD) in 2D and 3D, the Beam Propagation Method (BPM), Transfer Matrix / S-matrix methods, Rigorous Coupled-Wave Analysis (RCWA), finite-difference and FEM mode solvers, and geometric ray tracing — alongside a rich material database, type-safe unit system, and device library for silicon photonics.

The library is organized into 63 public modules spanning over 516 Rust source files (~145 K lines of implementation). It is part of the COOLJAPAN ecosystem and depends only on pure-Rust crates: linear algebra via **oxiblas**, FFT via **oxifft**, and standard scientific helpers. There are no C, Fortran, or unsafe-heavy native bindings in the default build.

With 4,304 passing tests and 10 runnable examples, oxiphoton is designed for correctness and ergonomics. The API is strongly typed throughout — wavelengths, frequencies, refractive indices, field components, and geometric quantities are all distinct types with compile-time unit safety and automatic conversions.

## Status (v0.1.3 — 2026-06-16)

| Metric | Value |
|--------|-------|
| Version | 0.1.3 |
| Release date | 2026-06-16 |
| Rust source files | 516 |
| Lines of code | 145,200 (178,677 total) |
| Public modules | 63 |
| Public API items | 7,595 |
| Public types / traits | 1,086 |
| Passing tests | 4,304 (zero failures) |
| Runnable examples | 10 |
| Criterion benchmarks | 4 |
| C/Fortran dependencies | 0 |
| MSRV | Rust 1.75, Edition 2021 |

## Key Features

- **FDTD 2D/3D** — Yee-cell engine, CPML absorbing boundaries, Bloch periodic BCs, dispersive media (Drude, Lorentz ADE), nonlinear FDTD (Kerr, Raman, SHG), anisotropic and thermal extensions, sweep infrastructure, field/flux/DFT/far-field monitors
- **Frequency-domain solvers** — Transfer Matrix Method (TMM), S-matrix eigenmode expansion, RCWA for gratings, finite-difference BPM
- **Mode solvers** — 2D finite-difference, FEM, coupled-mode theory, effective-index method
- **Silicon photonics device library** — waveguides (single-mode, multimode, tapered), directional couplers, ring/disk resonators, MZI modulators, plasma-dispersion modulators, photodetectors (PIN, avalanche), AWG, coupled-resonator chains
- **Fiber optics** — step-index, graded-index, photonic-crystal fibers; NLSE split-step solver; Bragg gratings; soliton propagation; supercontinuum generation; distributed sensing
- **Photonic crystals** — 2D/3D band structure, defect modes, DOS, slab PHC, topological PHC
- **Quantum photonics** — Fock states, boson sampling, Hong-Ou-Mandel effect, linear optical circuits, Jaynes-Cummings model, photon statistics, entanglement measures, quantum memory
- **Nonlinear optics** — nonlinear crystals (SHG, DFG, OPA), nonlinear microscopy (SHG/CARS imaging), ultrafast pulse characterization (FROG, SPIDER, autocorrelation, pulse shaping)
- **Dispersive materials** — Sellmeier, Drude, Drude-Lorentz, Cauchy, Brendel-Bormann, critical-point models; built-in database for Si, SiO2, Si3N4, TiO2, Au, Ag, Al, GaAs, InP, graphene, and more
- **Specialized domains** — plasmonics, metamaterials, metasurfaces/metalenses, topological photonics, polarimetry (Jones/Mueller), thin films, adaptive optics, structured light, frequency combs, laser rate equations, nano-lasers, polaritonics, THz photonics, X-ray optics
- **Applied photonics** — solar cell absorption (AM1.5), WDM interconnects, optical networks, free-space optical comms, optical CDMA, PIC design and simulation
- **Sensing and measurement** — optical coherence tomography (OCT), metrology, biophotonics, photoacoustics, photonic sensors, beam quality (M² analysis)
- **I/O** — Touchstone (.sNp), GDS layout, VTK, HDF5, Lumerical, OxiRS, STL
- **Inverse design** — adjoint gradient method, fabrication constraints, parametric/shape/topology optimization
- **Performance** — optional Rayon parallelism (`parallel` feature), SIMD acceleration (`simd` feature), GPU compute via wgpu (`gpu-wgpu` feature)

## Module Overview

| Group | Modules | Description |
|-------|---------|-------------|
| Foundations | `units`, `material`, `geometry`, `error`, `prelude` | Type-safe units, material models, mesh primitives |
| FDTD | `fdtd` | 2D/3D FDTD engine, CPML, Bloch BCs, dispersive/nonlinear/anisotropic/thermal extensions, sources, monitors, sweep, analysis |
| Frequency-domain | `smatrix`, `bpm` | TMM, RCWA, eigenmode expansion, BPM |
| Mode solving | `mode` | FD 2D, FEM, CMT, effective-index solvers |
| Silicon photonics | `devices` | Waveguides, couplers, resonators, modulators, detectors, AWG, MZI |
| Fiber optics | `fiber` | Step/graded/PCF fibers, NLSE, Bragg gratings, solitons, supercontinuum, sensing |
| Photonic crystals | `photonic_crystal` | Band structure, defect modes, DOS, slabs, topological PHC |
| Ray optics | `ray` | Ray tracing, aberrations, illumination |
| Quantum photonics | `quantum_photonics`, `quantum_optics`, `entanglement`, `quantum_memory`, `single_photon` | Fock states, boson sampling, HOM, JC model |
| Nonlinear optics | `nonlinear_crystal`, `nonlinear_microscopy`, `ultrafast` | SHG/CARS, FROG, SPIDER, pulse shaping |
| Specialized | `plasmonics`, `metamaterials`, `metasurface`, `topological_photonics`, `polarimetry`, `thin_film`, `adaptive_optics`, `structured_light`, `frequency_comb`, `laser`, `nanolaser`, `polaritonics`, `thz`, `xray` | Domain-specific simulation modules |
| Applied | `solar`, `interconnect`, `optical_network`, `comms`, `fso`, `optical_cdma`, `pic_design`, `pic_simulation` | System-level and application modules |
| Sensing | `metrology`, `oct`, `biophotonics`, `photoacoustics`, `photonic_sensors`, `beam_quality` | Measurement and sensing tools |
| Other | `coherence`, `diffractive`, `mems`, `nearfield`, `amplifiers`, `optical_force`, `optical_trapping`, `sdm`, `temporal_photonics`, `microwave_photonics`, `optical_computing`, `photonic_antenna`, `photonic_dsp`, `detector` | Additional specialized modules |
| I/O | `io` | Touchstone, GDS, VTK, HDF5, Lumerical, OxiRS, STL |
| Inverse design | `inverse` | Adjoint, fabrication constraints, parametric/shape/topology optimization |

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `fdtd` | yes | 2D FDTD engine: Yee solver, CPML, sources, monitors, dispersive and nonlinear media |
| `fdtd-3d` | yes | 3D FDTD extension with anisotropic, thermal, and advanced boundary support |
| `bpm` | yes | Finite-difference Beam Propagation Method |
| `smatrix` | yes | S-matrix and Transfer Matrix Method (TMM) |
| `rcwa` | yes | Rigorous Coupled-Wave Analysis for gratings and periodic structures |
| `mode-solver` | yes | Finite-difference and FEM waveguide mode solvers |
| `ray-optics` | yes | Geometric ray tracing, ABCD matrices, aberration analysis |
| `dispersive` | yes | Dispersive media models: Drude, Lorentz, Sellmeier, Cauchy, Brendel-Bormann |
| `nonlinear` | no | Nonlinear FDTD: Kerr, Raman, SHG |
| `photonic-crystal` | yes | Band structure calculations, defect modes, DOS |
| `siph-devices` | yes | Silicon photonics device library |
| `metalens` | no | Metasurface and metalens design tools |
| `fiber` | yes | Fiber optics: NLSE, solitons, supercontinuum, PCF, sensing |
| `interconnect` | no | WDM channel planning, link budget calculation |
| `solar-optics` | yes | Solar cell absorption with AM1.5 spectrum |
| `inverse-design` | no | Adjoint-based inverse design |
| `topology-opt` | no | Topology optimization |
| `materials-builtin` | yes | Built-in material database (Si, SiO2, Au, Ag, GaAs, InP, graphene, …) |
| `io-gds` | no | GDS layout I/O |
| `io-vtk` | no | VTK file I/O |
| `io-hdf5` | no | HDF5 file I/O |
| `io-oxirs` | no | OxiRS file format I/O |
| `parallel` | no | Rayon multi-threading for FDTD and mode solvers |
| `simd` | no | SIMD acceleration for inner loops |
| `gpu-wgpu` | no | GPU compute via wgpu |

## Installation

Add to your `Cargo.toml` with default features:

```toml
[dependencies]
oxiphoton = "0.1"
```

To enable specific optional features:

```toml
[dependencies]
oxiphoton = { version = "0.1", features = ["parallel", "simd"] }
```

To use only a minimal subset:

```toml
[dependencies]
oxiphoton = { version = "0.1", default-features = false, features = ["smatrix", "materials-builtin"] }
```

## Quick Start

### Unit conversions and physical constants

```rust
use oxiphoton::prelude::*;

// Construct wavelength and convert to frequency and wave number
let wl = Wavelength::from_nm(1550.0);          // telecom C-band
let freq = wl.to_frequency();                   // Frequency in Hz
let k0 = wl.to_wavenumber();                    // vacuum wave number in rad/m

println!("λ = {:.1} nm", wl.as_nm());
println!("f = {:.4e} Hz", freq.0);
println!("k₀ = {:.4e} rad/m", k0.0);
println!("c = {:.6e} m/s", SPEED_OF_LIGHT);
```

### Refractive index and material properties

```rust
use oxiphoton::prelude::*;

// Complex refractive index of a lossy material
let gold = RefractiveIndex::new(0.55, 11.5);    // Au at ~1550 nm
let eps = gold.to_permittivity_scalar();
let alpha = gold.absorption_coefficient(1550e-9);

println!("ε = {:.3} + {:.3}i", eps.re, eps.im);
println!("α = {:.4e} m⁻¹", alpha);

// Lossless dielectric
let sio2 = RefractiveIndex::real(1.444);
println!("n(SiO2) = {}", sio2.n);
```

### Transfer Matrix Method — antireflection coating

```rust
use oxiphoton::prelude::*;

// Quarter-wave SiO2 AR coating on Si at 1550 nm
let layers = vec![
    Layer::from_boxed(Box::new(Sellmeier::sio2()), 268e-9),  // λ/4 in SiO2
];

let wavelengths: Vec<Wavelength> = (1400..=1700)
    .step_by(10)
    .map(|nm| Wavelength::from_nm(nm as f64))
    .collect();

let results = TransferMatrix::spectrum(
    &layers,
    RefractiveIndex::real(1.0),    // air
    RefractiveIndex::real(3.48),   // Si substrate
    &wavelengths,
    Angle(0.0),
    Polarization::TE,
);

for (wl, r) in wavelengths.iter().zip(&results) {
    println!("{:.0} nm  R={:.4}  T={:.4}", wl.as_nm(), r.reflectance, r.transmittance);
}
```

## Examples

The `examples/` directory contains runnable demonstrations:

| Example | Description |
|---------|-------------|
| `anisotropic_crystal` | FDTD simulation of wave propagation in an anisotropic crystal |
| `bloch_band_structure` | Photonic crystal band structure with Bloch boundary conditions |
| `fdtd_3d_waveguide` | 3D FDTD simulation of a dielectric waveguide |
| `fdtd_gpu_acceleration` | CPU vs GPU FDTD comparison (2D TE + 3D), requires `gpu-wgpu` feature |
| `graphene_plasmon` | Graphene surface plasmon polariton simulation |
| `ring_modulator` | Silicon ring modulator frequency response |
| `ring_resonator` | Ring resonator transmission spectrum via TMM |
| `s_parameter_extraction` | S-parameter extraction from FDTD with Touchstone output |
| `solar_cell_absorption` | Solar cell absorption under AM1.5 spectrum |
| `waveguide_mode` | Waveguide mode field profiles via FD mode solver |

Run any example with:

```sh
cargo run --example ring_resonator
```

## Architecture

oxiphoton is built on the COOLJAPAN pure-Rust ecosystem:

```
oxiblas 0.2.1  (pure-Rust BLAS/LAPACK)
oxifft  0.3.2  (pure-Rust FFT)
    └── oxiphoton 0.1.3
```

**Core dependencies:**

| Crate | Version | Role |
|-------|---------|------|
| `num-complex` | 0.4 | Complex number arithmetic |
| `serde` | 1.0 | Serialization (optional on most types) |
| `thiserror` | 2.0 | Ergonomic error types |
| `bytemuck` | 1.25 | Safe transmutes for GPU/SIMD buffers |
| `log` | 0.4 | Structured logging |
| `oxiblas` | 0.2.1 | Pure-Rust BLAS/LAPACK (no OpenBLAS) |
| `oxifft` | 0.3.2 | Pure-Rust FFT (no FFTW) |
| `rayon` | 1.11 | Data parallelism (optional, `parallel` feature) |

There are zero C or Fortran dependencies in the default feature set. All numerical kernels — matrix factorization, FFT, eigensolvers — are implemented in pure Rust via the COOLJAPAN ecosystem crates.

## Performance

- **Parallelism**: Enable the `parallel` feature to activate Rayon-based threading across FDTD time steps, mode solver iterations, and spectrum sweeps. Scales linearly with core count on large grids.
- **SIMD**: The `simd` feature enables explicit SIMD intrinsics in Yee-cell update loops (x86-64 AVX2 and ARM NEON).
- **GPU**: The `gpu-wgpu` feature offloads FDTD field updates to the GPU via wgpu compute shaders, targeting both Vulkan and Metal backends.
- **Benchmarks**: Four Criterion benchmark suites cover 2D FDTD throughput, TMM spectral sweep, mode solver convergence, and GPU FDTD acceleration. Run with `cargo bench`.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

Copyright 2026 COOLJAPAN OU (Team Kitasan). All rights reserved.

Repository: https://github.com/cool-japan/oxiphoton
Documentation: https://docs.rs/oxiphoton

## Contributing

Contributions are welcome. Please open an issue before submitting large pull requests, and ensure all tests pass (`cargo test --all-features`) and code is formatted (`cargo fmt`) before review.
