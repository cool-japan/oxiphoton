# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - Unreleased

### Added

(nothing yet)

## [0.1.2] - 2026-06-16

### Added (Phase 15 — 2026-06-16)
- **GPU-accelerated FDTD via `wgpu` compute shaders** (`gpu-wgpu` feature, off by default): full wgpu 29 compute backend with zero CPU overhead between steps.
  - `Fdtd2dGpu` (`src/fdtd/gpu/fdtd_2d_gpu.rs`): 2D TE GPU solver (Hz/Ex/Ey); CPML psi arrays packed stride-6; 4 compute passes per step (inject, Hz, Ex, Ey).
  - `Fdtd2dTmGpu` (`src/fdtd/gpu/fdtd_2d_tm_gpu.rs`): 2D TM GPU solver (Ez/Hx/Hy); symmetric to TE.
  - `Fdtd3dGpu` (`src/fdtd/gpu/fdtd_3d_gpu.rs`): 3D GPU solver; all 6 fields (Ex,Ey,Ez,Hx,Hy,Hz) and 12 CPML ψ arrays on GPU; E/H packed as `array<vec4<f32>>`; per-pass storage bindings ≤ 8 (portable to WebGPU floor); 3 passes per step (H update → E update → Ez inject). Validated vs `Fdtd3d` oracle at 24³: L∞ < 2 × 10⁻³, L₂ < 1 × 10⁻³ on all 6 fields.
  - `GpuContext` (`src/fdtd/gpu/context.rs`): pollster-driven wgpu device/queue initialisation; `GpuError::NotAvailable` for graceful headless fallback.
  - Buffer helpers in `src/fdtd/gpu/buffers.rs`: `SrcUniform`, `SimDims`, `SimDims3d` (POD uniforms); `storage_rw_field`, `storage_rw_aux`, `uniform_from`, `readback`.
- **GPU example and benchmark**: `examples/fdtd_gpu_acceleration.rs` (CPU vs 2D/3D GPU agreement + timing); `benches/fdtd_gpu_bench.rs` (criterion; 2D TE + 3D 20³ GPU throughput); both adapter-gated.
- **New optional dependencies** (`gpu-wgpu` feature): `wgpu = "29"`, `pollster = "0.4"`.
- `oxifft` updated from `0.3` to `0.3.2`.

### Added (Phase 14 — 2026-06-10)
- **Full delayed Raman response in `NlseSolver`** (`src/fiber/nlse.rs`): replaced first-order T_R·dP/dT frequency-shift with proper Agrawal h_R(t) convolution via FFT (τ₁=12.2 fs, τ₂=32 fs). `dt` now passed through to nonlinear step.
- **Rigorous Mie scattering** (`src/smatrix/mie.rs`, new): Bohren-Huffman Lorenz-Mie with Lentz downward recurrence for D_n(ρ) and upward Riccati-Bessel/Hankel. Complex refractive index. Outputs Q_ext, Q_scat, Q_abs, Q_back. `SphereScatter` and `MieResult` exported from `smatrix`.
- **Genuine bidirectional BPM** (`src/bpm/bidirectional.rs`): `step_backward` (conjugated propagation phase + CN diffraction kernel); fixed convergence criterion; `BidirectionalBpm::reflectance()` computes cumulative Fresnel product (was always 0.0).
- **PWE-computed PhC defect mode** (`src/photonic_crystal/defect.rs`): `H1Defect::bandgap_center_from_pwe()` runs full TE band diagram (Γ→M→K→Γ, n_g=7) via `PhCrystal2d`; `resonance_frequency_rigorous()` uses actual bandgap center (replaces hard-coded 5% offset).
- **`n_bg` parameterized in slow-light nonlinear PhC** (`src/photonic_crystal/nonlinear_phc.rs`): `n_bg: f64` field on `PhCNonlinearEnhancement` and `SlowLightShg`; `::silicon()` convenience constructors; removed 4× hard-coded Si n=3.476.
- **Proper Kerr Picard iteration** (`src/fdtd/engine/nonlinear.rs`): `advance_picard` now iterates only the E-update, converging `ε_eff = ε_r + χ³·E²` self-consistently (max 5 iterations).
- **GNLSE adaptive RK4IP** (`src/fiber/supercontinuum.rs`): `propagate_adaptive(tol)`, `rk4ip_step`, `apply_linear_propagator`, `nl_operator` per Hult JLT 2007 + Sinkin step-doubling.
- **3D TFSF wired into `Fdtd3d`** (`src/fdtd/dims/fdtd_3d.rs`): `set_tfsf()` + k-major E/H corrections; `TfsfSource3d`, `Polarization3d`, `PropagationAxis` exported from `fdtd`.

### Added (Phase 13 — 2026-06-10)
- **Rigorous 1D RCWA** (`src/smatrix/rcwa.rs`): full Fourier-space eigensystem for TE and TM using Li inverse-rule factorization; 4n×4n stable boundary matching; Poynting-flux-weighted diffraction efficiencies. Energy conservation R+T=1 to <1e-3 for lossless gratings.
- **Full-vectorial 2D FD mode solver** (`src/mode/full_vectorial.rs`): `FullVectorialModeSolver2d` + `VectorMode` (Fallahkhair-Li-Murphy JLT 2008 Hx-Hy formulation); classifies quasi-TE/TM via Hx/Hy energy fraction.

### Fixed (Phase 13 — 2026-06-10)
- `bessel_j1` x≥5 branch upgraded from single-term to full DLMF 10.17.3 P/Q asymptotic (10 terms, ~1e-9 accuracy at x=10).
- `optomechanics coupling_enhancement` now uses `|cos(2π·q·fraction)|` with real cavity mode number q.
- Deleted dead `win_norm`/`i_approx` placeholder block in `brownian.rs`.
- `dft_ex_magnitude_sq` in `periodic.rs` replaced with `DftAccumulator` running-DFT struct.
- `io-vtk` and `io-hdf5` feature flags now properly gate their modules in `src/io/mod.rs`.
- Binary GDSII writer/reader now serialize/deserialize `PATHTYPE` and `TEXTHEIGHT`.

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

[0.1.3]: https://github.com/cool-japan/oxiphoton/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/cool-japan/oxiphoton/releases/tag/v0.1.2
[0.1.1]: https://github.com/cool-japan/oxiphoton/releases/tag/v0.1.1
[0.1.0]: https://github.com/cool-japan/oxiphoton/releases/tag/v0.1.0
