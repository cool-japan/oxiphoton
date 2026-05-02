# OxiPhoton Development Roadmap

## Phase 1: Foundation -- Transfer Matrix + Materials + Units

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

## Phase 4: 3D FDTD + RCWA ✓

- [x] 3D FDTD engine
- [x] Dispersive FDTD (ADE)
- [x] Bloch periodic boundary
- [x] Far-field transform
- [x] RCWA
- [x] EME (Eigenmode Expansion)

## Phase 5: Inverse Design + Advanced

- [x] Adjoint method
- [x] Topology optimization
- [x] Fabrication constraints
- [x] Photonic crystal band structure
- [x] Fiber modes and nonlinear propagation
- [x] Optical interconnect modeling
- [x] Solar cell optics
- [x] Metalens design
- [x] Ray tracing
- [x] I/O (GDSII, VTK, HDF5)

## Phase 6: Refinement & Closure

- [x] PML material model (UPML/CFS-PML profile: σ/κ/α profiles, optimal σ_max, cell_profiles, complex ε_eff)
  - **Goal:** `Pml` becomes the canonical CFS-PML profile authority — closed-form σ(x), κ(x), α(x) per Roden-Gedney, optimal σ_max via Berenger-Roden formula, per-cell profile arrays, and a correct complex-index `DispersiveMaterial` implementation.
  - **Design:** σ(s)=σ_max·(s/d)^m; κ(s)=1+(κ_max-1)·(s/d)^m; α(s)=α_max·(1-s/d)^m_a; σ_opt=(m+1)/(150·π·√ε_r·dx); complex ε_eff(s,ω)=ε_r·(κ+σ/(ε₀·(α+jωκ))). `cell_profiles(n, dx)->PmlCellProfiles`. Legacy `Pml::new()` preserved.
  - **Files:** `src/material/pml.rs` (32→~350 LoC), `src/material/mod.rs`, `tests/pml_material.rs`
  - **Tests:** polynomial_grading_monotone, boundary_values, optimal_sigma_max_matches_berenger_roden, cell_profiles_length_and_continuity, complex_eps_eff_attenuates, legacy_constructor_zero_alpha_kappa_one, dispersive_material_trait_returns_lossy_index
  - **Risk:** Branch cut of √(ε_eff) — select branch with k≥0.

- [x] Topology optimizer step() chain-rule completion
  - **Goal:** `TopologyOptimizer::step()` routes through the existing `filter_adjoint`/`projection_jacobian` infrastructure added in Phase 5. Caller passes `dfom_drbar` (gradient w.r.t projected ρ̄); chain rule applied internally.
  - **Design:** Rename param to `dfom_drbar`; body = `filter_density()` → `projection_jacobian` → element-wise multiply → `filter_adjoint` → steepest-ascent update. Add `step_with_raw_gradient` backward-compat shim.
  - **Files:** `src/inverse/topology.rs`, `tests/topology_opt_chainrule.rs` (extend)
  - **Tests:** step_applies_chain_rule_end_to_end, step_with_raw_gradient_preserves_legacy_semantics
  - **Risk:** Downstream callers that already applied chain rule; mitigated by `step_with_raw_gradient` shim.

- [x] Eye-diagram simulation (sub-scope (c) of optical-interconnect-modeling)
  - **Goal:** PRBS + raised-cosine pulse shaping + FFT through `SiPhLink` S21 cascade + AWGN + 2-UI eye fold → Q-factor / eye-opening / jitter metrics. Textbook Q matches noise-only ideal-channel case.
  - **Design:** LFSR PRBS (orders 7/9/11/15, ITU-T O.150 polynomials); RC filter span ±5 UI; oxifft forward; S21 interpolation at FFT bins; AWGN; UI fold; metrics: Q=(μ₁-μ₀)/(σ₁+σ₀), eye_opening=min(high)−max(low), jitter=std(zero-crossings), ber=0.5·erfc(Q/√2).
  - **Files:** `src/interconnect/eye_diagram.rs` (new, ~600 LoC), `src/interconnect/mod.rs`, `tests/eye_diagram.rs`
  - **Tests:** prbs_balance, raised_cosine_isi_free_at_sample_times, q_factor_matches_textbook_for_noise_only, eye_opening_shrinks_with_bandwidth_limit, jitter_increases_with_osnr_decrease, pam4_eye_has_three_levels, q_factor_to_ber_estimate_matches_erfc
  - **Risk:** oxifft API — read existing usage in bpm/fft_bpm.rs before coding.

- [x] Thermal FDTD convective Robin BC
  - **Goal:** `ThermalSimulator::step()` correctly applies Newton BC −k·∂T/∂n=h·(T−T_amb) on six faces using real dx/dy/dz and per-cell ρ·c_p (not the current dead-code closure with dummy ds=1.0).
  - **Design:** Add `volumetric_heat_capacity: Vec<f64>` field (default 1e6 J/(m³·K)); `set_rho_cp_region()` setter; per-face loop: `T_new[face] += -2·h·dt·(T_old-T_amb)/(rho_cp·ds)`; remove `let _ = (n, apply_face)` dead code; for_silicon()/for_copper() convenience ctors.
  - **Files:** `src/fdtd/engine/thermal.rs` (+~120 LoC), `tests/thermal_convection.rs`
  - **Tests:** steady_state_uniform_heat_with_convection, convection_decay_rate_matches_lumped_capacity, bc_does_nothing_when_h_is_zero, bc_active_on_all_six_faces
  - **Risk:** Robin BC CFL constraint Δt<ρc_p·dx/(2h) — document in method doc-comment.

## Phase 7: Closing the Proposed Follow-ups

- [x] Adjoint forward-field FDTD coupling
  - **Goal:** `AdjointOptimizer::compute_forward_field` returns the actual steady-state E_z field produced by a 2D TM FDTD run on the current design region, sampled at the design wavelength via DFT. Adjoint gradients computed downstream become real-quality (within discretisation).
  - **Design:** Add `DftBox2dTm` to `src/fdtd/monitors/dft.rs` (~80 LoC, mirrors existing TE `DftBox2d`). Add `AdjointSolver2d` struct to `src/inverse/adjoint.rs` that wraps `Fdtd2dTm` with design-region ε mapping, Gaussian-CW source injection, DFT monitoring, and `run_forward(region, source, wavelength) -> Vec<Complex64>`. Rewrite `compute_forward_field` (lines 389–423) to call `AdjointSolver2d::run_forward`. Gate FDTD path on `use_fdtd_forward: bool` (default true); keep old analytic path as `compute_forward_field_analytic`.
  - **Files:** `src/fdtd/monitors/dft.rs`, `src/fdtd/monitors/mod.rs`, `src/inverse/adjoint.rs`, `tests/adjoint_fdtd_forward.rs` (new)
  - **Tests:** `dft_box_2d_tm_records_correct_amplitude`, `forward_field_matches_analytic_for_uniform_region`, `gradient_finite_difference_check_2x2`
  - **Risk:** FDTD runtime in tests (~1 s/run); mitigated by `set_fdtd_steps()` knob.

- [x] Integrated SPDC source with real sinc phase-matching JSA
  - **Goal:** New `IntegratedSpdcSource` struct with waveguide A_eff, unifies the two `PhaseMatchingType` enums, and computes the JSA using real `sinc(ΔkL/2)` phase-matching instead of the simplified GVM ansatz. Returns correct pair rate, Schmidt decomposition, and HOM-dip visibility.
  - **Design:** Remove duplicate `PhaseMatchingType` at `src/entanglement/spdc.rs:145`; `pub use crate::nonlinear_crystal::PhaseMatchingType`. Refit JSA at lines 362–370: Δk(ω_s,ω_i)=k_p(ω_s+ω_i)−k_s(ω_s)−k_i(ω_i)−K_qpm; JSA=α_p·sinc(Δk·L/2). Add `IntegratedSpdcSource { crystal_length, effective_area, phase_matching, pump_wavelength, pump_bandwidth_fwhm, d_eff, n_p, n_s, n_i, n_g_p, n_g_s, n_g_i }` with methods `jsa()`, `schmidt_decomposition()`, `hom_visibility()`, `pair_generation_rate(pump_power_mw)`, `brightness_per_mw_per_nm()`. Pair rate uses waveguide formula with A_eff.
  - **Files:** `src/entanglement/spdc.rs`, `src/entanglement/mod.rs`, `src/nonlinear_crystal/mod.rs`, `tests/integrated_spdc.rs` (new)
  - **Tests:** `phase_matching_type_unified`, `jsa_uses_sinc_phase_matching`, `hom_visibility_unity_for_factorable_jsa`, `pair_rate_scales_linearly_with_pump_power`, `pair_rate_inverse_scales_with_a_eff`, `qpm_period_appears_in_delta_k`
  - **Risk:** Two PhaseMatchingType enums may have different variant lists; reviewed before merge.

- [x] Dammann grating real Fourier-coefficient optimization
  - **Goal:** `DammannGrating::optimize_transitions(n_orders)` returns transition points x_k that genuinely minimise diffraction-order intensity variance via LM iteration on Fourier coefficients. `efficiency()` and `uniformity()` read off optimised positions.
  - **Design:** Implement `fourier_coefficients(transitions, m_max) -> Vec<Complex64>` using `c_m = (1/(jπm))·(1+2·Σ_k(−1)^k·cos(2πm·x_k))`. Cost function J=variance of |c_m|² over non-DC orders + soft monotonicity penalty. LM with 5-point FD Jacobian (h=1e-6); stop at ‖grad‖_∞<1e-10 or 200 iterations. Seed from existing hard-coded table for N≤5. Add `nonuniformity_metric(n_orders) -> f64`. Update `efficiency()` and `uniformity()` to use optimised transitions.
  - **Files:** `src/diffractive/grating.rs`, `tests/dammann_optimization.rs` (new)
  - **Tests:** `optimised_transitions_3_orders_match_published_table`, `optimised_transitions_5_orders_match_published_table`, `efficiency_above_70_percent_for_optimised_design`, `uniformity_above_80_percent_for_optimised_design`, `fourier_coefficients_zero_for_even_orders`, `optimisation_does_not_violate_monotonicity`
  - **Risk:** LM local minima for N≥7; mitigated by 5-restart multi-start.

- [x] Bend-loss Marcuse 1971 cubic-exponent formula
  - **Goal:** `WaveguideBend::bend_loss_db_per_90deg` evaluates the Marcuse 1971 conformal-transformation formula `α_bend(R)=(κ_x²/(β·γ·(1+γ·a)))·exp(2γa)·exp(−(2/3)·(γ³/β²)·R)` [Np/m] using mode parameters from existing `n_eff`, `n_clad`, `core_width`, `wavelength` fields. Returns dB/90° = `α_bend·(πR/2)·10/ln(10)`.
  - **Design:** Add private `mode_parameters() -> (kappa_x, beta, gamma, a)` helper: k0=2π/λ, β=k0·n_eff, κ_x=√(k0²·n_eff²−β²).clamp(0,∞), γ=√(β²−k0²·n_clad²).clamp(1/(100·core_width),∞), a=core_width/2. Guard κ_x²<0 → return INFINITY. Document Marcuse 1971 Bell Syst Tech J eq. 26 in doc-comment. Note first-order accuracy for ridge/strip.
  - **Files:** `src/devices/waveguide/bend.rs`, `tests/bend_loss_marcuse.rs` (new)
  - **Tests:** `marcuse_loss_monotonic_with_radius`, `marcuse_loss_textbook_value_si_strip`, `marcuse_loss_explodes_for_subwavelength_radius`, `marcuse_loss_zero_for_extremely_large_radius`, `mode_parameters_satisfy_dispersion_relation`, `cutoff_returns_finite_or_infinite_consistently`
  - **Risk:** Slab/2D approximation for 3D ridge; documented in doc-comment.

## Phase 8: Closing the Adjoint Loop + Solar-Cell Optics

- [x] **adjoint-compute-adjoint-field** — close the 2D adjoint loop Phase 7 left half-open: add a real, FDTD-backed `compute_adjoint_field` method on `AdjointOptimizer` that symmetrically mirrors the existing `compute_forward_field` (planned 2026-04-27)
  - **Goal**: `AdjointOptimizer::compute_adjoint_field(region, fom_dconj_e)` runs a 2D TM FDTD simulation with adjoint sources at the monitor location (sources weighted by ∂FoM/∂E_z*) and returns the steady-state E_z field — making `compute_gradient(e_fwd, e_adj)` callable end-to-end without the caller having to manufacture `e_adj`. The full `forward + adjoint + gradient` pipeline becomes self-contained.
  - **Design**: Extend `AdjointSolver2d` with `run_adjoint(region, monitor_cells, fom_dconj_e, wavelength) -> Result<Vec<Complex64>>`: same Fdtd2dTm setup; inject adjoint sources at every monitor cell with Gaussian-modulated CW envelope weighted by fom_dconj_e; return DftBox2dTm accumulated E_z. Add `compute_adjoint_field` to `AdjointOptimizer` (FDTD-backed when `use_fdtd_forward`, analytic fallback `compute_adjoint_field_analytic` otherwise). Add `monitor_cells: Vec<(usize, usize)>` field to `AdjointOptimizer`.
  - **Files**: `src/inverse/adjoint.rs` (+180 LoC), `tests/adjoint_compute_adjoint_field.rs` (new, +150 LoC)
  - **Prerequisites**: Phase 7 AdjointSolver2d + DftBox2dTm infrastructure already in tree
  - **Tests**: `compute_adjoint_field_returns_correct_length`, `adjoint_field_decays_away_from_monitor`, `adjoint_reciprocity_check_uniform_region`, `gradient_matches_finite_difference_2x2_with_real_adjoint`
  - **Risk**: Reciprocity test — run 1500 steps minimum; normalisation must match forward source convention

- [x] **solar-ar-back-reflector-am15g** — implement AR coating + back-reflector pair design via TMM sweeping AM1.5G spectrum to maximise J_sc (planned 2026-04-27)
  - **Goal**: `optimize_ar_and_back_reflector(absorber, absorber_thickness, back_reflector, back_thickness, am15g) -> Result<ArBackReflectorDesign>` grid-searches n_ar ∈ [1.2, 2.5] (step 0.05) and d_ar ∈ [50nm, 200nm] (step 5nm); also expose `evaluate_design(ar_n, ar_d, ...) -> ArBackReflectorDesign` for single-point evaluation.
  - **Design**: Stack layout [air → AR(n_ar,d_ar) → absorber(λ-dep ε, d_abs) → back_reflector(complex ε, d_br) → substrate]. For each grid point: sweep λ ∈ [300,1200]nm in 5nm steps, run TMM, compute A(λ) in absorber layer, weight by flux_am15g(λ)·λ/(h·c), integrate to J_sc = q·∫A(λ)·flux(λ)dλ (EQE=A ideal).
  - **Files**: `src/solar/back_reflector.rs` (new, +250 LoC), `src/solar/mod.rs` (re-export), `tests/solar_ar_back_reflector.rs` (new)
  - **Prerequisites**: `SolarSpectrum::am15g()` in src/solar/spectrum.rs (confirmed present)
  - **Tests**: `quarter_wave_ar_minimises_reflection_at_design_wavelength`, `back_reflector_increases_long_wavelength_absorption`, `optimization_finds_better_than_uniform_design`, `jsc_below_amg15_short_circuit_limit_for_si`, `am15g_total_flux_consistent_with_constant`
  - **Risk**: ~150k TMM calls for full grid (~1.5s); EQE=A(λ) documented as ideal assumption

- [x] **solar-light-trapping-rcwa-tmm** — implement front-grating RCWA + back-stack TMM coupling for textured-Si integrated absorption vs Lambertian limit (planned 2026-04-27)
  - **Goal**: `evaluate_textured_absorption(period_m, depth_m, duty_cycle, absorber_thickness_m, am15g, n_orders) -> Result<TexturedAbsorptionResult>` returns jsc_ma_cm2, jsc_planar_ma_cm2, jsc_lambertian_ma_cm2, enhancement_factor, lambertian_fraction, absorption_spectrum.
  - **Design**: RCWA at front 1D grating → order-resolved transmission amplitudes t_m at angles θ_m=arcsin(m·λ/(n_si·Λ)); couple each transmitted order into back-stack TMM; A(λ)=Σ_m|t_m|²·A_TMM(λ,θ_m); J_sc=q·∫A·flux dλ. Clamp θ_m≤89°. Compare to lambertian_jsc_si for lambertian_fraction.
  - **Files**: `src/solar/textured_grating.rs` (new, +350 LoC), `src/solar/mod.rs` (re-export), `tests/solar_textured_grating.rs` (new)
  - **Prerequisites**: crate::rcwa with 1D-grating order-resolved transmission; lambertian_jsc_si accessible
  - **Tests**: `planar_limit_matches_zeroth_order_only`, `textured_enhances_over_planar`, `lambertian_fraction_below_unity`, `weak_grating_recovers_planar`, `n_orders_convergence`
  - **Risk**: Grazing-incidence RCWA — clamp θ_m; use coarse λ grid (every 20nm) in tests

- [x] **volume-grating-full-formulas** — replace Simplified shortcuts in VolumeGrating: (1) `first_order_efficiency_thin` J₁² truncation → full Raman-Nath J_m² order spectrum; (2) `reflection_spectrum` missing off-Bragg δ → full Kogelnik formula with detuning (planned 2026-04-27)
  - **Goal**: `VolumeGrating::diffraction_orders(m_max, wavelength_m) -> Vec<(i32,f64)>` returns η_m=J_m²(ν) for m∈{-m_max,…,m_max}; `VolumeGrating::reflection_spectrum(λ)` evaluates full Kogelnik 1969 off-Bragg formula with δ(λ)=π·d·(1/Λ_g−cos θ_B/λ)·(λ−λ_B)/λ_B². Add `raman_nath_modulation(λ) -> f64`. Keep existing method signatures backward-compatible.
  - **Design**: Bessel J_m via Miller's downward recurrence (A&S 9.1.27, ~50 LoC). Kogelnik two branches: κ²>δ² → sinh²/cosh²-based; δ²>κ² → sin²-based. `first_order_efficiency_thin` body becomes delegate to `diffraction_orders(1,λ)`.
  - **Files**: `src/diffractive/grating.rs` (+200 LoC), `tests/volume_grating_full.rs` (new)
  - **Prerequisites**: bessel_j1 pattern in src/photonic_crystal/pwe2d.rs (reuse recurrence)
  - **Tests**: `bessel_j_m_matches_known_values`, `raman_nath_orders_sum_to_unity`, `raman_nath_first_order_matches_old_thin_formula`, `kogelnik_at_bragg_matches_tanh_squared`, `kogelnik_off_bragg_dips_below_on_bragg`, `kogelnik_reflection_bounded_by_unity`, `raman_nath_orders_symmetric_about_m_zero`
  - **Risk**: Numerical instability for large m — Miller's downward recurrence; both δ²>κ² and κ²>δ² branches tested

## Phase 9: 3D Adjoint Coupling + Topology Eigenpairs + Interconnect Wrappers + Airy Recycling

- [x] **adjoint-3d-fdtd-coupling** — replace the analytic Gaussian × plane-wave shortcuts in `AdjointSolver3d` with a real `Fdtd3d` + `DftMonitor3d` pipeline, closing the 3D adjoint loop symmetrically with the Phase 8 2D version (planned 2026-04-27)
  - **Goal**: `AdjointSolver3d` has FDTD-backed `run_forward(region, wavelength)` and `run_adjoint(region, monitor_cells, fom_dconj_e, wavelength)` driving a full 3D Yee solver with CPML, injecting Ez via `inject_ez`, monitoring via `DftMonitor3d(FieldComp3d::Ez)`, returning steady-state complex Ez over the whole design region. `compute_forward_field`/`compute_adjoint_field` become thin FDTD-vs-analytic dispatchers via `use_fdtd: bool`. Gradient `compute_gradient(e_fwd, e_adj)` closes end-to-end on real 3D fields (Ez-only this round).
  - **Design**: Add `DesignRegion3d` with `epsilon(i,j,k)->f64`, `eps_min`/`eps_max`, `cell_idx(i,j,k)=i+j*nx+k*nx*ny`. Extend `AdjointSolver3d` with `use_fdtd: bool`, `monitor_cells: Vec<(usize,usize,usize)>`, `source_i/j/k: usize`. `run_forward`: create `Fdtd3d::new(nx,ny,nz,dx,dx,dx,BoundaryConfig::pml(clamp(8,nx/2..)))`, stamp ε, inject Ez at source cell with Gaussian-CW at ω=2πc/λ, attach `DftMonitor3d(MonitorRegion3d::SubVolume{full grid}, FieldComp3d::Ez, &[f0])`, run ≥2000 steps, return `Complex64::new(dft_re[0][cell], dft_im[0][cell])` per cell. `run_adjoint`: same setup; inject Ez at each monitor cell weighted by `Re{fom_dconj_e[m]·exp(jωt)}`. Existing analytic bodies become `compute_forward_field_analytic`/`compute_adjoint_field_analytic`. Gradient formula unchanged: `2·Re(e_z_fwd·conj(e_z_adj))·ω²ε₀·dx³·(ε_max−ε_min)`.
  - **Files**: `src/inverse/adjoint.rs` (+400 LoC), `tests/adjoint_3d_fdtd.rs` (new, +200 LoC)
  - **Prerequisites**: `Fdtd3d` (`inject_ez`, `step()`, `BoundaryConfig::pml`) and `DftMonitor3d` already in tree
  - **Tests**: `compute_forward_field_returns_correct_length` (4×4×4→len 64), `forward_field_decays_away_from_source`, `adjoint_reciprocity_check_4x4x4` (within 35%, ≥2000 steps), `gradient_matches_finite_difference_2x2x2_with_real_adjoint` (relative error <30%, ≥2000 steps)
  - **Risk**: CPML thickness clamp to `min(8, nx/2, ny/2, nz/2)`; source-stamp normalisation identical for forward and adjoint

- [x] **photonic-crystal-symmetric-eigensolver** — replace eigenvalues-only Sturm-bisection in `tridiagonal_eigenvalues` with full eigenpairs via `oxiblas::prelude::TridiagEvd`, add Bloch-vector producer for `BerryPhase::compute_wilson_loop` (planned 2026-04-27)
  - **Goal**: `SshPhotonicChain::eigenpairs(k) -> Result<(Vec<f64>, Mat<f64>), OxiPhotonError>` returns eigenvalues + column-major eigenvector matrix. `eigenfrequencies(k) -> Vec<f64>` preserved (delegates to eigenpairs). New `bloch_vector(k, band_index) -> Result<Vec<Complex64>, OxiPhotonError>` and `wilson_loop_band_n(k_path, band_index) -> Result<BerryPhase, OxiPhotonError>` close the Wilson-loop computation end-to-end.
  - **Design**: Add `use oxiblas::prelude::{TridiagEvd, Mat};` to `topology.rs`. Add `eigenpairs_inner(diag, off_diag) -> Result<(Vec<f64>, Mat<f64>), OxiPhotonError>` calling `TridiagEvd::compute`; replace `tridiagonal_eigenvalues` body with a call to it (eigenvalues-only callers still satisfied). `bloch_vector`: index `vecs[(row, band_index)]` for row in 0..n using `Index<(usize,usize)>` (column-major, padded — never assume contiguous rows), cast to `Complex64::new(v, 0.0)`. `wilson_loop_band_n`: collect Bloch vectors for each k, call `BerryPhase::compute_wilson_loop`. Add debug-only orthogonality assertion in `eigenpairs_inner`.
  - **Files**: `src/photonic_crystal/topology.rs` (+150 LoC), `tests/photonic_crystal_eigensolver.rs` (new)
  - **Prerequisites**: `oxiblas` 0.2.1 dep (Cargo.toml:22); `TridiagEvd::compute` re-exported at `oxiblas::prelude`
  - **Tests**: `eigenpairs_returns_orthonormal_basis` (6×6 SSH H, within 1e-6), `eigenfrequencies_unchanged_on_simple_case` (regression within 1e-9), `ssh_topological_phase_pi` (intra<inter → Berry phase ≈π within 0.05 rad), `ssh_trivial_phase_zero` (intra>inter → ≈0), `wilson_loop_independent_of_k_path_density` (50/100/200 k-points agree within 0.01 rad)
  - **Risk**: Inverse-iteration orthogonality for nearly-degenerate bands — safety-net assertion; if fires, escalate upstream to oxiblas not silence

- [x] **optical-interconnect-modeling-wrappers** — add `chip_to_chip_link_response` (multi-stage S-matrix cascade) and `ber_vs_osnr_sweep_for_link` (BER sweep with link-derived dispersion penalty) (planned 2026-04-27)
  - **Goal**: `chip_to_chip_link_response(stages: &[&SiPhLink], freq_grid_hz: &[f64]) -> Vec<[Complex64;4]>` cascades multiple SiPhLinks via Mason's-rule S-matrix composition; empty-stages returns identity matrices. `ber_vs_osnr_sweep_for_link(link, f_center_hz, modulation, bit_rate_gbps, osnr_db_grid) -> Result<BerOsnrCurve, OxiPhotonError>` derives dispersion penalty from `link.insertion_loss_db` at f_center and calls `BerOsnrCurve::compute`.
  - **Design**: `compose_s_matrices(a, b) -> [Complex64;4]` (module-private): `denom = 1 − a[3]·b[0]`; `S11=a[0]+a[1]·b[0]·a[2]/denom`, `S12=a[1]·b[1]/denom`, `S21=b[2]·a[2]/denom`, `S22=b[3]+b[2]·a[3]·b[1]/denom`. Return type is full `[Complex64;4]` not just S21 — preserves return-loss and reverse-isolation. `ber_vs_osnr_sweep_for_link`: get il_db from `link.insertion_loss_db(&[f_center_hz])`, pass as `dispersion_penalty_db` to `BerOsnrCurve::compute`.
  - **Files**: `src/interconnect/sparam_link.rs` (+50 LoC), `src/interconnect/ber_analysis.rs` (+30 LoC), `src/interconnect/mod.rs` (2 re-export lines), `tests/interconnect_composer.rs` (new, +80 LoC)
  - **Prerequisites**: `SiPhLink::cascade`, `SiPhLink::insertion_loss_db`, `BerOsnrCurve::compute`, `ModulationFormat` all in tree
  - **Tests**: `cascade_identity_recovers_single_link`, `cascade_two_lossless_links_doubles_phase`, `cascade_reduces_bandwidth`, `link_response_length_matches_freq_grid`, `ber_sweep_higher_loss_gives_worse_ber`, `ber_sweep_length_matches_osnr_grid`
  - **Risk**: `compose_s_matrices` divides by `(1−a[3]·b[0])` — near resonances this can approach singularity; tests use safely-detuned stages

- [x] **solar-airy-photon-recycling** — delete orphan Airy intermediates discarded with `let _ = (r01, r12, t01, t12, exp_i_delta, exp_2i_delta);` and replace thickness-blind `escape_probability = 1/(4n²)` with Tiedje-Yablonovitch `P_esc(α,d,n) = 1/(4n²) + (1−1/(4n²))·exp(−α·4n²·d)` (planned 2026-04-27)
  - **Goal**: `src/solar/absorption.rs` orphan Airy block removed (lines 295–304, 330). `photon_recycling_factor(q, alpha, thickness, n)` (signature change: `alpha: f64` inserted) implements the Tiedje-Yablonovitch escape probability. All callers updated in this block.
  - **Design**: Delete `let r01 = ...; let r12 = ...; let t01 = ...; let t12 = ...; let exp_i_delta = ...; let exp_2i_delta = ...;` block (lines 295–298, 303–304) and the `let _ = (r01, r12, t01, t12, exp_i_delta, exp_2i_delta);` discard (line 330). Rewrite `photon_recycling_factor`: `four_n2=4n²; escape_prob=1/four_n2 + (1−1/four_n2)·exp(−α·four_n2·d); return 1/(1−q·(1−escape_prob))`. Grep all callers, thread α through. Thin limit (α·4n²·d→0): escape_prob→1 (no recycling). Thick limit (→∞): escape_prob→1/(4n²) (Yablonovitch).
  - **Files**: `src/solar/absorption.rs` (−15/+20 LoC net +5), caller files (~5–10 LoC), `tests/solar_photon_recycling.rs` (new, +80 LoC)
  - **Prerequisites**: none (orphan deletion is a no-op vs matrix-method computation)
  - **Tests**: `recycling_thin_limit_escape_unity`, `recycling_thick_limit_yablonovitch`, `recycling_monotone_in_thickness`, `recycling_monotone_in_alpha`, `recycling_factor_at_least_unity`, `recycling_factor_q_zero_returns_unity`, `airy_orphan_block_removed` (static include_str! regression guard)
  - **Risk**: Caller audit may surface many callers — budget for cascade; `airy_orphan_block_removed` test prevents regression

## Phase 10: Silent-Correctness Sweep + Drift-Diffusion Keystone + Vector Adjoint + Spectral Response

- [x] **silent-correctness-eme-cpml-bloch** — fix three silent-correctness bugs uncovered by Phase 10 audit: (1) `SMatrix2x2::propagation` returns identity regardless of β·L because the struct can't represent complex S-parameters, (2) `Fdtd3d::update_{h,e}_parallel` CPML ψ accumulators have the b·ψ_old recursion term zeroed-out, (3) `bloch::update_{h,e}_3d` use `.min()` of curl coefficients with an extra `1/dx` factor (planned 2026-04-28)
  - **Goal**: (1) `SMatrix2x2` migrates to `Complex64` fields; `propagation(beta,length)` returns `Self { s11: ZERO, s12: e^{jβL}, s21: e^{jβL}, s22: ZERO }`. (2) `update_h_parallel`/`update_e_parallel` mirror serial-path CPML ψ recursion: `psi_new = b*psi_old + c*dcurl`; write back to struct via assign not `+=`. Restore `num/den` conductivity factor in E-update. (3) `bloch::update_h_3d`/`update_e_3d` use `coeff = dt/μ₀` (not `dt/(μ₀·dx)`); no `.min()`; drop the `let _ = (...)` discards.
  - **Design**: B.1 — change `SMatrix2x2 { s11: f64, ... }` to `{ s11: Complex64, ... }`; update `identity()`, `propagation()`, `from_overlap()`, `cascade()`. B.2 — snapshot all parallel-path ψ struct fields into local Vecs before par_iter; in closure, compute `psi_hx_y_new = b[j]*psi_hx_y_snap[idx] + c[j]*dez_dy`; after collect, `self.psi_hx_y[idx] = psi_hx_y_new` (assign); same for E-path's 6 ψ fields; restore `Ex_new = (num/den)*ex_snap[idx] + coeff_curl*(...)`. B.3 — replace `coeff_x/y/z = dt/(MU_0*dx)` triple with single `coeff = dt/MU_0`; `hx -= coeff*(dez_dy - dey_dz)`; same for E-path with `coeff = dt/EPSILON_0`.
  - **Files**: `src/smatrix/eigenmode.rs` (+~80 LoC), `tests/smatrix_validation.rs` (+~5 LoC fix), `src/fdtd/dims/fdtd_3d.rs` (+~100 LoC parallel path), `src/fdtd/boundary/bloch.rs` (+~20 LoC), `tests/silent_correctness_phase10.rs` (new, +150 LoC)
  - **Tests**: `eme_propagation_carries_phase`, `eme_propagation_zero_length_is_identity`, `eme_two_segment_cascade_doubles_phase`, `parallel_h_update_matches_serial_in_uniform_region`, `parallel_e_update_matches_serial_in_uniform_region`, `parallel_cpml_absorbs_outgoing_wave`, `bloch_h_update_unit_dimensional`, `bloch_e_update_unit_dimensional`, `bloch_resonance_unchanged`

- [x] **solar-drift-diffusion-mvp** — implement a Boltzmann-statistics 1D drift-diffusion EQE solver (Sze Ch. 2 fidelity): coupled Poisson + electron + hole continuity with Scharfetter-Gummel flux discretisation, damped Newton with line-search, SRH + radiative + Auger recombination, ohmic BCs, optical generation profile G(z), IV sweep + IQE extractor (planned 2026-04-28)
  - **Goal**: `DriftDiffusionDevice::new(material, doping_profile, thickness, n_grid_points)` constructs a 1D pn-junction. `solve_equilibrium()`, `solve_dark_iv(v_grid) -> Vec<(f64,f64)>`, `solve_illuminated_iv(v_grid, generation_profile)`, `compute_iqe(wavelength, absorber_alpha)`, `extract_jsc_voc(iv_curve)`. Damped Newton with backtracking line-search; convergence ‖F‖_∞ < 1e-8; max 200 iterations. Boltzmann stats. Bernoulli B(x) = x/(e^x-1) with Taylor branch |x|<1e-3. SRH+rad+Auger per cell. Ohmic Dirichlet BCs. No `unwrap()`.
  - **Design**: 3N state vector (ψ_i, n_i, p_i). Poisson residual via Voronoi discretisation. Scharfetter-Gummel flux `J_{n,i+1/2} = -(qD_n/dx)·[B(Δψ/V_T)·n_{i+1} - B(-Δψ/V_T)·n_i]`. SRH: `R = (np - n_i²)/(τ_p(n+n_1) + τ_n(p+p_1))`. Auger: `(C_n·n + C_p·p)·(np-n_i²)`. Block-tridiagonal 3N×3N LU (fallback to dense Gaussian elimination for N≤200). Equilibrium init: quadratic solve for n, p from `n·p=n_i²` and `n-p=N_d-N_a`.
  - **Files**: `src/solar/drift_diffusion/mod.rs` (~150), `material.rs` (~250), `poisson.rs` (~300), `continuity.rs` (~400), `recombination.rs` (~250), `newton.rs` (~450), `src/solar/mod.rs` (add pub mod), `tests/solar_drift_diffusion.rs` (~400)
  - **Tests**: `dark_diode_built_in_voltage_matches_kt_log_na_nd_over_ni_squared`, `depletion_width_matches_textbook_formula`, `equilibrium_satisfies_mass_action_n_p_equals_n_i_squared`, `dark_iv_diode_equation_in_low_bias`, `illuminated_iv_extracts_jsc_voc`, `iqe_matches_textbook_si_short_wavelength_quenching`, `recombination_profile_srh_matches_analytical`, `newton_converges_under_50_iterations_for_dark_equilibrium`, `current_continuity_satisfied_in_neutral_region`
  - **Risk**: Newton convergence — damped line-search; Bernoulli singularity at x=0 — Taylor branch; Boltzmann breaks down >1e19 cm⁻³ — document assumption

- [x] **adjoint-3d-vector-fields** — extend Phase 9 Block B `AdjointSolver3d` (Ez-only) to full (Ex, Ey, Ez) injection / monitoring / gradient; mode-source patterns at port plane; backward-compat: existing Ez-only API preserved (planned 2026-04-28)
  - **Goal**: `AdjointSolver3d::run_forward_vector(region, source: VectorSourcePattern, wavelength) -> Result<VectorField3d, OxiPhotonError>` returns 3-component complex field of length nx·ny·nz each. `run_adjoint_vector` takes per-component fom_dconj_e. `compute_gradient_vector(e_fwd, e_adj)` evaluates `g_v = 2·Re(E_fwd·conj(E_adj))·ω²ε₀·dx³·(ε_max−ε_min)`.
  - **Design**: Add `VectorField3d { ex, ey, ez: Vec<Complex64>; nx, ny, nz: usize }`, `VectorSourcePattern::PointSource { i, j, k, amplitude: [Complex64;3] }`, `VectorSourcePattern::ModeSource { port_plane: PortPlane, mode_pattern: VectorField3d }`, `PortPlane { XLow, XHigh, YLow, YHigh, ZLow, ZHigh }`. 3 DftMonitor3d instances (Ex, Ey, Ez). ≥2000 steps. If `Fdtd3d::inject_ex`/`inject_ey` missing, add them (mirror `inject_ez`). Existing `run_forward` (Ez-only) untouched.
  - **Files**: `src/inverse/adjoint.rs` (+~500 LoC), `src/inverse/mod.rs` (re-export), `tests/adjoint_3d_vector.rs` (new, ~200 LoC)
  - **Tests**: `vector_forward_field_returns_three_components`, `point_source_excites_only_target_component`, `vector_adjoint_reciprocity_check_4x4x4` (within 40%), `vector_gradient_finite_difference_2x2x2` (<35% error), `ez_only_subset_matches_phase9`, `vector_field_3d_indexing`

- [x] **solar-cell-spectral-response** — end-to-end EQE(λ) chaining Phase 8 AR/back-reflector + textured-grating + Phase 9 photon-recycling with self-consistent Würfel emission-reabsorption iteration (planned 2026-04-28)
  - **Goal**: `compute_spectral_response(cell_design: &SolarCellDesign, wavelengths: &[f64], am15g: &SolarSpectrum, recycling_iterations: usize) -> Result<SpectralResponse, OxiPhotonError>`. `SpectralResponse { wavelengths_m, eqe, iqe: Vec<f64>, jsc_ma_cm2, photon_recycling_factor: f64 }`. Self-consistent iteration: `A_{k+1} = A_optical + q·(1-EQE_k)·A_k·(1-escape_prob)`.
  - **Design**: `SolarCellDesign { absorber, ar_coating: Option<ArCoating>, back_reflector: Option<BackReflector>, texturing: Option<TexturingDesign>, temperature_k, quantum_yield }`. Per-λ: call `evaluate_textured_absorption` → order-resolved t_m; for each order call `evaluate_design` at θ_m; sum `A_optical = Σ|t_m|²·A_m`. Recycling via `photon_recycling_factor(q, alpha, thickness, n)`. Self-consistent iterate with relaxation α=0.5 if oscillating. J_sc via trapezoidal rule. IQE = EQE/(1-R).
  - **Files**: `src/solar/spectral_response.rs` (new, ~400 LoC), `src/solar/mod.rs` (re-export), `tests/solar_spectral_response.rs` (new, ~200 LoC)
  - **Tests**: `eqe_lambda_returns_correct_grid_length`, `eqe_below_unity_at_all_wavelengths`, `iqe_above_eqe`, `recycling_iteration_converges_in_5_iterations`, `textured_si_with_recycling_jsc_textbook` (within 15% of 30 mA/cm²), `cell_optimization_increases_jsc_over_baseline`, `recycling_factor_q_zero_no_iteration_change`, `eqe_zero_below_bandgap`

## Phase 11: Spectral-DD Coupling + Fermi-Dirac Stats + EME Orthogonality + Parallel-CPML Validation

- [x] **eme-mode-orthogonality-check** — after the Phase 10 Block B `SMatrix2x2`→`Complex64` migration, add a power-conservation orthogonality test for EME port modes across a cascade (verify `|S₁₁|² + |S₂₁|² ≤ 1` at every frequency) plus mode-overlap orthonormality. Multi-mode N×N S-matrix unitarity is the strongest check; single-segment from_overlap is trivial. (planned 2026-04-28)
  - **Goal**: A new `tests/eme_orthogonality.rs` file with six tests: (B.1) single-segment propagation unitary, (B.1) single-segment from_overlap energy-conserving, (B.1) full-overlap is identity, (B.2) multi-segment cascade below unity via Redheffer chain, (B.2) cascade power conservation at EmeSolver, (B.3) multi-mode N×N S-matrix unitarity S†·S ≈ I within 1e-9.
  - **Design**: Use `SMatrix2x2::propagation`, `from_overlap`, `cascade` from `src/smatrix/eigenmode.rs`. Multi-segment chain: identity→propagation(β=10,L=0.05)→from_overlap(0.9)→propagation(β=12,L=0.07)→from_overlap(0.8)→propagation(β=10,L=0.03). Assert |s11|²+|s21|² ≤ 1+1e-12. Multi-mode: `EigenmodeLayer::to_s_matrix_full()`, assemble 4×4 block S, check S†·S ≈ I within 1e-9.
  - **Files**: `tests/eme_orthogonality.rs` (new, ~200 LoC). No source changes.
  - **Tests**: 6 tests as above.

- [x] **parallel-cpml-3d-validation-suite** — after the Phase 10 Block B parallel-CPML ψ-recursion fix, add comprehensive `--features parallel` validation covering PML decay, interior energy conservation, plane-wave phase velocity, parallel-serial cross-check, PML thickness convergence. (planned 2026-04-28)
  - **Goal**: New `tests/parallel_cpml_validation.rs` gated `#[cfg(all(test, feature = "parallel"))]`. 5 tests: (C.1) PML residual energy decay below threshold, (C.2) interior energy conserved, (C.3) plane-wave phase velocity within 5%, (C.4) parallel-serial cross-check within 1%, (C.5) PML thickness convergence 4/8/12 cells.
  - **Design**: Use `Fdtd3d::update_h_parallel`, `update_e_parallel`, `inject_ez`, `total_energy`, `energy_e`, `energy_h`. Critical: parallel kernels bypass `apply_sources()` — must call `inject_ez(...)` manually each iteration. Mirror `tests/fdtd_3d_cpml.rs` pattern.
  - **Files**: `tests/parallel_cpml_validation.rs` (new, ~250 LoC). No source changes.
  - **Tests**: 5 tests as above.

- [x] **solar-dd-spectral-response-coupling** — bridge Phase 10 Block C (drift-diffusion TCAD) and Block E (EQE spectral response) with depth-resolved TMM optical generation profile driving per-λ DD solve. (planned 2026-04-28)
  - **Goal**: New `compute_spectral_response_dd(cell, device_config, wavelengths_m, am15g) → Result<SpectralResponse>` using depth-resolved Yeh-Hecht TMM for G(z,λ) → DD solve per λ → `EQE_DD(λ) = J_sc/(q·∫G dz)`.
  - **Design**: Add `single_wavelength_absorption_z` (depth-resolved TMM, tracks E_forward/E_backward amplitudes), `DriftDiffusionDeviceConfig`, `material_for_absorber`, `solve_dd_per_lambda`. Sanity: `Σ A_z(z)·dz ≈ absorptance` via debug_assert!. Per-λ: clone device to equilibrium state before each solve.
  - **Files**: `src/solar/spectral_response.rs` (+250 LoC), `tests/solar_spectral_response_dd.rs` (new, ~200 LoC).
  - **Tests**: 6 tests: eqe_below_unity, eqe_zero_below_bandgap, eqe_decreases_short_lambda, jsc_within_factor_of_optical, eqe_increases_with_diffusion_length, coupling_consistent_with_compute_iqe.

- [x] **solar-dd-fermi-dirac-stats** — extend drift-diffusion from Boltzmann to Fermi-Dirac F_{1/2}/F_{−1/2} statistics for degenerate doping > 1e19 cm⁻³. (planned 2026-04-28)
  - **Goal**: New `src/solar/drift_diffusion/fermi_dirac.rs` with `f_half(eta)`, `f_minus_half(eta)`, `joyce_dixon_eta(u)`. Antia (1993) 3-region rational approx (12-decimal accuracy). Joyce-Dixon 4-term inverse + Newton fallback. Conditional FD activation in material.rs (nc_at, nv_at, is_degenerate_for_doping, dn_cm2_s_fd, dp_cm2_s_fd) and degenerate branches in newton.rs + contact_bc in mod.rs.
  - **Files**: `src/solar/drift_diffusion/fermi_dirac.rs` (new ~250 LoC), `material.rs` (+50 LoC), `newton.rs` (+80 LoC), `mod.rs` (+30 LoC), `tests/solar_drift_diffusion_fermi_dirac.rs` (new ~200 LoC).
  - **Tests**: 8 tests: f_half(0) known value, f_minus_half(0) known value, derivative identity, Boltzmann limit, degenerate limit, Joyce-Dixon round-trip, Einstein relation above classical, Boltzmann recovered at low doping.

## Phase 12: BGN + Parallel-CPML-2D + EME Reciprocity + AM0 Spectrum

- [x] **solar-dd-bandgap-narrowing** — Slotboom/Klaassen BGN for heavily-doped Si emitters. (planned 2026-04-28)
  - **Goal**: New `bandgap_narrowing.rs` with `BgnModel { None, Slotboom, Klaassen }` enum and Slotboom-de Graaff 1976 / Klaassen 1992 ΔEg models; `n_ie_squared()` method on `SemiconductorMaterial`; per-node `ni_eff` array threading through Newton equilibrium residual/Jacobian, Gummel recombination call sites, and contact BCs.
  - **Files**: `src/solar/drift_diffusion/bandgap_narrowing.rs` (new), `material.rs` (+BgnModel, n_ie_squared), `newton.rs` (+per-node ni_eff), `mod.rs` (+pub mod, contact_bc update), `tests/solar_drift_diffusion_bgn.rs` (new, 9 tests).

- [x] **parallel-cpml-2d-validation-suite** — Implement `update_*_parallel` on Fdtd2dTm/Fdtd2dTe; 10-test validation suite. (planned 2026-04-28)
  - **Goal**: Add `update_h_parallel`/`update_e_parallel` to both 2D FDTD engines (TM and TE) using the 3D snapshot/par_iter/sequential-writeback pattern; CPML ψ-recursion `psi_new = b*psi_old + c*curl`; add `total_energy()` alias on `Fdtd2dTm`; validate with 10 tests (5 invariants × 2 polarizations: PML residual, energy conservation, CW buildup, determinism, PML thickness convergence).
  - **Files**: `src/fdtd/dims/fdtd_2d.rs` (+parallel methods + alias; run `splitrs` if > 2000 lines), `tests/parallel_cpml_2d_validation.rs` (new, 10 tests, `#[cfg(all(test, feature = "parallel"))]`).

- [x] **eme-multi-mode-scattering-suite** — S=Sᵀ reciprocity regression-guard for time-reversal-symmetric EME layers. (planned 2026-04-28)
  - **Goal**: 4 tests verifying `S = Sᵀ` (transpose, NOT conjugate) on `to_s_matrix_full`, `cascade_smatrices`, and `SMatrix2x2` infrastructure. No source-tree changes.
  - **Files**: `tests/eme_reciprocity.rs` (new, 4 tests).

- [x] **solar-dd-spectral-am15g-vs-am0** — AM0 spectrum builder at 1366 W/m² for space-cell J_sc. (planned 2026-04-28)
  - **Goal**: `SolarSpectrum::am0()` using Planck blackbody at T_eff=5778 K normalised to 1366 W/m² (ASTM E490 solar constant); same wavelength grid as `AM15G_DATA`; compatible with `compute_spectral_response_dd`.
  - **Files**: `src/solar/spectrum.rs` (+am0() constructor), `tests/solar_spectrum_am0.rs` (new, 4 tests).

## Proposed follow-ups

- [x] **solar-dd-fermi-dirac-wiring** — Wire Phase 11 FD scaffolding (`is_degenerate_for_doping`, `dn_cm2_s_fd`, `dp_cm2_s_fd`) into Newton residual and Jacobian. The FD module is correct in isolation but never called from `mod.rs` or `newton.rs`. ~150 LoC. (planned 2026-05-01)
  - **Goal:** Drift-diffusion uses `D_n = μ_n·V_T · 2·F_{1/2}(η)/F_{-1/2}(η)` wherever carrier density exceeds degeneracy threshold; Boltzmann is the fallback, selected per `StatisticsModel` field on `SemiconductorMaterial`.
  - **Design:** Add `enum StatisticsModel { Boltzmann, FermiDirac }` in `material.rs`; set `silicon()` to `Boltzmann`, `gaas()` to `FermiDirac`. Pre-compute per-node `dn_node`/`dp_node` arrays in `gummel_nonequil_solve` (frozen each outer Gummel iter). Modify `solve_n_tridiag`/`solve_p_tridiag` to accept `dn_per_edge`/`dp_per_edge` slices derived via harmonic mean. Bump `max_outer_iters` to 150 when FD is selected.
  - **Files:** `src/solar/drift_diffusion/material.rs`, `src/solar/drift_diffusion/newton.rs`, `src/solar/drift_diffusion/mod.rs`
  - **Tests:** New `tests/dd_fermi_dirac_wiring.rs`: `boltzmann_default_unchanged` (Si regression), `fermi_dirac_n_emitter_lowers_voc` (GaAs degenerate emitter V_oc drops 5–25 mV), `fd_diffusivity_factor_above_unity_at_degeneracy` (η=+5 → factor ≈ 1.5–2.0), `fd_converges_within_iter_cap` (stress-test cell ≤150 iters).
  - **Risk:** GaAs solar-cell V_oc may shift tens of mV when combined with Jain–Roulston BGN; full `tests/solar_*` suite must pass before accepting.
- [x] **eme-overlap-interface-smatrix** — Implement N×N interface S-matrix from inter-segment mode-overlap matrix so `to_s_matrix_full` exhibits real off-diagonal mode coupling. Required prerequisite for meaningful off-diagonal reciprocity tests. ~300 LoC. (planned 2026-05-01)
  - **Goal:** Cascading two `EigenmodeLayer`s of different widths produces non-trivial S11/S22 (reflection between mode subspaces) and correctly coupled S12/S21; `diag_inv_nd` bug in `cascade_smatrices` fixed to use a real matrix inverse.
  - **Design:** New `src/smatrix/interface.rs`: `interface_smatrix(modes_a, modes_b, dx, omega)` computes power-normalised overlap matrix `V[i][j]` and forms classical mode-matching S-matrix (Snyder & Love §31, Bienstman §2.2). Port `mat_inv_m` to `mat_inv_full_nd` for `Vec<Vec<Complex64>>`; replace `diag_inv_nd` calls in `cascade_smatrices`. Add `EmeStack::to_s_matrix_full` cascading layer + interface + layer. Add `EmeMode::power_norm(omega)` without touching existing `norm()`.
  - **Files:** NEW `src/smatrix/interface.rs`, `src/smatrix/eigenmode.rs` (fix cascade + add `power_norm`), `src/smatrix/mod.rs`
  - **Tests:** New `tests/eme_interface_smatrix.rs`: `same_layer_interface_is_identity`, `interface_unitary_lossless`, `step_width_reflection_nonzero`, `cascade_two_steps_returns_to_input`. Tighten existing `tests/eme_reciprocity.rs` to assert non-zero off-diagonal entries on a non-trivial stack.
  - **Risk:** Mode-overlap math sensitive to normalisation convention; validate `same_layer_interface_is_identity` first — if it fails, normalisation is wrong and all else is meaningless.
- [x] **solar-dd-bgn-jain-roulston-gaas** — Extend `BgnModel` with `JainRoulston` variant for GaAs (currently `BgnModel::None` for `gaas()`). ~80 LoC. (planned 2026-05-01)
  - **Goal:** GaAs degenerate-doping simulations use the Jain–Roulston (1991) BGN model; `SemiconductorMaterial::gaas()` defaults to it; `n_ie_squared` dispatch routes correctly.
  - **Design:** Add `BgnModel::JainRoulston` variant and `jain_roulston_gaas_delta_eg_ev(n_total_cm3, doping_type)` helper using form `ΔE_g = A·(N/1e18)^{1/3} + B·(N/1e18)^{1/4} + C·(N/1e18)^{1/2}` with coefficients from Jain & Roulston (1991) *Solid-State Electronics* 34(5):453 Table 2 (n/p-GaAs columns — must be transcribed verbatim, fabrication forbidden). Add dispatch arm in `n_ie_squared`; flip `gaas()` default.
  - **Files:** `src/solar/drift_diffusion/bandgap_narrowing.rs`, `src/solar/drift_diffusion/material.rs`
  - **Tests:** New `tests/bgn_jain_roulston_gaas.rs`: `bgn_zero_doping_returns_zero` (<1 meV at 1e14), `bgn_n_gaas_matches_published_at_1e19` (±5 meV vs Fig. 4), `bgn_p_gaas_larger_than_n_gaas`, `gaas_default_uses_jain_roulston`.
  - **Risk:** Flipping `gaas()` default may shift existing GaAs V_oc tests by 30+ mV; run full `tests/solar_*` and report deltas before accepting.
- [x] **solar-spectrum-am0-astm-e490-table** — Replace Planck-blackbody approximation in `am0()` with canonical ASTM E490 75-entry table for full-spectrum-shape accuracy. ~100 LoC. (planned 2026-05-01)
  - **Goal:** `SolarSpectrum::am0()` returns the canonical ASTM E490 / Wehrli 1985 WRC-615 extra-terrestrial spectrum re-sampled onto the AM15G grid, self-normalised to 1366.0 W/m²; spectral shape becomes physically correct (UV cut, Fraunhofer features).
  - **Design:** Add private `const AM0_E490_DATA: &[(f64, f64)]` (118-entry Wehrli 1985 table, verbatim from NREL ASTM E490-00a — fabrication forbidden). Rewrite `am0()` to build from table, re-sample via existing `interpolate_irradiance`, then self-normalise. Keep `planck_radiance` for downstream callers. Do not touch `total_irradiance()`.
  - **Files:** `src/solar/spectrum.rs`
  - **Tests:** Existing 4 tests in `tests/solar_spectrum_am0.rs` must pass unweakened. Append `am0_uv_visible_ratio_matches_e490` (integrated [200,400 nm]/[400,700 nm] within 10% of Wehrli ratio ~0.205) and `am0_no_blackbody_smoothness` (≥3 local minima in [380,450] nm from Fraunhofer features).
  - **Risk:** Transcription error in 118-entry table; mitigation: include checksum comment asserting total raw integral matches Wehrli's 1366.1 W/m² to within 0.5%.

## Verification (2026-04-27)

Workspace-wide `cargo nextest run --all-features` + `cargo clippy --all-features --all-targets -- -D warnings`: **PASS** (3908 tests, 0 failures, 0 warnings).

| Phase 4/5 item | Test files | Status |
|---|---|---|
| 3D FDTD engine | `tests/fdtd_3d.rs`, `tests/fdtd_3d_validation.rs`, `tests/fdtd_3d_cpml.rs` | ✓ |
| Dispersive FDTD (ADE) | `tests/fdtd_dispersive_3d.rs`, `tests/fdtd_3d_validation.rs` | ✓ |
| Bloch periodic boundary | `tests/bloch_bc.rs`, `tests/fdtd_3d.rs`, `tests/phase5.rs` | ✓ |
| Far-field transform | `tests/fdtd_3d.rs` | ✓ |
| RCWA | `tests/rcwa_validation.rs` | ✓ |
| EME (Eigenmode Expansion) | `tests/rcwa_validation.rs`, `tests/smatrix_validation.rs` | ✓ |
| Adjoint method | `src/inverse/adjoint.rs` (unit tests) | ✓ |
| Fabrication constraints | `tests/inverse_design.rs` | ✓ |
| Metalens design | `tests/phase5.rs`, `src/metasurface/metalens.rs` | ✓ |
| Ray tracing | `tests/phase5.rs`, inline tests | ✓ |
| Fiber modes / nonlinear propagation | `tests/fiber_validation.rs`, `tests/phase5.rs` | ✓ |

## Verification (2026-04-27, run 3)

Post-`cargo upgrade` validation (rayon 1.12 / oxiblas 0.2.1 / proptest 1.11) + Phase 7 items: PASS (4047 tests, 0 failures, 0 warnings).

## Verification (2026-04-27, run 4)
PASS (4075 tests, 0 failures, 0 warnings)

## Verification (2026-04-27, run 5)
PASS (4099 tests, 0 failures, 0 warnings) — Phase 9 (3D Adjoint FDTD coupling, SSH eigenpairs + Wilson-loop, interconnect link composer + BER sweep, Airy orphan deletion + Tiedje-Yablonovitch photon recycling)

## Verification (2026-04-28, run 6)
PASS (4156 tests, 0 failures, 0 warnings) — Phase 10 (SMatrix2x2→Complex64 + parallel CPML ψ fix + Bloch Yee curl fix, 1D drift-diffusion TCAD MVP [Scharfetter-Gummel + Newton + SRH/rad/Auger], AdjointSolver3d vector-field extension, end-to-end EQE(λ) spectral response with self-consistent photon-recycling iteration)

## Verification (2026-04-28, run 7)
PASS (4187+ tests, 0 failures, 0 warnings) — Phase 11 (EME mode orthogonality + power-conservation tests, parallel CPML 3D validation suite, solar DD spectral-response coupling [depth-resolved TMM → DD per λ], solar DD Fermi-Dirac statistics [Antia rational approx + Joyce-Dixon inverse + degenerate branches])

## Verification (2026-05-03)
PASS (4,264 tests, 0 failures, 0 warnings) — v0.1.1 stub-check fixes (GdsReader::parse() text-format reader + GdsWriter full-geometry emit, MetasurfaceFunction::Hologram Gerchberg-Saxton phase retrieval via OxiFFT, PhotonicNetwork::power_efficiency() ring-topology fix, HigherOrderSoliton wavelength propagation through fission, SemiconductorLaser::photon_density_ss() steady-state formula, OxirsConnection HTTP/SPARQL via ureq [feature io-oxirs])
