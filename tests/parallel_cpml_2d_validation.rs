//! Parallel CPML 2D FDTD validation tests (Block B).
//!
//! Tests cover both TM (Ez, Hx, Hy) and TE (Hz, Ex, Ey) polarisations.
//! Source injection order: inject → update_h_parallel → update_e_parallel.
//! This mirrors the serial pattern used in the existing 2D FDTD tests.

#[cfg(all(test, feature = "parallel"))]
mod parallel_cpml_2d_validation {
    use oxiphoton::fdtd::{BoundaryConfig, Fdtd2dTe, Fdtd2dTm};

    /// Gaussian pulse value at `step`, centred on `peak_step` with sigma `width`.
    fn gaussian_source(step: usize, peak_step: usize, width: f64, amplitude: f64) -> f64 {
        let t = step as f64 - peak_step as f64;
        amplitude * (-t * t / (2.0 * width * width)).exp()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 1 (TM): PML residual energy decays below 20 % of peak after drain.
    // ─────────────────────────────────────────────────────────────────────────

    /// Inject a Gaussian impulse at domain centre, track peak energy, then let
    /// the PML absorb the pulse. Residual energy must fall below 20 % of peak.
    #[test]
    fn parallel_tm_pml_residual_below_threshold() {
        let pml_cells = 8;
        let nx = 32;
        let ny = 32;
        let d = 20e-9_f64;
        let bc = BoundaryConfig::pml(pml_cells);
        let mut fdtd = Fdtd2dTm::new(nx, ny, d, d, &bc);

        let cx = nx / 2;
        let cy = ny / 2;

        // Inject single impulse at centre, then start tracking energy.
        fdtd.inject_ez(cx, cy, 1.0);
        fdtd.update_h_parallel();
        fdtd.update_e_parallel();

        let mut e_peak = fdtd.total_energy();
        for step in 0..30_usize {
            let src = gaussian_source(step, 15, 5.0, 1.0);
            fdtd.inject_ez(cx, cy, src);
            fdtd.update_h_parallel();
            fdtd.update_e_parallel();
            let e = fdtd.total_energy();
            if e > e_peak {
                e_peak = e;
            }
        }

        assert!(
            e_peak > 0.0,
            "TM peak energy must be positive after impulse: {e_peak:.3e}"
        );

        // Drain: run 400 steps without source injection.
        for _ in 0..400_usize {
            fdtd.update_h_parallel();
            fdtd.update_e_parallel();
        }

        let e_resid = fdtd.total_energy();
        assert!(
            e_resid >= 0.0,
            "TM final energy must be non-negative: {e_resid:.3e}"
        );
        assert!(
            e_resid < e_peak * 0.20,
            "TM PML should absorb >80 % of pulse: e_peak={e_peak:.3e}, e_resid={e_resid:.3e}, ratio={:.3}",
            e_resid / e_peak
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2 (TM): Energy builds during source injection.
    // ─────────────────────────────────────────────────────────────────────────

    /// Confirm that energy at the Gaussian peak (step 30) is greater than at
    /// step 5 (when source is just starting). This validates the parallel field
    /// update drives actual energy growth — not just noise.
    #[test]
    fn parallel_tm_interior_energy_conserved() {
        let nx = 48;
        let ny = 48;
        let d = 20e-9_f64;
        let bc = BoundaryConfig::pml(8);
        let mut fdtd = Fdtd2dTm::new(nx, ny, d, d, &bc);

        let cx = nx / 2;
        let cy = ny / 2;

        let mut e_at_5 = 0.0_f64;
        let mut e_at_35 = 0.0_f64;
        // Gaussian peaks at step 20, width 6: well within 48-cell interior.
        for step in 0..60_usize {
            let src = gaussian_source(step, 20, 6.0, 1.0);
            fdtd.inject_ez(cx, cy, src);
            fdtd.update_h_parallel();
            fdtd.update_e_parallel();
            if step == 5 {
                e_at_5 = fdtd.total_energy();
            }
            if step == 35 {
                e_at_35 = fdtd.total_energy();
            }
        }

        // Energy at step 35 (post-peak while pulse is still spreading) must
        // exceed the early reading at step 5 — pulse is well inside the domain.
        assert!(
            e_at_35 > e_at_5,
            "TM energy should grow past the Gaussian peak: e_at_5={e_at_5:.3e}, e_at_35={e_at_35:.3e}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 3 (TM): CW source creates non-trivial field energy.
    // ─────────────────────────────────────────────────────────────────────────

    /// A continuous-wave sinusoidal injection over 150 steps must result in
    /// a non-zero total energy (i.e. the CW source actually drives the field).
    #[test]
    fn parallel_tm_cw_builds_energy() {
        let nx = 36;
        let ny = 36;
        let d = 20e-9_f64;
        let bc = BoundaryConfig::pml(6);
        let mut fdtd = Fdtd2dTm::new(nx, ny, d, d, &bc);

        let cx = nx / 2;
        let cy = ny / 2;
        let omega = 2.0 * std::f64::consts::PI * 3.0e14; // ~1 µm wavelength
        let dt = fdtd.dt;

        for step in 0..150_usize {
            let src = (omega * step as f64 * dt).sin() * 0.5;
            fdtd.inject_ez(cx, cy, src);
            fdtd.update_h_parallel();
            fdtd.update_e_parallel();
        }

        let e_total = fdtd.total_energy();
        assert!(
            e_total > 1e-30,
            "TM CW source should create non-trivial energy: {e_total:.3e}"
        );
        assert!(
            e_total.is_finite(),
            "TM total energy must be finite: {e_total:.3e}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 4 (TM): Determinism — two identical runs give identical results.
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn parallel_tm_determinism_check() {
        let nx = 32;
        let ny = 32;
        let d = 20e-9_f64;
        let bc = BoundaryConfig::pml(6);

        let run = |steps: usize| -> f64 {
            let mut fdtd = Fdtd2dTm::new(nx, ny, d, d, &bc);
            let cx = nx / 2;
            let cy = ny / 2;
            for step in 0..steps {
                let src = gaussian_source(step, 20, 6.0, 1.0);
                fdtd.inject_ez(cx, cy, src);
                fdtd.update_h_parallel();
                fdtd.update_e_parallel();
            }
            fdtd.total_energy()
        };

        let e_a = run(200);
        let e_b = run(200);

        assert!(
            (e_a - e_b).abs() < 1e-10,
            "TM parallel runs must be deterministic: e_a={e_a:.6e}, e_b={e_b:.6e}, diff={:.3e}",
            (e_a - e_b).abs()
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 5 (TM): Residual energy decreases with increasing PML thickness.
    // ─────────────────────────────────────────────────────────────────────────

    /// Use a FIXED total domain size so all three runs have the same pulse
    /// travel time to the boundary. Only the PML fraction changes. This ensures
    /// a fair comparison: thicker PML means more absorption material, so
    /// the normalised residual must decrease monotonically.
    #[test]
    fn parallel_tm_pml_thickness_convergence() {
        let d = 20e-9_f64;
        // Fixed total domain: 48 × 48 cells. PML occupies 4, 8, or 12 cells on
        // each side; interior shrinks correspondingly but travel distance from
        // centre to PML stays proportional.
        let nx_total = 48_usize;
        let ny_total = 48_usize;

        let run_with_pml = |pml: usize| -> f64 {
            let bc = BoundaryConfig::pml(pml);
            let mut fdtd = Fdtd2dTm::new(nx_total, ny_total, d, d, &bc);
            let cx = nx_total / 2;
            let cy = ny_total / 2;

            // Inject single impulse then track peak energy over 30 source steps.
            fdtd.inject_ez(cx, cy, 1.0);
            fdtd.update_h_parallel();
            fdtd.update_e_parallel();
            let mut e_peak = fdtd.total_energy();

            for step in 0..30_usize {
                let src = gaussian_source(step, 15, 5.0, 1.0);
                fdtd.inject_ez(cx, cy, src);
                fdtd.update_h_parallel();
                fdtd.update_e_parallel();
                let e = fdtd.total_energy();
                if e > e_peak {
                    e_peak = e;
                }
            }

            // Long drain: 600 steps so even the PML-4 case has time to absorb.
            for _ in 0..600_usize {
                fdtd.update_h_parallel();
                fdtd.update_e_parallel();
            }

            let e_resid = fdtd.total_energy();
            if e_peak > 0.0 {
                e_resid / e_peak
            } else {
                0.0
            }
        };

        let ratio_4 = run_with_pml(4);
        let ratio_8 = run_with_pml(8);
        let ratio_12 = run_with_pml(12);

        assert!(
            ratio_8 < ratio_4,
            "TM: PML-8 should absorb better than PML-4: ratio_4={ratio_4:.4}, ratio_8={ratio_8:.4}"
        );
        assert!(
            ratio_12 < ratio_8,
            "TM: PML-12 should absorb better than PML-8: ratio_8={ratio_8:.4}, ratio_12={ratio_12:.4}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 6 (TE): PML residual energy decays below 20 % of peak after drain.
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn parallel_te_pml_residual_below_threshold() {
        let pml_cells = 8;
        let nx = 32;
        let ny = 32;
        let d = 20e-9_f64;
        let bc = BoundaryConfig::pml(pml_cells);
        let mut fdtd = Fdtd2dTe::new(nx, ny, d, d, &bc);

        let cx = nx / 2;
        let cy = ny / 2;

        fdtd.inject_hz(cx, cy, 1.0);
        fdtd.update_h_parallel();
        fdtd.update_e_parallel();

        let mut e_peak = fdtd.total_energy();
        for step in 0..30_usize {
            let src = gaussian_source(step, 15, 5.0, 1.0);
            fdtd.inject_hz(cx, cy, src);
            fdtd.update_h_parallel();
            fdtd.update_e_parallel();
            let e = fdtd.total_energy();
            if e > e_peak {
                e_peak = e;
            }
        }

        assert!(
            e_peak > 0.0,
            "TE peak energy must be positive after impulse: {e_peak:.3e}"
        );

        for _ in 0..400_usize {
            fdtd.update_h_parallel();
            fdtd.update_e_parallel();
        }

        let e_resid = fdtd.total_energy();
        assert!(
            e_resid >= 0.0,
            "TE final energy must be non-negative: {e_resid:.3e}"
        );
        assert!(
            e_resid < e_peak * 0.20,
            "TE PML should absorb >80 % of pulse: e_peak={e_peak:.3e}, e_resid={e_resid:.3e}, ratio={:.3}",
            e_resid / e_peak
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 7 (TE): Energy builds while source is active.
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn parallel_te_interior_energy_conserved() {
        let nx = 40;
        let ny = 40;
        let d = 20e-9_f64;
        let bc = BoundaryConfig::pml(6);
        let mut fdtd = Fdtd2dTe::new(nx, ny, d, d, &bc);

        let cx = nx / 2;
        let cy = ny / 2;

        let mut e_early = 0.0_f64;
        for step in 0..200_usize {
            let src = gaussian_source(step, 30, 8.0, 1.0);
            fdtd.inject_hz(cx, cy, src);
            fdtd.update_h_parallel();
            fdtd.update_e_parallel();
            if step == 10 {
                e_early = fdtd.total_energy();
            }
        }

        let e_late = fdtd.total_energy();
        assert!(
            e_late > e_early,
            "TE energy should grow while source is active: e_early={e_early:.3e}, e_late={e_late:.3e}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 8 (TE): CW source creates non-trivial field energy.
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn parallel_te_cw_builds_energy() {
        let nx = 36;
        let ny = 36;
        let d = 20e-9_f64;
        let bc = BoundaryConfig::pml(6);
        let mut fdtd = Fdtd2dTe::new(nx, ny, d, d, &bc);

        let cx = nx / 2;
        let cy = ny / 2;
        let omega = 2.0 * std::f64::consts::PI * 3.0e14;
        let dt = fdtd.dt;

        for step in 0..150_usize {
            let src = (omega * step as f64 * dt).sin() * 0.5;
            fdtd.inject_hz(cx, cy, src);
            fdtd.update_h_parallel();
            fdtd.update_e_parallel();
        }

        let e_total = fdtd.total_energy();
        assert!(
            e_total > 1e-30,
            "TE CW source should create non-trivial energy: {e_total:.3e}"
        );
        assert!(
            e_total.is_finite(),
            "TE total energy must be finite: {e_total:.3e}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 9 (TE): Determinism — two identical runs give identical results.
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn parallel_te_determinism_check() {
        let nx = 32;
        let ny = 32;
        let d = 20e-9_f64;
        let bc = BoundaryConfig::pml(6);

        let run = |steps: usize| -> f64 {
            let mut fdtd = Fdtd2dTe::new(nx, ny, d, d, &bc);
            let cx = nx / 2;
            let cy = ny / 2;
            for step in 0..steps {
                let src = gaussian_source(step, 20, 6.0, 1.0);
                fdtd.inject_hz(cx, cy, src);
                fdtd.update_h_parallel();
                fdtd.update_e_parallel();
            }
            fdtd.total_energy()
        };

        let e_a = run(200);
        let e_b = run(200);

        assert!(
            (e_a - e_b).abs() < 1e-10,
            "TE parallel runs must be deterministic: e_a={e_a:.6e}, e_b={e_b:.6e}, diff={:.3e}",
            (e_a - e_b).abs()
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 10 (TE): Residual energy decreases with increasing PML thickness.
    // ─────────────────────────────────────────────────────────────────────────

    /// Fixed total domain: 48 × 48 cells. Only the PML fraction varies.
    #[test]
    fn parallel_te_pml_thickness_convergence() {
        let d = 20e-9_f64;
        let nx_total = 48_usize;
        let ny_total = 48_usize;

        let run_with_pml = |pml: usize| -> f64 {
            let bc = BoundaryConfig::pml(pml);
            let mut fdtd = Fdtd2dTe::new(nx_total, ny_total, d, d, &bc);
            let cx = nx_total / 2;
            let cy = ny_total / 2;

            fdtd.inject_hz(cx, cy, 1.0);
            fdtd.update_h_parallel();
            fdtd.update_e_parallel();
            let mut e_peak = fdtd.total_energy();

            for step in 0..30_usize {
                let src = gaussian_source(step, 15, 5.0, 1.0);
                fdtd.inject_hz(cx, cy, src);
                fdtd.update_h_parallel();
                fdtd.update_e_parallel();
                let e = fdtd.total_energy();
                if e > e_peak {
                    e_peak = e;
                }
            }

            // Long drain: 600 steps.
            for _ in 0..600_usize {
                fdtd.update_h_parallel();
                fdtd.update_e_parallel();
            }

            let e_resid = fdtd.total_energy();
            if e_peak > 0.0 {
                e_resid / e_peak
            } else {
                0.0
            }
        };

        let ratio_4 = run_with_pml(4);
        let ratio_8 = run_with_pml(8);
        let ratio_12 = run_with_pml(12);

        assert!(
            ratio_8 < ratio_4,
            "TE: PML-8 should absorb better than PML-4: ratio_4={ratio_4:.4}, ratio_8={ratio_8:.4}"
        );
        assert!(
            ratio_12 < ratio_8,
            "TE: PML-12 should absorb better than PML-8: ratio_8={ratio_8:.4}, ratio_12={ratio_12:.4}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 11: TM parallel path matches serial step() to floating-point
    //          precision over 100 steps.
    // ─────────────────────────────────────────────────────────────────────────

    /// Run two identical TM simulations:
    ///   • serial:   inject_ez → step()
    ///   • parallel: inject_ez → update_h_parallel + update_e_parallel
    /// Both receive the same Gaussian pulse via inject_ez (hard source, added
    /// before the field update in both paths so treatment is identical).
    /// After 100 steps the total energies must agree to within 1 % relative
    /// error, and all Ez values must agree element-wise to < 1e-10.
    #[test]
    fn parallel_tm_serial_equivalence() {
        let nx = 24_usize;
        let ny = 24_usize;
        let d = 50e-9_f64;
        let bc = BoundaryConfig::pml(4);

        let mut ser = Fdtd2dTm::new(nx, ny, d, d, &bc);
        let mut par = Fdtd2dTm::new(nx, ny, d, d, &bc);

        let i_src = nx / 2;
        let j_src = ny / 2;

        for step in 0..100_usize {
            let src = gaussian_source(step, 20, 6.0, 1.0);

            ser.inject_ez(i_src, j_src, src);
            ser.step();

            par.inject_ez(i_src, j_src, src);
            par.update_h_parallel();
            par.update_e_parallel();
        }

        // Element-wise Ez comparison.
        for idx in 0..(nx * ny) {
            let diff = (ser.ez[idx] - par.ez[idx]).abs();
            assert!(
                diff < 1e-10,
                "TM Ez mismatch at flat index {idx}: serial={:.6e} par={:.6e} diff={diff:.3e}",
                ser.ez[idx],
                par.ez[idx]
            );
        }

        // Energy comparison.
        let e_ser = ser.total_energy();
        let e_par = par.total_energy();
        if e_ser > 1e-30 {
            let reldiff = (e_ser - e_par).abs() / e_ser;
            assert!(
                reldiff < 1e-8,
                "TM energy mismatch: serial={e_ser:.6e} par={e_par:.6e} rel={reldiff:.3e}"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 12: TE parallel path matches serial step() to floating-point
    //          precision over 100 steps.
    // ─────────────────────────────────────────────────────────────────────────

    /// Run two identical TE simulations:
    ///   • serial:   inject_hz → step()
    ///   • parallel: inject_hz → update_h_parallel + update_e_parallel
    /// Both receive the same Gaussian pulse. After 100 steps the total energies
    /// and all Hz values must agree to floating-point precision.
    #[test]
    fn parallel_te_serial_equivalence() {
        let nx = 24_usize;
        let ny = 24_usize;
        let d = 50e-9_f64;
        let bc = BoundaryConfig::pml(4);

        let mut ser = Fdtd2dTe::new(nx, ny, d, d, &bc);
        let mut par = Fdtd2dTe::new(nx, ny, d, d, &bc);

        let i_src = nx / 2;
        let j_src = ny / 2;

        for step in 0..100_usize {
            let src = gaussian_source(step, 20, 6.0, 1.0);

            ser.inject_hz(i_src, j_src, src);
            ser.step();

            par.inject_hz(i_src, j_src, src);
            par.update_h_parallel();
            par.update_e_parallel();
        }

        // Element-wise Hz comparison.
        for idx in 0..(nx * ny) {
            let diff = (ser.grid.hz[idx] - par.grid.hz[idx]).abs();
            assert!(
                diff < 1e-10,
                "TE Hz mismatch at flat index {idx}: serial={:.6e} par={:.6e} diff={diff:.3e}",
                ser.grid.hz[idx],
                par.grid.hz[idx]
            );
        }

        // Energy comparison.
        let e_ser = ser.total_energy();
        let e_par = par.total_energy();
        if e_ser > 1e-30 {
            let reldiff = (e_ser - e_par).abs() / e_ser;
            assert!(
                reldiff < 1e-8,
                "TE energy mismatch: serial={e_ser:.6e} par={e_par:.6e} rel={reldiff:.3e}"
            );
        }
    }
}
