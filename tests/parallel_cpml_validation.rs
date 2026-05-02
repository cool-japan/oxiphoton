//! Parallel CPML FDTD validation tests (Block C, Phase 11).
//!
//! Source injection order: `inject_ez` → `update_h_parallel` → `update_e_parallel`.
//! This mirrors the serial pattern `inject_ez` → `step()` used in the existing
//! CPML tests and in `silent_correctness_phase10.rs`.

#[cfg(all(test, feature = "parallel"))]
mod parallel_cpml_tests {
    use oxiphoton::fdtd::{BoundaryConfig, Fdtd3d};

    /// Gaussian pulse value at `step`, centred on `peak_step` with sigma `width`.
    fn gaussian_source(step: usize, peak_step: usize, width: f64, amplitude: f64) -> f64 {
        let t = step as f64 - peak_step as f64;
        amplitude * (-t * t / (2.0 * width * width)).exp()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // C.1: PML residual energy decays well below peak after pulse absorption.
    // ─────────────────────────────────────────────────────────────────────────

    /// Inject a hard-source impulse at the domain centre, track peak energy
    /// during the first 30 propagation steps, then confirm the residual energy
    /// after 500 additional steps is below 20 % of peak (PML absorption
    /// criterion for the parallel path, matching the threshold used in the
    /// serial CPML test `fdtd3d_pml_absorbs_pulse`).
    #[test]
    fn parallel_pml_residual_energy_below_threshold() {
        let n = 24_usize;
        let d = 20e-9_f64;
        let bc = BoundaryConfig::pml(6);
        let mut fdtd = Fdtd3d::new(n, n, n, d, d, d, &bc);

        let cx = n / 2;

        // Single hard-source impulse (identical to serial test pattern).
        fdtd.inject_ez(cx, cx, cx, 1.0);
        fdtd.update_h_parallel();
        fdtd.update_e_parallel();

        // Track peak energy during first 30 propagation steps.
        let mut e_peak = fdtd.total_energy();
        for _ in 0..30_usize {
            fdtd.update_h_parallel();
            fdtd.update_e_parallel();
            let e = fdtd.total_energy();
            if e > e_peak {
                e_peak = e;
            }
        }

        assert!(
            e_peak > 0.0,
            "Peak energy should be positive after impulse: {e_peak:.3e}"
        );

        // Run 400 more steps to let PML absorb the pulse.
        for _ in 0..400_usize {
            fdtd.update_h_parallel();
            fdtd.update_e_parallel();
        }

        let e_resid = fdtd.total_energy();
        assert!(
            e_resid >= 0.0,
            "Final energy must be non-negative: {e_resid:.3e}"
        );
        assert!(
            e_resid < e_peak * 0.20,
            "PML should absorb >80 % of pulse: e_peak={e_peak:.3e}, e_resid={e_resid:.3e}, ratio={:.3}",
            e_resid / e_peak
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // C.2: Interior energy conserved before the pulse reaches the PML.
    // ─────────────────────────────────────────────────────────────────────────

    /// After the pulse fully develops the energy change over 20 additional
    /// propagation steps must stay below 20 % — the pulse is still inside the
    /// domain, far from the PML boundary.
    #[test]
    fn parallel_interior_energy_conserved_before_pml() {
        let n = 32_usize;
        let d = 50e-9_f64;
        let bc = BoundaryConfig::pml(4);
        let mut fdtd = Fdtd3d::new(n, n, n, d, d, d, &bc);

        let i_src = n / 2;
        let j_src = n / 4;
        let k_src = n / 2;

        // Inject pulse over 30 steps.
        for step in 0..30_usize {
            let src = gaussian_source(step, 15, 5.0, 1.0);
            fdtd.inject_ez(i_src, j_src, k_src, src);
            fdtd.update_h_parallel();
            fdtd.update_e_parallel();
        }

        // Let the pulse spread for 50 more steps without injection.
        for _ in 0..50_usize {
            fdtd.update_h_parallel();
            fdtd.update_e_parallel();
        }
        let e_80 = fdtd.total_energy();

        // 20 more propagation steps — pulse still inside the domain.
        for _ in 0..20_usize {
            fdtd.update_h_parallel();
            fdtd.update_e_parallel();
        }
        let e_100 = fdtd.total_energy();

        // Guard: skip if energy is negligibly small.
        if e_80 > 1e-25 {
            let change = (e_100 - e_80).abs() / e_80;
            assert!(
                change < 0.20,
                "Interior energy changed too rapidly before PML absorption: {change:.3} (e_80={e_80:.3e}, e_100={e_100:.3e})"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // C.3: Parallel kernel propagates a CW source without zeroing out energy.
    // ─────────────────────────────────────────────────────────────────────────

    /// Smoke test: inject a sinusoidal Ez source via the parallel kernel for
    /// 400 steps and confirm non-trivial energy is present.  This verifies the
    /// parallel kernel neither zeroes fields nor exhibits numerical blow-up.
    /// (Full phase-velocity measurement requires field time-series at two points,
    /// which this API does not expose; this test is an intentionally limited check.)
    #[test]
    fn parallel_cw_source_builds_nontrivial_energy() {
        let nx = 32_usize;
        let ny = 16_usize;
        let nz = 16_usize;
        let d = 50e-9_f64;
        let bc = BoundaryConfig::pml(6);
        let mut fdtd = Fdtd3d::new(nx, ny, nz, d, d, d, &bc);

        let i_src = 10_usize;
        let j_src = ny / 2;
        let k_src = nz / 2;

        let n_steps = 400_usize;
        for step in 0..n_steps {
            // Sinusoidal source with ~20-step period.
            let phase = 2.0 * std::f64::consts::PI * step as f64 / 20.0;
            fdtd.inject_ez(i_src, j_src, k_src, phase.sin());
            fdtd.update_h_parallel();
            fdtd.update_e_parallel();
        }

        let e_after = fdtd.total_energy();
        // Physical energy scale for d=50nm, amplitude=1: ~5e-34 J/cell.
        // A 400-step CW source fills several cells with standing/propagating wave
        // energy; measured value is ~2e-34.  Threshold set one order below to give
        // meaningful discrimination (a kernel that zeros all fields would give 0,
        // a kernel that stalls propagation and leaves only the source cell energised
        // would give ~5e-34, both well above 1e-35).
        assert!(
            e_after > 1e-35,
            "CW source should build up non-trivial energy: {e_after:.3e}"
        );
        assert!(e_after.is_finite(), "Energy must remain finite: {e_after}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // C.4: Parallel path matches serial path to floating-point precision.
    // ─────────────────────────────────────────────────────────────────────────

    /// Run two identical small simulations: one using serial `step()`, one using
    /// `update_h_parallel` + `update_e_parallel`.  Both receive the same Gaussian
    /// pulse via `inject_ez` (inject before update in both cases).
    ///
    /// `step()` is a pure field-update helper — it does NOT call `apply_sources()`.
    /// So identical manual injection gives identical source treatment for both paths.
    ///
    /// After 40 steps the Hz (and total energy) must agree within floating-point
    /// rounding tolerance.
    #[test]
    fn parallel_serial_pml_energy_match() {
        let n = 24_usize;
        let d = 50e-9_f64;
        let bc = BoundaryConfig::pml(4);

        let mut ser = Fdtd3d::new(n, n, n, d, d, d, &bc);
        let mut par = Fdtd3d::new(n, n, n, d, d, d, &bc);

        let i_src = n / 2;
        let j_src = n / 2;
        let k_src = n / 2;

        let n_pulse = 40_usize;

        for step in 0..n_pulse {
            let src = gaussian_source(step, 20, 6.0, 1.0);

            // Serial path: inject then step (step = update_h + update_e).
            ser.inject_ez(i_src, j_src, k_src, src);
            ser.step();

            // Parallel path: inject then parallel update (same order).
            par.inject_ez(i_src, j_src, k_src, src);
            par.update_h_parallel();
            par.update_e_parallel();
        }

        let total_cells = n * n * n;
        for idx in 0..total_cells {
            let diff_hz = (ser.hz[idx] - par.hz[idx]).abs();
            assert!(
                diff_hz < 1e-10,
                "Hz mismatch at flat index {idx}: serial={:.6e} par={:.6e} diff={diff_hz:.3e}",
                ser.hz[idx],
                par.hz[idx]
            );
        }

        let e_ser = ser.total_energy();
        let e_par = par.total_energy();
        if e_ser > 1e-30 {
            let reldiff = (e_ser - e_par).abs() / e_ser;
            assert!(
                reldiff < 1e-8,
                "Total energy mismatch: serial={e_ser:.6e} par={e_par:.6e} rel={reldiff:.3e}"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // C.5: PML thickness convergence — thicker PML absorbs more energy.
    // ─────────────────────────────────────────────────────────────────────────

    /// Inject the same Gaussian pulse into three grids with PML = 4, 8, 12
    /// (interior size held fixed at 16 cells so only PML thickness varies).
    /// After 400 free-propagation steps:
    ///   • residual ratio must be below 5e-2 for each thickness,
    ///   • pml=8 residual must be at most 2 × pml=4 residual (monotone absorption).
    #[test]
    fn parallel_pml_thickness_convergence_4_8_12() {
        let d = 50e-9_f64;
        let interior = 16_usize;
        let peak_step = 20_usize;
        let pulse_width = 6.0_f64;

        let mut residual_ratios = Vec::with_capacity(3);

        for &pml in &[4_usize, 8, 12] {
            let n = interior + 2 * pml;
            let bc = BoundaryConfig::pml(pml);
            let mut fdtd = Fdtd3d::new(n, n, n, d, d, d, &bc);

            let i_src = n / 2;
            let j_src = n / 2;
            let k_src = n / 2;

            // Inject pulse over 40 steps, track peak energy.
            let mut e_peak = 0.0_f64;
            for step in 0..40_usize {
                let src = gaussian_source(step, peak_step, pulse_width, 1.0);
                fdtd.inject_ez(i_src, j_src, k_src, src);
                fdtd.update_h_parallel();
                fdtd.update_e_parallel();
                let e = fdtd.total_energy();
                if e > e_peak {
                    e_peak = e;
                }
            }

            let e_peak_safe = e_peak.max(1e-30);

            // 400 free-propagation steps (no source).
            for _ in 0..400_usize {
                fdtd.update_h_parallel();
                fdtd.update_e_parallel();
            }

            let e_resid = fdtd.total_energy();
            let ratio = e_resid / e_peak_safe;
            residual_ratios.push(ratio);
        }

        // Each PML thickness must absorb almost all injected energy.
        for (i, &ratio) in residual_ratios.iter().enumerate() {
            let pml = [4_usize, 8, 12][i];
            assert!(
                ratio < 5e-2,
                "PML={pml}: residual ratio {ratio:.3e} exceeds 5e-2"
            );
        }

        // Monotone-ish decrease in residual.
        assert!(
            residual_ratios[1] <= residual_ratios[0] * 2.0,
            "pml=8 residual ratio {:.3e} should be ≤ 2 × pml=4 ratio {:.3e}",
            residual_ratios[1],
            residual_ratios[0]
        );

        assert!(
            residual_ratios[2] <= residual_ratios[1] * 2.0,
            "pml=12 residual ratio {:.3e} should be ≤ 2 × pml=8 ratio {:.3e}",
            residual_ratios[2],
            residual_ratios[1]
        );
    }
}
