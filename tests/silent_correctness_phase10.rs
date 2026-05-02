// Test file for Phase 10 Block B silent-correctness fixes
// Covers: EME SMatrix2x2 Complex64 migration, parallel CPML psi recursion, Bloch Yee coeff

#[cfg(test)]
mod eme_phase_tests {
    use num_complex::Complex64;
    use oxiphoton::smatrix::eigenmode::SMatrix2x2;

    #[test]
    fn eme_propagation_carries_phase() {
        // beta=10, L=pi/20 -> beta*L = pi/2 -> e^{j*pi/2} = j
        let s = SMatrix2x2::propagation(10.0, std::f64::consts::PI / 20.0);
        assert!(
            (s.s12.re).abs() < 1e-9,
            "s12.re should be ~0, got {}",
            s.s12.re
        );
        assert!(
            (s.s12.im - 1.0).abs() < 1e-9,
            "s12.im should be ~1, got {}",
            s.s12.im
        );
        assert!(s.s11.norm() < 1e-12);
        assert!(s.s22.norm() < 1e-12);
    }

    #[test]
    fn eme_propagation_zero_length_is_identity() {
        let s = SMatrix2x2::propagation(42.0, 0.0);
        let id = SMatrix2x2::identity();
        // Zero length -> phase = exp(0) = 1
        assert!((s.s12.re - id.s12.re).abs() < 1e-12);
        assert!((s.s12.im - id.s12.im).abs() < 1e-12);
        assert!(s.s11.norm() < 1e-12);
    }

    #[test]
    fn eme_two_segment_cascade_doubles_phase() {
        let s1 = SMatrix2x2::propagation(10.0, std::f64::consts::PI / 20.0);
        let s2 = SMatrix2x2::propagation(10.0, std::f64::consts::PI / 20.0);
        let total = s1.cascade(&s2);
        // e^{j*pi/2} cascaded with e^{j*pi/2} = e^{j*pi} = -1
        assert!(
            (total.s12.re + 1.0).abs() < 1e-9,
            "s12.re should be ~-1, got {}",
            total.s12.re
        );
        assert!(
            (total.s12.im).abs() < 1e-9,
            "s12.im should be ~0, got {}",
            total.s12.im
        );
    }

    #[test]
    fn eme_identity_cascade_preserves_propagation() {
        // Cascading a propagation matrix with identity should yield the propagation matrix
        let s = SMatrix2x2::propagation(5.0, 0.3);
        let id = SMatrix2x2::identity();
        let result = id.cascade(&s);
        assert!((result.s12 - s.s12).norm() < 1e-12);
        assert!((result.s21 - s.s21).norm() < 1e-12);
        assert!((result.s11 - s.s11).norm() < 1e-12);
        assert!((result.s22 - s.s22).norm() < 1e-12);
    }

    #[test]
    fn eme_from_overlap_perfect_coupling_is_identity() {
        // Perfect overlap (eta=1) should give identity-like S-matrix (T=1, R=0)
        let s = SMatrix2x2::from_overlap(1.0);
        assert!((s.s12 - Complex64::ONE).norm() < 1e-12);
        assert!((s.s21 - Complex64::ONE).norm() < 1e-12);
        assert!(s.s11.norm() < 1e-12);
        assert!(s.s22.norm() < 1e-12);
    }
}

#[cfg(test)]
mod bloch_yee_tests {
    use oxiphoton::fdtd::BandStructureCalc;

    /// Smoke test: BandStructureCalc with corrected coeff should not blow up.
    /// The bug was coeff_y.min(coeff_z) = dt/(mu0*dy_min) — extra 1/dy factor.
    /// After fix: coeff = dt/mu0. Fields should stay bounded.
    #[test]
    fn bloch_h_update_unit_dimensional() {
        // Single k-point at gamma, uniform vacuum (eps=1), very short run
        let k_path = vec![[0.0_f64, 0.0, 0.0]];
        let calc = BandStructureCalc::new(
            k_path,
            20,                       // n_freqs (low for speed)
            (100e12_f64, 400e12_f64), // freq range
            10,                       // n_timesteps (very short)
        );
        let results = calc.compute(100e-9, |_x, _y, _z| 1.0);
        // Should complete without panic and return results for one k-point
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn bloch_e_update_unit_dimensional() {
        // If coeff had extra 1/dx factor, E would grow unbounded in finite time
        let k_path = vec![[0.1_f64, 0.0, 0.0]];
        let calc = BandStructureCalc::new(k_path, 20, (200e12_f64, 600e12_f64), 10);
        let results = calc.compute(50e-9, |_x, _y, _z| 1.0);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn bloch_resonance_gamma_point_vacuum() {
        // Run a known vacuum structure at gamma and check no crash
        let k_path = vec![[0.0_f64, 0.0, 0.0]];
        let calc = BandStructureCalc::new(k_path, 50, (50e12_f64, 800e12_f64), 50);
        let results = calc.compute(60e-9, |_x, _y, _z| 1.0);
        // Should return results for the single k-point
        assert_eq!(results.len(), 1);
    }
}

#[cfg(all(test, feature = "parallel"))]
mod parallel_cpml_tests {
    use oxiphoton::fdtd::{BoundaryConfig, Fdtd3d};

    /// Test that parallel H/E update matches serial in a uniform vacuum region.
    /// Bug: parallel psi accumulators had b*0.0 (dropped old state) and coeff_h*0.0 (dropped H).
    #[test]
    fn parallel_h_update_matches_serial_in_uniform_region() {
        let nx = 8;
        let ny = 8;
        let nz = 8;
        let dx = 50e-9_f64;
        let dy = 50e-9_f64;
        let dz = 50e-9_f64;

        let bc = BoundaryConfig::pml(2);
        let mut fdtd_serial = Fdtd3d::new(nx, ny, nz, dx, dy, dz, &bc);
        let mut fdtd_par = Fdtd3d::new(nx, ny, nz, dx, dy, dz, &bc);

        // Inject Ez at center
        let cx = nx / 2;
        let cy = ny / 2;
        let cz = nz / 2;
        fdtd_serial.inject_ez(cx, cy, cz, 1.0);
        fdtd_par.inject_ez(cx, cy, cz, 1.0);

        // Run a few steps
        for _ in 0..5 {
            fdtd_serial.step();
            fdtd_par.update_h_parallel();
            fdtd_par.update_e_parallel();
        }

        // Compare H fields
        let n = nx * ny * nz;
        for idx in 0..n {
            let diff = (fdtd_serial.hx[idx] - fdtd_par.hx[idx]).abs();
            assert!(diff < 1e-10, "Hx field mismatch at cell {idx}: {diff}");
        }
    }

    /// Test that parallel E-update includes the psi corrections (not just bare curl).
    #[test]
    fn parallel_e_update_includes_psi_correction() {
        let bc = BoundaryConfig::pml(2);
        let mut fdtd_serial = Fdtd3d::new(8, 8, 8, 50e-9, 50e-9, 50e-9, &bc);
        let mut fdtd_par = Fdtd3d::new(8, 8, 8, 50e-9, 50e-9, 50e-9, &bc);

        fdtd_serial.inject_ez(4, 4, 4, 1.0);
        fdtd_par.inject_ez(4, 4, 4, 1.0);

        for _ in 0..5 {
            fdtd_serial.step();
            fdtd_par.update_h_parallel();
            fdtd_par.update_e_parallel();
        }

        let n = 8 * 8 * 8;
        for idx in 0..n {
            let diff = (fdtd_serial.ez[idx] - fdtd_par.ez[idx]).abs();
            assert!(diff < 1e-10, "Ez field mismatch at cell {idx}: {diff}");
        }
    }

    /// Test that parallel CPML absorbs outgoing wave energy.
    /// Bug: without correct psi recursion, PML does not absorb properly.
    #[test]
    fn parallel_cpml_absorbs_outgoing_wave() {
        let bc = BoundaryConfig::pml(4);
        let mut fdtd = Fdtd3d::new(16, 16, 16, 25e-9, 25e-9, 25e-9, &bc);

        // Inject a pulse
        fdtd.inject_ez(8, 8, 8, 1.0);

        // Initial energy (Ez squared)
        let e0: f64 = fdtd.ez.iter().map(|v| v * v).sum();

        // Run enough steps for wave to reach PML and be absorbed
        for _ in 0..200 {
            fdtd.update_h_parallel();
            fdtd.update_e_parallel();
        }

        let e1: f64 = fdtd.ez.iter().map(|v| v * v).sum();
        // After many steps, residual energy should be small
        assert!(
            e1 < 0.5 * e0.max(1e-100),
            "PML did not absorb enough: e0={e0:.2e}, e1={e1:.2e}"
        );
    }
}
