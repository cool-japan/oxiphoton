/// Integration tests for Phase 10 full-vector (Ex, Ey, Ez) adjoint API.
///
/// These tests exercise `run_forward_vector`, `run_adjoint_vector`,
/// `compute_gradient_vector`, and the new types `VectorField3d`,
/// `VectorSourcePattern`, and `PortPlane`.
#[cfg(feature = "inverse-design")]
mod adjoint_3d_vector_tests {
    use num_complex::Complex64;
    use oxiphoton::inverse::{AdjointSolver3d, DesignRegion3d, VectorField3d, VectorSourcePattern};

    /// Uniform-ε region: ε = eps_target, so ρ = (ε - eps_min) / (eps_max - eps_min).
    fn make_region(nx: usize, ny: usize, nz: usize, dx: f64) -> DesignRegion3d {
        let eps_min = 1.0_f64;
        let eps_max = 12.0_f64;
        let eps_target = 2.25_f64;
        let rho_val = (eps_target - eps_min) / (eps_max - eps_min);
        let mut r = DesignRegion3d::new(nx, ny, nz, dx, eps_min, eps_max);
        for v in &mut r.rho {
            *v = rho_val;
        }
        r
    }

    /// FDTD solver with source at the given cell, reduced step count for speed.
    fn make_solver(nx: usize, ny: usize, nz: usize, dx: f64) -> AdjointSolver3d {
        let mut s = AdjointSolver3d::new_fdtd(nx, ny, nz, dx, nx / 2, ny / 2, nz / 2);
        // Reduce step count so tests finish quickly
        s.n_steps = 200;
        s
    }

    #[test]
    fn vector_forward_field_returns_three_components() {
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let dx = 50e-9;
        let solver = make_solver(nx, ny, nz, dx);
        let region = make_region(nx, ny, nz, dx);
        let source = VectorSourcePattern::PointSource {
            i: 2,
            j: 2,
            k: 2,
            amplitude: [Complex64::new(1.0, 0.0), Complex64::ZERO, Complex64::ZERO],
        };
        let wavelength = 1550e-9;
        let field = solver
            .run_forward_vector(&region, &source, wavelength)
            .expect("run_forward_vector should succeed");
        assert_eq!(field.ex.len(), 64, "ex should have 64 entries for 4x4x4");
        assert_eq!(field.ey.len(), 64);
        assert_eq!(field.ez.len(), 64);
        assert_eq!(field.nx, 4);
    }

    #[test]
    fn point_source_excites_primarily_target_component() {
        // Ez-only source -> Ez should dominate over Ex
        let nx = 6;
        let ny = 6;
        let nz = 6;
        let dx = 50e-9;
        let solver = make_solver(nx, ny, nz, dx);
        let region = make_region(nx, ny, nz, dx);
        let source = VectorSourcePattern::PointSource {
            i: 3,
            j: 3,
            k: 3,
            amplitude: [Complex64::ZERO, Complex64::ZERO, Complex64::new(1.0, 0.0)],
        };
        let field = solver
            .run_forward_vector(&region, &source, 1550e-9)
            .expect("run_forward_vector should succeed");

        let max_ez: f64 = field.ez.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
        let max_ex: f64 = field.ex.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
        // Ez should be significantly larger than Ex for a z-polarized source
        if max_ez > 1e-30 {
            assert!(
                max_ex / max_ez < 0.5,
                "Ex/Ez = {:.3}; expected Ez to dominate",
                max_ex / max_ez
            );
        }
    }

    #[test]
    fn vector_adjoint_reciprocity_check_4x4x4() {
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let dx = 50e-9;
        let solver = make_solver(nx, ny, nz, dx);
        let region = make_region(nx, ny, nz, dx);

        let amp = [Complex64::new(1.0, 0.0), Complex64::ZERO, Complex64::ZERO];
        let source = VectorSourcePattern::PointSource {
            i: 1,
            j: 1,
            k: 1,
            amplitude: amp,
        };
        let e_fwd = solver
            .run_forward_vector(&region, &source, 1550e-9)
            .expect("run_forward_vector should succeed");

        let monitor_cell = (2, 2, 2);
        let fom_amp = [Complex64::new(0.5, 0.5), Complex64::ZERO, Complex64::ZERO];
        let fom_x = vec![fom_amp[0]];
        let fom_y = vec![fom_amp[1]];
        let fom_z = vec![fom_amp[2]];
        let e_adj = solver
            .run_adjoint_vector(&region, &[monitor_cell], &fom_x, &fom_y, &fom_z, 1550e-9)
            .expect("run_adjoint_vector should succeed");

        // reciprocity: Σ_c E_fwd_c(monitor) · fom_c ≈ Σ_c E_adj_c(source) · amp_c
        let midx = e_fwd.cell_idx(2, 2, 2);
        let sidx = e_adj.cell_idx(1, 1, 1);

        let lhs =
            e_fwd.ex[midx] * fom_amp[0] + e_fwd.ey[midx] * fom_amp[1] + e_fwd.ez[midx] * fom_amp[2];
        let rhs = e_adj.ex[sidx] * amp[0] + e_adj.ey[sidx] * amp[1] + e_adj.ez[sidx] * amp[2];

        if lhs.norm() > 1e-30 && rhs.norm() > 1e-30 {
            let rel = (lhs.norm() - rhs.norm()).abs() / lhs.norm().max(rhs.norm());
            assert!(rel < 0.50, "Reciprocity rel error: {rel:.3}");
        }
    }

    #[test]
    fn vector_gradient_finite_difference_2x2x2() {
        let nx = 2;
        let ny = 2;
        let nz = 2;
        let dx = 50e-9;
        let solver = make_solver(nx, ny, nz, dx);
        let mut region = make_region(nx, ny, nz, dx);
        let wavelength = 1550e-9;

        let source = VectorSourcePattern::PointSource {
            i: 0,
            j: 0,
            k: 0,
            amplitude: [Complex64::ZERO, Complex64::ZERO, Complex64::new(1.0, 0.0)],
        };

        let e_fwd = solver
            .run_forward_vector(&region, &source, wavelength)
            .expect("run_forward_vector should succeed");
        let fom_z: Vec<Complex64> = e_fwd.ez.iter().map(|&e| e.conj()).collect();
        let fom_x = vec![Complex64::ZERO; 8];
        let fom_y = vec![Complex64::ZERO; 8];

        let monitor_cells: Vec<(usize, usize, usize)> = (0..8_usize)
            .map(|idx| (idx % 2, (idx / 2) % 2, idx / 4))
            .collect();
        let e_adj = solver
            .run_adjoint_vector(&region, &monitor_cells, &fom_x, &fom_y, &fom_z, wavelength)
            .expect("run_adjoint_vector should succeed");

        let g_adj = solver
            .compute_gradient_vector(&e_fwd, &e_adj, wavelength)
            .expect("compute_gradient_vector should succeed");

        // FD gradient check
        let delta_rho = 0.01_f64;
        let mut max_err = 0.0_f64;
        #[allow(clippy::needless_range_loop)]
        for v in 0..8 {
            let rho_orig = region.rho[v];
            region.rho[v] = (rho_orig + delta_rho).clamp(0.0, 1.0);
            let e_fwd_pert = solver
                .run_forward_vector(&region, &source, wavelength)
                .expect("perturbed run_forward_vector should succeed");
            region.rho[v] = rho_orig;

            let fom_orig: Complex64 = e_fwd.ez.iter().zip(fom_z.iter()).map(|(e, g)| e * g).sum();
            let fom_pert: Complex64 = e_fwd_pert
                .ez
                .iter()
                .zip(fom_z.iter())
                .map(|(e, g)| e * g)
                .sum();

            // Convert rho perturbation to eps perturbation for FD comparison
            let delta_eps = delta_rho * (region.eps_max - region.eps_min);
            let g_fd = (fom_pert.re - fom_orig.re) / delta_eps;
            if g_fd.abs() > 1e-30 {
                let rel = (g_adj[v] - g_fd).abs() / g_fd.abs().max(g_adj[v].abs());
                max_err = max_err.max(rel);
            }
        }
        assert!(max_err < 0.40, "FD vs adjoint max rel error: {max_err:.3}");
    }

    #[test]
    fn ez_only_subset_matches_phase9() {
        // run_forward_vector with Z-polarized source should produce Ez
        // compatible with run_forward (which uses the same source coordinates)
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let dx = 50e-9;
        let solver = make_solver(nx, ny, nz, dx);
        let region = make_region(nx, ny, nz, dx);
        let wavelength = 1550e-9;

        let vector_source = VectorSourcePattern::PointSource {
            i: nx / 2,
            j: ny / 2,
            k: nz / 2,
            amplitude: [Complex64::ZERO, Complex64::ZERO, Complex64::new(1.0, 0.0)],
        };

        let vec_field = solver
            .run_forward_vector(&region, &vector_source, wavelength)
            .expect("run_forward_vector should succeed");

        // Phase 9 Ez-only path (source at source_i, source_j, source_k = nx/2, ny/2, nz/2)
        let ez_only = solver
            .run_forward(&region, wavelength)
            .expect("run_forward should succeed");

        // Both paths should produce Ez fields of similar magnitude.
        // Note: run_forward injects via sin(ω₀t) while run_forward_vector uses
        // Re(amp · exp(iω₀t)) = cos(ω₀t) for a real amplitude — a 90° phase shift.
        // We therefore compare field MAGNITUDES rather than complex values.
        let n = nx * ny * nz;
        let mag_vec: Vec<f64> = (0..n).map(|i| vec_field.ez[i].norm()).collect();
        let mag_ref: Vec<f64> = (0..n).map(|i| ez_only[i].norm()).collect();

        let scale = mag_ref.iter().cloned().fold(0.0_f64, f64::max);
        let max_diff: f64 = mag_vec
            .iter()
            .zip(mag_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        // Sanity: both should produce non-trivial fields or both trivial
        let vec_scale = mag_vec.iter().cloned().fold(0.0_f64, f64::max);
        if scale > 1e-30 || vec_scale > 1e-30 {
            assert!(
                scale > 1e-30,
                "Phase 9 Ez field is trivially zero (max={scale:.3e})"
            );
            assert!(
                vec_scale > 1e-30,
                "Vector path Ez field is trivially zero (max={vec_scale:.3e})"
            );
            // Magnitudes should agree within 20%.
            // (run_forward uses sin-carrier; run_forward_vector uses cos-carrier for
            //  a real amplitude.  In a short transient run the DFT magnitudes are
            //  similar but not identical — 20% is a generous but meaningful bound.)
            assert!(
                max_diff / scale < 0.20,
                "Ez magnitude mismatch: max_diff/scale = {:.4}",
                max_diff / scale
            );
        }
    }

    #[test]
    fn vector_field_3d_indexing() {
        let field = VectorField3d::new(3, 4, 5);
        assert_eq!(field.cell_idx(0, 0, 0), 0);
        assert_eq!(field.cell_idx(1, 0, 0), 1);
        assert_eq!(field.cell_idx(0, 1, 0), 3);
        assert_eq!(field.cell_idx(0, 0, 1), 12);
        let comp = field.at(1, 2, 3);
        assert_eq!(comp.len(), 3);
    }
}
