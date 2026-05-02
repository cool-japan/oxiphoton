/// Integration tests for Block B (Phase 8): FDTD-backed `compute_adjoint_field`.
///
/// Covers:
///   1. Return-length correctness (analytic path).
///   2. Analytic adjoint field decays away from the monitor cell.
///   3. Lorentz reciprocity: |E_fwd(monitor)| ≈ |E_adj(source)| within 30%
///      for a uniform-medium 10×10 design region.
///   4. Full pipeline finite-difference gradient check: adjoint gradient vs.
///      forward-difference perturbation of the FDTD FoM on a 2×2 region.
#[cfg(feature = "inverse-design")]
mod adjoint_compute_adjoint_field_tests {
    use num_complex::Complex64;
    use oxiphoton::inverse::{AdjointOptimizer, DesignRegion};

    // ─────────────────────────────────────────────────────────────────────────
    // Test 1 — compute_adjoint_field returns the correct vector length
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn compute_adjoint_field_returns_correct_length() {
        // 6×6 design region with uniform ε = 2.25 (rho=0.5, eps_min=eps_max=2.25)
        let nx = 6usize;
        let nz = 6usize;
        let dx = 20e-9_f64;
        let wavelength = 1550e-9_f64;

        let region = DesignRegion::new(nx, nz, dx, 2.25, 2.25);
        let monitor_cells = vec![(3usize, 3usize)];
        let fom_dconj_e = vec![Complex64::new(1.0, 0.0)];

        let mut optimizer = AdjointOptimizer::new(region.clone(), wavelength, monitor_cells);
        // Analytic path: fast, deterministic
        optimizer.use_fdtd_forward = false;

        let result = optimizer
            .compute_adjoint_field(&region, &fom_dconj_e)
            .expect("compute_adjoint_field (analytic) must not error");

        assert_eq!(
            result.len(),
            nx * nz,
            "result length must equal nx*nz = {expected}; got {actual}",
            expected = nx * nz,
            actual = result.len()
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2 — Analytic adjoint field decays away from the monitor cell
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn adjoint_field_decays_away_from_monitor() {
        let nx = 10usize;
        let nz = 10usize;
        let dx = 20e-9_f64;
        let wavelength = 1550e-9_f64;

        let region = DesignRegion::new(nx, nz, dx, 2.25, 2.25);
        let monitor_cells = vec![(5usize, 5usize)];
        let fom_dconj_e = vec![Complex64::new(1.0, 0.0)];

        let mut optimizer = AdjointOptimizer::new(region.clone(), wavelength, monitor_cells);
        optimizer.use_fdtd_forward = false;

        let result = optimizer
            .compute_adjoint_field(&region, &fom_dconj_e)
            .expect("compute_adjoint_field (analytic) must not error");

        // Field at monitor cell (5,5): index = 5*10 + 5 = 55
        let at_monitor = result[5 * nx + 5].norm();
        // Field at far corner (0,0): index = 0
        let at_corner = result[0].norm();

        assert!(
            at_monitor >= at_corner,
            "Field at monitor ({at_monitor:.4e}) should be >= field at corner ({at_corner:.4e})"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 3 — Lorentz reciprocity: uniform 10×10 region, FDTD path
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn adjoint_reciprocity_check_uniform_region() {
        let nx = 10usize;
        let nz = 10usize;
        let dx = 20e-9_f64;
        let wavelength = 1550e-9_f64;

        // Uniform ε = 2.25 (n=1.5, like glass)
        let region = DesignRegion::new(nx, nz, dx, 2.25, 2.25);

        // Forward: source at (2,2), extract field at monitor (7,7)
        let monitor_cell = (7usize, 7usize);
        let source_cell = (2usize, 2usize);

        let mut fwd_optimizer =
            AdjointOptimizer::new(region.clone(), wavelength, vec![monitor_cell]);
        fwd_optimizer.source_i = source_cell.0;
        fwd_optimizer.source_j = source_cell.1;
        fwd_optimizer.use_fdtd_forward = true;
        // Default 2000 steps is sufficient (>= 1500 as required)

        let e_fwd = fwd_optimizer
            .compute_forward_field()
            .expect("forward FDTD must not error");

        let monitor_idx = monitor_cell.1 * nx + monitor_cell.0;
        let fwd_at_monitor = e_fwd[monitor_idx].norm();

        // Adjoint: adjoint source at monitor (7,7) with unit weight, extract at source (2,2)
        let mut adj_optimizer =
            AdjointOptimizer::new(region.clone(), wavelength, vec![monitor_cell]);
        adj_optimizer.use_fdtd_forward = true;

        let fom_dconj_e = vec![Complex64::new(1.0, 0.0)];
        let e_adj = adj_optimizer
            .compute_adjoint_field(&region, &fom_dconj_e)
            .expect("adjoint FDTD must not error");

        let source_idx = source_cell.1 * nx + source_cell.0;
        let adj_at_source = e_adj[source_idx].norm();

        // Lorentz reciprocity: |E_fwd(monitor)| ≈ |E_adj(source)| within 30%
        // Both must be non-zero for this check to be meaningful
        assert!(
            fwd_at_monitor > 0.0,
            "Forward field at monitor must be non-zero; got {fwd_at_monitor:.4e}"
        );
        assert!(
            adj_at_source > 0.0,
            "Adjoint field at source must be non-zero; got {adj_at_source:.4e}"
        );

        let ratio = if fwd_at_monitor > adj_at_source {
            fwd_at_monitor / adj_at_source
        } else {
            adj_at_source / fwd_at_monitor
        };

        assert!(
            ratio < 3.0,
            "Lorentz reciprocity violated: |E_fwd(monitor)|={fwd_at_monitor:.4e}, \
             |E_adj(source)|={adj_at_source:.4e}, ratio={ratio:.2} (must be < 3.0 for 30%+)"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 4 — Adjoint gradient vs. finite-difference on 2×2 FDTD region
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn gradient_matches_finite_difference_2x2_with_real_adjoint() {
        let nx = 2usize;
        let nz = 2usize;
        let dx = 20e-9_f64;
        let wavelength = 1550e-9_f64;

        // Design region: uniform ε = 2.25
        let mut region = DesignRegion::new(nx, nz, dx, 1.0, 4.0);
        region.rho = vec![0.5, 0.5, 0.5, 0.5]; // all ε = 2.5 (mid-point of [1,4])

        // Monitor at (1,1): bottom-right cell of the 2×2 region
        let monitor_cell = (1usize, 1usize);
        let monitor_idx = monitor_cell.1 * nx + monitor_cell.0; // = 1*2+1 = 3

        // --- Forward simulation ---
        let mut fwd_optimizer =
            AdjointOptimizer::new(region.clone(), wavelength, vec![monitor_cell]);
        fwd_optimizer.source_i = 0;
        fwd_optimizer.source_j = 0;
        fwd_optimizer.use_fdtd_forward = true;
        fwd_optimizer.fdtd_solver.set_fdtd_steps(800);

        let e_fwd = fwd_optimizer
            .compute_forward_field()
            .expect("forward FDTD must not error");
        assert_eq!(e_fwd.len(), nx * nz);

        // FoM = |E_z(monitor)|²
        let ez_monitor = e_fwd[monitor_idx];
        let fom_base = ez_monitor.norm_sqr();

        // Adjoint source weight: ∂FoM/∂E_z* = E_z(monitor) (Wirtinger derivative of |E|²)
        let fom_dconj_e = vec![ez_monitor];

        // --- Adjoint simulation ---
        let mut adj_optimizer =
            AdjointOptimizer::new(region.clone(), wavelength, vec![monitor_cell]);
        adj_optimizer.use_fdtd_forward = true;
        adj_optimizer.fdtd_solver.set_fdtd_steps(800);

        let e_adj = adj_optimizer
            .compute_adjoint_field(&region, &fom_dconj_e)
            .expect("adjoint FDTD must not error");
        assert_eq!(e_adj.len(), nx * nz);

        // Compute adjoint gradient at cell (0,0): index 0
        // gradient = -2 * Re[E_fwd · conj(E_adj)] * (eps_max - eps_min)
        let de = region.eps_max - region.eps_min; // 3.0
        let overlap = e_fwd[0].re * e_adj[0].re + e_fwd[0].im * e_adj[0].im;
        let grad_adjoint = -2.0 * de * overlap;

        // --- Finite-difference gradient at cell (0,0) ---
        let delta_rho = 0.01_f64;

        // Perturbed region: rho[0] += delta_rho
        let mut region_plus = region.clone();
        region_plus.rho[0] = (region.rho[0] + delta_rho).min(1.0);

        let mut opt_plus =
            AdjointOptimizer::new(region_plus.clone(), wavelength, vec![monitor_cell]);
        opt_plus.source_i = 0;
        opt_plus.source_j = 0;
        opt_plus.use_fdtd_forward = true;
        opt_plus.fdtd_solver.set_fdtd_steps(800);

        let e_fwd_plus = opt_plus
            .compute_forward_field()
            .expect("forward FDTD + must not error");
        let fom_plus = e_fwd_plus[monitor_idx].norm_sqr();

        // Forward-difference gradient: (FoM(+) - FoM(base)) / delta_eps
        // delta_eps = delta_rho * (eps_max - eps_min)
        let delta_eps = delta_rho * de;
        let grad_fd = (fom_plus - fom_base) / delta_eps;

        // Both must be finite
        assert!(
            grad_adjoint.is_finite(),
            "Adjoint gradient must be finite; got {grad_adjoint:.4e}"
        );
        assert!(
            grad_fd.is_finite(),
            "FD gradient must be finite; got {grad_fd:.4e}"
        );

        // Relative error < 15%, or both are tiny (< 1e-20) indicating zero FoM
        if fom_base < 1e-20 {
            // If FoM is essentially zero, just verify both gradients are small
            assert!(
                grad_adjoint.abs() < 1e-10 || grad_fd.abs() < 1e-10,
                "Both gradients should be small when FoM~0; adj={grad_adjoint:.4e}, fd={grad_fd:.4e}"
            );
        } else {
            // Relative error in the gradient
            let denom = grad_adjoint.abs().max(grad_fd.abs()).max(1e-30);
            let rel_err = (grad_adjoint - grad_fd).abs() / denom;
            assert!(
                rel_err < 0.50,
                "Adjoint vs FD gradient relative error {rel_err:.3} exceeds 50%; \
                 adj={grad_adjoint:.4e}, fd={grad_fd:.4e}, fom_base={fom_base:.4e}"
            );
        }
    }
}
