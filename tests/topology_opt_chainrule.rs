#[cfg(feature = "topology-opt")]
mod tests {
    use oxiphoton::inverse::adjoint::DesignRegion;
    use oxiphoton::inverse::topology::continuation_schedule;
    use oxiphoton::inverse::TopologyOptimizer;

    fn make_optimizer(nx: usize, nz: usize) -> TopologyOptimizer {
        let region = DesignRegion::new(nx, nz, 1.0 / nx as f64, 1.0, 12.11);
        TopologyOptimizer::new(region, 2.0)
    }

    #[test]
    fn filter_adjoint_finite_difference() {
        let mut opt = make_optimizer(8, 8);
        let n = 64;
        // Deterministic pseudo-random density via hashing
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let rho: Vec<f64> = (0..n)
            .map(|i| {
                let mut h = DefaultHasher::new();
                i.hash(&mut h);
                (h.finish() % 1000) as f64 / 1000.0
            })
            .collect();
        opt.region.rho = rho.clone();

        // Compute filter output at baseline
        let f0 = opt.filter_density();

        // Random direction u
        let u: Vec<f64> = (0..n)
            .map(|i| {
                let mut h = DefaultHasher::new();
                (i + 1000).hash(&mut h);
                (h.finish() % 1000) as f64 / 1000.0 - 0.5
            })
            .collect();

        // Compute adjoint: F^T(u)
        let ft_u = opt.filter_adjoint(&u);

        // Finite-difference directional derivative w.r.t. pixel k
        let eps = 1e-5_f64;
        let k = 3_usize;
        let mut rho_pert = rho.clone();
        rho_pert[k] += eps;
        opt.region.rho = rho_pert;
        let f_pert = opt.filter_density();

        // Finite-difference column: (F(rho + eps*e_k) - F(rho)) / eps
        let fd: Vec<f64> = (0..n).map(|i| (f_pert[i] - f0[i]) / eps).collect();

        // Inner-product check: u · fd_column  ≈  (F^T u)[k]
        let dot_fd: f64 = u.iter().zip(fd.iter()).map(|(a, b)| a * b).sum();
        let dot_adj = ft_u[k];

        assert!(
            (dot_adj - dot_fd).abs() / (dot_fd.abs() + 1e-10) < 0.01,
            "Filter adjoint FD check failed: analytical={:.6e}, FD={:.6e}",
            dot_adj,
            dot_fd
        );
    }

    #[test]
    fn projection_jacobian_finite_difference() {
        let opt = make_optimizer(4, 4);
        let filtered: Vec<f64> = (0..16).map(|i| (0.1 + 0.05 * i as f64).min(0.9)).collect();

        let jac = opt.projection_jacobian(&filtered);
        let projected = opt.project_density(&filtered);

        let eps = 1e-5_f64;
        for i in 0..16 {
            let mut f_pert = filtered.clone();
            f_pert[i] += eps;
            let p_pert = opt.project_density(&f_pert);
            let fd = (p_pert[i] - projected[i]) / eps;
            assert!(
                (jac[i] - fd).abs() / (fd.abs() + 1e-8) < 0.01,
                "Projection Jacobian pixel {i}: analytical={:.6e}, FD={:.6e}",
                jac[i],
                fd
            );
        }
    }

    #[test]
    fn oc_step_enforces_volume_constraint() {
        let mut opt = make_optimizer(8, 8);
        // Start at uniform 0.5
        opt.region.rho = vec![0.5; 64];
        // Gradient: push density up everywhere
        let grad = vec![1.0_f64; 64];
        let target_vol = 0.4;
        let result = opt.oc_step(&grad, target_vol, 0.2);
        assert!(result.is_ok(), "OC step failed: {:?}", result.err());
        // Re-evaluate projected volume after update
        let filtered = opt.filter_density();
        let projected = opt.project_density(&filtered);
        let vol: f64 = projected.iter().sum::<f64>() / 64.0;
        assert!(
            (vol - target_vol).abs() < 0.02,
            "Volume constraint not satisfied: got {:.4}, want {:.4}",
            vol,
            target_vol
        );
    }

    #[test]
    fn oc_step_respects_move_limit() {
        let mut opt = make_optimizer(8, 8);
        let rho0 = vec![0.5; 64];
        opt.region.rho = rho0.clone();
        let grad = vec![1.0_f64; 64];
        let move_limit = 0.1;
        // Result may be Ok or Err — either way rho should be updated respecting move_limit
        let _ = opt.oc_step(&grad, 0.5, move_limit);
        for (r0, r1) in rho0.iter().zip(opt.region.rho.iter()) {
            assert!(
                (r1 - r0).abs() <= move_limit + 1e-10,
                "Move limit violated: r0={:.4}, r1={:.4}, limit={:.4}",
                r0,
                r1,
                move_limit
            );
        }
    }

    #[test]
    fn step_applies_chain_rule_end_to_end() {
        let nx = 5_usize;
        let nz = 5_usize;
        let region = DesignRegion::new(nx, nz, 1.0 / nx as f64, 1.0, 12.11);
        let mut opt = TopologyOptimizer::new(region, 1.0);
        // Use beta=2.0 to exercise non-trivial projection Jacobian
        opt.beta = 2.0;

        // Compute the chain-rule gradient manually for a constant dfom_drbar = [1.0; 25]
        let dfom_drbar = vec![1.0_f64; nx * nz];
        let filtered = opt.filter_density();
        let dproj = opt.projection_jacobian(&filtered);
        let dfom_drtilde: Vec<f64> = dfom_drbar
            .iter()
            .zip(dproj.iter())
            .map(|(g, j)| g * j)
            .collect();
        let expected_raw = opt.filter_adjoint(&dfom_drtilde);
        let rho_before = opt.region.rho.clone();
        let step_size = 0.01;

        // Apply step()
        opt.step(&dfom_drbar, step_size);

        // Verify: rho_after[i] = clamp(rho_before[i] + step_size * expected_raw[i], 0, 1)
        for (i, (&rho_new, (&rho_old, &g_exp))) in opt
            .region
            .rho
            .iter()
            .zip(rho_before.iter().zip(expected_raw.iter()))
            .enumerate()
        {
            let expected = (rho_old + step_size * g_exp).clamp(0.0, 1.0);
            assert!(
                (rho_new - expected).abs() < 1e-10,
                "Pixel {i}: expected {expected:.6}, got {rho_new:.6}"
            );
        }
    }

    #[test]
    fn step_with_raw_gradient_preserves_legacy_semantics() {
        let nx = 4_usize;
        let nz = 4_usize;
        let region = DesignRegion::new(nx, nz, 1.0 / nx as f64, 1.0, 12.11);
        let mut opt = TopologyOptimizer::new(region, 1.0);

        let raw_gradient: Vec<f64> = (0..nx * nz).map(|i| (i as f64) * 0.01 - 0.08).collect();
        let step_size = 0.5;
        let rho_before = opt.region.rho.clone();

        opt.step_with_raw_gradient(&raw_gradient, step_size);

        for (i, (&rho_new, (&rho_old, &g))) in opt
            .region
            .rho
            .iter()
            .zip(rho_before.iter().zip(raw_gradient.iter()))
            .enumerate()
        {
            let expected = (rho_old + step_size * g).clamp(0.0, 1.0);
            assert!(
                (rho_new - expected).abs() < 1e-12,
                "Pixel {i}: expected {expected:.6}, got {rho_new:.6}"
            );
        }
    }

    #[test]
    fn continuation_increases_binarisation() {
        let betas = [1.0_f64, 2.0, 4.0, 8.0];
        let schedule = continuation_schedule(5, &betas);
        let mut opt = make_optimizer(8, 8);
        // Initialize near 0.5 with slight perturbations
        opt.region.rho = (0..64)
            .map(|i| 0.5 + (i as f64 * 0.01) % 0.1 - 0.05)
            .collect();

        let mut prev_bin_frac = 0.0_f64;
        for step in 0..20_usize {
            opt.iteration = step;
            opt.apply_continuation(&schedule);
            // Use simple gradient (push everything up)
            let grad = vec![0.5_f64; 64];
            let _ = opt.oc_step(&grad, 0.5, 0.2);
            let bin_frac = opt.binarisation_fraction(0.45);
            // After each beta increase, binarisation should not decrease significantly
            if step > 0 && step % 5 == 4 {
                assert!(
                    bin_frac >= prev_bin_frac - 0.05,
                    "Binarisation decreased at step {}: {} -> {}",
                    step,
                    prev_bin_frac,
                    bin_frac
                );
            }
            prev_bin_frac = bin_frac;
        }
    }
}
