#[cfg(feature = "topology-opt")]
mod tests {
    use oxiphoton::inverse::adjoint::DesignRegion;
    use oxiphoton::inverse::topology::continuation_schedule;
    use oxiphoton::inverse::{Pseudo2dFom, TopologyOptimizer};

    #[test]
    fn splitter_fom_improves_after_n_iters() {
        let nx = 16_usize;
        let nz = 16_usize;
        let region = DesignRegion::new(nx, nz, 1.0 / nx as f64, 1.0, 12.11);
        let mut opt = TopologyOptimizer::new(region, 2.0);
        let fom_obj = Pseudo2dFom::new(nx, nz);

        // Initialize uniform 0.5
        opt.region.rho = vec![0.5; nx * nz];

        let betas = [1.0_f64, 2.0, 4.0];
        let schedule = continuation_schedule(7, &betas);

        let mut fom_initial = 0.0_f64;
        for step in 0..21_usize {
            opt.iteration = step;
            opt.apply_continuation(&schedule);

            // Evaluate FOM and gradient at current design
            let filtered = opt.filter_density();
            let projected = opt.project_density(&filtered);
            let (fom, grad_proj) = fom_obj.evaluate(&projected);

            if step == 0 {
                fom_initial = fom;
            }
            opt.record_fom(fom);

            // Convert projected gradient to raw gradient via full chain rule
            let raw_grad = opt.raw_gradient(&grad_proj);
            let _ = opt.oc_step(&raw_grad, 0.5, 0.2);
        }

        let fom_final = opt.fom_history.last().copied().unwrap_or(fom_initial);
        assert!(
            fom_final > fom_initial,
            "FOM did not improve: initial={:.4}, final={:.4}",
            fom_initial,
            fom_final
        );
    }
}
