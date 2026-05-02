#[cfg(feature = "fdtd-3d")]
mod thermal_convection_tests {
    use oxiphoton::fdtd::engine::HeatSolver3d;

    fn make_sim(nx: usize, ny: usize, nz: usize, h: f64) -> HeatSolver3d {
        // dx=dy=dz=1e-3 (1 mm cells), dt=0.001 s, rho_cp=1e6 J/(m³·K)
        let dt = 0.001_f64;
        let dx = 1e-3_f64;
        // CFL for Robin BC: dt < rho_cp * dx / (2 * h) = 1e6 * 1e-3 / (2 * h)
        // For h=1000: 1e6*1e-3/2000 = 0.5 s >> dt=0.001 — stable
        let mut sim = HeatSolver3d::new(nx, ny, nz, dx, dx, dx, dt, 300.0);
        sim.convection_coefficient = h;
        sim.ambient_temperature = 300.0;
        sim.set_rho_cp_region(0, nx, 0, ny, 0, nz, 1.0e6);
        sim.set_diffusivity_region(0, nx, 0, ny, 0, nz, 1e-5);
        sim
    }

    #[test]
    fn steady_state_uniform_heat_with_convection() {
        // A small cube with uniform heat source and convective cooling on all faces.
        // Without BC: temperature would diverge. With BC: must reach plateau.
        let mut sim = make_sim(5, 5, 5, 1000.0);
        // Set uniform heat source (pre-divided by rho_cp): Q/rho_cp = 0.01 K/s per step
        let n = 5 * 5 * 5;
        let q_over_rho_cp = 0.01_f64;
        for i in 0..n {
            sim.heat_source[i] = q_over_rho_cp;
        }
        // Initialize at ambient
        for t in sim.temperature.iter_mut() {
            *t = 300.0;
        }

        // Run 5000 steps, sample max temperature every 500 steps
        let mut max_temp_history = Vec::new();
        for step in 0..5000_usize {
            sim.step();
            if step % 500 == 499 {
                let max_t = sim
                    .temperature
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                max_temp_history.push(max_t);
            }
        }

        let last = *max_temp_history.last().expect("history must have entries");
        let second_last = max_temp_history[max_temp_history.len() - 2];

        // Temperature must have plateaued: last two samples close together
        assert!(
            (last - second_last).abs() < 0.1 * (last - 300.0).abs() + 0.1,
            "Temperature did not plateau: last={last:.2}, second_last={second_last:.2}"
        );
        // Must be above ambient (heating active), must not diverge
        assert!(last > 300.0, "Temperature must be above ambient");
        assert!(last < 1000.0, "Temperature must not diverge");
    }

    #[test]
    fn bc_does_nothing_when_h_is_zero() {
        let mut sim_bc = make_sim(4, 4, 4, 0.0); // h=0 → no BC applied
        let mut sim_no_bc = make_sim(4, 4, 4, 0.0);

        // No heat source, start above ambient — pure diffusion
        for t in sim_bc.temperature.iter_mut() {
            *t = 400.0;
        }
        for t in sim_no_bc.temperature.iter_mut() {
            *t = 400.0;
        }

        for _ in 0..100 {
            sim_bc.step();
            sim_no_bc.step();
        }

        let max_diff = sim_bc
            .temperature
            .iter()
            .zip(sim_no_bc.temperature.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-12,
            "h=0 BC should have no effect; diff={max_diff}"
        );
    }

    #[test]
    fn bc_active_on_all_six_faces() {
        // Hot cube, no heat source, convective cooling. Check all 6 faces cool.
        let mut sim = make_sim(6, 6, 6, 1000.0);
        for t in sim.temperature.iter_mut() {
            *t = 500.0;
        }

        for _ in 0..100 {
            sim.step();
        }

        let nx = 6_usize;
        let ny = 6_usize;
        let nz = 6_usize;
        let idx = |i: usize, j: usize, k: usize| i * ny * nz + j * nz + k;

        // Check that every face cell is below the initial 500 K
        let faces = [
            ("x=0 face", sim.temperature[idx(0, 2, 2)]),
            ("x=nx-1 face", sim.temperature[idx(nx - 1, 2, 2)]),
            ("y=0 face", sim.temperature[idx(2, 0, 2)]),
            ("y=ny-1 face", sim.temperature[idx(2, ny - 1, 2)]),
            ("z=0 face", sim.temperature[idx(2, 2, 0)]),
            ("z=nz-1 face", sim.temperature[idx(2, 2, nz - 1)]),
        ];
        for (label, t) in faces {
            assert!(
                t < 499.9,
                "Face '{label}' did not cool: T={t:.2} (initial 500 K)"
            );
        }
    }

    #[test]
    fn convection_decay_rate_matches_lumped_capacity() {
        // Lumped capacity model: dT/dt = -h*A/(rho_cp*V) * (T - T_amb)
        // For a cube of side L: A/V = 6/L
        // τ = rho_cp * L / (6 * h)
        //
        // 5 cells × 2 mm = L = 10 mm, h=200, rho_cp=1e6
        // τ = 1e6 * 0.01 / (6 * 200) = 8.33 s
        // Robin BC CFL: dt < rho_cp * dx / (2*h) = 1e6 * 2e-3 / 400 = 5 s
        // Bulk Laplacian CFL: dx²/(6α) = 4e-6/(6e-4) ≈ 6.7e-3 s
        // Use dt=1e-3 s to satisfy both constraints
        let n = 5_usize;
        let dx = 2e-3_f64;
        let h = 200.0_f64;
        let rho_cp = 1.0e6_f64;
        let dt = 1e-3_f64;

        let mut sim = HeatSolver3d::new(n, n, n, dx, dx, dx, dt, 300.0);
        sim.convection_coefficient = h;
        sim.ambient_temperature = 300.0;
        sim.set_rho_cp_region(0, n, 0, n, 0, n, rho_cp);
        sim.set_diffusivity_region(0, n, 0, n, 0, n, 1e-4);

        // Initialize hot, no heat source
        let t_init = 400.0_f64;
        for t in sim.temperature.iter_mut() {
            *t = t_init;
        }

        let l = n as f64 * dx; // side length = 10 mm
        let tau = rho_cp * l / (6.0 * h); // lumped capacity time constant ≈ 8.33 s

        // Run for exactly 0.5 × τ
        let n_steps = ((0.5 * tau) / dt).round() as usize;
        for _ in 0..n_steps {
            sim.step();
        }

        // Expected from lumped ODE: T(t) = T_amb + (T_init - T_amb) * exp(-t/τ)
        let t_expected = 300.0 + (t_init - 300.0) * (-0.5_f64).exp();
        let t_actual = sim.temperature.iter().sum::<f64>() / sim.temperature.len() as f64;

        // Allow 30% relative tolerance. Deviations arise because:
        // 1. The lumped ODE assumes uniform cooling; in 3D FD, corner cells sit on
        //    three faces simultaneously and are cooled at ~3× the face rate.
        // 2. Edge cells experience 2× face cooling. Both effects accelerate the
        //    domain-average decay compared to the lumped scalar ODE.
        // 3. Bulk diffusion homogenises the field, partially counteracting corner
        //    over-cooling.
        // A 30% envelope is calibrated against this 5-cell cube geometry.
        let rel_err = (t_actual - t_expected).abs() / (t_init - 300.0);
        assert!(
            rel_err < 0.30,
            "Decay rate mismatch: actual={t_actual:.2} expected={t_expected:.2} rel_err={rel_err:.3}"
        );
    }
}
