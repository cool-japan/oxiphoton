//! Integration tests for 2D material models and thermal FDTD (Wave 5).
//!
//! Tests cover: GrapheneSheet optical conductivity, DC scaling, universal absorption,
//! plasmon confinement, hBN hyperbolic range, MoS2 monolayer direct gap and layer
//! dependence, BlackPhosphorus anisotropy, ThermalFdtd3d temperature increase, and
//! HeatSolver3d diffusion of a hot spot.

mod graphene_tests {
    use oxiphoton::material::{BPDirection, BlackPhosphorus, GrapheneSheet, HexagonalBN, MoS2};
    use std::f64::consts::PI;

    // Physical constants matching the material module
    const SPEED_OF_LIGHT: f64 = 2.997_924_58e8;

    // ── test 1: graphene conductivity is non-zero at optical frequencies ──────

    /// Surface conductivity of graphene at visible/near-IR frequencies should
    /// be non-zero in both real and imaginary parts.
    #[test]
    fn test_graphene_conductivity_optical() {
        let graphene = GrapheneSheet::new(0.3, 1.0, 300.0);
        let omega = 2.0 * PI * 3.0e14; // 300 THz ≈ 1 µm
        let sigma = graphene.surface_conductivity(omega);
        assert!(
            sigma.norm() > 0.0,
            "Graphene surface conductivity must be non-zero at optical frequencies: |σ| = {}",
            sigma.norm()
        );
    }

    // ── test 2: DC conductivity scaling with Fermi energy ────────────────────

    /// DC conductivity σ_DC ∝ E_F: doubling E_F must double σ_DC.
    #[test]
    fn test_graphene_dc_scaling() {
        let g1 = GrapheneSheet::new(0.2, 1.0, 300.0);
        let g2 = GrapheneSheet::new(0.4, 1.0, 300.0); // doubled E_F

        let dc1 = g1.dc_conductivity();
        let dc2 = g2.dc_conductivity();

        assert!(dc1 > 0.0, "DC conductivity must be positive for E_F > 0");
        assert!(
            dc2 > 0.0,
            "DC conductivity must be positive for doubled E_F"
        );

        let ratio = dc2 / dc1;
        assert!(
            (ratio - 2.0).abs() < 1e-9,
            "DC conductivity must scale linearly with E_F: ratio = {ratio}, expected 2.0"
        );
    }

    // ── test 3: universal absorption ≈ 2.3% ──────────────────────────────────

    /// πα ≈ 2.293%, where α = fine-structure constant ≈ 1/137.
    #[test]
    fn test_graphene_universal_absorption() {
        let abs = GrapheneSheet::universal_absorption();
        // πα ≈ 0.02293
        assert!(
            (abs - 0.02293).abs() < 5e-4,
            "Universal absorption should be ≈ 2.3% (πα), got {:.5}",
            abs
        );
        assert!(abs > 0.02, "Universal absorption must be > 2%");
        assert!(abs < 0.03, "Universal absorption must be < 3%");
    }

    // ── test 4: graphene plasmon confinement at THz ───────────────────────────

    /// At mid-IR/THz frequencies the graphene plasmon wavevector |k_sp| must
    /// greatly exceed the free-space wavevector k₀ (strong confinement).
    #[test]
    fn test_graphene_plasmon_confined() {
        // High-quality graphene (long relaxation time) for strong confinement
        let graphene = GrapheneSheet::new(0.3, 10.0, 300.0);
        let omega = 2.0 * PI * 30.0e12; // 30 THz
        let k_sp = graphene.plasmon_wavevector(omega, 1.0, 1.0);
        let k0 = omega / SPEED_OF_LIGHT;

        let confinement = k_sp.norm() / k0;
        assert!(
            confinement > 10.0,
            "Graphene plasmon must be highly confined: |k_sp|/k₀ = {confinement:.2}, expected > 10"
        );
    }

    // ── test 5: hBN has hyperbolic frequency ranges ───────────────────────────

    /// hBN must have at least two hyperbolic Reststrahlen bands (Type I and II).
    #[test]
    fn test_hbn_has_hyperbolic_range() {
        let hbn = HexagonalBN::new(10);
        let ranges = hbn.hyperbolic_frequency_range();

        assert!(
            ranges.len() >= 2,
            "hBN must have at least two hyperbolic bands, found {}",
            ranges.len()
        );

        for (i, &(lo, hi)) in ranges.iter().enumerate() {
            assert!(
                hi > lo,
                "Hyperbolic band {i}: upper frequency {hi:.3e} must exceed lower {lo:.3e}"
            );
        }

        // Check that a frequency inside the upper Reststrahlen band is hyperbolic
        // (~1490 cm⁻¹, between TO=1370 and LO=1610 cm⁻¹)
        let cm1_to_rads = 2.0 * PI * SPEED_OF_LIGHT * 100.0;
        let omega_upper = 1490.0 * cm1_to_rads;
        assert!(
            hbn.is_hyperbolic(omega_upper),
            "hBN should be hyperbolic at ~1490 cm⁻¹ (upper Reststrahlen band)"
        );
    }

    // ── test 6: MoS2 monolayer is direct gap ─────────────────────────────────

    /// MoS₂ with n_layers = 1 must report is_direct_bandgap() = true and
    /// a bandgap near 1.80 eV.
    #[test]
    fn test_mos2_monolayer_direct() {
        let mos2 = MoS2::new(1, 0.0);
        assert!(
            mos2.is_direct_bandgap(),
            "MoS₂ monolayer must have a direct bandgap"
        );

        let eg = mos2.bandgap_ev();
        assert!(
            (eg - 1.80).abs() < 0.05,
            "Monolayer MoS₂ bandgap should be ≈ 1.80 eV, got {eg:.3} eV"
        );
    }

    // ── test 7: MoS2 bandgap decreases with layers ───────────────────────────

    /// Quantum confinement: bandgap must strictly decrease as layer count rises.
    #[test]
    fn test_mos2_layer_dependence() {
        let eg1 = MoS2::new(1, 0.0).bandgap_ev();
        let eg2 = MoS2::new(2, 0.0).bandgap_ev();
        let eg3 = MoS2::new(3, 0.0).bandgap_ev();

        assert!(
            eg1 > eg2,
            "Monolayer gap ({eg1:.3} eV) must exceed bilayer gap ({eg2:.3} eV)"
        );
        assert!(
            eg2 > eg3,
            "Bilayer gap ({eg2:.3} eV) must exceed trilayer gap ({eg3:.3} eV)"
        );

        // Bulk (≥3 layers) must not be a direct bandgap
        assert!(
            !MoS2::new(3, 0.0).is_direct_bandgap(),
            "MoS₂ with ≥3 layers must NOT be a direct bandgap semiconductor"
        );
    }

    // ── test 8: BlackPhosphorus armchair vs zigzag anisotropy ─────────────────

    /// Permittivity must differ between armchair and zigzag directions at
    /// optical frequencies due to the strong in-plane anisotropy of BP.
    #[test]
    fn test_bp_anisotropy() {
        let omega = 2.0 * PI * 4.0e14; // ~750 nm (visible)
        let bp_ac = BlackPhosphorus::new(1, BPDirection::Armchair);
        let bp_zz = BlackPhosphorus::new(1, BPDirection::Zigzag);

        let eps_ac = bp_ac.permittivity(omega);
        let eps_zz = bp_zz.permittivity(omega);

        let diff = (eps_ac - eps_zz).norm();
        assert!(
            diff > 1e-3,
            "Armchair and zigzag permittivities must differ at optical frequencies, |Δε| = {diff:.4e}"
        );

        // Armchair effective mass must be lighter (less confined → higher mobility)
        assert!(
            bp_ac.effective_mass_ratio() < bp_zz.effective_mass_ratio(),
            "Armchair m* ({}) must be lighter than zigzag m* ({})",
            bp_ac.effective_mass_ratio(),
            bp_zz.effective_mass_ratio()
        );
    }
}

// ── Thermal FDTD tests ────────────────────────────────────────────────────────

#[cfg(feature = "fdtd")]
mod thermal_tests {
    use oxiphoton::fdtd::engine::thermal::{HeatSolver3d, ThermalFdtd3d};

    // ── test 9: thermal FDTD temperature increase ─────────────────────────────

    /// After setting a uniform above-ambient temperature and calling
    /// apply_temperature_field(), eps_current should change relative to
    /// the base-temperature permittivity for a cell with non-zero dn/dT.
    #[test]
    fn test_thermal_fdtd_temperature_increase() {
        let base_temp = 300.0_f64;
        let mut thermal = ThermalFdtd3d::new(8, 8, 8, 10e-9, 10e-9, 10e-9, base_temp);

        // Set a non-zero thermo-optic coefficient dn/dT for the whole grid
        let dn_dt = 1.84e-4_f64; // Si-like: ~1.84×10⁻⁴ /K
        thermal.set_thermo_optic(0, 8, 0, 8, 0, 8, dn_dt);

        // Set base permittivity to Si-like value (n_Si ≈ 3.5 → ε ≈ 12.25)
        thermal.set_eps_region(0, 8, 0, 8, 0, 8, 12.25);

        // Raise temperature by 100 K across the whole domain
        let hot_temp = base_temp + 100.0;
        thermal.set_uniform_temperature(hot_temp);

        // Recompute permittivity from the new temperature field
        thermal.apply_temperature_field();

        // eps_current should differ from eps_base due to thermo-optic effect
        let idx = thermal.idx(4, 4, 4);
        let eps_base = thermal.eps_base[idx];
        let eps_now = thermal.eps_current[idx];

        assert!(
            (eps_now - eps_base).abs() > 1e-6,
            "eps_current should change after temperature increase; Δε = {}",
            eps_now - eps_base
        );

        // The refractive index should have increased (for positive dn/dT)
        let n_base = eps_base.sqrt();
        let n_now = thermal.refractive_index_at(4, 4, 4);
        assert!(
            n_now > n_base,
            "Refractive index should increase with temperature (dn/dT > 0): n₀ = {n_base:.6}, n_now = {n_now:.6}"
        );

        // max_delta_t should reflect the 100 K increase
        let max_dt = thermal.max_delta_t();
        assert!(
            (max_dt - 100.0).abs() < 1e-9,
            "max_delta_t should be 100 K, got {max_dt}"
        );
    }

    // ── test 10: heat solver diffuses hot spot ────────────────────────────────

    /// Place a hot spot in the centre of a small 3D domain.  After several
    /// explicit Euler steps the temperature at the hot-spot centre must
    /// decrease while the neighbouring cells' temperature must increase.
    #[test]
    fn test_heat_solver_diffusion() {
        let nx = 10usize;
        let ny = 10usize;
        let nz = 10usize;
        let dx = 1e-6_f64; // 1 µm cells

        // Use Si thermal diffusivity: α ≈ 8.8e-5 m²/s
        let alpha_si = 8.8e-5_f64;

        // CFL stability: dt < dx² / (6 α) ≈ (1e-6)² / (6 × 8.8e-5) ≈ 1.89 ns
        let dt_cfl = dx * dx / (6.0 * alpha_si);
        let dt = 0.4 * dt_cfl; // conservative safety factor

        let ambient = 300.0_f64;
        let mut solver = HeatSolver3d::new(nx, ny, nz, dx, dx, dx, dt, ambient);

        // Set Si diffusivity everywhere
        solver.set_diffusivity_region(0, nx, 0, ny, 0, nz, alpha_si);

        // Create a hot spot of 400 K at the grid centre
        let ic = nx / 2;
        let jc = ny / 2;
        let kc = nz / 2;
        let hot_idx = solver.idx(ic, jc, kc);
        solver.temperature[hot_idx] = 400.0;

        let t_centre_before = solver.temperature[hot_idx];

        // Neighbour before (should be ambient)
        let nb_idx = solver.idx(ic + 1, jc, kc);
        let t_nb_before = solver.temperature[nb_idx];
        assert!(
            (t_nb_before - ambient).abs() < 1e-9,
            "neighbour should start at ambient"
        );

        // Run enough steps for diffusion to be clearly visible
        let n_steps = 20;
        for _ in 0..n_steps {
            solver.step();
        }

        let t_centre_after = solver.temperature[hot_idx];
        let t_nb_after = solver.temperature[nb_idx];

        // Hot spot must cool down
        assert!(
            t_centre_after < t_centre_before,
            "Hot-spot centre must cool: T_before = {t_centre_before:.2} K, T_after = {t_centre_after:.2} K"
        );

        // Neighbour must warm up
        assert!(
            t_nb_after > t_nb_before,
            "Neighbour must warm up: T_before = {t_nb_before:.2} K, T_after = {t_nb_after:.2} K"
        );

        // Maximum temperature must remain finite
        let t_max = solver.max_temperature();
        assert!(t_max.is_finite(), "Max temperature must be finite: {t_max}");
        assert!(
            t_max <= 400.0 + 1e-6,
            "Max temperature must not exceed initial hot-spot: {t_max}"
        );
    }
}
