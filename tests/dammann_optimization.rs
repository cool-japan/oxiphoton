use oxiphoton::diffractive::grating::DammannGrating;

// Helper: build a default DammannGrating with n_spots spots.
// period_um and wavelength_nm are arbitrary; they do not affect the optimiser.
fn make_dammann(n_spots: usize) -> DammannGrating {
    DammannGrating::new(10.0, n_spots, 1064.0)
}

#[test]
fn optimised_transitions_3_orders_converges() {
    // n_orders=3 → optimises for odd diffraction orders m = 1, 3, 5.
    // The bilateral efficiency (counting both +m and −m) should exceed 0.70
    // and uniformity across odd orders should exceed 0.80.
    let mut g = make_dammann(3);
    let j = g.optimize_transitions(3).expect("LM should converge");
    assert!(j.is_finite(), "cost should be finite, got {j}");
    let eff = g.efficiency();
    assert!(eff > 0.70, "efficiency after optimisation: {eff}");
    let uni = g.uniformity();
    assert!(uni > 0.80, "uniformity after optimisation: {uni}");
}

#[test]
fn efficiency_above_70_percent_for_optimised_design() {
    let mut g = make_dammann(3);
    g.optimize_transitions(3).expect("LM should converge");
    assert!(g.efficiency() > 0.70, "efficiency = {}", g.efficiency());
}

#[test]
fn uniformity_above_80_percent_for_optimised_design() {
    let mut g = make_dammann(3);
    g.optimize_transitions(3).expect("LM should converge");
    assert!(g.uniformity() > 0.80, "uniformity = {}", g.uniformity());
}

#[test]
fn fourier_coefficients_zero_for_even_orders() {
    // After half-wave-symmetric optimisation, even non-zero Fourier orders
    // should vanish by construction (f(x+0.5) = −f(x)).
    let mut g = make_dammann(3);
    g.optimize_transitions(3).expect("LM should converge");
    // Compute enough coefficients to cover m=2,4,6.
    let coeffs = g.fourier_coefficients(6);
    // coeffs[0]=DC, coeffs[1]=m=1, coeffs[2]=m=2, …
    for m in [2usize, 4, 6] {
        let c = &coeffs[m];
        assert!(
            c.norm() < 1e-10,
            "c_{m} should be ~0 (half-wave symmetry), got {:.2e}",
            c.norm()
        );
    }
}

#[test]
fn optimisation_does_not_violate_monotonicity() {
    let mut g = make_dammann(3);
    g.optimize_transitions(3).expect("LM should converge");
    // Full optimised transition set must be sorted.
    if let Some(t) = g.get_optimised_transitions() {
        for pair in t.windows(2) {
            assert!(
                pair[0] <= pair[1] + 1e-10,
                "transitions should be monotone: {} > {}",
                pair[0],
                pair[1]
            );
        }
    }
}

#[test]
fn fourier_coefficients_without_optimisation_works() {
    // Before any optimisation, fourier_coefficients() falls back to the
    // hard-coded table and should return a valid vector.
    let g = make_dammann(3);
    let coeffs = g.fourier_coefficients(3);
    // DC + m=1, m=2, m=3 → 4 entries
    assert_eq!(coeffs.len(), 4, "should have DC + m=1,2,3");
    // At least one coefficient should be non-trivial for a real grating.
    let total: f64 = coeffs.iter().map(|c| c.norm_sqr()).sum();
    assert!(
        total > 1e-6,
        "coefficients should not all be zero: total={total}"
    );
}

#[test]
fn optimize_transitions_zero_n_orders_returns_error() {
    let mut g = make_dammann(3);
    let result = g.optimize_transitions(0);
    assert!(result.is_err(), "n_orders=0 should return an error");
}

#[test]
fn nonuniformity_metric_is_complement_of_uniformity() {
    let mut g = make_dammann(3);
    g.optimize_transitions(3).expect("LM should converge");
    let non_uni = g.nonuniformity_metric(3);
    let uni = g.uniformity();
    assert!(
        (non_uni - (1.0 - uni)).abs() < 1e-10,
        "nonuniformity_metric should be 1 - uniformity, got {non_uni} vs {}",
        1.0 - uni
    );
}

#[test]
fn transition_points_returns_optimised_after_optimise() {
    let mut g = make_dammann(3);
    g.optimize_transitions(3).expect("LM should converge");
    let tp = g.transition_points();
    let opt = g
        .get_optimised_transitions()
        .expect("should have optimised transitions");
    assert_eq!(
        tp, opt,
        "transition_points() should return optimised transitions"
    );
}

#[test]
fn half_wave_symmetry_full_transitions_count() {
    // optimize_transitions(K) should store 2*K full-period transitions.
    let mut g = make_dammann(3);
    g.optimize_transitions(3).expect("LM should converge");
    if let Some(t) = g.get_optimised_transitions() {
        assert_eq!(
            t.len(),
            6,
            "K=3 half params → 2*3=6 full-period transitions"
        );
    }
}

#[test]
fn all_transitions_in_unit_interval() {
    let mut g = make_dammann(3);
    g.optimize_transitions(3).expect("LM should converge");
    if let Some(t) = g.get_optimised_transitions() {
        for &x in t {
            assert!(x > 0.0 && x < 1.0, "transition {x} outside (0, 1)");
        }
    }
}
