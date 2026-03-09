use oxiphoton::smatrix::{
    EmeMode, EmeSegment, EmeSolver, GratingLayer, Polarization, RcwaSolver, SMatrix2x2,
};

// ── RCWA tests ────────────────────────────────────────────────────────────────

#[test]
fn rcwa_eps_fourier_dc_term_correct() {
    // For a 50% fill factor grating: eps_0 = (eps1 + eps2)/2
    let layer = GratingLayer::new(500e-9, 200e-9, 2.0, 1.0, 0.5);
    let coeffs = layer.eps_fourier(3);
    let eps_0 = coeffs[3]; // m=0 at index n_orders
    let expected = 2.0 * 2.0 * 0.5 + 1.0 * 0.5; // n1²*f + n2²*(1-f)
    assert!(
        (eps_0.re - expected).abs() < 1e-6,
        "eps_0={:.4} expected={expected:.4}",
        eps_0.re
    );
}

#[test]
fn rcwa_eps_fourier_symmetry() {
    // For real dielectrics, eps_{-m} = conj(eps_m) = eps_m (since eps is real)
    let layer = GratingLayer::new(500e-9, 200e-9, 1.5, 1.0, 0.3);
    let n = 4;
    let coeffs = layer.eps_fourier(n);
    // eps_{-m} and eps_{+m} should have same real part (symmetric)
    for m in 1..=n {
        let neg = coeffs[n - m]; // index for -m
        let pos = coeffs[n + m]; // index for +m
        assert!(
            (neg.re - pos.re).abs() < 1e-10,
            "eps_{{-{m}}} real part should equal eps_{{{m}}} real part"
        );
    }
}

#[test]
fn rcwa_solve_returns_physical_result() {
    let solver = RcwaSolver::new(3, 1.0, 1.5);
    let layer = GratingLayer::new(800e-9, 300e-9, 3.476, 1.0, 0.4);
    let result = solver.solve(&layer, 1550e-9, 0.0, Polarization::TE);

    assert_eq!(result.r_eff.len(), 7); // 2*3+1
    assert!(result.r_total >= 0.0, "R must be >= 0");
    assert!(result.r_total <= 1.0, "R must be <= 1");
    assert!(result.t_total >= 0.0, "T must be >= 0");
    assert!(result.t_total <= 1.0, "T must be <= 1");
}

#[test]
fn rcwa_spectrum_monotonic_wavelength() {
    let solver = RcwaSolver::new(2, 1.0, 1.5);
    let layer = GratingLayer::new(500e-9, 200e-9, 2.0, 1.0, 0.5);
    let wls: Vec<f64> = (5..15).map(|i| i as f64 * 100e-9).collect();
    let results = solver.spectrum(&layer, &wls, 0.0, Polarization::TE);
    assert_eq!(results.len(), wls.len());
    for (r, &wl) in results.iter().zip(wls.iter()) {
        assert_eq!(r.wavelength, wl);
        assert!(r.r_total >= 0.0 && r.r_total <= 1.0);
    }
}

#[test]
fn rcwa_tm_polarization_runs() {
    let solver = RcwaSolver::new(3, 1.0, 1.5);
    let layer = GratingLayer::new(500e-9, 200e-9, 1.5, 1.0, 0.5);
    let result = solver.solve(&layer, 800e-9, 0.0, Polarization::TM);
    assert!(result.r_total >= 0.0 && result.r_total <= 1.0);
}

#[test]
fn rcwa_oblique_incidence_runs() {
    let solver = RcwaSolver::new(5, 1.0, 1.5);
    let layer = GratingLayer::new(600e-9, 250e-9, 2.0, 1.0, 0.4);
    // 15° incidence
    let result = solver.solve(&layer, 1000e-9, 15.0_f64.to_radians(), Polarization::TE);
    assert!(result.r_total >= 0.0 && result.r_total <= 1.0);
}

#[test]
fn rcwa_grating_layer_fill_factor_bounds() {
    // Fill factor must be in (0,1)
    let g = GratingLayer::new(500e-9, 200e-9, 2.0, 1.0, 0.5);
    assert!(g.fill_factor > 0.0 && g.fill_factor < 1.0);
}

// ── EME tests ─────────────────────────────────────────────────────────────────

#[test]
fn eme_segment_finds_si_sio2_mode() {
    let seg = EmeSegment::new(10e-6, 3.476, 1.444, 500e-9);
    let modes = seg.find_modes(1550e-9, 5, 200);
    assert!(!modes.is_empty(), "Si/SiO2 slab should have guided modes");
    assert!(modes[0].n_eff > 1.444 && modes[0].n_eff < 3.476);
}

#[test]
fn eme_transmission_uniform_waveguide() {
    // Single straight segment: should have T ≈ 1 (lossless propagation)
    let mut solver = EmeSolver::new(1550e-9, 1, 200);
    solver.add_segment(EmeSegment::new(10e-6, 3.476, 1.444, 500e-9));
    let t = solver.transmission();
    // For a single segment, propagation S-matrix has |S21|=1
    assert!(
        (t - 1.0).abs() < 1e-6,
        "Single segment T should be 1, got {t:.4}"
    );
}

#[test]
fn eme_transmission_in_unit_range() {
    let mut solver = EmeSolver::new(1550e-9, 2, 150);
    solver.add_segment(EmeSegment::new(5e-6, 3.476, 1.444, 400e-9));
    solver.add_segment(EmeSegment::new(5e-6, 3.476, 1.444, 800e-9));
    let t = solver.transmission();
    assert!((0.0..=1.0).contains(&t), "T={t:.4} not in [0,1]");
}

#[test]
fn smatrix_2x2_identity_cascade() {
    let s1 = SMatrix2x2::identity();
    let s2 = SMatrix2x2::identity();
    let combined = s1.cascade(&s2);
    assert!((combined.s21 - 1.0).abs() < 1e-10);
    assert!((combined.s12 - 1.0).abs() < 1e-10);
    assert!(combined.s11.abs() < 1e-10);
    assert!(combined.s22.abs() < 1e-10);
}

#[test]
fn smatrix_from_overlap_unity() {
    let s = SMatrix2x2::from_overlap(1.0);
    assert!((s.s21 - 1.0).abs() < 1e-10, "Perfect overlap → T=1");
    assert!(s.s11.abs() < 1e-10, "Perfect overlap → R=0");
}

#[test]
fn smatrix_from_overlap_zero() {
    let s = SMatrix2x2::from_overlap(0.0);
    assert!(s.s21.abs() < 1e-10, "Zero overlap → T=0");
    assert!((s.s11 - 1.0).abs() < 1e-10, "Zero overlap → R=1");
}

#[test]
fn eme_mode_overlap_self_is_unity() {
    let field = vec![1.0f64; 100];
    let dx = 10e-9;
    let mode = EmeMode {
        n_eff: 2.5,
        beta: 1e7,
        field,
        dx,
    };
    let ov = mode.overlap(&mode);
    assert!((ov - 1.0).abs() < 1e-10, "Self-overlap={ov:.6}");
}
