//! S-matrix and RCWA validation tests.
//!
//! Validates the S-matrix, transfer matrix, and EME solver implementations
//! against known analytical solutions.

use num_complex::Complex64;
use oxiphoton::smatrix::eigenmode::{
    cascade_smatrices, extract_mode_amplitudes, eye_nd, mat_add_nd, mat_mul_nd, EigenMode, EmeMode,
    EmeSegment, EmeSolver, SMatrix2x2, ThinFilmLayer, TransferMatrixSystem,
};

// ─────────────────────────────────────────────────────────────────────────────
// Transfer matrix / Fresnel coefficient tests
// ─────────────────────────────────────────────────────────────────────────────

/// Transfer matrix for single interface gives correct Fresnel coefficients.
#[test]
fn single_interface_fresnel_reflection() {
    let n1 = 1.0_f64;
    let n2 = 1.5_f64;
    let tmm = TransferMatrixSystem::new(n1, n2);
    let r = tmm.reflectance();
    let r_fresnel = ((n1 - n2) / (n1 + n2)).powi(2);
    assert!(
        (r - r_fresnel).abs() < 1e-10,
        "R={r} vs Fresnel={r_fresnel}"
    );
}

/// Air→glass→air symmetric stack: overall R + T = 1.
#[test]
fn symmetric_stack_energy_conservation() {
    let mut tmm = TransferMatrixSystem::new(1.0, 1.0);
    tmm.add_layer(ThinFilmLayer::new(1.5, 200e-9, 1550e-9));
    let r = tmm.reflectance();
    let t = tmm.transmittance();
    assert!((r + t - 1.0).abs() < 1e-10, "R+T={}", r + t);
}

/// Bragg mirror high-reflection band: 5-period λ/4 stack should have high R.
#[test]
fn bragg_mirror_high_reflection_at_center() {
    let lambda = 1550e-9_f64;
    let n_h = 2.3_f64; // TiO2
    let n_l = 1.46_f64; // SiO2
    let d_h = lambda / (4.0 * n_h);
    let d_l = lambda / (4.0 * n_l);
    let mut tmm = TransferMatrixSystem::new(1.0, 1.5);
    for _ in 0..5 {
        tmm.add_layer(ThinFilmLayer::new(n_h, d_h, lambda));
        tmm.add_layer(ThinFilmLayer::new(n_l, d_l, lambda));
    }
    let r = tmm.reflectance();
    assert!(
        r > 0.9,
        "5-period Bragg mirror should have R > 90%, got R={r:.4}"
    );
}

/// Quarter-wave anti-reflection coating: R → 0.
#[test]
fn quarter_wave_ar_coating_near_zero_reflection() {
    let n1 = 1.0_f64;
    let n2 = 1.5_f64;
    let n_ar = (n1 * n2).sqrt();
    let lambda = 1550e-9_f64;
    let d_ar = lambda / (4.0 * n_ar);
    let mut tmm = TransferMatrixSystem::new(n1, n2);
    tmm.add_layer(ThinFilmLayer::new(n_ar, d_ar, lambda));
    let r = tmm.reflectance();
    assert!(r < 1e-10, "Ideal AR coating: R={r}");
}

/// Half-wave layer is transparent (R same as bare interface).
#[test]
fn half_wave_layer_transparent() {
    let lambda = 1550e-9_f64;
    let n_layer = 2.0_f64;
    let d_hw = lambda / (2.0 * n_layer); // half-wave layer
    let n1 = 1.0_f64;
    let n2 = 1.5_f64;
    let tmm_bare = TransferMatrixSystem::new(n1, n2);
    let r_bare = tmm_bare.reflectance();
    let mut tmm_hw = TransferMatrixSystem::new(n1, n2);
    tmm_hw.add_layer(ThinFilmLayer::new(n_layer, d_hw, lambda));
    let r_hw = tmm_hw.reflectance();
    assert!(
        (r_hw - r_bare).abs() < 1e-8,
        "HW layer: R={r_hw} vs bare R={r_bare}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// EME solver tests
// ─────────────────────────────────────────────────────────────────────────────

/// EME total transmission + reflection ≤ 1 (energy conservation, inequality
/// due to imperfect mode coverage).
#[test]
fn eme_energy_conservation_bound() {
    let mut solver = EmeSolver::new(1550e-9, 2, 100);
    solver.add_segment(EmeSegment::new(5e-6, 3.476, 1.444, 500e-9));
    solver.add_segment(EmeSegment::new(5e-6, 3.476, 1.444, 800e-9));
    let s = solver.solve_fundamental();
    let t = s.s21 * s.s21;
    let r = s.s11 * s.s11;
    assert!(t + r <= 1.0 + 1e-10, "T+R={} > 1", t + r);
}

/// S-matrix cascade: two identical sections should give near-unity transmission.
#[test]
fn smatrix_cascade_identical_sections() {
    let mut solver = EmeSolver::new(1550e-9, 2, 150);
    solver.add_segment(EmeSegment::new(10e-6, 3.476, 1.444, 500e-9));
    solver.add_segment(EmeSegment::new(10e-6, 3.476, 1.444, 500e-9));
    let t = solver.transmission();
    assert!(t > 0.9, "Identical sections: T={t}");
}

/// S-matrix identity cascade: cascading with itself leaves S unchanged.
#[test]
fn smatrix_2x2_cascade_identity_left() {
    let s = SMatrix2x2 {
        s11: 0.1,
        s12: 0.9,
        s21: 0.9,
        s22: 0.1,
    };
    let id = SMatrix2x2::identity();
    let result = id.cascade(&s);
    assert!(
        (result.s11 - s.s11).abs() < 1e-10,
        "S11: {} vs {}",
        result.s11,
        s.s11
    );
    assert!(
        (result.s21 - s.s21).abs() < 1e-10,
        "S21: {} vs {}",
        result.s21,
        s.s21
    );
}

/// S-matrix cascade: reflective interface reduces transmission.
#[test]
fn smatrix_cascade_interface_reduces_transmission() {
    let s_lossless = SMatrix2x2::identity();
    let s_partial = SMatrix2x2::from_overlap(0.7); // 70% overlap
    let combined = s_lossless.cascade(&s_partial);
    let t = combined.s21 * combined.s21;
    assert!(t < 1.0, "Combined transmission={t} should be < 1");
    assert!(t > 0.0, "Combined transmission={t} should be > 0");
}

// ─────────────────────────────────────────────────────────────────────────────
// N-port S-matrix cascade tests
// ─────────────────────────────────────────────────────────────────────────────

/// Cascade of two identity N×N S-matrices yields identity.
#[test]
fn nd_cascade_two_identities() {
    let n = 3_usize;
    let id = eye_nd(n);
    let zero = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    let (s11, _s12, s21, s22) = cascade_smatrices(&zero, &id, &id, &zero, &zero, &id, &id, &zero);
    for i in 0..n {
        assert!(
            (s21[i][i].re - 1.0).abs() < 1e-10,
            "S21[{i}][{i}] should be 1"
        );
        assert!(s11[i][i].norm() < 1e-10, "S11[{i}][{i}] should be 0");
        assert!(s22[i][i].norm() < 1e-10, "S22[{i}][{i}] should be 0");
    }
}

/// mat_mul_nd: identity × A = A.
#[test]
fn mat_mul_nd_identity_right() {
    let n = 4_usize;
    let id = eye_nd(n);
    let a: Vec<Vec<Complex64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| Complex64::new((i * n + j) as f64, i as f64))
                .collect()
        })
        .collect();
    let result = mat_mul_nd(&a, &id);
    for i in 0..n {
        for j in 0..n {
            assert!((result[i][j] - a[i][j]).norm() < 1e-10);
        }
    }
}

/// mat_add_nd: A + 0 = A.
#[test]
fn mat_add_nd_zero() {
    let n = 3_usize;
    let a: Vec<Vec<Complex64>> = (0..n)
        .map(|i| (0..n).map(|j| Complex64::new(i as f64, j as f64)).collect())
        .collect();
    let zero = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    let result = mat_add_nd(&a, &zero);
    for i in 0..n {
        for j in 0..n {
            assert!((result[i][j] - a[i][j]).norm() < 1e-14);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mode extraction tests
// ─────────────────────────────────────────────────────────────────────────────

/// Extract mode amplitudes: field = mode → amplitude = 1.
#[test]
fn extract_amplitudes_self_amplitude_one() {
    let n = 64_usize;
    let dx = 10e-9_f64;
    let field: Vec<f64> = (0..n)
        .map(|i| (std::f64::consts::PI * i as f64 / n as f64).sin())
        .collect();
    let mode = EmeMode {
        n_eff: 2.5,
        beta: 1e7,
        field: field.clone(),
        dx,
    };
    let amps = extract_mode_amplitudes(&field, &[mode], dx);
    assert!((amps[0].re - 1.0).abs() < 1e-10, "amplitude={}", amps[0].re);
}

/// Extract mode amplitudes: orthogonal field → zero amplitude.
///
/// The even Gaussian f(x) = exp(-x²/σ²) is orthogonal to the odd function
/// g(x) = x·exp(-x²/σ²) because ∫f·g dx = ∫x·exp(-2x²/σ²) dx = 0 by symmetry.
/// To avoid boundary artefacts the Gaussian width σ must be small enough that
/// both functions are numerically zero at the grid edges (± n/2 · dx).
#[test]
fn extract_amplitudes_orthogonal_near_zero() {
    let n = 128_usize;
    let dx = 10e-9_f64;
    // σ = 100 nm: at boundary x = ±640 nm the Gaussian is exp(-(640/100)²) ≈ 2e-18,
    // so the discrete grid is effectively symmetric and the overlap cancels to < 1e-6.
    let sigma = 100e-9_f64;
    let field: Vec<f64> = (0..n)
        .map(|i| {
            let x = (i as f64 - n as f64 / 2.0) * dx;
            (-x * x / (sigma * sigma)).exp()
        })
        .collect();
    let mode_field: Vec<f64> = (0..n)
        .map(|i| {
            let x = (i as f64 - n as f64 / 2.0) * dx;
            x * (-x * x / (sigma * sigma)).exp()
        })
        .collect();
    let mode = EmeMode {
        n_eff: 2.3,
        beta: 9e6,
        field: mode_field,
        dx,
    };
    let amps = extract_mode_amplitudes(&field, &[mode], dx);
    assert!(
        amps[0].norm() < 1e-6,
        "orthogonal amplitude={}",
        amps[0].norm()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// DWDM channel plan tests
// ─────────────────────────────────────────────────────────────────────────────

mod wdm_tests {
    use oxiphoton::interconnect::wdm::DwdmChannelPlan;

    /// DWDM channel plan: correct 100 GHz spacing.
    #[test]
    fn dwdm_channel_plan_spacing_100ghz() {
        let plan = DwdmChannelPlan::new(1550.0, 100.0, 8);
        let freqs = plan.channel_frequencies_thz();
        for w in freqs.windows(2) {
            let df_ghz = (w[1] - w[0]) * 1e3;
            assert!((df_ghz - 100.0).abs() < 0.01, "spacing={df_ghz} GHz");
        }
    }

    /// Channel wavelengths in the C-band.
    #[test]
    fn dwdm_channel_wavelengths_c_band() {
        let plan = DwdmChannelPlan::new(1550.0, 100.0, 8);
        let wls = plan.channel_wavelengths_nm();
        for wl in wls {
            assert!(wl > 1530.0 && wl < 1570.0, "wl={wl} nm outside C-band");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Link budget tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn link_budget_margin_calculation() {
    use oxiphoton::interconnect::wdm::LinkBudget;
    let mut budget = LinkBudget::new();
    budget.add_fiber(80.0, 0.2); // -16 dB
    budget.add_amplifier(20.0); // +20 dB
    budget.add_fiber(80.0, 0.2); // -16 dB
                                 // Net = 20 - 16 - 16 = -12 dB
                                 // Margin = 0 + (-12) - (-25) = 13 dB
    let margin = budget.margin_db(-25.0, 0.0);
    assert!((margin - 13.0).abs() < 1e-8, "margin={margin}");
}

#[test]
fn link_budget_total_loss_correct() {
    use oxiphoton::interconnect::wdm::LinkBudget;
    let mut budget = LinkBudget::new();
    budget.add_fiber(100.0, 0.2); // 20 dB loss
    budget.add_component("Connector", -0.5);
    assert!(
        (budget.total_loss_db() - 20.5).abs() < 1e-8,
        "total_loss={}",
        budget.total_loss_db()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// EigenMode propagator energy conservation
// ─────────────────────────────────────────────────────────────────────────────

/// Lossless eigenmode propagator preserves power.
#[test]
fn eigenmode_propagator_lossless_energy_conservation() {
    use oxiphoton::smatrix::eigenmode::EigenmodePropagator;
    let n = 32_usize;
    let dx = 5e-9_f64;
    let field: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((std::f64::consts::PI * i as f64 / n as f64).sin(), 0.0))
        .collect();
    let mode = EigenMode {
        beta: Complex64::new(1.2e7, 0.0),
        field: field.clone(),
        dx,
    };
    let prop = EigenmodePropagator::new(vec![mode]);
    let out = prop.propagate_forward(&field, 5e-6);
    let p_in: f64 = field.iter().map(|e| e.norm_sqr()).sum();
    let p_out: f64 = out.iter().map(|e| e.norm_sqr()).sum();
    assert!(
        (p_out / p_in - 1.0).abs() < 1e-6,
        "power ratio={}",
        p_out / p_in
    );
}

/// Lossy eigenmode propagator reduces power.
#[test]
fn eigenmode_propagator_lossy_reduces_power() {
    use oxiphoton::smatrix::eigenmode::EigenmodePropagator;
    let n = 32_usize;
    let dx = 5e-9_f64;
    let field: Vec<Complex64> = vec![Complex64::new(1.0, 0.0); n];
    let mode = EigenMode {
        beta: Complex64::new(1.2e7, 1e5), // lossy: β_i = 1e5 rad/m
        field: field.clone(),
        dx,
    };
    let prop = EigenmodePropagator::new(vec![mode]);
    let out = prop.propagate_forward(&field, 1e-5);
    let p_in: f64 = field.iter().map(|e| e.norm_sqr()).sum();
    let p_out: f64 = out.iter().map(|e| e.norm_sqr()).sum();
    assert!(p_out < p_in, "Lossy: p_out={p_out} should be < p_in={p_in}");
}
