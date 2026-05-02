use oxiphoton::solar::absorption::photon_recycling_factor;

#[test]
fn recycling_thin_limit_escape_unity() {
    // alpha * 4n2 * d = 1e-6: escape_prob ≈ 1, factor ≈ 1
    let n = 3.5_f64;
    let four_n2 = 4.0 * n * n;
    let d = 1e-6 / (1e2 * four_n2); // gives alpha*4n2*d = 1e-6
    let factor = photon_recycling_factor(0.5, 1e2, d, n);
    assert!((factor - 1.0).abs() < 0.01, "thin limit: factor={factor}");
}

#[test]
fn recycling_thick_limit_yablonovitch() {
    // alpha * 4n2 * d very large: escape_prob ≈ 1/(4n2)
    let n = 3.5_f64;
    let four_n2 = 4.0 * n * n;
    let alpha = 1e5_f64;
    let d = 1e3 / (alpha * four_n2); // alpha*4n2*d = 1e3
    let q = 0.99_f64;
    let factor = photon_recycling_factor(q, alpha, d, n);
    let expected = 1.0 / (1.0 - q * (1.0 - 1.0 / four_n2));
    assert!(
        (factor - expected).abs() / expected < 1e-3,
        "thick limit: factor={factor}, expected={expected}"
    );
}

#[test]
fn recycling_monotone_in_thickness() {
    let n = 3.5_f64;
    let alpha = 1e4_f64;
    let q = 0.9_f64;
    let thicknesses: Vec<f64> = (0..20)
        .map(|i| 10e-9 * (100f64).powf(i as f64 / 19.0))
        .collect();
    let factors: Vec<f64> = thicknesses
        .iter()
        .map(|&d| photon_recycling_factor(q, alpha, d, n))
        .collect();
    for w in factors.windows(2) {
        assert!(
            w[1] >= w[0] - 1e-10,
            "not monotone: {:.6} > {:.6}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn recycling_monotone_in_alpha() {
    let n = 3.5_f64;
    let d = 200e-6_f64;
    let q = 0.9_f64;
    let alphas: Vec<f64> = (0..20)
        .map(|i| 1e2 * (1e3f64).powf(i as f64 / 19.0))
        .collect();
    let factors: Vec<f64> = alphas
        .iter()
        .map(|&a| photon_recycling_factor(q, a, d, n))
        .collect();
    for w in factors.windows(2) {
        assert!(
            w[1] >= w[0] - 1e-10,
            "not monotone: {:.6} > {:.6}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn recycling_factor_at_least_unity() {
    for &q in &[0.0, 0.5, 0.99] {
        for &alpha in &[1e2, 1e4, 1e6] {
            for &d in &[1e-7, 1e-5, 1e-3] {
                let f = photon_recycling_factor(q, alpha, d, 3.5);
                assert!(
                    f >= 1.0 - 1e-10,
                    "factor < 1: q={q}, alpha={alpha}, d={d}, f={f}"
                );
            }
        }
    }
}

#[test]
fn recycling_factor_q_zero_returns_unity() {
    let f = photon_recycling_factor(0.0, 1e4, 200e-6, 3.5);
    assert!((f - 1.0).abs() < 1e-12, "q=0 should give factor=1, got {f}");
}

#[test]
fn airy_orphan_block_removed() {
    let source = include_str!("../src/solar/absorption.rs");
    assert!(
        !source.contains("let _ = (r01"),
        "orphan Airy discard `let _ = (r01, ...)` is still present — remove it"
    );
}
