use oxiphoton::photonic_crystal::pwe2d::jacobi_eigh;
use oxiphoton::photonic_crystal::{
    bessel_j1, kpath_hexagonal, kpath_square, BandStructure, PhCrystal2d, Polarization,
};

#[test]
fn bessel_j1_known_values() {
    assert!((bessel_j1(0.0) - 0.0).abs() < 1e-14);
    assert!((bessel_j1(1.0) - 0.4400505857449335).abs() < 1e-8);
    // First zero near 3.8317
    assert!(bessel_j1(3.8317).abs() < 1e-3);
}

#[test]
fn kpath_square_point_count() {
    let n = 10;
    let path = kpath_square(n);
    assert_eq!(path.len(), n * 3 + 1);
    // Check endpoints
    assert!((path[0][0]).abs() < 1e-14 && path[0][1].abs() < 1e-14); // Γ
    assert!((path[n][0] - 0.5).abs() < 1e-14); // X
    assert!((path[n * 2][0] - 0.5).abs() < 1e-14 && (path[n * 2][1] - 0.5).abs() < 1e-14); // M
    assert!((path[n * 3][0]).abs() < 1e-14 && path[n * 3][1].abs() < 1e-14); // Γ
}

#[test]
fn jacobi_eigh_4x4_diagonal() {
    let n = 4;
    let diag = [4.0, 2.0, 1.0, 3.0];
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        a[i * n + i] = diag[i];
    }
    let eigs = jacobi_eigh(&mut a, n, 1e-12, 50);
    // Should return eigenvalues sorted ascending
    assert_eq!(eigs.len(), n);
    assert!((eigs[0] - 1.0).abs() < 1e-10);
    assert!((eigs[1] - 2.0).abs() < 1e-10);
    assert!((eigs[2] - 3.0).abs() < 1e-10);
    assert!((eigs[3] - 4.0).abs() < 1e-10);
}

#[test]
fn empty_lattice_gamma_point_zero_freq() {
    // ε = 1 uniform: TM at Γ (k=[0,0]) should have lowest eigenvalue = 0
    let crystal = PhCrystal2d::square_rods(1.0, 1.0, 0.3, 4);
    let freqs = crystal.solve_tm([0.0, 0.0]);
    assert!(
        freqs[0].abs() < 1e-8,
        "lowest TM band at Gamma = {}",
        freqs[0]
    );
}

#[test]
fn square_lattice_si_rods_tm_gap() {
    // Joannopoulos Si rods: eps_rod=11.4, r/a=0.2, fill=π*0.2²≈0.1257
    let fill = std::f64::consts::PI * 0.2f64.powi(2);
    let crystal = PhCrystal2d::square_rods(11.4, 1.0, fill, 6);
    let k_path = kpath_square(20);
    let bands = crystal.band_diagram(&k_path, Polarization::TM);
    // Find gap between band 0 and band 1
    let band0_max = bands.bands[0]
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let band1_min = bands.bands[1].iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        band1_min > band0_max,
        "No TM gap found: band0_max={:.4}, band1_min={:.4}",
        band0_max,
        band1_min
    );
    let midgap = (band0_max + band1_min) / 2.0;
    let gap_frac = (band1_min - band0_max) / midgap;
    // PWE with n_g=6 gives midgap ≈ 0.375 (converging toward MPB value ~0.372)
    assert!(
        midgap > 0.33 && midgap < 0.42,
        "TM midgap {:.4} not in expected range (0.33, 0.42)",
        midgap
    );
    assert!(gap_frac > 0.20, "TM gap fraction {:.4} < 20%", gap_frac);
}

#[test]
fn hex_lattice_air_holes_te_gap() {
    // Joannopoulos hex air holes: eps_bg=12, eps_hole=1, r/a=0.45
    // fill = π * 0.45² / (sqrt(3)/2) ≈ 0.6545 (hex cell area = sqrt(3)/2)
    let r_over_a = 0.45f64;
    let cell_area = (3.0f64.sqrt()) / 2.0; // hexagonal unit cell area for a=1
    let fill = std::f64::consts::PI * r_over_a.powi(2) / cell_area;
    // Clamp fill to valid range
    let fill = fill.min(0.99);
    let crystal = PhCrystal2d::hex_holes(12.0, 1.0, fill, 5);
    let k_path = kpath_hexagonal(20);

    let te_bands = crystal.band_diagram(&k_path, Polarization::TE);
    let te_b0_max = te_bands.bands[0]
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let te_b1_min = te_bands.bands[1]
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    assert!(
        te_b1_min > te_b0_max,
        "Expected TE gap: band0_max={:.4}, band1_min={:.4}",
        te_b0_max,
        te_b1_min
    );
}

#[test]
fn convergence_in_n_g() {
    // TM midgap of square Si rods should be stable to <5% between n_g=4 and n_g=6.
    // (PWE converges from above toward MPB value ~0.372; n_g=4→6 difference ~4%.)
    let fill = std::f64::consts::PI * 0.2f64.powi(2);
    let k_path = kpath_square(15);

    let c1 = PhCrystal2d::square_rods(11.4, 1.0, fill, 4);
    let b1 = c1.band_diagram(&k_path, Polarization::TM);
    let g1 = (b1.bands[0]
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        + b1.bands[1].iter().cloned().fold(f64::INFINITY, f64::min))
        / 2.0;

    let c2 = PhCrystal2d::square_rods(11.4, 1.0, fill, 6);
    let b2 = c2.band_diagram(&k_path, Polarization::TM);
    let g2 = (b2.bands[0]
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        + b2.bands[1].iter().cloned().fold(f64::INFINITY, f64::min))
        / 2.0;

    let rel_diff = ((g1 - g2) / g2).abs();
    assert!(
        rel_diff < 0.05,
        "Midgap convergence: n_g=4 gives {:.4}, n_g=6 gives {:.4}, diff={:.2}%",
        g1,
        g2,
        rel_diff * 100.0
    );
}

#[test]
fn band_structure_has_gaps_field() {
    let fill = std::f64::consts::PI * 0.2f64.powi(2);
    let crystal = PhCrystal2d::square_rods(11.4, 1.0, fill, 4);
    let k_path = kpath_square(10);
    let mut bands: BandStructure = crystal.band_diagram(&k_path, Polarization::TM);
    // find_gaps is called internally, but also works if called again
    bands.find_gaps();
    // Gaps vec should exist (may be empty or not)
    let _ = bands.gaps.len();
}
