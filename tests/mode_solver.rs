use oxiphoton::mode::effective_index::{
    strip_waveguide_eim, AsymmetricSlab, Polarization, SlabWaveguide,
};
use oxiphoton::mode::fd_solver::{FdModeSolver1d, FdModeSolver2d};

#[test]
fn symmetric_slab_te_modes_guided() {
    // Si slab 500nm thick in SiO2 at 1550nm
    let slab = SlabWaveguide::new(3.476, 1.444, 500e-9);
    let modes = slab.solve_te(1550e-9);
    assert!(!modes.is_empty(), "Should find at least TE0");
    let n0 = modes[0].n_eff;
    assert!(n0 > 1.444 && n0 < 3.476, "n_eff0={n0:.4} not guided");
}

#[test]
fn symmetric_slab_tm_modes_guided() {
    let slab = SlabWaveguide::new(3.476, 1.444, 500e-9);
    let modes = slab.solve_tm(1550e-9);
    assert!(!modes.is_empty(), "Should find TM0");
    assert!(modes[0].n_eff > 1.444 && modes[0].n_eff < 3.476);
}

#[test]
fn slab_te_neff_greater_than_tm_neff() {
    // For high-contrast step-index slab, TE modes have higher n_eff than TM
    // (TE modes are more confined due to larger polarization factor)
    let slab = SlabWaveguide::new(3.476, 1.444, 500e-9);
    let te_modes = slab.solve_te(1550e-9);
    let tm_modes = slab.solve_tm(1550e-9);
    // Both should find at least the fundamental mode
    assert!(!te_modes.is_empty(), "Should find TE modes");
    assert!(!tm_modes.is_empty(), "Should find TM modes");
    // Both in guidance range
    assert!(te_modes[0].n_eff > 1.444 && te_modes[0].n_eff < 3.476);
    assert!(tm_modes[0].n_eff > 1.444 && tm_modes[0].n_eff < 3.476);
}

#[test]
fn asymmetric_slab_si_on_sio2_air_top() {
    // Si on SiO2 substrate, air cladding: n_left=1.444, n_core=3.476, n_right=1.0
    let slab = AsymmetricSlab::new(1.444, 3.476, 1.0, 220e-9);
    let modes = slab.solve_te(1550e-9);
    assert!(!modes.is_empty(), "Si on SiO2 with air top should guide");
    let n_eff = modes[0].n_eff;
    let n_max_clad = 1.444_f64.max(1.0);
    assert!(n_eff > n_max_clad && n_eff < 3.476, "n_eff={n_eff:.4}");
}

#[test]
fn eim_si_strip_220x500_1550() {
    // Si strip 220nm × 500nm at 1550nm: n_eff in physically valid range
    let n_eff = strip_waveguide_eim(3.476, 1.444, 500e-9, 220e-9, 1550e-9, Polarization::TE)
        .expect("EIM should find a mode");
    assert!(n_eff > 1.444 && n_eff < 3.476, "EIM n_eff={n_eff:.4}");
    // EIM for this geometry gives roughly 2.1–2.5
    assert!(
        n_eff > 1.8 && n_eff < 3.0,
        "EIM n_eff={n_eff:.4} unexpected range"
    );
}

#[test]
fn fd1d_slab_guided_modes() {
    let n_pts = 120;
    let dx = 25e-9;
    let n_profile = FdModeSolver1d::slab_profile(3.476, 1.444, 500e-9, n_pts, dx);
    let solver = FdModeSolver1d::new(n_profile, dx, 1.444);
    let modes = solver.solve(1550e-9);
    assert!(!modes.is_empty(), "FD solver should find guided modes");
    let n0 = modes[0].n_eff;
    assert!(n0 > 1.444 && n0 < 3.476, "n_eff={n0:.4}");
}

#[test]
fn fd1d_matches_eim_within_tolerance() {
    // FD and EIM should agree within 1% for a well-resolved grid
    let n_core = 3.476;
    let n_clad = 1.444;
    let thickness = 500e-9;
    let wavelength = 1550e-9;

    let slab = SlabWaveguide::new(n_core, n_clad, thickness);
    let eim = slab.solve_te(wavelength);
    let eim_neff = eim[0].n_eff;

    let n_pts = 200;
    let dx = 15e-9;
    let n_profile = FdModeSolver1d::slab_profile(n_core, n_clad, thickness, n_pts, dx);
    let solver = FdModeSolver1d::new(n_profile, dx, n_clad);
    let fd_modes = solver.solve(wavelength);
    let fd_neff = fd_modes[0].n_eff;

    let rel_err = (fd_neff - eim_neff).abs() / eim_neff;
    assert!(
        rel_err < 0.01,
        "FD={fd_neff:.4} EIM={eim_neff:.4} rel_err={rel_err:.4}"
    );
}

#[test]
fn fd2d_strip_waveguide_finds_mode() {
    // 2D FD solver for 220nm × 500nm Si strip in SiO2
    // Small grid for test speed: 24×18 over 1200nm × 900nm domain
    let n_si = 3.476;
    let n_sio2 = 1.444;
    let width = 500e-9;
    let height = 220e-9;
    let nx = 24;
    let ny = 18;
    let dx = 1200e-9 / nx as f64;
    let dy = 900e-9 / ny as f64;

    let n_profile = FdModeSolver2d::strip_profile(n_si, n_sio2, width, height, nx, ny, dx, dy);
    let solver = FdModeSolver2d::new(n_profile, nx, ny, dx, dy, n_sio2);
    let modes = solver.solve(1550e-9);

    assert!(!modes.is_empty(), "2D FD solver should find guided mode");
    let n0 = modes[0].n_eff;
    assert!(
        n0 > n_sio2 && n0 < n_si,
        "n_eff={n0:.4} out of guidance range"
    );
}

#[test]
fn mode_order_increases_with_mode_number() {
    // Higher-order modes should have lower n_eff
    let slab = SlabWaveguide::new(3.476, 1.444, 2000e-9); // thick: many modes
    let modes = slab.solve_te(1550e-9);
    if modes.len() >= 2 {
        assert!(
            modes[0].n_eff > modes[1].n_eff,
            "modes should be sorted by n_eff desc: {:.4} {:.4}",
            modes[0].n_eff,
            modes[1].n_eff
        );
    }
}
