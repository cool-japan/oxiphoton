//! Waveguide mode analysis example.
//!
//! Demonstrates effective index method and FD mode solver for a
//! Silicon strip waveguide at 1550 nm:
//!   - Core: Si, n ≈ 3.48
//!   - Cladding: SiO2, n ≈ 1.44
//!   - Dimensions: 500 nm wide × 220 nm tall

use oxiphoton::material::MaterialDatabase;
use oxiphoton::mode::{strip_waveguide_eim, FdModeSolver1d, Polarization, SlabWaveguide};
use oxiphoton::units::Wavelength;

fn main() {
    let wavelength = Wavelength::from_nm(1550.0);
    let db = MaterialDatabase::load_default();
    let si = db.get("Si").unwrap();
    let sio2 = db.get("SiO2").unwrap();

    let wl = wavelength.0;
    let n_core = si.refractive_index(wavelength).n;
    let n_clad = sio2.refractive_index(wavelength).n;
    let width = 500e-9;
    let height = 220e-9;

    println!("=== Silicon Strip Waveguide Mode Analysis ===");
    println!("λ = {:.0} nm", wavelength.as_nm());
    println!("n_core (Si)   = {n_core:.4}");
    println!("n_clad (SiO2) = {n_clad:.4}");
    println!(
        "Width × Height = {:.0} nm × {:.0} nm",
        width * 1e9,
        height * 1e9
    );
    println!();

    // EIM for strip waveguide (approximate 2D effective index)
    match strip_waveguide_eim(n_core, n_clad, width, height, wl, Polarization::TE) {
        Some(n_eff) => println!("EIM effective index (TE): n_eff ≈ {n_eff:.4}"),
        None => println!("EIM: no guided TE mode found"),
    }
    match strip_waveguide_eim(n_core, n_clad, width, height, wl, Polarization::TM) {
        Some(n_eff) => println!("EIM effective index (TM): n_eff ≈ {n_eff:.4}"),
        None => println!("EIM: no guided TM mode found"),
    }
    println!();

    // 1D slab waveguide — vertical direction (height)
    println!(
        "--- 1D Slab (vertical, height = {:.0} nm) ---",
        height * 1e9
    );
    let slab_v = SlabWaveguide::new(n_core, n_clad, height);
    let modes_v = slab_v.solve_te(wl);
    println!("Found {} TE mode(s):", modes_v.len());
    for m in &modes_v {
        println!("  TE{}: n_eff = {:.4}", m.order, m.n_eff);
    }
    println!();

    // 1D slab waveguide — horizontal direction (width)
    println!(
        "--- 1D Slab (horizontal, width = {:.0} nm) ---",
        width * 1e9
    );
    let slab_h = SlabWaveguide::new(n_core, n_clad, width);
    let modes_h = slab_h.solve_te(wl);
    println!("Found {} TE mode(s):", modes_h.len());
    for m in &modes_h {
        println!("  TE{}: n_eff = {:.4}", m.order, m.n_eff);
    }
    println!();

    // FD 1D mode solver for vertical slab (using slab_profile helper)
    println!("--- FD 1D Mode Solver (vertical slab) ---");
    let dx = 5e-9; // 5 nm grid
    let n_pts = 201;
    let n_profile = FdModeSolver1d::slab_profile(n_core, n_clad, height, n_pts, dx);
    let solver = FdModeSolver1d::new(n_profile, dx, n_clad);
    let fd_modes = solver.solve(wl);
    println!("Found {} FD guided mode(s):", fd_modes.len());
    for (i, m) in fd_modes.iter().enumerate() {
        let beta = 2.0 * std::f64::consts::PI * m.n_eff / wl;
        println!(
            "  Mode {i}: n_eff = {:.4},  β = {:.4e} rad/m",
            m.n_eff, beta
        );
    }

    println!("\nDone.");
}
