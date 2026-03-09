//! Anisotropic Crystal FDTD Example — birefringent LiNbO₃ waveguide simulation.
//!
//! Demonstrates how a uniaxial crystal (LiNbO₃) causes ordinary and extraordinary
//! polarised waves to propagate at different phase velocities, the core physical
//! effect exploited in electro-optic modulators and second-harmonic generators.
//!
//! Physics recap
//! -------------
//! In a uniaxial crystal with optic axis along z the permittivity tensor is diagonal:
//!   ε = diag(no², no², ne²)
//! The ordinary ray (E ⊥ c-axis) sees n = no, the extraordinary ray (E ∥ c-axis) n = ne.
//! The phase velocity difference Δv = c/no − c/ne produces a walk-off between the two
//! polarisations after propagation through a birefringent medium.

use std::f64::consts::PI;

use oxiphoton::fdtd::{
    fill_uniaxial_crystal, AnisotropicFdtd3d, GyroelectricMedium, UniaxialCrystal,
};

const C0: f64 = 2.99792458e8; // m/s — speed of light in vacuum

fn main() {
    println!("=== Anisotropic Crystal FDTD Example ===");
    println!();

    // ── 1. Create UniaxialCrystal for LiNbO₃ ────────────────────────────────
    // LiNbO₃ refractive indices at λ = 1550 nm (telecom C-band)
    //   Ordinary (e-polarisation in x-y plane): no = 2.211
    //   Extraordinary (e-polarisation along z / optic axis): ne = 2.138
    // The crystal is negative uniaxial (ne < no).
    let no = 2.211_f64; // ordinary index
    let ne = 2.138_f64; // extraordinary index

    let linbo3 = UniaxialCrystal::new(no, ne, [0.0, 0.0, 1.0])
        .expect("LiNbO3 crystal creation should succeed");

    println!("LiNbO3 (lithium niobate) at λ = 1550 nm");
    println!("  Ordinary index   no = {:.3}", no);
    println!("  Extraordinary index ne = {:.3}", ne);
    println!("  Birefringence |Δn| = {:.4}", linbo3.birefringence());
    println!();

    // ── 2. Print permittivity tensor ─────────────────────────────────────────
    // For optic axis along z: ε = diag(no², no², ne²)
    let tensor = linbo3.permittivity_tensor();
    println!("Permittivity tensor ε (relative, optic axis ∥ z):");
    for (i, row) in tensor.iter().enumerate() {
        println!("  [{:.4}  {:.4}  {:.4}]", row[0], row[1], row[2]);
        let _ = i;
    }
    println!();

    // ── 3. Show phase velocity difference ────────────────────────────────────
    // Ordinary ray: E in x-y plane, propagating along z → sees ε_xx = no²
    // Extraordinary ray: E along z, propagating in x-y plane → sees ε_zz = ne²
    let v_o = linbo3.phase_velocity_ordinary();
    let v_e = linbo3.phase_velocity_extraordinary();
    println!("Phase velocities:");
    println!("  Ordinary ray:      v_o = c / no = {:.4e} m/s", v_o);
    println!("  Extraordinary ray: v_e = c / ne = {:.4e} m/s", v_e);
    let delta_v = (v_o - v_e).abs();
    println!("  Phase velocity difference |Δv| = {:.4e} m/s", delta_v);
    println!("  Relative difference |Δv|/c      = {:.4e}", delta_v / C0);
    println!();

    // Walk-off length: the distance after which o and e wave accumulate π phase
    // difference → L_pi = λ / (2 Δn)
    let lambda = 1550e-9_f64; // 1550 nm
    let delta_n = linbo3.birefringence();
    let l_pi = lambda / (2.0 * delta_n);
    println!("Half-wave (π) walk-off length at 1550 nm:");
    println!("  L_π = λ / (2Δn) = {:.2} μm", l_pi * 1e6);
    println!();

    // ── 4. Run small AnisotropicFdtd3d simulation (15×15×15) ────────────────
    let nx = 15usize;
    let ny = 15usize;
    let nz = 15usize;

    // Cell size 20 nm — resolves 1550 nm wavelength with ~77 cells/wavelength
    let dx = 20.0e-9_f64;
    let dy = dx;
    let dz = dx;

    // Courant-stable time step for 3D anisotropic medium:
    //   dt ≤ dx / (c√3 · max(√ε))
    // For LiNbO₃ max √ε ≈ no ≈ 2.211, so we use a conservative factor.
    let dt = 0.9 * dx / (C0 * 3.0_f64.sqrt() * no);

    println!(
        "FDTD simulation: {}×{}×{} cells, dx = {:.0} nm, dt = {:.3e} s",
        nx,
        ny,
        nz,
        dx * 1e9,
        dt
    );

    let mut fdtd = AnisotropicFdtd3d::new(nx, ny, nz, dx, dy, dz, dt);

    // ── 5. Fill centre with LiNbO₃ (optic axis along z) ────────────────────
    // Fill a 9×9×9 box centred on the grid (indices 3..12 in each direction)
    let margin = 3usize;
    fill_uniaxial_crystal(
        &mut fdtd,
        &linbo3,
        margin,
        nx - margin,
        margin,
        ny - margin,
        margin,
        nz - margin,
    );

    // Verify material assignment at grid centre
    let cx = nx / 2;
    let cy = ny / 2;
    let cz = nz / 2;
    let centre = fdtd.idx(cx, cy, cz);
    println!("Centre cell permittivity (should be LiNbO3):");
    println!(
        "  ε_xx = {:.4} (expected: no² = {:.4})",
        fdtd.eps_xx[centre],
        no * no
    );
    println!(
        "  ε_yy = {:.4} (expected: no² = {:.4})",
        fdtd.eps_yy[centre],
        no * no
    );
    println!(
        "  ε_zz = {:.4} (expected: ne² = {:.4})",
        fdtd.eps_zz[centre],
        ne * ne
    );
    println!();

    // ── 6. Inject point source and run 50 steps ──────────────────────────────
    // Soft Gaussian pulse as Ez source at grid centre.
    // Source wavelength λ = 1550 nm → angular frequency ω₀ = 2πc/λ
    let omega0 = 2.0 * PI * C0 / lambda;
    let t_width = 3.0 / omega0; // ~3 optical cycles wide (broadband)
    let t_peak = 5.0 * t_width;

    let n_steps = 50usize;
    for step in 0..n_steps {
        let t = step as f64 * dt;
        // Modulated Gaussian pulse
        let envelope = (-0.5 * ((t - t_peak) / t_width).powi(2)).exp();
        let source_val = envelope * (omega0 * (t - t_peak)).cos();
        fdtd.set_point_source_ez(cx, cy, cz, source_val);
        fdtd.step();
    }

    // ── 7. Print final Ez field energy ───────────────────────────────────────
    let ez_energy: f64 = fdtd.ez.iter().map(|&v| v * v).sum();
    let ex_energy: f64 = fdtd.ex.iter().map(|&v| v * v).sum();
    let ey_energy: f64 = fdtd.ey.iter().map(|&v| v * v).sum();
    let total_e_energy = ex_energy + ey_energy + ez_energy;

    println!(
        "After {} FDTD time steps (t = {:.3} fs):",
        n_steps,
        fdtd.current_time() * 1e15
    );
    println!("  Ez field energy (Σ Ez²) = {:.4e}", ez_energy);
    println!("  Ex field energy (Σ Ex²) = {:.4e}", ex_energy);
    println!("  Ey field energy (Σ Ey²) = {:.4e}", ey_energy);
    println!("  Total E field energy     = {:.4e}", total_e_energy);
    // Energy should be non-zero (source injected successfully)
    if total_e_energy > 0.0 {
        println!("  [OK] Fields are non-zero — source injection and propagation verified.");
    }
    println!();

    // ── 8. Create GyroelectricMedium — Faraday rotation in magneto-optics ───
    // Yttrium iron garnet (YIG) — a common magneto-optic material at telecom wavelengths.
    // Gyrotropic parameters (illustrative values):
    //   ε_d = 5.0 (background permittivity, n ≈ 2.24)
    //   ε_g = 0.1 (gyrotropy parameter, proportional to applied magnetic field)
    let eps_d = 5.0_f64;
    let eps_g = 0.1_f64;
    let gyro = GyroelectricMedium::new(eps_d, eps_g, [0.0, 0.0, 1.0])
        .expect("GyroelectricMedium creation should succeed");

    let (n_plus, n_minus) = gyro.circular_indices();
    println!("Gyroelectric medium (magneto-optic, YIG-like):");
    println!("  ε_d = {:.2}, ε_g = {:.2}", eps_d, eps_g);
    println!("  n₊ (RCP) = {:.4}", n_plus);
    println!("  n₋ (LCP) = {:.4}", n_minus);

    // ── 9. Print Faraday rotation rate ───────────────────────────────────────
    let omega_telecom = 2.0 * PI * C0 / lambda;
    let faraday_rate = gyro.faraday_rotation_rate(omega_telecom);
    println!("  Faraday rotation rate at 1550 nm:");
    println!(
        "    θ_F = {:.2} rad/m  ({:.2} deg/mm)",
        faraday_rate,
        faraday_rate.to_degrees() * 1e-3
    );
    println!();

    // Verdet constant estimate for B = 0.1 T (typical for YIG)
    let b_field = 0.1_f64; // Tesla
    let verdet = gyro
        .verdet_constant(omega_telecom, b_field)
        .expect("Verdet constant computation should succeed");
    println!(
        "  Verdet constant V = {:.1} rad/(T·m)  (B = {:.2} T)",
        verdet, b_field
    );
    println!(
        "  (45° rotation length: {:.2} mm)",
        (PI / 4.0 / verdet / b_field) * 1e3
    );

    println!();
    println!("=== Example complete ===");
}
