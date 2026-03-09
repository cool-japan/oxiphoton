//! Bloch Boundary Condition and Band Structure Example.
//!
//! Photonic crystals are periodic dielectric structures that exhibit photonic
//! band gaps — frequency ranges where light cannot propagate. Computing the
//! band structure (dispersion relation ω vs k) requires sampling the Brillouin
//! zone using Bloch periodic boundary conditions.
//!
//! This example demonstrates:
//!   1. Bloch BC phase factors at high-symmetry points of the 2D square lattice
//!   2. Building the Γ–X–M–Γ k-path
//!   3. Running BandStructureCalc on a simple ε = 12 background with circular
//!      holes (r = 0.2a) to identify photonic band edges
//!   4. Wavelength sweeps with WavelengthSweep
//!   5. Convergence studies with ConvergenceSweep

use std::f64::consts::PI;

use oxiphoton::fdtd::{
    BandStructureCalc, BlochBc3d, ConvergenceSweep, ParamSweep, WavelengthSweep,
};

const C0: f64 = 2.99792458e8; // m/s

fn main() {
    println!("=== Bloch BC and Photonic Band Structure Example ===");
    println!();

    // ── 1. Bloch BC phase factors at high-symmetry k-points ─────────────────
    // For a square lattice with period a, the Brillouin zone high-symmetry points
    // (in 2D, projected to the xy-plane) are:
    //   Γ = (0, 0)              — zone centre
    //   X = (π/a, 0)            — zone boundary midpoint
    //   M = (π/a, π/a)          — zone corner
    let a = 500.0e-9_f64; // lattice constant 500 nm
    let nx = 20usize;
    let ny = 20usize;
    let nz = 1usize; // single layer (2D PhC)

    println!(
        "Square lattice: a = {:.0} nm, grid {}×{}×{}",
        a * 1e9,
        nx,
        ny,
        nz
    );
    println!();

    // Γ point — zero k-vector, phase factor = 1 everywhere
    let gamma_bc = BlochBc3d::square_lattice_gamma(a, nx, ny, nz);
    let pfx_gamma = gamma_bc.phase_factor_x();
    let pfy_gamma = gamma_bc.phase_factor_y();
    println!("High-symmetry k-points and Bloch phase factors exp(i·k·L):");
    println!(
        "  Γ = (0, 0):        pfx = {:.4}+{:.4}i,  pfy = {:.4}+{:.4}i",
        pfx_gamma.re, pfx_gamma.im, pfy_gamma.re, pfy_gamma.im
    );

    // X point — k = (π/a, 0) → phase exp(i·π) = -1 in x, exp(0) = 1 in y
    let x_bc = BlochBc3d::square_lattice_x(a, nx, ny, nz);
    let pfx_x = x_bc.phase_factor_x();
    let pfy_x = x_bc.phase_factor_y();
    println!(
        "  X = (π/a, 0):      pfx = {:.4}+{:.4}i,  pfy = {:.4}+{:.4}i",
        pfx_x.re, pfx_x.im, pfy_x.re, pfy_x.im
    );

    // M point — k = (π/a, π/a) → phase = -1 in both x and y
    let m_bc = BlochBc3d::square_lattice_m(a, nx, ny, nz);
    let pfx_m = m_bc.phase_factor_x();
    let pfy_m = m_bc.phase_factor_y();
    println!(
        "  M = (π/a, π/a):    pfx = {:.4}+{:.4}i,  pfy = {:.4}+{:.4}i",
        pfx_m.re, pfx_m.im, pfy_m.re, pfy_m.im
    );
    println!("  (At M-point both phase factors → −1, as exp(iπ) = −1 ✓)");
    println!();

    // ── 2. Build the Γ–X–M–Γ k-path for a 2D square lattice ─────────────────
    // The irreducible Brillouin zone (IBZ) of the square lattice has the path
    // Γ → X → M → Γ. Sampling this path gives the photonic band structure.
    let n_k = 15usize; // total k-points along the path
    let k_path = BandStructureCalc::square_lattice_path(n_k, a);

    println!("Γ–X–M–Γ k-path ({} k-points):", k_path.len());
    println!(
        "  {:>4}  {:>14}  {:>14}  Label",
        "idx", "kx (m⁻¹)", "ky (m⁻¹)"
    );
    println!("  {}", "-".repeat(50));
    let pi_over_a = PI / a;
    for (i, &[kx, ky, _kz]) in k_path.iter().enumerate() {
        // Label high-symmetry points
        let label = if kx.abs() < 1e-6 * pi_over_a && ky.abs() < 1e-6 * pi_over_a {
            "Γ"
        } else if (kx - pi_over_a).abs() < 1e-6 * pi_over_a && ky.abs() < 1e-6 * pi_over_a {
            "X"
        } else if (kx - pi_over_a).abs() < 1e-6 * pi_over_a
            && (ky - pi_over_a).abs() < 1e-6 * pi_over_a
        {
            "M"
        } else {
            ""
        };
        println!("  {:>4}  {:>14.4e}  {:>14.4e}  {}", i, kx, ky, label);
    }
    println!();

    // ── 3. BandStructureCalc: 2D PhC with ε = 12 background, r = 0.2a holes ─
    // This is a classic textbook model: dielectric background (ε = 12, like Si
    // at 1550 nm) with circular air holes (ε = 1) of radius r = 0.2a.
    // The first photonic band gap in TM polarisation opens around normalised
    // frequency fa/c ≈ 0.3.
    let bg_eps = 12.0_f64; // background permittivity (Si-like)
    let hole_r = 0.2 * a; // hole radius

    // Dielectric function: ε = 1 inside holes, ε = bg_eps outside
    let eps_fn = move |x: f64, y: f64, _z: f64| -> f64 {
        // Centre of unit cell at (a/2, a/2)
        let cx = a / 2.0;
        let cy = a / 2.0;
        let dist = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
        if dist < hole_r {
            1.0
        } else {
            bg_eps
        }
    };

    // Small calculation: few k-points, short time steps for quick demo
    let n_k_calc = 6usize;
    let k_path_short = BandStructureCalc::square_lattice_path(n_k_calc, a);

    // Frequency range: normalised frequency fa/c ∈ [0, 0.8]
    // → physical frequency f ∈ [0, 0.8·c/a]
    let f_max = 0.8 * C0 / a;
    let freq_range = (1e12_f64, f_max);

    let calc = BandStructureCalc::new(
        k_path_short.clone(),
        64, // n_freqs (DFT bins)
        freq_range,
        200, // n_timesteps (short run for demo)
    );

    println!(
        "BandStructureCalc: ε_bg = {:.0}, r/a = {:.2}, {} k-points",
        bg_eps,
        hole_r / a,
        k_path_short.len()
    );
    println!("Running FDTD band structure computation (short demo)...");

    let bands = calc.compute(a, eps_fn);
    let bands_norm = BandStructureCalc::normalized_frequencies(&bands, a);

    println!("Normalised resonant frequencies fa/c at each k-point:");
    println!(
        "  {:>4}  {:>8}  {:>8}  Resonances (fa/c)",
        "idx", "kx/π·a", "ky/π·a"
    );
    println!("  {}", "-".repeat(60));
    for (i, (&[kx, ky, _], norm_freqs)) in k_path_short.iter().zip(bands_norm.iter()).enumerate() {
        let kx_norm = kx * a / PI;
        let ky_norm = ky * a / PI;
        let freq_strs: Vec<String> = norm_freqs
            .iter()
            .take(4) // show at most 4 bands
            .map(|&f| format!("{:.3}", f))
            .collect();
        println!(
            "  {:>4}  {:>8.3}  {:>8.3}  [{}]",
            i,
            kx_norm,
            ky_norm,
            freq_strs.join(", ")
        );
    }
    println!("  (Short run: resonance detection accuracy improves with more time steps)");
    println!();

    // ── 4. WavelengthSweep: telecom C-band, 1000–1600 nm ────────────────────
    // This API converts between wavelengths (nm) and frequencies (Hz)
    // which is useful for connecting FDTD results to spectroscopic measurements.
    let wl_sweep = WavelengthSweep::new(1000.0, 1600.0, 7);
    let lambdas = wl_sweep.wavelengths_nm();
    let freqs_hz = wl_sweep.frequencies_hz();
    let omegas = wl_sweep.angular_frequencies();

    println!(
        "WavelengthSweep: 1000–1600 nm, {} points",
        wl_sweep.n_points
    );
    println!("  {:>12}  {:>16}  {:>18}", "λ (nm)", "f (THz)", "ω (rad/s)");
    println!("  {}", "-".repeat(52));
    for i in 0..lambdas.len() {
        println!(
            "  {:>12.1}  {:>16.4}  {:>18.6e}",
            lambdas[i],
            freqs_hz[i] * 1e-12,
            omegas[i]
        );
    }
    println!();

    // Demonstrate the run() API — compute effective index vs wavelength
    // (simplified: n_eff = constant 2.4, illustrating the sweep API)
    let n_eff_approx = 2.4_f64;
    let phase_lengths = wl_sweep.run(|lambda_m| {
        // Phase accumulated over 1 mm propagation: φ = 2π n_eff / λ × L
        let l = 1e-3_f64; // 1 mm
        2.0 * PI * n_eff_approx / lambda_m * l
    });
    println!(
        "Phase accumulation over 1 mm (n_eff = {:.1}):",
        n_eff_approx
    );
    for (lambda_m, phase) in &phase_lengths {
        println!(
            "  λ = {:.0} nm  →  φ = {:.2} rad  ({:.2}π)",
            lambda_m * 1e9,
            phase,
            phase / PI
        );
    }
    println!();

    // ── 5. ParamSweep: gap sweep for a coupling optimisation ─────────────────
    // Sweep the PhC hole radius from 0.1a to 0.4a and compute the filling
    // fraction of the unit cell (a measure of the average dielectric constant).
    let radius_sweep = ParamSweep::linspace("hole_radius_over_a", 0.1, 0.4, 7);
    let fill_fractions = radius_sweep.run(|r_over_a| -> f64 {
        // Filling fraction of circular hole in square unit cell
        PI * r_over_a * r_over_a
    });

    println!("ParamSweep: hole radius r/a from 0.1 to 0.4");
    println!(
        "  {:>10}  {:>16}  {:>14}",
        "r/a", "Fill fraction", "Avg ε (eff.)"
    );
    println!("  {}", "-".repeat(44));
    for (&r_over_a, &ff) in radius_sweep.values.iter().zip(fill_fractions.iter()) {
        let eps_eff = ff * 1.0 + (1.0 - ff) * bg_eps; // effective medium
        println!("  {:>10.3}  {:>16.4}  {:>14.3}", r_over_a, ff, eps_eff);
    }
    println!();

    // Minimise effective permittivity (find widest gap estimate)
    let (best_r, min_eps) = radius_sweep.minimize(|r_over_a| {
        // Effective eps — lower means stronger index contrast → wider gap
        let ff = PI * r_over_a * r_over_a;
        ff * 1.0 + (1.0 - ff) * bg_eps
    });
    println!(
        "  Best r/a = {:.3} → minimum ε_eff = {:.3}",
        best_r, min_eps
    );
    println!();

    // ── 6. ConvergenceSweep: spatial resolution convergence ──────────────────
    // Study how the Courant time step dt(dx) converges as the grid is refined.
    // The 3D Courant condition: dt = 0.5 · dx / (c√3)
    // We sweep dx from 20 nm to 1.25 nm (6 doublings), checking the Courant dt.
    let conv_sweep = ConvergenceSweep::new("cells_per_wavelength", 8.0, 8, 1e-3);

    let result = conv_sweep.run(|n_cells| {
        // Convergence metric: phase error of a wave over 10 wavelengths
        // φ_numerical = n_cells × arcsin(sin(π/n_cells)) ≈ π at large n
        let dx_over_lambda = 1.0 / n_cells;
        let k_numerical = (PI * dx_over_lambda).sin() / dx_over_lambda; // dispersion
        let k_exact = PI; // exact wavenumber (normalised)
        ((k_numerical - k_exact) / k_exact).abs()
    });

    match result {
        Ok((converged_n, error)) => {
            println!("ConvergenceSweep: phase error vs cells per wavelength");
            println!(
                "  Converged at {} cells/λ with phase error = {:.2e}",
                converged_n, error
            );
            println!("  (Error < 0.1% → sufficient for typical FDTD simulations)");
        }
        Err(e) => {
            println!(
                "ConvergenceSweep did not converge (expected for this demo): {}",
                e
            );
        }
    }
    println!();

    println!("=== Example complete ===");
}
