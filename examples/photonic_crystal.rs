//! Photonic crystal band structure calculation example.
//!
//! Computes the band structure of a 1D Si/SiO2 photonic crystal (Bragg stack)
//! using the Transfer Matrix dispersion relation, and finds the photonic band gap.
//!
//! Parameters:
//!   - n_Si = 3.48, n_SiO2 = 1.44
//!   - Quarter-wave stack at λ = 1550 nm

use oxiphoton::photonic_crystal::bandstructure::{quarter_wave_gap, PhCrystal1d};
use oxiphoton::photonic_crystal::defect::Pc1dCavity;
use std::f64::consts::PI;

fn main() {
    let n_si = 3.48_f64;
    let n_sio2 = 1.44_f64;
    let lambda_c = 1550e-9;
    let c = 2.998e8_f64;

    println!("=== 1D Photonic Crystal Band Structure ===");
    println!(
        "Si/SiO2 quarter-wave Bragg stack at λ = {:.0} nm",
        lambda_c * 1e9
    );
    println!("n_Si = {n_si:.2},  n_SiO2 = {n_sio2:.2}");

    // Quarter-wave stack: d_i = λ_c / (4·n_i)
    let pc = PhCrystal1d::quarter_wave(n_si, n_sio2, lambda_c);
    println!("Period Λ = {:.1} nm", pc.period * 1e9);
    println!("  Layer 1 (Si):   d₁ = {:.1} nm", pc.d1 * 1e9);
    println!("  Layer 2 (SiO2): d₂ = {:.1} nm", pc.d2 * 1e9);
    println!();

    let omega_c = 2.0 * PI * c / lambda_c;
    let omega_max = 3.0 * omega_c;

    // Print dispersion at key frequency fractions
    println!("--- Bloch dispersion at key frequencies ---");
    println!("{:>10}  {:>15}  {:>10}", "ω/ωc", "cos(k_B·Λ)", "status");
    for frac in [0.5f64, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0] {
        let omega = frac * omega_c;
        let cos_kb = pc.cos_kbloch(omega);
        let status = if cos_kb.abs() > 1.0 { "GAP" } else { "band" };
        println!("{:>10.3}  {:>15.4}  {:>10}", frac, cos_kb, status);
    }
    println!();

    // Find allowed bands by scanning
    let bands = pc.find_bands(omega_max, 2000);
    println!(
        "--- Allowed photonic bands (first {}) ---",
        bands.len().min(4)
    );
    for (i, (o_lo, o_hi)) in bands.iter().take(4).enumerate() {
        let lam_lo = 2.0 * PI * c / o_hi;
        let lam_hi = 2.0 * PI * c / o_lo;
        println!(
            "  Band {}: λ = {:.1}–{:.1} nm",
            i + 1,
            lam_lo * 1e9,
            lam_hi * 1e9
        );
    }
    println!();

    // Band gap (analytic quarter-wave result)
    let (gap_omega, gap_width) = quarter_wave_gap(n_si, n_sio2, pc.period);
    let gap_frac = gap_width / gap_omega;
    println!("--- Band Gap (analytic, quarter-wave) ---");
    println!("Gap center:   ω = {:.3}·ωc", gap_omega / omega_c);
    println!(
        "Gap width:    Δω/ω_c = {:.3} ({:.1}%)",
        gap_frac,
        gap_frac * 100.0
    );
    let lambda_gap = 2.0 * PI * c / gap_omega;
    println!("Gap center λ: {:.1} nm", lambda_gap * 1e9);
    println!();

    // DOS: density of states from numerical dispersion
    let dos_data = pc.density_of_states(omega_max, 300);
    let n_gap_pts = dos_data.iter().filter(|(_, d)| *d < 1e-30).count();
    println!(
        "DOS: {} zero-DOS points (gap region) out of {}",
        n_gap_pts,
        dos_data.len()
    );
    println!();

    // 1D PC cavity
    let cavity = Pc1dCavity::half_wave_cavity_1550nm();
    let lambda_res = cavity.resonance_wavelength();
    println!("--- 1D PC Fabry-Pérot Cavity (half-wave) ---");
    println!("Mirror R:        {:.3}", cavity.r_mirror);
    println!("Q factor:        {:.0}", cavity.quality_factor());
    println!("Finesse:         {:.1}", cavity.finesse());
    println!("Resonance λ:     {:.2} nm", lambda_res * 1e9);
    println!(
        "Decay length:    {:.1} nm",
        cavity.evanescent_decay_length() * 1e9
    );
    let t_peak = cavity.transmission(cavity.resonance_frequency());
    println!("Peak T:          {:.4}", t_peak);

    println!("\nDone.");
}
