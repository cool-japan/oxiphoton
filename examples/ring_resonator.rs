//! Ring resonator spectrum calculation example.
//!
//! Computes transmission spectrum of a Silicon ring resonator:
//!   - Radius: 5 µm
//!   - n_eff = 2.4,  n_group = 4.2
//!   - κ² = 0.05 (power coupling coefficient)
//!   - Wavelength range: 1540–1560 nm
//!
//! Demonstrates FSR, Q factor, and through-port spectrum.

use oxiphoton::devices::resonator::ring::RingResonator;

fn main() {
    let ring = RingResonator::new(
        5e-6,  // radius = 5 µm
        2.4,   // n_eff
        4.2,   // n_group
        0.05,  // kappa^2 = 5% power coupling
        200.0, // propagation loss 200 /m (≈ 1 dB/cm)
    );

    println!("=== Ring Resonator Transmission Spectrum ===");
    println!("Radius:       {:.1} µm", ring.radius * 1e6);
    println!("n_eff = {:.3},  n_group = {:.3}", ring.n_eff, ring.n_group);
    println!("κ² = {:.3}  (power coupling coefficient)", ring.kappa_sq);
    println!("Loss: {:.1} /m", ring.alpha);
    println!();

    let wl0 = 1550e-9;
    let fsr = ring.fsr(wl0);
    let q = ring.quality_factor(wl0);
    let resonances = ring.resonances(wl0, 3);

    println!("FSR at 1550 nm:    {:.3} nm", fsr * 1e9);
    println!("Q factor at 1550:  {:.0}", q);
    println!("Resonances near 1550 nm:");
    for wl in &resonances {
        println!("  {:.3} nm", wl * 1e9);
    }
    println!();

    // Through-port transmission spectrum sweep
    let wavelengths: Vec<f64> = (0..=80)
        .map(|i| (1540.0 + i as f64 * 0.25) * 1e-9)
        .collect();
    let t_through = ring.transmission_through(&wavelengths);

    println!("--- Through-port spectrum (1540–1560 nm, Δλ = 0.25 nm) ---");
    println!("{:>10}  {:>12}", "λ (nm)", "T_through");
    // Print every 8th point for brevity
    for (i, (wl, t)) in wavelengths.iter().zip(&t_through).enumerate() {
        if i % 8 == 0 {
            println!("{:>10.2}  {:>12.4}", wl * 1e9, t);
        }
    }

    // Find minimum transmission (resonance dip)
    let (min_idx, &t_min) = t_through
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    println!(
        "\nMinimum T = {t_min:.4} at λ = {:.3} nm",
        wavelengths[min_idx] * 1e9
    );
    let er_db = -10.0 * t_min.log10();
    println!("Extinction ratio: {er_db:.1} dB");

    println!("\nDone.");
}
