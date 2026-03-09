//! Metalens phase profile design example.
//!
//! Designs a metalens for focusing at λ = 532 nm (visible):
//!   - Diameter: 20 µm
//!   - Focal length: 50 µm
//!   - TiO₂ nanoposts on glass
//!
//! Shows the hyperbolic phase profile, nanopost library, and layout.

use oxiphoton::devices::metalens::layout::MetalensLayout;
use oxiphoton::devices::metalens::nanopost::NanopostLibrary;
use oxiphoton::devices::metalens::phase_profile::MetalensPhaseFocusing;
use std::f64::consts::PI;

fn main() {
    let wavelength = 532e-9; // 532 nm
    let diameter = 20e-6; // 20 µm aperture
    let focal_length = 50e-6; // 50 µm focal length

    println!("=== Metalens Phase Profile Design ===");
    println!("Wavelength:   {:.0} nm", wavelength * 1e9);
    println!("Diameter:     {:.1} µm", diameter * 1e6);
    println!("Focal length: {:.1} µm", focal_length * 1e6);
    let half: f64 = diameter / 2.0;
    let na = half / (half * half + focal_length * focal_length).sqrt();
    println!("NA:           {:.3}", na);
    println!();

    // Hyperbolic phase profile φ(r) = -k₀·(√(r²+f²) - f)
    let profile = MetalensPhaseFocusing::new(focal_length, wavelength, diameter / 2.0);
    println!("--- Phase profile at radial positions ---");
    println!("{:>10}  {:>15}  {:>12}", "r (µm)", "φ (rad)", "wraps (2π)");
    for r_um in [0.0f64, 2.0, 4.0, 6.0, 8.0, 10.0] {
        let r = r_um * 1e-6;
        let phi = profile.phase_continuous(r);
        let phi_wrapped = profile.phase_wrapped(r);
        println!(
            "{:>10.1}  {:>15.4}  {:>12.3}",
            r_um,
            phi,
            phi_wrapped / (2.0 * PI)
        );
    }
    println!();

    // TiO₂ nanopost library at 532 nm
    let library = NanopostLibrary::tio2_532nm();
    println!("Nanopost library: {} entries", library.diameters.len());
    println!(
        "Phase range: {:.2} rad ({:.1}×2π)",
        library.phase_range(),
        library.phase_range() / (2.0 * PI)
    );
    println!("Full 2π coverage: {}", library.has_full_phase_coverage());
    let (d_min, d_max) = (
        *library.diameters.first().unwrap(),
        *library.diameters.last().unwrap(),
    );
    println!("Diameter range: {:.0}–{:.0} nm", d_min * 1e9, d_max * 1e9);
    println!(
        "Average transmittance: {:.1}%",
        library.average_transmittance() * 100.0
    );
    println!();

    // Example phase-to-diameter lookup
    println!("--- Phase → Diameter mapping ---");
    println!("{:>12}  {:>15}", "Phase (rad)", "Diameter (nm)");
    for frac in [0.0f64, 0.25, 0.5, 0.75, 1.0] {
        let phase = frac * 2.0 * PI;
        let d = library.diameter_for_phase(phase).unwrap_or(0.0);
        println!("{:>12.4}  {:>15.1}", phase, d * 1e9);
    }
    println!();

    // Generate focusing metalens layout
    let pitch = 350e-9;
    let layout = MetalensLayout::focusing(diameter, focal_length, pitch, wavelength);
    println!("Metalens layout:");
    println!("  Posts:  {}", layout.n_posts());
    println!("  Pitch:  {:.0} nm", layout.pitch * 1e9);
    println!("  NA:     {:.3}", layout.numerical_aperture());
    println!("  Airy radius: {:.2} µm", layout.airy_radius() * 1e6);
    let (x_min, x_max, y_min, y_max) = layout.bounding_box();
    println!(
        "  Extent: [{:.1}, {:.1}] × [{:.1}, {:.1}] µm",
        x_min * 1e6,
        x_max * 1e6,
        y_min * 1e6,
        y_max * 1e6
    );

    println!("\nDone.");
}
