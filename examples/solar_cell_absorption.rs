//! Solar cell absorption spectrum example.
//!
//! Computes absorption in a crystalline Si solar cell:
//!   - 300 µm thick Si absorber
//!   - AM1.5G solar spectrum
//!   - SiNx anti-reflection coating (n=2.0, 75 nm)
//!   - Estimates Jsc improvement with ARC

use oxiphoton::solar::absorption::AbsorptionMaterial;
use oxiphoton::solar::antireflection::SingleLayerArc;
use oxiphoton::solar::spectrum::SolarSpectrum;

fn main() {
    let thickness = 300e-6; // 300 µm Si wafer

    println!("=== Silicon Solar Cell Absorption ===");
    println!("Absorber: c-Si, thickness = {:.0} µm", thickness * 1e6);
    println!();

    // Materials and spectrum
    let si = AbsorptionMaterial::crystalline_silicon();
    let arc = SingleLayerArc::sinx_on_silicon();
    let spectrum = SolarSpectrum::am15g();

    println!("Solar spectrum (AM1.5G):");
    let total_irr = spectrum.integrate(300e-9, 1200e-9, 1000);
    println!("  Total irradiance (300–1200 nm): {:.1} W/m²", total_irr);
    println!();

    // Absorption coefficient and single-pass absorptance
    println!("--- Absorption Properties ---");
    println!(
        "{:>10}  {:>12}  {:>12}  {:>12}  {:>10}",
        "λ (nm)", "α (cm⁻¹)", "α (m⁻¹)", "A_bare (%)", "R_ARC (%)"
    );
    let wavelengths_nm = [400.0f64, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0];
    for &lam_nm in &wavelengths_nm {
        let alpha_m = si.alpha_at_nm(lam_nm);
        let alpha_cm = alpha_m / 100.0;
        let absorptance = si.absorptance(lam_nm, thickness);
        let r_arc = arc.reflectance(lam_nm * 1e-9);
        println!(
            "{:>10.0}  {:>12.2e}  {:>12.2e}  {:>12.1}  {:>10.3}",
            lam_nm,
            alpha_cm,
            alpha_m,
            absorptance * 100.0,
            r_arc * 100.0
        );
    }
    println!();

    // Jsc estimate (bare Si)
    let jsc_bare = si.jsc_am15g(thickness, 0.35); // 35% reflection without ARC
    println!("--- Short-Circuit Current Density ---");
    println!("Jsc (bare, R=35%):        {:.2} mA/cm²", jsc_bare * 0.1);

    // Jsc with ARC (weighted reflectance ~5%)
    let jsc_arc = si.jsc_am15g(thickness, 0.05);
    println!("Jsc (with SiNx ARC, R≈5%): {:.2} mA/cm²", jsc_arc * 0.1);
    println!(
        "ARC improvement:           +{:.1}%",
        (jsc_arc / jsc_bare - 1.0) * 100.0
    );
    println!();

    // ARC properties
    println!("--- Anti-Reflection Coating (SiNx) ---");
    println!(
        "n_ARC = {:.1},  thickness = {:.0} nm",
        arc.n_arc,
        arc.thickness * 1e9
    );
    println!("Optimal index (√n_inc·n_sub): {:.3}", arc.optimal_index());
    println!(
        "Optimal thickness at 600nm:   {:.0} nm",
        arc.optimal_thickness(600e-9) * 1e9
    );
    println!();

    // Bandgap wavelength
    println!("--- Silicon Properties ---");
    println!("Bandgap Eg = {:.2} eV", si.bandgap_ev);
    println!("Bandgap λg = {:.0} nm", si.bandgap_wavelength_nm());
    println!(
        "Absorption depth at 1000nm: {:.1} µm",
        si.absorption_depth_m(1000.0) * 1e6
    );
    println!(
        "Absorption depth at 600nm:  {:.2} µm",
        si.absorption_depth_m(600.0) * 1e6
    );

    println!("\nDone.");
}
