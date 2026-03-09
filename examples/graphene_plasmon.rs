//! Graphene Plasmon Example — surface plasmon propagation on gated graphene.
//!
//! Graphene plasmons are extraordinary electromagnetic modes that are confined
//! to a two-dimensional sheet of carbon atoms. Compared to plasmons in noble
//! metals, graphene plasmons offer:
//!   * Extreme spatial confinement: λ_sp ≪ λ₀ (confinement factor up to 300×)
//!   * Gate-tuneable Fermi energy → frequency tuning without re-fabrication
//!   * Long lifetime in high-quality graphene (Drude scattering time τ ~ 1 ps)
//!
//! This example computes the graphene surface conductivity (Kubo formula),
//! the plasmon wavevector, and compares intraband vs interband contributions
//! across the THz range. It also illustrates key properties of hBN and MoS₂.

use std::f64::consts::PI;

use oxiphoton::material::{BPDirection, BlackPhosphorus, GrapheneSheet, HexagonalBN, MoS2};

const C0: f64 = 2.99792458e8; // m/s

fn main() {
    println!("=== Graphene Plasmon Propagation Example ===");
    println!();

    // ── 1. Create GrapheneSheet (E_F = 0.5 eV, τ = 1 ps, T = 300 K) ────────
    // E_F = 0.5 eV corresponds to carrier density n ≈ 1.9 × 10¹³ cm⁻²,
    // achievable with a back-gate voltage of ~40 V on 300 nm SiO₂.
    let fermi_ev = 0.5_f64; // Fermi energy (eV)
    let tau_ps = 1.0_f64; // scattering time (ps)
    let temp_k = 300.0_f64; // temperature (K)

    let graphene = GrapheneSheet::new(fermi_ev, tau_ps, temp_k);

    println!("Graphene parameters:");
    println!("  Fermi energy E_F  = {:.2} eV", fermi_ev);
    println!("  Scattering time τ = {:.1} ps", tau_ps);
    println!("  Temperature T     = {:.0} K", temp_k);
    println!(
        "  DC conductivity   = {:.3e} S/sq",
        graphene.dc_conductivity()
    );
    println!(
        "  Universal absorption = {:.4} ({:.2}%)",
        GrapheneSheet::universal_absorption(),
        GrapheneSheet::universal_absorption() * 100.0
    );
    println!();

    // ── 2. Compute conductivity at THz frequencies (1–10 THz) ───────────────
    println!("Surface conductivity σ(ω) at THz frequencies:");
    println!(
        "  {:>8}  {:>18}  {:>18}  {:>14}",
        "f (THz)", "σ_total (S/sq)", "σ_intra (S/sq)", "σ_inter (S/sq)"
    );
    println!("  {}", "-".repeat(66));

    let thz_freqs: Vec<f64> = (1..=10).map(|i| i as f64 * 1e12).collect();

    for &f in &thz_freqs {
        let omega = 2.0 * PI * f;
        let sigma_tot = graphene.surface_conductivity(omega);
        let sigma_intra = graphene.intraband(omega);
        let sigma_inter = graphene.interband(omega);

        // At THz frequencies, intraband (Drude) dominates for E_F > ℏω/2
        // Intraband: large imaginary part → plasmonic character
        // Interband: nearly zero below E_F threshold ℏω < 2E_F ≈ 1 eV
        println!(
            "  {:>8.1}  {:>8.3e}+{:.3e}i  {:>8.3e}+{:.3e}i  {:>8.3e}+{:.3e}i",
            f * 1e-12,
            sigma_tot.re,
            sigma_tot.im,
            sigma_intra.re,
            sigma_intra.im,
            sigma_inter.re,
            sigma_inter.im
        );
    }
    println!();

    // ── 3. Compute plasmon wavevector — show |k_sp| >> k₀ ───────────────────
    // Dispersion relation for TM graphene plasmon on substrate (ε_r above/below = 1):
    //   k_sp = i ω ε₀ (ε_above + ε_below) / (2 σ)
    // The ratio |k_sp| / k₀ is the confinement factor — values > 1 indicate
    // sub-wavelength confinement (typical: 10–300 at THz / mid-IR).
    println!("Plasmon confinement factor |k_sp| / k₀ (air / air substrate):");
    println!(
        "  {:>8}  {:>14}  {:>14}  {:>14}",
        "f (THz)", "|k_sp| (m⁻¹)", "k₀ (m⁻¹)", "Confinement"
    );
    println!("  {}", "-".repeat(58));

    for &f in &thz_freqs {
        let omega = 2.0 * PI * f;
        let k0 = omega / C0;
        let k_sp = graphene.plasmon_wavevector(omega, 1.0, 1.0);
        let confinement = k_sp.norm() / k0;
        let lambda_sp = 2.0 * PI / k_sp.re; // plasmon wavelength (m)

        println!(
            "  {:>8.1}  {:>14.4e}  {:>14.4e}  {:>10.1}×  (λ_sp={:.2} nm)",
            f * 1e-12,
            k_sp.norm(),
            k0,
            confinement,
            lambda_sp.abs() * 1e9
        );
    }
    println!();

    // ── 4. Compare intraband vs interband at mid-IR (30 THz) ─────────────────
    // At 30 THz (λ₀ ≈ 10 μm), graphene with E_F = 0.5 eV is well in the
    // intraband-dominated regime (ℏ·30THz ≈ 0.12 eV ≪ 2E_F = 1 eV).
    let f_midir = 30e12_f64;
    let omega_midir = 2.0 * PI * f_midir;
    let s_intra = graphene.intraband(omega_midir);
    let s_inter = graphene.interband(omega_midir);
    let s_total = graphene.surface_conductivity(omega_midir);
    println!("Intraband vs interband at {:.0} THz:", f_midir * 1e-12);
    println!(
        "  σ_intra = {:.4e} + {:.4e}i  (magnitude: {:.4e})",
        s_intra.re,
        s_intra.im,
        s_intra.norm()
    );
    println!(
        "  σ_inter = {:.4e} + {:.4e}i  (magnitude: {:.4e})",
        s_inter.re,
        s_inter.im,
        s_inter.norm()
    );
    println!("  σ_total = {:.4e} + {:.4e}i", s_total.re, s_total.im);
    println!(
        "  Intraband fraction: {:.1}%",
        100.0 * s_intra.norm() / s_total.norm()
    );
    println!();

    // ── 5. Plasmon group velocity ─────────────────────────────────────────────
    println!("Graphene plasmon group velocity at selected frequencies:");
    for &f_thz in &[1.0_f64, 3.0, 5.0, 10.0] {
        let omega = 2.0 * PI * f_thz * 1e12;
        let vg = graphene.plasmon_group_velocity(omega);
        println!(
            "  f = {:5.1} THz  →  v_g = {:.3e} m/s  ({:.3}×c)",
            f_thz,
            vg,
            vg / C0
        );
    }
    println!();

    // ── 6. Hexagonal boron nitride — hyperbolic phonon polaritons ─────────────
    // hBN is a natural hyperbolic material with two Reststrahlen bands where
    // ε_∥ and ε_⊥ take opposite signs, enabling hyperbolic dispersion.
    let hbn = HexagonalBN::new(10); // 10-layer slab
    let ranges = hbn.hyperbolic_frequency_range();

    println!("Hexagonal BN (hBN) — hyperbolic phonon polariton medium:");
    println!("  Hyperbolic frequency ranges:");
    for (i, &(omega_lo, omega_hi)) in ranges.iter().enumerate() {
        let f_lo = omega_lo / (2.0 * PI) * 1e-12;
        let f_hi = omega_hi / (2.0 * PI) * 1e-12;
        let band_type = if i == 0 {
            "Type I  (ε_⊥ < 0, ε_∥ > 0)"
        } else {
            "Type II (ε_∥ < 0, ε_⊥ > 0)"
        };
        println!(
            "    Band {}: {:.2}–{:.2} THz  [{}]",
            i + 1,
            f_lo,
            f_hi,
            band_type
        );
    }

    // Check whether mid-upper-band frequency is flagged hyperbolic
    let omega_test = ranges[1].0 + 0.5 * (ranges[1].1 - ranges[1].0);
    let eps_par = hbn.permittivity_in_plane(omega_test);
    let eps_perp = hbn.permittivity_out_of_plane(omega_test);
    let f_test = omega_test / (2.0 * PI) * 1e-12;
    println!("  At f = {:.2} THz (mid upper band):", f_test);
    println!(
        "    ε_∥ = {:.3}+{:.3}i  (negative real → hyperbolic ✓)",
        eps_par.re, eps_par.im
    );
    println!(
        "    ε_⊥ = {:.3}+{:.3}i  (positive real)",
        eps_perp.re, eps_perp.im
    );
    println!("    is_hyperbolic: {}", hbn.is_hyperbolic(omega_test));
    println!();

    // ── 7. MoS₂ monolayer — direct bandgap and exciton energies ──────────────
    let mos2_mono = MoS2::new(1, 0.0); // monolayer, no strain
    let mos2_bilayer = MoS2::new(2, 0.0);
    let mos2_bulk = MoS2::new(5, 0.0);

    println!("MoS₂ layer-dependent properties:");
    println!(
        "  {:>10}  {:>14}  {:>14}  {:>14}  {:>12}",
        "# layers", "Eg (eV)", "E_A (eV)", "E_B (eV)", "Direct gap?"
    );
    println!("  {}", "-".repeat(70));
    for (label, mos2) in [
        ("Monolayer", &mos2_mono),
        ("Bilayer", &mos2_bilayer),
        ("5-layer", &mos2_bulk),
    ] {
        println!(
            "  {:>10}  {:>14.3}  {:>14.3}  {:>14.3}  {:>12}",
            label,
            mos2.bandgap_ev(),
            mos2.a_exciton_energy_ev(),
            mos2.b_exciton_energy_ev(),
            mos2.is_direct_bandgap()
        );
    }
    println!();

    // Valley polarisation under σ+ excitation
    let p_valley = mos2_mono.valley_polarization(1.0);
    println!(
        "  Monolayer valley polarisation (σ+ excitation): {:.1}%",
        p_valley * 100.0
    );
    println!("  (Due to K/K' valley degeneracy lifting under circularly polarised light)");
    println!();

    // ── 8. Summary table of 2D material properties ───────────────────────────
    println!("── Summary: 2D materials comparison ───────────────────────────────────");
    println!(
        "  {:>18}  {:>14}  {:>18}  {:>20}",
        "Material", "Eg (eV)", "Key feature", "Spectral range"
    );
    println!("  {}", "-".repeat(76));

    let bp_ac = BlackPhosphorus::new(1, BPDirection::Armchair);
    let rows: &[(&str, f64, &str, &str)] = &[
        ("Graphene", 0.0, "Plasmonic (THz/MIR)", "THz – mid-IR"),
        ("hBN", 0.0, "Hyperbolic phonon", "Mid-IR"),
        (
            "MoS2 (1L)",
            mos2_mono.bandgap_ev(),
            "Direct gap, valley",
            "Visible (~660 nm)",
        ),
        (
            "BP (1L, AC)",
            bp_ac.bandgap_ev(),
            "Anisotropic, tunable",
            "NIR (~620 nm)",
        ),
    ];
    for &(material, eg, feature, spectral) in rows {
        if eg > 0.0 {
            println!(
                "  {:>18}  {:>14.3}  {:>18}  {:>20}",
                material, eg, feature, spectral
            );
        } else {
            println!(
                "  {:>18}  {:>14}  {:>18}  {:>20}",
                material, "gapless", feature, spectral
            );
        }
    }
    println!();
    println!("=== Example complete ===");
}
