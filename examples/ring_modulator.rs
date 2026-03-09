//! Ring modulator performance analysis example.
//!
//! Computes transmission spectrum, modulation efficiency, and bandwidth
//! for a silicon ring modulator using plasma dispersion.
//!
//! The ring modulator uses free-carrier injection (PIN diode) or depletion
//! to shift the resonance wavelength, providing intensity modulation at
//! a fixed laser wavelength.
//!
//! Run with: cargo run --example ring_modulator --all-features

use oxiphoton::devices::modulator::plasma_dispersion::{PinDiodeModel, SiPlasmaDispersion};
use oxiphoton::devices::resonator::ring::RingResonator;

fn main() {
    println!("=== Silicon Ring Modulator Performance Analysis ===");
    println!();

    // ── Ring resonator parameters ────────────────────────────────────────────
    let radius = 10e-6; // 10 μm radius
    let n_eff = 2.4; // effective index at operating wavelength
    let n_group = 4.2; // group index
    let kappa_sq = 0.03; // power coupling coefficient (3%)
    let alpha_per_m = 50.0; // propagation loss 50 /m ≈ 2.2 dB/cm

    let ring = RingResonator::new(radius, n_eff, n_group, kappa_sq, alpha_per_m);

    let lambda0 = 1550e-9; // operating wavelength

    // ── Basic resonator properties ───────────────────────────────────────────
    let fsr = ring.fsr(lambda0);
    let q = ring.quality_factor(lambda0);
    let finesse = ring.finesse();
    let linewidth = lambda0 / q;
    let circ = ring.circumference();

    println!("=== Ring Resonator Properties ===");
    println!("Radius:         {:.1} μm", radius * 1e6);
    println!("Circumference:  {:.2} μm", circ * 1e6);
    println!("n_eff:          {:.4}", n_eff);
    println!("n_group:        {:.4}", n_group);
    println!("κ² (coupling):  {:.3}", kappa_sq);
    println!(
        "Loss (α):       {:.1} /m = {:.2} dB/cm",
        alpha_per_m,
        alpha_per_m * 0.01 * 4.343
    );
    println!();
    println!("FSR @ 1550nm:   {:.3} nm", fsr * 1e9);
    println!("Q factor:       {:.0}", q);
    println!("Finesse:        {:.1}", finesse);
    println!("Linewidth:      {:.4} nm", linewidth * 1e9);
    println!();

    // ── Resonance wavelengths ────────────────────────────────────────────────
    let resonances = ring.resonances(lambda0, 3);
    println!("Nearest resonances:");
    for (i, &wl) in resonances.iter().enumerate() {
        println!(
            "  Resonance {}: {:.4} nm (Δλ = {:+.4} nm from 1550nm)",
            i + 1,
            wl * 1e9,
            (wl - lambda0) * 1e9
        );
    }
    println!();

    // ── Transmission spectrum ────────────────────────────────────────────────
    // Sample a 2×FSR window around the nearest resonance
    let res0 = resonances[0];
    let span = fsr * 2.0;
    let n_pts = 201;
    let wavelengths: Vec<f64> = (0..n_pts)
        .map(|i| res0 - span / 2.0 + i as f64 / (n_pts - 1) as f64 * span)
        .collect();
    let t_through = ring.transmission_through(&wavelengths);

    let min_t = t_through.iter().cloned().fold(f64::INFINITY, f64::min);
    let min_wl = wavelengths[t_through.iter().position(|&v| v == min_t).unwrap_or(0)];
    let max_t = t_through.iter().cloned().fold(0.0_f64, f64::max);

    println!("=== Through-Port Transmission ===");
    println!("Resonance dip at:      {:.4} nm", min_wl * 1e9);
    println!(
        "Minimum transmission:  {:.4} ({:.2} dB)",
        min_t,
        10.0 * min_t.log10()
    );
    println!("Maximum transmission:  {:.4}", max_t);
    println!(
        "Extinction ratio:      {:.2} dB",
        10.0 * (max_t / min_t.max(1e-10)).log10()
    );
    println!();

    // Print spectrum (every 20th point)
    println!("--- Through-Port Spectrum (selected points) ---");
    println!("{:>10}  {:>12}  {:>10}", "λ (nm)", "T_through", "T (dB)");
    for i in (0..n_pts).step_by(20) {
        let t = t_through[i];
        let t_db = if t > 1e-30 { 10.0 * t.log10() } else { -100.0 };
        println!(
            "{:>10.3}  {:>12.6}  {:>10.3}",
            wavelengths[i] * 1e9,
            t,
            t_db
        );
    }
    println!();

    // ── Plasma dispersion modulation ─────────────────────────────────────────
    println!("=== Plasma Dispersion Modulation (Soref-Bennett) ===");

    // Modulator parameters
    let confinement = 0.85; // mode confinement in Si
    let mod_length = circ; // effective length = circumference

    // Sweep carrier concentration from 0 to 10^18 cm^-3
    let carrier_densities = [1e15, 1e16, 5e16, 1e17, 3e17, 1e18];

    println!(
        "{:>14}  {:>12}  {:>14}  {:>12}  {:>14}",
        "ΔN (cm⁻³)", "Δn_eff", "Δφ (rad)", "FCA (dB)", "ΔΛ_res (pm)"
    );
    println!("{}", "-".repeat(70));

    for &carrier in &carrier_densities {
        let modulator = SiPlasmaDispersion::new(lambda0, confinement, mod_length)
            .with_carriers(carrier, carrier);

        let dn_eff = modulator.delta_n_eff();
        let dphi = modulator.phase_shift_rad();
        let fca_db = modulator.fca_loss_db();

        // Resonance wavelength shift: Δλ_res = (λ/n_g) · Δn_eff
        let delta_lambda_res = lambda0 / n_group * dn_eff;

        println!(
            "{:>14.2e}  {:>+12.6}  {:>+14.6}  {:>12.4}  {:>+14.3}",
            carrier,
            dn_eff,
            dphi,
            fca_db,
            delta_lambda_res * 1e12
        );
    }
    println!();

    // ── PIN diode bandwidth analysis ─────────────────────────────────────────
    println!("=== PIN Diode Bandwidth Analysis ===");

    let pin = PinDiodeModel::silicon();
    let built_in_v = pin.built_in_voltage();

    println!("Built-in voltage:  {:.3} V", built_in_v);
    println!();

    let rs_ohm = 50.0; // 50 Ω series resistance
    let biases = [-2.0, -1.0, 0.0, 0.5];

    println!(
        "{:>10}  {:>18}  {:>14}  {:>14}",
        "V_bias (V)", "W_dep (nm)", "C_j (fF/μm)", "BW_RC (GHz)"
    );
    println!("{}", "-".repeat(60));

    for &v in &biases {
        let w_dep_nm = pin.depletion_width(v) * 1e9;
        let c_per_um = pin.junction_capacitance_ff_per_um(v);
        let bw_ghz = pin.rc_bandwidth_ghz(v, rs_ohm);
        println!(
            "{:>10.1}  {:>18.2}  {:>14.3}  {:>14.3}",
            v, w_dep_nm, c_per_um, bw_ghz
        );
    }
    println!();

    // ── Carrier lifetime bandwidth ───────────────────────────────────────────
    println!("=== Carrier Lifetime Bandwidth ===");

    let lifetimes_ns = [0.1, 0.3, 1.0, 3.0, 10.0];
    println!("{:>16}  {:>16}", "τ_c (ns)", "f_3dB (GHz)");
    println!("{}", "-".repeat(36));
    for &tau_ns in &lifetimes_ns {
        let tau_s = tau_ns * 1e-9;
        let bw_ghz = SiPlasmaDispersion::bandwidth_3db(tau_s) / 1e9;
        println!("{:>16.2}  {:>16.3}", tau_ns, bw_ghz);
    }
    println!();

    // ── Modulation efficiency table ──────────────────────────────────────────
    println!("=== Modulation Efficiency V_π·L ===");

    // For a phase-shift of π, we need |Δφ| = π
    // V_pi·L = V_pi × L where V_pi is the voltage for π phase shift
    // For carrier injection: Δφ = (2π/λ) × Γ × Δn(ΔN) × L
    // At a typical operating point ΔN = 1e17 cm^-3:
    let dn_at_1e17 = {
        let m = SiPlasmaDispersion::new(lambda0, confinement, 1.0).with_carriers(1e17, 1e17);
        m.delta_n()
    };

    // Length needed for π phase shift at this ΔN:
    // L_pi = λ / (2 × Γ × |Δn|)
    let l_pi_m = lambda0 / (2.0 * confinement * dn_at_1e17.abs());
    let vpi_l_typical = 1.0 * l_pi_m; // assuming ~1V drive

    println!("At ΔN = 1×10¹⁷ cm⁻³:");
    println!("  |Δn_eff|:    {:.4e}", dn_at_1e17.abs() * confinement);
    println!("  L_π:         {:.3} mm (for π phase shift)", l_pi_m * 1e3);
    println!(
        "  V_π·L (est): {:.4} V·cm (assuming 1V drive)",
        vpi_l_typical * 100.0
    );
    println!();

    // ── Energy per bit (EO modulator) ────────────────────────────────────────
    println!("=== Energy per Bit Analysis ===");
    let v_pp = 2.0; // 2V peak-to-peak
    let cap_ff_per_um = pin.junction_capacitance_ff_per_um(0.0);
    let mod_length_um = circ * 1e6;
    let c_total_ff = cap_ff_per_um * mod_length_um;

    let e_per_bit_fj = SiPlasmaDispersion::energy_per_bit_fj(c_total_ff, v_pp);
    println!("Modulator length:     {:.2} μm", mod_length_um);
    println!(
        "Junction capacitance: {:.3} fF/μm × {:.2} μm = {:.2} fF",
        cap_ff_per_um, mod_length_um, c_total_ff
    );
    println!("Drive voltage (V_pp): {:.1} V", v_pp);
    println!("Energy per bit:       {:.3} fJ/bit", e_per_bit_fj);
    println!();

    println!("=== Summary ===");
    println!("OxiPhoton ring modulator analysis complete.");
    println!(
        "FSR = {:.3} nm, Q = {:.0}, Finesse = {:.1}",
        fsr * 1e9,
        q,
        finesse
    );
    println!(
        "Carrier-induced Δλ_res @ 10¹⁷ cm⁻³ = {:.3} pm",
        (lambda0 / n_group
            * (SiPlasmaDispersion::new(lambda0, confinement, circ)
                .with_carriers(1e17, 1e17)
                .delta_n_eff()))
            * 1e12
    );
}
