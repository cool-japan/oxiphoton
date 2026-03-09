//! S-Parameter Extraction and Touchstone Export Example.
//!
//! S-parameters (scattering parameters) are the standard description of linear
//! N-port microwave / photonic networks. In photonics they capture both
//! amplitude and phase of transmission and reflection coefficients.
//!
//! This example demonstrates:
//!   1. Creating synthetic 2-port S-parameters for a Fabry-Perot resonator
//!   2. Writing the data to a Touchstone (.s2p) file in /tmp/
//!   3. Reading back and verifying the round-trip
//!   4. Computing insertion loss, return loss, group delay, and 3-dB bandwidth
//!   5. Demonstrating the EigenModeMonitor API
//!   6. Cascading two identical 3-dB splitters using the T-matrix product

use std::f64::consts::PI;

use num_complex::Complex64;

use oxiphoton::fdtd::monitor::eigenmode_decomp::{EigenModeMonitor, EigenModeProfile};
use oxiphoton::fdtd::monitor::flux::FluxNormal;
use oxiphoton::io::{cascade_two_port, TouchstoneReader, TouchstoneWriter};

// ── Fabry-Perot resonator S-parameters ──────────────────────────────────────
//
// A Fabry-Perot cavity of length L with mirror reflectivity r (field amplitude)
// has the transfer function:
//   t(f) = (1 - r²) / (1 - r² · exp(2iφ))   where φ = π f / FSR
//   s11(f) = r · (exp(2iφ) - 1) / (1 - r² · exp(2iφ))
//
// Resonances occur when φ = nπ (f = n·FSR), where the transmission peaks
// and reflection dips.
fn fabry_perot_s2port(freq_hz: f64, fsr_hz: f64, r_mirror: f64) -> Vec<Vec<Complex64>> {
    let phi = PI * freq_hz / fsr_hz;
    let exp2i_phi = Complex64::new((2.0 * phi).cos(), (2.0 * phi).sin());
    let r2 = r_mirror * r_mirror;
    let one = Complex64::new(1.0, 0.0);

    let denom = one - Complex64::new(r2, 0.0) * exp2i_phi;
    // Transmission S21 = S12
    let s21 = Complex64::new(1.0 - r2, 0.0) / denom;
    // Reflection S11 = S22
    let s11 = Complex64::new(r_mirror, 0.0) * (exp2i_phi - one) / denom;

    vec![vec![s11, s21], vec![s21, s11]]
}

fn main() {
    println!("=== S-Parameter Extraction and Touchstone Export Example ===");
    println!();

    // ── 1. Create synthetic 2-port S-parameters (Fabry-Perot resonator) ──────
    // Parameters for a telecom-band FP cavity:
    //   FSR (free spectral range) = 50 GHz (corresponds to L ≈ 2 mm in Si)
    //   Mirror reflectivity r = 0.5 (power reflectivity R = 25%)
    //   Frequency range: 185.0–195.0 THz (C-band 1535–1622 nm)
    let fsr_hz = 50.0e9_f64; // 50 GHz free spectral range
    let r_mirror = 0.5_f64; // field amplitude reflectivity
    let f_start = 185.0e12_f64; // 185 THz
    let f_end = 195.0e12_f64; // 195 THz
    let n_freq = 201usize;

    println!("Fabry-Perot resonator parameters:");
    println!("  FSR               = {:.1} GHz", fsr_hz * 1e-9);
    println!(
        "  Mirror reflectivity r = {:.2}  (R = {:.0}%)",
        r_mirror,
        r_mirror * r_mirror * 100.0
    );
    println!(
        "  Finesse F ≈ π√R / (1-R) ≈ {:.1}",
        PI * r_mirror / (1.0 - r_mirror * r_mirror)
    );
    println!(
        "  Frequency range:  {:.1}–{:.1} THz",
        f_start * 1e-12,
        f_end * 1e-12
    );
    println!("  Number of points: {}", n_freq);
    println!();

    let mut writer = TouchstoneWriter::new(2, 50.0);
    let df = (f_end - f_start) / (n_freq - 1) as f64;

    for i in 0..n_freq {
        let f = f_start + i as f64 * df;
        let s_mat = fabry_perot_s2port(f, fsr_hz, r_mirror);
        writer.add_frequency_point(f, s_mat);
    }

    // ── 2. Write to Touchstone file in /tmp/ ──────────────────────────────────
    let tmp_path = {
        let mut p = std::env::temp_dir();
        p.push("fabry_perot.s2p");
        p.to_string_lossy().into_owned()
    };

    writer
        .write_to_file(&tmp_path)
        .expect("Writing Touchstone file should succeed");
    println!("Touchstone file written to: {}", tmp_path);

    // ── 3. Read back and verify round-trip ────────────────────────────────────
    let file_content =
        std::fs::read_to_string(&tmp_path).expect("Reading back Touchstone file should succeed");
    let reader =
        TouchstoneReader::parse(&file_content).expect("Parsing Touchstone file should succeed");

    println!("Round-trip verification:");
    println!(
        "  Written frequencies: {}, Read frequencies: {}",
        writer.frequencies.len(),
        reader.frequencies.len()
    );

    // Verify a few S-parameter values
    let mut max_err = 0.0_f64;
    for i in 0..n_freq {
        let f_w = writer.frequencies[i];
        let f_r = reader.frequencies[i];
        let freq_err = (f_w - f_r).abs() / f_w.max(1.0);
        max_err = max_err.max(freq_err);

        let s21_w = writer.s_params[i][1][0];
        let s21_r = reader.s_params[i][1][0];
        let s_err = (s21_w - s21_r).norm();
        max_err = max_err.max(s_err);
    }
    println!("  Maximum round-trip error (freq + S21): {:.3e}", max_err);
    if max_err < 1e-4 {
        println!("  [OK] Round-trip accuracy verified (< 0.01%).");
    }
    println!();

    // ── 4. Compute insertion loss, return loss, group delay ───────────────────
    let il_db = reader.insertion_loss_db(0, 1); // S21: port 0→1
    let rl_db = reader.return_loss_db(0); // S11
    let gd = reader.group_delay(0, 1); // S21 group delay

    // Find resonances (local minima of insertion loss = local maxima of |S21|)
    let mut resonances: Vec<(f64, f64)> = Vec::new(); // (freq_THz, IL_dB)
    let s21_mag = reader.magnitude(0, 1);
    for i in 1..n_freq.saturating_sub(1) {
        if s21_mag[i] > s21_mag[i - 1] && s21_mag[i] > s21_mag[i + 1] {
            resonances.push((reader.frequencies[i] * 1e-12, il_db[i]));
        }
    }

    println!("Insertion loss S21 analysis:");
    println!(
        "  Min IL = {:.3} dB  (at resonance: near-zero loss)",
        il_db.iter().cloned().fold(f64::INFINITY, f64::min)
    );
    println!(
        "  Max IL = {:.3} dB  (at anti-resonance)",
        il_db.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    println!("  Return loss S11 analysis:");
    println!(
        "    Min RL = {:.3} dB  (at resonance: near-zero reflection)",
        rl_db.iter().cloned().fold(f64::INFINITY, f64::min)
    );
    println!(
        "    Max RL = {:.3} dB  (at anti-resonance)",
        rl_db.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    println!(
        "  First {} resonances detected (f, IL_dB):",
        resonances.len().min(5)
    );
    for &(f_thz, il) in resonances.iter().take(5) {
        println!("    f = {:.3} THz,  IL = {:.4} dB", f_thz, il);
    }

    // Group delay at first resonance
    if let Some(i_res) = s21_mag
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
    {
        println!(
            "  Group delay at peak transmission: {:.3} ps",
            gd[i_res] * 1e12
        );
    }
    println!();

    // ── 5. Find 3-dB bandwidth ─────────────────────────────────────────────────
    // For the S21 peak, the 3-dB bandwidth equals FSR / Finesse.
    // Expected: BW_3dB ≈ FSR × (1 - r²) / (π√r²) = FSR / Finesse
    let bw_3db = reader.bandwidth_3db(0, 1);
    let finesse = PI * r_mirror / (1.0 - r_mirror * r_mirror);
    let expected_bw = fsr_hz / finesse;

    match bw_3db {
        Some(bw) => {
            println!("3-dB bandwidth analysis:");
            println!("  Measured BW_3dB = {:.3} GHz", bw * 1e-9);
            println!(
                "  Expected  BW_3dB = FSR/F = {:.3} GHz  (FSR={:.0} GHz, F={:.1})",
                expected_bw * 1e-9,
                fsr_hz * 1e-9,
                finesse
            );
            let bw_err = (bw - expected_bw).abs() / expected_bw;
            println!(
                "  Agreement: {:.1}%  (limited by frequency sampling resolution)",
                (1.0 - bw_err) * 100.0
            );
        }
        None => {
            println!("3-dB bandwidth: insufficient data (increase n_freq for accurate result)");
        }
    }
    println!();

    // ── 6. EigenModeMonitor creation ──────────────────────────────────────────
    // Create a monitor at a waveguide cross-section (z = 20 in a 40×40×80 grid),
    // monitoring at C-band frequencies.
    let monitor_freqs = vec![191.0e12_f64, 192.0e12, 193.0e12]; // THz frequencies
    let dt = 1e-17_f64; // 10 as time step

    let mut emon = EigenModeMonitor::new(
        FluxNormal::Z, // monitor in xy-plane (normal = z)
        20,            // z-index (slice through the waveguide)
        (5, 35),       // i-range (transverse x)
        (5, 35),       // j-range (transverse y)
        monitor_freqs.clone(),
        dt,
    );

    // Register the fundamental TE mode as a Gaussian profile (approximate)
    // A real simulation would extract this from a mode solver; here we use
    // the built-in Gaussian helper for illustration.
    let ni = 30usize; // 35 - 5
    let nj = 30usize;
    let n_eff_te = 2.35_f64; // effective index of Si waveguide TE mode at 1550 nm
    let sigma_grid = 4.0_f64; // mode width in grid cells

    let mut te_mode = EigenModeProfile::gaussian(n_eff_te, ni, nj, sigma_grid, sigma_grid);
    te_mode
        .normalize()
        .expect("Mode normalization should succeed for non-zero Gaussian mode");
    te_mode.mode_number = 0;

    emon.add_mode(te_mode);

    println!("EigenModeMonitor setup:");
    println!(
        "  Monitor plane: z = 20, transverse range 5..35 × 5..35 ({} × {} cells)",
        ni, nj
    );
    println!(
        "  Frequencies: {} points ({:.0}–{:.0} THz)",
        monitor_freqs.len(),
        monitor_freqs.first().copied().unwrap_or(0.0) * 1e-12,
        monitor_freqs.last().copied().unwrap_or(0.0) * 1e-12
    );
    println!(
        "  Registered modes: {} (TE fundamental, n_eff = {:.3})",
        emon.modes.len(),
        n_eff_te
    );
    println!(
        "  Mode self-overlap (after normalise): {:.6}",
        emon.modes[0].self_overlap()
    );
    println!();

    // ── 7. Cascade two identical 3-dB splitters ───────────────────────────────
    // A 3-dB (50:50) optical splitter has S-parameters:
    //   S11 = 0  (matched input, no reflection)
    //   S21 = 1/√2 · exp(iπ/2) = i/√2  (half power, 90° phase shift)
    //   S12 = S21  (reciprocal)
    //   S22 = 0
    //
    // Two such splitters in cascade (port 2 of S1 → port 1 of S2) give a
    // 3-dB + 3-dB = 6-dB (25% power) through path with additional phase shift.
    let amp = (0.5_f64).sqrt(); // 1/√2 amplitude for each splitter
    let s_splitter: Vec<Vec<Complex64>> = vec![
        vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, amp)],
        vec![Complex64::new(0.0, amp), Complex64::new(0.0, 0.0)],
    ];

    let s_cascade = cascade_two_port(&s_splitter, &s_splitter);

    let s11_c = s_cascade[0][0];
    let s21_c = s_cascade[1][0];
    let s12_c = s_cascade[0][1];
    let s22_c = s_cascade[1][1];

    println!("Cascade of two identical 3-dB splitters:");
    println!("  Each splitter: S21 = i/√2  (|S21|² = 0.5 → 3 dB)");
    println!("  Cascaded result:");
    println!(
        "    S11 = {:.4}+{:.4}i  (|S11|² = {:.4})",
        s11_c.re,
        s11_c.im,
        s11_c.norm_sqr()
    );
    println!(
        "    S21 = {:.4}+{:.4}i  (|S21|² = {:.4} → {:.1} dB)",
        s21_c.re,
        s21_c.im,
        s21_c.norm_sqr(),
        -10.0 * s21_c.norm_sqr().log10()
    );
    println!(
        "    S12 = {:.4}+{:.4}i  (|S12|² = {:.4})",
        s12_c.re,
        s12_c.im,
        s12_c.norm_sqr()
    );
    println!(
        "    S22 = {:.4}+{:.4}i  (|S22|² = {:.4})",
        s22_c.re,
        s22_c.im,
        s22_c.norm_sqr()
    );
    println!(
        "  Phase of S21: {:.1}°  (expected: 180° = 2×90°)",
        s21_c.arg().to_degrees()
    );
    println!(
        "  Energy conservation: |S21|² + |S11|² = {:.4}  (should be ≤ 1)",
        s21_c.norm_sqr() + s11_c.norm_sqr()
    );
    println!();

    // ── 8. S-parameter summary ────────────────────────────────────────────────
    println!("── S-parameter analysis summary ─────────────────────────────────────");
    println!(
        "  FP resonator (r={:.2}, FSR={:.0} GHz):",
        r_mirror,
        fsr_hz * 1e-9
    );
    println!(
        "    Peak |S21|:         {:.3} (at resonance)",
        reader
            .magnitude(0, 1)
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max)
    );
    println!(
        "    Min |S11|:          {:.3} (at resonance)",
        reader
            .magnitude(0, 0)
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
    );
    println!(
        "    Peak group delay:   {:.3} ps",
        gd.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 1e12
    );
    println!("    Resonances found:   {}", resonances.len());
    if let Some(bw) = bw_3db {
        println!("    3-dB bandwidth:     {:.3} GHz", bw * 1e-9);
    }
    println!("  Touchstone file:  {}", tmp_path);
    println!();
    println!("=== Example complete ===");
}
