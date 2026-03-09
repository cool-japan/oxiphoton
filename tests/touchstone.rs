//! Integration tests for Touchstone S-parameter I/O (Wave 5).
//!
//! Tests cover: 2-port and 4-port roundtrip, insertion loss dB, return loss dB,
//! group delay from linear phase, cascade identity matrices, S→T→S roundtrip,
//! and Touchstone header format.

use num_complex::Complex64;
use oxiphoton::io::touchstone::{
    cascade_two_port, s_to_t_matrix, t_to_s_matrix, TouchstoneFormat, TouchstoneReader,
    TouchstoneWriter,
};
use std::f64::consts::PI;

const TOL: f64 = 1e-6;

// ── helper: identity 2-port S-matrix (perfect through) ─────────────────────

fn identity_s2() -> Vec<Vec<Complex64>> {
    vec![
        vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ]
}

// ── test 1: write and parse 2-port S-parameters roundtrip ───────────────────

#[test]
fn test_touchstone_2port_roundtrip() {
    let mut writer = TouchstoneWriter::new(2, 50.0);

    let freqs = [1.0e9_f64, 2.0e9, 3.0e9, 5.0e9, 10.0e9];
    for &f in &freqs {
        let s11 = Complex64::new(0.05, 0.02);
        let s21 = Complex64::new(0.70, 0.10);
        let s12 = Complex64::new(0.70, -0.10);
        let s22 = Complex64::new(0.05, -0.02);
        let s = vec![vec![s11, s12], vec![s21, s22]];
        writer.add_frequency_point(f, s);
    }

    let content = writer.write_to_string();
    let reader =
        TouchstoneReader::parse(&content).expect("2-port Touchstone roundtrip parse must succeed");

    assert_eq!(reader.n_ports, 2, "n_ports must be 2");
    assert_eq!(
        reader.frequencies.len(),
        freqs.len(),
        "frequency count must match"
    );

    // Frequencies must round-trip accurately (relative error < 1e-6)
    for (i, (&f_w, &f_r)) in writer
        .frequencies
        .iter()
        .zip(reader.frequencies.iter())
        .enumerate()
    {
        let rel_err = (f_w - f_r).abs() / f_w;
        assert!(
            rel_err < 1e-5,
            "Frequency[{i}] roundtrip error: wrote {f_w:.6e} Hz, read {f_r:.6e} Hz, rel_err={rel_err:.2e}"
        );
    }

    // S-matrix values must roundtrip with tolerance from ASCII formatting
    for fi in 0..freqs.len() {
        for row in 0..2 {
            for col in 0..2 {
                let orig = writer.s_params[fi][row][col];
                let parsed = reader.s_params[fi][row][col];
                let diff = (orig - parsed).norm();
                assert!(
                    diff < 1e-4,
                    "S[{fi}][{row}][{col}] roundtrip: orig={orig:.6}, parsed={parsed:.6}, diff={diff:.2e}"
                );
            }
        }
    }
}

// ── test 2: write and parse 4-port roundtrip ────────────────────────────────

#[test]
fn test_touchstone_4port_roundtrip() {
    let mut writer = TouchstoneWriter::new(4, 50.0);

    // Build a simple 4-port S-matrix (directional coupler-like)
    let s = |i: usize, j: usize| -> Complex64 {
        if i == j {
            Complex64::new(0.01, 0.0) // small reflection
        } else if (i + j) == 3 {
            Complex64::new(0.7, 0.0) // through ports
        } else {
            Complex64::new(0.05, 0.0) // cross coupling
        }
    };

    let s_matrix: Vec<Vec<Complex64>> = (0..4).map(|i| (0..4).map(|j| s(i, j)).collect()).collect();

    let freqs = [1.0e9_f64, 10.0e9, 100.0e9];
    for &f in &freqs {
        writer.add_frequency_point(f, s_matrix.clone());
    }

    let content = writer.write_to_string();
    let reader =
        TouchstoneReader::parse(&content).expect("4-port Touchstone roundtrip parse must succeed");

    assert_eq!(reader.n_ports, 4, "n_ports must be 4");
    assert_eq!(
        reader.frequencies.len(),
        freqs.len(),
        "frequency count must match"
    );

    // Spot-check a few S-parameter values
    let s21_orig = writer.s_params[0][1][0];
    let s21_parsed = reader.s_params[0][1][0];
    let diff = (s21_orig - s21_parsed).norm();
    assert!(
        diff < 1e-4,
        "4-port S[0][1][0] roundtrip diff {diff:.2e} exceeds tolerance"
    );
}

// ── test 3: insertion loss dB calculation ───────────────────────────────────

/// |S21| = 0.5 → IL = −20 log10(0.5) ≈ 6.0206 dB
#[test]
fn test_insertion_loss_calculation() {
    let mut writer = TouchstoneWriter::new(2, 50.0);
    writer.add_frequency_point(
        1.0e9,
        vec![
            vec![Complex64::new(0.0, 0.0), Complex64::new(0.5, 0.0)],
            vec![Complex64::new(0.5, 0.0), Complex64::new(0.0, 0.0)],
        ],
    );

    let content = writer.write_to_string();
    let reader = TouchstoneReader::parse(&content).expect("insertion loss parse must succeed");

    // from_port=0, to_port=1 → S21
    let il = reader.insertion_loss_db(0, 1);
    assert_eq!(il.len(), 1, "one frequency point → one IL value");

    let expected_db = 20.0 * (1.0_f64 / 0.5).log10(); // ≈ 6.0206
    assert!(
        (il[0] - expected_db).abs() < 0.01,
        "IL of |S21|=0.5 should be {expected_db:.4} dB, got {:.4} dB",
        il[0]
    );
}

// ── test 4: return loss dB ───────────────────────────────────────────────────

/// |S11| = 0.1 → RL = −20 log10(0.1) = 20.0 dB
#[test]
fn test_return_loss_db() {
    let mut writer = TouchstoneWriter::new(2, 50.0);
    writer.add_frequency_point(
        1.0e9,
        vec![
            vec![Complex64::new(0.1, 0.0), Complex64::new(0.9, 0.0)],
            vec![Complex64::new(0.9, 0.0), Complex64::new(0.1, 0.0)],
        ],
    );

    let content = writer.write_to_string();
    let reader = TouchstoneReader::parse(&content).expect("return loss parse must succeed");

    let rl = reader.return_loss_db(0);
    assert_eq!(rl.len(), 1);

    let expected_db = 20.0_f64; // −20 log10(0.1) = 20 dB
    assert!(
        (rl[0] - expected_db).abs() < 0.01,
        "RL of |S11|=0.1 should be {expected_db:.1} dB, got {:.4} dB",
        rl[0]
    );
}

// ── test 5: group delay from linear phase ────────────────────────────────────

/// For S21(f) = exp(−2πi·f·τ) the group delay must be τ at every interior
/// frequency point (central-difference estimate).
#[test]
fn test_group_delay_linear_phase() {
    let tau = 10e-12_f64; // 10 ps
                          // Use frequencies where phase = 2π·f·τ stays below π to avoid wrap
                          // Maximum phase: 2π × 20e9 × 10e-12 = 0.4π  (well within ±π)
    let freqs: Vec<f64> = (1..=20).map(|i| i as f64 * 1.0e9).collect();

    // Build TouchstoneReader directly to avoid ASCII formatting precision loss
    let s_params: Vec<Vec<Vec<Complex64>>> = freqs
        .iter()
        .map(|&f| {
            let phase = -2.0 * PI * f * tau;
            let s21 = Complex64::new(phase.cos(), phase.sin());
            vec![
                vec![Complex64::new(0.0, 0.0), s21],
                vec![s21, Complex64::new(0.0, 0.0)],
            ]
        })
        .collect();

    let reader = TouchstoneReader {
        n_ports: 2,
        frequencies: freqs.clone(),
        s_params,
        reference_impedance: 50.0,
    };

    let gd = reader.group_delay(0, 1);
    assert_eq!(
        gd.len(),
        freqs.len(),
        "group delay vector must have same length as frequencies"
    );

    // Check interior points (central differences); skip edges
    for (i, &gd_val) in gd.iter().enumerate().take(gd.len() - 1).skip(1) {
        assert!(
            (gd_val - tau).abs() < tau * 0.01,
            "group_delay[{i}] = {:.4e} s, expected {tau:.4e} s (1% tol)",
            gd_val
        );
    }
}

// ── test 6: cascade identity matrices ────────────────────────────────────────

/// Cascading two identity (perfect-through) 2-port networks must produce
/// the identity S-matrix: S11=0, S21=1, S12=1, S22=0.
#[test]
fn test_cascade_identity() {
    let id = identity_s2();
    let result = cascade_two_port(&id, &id);

    assert_eq!(result.len(), 2, "cascaded result must be 2×2");
    assert_eq!(result[0].len(), 2);

    assert!(
        result[0][0].norm() < TOL,
        "S11 of cascaded identities must be 0: |S11| = {:.2e}",
        result[0][0].norm()
    );
    assert!(
        result[1][1].norm() < TOL,
        "S22 of cascaded identities must be 0: |S22| = {:.2e}",
        result[1][1].norm()
    );
    assert!(
        (result[1][0].norm() - 1.0).abs() < TOL,
        "S21 of cascaded identities must be 1: |S21| = {:.6}",
        result[1][0].norm()
    );
    assert!(
        (result[0][1].norm() - 1.0).abs() < TOL,
        "S12 of cascaded identities must be 1: |S12| = {:.6}",
        result[0][1].norm()
    );
}

// ── test 7: S to T to S roundtrip ────────────────────────────────────────────

/// S → T → S must recover the original S-matrix to within numerical precision.
#[test]
fn test_s_t_s_roundtrip() {
    let s_orig = vec![
        vec![Complex64::new(0.15, 0.05), Complex64::new(0.72, 0.08)],
        vec![Complex64::new(0.72, -0.08), Complex64::new(0.15, -0.05)],
    ];

    let t = s_to_t_matrix(&s_orig);
    let s_back = t_to_s_matrix(&t);

    for row in 0..2 {
        for col in 0..2 {
            let diff = (s_orig[row][col] - s_back[row][col]).norm();
            assert!(
                diff < 1e-10,
                "S→T→S roundtrip error at [{row}][{col}]: orig={}, recovered={}, diff={diff:.2e}",
                s_orig[row][col],
                s_back[row][col]
            );
        }
    }
}

// ── test 8: touchstone header format ─────────────────────────────────────────

/// write_to_string() must contain a comment line starting with '!' and
/// an options line starting with '# GHz'.
#[test]
fn test_touchstone_header_format() {
    let mut writer = TouchstoneWriter::with_format(2, 50.0, TouchstoneFormat::RI);
    writer.add_frequency_point(
        1.0e9,
        vec![
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ],
    );

    let content = writer.write_to_string();

    // Must contain at least one comment line (starts with '!')
    let has_comment = content.lines().any(|l| l.trim_start().starts_with('!'));
    assert!(
        has_comment,
        "Touchstone output must contain at least one '!' comment line"
    );

    // Must contain the standard options line '# GHz S RI R 50'
    let has_options = content.lines().any(|l| {
        let t = l.trim();
        t.starts_with('#') && t.contains("GHz") && t.contains("RI")
    });
    assert!(
        has_options,
        "Touchstone output must contain '#  GHz S RI ...' options line"
    );

    // Must contain the [Version] 2.0 tag
    assert!(
        content.contains("[Version] 2.0"),
        "Touchstone 2.0 output must contain '[Version] 2.0' header"
    );

    // Must contain [Network Data] keyword
    assert!(
        content.contains("[Network Data]"),
        "Touchstone 2.0 output must contain '[Network Data]' keyword"
    );
}
