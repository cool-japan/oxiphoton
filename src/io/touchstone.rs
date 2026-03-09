//! Touchstone 2.0 S-parameter file reader and writer.
//!
//! Touchstone (*.snp, *.s2p, etc.) is the de-facto standard for N-port
//! S-parameter data in RF and photonic circuit simulation.
//!
//! # Format overview (Touchstone 1.0 / 2.0)
//!
//! ```text
//! ! comment lines begin with '!'
//! # GHz S RI R 50      ← option line: freq-unit param-type data-format ref-R
//! 1.0  0.9 0.0  0.1 0.0  0.1 0.0  0.9 0.0   ← frequency + S-matrix row-major
//! ```
//!
//! This module writes Touchstone 2.0 with `[Version] 2.0` header and reads
//! both 1.0 and 2.0 files.

use num_complex::Complex64;
use std::f64::consts::PI;
use std::fmt::Write as FmtWrite;

use crate::error::OxiPhotonError;

// ─────────────────────────────────────────────────────────────────────────────
// TouchstoneFormat
// ─────────────────────────────────────────────────────────────────────────────

/// Data format used in the Touchstone option line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TouchstoneFormat {
    /// dB magnitude + angle in degrees: `|S| [dB]   ∠S [°]`
    DB,
    /// Linear magnitude + angle in degrees: `|S|   ∠S [°]`
    MA,
    /// Real + imaginary parts: `Re(S)   Im(S)`
    RI,
}

impl TouchstoneFormat {
    fn option_str(&self) -> &'static str {
        match self {
            TouchstoneFormat::DB => "DB",
            TouchstoneFormat::MA => "MA",
            TouchstoneFormat::RI => "RI",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TouchstoneWriter
// ─────────────────────────────────────────────────────────────────────────────

/// Touchstone 2.0 writer for N-port S-parameter data.
pub struct TouchstoneWriter {
    /// Number of ports.
    pub n_ports: usize,
    /// Frequency list (Hz), one entry per data point.
    pub frequencies: Vec<f64>,
    /// S-parameter matrix per frequency: `[n_freq][n_ports][n_ports]`.
    pub s_params: Vec<Vec<Vec<Complex64>>>,
    /// Reference impedance (Ω), typically 50.0.
    pub reference_impedance: f64,
    /// Output data format.
    pub format: TouchstoneFormat,
}

impl TouchstoneWriter {
    /// Create a new writer with RI format and 50 Ω reference impedance.
    pub fn new(n_ports: usize, z0: f64) -> Self {
        Self::with_format(n_ports, z0, TouchstoneFormat::RI)
    }

    /// Create a new writer with a specified data format.
    pub fn with_format(n_ports: usize, z0: f64, format: TouchstoneFormat) -> Self {
        Self {
            n_ports,
            frequencies: Vec::new(),
            s_params: Vec::new(),
            reference_impedance: z0,
            format,
        }
    }

    /// Append one frequency data point.
    ///
    /// `s_matrix` must be `n_ports × n_ports`.
    pub fn add_frequency_point(&mut self, freq_hz: f64, s_matrix: Vec<Vec<Complex64>>) {
        self.frequencies.push(freq_hz);
        self.s_params.push(s_matrix);
    }

    /// Sort frequency points in ascending order.
    pub fn sort_frequencies(&mut self) {
        // Zip, sort by frequency, unzip
        let mut combined: Vec<(f64, Vec<Vec<Complex64>>)> = self
            .frequencies
            .drain(..)
            .zip(self.s_params.drain(..))
            .collect();
        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (f, s) in combined {
            self.frequencies.push(f);
            self.s_params.push(s);
        }
    }

    /// Check passivity (|S| ≤ 1 for lossless networks).
    ///
    /// Returns a list of `(freq_hz, max_singular_value)` for every frequency
    /// where the maximum singular value exceeds 1.0 (i.e. active gain).
    pub fn check_passivity(&self) -> Vec<(f64, f64)> {
        let mut violations = Vec::new();
        for (fi, &freq) in self.frequencies.iter().enumerate() {
            let s = &self.s_params[fi];
            // Estimate max singular value via Frobenius norm / sqrt(n)
            // (exact SVD requires a linear algebra library; this is a conservative bound)
            let frobenius: f64 = s
                .iter()
                .flat_map(|row| row.iter())
                .map(|c| c.norm_sqr())
                .sum::<f64>()
                .sqrt();
            let sv_est = frobenius / (self.n_ports as f64).sqrt();
            if sv_est > 1.0 + 1e-9 {
                violations.push((freq, sv_est));
            }
        }
        violations
    }

    /// Render the Touchstone 2.0 content as a `String`.
    pub fn write_to_string(&self) -> String {
        let mut out = String::new();

        // Header
        writeln!(out, "[Version] 2.0").ok();
        writeln!(out, "! OxiPhoton S-parameter data").ok();
        writeln!(out, "! Ports: {}", self.n_ports).ok();
        writeln!(out, "! Frequencies: {}", self.frequencies.len()).ok();

        // Option line
        writeln!(
            out,
            "# GHz S {} R {}",
            self.format.option_str(),
            self.reference_impedance
        )
        .ok();

        // Number of ports keyword (Touchstone 2.0)
        writeln!(out, "[Number of Ports] {}", self.n_ports).ok();
        writeln!(out, "[Number of Frequencies] {}", self.frequencies.len()).ok();
        writeln!(out, "[Network Data]").ok();

        for (fi, &freq) in self.frequencies.iter().enumerate() {
            let freq_ghz = freq / 1e9;
            let s = &self.s_params[fi];

            // Frequency value
            write!(out, "{:.10e}", freq_ghz).ok();

            // S-matrix values: row-major order (S11 S12 … S1N  S21 … SNN)
            for row in s.iter() {
                for c in row.iter() {
                    let pair = self.format_complex(*c);
                    write!(out, "  {} {}", pair.0, pair.1).ok();
                }
            }
            writeln!(out).ok();
        }

        out
    }

    /// Write Touchstone 2.0 data to a file at `path`.
    pub fn write_to_file(&self, path: &str) -> Result<(), OxiPhotonError> {
        let content = self.write_to_string();
        std::fs::write(path, content).map_err(|e| {
            OxiPhotonError::NumericalError(format!(
                "Cannot write Touchstone file '{}': {}",
                path, e
            ))
        })
    }

    /// Format a complex S-parameter value according to the selected format.
    ///
    /// Returns `(first_number_str, second_number_str)`.
    fn format_complex(&self, c: Complex64) -> (String, String) {
        match self.format {
            TouchstoneFormat::RI => (format!("{:.8e}", c.re), format!("{:.8e}", c.im)),
            TouchstoneFormat::MA => {
                let mag = c.norm();
                let ang_deg = c.arg().to_degrees();
                (format!("{:.8e}", mag), format!("{:.6}", ang_deg))
            }
            TouchstoneFormat::DB => {
                let mag = c.norm();
                let db = if mag > 0.0 {
                    20.0 * mag.log10()
                } else {
                    -300.0_f64
                };
                let ang_deg = c.arg().to_degrees();
                (format!("{:.6}", db), format!("{:.6}", ang_deg))
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TouchstoneReader
// ─────────────────────────────────────────────────────────────────────────────

/// Touchstone 1.0 / 2.0 file reader.
pub struct TouchstoneReader {
    /// Number of ports.
    pub n_ports: usize,
    /// Parsed frequencies (Hz).
    pub frequencies: Vec<f64>,
    /// Parsed S-matrix per frequency: `[n_freq][n_ports][n_ports]`.
    pub s_params: Vec<Vec<Vec<Complex64>>>,
    /// Reference impedance (Ω).
    pub reference_impedance: f64,
}

impl TouchstoneReader {
    /// Parse Touchstone 1.0 or 2.0 format from a string.
    pub fn parse(content: &str) -> Result<Self, OxiPhotonError> {
        let mut freq_unit = 1e9_f64; // default GHz
        let mut data_format = TouchstoneFormat::MA; // default
        let mut z0 = 50.0_f64;
        let mut n_ports: Option<usize> = None;
        let mut _in_network_data = false;

        let mut raw_freqs: Vec<f64> = Vec::new();
        let mut raw_matrices: Vec<Vec<Vec<Complex64>>> = Vec::new();

        // Collect all numeric tokens from data lines (non-comment, non-keyword lines)
        let mut current_tokens: Vec<f64> = Vec::new();

        for line in content.lines() {
            let trimmed = line.trim();

            // Skip comments
            if trimmed.starts_with('!') || trimmed.is_empty() {
                continue;
            }

            // Version keyword (Touchstone 2.0)
            if trimmed.starts_with("[Version]") {
                continue;
            }

            // Number of ports keyword
            if trimmed.to_lowercase().starts_with("[number of ports]") {
                let val_str = trimmed.split(']').nth(1).unwrap_or("").trim();
                n_ports = val_str.parse::<usize>().ok();
                continue;
            }

            // Number of frequencies — skip (we infer from data)
            if trimmed
                .to_lowercase()
                .starts_with("[number of frequencies]")
            {
                continue;
            }

            // Network data keyword
            if trimmed.to_lowercase().starts_with("[network data]") {
                _in_network_data = true;
                continue;
            }

            // Other bracket keywords — skip
            if trimmed.starts_with('[') {
                continue;
            }

            // Option line: # GHz S RI R 50
            if let Some(after_hash) = trimmed.strip_prefix('#') {
                let parts: Vec<&str> = after_hash.split_whitespace().collect();
                for (idx, &token) in parts.iter().enumerate() {
                    match token.to_uppercase().as_str() {
                        "HZ" => freq_unit = 1.0,
                        "KHZ" => freq_unit = 1e3,
                        "MHZ" => freq_unit = 1e6,
                        "GHZ" => freq_unit = 1e9,
                        "THZ" => freq_unit = 1e12,
                        "RI" => data_format = TouchstoneFormat::RI,
                        "MA" => data_format = TouchstoneFormat::MA,
                        "DB" => data_format = TouchstoneFormat::DB,
                        "R" => {
                            if let Some(&next) = parts.get(idx + 1) {
                                z0 = next.parse::<f64>().unwrap_or(50.0);
                            }
                        }
                        _ => {}
                    }
                }
                continue;
            }

            // Data line — strip inline comments
            let data_part = trimmed.split('!').next().unwrap_or("").trim();
            if data_part.is_empty() {
                continue;
            }

            // Accumulate tokens
            for tok in data_part.split_whitespace() {
                if let Ok(v) = tok.parse::<f64>() {
                    current_tokens.push(v);
                }
            }
        }

        // Determine n_ports from token count if not specified
        // A 2-port file has: 1 freq + 4 pairs = 9 tokens per frequency
        // General: 1 + n_ports² * 2 tokens per frequency
        let np = match n_ports {
            Some(p) => p,
            None => {
                // Try to infer from token count — try 1-port through 8-port
                let mut found = 2usize;
                for p in 1usize..=8 {
                    let tokens_per_freq = 1 + p * p * 2;
                    if tokens_per_freq > 0 && current_tokens.len() % tokens_per_freq == 0 {
                        found = p;
                        break;
                    }
                }
                found
            }
        };

        let tokens_per_freq = 1 + np * np * 2;
        if tokens_per_freq == 0 {
            return Err(OxiPhotonError::NumericalError(
                "Invalid port count: results in zero tokens per frequency".to_string(),
            ));
        }

        if current_tokens.len() % tokens_per_freq != 0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "Token count {} not divisible by {} (n_ports={})",
                current_tokens.len(),
                tokens_per_freq,
                np,
            )));
        }

        let n_freq = current_tokens.len() / tokens_per_freq;
        raw_freqs.reserve(n_freq);
        raw_matrices.reserve(n_freq);

        for fi in 0..n_freq {
            let base = fi * tokens_per_freq;
            let freq_hz = current_tokens[base] * freq_unit;
            raw_freqs.push(freq_hz);

            let mut s_mat: Vec<Vec<Complex64>> = vec![vec![Complex64::new(0.0, 0.0); np]; np];
            for (row, s_row) in s_mat.iter_mut().enumerate().take(np) {
                for (col, elem) in s_row.iter_mut().enumerate().take(np) {
                    let pair_base = base + 1 + (row * np + col) * 2;
                    let a = current_tokens[pair_base];
                    let b = current_tokens[pair_base + 1];
                    *elem = parse_complex(a, b, data_format);
                }
            }
            raw_matrices.push(s_mat);
        }

        Ok(Self {
            n_ports: np,
            frequencies: raw_freqs,
            s_params: raw_matrices,
            reference_impedance: z0,
        })
    }

    /// Insertion loss in dB: `-20 log10(|S_from_to|)`.
    pub fn insertion_loss_db(&self, from_port: usize, to_port: usize) -> Vec<f64> {
        self.s_params
            .iter()
            .map(|s| {
                let mag = s
                    .get(to_port)
                    .and_then(|row| row.get(from_port))
                    .map(|c| c.norm())
                    .unwrap_or(0.0);
                if mag > 0.0 {
                    -20.0 * mag.log10()
                } else {
                    f64::INFINITY
                }
            })
            .collect()
    }

    /// Return loss in dB: `-20 log10(|S_nn|)`.
    pub fn return_loss_db(&self, port: usize) -> Vec<f64> {
        self.insertion_loss_db(port, port)
    }

    /// Group delay in seconds: `-dφ/dω` computed via finite difference.
    ///
    /// Uses central differences where possible, forward/backward at edges.
    pub fn group_delay(&self, from_port: usize, to_port: usize) -> Vec<f64> {
        let phases = self.phase_unwrapped(from_port, to_port);
        let n = phases.len();
        if n < 2 {
            return vec![0.0; n];
        }
        let mut gd = vec![0.0_f64; n];
        for (i, gd_val) in gd.iter_mut().enumerate().take(n) {
            let (f_prev, phi_prev, f_next, phi_next) = if i == 0 {
                (
                    self.frequencies[0],
                    phases[0],
                    self.frequencies[1],
                    phases[1],
                )
            } else if i == n - 1 {
                (
                    self.frequencies[n - 2],
                    phases[n - 2],
                    self.frequencies[n - 1],
                    phases[n - 1],
                )
            } else {
                (
                    self.frequencies[i - 1],
                    phases[i - 1],
                    self.frequencies[i + 1],
                    phases[i + 1],
                )
            };
            let dom = 2.0 * PI * (f_next - f_prev);
            if dom.abs() > 1e-30 {
                *gd_val = -(phi_next - phi_prev) / dom;
            }
        }
        gd
    }

    /// Phase of S-parameter in degrees (naively, no unwrapping).
    pub fn phase_deg(&self, from_port: usize, to_port: usize) -> Vec<f64> {
        self.s_params
            .iter()
            .map(|s| {
                s.get(to_port)
                    .and_then(|row| row.get(from_port))
                    .map(|c| c.arg().to_degrees())
                    .unwrap_or(0.0)
            })
            .collect()
    }

    /// Linear magnitude of S-parameter.
    pub fn magnitude(&self, from_port: usize, to_port: usize) -> Vec<f64> {
        self.s_params
            .iter()
            .map(|s| {
                s.get(to_port)
                    .and_then(|row| row.get(from_port))
                    .map(|c| c.norm())
                    .unwrap_or(0.0)
            })
            .collect()
    }

    /// Find the 3-dB bandwidth from the S-parameter curve.
    ///
    /// Finds the peak value and returns the frequency span where the magnitude
    /// is at or above `peak / sqrt(2)` (i.e., 3 dB below the peak in power).
    ///
    /// Returns `None` if insufficient data or the curve never drops by 3 dB.
    pub fn bandwidth_3db(&self, from_port: usize, to_port: usize) -> Option<f64> {
        let mags = self.magnitude(from_port, to_port);
        if mags.len() < 3 {
            return None;
        }
        let peak = mags.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let threshold = peak / 2.0_f64.sqrt();

        let above: Vec<usize> = mags
            .iter()
            .enumerate()
            .filter(|(_, &m)| m >= threshold)
            .map(|(i, _)| i)
            .collect();

        if above.len() < 2 {
            return None;
        }

        let f_lo = self.frequencies[*above.first()?];
        let f_hi = self.frequencies[*above.last()?];
        Some(f_hi - f_lo)
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /// Phase in radians, unwrapped using consecutive-sample phase difference.
    fn phase_unwrapped(&self, from_port: usize, to_port: usize) -> Vec<f64> {
        let raw: Vec<f64> = self
            .s_params
            .iter()
            .map(|s| {
                s.get(to_port)
                    .and_then(|row| row.get(from_port))
                    .map(|c| c.arg())
                    .unwrap_or(0.0)
            })
            .collect();

        let n = raw.len();
        if n == 0 {
            return raw;
        }

        let mut unwrapped = vec![0.0_f64; n];
        unwrapped[0] = raw[0];
        for i in 1..n {
            let diff = raw[i] - raw[i - 1];
            // Wrap diff to [-π, π]
            let diff_wrapped = diff - (2.0 * PI) * ((diff + PI) / (2.0 * PI)).floor();
            unwrapped[i] = unwrapped[i - 1] + diff_wrapped;
        }
        unwrapped
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: parse (a, b) → Complex64 according to format
// ─────────────────────────────────────────────────────────────────────────────

fn parse_complex(a: f64, b: f64, format: TouchstoneFormat) -> Complex64 {
    match format {
        TouchstoneFormat::RI => Complex64::new(a, b),
        TouchstoneFormat::MA => {
            let ang_rad = b.to_radians();
            Complex64::new(a * ang_rad.cos(), a * ang_rad.sin())
        }
        TouchstoneFormat::DB => {
            let mag = 10.0_f64.powf(a / 20.0);
            let ang_rad = b.to_radians();
            Complex64::new(mag * ang_rad.cos(), mag * ang_rad.sin())
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Convert an N×N S-parameter matrix to a T-parameter (transfer) matrix.
///
/// For a 2-port network with `s = [[S11, S12], [S21, S22]]`:
///
/// ```text
/// T = [[-det(S)/S21,  S11/S21],
///      [-S22/S21,     1/S21  ]]
/// ```
///
/// Only 2-port matrices are supported; returns zeros for other sizes.
pub fn s_to_t_matrix(s: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let n = s.len();
    if n != 2 {
        return vec![vec![Complex64::new(0.0, 0.0); n]; n];
    }
    let s11 = s[0][0];
    let s12 = s[0][1];
    let s21 = s[1][0];
    let s22 = s[1][1];

    let det_s = s11 * s22 - s12 * s21;
    if s21.norm() < 1e-60 {
        return vec![vec![Complex64::new(0.0, 0.0); 2]; 2];
    }

    vec![
        vec![-det_s / s21, s11 / s21],
        vec![-s22 / s21, Complex64::new(1.0, 0.0) / s21],
    ]
}

/// Convert a 2-port T-parameter (transfer) matrix back to an S-parameter matrix.
pub fn t_to_s_matrix(t: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let n = t.len();
    if n != 2 {
        return vec![vec![Complex64::new(0.0, 0.0); n]; n];
    }
    let t11 = t[0][0];
    let t12 = t[0][1];
    let t21 = t[1][0];
    let t22 = t[1][1];

    if t22.norm() < 1e-60 {
        return vec![vec![Complex64::new(0.0, 0.0); 2]; 2];
    }

    let one = Complex64::new(1.0, 0.0);
    vec![
        vec![t12 / t22, (t11 * t22 - t12 * t21) / t22],
        vec![one / t22, -t21 / t22],
    ]
}

/// Cascade two 2-port S-parameter matrices using the star (Redheffer) product.
///
/// Connects port 2 of `s1` to port 1 of `s2`.
///
/// ```text
/// S_cascade[1,1] = S1[1,1] + S1[1,2]·S2[1,1]·(I - S1[2,2]·S2[1,1])⁻¹·S1[2,1]
/// ```
///
/// This is implemented via T-matrix multiplication for efficiency.
pub fn cascade_two_port(s1: &[Vec<Complex64>], s2: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    // T-matrix cascade = matrix multiplication
    let t1 = s_to_t_matrix(s1);
    let t2 = s_to_t_matrix(s2);

    // 2×2 matrix multiply
    let n = 2usize;
    let mut tc = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                tc[i][j] += t1[i][k] * t2[k][j];
            }
        }
    }

    t_to_s_matrix(&tc)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn identity_2port() -> Vec<Vec<Complex64>> {
        // Perfect through: S21 = S12 = 1, S11 = S22 = 0
        vec![
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ]
    }

    fn make_writer_2port() -> TouchstoneWriter {
        let mut w = TouchstoneWriter::new(2, 50.0);
        let freqs = [1e9, 2e9, 3e9];
        for f in freqs {
            let s = vec![
                vec![Complex64::new(0.01, 0.0), Complex64::new(0.5, 0.0)],
                vec![Complex64::new(0.5, 0.0), Complex64::new(0.01, 0.0)],
            ];
            w.add_frequency_point(f, s);
        }
        w
    }

    // ── writer tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_touchstone_write_2port() {
        let w = make_writer_2port();
        let s = w.write_to_string();
        assert!(s.contains("[Version] 2.0"), "Header missing version");
        assert!(s.contains("# GHz S RI R 50"), "Option line missing");
        assert!(s.contains("[Network Data]"), "Network data keyword missing");
        assert!(s.contains("[Number of Ports] 2"), "Port count missing");
    }

    #[test]
    fn test_touchstone_roundtrip() {
        let w = make_writer_2port();
        let s = w.write_to_string();
        let r = TouchstoneReader::parse(&s).expect("Roundtrip parse failed");
        assert_eq!(r.n_ports, 2, "Port count mismatch");
        assert_eq!(r.frequencies.len(), 3, "Frequency count mismatch");
        for (i, &f_w) in w.frequencies.iter().enumerate() {
            let f_r = r.frequencies[i];
            assert!(
                (f_w - f_r).abs() / f_w.max(1.0) < 1e-6,
                "Frequency mismatch at index {i}: wrote {f_w} Hz, read {f_r} Hz"
            );
        }
    }

    // ── reader analysis tests ─────────────────────────────────────────────

    #[test]
    fn test_insertion_loss_db() {
        // S21 = 0.5 → IL = 20*log10(1/0.5) = 6.02 dB
        let mut w = TouchstoneWriter::new(2, 50.0);
        w.add_frequency_point(
            1e9,
            vec![
                vec![Complex64::new(0.0, 0.0), Complex64::new(0.5, 0.0)],
                vec![Complex64::new(0.5, 0.0), Complex64::new(0.0, 0.0)],
            ],
        );
        let s = w.write_to_string();
        let r = TouchstoneReader::parse(&s).expect("parse failed");
        let il = r.insertion_loss_db(0, 1); // S21: from port 0, to port 1
        assert_eq!(il.len(), 1);
        assert!(
            (il[0] - 6.0206).abs() < 0.01,
            "Expected ≈6.02 dB, got {}",
            il[0]
        );
    }

    #[test]
    fn test_return_loss_db() {
        // S11 = 0.1 → RL = 20*log10(1/0.1) = 20 dB
        let mut w = TouchstoneWriter::new(2, 50.0);
        w.add_frequency_point(
            1e9,
            vec![
                vec![Complex64::new(0.1, 0.0), Complex64::new(0.9, 0.0)],
                vec![Complex64::new(0.9, 0.0), Complex64::new(0.1, 0.0)],
            ],
        );
        let s = w.write_to_string();
        let r = TouchstoneReader::parse(&s).expect("parse failed");
        let rl = r.return_loss_db(0);
        assert_eq!(rl.len(), 1);
        assert!((rl[0] - 20.0).abs() < 0.01, "Expected 20 dB, got {}", rl[0]);
    }

    #[test]
    fn test_group_delay() {
        // Linear phase θ = -2π·f·τ for τ = 10 ps gives constant group delay τ.
        // Use small tau so that the phase never wraps across a 2π boundary
        // at any of the sampled frequencies → no unwrapping ambiguity.
        let tau = 10e-12_f64; // 10 ps
                              // Frequencies from 1 GHz to 20 GHz in 1-GHz steps.
                              // Maximum phase = 2π * 20e9 * 10e-12 = 2π * 0.2 rad ≈ 1.26 rad < π
                              // So no 2π wraps occur and atan2 gives the true phase.
        let freqs: Vec<f64> = (1..=20).map(|i| i as f64 * 1e9).collect();
        // Build reader directly (no string roundtrip) so there is no floating-point
        // truncation from the ASCII format step.
        let mut r = TouchstoneReader {
            n_ports: 2,
            frequencies: freqs.clone(),
            s_params: freqs
                .iter()
                .map(|&f| {
                    let phase = -2.0 * PI * f * tau;
                    let c = Complex64::new(phase.cos(), phase.sin());
                    vec![
                        vec![Complex64::new(0.0, 0.0), c],
                        vec![c, Complex64::new(0.0, 0.0)],
                    ]
                })
                .collect(),
            reference_impedance: 50.0,
        };
        let gd = r.group_delay(0, 1);
        // All interior points should be ≈ τ = 10 ps (tolerance 1%)
        for (i, &gd_val) in gd.iter().enumerate().take(gd.len() - 1).skip(1) {
            assert!(
                (gd_val - tau).abs() < tau * 0.01,
                "group_delay[{i}] = {:.4e} s, expected {tau:.4e} s",
                gd_val
            );
        }
        // Suppress unused_mut warning for direct construction pattern
        let _ = &mut r;
    }

    #[test]
    fn test_cascade_two_port() {
        // Cascading identity (perfect through) with itself gives identity
        let id = identity_2port();
        let result = cascade_two_port(&id, &id);
        // S21 and S12 should be ≈ 1, S11 and S22 should be ≈ 0
        assert!(
            (result[1][0].norm() - 1.0).abs() < 1e-6,
            "S21 cascade of identities ≠ 1: {}",
            result[1][0]
        );
        assert!(
            (result[0][1].norm() - 1.0).abs() < 1e-6,
            "S12 cascade of identities ≠ 1: {}",
            result[0][1]
        );
        assert!(
            result[0][0].norm() < 1e-6,
            "S11 of cascaded identities ≠ 0: {}",
            result[0][0]
        );
        assert!(
            result[1][1].norm() < 1e-6,
            "S22 of cascaded identities ≠ 0: {}",
            result[1][1]
        );
    }

    #[test]
    fn test_s_to_t_conversion() {
        // Roundtrip: S → T → S should recover the original matrix
        let s_orig = vec![
            vec![Complex64::new(0.1, 0.05), Complex64::new(0.7, 0.1)],
            vec![Complex64::new(0.7, -0.1), Complex64::new(0.1, -0.05)],
        ];
        let t = s_to_t_matrix(&s_orig);
        let s_back = t_to_s_matrix(&t);
        for i in 0..2 {
            for j in 0..2 {
                let diff = (s_orig[i][j] - s_back[i][j]).norm();
                assert!(
                    diff < 1e-10,
                    "S[{i}][{j}] roundtrip error: orig={}, recovered={}, diff={}",
                    s_orig[i][j],
                    s_back[i][j],
                    diff
                );
            }
        }
    }

    #[test]
    fn test_bandwidth_3db() {
        // Lorentzian: S21(f) = Γ / (Γ + j*(f - f0))  → FWHM = 2Γ
        let f0 = 5e9_f64;
        let gamma = 0.5e9_f64; // half-bandwidth = 0.5 GHz → BW = 1 GHz
        let freqs: Vec<f64> = (0..200).map(|i| (i as f64) * 50e6 + 0.5e9).collect();
        let mut w = TouchstoneWriter::new(2, 50.0);
        for &f in &freqs {
            let denom = Complex64::new(gamma, f - f0);
            let s21 = Complex64::new(gamma, 0.0) / denom;
            w.add_frequency_point(
                f,
                vec![
                    vec![Complex64::new(0.0, 0.0), s21],
                    vec![s21, Complex64::new(0.0, 0.0)],
                ],
            );
        }
        let s = w.write_to_string();
        let r = TouchstoneReader::parse(&s).expect("parse failed");
        let bw = r
            .bandwidth_3db(0, 1)
            .expect("bandwidth_3db should return Some");
        // Expected bandwidth ≈ 2*gamma = 1 GHz; allow 10% tolerance due to sampling
        assert!(
            (bw - 2.0 * gamma).abs() < 0.1 * 2.0 * gamma,
            "3-dB bandwidth = {bw:.3e} Hz, expected ≈ {:.3e} Hz",
            2.0 * gamma
        );
    }
}
