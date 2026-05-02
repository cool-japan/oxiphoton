//! End-to-end SiPh link S-parameter cascade.
//!
//! Provides a unified pipeline from device physics through cascaded S-matrix
//! to link performance metrics (insertion loss, group delay, bandwidth).
//!
//! # Architecture
//!
//! - [`SiPhElement`] — trait for any 2-port optical element
//! - [`WaveguideSection`] — passive waveguide with loss and dispersion
//! - [`Splitter50_50`] — ideal 50:50 Y-junction
//! - [`DirectionalCoupler`] — coupler with configurable ratio and excess loss
//! - [`SiPhLink`] — sequence of elements; cascades S-matrices across frequency

#![cfg(feature = "interconnect")]

use num_complex::Complex64;
use std::f64::consts::PI;

// Physical constants
const C_LIGHT: f64 = 2.997_924_58e8; // m/s

// ─────────────────────────────────────────────────────────────────────────────
// SiPhElement trait
// ─────────────────────────────────────────────────────────────────────────────

/// A SiPh element that can be represented as a frequency-dependent 2-port
/// S-matrix.
///
/// Each element returns `(S11, S21, S12, S22)` at each requested frequency.
/// All entries are complex-valued.
pub trait SiPhElement {
    /// Returns `[S11, S21, S12, S22]` at each frequency in `freq_hz`.
    fn s_params(&self, freq_hz: &[f64]) -> Vec<[Complex64; 4]>;
}

// ─────────────────────────────────────────────────────────────────────────────
// WaveguideSection
// ─────────────────────────────────────────────────────────────────────────────

/// Passive straight waveguide segment with propagation loss and group delay.
///
/// The element is modelled as a lossless-phase-shift + attenuation:
///
/// ```text
/// S11 = S22 = 0       (no reflections for ideal straight guide)
/// S21 = S12 = A · exp(i·φ)
///   A   = 10^(-loss_dB / 20)
///   φ   = 2π·n_g·f/c · L_m
/// ```
///
/// The dispersion parameter is used only to select a dispersion model
/// in future extensions; the current implementation uses group index
/// dispersion only (first-order dispersion).
#[derive(Debug, Clone)]
pub struct WaveguideSection {
    /// Waveguide physical length (µm)
    pub length_um: f64,
    /// Propagation loss coefficient (dB/cm)
    pub loss_db_per_cm: f64,
    /// Group index (dimensionless, typically 3.5–4.5 for Si waveguide)
    pub group_index: f64,
    /// Chromatic dispersion parameter (ps/nm/km) — for future use
    pub dispersion_ps_per_nm_per_km: f64,
}

impl SiPhElement for WaveguideSection {
    fn s_params(&self, freq_hz: &[f64]) -> Vec<[Complex64; 4]> {
        let length_m = self.length_um * 1e-6;
        let length_cm = self.length_um * 1e-4;
        // Total insertion loss (dB), independent of frequency for this model
        let loss_db = self.loss_db_per_cm * length_cm;
        // Amplitude factor: |S21| = 10^(-loss_dB / 20)
        let amplitude = 10.0_f64.powf(-loss_db / 20.0);

        freq_hz
            .iter()
            .map(|&f| {
                // Propagation phase: φ = 2π · n_g · f / c · L
                // Use the physics convention S21 = A · exp(-i·φ) so that
                // group delay τ_g = -d(phase)/dω = +n_g·L/c > 0 (causal)
                let phi = 2.0 * PI * self.group_index * f / C_LIGHT * length_m;
                let s21 = Complex64::from_polar(amplitude, -phi);
                [
                    Complex64::new(0.0, 0.0), // S11
                    s21,                      // S21
                    s21,                      // S12 (reciprocal)
                    Complex64::new(0.0, 0.0), // S22
                ]
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Splitter50_50
// ─────────────────────────────────────────────────────────────────────────────

/// Ideal 1×2 Y-junction splitter with perfect 50:50 power split.
///
/// Modelled as a 2-port device with:
/// ```text
/// S21 = S12 = 1/√2  (3 dB amplitude split, no excess loss)
/// S11 = S22 = 0
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Splitter50_50;

impl SiPhElement for Splitter50_50 {
    fn s_params(&self, freq_hz: &[f64]) -> Vec<[Complex64; 4]> {
        let s21 = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        freq_hz
            .iter()
            .map(|_| {
                [
                    Complex64::new(0.0, 0.0), // S11
                    s21,                      // S21
                    s21,                      // S12
                    Complex64::new(0.0, 0.0), // S22
                ]
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DirectionalCoupler
// ─────────────────────────────────────────────────────────────────────────────

/// Directional coupler with configurable coupling ratio and excess loss.
///
/// The 2-port model exposes the **through port** only.  The cross port power
/// is dissipated into the other output arm and is not tracked by this model.
///
/// Through-port amplitude:
/// ```text
/// S21_through = sqrt(1 - κ) · 10^(-excess_loss_dB/20)
/// ```
/// where `κ = coupling_ratio` (fraction of power to the cross port).
#[derive(Debug, Clone, Copy)]
pub struct DirectionalCoupler {
    /// Power coupling fraction to the cross port (0.0 = no coupling, 1.0 = full cross)
    pub coupling_ratio: f64,
    /// Excess insertion loss of the coupler (dB)
    pub excess_loss_db: f64,
}

impl SiPhElement for DirectionalCoupler {
    fn s_params(&self, freq_hz: &[f64]) -> Vec<[Complex64; 4]> {
        let kappa = self.coupling_ratio.clamp(0.0, 1.0);
        let excess_amplitude = 10.0_f64.powf(-self.excess_loss_db / 20.0);
        let through_amplitude = (1.0 - kappa).sqrt() * excess_amplitude;
        let s21 = Complex64::new(through_amplitude, 0.0);

        freq_hz
            .iter()
            .map(|_| {
                [
                    Complex64::new(0.0, 0.0), // S11
                    s21,                      // S21 (through port)
                    s21,                      // S12
                    Complex64::new(0.0, 0.0), // S22
                ]
            })
            .collect()
    }
}

impl DirectionalCoupler {
    /// Insertion loss to the cross port (dB).
    ///
    /// Power coupled to the cross port is:
    /// ```text
    /// |S21_cross|^2 = κ · 10^(-excess_loss_dB/10)
    /// loss_cross_dB = -10 · log10(κ · 10^(-excess/10))
    /// ```
    pub fn cross_port_loss_db(&self) -> f64 {
        let kappa = self.coupling_ratio.clamp(1e-20, 1.0);
        let power = kappa * 10.0_f64.powf(-self.excess_loss_db / 10.0);
        -10.0 * power.max(1e-40).log10()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2-port cascade helper
// ─────────────────────────────────────────────────────────────────────────────

/// Cascade two 2-port S-matrices in series (signal flows A → B).
///
/// Let port indices be: 1 = input, 2 = output.
/// Cascade formula (Pozar §4.4):
///
/// ```text
/// D       = 1 - S11_B · S22_A
/// S21_tot = S21_B · S21_A / D
/// S12_tot = S12_A · S12_B / D
/// S11_tot = S11_A + S21_A · S12_A · S11_B / D
/// S22_tot = S22_B + S21_B · S12_B · S22_A / D
/// ```
///
/// Returns `[S11, S21, S12, S22]` of the cascaded network.
#[inline]
fn cascade_two(a: [Complex64; 4], b: [Complex64; 4]) -> [Complex64; 4] {
    let [s11a, s21a, s12a, s22a] = a;
    let [s11b, s21b, s12b, s22b] = b;

    let denom = Complex64::new(1.0, 0.0) - s11b * s22a;

    // Avoid division by exactly zero — if denom is zero the chain is
    // resonant / ill-conditioned; return a sentinel (high loss).
    if denom.norm() < 1e-30 {
        return [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
    }

    let s21_tot = s21b * s21a / denom;
    let s12_tot = s12a * s12b / denom;
    let s11_tot = s11a + s21a * s12a * s11b / denom;
    let s22_tot = s22b + s21b * s12b * s22a / denom;

    [s11_tot, s21_tot, s12_tot, s22_tot]
}

// ─────────────────────────────────────────────────────────────────────────────
// SiPhLink
// ─────────────────────────────────────────────────────────────────────────────

/// A cascaded SiPh link composed of a sequence of 2-port SiPh elements.
///
/// Elements are stored in signal-propagation order.  The cascade is computed
/// by applying the 2-port cascade formula iteratively across all elements.
///
/// An empty link is the identity (S21 = S12 = 1, S11 = S22 = 0).
// Note: no #[derive(Debug)] because Box<dyn SiPhElement> is not Debug
pub struct SiPhLink {
    elements: Vec<Box<dyn SiPhElement>>,
}

impl SiPhLink {
    /// Create a new, empty link (identity element).
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
        }
    }

    /// Append a [`SiPhElement`] to the end of the link (builder pattern).
    pub fn push(mut self, el: impl SiPhElement + 'static) -> Self {
        self.elements.push(Box::new(el));
        self
    }

    /// Compute the cascaded S-matrix at each frequency.
    ///
    /// Returns `[S11, S21, S12, S22]` at each frequency in `freq_hz`.
    /// An empty link returns identity S-parameters.
    pub fn cascade(&self, freq_hz: &[f64]) -> Vec<[Complex64; 4]> {
        let n_freq = freq_hz.len();
        if n_freq == 0 {
            return Vec::new();
        }

        // Identity S-matrix: S21=S12=1, S11=S22=0
        let identity = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let mut result: Vec<[Complex64; 4]> = vec![[zero, identity, identity, zero]; n_freq];

        for element in &self.elements {
            let s = element.s_params(freq_hz);
            for (i, &sp) in s.iter().enumerate() {
                result[i] = cascade_two(result[i], sp);
            }
        }

        result
    }

    /// Insertion loss (dB) at each frequency: IL = −20·log₁₀(|S21|).
    pub fn insertion_loss_db(&self, freq_hz: &[f64]) -> Vec<f64> {
        self.cascade(freq_hz)
            .into_iter()
            .map(|[_, s21, _, _]| {
                let mag = s21.norm();
                if mag < 1e-40 {
                    400.0 // practical floor (very high loss)
                } else {
                    -20.0 * mag.log10()
                }
            })
            .collect()
    }

    /// Group delay (ps) at each frequency, derived from dφ/dω of S21.
    ///
    /// Computed analytically from the unwrapped phase of S21 using central
    /// differences.  The phase is accumulated from consecutive frequency
    /// points using `arg(S21[k+1] / S21[k])` to avoid wrap-around.
    ///
    /// Returns an empty vector if fewer than 2 frequency points are provided.
    pub fn group_delay_ps(&self, freq_hz: &[f64]) -> Vec<f64> {
        if freq_hz.len() < 2 {
            return Vec::new();
        }

        let params = self.cascade(freq_hz);

        // Accumulate unwrapped phase: φ[k] by summing phase increments
        let mut phi = vec![0.0_f64; freq_hz.len()];
        phi[0] = params[0][1].arg(); // initial phase at first frequency
        for i in 1..freq_hz.len() {
            // Phase increment: arg(S21[i] / S21[i-1]) unwraps automatically
            let ratio = params[i][1] / params[i - 1][1];
            // If S21 is zero, treat as zero phase increment
            let increment = if ratio.norm() < 1e-40 {
                0.0
            } else {
                ratio.arg()
            };
            phi[i] = phi[i - 1] + increment;
        }

        // Group delay τ_g = -dφ/dω = -dφ/(2π·df)
        // Use central differences, edge points use one-sided difference
        let n = freq_hz.len();
        let mut gd = vec![0.0_f64; n];

        // Interior points: central difference
        for i in 1..(n - 1) {
            let dphi = phi[i + 1] - phi[i - 1];
            let domega = 2.0 * PI * (freq_hz[i + 1] - freq_hz[i - 1]);
            if domega.abs() > 0.0 {
                gd[i] = -dphi / domega * 1e12; // convert s → ps
            }
        }
        // Edge points: one-sided difference
        if n >= 2 {
            let dphi0 = phi[1] - phi[0];
            let domega0 = 2.0 * PI * (freq_hz[1] - freq_hz[0]);
            if domega0.abs() > 0.0 {
                gd[0] = -dphi0 / domega0 * 1e12;
            }
            let dphi_n = phi[n - 1] - phi[n - 2];
            let domega_n = 2.0 * PI * (freq_hz[n - 1] - freq_hz[n - 2]);
            if domega_n.abs() > 0.0 {
                gd[n - 1] = -dphi_n / domega_n * 1e12;
            }
        }

        gd
    }

    /// 3-dB optical bandwidth (Hz) of the link from the S21 response.
    ///
    /// Searches for the first frequency point where |S21|² drops below half
    /// of its peak value.  Returns `None` if the sweep is too narrow to
    /// determine the bandwidth.
    pub fn bandwidth_3db_hz(&self, freq_hz: &[f64]) -> Option<f64> {
        if freq_hz.len() < 2 {
            return None;
        }

        let params = self.cascade(freq_hz);
        let powers: Vec<f64> = params.iter().map(|p| p[1].norm_sqr()).collect();

        // Find peak power
        let peak = powers.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if peak <= 0.0 {
            return None;
        }

        let half_peak = peak / 2.0;

        // Find the index of the peak
        let peak_idx = powers
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)?;

        // Search right of peak for -3 dB point
        let right_idx = (peak_idx + 1..freq_hz.len()).find(|&i| powers[i] < half_peak)?;

        // Linearly interpolate between right_idx-1 and right_idx
        let p_lo = powers[right_idx - 1];
        let p_hi = powers[right_idx];
        let f_lo = freq_hz[right_idx - 1];
        let f_hi = freq_hz[right_idx];

        // Avoid division by zero
        if (p_hi - p_lo).abs() < 1e-40 {
            return None;
        }

        // Interpolate to find exact frequency where power = half_peak
        let t = (half_peak - p_lo) / (p_hi - p_lo);
        let f_3db_right = f_lo + t * (f_hi - f_lo);

        // Bandwidth = 2 × (f_3dB_right - f_peak) for symmetric response
        // or just use f_peak_center relative to f_3dB_right if only one side
        let f_peak = freq_hz[peak_idx];
        Some(2.0 * (f_3db_right - f_peak).abs())
    }

    /// Number of elements in the link.
    pub fn n_elements(&self) -> usize {
        self.elements.len()
    }
}

impl Default for SiPhLink {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-stage chip-to-chip link cascade
// ─────────────────────────────────────────────────────────────────────────────

/// Compose two 2-port S-matrices in series using Mason's rule.
///
/// Layout convention (same as [`cascade_two`]): `[S11, S21, S12, S22]`.
///
/// ```text
/// denom   = 1 − a_S22 · b_S11
/// S11_tot = a_S11 + a_S21 · a_S12 · b_S11 / denom
/// S21_tot = b_S21 · a_S21 / denom
/// S12_tot = a_S12 · b_S12 / denom
/// S22_tot = b_S22 + b_S21 · b_S12 · a_S22 / denom
/// ```
///
/// When the denominator approaches zero (resonant ill-conditioning) returns
/// the same high-loss sentinel as [`cascade_two`].
fn compose_s_matrices(a: [Complex64; 4], b: [Complex64; 4]) -> [Complex64; 4] {
    // a = [S11_a, S21_a, S12_a, S22_a]  (indices 0,1,2,3)
    // b = [S11_b, S21_b, S12_b, S22_b]
    let denom = Complex64::new(1.0, 0.0) - a[3] * b[0]; // 1 - S22_a * S11_b

    if denom.norm() < 1e-30 {
        // Resonant / ill-conditioned: return high-loss sentinel (identity with 0 transmission)
        return [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
    }

    [
        a[0] + a[1] * a[2] * b[0] / denom, // S11_tot
        b[1] * a[1] / denom,               // S21_tot
        a[2] * b[2] / denom,               // S12_tot
        b[3] + b[1] * b[2] * a[3] / denom, // S22_tot
    ]
}

/// Cascade multiple [`SiPhLink`] stages into a single end-to-end S-matrix response.
///
/// Each element of the returned `Vec` is `[S11, S21, S12, S22]` at the
/// corresponding frequency in `freq_grid_hz`.
///
/// # Arguments
///
/// * `stages`       — ordered slice of link references (signal flows left to right)
/// * `freq_grid_hz` — frequency grid (Hz)
///
/// # Returns
///
/// - Empty `stages` → identity S-matrices (`[1, 0, 0, 1]`) at each frequency.
/// - Single stage → identical to calling `stage.cascade(freq_grid_hz)`.
/// - Multiple stages → pairwise Mason's-rule composition along the chain.
pub fn chip_to_chip_link_response(
    stages: &[&SiPhLink],
    freq_grid_hz: &[f64],
) -> Vec<[Complex64; 4]> {
    if stages.is_empty() {
        // Identity S-matrix: S11=0, S21=1, S12=1, S22=0 — but convention here is
        // [S11, S21, S12, S22] so identity has S21=S12=1, S11=S22=0.
        return freq_grid_hz
            .iter()
            .map(|_| {
                [
                    Complex64::new(0.0, 0.0), // S11
                    Complex64::new(1.0, 0.0), // S21
                    Complex64::new(1.0, 0.0), // S12
                    Complex64::new(0.0, 0.0), // S22
                ]
            })
            .collect();
    }

    let mut acc = stages[0].cascade(freq_grid_hz);
    for stage in &stages[1..] {
        let next = stage.cascade(freq_grid_hz);
        for (cur, &nxt) in acc.iter_mut().zip(next.iter()) {
            *cur = compose_s_matrices(*cur, nxt);
        }
    }
    acc
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_link_zero_insertion_loss() {
        let link = SiPhLink::new();
        let freqs = [193.41e12_f64];
        let il = link.insertion_loss_db(&freqs);
        assert!(
            il[0].abs() < 1e-9,
            "empty link should have 0 dB IL, got {:.6}",
            il[0]
        );
    }

    #[test]
    fn waveguide_10mm_at_3db_per_cm() {
        let wg = WaveguideSection {
            length_um: 10_000.0,
            loss_db_per_cm: 3.0,
            group_index: 4.2,
            dispersion_ps_per_nm_per_km: 0.0,
        };
        let link = SiPhLink::new().push(wg);
        let freqs = [193.41e12_f64];
        let il = link.insertion_loss_db(&freqs);
        // 10 mm = 1 cm → 3 dB/cm × 1 cm = 3 dB
        assert!(
            (il[0] - 3.0).abs() < 0.01,
            "Expected 3 dB IL, got {:.4}",
            il[0]
        );
    }

    #[test]
    fn splitter_3db_loss() {
        let link = SiPhLink::new().push(Splitter50_50);
        let freqs = [193.41e12_f64];
        let il = link.insertion_loss_db(&freqs);
        // Ideal 3 dB splitter: IL = -20*log10(1/sqrt(2)) = 10*log10(2) ≈ 3.0103
        assert!(
            (il[0] - 3.0103).abs() < 0.01,
            "Splitter50_50 IL should be ~3.01 dB, got {:.4}",
            il[0]
        );
    }

    #[test]
    fn directional_coupler_through_loss() {
        let dc = DirectionalCoupler {
            coupling_ratio: 0.5,
            excess_loss_db: 0.0,
        };
        let link = SiPhLink::new().push(dc);
        let freqs = [193.41e12_f64];
        let il = link.insertion_loss_db(&freqs);
        // Through port: S21 = sqrt(0.5) → IL = 3.01 dB
        assert!(
            (il[0] - 3.0103).abs() < 0.01,
            "DirectionalCoupler IL should be ~3.01 dB, got {:.4}",
            il[0]
        );
    }
}
