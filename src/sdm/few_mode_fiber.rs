//! Few-Mode Fiber (FMF) and Multicore Fiber (MCF) propagation models.
//!
//! # Few-Mode Fiber
//! Models differential group delay (DGD), chromatic dispersion, and MIMO capacity
//! for fibers guiding 2–20 LP modes.
//!
//! # Multicore Fiber
//! Models inter-core cross-talk, capacity, and core layout geometries for
//! 7-core and 19-core fiber designs.
//!
//! Reference:
//! - Ryf et al., "Mode-Division Multiplexing Over 96 km of Few-Mode Fiber", JLT 2012
//! - Saitoh & Matsuo, "Multicore Fiber Technology", JLT 2016

use num_complex::Complex64;
use std::f64::consts::PI;

// ── Physical constants ────────────────────────────────────────────────────────

/// Speed of light in vacuum \[m/s\]
const C_LIGHT: f64 = 2.997_924_58e8;

// ── Few-Mode Fiber ────────────────────────────────────────────────────────────

/// Few-mode fiber specification and propagation model.
///
/// The β-coefficients follow the Taylor expansion of the propagation constant:
///   β(ω) = β₀ + β₁(ω−ω₀) + β₂/2·(ω−ω₀)² + β₃/6·(ω−ω₀)³
///
/// where:
///   β₀ \[rad/m\]: phase constant at centre frequency
///   β₁ \[s/m\]:   inverse group velocity (1/v_g)
///   β₂ \[s²/m\]:  group velocity dispersion (GVD)
///   β₃ \[s³/m\]:  third-order dispersion (TOD)
pub struct FewModeFiber {
    /// Total spatial modes per polarisation.
    pub n_modes: usize,
    /// Core radius \[µm\].
    pub core_radius_um: f64,
    /// Core refractive index.
    pub n_core: f64,
    /// Cladding refractive index.
    pub n_clad: f64,
    /// Fiber length \[km\].
    pub length_km: f64,
    /// Per-mode attenuation \[dB/km\]. Length = `n_modes`.
    pub loss_db_per_km: Vec<f64>,
    /// Taylor coefficients `[mode][order]` where order ∈ {0,1,2,3}.
    /// β₁ in s/m, β₂ in s²/m, etc.
    pub beta_coeffs: Vec<Vec<f64>>,
    /// Centre wavelength \[m\].
    pub wavelength: f64,
}

impl FewModeFiber {
    /// 2-mode fiber: LP01 + LP11 (6 spatial + polarisation modes total).
    ///
    /// Parameters representative of a commercially designed 2-mode fiber at 1550 nm:
    /// - β₁ difference ≈ 200 ps/km between LP01 and LP11
    /// - D ≈ 18 ps/(nm·km) for LP01, 17 ps/(nm·km) for LP11
    pub fn new_2mode(length_km: f64, wavelength: f64) -> Self {
        let n_modes = 2;
        let k0 = 2.0 * PI / wavelength;
        let n_core = 1.455_f64;
        let n_clad = 1.444_f64;
        // beta0 from n_eff·k0
        let b0_01 = n_core * k0;
        let b0_11 = (n_core - 0.002) * k0;
        // beta1 = n_g/c; DGD ≈ 200 ps/km → Δβ₁ = 2e-10 s/m
        let b1_01 = 1.4682_f64 / C_LIGHT;
        let b1_11 = b1_01 + 2.0e-10_f64;
        // beta2: D = -λ²/(2πc) * beta2 → beta2 = -D*λ²/(2πc)  [D in s/m²]
        // D=18 ps/(nm·km) = 18e-6 s/m² → β₂ = -D*λ²/(2πc)
        let d_01 = 18.0e-6_f64; // s/m²
        let d_11 = 17.0e-6_f64;
        let b2_01 = -d_01 * wavelength * wavelength / (2.0 * PI * C_LIGHT);
        let b2_11 = -d_11 * wavelength * wavelength / (2.0 * PI * C_LIGHT);
        // beta3: TOD ~ 0.08 ps/(nm²·km) = 0.08e3 s³/m
        let b3 = 0.08e-39_f64; // s³/m typical silica value

        Self {
            n_modes,
            core_radius_um: 9.0,
            n_core,
            n_clad,
            length_km,
            loss_db_per_km: vec![0.20, 0.21],
            beta_coeffs: vec![vec![b0_01, b1_01, b2_01, b3], vec![b0_11, b1_11, b2_11, b3]],
            wavelength,
        }
    }

    /// 6-mode fiber: LP01, LP11a, LP11b, LP21a, LP21b, LP02.
    ///
    /// β₁ values chosen to give realistic DGD values between mode groups.
    pub fn new_6mode(length_km: f64, wavelength: f64) -> Self {
        let n_modes = 6;
        let k0 = 2.0 * PI / wavelength;
        let n_core = 1.457_f64;
        let n_clad = 1.440_f64;
        let b2_base = -2.0e-26_f64; // s²/m typical anomalous dispersion
        let b3 = 0.08e-39_f64;
        // β₁ increments: 0, +100, +100, +300, +300, +350 ps/km in s/m
        let delta_b1: [f64; 6] = [0.0, 1.0e-10, 1.0e-10, 3.0e-10, 3.0e-10, 3.5e-10];
        let b1_base = 1.4700_f64 / C_LIGHT;
        let loss = [0.20, 0.21, 0.21, 0.22, 0.22, 0.22_f64];
        let coeffs: Vec<Vec<f64>> = (0..n_modes)
            .map(|i| {
                vec![
                    n_core * k0 - i as f64 * 50.0, // rough β₀ offset
                    b1_base + delta_b1[i],
                    b2_base,
                    b3,
                ]
            })
            .collect();
        Self {
            n_modes,
            core_radius_um: 13.0,
            n_core,
            n_clad,
            length_km,
            loss_db_per_km: loss.to_vec(),
            beta_coeffs: coeffs,
            wavelength,
        }
    }

    /// 10-mode fiber: LP01, LP11(×2), LP21(×2), LP02, LP31(×2), LP12(×2).
    pub fn new_10mode(length_km: f64, wavelength: f64) -> Self {
        let n_modes = 10;
        let k0 = 2.0 * PI / wavelength;
        let n_core = 1.460_f64;
        let n_clad = 1.440_f64;
        let b2_base = -2.0e-26_f64;
        let b3 = 0.08e-39_f64;
        let b1_base = 1.472_f64 / C_LIGHT;
        // Group delays: each mode group has slightly different β₁
        let delta_b1_ps_km: [f64; 10] = [
            0.0, 80.0, 80.0, 200.0, 200.0, 220.0, 400.0, 400.0, 500.0, 500.0,
        ];
        let coeffs: Vec<Vec<f64>> = (0..n_modes)
            .map(|i| {
                let db1 = delta_b1_ps_km[i] * 1.0e-12 / 1.0e3; // ps/km → s/m
                vec![n_core * k0 - i as f64 * 30.0, b1_base + db1, b2_base, b3]
            })
            .collect();
        let loss: Vec<f64> = (0..n_modes).map(|i| 0.20 + i as f64 * 0.005).collect();
        Self {
            n_modes,
            core_radius_um: 17.0,
            n_core,
            n_clad,
            length_km,
            loss_db_per_km: loss,
            beta_coeffs: coeffs,
            wavelength,
        }
    }

    /// Differential Group Delay between mode `a` and mode `b`:
    ///   DGD_{ab} = |β₁_a − β₁_b| · L   \[ps\]
    pub fn dgd_ps(&self, mode_a: usize, mode_b: usize) -> f64 {
        if mode_a >= self.n_modes || mode_b >= self.n_modes {
            return 0.0;
        }
        let b1_a = self.beta_coeffs[mode_a].get(1).copied().unwrap_or(0.0);
        let b1_b = self.beta_coeffs[mode_b].get(1).copied().unwrap_or(0.0);
        (b1_a - b1_b).abs() * self.length_km * 1.0e3 * 1.0e12 // s/m * m → s → ps
    }

    /// Maximum DGD across all mode pairs \[ps\].
    pub fn group_delay_spread_ps(&self) -> f64 {
        let b1s: Vec<f64> = self
            .beta_coeffs
            .iter()
            .map(|c| c.get(1).copied().unwrap_or(0.0))
            .collect();
        let b1_max = b1s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let b1_min = b1s.iter().cloned().fold(f64::INFINITY, f64::min);
        (b1_max - b1_min) * self.length_km * 1.0e3 * 1.0e12
    }

    /// Chromatic dispersion for mode `i` \[ps/nm\] over the full fiber length.
    ///
    /// D \[ps/(nm·km)\] = −(λ²)/(2πc) · β₂ · 2π/λ² · 1e−12/(1e−9 · 1e3)
    /// Simplified: D\[ps/nm/km\] = -2πc/λ² * β₂ * 1e3 converted units.
    pub fn dispersion_ps_per_nm(&self, mode: usize) -> f64 {
        if mode >= self.n_modes {
            return 0.0;
        }
        let b2 = self.beta_coeffs[mode].get(2).copied().unwrap_or(0.0);
        // D [s/m²] = -2πc/λ² * β₂  → [ps/(nm·km)] = D * 1e3 (km→m) * 1e12 (s→ps) / 1e9 (m→nm)
        let d_si = -2.0 * PI * C_LIGHT / (self.wavelength * self.wavelength) * b2;
        // d_si in s/m²; convert: 1 s/m² = 1e6 ps/(nm·km)
        let d_ps_nm_km = d_si * 1.0e6;
        d_ps_nm_km * self.length_km // total [ps/nm]
    }

    /// MIMO-equalised capacity using Shannon formula.
    ///
    /// C = n_modes · log₂(1 + SNR) · B   \[Tb/s\]
    /// where SNR is per-mode signal-to-noise ratio.
    pub fn mimo_capacity_tbps(&self, launch_power_dbm: f64, snr_db: f64, baud_gbaud: f64) -> f64 {
        let snr_linear = 10.0_f64.powf(snr_db / 10.0);
        let bits_per_symbol = (1.0 + snr_linear).log2();
        // Subtract mode-averaged loss
        let avg_loss = self.average_loss_db_per_km() * self.length_km;
        let power_at_rx_dbm = launch_power_dbm - avg_loss;
        // If received power is below threshold, capacity degrades
        let power_penalty = if power_at_rx_dbm < -30.0 { 0.1 } else { 1.0 };
        let bw_hz = baud_gbaud * 1.0e9;
        self.n_modes as f64 * bw_hz * bits_per_symbol * power_penalty * 1.0e-12
    }

    /// Average loss across all modes \[dB/km\].
    pub fn average_loss_db_per_km(&self) -> f64 {
        if self.loss_db_per_km.is_empty() {
            return 0.0;
        }
        self.loss_db_per_km.iter().sum::<f64>() / self.loss_db_per_km.len() as f64
    }

    /// Complex propagation factor for mode `i` at angular frequency offset `Δω`:
    ///   H(Δω) = exp(i · \[β₁·Δω + β₂/2·Δω² + β₃/6·Δω³\] · L − α/2·L)
    pub fn propagation_matrix(&self, mode: usize, freq_offset_rad: f64) -> Complex64 {
        if mode >= self.n_modes {
            return Complex64::new(0.0, 0.0);
        }
        let coeffs = &self.beta_coeffs[mode];
        let b1 = coeffs.get(1).copied().unwrap_or(0.0);
        let b2 = coeffs.get(2).copied().unwrap_or(0.0);
        let b3 = coeffs.get(3).copied().unwrap_or(0.0);
        let dw = freq_offset_rad;
        let l_m = self.length_km * 1.0e3;
        let phase = (b1 * dw + 0.5 * b2 * dw * dw + b3 / 6.0 * dw * dw * dw) * l_m;
        let alpha_lin =
            self.loss_db_per_km[mode] * self.length_km / (10.0 / std::f64::consts::LN_10);
        let amplitude = (-alpha_lin / 2.0).exp();
        Complex64::new(0.0, phase).exp() * amplitude
    }
}

// ── Multicore Fiber ───────────────────────────────────────────────────────────

/// Core arrangement geometry for a multicore fiber.
#[derive(Debug, Clone)]
pub enum CoreLayout {
    /// Cores arranged in a straight line.
    Linear,
    /// Hexagonally packed cores with the specified number of rings.
    /// 1 ring → 7 cores, 2 rings → 19 cores, 3 rings → 37 cores.
    Hexagonal { rings: usize },
    /// Square lattice with `side × side` cores.
    Square { side: usize },
    /// Cores arranged uniformly on a ring of given radius.
    Ring { n: usize, radius_um: f64 },
}

/// Multicore fiber specification.
///
/// Each core is assumed identical (homogeneous MCF). The inter-core cross-talk
/// is modelled using coupled-power theory:
///
///   XT \[dB\] = 10·log₁₀(2·κ²·L / β)
///
/// where κ is the coupling coefficient between adjacent cores.
pub struct MulticoreFiber {
    /// Number of cores.
    pub n_cores: usize,
    /// Centre-to-centre spacing between adjacent cores \[µm\].
    pub core_pitch_um: f64,
    /// Individual core radius \[µm\].
    pub core_radius_um: f64,
    /// Core refractive index.
    pub n_core: f64,
    /// Cladding refractive index.
    pub n_clad: f64,
    /// Fiber length \[km\].
    pub length_km: f64,
    /// Core layout geometry.
    pub layout: CoreLayout,
}

impl MulticoreFiber {
    /// Standard 7-core hexagonal MCF (1 central + 6 surrounding cores).
    ///
    /// Typical parameters: pitch = 40 µm, a = 4.5 µm, n_core = 1.455
    pub fn new_7core(length_km: f64, _wavelength: f64) -> Self {
        Self {
            n_cores: 7,
            core_pitch_um: 40.0,
            core_radius_um: 4.5,
            n_core: 1.4550,
            n_clad: 1.4440,
            length_km,
            layout: CoreLayout::Hexagonal { rings: 1 },
        }
    }

    /// Standard 19-core hexagonal MCF (1 + 6 + 12 cores).
    pub fn new_19core(length_km: f64, _wavelength: f64) -> Self {
        Self {
            n_cores: 19,
            core_pitch_um: 35.0,
            core_radius_um: 4.2,
            n_core: 1.4550,
            n_clad: 1.4440,
            length_km,
            layout: CoreLayout::Hexagonal { rings: 2 },
        }
    }

    /// Returns (x, y) positions \[µm\] of each core centre.
    pub fn core_positions(&self) -> Vec<[f64; 2]> {
        match &self.layout {
            CoreLayout::Linear => {
                let d = self.core_pitch_um;
                let offset = (self.n_cores as f64 - 1.0) / 2.0 * d;
                (0..self.n_cores)
                    .map(|i| [i as f64 * d - offset, 0.0])
                    .collect()
            }

            CoreLayout::Hexagonal { rings } => {
                let d = self.core_pitch_um;
                let mut pos = vec![[0.0_f64; 2]];
                for ring in 1..=*rings {
                    let r = ring as f64;
                    // 6 sides of hexagon, each side has `ring` cores
                    // Direction vectors for the 6 sides
                    let dirs: [(f64, f64); 6] = [
                        (1.0, 0.0),
                        (0.5, -(3.0_f64).sqrt() / 2.0),
                        (-0.5, -(3.0_f64).sqrt() / 2.0),
                        (-1.0, 0.0),
                        (-0.5, (3.0_f64).sqrt() / 2.0),
                        (0.5, (3.0_f64).sqrt() / 2.0),
                    ];
                    // Start position for this ring
                    let mut x = r * d;
                    let mut y = 0.0_f64;
                    for (dx, dy) in dirs {
                        for _ in 0..ring {
                            if pos.len() < self.n_cores {
                                pos.push([x, y]);
                            }
                            x += dx * d;
                            y += dy * d;
                        }
                    }
                }
                pos.truncate(self.n_cores);
                pos
            }

            CoreLayout::Square { side } => {
                let d = self.core_pitch_um;
                let offset = (*side as f64 - 1.0) / 2.0 * d;
                let mut pos = Vec::new();
                'outer: for row in 0..*side {
                    for col in 0..*side {
                        if pos.len() >= self.n_cores {
                            break 'outer;
                        }
                        pos.push([col as f64 * d - offset, row as f64 * d - offset]);
                    }
                }
                pos
            }

            CoreLayout::Ring { n, radius_um } => (0..*n)
                .map(|i| {
                    let theta = 2.0 * PI * i as f64 / *n as f64;
                    [radius_um * theta.cos(), radius_um * theta.sin()]
                })
                .collect(),
        }
    }

    /// Coupling coefficient κ between adjacent cores \[1/m\].
    ///
    /// Uses the simplified formula:
    ///   κ ≈ κ₀ · exp(−α_eff · (d − 2a))
    /// where d is the pitch, a is core radius, and α_eff is the evanescent
    /// decay constant of the fundamental mode in the cladding.
    pub fn coupling_coefficient(&self, wavelength: f64) -> f64 {
        let a = self.core_radius_um * 1.0e-6;
        let d = self.core_pitch_um * 1.0e-6;
        let k0 = 2.0 * PI / wavelength;
        let na2 = self.n_core * self.n_core - self.n_clad * self.n_clad;
        let na = if na2 > 0.0 { na2.sqrt() } else { 0.01 };
        let v = k0 * a * na;
        // Evanescent decay: w = sqrt(β² − k0²·n2²) ≈ k0·NA·sqrt(1−(u/V)²)
        // For LP01: u ≈ 2.0 (first approximation)
        let u = if v > 2.0 { 2.0 } else { v * 0.8 };
        let w = (v * v - u * u).max(0.0).sqrt();
        let alpha_eff = w / a; // 1/m
        let gap = d - 2.0 * a;
        // κ₀: pre-factor proportional to u²/(V²·a·K1²(w)·K0²(w))
        let k0_prefactor = (u * u / (v * v * a)).abs();
        k0_prefactor * (-alpha_eff * gap.max(0.0)).exp()
    }

    /// Mean inter-core cross-talk using coupled-power theory \[dB\].
    ///
    ///   XT \[dB\] = 10·log₁₀(2·κ²·L / (β·1))
    ///           ≈ 10·log₁₀(2·h·L)  with h = κ²/β
    pub fn inter_core_xt_db(&self, wavelength: f64) -> f64 {
        let kappa = self.coupling_coefficient(wavelength);
        let k0 = 2.0 * PI / wavelength;
        let beta = k0 * self.n_core; // approximate β ≈ k0·n_core
        let l_m = self.length_km * 1.0e3;
        let h = kappa * kappa / beta;
        let xt = 2.0 * h * l_m;
        if xt <= 0.0 {
            return f64::NEG_INFINITY;
        }
        10.0 * xt.log10()
    }

    /// Cross-talk between a specific pair of cores \[dB\].
    ///
    /// Uses the same coupled-power formula but scales by the inverse of the
    /// squared inter-core distance relative to the pitch.
    pub fn crosstalk_between(&self, core1: usize, core2: usize, wavelength: f64) -> f64 {
        let positions = self.core_positions();
        if core1 >= positions.len() || core2 >= positions.len() {
            return f64::NEG_INFINITY;
        }
        let p1 = positions[core1];
        let p2 = positions[core2];
        let dist_um = ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2)).sqrt();
        let pitch = self.core_pitch_um;
        // κ decays exponentially with gap; scale from adjacent value
        let kappa_adj = self.coupling_coefficient(wavelength);
        let gap_adj = pitch - 2.0 * self.core_radius_um; // gap for adjacent cores
        let gap_pair = dist_um - 2.0 * self.core_radius_um;
        // α_eff: extract from kappa_adj = k0_pre * exp(-α_eff * gap_adj)
        let a = self.core_radius_um * 1.0e-6;
        let k0 = 2.0 * PI / wavelength;
        let na = (self.n_core * self.n_core - self.n_clad * self.n_clad)
            .max(0.0)
            .sqrt();
        let v = k0 * a * na;
        let u = if v > 2.0 { 2.0 } else { v * 0.8 };
        let w = (v * v - u * u).max(0.0).sqrt();
        let alpha_eff = w / a; // 1/m
        let gap_adj_m = gap_adj.max(0.0) * 1.0e-6;
        let gap_pair_m = gap_pair.max(0.0) * 1.0e-6;
        let kappa = kappa_adj * ((-alpha_eff * (gap_pair_m - gap_adj_m)).exp());
        let beta = k0 * self.n_core;
        let l_m = self.length_km * 1.0e3;
        let h = kappa * kappa / beta;
        let xt = 2.0 * h * l_m;
        if xt <= 0.0 {
            return f64::NEG_INFINITY;
        }
        10.0 * xt.log10()
    }

    /// Total capacity of the MCF system \[Pb/s\].
    ///
    ///   C = n_cores · SE · B_WDM
    ///
    /// where SE is spectral efficiency \[b/s/Hz\] and B_WDM is the WDM bandwidth \[Hz\].
    pub fn total_capacity_pbps(&self, se_bps_per_hz: f64, bandwidth_thz: f64) -> f64 {
        self.n_cores as f64 * se_bps_per_hz * bandwidth_thz * 1.0e12 * 1.0e-15
    }

    /// Effective cladding diameter \[µm\].
    ///
    /// Estimated from core positions plus a safety margin of 2× core radius.
    pub fn cladding_diameter_um(&self) -> f64 {
        let positions = self.core_positions();
        let r_max = positions
            .iter()
            .map(|p| (p[0] * p[0] + p[1] * p[1]).sqrt())
            .fold(0.0_f64, f64::max);
        // Add core radius + minimum cladding thickness (typically 35 µm)
        (r_max + self.core_radius_um + 35.0) * 2.0
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_2mode_dgd() {
        let fmf = FewModeFiber::new_2mode(100.0, 1.55e-6);
        let dgd = fmf.dgd_ps(0, 1);
        // DGD should be ~200 ps/km × 100 km = 20000 ps = 20 ns, or smaller
        assert!(dgd > 0.0, "DGD should be positive");
        assert!(dgd < 1.0e8, "DGD should be physically plausible");
    }

    #[test]
    fn test_group_delay_spread_monotone() {
        let fmf6 = FewModeFiber::new_6mode(100.0, 1.55e-6);
        let fmf2 = FewModeFiber::new_2mode(100.0, 1.55e-6);
        let spread6 = fmf6.group_delay_spread_ps();
        let spread2 = fmf2.group_delay_spread_ps();
        // 6-mode fiber should have larger group delay spread
        assert!(spread6 >= spread2, "6-mode should have >= spread vs 2-mode");
    }

    #[test]
    fn test_propagation_matrix_unit_amplitude_zero_offset() {
        let fmf = FewModeFiber::new_2mode(0.0, 1.55e-6); // 0 km → no loss/dispersion
        let h = fmf.propagation_matrix(0, 0.0);
        // At zero frequency offset and zero length, |H| ≈ 1
        assert_abs_diff_eq!(h.norm(), 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_7core_positions_count() {
        let mcf = MulticoreFiber::new_7core(100.0, 1.55e-6);
        let pos = mcf.core_positions();
        assert_eq!(pos.len(), 7, "7-core MCF should have 7 positions");
    }

    #[test]
    fn test_19core_positions_count() {
        let mcf = MulticoreFiber::new_19core(100.0, 1.55e-6);
        let pos = mcf.core_positions();
        assert_eq!(pos.len(), 19, "19-core MCF should have 19 positions");
    }

    #[test]
    fn test_average_loss_positive() {
        let fmf = FewModeFiber::new_6mode(100.0, 1.55e-6);
        let loss = fmf.average_loss_db_per_km();
        assert!(
            loss > 0.0 && loss < 1.0,
            "Loss should be 0.2–0.3 dB/km, got {}",
            loss
        );
    }

    #[test]
    fn test_mimo_capacity_positive() {
        let fmf = FewModeFiber::new_6mode(100.0, 1.55e-6);
        let cap = fmf.mimo_capacity_tbps(0.0, 20.0, 32.0);
        assert!(cap > 0.0, "Capacity should be positive");
    }

    #[test]
    fn test_cladding_diameter_7core() {
        let mcf = MulticoreFiber::new_7core(100.0, 1.55e-6);
        let diam = mcf.cladding_diameter_um();
        // 7-core with pitch 40 µm: outer cores at 40 µm from centre → diam ≈ 150–200 µm
        assert!(
            diam > 80.0 && diam < 400.0,
            "Cladding diameter out of range: {} µm",
            diam
        );
    }

    #[test]
    fn test_capacity_scales_with_cores() {
        let mcf7 = MulticoreFiber::new_7core(100.0, 1.55e-6);
        let mcf19 = MulticoreFiber::new_19core(100.0, 1.55e-6);
        let c7 = mcf7.total_capacity_pbps(8.0, 4.0);
        let c19 = mcf19.total_capacity_pbps(8.0, 4.0);
        assert!(c19 > c7, "19-core should have higher capacity than 7-core");
    }
}
