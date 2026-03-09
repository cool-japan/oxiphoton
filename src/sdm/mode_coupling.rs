//! Mode Coupling, MIMO Equalization, and Mode Conversion for SDM Fibers.
//!
//! # Contents
//! - `RandomModeCoupling`: statistical model for random mode coupling along FMF
//! - `FmfMimoEqualizer`: digital MIMO equalizer (CMA adaptation)
//! - `PhotonicLantern`: spatial mode multiplexer / demultiplexer
//! - `LpgModeConverter`: long-period grating mode converter
//!
//! # References
//! - Antonelli et al., "Stokes-Space Analysis of Modal Dispersion", OE 2012
//! - Ryf et al., "Mode-Division Multiplexing", JLT 2012
//! - Birks et al., "The Photonic Lantern", Adv. Opt. Photon. 2015

use num_complex::Complex64;
use std::f64::consts::PI;

// ── Random Mode Coupling ──────────────────────────────────────────────────────

/// Statistical model of random mode coupling along a few-mode fiber.
///
/// Based on coupled-power equations (Gloge's theory):
///   dP_i/dz = Σ_j h_{ij} (P_j − P_i)
///
/// where h_{ij} = 2κ²·l_c is the coupling power coefficient,
/// κ is the field coupling coefficient, and l_c is the correlation length.
pub struct RandomModeCoupling {
    /// Number of spatial modes (per polarisation).
    pub n_modes: usize,
    /// Field coupling coefficient h = κ² · l_c \[1/m\].
    pub coupling_strength: f64,
    /// Perturbation correlation length l_c \[m\].
    pub correlation_length_m: f64,
}

impl RandomModeCoupling {
    /// Construct a new coupling model.
    ///
    /// # Arguments
    /// - `n_modes`: number of distinct spatial modes
    /// - `h`: power coupling coefficient \[1/m\] (h = κ²·l_c)
    /// - `lc`: perturbation correlation length \[m\]
    pub fn new(n_modes: usize, h: f64, lc: f64) -> Self {
        Self {
            n_modes,
            coupling_strength: h,
            correlation_length_m: lc,
        }
    }

    /// Power coupling matrix M for propagation step `dz` \[m\].
    ///
    /// Off-diagonal: M_{ij} = h·dz  (i ≠ j)
    /// Diagonal:     M_{ii} = 1 − (n_modes−1)·h·dz
    ///
    /// Returns an n×n matrix as a `Vec<Vec<f64>>`.
    pub fn coupling_matrix(&self, dz: f64) -> Vec<Vec<f64>> {
        let n = self.n_modes;
        let h_dz = self.coupling_strength * dz;
        let off = h_dz.min(1.0 / (n as f64 + 1.0)); // clamp to preserve positivity
        let diag = 1.0 - (n as f64 - 1.0) * off;
        let mut m = vec![vec![0.0_f64; n]; n];
        for (i, row) in m.iter_mut().enumerate().take(n) {
            for (j, elem) in row.iter_mut().enumerate().take(n) {
                if i == j {
                    *elem = diag.max(0.0);
                } else {
                    *elem = off;
                }
            }
        }
        m
    }

    /// Mode mixing length L_mix = 1 / (2·h·(n_modes−1)) \[m\], converted to km.
    ///
    /// Beyond L_mix the mode powers equalise ("strong coupling regime").
    pub fn mixing_length_km(&self) -> f64 {
        let n = self.n_modes;
        if self.coupling_strength <= 0.0 || n <= 1 {
            return f64::INFINITY;
        }
        let l_mix_m = 1.0 / (2.0 * self.coupling_strength * (n as f64 - 1.0));
        l_mix_m / 1.0e3
    }

    /// Returns `true` if the fiber is in the strong coupling regime:
    ///   L ≫ L_mix.
    pub fn is_strongly_coupled(&self, fiber_length_km: f64) -> bool {
        let l_mix = self.mixing_length_km();
        fiber_length_km > 10.0 * l_mix
    }

    /// Effective DGD in the strong coupling regime.
    ///
    /// In the strongly-coupled regime, mode mixing reduces the DGD as:
    ///   DGD_eff = DGD_uncoupled / √(L / L_mix)
    ///
    /// In the weak coupling regime, DGD_eff ≈ DGD_uncoupled.
    pub fn effective_dgd_ps(&self, dgd_uncoupled_ps: f64, fiber_length_km: f64) -> f64 {
        let l_mix = self.mixing_length_km();
        if l_mix <= 0.0 || l_mix.is_infinite() || fiber_length_km <= l_mix {
            // Weak coupling: no significant reduction
            return dgd_uncoupled_ps;
        }
        let ratio = fiber_length_km / l_mix;
        dgd_uncoupled_ps / ratio.sqrt()
    }

    /// Mode-dependent loss (MDL) in the presence of random mode coupling.
    ///
    /// Strong coupling reduces MDL as:
    ///   MDL_eff \[dB\] = MDL_0 / √(n_modes · L / L_mix)
    pub fn effective_mdl_db(&self, mdl_uncoupled_db: f64, fiber_length_km: f64) -> f64 {
        let l_mix = self.mixing_length_km();
        let n = self.n_modes as f64;
        if l_mix <= 0.0 || l_mix.is_infinite() || fiber_length_km <= l_mix {
            return mdl_uncoupled_db;
        }
        let denom = (n * fiber_length_km / l_mix).sqrt();
        mdl_uncoupled_db / denom
    }

    /// Steady-state power distribution after strong coupling.
    /// In the strongly-coupled regime, all mode powers equalise: P_i = P_total / n.
    pub fn steady_state_powers(&self, total_power: f64) -> Vec<f64> {
        vec![total_power / self.n_modes as f64; self.n_modes]
    }
}

// ── MIMO Equalizer ────────────────────────────────────────────────────────────

/// Digital MIMO equalizer for few-mode fiber coherent receivers.
///
/// Architecture: n_modes × n_modes matrix of FIR filters, each with n_taps taps.
/// Adaptation uses the Constant Modulus Algorithm (CMA), suitable for DP-QPSK
/// and higher-order QAM constellations.
pub struct FmfMimoEqualizer {
    /// Number of spatial modes (equals number of receiver ports).
    pub n_modes: usize,
    /// Number of FIR taps per filter.
    pub n_taps: usize,
    /// Oversampling factor (typically 2 for T/2-spaced equalizers).
    pub tap_spacing_samples: usize,
    /// Equalizer coefficients: `[output_mode][input_mode][tap]`.
    pub equalizer_matrix: Vec<Vec<Vec<Complex64>>>,
    /// CMA adaptation step size μ (controls convergence speed vs. noise).
    pub mu: f64,
}

impl FmfMimoEqualizer {
    /// Initialise a MIMO equalizer.
    ///
    /// The center taps are set to identity (output mode `i` uses input mode `i`)
    /// to provide a known starting point for adaptation.
    pub fn new(n_modes: usize, n_taps: usize, mu: f64) -> Self {
        let center = n_taps / 2;
        let equalizer_matrix: Vec<Vec<Vec<Complex64>>> = (0..n_modes)
            .map(|out| {
                (0..n_modes)
                    .map(|inp| {
                        let mut taps = vec![Complex64::new(0.0, 0.0); n_taps];
                        if inp == out {
                            taps[center] = Complex64::new(1.0, 0.0);
                        }
                        taps
                    })
                    .collect()
            })
            .collect();
        Self {
            n_modes,
            n_taps,
            tap_spacing_samples: 2,
            equalizer_matrix,
            mu,
        }
    }

    /// Apply the equalizer to a block of received samples.
    ///
    /// # Arguments
    /// - `received`: `[n_modes][n_samples]` — complex received signal per mode
    ///
    /// # Returns
    /// Equalised output vector of length `n_modes`, one symbol per call.
    /// (Processes only the first usable symbol from the block.)
    pub fn apply(&self, received: &[Vec<Complex64>]) -> Vec<Complex64> {
        let n = self.n_modes.min(received.len());
        let n_samp = received.first().map(|v| v.len()).unwrap_or(0);
        let mut output = vec![Complex64::new(0.0, 0.0); n];
        for (out, out_val) in output.iter_mut().enumerate().take(n) {
            for (inp, recv_mode) in received.iter().enumerate().take(n) {
                let taps = &self.equalizer_matrix[out][inp];
                // Convolve taps with received signal (at time 0)
                for (k, &tap) in taps.iter().enumerate() {
                    let idx = k * self.tap_spacing_samples;
                    if idx < n_samp {
                        *out_val += tap * recv_mode[idx];
                    }
                }
            }
        }
        output
    }

    /// CMA coefficient update for one received symbol.
    ///
    /// Error signal: e_i = (R² − |y_i|²) · y_i   (R = 1 for normalised QPSK)
    /// Update: w_{ij}\[k\] += μ · e_i · x_j*\[k\]
    pub fn update_cma(&mut self, received: &[Complex64], equalized: &[Complex64]) {
        let r_squared = 1.0_f64; // CMA radius = 1 (normalised constellation)
        let n = self.n_modes.min(equalized.len()).min(received.len());
        for (out, &y) in equalized.iter().enumerate().take(n) {
            let error = (r_squared - y.norm_sqr()) * y;
            for (inp, &x) in received.iter().enumerate().take(n) {
                let mu_err = self.mu * error;
                for k in 0..self.n_taps {
                    // w += μ · e · x*
                    self.equalizer_matrix[out][inp][k] += mu_err * x.conj();
                }
            }
        }
    }

    /// Total multiply-accumulate (MAC) operations per symbol.
    ///
    ///   Complexity = n_modes² × n_taps × (taps per mode)
    pub fn complexity_multiplications_per_symbol(&self) -> usize {
        self.n_modes * self.n_modes * self.n_taps
    }

    /// Minimum required tap count to equalise a given DGD and symbol rate.
    ///
    ///   N_taps ≥ DGD_total / T_symbol  (with 20% margin)
    ///
    /// where T_symbol = 1 / (baud_rate · oversampling).
    pub fn required_taps(dgd_total_ps: f64, symbol_rate_gbaud: f64) -> usize {
        if symbol_rate_gbaud <= 0.0 {
            return 1;
        }
        let t_sym_ps = 1.0 / (symbol_rate_gbaud * 1.0e9) * 1.0e12; // ps
        let n_taps = (dgd_total_ps / t_sym_ps * 1.2).ceil() as usize;
        n_taps.max(3)
    }

    /// Estimated convergence time in symbols: ≈ 1 / (μ · n_modes).
    pub fn convergence_symbols(&self) -> usize {
        if self.mu <= 0.0 {
            return usize::MAX;
        }
        (1.0 / (self.mu * self.n_modes as f64)).ceil() as usize
    }
}

// ── Photonic Lantern ──────────────────────────────────────────────────────────

/// Photonic lantern: a tapered waveguide bundle that adiabatically transforms
/// N single-mode fibers into N spatial modes of a few-mode fiber.
///
/// Key performance parameters:
/// - Insertion loss: total excess loss \[dB\]
/// - Mode-dependent loss (MDL): variation of loss across modes \[dB\]
/// - Crosstalk: inter-mode interference at the output \[dB\]
pub struct PhotonicLantern {
    /// Number of modes (= number of SMF ports).
    pub n_modes: usize,
    /// Adiabatic taper length \[mm\].
    pub transition_length_mm: f64,
    /// Insertion loss \[dB\] (averaged across modes).
    pub insertion_loss_db: f64,
    /// Peak mode-dependent loss \[dB\].
    pub mode_dependent_loss_db: f64,
    /// Inter-mode crosstalk \[dB\].
    pub crosstalk_db: f64,
}

impl PhotonicLantern {
    /// Construct a photonic lantern with typical fabricated parameters.
    ///
    /// Scaling relations from literature (Birks et al. 2015):
    /// - Transition length scales with √n_modes
    /// - Insertion loss ≈ 0.3 + 0.1·log2(n_modes) dB
    /// - Crosstalk improves with longer taper
    pub fn new(n_modes: usize) -> Self {
        let transition_length_mm = 25.0 * (n_modes as f64).sqrt();
        let insertion_loss_db = 0.3 + 0.1 * (n_modes as f64).log2();
        let mode_dependent_loss_db = 0.5 + 0.05 * n_modes as f64;
        // Crosstalk: −20 to −30 dB for well-fabricated lanterns
        let crosstalk_db = -25.0 - 5.0 * (n_modes as f64).log2();
        Self {
            n_modes,
            transition_length_mm,
            insertion_loss_db,
            mode_dependent_loss_db,
            crosstalk_db,
        }
    }

    /// Mode selectivity (dB): isolation between desired and spurious modes.
    ///   selectivity = |crosstalk| − MDL
    pub fn mode_selectivity_db(&self) -> f64 {
        self.crosstalk_db.abs() - self.mode_dependent_loss_db
    }

    /// Power transfer efficiency (fraction ∈ \[0, 1\]).
    pub fn efficiency(&self) -> f64 {
        10.0_f64.powf(-self.insertion_loss_db / 10.0)
    }

    /// 3-dB operational bandwidth \[nm\].
    ///
    /// Adiabatic lanterns have broad bandwidth; typical values 100–300 nm.
    pub fn bandwidth_nm(&self) -> f64 {
        // Longer transition → better adiabaticity → broader bandwidth
        let base_bw = 150.0_f64;
        base_bw * (self.transition_length_mm / 25.0).sqrt()
    }

    /// Transfer matrix T (n_modes × n_modes) from SMF ports to FMF modes.
    ///
    /// Diagonal: √(η_mode), Off-diagonal: ε (crosstalk amplitude).
    /// The matrix is approximately unitary for a lossless lantern.
    pub fn transfer_matrix(&self) -> Vec<Vec<Complex64>> {
        let n = self.n_modes;
        let eta = self.efficiency();
        // Amplitude: sqrt(efficiency) per mode diagonal
        let diag_amp = eta.sqrt();
        // Crosstalk amplitude: from crosstalk_db = 20·log10(ε/diag_amp)
        let xt_ratio = 10.0_f64.powf(self.crosstalk_db / 20.0);
        let xt_amp = diag_amp * xt_ratio;
        let mut t = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for (i, t_row) in t.iter_mut().enumerate().take(n) {
            for (j, elem) in t_row.iter_mut().enumerate().take(n) {
                if i == j {
                    *elem = Complex64::new(diag_amp, 0.0);
                } else {
                    // Distribute crosstalk with a small phase offset per pair
                    let phase = 2.0 * PI * (i * n + j) as f64 / (n * n) as f64;
                    *elem = Complex64::new(xt_amp * phase.cos(), xt_amp * phase.sin());
                }
            }
        }
        t
    }
}

// ── Long-Period Grating Mode Converter ───────────────────────────────────────

/// Long-period grating (LPG) mode converter.
///
/// An LPG inscribed in a few-mode fiber resonantly couples two LP modes when
/// the grating period satisfies the phase-matching condition:
///
///   Λ = 2π / |β₁ − β₂|   (resonance period)
///
/// Conversion efficiency at resonance: η_max = sin²(κ·L)
pub struct LpgModeConverter {
    /// Grating period Λ \[mm\].
    pub period_mm: f64,
    /// Grating length L \[mm\].
    pub length_mm: f64,
    /// Coupling coefficient κ \[1/m\] (field coupling, not power).
    pub coupling_coefficient: f64,
    /// Pair of mode indices being coupled, e.g. (0, 1) for LP01↔LP11.
    pub mode_pair: (usize, usize),
}

impl LpgModeConverter {
    /// Construct an LPG mode converter.
    ///
    /// Default coupling coefficient: κ ≈ 1–10 m⁻¹ for a silica FMF LPG.
    pub fn new(period_mm: f64, length_mm: f64, mode_pair: (usize, usize)) -> Self {
        // Typical κ for a UV-written LPG in silica
        let coupling_coefficient = PI / (2.0 * length_mm * 1.0e-3) * 0.9; // near-100% conversion
        Self {
            period_mm,
            length_mm,
            coupling_coefficient,
            mode_pair,
        }
    }

    /// Power coupling efficiency η(Δβ) as a function of phase detuning Δβ_eff \[rad/m\].
    ///
    ///   η = (κ/Ω)² · sin²(Ω·L)
    ///   Ω = sqrt(κ² + (Δβ_eff/2)²)
    ///   Δβ_eff = (β₁−β₂) − 2π/Λ
    pub fn efficiency(&self, delta_beta: f64) -> f64 {
        let kappa = self.coupling_coefficient;
        let l_m = self.length_mm * 1.0e-3;
        let detuning = delta_beta / 2.0; // half-detuning
        let omega = (kappa * kappa + detuning * detuning).sqrt();
        if omega < 1.0e-30 {
            return 0.0;
        }
        (kappa / omega).powi(2) * (omega * l_m).sin().powi(2)
    }

    /// Peak efficiency at exact resonance (Δβ_eff = 0):
    ///   η_max = sin²(κ·L)
    pub fn peak_efficiency(&self) -> f64 {
        let kappa = self.coupling_coefficient;
        let l_m = self.length_mm * 1.0e-3;
        (kappa * l_m).sin().powi(2)
    }

    /// 3-dB bandwidth of the conversion spectrum \[nm\].
    ///
    ///   Δλ_3dB ≈ 0.886 · λ² / (L · |dΔβ/dλ|)
    ///
    /// # Arguments
    /// - `dispersion_slope`: differential group delay slope dΔβ/dλ \[rad/m per m\] = \[rad/m²\]
    pub fn bandwidth_nm(&self, dispersion_slope: f64) -> f64 {
        let l_m = self.length_mm * 1.0e-3;
        if dispersion_slope.abs() < 1.0e-30 || l_m < 1.0e-30 {
            return 0.0;
        }
        // Numerator in metres, convert to nm
        let lambda_center = 1.55e-6_f64;
        let bw_m = 0.886 * lambda_center * lambda_center / (l_m * dispersion_slope.abs());
        bw_m * 1.0e9 // m → nm
    }

    /// Resonance wavelength from the phase-matching condition:
    ///   Λ = 2π / (β₁ − β₂)   →   λ_res = Λ · (β₁ − β₂) / (2π)  ← circular
    ///
    /// In practice, given β₁(λ) and β₂(λ), this is solved implicitly.
    /// Here we invert: λ\_res \\[m\\] = 2π·n\_g\_diff / (β₁ − β₂) using supplied values.
    ///
    /// # Arguments
    /// - `beta1`: propagation constant of mode 1 \[rad/m\]
    /// - `beta2`: propagation constant of mode 2 \[rad/m\]
    pub fn resonance_wavelength(&self, beta1: f64, beta2: f64) -> f64 {
        let delta_beta = (beta1 - beta2).abs();
        let lambda_m = self.period_mm * 1.0e-3 * delta_beta / (2.0 * PI);
        // lambda_m is wavelength in m if Λ·Δβ = 2π (exactly); more precisely:
        // resonance when grating k-vector = Δβ  →  λ = 2π / k_grating * n_eff_avg
        // Simpler: return grating-implied wavelength
        lambda_m * 1.0e9 // convert m → nm for output clarity, keep as m
        // Actually return in metres for consistency with the rest of the API
            * 1.0e-9
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_coupling_matrix_rows_sum_to_one() {
        let coupler = RandomModeCoupling::new(4, 1.0e-4, 0.5);
        let m = coupler.coupling_matrix(10.0);
        for row in &m {
            let row_sum: f64 = row.iter().sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1.0e-9);
        }
    }

    #[test]
    fn test_mixing_length_decreases_with_coupling() {
        let weak = RandomModeCoupling::new(4, 1.0e-5, 0.5);
        let strong = RandomModeCoupling::new(4, 1.0e-3, 0.5);
        assert!(strong.mixing_length_km() < weak.mixing_length_km());
    }

    #[test]
    fn test_effective_dgd_reduced_in_strong_coupling() {
        let coupler = RandomModeCoupling::new(4, 1.0e-3, 1.0);
        let l_mix = coupler.mixing_length_km();
        let fiber_len = 100.0 * l_mix; // strongly coupled
        let dgd_eff = coupler.effective_dgd_ps(1000.0, fiber_len);
        assert!(
            dgd_eff < 1000.0,
            "DGD should be reduced by coupling, got {}",
            dgd_eff
        );
    }

    #[test]
    fn test_mimo_apply_identity() {
        let eq = FmfMimoEqualizer::new(2, 5, 1.0e-4);
        // Identity initialisation: output of mode i ≈ received sample at center
        let received: Vec<Vec<Complex64>> = (0..2)
            .map(|i| vec![Complex64::new(i as f64 + 1.0, 0.0); 10])
            .collect();
        let output = eq.apply(&received);
        assert_eq!(output.len(), 2);
        // Mode 0 should pass through signal ≈ 1
        assert_abs_diff_eq!(output[0].re, 1.0, epsilon = 1.0e-9);
        assert_abs_diff_eq!(output[1].re, 2.0, epsilon = 1.0e-9);
    }

    #[test]
    fn test_required_taps_positive() {
        let n_taps = FmfMimoEqualizer::required_taps(1000.0, 32.0);
        assert!(n_taps >= 3, "Should need at least 3 taps");
    }

    #[test]
    fn test_convergence_symbols_finite() {
        let eq = FmfMimoEqualizer::new(4, 32, 1.0e-3);
        let conv = eq.convergence_symbols();
        assert!(conv > 0 && conv < usize::MAX);
    }

    #[test]
    fn test_photonic_lantern_efficiency() {
        let pl = PhotonicLantern::new(6);
        let eta = pl.efficiency();
        assert!(
            eta > 0.0 && eta <= 1.0,
            "Efficiency must be in (0,1]: {}",
            eta
        );
    }

    #[test]
    fn test_photonic_lantern_transfer_matrix_size() {
        let pl = PhotonicLantern::new(4);
        let t = pl.transfer_matrix();
        assert_eq!(t.len(), 4);
        assert_eq!(t[0].len(), 4);
    }

    #[test]
    fn test_lpg_peak_efficiency_range() {
        let lpg = LpgModeConverter::new(0.5, 30.0, (0, 1));
        let eta = lpg.peak_efficiency();
        assert!(
            (0.0..=1.0).contains(&eta),
            "Peak efficiency must be in [0,1]: {}",
            eta
        );
    }

    #[test]
    fn test_lpg_efficiency_at_resonance_equals_peak() {
        let lpg = LpgModeConverter::new(0.5, 30.0, (0, 1));
        let eta_res = lpg.efficiency(0.0);
        let eta_peak = lpg.peak_efficiency();
        assert_abs_diff_eq!(eta_res, eta_peak, epsilon = 1.0e-9);
    }

    #[test]
    fn test_lpg_efficiency_decreases_off_resonance() {
        let lpg = LpgModeConverter::new(0.5, 30.0, (0, 1));
        let eta_on = lpg.efficiency(0.0);
        let eta_off = lpg.efficiency(1000.0); // large detuning
        assert!(eta_off < eta_on, "Off-resonance efficiency must be lower");
    }
}
