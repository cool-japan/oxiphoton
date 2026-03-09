//! Erbium-Doped Fiber Amplifier (EDFA) rate equation model.
//!
//! Models gain, noise figure, ASE, and OSNR for C-band EDFAs based on
//! the two-level rate equation approximation. References:
//! - Desurvire, "Erbium-Doped Fiber Amplifiers", Wiley 2002
//! - Saleh & Teich, "Fundamentals of Photonics", 3rd ed., §15
use std::f64::consts::PI;

/// Planck constant (J·s)
const H_PLANCK: f64 = 6.626_070_15e-34;
/// Speed of light (m/s)
const C_LIGHT: f64 = 2.997_924_58e8;
/// Boltzmann constant (J/K)
const _K_B: f64 = 1.380_649e-23;
/// Upper-state lifetime of Er³⁺ ⁴I₁₃/₂ manifold (s)
const TAU_SP: f64 = 10e-3;
/// C-band peak emission wavelength (m)
const PEAK_EMISSION_WL: f64 = 1530e-9;
/// C-band emission bandwidth (half-width) (m)
const EMISSION_BW: f64 = 20e-9;

/// Pump direction configuration for optical amplifiers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PumpDirection {
    /// Pump co-propagates with signal (forward pumping).
    Copropagating,
    /// Pump counter-propagates with signal (backward pumping).
    Counterpropagating,
    /// Both forward and backward pumps; `forward_fraction` ∈ [0, 1].
    Bidirectional { forward_fraction: f64 },
}

/// Erbium-Doped Fiber Amplifier rate equation model.
///
/// Uses a homogeneous two-level approximation:
///   dN₂/dt = R_p·N₁ - (A₂₁ + W_e·I_s)·N₂ + W_a·I_s·N₁ = 0
/// giving inversion fraction n₂ = N₂/N_total as a function of pump
/// and signal intensities.
#[derive(Debug, Clone)]
pub struct Edfa {
    /// Active fiber length (m).
    pub fiber_length_m: f64,
    /// Er³⁺ doping concentration (m⁻³); typical 1 × 10²⁴ – 10²⁵.
    pub er_doping_concentration: f64,
    /// Mode-field diameter (μm).
    pub mode_field_diameter_um: f64,
    /// Pump wavelength (m); typically 980 nm or 1480 nm.
    pub pump_wavelength: f64,
    /// Launched pump power (mW).
    pub pump_power_mw: f64,
    /// Signal wavelength (m); C-band 1530–1565 nm.
    pub signal_wavelength: f64,
    /// Pump propagation direction.
    pub pump_direction: PumpDirection,
}

impl Edfa {
    /// Construct a standard C-band EDFA pumped at 980 nm.
    ///
    /// Default parameters: MFD = 3 μm, N_Er = 3 × 10²⁴ m⁻³, signal @ 1550 nm,
    /// copropagating pump.
    pub fn new_c_band_980nm(pump_power_mw: f64, length_m: f64) -> Self {
        Self {
            fiber_length_m: length_m,
            er_doping_concentration: 3.0e24,
            mode_field_diameter_um: 3.0,
            pump_wavelength: 980e-9,
            pump_power_mw,
            signal_wavelength: 1550e-9,
            pump_direction: PumpDirection::Copropagating,
        }
    }

    /// Construct a standard C-band EDFA pumped at 1480 nm.
    ///
    /// Default parameters: MFD = 4 μm, N_Er = 2 × 10²⁴ m⁻³, signal @ 1550 nm,
    /// counter-propagating pump.
    pub fn new_c_band_1480nm(pump_power_mw: f64, length_m: f64) -> Self {
        Self {
            fiber_length_m: length_m,
            er_doping_concentration: 2.0e24,
            mode_field_diameter_um: 4.0,
            pump_wavelength: 1480e-9,
            pump_power_mw,
            signal_wavelength: 1550e-9,
            pump_direction: PumpDirection::Counterpropagating,
        }
    }

    /// Effective mode area A_eff = π (MFD/2)² (m²).
    pub fn effective_area_m2(&self) -> f64 {
        let r = self.mode_field_diameter_um * 1e-6 / 2.0;
        PI * r * r
    }

    /// Emission cross-section σ_e(λ) in m² using a Gaussian spectral shape.
    ///
    /// Peak value σ_e0 ≈ 5 × 10⁻²⁵ m² at 1530 nm.
    pub fn emission_cross_section(&self, wavelength: f64) -> f64 {
        let sigma_e0 = 5.0e-25_f64; // peak m²
        let delta = wavelength - PEAK_EMISSION_WL;
        sigma_e0 * (-delta * delta / (2.0 * EMISSION_BW * EMISSION_BW)).exp()
    }

    /// Absorption cross-section σ_a(λ) in m².
    ///
    /// At 980 nm: σ_a ≈ 2.5 × 10⁻²⁵ m² (strong GSA).
    /// At 1480 nm: σ_a ≈ 3.5 × 10⁻²⁵ m² (ESA / GSA overlap).
    /// In the signal band (1530–1565 nm): Gaussian centred at 1530 nm.
    pub fn absorption_cross_section(&self, wavelength: f64) -> f64 {
        if (wavelength - 980e-9).abs() < 10e-9 {
            2.5e-25
        } else if (wavelength - 1480e-9).abs() < 15e-9 {
            3.5e-25
        } else {
            // Signal-band absorption: σ_a ≈ σ_e * exp(hν/(kT)) at RT → simplified
            let sigma_e = self.emission_cross_section(wavelength);
            // McCumber relation approximation: σ_a ≈ 0.7 * σ_e in C-band
            0.7 * sigma_e
        }
    }

    /// Pump absorption cross-section σ_ap (m²) at the pump wavelength.
    fn pump_absorption_cross_section(&self) -> f64 {
        self.absorption_cross_section(self.pump_wavelength)
    }

    /// Pump rate coefficient R_p (s⁻¹) per ion when all are in ground state.
    ///
    /// R_p = σ_ap · Φ_pump, where Φ_pump = P_pump / (A_eff · h·ν_pump)
    fn pump_rate(&self) -> f64 {
        let a_eff = self.effective_area_m2();
        let nu_pump = C_LIGHT / self.pump_wavelength;
        let phi_pump = (self.pump_power_mw * 1e-3) / (a_eff * H_PLANCK * nu_pump);
        self.pump_absorption_cross_section() * phi_pump
    }

    /// Upper-state population fraction n₂ = N₂/N_total at steady state.
    ///
    /// From rate equation with pump rate R_p, stimulated rates W_e, W_a and
    /// spontaneous rate A₂₁ = 1/τ_sp:
    ///   n₂ = R_p / (R_p + A₂₁ + W_e + W_a)
    /// (signal field assumed small-signal: W_e, W_a → 0 for unsaturated gain)
    pub fn inversion_parameter(&self) -> f64 {
        let r_p = self.pump_rate();
        let a21 = 1.0 / TAU_SP;
        // Small-signal (no saturation from signal)
        let n2 = r_p / (r_p + a21);
        // n_sp = N₂ / (N₂ - N₁) = n₂ / (2*n₂ - 1) for symmetric 2-level
        // Clamp to avoid division by zero or negative values
        n2.max(0.501)
    }

    /// Upper-state fraction n₂ ∈ (0, 1).
    fn n2_fraction(&self) -> f64 {
        let r_p = self.pump_rate();
        let a21 = 1.0 / TAU_SP;
        r_p / (r_p + a21)
    }

    /// Confinement factor Γ (overlap of optical mode with doped core).
    ///
    /// Approximated as Γ ≈ 1 - exp(-2 r_core² / w²) with r_core = MFD/4
    /// (doped region ≈ half the MFD).
    fn confinement_factor(&self) -> f64 {
        // For simplicity use Γ ≈ 0.6–0.8 depending on MFD
        let mfd = self.mode_field_diameter_um;
        // Larger MFD → smaller overlap
        (1.0 - (-2.0_f64 * (2.0 / mfd) * (2.0 / mfd)).exp()).clamp(0.3, 0.95)
    }

    /// Small-signal gain coefficient g₀ (1/m).
    ///
    /// g₀ = Γ · N_total · (σ_e · n₂ - σ_a · n₁)
    pub fn small_signal_gain_db_per_m(&self) -> f64 {
        let n2 = self.n2_fraction();
        let n1 = 1.0 - n2;
        let sigma_e = self.emission_cross_section(self.signal_wavelength);
        let sigma_a = self.absorption_cross_section(self.signal_wavelength);
        let gamma = self.confinement_factor();
        let g0_per_m = gamma * self.er_doping_concentration * (sigma_e * n2 - sigma_a * n1);
        g0_per_m * 10.0 / 2.0_f64.ln() // convert to dB/m: g[dB/m] = g[1/m] * 10/ln(10)
    }

    /// On-off small-signal gain G_ss = exp(g₀ · L) in dB.
    pub fn small_signal_gain_db(&self) -> f64 {
        self.small_signal_gain_db_per_m() * self.fiber_length_m
    }

    /// Saturation power P_sat (mW).
    ///
    /// P_sat = h·ν_s · A_eff / ((σ_e + σ_a) · τ_sp · Γ)
    pub fn saturation_power_mw(&self) -> f64 {
        let nu_s = C_LIGHT / self.signal_wavelength;
        let a_eff = self.effective_area_m2();
        let sigma_e = self.emission_cross_section(self.signal_wavelength);
        let sigma_a = self.absorption_cross_section(self.signal_wavelength);
        let gamma = self.confinement_factor();
        let p_sat_w = H_PLANCK * nu_s * a_eff / ((sigma_e + sigma_a) * TAU_SP * gamma);
        p_sat_w * 1e3
    }

    /// Gain at given input signal power (dB), accounting for gain compression.
    ///
    /// Uses the Saleh model: G(P_in) = G_ss / (1 + P_in/P_sat).
    /// This is an approximation; exact solution requires transcendental equation.
    pub fn gain_db(&self, signal_power_dbm: f64) -> f64 {
        let p_in_mw = 10.0_f64.powf(signal_power_dbm / 10.0);
        let p_sat = self.saturation_power_mw();
        let g_ss_linear = 10.0_f64.powf(self.small_signal_gain_db() / 10.0);
        let g_linear = g_ss_linear / (1.0 + p_in_mw / p_sat);
        10.0 * g_linear.log10()
    }

    /// Output signal power P_out = P_in · G(P_in) in dBm.
    pub fn output_power_dbm(&self, input_dbm: f64) -> f64 {
        input_dbm + self.gain_db(input_dbm)
    }

    /// Population inversion parameter n_sp = N₂ / (N₂ - N₁).
    ///
    /// n_sp = 1 for complete inversion, n_sp → ∞ as inversion → 0.5.
    pub fn nsp(&self) -> f64 {
        let n2 = self.n2_fraction();
        let n1 = 1.0 - n2;
        if n2 <= n1 {
            return 1e6; // no net inversion
        }
        n2 / (n2 - n1)
    }

    /// Noise figure NF ≈ 2·n_sp (quantum-limited: 3 dB for complete inversion).
    pub fn noise_figure_db(&self) -> f64 {
        let nsp = self.nsp();
        10.0 * (2.0 * nsp).log10()
    }

    /// ASE power (both polarisations) in an optical bandwidth B_o.
    ///
    /// P_ASE = 2 · n_sp · h·ν · (G - 1) · B_o
    /// where B_o (Hz) corresponds to `bandwidth_nm` around the signal wavelength.
    pub fn ase_power_dbm(&self, bandwidth_nm: f64) -> f64 {
        let nu = C_LIGHT / self.signal_wavelength;
        // Convert bandwidth from nm to Hz: ΔB_Hz ≈ c/λ² · Δλ
        let bw_hz =
            C_LIGHT / (self.signal_wavelength * self.signal_wavelength) * (bandwidth_nm * 1e-9);
        let g_ss = 10.0_f64.powf(self.small_signal_gain_db() / 10.0);
        let nsp = self.nsp();
        let p_ase_w = 2.0 * nsp * H_PLANCK * nu * (g_ss - 1.0) * bw_hz;
        10.0 * (p_ase_w * 1e3).log10()
    }

    /// Output OSNR (dB) for a given input signal power.
    ///
    /// OSNR = P_signal_out / P_ASE, measured in 0.1 nm reference bandwidth.
    pub fn output_osnr_db(&self, input_power_dbm: f64) -> f64 {
        let p_out_dbm = self.output_power_dbm(input_power_dbm);
        let p_ase_dbm = self.ase_power_dbm(0.1); // 0.1 nm reference BW
        p_out_dbm - p_ase_dbm
    }

    /// Normalised gain spectrum shape (0–1) at `wavelength_nm` across C-band.
    ///
    /// Uses a simplified double-Gaussian profile matching measured Er spectra.
    pub fn gain_spectrum(&self, wavelength_nm: f64) -> f64 {
        let lambda = wavelength_nm * 1e-9;
        // Two Gaussian peaks: 1530 nm and 1550 nm (typical Er emission)
        let peak1 = (-((lambda - 1530e-9).powi(2)) / (2.0 * (5e-9_f64).powi(2))).exp();
        let peak2 = 0.7 * (-((lambda - 1550e-9).powi(2)) / (2.0 * (8e-9_f64).powi(2))).exp();
        (peak1 + peak2).min(1.0)
    }

    /// Estimate optimal fiber length L_opt for maximum small-signal gain.
    ///
    /// L_opt minimises re-absorption past the gain peak; approximated as the
    /// length at which absorbed pump power equals that required for transparency.
    pub fn optimal_length_m(&self) -> f64 {
        let sigma_ap = self.pump_absorption_cross_section();
        let gamma = self.confinement_factor();
        let alpha_p = gamma * self.er_doping_concentration * sigma_ap; // pump absorption (1/m)
                                                                       // L_opt ≈ 1/α_p (1/e pump absorption length)
        if alpha_p > 0.0 {
            1.0 / alpha_p
        } else {
            self.fiber_length_m
        }
    }

    /// Output power for a gain-clamped EDFA (feedback laser fixes gain at `clamp_gain_db`).
    pub fn gain_clamped_output(&self, input_dbm: f64, clamp_gain_db: f64) -> f64 {
        // For a gain-clamped EDFA the gain is fixed regardless of input power
        // (until the input saturates the clamping laser itself).
        input_dbm + clamp_gain_db
    }
}

// ─── WDM channel ─────────────────────────────────────────────────────────────

/// A single WDM channel descriptor.
#[derive(Debug, Clone)]
pub struct WdmChannel {
    /// Centre wavelength (nm).
    pub wavelength_nm: f64,
    /// Channel launch power into the amplifier (dBm).
    pub power_dbm: f64,
}

impl WdmChannel {
    /// Construct a WDM channel.
    pub fn new(wavelength_nm: f64, power_dbm: f64) -> Self {
        Self {
            wavelength_nm,
            power_dbm,
        }
    }
}

/// Multi-channel EDFA for WDM amplification.
#[derive(Debug, Clone)]
pub struct WdmEdfa {
    /// Underlying EDFA model.
    pub edfa: Edfa,
    /// WDM channel list.
    pub channels: Vec<WdmChannel>,
}

impl WdmEdfa {
    /// Construct a WDM-EDFA.
    pub fn new(edfa: Edfa, channels: Vec<WdmChannel>) -> Self {
        Self { edfa, channels }
    }

    /// Per-channel gain (dB), accounting for gain spectral shape.
    ///
    /// The gain at each channel is scaled by the normalised gain spectrum
    /// relative to the EDFA's design signal wavelength.
    pub fn channel_gains_db(&self) -> Vec<f64> {
        let ref_gain = self.edfa.small_signal_gain_db();
        let ref_spectrum = self.edfa.gain_spectrum(self.edfa.signal_wavelength / 1e-9);
        self.channels
            .iter()
            .map(|ch| {
                let s = self.edfa.gain_spectrum(ch.wavelength_nm);
                let scale = if ref_spectrum > 1e-12 {
                    s / ref_spectrum
                } else {
                    1.0
                };
                ref_gain * scale
            })
            .collect()
    }

    /// Gain tilt (dB): gain difference between shortest and longest wavelength channel.
    pub fn gain_tilt_db(&self) -> f64 {
        if self.channels.len() < 2 {
            return 0.0;
        }
        let gains = self.channel_gains_db();
        let max_g = gains.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_g = gains.iter().cloned().fold(f64::INFINITY, f64::min);
        max_g - min_g
    }

    /// Total output power (dBm) by summing linear channel powers.
    pub fn total_output_power_dbm(&self) -> f64 {
        let gains = self.channel_gains_db();
        let total_mw: f64 = self
            .channels
            .iter()
            .zip(gains.iter())
            .map(|(ch, &g)| {
                let p_out_dbm = ch.power_dbm + g;
                10.0_f64.powf(p_out_dbm / 10.0)
            })
            .sum();
        if total_mw <= 0.0 {
            return f64::NEG_INFINITY;
        }
        10.0 * total_mw.log10()
    }

    /// Per-channel gain equalisation correction (dB) needed to achieve flat gain.
    ///
    /// Returns the negative correction (attenuation) each channel needs so that
    /// all channels have the same output power as the weakest channel.
    pub fn gain_equalization_needed_db(&self) -> Vec<f64> {
        let gains = self.channel_gains_db();
        let min_g = gains.iter().cloned().fold(f64::INFINITY, f64::min);
        gains.iter().map(|&g| -(g - min_g)).collect()
    }
}

// ─── EDFA cascade ────────────────────────────────────────────────────────────

/// Cascaded EDFA chain for long-haul transmission modelling.
#[derive(Debug, Clone)]
pub struct EdfaCascade {
    /// Amplifiers in the chain.
    pub amplifiers: Vec<Edfa>,
    /// Span losses (dB) between successive amplifiers.  Length = amplifiers.len() - 1,
    /// or len() == amplifiers.len() when loss precedes each amp (use case dependent).
    pub span_losses_db: Vec<f64>,
}

impl EdfaCascade {
    /// Build a uniform chain: `n_spans` identical spans, each with `span_loss_db`
    /// followed by a copy of `edfa`.
    pub fn new_uniform(n_spans: usize, span_loss_db: f64, edfa: Edfa) -> Self {
        let amplifiers = (0..n_spans).map(|_| edfa.clone()).collect();
        let span_losses_db = vec![span_loss_db; n_spans];
        Self {
            amplifiers,
            span_losses_db,
        }
    }

    /// Accumulated ASE noise power (dBm) after the full cascade, in `bw_nm`.
    ///
    /// Uses Friis chaining: P_ASE_total = Σ_n P_ASE_n · G_{n+1} · ... · G_N
    /// Simplified here as Σ P_ASE_n (assuming each amp re-amplifies to original level).
    pub fn accumulated_ase_dbm(&self, bw_nm: f64) -> f64 {
        // For a loss-compensated chain (G_amp = L_span):
        // P_ASE_total ≈ N · P_ASE_1
        let n = self.amplifiers.len() as f64;
        if n == 0.0 {
            return f64::NEG_INFINITY;
        }
        let p_ase_single = self.amplifiers[0].ase_power_dbm(bw_nm);
        // Linear sum: P_ase_total = N * P_ase_1
        let p_ase_total_mw = n * 10.0_f64.powf(p_ase_single / 10.0);
        10.0 * p_ase_total_mw.log10()
    }

    /// Total OSNR (dB) after `n_spans` amplifiers with `launch_power_dbm` per channel.
    ///
    /// Uses: 1/OSNR_total = Σ_n (1/OSNR_n)
    /// For identical spans: OSNR_total = OSNR_single / N
    pub fn total_osnr_db(&self, launch_power_dbm: f64) -> f64 {
        let n = self.amplifiers.len();
        if n == 0 {
            return f64::INFINITY;
        }
        // OSNR per span (each amp sees the launch power after loss compensation)
        let sum_inv_osnr: f64 = self
            .amplifiers
            .iter()
            .map(|amp| {
                let osnr_db = amp.output_osnr_db(launch_power_dbm);
                10.0_f64.powf(-osnr_db / 10.0)
            })
            .sum();
        if sum_inv_osnr <= 0.0 {
            return f64::INFINITY;
        }
        -10.0 * sum_inv_osnr.log10()
    }

    /// Maximum number of spans achievable while maintaining `target_osnr_db`.
    ///
    /// With identical spans: N_max = OSNR_single / OSNR_target (linear).
    pub fn max_spans(&self, target_osnr_db: f64, launch_power_dbm: f64) -> usize {
        if self.amplifiers.is_empty() {
            return 0;
        }
        let osnr_single_db = self.amplifiers[0].output_osnr_db(launch_power_dbm);
        let ratio = 10.0_f64.powf((osnr_single_db - target_osnr_db) / 10.0);
        ratio.floor().max(0.0) as usize
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_edfa_980nm_positive_gain() {
        let edfa = Edfa::new_c_band_980nm(100.0, 10.0);
        let gain = edfa.small_signal_gain_db();
        assert!(gain > 0.0, "EDFA should have positive gain: got {gain}");
    }

    #[test]
    fn test_edfa_noise_figure_above_3db() {
        let edfa = Edfa::new_c_band_980nm(100.0, 10.0);
        let nf = edfa.noise_figure_db();
        // Quantum limit is 3 dB; any physical EDFA is ≥ 3 dB
        assert!(nf >= 3.0, "NF must be ≥ 3 dB; got {nf}");
    }

    #[test]
    fn test_edfa_saturation_power_positive() {
        let edfa = Edfa::new_c_band_1480nm(200.0, 20.0);
        let p_sat = edfa.saturation_power_mw();
        assert!(
            p_sat > 0.0,
            "Saturation power must be positive; got {p_sat}"
        );
    }

    #[test]
    fn test_output_power_greater_than_input() {
        let edfa = Edfa::new_c_band_980nm(150.0, 15.0);
        let p_out = edfa.output_power_dbm(-20.0);
        assert!(p_out > -20.0, "Output power must exceed input; got {p_out}");
    }

    #[test]
    fn test_gain_compression_under_saturation() {
        let edfa = Edfa::new_c_band_980nm(50.0, 10.0);
        let g_small = edfa.gain_db(-30.0);
        let g_large = edfa.gain_db(0.0);
        assert!(
            g_small >= g_large,
            "Gain must decrease with input power; g(-30 dBm)={g_small}, g(0 dBm)={g_large}"
        );
    }

    #[test]
    fn test_ase_power_increases_with_bandwidth() {
        let edfa = Edfa::new_c_band_980nm(100.0, 10.0);
        let ase_01 = edfa.ase_power_dbm(0.1);
        let ase_1 = edfa.ase_power_dbm(1.0);
        assert!(ase_1 > ase_01, "ASE must increase with bandwidth");
    }

    #[test]
    fn test_gain_spectrum_normalised() {
        let edfa = Edfa::new_c_band_980nm(100.0, 10.0);
        for wl_nm in [1530.0, 1540.0, 1550.0, 1565.0] {
            let s = edfa.gain_spectrum(wl_nm);
            assert!(
                (0.0..=1.0).contains(&s),
                "Spectrum must be in [0,1] at {wl_nm} nm; got {s}"
            );
        }
    }

    #[test]
    fn test_wdm_edfa_gain_tilt() {
        let edfa = Edfa::new_c_band_980nm(100.0, 10.0);
        let channels = vec![
            WdmChannel::new(1530.0, -20.0),
            WdmChannel::new(1550.0, -20.0),
            WdmChannel::new(1565.0, -20.0),
        ];
        let wdm = WdmEdfa::new(edfa, channels);
        let tilt = wdm.gain_tilt_db();
        assert!(tilt >= 0.0, "Gain tilt must be non-negative; got {tilt}");
    }

    #[test]
    fn test_cascade_ase_accumulation() {
        let edfa = Edfa::new_c_band_980nm(100.0, 10.0);
        let cascade_1 = EdfaCascade::new_uniform(1, 10.0, edfa.clone());
        let cascade_4 = EdfaCascade::new_uniform(4, 10.0, edfa);
        let ase_1 = cascade_1.accumulated_ase_dbm(0.1);
        let ase_4 = cascade_4.accumulated_ase_dbm(0.1);
        // 4-span cascade should have ~6 dB more ASE
        assert!(
            ase_4 > ase_1,
            "Accumulated ASE must increase with number of spans; 1-span={ase_1}, 4-span={ase_4}"
        );
    }

    #[test]
    fn test_effective_area() {
        let edfa = Edfa::new_c_band_980nm(100.0, 10.0);
        let area = edfa.effective_area_m2();
        // MFD = 3 μm → A_eff = π*(1.5e-6)² ≈ 7.07e-12 m²
        assert_abs_diff_eq!(area, PI * (1.5e-6_f64).powi(2), epsilon = 1e-15);
    }
}
