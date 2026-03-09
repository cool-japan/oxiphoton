/// Optical frequency comb physics.
///
/// An optical frequency comb is a laser source whose spectrum consists of a
/// series of discrete, equally spaced frequency lines described by:
///
/// ```text
/// f_n = f_CEO + n · f_rep
/// ```
///
/// where `f_CEO` is the carrier-envelope offset frequency, `f_rep` is the
/// repetition rate, and `n` is an integer mode number.
///
/// Covers Ti:Sapphire mode-locked lasers, erbium-fiber combs, and Kerr
/// microresonator combs (dissipative Kerr solitons).
use crate::error::OxiPhotonError;

// ─── Physical constants ─────────────────────────────────────────────────────
/// Speed of light in vacuum (m/s).
pub const C0: f64 = 2.997_924_58e8;
/// Planck's constant (J·s).
const H_PLANCK: f64 = 6.626_070_15e-34;

// ─── FrequencyComb ──────────────────────────────────────────────────────────

/// Optical frequency comb with equally spaced spectral modes.
///
/// The mode frequencies follow `f_n = f_CEO + n · f_rep`. The struct captures
/// the key parameters that define the comb's temporal and spectral properties.
#[derive(Debug, Clone)]
pub struct FrequencyComb {
    /// Pulse repetition rate (Hz). Typical range: 100 MHz – 10 GHz.
    pub f_rep: f64,
    /// Carrier-envelope offset frequency (Hz). Must satisfy 0 ≤ f_CEO < f_rep.
    pub f_ceo: f64,
    /// Center wavelength (m).
    pub center_wavelength: f64,
    /// Full-width at half-maximum spectral bandwidth (nm).
    pub bandwidth_nm: f64,
    /// Peak power per pulse (W).
    pub peak_power_w: f64,
    /// Pulse duration, full-width at half-maximum (fs).
    pub pulse_duration_fs: f64,
}

impl FrequencyComb {
    /// Construct a generic frequency comb from explicit parameters.
    ///
    /// # Arguments
    /// * `f_rep`              — repetition rate (Hz)
    /// * `f_ceo`              — carrier-envelope offset frequency (Hz)
    /// * `center_wavelength`  — center wavelength (m)
    /// * `bandwidth_nm`       — FWHM spectral bandwidth (nm)
    /// * `peak_power_w`       — peak power per pulse (W)
    /// * `pulse_duration_fs`  — FWHM pulse duration (fs)
    pub fn new(
        f_rep: f64,
        f_ceo: f64,
        center_wavelength: f64,
        bandwidth_nm: f64,
        peak_power_w: f64,
        pulse_duration_fs: f64,
    ) -> Self {
        Self {
            f_rep,
            f_ceo,
            center_wavelength,
            bandwidth_nm,
            peak_power_w,
            pulse_duration_fs,
        }
    }

    /// Standard Ti:Sapphire frequency comb.
    ///
    /// Centered at 800 nm with 100 fs pulses and 10 nm FWHM bandwidth.
    /// Typical parameters for a Kerr-lens mode-locked Ti:Sa oscillator.
    ///
    /// # Arguments
    /// * `f_rep` — repetition rate (Hz), commonly 80–1000 MHz
    /// * `f_ceo` — carrier-envelope offset frequency (Hz)
    pub fn new_ti_sapphire(f_rep: f64, f_ceo: f64) -> Self {
        Self {
            f_rep,
            f_ceo,
            center_wavelength: 800e-9, // m
            bandwidth_nm: 10.0,        // nm FWHM → transform-limited ~30 fs at 800 nm
            peak_power_w: 100e3,       // 100 kW typical for 100 fs, 10 nJ pulse
            pulse_duration_fs: 100.0,  // fs FWHM
        }
    }

    /// Standard erbium-fiber frequency comb at 1550 nm.
    ///
    /// Represents a passively mode-locked Er:fiber oscillator common in
    /// telecom-band frequency metrology.
    ///
    /// # Arguments
    /// * `f_rep` — repetition rate (Hz), commonly 100–500 MHz
    /// * `f_ceo` — carrier-envelope offset frequency (Hz)
    pub fn new_erbium_fiber(f_rep: f64, f_ceo: f64) -> Self {
        Self {
            f_rep,
            f_ceo,
            center_wavelength: 1550e-9, // m
            bandwidth_nm: 8.0,          // nm FWHM
            peak_power_w: 50e3,         // 50 kW typical
            pulse_duration_fs: 150.0,   // fs FWHM
        }
    }

    /// Kerr microresonator frequency comb (dissipative Kerr soliton).
    ///
    /// Models a chip-integrated comb with THz-range repetition rate generated
    /// by parametric oscillation in a high-Q resonator.
    ///
    /// # Arguments
    /// * `f_rep_thz`      — repetition rate (THz)
    /// * `wavelength_nm`  — pump wavelength (nm)
    pub fn new_microresonator(f_rep_thz: f64, wavelength_nm: f64) -> Self {
        let f_rep = f_rep_thz * 1e12; // THz → Hz
                                      // Microcomb CEO is typically half the FSR (f_CEO = f_rep/2) for soliton state
        let f_ceo = f_rep * 0.5;
        Self {
            f_rep,
            f_ceo,
            center_wavelength: wavelength_nm * 1e-9,
            bandwidth_nm: 50.0,       // typical soliton bandwidth
            peak_power_w: 1.0,        // much lower than bulk lasers
            pulse_duration_fs: 100.0, // sub-ps soliton
        }
    }

    /// Frequency of the nth comb tooth: f_n = f_CEO + n · f_rep.
    ///
    /// # Arguments
    /// * `n` — integer mode number
    pub fn tooth_frequency(&self, n: i64) -> f64 {
        self.f_ceo + n as f64 * self.f_rep
    }

    /// Integer mode number of the comb tooth nearest to the center wavelength.
    ///
    /// Computed as n₀ = round((c/λ − f_CEO) / f_rep).
    pub fn center_mode_number(&self) -> i64 {
        let f_center = C0 / self.center_wavelength;
        ((f_center - self.f_ceo) / self.f_rep).round() as i64
    }

    /// Number of comb teeth within the FWHM spectral bandwidth.
    ///
    /// Converts the bandwidth from nm to Hz at the center wavelength and
    /// divides by f_rep.
    pub fn n_teeth(&self) -> usize {
        // Δf = (c / λ²) · Δλ  (in Hz, Δλ in m)
        let delta_lambda_m = self.bandwidth_nm * 1e-9;
        let delta_f_hz = C0 / (self.center_wavelength * self.center_wavelength) * delta_lambda_m;
        let n = (delta_f_hz / self.f_rep).round() as i64;
        n.max(1) as usize
    }

    /// Check whether the pulse is transform-limited (Gaussian pulse).
    ///
    /// For a Gaussian pulse the time-bandwidth product is TBP = 0.4413.
    /// A pulse is considered transform-limited when its actual TBP is within
    /// 10 % of the ideal value.
    pub fn is_transform_limited(&self) -> bool {
        // Spectral bandwidth in Hz
        let delta_lambda_m = self.bandwidth_nm * 1e-9;
        let delta_nu_hz = C0 / (self.center_wavelength * self.center_wavelength) * delta_lambda_m;
        // Pulse duration in seconds
        let tau_s = self.pulse_duration_fs * 1e-15;
        // Time-bandwidth product
        let tbp = delta_nu_hz * tau_s;
        // Transform-limited Gaussian: TBP = 0.4413
        (tbp - 0.4413).abs() < 0.044 // within 10 %
    }

    /// Average power: P_avg = P_peak · τ_pulse · f_rep.
    ///
    /// Uses the Gaussian pulse approximation where the pulse area factor is
    /// √(π / (4 ln 2)) ≈ 0.9394 · τ_FWHM.
    pub fn average_power_w(&self) -> f64 {
        // For a Gaussian pulse: P_avg = P_peak · τ_FWHM · f_rep · sqrt(π/(4 ln2))
        // The factor sqrt(π/(4 ln2)) ≈ 0.9394
        let gaussian_factor = (std::f64::consts::PI / (4.0 * 2_f64.ln())).sqrt();
        let tau_s = self.pulse_duration_fs * 1e-15;
        self.peak_power_w * tau_s * self.f_rep * gaussian_factor
    }

    /// Pulse energy (nJ): E = P_avg / f_rep.
    pub fn pulse_energy_nj(&self) -> f64 {
        self.average_power_w() / self.f_rep * 1e9 // J → nJ
    }

    /// CEO phase noise requirement for optical clock applications.
    ///
    /// For an optical clock, the CEO phase noise must be below 1 mrad/√Hz to
    /// avoid degrading the clock instability below the 10⁻¹⁸ level.
    /// Returns the requirement in rad/√Hz.
    pub fn ceo_phase_noise_requirement(&self) -> f64 {
        // 1 mrad/√Hz = 1e-3 rad/√Hz
        1e-3
    }

    /// Frequency uncertainty of the nth comb tooth propagated from
    /// uncertainties in f_rep and f_CEO.
    ///
    /// δf_n = √((n · δf_rep)² + δf_CEO²)
    ///
    /// # Arguments
    /// * `n`          — mode number
    /// * `delta_frep` — uncertainty in repetition rate (Hz)
    /// * `delta_fceo` — uncertainty in CEO frequency (Hz)
    pub fn tooth_frequency_uncertainty(&self, n: i64, delta_frep: f64, delta_fceo: f64) -> f64 {
        let term_rep = (n as f64 * delta_frep).powi(2);
        let term_ceo = delta_fceo.powi(2);
        (term_rep + term_ceo).sqrt()
    }

    /// Photon energy at the center wavelength (J).
    pub fn photon_energy_j(&self) -> f64 {
        H_PLANCK * C0 / self.center_wavelength
    }

    /// Estimate the spectral coverage (optical octave fraction).
    ///
    /// Returns the ratio of FWHM bandwidth to center frequency. A value ≥ 0.67
    /// (i.e., 2/3 octave) is needed for direct octave-spanning without external
    /// spectral broadening.
    pub fn octave_fraction(&self) -> f64 {
        let f_center = C0 / self.center_wavelength;
        let delta_lambda_m = self.bandwidth_nm * 1e-9;
        let delta_f = C0 / (self.center_wavelength * self.center_wavelength) * delta_lambda_m;
        delta_f / f_center
    }
}

// ─── CombState ──────────────────────────────────────────────────────────────

/// State of a Kerr microresonator frequency comb.
#[derive(Debug, Clone, PartialEq)]
pub enum CombState {
    /// Pump below threshold; no parametric oscillation.
    Off,
    /// Primary comb sidebands appear at multiples of the FSR.
    PrimaryComb,
    /// Sub-comb clusters form between primary teeth.
    SubComb,
    /// Dissipative Kerr soliton state with `n` intracavity solitons.
    Soliton(usize),
    /// Chaotic/noisy comb state.
    Chaos,
}

// ─── KerrMicrocomb ──────────────────────────────────────────────────────────

/// Kerr microresonator frequency comb (dissipative soliton model).
///
/// Describes a high-Q optical microresonator driven by a CW pump laser. Above
/// the parametric threshold, modulation-instability gain generates sidebands
/// that can evolve into a dissipative Kerr soliton (DKS) comb state.
///
/// The model uses the Lugiato-Lefever equation (LLE) phenomenology for
/// threshold and soliton-existence-range estimates.
#[derive(Debug, Clone)]
pub struct KerrMicrocomb {
    /// Resonator radius (μm).
    pub resonator_radius_um: f64,
    /// Effective refractive index at pump wavelength.
    pub n_eff: f64,
    /// Nonlinear refractive index n₂ (m²/W).
    pub n2: f64,
    /// Loaded quality factor of the resonator.
    pub q_factor: f64,
    /// Pump wavelength (m).
    pub wavelength: f64,
    /// On-chip pump power (mW).
    pub pump_power_mw: f64,
}

impl KerrMicrocomb {
    /// Construct a Kerr microcomb.
    ///
    /// # Arguments
    /// * `radius_um`  — resonator radius (μm)
    /// * `n_eff`      — effective refractive index
    /// * `q`          — loaded Q factor
    /// * `wavelength` — pump wavelength (m)
    pub fn new(radius_um: f64, n_eff: f64, q: f64, wavelength: f64) -> Self {
        Self {
            resonator_radius_um: radius_um,
            n_eff,
            n2: 2.4e-19, // m²/W — typical Si₃N₄
            q_factor: q,
            wavelength,
            pump_power_mw: 0.0,
        }
    }

    /// Silicon nitride microring at 1550 nm with typical parameters.
    ///
    /// Radius 100 μm, Q = 10⁶, n_eff = 1.9.
    pub fn si3n4_standard() -> Self {
        Self::new(100.0, 1.9, 1e6, 1550e-9)
    }

    /// Free spectral range (THz): FSR = c / (2π R n_g).
    ///
    /// Uses n_g ≈ n_eff (group index approximated as phase index for brevity).
    pub fn fsr_thz(&self) -> f64 {
        let radius_m = self.resonator_radius_um * 1e-6;
        let circumference = 2.0 * std::f64::consts::PI * radius_m;
        C0 / (circumference * self.n_eff) * 1e-12 // Hz → THz
    }

    /// Resonator linewidth (Hz): κ/2π = ω₀ / Q.
    pub fn linewidth_hz(&self) -> f64 {
        let omega0 = 2.0 * std::f64::consts::PI * C0 / self.wavelength;
        omega0 / (2.0 * std::f64::consts::PI * self.q_factor)
    }

    /// Parametric oscillation threshold power (mW).
    ///
    /// Based on the LLE threshold condition:
    /// P_th = (κ/2)² · n_eff · V_eff / (η_c · ω₀ · n₂ · c)
    ///
    /// Simplified form using the coupling efficiency η_c = 0.5 (critical coupling)
    /// and mode volume V_eff = π R · A_eff with A_eff ≈ 1 μm².
    pub fn threshold_power_mw(&self) -> f64 {
        let kappa_hz = self.linewidth_hz();
        let kappa = 2.0 * std::f64::consts::PI * kappa_hz; // rad/s
        let omega0 = 2.0 * std::f64::consts::PI * C0 / self.wavelength;
        let radius_m = self.resonator_radius_um * 1e-6;
        // Effective mode area A_eff ≈ 1 μm² for a typical waveguide
        let a_eff_m2 = 1e-12;
        // Mode volume V_eff = 2π R · A_eff
        let v_eff_m3 = 2.0 * std::f64::consts::PI * radius_m * a_eff_m2;
        // At critical coupling η_c = 1/2 → η_c² = 1/4
        let eta_c = 0.5_f64;
        // P_th = (κ/2)² · n_eff² · V_eff / (η_c · ω₀ · n₂ · c)
        let numerator = (kappa / 2.0).powi(2) * self.n_eff * self.n_eff * v_eff_m3;
        let denominator = eta_c * omega0 * self.n2 * C0;
        (numerator / denominator) * 1e3 // W → mW
    }

    /// Single-pass parametric gain coefficient g = g₀ · P_pump (1/m).
    ///
    /// g₀ = 2π n₂ / (λ A_eff) — the nonlinear coefficient γ.
    pub fn parametric_gain(&self, pump_power_mw: f64) -> f64 {
        let pump_w = pump_power_mw * 1e-3;
        let a_eff_m2 = 1e-12;
        // Nonlinear coefficient γ = 2π n₂ / (λ A_eff)  (1/(W·m))
        let gamma = 2.0 * std::f64::consts::PI * self.n2 / (self.wavelength * a_eff_m2);
        // Parametric gain = 2 γ P  (1/m) — the MI gain peak
        2.0 * gamma * pump_w
    }

    /// Classify the comb state based on pump power relative to threshold.
    ///
    /// The classification follows the experimentally observed sequence:
    /// 0 → Off, ~1× threshold → Primary comb, ~2× → SubComb,
    /// 3–5× → Soliton (transitions through chaos), >5× → Chaos.
    pub fn comb_state(&self, pump_power_mw: f64) -> CombState {
        let p_th = self.threshold_power_mw();
        if pump_power_mw < p_th {
            CombState::Off
        } else if pump_power_mw < 1.5 * p_th {
            CombState::PrimaryComb
        } else if pump_power_mw < 2.5 * p_th {
            CombState::SubComb
        } else if pump_power_mw < 6.0 * p_th {
            // Soliton existence window (depends on detuning sweep)
            let n_sol = self.n_solitons(pump_power_mw);
            CombState::Soliton(n_sol)
        } else {
            CombState::Chaos
        }
    }

    /// Soliton existence range as (lower, upper) detuning bounds in units of
    /// the cavity linewidth κ/2.
    ///
    /// From LLE theory, solitons exist for pump detuning δ satisfying:
    /// δ_min ≈ 1  and  δ_max ≈ π²/8 · (P/P_th)
    ///
    /// Returns (δ_min, δ_max) in units of κ/2.
    pub fn soliton_existence_range(&self) -> (f64, f64) {
        let p_th = self.threshold_power_mw();
        let p = self.pump_power_mw.max(p_th * 1.01);
        let delta_min = 1.0;
        // δ_max ≈ π²/8 · (P/P_th) from LLE soliton theory
        let delta_max = (std::f64::consts::PI * std::f64::consts::PI / 8.0) * (p / p_th);
        (delta_min, delta_max.max(delta_min + 0.1))
    }

    /// Estimated number of intracavity solitons for the given pump power.
    ///
    /// Uses the approximate relation N_sol ~ floor(P / (2 P_th)) clipped to [1, 8].
    pub fn n_solitons(&self, pump_power_mw: f64) -> usize {
        let p_th = self.threshold_power_mw();
        let n = ((pump_power_mw / (2.0 * p_th)).floor() as usize).max(1);
        n.min(8)
    }

    /// Convert the Kerr microcomb into a `FrequencyComb` representation.
    pub fn to_frequency_comb(&self) -> Result<FrequencyComb, OxiPhotonError> {
        let f_rep = self.fsr_thz() * 1e12; // THz → Hz
        if f_rep <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "FSR must be positive".into(),
            ));
        }
        Ok(FrequencyComb {
            f_rep,
            f_ceo: f_rep * 0.25, // typical
            center_wavelength: self.wavelength,
            bandwidth_nm: 50.0,
            peak_power_w: self.pump_power_mw * 1e-3,
            pulse_duration_fs: 100.0,
        })
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_tooth_frequency_formula() {
        let comb = FrequencyComb::new_ti_sapphire(100e6, 20e6);
        // f_0 = f_CEO = 20 MHz
        assert_abs_diff_eq!(comb.tooth_frequency(0), 20e6, epsilon = 1.0);
        // f_1 = 20 MHz + 100 MHz = 120 MHz
        assert_abs_diff_eq!(comb.tooth_frequency(1), 120e6, epsilon = 1.0);
        // f_n = f_CEO + n * f_rep for negative n
        let n: i64 = -5;
        let expected = 20e6 + n as f64 * 100e6;
        assert_abs_diff_eq!(comb.tooth_frequency(n), expected, epsilon = 1.0);
    }

    #[test]
    fn test_center_mode_number_ti_sa() {
        // Ti:Sa at 800 nm, f_rep = 100 MHz, f_CEO = 0
        let comb = FrequencyComb::new_ti_sapphire(100e6, 0.0);
        let n0 = comb.center_mode_number();
        // f_center = c/λ = 3e8/800e-9 ≈ 375 THz; n0 = 375e12/100e6 = 3_750_000
        let expected: i64 = (C0 / 800e-9 / 100e6).round() as i64;
        assert_eq!(n0, expected);
    }

    #[test]
    fn test_average_power_positive() {
        let comb = FrequencyComb::new_erbium_fiber(250e6, 30e6);
        let p_avg = comb.average_power_w();
        assert!(p_avg > 0.0, "average power must be positive: {p_avg}");
        // Sanity: P_avg < P_peak for any reasonable duty cycle
        assert!(p_avg < comb.peak_power_w);
    }

    #[test]
    fn test_pulse_energy_consistency() {
        let comb = FrequencyComb::new_erbium_fiber(250e6, 0.0);
        let e_nj = comb.pulse_energy_nj();
        // E = P_avg / f_rep; must be positive
        assert!(e_nj > 0.0, "pulse energy must be positive: {e_nj} nJ");
        // Round-trip: E * f_rep ≈ P_avg (in nJ * Hz = nW, × 1e-9 = W)
        let p_avg_from_e = e_nj * 1e-9 * comb.f_rep;
        assert_abs_diff_eq!(p_avg_from_e, comb.average_power_w(), epsilon = 1e-15);
    }

    #[test]
    fn test_tooth_uncertainty_propagation() {
        let comb = FrequencyComb::new_ti_sapphire(100e6, 20e6);
        let n: i64 = 3_750_000;
        let delta_frep = 1.0; // 1 Hz
        let delta_fceo = 10.0; // 10 Hz
        let uncertainty = comb.tooth_frequency_uncertainty(n, delta_frep, delta_fceo);
        let expected = ((n as f64 * delta_frep).powi(2) + delta_fceo.powi(2)).sqrt();
        assert_abs_diff_eq!(uncertainty, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_n_teeth_positive() {
        let comb = FrequencyComb::new_ti_sapphire(100e6, 0.0);
        let n = comb.n_teeth();
        assert!(n >= 1, "must have at least one tooth: {n}");
    }

    #[test]
    fn test_kerr_microcomb_fsr() {
        // R = 100 μm, n_eff = 1.9 → FSR ≈ c / (2π * 100e-6 * 1.9)
        let mc = KerrMicrocomb::si3n4_standard();
        let fsr = mc.fsr_thz();
        let expected = C0 / (2.0 * std::f64::consts::PI * 100e-6 * 1.9) * 1e-12;
        assert_abs_diff_eq!(fsr, expected, epsilon = 1e-3); // THz precision
    }

    #[test]
    fn test_kerr_comb_state_transitions() {
        let mc = KerrMicrocomb::si3n4_standard();
        let p_th = mc.threshold_power_mw();
        assert_eq!(mc.comb_state(p_th * 0.5), CombState::Off);
        assert_eq!(mc.comb_state(p_th * 1.2), CombState::PrimaryComb);
        assert_eq!(mc.comb_state(p_th * 2.0), CombState::SubComb);
        // At 4× threshold → Soliton state
        match mc.comb_state(p_th * 4.0) {
            CombState::Soliton(_) => {}
            other => panic!("expected Soliton, got {other:?}"),
        }
    }

    #[test]
    fn test_soliton_existence_range_ordering() {
        let mut mc = KerrMicrocomb::si3n4_standard();
        mc.pump_power_mw = mc.threshold_power_mw() * 4.0;
        let (lo, hi) = mc.soliton_existence_range();
        assert!(hi > lo, "soliton range must have hi > lo: lo={lo}, hi={hi}");
        assert!(lo >= 1.0, "lower bound must be ≥ 1 linewidth: {lo}");
    }

    #[test]
    fn test_to_frequency_comb() {
        let mc = KerrMicrocomb::si3n4_standard();
        let fc = mc.to_frequency_comb();
        assert!(fc.is_ok(), "conversion should succeed");
        let fc = fc.expect("already checked");
        assert!(fc.f_rep > 0.0);
        assert_abs_diff_eq!(fc.center_wavelength, 1550e-9, epsilon = 1e-12);
    }
}
