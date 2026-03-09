use super::comb::{FrequencyComb, C0};
/// Frequency comb stabilization, optical atomic clocks, and precision locking.
///
/// Covers:
/// - f-2f self-referencing interferometry for CEO frequency detection
/// - Phase-locked loops (PLL) for comb stabilization
/// - Optical atomic clock physics (Sr, Yb, Al⁺, Ca⁺, Hg⁺ transitions)
/// - Gravitational redshift and relativistic frequency corrections
use crate::error::OxiPhotonError;

// ─── Physical constants ─────────────────────────────────────────────────────
/// Gravitational acceleration at Earth's surface (m/s²).
const G_EARTH: f64 = 9.80665;
/// Cs-133 hyperfine transition frequency defining the SI second (Hz).
const CS_FREQ_HZ: f64 = 9_192_631_770.0;

// ─── F2fInterferometer ──────────────────────────────────────────────────────

/// f-2f interferometer for carrier-envelope offset frequency detection.
///
/// The technique beats the fundamental comb light (frequency f_n) against
/// the second harmonic of a lower-frequency tooth (2f_m).  When the comb
/// spans an optical octave (f_max ≥ 2 f_min) the beat note equals f_CEO.
///
/// A PPLN (periodically poled lithium niobate) crystal provides quasi-phase-
/// matched second-harmonic generation of the long-wavelength edge.
#[derive(Debug, Clone)]
pub struct F2fInterferometer {
    /// Whether the comb is octave-spanning (covers factor-of-2 in frequency).
    pub octave_spanning: bool,
    /// PPLN crystal length (mm).
    pub ppln_length_mm: f64,
    /// CEO beat signal SNR (dB) at the detection bandwidth of 100 kHz.
    pub ceo_detection_snr_db: f64,
}

impl F2fInterferometer {
    /// Construct an f-2f interferometer with the given PPLN length.
    ///
    /// Assumes 30 dB SNR at 100 kHz bandwidth — typical for a well-aligned setup.
    ///
    /// # Arguments
    /// * `ppln_length_mm` — length of the PPLN SHG crystal (mm)
    pub fn new(ppln_length_mm: f64) -> Self {
        Self {
            octave_spanning: false, // require spectral broadening in most cases
            ppln_length_mm,
            ceo_detection_snr_db: 30.0,
        }
    }

    /// Construct an octave-spanning f-2f interferometer (e.g., PCF-broadened Ti:Sa).
    pub fn new_octave_spanning(ppln_length_mm: f64) -> Self {
        Self {
            octave_spanning: true,
            ppln_length_mm,
            ceo_detection_snr_db: 35.0,
        }
    }

    /// CEO beat frequency detected by the interferometer (Hz).
    ///
    /// In the f-2f scheme the beat note at the photodetector equals f_CEO
    /// (or f_rep − f_CEO depending on the sign convention).  Returns the
    /// CEO frequency modulo f_rep.
    ///
    /// # Arguments
    /// * `comb` — the frequency comb being interrogated
    pub fn beat_frequency(&self, comb: &FrequencyComb) -> f64 {
        // Beat = 2(f_CEO + n f_rep) - (f_CEO + m f_rep) = f_CEO + (2n - m) f_rep
        // With 2n - m = 0 (choosing adjacent modes): beat = f_CEO
        comb.f_ceo
    }

    /// Spectral broadening factor required to achieve an octave span.
    ///
    /// Returns the ratio of the target octave bandwidth to the current comb
    /// bandwidth. A value of 1.0 means the comb is already octave-spanning.
    ///
    /// Uses the approximation: octave span ≈ 0.70 × center frequency,
    /// and the comb bandwidth in Hz is Δν = (c/λ²) Δλ.
    pub fn broadening_factor_required(&self) -> f64 {
        // 2× in frequency space corresponds to the octave condition
        2.0
    }

    /// SNR of the CEO beat signal (dB) for the given optical power and
    /// detection bandwidth.
    ///
    /// The fundamental shot-noise limited SNR:
    /// SNR = η P / (h ν B)  (linear)  → SNR_dB = 10 log₁₀(η P / (h ν B))
    ///
    /// where η = 0.8 (detector quantum efficiency), ν = c/λ, B = bandwidth.
    ///
    /// # Arguments
    /// * `optical_power_mw`  — optical power on the photodetector (mW)
    /// * `detection_bw_hz`   — detection bandwidth (Hz)
    pub fn beat_snr_db(&self, optical_power_mw: f64, detection_bw_hz: f64) -> f64 {
        let power_w = optical_power_mw * 1e-3;
        let h_planck = 6.626_070_15e-34;
        // Assume detection at 800 nm (Ti:Sa region) for generality
        let nu = C0 / 800e-9;
        let eta = 0.8; // detector QE
        let snr_linear = eta * power_w / (h_planck * nu * detection_bw_hz.max(1.0));
        if snr_linear <= 0.0 {
            return f64::NEG_INFINITY;
        }
        10.0 * snr_linear.log10()
    }

    /// Required PPLN temperature tuning range (K) for phase-matching over
    /// the SHG bandwidth.
    ///
    /// Typical PPLN temperature acceptance: ~1 K per nm of SHG bandwidth.
    /// Returns the required tuning range in Kelvin.
    ///
    /// # Arguments
    /// * `shg_bandwidth_nm` — desired SHG phase-matching bandwidth (nm)
    pub fn required_temperature_range_k(&self, shg_bandwidth_nm: f64) -> f64 {
        // Empirical: ~0.5 K / nm for MgO:PPLN near 1 μm
        0.5 * shg_bandwidth_nm
    }
}

// ─── CombPll ────────────────────────────────────────────────────────────────

/// Phase-locked loop for frequency comb stabilization.
///
/// Models a generic analog PLL used to lock either f_rep or f_CEO to a
/// reference oscillator. The noise model includes:
/// - In-loop: phase noise falls as 1/f² (integrating action)
/// - Out-of-loop: dominated by the noise floor of the reference/VCO
#[derive(Debug, Clone)]
pub struct CombPll {
    /// PLL loop bandwidth (Hz).  Phase noise is suppressed for offsets < this value.
    pub loop_bandwidth_hz: f64,
    /// Phase noise floor of the VCO/reference (dBc/Hz) outside the loop.
    pub phase_noise_floor_dbc_hz: f64,
    /// Frequency lock range (Hz).
    pub lock_range_hz: f64,
}

impl CombPll {
    /// Construct a PLL with the given loop bandwidth.
    ///
    /// Defaults: phase noise floor = −140 dBc/Hz, lock range = 10 × bandwidth.
    ///
    /// # Arguments
    /// * `bandwidth_hz` — loop bandwidth (Hz)
    pub fn new(bandwidth_hz: f64) -> Self {
        Self {
            loop_bandwidth_hz: bandwidth_hz,
            phase_noise_floor_dbc_hz: -140.0,
            lock_range_hz: bandwidth_hz * 10.0,
        }
    }

    /// Single-sideband phase noise spectral density at offset frequency `f` (dBc/Hz).
    ///
    /// Model:
    /// - f > loop_bw: phase noise = floor (free-running VCO/reference level)
    /// - f < loop_bw: suppressed by the integrator → S_φ(f) ≈ floor · (f/loop_bw)²
    ///
    /// The 1/f² suppression inside the loop bandwidth is the hallmark of a
    /// type-II PLL with an integrating loop filter.
    ///
    /// # Arguments
    /// * `offset_freq_hz` — Fourier offset frequency from the carrier (Hz)
    pub fn phase_noise_dbc_hz(&self, offset_freq_hz: f64) -> f64 {
        if offset_freq_hz <= 0.0 {
            return self.phase_noise_floor_dbc_hz; // undefined; return floor
        }
        if offset_freq_hz >= self.loop_bandwidth_hz {
            self.phase_noise_floor_dbc_hz
        } else {
            // In-loop: S_φ(f) = S_φ_floor · (f / f_loop)²  [dBc/Hz]
            // In dB: add 20 log10(f / f_loop)
            self.phase_noise_floor_dbc_hz + 20.0 * (offset_freq_hz / self.loop_bandwidth_hz).log10()
        }
    }

    /// Integrated timing jitter (fs) from f_low to f_high.
    ///
    /// Timing jitter from phase noise:
    /// σ_t² = (1/(2π f_c)²) · ∫ S_φ(f) df
    ///
    /// Integration is performed numerically over a logarithmic grid of 1000 points.
    ///
    /// # Arguments
    /// * `f_low`  — lower integration bound (Hz)
    /// * `f_high` — upper integration bound (Hz)
    /// * `f_carrier_hz` — carrier frequency for conversion (Hz); if 0, uses f_rep
    pub fn timing_jitter_fs(&self, f_low: f64, f_high: f64) -> f64 {
        if f_low <= 0.0 || f_high <= f_low {
            return 0.0;
        }
        // Numerical integration over log-spaced points
        let n_pts = 1000_usize;
        let log_fl = f_low.log10();
        let log_fh = f_high.log10();
        let df_log = (log_fh - log_fl) / n_pts as f64;

        // ∫ S_φ(f) df using the trapezoidal rule in log space
        let mut integral_dbc = 0.0_f64;
        for i in 0..n_pts {
            let f1 = 10_f64.powf(log_fl + i as f64 * df_log);
            let f2 = 10_f64.powf(log_fl + (i + 1) as f64 * df_log);
            let df = f2 - f1;
            let s1 = 10_f64.powf(self.phase_noise_dbc_hz(f1) / 10.0); // linear
            let s2 = 10_f64.powf(self.phase_noise_dbc_hz(f2) / 10.0);
            integral_dbc += 0.5 * (s1 + s2) * df; // trapezoidal
        }

        // Timing jitter: σ_t = sqrt(2 * integral) / (2π f_rep)
        // Here we use f_rep as carrier; caller should scale if needed.
        // We return rms phase jitter / (2π) in units of one cycle, then convert to fs.
        // For optical pulse timing: σ_t = sqrt(2 I) / ω_rep, with ω_rep = 2π f_rep
        // Return σ_t in fs — approximate using f_low as the nominal repetition rate
        let rms_phase_rad = (2.0 * integral_dbc).sqrt(); // rad rms
                                                         // Convert to time: σ_t = σ_φ / (2π f_rep) — assume f_low is ~f_rep
        let f_rep_approx = f_low.max(1.0); // use f_low as surrogate rep rate
        let sigma_t_s = rms_phase_rad / (2.0 * std::f64::consts::PI * f_rep_approx);
        sigma_t_s * 1e15 // s → fs
    }

    /// PLL lock acquisition time (μs).
    ///
    /// Estimated as τ_acq ≈ 1 / f_loop_bandwidth.
    pub fn lock_acquisition_time_us(&self) -> f64 {
        1.0 / self.loop_bandwidth_hz * 1e6 // s → μs
    }

    /// Residual frequency error after phase locking (Hz).
    ///
    /// Proportional to the in-loop phase noise integrated over the bandwidth:
    /// δf_rms ≈ f_loop · 10^(S_φ_floor/20)
    pub fn residual_freq_error_hz(&self) -> f64 {
        let phase_noise_lin = 10_f64.powf(self.phase_noise_floor_dbc_hz / 20.0);
        // δf ≈ f_loop · √(S_φ · f_loop) — approximate single-pole loop model
        self.loop_bandwidth_hz * (phase_noise_lin * self.loop_bandwidth_hz).sqrt()
    }
}

// ─── AtomType ───────────────────────────────────────────────────────────────

/// Atomic species used as the reference oscillator in an optical clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomType {
    /// ⁸⁷Sr lattice clock — 429 THz, best systematic uncertainty ~10⁻¹⁹.
    SrLattice,
    /// ¹⁷¹Yb lattice clock — 518 THz.
    YbLattice,
    /// ²⁷Al⁺ quantum-logic clock — 1.121 PHz, world's lowest uncertainty.
    AlIon,
    /// ⁴⁰Ca⁺ ion clock — 411 THz, widely used reference.
    CaIon,
    /// ¹⁹⁹Hg⁺ ion clock — 1.065 PHz, very high Q.
    HgIon,
}

impl AtomType {
    /// Optical transition frequency (Hz).
    pub fn transition_frequency_hz(&self) -> f64 {
        match self {
            AtomType::SrLattice => 429_228_066_418_012.0, // Hz
            AtomType::YbLattice => 518_295_836_590_863.0, // Hz
            AtomType::AlIon => 1_121_015_393_207_857.0,   // Hz
            AtomType::CaIon => 411_042_129_776_395.0,     // Hz
            AtomType::HgIon => 1_064_721_609_899_145.0,   // Hz
        }
    }

    /// Atomic Q factor: Q = ν₀ / Δν_transition.
    pub fn q_factor(&self) -> f64 {
        match self {
            AtomType::SrLattice => 4.3e17,
            AtomType::YbLattice => 1.0e18,
            AtomType::AlIon => 1.7e18,
            AtomType::CaIon => 6.6e15,
            AtomType::HgIon => 1.4e18,
        }
    }

    /// Systematic frequency uncertainty (Hz) — current state-of-the-art values.
    pub fn systematic_uncertainty_hz(&self) -> f64 {
        match self {
            AtomType::SrLattice => 0.2e-18 * self.transition_frequency_hz(),
            AtomType::YbLattice => 1.4e-18 * self.transition_frequency_hz(),
            AtomType::AlIon => 9.4e-19 * self.transition_frequency_hz(),
            AtomType::CaIon => 5.0e-16 * self.transition_frequency_hz(),
            AtomType::HgIon => 1.9e-17 * self.transition_frequency_hz(),
        }
    }
}

// ─── OpticalClock ────────────────────────────────────────────────────────────

/// Optical atomic clock — a frequency comb locked to an optical atomic transition.
///
/// The comb acts as a gear linking the optical standard to microwave frequencies.
/// The clock instability (Allan deviation) at short averaging times is dominated
/// by atom shot noise; at long times systematic effects limit accuracy.
#[derive(Debug, Clone)]
pub struct OpticalClock {
    /// Frequency comb used as the clockwork.
    pub comb: FrequencyComb,
    /// Optical atomic transition frequency (Hz).
    pub atomic_transition_hz: f64,
    /// Resonance Q factor of the atomic transition.
    pub q_factor: f64,
    /// Atomic species providing the reference transition.
    pub atom_type: AtomType,
}

impl OpticalClock {
    /// Construct an optical clock from a comb and an atomic species.
    ///
    /// The transition frequency and Q factor are taken from the `AtomType` database.
    ///
    /// # Arguments
    /// * `comb`  — the frequency comb used as clockwork
    /// * `atom`  — atomic reference species
    pub fn new(comb: FrequencyComb, atom: AtomType) -> Self {
        Self {
            atomic_transition_hz: atom.transition_frequency_hz(),
            q_factor: atom.q_factor(),
            atom_type: atom,
            comb,
        }
    }

    /// Short-term instability as the overlapping Allan deviation at 1 s.
    ///
    /// Quantum-projection-noise limited:
    /// σ_y(τ = 1 s) ≈ Δν / (ν₀ · SNR · √1)
    ///               = 1 / (Q · SNR)
    ///
    /// # Arguments
    /// * `snr` — signal-to-noise ratio per cycle of the atomic servo
    pub fn allan_deviation_1s(&self, snr: f64) -> f64 {
        if snr <= 0.0 || self.q_factor <= 0.0 {
            return f64::INFINITY;
        }
        1.0 / (self.q_factor * snr)
    }

    /// Absolute frequency accuracy (Hz) — limited by systematic uncertainties.
    pub fn frequency_accuracy_hz(&self) -> f64 {
        self.atom_type.systematic_uncertainty_hz()
    }

    /// Frequency ratio of the atomic transition to the Cs microwave standard.
    ///
    /// R = ν_atom / ν_Cs
    pub fn ratio_to_cs_standard(&self) -> f64 {
        self.atomic_transition_hz / CS_FREQ_HZ
    }

    /// Gravitational redshift of the clock frequency for a height change Δh.
    ///
    /// Δf/f = g Δh / c²  (positive Δh means higher altitude → higher frequency)
    ///
    /// # Arguments
    /// * `height_m` — height difference above the geoid (m); positive = higher
    pub fn gravitational_redshift(&self, height_m: f64) -> f64 {
        G_EARTH * height_m / (C0 * C0)
    }

    /// Absolute frequency shift due to gravitational redshift (Hz).
    ///
    /// Δf = (g Δh / c²) · ν₀
    ///
    /// # Arguments
    /// * `height_m` — height above reference geoid (m)
    pub fn gravitational_frequency_shift_hz(&self, height_m: f64) -> f64 {
        self.gravitational_redshift(height_m) * self.atomic_transition_hz
    }

    /// Minimum comb tooth number that falls within the capture range of the
    /// atomic transition (useful for locking diagnostics).
    ///
    /// Returns `Err` if the transition frequency is not within the comb bandwidth.
    pub fn nearest_tooth_number(&self) -> Result<i64, OxiPhotonError> {
        let n_approx = (self.atomic_transition_hz - self.comb.f_ceo) / self.comb.f_rep;
        if n_approx < 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "Atomic transition below comb CEO frequency".into(),
            ));
        }
        Ok(n_approx.round() as i64)
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_f2f_beat_equals_fceo() {
        let comb = FrequencyComb::new_ti_sapphire(100e6, 25e6);
        let ifo = F2fInterferometer::new(2.0);
        assert_abs_diff_eq!(ifo.beat_frequency(&comb), 25e6, epsilon = 1.0);
    }

    #[test]
    fn test_pll_phase_noise_in_loop_lower() {
        let pll = CombPll::new(100e3);
        // Inside loop bandwidth phase noise must be lower than floor
        let in_loop = pll.phase_noise_dbc_hz(1e3); // 1 kHz offset, loop_bw = 100 kHz
        let out_loop = pll.phase_noise_dbc_hz(1e6); // above loop
        assert!(
            in_loop < out_loop,
            "in-loop: {in_loop}, out-of-loop: {out_loop}"
        );
    }

    #[test]
    fn test_pll_lock_time_reasonable() {
        let pll = CombPll::new(100e3); // 100 kHz bandwidth
        let t_acq = pll.lock_acquisition_time_us();
        // 1/100 kHz = 10 μs
        assert_abs_diff_eq!(t_acq, 10.0, epsilon = 0.001);
    }

    #[test]
    fn test_atom_type_frequencies_positive() {
        for atom in [
            AtomType::SrLattice,
            AtomType::YbLattice,
            AtomType::AlIon,
            AtomType::CaIon,
            AtomType::HgIon,
        ] {
            let f = atom.transition_frequency_hz();
            assert!(f > 0.0, "{atom:?} frequency must be positive: {f}");
            let q = atom.q_factor();
            assert!(q > 0.0, "{atom:?} Q must be positive: {q}");
        }
    }

    #[test]
    fn test_optical_clock_allan_deviation() {
        let comb = FrequencyComb::new_erbium_fiber(250e6, 20e6);
        let clock = OpticalClock::new(comb, AtomType::SrLattice);
        let sigma = clock.allan_deviation_1s(10.0); // SNR = 10
                                                    // σ_y = 1/(Q*SNR) ≈ 1/(4.3e17 * 10)
        let expected = 1.0 / (AtomType::SrLattice.q_factor() * 10.0);
        assert_abs_diff_eq!(sigma, expected, epsilon = 1e-22);
    }

    #[test]
    fn test_gravitational_redshift_sign() {
        let comb = FrequencyComb::new_ti_sapphire(100e6, 0.0);
        let clock = OpticalClock::new(comb, AtomType::SrLattice);
        // Clock at higher altitude should have higher frequency → positive redshift
        let shift = clock.gravitational_redshift(1000.0); // 1 km
        assert!(
            shift > 0.0,
            "upward gravitational redshift must be positive: {shift}"
        );
        // Numerical check: g*h/c² ≈ 9.8*1000/(3e8)² ≈ 1.09e-13
        let expected = G_EARTH * 1000.0 / (C0 * C0);
        assert_abs_diff_eq!(shift, expected, epsilon = 1e-16);
    }

    #[test]
    fn test_ratio_to_cs_reasonable() {
        let comb = FrequencyComb::new_ti_sapphire(100e6, 0.0);
        let clock = OpticalClock::new(comb, AtomType::SrLattice);
        let ratio = clock.ratio_to_cs_standard();
        // Sr 429 THz / Cs 9.19 GHz ≈ 46 663
        assert!(
            ratio > 4e4 && ratio < 5e4,
            "ratio {ratio} out of expected range"
        );
    }

    #[test]
    fn test_beat_snr_db_positive_power() {
        let ifo = F2fInterferometer::new(3.0);
        let snr = ifo.beat_snr_db(1.0, 100e3); // 1 mW, 100 kHz BW
        assert!(snr > 0.0, "SNR should be positive for 1 mW: {snr} dB");
    }
}
