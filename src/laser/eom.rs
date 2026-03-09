/// Electro-optic and acousto-optic modulators for OxiPhoton.
///
/// Provides:
/// * [`PockelsEom`] — Pockels-effect EOM (LiNbO₃ and general configurations)
/// * [`IqModulator`] — dual-parallel MZI for coherent (IQ) modulation
/// * [`AcoustomOpticModulator`] — acousto-optic modulator (AOM) model
///
/// # Physical background
///
/// The Pockels (linear electro-optic) effect in crystals such as LiNbO₃ causes
/// a refractive-index change Δn proportional to the applied electric field,
/// yielding a phase shift:
///
/// ```text
/// Δφ = π · V / Vπ
/// ```
///
/// For a Mach–Zehnder intensity modulator biased at quadrature:
/// ```text
/// T(V) = cos²(π·V / (2·Vπ) + π/4)
/// ```
use std::f64::consts::PI;

use num_complex::Complex64;

use crate::error::OxiPhotonError;

// ---------------------------------------------------------------------------
// EomType
// ---------------------------------------------------------------------------

/// Electro-optic modulator architecture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EomType {
    /// Simple phase modulator: A_out = A_in · exp(i·Δφ).
    Phase,
    /// Mach–Zehnder intensity modulator.
    Intensity,
    /// Dual-parallel MZI for IQ (complex) modulation.
    IQ,
}

// ---------------------------------------------------------------------------
// PockelsEom
// ---------------------------------------------------------------------------

/// Pockels-effect electro-optic modulator.
///
/// Models a travelling-wave LiNbO₃ or similar crystal modulator with a
/// specified half-wave voltage Vπ, insertion loss, and 3-dB electrical
/// bandwidth.  The bandwidth limitation is modelled by a first-order
/// (single-pole) frequency response.
#[derive(Debug, Clone)]
pub struct PockelsEom {
    /// Half-wave voltage Vπ (V) — voltage required for π phase shift.
    pub v_pi: f64,
    /// Insertion loss (dB).
    pub insertion_loss_db: f64,
    /// 3-dB electrical bandwidth (GHz).
    pub bandwidth_ghz: f64,
    /// Centre wavelength (nm).
    pub center_wavelength_nm: f64,
    /// Modulator architecture.
    pub modulator_type: EomType,
}

impl PockelsEom {
    /// Construct a Pockels EOM with specified parameters.
    pub fn new(v_pi: f64, loss_db: f64, bw_ghz: f64, lambda_nm: f64, eom_type: EomType) -> Self {
        Self {
            v_pi,
            insertion_loss_db: loss_db,
            bandwidth_ghz: bw_ghz,
            center_wavelength_nm: lambda_nm,
            modulator_type: eom_type,
        }
    }

    /// Canonical LiNbO₃ phase modulator at 1550 nm.
    ///
    /// Typical specifications: Vπ ≈ 3 V, insertion loss ≈ 3 dB.
    pub fn linbo3_phase(bw_ghz: f64) -> Self {
        Self::new(3.0, 3.0, bw_ghz, 1550.0, EomType::Phase)
    }

    /// Canonical LiNbO₃ intensity (MZI) modulator at 1550 nm.
    ///
    /// Typical: Vπ ≈ 3.5 V, insertion loss ≈ 4 dB.
    pub fn linbo3_intensity(bw_ghz: f64) -> Self {
        Self::new(3.5, 4.0, bw_ghz, 1550.0, EomType::Intensity)
    }

    /// Phase shift (rad) for applied DC voltage V: Δφ = π·V/Vπ.
    pub fn phase_shift_rad(&self, voltage_v: f64) -> f64 {
        if self.v_pi.abs() < 1.0e-15 {
            return 0.0;
        }
        PI * voltage_v / self.v_pi
    }

    /// Complex field transmission for a phase modulator.
    ///
    /// t = √(η) · exp(i·Δφ)
    /// where η = 10^(−loss_dB/10) is the insertion loss power factor.
    pub fn phase_transmission(&self, voltage_v: f64) -> Complex64 {
        let eta = 10.0_f64.powf(-self.insertion_loss_db / 10.0);
        let phi = self.phase_shift_rad(voltage_v);
        Complex64::new(0.0, phi).exp() * eta.sqrt()
    }

    /// Intensity transmission for a Mach–Zehnder EOM biased at quadrature.
    ///
    /// T(V) = η · cos²(π·V/(2·Vπ) + π/4)
    ///
    /// At V = 0 (quadrature): T = η · 0.5 (3 dB splitting).
    /// At V = −Vπ/2: T = η (fully on).
    /// At V = +Vπ/2: T = 0 (fully off).
    pub fn intensity_transmission(&self, voltage_v: f64) -> f64 {
        let eta = 10.0_f64.powf(-self.insertion_loss_db / 10.0);
        if self.v_pi.abs() < 1.0e-15 {
            return eta;
        }
        let arg = PI * voltage_v / (2.0 * self.v_pi) + PI / 4.0;
        eta * arg.cos() * arg.cos()
    }

    /// Optical extinction ratio (dB) for an ideal MZI intensity modulator.
    ///
    /// For an ideal MZI: T_max/T_min → ∞ (perfect null).
    /// In practice, finite extinction is modelled as:
    ///   ER (dB) = 10·log10(T_max / T_min)
    ///
    /// For an ideal modulator T_min = 0, so we return a representative
    /// large value of 40 dB.
    pub fn extinction_ratio_db(&self) -> f64 {
        40.0 // dB — ideal MZI
    }

    /// Chirp parameter α for a dual-drive push-pull MZI modulator.
    ///
    /// For an ideal push-pull MZI: α = 0 (chirp-free).
    /// For a single-drive modulator: α = 1.
    pub fn chirp_parameter(&self) -> f64 {
        match self.modulator_type {
            EomType::Intensity => 0.0, // ideal push-pull
            EomType::Phase => 1.0,     // pure phase → maximum chirp
            EomType::IQ => 0.0,
        }
    }

    /// Normalised frequency response |H(f)| at electrical frequency f (GHz).
    ///
    /// First-order (single-pole) Butterworth model:
    ///   |H(f)|² = 1 / (1 + (f / f_3dB)²)
    pub fn frequency_response(&self, freq_ghz: f64) -> f64 {
        if self.bandwidth_ghz <= 0.0 {
            return 0.0;
        }
        let x = freq_ghz / self.bandwidth_ghz;
        1.0 / (1.0 + x * x).sqrt()
    }

    /// Effective Vπ at electrical frequency f (GHz).
    ///
    /// Vπ(f) = Vπ(0) / |H(f)|
    pub fn vpi_at_frequency(&self, freq_ghz: f64) -> f64 {
        let h = self.frequency_response(freq_ghz);
        if h < 1.0e-15 {
            return f64::INFINITY;
        }
        self.v_pi / h
    }

    /// Modulation efficiency η = π / Vπ (rad/V).
    pub fn modulation_efficiency(&self) -> f64 {
        if self.v_pi.abs() < 1.0e-15 {
            return 0.0;
        }
        PI / self.v_pi
    }

    /// Apply phase modulation to an optical pulse.
    ///
    /// `amplitude` and `voltage_waveform` must have the same length.
    /// A_out(t) = A_in(t) · exp(i·Δφ(t)) · √η
    pub fn modulate_phase(
        &self,
        amplitude: &[Complex64],
        voltage_waveform: &[f64],
    ) -> Result<Vec<Complex64>, OxiPhotonError> {
        if amplitude.len() != voltage_waveform.len() {
            return Err(OxiPhotonError::NumericalError(format!(
                "amplitude length {} != voltage waveform length {}",
                amplitude.len(),
                voltage_waveform.len()
            )));
        }
        let eta = 10.0_f64.powf(-self.insertion_loss_db / 10.0);
        let amp_factor = eta.sqrt();
        let result = amplitude
            .iter()
            .zip(voltage_waveform.iter())
            .map(|(&a, &v)| {
                let phi = self.phase_shift_rad(v);
                a * Complex64::new(0.0, phi).exp() * amp_factor
            })
            .collect();
        Ok(result)
    }

    /// Apply intensity modulation to an optical pulse (MZI-based).
    ///
    /// A_out(t) = A_in(t) · √(T(V(t)))
    pub fn modulate_intensity(
        &self,
        amplitude: &[Complex64],
        voltage_waveform: &[f64],
    ) -> Result<Vec<Complex64>, OxiPhotonError> {
        if amplitude.len() != voltage_waveform.len() {
            return Err(OxiPhotonError::NumericalError(format!(
                "amplitude length {} != voltage waveform length {}",
                amplitude.len(),
                voltage_waveform.len()
            )));
        }
        let result = amplitude
            .iter()
            .zip(voltage_waveform.iter())
            .map(|(&a, &v)| {
                let t = self.intensity_transmission(v).max(0.0);
                a * t.sqrt()
            })
            .collect();
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// IqModulator
// ---------------------------------------------------------------------------

/// IQ (complex) optical modulator based on dual-parallel MZI architecture.
///
/// The in-phase (I) and quadrature (Q) arms provide access to the full
/// complex optical plane, enabling coherent modulation formats.
///
/// The optical field output is:
///   E_out = E_in / 2 · [cos(π·V_I/(2·Vπ)) + i·cos(π·V_Q/(2·Vπ))]
#[derive(Debug, Clone)]
pub struct IqModulator {
    /// In-phase MZI arm.
    pub eom_i: PockelsEom,
    /// Quadrature MZI arm.
    pub eom_q: PockelsEom,
    /// Phase shift between I and Q arms (ideally π/2).
    pub hybrid_phase: f64,
}

impl IqModulator {
    /// Construct a symmetric IQ modulator.
    ///
    /// Both I and Q arms share the same Vπ, loss and bandwidth.
    pub fn new(v_pi: f64, loss_db: f64, bw_ghz: f64, lambda_nm: f64) -> Self {
        Self {
            eom_i: PockelsEom::new(v_pi, loss_db, bw_ghz, lambda_nm, EomType::Intensity),
            eom_q: PockelsEom::new(v_pi, loss_db, bw_ghz, lambda_nm, EomType::Intensity),
            hybrid_phase: PI / 2.0,
        }
    }

    /// Generate the optical field for given I and Q drive voltages.
    ///
    /// E = (I_field + exp(i·π/2) · Q_field) / 2
    ///   = (I_field + i · Q_field) / 2
    pub fn generate_field(&self, vi: f64, vq: f64) -> Complex64 {
        let eta_i = 10.0_f64.powf(-self.eom_i.insertion_loss_db / 10.0);
        let eta_q = 10.0_f64.powf(-self.eom_q.insertion_loss_db / 10.0);
        // Each MZI arm: cos(π·V/(2·Vπ))
        let i_field = if self.eom_i.v_pi.abs() > 1.0e-15 {
            (PI * vi / (2.0 * self.eom_i.v_pi)).cos() * eta_i.sqrt()
        } else {
            0.0
        };
        let q_field = if self.eom_q.v_pi.abs() > 1.0e-15 {
            (PI * vq / (2.0 * self.eom_q.v_pi)).cos() * eta_q.sqrt()
        } else {
            0.0
        };
        let phase_q = Complex64::new(0.0, self.hybrid_phase).exp();
        (Complex64::new(i_field, 0.0) + phase_q * q_field) / 2.0
    }

    /// QPSK symbol constellation (4 symbols), normalised to ±Vπ/2.
    ///
    /// Returns (V_I, V_Q) drive voltage pairs for QPSK symbols.
    pub fn qpsk_symbols() -> Vec<(f64, f64)> {
        // Vπ = 1 (normalised): symbols at ±0.5
        let v = 0.5_f64;
        vec![(v, v), (-v, v), (-v, -v), (v, -v)]
    }

    /// 16-QAM symbol constellation (16 symbols), normalised voltage levels.
    ///
    /// Returns (V_I, V_Q) drive voltage pairs for 16-QAM symbols.
    pub fn qam16_symbols() -> Vec<(f64, f64)> {
        let levels = [-0.75_f64, -0.25, 0.25, 0.75];
        let mut symbols = Vec::with_capacity(16);
        for &i in &levels {
            for &q in &levels {
                symbols.push((i, q));
            }
        }
        symbols
    }

    /// Modulate a sequence of IQ symbol pairs into optical fields.
    pub fn modulate_symbols(&self, symbols: &[(f64, f64)]) -> Vec<Complex64> {
        symbols
            .iter()
            .map(|&(vi, vq)| self.generate_field(vi, vq))
            .collect()
    }

    /// Error vector magnitude (EVM) for an ideal QPSK constellation.
    ///
    /// For an ideal (lossless, perfectly balanced) IQ modulator, the EVM
    /// is determined by the amplitude imbalance between I and Q arms.
    /// Returns the RMS EVM as a fraction of the reference amplitude.
    pub fn evm_ideal_qpsk(&self) -> f64 {
        let symbols = Self::qpsk_symbols();
        let fields: Vec<Complex64> = self.modulate_symbols(&symbols);

        // Reference amplitude: mean magnitude of output symbols
        let ref_amp = fields.iter().map(|f| f.norm()).sum::<f64>() / fields.len() as f64;

        if ref_amp < 1.0e-20 {
            return 1.0;
        }

        // Ideal QPSK symbols are equally spaced on a circle
        let ideal_amp = ref_amp;
        let evm_sq = fields
            .iter()
            .map(|f| {
                let err = f.norm() - ideal_amp;
                err * err
            })
            .sum::<f64>()
            / fields.len() as f64;

        evm_sq.sqrt() / ref_amp
    }
}

// ---------------------------------------------------------------------------
// AcoustomOpticModulator
// ---------------------------------------------------------------------------

/// Acousto-optic modulator (AOM) model.
///
/// An AOM diffracts light using a travelling acoustic wave (phonon grating).
/// The first-order diffracted beam is frequency-shifted by ±f_drive.
///
/// Key parameters follow the Bragg diffraction regime:
///   sin(θ_B) = λ · f_acoustic / (2 · v_s)
///
/// where v_s is the acoustic velocity in the medium (TeO₂: ~4200 m/s).
#[derive(Debug, Clone)]
pub struct AcoustomOpticModulator {
    /// RF drive frequency (MHz) — sets the frequency shift of the diffracted order.
    pub center_frequency_mhz: f64,
    /// 3-dB bandwidth of the RF input (MHz).
    pub bandwidth_mhz: f64,
    /// Peak diffraction efficiency η (0–1, typically 0.7–0.9).
    pub diffraction_efficiency: f64,
    /// Acoustic transit time across the beam (ns) — sets the rise/fall time.
    pub rise_time_ns: f64,
    /// Operating wavelength (nm).
    pub wavelength_nm: f64,
}

impl AcoustomOpticModulator {
    /// Construct an AOM model.
    pub fn new(
        center_mhz: f64,
        bw_mhz: f64,
        efficiency: f64,
        rise_ns: f64,
        lambda_nm: f64,
    ) -> Self {
        Self {
            center_frequency_mhz: center_mhz,
            bandwidth_mhz: bw_mhz,
            diffraction_efficiency: efficiency.clamp(0.0, 1.0),
            rise_time_ns: rise_ns,
            wavelength_nm: lambda_nm,
        }
    }

    /// Frequency shift of the m-th diffraction order (Hz).
    ///
    /// Δf = m · f_drive
    /// where m = +1 (upshift) or m = −1 (downshift).
    pub fn frequency_shift_hz(&self, order: i32) -> f64 {
        order as f64 * self.center_frequency_mhz * 1.0e6
    }

    /// Bragg angle θ_B (rad) for the optical beam inside the medium.
    ///
    /// θ_B = arcsin(λ · f_acoustic / (2 · v_s))
    ///
    /// For TeO₂: v_s ≈ 4200 m/s (shear mode at 780 nm).
    pub fn bragg_angle_rad(&self, sound_velocity_m_per_s: f64) -> f64 {
        let lambda_m = self.wavelength_nm * 1.0e-9;
        let f_hz = self.center_frequency_mhz * 1.0e6;
        if sound_velocity_m_per_s < 1.0e-10 {
            return 0.0;
        }
        let arg = (lambda_m * f_hz / (2.0 * sound_velocity_m_per_s)).clamp(-1.0, 1.0);
        arg.asin()
    }

    /// Diffracted power for a given input power (W).
    ///
    /// P_out = η · P_in
    pub fn diffracted_power_w(&self, input_power_w: f64) -> f64 {
        self.diffraction_efficiency * input_power_w
    }

    /// Normalised frequency response |H(f)| at modulation frequency f (MHz).
    ///
    /// Combines:
    ///   1. Bandwidth limit (first-order): 1/sqrt(1+(f/BW)²)
    ///   2. Rise-time sinc rolloff: sinc(f·τ_rise) where τ = rise_time_ns
    pub fn frequency_response(&self, freq_mhz: f64) -> f64 {
        // Bandwidth (single-pole)
        let bw_term = if self.bandwidth_mhz > 0.0 {
            1.0 / (1.0 + (freq_mhz / self.bandwidth_mhz).powi(2)).sqrt()
        } else {
            0.0
        };
        // Rise-time sinc rolloff
        let tau_us = self.rise_time_ns * 1.0e-3; // ns → μs
        let ft = freq_mhz * tau_us; // MHz * μs = dimensionless
        let sinc_term = if ft.abs() < 1.0e-10 {
            1.0
        } else {
            (PI * ft).sin() / (PI * ft)
        };
        (bw_term * sinc_term.abs()).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // -----------------------------------------------------------------------
    // PockelsEom tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pockels_vpi_phase() {
        // At V = Vπ, the phase shift must equal π
        let eom = PockelsEom::linbo3_phase(10.0);
        let phi = eom.phase_shift_rad(eom.v_pi);
        assert_relative_eq!(phi, PI, max_relative = 1.0e-12);
    }

    #[test]
    fn test_phase_modulator_intensity_constant() {
        // Phase modulation must not change optical power (|t|² = η)
        let eom = PockelsEom::linbo3_phase(10.0);
        let eta = 10.0_f64.powf(-eom.insertion_loss_db / 10.0);
        for v in [-5.0, -2.0, 0.0, 1.5, 3.0, 6.0] {
            let t = eom.phase_transmission(v);
            let power = t.norm_sqr();
            assert!(
                (power - eta).abs() < 1.0e-10,
                "Phase mod power at V={v}: expected η={eta:.6}, got {power:.6}"
            );
        }
    }

    #[test]
    fn test_intensity_mod_at_zero() {
        // At V=0, quadrature-biased MZI: T = η * cos²(π/4) = η/2
        let eom = PockelsEom::linbo3_intensity(10.0);
        let t = eom.intensity_transmission(0.0);
        let eta = 10.0_f64.powf(-eom.insertion_loss_db / 10.0);
        let expected = eta * (PI / 4.0).cos().powi(2); // = η/2
        assert!(
            (t - expected).abs() < 1.0e-10,
            "Intensity at V=0: expected {expected:.6}, got {t:.6}"
        );
    }

    #[test]
    fn test_intensity_mod_at_vpi() {
        // At V = Vπ: arg = π/2 + π/4 = 3π/4, cos(3π/4)² = 0.5 → small
        // At V = -Vπ/2: arg = -π/4 + π/4 = 0 → cos(0)² = 1 → maximum
        // At V = +Vπ/2: arg = π/4 + π/4 = π/2 → cos(π/2)² = 0 → off
        let eom = PockelsEom::linbo3_intensity(10.0);
        let t_off = eom.intensity_transmission(eom.v_pi / 2.0);
        // Off state: cos²(π/4 + π/4) = cos²(π/2) = 0
        assert!(
            t_off < 1.0e-10,
            "Off-state transmission should be ≈0, got {t_off:.6e}"
        );
        // On state at -Vπ/2
        let t_on = eom.intensity_transmission(-eom.v_pi / 2.0);
        let eta = 10.0_f64.powf(-eom.insertion_loss_db / 10.0);
        assert!(
            (t_on - eta).abs() < 1.0e-10,
            "On-state transmission: expected η={eta:.4}, got {t_on:.4}"
        );
    }

    #[test]
    fn test_frequency_response_at_dc() {
        // H(0) = 1 (normalised to DC)
        let eom = PockelsEom::linbo3_phase(10.0);
        let h0 = eom.frequency_response(0.0);
        assert_relative_eq!(h0, 1.0, max_relative = 1.0e-12);
    }

    #[test]
    fn test_frequency_response_at_3db() {
        // H(BW) = 1/sqrt(2) ≈ 0.7071 for single-pole model
        let bw = 20.0;
        let eom = PockelsEom::linbo3_phase(bw);
        let h_at_bw = eom.frequency_response(bw);
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!(
            (h_at_bw - expected).abs() < 1.0e-10,
            "H(BW) should be 1/√2 ≈ {expected:.6}, got {h_at_bw:.6}"
        );
    }

    // -----------------------------------------------------------------------
    // IqModulator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_iq_qpsk_symbols() {
        // There must be exactly 4 QPSK symbols
        let syms = IqModulator::qpsk_symbols();
        assert_eq!(syms.len(), 4, "QPSK must have 4 symbols");
        // All symbols should have equal magnitude voltage
        let mag0 = (syms[0].0.powi(2) + syms[0].1.powi(2)).sqrt();
        for &(vi, vq) in &syms {
            let mag = (vi.powi(2) + vq.powi(2)).sqrt();
            assert!(
                (mag - mag0).abs() < 1.0e-10,
                "QPSK symbols should have equal magnitude: {mag:.6} vs {mag0:.6}"
            );
        }
    }

    #[test]
    fn test_iq_16qam_symbols() {
        // 16-QAM must have exactly 16 symbols
        let syms = IqModulator::qam16_symbols();
        assert_eq!(syms.len(), 16, "16-QAM must have 16 symbols");
    }

    #[test]
    fn test_iq_modulate_symbols_output_length() {
        let iq = IqModulator::new(3.5, 4.0, 20.0, 1550.0);
        let syms = IqModulator::qpsk_symbols();
        let fields = iq.modulate_symbols(&syms);
        assert_eq!(
            fields.len(),
            syms.len(),
            "Output length must match input length"
        );
    }

    // -----------------------------------------------------------------------
    // AcoustomOpticModulator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_aom_frequency_shift() {
        // Frequency shift must equal ±f_drive
        let center_mhz = 80.0;
        let aom = AcoustomOpticModulator::new(center_mhz, 10.0, 0.85, 50.0, 532.0);
        let shift_p1 = aom.frequency_shift_hz(1);
        let shift_m1 = aom.frequency_shift_hz(-1);
        assert_relative_eq!(shift_p1, center_mhz * 1.0e6, max_relative = 1.0e-12);
        assert_relative_eq!(shift_m1, -center_mhz * 1.0e6, max_relative = 1.0e-12);
    }

    #[test]
    fn test_aom_bragg_angle_positive() {
        let aom = AcoustomOpticModulator::new(80.0, 10.0, 0.85, 50.0, 780.0);
        let theta = aom.bragg_angle_rad(4200.0);
        assert!(theta > 0.0, "Bragg angle must be positive, got {theta:.4e}");
        assert!(
            theta < PI / 2.0,
            "Bragg angle must be < π/2, got {theta:.4e}"
        );
    }

    #[test]
    fn test_aom_diffracted_power_scales_linearly() {
        let aom = AcoustomOpticModulator::new(80.0, 10.0, 0.85, 50.0, 1064.0);
        let p1 = aom.diffracted_power_w(1.0);
        let p2 = aom.diffracted_power_w(2.0);
        assert_relative_eq!(p2 / p1, 2.0, max_relative = 1.0e-12);
    }

    #[test]
    fn test_aom_frequency_response_at_dc() {
        // At f → 0 the sinc term → 1 and bandwidth term → 1
        let aom = AcoustomOpticModulator::new(80.0, 10.0, 0.85, 50.0, 1064.0);
        let h = aom.frequency_response(0.0);
        assert_relative_eq!(h, 1.0, max_relative = 1.0e-10);
    }

    #[test]
    fn test_phase_modulate_preserves_length() {
        let eom = PockelsEom::linbo3_phase(10.0);
        let n = 128;
        let amplitude: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.1).cos(), 0.0))
            .collect();
        let voltages: Vec<f64> = (0..n).map(|i| i as f64 * 0.05).collect();
        let out = eom
            .modulate_phase(&amplitude, &voltages)
            .expect("modulate_phase");
        assert_eq!(out.len(), n, "Output length must match input length");
    }

    #[test]
    fn test_intensity_modulate_preserves_length() {
        let eom = PockelsEom::linbo3_intensity(10.0);
        let n = 64;
        let amplitude: Vec<Complex64> = vec![Complex64::new(1.0, 0.0); n];
        let voltages: Vec<f64> = vec![0.0; n];
        let out = eom
            .modulate_intensity(&amplitude, &voltages)
            .expect("modulate_intensity");
        assert_eq!(out.len(), n);
    }
}
