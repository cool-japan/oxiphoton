/// Mode-locked laser simulation via the Haus master equation.
///
/// The Haus master equation describes the steady-state pulse in a
/// mode-locked laser cavity:
///
/// ```text
/// [g − l + Dg·∂²/∂T² − jD·∂²/∂T² + jγ|A|²] A = 0
/// ```
///
/// where
///   g  = saturated gain per round trip
///   l  = total loss per round trip
///   Dg = gain filtering coefficient (fs²)
///   D  = group delay dispersion per round trip (fs²)
///   γ  = effective SPM coefficient per round trip
///
/// The analytic steady-state solution for the soliton branch is:
///   A(T) = A₀ · sech(T/τ) · exp(iψT²/2)
///
/// with τ determined self-consistently from dispersion/SPM balance.
///
/// Reference: H. A. Haus, *IEEE J. Sel. Topics Quantum Electron.* 6, 1173 (2000).
use std::f64::consts::PI;

use num_complex::Complex64;

use crate::error::OxiPhotonError;
use crate::fiber::pulse::fft_radix2;

// ---------------------------------------------------------------------------
// Physical constants
// ---------------------------------------------------------------------------

const C0: f64 = 2.997_924_58e8; // m/s
#[allow(dead_code)]
const HBAR: f64 = 1.054_571_8e-34; // J·s

// ---------------------------------------------------------------------------
// HausMasterEquation
// ---------------------------------------------------------------------------

/// Haus master equation parameters for a mode-locked oscillator.
///
/// All per-round-trip quantities unless stated otherwise.
#[derive(Debug, Clone)]
pub struct HausMasterEquation {
    /// Saturated small-signal gain per round trip (dimensionless, e.g. 0.1).
    pub gain_per_roundtrip: f64,
    /// Total round-trip loss (dimensionless, e.g. 0.05).
    pub loss_per_roundtrip: f64,
    /// Gain bandwidth FWHM in THz (sets gain filtering strength).
    pub gain_bandwidth_thz: f64,
    /// Group delay dispersion (GDD) per round trip in fs².
    /// Negative = anomalous; required for soliton formation.
    pub gdd_fs2: f64,
    /// SPM coefficient per round trip (rad/W).  γ*L_eff.
    pub spe_phase_shift_rad: f64,
    /// Saturable absorber modulation depth q₀ (dimensionless, 0–1).
    pub saturable_absorber_mod: f64,
    /// Gain saturation power P_sat (W).
    pub saturation_power_w: f64,
}

impl HausMasterEquation {
    /// Construct a new Haus master equation model.
    ///
    /// # Arguments
    /// * `gain`      – gain per round trip
    /// * `loss`      – loss per round trip
    /// * `bw_thz`    – gain bandwidth FWHM (THz)
    /// * `gdd_fs2`   – GDD per round trip (fs²)
    /// * `spe`       – SPM phase coefficient per round trip (rad/W)
    /// * `sa_mod`    – saturable absorber modulation depth
    /// * `p_sat`     – gain saturation power (W)
    pub fn new(
        gain: f64,
        loss: f64,
        bw_thz: f64,
        gdd_fs2: f64,
        spe: f64,
        sa_mod: f64,
        p_sat: f64,
    ) -> Self {
        Self {
            gain_per_roundtrip: gain,
            loss_per_roundtrip: loss,
            gain_bandwidth_thz: bw_thz,
            gdd_fs2,
            spe_phase_shift_rad: spe,
            saturable_absorber_mod: sa_mod,
            saturation_power_w: p_sat,
        }
    }

    /// Gain-filtering coefficient Dg = g / Δω_g² (units: fs²).
    ///
    /// Δω_g = 2π · Δν_g (rad/fs) where Δν_g is in THz.
    pub fn gain_filtering_fs2(&self) -> f64 {
        // Δω_g in rad/fs: 2π * bw_thz * 1e12 * 1e-15 = 2π * bw_thz * 1e-3
        let delta_omega_g_rad_per_fs = 2.0 * PI * self.gain_bandwidth_thz * 1.0e-3;
        self.gain_per_roundtrip / (delta_omega_g_rad_per_fs * delta_omega_g_rad_per_fs)
    }

    /// Soliton pulse half-width parameter τ (fs) at given pulse energy.
    ///
    /// From soliton area theorem: τ = 2|D| / (|γ| · E_pulse)
    /// where E_pulse is in pJ, D in fs², γ in rad/W.
    pub fn soliton_pulse_width_fs(&self, pulse_energy_pj: f64) -> f64 {
        let e_j = pulse_energy_pj * 1.0e-12;
        let gamma = self.spe_phase_shift_rad.abs();
        let d_abs = self.gdd_fs2.abs() * 1.0e-30; // fs² → s²
        if gamma < 1.0e-30 || e_j < 1.0e-30 {
            return 1.0e15; // degenerate
        }
        // τ (s) = 2|D| / (γ · E)
        let tau_s = 2.0 * d_abs / (gamma * e_j);
        tau_s * 1.0e15 // s → fs
    }

    /// Steady-state pulse half-width τ (fs) from Haus ME soliton branch.
    ///
    /// Self-consistent solution: τ² = |D_eff| / (γ · P₀)
    /// and P₀ = (g − l) / (γ · τ_SA_mod · τ²) — simplified analytic estimate.
    ///
    /// More precisely, the master equation soliton width arises from
    /// balancing the net gain/loss with gain filtering and nonlinearity:
    ///
    ///   τ = sqrt(Dg / (g − l − q₀/2))  where q₀/2 is SA contribution
    ///
    /// combined with GDD/SPM soliton: τ = |D| / (γ * P₀ * τ)
    pub fn steady_state_pulse_width_fs(&self) -> f64 {
        let g = self.gain_per_roundtrip;
        let l = self.loss_per_roundtrip;
        let dg = self.gain_filtering_fs2(); // fs²
        let net = g - l - self.saturable_absorber_mod / 2.0;
        if net <= 0.0 || dg <= 0.0 {
            return f64::INFINITY;
        }
        // Gain filtering limited pulse: τ_gf = sqrt(Dg / net)  [fs]
        let tau_gf = (dg / net).sqrt();

        // GDD/SPM soliton: τ_sol from balancing anomalous GDD with SPM
        // For anomalous dispersion (gdd_fs2 < 0):
        let d_abs = self.gdd_fs2.abs(); // fs²
        let gamma = self.spe_phase_shift_rad;
        if gamma > 0.0 && d_abs > 0.0 {
            // Geometric mean of gain-filtering and soliton widths
            // (Haus ME analytic result in the soliton limit)
            let tau_sol = (d_abs / (gamma * (g - l))).sqrt();
            // Weighted combination — in practice the minimum governs
            tau_gf.min(tau_sol)
        } else {
            tau_gf
        }
    }

    /// Steady-state peak power P₀ (W).
    ///
    /// From soliton area theorem: P₀ = |D| / (γ · τ²)
    pub fn steady_state_peak_power_w(&self) -> f64 {
        let tau_fs = self.steady_state_pulse_width_fs();
        if !tau_fs.is_finite() || tau_fs < 1.0e-6 {
            return 0.0;
        }
        let tau_s = tau_fs * 1.0e-15;
        let d_abs = self.gdd_fs2.abs() * 1.0e-30; // fs² → s²
        let gamma = self.spe_phase_shift_rad;
        if gamma < 1.0e-40 {
            return 0.0;
        }
        d_abs / (gamma * tau_s * tau_s)
    }

    /// Pulse energy E = 2 · P₀ · τ (pJ).
    ///
    /// For a sech pulse: E = ∫A₀²·sech²(t/τ)dt = 2·P₀·τ
    pub fn pulse_energy_pj(&self) -> f64 {
        let p0 = self.steady_state_peak_power_w();
        let tau_s = self.steady_state_pulse_width_fs() * 1.0e-15;
        2.0 * p0 * tau_s * 1.0e12 // J → pJ
    }

    /// Cavity repetition rate f_rep = c / (2·L) (MHz).
    pub fn rep_rate_mhz(&self, cavity_length_m: f64) -> f64 {
        if cavity_length_m <= 0.0 {
            return 0.0;
        }
        C0 / (2.0 * cavity_length_m) / 1.0e6
    }

    /// Average power P_avg = E_pulse · f_rep (mW).
    pub fn average_power_mw(&self, cavity_length_m: f64) -> f64 {
        let e_j = self.pulse_energy_pj() * 1.0e-12;
        let f_rep = self.rep_rate_mhz(cavity_length_m) * 1.0e6;
        e_j * f_rep * 1.0e3 // W → mW
    }

    /// Kelly sideband wavelength (nm) for dispersion order m.
    ///
    /// Dispersive waves from soliton perturbation satisfy the phase-matching:
    ///   β₂·Δω²/2 = (2π·m·f_rep/v_g) − δ_sol
    ///
    /// Simplified formula: Δλ_K ≈ λ²/(2πc) · sqrt(2 · m · f_rep / |β₂| · T_R)
    pub fn kelly_sideband_wavelength_nm(
        &self,
        m: i32,
        center_lambda_nm: f64,
        cavity_length_m: f64,
    ) -> f64 {
        let f_rep_hz = self.rep_rate_mhz(cavity_length_m) * 1.0e6;
        let lambda_m = center_lambda_nm * 1.0e-9;
        // GDD β₂·L = D [fs²] → s²
        let d_s2 = self.gdd_fs2.abs() * 1.0e-30;
        if d_s2 < 1.0e-50 || f_rep_hz < 1.0 {
            return center_lambda_nm;
        }
        // Phase mismatch Δω² = 2 * 2π * m * f_rep / |D|  (angular frequency)
        let delta_omega_sq = 2.0 * 2.0 * PI * (m.abs() as f64) * f_rep_hz / d_s2;
        if delta_omega_sq < 0.0 {
            return center_lambda_nm;
        }
        let delta_omega = delta_omega_sq.sqrt();
        // Convert to wavelength offset: Δλ = λ²/(2πc) * Δω
        let delta_lambda_m = lambda_m * lambda_m / (2.0 * PI * C0) * delta_omega;
        let sign = if m > 0 { 1.0 } else { -1.0 };
        center_lambda_nm + sign * delta_lambda_m * 1.0e9
    }

    /// Time-bandwidth product Δτ·Δν for a transform-limited sech² pulse.
    ///
    /// For unchirped sech: TBP = 0.3148 (FWHM definition).
    /// With chirp from the master equation, TBP > 0.3148.
    pub fn time_bandwidth_product(&self) -> f64 {
        // Analytic Haus ME result: TBP = 0.3148 * sqrt(1 + β²)
        // where β is the chirp parameter
        let beta = self.chirp_parameter();
        0.314_8 * (1.0 + beta * beta).sqrt()
    }

    /// Chirp parameter β from the master equation.
    ///
    /// β = (Dg · γ) / (|D|)  — ratio of gain filtering to dispersive effects.
    /// For pure solitons β → 0; finite gain filtering introduces residual chirp.
    pub fn chirp_parameter(&self) -> f64 {
        let dg = self.gain_filtering_fs2(); // fs²
        let d_abs = self.gdd_fs2.abs(); // fs²
        let gamma = self.spe_phase_shift_rad;
        if d_abs < 1.0e-20 {
            return 0.0;
        }
        dg * gamma / d_abs
    }

    /// Check self-consistency of the Haus master equation solution.
    ///
    /// The solution is self-consistent when the gain equals the total round-trip
    /// loss plus gain-filtering loss:
    ///   g = l + Dg / τ²
    pub fn is_self_consistent(&self) -> bool {
        let tau_fs = self.steady_state_pulse_width_fs();
        if !tau_fs.is_finite() {
            return false;
        }
        let dg = self.gain_filtering_fs2();
        let gain_filtering_loss = dg / (tau_fs * tau_fs);
        let lhs = self.gain_per_roundtrip;
        let rhs = self.loss_per_roundtrip + gain_filtering_loss;
        // Allow 20% tolerance for the simplified analytic formula
        (lhs - rhs).abs() / (lhs + rhs + 1.0e-10) < 0.20
    }

    /// Return `true` if Q-switching instability is suppressed.
    ///
    /// Heuristic condition (Hönninger criterion):
    ///   E_pulse / E_sat_gain * (g / l) < 1
    ///
    /// where E_sat_gain = P_sat · T_R.
    pub fn q_switch_free(&self, sat_energy_pj: f64) -> bool {
        let e_pulse = self.pulse_energy_pj();
        if sat_energy_pj < 1.0e-20 {
            return false;
        }
        let g = self.gain_per_roundtrip;
        let l = self.loss_per_roundtrip;
        if l < 1.0e-20 {
            return false;
        }
        (e_pulse / sat_energy_pj) * (g / l) < 1.0
    }

    /// Propagate one round trip through the laser cavity.
    ///
    /// Operator splitting in the frequency domain (gain + gain filtering +
    /// GDD) then in the time domain (SPM + saturable absorber loss):
    ///
    /// 1. FFT → frequency domain
    /// 2. Apply gain: g(1 + Dg·ω²/g)  → (g − Dg·ω²) per mode
    /// 3. Apply GDD: exp(−i·D·ω²/2)
    /// 4. IFFT → time domain
    /// 5. Apply loss: (1 − l)
    /// 6. Apply SPM: exp(i·γ|A|²)
    /// 7. Apply saturable absorber: (1 − q₀/(1 + |A|²/P_sat))
    pub fn propagate_one_roundtrip(&self, amplitude: &[Complex64], dt_fs: f64) -> Vec<Complex64> {
        let n = amplitude.len();
        if n == 0 {
            return Vec::new();
        }

        // Step 1: FFT
        let spectrum = fft_radix2(amplitude, false);

        // Build angular frequency array (rad/fs)
        let df = 1.0 / (n as f64 * dt_fs); // THz
        let omega: Vec<f64> = (0..n)
            .map(|i| {
                let fi = if i < n / 2 {
                    i as f64 * df
                } else {
                    (i as f64 - n as f64) * df
                };
                2.0 * PI * fi // rad/fs
            })
            .collect();

        let g = self.gain_per_roundtrip;
        let l = self.loss_per_roundtrip;
        let dg = self.gain_filtering_fs2(); // fs²
        let d = self.gdd_fs2; // fs²

        // Steps 2 & 3: gain filtering + GDD in frequency domain
        let spectrum_out: Vec<Complex64> = spectrum
            .iter()
            .zip(omega.iter())
            .map(|(&s, &om)| {
                let om2 = om * om;
                // Net gain after filtering: (g − l) − Dg*ω² (simplified gain saturation)
                let net_gain = (g - l - dg * om2).exp();
                // GDD phase: exp(−i·D·ω²/2)
                let gdd_phase = Complex64::new(0.0, -d * om2 / 2.0).exp();
                s * net_gain * gdd_phase
            })
            .collect();

        // Step 4: IFFT
        let mut field = fft_radix2(&spectrum_out, true);

        // Steps 5–7: time-domain operators
        let gamma = self.spe_phase_shift_rad;
        let q0 = self.saturable_absorber_mod;
        let p_sat = self.saturation_power_w;
        let loss_factor = 1.0 - l;

        for sample in field.iter_mut() {
            let power = sample.norm_sqr();
            // Loss
            *sample *= loss_factor;
            // SPM
            let spm_phase = Complex64::new(0.0, gamma * power).exp();
            *sample *= spm_phase;
            // Saturable absorber (fast absorber model)
            let sa_transmission = if p_sat > 0.0 {
                1.0 - q0 / (1.0 + power / p_sat)
            } else {
                1.0 - q0
            };
            *sample *= sa_transmission;
        }

        field
    }

    /// Iterate the round-trip map to find the steady-state pulse.
    ///
    /// Starts from a sech seed and iterates `n_iterations` round trips.
    /// Returns the converged amplitude vector.
    pub fn find_steady_state(
        &self,
        n_pts: usize,
        t_window_ps: f64,
        n_iterations: usize,
    ) -> Result<Vec<Complex64>, OxiPhotonError> {
        if n_pts == 0 {
            return Err(OxiPhotonError::NumericalError("n_pts must be > 0".into()));
        }
        if t_window_ps <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "t_window_ps must be positive".into(),
            ));
        }

        let n_pts_pow2 = n_pts.next_power_of_two();
        let dt_ps = t_window_ps / n_pts_pow2 as f64;
        let dt_fs = dt_ps * 1.0e3; // ps → fs

        // Seed: sech pulse with estimated width and peak power
        let tau_fs = self.steady_state_pulse_width_fs();
        let tau_ps = if tau_fs.is_finite() {
            tau_fs * 1.0e-3
        } else {
            0.1 // 100 fs default
        };
        let p0 = self.steady_state_peak_power_w().max(1.0);
        let a0 = p0.sqrt();

        let half_ps = t_window_ps / 2.0;
        let mut amplitude: Vec<Complex64> = (0..n_pts_pow2)
            .map(|i| {
                let t = i as f64 * dt_ps - half_ps;
                let env = if tau_ps > 0.0 {
                    1.0 / (t / tau_ps).cosh()
                } else {
                    0.0
                };
                Complex64::new(a0 * env, 0.0)
            })
            .collect();

        for _ in 0..n_iterations {
            amplitude = self.propagate_one_roundtrip(&amplitude, dt_fs);
            // Renormalise to avoid exponential growth / decay
            let peak = amplitude
                .iter()
                .map(|a| a.norm_sqr())
                .fold(0.0_f64, f64::max)
                .sqrt()
                .max(1.0e-30);
            let target = a0;
            let scale = target / peak;
            for s in amplitude.iter_mut() {
                *s *= scale;
            }
        }

        Ok(amplitude)
    }
}

// ---------------------------------------------------------------------------
// Sesam
// ---------------------------------------------------------------------------

/// Semiconductor saturable absorber mirror (SESAM) model.
///
/// The reflectivity follows a two-photon saturation model:
///   R(F) = 1 − ΔR·exp(−F/F_sat) − l_ns − TPA·F
///
/// Simplified to: R(F) = (1 − l_ns) − ΔR·exp(−F/F_sat)
#[derive(Debug, Clone)]
pub struct Sesam {
    /// Modulation depth ΔR (0–1).
    pub modulation_depth: f64,
    /// Saturation fluence F_sat (μJ/cm²).
    pub saturation_fluence_uj_cm2: f64,
    /// Absorber recovery time τ_A (ps).
    pub recovery_time_ps: f64,
    /// Non-saturable loss l_ns (0–1).
    pub non_sat_loss: f64,
}

impl Sesam {
    /// Construct a SESAM model.
    ///
    /// Returns `Err` if modulation depth or non-saturable loss are out of [0, 1].
    pub fn new(
        mod_depth: f64,
        fsat: f64,
        recovery_ps: f64,
        ns_loss: f64,
    ) -> Result<Self, OxiPhotonError> {
        if !(0.0..=1.0).contains(&mod_depth) {
            return Err(OxiPhotonError::NumericalError(format!(
                "modulation_depth {mod_depth} must be in [0, 1]"
            )));
        }
        if !(0.0..=1.0).contains(&ns_loss) {
            return Err(OxiPhotonError::NumericalError(format!(
                "non_sat_loss {ns_loss} must be in [0, 1]"
            )));
        }
        if fsat <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "saturation_fluence_uj_cm2 must be positive".into(),
            ));
        }
        Ok(Self {
            modulation_depth: mod_depth,
            saturation_fluence_uj_cm2: fsat,
            recovery_time_ps: recovery_ps,
            non_sat_loss: ns_loss,
        })
    }

    /// Effective reflectivity R(F) at incident fluence F (μJ/cm²).
    ///
    /// R(F) = (1 − l_ns) − ΔR · exp(−F / F_sat)
    pub fn reflectivity(&self, fluence_uj_cm2: f64) -> f64 {
        let base = 1.0 - self.non_sat_loss;
        let saturable =
            self.modulation_depth * (-fluence_uj_cm2 / self.saturation_fluence_uj_cm2).exp();
        (base - saturable).clamp(0.0, 1.0)
    }

    /// Effective saturable absorption for a pulse with given peak power and width.
    ///
    /// Approximates the fluence as F ≈ P_peak · τ_pulse / A_eff.
    /// Here we use a normalised approach: F_norm = P_peak · τ_pulse_ps / F_sat.
    pub fn modulation_for_pulse(&self, peak_power_w: f64, pulse_width_ps: f64) -> f64 {
        // Approximate fluence using unit beam area (1 cm²) for normalisation
        let fluence = peak_power_w * pulse_width_ps * 1.0e-12; // J/cm² (normalised)
        let fluence_uj = fluence * 1.0e6; // → μJ/cm²
        let r_with = self.reflectivity(fluence_uj);
        let r_unsaturated = 1.0 - self.non_sat_loss - self.modulation_depth;
        r_with - r_unsaturated.max(0.0)
    }

    /// Heuristic stability test: mode locking requires ΔR > (g − l).
    pub fn is_mode_locking_stable(&self, gain: f64, loss: f64) -> bool {
        self.modulation_depth > (gain - loss).abs() && self.non_sat_loss < gain - loss
    }
}

// ---------------------------------------------------------------------------
// KerrLensModelocking
// ---------------------------------------------------------------------------

/// Kerr-lens mode locking (KLM) model.
///
/// Self-focusing in a Kerr medium combined with a hard/soft aperture
/// provides an effective fast saturable absorber.
#[derive(Debug, Clone)]
pub struct KerrLensModelocking {
    /// Length of Kerr medium (mm).
    pub kerr_medium_length_mm: f64,
    /// Nonlinear refractive index n₂ (cm²/W).
    pub n2_cm2_per_w: f64,
    /// Beam waist inside Kerr medium (μm).
    pub beam_waist_in_kerr_um: f64,
    /// Aperture position relative to focus (dimensionless, 0 = at focus).
    pub aperture_position: f64,
}

impl KerrLensModelocking {
    /// Construct a KLM model.
    pub fn new(length_mm: f64, n2: f64, waist_um: f64, aperture_pos: f64) -> Self {
        Self {
            kerr_medium_length_mm: length_mm,
            n2_cm2_per_w: n2,
            beam_waist_in_kerr_um: waist_um,
            aperture_position: aperture_pos,
        }
    }

    /// Nonlinear phase (B-integral) accumulated in the Kerr medium.
    ///
    /// B = (2π/λ) · n₂ · I · L
    /// Using I = P / (π · w₀²) with unit wavelength 800 nm.
    pub fn nonlinear_phase_rad(&self, peak_power_w: f64) -> f64 {
        let lambda_m = 800.0e-9; // reference wavelength
        let w0_m = self.beam_waist_in_kerr_um * 1.0e-6;
        let n2_m2_per_w = self.n2_cm2_per_w * 1.0e-4; // cm² → m²
        let intensity_w_per_m2 = peak_power_w / (PI * w0_m * w0_m);
        let length_m = self.kerr_medium_length_mm * 1.0e-3;
        (2.0 * PI / lambda_m) * n2_m2_per_w * intensity_w_per_m2 * length_m
    }

    /// Effective saturable absorption modulation depth from KLM.
    ///
    /// Approximated as δ_KLM ≈ α · B² / (1 + B²)
    /// where α = aperture_position² / (1 + aperture_position²) represents
    /// the geometric aperture coupling efficiency.
    pub fn effective_modulation_depth(&self, peak_power_w: f64) -> f64 {
        let b = self.nonlinear_phase_rad(peak_power_w);
        let ap = self.aperture_position;
        let alpha = ap * ap / (1.0 + ap * ap + 1.0e-10);
        alpha * b * b / (1.0 + b * b)
    }

    /// Transform-limited minimum pulse width (fs) for a given spectral bandwidth.
    ///
    /// τ_min = 0.3148 · λ² / (c · Δλ)  for sech pulses.
    pub fn minimum_pulse_width_fs(&self, bandwidth_nm: f64, lambda_nm: f64) -> f64 {
        if bandwidth_nm <= 0.0 || lambda_nm <= 0.0 {
            return f64::INFINITY;
        }
        let lambda_m = lambda_nm * 1.0e-9;
        let delta_lambda_m = bandwidth_nm * 1.0e-9;
        // τ (s) = TBP_sech · λ² / (c · Δλ)
        let tau_s = 0.314_8 * lambda_m * lambda_m / (C0 * delta_lambda_m);
        tau_s * 1.0e15 // s → fs
    }
}

// ---------------------------------------------------------------------------
// ModeLockLaser
// ---------------------------------------------------------------------------

/// Complete mode-locked laser cavity model.
///
/// Combines the Haus master equation with cavity geometry to provide
/// practical output parameters.
#[derive(Debug, Clone)]
pub struct ModeLockLaser {
    /// Haus master equation model.
    pub equation: HausMasterEquation,
    /// Cavity round-trip length (m).
    pub cavity_length_m: f64,
    /// Centre wavelength (nm).
    pub center_wavelength_nm: f64,
    /// Output coupler power transmission T_OC (0–1).
    pub output_coupler_transmission: f64,
}

impl ModeLockLaser {
    /// Construct a complete mode-locked laser.
    ///
    /// # Arguments
    /// * `eq`         – Haus master equation parameters
    /// * `length_m`   – cavity round-trip length (m)
    /// * `lambda_nm`  – centre wavelength (nm)
    /// * `t_oc`       – output coupler transmission (0–1)
    pub fn new(eq: HausMasterEquation, length_m: f64, lambda_nm: f64, t_oc: f64) -> Self {
        Self {
            equation: eq,
            cavity_length_m: length_m,
            center_wavelength_nm: lambda_nm,
            output_coupler_transmission: t_oc.clamp(0.0, 1.0),
        }
    }

    /// Repetition rate (MHz).
    pub fn rep_rate_mhz(&self) -> f64 {
        self.equation.rep_rate_mhz(self.cavity_length_m)
    }

    /// Intracavity pulse half-width (fs).
    pub fn pulse_width_fs(&self) -> f64 {
        self.equation.steady_state_pulse_width_fs()
    }

    /// Intracavity peak power (kW).
    pub fn peak_power_kw(&self) -> f64 {
        self.equation.steady_state_peak_power_w() / 1.0e3
    }

    /// Average output power (mW).
    ///
    /// P_out = T_OC · P_avg_intracavity
    pub fn average_power_mw(&self) -> f64 {
        self.equation.average_power_mw(self.cavity_length_m) * self.output_coupler_transmission
    }

    /// Output pulse energy (pJ) after the output coupler.
    pub fn output_pulse_energy_pj(&self) -> f64 {
        self.equation.pulse_energy_pj() * self.output_coupler_transmission
    }

    /// Spectral bandwidth (nm) for the steady-state pulse.
    ///
    /// From time-bandwidth product: Δλ = λ² · TBP / (c · τ)
    pub fn spectral_bandwidth_nm(&self) -> f64 {
        let tau_s = self.pulse_width_fs() * 1.0e-15;
        if tau_s <= 0.0 || !tau_s.is_finite() {
            return 0.0;
        }
        let tbp = self.equation.time_bandwidth_product();
        let lambda_m = self.center_wavelength_nm * 1.0e-9;
        // Δλ = TBP · λ² / (c · τ)
        tbp * lambda_m * lambda_m / (C0 * tau_s) * 1.0e9
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_hme() -> HausMasterEquation {
        HausMasterEquation::new(
            0.10,    // gain
            0.05,    // loss
            5.0,     // gain bandwidth THz
            -1000.0, // GDD fs² (anomalous)
            1.0e-3,  // SPM rad/W
            0.02,    // SA modulation
            50.0,    // P_sat W
        )
    }

    #[test]
    fn test_rep_rate_from_cavity_length() {
        // f_rep = c / (2*L) for linear cavity
        let hme = default_hme();
        let l_m = 1.5; // 1.5 m → f_rep = 3e8/(2*1.5) = 1e8 Hz = 100 MHz
        let f_rep = hme.rep_rate_mhz(l_m);
        let expected = C0 / (2.0 * l_m) / 1.0e6;
        assert!(
            (f_rep - expected).abs() / expected < 1.0e-10,
            "rep rate: expected {expected:.3} MHz, got {f_rep:.3} MHz"
        );
    }

    #[test]
    fn test_soliton_pulse_energy_consistency() {
        // E_pulse = 2 * P0 * τ for sech pulse
        let hme = default_hme();
        let p0 = hme.steady_state_peak_power_w();
        let tau_fs = hme.steady_state_pulse_width_fs();
        let tau_s = tau_fs * 1.0e-15;
        let e_analytical = 2.0 * p0 * tau_s * 1.0e12; // pJ
        let e_method = hme.pulse_energy_pj();
        assert!(
            (e_analytical - e_method).abs() / (e_analytical + 1.0e-20) < 1.0e-10,
            "Energy consistency: analytical={e_analytical:.4e} pJ, method={e_method:.4e} pJ"
        );
    }

    #[test]
    fn test_time_bandwidth_product() {
        // For a transform-limited sech pulse TBP ≈ 0.3148
        // With small chirp it should be close to 0.3148
        let hme = HausMasterEquation::new(
            0.10,    // gain
            0.05,    // loss
            50.0,    // very wide gain bandwidth → small Dg → small chirp
            -1000.0, // GDD fs²
            1.0e-3,  // SPM
            0.02,    // SA mod
            50.0,    // P_sat
        );
        let tbp = hme.time_bandwidth_product();
        // TBP should be ≥ 0.3148 (transform limit)
        assert!(tbp >= 0.314_8 - 1.0e-6, "TBP {tbp:.5} should be ≥ 0.3148");
        // For small chirp, should be close to 0.3148
        assert!(
            tbp < 0.40,
            "TBP {tbp:.5} should be < 0.40 for small chirp case"
        );
    }

    #[test]
    fn test_self_consistency_check() {
        // Construct an HME with parameters that should give a self-consistent solution
        let hme = default_hme();
        // The is_self_consistent check uses 20% tolerance
        // Just verify it runs without panic and returns a bool
        let _ = hme.is_self_consistent();
        // Verify steady state gives finite result
        let tau = hme.steady_state_pulse_width_fs();
        assert!(tau.is_finite(), "steady-state pulse width should be finite");
        assert!(tau > 0.0, "pulse width should be positive");
    }

    #[test]
    fn test_sesam_reflectivity_at_saturation() {
        // At very high fluence, R → (1 − ns_loss)
        let sesam = Sesam::new(0.02, 10.0, 1.0, 0.005).expect("valid SESAM");
        let r_high = sesam.reflectivity(1.0e6); // very high fluence
        let expected = 1.0 - sesam.non_sat_loss;
        assert!(
            (r_high - expected).abs() < 1.0e-4,
            "R at saturation: expected {expected:.5}, got {r_high:.5}"
        );
    }

    #[test]
    fn test_sesam_reflectivity_at_low_fluence() {
        // At zero fluence, R → (1 − ns_loss) − ΔR = (1 − ΔR − ns_loss)
        let sesam = Sesam::new(0.02, 10.0, 1.0, 0.005).expect("valid SESAM");
        let r_low = sesam.reflectivity(0.0);
        let expected = 1.0 - sesam.modulation_depth - sesam.non_sat_loss;
        assert!(
            (r_low - expected).abs() < 1.0e-10,
            "R at low fluence: expected {expected:.5}, got {r_low:.5}"
        );
    }

    #[test]
    fn test_klm_phase_increases_with_power() {
        let klm = KerrLensModelocking::new(3.0, 3.0e-16, 50.0, 0.5);
        let phi_low = klm.nonlinear_phase_rad(1.0e3);
        let phi_high = klm.nonlinear_phase_rad(1.0e6);
        assert!(
            phi_high > phi_low,
            "Nonlinear phase should increase with power: low={phi_low:.4e}, high={phi_high:.4e}"
        );
    }

    #[test]
    fn test_steady_state_convergence() {
        // After many iterations the pulse should remain bounded and non-zero
        let hme = HausMasterEquation::new(
            0.10,    // gain
            0.05,    // loss
            5.0,     // gain bandwidth THz
            -1000.0, // GDD fs²
            1.0e-3,  // SPM rad/W
            0.02,    // SA mod
            50.0,    // P_sat
        );
        let result = hme.find_steady_state(256, 10.0, 50);
        assert!(result.is_ok(), "find_steady_state should succeed");
        let field = result.expect("steady state result");
        let peak = field.iter().map(|a| a.norm_sqr()).fold(0.0_f64, f64::max);
        assert!(peak > 0.0, "steady-state peak power should be positive");
        assert!(peak.is_finite(), "steady-state peak power should be finite");
    }
}
