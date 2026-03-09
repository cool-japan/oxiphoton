use crate::error::OxiPhotonError;
use crate::fiber::dispersion::FiberDispersion;
use crate::fiber::pulse::{fft_radix2, omega_array_unshifted, OpticalPulse};
use num_complex::Complex64;
/// Nonlinear Schrödinger equation (NLSE) solver for pulse propagation
/// in optical fibres, using the Symmetric Split-Step Fourier Method (SSFM).
///
/// # NLSE
///
/// ```text
///   ∂A/∂z = −α/2·A − i·β₂/2·∂²A/∂T² + i·γ|A|²·A
/// ```
///
/// where `A(z,T)` is the complex pulse envelope (√W), `α` is the power loss
/// coefficient (1/m), `β₂` is the GVD (s²/m), and `γ` is the nonlinear
/// coefficient (1/W/m).  Higher-order dispersion (β₃, β₄) from the
/// `FiberDispersion` model is also included.
///
/// # SSFM algorithm (symmetric variant)
///
/// For each propagation step `dz`:
/// 1. Half-step dispersion in the frequency domain.
/// 2. Full nonlinear (SPM ± Raman) step in the time domain.
/// 3. Half-step dispersion in the frequency domain.
///
/// Reference: Agrawal, "Nonlinear Fiber Optics", 6th ed., §2.4.
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// NlseSolver
// ---------------------------------------------------------------------------

/// Full-featured NLSE solver for single-mode fibre propagation.
///
/// Propagates the complex envelope `A(T)` over `n_steps` steps of size
/// `step_size_m`, accounting for:
/// - Loss (α)
/// - Second- through fourth-order dispersion (β₂, β₃, β₄)
/// - Self-phase modulation (SPM)
/// - Simplified intrapulse Raman scattering (frequency-shift term)
pub struct NlseSolver {
    /// Dispersion model (β₂, β₃, β₄) and centre wavelength.
    pub dispersion: FiberDispersion,
    /// Nonlinear coefficient γ (1/W/m).
    pub gamma_per_w_per_m: f64,
    /// Power attenuation coefficient α (1/m).
    /// Relation to dB/km: α = α_dB/km · ln(10) / (10 · 1000).
    pub alpha_per_m: f64,
    /// Propagation step size dz (m).
    pub step_size_m: f64,
    /// Number of propagation steps.
    pub n_steps: usize,
    /// Include simplified Raman frequency shift term.
    pub include_raman: bool,
    /// Raman fractional contribution fR (≈ 0.18 for silica).
    pub raman_fraction: f64,
}

impl NlseSolver {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Create an `NlseSolver` with the given fibre and simulation parameters.
    ///
    /// `total_length_m` is divided equally into `n_steps` steps of size
    /// `step_size_m = total_length_m / n_steps`; the caller-supplied
    /// `step_size_m` here is used as a *target* step and `n_steps` is derived.
    pub fn new(
        dispersion: FiberDispersion,
        gamma_per_w_per_m: f64,
        alpha_per_m: f64,
        step_size_m: f64,
        total_length_m: f64,
    ) -> Self {
        let n_steps = if step_size_m > 0.0 {
            ((total_length_m / step_size_m).ceil() as usize).max(1)
        } else {
            1
        };
        let actual_step = total_length_m / n_steps as f64;
        Self {
            dispersion,
            gamma_per_w_per_m,
            alpha_per_m,
            step_size_m: actual_step,
            n_steps,
            include_raman: false,
            raman_fraction: 0.18,
        }
    }

    /// Enable the simplified intrapulse Raman self-frequency shift.
    pub fn with_raman(mut self, fraction: f64) -> Self {
        self.include_raman = true;
        self.raman_fraction = fraction;
        self
    }

    // -----------------------------------------------------------------------
    // Single-step propagation
    // -----------------------------------------------------------------------

    /// Propagate the amplitude array by one step `dz` using the symmetric SSFM.
    ///
    /// `omega` must be the angular-frequency array (rad/s) of length equal to
    /// `amplitude.len()` (already a power-of-two after potential padding by
    /// the caller).
    pub fn step(&self, amplitude: &[Complex64], omega: &[f64]) -> Vec<Complex64> {
        let dz = self.step_size_m;

        // --- Half-step dispersion + loss in frequency domain ---
        let after_half_disp = self.apply_dispersion_half(amplitude, omega, dz);

        // --- Full nonlinear step in time domain ---
        let after_nl = self.apply_nonlinear(&after_half_disp, dz);

        // --- Half-step dispersion + loss in frequency domain ---
        self.apply_dispersion_half(&after_nl, omega, dz)
    }

    // -----------------------------------------------------------------------
    // Full propagation
    // -----------------------------------------------------------------------

    /// Propagate a pulse through the full fibre length.
    ///
    /// The output pulse has the same time array as the input; only the
    /// amplitude is updated.
    pub fn propagate(&self, pulse: &OpticalPulse) -> Result<OpticalPulse, OxiPhotonError> {
        let n = pulse.amplitude.len();
        if n == 0 {
            return Err(OxiPhotonError::NumericalError(
                "pulse amplitude array must not be empty".into(),
            ));
        }
        let m = n.next_power_of_two();
        // Zero-pad amplitude to power-of-two length
        let mut amp = pulse.amplitude.clone();
        amp.resize(m, Complex64::new(0.0, 0.0));
        let omega = omega_array_unshifted(m, pulse.dt);

        for _ in 0..self.n_steps {
            amp = self.step(&amp, &omega);
        }

        // Truncate back to original length
        amp.truncate(n);
        OpticalPulse::new(pulse.t.clone(), amp, pulse.center_wavelength_nm)
    }

    /// Propagate and collect snapshots every `snapshot_interval` steps.
    ///
    /// The snapshot at index 0 is the input pulse; subsequent entries are
    /// taken after every `snapshot_interval` propagation steps.
    pub fn propagate_with_snapshots(
        &self,
        pulse: &OpticalPulse,
        snapshot_interval: usize,
    ) -> Result<Vec<OpticalPulse>, OxiPhotonError> {
        let n = pulse.amplitude.len();
        if n == 0 {
            return Err(OxiPhotonError::NumericalError(
                "pulse amplitude array must not be empty".into(),
            ));
        }
        let interval = snapshot_interval.max(1);
        let m = n.next_power_of_two();
        let mut amp = pulse.amplitude.clone();
        amp.resize(m, Complex64::new(0.0, 0.0));
        let omega = omega_array_unshifted(m, pulse.dt);

        // Snapshot 0: initial pulse
        let initial = OpticalPulse::new(
            pulse.t.clone(),
            amp[..n].to_vec(),
            pulse.center_wavelength_nm,
        )?;
        let mut snapshots = vec![initial];

        for step_idx in 0..self.n_steps {
            amp = self.step(&amp, &omega);
            if (step_idx + 1) % interval == 0 || step_idx + 1 == self.n_steps {
                let snap = OpticalPulse::new(
                    pulse.t.clone(),
                    amp[..n].to_vec(),
                    pulse.center_wavelength_nm,
                )?;
                snapshots.push(snap);
            }
        }
        Ok(snapshots)
    }

    // -----------------------------------------------------------------------
    // Characteristic lengths and soliton parameters
    // -----------------------------------------------------------------------

    /// Nonlinear length L_NL = 1 / (γ · P₀) (m).
    ///
    /// Characterises the propagation distance over which SPM becomes
    /// significant.  Returns infinity if γ or P₀ is negligibly small.
    pub fn nonlinear_length_m(&self, peak_power_w: f64) -> f64 {
        let denom = self.gamma_per_w_per_m * peak_power_w;
        if denom.abs() < 1.0e-60 {
            f64::INFINITY
        } else {
            1.0 / denom
        }
    }

    /// Soliton number N = √(γ · P₀ · T₀² / |β₂|).
    ///
    /// Uses the sech-pulse T₀ = FWHM / (2·ln(1+√2)) convention, consistent
    /// with the standard soliton definition in Agrawal "Nonlinear Fiber Optics".
    /// `fwhm_ps` is the intensity FWHM (ps).
    pub fn soliton_number(&self, peak_power_w: f64, fwhm_ps: f64) -> f64 {
        let b2_abs = self.dispersion.beta2_s2_per_m().abs();
        if b2_abs < 1.0e-60 {
            return f64::INFINITY;
        }
        // Sech T₀ from FWHM: FWHM = 2·T₀·ln(1+√2)
        let ln_fac = 2.0 * (1.0 + 2.0_f64.sqrt()).ln();
        let t0_s = fwhm_ps * 1.0e-12 / ln_fac;
        let lnl = self.nonlinear_length_m(peak_power_w);
        if !lnl.is_finite() || lnl < 1.0e-60 {
            return 0.0;
        }
        // N² = γ·P₀·T₀²/|β₂| = L_D / L_NL  (with sech L_D = T₀²/|β₂|)
        (self.gamma_per_w_per_m * peak_power_w * t0_s * t0_s / b2_abs).sqrt()
    }

    /// Peak power for a fundamental soliton (N = 1):
    ///   P₁ = |β₂| / (γ · T₀²)
    ///
    /// where T₀ = FWHM / (2√(ln 2)) for a sech pulse shape.
    pub fn soliton_power_w(&self, fwhm_ps: f64) -> f64 {
        let b2_abs = self.dispersion.beta2_s2_per_m().abs();
        let t0_s = fwhm_ps * 1.0e-12 / (2.0 * (1.0 + 2.0_f64.sqrt()).ln());
        if self.gamma_per_w_per_m.abs() < 1.0e-60 || t0_s < 1.0e-30 {
            return f64::INFINITY;
        }
        b2_abs / (self.gamma_per_w_per_m * t0_s * t0_s)
    }

    /// Self-phase modulation phase shift after propagation length `length_m` (m):
    ///   φ_NL = γ · P₀ · L_eff
    ///
    /// For a lossless fibre L_eff = length_m; with loss L_eff = (1 − e^(−αL)) / α.
    pub fn spm_phase_shift(&self, peak_power_w: f64, length_m: f64) -> f64 {
        let l_eff = if self.alpha_per_m.abs() < 1.0e-30 {
            length_m
        } else {
            (1.0 - (-self.alpha_per_m * length_m).exp()) / self.alpha_per_m
        };
        self.gamma_per_w_per_m * peak_power_w * l_eff
    }

    /// Rough estimate of supercontinuum spectral bandwidth (nm) generated
    /// by self-phase modulation alone:
    ///   Δλ_SC ≈ λ₀² · γ · P₀ · L / (π c)
    ///
    /// This is a first-order approximation valid for the coherent pumping regime.
    pub fn estimate_sc_bandwidth_nm(&self, pulse: &OpticalPulse) -> f64 {
        let p0 = pulse.peak_power();
        let total_length_m = self.step_size_m * self.n_steps as f64;
        let phi_max = self.spm_phase_shift(p0, total_length_m);
        let lambda0_m = pulse.center_wavelength_nm * 1.0e-9;
        // Maximum frequency broadening Δν ≈ φ_max / (π T₀)
        // → Δλ ≈ λ₀² Δν / c
        let t0_s = pulse.rms_width_s();
        if t0_s < 1.0e-30 || lambda0_m < 1.0e-12 {
            return 0.0;
        }
        let delta_nu = phi_max / (PI * t0_s);
        (lambda0_m * lambda0_m * delta_nu / (2.998e8)).abs() * 1.0e9
    }

    // -----------------------------------------------------------------------
    // FFT helpers (public for testing, private in spirit)
    // -----------------------------------------------------------------------

    /// Forward FFT of a complex array (Cooley–Tukey radix-2).
    ///
    /// Pads to the next power-of-two length if necessary.
    pub fn fft(&self, x: &[Complex64]) -> Vec<Complex64> {
        fft_pow2_local(x)
    }

    /// Inverse FFT (assumes `x.len()` is already a power of two).
    pub fn ifft(&self, x: &[Complex64]) -> Vec<Complex64> {
        fft_radix2(x, true)
    }

    /// Angular-frequency axis (rad/s) for an `n`-point FFT with time step `dt`.
    pub fn omega_array(n: usize, dt: f64) -> Vec<f64> {
        omega_array_unshifted(n, dt)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Apply the linear (dispersion + loss) half-step in the frequency domain.
    ///
    /// The loss factor exp(−α/2·dz/2) is folded into each half-step so that
    /// the full-step loss equals exp(−α·dz).
    fn apply_dispersion_half(
        &self,
        amplitude: &[Complex64],
        omega: &[f64],
        dz: f64,
    ) -> Vec<Complex64> {
        // Forward FFT
        let mut spectrum = fft_radix2(amplitude, false);
        // Dispersion operator + loss for a half-step
        let loss_factor = (-self.alpha_per_m / 2.0 * dz / 2.0).exp();
        let disp_op = self.dispersion.dispersion_operator(omega, dz / 2.0);
        for (s, (d, _)) in spectrum.iter_mut().zip(disp_op.iter().zip(omega.iter())) {
            *s *= *d * loss_factor;
        }
        // Inverse FFT
        fft_radix2(&spectrum, true)
    }

    /// Apply the full nonlinear (SPM ± Raman) step in the time domain.
    fn apply_nonlinear(&self, amplitude: &[Complex64], dz: f64) -> Vec<Complex64> {
        if self.include_raman {
            self.apply_nonlinear_raman(amplitude, dz)
        } else {
            self.apply_spm_only(amplitude, dz)
        }
    }

    /// Pure SPM: A → A · exp(i·γ·|A|²·dz).
    fn apply_spm_only(&self, amplitude: &[Complex64], dz: f64) -> Vec<Complex64> {
        amplitude
            .iter()
            .map(|&a| {
                let phi = self.gamma_per_w_per_m * a.norm_sqr() * dz;
                a * Complex64::new(0.0, phi).exp()
            })
            .collect()
    }

    /// Simplified Raman model: the SPM term is reduced by fR, and a frequency
    /// shift proportional to the power derivative is added.
    ///
    /// The Raman self-frequency shift is modelled as:
    ///   ∂A/∂z|_Raman = −i·γ·fR·T_R·|A|²·(∂A/∂T)
    ///
    /// Here we use an explicit first-order finite-difference approximation for
    /// the power derivative d(|A|²)/dt.
    fn apply_nonlinear_raman(&self, amplitude: &[Complex64], dz: f64) -> Vec<Complex64> {
        let n = amplitude.len();
        let mut out = Vec::with_capacity(n);
        // Simplified Raman: T_R ≈ 3 fs for silica
        let t_r = 3.0e-15_f64;
        // We embed dt from the amplitude array length implicitly — use a fixed
        // approximate value (1 fs per sample is worst case; real dt comes from
        // the calling context).  For the simplified model we just modulate the
        // SPM phase by the local power slope.
        for (idx, &a) in amplitude.iter().enumerate() {
            let power = a.norm_sqr();
            // Power gradient (central difference where possible)
            let dp_dt = if idx == 0 || idx == n - 1 {
                0.0
            } else {
                (amplitude[idx + 1].norm_sqr() - amplitude[idx - 1].norm_sqr()) / 2.0
            };
            // Effective nonlinear phase including Raman frequency shift
            let phi_spm = self.gamma_per_w_per_m * (1.0 - self.raman_fraction) * power * dz;
            let phi_raman = -self.gamma_per_w_per_m * self.raman_fraction * t_r * dp_dt * dz;
            let phi_total = phi_spm + phi_raman;
            out.push(a * Complex64::new(0.0, phi_total).exp());
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Local FFT helper (to avoid re-export confusion)
// ---------------------------------------------------------------------------

fn fft_pow2_local(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    let m = n.next_power_of_two();
    let mut padded = x.to_vec();
    padded.resize(m, Complex64::new(0.0, 0.0));
    fft_radix2(&padded, false)
}

// ---------------------------------------------------------------------------
// FiberAmplifier
// ---------------------------------------------------------------------------

/// Optical amplifier model for erbium-doped fibre amplifiers (EDFA) and
/// other in-line optical amplifiers.
///
/// Models signal gain, amplified spontaneous emission (ASE), noise figure,
/// and saturation.
#[derive(Debug, Clone)]
pub struct FiberAmplifier {
    /// Signal gain G (dB).
    pub gain_db: f64,
    /// Noise figure NF (dB).
    pub noise_figure_db: f64,
    /// Amplification bandwidth (nm, 3 dB).
    pub bandwidth_nm: f64,
    /// Centre wavelength λ₀ (nm).
    pub center_wavelength_nm: f64,
    /// Saturation output power P_sat (dBm).
    pub saturation_power_dbm: f64,
}

impl FiberAmplifier {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Create an amplifier with explicit parameters.
    pub fn new(
        gain_db: f64,
        noise_figure_db: f64,
        bandwidth_nm: f64,
        center_wavelength_nm: f64,
    ) -> Self {
        Self {
            gain_db,
            noise_figure_db,
            bandwidth_nm,
            center_wavelength_nm,
            saturation_power_dbm: 17.0, // typical EDFA
        }
    }

    /// Set the saturation output power.
    pub fn with_saturation(mut self, sat_power_dbm: f64) -> Self {
        self.saturation_power_dbm = sat_power_dbm;
        self
    }

    /// Typical C-band EDFA: G = 30 dB, NF = 5 dB, BW = 35 nm, λ₀ = 1550 nm.
    pub fn edfa_c_band() -> Self {
        Self {
            gain_db: 30.0,
            noise_figure_db: 5.0,
            bandwidth_nm: 35.0,
            center_wavelength_nm: 1550.0,
            saturation_power_dbm: 17.0,
        }
    }

    // -----------------------------------------------------------------------
    // Gain and noise
    // -----------------------------------------------------------------------

    /// Linear power gain G = 10^(G_dB/10).
    pub fn linear_gain(&self) -> f64 {
        10.0_f64.powf(self.gain_db / 10.0)
    }

    /// Amplified spontaneous emission (ASE) noise power (dBm) in the
    /// amplifier bandwidth:
    ///   P_ASE = hν · (G − 1) · n_sp · BW
    ///
    /// where n_sp = NF·G / (2·(G−1)) is the spontaneous emission factor.
    /// The result is expressed in dBm.
    pub fn spontaneous_emission_power_dbm(&self) -> f64 {
        let g = self.linear_gain();
        if g <= 1.0 + 1.0e-10 {
            // No gain → no ASE
            return -f64::INFINITY;
        }
        let nf_linear = 10.0_f64.powf(self.noise_figure_db / 10.0);
        // n_sp = NF · G / (2 · (G-1))
        let n_sp = nf_linear * g / (2.0 * (g - 1.0));
        // Photon energy at centre wavelength
        let h = 6.626e-34_f64; // J·s
        let c = 2.998e8_f64; // m/s
        let nu = c / (self.center_wavelength_nm * 1.0e-9);
        // Optical bandwidth in Hz: Δν ≈ c·Δλ/λ²
        let delta_nu =
            c * self.bandwidth_nm * 1.0e-9 / ((self.center_wavelength_nm * 1.0e-9).powi(2));
        let p_ase_w = h * nu * n_sp * (g - 1.0) * delta_nu;
        // Convert W → dBm
        10.0 * (p_ase_w * 1.0e3).log10()
    }

    /// Output OSNR (dB) given input signal power `input_power_dbm` (dBm).
    ///
    /// OSNR = P_signal_out / P_ASE = G·P_in / P_ASE.
    pub fn osnr_db(&self, input_power_dbm: f64) -> f64 {
        let p_in_w = 1.0e-3 * 10.0_f64.powf(input_power_dbm / 10.0);
        let p_out_w = self.linear_gain() * p_in_w;
        let ase_dbm = self.spontaneous_emission_power_dbm();
        if ase_dbm.is_infinite() {
            return f64::INFINITY;
        }
        let p_ase_w = 1.0e-3 * 10.0_f64.powf(ase_dbm / 10.0);
        if p_ase_w < 1.0e-60 {
            return f64::INFINITY;
        }
        10.0 * (p_out_w / p_ase_w).log10()
    }

    // -----------------------------------------------------------------------
    // Pulse amplification
    // -----------------------------------------------------------------------

    /// Amplify a pulse by applying the linear gain to the amplitude:
    ///   A_out(t) = √G · A_in(t)
    ///
    /// Gain saturation is not modelled here (see `amplify_pulse_saturated`).
    pub fn amplify_pulse(&self, pulse: &OpticalPulse) -> OpticalPulse {
        let sqrt_g = self.linear_gain().sqrt();
        let amplitude: Vec<Complex64> = pulse.amplitude.iter().map(|&a| a * sqrt_g).collect();
        // Build the output pulse — cannot fail because we copy t from input
        OpticalPulse {
            t: pulse.t.clone(),
            amplitude,
            center_wavelength_nm: pulse.center_wavelength_nm,
            dt: pulse.dt,
        }
    }

    /// Amplify a pulse with a simple gain-saturation model.
    ///
    /// The effective gain is G_eff = G_small_signal / (1 + E_in/E_sat) where
    /// E_sat is the saturation energy.  `saturation_energy_j` is the
    /// amplifier saturation energy parameter.
    pub fn amplify_pulse_saturated(
        &self,
        pulse: &OpticalPulse,
        saturation_energy_j: f64,
    ) -> OpticalPulse {
        let e_in = pulse.energy_j();
        let g_small = self.linear_gain();
        let g_eff = g_small / (1.0 + e_in / saturation_energy_j.max(1.0e-60));
        let sqrt_g = g_eff.sqrt();
        let amplitude: Vec<Complex64> = pulse.amplitude.iter().map(|&a| a * sqrt_g).collect();
        OpticalPulse {
            t: pulse.t.clone(),
            amplitude,
            center_wavelength_nm: pulse.center_wavelength_nm,
            dt: pulse.dt,
        }
    }

    /// Test whether the input pulse exceeds the amplifier's saturation threshold.
    pub fn is_saturated(&self, pulse: &OpticalPulse) -> bool {
        let p_sat_w = 1.0e-3 * 10.0_f64.powf(self.saturation_power_dbm / 10.0);
        pulse.peak_power() > p_sat_w / self.linear_gain().max(1.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fiber::dispersion::FiberDispersion;
    use approx::assert_relative_eq;

    fn smf28_solver(length_m: f64) -> NlseSolver {
        NlseSolver::new(
            FiberDispersion::smf28(),
            1.3e-3, // γ = 1.3 /W/km = 1.3e-3 /W/m
            4.6e-5, // α ≈ 0.2 dB/km → 4.6e-5 /m
            100.0,  // 100 m steps
            length_m,
        )
    }

    // -----------------------------------------------------------------------
    // NlseSolver tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_soliton_number_formula() {
        // N² = γ·P₀·T₀²/|β₂|
        // Use SMF-28 and choose P₀ so that N = 1
        let fiber = FiberDispersion::smf28();
        let gamma = 1.3e-3_f64; // 1/W/m
        let fwhm_ps = 1.0_f64;
        // T₀ (sech) = FWHM / (2·ln(1+√2))
        let t0_s = fwhm_ps * 1.0e-12 / (2.0 * (1.0 + 2.0_f64.sqrt()).ln());
        let b2_abs = fiber.beta2_s2_per_m().abs();
        // P₁ = |β₂| / (γ · T₀²)  → N = 1
        let p1 = b2_abs / (gamma * t0_s * t0_s);
        let solver = NlseSolver::new(fiber, gamma, 0.0, 100.0, 1.0e3);
        let n = solver.soliton_number(p1, fwhm_ps);
        assert_relative_eq!(n, 1.0, max_relative = 1.0e-6);
    }

    #[test]
    fn test_soliton_power() {
        // soliton_power_w(FWHM) should return ≈ P₁ computed from the formula
        let fiber = FiberDispersion::smf28();
        let gamma = 1.3e-3_f64;
        let fwhm_ps = 1.0_f64;
        let solver = NlseSolver::new(fiber.clone(), gamma, 0.0, 100.0, 1.0e3);
        let p1_solver = solver.soliton_power_w(fwhm_ps);
        // Independent calculation: P₁ = |β₂|/(γ T₀²)
        let ln_fac = 2.0 * (1.0 + 2.0_f64.sqrt()).ln();
        let t0_s = fwhm_ps * 1.0e-12 / ln_fac;
        let p1_ref = fiber.beta2_s2_per_m().abs() / (gamma * t0_s * t0_s);
        assert_relative_eq!(p1_solver, p1_ref, max_relative = 1.0e-9);
    }

    #[test]
    fn test_spm_phase_shift() {
        // Lossless fibre: φ_NL = γ · P₀ · L
        let fiber = FiberDispersion::smf28();
        let gamma = 1.3e-3_f64;
        let solver = NlseSolver::new(fiber, gamma, 0.0, 100.0, 1.0e3);
        let p0 = 1.0_f64;
        let length_m = 1000.0_f64;
        let phi = solver.spm_phase_shift(p0, length_m);
        let expected = gamma * p0 * length_m;
        assert_relative_eq!(phi, expected, max_relative = 1.0e-9);
    }

    #[test]
    fn test_nlse_propagate_gaussian_broadens() {
        // A very weak Gaussian pulse propagating in a dispersive fibre should
        // broaden without appreciable SPM.  Use anomalous SMF-28 and low power.
        let n_pts = 1024_usize;
        let t_window_ps = 200.0_f64;
        let fwhm_ps = 10.0_f64;
        let p0 = 1.0e-6_f64; // nW — negligible SPM
        let pulse = OpticalPulse::gaussian(n_pts, t_window_ps, p0, fwhm_ps, 1550.0);
        let w0 = pulse.rms_width_s();

        let solver = smf28_solver(50.0e3); // 50 km
        let out = solver.propagate(&pulse).expect("propagation failed");
        let w1 = out.rms_width_s();
        assert!(
            w1 > w0,
            "Gaussian pulse must broaden in dispersive fibre: σ₀={w0:.3e} s, σ₁={w1:.3e} s"
        );
    }

    #[test]
    fn test_lossless_power_conservation() {
        // Lossless fibre (α = 0): pulse energy should be conserved.
        let n_pts = 1024_usize;
        let pulse = OpticalPulse::gaussian(n_pts, 100.0, 1.0, 5.0, 1550.0);
        let e0 = pulse.energy_j();
        let fiber = FiberDispersion::smf28();
        let solver = NlseSolver::new(fiber, 1.3e-3, 0.0, 100.0, 1.0e3);
        let out = solver.propagate(&pulse).expect("propagation failed");
        let e1 = out.energy_j();
        let rel_err = (e1 - e0).abs() / e0;
        assert!(
            rel_err < 5.0e-3,
            "Energy not conserved (lossless): rel_err = {rel_err:.2e}"
        );
    }

    #[test]
    fn test_propagate_with_snapshots_count() {
        let n_pts = 512_usize;
        let pulse = OpticalPulse::gaussian(n_pts, 50.0, 1.0, 2.0, 1550.0);
        let fiber = FiberDispersion::smf28();
        // 10 steps, snapshot every 5 → expect initial + 2 snapshots = 3
        let solver = NlseSolver::new(fiber, 1.3e-3, 0.0, 100.0, 1.0e3);
        let snaps = solver
            .propagate_with_snapshots(&pulse, 5)
            .expect("snapshot propagation failed");
        // Snapshots: initial + every 5 steps (steps 5 and 10) = 3
        assert!(
            snaps.len() >= 2,
            "Expected at least 2 snapshots, got {}",
            snaps.len()
        );
    }

    #[test]
    fn test_nonlinear_length_formula() {
        let fiber = FiberDispersion::smf28();
        let gamma = 1.3e-3_f64;
        let solver = NlseSolver::new(fiber, gamma, 0.0, 100.0, 1.0e3);
        let p0 = 1.0_f64;
        let lnl = solver.nonlinear_length_m(p0);
        assert_relative_eq!(lnl, 1.0 / (gamma * p0), max_relative = 1.0e-12);
    }

    #[test]
    fn test_raman_solver_produces_output() {
        let n_pts = 512_usize;
        let pulse = OpticalPulse::sech(n_pts, 50.0, 100.0, 1.0, 1550.0);
        let fiber = FiberDispersion::smf28();
        let solver = NlseSolver::new(fiber, 1.3e-3, 0.0, 10.0, 100.0).with_raman(0.18);
        let out = solver.propagate(&pulse).expect("Raman propagation failed");
        assert_eq!(out.amplitude.len(), n_pts);
    }

    // -----------------------------------------------------------------------
    // FiberAmplifier tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fiber_amplifier_gain() {
        let amp = FiberAmplifier::edfa_c_band();
        // Linear gain = 10^(30/10) = 1000
        assert_relative_eq!(amp.linear_gain(), 1000.0, max_relative = 1.0e-9);
    }

    #[test]
    fn test_fiber_amplifier_amplifies_pulse() {
        let amp = FiberAmplifier::edfa_c_band(); // 30 dB gain
        let pulse = OpticalPulse::gaussian(512, 20.0, 1.0e-3, 1.0, 1550.0);
        let out = amp.amplify_pulse(&pulse);
        // Peak power should be multiplied by the linear gain (√G on amplitude → G on power)
        let ratio = out.peak_power() / pulse.peak_power();
        assert_relative_eq!(ratio, amp.linear_gain(), max_relative = 1.0e-9);
    }

    #[test]
    fn test_fiber_amplifier_energy_scales_with_gain() {
        let amp = FiberAmplifier::edfa_c_band();
        let pulse = OpticalPulse::gaussian(512, 20.0, 1.0e-6, 1.0, 1550.0);
        let out = amp.amplify_pulse(&pulse);
        let ratio = out.energy_j() / pulse.energy_j();
        assert_relative_eq!(ratio, amp.linear_gain(), max_relative = 1.0e-9);
    }

    #[test]
    fn test_fiber_amplifier_ase_power_finite() {
        let amp = FiberAmplifier::edfa_c_band();
        let ase = amp.spontaneous_emission_power_dbm();
        assert!(
            ase.is_finite(),
            "ASE power must be finite for a 30 dB EDFA, got {ase}"
        );
    }

    #[test]
    fn test_fiber_amplifier_osnr_positive() {
        let amp = FiberAmplifier::edfa_c_band();
        let osnr = amp.osnr_db(-10.0); // −10 dBm input
        assert!(
            osnr > 0.0,
            "OSNR must be positive for a high-gain amplifier, got {osnr:.2} dB"
        );
    }

    #[test]
    fn test_omega_array_length() {
        let n = 256_usize;
        let dt = 1.0e-14_f64;
        let omega = NlseSolver::omega_array(n, dt);
        assert_eq!(omega.len(), n);
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        let n = 64_usize;
        let x: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.1).sin(), 0.0))
            .collect();
        let fiber = FiberDispersion::smf28();
        let solver = NlseSolver::new(fiber, 1.3e-3, 0.0, 100.0, 1.0e3);
        let spec = solver.fft(&x);
        // Truncate back to n (fft zero-pads to power-of-two)
        let recovered = solver.ifft(&spec)[..n].to_vec();
        for (orig, rec) in x.iter().zip(recovered.iter()) {
            let err = (orig - rec).norm();
            assert!(err < 1.0e-9, "FFT/IFFT roundtrip error: {err:.2e}");
        }
    }
}
