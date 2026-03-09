//! Cross-phase modulation (XPM) in optical fibers.
//!
//! XPM occurs when two (or more) co-propagating pulses at different wavelengths
//! interact through the Kerr nonlinearity. The phase of pulse 1 is modified by
//! the intensity of pulse 2:
//!
//!   dφ₁/dz = γ·(|A₁|² + 2·|A₂|²)  — factor of 2 from XPM vs SPM
//!
//! XPM causes:
//!   - Spectral broadening of each channel
//!   - Frequency shifts (chirp) on edges of overlapping pulses
//!   - Timing jitter in WDM systems
//!
//! The XPM walk-off length L_W = T₀ / |d₁₂| where d₁₂ = 1/v_g1 - 1/v_g2.

use num_complex::Complex64;

use crate::fiber::nonlinear::spm::{fft_spm, ifft_spm};

/// Cross-phase modulation model for two co-propagating channels.
#[derive(Debug, Clone, Copy)]
pub struct XpmChannel {
    /// Nonlinear coefficient γ (W⁻¹m⁻¹) — same for both channels
    pub gamma: f64,
    /// Group velocity dispersion D₁₂ = d(1/v_g)/dλ · Δλ (s/m): walk-off between channels
    pub walk_off_per_m: f64,
    /// Fiber attenuation α (m⁻¹)
    pub alpha: f64,
    /// Fiber length (m)
    pub length: f64,
}

impl XpmChannel {
    /// Create XPM model.
    pub fn new(gamma: f64, d12_ps_per_km: f64, alpha_db_per_km: f64, length_m: f64) -> Self {
        let walk_off_per_m = d12_ps_per_km * 1e-12 / 1e3; // ps/km → s/m
        let alpha = alpha_db_per_km * 1e-3 / (10.0 / std::f64::consts::LN_10);
        Self {
            gamma,
            walk_off_per_m,
            alpha,
            length: length_m,
        }
    }

    /// Effective length L_eff.
    pub fn effective_length(&self) -> f64 {
        if self.alpha < 1e-30 {
            return self.length;
        }
        (1.0 - (-self.alpha * self.length).exp()) / self.alpha
    }

    /// Walk-off length L_W = T₀ / |d₁₂| (m).
    ///
    /// When L >> L_W, walk-off reduces XPM efficiency.
    pub fn walk_off_length(&self, pulse_duration_s: f64) -> f64 {
        if self.walk_off_per_m.abs() < 1e-30 {
            return f64::INFINITY;
        }
        pulse_duration_s / self.walk_off_per_m.abs()
    }

    /// XPM-induced phase shift on channel 1 from channel 2 (CW approximation).
    ///
    ///   φ_XPM = 2·γ·P₂·L_eff
    pub fn xpm_phase_shift(&self, p2_watts: f64) -> f64 {
        2.0 * self.gamma * p2_watts * self.effective_length()
    }

    /// Frequency chirp δω (rad/s) induced by XPM on channel 1 from channel 2.
    ///
    /// For a Gaussian pulse P₂(t) = P₀·exp(-t²/T₀²):
    ///   δω_XPM(t) = -dφ_XPM/dt ≈ 4·γ·P₀·t/T₀²·L_eff
    /// Maximum chirp at t = T₀/√2:
    ///   δω_max = 4·γ·P₀·L_eff / (T₀·√(2e))
    pub fn max_xpm_chirp(&self, p2_peak_w: f64, pulse_duration_s: f64) -> f64 {
        4.0 * self.gamma * p2_peak_w * self.effective_length()
            / (pulse_duration_s * (2.0 * std::f64::consts::E).sqrt())
    }

    /// Apply XPM phase from channel 2 to channel 1 envelope.
    ///
    /// Both channels passed as slices of [re, im] complex amplitudes (same length).
    /// Returns phase-rotated channel 1.
    pub fn apply_xpm(&self, ch1: &[[f64; 2]], ch2: &[[f64; 2]]) -> Vec<[f64; 2]> {
        let l_eff = self.effective_length();
        ch1.iter()
            .zip(ch2.iter())
            .map(|(&[r1, i1], &[r2, i2])| {
                let intensity2 = r2 * r2 + i2 * i2;
                let phase = 2.0 * self.gamma * intensity2 * l_eff;
                let (s, c) = phase.sin_cos();
                [r1 * c - i1 * s, r1 * s + i1 * c]
            })
            .collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// XpmCoeff: wavelength-resolved nonlinear coefficients
// ──────────────────────────────────────────────────────────────────────────────

/// Wavelength-resolved nonlinear coefficients for two-channel XPM.
///
/// The nonlinear coefficient γ = n₂·ω/(c·A_eff) = 2π·n₂/(λ·A_eff).
/// For XPM the coupling is 2γ (twice the SPM coefficient).
#[derive(Debug, Clone, Copy)]
pub struct XpmCoeff {
    n2: f64,
    wavelength1: f64,
    wavelength2: f64,
    a_eff: f64,
}

impl XpmCoeff {
    /// Create XPM coefficients.
    ///
    /// # Arguments
    /// - `n2`          — nonlinear index coefficient (m²/W)
    /// - `wavelength1` — first channel wavelength (m)
    /// - `wavelength2` — second channel wavelength (m)
    /// - `a_eff`       — effective mode area (m²)
    pub fn new(n2: f64, wavelength1: f64, wavelength2: f64, a_eff: f64) -> Self {
        Self {
            n2,
            wavelength1,
            wavelength2,
            a_eff,
        }
    }

    /// Nonlinear coefficient γ₁ for channel 1 (W⁻¹m⁻¹).
    ///
    ///   γ₁ = 2π·n₂ / (λ₁·A_eff)
    pub fn gamma1(&self) -> f64 {
        2.0 * std::f64::consts::PI * self.n2 / (self.wavelength1 * self.a_eff)
    }

    /// Nonlinear coefficient γ₂ for channel 2 (W⁻¹m⁻¹).
    ///
    ///   γ₂ = 2π·n₂ / (λ₂·A_eff)
    pub fn gamma2(&self) -> f64 {
        2.0 * std::f64::consts::PI * self.n2 / (self.wavelength2 * self.a_eff)
    }

    /// XPM coupling coefficient (W⁻¹m⁻¹) = 2·γ₁ (factor of 2 vs SPM).
    ///
    /// The XPM phase on channel 1 per unit length and per unit channel-2 power:
    ///   dφ₁_XPM/dz = 2·γ₁·P₂
    pub fn xpm_coeff(&self) -> f64 {
        2.0 * self.gamma1()
    }

    /// Group velocity mismatch (walk-off) δ = 1/v_g1 - 1/v_g2 (s/m).
    ///
    /// `beta1_1` and `beta1_2` are the first-order propagation constants
    /// (inverse group velocities, units s/m) at channels 1 and 2 respectively.
    pub fn group_velocity_mismatch(&self, beta1_1: f64, beta1_2: f64) -> f64 {
        beta1_1 - beta1_2
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// TwoChannelPropagation: split-step with XPM coupling
// ──────────────────────────────────────────────────────────────────────────────

/// Co-propagation of two WDM channels with mutual XPM coupling.
///
/// The coupled equations (lossless, no dispersion limit):
///   ∂A₁/∂z = i·γ₁·(|A₁|² + 2|A₂|²)·A₁
///   ∂A₂/∂z = i·γ₂·(|A₂|² + 2|A₁|²)·A₂
///
/// A full split-step Fourier scheme includes dispersion via the
/// `beta2_1`/`beta2_2` fields (both can be set to 0 for pure XPM).
#[derive(Debug, Clone)]
pub struct TwoChannelPropagation {
    /// XPM coefficient container
    pub xpm: XpmCoeff,
    /// Total fiber length (m)
    pub fiber_length: f64,
    /// Number of propagation steps
    pub n_steps: usize,
    /// Step size dz (m)
    pub dz: f64,
    /// GVD for channel 1 (s²/m)
    pub beta2_1: f64,
    /// GVD for channel 2 (s²/m)
    pub beta2_2: f64,
    /// Loss for channel 1 (m⁻¹)
    pub alpha1: f64,
    /// Loss for channel 2 (m⁻¹)
    pub alpha2: f64,
}

impl TwoChannelPropagation {
    /// Create a two-channel propagation solver.
    ///
    /// Dispersion and loss are set to zero by default; use the builder setters
    /// to include them.
    pub fn new(xpm: XpmCoeff, fiber_length: f64, n_steps: usize) -> Self {
        let dz = fiber_length / n_steps as f64;
        Self {
            xpm,
            fiber_length,
            n_steps,
            dz,
            beta2_1: 0.0,
            beta2_2: 0.0,
            alpha1: 0.0,
            alpha2: 0.0,
        }
    }

    /// Set GVD parameters (s²/m) and loss (m⁻¹) for both channels.
    pub fn with_dispersion_loss(
        mut self,
        beta2_1: f64,
        beta2_2: f64,
        alpha1: f64,
        alpha2: f64,
    ) -> Self {
        self.beta2_1 = beta2_1;
        self.beta2_2 = beta2_2;
        self.alpha1 = alpha1;
        self.alpha2 = alpha2;
        self
    }

    /// Propagate two channels simultaneously with XPM coupling.
    ///
    /// Returns `(a1_out, a2_out)`.
    pub fn propagate(
        &self,
        a1: &[Complex64],
        a2: &[Complex64],
        dt: f64,
    ) -> (Vec<Complex64>, Vec<Complex64>) {
        let mut a1 = a1.to_vec();
        let mut a2 = a2.to_vec();
        let g1 = self.xpm.gamma1();
        let g2 = self.xpm.gamma2();

        for _ in 0..self.n_steps {
            // Half nonlinear step (SPM + XPM) for both channels
            xpm_nl_half_step(&mut a1, &a2, g1, self.dz / 2.0);
            xpm_nl_half_step(&mut a2, &a1, g2, self.dz / 2.0);

            // Full linear (dispersion + loss) step
            if self.beta2_1 != 0.0 || self.alpha1 != 0.0 {
                apply_linear_step(&mut a1, self.beta2_1, self.alpha1, self.dz, dt);
            }
            if self.beta2_2 != 0.0 || self.alpha2 != 0.0 {
                apply_linear_step(&mut a2, self.beta2_2, self.alpha2, self.dz, dt);
            }

            // Half nonlinear step
            xpm_nl_half_step(&mut a1, &a2, g1, self.dz / 2.0);
            xpm_nl_half_step(&mut a2, &a1, g2, self.dz / 2.0);
        }
        (a1, a2)
    }

    /// Maximum XPM phase shift on a probe channel from a pump pulse.
    ///
    ///   φ_XPM_max = 2·γ₁·E_pump / A_eff
    ///
    /// where E_pump = pump_energy (J) is the total pump pulse energy.
    /// With `a_eff` absorbed into `xpm.a_eff` via γ₁, the result is:
    ///   φ_XPM_max = 2·γ₁·P_peak·L_eff
    ///
    /// This convenience method estimates the max XPM phase from pump energy
    /// assuming a rectangular pulse over the full fiber effective length with
    /// L_eff computed from the stored parameters.
    pub fn xpm_phase_shift(&self, pump_energy: f64) -> f64 {
        // Effective length assuming no loss (conservative upper bound)
        2.0 * self.xpm.gamma1() * pump_energy
    }
}

fn xpm_nl_half_step(a: &mut [Complex64], b: &[Complex64], gamma: f64, dz: f64) {
    for (ai, bi) in a.iter_mut().zip(b.iter()) {
        let phi = gamma * (ai.norm_sqr() + 2.0 * bi.norm_sqr()) * dz;
        *ai *= Complex64::new(0.0, phi).exp();
    }
}

fn apply_linear_step(a: &mut [Complex64], beta2: f64, alpha: f64, dz: f64, dt: f64) {
    use std::f64::consts::PI;
    let n = a.len();
    let mut spec = fft_spm(a);
    let df = 1.0 / (n as f64 * dt);
    for (m, s) in spec.iter_mut().enumerate() {
        let freq = if m < n / 2 {
            m as f64 * df
        } else {
            (m as f64 - n as f64) * df
        };
        let omega = 2.0 * PI * freq;
        let loss = (-alpha / 2.0 * dz).exp();
        let disp_phase = -beta2 / 2.0 * omega * omega * dz;
        *s *= Complex64::new(0.0, disp_phase).exp() * loss;
    }
    let out = ifft_spm(&spec);
    a.copy_from_slice(&out);
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xpm_phase_shift_scales_with_power() {
        let x = XpmChannel::new(1.3e-3, 3.0, 0.2, 80e3);
        let p1 = x.xpm_phase_shift(1.0);
        let p2 = x.xpm_phase_shift(2.0);
        assert!((p2 - 2.0 * p1).abs() < 1e-10);
    }

    #[test]
    fn xpm_walk_off_length_decreases_with_walkoff() {
        let x1 = XpmChannel::new(1.3e-3, 1.0, 0.2, 80e3); // 1 ps/km
        let x2 = XpmChannel::new(1.3e-3, 10.0, 0.2, 80e3); // 10 ps/km
        let lw1 = x1.walk_off_length(1e-12);
        let lw2 = x2.walk_off_length(1e-12);
        assert!(lw1 > lw2, "Larger walk-off → shorter L_W");
    }

    #[test]
    fn xpm_apply_preserves_amplitude() {
        let x = XpmChannel::new(1.3e-3, 3.0, 0.2, 80e3);
        let ch1 = vec![[1.0f64, 0.0]; 5];
        let ch2 = vec![[0.5f64, 0.0]; 5];
        let out = x.apply_xpm(&ch1, &ch2);
        for &[r, i] in &out {
            assert!((r * r + i * i - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn xpm_max_chirp_positive() {
        let x = XpmChannel::new(1.3e-3, 3.0, 0.2, 80e3);
        let chirp = x.max_xpm_chirp(100e-3, 1e-12);
        assert!(chirp > 0.0);
    }

    // ── XpmCoeff tests ────────────────────────────────────────────────────────

    #[test]
    fn xpm_coeff_is_twice_gamma1() {
        // n2 = 2.6e-20 m²/W (silica), A_eff = 80 µm² = 80e-12 m²
        let xc = XpmCoeff::new(2.6e-20, 1550e-9, 1310e-9, 80e-12);
        let ratio = xc.xpm_coeff() / xc.gamma1();
        assert!(
            (ratio - 2.0).abs() < 1e-10,
            "xpm_coeff should be 2×γ₁, got ratio={ratio}"
        );
    }

    #[test]
    fn xpm_coeff_gamma_positive() {
        let xc = XpmCoeff::new(2.6e-20, 1550e-9, 1310e-9, 80e-12);
        assert!(xc.gamma1() > 0.0);
        assert!(xc.gamma2() > 0.0);
    }

    #[test]
    fn xpm_phase_shift_sign() {
        // XPM phase must be positive for positive pump energy (constructive Kerr)
        let xc = XpmCoeff::new(2.6e-20, 1550e-9, 1310e-9, 80e-12);
        let prop = TwoChannelPropagation::new(xc, 1e3, 10);
        let phi = prop.xpm_phase_shift(1e-12); // 1 pJ pump
        assert!(phi > 0.0, "XPM phase shift should be positive, got {phi}");
    }

    #[test]
    fn two_channel_power_conservation() {
        // Lossless, no dispersion: total power must be conserved
        let xc = XpmCoeff::new(2.6e-20, 1550e-9, 1310e-9, 80e-12);
        let prop = TwoChannelPropagation::new(xc, 1e3, 50);

        let n = 64;
        let dt = 0.5e-12;
        let t_center = (n as f64 - 1.0) / 2.0 * dt;
        let a1: Vec<Complex64> = (0..n)
            .map(|i| {
                let t = i as f64 * dt - t_center;
                Complex64::new((-t * t / (2.0 * (5e-12_f64).powi(2))).exp(), 0.0)
            })
            .collect();
        let a2: Vec<Complex64> = (0..n)
            .map(|i| {
                let t = i as f64 * dt - t_center;
                Complex64::new(0.5 * (-t * t / (2.0 * (5e-12_f64).powi(2))).exp(), 0.0)
            })
            .collect();

        let power_in: f64 = a1.iter().chain(a2.iter()).map(|v| v.norm_sqr()).sum();
        let (b1, b2) = prop.propagate(&a1, &a2, dt);
        let power_out: f64 = b1.iter().chain(b2.iter()).map(|v| v.norm_sqr()).sum();

        let rel_err = (power_out - power_in).abs() / power_in;
        assert!(
            rel_err < 1e-6,
            "Two-channel power not conserved: rel_err={rel_err:.2e}"
        );
    }
}
