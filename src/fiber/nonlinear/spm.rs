//! Self-phase modulation (SPM) in optical fibers.
//!
//! SPM arises from the intensity-dependent refractive index (Kerr effect):
//!   n = n₀ + n₂·I
//!
//! This causes an intensity-dependent phase shift:
//!   φ_NL(z) = γ · P(z) · dz   per unit length
//!
//! In the absence of dispersion, SPM causes spectral broadening:
//!   Δω_SPM ≈ 2·γ·P₀·L_eff
//!
//! where L_eff = (1 - exp(-αL))/α is the effective length.
//!
//! The nonlinear phase accumulated over fiber length L:
//!   φ_max = γ · P₀ · L_eff

use num_complex::Complex64;
use std::f64::consts::PI;

/// SPM model for an optical fiber.
#[derive(Debug, Clone, Copy)]
pub struct SpmFiber {
    /// Nonlinear coefficient γ (W⁻¹m⁻¹)
    pub gamma: f64,
    /// Fiber attenuation α (m⁻¹)
    pub alpha: f64,
    /// Fiber length L (m)
    pub length: f64,
}

impl SpmFiber {
    /// Create an SPM fiber model.
    pub fn new(gamma: f64, alpha_db_per_km: f64, length_m: f64) -> Self {
        let alpha = alpha_db_per_km * 1e-3 / (10.0 / std::f64::consts::LN_10); // dB/km → m⁻¹
        Self {
            gamma,
            alpha,
            length: length_m,
        }
    }

    /// Standard SMF-28 fiber parameters at 1550 nm.
    ///
    /// γ ≈ 1.3 W⁻¹km⁻¹, α = 0.2 dB/km, typical 80 km span.
    pub fn smf28_80km() -> Self {
        Self::new(1.3e-3, 0.2, 80e3)
    }

    /// Effective length L_eff = (1 - exp(-αL))/α.
    pub fn effective_length(&self) -> f64 {
        if self.alpha < 1e-30 {
            return self.length;
        }
        (1.0 - (-self.alpha * self.length).exp()) / self.alpha
    }

    /// Maximum nonlinear phase shift (rad) for peak power P₀ (W).
    ///
    ///   φ_max = γ · P₀ · L_eff
    pub fn max_nonlinear_phase(&self, p0_watts: f64) -> f64 {
        self.gamma * p0_watts * self.effective_length()
    }

    /// SPM spectral broadening factor (approximate).
    ///
    /// For a Gaussian pulse with φ_max >> 1:
    ///   Δω_spm / Δω_0 ≈ 1 + 4·(φ_max/3√3)^2 )^{1/2}  (Agrawal approx.)
    ///
    /// Simplified: Δω/Δω₀ ≈ √(1 + (4/3√3 · φ_max)²)
    pub fn spectral_broadening_factor(&self, p0_watts: f64) -> f64 {
        let phi = self.max_nonlinear_phase(p0_watts);
        (1.0 + (4.0 * phi / (3.0 * 3.0_f64.sqrt())).powi(2)).sqrt()
    }

    /// Critical power for self-focusing (bulk): P_cr = λ²/(2π·n₀·n₂·A_eff).
    /// Here as reference: maximum power for φ_max = π.
    pub fn power_at_pi_phase(&self) -> f64 {
        std::f64::consts::PI / (self.gamma * self.effective_length())
    }

    /// Apply SPM phase to a CW field amplitude envelope u(z) over fiber.
    ///
    /// Returns phase-rotated field: u_out = u_in · exp(i · γ·|u_in|²·L_eff)
    pub fn apply_spm_cw(&self, field_re: f64, field_im: f64) -> (f64, f64) {
        let intensity = field_re * field_re + field_im * field_im;
        let phase = self.gamma * intensity * self.effective_length();
        let (s, c) = phase.sin_cos();
        (field_re * c - field_im * s, field_re * s + field_im * c)
    }

    /// Apply SPM to a pulse envelope (array of complex amplitudes).
    ///
    /// Each element is [re, im]; returns phase-rotated envelope.
    pub fn apply_spm_pulse(&self, envelope: &[[f64; 2]]) -> Vec<[f64; 2]> {
        let l_eff = self.effective_length();
        envelope
            .iter()
            .map(|&[re, im]| {
                let intensity = re * re + im * im;
                let phase = self.gamma * intensity * l_eff;
                let (s, c) = phase.sin_cos();
                [re * c - im * s, re * s + im * c]
            })
            .collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Split-step NLSE solver
// ──────────────────────────────────────────────────────────────────────────────

/// Nonlinear Schrödinger Equation (NLSE) split-step Fourier solver.
///
/// Propagates a complex pulse envelope through a fiber described by:
///   ∂A/∂z = -α/2·A - i·β₂/2·∂²A/∂T² + i·γ|A|²·A
///
/// The split-step Fourier method alternates:
///   1. Half-step nonlinear phase (SPM) in time domain
///   2. Full linear step (dispersion + loss) in frequency domain
///   3. Half-step nonlinear phase
#[derive(Debug, Clone)]
pub struct SplitStepNls {
    /// Nonlinear coefficient γ (W⁻¹m⁻¹)
    pub gamma: f64,
    /// Group velocity dispersion β₂ (ps²/km stored as s²/m internally)
    pub beta2: f64,
    /// Loss α (dB/km stored as m⁻¹ internally)
    pub alpha: f64,
    /// Step size dz (m)
    pub dz: f64,
    /// Number of propagation steps
    pub n_steps: usize,
}

impl SplitStepNls {
    /// Create a new split-step NLSE solver.
    ///
    /// # Arguments
    /// - `gamma`        — nonlinear coefficient (W⁻¹m⁻¹)
    /// - `beta2`        — GVD in ps²/km (negative = anomalous)
    /// - `alpha_db_km`  — fiber loss in dB/km
    /// - `fiber_length` — total fiber length in m
    /// - `n_steps`      — number of propagation steps
    pub fn new(
        gamma: f64,
        beta2_ps2_per_km: f64,
        alpha_db_km: f64,
        fiber_length: f64,
        n_steps: usize,
    ) -> Self {
        // Convert units
        let beta2 = beta2_ps2_per_km * 1e-24; // ps²/km → s²/m
        let alpha = alpha_db_km * 1e-3 / (10.0 / std::f64::consts::LN_10); // dB/km → m⁻¹
        let dz = fiber_length / n_steps as f64;
        Self {
            gamma,
            beta2,
            alpha,
            dz,
            n_steps,
        }
    }

    /// Propagate pulse envelope `a0` (sqrt(W)) through the fiber.
    ///
    /// `dt` is the time-domain sample spacing (s). The input slice length must be
    /// a power of 2 for best performance (the internal Cooley-Tukey FFT requires it).
    ///
    /// Returns the output pulse envelope.
    pub fn propagate(&self, a0: &[Complex64], dt: f64) -> Vec<Complex64> {
        let mut a = a0.to_vec();
        for _ in 0..self.n_steps {
            // Half nonlinear step
            apply_nl_step(&mut a, self.gamma, self.dz / 2.0);
            // Full linear (dispersion + loss) step
            apply_disp_step(&mut a, self.beta2, self.alpha, self.dz, dt);
            // Half nonlinear step
            apply_nl_step(&mut a, self.gamma, self.dz / 2.0);
        }
        a
    }

    /// RMS pulse width √(〈T²〉 - 〈T〉²) from |A(T)|² intensity profile.
    pub fn pulse_width_rms(a: &[Complex64], dt: f64) -> f64 {
        let n = a.len();
        let t_center = (n as f64 - 1.0) / 2.0 * dt;
        let power: Vec<f64> = a.iter().map(|v| v.norm_sqr()).collect();
        let total: f64 = power.iter().sum();
        if total < 1e-60 {
            return 0.0;
        }
        let mean_t = power
            .iter()
            .enumerate()
            .map(|(i, &p)| (i as f64 * dt - t_center) * p)
            .sum::<f64>()
            / total;
        let var = power
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let t = i as f64 * dt - t_center - mean_t;
                t * t * p
            })
            .sum::<f64>()
            / total;
        var.sqrt()
    }

    /// Instantaneous frequency chirp parameter C from d(arg A)/dt.
    ///
    /// Returns the mean signed chirp integrated over the pulse profile, normalised
    /// by Δω_rms of the input spectrum.  Positive C = up-chirp.
    pub fn chirp_parameter(a: &[Complex64], dt: f64) -> f64 {
        let n = a.len();
        if n < 2 {
            return 0.0;
        }
        // Instantaneous angular frequency: ω_inst(t) = d(arg A)/dt ≈ Im[ A*(t) · (A(t+1)-A(t-1))/(2dt) ] / |A|²
        let power: Vec<f64> = a.iter().map(|v| v.norm_sqr()).collect();
        let total: f64 = power.iter().sum();
        if total < 1e-60 {
            return 0.0;
        }

        let mut chirp_sum = 0.0_f64;
        for i in 1..n - 1 {
            if power[i] < 1e-30 * total {
                continue;
            }
            // Central difference for dA/dt
            let da_re = (a[i + 1].re - a[i - 1].re) / (2.0 * dt);
            let da_im = (a[i + 1].im - a[i - 1].im) / (2.0 * dt);
            // Im[ A* · dA/dt ] = Re·da_im - Im·da_re
            let inst_omega = (a[i].re * da_im - a[i].im * da_re) / power[i];
            chirp_sum += inst_omega * power[i];
        }
        let mean_omega = chirp_sum / total;

        // Normalise by the RMS spectral width of the field
        let spec = fft_spm(a);
        let n_f = spec.len() as f64;
        let df = 1.0 / (n as f64 * dt);
        let spec_power: Vec<f64> = spec.iter().map(|v| v.norm_sqr()).collect();
        let spec_total: f64 = spec_power.iter().sum();
        if spec_total < 1e-60 {
            return 0.0;
        }
        let mean_freq = spec_power
            .iter()
            .enumerate()
            .map(|(k, &p)| {
                let f = if k < spec.len() / 2 {
                    k as f64 * df
                } else {
                    (k as f64 - n_f) * df
                };
                f * p
            })
            .sum::<f64>()
            / spec_total;
        let var_freq = spec_power
            .iter()
            .enumerate()
            .map(|(k, &p)| {
                let f = if k < spec.len() / 2 {
                    k as f64 * df
                } else {
                    (k as f64 - n_f) * df
                };
                let df2 = f - mean_freq;
                df2 * df2 * p
            })
            .sum::<f64>()
            / spec_total;
        let omega_rms = (2.0 * PI * var_freq.sqrt()).max(1e-30);
        mean_omega / omega_rms
    }

    /// Short-time Fourier transform (spectrogram) using a Hann gate window.
    ///
    /// Returns a `n_time × n_freq` matrix of spectral power densities where
    /// - `n_time` = number of time-gate positions (equals `a.len()`)
    /// - `n_freq` = number of frequency bins requested
    ///
    /// `gate_width` is the 1-σ width of the Hann window in seconds.
    pub fn spectrogram(a: &[Complex64], dt: f64, gate_width: f64, n_freq: usize) -> Vec<Vec<f64>> {
        let n_t = a.len();
        let gate_samples = ((gate_width / dt) as usize).max(4).min(n_t);
        // Number of time positions: stride by 1 (full resolution)
        let mut result = vec![vec![0.0_f64; n_freq]; n_t];
        let half = gate_samples / 2;

        // t_center is used as arithmetic offset (t_center + k - half), not just an iterator index.
        // k is used to compute Hann window weights — both loops need the numeric index.
        #[allow(clippy::needless_range_loop)]
        for t_center in 0..n_t {
            // Extract windowed slice, zero-padded to n_freq
            let mut windowed = vec![Complex64::new(0.0, 0.0); n_freq];
            for k in 0..gate_samples {
                let t_idx = t_center + k;
                if t_idx < half {
                    continue;
                }
                let t_idx = t_idx - half;
                if t_idx >= n_t {
                    continue;
                }
                // Hann window: w(k) = sin²(π·k/(N-1))
                let w = if gate_samples > 1 {
                    let arg = PI * k as f64 / (gate_samples - 1) as f64;
                    arg.sin().powi(2)
                } else {
                    1.0
                };
                if k < n_freq {
                    windowed[k] = a[t_idx] * w;
                }
            }
            // FFT of windowed slice
            let spec = fft_spm(&windowed);
            for (fi, s) in spec.iter().enumerate().take(n_freq) {
                result[t_center][fi] = s.norm_sqr();
            }
        }
        result
    }
}

// ── internal helpers ──────────────────────────────────────────────────────────

fn apply_nl_step(a: &mut [Complex64], gamma: f64, dz: f64) {
    for v in a.iter_mut() {
        let phi = gamma * v.norm_sqr() * dz;
        *v *= Complex64::new(0.0, phi).exp();
    }
}

fn apply_disp_step(a: &mut [Complex64], beta2: f64, alpha: f64, dz: f64, dt: f64) {
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

/// Cooley-Tukey radix-2 FFT (in-place style, recursive).
pub(crate) fn fft_spm(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    if n == 1 {
        return x.to_vec();
    }
    let half = n / 2;
    let even: Vec<Complex64> = (0..half).map(|k| x[2 * k]).collect();
    let odd: Vec<Complex64> = (0..half).map(|k| x[2 * k + 1]).collect();
    let fe = fft_spm(&even);
    let fo = fft_spm(&odd);
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    for k in 0..half {
        let angle = -2.0 * PI * k as f64 / n as f64;
        let tw = Complex64::new(angle.cos(), angle.sin());
        out[k] = fe[k] + tw * fo[k];
        out[k + half] = fe[k] - tw * fo[k];
    }
    out
}

pub(crate) fn ifft_spm(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    let conj_x: Vec<Complex64> = x.iter().map(|v| v.conj()).collect();
    let fft_conj = fft_spm(&conj_x);
    fft_conj.iter().map(|v| v.conj() / n as f64).collect()
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spm_effective_length_lossless() {
        let f = SpmFiber {
            gamma: 1.3e-3,
            alpha: 0.0,
            length: 1000.0,
        };
        assert!((f.effective_length() - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn spm_effective_length_lossy() {
        let f = SpmFiber::smf28_80km();
        let l_eff = f.effective_length();
        assert!(l_eff < f.length, "L_eff should be < L for lossy fiber");
        assert!(l_eff > 0.0);
    }

    #[test]
    fn spm_nonlinear_phase_scales_with_power() {
        let f = SpmFiber::smf28_80km();
        let p1 = f.max_nonlinear_phase(1.0);
        let p2 = f.max_nonlinear_phase(2.0);
        assert!((p2 - 2.0 * p1).abs() < 1e-10);
    }

    #[test]
    fn spm_spectral_broadening_ge_1() {
        let f = SpmFiber::smf28_80km();
        let factor = f.spectral_broadening_factor(100e-3);
        assert!(factor >= 1.0);
    }

    #[test]
    fn spm_apply_cw_preserves_amplitude() {
        let f = SpmFiber::smf28_80km();
        let (re_out, im_out) = f.apply_spm_cw(1.0, 0.0);
        let amp_sq = re_out * re_out + im_out * im_out;
        assert!((amp_sq - 1.0).abs() < 1e-10);
    }

    #[test]
    fn spm_apply_pulse_preserves_envelope_amplitude() {
        let f = SpmFiber::smf28_80km();
        let envelope: Vec<[f64; 2]> = (0..10)
            .map(|i| {
                let a = (-(i as f64 - 5.0).powi(2) / 4.0).exp();
                [a, 0.0]
            })
            .collect();
        let out = f.apply_spm_pulse(&envelope);
        for (i, (&[ri, ii], &[ro, io])) in envelope.iter().zip(out.iter()).enumerate() {
            let amp_in = ri * ri + ii * ii;
            let amp_out = ro * ro + io * io;
            assert!(
                (amp_out - amp_in).abs() < 1e-10,
                "Element {i} amplitude changed"
            );
        }
    }

    // ── SplitStepNls tests ────────────────────────────────────────────────────

    fn gaussian_pulse(n: usize, dt: f64, t0: f64, peak: f64) -> Vec<Complex64> {
        let t_center = (n as f64 - 1.0) / 2.0 * dt;
        (0..n)
            .map(|i| {
                let t = i as f64 * dt - t_center;
                let env = peak * (-t * t / (2.0 * t0 * t0)).exp();
                Complex64::new(env, 0.0)
            })
            .collect()
    }

    #[test]
    fn spm_broadens_spectrum() {
        // Lossless, no dispersion (beta2=0), strong SPM → spectrum must broaden
        let n = 256;
        let dt = 0.5e-12; // 0.5 ps per sample
        let t0 = 5e-12; // 5 ps pulse
        let peak = 10.0_f64.sqrt(); // sqrt(10 W)

        let a0 = gaussian_pulse(n, dt, t0, peak);

        // Compute initial spectral bandwidth (RMS)
        let spec0 = fft_spm(&a0);
        let spec_power0: Vec<f64> = spec0.iter().map(|v| v.norm_sqr()).collect();
        let total0: f64 = spec_power0.iter().sum();
        let df = 1.0 / (n as f64 * dt);
        let rms_bw0 = {
            let mean_f = spec_power0
                .iter()
                .enumerate()
                .map(|(k, &p)| {
                    let f = if k < n / 2 {
                        k as f64 * df
                    } else {
                        (k as f64 - n as f64) * df
                    };
                    f * p
                })
                .sum::<f64>()
                / total0;
            let var_f = spec_power0
                .iter()
                .enumerate()
                .map(|(k, &p)| {
                    let f = if k < n / 2 {
                        k as f64 * df
                    } else {
                        (k as f64 - n as f64) * df
                    };
                    (f - mean_f).powi(2) * p
                })
                .sum::<f64>()
                / total0;
            var_f.sqrt()
        };

        // Strong SPM: γ=10 W⁻¹km⁻¹ = 1e-2 W⁻¹m⁻¹, β₂=0, L=1km, 200 steps
        let solver = SplitStepNls::new(1e-2, 0.0, 0.0, 1e3, 200);
        let a_out = solver.propagate(&a0, dt);

        let spec1 = fft_spm(&a_out);
        let spec_power1: Vec<f64> = spec1.iter().map(|v| v.norm_sqr()).collect();
        let total1: f64 = spec_power1.iter().sum();
        let rms_bw1 = {
            let mean_f = spec_power1
                .iter()
                .enumerate()
                .map(|(k, &p)| {
                    let f = if k < n / 2 {
                        k as f64 * df
                    } else {
                        (k as f64 - n as f64) * df
                    };
                    f * p
                })
                .sum::<f64>()
                / total1;
            let var_f = spec_power1
                .iter()
                .enumerate()
                .map(|(k, &p)| {
                    let f = if k < n / 2 {
                        k as f64 * df
                    } else {
                        (k as f64 - n as f64) * df
                    };
                    (f - mean_f).powi(2) * p
                })
                .sum::<f64>()
                / total1;
            var_f.sqrt()
        };

        assert!(
            rms_bw1 > rms_bw0,
            "SPM should broaden spectrum: bw0={rms_bw0:.3e} bw1={rms_bw1:.3e}"
        );
    }

    #[test]
    fn pulse_width_gaussian() {
        // A(t) = exp(-t²/(2T₀²))  →  |A(t)|² = exp(-t²/T₀²)
        // The RMS width of |A|² is σ = T₀/√2.
        let n = 512;
        let dt = 0.5e-12;
        let t0 = 10e-12; // field 1/e half-width
        let a = gaussian_pulse(n, dt, t0, 1.0);
        let rms = SplitStepNls::pulse_width_rms(&a, dt);
        let expected = t0 / 2.0_f64.sqrt(); // σ = T₀/√2
        let rel_err = (rms - expected).abs() / expected;
        assert!(
            rel_err < 0.02,
            "RMS width: expected {expected:.2e}, got {rms:.2e}, rel_err={rel_err:.3}"
        );
    }
}
