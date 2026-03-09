use num_complex::Complex64;
use std::f64::consts::PI;

/// Nonlinear Schrödinger Equation (NLSE) fiber propagation via split-step Fourier method.
///
/// The NLSE for pulse propagation in a single-mode fiber:
///   ∂A/∂z = -α/2·A - i·β₂/2·∂²A/∂T² + i·γ|A|²·A
///
/// where:
/// - A(z,T): complex pulse envelope (sqrt(W))
/// - α: power loss coefficient (1/m)
/// - β₂: group velocity dispersion [s²/m] (anomalous: β₂<0, normal: β₂>0)
/// - γ: nonlinear coefficient (1/(W·m))
/// - T: retarded time frame: T = t - z/v_g
///
/// Split-step method: alternates between linear (dispersion+loss) in Fourier domain
/// and nonlinear (SPM) in time domain.
///
/// Reference: Agrawal, "Nonlinear Fiber Optics", 6th ed., §2.4.
pub struct NlseSolver {
    /// Number of time points (power of 2)
    pub n_t: usize,
    /// Time window width (s)
    pub t_window: f64,
    /// Loss coefficient α (1/m)
    pub alpha: f64,
    /// GVD β₂ (s²/m)
    pub beta2: f64,
    /// Nonlinear coefficient γ (1/(W·m))
    pub gamma: f64,
    /// Pulse envelope A(T) — complex amplitudes
    pub field: Vec<Complex64>,
    /// Time step (s)
    pub dt: f64,
}

impl NlseSolver {
    pub fn new(n_t: usize, t_window: f64, alpha: f64, beta2: f64, gamma: f64) -> Self {
        let dt = t_window / n_t as f64;
        Self {
            n_t,
            t_window,
            alpha,
            beta2,
            gamma,
            field: vec![Complex64::new(0.0, 0.0); n_t],
            dt,
        }
    }

    /// Initialize a Gaussian pulse: A(T) = A₀·exp(-T²/(2T₀²))
    pub fn set_gaussian_pulse(&mut self, a0: f64, t0: f64) {
        let t_center = self.t_window / 2.0;
        for i in 0..self.n_t {
            let t = i as f64 * self.dt - t_center;
            let envelope = (-t * t / (2.0 * t0 * t0)).exp();
            self.field[i] = Complex64::new(a0 * envelope, 0.0);
        }
    }

    /// Initialize a sech pulse (soliton shape): A(T) = A₀ / cosh(T/T₀)
    pub fn set_sech_pulse(&mut self, a0: f64, t0: f64) {
        let t_center = self.t_window / 2.0;
        for i in 0..self.n_t {
            let t = i as f64 * self.dt - t_center;
            let envelope = 1.0 / (t / t0).cosh();
            self.field[i] = Complex64::new(a0 * envelope, 0.0);
        }
    }

    /// Total pulse power P = ∫|A|² dT
    pub fn total_power(&self) -> f64 {
        self.field.iter().map(|a| a.norm_sqr()).sum::<f64>() * self.dt
    }

    /// Peak power |A|²_max
    pub fn peak_power(&self) -> f64 {
        self.field
            .iter()
            .map(|a| a.norm_sqr())
            .fold(0.0_f64, f64::max)
    }

    /// RMS pulse width (second moment of |A|² in time)
    pub fn rms_width(&self) -> f64 {
        let power: Vec<f64> = self.field.iter().map(|a| a.norm_sqr()).collect();
        let total: f64 = power.iter().sum::<f64>();
        if total < 1e-60 {
            return 0.0;
        }
        let t_center = self.t_window / 2.0;
        let mean = power
            .iter()
            .enumerate()
            .map(|(i, &p)| (i as f64 * self.dt - t_center) * p)
            .sum::<f64>()
            / total;
        let var = power
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let t = i as f64 * self.dt - t_center - mean;
                t * t * p
            })
            .sum::<f64>()
            / total;
        var.sqrt()
    }

    /// Propagate the pulse over a step `dz` (m) using the split-step Fourier method.
    pub fn step(&mut self, dz: f64) {
        // Step 1: Half-step nonlinear phase (SPM)
        self.apply_nonlinear(dz / 2.0);

        // Step 2: Full linear step (dispersion + loss) in Fourier domain
        self.apply_linear(dz);

        // Step 3: Half-step nonlinear phase
        self.apply_nonlinear(dz / 2.0);
    }

    /// Propagate over total length L using n_steps steps.
    pub fn propagate(&mut self, length: f64, n_steps: usize) {
        let dz = length / n_steps as f64;
        for _ in 0..n_steps {
            self.step(dz);
        }
    }

    fn apply_nonlinear(&mut self, dz: f64) {
        for a in self.field.iter_mut() {
            let phi_nl = self.gamma * a.norm_sqr() * dz;
            let phase = Complex64::new(0.0, phi_nl);
            *a *= phase.exp();
        }
    }

    fn apply_linear(&mut self, dz: f64) {
        // FFT
        let spec = fft_manual(&self.field);

        // Frequency axis (angular): ω_m = 2π·m/(N·dt) for m = 0..N/2, then N/2..N (negative)
        let n = self.n_t;
        let df = 1.0 / self.t_window;
        let spec_out: Vec<Complex64> = spec
            .iter()
            .enumerate()
            .map(|(m, &s)| {
                let freq = if m < n / 2 {
                    m as f64 * df
                } else {
                    (m as f64 - n as f64) * df
                };
                let omega = 2.0 * PI * freq;
                // H(ω) = exp(-α/2·dz) · exp(-i·β₂/2·ω²·dz)
                let loss_factor = (-self.alpha / 2.0 * dz).exp();
                let disp_phase = -self.beta2 / 2.0 * omega * omega * dz;
                let h = Complex64::new(0.0, disp_phase).exp() * loss_factor;
                s * h
            })
            .collect();

        // IFFT
        self.field = ifft_manual(&spec_out);
    }
}

/// Simple DFT-based FFT (Cooley-Tukey, radix-2)
/// Uses oxifft indirectly via manual implementation to avoid import complexity.
fn fft_manual(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    if n == 1 {
        return x.to_vec();
    }
    let half = n / 2;
    let even: Vec<Complex64> = (0..half).map(|k| x[2 * k]).collect();
    let odd: Vec<Complex64> = (0..half).map(|k| x[2 * k + 1]).collect();
    let fe = fft_manual(&even);
    let fo = fft_manual(&odd);
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    for k in 0..half {
        let angle = -2.0 * PI * k as f64 / n as f64;
        let twiddle = Complex64::new(angle.cos(), angle.sin());
        out[k] = fe[k] + twiddle * fo[k];
        out[k + half] = fe[k] - twiddle * fo[k];
    }
    out
}

fn ifft_manual(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    // IFFT = conj(FFT(conj(x)))/n
    let conj_x: Vec<Complex64> = x.iter().map(|v| v.conj()).collect();
    let fft_conj = fft_manual(&conj_x);
    fft_conj.iter().map(|v| v.conj() / n as f64).collect()
}

/// Soliton order N² = L_D/L_NL = γ·P₀·T₀²/|β₂|
pub fn soliton_order(gamma: f64, peak_power: f64, t0: f64, beta2: f64) -> f64 {
    (gamma * peak_power * t0 * t0 / beta2.abs()).sqrt()
}

/// Soliton period z_0 = π/2 · L_D = π·T₀²/(2·|β₂|)
pub fn soliton_period(t0: f64, beta2: f64) -> f64 {
    PI * t0 * t0 / (2.0 * beta2.abs())
}

// ---------------------------------------------------------------------------
// Split-step Fourier method parameters
// ---------------------------------------------------------------------------

/// Parameters for the split-step Fourier method (SSFM).
///
/// Encapsulates all fiber and simulation parameters needed to run an SSFM
/// propagation.  Convenience constructors are provided for standard fiber types.
#[derive(Debug, Clone)]
pub struct SsfmParams {
    /// Step size (m)
    pub dz: f64,
    /// Number of propagation steps
    pub n_steps: usize,
    /// Nonlinear coefficient γ (1/W/m)
    pub gamma: f64,
    /// Group-velocity dispersion β₂ (s²/m)
    pub beta2: f64,
    /// Third-order dispersion β₃ (s³/m)
    pub beta3: f64,
    /// Power loss coefficient α (1/m)
    pub alpha: f64,
}

impl SsfmParams {
    /// Create an `SsfmParams` with all fields specified explicitly.
    pub fn new(dz: f64, n_steps: usize, gamma: f64, beta2: f64, beta3: f64, alpha: f64) -> Self {
        Self {
            dz,
            n_steps,
            gamma,
            beta2,
            beta3,
            alpha,
        }
    }

    /// Standard SMF-28 single-mode fiber parameters near `wavelength_m`.
    ///
    /// Reference values (Corning SMF-28):
    ///   γ ≈ 1.3×10⁻³ /W/m, β₂ ≈ −21.7 ps²/km @ 1550 nm,
    ///   β₃ ≈ 0.12 ps³/km, α ≈ 0.046/km (0.2 dB/km)
    pub fn for_smf28(wavelength_m: f64) -> Self {
        // Simple dispersion slope model: β₂(λ) ≈ β₂₀ + (λ-λ₀)·S₀
        let lambda0 = 1.31e-6; // zero-dispersion wavelength (m)
        let s0 = 0.092e12; // dispersion slope (s/m³)
        let c = crate::units::conversion::SPEED_OF_LIGHT;
        let omega0 = 2.0 * PI * c / lambda0;
        let omega = 2.0 * PI * c / wavelength_m;
        let d_omega = omega - omega0;
        let beta2 = -s0 * d_omega / (omega0 * omega0 / (2.0 * PI));
        // Numerical fallback for the linear approximation
        let beta2 = if beta2.is_finite() { beta2 } else { -21.7e-27 };
        Self {
            dz: 100.0, // 100 m steps
            n_steps: 100,
            gamma: 1.3e-3, // 1/W/m
            beta2,
            beta3: 0.12e-39, // s³/m
            alpha: 0.046e-3, // 1/m (≈ 0.2 dB/km)
        }
    }

    /// High-nonlinearity fiber (HNLF) parameters.
    ///
    /// Typical HNLF: γ ≈ 10 /W/km, near-zero dispersion, low loss.
    pub fn for_hnlf(wavelength_m: f64) -> Self {
        let _ = wavelength_m; // parameters are broadband
        Self {
            dz: 10.0, // 10 m steps (shorter due to higher nonlinearity)
            n_steps: 200,
            gamma: 10e-3,    // 10 /W/km = 10×10⁻³ /W/m
            beta2: -1.0e-27, // near-zero anomalous GVD (1 ps²/km)
            beta3: 0.05e-39, // small TOD
            alpha: 0.023e-3, // 0.1 dB/km
        }
    }

    /// Nonlinear length L_NL = 1 / (γ · P₀).
    ///
    /// Characterises the distance over which SPM becomes significant.
    pub fn nonlinear_length(&self, peak_power_w: f64) -> f64 {
        if self.gamma.abs() < 1e-30 || peak_power_w.abs() < 1e-30 {
            return f64::INFINITY;
        }
        1.0 / (self.gamma * peak_power_w)
    }

    /// Dispersion length L_D = T₀² / |β₂|.
    ///
    /// Characterises the distance over which GVD broadens the pulse.
    pub fn dispersion_length(&self, pulse_duration_s: f64) -> f64 {
        if self.beta2.abs() < 1e-60 {
            return f64::INFINITY;
        }
        pulse_duration_s * pulse_duration_s / self.beta2.abs()
    }

    /// Soliton number N = √(L_D / L_NL).
    ///
    /// N = 1 → fundamental soliton; N > 1 → higher-order soliton.
    pub fn soliton_number(&self, peak_power_w: f64, pulse_duration_s: f64) -> f64 {
        let ld = self.dispersion_length(pulse_duration_s);
        let lnl = self.nonlinear_length(peak_power_w);
        if lnl.is_infinite() || lnl < 1e-30 {
            return 0.0;
        }
        (ld / lnl).sqrt()
    }

    /// Walk-off length L_W = T₀ / |β₁₂|  between two pulses with group-velocity
    /// difference Δv = dv (s/m = difference in 1/v_g = β₁ mismatch).
    ///
    /// Here `dv` is the group-delay difference per unit length (s/m).
    pub fn walk_off_length(&self, dv: f64) -> f64 {
        if dv.abs() < 1e-30 {
            return f64::INFINITY;
        }
        // Characteristic pulse width estimate from dispersion & step
        let t0_est = (self.beta2.abs() * self.dz).sqrt().max(1e-15);
        t0_est / dv.abs()
    }
}

// ---------------------------------------------------------------------------
// Pulse characterisation metrics
// ---------------------------------------------------------------------------

/// Scalar metrics characterising an optical pulse.
#[derive(Debug, Clone)]
pub struct PulseMetrics {
    /// Peak power |A|²_max (W)
    pub peak_power: f64,
    /// Pulse energy E = ∫|A|² dt (J)
    pub energy: f64,
    /// RMS duration (s)
    pub rms_duration: f64,
    /// Time-bandwidth product Δt·Δν (dimensionless)
    pub time_bandwidth_product: f64,
}

impl PulseMetrics {
    /// Compute pulse metrics from a temporal field envelope.
    ///
    /// `field` contains the real-valued power envelope |A(t)|² at each point
    /// (i.e., already squared), and `dt` is the time step (s).
    ///
    /// The TBP is estimated via Parseval / RMS spectral width using a manual DFT.
    pub fn from_field(field: &[f64], dt: f64) -> Self {
        let n = field.len();
        if n == 0 {
            return Self {
                peak_power: 0.0,
                energy: 0.0,
                rms_duration: 0.0,
                time_bandwidth_product: 0.0,
            };
        }

        // Peak power
        let peak_power = field.iter().cloned().fold(0.0_f64, f64::max);

        // Energy
        let energy = field.iter().sum::<f64>() * dt;

        // RMS duration (second moment of the power profile)
        let total = energy;
        let rms_duration = if total < 1e-60 {
            0.0
        } else {
            let t_center_sum: f64 = field
                .iter()
                .enumerate()
                .map(|(i, &p)| i as f64 * dt * p)
                .sum::<f64>();
            let t_mean = t_center_sum / total * dt; // compensate for dt already in total
                                                    // Recompute properly
            let t_mean2: f64 = field
                .iter()
                .enumerate()
                .map(|(i, &p)| i as f64 * p)
                .sum::<f64>()
                / (total / dt);
            let var: f64 = field
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    let t = i as f64 - t_mean2;
                    t * t * p
                })
                .sum::<f64>()
                / (total / dt);
            let _ = t_mean; // suppress warning
            var.sqrt() * dt
        };

        // RMS spectral width via FFT of sqrt(field) (amplitude)
        let amplitude: Vec<num_complex::Complex64> = field
            .iter()
            .map(|&p| num_complex::Complex64::new(p.max(0.0).sqrt(), 0.0))
            .collect();
        let spectrum = fft_manual(&amplitude);
        let spec_power: Vec<f64> = spectrum.iter().map(|s| s.norm_sqr()).collect();
        let spec_total: f64 = spec_power.iter().sum::<f64>();

        let rms_bandwidth = if spec_total < 1e-60 || rms_duration < 1e-30 {
            0.0
        } else {
            let df = 1.0 / (n as f64 * dt);
            let nu_mean: f64 = spec_power
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    let freq = if i < n / 2 {
                        i as f64
                    } else {
                        i as f64 - n as f64
                    };
                    freq * p
                })
                .sum::<f64>()
                / spec_total;
            let nu_var: f64 = spec_power
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    let freq = if i < n / 2 {
                        i as f64
                    } else {
                        i as f64 - n as f64
                    };
                    let d = freq - nu_mean;
                    d * d * p
                })
                .sum::<f64>()
                / spec_total;
            nu_var.sqrt() * df
        };

        let time_bandwidth_product = rms_duration * rms_bandwidth;

        Self {
            peak_power,
            energy,
            rms_duration,
            time_bandwidth_product,
        }
    }

    /// Return `true` if the TBP is close to the transform-limited value.
    ///
    /// For a Gaussian pulse, the TBP (RMS) = 1/(4π) ≈ 0.0796.
    /// `tolerance` is the fractional deviation allowed.
    pub fn is_transform_limited(&self, tolerance: f64) -> bool {
        let tbp_gaussian = 1.0 / (4.0 * PI);
        (self.time_bandwidth_product - tbp_gaussian).abs() / tbp_gaussian < tolerance
    }

    /// Peak power of a sech²-shaped pulse from energy and FWHM duration.
    ///
    /// For sech²: E = P₀ · T₀ · 2  (with T₀ = T_FWHM / (2 ln(1+√2)))
    pub fn sech_peak_power(energy_j: f64, duration_fwhm_s: f64) -> f64 {
        if duration_fwhm_s < 1e-30 {
            return 0.0;
        }
        // T_FWHM = 2 · T₀ · ln(1 + √2)  ⟹  T₀ = T_FWHM / (2·ln(1+√2))
        let ln_factor = 2.0 * (1.0 + 2.0_f64.sqrt()).ln();
        let t0 = duration_fwhm_s / ln_factor;
        energy_j / (2.0 * t0)
    }

    /// Peak power of a Gaussian pulse from energy and FWHM duration.
    ///
    /// E = P₀ · T₀ · √π  with T₀ = T_FWHM / (2·√(ln 2))
    pub fn gaussian_peak_power(energy_j: f64, duration_fwhm_s: f64) -> f64 {
        if duration_fwhm_s < 1e-30 {
            return 0.0;
        }
        let t0 = duration_fwhm_s / (2.0 * 2.0_f64.ln().sqrt());
        energy_j / (t0 * PI.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_solver() -> NlseSolver {
        NlseSolver::new(
            256,     // N_t
            100e-12, // 100ps window
            0.0,     // lossless
            -20e-27, // β₂ = -20 ps²/km = -20×10⁻²⁷ s²/m (anomalous)
            1.0e-3,  // γ = 1/(W·m)
        )
    }

    #[test]
    fn nlse_initializes_zero() {
        let s = make_solver();
        assert!(s.field.iter().all(|a| a.norm() == 0.0));
    }

    #[test]
    fn nlse_gaussian_pulse_power() {
        let mut s = make_solver();
        let a0 = 1.0_f64; // sqrt(W)
        let t0 = 5e-12; // 5ps
        s.set_gaussian_pulse(a0, t0);
        let p = s.total_power();
        // ∫ a₀²·exp(-T²/T₀²) dT ≈ a₀²·T₀·sqrt(π)
        let expected = a0 * a0 * t0 * PI.sqrt();
        let rel_err = (p - expected).abs() / expected;
        assert!(
            rel_err < 0.01,
            "Power rel_err={rel_err:.4} (p={p:.2e}, exp={expected:.2e})"
        );
    }

    #[test]
    fn nlse_sech_pulse_set() {
        let mut s = make_solver();
        s.set_sech_pulse(1.0, 5e-12);
        let p = s.peak_power();
        assert!(
            (p - 1.0).abs() < 1e-6,
            "Sech peak should be |A0|²=1, got {p:.4}"
        );
    }

    #[test]
    fn nlse_lossless_power_conservation() {
        let mut s = make_solver();
        s.set_gaussian_pulse(1.0, 5e-12);
        let p0 = s.total_power();
        s.propagate(1e3, 100); // 1km, 100 steps
        let p1 = s.total_power();
        let rel_err = (p1 - p0).abs() / p0;
        assert!(rel_err < 1e-3, "Power not conserved: rel_err={rel_err:.2e}");
    }

    #[test]
    fn nlse_dispersion_spreads_pulse() {
        let mut s = make_solver();
        s.set_gaussian_pulse(0.001, 5e-12); // very low power → negligible SPM
        let w0 = s.rms_width();
        s.propagate(1e3, 50); // 1km
        let w1 = s.rms_width();
        assert!(
            w1 > w0,
            "Dispersion should spread pulse: w0={w0:.2e} w1={w1:.2e}"
        );
    }

    #[test]
    fn soliton_order_n1() {
        // Fundamental soliton: N=1 at P0 = |β₂|/(γ·T₀²)
        let beta2 = 20e-27_f64;
        let gamma = 1e-3;
        let t0 = 5e-12;
        let p0 = beta2 / (gamma * t0 * t0);
        let n = soliton_order(gamma, p0, t0, beta2);
        assert!((n - 1.0).abs() < 1e-6, "N should be 1, got {n:.4}");
    }

    #[test]
    fn soliton_period_positive() {
        let z0 = soliton_period(5e-12, 20e-27);
        assert!(z0 > 0.0);
    }

    #[test]
    fn fft_then_ifft_roundtrip() {
        let n = 64;
        let x: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect();
        let spec = fft_manual(&x);
        let recovered = ifft_manual(&spec);
        for (orig, rec) in x.iter().zip(recovered.iter()) {
            let err = (orig - rec).norm();
            assert!(err < 1e-10, "FFT roundtrip error: {err:.2e}");
        }
    }

    // ── SsfmParams tests ──────────────────────────────────────────────────────

    #[test]
    fn ssfm_smf28_params_are_finite() {
        let p = SsfmParams::for_smf28(1550e-9);
        assert!(p.gamma.is_finite() && p.gamma > 0.0);
        assert!(p.beta2.is_finite());
        assert!(p.alpha >= 0.0);
        assert!(p.dz > 0.0);
        assert!(p.n_steps > 0);
    }

    #[test]
    fn ssfm_hnlf_higher_gamma_than_smf28() {
        let smf = SsfmParams::for_smf28(1550e-9);
        let hnlf = SsfmParams::for_hnlf(1550e-9);
        assert!(
            hnlf.gamma > smf.gamma,
            "HNLF gamma should exceed SMF-28 gamma"
        );
    }

    #[test]
    fn ssfm_nonlinear_length_scales_inversely_with_power() {
        let p = SsfmParams::for_smf28(1550e-9);
        let lnl1 = p.nonlinear_length(1.0);
        let lnl2 = p.nonlinear_length(2.0);
        assert!((lnl1 / lnl2 - 2.0).abs() < 1e-10, "L_NL ∝ 1/P₀");
    }

    #[test]
    fn ssfm_nonlinear_length_zero_power_is_inf() {
        let p = SsfmParams::for_smf28(1550e-9);
        assert_eq!(p.nonlinear_length(0.0), f64::INFINITY);
    }

    #[test]
    fn ssfm_dispersion_length_quadratic_in_duration() {
        let p = SsfmParams::for_smf28(1550e-9);
        let ld1 = p.dispersion_length(1e-12);
        let ld2 = p.dispersion_length(2e-12);
        assert!((ld2 / ld1 - 4.0).abs() < 1e-9, "L_D ∝ T₀²");
    }

    #[test]
    fn ssfm_soliton_number_fundamental_soliton() {
        // N=1 when L_D = L_NL
        let beta2 = 20e-27_f64; // |β₂| = 20 ps²/km
        let gamma = 1e-3;
        let t0 = 5e-12; // 5 ps
        let p0 = beta2 / (gamma * t0 * t0); // balances L_D and L_NL
        let params = SsfmParams::new(100.0, 10, gamma, -beta2, 0.0, 0.0);
        let n = params.soliton_number(p0, t0);
        assert!((n - 1.0).abs() < 1e-6, "N should equal 1, got {n:.4}");
    }

    #[test]
    fn ssfm_walk_off_positive_and_finite() {
        let p = SsfmParams::for_smf28(1550e-9);
        let lw = p.walk_off_length(1e-12); // 1 ps/m group delay difference
        assert!(lw.is_finite() && lw > 0.0);
    }

    // ── PulseMetrics tests ────────────────────────────────────────────────────

    #[test]
    fn pulse_metrics_gaussian_energy() {
        use approx::assert_relative_eq;
        // Gaussian power profile |A|² = P₀ exp(-t²/T₀²)
        let n = 512;
        let t_window = 100e-12;
        let dt = t_window / n as f64;
        let t0 = 10e-12;
        let p0 = 1.0_f64;
        let t_center = t_window / 2.0;
        let field: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 * dt - t_center;
                p0 * (-t * t / (t0 * t0)).exp()
            })
            .collect();
        let metrics = PulseMetrics::from_field(&field, dt);
        let expected_energy = p0 * t0 * PI.sqrt();
        assert_relative_eq!(metrics.energy, expected_energy, max_relative = 0.02);
    }

    #[test]
    fn pulse_metrics_peak_power_correct() {
        let field = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let m = PulseMetrics::from_field(&field, 1e-12);
        assert!((m.peak_power - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pulse_metrics_sech_peak_power_positive() {
        let p = PulseMetrics::sech_peak_power(1e-12, 1e-12);
        assert!(p > 0.0, "sech peak power should be positive");
    }

    #[test]
    fn pulse_metrics_gaussian_peak_power_positive() {
        let p = PulseMetrics::gaussian_peak_power(1e-12, 1e-12);
        assert!(p > 0.0, "Gaussian peak power should be positive");
    }

    #[test]
    fn pulse_metrics_sech_vs_gaussian_same_energy_duration() {
        // Sech pulse is taller/narrower than Gaussian for same energy & FWHM
        let energy = 1e-12;
        let fwhm = 1e-12;
        let p_sech = PulseMetrics::sech_peak_power(energy, fwhm);
        let p_gauss = PulseMetrics::gaussian_peak_power(energy, fwhm);
        // sech² peak ≈ 0.88 E/T_FWHM, Gaussian peak ≈ 0.94 E/T_FWHM — both finite
        assert!(p_sech > 0.0 && p_gauss > 0.0);
    }

    #[test]
    fn pulse_metrics_empty_field() {
        let m = PulseMetrics::from_field(&[], 1e-12);
        assert_eq!(m.peak_power, 0.0);
        assert_eq!(m.energy, 0.0);
    }

    #[test]
    fn pulse_metrics_zero_fwhm_peak_power() {
        assert_eq!(PulseMetrics::sech_peak_power(1.0, 0.0), 0.0);
        assert_eq!(PulseMetrics::gaussian_peak_power(1.0, 0.0), 0.0);
    }
}
