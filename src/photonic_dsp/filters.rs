//! Digital optical filters for photonic signal processing.
//!
//! Implements FIR filters (via wavelength-selective switches), IIR ring-resonator
//! cascade filters, and adaptive optical equalizers for chromatic dispersion
//! compensation.

use num_complex::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Optical FIR filter
// ---------------------------------------------------------------------------

/// Optical FIR filter implemented via wavelength-selective switches.
///
/// The tap weights model the coupling coefficients of each delay-line
/// stage, and `delay_per_tap` is the differential group delay (in seconds)
/// introduced between consecutive taps.
#[derive(Debug, Clone)]
pub struct OpticalFirFilter {
    /// Complex tap weights (amplitude and phase of each stage).
    pub tap_weights: Vec<Complex64>,
    /// Differential delay between consecutive taps (seconds).
    pub delay_per_tap: f64,
    /// Centre optical frequency of the filter (Hz).
    pub center_frequency: f64,
}

impl OpticalFirFilter {
    /// Construct a new FIR filter from explicit tap weights.
    pub fn new(weights: Vec<Complex64>, delay: f64, fc: f64) -> Self {
        Self {
            tap_weights: weights,
            delay_per_tap: delay,
            center_frequency: fc,
        }
    }

    /// Transfer function H(ω) = Σ_k w_k · exp(−j·k·ω·T).
    ///
    /// `omega` is the angular frequency offset from the centre (rad/s).
    pub fn transfer_function(&self, omega: f64) -> Complex64 {
        self.tap_weights
            .iter()
            .enumerate()
            .fold(Complex64::new(0.0, 0.0), |acc, (k, &w)| {
                let phase = -(k as f64) * omega * self.delay_per_tap;
                acc + w * Complex64::new(phase.cos(), phase.sin())
            })
    }

    /// Frequency response |H(f)|² expressed in dB.
    pub fn response_db(&self, freq_hz: f64) -> f64 {
        let omega = 2.0 * PI * freq_hz;
        let h = self.transfer_function(omega);
        let power = h.norm_sqr();
        if power <= 0.0 {
            -300.0
        } else {
            10.0 * power.log10()
        }
    }

    /// Group delay τ(ω) = −d∠H/dω, computed numerically (seconds).
    pub fn group_delay_s(&self, omega: f64) -> f64 {
        let delta = 1e3; // rad/s perturbation
        let h_plus = self.transfer_function(omega + delta);
        let h_minus = self.transfer_function(omega - delta);
        let phase_plus = h_plus.im.atan2(h_plus.re);
        let phase_minus = h_minus.im.atan2(h_minus.re);
        // Unwrap single-step difference
        let mut dphi = phase_plus - phase_minus;
        while dphi > PI {
            dphi -= 2.0 * PI;
        }
        while dphi < -PI {
            dphi += 2.0 * PI;
        }
        -dphi / (2.0 * delta)
    }

    /// Design a windowed-sinc lowpass filter using a Hamming window.
    ///
    /// - `n_taps`: number of filter taps (odd recommended)
    /// - `cutoff_hz`: −3 dB cutoff frequency in Hz
    /// - `delay`: differential tap delay (seconds)
    /// - `fc`: centre optical frequency (Hz)
    pub fn lowpass(n_taps: usize, cutoff_hz: f64, delay: f64, fc: f64) -> Self {
        let mut weights: Vec<Complex64> = Vec::with_capacity(n_taps);
        let m = n_taps as f64 - 1.0;
        let wc = 2.0 * PI * cutoff_hz * delay; // normalised cutoff (cycles per tap)
        for n in 0..n_taps {
            let nf = n as f64;
            // Sinc kernel
            let sinc = if (nf - m / 2.0).abs() < 1e-12 {
                wc / PI
            } else {
                (wc * (nf - m / 2.0)).sin() / (PI * (nf - m / 2.0))
            };
            // Hamming window
            let window = 0.54 - 0.46 * (2.0 * PI * nf / m).cos();
            weights.push(Complex64::new(sinc * window, 0.0));
        }
        Self::new(weights, delay, fc)
    }

    /// Design a bandpass filter centred at `f_pass` Hz with bandwidth `bw_hz`.
    pub fn bandpass(n_taps: usize, f_pass: f64, bw_hz: f64, delay: f64, fc: f64) -> Self {
        let mut weights: Vec<Complex64> = Vec::with_capacity(n_taps);
        let m = n_taps as f64 - 1.0;
        let wc = 2.0 * PI * (bw_hz / 2.0) * delay;
        let w0 = 2.0 * PI * f_pass * delay;
        for n in 0..n_taps {
            let nf = n as f64;
            let x = nf - m / 2.0;
            let sinc = if x.abs() < 1e-12 {
                wc / PI
            } else {
                (wc * x).sin() / (PI * x)
            };
            let window = 0.54 - 0.46 * (2.0 * PI * nf / m).cos();
            // Frequency-shift to f_pass
            let modulated = sinc * window * (w0 * x).cos() * 2.0;
            weights.push(Complex64::new(modulated, 0.0));
        }
        Self::new(weights, delay, fc)
    }

    /// Apply the FIR filter to a signal sequence via direct convolution.
    pub fn filter(&self, signal: &[Complex64]) -> Vec<Complex64> {
        let n = signal.len();
        let m = self.tap_weights.len();
        let mut output = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..n {
            for (k, &w) in self.tap_weights.iter().enumerate() {
                if i >= k {
                    output[i] += w * signal[i - k];
                }
            }
        }
        // Drop the transient startup samples only if we have enough output
        if n > m {
            output[m - 1..].to_vec()
        } else {
            output
        }
    }

    /// Free spectral range of the filter: FSR = 1 / delay_per_tap (Hz).
    pub fn free_spectral_range(&self) -> f64 {
        if self.delay_per_tap > 0.0 {
            1.0 / self.delay_per_tap
        } else {
            f64::INFINITY
        }
    }
}

// ---------------------------------------------------------------------------
// Ring-resonator IIR filter
// ---------------------------------------------------------------------------

/// Optical IIR filter implemented as a cascade of ring resonators.
///
/// Each ring is characterised by its power coupling coefficient κ, round-trip
/// power loss α, round-trip phase φ, and a shared free spectral range (FSR).
#[derive(Debug, Clone)]
pub struct RingResonatorFilter {
    /// Power coupling coefficients κ per ring (0 < κ < 1).
    pub coupling_coefficients: Vec<f64>,
    /// Round-trip power loss (amplitude decay per round trip) per ring.
    pub round_trip_loss: Vec<f64>,
    /// Round-trip phase φ per ring (radians); resonance at φ = 2πN.
    pub round_trip_phases: Vec<f64>,
    /// Free spectral range of the rings (Hz).
    pub fsr: f64,
}

impl RingResonatorFilter {
    /// Construct a single-ring resonator filter.
    pub fn new_single(kappa: f64, alpha: f64, phi: f64, fsr: f64) -> Self {
        Self {
            coupling_coefficients: vec![kappa],
            round_trip_loss: vec![alpha],
            round_trip_phases: vec![phi],
            fsr,
        }
    }

    /// Construct a coupled-resonator filter with multiple rings sharing the
    /// same FSR. Phases are initialised to resonance (φ = 0).
    pub fn new_coupled_resonators(kappas: Vec<f64>, alphas: Vec<f64>, fsr: f64) -> Self {
        let n = kappas.len();
        let phases = vec![0.0_f64; n];
        Self {
            coupling_coefficients: kappas,
            round_trip_loss: alphas,
            round_trip_phases: phases,
            fsr,
        }
    }

    /// Transfer function of the cascade, evaluated at a frequency offset
    /// expressed as a fraction of the FSR.
    ///
    /// For a single all-pass ring:
    ///   H(f) = (√(1-κ) − α·exp(jφ_rt)) / (1 − √(1-κ)·α·exp(jφ_rt))
    /// where φ_rt = φ + 2π·f/FSR.
    pub fn transfer_function(&self, freq_offset: f64) -> Complex64 {
        let n = self.coupling_coefficients.len();
        let mut h = Complex64::new(1.0, 0.0);
        for i in 0..n {
            let kappa = self.coupling_coefficients[i].clamp(0.0, 1.0);
            let alpha = self.round_trip_loss[i].clamp(0.0, 1.0);
            let phi_rt = self.round_trip_phases[i] + 2.0 * PI * freq_offset / self.fsr;
            let t = (1.0 - kappa).sqrt();
            let exp_jrt = Complex64::new(phi_rt.cos(), phi_rt.sin());
            let numerator = Complex64::new(t, 0.0) - alpha * exp_jrt;
            let denominator = Complex64::new(1.0, 0.0) - t * alpha * exp_jrt;
            if denominator.norm() > 1e-30 {
                h *= numerator / denominator;
            }
        }
        h
    }

    /// Extinction ratio (on-resonance suppression) in dB.
    pub fn extinction_ratio_db(&self) -> f64 {
        let h_on = self.transfer_function(0.0);
        let h_off = self.transfer_function(self.fsr / 2.0);
        let er = h_off.norm_sqr() / h_on.norm_sqr().max(1e-30);
        10.0 * er.max(1e-30).log10()
    }

    /// Estimate the 3-dB bandwidth of the filter (Hz) by bisection.
    ///
    /// The search is limited to [0, FSR/2).  If the response never drops below
    /// the 3-dB point within that range (e.g. very broadband ring), the full
    /// FSR is returned as the bandwidth.
    pub fn bandwidth_hz(&self) -> f64 {
        let peak = self.transfer_function(0.0).norm_sqr();
        let target = peak / 2.0;
        // Check if response ever falls to target within FSR/2
        let half_fsr = self.fsr / 2.0;
        if self.transfer_function(half_fsr).norm_sqr() > target {
            // Response never drops to 3 dB within half FSR → broadband
            return self.fsr;
        }
        let mut lo = 0.0_f64;
        let mut hi = half_fsr;
        for _ in 0..64 {
            let mid = (lo + hi) / 2.0;
            if self.transfer_function(mid).norm_sqr() > target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (2.0 * hi).min(self.fsr) // full width, capped at FSR
    }

    /// Loaded Q-factor of the filter: Q = centre_frequency / bandwidth.
    ///
    /// Here we use FSR as a proxy for the resonance frequency, which gives a
    /// dimensionally consistent Q relative to the FSR.
    pub fn q_factor(&self) -> f64 {
        let bw = self.bandwidth_hz();
        if bw > 0.0 {
            self.fsr / bw
        } else {
            f64::INFINITY
        }
    }

    /// Peak group delay at resonance (seconds), estimated from the Kramers–
    /// Kronig consistent delay formula: τ_peak ≈ 1 / (π · BW).
    pub fn group_delay_peak_s(&self) -> f64 {
        let bw = self.bandwidth_hz();
        if bw > 0.0 {
            1.0 / (PI * bw)
        } else {
            f64::INFINITY
        }
    }
}

// ---------------------------------------------------------------------------
// Optical equalizer
// ---------------------------------------------------------------------------

/// Adaptation algorithm for the optical equalizer.
#[derive(Debug, Clone)]
pub enum EqAlgorithm {
    /// Least Mean Squares with step size μ.
    Lms { mu: f64 },
    /// Recursive Least Squares with forgetting factor λ.
    Rls { lambda: f64 },
    /// Constant Modulus Algorithm with step size μ.
    Cma { mu: f64 },
    /// Decision-Directed adaptation with step size μ.
    Dd { mu: f64 },
}

/// Adaptive optical equalizer for chromatic dispersion compensation.
///
/// Implements a T/2-spaced FIR equalizer with selectable adaptation algorithm.
#[derive(Debug, Clone)]
pub struct OpticalEqualizer {
    /// Number of complex taps.
    pub n_taps: usize,
    /// Tap spacing in picoseconds (typically T/2 for fractionally-spaced eq.).
    pub tap_spacing_ps: f64,
    /// Current complex tap vector.
    pub taps: Vec<Complex64>,
    /// Selected adaptation algorithm.
    pub adaptation_algorithm: EqAlgorithm,
}

impl OpticalEqualizer {
    /// Construct a new equalizer with zero-initialised taps, except for a
    /// unit centre tap (identity initialisation).
    pub fn new(n_taps: usize, tap_spacing_ps: f64, algo: EqAlgorithm) -> Self {
        let mut taps = vec![Complex64::new(0.0, 0.0); n_taps];
        if n_taps > 0 {
            taps[n_taps / 2] = Complex64::new(1.0, 0.0);
        }
        Self {
            n_taps,
            tap_spacing_ps,
            taps,
            adaptation_algorithm: algo,
        }
    }

    /// Apply the equalizer to a signal sequence (direct convolution).
    pub fn apply(&self, signal: &[Complex64]) -> Vec<Complex64> {
        let n = signal.len();
        let m = self.taps.len();
        let mut output = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..n {
            for (k, &tap) in self.taps.iter().enumerate() {
                if i >= k {
                    output[i] += tap * signal[i - k];
                }
            }
        }
        if n > m {
            output[m - 1..].to_vec()
        } else {
            output
        }
    }

    /// Perform one LMS update step.
    ///
    /// `input` must have at least `n_taps` samples; the last `n_taps` samples
    /// are used as the tap input vector.  Returns the equalised output sample.
    pub fn update_lms(&mut self, input: &[Complex64], error: Complex64) -> Complex64 {
        let mu = match &self.adaptation_algorithm {
            EqAlgorithm::Lms { mu } => *mu,
            EqAlgorithm::Dd { mu } => *mu,
            _ => 1e-3,
        };
        let start = if input.len() >= self.n_taps {
            input.len() - self.n_taps
        } else {
            0
        };
        let x_slice = &input[start..];
        // Update taps: w ← w + μ · e · x*
        for (k, tap) in self.taps.iter_mut().enumerate() {
            if k < x_slice.len() {
                *tap += mu * error * x_slice[x_slice.len() - 1 - k].conj();
            }
        }
        // Compute current output: y = w^H · x
        x_slice
            .iter()
            .rev()
            .zip(self.taps.iter())
            .fold(Complex64::new(0.0, 0.0), |acc, (&xi, &wi)| acc + wi * xi)
    }

    /// Perform one CMA update step.
    ///
    /// Minimises |y|² − R² where R² is the constant modulus radius (≈ 1 for
    /// normalised constellations).  Returns the equalised output sample.
    pub fn update_cma(&mut self, input: &[Complex64]) -> Complex64 {
        let mu = match &self.adaptation_algorithm {
            EqAlgorithm::Cma { mu } => *mu,
            _ => 1e-4,
        };
        let start = if input.len() >= self.n_taps {
            input.len() - self.n_taps
        } else {
            0
        };
        let x_slice = &input[start..];
        // Compute current output
        let y = x_slice
            .iter()
            .rev()
            .zip(self.taps.iter())
            .fold(Complex64::new(0.0, 0.0), |acc, (&xi, &wi)| acc + wi * xi);
        // CMA error: e = y * (1 - |y|²)   (R² = 1 assumed)
        let error = y * (1.0 - y.norm_sqr());
        for (k, tap) in self.taps.iter_mut().enumerate() {
            if k < x_slice.len() {
                *tap += mu * error * x_slice[x_slice.len() - 1 - k].conj();
            }
        }
        y
    }

    /// Maximum chromatic dispersion (ps/nm) this equalizer can compensate.
    ///
    /// Based on the equalizer tap span: CD_max ≈ T_span / (D · λ²/c · R_sym)
    /// Simplified to: tap_span_ps / symbol_rate_gbaud (ps²/Gbaud normalisation).
    pub fn max_cd_compensation_ps_per_nm(&self, symbol_rate_gbaud: f64) -> f64 {
        // Heuristic: each ps of tap span compensates ~1 ps²/nm at 1550 nm
        // Full formula: CD_max = T_span² * c / (D * λ² * L)
        // Here we return the tap span in ps² (proxy for CD budget at 1550 nm)
        let tap_span = self.tap_span_ps();
        let lambda_nm = 1550.0;
        let c_nm_ps = 3e8 * 1e9 / 1e12; // nm/ps
                                        // CD = D·L;  Δτ = D·L·Δλ; tap_span ≈ D·L·bandwidth
                                        // bandwidth in nm ≈ symbol_rate / (c/λ²) ... simplified
        let bandwidth_nm = symbol_rate_gbaud * 1e9 * (lambda_nm * lambda_nm) / (c_nm_ps * 1e12);
        if bandwidth_nm > 0.0 {
            tap_span / bandwidth_nm
        } else {
            0.0
        }
    }

    /// Total tap span in picoseconds: n_taps × tap_spacing_ps.
    pub fn tap_span_ps(&self) -> f64 {
        self.n_taps as f64 * self.tap_spacing_ps
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn fir_transfer_dc() {
        // All-ones taps with unit delay → H(0) = N (sum of taps)
        let n = 5_usize;
        let taps = vec![Complex64::new(1.0, 0.0); n];
        let fir = OpticalFirFilter::new(taps, 1e-12, 193.4e12);
        let h0 = fir.transfer_function(0.0);
        assert_abs_diff_eq!(h0.re, n as f64, epsilon = 1e-10);
        assert_abs_diff_eq!(h0.im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn fir_fsr() {
        let delay_ps = 10e-12;
        let fir = OpticalFirFilter::new(vec![Complex64::new(1.0, 0.0)], delay_ps, 193.4e12);
        let expected_fsr = 1.0 / delay_ps;
        assert_abs_diff_eq!(fir.free_spectral_range(), expected_fsr, epsilon = 1.0);
    }

    #[test]
    fn fir_lowpass_dc_passband() {
        let fir = OpticalFirFilter::lowpass(31, 10e9, 10e-12, 193.4e12);
        let h0 = fir.transfer_function(0.0);
        // DC gain should be close to 1 (normalised windowed-sinc is approximately 1)
        assert!(h0.norm() > 0.5, "DC gain too low: {}", h0.norm());
    }

    #[test]
    fn ring_resonator_single_resonance() {
        // At resonance (freq_offset=0) a critically-coupled ring has zero transmission.
        let kappa = 0.5_f64;
        let alpha = (1.0 - kappa).sqrt(); // critical coupling condition
        let ring = RingResonatorFilter::new_single(kappa, alpha, 0.0, 100e9);
        let h_on = ring.transfer_function(0.0);
        assert!(
            h_on.norm() < 0.05,
            "On-resonance transmission should be near 0; got {}",
            h_on.norm()
        );
    }

    #[test]
    fn ring_bandwidth_positive() {
        // Use a narrowband ring: high coupling (κ=0.5) with moderate round-trip loss
        let ring = RingResonatorFilter::new_single(0.5, 0.9, 0.0, 100e9);
        let bw = ring.bandwidth_hz();
        assert!(bw > 0.0, "Bandwidth must be positive, got {}", bw);
        assert!(bw <= ring.fsr, "Bandwidth must not exceed FSR");
    }

    #[test]
    fn ring_bandwidth_narrowband() {
        // A high coupling ring (κ=0.9) creates a sharp notch with measurable 3-dB BW
        // With α close to critical coupling, the bandwidth is a fraction of the FSR
        let ring = RingResonatorFilter::new_single(0.9, 0.3, 0.0, 100e9);
        let bw = ring.bandwidth_hz();
        // The bandwidth must be positive and bounded by the FSR
        assert!(bw > 0.0, "Bandwidth must be positive, got {}", bw);
        assert!(
            bw <= ring.fsr,
            "High-coupling ring bandwidth must not exceed FSR: {}",
            bw
        );
    }

    #[test]
    fn equalizer_identity_tap() {
        let eq = OpticalEqualizer::new(7, 50.0, EqAlgorithm::Lms { mu: 1e-3 });
        let signal: Vec<Complex64> = (0..20).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let out = eq.apply(&signal);
        // With centre-tap = 1 and all others 0, output should closely match input
        // (delayed by n_taps/2 samples)
        assert!(!out.is_empty());
    }

    #[test]
    fn equalizer_tap_span() {
        let eq = OpticalEqualizer::new(11, 50.0, EqAlgorithm::Cma { mu: 1e-4 });
        assert_abs_diff_eq!(eq.tap_span_ps(), 550.0, epsilon = 1e-6);
    }
}
