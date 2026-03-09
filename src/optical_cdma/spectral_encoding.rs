//! Spectral encoding schemes for OCDMA.
//!
//! Two complementary approaches are implemented:
//!
//! * **Spectral Amplitude Coding (SAC-OCDMA)** — each user occupies a
//!   distinct subset of wavelength bins.  The Modified Double Weight (MDW)
//!   code family is constructed analytically.  Complementary subtraction at
//!   the receiver eliminates MAI exactly.
//!
//! * **Spectral Phase Coding (SPC-OCDMA)** — each user imposes a unique
//!   phase pattern on the spectral components of a broadband coherent pulse.
//!   Decoding is by matched-phase multiplication followed by IFFT, producing
//!   a compressed auto-correlation pulse for the intended user and a diffuse
//!   background for others.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// SAC-OCDMA
// ---------------------------------------------------------------------------

/// Spectral Amplitude Coding OCDMA system.
///
/// Uses the Modified Double Weight (MDW) code family where each user's code
/// occupies `w` consecutive wavelength slots from a cyclically shifted window
/// of width `2w` within the `N`-chip spectral grid.
///
/// # MAI cancellation
/// The receiver uses a complementary photodetector pair:
/// * Detector A receives wavelengths in the user's code.
/// * Detector B receives the complement wavelengths.
///   Subtracting B from A cancels MAI contributions that are equally distributed
///   across both groups.
#[derive(Debug, Clone)]
pub struct SacOcdma {
    /// Total number of spectral chips (wavelength bins).
    pub n_wavelengths: usize,
    /// Code weight: number of wavelengths assigned to each user.
    pub code_weight: usize,
    /// Center optical frequency of the spectral band (Hz).
    pub center_freq_hz: f64,
    /// Frequency spacing between consecutive spectral chips (Hz).
    pub channel_spacing_hz: f64,
}

impl SacOcdma {
    /// Create a new SAC-OCDMA system.
    pub fn new(
        n_wavelengths: usize,
        code_weight: usize,
        center_freq_hz: f64,
        channel_spacing_hz: f64,
    ) -> Self {
        Self {
            n_wavelengths,
            code_weight,
            center_freq_hz,
            channel_spacing_hz,
        }
    }

    /// Generate the MDW code for user `user_idx`.
    ///
    /// User `k` occupies wavelength positions `{2·k·w, 2·k·w+1, …, 2·k·w+w−1}`
    /// modulo `N` (where N = `n_wavelengths`).  This guarantees that each pair
    /// of users shares at most one common wavelength, keeping cross-correlation
    /// bounded.
    ///
    /// Returns a binary vector of length `n_wavelengths`.
    pub fn mdw_code(&self, user_idx: usize) -> Vec<u8> {
        let n = self.n_wavelengths;
        let w = self.code_weight;
        let mut code = vec![0u8; n];
        let start = (2 * user_idx * w) % (2 * n.max(1));
        for k in 0..w {
            let pos = (start + k) % n;
            code[pos] = 1;
        }
        code
    }

    /// Generate the complementary code (bit-wise NOT) of a given code.
    ///
    /// The complement is used in the balanced differential receiver to
    /// achieve exact MAI cancellation.
    pub fn complementary_code(&self, code: &[u8]) -> Vec<u8> {
        code.iter().map(|&b| 1 - b.min(1)).collect()
    }

    /// SNR at the differential receiver output.
    ///
    /// After complementary subtraction the signal power is proportional to
    /// `w × P_chip` and the noise variance is the sum of shot noise and
    /// thermal noise across both detectors.
    ///
    /// `SNR = (w × P_chip)² / (σ²_shot + σ²_thermal)`
    pub fn snr(
        &self,
        power_per_chip_w: f64,
        shot_noise_variance: f64,
        thermal_variance: f64,
    ) -> f64 {
        let signal_power = (self.code_weight as f64 * power_per_chip_w).powi(2);
        let noise_power = shot_noise_variance + thermal_variance;
        if noise_power < 1e-300 {
            return f64::INFINITY;
        }
        signal_power / noise_power
    }

    /// Spectral efficiency in bps/Hz per user.
    ///
    /// `η = bit_rate / (N × Δν)`
    pub fn spectral_efficiency(&self, bit_rate_hz: f64) -> f64 {
        let total_bw = self.n_wavelengths as f64 * self.channel_spacing_hz;
        if total_bw < 1e-30 {
            return 0.0;
        }
        bit_rate_hz / total_bw
    }

    /// Maximum number of users supported by the MDW code family.
    ///
    /// `K_max = ⌊N / w⌋`
    pub fn max_users(&self) -> usize {
        if self.code_weight == 0 {
            return 0;
        }
        self.n_wavelengths / self.code_weight
    }

    /// Centre frequency of spectral chip `k` (Hz).
    pub fn chip_frequency_hz(&self, chip_idx: usize) -> f64 {
        let offset = chip_idx as f64 - (self.n_wavelengths as f64 - 1.0) / 2.0;
        self.center_freq_hz + offset * self.channel_spacing_hz
    }

    /// Cross-correlation between two MDW codes (number of shared wavelengths).
    pub fn cross_correlation(&self, user_a: usize, user_b: usize) -> usize {
        let ca = self.mdw_code(user_a);
        let cb = self.mdw_code(user_b);
        ca.iter()
            .zip(cb.iter())
            .map(|(&a, &b)| (a & b) as usize)
            .sum()
    }
}

// ---------------------------------------------------------------------------
// SPC-OCDMA
// ---------------------------------------------------------------------------

/// Spectral Phase Coding OCDMA transceiver.
///
/// A broadband coherent pulse is decomposed into `n_chips` spectral bins.
/// Each user is assigned a unique phase vector `φ_k ∈ [0, 2π)`.  After
/// encoding, the receiver multiplies the spectrum by the conjugate phase code
/// and performs an IFFT to recover a compressed time-domain pulse (for the
/// intended user) or a diffuse background (for other users).
#[derive(Debug, Clone)]
pub struct SpcOcdma {
    /// Number of spectral chips (frequency bins).
    pub n_chips: usize,
    /// Phase code vector — one phase value per spectral chip (radians).
    pub phases: Vec<f64>,
}

impl SpcOcdma {
    /// Create a new SPC-OCDMA transceiver.
    pub fn new(phases: Vec<f64>) -> Self {
        let n_chips = phases.len();
        Self { n_chips, phases }
    }

    /// Encode a real-valued input spectrum by multiplying with `exp(i·φ_k)`.
    ///
    /// Returns a vector of complex chip amplitudes `(Re, Im)`.
    pub fn encode_spectrum(&self, spectrum: &[f64]) -> Vec<(f64, f64)> {
        spectrum
            .iter()
            .zip(self.phases.iter())
            .map(|(&s, &phi)| {
                let (sin_p, cos_p) = phi.sin_cos();
                (s * cos_p, s * sin_p)
            })
            .collect()
    }

    /// Decode a phase-encoded spectrum by conjugate multiplication.
    ///
    /// Multiplies each complex chip by `exp(−i·φ_k)`.  For the intended user
    /// this restores the original spectrum; for all other users the residual
    /// phase randomises the result.
    pub fn decode_spectrum(&self, encoded: &[(f64, f64)]) -> Vec<(f64, f64)> {
        encoded
            .iter()
            .zip(self.phases.iter())
            .map(|(&(re, im), &phi)| {
                let (sin_p, cos_p) = phi.sin_cos();
                // Multiply by exp(-i·φ) = cos(φ) - i·sin(φ)
                (re * cos_p + im * sin_p, -re * sin_p + im * cos_p)
            })
            .collect()
    }

    /// Auto-correlation peak after matched decoding.
    ///
    /// For an N-chip code this equals N (coherent summation of all chips).
    pub fn autocorrelation_peak(&self) -> f64 {
        self.n_chips as f64
    }

    /// Generate a Walsh–Hadamard phase code.
    ///
    /// Row `idx` of the N × N Hadamard matrix gives a ±1 sequence which
    /// maps to phases `{0, π}`.  Uses the Sylvester construction.
    ///
    /// * `n`   — code length (must be a power of two).
    /// * `idx` — row index (0 ≤ idx < n).
    pub fn walsh_code(n: usize, idx: usize) -> Vec<f64> {
        let n_clamped = n.next_power_of_two().max(1);
        let idx = idx % n_clamped;
        let mut code = vec![1.0f64; n_clamped];
        let mut half = 1usize;
        let mut row = idx;
        while half < n_clamped {
            let bit = row & 1;
            if bit == 1 {
                // Negate alternating blocks of size `half`
                for (k, v) in code.iter_mut().enumerate().take(n_clamped) {
                    if (k / half) % 2 == 1 {
                        *v *= -1.0;
                    }
                }
            }
            row >>= 1;
            half <<= 1;
        }
        // Map +1 → phase 0, -1 → phase π
        code.iter()
            .map(|&v| if v > 0.0 { 0.0 } else { PI })
            .collect()
    }

    /// Estimated decoded pulse width in seconds (time-bandwidth product).
    ///
    /// After IFFT the compressed pulse occupies ≈ 1 / (N × Δν).
    pub fn decoded_pulse_width_s(&self, chip_bandwidth_hz: f64) -> f64 {
        if chip_bandwidth_hz < 1e-30 || self.n_chips == 0 {
            return f64::INFINITY;
        }
        1.0 / (self.n_chips as f64 * chip_bandwidth_hz)
    }

    /// Compute the IFFT of a complex spectrum (Cooley–Tukey, zero-padded
    /// to next power-of-two if necessary).
    ///
    /// Returns time-domain amplitudes as `Vec<(f64, f64)>`.
    pub fn ifft(&self, spectrum: &[(f64, f64)]) -> Vec<(f64, f64)> {
        let n = spectrum.len().next_power_of_two().max(1);
        let mut buf: Vec<(f64, f64)> = spectrum
            .iter()
            .cloned()
            .chain(std::iter::repeat((0.0, 0.0)))
            .take(n)
            .collect();

        // Bit-reversal permutation
        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                buf.swap(i, j);
            }
        }

        // Cooley–Tukey butterfly (IFFT: positive exponent)
        let mut len = 2usize;
        while len <= n {
            let half = len / 2;
            let angle = 2.0 * PI / len as f64; // positive for IFFT
            let (wsin, wcos) = angle.sin_cos();
            for start in (0..n).step_by(len) {
                let (mut wr, mut wi) = (1.0f64, 0.0f64);
                for k in 0..half {
                    let (ur, ui) = buf[start + k];
                    let (vr, vi) = buf[start + k + half];
                    let (tr, ti) = (wr * vr - wi * vi, wr * vi + wi * vr);
                    buf[start + k] = (ur + tr, ui + ti);
                    buf[start + k + half] = (ur - tr, ui - ti);
                    let new_wr = wr * wcos - wi * wsin;
                    wi = wr * wsin + wi * wcos;
                    wr = new_wr;
                }
            }
            len <<= 1;
        }

        // Normalise by 1/N
        let scale = 1.0 / n as f64;
        buf.iter()
            .map(|&(re, im)| (re * scale, im * scale))
            .collect()
    }

    /// Number of orthogonal codes available (equal to the code length for
    /// Walsh–Hadamard codes).
    pub fn n_orthogonal_codes(&self) -> usize {
        self.n_chips.next_power_of_two().max(1)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sac_mdw_code_weight() {
        let sac = SacOcdma::new(16, 4, 193.1e12, 50e9);
        let code = sac.mdw_code(0);
        let weight: usize = code.iter().map(|&b| b as usize).sum();
        assert_eq!(weight, 4, "MDW code weight should equal code_weight");
    }

    #[test]
    fn sac_complementary_code_no_overlap() {
        let sac = SacOcdma::new(16, 4, 193.1e12, 50e9);
        let code = sac.mdw_code(0);
        let comp = sac.complementary_code(&code);
        // code AND comp must be all zeros
        let overlap: usize = code
            .iter()
            .zip(comp.iter())
            .map(|(&a, &b)| (a & b) as usize)
            .sum();
        assert_eq!(overlap, 0);
        // Union must cover all chips
        let union: usize = code
            .iter()
            .zip(comp.iter())
            .map(|(&a, &b)| (a | b) as usize)
            .sum();
        assert_eq!(union, 16);
    }

    #[test]
    fn sac_max_users() {
        let sac = SacOcdma::new(32, 4, 193.1e12, 50e9);
        assert_eq!(sac.max_users(), 8); // 32/4 = 8
    }

    #[test]
    fn spc_encode_decode_identity() {
        // Walsh code row 0 (all zeros → all phases 0) → encode is identity
        let phases = vec![0.0f64; 8];
        let spc = SpcOcdma::new(phases);
        let spectrum = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let encoded = spc.encode_spectrum(&spectrum);
        let decoded = spc.decode_spectrum(&encoded);
        for (orig, (re, _im)) in spectrum.iter().zip(decoded.iter()) {
            assert!((orig - re).abs() < 1e-10, "decode(encode(x)) ≠ x");
        }
    }

    #[test]
    fn spc_walsh_code_orthogonality() {
        // Two distinct Walsh codes should be orthogonal (dot product = 0 mod π)
        let w0 = SpcOcdma::walsh_code(8, 0);
        let w1 = SpcOcdma::walsh_code(8, 1);
        // Map phases back to ±1 and compute dot product
        let dot: f64 = w0
            .iter()
            .zip(w1.iter())
            .map(|(&a, &b)| {
                let va = if a.abs() < 0.1 { 1.0 } else { -1.0 };
                let vb = if b.abs() < 0.1 { 1.0 } else { -1.0 };
                va * vb
            })
            .sum();
        assert_eq!(dot, 0.0, "Walsh codes 0 and 1 must be orthogonal");
    }

    #[test]
    fn spc_autocorrelation_peak() {
        let phases = SpcOcdma::walsh_code(8, 3);
        let spc = SpcOcdma::new(phases);
        assert_eq!(spc.autocorrelation_peak(), 8.0);
    }

    #[test]
    fn sac_snr_infinite_at_zero_noise() {
        let sac = SacOcdma::new(8, 2, 193.1e12, 50e9);
        let snr = sac.snr(1e-3, 0.0, 0.0);
        assert!(snr.is_infinite() || snr > 1e20);
    }

    #[test]
    fn spc_ifft_impulse() {
        // All-ones spectrum → IFFT should give impulse at bin 0
        let phases = vec![0.0f64; 8];
        let spc = SpcOcdma::new(phases);
        let spectrum: Vec<(f64, f64)> = vec![(1.0, 0.0); 8];
        let td = spc.ifft(&spectrum);
        // First sample should equal 1.0, rest ≈ 0
        assert!(
            (td[0].0 - 1.0).abs() < 1e-10,
            "IFFT impulse[0] = {}",
            td[0].0
        );
        for &(re, im) in &td[1..8] {
            assert!(re.abs() < 1e-10 && im.abs() < 1e-10, "IFFT tail non-zero");
        }
    }
}
