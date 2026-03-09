//! Modulation formats, constellation generation, and coherent receiver modelling.
//!
//! # Modulation formats
//!
//! | Format  | bits/symbol | Spectral eff. (pol-mux) |
//! |---------|-------------|-------------------------|
//! | OOK     | 1           | 1 bit/s/Hz              |
//! | BPSK    | 1           | 2 bit/s/Hz              |
//! | DPSK    | 1           | 2 bit/s/Hz              |
//! | QPSK    | 2           | 4 bit/s/Hz              |
//! | DQPSK   | 2           | 4 bit/s/Hz              |
//! | 16-QAM  | 4           | 8 bit/s/Hz              |
//! | 64-QAM  | 6           | 12 bit/s/Hz             |
//! | 256-QAM | 8           | 16 bit/s/Hz             |
//! | PAM-4   | 2           | 2 bit/s/Hz (single-pol) |
//!
//! # References
//!
//! - Proakis & Salehi, "Digital Communications", 5th ed., McGraw-Hill, 2008
//! - Savory, "Digital Coherent Optical Receivers", IEEE JSTQE, 2010

use num_complex::Complex64;
use std::f64::consts::PI;

use crate::comms::metrics::BerCalculator;

// ──────────────────────────────────────────────────────────────────────────────
// ModulationFormat
// ──────────────────────────────────────────────────────────────────────────────

/// Standard optical modulation format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModulationFormat {
    /// On-off keying (binary intensity modulation), direct or coherent detection.
    Ook,
    /// Binary phase-shift keying (coherent).
    Bpsk,
    /// Differential BPSK (self-coherent).
    Dpsk,
    /// Quadrature phase-shift keying (coherent).
    Qpsk,
    /// Differential QPSK (self-coherent).
    Dqpsk,
    /// 16-level quadrature amplitude modulation (coherent).
    Qam16,
    /// 64-level QAM (coherent).
    Qam64,
    /// 256-level QAM (coherent).
    Qam256,
    /// 4-level pulse-amplitude modulation (short-reach direct detection).
    Pam4,
}

impl ModulationFormat {
    /// Number of bits carried per modulation symbol: log₂(M).
    pub fn bits_per_symbol(&self) -> usize {
        match self {
            ModulationFormat::Ook => 1,
            ModulationFormat::Bpsk => 1,
            ModulationFormat::Dpsk => 1,
            ModulationFormat::Qpsk => 2,
            ModulationFormat::Dqpsk => 2,
            ModulationFormat::Qam16 => 4,
            ModulationFormat::Qam64 => 6,
            ModulationFormat::Qam256 => 8,
            ModulationFormat::Pam4 => 2,
        }
    }

    /// Spectral efficiency (bit/s/Hz) assuming dual-polarisation transmission.
    ///
    /// For PAM-4, which is inherently single-polarisation, the factor of 2 for
    /// polarisation multiplexing is not applied.
    pub fn spectral_efficiency_bps_per_hz(&self) -> f64 {
        match self {
            ModulationFormat::Pam4 => self.bits_per_symbol() as f64,
            _ => 2.0 * self.bits_per_symbol() as f64,
        }
    }

    /// Human-readable format name.
    pub fn name(&self) -> &str {
        match self {
            ModulationFormat::Ook => "OOK",
            ModulationFormat::Bpsk => "BPSK",
            ModulationFormat::Dpsk => "DPSK",
            ModulationFormat::Qpsk => "QPSK",
            ModulationFormat::Dqpsk => "DQPSK",
            ModulationFormat::Qam16 => "16-QAM",
            ModulationFormat::Qam64 => "64-QAM",
            ModulationFormat::Qam256 => "256-QAM",
            ModulationFormat::Pam4 => "PAM-4",
        }
    }

    /// Approximate required OSNR (dB, in 0.1 nm bandwidth) for the target BER.
    ///
    /// Uses analytic closed-form approximations based on the relationship
    ///   OSNR_req ≈ Eb/N₀_req × (2 × B_symbol / B_ref)
    /// with B_ref = 12.5 GHz (≡ 0.1 nm at 1550 nm) and B_symbol ≈ B_ref/2
    /// (assumes Nyquist-limited signalling at typical 10–40 Gbaud rates).
    ///
    /// # Arguments
    /// * `ber` – target BER (e.g. 1e-3 for pre-FEC, 1e-12 for post-FEC)
    pub fn required_osnr_db(&self, ber: f64) -> f64 {
        // Get required Eb/N0 from binary-search (BerCalculator)
        let eb_n0_db = BerCalculator::required_eb_n0_db(self, ber);
        let eb_n0_lin = 10.0_f64.powf(eb_n0_db / 10.0);
        // OSNR_linear = Eb/N0 × (2 × Rs) / B_ref
        // With Rs ≈ B_ref for a single-carrier occupying the reference BW:
        // OSNR_lin = Eb/N0 × 2  (assuming Rs = B_ref)
        // In practice the ratio Rs/B_ref depends on baud rate; here we use
        // a canonical 12.5 GHz B_ref with Rs/B_ref = 1.
        let bits = self.bits_per_symbol() as f64;
        // Convert Eb/N0 → Es/N0 → OSNR
        // OSNR = (Es/N0) / (2 · B_ref/B_symbol) for pol-mux; simplified:
        let osnr_lin = eb_n0_lin * bits / 2.0; // factor of 2 for pol-mux
        10.0 * osnr_lin.max(1e-40).log10()
    }

    /// Complex baseband constellation points (normalised to unit average energy).
    ///
    /// Points are enumerated in Gray-coded order where applicable.
    pub fn constellation_points(&self) -> Vec<Complex64> {
        match self {
            ModulationFormat::Ook => {
                vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]
            }
            ModulationFormat::Bpsk | ModulationFormat::Dpsk => {
                vec![Complex64::new(-1.0, 0.0), Complex64::new(1.0, 0.0)]
            }
            ModulationFormat::Qpsk | ModulationFormat::Dqpsk => {
                let s = (0.5_f64).sqrt();
                vec![
                    Complex64::new(s, s),
                    Complex64::new(-s, s),
                    Complex64::new(-s, -s),
                    Complex64::new(s, -s),
                ]
            }
            ModulationFormat::Qam16 => square_qam_constellation(4),
            ModulationFormat::Qam64 => square_qam_constellation(8),
            ModulationFormat::Qam256 => square_qam_constellation(16),
            ModulationFormat::Pam4 => {
                // PAM-4: 4 real amplitude levels, normalised
                // Levels: −3, −1, +1, +3 (normalised by √5)
                let norm = (5.0_f64).sqrt();
                vec![
                    Complex64::new(-3.0 / norm, 0.0),
                    Complex64::new(-1.0 / norm, 0.0),
                    Complex64::new(1.0 / norm, 0.0),
                    Complex64::new(3.0 / norm, 0.0),
                ]
            }
        }
    }

    /// Minimum Euclidean distance between constellation points (dimensionless).
    ///
    /// Computed from the actual normalised constellation.
    pub fn min_distance(&self) -> f64 {
        let pts = self.constellation_points();
        if pts.len() < 2 {
            return f64::INFINITY;
        }
        let mut d_min = f64::INFINITY;
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                let d = (pts[i] - pts[j]).norm();
                if d < d_min {
                    d_min = d;
                }
            }
        }
        d_min
    }

    /// Approximate coding gain (dB) over OOK at the same spectral efficiency.
    ///
    /// This represents the SNR advantage of coherent modulation.
    pub fn coding_gain_db(&self) -> f64 {
        match self {
            ModulationFormat::Ook => 0.0,
            ModulationFormat::Bpsk | ModulationFormat::Dpsk => 3.0,
            ModulationFormat::Qpsk | ModulationFormat::Dqpsk => 3.0,
            ModulationFormat::Qam16 => 7.0,
            ModulationFormat::Qam64 => 11.0,
            ModulationFormat::Qam256 => 15.0,
            ModulationFormat::Pam4 => -1.76,
        }
    }
}

/// Generate a square M×M QAM constellation (M must be a power of 2 ≥ 2).
///
/// Levels are the odd integers ±1, ±3, …, ±(M−1) normalised to unit average
/// symbol energy.
///
/// # Arguments
/// * `side` – number of points per side (e.g., 4 for 16-QAM, 8 for 64-QAM)
fn square_qam_constellation(side: usize) -> Vec<Complex64> {
    debug_assert!(
        side >= 2 && side.is_power_of_two(),
        "QAM side must be power-of-2 ≥ 2"
    );
    let levels: Vec<f64> = (0..side)
        .map(|k| 2.0 * k as f64 - (side as f64 - 1.0))
        .collect();
    let n_pts = side * side;
    // Average energy: E_s = 2/3 · (M−1) for M-ary PAM; for square QAM E_s = 2·(M-1)/3
    let m = side as f64;
    let e_s = 2.0 * (m * m - 1.0) / 3.0;
    let norm = e_s.sqrt();
    let mut pts = Vec::with_capacity(n_pts);
    for &q in &levels {
        for &i in &levels {
            pts.push(Complex64::new(i / norm, q / norm));
        }
    }
    pts
}

// ──────────────────────────────────────────────────────────────────────────────
// CoherentReceiver
// ──────────────────────────────────────────────────────────────────────────────

/// Coherent optical receiver model including DSP impairment budget.
///
/// Covers phase noise from local oscillator linewidth, chromatic dispersion
/// compensation filter design, and ADC quantisation effects.
#[derive(Debug, Clone)]
pub struct CoherentReceiver {
    /// Modulation format
    pub format: ModulationFormat,
    /// Local oscillator (laser) linewidth (kHz)
    pub lo_linewidth_khz: f64,
    /// Electrical bandwidth of the receiver (GHz)
    pub rx_bandwidth_ghz: f64,
    /// ADC resolution (bits)
    pub adc_bits: usize,
    /// Photodetector responsivity R (A/W)
    pub responsivity: f64,
}

impl CoherentReceiver {
    /// Construct a new coherent receiver.
    ///
    /// # Arguments
    /// * `format`           – modulation format
    /// * `lo_linewidth_khz` – local oscillator linewidth (kHz)
    /// * `rx_bw_ghz`        – electrical bandwidth (GHz)
    /// * `adc_bits`         – ADC resolution (bits)
    /// * `responsivity`     – detector responsivity (A/W)
    pub fn new(
        format: ModulationFormat,
        lo_linewidth_khz: f64,
        rx_bw_ghz: f64,
        adc_bits: usize,
        responsivity: f64,
    ) -> Self {
        Self {
            format,
            lo_linewidth_khz,
            rx_bandwidth_ghz: rx_bw_ghz,
            adc_bits,
            responsivity,
        }
    }

    /// OSNR penalty (dB) from LO phase noise.
    ///
    /// Approximation for QPSK/QAM systems:
    ///   penalty ≈ 10·log₁₀(1 + π²/3 · (Δν · T_s))
    ///
    /// where Δν is the combined linewidth (laser + LO) and T_s = 1/R_s is the
    /// symbol period.  A factor of 2 is included to account for both laser and
    /// LO linewidths being equal to `lo_linewidth_khz`.
    ///
    /// # Arguments
    /// * `symbol_rate_gbaud` – symbol rate (Gbaud)
    pub fn phase_noise_penalty_db(&self, symbol_rate_gbaud: f64) -> f64 {
        let delta_nu_hz = 2.0 * self.lo_linewidth_khz * 1e3; // combined linewidth
        let t_symbol_s = 1.0 / (symbol_rate_gbaud * 1e9);
        let delta_nu_t = delta_nu_hz * t_symbol_s;
        // Phase error variance: σ²_φ ≈ 2π · Δν · T_s
        let sigma_phi_sq = 2.0 * PI * delta_nu_t;
        // OSNR penalty from phase noise (first-order approximation)
        let penalty_lin = 1.0 + (PI * PI / 3.0) * sigma_phi_sq;
        10.0 * penalty_lin.log10()
    }

    /// Required OSNR (dB) accounting for phase noise penalty.
    ///
    /// # Arguments
    /// * `ber_target`        – target BER
    /// * `symbol_rate_gbaud` – symbol rate (Gbaud)
    pub fn required_osnr_db(&self, ber_target: f64, symbol_rate_gbaud: f64) -> f64 {
        let base_osnr = self.format.required_osnr_db(ber_target);
        let pn_penalty = self.phase_noise_penalty_db(symbol_rate_gbaud);
        base_osnr + pn_penalty
    }

    /// Hard-decision detection in the presence of AWGN.
    ///
    /// Each received sample is mapped to the nearest constellation point
    /// (minimum Euclidean distance).
    ///
    /// # Arguments
    /// * `received` – slice of received complex samples (before noise is added)
    /// * `snr_linear` – SNR in linear units; currently unused (noise already present)
    pub fn detect(&self, received: &[Complex64], _snr_linear: f64) -> Vec<Complex64> {
        let constellation = self.format.constellation_points();
        received
            .iter()
            .map(|&sample| {
                constellation
                    .iter()
                    .min_by(|&&a, &&b| {
                        let da = (sample - a).norm_sqr();
                        let db = (sample - b).norm_sqr();
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .copied()
                    .unwrap_or(Complex64::new(0.0, 0.0))
            })
            .collect()
    }

    /// Raised-cosine matched filter frequency response at normalised frequency f_n.
    ///
    ///   H_RC(f_n) = { 1,                                       |f_n| ≤ (1−β)/2
    ///               { ½·[1 + cos(π/β·(|f_n| − (1−β)/2))],     (1−β)/2 < |f_n| ≤ (1+β)/2
    ///               { 0,                                        |f_n| > (1+β)/2
    ///
    /// where β = 0.1 (roll-off factor, Nyquist signalling).
    ///
    /// # Arguments
    /// * `freq_normalized` – normalised frequency f/Rs ∈ [0, 1]
    pub fn matched_filter_response(&self, freq_normalized: f64) -> f64 {
        let beta = 0.1_f64; // roll-off factor
        let f = freq_normalized.abs();
        let f_low = (1.0 - beta) / 2.0;
        let f_high = (1.0 + beta) / 2.0;
        if f <= f_low {
            1.0
        } else if f <= f_high {
            0.5 * (1.0 + (PI / beta * (f - f_low)).cos())
        } else {
            0.0
        }
    }

    /// Chromatic dispersion compensation (CDC) filter coefficients in the time domain.
    ///
    /// The frequency-domain transfer function is:
    ///   H_CD(f) = exp(j·π·D·λ²/c · f²)
    ///
    /// where D (ps/nm) is the accumulated dispersion and f is in Hz.  The filter
    /// is computed at `n_taps` uniformly spaced frequency bins over ±bandwidth/2.
    ///
    /// The returned vector contains the complex time-domain taps (IFFT of H_CD).
    ///
    /// # Arguments
    /// * `accumulated_dispersion_ps_per_nm` – total accumulated dispersion (ps/nm)
    /// * `lambda_nm`    – signal centre wavelength (nm)
    /// * `bandwidth_ghz` – signal bandwidth (GHz) over which to design the filter
    /// * `n_taps`        – number of filter taps (should be odd, typically 31–511)
    pub fn cd_compensation_filter(
        &self,
        accumulated_dispersion_ps_per_nm: f64,
        lambda_nm: f64,
        bandwidth_ghz: f64,
        n_taps: usize,
    ) -> Vec<Complex64> {
        let n = n_taps.max(1);
        let lambda_m = lambda_nm * 1e-9;
        let bw_hz = bandwidth_ghz * 1e9;
        // D in s/m²: convert ps/nm → s/m²
        // [ps/nm] × 1e-12/1e-9 = 1e-3 s/m² — then /λ² gives rad/Hz²
        let d_s_m2 = accumulated_dispersion_ps_per_nm * 1e-3; // ps/nm → s/m (dispersion slope)
                                                              // Phase coefficient: φ(f) = π·D·λ²/c · f²
                                                              // D here is in ps/(nm·km) integrated over distance, units: s/m
                                                              // λ in m, c in m/s → phase coefficient α in s² (group delay dispersion)
                                                              // α = D_accum [s/m] · λ² [m²] / c [m/s] = D_accum · λ²/c [s]
                                                              // accumulated D_accum = D [ps/nm] · L [km] converted:
                                                              // D [ps/nm] → [s/m] via ×1e-12/1e-9 = ×1e-3
        let alpha_s2 = d_s_m2 * lambda_m * lambda_m / (2.998e8_f64);
        // Frequency axis: −BW/2 … +BW/2 with n_taps points
        let mut h_freq = Vec::with_capacity(n);
        for k in 0..n {
            let f = (k as f64 - (n / 2) as f64) * bw_hz / n as f64;
            let phase = PI * alpha_s2 * f * f;
            h_freq.push(Complex64::new(phase.cos(), phase.sin()));
        }
        // IDFT (brute force, O(n²)) — for production use OxiFFT
        let mut taps = Vec::with_capacity(n);
        for m in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for (k, &hk) in h_freq.iter().enumerate().take(n) {
                let angle = 2.0 * PI * (k as f64) * (m as f64) / n as f64;
                let tw = Complex64::new(angle.cos(), angle.sin());
                sum += hk * tw;
            }
            taps.push(sum / n as f64);
        }
        taps
    }

    /// Effective number of bits (ENOB) of the ADC as a function of frequency.
    ///
    /// Models frequency-dependent ENOB degradation:
    ///   ENOB(f) = N_bits − 0.5·log₂(1 + (f/f_3dB)²)
    ///
    /// where f_3dB is approximated as 60% of the receiver bandwidth.
    ///
    /// # Arguments
    /// * `freq_ghz` – frequency at which to evaluate ENOB (GHz)
    pub fn enob(&self, freq_ghz: f64) -> f64 {
        let f_3db = 0.6 * self.rx_bandwidth_ghz;
        let ratio = freq_ghz / f_3db.max(1e-10);
        let degradation = 0.5 * (1.0 + ratio * ratio).log2();
        (self.adc_bits as f64 - degradation).max(0.0)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// AmplifierChain
// ──────────────────────────────────────────────────────────────────────────────

/// Optical amplifier chain (cascade of N identical spans).
///
/// Each span consists of a fiber segment (loss = `span_loss_db`) followed by
/// an optical amplifier (gain = `gain_db`, NF = `noise_figure_db`).
#[derive(Debug, Clone)]
pub struct AmplifierChain {
    /// Number of amplifiers (= number of spans)
    pub n_amplifiers: usize,
    /// Per-amplifier gain (dB)
    pub gain_db: f64,
    /// Per-amplifier noise figure (dB)
    pub noise_figure_db: f64,
    /// Span loss (fiber + connectors, dB)
    pub span_loss_db: f64,
}

impl AmplifierChain {
    /// Construct an amplifier chain.
    pub fn new(n_amplifiers: usize, gain_db: f64, noise_figure_db: f64, span_loss_db: f64) -> Self {
        Self {
            n_amplifiers,
            gain_db,
            noise_figure_db,
            span_loss_db,
        }
    }

    /// OSNR at the chain output (dB).
    ///
    /// Assumes each amplifier exactly compensates span loss (G = span_loss_db).
    /// Uses the standard multi-span OSNR formula:
    ///
    ///   OSNR = P_in / (n_sp · h·ν · B_ref · N · (G−1))
    ///
    /// where N is the number of amplifiers and n_sp ≈ NF·G/(2·(G−1)).
    ///
    /// # Arguments
    /// * `input_power_dbm` – launch power per channel into first span (dBm)
    /// * `lambda_nm`       – signal wavelength (nm)
    /// * `ref_bw_nm`       – OSNR reference bandwidth (nm); typically 0.1 nm
    pub fn output_osnr_db(&self, input_power_dbm: f64, lambda_nm: f64, ref_bw_nm: f64) -> f64 {
        if self.n_amplifiers == 0 {
            return f64::INFINITY;
        }
        let per_span_ase = crate::comms::metrics::OsnrAnalysis::ase_per_span_dbm(
            self.gain_db,
            self.noise_figure_db,
            lambda_nm,
            ref_bw_nm,
        );
        let total_ase_dbm = crate::comms::metrics::OsnrAnalysis::accumulated_ase_dbm(
            self.n_amplifiers,
            per_span_ase,
        );
        // Signal power stays at input power (gain exactly compensates loss)
        input_power_dbm - total_ase_dbm
    }

    /// Effective total noise figure of the cascaded amplifier chain (dB).
    ///
    /// Uses the Friis cascaded NF formula (each stage: NF_i, G_i):
    ///   NF_total = NF₁ + (NF₂−1)/G₁ + (NF₃−1)/(G₁G₂) + …
    ///
    /// For N identical stages:
    ///   NF_total ≈ NF₁ · (1 + 1/G + 1/G² + …) = NF₁ · G/(G−1)  (large G limit)
    pub fn total_noise_figure_db(&self) -> f64 {
        if self.n_amplifiers == 0 {
            return 0.0;
        }
        let g = 10.0_f64.powf(self.gain_db / 10.0);
        let nf = 10.0_f64.powf(self.noise_figure_db / 10.0);
        // Friis sum: NF_total = NF · Σ_{k=0}^{N-1} G^{-k}
        let n = self.n_amplifiers as f64;
        let total_nf_lin = if (g - 1.0).abs() < 1e-10 {
            nf * n
        } else {
            nf * (1.0 - g.powf(-n)) / (1.0 - 1.0 / g)
        };
        10.0 * total_nf_lin.max(1e-40).log10()
    }

    /// Maximum achievable transmission distance (km) for a target OSNR.
    ///
    /// Solves for the number of spans N such that OSNR_out ≥ target_osnr_db, then
    /// converts: distance = N × span_length_km.
    ///
    /// # Arguments
    /// * `span_length_km`  – physical length of each span (km)
    /// * `target_osnr_db`  – minimum acceptable OSNR (dB)
    /// * `input_power_dbm` – per-channel launch power (dBm)
    /// * `lambda_nm`       – signal wavelength (nm)
    pub fn max_distance_km(
        &self,
        span_length_km: f64,
        target_osnr_db: f64,
        input_power_dbm: f64,
        lambda_nm: f64,
    ) -> f64 {
        // Binary search over number of spans
        let mut lo: usize = 1;
        let mut hi: usize = 10_000;
        // Check feasibility: even 1 span must beat target
        let chain1 = AmplifierChain::new(1, self.gain_db, self.noise_figure_db, self.span_loss_db);
        if chain1.output_osnr_db(input_power_dbm, lambda_nm, 0.1) < target_osnr_db {
            return 0.0;
        }
        // Check if even 10000 spans is fine (edge case)
        let chain_max =
            AmplifierChain::new(hi, self.gain_db, self.noise_figure_db, self.span_loss_db);
        if chain_max.output_osnr_db(input_power_dbm, lambda_nm, 0.1) >= target_osnr_db {
            return hi as f64 * span_length_km;
        }
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            let chain =
                AmplifierChain::new(mid, self.gain_db, self.noise_figure_db, self.span_loss_db);
            let osnr = chain.output_osnr_db(input_power_dbm, lambda_nm, 0.1);
            if osnr >= target_osnr_db {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo as f64 * span_length_km
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qpsk_bits_per_symbol() {
        assert_eq!(ModulationFormat::Qpsk.bits_per_symbol(), 2);
    }

    #[test]
    fn test_qam16_bits_per_symbol() {
        assert_eq!(ModulationFormat::Qam16.bits_per_symbol(), 4);
    }

    #[test]
    fn test_ook_constellation_has_2_points() {
        assert_eq!(ModulationFormat::Ook.constellation_points().len(), 2);
    }

    #[test]
    fn test_qpsk_constellation_has_4_points() {
        assert_eq!(ModulationFormat::Qpsk.constellation_points().len(), 4);
    }

    #[test]
    fn test_qam16_constellation_has_16_points() {
        assert_eq!(ModulationFormat::Qam16.constellation_points().len(), 16);
    }

    #[test]
    fn test_qam64_constellation_has_64_points() {
        assert_eq!(ModulationFormat::Qam64.constellation_points().len(), 64);
    }

    /// Phase noise penalty must be strictly larger for a wider linewidth.
    #[test]
    fn test_phase_noise_penalty_increases_with_linewidth() {
        let rx1 = CoherentReceiver::new(ModulationFormat::Qpsk, 100.0, 50.0, 8, 0.8);
        let rx2 = CoherentReceiver::new(ModulationFormat::Qpsk, 1000.0, 50.0, 8, 0.8);
        let p1 = rx1.phase_noise_penalty_db(32.0);
        let p2 = rx2.phase_noise_penalty_db(32.0);
        assert!(
            p2 > p1,
            "wider linewidth → larger penalty: p2={p2:.4} vs p1={p1:.4}"
        );
    }

    /// CDC filter length must match requested n_taps.
    #[test]
    fn test_cd_compensation_filter_length() {
        let rx = CoherentReceiver::new(ModulationFormat::Qpsk, 100.0, 50.0, 8, 0.8);
        let taps = rx.cd_compensation_filter(1000.0, 1550.0, 50.0, 63);
        assert_eq!(taps.len(), 63);
    }

    /// OSNR decreases (degrades) as the number of amplifier spans increases.
    #[test]
    fn test_amplifier_chain_osnr_decreases_with_spans() {
        let chain_short = AmplifierChain::new(5, 20.0, 5.0, 20.0);
        let chain_long = AmplifierChain::new(20, 20.0, 5.0, 20.0);
        let osnr_short = chain_short.output_osnr_db(0.0, 1550.0, 0.1);
        let osnr_long = chain_long.output_osnr_db(0.0, 1550.0, 0.1);
        assert!(
            osnr_long < osnr_short,
            "More spans → lower OSNR: {osnr_long:.2} vs {osnr_short:.2}"
        );
    }

    /// Spectral efficiency for QPSK with pol-mux should be 4 bit/s/Hz.
    #[test]
    fn test_qpsk_spectral_efficiency() {
        let se = ModulationFormat::Qpsk.spectral_efficiency_bps_per_hz();
        assert!((se - 4.0).abs() < 1e-10, "QPSK SE should be 4, got {se}");
    }

    /// PAM-4 spectral efficiency (single-pol) = 2 bit/s/Hz.
    #[test]
    fn test_pam4_spectral_efficiency() {
        let se = ModulationFormat::Pam4.spectral_efficiency_bps_per_hz();
        assert!((se - 2.0).abs() < 1e-10, "PAM-4 SE should be 2, got {se}");
    }

    /// min_distance for BPSK (normalised) should be 2.0.
    #[test]
    fn test_bpsk_min_distance() {
        let d = ModulationFormat::Bpsk.min_distance();
        assert!(
            (d - 2.0).abs() < 1e-10,
            "BPSK min_distance should be 2, got {d}"
        );
    }

    /// ENOB at DC must equal full ADC resolution.
    #[test]
    fn test_enob_at_dc() {
        let rx = CoherentReceiver::new(ModulationFormat::Qam16, 100.0, 50.0, 8, 0.8);
        let enob = rx.enob(0.0);
        assert!(
            (enob - 8.0).abs() < 0.01,
            "ENOB at DC should be ~8, got {enob}"
        );
    }

    /// Friis NF formula: cascaded NF must be ≥ single-stage NF.
    #[test]
    fn test_amplifier_chain_friis_nf() {
        let chain = AmplifierChain::new(4, 20.0, 5.0, 20.0);
        let total_nf = chain.total_noise_figure_db();
        assert!(
            total_nf >= 5.0,
            "Cascaded NF must be ≥ single-stage NF: {total_nf:.2}"
        );
    }
}
