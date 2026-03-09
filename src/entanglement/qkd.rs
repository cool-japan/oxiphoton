//! Quantum Key Distribution (QKD) protocol models.
//!
//! Implements key-rate and security analysis for:
//! - BB84 (Bennett & Brassard 1984): prepare-and-measure with single photons
//! - E91 (Ekert 1991): entanglement-based using SPDC photon pairs
//! - CV-QKD (Grosshans & Grangier 2002): continuous-variable with coherent states
//!
//! References:
//! - Shor & Preskill, PRL 85, 441 (2000): BB84 security proof
//! - Ekert, PRL 67, 661 (1991): E91 protocol
//! - Grosshans & Grangier, PRL 88, 057902 (2002): CV-QKD
//! - Pirandola et al., Nature Photon. 9, 397 (2015): CV-QKD distance record

use crate::entanglement::bell_inequality::ChshTest;
use crate::entanglement::entanglement_measures::binary_entropy;
use crate::entanglement::spdc::SpdcSource;

/// Fibre loss coefficient (dB/km) for standard SMF-28 at 1550 nm.
const FIBER_LOSS_DB_PER_KM: f64 = 0.2;

// ─── BB84 protocol ────────────────────────────────────────────────────────────

/// BB84 quantum key distribution protocol model.
///
/// Uses prepare-and-measure with single photons in two conjugate bases
/// (rectilinear Z and diagonal X).  Security is guaranteed by the
/// Shor-Preskill bound when QBER < 11%.
#[derive(Debug, Clone)]
pub struct Bb84Protocol {
    /// Target secure key length (bits)
    pub key_length: usize,
    /// Measured quantum bit error rate (fraction, e.g. 0.03 for 3%)
    pub qber: f64,
    /// Single-photon detector efficiency (fraction)
    pub detection_efficiency: f64,
    /// Dark count rate per gate interval
    pub dark_count_rate: f64,
    /// Total channel loss (dB)
    pub channel_loss_db: f64,
}

impl Bb84Protocol {
    /// Construct a BB84 session with given error rate and channel parameters.
    pub fn new(qber: f64, channel_loss_db: f64, efficiency: f64) -> Self {
        Self {
            key_length: 1_000_000,
            qber: qber.clamp(0.0, 0.5),
            detection_efficiency: efficiency.clamp(0.0, 1.0),
            dark_count_rate: 1e-6,
            channel_loss_db,
        }
    }

    /// Optical channel transmittance T = 10^(−loss_dB / 10).
    fn channel_transmittance(&self) -> f64 {
        10.0_f64.powf(-self.channel_loss_db / 10.0)
    }

    /// Raw (sifted) key rate in bits/s.
    ///
    /// R_sifted = R_pulse × T_channel × η_detector × sift_factor
    ///
    /// The sift factor is 1/2 (Bob measures in the correct basis 50% of the time
    /// in BB84 with two bases, or asymptotically 1 for decoy-state BB84).
    pub fn raw_key_rate_hz(&self, pulse_rate_mhz: f64) -> f64 {
        let r_pulse = pulse_rate_mhz * 1e6; // pulses/s
        let t = self.channel_transmittance();
        let eta = self.detection_efficiency;
        let sift = 0.5; // symmetric BB84 sifting
        r_pulse * t * eta * sift
    }

    /// Expected QBER from physical parameters.
    ///
    /// QBER = QBER_optical + contribution from dark counts:
    /// e_dark = d / (d + η·T·μ) / 2
    /// where d is dark count rate, μ is mean photon number per pulse.
    pub fn expected_qber(&self, mean_photon_number: f64) -> f64 {
        let t = self.channel_transmittance();
        let eta = self.detection_efficiency;
        let d = self.dark_count_rate;
        let signal_rate = eta * t * mean_photon_number;
        // Dark count QBER contribution (dark counts contribute with 50% error prob)
        let qber_dark = if signal_rate + d > 0.0 {
            d / (2.0 * (signal_rate + d))
        } else {
            0.0
        };
        // Add optical alignment/intrinsic error (taken as the stored qber)
        (self.qber + qber_dark).clamp(0.0, 0.5)
    }

    /// Secret key fraction per sifted bit (Shor-Preskill bound).
    ///
    /// r = 1 − h(e_bit) − h(e_phase) ≈ 1 − 2·h(QBER)
    ///
    /// (Assumes bit and phase error rates are equal, valid for symmetric attacks.)
    pub fn secret_key_fraction(&self) -> f64 {
        let r = 1.0 - 2.0 * binary_entropy(self.qber);
        r.max(0.0)
    }

    /// Maximum tolerable QBER for one-way error correction (Shor-Preskill): ~11%.
    pub fn max_tolerable_qber() -> f64 {
        // Solve 1 - 2*h(e) = 0: h(e) = 0.5 → e ≈ 11.0%
        0.110_028 // numerical solution of h(e) = 0.5
    }

    /// Secure key rate after error correction and privacy amplification (bits/s).
    ///
    /// K_secure = R_sifted × r_secret
    pub fn secure_key_rate_bps(&self, pulse_rate_mhz: f64) -> f64 {
        let r_sifted = self.raw_key_rate_hz(pulse_rate_mhz);
        let r_secret = self.secret_key_fraction();
        r_sifted * r_secret
    }

    /// Secure key rate at a given fibre distance (km).
    ///
    /// Uses standard SMF-28 loss (0.2 dB/km) if not overriding the stored channel loss.
    pub fn key_rate_at_distance_km(&self, distance_km: f64, pulse_rate_mhz: f64) -> f64 {
        let loss_db = distance_km * FIBER_LOSS_DB_PER_KM;
        let modified = Self {
            channel_loss_db: loss_db,
            ..self.clone()
        };
        modified.secure_key_rate_bps(pulse_rate_mhz)
    }

    /// Maximum distance at which a secret key rate exceeds 1 bit/s.
    ///
    /// Uses a 1 bit/s threshold since the exponential decay never reaches exactly zero.
    pub fn max_secure_distance_km(&self, pulse_rate_mhz: f64) -> f64 {
        // Binary search from 0 to 500 km
        let threshold_bps = 1.0_f64; // minimum useful rate (1 bit/s)
        let mut lo = 0.0_f64;
        let mut hi = 500.0_f64;
        // Quick check that rate is above threshold at lo
        if self.key_rate_at_distance_km(lo, pulse_rate_mhz) < threshold_bps {
            return 0.0;
        }
        for _ in 0..60 {
            let mid = (lo + hi) / 2.0;
            let rate = self.key_rate_at_distance_km(mid, pulse_rate_mhz);
            if rate > threshold_bps {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

// ─── E91 protocol ────────────────────────────────────────────────────────────

/// Ekert 1991 (E91) entanglement-based QKD protocol.
///
/// Uses SPDC photon pairs; security is guaranteed by violation of the CHSH
/// Bell inequality.  Any eavesdropping reduces entanglement, which is
/// detected through Bell measurements.
#[derive(Debug, Clone)]
pub struct E91Protocol {
    /// SPDC photon pair source
    pub pair_source: SpdcSource,
    /// Fibre loss coefficient (dB/km)
    pub channel_loss_db_per_km: f64,
    /// Channel distance (km, symmetric Alice-Bob)
    pub distance_km: f64,
    /// Measured QBER (may differ from Bell-inequality-derived bound)
    pub qber: f64,
}

impl E91Protocol {
    /// Construct an E91 session over a given fibre distance.
    pub fn new(source: SpdcSource, loss_db_km: f64, distance_km: f64) -> Self {
        Self {
            pair_source: source,
            channel_loss_db_per_km: loss_db_km,
            distance_km,
            qber: 0.03,
        }
    }

    /// Total channel transmittance for both arms (pairs traverse half the distance each).
    fn total_transmittance(&self) -> f64 {
        let loss_db_total = self.channel_loss_db_per_km * self.distance_km;
        10.0_f64.powf(-loss_db_total / 10.0)
    }

    /// Coincidence detection rate (pairs/s) accounting for channel loss.
    ///
    /// Assumes symmetric 50/50 split: each photon travels distance/2.
    pub fn pair_detection_rate(&self) -> f64 {
        let r_pairs = self.pair_source.pair_rate_per_second();
        let t = self.total_transmittance();
        // Both photons must be transmitted and detected; assume η_det = 0.8
        let eta_det = 0.8_f64;
        r_pairs * t * eta_det * eta_det
    }

    /// Expected CHSH |S| parameter as a function of source and channel quality.
    ///
    /// |S| ≈ 2√2 × F_bell × (1 - 2·QBER) where F_bell is the Bell-state fidelity.
    pub fn chsh_violation(&self) -> f64 {
        // Estimate Bell-state fidelity from QBER: F ≈ 1 - 2*QBER
        let f_bell = (1.0 - 2.0 * self.qber).clamp(0.0, 1.0);
        ChshTest::expected_s_for_fidelity(f_bell)
    }

    /// Secret key rate (bits/s) using CHSH-based security (one-way EC).
    ///
    /// K = R_coinc × (1 − 2·h(QBER))
    pub fn secret_key_rate_bps(&self) -> f64 {
        let r_coinc = self.pair_detection_rate();
        // Sifting: E91 uses three bases, sift factor ≈ 1/3 for key bits
        let sift = 1.0 / 3.0;
        let r_secret = (1.0 - 2.0 * binary_entropy(self.qber)).max(0.0);
        r_coinc * sift * r_secret
    }

    /// Returns `true` if |S| > 2 (eavesdropper detectable via Bell inequality).
    pub fn eavesdropper_detectable(&self) -> bool {
        self.chsh_violation() > 2.0
    }
}

// ─── CV-QKD ──────────────────────────────────────────────────────────────────

/// Continuous-variable QKD (CV-QKD) with Gaussian-modulated coherent states.
///
/// Alice encodes a random displacement (x, p) drawn from a Gaussian
/// distribution with variance V_A (shot noise units).  Bob performs homodyne
/// or heterodyne detection.  Security analysis uses the Gaussian optimality
/// theorem (García-Patrón & Cerf 2006).
///
/// Channel model: bosonic lossy channel with transmittance T and excess noise ξ.
#[derive(Debug, Clone)]
pub struct CvQkd {
    /// Modulation variance V_A (shot noise units, N₀ = 1)
    pub modulation_variance: f64,
    /// Channel transmittance T = 10^(−αL/10)
    pub channel_transmittance: f64,
    /// Excess noise ξ (shot noise units, referred to the channel input)
    pub excess_noise: f64,
    /// Reconciliation efficiency β (0 ≤ β ≤ 1, typically 0.95)
    pub reconciliation_efficiency: f64,
}

impl CvQkd {
    /// Construct a CV-QKD instance for a given propagation distance.
    ///
    /// Loss is computed using standard SMF-28: α = loss_db_per_km dB/km.
    pub fn new(v_a: f64, distance_km: f64, loss_db_per_km: f64, excess_noise: f64) -> Self {
        let loss_db = loss_db_per_km * distance_km;
        let t = 10.0_f64.powf(-loss_db / 10.0);
        Self {
            modulation_variance: v_a.max(0.01),
            channel_transmittance: t.clamp(1e-10, 1.0),
            excess_noise: excess_noise.max(0.0),
            reconciliation_efficiency: 0.95,
        }
    }

    /// Channel-added noise (referred to Bob's input): χ = (1−T)/T + ξ/T.
    pub fn channel_noise(&self) -> f64 {
        let t = self.channel_transmittance;
        (1.0 - t) / t + self.excess_noise / t
    }

    /// Signal-to-noise ratio at Bob's detector: SNR = T·V_A / (1 + T·χ).
    fn snr(&self) -> f64 {
        let t = self.channel_transmittance;
        let v_a = self.modulation_variance;
        let chi = self.channel_noise();
        t * v_a / (1.0 + t * chi).max(1e-30)
    }

    /// Mutual information I_AB between Alice and Bob (homodyne detection, one quadrature).
    ///
    /// I_AB = (1/2) log₂(1 + SNR)  [bits per mode]
    pub fn mutual_information_ab(&self) -> f64 {
        0.5 * (1.0 + self.snr()).max(1.0).log2()
    }

    /// Symplectic entropy function g(v) = (v+1)/2 · log₂((v+1)/2) − (v-1)/2 · log₂((v-1)/2).
    ///
    /// This is the von Neumann entropy of a single-mode Gaussian state with
    /// symplectic eigenvalue v ≥ 1.
    fn g_entropy(v: f64) -> f64 {
        let v = v.max(1.0); // enforce physical constraint ν ≥ 1
        let vp = (v + 1.0) / 2.0;
        let vm = (v - 1.0) / 2.0;
        let term_plus = vp * vp.log2();
        let term_minus = if vm < 1e-15 { 0.0 } else { vm * vm.log2() };
        term_plus - term_minus
    }

    /// Holevo information χ_BE (Eve's accessible information, collective attacks).
    ///
    /// Uses the symplectic eigenvalue formalism for two-mode Gaussian states
    /// (Weedbrook et al., Rev. Mod. Phys. 84, 621 (2012), Section VI.B).
    ///
    /// For homodyne detection on Bob's x-quadrature:
    /// χ_BE = g(ν₁) + g(ν₂) − g(ν₃)
    pub fn holevo_information_be(&self) -> f64 {
        let v_a = self.modulation_variance;
        let t = self.channel_transmittance;
        let xi = self.excess_noise;
        // V = V_A + 1 is Alice's total quadrature variance (signal + vacuum)
        let v = v_a + 1.0;
        // Channel noise at the input: χ_line = (1-T)/T + ξ
        let chi_line = (1.0 - t) / t + xi;

        // Bob's diagonal CM element: B = T(V + χ_line)
        let b_bob = t * (v + chi_line);
        // Off-diagonal (correlation): C² = T(V²-1)
        let c_sq = t * (v * v - 1.0);

        // Symplectic eigenvalues of the two-mode CM
        // Δ = V² + B² - 2·T·(V²-1)
        let delta_cm = v * v + b_bob * b_bob - 2.0 * t * (v * v - 1.0);
        // D² = (V·B - C²)² = det(γ_AB)
        let det_ab = v * b_bob - c_sq;
        let det_ab_sq = det_ab * det_ab;

        let disc = (delta_cm * delta_cm - 4.0 * det_ab_sq).max(0.0).sqrt();
        let nu1 = ((delta_cm + disc) / 2.0).max(1.0).sqrt();
        let nu2 = ((delta_cm - disc) / 2.0).max(1.0).sqrt();

        // Conditional symplectic eigenvalue for homodyne detection on mode B (x-quadrature):
        // ν₃² = V - C²/B  (conditional variance of mode A given homodyne on B)
        let nu3 = (v - c_sq / b_bob.max(1e-30)).max(1.0).sqrt();

        // χ_BE = g(ν₁) + g(ν₂) − g(ν₃)
        let chi_be = Self::g_entropy(nu1) + Self::g_entropy(nu2) - Self::g_entropy(nu3);
        chi_be.max(0.0)
    }

    /// Secret key rate per mode: K = β · I_AB − χ_BE.
    ///
    /// Returns 0 if Eve's information exceeds Alice-Bob's (no positive key possible).
    pub fn secret_key_rate_bits_per_mode(&self) -> f64 {
        let i_ab = self.mutual_information_ab();
        let chi_be = self.holevo_information_be();
        (self.reconciliation_efficiency * i_ab - chi_be).max(0.0)
    }

    /// Maximum secure distance (km) assuming SMF-28 loss (0.2 dB/km).
    ///
    /// Performs a binary search over distance to find where K drops to zero.
    pub fn max_secure_distance_km(&self) -> f64 {
        let v_a = self.modulation_variance;
        let xi = self.excess_noise;
        let beta = self.reconciliation_efficiency;
        let loss_db_per_km = FIBER_LOSS_DB_PER_KM;
        // Check that key rate is positive at zero distance
        let at_zero = CvQkd {
            modulation_variance: v_a,
            channel_transmittance: 1.0,
            excess_noise: xi,
            reconciliation_efficiency: beta,
        };
        if at_zero.secret_key_rate_bits_per_mode() <= 0.0 {
            return 0.0;
        }
        let mut lo = 0.0_f64;
        let mut hi = 300.0_f64;
        for _ in 0..80 {
            let mid = (lo + hi) / 2.0;
            let t_mid = 10.0_f64.powf(-loss_db_per_km * mid / 10.0);
            let candidate = CvQkd {
                modulation_variance: v_a,
                channel_transmittance: t_mid,
                excess_noise: xi,
                reconciliation_efficiency: beta,
            };
            if candidate.secret_key_rate_bits_per_mode() > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bb84_secret_key_fraction_low_qber() {
        let bb84 = Bb84Protocol::new(0.02, 3.0, 0.85);
        let r = bb84.secret_key_fraction();
        // At 2% QBER, r = 1 - 2*h(0.02) ≈ 1 - 2*0.1414 ≈ 0.717
        assert!(
            r > 0.5 && r < 1.0,
            "Secret key fraction at 2% QBER should be ~0.7, got {r}"
        );
    }

    #[test]
    fn test_bb84_zero_rate_above_max_qber() {
        // Above 11% QBER no secure key is possible
        let bb84 = Bb84Protocol::new(0.12, 3.0, 0.85);
        let r = bb84.secret_key_fraction();
        assert_eq!(r, 0.0, "Secret key fraction should be 0 at QBER > 11%");
    }

    #[test]
    fn test_bb84_max_tolerable_qber() {
        let e_max = Bb84Protocol::max_tolerable_qber();
        // h(e_max) should be ≈ 0.5
        let h = binary_entropy(e_max);
        assert!((h - 0.5).abs() < 0.01, "h(e_max) ≈ 0.5, got h={h}");
    }

    #[test]
    fn test_bb84_key_rate_positive() {
        let bb84 = Bb84Protocol::new(0.03, 10.0, 0.85);
        let rate = bb84.secure_key_rate_bps(100.0);
        assert!(
            rate > 0.0,
            "Secure key rate should be positive at low QBER and loss"
        );
    }

    #[test]
    fn test_bb84_max_distance_positive() {
        let bb84 = Bb84Protocol::new(0.03, 0.0, 0.85);
        let d_max = bb84.max_secure_distance_km(1000.0);
        // At 1000 MHz, 85% efficiency, 0.2 dB/km, 3% QBER, threshold=1 bps:
        // loss at 400 km = 80 dB → T = 10^-8 → rate ~ 1000e6 * 10^-8 * 0.85 * 0.5 * 0.61 ≈ 2.6 bps
        // loss at 500 km = 100 dB → T = 10^-10 → rate ~ 0.026 bps < 1 bps
        assert!(
            d_max > 100.0 && d_max < 500.0,
            "Max distance should be 100–500 km, got {d_max}"
        );
    }

    #[test]
    fn test_e91_chsh_violation() {
        let source = SpdcSource::new_ppktp_1550(1.0, 10.0);
        let e91 = E91Protocol::new(source, 0.2, 10.0);
        let s = e91.chsh_violation();
        // At 3% QBER: F ≈ 0.94, |S| ≈ 2√2 × 0.94 ≈ 2.66
        assert!(s > 2.0, "E91 should violate CHSH at 3% QBER, S={s}");
    }

    #[test]
    fn test_e91_eavesdropper_detectable() {
        let source = SpdcSource::new_ppktp_1550(1.0, 10.0);
        let mut e91 = E91Protocol::new(source, 0.2, 10.0);
        e91.qber = 0.03;
        assert!(
            e91.eavesdropper_detectable(),
            "E91 with low QBER should detect eavesdroppers"
        );
    }

    #[test]
    fn test_cv_qkd_channel_noise() {
        // At T = 1 (no loss) and ξ = 0: channel noise = 0
        let cv = CvQkd {
            modulation_variance: 10.0,
            channel_transmittance: 1.0,
            excess_noise: 0.0,
            reconciliation_efficiency: 0.95,
        };
        let chi = cv.channel_noise();
        assert!(chi.abs() < 1e-10, "No-loss channel noise = 0, got {chi}");
    }

    #[test]
    fn test_cv_qkd_positive_key_rate_short_distance() {
        // Very short distance (0.1 km = 100 m), low excess noise → positive key rate
        let cv = CvQkd::new(10.0, 0.1, 0.2, 0.001);
        let k = cv.secret_key_rate_bits_per_mode();
        assert!(
            k > 0.0,
            "CV-QKD should have positive key rate at 100 m, got {k}"
        );
    }

    #[test]
    fn test_cv_qkd_max_distance_sensible() {
        // With V_A=10, xi=0.01, beta=0.95 and 0.2 dB/km loss:
        // practical range is 2–3 km (limited by excess noise relative to SNR)
        let cv = CvQkd::new(10.0, 0.0, 0.2, 0.01);
        let d_max = cv.max_secure_distance_km();
        assert!(
            d_max > 0.5 && d_max < 50.0,
            "CV-QKD max distance should be 0.5–50 km for these params, got {d_max}"
        );
    }
}
