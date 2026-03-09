//! Classical and quantum coherence theory.
//!
//! Implements:
//! - First-order temporal coherence g¹(τ) for single-mode and thermal sources
//! - Second-order coherence g²(τ) for laser, thermal, single-photon, and squeezed sources
//! - Spatial coherence via the Van Cittert-Zernike theorem
//! - Hanbury-Brown–Twiss (HBT) experiment modelling including detector imperfections
//!
//! Physical constants used throughout:
//!   C0 = 2.997 924 58 × 10⁸ m s⁻¹
//!   π = std::f64::consts::PI

use num_complex::Complex64;

/// Speed of light in vacuum (m s⁻¹)
const C0: f64 = 2.997_924_58e8;

// ─── First-order coherence ────────────────────────────────────────────────────

/// First-order (temporal) coherence function g¹(τ).
///
/// Encodes the spectral/temporal coherence of a light source characterised by
/// a coherence time τ_c and a central wavelength λ₀.
#[derive(Debug, Clone)]
pub struct FirstOrderCoherence {
    /// Coherence time τ_c (s)
    pub coherence_time_s: f64,
    /// Central wavelength λ₀ (nm)
    pub center_wavelength_nm: f64,
}

impl FirstOrderCoherence {
    /// Construct from coherence time and central wavelength.
    pub fn new(coherence_time_s: f64, center_wavelength_nm: f64) -> Self {
        Self {
            coherence_time_s,
            center_wavelength_nm,
        }
    }

    /// Central angular frequency ω₀ = 2πc / λ₀.
    #[inline]
    fn omega0(&self) -> f64 {
        2.0 * std::f64::consts::PI * C0 / (self.center_wavelength_nm * 1e-9)
    }

    /// g¹(τ) for an ideal single-mode (monochromatic) laser.
    ///
    /// g¹(τ) = exp(−iω₀τ − |τ|/τ_c)
    ///
    /// This is a Lorentzian spectrum (exponential decay in time domain).
    pub fn g1_single_mode(&self, tau_s: f64) -> Complex64 {
        let phase = -self.omega0() * tau_s;
        let decay = -tau_s.abs() / self.coherence_time_s;
        Complex64::from_polar(decay.exp(), phase)
    }

    /// g¹(τ) for a thermal (chaotic / multi-mode) source.
    ///
    /// For a Gaussian spectral profile the envelope is Gaussian:
    ///   g¹(τ) = exp(−iω₀τ) · exp(−|τ|²/(2τ_c²))
    ///
    /// This corresponds to a Gaussian spectrum (most common model for incoherent sources).
    pub fn g1_thermal(&self, tau_s: f64) -> Complex64 {
        let phase = -self.omega0() * tau_s;
        let decay = -(tau_s * tau_s) / (2.0 * self.coherence_time_s * self.coherence_time_s);
        Complex64::from_polar(decay.exp(), phase)
    }

    /// Coherence length L_c = c · τ_c (m).
    #[inline]
    pub fn coherence_length_m(&self) -> f64 {
        C0 * self.coherence_time_s
    }

    /// Spectral linewidth Δν (FWHM) from coherence time.
    ///
    /// For a Lorentzian spectrum (single-mode laser): Δν = 1 / (π τ_c).
    #[inline]
    pub fn linewidth_hz(&self) -> f64 {
        1.0 / (std::f64::consts::PI * self.coherence_time_s)
    }

    /// Visibility of Young's double-slit fringes V = |g¹(τ)|.
    ///
    /// For a single-mode source: V = exp(−|τ|/τ_c).
    pub fn visibility(&self, tau_s: f64) -> f64 {
        // |g¹(τ)| = exp(−|τ|/τ_c) for single-mode
        (-tau_s.abs() / self.coherence_time_s).exp()
    }

    /// Integrated degree of coherence: μ₁₂ = 1 (perfect coherence for single mode).
    ///
    /// By definition, a single-mode source has μ₁₂ = 1; the value encodes the maximum
    /// achievable fringe visibility.
    #[inline]
    pub fn degree_of_coherence(&self) -> f64 {
        1.0
    }
}

// ─── Second-order coherence ───────────────────────────────────────────────────

/// Classification of a light source by its photon statistics.
#[derive(Debug, Clone, PartialEq)]
pub enum LightSource {
    /// Ideal coherent laser — Poissonian statistics.
    Laser,
    /// Thermal / chaotic source — Bose-Einstein statistics, photon bunching.
    Thermal,
    /// Perfect single-photon emitter — photon antibunching.
    SinglePhoton,
    /// Squeezed light — sub-Poissonian statistics.
    Squeezed {
        /// Squeezing factor r (g²(0) ≈ 1 − 1/cosh(2r))
        squeezing_r: f64,
    },
}

/// Second-order (intensity) coherence function g²(τ).
///
/// g²(τ) = ⟨I(t) I(t+τ)⟩ / ⟨I⟩²
///
/// At τ = 0:
///   Laser        → g²(0) = 1
///   Thermal      → g²(0) = 2
///   SinglePhoton → g²(0) = 0
///   Squeezed     → g²(0) < 1
#[derive(Debug, Clone)]
pub struct SecondOrderCoherence {
    /// Type of light source
    pub source_type: LightSource,
    /// Coherence time τ_c (s)
    pub coherence_time_s: f64,
}

impl SecondOrderCoherence {
    /// Construct a second-order coherence model.
    pub fn new(source_type: LightSource, coherence_time_s: f64) -> Self {
        Self {
            source_type,
            coherence_time_s,
        }
    }

    /// g²(0) — the zero-delay second-order coherence.
    pub fn g2_at_zero(&self) -> f64 {
        match &self.source_type {
            LightSource::Laser => 1.0,
            LightSource::Thermal => 2.0,
            LightSource::SinglePhoton => 0.0,
            LightSource::Squeezed { squeezing_r } => {
                // For squeezed vacuum: g²(0) = 3 − 2/cosh(r)²  (Mandel & Wolf 1995)
                // More carefully: g²(0) = (2 sinh⁴r + 2 sinh²r cosh²r) / sinh⁴r
                //   = (2 cosh²r) / sinh²r  which diverges as r→0 for vacuum.
                // For a displaced squeezed state with large coherent amplitude:
                //   g²(0) ≈ 1 − 1/|α|² (practically 1 for bright squeezing).
                // We use the bright-squeezing approximation with the noise reduction factor:
                //   g²(0) = 1 − (1 − exp(−2r)) ≡ exp(−2r)
                let r = squeezing_r;
                (-2.0 * r).exp()
            }
        }
    }

    /// g²(τ) at finite delay.
    ///
    /// - Laser:        g²(τ) = 1 (all τ)
    /// - Thermal:      g²(τ) = 1 + exp(−2|τ|/τ_c)  [Siegert relation]
    /// - SinglePhoton: g²(τ) = 1 − exp(−|τ|/τ_c)
    /// - Squeezed:     g²(τ) = 1 + [g²(0)−1] · exp(−|τ|/τ_c)
    pub fn g2_at_tau(&self, tau_s: f64) -> f64 {
        match &self.source_type {
            LightSource::Laser => 1.0,
            LightSource::Thermal => 1.0 + (-2.0 * tau_s.abs() / self.coherence_time_s).exp(),
            LightSource::SinglePhoton => 1.0 - (-tau_s.abs() / self.coherence_time_s).exp(),
            LightSource::Squeezed { .. } => {
                let g2_0 = self.g2_at_zero();
                1.0 + (g2_0 - 1.0) * (-tau_s.abs() / self.coherence_time_s).exp()
            }
        }
    }

    /// True if g²(0) < 1 (antibunching — non-classical light).
    pub fn is_antibunching(&self) -> bool {
        self.g2_at_zero() < 1.0
    }

    /// True if g²(0) > 1 (bunching — super-Poissonian fluctuations).
    pub fn is_bunching(&self) -> bool {
        self.g2_at_zero() > 1.0
    }

    /// Visibility measured in a Hanbury-Brown–Twiss setup.
    ///
    /// Defined as V_HBT = [g²(0) − g²(∞)] / g²(∞) = g²(0) − 1  (since g²(∞) = 1).
    pub fn hanbury_brown_twiss_visibility(&self) -> f64 {
        self.g2_at_zero() - 1.0
    }
}

// ─── Spatial coherence ────────────────────────────────────────────────────────

/// Spatial coherence of a quasi-monochromatic source via the Van Cittert-Zernike theorem.
///
/// For a spatially incoherent, uniform disk source of diameter D at distance z,
/// the mutual coherence at the observation plane is given by the Fourier transform of
/// the source intensity distribution, which for a disk source yields a jinc function.
/// We use the simpler sinc approximation valid for a slit source of width D:
///
///   γ(Δr) = sinc(π D Δr / (λ z))
#[derive(Debug, Clone)]
pub struct SpatialCoherence {
    /// Central wavelength λ₀ (nm)
    pub wavelength_nm: f64,
    /// Source spatial extent / diameter D (mm)
    pub source_size_mm: f64,
    /// Propagation distance z (m)
    pub propagation_distance_m: f64,
}

impl SpatialCoherence {
    /// Construct a spatial coherence model.
    pub fn new(wavelength_nm: f64, source_size_mm: f64, propagation_distance_m: f64) -> Self {
        Self {
            wavelength_nm,
            source_size_mm,
            propagation_distance_m,
        }
    }

    /// Coherence radius r_c (mm) from the Van Cittert-Zernike theorem.
    ///
    /// r_c = λ z / (π D)
    ///
    /// where λ and D must be in consistent units.  Here λ in m, z in m, D in m → r_c in m → mm.
    pub fn coherence_radius_mm(&self) -> f64 {
        let lambda_m = self.wavelength_nm * 1e-9;
        let d_m = self.source_size_mm * 1e-3;
        let r_c_m = lambda_m * self.propagation_distance_m / (std::f64::consts::PI * d_m);
        r_c_m * 1e3
    }

    /// Normalised mutual coherence γ(Δr) for a slit source (sinc approximation).
    ///
    /// γ(Δr) = sinc(π D Δr / (λ z))  where sinc(x) = sin(x)/x
    pub fn mutual_coherence(&self, separation_mm: f64) -> f64 {
        let lambda_m = self.wavelength_nm * 1e-9;
        let d_m = self.source_size_mm * 1e-3;
        let delta_r_m = separation_mm * 1e-3;
        let x = std::f64::consts::PI * d_m * delta_r_m / (lambda_m * self.propagation_distance_m);
        if x.abs() < 1e-12 {
            1.0
        } else {
            x.sin() / x
        }
    }

    /// Speckle grain size ≈ coherence radius (mm).
    #[inline]
    pub fn speckle_size_mm(&self) -> f64 {
        self.coherence_radius_mm()
    }

    /// Number of coherence cells across a detector of diameter D_det (mm).
    ///
    /// N = (D_det / r_c)²  (in 2D; ratio of areas)
    pub fn coherence_cells(&self, detector_diameter_mm: f64) -> f64 {
        let r_c = self.coherence_radius_mm();
        if r_c <= 0.0 {
            return f64::INFINITY;
        }
        (detector_diameter_mm / r_c).powi(2)
    }

    /// Coherence volume V_c = A_c · L_c (mm³).
    ///
    /// A_c = π r_c²  (coherence area, mm²)
    /// L_c = c · τ_c  (coherence length, converted to mm)
    pub fn coherence_volume_mm3(&self, coherence_time_s: f64) -> f64 {
        let r_c = self.coherence_radius_mm();
        let area_mm2 = std::f64::consts::PI * r_c * r_c;
        let lc_m = C0 * coherence_time_s;
        let lc_mm = lc_m * 1e3;
        area_mm2 * lc_mm
    }
}

// ─── HBT experiment ───────────────────────────────────────────────────────────

/// Model of a Hanbury-Brown–Twiss (HBT) photon correlation experiment.
///
/// Accounts for:
/// - Detector timing jitter (Gaussian broadening of the coincidence histogram)
/// - Detector dead time (suppresses high count rates)
/// - Dark count background (elevates the baseline of g²)
#[derive(Debug, Clone)]
pub struct HBTExperiment {
    /// Quantum optical source characterised by g²(τ)
    pub source: SecondOrderCoherence,
    /// Single-photon detector timing jitter σ_t (ps)
    pub detector_jitter_ps: f64,
    /// Detector dead time t_d (ns)
    pub dead_time_ns: f64,
    /// Dark count rate d (counts s⁻¹)
    pub dark_count_rate: f64,
}

impl HBTExperiment {
    /// Construct an HBT experiment model.
    pub fn new(
        source: SecondOrderCoherence,
        jitter_ps: f64,
        dead_time_ns: f64,
        dark_count_rate: f64,
    ) -> Self {
        Self {
            source,
            detector_jitter_ps: jitter_ps,
            dead_time_ns,
            dark_count_rate,
        }
    }

    /// Measured g²(τ) including detector imperfections.
    ///
    /// The dark-count-corrected g²(τ) is:
    ///
    ///   g²_meas(τ) = ρ² · g²_true(τ) + 2ρ(1−ρ) + (1−ρ)²
    ///
    /// where ρ = S/(S + d) is the signal purity and S is the signal count rate.
    ///
    /// Timing jitter broadens the peak but does not change g²(0) in an averaged sense;
    /// here we include the jitter as a convolution approximation that smoothes the
    /// τ = 0 feature by evaluating at an effective delay |τ| + σ_t.
    pub fn measured_g2(&self, tau_s: f64, signal_rate: f64) -> f64 {
        let total_rate = signal_rate + self.dark_count_rate;
        let rho = if total_rate > 0.0 {
            signal_rate / total_rate
        } else {
            1.0
        };

        // Jitter-broadened effective delay
        let jitter_s = self.detector_jitter_ps * 1e-12;
        let tau_eff = (tau_s.abs() + jitter_s).max(0.0);

        let g2_true = self.source.g2_at_tau(tau_eff);

        // Dark-count correction (two-channel, balanced, see Grangier et al.)
        rho * rho * g2_true + 2.0 * rho * (1.0 - rho) + (1.0 - rho) * (1.0 - rho)
    }

    /// Signal-to-noise ratio for a g²(0) measurement.
    ///
    /// For a photon-correlation histogram with coincidence window Δt, the SNR scales as
    ///
    ///   SNR = C_signal / √C_accidental = (S² Δt T) / √(S² Δt T · g²(0) + R_acc T)
    ///
    /// In the shot-noise-limited regime (many signal photons):
    ///
    ///   SNR ≈ S · √(Δt · T)  / √g²(0)   (for g²(0) > 0)
    ///
    /// We adopt a practical formula from the HBT literature:
    ///   SNR = S² · τ_c · T / √(S · T)  = S^{3/2} √T · τ_c
    ///
    /// where τ_c = coherence_time_s.
    pub fn snr(&self, signal_rate: f64, measurement_time_s: f64) -> f64 {
        let tau_c = self.source.coherence_time_s;
        // Coincidence rate ∝ S²·τ_c, Poisson noise ∝ √(S·T)
        signal_rate.powf(1.5) * measurement_time_s.sqrt() * tau_c
    }

    /// Minimum measurement time to achieve SNR > 3 (3σ significance).
    ///
    /// From SNR = S^{3/2} √T τ_c = 3 → T_min = 9 / (S³ τ_c²)
    pub fn required_measurement_time(&self, signal_rate: f64) -> f64 {
        let tau_c = self.source.coherence_time_s;
        9.0 / (signal_rate.powi(3) * tau_c * tau_c)
    }

    /// Accidental coincidence rate (Hz) for a given coincidence window Δt (ns).
    ///
    /// R_acc = 2 · (S + d)² · Δt
    pub fn accidental_rate(&self, signal_rate: f64, window_ns: f64) -> f64 {
        let total = signal_rate + self.dark_count_rate;
        let window_s = window_ns * 1e-9;
        2.0 * total * total * window_s
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ── g¹(τ) ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_g1_at_zero() {
        let coh = FirstOrderCoherence::new(1e-12, 800.0);
        let g1_0 = coh.g1_single_mode(0.0);
        // |g¹(0)| = exp(0) = 1
        assert_relative_eq!(g1_0.norm(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_g1_thermal_at_zero() {
        let coh = FirstOrderCoherence::new(1e-12, 800.0);
        let g1_0 = coh.g1_thermal(0.0);
        assert_relative_eq!(g1_0.norm(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_coherence_length() {
        let tau_c = 1e-12; // 1 ps
        let coh = FirstOrderCoherence::new(tau_c, 800.0);
        let lc = coh.coherence_length_m();
        assert_relative_eq!(lc, C0 * tau_c, epsilon = 1e-5);
    }

    #[test]
    fn test_linewidth_from_coherence() {
        let tau_c = 1e-9; // 1 ns coherence time
        let coh = FirstOrderCoherence::new(tau_c, 800.0);
        let dnu = coh.linewidth_hz();
        // Δν = 1/(π τ_c) ≈ 318.3 MHz
        assert_relative_eq!(dnu, 1.0 / (std::f64::consts::PI * tau_c), epsilon = 1e-5);
    }

    // ── g²(τ) ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_g2_zero_laser() {
        let src = SecondOrderCoherence::new(LightSource::Laser, 1e-12);
        assert_relative_eq!(src.g2_at_zero(), 1.0, epsilon = 1e-15);
        assert!(!src.is_antibunching());
        assert!(!src.is_bunching());
    }

    #[test]
    fn test_g2_zero_thermal() {
        let src = SecondOrderCoherence::new(LightSource::Thermal, 1e-12);
        assert_relative_eq!(src.g2_at_zero(), 2.0, epsilon = 1e-15);
        assert!(src.is_bunching());
    }

    #[test]
    fn test_g2_zero_single_photon() {
        let src = SecondOrderCoherence::new(LightSource::SinglePhoton, 1e-9);
        assert_relative_eq!(src.g2_at_zero(), 0.0, epsilon = 1e-15);
        assert!(src.is_antibunching());
    }

    #[test]
    fn test_g2_thermal_large_tau() {
        // g²(τ → ∞) → 1 for thermal
        let tau_c = 1e-12;
        let src = SecondOrderCoherence::new(LightSource::Thermal, tau_c);
        let g2_large = src.g2_at_tau(1e6 * tau_c);
        assert_relative_eq!(g2_large, 1.0, epsilon = 1e-6);
    }

    // ── Spatial coherence ─────────────────────────────────────────────────────

    #[test]
    fn test_coherence_radius_vczernike() {
        // λ = 633 nm, D = 1 mm, z = 1 m
        let sc = SpatialCoherence::new(633.0, 1.0, 1.0);
        let r_c = sc.coherence_radius_mm();
        // Expected: λz/(πD) = 633e-9 * 1 / (π * 1e-3) = 201.5 µm ≈ 0.2015 mm
        let expected_m = 633e-9 * 1.0 / (std::f64::consts::PI * 1e-3);
        assert_relative_eq!(r_c, expected_m * 1e3, epsilon = 1e-6);
    }

    #[test]
    fn test_mutual_coherence_at_zero_separation() {
        let sc = SpatialCoherence::new(800.0, 2.0, 0.5);
        // At zero separation, γ = 1
        assert_relative_eq!(sc.mutual_coherence(0.0), 1.0, epsilon = 1e-10);
    }

    // ── HBT experiment ────────────────────────────────────────────────────────

    #[test]
    fn test_hbt_accidental_rate() {
        let src = SecondOrderCoherence::new(LightSource::SinglePhoton, 1e-9);
        let hbt = HBTExperiment::new(src, 50.0, 10.0, 100.0);
        let rate = hbt.accidental_rate(1e6, 1.0); // S = 1 MHz, Δt = 1 ns
                                                  // R_acc = 2 * (1e6 + 100)² * 1e-9
        let total = 1e6_f64 + 100.0;
        let expected = 2.0 * total * total * 1e-9;
        assert_relative_eq!(rate, expected, epsilon = 1e-3);
    }

    #[test]
    fn test_hbt_g2_ideal_single_photon_at_zero() {
        let src = SecondOrderCoherence::new(LightSource::SinglePhoton, 1e-9);
        let hbt = HBTExperiment::new(src, 0.0, 0.0, 0.0); // ideal detector
                                                          // Ideal case: g²_meas(0) = g²_true(0+ε) → 0
        let g2 = hbt.measured_g2(0.0, 1e6);
        assert!(
            g2 < 0.1,
            "Expected near-zero g²(0) for single photon, got {g2}"
        );
    }

    #[test]
    fn test_hbt_snr_increases_with_time() {
        let src = SecondOrderCoherence::new(LightSource::Thermal, 1e-9);
        let hbt = HBTExperiment::new(src, 50.0, 10.0, 100.0);
        let snr_1s = hbt.snr(1e6, 1.0);
        let snr_100s = hbt.snr(1e6, 100.0);
        assert!(snr_100s > snr_1s);
    }
}
