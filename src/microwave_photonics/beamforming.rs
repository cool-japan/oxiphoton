/// Photonic beamforming for phased array antennas.
///
/// Provides true-time-delay (TTD) beamforming using optical delay lines,
/// array factor computation, beam pattern analysis, and optical beamforming
/// network (BFN) models for Butler, Blass, and Nolen matrix architectures.
use num_complex::Complex64;
use std::f64::consts::PI;

/// Speed of light in vacuum \[m/s\].
const SPEED_OF_LIGHT: f64 = 2.997_924_58e8;

// ─── PhotonicBeamformer ───────────────────────────────────────────────────────

/// True-time-delay photonic beamformer for a 1-D phased array.
///
/// Optical delay lines provide the per-element time delays needed for
/// squint-free beamsteering over the full RF bandwidth (true time delay).
#[derive(Debug, Clone)]
pub struct PhotonicBeamformer {
    /// Number of antenna elements.
    pub n_elements: usize,
    /// Inter-element spacing \[m\].
    pub element_spacing: f64,
    /// RF wavelength at the design frequency \[m\].
    pub wavelength_rf: f64,
    /// Optical carrier wavelength \[m\].
    pub wavelength_optical: f64,
    /// Current per-element time delays \[s\].
    pub delays: Vec<f64>,
}

impl PhotonicBeamformer {
    /// Create a photonic beamformer with N elements and half-wavelength spacing.
    ///
    /// # Arguments
    /// * `n` – number of antenna elements
    /// * `spacing` – inter-element spacing \[m\]
    /// * `lambda_rf` – RF wavelength at the design frequency \[m\]
    /// * `lambda_opt` – optical carrier wavelength \[m\]
    pub fn new(n: usize, spacing: f64, lambda_rf: f64, lambda_opt: f64) -> Self {
        PhotonicBeamformer {
            n_elements: n,
            element_spacing: spacing,
            wavelength_rf: lambda_rf,
            wavelength_optical: lambda_opt,
            delays: vec![0.0; n],
        }
    }

    /// Create a beamformer pre-steered to a given angle.
    pub fn new_steered(
        n: usize,
        spacing: f64,
        lambda_rf: f64,
        lambda_opt: f64,
        theta_deg: f64,
    ) -> Self {
        let mut bf = Self::new(n, spacing, lambda_rf, lambda_opt);
        bf.set_steering_angle(theta_deg);
        bf
    }

    /// Set the beam steering angle and update all element delays.
    ///
    /// For element k the required true-time delay is:
    ///   τₖ = k · d · sin(θ) / c
    pub fn set_steering_angle(&mut self, theta_deg: f64) {
        let theta_rad = theta_deg.to_radians();
        for k in 0..self.n_elements {
            self.delays[k] = self.required_delay(k, theta_deg);
            let _ = (theta_rad, k); // used inside required_delay
        }
    }

    /// Required true-time delay for element `k` to steer to angle `theta_deg`.
    ///
    ///   τₖ = k · d · sin(θ) / c
    pub fn required_delay(&self, element: usize, theta_deg: f64) -> f64 {
        let theta_rad = theta_deg.to_radians();
        element as f64 * self.element_spacing * theta_rad.sin() / SPEED_OF_LIGHT
    }

    /// Array factor AF(θ) as a complex phasor.
    ///
    ///   AF(θ) = Σₖ exp(j · 2π/λ_RF · k · d · (sin(θ) − sin(θ₀)))
    ///
    /// where θ₀ is the current steering angle (encoded in `self.delays`).
    pub fn array_factor(&self, theta_deg: f64) -> Complex64 {
        let theta_rad = theta_deg.to_radians();
        let k_rf = 2.0 * PI / self.wavelength_rf;

        self.delays
            .iter()
            .enumerate()
            .fold(Complex64::new(0.0, 0.0), |acc, (k, &tau_k)| {
                // Spatial phase from actual steering delay
                let steering_phase = 2.0 * PI * SPEED_OF_LIGHT * tau_k / self.wavelength_rf;
                // Phase from element position relative to arrival direction
                let spatial_phase = k_rf * k as f64 * self.element_spacing * theta_rad.sin();
                let total_phase = spatial_phase - steering_phase;
                acc + Complex64::new(total_phase.cos(), total_phase.sin())
            })
    }

    /// Beam pattern: returns `(angle_deg, power_db)` pairs over 0° … 180°.
    ///
    /// Power is normalized to the main beam peak.
    pub fn beam_pattern(&self, n_angles: usize) -> Vec<(f64, f64)> {
        if n_angles == 0 {
            return Vec::new();
        }
        let mut pattern: Vec<(f64, f64)> = (0..n_angles)
            .map(|i| {
                let theta = 180.0 * i as f64 / (n_angles as f64 - 1.0).max(1.0);
                let power = self.array_factor(theta).norm_sqr();
                (theta, power)
            })
            .collect();

        // Normalize to peak
        let peak = pattern.iter().map(|(_, p)| *p).fold(0.0_f64, f64::max);

        if peak > 0.0 {
            for (_, p) in &mut pattern {
                let normalized = *p / peak;
                *p = if normalized > 1e-12 {
                    10.0 * normalized.log10()
                } else {
                    -120.0
                };
            }
        }
        pattern
    }

    /// Half-power beamwidth (HPBW) estimated analytically \[degrees\].
    ///
    ///   HPBW ≈ 0.886 · λ_RF / (N · d)  \[radians\], converted to degrees.
    pub fn hpbw_deg(&self) -> f64 {
        let hpbw_rad = 0.886 * self.wavelength_rf / (self.n_elements as f64 * self.element_spacing);
        hpbw_rad.to_degrees()
    }

    /// Beam squint: frequency-induced beam displacement \[degrees/GHz\].
    ///
    /// For a phase-steered (non-TTD) system:
    ///   squint ≈ −θ₀ / f_RF \[deg/GHz\]
    ///
    /// For a true-time-delay system, squint is essentially zero.
    /// This returns the phase-steered squint for reference.
    pub fn beam_squint_deg_per_ghz(&self) -> f64 {
        // For phase shifter (non-TTD) beam squint:
        // Δθ ≈ -θ₀ / f_RF_GHz
        // For TTD (this implementation) squint → 0 by design
        // Return the theoretical phase-steered squint magnitude for comparison
        let theta_0 = self.current_steering_angle_deg();
        let f_rf_ghz = SPEED_OF_LIGHT / self.wavelength_rf * 1e-9;
        -theta_0 / f_rf_ghz
    }

    /// First sidelobe level relative to main beam peak \[dB\].
    ///
    /// For a uniform linear array: first sidelobe ≈ −13.3 dB.
    pub fn first_sidelobe_db(&self) -> f64 {
        -13.26 // theoretical value for uniform amplitude weighting
    }

    /// Estimate the current steering angle from the delay profile \[degrees\].
    fn current_steering_angle_deg(&self) -> f64 {
        if self.n_elements < 2 || self.element_spacing <= 0.0 {
            return 0.0;
        }
        // Use the slope of the delay profile
        let delta_tau = self.delays.last().copied().unwrap_or(0.0)
            - self.delays.first().copied().unwrap_or(0.0);
        let sin_theta =
            delta_tau * SPEED_OF_LIGHT / ((self.n_elements as f64 - 1.0) * self.element_spacing);
        sin_theta.clamp(-1.0, 1.0).asin().to_degrees()
    }
}

// ─── OpticalBfn ───────────────────────────────────────────────────────────────

/// Beamforming network architecture type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BfnArchitecture {
    /// Butler matrix — produces a set of orthogonal beams. N ports → N beams.
    /// Requires log₂(N) stages of 3-dB couplers and phase shifters.
    Butler,
    /// Blass matrix — arbitrary beam directions using cascaded transmission lines.
    Blass,
    /// Nolen matrix — a variant of the Butler matrix with simplified construction.
    Nolen,
}

/// Optical beamforming network (BFN) connecting N antenna ports to N beams.
#[derive(Debug, Clone)]
pub struct OpticalBfn {
    /// Number of input/output ports.
    pub n_ports: usize,
    /// BFN architecture.
    pub architecture: BfnArchitecture,
}

impl OpticalBfn {
    /// Create an optical BFN.
    ///
    /// # Arguments
    /// * `n` – number of ports (for Butler must be a power of 2)
    /// * `arch` – beamforming network architecture
    pub fn new(n: usize, arch: BfnArchitecture) -> Self {
        OpticalBfn {
            n_ports: n,
            architecture: arch,
        }
    }

    /// Number of simultaneous independent beams the BFN can form.
    pub fn n_beams(&self) -> usize {
        match self.architecture {
            BfnArchitecture::Butler => {
                // Butler: N × N orthogonal beams for N = 2^m
                self.n_ports
            }
            BfnArchitecture::Blass => {
                // Blass: arbitrary number of beams; typically N beams from N ports
                self.n_ports
            }
            BfnArchitecture::Nolen => self.n_ports,
        }
    }

    /// Estimate the total insertion loss of the BFN \[dB\].
    ///
    /// Based on component count and typical per-element loss.
    pub fn insertion_loss_db(&self) -> f64 {
        match self.architecture {
            BfnArchitecture::Butler => {
                // log₂(N) stages, each with 2 dB coupler loss + 0.5 dB phase shifter
                let stages = (self.n_ports as f64).log2().ceil();
                stages * (2.0 + 0.5)
            }
            BfnArchitecture::Blass => {
                // Blass has higher loss due to signal distribution via couplers
                let stages = self.n_ports as f64;
                stages * 1.5
            }
            BfnArchitecture::Nolen => {
                // Nolen: similar to Butler but potentially fewer stages
                let stages = (self.n_ports as f64).log2().ceil();
                stages * 2.2
            }
        }
    }

    /// Theoretical beam directions \[degrees\] for a Butler matrix BFN.
    ///
    /// Returns the N orthogonal beam positions for a half-wavelength spaced array.
    pub fn butler_beam_directions(&self) -> Vec<f64> {
        // For an N-element Butler matrix with d = λ/2:
        //   sin(θₘ) = (2m − 1) / N,  m = 1 … N/2   (positive and negative)
        let n = self.n_ports;
        let mut directions = Vec::with_capacity(n);
        for m in 1..=(n / 2) {
            let sin_theta = (2 * m - 1) as f64 / n as f64;
            if sin_theta.abs() <= 1.0 {
                directions.push(-sin_theta.asin().to_degrees());
                directions.push(sin_theta.asin().to_degrees());
            }
        }
        directions.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        directions
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn default_beamformer() -> PhotonicBeamformer {
        // 8-element array, λ/2 spacing at 10 GHz
        let lambda_rf = SPEED_OF_LIGHT / 10.0e9;
        PhotonicBeamformer::new(8, lambda_rf / 2.0, lambda_rf, 1550e-9)
    }

    #[test]
    fn test_array_factor_broadside_is_n() {
        let bf = default_beamformer();
        // At broadside (0°) and zero delays, AF = N (all phasors aligned)
        let af = bf.array_factor(90.0); // 90° = broadside for this convention
                                        // All delays are zero initially → spatial phases differ
                                        // At θ where k·d·sin(θ) = 0 → θ = 0 for sine-based formula
        let af0 = bf.array_factor(0.0);
        assert_abs_diff_eq!(af0.norm(), bf.n_elements as f64, epsilon = 1e-6);
        let _ = af;
    }

    #[test]
    fn test_required_delay_element_zero() {
        let bf = default_beamformer();
        // Element 0 always has zero delay
        assert_abs_diff_eq!(bf.required_delay(0, 30.0), 0.0, epsilon = 1e-20);
    }

    #[test]
    fn test_required_delay_positive_angle() {
        let bf = default_beamformer();
        let lambda_rf = SPEED_OF_LIGHT / 10.0e9;
        let d = lambda_rf / 2.0;
        let tau = bf.required_delay(3, 30.0);
        let expected = 3.0 * d * (30.0_f64.to_radians().sin()) / SPEED_OF_LIGHT;
        assert_abs_diff_eq!(tau, expected, epsilon = 1e-15);
    }

    #[test]
    fn test_hpbw_decreases_with_more_elements() {
        let lambda_rf = SPEED_OF_LIGHT / 10.0e9;
        let d = lambda_rf / 2.0;
        let bf4 = PhotonicBeamformer::new(4, d, lambda_rf, 1550e-9);
        let bf16 = PhotonicBeamformer::new(16, d, lambda_rf, 1550e-9);
        assert!(
            bf16.hpbw_deg() < bf4.hpbw_deg(),
            "HPBW should decrease with more elements"
        );
    }

    #[test]
    fn test_hpbw_formula() {
        let lambda_rf = SPEED_OF_LIGHT / 10.0e9;
        let d = lambda_rf / 2.0;
        let n = 8;
        let bf = PhotonicBeamformer::new(n, d, lambda_rf, 1550e-9);
        let expected_rad = 0.886 * lambda_rf / (n as f64 * d);
        assert_abs_diff_eq!(bf.hpbw_deg(), expected_rad.to_degrees(), epsilon = 1e-6);
    }

    #[test]
    fn test_beam_pattern_normalized_to_zero_db() {
        let bf = default_beamformer();
        let pattern = bf.beam_pattern(181);
        // Peak of normalized pattern must be 0 dB
        let peak = pattern
            .iter()
            .map(|(_, p)| *p)
            .fold(f64::NEG_INFINITY, f64::max);
        assert_abs_diff_eq!(peak, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_beam_pattern_length() {
        let bf = default_beamformer();
        let pattern = bf.beam_pattern(91);
        assert_eq!(pattern.len(), 91);
    }

    #[test]
    fn test_set_steering_angle_updates_delays() {
        let mut bf = default_beamformer();
        bf.set_steering_angle(30.0);
        // All delays should be ≥ 0 for positive steering angles
        assert!(bf.delays.iter().all(|&d| d >= 0.0));
        // Delays should be monotonically increasing for positive angle
        for i in 1..bf.n_elements {
            assert!(bf.delays[i] >= bf.delays[i - 1]);
        }
    }

    #[test]
    fn test_first_sidelobe_level() {
        let bf = default_beamformer();
        // Uniform array theoretical first sidelobe: -13.26 dB
        assert_abs_diff_eq!(bf.first_sidelobe_db(), -13.26, epsilon = 0.01);
    }

    // ── OpticalBfn ──────────────────────────────────────────────────────────

    #[test]
    fn test_butler_n_beams() {
        let bfn = OpticalBfn::new(8, BfnArchitecture::Butler);
        assert_eq!(bfn.n_beams(), 8);
    }

    #[test]
    fn test_butler_insertion_loss_increases_with_n() {
        let bfn4 = OpticalBfn::new(4, BfnArchitecture::Butler);
        let bfn16 = OpticalBfn::new(16, BfnArchitecture::Butler);
        assert!(bfn16.insertion_loss_db() > bfn4.insertion_loss_db());
    }

    #[test]
    fn test_butler_beam_directions_count() {
        let bfn = OpticalBfn::new(8, BfnArchitecture::Butler);
        let dirs = bfn.butler_beam_directions();
        // Should produce 8 beam directions (symmetric ±4 pairs)
        assert_eq!(dirs.len(), 8, "8-port Butler: expect 8 beam directions");
    }

    #[test]
    fn test_blass_insertion_loss_positive() {
        let bfn = OpticalBfn::new(4, BfnArchitecture::Blass);
        assert!(bfn.insertion_loss_db() > 0.0);
    }
}
