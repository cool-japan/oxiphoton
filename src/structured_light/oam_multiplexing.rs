//! Orbital Angular Momentum (OAM) multiplexing for optical communications.
//!
//! Provides models for OAM-multiplexed links, mode sorters, spiral phase plates
//! and fractional OAM spectra.  All functions are pure Rust; no external
//! linear-algebra or BLAS dependencies.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// OAM-Multiplexed Optical Link
// ---------------------------------------------------------------------------

/// Model of an OAM-multiplexed free-space optical (FSO) link.
///
/// Each OAM channel l carries an independent data stream.  The link budget
/// accounts for Gaussian-beam diffraction loss at the receiver aperture and
/// atmospheric turbulence-induced crosstalk.
#[derive(Debug, Clone)]
pub struct OamMultiplexLink {
    /// OAM quantum numbers ℓ of all active channels.
    pub oam_channels: Vec<i32>,
    /// Centre wavelength (metres).
    pub wavelength: f64,
    /// Transmit beam waist w₀ (metres).
    pub beam_waist: f64,
    /// Link distance z (metres).
    pub distance: f64,
    /// Receiver aperture diameter D (metres).
    pub aperture: f64,
}

impl OamMultiplexLink {
    /// Construct a new OAM link.
    pub fn new(channels: Vec<i32>, wavelength: f64, w0: f64, distance: f64, aperture: f64) -> Self {
        Self {
            oam_channels: channels,
            wavelength,
            beam_waist: w0,
            distance,
            aperture,
        }
    }

    /// Number of active OAM channels.
    pub fn n_channels(&self) -> usize {
        self.oam_channels.len()
    }

    /// Check that all OAM channels are mutually orthogonal (all distinct).
    pub fn modes_orthogonal(&self) -> bool {
        let mut sorted = self.oam_channels.clone();
        sorted.sort_unstable();
        sorted.windows(2).all(|w| w[0] != w[1])
    }

    /// Aggregate channel capacity in bits/s/Hz (Shannon bound).
    ///
    /// C = Σ_k log₂(1 + SNR_k)
    ///
    /// Here we assume all channels have equal SNR and equal aperture efficiency.
    pub fn capacity_bits_per_s_per_hz(&self, snr_db: f64) -> f64 {
        let snr_linear = 10.0_f64.powf(snr_db / 10.0);
        self.oam_channels
            .iter()
            .map(|&ell| {
                let eta = self.aperture_efficiency(ell);
                (1.0 + snr_linear * eta).log2()
            })
            .sum()
    }

    /// Atmospheric crosstalk between OAM modes ℓ₁ and ℓ₂ in dB.
    ///
    /// Model (simplified after Tyler & Boyd 2009):
    ///   XT(Δl) ≈ −10 log₁₀(C_XT) where C_XT = (w/r₀)^(5/3) · f(Δl)
    /// and f(Δl) = 1/(1 + Δl²).
    ///
    /// `r0` is the Fried coherence parameter (metres).
    pub fn crosstalk_db(&self, l1: i32, l2: i32, r0: f64) -> f64 {
        if l1 == l2 {
            return 0.0; // same mode — no crosstalk
        }
        let w_at_z = self.beam_radius_at_distance();
        let delta_l = (l1 - l2).unsigned_abs() as f64;
        // Turbulence strength: (w/r0)^{5/3}
        let turb_strength = (w_at_z / r0).powf(5.0 / 3.0);
        let spectral_weight = 1.0 / (1.0 + delta_l * delta_l);
        let xt_linear = turb_strength * spectral_weight;
        // Return as a negative dB value (crosstalk into adjacent mode)
        10.0 * xt_linear.log10()
    }

    /// Mode purity after atmospheric propagation (fraction of power remaining in mode ℓ).
    ///
    /// Approximation: P_ℓ ≈ exp[−(w/r₀)^{5/3} · C_mode]
    /// where C_mode ≈ 1.3 (empirical from numerical studies).
    pub fn mode_purity_after_atmosphere(&self, ell: i32, r0: f64) -> f64 {
        let w = self.beam_radius_at_distance();
        let turb = (w / r0).powf(5.0 / 3.0);
        let l_abs = ell.unsigned_abs() as f64;
        // Higher |l| modes are slightly more affected
        let c_mode = 1.3 * (1.0 + 0.1 * l_abs);
        (-turb * c_mode).exp().clamp(0.0, 1.0)
    }

    /// Fraction of transmitted power captured by the receiver aperture for mode ℓ.
    ///
    /// For LG_{0}^{l} at distance z, the beam radius is w(z).  The power within
    /// aperture radius R_a = aperture/2 is
    ///
    ///   η = 1 − Γ(|l|+1, 2(R_a/w)²) / |l|!   (regularised incomplete Gamma)
    ///
    /// We use a simple exponential approximation valid for R_a ≈ w(z):
    ///   η ≈ 1 − exp(−2 R_a² / w²) · Σ_{k=0}^{|l|} (2 R_a²/w²)^k / k!
    pub fn aperture_efficiency(&self, ell: i32) -> f64 {
        let w = self.beam_radius_at_distance();
        let ra = self.aperture / 2.0;
        let u = 2.0 * ra * ra / (w * w);
        let l_abs = ell.unsigned_abs() as usize;
        // Partial sum of Poisson CDF terms
        let mut partial = 0.0_f64;
        let mut term = (-u).exp();
        partial += term;
        for k in 1..=l_abs {
            term *= u / k as f64;
            partial += term;
        }
        // Fraction of power outside aperture = partial → efficiency = 1 − partial
        (1.0 - partial).clamp(0.0, 1.0)
    }

    /// Beam radius w(z) of an LG beam at distance z from its waist.
    fn beam_radius_at_distance(&self) -> f64 {
        let zr = PI * self.beam_waist * self.beam_waist / self.wavelength;
        self.beam_waist * (1.0 + (self.distance / zr).powi(2)).sqrt()
    }
}

// ---------------------------------------------------------------------------
// OAM Mode Sorter
// ---------------------------------------------------------------------------

/// OAM mode sorter — an optical device that separates OAM modes to distinct
/// spatial positions using a log-polar coordinate transformation.
///
/// # Implementation model
/// The mode sorter maps azimuthal phase ℓφ → transverse position y ∝ ℓ,
/// enabling simultaneous detection of all OAM channels on a detector array.
#[derive(Debug, Clone)]
pub struct OamModeSorter {
    /// Number of OAM modes the sorter is designed for.
    pub n_modes: usize,
    /// Minimum OAM quantum number (may be negative).
    pub min_oam: i32,
    /// Maximum OAM quantum number.
    pub max_oam: i32,
    /// Current efficiency (0–1).
    pub efficiency: f64,
    /// Crosstalk in dB (negative: e.g. −20 dB).
    pub crosstalk_db: f64,
}

impl OamModeSorter {
    /// Construct a sorter covering OAM modes from `min_l` to `max_l` (inclusive).
    pub fn new(min_l: i32, max_l: i32) -> Self {
        let n_modes = (max_l - min_l).unsigned_abs() as usize + 1;
        Self {
            n_modes,
            min_oam: min_l,
            max_oam: max_l,
            efficiency: 0.90,    // realistic value ~90%
            crosstalk_db: -20.0, // −20 dB typical
        }
    }

    /// Normalised output position (0–1) for OAM mode ℓ in the output plane.
    ///
    /// The log-polar transformation maps ℓ → y position linearly.
    pub fn output_position(&self, ell: i32) -> f64 {
        if self.max_oam == self.min_oam {
            return 0.5;
        }
        (ell - self.min_oam) as f64 / (self.max_oam - self.min_oam) as f64
    }

    /// Mode separation in the output plane (mm) given detector pixel pitch (mm).
    pub fn mode_separation_mm(&self, pixel_pitch: f64) -> f64 {
        // Each mode maps to one pixel-pitch separation in the ideal case
        pixel_pitch
    }

    /// Theoretical efficiency of the log-polar mode sorter (ideal: 100%).
    pub fn theoretical_efficiency(&self) -> f64 {
        // With perfect optical components the efficiency approaches 1.
        // Practical limit set by phase-ramp discretisation ≈ 95%.
        0.95
    }

    /// Channel isolation (dB) — the ratio of power in the desired mode to
    /// leaked power in adjacent modes.
    pub fn channel_isolation_db(&self) -> f64 {
        -self.crosstalk_db // expressed as a positive isolation figure
    }
}

// ---------------------------------------------------------------------------
// Spiral Phase Plate
// ---------------------------------------------------------------------------

/// Spiral phase plate (SPP) — a transmissive optical element that imprints
/// an azimuthal phase ℓφ on a passing beam to create an OAM mode.
#[derive(Debug, Clone)]
pub struct SpiralPhasePlate {
    /// Topological charge ℓ (integer or fractional).
    pub topological_charge: i32,
    /// Outer diameter D (metres).
    pub diameter: f64,
    /// Physical step height h of the 2π discontinuity (metres).
    pub step_height: f64,
    /// Refractive index of the plate material.
    pub n_material: f64,
    /// Design wavelength (metres).
    pub wavelength: f64,
}

impl SpiralPhasePlate {
    /// Construct an SPP for integer topological charge.  Step height is
    /// computed automatically from λ/(n − 1).
    pub fn new(charge: i32, wavelength: f64, n: f64) -> Self {
        let h = wavelength / (n - 1.0);
        Self {
            topological_charge: charge,
            diameter: 10e-3, // default 10 mm
            step_height: h,
            n_material: n,
            wavelength,
        }
    }

    /// Phase imparted at azimuthal angle φ: φ_SPP = ℓ · φ (radians).
    pub fn phase_at_angle(&self, phi: f64) -> f64 {
        self.topological_charge as f64 * phi
    }

    /// Required step height h = λ / (n − 1) for a 2π phase shift.
    pub fn step_height_m(&self) -> f64 {
        if (self.n_material - 1.0).abs() < 1e-30 {
            return f64::INFINITY;
        }
        self.wavelength / (self.n_material - 1.0)
    }

    /// Diffraction efficiency for integer topological charge.
    ///
    /// For a perfect SPP with integer ℓ the efficiency into the LG_{0}^{l}
    /// mode is ≈ 100%.  Fabrication imperfections limit this to ~95%.
    pub fn efficiency(&self) -> f64 {
        // Simple model: η = sinc²(ε/2π) where ε is the residual phase error.
        // For an ideal integer plate ε = 0 → η = 1.0.
        1.0
    }

    /// Compute the OAM power spectrum for a *fractional* topological charge ℓ_f.
    ///
    /// A fractional SPP with charge ℓ_f decomposes into a superposition of
    /// integer OAM states.  The amplitude of each integer mode n is
    ///
    ///   c_n = sin(π(ℓ_f − n)) / (π(ℓ_f − n))  · exp(i π(ℓ_f − n))
    ///
    /// which is the Fourier series of the discontinuous phase ramp.
    ///
    /// Returns a `Vec<(integer_mode, power_fraction)>` centred on ℓ_f
    /// spanning ±n_modes/2 modes.
    pub fn fractional_mode_spectrum(charge: f64, n_modes: usize) -> Vec<(i32, f64)> {
        let l_center = charge.round() as i32;
        let half = (n_modes / 2) as i32;
        let mut spectrum: Vec<(i32, f64)> = Vec::with_capacity(n_modes);
        let mut total_power = 0.0_f64;

        for n in (l_center - half)..=(l_center + half) {
            let diff = charge - n as f64;
            let amplitude_sq = if diff.abs() < 1e-12 {
                1.0
            } else {
                let sinc = (PI * diff).sin() / (PI * diff);
                sinc * sinc
            };
            spectrum.push((n, amplitude_sq));
            total_power += amplitude_sq;
        }
        // Normalise so that total power = 1
        if total_power > 1e-30 {
            for entry in spectrum.iter_mut() {
                entry.1 /= total_power;
            }
        }
        spectrum
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // --- OamMultiplexLink ---
    #[test]
    fn oam_link_modes_orthogonal() {
        let link = OamMultiplexLink::new(vec![-2, -1, 0, 1, 2], 1550e-9, 0.05, 1000.0, 0.2);
        assert!(link.modes_orthogonal());
    }

    #[test]
    fn oam_link_modes_not_orthogonal_if_duplicate() {
        let link = OamMultiplexLink::new(vec![1, 1, 2], 1550e-9, 0.05, 1000.0, 0.2);
        assert!(!link.modes_orthogonal());
    }

    #[test]
    fn oam_link_capacity_increases_with_snr() {
        let link = OamMultiplexLink::new(vec![-1, 0, 1], 1550e-9, 0.05, 100.0, 0.1);
        let c_low = link.capacity_bits_per_s_per_hz(10.0);
        let c_high = link.capacity_bits_per_s_per_hz(30.0);
        assert!(c_high > c_low);
    }

    #[test]
    fn oam_link_aperture_efficiency_in_bounds() {
        let link = OamMultiplexLink::new(vec![0], 1550e-9, 0.02, 500.0, 0.1);
        let eta = link.aperture_efficiency(0);
        assert!((0.0..=1.0).contains(&eta));
    }

    #[test]
    fn oam_crosstalk_same_mode_zero() {
        let link = OamMultiplexLink::new(vec![0, 1], 1550e-9, 0.02, 100.0, 0.1);
        let xt = link.crosstalk_db(1, 1, 0.1);
        assert_abs_diff_eq!(xt, 0.0, epsilon = 1e-12);
    }

    // --- OamModeSorter ---
    #[test]
    fn sorter_n_modes() {
        let sorter = OamModeSorter::new(-3, 3);
        assert_eq!(sorter.n_modes, 7);
    }

    #[test]
    fn sorter_output_position_extremes() {
        let sorter = OamModeSorter::new(-2, 2);
        assert_abs_diff_eq!(sorter.output_position(-2), 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(sorter.output_position(2), 1.0, epsilon = 1e-14);
    }

    #[test]
    fn sorter_theoretical_efficiency() {
        let sorter = OamModeSorter::new(-5, 5);
        assert!(sorter.theoretical_efficiency() > 0.9);
    }

    // --- SpiralPhasePlate ---
    #[test]
    fn spp_step_height() {
        let spp = SpiralPhasePlate::new(1, 1064e-9, 1.5);
        assert_abs_diff_eq!(spp.step_height_m(), 1064e-9 / 0.5, epsilon = 1e-18);
    }

    #[test]
    fn spp_phase_at_pi() {
        let spp = SpiralPhasePlate::new(2, 1064e-9, 1.5);
        use std::f64::consts::PI;
        assert_abs_diff_eq!(spp.phase_at_angle(PI), 2.0 * PI, epsilon = 1e-14);
    }

    #[test]
    fn spp_fractional_spectrum_normalised() {
        let spectrum = SpiralPhasePlate::fractional_mode_spectrum(2.5, 11);
        let total: f64 = spectrum.iter().map(|&(_, p)| p).sum();
        assert_abs_diff_eq!(total, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn spp_fractional_spectrum_peak_at_nearest_integer() {
        // For charge = 3.0 (integer), the dominant mode should be l = 3
        let spectrum = SpiralPhasePlate::fractional_mode_spectrum(3.0, 11);
        let peak = spectrum
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        assert!(peak.is_some());
        assert_eq!(peak.unwrap().0, 3);
    }
}
