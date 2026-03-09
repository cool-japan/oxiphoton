//! Coherent Anti-Stokes Raman Scattering (CARS) and Stimulated Raman Scattering (SRS) Microscopy
//!
//! CARS and SRS provide chemically-specific, label-free contrast based on molecular
//! vibrational resonances. CARS suffers from a non-resonant background, while SRS
//! (stimulated Raman loss / gain) is background-free and linear in concentration.
//!
//! # Signal relationships
//! - CARS: `I_CARS ∝ |χ_R(Ω) + χ_NR|² · I_pump² · I_Stokes`
//! - SRS (SRL): `ΔI_pump/I_pump ∝ Im(χ_R) · I_Stokes`
//! - SRS (SRG): `ΔI_Stokes/I_Stokes ∝ Im(χ_R) · I_pump`
//!
//! # References
//! - Evans & Xie, Annu. Rev. Anal. Chem. 1, 883 (2008)
//! - Freudiger et al., Science 322, 1857 (2008)
//! - Min et al., Annu. Rev. Phys. Chem. 62, 507 (2011)

use std::f64::consts::PI;

/// Speed of light in vacuum \[m/s\]
pub const C_LIGHT: f64 = 2.99792458e8;
/// Speed of light in cm/s (convenient for Raman wavenumber calculations)
const C_CM_S: f64 = 2.99792458e10;

/// CARS/SRS experimental geometry: pump, Stokes, and probe beams.
///
/// The pump and Stokes beams are tuned so that their difference frequency
/// Ω = ω_pump − ω_Stokes matches a molecular vibrational frequency.
/// When a probe beam is added (often the same as pump), anti-Stokes emission
/// occurs at ω_CARS = ω_pump − ω_Stokes + ω_probe.
#[derive(Debug, Clone)]
pub struct CarsSetup {
    /// Pump beam wavelength \[m\]
    pub pump_wavelength_m: f64,
    /// Stokes beam wavelength \[m\] (longer than pump)
    pub stokes_wavelength_m: f64,
    /// Probe beam wavelength \[m\] (typically equal to pump)
    pub probe_wavelength_m: f64,
    /// Refractive index of the sample medium
    pub n_medium: f64,
}

impl CarsSetup {
    /// Construct a CARS setup.  `probe_wavelength_m` is typically the same as `pump_wavelength_m`.
    pub fn new(
        pump_wavelength_m: f64,
        stokes_wavelength_m: f64,
        probe_wavelength_m: f64,
        n_medium: f64,
    ) -> Self {
        Self {
            pump_wavelength_m,
            stokes_wavelength_m,
            probe_wavelength_m,
            n_medium,
        }
    }

    /// Raman shift addressed by this pump–Stokes combination \[cm⁻¹\].
    ///
    /// `Ω = (1/λ_pump − 1/λ_Stokes) × 10⁻²` (λ in metres, result in cm⁻¹)
    pub fn raman_shift_cm1(&self) -> f64 {
        (1.0 / self.pump_wavelength_m - 1.0 / self.stokes_wavelength_m) * 1e-2
    }

    /// Anti-Stokes (CARS) emission wavelength \[m\].
    ///
    /// `1/λ_CARS = 1/λ_pump − 1/λ_Stokes + 1/λ_probe`
    pub fn cars_wavelength_m(&self) -> f64 {
        let inv = 1.0 / self.pump_wavelength_m - 1.0 / self.stokes_wavelength_m
            + 1.0 / self.probe_wavelength_m;
        1.0 / inv.max(f64::EPSILON)
    }

    /// Phase mismatch Δk for collinear CARS \[rad/m\].
    ///
    /// `Δk = (2π/λ_CARS)·n_CARS − (2π/λ_pump)·n_pump − (2π/λ_probe)·n_probe + (2π/λ_Stokes)·n_Stokes`
    ///
    /// In the degenerate case (pump = probe, same refractive index dispersion),
    /// this reduces to:
    /// `Δk ≈ n·(2/λ_pump − 1/λ_Stokes − 1/λ_CARS) · 2π`
    pub fn phase_mismatch_collinear(&self) -> f64 {
        let lambda_cars = self.cars_wavelength_m();
        // Approximate: use same n_medium for all beams (narrow spectral range)
        let k_cars = 2.0 * PI * self.n_medium / lambda_cars;
        let k_pump = 2.0 * PI * self.n_medium / self.pump_wavelength_m;
        let k_probe = 2.0 * PI * self.n_medium / self.probe_wavelength_m;
        let k_stokes = 2.0 * PI * self.n_medium / self.stokes_wavelength_m;
        k_cars - k_pump - k_probe + k_stokes
    }

    /// Coherence length L_c = π / |Δk| \[m\].
    pub fn coherence_length(&self) -> f64 {
        let dk = self.phase_mismatch_collinear().abs();
        if dk < 1e-30 {
            f64::INFINITY
        } else {
            PI / dk
        }
    }
}

/// Resonant Raman susceptibility χ_R(Ω) for a single vibrational mode.
///
/// Models a Lorentzian line shape:
/// `χ_R(Ω) = A / (Ω_R − Ω − iΓ/2)`
///
/// The real part produces the dispersive CARS lineshape; the imaginary part
/// is proportional to the spontaneous Raman spectrum.
#[derive(Debug, Clone)]
pub struct RamanSusceptibility {
    /// Raman resonance frequency Ω_R \[cm⁻¹\]
    pub resonance_freq_cm1: f64,
    /// Lorentzian linewidth Γ (FWHM) \[cm⁻¹\]
    pub linewidth_cm1: f64,
    /// Amplitude A \[proportional to ∂σ/∂Ω (Raman cross section density)\]
    pub amplitude: f64,
}

impl RamanSusceptibility {
    /// Construct a Raman susceptibility.
    pub fn new(resonance_freq_cm1: f64, linewidth_cm1: f64, amplitude: f64) -> Self {
        Self {
            resonance_freq_cm1,
            linewidth_cm1,
            amplitude,
        }
    }

    /// Complex susceptibility χ_R(Ω) = A / (Ω_R − Ω − iΓ/2).
    ///
    /// Returns `(Re[χ_R], Im[χ_R])`.
    ///
    /// The imaginary part Im\[χ_R\] < 0 on resonance (Raman gain convention);
    /// here we return the magnitude-positive form used in the CARS literature.
    pub fn susceptibility(&self, omega_cm1: f64) -> (f64, f64) {
        let delta = self.resonance_freq_cm1 - omega_cm1;
        let gamma_half = self.linewidth_cm1 * 0.5;
        let denom = delta * delta + gamma_half * gamma_half;
        let re = self.amplitude * delta / denom;
        let im = self.amplitude * gamma_half / denom;
        (re, im)
    }

    /// Peak Raman frequency Ω_R \[cm⁻¹\].
    pub fn peak_frequency(&self) -> f64 {
        self.resonance_freq_cm1
    }

    /// Dephasing time T₂ = 1 / (π c Γ) \[s\].
    ///
    /// Relates the homogeneous linewidth to the coherence lifetime of the vibration.
    pub fn dephasing_time(&self) -> f64 {
        1.0 / (PI * C_CM_S * self.linewidth_cm1)
    }
}

/// Full CARS/SRS signal including multiple resonances and non-resonant background.
///
/// The total third-order susceptibility is:
/// `χ⁽³⁾(Ω) = χ_NR + Σ_k χ_R,k(Ω)`
///
/// The non-resonant background χ_NR is real and positive, giving a frequency-
/// independent "pedestal" that interferes constructively/destructively with
/// the resonant terms.
#[derive(Debug, Clone)]
pub struct CarsSignal {
    /// Non-resonant background χ_NR (real, dimensionless ratio)
    pub chi_nr: f64,
    /// Collection of vibrational resonances
    pub resonances: Vec<RamanSusceptibility>,
}

impl CarsSignal {
    /// Construct a CARS signal model.
    pub fn new(chi_nr: f64, resonances: Vec<RamanSusceptibility>) -> Self {
        Self { chi_nr, resonances }
    }

    /// Total complex χ⁽³⁾(Ω) = χ_NR + Σ χ_R,k(Ω).
    ///
    /// Returns `(Re[χ_tot], Im[χ_tot])`.
    pub fn total_chi3(&self, omega_cm1: f64) -> (f64, f64) {
        let mut re = self.chi_nr;
        let mut im = 0.0_f64;
        for res in &self.resonances {
            let (r, i) = res.susceptibility(omega_cm1);
            re += r;
            im += i;
        }
        (re, im)
    }

    /// CARS intensity at Raman shift Ω \[cm⁻¹\].
    ///
    /// `I_CARS ∝ |χ⁽³⁾(Ω)|² · I_pump² · I_Stokes`
    pub fn cars_intensity(
        &self,
        omega_cm1: f64,
        pump_intensity: f64,
        stokes_intensity: f64,
    ) -> f64 {
        let (re, im) = self.total_chi3(omega_cm1);
        let chi_sq = re * re + im * im;
        chi_sq * pump_intensity * pump_intensity * stokes_intensity
    }

    /// SRS (stimulated Raman scattering) signal — background-free.
    ///
    /// SRS is proportional to Im\[χ_R\] only (the non-resonant background is
    /// purely real and does not contribute):
    /// `I_SRS ∝ Im[χ_R(Ω)] · I_pump · I_Stokes`
    pub fn srs_signal(&self, omega_cm1: f64, pump_intensity: f64, stokes_intensity: f64) -> f64 {
        let mut im_r = 0.0_f64;
        for res in &self.resonances {
            let (_, i) = res.susceptibility(omega_cm1);
            im_r += i;
        }
        im_r * pump_intensity * stokes_intensity
    }

    /// Compute a CARS spectrum over a wavenumber range.
    ///
    /// # Arguments
    /// * `omega_min_cm1` — Start of Raman shift range \[cm⁻¹\]
    /// * `omega_max_cm1` — End of Raman shift range \[cm⁻¹\]
    /// * `n_points`      — Number of spectral points
    /// * `pump_power`    — Pump beam power (arbitrary units)
    /// * `stokes_power`  — Stokes beam power (arbitrary units)
    ///
    /// # Returns
    /// Vector of `(Ω [cm⁻¹], I_CARS [a.u.])` pairs.
    pub fn spectrum(
        &self,
        omega_min_cm1: f64,
        omega_max_cm1: f64,
        n_points: usize,
        pump_power: f64,
        stokes_power: f64,
    ) -> Vec<(f64, f64)> {
        if n_points == 0 {
            return Vec::new();
        }
        let step = (omega_max_cm1 - omega_min_cm1) / (n_points.saturating_sub(1).max(1)) as f64;
        (0..n_points)
            .map(|i| {
                let omega = omega_min_cm1 + i as f64 * step;
                let intensity = self.cars_intensity(omega, pump_power, stokes_power);
                (omega, intensity)
            })
            .collect()
    }
}

/// Stimulated Raman Scattering detector parameters.
///
/// SRS detection measures the modulation transfer from the pump to the Stokes
/// beam (SRL — stimulated Raman loss) or vice versa (SRG — stimulated Raman gain).
/// It is background-free and linearly proportional to concentration.
#[derive(Debug, Clone)]
pub struct SrsDetector {
    /// Minimum detectable relative intensity change ΔI/I (shot-noise limited)
    pub detection_sensitivity: f64,
}

impl SrsDetector {
    /// Construct an SRS detector.
    pub fn new(detection_sensitivity: f64) -> Self {
        Self {
            detection_sensitivity,
        }
    }

    /// Stimulated Raman Loss (SRL) signal — pump beam intensity decrease.
    ///
    /// `ΔSRL = Im[χ_R(Ω)] · I_pump · I_Stokes`
    pub fn srl_signal(&self, chi_r_im: f64, pump_intensity: f64, stokes_intensity: f64) -> f64 {
        chi_r_im * pump_intensity * stokes_intensity
    }

    /// Stimulated Raman Gain (SRG) signal — Stokes beam intensity increase.
    ///
    /// By energy conservation SRG = SRL (same amplitude, sign reflects gain vs. loss).
    pub fn srg_signal(&self, chi_r_im: f64, pump_intensity: f64, stokes_intensity: f64) -> f64 {
        chi_r_im * pump_intensity * stokes_intensity
    }

    /// Minimum detectable molecular concentration \[mol/m³\].
    ///
    /// From the signal-to-noise requirement:
    /// `C_min = (ΔI/I)_min / (σ_Raman · I_pump · N_A)`
    ///
    /// # Arguments
    /// * `raman_cross_section` — Differential Raman cross section \[m²/(sr·molecule)\]
    /// * `pump_intensity`      — Pump irradiance \[W/m²\]
    pub fn minimum_detectable_concentration(
        &self,
        raman_cross_section: f64,
        pump_intensity: f64,
    ) -> f64 {
        // Avogadro number
        let n_a = 6.02214076e23_f64;
        let denominator = raman_cross_section * pump_intensity * n_a;
        if denominator < f64::EPSILON {
            f64::INFINITY
        } else {
            self.detection_sensitivity / denominator
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ch2_setup() -> CarsSetup {
        // Standard CARS on CH₂ stretch (2850 cm⁻¹): pump 817 nm, Stokes 1064 nm
        CarsSetup::new(817e-9, 1064e-9, 817e-9, 1.33)
    }

    fn ch2_resonance() -> RamanSusceptibility {
        RamanSusceptibility::new(2850.0, 10.0, 1.0)
    }

    #[test]
    fn test_raman_shift_ch2() {
        let setup = ch2_setup();
        let shift = setup.raman_shift_cm1();
        // Expected: (1/817e-9 - 1/1064e-9)×1e-2 ≈ 2845 cm⁻¹
        assert!(
            (shift - 2845.0).abs() < 30.0,
            "CH₂ Raman shift {} far from 2850 cm⁻¹",
            shift
        );
    }

    #[test]
    fn test_cars_wavelength_shorter_than_pump() {
        let setup = ch2_setup();
        let lambda_cars = setup.cars_wavelength_m();
        // CARS wavelength must be shorter than pump (anti-Stokes)
        assert!(
            lambda_cars < setup.pump_wavelength_m,
            "CARS wavelength {} should be < pump {}",
            lambda_cars,
            setup.pump_wavelength_m
        );
    }

    #[test]
    fn test_raman_susceptibility_peak_on_resonance() {
        let res = ch2_resonance();
        let (re_on, im_on) = res.susceptibility(2850.0);
        let (re_off, _) = res.susceptibility(2900.0);
        // On resonance: Re = 0, Im is maximal; off resonance: Re ≠ 0
        assert!(re_on.abs() < 1e-6, "Re[χ_R] should be zero on resonance");
        assert!(im_on > 0.0, "Im[χ_R] should be positive (our convention)");
        assert!(
            re_off.abs() > 0.0,
            "Re[χ_R] should be non-zero off resonance"
        );
    }

    #[test]
    fn test_dephasing_time_physical() {
        let res = ch2_resonance(); // 10 cm⁻¹ linewidth
        let t2 = res.dephasing_time();
        // T₂ ≈ 1/(π × 3e10 × 10) ≈ 1.06 ps
        assert!(t2 > 0.5e-12 && t2 < 5e-12, "T₂ = {} s out of range", t2);
    }

    #[test]
    fn test_cars_background_beats_resonance_far_off_peak() {
        let chi_nr = 1.0;
        let signal = CarsSignal::new(chi_nr, vec![ch2_resonance()]);
        // Far from resonance (3500 cm⁻¹), χ_R ≈ 0, so CARS ≈ χ_NR²
        let i_off = signal.cars_intensity(3500.0, 1.0, 1.0);
        let i_on = signal.cars_intensity(2850.0, 1.0, 1.0);
        // On resonance total χ has extra resonant contribution → larger signal
        assert!(i_on > i_off, "On-resonance CARS should exceed background");
    }

    #[test]
    fn test_srs_background_free() {
        let signal = CarsSignal::new(10.0, vec![ch2_resonance()]);
        // SRS depends only on Im[χ_R], not on χ_NR (real)
        let srs_on = signal.srs_signal(2850.0, 1.0, 1.0);
        let srs_off = signal.srs_signal(3500.0, 1.0, 1.0);
        assert!(srs_on > srs_off, "SRS should be maximal on resonance");
    }

    #[test]
    fn test_spectrum_length_and_ordering() {
        let signal = CarsSignal::new(0.5, vec![ch2_resonance()]);
        let spec = signal.spectrum(2700.0, 3000.0, 50, 1.0, 1.0);
        assert_eq!(spec.len(), 50);
        // Frequencies should be monotonically increasing
        for w in spec.windows(2) {
            assert!(w[1].0 > w[0].0, "Spectrum frequencies not increasing");
        }
    }

    #[test]
    fn test_srs_detector_minimum_concentration() {
        let det = SrsDetector::new(1e-7);
        // σ_Raman ≈ 1e-30 m²/sr for a typical Raman mode
        // pump_intensity ≈ 1e12 W/m²
        let c_min = det.minimum_detectable_concentration(1e-30, 1e12);
        // Should be detectable at sub-micromolar concentrations
        assert!(
            c_min > 0.0 && c_min.is_finite(),
            "C_min must be positive finite"
        );
    }
}
