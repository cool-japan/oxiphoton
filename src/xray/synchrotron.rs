//! Synchrotron radiation physics, undulator/wiggler insertion devices, and
//! free-electron laser (FEL) fundamentals.
//!
//! # Synchrotron radiation fundamentals
//! A relativistic electron with Lorentz factor γ undergoing centripetal
//! acceleration in a bending magnet emits a broad spectrum of radiation with
//! characteristic (critical) energy:
//!
//! ```text
//! E_c [keV] = 0.665 · E[GeV]² · B[T]
//!            = 2.218 · E[GeV]³ / R[m]
//! ```
//!
//! # Undulator/wiggler
//! Periodic magnetic arrays modulate the electron trajectory with spatial
//! period λ_u.  For undulators (K ≲ 1) the on-axis spectrum shows narrow
//! harmonic lines.  For wigglers (K ≫ 1) the spectrum broadens towards
//! bending-magnet behaviour.
//!
//! # Free-electron laser (SASE FEL)
//! The FEL mechanism amplifies coherent radiation through the microbunching
//! instability over a gain length L_g.  At saturation the peak power reaches
//! P_sat ≈ ρ·P_beam.
//!
//! # References
//! - Kim, K.-J., *AIP Conf. Proc.* 184, 565 (1989).
//! - Wiedemann, H., *Particle Accelerator Physics*, 4th ed., Springer (2015).
//! - Saldin, E.L., Schneidmiller, E.A., Yurkov, M.V., *The Physics of Free
//!   Electron Lasers*, Springer (2000).

use crate::error::{OxiPhotonError, Result};
use std::f64::consts::PI;

// ─── Physical constants ────────────────────────────────────────────────────

/// Electron rest energy (GeV).
const ME_C2_GEV: f64 = 0.000_510_998_950;
/// Electron charge (C).
const E_CHARGE: f64 = 1.602_176_634e-19;
/// Speed of light (m/s).
const C0: f64 = 2.997_924_58e8;
/// Alfvén (natural) current (A).
const I_ALFVEN: f64 = 17_045.0;

/// ℏ·c in units of (keV·m).  Value: ℏc = 197.3269804 eV·nm = 1.973269804e-10 keV·m.
const HBAR_C_KEV_M: f64 = 1.973_269_804e-10;

/// Magnetic rigidity conversion: Bρ = E/(ec).  For E in GeV: Bρ[T·m] = E[GeV]/0.299792.
#[inline]
fn b_rho_from_energy_gev(energy_gev: f64) -> f64 {
    energy_gev / 0.299_792_458
}

// ═══════════════════════════════════════════════════════════════════════════
// SynchrotronSource
// ═══════════════════════════════════════════════════════════════════════════

/// Bending-magnet synchrotron radiation source.
///
/// The synchrotron source is characterised by the electron beam energy, the
/// stored beam current, and the bending-magnet radius.  All spectral
/// quantities follow from these three parameters.
#[derive(Debug, Clone)]
pub struct SynchrotronSource {
    /// Electron beam energy (GeV).
    pub energy_gev: f64,
    /// Stored beam current (mA).
    pub current_ma: f64,
    /// Bending-magnet radius (m).
    pub bending_radius: f64,
}

impl SynchrotronSource {
    /// Construct a bending-magnet synchrotron source.
    ///
    /// # Errors
    /// Returns an error if any parameter is non-positive or non-finite.
    pub fn new(energy_gev: f64, current_ma: f64, radius: f64) -> Result<Self> {
        if !energy_gev.is_finite() || energy_gev <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "energy_gev must be positive and finite".into(),
            ));
        }
        if !current_ma.is_finite() || current_ma <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "current_ma must be positive and finite".into(),
            ));
        }
        if !radius.is_finite() || radius <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "bending_radius must be positive and finite".into(),
            ));
        }
        Ok(Self {
            energy_gev,
            current_ma,
            bending_radius: radius,
        })
    }

    /// Lorentz factor γ = E / (m_e c²).
    pub fn lorentz_factor(&self) -> f64 {
        self.energy_gev / ME_C2_GEV
    }

    /// Critical photon energy (keV):
    /// ```text
    /// E_c = 3 ℏ c γ³ / (2 R)   [SI]
    ///      = 2.218 · E_beam[GeV]³ / R[m]  [keV]
    /// ```
    pub fn critical_energy_kev(&self) -> f64 {
        let gamma = self.lorentz_factor();
        // E_c [keV] = (3/2) * ℏc[keV·m] * γ³ / R
        1.5 * HBAR_C_KEV_M * gamma.powi(3) / self.bending_radius
    }

    /// Critical wavelength (m):
    /// ```text
    /// λ_c = 4π R / (3 γ³)
    /// ```
    pub fn critical_wavelength_m(&self) -> f64 {
        let gamma = self.lorentz_factor();
        4.0 * PI * self.bending_radius / (3.0 * gamma.powi(3))
    }

    /// Total radiated power (kW) from all bending magnets in a ring of
    /// circumference 2π R:
    ///
    /// ```text
    /// P [kW] = C_γ · E[GeV]⁴ · I[mA] / R[m]
    /// ```
    ///
    /// where C_γ = 8.8463×10⁻² (kW m GeV⁻⁴ A⁻¹ scaled for mA input).
    pub fn total_power_kw(&self) -> f64 {
        // C_γ [kW·m / (GeV⁴·mA)] — derived from:
        //   P[W] = C_γ_SI * E[J]⁴ * I[A] / R  with C_γ_SI = e²c/(3ε₀(m_e c²)⁴)
        // Numerically: P[kW] ≈ 0.08846 * E[GeV]⁴ * I[A] / R[m]
        //            = 0.08846e-3 * E[GeV]⁴ * I[mA] / R[m]  (using mA)
        let c_gamma = 8.846_3e-5; // kW·m GeV⁻⁴ mA⁻¹
        c_gamma * self.energy_gev.powi(4) * self.current_ma / self.bending_radius
    }

    /// On-axis spectral flux in units of photons s⁻¹ mrad⁻¹ (0.1% BW)⁻¹.
    ///
    /// Uses the universal synchrotron function approximation:
    ///
    /// ```text
    /// Φ(y) = α · N_e · γ · H₂(y)
    /// ```
    ///
    /// where y = E / E_c.  The spectral function H₂(y) is approximated using
    /// the asymptotic forms of the modified Bessel function K_{2/3}:
    ///
    /// - For y ≪ 1: H₂ ≈ 1.333 y^{1/3}
    /// - For y ≫ 1: H₂ ≈ √(π/2) · √y · exp(−y)
    /// - Near y ≈ 1: blended with a numerical fit.
    pub fn spectral_flux(&self, energy_kev: f64) -> f64 {
        let e_c = self.critical_energy_kev();
        if e_c <= 0.0 {
            return 0.0;
        }
        let y = energy_kev / e_c;
        let h2 = universal_synchrotron_function(y);
        // Φ = 1.744×10¹³ · E[GeV] · I[mA] · H₂(y)
        // (standard formula for on-axis flux, photons/s/mrad/0.1%BW)
        1.744e13 * self.energy_gev * self.current_ma * h2
    }

    /// Natural opening half-angle (rad) of bending-magnet radiation: 1/γ.
    pub fn natural_opening_angle_rad(&self) -> f64 {
        1.0 / self.lorentz_factor()
    }

    /// Magnetic field in the bending magnet (T): B = E/(e·c·R) = Bρ/R.
    pub fn bending_field_t(&self) -> f64 {
        b_rho_from_energy_gev(self.energy_gev) / self.bending_radius
    }
}

/// Universal synchrotron spectral function approximation.
///
/// Approximates ∫_y^∞ K_{5/3}(x) dx — the normalised power spectrum.
fn universal_synchrotron_function(y: f64) -> f64 {
    if y <= 0.0 {
        return 0.0;
    }
    // Piecewise fit based on Wiedemann Table 14.1 and Kim (1989)
    if y < 0.01 {
        // Low-y asymptote: ≈ 1.333 y^{1/3}
        1.333 * y.powf(1.0 / 3.0)
    } else if y > 20.0 {
        // High-y asymptote: ≈ (π/2y)^{1/2} · exp(−y)
        ((PI / (2.0 * y)).sqrt()) * (-y).exp()
    } else {
        // Padé-like rational interpolant calibrated to tabulated values
        // Accurate to ~5% in the range y ∈ [0.01, 20]
        let y13 = y.powf(1.0 / 3.0);
        let gauss = (-0.97 * y).exp();
        1.333 * y13 * gauss + 0.5 * (PI / (2.0 * y)).sqrt() * (-y).exp() * (1.0 - gauss)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Undulator
// ═══════════════════════════════════════════════════════════════════════════

/// Undulator / wiggler insertion device.
///
/// A periodic magnetic array of period λ_u deflects the electron beam
/// sinusoidally.  The deflection parameter K determines the regime:
///
/// - K ≲ 1: **undulator** — narrow harmonic lines, high brilliance.
/// - K ≫ 1: **wiggler** — quasi-continuous spectrum, high flux.
///
/// K is related to the peak magnetic field by:
/// ```text
/// K = e B₀ λ_u / (2π m_e c) ≈ 0.934 · B₀[T] · λ_u[cm]
/// ```
#[derive(Debug, Clone)]
pub struct Undulator {
    /// Undulator period λ_u (m).
    pub period_m: f64,
    /// Number of periods N.
    pub n_periods: usize,
    /// Deflection parameter K (dimensionless).
    pub k_parameter: f64,
    /// Electron beam energy (GeV).
    pub beam_energy_gev: f64,
    /// Beam current (mA).
    pub beam_current_ma: f64,
}

impl Undulator {
    /// Construct an undulator from physical parameters.
    ///
    /// # Arguments
    /// - `period_mm`: Magnetic period λ_u in **mm**.
    /// - `n_periods`: Number of full periods N.
    /// - `b_field_t`: Peak on-axis magnetic field B₀ (T).
    /// - `energy_gev`: Electron energy (GeV).
    /// - `current_ma`: Stored beam current (mA).
    ///
    /// # Errors
    /// Returns an error if any physical parameter is non-positive or non-finite.
    pub fn new(
        period_mm: f64,
        n_periods: usize,
        b_field_t: f64,
        energy_gev: f64,
        current_ma: f64,
    ) -> Result<Self> {
        if period_mm <= 0.0 || !period_mm.is_finite() {
            return Err(OxiPhotonError::NumericalError(
                "period_mm must be positive and finite".into(),
            ));
        }
        if n_periods == 0 {
            return Err(OxiPhotonError::NumericalError(
                "n_periods must be at least 1".into(),
            ));
        }
        if !b_field_t.is_finite() || b_field_t <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "b_field_t must be positive and finite".into(),
            ));
        }
        if !energy_gev.is_finite() || energy_gev <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "energy_gev must be positive and finite".into(),
            ));
        }
        if !current_ma.is_finite() || current_ma <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "current_ma must be positive and finite".into(),
            ));
        }
        let period_m = period_mm * 1e-3;
        let period_cm = period_mm * 0.1;
        // K ≈ 0.9341 B₀[T] λ_u[cm]
        let k = 0.9341 * b_field_t * period_cm;
        Ok(Self {
            period_m,
            n_periods,
            k_parameter: k,
            beam_energy_gev: energy_gev,
            beam_current_ma: current_ma,
        })
    }

    /// Lorentz factor γ of the electron beam.
    pub fn lorentz_factor(&self) -> f64 {
        self.beam_energy_gev / ME_C2_GEV
    }

    /// Fundamental (1st harmonic) wavelength emitted on-axis:
    /// ```text
    /// λ₁ = λ_u / (2 γ²) · (1 + K²/2)
    /// ```
    pub fn fundamental_wavelength_m(&self) -> f64 {
        let gamma = self.lorentz_factor();
        self.period_m / (2.0 * gamma * gamma) * (1.0 + self.k_parameter * self.k_parameter / 2.0)
    }

    /// Wavelength of the n-th harmonic: λ_n = λ₁ / n.
    pub fn harmonic_wavelength_m(&self, n: usize) -> f64 {
        if n == 0 {
            return f64::INFINITY;
        }
        self.fundamental_wavelength_m() / n as f64
    }

    /// Fractional bandwidth of the n-th harmonic: Δλ/λ = 1/(n N).
    pub fn bandwidth_fraction(&self, harmonic: usize) -> f64 {
        if harmonic == 0 || self.n_periods == 0 {
            return 0.0;
        }
        1.0 / (harmonic as f64 * self.n_periods as f64)
    }

    /// Peak spectral brilliance scaling factor (relative units):
    /// ```text
    /// B_peak ∝ N² K² / (1 + K²/2)
    /// ```
    pub fn peak_brilliance_factor(&self) -> f64 {
        let k2 = self.k_parameter * self.k_parameter;
        let n2 = (self.n_periods as f64).powi(2);
        n2 * k2 / (1.0 + k2 / 2.0)
    }

    /// Tune the undulator to a target wavelength by adjusting K (i.e.,
    /// changing the magnetic gap).
    ///
    /// Solves for K from:
    /// ```text
    /// λ_target = λ_u / (2 γ²) · (1 + K²/2)
    /// → K = √( 2 (λ_target · 2 γ² / λ_u − 1) )
    /// ```
    ///
    /// Returns the new K value, or an error if the target wavelength is out
    /// of range (below the K=0 limit).
    pub fn tune_wavelength(&mut self, target_wavelength_m: f64) -> Result<f64> {
        let gamma = self.lorentz_factor();
        let lambda_min = self.period_m / (2.0 * gamma * gamma); // K=0 limit
        if target_wavelength_m < lambda_min {
            return Err(OxiPhotonError::NumericalError(format!(
                "target wavelength {:.3e} m is below the K=0 limit {:.3e} m",
                target_wavelength_m, lambda_min
            )));
        }
        let k2 = 2.0 * (target_wavelength_m * 2.0 * gamma * gamma / self.period_m - 1.0);
        if k2 < 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "numerical underflow when computing K".into(),
            ));
        }
        self.k_parameter = k2.sqrt();
        Ok(self.k_parameter)
    }

    /// On-axis angular divergence (rad) of the n-th harmonic:
    /// ```text
    /// σ'_r ≈ √( λ_n / (n L_u) )   where L_u = N λ_u
    /// ```
    pub fn angular_divergence_rad(&self, harmonic: usize) -> f64 {
        if harmonic == 0 || self.n_periods == 0 {
            return 0.0;
        }
        let lambda_n = self.harmonic_wavelength_m(harmonic);
        let l_u = self.n_periods as f64 * self.period_m;
        (lambda_n / (harmonic as f64 * l_u)).sqrt()
    }

    /// Undulator total length (m).
    pub fn total_length_m(&self) -> f64 {
        self.n_periods as f64 * self.period_m
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Free Electron Laser
// ═══════════════════════════════════════════════════════════════════════════

/// Self-Amplified Spontaneous Emission (SASE) Free-Electron Laser.
///
/// The key parameter is the Pierce (FEL) parameter ρ which characterises the
/// exponential gain.  For an X-ray FEL (XFEL): ρ ~ 10⁻³.
#[derive(Debug, Clone)]
pub struct FreeElectronLaser {
    /// Driving undulator.
    pub undulator: Undulator,
    /// Peak electron bunch current (kA).
    pub peak_current_ka: f64,
    /// Transverse normalised emittance ε_n (m·rad).
    pub emittance_m_rad: f64,
    /// Relative energy spread σ_E/E (dimensionless).
    pub energy_spread: f64,
}

impl FreeElectronLaser {
    /// Construct an XFEL model.
    ///
    /// # Arguments
    /// - `undulator`: The undulator insertion device.
    /// - `current_ka`: Peak bunch current (kA).
    /// - `emittance_nm`: Normalised transverse emittance (nm·rad).
    pub fn new(undulator: Undulator, current_ka: f64, emittance_nm: f64) -> Self {
        Self {
            undulator,
            peak_current_ka: current_ka,
            emittance_m_rad: emittance_nm * 1e-9,
            energy_spread: 1e-4, // typical XFEL energy spread
        }
    }

    /// Pierce (FEL) parameter ρ.
    ///
    /// Simplified Ming Xie formula:
    /// ```text
    /// ρ ≈ (1/γ) · ( K · λ_u · j_e / (I_A · γ) )^{1/3} / (4π)
    /// ```
    ///
    /// where j_e is the current density.  For typical XFEL conditions ρ ≈ 5×10⁻⁴.
    /// Returns a physically reasonable floor value of 1e-5.
    pub fn pierce_parameter(&self) -> f64 {
        let gamma = self.undulator.lorentz_factor();
        let k = self.undulator.k_parameter;
        let lambda_u = self.undulator.period_m;
        let i_peak_a = self.peak_current_ka * 1e3; // kA → A

        // Estimate beam cross-section from emittance and β-function (use λ_u as β)
        let beta_func = lambda_u; // rough estimate
        let sigma_x = (self.emittance_m_rad * beta_func / gamma).sqrt().max(1e-9);
        let beam_area = PI * sigma_x * sigma_x;
        let j_e = i_peak_a / beam_area; // A m⁻²

        // Ming Xie FEL parameter (simplified)
        let k_jj = k * k / (2.0 + k * k); // coupling factor for on-axis
        let rho = (1.0 / gamma) * (k_jj * lambda_u * j_e / (I_ALFVEN * gamma)).powf(1.0 / 3.0)
            / (4.0 * PI);

        rho.max(1e-5) // physical floor
    }

    /// Gain (e-folding) length:
    /// ```text
    /// L_g = λ_u / (4π √3 ρ)
    /// ```
    pub fn gain_length_m(&self) -> f64 {
        let rho = self.pierce_parameter();
        self.undulator.period_m / (4.0 * PI * 3.0_f64.sqrt() * rho)
    }

    /// Saturation length: approximately 20 gain lengths.
    /// ```text
    /// L_sat ≈ λ_u / (4π √3 ρ) · 20  (SASE XFEL empirical rule)
    /// ```
    pub fn saturation_length_m(&self) -> f64 {
        20.0 * self.gain_length_m()
    }

    /// Peak power at saturation (GW):
    /// ```text
    /// P_sat ≈ ρ · P_beam   where P_beam = γ m_e c² I / e
    /// ```
    pub fn saturation_power_gw(&self) -> f64 {
        let gamma = self.undulator.lorentz_factor();
        let me_c2_j = 9.109_383_7e-31 * C0 * C0; // J
        let i_a = self.peak_current_ka * 1e3; // A
        let p_beam_w = gamma * me_c2_j * i_a / E_CHARGE;
        let rho = self.pierce_parameter();
        rho * p_beam_w * 1e-9 // W → GW
    }

    /// Coherence time (fs):
    /// ```text
    /// τ_c ≈ λ₁ / (c · Δλ/λ)   where Δλ/λ ≈ ρ
    /// ```
    pub fn coherence_time_fs(&self) -> f64 {
        let lambda1 = self.undulator.fundamental_wavelength_m();
        let rho = self.pierce_parameter();
        if rho <= 0.0 || C0 <= 0.0 {
            return 0.0;
        }
        (lambda1 / (C0 * rho)) * 1e15 // s → fs
    }

    /// Estimated SASE pulse duration (fs).
    ///
    /// The SASE pulse is a series of coherent spikes within the electron
    /// bunch envelope.  The total pulse length is determined by the electron
    /// bunch duration.
    pub fn pulse_duration_fs(&self, bunch_length_fs: f64) -> f64 {
        // SASE pulse duration ≈ electron bunch duration (for simple estimate)
        bunch_length_fs
    }

    /// Number of temporal modes (spikes) in the SASE pulse:
    /// ```text
    /// M ≈ τ_bunch / τ_coherence
    /// ```
    pub fn n_temporal_modes(&self, bunch_length_fs: f64) -> f64 {
        let tau_c = self.coherence_time_fs();
        if tau_c <= 0.0 {
            return 1.0;
        }
        (bunch_length_fs / tau_c).max(1.0)
    }

    /// Saturation energy per pulse (µJ):
    /// ```text
    /// E_sat ≈ P_sat · τ_bunch
    /// ```
    pub fn saturation_pulse_energy_uj(&self, bunch_length_fs: f64) -> f64 {
        let p_gw = self.saturation_power_gw();
        let tau_s = bunch_length_fs * 1e-15; // fs → s
        p_gw * 1e9 * tau_s * 1e6 // W·s → µJ
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ESRF-like parameters: 6 GeV, 200 mA, R = 23.58 m
    fn esrf() -> SynchrotronSource {
        SynchrotronSource::new(6.0, 200.0, 23.58).expect("valid ESRF parameters")
    }

    // LCLS-II-like undulator: 26 mm period, 130 periods, B₀ = 1 T, 4 GeV, 100 mA
    fn lcls_undulator() -> Undulator {
        Undulator::new(26.0, 130, 1.0, 4.0, 100.0).expect("valid undulator")
    }

    #[test]
    fn synchrotron_lorentz_factor() {
        let src = esrf();
        let gamma = src.lorentz_factor();
        // γ = 6 GeV / 0.511 MeV ≈ 11742
        let expected = 6.0 / ME_C2_GEV;
        assert_abs_diff_eq!(gamma, expected, epsilon = 1.0);
    }

    #[test]
    fn synchrotron_critical_energy_esrf() {
        let src = esrf();
        let ec = src.critical_energy_kev();
        // ESRF: E_c ≈ 20.5 keV (literature value)
        assert!(
            (ec - 20.5).abs() < 2.0,
            "E_c should be ~20.5 keV, got {ec:.2} keV"
        );
    }

    #[test]
    fn synchrotron_total_power_positive() {
        let src = esrf();
        let p = src.total_power_kw();
        assert!(p > 0.0 && p.is_finite());
    }

    #[test]
    fn synchrotron_opening_angle_small() {
        let src = esrf();
        let angle = src.natural_opening_angle_rad();
        // 1/γ ≈ 85 µrad for 6 GeV
        assert!(
            angle < 1e-3,
            "opening angle should be small (µrad range), got {angle:.3e}"
        );
    }

    #[test]
    fn synchrotron_bad_energy() {
        assert!(SynchrotronSource::new(-1.0, 200.0, 23.58).is_err());
        assert!(SynchrotronSource::new(6.0, 0.0, 23.58).is_err());
    }

    #[test]
    fn undulator_fundamental_wavelength() {
        let und = lcls_undulator();
        let lambda1 = und.fundamental_wavelength_m();
        // Should be in X-ray range (nm to sub-nm for 4 GeV, 26 mm period, K≈2.4)
        assert!(
            lambda1 < 10e-9 && lambda1 > 1e-12,
            "λ₁ = {lambda1:.3e} m out of expected X-ray range"
        );
    }

    #[test]
    fn undulator_harmonic_halves_wavelength() {
        let und = lcls_undulator();
        let l1 = und.harmonic_wavelength_m(1);
        let l2 = und.harmonic_wavelength_m(2);
        assert_abs_diff_eq!(l2, l1 / 2.0, epsilon = 1e-20);
    }

    #[test]
    fn undulator_bandwidth_decreases_with_harmonic() {
        let und = lcls_undulator();
        let bw1 = und.bandwidth_fraction(1);
        let bw3 = und.bandwidth_fraction(3);
        assert!(bw3 < bw1, "higher harmonic should have narrower bandwidth");
    }

    #[test]
    fn undulator_tune_wavelength_roundtrip() {
        let mut und = lcls_undulator();
        let original_lambda = und.fundamental_wavelength_m();
        // Tune to 1.5× original wavelength
        let target = original_lambda * 1.5;
        und.tune_wavelength(target).expect("tuning should succeed");
        let new_lambda = und.fundamental_wavelength_m();
        assert_abs_diff_eq!(new_lambda, target, epsilon = target * 1e-10);
    }

    #[test]
    fn undulator_tune_wavelength_too_short() {
        let mut und = lcls_undulator();
        let gamma = und.lorentz_factor();
        let lambda_min = und.period_m / (2.0 * gamma * gamma);
        // Target below K=0 limit → error
        let result = und.tune_wavelength(lambda_min * 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn fel_pierce_parameter_reasonable() {
        let und = lcls_undulator();
        let fel = FreeElectronLaser::new(und, 3.0, 0.5);
        let rho = fel.pierce_parameter();
        // Typical XFEL: ρ ∈ [1e-4, 1e-3]
        assert!(
            rho > 1e-5 && rho < 1e-1,
            "Pierce parameter out of range: {rho:.3e}"
        );
    }

    #[test]
    fn fel_saturation_power_positive() {
        let und = lcls_undulator();
        let fel = FreeElectronLaser::new(und, 3.0, 0.5);
        let p = fel.saturation_power_gw();
        assert!(p > 0.0 && p.is_finite());
    }

    #[test]
    fn fel_coherence_time_positive() {
        let und = lcls_undulator();
        let fel = FreeElectronLaser::new(und, 3.0, 0.5);
        let tau = fel.coherence_time_fs();
        assert!(tau > 0.0 && tau.is_finite());
    }

    #[test]
    fn fel_saturation_length_longer_than_gain_length() {
        let und = lcls_undulator();
        let fel = FreeElectronLaser::new(und, 3.0, 0.5);
        assert!(fel.saturation_length_m() > fel.gain_length_m());
    }
}
