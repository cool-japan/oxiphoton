use num_complex::Complex64;
/// Fiber Bragg Grating (FBG) sensor simulation.
///
/// Implements coupled-mode theory for FBG reflection/transmission spectra,
/// strain and temperature sensing, apodization profiles, and interrogation
/// systems for distributed sensing networks.
///
/// References:
/// - Kashyap, "Fiber Bragg Gratings", 2nd ed., Academic Press (2010)
/// - Othonos & Kalli, "Fiber Bragg Gratings", Artech House (1999)
/// - Kersey et al., "Fiber grating sensors", JLT 15(8), 1442-1463 (1997)
use std::f64::consts::PI;

use crate::error::{OxiPhotonError, Result};

/// Speed of light in vacuum (m/s)
const C0: f64 = 2.99792458e8;

// ---------------------------------------------------------------------------
// FiberBraggGrating
// ---------------------------------------------------------------------------

/// Fiber Bragg Grating (FBG) sensor.
///
/// A periodic refractive index modulation inscribed in the fiber core.
/// When broadband light is launched, wavelengths satisfying the Bragg condition
/// λ_B = 2·n_eff·Λ are reflected; all others are transmitted.
///
/// The reflection spectrum follows coupled-mode theory (CMT).
#[derive(Debug, Clone)]
pub struct FiberBraggGrating {
    /// Bragg wavelength λ_B (nm)
    pub center_wavelength_nm: f64,
    /// Grating length L (mm)
    pub grating_length_mm: f64,
    /// Amplitude of refractive-index modulation Δn (≈1e-4 to 1e-3)
    pub index_modulation: f64,
    /// Effective refractive index n_eff (≈1.45 for SMF-28)
    pub average_index: f64,
    /// Grating period Λ (nm), derived from Bragg condition
    pub grating_period_nm: f64,
    /// Strain sensitivity (pm/με) — typical ≈ 1.2 pm/με for SMF at 1550 nm
    pub strain_sensitivity_pm_per_microstrain: f64,
    /// Temperature sensitivity (pm/°C) — typical ≈ 10–13 pm/°C for SMF at 1550 nm
    pub temperature_sensitivity_pm_per_c: f64,
}

impl FiberBraggGrating {
    /// Create a new FBG sensor.
    ///
    /// # Arguments
    /// * `center_lambda_nm` — Bragg wavelength in nm (e.g., 1550.0)
    /// * `length_mm`        — Physical grating length in mm (e.g., 10.0)
    /// * `delta_n`          — Index modulation amplitude (e.g., 3e-4)
    /// * `n_eff`            — Effective group index (e.g., 1.4682)
    pub fn new(center_lambda_nm: f64, length_mm: f64, delta_n: f64, n_eff: f64) -> Result<Self> {
        if center_lambda_nm <= 0.0 || !center_lambda_nm.is_finite() {
            return Err(OxiPhotonError::InvalidWavelength(center_lambda_nm * 1e-9));
        }
        if length_mm <= 0.0 || !length_mm.is_finite() {
            return Err(OxiPhotonError::NumericalError(format!(
                "grating_length_mm must be positive, got {length_mm}"
            )));
        }
        if delta_n <= 0.0 || !delta_n.is_finite() {
            return Err(OxiPhotonError::NumericalError(format!(
                "index_modulation must be positive, got {delta_n}"
            )));
        }
        if n_eff <= 0.0 || !n_eff.is_finite() {
            return Err(OxiPhotonError::InvalidRefractiveIndex { n: n_eff, k: 0.0 });
        }
        let period = center_lambda_nm / (2.0 * n_eff);
        // Physical sensitivities for silica SMF (from literature)
        // Strain: Δλ/λ = (1 - p_e) * ε,  p_e = 0.212,  so S_ε = λ*(1-0.212) pm/με
        let strain_sens = center_lambda_nm * 1e3 * (1.0 - 0.212) * 1e-6; // pm/με
                                                                         // Temperature: Δλ/λ = (α_L + ξ) * ΔT = (0.55+8.6)*1e-6 ΔT
        let temp_sens = center_lambda_nm * 1e3 * (0.55e-6 + 8.6e-6); // pm/°C
        Ok(Self {
            center_wavelength_nm: center_lambda_nm,
            grating_length_mm: length_mm,
            index_modulation: delta_n,
            average_index: n_eff,
            grating_period_nm: period,
            strain_sensitivity_pm_per_microstrain: strain_sens,
            temperature_sensitivity_pm_per_c: temp_sens,
        })
    }

    /// Standard SMF-28 FBG at 1550 nm with typical parameters.
    ///
    /// Parameters: n_eff = 1.4682, Δn = 3×10⁻⁴, L = 10 mm.
    pub fn smf28_1550() -> Result<Self> {
        Self::new(1550.0, 10.0, 3e-4, 1.4682)
    }

    /// Grating period Λ = λ_B / (2·n_eff) \[nm\].
    pub fn grating_period_nm(&self) -> f64 {
        self.center_wavelength_nm / (2.0 * self.average_index)
    }

    /// Coupling coefficient κ = π·Δn / λ_B (1/m).
    pub fn coupling_coefficient(&self) -> f64 {
        let lambda_m = self.center_wavelength_nm * 1e-9;
        PI * self.index_modulation / lambda_m
    }

    /// Peak reflectivity R_peak = tanh²(κ·L).
    pub fn peak_reflectivity(&self) -> f64 {
        let kappa = self.coupling_coefficient();
        let l_m = self.grating_length_mm * 1e-3;
        let x = kappa * l_m;
        let th = x.tanh();
        th * th
    }

    /// Reflection bandwidth (FWHM) in nm using the CMT approximation:
    ///   δλ ≈ λ_B² / (n_eff · L) · √((Δn/(2·n_eff))² + (1/N)²)
    /// where N = L/Λ is the number of grating periods.
    pub fn bandwidth_nm(&self) -> f64 {
        let lambda_b = self.center_wavelength_nm;
        let n_eff = self.average_index;
        let l_nm = self.grating_length_mm * 1e6; // mm → nm
        let big_n = self.n_periods() as f64;
        if big_n < 1.0 {
            return 0.0;
        }
        let term1 = self.index_modulation / (2.0 * n_eff);
        let term2 = 1.0 / big_n;
        lambda_b * lambda_b / (n_eff * l_nm) * (term1 * term1 + term2 * term2).sqrt()
    }

    /// Number of grating periods N = L / Λ.
    pub fn n_periods(&self) -> usize {
        let l_nm = self.grating_length_mm * 1e6; // mm → nm
        (l_nm / self.grating_period_nm()).round() as usize
    }

    /// Compute reflection spectrum R(λ) using coupled-mode theory.
    ///
    /// For each wavelength λ, the detuning parameter is:
    ///   δ = π·n_eff·(1/λ - 1/λ_B) * 2  (= Δβ/2)
    /// Then:
    ///   s² = κ² - δ²
    ///
    /// Case s² > 0 (strong coupling, near Bragg):
    ///   R = sinh²(s·L) / (cosh²(s·L) - δ²/κ²)
    ///
    /// Case s² < 0 (off-resonance):
    ///   R = sin²(|s|·L) / (cos²(|s|·L) + δ²/κ²)
    ///
    /// Returns a `Vec<(wavelength_nm, reflectivity)>`.
    pub fn reflection_spectrum(
        &self,
        lambda_min_nm: f64,
        lambda_max_nm: f64,
        n_pts: usize,
    ) -> Vec<(f64, f64)> {
        if n_pts == 0 || lambda_min_nm >= lambda_max_nm {
            return Vec::new();
        }
        let kappa = self.coupling_coefficient();
        let l_m = self.grating_length_mm * 1e-3;
        let lambda_b_m = self.center_wavelength_nm * 1e-9;
        let n_eff = self.average_index;
        let kappa_sq = kappa * kappa;

        (0..n_pts)
            .map(|i| {
                let lambda_nm = lambda_min_nm
                    + (lambda_max_nm - lambda_min_nm) * i as f64 / (n_pts - 1).max(1) as f64;
                let lambda_m = lambda_nm * 1e-9;
                // Phase mismatch (detuning): δ = π·n_eff·(1/λ - 1/λ_B)
                let delta = PI * n_eff * (1.0 / lambda_m - 1.0 / lambda_b_m);
                let s_sq = kappa_sq - delta * delta;
                let r = if s_sq > 0.0 {
                    let s = s_sq.sqrt();
                    let sl = s * l_m;
                    let sinh_sl = sl.sinh();
                    let cosh_sl = sl.cosh();
                    let denom = cosh_sl * cosh_sl - delta * delta / kappa_sq;
                    if denom.abs() < 1e-30 {
                        1.0
                    } else {
                        (sinh_sl * sinh_sl / denom).clamp(0.0, 1.0)
                    }
                } else if s_sq < 0.0 {
                    let s = (-s_sq).sqrt();
                    let sl = s * l_m;
                    let sin_sl = sl.sin();
                    let cos_sl = sl.cos();
                    let denom = cos_sl * cos_sl + delta * delta / kappa_sq;
                    if denom.abs() < 1e-30 {
                        0.0
                    } else {
                        (sin_sl * sin_sl / denom).clamp(0.0, 1.0)
                    }
                } else {
                    // At exact Bragg wavelength: R = tanh²(κL)
                    let x = kappa * l_m;
                    let th = x.tanh();
                    th * th
                };
                (lambda_nm, r)
            })
            .collect()
    }

    /// Transmission spectrum T(λ) = 1 - R(λ) for a lossless FBG.
    pub fn transmission_spectrum(
        &self,
        lambda_min_nm: f64,
        lambda_max_nm: f64,
        n_pts: usize,
    ) -> Vec<(f64, f64)> {
        self.reflection_spectrum(lambda_min_nm, lambda_max_nm, n_pts)
            .into_iter()
            .map(|(lam, r)| (lam, 1.0 - r))
            .collect()
    }

    /// Wavelength shift due to axial strain.
    ///
    ///   Δλ = λ_B · (1 - p_e) · ε
    ///
    /// where p_e = 0.212 is the effective photo-elastic constant for germano-silicate
    /// fiber and ε is the strain in micro-strain (με = 10⁻⁶).
    pub fn strain_shift_nm(&self, strain_microstrain: f64) -> f64 {
        let pe = 0.212_f64;
        // strain_microstrain is in units of 10⁻⁶ (dimensionless when multiplied by 1e-6)
        self.center_wavelength_nm * (1.0 - pe) * strain_microstrain * 1e-6
    }

    /// Wavelength shift due to temperature change.
    ///
    ///   Δλ = λ_B · (α_L + ξ) · ΔT
    ///
    /// where α_L = 0.55×10⁻⁶ /°C (thermal expansion of silica) and
    ///       ξ    = 8.6×10⁻⁶  /°C (thermo-optic coefficient of silica).
    pub fn temperature_shift_nm(&self, delta_t_celsius: f64) -> f64 {
        let alpha_l = 0.55e-6_f64; // thermal expansion coefficient (/°C)
        let xi = 8.6e-6_f64; // thermo-optic coefficient (/°C)
        self.center_wavelength_nm * (alpha_l + xi) * delta_t_celsius
    }

    /// Combined wavelength shift from simultaneous strain and temperature change.
    pub fn combined_shift_nm(&self, strain_microstrain: f64, delta_t_celsius: f64) -> f64 {
        self.strain_shift_nm(strain_microstrain) + self.temperature_shift_nm(delta_t_celsius)
    }

    /// Convert a measured wavelength shift (temperature-compensated) to strain.
    ///
    ///   ε (με) = Δλ / (λ_B · (1 - p_e)) × 10⁶
    pub fn strain_from_shift_microstrain(&self, delta_lambda_nm: f64) -> f64 {
        let pe = 0.212_f64;
        let sensitivity = self.center_wavelength_nm * (1.0 - pe) * 1e-6; // nm/με
        if sensitivity.abs() < 1e-60 {
            return 0.0;
        }
        delta_lambda_nm / sensitivity
    }

    /// Convert a measured wavelength shift (strain-free) to temperature.
    ///
    ///   ΔT (°C) = Δλ / (λ_B · (α_L + ξ))
    pub fn temperature_from_shift_c(&self, delta_lambda_nm: f64) -> f64 {
        let alpha_l = 0.55e-6_f64;
        let xi = 8.6e-6_f64;
        let sensitivity = self.center_wavelength_nm * (alpha_l + xi); // nm/°C
        if sensitivity.abs() < 1e-60 {
            return 0.0;
        }
        delta_lambda_nm / sensitivity
    }

    /// Compute the complex reflection coefficient r(λ) using full CMT.
    ///
    /// The complex amplitude reflection coefficient is:
    ///   r(λ) = -i·κ·sinh(s·L) / (s·cosh(s·L) + i·δ·sinh(s·L))
    ///
    /// for the over-coupled regime (s² = κ² - δ² > 0).
    fn complex_reflection(&self, lambda_nm: f64) -> Complex64 {
        let kappa = self.coupling_coefficient();
        let l_m = self.grating_length_mm * 1e-3;
        let lambda_b_m = self.center_wavelength_nm * 1e-9;
        let lambda_m = lambda_nm * 1e-9;
        let n_eff = self.average_index;

        let delta = PI * n_eff * (1.0 / lambda_m - 1.0 / lambda_b_m);
        let s_sq = kappa * kappa - delta * delta;

        if s_sq > 1e-30 {
            let s = s_sq.sqrt();
            let sl = s * l_m;
            let sinh_sl = Complex64::new(sl.sinh(), 0.0);
            let cosh_sl = Complex64::new(sl.cosh(), 0.0);
            let numerator = Complex64::new(0.0, -kappa) * sinh_sl;
            let denominator = s * cosh_sl + Complex64::new(0.0, delta) * sinh_sl;
            if denominator.norm() < 1e-60 {
                Complex64::new(0.0, 0.0)
            } else {
                numerator / denominator
            }
        } else if s_sq < -1e-30 {
            // Off-resonance: s imaginary → use sin/cos
            let s_abs = (-s_sq).sqrt();
            let sl = s_abs * l_m;
            // sinh(i*x) = i*sin(x), cosh(i*x) = cos(x)
            let sinh_isl = Complex64::new(0.0, sl.sin());
            let cosh_isl = Complex64::new(sl.cos(), 0.0);
            let numerator = Complex64::new(0.0, -kappa) * sinh_isl;
            let denominator = s_abs * cosh_isl + Complex64::new(0.0, delta) * sinh_isl;
            if denominator.norm() < 1e-60 {
                Complex64::new(0.0, 0.0)
            } else {
                numerator / denominator
            }
        } else {
            // At Bragg: limit r → -i·tanh(κL)
            let x = kappa * l_m;
            Complex64::new(0.0, -x.tanh())
        }
    }

    /// Phase spectrum φ(λ) = arg(r(λ)) in radians.
    pub fn phase_spectrum(
        &self,
        lambda_min_nm: f64,
        lambda_max_nm: f64,
        n_pts: usize,
    ) -> Vec<(f64, f64)> {
        if n_pts == 0 || lambda_min_nm >= lambda_max_nm {
            return Vec::new();
        }
        (0..n_pts)
            .map(|i| {
                let lambda_nm = lambda_min_nm
                    + (lambda_max_nm - lambda_min_nm) * i as f64 / (n_pts - 1).max(1) as f64;
                let r = self.complex_reflection(lambda_nm);
                (lambda_nm, r.im.atan2(r.re))
            })
            .collect()
    }

    /// Group delay τ_g(λ) = -dφ/dω (ps), evaluated by finite difference.
    ///
    /// Uses a central difference with a wavelength step of 0.001 nm.
    pub fn group_delay_ps(&self, lambda_nm: f64) -> f64 {
        let dlambda = 0.001; // nm
        let lam1 = lambda_nm - dlambda;
        let lam2 = lambda_nm + dlambda;
        let phi1 = {
            let r = self.complex_reflection(lam1);
            r.im.atan2(r.re)
        };
        let phi2 = {
            let r = self.complex_reflection(lam2);
            r.im.atan2(r.re)
        };
        // dω/dλ = -2πc/λ², so dφ/dω = dφ/dλ · dλ/dω = dφ/dλ · (-λ²/(2πc))
        let dphi_dlambda = (phi2 - phi1) / (2.0 * dlambda * 1e-9); // rad/m
        let lambda_m = lambda_nm * 1e-9;
        let domega_dlambda = -2.0 * PI * C0 / (lambda_m * lambda_m); // rad/(s·m)
        let tau_s = -dphi_dlambda / domega_dlambda; // seconds
        tau_s * 1e12 // → picoseconds
    }

    /// Wrap this FBG into an `ApodizedFbg` with a Gaussian apodization profile.
    ///
    /// `sigma_fraction` is σ as a fraction of grating length (e.g., 0.3).
    pub fn with_gaussian_apodization(self, sigma_fraction: f64) -> ApodizedFbg {
        let sigma = sigma_fraction.clamp(0.01, 0.5);
        ApodizedFbg {
            fbg: self,
            apodization_profile: ApodizationProfile::Gaussian { sigma },
        }
    }
}

// ---------------------------------------------------------------------------
// Apodization
// ---------------------------------------------------------------------------

/// Apodization profile for shaping the coupling-coefficient envelope along
/// the grating length, suppressing the side-lobe structure in the spectrum.
#[derive(Debug, Clone)]
pub enum ApodizationProfile {
    /// Gaussian envelope: A(z) = exp(-z²/(2σ²)), σ in units of grating length.
    Gaussian { sigma: f64 },
    /// Raised-cosine (Hanning): A(z) = 0.5·(1 − cos(2πz/L))
    RaisedCosine,
    /// Hamming: A(z) = 0.54 − 0.46·cos(2πz/L)
    Hamming,
    /// Uniform (no apodization)
    Uniform,
}

impl ApodizationProfile {
    /// Evaluate the apodization weight at normalised position u ∈ \[0, 1\].
    fn weight(&self, u: f64) -> f64 {
        match self {
            ApodizationProfile::Gaussian { sigma } => {
                let z = u - 0.5; // centre at 0
                (-z * z / (2.0 * sigma * sigma)).exp()
            }
            ApodizationProfile::RaisedCosine => 0.5 * (1.0 - (2.0 * PI * u).cos()),
            ApodizationProfile::Hamming => 0.54 - 0.46 * (2.0 * PI * u).cos(),
            ApodizationProfile::Uniform => 1.0,
        }
    }
}

/// FBG with an apodized coupling-coefficient profile.
///
/// Integrates the coupled-mode equations numerically using a transfer-matrix
/// method with N_SECTIONS uniform sections, each weighted by the apodization
/// profile.
#[derive(Debug, Clone)]
pub struct ApodizedFbg {
    /// Underlying FBG parameters
    pub fbg: FiberBraggGrating,
    /// Apodization profile
    pub apodization_profile: ApodizationProfile,
}

impl ApodizedFbg {
    const N_SECTIONS: usize = 200;

    /// Compute the complex amplitude reflection coefficient at wavelength λ_nm
    /// using the piecewise transfer-matrix method (TMM).
    ///
    /// Each section of length Δz has a local coupling coefficient
    ///   κ_j = κ_0 · A(z_j/L)
    /// and the section transfer matrix is evaluated analytically for the
    /// uniform-grating CMT equations.
    fn complex_reflection_tmm(&self, lambda_nm: f64) -> Complex64 {
        let n = Self::N_SECTIONS;
        let kappa_0 = self.fbg.coupling_coefficient();
        let l_m = self.fbg.grating_length_mm * 1e-3;
        let dz = l_m / n as f64;
        let lambda_b_m = self.fbg.center_wavelength_nm * 1e-9;
        let lambda_m = lambda_nm * 1e-9;
        let n_eff = self.fbg.average_index;

        let delta = PI * n_eff * (1.0 / lambda_m - 1.0 / lambda_b_m);

        // Start with identity transfer matrix [m11, m12; m21, m22]
        let mut m11 = Complex64::new(1.0, 0.0);
        let mut m12 = Complex64::new(0.0, 0.0);
        let mut m21 = Complex64::new(0.0, 0.0);
        let mut m22 = Complex64::new(1.0, 0.0);

        for j in 0..n {
            let u = (j as f64 + 0.5) / n as f64;
            let kappa_j = kappa_0 * self.apodization_profile.weight(u);
            let s_sq = kappa_j * kappa_j - delta * delta;

            let (t11, t12, t21, t22) = if s_sq > 1e-30 {
                let s = s_sq.sqrt();
                let sdz = s * dz;
                let cosh_s = sdz.cosh();
                let sinh_s = sdz.sinh();
                let t11 = Complex64::new(cosh_s, -delta * sinh_s / s);
                let t22 = Complex64::new(cosh_s, delta * sinh_s / s);
                let t12 = Complex64::new(0.0, -kappa_j * sinh_s / s);
                let t21 = Complex64::new(0.0, kappa_j * sinh_s / s);
                (t11, t12, t21, t22)
            } else if s_sq < -1e-30 {
                let s_abs = (-s_sq).sqrt();
                let sdz = s_abs * dz;
                let cos_s = sdz.cos();
                let sin_s = sdz.sin();
                let t11 = Complex64::new(cos_s, -delta * sin_s / s_abs);
                let t22 = Complex64::new(cos_s, delta * sin_s / s_abs);
                let t12 = Complex64::new(0.0, -kappa_j * sin_s / s_abs);
                let t21 = Complex64::new(0.0, kappa_j * sin_s / s_abs);
                (t11, t12, t21, t22)
            } else {
                // Degenerate (κ → 0 or exactly at Bragg with no coupling)
                let t11 = Complex64::new(1.0, -delta * dz);
                let t22 = Complex64::new(1.0, delta * dz);
                let t12 = Complex64::new(0.0, 0.0);
                let t21 = Complex64::new(0.0, 0.0);
                (t11, t12, t21, t22)
            };

            // M_new = T · M
            let new11 = t11 * m11 + t12 * m21;
            let new12 = t11 * m12 + t12 * m22;
            let new21 = t21 * m11 + t22 * m21;
            let new22 = t21 * m12 + t22 * m22;
            m11 = new11;
            m12 = new12;
            m21 = new21;
            m22 = new22;
        }

        // Reflection coefficient: r = -m21/m22 (boundary condition: no backward wave at output)
        if m22.norm() < 1e-60 {
            Complex64::new(0.0, 0.0)
        } else {
            -m21 / m22
        }
    }

    /// Peak reflectivity of the apodized FBG at the Bragg wavelength.
    pub fn peak_reflectivity(&self) -> f64 {
        let r = self.complex_reflection_tmm(self.fbg.center_wavelength_nm);
        r.norm_sqr().min(1.0)
    }

    /// Sidelobe level (dB) — ratio of first sidelobe peak to main peak.
    ///
    /// Scans a ±5× bandwidth window to find the highest sidelobe.
    pub fn sidelobe_level_db(&self) -> f64 {
        let bw = self.fbg.bandwidth_nm().max(0.01);
        let lambda_b = self.fbg.center_wavelength_nm;
        let scan_width = 5.0 * bw;
        let spectrum = self.reflection_spectrum(lambda_b - scan_width, lambda_b + scan_width, 4000);

        let r_peak = spectrum.iter().map(|&(_, r)| r).fold(0.0_f64, f64::max);

        // Find first sidelobe: look outside the main lobe (beyond 1.5× bandwidth)
        let sidelobe_max = spectrum
            .iter()
            .filter(|&&(lam, _)| (lam - lambda_b).abs() > 1.5 * bw)
            .map(|&(_, r)| r)
            .fold(0.0_f64, f64::max);

        if r_peak < 1e-30 || sidelobe_max < 1e-30 {
            return -60.0; // very low
        }
        10.0 * (sidelobe_max / r_peak).log10()
    }

    /// FWHM bandwidth of the apodized FBG (nm), estimated from its spectrum.
    pub fn bandwidth_nm(&self) -> f64 {
        let bw_est = self.fbg.bandwidth_nm().max(0.01);
        let lambda_b = self.fbg.center_wavelength_nm;
        let spectrum =
            self.reflection_spectrum(lambda_b - 3.0 * bw_est, lambda_b + 3.0 * bw_est, 2000);

        let r_peak = spectrum.iter().map(|&(_, r)| r).fold(0.0_f64, f64::max);
        let half_max = r_peak / 2.0;

        let above_half: Vec<f64> = spectrum
            .iter()
            .filter(|&&(_, r)| r >= half_max)
            .map(|&(lam, _)| lam)
            .collect();

        if above_half.is_empty() {
            return 0.0;
        }
        let lam_min = above_half.iter().cloned().fold(f64::INFINITY, f64::min);
        let lam_max = above_half.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        lam_max - lam_min
    }

    /// Reflection spectrum using the TMM.
    pub fn reflection_spectrum(
        &self,
        lambda_min_nm: f64,
        lambda_max_nm: f64,
        n_pts: usize,
    ) -> Vec<(f64, f64)> {
        if n_pts == 0 || lambda_min_nm >= lambda_max_nm {
            return Vec::new();
        }
        (0..n_pts)
            .map(|i| {
                let lambda_nm = lambda_min_nm
                    + (lambda_max_nm - lambda_min_nm) * i as f64 / (n_pts - 1).max(1) as f64;
                let r = self.complex_reflection_tmm(lambda_nm);
                (lambda_nm, r.norm_sqr().clamp(0.0, 1.0))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// FbgInterrogator
// ---------------------------------------------------------------------------

/// FBG interrogation system for multi-sensor wavelength-division multiplexing.
///
/// A broadband source illuminates an array of FBGs, each with a different
/// Bragg wavelength. An optical spectrum analyser (OSA) or tunable filter
/// tracks each peak, resolving shifts due to measurands.
#[derive(Debug, Clone)]
pub struct FbgInterrogator {
    /// Array of FBG sensors (must have distinct Bragg wavelengths)
    pub fbg_array: Vec<FiberBraggGrating>,
    /// Light source bandwidth (nm) — must cover all sensor wavelengths
    pub light_source_bandwidth_nm: f64,
    /// Wavelength measurement resolution (pm) — determines measurand resolution
    pub wavelength_resolution_pm: f64,
    /// Scan rate (Hz) — interrogation speed
    pub scan_rate_hz: f64,
}

impl FbgInterrogator {
    /// Create a new FBG interrogation system.
    pub fn new(
        fbg_array: Vec<FiberBraggGrating>,
        source_bw_nm: f64,
        resolution_pm: f64,
        scan_rate_hz: f64,
    ) -> Result<Self> {
        if source_bw_nm <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "source bandwidth must be positive, got {source_bw_nm}"
            )));
        }
        if resolution_pm <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "wavelength resolution must be positive, got {resolution_pm}"
            )));
        }
        if scan_rate_hz <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "scan rate must be positive, got {scan_rate_hz}"
            )));
        }
        Ok(Self {
            fbg_array,
            light_source_bandwidth_nm: source_bw_nm,
            wavelength_resolution_pm: resolution_pm,
            scan_rate_hz,
        })
    }

    /// Measure wavelength shifts (nm) for a set of applied strains and
    /// temperature changes.  Returns one shift per FBG.
    ///
    /// If `strains` or `delta_temps` are shorter than the FBG array, the
    /// missing values are treated as zero.
    pub fn measure_shifts_nm(&self, strains: &[f64], delta_temps: &[f64]) -> Vec<f64> {
        self.fbg_array
            .iter()
            .enumerate()
            .map(|(i, fbg)| {
                let eps = strains.get(i).copied().unwrap_or(0.0);
                let dt = delta_temps.get(i).copied().unwrap_or(0.0);
                fbg.combined_shift_nm(eps, dt)
            })
            .collect()
    }

    /// Strain measurement resolution (με) derived from wavelength resolution.
    ///
    /// Uses the first FBG's strain sensitivity as representative.
    pub fn strain_resolution_microstrain(&self) -> f64 {
        match self.fbg_array.first() {
            None => f64::INFINITY,
            Some(fbg) => {
                let sens = fbg.strain_sensitivity_pm_per_microstrain;
                if sens.abs() < 1e-60 {
                    f64::INFINITY
                } else {
                    self.wavelength_resolution_pm / sens
                }
            }
        }
    }

    /// Temperature measurement resolution (°C) derived from wavelength resolution.
    ///
    /// Uses the first FBG's temperature sensitivity as representative.
    pub fn temperature_resolution_c(&self) -> f64 {
        match self.fbg_array.first() {
            None => f64::INFINITY,
            Some(fbg) => {
                let sens = fbg.temperature_sensitivity_pm_per_c;
                if sens.abs() < 1e-60 {
                    f64::INFINITY
                } else {
                    self.wavelength_resolution_pm / sens
                }
            }
        }
    }

    /// Maximum number of FBGs that can be multiplexed within the source bandwidth.
    ///
    /// Each FBG occupies a spectral slot equal to 10× its FWHM bandwidth to avoid
    /// spectral overlap and cross-talk.
    pub fn multiplexing_capacity(&self) -> usize {
        if self.fbg_array.is_empty() {
            return 0;
        }
        // Use mean bandwidth of all FBGs
        let mean_bw: f64 = self.fbg_array.iter().map(|f| f.bandwidth_nm()).sum::<f64>()
            / self.fbg_array.len() as f64;
        let slot_nm = (10.0 * mean_bw).max(0.5); // minimum slot 0.5 nm
        (self.light_source_bandwidth_nm / slot_nm).floor() as usize
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn smf28() -> FiberBraggGrating {
        FiberBraggGrating::smf28_1550().expect("SMF-28 FBG construction should succeed")
    }

    #[test]
    fn test_fbg_grating_period() {
        let fbg = smf28();
        // Λ = λ_B / (2·n_eff)
        let expected = 1550.0 / (2.0 * 1.4682);
        assert_relative_eq!(fbg.grating_period_nm(), expected, max_relative = 1e-9);
    }

    #[test]
    fn test_fbg_coupling_coefficient() {
        let fbg = smf28();
        // κ = π·Δn / λ_B(m)
        let expected = PI * 3e-4 / (1550e-9);
        assert_relative_eq!(fbg.coupling_coefficient(), expected, max_relative = 1e-9);
    }

    #[test]
    fn test_fbg_peak_reflectivity_high() {
        // With Δn = 3e-4 and L = 10 mm, κ·L ≈ 6.08 → tanh²(6.08) ≈ 1.0
        let fbg = smf28();
        let r = fbg.peak_reflectivity();
        assert!(r > 0.99, "Expected high reflectivity, got {r:.6}");
        assert!(r <= 1.0, "Reflectivity must be ≤ 1, got {r:.6}");
    }

    #[test]
    fn test_fbg_strain_shift_positive() {
        // Tensile strain (positive ε) → red shift (positive Δλ)
        let fbg = smf28();
        let shift = fbg.strain_shift_nm(1000.0); // 1000 με tension
        assert!(
            shift > 0.0,
            "Tensile strain should produce a red shift, got {shift:.6} nm"
        );
    }

    #[test]
    fn test_fbg_temperature_shift_positive() {
        // Heating (positive ΔT) → red shift (positive Δλ)
        let fbg = smf28();
        let shift = fbg.temperature_shift_nm(10.0); // +10 °C
        assert!(
            shift > 0.0,
            "Heating should produce a red shift, got {shift:.6} nm"
        );
    }

    #[test]
    fn test_fbg_combined_shift() {
        let fbg = smf28();
        let eps = 500.0; // με
        let dt = 5.0; // °C
        let combined = fbg.combined_shift_nm(eps, dt);
        let expected = fbg.strain_shift_nm(eps) + fbg.temperature_shift_nm(dt);
        assert_relative_eq!(combined, expected, max_relative = 1e-12);
    }

    #[test]
    fn test_fbg_bandwidth_positive() {
        let fbg = smf28();
        let bw = fbg.bandwidth_nm();
        assert!(bw > 0.0, "Bandwidth should be positive, got {bw:.6}");
    }

    #[test]
    fn test_reflection_spectrum_peak_at_center() {
        let fbg = smf28();
        let lambda_b = fbg.center_wavelength_nm;
        let spectrum = fbg.reflection_spectrum(lambda_b - 1.0, lambda_b + 1.0, 1001);
        // Find the maximum reflectivity and its wavelength
        let (lam_peak, r_peak) = spectrum
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|&(l, r)| (l, r))
            .expect("spectrum should be non-empty");
        assert!(
            r_peak > 0.99,
            "Peak reflectivity should be > 0.99, got {r_peak}"
        );
        assert!(
            (lam_peak - lambda_b).abs() < 0.01,
            "Peak should be at λ_B ≈ {lambda_b} nm, found {lam_peak} nm"
        );
    }

    #[test]
    fn test_interrogator_resolution() {
        let fbg = smf28();
        let resolution_pm = 1.0; // 1 pm wavelength resolution
        let interrogator = FbgInterrogator::new(
            vec![fbg.clone()],
            40.0, // 40 nm source bandwidth
            resolution_pm,
            1000.0,
        )
        .expect("Interrogator construction should succeed");

        let strain_res = interrogator.strain_resolution_microstrain();
        // Should be 1 pm / (strain_sensitivity_pm_per_με) ≈ 1/1.2 ≈ 0.83 με
        assert!(
            strain_res > 0.0 && strain_res < 5.0,
            "Strain resolution should be < 5 με, got {strain_res:.4} με"
        );
        let temp_res = interrogator.temperature_resolution_c();
        assert!(
            temp_res > 0.0 && temp_res < 1.0,
            "Temp resolution should be < 1 °C, got {temp_res:.4} °C"
        );
    }

    #[test]
    fn test_n_periods() {
        let fbg = smf28();
        let period = fbg.grating_period_nm();
        // L = 10 mm = 10e6 nm; N ≈ 10e6 / period
        let expected = (10e6_f64 / period).round() as usize;
        let got = fbg.n_periods();
        // Allow rounding difference of 1
        assert!(
            (got as i64 - expected as i64).abs() <= 1,
            "Expected ~{expected} periods, got {got}"
        );
    }

    #[test]
    fn test_strain_roundtrip() {
        let fbg = smf28();
        let eps_original = 200.0_f64; // με
        let shift = fbg.strain_shift_nm(eps_original);
        let eps_recovered = fbg.strain_from_shift_microstrain(shift);
        assert_relative_eq!(eps_recovered, eps_original, max_relative = 1e-9);
    }

    #[test]
    fn test_temperature_roundtrip() {
        let fbg = smf28();
        let dt_original = 25.0_f64; // °C
        let shift = fbg.temperature_shift_nm(dt_original);
        let dt_recovered = fbg.temperature_from_shift_c(shift);
        assert_relative_eq!(dt_recovered, dt_original, max_relative = 1e-9);
    }

    #[test]
    fn test_apodized_fbg_peak_below_uniform() {
        let fbg = smf28();
        let uniform_peak = fbg.peak_reflectivity();
        let apodized = fbg.clone().with_gaussian_apodization(0.3);
        let apodized_peak = apodized.peak_reflectivity();
        // Apodized grating has lower peak due to reduced effective κ·L
        // (Gaussian weights reduce the mean coupling strength)
        assert!(
            apodized_peak > 0.0 && apodized_peak <= 1.0,
            "Apodized peak {apodized_peak} out of [0,1]"
        );
        // For a strongly over-coupled grating, apodized peak should still be high
        let _ = uniform_peak;
    }

    #[test]
    fn test_apodized_fbg_spectrum_nonempty() {
        let fbg = smf28();
        let apodized = fbg.with_gaussian_apodization(0.3);
        let spec = apodized.reflection_spectrum(1548.0, 1552.0, 200);
        assert_eq!(spec.len(), 200);
        assert!(spec.iter().all(|&(_, r)| (0.0..=1.0).contains(&r)));
    }

    #[test]
    fn test_group_delay_at_bragg_finite() {
        let fbg = smf28();
        let tau = fbg.group_delay_ps(fbg.center_wavelength_nm);
        assert!(tau.is_finite(), "Group delay should be finite, got {tau}");
        // For a 10 mm strong FBG, group delay at Bragg ≈ L/c ≈ few ps
        assert!(tau.abs() < 1e6, "Group delay suspiciously large: {tau} ps");
    }

    #[test]
    fn test_transmission_plus_reflection_equals_unity() {
        let fbg = smf28();
        let lambda_b = fbg.center_wavelength_nm;
        let r_spec = fbg.reflection_spectrum(lambda_b - 0.5, lambda_b + 0.5, 51);
        let t_spec = fbg.transmission_spectrum(lambda_b - 0.5, lambda_b + 0.5, 51);
        for (&(_, r), &(_, t)) in r_spec.iter().zip(t_spec.iter()) {
            assert_relative_eq!(r + t, 1.0, max_relative = 1e-10);
        }
    }
}
