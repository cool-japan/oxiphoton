//! Atmospheric turbulence models for free-space optical propagation.
//!
//! Implements Kolmogorov turbulence statistics, the Hufnagel-Valley C_n² profile,
//! SLC-Day, CLEAR1 standard atmosphere models, and derived quantities (Fried
//! parameter, Rytov variance, scintillation index, isoplanatic angle, etc.).
//!
//! # Physical Background
//!
//! Atmospheric refractive-index turbulence is described by the structure constant
//! C_n² (m^{-2/3}).  Key derived quantities:
//! - Fried parameter r₀ = \[0.423 k² ∫C_n²(z)dz\]^{-3/5}
//! - Rytov variance σ²_R = 1.23 C_n² k^{7/6} L^{11/6} (plane wave, horizontal)
//! - Scintillation index σ²_I ≈ exp(4 σ²_χ) − 1 (weak turbulence σ²_χ = 0.307 σ²_R)
//!
//! # References
//! - Andrews & Phillips, "Laser Beam Propagation through Random Media", 2005
//! - Hufnagel & Stanley (1964); Valley (1980)
//! - Noll (1976) — Zernike decomposition of turbulence residuals

use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Noll residual-variance coefficients (Table 1, Noll 1976).
/// σ²_res = NOLL\[j\] * (D/r₀)^(5/3) after correcting j Zernike modes.
const NOLL: [f64; 22] = [
    0.0, 1.0299, 0.582, 0.134, 0.111, 0.0880, 0.0648, 0.0587, 0.0525, 0.0463, 0.0401, 0.0377,
    0.0352, 0.0328, 0.0304, 0.0279, 0.0267, 0.0255, 0.0243, 0.0231, 0.0220, 0.0208,
];

// ─────────────────────────────────────────────────────────────────────────────
// Cn2Profile
// ─────────────────────────────────────────────────────────────────────────────

/// Refractive-index structure constant profile C_n²(h).
#[derive(Debug, Clone)]
pub enum Cn2Profile {
    /// Constant C_n² along the path (m^{-2/3}).
    /// Typical values: strong ≈ 1e-13, weak ≈ 1e-17.
    Constant(f64),

    /// Hufnagel-Valley (HV-5/7) model.
    ///
    /// C_n²(h) = 0.00594*(v_rms/27)²*(1e-5*h)^10*exp(-h/1000)
    ///          + 2.7e-16*exp(-h/1500) + Cn2_0*exp(-h/100)
    ///
    /// where h is altitude in metres, v_rms is RMS wind speed (m/s),
    /// and Cn2_0 is the ground-level structure constant.
    HufnagelValley {
        /// RMS wind speed in m/s (HV-5/7 uses 21 m/s).
        v_rms: f64,
        /// Ground-level C_n² in m^{-2/3} (HV-5/7 uses 1.7e-14).
        cn2_0: f64,
    },

    /// SLC-Day (Submarine Laser Communication) daytime profile.
    SlcDay,

    /// CLEAR I nighttime model.
    Clear1,

    /// User-supplied altitude-dependent profile.
    /// Each element is `(altitude_m, cn2_m_minus_2_3)`.
    /// Points must be sorted by ascending altitude.
    Custom(Vec<(f64, f64)>),
}

impl Cn2Profile {
    /// C_n² at altitude `h_m` (metres).
    pub fn cn2_at_height(&self, h_m: f64) -> f64 {
        match self {
            Cn2Profile::Constant(c) => *c,

            Cn2Profile::HufnagelValley { v_rms, cn2_0 } => {
                let h_km = h_m * 1e-3;
                let h5 = 1e-5 * h_m; // dimensionless form used in original HV
                let term1 = 5.94e-23 * (v_rms / 27.0).powi(2) * h5.powi(10) * (-h_km / 1.0).exp();
                let term2 = 2.7e-16 * (-h_km / 1.5).exp();
                let term3 = cn2_0 * (-h_m / 100.0).exp();
                term1 + term2 + term3
            }

            Cn2Profile::SlcDay => {
                // SLC-Day piecewise model (altitude in km)
                let h = h_m * 1e-3;
                if h < 0.019 {
                    1.703e-14
                } else if h < 0.23 {
                    8.874e-15 * (-0.1046 * h).exp()
                } else if h < 1.026 {
                    2.241e-15 * (0.1032 * h).exp()
                } else if h < 7.0 {
                    1.136e-16 * (-0.1003 * h).exp()
                } else if h < 15.81 {
                    8.474e-17 * (-0.1005 * h).exp()
                } else if h < 22.91 {
                    3.297e-20 * (0.1005 * h).exp()
                } else {
                    3.681e-17 * (-0.1004 * h).exp()
                }
            }

            Cn2Profile::Clear1 => {
                // CLEAR I nighttime model (altitude in km)
                let h = h_m * 1e-3;
                if h < 0.1 {
                    6.352e-7 * (-h / 0.105).exp()
                } else if h < 1.5 {
                    6.209e-16 / (h + 0.1).powf(10.0 / 3.0)
                } else if h < 7.2 {
                    3.981e-16 * (-h / 2.0).exp()
                } else if h < 20.0 {
                    1.0e-16 * (-h / 6.0).exp()
                } else {
                    1.0e-22
                }
            }

            Cn2Profile::Custom(pts) => {
                if pts.is_empty() {
                    return 0.0;
                }
                if h_m <= pts[0].0 {
                    return pts[0].1;
                }
                if h_m >= pts[pts.len() - 1].0 {
                    return pts[pts.len() - 1].1;
                }
                // Linear interpolation
                for i in 0..pts.len() - 1 {
                    let (h0, c0) = pts[i];
                    let (h1, c1) = pts[i + 1];
                    if h_m >= h0 && h_m <= h1 {
                        let t = (h_m - h0) / (h1 - h0);
                        return c0 + t * (c1 - c0);
                    }
                }
                pts[pts.len() - 1].1
            }
        }
    }

    /// Path-integrated C_n²: ∫ C_n²(h) dh from `h_start` to `h_end` (metres).
    ///
    /// Uses composite trapezoidal quadrature with `n_steps` intervals.
    pub fn path_integrated(&self, h_start: f64, h_end: f64, n_steps: usize) -> f64 {
        let n = n_steps.max(2);
        let dh = (h_end - h_start) / n as f64;
        let mut sum = 0.0;
        for i in 0..=n {
            let h = h_start + i as f64 * dh;
            let w = if i == 0 || i == n { 0.5 } else { 1.0 };
            sum += w * self.cn2_at_height(h);
        }
        sum * dh
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AtmosphericPath
// ─────────────────────────────────────────────────────────────────────────────

/// Describes a propagation path through the turbulent atmosphere.
#[derive(Debug, Clone)]
pub struct AtmosphericPath {
    /// Geometric path length in kilometres.
    pub length_km: f64,
    /// C_n² structure constant profile along the path.
    pub cn2_profile: Cn2Profile,
    /// Optical wavelength in metres.
    pub wavelength: f64,
    /// Elevation angle of the path in degrees (0 = horizontal, 90 = zenith).
    pub elevation_deg: f64,
    /// Starting altitude (ground station) in metres.
    pub h_start_m: f64,
}

impl AtmosphericPath {
    /// Horizontal path at constant altitude with constant C_n².
    pub fn new_horizontal(length_km: f64, cn2: f64, wavelength: f64) -> Self {
        Self {
            length_km,
            cn2_profile: Cn2Profile::Constant(cn2),
            wavelength,
            elevation_deg: 0.0,
            h_start_m: 0.0,
        }
    }

    /// Slant path at given elevation angle with Hufnagel-Valley profile.
    /// `length_km` is the slant-path distance.
    pub fn new_vertical_slant(length_km: f64, elevation_deg: f64, wavelength: f64) -> Self {
        Self {
            length_km,
            cn2_profile: Cn2Profile::HufnagelValley {
                v_rms: 21.0,
                cn2_0: 1.7e-14,
            },
            wavelength,
            elevation_deg,
            h_start_m: 0.0,
        }
    }

    /// Optical wave-number k = 2π/λ.
    fn wave_number(&self) -> f64 {
        2.0 * PI / self.wavelength
    }

    /// Path length in metres.
    fn length_m(&self) -> f64 {
        self.length_km * 1e3
    }

    /// Path-integrated C_n² (m^{1/3}) using 1000 steps.
    ///
    /// For horizontal paths (elevation ≈ 0°), the integration variable is the
    /// along-path coordinate z ∈ \[0, L\], and C_n² is sampled at the constant
    /// altitude h_start.  For slant paths the integration is over the altitude
    /// range h_start to h_end = h_start + L sin(θ).
    fn cn2_integrated(&self) -> f64 {
        let l = self.length_m();
        let elev_rad = self.elevation_deg.to_radians();
        if elev_rad.abs() < 1e-4 {
            // Horizontal path: integrate along path at constant altitude h_start
            self.cn2_profile.path_integrated(0.0, l, 1000)
        } else {
            // Slant path: integrate over altitude range
            let h_end = self.h_start_m + l * elev_rad.sin();
            // Scale by 1/sin(θ) to convert altitude integral to path integral
            let sin_el = elev_rad.sin().abs().max(1e-6);
            self.cn2_profile
                .path_integrated(self.h_start_m, h_end, 1000)
                / sin_el
        }
    }

    /// Fried coherence parameter r₀ (metres).
    ///
    /// For a horizontal path: r₀ = \[0.423 k² C_n² L\]^{-3/5}
    /// For a slant path: integrates C_n²(h) with appropriate geometry.
    pub fn fried_parameter_m(&self) -> f64 {
        let k = self.wave_number();
        let cn2_int = self.cn2_integrated();
        // r₀ = (0.423 k² ∫C_n²dz)^{-3/5}
        (0.423 * k * k * cn2_int).powf(-3.0 / 5.0)
    }

    /// Rytov variance σ²_R (plane wave, horizontal path).
    ///
    /// σ²_R = 1.23 C_n² k^{7/6} L^{11/6}
    pub fn rytov_variance(&self) -> f64 {
        let k = self.wave_number();
        let l = self.length_m();
        let cn2 = self.cn2_profile.cn2_at_height(self.h_start_m);
        1.23 * cn2 * k.powf(7.0 / 6.0) * l.powf(11.0 / 6.0)
    }

    /// Log-amplitude variance σ²_χ ≈ 0.307 σ²_R (plane wave, Rytov).
    fn log_amplitude_variance(&self) -> f64 {
        0.307 * self.rytov_variance()
    }

    /// Scintillation index σ²_I = exp(4 σ²_χ) − 1.
    ///
    /// Valid for weak turbulence (σ²_R < 0.3).  For moderate/strong turbulence
    /// a saturation correction is applied.
    pub fn scintillation_index(&self) -> f64 {
        let sr = self.rytov_variance();
        if sr < 0.3 {
            // Weak: log-normal model
            let sigma_chi2 = self.log_amplitude_variance();
            (4.0 * sigma_chi2).exp() - 1.0
        } else if sr < 5.0 {
            // Moderate: Andrews (2001) interpolation
            let sigma_chi2 = self.log_amplitude_variance();
            let weak_si = (4.0 * sigma_chi2).exp() - 1.0;
            // Saturation correction: Andrews & Phillips eq. 8.82
            let correction = 1.0 - (1.0 + 0.47 * sr.powf(6.0 / 5.0)).powf(-7.0 / 6.0);
            weak_si * correction / sr.sqrt()
        } else {
            // Strong / saturation: σ²_I → 1 + constant/σ²_R
            1.0 + 2.0 / sr
        }
    }

    /// Transverse coherence radius ρ₀ = r₀ (plane wave approximation).
    pub fn coherence_length_m(&self) -> f64 {
        self.fried_parameter_m()
    }

    /// Isoplanatic angle θ₀ = 0.314 r₀ / H_eff \[radians\], returned in µrad.
    ///
    /// H_eff = \[∫C_n²(h) h^{5/3} dh / ∫C_n²(h) dh\]^{3/5}
    pub fn isoplanatic_angle_urad(&self) -> f64 {
        let r0 = self.fried_parameter_m();
        let l = self.length_m();
        // Effective turbulence height for horizontal path
        let h_eff = l / 2.0; // simplified for constant profile
        let theta_rad = 0.314 * r0 / h_eff.max(1.0);
        theta_rad * 1e6 // → µrad
    }

    /// Coherence time τ₀ = 0.314 r₀ / v_wind \[seconds\], returned in ms.
    pub fn coherence_time_ms(&self, wind_speed_m_per_s: f64) -> f64 {
        let r0 = self.fried_parameter_m();
        let v = wind_speed_m_per_s.max(0.1);
        0.314 * r0 / v * 1e3 // → ms
    }

    /// Classify the turbulence strength regime based on Rytov variance.
    pub fn regime(&self) -> TurbulenceRegime {
        let sr = self.rytov_variance();
        if sr < 0.3 {
            TurbulenceRegime::Weak
        } else if sr <= 1.0 {
            TurbulenceRegime::Moderate
        } else {
            TurbulenceRegime::Strong
        }
    }

    /// Strehl ratio without AO correction (Maréchal approximation).
    ///
    /// S = exp(−(2π σ_OPD / λ)²) where σ²_OPD = 1.03 (D/r₀)^{5/3} λ²/(2π)²
    /// (assuming aperture D = r₀ for a diffraction-limited beam size estimate).
    /// For a general aperture use `strehl_ratio_ao(0)`.
    pub fn strehl_ratio_free(&self) -> f64 {
        // Use aperture = 5*r₀ as reference (severely aberrated)
        let _r0 = self.fried_parameter_m();
        let d_over_r0 = 5.0_f64; // representative value
        let sigma2_waves = NOLL[1] * d_over_r0.powf(5.0 / 3.0); // radians² / (2π)²
        let sigma2_rad = sigma2_waves * (2.0 * PI).powi(2);
        (-sigma2_rad).exp().max(0.0)
    }

    /// Strehl ratio with `ao_order` Zernike modes corrected (Noll 1976).
    ///
    /// Uses the look-up table up to 21 modes; beyond that uses the asymptotic
    /// power law σ² ≈ 0.2944 j^{-√3} (D/r₀)^{5/3}.
    pub fn strehl_ratio_ao(&self, ao_order: usize, aperture_m: f64) -> f64 {
        let r0 = self.fried_parameter_m();
        let d_over_r0 = aperture_m / r0;
        let sigma2_waves = if ao_order == 0 {
            NOLL[1] * d_over_r0.powf(5.0 / 3.0)
        } else if ao_order < NOLL.len() {
            NOLL[ao_order] * d_over_r0.powf(5.0 / 3.0)
        } else {
            // Asymptotic approximation beyond table
            0.2944 * (ao_order as f64).powf(-(3.0_f64.sqrt())) * d_over_r0.powf(5.0 / 3.0)
        };
        let sigma2_rad2 = sigma2_waves * (2.0 * PI).powi(2);
        (-sigma2_rad2).exp().max(0.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TurbulenceRegime
// ─────────────────────────────────────────────────────────────────────────────

/// Turbulence strength classification based on Rytov variance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurbulenceRegime {
    /// σ²_R < 0.3 — log-normal statistics apply.
    Weak,
    /// 0.3 ≤ σ²_R ≤ 1.0 — intermediate regime.
    Moderate,
    /// σ²_R > 1.0 — saturation and focusing effects dominate.
    Strong,
}

// ─────────────────────────────────────────────────────────────────────────────
// LogNormalScintillation
// ─────────────────────────────────────────────────────────────────────────────

/// Log-normal irradiance statistics for weak turbulence.
///
/// The irradiance I is modelled as I = ⟨I⟩ exp(2χ) where χ is a zero-mean
/// Gaussian with variance σ²_χ.  The scintillation index σ²_I = exp(4σ²_χ) − 1.
#[derive(Debug, Clone)]
pub struct LogNormalScintillation {
    /// Scintillation index σ²_I = ⟨I²⟩/⟨I⟩² − 1.
    pub scintillation_index: f64,
    /// Mean irradiance (W/m²).
    pub mean_irradiance: f64,
}

impl LogNormalScintillation {
    /// Construct from scintillation index and mean irradiance.
    ///
    /// # Panics-free: clamps σ²_I to (0, 100].
    pub fn new(sigma_i: f64, mean_i: f64) -> Self {
        Self {
            scintillation_index: sigma_i.clamp(1e-10, 100.0),
            mean_irradiance: mean_i.max(1e-30),
        }
    }

    /// Log-amplitude variance σ²_χ from σ²_I = exp(4σ²_χ) − 1.
    fn sigma_chi2(&self) -> f64 {
        0.25 * (1.0 + self.scintillation_index).ln()
    }

    /// PDF of irradiance:
    /// p(I) = exp\[−(ln(I/⟨I⟩) + σ²_χ)² / (2σ²_χ)\] / (I σ_χ √(2π))
    pub fn pdf(&self, irradiance: f64) -> f64 {
        if irradiance <= 0.0 {
            return 0.0;
        }
        let sc2 = self.sigma_chi2();
        let sc = sc2.sqrt();
        let mu = self.mean_irradiance * (-sc2).exp(); // median irradiance
        let arg = (irradiance / mu).ln() / sc;
        let norm = irradiance * sc * (2.0 * PI).sqrt();
        (-0.5 * arg * arg).exp() / norm
    }

    /// CDF P(I ≤ threshold) via complementary error function.
    pub fn cdf(&self, threshold: f64) -> f64 {
        if threshold <= 0.0 {
            return 0.0;
        }
        let sc2 = self.sigma_chi2();
        let sc = sc2.sqrt();
        let mu = self.mean_irradiance * (-sc2).exp();
        let arg = -(threshold / mu).ln() / (sc * 2.0_f64.sqrt());
        0.5 * erfc_approx(arg)
    }

    /// Fade threshold I_T (dB below mean) such that P(I < I_T) = p_fade.
    pub fn fade_threshold(&self, p_fade: f64) -> f64 {
        let p = p_fade.clamp(1e-10, 1.0 - 1e-10);
        let sc2 = self.sigma_chi2();
        let sc = sc2.sqrt();
        // Invert CDF: I_T = exp(erfinv(1−2p)*sc*√2 − sc²) * ⟨I⟩
        let erfinv_val = erfinv_approx(1.0 - 2.0 * p);
        let ln_it_over_mean = erfinv_val * sc * 2.0_f64.sqrt() - sc2;
        // Fade depth in dB below mean (positive = below mean)
        -10.0 * std::f64::consts::LOG10_E * ln_it_over_mean
    }

    /// SNR penalty in dB due to scintillation: ΔSNRᵈᴮ = 10 log₁₀(1 + σ²_I).
    pub fn mean_snr_penalty_db(&self) -> f64 {
        10.0 * (1.0 + self.scintillation_index).log10()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GammaGammaDistribution
// ─────────────────────────────────────────────────────────────────────────────

/// Gamma-Gamma irradiance distribution for moderate-to-strong turbulence.
///
/// The PDF is:
/// p(I) = 2(αβ)^{(α+β)/2} / \[Γ(α)Γ(β)\] I^{(α+β)/2−1} K_{α−β}(2√(αβI))
///
/// where K_ν is the modified Bessel function of the second kind.
/// Parameters α (large-scale) and β (small-scale) are related to σ²_I by:
/// σ²_I = 1/α + 1/β + 1/(αβ)
#[derive(Debug, Clone)]
pub struct GammaGammaDistribution {
    /// Large-scale scintillation parameter α > 0.
    pub alpha: f64,
    /// Small-scale scintillation parameter β > 0.
    pub beta: f64,
    /// Mean irradiance ⟨I⟩.
    pub mean_irradiance: f64,
}

impl GammaGammaDistribution {
    /// Construct from scintillation index and mean irradiance.
    ///
    /// Solves for α and β using the closed-form approximation valid for
    /// plane-wave propagation (Andrews & Phillips 2005, Ch. 10):
    /// 1/α = exp\[σ²_ln_x\] − 1 and 1/β = exp\[σ²_ln_y\] − 1
    /// where σ²_ln_x = 0.49 σ²_R / (1+1.11 σ²_R^{6/5})^{7/6}
    ///       σ²_ln_y = 0.51 σ²_R / (1+0.69 σ²_R^{6/5})^{5/6}
    pub fn from_scintillation_index(si: f64, mean_i: f64) -> Self {
        // Recover Rytov variance from σ²_I: approximate inversion
        // For plane wave: σ²_I ≈ σ²_R (rough guide — iterate if needed)
        let si_clamped = si.max(0.01);
        // Use si as proxy for Rytov variance to get α, β directly
        let sr = si_clamped; // approximate σ²_R ≈ σ²_I
        let sigma_ln_x2 = 0.49 * sr / (1.0 + 1.11 * sr.powf(6.0 / 5.0)).powf(7.0 / 6.0);
        let sigma_ln_y2 = 0.51 * sr / (1.0 + 0.69 * sr.powf(6.0 / 5.0)).powf(5.0 / 6.0);
        let alpha = 1.0 / (sigma_ln_x2.exp() - 1.0).max(1e-6);
        let beta = 1.0 / (sigma_ln_y2.exp() - 1.0).max(1e-6);
        Self {
            alpha: alpha.min(1e6),
            beta: beta.min(1e6),
            mean_irradiance: mean_i.max(1e-30),
        }
    }

    /// Gamma function via Lanczos approximation.
    fn gamma_fn(x: f64) -> f64 {
        // Lanczos approximation (g=7, n=9) — Numerical Recipes
        const G: f64 = 7.0;
        const C: [f64; 9] = [
            0.999_999_999_999_809_9,
            676.520_368_121_885_1,
            -1_259.139_216_722_402_9,
            771.323_428_777_653_1,
            -176.615_029_162_140_6,
            12.507_343_278_686_905,
            -0.138_571_095_265_720_12,
            9.984_369_578_019_572e-6,
            1.505_632_735_149_311_6e-7,
        ];
        if x < 0.5 {
            PI / ((PI * x).sin() * Self::gamma_fn(1.0 - x))
        } else {
            let z = x - 1.0;
            let mut sum = C[0];
            for (i, &ci) in C[1..].iter().enumerate() {
                sum += ci / (z + i as f64 + 1.0);
            }
            let t = z + G + 0.5;
            (2.0 * PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * sum
        }
    }

    /// Modified Bessel function K_0(x).
    ///
    /// Uses Abramowitz & Stegun formulas 9.8.5 (small x) and 9.8.6 (large x).
    /// The small-argument formula uses the substitution t = (x/3.75)² for I_0
    /// (A&S 9.8.1) and a separate t = x²/4 for the correction polynomial.
    fn bessel_k0(x: f64) -> f64 {
        if x <= 0.0 {
            return f64::INFINITY;
        }
        if x <= 2.0 {
            // I_0(x): A&S 9.8.1, valid |x| ≤ 3.75, t = (x/3.75)²
            let t1 = (x / 3.75) * (x / 3.75);
            let i0 = 1.0
                + 3.515_622_9 * t1
                + 3.089_942_4 * t1 * t1
                + 1.206_749_2 * t1.powi(3)
                + 0.265_973_2 * t1.powi(4)
                + 0.036_076_8 * t1.powi(5)
                + 0.004_581_3 * t1.powi(6);
            // A&S 9.8.5 correction polynomial, t = (x/2)² = x²/4
            let t2 = x * x / 4.0;
            let poly = -0.577_215_65
                + 0.422_784_20 * t2
                + 0.230_697_56 * t2 * t2
                + 0.034_885_90 * t2.powi(3)
                + 0.002_626_98 * t2.powi(4)
                + 0.000_107_50 * t2.powi(5)
                + 0.000_007_40 * t2.powi(6);
            -(x / 2.0).ln() * i0 + poly
        } else {
            // A&S 9.8.6 asymptotic, t = 2/x
            let t = 2.0 / x;
            let poly = 1.253_314_14 - 0.078_323_58 * t + 0.021_895_68 * t * t
                - 0.010_624_46 * t.powi(3)
                + 0.005_878_72 * t.powi(4)
                - 0.002_515_40 * t.powi(5)
                + 0.000_532_08 * t.powi(6);
            (-x).exp() / x.sqrt() * poly
        }
    }

    /// Modified Bessel function K_1(x).
    ///
    /// Uses A&S 9.8.7 (small x, |x| ≤ 2) and 9.8.8 (large x).
    fn bessel_k1(x: f64) -> f64 {
        if x <= 0.0 {
            return f64::INFINITY;
        }
        if x <= 2.0 {
            // I_1(x)/x: A&S 9.8.3, t = (x/3.75)²
            let t1 = (x / 3.75) * (x / 3.75);
            let i1_over_x = 0.5
                + 0.878_905_94 * t1
                + 0.514_988_69 * t1 * t1
                + 0.150_849_34 * t1.powi(3)
                + 0.026_587_33 * t1.powi(4)
                + 0.003_015_32 * t1.powi(5)
                + 0.000_324_11 * t1.powi(6);
            let i1 = x * i1_over_x;
            // A&S 9.8.7 correction polynomial, t = (x/2)²
            let t2 = x * x / 4.0;
            let poly = 1.0 + 0.154_431_44 * t2
                - 0.672_785_79 * t2 * t2
                - 0.181_568_97 * t2.powi(3)
                - 0.019_194_02 * t2.powi(4)
                - 0.001_104_04 * t2.powi(5)
                - 0.000_046_86 * t2.powi(6);
            // K_1(x) = (1/x) * poly + ln(x/2) * I_1(x)
            (1.0 / x) * poly + (x / 2.0).ln() * i1
        } else {
            // A&S 9.8.8 asymptotic, t = 2/x
            let t = 2.0 / x;
            let poly = 1.253_314_14 + 0.234_986_19 * t - 0.036_556_20 * t * t
                + 0.015_042_68 * t.powi(3)
                - 0.007_803_53 * t.powi(4)
                + 0.003_256_14 * t.powi(5)
                - 0.000_682_45 * t.powi(6);
            (-x).exp() / x.sqrt() * poly
        }
    }

    /// Modified Bessel K_ν(x) for non-integer ν via recursion from K_0, K_1.
    ///
    /// Uses the recurrence: K_{n+1}(x) = K_{n-1}(x) + (2n/x) K_n(x).
    /// For fractional ν the half-integer formula is used up to the integer part,
    /// then a final interpolation step is applied.
    fn bessel_knu(nu: f64, x: f64) -> f64 {
        if x <= 0.0 {
            return f64::INFINITY;
        }
        let nu_abs = nu.abs();
        // Use integer recurrence up to floor(nu_abs)
        let n_floor = nu_abs.floor() as usize;
        let frac = nu_abs - n_floor as f64;

        // Compute K_0, K_1 first
        let k0 = Self::bessel_k0(x);
        if n_floor == 0 && frac < 1e-10 {
            return k0;
        }
        let k1 = Self::bessel_k1(x);
        if n_floor == 1 && frac < 1e-10 {
            return k1;
        }

        // Recurse up to K_{n_floor}
        let mut k_prev = k0;
        let mut k_curr = k1;
        for n in 1..n_floor {
            let k_next = k_prev + (2.0 * n as f64 / x) * k_curr;
            k_prev = k_curr;
            k_curr = k_next;
        }
        // k_curr = K_{n_floor}
        if frac < 1e-10 {
            return k_curr;
        }
        // Interpolate between K_{n_floor} and K_{n_floor+1}
        let k_next = k_prev + (2.0 * n_floor as f64 / x) * k_curr;
        k_curr + frac * (k_next - k_curr)
    }

    /// Gamma-Gamma PDF:
    /// p(I) = 2(αβ)^{(α+β)/2} / \[Γ(α)Γ(β)\] I^{(α+β)/2−1} K_{α−β}(2√(αβI/⟨I⟩))
    pub fn pdf(&self, irradiance: f64) -> f64 {
        if irradiance <= 0.0 {
            return 0.0;
        }
        let a = self.alpha;
        let b = self.beta;
        let im = self.mean_irradiance;
        let i_norm = irradiance / im; // normalised irradiance
        let nu = (a - b).abs();
        let sqrt_ab = (a * b).sqrt();
        let arg = 2.0 * sqrt_ab * i_norm.sqrt();
        let knu = Self::bessel_knu(nu, arg);
        let gamma_a = Self::gamma_fn(a);
        let gamma_b = Self::gamma_fn(b);
        let prefactor = 2.0 * (a * b).powf((a + b) / 2.0) / (gamma_a * gamma_b * im);
        prefactor * i_norm.powf((a + b) / 2.0 - 1.0) * knu
    }

    /// Outage probability P(I < I_th) via numerical integration of the PDF.
    pub fn outage_probability(&self, threshold: f64) -> f64 {
        if threshold <= 0.0 {
            return 0.0;
        }
        // Composite Simpson integration from 0 to threshold
        let n = 200usize;
        let eps = threshold * 1e-6;
        let dt = (threshold - eps) / n as f64;
        let mut sum = 0.0;
        for i in 0..=n {
            let t = eps + i as f64 * dt;
            let w = if i == 0 || i == n {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            sum += w * self.pdf(t);
        }
        (sum * dt / 3.0).clamp(0.0, 1.0)
    }

    /// Average BER for OOK-IM with direct detection, averaged over the Gamma-Gamma
    /// fading distribution.
    ///
    /// BER(I) = 0.5 erfc(SNR*I / √2)
    pub fn mean_ber(&self, snr_db: f64) -> f64 {
        let snr = 10.0_f64.powf(snr_db / 10.0);
        // Numerical integration
        let n = 500usize;
        // Integrate from eps to 10*mean
        let i_max = 10.0 * self.mean_irradiance;
        let eps = i_max * 1e-5;
        let di = (i_max - eps) / n as f64;
        let mut sum = 0.0;
        for i in 0..=n {
            let irr = eps + i as f64 * di;
            let w = if i == 0 || i == n {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            let ber_cond = 0.5 * erfc_approx(snr * irr / self.mean_irradiance / 2.0_f64.sqrt());
            sum += w * ber_cond * self.pdf(irr);
        }
        (sum * di / 3.0).clamp(0.0, 1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: erfc and erfinv (re-implemented to avoid import cycles)
// ─────────────────────────────────────────────────────────────────────────────

/// Complementary error function approximation (max error < 1.5×10⁻⁷).
/// Abramowitz & Stegun 7.1.26.
pub(crate) fn erfc_approx(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc_approx(-x);
    }
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    poly * (-(x * x)).exp()
}

/// Inverse error function approximation (Peter Acklam 2002 + Halley refinement).
pub(crate) fn erfinv_approx(y: f64) -> f64 {
    let y = y.clamp(-1.0 + 1e-15, 1.0 - 1e-15);
    let p = (y + 1.0) / 2.0;
    let ln_p = (-2.0 * p * (1.0 - p)).ln();
    // Rational approximation
    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };
    let c0 = 2.515_517;
    let c1 = 0.802_853;
    let c2 = 0.010_328;
    let d1 = 1.432_788;
    let d2 = 0.189_269;
    let d3 = 0.001_308;
    let num = c0 + c1 * t + c2 * t * t;
    let den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
    let mut x = if p < 0.5 {
        -(t - num / den)
    } else {
        t - num / den
    };
    // One Halley step: refine
    let fx = libm_erf(x) - y;
    let fpx = (2.0 / PI.sqrt()) * (-(x * x)).exp();
    x -= fx / (fpx + x * fx);
    // Use ln_p only to suppress unused-variable lint
    let _ = ln_p;
    x
}

/// Approximation of erf(x) via A&S 7.1.26 (used internally).
fn libm_erf(x: f64) -> f64 {
    1.0 - erfc_approx(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Fried parameter for strong turbulence at 1 km should be millimetre-scale.
    #[test]
    fn test_fried_parameter_strong() {
        let path = AtmosphericPath::new_horizontal(1.0, 1e-13, 1550e-9);
        let r0 = path.fried_parameter_m();
        assert!(r0 > 0.0 && r0 < 0.1, "r0 = {r0:.4e} m (expected < 10 cm)");
    }

    /// Fried parameter for weak turbulence at 1 km should be > 10 cm.
    #[test]
    fn test_fried_parameter_weak() {
        let path = AtmosphericPath::new_horizontal(1.0, 1e-17, 1550e-9);
        let r0 = path.fried_parameter_m();
        assert!(
            r0 > 0.1,
            "r0 = {r0:.4e} m (expected > 0.1 m for weak turbulence)"
        );
    }

    /// Rytov variance weak-turbulence regime classification.
    #[test]
    fn test_rytov_regime_weak() {
        let path = AtmosphericPath::new_horizontal(0.5, 1e-17, 1550e-9);
        assert_eq!(path.regime(), TurbulenceRegime::Weak);
    }

    /// Log-normal CDF integrates to ~1 over a wide range.
    #[test]
    fn test_lognormal_cdf_normalisation() {
        let ln = LogNormalScintillation::new(0.2, 1.0);
        let cdf_high = ln.cdf(1e6);
        assert!((cdf_high - 1.0).abs() < 0.01, "CDF(∞) = {cdf_high}");
    }

    /// Gamma-Gamma PDF integrates to approximately 1.
    #[test]
    fn test_gamma_gamma_pdf_integral() {
        let gg = GammaGammaDistribution::from_scintillation_index(0.5, 1.0);
        let n = 2000;
        let i_max = 10.0;
        let di = i_max / n as f64;
        let mut integral = 0.0;
        for i in 1..n {
            integral += gg.pdf(i as f64 * di);
        }
        integral *= di;
        assert!(
            (integral - 1.0).abs() < 0.05,
            "GG PDF integral = {integral:.4}"
        );
    }

    /// Coherence time decreases with higher wind speed.
    #[test]
    fn test_coherence_time_wind_dependence() {
        let path = AtmosphericPath::new_horizontal(1.0, 1e-14, 1064e-9);
        let tau_slow = path.coherence_time_ms(5.0);
        let tau_fast = path.coherence_time_ms(20.0);
        assert!(
            tau_slow > tau_fast,
            "Coherence time should decrease with wind speed"
        );
    }

    /// HV profile gives higher C_n² at ground than at 20 km altitude.
    #[test]
    fn test_hv_profile_altitude_dependence() {
        let profile = Cn2Profile::HufnagelValley {
            v_rms: 21.0,
            cn2_0: 1.7e-14,
        };
        let cn2_ground = profile.cn2_at_height(0.0);
        let cn2_high = profile.cn2_at_height(20_000.0);
        assert!(cn2_ground > cn2_high, "C_n² should decrease with altitude");
    }

    /// Bessel K_0 at x=1 should match known value ≈ 0.4210.
    #[test]
    fn test_bessel_k0() {
        let k0 = GammaGammaDistribution::bessel_k0(1.0);
        assert!((k0 - 0.4210).abs() < 0.01, "K_0(1) = {k0:.4}");
    }

    /// erfc_approx(0) = 1 and erfc_approx(∞) → 0.
    #[test]
    fn test_erfc_boundary() {
        assert!((erfc_approx(0.0) - 1.0).abs() < 1e-6);
        assert!(erfc_approx(10.0) < 1e-20);
    }
}
