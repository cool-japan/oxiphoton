/// THz material database, Drude model at THz frequencies, and atmospheric
/// THz absorption from water-vapour resonances.
use num_complex::Complex64;

#[cfg(test)]
use crate::error::OxiPhotonError;

// ─── Physical constants ────────────────────────────────────────────────────
const EPS0: f64 = 8.854_187_817e-12; // F/m
const C0: f64 = 2.997_924_58e8; // m/s
const MU0: f64 = 1.256_637_061e-6; // H/m

// ─── THz Material Database ────────────────────────────────────────────────

/// Static THz material entry: real refractive index and power-absorption
/// coefficient α (cm⁻¹) evaluated near 1 THz.
#[derive(Debug, Clone)]
pub struct ThzMaterial {
    /// Material name.
    pub name: String,
    /// Real part of refractive index at ~1 THz.
    pub n_real: f64,
    /// Power absorption coefficient at ~1 THz (cm⁻¹).
    pub alpha_cm: f64,
}

impl ThzMaterial {
    /// Construct from raw parameters.
    pub fn new(name: impl Into<String>, n_real: f64, alpha_cm: f64) -> Self {
        Self {
            name: name.into(),
            n_real,
            alpha_cm,
        }
    }

    /// High-resistivity silicon — nearly dispersion-free THz window material.
    ///
    /// n ≈ 3.42, α < 0.05 cm⁻¹ at 1 THz.
    pub fn silicon() -> Self {
        Self::new("Silicon (HR)", 3.42, 0.02)
    }

    /// Semi-insulating GaAs — common THz substrate/window.
    ///
    /// n ≈ 3.56, α ≈ 0.5 cm⁻¹ at 1 THz.
    pub fn gaas() -> Self {
        Self::new("GaAs (SI)", 3.56, 0.5)
    }

    /// Liquid water at room temperature — strong THz absorber.
    ///
    /// n ≈ 2.1, α ≈ 200 cm⁻¹ at 1 THz.
    pub fn water() -> Self {
        Self::new("Water (liquid)", 2.1, 200.0)
    }

    /// PTFE (Teflon) — low-loss THz optical element material.
    ///
    /// n ≈ 1.43, α ≈ 0.2 cm⁻¹ at 1 THz.
    pub fn ptfe() -> Self {
        Self::new("PTFE (Teflon)", 1.43, 0.2)
    }

    /// High-density polyethylene (HDPE) — used for THz lenses and windows.
    ///
    /// n ≈ 1.54, α ≈ 0.2 cm⁻¹ at 1 THz.
    pub fn polyethylene() -> Self {
        Self::new("HDPE", 1.54, 0.2)
    }

    /// Fused silica (amorphous SiO₂) — moderate THz transmission.
    ///
    /// n ≈ 1.95, α ≈ 0.1 cm⁻¹ at 1 THz.
    pub fn quartz_fused() -> Self {
        Self::new("Fused Silica", 1.95, 0.1)
    }

    /// ZnTe — widely used nonlinear crystal for THz OR and electro-optic sampling.
    ///
    /// n ≈ 2.85, α ≈ 1.2 cm⁻¹ at 1 THz.
    pub fn znte() -> Self {
        Self::new("ZnTe", 2.85, 1.2)
    }

    /// Complex refractive index N = n + i·k where k = α·c/(2·ω).
    pub fn refractive_index(&self) -> Complex64 {
        let freq_thz = 1.0;
        let omega = 2.0 * std::f64::consts::PI * freq_thz * 1e12;
        let k = self.alpha_cm * 1e2 * C0 / (2.0 * omega); // cm⁻¹ → m⁻¹, then k
        Complex64::new(self.n_real, k)
    }

    /// Return absorption coefficient (cm⁻¹).
    pub fn absorption_coefficient_cm(&self) -> f64 {
        self.alpha_cm
    }

    /// Single-pass intensity transmission exp(−α · L) through thickness L (mm).
    pub fn transmission(length_mm: f64, alpha_cm: f64) -> f64 {
        let l_cm = length_mm / 10.0;
        (-alpha_cm * l_cm).exp()
    }
}

// ─── Drude Model ──────────────────────────────────────────────────────────

/// Drude model permittivity and optical constants for metals and doped
/// semiconductors at THz frequencies.
///
/// ε(ω) = ε_∞ − ω_p² / (ω² + i·γ·ω)
///
/// where ω_p = plasma frequency and γ = scattering rate.
#[derive(Debug, Clone)]
pub struct DrudeTHz {
    /// Plasma frequency ω_p (THz).
    pub plasma_freq_thz: f64,
    /// Scattering rate (Drude damping) γ = 1/τ (THz).
    pub scattering_rate_thz: f64,
    /// High-frequency (background) dielectric constant ε_∞.
    pub eps_inf: f64,
}

impl DrudeTHz {
    /// General constructor.
    pub fn new(plasma_freq_thz: f64, scattering_rate_thz: f64, eps_inf: f64) -> Self {
        Self {
            plasma_freq_thz,
            scattering_rate_thz,
            eps_inf,
        }
    }

    /// Gold at THz frequencies.
    ///
    /// ω_p ≈ 4300 THz, γ ≈ 6.5 THz (from Drude fit to optical data).
    pub fn gold_thz() -> Self {
        Self::new(4300.0, 6.5, 1.0)
    }

    /// Copper at THz frequencies.
    ///
    /// ω_p ≈ 3700 THz, γ ≈ 8.3 THz.
    pub fn copper_thz() -> Self {
        Self::new(3700.0, 8.3, 1.0)
    }

    /// Drude model for a doped semiconductor with free-carrier density n_c (m⁻³).
    ///
    /// Uses the effective mass of GaAs (m* = 0.067 m_e) and carrier mobility
    /// μ = 8000 cm² V⁻¹ s⁻¹ as representative values.
    ///
    /// # Arguments
    /// * `n_carriers` — free carrier density (m⁻³).
    pub fn doped_silicon(n_carriers: f64) -> Self {
        // m* = 0.26 m_e for silicon
        let m_e = 9.109_383_7e-31; // kg
        let m_star = 0.26 * m_e;
        let e = 1.602_176_634e-19; // C
        let eps_r_si = 11.7; // background permittivity

        // ω_p² = n e² / (ε₀ ε_r m*)
        let omega_p_sq = n_carriers * e * e / (EPS0 * eps_r_si * m_star);
        let omega_p_thz = omega_p_sq.sqrt() / (2.0 * std::f64::consts::PI * 1e12);

        // γ = e / (m* μ)  with μ = 1400 cm²/Vs for Si
        let mu_si = 1400.0 * 1e-4; // m²/Vs
        let gamma_rad_per_s = e / (m_star * mu_si);
        let gamma_thz = gamma_rad_per_s / (2.0 * std::f64::consts::PI * 1e12);

        Self::new(omega_p_thz, gamma_thz, eps_r_si)
    }

    /// Complex permittivity at frequency `freq_thz` (THz).
    ///
    /// ε(ω) = ε_∞ − ω_p² / (ω(ω + iγ))
    pub fn permittivity(&self, freq_thz: f64) -> Complex64 {
        let omega = 2.0 * std::f64::consts::PI * freq_thz * 1e12;
        let omega_p = 2.0 * std::f64::consts::PI * self.plasma_freq_thz * 1e12;
        let gamma = 2.0 * std::f64::consts::PI * self.scattering_rate_thz * 1e12;

        // ε = ε_∞ − ω_p² / (ω² + iγω)
        let denom = Complex64::new(omega * omega - 0.0, gamma * omega); // ω(ω+iγ)
                                                                        // More precisely: denom = ω² + iγω
        let drude_term = Complex64::new(omega_p * omega_p, 0.0) / denom;
        Complex64::new(self.eps_inf, 0.0) - drude_term
    }

    /// Complex refractive index N = sqrt(ε).
    pub fn refractive_index(&self, freq_thz: f64) -> Complex64 {
        let eps = self.permittivity(freq_thz);
        // Principal square root of complex number
        complex_sqrt(eps)
    }

    /// Skin depth δ = 1 / (ω · Im(N) / c)  (μm).
    pub fn skin_depth_um(&self, freq_thz: f64) -> f64 {
        let n = self.refractive_index(freq_thz);
        let omega = 2.0 * std::f64::consts::PI * freq_thz * 1e12;
        let kappa = n.im.abs(); // extinction coefficient
        if kappa < 1e-30 {
            return f64::INFINITY;
        }
        let delta_m = C0 / (omega * kappa); // m
        delta_m * 1e6 // m → μm
    }

    /// Surface resistance Rs = sqrt(π f μ₀ / σ)  (Ω/sq).
    ///
    /// σ is recovered from Im(ε): σ = ε₀ · ω · Im(ε).
    pub fn surface_resistance_ohm(&self, freq_thz: f64) -> f64 {
        let eps = self.permittivity(freq_thz);
        let omega = 2.0 * std::f64::consts::PI * freq_thz * 1e12;
        let freq_hz = freq_thz * 1e12;

        // σ from Drude: ε_im = σ/(ε₀ ω)  →  σ = ε₀ ω ε_im
        let sigma = EPS0 * omega * eps.im.abs();
        if sigma < 1e-20 {
            return f64::INFINITY;
        }
        (std::f64::consts::PI * freq_hz * MU0 / sigma).sqrt()
    }
}

// ─── Atmospheric Absorption ───────────────────────────────────────────────

/// Atmospheric THz absorption model driven by water-vapour resonance lines.
///
/// The absorption coefficient is computed by summing Lorentzian-shaped
/// contributions from the principal H₂O rotational lines in the THz band.
/// This is a simplified van Vleck–Weisskopf model — sufficient for
/// engineering estimates.
#[derive(Debug, Clone)]
pub struct AtmosphericAbsorption {
    /// Relative humidity (%).
    pub humidity_percent: f64,
    /// Temperature (°C).
    pub temperature_c: f64,
    /// Pressure (atm).
    pub pressure_atm: f64,
}

/// Water vapour H₂O rotational absorption lines: (centre frequency THz,
/// line-strength factor relative to 0.557 THz line).
///
/// Lines taken from HITRAN database (JPL catalogue subset used in THz community).
const H2O_LINES: &[(f64, f64)] = &[
    (0.557, 1.000), // 5₁₅ ← 4₂₂  (strongest THz line)
    (0.752, 0.080),
    (0.988, 0.040),
    (1.097, 0.180),
    (1.113, 0.040),
    (1.163, 0.070),
    (1.207, 0.130),
    (1.229, 0.025),
    (1.411, 0.020),
    (1.602, 0.010),
    (1.661, 0.015),
    (1.717, 0.025),
    (1.763, 0.010),
    (1.797, 0.008),
    (1.919, 0.012),
    (2.074, 0.018),
    (2.164, 0.008),
    (2.221, 0.015),
    (2.264, 0.010),
    (2.344, 0.025),
    (2.640, 0.035),
    (2.774, 0.020),
    (2.968, 0.018),
];

/// Peak absorption coefficient of the 0.557 THz line at standard conditions
/// (50 % RH, 20 °C, 1 atm) — calibrated to experimental value ≈ 0.35 cm⁻¹.
const ALPHA_PEAK_STD: f64 = 0.35; // cm⁻¹

/// Lorentzian linewidth of water vapour lines at 1 atm (THz).
const LINEWIDTH_THZ: f64 = 0.005; // THz ≈ 5 GHz at 1 atm

impl AtmosphericAbsorption {
    /// Construct with specified humidity, temperature and pressure.
    pub fn new(humidity_percent: f64, temperature_c: f64, pressure_atm: f64) -> Self {
        Self {
            humidity_percent,
            temperature_c,
            pressure_atm,
        }
    }

    /// Standard laboratory conditions: 50 % RH, 20 °C, 1 atm.
    pub fn standard_conditions() -> Self {
        Self::new(50.0, 20.0, 1.0)
    }

    /// Saturated water-vapour pressure (hPa) — Magnus formula.
    fn saturation_pressure_hpa(&self) -> f64 {
        let t = self.temperature_c;
        6.112 * ((17.67 * t) / (t + 243.5)).exp()
    }

    /// Partial pressure of water vapour (hPa).
    fn water_vapour_pressure_hpa(&self) -> f64 {
        self.saturation_pressure_hpa() * self.humidity_percent / 100.0
    }

    /// Relative water-vapour density compared to standard conditions.
    fn relative_humidity_factor(&self) -> f64 {
        // At standard conditions: 50% RH, 20°C → p_w_std = 0.5 × 23.37 ≈ 11.7 hPa
        let p_w = self.water_vapour_pressure_hpa();
        let p_std = 6.112_f64 * ((17.67_f64 * 20.0_f64) / (20.0_f64 + 243.5_f64)).exp() * 0.5;
        let pressure_factor = self.pressure_atm; // collision broadening
        (p_w / p_std) * pressure_factor
    }

    /// Lorentzian linewidth at the current pressure (THz).
    fn linewidth_at_pressure(&self) -> f64 {
        LINEWIDTH_THZ * self.pressure_atm
    }

    /// Power absorption coefficient α(f) at frequency `freq_thz` (cm⁻¹).
    ///
    /// Computed as a sum of Lorentzian profiles over the tabulated H₂O lines.
    pub fn absorption_coefficient_cm(&self, freq_thz: f64) -> f64 {
        let rh_factor = self.relative_humidity_factor();
        let gamma = self.linewidth_at_pressure();

        let mut alpha = 0.0_f64;
        for &(f0, strength) in H2O_LINES {
            // Lorentzian: L(f) = (γ/π) / ((f-f0)² + γ²)
            let df = freq_thz - f0;
            let lorentz = (gamma / std::f64::consts::PI) / (df * df + gamma * gamma);
            // Peak coefficient = ALPHA_PEAK_STD × strength × rh_factor × (γ_std/γ)
            // so that the total area is invariant under pressure broadening.
            let peak = ALPHA_PEAK_STD * strength * rh_factor;
            // Area-normalised: multiply by π γ (area of unit Lorentzian peak)
            alpha += peak * std::f64::consts::PI * gamma * lorentz;
        }
        alpha
    }

    /// Single-pass intensity transmission exp(−α·L) over path length L (m).
    pub fn transmission(&self, freq_thz: f64, path_length_m: f64) -> f64 {
        let alpha = self.absorption_coefficient_cm(freq_thz);
        let l_cm = path_length_m * 100.0;
        (-alpha * l_cm).exp()
    }

    /// Transmission spectrum — returns `(freq_thz, transmission)` pairs.
    ///
    /// # Arguments
    /// * `f_min`  — start frequency (THz).
    /// * `f_max`  — end frequency (THz).
    /// * `n_pts`  — number of frequency samples.
    /// * `path_m` — propagation path (m), pass 1.0 for normalised α.
    ///
    /// Uses a 1 m path if you want α in cm⁻¹; override externally.
    pub fn transmission_spectrum(&self, f_min: f64, f_max: f64, n_pts: usize) -> Vec<(f64, f64)> {
        if n_pts == 0 {
            return Vec::new();
        }
        let path_m = 1.0; // 1 m standard path
        let df = if n_pts > 1 {
            (f_max - f_min) / (n_pts - 1) as f64
        } else {
            0.0
        };
        (0..n_pts)
            .map(|i| {
                let f = f_min + i as f64 * df;
                let t = self.transmission(f, path_m);
                (f, t)
            })
            .collect()
    }

    /// Identify THz transmission windows — frequency intervals (THz) where the
    /// 1 m path transmission exceeds 50 %.
    ///
    /// The three principal windows in dry/semi-dry air are roughly:
    /// - 0.1 – 0.5 THz
    /// - 0.7 – 1.1 THz
    /// - 1.2 – 1.4 THz
    pub fn transmission_windows(&self) -> Vec<(f64, f64)> {
        let threshold = 0.5; // 50 % transmission criterion
        let spectrum = self.transmission_spectrum(0.05, 3.0, 600);

        let mut windows: Vec<(f64, f64)> = Vec::new();
        let mut in_window = false;
        let mut win_start = 0.0_f64;

        for &(f, t) in &spectrum {
            if t >= threshold && !in_window {
                in_window = true;
                win_start = f;
            } else if t < threshold && in_window {
                in_window = false;
                windows.push((win_start, f));
            }
        }
        // Close an open window at the end of the scan
        if in_window {
            if let Some(&(f_last, _)) = spectrum.last() {
                windows.push((win_start, f_last));
            }
        }
        windows
    }

    /// Maximum propagation range (m) for a given maximum path loss (dB).
    ///
    /// L_max = loss_db / (α_cm · 4.343 · 100)
    ///
    /// (4.343 ≈ 1/ln(10)·10 converts Neper to dB; ×100 converts cm → m)
    pub fn max_range_m(&self, freq_thz: f64, max_loss_db: f64) -> f64 {
        let alpha_cm = self.absorption_coefficient_cm(freq_thz);
        if alpha_cm < 1e-30 {
            return f64::INFINITY;
        }
        // α_cm [cm⁻¹] × L_cm = − ln(T)  → L_cm = loss_db / (α_cm × 4.343)
        let l_cm = max_loss_db / (alpha_cm * 4.342_944_8);
        l_cm / 100.0 // cm → m
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────

/// Principal square root of a complex number.
///
/// Returns the root with non-negative real part.
fn complex_sqrt(z: Complex64) -> Complex64 {
    let r = z.norm().sqrt();
    let theta = z.arg() / 2.0;
    Complex64::new(r * theta.cos(), r * theta.sin())
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silicon_low_absorption() {
        let si = ThzMaterial::silicon();
        assert!(
            si.alpha_cm < 0.1,
            "Si absorption should be <0.1 cm⁻¹, got {}",
            si.alpha_cm
        );
    }

    #[test]
    fn test_water_high_absorption() {
        let h2o = ThzMaterial::water();
        assert!(
            h2o.alpha_cm > 100.0,
            "Water absorption should be >100 cm⁻¹, got {}",
            h2o.alpha_cm
        );
    }

    #[test]
    fn test_drude_gold_negative_real_eps() {
        let gold = DrudeTHz::gold_thz();
        let eps = gold.permittivity(1.0); // at 1 THz
        assert!(
            eps.re < 0.0,
            "Gold ε' should be negative at THz, got Re(ε)={}",
            eps.re
        );
    }

    #[test]
    fn test_skin_depth_positive() {
        let gold = DrudeTHz::gold_thz();
        let delta = gold.skin_depth_um(1.0);
        assert!(delta > 0.0, "Skin depth must be positive, got {delta}");
    }

    #[test]
    fn test_atmospheric_transmission_decreases_with_distance() {
        let atm = AtmosphericAbsorption::standard_conditions();
        let t1 = atm.transmission(0.557, 1.0);
        let t2 = atm.transmission(0.557, 10.0);
        assert!(
            t2 < t1,
            "Transmission should decrease with distance: T(1m)={t1}, T(10m)={t2}"
        );
    }

    #[test]
    fn test_transmission_windows_exist() {
        let atm = AtmosphericAbsorption::standard_conditions();
        let windows = atm.transmission_windows();
        assert!(
            !windows.is_empty(),
            "At least one THz transmission window must exist"
        );
    }

    #[test]
    fn test_surface_resistance_positive() {
        let gold = DrudeTHz::gold_thz();
        let rs = gold.surface_resistance_ohm(1.0);
        assert!(rs > 0.0, "Surface resistance must be positive, got {rs}");
    }

    // Ensure the OxiPhotonError type is reachable from this module (unused import guard).
    fn _assert_error_import(_: OxiPhotonError) {}
}
