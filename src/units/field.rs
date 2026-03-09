//! Field unit conversions and intensity calculations for photonics.
//!
//! Provides functions to convert between:
//! - Electric field amplitude (V/m) and irradiance (W/m²)
//! - Optical power (W) and photon flux / count rate
//! - Pulse energy, peak intensity, and fluence
//! - Nonlinear phase (B-integral) and field enhancement

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use super::conversion::{PLANCK, SPEED_OF_LIGHT, Z0};

// ─── Strongly-typed field wrappers ────────────────────────────────────────────

/// Electric field vector (Ex, Ey, Ez) — complex-valued.
#[derive(Debug, Clone, Copy)]
pub struct ElectricField(pub [Complex64; 3]);

/// Magnetic field vector (Hx, Hy, Hz) — complex-valued.
#[derive(Debug, Clone, Copy)]
pub struct MagneticField(pub [Complex64; 3]);

/// Optical intensity (irradiance) in W/m².
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Intensity(pub f64);

/// Poynting vector magnitude |S| in W/m².
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Poynting(pub f64);

impl ElectricField {
    /// Zero field.
    pub fn zero() -> Self {
        Self([Complex64::new(0.0, 0.0); 3])
    }

    /// Compute |E|² = |Ex|² + |Ey|² + |Ez|².
    pub fn intensity_squared(&self) -> f64 {
        self.0.iter().map(|c| c.norm_sqr()).sum()
    }

    /// Time-averaged irradiance I = n·|E|²/(2·Z₀) for a plane wave in medium n.
    pub fn irradiance(&self, n: f64) -> f64 {
        0.5 * n * self.intensity_squared() / Z0
    }
}

impl MagneticField {
    /// Zero field.
    pub fn zero() -> Self {
        Self([Complex64::new(0.0, 0.0); 3])
    }

    /// Compute |H|².
    pub fn intensity_squared(&self) -> f64 {
        self.0.iter().map(|c| c.norm_sqr()).sum()
    }
}

// ─── Irradiance / E-field conversion ─────────────────────────────────────────

/// Compute irradiance (W/m²) from peak E-field amplitude (V/m) in a medium of index n.
///
/// I = n · |E|² / (2 · Z₀)
///
/// # Example
/// ```
/// use oxiphoton::units::field::irradiance_from_e_field;
/// let i = irradiance_from_e_field(1e6, 1.0);
/// assert!(i > 0.0);
/// ```
pub fn irradiance_from_e_field(e_amplitude: f64, n: f64) -> f64 {
    0.5 * n * e_amplitude * e_amplitude / Z0
}

/// Compute E-field amplitude (V/m) from irradiance (W/m²) and refractive index n.
///
/// E = √(2 · Z₀ · I / n)
pub fn e_field_from_irradiance(irradiance: f64, n: f64) -> f64 {
    (2.0 * Z0 * irradiance / n).sqrt()
}

/// Convert E-field (V/m) to power density (W/m²) — alias for `irradiance_from_e_field`.
pub fn e_field_to_power_density(e: f64, n: f64) -> f64 {
    irradiance_from_e_field(e, n)
}

/// Magnetic field H (A/m) from E-field amplitude in medium of index n.
///
/// H = n · E / Z₀
pub fn h_from_e(e: f64, n: f64) -> f64 {
    n * e / Z0
}

// ─── Time-averaged Poynting vector ────────────────────────────────────────────

/// Time-averaged Poynting vector magnitude ⟨S⟩ = n · |E|² / (2 · Z₀) (W/m²).
///
/// For a plane wave in a medium of index n, this equals the irradiance.
pub fn time_averaged_poynting(e_amplitude: f64, n: f64) -> f64 {
    irradiance_from_e_field(e_amplitude, n)
}

// ─── Pulse / beam intensity ───────────────────────────────────────────────────

/// Peak intensity (W/m²) from pulse energy, duration, and beam area.
///
/// Assumes a Gaussian pulse shape: I_peak = E / (√(π/2) · τ · A)
/// For a square-pulse approximation: I_peak = E / (τ · A).
///
/// This function uses the simple rectangular approximation.
///
/// # Arguments
/// * `energy_j` — pulse energy (J)
/// * `duration_s` — pulse duration (FWHM, s)
/// * `area_m2` — beam cross-sectional area (m²)
pub fn peak_intensity_from_pulse(energy_j: f64, duration_s: f64, area_m2: f64) -> f64 {
    energy_j / (duration_s * area_m2)
}

/// Fluence (J/m²) from intensity (W/m²) and duration (s).
///
/// F = I · τ
pub fn fluence(intensity_wm2: f64, duration_s: f64) -> f64 {
    intensity_wm2 * duration_s
}

/// Intensity required for a given fluence and pulse duration.
pub fn intensity_from_fluence(fluence_jm2: f64, duration_s: f64) -> f64 {
    fluence_jm2 / duration_s
}

// ─── Photon counting ──────────────────────────────────────────────────────────

/// Photon flux density (photons·m⁻²·s⁻¹) from irradiance and photon energy.
///
/// Φ = I / E_photon
pub fn photon_flux(irradiance_wm2: f64, photon_energy_j: f64) -> f64 {
    irradiance_wm2 / photon_energy_j
}

/// Convert optical power (W) to photon count rate (photons/s) at wavelength λ (m).
///
/// Ṅ = P · λ / (h · c)
pub fn power_to_photon_rate(power_w: f64, wavelength_m: f64) -> f64 {
    power_w * wavelength_m / (PLANCK * SPEED_OF_LIGHT)
}

/// Convert photon count rate (photons/s) to optical power (W) at wavelength λ (m).
pub fn photon_rate_to_power(rate: f64, wavelength_m: f64) -> f64 {
    rate * PLANCK * SPEED_OF_LIGHT / wavelength_m
}

// ─── Waveguide / fibre mode ───────────────────────────────────────────────────

/// Optical power (W) from peak E-field amplitude (V/m), mode area (m²), and index n.
///
/// P = I · A_eff = n · |E|² · A_eff / (2 · Z₀)
pub fn optical_power(e_amplitude: f64, mode_area_m2: f64, n: f64) -> f64 {
    irradiance_from_e_field(e_amplitude, n) * mode_area_m2
}

/// Peak E-field amplitude (V/m) in a waveguide mode from power and effective mode area.
pub fn mode_e_field_amplitude(power_w: f64, mode_area_m2: f64, n: f64) -> f64 {
    e_field_from_irradiance(power_w / mode_area_m2, n)
}

// ─── Nonlinear optics ─────────────────────────────────────────────────────────

/// B-integral (accumulated nonlinear phase) along a propagation path.
///
/// B = (2π / λ) · n₂ · I · L
///
/// # Arguments
/// * `intensity` — peak intensity (W/m²)
/// * `n2` — nonlinear refractive index (m²/W); ~2.6e-20 for silica
/// * `length` — propagation length (m)
/// * `lambda` — free-space wavelength (m)
pub fn b_integral(intensity: f64, n2: f64, length: f64, lambda: f64) -> f64 {
    2.0 * PI / lambda * n2 * intensity * length
}

/// Self-phase-modulation phase shift: φ_SPM = γ · P · L_eff
///
/// # Arguments
/// * `gamma` — nonlinear coefficient γ = n₂·ω/(c·A_eff) (rad·W⁻¹·m⁻¹)
/// * `power_w` — peak power (W)
/// * `l_eff` — effective length L_eff = (1 − e^{−αL})/α (m)
pub fn spm_phase_shift(gamma: f64, power_w: f64, l_eff: f64) -> f64 {
    gamma * power_w * l_eff
}

/// Effective nonlinear length L_eff = (1 − exp(−α·L)) / α (m).
///
/// Accounts for propagation loss α (m⁻¹) over length L (m).
pub fn effective_length(alpha: f64, length: f64) -> f64 {
    if alpha.abs() < 1e-30 {
        length
    } else {
        (1.0 - (-alpha * length).exp()) / alpha
    }
}

// ─── Resonator field enhancement ──────────────────────────────────────────────

/// Field enhancement factor in a resonator (cavity or photonic crystal).
///
/// ξ = √(Q · λ³ / (n³ · V_mode))
///
/// where V_mode is the mode volume (m³) and n is the refractive index.
/// This is a simplified figure of merit related to Purcell factor.
///
/// # Arguments
/// * `q` — quality factor
/// * `mode_volume` — mode volume (m³)
/// * `lambda` — free-space wavelength (m)
/// * `n` — refractive index of the cavity medium
pub fn field_enhancement_resonator(q: f64, mode_volume: f64, lambda: f64, n: f64) -> f64 {
    let lambda_n = lambda / n;
    (q * lambda_n * lambda_n * lambda_n / mode_volume).sqrt()
}

/// Purcell factor F_P = (3/(4π²)) · (λ/n)³ · (Q / V_mode).
///
/// Describes the enhancement of spontaneous emission into a cavity mode.
pub fn purcell_factor(q: f64, mode_volume: f64, lambda: f64, n: f64) -> f64 {
    let lambda_n = lambda / n;
    (3.0 / (4.0 * PI * PI)) * lambda_n * lambda_n * lambda_n * q / mode_volume
}

// ─── Damage thresholds ────────────────────────────────────────────────────────

/// Number of photons per unit volume (m⁻³) from irradiance and wavelength.
///
/// ρ_photon = I / (c · E_photon) = I · λ / (c · h · c) = I · λ / (h · c²)
pub fn photon_density(irradiance_wm2: f64, wavelength_m: f64) -> f64 {
    irradiance_wm2 * wavelength_m / (PLANCK * SPEED_OF_LIGHT * SPEED_OF_LIGHT)
}

/// Multi-photon absorption probability (relative, dimensionless prefactor).
///
/// Γ_MPA ∝ σ_N · I^N  — returns I^N normalised to I_ref^(N-1).
///
/// # Arguments
/// * `intensity` — peak intensity (W/m²)
/// * `order` — number of photons N (e.g., 2 for two-photon absorption)
pub fn multiphoton_absorption_scaling(intensity: f64, order: u32) -> f64 {
    intensity.powi(order as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn irradiance_roundtrip() {
        let e0 = 1e6; // V/m
        let n = 1.5;
        let i = irradiance_from_e_field(e0, n);
        let e0_back = e_field_from_irradiance(i, n);
        assert_relative_eq!(e0, e0_back, epsilon = 1e-8);
    }

    #[test]
    fn irradiance_air_formula() {
        // E=1 V/m, n=1 → I = 1/(2·Z0)
        let i = irradiance_from_e_field(1.0, 1.0);
        assert_relative_eq!(i, 0.5 / Z0, epsilon = 1e-14);
    }

    #[test]
    fn h_from_e_vacuum() {
        // n=1 → H = E/Z0
        let h = h_from_e(376.73, 1.0);
        assert_relative_eq!(h, 1.0, epsilon = 1e-3);
    }

    #[test]
    fn poynting_equals_irradiance() {
        let e = 1e5;
        let n = 1.5;
        let i = irradiance_from_e_field(e, n);
        let s = time_averaged_poynting(e, n);
        assert_relative_eq!(i, s, epsilon = 1e-12);
    }

    #[test]
    fn peak_intensity_basic() {
        // 1 µJ pulse, 1 ps duration, 1 mm² area → I = 1e-6/(1e-12·1e-6) = 1e12 W/m²
        let i = peak_intensity_from_pulse(1e-6, 1e-12, 1e-6);
        assert_relative_eq!(i, 1e12, epsilon = 1.0);
    }

    #[test]
    fn fluence_basic() {
        // I=1e12 W/m², τ=1 ps → F=1e12·1e-12=1 J/m²
        let f = fluence(1e12, 1e-12);
        assert_relative_eq!(f, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn photon_rate_roundtrip() {
        let power = 1e-3; // 1 mW
        let wl = 1550e-9;
        let rate = power_to_photon_rate(power, wl);
        let power_back = photon_rate_to_power(rate, wl);
        assert_relative_eq!(power, power_back, epsilon = 1e-17);
    }

    #[test]
    fn photon_rate_1mw_at_1550nm() {
        // P=1mW, λ=1550nm → Ṅ ≈ 7.8e15 photons/s
        let rate = power_to_photon_rate(1e-3, 1550e-9);
        assert!(rate > 7e15 && rate < 9e15, "Ṅ={rate:.2e}");
    }

    #[test]
    fn optical_power_from_e_field() {
        let e = 1e6;
        let area = 1e-12; // 1 µm²
        let n = 3.48; // Silicon
        let p = optical_power(e, area, n);
        let i = irradiance_from_e_field(e, n);
        assert_relative_eq!(p, i * area, epsilon = 1e-10);
    }

    #[test]
    fn b_integral_silica() {
        // n2=2.6e-20 m²/W, I=1e13 W/m², L=1m, λ=1550nm → B ≈ 1.08 rad
        let b = b_integral(1e13, 2.6e-20, 1.0, 1550e-9);
        assert!(b > 0.5 && b < 2.0, "B={b:.3}");
    }

    #[test]
    fn effective_length_lossless() {
        let l_eff = effective_length(0.0, 10.0);
        assert_relative_eq!(l_eff, 10.0, epsilon = 1e-12);
    }

    #[test]
    fn effective_length_high_loss() {
        // α → ∞ → L_eff → 1/α (short)
        let alpha = 1e6;
        let l = 100.0;
        let l_eff = effective_length(alpha, l);
        assert_relative_eq!(l_eff, 1.0 / alpha, epsilon = 1e-10);
    }

    #[test]
    fn purcell_factor_reasonable() {
        // Q=1e6, V=1 (λ/n)³, λ=1.5µm, n=3.5 → F_P = 3/(4π²) ≈ 0.076 >> 1 for small V
        let lambda: f64 = 1.5e-6;
        let n: f64 = 3.5;
        let v_mode = (lambda / n).powi(3);
        let fp = purcell_factor(1e4, v_mode, lambda, n);
        assert!(fp > 100.0, "F_P={fp:.1}");
    }

    #[test]
    fn electric_field_zero() {
        let e = ElectricField::zero();
        assert_relative_eq!(e.intensity_squared(), 0.0, epsilon = 1e-30);
    }

    #[test]
    fn magnetic_field_zero() {
        let h = MagneticField::zero();
        assert_relative_eq!(h.intensity_squared(), 0.0, epsilon = 1e-30);
    }
}
