//! Electromagnetic wave types and engineering utilities.
//!
//! Contains:
//! - Strongly-typed wrappers for `Wavelength`, `Frequency`, `WaveNumber`
//! - EM material and conductor functions (skin depth, plasma frequency, …)
//! - Waveguide dispersion relations (rectangular, circular)
//! - Quality-factor, boundary-condition, and power-flow helpers

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use super::conversion::{BOLTZMANN, ELECTRON_CHARGE, ELECTRON_MASS, SPEED_OF_LIGHT, Z0};

// ─── Strongly-typed EM quantities ────────────────────────────────────────────

/// Wavelength in metres (SI).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Wavelength(pub f64);

/// Frequency in Hz.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Frequency(pub f64);

/// Wave number in rad/m.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct WaveNumber(pub f64);

impl Wavelength {
    /// Construct from nanometres.
    pub fn from_nm(nm: f64) -> Self {
        Wavelength(nm * 1e-9)
    }

    /// Return value in nanometres.
    pub fn as_nm(self) -> f64 {
        self.0 * 1e9
    }

    /// Construct from micrometres.
    pub fn from_um(um: f64) -> Self {
        Wavelength(um * 1e-6)
    }

    /// Return value in micrometres.
    pub fn as_um(self) -> f64 {
        self.0 * 1e6
    }

    /// Convert to temporal frequency f = c / λ.
    pub fn to_frequency(self) -> Frequency {
        Frequency(SPEED_OF_LIGHT / self.0)
    }

    /// Convert to vacuum wave number k₀ = 2π / λ.
    pub fn to_wavenumber(self) -> WaveNumber {
        WaveNumber(2.0 * PI / self.0)
    }
}

impl Frequency {
    /// Convert to free-space wavelength.
    pub fn to_wavelength(self) -> Wavelength {
        Wavelength(SPEED_OF_LIGHT / self.0)
    }

    /// Convert to vacuum wave number k₀ = 2πf/c.
    pub fn to_wavenumber(self) -> WaveNumber {
        WaveNumber(2.0 * PI * self.0 / SPEED_OF_LIGHT)
    }
}

impl WaveNumber {
    /// Convert to free-space wavelength.
    pub fn to_wavelength(self) -> Wavelength {
        Wavelength(2.0 * PI / self.0)
    }

    /// Convert to temporal frequency.
    pub fn to_frequency(self) -> Frequency {
        Frequency(self.0 * SPEED_OF_LIGHT / (2.0 * PI))
    }
}

// ─── Conductor / plasma physics ───────────────────────────────────────────────

/// Skin depth δ = 1/√(π·f·μ_r·μ₀·σ) (m) in a conductor.
///
/// # Arguments
/// * `sigma` — electrical conductivity (S/m)
/// * `mu_r` — relative permeability (dimensionless; 1.0 for non-magnetic)
/// * `freq_hz` — frequency (Hz)
///
/// # Example
/// ```
/// use oxiphoton::units::electromagnetic::skin_depth;
/// // Copper: σ≈5.8e7 S/m, μ_r=1, f=1 GHz → δ ≈ 2.1 µm
/// let delta = skin_depth(5.8e7, 1.0, 1e9);
/// assert!(delta > 1e-6 && delta < 5e-6);
/// ```
pub fn skin_depth(sigma: f64, mu_r: f64, freq_hz: f64) -> f64 {
    let mu_0 = 4.0 * PI * 1e-7; // H/m
    (1.0 / (PI * freq_hz * mu_r * mu_0 * sigma)).sqrt()
}

/// Plasma angular frequency ω_p = √(n·q²/(ε₀·m·mass_ratio)) (rad/s).
///
/// # Arguments
/// * `n_carriers` — carrier density (m⁻³)
/// * `mass_ratio` — effective-mass ratio m*/m_e (dimensionless; 1.0 for electrons)
pub fn plasma_frequency(n_carriers: f64, mass_ratio: f64) -> f64 {
    let eps_0 = 8.854_187_812_8e-12;
    (n_carriers * ELECTRON_CHARGE * ELECTRON_CHARGE / (eps_0 * ELECTRON_MASS * mass_ratio)).sqrt()
}

/// Debye length λ_D = √(ε₀·k_B·T / (n·q²)) (m).
///
/// Length scale over which charge perturbations are screened in a plasma.
///
/// # Arguments
/// * `temperature_k` — electron temperature (K)
/// * `n_carriers` — carrier density (m⁻³)
pub fn debye_length(temperature_k: f64, n_carriers: f64) -> f64 {
    let eps_0 = 8.854_187_812_8e-12;
    (eps_0 * BOLTZMANN * temperature_k / (n_carriers * ELECTRON_CHARGE * ELECTRON_CHARGE)).sqrt()
}

// ─── EM wave impedance ────────────────────────────────────────────────────────

/// Wave impedance in a medium η = Z₀·√(μ_r / ε_r) (Ω).
///
/// For air (ε_r=1, μ_r=1) this returns Z₀ ≈ 376.73 Ω.
pub fn wave_impedance(eps_r: f64, mu_r: f64) -> f64 {
    Z0 * (mu_r / eps_r).sqrt()
}

// ─── Quality factor ──────────────────────────────────────────────────────────

/// Quality factor Q = f₀ / FWHM.
///
/// # Arguments
/// * `f0_hz` — resonant frequency (Hz)
/// * `fwhm_hz` — full-width at half-maximum linewidth (Hz)
pub fn quality_factor(f0_hz: f64, fwhm_hz: f64) -> f64 {
    f0_hz / fwhm_hz
}

/// Loaded Q-factor when an unloaded resonator is coupled with coupling coefficient κ.
///
/// Q_L = Q_0 / (1 + κ)
///
/// where κ is the coupling coefficient (κ=1 for critical coupling).
pub fn loaded_q(q_unloaded: f64, coupling: f64) -> f64 {
    q_unloaded / (1.0 + coupling)
}

/// Coupling coefficient κ from loaded and unloaded Q.
///
/// κ = Q_0/Q_L − 1
pub fn coupling_from_q(q_loaded: f64, q_unloaded: f64) -> f64 {
    q_unloaded / q_loaded - 1.0
}

// ─── Antenna / radiation resistance ──────────────────────────────────────────

/// Radiation resistance of a Hertzian (short) dipole (Ω).
///
/// R_rad = 80·π²·(L/λ)²
///
/// # Arguments
/// * `length_m` — dipole physical length (m), assumed ≪ λ
/// * `lambda_m` — free-space wavelength (m)
pub fn hertzian_dipole_radiation_resistance(length_m: f64, lambda_m: f64) -> f64 {
    80.0 * PI * PI * (length_m / lambda_m).powi(2)
}

/// Maximum directivity of a Hertzian dipole (dimensionless).
///
/// D = 1.5  at θ = π/2 (broadside).
pub fn hertzian_dipole_directivity() -> f64 {
    1.5
}

/// Effective aperture A_eff = D·λ²/(4π) from directivity and wavelength (m²).
pub fn effective_aperture(directivity: f64, lambda_m: f64) -> f64 {
    directivity * lambda_m * lambda_m / (4.0 * PI)
}

// ─── Noise ────────────────────────────────────────────────────────────────────

/// Noise figure (dB) of a two-port network from input/output signal and noise powers.
///
/// NF = 10·log10((S_in/N_in) / (S_out/N_out))
///
/// # Arguments
/// * `signal_in`, `noise_in` — input signal and noise powers (W)
/// * `signal_out`, `noise_out` — output signal and noise powers (W)
pub fn noise_figure_db(signal_in: f64, noise_in: f64, signal_out: f64, noise_out: f64) -> f64 {
    let snr_in = signal_in / noise_in;
    let snr_out = signal_out / noise_out;
    10.0 * (snr_in / snr_out).log10()
}

/// Thermal Johnson-Nyquist noise power spectral density N₀ = k_B·T (W/Hz).
pub fn johnson_noise_psd(temperature_k: f64) -> f64 {
    BOLTZMANN * temperature_k
}

// ─── EM boundary conditions ───────────────────────────────────────────────────

/// Check that tangential E-field is continuous across an interface (within tolerance).
///
/// Returns `true` if |E_tan1 − E_tan2| < tolerance.
pub fn check_tangential_e_continuity(e_tan1: f64, e_tan2: f64, tolerance: f64) -> bool {
    (e_tan1 - e_tan2).abs() < tolerance
}

/// Normal D-field jump across an interface due to free surface charge ρ_s (C/m²).
///
/// ε₁·E_n1 − ε₂·E_n2 = ρ_s  →  returns ε₁·E_n1 − ε₂·E_n2
pub fn normal_d_jump(eps1: f64, en1: f64, eps2: f64, en2: f64) -> f64 {
    eps1 * en1 - eps2 * en2
}

// ─── Rectangular waveguide ────────────────────────────────────────────────────

/// Longitudinal wave number k_z of TE/TM_{mn} mode in a rectangular waveguide (rad/m).
///
/// k_z = √(k₀² − (mπ/a)²) for the dominant TE₁₀ mode (n=0 term omitted for simplicity).
///
/// Returns `None` if the mode is below cutoff (k₀ < mπ/a).
///
/// # Arguments
/// * `k0` — free-space wave number 2π/λ (rad/m)
/// * `a` — broad-wall dimension (m)
/// * `m` — mode index (m ≥ 1)
pub fn rectangular_waveguide_kz(k0: f64, a: f64, m: u32) -> Option<f64> {
    let kc_sq = (m as f64 * PI / a).powi(2);
    let kz_sq = k0 * k0 - kc_sq;
    if kz_sq < 0.0 {
        None
    } else {
        Some(kz_sq.sqrt())
    }
}

/// Cutoff frequency of TE_{mn} mode in a rectangular waveguide (Hz).
///
/// f_c = c/(2π) · √((mπ/a)² + (nπ/b)²)
///
/// # Arguments
/// * `a` — broad-wall width (m)
/// * `b` — narrow-wall height (m)
/// * `m`, `n` — mode indices
pub fn rectangular_waveguide_cutoff_te(a: f64, b: f64, m: u32, n: u32) -> f64 {
    let kc = ((m as f64 * PI / a).powi(2) + (n as f64 * PI / b).powi(2)).sqrt();
    SPEED_OF_LIGHT * kc / (2.0 * PI)
}

/// Cutoff frequency of the TE₁₁ mode in a circular waveguide of radius r (Hz).
///
/// f_c = c · 1.841_18 / (2π·r),   where 1.841_18 ≈ first zero of J₁'.
pub fn circular_waveguide_te11_cutoff(radius: f64) -> f64 {
    SPEED_OF_LIGHT * 1.841_18 / (2.0 * PI * radius)
}

/// Cutoff frequency of the TM₀₁ mode in a circular waveguide of radius r (Hz).
///
/// f_c = c · 2.404_83 / (2π·r),   where 2.404_83 ≈ first zero of J₀.
pub fn circular_waveguide_tm01_cutoff(radius: f64) -> f64 {
    SPEED_OF_LIGHT * 2.404_83 / (2.0 * PI * radius)
}

/// Time-averaged power flow (W) of TE₁₀ mode in a rectangular waveguide.
///
/// P = (a·b / (4·Z_TE)) · |E_max|²
/// where Z_TE = Z₀·k₀/k_z is the TE wave impedance and E_max is the peak E-field (V/m).
///
/// # Arguments
/// * `e_max` — peak E-field amplitude (V/m)
/// * `a` — broad-wall width (m)
/// * `b` — narrow-wall height (m)
/// * `kz` — longitudinal wave number (rad/m)
/// * `k0` — free-space wave number (rad/m)
pub fn rectangular_waveguide_te10_power(e_max: f64, a: f64, b: f64, kz: f64, k0: f64) -> f64 {
    let z_te = Z0 * k0 / kz;
    a * b * e_max * e_max / (4.0 * z_te)
}

// ─── Dispersion / group velocity ─────────────────────────────────────────────

/// Group velocity v_g = dω/dβ ≈ Δω / Δβ (m/s).
///
/// Computed by finite differences from two (ω, k) points.
///
/// # Arguments
/// * `omega`, `k` — first point (rad/s, rad/m)
/// * `d_omega`, `d_k` — differences Δω = ω₂−ω₁, Δk = k₂−k₁
pub fn group_velocity(omega: f64, k: f64, d_omega: f64, d_k: f64) -> f64 {
    let _ = (omega, k); // present for context; d_omega/d_k carry the derivative
    if d_k.abs() < 1e-30 {
        SPEED_OF_LIGHT
    } else {
        d_omega / d_k
    }
}

/// Phase velocity v_ph = ω / β (m/s).
pub fn phase_velocity(omega: f64, k: f64) -> f64 {
    omega / k
}

/// Waveguide TE mode phase velocity v_ph = ω / k_z (m/s).
pub fn waveguide_phase_velocity(omega: f64, kz: f64) -> f64 {
    omega / kz
}

/// Waveguide TE mode group velocity v_g = c²·k_z / ω = c·(1-(f_c/f)²)^{1/2} (m/s).
pub fn waveguide_group_velocity(omega: f64, kz: f64) -> f64 {
    SPEED_OF_LIGHT * SPEED_OF_LIGHT * kz / omega
}

// ─── Fresnel reflection losses ────────────────────────────────────────────────

/// Power reflectance at a planar interface, averaged over polarisations,
/// from a medium of index n1 to n2 at angle θ_i (rad).
///
/// Returns (R_s, R_p) — individual polarisation reflectances.
pub fn fresnel_power_coefficients(n1: f64, n2: f64, theta_i_rad: f64) -> (f64, f64) {
    let sin_t = n1 / n2 * theta_i_rad.sin();
    if sin_t.abs() > 1.0 {
        return (1.0, 1.0); // TIR
    }
    let cos_i = theta_i_rad.cos();
    let cos_t = (1.0 - sin_t * sin_t).sqrt();
    let rs = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)).powi(2);
    let rp = ((n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)).powi(2);
    (rs, rp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn wavelength_nm_roundtrip() {
        let wl = Wavelength::from_nm(1550.0);
        assert_relative_eq!(wl.as_nm(), 1550.0, epsilon = 1e-10);
    }

    #[test]
    fn wavelength_um_roundtrip() {
        let wl = Wavelength::from_um(1.55);
        assert_relative_eq!(wl.as_um(), 1.55, epsilon = 1e-10);
    }

    #[test]
    fn wavelength_frequency_roundtrip() {
        let wl = Wavelength::from_nm(1550.0);
        let f = wl.to_frequency();
        let wl2 = f.to_wavelength();
        assert_relative_eq!(wl.0, wl2.0, epsilon = 1e-20);
    }

    #[test]
    fn wavelength_wavenumber_roundtrip() {
        let wl = Wavelength::from_nm(632.8);
        let k = wl.to_wavenumber();
        let wl2 = k.to_wavelength();
        assert_relative_eq!(wl.0, wl2.0, epsilon = 1e-20);
    }

    #[test]
    fn frequency_wavenumber_roundtrip() {
        let f = Frequency(193.4e12);
        let k = f.to_wavenumber();
        let f2 = k.to_frequency();
        assert_relative_eq!(f.0, f2.0, epsilon = 1e-2);
    }

    #[test]
    fn skin_depth_copper_1ghz() {
        // σ(Cu) ≈ 5.8e7 S/m, μ_r=1, f=1 GHz → δ ≈ 2.09 µm
        let delta = skin_depth(5.8e7, 1.0, 1e9);
        assert!(delta > 1.5e-6 && delta < 3.0e-6, "δ(Cu,1GHz)={delta:.2e} m");
    }

    #[test]
    fn skin_depth_decreases_with_freq() {
        let d1 = skin_depth(5.8e7, 1.0, 1e6);
        let d2 = skin_depth(5.8e7, 1.0, 1e9);
        assert!(d1 > d2, "Skin depth should decrease with frequency");
    }

    #[test]
    fn plasma_frequency_electron_density() {
        // n = 1e18 m⁻³, mass_ratio=1 → ω_p ≈ 1.78e9 rad/s
        let wp = plasma_frequency(1e18, 1.0);
        assert!(wp > 1e8 && wp < 1e11, "ω_p={wp:.2e}");
    }

    #[test]
    fn debye_length_basic() {
        // T=300 K, n=1e15 m⁻³ → λ_D in µm range
        let ld = debye_length(300.0, 1e15);
        assert!(ld > 1e-7 && ld < 1e-2, "λ_D={ld:.2e} m");
    }

    #[test]
    fn wave_impedance_vacuum() {
        let eta = wave_impedance(1.0, 1.0);
        assert_relative_eq!(eta, Z0, epsilon = 1e-10);
    }

    #[test]
    fn wave_impedance_glass() {
        // n=1.5, mu_r=1 → η = Z0/1.5
        let eta = wave_impedance(1.5 * 1.5, 1.0);
        assert_relative_eq!(eta, Z0 / 1.5, epsilon = 1e-6);
    }

    #[test]
    fn quality_factor_basic() {
        let q = quality_factor(193e12, 193e6); // Q=1e6
        assert_relative_eq!(q, 1e6, epsilon = 1.0);
    }

    #[test]
    fn loaded_q_basic() {
        let ql = loaded_q(1e6, 1.0); // critical coupling → Q_L = Q_0/2
        assert_relative_eq!(ql, 5e5, epsilon = 1.0);
    }

    #[test]
    fn hertzian_dipole_rrad() {
        // L=0.01λ → R_rad = 80π²·0.01² ≈ 0.0789 Ω
        let r = hertzian_dipole_radiation_resistance(0.01, 1.0);
        assert_relative_eq!(r, 80.0 * PI * PI * 1e-4, epsilon = 1e-10);
    }

    #[test]
    fn waveguide_cutoff_te10() {
        // WR-90: a=22.86mm, b=10.16mm → f_c(TE10) ≈ 6.557 GHz
        let fc = rectangular_waveguide_cutoff_te(22.86e-3, 10.16e-3, 1, 0);
        assert_relative_eq!(fc, 6.557e9, epsilon = 0.01e9);
    }

    #[test]
    fn waveguide_kz_below_cutoff_is_none() {
        // TE10 in WR-90 at 5 GHz (below f_c ≈ 6.56 GHz)
        let k0 = 2.0 * PI * 5e9 / SPEED_OF_LIGHT;
        let result = rectangular_waveguide_kz(k0, 22.86e-3, 1);
        assert!(result.is_none());
    }

    #[test]
    fn waveguide_kz_above_cutoff() {
        // At 10 GHz, well above cutoff
        let k0 = 2.0 * PI * 10e9 / SPEED_OF_LIGHT;
        let kz = rectangular_waveguide_kz(k0, 22.86e-3, 1).expect("propagating mode");
        assert!(kz > 0.0);
    }

    #[test]
    fn circular_waveguide_te11() {
        // r=10mm → f_c = c·1.841/(2π·0.01) ≈ 8.79 GHz
        let fc = circular_waveguide_te11_cutoff(10e-3);
        assert_relative_eq!(fc, 8.79e9, epsilon = 0.01e9);
    }

    #[test]
    fn phase_velocity_equals_c_in_vacuum() {
        let omega = 2.0 * PI * 300e12;
        let k = omega / SPEED_OF_LIGHT;
        let vph = phase_velocity(omega, k);
        assert_relative_eq!(vph, SPEED_OF_LIGHT, epsilon = 1.0);
    }

    #[test]
    fn fresnel_tir_at_grazing() {
        // TIR from glass (n1=1.5) to air (n2=1) at θ=90°
        let (rs, rp) = fresnel_power_coefficients(1.5, 1.0, std::f64::consts::FRAC_PI_2);
        assert_relative_eq!(rs, 1.0, epsilon = 1e-10);
        assert_relative_eq!(rp, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn johnson_noise_psd_room_temp() {
        let n0 = johnson_noise_psd(290.0);
        // k_B·T ≈ 4.0e-21 W/Hz at 290 K
        assert_relative_eq!(n0, 4.0e-21, epsilon = 0.1e-21);
    }

    #[test]
    fn boundary_condition_passes() {
        assert!(check_tangential_e_continuity(1.0, 1.0 + 1e-10, 1e-9));
    }

    #[test]
    fn boundary_condition_fails() {
        assert!(!check_tangential_e_continuity(1.0, 2.0, 1e-9));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn wavelength_frequency_roundtrip(wl_nm in 100.0..20000.0_f64) {
            let wl = Wavelength::from_nm(wl_nm);
            let f = wl.to_frequency();
            let wl2 = f.to_wavelength();
            prop_assert!((wl.0 - wl2.0).abs() / wl.0 < 1e-12);
        }

        #[test]
        fn wavelength_wavenumber_roundtrip(wl_nm in 100.0..20000.0_f64) {
            let wl = Wavelength::from_nm(wl_nm);
            let k = wl.to_wavenumber();
            let wl2 = k.to_wavelength();
            prop_assert!((wl.0 - wl2.0).abs() / wl.0 < 1e-12);
        }

        #[test]
        fn frequency_wavenumber_roundtrip(freq in 1e12..3e15_f64) {
            let f = Frequency(freq);
            let k = f.to_wavenumber();
            let f2 = k.to_frequency();
            prop_assert!((f.0 - f2.0).abs() / f.0 < 1e-12);
        }
    }
}
