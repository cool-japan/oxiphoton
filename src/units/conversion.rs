//! Physical constants and unit conversion utilities for photonics and optics.
//!
//! All values are in SI units unless otherwise noted. Constants are taken from
//! CODATA 2018 / NIST recommended values.

use std::f64::consts::PI;

// ─── Fundamental physical constants ──────────────────────────────────────────

/// Speed of light in vacuum (m/s)
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Vacuum permittivity ε₀ (F/m)
pub const EPSILON_0: f64 = 8.854_187_812_8e-12;

/// Vacuum permeability μ₀ (H/m)
pub const MU_0: f64 = 1.256_637_062_12e-6;

/// Impedance of free space Z₀ = √(μ₀/ε₀) ≈ 376.730 Ω
pub const Z0: f64 = 376.730_313_668;

/// Planck constant h (J·s)
pub const PLANCK: f64 = 6.626_070_15e-34;

/// Reduced Planck constant ℏ = h / (2π) (J·s)
pub const HBAR: f64 = 1.054_571_817e-34;

/// Boltzmann constant k_B (J/K)
pub const BOLTZMANN: f64 = 1.380_649e-23;

/// Elementary charge e (C)
pub const ELECTRON_CHARGE: f64 = 1.602_176_634e-19;

/// Electron rest mass m_e (kg)
pub const ELECTRON_MASS: f64 = 9.109_383_701_5e-31;

/// Avogadro constant N_A (mol⁻¹)
pub const AVOGADRO: f64 = 6.022_140_76e23;

/// Fine-structure constant α ≈ 1/137 (dimensionless)
pub const FINE_STRUCTURE: f64 = 7.297_352_569_3e-3;

/// Stefan–Boltzmann constant σ (W·m⁻²·K⁻⁴)
pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;

// ─── Length conversion factors ────────────────────────────────────────────────

/// Nanometres to metres
pub const NM_TO_M: f64 = 1e-9;

/// Micrometres to metres
pub const UM_TO_M: f64 = 1e-6;

/// Centimetres to metres
pub const CM_TO_M: f64 = 1e-2;

/// Millimetres to metres
pub const MM_TO_M: f64 = 1e-3;

// ─── Energy conversion factors ───────────────────────────────────────────────

/// Electron-volts to joules
pub const EV_TO_J: f64 = ELECTRON_CHARGE; // 1.602_176_634e-19

/// Joules to electron-volts
pub const J_TO_EV: f64 = 1.0 / EV_TO_J;

// ─── Frequency / wavelength conversions ──────────────────────────────────────

/// Convert wavelength λ (m) to frequency f (Hz): f = c / λ
///
/// # Panics-free
/// Returns `f64::INFINITY` if `lambda_m == 0`.
///
/// # Example
/// ```
/// use oxiphoton::units::conversion::wavelength_to_frequency;
/// let f = wavelength_to_frequency(1550e-9); // ~193.4 THz
/// assert!((f - 193.4e12).abs() / 193.4e12 < 1e-3);
/// ```
pub fn wavelength_to_frequency(lambda_m: f64) -> f64 {
    SPEED_OF_LIGHT / lambda_m
}

/// Convert frequency f (Hz) to wavelength λ (m): λ = c / f
pub fn frequency_to_wavelength(freq_hz: f64) -> f64 {
    SPEED_OF_LIGHT / freq_hz
}

/// Convert wavelength λ (m) to angular frequency ω (rad/s): ω = 2π·c / λ
pub fn wavelength_to_omega(lambda_m: f64) -> f64 {
    2.0 * PI * SPEED_OF_LIGHT / lambda_m
}

/// Convert angular frequency ω (rad/s) to wavelength λ (m): λ = 2π·c / ω
pub fn omega_to_wavelength(omega: f64) -> f64 {
    2.0 * PI * SPEED_OF_LIGHT / omega
}

// ─── Energy ↔ wavelength ─────────────────────────────────────────────────────

/// Convert photon energy in eV to wavelength in nm.
///
/// Uses hc ≈ 1239.84193 eV·nm.
///
/// # Example
/// ```
/// use oxiphoton::units::conversion::ev_to_wavelength_nm;
/// let wl = ev_to_wavelength_nm(0.8); // ~1550 nm
/// assert!((wl - 1549.8).abs() < 1.0);
/// ```
pub fn ev_to_wavelength_nm(ev: f64) -> f64 {
    // hc = (6.62607015e-34 J·s × 2.99792458e8 m/s) / (1.602176634e-19 J/eV) × 1e9 nm/m
    1_239.841_93 / ev
}

/// Convert wavelength in nm to photon energy in eV.
pub fn wavelength_nm_to_ev(nm: f64) -> f64 {
    1_239.841_93 / nm
}

// ─── Power (dB / linear) ─────────────────────────────────────────────────────

/// Convert optical power from dBm to watts.
///
/// # Example
/// ```
/// use oxiphoton::units::conversion::dbm_to_watts;
/// let w = dbm_to_watts(0.0); // 0 dBm = 1 mW
/// assert!((w - 1e-3).abs() < 1e-15);
/// ```
pub fn dbm_to_watts(dbm: f64) -> f64 {
    1e-3 * 10_f64.powf(dbm / 10.0)
}

/// Convert optical power from watts to dBm.
///
/// # Example
/// ```
/// use oxiphoton::units::conversion::watts_to_dbm;
/// let dbm = watts_to_dbm(1e-3); // 1 mW = 0 dBm
/// assert!(dbm.abs() < 1e-10);
/// ```
pub fn watts_to_dbm(w: f64) -> f64 {
    10.0 * (w / 1e-3).log10()
}

// ─── Propagation constant ─────────────────────────────────────────────────────

/// Convert effective index n_eff to propagation constant β (rad/m):
/// β = 2π·n_eff / λ
///
/// # Arguments
/// * `n_eff` — effective refractive index (dimensionless)
/// * `lambda_m` — free-space wavelength (m)
pub fn neff_to_beta(n_eff: f64, lambda_m: f64) -> f64 {
    2.0 * PI * n_eff / lambda_m
}

/// Convert propagation constant β (rad/m) back to effective index:
/// n_eff = β·λ / (2π)
pub fn beta_to_neff(beta: f64, lambda_m: f64) -> f64 {
    beta * lambda_m / (2.0 * PI)
}

// ─── Temperature ─────────────────────────────────────────────────────────────

/// Convert Celsius to Kelvin.
pub fn temperature_c_to_k(c: f64) -> f64 {
    c + 273.15
}

/// Convert Kelvin to Celsius.
pub fn temperature_k_to_c(k: f64) -> f64 {
    k - 273.15
}

// ─── Loss / attenuation ───────────────────────────────────────────────────────

/// Convert loss coefficient (m⁻¹) to dB/cm.
///
/// For a power propagating as exp(-α·z), the loss in dB/cm is
/// α_dB = α · 10 / (ln10 · 100).
pub fn loss_per_m_to_db_per_cm(loss_per_m: f64) -> f64 {
    loss_per_m * 10.0 / (10_f64.ln() * 100.0)
}

/// Convert loss coefficient in dB/cm to m⁻¹.
pub fn db_per_cm_to_loss_per_m(db_per_cm: f64) -> f64 {
    db_per_cm * 10_f64.ln() * 100.0 / 10.0
}

/// Convert attenuation coefficient in dB/km to m⁻¹ (Neper per metre).
pub fn db_per_km_to_loss_per_m(db_per_km: f64) -> f64 {
    db_per_km * 10_f64.ln() / (10.0 * 1000.0)
}

// ─── Resonator / cavity ───────────────────────────────────────────────────────

/// Compute Q-factor from finesse, FSR (Hz), and resonance frequency (Hz).
///
/// Q = F · f₀ / FSR,  where F is the finesse.
///
/// # Arguments
/// * `finesse` — cavity finesse (dimensionless)
/// * `fsr_hz` — free spectral range (Hz)
/// * `resonance_hz` — resonance frequency (Hz)
pub fn finesse_to_q(finesse: f64, fsr_hz: f64, resonance_hz: f64) -> f64 {
    finesse * resonance_hz / fsr_hz
}

/// Compute FWHM linewidth (Hz) from Q-factor and resonance frequency (Hz).
///
/// Δf = f₀ / Q
pub fn q_to_linewidth_hz(q: f64, resonance_hz: f64) -> f64 {
    resonance_hz / q
}

// ─── Group velocity ───────────────────────────────────────────────────────────

/// Convert group index n_g to group velocity v_g = c / n_g (m/s).
pub fn group_index_to_group_velocity(ng: f64) -> f64 {
    SPEED_OF_LIGHT / ng
}

/// Convert GVD in ps/(nm·km) to SI units (s/m²).
///
/// D \[ps/(nm·km)\] × 10⁻¹² s/ps × 1/(10⁻⁹ m/nm) × 1/(10³ m/km)
///               = D × 10⁻²⁷ / 10⁻⁹ / 10³ s/m²
///               = D × 10⁻²⁷ × 10⁹ / 10³ s/m² = D × 10⁻²¹ s/m²
///               => D_SI = D × 1e-6 / 1e6 = D × 1e-3 / 1e12
///
/// More precisely: D \[ps/nm/km\] → D × 1e-12 / (1e-9 × 1e3) s/m² = D × 1e-6 s/m²
///               but conventional factor is ×10⁻²¹ / 10⁻⁹·10³ … let's derive explicitly:
///
/// Δτ \[ps\] / (Δλ \[nm\] · L \[km\]) = D
/// → Δτ \[s\] / (Δλ \[m\] · L \[m\]) = D × 10⁻¹² / (10⁻⁹ × 10³) = D × 10⁻⁶  s/m²
pub fn dispersion_ps_per_nm_per_km_to_si(d: f64) -> f64 {
    d * 1e-6
}

/// Convert dispersion from SI (s/m²) back to ps/(nm·km).
pub fn dispersion_si_to_ps_per_nm_per_km(d_si: f64) -> f64 {
    d_si * 1e6
}

// ─── Photon energy ────────────────────────────────────────────────────────────

/// Photon energy (J) for a given wavelength (m): E = hc/λ
pub fn photon_energy_j(lambda_m: f64) -> f64 {
    PLANCK * SPEED_OF_LIGHT / lambda_m
}

/// Photon energy (eV) for a given wavelength (nm).
pub fn photon_energy_ev(lambda_nm: f64) -> f64 {
    wavelength_nm_to_ev(lambda_nm)
}

// ─── Thermal energy ──────────────────────────────────────────────────────────

/// Thermal voltage V_T = k_B·T/q (V) — useful for diode equations.
pub fn thermal_voltage(temperature_k: f64) -> f64 {
    BOLTZMANN * temperature_k / ELECTRON_CHARGE
}

/// Thermal energy k_B·T (J).
pub fn thermal_energy_j(temperature_k: f64) -> f64 {
    BOLTZMANN * temperature_k
}

/// Thermal energy k_B·T (eV).
pub fn thermal_energy_ev(temperature_k: f64) -> f64 {
    thermal_energy_j(temperature_k) * J_TO_EV
}

// ─── Angle conversions ────────────────────────────────────────────────────────

/// Convert degrees to radians.
#[inline]
pub fn deg_to_rad(deg: f64) -> f64 {
    deg * PI / 180.0
}

/// Convert radians to degrees.
#[inline]
pub fn rad_to_deg(rad: f64) -> f64 {
    rad * 180.0 / PI
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn z0_from_constants() {
        let z0_calc = (MU_0 / EPSILON_0).sqrt();
        assert_relative_eq!(z0_calc, Z0, epsilon = 1e-3);
    }

    #[test]
    fn c_from_mu_eps() {
        let c_calc = 1.0 / (MU_0 * EPSILON_0).sqrt();
        assert_relative_eq!(c_calc, SPEED_OF_LIGHT, epsilon = 1.0);
    }

    #[test]
    fn wavelength_frequency_roundtrip() {
        let lambda = 1550e-9;
        let f = wavelength_to_frequency(lambda);
        let lambda2 = frequency_to_wavelength(f);
        assert_relative_eq!(lambda, lambda2, epsilon = 1e-25);
    }

    #[test]
    fn wavelength_omega_roundtrip() {
        let lambda = 800e-9;
        let omega = wavelength_to_omega(lambda);
        let lambda2 = omega_to_wavelength(omega);
        assert_relative_eq!(lambda, lambda2, epsilon = 1e-25);
    }

    #[test]
    fn ev_wavelength_roundtrip() {
        // 1550 nm ≈ 0.7999 eV
        let ev = wavelength_nm_to_ev(1550.0);
        let nm = ev_to_wavelength_nm(ev);
        assert_relative_eq!(nm, 1550.0, epsilon = 1e-4);
    }

    #[test]
    fn dbm_zero_is_one_mw() {
        let w = dbm_to_watts(0.0);
        assert_relative_eq!(w, 1e-3, epsilon = 1e-15);
    }

    #[test]
    fn one_mw_is_zero_dbm() {
        let dbm = watts_to_dbm(1e-3);
        assert!(dbm.abs() < 1e-10);
    }

    #[test]
    fn dbm_roundtrip() {
        let dbm = 10.0;
        let w = dbm_to_watts(dbm);
        let dbm2 = watts_to_dbm(w);
        assert_relative_eq!(dbm, dbm2, epsilon = 1e-10);
    }

    #[test]
    fn neff_beta_roundtrip() {
        let n_eff = 2.5;
        let lambda = 1550e-9;
        let beta = neff_to_beta(n_eff, lambda);
        let n_eff2 = beta_to_neff(beta, lambda);
        assert_relative_eq!(n_eff, n_eff2, epsilon = 1e-14);
    }

    #[test]
    fn temperature_c_k_roundtrip() {
        let c = 25.0;
        let k = temperature_c_to_k(c);
        let c2 = temperature_k_to_c(k);
        assert_relative_eq!(c, c2, epsilon = 1e-12);
    }

    #[test]
    fn loss_db_cm_roundtrip() {
        let loss = 10.0; // m⁻¹
        let db = loss_per_m_to_db_per_cm(loss);
        let loss2 = db_per_cm_to_loss_per_m(db);
        assert_relative_eq!(loss, loss2, epsilon = 1e-12);
    }

    #[test]
    fn finesse_to_q_basic() {
        // F=100, FSR=100e9 Hz, f0=193e12 Hz → Q = 100 * 193e12 / 100e9 = 193000
        let q = finesse_to_q(100.0, 100e9, 193e12);
        assert_relative_eq!(q, 193_000.0, epsilon = 1.0);
    }

    #[test]
    fn q_to_linewidth_basic() {
        let q = 1e6;
        let f0 = 200e12;
        let lw = q_to_linewidth_hz(q, f0);
        assert_relative_eq!(lw, 200e6, epsilon = 1.0);
    }

    #[test]
    fn group_velocity_silicon_ng4() {
        let vg = group_index_to_group_velocity(4.0);
        assert_relative_eq!(vg, SPEED_OF_LIGHT / 4.0, epsilon = 1.0);
    }

    #[test]
    fn dispersion_roundtrip() {
        let d = 17.0; // ps/nm/km (typical SMF-28)
        let d_si = dispersion_ps_per_nm_per_km_to_si(d);
        let d2 = dispersion_si_to_ps_per_nm_per_km(d_si);
        assert_relative_eq!(d, d2, epsilon = 1e-10);
    }

    #[test]
    fn photon_energy_visible() {
        // 550 nm green photon ≈ 2.25 eV
        let e = photon_energy_ev(550.0);
        assert!(e > 2.2 && e < 2.3, "E(550nm)={e:.4} eV");
    }

    #[test]
    fn thermal_voltage_room_temp() {
        // At 300 K, V_T ≈ 25.85 mV
        let vt = thermal_voltage(300.0);
        assert_relative_eq!(vt, 0.025_852, epsilon = 1e-5);
    }

    #[test]
    fn deg_rad_roundtrip() {
        let deg = 45.0;
        let rad = deg_to_rad(deg);
        let deg2 = rad_to_deg(rad);
        assert_relative_eq!(deg, deg2, epsilon = 1e-13);
    }

    #[test]
    fn hbar_from_planck() {
        // HBAR is defined as exact constant; verify it matches h/(2π) to within
        // the last digit of PLANCK's precision (relative tolerance 1e-9)
        assert_relative_eq!(HBAR, PLANCK / (2.0 * PI), epsilon = 1e-36);
    }
}
