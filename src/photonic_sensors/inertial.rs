//! Photonic inertial and mechanical sensors.
//!
//! Models:
//! - Photonic MEMS accelerometer (optical-readout proof-mass).
//! - Photonic pressure sensor (Fabry-Pérot cavity on clamped circular membrane).
//! - Integrated optical strain gauge (waveguide Bragg grating, silicon nitride).

use std::f64::consts::PI;

/// Boltzmann constant (J/K).
const K_B: f64 = 1.380_649e-23;
/// Standard gravity (m/s²).
const G_ACCEL: f64 = 9.80665;
// ---------------------------------------------------------------------------
// Photonic MEMS Accelerometer
// ---------------------------------------------------------------------------

/// Photonic MEMS accelerometer with optical readout.
///
/// A proof mass on a flexural spring is displaced by inertial forces.  An
/// optical transducer (e.g. evanescent waveguide gap or Fabry-Pérot cavity)
/// converts displacement to an optical signal.
#[derive(Debug, Clone)]
pub struct PhotonicAccelerometer {
    /// Proof mass (kg).
    pub test_mass_kg: f64,
    /// Effective spring constant (N/m).
    pub spring_constant: f64,
    /// Mechanical resonant frequency (Hz).
    pub resonant_frequency: f64,
    /// Mechanical quality factor.
    pub q_factor: f64,
    /// Optical readout sensitivity: change in optical power per unit
    /// displacement (W/m).
    pub optical_sensitivity: f64,
    /// Operating wavelength (m).
    pub wavelength: f64,
}

impl PhotonicAccelerometer {
    /// Construct an accelerometer; resonant frequency is computed from
    /// `ω₀ = √(k/m)`.
    ///
    /// # Arguments
    /// * `mass_kg` – Proof mass (kg).
    /// * `spring_n_per_m` – Spring constant (N/m).
    /// * `q` – Mechanical Q-factor.
    /// * `wavelength` – Operating wavelength (m).
    pub fn new(mass_kg: f64, spring_n_per_m: f64, q: f64, wavelength: f64) -> Self {
        let omega0 = (spring_n_per_m / mass_kg).sqrt();
        let f0 = omega0 / (2.0 * PI);
        Self {
            test_mass_kg: mass_kg,
            spring_constant: spring_n_per_m,
            resonant_frequency: f0,
            q_factor: q,
            optical_sensitivity: 1e-3,
            wavelength,
        }
    }

    /// Proof-mass displacement for a quasi-static acceleration.
    ///
    /// `x = m · a / k`
    ///
    /// # Arguments
    /// * `accel_g` – Input acceleration in units of standard gravity *g*.
    ///
    /// # Returns
    /// Proof-mass displacement (m).
    pub fn displacement_from_acceleration(&self, accel_g: f64) -> f64 {
        let a_m_s2 = accel_g * G_ACCEL;
        self.test_mass_kg * a_m_s2 / self.spring_constant
    }

    /// Thermal (Brownian) displacement noise floor.
    ///
    /// `Sx_therm = 4·k_B·T·k / (ω₀⁴·Q·m²)` → expressed as ASD:
    ///
    /// `x_n = √(4·k_B·T / (k·ω₀·Q))` [m/√Hz]
    ///
    /// This is the fluctuation-dissipation-theorem result for a damped
    /// harmonic oscillator far below resonance.
    ///
    /// # Arguments
    /// * `temp_k` – Temperature (K).
    ///
    /// # Returns
    /// Brownian noise ASD (m/√Hz).
    pub fn brownian_noise_floor_m_per_sqrthz(&self, temp_k: f64) -> f64 {
        let omega0 = 2.0 * PI * self.resonant_frequency;
        let psd = 4.0 * K_B * temp_k / (self.spring_constant * omega0 * self.q_factor);
        psd.sqrt()
    }

    /// Optical readout displacement noise.
    ///
    /// `x_opt = NEP / S_opt`
    ///
    /// where `S_opt` is the optical sensitivity (W/m) and NEP is converted to
    /// a displacement noise via division by sensitivity.
    ///
    /// # Arguments
    /// * `nep` – Noise-equivalent power of the detector (W/√Hz).
    ///
    /// # Returns
    /// Optical readout noise ASD (m/√Hz).
    pub fn optical_readout_noise_m_per_sqrthz(&self, nep: f64) -> f64 {
        if self.optical_sensitivity <= 0.0 {
            return f64::INFINITY;
        }
        nep / self.optical_sensitivity
    }

    /// Total noise floor as quadrature sum of thermal and optical noise.
    ///
    /// # Arguments
    /// * `temp_k` – Temperature (K).
    /// * `nep` – Detector NEP (W/√Hz).
    ///
    /// # Returns
    /// Total displacement noise ASD (m/√Hz).
    pub fn total_noise_m_per_sqrthz(&self, temp_k: f64, nep: f64) -> f64 {
        let x_th = self.brownian_noise_floor_m_per_sqrthz(temp_k);
        let x_opt = self.optical_readout_noise_m_per_sqrthz(nep);
        (x_th * x_th + x_opt * x_opt).sqrt()
    }

    /// Noise-equivalent acceleration (μg/√Hz).
    ///
    /// `a_n = x_n · k / m` converted to μg.
    ///
    /// # Arguments
    /// * `temp_k` – Temperature (K).
    /// * `nep` – Detector NEP (W/√Hz).
    ///
    /// # Returns
    /// Noise-equivalent acceleration (μg/√Hz).
    pub fn noise_equivalent_acceleration_ug(&self, temp_k: f64, nep: f64) -> f64 {
        let x_n = self.total_noise_m_per_sqrthz(temp_k, nep);
        let a_n_m_s2 = x_n * self.spring_constant / self.test_mass_kg;
        a_n_m_s2 / G_ACCEL * 1e6 // convert to μg/√Hz
    }

    /// Dynamic range (dB) between peak acceleration and noise floor.
    ///
    /// `DR = 20 · log₁₀(a_max / a_noise)`
    ///
    /// # Arguments
    /// * `max_accel_g` – Full-scale input acceleration (*g*).
    /// * `temp_k` – Temperature (K).
    /// * `nep` – Detector NEP (W/√Hz).
    ///
    /// # Returns
    /// Dynamic range (dB).
    pub fn dynamic_range_db(&self, max_accel_g: f64, temp_k: f64, nep: f64) -> f64 {
        let noise_ug = self.noise_equivalent_acceleration_ug(temp_k, nep);
        if noise_ug <= 0.0 {
            return f64::INFINITY;
        }
        let max_ug = max_accel_g * 1e6;
        20.0 * (max_ug / noise_ug).log10()
    }
}

// ---------------------------------------------------------------------------
// Photonic Pressure Sensor
// ---------------------------------------------------------------------------

/// Photonic pressure sensor based on a Fabry-Pérot microcavity formed
/// between a fixed mirror and a pressure-deflecting membrane.
///
/// Pressure deflects the membrane, changing the cavity length and hence the
/// resonance wavelength.
#[derive(Debug, Clone)]
pub struct PhotonicPressureSensor {
    /// Membrane radius (m).
    pub membrane_radius: f64,
    /// Membrane thickness (m).
    pub membrane_thickness: f64,
    /// Young's modulus of the membrane material (Pa).
    pub youngs_modulus: f64,
    /// Nominal Fabry-Pérot cavity length (m).
    pub cavity_length: f64,
    /// Fabry-Pérot finesse.
    pub finesse: f64,
    /// Operating wavelength (m).
    pub wavelength: f64,
}

impl PhotonicPressureSensor {
    /// Construct a silicon membrane pressure sensor.
    ///
    /// Silicon: E = 130 GPa, ν ≈ 0.28.
    ///
    /// # Arguments
    /// * `radius` – Membrane radius (m).
    /// * `thickness` – Membrane thickness (m).
    /// * `cavity_length` – F-P cavity gap (m).
    /// * `finesse` – Cavity finesse.
    /// * `wavelength` – Operating wavelength (m).
    pub fn new_silicon(
        radius: f64,
        thickness: f64,
        cavity_length: f64,
        finesse: f64,
        wavelength: f64,
    ) -> Self {
        Self {
            membrane_radius: radius,
            membrane_thickness: thickness,
            youngs_modulus: 130e9,
            cavity_length,
            finesse,
            wavelength,
        }
    }

    /// Centre deflection of a clamped circular plate under uniform pressure.
    ///
    /// `δ = P · R⁴ · 3 · (1 – ν²) / (16 · E · h³)`
    ///
    /// Silicon Poisson ratio ν ≈ 0.28.
    ///
    /// # Arguments
    /// * `pressure_pa` – Applied differential pressure (Pa).
    /// * `poissons_ratio` – Membrane material Poisson ratio.
    ///
    /// # Returns
    /// Centre deflection (m); positive for pressure pushing into the cavity.
    pub fn deflection_m(&self, pressure_pa: f64, poissons_ratio: f64) -> f64 {
        let r4 = self.membrane_radius.powi(4);
        let h3 = self.membrane_thickness.powi(3);
        3.0 * pressure_pa * r4 * (1.0 - poissons_ratio * poissons_ratio)
            / (16.0 * self.youngs_modulus * h3)
    }

    /// Cavity resonance wavelength shift from membrane deflection.
    ///
    /// For a Fabry-Pérot cavity the FSR is `Δλ_FSR = λ² / (2·L)` and the
    /// wavelength shift per unit length change is `dλ/dL = λ/L`:
    ///
    /// `Δλ = 2 · δ · λ / (2 · L) = δ · λ / L`
    ///
    /// expressed in pm.
    ///
    /// # Arguments
    /// * `pressure_pa` – Applied pressure (Pa).
    ///
    /// # Returns
    /// Resonance wavelength shift (pm).
    pub fn wavelength_shift_pm(&self, pressure_pa: f64) -> f64 {
        let nu = 0.28_f64; // silicon
        let delta = self.deflection_m(pressure_pa, nu);
        // Δλ = δ · λ / L
        delta * self.wavelength / self.cavity_length * 1e12
    }

    /// Pressure sensitivity (pm/Pa).
    ///
    /// Numerical derivative at 1 Pa applied pressure.
    pub fn sensitivity_pm_per_pa(&self) -> f64 {
        self.wavelength_shift_pm(1.0)
    }

    /// Minimum detectable pressure (photon-noise limited).
    ///
    /// `P_min = noise_pm / sensitivity`
    ///
    /// # Arguments
    /// * `noise_pm` – Wavelength tracking noise (pm RMS).
    ///
    /// # Returns
    /// Minimum detectable pressure change (Pa).
    pub fn detection_limit_pa(&self, noise_pm: f64) -> f64 {
        let s = self.sensitivity_pm_per_pa();
        if s.abs() < f64::EPSILON {
            return f64::INFINITY;
        }
        noise_pm / s
    }

    /// Fundamental resonant frequency of the clamped circular membrane.
    ///
    /// `f₀ = 10.22 / (2π · R²) · √(E·h² / (12·ρ·(1–ν²)))` (first mode)
    ///
    /// # Arguments
    /// * `density` – Membrane material density (kg/m³).
    ///
    /// # Returns
    /// Resonant frequency (Hz).
    pub fn resonant_frequency_hz(&self, density: f64) -> f64 {
        let nu = 0.28_f64;
        if density <= 0.0 || self.membrane_radius <= 0.0 {
            return 0.0;
        }
        let h = self.membrane_thickness;
        let r = self.membrane_radius;
        // Exact first-mode coefficient κ₁ = 10.22 for clamped circular plate
        let kappa = 10.22_f64;
        kappa / (2.0 * PI * r * r)
            * (self.youngs_modulus * h * h / (12.0 * density * (1.0 - nu * nu))).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Integrated Optical Strain Gauge
// ---------------------------------------------------------------------------

/// Integrated waveguide strain gauge based on a Bragg-like resonance shift.
///
/// Mechanical strain changes the effective optical path length via the
/// elasto-optic effect.  The resonance wavelength shift is:
///
/// `Δλ/λ = (1 – p_e) · ε`
///
/// where `p_e` is the effective strain-optic coefficient (≈ 0.22 for silica,
/// ≈ 0.18 for silicon nitride).
#[derive(Debug, Clone)]
pub struct IntegratedStrainGauge {
    /// Sensing gauge length (mm).
    pub gauge_length_mm: f64,
    /// Effective photo-elastic coefficient (`p_e`, dimensionless).
    pub waveguide_strain_optic_coeff: f64,
    /// Operating wavelength (m).
    pub wavelength: f64,
    /// Nominal Bragg period (nm).
    pub baseline_period_nm: f64,
}

impl IntegratedStrainGauge {
    /// Construct a silicon nitride waveguide strain gauge.
    ///
    /// Silicon nitride parameters: `p_e ≈ 0.18`, Bragg period derived from
    /// `λ = 2 · n_eff · Λ` with `n_eff = 1.98`.
    ///
    /// # Arguments
    /// * `gauge_length_mm` – Gauge length (mm).
    /// * `wavelength` – Operating wavelength (m).
    pub fn new_silicon_nitride(gauge_length_mm: f64, wavelength: f64) -> Self {
        let n_eff = 1.98_f64;
        let period_nm = wavelength * 1e9 / (2.0 * n_eff);
        Self {
            gauge_length_mm,
            waveguide_strain_optic_coeff: 0.18,
            wavelength,
            baseline_period_nm: period_nm,
        }
    }

    /// Wavelength shift for an applied strain.
    ///
    /// `Δλ = λ · (1 – p_e) · ε`
    ///
    /// # Arguments
    /// * `strain_microstrain` – Applied strain (με = 10⁻⁶ m/m).
    ///
    /// # Returns
    /// Wavelength shift (pm).
    pub fn wavelength_shift_pm(&self, strain_microstrain: f64) -> f64 {
        let strain = strain_microstrain * 1e-6;
        self.wavelength * 1e12 * (1.0 - self.waveguide_strain_optic_coeff) * strain
    }

    /// Recover strain from a measured wavelength shift.
    ///
    /// # Arguments
    /// * `shift_pm` – Measured wavelength shift (pm).
    ///
    /// # Returns
    /// Strain (με).
    pub fn strain_from_shift_microstrain(&self, shift_pm: f64) -> f64 {
        let sensitivity = self.wavelength * 1e12 * (1.0 - self.waveguide_strain_optic_coeff);
        if sensitivity.abs() < f64::EPSILON {
            return 0.0;
        }
        shift_pm / sensitivity * 1e6
    }

    /// Strain resolution limited by wavelength noise.
    ///
    /// `δε = δλ / (λ · (1 – p_e))` expressed in nε/√Hz.
    ///
    /// # Arguments
    /// * `wavelength_noise_pm` – Wavelength measurement noise (pm/√Hz).
    ///
    /// # Returns
    /// Strain resolution (nε/√Hz).
    pub fn resolution_nstrain(&self, wavelength_noise_pm: f64) -> f64 {
        let sensitivity = self.wavelength * 1e12 * (1.0 - self.waveguide_strain_optic_coeff);
        if sensitivity.abs() < f64::EPSILON {
            return f64::INFINITY;
        }
        // δε in micro-strain, converted to nano-strain
        (wavelength_noise_pm / sensitivity) * 1e3
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // --- Accelerometer ---

    #[test]
    fn accelerometer_displacement_from_1g() {
        // 1 μg mass, 100 N/m spring
        let acc = PhotonicAccelerometer::new(1e-6, 100.0, 1000.0, 1550e-9);
        let x = acc.displacement_from_acceleration(1.0); // 1g
        let expected = 1e-6 * G_ACCEL / 100.0;
        assert_abs_diff_eq!(x, expected, epsilon = 1e-20);
    }

    #[test]
    fn accelerometer_brownian_noise_positive() {
        let acc = PhotonicAccelerometer::new(1e-6, 100.0, 1000.0, 1550e-9);
        let xn = acc.brownian_noise_floor_m_per_sqrthz(300.0);
        assert!(xn > 0.0, "Thermal noise should be positive");
    }

    #[test]
    fn accelerometer_dynamic_range_positive() {
        let acc = PhotonicAccelerometer::new(1e-6, 100.0, 1000.0, 1550e-9);
        let dr = acc.dynamic_range_db(100.0, 300.0, 1e-12);
        assert!(
            dr > 0.0 && dr < 300.0,
            "Dynamic range out of range: {} dB",
            dr
        );
    }

    #[test]
    fn accelerometer_nea_improves_with_lower_nep() {
        let acc = PhotonicAccelerometer::new(1e-6, 100.0, 1000.0, 1550e-9);
        let nea_hi = acc.noise_equivalent_acceleration_ug(300.0, 1e-12);
        let nea_lo = acc.noise_equivalent_acceleration_ug(300.0, 1e-15);
        assert!(nea_lo <= nea_hi, "Lower NEP should give better sensitivity");
    }

    // --- Pressure Sensor ---

    #[test]
    fn pressure_deflection_scales_with_radius() {
        // Larger radius → more deflection for same pressure
        let s1 = PhotonicPressureSensor::new_silicon(100e-6, 2e-6, 5e-6, 100.0, 1550e-9);
        let s2 = PhotonicPressureSensor::new_silicon(200e-6, 2e-6, 5e-6, 100.0, 1550e-9);
        let d1 = s1.deflection_m(1000.0, 0.28);
        let d2 = s2.deflection_m(1000.0, 0.28);
        // R scales as R⁴ → 2× radius → 16× deflection
        assert_abs_diff_eq!(d2 / d1, 16.0, epsilon = 0.1);
    }

    #[test]
    fn pressure_wavelength_shift_positive() {
        let s = PhotonicPressureSensor::new_silicon(200e-6, 2e-6, 10e-6, 50.0, 1550e-9);
        let shift = s.wavelength_shift_pm(1000.0);
        assert!(
            shift > 0.0,
            "Wavelength shift should be positive for positive pressure"
        );
    }

    #[test]
    fn pressure_resonant_frequency_silicon() {
        // Silicon density ~2329 kg/m³
        let s = PhotonicPressureSensor::new_silicon(100e-6, 2e-6, 5e-6, 100.0, 1550e-9);
        let f0 = s.resonant_frequency_hz(2329.0);
        // Typical Si membrane: hundreds of kHz to few MHz
        assert!(
            f0 > 1e4 && f0 < 1e8,
            "Resonant frequency out of expected range: {} Hz",
            f0
        );
    }

    // --- Strain Gauge ---

    #[test]
    fn strain_gauge_shift_roundtrip() {
        let sg = IntegratedStrainGauge::new_silicon_nitride(10.0, 1550e-9);
        let strain = 100.0_f64; // 100 με
        let shift = sg.wavelength_shift_pm(strain);
        let recovered = sg.strain_from_shift_microstrain(shift);
        assert_abs_diff_eq!(recovered, strain, epsilon = 1e-6);
    }

    #[test]
    fn strain_gauge_resolution_decreases_with_less_noise() {
        let sg = IntegratedStrainGauge::new_silicon_nitride(10.0, 1550e-9);
        let r1 = sg.resolution_nstrain(1.0);
        let r2 = sg.resolution_nstrain(0.01);
        assert!(r2 < r1, "Finer wavelength noise → better strain resolution");
        assert_abs_diff_eq!(r1 / r2, 100.0, epsilon = 1.0);
    }

    #[test]
    fn strain_gauge_period_reasonable() {
        let sg = IntegratedStrainGauge::new_silicon_nitride(10.0, 1550e-9);
        // For n_eff=1.98, λ=1550nm: Λ = 1550/(2*1.98) ≈ 391 nm
        assert_abs_diff_eq!(sg.baseline_period_nm, 391.4, epsilon = 1.0);
    }
}
