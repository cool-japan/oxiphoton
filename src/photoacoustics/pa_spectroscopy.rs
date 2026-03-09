//! Photoacoustic spectroscopy and gas sensing
//!
//! Implements:
//! - Resonant PA cell design (longitudinal and radial acoustic modes)
//! - Minimum detectable absorption and NNEA figures of merit
//! - Gas concentration sensing via PA signal modelling
//! - Common target gas reference data (CH₄, CO₂, NO₂)

use std::f64::consts::PI;

/// Boltzmann constant (J/K)
const KB: f64 = 1.380649e-23;

/// Avogadro's number (mol⁻¹)
#[allow(dead_code)]
const NA: f64 = 6.02214076e23;

/// Speed of light in vacuum (m/s)
#[allow(dead_code)]
const C_LIGHT: f64 = 2.99792458e8;

/// Resonant photoacoustic cell for gas absorption sensing.
///
/// A resonant PA cell is an acoustic cavity tuned so that the modulated laser
/// excites a standing acoustic wave, which amplifies the PA signal by the
/// acoustic Q factor. Common designs: longitudinal (H-type), radial, or
/// azimuthal resonance.
#[derive(Debug, Clone)]
pub struct PaCell {
    /// Cavity length (m)
    pub length_m: f64,
    /// Cell inner radius (m)
    pub radius_m: f64,
    /// Speed of sound in the fill gas (m/s)
    pub c_sound: f64,
    /// Acoustic quality factor Q (dimensionless)
    pub q_factor: f64,
}

impl PaCell {
    /// Fundamental longitudinal resonance frequency (Hz).
    ///
    /// f₀ = c_s / (2L)
    ///
    /// This is the lowest mode for a cylindrical cavity open at both ends.
    pub fn fundamental_frequency_hz(&self) -> f64 {
        self.c_sound / (2.0 * self.length_m)
    }

    /// First radial resonance frequency (Hz).
    ///
    /// f_rad = χ₀₁ × c_s / (2π R)  where χ₀₁ ≈ 2.4048 is the first root
    /// of J₀, the zeroth-order Bessel function.
    ///
    /// Approximated as f_rad ≈ 1.22 c_s / (2R) which follows from χ₀₁/(π) ≈ 0.7655,
    /// and 2 × 0.7655 / (2π) × c_s / R = 1.22 × c_s / (2R).
    pub fn radial_resonance_hz(&self) -> f64 {
        // χ₀₁ / (2π) = 2.4048 / (2π) ≈ 0.38274
        // f_rad = 0.38274 × c_s / R = (0.38274 × 2) × c_s / (2R) ≈ 0.765 × c_s / R
        // Equivalently: 1.22 × c_s / (2R)
        1.22 * self.c_sound / (2.0 * self.radius_m)
    }

    /// Signal enhancement factor at resonance.
    ///
    /// At the resonance frequency the acoustic pressure amplitude is amplified
    /// by the Q factor relative to the non-resonant case: S_res = Q × S_non_res.
    pub fn signal_enhancement(&self) -> f64 {
        self.q_factor
    }

    /// Minimum detectable absorption coefficient (m⁻¹).
    ///
    /// Derived from the noise equivalent pressure of the microphone and the
    /// PA signal sensitivity:
    ///
    ///   μ_a_min = NEP_mic / (Γ × Q × F × V_cell)
    ///
    /// where Γ ≈ 0.15 (Grueneisen parameter of gas), F = P × τ / A is fluence,
    /// and V_cell = π R² L.
    ///
    /// # Arguments
    /// * `power_w`        — laser power (W)
    /// * `nep_pa_hz`      — microphone noise equivalent pressure density (Pa/√Hz)
    pub fn min_detectable_absorption(&self, power_w: f64, nep_pa_hz: f64) -> f64 {
        // Typical Grueneisen parameter for gas (approximated)
        let gamma_gas = 0.40; // for air/N₂-like gas
        let v_cell = PI * self.radius_m * self.radius_m * self.length_m;
        let denom = gamma_gas * self.q_factor * power_w * v_cell;
        if denom.abs() < 1.0e-30 {
            return f64::INFINITY;
        }
        nep_pa_hz / denom
    }

    /// Normalised noise equivalent absorption (NNEA) — figure of merit for PA cells.
    ///
    /// NNEA = noise_amplitude / (power × sensitivity) [W cm⁻¹ Hz^{−1/2}]
    ///
    /// A smaller NNEA indicates a more sensitive system.
    ///
    /// # Arguments
    /// * `noise_pa_hz`  — microphone noise spectral density (Pa/√Hz)
    /// * `power_w`      — laser power at sample (W)
    pub fn nnea(&self, noise_pa_hz: f64, power_w: f64) -> f64 {
        if power_w <= 0.0 {
            return f64::INFINITY;
        }
        // NNEA = (noise_pa / √Hz) / (sensitivity × W)
        // Sensitivity here is Q (dimensionless enhancement), so NNEA ∝ noise/P/Q
        // Units normalised to cm⁻¹ W Hz^{-1/2} by including cell volume factor
        let v_cell_cm3 = PI * (self.radius_m * 100.0).powi(2) * (self.length_m * 100.0);
        noise_pa_hz / (power_w * self.q_factor * v_cell_cm3.max(1.0e-20))
    }
}

/// Gas concentration sensor using photoacoustic spectroscopy.
///
/// The laser wavelength is tuned to a gas absorption line, and the
/// PA signal amplitude is proportional to the gas concentration.
///
/// # Physics
/// The number density of target molecules at partial pressure p_gas:
///   N_gas = p_gas / (k_B T)   [molecules/m³]
///
/// Absorption coefficient:  μ_a = N_gas × σ_abs
///
/// PA signal:  S = Γ × μ_a × F × Q_cell
///           = Γ × C_ppm × 1e-6 × (p_total / k_B T) × σ_abs × F × Q_cell
#[derive(Debug, Clone)]
pub struct PaGasSensor {
    /// Laser wavelength (nm)
    pub laser_wavelength_nm: f64,
    /// Absorption cross section σ_abs of target molecule (cm²)
    pub absorption_cross_section_cm2: f64,
    /// Resonant PA cell
    pub cell: PaCell,
    /// Laser power at sample (W)
    pub power_w: f64,
}

impl PaGasSensor {
    /// PA signal amplitude per ppm of target gas (Pa/ppm).
    ///
    /// S_ppm = Γ × (1e-6 × p / k_B T) × σ_abs_m2 × F × Q
    ///
    /// # Arguments
    /// * `temperature_k`  — gas temperature (K)
    /// * `pressure_pa`    — total gas pressure (Pa)
    pub fn signal_per_ppm(&self, temperature_k: f64, pressure_pa: f64) -> f64 {
        // Number density at 1 ppm of target gas: N = 1e-6 × p / (k_B T)
        let n_per_m3_per_ppm = 1.0e-6 * pressure_pa / (KB * temperature_k.max(1.0));

        // Convert σ from cm² to m²
        let sigma_m2 = self.absorption_cross_section_cm2 * 1.0e-4;

        // Absorption coefficient contribution per ppm:
        let mu_a_per_ppm = n_per_m3_per_ppm * sigma_m2;

        // Grueneisen parameter for gas (γ−1 ≈ 0.4 for diatomic gas)
        let gamma_gas = 0.40;

        // Fluence = P × τ / A_beam ≈ P / (beam area); for spectroscopy, use power density
        // In PA spectroscopy, the signal scales linearly with power: S ∝ Γ μ_a P Q
        gamma_gas * mu_a_per_ppm * self.power_w * self.cell.q_factor
    }

    /// 1-σ detection limit in ppb.
    ///
    /// C_min (ppb) = NEP_pa / (dS/dC × 1e3)
    ///
    /// The 1e3 factor converts ppm → ppb sensitivity.
    pub fn detection_limit_ppb(&self, nep_pa: f64, temperature_k: f64, pressure_pa: f64) -> f64 {
        let sens_per_ppm = self.signal_per_ppm(temperature_k, pressure_pa);
        if sens_per_ppm <= 0.0 {
            return f64::INFINITY;
        }
        // sens_per_ppm is in Pa/ppm; nep in Pa; limit = nep/sens in ppm × 1000 = ppb
        nep_pa / sens_per_ppm * 1000.0
    }

    /// Reference data for methane (CH₄) at the 3311 nm R-branch absorption line.
    ///
    /// Returns `(σ_abs [cm²], λ [nm])`.
    /// σ ≈ 2.0×10⁻¹⁹ cm² (HITRAN-derived, air-broadened, 296 K)
    pub fn methane_at_3311nm() -> (f64, f64) {
        (2.0e-19, 3311.0)
    }

    /// Reference data for CO₂ at the 4260 nm P-branch absorption line.
    ///
    /// Returns `(σ_abs [cm²], λ [nm])`.
    /// σ ≈ 1.6×10⁻¹⁸ cm² (HITRAN-derived)
    pub fn co2_at_4260nm() -> (f64, f64) {
        (1.6e-18, 4260.0)
    }

    /// Reference data for NO₂ at the 450 nm visible absorption band.
    ///
    /// Returns `(σ_abs [cm²], λ [nm])`.
    /// σ ≈ 6.0×10⁻¹⁹ cm² (UV/vis cross-section, Vandaele et al.)
    pub fn no2_at_450nm() -> (f64, f64) {
        (6.0e-19, 450.0)
    }

    /// Build a sensor pre-configured for methane detection at 3311 nm.
    pub fn for_methane(cell: PaCell, power_w: f64) -> Self {
        let (sigma, lambda) = Self::methane_at_3311nm();
        Self {
            laser_wavelength_nm: lambda,
            absorption_cross_section_cm2: sigma,
            cell,
            power_w,
        }
    }

    /// Build a sensor pre-configured for CO₂ detection at 4260 nm.
    pub fn for_co2(cell: PaCell, power_w: f64) -> Self {
        let (sigma, lambda) = Self::co2_at_4260nm();
        Self {
            laser_wavelength_nm: lambda,
            absorption_cross_section_cm2: sigma,
            cell,
            power_w,
        }
    }

    /// Absorption coefficient at a given gas concentration (m⁻¹).
    ///
    /// μ_a = C_ppm × 1e-6 × (p / k_B T) × σ_m2
    pub fn absorption_coeff_per_m(
        &self,
        concentration_ppm: f64,
        temperature_k: f64,
        pressure_pa: f64,
    ) -> f64 {
        let n_total = pressure_pa / (KB * temperature_k.max(1.0)); // total molecules/m³
        let n_gas = concentration_ppm * 1.0e-6 * n_total;
        let sigma_m2 = self.absorption_cross_section_cm2 * 1.0e-4;
        n_gas * sigma_m2
    }

    /// Estimated PA signal (Pa) for a known gas concentration.
    ///
    /// S = Γ × μ_a × P × Q
    pub fn signal_pa(&self, concentration_ppm: f64, temperature_k: f64, pressure_pa: f64) -> f64 {
        let gamma_gas = 0.40;
        let mu_a = self.absorption_coeff_per_m(concentration_ppm, temperature_k, pressure_pa);
        gamma_gas * mu_a * self.power_w * self.cell.q_factor
    }
}

/// Beer-Lambert gas absorptance over a path length L.
///
/// A = 1 − exp(−μ_a L)
pub fn beer_lambert_absorptance(mu_a_per_m: f64, path_length_m: f64) -> f64 {
    1.0 - (-mu_a_per_m * path_length_m).exp()
}

/// Convert absorption cross section from cm² to m².
pub fn cross_section_cm2_to_m2(sigma_cm2: f64) -> f64 {
    sigma_cm2 * 1.0e-4
}

/// Number density of ideal gas (molecules/m³).
///
/// N = p / (k_B T)
pub fn ideal_gas_number_density(pressure_pa: f64, temperature_k: f64) -> f64 {
    pressure_pa / (KB * temperature_k.max(1.0))
}

/// Mole fraction to partial pressure (Pa).
pub fn mole_fraction_to_partial_pressure(mole_fraction: f64, total_pressure_pa: f64) -> f64 {
    mole_fraction * total_pressure_pa
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cell() -> PaCell {
        PaCell {
            length_m: 0.15,  // 15 cm
            radius_m: 0.005, // 5 mm
            c_sound: 343.0,  // air at ~20 °C
            q_factor: 50.0,
        }
    }

    #[test]
    fn pa_cell_longitudinal_resonance() {
        let cell = test_cell();
        let f0 = cell.fundamental_frequency_hz();
        // f₀ = 343/(2×0.15) = 1143 Hz
        let expected = 343.0 / (2.0 * 0.15);
        assert!(
            (f0 - expected).abs() < 1.0e-9,
            "f0={}Hz expected={}Hz",
            f0,
            expected
        );
        assert!(
            f0 > 1000.0 && f0 < 2000.0,
            "Longitudinal resonance out of range: {}",
            f0
        );
    }

    #[test]
    fn pa_cell_radial_resonance() {
        let cell = test_cell();
        let f_rad = cell.radial_resonance_hz();
        // f_rad = 1.22 × 343 / (2 × 0.005) = 41.846 kHz
        let expected = 1.22 * 343.0 / (2.0 * 0.005);
        assert!((f_rad - expected).abs() < 1.0e-9, "f_rad={}", f_rad);
        assert!(f_rad > 10.0e3 && f_rad < 100.0e3, "f_rad={}", f_rad);
    }

    #[test]
    fn signal_enhancement_equals_q() {
        let cell = test_cell();
        assert!((cell.signal_enhancement() - cell.q_factor).abs() < 1.0e-10);
    }

    #[test]
    fn gas_sensor_signal_scales_with_concentration() {
        let sensor = PaGasSensor::for_methane(test_cell(), 0.01);
        let s1 = sensor.signal_pa(1.0, 296.0, 101325.0);
        let s10 = sensor.signal_pa(10.0, 296.0, 101325.0);
        // Signal should scale linearly with concentration
        assert!(s10 > 0.0, "Signal must be positive");
        assert!(
            (s10 / s1 - 10.0).abs() < 1.0e-8,
            "Signal should be linear in concentration"
        );
    }

    #[test]
    fn detection_limit_ppb_methane() {
        let sensor = PaGasSensor::for_methane(test_cell(), 0.01);
        // NEP = 1 µPa/√Hz at 1143 Hz (typical for sensitive microphone)
        let limit = sensor.detection_limit_ppb(1.0e-6, 296.0, 101325.0);
        assert!(
            limit > 0.0 && limit.is_finite(),
            "Detection limit must be positive finite"
        );
        // Typical PA gas sensors achieve sub-ppm to ppb levels
        // With these parameters, limit should be in range 0.001–1000 ppb
        assert!(
            limit < 1.0e9,
            "Detection limit unreasonably large: {}ppb",
            limit
        );
    }

    #[test]
    fn beer_lambert_absorptance_small() {
        // For μ_a L ≪ 1: A ≈ μ_a L
        let mu_a = 0.001; // m⁻¹
        let l = 0.1; // 10 cm
        let a = beer_lambert_absorptance(mu_a, l);
        let approx = mu_a * l;
        assert!((a - approx).abs() < 1.0e-6, "A={}≈{}", a, approx);
    }

    #[test]
    fn ideal_gas_density_stp() {
        // At STP (273.15 K, 101325 Pa): N ≈ 2.687×10²⁵ m⁻³ (Loschmidt's number)
        let n = ideal_gas_number_density(101325.0, 273.15);
        let loschmidt = 2.6867774e25;
        let rel_err = (n - loschmidt).abs() / loschmidt;
        assert!(rel_err < 0.01, "N={:.3e} expected {:.3e}", n, loschmidt);
    }

    #[test]
    fn gas_references_positive_sigma() {
        let (sigma_ch4, lambda_ch4) = PaGasSensor::methane_at_3311nm();
        assert!(sigma_ch4 > 0.0 && lambda_ch4 > 0.0);
        let (sigma_co2, lambda_co2) = PaGasSensor::co2_at_4260nm();
        assert!(sigma_co2 > 0.0 && lambda_co2 > 0.0);
        let (sigma_no2, lambda_no2) = PaGasSensor::no2_at_450nm();
        assert!(sigma_no2 > 0.0 && lambda_no2 > 0.0);
        // Wavelength ordering: NO₂ < CH₄ < CO₂
        assert!(lambda_no2 < lambda_ch4 && lambda_ch4 < lambda_co2);
    }

    #[test]
    fn signal_per_ppm_positive() {
        let sensor = PaGasSensor::for_co2(test_cell(), 0.1);
        let s = sensor.signal_per_ppm(296.0, 101325.0);
        assert!(s > 0.0 && s.is_finite(), "signal_per_ppm={}", s);
    }
}
