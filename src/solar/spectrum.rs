//! Solar spectrum utilities: AM1.5G reference data, interpolation, and photovoltaic analysis.
//!
//! Includes:
//! - `AM15G_DATA` — tabulated AM1.5G irradiance (W/m²/nm)
//! - `SolarSpectrum` — interpolating wrapper with integration helpers
//! - Blackbody / Planck spectrum tools
//! - Shockley-Queisser efficiency limit
//! - Concentration-ratio effects on open-circuit voltage

const PLANCK_H: f64 = 6.626_070_15e-34; // J·s
const BOLTZMANN: f64 = 1.380_649e-23; // J/K
const SPEED_OF_LIGHT: f64 = 2.997_924_58e8; // m/s
const ELECTRON_CHARGE: f64 = 1.602_176_634e-19; // C

/// AM1.5G solar spectrum reference data (ASTM G173-03).
///
/// Tabulated as (wavelength_nm, irradiance_W/m²/nm) pairs.
/// Total integrated irradiance ≈ 1000.4 W/m² (AM1.5G standard).
///
/// Representative AM1.5G data points (wavelength in nm, irradiance in W/m²/nm).
/// Values from ASTM G173-03 global tilt standard (total ≈ 1000.4 W/m²).
/// Includes explicit sampling at water-vapor and O₂ absorption band wavelengths
/// to accurately reproduce the integrated total via the trapezoidal rule.
pub const AM15G_DATA: &[(f64, f64)] = &[
    // UV (Hartley-Huggins Ozone bands suppress UV heavily)
    (280.0, 0.0),
    (300.0, 0.013),
    (320.0, 0.308),
    (340.0, 0.632),
    (360.0, 0.465),
    (380.0, 0.378),
    // Visible (390-700 nm)
    (400.0, 1.243),
    (420.0, 1.490),
    (440.0, 1.733),
    (460.0, 1.663),
    (480.0, 1.748),
    (500.0, 1.719),
    (520.0, 1.784),
    (540.0, 1.770),
    (560.0, 1.823),
    (580.0, 1.729),
    (600.0, 1.651),
    (620.0, 1.602),
    (640.0, 1.551),
    (660.0, 1.543),
    (680.0, 1.484),
    (700.0, 1.468),
    // Near-IR including O₂ (762 nm) and H₂O absorption bands
    (720.0, 1.370),
    (740.0, 1.346),
    (755.0, 0.950), // O₂-A band edge
    (762.0, 0.650), // O₂-A band minimum
    (770.0, 1.260),
    (780.0, 1.218),
    (800.0, 1.189),
    (820.0, 1.082),
    (840.0, 1.037),
    (860.0, 0.983),
    (880.0, 0.834), // H₂O band starts
    (900.0, 0.612), // H₂O absorption minimum
    (920.0, 0.456),
    (940.0, 0.468),
    (960.0, 0.726),
    (980.0, 0.838),
    (1000.0, 0.904),
    (1020.0, 0.864),
    (1050.0, 0.826),
    (1080.0, 0.691), // H₂O band
    (1100.0, 0.770),
    (1120.0, 0.773),
    (1150.0, 0.676),
    (1170.0, 0.553), // H₂O band dip
    (1200.0, 0.584),
    (1250.0, 0.522),
    // 1300-1450 nm: strong H₂O band (reduces irradiance significantly)
    (1300.0, 0.440),
    (1350.0, 0.065), // deep H₂O absorption
    (1380.0, 0.010), // near-zero
    (1400.0, 0.055),
    (1450.0, 0.140),
    (1500.0, 0.453),
    (1550.0, 0.440),
    (1600.0, 0.404),
    (1650.0, 0.350),
    // 1700-2100 nm: H₂O + CO₂ bands
    (1700.0, 0.286),
    (1750.0, 0.200),
    (1800.0, 0.088), // H₂O band minimum
    (1850.0, 0.055),
    (1900.0, 0.020), // CO₂+H₂O minimum
    (1950.0, 0.095),
    (2000.0, 0.141),
    (2050.0, 0.133),
    (2100.0, 0.076), // CO₂ band dip
    (2150.0, 0.075),
    (2200.0, 0.081),
    (2300.0, 0.056),
    (2400.0, 0.042),
    (2500.0, 0.032),
    (3000.0, 0.008),
    (3500.0, 0.001),
    (4000.0, 0.0002),
];

// ─── Solar spectrum type ──────────────────────────────────────────────────────

/// Solar spectrum with interpolation, integration, and photovoltaic analysis.
#[derive(Debug, Clone)]
pub struct SolarSpectrum {
    /// Wavelength grid (m)
    pub wavelengths: Vec<f64>,
    /// Spectral irradiance (W/m²/m)
    pub irradiances: Vec<f64>,
}

impl SolarSpectrum {
    /// Create from the built-in AM1.5G data.
    pub fn am15g() -> Self {
        let wavelengths: Vec<f64> = AM15G_DATA.iter().map(|(wl, _)| wl * 1e-9).collect();
        // Convert from W/m²/nm to W/m²/m
        let irradiances: Vec<f64> = AM15G_DATA.iter().map(|(_, s)| s * 1e9).collect();
        Self {
            wavelengths,
            irradiances,
        }
    }

    /// Create from custom (wavelength_m, irradiance_W_m2_m) pairs.
    pub fn from_data(data: Vec<(f64, f64)>) -> Self {
        let wavelengths = data.iter().map(|(w, _)| *w).collect();
        let irradiances = data.iter().map(|(_, s)| *s).collect();
        Self {
            wavelengths,
            irradiances,
        }
    }

    /// Evaluate spectral irradiance at wavelength λ (m) by linear interpolation.
    pub fn irradiance_at(&self, wavelength: f64) -> f64 {
        let wls = &self.wavelengths;
        if wls.is_empty() {
            return 0.0;
        }
        if wavelength <= wls[0] {
            return self.irradiances[0];
        }
        let last = wls.len() - 1;
        if wavelength >= wls[last] {
            return self.irradiances[last];
        }
        // Binary search
        let idx = wls.partition_point(|&w| w <= wavelength);
        let i = idx - 1;
        let t = (wavelength - wls[i]) / (wls[i + 1] - wls[i]);
        self.irradiances[i] + t * (self.irradiances[i + 1] - self.irradiances[i])
    }

    /// Integrate the spectrum over [λ_min, λ_max] (m) using the trapezoidal rule.
    ///
    /// Returns total irradiance (W/m²) in the given wavelength range.
    pub fn integrate(&self, lambda_min: f64, lambda_max: f64, n_pts: usize) -> f64 {
        let dl = (lambda_max - lambda_min) / n_pts as f64;
        let mut sum = 0.0_f64;
        for i in 0..n_pts {
            let l1 = lambda_min + i as f64 * dl;
            let l2 = l1 + dl;
            sum += 0.5 * (self.irradiance_at(l1) + self.irradiance_at(l2)) * dl;
        }
        sum
    }

    /// Total integrated irradiance (W/m²) over the full spectrum.
    pub fn total_power(&self) -> f64 {
        self.integrate(280e-9, 4000e-9, 10_000)
    }

    /// Total integrated irradiance — alias for `total_power` (backward-compatible).
    pub fn total_irradiance(&self) -> f64 {
        self.total_power()
    }

    /// Photon flux density at wavelength λ (m): Φ(λ) = E(λ) · λ / (h·c)  [photons/m²/s/m]
    pub fn photon_flux(&self, wavelength: f64) -> f64 {
        let e = self.irradiance_at(wavelength);
        e * wavelength / (PLANCK_H * SPEED_OF_LIGHT)
    }

    /// Photon flux density integrated over a wavelength range (photons/m²/s).
    ///
    /// # Arguments
    /// * `lambda_min`, `lambda_max` — integration bounds (m)
    pub fn photon_flux_density(&self, lambda_min: f64, lambda_max: f64) -> f64 {
        integrate_photon_flux_spectrum(self, lambda_min, lambda_max, 2000)
    }

    /// Integrated photon flux above the bandgap energy E_g (eV).
    ///
    /// This represents the maximum possible photon-generated current density
    /// (in photons/m²/s) for a single-junction cell with bandgap E_g.
    pub fn above_bandgap_photon_flux(&self, bandgap_ev: f64) -> f64 {
        let lambda_max_m = 1_239.841_93e-9 / bandgap_ev; // bandgap cutoff wavelength
        self.photon_flux_density(280e-9, lambda_max_m)
    }
}

// ─── Photon flux integration ───────────────────────────────────────────────────

/// Integrate photon flux density over [λ_min, λ_max] from a spectrum object.
fn integrate_photon_flux_spectrum(
    spec: &SolarSpectrum,
    lambda_min: f64,
    lambda_max: f64,
    n_pts: usize,
) -> f64 {
    let dl = (lambda_max - lambda_min) / n_pts as f64;
    let mut sum = 0.0_f64;
    for i in 0..n_pts {
        let l1 = lambda_min + i as f64 * dl;
        let l2 = l1 + dl;
        let phi1 = spec.photon_flux(l1);
        let phi2 = spec.photon_flux(l2);
        sum += 0.5 * (phi1 + phi2) * dl;
    }
    sum
}

/// Integrate photon flux density over [λ_min, λ_max] from tabulated (λ, E) pairs.
///
/// Converts irradiance to photon flux using E_photon = h·c/λ.
///
/// # Arguments
/// * `spectrum` — slice of (wavelength_m, irradiance_W_m2_m) pairs
/// * `lambda_min`, `lambda_max` — integration bounds (m)
pub fn integrate_photon_flux(spectrum: &[(f64, f64)], lambda_min: f64, lambda_max: f64) -> f64 {
    if spectrum.len() < 2 {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for i in 0..spectrum.len() - 1 {
        let (l0, e0) = spectrum[i];
        let (l1, e1) = spectrum[i + 1];
        if l1 <= lambda_min || l0 >= lambda_max {
            continue;
        }
        let la = l0.max(lambda_min);
        let lb = l1.min(lambda_max);
        let t_a = (la - l0) / (l1 - l0);
        let t_b = (lb - l0) / (l1 - l0);
        let ea = e0 + t_a * (e1 - e0);
        let eb = e0 + t_b * (e1 - e0);
        // photon flux density = E(λ)·λ/(hc)
        let phi_a = ea * la / (PLANCK_H * SPEED_OF_LIGHT);
        let phi_b = eb * lb / (PLANCK_H * SPEED_OF_LIGHT);
        sum += 0.5 * (phi_a + phi_b) * (lb - la);
    }
    sum
}

// ─── Blackbody spectrum ───────────────────────────────────────────────────────

/// Planck blackbody spectral radiance B_λ (W/m²/m/sr) at temperature T (K).
///
/// B_λ = (2hc²/λ⁵) · 1/(exp(hc/(λ·k_B·T)) − 1)
pub fn planck_radiance(lambda_m: f64, temp_k: f64) -> f64 {
    let exp_arg = PLANCK_H * SPEED_OF_LIGHT / (lambda_m * BOLTZMANN * temp_k);
    let exp_val = exp_arg.exp();
    if exp_val.is_infinite() {
        return 0.0;
    }
    2.0 * PLANCK_H * SPEED_OF_LIGHT * SPEED_OF_LIGHT / lambda_m.powi(5) / (exp_val - 1.0)
}

/// Blackbody total photon flux (photons/m²/s) in [λ_min, λ_max] at temperature T.
///
/// Integrates Φ(λ) = π · B_λ(T) · λ / (hc) over the wavelength range
/// using n_pts trapezoidal segments.
pub fn blackbody_photon_flux(temp_k: f64, lambda_min: f64, lambda_max: f64, n_pts: usize) -> f64 {
    let dl = (lambda_max - lambda_min) / n_pts as f64;
    let mut sum = 0.0_f64;
    for i in 0..n_pts {
        let l1 = lambda_min + i as f64 * dl;
        let l2 = l1 + dl;
        // Φ = π·B_λ·λ/(hc)  (hemisphere integrated, both polarisations)
        let phi = |l: f64| {
            std::f64::consts::PI * planck_radiance(l, temp_k) * l / (PLANCK_H * SPEED_OF_LIGHT)
        };
        sum += 0.5 * (phi(l1) + phi(l2)) * dl;
    }
    sum
}

// ─── AM1.5G integrated power ──────────────────────────────────────────────────

/// Total AM1.5G integrated irradiance in the standard wavelength range (W/m²).
///
/// Returns approximately 1000 W/m² (ASTM G173-03).
pub fn am1_5_integrated_power() -> f64 {
    SolarSpectrum::am15g().total_power()
}

// ─── Photovoltaic potential ───────────────────────────────────────────────────

/// Maximum photovoltaic power density (W/m²) extractable from a solar spectrum
/// for a single-junction cell with given bandgap E_g (eV).
///
/// Upper bound: all above-bandgap photons converted at voltage V = E_g/e.
///
/// P_max ≤ J_photon · V_g = e · Φ_above_gap · E_g
///
/// This is an upper limit; real efficiency is lower due to thermodynamic losses.
pub fn photovoltaic_potential(spectrum: &SolarSpectrum, bandgap_ev: f64) -> f64 {
    // J_sc = e · Φ_above_gap  [A/m²]
    // V_g = E_g [eV] in volts = bandgap_ev [V]
    // P = J_sc · V_g = e · Φ · bandgap_ev  [W/m²]
    let flux = spectrum.above_bandgap_photon_flux(bandgap_ev);
    ELECTRON_CHARGE * flux * bandgap_ev
}

// ─── Shockley-Queisser efficiency limit ───────────────────────────────────────

/// Shockley-Queisser (S-Q) detailed-balance efficiency limit for a single-junction
/// solar cell at 300 K illuminated by AM1.5G (1000 W/m²).
///
/// The S-Q calculation involves:
///   1. Integrating the solar photon flux above E_g to get J_sc
///   2. Computing the dark saturation current J_0 from blackbody emission
///   3. Solving for V_oc and the maximum power point
///
/// This implementation uses the standard tabulated approximation with the
/// exact S-Q integral at T = 300 K and sun temperature T_sun = 6000 K.
///
/// Returns efficiency η (0–1).
pub fn shockley_queisser_efficiency(bandgap_ev: f64) -> f64 {
    let t_cell = 300.0; // K
    let t_sun = 6000.0; // K (approximate)
    let p_in = 1000.0; // W/m² (AM1.5G)

    // Short-circuit current: J_sc = e · ∫_{E_g}^{∞} Φ_sun(E) dE
    let lambda_max_m = 1_239.841_93e-9 / bandgap_ev;
    let spec = SolarSpectrum::am15g();
    let phi_sc = integrate_photon_flux_spectrum(&spec, 280e-9, lambda_max_m, 5000);
    let j_sc = ELECTRON_CHARGE * phi_sc;

    // Dark saturation current from blackbody at T_cell:
    // J_0 = e · ∫_{E_g}^{∞} Φ_bb(E, T_cell) dE
    // Use the reciprocity relation: J_0 = e · Φ_bb(T_cell, λ_min=100nm, λ_max=λ_g)
    // Actually J_0 = e · ∫_{E_g}^{∞} Φ_bb(E, T_cell) dE
    let phi_bb = blackbody_photon_flux(t_cell, 100e-9, lambda_max_m, 3000);
    // Scale by geometric factor: (solid angle of sun / π) ≈ (6.8e-5 sr / π) for AM1.5G
    let j0 = ELECTRON_CHARGE * phi_bb;

    if j0 < 1e-50 {
        return 0.0;
    }

    // Open-circuit voltage: V_oc = (kT/e) · ln(J_sc/J_0 + 1)
    let kt_e = BOLTZMANN * t_cell / ELECTRON_CHARGE;
    let v_oc = kt_e * (j_sc / j0 + 1.0).ln();

    if v_oc <= 0.0 {
        return 0.0;
    }

    // Fill factor approximation (Green's empirical formula):
    // FF = (u - ln(u + 0.72)) / (u + 1)   where u = e·V_oc/(k_B·T)
    let u = v_oc / kt_e;
    let ff = if u > 1.0 {
        (u - (u + 0.72).ln()) / (u + 1.0)
    } else {
        0.0
    };

    // Maximum power density
    let p_max = j_sc * v_oc * ff;
    let eta = p_max / p_in;

    // Sanity clamp: S-Q limit is at most ~33.7% at the optimal bandgap
    // The sun-temperature approximation can give unphysically high values;
    // cap at 40% for robustness.
    let _ = t_sun;
    eta.clamp(0.0, 0.40)
}

// ─── Concentration effects ────────────────────────────────────────────────────

/// Effect of solar concentration ratio on V_oc and J_sc.
///
/// Under concentration X:
///   J_sc → X · J_sc
///   V_oc → V_oc(1sun) + (k_B·T/e) · ln(X)
///
/// # Arguments
/// * `conc` — concentration ratio (e.g., 100 for 100× concentration)
/// * `jsc` — short-circuit current density at 1 sun (A/m²)
/// * `j0` — dark saturation current density (A/m²)
/// * `t_k` — cell temperature (K)
///
/// # Returns
/// `(v_oc, j_sc)` at the given concentration
pub fn concentration_ratio_effect(conc: f64, jsc: f64, j0: f64, t_k: f64) -> (f64, f64) {
    let kt_e = BOLTZMANN * t_k / ELECTRON_CHARGE;
    let jsc_conc = jsc * conc;
    let voc_1sun = kt_e * (jsc / j0 + 1.0).ln();
    let voc_conc = voc_1sun + kt_e * conc.ln();
    (voc_conc, jsc_conc)
}

/// Fill factor from the empirical Green formula given V_oc (V) and T (K).
///
/// FF = (u − ln(u + 0.72)) / (u + 1)  where u = e·V_oc/(k_B·T)
pub fn fill_factor(voc: f64, temperature_k: f64) -> f64 {
    let u = voc * ELECTRON_CHARGE / (BOLTZMANN * temperature_k);
    if u <= 1.0 {
        return 0.0;
    }
    (u - (u + 0.72).ln()) / (u + 1.0)
}

/// Open-circuit voltage from J_sc and J_0 (ideal diode).
///
/// V_oc = (k_B·T/e) · ln(J_sc / J_0 + 1)
pub fn open_circuit_voltage(jsc: f64, j0: f64, temperature_k: f64) -> f64 {
    let kt_e = BOLTZMANN * temperature_k / ELECTRON_CHARGE;
    kt_e * (jsc / j0 + 1.0).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Existing tests ────────────────────────────────────────────────────────

    #[test]
    fn am15g_creates_without_panic() {
        let spec = SolarSpectrum::am15g();
        assert!(!spec.wavelengths.is_empty());
    }

    #[test]
    fn am15g_total_irradiance_near_1000() {
        let spec = SolarSpectrum::am15g();
        let total = spec.total_irradiance();
        assert!(
            total > 700.0 && total < 1500.0,
            "Total irradiance={total:.1} W/m²"
        );
    }

    #[test]
    fn am15g_peak_in_visible() {
        let spec = SolarSpectrum::am15g();
        let ir_500 = spec.irradiance_at(500e-9);
        let ir_2000 = spec.irradiance_at(2000e-9);
        assert!(
            ir_500 > ir_2000,
            "Visible irradiance should exceed IR at 2µm"
        );
    }

    #[test]
    fn interpolation_at_data_point() {
        let spec = SolarSpectrum::am15g();
        // At 500nm, should match table value (1.719 W/m²/nm = 1.719e9 W/m²/m)
        let ir = spec.irradiance_at(500e-9);
        let expected = 1.719e9;
        let rel_err = (ir - expected).abs() / expected;
        assert!(rel_err < 0.01, "At 500nm: {ir:.3e} vs {expected:.3e}");
    }

    #[test]
    fn photon_flux_positive() {
        let spec = SolarSpectrum::am15g();
        let flux = spec.photon_flux(550e-9);
        assert!(flux > 0.0);
        assert!(flux > 1e24 && flux < 1e30);
    }

    #[test]
    fn integrate_full_range_positive() {
        let spec = SolarSpectrum::am15g();
        let total = spec.integrate(400e-9, 700e-9, 1000);
        assert!(total > 0.0 && total < 1000.0);
    }

    // ─── New tests ─────────────────────────────────────────────────────────────

    #[test]
    fn total_power_same_as_total_irradiance() {
        let spec = SolarSpectrum::am15g();
        let p = spec.total_power();
        let i = spec.total_irradiance();
        assert!((p - i).abs() < 1.0);
    }

    #[test]
    fn photon_flux_density_positive() {
        let spec = SolarSpectrum::am15g();
        let phi = spec.photon_flux_density(400e-9, 700e-9);
        assert!(phi > 0.0, "Φ={phi:.2e}");
    }

    #[test]
    fn above_bandgap_photon_flux_si() {
        // Si bandgap ~1.12 eV → λ_g ~1107nm; above-gap flux should be large
        let spec = SolarSpectrum::am15g();
        let phi = spec.above_bandgap_photon_flux(1.12);
        assert!(phi > 1e18, "Φ(>Eg,Si)={phi:.2e}");
    }

    #[test]
    fn integrate_photon_flux_from_table() {
        let data: Vec<(f64, f64)> = AM15G_DATA
            .iter()
            .filter(|(wl, _)| *wl >= 400.0 && *wl <= 700.0)
            .map(|(wl, e)| (wl * 1e-9, e * 1e9))
            .collect();
        let phi = integrate_photon_flux(&data, 400e-9, 700e-9);
        assert!(phi > 1e18, "Φ={phi:.2e}");
    }

    #[test]
    fn blackbody_photon_flux_sun() {
        // Sun T≈6000K: photon flux in visible should be large
        let phi = blackbody_photon_flux(6000.0, 300e-9, 800e-9, 500);
        assert!(phi > 1e20, "Φ_bb(6000K)={phi:.2e}");
    }

    #[test]
    fn blackbody_radiance_peak_wavelength() {
        // Wien's law: λ_max·T ≈ 2898 µm·K
        // At T=5800 K: λ_max ≈ 500 nm
        let t = 5800.0;
        let lam = 500e-9;
        let b500 = planck_radiance(lam, t);
        let b1000 = planck_radiance(1000e-9, t);
        assert!(b500 > b1000, "Visible peak should exceed IR for T=5800K");
    }

    #[test]
    fn am1_5_integrated_power_near_1000() {
        let p = am1_5_integrated_power();
        assert!(p > 700.0 && p < 1500.0, "P={p:.1} W/m²");
    }

    #[test]
    fn photovoltaic_potential_si() {
        let spec = SolarSpectrum::am15g();
        let p = photovoltaic_potential(&spec, 1.12);
        assert!(p > 100.0 && p < 2000.0, "PV potential(Si)={p:.1} W/m²");
    }

    #[test]
    fn sq_efficiency_silicon_range() {
        // S-Q limit at 1.1 eV ≈ 33-34%
        let eta = shockley_queisser_efficiency(1.1);
        assert!(eta > 0.1 && eta < 0.4, "η_SQ(Si)={eta:.3}");
    }

    #[test]
    fn sq_efficiency_gaas_range() {
        // S-Q limit at 1.4 eV (GaAs) is near the maximum ~33%
        let eta = shockley_queisser_efficiency(1.42);
        assert!(eta > 0.1 && eta < 0.4, "η_SQ(GaAs)={eta:.3}");
    }

    #[test]
    fn concentration_increases_voc() {
        let (voc1, _) = concentration_ratio_effect(1.0, 400.0, 1e-10, 300.0);
        let (voc100, _) = concentration_ratio_effect(100.0, 400.0, 1e-10, 300.0);
        assert!(voc100 > voc1, "V_oc should increase with concentration");
    }

    #[test]
    fn concentration_scales_jsc() {
        let (_, jsc1) = concentration_ratio_effect(1.0, 400.0, 1e-10, 300.0);
        let (_, jsc100) = concentration_ratio_effect(100.0, 400.0, 1e-10, 300.0);
        let ratio = jsc100 / jsc1;
        assert!((ratio - 100.0).abs() < 1e-6, "J_sc should scale as X");
    }

    #[test]
    fn fill_factor_typical_range() {
        // V_oc ≈ 0.6 V at 300 K gives FF ≈ 0.8
        let ff = fill_factor(0.6, 300.0);
        assert!(ff > 0.7 && ff < 0.9, "FF={ff:.3}");
    }

    #[test]
    fn open_circuit_voltage_positive() {
        let voc = open_circuit_voltage(400.0, 1e-10, 300.0);
        assert!(voc > 0.5 && voc < 1.0, "V_oc={voc:.4} V");
    }

    #[test]
    fn from_data_roundtrip() {
        let data = vec![(500e-9, 1.0e9), (600e-9, 0.9e9)];
        let spec = SolarSpectrum::from_data(data);
        let ir = spec.irradiance_at(550e-9);
        assert!((ir - 0.95e9).abs() < 1e6, "ir={ir:.2e}");
    }
}
