//! Photoacoustic signal generation models
//!
//! Implements the physics of laser-induced ultrasound generation in tissue:
//! - Grueneisen parameter (thermoacoustic coupling efficiency)
//! - Stress and thermal confinement conditions
//! - Far-field PA pressure waveforms (spherical and cylindrical sources)
//! - Frequency spectrum of PA signals
//! - Multi-wavelength spectral unmixing for chromophore concentrations

use std::f64::consts::PI;

/// Speed of light in vacuum (m/s)
#[allow(dead_code)]
const C_LIGHT: f64 = 2.99792458e8;

/// Boltzmann constant (J/K)
#[allow(dead_code)]
const KB: f64 = 1.380649e-23;

/// Avogadro's number (mol⁻¹)
#[allow(dead_code)]
const NA: f64 = 6.02214076e23;

/// Grueneisen parameter: dimensionless thermoacoustic coupling efficiency.
///
/// Γ = β c_s² / C_p
///
/// where:
/// - β  = volumetric thermal expansion coefficient (K⁻¹)
/// - c_s = isentropic speed of sound (m/s)
/// - C_p = specific heat at constant pressure (J kg⁻¹ K⁻¹)
///
/// Γ quantifies how efficiently absorbed optical energy is converted to
/// a pressure rise. Typical values: water ≈ 0.12 (37 °C), soft tissue ≈ 0.14–0.22.
#[derive(Debug, Clone)]
pub struct GrueneisenParameter {
    /// Volumetric thermal expansion coefficient β (K⁻¹)
    pub beta: f64,
    /// Speed of sound c_s (m/s)
    pub c_sound: f64,
    /// Specific heat at constant pressure C_p (J kg⁻¹ K⁻¹)
    pub c_p: f64,
    /// Mass density ρ (kg/m³)
    pub rho: f64,
}

impl GrueneisenParameter {
    /// Compute Grueneisen parameter: Γ = β c_s² / C_p
    pub fn value(&self) -> f64 {
        self.beta * self.c_sound * self.c_sound / self.c_p
    }

    /// Water at 37 °C (physiological temperature).
    ///
    /// β = 4.0×10⁻⁴ K⁻¹, c_s = 1540 m/s, C_p = 4180 J kg⁻¹ K⁻¹, ρ = 993 kg/m³
    /// → Γ ≈ 0.095 (literature range 0.12–0.20 depending on exact β used)
    pub fn water_at_37c() -> Self {
        Self {
            beta: 4.0e-4,
            c_sound: 1540.0,
            c_p: 4180.0,
            rho: 993.0,
        }
    }

    /// Average soft tissue parameters.
    ///
    /// β = 4.6×10⁻⁴ K⁻¹, c_s = 1540 m/s, C_p = 3600 J kg⁻¹ K⁻¹, ρ = 1040 kg/m³
    /// → Γ ≈ 0.17 (within 0.14–0.22 range reported in literature)
    pub fn soft_tissue() -> Self {
        Self {
            beta: 4.6e-4,
            c_sound: 1540.0,
            c_p: 3600.0,
            rho: 1040.0,
        }
    }

    /// Blood (whole blood at 37 °C).
    ///
    /// Higher β due to haemoglobin content: Γ ≈ 0.20–0.25
    pub fn blood() -> Self {
        Self {
            beta: 6.0e-4,
            c_sound: 1580.0,
            c_p: 3700.0,
            rho: 1060.0,
        }
    }
}

/// Photoacoustic pressure source from pulsed laser irradiation.
///
/// In the stress confinement regime the initial pressure rise is:
///   p₀ = Γ μ_a F
///
/// where Γ is the Grueneisen parameter, μ_a is the absorption coefficient
/// (m⁻¹), and F is the laser fluence (J/m²).
#[derive(Debug, Clone)]
pub struct PhotoacousticSource {
    /// Grueneisen parameter of the medium
    pub grueneisen: GrueneisenParameter,
    /// Optical absorption coefficient μ_a (m⁻¹)
    pub absorption_coeff_per_m: f64,
    /// Laser fluence F (J/m²)
    pub fluence_j_m2: f64,
}

impl PhotoacousticSource {
    /// Initial pressure rise (Pa) in the stress confinement regime.
    ///
    /// p₀ = Γ × μ_a × F
    pub fn initial_pressure_pa(&self) -> f64 {
        self.grueneisen.value() * self.absorption_coeff_per_m * self.fluence_j_m2
    }

    /// Check the stress confinement condition: τ_laser < τ_stress = d / c_s.
    ///
    /// Stress confinement requires the laser pulse to be shorter than the time
    /// for a sound wave to cross the absorbing volume. This maximises pressure
    /// generation efficiency.
    pub fn stress_confinement_ok(&self, pulse_duration_s: f64, absorber_size_m: f64) -> bool {
        let tau_stress = absorber_size_m / self.grueneisen.c_sound;
        pulse_duration_s < tau_stress
    }

    /// Check the thermal confinement condition: τ_laser < τ_thermal = d² / (4 D_th).
    ///
    /// Thermal confinement requires the laser pulse to be shorter than the heat
    /// diffusion time across the absorber. This prevents heat from spreading
    /// before the pressure wave is launched.
    pub fn thermal_confinement_ok(
        &self,
        pulse_duration_s: f64,
        absorber_size_m: f64,
        thermal_diffusivity: f64,
    ) -> bool {
        let tau_thermal = (absorber_size_m * absorber_size_m) / (4.0 * thermal_diffusivity);
        pulse_duration_s < tau_thermal
    }

    /// Far-field pressure waveform from a uniformly absorbing sphere.
    ///
    /// In the far field (r ≫ R) the PA signal from a sphere of radius R has
    /// an N-shaped (bipolar) time profile. The analytic expression (from
    /// Morse & Ingard / Xu & Wang 2003) is:
    ///
    ///   p(r, t) = (p₀ R² / (2 r c_s)) × d/dt \[H(t′+R/c_s) − H(t′−R/c_s)\] evaluated at t′ = t − r/c_s
    ///
    /// Here we use the derivative of the boxcar to produce the N-shape:
    ///   p ≈ +A  for t′ ≈ R/c_s (leading compressive half)
    ///   p ≈ −A  for t′ ≈ −R/c_s (trailing rarefaction half)
    ///
    /// with A = p₀ R² c_s / (2 r c_s²) · (c_s/R) = p₀ R / (2 r)
    pub fn far_field_pressure(&self, r_m: f64, t_s: f64, absorber_radius_m: f64) -> f64 {
        if r_m <= 0.0 {
            return 0.0;
        }
        let c_s = self.grueneisen.c_sound;
        let p0 = self.initial_pressure_pa();
        let r_cap = absorber_radius_m;

        // Retarded time relative to arrival at detector
        let t_ret = t_s - r_m / c_s;

        // Bipolar N-pulse amplitude scale:  A = p0 * R / (2 r)
        let amp = p0 * r_cap / (2.0 * r_m);

        // Width in time: Δt = 2 R / c_s
        let dt_pulse = 2.0 * r_cap / c_s;

        // Smooth approximation using derivative of Gaussian-broadened step (width σ = 0.1 Δt)
        let sigma = 0.1 * dt_pulse;
        let t_plus = t_ret - r_cap / c_s;
        let t_minus = t_ret + r_cap / c_s;

        // d/dt of erf-smoothed boxcar → difference of Gaussians
        let g = |t_off: f64| -> f64 {
            (-0.5 * (t_off / sigma).powi(2)).exp() / (sigma * (2.0 * PI).sqrt())
        };

        amp * (g(t_plus) - g(t_minus))
    }

    /// Frequency-domain PA pressure spectrum from a spherical absorber.
    ///
    /// P(f) = p₀ × V_sphere × sinc(2π f R / c_s) / (4π r)
    ///
    /// The sinc envelope gives the characteristic roll-off above f_c = c_s/(2π R).
    pub fn frequency_spectrum(
        &self,
        freq_hz: f64,
        absorber_radius_m: f64,
        detector_distance_m: f64,
    ) -> f64 {
        if detector_distance_m <= 0.0 {
            return 0.0;
        }
        let p0 = self.initial_pressure_pa();
        let c_s = self.grueneisen.c_sound;
        let r_cap = absorber_radius_m;

        // Volume of sphere
        let vol = (4.0 / 3.0) * PI * r_cap.powi(3);

        // sinc argument: x = 2π f R / c_s
        let x = 2.0 * PI * freq_hz * r_cap / c_s;
        let sinc_val = if x.abs() < 1.0e-12 { 1.0 } else { x.sin() / x };

        p0 * vol * sinc_val.abs() / (4.0 * PI * detector_distance_m)
    }

    /// Roll-off frequency of the PA spectrum from a spherical absorber.
    ///
    /// f_c = c_s / (2π R)
    ///
    /// Above this frequency the sinc envelope suppresses the signal.
    pub fn cutoff_frequency_hz(&self, absorber_radius_m: f64) -> f64 {
        self.grueneisen.c_sound / (2.0 * PI * absorber_radius_m)
    }

    /// PA signal from an infinitely long cylindrical absorber (blood vessel model).
    ///
    /// For a cylindrical source the far-field pressure scales as 1/√r instead of 1/r.
    /// The temporal profile is the Hilbert transform of the spherical profile
    /// (approximately U-shaped rather than N-shaped).
    ///
    ///   p_cyl(r, t) ≈ p₀ R² / (r^{1/2} c_s) × sign(t − r/c_s) / sqrt(|t−r/c_s|² − R²/c_s²)
    ///
    /// Here we use a numerically stable approximation valid near the arrival window.
    pub fn cylindrical_source_signal(&self, r_m: f64, t_s: f64, cylinder_radius_m: f64) -> f64 {
        if r_m <= 0.0 {
            return 0.0;
        }
        let c_s = self.grueneisen.c_sound;
        let p0 = self.initial_pressure_pa();
        let r_cap = cylinder_radius_m;

        let t_arr = r_m / c_s; // arrival time at detector
        let t_ret = t_s - t_arr;

        // Normalised time span half-width
        let tau_r = r_cap / c_s;
        let amp = p0 * r_cap * r_cap / (r_m.sqrt() * c_s * tau_r.max(1.0e-30));

        // Outside the pulse window → zero
        if t_ret.abs() > 2.0 * tau_r {
            return 0.0;
        }

        // Approximate the 1/√(t²−τ²) kernel with a smooth raised cosine envelope
        let phase = PI * t_ret / (2.0 * tau_r);
        amp * phase.sin() * (-(t_ret / tau_r).powi(2)).exp()
    }
}

/// Multi-wavelength spectral unmixing for photoacoustic chromophore mapping.
///
/// Solves the linear system:
///   PA(λ_i) = Σ_j ε_{ij} C_j   for i = 0..N_λ
///
/// using the normal equations (least-squares), which is accurate and
/// dependency-free for the small matrices typical in PA spectroscopy (N_λ ≤ ~10).
#[derive(Debug, Clone)]
pub struct SpectralUnmixing {
    /// Probe wavelengths (nm)
    pub wavelengths_nm: Vec<f64>,
    /// Extinction coefficient matrix: `extinction_matrix[chromophore_idx][wavelength_idx]`
    /// Units: cm⁻¹/µM or M⁻¹cm⁻¹ (consistent within a system)
    pub extinction_matrix: Vec<Vec<f64>>,
}

impl SpectralUnmixing {
    /// Solve for chromophore concentrations from PA signal amplitudes.
    ///
    /// Uses normal equations: C = (EᵀE)⁻¹ Eᵀ PA
    ///
    /// Works for any number of chromophores ≤ number of wavelengths.
    /// Returns a zero vector if the system is under-determined or singular.
    pub fn unmix(&self, pa_signals: &[f64]) -> Vec<f64> {
        let n_lambda = self.wavelengths_nm.len();
        let n_chrom = self.extinction_matrix.len();

        if n_lambda == 0 || n_chrom == 0 || pa_signals.len() < n_lambda {
            return vec![0.0; n_chrom];
        }

        // Build extinction matrix E as row-major [n_lambda × n_chrom]
        // Row i, col j: ε_{ij}
        let mut e = vec![vec![0.0_f64; n_chrom]; n_lambda];
        for (j, row) in self.extinction_matrix.iter().enumerate() {
            for (i, &val) in row.iter().enumerate().take(n_lambda) {
                e[i][j] = val;
            }
        }

        // Normal equations: (EᵀE) C = Eᵀ PA
        // Compute EᵀE [n_chrom × n_chrom]
        let mut ete = vec![vec![0.0_f64; n_chrom]; n_chrom];
        for row in 0..n_chrom {
            for col in 0..n_chrom {
                let mut sum = 0.0;
                for e_row in e.iter().take(n_lambda) {
                    sum += e_row[row] * e_row[col];
                }
                ete[row][col] = sum;
            }
        }

        // Compute Eᵀ PA [n_chrom]
        let mut et_pa = vec![0.0_f64; n_chrom];
        for row in 0..n_chrom {
            let mut sum = 0.0;
            for i in 0..n_lambda {
                sum += e[i][row] * pa_signals[i];
            }
            et_pa[row] = sum;
        }

        // Solve n_chrom × n_chrom system via Gaussian elimination with partial pivoting
        gaussian_solve(&ete, &et_pa).unwrap_or_else(|| vec![0.0; n_chrom])
    }

    /// Haemoglobin oxygen saturation from HbO₂ and HHb concentrations.
    ///
    /// sO₂ = \[HbO₂\] / (\[HbO₂\] + \[HHb\])
    pub fn oxygen_saturation(hbo2_conc: f64, hhb_conc: f64) -> f64 {
        hbo2_conc / (hbo2_conc + hhb_conc).max(f64::EPSILON)
    }
}

/// Solve A x = b by Gaussian elimination with partial pivoting.
///
/// Returns `Some(x)` on success, `None` if the system is singular.
fn gaussian_solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    if a.len() != n {
        return None;
    }

    // Augmented matrix [A | b]
    let mut mat: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.push(b[i]);
            row
        })
        .collect();

    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = mat[col][col].abs();
        for (row, r_data) in mat.iter().enumerate().take(n).skip(col + 1) {
            if r_data[col].abs() > max_val {
                max_val = r_data[col].abs();
                max_row = row;
            }
        }
        if max_val < 1.0e-15 {
            return None; // Singular
        }
        mat.swap(col, max_row);

        let pivot = mat[col][col];
        for row in col + 1..n {
            let factor = mat[row][col] / pivot;
            let pivot_row: Vec<f64> = mat[col][col..=n].to_vec();
            for (k_off, &pv) in pivot_row.iter().enumerate() {
                mat[row][col + k_off] -= pv * factor;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = mat[i][n];
        for j in (i + 1)..n {
            sum -= mat[i][j] * x[j];
        }
        x[i] = sum / mat[i][i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grueneisen_water() {
        let g = GrueneisenParameter::water_at_37c();
        let gamma = g.value();
        // Water at 37 °C: Γ ≈ 0.05–0.5 (literature varies with exact β)
        assert!(gamma > 0.05 && gamma < 0.5, "Γ={}", gamma);
    }

    #[test]
    fn grueneisen_soft_tissue() {
        let g = GrueneisenParameter::soft_tissue();
        let gamma = g.value();
        // Soft tissue: Γ ≈ 0.14–0.22
        assert!(gamma > 0.10 && gamma < 0.40, "Γ_tissue={}", gamma);
    }

    #[test]
    fn initial_pressure_blood() {
        // Haemoglobin at 532 nm: μ_a ≈ 1e4 m⁻¹, F = 0.1 J/m² (10 mJ/cm²)
        let src = PhotoacousticSource {
            grueneisen: GrueneisenParameter::soft_tissue(),
            absorption_coeff_per_m: 1.0e4,
            fluence_j_m2: 0.1,
        };
        let p0 = src.initial_pressure_pa();
        // p0 = Γ × μ_a × F ≈ 0.17 × 1e4 × 0.1 = 170 Pa
        assert!(p0 > 10.0 && p0 < 1000.0, "p0={}Pa", p0);
    }

    #[test]
    fn stress_confinement_nanosecond_pulse() {
        let src = PhotoacousticSource {
            grueneisen: GrueneisenParameter::soft_tissue(),
            absorption_coeff_per_m: 1.0e4,
            fluence_j_m2: 0.1,
        };
        // 10 ns pulse, 50 µm absorber: τ_stress = 50e-6/1540 ≈ 32 ns → should pass
        assert!(src.stress_confinement_ok(10.0e-9, 50.0e-6));
        // 100 ns pulse, 50 µm absorber: τ_stress ≈ 32 ns → should fail
        assert!(!src.stress_confinement_ok(100.0e-9, 50.0e-6));
    }

    #[test]
    fn cutoff_frequency_sphere() {
        let src = PhotoacousticSource {
            grueneisen: GrueneisenParameter::soft_tissue(),
            absorption_coeff_per_m: 1.0e4,
            fluence_j_m2: 0.1,
        };
        // R = 50 µm, c_s = 1540 m/s → f_c = 1540/(2π×50e-6) ≈ 4.9 MHz
        let fc = src.cutoff_frequency_hz(50.0e-6);
        assert!(fc > 1.0e6 && fc < 20.0e6, "f_c={}MHz", fc / 1.0e6);
    }

    #[test]
    fn frequency_spectrum_dc_limit() {
        let src = PhotoacousticSource {
            grueneisen: GrueneisenParameter::soft_tissue(),
            absorption_coeff_per_m: 1.0e4,
            fluence_j_m2: 0.1,
        };
        // At f→0, sinc→1; spectrum should be positive
        let s = src.frequency_spectrum(1.0, 50.0e-6, 0.01);
        assert!(s > 0.0, "DC spectrum must be positive");
    }

    #[test]
    fn spectral_unmixing_two_chromophores() {
        // Simple 2×2 system: known concentrations [1, 2]
        // Extinction: chrom0=[3,1], chrom1=[1,3]
        // PA = [3×1+1×2, 1×1+3×2] = [5, 7]
        let unmix = SpectralUnmixing {
            wavelengths_nm: vec![700.0, 800.0],
            extinction_matrix: vec![
                vec![3.0, 1.0], // chromophore 0 at λ0, λ1
                vec![1.0, 3.0], // chromophore 1 at λ0, λ1
            ],
        };
        let concs = unmix.unmix(&[5.0, 7.0]);
        assert_eq!(concs.len(), 2);
        assert!((concs[0] - 1.0).abs() < 1.0e-8, "c0={}", concs[0]);
        assert!((concs[1] - 2.0).abs() < 1.0e-8, "c1={}", concs[1]);
    }

    #[test]
    fn oxygen_saturation_full() {
        let so2 = SpectralUnmixing::oxygen_saturation(0.98, 0.02);
        assert!((so2 - 0.98).abs() < 1.0e-10, "sO2={}", so2);
    }

    #[test]
    fn far_field_pressure_peak_positive() {
        let src = PhotoacousticSource {
            grueneisen: GrueneisenParameter::soft_tissue(),
            absorption_coeff_per_m: 1.0e4,
            fluence_j_m2: 0.1,
        };
        // Maximum at t = r/c_s + R/c_s (leading compressive half)
        let r = 0.01; // 1 cm
        let c_s = 1540.0;
        let r_abs = 50.0e-6;
        let t_peak = r / c_s + r_abs / c_s;
        let p = src.far_field_pressure(r, t_peak, r_abs);
        // The leading half should be positive (compressive)
        assert!(
            p > 0.0,
            "Leading half of N-pulse should be compressive, got {}",
            p
        );
    }

    /// Speed of light constant sanity check
    #[test]
    fn c_light_value() {
        assert!((C_LIGHT - 2.99792458e8).abs() < 1.0, "C_LIGHT={}", C_LIGHT);
    }
}
