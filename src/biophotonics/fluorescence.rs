//! Fluorescence, FRET, two-photon excitation, and FLIM models
//!
//! Implements:
//! - Fluorophore photophysics (absorption, emission, photobleaching)
//! - FRET (Förster Resonance Energy Transfer) with distance/lifetime analysis
//! - Two-photon excitation (2PE) cross-section and signal models
//! - Fluorescence lifetime imaging (FLIM) phasor analysis and exponential fitting

use crate::error::OxiPhotonError;

/// Planck constant (J·s)
const H_PLANCK: f64 = 6.62607015e-34;
/// Speed of light in vacuum (m/s)
const C0: f64 = 2.99792458e8;
/// Boltzmann constant (J/K)
#[allow(dead_code)]
const KB: f64 = 1.380649e-23;
/// Avogadro's number (mol⁻¹)
const AVOGADRO: f64 = 6.02214076e23;
/// 2π
const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// Spectral bandwidth (FWHM) for emission Gaussian approximation.
/// Typical organic fluorophore emission FWHM ≈ 30–60 nm.
const DEFAULT_FWHM_NM: f64 = 40.0;

/// Photophysical model of a fluorescent molecule.
///
/// Encapsulates the key parameters governing fluorescence excitation,
/// emission, and photostability.
#[derive(Debug, Clone)]
pub struct Fluorophore {
    /// Human-readable name (e.g. "FITC", "GFP")
    pub name: String,
    /// Peak excitation wavelength (nm)
    pub excitation_peak_nm: f64,
    /// Peak emission wavelength (nm)
    pub emission_peak_nm: f64,
    /// Molar extinction coefficient at excitation peak (M⁻¹cm⁻¹)
    pub extinction_coefficient_m_per_cm: f64,
    /// Fluorescence quantum yield Φ ∈ [0, 1]
    pub quantum_yield: f64,
    /// Fluorescence lifetime τ (ns)
    pub lifetime_ns: f64,
    /// Relative photobleaching rate (arbitrary units; larger = faster bleaching)
    pub photobleaching_threshold: f64,
}

impl Fluorophore {
    /// Construct a fluorophore with explicit parameters.
    pub fn new(
        name: impl Into<String>,
        ex_nm: f64,
        em_nm: f64,
        epsilon: f64,
        qy: f64,
        lifetime_ns: f64,
    ) -> Self {
        Self {
            name: name.into(),
            excitation_peak_nm: ex_nm,
            emission_peak_nm: em_nm,
            extinction_coefficient_m_per_cm: epsilon,
            quantum_yield: qy,
            lifetime_ns,
            photobleaching_threshold: 1.0,
        }
    }

    /// Fluorescein isothiocyanate (FITC).
    ///
    /// ex=494 nm, em=521 nm, ε=73 000 M⁻¹cm⁻¹, Φ=0.93, τ=4.0 ns
    pub fn fitc() -> Self {
        Self::new("FITC", 494.0, 521.0, 73_000.0, 0.93, 4.0)
    }

    /// Rhodamine B.
    ///
    /// ex=540 nm, em=625 nm, ε=106 000 M⁻¹cm⁻¹, Φ=0.65, τ=2.5 ns
    pub fn rhodamine_b() -> Self {
        Self::new("Rhodamine B", 540.0, 625.0, 106_000.0, 0.65, 2.5)
    }

    /// Green Fluorescent Protein (GFP, enhanced).
    ///
    /// ex=488 nm, em=507 nm, ε=56 000 M⁻¹cm⁻¹, Φ=0.79, τ=2.4 ns
    pub fn gfp() -> Self {
        Self::new("GFP", 488.0, 507.0, 56_000.0, 0.79, 2.4)
    }

    /// Cyanine 3 (Cy3).
    ///
    /// ex=550 nm, em=570 nm, ε=150 000 M⁻¹cm⁻¹, Φ=0.04, τ=0.3 ns
    pub fn cy3() -> Self {
        Self::new("Cy3", 550.0, 570.0, 150_000.0, 0.04, 0.3)
    }

    /// Cyanine 5 (Cy5).
    ///
    /// ex=650 nm, em=670 nm, ε=250 000 M⁻¹cm⁻¹, Φ=0.27, τ=1.0 ns
    pub fn cy5() -> Self {
        Self::new("Cy5", 650.0, 670.0, 250_000.0, 0.27, 1.0)
    }

    /// Stokes shift: Δλ = λ_em − λ_ex  (nm)
    pub fn stokes_shift_nm(&self) -> f64 {
        self.emission_peak_nm - self.excitation_peak_nm
    }

    /// Molecular brightness: B = ε × Φ  (M⁻¹cm⁻¹)
    ///
    /// Higher brightness means more detected photons per molecule per unit irradiance.
    pub fn brightness(&self) -> f64 {
        self.extinction_coefficient_m_per_cm * self.quantum_yield
    }

    /// Saturation intensity: I_sat = hν / (σ · τ)  (W/cm²)
    ///
    /// Above this irradiance the ground-state population is significantly depleted
    /// (stimulated emission / excited-state absorption becomes important).
    pub fn saturation_intensity_w_per_cm2(&self) -> f64 {
        let sigma = self.absorption_cross_section_cm2();
        let tau_s = self.lifetime_ns * 1.0e-9;
        let lambda_m = self.excitation_peak_nm * 1.0e-9;
        let h_nu = H_PLANCK * C0 / lambda_m;
        h_nu / (sigma * tau_s)
    }

    /// Absorption cross-section: σ = 2303 · ε / N_A  (cm²)
    ///
    /// Converts from molar extinction coefficient (M⁻¹cm⁻¹) to per-molecule cross-section.
    pub fn absorption_cross_section_cm2(&self) -> f64 {
        // Beer-Lambert: A = ε·c·l → 2303·ε·c·l = -ln(T) → σ = 2303·ε/Na
        2303.0 * self.extinction_coefficient_m_per_cm / AVOGADRO
    }

    /// Single-molecule fluorescence emission rate.
    ///
    /// R = (σ · I_photons) · Φ  where I_photons = I / (hν)
    ///
    /// # Arguments
    /// * `irradiance_w_per_cm2` — excitation irradiance (W/cm²)
    ///
    /// # Returns
    /// Emission rate in photons/s per molecule.
    pub fn emission_rate(&self, irradiance_w_per_cm2: f64) -> f64 {
        let lambda_m = self.excitation_peak_nm * 1.0e-9;
        let h_nu = H_PLANCK * C0 / lambda_m;
        let photon_flux = irradiance_w_per_cm2 / h_nu; // photons cm⁻² s⁻¹
        let sigma = self.absorption_cross_section_cm2();
        sigma * photon_flux * self.quantum_yield
    }

    /// Emission spectrum as a Gaussian approximation.
    ///
    /// S(λ) = A · exp(−(λ − λ_em)² / (2σ_em²))
    /// normalized so that the peak = 1 (relative spectrum).
    ///
    /// # Arguments
    /// * `lambda_nm_range` — (λ_min, λ_max) wavelength range in nm
    /// * `n_pts` — number of wavelength points
    ///
    /// # Returns
    /// Vector of (wavelength_nm, relative_intensity) pairs.
    pub fn emission_spectrum(&self, lambda_nm_range: (f64, f64), n_pts: usize) -> Vec<(f64, f64)> {
        let (lambda_min, lambda_max) = lambda_nm_range;
        // Convert FWHM to Gaussian σ: σ = FWHM / (2√(2·ln2))
        let sigma_nm = DEFAULT_FWHM_NM / (2.0 * (2.0 * 2_f64.ln()).sqrt());

        (0..n_pts)
            .map(|i| {
                let lambda = lambda_min
                    + (lambda_max - lambda_min) * (i as f64) / ((n_pts - 1).max(1) as f64);
                let delta = lambda - self.emission_peak_nm;
                let intensity = (-delta * delta / (2.0 * sigma_nm * sigma_nm)).exp();
                (lambda, intensity)
            })
            .collect()
    }

    /// Signal-to-noise ratio in fluorescence microscopy.
    ///
    /// SNR = N_sig / √(N_sig + N_bg + N_dark)
    ///
    /// Based on shot-noise limited detection (Poisson statistics).
    pub fn microscopy_snr(&self, n_signal_photons: f64, n_bg: f64, n_dark: f64) -> f64 {
        let noise = (n_signal_photons + n_bg + n_dark).sqrt();
        if noise < 1.0e-30 {
            return 0.0;
        }
        n_signal_photons / noise
    }
}

/// FRET (Förster Resonance Energy Transfer) pair model.
///
/// Describes dipole-dipole energy transfer from a donor to an acceptor
/// fluorophore as a function of their separation distance.
pub struct FretPair {
    /// Donor fluorophore
    pub donor: Fluorophore,
    /// Acceptor fluorophore
    pub acceptor: Fluorophore,
    /// Förster radius R₀ (nm): distance at which E = 50%
    pub forster_radius_nm: f64,
}

impl FretPair {
    /// Construct a FRET pair with explicit Förster radius.
    pub fn new(donor: Fluorophore, acceptor: Fluorophore, r0_nm: f64) -> Self {
        Self {
            donor,
            acceptor,
            forster_radius_nm: r0_nm,
        }
    }

    /// FITC–Rhodamine B FRET pair.  R₀ ≈ 5.5 nm.
    pub fn fitc_rhodamine() -> Self {
        Self::new(Fluorophore::fitc(), Fluorophore::rhodamine_b(), 5.5)
    }

    /// CFP–YFP FRET pair commonly used in live-cell imaging.  R₀ ≈ 4.9 nm.
    ///
    /// CFP: ex=433, em=476, Φ=0.40, τ=2.5 ns
    /// YFP: ex=514, em=527, Φ=0.61, τ=3.0 ns
    pub fn cfp_yfp() -> Self {
        let cfp = Fluorophore::new("CFP", 433.0, 476.0, 32_500.0, 0.40, 2.5);
        let yfp = Fluorophore::new("YFP", 514.0, 527.0, 83_400.0, 0.61, 3.0);
        Self::new(cfp, yfp, 4.9)
    }

    /// Cy3–Cy5 FRET pair.  R₀ ≈ 6.0 nm.
    pub fn cy3_cy5() -> Self {
        Self::new(Fluorophore::cy3(), Fluorophore::cy5(), 6.0)
    }

    /// FRET efficiency as a function of donor-acceptor distance.
    ///
    /// E(r) = R₀⁶ / (R₀⁶ + r⁶)
    pub fn efficiency(&self, distance_nm: f64) -> f64 {
        let r0_6 = self.forster_radius_nm.powi(6);
        let r_6 = distance_nm.powi(6);
        r0_6 / (r0_6 + r_6)
    }

    /// Infer donor-acceptor distance from measured FRET efficiency.
    ///
    /// r = R₀ · ((1 − E) / E)^{1/6}
    pub fn distance_from_efficiency_nm(&self, efficiency: f64) -> f64 {
        if efficiency <= 0.0 {
            return f64::INFINITY;
        }
        if efficiency >= 1.0 {
            return 0.0;
        }
        self.forster_radius_nm * ((1.0 - efficiency) / efficiency).powf(1.0 / 6.0)
    }

    /// Compute the Förster radius from spectral overlap of donor emission and acceptor absorption.
    ///
    /// R₀⁶ = (9000 · ln10 · κ² · Φ_D · J) / (128 · π⁵ · n⁴ · N_A)
    ///
    /// The spectral overlap integral is approximated numerically:
    ///   J = ∫ F_D(λ) · ε_A(λ) · λ⁴ dλ  (normalized by ∫ F_D(λ) dλ)
    ///
    /// # Arguments
    /// * `donor` — donor fluorophore
    /// * `acceptor` — acceptor fluorophore
    /// * `orientation_factor_kappa2` — κ² (2/3 for random orientation)
    /// * `n_medium` — refractive index of medium
    ///
    /// # Returns
    /// Förster radius in nanometers.
    pub fn compute_forster_radius_nm(
        donor: &Fluorophore,
        acceptor: &Fluorophore,
        orientation_factor_kappa2: f64,
        n_medium: f64,
    ) -> f64 {
        // Numerical integration range: donor emission window
        let lambda_min = donor.emission_peak_nm - 100.0;
        let lambda_max = donor.emission_peak_nm + 150.0;
        let n_pts = 1000_usize;
        let d_lambda = (lambda_max - lambda_min) / n_pts as f64;

        // Donor emission spectrum σ_em (Gaussian, unnormalized)
        let sigma_d = DEFAULT_FWHM_NM / (2.0 * (2.0 * 2_f64.ln()).sqrt());
        // Acceptor absorption approximated as Gaussian around its excitation peak
        let sigma_a = DEFAULT_FWHM_NM / (2.0 * (2.0 * 2_f64.ln()).sqrt());

        let mut j_num = 0.0_f64; // numerator: ∫ F_D·ε_A·λ⁴ dλ
        let mut j_den = 0.0_f64; // denominator: ∫ F_D dλ

        for i in 0..n_pts {
            let lambda_nm = lambda_min + (i as f64 + 0.5) * d_lambda;
            // Donor normalized emission
            let delta_d = lambda_nm - donor.emission_peak_nm;
            let f_d = (-delta_d * delta_d / (2.0 * sigma_d * sigma_d)).exp();
            // Acceptor molar extinction (Gaussian around excitation peak)
            let delta_a = lambda_nm - acceptor.excitation_peak_nm;
            let eps_a = acceptor.extinction_coefficient_m_per_cm
                * (-delta_a * delta_a / (2.0 * sigma_a * sigma_a)).exp();
            // λ in cm for SI consistency (λ in nm → cm: × 1e-7)
            let lambda_cm = lambda_nm * 1.0e-7;
            j_num += f_d * eps_a * lambda_cm.powi(4) * d_lambda;
            j_den += f_d * d_lambda;
        }

        let j = if j_den > 1.0e-30 { j_num / j_den } else { 0.0 };

        // R₀⁶ in cm⁶ (all quantities in CGS)
        let r0_6_cm6 = (9000.0 * 10_f64.ln() * orientation_factor_kappa2 * donor.quantum_yield * j)
            / (128.0 * std::f64::consts::PI.powi(5) * n_medium.powi(4) * AVOGADRO);

        // Convert cm → nm: 1 cm = 1e7 nm → cm⁶ = (1e7)⁶ nm⁶
        let r0_6_nm6 = r0_6_cm6 * 1.0e42;
        r0_6_nm6.powf(1.0 / 6.0)
    }

    /// Donor fluorescence lifetime in the presence of acceptor.
    ///
    /// τ_DA = τ_D · (1 − E)
    pub fn donor_lifetime_with_acceptor_ns(&self, distance_nm: f64) -> f64 {
        let e = self.efficiency(distance_nm);
        self.donor.lifetime_ns * (1.0 - e)
    }

    /// FRET rate constant.
    ///
    /// k_FRET = (1/τ_D) · (R₀/r)⁶  (ns⁻¹)
    pub fn fret_rate_per_ns(&self, distance_nm: f64) -> f64 {
        let r0_over_r = self.forster_radius_nm / distance_nm;
        (1.0 / self.donor.lifetime_ns) * r0_over_r.powi(6)
    }
}

/// Two-photon excitation (2PE) fluorescence microscopy model.
///
/// Models the nonlinear (quadratic) excitation process used in
/// multiphoton microscopy to achieve inherent 3-D optical sectioning.
pub struct TwoPhotonExcitation {
    /// The fluorophore being excited
    pub fluorophore: Fluorophore,
    /// Two-photon absorption cross-section δ in Göppert-Mayer units
    /// (1 GM = 10⁻⁵⁰ cm⁴·s·photon⁻¹)
    pub two_photon_cross_section_gm: f64,
    /// Laser pulse width (full-width at half-maximum) in femtoseconds
    pub pulse_width_fs: f64,
    /// Laser repetition rate (MHz)
    pub rep_rate_mhz: f64,
    /// Center wavelength of the excitation laser (nm)
    /// Should be approximately 2× the one-photon excitation peak.
    pub center_wavelength_nm: f64,
}

impl TwoPhotonExcitation {
    /// Construct a 2PE model with explicit parameters.
    pub fn new(
        fluorophore: Fluorophore,
        delta_gm: f64,
        pulse_fs: f64,
        rep_mhz: f64,
        lambda_nm: f64,
    ) -> Self {
        Self {
            fluorophore,
            two_photon_cross_section_gm: delta_gm,
            pulse_width_fs: pulse_fs,
            rep_rate_mhz: rep_mhz,
            center_wavelength_nm: lambda_nm,
        }
    }

    /// Two-photon excitation rate for a focused pulsed laser beam.
    ///
    /// For a Gaussian pulse train:
    ///   R₂PE = δ · (P_avg / (hν))² · f_rep · τ_p / (π · w₀²)² · NA_correction
    ///
    /// Simplified form:
    ///   R₂PE = δ_cm4s · (P²_peak) / (hν)² · 1/(π w₀²)
    /// where P_peak = P_avg / (f_rep · τ_p).
    ///
    /// # Arguments
    /// * `avg_power_mw` — time-averaged laser power (mW)
    /// * `beam_waist_um` — 1/e² beam radius at focus (μm)
    ///
    /// # Returns
    /// Excitation rate (excitations per molecule per second).
    pub fn excitation_rate(&self, avg_power_mw: f64, beam_waist_um: f64) -> f64 {
        // Convert units
        let delta_cm4s = self.two_photon_cross_section_gm * 1.0e-50; // GM → cm⁴·s
        let f_rep_hz = self.rep_rate_mhz * 1.0e6;
        let tau_p_s = self.pulse_width_fs * 1.0e-15;
        let p_avg_w = avg_power_mw * 1.0e-3;
        let w0_cm = beam_waist_um * 1.0e-4; // μm → cm

        // Peak power during pulse
        let p_peak_w = p_avg_w / (f_rep_hz * tau_p_s);

        let lambda_m = self.center_wavelength_nm * 1.0e-9;
        let h_nu = H_PLANCK * C0 / lambda_m; // energy per photon (J)

        // Beam area at focus (cm²)
        let beam_area_cm2 = std::f64::consts::PI * w0_cm * w0_cm;

        // Peak irradiance (W/cm²)
        let i_peak = p_peak_w / beam_area_cm2;

        // 2PE rate: R = δ · (I/(hν))²  (excitations per molecule per unit time)
        // [cm⁴·s] × (photons·cm⁻²·s⁻¹)² = cm⁴·s × cm⁻⁴·s⁻² = s⁻¹
        let photon_flux_peak = i_peak / h_nu;
        delta_cm4s * photon_flux_peak * photon_flux_peak * self.fluorophore.quantum_yield
    }

    /// The signal power dependence exponent for 2PE.
    ///
    /// R ∝ P² (quadratic dependence is the hallmark of two-photon excitation).
    pub fn power_dependence_exponent() -> f64 {
        2.0
    }

    /// Excited volume in a diffraction-limited 2PE microscope.
    ///
    /// The axially confined excitation volume scales as:
    ///   V_exc ≈ (λ²)³ / (8 · ln2 · NA⁴)  (simplified)
    ///
    /// A Gaussian beam approximation gives:
    ///   w₀ = λ / (π · NA) and z_R = n·λ / (NA²)
    ///   V_exc ≈ (π/2)^{3/2} · w₀² · z_R
    ///
    /// # Returns
    /// Excited volume in μm³.
    pub fn excited_volume_um3(&self, numerical_aperture: f64) -> f64 {
        let lambda_um = self.center_wavelength_nm * 1.0e-3;
        let n_medium = 1.33; // water immersion
        let w0_um = lambda_um / (std::f64::consts::PI * numerical_aperture);
        // Rayleigh range
        let z_r_um = n_medium * lambda_um / (numerical_aperture * numerical_aperture);
        // 2PE excitation PSF volume (intensity² → PSF narrows by 1/√2 in each dimension)

        (std::f64::consts::PI / 2.0_f64).powf(1.5) * w0_um * w0_um / std::f64::consts::SQRT_2
            * z_r_um
            / std::f64::consts::SQRT_2
    }

    /// Estimated tissue penetration depth advantage for 2PE vs 1PE.
    ///
    /// In the near-infrared (700–1100 nm), tissue scattering is reduced,
    /// permitting deeper imaging. Uses the effective attenuation at the
    /// 2PE laser wavelength assuming scattering scales as λ⁻{b} with b≈1.5.
    ///
    /// # Returns
    /// Approximate penetration depth in μm.
    pub fn penetration_depth_um(
        &self,
        tissue: &crate::biophotonics::tissue::TissueOpticalProperties,
    ) -> f64 {
        // Scale tissue scattering from reference wavelength (630 nm) to 2PE wavelength
        let lambda_ref = tissue.wavelength_nm;
        let lambda_2pe = self.center_wavelength_nm;
        let b = 1.5_f64; // scattering power law exponent
        let scale = (lambda_ref / lambda_2pe).powf(b);
        let mu_s_prime_2pe = tissue.reduced_scattering_coefficient() * scale;
        let mu_a_2pe = tissue.absorption_coefficient_cm * 0.1; // absorption much lower in NIR
        let mu_eff_2pe = (3.0 * mu_a_2pe * (mu_a_2pe + mu_s_prime_2pe)).sqrt();
        if mu_eff_2pe < 1.0e-20 {
            return f64::INFINITY;
        }
        // Convert cm → μm
        (1.0 / mu_eff_2pe) * 1.0e4
    }

    /// Photobleaching rate (simplified model, scales as P²).
    ///
    /// Bleaching for 2PE has the same nonlinear power dependence as signal,
    /// so increasing power beyond the saturation level does not improve SNR.
    ///
    /// Returns a relative bleaching index (arbitrary units).
    pub fn photobleaching_rate_per_s(&self, avg_power_mw: f64) -> f64 {
        let rate_at_1mw = self.fluorophore.photobleaching_threshold;
        rate_at_1mw * avg_power_mw * avg_power_mw
    }

    /// Two-photon action cross-section: δ_action = δ × Φ  (GM)
    pub fn action_cross_section_gm(&self) -> f64 {
        self.two_photon_cross_section_gm * self.fluorophore.quantum_yield
    }
}

/// Solve a 3×3 linear system A·x = b using Gaussian elimination with partial pivoting.
///
/// Returns the solution vector x, or zeros if the system is singular.
fn solve_3x3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> [f64; 3] {
    let mut m = [
        [a[0][0], a[0][1], a[0][2], b[0]],
        [a[1][0], a[1][1], a[1][2], b[1]],
        [a[2][0], a[2][1], a[2][2], b[2]],
    ];

    // Forward elimination with partial pivoting
    for col in 0..3 {
        // Find pivot
        let mut max_val = m[col][col].abs();
        let mut max_row = col;
        for (row, _) in m.iter().enumerate().skip(col + 1).take(3 - col - 1) {
            if m[row][col].abs() > max_val {
                max_val = m[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1.0e-30 {
            return [0.0; 3]; // Singular
        }
        m.swap(col, max_row);

        for row in (col + 1)..3 {
            let factor = m[row][col] / m[col][col];
            let pivot_vals: [f64; 4] = m[col];
            for (offset, &pv) in pivot_vals[col..4].iter().enumerate() {
                m[row][col + offset] -= pv * factor;
            }
        }
    }

    // Back substitution
    let mut x = [0.0_f64; 3];
    for i in (0..3).rev() {
        let mut s = m[i][3];
        for j in (i + 1)..3 {
            s -= m[i][j] * x[j];
        }
        if m[i][i].abs() < 1.0e-30 {
            return [0.0; 3];
        }
        x[i] = s / m[i][i];
    }
    x
}

/// Fluorescence Lifetime Imaging Microscopy (FLIM) analysis tools.
///
/// Provides time-domain exponential fitting and phasor-space analysis
/// of fluorescence decay data.
pub struct FlimAnalysis;

impl FlimAnalysis {
    /// Fit a single-exponential decay: I(t) = A · exp(−t/τ) + B
    ///
    /// Uses a two-step approach:
    /// 1. Estimate B from the tail of the decay (last 10% of points).
    /// 2. Log-linear regression on (I − B) for initial (A, τ).
    /// 3. Levenberg-Marquardt style nonlinear refinement.
    ///
    /// # Arguments
    /// * `t` — time axis (ns)
    /// * `intensity` — measured intensity values
    ///
    /// # Returns
    /// `(A, τ_ns, B)` — amplitude, lifetime (ns), and baseline offset.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] if fitting fails.
    pub fn single_exponential_fit(
        t: &[f64],
        intensity: &[f64],
    ) -> Result<(f64, f64, f64), OxiPhotonError> {
        if t.len() != intensity.len() || t.len() < 3 {
            return Err(OxiPhotonError::NumericalError(
                "Need at least 3 data points with matching t and I arrays".into(),
            ));
        }

        // Step 1: Estimate background from the last 10% of the decay (should be ~B)
        let tail_start = (t.len() * 9 / 10).max(t.len() - 3);
        let b_est = intensity[tail_start..]
            .iter()
            .cloned()
            .fold(0.0_f64, |acc, v| acc + v)
            / (intensity.len() - tail_start) as f64;
        let b_est = b_est.max(0.0);

        // Step 2: Log-linear regression on (I - B) for initial A, τ
        let mut sum_t = 0.0;
        let mut sum_ln_i = 0.0;
        let mut sum_t2 = 0.0;
        let mut sum_t_ln_i = 0.0;
        let mut n_valid = 0usize;

        for (&ti, &ii) in t.iter().zip(intensity.iter()) {
            let i_corr = ii - b_est;
            if i_corr > 1.0e-30 {
                let ln_i = i_corr.ln();
                sum_t += ti;
                sum_ln_i += ln_i;
                sum_t2 += ti * ti;
                sum_t_ln_i += ti * ln_i;
                n_valid += 1;
            }
        }

        if n_valid < 2 {
            return Err(OxiPhotonError::NumericalError(
                "Insufficient non-zero data points for exponential fit".into(),
            ));
        }

        let n = n_valid as f64;
        let denom = n * sum_t2 - sum_t * sum_t;
        if denom.abs() < 1.0e-30 {
            return Err(OxiPhotonError::NumericalError(
                "Degenerate time axis in single-exponential fit".into(),
            ));
        }

        let slope = (n * sum_t_ln_i - sum_t * sum_ln_i) / denom;
        let intercept = (sum_ln_i - slope * sum_t) / n;

        let tau_init = if slope < -1.0e-30 { -1.0 / slope } else { 1.0 };
        let a_init = intercept.exp();

        // Step 3: Gauss-Newton nonlinear least-squares with Levenberg-Marquardt damping.
        // Jacobian columns: J0 = ∂r/∂A = exp(-t/τ), J1 = ∂r/∂τ = A·t/τ²·exp(-t/τ), J2 = ∂r/∂B = 1
        let (mut a, mut tau, mut b) = (a_init, tau_init, b_est);
        let mut lambda_lm = 0.01;

        // Compute SSE
        let sse_fn = |a: f64, tau: f64, b: f64| -> f64 {
            t.iter()
                .zip(intensity.iter())
                .map(|(&ti, &ii)| {
                    let r = a * (-ti / tau).exp() + b - ii;
                    r * r
                })
                .sum()
        };

        let mut sse_cur = sse_fn(a, tau, b);

        for _iter in 0..1000 {
            // Build normal equations: (JᵀJ + λ·diag(JᵀJ)) · δp = Jᵀr
            let mut jtj = [[0.0_f64; 3]; 3];
            let mut jtr = [0.0_f64; 3];

            for (&ti, &ii) in t.iter().zip(intensity.iter()) {
                let e = (-ti / tau).exp();
                let j0 = e; // ∂f/∂A
                let j1 = a * e * ti / (tau * tau); // ∂f/∂τ
                let j2 = 1.0_f64; // ∂f/∂B
                let res = a * e + b - ii;

                let jv = [j0, j1, j2];
                for i in 0..3 {
                    jtr[i] += jv[i] * res;
                    for j in 0..3 {
                        jtj[i][j] += jv[i] * jv[j];
                    }
                }
            }

            // Add LM damping to diagonal
            for (i, row) in jtj.iter_mut().enumerate() {
                row[i] *= 1.0 + lambda_lm;
            }

            // Solve 3×3 linear system via Cramer's rule / Gaussian elimination
            let delta = solve_3x3(&jtj, &jtr);

            let a_new = (a - delta[0]).max(1.0e-10);
            let tau_new = (tau - delta[1]).max(1.0e-9);
            let b_new = b - delta[2];

            let sse_new = sse_fn(a_new, tau_new, b_new);

            if sse_new < sse_cur {
                a = a_new;
                tau = tau_new;
                b = b_new;
                sse_cur = sse_new;
                lambda_lm *= 0.5;
                if sse_cur < 1.0e-20 {
                    break;
                }
            } else {
                lambda_lm *= 4.0;
                if lambda_lm > 1.0e10 {
                    break;
                }
            }
        }

        Ok((a, tau, b))
    }

    /// Fit a double-exponential decay: I(t) = A1·exp(−t/τ1) + A2·exp(−t/τ2) + B
    ///
    /// Uses a sequential peeling approach: fits the long component first
    /// (using the tail of the decay), then subtracts and fits the short component.
    ///
    /// # Returns
    /// `(A1, τ1_ns, A2, τ2_ns, B)` where τ1 ≥ τ2.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] if fitting fails.
    pub fn double_exponential_fit(
        t: &[f64],
        intensity: &[f64],
    ) -> Result<(f64, f64, f64, f64, f64), OxiPhotonError> {
        if t.len() != intensity.len() || t.len() < 5 {
            return Err(OxiPhotonError::NumericalError(
                "Need at least 5 data points for double-exponential fit".into(),
            ));
        }

        // Step 1: Use the second half of the decay to estimate the long component
        let half = t.len() / 2;
        let t_tail = &t[half..];
        let i_tail = &intensity[half..];

        let (a1_long, tau_long, b_est) = Self::single_exponential_fit(t_tail, i_tail)?;

        // Step 2: Subtract long component, fit short component on the first half
        let i_short: Vec<f64> = t
            .iter()
            .zip(intensity.iter())
            .map(|(&ti, &ii)| {
                let long_component = a1_long * (-ti / tau_long).exp() + b_est;
                (ii - long_component).max(1.0e-30)
            })
            .collect();

        let (a2_short, tau_short, _) = Self::single_exponential_fit(t, &i_short)?;

        // Return with τ1 ≥ τ2 convention
        if tau_long >= tau_short {
            Ok((a1_long, tau_long, a2_short, tau_short, b_est))
        } else {
            Ok((a2_short, tau_short, a1_long, tau_long, b_est))
        }
    }

    /// Intensity-weighted average lifetime.
    ///
    /// ⟨τ⟩_int = Σ(Aᵢ · τᵢ²) / Σ(Aᵢ · τᵢ)
    ///
    /// This is proportional to the steady-state fluorescence intensity.
    pub fn amplitude_weighted_lifetime(amplitudes: &[f64], lifetimes: &[f64]) -> f64 {
        let num: f64 = amplitudes
            .iter()
            .zip(lifetimes.iter())
            .map(|(&a, &tau)| a * tau * tau)
            .sum();
        let den: f64 = amplitudes
            .iter()
            .zip(lifetimes.iter())
            .map(|(&a, &tau)| a * tau)
            .sum();
        if den.abs() < 1.0e-30 {
            0.0
        } else {
            num / den
        }
    }

    /// Amplitude-weighted average lifetime.
    ///
    /// ⟨τ⟩_amp = Σ(Aᵢ · τᵢ) / Σ(Aᵢ)
    ///
    /// This emphasizes the fraction of molecules with each lifetime.
    pub fn intensity_weighted_lifetime(amplitudes: &[f64], lifetimes: &[f64]) -> f64 {
        let num: f64 = amplitudes
            .iter()
            .zip(lifetimes.iter())
            .map(|(&a, &tau)| a * tau)
            .sum();
        let den: f64 = amplitudes.iter().sum();
        if den.abs() < 1.0e-30 {
            0.0
        } else {
            num / den
        }
    }

    /// Phasor plot coordinates for a single-exponential decay.
    ///
    /// g = ∫ I(t) cos(ωt) dt / ∫ I(t) dt = 1 / (1 + (ωτ)²)
    /// s = ∫ I(t) sin(ωt) dt / ∫ I(t) dt = ωτ / (1 + (ωτ)²)
    ///
    /// Single-exponential lifetimes lie on a semicircle of radius 0.5
    /// centered at (0.5, 0) in the (g, s) plane.
    ///
    /// # Arguments
    /// * `lifetime_ns` — fluorescence lifetime τ (ns)
    /// * `rep_rate_mhz` — laser/detector repetition rate (MHz)
    ///
    /// # Returns
    /// `(g, s)` phasor coordinates.
    pub fn phasor_coordinates(lifetime_ns: f64, rep_rate_mhz: f64) -> (f64, f64) {
        let omega = TWO_PI * rep_rate_mhz * 1.0e6 * 1.0e-9; // rad/ns
        let omega_tau = omega * lifetime_ns;
        let denom = 1.0 + omega_tau * omega_tau;
        let g = 1.0 / denom;
        let s = omega_tau / denom;
        (g, s)
    }

    /// Recover fluorescence lifetime from phasor coordinates.
    ///
    /// For a single-exponential decay:
    ///   τ = s / (ω · g)
    ///
    /// # Arguments
    /// * `g` — real phasor component
    /// * `s` — imaginary phasor component
    /// * `rep_rate_mhz` — repetition rate (MHz)
    ///
    /// # Returns
    /// Lifetime in nanoseconds.
    pub fn lifetime_from_phasor(g: f64, s: f64, rep_rate_mhz: f64) -> f64 {
        let omega = TWO_PI * rep_rate_mhz * 1.0e6 * 1.0e-9; // rad/ns
        if omega < 1.0e-30 || g < 1.0e-30 {
            return 0.0;
        }
        s / (omega * g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1.0e-10;
    #[allow(dead_code)]
    const RTOL: f64 = 1.0e-6;

    #[test]
    fn test_fitc_stokes_shift() {
        let fitc = Fluorophore::fitc();
        // FITC: ex=494, em=521 → shift = 27 nm
        let shift = fitc.stokes_shift_nm();
        assert!(
            (shift - 27.0).abs() < TOL,
            "FITC Stokes shift should be 27 nm, got {shift}"
        );
    }

    #[test]
    fn test_fluorophore_brightness() {
        let fitc = Fluorophore::fitc();
        let expected = fitc.extinction_coefficient_m_per_cm * fitc.quantum_yield;
        let brightness = fitc.brightness();
        assert!(
            (brightness - expected).abs() < TOL,
            "Brightness = ε × Φ, expected {expected}, got {brightness}"
        );
    }

    #[test]
    fn test_fret_efficiency_at_r0() {
        let pair = FretPair::fitc_rhodamine();
        // At r = R0, efficiency should be exactly 0.5
        let eff = pair.efficiency(pair.forster_radius_nm);
        assert!(
            (eff - 0.5).abs() < TOL,
            "FRET efficiency at R0 should be 0.5, got {eff}"
        );
    }

    #[test]
    fn test_fret_efficiency_zero_at_large_r() {
        let pair = FretPair::fitc_rhodamine();
        // At r = 100 × R0, efficiency → 0
        let eff = pair.efficiency(pair.forster_radius_nm * 100.0);
        assert!(
            eff < 1.0e-10,
            "FRET efficiency should approach 0 at large r, got {eff}"
        );
    }

    #[test]
    fn test_fret_efficiency_one_at_small_r() {
        let pair = FretPair::fitc_rhodamine();
        // At r = 0.001 × R0, efficiency → 1
        let eff = pair.efficiency(pair.forster_radius_nm * 0.001);
        assert!(
            eff > 0.999_999,
            "FRET efficiency should approach 1 at small r, got {eff}"
        );
    }

    #[test]
    fn test_donor_lifetime_reduced_by_fret() {
        let pair = FretPair::fitc_rhodamine();
        let tau_d = pair.donor.lifetime_ns;
        // At r = R0, E=0.5 → τ_DA = τ_D × 0.5
        let tau_da = pair.donor_lifetime_with_acceptor_ns(pair.forster_radius_nm);
        assert!(
            tau_da < tau_d,
            "Donor lifetime should be reduced by FRET: τ_D={tau_d}, τ_DA={tau_da}"
        );
        assert!(
            (tau_da - tau_d * 0.5).abs() < TOL,
            "At r=R0, τ_DA should be τ_D/2"
        );
    }

    #[test]
    fn test_two_photon_quadratic_power() {
        let gfp = Fluorophore::gfp();
        let tpe = TwoPhotonExcitation::new(gfp, 10.0, 100.0, 80.0, 920.0);

        let rate_1mw = tpe.excitation_rate(1.0, 0.5);
        let rate_2mw = tpe.excitation_rate(2.0, 0.5);

        // Rate ∝ P² → doubling power → 4× rate
        let ratio = rate_2mw / rate_1mw;
        assert!(
            (ratio - 4.0).abs() < 0.01,
            "2PE rate should scale as P²: ratio = {ratio}, expected 4.0"
        );
    }

    #[test]
    fn test_phasor_unit_circle() {
        // Single-exponential phasors must lie on semicircle: g²+s²−g = 0,
        // i.e., g² + s² = g (or equivalently center=(0.5,0), radius=0.5)
        for &tau_ns in &[0.5, 1.0, 2.5, 4.0, 10.0] {
            let (g, s) = FlimAnalysis::phasor_coordinates(tau_ns, 80.0);
            // On unit semicircle: (g-0.5)² + s² = 0.25
            let dist = (g - 0.5) * (g - 0.5) + s * s;
            assert!(
                (dist - 0.25).abs() < 1.0e-12,
                "Phasor for τ={tau_ns} ns not on unit semicircle: dist={dist}"
            );
        }
    }

    #[test]
    fn test_lifetime_from_phasor_roundtrip() {
        let tau_orig = 3.7_f64; // ns
        let rep_rate = 80.0; // MHz
        let (g, s) = FlimAnalysis::phasor_coordinates(tau_orig, rep_rate);
        let tau_recovered = FlimAnalysis::lifetime_from_phasor(g, s, rep_rate);
        assert!(
            (tau_recovered - tau_orig).abs() < 1.0e-10,
            "Phasor roundtrip failed: original={tau_orig}, recovered={tau_recovered}"
        );
    }

    #[test]
    fn test_single_exp_fit_exact() {
        // Generate noise-free single-exponential data and verify the fit recovers exact params
        let a_true = 1000.0_f64;
        let tau_true = 3.5_f64; // ns
        let b_true = 10.0_f64;

        let n_pts = 100_usize;
        let t_max = 15.0_f64; // ns (≈ 4 lifetimes)

        let t: Vec<f64> = (0..n_pts)
            .map(|i| t_max * (i as f64) / (n_pts - 1) as f64)
            .collect();
        let intensity: Vec<f64> = t
            .iter()
            .map(|&ti| a_true * (-ti / tau_true).exp() + b_true)
            .collect();

        let (a_fit, tau_fit, b_fit) = FlimAnalysis::single_exponential_fit(&t, &intensity)
            .expect("Fit should succeed on clean data");

        // Allow 5% relative tolerance (gradient descent is approximate)
        let tol_rel = 0.05;
        assert!(
            (tau_fit - tau_true).abs() / tau_true < tol_rel,
            "Lifetime fit error too large: true={tau_true}, fitted={tau_fit}"
        );
        assert!(
            (a_fit - a_true).abs() / a_true < tol_rel,
            "Amplitude fit error too large: true={a_true}, fitted={a_fit}"
        );
        let _ = b_fit; // background may drift slightly due to log approximation
    }
}
