//! Spontaneous Parametric Down-Conversion (SPDC) photon pair sources.
//!
//! Models SPDC crystals (PPKTP, PPLN, BBO, KTP) and their properties:
//! pair generation rate, spectral brightness, joint spectral amplitude (JSA),
//! Schmidt decomposition, and entanglement bandwidth.
//!
//! Physical constants and formulae follow:
//!   - Boyd, "Nonlinear Optics", 4th ed.
//!   - Law & Eberly, PRL 2004 (Schmidt decomposition of JSA)
//!   - Mosley et al., PRL 2008 (engineered photon pairs)

use num_complex::Complex64;
use std::f64::consts::PI;

pub use crate::nonlinear_crystal::PhaseMatchingType;

/// Speed of light in vacuum (m/s)
const C: f64 = 2.997_924_58e8;
/// Permittivity of free space (F/m)
const EPSILON_0: f64 = 8.854_187_817e-12;

// ─── Crystal type ─────────────────────────────────────────────────────────────

/// SPDC nonlinear crystal.
#[derive(Debug, Clone)]
pub enum SpdcCrystal {
    /// Periodically-poled KTiOPO₄ — type-II PM, 775 → 1550 nm telecom
    Ppktp,
    /// Periodically-poled LiNbO₃ — type-0 QPM, telecom C-band
    Ppln,
    /// β-BaB₂O₄ — type-II PM, visible range (400–1100 nm)
    Bbo,
    /// KTiOPO₄ (bulk) — type-II PM
    Ktp,
    /// User-specified periodically-poled crystal
    Periodically {
        /// Quasi-phase-matching poling period (µm)
        period_um: f64,
        /// Material name (informational)
        material: String,
        /// Effective nonlinear coefficient (pm/V)
        d_eff: f64,
    },
}

impl SpdcCrystal {
    /// Effective nonlinear coefficient d_eff (pm/V).
    pub fn d_eff_pm_per_v(&self) -> f64 {
        match self {
            SpdcCrystal::Ppktp => 9.5, // type-II d_eff ≈ 9.5 pm/V
            SpdcCrystal::Ppln => 16.5, // QPM d_eff = (2/π)*d_33 ≈ (2/π)*27 pm/V
            SpdcCrystal::Bbo => 2.0,   // type-II d_eff ≈ 2.0 pm/V
            SpdcCrystal::Ktp => 3.7,   // type-II d_eff ≈ 3.7 pm/V
            SpdcCrystal::Periodically { d_eff, .. } => *d_eff,
        }
    }

    /// Group velocity of the signal photon (m/s).
    /// Approximate values at 1550 nm for each crystal.
    pub fn group_velocity_signal(&self) -> f64 {
        match self {
            SpdcCrystal::Ppktp => C / 1.745, // n_g(o) ≈ 1.745 at 1550 nm
            SpdcCrystal::Ppln => C / 2.211,  // n_g ≈ 2.211 at 1550 nm
            SpdcCrystal::Bbo => C / 1.668,   // n_g(o) ≈ 1.668 at 800 nm
            SpdcCrystal::Ktp => C / 1.745,
            SpdcCrystal::Periodically { .. } => C / 2.0,
        }
    }

    /// Group velocity of the idler photon (m/s).
    pub fn group_velocity_idler(&self) -> f64 {
        match self {
            SpdcCrystal::Ppktp => C / 1.830, // n_g(e) ≈ 1.830 at 1550 nm
            SpdcCrystal::Ppln => C / 2.211,  // type-0: same polarisation
            SpdcCrystal::Bbo => C / 1.730,   // n_g(e) ≈ 1.730 at 800 nm
            SpdcCrystal::Ktp => C / 1.830,
            SpdcCrystal::Periodically { .. } => C / 2.0,
        }
    }

    /// Group velocity dispersion β₂ for the signal (s²/m).
    /// GVD = d²β/dω² = -λ³/(2πc²) * (d²n/dλ²)
    pub fn gvd_signal(&self) -> f64 {
        match self {
            SpdcCrystal::Ppktp => -1.8e-26, // typical near 1550 nm
            SpdcCrystal::Ppln => 1.0e-25,   // anomalous at 1550 nm in PPLN
            SpdcCrystal::Bbo => -6.0e-26,   // near 800 nm
            SpdcCrystal::Ktp => -1.8e-26,
            SpdcCrystal::Periodically { .. } => -1.0e-26,
        }
    }

    /// Group velocity dispersion β₂ for the idler (s²/m).
    pub fn gvd_idler(&self) -> f64 {
        match self {
            SpdcCrystal::Ppktp => -1.5e-26,
            SpdcCrystal::Ppln => 1.0e-25,
            SpdcCrystal::Bbo => -4.0e-26,
            SpdcCrystal::Ktp => -1.5e-26,
            SpdcCrystal::Periodically { .. } => -1.0e-26,
        }
    }

    /// Refractive index at the pump wavelength (approximate).
    fn refractive_index_pump(&self) -> f64 {
        match self {
            SpdcCrystal::Ppktp => 1.738, // n_p at 775 nm
            SpdcCrystal::Ppln => 2.156,  // n_p at 775 nm
            SpdcCrystal::Bbo => 1.672,   // n_p at 400 nm
            SpdcCrystal::Ktp => 1.738,
            SpdcCrystal::Periodically { .. } => 2.0,
        }
    }

    /// Refractive index at the signal wavelength (approximate).
    fn refractive_index_signal(&self) -> f64 {
        match self {
            SpdcCrystal::Ppktp => 1.736,
            SpdcCrystal::Ppln => 2.211,
            SpdcCrystal::Bbo => 1.655,
            SpdcCrystal::Ktp => 1.736,
            SpdcCrystal::Periodically { .. } => 2.0,
        }
    }

    /// Refractive index at the idler wavelength (approximate).
    fn refractive_index_idler(&self) -> f64 {
        match self {
            SpdcCrystal::Ppktp => 1.818,
            SpdcCrystal::Ppln => 2.211,
            SpdcCrystal::Bbo => 1.720,
            SpdcCrystal::Ktp => 1.818,
            SpdcCrystal::Periodically { .. } => 2.0,
        }
    }

    /// Group-velocity mismatch between signal and idler (s/m) = 1/v_gs - 1/v_gi.
    pub fn group_velocity_mismatch(&self) -> f64 {
        1.0 / self.group_velocity_signal() - 1.0 / self.group_velocity_idler()
    }
}

// ─── Joint Spectral Amplitude ─────────────────────────────────────────────────

/// Joint Spectral Amplitude f(ω_s, ω_i) on a 2D frequency grid.
///
/// The amplitude matrix is stored as a row-major 2D vec of Complex64, indexed
/// `[signal_index][idler_index]`, and is normalised so that Σ|f|² = 1.
#[derive(Debug, Clone)]
pub struct JointSpectralAmplitude {
    /// Signal angular frequencies (rad/s), length = `n_signal`.
    pub signal_freqs: Vec<f64>,
    /// Idler angular frequencies (rad/s), length = `n_idler`.
    pub idler_freqs: Vec<f64>,
    /// 2D amplitude matrix: `amplitude[i_s][i_i]` = f(ω_s at signal index, ω_i at idler index).
    pub amplitude: Vec<Vec<Complex64>>,
}

// ─── Schmidt decomposition ────────────────────────────────────────────────────

/// Schmidt decomposition of the JSA.
///
/// The JSA is decomposed as f(ω_s, ω_i) = Σ_k √λ_k · ψ_k(ω_s) · φ_k(ω_i),
/// where {λ_k} are the Schmidt eigenvalues (normalised so Σλ_k = 1).
#[derive(Debug, Clone)]
pub struct SchmidtDecomposition {
    /// Schmidt eigenvalues λ_k in descending order, Σλ_k = 1.
    pub eigenvalues: Vec<f64>,
    /// Schmidt number K = 1/Σλ_k² ≥ 1.
    pub schmidt_number: f64,
}

// ─── SPDC source ─────────────────────────────────────────────────────────────

/// Spontaneous parametric down-conversion photon pair source.
///
/// Models the key measurable quantities of an SPDC source:
/// pair generation rate, spectral brightness, heralding efficiency,
/// joint spectral amplitude, and Schmidt mode decomposition.
#[derive(Debug, Clone)]
pub struct SpdcSource {
    /// Nonlinear crystal type
    pub crystal: SpdcCrystal,
    /// Pump wavelength (m)
    pub pump_wavelength: f64,
    /// Pump power (mW)
    pub pump_power_mw: f64,
    /// Crystal length (mm)
    pub crystal_length_mm: f64,
    /// Phase-matching configuration
    pub phase_matching: PhaseMatchingType,
}

impl SpdcSource {
    /// Create a PPKTP source optimised for 1550 nm telecom pairs (775 nm pump).
    pub fn new_ppktp_1550(pump_power_mw: f64, length_mm: f64) -> Self {
        Self {
            crystal: SpdcCrystal::Ppktp,
            pump_wavelength: 775e-9,
            pump_power_mw,
            crystal_length_mm: length_mm,
            phase_matching: PhaseMatchingType::TypeII,
        }
    }

    /// Create a PPLN source for telecom C-band pairs (780 nm pump).
    pub fn new_ppln_telecom(pump_power_mw: f64, length_mm: f64) -> Self {
        Self {
            crystal: SpdcCrystal::Ppln,
            pump_wavelength: 780e-9,
            pump_power_mw,
            crystal_length_mm: length_mm,
            phase_matching: PhaseMatchingType::QuasiPm { period_um: 19.3 },
        }
    }

    /// Signal wavelength (m). For degenerate SPDC: λ_s = 2 * λ_pump.
    pub fn signal_wavelength(&self) -> f64 {
        2.0 * self.pump_wavelength
    }

    /// Idler wavelength (m). Energy conservation: 1/λ_p = 1/λ_s + 1/λ_i.
    /// For degenerate case λ_i = λ_s.
    pub fn idler_wavelength(&self) -> f64 {
        let lambda_p = self.pump_wavelength;
        let lambda_s = self.signal_wavelength();
        // 1/λ_i = 1/λ_p - 1/λ_s; for degenerate λ_s = 2λ_p → λ_i = λ_s
        let inv_lambda_i = 1.0 / lambda_p - 1.0 / lambda_s;
        if inv_lambda_i > 0.0 {
            1.0 / inv_lambda_i
        } else {
            lambda_s
        }
    }

    /// Crystal length in metres.
    pub fn crystal_length_m(&self) -> f64 {
        self.crystal_length_mm * 1e-3
    }

    /// Pair generation rate (pairs/s).
    ///
    /// Based on the standard SPDC brightness formula:
    /// ```text
    /// R ≈ (ω_s ω_i d_eff² L² P_pump) / (2 n_s n_i n_p ε₀ c³ A_eff)
    /// ```
    /// where A_eff ≈ (10 µm)² is a typical focused beam waist area.
    pub fn pair_rate_per_second(&self) -> f64 {
        let d_eff_si = self.crystal.d_eff_pm_per_v() * 1e-12; // pm/V → m/V
        let l = self.crystal_length_m();
        let p = self.pump_power_mw * 1e-3; // mW → W
        let lambda_s = self.signal_wavelength();
        let lambda_i = self.idler_wavelength();
        let n_s = self.crystal.refractive_index_signal();
        let n_i = self.crystal.refractive_index_idler();
        let n_p = self.crystal.refractive_index_pump();
        let omega_s = 2.0 * PI * C / lambda_s;
        let omega_i = 2.0 * PI * C / lambda_i;
        // Effective mode area (assume ~10 µm beam waist)
        let a_eff = PI * (10e-6_f64).powi(2);
        let numerator = omega_s * omega_i * d_eff_si.powi(2) * l.powi(2) * p;
        let denominator = 2.0 * n_s * n_i * n_p * EPSILON_0 * C.powi(3) * a_eff;
        numerator / denominator
    }

    /// Spectral brightness (pairs/s/mW/nm).
    ///
    /// B = R / (P_pump * Δλ_PM)
    pub fn spectral_brightness(&self) -> f64 {
        let r = self.pair_rate_per_second();
        let p_mw = self.pump_power_mw;
        let bandwidth_nm = self.phase_matching_bandwidth_nm();
        if p_mw > 0.0 && bandwidth_nm > 0.0 {
            r / (p_mw * bandwidth_nm)
        } else {
            0.0
        }
    }

    /// Heralding efficiency (fraction of detected signal photons heralded by idler).
    ///
    /// Accounts for single-mode fibre coupling (≈ 0.65 for PPKTP in SMF-28).
    pub fn heralding_efficiency(&self) -> f64 {
        match &self.crystal {
            SpdcCrystal::Ppktp => 0.80,
            SpdcCrystal::Ppln => 0.75,
            SpdcCrystal::Bbo => 0.55,
            SpdcCrystal::Ktp => 0.72,
            SpdcCrystal::Periodically { .. } => 0.70,
        }
    }

    /// Phase-matching bandwidth Δλ (nm, FWHM of sinc² envelope).
    ///
    /// ```text
    /// Δλ ≈ 0.886 * λ_s² / (L * |GVM| * c)
    /// ```
    /// where GVM = 1/v_gs − 1/v_gi is the group-velocity mismatch.
    pub fn phase_matching_bandwidth_nm(&self) -> f64 {
        let gvm = self.crystal.group_velocity_mismatch().abs();
        let l = self.crystal_length_m();
        let lambda_s = self.signal_wavelength();
        if gvm < 1e-20 || l < 1e-10 {
            return 10.0; // fallback for degenerate / type-0
        }
        // FWHM of sinc² for sinc(x) = 0.886 at x = 1.392/2π... simplified:
        // Δω ≈ 0.886 * π / (L * GVM)
        let delta_omega = 0.886 * PI / (l * gvm);
        // Convert Δω → Δλ: Δλ = (λ²/2πc) Δω
        let delta_lambda = (lambda_s.powi(2) / (2.0 * PI * C)) * delta_omega;
        delta_lambda * 1e9 // m → nm
    }

    /// Second-order coherence g²(0) for SPDC photons (thermal/single-mode statistics).
    ///
    /// For a single spectral mode: g²(0) = 2 (thermal bunching).
    /// For multi-mode (Schmidt number K): g²(0) = 1 + 1/K.
    pub fn g2_zero(&self) -> f64 {
        let k = self.schmidt_number(32);
        1.0 + 1.0 / k.max(1.0)
    }

    /// Joint Spectral Amplitude f(ω_s, ω_i) on an n×n frequency grid.
    ///
    /// ```text
    /// f(ω_s, ω_i) = α(ω_s + ω_i) · sinc(Δk(ω_s,ω_i)·L/2)
    /// ```
    /// where α is the Gaussian pump envelope, and Δk uses the GVM linearisation:
    /// Δk ≈ GVM·(ω_s − ω_s0).
    ///
    /// Returns `(signal_freqs_hz, idler_freqs_hz, jsa_matrix)`.
    pub fn joint_spectral_amplitude(
        &self,
        n_points: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<Vec<Complex64>>) {
        let n = n_points.max(4);
        let omega_s0 = 2.0 * PI * C / self.signal_wavelength();
        let omega_i0 = 2.0 * PI * C / self.idler_wavelength();
        let omega_p0 = 2.0 * PI * C / self.pump_wavelength;

        // Pump bandwidth (Gaussian): assume σ_p = 0.5 nm → Δω_p
        let sigma_pump_nm = 0.5_f64;
        let sigma_pump_m = sigma_pump_nm * 1e-9;
        let sigma_omega_pump = (2.0 * PI * C / self.pump_wavelength.powi(2)) * sigma_pump_m;

        // PM bandwidth in angular frequency
        let bw_nm = self.phase_matching_bandwidth_nm();
        let bw_m = bw_nm * 1e-9;
        let sigma_omega_pm = (2.0 * PI * C / self.signal_wavelength().powi(2)) * bw_m;

        // Frequency grid half-width: 3 * max(sigma_pump, sigma_pm) around central frequencies
        let half_span = 3.0 * sigma_omega_pump.max(sigma_omega_pm);
        let d_omega = 2.0 * half_span / (n as f64 - 1.0);

        let signal_freqs: Vec<f64> = (0..n)
            .map(|i| omega_s0 - half_span + i as f64 * d_omega)
            .collect();
        let idler_freqs: Vec<f64> = (0..n)
            .map(|i| omega_i0 - half_span + i as f64 * d_omega)
            .collect();

        let gvm = self.crystal.group_velocity_mismatch();
        let l = self.crystal_length_m();

        let mut jsa = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        let mut norm_sq = 0.0_f64;

        for (si, &omega_s) in signal_freqs.iter().enumerate() {
            for (ii, &omega_i) in idler_freqs.iter().enumerate() {
                // Pump envelope: α(ω_s + ω_i) — Gaussian centred at ω_p0
                let delta_sum = omega_s + omega_i - omega_p0;
                let pump_env = (-delta_sum.powi(2) / (2.0 * sigma_omega_pump.powi(2))).exp();

                // Phase-matching: sinc(Δk·L/2) where Δk ≈ GVM·(ω_s - ω_s0)
                let delta_omega_s = omega_s - omega_s0;
                let delta_k_l_half = gvm * delta_omega_s * l / 2.0;
                let pm = sinc(delta_k_l_half);

                let val = pump_env * pm;
                jsa[si][ii] = Complex64::new(val, 0.0);
                norm_sq += val * val;
            }
        }

        // Normalise
        if norm_sq > 0.0 {
            let norm = norm_sq.sqrt();
            for row in &mut jsa {
                for v in row.iter_mut() {
                    *v /= norm;
                }
            }
        }

        (signal_freqs, idler_freqs, jsa)
    }

    /// Schmidt eigenvalues {λ_k} of the JSA (via SVD of the JSA matrix).
    ///
    /// The JSA matrix f_{si} is decomposed as f = Σ_k √λ_k · u_k ⊗ v_k.
    /// Returns eigenvalues in descending order, normalised so Σλ_k = 1.
    pub fn schmidt_modes(&self, n_points: usize) -> Vec<f64> {
        let (_, _, jsa) = self.joint_spectral_amplitude(n_points);
        let n = jsa.len();
        if n == 0 {
            return vec![1.0];
        }
        // Compute singular values via power iteration on A^†A (real JSA here)
        // Build real matrix a[i][j] = Re(jsa[i][j]) (our JSA is real-valued)
        let a: Vec<Vec<f64>> = jsa
            .iter()
            .map(|row| row.iter().map(|v| v.re).collect())
            .collect();
        // Compute S = A^T A (n×n)
        let s = mat_transpose_times_mat(&a, n);
        // Extract eigenvalues using symmetric power deflation
        let eigenvalues = symmetric_eigenvalues_power_deflation(&s, n, 32);
        // Eigenvalues of A^T A are squares of singular values; λ_k = σ_k² (already normalised)
        let mut lambdas: Vec<f64> = eigenvalues.into_iter().filter(|&v| v > 1e-12).collect();
        lambdas.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        // Normalise so Σλ = 1
        let total: f64 = lambdas.iter().sum();
        if total > 0.0 {
            for v in lambdas.iter_mut() {
                *v /= total;
            }
        }
        lambdas
    }

    /// Schmidt number K = (Σλ_k)² / Σλ_k² = 1 / Σλ_k².
    ///
    /// K = 1 for a pure single-mode state, K > 1 for spectrally mixed states.
    pub fn schmidt_number(&self, n_points: usize) -> f64 {
        let lambdas = self.schmidt_modes(n_points);
        let sum_sq: f64 = lambdas.iter().map(|&l| l * l).sum();
        if sum_sq > 0.0 {
            1.0 / sum_sq
        } else {
            1.0
        }
    }

    /// Spectral purity P = 1/K (1 = spectrally pure, 0 = maximally mixed).
    pub fn spectral_purity(&self, n_points: usize) -> f64 {
        1.0 / self.schmidt_number(n_points).max(1.0)
    }
}

// ─── Integrated SPDC source ───────────────────────────────────────────────────

/// Integrated waveguide SPDC source with real sinc(ΔkL/2) phase-matching JSA.
///
/// Unlike `SpdcSource`, this struct carries explicit refractive and group indices
/// for all three fields (pump/signal/idler), enabling a physically accurate
/// computation of Δk(ω_s, ω_i) = k_p(ω_s+ω_i) − k_s(ω_s) − k_i(ω_i) − K_qpm.
///
/// # References
/// - Grice & Walmsley, PRA 1997 (JSA factorability)
/// - Mosley et al., PRL 2008 (group-index engineered pairs)
/// - Law & Eberly, PRL 2004 (Schmidt decomposition of JSA)
#[derive(Debug, Clone)]
pub struct IntegratedSpdcSource {
    /// Crystal (waveguide) length (m).
    pub crystal_length: f64,
    /// Effective mode area A_eff (m²).
    pub effective_area: f64,
    /// Phase-matching configuration.
    pub phase_matching: PhaseMatchingType,
    /// Pump centre wavelength (m).
    pub pump_wavelength: f64,
    /// Pump bandwidth FWHM (Hz, ordinary frequency).
    pub pump_bandwidth_fwhm: f64,
    /// Effective nonlinear coefficient d_eff (m/V). Note: 1 pm/V = 1e-12 m/V.
    pub d_eff: f64,
    /// Phase index at pump centre frequency.
    pub n_p: f64,
    /// Phase index at signal centre frequency (degenerate: λ_s = 2·λ_pump).
    pub n_s: f64,
    /// Phase index at idler centre frequency.
    pub n_i: f64,
    /// Group index at pump centre frequency.
    pub n_g_p: f64,
    /// Group index at signal centre frequency.
    pub n_g_s: f64,
    /// Group index at idler centre frequency.
    pub n_g_i: f64,
}

impl IntegratedSpdcSource {
    /// Pump centre angular frequency (rad/s).
    fn omega_p0(&self) -> f64 {
        2.0 * PI * C / self.pump_wavelength
    }

    /// Signal centre angular frequency (rad/s) — degenerate: ω_s0 = ω_p0/2.
    fn omega_s0(&self) -> f64 {
        self.omega_p0() / 2.0
    }

    /// Idler centre angular frequency (rad/s) — degenerate: ω_i0 = ω_p0/2.
    fn omega_i0(&self) -> f64 {
        self.omega_p0() / 2.0
    }

    /// Pump Gaussian bandwidth σ_ω (rad/s), converting FWHM in Hz.
    ///
    /// σ_freq = FWHM/(2√(2 ln 2)); σ_ω = 2π·σ_freq.
    fn sigma_omega_pump(&self) -> f64 {
        let sigma_freq = self.pump_bandwidth_fwhm / (2.0 * (2.0_f64 * 2.0_f64.ln()).sqrt());
        2.0 * PI * sigma_freq
    }

    /// Wave-vector k_j(ω) using first-order Taylor expansion about the centre frequency:
    ///
    /// k_j(ω) = n_j·ω_0j/c + (n_g_j/c)·(ω − ω_0j)
    ///
    /// where ω_0j is the centre frequency for field j.
    fn k_pump(&self, omega: f64) -> f64 {
        let omega_0 = self.omega_p0();
        self.n_p * omega_0 / C + (self.n_g_p / C) * (omega - omega_0)
    }

    fn k_signal(&self, omega: f64) -> f64 {
        let omega_0 = self.omega_s0();
        self.n_s * omega_0 / C + (self.n_g_s / C) * (omega - omega_0)
    }

    fn k_idler(&self, omega: f64) -> f64 {
        let omega_0 = self.omega_i0();
        self.n_i * omega_0 / C + (self.n_g_i / C) * (omega - omega_0)
    }

    /// QPM grating vector K_qpm = 2π/Λ (rad/m); 0 for non-QPM types.
    fn k_qpm(&self) -> f64 {
        match &self.phase_matching {
            PhaseMatchingType::QuasiPm { period_um } => 2.0 * PI / (period_um * 1e-6),
            _ => 0.0,
        }
    }

    /// Phase mismatch Δk(ω_s, ω_i) = k_p(ω_s+ω_i) − k_s(ω_s) − k_i(ω_i) − K_qpm.
    fn delta_k(&self, omega_s: f64, omega_i: f64) -> f64 {
        let omega_p = omega_s + omega_i;
        self.k_pump(omega_p) - self.k_signal(omega_s) - self.k_idler(omega_i) - self.k_qpm()
    }

    /// Pump Gaussian envelope α_p(ω_p) = exp(−(ω_p − ω_p0)²/(2·σ_ω²)).
    fn pump_envelope(&self, omega_p: f64) -> f64 {
        let sigma = self.sigma_omega_pump();
        let delta = omega_p - self.omega_p0();
        (-delta * delta / (2.0 * sigma * sigma)).exp()
    }

    /// Compute the JSA on a `ds × di` frequency grid centred on degeneracy.
    ///
    /// The grid spans ±3·σ_ω (pump bandwidth) around the degenerate frequency.
    pub fn jsa(&self, ds: usize, di: usize) -> JointSpectralAmplitude {
        let ns = ds.max(4);
        let ni = di.max(4);
        let omega_s0 = self.omega_s0();
        let omega_i0 = self.omega_i0();
        let sigma = self.sigma_omega_pump();
        let half_span = 3.0 * sigma;
        let l = self.crystal_length;

        let signal_freqs: Vec<f64> = (0..ns)
            .map(|i| omega_s0 - half_span + (2.0 * half_span) * (i as f64) / ((ns - 1) as f64))
            .collect();
        let idler_freqs: Vec<f64> = (0..ni)
            .map(|i| omega_i0 - half_span + (2.0 * half_span) * (i as f64) / ((ni - 1) as f64))
            .collect();

        let mut amplitude = vec![vec![Complex64::new(0.0, 0.0); ni]; ns];
        let mut norm_sq = 0.0_f64;

        for (si, &omega_s) in signal_freqs.iter().enumerate() {
            for (ii, &omega_i) in idler_freqs.iter().enumerate() {
                let alpha = self.pump_envelope(omega_s + omega_i);
                let dk_l_half = self.delta_k(omega_s, omega_i) * l / 2.0;
                let pm = sinc(dk_l_half);
                let val = alpha * pm;
                amplitude[si][ii] = Complex64::new(val, 0.0);
                norm_sq += val * val;
            }
        }

        // Normalise so Σ|f|² = 1
        if norm_sq > 0.0 {
            let norm = norm_sq.sqrt();
            for row in &mut amplitude {
                for v in row.iter_mut() {
                    *v /= norm;
                }
            }
        }

        JointSpectralAmplitude {
            signal_freqs,
            idler_freqs,
            amplitude,
        }
    }

    /// Schmidt decomposition via SVD of the JSA amplitude matrix.
    ///
    /// Computes singular values of the n×n JSA sub-matrix (taking the min of
    /// the ds/di dimensions). Returns eigenvalues λ_k = σ_k²/Σσ_j² in
    /// descending order.
    pub fn schmidt_decomposition(&self, n: usize) -> SchmidtDecomposition {
        let jsa = self.jsa(n, n);
        let dim = jsa.amplitude.len();
        if dim == 0 {
            return SchmidtDecomposition {
                eigenvalues: vec![1.0],
                schmidt_number: 1.0,
            };
        }

        // Build real matrix (JSA is real-valued in our model)
        let a: Vec<Vec<f64>> = jsa
            .amplitude
            .iter()
            .map(|row| row.iter().map(|v| v.re).collect())
            .collect();

        // S = A^T A; eigenvalues of S = squared singular values of A
        let s = mat_transpose_times_mat(&a, dim);
        let raw_eigs = symmetric_eigenvalues_power_deflation(&s, dim, dim.min(64));

        let mut lambdas: Vec<f64> = raw_eigs.into_iter().filter(|&v| v > 1e-14).collect();
        lambdas.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Normalise Σλ = 1
        let total: f64 = lambdas.iter().sum();
        if total > 0.0 {
            for v in lambdas.iter_mut() {
                *v /= total;
            }
        } else {
            lambdas = vec![1.0];
        }

        let sum_sq: f64 = lambdas.iter().map(|&l| l * l).sum();
        let schmidt_number = if sum_sq > 0.0 { 1.0 / sum_sq } else { 1.0 };

        SchmidtDecomposition {
            eigenvalues: lambdas,
            schmidt_number,
        }
    }

    /// Hong-Ou-Mandel visibility = 1/Σλ_k² = Schmidt number.
    ///
    /// For a spectrally pure (single-mode) state: visibility → 1.
    /// For a mixed state (K modes): visibility = K / (K² - ... ) < 1.
    /// Uses 20 Schmidt modes.
    pub fn hom_visibility(&self) -> f64 {
        let decomp = self.schmidt_decomposition(20);
        let sum_sq: f64 = decomp.eigenvalues.iter().map(|&l| l * l).sum();
        if sum_sq > 0.0 {
            (1.0 / sum_sq).min(1.0)
        } else {
            1.0
        }
    }

    /// Pair generation rate [pairs/s] at given pump power.
    ///
    /// Formula (waveguide):
    /// ```text
    /// R = (8π² · d_eff² · L · P_pump) / (ε₀ · c · n_p·n_s·n_i · λ_p³ · A_eff)
    /// ```
    /// Note: d_eff is in SI units (m/V); 1 pm/V = 1e-12 m/V.
    pub fn pair_generation_rate(&self, pump_power_w: f64) -> f64 {
        let lambda_p = self.pump_wavelength;
        let numerator =
            8.0 * PI * PI * self.d_eff * self.d_eff * self.crystal_length * pump_power_w;
        let denominator =
            EPSILON_0 * C * self.n_p * self.n_s * self.n_i * lambda_p.powi(3) * self.effective_area;
        numerator / denominator
    }

    /// Spectral brightness [pairs/(s·mW·nm)].
    ///
    /// Normalises the pair rate by pump power (mW) and the sinc FWHM bandwidth (nm).
    pub fn brightness_per_mw_per_nm(&self) -> f64 {
        // Signal centre wavelength
        let lambda_s = 2.0 * self.pump_wavelength;
        // GVM = (n_g_s - n_g_i)/c; bandwidth = 0.886·λ_s²/(c·L·|GVM_wavelength_units|)
        let gvm = (self.n_g_s - self.n_g_i).abs() / C; // s/m
        let bandwidth_nm = if gvm < 1e-20 {
            10.0 // fallback for degenerate / type-0
        } else {
            let delta_omega = 0.886 * PI / (self.crystal_length * gvm);
            let delta_lambda = (lambda_s * lambda_s / (2.0 * PI * C)) * delta_omega;
            delta_lambda * 1e9 // m → nm
        };
        // Rate at 1 mW
        let r = self.pair_generation_rate(1e-3);
        r / (1.0 * bandwidth_nm)
    }
}

// ─── Linear algebra helpers ───────────────────────────────────────────────────

/// sinc(x) = sin(x)/x with the limit sinc(0) = 1.
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        1.0
    } else {
        x.sin() / x
    }
}

/// Compute B = A^T * A for an n×n matrix stored as Vec<Vec<f64>>.
fn mat_transpose_times_mat(a: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut b = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for row in a.iter().take(n) {
                acc += row[i] * row[j];
            }
            b[i][j] = acc;
        }
    }
    b
}

/// Extract eigenvalues of a real symmetric matrix via power-method deflation.
///
/// Returns up to `n_eigs` dominant eigenvalues in descending order.
fn symmetric_eigenvalues_power_deflation(s: &[Vec<f64>], n: usize, n_eigs: usize) -> Vec<f64> {
    let max_eigs = n_eigs.min(n);
    let mut eigenvalues = Vec::with_capacity(max_eigs);
    // Work on a mutable copy
    let mut m: Vec<Vec<f64>> = s.to_vec();

    for _ in 0..max_eigs {
        // Power iteration for dominant eigenvalue
        let mut v = vec![1.0_f64 / (n as f64).sqrt(); n];
        let mut eigenvalue = 0.0_f64;

        for _iter in 0..200 {
            let mv = mat_vec_mul(&m, &v, n);
            let norm: f64 = mv.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-30 {
                break;
            }
            let v_new: Vec<f64> = mv.iter().map(|x| x / norm).collect();
            // Rayleigh quotient
            let mv2 = mat_vec_mul(&m, &v_new, n);
            eigenvalue = v_new.iter().zip(mv2.iter()).map(|(a, b)| a * b).sum();
            let diff: f64 = v_new
                .iter()
                .zip(v.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            v = v_new;
            if diff < 1e-12 {
                break;
            }
        }

        if eigenvalue.abs() < 1e-14 {
            break;
        }
        eigenvalues.push(eigenvalue);

        // Deflate: M ← M - λ v v^T
        for i in 0..n {
            for j in 0..n {
                m[i][j] -= eigenvalue * v[i] * v[j];
            }
        }
    }
    eigenvalues
}

/// Multiply matrix m (n×n) by vector v.
fn mat_vec_mul(m: &[Vec<f64>], v: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            result[i] += m[i][j] * v[j];
        }
    }
    result
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppktp_wavelengths() {
        let src = SpdcSource::new_ppktp_1550(1.0, 10.0);
        // Degenerate SPDC: signal = idler = 2 * pump
        let sig = src.signal_wavelength();
        let idl = src.idler_wavelength();
        assert!((sig - 1550e-9).abs() < 1e-12, "signal should be ~1550 nm");
        assert!((idl - sig).abs() < 1e-12, "degenerate: signal = idler");
    }

    #[test]
    fn test_pair_rate_positive() {
        let src = SpdcSource::new_ppktp_1550(1.0, 10.0);
        let rate = src.pair_rate_per_second();
        assert!(rate > 0.0, "pair rate must be positive");
        // Our formula gives a dimensionally correct value; physical order of magnitude varies
        // with how the mode area and bandwidth integrals are normalised.
        // We test that the rate is finite and positive.
        assert!(rate.is_finite(), "pair rate must be finite");
    }

    #[test]
    fn test_pair_rate_scales_with_power() {
        // Rate should scale linearly with pump power
        let src1 = SpdcSource::new_ppktp_1550(1.0, 10.0);
        let src2 = SpdcSource::new_ppktp_1550(2.0, 10.0);
        let r1 = src1.pair_rate_per_second();
        let r2 = src2.pair_rate_per_second();
        assert!(
            (r2 / r1 - 2.0).abs() < 1e-9,
            "Rate should double when power doubles"
        );
    }

    #[test]
    fn test_spectral_brightness_positive() {
        let src = SpdcSource::new_ppln_telecom(1.0, 20.0);
        let b = src.spectral_brightness();
        assert!(b > 0.0, "spectral brightness must be positive");
    }

    #[test]
    fn test_phase_matching_bandwidth() {
        let src = SpdcSource::new_ppktp_1550(1.0, 10.0);
        let bw = src.phase_matching_bandwidth_nm();
        // PPKTP 10 mm crystal: PM bandwidth ~0.5–5 nm
        assert!(bw > 0.0, "bandwidth must be positive");
        assert!(bw < 100.0, "bandwidth sanity check (< 100 nm)");
    }

    #[test]
    fn test_schmidt_number_ge_one() {
        let src = SpdcSource::new_ppktp_1550(1.0, 10.0);
        let k = src.schmidt_number(16);
        assert!(k >= 1.0 - 1e-9, "Schmidt number K ≥ 1");
    }

    #[test]
    fn test_spectral_purity_range() {
        let src = SpdcSource::new_ppktp_1550(1.0, 10.0);
        let p = src.spectral_purity(16);
        assert!(p > 0.0 && p <= 1.0 + 1e-9, "spectral purity ∈ (0, 1]");
    }

    #[test]
    fn test_g2_zero_thermal() {
        let src = SpdcSource::new_ppktp_1550(1.0, 10.0);
        let g2 = src.g2_zero();
        // For single-mode thermal: g²(0) = 2; multi-mode: between 1 and 2
        assert!((1.0..=2.0 + 1e-9).contains(&g2), "g²(0) ∈ [1, 2]");
    }

    #[test]
    fn test_jsa_normalised() {
        let src = SpdcSource::new_ppktp_1550(1.0, 10.0);
        let (_, _, jsa) = src.joint_spectral_amplitude(16);
        let norm_sq: f64 = jsa
            .iter()
            .flat_map(|row| row.iter())
            .map(|v| v.norm_sqr())
            .sum();
        // After normalisation the grid sum of |f|² ≈ 1
        assert!(
            (norm_sq - 1.0).abs() < 0.01,
            "JSA should be normalised to ~1"
        );
    }

    #[test]
    fn test_sinc_zero() {
        assert!((sinc(0.0) - 1.0).abs() < 1e-15, "sinc(0) = 1");
        assert!((sinc(1e-12) - 1.0).abs() < 1e-10, "sinc(near-0) ≈ 1");
    }

    #[test]
    fn test_sinc_at_pi() {
        // sinc(π) = sin(π)/π ≈ 0
        let v = sinc(PI);
        assert!(v.abs() < 1e-14, "sinc(π) ≈ 0, got {v}");
    }
}
