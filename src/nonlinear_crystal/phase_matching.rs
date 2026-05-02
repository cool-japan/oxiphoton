//! Phase matching analysis for nonlinear optical frequency conversion.
//!
//! Covers:
//! - Type I/II/III phase matching in uniaxial crystals
//! - Quasi-phase matching (QPM) with poling period
//! - SHG phase mismatch Δk, coherence length, walk-off
//! - Acceptance bandwidths (angular, spectral, thermal)
//! - SHG conversion efficiency (plane-wave, low-depletion)
//! - Boyd-Kleinman optimum focus parameter
//! - General optical frequency conversion framework (SHG, SFG, DFG, OPA, OPO, THG)

use crate::error::OxiPhotonError;
use crate::nonlinear_crystal::crystals::NloCrystal;

/// Speed of light in vacuum (m/s).
const C0: f64 = 2.99792458e8;
/// Permittivity of free space (F/m).
const EPS0: f64 = 8.854187817e-12;

// ─── Phase matching type ───────────────────────────────────────────────────

/// Phase matching type for a nonlinear interaction.
#[derive(Debug, Clone)]
pub enum PhaseMatchingType {
    /// Type I: both input photons have the same polarization (e.g., o + o → e).
    TypeI,
    /// Type II: input photons have orthogonal polarizations (e.g., o + e → e).
    TypeII,
    /// Type III: non-collinear / e + e → e (less common label; used for NCPM or
    /// non-collinear geometries).
    TypeIII,
    /// Type 0: all fields share the same polarization (e.g., e → e + e via d33).
    /// Requires QPM to satisfy phase matching; common in waveguide PPLN.
    TypeZero,
    /// Quasi-phase matching (QPM) using periodic domain inversion with given poling period Λ (μm).
    QuasiPm { period_um: f64 },
}

// ─── SHG Phase Matching ────────────────────────────────────────────────────

/// Collinear SHG phase matching analysis for a uniaxial crystal.
///
/// Computes phase mismatch Δk, phase matching angle, walk-off, coherence length,
/// acceptance bandwidths (angular, spectral, thermal), and plane-wave efficiency.
pub struct SHGPhaseMatching {
    /// Crystal to use for phase matching analysis.
    pub crystal: NloCrystal,
    /// Phase matching type (Type I, II, QPM, …).
    pub pm_type: PhaseMatchingType,
    /// Fundamental wavelength λ₁ (nm).
    pub fundamental_wavelength_nm: f64,
}

impl SHGPhaseMatching {
    /// Construct a new SHG phase matching object.
    pub fn new(crystal: NloCrystal, pm_type: PhaseMatchingType, lambda_nm: f64) -> Self {
        Self {
            crystal,
            pm_type,
            fundamental_wavelength_nm: lambda_nm,
        }
    }

    /// SHG wavelength λ₂ = λ₁/2 (nm).
    pub fn shg_wavelength_nm(&self) -> f64 {
        self.fundamental_wavelength_nm / 2.0
    }

    /// Wave-vector k = 2πn/λ (m⁻¹) for the ordinary beam at λ (nm).
    fn k_ordinary(&self, lambda_nm: f64) -> f64 {
        let n = self.crystal.n_ordinary(lambda_nm);
        2.0 * std::f64::consts::PI * n / (lambda_nm * 1e-9)
    }

    /// Wave-vector k (m⁻¹) for the extraordinary beam at angle θ (rad) and λ (nm).
    fn k_extraordinary(&self, lambda_nm: f64, theta_rad: f64) -> f64 {
        let n = self.crystal.n_extraordinary(lambda_nm, theta_rad);
        2.0 * std::f64::consts::PI * n / (lambda_nm * 1e-9)
    }

    /// Phase mismatch Δk = k₂ − 2k₁ (rad/m) for given crystal angle θ (rad).
    ///
    /// For Type-I SHG: k₁ is ordinary (fundamental), k₂ is extraordinary (SHG).
    /// For Type-II SHG: k₁_o + k₁_e → k₂_e.
    pub fn phase_mismatch(&self, theta_rad: f64) -> f64 {
        let lambda1 = self.fundamental_wavelength_nm;
        let lambda2 = self.shg_wavelength_nm();
        match &self.pm_type {
            PhaseMatchingType::TypeI => {
                // o + o → e: k2_e(θ) - 2 * k1_o
                let k1 = self.k_ordinary(lambda1);
                let k2 = self.k_extraordinary(lambda2, theta_rad);
                k2 - 2.0 * k1
            }
            PhaseMatchingType::TypeII => {
                // o + e → e: k2_e(θ) - k1_o - k1_e(θ)
                let k1_o = self.k_ordinary(lambda1);
                let k1_e = self.k_extraordinary(lambda1, theta_rad);
                let k2_e = self.k_extraordinary(lambda2, theta_rad);
                k2_e - k1_o - k1_e
            }
            PhaseMatchingType::TypeIII => {
                // e + e → o: k2_o - 2 * k1_e(θ)
                let k1_e = self.k_extraordinary(lambda1, theta_rad);
                let k2_o = self.k_ordinary(lambda2);
                k2_o - 2.0 * k1_e
            }
            PhaseMatchingType::TypeZero => {
                // e → e + e: all ordinary (same polarisation, e.g. d33 in PPLN)
                let k1 = self.k_ordinary(lambda1);
                let k2 = self.k_ordinary(lambda2);
                k2 - 2.0 * k1
            }
            PhaseMatchingType::QuasiPm { period_um } => {
                // Type-I with QPM compensation: Δk_eff = Δk_free - 2π/Λ
                let k1 = self.k_ordinary(lambda1);
                let k2 = self.k_ordinary(lambda2); // both ordinary in QPM
                let delta_k_free = k2 - 2.0 * k1;
                let k_qpm = 2.0 * std::f64::consts::PI / (period_um * 1e-6);
                delta_k_free - k_qpm
            }
        }
    }

    /// Phase matching angle θ_PM (rad) where Δk(θ) = 0.
    ///
    /// Uses bisection search over [1°, 89°]. Returns an error if no root is found.
    pub fn phase_matching_angle_rad(&self) -> Result<f64, OxiPhotonError> {
        let lo = 1.0_f64.to_radians();
        let hi = 89.0_f64.to_radians();
        let f_lo = self.phase_mismatch(lo);
        let f_hi = self.phase_mismatch(hi);

        // Check that there is a sign change → root exists
        if f_lo * f_hi > 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "No phase matching angle found in [1°, 89°] — check crystal and wavelength"
                    .to_string(),
            ));
        }

        let mut a = lo;
        let mut b = hi;
        for _ in 0..64 {
            let mid = 0.5 * (a + b);
            if self.phase_mismatch(mid) * f_lo <= 0.0 {
                b = mid;
            } else {
                a = mid;
            }
            if (b - a).abs() < 1e-10 {
                break;
            }
        }
        Ok(0.5 * (a + b))
    }

    /// Walk-off angle ρ (rad) between the Poynting vector and the k-vector for the
    /// extraordinary beam at angle θ.
    ///
    /// tan(ρ) = −(1/n_e(θ)) · dn_e(θ)/dθ
    pub fn walkoff_angle_rad(&self, theta_rad: f64) -> f64 {
        let lambda2 = self.shg_wavelength_nm();
        let dtheta = 1e-7_f64;
        let ne_p = self.crystal.n_extraordinary(lambda2, theta_rad + dtheta);
        let ne_m = self.crystal.n_extraordinary(lambda2, theta_rad - dtheta);
        let dne_dtheta = (ne_p - ne_m) / (2.0 * dtheta);
        let ne = self.crystal.n_extraordinary(lambda2, theta_rad);
        if ne.abs() < 1e-10 {
            return 0.0;
        }
        (-dne_dtheta / ne).atan()
    }

    /// Walk-off length L_w (mm): propagation distance over which a Gaussian beam of
    /// waist w₀ (μm) walks off due to birefringence.
    ///
    /// L_w = w₀ / |tan(ρ)|
    pub fn walkoff_length_mm(&self, beam_waist_um: f64, theta_rad: f64) -> f64 {
        let rho = self.walkoff_angle_rad(theta_rad).abs();
        if rho < 1e-12 {
            return f64::INFINITY;
        }
        let w0_mm = beam_waist_um * 1e-3;
        w0_mm / rho.tan()
    }

    /// Coherence length L_c (mm) = π/|Δk|.
    ///
    /// At perfect phase matching (Δk → 0), L_c → ∞.
    pub fn coherence_length_mm(&self, theta_rad: f64) -> f64 {
        let dk = self.phase_mismatch(theta_rad).abs();
        if dk < 1e-6 {
            return f64::INFINITY;
        }
        std::f64::consts::PI / dk * 1e3 // convert m to mm
    }

    /// Angular acceptance bandwidth δθ (mrad) for a crystal of length L (mm).
    ///
    /// δθ such that Δk(θ_PM ± δθ/2) = π/L (sinc² drops to first zero).
    /// Uses numerical derivative of Δk w.r.t. θ.
    pub fn angular_acceptance_mrad(&self, crystal_length_mm: f64, theta_pm: f64) -> f64 {
        let l_m = crystal_length_mm * 1e-3;
        let dtheta = 1e-7_f64;
        let dk_p = self.phase_mismatch(theta_pm + dtheta);
        let dk_m = self.phase_mismatch(theta_pm - dtheta);
        let ddk_dtheta = (dk_p - dk_m) / (2.0 * dtheta);
        if ddk_dtheta.abs() < 1e-20 {
            return f64::INFINITY;
        }
        // |dDk/dθ| * δθ/2 = π/L → δθ = 2π/(L * |dDk/dθ|)
        let delta_theta_rad = 2.0 * std::f64::consts::PI / (l_m * ddk_dtheta.abs());
        delta_theta_rad * 1e3 // rad → mrad
    }

    /// Spectral acceptance bandwidth δλ (nm) for a crystal of length L (mm).
    ///
    /// δλ such that Δk(λ±δλ/2) · L = π (first sinc zero).
    /// Uses numerical derivative of Δk w.r.t. λ at the phase-matching angle.
    pub fn spectral_acceptance_nm(&self, crystal_length_mm: f64) -> f64 {
        let theta_pm = self
            .phase_matching_angle_rad()
            .unwrap_or(std::f64::consts::FRAC_PI_4);
        let l_m = crystal_length_mm * 1e-3;
        let dlambda_nm = 0.01_f64; // 10 pm step

        // Temporarily shift wavelength via a mini closure
        let dk_at_lambda = |lam_nm: f64| -> f64 {
            let crystal_tmp = self.crystal.clone();
            let pm_tmp = self.pm_type.clone();
            let shg_tmp = SHGPhaseMatching::new(crystal_tmp, pm_tmp, lam_nm);
            shg_tmp.phase_mismatch(theta_pm)
        };

        let lam0 = self.fundamental_wavelength_nm;
        let ddk_dl = (dk_at_lambda(lam0 + dlambda_nm) - dk_at_lambda(lam0 - dlambda_nm))
            / (2.0 * dlambda_nm * 1e-9); // rad/(m·m) → need d(Δk)/dλ in 1/m per m

        if ddk_dl.abs() < 1e-10 {
            return f64::INFINITY;
        }
        // |dDk/dlambda| * δλ/2 = π/L
        let delta_lambda_m = 2.0 * std::f64::consts::PI / (l_m * ddk_dl.abs());
        delta_lambda_m * 1e9 // m → nm
    }

    /// Temperature acceptance bandwidth δT (°C) for a crystal of length L (mm).
    ///
    /// Estimated from the thermo-optic coefficient via
    /// δT = 2π / (L · |dΔk/dT|).
    pub fn temperature_acceptance_degc(&self, crystal_length_mm: f64, _theta_pm: f64) -> f64 {
        let l_m = crystal_length_mm * 1e-3;
        let lambda_m = self.fundamental_wavelength_nm * 1e-9;
        let lambda2_m = lambda_m / 2.0;
        // dΔk/dT ≈ (2π/λ₂) · dn_e/dT − 2·(2π/λ₁) · dn_o/dT
        let dn_dt = self.crystal.thermo_optic_dn_dt;
        let ddk_dt = (2.0 * std::f64::consts::PI / lambda2_m
            - 2.0 * 2.0 * std::f64::consts::PI / lambda_m)
            * dn_dt;
        if ddk_dt.abs() < 1e-30 {
            return f64::INFINITY;
        }
        2.0 * std::f64::consts::PI / (l_m * ddk_dt.abs())
    }

    /// Normalized SHG efficiency sinc²(ΔkL/2) ∈ [0, 1].
    pub fn sinc_efficiency(&self, crystal_length_mm: f64, theta_rad: f64) -> f64 {
        let dk = self.phase_mismatch(theta_rad);
        let l_m = crystal_length_mm * 1e-3;
        let x = dk * l_m / 2.0;
        if x.abs() < 1e-12 {
            1.0
        } else {
            (x.sin() / x).powi(2)
        }
    }

    /// Boyd-Kleinman optimum focus parameter ξ_opt ≈ 2.84.
    ///
    /// Defined as ξ = L/(2z_R) where z_R is the Rayleigh range.
    /// The optimum focusing maximizes the tightly focused SHG conversion.
    pub fn optimal_focus_parameter() -> f64 {
        2.84
    }

    /// Plane-wave SHG conversion efficiency η (fraction, 0–1) in the low-depletion limit.
    ///
    /// Uses the formula:
    /// η = (8π²·d_eff²·L²·P_in) / (n₁²·n₂·ε₀·c·λ₁²·A)
    ///
    /// Parameters:
    /// - `crystal_length_mm`: crystal length (mm)
    /// - `peak_power_w`: input peak power (W)
    /// - `beam_area_um2`: beam cross-section area (μm²)
    /// - `d_eff_pm_per_v`: effective nonlinear coefficient (pm/V)
    /// - `theta_rad`: crystal angle (rad) — applies sinc² phase mismatch factor
    pub fn plane_wave_efficiency(
        &self,
        crystal_length_mm: f64,
        peak_power_w: f64,
        beam_area_um2: f64,
        d_eff_pm_per_v: f64,
        theta_rad: f64,
    ) -> f64 {
        let lambda1_m = self.fundamental_wavelength_nm * 1e-9;
        let lambda2_m = self.shg_wavelength_nm() * 1e-9;
        let l_m = crystal_length_mm * 1e-3;
        let area_m2 = beam_area_um2 * 1e-12;
        let d_eff = d_eff_pm_per_v * 1e-12; // pm/V → m/V
        let n1 = self.crystal.n_ordinary(self.fundamental_wavelength_nm);
        let n2 = self
            .crystal
            .n_extraordinary(self.shg_wavelength_nm(), theta_rad);

        // η = (8π²·d_eff²·L²·P) / (n1²·n2·ε0·c·λ1²·A) · sinc²(ΔkL/2)
        let numerator =
            8.0 * std::f64::consts::PI.powi(2) * d_eff * d_eff * l_m * l_m * peak_power_w;
        let denominator = n1 * n1 * n2 * EPS0 * C0 * lambda2_m * lambda1_m * lambda1_m * area_m2;
        if denominator < 1e-100 {
            return 0.0;
        }
        let eta_pw = numerator / denominator;
        let sinc2 = self.sinc_efficiency(crystal_length_mm, theta_rad);
        (eta_pw * sinc2).min(1.0)
    }

    /// Effective nonlinear coefficient d_eff (pm/V) for given crystal angles θ, φ.
    ///
    /// For Type-I in a uniaxial crystal with 3m symmetry (e.g., BBO):
    /// d_eff = d_22·cos(θ)·cos(3φ) − d_31·sin(θ)
    /// This approximation is valid for BBO in the principal plane (φ=0).
    pub fn effective_d_coefficient(&self, theta_rad: f64, phi_rad: f64) -> f64 {
        // d_22 and d_31 from the d-tensor (column indices: Voigt notation)
        // For BBO: d_22 = d_tensor[1][1], d_31 = d_tensor[2][0]
        let d22 = self.crystal.d_tensor[1][1].abs();
        let d31 = self.crystal.d_tensor[2][0].abs();
        match &self.pm_type {
            PhaseMatchingType::TypeI => {
                // d_eff(Type-I, ooe) = d_22*cos(θ)*cos(3φ) - d_31*sin(θ)
                d22 * theta_rad.cos() * (3.0 * phi_rad).cos() - d31 * theta_rad.sin()
            }
            PhaseMatchingType::TypeII => {
                // d_eff(Type-II, oee) = d_22*cos(2φ)*cos²(θ) − d_31*(something)
                // Simplified for φ=0 principal plane:
                d22 * (2.0 * phi_rad).cos() * theta_rad.cos() * theta_rad.cos()
                    - d31 * theta_rad.sin()
            }
            _ => d22 * theta_rad.cos(),
        }
    }

    /// QPM poling period Λ (μm) required to achieve first-order quasi-phase matching
    /// at the given temperature (°C).
    ///
    /// Λ = 2π / |Δk_free| where Δk_free is the free (non-poled) phase mismatch.
    pub fn qpm_period_um(&self, temperature_c: f64) -> f64 {
        // Temperature shifts the refractive index slightly
        let dn_dt = self.crystal.thermo_optic_dn_dt;
        let dt = temperature_c - 25.0; // reference temperature 25°C
        let lambda1_nm = self.fundamental_wavelength_nm;
        let lambda2_nm = self.shg_wavelength_nm();
        let lambda1_m = lambda1_nm * 1e-9;
        let lambda2_m = lambda2_nm * 1e-9;

        // Shift refractive index with temperature (approximate)
        let n_o1 = self.crystal.n_ordinary(lambda1_nm) + dn_dt * dt;
        let n_o2 = self.crystal.n_ordinary(lambda2_nm) + dn_dt * dt;
        let k1 = 2.0 * std::f64::consts::PI * n_o1 / lambda1_m;
        let k2 = 2.0 * std::f64::consts::PI * n_o2 / lambda2_m;
        let dk_free = (k2 - 2.0 * k1).abs();
        if dk_free < 1e-6 {
            return f64::INFINITY;
        }
        // First-order QPM: Λ = 2π/|Δk_free|, convert m to μm
        2.0 * std::f64::consts::PI / dk_free * 1e6
    }
}

// ─── Frequency conversion process taxonomy ─────────────────────────────────

/// Type of optical frequency conversion process.
#[derive(Debug, Clone, PartialEq)]
pub enum ConversionProcess {
    /// Second harmonic generation: λ₃ = λ₁/2.
    SHG,
    /// Sum frequency generation: 1/λ₃ = 1/λ₁ + 1/λ₂.
    SFG,
    /// Difference frequency generation: 1/λ₃ = 1/λ₁ − 1/λ₂.
    DFG,
    /// Optical parametric amplification (pump → signal + idler).
    OPA,
    /// Optical parametric oscillation (cavity-enhanced OPA).
    OPO,
    /// Third harmonic generation: λ₃ = λ₁/3.
    THG,
}

/// General optical frequency conversion framework.
///
/// Stores pump/signal/idler wavelengths and the conversion process type,
/// with energy conservation and Manley-Rowe utilities.
pub struct FrequencyConversion {
    /// Pump wavelength λ₁ (nm).
    pub lambda1_nm: f64,
    /// Signal wavelength λ₂ (nm). For SHG, equal to λ₁.
    pub lambda2_nm: f64,
    /// Output/idler wavelength λ₃ (nm).
    pub lambda3_nm: f64,
    /// Conversion process.
    pub process: ConversionProcess,
}

impl FrequencyConversion {
    /// Create from pump and signal wavelengths; compute output for the given process.
    pub fn new(lambda1_nm: f64, lambda2_nm: f64, process: ConversionProcess) -> Self {
        let lambda3_nm = match &process {
            ConversionProcess::SHG => lambda1_nm / 2.0,
            ConversionProcess::THG => lambda1_nm / 3.0,
            ConversionProcess::SFG => {
                // 1/λ₃ = 1/λ₁ + 1/λ₂
                1.0 / (1.0 / lambda1_nm + 1.0 / lambda2_nm)
            }
            ConversionProcess::DFG | ConversionProcess::OPA | ConversionProcess::OPO => {
                // 1/λ₃ = 1/λ₁ - 1/λ₂ (idler)
                let inv = 1.0 / lambda1_nm - 1.0 / lambda2_nm;
                if inv > 0.0 {
                    1.0 / inv
                } else {
                    f64::INFINITY
                }
            }
        };
        Self {
            lambda1_nm,
            lambda2_nm,
            lambda3_nm,
            process,
        }
    }

    /// Construct an SHG interaction: λ₃ = λ₁/2.
    pub fn shg(lambda_nm: f64) -> Self {
        Self::new(lambda_nm, lambda_nm, ConversionProcess::SHG)
    }

    /// Construct an SFG interaction: 1/λ₃ = 1/λ₁ + 1/λ₂.
    pub fn sfg(lambda1_nm: f64, lambda2_nm: f64) -> Self {
        Self::new(lambda1_nm, lambda2_nm, ConversionProcess::SFG)
    }

    /// Construct a DFG interaction (idler wavelength computed automatically).
    pub fn dfg(pump_nm: f64, signal_nm: f64) -> Self {
        Self::new(pump_nm, signal_nm, ConversionProcess::DFG)
    }

    /// Idler wavelength λ₃ (nm).
    pub fn idler_wavelength_nm(&self) -> f64 {
        self.lambda3_nm
    }

    /// Check energy conservation: 1/λ₁ ≈ 1/λ₂ + 1/λ₃ (for SFG/DFG).
    pub fn energy_conservation_check(&self) -> bool {
        match &self.process {
            ConversionProcess::SHG => {
                let err = (self.lambda1_nm / 2.0 - self.lambda3_nm).abs() / self.lambda3_nm;
                err < 1e-6
            }
            ConversionProcess::THG => {
                let err = (self.lambda1_nm / 3.0 - self.lambda3_nm).abs() / self.lambda3_nm;
                err < 1e-6
            }
            ConversionProcess::SFG => {
                let lhs = 1.0 / self.lambda1_nm + 1.0 / self.lambda2_nm;
                let rhs = 1.0 / self.lambda3_nm;
                (lhs - rhs).abs() / rhs < 1e-6
            }
            ConversionProcess::DFG | ConversionProcess::OPA | ConversionProcess::OPO => {
                let lhs = 1.0 / self.lambda1_nm;
                let rhs = 1.0 / self.lambda2_nm + 1.0 / self.lambda3_nm;
                (lhs - rhs).abs() / lhs < 1e-6
            }
        }
    }

    /// Manley-Rowe photon flux ratio n₃/n₁ (output photons / pump photons).
    ///
    /// For SHG: 0.5 (2 input photons → 1 output photon).
    /// For DFG/OPA: 1.0 (1 pump → 1 signal + 1 idler).
    pub fn manley_rowe_ratio(&self) -> f64 {
        match &self.process {
            ConversionProcess::SHG => 0.5,
            ConversionProcess::THG => 1.0 / 3.0,
            ConversionProcess::SFG => {
                // λ₃ < λ₁, λ₂ — output photon has higher energy
                self.lambda1_nm / self.lambda3_nm
            }
            ConversionProcess::DFG | ConversionProcess::OPA | ConversionProcess::OPO => {
                // For every pump photon: 1 signal + 1 idler
                1.0
            }
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nonlinear_crystal::crystals::NloCrystal;

    fn bbo_shg_type1() -> SHGPhaseMatching {
        SHGPhaseMatching::new(NloCrystal::bbo(), PhaseMatchingType::TypeI, 1064.0)
    }

    #[test]
    fn test_shg_wavelength_is_half() {
        let shg = bbo_shg_type1();
        let lambda_shg = shg.shg_wavelength_nm();
        assert!(
            (lambda_shg - 532.0).abs() < 1e-10,
            "SHG wavelength {:.2} should be 532 nm",
            lambda_shg
        );
    }

    #[test]
    fn test_dfg_idler_wavelength() {
        // 1/λ_i = 1/λ_p - 1/λ_s
        let pump_nm = 532.0;
        let signal_nm = 800.0;
        let fc = FrequencyConversion::dfg(pump_nm, signal_nm);
        let idler = fc.idler_wavelength_nm();
        let expected = 1.0 / (1.0 / pump_nm - 1.0 / signal_nm);
        assert!(
            (idler - expected).abs() < 0.01,
            "DFG idler {:.2} nm, expected {:.2} nm",
            idler,
            expected
        );
    }

    #[test]
    fn test_energy_conservation() {
        let fc_shg = FrequencyConversion::shg(1064.0);
        assert!(
            fc_shg.energy_conservation_check(),
            "SHG energy conservation failed"
        );

        let fc_sfg = FrequencyConversion::sfg(1064.0, 532.0);
        assert!(
            fc_sfg.energy_conservation_check(),
            "SFG energy conservation failed"
        );

        let fc_dfg = FrequencyConversion::dfg(532.0, 800.0);
        assert!(
            fc_dfg.energy_conservation_check(),
            "DFG energy conservation failed"
        );
    }

    #[test]
    fn test_coherence_length_at_phase_match() {
        // At the phase matching angle, Δk ≈ 0 and L_c → ∞
        let shg = bbo_shg_type1();
        if let Ok(theta_pm) = shg.phase_matching_angle_rad() {
            let lc = shg.coherence_length_mm(theta_pm);
            // L_c should be very large (>> 1 m) at phase match
            assert!(
                lc > 1e3,
                "Coherence length at PM angle should be >> 1 m, got {:.2} mm",
                lc
            );
        }
    }

    #[test]
    fn test_sinc_efficiency_at_zero_mismatch() {
        let shg = bbo_shg_type1();
        // At exact phase matching angle, sinc² = 1
        if let Ok(theta_pm) = shg.phase_matching_angle_rad() {
            let eta = shg.sinc_efficiency(10.0, theta_pm);
            assert!(
                (eta - 1.0).abs() < 0.01,
                "sinc² at PM angle should be ≈ 1, got {:.4}",
                eta
            );
        }
    }

    #[test]
    fn test_sinc_efficiency_decreases_with_mismatch() {
        let shg = bbo_shg_type1();
        // Away from PM angle, sinc² < 1
        let eta_on = shg.sinc_efficiency(10.0, 22.8_f64.to_radians());
        let eta_off = shg.sinc_efficiency(10.0, 30.0_f64.to_radians());
        // Off-angle should have lower efficiency
        assert!(
            eta_off <= eta_on + 1e-10,
            "Off-angle sinc²={:.4} should not exceed on-angle {:.4}",
            eta_off,
            eta_on
        );
    }

    #[test]
    fn test_qpm_period_physical() {
        // QPM poling period for LiNbO3 SHG at 1064 nm should be ~19 μm
        let shg = SHGPhaseMatching::new(
            NloCrystal::linbo3(),
            PhaseMatchingType::QuasiPm { period_um: 19.0 },
            1064.0,
        );
        let period = shg.qpm_period_um(25.0);
        assert!(
            period > 1.0 && period < 100.0,
            "QPM period {:.2} μm should be in [1, 100] μm",
            period
        );
    }
}
