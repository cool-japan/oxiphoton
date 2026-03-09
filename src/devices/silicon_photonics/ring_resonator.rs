//! Ring resonator models for silicon photonics.
//!
//! Implements all-pass and add-drop ring resonators using the transfer-matrix /
//! coupled-mode-theory formalism.  Physical quantities follow SI conventions
//! internally; user-facing lengths are in μm and nm for compactness.
//!
//! # References
//! - Yariv, "Universal relations for coupling of optical power between
//!   microresonators and dielectric waveguides", Electron. Lett. 36 (2000)
//! - Bogaerts et al., "Silicon microring resonators", Laser Photon. Rev. (2012)

use num_complex::Complex64;
use std::f64::consts::PI;

use crate::error::{OxiPhotonError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Speed of light in nm/s (c = 2.998 × 10^17 nm/s)
#[cfg_attr(not(test), allow(dead_code))]
const C_NM_PER_S: f64 = 2.997_924_58e17;

// ─────────────────────────────────────────────────────────────────────────────
// AllPassRing
// ─────────────────────────────────────────────────────────────────────────────

/// All-pass ring resonator (single bus waveguide).
///
/// The through-port transfer function is derived from the round-trip transfer
/// matrix:
///
/// ```text
/// E_t     t - a · exp(iφ)
/// ─── = ─────────────────────
/// E_in  1 - t·a · exp(iφ)
/// ```
///
/// where:
/// - `t = sqrt(1 - κ²)` is the self-coupling coefficient (field transmission),
/// - `a = exp(-α L / 2)` is the round-trip field loss amplitude,
/// - `φ = 2π n_eff L / λ` is the round-trip phase.
#[derive(Debug, Clone)]
pub struct AllPassRing {
    /// Ring radius (μm).
    pub radius_um: f64,
    /// Effective refractive index (dimensionless).
    pub n_eff: f64,
    /// Group refractive index (dimensionless).
    pub n_g: f64,
    /// Propagation loss (dB/cm).
    pub loss_db_per_cm: f64,
    /// Field coupling coefficient κ ∈ (0, 1).  Power coupling = κ².
    pub coupling_coeff: f64,
    /// Design / center wavelength (nm).
    pub wavelength_nm: f64,
}

impl AllPassRing {
    /// Create a new all-pass ring resonator.
    ///
    /// # Arguments
    /// * `radius_um`       – ring radius in μm
    /// * `n_eff`           – effective refractive index
    /// * `n_g`             – group refractive index
    /// * `loss_db_per_cm`  – waveguide propagation loss in dB/cm
    /// * `coupling_coeff`  – field coupling coefficient κ (0 < κ < 1)
    /// * `wavelength_nm`   – design center wavelength in nm
    pub fn new(
        radius_um: f64,
        n_eff: f64,
        n_g: f64,
        loss_db_per_cm: f64,
        coupling_coeff: f64,
        wavelength_nm: f64,
    ) -> Self {
        Self {
            radius_um,
            n_eff,
            n_g,
            loss_db_per_cm,
            coupling_coeff,
            wavelength_nm,
        }
    }

    /// Ring circumference in nm.
    #[inline]
    fn circumference_nm(&self) -> f64 {
        2.0 * PI * self.radius_um * 1_000.0 // μm → nm
    }

    /// Round-trip optical phase at wavelength `lambda_nm` (radians).
    ///
    /// φ = 2π · n_eff · L / λ
    pub fn round_trip_phase(&self, lambda_nm: f64) -> f64 {
        2.0 * PI * self.n_eff * self.circumference_nm() / lambda_nm
    }

    /// Round-trip field amplitude loss factor `a = exp(-α L / 2)`.
    ///
    /// α (field) is half the power attenuation coefficient.
    /// α_power [1/nm] = loss_db_per_cm * ln(10)/10 / (10^7)  [cm→nm]
    pub fn round_trip_loss(&self) -> f64 {
        let alpha_power_per_nm = self.loss_db_per_cm * 10.0_f64.ln() / 10.0 / 1.0e7;
        let l = self.circumference_nm();
        (-alpha_power_per_nm * l / 2.0).exp()
    }

    /// Through-port intensity transmission |E_t/E_in|² at `lambda_nm`.
    ///
    /// T = |t - a exp(iφ)|² / |1 - t·a exp(iφ)|²
    pub fn through_transmission(&self, lambda_nm: f64) -> f64 {
        let t = (1.0 - self.coupling_coeff * self.coupling_coeff).sqrt();
        let a = self.round_trip_loss();
        let phi = self.round_trip_phase(lambda_nm);
        let exp_iphi = Complex64::new(phi.cos(), phi.sin());
        let numerator = Complex64::new(t, 0.0) - a * exp_iphi;
        let denominator = Complex64::new(1.0, 0.0) - t * a * exp_iphi;
        (numerator / denominator).norm_sqr()
    }

    /// Drop-port intensity transmission in add-drop configuration.
    ///
    /// Uses the two-coupler transfer matrix with this ring's coupling as κ₁
    /// and `coupling_coeff2` as κ₂.
    ///
    /// T_drop = (1-t₁²)(1-t₂²) · a / |1 - t₁ t₂ a exp(iφ)|²
    ///        = κ₁² κ₂² a / |1 - t₁ t₂ a exp(iφ)|²
    pub fn drop_transmission(&self, lambda_nm: f64, coupling_coeff2: f64) -> f64 {
        let t1 = (1.0 - self.coupling_coeff * self.coupling_coeff).sqrt();
        let t2 = (1.0 - coupling_coeff2 * coupling_coeff2).sqrt();
        let a = self.round_trip_loss();
        let phi = self.round_trip_phase(lambda_nm);
        let exp_iphi = Complex64::new(phi.cos(), phi.sin());
        let kappa1_sq = 1.0 - t1 * t1;
        let kappa2_sq = 1.0 - t2 * t2;
        let denominator = Complex64::new(1.0, 0.0) - t1 * t2 * a * exp_iphi;
        kappa1_sq * kappa2_sq * a / denominator.norm_sqr()
    }

    /// Free spectral range (FSR) in nm.
    ///
    /// FSR = λ² / (n_g · L)
    pub fn fsr_nm(&self) -> f64 {
        let l = self.circumference_nm();
        self.wavelength_nm * self.wavelength_nm / (self.n_g * l)
    }

    /// FWHM linewidth in nm (power linewidth of the Lorentzian resonance).
    ///
    /// Δλ = FSR · (1 - t·a) / (π · sqrt(t·a))
    pub fn linewidth_nm(&self) -> f64 {
        let t = (1.0 - self.coupling_coeff * self.coupling_coeff).sqrt();
        let a = self.round_trip_loss();
        let rt = t * a;
        // Exact formula from transfer matrix analysis
        let fsr = self.fsr_nm();
        fsr * (1.0 - rt) / (PI * rt.sqrt())
    }

    /// Quality factor Q = λ / Δλ.
    pub fn quality_factor(&self) -> f64 {
        self.wavelength_nm / self.linewidth_nm()
    }

    /// Finesse F = FSR / FWHM.
    pub fn finesse(&self) -> f64 {
        self.fsr_nm() / self.linewidth_nm()
    }

    /// Extinction ratio at resonance in dB.
    ///
    /// ER = 10 log₁₀(T_max / T_min)
    /// For all-pass ring, T_min = ((t - a)/(1 - t·a))²
    pub fn extinction_ratio_db(&self) -> f64 {
        let t = (1.0 - self.coupling_coeff * self.coupling_coeff).sqrt();
        let a = self.round_trip_loss();
        // At resonance: φ = 0 (2π), cos(φ)=1
        let t_min_field = (t - a) / (1.0 - t * a);
        let t_min = t_min_field * t_min_field;
        // Off resonance (worst case): φ = π
        let t_max_field = (t + a) / (1.0 + t * a);
        let t_max = t_max_field * t_max_field;
        if t_min <= 0.0 {
            return f64::INFINITY;
        }
        10.0 * (t_max / t_min).log10()
    }

    /// Returns `true` when the critical coupling condition κ² ≈ 1 - a² holds
    /// within 1% tolerance.
    pub fn is_critically_coupled(&self) -> bool {
        let a = self.round_trip_loss();
        let kappa_sq = self.coupling_coeff * self.coupling_coeff;
        let critical_kappa_sq = 1.0 - a * a;
        (kappa_sq - critical_kappa_sq).abs() / critical_kappa_sq.max(1e-12) < 0.01
    }

    /// Compute the through-port transmission spectrum.
    ///
    /// Returns a `Vec<(lambda_nm, transmission)>` over the wavelength range
    /// \[`lambda_start_nm`, `lambda_end_nm`\] with `n_points` samples.
    ///
    /// # Errors
    /// Returns `OxiPhotonError::NumericalError` if `n_points < 2` or the
    /// wavelength range is invalid.
    pub fn spectrum(
        &self,
        lambda_start_nm: f64,
        lambda_end_nm: f64,
        n_points: usize,
    ) -> Result<Vec<(f64, f64)>> {
        if n_points < 2 {
            return Err(OxiPhotonError::NumericalError(
                "n_points must be >= 2".to_owned(),
            ));
        }
        if lambda_start_nm >= lambda_end_nm || lambda_start_nm <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "invalid wavelength range: [{lambda_start_nm}, {lambda_end_nm}]"
            )));
        }
        let step = (lambda_end_nm - lambda_start_nm) / (n_points - 1) as f64;
        let result = (0..n_points)
            .map(|i| {
                let lam = lambda_start_nm + i as f64 * step;
                (lam, self.through_transmission(lam))
            })
            .collect();
        Ok(result)
    }

    /// Find resonance wavelengths in \[`lambda_start_nm`, `lambda_end_nm`\].
    ///
    /// Resonances are wavelengths where the round-trip phase is a multiple of
    /// 2π: n_eff · L = m · λ  →  λ_m = n_eff · L / m.
    pub fn resonance_wavelengths(&self, lambda_start_nm: f64, lambda_end_nm: f64) -> Vec<f64> {
        let l = self.circumference_nm();
        // m ranges: λ = n_eff·L/m  →  m ∈ [n_eff·L/λ_end, n_eff·L/λ_start]
        let m_min = (self.n_eff * l / lambda_end_nm).ceil() as i64;
        let m_max = (self.n_eff * l / lambda_start_nm).floor() as i64;
        let mut resonances = Vec::new();
        for m in m_min..=m_max {
            if m <= 0 {
                continue;
            }
            let lam = self.n_eff * l / m as f64;
            if lam >= lambda_start_nm && lam <= lambda_end_nm {
                resonances.push(lam);
            }
        }
        resonances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        resonances
    }

    /// Thermo-optic wavelength shift: Δλ = λ · (dn/dT) / n_g · ΔT.
    ///
    /// For silicon: dn/dT ≈ 1.86 × 10⁻⁴ K⁻¹.
    pub fn thermal_shift_nm(&self, delta_t_k: f64, dn_dt: f64) -> f64 {
        self.wavelength_nm * dn_dt / self.n_g * delta_t_k
    }

    /// Electro-optic wavelength shift via plasma dispersion (Soref & Bennett model).
    ///
    /// Δn_e from free electrons, Δn_h from free holes (both negative for carrier
    /// injection).  Returns Δλ = λ · Δn / n_g.
    ///
    /// Soref model at 1550 nm:
    /// Δn_e = -8.8 × 10⁻²² · ΔN_e
    /// Δn_h = -8.5 × 10⁻¹⁸ · ΔN_h^0.8
    ///
    /// Arguments are carrier concentration changes (cm⁻³).
    pub fn electro_optic_shift_nm(&self, delta_n_e: f64, delta_n_h: f64) -> f64 {
        // Soref & Bennett (1987) coefficients at 1550 nm
        let delta_n_electrons = -8.8e-22 * delta_n_e;
        let delta_n_holes = if delta_n_h >= 0.0 {
            -8.5e-18 * delta_n_h.powf(0.8)
        } else {
            8.5e-18 * (-delta_n_h).powf(0.8)
        };
        let delta_n_total = delta_n_electrons + delta_n_holes;
        self.wavelength_nm * delta_n_total / self.n_g
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AddDropRing
// ─────────────────────────────────────────────────────────────────────────────

/// Add-drop ring resonator (two bus waveguides).
///
/// The through and drop port transfer functions are:
/// ```text
/// T_through = |t₁ - t₂·a·exp(iφ)|² / |1 - t₁·t₂·a·exp(iφ)|²
/// T_drop    = κ₁²·κ₂²·a / |1 - t₁·t₂·a·exp(iφ)|²
/// ```
#[derive(Debug, Clone)]
pub struct AddDropRing {
    /// The primary ring (contains the first coupler parameters).
    pub ring: AllPassRing,
    /// Field coupling coefficient of the second (drop) coupler κ₂.
    pub coupling_coeff2: f64,
}

impl AddDropRing {
    /// Create an add-drop ring from an `AllPassRing` and a second coupler.
    pub fn new(ring: AllPassRing, coupling_coeff2: f64) -> Self {
        Self {
            ring,
            coupling_coeff2,
        }
    }

    /// Through-port intensity transmission at `lambda_nm`.
    pub fn through_transmission(&self, lambda_nm: f64) -> f64 {
        let t1 = (1.0 - self.ring.coupling_coeff * self.ring.coupling_coeff).sqrt();
        let t2 = (1.0 - self.coupling_coeff2 * self.coupling_coeff2).sqrt();
        let a = self.ring.round_trip_loss();
        let phi = self.ring.round_trip_phase(lambda_nm);
        let exp_iphi = Complex64::new(phi.cos(), phi.sin());
        let numerator = Complex64::new(t1, 0.0) - t2 * a * exp_iphi;
        let denominator = Complex64::new(1.0, 0.0) - t1 * t2 * a * exp_iphi;
        (numerator / denominator).norm_sqr()
    }

    /// Drop-port intensity transmission at `lambda_nm`.
    pub fn drop_transmission(&self, lambda_nm: f64) -> f64 {
        self.ring.drop_transmission(lambda_nm, self.coupling_coeff2)
    }

    /// Through-port spectrum: `Vec<(lambda_nm, T_through)>`.
    pub fn through_spectrum(
        &self,
        lambda_start_nm: f64,
        lambda_end_nm: f64,
        n_points: usize,
    ) -> Result<Vec<(f64, f64)>> {
        if n_points < 2 {
            return Err(OxiPhotonError::NumericalError(
                "n_points must be >= 2".to_owned(),
            ));
        }
        if lambda_start_nm >= lambda_end_nm || lambda_start_nm <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "invalid wavelength range: [{lambda_start_nm}, {lambda_end_nm}]"
            )));
        }
        let step = (lambda_end_nm - lambda_start_nm) / (n_points - 1) as f64;
        Ok((0..n_points)
            .map(|i| {
                let lam = lambda_start_nm + i as f64 * step;
                (lam, self.through_transmission(lam))
            })
            .collect())
    }

    /// Drop-port spectrum: `Vec<(lambda_nm, T_drop)>`.
    pub fn drop_spectrum(
        &self,
        lambda_start_nm: f64,
        lambda_end_nm: f64,
        n_points: usize,
    ) -> Result<Vec<(f64, f64)>> {
        if n_points < 2 {
            return Err(OxiPhotonError::NumericalError(
                "n_points must be >= 2".to_owned(),
            ));
        }
        if lambda_start_nm >= lambda_end_nm || lambda_start_nm <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "invalid wavelength range: [{lambda_start_nm}, {lambda_end_nm}]"
            )));
        }
        let step = (lambda_end_nm - lambda_start_nm) / (n_points - 1) as f64;
        Ok((0..n_points)
            .map(|i| {
                let lam = lambda_start_nm + i as f64 * step;
                (lam, self.drop_transmission(lam))
            })
            .collect())
    }

    /// Insertion loss at the drop port at `lambda_nm` (dB, positive = loss).
    pub fn insertion_loss_db(&self, lambda_nm: f64) -> f64 {
        let t = self.drop_transmission(lambda_nm);
        if t <= 0.0 {
            return f64::INFINITY;
        }
        -10.0 * t.log10()
    }

    /// Crosstalk: ratio of off-resonance drop power to on-resonance drop power (dB).
    ///
    /// Evaluates drop at λ + FSR/2 (midway between resonances) vs. at resonance.
    pub fn crosstalk_db(&self, lambda_nm: f64) -> f64 {
        let fsr = self.ring.fsr_nm();
        let t_on = self.drop_transmission(lambda_nm);
        let t_off = self.drop_transmission(lambda_nm + fsr / 2.0);
        if t_on <= 0.0 || t_off <= 0.0 {
            return f64::NEG_INFINITY;
        }
        10.0 * (t_off / t_on).log10()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RingModulator
// ─────────────────────────────────────────────────────────────────────────────

/// Ring resonator modulator (carrier-depletion type).
///
/// Modulation is achieved by shifting the ring resonance via the plasma
/// dispersion effect.  The modulation depth is determined by the voltage swing
/// relative to the resonance linewidth.
#[derive(Debug, Clone)]
pub struct RingModulator {
    /// Underlying add-drop ring (defines Q, loss, coupling).
    pub ring: AddDropRing,
    /// Half-wave voltage Vπ (V) — voltage to shift resonance by one linewidth.
    pub vpi: f64,
    /// DC bias voltage (V).
    pub v_bias: f64,
    /// Electrical 3-dB bandwidth (GHz).
    pub bandwidth_ghz: f64,
}

impl RingModulator {
    /// Create a ring modulator.
    pub fn new(ring: AddDropRing, vpi: f64, v_bias: f64, bandwidth_ghz: f64) -> Self {
        Self {
            ring,
            vpi,
            v_bias,
            bandwidth_ghz,
        }
    }

    /// Modulation depth (extinction ratio in dB) for a voltage swing `delta_v` (V).
    ///
    /// The resonance shifts by Δλ = linewidth_nm * delta_v / Vπ, causing the
    /// drop transmission to vary between on-resonance and shifted states.
    pub fn modulation_depth(&self, delta_v: f64) -> f64 {
        let linewidth = self.ring.ring.linewidth_nm();
        let lambda0 = self.ring.ring.wavelength_nm;
        // Resonance shift due to voltage swing
        let delta_lambda = linewidth * delta_v / self.vpi;
        let t_on = self.ring.drop_transmission(lambda0);
        let t_off = self.ring.drop_transmission(lambda0 + delta_lambda);
        if t_on <= 0.0 || t_off <= 0.0 {
            return 0.0;
        }
        (10.0 * (t_on / t_off).log10()).abs()
    }

    /// Eye opening in dB for a peak-to-peak voltage swing `v_swing` (V).
    ///
    /// Eye opening ≈ ER for an ideal NRZ signal.
    pub fn eye_opening_db(&self, v_swing: f64) -> f64 {
        self.modulation_depth(v_swing)
    }

    /// Normalized electrical frequency response |H(f)|² (linear, 0–1).
    ///
    /// Single-pole RC model: |H(f)|² = 1 / (1 + (f/f_3dB)²).
    pub fn frequency_response(&self, freq_ghz: f64) -> f64 {
        let f_ratio = freq_ghz / self.bandwidth_ghz;
        1.0 / (1.0 + f_ratio * f_ratio)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a canonical silicon ring (R=10 μm, n_eff=2.4, n_g=4.2,
    /// loss=3 dB/cm, κ=0.1, λ=1550 nm).
    fn si_ring() -> AllPassRing {
        AllPassRing::new(10.0, 2.4, 4.2, 3.0, 0.1, 1550.0)
    }

    #[test]
    fn test_ring_fsr() {
        let ring = si_ring();
        let l_nm = 2.0 * PI * 10.0 * 1_000.0; // nm
        let expected_fsr = 1550.0_f64.powi(2) / (4.2 * l_nm);
        let fsr = ring.fsr_nm();
        assert!(
            (fsr - expected_fsr).abs() / expected_fsr < 1e-10,
            "FSR mismatch: got {fsr:.4} nm, expected {expected_fsr:.4} nm"
        );
    }

    #[test]
    fn test_ring_quality_factor() {
        let ring = si_ring();
        let q = ring.quality_factor();
        let linewidth = ring.linewidth_nm();
        let q_check = 1550.0 / linewidth;
        assert!(
            (q - q_check).abs() / q < 1e-10,
            "Q inconsistency: Q={q:.1}, λ/Δλ={q_check:.1}"
        );
        // Silicon ring resonators at 1550 nm typically have Q > 1000
        assert!(q > 1_000.0, "Q too low for silicon ring: Q={q:.1}");
    }

    #[test]
    fn test_critical_coupling() {
        // Build a ring where κ² = 1 - a²
        let ring_base = AllPassRing::new(10.0, 2.4, 4.2, 3.0, 0.1, 1550.0);
        let a = ring_base.round_trip_loss();
        let kappa_critical = (1.0 - a * a).sqrt();
        let ring_cc = AllPassRing::new(10.0, 2.4, 4.2, 3.0, kappa_critical, 1550.0);
        // At resonance, through transmission should be ~0
        // Resonance: φ = 2π*m, use wavelength_nm as approximate resonance
        // Find nearest resonance
        let resonances = ring_cc.resonance_wavelengths(1545.0, 1555.0);
        assert!(!resonances.is_empty(), "No resonances found in range");
        let lam_res = resonances[resonances.len() / 2];
        let t_res = ring_cc.through_transmission(lam_res);
        assert!(
            t_res < 0.001,
            "Critical coupling: T at resonance should be ~0, got {t_res:.6}"
        );
        assert!(
            ring_cc.is_critically_coupled(),
            "is_critically_coupled() returned false"
        );
    }

    #[test]
    fn test_through_transmission_off_resonance() {
        let ring = si_ring();
        // Evaluate at a wavelength that is half FSR away from any resonance.
        // At φ = π (anti-resonance), T should be close to 1 for low-loss ring.
        let fsr = ring.fsr_nm();
        let resonances = ring.resonance_wavelengths(1540.0, 1560.0);
        assert!(!resonances.is_empty());
        let lam_anti = resonances[0] + fsr / 2.0;
        let t_anti = ring.through_transmission(lam_anti);
        assert!(
            t_anti > 0.95,
            "Anti-resonance transmission should be > 0.95, got {t_anti:.4}"
        );
    }

    #[test]
    fn test_add_drop_energy_conservation() {
        let ring_base = AllPassRing::new(10.0, 2.4, 4.2, 3.0, 0.15, 1550.0);
        let add_drop = AddDropRing::new(ring_base, 0.15);
        // At every wavelength, T_through + T_drop <= 1 (loss can absorb the rest)
        let n_pts = 500;
        let start = 1548.0;
        let end = 1552.0;
        for i in 0..n_pts {
            let lam = start + (end - start) * i as f64 / (n_pts - 1) as f64;
            let t_t = add_drop.through_transmission(lam);
            let t_d = add_drop.drop_transmission(lam);
            let total = t_t + t_d;
            assert!(
                total <= 1.0 + 1e-9,
                "Energy conservation violated at λ={lam:.2} nm: T_t+T_d={total:.6}"
            );
        }
    }

    #[test]
    fn test_thermal_shift() {
        let ring = si_ring();
        // dn/dT for silicon ≈ 1.86e-4 K⁻¹ → positive heating gives red shift
        let dn_dt = 1.86e-4_f64;
        let shift = ring.thermal_shift_nm(10.0, dn_dt); // +10 K
        assert!(
            shift > 0.0,
            "Positive ΔT with positive dn/dT should give red shift: {shift}"
        );
        let shift_neg = ring.thermal_shift_nm(-10.0, dn_dt);
        assert!(
            shift_neg < 0.0,
            "Negative ΔT should give blue shift: {shift_neg}"
        );
    }

    #[test]
    fn test_spectrum_has_dips() {
        let ring = si_ring();
        let spec = ring
            .spectrum(1540.0, 1560.0, 2000)
            .expect("spectrum should succeed");
        // Find the minimum transmission — it should be well below 1
        let t_min = spec.iter().map(|(_, t)| *t).fold(f64::INFINITY, f64::min);
        assert!(
            t_min < 0.5,
            "Spectrum should have dips below 0.5, min={t_min:.4}"
        );
        // And a maximum near 1.0
        let t_max = spec
            .iter()
            .map(|(_, t)| *t)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            t_max > 0.9,
            "Spectrum maximum should be > 0.9, max={t_max:.4}"
        );
    }

    #[test]
    fn test_ring_modulator_frequency_response() {
        let ring_base = AllPassRing::new(5.0, 2.4, 4.2, 2.0, 0.2, 1550.0);
        let add_drop = AddDropRing::new(ring_base, 0.2);
        let modulator = RingModulator::new(add_drop, 1.0, -1.0, 40.0);
        // At DC, response = 1
        let h_dc = modulator.frequency_response(0.0);
        assert!(
            (h_dc - 1.0).abs() < 1e-10,
            "DC response should be 1, got {h_dc}"
        );
        // At 3-dB bandwidth, response = 0.5
        let h_3db = modulator.frequency_response(40.0);
        assert!(
            (h_3db - 0.5).abs() < 1e-10,
            "3-dB response should be 0.5, got {h_3db}"
        );
        // At 2× bandwidth, response < 0.5
        let h_2bw = modulator.frequency_response(80.0);
        assert!(
            h_2bw < 0.5,
            "Response above bandwidth should be < 0.5, got {h_2bw}"
        );
    }

    #[test]
    fn test_resonance_wavelengths_count() {
        let ring = si_ring();
        // The resonance condition is n_eff * L = m * λ.
        // Consecutive resonances are separated by λ²/(n_eff * L) (phase-index FSR).
        let l_nm = 2.0 * PI * ring.radius_um * 1_000.0;
        let fsr_phase = ring.wavelength_nm * ring.wavelength_nm / (ring.n_eff * l_nm);
        // Scan over 10× phase FSR to find ~10 resonances
        let span = 10.0 * fsr_phase;
        let resonances = ring.resonance_wavelengths(1550.0 - span / 2.0, 1550.0 + span / 2.0);
        assert!(
            resonances.len() >= 8 && resonances.len() <= 12,
            "Expected ~10 resonances in 10×phase-FSR span, found {}",
            resonances.len()
        );
        // Verify that consecutive resonances have approximately phase-index FSR spacing
        if resonances.len() >= 2 {
            for i in 1..resonances.len() {
                let spacing = resonances[i] - resonances[i - 1];
                // Use local FSR at average wavelength between the two resonances
                let lam_avg = (resonances[i] + resonances[i - 1]) / 2.0;
                let local_fsr = lam_avg * lam_avg / (ring.n_eff * l_nm);
                assert!(
                    (spacing - local_fsr).abs() / local_fsr < 0.05,
                    "Resonance spacing {spacing:.4} nm differs from local phase FSR {local_fsr:.4} nm"
                );
            }
        }
    }

    #[test]
    fn test_electro_optic_shift_direction() {
        let ring = si_ring();
        // Carrier injection (positive ΔN) → negative Δn → blue shift (negative Δλ)
        let shift = ring.electro_optic_shift_nm(1e18, 1e18);
        assert!(
            shift < 0.0,
            "Carrier injection should give blue shift: {shift:.6}"
        );
    }

    #[test]
    fn test_round_trip_loss_lossless_limit() {
        // Zero loss → a = 1
        let ring = AllPassRing::new(10.0, 2.4, 4.2, 0.0, 0.1, 1550.0);
        let a = ring.round_trip_loss();
        assert!(
            (a - 1.0).abs() < 1e-12,
            "Zero-loss ring should have a=1, got {a}"
        );
    }

    #[test]
    fn test_c_nm_per_s_constant() {
        // Sanity check: C_NM_PER_S corresponds to c in nm/s
        // c ≈ 2.998e8 m/s = 2.998e17 nm/s
        assert!((C_NM_PER_S - 2.997_924_58e17).abs() / 2.997_924_58e17 < 1e-8);
    }
}
