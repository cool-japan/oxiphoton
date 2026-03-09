//! Mach-Zehnder Interferometer (MZI) models for silicon photonics.
//!
//! Implements the MZI using 2×2 transfer matrices for the directional couplers
//! and a phase-shift arm matrix.  Both balanced (50:50) and imbalanced splits
//! are supported.
//!
//! # Transfer matrix convention
//!
//! Each directional coupler with power coupling ratio κ² is modelled as:
//! ```text
//! [E_out1]   [  t    iκ ] [E_in1]
//! [E_out2] = [ iκ     t ] [E_in2]
//! ```
//! where `t = sqrt(1 - κ²)` and `κ = sqrt(coupling_ratio)`.
//!
//! The arm propagation matrix is diagonal:
//! ```text
//! P = diag(exp(iβL₁·a₁), exp(iβL₂·a₂))
//! ```
//! where the differential path length ΔL determines the interference.

use num_complex::Complex64;
use std::f64::consts::PI;

use crate::error::{OxiPhotonError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Matrix helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Multiply two 2×2 complex matrices.
#[inline]
fn mat2_mul(a: [[Complex64; 2]; 2], b: [[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// MachZehnderInterferometer
// ─────────────────────────────────────────────────────────────────────────────

/// Mach-Zehnder Interferometer.
///
/// The MZI consists of:
/// 1. Input directional coupler (splitting ratio `coupling_ratio1`)
/// 2. Two arm waveguides with path difference `arm_length_um`
/// 3. Output directional coupler (combining ratio `coupling_ratio2`)
///
/// A static phase offset `delta_phi` (rad) can be applied to emulate biasing.
///
/// # Sign convention
/// The differential arm length is ΔL = L₂ − L₁ where arm 2 is the longer
/// arm.  The phase difference at wavelength λ is:
/// ```text
/// Δφ = 2π · n_eff · ΔL / λ + delta_phi
/// ```
#[derive(Debug, Clone)]
pub struct MachZehnderInterferometer {
    /// Effective refractive index of arm waveguides.
    pub n_eff: f64,
    /// Group refractive index of arm waveguides.
    pub n_g: f64,
    /// Differential arm length ΔL = L₂ − L₁ (μm).
    pub arm_length_um: f64,
    /// Propagation loss (dB/cm).
    pub loss_db_per_cm: f64,
    /// Input coupler power splitting ratio κ₁² ∈ (0, 1).
    pub coupling_ratio1: f64,
    /// Output coupler power splitting ratio κ₂² ∈ (0, 1).
    pub coupling_ratio2: f64,
    /// Static (bias) phase offset (rad).
    pub delta_phi: f64,
}

impl MachZehnderInterferometer {
    /// Create a new MZI with default zero static phase.
    ///
    /// # Arguments
    /// * `n_eff`          – effective index
    /// * `n_g`            – group index
    /// * `arm_length_um`  – differential arm length ΔL (μm)
    /// * `loss_db_per_cm` – waveguide propagation loss
    /// * `split1`         – input coupler power coupling ratio (0.5 = 50:50)
    /// * `split2`         – output coupler power coupling ratio
    pub fn new(
        n_eff: f64,
        n_g: f64,
        arm_length_um: f64,
        loss_db_per_cm: f64,
        split1: f64,
        split2: f64,
    ) -> Self {
        Self {
            n_eff,
            n_g,
            arm_length_um,
            loss_db_per_cm,
            coupling_ratio1: split1,
            coupling_ratio2: split2,
            delta_phi: 0.0,
        }
    }

    /// 2×2 transfer matrix for a directional coupler with power coupling κ².
    ///
    /// ```text
    /// DC = \[ t   iκ \]
    ///      \[ iκ   t \]
    /// ```
    /// where `t = sqrt(1 - kappa_sq)` and `κ = sqrt(kappa_sq)`.
    pub fn coupler_matrix(kappa_sq: f64) -> [[Complex64; 2]; 2] {
        let kappa_sq = kappa_sq.clamp(0.0, 1.0);
        let t = (1.0 - kappa_sq).sqrt();
        let kappa = kappa_sq.sqrt();
        [
            [Complex64::new(t, 0.0), Complex64::new(0.0, kappa)],
            [Complex64::new(0.0, kappa), Complex64::new(t, 0.0)],
        ]
    }

    /// Round-trip field amplitude propagation loss for arm of length `l_nm`.
    fn arm_field_loss(&self, l_nm: f64) -> f64 {
        let alpha_per_nm = self.loss_db_per_cm * 10.0_f64.ln() / 10.0 / 1.0e7;
        (-alpha_per_nm * l_nm / 2.0).exp()
    }

    /// Propagation phase for arm of effective optical path `n_eff * l_nm` at λ.
    #[inline]
    fn arm_phase(&self, l_nm: f64, lambda_nm: f64) -> f64 {
        2.0 * PI * self.n_eff * l_nm / lambda_nm
    }

    /// Full 2×2 MZI transfer matrix at `lambda_nm`.
    ///
    /// M = DC₂ · P · DC₁
    ///
    /// where P is the arm propagation matrix with differential phase delay and
    /// loss applied to arm 2 relative to arm 1 (arm 1 is the reference arm).
    pub fn transfer_matrix(&self, lambda_nm: f64) -> [[Complex64; 2]; 2] {
        let dc1 = Self::coupler_matrix(self.coupling_ratio1);
        let dc2 = Self::coupler_matrix(self.coupling_ratio2);

        // Differential arm length in nm
        let delta_l_nm = self.arm_length_um * 1_000.0;

        // Arm 1: reference arm (zero differential path, unity field loss reference)
        // Arm 2: extra path ΔL with phase Δφ = 2π n_eff ΔL / λ + delta_phi
        let phi2 = self.arm_phase(delta_l_nm, lambda_nm) + self.delta_phi;
        let a2 = self.arm_field_loss(delta_l_nm);

        // Propagation matrix: P = diag(1, a2 * exp(i*phi2))
        let p: [[Complex64; 2]; 2] = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [
                Complex64::new(0.0, 0.0),
                a2 * Complex64::new(phi2.cos(), phi2.sin()),
            ],
        ];

        // M = DC2 · P · DC1
        mat2_mul(dc2, mat2_mul(p, dc1))
    }

    /// Bar-port (through) intensity transmission |S₂₁|² = |M\[1\]\[0\]|².
    ///
    /// Input is port 1 (index 0); bar output is port 2 (index 1).
    pub fn bar_transmission(&self, lambda_nm: f64) -> f64 {
        let m = self.transfer_matrix(lambda_nm);
        m[1][0].norm_sqr()
    }

    /// Cross-port intensity transmission |S₁₁|² = |M\[0\]\[0\]|².
    ///
    /// Input is port 1 (index 0); cross output is port 1 on the other side
    /// (index 0 of output).
    pub fn cross_transmission(&self, lambda_nm: f64) -> f64 {
        let m = self.transfer_matrix(lambda_nm);
        m[0][0].norm_sqr()
    }

    /// Transmission spectrum.
    ///
    /// Returns `Vec<(lambda_nm, T_bar, T_cross)>`.
    pub fn spectrum(
        &self,
        lambda_start_nm: f64,
        lambda_end_nm: f64,
        n_pts: usize,
    ) -> Result<Vec<(f64, f64, f64)>> {
        if n_pts < 2 {
            return Err(OxiPhotonError::NumericalError(
                "n_pts must be >= 2".to_owned(),
            ));
        }
        if lambda_start_nm >= lambda_end_nm || lambda_start_nm <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "invalid wavelength range: [{lambda_start_nm}, {lambda_end_nm}]"
            )));
        }
        let step = (lambda_end_nm - lambda_start_nm) / (n_pts - 1) as f64;
        Ok((0..n_pts)
            .map(|i| {
                let lam = lambda_start_nm + i as f64 * step;
                (
                    lam,
                    self.bar_transmission(lam),
                    self.cross_transmission(lam),
                )
            })
            .collect())
    }

    /// Free spectral range of the MZI: FSR = λ² / (n_g · ΔL).
    ///
    /// Returns `None` if `arm_length_um` is zero (balanced MZI has no fringes).
    pub fn fsr_nm(&self) -> Option<f64> {
        if self.arm_length_um == 0.0 {
            return None;
        }
        let delta_l_nm = self.arm_length_um * 1_000.0;
        // Use the center wavelength from design; caller can set coupling_ratio1 center
        // We use 1550 nm as default reference wavelength.
        let lambda_ref = 1550.0_f64;
        Some(lambda_ref * lambda_ref / (self.n_g * delta_l_nm))
    }

    /// Free spectral range at a given reference wavelength.
    pub fn fsr_at_nm(&self, lambda_nm: f64) -> Option<f64> {
        if self.arm_length_um == 0.0 {
            return None;
        }
        let delta_l_nm = self.arm_length_um * 1_000.0;
        Some(lambda_nm * lambda_nm / (self.n_g * delta_l_nm))
    }

    /// Peak extinction ratio in dB for the bar port.
    ///
    /// For a symmetric MZI (κ₁² = κ₂² = 0.5), ideal ER → ∞.
    /// Computed analytically from the coupler ratios and arm loss for maximum
    /// accuracy, scanning over one FSR to locate the true max and min.
    pub fn extinction_ratio_db(&self) -> f64 {
        // For a lossless 2×2 MZI the bar-port amplitude is:
        //   E_bar = -κ₁κ₂ + t₁t₂ exp(iΔφ)  (schematic)
        // The maximum and minimum occur at Δφ = 0 and Δφ = π respectively.
        // We evaluate at two phase-quadrature wavelengths and the direct
        // analytical max/min for lossless case.
        let fsr = match self.fsr_nm() {
            Some(f) => f,
            None => return 0.0,
        };
        // Scan densely over TWO FSRs starting from a reference point that
        // guarantees capturing both the peak and trough of the fringe pattern.
        let lambda_ref = 1550.0_f64;
        let n_scan = 1000_usize;
        let scan_span = 2.0 * fsr;
        let step = scan_span / (n_scan - 1) as f64;
        let mut t_max = f64::NEG_INFINITY;
        let mut t_min = f64::INFINITY;
        for i in 0..n_scan {
            let lam = lambda_ref + i as f64 * step;
            let t = self.bar_transmission(lam);
            if t > t_max {
                t_max = t;
            }
            if t < t_min {
                t_min = t;
            }
        }
        if t_min <= 1e-30 {
            return f64::INFINITY;
        }
        10.0 * (t_max / t_min).log10()
    }

    /// Apply an additional static phase shift (rad) to arm 2.
    pub fn apply_phase_shift(&mut self, delta_phi: f64) {
        self.delta_phi += delta_phi;
    }

    /// Required effective index change Δn to achieve a π phase shift.
    ///
    /// Δφ = 2π ΔL Δn / λ = π  →  Δn = λ / (2 ΔL)
    pub fn index_change_for_pi(&self, lambda_nm: f64) -> f64 {
        let delta_l_nm = self.arm_length_um * 1_000.0;
        if delta_l_nm == 0.0 {
            return f64::INFINITY;
        }
        lambda_nm / (2.0 * delta_l_nm)
    }

    /// Cascade this MZI with a `second` MZI: M_total = M_second · M_first.
    pub fn cascade(
        &self,
        second: &MachZehnderInterferometer,
        lambda_nm: f64,
    ) -> [[Complex64; 2]; 2] {
        let m1 = self.transfer_matrix(lambda_nm);
        let m2 = second.transfer_matrix(lambda_nm);
        mat2_mul(m2, m1)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MziSwitch
// ─────────────────────────────────────────────────────────────────────────────

/// Switch state of an MZI optical switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchState {
    /// Output appears at the bar port (straight-through).
    Bar,
    /// Output appears at the cross port.
    Cross,
}

/// MZI used as a 2×2 optical switch.
///
/// The switch state is set by adjusting the bias phase of the MZI.  Switching
/// voltage is determined by the required π phase shift.
#[derive(Debug, Clone)]
pub struct MziSwitch {
    /// The underlying MZI.
    pub mzi: MachZehnderInterferometer,
    /// Current switch state.
    pub state: SwitchState,
}

impl MziSwitch {
    /// Create an MZI switch in the Bar state.
    pub fn new(mzi: MachZehnderInterferometer) -> Self {
        Self {
            mzi,
            state: SwitchState::Bar,
        }
    }

    /// Set the switch state by adjusting the static phase bias.
    ///
    /// Bar state: Δφ = 0 (or 2π) — bar port has maximum transmission.
    /// Cross state: Δφ = π — cross port has maximum transmission.
    pub fn set_state(&mut self, state: SwitchState) {
        let old_phi = self.mzi.delta_phi;
        // Remove old state bias
        let base_phi = match self.state {
            SwitchState::Bar => old_phi,
            SwitchState::Cross => old_phi - PI,
        };
        self.state = state;
        self.mzi.delta_phi = match state {
            SwitchState::Bar => base_phi,
            SwitchState::Cross => base_phi + PI,
        };
    }

    /// Required voltage swing to switch (assuming Vπ is defined externally).
    ///
    /// Returns the normalised switching voltage = 1 Vπ (caller scales by Vπ).
    pub fn switching_voltage(&self) -> f64 {
        1.0 // 1 × Vπ
    }

    /// Isolation (dB) at `lambda_nm`: ratio of intended to unintended port power.
    pub fn isolation_db(&self, lambda_nm: f64) -> f64 {
        let t_bar = self.mzi.bar_transmission(lambda_nm);
        let t_cross = self.mzi.cross_transmission(lambda_nm);
        match self.state {
            SwitchState::Bar => {
                if t_cross <= 0.0 {
                    return f64::INFINITY;
                }
                10.0 * (t_bar / t_cross).log10()
            }
            SwitchState::Cross => {
                if t_bar <= 0.0 {
                    return f64::INFINITY;
                }
                10.0 * (t_cross / t_bar).log10()
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Canonical MZI: 50:50 splitters, ΔL = 100 μm, n_eff=2.4, n_g=4.2.
    fn balanced_mzi() -> MachZehnderInterferometer {
        MachZehnderInterferometer::new(2.4, 4.2, 100.0, 0.0, 0.5, 0.5)
    }

    /// Lossless 50:50 coupler matrix verification.
    #[test]
    fn test_50_50_coupler_matrix() {
        let m = MachZehnderInterferometer::coupler_matrix(0.5);
        // t = 1/sqrt(2), κ = 1/sqrt(2)
        let t = (0.5_f64).sqrt();
        assert!((m[0][0].re - t).abs() < 1e-12, "m[0][0].re should be t");
        assert!((m[0][0].im).abs() < 1e-12, "m[0][0].im should be 0");
        assert!((m[0][1].re).abs() < 1e-12, "m[0][1].re should be 0");
        assert!(
            (m[0][1].im - t).abs() < 1e-12,
            "m[0][1].im should be κ=t for 50:50"
        );
        // Check unitarity: M† M = I
        let m00_sq = m[0][0].norm_sqr() + m[1][0].norm_sqr();
        let m11_sq = m[0][1].norm_sqr() + m[1][1].norm_sqr();
        assert!(
            (m00_sq - 1.0).abs() < 1e-12,
            "Column 0 norm sq should be 1, got {m00_sq}"
        );
        assert!(
            (m11_sq - 1.0).abs() < 1e-12,
            "Column 1 norm sq should be 1, got {m11_sq}"
        );
    }

    #[test]
    fn test_mzi_bar_cross_sum() {
        let mzi = balanced_mzi();
        let n_pts = 300;
        let start = 1540.0_f64;
        let end = 1560.0_f64;
        for i in 0..n_pts {
            let lam = start + (end - start) * i as f64 / (n_pts - 1) as f64;
            let t_bar = mzi.bar_transmission(lam);
            let t_cross = mzi.cross_transmission(lam);
            let total = t_bar + t_cross;
            assert!(
                total <= 1.0 + 1e-9,
                "Energy violation at λ={lam:.2}: T_bar+T_cross={total:.8}"
            );
            // For lossless MZI, total should be very close to 1
            assert!(
                total > 0.999,
                "Lossless MZI should conserve energy at λ={lam:.2}: total={total:.8}"
            );
        }
    }

    #[test]
    fn test_mzi_fsr() {
        let mzi = balanced_mzi();
        let fsr = mzi.fsr_at_nm(1550.0).expect("should have FSR");
        let expected = 1550.0_f64.powi(2) / (4.2 * 100.0 * 1_000.0);
        assert!(
            (fsr - expected).abs() / expected < 1e-10,
            "FSR mismatch: got {fsr:.4} nm, expected {expected:.4} nm"
        );
    }

    #[test]
    fn test_mzi_extinction() {
        // Ideal lossless 50:50 MZI should have very high extinction ratio
        let mzi = balanced_mzi();
        let er = mzi.extinction_ratio_db();
        assert!(
            er > 30.0,
            "Ideal 50:50 MZI should have ER > 30 dB, got {er:.1} dB"
        );
    }

    #[test]
    fn test_transfer_matrix_unitary() {
        // Lossless MZI: M†M should be identity
        let mzi = MachZehnderInterferometer::new(2.4, 4.2, 50.0, 0.0, 0.5, 0.5);
        let lambda = 1550.0_f64;
        let m = mzi.transfer_matrix(lambda);
        // M†M: compute manually
        // (M†)_{ij} = conj(M_{ji})
        // (M†M)_{ij} = sum_k conj(M_{ki}) * M_{kj}
        let m_dag_m_00 = m[0][0].conj() * m[0][0] + m[1][0].conj() * m[1][0];
        let m_dag_m_11 = m[0][1].conj() * m[0][1] + m[1][1].conj() * m[1][1];
        let m_dag_m_01 = m[0][0].conj() * m[0][1] + m[1][0].conj() * m[1][1];
        assert!(
            (m_dag_m_00.re - 1.0).abs() < 1e-12 && m_dag_m_00.im.abs() < 1e-12,
            "M†M[0][0] should be 1, got {m_dag_m_00}"
        );
        assert!(
            (m_dag_m_11.re - 1.0).abs() < 1e-12 && m_dag_m_11.im.abs() < 1e-12,
            "M†M[1][1] should be 1, got {m_dag_m_11}"
        );
        assert!(
            m_dag_m_01.norm() < 1e-12,
            "M†M[0][1] should be 0, got {m_dag_m_01}"
        );
    }

    #[test]
    fn test_mzi_switch_bar_cross() {
        let mzi = MachZehnderInterferometer::new(2.4, 4.2, 100.0, 0.0, 0.5, 0.5);
        let mut sw = MziSwitch::new(mzi);
        // In Bar state, bar transmission should dominate
        let t_bar_bar = sw.mzi.bar_transmission(1550.0);
        let t_cross_bar = sw.mzi.cross_transmission(1550.0);
        // Switch to Cross state
        sw.set_state(SwitchState::Cross);
        let t_bar_cross = sw.mzi.bar_transmission(1550.0);
        let t_cross_cross = sw.mzi.cross_transmission(1550.0);
        // After switching to Cross, cross should be larger than bar
        assert!(
            t_cross_cross > t_bar_cross,
            "In Cross state, cross port should dominate: T_cross={t_cross_cross:.4}, T_bar={t_bar_cross:.4}"
        );
        // And it should be approximately what bar was before switching
        assert!(
            (t_cross_cross - t_bar_bar).abs() < 0.01,
            "Cross state cross transmission should match Bar state bar transmission"
        );
        // Suppress unused warning
        let _ = t_cross_bar;
    }

    #[test]
    fn test_index_change_for_pi() {
        let mzi = balanced_mzi();
        let delta_n = mzi.index_change_for_pi(1550.0);
        // Δn = λ/(2 ΔL) = 1550 nm / (2 × 100 μm × 1000 nm/μm) = 1550/200000 = 7.75e-3
        let expected = 1550.0 / (2.0 * 100.0 * 1_000.0);
        assert!(
            (delta_n - expected).abs() / expected < 1e-10,
            "Δn for π shift: got {delta_n:.6}, expected {expected:.6}"
        );
    }

    #[test]
    fn test_cascade_matrix() {
        // Cascading two identical balanced MZIs should give a valid matrix
        let mzi1 = balanced_mzi();
        let mzi2 = balanced_mzi();
        let m_cascade = mzi1.cascade(&mzi2, 1550.0);
        // Check that cascade matrix preserves unitarity (lossless case)
        let m00_sq = m_cascade[0][0].norm_sqr() + m_cascade[1][0].norm_sqr();
        assert!(
            (m00_sq - 1.0).abs() < 1e-10,
            "Cascade matrix column 0 should be unit: {m00_sq:.8}"
        );
    }

    #[test]
    fn test_spectrum_energy_conservation() {
        let mzi = balanced_mzi();
        let spec = mzi.spectrum(1540.0, 1560.0, 500).expect("spectrum");
        for (lam, t_bar, t_cross) in &spec {
            let total = t_bar + t_cross;
            assert!(
                total <= 1.0 + 1e-9 && total > 0.999,
                "Energy violation at λ={lam:.2}: total={total:.8}"
            );
        }
    }
}
