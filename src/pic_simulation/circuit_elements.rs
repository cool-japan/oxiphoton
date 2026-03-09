/// PIC circuit building blocks: directional couplers, microring resonators, MZIs.
use std::f64::consts::PI;

/// Complex number (Re, Im) — lightweight, no external dependency.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex(pub f64, pub f64);

impl Complex {
    /// Create from Cartesian components.
    #[inline]
    pub fn new(re: f64, im: f64) -> Self {
        Self(re, im)
    }

    /// Create from polar form r·e^{iθ}.
    #[inline]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self(r * theta.cos(), r * theta.sin())
    }

    /// Absolute value (modulus).
    #[inline]
    pub fn abs(&self) -> f64 {
        (self.0 * self.0 + self.1 * self.1).sqrt()
    }

    /// Squared modulus |z|².
    #[inline]
    pub fn abs2(&self) -> f64 {
        self.0 * self.0 + self.1 * self.1
    }

    /// Complex conjugate.
    #[inline]
    pub fn conj(&self) -> Self {
        Self(self.0, -self.1)
    }

    /// Complex multiplication.
    #[inline]
    pub fn mul(&self, other: &Self) -> Self {
        Self(
            self.0 * other.0 - self.1 * other.1,
            self.0 * other.1 + self.1 * other.0,
        )
    }

    /// Complex addition.
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        Self(self.0 + other.0, self.1 + other.1)
    }

    /// Scalar multiplication.
    #[inline]
    pub fn scale(&self, s: f64) -> Self {
        Self(self.0 * s, self.1 * s)
    }

    /// Subtraction.
    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        Self(self.0 - other.0, self.1 - other.1)
    }

    /// Reciprocal 1/z.
    #[inline]
    pub fn recip(&self) -> Self {
        let d = self.abs2();
        Self(self.0 / d, -self.1 / d)
    }

    /// Division self / other.
    #[inline]
    pub fn div(&self, other: &Self) -> Self {
        self.mul(&other.recip())
    }
}

impl std::fmt::Display for Complex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.1 >= 0.0 {
            write!(f, "{:.6}+{:.6}i", self.0, self.1)
        } else {
            write!(f, "{:.6}{:.6}i", self.0, self.1)
        }
    }
}

// ---------------------------------------------------------------------------
// 2×2 Transfer Matrix
// ---------------------------------------------------------------------------

/// 2×2 transfer matrix for a 2-port photonic element.
///
/// Convention: `[b1, b2] = M × [a1, a2]` where a_i are input amplitudes
/// and b_i are output amplitudes.
#[derive(Clone, Copy, Debug)]
pub struct TransferMatrix2x2 {
    pub m: [[Complex; 2]; 2],
}

impl TransferMatrix2x2 {
    /// Identity matrix.
    pub fn identity() -> Self {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        Self {
            m: [[o, z], [z, o]],
        }
    }

    /// Cascade two matrices: M_total = M_other × M_self (self propagates first).
    pub fn cascade(&self, other: &TransferMatrix2x2) -> TransferMatrix2x2 {
        // Standard 2×2 matrix multiplication: result[i][j] = sum_k other[i][k] * self[k][j]
        let mut result = [[Complex::new(0.0, 0.0); 2]; 2];
        for (i, res_row) in result.iter_mut().enumerate() {
            for (j, cell) in res_row.iter_mut().enumerate() {
                let mut sum = Complex::new(0.0, 0.0);
                for k in 0..2 {
                    sum = sum.add(&other.m[i][k].mul(&self.m[k][j]));
                }
                *cell = sum;
            }
        }
        TransferMatrix2x2 { m: result }
    }

    /// Apply matrix to input amplitudes: returns (b1, b2).
    pub fn apply(&self, a1: Complex, a2: Complex) -> (Complex, Complex) {
        let b1 = self.m[0][0].mul(&a1).add(&self.m[0][1].mul(&a2));
        let b2 = self.m[1][0].mul(&a1).add(&self.m[1][1].mul(&a2));
        (b1, b2)
    }

    /// Return the S-matrix (same as the transfer matrix for this convention).
    pub fn s_matrix(&self) -> [[Complex; 2]; 2] {
        self.m
    }

    /// Through transmission (port 1 in → port 1 out) with unit input at port 1.
    pub fn through_power(&self) -> f64 {
        let (b1, _) = self.apply(Complex::new(1.0, 0.0), Complex::new(0.0, 0.0));
        b1.abs2()
    }

    /// Cross transmission (port 1 in → port 2 out) with unit input at port 1.
    pub fn cross_power(&self) -> f64 {
        let (_, b2) = self.apply(Complex::new(1.0, 0.0), Complex::new(0.0, 0.0));
        b2.abs2()
    }
}

// ---------------------------------------------------------------------------
// Directional Coupler
// ---------------------------------------------------------------------------

/// Directional coupler / beam splitter.
///
/// Transfer matrix (lossless, with phase convention):
/// ```text
/// M = [[cos θ, i sin θ],
///      [i sin θ, cos θ]]
/// ```
/// where θ = arcsin(√κ) and κ is the power coupling ratio.
#[derive(Clone, Debug)]
pub struct DirectionalCoupler {
    /// Power coupling ratio κ ∈ [0, 1].
    pub coupling_ratio: f64,
    /// Excess (insertion) loss in dB.
    pub excess_loss_db: f64,
    /// Phase imbalance between through and cross paths (rad).
    pub phase_imbalance_rad: f64,
}

impl DirectionalCoupler {
    /// Create a new directional coupler with given coupling ratio.
    pub fn new(coupling_ratio: f64) -> Self {
        Self {
            coupling_ratio: coupling_ratio.clamp(0.0, 1.0),
            excess_loss_db: 0.0,
            phase_imbalance_rad: 0.0,
        }
    }

    /// 50/50 coupler (3 dB).
    pub fn new_50_50() -> Self {
        Self::new(0.5)
    }

    /// Coupling angle θ = arcsin(√κ).
    pub fn coupling_angle_rad(&self) -> f64 {
        self.coupling_ratio.sqrt().asin()
    }

    /// Through port power fraction.
    pub fn through_transmission(&self) -> f64 {
        1.0 - self.coupling_ratio
    }

    /// Cross port power fraction.
    pub fn cross_transmission(&self) -> f64 {
        self.coupling_ratio
    }

    /// Transfer matrix including excess loss and phase imbalance.
    pub fn transfer_matrix(&self) -> TransferMatrix2x2 {
        let theta = self.coupling_angle_rad();
        let loss_amp = 10.0_f64.powf(-self.excess_loss_db / 20.0);
        let cos_t = Complex::new(theta.cos() * loss_amp, 0.0);
        let i_sin_t = Complex::new(0.0, theta.sin() * loss_amp);
        // Apply phase imbalance to off-diagonal elements
        let phi = self.phase_imbalance_rad;
        let phase = Complex::from_polar(1.0, phi);
        let i_sin_phased = i_sin_t.mul(&phase);
        TransferMatrix2x2 {
            m: [[cos_t, i_sin_phased], [i_sin_phased, cos_t]],
        }
    }
}

// ---------------------------------------------------------------------------
// Microring Resonator
// ---------------------------------------------------------------------------

/// Microring resonator add-drop filter.
///
/// Coupled-mode theory model with bus coupling coefficients κ_1, κ_2
/// and round-trip amplitude loss α_rt.
#[derive(Clone, Debug)]
pub struct MicroringResonator {
    /// Ring radius (m).
    pub radius_m: f64,
    /// Bus-to-ring power coupling ratio κ_1.
    pub coupling_through: f64,
    /// Drop-port power coupling ratio κ_2.
    pub coupling_drop: f64,
    /// Round-trip amplitude transmission α_rt ∈ (0, 1].
    pub round_trip_loss: f64,
    /// Effective refractive index n_eff.
    pub effective_index: f64,
    /// Operating wavelength (m).
    pub wavelength_m: f64,
}

impl MicroringResonator {
    /// Create a new ring resonator with default coupling and loss values.
    pub fn new(radius_m: f64, n_eff: f64, wavelength_m: f64) -> Self {
        Self {
            radius_m,
            coupling_through: 0.1,
            coupling_drop: 0.1,
            round_trip_loss: 0.99,
            effective_index: n_eff,
            wavelength_m,
        }
    }

    /// Round-trip phase φ_rt = 2π n_eff L_rt / λ.
    pub fn round_trip_phase(&self) -> f64 {
        2.0 * PI * self.effective_index * 2.0 * PI * self.radius_m / self.wavelength_m
    }

    /// Free spectral range: FSR = λ² / (n_g × L_rt) [converted to nm].
    pub fn fsr_nm(&self, n_group: f64) -> f64 {
        self.wavelength_m * self.wavelength_m / (n_group * 2.0 * PI * self.radius_m) * 1e9
    }

    /// Loaded quality factor Q ≈ π √(α_rt t1 t2) / (1 − α_rt t1 t2) × n_eff L / λ.
    pub fn loaded_q(&self) -> f64 {
        let t1 = (1.0 - self.coupling_through).sqrt();
        let t2 = (1.0 - self.coupling_drop).sqrt();
        let a = self.round_trip_loss;
        let round_trip = a * t1 * t2;
        if round_trip >= 1.0 {
            return f64::INFINITY;
        }
        // Q = π √(round_trip) / (1 - round_trip) × n_eff × L_rt / λ
        let l_rt = 2.0 * PI * self.radius_m;
        PI * round_trip.sqrt() / (1.0 - round_trip) * self.effective_index * l_rt
            / self.wavelength_m
    }

    /// Through port transmission T_through(φ) = |t1 − a t2 e^{iφ}|² / |1 − a t1 t2 e^{iφ}|².
    pub fn through_transmission(&self, detuning_rad: f64) -> f64 {
        let t1 = (1.0 - self.coupling_through).sqrt();
        let t2 = (1.0 - self.coupling_drop).sqrt();
        let a = self.round_trip_loss;
        let phi = self.round_trip_phase() + detuning_rad;
        let exp_phi = Complex::from_polar(1.0, phi);
        // numerator: t1 - a*t2*e^{iφ}
        let num = Complex::new(t1, 0.0).sub(&exp_phi.scale(a * t2));
        // denominator: 1 - a*t1*t2*e^{iφ}
        let den = Complex::new(1.0, 0.0).sub(&exp_phi.scale(a * t1 * t2));
        num.abs2() / den.abs2().max(1e-30)
    }

    /// Drop port transmission T_drop(φ) = κ_1 κ_2 a / |1 − a t1 t2 e^{iφ}|².
    pub fn drop_transmission(&self, detuning_rad: f64) -> f64 {
        let t1 = (1.0 - self.coupling_through).sqrt();
        let t2 = (1.0 - self.coupling_drop).sqrt();
        let a = self.round_trip_loss;
        let phi = self.round_trip_phase() + detuning_rad;
        let exp_phi = Complex::from_polar(1.0, phi);
        let den = Complex::new(1.0, 0.0).sub(&exp_phi.scale(a * t1 * t2));
        (self.coupling_through * self.coupling_drop * a) / den.abs2().max(1e-30)
    }

    /// Extinction ratio at resonance (detuning = 0) in dB.
    pub fn extinction_ratio_db(&self) -> f64 {
        let t_on = self.through_transmission(0.0);
        let t_off = self.through_transmission(PI);
        if t_on < 1e-20 {
            return 60.0; // clamp at 60 dB
        }
        10.0 * (t_off / t_on.max(1e-20)).log10()
    }

    /// Finesse: F = π / (1 − √((1−κ1)(1−κ2)) × α_rt).
    pub fn finesse(&self) -> f64 {
        let t1 = (1.0 - self.coupling_through).sqrt();
        let t2 = (1.0 - self.coupling_drop).sqrt();
        let a = self.round_trip_loss;
        let denom = 1.0 - a * t1 * t2;
        if denom <= 0.0 {
            return f64::INFINITY;
        }
        PI / denom
    }
}

// ---------------------------------------------------------------------------
// Mach-Zehnder Interferometer
// ---------------------------------------------------------------------------

/// Mach-Zehnder Interferometer (MZI).
///
/// Consists of two directional couplers with a phase shift between arms.
#[derive(Clone, Debug)]
pub struct MachZehnderInterferometer {
    /// Input coupler.
    pub coupler1: DirectionalCoupler,
    /// Output coupler.
    pub coupler2: DirectionalCoupler,
    /// Electro-optic or thermo-optic phase bias in upper arm (rad).
    pub delta_phi: f64,
    /// Physical arm length difference ΔL (m).
    pub delta_length_m: f64,
    /// Effective refractive index of arms.
    pub n_eff: f64,
    /// Operating wavelength (m).
    pub wavelength_m: f64,
}

impl MachZehnderInterferometer {
    /// Symmetric MZI with 50/50 couplers and zero arm imbalance.
    pub fn symmetric() -> Self {
        Self {
            coupler1: DirectionalCoupler::new_50_50(),
            coupler2: DirectionalCoupler::new_50_50(),
            delta_phi: 0.0,
            delta_length_m: 0.0,
            n_eff: 2.4,
            wavelength_m: 1550e-9,
        }
    }

    /// Phase contribution from arm length difference: Δφ_ΔL = 2π n_eff ΔL / λ.
    pub fn phase_from_length_difference(&self) -> f64 {
        2.0 * PI * self.n_eff * self.delta_length_m / self.wavelength_m
    }

    /// Total phase including bias.
    pub fn total_phase(&self) -> f64 {
        self.phase_from_length_difference() + self.delta_phi
    }

    /// Through port power transmission: T = cos²(Δφ/2) for ideal 50/50 MZI.
    pub fn through_transmission(&self) -> f64 {
        (self.total_phase() / 2.0).cos().powi(2)
    }

    /// Cross port power transmission: T_cross = sin²(Δφ/2).
    pub fn cross_transmission(&self) -> f64 {
        (self.total_phase() / 2.0).sin().powi(2)
    }

    /// Full transfer matrix of the MZI (coupler1 → phase shift → coupler2).
    pub fn transfer_matrix(&self) -> TransferMatrix2x2 {
        let m1 = self.coupler1.transfer_matrix();
        // Phase shift matrix: arm 1 gets exp(iΔφ), arm 2 gets 1
        let phi = self.total_phase();
        let phase_m = TransferMatrix2x2 {
            m: [
                [Complex::from_polar(1.0, phi), Complex::new(0.0, 0.0)],
                [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
            ],
        };
        let m2 = self.coupler2.transfer_matrix();
        // Total: M2 × phase × M1
        let m_mid = phase_m.cascade(&m1);
        m2.cascade(&m_mid)
    }

    /// Electro-optic modulator response: total phase at given drive voltage.
    pub fn eo_modulator_response(&self, voltage_v: f64, v_pi: f64) -> f64 {
        self.total_phase() + PI * voltage_v / v_pi
    }

    /// Extinction ratio in dB for ideal balanced MZI (T_max / T_min).
    pub fn extinction_ratio_db(&self) -> f64 {
        // For ideal 50/50 MZI: T_max = 1, T_min → 0 (theoretically infinite ER)
        // In practice, limited by coupler imbalance
        let kappa1 = self.coupler1.coupling_ratio;
        let kappa2 = self.coupler2.coupling_ratio;
        let t1 = (1.0 - kappa1).sqrt();
        let c1 = kappa1.sqrt();
        let t2 = (1.0 - kappa2).sqrt();
        let c2 = kappa2.sqrt();
        // Maximum transmission
        let t_max = (t1 * t2 + c1 * c2).powi(2);
        // Minimum transmission
        let t_min = (t1 * t2 - c1 * c2).powi(2);
        if t_min < 1e-30 {
            return 60.0; // clamp
        }
        10.0 * (t_max / t_min).log10()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_polar_roundtrip() {
        let z = Complex::from_polar(2.0, 1.0);
        assert!((z.abs() - 2.0).abs() < 1e-12, "abs={}", z.abs());
    }

    #[test]
    fn complex_mul_conj_is_abs2() {
        let z = Complex::new(3.0, 4.0);
        let prod = z.mul(&z.conj());
        assert!((prod.0 - z.abs2()).abs() < 1e-12);
        assert!(prod.1.abs() < 1e-12);
    }

    #[test]
    fn directional_coupler_unitary() {
        let dc = DirectionalCoupler::new(0.3);
        let m = dc.transfer_matrix();
        let (b1, b2) = m.apply(Complex::new(1.0, 0.0), Complex::new(0.0, 0.0));
        let total = b1.abs2() + b2.abs2();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Power not conserved: {}",
            total
        );
    }

    #[test]
    fn directional_coupler_50_50_splits_equally() {
        let dc = DirectionalCoupler::new_50_50();
        let m = dc.transfer_matrix();
        let (b1, b2) = m.apply(Complex::new(1.0, 0.0), Complex::new(0.0, 0.0));
        assert!((b1.abs2() - 0.5).abs() < 1e-10, "through={}", b1.abs2());
        assert!((b2.abs2() - 0.5).abs() < 1e-10, "cross={}", b2.abs2());
    }

    #[test]
    fn ring_fsr_reasonable() {
        let ring = MicroringResonator::new(5e-6, 2.4, 1550e-9);
        let fsr = ring.fsr_nm(4.0);
        // FSR = 1550² / (4 × 2π × 5e-6) nm ≈ 19.1 nm
        assert!(fsr > 10.0 && fsr < 30.0, "FSR={} nm", fsr);
    }

    #[test]
    fn ring_drop_at_resonance() {
        let mut ring = MicroringResonator::new(5e-6, 2.4, 1550e-9);
        ring.coupling_through = 0.2;
        ring.coupling_drop = 0.2;
        ring.round_trip_loss = 0.98;
        // At exact resonance (detuning = -round_trip_phase) drop should be significant
        let phi_rt = ring.round_trip_phase();
        let t_drop = ring.drop_transmission(-phi_rt);
        assert!(t_drop > 0.0 && t_drop <= 1.0, "T_drop={}", t_drop);
    }

    #[test]
    fn mzi_on_state_full_transmission() {
        let mzi = MachZehnderInterferometer::symmetric();
        // At Δφ = 0: through = cos²(0) = 1
        assert!(
            (mzi.through_transmission() - 1.0).abs() < 1e-10,
            "T={}",
            mzi.through_transmission()
        );
    }

    #[test]
    fn mzi_off_state_zero_transmission() {
        let mut mzi = MachZehnderInterferometer::symmetric();
        mzi.delta_phi = PI;
        assert!(
            mzi.through_transmission() < 0.01,
            "T={}",
            mzi.through_transmission()
        );
    }

    #[test]
    fn transfer_matrix_cascade_identity() {
        let id = TransferMatrix2x2::identity();
        let dc = DirectionalCoupler::new(0.3).transfer_matrix();
        let result = dc.cascade(&id);
        let (b1_dc, b2_dc) = dc.apply(Complex::new(1.0, 0.0), Complex::new(0.0, 0.0));
        let (b1_r, b2_r) = result.apply(Complex::new(1.0, 0.0), Complex::new(0.0, 0.0));
        assert!((b1_dc.abs2() - b1_r.abs2()).abs() < 1e-12);
        assert!((b2_dc.abs2() - b2_r.abs2()).abs() < 1e-12);
    }
}
