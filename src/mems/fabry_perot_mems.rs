//! Tunable MEMS Fabry-Pérot cavity with electrostatic actuation.
//!
//! Models a parallel-plate MEMS cavity in which one mirror is suspended by a
//! mechanical spring. Applying a voltage between the plates deflects the
//! mirror electrostatically, tuning the cavity resonance. Beyond the
//! "pull-in" voltage the restoring spring force cannot overcome the
//! electrostatic attraction and the device collapses.
//!
//! # References
//! - Senturia, S.D. (2001). *Microsystem Design*. Kluwer.
//! - Yariv, A. & Yeh, P. (2007). *Photonics*. Oxford.

use std::f64::consts::PI;

/// Speed of light in vacuum (m/s).
const C: f64 = 299_792_458.0;
/// Permittivity of free space (F/m).
const EPSILON_0: f64 = 8.854_187_812_8e-12;

/// Error type for MEMS Fabry-Pérot operations.
#[derive(Debug, Clone, PartialEq)]
pub enum FpMemsError {
    /// The requested voltage exceeds the pull-in voltage, causing device collapse.
    PullIn {
        /// The voltage that was requested (V).
        requested: f64,
        /// The pull-in threshold (V).
        pull_in: f64,
    },
    /// The requested cavity length is outside the physically achievable range.
    OutOfRange {
        /// Requested cavity length (m).
        requested: f64,
        /// Minimum achievable length (m).
        min: f64,
        /// Maximum achievable length (m).
        max: f64,
    },
    /// Numerical solver failed to converge.
    ConvergenceFailure,
}

impl std::fmt::Display for FpMemsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FpMemsError::PullIn { requested, pull_in } => write!(
                f,
                "pull-in instability: requested {requested:.3} V exceeds pull-in voltage {pull_in:.3} V"
            ),
            FpMemsError::OutOfRange { requested, min, max } => write!(
                f,
                "cavity length {requested:.3e} m is outside achievable range [{min:.3e}, {max:.3e}] m"
            ),
            FpMemsError::ConvergenceFailure => write!(f, "electrostatic solver failed to converge"),
        }
    }
}

impl std::error::Error for FpMemsError {}

/// Tunable MEMS Fabry-Pérot etalon.
///
/// One mirror is fixed; the other is suspended on a mechanical spring and
/// displaced by electrostatic actuation. The cavity length `d` changes as
/// the movable mirror deflects, tuning all resonances.
///
/// # Example
/// ```
/// use oxiphoton::mems::fabry_perot_mems::MemosFabryPerot;
/// let mut fp = MemosFabryPerot::new(5e-6, 0.95, 0.95, 50.0, 100e-12);
/// let vpi = fp.pull_in_voltage();
/// assert!(vpi > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct MemosFabryPerot {
    /// Current cavity length (m).
    pub cavity_length: f64,
    /// Minimum achievable cavity length (m) (typically 2/3 of d0 at pull-in).
    pub min_length: f64,
    /// Maximum cavity length (m) = initial gap d0.
    pub max_length: f64,
    /// Reflectivity of mirror 1 (power, 0..1).
    pub r1: f64,
    /// Reflectivity of mirror 2 (power, 0..1).
    pub r2: f64,
    /// Refractive index of cavity medium.
    pub medium_index: f64,
    /// Computed finesse.
    pub finesse: f64,
    /// Mechanical spring constant (N/m).
    pub spring_constant: f64,
    /// Current actuation voltage (V).
    pub actuation_voltage: f64,
    /// Pull-in voltage threshold (V).
    pub pull_in_voltage: f64,
    /// Electrode area for electrostatic actuation (m²).
    electrode_area: f64,
    /// Initial (zero-voltage) gap (m).
    d0: f64,
}

impl MemosFabryPerot {
    /// Construct a new MEMS Fabry-Pérot cavity.
    ///
    /// # Arguments
    /// * `d0` - Initial (zero-voltage) cavity length in meters.
    /// * `r1`, `r2` - Power reflectivities of the two mirrors (0..1).
    /// * `k` - Mechanical spring constant of the movable mirror suspension (N/m).
    /// * `area` - Electrostatic electrode area (m²).
    pub fn new(d0: f64, r1: f64, r2: f64, k: f64, area: f64) -> Self {
        let r1 = r1.clamp(0.0, 1.0);
        let r2 = r2.clamp(0.0, 1.0);
        let finesse = Self::compute_finesse(r1, r2);
        // Pull-in occurs at d = 2d0/3; voltage at that point is V_pi
        let pull_in = Self::compute_pull_in_voltage(k, d0, area);
        Self {
            cavity_length: d0,
            min_length: 2.0 * d0 / 3.0,
            max_length: d0,
            r1,
            r2,
            medium_index: 1.0,
            finesse,
            spring_constant: k,
            actuation_voltage: 0.0,
            pull_in_voltage: pull_in,
            electrode_area: area,
            d0,
        }
    }

    /// Finesse of the cavity: π·(R₁·R₂)^(1/4) / (1 − √(R₁·R₂)).
    pub fn finesse(&self) -> f64 {
        self.finesse
    }

    fn compute_finesse(r1: f64, r2: f64) -> f64 {
        let r_eff = (r1 * r2).sqrt();
        if (1.0 - r_eff).abs() < f64::EPSILON {
            f64::INFINITY
        } else {
            PI * (r1 * r2).powf(0.25) / (1.0 - r_eff)
        }
    }

    /// Free spectral range (Hz): Δν = c / (2·n·L).
    pub fn fsr(&self) -> f64 {
        C / (2.0 * self.medium_index * self.cavity_length)
    }

    /// FWHM linewidth of a resonance (Hz): δν = FSR / F.
    pub fn linewidth(&self) -> f64 {
        self.fsr() / self.finesse
    }

    /// Wavelength (m) of the m-th longitudinal resonance.
    ///
    /// λ_m = 2·n·L / m
    ///
    /// Returns `None` if `order` is zero.
    pub fn resonant_wavelength(&self, order: usize) -> Option<f64> {
        if order == 0 {
            return None;
        }
        Some(2.0 * self.medium_index * self.cavity_length / order as f64)
    }

    /// Compute the pull-in voltage V_pi = √(8·k·d0³ / (27·ε₀·A)).
    fn compute_pull_in_voltage(k: f64, d0: f64, area: f64) -> f64 {
        (8.0 * k * d0.powi(3) / (27.0 * EPSILON_0 * area)).sqrt()
    }

    /// Return the current pull-in voltage (V) based on current spring/geometry.
    pub fn pull_in_voltage(&self) -> f64 {
        self.pull_in_voltage
    }

    /// Solve for the equilibrium gap under a given voltage using Newton-Raphson.
    ///
    /// Electrostatic balance: k·(d0 − d) = ε₀·A·V² / (2·d²)
    ///
    /// Rearranged: f(d) = k·(d0-d)·d² − ε₀·A·V²/2 = 0
    ///
    /// Returns the new cavity length on success, or `Err` at pull-in.
    fn solve_gap_newton(&self, voltage: f64) -> Result<f64, FpMemsError> {
        let vv = EPSILON_0 * self.electrode_area * voltage * voltage / 2.0;
        let mut d = self.d0; // initial guess at full gap
        for _ in 0..500 {
            let fd = self.spring_constant * (self.d0 - d) * d * d - vv;
            // f'(d) = k*(d0*2d - 3d²) = k*d*(2d0 - 3d)
            let fpd = self.spring_constant * d * (2.0 * self.d0 - 3.0 * d);
            if fpd.abs() < 1e-50 {
                return Err(FpMemsError::ConvergenceFailure);
            }
            let d_next = d - fd / fpd;
            // clamp to physical region (pull-in at 2d0/3)
            let d_next = d_next.max(2.0 * self.d0 / 3.0 + 1e-15);
            if (d_next - d).abs() < 1e-18 {
                return Ok(d_next);
            }
            d = d_next;
        }
        Err(FpMemsError::ConvergenceFailure)
    }

    /// Apply a voltage to electrostatically tune the cavity.
    ///
    /// Solves the electromechanical equilibrium and updates `cavity_length` and
    /// `actuation_voltage`. Returns the new cavity length (m) or an error if
    /// pull-in instability is triggered.
    pub fn tune_voltage(&mut self, voltage: f64) -> Result<f64, FpMemsError> {
        let vpi = self.pull_in_voltage;
        if voltage >= vpi {
            return Err(FpMemsError::PullIn {
                requested: voltage,
                pull_in: vpi,
            });
        }
        let new_d = self.solve_gap_newton(voltage)?;
        self.cavity_length = new_d;
        self.actuation_voltage = voltage;
        Ok(new_d)
    }

    /// Reset actuation voltage to zero and restore the initial gap.
    pub fn reset(&mut self) {
        self.cavity_length = self.d0;
        self.actuation_voltage = 0.0;
    }

    /// Transmission of the cavity at wavelength `lambda` (m) using the Airy function.
    ///
    /// T(λ) = T_max / (1 + F·sin²(δ/2))
    ///
    /// where δ = 4π·n·L/λ and F = 4R/(1-R)² (coefficient of finesse), and
    /// T_max = (1-R)² / (1-R)² for lossless mirrors.
    pub fn transmission(&self, wavelength: f64) -> f64 {
        // Use geometric mean reflectivity
        let r = (self.r1 * self.r2).sqrt();
        let phase = 4.0 * PI * self.medium_index * self.cavity_length / wavelength;
        let f_coeff = 4.0 * r / (1.0 - r).powi(2);
        let t_max = (1.0 - self.r1) * (1.0 - self.r2) / (1.0 - r).powi(2);
        t_max / (1.0 + f_coeff * (phase / 2.0).sin().powi(2))
    }

    /// Tuning range: returns (min_wavelength, max_wavelength) for a given
    /// longitudinal mode order achievable by electrostatic actuation.
    ///
    /// At pull-in the gap decreases to 2d0/3, so the wavelength shifts by the
    /// same factor.
    pub fn tuning_range(&self, order: usize) -> Option<(f64, f64)> {
        if order == 0 {
            return None;
        }
        let lambda_max = 2.0 * self.medium_index * self.d0 / order as f64;
        let lambda_min = 2.0 * self.medium_index * self.min_length / order as f64;
        Some((lambda_min, lambda_max))
    }

    /// Electrostatic force at the current actuation voltage (N).
    ///
    /// F_es = ε₀·A·V² / (2·d²)
    pub fn electrostatic_force(&self) -> f64 {
        if self.actuation_voltage == 0.0 {
            return 0.0;
        }
        EPSILON_0 * self.electrode_area * self.actuation_voltage.powi(2)
            / (2.0 * self.cavity_length.powi(2))
    }

    /// Mechanical restoring force at the current gap (N).
    pub fn restoring_force(&self) -> f64 {
        self.spring_constant * (self.d0 - self.cavity_length)
    }

    /// Voltage-to-wavelength sensitivity at the current operating point (m/V).
    ///
    /// dλ/dV = dλ/dd · dd/dV
    ///
    /// Uses a finite-difference approximation with a small voltage step.
    pub fn wavelength_sensitivity(&self, order: usize) -> Option<f64> {
        if order == 0 {
            return None;
        }
        // Numerical dλ/dV around current voltage with a tiny perturbation
        let dv = self.pull_in_voltage * 1e-5;
        let v0 = self.actuation_voltage;

        let mut fp_lo = self.clone();
        let mut fp_hi = self.clone();

        let v_lo = (v0 - dv).max(0.0);
        let v_hi = v0 + dv;

        // Only compute if the high voltage is below pull-in
        if v_hi >= self.pull_in_voltage {
            return None;
        }
        if fp_lo.tune_voltage(v_lo).is_err() || fp_hi.tune_voltage(v_hi).is_err() {
            return None;
        }
        let lambda_lo = 2.0 * self.medium_index * fp_lo.cavity_length / order as f64;
        let lambda_hi = 2.0 * self.medium_index * fp_hi.cavity_length / order as f64;
        Some((lambda_hi - lambda_lo) / (v_hi - v_lo))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_fp() -> MemosFabryPerot {
        // d0 = 5 µm, R1=R2=0.95, k=50 N/m, A=100 µm²
        MemosFabryPerot::new(5e-6, 0.95, 0.95, 50.0, 100e-12)
    }

    #[test]
    fn test_finesse() {
        let fp = make_fp();
        // F = π*(0.95)^0.5 / (1 - 0.95) = π*0.9747/0.05 ≈ 61.3
        let f = fp.finesse();
        assert!(f > 50.0 && f < 80.0, "finesse {f} out of expected range");
    }

    #[test]
    fn test_fsr() {
        let fp = make_fp();
        // FSR = c/(2nL) = 3e8/(2*1*5e-6) = 3e13 Hz = 30 THz
        let fsr = fp.fsr();
        assert_abs_diff_eq!(fsr, C / (2.0 * 5e-6), epsilon = 1e6);
    }

    #[test]
    fn test_resonant_wavelength() {
        let fp = make_fp();
        // m=10 => λ = 2*1*5e-6/10 = 1e-6 m = 1 µm
        let lambda = fp.resonant_wavelength(10).expect("order 10 should work");
        assert_abs_diff_eq!(lambda, 1.0e-6, epsilon = 1e-12);
    }

    #[test]
    fn test_pull_in_voltage_positive() {
        let fp = make_fp();
        let vpi = fp.pull_in_voltage();
        assert!(vpi > 0.0, "pull-in voltage must be positive");
    }

    #[test]
    fn test_tune_below_pull_in() {
        let mut fp = make_fp();
        let vpi = fp.pull_in_voltage();
        let v_safe = vpi * 0.5;
        let result = fp.tune_voltage(v_safe);
        assert!(result.is_ok(), "voltage below pull-in should succeed");
        let new_d = result.expect("already checked ok");
        assert!(
            new_d < fp.d0,
            "cavity should compress under positive voltage"
        );
        assert!(
            new_d > fp.min_length - 1e-12,
            "gap should not collapse below pull-in point"
        );
    }

    #[test]
    fn test_tune_at_pull_in_fails() {
        let mut fp = make_fp();
        let vpi = fp.pull_in_voltage();
        let result = fp.tune_voltage(vpi);
        assert!(
            matches!(result, Err(FpMemsError::PullIn { .. })),
            "voltage at pull-in should return PullIn error"
        );
    }

    #[test]
    fn test_transmission_at_resonance() {
        let fp = make_fp();
        // Find a resonant wavelength and verify high transmission
        let lambda = fp.resonant_wavelength(10).expect("valid order");
        let t = fp.transmission(lambda);
        // At resonance with equal mirrors, T_max = (1-R)^2/(1-R)^2 = 1
        assert!(t > 0.9, "transmission at resonance should be high, got {t}");
    }

    #[test]
    fn test_transmission_off_resonance() {
        let fp = make_fp();
        // Half-FSR away from resonance -> near minimum transmission
        let lambda = fp.resonant_wavelength(10).expect("valid order");
        let lambda_off = lambda * (1.0 + 0.5 / fp.finesse());
        let t_off = fp.transmission(lambda_off);
        let t_on = fp.transmission(lambda);
        assert!(t_off < t_on, "off-resonance transmission should be lower");
    }

    #[test]
    fn test_tuning_range_consistent() {
        let fp = make_fp();
        let (lmin, lmax) = fp.tuning_range(10).expect("valid order");
        assert!(lmin < lmax, "min wavelength must be less than max");
        // Max = resonant at rest, min = 2/3 of that
        assert_abs_diff_eq!(lmax / lmin, fp.d0 / fp.min_length, epsilon = 1e-10);
    }

    #[test]
    fn test_reset_restores_gap() {
        let mut fp = make_fp();
        let vpi = fp.pull_in_voltage();
        fp.tune_voltage(vpi * 0.5).expect("tune should succeed");
        fp.reset();
        assert_abs_diff_eq!(fp.cavity_length, fp.d0, epsilon = 1e-18);
        assert_abs_diff_eq!(fp.actuation_voltage, 0.0, epsilon = 1e-18);
    }
}
