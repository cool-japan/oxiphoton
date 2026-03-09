/// Stokes vector and polarization state analysis.
///
/// The Stokes vector S = [S0, S1, S2, S3] is a four-element real vector that
/// completely characterizes the polarization state of any beam of light, including
/// partially polarized and unpolarized light.
///
/// # Definitions
/// - S0: total intensity
/// - S1: linear horizontal (H) minus linear vertical (V) component
/// - S2: linear +45° minus linear −45° component
/// - S3: right circular minus left circular component
///
/// Physical constraint: S0² ≥ S1² + S2² + S3²
use crate::error::OxiPhotonError;

/// Four-element Stokes vector representing a polarization state.
///
/// All states satisfy S0 ≥ 0 and S0² ≥ S1² + S2² + S3² (physical realizability).
/// Equality holds for fully polarized (pure) states.
#[derive(Debug, Clone, PartialEq)]
pub struct StokesVector {
    /// [S0, S1, S2, S3] Stokes parameters
    pub s: [f64; 4],
}

impl StokesVector {
    /// Construct a Stokes vector from the four Stokes parameters.
    ///
    /// # Errors
    /// Returns `InvalidLayer` if the parameters violate the physical constraint
    /// S0 ≥ 0 and S0² ≥ S1² + S2² + S3² (within numerical tolerance).
    pub fn new(s0: f64, s1: f64, s2: f64, s3: f64) -> Result<Self, OxiPhotonError> {
        if s0 < -1e-12 {
            return Err(OxiPhotonError::InvalidLayer(format!(
                "Stokes S0 must be non-negative, got {s0}"
            )));
        }
        let pol_sq = s1 * s1 + s2 * s2 + s3 * s3;
        if pol_sq > s0 * s0 + 1e-10 {
            return Err(OxiPhotonError::InvalidLayer(format!(
                "Stokes vector violates physical realizability: S0²={} < S1²+S2²+S3²={}",
                s0 * s0,
                pol_sq
            )));
        }
        Ok(Self {
            s: [s0, s1, s2, s3],
        })
    }

    /// Construct without validation (for internal use where validity is guaranteed).
    #[inline]
    fn new_unchecked(s0: f64, s1: f64, s2: f64, s3: f64) -> Self {
        Self {
            s: [s0, s1, s2, s3],
        }
    }

    /// Construct without validation, accessible within the `polarimetry` crate modules.
    ///
    /// Used by `jones` and `mueller` sub-modules to build Stokes vectors from
    /// values that are mathematically guaranteed to be physically valid.
    #[inline]
    pub(crate) fn new_unchecked_pub(s0: f64, s1: f64, s2: f64, s3: f64) -> Self {
        Self {
            s: [s0, s1, s2, s3],
        }
    }

    // ── Canonical fully-polarized basis states ──────────────────────────────

    /// Horizontally linearly polarized light: S = [1, 1, 0, 0].
    pub fn horizontal() -> Self {
        Self::new_unchecked(1.0, 1.0, 0.0, 0.0)
    }

    /// Vertically linearly polarized light: S = [1, −1, 0, 0].
    pub fn vertical() -> Self {
        Self::new_unchecked(1.0, -1.0, 0.0, 0.0)
    }

    /// Linearly polarized at +45°: S = [1, 0, 1, 0].
    pub fn diagonal_p45() -> Self {
        Self::new_unchecked(1.0, 0.0, 1.0, 0.0)
    }

    /// Linearly polarized at −45°: S = [1, 0, −1, 0].
    pub fn diagonal_m45() -> Self {
        Self::new_unchecked(1.0, 0.0, -1.0, 0.0)
    }

    /// Right-hand circularly polarized light (RCP): S = [1, 0, 0, 1].
    ///
    /// Convention: S3 > 0 corresponds to right-hand circular polarization
    /// (physics/optics convention; the opposite of some engineering texts).
    pub fn right_circular() -> Self {
        Self::new_unchecked(1.0, 0.0, 0.0, 1.0)
    }

    /// Left-hand circularly polarized light (LCP): S = [1, 0, 0, −1].
    pub fn left_circular() -> Self {
        Self::new_unchecked(1.0, 0.0, 0.0, -1.0)
    }

    /// Completely unpolarized light with the given total intensity.
    ///
    /// S = [intensity, 0, 0, 0].
    pub fn unpolarized(intensity: f64) -> Self {
        Self::new_unchecked(intensity.max(0.0), 0.0, 0.0, 0.0)
    }

    // ── Degree-of-polarization metrics ──────────────────────────────────────

    /// Degree of polarization (DOP): fraction of intensity that is polarized.
    ///
    /// DOP = √(S1² + S2² + S3²) / S0.
    /// Range: 0 (fully unpolarized) to 1 (fully polarized).
    pub fn dop(&self) -> f64 {
        if self.s[0].abs() < f64::EPSILON {
            return 0.0;
        }
        let pol = (self.s[1] * self.s[1] + self.s[2] * self.s[2] + self.s[3] * self.s[3]).sqrt();
        (pol / self.s[0]).min(1.0)
    }

    /// Degree of linear polarization (DOLP): fraction in any linear state.
    ///
    /// DOLP = √(S1² + S2²) / S0.
    pub fn dolp(&self) -> f64 {
        if self.s[0].abs() < f64::EPSILON {
            return 0.0;
        }
        let lin = (self.s[1] * self.s[1] + self.s[2] * self.s[2]).sqrt();
        (lin / self.s[0]).min(1.0)
    }

    /// Degree of circular polarization (DOCP): fraction in circular states.
    ///
    /// DOCP = |S3| / S0.
    pub fn docp(&self) -> f64 {
        if self.s[0].abs() < f64::EPSILON {
            return 0.0;
        }
        (self.s[3].abs() / self.s[0]).min(1.0)
    }

    // ── Polarization ellipse parameters ────────────────────────────────────

    /// Ellipticity angle χ (chi) in radians.
    ///
    /// χ = ½ · atan2(S3, √(S1² + S2²)).
    /// Range: [−π/4, π/4]. χ = 0 ⟹ linear; χ = ±π/4 ⟹ circular.
    pub fn ellipticity_angle_rad(&self) -> f64 {
        let lin = (self.s[1] * self.s[1] + self.s[2] * self.s[2]).sqrt();
        0.5 * self.s[3].atan2(lin)
    }

    /// Azimuth (orientation) angle ψ (psi) of the polarization ellipse, in radians.
    ///
    /// ψ = ½ · atan2(S2, S1).
    /// Range: [−π/2, π/2].
    pub fn azimuth_rad(&self) -> f64 {
        0.5 * self.s[2].atan2(self.s[1])
    }

    /// Ellipticity ratio b/a (minor-to-major semi-axis).
    ///
    /// Range: 0 (linear) to 1 (circular).
    pub fn ellipticity(&self) -> f64 {
        let chi = self.ellipticity_angle_rad();
        chi.abs().tan().min(1.0)
    }

    /// Axial ratio a/b (major-to-minor semi-axis).
    ///
    /// Returns `f64::INFINITY` for linear polarization.
    pub fn axial_ratio(&self) -> f64 {
        let e = self.ellipticity();
        if e.abs() < f64::EPSILON {
            f64::INFINITY
        } else {
            1.0 / e
        }
    }

    // ── Derived quantities ──────────────────────────────────────────────────

    /// Total intensity (= S0).
    #[inline]
    pub fn intensity(&self) -> f64 {
        self.s[0]
    }

    /// Normalize the Stokes vector to unit intensity (S0 = 1).
    ///
    /// Returns the vector unchanged if S0 ≈ 0.
    pub fn normalize(&self) -> Self {
        if self.s[0].abs() < f64::EPSILON {
            return self.clone();
        }
        let inv = 1.0 / self.s[0];
        Self::new_unchecked(1.0, self.s[1] * inv, self.s[2] * inv, self.s[3] * inv)
    }

    /// Add two incoherent beams (intensities simply add component-wise).
    pub fn add(&self, other: &StokesVector) -> Self {
        Self::new_unchecked(
            self.s[0] + other.s[0],
            self.s[1] + other.s[1],
            self.s[2] + other.s[2],
            self.s[3] + other.s[3],
        )
    }

    /// Extract the fully polarized component of this beam.
    ///
    /// Scaled by DOP so that `polarized + unpolarized = self`.
    pub fn polarized_component(&self) -> Self {
        let dop = self.dop();
        let s0_pol = self.s[0] * dop;
        // The polarized fraction carries the same S1–S3 proportions.
        Self::new_unchecked(s0_pol, self.s[1], self.s[2], self.s[3])
    }

    /// Extract the unpolarized component of this beam.
    pub fn unpolarized_component(&self) -> Self {
        let dop = self.dop();
        let s0_unpol = self.s[0] * (1.0 - dop);
        Self::new_unchecked(s0_unpol, 0.0, 0.0, 0.0)
    }

    /// Poincaré sphere coordinates (2χ, 2ψ) in radians.
    ///
    /// Returns `(latitude, longitude)` = `(2χ, 2ψ)`.
    /// - North pole: right circular
    /// - South pole: left circular
    /// - Equator: linear states
    pub fn poincare_coords(&self) -> (f64, f64) {
        let chi2 = 2.0 * self.ellipticity_angle_rad();
        let psi2 = 2.0 * self.azimuth_rad();
        (chi2, psi2)
    }
}

/// Factory for constructing common polarization states.
pub struct PolarizationState;

impl PolarizationState {
    /// Linearly polarized at angle `angle_rad` from the horizontal axis.
    ///
    /// S = [1, cos(2θ), sin(2θ), 0].
    pub fn linear(angle_rad: f64) -> StokesVector {
        let two_theta = 2.0 * angle_rad;
        StokesVector::new_unchecked(1.0, two_theta.cos(), two_theta.sin(), 0.0)
    }

    /// Elliptically polarized state with given azimuth `ψ` and ellipticity angle `χ`.
    ///
    /// # Parameters
    /// - `azimuth_rad`: orientation of the ellipse major axis (ψ), in radians
    /// - `ellipticity_rad`: ellipticity angle (χ), in radians; χ ∈ [−π/4, π/4]
    ///
    /// S = [1, cos(2χ)cos(2ψ), cos(2χ)sin(2ψ), sin(2χ)].
    pub fn elliptical(azimuth_rad: f64, ellipticity_rad: f64) -> StokesVector {
        let two_chi = 2.0 * ellipticity_rad;
        let two_psi = 2.0 * azimuth_rad;
        let cos2chi = two_chi.cos();
        let sin2chi = two_chi.sin();
        StokesVector::new_unchecked(
            1.0,
            cos2chi * two_psi.cos(),
            cos2chi * two_psi.sin(),
            sin2chi,
        )
    }

    /// Partially polarized state: `dop` fraction fully polarized, rest unpolarized.
    ///
    /// The fully polarized portion retains the polarization ellipse parameters of `polarized`.
    /// The total intensity is taken from `polarized.intensity()`.
    ///
    /// # Parameters
    /// - `polarized`: the fully polarized reference state (need not be normalized)
    /// - `dop`: degree of polarization, clamped to [0, 1]
    pub fn partial(polarized: StokesVector, dop: f64) -> StokesVector {
        let dop = dop.clamp(0.0, 1.0);
        let i = polarized.s[0];
        let norm = polarized.normalize();
        // polarized fraction
        let i_pol = i * dop;
        // unpolarized fraction
        let i_unpol = i * (1.0 - dop);
        StokesVector::new_unchecked(
            i_pol * norm.s[0] + i_unpol,
            i_pol * norm.s[1],
            i_pol * norm.s[2],
            i_pol * norm.s[3],
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-10;

    #[test]
    fn test_horizontal_stokes() {
        let s = StokesVector::horizontal();
        assert!((s.s[0] - 1.0).abs() < EPS);
        assert!((s.s[1] - 1.0).abs() < EPS);
        assert!(s.s[2].abs() < EPS);
        assert!(s.s[3].abs() < EPS);
        assert!((s.dop() - 1.0).abs() < EPS);
    }

    #[test]
    fn test_right_circular_stokes() {
        let s = StokesVector::right_circular();
        assert!((s.s[0] - 1.0).abs() < EPS);
        assert!(s.s[1].abs() < EPS);
        assert!(s.s[2].abs() < EPS);
        assert!((s.s[3] - 1.0).abs() < EPS);
        assert!((s.docp() - 1.0).abs() < EPS);
        assert!(s.dolp() < EPS);
    }

    #[test]
    fn test_unpolarized_dop() {
        let s = StokesVector::unpolarized(3.5);
        assert!((s.s[0] - 3.5).abs() < EPS);
        assert!(s.dop() < EPS);
        assert!(s.dolp() < EPS);
        assert!(s.docp() < EPS);
    }

    #[test]
    fn test_dop_invariant_to_scaling() {
        let s = StokesVector::new(2.0, 2.0, 0.0, 0.0).expect("valid");
        assert!((s.dop() - 1.0).abs() < EPS);
        // scale by 2
        let s2 = StokesVector::new(4.0, 4.0, 0.0, 0.0).expect("valid");
        assert!((s2.dop() - s.dop()).abs() < EPS);
    }

    #[test]
    fn test_poincare_coords_horizontal() {
        let s = StokesVector::horizontal();
        let (lat, lon) = s.poincare_coords();
        // 2χ = 0 (equator), 2ψ = 0 (horizontal)
        assert!(lat.abs() < EPS);
        assert!(lon.abs() < EPS);
    }

    #[test]
    fn test_add_two_unpolarized() {
        let a = StokesVector::unpolarized(1.0);
        let b = StokesVector::unpolarized(2.0);
        let sum = a.add(&b);
        assert!((sum.intensity() - 3.0).abs() < EPS);
        assert!(sum.dop() < EPS);
    }

    #[test]
    fn test_ellipticity_circular() {
        let s = StokesVector::right_circular();
        let e = s.ellipticity();
        // circular: b/a = 1
        assert!((e - 1.0).abs() < 1e-9, "ellipticity={e}");
    }

    #[test]
    fn test_partial_polarization() {
        let pol = StokesVector::horizontal();
        let partial = PolarizationState::partial(pol, 0.5);
        let dop = partial.dop();
        assert!((dop - 0.5).abs() < 1e-10, "DOP={dop}");
    }

    #[test]
    fn test_linear_polarization_state() {
        // 0° = horizontal
        let h = PolarizationState::linear(0.0);
        assert!((h.s[1] - 1.0).abs() < EPS);
        assert!(h.s[2].abs() < EPS);

        // 90° = vertical
        let v = PolarizationState::linear(PI / 2.0);
        assert!((v.s[1] + 1.0).abs() < 1e-9);

        // 45°
        let d = PolarizationState::linear(PI / 4.0);
        assert!(d.s[1].abs() < 1e-10);
        assert!((d.s[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_elliptical_state_unit_intensity() {
        let s = PolarizationState::elliptical(PI / 6.0, PI / 8.0);
        assert!((s.intensity() - 1.0).abs() < EPS);
        assert!((s.dop() - 1.0).abs() < EPS);
    }

    #[test]
    fn test_stokes_new_rejects_invalid() {
        // S1² > S0²
        let res = StokesVector::new(0.5, 1.0, 0.0, 0.0);
        assert!(res.is_err());

        // Negative S0
        let res2 = StokesVector::new(-1.0, 0.0, 0.0, 0.0);
        assert!(res2.is_err());
    }
}
