/// Jones vector and Jones matrix calculus for fully polarized coherent light.
///
/// Jones calculus uses 2-component complex electric field vectors and 2×2 complex
/// matrices to describe the transformation of fully polarized light through optical
/// elements.  It cannot represent partially polarized or unpolarized light; use the
/// Mueller–Stokes formalism for those cases.
///
/// # Conventions
/// - The Jones vector is \[Ex, Ey\] in a right-handed xyz coordinate system with
///   propagation along +z.
/// - Phase factors use the exp(iωt − ikz) convention (physics convention).
/// - Right-hand circular polarization (RCP): electric field rotates clockwise
///   when viewed facing the source, i.e., \[1, −i\]/√2.
///
/// # References
/// - Hecht, E. "Optics", 5th ed., §8.1 (Jones calculus).
/// - Chipman, R. A. "Polarimetry," Handbook of Optics Vol. 2 (1995).
use num_complex::Complex64;

use crate::polarimetry::stokes::StokesVector;

// ── Jones Vector ─────────────────────────────────────────────────────────────

/// Two-component complex Jones vector \[Ex, Ey\] for fully polarized light.
///
/// The electric field components Ex, Ey are complex amplitudes (phasor representation).
/// Physical intensity ∝ |Ex|² + |Ey|².
#[derive(Debug, Clone)]
pub struct JonesVector {
    /// x-component of the electric field phasor
    pub ex: Complex64,
    /// y-component of the electric field phasor
    pub ey: Complex64,
}

impl JonesVector {
    /// Construct a Jones vector from complex components.
    pub fn new(ex: Complex64, ey: Complex64) -> Self {
        Self { ex, ey }
    }

    // ── Canonical polarization states ─────────────────────────────────────

    /// Horizontally polarized: \[1, 0\].
    pub fn horizontal() -> Self {
        Self::new(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0))
    }

    /// Vertically polarized: \[0, 1\].
    pub fn vertical() -> Self {
        Self::new(Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0))
    }

    /// Linear +45°: \[1, 1\]/√2.
    pub fn diagonal_p45() -> Self {
        let a = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        Self::new(a, a)
    }

    /// Linear −45°: \[1, −1\]/√2.
    pub fn diagonal_m45() -> Self {
        let a = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        Self::new(a, -a)
    }

    /// Right-hand circular polarization (RCP): \[1, i\]/√2.
    ///
    /// Uses the physics/optics convention (exp(−iωt) time dependence): the
    /// electric field of RCP light rotates counter-clockwise when viewed facing
    /// the source (i.e., clockwise to the observer at the detector).
    ///
    /// This choice ensures that `to_stokes()` maps RCP → S3 = +1, consistent
    /// with the standard Stokes/Mueller formalism.
    pub fn right_circular() -> Self {
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        Self::new(
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(0.0, inv_sqrt2),
        )
    }

    /// Left-hand circular polarization (LCP): \[1, −i\]/√2.
    ///
    /// Uses the physics/optics convention; `to_stokes()` maps LCP → S3 = −1.
    pub fn left_circular() -> Self {
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        Self::new(
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(0.0, -inv_sqrt2),
        )
    }

    /// Linearly polarized at angle `angle_rad` from the horizontal axis.
    ///
    /// \[cos θ, sin θ\].
    pub fn linear(angle_rad: f64) -> Self {
        Self::new(
            Complex64::new(angle_rad.cos(), 0.0),
            Complex64::new(angle_rad.sin(), 0.0),
        )
    }

    // ── Derived quantities ───────────────────────────────────────────────

    /// Total intensity: |Ex|² + |Ey|².
    pub fn intensity(&self) -> f64 {
        self.ex.norm_sqr() + self.ey.norm_sqr()
    }

    /// Normalize to unit intensity.
    ///
    /// Returns the vector unchanged if intensity ≈ 0.
    pub fn normalize(&self) -> Self {
        let i = self.intensity();
        if i < f64::EPSILON {
            return self.clone();
        }
        let scale = 1.0 / i.sqrt();
        Self::new(self.ex * scale, self.ey * scale)
    }

    /// Phase difference δ = arg(Ey) − arg(Ex), in radians ∈ (−π, π].
    pub fn phase_difference(&self) -> f64 {
        let phi_y = self.ey.arg();
        let phi_x = self.ex.arg();
        phi_y - phi_x
    }

    /// Amplitude ratio |Ey| / |Ex|.
    ///
    /// Returns `f64::INFINITY` for horizontal polarization (|Ex| = 0 handled as ∞).
    pub fn amplitude_ratio(&self) -> f64 {
        let ax = self.ex.norm();
        if ax < f64::EPSILON {
            return f64::INFINITY;
        }
        self.ey.norm() / ax
    }

    /// Convert Jones vector to an equivalent Stokes vector.
    ///
    /// Valid only for fully polarized light (DOP = 1).
    ///
    /// Using the standard definitions (Chipman, Handbook of Optics, 2nd ed.):
    ///
    ///   S0 = |Ex|² + |Ey|²
    ///   S1 = |Ex|² − |Ey|²
    ///   S2 = 2·Re(Ex*·Ey)  =  2·Re(Ex·Ey*)
    ///   S3 = 2·Im(Ex*·Ey)  = −2·Im(Ex·Ey*)
    ///
    /// Physics convention: right-hand circular polarization \[1, −i\]/√2 → S3 = +1.
    pub fn to_stokes(&self) -> StokesVector {
        let s0 = self.ex.norm_sqr() + self.ey.norm_sqr();
        let s1 = self.ex.norm_sqr() - self.ey.norm_sqr();
        // cross = Ex* · Ey
        let cross = self.ex.conj() * self.ey;
        let s2 = 2.0 * cross.re;
        let s3 = 2.0 * cross.im;
        // Safety: constructed from a valid Jones vector → always physical
        StokesVector::new_unchecked_pub(s0, s1, s2, s3)
    }
}

// ── Jones Matrix ─────────────────────────────────────────────────────────────

/// 2×2 complex Jones matrix representing a linear optical element.
///
/// The Jones matrix M transforms a Jones vector v according to v_out = M · v_in.
/// Cascaded elements are composed as M_total = M_N · … · M_2 · M_1.
#[derive(Debug, Clone)]
pub struct JonesMatrix {
    /// Row-major 2×2 complex matrix: m\[row\]\[col\]
    pub m: [[Complex64; 2]; 2],
}

impl JonesMatrix {
    /// Construct a Jones matrix from a row-major 2×2 array.
    pub fn new(m: [[Complex64; 2]; 2]) -> Self {
        Self { m }
    }

    /// 2×2 identity matrix.
    pub fn identity() -> Self {
        let o = Complex64::new(0.0, 0.0);
        let i = Complex64::new(1.0, 0.0);
        Self::new([[i, o], [o, i]])
    }

    /// Apply the Jones matrix to a Jones vector: v_out = M · v.
    pub fn apply(&self, v: &JonesVector) -> JonesVector {
        JonesVector::new(
            self.m[0][0] * v.ex + self.m[0][1] * v.ey,
            self.m[1][0] * v.ex + self.m[1][1] * v.ey,
        )
    }

    /// Cascade two elements: `self` acts first, `next` acts second.
    ///
    /// Returns `next.M × self.M`.
    pub fn cascade(&self, next: &JonesMatrix) -> JonesMatrix {
        mat2x2_mul(next, self)
    }

    // ── Canonical optical elements ───────────────────────────────────────

    /// Linear polarizer with transmission axis at angle `angle_rad` from horizontal.
    ///
    /// M = \[\[cos²θ, cosθ·sinθ\\], \[cosθ·sinθ, sin²θ\]]
    pub fn linear_polarizer(angle_rad: f64) -> Self {
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        let cc = Complex64::new(c * c, 0.0);
        let ss = Complex64::new(s * s, 0.0);
        let cs = Complex64::new(c * s, 0.0);
        Self::new([[cc, cs], [cs, ss]])
    }

    /// Half-wave plate (HWP) with fast axis at angle `fast_axis_rad`.
    ///
    /// Phase retardation δ = π (fast axis gains no phase; slow axis gains π).
    pub fn half_wave_plate(fast_axis_rad: f64) -> Self {
        Self::wave_plate(std::f64::consts::PI, fast_axis_rad)
    }

    /// Quarter-wave plate (QWP) with fast axis at angle `fast_axis_rad`.
    ///
    /// Phase retardation δ = π/2.
    pub fn quarter_wave_plate(fast_axis_rad: f64) -> Self {
        Self::wave_plate(std::f64::consts::FRAC_PI_2, fast_axis_rad)
    }

    /// General wave plate (retarder) with retardation `phase_rad` and
    /// fast axis at `fast_axis_rad`.
    ///
    /// The Jones matrix for a retarder rotated by θ with retardation δ is:
    ///
    /// M = R(−θ) · diag(e^{iδ/2}, e^{−iδ/2}) · R(θ)
    ///
    /// where R(θ) is the rotation matrix.
    pub fn wave_plate(phase_rad: f64, fast_axis_rad: f64) -> Self {
        let theta = fast_axis_rad;
        let delta = phase_rad;
        let half_delta = delta / 2.0;
        // eigenvalues: fast axis → e^{+iδ/2}, slow axis → e^{−iδ/2}
        let lam_fast = Complex64::new(0.0, half_delta).exp();
        let lam_slow = Complex64::new(0.0, -half_delta).exp();
        let c = theta.cos();
        let s = theta.sin();
        let c2 = c * c;
        let s2 = s * s;
        let cs = c * s;
        // M = lam_fast * [[c²,cs],[cs,s²]] + lam_slow * [[s²,-cs],[-cs,c²]]
        let m00 = lam_fast * c2 + lam_slow * s2;
        let m01 = (lam_fast - lam_slow) * cs;
        let m10 = m01;
        let m11 = lam_fast * s2 + lam_slow * c2;
        Self::new([[m00, m01], [m10, m11]])
    }

    /// Optical rotator by angle `angle_rad` (e.g., Faraday rotator, optically active medium).
    ///
    /// M = \[\[cos θ, −sin θ\\], \[sin θ, cos θ\]]
    pub fn rotator(angle_rad: f64) -> Self {
        let c = Complex64::new(angle_rad.cos(), 0.0);
        let s = Complex64::new(angle_rad.sin(), 0.0);
        Self::new([[c, -s], [s, c]])
    }

    /// Lossless beam splitter (amplitude transmission `t`, reflection `r = √(1−t²)`).
    ///
    /// Modelled as a scalar amplitude multiplier: M = t · I₂.
    /// Full beam-splitter modelling requires a 4-port description; this element
    /// captures only the transmitted field.
    pub fn beam_splitter(transmission: f64) -> Self {
        let t = Complex64::new(transmission.clamp(0.0, 1.0), 0.0);
        let o = Complex64::new(0.0, 0.0);
        Self::new([[t, o], [o, t]])
    }

    /// Phase delay: independent phase shifts δx and δy on the x and y components.
    ///
    /// M = \[\[e^{iδx}, 0\\], \[0, e^{iδy}\]]
    pub fn phase_delay(delta_x: f64, delta_y: f64) -> Self {
        let px = Complex64::new(0.0, delta_x).exp();
        let py = Complex64::new(0.0, delta_y).exp();
        let o = Complex64::new(0.0, 0.0);
        Self::new([[px, o], [o, py]])
    }

    /// Amplitude (intensity) attenuator by a real factor `amplitude_factor` ∈ \[0, 1\].
    ///
    /// M = amplitude_factor · I₂.
    pub fn attenuator(amplitude_factor: f64) -> Self {
        let a = Complex64::new(amplitude_factor.clamp(0.0, 1.0), 0.0);
        let o = Complex64::new(0.0, 0.0);
        Self::new([[a, o], [o, a]])
    }

    // ── Matrix analysis ──────────────────────────────────────────────────

    /// Eigenvalues of the 2×2 Jones matrix via the quadratic characteristic equation.
    ///
    /// Returns (λ₁, λ₂) ordered so that |λ₁| ≥ |λ₂|.
    pub fn eigenvalues(&self) -> (Complex64, Complex64) {
        let a = self.m[0][0];
        let b = self.m[0][1];
        let c = self.m[1][0];
        let d = self.m[1][1];
        let trace = a + d;
        let det = a * d - b * c;
        // λ = (tr ± √(tr² − 4·det)) / 2
        let discriminant = trace * trace - Complex64::new(4.0, 0.0) * det;
        let sqrt_disc = csqrt(discriminant);
        let lam1 = (trace + sqrt_disc) / 2.0;
        let lam2 = (trace - sqrt_disc) / 2.0;
        if lam1.norm() >= lam2.norm() {
            (lam1, lam2)
        } else {
            (lam2, lam1)
        }
    }

    /// Power transmission: ratio of output to input intensity for given input.
    pub fn transmission(&self, input: &JonesVector) -> f64 {
        let i_in = input.intensity();
        if i_in < f64::EPSILON {
            return 0.0;
        }
        let output = self.apply(input);
        output.intensity() / i_in
    }

    /// Convert Jones matrix to the equivalent Mueller matrix.
    ///
    /// Uses the relation M_Mueller = A (J ⊗ J*) A⁻¹ where A is the coherency
    /// transformation matrix.
    ///
    /// Ref: Chipman (1995), Eq. (22.44).
    pub fn to_mueller(&self) -> crate::polarimetry::mueller::MuellerMatrix {
        jones_to_mueller(self)
    }
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Matrix multiplication of two 2×2 Jones matrices: result = a × b.
fn mat2x2_mul(a: &JonesMatrix, b: &JonesMatrix) -> JonesMatrix {
    let mut r = [[Complex64::new(0.0, 0.0); 2]; 2];
    for (i, r_row) in r.iter_mut().enumerate() {
        for (j, r_cell) in r_row.iter_mut().enumerate() {
            for k in 0..2 {
                *r_cell += a.m[i][k] * b.m[k][j];
            }
        }
    }
    JonesMatrix::new(r)
}

/// Complex square root via polar decomposition.
fn csqrt(z: Complex64) -> Complex64 {
    let r = z.norm().sqrt();
    let theta = z.arg() / 2.0;
    Complex64::new(r * theta.cos(), r * theta.sin())
}

/// Convert a Jones matrix J to a Mueller matrix M via the Pauli trace formula.
///
/// The Mueller matrix element M_{row,col} is given by:
///
///   M_{row,col} = (1/2) · Re\[ Tr(σ_{row} · J · σ_{col} · J†) \]
///
/// where σ_0 = I₂, σ_1, σ_2, σ_3 are the Pauli matrices and J† is the
/// conjugate transpose of J.
///
/// This formula is exact, numerically stable, and free of normalization ambiguity.
/// Reference: Chipman, "Polarimetry," Handbook of Optics Vol. 2, §22.5 (1995).
pub(crate) fn jones_to_mueller(j: &JonesMatrix) -> crate::polarimetry::mueller::MuellerMatrix {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let im = Complex64::new(0.0, 1.0);

    // Pauli matrices σ_0..σ_3 (2×2 complex, row-major)
    let sigma: [[[Complex64; 2]; 2]; 4] = [
        // σ_0 = I₂
        [[one, zero], [zero, one]],
        // σ_1 = [[0,1],[1,0]]
        [[zero, one], [one, zero]],
        // σ_2 = [[0,-i],[i,0]]
        [[zero, -im], [im, zero]],
        // σ_3 = [[1,0],[0,-1]]
        [[one, zero], [zero, -one]],
    ];

    // J† (conjugate transpose of J)
    let j_dag: [[Complex64; 2]; 2] = [
        [j.m[0][0].conj(), j.m[1][0].conj()],
        [j.m[0][1].conj(), j.m[1][1].conj()],
    ];

    let mut m = [[0.0f64; 4]; 4];

    for row in 0..4 {
        for col in 0..4 {
            // Compute Tr(σ_{row} · J · σ_{col} · J†) = Σ_i [σ_{row} · J · σ_{col} · J†]_{ii}
            //
            // Step 1: A_{ik} = (σ_{row} · J)_{ik} = Σ_l σ_{row}[i][l] · J[l][k]
            let mut a_mat = [[zero; 2]; 2];
            for (i, a_row) in a_mat.iter_mut().enumerate() {
                for (k, a_cell) in a_row.iter_mut().enumerate() {
                    for (l, &s_il) in sigma[row][i].iter().enumerate() {
                        *a_cell += s_il * j.m[l][k];
                    }
                }
            }
            // Step 2: B_{ik} = (σ_{col} · J†)_{ik} = Σ_l σ_{col}[i][l] · J†[l][k]
            let mut b_mat = [[zero; 2]; 2];
            for i in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        b_mat[i][k] += sigma[col][i][l] * j_dag[l][k];
                    }
                }
            }
            // Step 3: Tr(A · B) = Σ_i Σ_l A[i][l] · B[l][i]
            let mut tr = zero;
            for i in 0..2 {
                for l in 0..2 {
                    tr += a_mat[i][l] * b_mat[l][i];
                }
            }
            m[row][col] = 0.5 * tr.re;
        }
    }

    crate::polarimetry::mueller::MuellerMatrix::new(m)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    const EPS: f64 = 1e-10;
    const LOOSE: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_jones_horizontal_intensity() {
        let h = JonesVector::horizontal();
        assert!(approx_eq(h.intensity(), 1.0, EPS));
    }

    #[test]
    fn test_jones_to_stokes_circular() {
        // RCP: [1,-i]/√2 → S = [1, 0, 0, 1]
        let rcp = JonesVector::right_circular();
        let s = rcp.to_stokes();
        assert!(approx_eq(s.s[0], 1.0, LOOSE), "S0={}", s.s[0]);
        assert!(s.s[1].abs() < LOOSE, "S1={}", s.s[1]);
        assert!(s.s[2].abs() < LOOSE, "S2={}", s.s[2]);
        assert!(approx_eq(s.s[3], 1.0, LOOSE), "S3={}", s.s[3]);

        // LCP: [1,i]/√2 → S = [1, 0, 0, -1]
        let lcp = JonesVector::left_circular();
        let sl = lcp.to_stokes();
        assert!(approx_eq(sl.s[3], -1.0, LOOSE), "S3={}", sl.s[3]);
    }

    #[test]
    fn test_hwp_rotates_linear() {
        // HWP with fast axis at 22.5° transforms H → +45°
        let hwp = JonesMatrix::half_wave_plate(FRAC_PI_4 / 2.0);
        let h = JonesVector::horizontal();
        let out = hwp.apply(&h);
        let s = out.to_stokes();
        // +45° state: S1 ≈ 0, S2 ≈ 1·S0
        assert!(s.s[1].abs() < 1e-9, "S1={}", s.s[1]);
        assert!(
            approx_eq(s.s[2] / s.s[0], 1.0, 1e-9),
            "S2/S0={}",
            s.s[2] / s.s[0]
        );
    }

    #[test]
    fn test_qwp_h_gives_circular() {
        // QWP fast axis at 45° on H → right circular
        let qwp = JonesMatrix::quarter_wave_plate(FRAC_PI_4);
        let h = JonesVector::horizontal();
        let out = qwp.apply(&h);
        let s = out.to_stokes();
        // S3 should be non-zero (circular), S1 and S2 should be small
        assert!(s.s[1].abs() < 1e-9, "S1={}", s.s[1]);
        assert!(s.s[2].abs() < 1e-9, "S2={}", s.s[2]);
        assert!(s.s[3].abs() > 0.9 * s.s[0], "S3={} S0={}", s.s[3], s.s[0]);
    }

    #[test]
    fn test_polarizer_blocks_perp() {
        // Vertical polarizer on horizontal input → zero intensity
        let pol_v = JonesMatrix::linear_polarizer(FRAC_PI_2);
        let h = JonesVector::horizontal();
        let out = pol_v.apply(&h);
        assert!(out.intensity() < 1e-20, "intensity={}", out.intensity());
    }

    #[test]
    fn test_jones_cascade_two_hwp() {
        // Two identical HWPs with the same axis compose to the identity (up to global phase).
        let hwp = JonesMatrix::half_wave_plate(0.0);
        let hwp2 = hwp.cascade(&hwp);
        // Apply to arbitrary input H and compare to identity
        let h = JonesVector::horizontal();
        let out = hwp2.apply(&h);
        // |Ex|² + |Ey|² should equal input intensity; direction should be restored
        assert!(approx_eq(out.intensity(), h.intensity(), 1e-10));
        // Ex ≈ ±1, Ey ≈ 0
        assert!(out.ey.norm() < 1e-10, "Ey={}", out.ey.norm());
    }

    #[test]
    fn test_jones_to_mueller_identity() {
        let j = JonesMatrix::identity();
        let m = j.to_mueller();
        let identity_row = [1.0, 0.0, 0.0, 0.0];
        for r in 0..4 {
            for c in 0..4 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(m.m[r][c], expected, 1e-10),
                    "M[{r}][{c}]={} expected {expected}",
                    m.m[r][c]
                );
            }
        }
        let _ = identity_row; // suppress unused warning
    }

    #[test]
    fn test_wave_plate_general() {
        // A wave plate with δ = 2π should be the identity (up to global phase e^{iπ} = -1,
        // but intensity is unaffected).
        let wp = JonesMatrix::wave_plate(2.0 * PI, 0.0);
        let h = JonesVector::horizontal();
        let out = wp.apply(&h);
        assert!(approx_eq(out.intensity(), h.intensity(), 1e-10));
    }

    #[test]
    fn test_eigenvalues_identity() {
        let j = JonesMatrix::identity();
        let (l1, l2) = j.eigenvalues();
        assert!(approx_eq(l1.re, 1.0, 1e-10));
        assert!(approx_eq(l2.re, 1.0, 1e-10));
        assert!(l1.im.abs() < 1e-10);
        assert!(l2.im.abs() < 1e-10);
    }
}
