/// Mutual Coherence Function and Cross-Spectral Density
///
/// Implements the Wolf coherence theory framework:
/// - Mutual coherence function W(r1, r2, ω) = <E*(r1,ω) E(r2,ω)>
/// - Cross-spectral density with Hermitian + positive semi-definite constraints
/// - Polarization coherence matrix (2×2 Jones coherence matrix)
/// - Van Cittert-Zernike theorem for incoherent sources
/// - Schell-model beams with Gaussian coherence kernels
use num_complex::Complex64;
use std::f64::consts::PI;

/// Error type for coherence calculations.
#[derive(Debug, Clone, PartialEq)]
pub enum CoherenceError {
    /// The provided arrays have mismatched lengths.
    LengthMismatch { expected: usize, got: usize },
    /// A physical parameter is out of its valid range.
    InvalidParameter(String),
    /// Division by zero or near-zero denominator encountered.
    DivisionByZero,
}

impl std::fmt::Display for CoherenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch { expected, got } => {
                write!(f, "Length mismatch: expected {expected}, got {got}")
            }
            Self::InvalidParameter(msg) => write!(f, "Invalid parameter: {msg}"),
            Self::DivisionByZero => write!(f, "Division by zero encountered"),
        }
    }
}

impl std::error::Error for CoherenceError {}

// ─────────────────────────────────────────────────────────────────────────────
// MutualCoherenceFunction
// ─────────────────────────────────────────────────────────────────────────────

/// Mutual coherence function W(r1, r2, ω) = <E*(r1,ω) E(r2,ω)>.
///
/// Encodes the statistical correlation of the electric field between two
/// spatial positions r1 and r2 at angular frequency ω.
#[derive(Debug, Clone)]
pub struct MutualCoherenceFunction {
    /// Angular frequency \[rad/s\] at which this function is evaluated.
    pub omega: f64,
    /// Spatial positions r1 \[m\].
    pub r1: [f64; 3],
    /// Spatial positions r2 \[m\].
    pub r2: [f64; 3],
    /// Complex value W(r1, r2, ω).
    pub value: Complex64,
}

impl MutualCoherenceFunction {
    /// Construct a mutual coherence function from explicit field samples.
    ///
    /// The value is W = <E*(r1) · E(r2)> taken as the direct product of
    /// complex-conjugate amplitudes when only two samples are available.
    pub fn new(
        omega: f64,
        r1: [f64; 3],
        r2: [f64; 3],
        e_at_r1: Complex64,
        e_at_r2: Complex64,
    ) -> Self {
        Self {
            omega,
            r1,
            r2,
            value: e_at_r1.conj() * e_at_r2,
        }
    }

    /// Normalised degree of coherence |μ(r1, r2)| ∈ \[0, 1\].
    ///
    /// Returns `None` when either self-coherence (spectral density) is zero.
    pub fn degree_of_coherence(
        &self,
        w_r1r1: &MutualCoherenceFunction,
        w_r2r2: &MutualCoherenceFunction,
    ) -> Option<f64> {
        let denom = (w_r1r1.value.re * w_r2r2.value.re).sqrt();
        if denom < f64::EPSILON {
            None
        } else {
            Some((self.value.norm() / denom).min(1.0))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CrossSpectralDensity
// ─────────────────────────────────────────────────────────────────────────────

/// Cross-spectral density matrix W(r1, r2, ω).
///
/// Encodes the full spatial coherence at a single frequency.  The matrix is
/// sampled on a 1-D grid of positions; `values[i][j]` is W(x_i, x_j, ω).
///
/// # Physical constraints
/// - Hermitian: W(r1, r2) = W*(r2, r1)
/// - Positive semi-definite: all eigenvalues ≥ 0
#[derive(Debug, Clone)]
pub struct CrossSpectralDensity {
    /// Angular frequency \[rad/s\].
    pub omega: f64,
    /// Sampling positions \[m\].
    pub positions: Vec<f64>,
    /// N×N matrix of cross-spectral density values (row-major).
    pub values: Vec<Vec<Complex64>>,
}

impl CrossSpectralDensity {
    /// Build a cross-spectral density from a Schell-model kernel.
    ///
    /// W(x1, x2, ω) = √(S(x1)) · μ(x1-x2) · √(S(x2))
    /// where S(x) is the spectral density and μ the spectence kernel.
    pub fn from_schell_model(
        positions: Vec<f64>,
        spectral_density: &dyn Fn(f64) -> f64,
        coherence_kernel: &dyn Fn(f64) -> f64,
    ) -> Self {
        let n = positions.len();
        let mut values = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for i in 0..n {
            let si = spectral_density(positions[i]).max(0.0).sqrt();
            for j in 0..n {
                let sj = spectral_density(positions[j]).max(0.0).sqrt();
                let mu = coherence_kernel(positions[i] - positions[j]);
                values[i][j] = Complex64::new(si * mu * sj, 0.0);
            }
        }
        // Symmetrise to enforce Hermitian property numerically.
        for (i, _) in (0..n).zip(std::iter::repeat(())) {
            for (j, _) in (i + 1..n).zip(std::iter::repeat(())) {
                let avg = (values[i][j] + values[j][i].conj()) * 0.5;
                values[i][j] = avg;
                values[j][i] = avg.conj();
            }
        }
        Self {
            omega: 0.0,
            positions,
            values,
        }
    }

    /// Return the spectral density (diagonal) at position index `i`.
    pub fn spectral_density_at(&self, i: usize) -> f64 {
        self.values
            .get(i)
            .and_then(|row| row.get(i))
            .map(|v| v.re)
            .unwrap_or(0.0)
    }

    /// Verify the Hermitian property to within a tolerance.
    pub fn is_hermitian(&self, tol: f64) -> bool {
        let n = self.values.len();
        for i in 0..n {
            for j in 0..n {
                let diff = (self.values[i][j] - self.values[j][i].conj()).norm();
                if diff > tol {
                    return false;
                }
            }
        }
        true
    }
}

/// Compute the degree of coherence |μ(r1, r2)| from a `CrossSpectralDensity`.
///
/// Uses the formula μ(r1,r2) = W(r1,r2) / √(W(r1,r1) · W(r2,r2)).
///
/// Returns 0 when the spectral density at either point is non-positive.
pub fn degree_of_coherence(w: &CrossSpectralDensity, r1: [f64; 3], r2: [f64; 3]) -> f64 {
    // For a 1-D CSD grid we map the 3-D query onto the nearest grid point.
    let find_nearest = |x: f64| -> usize {
        w.positions
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((*a - x).abs())
                    .partial_cmp(&((*b - x).abs()))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    };

    let i = find_nearest(r1[0]);
    let j = find_nearest(r2[0]);

    let s1 = w.spectral_density_at(i);
    let s2 = w.spectral_density_at(j);

    if s1 <= 0.0 || s2 <= 0.0 {
        return 0.0;
    }

    let w12 = w
        .values
        .get(i)
        .and_then(|row| row.get(j))
        .copied()
        .unwrap_or(Complex64::new(0.0, 0.0));
    (w12.norm() / (s1 * s2).sqrt()).min(1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// CoherenceMatrix (2×2 polarization)
// ─────────────────────────────────────────────────────────────────────────────

/// 2×2 polarization coherence matrix (Jones coherence matrix).
///
/// J = \[\[Jxx, Jxy\\], \[Jyx, Jyy\]]
///
/// where Jij = <Ei*(t) Ej(t)>.  Diagonal elements are the intensities
/// of the x and y polarization components; off-diagonal elements encode
/// polarization correlations.
#[derive(Debug, Clone)]
pub struct CoherenceMatrix {
    /// Jxx = <|Ex|²>
    pub jxx: Complex64,
    /// Jxy = <Ex* Ey>
    pub jxy: Complex64,
    /// Jyx = <Ey* Ex> = Jxy*
    pub jyx: Complex64,
    /// Jyy = <|Ey|²>
    pub jyy: Complex64,
}

impl CoherenceMatrix {
    /// Construct the coherence matrix from four complex entries.
    ///
    /// The off-diagonal elements are symmetrised to enforce Hermiticity.
    pub fn new(jxx: Complex64, jxy: Complex64, jyy: Complex64) -> Self {
        Self {
            jxx,
            jxy,
            jyx: jxy.conj(),
            jyy,
        }
    }

    /// Total intensity Tr(J) = Jxx + Jyy.
    pub fn total_intensity(&self) -> f64 {
        self.jxx.re + self.jyy.re
    }

    /// Degree of polarization P ∈ \[0, 1\].
    ///
    /// P = √(1 − 4 det(J) / Tr(J)²)
    pub fn degree_of_polarization(&self) -> f64 {
        let tr = self.total_intensity();
        if tr < f64::EPSILON {
            return 0.0;
        }
        let det = self.jxx * self.jyy - self.jxy * self.jyx;
        let discriminant = (1.0 - 4.0 * det.re / (tr * tr)).max(0.0);
        discriminant.sqrt()
    }

    /// Degree of coherence between the two polarization components.
    ///
    /// η = |Jxy| / √(Jxx · Jyy)
    pub fn polarization_degree_of_coherence(&self) -> f64 {
        let denom = (self.jxx.re * self.jyy.re).sqrt();
        if denom < f64::EPSILON {
            0.0
        } else {
            (self.jxy.norm() / denom).min(1.0)
        }
    }

    /// Stokes parameters \[S0, S1, S2, S3\].
    pub fn stokes(&self) -> [f64; 4] {
        let s0 = self.jxx.re + self.jyy.re;
        let s1 = self.jxx.re - self.jyy.re;
        let s2 = 2.0 * self.jxy.re;
        let s3 = -2.0 * self.jxy.im;
        [s0, s1, s2, s3]
    }

    /// Check whether the matrix is valid (positive semi-definite and Hermitian).
    pub fn is_valid(&self) -> bool {
        // Hermitian check
        let hermitian = (self.jyx - self.jxy.conj()).norm() < 1e-12;
        // PSD check: both eigenvalues ≥ 0 ⟺ Tr ≥ 0 and det ≥ 0
        let tr = self.jxx.re + self.jyy.re;
        let det = self.jxx * self.jyy - self.jxy * self.jyx;
        hermitian && tr >= 0.0 && det.re >= -1e-12
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Van Cittert-Zernike theorem
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the far-field mutual coherence from an incoherent planar source
/// using the Van Cittert-Zernike theorem.
///
/// For a quasi-monochromatic incoherent source with intensity distribution
/// I(r_s) at positions r_s on the source plane, the mutual coherence in the
/// far-field at positions p1 and p2 (in the observation plane at distance z) is:
///
/// W(p1, p2) ∝ ∫ I(r_s) exp(ik \[|p2 - r_s| - |p1 - r_s|\] / z) d²r_s
///
/// (paraxial Fresnel approximation)
///
/// # Parameters
/// - `source_intensity` — intensity weights for each source element.
/// - `source_positions` — 2-D positions \[x, y\] of the source elements \[m\].
/// - `wavelength` — free-space wavelength \[m\].
/// - `z` — propagation distance from source to observation plane \[m\].
///
/// Returns a symmetric N×N matrix of complex coherence values where N is the
/// number of source positions (used as observation probe positions as well for
/// self-consistency).
pub fn van_cittert_zernike_theorem(
    source_intensity: &[f64],
    source_positions: &[[f64; 2]],
    wavelength: f64,
    z: f64,
) -> Result<Vec<Vec<Complex64>>, CoherenceError> {
    let n_src = source_intensity.len();
    if source_positions.len() != n_src {
        return Err(CoherenceError::LengthMismatch {
            expected: n_src,
            got: source_positions.len(),
        });
    }
    if wavelength <= 0.0 {
        return Err(CoherenceError::InvalidParameter(
            "wavelength must be positive".into(),
        ));
    }
    if z <= 0.0 {
        return Err(CoherenceError::InvalidParameter(
            "propagation distance z must be positive".into(),
        ));
    }

    let k = 2.0 * PI / wavelength;

    // We use the source positions as observation positions (self-consistency).
    let n = n_src;
    let mut w = vec![vec![Complex64::new(0.0, 0.0); n]; n];

    for i in 0..n {
        let p1 = source_positions[i];
        for j in 0..n {
            let p2 = source_positions[j];
            let mut acc = Complex64::new(0.0, 0.0);
            for s in 0..n_src {
                let rs = source_positions[s];
                let i_s = source_intensity[s];
                if i_s <= 0.0 {
                    continue;
                }
                // Paraxial phase difference: k/(2z) * (|p2-rs|² - |p1-rs|²)
                let d1_sq = (p1[0] - rs[0]).powi(2) + (p1[1] - rs[1]).powi(2);
                let d2_sq = (p2[0] - rs[0]).powi(2) + (p2[1] - rs[1]).powi(2);
                let phase = k / (2.0 * z) * (d2_sq - d1_sq);
                acc += Complex64::new(0.0, phase).exp() * i_s;
            }
            w[i][j] = acc;
        }
    }
    Ok(w)
}

// ─────────────────────────────────────────────────────────────────────────────
// SchellModelBeam
// ─────────────────────────────────────────────────────────────────────────────

/// Schell-model beam: a spatially partially coherent beam whose spectral
/// degree of coherence depends only on the difference r1 − r2.
///
/// The beam has a Gaussian intensity profile and a Gaussian coherence kernel:
///   S(r)     = exp(−r²/w²)
///   μ(r1,r2) = exp(−|r1−r2|²/(2 lc²))
///
/// This is the Gaussian Schell-model (GSM) beam, widely used in partial
/// coherence theory (Mandel & Wolf, 1995).
#[derive(Debug, Clone)]
pub struct SchellModelBeam {
    /// 1/e² beam radius w \[m\].
    pub beam_radius: f64,
    /// Transverse coherence length lc \[m\].
    pub coherence_length: f64,
    /// Peak intensity \[W/m²\].
    pub peak_intensity: f64,
}

impl SchellModelBeam {
    /// Create a Gaussian Schell-model beam.
    ///
    /// # Errors
    /// Returns `CoherenceError::InvalidParameter` if either radius is non-positive.
    pub fn new(beam_radius: f64, coherence_length: f64) -> Result<Self, CoherenceError> {
        if beam_radius <= 0.0 {
            return Err(CoherenceError::InvalidParameter(
                "beam_radius must be positive".into(),
            ));
        }
        if coherence_length <= 0.0 {
            return Err(CoherenceError::InvalidParameter(
                "coherence_length must be positive".into(),
            ));
        }
        Ok(Self {
            beam_radius,
            coherence_length,
            peak_intensity: 1.0,
        })
    }

    /// Set the peak intensity \[W/m²\].
    pub fn with_peak_intensity(mut self, i0: f64) -> Self {
        self.peak_intensity = i0;
        self
    }

    /// Gaussian intensity profile S(x, y) = I₀ exp(−(x²+y²)/w²).
    pub fn intensity(&self, x: f64, y: f64) -> f64 {
        let r2 = x * x + y * y;
        self.peak_intensity * (-r2 / (self.beam_radius * self.beam_radius)).exp()
    }

    /// Degree of coherence (Gaussian kernel) μ(Δx, Δy) = exp(−|Δr|²/(2lc²)).
    pub fn degree_of_coherence(&self, dx: f64, dy: f64) -> f64 {
        let dr2 = dx * dx + dy * dy;
        let lc2 = self.coherence_length * self.coherence_length;
        (-dr2 / (2.0 * lc2)).exp()
    }

    /// Propagate the GSM beam through free space of length `z` using the
    /// Wigner-distribution (phase-space) propagation rules.
    ///
    /// Under free-space propagation the effective beam radius and coherence
    /// length evolve as:
    ///   w²(z) = w₀² + (z/k)² (1/w₀² + 2/lc²)    \[Gaussian Schell-model\]
    ///
    /// where k = 2π/λ.  Since λ is not stored in the beam itself we use the
    /// far-field divergence parameterised by the quality-factor M².
    ///
    /// The coherence length evolves such that the effective coherence number is
    /// preserved; in the paraxial scalar theory for a GSM beam:
    ///   1/lc²(z) = 1/lc₀² − propagation correction (self-consistent)
    ///
    /// For the free-space case the closed-form result (in angular-frequency
    /// formulation, Gbur & Wolf 2002) is:
    ///   q(z) = q₀ + z/k   with  q = w²lc² / (w² + lc²/2)
    ///
    /// Here we apply the beam-propagation factor in a simplified form that
    /// preserves the ratio lc/w while scaling both with the Rayleigh range of
    /// the effective source sigma_s = w·lc/√(w²+lc²).
    pub fn propagate_wigner(&self, z: f64, wavelength: f64) -> Result<Self, CoherenceError> {
        if wavelength <= 0.0 {
            return Err(CoherenceError::InvalidParameter(
                "wavelength must be positive".into(),
            ));
        }
        let k = 2.0 * PI / wavelength;
        let w0 = self.beam_radius;
        let lc = self.coherence_length;

        // Effective source size for the GSM beam (coherent-mode decomposition).
        let sigma_s_sq = 1.0 / (1.0 / (w0 * w0) + 2.0 / (lc * lc));
        let sigma_s = sigma_s_sq.sqrt();

        // Rayleigh range of the effective source.
        let z_r = k * sigma_s * sigma_s;

        // Scale factor for the beam radius: same as free Gaussian beam.
        let scale = (1.0 + (z / z_r).powi(2)).sqrt();

        let w_z = w0 * scale;
        // Coherence length scales identically in the GSM model.
        let lc_z = lc * scale;

        Self::new(w_z, lc_z)
    }

    /// Global degree of coherence ζ ∈ \[0, 1\] (ratio lc/w, normalised).
    ///
    /// ζ = 1/√(1 + 2(w/lc)²)
    pub fn global_coherence(&self) -> f64 {
        let ratio = self.beam_radius / self.coherence_length;
        1.0 / (1.0 + 2.0 * ratio * ratio).sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn mcf_self_coherence_is_real_and_positive() {
        let e = Complex64::new(2.0, 1.0);
        let r = [0.0f64; 3];
        let mcf = MutualCoherenceFunction::new(3e14, r, r, e, e);
        // W(r, r) = |E|² must be real and positive.
        assert!(mcf.value.re > 0.0);
        assert_abs_diff_eq!(mcf.value.im, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn csd_hermitian_symmetry() {
        let positions = vec![-1.0e-6, 0.0, 1.0e-6];
        let lc = 0.5e-6_f64;
        let csd = CrossSpectralDensity::from_schell_model(
            positions,
            &|x| (-x * x / (2.0 * 1e-12)).exp(),
            &|dx| (-dx * dx / (2.0 * lc * lc)).exp(),
        );
        assert!(csd.is_hermitian(1e-12));
    }

    #[test]
    fn degree_of_coherence_self_is_unity() {
        let positions: Vec<f64> = (-5..=5).map(|i| i as f64 * 1e-6).collect();
        let lc = 2e-6_f64;
        let csd = CrossSpectralDensity::from_schell_model(positions.clone(), &|_| 1.0, &|dx| {
            (-dx * dx / (2.0 * lc * lc)).exp()
        });
        // Self-coherence at index 5 (x = 0).
        let r0 = [positions[5], 0.0, 0.0];
        let mu = degree_of_coherence(&csd, r0, r0);
        assert_abs_diff_eq!(mu, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn coherence_matrix_degree_of_polarization() {
        // Fully polarised beam: Jxy = √(Jxx·Jyy), det = 0 → P = 1.
        let jxx = Complex64::new(1.0, 0.0);
        let jyy = Complex64::new(1.0, 0.0);
        let jxy = Complex64::new(1.0, 0.0); // |jxy| = √(Jxx·Jyy)
        let cm = CoherenceMatrix::new(jxx, jxy, jyy);
        assert_abs_diff_eq!(cm.degree_of_polarization(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn coherence_matrix_unpolarised_p_zero() {
        // Unpolarised: Jxx = Jyy = I/2, Jxy = 0 → P = 0.
        let jxx = Complex64::new(0.5, 0.0);
        let jyy = Complex64::new(0.5, 0.0);
        let jxy = Complex64::new(0.0, 0.0);
        let cm = CoherenceMatrix::new(jxx, jxy, jyy);
        assert_abs_diff_eq!(cm.degree_of_polarization(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn van_cittert_zernike_diagonal_is_total_source_power() {
        // For a single point source the far-field W(p, p) = I_source everywhere.
        let intensities = vec![1.0_f64];
        let positions = vec![[0.0_f64, 0.0]];
        let w = van_cittert_zernike_theorem(&intensities, &positions, 633e-9, 1.0)
            .expect("VCZ should succeed");
        // W(0, 0) = intensity (phase = 0).
        assert_abs_diff_eq!(w[0][0].re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn schell_model_beam_coherence_at_zero_is_unity() {
        let beam = SchellModelBeam::new(1e-3, 0.5e-3).expect("valid params");
        let mu = beam.degree_of_coherence(0.0, 0.0);
        assert_abs_diff_eq!(mu, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn schell_model_beam_coherence_decays_with_separation() {
        let lc = 0.5e-3_f64;
        let beam = SchellModelBeam::new(1e-3, lc).expect("valid params");
        let mu_near = beam.degree_of_coherence(lc * 0.1, 0.0);
        let mu_far = beam.degree_of_coherence(lc * 2.0, 0.0);
        assert!(mu_near > mu_far, "coherence must decay with separation");
    }

    #[test]
    fn schell_model_propagation_increases_beam_size() {
        let w0 = 1e-3_f64;
        let lc = 0.5e-3_f64;
        let wavelength = 633e-9_f64;
        let beam = SchellModelBeam::new(w0, lc).expect("valid params");
        let propagated = beam
            .propagate_wigner(1.0, wavelength)
            .expect("propagation should succeed");
        assert!(
            propagated.beam_radius > w0,
            "beam must expand after propagation"
        );
    }
}
