/// Mueller matrix calculus for partially polarized and unpolarized light.
///
/// The Mueller matrix M is a 4×4 real matrix that transforms Stokes vectors:
/// S_out = M · S_in.
///
/// Mueller matrices are applicable to all polarization states including
/// partially polarized and completely unpolarized light, and can represent
/// depolarizing optical elements (unlike Jones matrices).
///
/// # Validity
/// A physically realizable Mueller matrix must satisfy the Cloude coherency
/// criterion: the 4×4 coherency matrix H(M) must be positive semi-definite,
/// i.e., all eigenvalues ≥ 0.
///
/// # Polar Decomposition
/// Any Mueller matrix can be uniquely decomposed (Lu–Chipman decomposition) as:
/// M = M_Δ · M_R · M_D
/// where M_Δ = depolarizer, M_R = pure retarder, M_D = pure diattenuator.
///
/// # References
/// - Lu, S.-Y. & Chipman, R. A., "Interpretation of Mueller matrices based on
///   polar decomposition," JOSA A 13, 1106–1113 (1996).
/// - Chipman, R. A., "Polarimetry," Handbook of Optics Vol. 2 (1995).
use crate::error::OxiPhotonError;
use crate::polarimetry::stokes::StokesVector;

// ── Mueller Matrix ────────────────────────────────────────────────────────────

/// 4×4 real Mueller matrix for an optical system.
///
/// Works for any polarization state of the input light, including partially
/// polarized and unpolarized light.
#[derive(Debug, Clone)]
pub struct MuellerMatrix {
    /// Row-major 4×4 real matrix: m\[row\]\[col\]
    pub m: [[f64; 4]; 4],
}

impl MuellerMatrix {
    /// Construct a Mueller matrix from a row-major 4×4 array.
    pub fn new(m: [[f64; 4]; 4]) -> Self {
        Self { m }
    }

    /// 4×4 identity Mueller matrix.
    pub fn identity() -> Self {
        let mut m = [[0.0f64; 4]; 4];
        for (i, row) in m.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        Self::new(m)
    }

    /// Zero Mueller matrix.
    pub fn zero() -> Self {
        Self::new([[0.0; 4]; 4])
    }

    /// Apply the Mueller matrix to a Stokes vector: S_out = M · S.
    pub fn apply(&self, s: &StokesVector) -> StokesVector {
        let mut out = [0.0f64; 4];
        for (r, out_r) in out.iter_mut().enumerate() {
            for c in 0..4 {
                *out_r += self.m[r][c] * s.s[c];
            }
        }
        // Output may not be strictly valid due to floating-point errors;
        // clamp S0 to be non-negative to prevent spurious failures.
        StokesVector::new_unchecked_pub(out[0].max(0.0), out[1], out[2], out[3])
    }

    /// Cascade two optical elements: `self` acts first, `next` acts second.
    ///
    /// Returns `next · self`.
    pub fn cascade(&self, next: &MuellerMatrix) -> MuellerMatrix {
        mat4x4_mul(next, self)
    }

    // ── Canonical Mueller matrices for common optical elements ───────────

    /// Horizontal linear polarizer (transmission axis along x).
    ///
    /// M = (1/2) \[\[1,1,0,0\\],\[1,1,0,0\],\[0,0,0,0\],\[0,0,0,0\]]
    pub fn linear_polarizer_h() -> Self {
        let h = 0.5;
        Self::new([
            [h, h, 0.0, 0.0],
            [h, h, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
    }

    /// Vertical linear polarizer (transmission axis along y).
    pub fn linear_polarizer_v() -> Self {
        let h = 0.5;
        Self::new([
            [h, -h, 0.0, 0.0],
            [-h, h, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
    }

    /// Linear polarizer with transmission axis at angle `angle_rad` from horizontal.
    ///
    /// M = (1/2) \[\[1, cos2θ, sin2θ, 0\\],
    ///            \[cos2θ, cos²2θ, cos2θ·sin2θ, 0\],
    ///            \[sin2θ, cos2θ·sin2θ, sin²2θ, 0\],
    ///            \[0, 0, 0, 0\]]
    pub fn linear_polarizer(angle_rad: f64) -> Self {
        let two_theta = 2.0 * angle_rad;
        let c = two_theta.cos();
        let s = two_theta.sin();
        let h = 0.5;
        Self::new([
            [h, h * c, h * s, 0.0],
            [h * c, h * c * c, h * c * s, 0.0],
            [h * s, h * c * s, h * s * s, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
    }

    /// Half-wave plate (HWP) with fast axis at angle `fast_axis_rad`.
    ///
    /// Retardation δ = π.
    pub fn half_wave_plate(fast_axis_rad: f64) -> Self {
        Self::wave_plate(std::f64::consts::PI, fast_axis_rad)
    }

    /// Quarter-wave plate (QWP) with fast axis at angle `fast_axis_rad`.
    ///
    /// Retardation δ = π/2.
    pub fn quarter_wave_plate(fast_axis_rad: f64) -> Self {
        Self::wave_plate(std::f64::consts::FRAC_PI_2, fast_axis_rad)
    }

    /// General linear retarder (wave plate) with retardation `delta_rad` and
    /// fast axis at angle `fast_axis_rad` from horizontal.
    ///
    /// Exact Mueller matrix for a pure retarder (no diattenuation).
    pub fn wave_plate(delta_rad: f64, fast_axis_rad: f64) -> Self {
        let theta = fast_axis_rad;
        let delta = delta_rad;
        let two_t = 2.0 * theta;
        let c2 = two_t.cos();
        let s2 = two_t.sin();
        let cd = delta.cos();
        let sd = delta.sin();

        // M = [[1, 0, 0, 0],
        //      [0, c²2θ + s²2θ·cos δ, c2θ·s2θ(1−cos δ), −s2θ·sin δ],
        //      [0, c2θ·s2θ(1−cos δ), s²2θ + c²2θ·cos δ,  c2θ·sin δ],
        //      [0, s2θ·sin δ,        −c2θ·sin δ,           cos δ     ]]
        let m00 = 1.0;
        let m11 = c2 * c2 + s2 * s2 * cd;
        let m12 = c2 * s2 * (1.0 - cd);
        let m13 = -s2 * sd;
        let m21 = c2 * s2 * (1.0 - cd);
        let m22 = s2 * s2 + c2 * c2 * cd;
        let m23 = c2 * sd;
        let m31 = s2 * sd;
        let m32 = -c2 * sd;
        let m33 = cd;

        Self::new([
            [m00, 0.0, 0.0, 0.0],
            [0.0, m11, m12, m13],
            [0.0, m21, m22, m23],
            [0.0, m31, m32, m33],
        ])
    }

    /// Optical rotator by angle `angle_rad` (e.g., Faraday effect, optically active medium).
    ///
    /// M = \[\[1, 0, 0, 0\\],
    ///      \[0, cos2θ, −sin2θ, 0\],
    ///      \[0, sin2θ,  cos2θ, 0\],
    ///      \[0, 0, 0, 1\]]
    pub fn rotator(angle_rad: f64) -> Self {
        let two_t = 2.0 * angle_rad;
        let c = two_t.cos();
        let s = two_t.sin();
        Self::new([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// Partial polarizer / attenuator with independent amplitude transmission
    /// coefficients `px` (x-axis) and `py` (y-axis) for intensity.
    ///
    /// M = (1/2) \[\[px+py, px-py, 0, 0\\],
    ///            \[px-py, px+py, 0, 0\],
    ///            \[0, 0, 2√(px·py), 0\],
    ///            \[0, 0, 0, 2√(px·py)\]]
    ///
    /// where px, py are *intensity* transmittances (not amplitude).
    pub fn attenuator(px: f64, py: f64) -> Self {
        let sum = px + py;
        let diff = px - py;
        let cross = 2.0 * (px * py).sqrt();
        Self::new([
            [0.5 * sum, 0.5 * diff, 0.0, 0.0],
            [0.5 * diff, 0.5 * sum, 0.0, 0.0],
            [0.0, 0.0, cross * 0.5, 0.0],
            [0.0, 0.0, 0.0, cross * 0.5],
        ])
    }

    /// Partial polarizer: alias of `attenuator` with explicit naming.
    ///
    /// `px` and `py` are intensity transmittances along the x and y axes.
    pub fn partial_polarizer(px: f64, py: f64) -> Self {
        Self::attenuator(px, py)
    }

    /// Ideal depolarizer scaled by `depol_factor`.
    ///
    /// When `depol_factor = 0`: complete depolarizer, only S0 passes through.
    /// When `depol_factor = 1`: identity matrix (no depolarization).
    ///
    /// M = diag(1, depol_factor, depol_factor, depol_factor).
    pub fn depolarizer(depol_factor: f64) -> Self {
        let d = depol_factor.clamp(0.0, 1.0);
        Self::new([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, d, 0.0, 0.0],
            [0.0, 0.0, d, 0.0],
            [0.0, 0.0, 0.0, d],
        ])
    }

    /// Ideal planar mirror (normal along z, reflection in xy plane).
    ///
    /// Reflection reverses the handedness of circular polarization and negates S3.
    ///
    /// M = diag(1, 1, −1, −1)  (one sign convention; consistent with Born & Wolf).
    pub fn mirror() -> Self {
        Self::new([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
        ])
    }

    // ── Matrix properties ────────────────────────────────────────────────

    /// Total power transmittance: m\[0\]\[0\] (response to unpolarized input).
    pub fn transmittance(&self) -> f64 {
        self.m[0][0]
    }

    /// Diattenuation: D = √(m01² + m02² + m03²) / m00.
    ///
    /// D = 0: no diattenuation; D = 1: perfect polarizer.
    pub fn diattenuation(&self) -> f64 {
        let m00 = self.m[0][0];
        if m00.abs() < f64::EPSILON {
            return 0.0;
        }
        let d = (self.m[0][1] * self.m[0][1]
            + self.m[0][2] * self.m[0][2]
            + self.m[0][3] * self.m[0][3])
            .sqrt();
        (d / m00).min(1.0)
    }

    /// Polarizance: P = √(m10² + m20² + m30²) / m00.
    ///
    /// Degree of polarization produced from an unpolarized input.
    pub fn polarizance(&self) -> f64 {
        let m00 = self.m[0][0];
        if m00.abs() < f64::EPSILON {
            return 0.0;
        }
        let p = (self.m[1][0] * self.m[1][0]
            + self.m[2][0] * self.m[2][0]
            + self.m[3][0] * self.m[3][0])
            .sqrt();
        (p / m00).min(1.0)
    }

    /// Net phase retardance in radians, extracted from the lower-right 3×3 sub-matrix.
    ///
    /// For a pure retarder, this returns the retardation δ.
    /// For a general Mueller matrix, this is the retardance of the retarder
    /// component in the polar decomposition (Lu–Chipman).
    ///
    /// Uses the formula: δ = arccos\[(tr(m_R)/2) - 1\] where m_R is the upper-left
    /// 3×3 rotation part.
    pub fn retardance_rad(&self) -> f64 {
        // For a pure retarder M the lower-right 3×3 block is a rotation matrix R.
        // tr(R) = 1 + 2·cos(δ) → δ = arccos((tr(R) − 1)/2)
        let trace_sub = self.m[1][1] + self.m[2][2] + self.m[3][3];
        let arg = ((trace_sub - 1.0) / 2.0).clamp(-1.0, 1.0);
        arg.acos()
    }

    /// Check whether this matrix is a physically realizable Mueller matrix.
    ///
    /// Uses the Cloude coherency matrix criterion: H(M) must be positive semi-definite.
    /// All eigenvalues of H must be ≥ −ε (numerical tolerance).
    pub fn is_physical(&self) -> bool {
        let h = self.cloude_coherency_matrix();
        // Check all eigenvalues of the 4×4 Hermitian matrix H ≥ -tol
        // We use the fact that H is 4×4 Hermitian with real diagonal.
        // Compute eigenvalues via characteristic polynomial (4th degree) — expensive
        // but exact for this small matrix.  Use the simpler Gershgorin check as a
        // fast pre-filter, then compute eigenvalues for definitive answer.
        eigenvalues_hermitian_4x4_non_negative(&h, 1e-9)
    }

    /// Alias of `is_physical` (Cloude's physical realizability criterion).
    pub fn satisfies_physically_realizable(&self) -> bool {
        self.is_physical()
    }

    /// Polar decomposition of the Mueller matrix: M = M_Δ · M_R · M_D.
    ///
    /// Implements the Lu–Chipman algorithm.
    pub fn polar_decompose(&self) -> PolarDecomposition {
        lu_chipman_decompose(self)
    }

    // ── Internal helpers ─────────────────────────────────────────────────

    /// Build the 4×4 Cloude coherency (covariance) matrix H(M).
    ///
    /// H = (1/4) Σ_{ij} m_{ij} (σ_i ⊗ σ_j*)
    ///
    /// where σ_0 = I₂, σ_1 = Pauli X, σ_2 = Pauli Y, σ_3 = Pauli Z.
    /// H is Hermitian positive semi-definite iff M is a valid Mueller matrix.
    fn cloude_coherency_matrix(&self) -> [[num_complex::Complex64; 4]; 4] {
        use num_complex::Complex64;

        let zero = Complex64::new(0.0, 0.0);
        let mut h = [[zero; 4]; 4];

        // Pauli basis matrices (2×2 complex), row-major:
        // σ0 = I, σ1 = [[0,1],[1,0]], σ2 = [[0,-i],[i,0]], σ3 = [[1,0],[0,-1]]
        let pauli: [[[Complex64; 2]; 2]; 4] = [
            // σ0
            [
                [Complex64::new(1.0, 0.0), zero],
                [zero, Complex64::new(1.0, 0.0)],
            ],
            // σ1
            [
                [zero, Complex64::new(1.0, 0.0)],
                [Complex64::new(1.0, 0.0), zero],
            ],
            // σ2
            [
                [zero, Complex64::new(0.0, -1.0)],
                [Complex64::new(0.0, 1.0), zero],
            ],
            // σ3
            [
                [Complex64::new(1.0, 0.0), zero],
                [zero, Complex64::new(-1.0, 0.0)],
            ],
        ];

        // Kronecker product σ_i ⊗ σ_j* in the 4×4 basis
        // row indices: (a,b) in {0,1}×{0,1} mapped to a*2+b
        for i in 0..4 {
            for j in 0..4 {
                let mij = Complex64::new(self.m[i][j], 0.0);
                // Add mij * (σ_i ⊗ σ_j*) to H
                for ra in 0..2usize {
                    for rb in 0..2usize {
                        for ca in 0..2usize {
                            for cb in 0..2usize {
                                let row = ra * 2 + rb;
                                let col = ca * 2 + cb;
                                h[row][col] += mij * pauli[i][ra][ca] * pauli[j][rb][cb].conj();
                            }
                        }
                    }
                }
            }
        }

        // Scale by 1/4
        for row in h.iter_mut() {
            for val in row.iter_mut() {
                *val *= Complex64::new(0.25, 0.0);
            }
        }

        h
    }
}

// ── Polar decomposition result ────────────────────────────────────────────────

/// Result of the Lu–Chipman polar decomposition of a Mueller matrix.
///
/// M = M_Δ · M_R · M_D  (depolarizer × retarder × diattenuator).
pub struct PolarDecomposition {
    /// Depolarizer component M_Δ
    pub depolarizer: MuellerMatrix,
    /// Pure retarder component M_R
    pub retarder: MuellerMatrix,
    /// Pure diattenuator component M_D
    pub diattenuator: MuellerMatrix,
    /// Scalar diattenuation magnitude D ∈ \[0, 1\]
    pub diattenuation: f64,
    /// Scalar retardance δ in radians
    pub retardance_rad: f64,
}

// ── Stokes Polarimeter ────────────────────────────────────────────────────────

/// Stokes polarimeter: reconstructs the Stokes vector from intensity measurements.
///
/// In a Stokes polarimeter, the input light passes through N different analysis
/// states (polarizer + wave plate combinations), and the transmitted intensity
/// for each setting is recorded:
///
///   I_k = (1/2) A_k · S
///
/// where A_k is the analysis Stokes vector for measurement k.
/// Stacking all measurements: I = W · S, where W is the N×4 measurement matrix.
/// Reconstruction: S = W⁺ · I (pseudo-inverse).
pub struct StokesPolarimeter {
    /// Analysis Stokes vectors (one per measurement channel).
    pub analysis_vectors: Vec<StokesVector>,
}

impl StokesPolarimeter {
    /// Standard four-state Stokes polarimeter using H, V, +45°, RCP analysis states.
    ///
    /// This is the minimal complete Stokes polarimeter (4 measurements for 4 unknowns).
    pub fn standard_four_state() -> Self {
        Self {
            analysis_vectors: vec![
                StokesVector::horizontal(),
                StokesVector::vertical(),
                StokesVector::diagonal_p45(),
                StokesVector::right_circular(),
            ],
        }
    }

    /// Reconstruct the Stokes vector from measured intensities.
    ///
    /// Solves the linear system I = (1/2) W · S for S using the pseudo-inverse of W.
    ///
    /// # Parameters
    /// - `intensities`: measured intensity values, one per analysis state.
    ///
    /// # Errors
    /// Returns an error if the number of intensities doesn't match the number of
    /// analysis states, or if the measurement matrix is singular.
    pub fn reconstruct(&self, intensities: &[f64]) -> Result<StokesVector, OxiPhotonError> {
        let n = self.analysis_vectors.len();
        if intensities.len() != n {
            return Err(OxiPhotonError::NumericalError(format!(
                "Expected {n} intensity measurements, got {}",
                intensities.len()
            )));
        }
        if n < 4 {
            return Err(OxiPhotonError::NumericalError(
                "At least 4 analysis states required for Stokes reconstruction".into(),
            ));
        }

        // Build measurement matrix W (n × 4), row k = analysis_vectors[k].s
        let w = self.measurement_matrix();
        // Solve W · s = 2 * I (the factor of 2 comes from convention I = 0.5 * A · S)
        let rhs: Vec<f64> = intensities.iter().map(|&x| 2.0 * x).collect();

        // Pseudo-inverse: W⁺ = (WᵀW)⁻¹Wᵀ
        // WᵀW is 4×4, Wᵀ·rhs is 4-vector
        let wt_w = mat_ata(&w, n, 4);
        let wt_b = mat_at_b(&w, n, 4, &rhs);

        let s_vec = solve_4x4(&wt_w, &wt_b).ok_or_else(|| {
            OxiPhotonError::NumericalError(
                "Measurement matrix is singular; cannot reconstruct Stokes vector".into(),
            )
        })?;

        StokesVector::new(s_vec[0], s_vec[1], s_vec[2], s_vec[3]).map_err(|e| {
            OxiPhotonError::NumericalError(format!("Reconstructed Stokes vector invalid: {e}"))
        })
    }

    /// Condition number of the measurement matrix W, estimated as σ_max / σ_min.
    ///
    /// Lower values indicate a more robust polarimeter configuration.
    pub fn condition_number(&self) -> f64 {
        let n = self.analysis_vectors.len();
        let w = self.measurement_matrix();
        // Estimate condition number via SVD of WᵀW (4×4 symmetric positive semi-definite).
        // Condition of W = sqrt(condition of WᵀW).
        let wt_w = mat_ata(&w, n, 4);
        let eigenvals = symmetric_4x4_eigenvalues(&wt_w);
        let max_ev = eigenvals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_ev = eigenvals.iter().cloned().fold(f64::INFINITY, f64::min);
        if min_ev.abs() < f64::EPSILON {
            return f64::INFINITY;
        }
        (max_ev / min_ev).abs().sqrt()
    }

    // ── Internal helpers ─────────────────────────────────────────────────

    /// Build the n×4 measurement matrix from the analysis vectors.
    fn measurement_matrix(&self) -> Vec<Vec<f64>> {
        self.analysis_vectors.iter().map(|a| a.s.to_vec()).collect()
    }
}

// ── Numerical helpers ─────────────────────────────────────────────────────────

/// 4×4 matrix multiplication result = a × b (both row-major as flat Vec).
fn mat4x4_mul(a: &MuellerMatrix, b: &MuellerMatrix) -> MuellerMatrix {
    let mut r = [[0.0f64; 4]; 4];
    for (i, r_row) in r.iter_mut().enumerate() {
        for (j, r_ij) in r_row.iter_mut().enumerate() {
            for k in 0..4 {
                *r_ij += a.m[i][k] * b.m[k][j];
            }
        }
    }
    MuellerMatrix::new(r)
}

/// Compute AᵀA for an n×4 matrix A (stored as Vec of row Vecs).
fn mat_ata(a: &[Vec<f64>], n: usize, cols: usize) -> [[f64; 4]; 4] {
    let mut r = [[0.0f64; 4]; 4];
    for (i, r_row) in r.iter_mut().enumerate().take(cols) {
        for (j, r_ij) in r_row.iter_mut().enumerate().take(cols) {
            for a_row in a.iter().take(n) {
                *r_ij += a_row[i] * a_row[j];
            }
        }
    }
    r
}

/// Compute Aᵀb for an n×4 matrix A and length-n vector b.
fn mat_at_b(a: &[Vec<f64>], n: usize, cols: usize, b: &[f64]) -> [f64; 4] {
    let mut r = [0.0f64; 4];
    for j in 0..cols {
        for k in 0..n {
            r[j] += a[k][j] * b[k];
        }
    }
    r
}

/// Solve a 4×4 linear system Ax = b using Gaussian elimination with partial pivoting.
/// Returns None if A is singular.
#[allow(clippy::needless_range_loop)]
fn solve_4x4(a: &[[f64; 4]; 4], b: &[f64; 4]) -> Option<[f64; 4]> {
    let mut aug = [[0.0f64; 5]; 4];
    for (r, aug_r) in aug.iter_mut().enumerate() {
        for (c, aug_rc) in aug_r.iter_mut().enumerate().take(4) {
            *aug_rc = a[r][c];
        }
        aug_r[4] = b[r];
    }

    for col in 0..4 {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for (row, aug_row) in aug.iter().enumerate().skip(col + 1) {
            if aug_row[col].abs() > max_val {
                max_val = aug_row[col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return None; // singular
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for row in (col + 1)..4 {
            let factor = aug[row][col] / pivot;
            let (left, right) = aug.split_at_mut(row);
            for (piv_c, elim_c) in left[col][col..5].iter().zip(right[0][col..5].iter_mut()) {
                *elim_c -= piv_c * factor;
            }
        }
    }

    // Back substitution
    let mut x = [0.0f64; 4];
    for row in (0..4).rev() {
        x[row] = aug[row][4];
        for (c, aug_row_c) in aug[row].iter().enumerate().skip(row + 1).take(4 - row - 1) {
            x[row] -= aug_row_c * x[c];
        }
        if aug[row][row].abs() < 1e-14 {
            return None;
        }
        x[row] /= aug[row][row];
    }
    Some(x)
}

/// Compute eigenvalues of a real symmetric 4×4 matrix using the Jacobi eigenvalue algorithm.
///
/// The classical Jacobi method iteratively applies Givens rotations to zero off-diagonal
/// elements.  It is robust, converges quadratically, and correct for small matrices.
///
/// Returns all four real eigenvalues (unordered).
#[allow(clippy::needless_range_loop)]
fn symmetric_4x4_eigenvalues(a: &[[f64; 4]; 4]) -> [f64; 4] {
    const N: usize = 4;
    const MAX_SWEEPS: usize = 50;
    const TOL: f64 = 1e-15;

    let mut s = *a; // working copy

    for _ in 0..MAX_SWEEPS {
        // Sum of squares of off-diagonal elements
        let mut off_sq = 0.0f64;
        for (i, s_row) in s.iter().enumerate().take(N) {
            for j in (i + 1)..N {
                off_sq += s_row[j] * s_row[j];
            }
        }
        if off_sq < TOL {
            break;
        }

        // One Jacobi sweep: rotate all (p,q) pairs with p < q
        for p in 0..N {
            for q in (p + 1)..N {
                if s[p][q].abs() < TOL {
                    continue;
                }
                // Compute Jacobi rotation angle θ
                let tau = (s[q][q] - s[p][p]) / (2.0 * s[p][q]);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    1.0 / (tau - (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let sn = t * c;

                // Update diagonal
                let spp = s[p][p];
                let sqq = s[q][q];
                let spq = s[p][q];
                s[p][p] = spp - t * spq;
                s[q][q] = sqq + t * spq;
                s[p][q] = 0.0;
                s[q][p] = 0.0;

                // Update off-diagonal rows/columns
                for r in 0..N {
                    if r == p || r == q {
                        continue;
                    }
                    let srp = s[r][p];
                    let srq = s[r][q];
                    let new_rp = c * srp - sn * srq;
                    let new_rq = sn * srp + c * srq;
                    s[r][p] = new_rp;
                    s[p][r] = new_rp;
                    s[r][q] = new_rq;
                    s[q][r] = new_rq;
                }
            }
        }
    }

    [s[0][0], s[1][1], s[2][2], s[3][3]]
}

/// Check whether all eigenvalues of a 4×4 complex Hermitian matrix are ≥ −tolerance.
///
/// A 4×4 Hermitian matrix H = A + iB with A = Aᵀ, B = −Bᵀ can be embedded into
/// an 8×8 real symmetric matrix:
///
///   M_real = \[\[A, −B\\], \[B, A\]]
///
/// Eigenvalues of M_real come in pairs equal to the eigenvalues of H (each with
/// multiplicity 2).  We run the Jacobi algorithm on the 8×8 real symmetric matrix
/// and check that all eigenvalues are ≥ −tol.
#[allow(clippy::needless_range_loop)]
fn eigenvalues_hermitian_4x4_non_negative(h: &[[num_complex::Complex64; 4]; 4], tol: f64) -> bool {
    const N8: usize = 8;

    // Build the 8×8 real representation [[A, -B],[B, A]]
    let mut real8 = [[0.0f64; N8]; N8];
    for r in 0..4 {
        for c in 0..4 {
            let a_rc = h[r][c].re;
            let b_rc = h[r][c].im; // H = A + iB
            real8[r][c] = a_rc; // top-left: A
            real8[r][c + 4] = -b_rc; // top-right: -B
            real8[r + 4][c] = b_rc; // bottom-left: B
            real8[r + 4][c + 4] = a_rc; // bottom-right: A
        }
    }

    // Jacobi eigenvalue algorithm for 8×8 real symmetric matrix
    const MAX_SWEEPS: usize = 80;
    const TOL_J: f64 = 1e-14;

    for _ in 0..MAX_SWEEPS {
        let mut off_sq = 0.0f64;
        for (i, row_i) in real8.iter().enumerate().take(N8) {
            for j in (i + 1)..N8 {
                off_sq += row_i[j] * row_i[j];
            }
        }
        if off_sq < TOL_J {
            break;
        }

        for p in 0..N8 {
            for q in (p + 1)..N8 {
                if real8[p][q].abs() < TOL_J {
                    continue;
                }
                let tau = (real8[q][q] - real8[p][p]) / (2.0 * real8[p][q]);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    1.0 / (tau - (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let sn = t * c;

                let spp = real8[p][p];
                let sqq = real8[q][q];
                let spq = real8[p][q];
                real8[p][p] = spp - t * spq;
                real8[q][q] = sqq + t * spq;
                real8[p][q] = 0.0;
                real8[q][p] = 0.0;

                for r in 0..N8 {
                    if r == p || r == q {
                        continue;
                    }
                    let srp = real8[r][p];
                    let srq = real8[r][q];
                    let new_rp = c * srp - sn * srq;
                    let new_rq = sn * srp + c * srq;
                    real8[r][p] = new_rp;
                    real8[p][r] = new_rp;
                    real8[r][q] = new_rq;
                    real8[q][r] = new_rq;
                }
            }
        }
    }

    // Check all diagonal elements (eigenvalues)
    (0..N8).all(|i| real8[i][i] >= -tol)
}

/// Lu–Chipman polar decomposition: M = M_Δ · M_R · M_D.
///
/// Ref: Lu & Chipman, JOSA A 13, 1106–1113 (1996).
fn lu_chipman_decompose(m: &MuellerMatrix) -> PolarDecomposition {
    // Step 1: Extract diattenuation vector d = [m01, m02, m03] / m00
    let m00 = m.m[0][0];
    let d_vec = if m00.abs() > f64::EPSILON {
        [m.m[0][1] / m00, m.m[0][2] / m00, m.m[0][3] / m00]
    } else {
        [0.0; 3]
    };
    let d_mag = (d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2]).sqrt();

    // Step 2: Build the diattenuator M_D
    let m_d = build_diattenuator(m00, &d_vec, d_mag);

    // Step 3: M' = M · M_D^{-1}
    // For a pure diattenuator, M_D^{-1} is easy to compute.
    let m_d_inv = diattenuator_inverse(&m_d, d_mag, m00);
    let m_prime = mat4x4_mul(m, &m_d_inv);

    // Step 4: Extract retarder M_R from M' (upper-left 3×3 block is an SO(3) rotation)
    // Extract M_R as the closest rotation to the upper-left 3×3 of M'
    let m_r = extract_retarder(&m_prime);

    // Step 5: M_delta = M' · M_R^{-1} = M' · M_R^T (since M_R is orthogonal)
    let m_r_t = retarder_transpose(&m_r);
    let m_delta = mat4x4_mul(&m_prime, &m_r_t);

    let retardance = m_r.retardance_rad();

    PolarDecomposition {
        depolarizer: m_delta,
        retarder: m_r,
        diattenuator: m_d,
        diattenuation: d_mag.min(1.0),
        retardance_rad: retardance,
    }
}

/// Build the diattenuator Mueller matrix M_D from diattenuation vector d and magnitude.
fn build_diattenuator(m00: f64, d_vec: &[f64; 3], d_mag: f64) -> MuellerMatrix {
    // M_D = m00 * [[1, dᵀ],[d, (1 + sqrt(1-D²))/2 · I + (1 - sqrt(1-D²))/2 · ddᵀ/D² ]]
    // Simplified for D < 1:
    let factor = if d_mag > f64::EPSILON {
        (1.0 - d_mag * d_mag).max(0.0).sqrt()
    } else {
        1.0
    };

    let mut md = [[0.0f64; 4]; 4];
    md[0][0] = m00;
    for j in 0..3 {
        md[0][j + 1] = m00 * d_vec[j];
        md[j + 1][0] = m00 * d_vec[j];
    }

    let a = (1.0 + factor) / 2.0;
    let b = if d_mag > f64::EPSILON {
        (1.0 - factor) / (2.0 * d_mag * d_mag)
    } else {
        0.0
    };

    for i in 0..3 {
        for j in 0..3 {
            let delta = if i == j { 1.0 } else { 0.0 };
            md[i + 1][j + 1] = m00 * (a * delta + b * d_vec[i] * d_vec[j]);
        }
    }

    MuellerMatrix::new(md)
}

/// Compute the inverse of a diattenuator Mueller matrix.
#[allow(clippy::needless_range_loop)]
fn diattenuator_inverse(m_d: &MuellerMatrix, d_mag: f64, m00: f64) -> MuellerMatrix {
    // M_D is symmetric; its inverse can be computed by treating it as a 4×4 system.
    // For small matrices, use direct inversion via Gaussian elimination.
    let mut aug = [[0.0f64; 8]; 4];
    for (r, aug_r) in aug.iter_mut().enumerate() {
        for (c, aug_rc) in aug_r.iter_mut().enumerate().take(4) {
            *aug_rc = m_d.m[r][c];
        }
        aug_r[r + 4] = 1.0;
    }

    // Gaussian elimination
    for col in 0..4 {
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for (row, aug_row) in aug.iter().enumerate().skip(col + 1).take(4 - col - 1) {
            if aug_row[col].abs() > max_val {
                max_val = aug_row[col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            // Singular: fall back to a scaled identity
            let scale = if m00.abs() > f64::EPSILON {
                1.0 / m00
            } else {
                1.0
            };
            let _ = d_mag; // suppress warning
            return MuellerMatrix::identity().scale(scale);
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for aug_col_c in aug[col].iter_mut() {
            *aug_col_c /= pivot;
        }
        for row in 0..4 {
            if row != col {
                let factor = aug[row][col];
                let col_row_copy = aug[col];
                for (aug_row_c, &col_c) in aug[row].iter_mut().zip(col_row_copy.iter()) {
                    *aug_row_c -= col_c * factor;
                }
            }
        }
    }

    let mut inv = [[0.0f64; 4]; 4];
    for (r, inv_r) in inv.iter_mut().enumerate() {
        for (c, inv_rc) in inv_r.iter_mut().enumerate() {
            *inv_rc = aug[r][c + 4];
        }
    }
    MuellerMatrix::new(inv)
}

/// Extract the retarder (pure rotation) component from a Mueller matrix M'
/// whose top-left 3×3 sub-block is approximately a rotation matrix.
fn extract_retarder(m: &MuellerMatrix) -> MuellerMatrix {
    // The retarder M_R has the form:
    // [[1, 0ᵀ], [0, R_3x3]] where R is SO(3).
    // We extract the 3×3 sub-block and orthogonalize via the polar decomposition of R.
    let mut r33 = [[0.0f64; 3]; 3];
    for (i, r33_row) in r33.iter_mut().enumerate() {
        for (j, r33_ij) in r33_row.iter_mut().enumerate() {
            *r33_ij = m.m[i + 1][j + 1];
        }
    }

    // Polar decomposition of R33: R33 = U · S where U is orthogonal.
    // We use the iterative method: U_k+1 = 0.5 * (U_k + (U_kᵀ)^{-1})
    let u33 = orthogonalize_3x3(&r33);

    let mut mr = [[0.0f64; 4]; 4];
    mr[0][0] = 1.0;
    for i in 0..3 {
        for j in 0..3 {
            mr[i + 1][j + 1] = u33[i][j];
        }
    }
    MuellerMatrix::new(mr)
}

/// Transpose of a pure retarder (= inverse, since M_R is orthogonal).
fn retarder_transpose(m_r: &MuellerMatrix) -> MuellerMatrix {
    let mut r = [[0.0f64; 4]; 4];
    for (i, mr_row) in m_r.m.iter().enumerate() {
        for (j, &val) in mr_row.iter().enumerate() {
            r[j][i] = val;
        }
    }
    MuellerMatrix::new(r)
}

/// Compute the nearest orthogonal matrix to a 3×3 real matrix via Cayley iteration.
/// Converges quadratically to U = R · (RᵀR)^{-1/2}.
fn orthogonalize_3x3(a: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut u = *a;

    for _ in 0..20 {
        // u_next = 0.5 * (u + (uᵀ)^{-1}) = 0.5 * (u + adj(u)ᵀ / det(u))
        let det = det3x3(&u);
        if det.abs() < 1e-14 {
            break;
        }
        let inv = inv3x3(&u, det);
        // u = 0.5*(u + invᵀ)
        let mut u_next = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                u_next[i][j] = 0.5 * (u[i][j] + inv[j][i]);
            }
        }
        // Check convergence
        let mut diff = 0.0f64;
        for i in 0..3 {
            for j in 0..3 {
                diff += (u_next[i][j] - u[i][j]).powi(2);
            }
        }
        u = u_next;
        if diff.sqrt() < 1e-12 {
            break;
        }
    }
    u
}

fn det3x3(a: &[[f64; 3]; 3]) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

fn inv3x3(a: &[[f64; 3]; 3], det: f64) -> [[f64; 3]; 3] {
    let inv_det = 1.0 / det;
    [
        [
            (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * inv_det,
            (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv_det,
            (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv_det,
        ],
        [
            (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * inv_det,
            (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv_det,
            (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv_det,
        ],
        [
            (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * inv_det,
            (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv_det,
            (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv_det,
        ],
    ]
}

impl MuellerMatrix {
    /// Scale all elements of the matrix by a scalar.
    fn scale(&self, factor: f64) -> Self {
        let mut m = self.m;
        for row in m.iter_mut() {
            for val in row.iter_mut() {
                *val *= factor;
            }
        }
        Self::new(m)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn stokes_eq(a: &StokesVector, b: &StokesVector, tol: f64) -> bool {
        a.s.iter().zip(b.s.iter()).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn test_mueller_identity_preserves_stokes() {
        let id = MuellerMatrix::identity();
        let states = [
            StokesVector::horizontal(),
            StokesVector::vertical(),
            StokesVector::diagonal_p45(),
            StokesVector::right_circular(),
            StokesVector::unpolarized(1.0),
        ];
        for s in &states {
            let out = id.apply(s);
            assert!(stokes_eq(&out, s, EPS), "identity failed for {:?}", s);
        }
    }

    #[test]
    fn test_horizontal_polarizer_kills_vertical() {
        let pol_h = MuellerMatrix::linear_polarizer_h();
        let v = StokesVector::vertical();
        let out = pol_h.apply(&v);
        assert!(out.intensity() < EPS, "intensity={}", out.intensity());
    }

    #[test]
    fn test_hwp_mueller_effect() {
        // HWP with fast axis at 0° flips S3 sign (converts RCP to LCP)
        let hwp = MuellerMatrix::half_wave_plate(0.0);
        let rcp = StokesVector::right_circular();
        let out = hwp.apply(&rcp);
        assert!(approx_eq(out.s[3], -1.0, EPS), "S3={}", out.s[3]);
        assert!(approx_eq(out.s[0], 1.0, EPS), "S0={}", out.s[0]);
    }

    #[test]
    fn test_cascade_two_qwp() {
        // Two QWPs at the same angle = HWP
        let qwp = MuellerMatrix::quarter_wave_plate(0.0);
        let two_qwp = qwp.cascade(&qwp);
        let hwp = MuellerMatrix::half_wave_plate(0.0);
        for r in 0..4 {
            for c in 0..4 {
                assert!(
                    approx_eq(two_qwp.m[r][c], hwp.m[r][c], 1e-10),
                    "m[{r}][{c}]: 2QWP={} HWP={}",
                    two_qwp.m[r][c],
                    hwp.m[r][c]
                );
            }
        }
    }

    #[test]
    fn test_depolarizer_zero() {
        // Full depolarizer (depol_factor=0) reduces any input to unpolarized
        let depol = MuellerMatrix::depolarizer(0.0);
        let states = [
            StokesVector::horizontal(),
            StokesVector::right_circular(),
            StokesVector::diagonal_p45(),
        ];
        for s in &states {
            let out = depol.apply(s);
            assert!(out.dop() < EPS, "DOP={}", out.dop());
        }
    }

    #[test]
    fn test_diattenuation_linear_polarizer() {
        let pol = MuellerMatrix::linear_polarizer(0.0);
        let d = pol.diattenuation();
        assert!(approx_eq(d, 1.0, EPS), "D={d}");
    }

    #[test]
    fn test_mueller_physical_check() {
        // Known physical matrices should pass
        assert!(MuellerMatrix::identity().is_physical());
        assert!(MuellerMatrix::linear_polarizer_h().is_physical());
        assert!(MuellerMatrix::half_wave_plate(0.0).is_physical());
        assert!(MuellerMatrix::quarter_wave_plate(FRAC_PI_2).is_physical());
    }

    #[test]
    fn test_polar_decompose_hwp() {
        // HWP polar decomposition: retardance ≈ π
        let hwp = MuellerMatrix::half_wave_plate(0.0);
        let decomp = hwp.polar_decompose();
        let ret = decomp.retardance_rad;
        assert!(approx_eq(ret, PI, 1e-6), "retardance={ret} expected π={PI}");
    }

    #[test]
    fn test_rotator_mueller() {
        // Rotation by 45° should convert H → +45° state
        let rot = MuellerMatrix::rotator(PI / 4.0);
        let h = StokesVector::horizontal();
        let out = rot.apply(&h);
        // +45°: S1 ≈ 0, S2 ≈ 1
        assert!(out.s[1].abs() < 1e-9, "S1={}", out.s[1]);
        assert!(approx_eq(out.s[2], 1.0, 1e-9), "S2={}", out.s[2]);
    }

    #[test]
    fn test_transmittance_identity() {
        let id = MuellerMatrix::identity();
        assert!(approx_eq(id.transmittance(), 1.0, EPS));
    }

    #[test]
    fn test_polarizance_linear_polarizer() {
        let pol = MuellerMatrix::linear_polarizer_h();
        let p = pol.polarizance();
        assert!(approx_eq(p, 1.0, EPS), "P={p}");
    }

    #[test]
    fn test_stokes_polarimeter_reconstruct() {
        let pol = StokesPolarimeter::standard_four_state();
        // Simulate measurement of horizontal light: I_k = 0.5 * A_k · S
        let s_true = StokesVector::horizontal();
        let intensities: Vec<f64> = pol
            .analysis_vectors
            .iter()
            .map(|a| {
                0.5 * (a.s[0] * s_true.s[0]
                    + a.s[1] * s_true.s[1]
                    + a.s[2] * s_true.s[2]
                    + a.s[3] * s_true.s[3])
            })
            .collect();
        let s_rec = pol.reconstruct(&intensities).expect("reconstruct");
        assert!(
            stokes_eq(&s_rec, &s_true, 1e-8),
            "reconstructed={:?} expected={:?}",
            s_rec,
            s_true
        );
    }
}
