//! MZI Mesh architectures for optical matrix-vector multiplication.
//!
//! Implements the Clements (rectangular) and Reck (triangular) universal
//! unitary decomposition architectures using Mach-Zehnder interferometers.

use num_complex::Complex64;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// MZI unit cell
// ─────────────────────────────────────────────────────────────────────────────

/// A single Mach-Zehnder interferometer (MZI) unit cell.
///
/// The transfer matrix is:
/// ```text
/// T = [[cos(θ/2)·e^{iφ},  i·sin(θ/2)·e^{iφ}],
///       [i·sin(θ/2),       cos(θ/2)         ]]
/// ```
#[derive(Debug, Clone)]
pub struct MziCell {
    /// Internal phase shift θ controlling splitting ratio.
    pub theta: f64,
    /// External phase shift φ applied to the upper arm.
    pub phi: f64,
}

impl MziCell {
    /// Create a new MZI cell with given phase shifts.
    pub fn new(theta: f64, phi: f64) -> Self {
        Self { theta, phi }
    }

    /// Compute the 2×2 transfer matrix of this MZI.
    ///
    /// ```text
    /// T = [[cos(θ/2)·e^{iφ},  i·sin(θ/2)·e^{iφ}],
    ///       [i·sin(θ/2),       cos(θ/2)         ]]
    /// ```
    pub fn transfer_matrix(&self) -> [[Complex64; 2]; 2] {
        let half = self.theta / 2.0;
        let c = half.cos();
        let s = half.sin();
        let ep = Complex64::from_polar(1.0, self.phi);
        let i = Complex64::new(0.0, 1.0);
        [[ep * c, i * ep * s], [i * s, Complex64::new(c, 0.0)]]
    }

    /// Power transmissivity of the cross port: T = cos²(θ/2).
    pub fn transmissivity(&self) -> f64 {
        let c = (self.theta / 2.0).cos();
        c * c
    }

    /// Set power transmissivity (0..=1); computes θ = 2·acos(√t).
    pub fn set_transmissivity(&mut self, t: f64) {
        let t_clamped = t.clamp(0.0, 1.0);
        self.theta = 2.0 * t_clamped.sqrt().acos();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Clements (rectangular) architecture
// ─────────────────────────────────────────────────────────────────────────────

/// N×N rectangular MZI mesh using the Clements decomposition.
///
/// The Clements architecture achieves a universal N×N unitary with n(n-1)/2
/// MZI cells arranged in n columns of alternating even/odd pairs, with depth n.
pub struct ClementsArch {
    /// Port count.
    pub n: usize,
    /// Columns of MZI cells. `columns[col][row]` is `Some` when that position
    /// holds an active MZI, `None` otherwise (structural zeros in the mesh).
    pub columns: Vec<Vec<Option<MziCell>>>,
    /// Per-output diagonal phase shifts (length n).
    pub diag_phases: Vec<f64>,
}

impl ClementsArch {
    /// Construct an identity mesh: all θ = 0, φ = 0.
    pub fn new(n: usize) -> Self {
        let mut columns = Vec::with_capacity(n);
        for col in 0..n {
            // Alternating even/odd columns: even cols pair (0,1),(2,3),...
            // odd cols pair (1,2),(3,4),...
            let start = col % 2;
            let mut column = vec![None; n - 1];
            let mut row = start;
            while row + 1 < n {
                column[row] = Some(MziCell::new(0.0, 0.0));
                row += 2;
            }
            columns.push(column);
        }
        Self {
            n,
            columns,
            diag_phases: vec![0.0; n],
        }
    }

    /// Number of MZI cells: n(n-1)/2.
    pub fn n_mzis(&self) -> usize {
        self.n * (self.n - 1) / 2
    }

    /// Depth (number of columns): n.
    pub fn depth(&self) -> usize {
        self.n
    }

    /// Apply a 2×2 MZI transfer matrix at rows (r, r+1) of vector `v`.
    fn apply_mzi_inplace(v: &mut [Complex64], r: usize, mzi: &MziCell) {
        let t = mzi.transfer_matrix();
        let a = v[r];
        let b = v[r + 1];
        v[r] = t[0][0] * a + t[0][1] * b;
        v[r + 1] = t[1][0] * a + t[1][1] * b;
    }

    /// Apply the full Clements mesh to an input vector.
    pub fn apply(&self, input: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(input.len(), self.n, "input length must equal n");
        let mut v: Vec<Complex64> = input.to_vec();

        for col in &self.columns {
            for (row, cell_opt) in col.iter().enumerate() {
                if let Some(mzi) = cell_opt {
                    Self::apply_mzi_inplace(&mut v, row, mzi);
                }
            }
        }

        // Apply diagonal phases
        for (i, &phase) in self.diag_phases.iter().enumerate() {
            v[i] *= Complex64::from_polar(1.0, phase);
        }

        v
    }

    /// Compute the full N×N unitary matrix by applying the mesh to each
    /// standard basis vector.
    pub fn to_unitary(&self) -> Vec<Vec<Complex64>> {
        let n = self.n;
        let zero = Complex64::new(0.0, 0.0);
        (0..n)
            .map(|j| {
                let mut e = vec![zero; n];
                e[j] = Complex64::new(1.0, 0.0);
                self.apply(&e)
            })
            .collect()
    }

    /// Decompose a unitary matrix into Clements mesh parameters.
    ///
    /// Implements the Clements et al. (Optica 2016) algorithm: nulling
    /// off-diagonal elements column by column using T_{pq}·U operations.
    pub fn from_unitary(u: &[Vec<Complex64>]) -> Self {
        let n = u.len();
        assert!(n >= 2, "matrix must be at least 2×2");
        for row in u {
            assert_eq!(row.len(), n, "matrix must be square");
        }

        let mut arch = Self::new(n);
        let mut work: Vec<Vec<Complex64>> = u.to_vec();

        // Clements decomposition: alternating left-multiply and right-multiply
        // by T^† to zero sub-diagonal elements.
        // We record (col_index, row_index, theta, phi) for each MZI.
        let mut mzi_params: Vec<(usize, usize, f64, f64)> = Vec::new();
        let diag_left: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n]; // diagonal phases from left ops

        for col in 0..n {
            if col % 2 == 0 {
                // Zero work[n-1-col+1 .. n-1][col] from bottom by left-multiply T·w
                let top = col / 2;
                let bottom = n - 1 - top;
                let mut r = bottom;
                while r > col {
                    // Null element work[r][col] using rows (r-1, r)
                    let a = work[r - 1][col];
                    let b = work[r][col];
                    let (theta, phi) = nulling_angles(a, b);
                    // Left multiply rows (r-1, r) by T(theta, phi)
                    let mzi = MziCell::new(theta, phi);
                    left_multiply_rows(&mut work, n, r - 1, &mzi);
                    mzi_params.push((col, r - 1, theta, phi));
                    if r >= 2 {
                        r -= 2;
                    } else {
                        break;
                    }
                }
            } else {
                // Zero work[col][col+1 .. n-1] from right by right-multiply w·T^†
                let c = col / 2;
                let mut s = n - 1 - c;
                while s > col {
                    let a = work[col][s - 1];
                    let b = work[col][s];
                    let (theta, phi) = nulling_angles_right(a, b);
                    let mzi = MziCell::new(theta, phi);
                    right_multiply_cols(&mut work, n, s - 1, &mzi);
                    mzi_params.push((col, s - 1, theta, phi));
                    if s >= 2 {
                        s -= 2;
                    } else {
                        break;
                    }
                }
            }
        }

        // Extract diagonal phases from what remains (should be diagonal unitary)
        let mut diag = vec![0.0f64; n];
        for i in 0..n {
            if work[i][i].norm() > 1e-10 {
                diag[i] = work[i][i].arg();
            }
        }
        // Suppress unused variable warning
        let _ = diag_left;

        // Program the arch with extracted MZI parameters.
        // Map (col, row) back to the mesh columns.
        for (col_idx, row_idx, theta, phi) in &mzi_params {
            let col = col_idx % arch.columns.len();
            if row_idx + 1 < arch.n
                && col < arch.columns.len()
                && *row_idx < arch.columns[col].len()
            {
                arch.columns[col][*row_idx] = Some(MziCell::new(*theta, *phi));
            }
        }
        arch.diag_phases = diag;
        arch
    }

    /// Program the mesh for a target unitary (calls from_unitary).
    pub fn program(&mut self, target_unitary: &[Vec<Complex64>]) {
        let new_arch = Self::from_unitary(target_unitary);
        self.columns = new_arch.columns;
        self.diag_phases = new_arch.diag_phases;
    }

    /// Phase sensitivity: partial derivative of output power w.r.t. θ at (col,row).
    ///
    /// Estimated as the Frobenius-norm change in the unitary column for a small
    /// perturbation dθ = 1e-6 rad.
    pub fn phase_sensitivity(&self, col: usize, row: usize) -> f64 {
        let eps = 1e-6_f64;
        if col >= self.columns.len() || row >= self.columns[col].len() {
            return 0.0;
        }
        if self.columns[col][row].is_none() {
            return 0.0;
        }

        // Perturb theta at (col, row) and measure ‖ΔU‖_F
        let u0 = self.to_unitary();

        let mut perturbed = Self {
            n: self.n,
            columns: self.columns.clone(),
            diag_phases: self.diag_phases.clone(),
        };
        if let Some(ref mut mzi) = perturbed.columns[col][row] {
            mzi.theta += eps;
        }
        let u1 = perturbed.to_unitary();

        let mut norm_sq = 0.0_f64;
        for i in 0..self.n {
            for j in 0..self.n {
                let d = u1[i][j] - u0[i][j];
                norm_sq += d.norm_sqr();
            }
        }
        norm_sq.sqrt() / eps
    }

    /// Total insertion loss (dB) along the longest path through the mesh.
    ///
    /// The depth equals `n`, so the total loss is `n * loss_per_mzi_db`.
    pub fn total_insertion_loss_db(&self, loss_per_mzi_db: f64) -> f64 {
        (self.n as f64) * loss_per_mzi_db
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reck (triangular) architecture
// ─────────────────────────────────────────────────────────────────────────────

/// N×N triangular MZI mesh using the Reck decomposition.
///
/// The Reck architecture has depth 2n-3 and n(n-1)/2 MZIs arranged in a
/// lower-triangular pattern.
pub struct ReckArch {
    /// Port count.
    pub n: usize,
    /// `cells[i][j]` covers the (i,j) position in the lower triangle.
    /// Outer index is the diagonal, inner index is the column.
    pub cells: Vec<Vec<MziCell>>,
    /// Output phase shifts (length n).
    pub output_phases: Vec<f64>,
}

impl ReckArch {
    /// Construct an identity mesh.
    pub fn new(n: usize) -> Self {
        let mut cells = Vec::with_capacity(n - 1);
        for diag in 1..n {
            cells.push(vec![MziCell::new(0.0, 0.0); diag]);
        }
        Self {
            n,
            cells,
            output_phases: vec![0.0; n],
        }
    }

    /// Compute the full N×N unitary matrix.
    pub fn to_unitary(&self) -> Vec<Vec<Complex64>> {
        let n = self.n;
        let zero = Complex64::new(0.0, 0.0);
        (0..n)
            .map(|j| {
                let mut e = vec![zero; n];
                e[j] = Complex64::new(1.0, 0.0);
                self.apply(&e)
            })
            .collect()
    }

    /// Decompose a unitary into the Reck triangular mesh.
    ///
    /// Implements the Reck et al. (PRL 1994) algorithm by successive nulling
    /// of elements using 2×2 rotations from the bottom-right corner.
    pub fn from_unitary(u: &[Vec<Complex64>]) -> Self {
        let n = u.len();
        assert!(n >= 2, "matrix must be at least 2×2");

        let mut arch = Self::new(n);
        let mut work: Vec<Vec<Complex64>> = u.to_vec();

        // Null elements column by column from right to left.
        for col in (0..n - 1).rev() {
            for r in (col + 1..n).rev() {
                // Null work[r][col] using rows (r-1, r)
                let a = work[r - 1][col];
                let b = work[r][col];
                let (theta, phi) = nulling_angles(a, b);
                let mzi = MziCell::new(theta, phi);
                left_multiply_rows(&mut work, n, r - 1, &mzi);
                // Store in the triangular cell array
                let diag_idx = r - 1; // diagonal index (0 = first off-diagonal)
                let col_in_diag = col;
                if diag_idx < arch.cells.len() && col_in_diag < arch.cells[diag_idx].len() {
                    arch.cells[diag_idx][col_in_diag] = MziCell::new(theta, phi);
                }
            }
        }

        // Extract diagonal phases
        for (i, work_row) in work.iter().enumerate().take(n) {
            if work_row[i].norm() > 1e-10 {
                arch.output_phases[i] = work_row[i].arg();
            }
        }

        arch
    }

    /// Apply the Reck mesh to an input vector.
    pub fn apply(&self, input: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(input.len(), self.n);
        let mut v: Vec<Complex64> = input.to_vec();

        // Apply each diagonal layer
        for (diag_idx, diag_cells) in self.cells.iter().enumerate() {
            let r_base = diag_idx + 1; // bottom row of the pair
            for (col, mzi) in diag_cells.iter().enumerate() {
                let r = r_base - col; // row index (decreasing for each column)
                if r > 0 && r < self.n {
                    ClementsArch::apply_mzi_inplace(&mut v, r - 1, mzi);
                }
            }
        }

        // Apply output phases
        for (i, &phase) in self.output_phases.iter().enumerate() {
            v[i] *= Complex64::from_polar(1.0, phase);
        }

        v
    }

    /// Depth of the Reck mesh: 2n-3.
    pub fn depth(&self) -> usize {
        if self.n < 2 {
            0
        } else {
            2 * self.n - 3
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions for decomposition
// ─────────────────────────────────────────────────────────────────────────────

/// Compute (θ, φ) such that a 2×2 MZI T(θ,φ) zeros the second element
/// of [a; b] → [*, 0].
///
/// Strategy: θ = 2·atan2(|b|, |a|),  φ = arg(a) - arg(b) - π/2.
fn nulling_angles(a: Complex64, b: Complex64) -> (f64, f64) {
    let r_a = a.norm();
    let r_b = b.norm();
    let theta = 2.0 * r_b.atan2(r_a);
    let phi = if r_a < 1e-15 && r_b < 1e-15 {
        0.0
    } else {
        a.arg() - b.arg() + PI / 2.0
    };
    (theta, phi)
}

/// Compute (θ, φ) for right-multiplication nulling: zeros the second element
/// of [a, b] · T^†.
fn nulling_angles_right(a: Complex64, b: Complex64) -> (f64, f64) {
    let r_a = a.norm();
    let r_b = b.norm();
    let theta = 2.0 * r_b.atan2(r_a);
    let phi = if r_a < 1e-15 && r_b < 1e-15 {
        0.0
    } else {
        b.arg() - a.arg() + PI / 2.0
    };
    (theta, phi)
}

/// Left-multiply rows (r, r+1) of matrix `m` (size n×n) by T(theta,phi).
fn left_multiply_rows(m: &mut [Vec<Complex64>], n: usize, r: usize, mzi: &MziCell) {
    let t = mzi.transfer_matrix();
    let (rows_lo, rows_hi) = m.split_at_mut(r + 1);
    let row_r = &mut rows_lo[r];
    let row_r1 = &mut rows_hi[0];
    for (ar, br) in row_r.iter_mut().zip(row_r1.iter_mut()).take(n) {
        let a = *ar;
        let b = *br;
        *ar = t[0][0] * a + t[0][1] * b;
        *br = t[1][0] * a + t[1][1] * b;
    }
}

/// Right-multiply columns (c, c+1) of matrix `m` (size n×n) by T^†(theta,phi).
fn right_multiply_cols(m: &mut [Vec<Complex64>], n: usize, c: usize, mzi: &MziCell) {
    let t = mzi.transfer_matrix();
    // T^† = conj transpose
    for row_vec in m.iter_mut().take(n) {
        let a = row_vec[c];
        let b = row_vec[c + 1];
        row_vec[c] = t[0][0].conj() * a + t[1][0].conj() * b;
        row_vec[c + 1] = t[0][1].conj() * a + t[1][1].conj() * b;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_c(a: Complex64, b: Complex64, tol: f64) -> bool {
        (a - b).norm() < tol
    }

    fn mat_approx_eq(a: &[Vec<Complex64>], b: &[Vec<Complex64>], tol: f64) -> bool {
        a.iter().zip(b.iter()).all(|(ra, rb)| {
            ra.iter()
                .zip(rb.iter())
                .all(|(x, y)| approx_eq_c(*x, *y, tol))
        })
    }

    #[test]
    fn mzi_cell_identity() {
        // θ=π, φ=0 → pure cross state (T = i*σ_x)
        let mzi = MziCell::new(PI, 0.0);
        let t = mzi.transfer_matrix();
        // cos(π/2) = 0, sin(π/2) = 1
        assert!(approx_eq_c(t[0][0], Complex64::new(0.0, 0.0), 1e-12));
        assert!(approx_eq_c(t[0][1], Complex64::new(0.0, 1.0), 1e-12));
        assert!(approx_eq_c(t[1][0], Complex64::new(0.0, 1.0), 1e-12));
        assert!(approx_eq_c(t[1][1], Complex64::new(0.0, 0.0), 1e-12));
    }

    #[test]
    fn mzi_transmissivity_roundtrip() {
        let mut mzi = MziCell::new(0.0, 0.0);
        mzi.set_transmissivity(0.25);
        let t = mzi.transmissivity();
        assert!((t - 0.25).abs() < 1e-12, "got {t}");
    }

    #[test]
    fn clements_identity_apply() {
        let n = 4;
        let arch = ClementsArch::new(n);
        let input: Vec<Complex64> = (0..n).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let output = arch.apply(&input);
        for (i, (a, b)) in input.iter().zip(output.iter()).enumerate() {
            assert!(
                approx_eq_c(*a, *b, 1e-12),
                "mismatch at index {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn clements_to_unitary_is_unitary() {
        // A random-ish Clements mesh should produce a unitary matrix.
        let n = 3;
        let mut arch = ClementsArch::new(n);
        // Program some phases
        arch.columns[0][0] = Some(MziCell::new(0.5, 0.3));
        arch.columns[1][1] = Some(MziCell::new(1.2, -0.7));
        arch.diag_phases = vec![0.1, -0.2, 0.4];

        let u = arch.to_unitary();
        // Check U†U = I
        for i in 0..n {
            for j in 0..n {
                let mut s = Complex64::new(0.0, 0.0);
                for u_row in u.iter().take(n) {
                    s += u_row[i].conj() * u_row[j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (s.re - expected).abs() < 1e-10 && s.im.abs() < 1e-10,
                    "U†U[{i}][{j}] = {s}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn clements_insertion_loss() {
        let arch = ClementsArch::new(4);
        let loss = arch.total_insertion_loss_db(0.5);
        assert!((loss - 2.0).abs() < 1e-12);
    }

    #[test]
    fn reck_identity_apply() {
        let n = 3;
        let arch = ReckArch::new(n);
        let input: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(i as f64 + 1.0, 0.0))
            .collect();
        let output = arch.apply(&input);
        for (i, (a, b)) in input.iter().zip(output.iter()).enumerate() {
            assert!(
                approx_eq_c(*a, *b, 1e-12),
                "Reck identity mismatch at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn reck_depth() {
        assert_eq!(ReckArch::new(4).depth(), 5);
        assert_eq!(ReckArch::new(2).depth(), 1);
    }

    #[test]
    fn clements_n_mzis() {
        let n = 5;
        let arch = ClementsArch::new(n);
        assert_eq!(arch.n_mzis(), n * (n - 1) / 2);
    }

    #[test]
    fn clements_phase_sensitivity() {
        let n = 3;
        let mut arch = ClementsArch::new(n);
        arch.columns[0][0] = Some(MziCell::new(0.8, 0.3));
        let s = arch.phase_sensitivity(0, 0);
        assert!(s > 0.0, "sensitivity should be positive, got {s}");
    }

    #[test]
    fn mat_approx_eq_helper() {
        let a = vec![vec![Complex64::new(1.0, 0.0)]];
        let b = vec![vec![Complex64::new(1.0, 0.0)]];
        assert!(mat_approx_eq(&a, &b, 1e-12));
    }
}
