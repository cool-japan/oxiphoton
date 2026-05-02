//! EME interface S-matrix from mode overlap integrals.
//!
//! Implements the classical mode-matching method for computing the scattering
//! matrix at an interface between two waveguide cross-sections with different
//! sets of guided modes.
//!
//! ## Reference
//! Snyder & Love, "Optical Waveguide Theory", Ch. 31;
//! Bienstman, PhD thesis (Ghent 2001), §2.2.

use num_complex::Complex64;
use std::f64::consts::PI;

use super::eigenmode::{cascade_smatrices, EigenmodeLayer, EmeMode, SMatrixBlocks};

// ── Error type ────────────────────────────────────────────────────────────────

/// Error type for interface S-matrix computation.
#[derive(Debug, Clone)]
pub enum InterfaceError {
    /// Matrix inversion failed due to a near-singular pivot.
    Singular { row: usize, pivot_norm: f64 },
    /// No modes were provided on one or both sides.
    NoModes,
    /// Field lengths mismatch between mode sets.
    GridMismatch { len_a: usize, len_b: usize },
}

impl std::fmt::Display for InterfaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InterfaceError::Singular { row, pivot_norm } => {
                write!(
                    f,
                    "singular matrix at pivot row {row} (|pivot|={pivot_norm:.3e})"
                )
            }
            InterfaceError::NoModes => write!(f, "no modes provided for interface S-matrix"),
            InterfaceError::GridMismatch { len_a, len_b } => {
                write!(
                    f,
                    "grid length mismatch: modes_a has {len_a} points, modes_b has {len_b}"
                )
            }
        }
    }
}

impl std::error::Error for InterfaceError {}

// ── Matrix helpers (Vec<Vec<Complex64>>) ─────────────────────────────────────

/// Full Gauss-Jordan matrix inverse with partial pivoting.
/// On a near-singular pivot (|piv| < 1e-30), returns Err(InterfaceError::Singular).
pub(crate) fn mat_inv_full_nd(
    m: Vec<Vec<Complex64>>,
) -> Result<Vec<Vec<Complex64>>, InterfaceError> {
    let n = m.len();
    // Augment [m | I]
    let mut aug: Vec<Vec<Complex64>> = (0..n)
        .map(|i| {
            let mut row = m[i].clone();
            row.resize(2 * n, Complex64::new(0.0, 0.0));
            row[n + i] = Complex64::new(1.0, 0.0);
            row
        })
        .collect();

    for col in 0..n {
        // Partial pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].norm();
        for (row, aug_row) in aug.iter().enumerate().take(n).skip(col + 1) {
            let v = aug_row[col].norm();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_row != col {
            aug.swap(col, max_row);
        }

        let piv = aug[col][col];
        if piv.norm() < 1e-30 {
            return Err(InterfaceError::Singular {
                row: col,
                pivot_norm: piv.norm(),
            });
        }

        // Normalise pivot row
        for elem in aug[col].iter_mut() {
            *elem /= piv;
        }

        // Eliminate column in all other rows (need index, so clone pivot row first)
        let pivot_row: Vec<Complex64> = aug[col].clone();
        for (row, aug_row) in aug.iter_mut().enumerate().take(n) {
            if row == col {
                continue;
            }
            let factor = aug_row[col];
            for (a_elem, p_elem) in aug_row.iter_mut().zip(pivot_row.iter()) {
                *a_elem -= factor * p_elem;
            }
        }
    }

    // Extract the right half (the inverse)
    let inv: Vec<Vec<Complex64>> = aug.into_iter().map(|row| row[n..].to_vec()).collect();
    Ok(inv)
}

/// Multiply two n×m and m×p matrices (supports non-square).
fn mat_mul_cc(a: &[Vec<Complex64>], b: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let rows_a = a.len();
    if rows_a == 0 || b.is_empty() {
        return vec![];
    }
    let cols_b = b[0].len();
    let inner = b.len();
    let mut c = vec![vec![Complex64::new(0.0, 0.0); cols_b]; rows_a];
    for i in 0..rows_a {
        for k in 0..inner {
            let aik = a[i][k];
            for j in 0..cols_b {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

fn mat_add_cc(a: &[Vec<Complex64>], b: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let n = a.len();
    (0..n)
        .map(|i| (0..a[i].len()).map(|j| a[i][j] + b[i][j]).collect())
        .collect()
}

fn mat_sub_cc(a: &[Vec<Complex64>], b: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let n = a.len();
    (0..n)
        .map(|i| (0..a[i].len()).map(|j| a[i][j] - b[i][j]).collect())
        .collect()
}

fn identity_cc(n: usize) -> Vec<Vec<Complex64>> {
    let mut m = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for (i, row) in m.iter_mut().enumerate().take(n) {
        row[i] = Complex64::new(1.0, 0.0);
    }
    m
}

fn scalar_mul_cc(s: Complex64, m: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    m.iter()
        .map(|row| row.iter().map(|&v| s * v).collect())
        .collect()
}

fn transpose_cc(m: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    if m.is_empty() {
        return vec![];
    }
    let rows = m.len();
    let cols = m[0].len();
    let mut t = vec![vec![Complex64::new(0.0, 0.0); rows]; cols];
    for (i, row) in m.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            t[j][i] = v;
        }
    }
    t
}

// ── Real-field overlap ────────────────────────────────────────────────────────

/// Raw E·E overlap integral ∫ E_a(x) · E_b(x) dx (trapezoidal rule).
/// Both field arrays must have the same length.
fn overlap_real(field_a: &[f64], field_b: &[f64], dx: f64) -> f64 {
    let n = field_a.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return field_a[0] * field_b[0] * dx;
    }
    let ends = field_a[0] * field_b[0] + field_a[n - 1] * field_b[n - 1];
    let middle: f64 = field_a[1..n - 1]
        .iter()
        .zip(field_b[1..n - 1].iter())
        .map(|(&a, &b)| 2.0 * a * b)
        .sum();
    (ends + middle) * (dx / 2.0)
}

/// Power normalisation factor for a TE-slab mode: ωμ₀/(β) · ∫|E_y|² dx.
///
/// The reciprocal is used to scale fields so that V[i][i] = 1 when modes_a == modes_b.
/// Each mode carries unit power when its field is divided by sqrt(power_norm).
fn power_norm_factor(mode: &EmeMode, omega: f64) -> f64 {
    const MU_0: f64 = 4.0 * PI * 1e-7;
    let integral = overlap_real(&mode.field, &mode.field, mode.dx);
    // Poynting flux: (β / 2·ω·μ₀) · ∫|E_y|² dx
    mode.beta / (2.0 * omega * MU_0) * integral
}

// ── Core interface S-matrix ───────────────────────────────────────────────────

/// Compute the interface S-matrix between two sets of TE-slab modes.
///
/// Uses the classical mode-matching method (Bienstman/Snyder-Love).
/// The returned tuple `(S11, S12, S21, S22)` follows the same `SMatrixBlocks`
/// convention as `EigenmodeLayer::to_s_matrix_full()`:
///   - S12, S21 are na×nb and nb×na respectively
///   - S11 is na×na (reflection on the left)
///   - S22 is nb×nb (reflection on the right)
///
/// Both mode sets must have fields sampled on a grid with the same number of
/// points and the same grid spacing `dx`.
///
/// `omega` is the angular frequency (rad/s), used for power normalisation.
pub fn interface_smatrix(
    modes_a: &[EmeMode],
    modes_b: &[EmeMode],
    omega: f64,
) -> Result<SMatrixBlocks, InterfaceError> {
    if modes_a.is_empty() || modes_b.is_empty() {
        return Err(InterfaceError::NoModes);
    }

    let dx_a = modes_a[0].dx;
    let len_a = modes_a[0].field.len();
    let len_b = modes_b[0].field.len();
    if len_a != len_b {
        return Err(InterfaceError::GridMismatch { len_a, len_b });
    }

    let na = modes_a.len();
    let nb = modes_b.len();

    // Compute power normalisation for each mode: p_a[i] = (β/(2ωμ₀)) ∫|E_i|² dx
    let p_a: Vec<f64> = modes_a
        .iter()
        .map(|m| power_norm_factor(m, omega))
        .collect();
    let p_b: Vec<f64> = modes_b
        .iter()
        .map(|m| power_norm_factor(m, omega))
        .collect();

    // Build the power-normalised overlap matrix V[i][j]:
    //
    //   V[i][j] = (β_a_i + β_b_j)/(4ωμ₀) · c[i][j] / sqrt(p_a_i · p_b_j)
    //
    // where c[i][j] = ∫ E_ai · E_bj dx  and
    //       p_a_i = (β_a_i/(2ωμ₀)) ∫|E_ai|² dx  (Poynting power).
    //
    // For same modes (i=j, modes_a == modes_b):
    //   V[i][i] = (2β_i/(4ωμ₀)) · N_i / sqrt(p_i²)
    //           = (β_i/(2ωμ₀) · N_i) / p_i
    //           = p_i / p_i = 1  ✓
    // so V = I when modes_a == modes_b, giving S11 = 0 and S12 = I (transparent interface).
    const MU_0: f64 = 4.0 * PI * 1e-7;

    let mut v: Vec<Vec<Complex64>> = vec![vec![Complex64::new(0.0, 0.0); nb]; na];
    for (i, m_a) in modes_a.iter().enumerate() {
        for (j, m_b) in modes_b.iter().enumerate() {
            let c_ij = overlap_real(&m_a.field, &m_b.field, dx_a);
            let denom = (p_a[i] * p_b[j]).sqrt();
            if denom < 1e-60 {
                v[i][j] = Complex64::new(0.0, 0.0);
            } else {
                let factor = (m_a.beta + m_b.beta) / (4.0 * omega * MU_0);
                v[i][j] = Complex64::new(factor * c_ij / denom, 0.0);
            }
        }
    }

    let vt = transpose_cc(&v); // nb × na
    let vvt = mat_mul_cc(&v, &vt); // na × na
    let vtv = mat_mul_cc(&vt, &v); // nb × nb
    let i_a = identity_cc(na);
    let i_b = identity_cc(nb);

    // Interface S-matrix (mode-matching formula):
    //   S11 = (V·Vᵀ + I_a)⁻¹ · (V·Vᵀ − I_a)   [na × na]
    //   S12 = 2·(V·Vᵀ + I_a)⁻¹ · V             [na × nb]
    //   S21 = 2·(Vᵀ·V + I_b)⁻¹ · Vᵀ            [nb × na]
    //   S22 = −(Vᵀ·V + I_b)⁻¹ · (Vᵀ·V − I_b)  [nb × nb]
    let m_pp_a = mat_add_cc(&vvt, &i_a);
    let m_mm_a = mat_sub_cc(&vvt, &i_a);
    let m_pp_b = mat_add_cc(&vtv, &i_b);
    let m_mm_b = mat_sub_cc(&vtv, &i_b);

    // Fall back to identity inverse on singular matrices (consistent with mat_inv_m in smatrix.rs)
    let inv_pp_a = mat_inv_full_nd(m_pp_a).unwrap_or_else(|_| identity_cc(na));
    let inv_pp_b = mat_inv_full_nd(m_pp_b).unwrap_or_else(|_| identity_cc(nb));

    let s11 = mat_mul_cc(&inv_pp_a, &m_mm_a);
    let two = Complex64::new(2.0, 0.0);
    let s12 = scalar_mul_cc(two, &mat_mul_cc(&inv_pp_a, &v));
    let s21 = scalar_mul_cc(two, &mat_mul_cc(&inv_pp_b, &vt));
    let s22 = scalar_mul_cc(Complex64::new(-1.0, 0.0), &mat_mul_cc(&inv_pp_b, &m_mm_b));

    Ok((s11, s12, s21, s22))
}

// ── EmeStack ──────────────────────────────────────────────────────────────────

/// A sequence of `EigenmodeLayer` sections for cascaded EME simulation.
///
/// Computes the end-to-end S-matrix by cascading each layer's propagation
/// S-matrix with mode-matching interface S-matrices at layer boundaries.
pub struct EmeStack {
    pub layers: Vec<EigenmodeLayer>,
}

impl EmeStack {
    pub fn new(layers: Vec<EigenmodeLayer>) -> Self {
        Self { layers }
    }

    /// Cascade all layers with interface S-matrices between adjacent sections.
    ///
    /// Returns the total `SMatrixBlocks` in the same format as
    /// `EigenmodeLayer::to_s_matrix_full()`.  Returns `Err` if any layer has
    /// no guided modes.
    pub fn to_s_matrix_full(&self, omega: f64) -> Result<SMatrixBlocks, InterfaceError> {
        if self.layers.is_empty() {
            return Err(InterfaceError::NoModes);
        }

        // Compute S-matrix and guided modes for each layer up-front.
        let layer_data: Vec<_> = self
            .layers
            .iter()
            .map(|layer| {
                use super::eigenmode::EmeSegment;
                let seg =
                    EmeSegment::new(layer.thickness, layer.n_core, layer.n_clad, layer.thickness);
                let modes = seg.find_modes(layer.wavelength, layer.n_modes, layer.n_pts);
                let s = layer.to_s_matrix_full();
                (s, modes)
            })
            .collect();

        // Start with the first layer's S-matrix.
        let (mut total, _) = layer_data
            .first()
            .map(|(s, m)| (s.clone(), m.clone()))
            .ok_or(InterfaceError::NoModes)?;

        for i in 1..layer_data.len() {
            let (_, modes_a) = &layer_data[i - 1];
            let (s_b, modes_b) = &layer_data[i];

            if modes_a.is_empty() || modes_b.is_empty() {
                return Err(InterfaceError::NoModes);
            }

            // Interface S-matrix between layer (i-1) and layer i.
            let s_iface = interface_smatrix(modes_a, modes_b, omega)?;

            // Cascade: total ⋆ interface ⋆ layer_i
            let after_iface = cascade_blocks(&total, &s_iface);
            total = cascade_blocks(&after_iface, s_b);
        }

        Ok(total)
    }
}

/// Cascade two `SMatrixBlocks` via Redheffer star product.
///
/// When both blocks have the same square dimension, uses the existing
/// `cascade_smatrices` fast path. Otherwise falls back to the general
/// rectangular Redheffer formula.
fn cascade_blocks(a: &SMatrixBlocks, b: &SMatrixBlocks) -> SMatrixBlocks {
    let na = a.0.len();
    let nb = b.0.len();

    if na == nb {
        let (s11_a, s12_a, s21_a, s22_a) = a;
        let (s11_b, s12_b, s21_b, s22_b) = b;
        cascade_smatrices(s11_a, s12_a, s21_a, s22_a, s11_b, s12_b, s21_b, s22_b)
    } else {
        rectangular_redheffer(a, b)
    }
}

/// General Redheffer star product for potentially rectangular S-matrices.
///
/// Dimensions:
///   a: na_in × na_in S11,  na_in × na_out S12,  na_out × na_in S21,  na_out × na_out S22
///   b: na_out × na_out S11, na_out × nb_out S12, nb_out × na_out S21, nb_out × nb_out S22
///
/// Result:
///   S11: na_in × na_in,  S12: na_in × nb_out,
///   S21: nb_out × na_in, S22: nb_out × nb_out
fn rectangular_redheffer(a: &SMatrixBlocks, b: &SMatrixBlocks) -> SMatrixBlocks {
    let (s11_a, s12_a, s21_a, s22_a) = a;
    let (s11_b, s12_b, s21_b, s22_b) = b;

    // mid dimension (shared between A's output and B's input)
    let mid = s22_a.len();
    let i_mid = identity_cc(mid);

    // D1 = (I_mid − S22_A · S11_B)⁻¹   [mid × mid]
    let s22a_s11b = mat_mul_cc(s22_a, s11_b);
    let id_minus1 = mat_sub_cc(&i_mid, &s22a_s11b);
    let d1 = mat_inv_full_nd(id_minus1).unwrap_or_else(|_| identity_cc(mid));

    // D2 = (I_mid − S11_B · S22_A)⁻¹   [mid × mid]
    let s11b_s22a = mat_mul_cc(s11_b, s22_a);
    let id_minus2 = mat_sub_cc(&i_mid, &s11b_s22a);
    let d2 = mat_inv_full_nd(id_minus2).unwrap_or_else(|_| identity_cc(mid));

    // new S11 = S11_A + S12_A · D1 · S11_B · S21_A
    let new_s11 = mat_add_cc(
        s11_a,
        &mat_mul_cc(&mat_mul_cc(&mat_mul_cc(s12_a, &d1), s11_b), s21_a),
    );
    // new S22 = S22_B + S21_B · D1 · S22_A · S12_B
    // Standard Redheffer: new_S22 uses D1 (not D2).  Reference: eq (6) in
    // Redheffer (1961); D1 = (I - S22_A S11_B)^{-1}.
    let new_s22 = mat_add_cc(
        s22_b,
        &mat_mul_cc(&mat_mul_cc(&mat_mul_cc(s21_b, &d1), s22_a), s12_b),
    );
    // new S21 = S21_B · D1 · S21_A
    let new_s21 = mat_mul_cc(&mat_mul_cc(s21_b, &d1), s21_a);
    // new S12 = S12_A · D2 · S12_B
    let new_s12 = mat_mul_cc(&mat_mul_cc(s12_a, &d2), s12_b);

    (new_s11, new_s12, new_s21, new_s22)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: make a simple EmeMode with uniform field of value 1/√n
    fn make_uniform_mode(beta: f64, n_pts: usize, dx: f64) -> EmeMode {
        EmeMode {
            n_eff: beta / (2.0 * PI / 1550e-9),
            beta,
            field: vec![1.0 / (n_pts as f64).sqrt(); n_pts],
            dx,
        }
    }

    #[test]
    fn mat_inv_full_nd_identity() {
        let id: Vec<Vec<Complex64>> = vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        let inv = mat_inv_full_nd(id).expect("identity is invertible");
        assert!((inv[0][0] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        assert!(inv[0][1].norm() < 1e-12);
        assert!(inv[1][0].norm() < 1e-12);
        assert!((inv[1][1] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn mat_inv_full_nd_2x2() {
        // Inverse of [[2,1],[1,3]]: det=5, inv=[[3,-1],[-1,2]]/5
        let m: Vec<Vec<Complex64>> = vec![
            vec![Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0)],
            vec![Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];
        let inv = mat_inv_full_nd(m).expect("2x2 is invertible");
        assert!((inv[0][0] - Complex64::new(3.0 / 5.0, 0.0)).norm() < 1e-12);
        assert!((inv[0][1] - Complex64::new(-1.0 / 5.0, 0.0)).norm() < 1e-12);
        assert!((inv[1][0] - Complex64::new(-1.0 / 5.0, 0.0)).norm() < 1e-12);
        assert!((inv[1][1] - Complex64::new(2.0 / 5.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn transpose_cc_correctness() {
        let m = vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            vec![Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)],
            vec![Complex64::new(5.0, 0.0), Complex64::new(6.0, 0.0)],
        ];
        let t = transpose_cc(&m);
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].len(), 3);
        assert!((t[0][2] - Complex64::new(5.0, 0.0)).norm() < 1e-12);
        assert!((t[1][0] - Complex64::new(2.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn interface_smatrix_no_modes_error() {
        let modes: Vec<EmeMode> = vec![];
        let other = vec![make_uniform_mode(1e7, 10, 1e-8)];
        let r = interface_smatrix(&modes, &other, 1.2e15);
        assert!(matches!(r, Err(InterfaceError::NoModes)));
    }

    #[test]
    fn overlap_real_self_equals_norm() {
        let f = vec![1.0, 2.0, 3.0, 4.0];
        let dx = 0.1;
        let ov = overlap_real(&f, &f, dx);
        // Trapezoidal: (1+16)/2 + 4 + 9 = 8.5 + 13 = 21.5, times dx=0.1 → 2.15? Let's check
        // (f[0]*f[0] + f[3]*f[3])/2 + f[1]*f[1] + f[2]*f[2]) * dx
        // = (1+16)/2 + 4 + 9) * 0.1 = (8.5 + 13)*0.1 = 2.15
        assert!((ov - 2.15).abs() < 1e-12);
    }
}
