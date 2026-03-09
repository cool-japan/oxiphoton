//! General N-port S-matrix with Redheffer star product cascade.
//!
//! The S-matrix relates incoming wave amplitudes to outgoing wave amplitudes
//! at each port of a microwave / photonic network:
//!
//!   \[b\] = S \[a\]
//!
//! where `a` = incoming amplitudes, `b` = outgoing amplitudes.
//!
//! ## Redheffer Star Product
//! For two cascaded networks A and B (ports partitioned into left and right):
//!
//!   S_AB = A ★ B
//!
//! using the standard Redheffer formula:
//!   S11 = A11 + A12 (I - B11 A22)^{-1} B11 A21
//!   S12 = A12 (I - B11 A22)^{-1} B12
//!   S21 = B21 (I - A22 B11)^{-1} A21
//!   S22 = B22 + B21 (I - A22 B11)^{-1} A22 B12
//!
//! ## S ↔ T Matrix Conversion (2-port)
//! For a 2-port network (N=2):
//!   T = [T11 T12; T21 T22]
//!   S11 = T21/T11, S12 = det(T)/T11, S21 = 1/T11, S22 = -T12/T11

use num_complex::Complex64;

/// N-port scattering matrix.
#[derive(Debug, Clone)]
pub struct SMatrixN {
    /// Matrix size (number of ports).
    pub n_ports: usize,
    /// S-matrix entries, row-major: data[i*n_ports + j] = S_{ij}
    pub data: Vec<Complex64>,
}

impl SMatrixN {
    /// Create an identity S-matrix (all zeros, then S_{ii} = 0 means fully transparent).
    ///
    /// An N-port "through" (identity) means: signal out at port j equals signal in at port j.
    pub fn identity(n: usize) -> Self {
        let mut data = vec![Complex64::new(0.0, 0.0); n * n];
        for i in 0..n {
            data[i * n + i] = Complex64::new(1.0, 0.0);
        }
        Self { n_ports: n, data }
    }

    /// Create a zero S-matrix.
    pub fn zeros(n: usize) -> Self {
        Self {
            n_ports: n,
            data: vec![Complex64::new(0.0, 0.0); n * n],
        }
    }

    /// Get S_{row, col}.
    pub fn get(&self, row: usize, col: usize) -> Complex64 {
        self.data[row * self.n_ports + col]
    }

    /// Set S_{row, col}.
    pub fn set(&mut self, row: usize, col: usize, val: Complex64) {
        self.data[row * self.n_ports + col] = val;
    }

    /// 2-port S-matrix from parameters.
    ///
    /// Ports: 0 = left input/output, 1 = right input/output.
    pub fn two_port(s11: Complex64, s12: Complex64, s21: Complex64, s22: Complex64) -> Self {
        let mut m = Self::zeros(2);
        m.set(0, 0, s11);
        m.set(0, 1, s12);
        m.set(1, 0, s21);
        m.set(1, 1, s22);
        m
    }

    /// Symmetric lossless 2-port (S11=0, S12=S21=exp(iφ), S22=0).
    pub fn lossless_two_port(phase: f64) -> Self {
        let t = Complex64::from_polar(1.0, phase);
        Self::two_port(Complex64::new(0.0, 0.0), t, t, Complex64::new(0.0, 0.0))
    }

    /// Reflective 2-port: pure reflector with reflection coefficient `r`.
    pub fn reflector(r: Complex64) -> Self {
        let t = (Complex64::new(1.0, 0.0) - r * r.conj()).sqrt();
        Self::two_port(r, t, t, -r.conj())
    }

    /// Redheffer star product (cascade): `self ★ other`.
    ///
    /// Both matrices must have even port count (N = 2M), split as M left + M right ports.
    /// Ports 0..M-1 are "left" (input side), M..2M-1 are "right" (output side).
    ///
    /// # Panics
    /// Panics if n_ports is not even or if n_ports differ.
    pub fn star(&self, other: &SMatrixN) -> SMatrixN {
        assert_eq!(
            self.n_ports, other.n_ports,
            "S-matrices must have same port count for star product"
        );
        assert_eq!(
            self.n_ports % 2,
            0,
            "n_ports must be even for Redheffer star product"
        );
        let n = self.n_ports;
        let m = n / 2;

        // Extract blocks: A = self, B = other
        // A = [[A11, A12], [A21, A22]]  where each block is m×m
        let a11 = extract_block(&self.data, n, 0, 0, m, m);
        let a12 = extract_block(&self.data, n, 0, m, m, m);
        let a21 = extract_block(&self.data, n, m, 0, m, m);
        let a22 = extract_block(&self.data, n, m, m, m, m);

        let b11 = extract_block(&other.data, n, 0, 0, m, m);
        let b12 = extract_block(&other.data, n, 0, m, m, m);
        let b21 = extract_block(&other.data, n, m, 0, m, m);
        let b22 = extract_block(&other.data, n, m, m, m, m);

        // D1 = (I - B11 A22)^{-1},  D2 = (I - A22 B11)^{-1}
        let d1 = mat_inv_m(&mat_sub_m(&identity_m(m), &mat_mul_m(&b11, &a22, m), m), m);
        let d2 = mat_inv_m(&mat_sub_m(&identity_m(m), &mat_mul_m(&a22, &b11, m), m), m);

        // S11 = A11 + A12 D1 B11 A21
        let s11 = mat_add_m(
            &a11,
            &mat_mul_m(&mat_mul_m(&a12, &d1, m), &mat_mul_m(&b11, &a21, m), m),
            m,
        );
        // S12 = A12 D1 B12
        let s12 = mat_mul_m(&mat_mul_m(&a12, &d1, m), &b12, m);
        // S21 = B21 D2 A21
        let s21 = mat_mul_m(&mat_mul_m(&b21, &d2, m), &a21, m);
        // S22 = B22 + B21 D2 A22 B12
        let s22 = mat_add_m(
            &b22,
            &mat_mul_m(&mat_mul_m(&b21, &d2, m), &mat_mul_m(&a22, &b12, m), m),
            m,
        );

        assemble_blocks(s11, s12, s21, s22, m)
    }

    /// Convert 2-port S-matrix to transfer (T) matrix.
    ///
    /// T-matrix definition:
    ///   [b1, a1]^T = T [a2, b2]^T  (using Pozar convention)
    ///
    /// T11 = 1/S21, T12 = -S22/S21, T21 = S11/S21, T22 = S12 - S11*S22/S21
    ///
    /// Returns a 2x2 T-matrix as flat row-major [T11, T12, T21, T22].
    pub fn to_transfer_matrix(&self) -> Option<[Complex64; 4]> {
        if self.n_ports != 2 {
            return None;
        }
        let s11 = self.get(0, 0);
        let s12 = self.get(0, 1);
        let s21 = self.get(1, 0);
        let s22 = self.get(1, 1);

        if s21.norm() < 1e-20 {
            return None;
        }
        let one = Complex64::new(1.0, 0.0);
        let t11 = one / s21;
        let t12 = -s22 / s21;
        let t21 = s11 / s21;
        let t22 = s12 - s11 * s22 / s21;
        Some([t11, t12, t21, t22])
    }

    /// Convert a 2-port T-matrix (flat row-major) back to S-matrix.
    pub fn from_transfer_matrix(t: [Complex64; 4]) -> Self {
        let [t11, t12, t21, t22] = t;
        // S21 = 1/T11, S11 = T21/T11, S22 = -T12/T11, S12 = (T11 T22 - T12 T21)/T11
        let s21 = Complex64::new(1.0, 0.0) / t11;
        let s11 = t21 / t11;
        let s22 = -t12 / t11;
        let s12 = (t11 * t22 - t12 * t21) / t11;
        Self::two_port(s11, s12, s21, s22)
    }

    /// Reorder ports according to a permutation.
    ///
    /// `perm[i] = j` means new port `i` maps to old port `j`.
    pub fn reorder_ports(&self, perm: &[usize]) -> Self {
        let n = self.n_ports;
        assert_eq!(perm.len(), n, "Permutation length must match n_ports");
        let mut out = Self::zeros(n);
        for i in 0..n {
            for j in 0..n {
                out.set(i, j, self.get(perm[i], perm[j]));
            }
        }
        out
    }

    /// Power transmittance |S_{out_port, in_port}|²
    pub fn transmittance(&self, out_port: usize, in_port: usize) -> f64 {
        let s = self.get(out_port, in_port);
        s.norm_sqr()
    }

    /// Power reflectance |S_{in_port, in_port}|²
    pub fn reflectance(&self, in_port: usize) -> f64 {
        self.transmittance(in_port, in_port)
    }

    /// Check if S-matrix is unitary to within `tol` (power conservation).
    pub fn is_unitary(&self, tol: f64) -> bool {
        let n = self.n_ports;
        // Check S† S ≈ I
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    sum += self.get(k, i).conj() * self.get(k, j);
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                if (sum - expected).norm() > tol {
                    return false;
                }
            }
        }
        true
    }
}

// ── Matrix helper functions (dense, m×m, row-major) ──────────────────────────

fn identity_m(m: usize) -> Vec<Complex64> {
    let mut v = vec![Complex64::new(0.0, 0.0); m * m];
    for i in 0..m {
        v[i * m + i] = Complex64::new(1.0, 0.0);
    }
    v
}

fn mat_mul_m(a: &[Complex64], b: &[Complex64], m: usize) -> Vec<Complex64> {
    let mut c = vec![Complex64::new(0.0, 0.0); m * m];
    for i in 0..m {
        for k in 0..m {
            for j in 0..m {
                c[i * m + j] += a[i * m + k] * b[k * m + j];
            }
        }
    }
    c
}

fn mat_add_m(a: &[Complex64], b: &[Complex64], m: usize) -> Vec<Complex64> {
    (0..m * m).map(|i| a[i] + b[i]).collect()
}

fn mat_sub_m(a: &[Complex64], b: &[Complex64], m: usize) -> Vec<Complex64> {
    (0..m * m).map(|i| a[i] - b[i]).collect()
}

/// Simple Gauss-Jordan inversion for small m×m complex matrix.
fn mat_inv_m(a: &[Complex64], m: usize) -> Vec<Complex64> {
    let mut aug: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); m * 2 * m];
    // Copy a into left half, identity into right half
    for i in 0..m {
        for j in 0..m {
            aug[i * 2 * m + j] = a[i * m + j];
        }
        aug[i * 2 * m + m + i] = Complex64::new(1.0, 0.0);
    }
    for col in 0..m {
        // Partial pivot
        let mut max_row = col;
        let mut max_val = aug[col * 2 * m + col].norm();
        for row in col + 1..m {
            let v = aug[row * 2 * m + col].norm();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_row != col {
            for j in 0..2 * m {
                aug.swap(col * 2 * m + j, max_row * 2 * m + j);
            }
        }
        let piv = aug[col * 2 * m + col];
        if piv.norm() < 1e-30 {
            // Singular → return identity (fallback)
            return identity_m(m);
        }
        for j in 0..2 * m {
            aug[col * 2 * m + j] /= piv;
        }
        for row in 0..m {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * m + col];
            for j in 0..2 * m {
                let sub = factor * aug[col * 2 * m + j];
                aug[row * 2 * m + j] -= sub;
            }
        }
    }
    // Extract right half
    let mut inv = vec![Complex64::new(0.0, 0.0); m * m];
    for i in 0..m {
        for j in 0..m {
            inv[i * m + j] = aug[i * 2 * m + m + j];
        }
    }
    inv
}

fn extract_block(
    data: &[Complex64],
    n: usize,
    r0: usize,
    c0: usize,
    rows: usize,
    cols: usize,
) -> Vec<Complex64> {
    let mut blk = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            blk.push(data[(r0 + i) * n + (c0 + j)]);
        }
    }
    blk
}

fn assemble_blocks(
    s11: Vec<Complex64>,
    s12: Vec<Complex64>,
    s21: Vec<Complex64>,
    s22: Vec<Complex64>,
    m: usize,
) -> SMatrixN {
    let n = 2 * m;
    let mut out = SMatrixN::zeros(n);
    for i in 0..m {
        for j in 0..m {
            out.set(i, j, s11[i * m + j]);
            out.set(i, j + m, s12[i * m + j]);
            out.set(i + m, j, s21[i * m + j]);
            out.set(i + m, j + m, s22[i * m + j]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    #[test]
    fn smatrix_identity_is_identity() {
        let s = SMatrixN::identity(4);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(s.get(i, j).re, expected, epsilon = 1e-12);
                assert_relative_eq!(s.get(i, j).im, 0.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn smatrix_two_port_accessors() {
        let s = SMatrixN::two_port(c(0.1, 0.0), c(0.9, 0.0), c(0.9, 0.0), c(0.05, 0.0));
        assert_relative_eq!(s.get(0, 0).re, 0.1, epsilon = 1e-12);
        assert_relative_eq!(s.get(0, 1).re, 0.9, epsilon = 1e-12);
        assert_relative_eq!(s.get(1, 0).re, 0.9, epsilon = 1e-12);
    }

    #[test]
    fn lossless_two_port_transmittance_unity() {
        let s = SMatrixN::lossless_two_port(0.3);
        assert_relative_eq!(s.transmittance(0, 1), 1.0, epsilon = 1e-12);
        assert_relative_eq!(s.transmittance(1, 0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn star_product_of_two_through_is_through() {
        // Through: S = [[0, 1], [1, 0]]
        let s = SMatrixN::two_port(c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0));
        let result = s.star(&s);
        // S12 of cascade should still be 1 (two lossless throughs)
        assert_relative_eq!(result.get(0, 1).norm(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(1, 0).norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn star_product_with_phase() {
        let s = SMatrixN::lossless_two_port(0.5);
        let result = s.star(&s);
        // Phase accumulated twice
        assert_relative_eq!(result.get(0, 1).norm(), 1.0, epsilon = 1e-10);
        let phase = result.get(0, 1).arg();
        assert_relative_eq!(phase, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn s_to_t_matrix_roundtrip() {
        let s = SMatrixN::two_port(c(0.2, 0.1), c(0.8, 0.0), c(0.8, 0.0), c(0.1, 0.05));
        let t = s.to_transfer_matrix().unwrap();
        let s_back = SMatrixN::from_transfer_matrix(t);
        for i in 0..2 {
            for j in 0..2 {
                let diff = (s.get(i, j) - s_back.get(i, j)).norm();
                assert!(diff < 1e-10, "S[{i},{j}] roundtrip error={diff:.2e}");
            }
        }
    }

    #[test]
    fn reorder_ports_swap() {
        let s = SMatrixN::two_port(c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0));
        let perm = [1, 0];
        let r = s.reorder_ports(&perm);
        // After swap: new S[0,0] = old S[1,1] = 4
        assert_relative_eq!(r.get(0, 0).re, 4.0, epsilon = 1e-12);
        assert_relative_eq!(r.get(1, 1).re, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn lossless_two_port_is_unitary() {
        let s = SMatrixN::lossless_two_port(0.7);
        assert!(s.is_unitary(1e-10));
    }

    #[test]
    fn mat_inv_identity_check() {
        let id = identity_m(3);
        let inv = mat_inv_m(&id, 3);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(inv[i * 3 + j].re, expected, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn transmittance_and_reflectance() {
        let s = SMatrixN::lossless_two_port(0.0);
        assert_relative_eq!(s.transmittance(0, 1), 1.0, epsilon = 1e-12);
        assert_relative_eq!(s.reflectance(0), 0.0, epsilon = 1e-12);
    }
}
