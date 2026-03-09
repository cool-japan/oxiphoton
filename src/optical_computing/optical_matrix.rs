//! Optical matrix-vector multiplication implementations.
//!
//! Covers WDM-based multiply-accumulate (MAC) units, optical outer products,
//! and photonic systolic array architectures for dense linear algebra.

// ─────────────────────────────────────────────────────────────────────────────
// WDM-based MAC
// ─────────────────────────────────────────────────────────────────────────────

/// Wavelength-Division Multiplexing multiply-accumulate (MAC) unit.
///
/// Each WDM channel λ_k carries one vector element x_k; the ring-resonator
/// coupling coefficient sets the weight w_k. The photodetector sums all
/// weighted intensities on the bus.
///
/// Architecture: N channels, equally spaced by `channel_spacing_ghz`.
pub struct WdmMac {
    /// Number of WDM channels (= vector dimension).
    pub n_channels: usize,
    /// Channel spacing in GHz.
    pub channel_spacing_ghz: f64,
    /// Achievable weight range [w_min, w_max].
    pub weight_range: (f64, f64),
}

impl WdmMac {
    /// Create a new WDM MAC unit.
    pub fn new(n: usize, spacing_ghz: f64) -> Self {
        Self {
            n_channels: n,
            channel_spacing_ghz: spacing_ghz,
            weight_range: (0.0, 1.0),
        }
    }

    /// Compute the optical dot product result = Σ_k w_k · x_k.
    ///
    /// Weights are clamped to `weight_range` to respect the physical constraints
    /// of ring resonator coupling coefficients.
    pub fn dot_product(&self, weights: &[f64], inputs: &[f64]) -> f64 {
        assert_eq!(weights.len(), self.n_channels);
        assert_eq!(inputs.len(), self.n_channels);

        let (w_min, w_max) = self.weight_range;
        weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, x)| w.clamp(w_min, w_max) * x)
            .sum()
    }

    /// Compute the matrix-vector product y = W · x using one WDM bus per row.
    ///
    /// Each row of `weight_matrix` is broadcast on a separate bus waveguide.
    /// `weight_matrix` is (n_rows × n_channels).
    pub fn matrix_vector(&self, weight_matrix: &[Vec<f64>], input: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.n_channels);
        weight_matrix
            .iter()
            .map(|row| {
                assert_eq!(
                    row.len(),
                    self.n_channels,
                    "each weight row must have n_channels elements"
                );
                self.dot_product(row, input)
            })
            .collect()
    }

    /// Effective weight precision in bits, derived from the dynamic range.
    ///
    /// Precision = log₂(w_max / w_min) bits (floor, at least 0).
    /// If w_min ≤ 0, returns 0 (undefined / single-sided).
    pub fn weight_precision_bits(&self) -> f64 {
        let (w_min, w_max) = self.weight_range;
        if w_min <= 0.0 || w_max <= w_min {
            return 0.0;
        }
        (w_max / w_min).log2()
    }

    /// Total optical bandwidth occupied: n_channels × channel_spacing_ghz (GHz).
    pub fn total_bandwidth_ghz(&self) -> f64 {
        self.n_channels as f64 * self.channel_spacing_ghz
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Optical outer product
// ─────────────────────────────────────────────────────────────────────────────

/// Optical vector outer product C = a ⊗ b, implemented via a 2D array of
/// optical multipliers (e.g. MZI pixel pairs or ring weight banks).
pub struct OpticalOuterProduct {
    /// Number of rows (length of vector a).
    pub n_rows: usize,
    /// Number of columns (length of vector b).
    pub n_cols: usize,
}

impl OpticalOuterProduct {
    /// Create a new outer-product unit.
    pub fn new(n: usize, m: usize) -> Self {
        Self {
            n_rows: n,
            n_cols: m,
        }
    }

    /// Compute C_{ij} = a_i · b_j.
    pub fn compute(&self, a: &[f64], b: &[f64]) -> Vec<Vec<f64>> {
        assert_eq!(a.len(), self.n_rows);
        assert_eq!(b.len(), self.n_cols);
        a.iter()
            .map(|&ai| b.iter().map(|&bj| ai * bj).collect())
            .collect()
    }

    /// Rank-1 update: M ← M + α · (a ⊗ b).
    ///
    /// Adds α times the outer product of a and b to the existing matrix M.
    /// M must be (n_rows × n_cols).
    pub fn rank1_update(&self, matrix: &mut [Vec<f64>], a: &[f64], b: &[f64], alpha: f64) {
        assert_eq!(matrix.len(), self.n_rows);
        assert_eq!(a.len(), self.n_rows);
        assert_eq!(b.len(), self.n_cols);

        for (i, row) in matrix.iter_mut().enumerate() {
            assert_eq!(row.len(), self.n_cols);
            for (j, cell) in row.iter_mut().enumerate() {
                *cell += alpha * a[i] * b[j];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Optical systolic array
// ─────────────────────────────────────────────────────────────────────────────

/// Photonic systolic array for dense matrix-matrix multiplication.
///
/// Analogous to a Google TPU or Intel NNP systolic array, but implemented
/// with optical delay lines and photodetector integration. Each PE (processing
/// element) computes one partial product per clock cycle.
pub struct OpticalSystolicArray {
    /// Number of rows (output rows = rows of A).
    pub n_rows: usize,
    /// Number of columns (output cols = cols of B).
    pub n_cols: usize,
    /// Clock rate in GHz.
    pub clock_rate_ghz: f64,
}

impl OpticalSystolicArray {
    /// Create a new systolic array.
    pub fn new(n: usize, m: usize, clock_ghz: f64) -> Self {
        Self {
            n_rows: n,
            n_cols: m,
            clock_rate_ghz: clock_ghz,
        }
    }

    /// Compute C = A · B.
    ///
    /// A is (n_rows × k), B is (k × n_cols).
    pub fn matrix_multiply(&self, a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = a.len();
        assert_eq!(n, self.n_rows, "A rows must equal n_rows");
        if n == 0 {
            return Vec::new();
        }
        let k = a[0].len();
        assert_eq!(b.len(), k, "B rows must equal inner dimension k");
        let m = if b.is_empty() { 0 } else { b[0].len() };
        assert_eq!(m, self.n_cols, "B cols must equal n_cols");

        (0..n)
            .map(|i| {
                (0..m)
                    .map(|j| (0..k).map(|l| a[i][l] * b[l][j]).sum())
                    .collect()
            })
            .collect()
    }

    /// Peak throughput in TOPS (tera operations per second).
    ///
    /// T = 2 · n_rows · n_cols · clock_rate_ghz × 10⁻³ TOPS
    /// (factor 2 for multiply + accumulate).
    pub fn throughput_tops(&self) -> f64 {
        2.0 * self.n_rows as f64 * self.n_cols as f64 * self.clock_rate_ghz * 1e-3
    }

    /// Latency in nanoseconds for a matrix multiply with inner dimension k.
    ///
    /// The systolic fill latency is (n_rows + k - 1) clock cycles.
    pub fn latency_ns(&self, k: usize) -> f64 {
        if self.clock_rate_ghz <= 0.0 {
            return f64::INFINITY;
        }
        let clock_period_ns = 1.0 / self.clock_rate_ghz; // ns per cycle
        let n_cycles = self.n_rows + k - 1;
        n_cycles as f64 * clock_period_ns
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wdm_dot_product() {
        let mac = WdmMac::new(3, 100.0);
        let w = vec![0.5, 0.3, 0.2];
        let x = vec![2.0, 4.0, 6.0];
        let result = mac.dot_product(&w, &x);
        let expected = 0.5 * 2.0 + 0.3 * 4.0 + 0.2 * 6.0; // 1.0+1.2+1.2 = 3.4
        assert!((result - expected).abs() < 1e-12, "got {result}");
    }

    #[test]
    fn wdm_matrix_vector() {
        let mac = WdmMac::new(2, 100.0);
        let w = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let x = vec![3.0, 5.0];
        let y = mac.matrix_vector(&w, &x);
        assert!((y[0] - 3.0).abs() < 1e-12);
        assert!((y[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn wdm_bandwidth() {
        let mac = WdmMac::new(16, 100.0);
        let bw = mac.total_bandwidth_ghz();
        assert!((bw - 1600.0).abs() < 1e-9);
    }

    #[test]
    fn wdm_weight_precision() {
        let mut mac = WdmMac::new(4, 100.0);
        mac.weight_range = (0.001, 1.0);
        let bits = mac.weight_precision_bits();
        // log2(1000) ≈ 9.97
        assert!(bits > 9.0 && bits < 11.0, "got {bits}");
    }

    #[test]
    fn outer_product_correctness() {
        let op = OpticalOuterProduct::new(2, 3);
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0, 5.0];
        let c = op.compute(&a, &b);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].len(), 3);
        assert!((c[0][0] - 3.0).abs() < 1e-12);
        assert!((c[1][2] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn rank1_update() {
        let op = OpticalOuterProduct::new(2, 2);
        let mut m = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        op.rank1_update(&mut m, &a, &b, 1.0);
        // M = a⊗b = [[3,4],[6,8]]
        assert!((m[0][0] - 3.0).abs() < 1e-12);
        assert!((m[1][1] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn systolic_matrix_multiply() {
        let sa = OpticalSystolicArray::new(2, 2, 10.0);
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let c = sa.matrix_multiply(&a, &b);
        // C = A*B = [[19,22],[43,50]]
        assert!((c[0][0] - 19.0).abs() < 1e-12, "got {}", c[0][0]);
        assert!((c[0][1] - 22.0).abs() < 1e-12, "got {}", c[0][1]);
        assert!((c[1][0] - 43.0).abs() < 1e-12, "got {}", c[1][0]);
        assert!((c[1][1] - 50.0).abs() < 1e-12, "got {}", c[1][1]);
    }

    #[test]
    fn systolic_throughput() {
        let sa = OpticalSystolicArray::new(4, 4, 10.0);
        let tops = sa.throughput_tops();
        // 2 * 4 * 4 * 10 * 1e-3 = 0.32 TOPS
        assert!((tops - 0.32).abs() < 1e-9, "got {tops}");
    }

    #[test]
    fn systolic_latency() {
        let sa = OpticalSystolicArray::new(4, 4, 10.0);
        // k=4: (4 + 4 - 1) = 7 cycles × (1/10) ns = 0.7 ns
        let lat = sa.latency_ns(4);
        assert!((lat - 0.7).abs() < 1e-9, "got {lat}");
    }
}
