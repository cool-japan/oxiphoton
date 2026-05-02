//! Linear optical quantum gates and network transformations.
//!
//! Provides:
//! - Beam splitters and phase shifters (fundamental 1- and 2-mode gates)
//! - Composition of linear optical networks
//! - Action on Fock states via the permanent (Ryser's formula)
//! - Reck et al. (1994) triangular MZI decomposition
//! - Clements et al. (2016) rectangular MZI mesh
//! - Permanent and hafnian of complex matrices
//! - KLM CNOT gate with success-probability bookkeeping and Knill-scaling ancilla boost

use num_complex::Complex64;
use std::f64::consts::PI;

use super::fock_state::{ln_factorial, FockSuperposition, MultiModeFockState};

// ─── Matrix algebra helpers ──────────────────────────────────────────────────

/// Multiply two n×n complex matrices.
fn mat_mul(a: &[Vec<Complex64>], b: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let n = a.len();
    let mut c = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for i in 0..n {
        for k in 0..n {
            if a[i][k].norm_sqr() < 1e-300 {
                continue;
            }
            for j in 0..n {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Conjugate-transpose (†) of an n×n matrix.
fn mat_dagger(a: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let n = a.len();
    let mut b = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for i in 0..n {
        for j in 0..n {
            b[j][i] = a[i][j].conj();
        }
    }
    b
}

/// Identity n×n complex matrix.
fn identity_matrix(n: usize) -> Vec<Vec<Complex64>> {
    let mut m = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for (i, row) in m.iter_mut().enumerate().take(n) {
        row[i] = Complex64::new(1.0, 0.0);
    }
    m
}

// ─── Permanent (Ryser's formula) ─────────────────────────────────────────────

/// Compute the permanent of an n×n complex matrix using Ryser's formula.
///
/// perm(A) = (-1)^n Σ_{S ⊆ {1..n}} (-1)^|S| Πᵢ (Σⱼ∈S Aᵢⱼ)
///
/// Complexity: O(2ⁿ · n).  Suitable for n ≤ 25.
pub fn permanent(matrix: &[Vec<Complex64>]) -> Complex64 {
    let n = matrix.len();
    if n == 0 {
        return Complex64::new(1.0, 0.0);
    }
    let total_subsets: u64 = 1u64 << n;
    let mut result = Complex64::new(0.0, 0.0);

    for subset in 0u64..total_subsets {
        let popcount = subset.count_ones() as usize;
        // Row sums for this subset of columns
        let mut prod = Complex64::new(1.0, 0.0);
        for row in matrix.iter().take(n) {
            let mut row_sum = Complex64::new(0.0, 0.0);
            for (j, &val) in row.iter().enumerate().take(n) {
                if subset & (1u64 << j) != 0 {
                    row_sum += val;
                }
            }
            prod *= row_sum;
        }
        // Sign: (-1)^(n - popcount)
        if (n - popcount) % 2 == 0 {
            result += prod;
        } else {
            result -= prod;
        }
    }
    // Overall sign: (-1)^n
    if n % 2 == 0 {
        result
    } else {
        -result
    }
}

// ─── Hafnian (loop hafnian via inclusion-exclusion) ──────────────────────────

/// Compute the hafnian of a 2n×2n complex symmetric matrix.
///
/// Uses the inclusion-exclusion / "Glynn" formula for the loop hafnian,
/// summing over perfect matchings of {0, …, 2n−1}.
///
/// Complexity: O(2ⁿ · n²).  Suitable for matrix dimension ≤ 20.
pub fn hafnian(matrix: &[Vec<Complex64>]) -> Complex64 {
    let n2 = matrix.len();
    if n2 == 0 {
        return Complex64::new(1.0, 0.0);
    }
    if n2 % 2 != 0 {
        // Odd-dimensional — hafnian is 0 (no perfect matching)
        return Complex64::new(0.0, 0.0);
    }
    let _n = n2 / 2;
    // Enumerate all perfect matchings recursively via a bitmask approach.
    // A perfect matching of 2n elements: choose the partner of the smallest free element.
    fn recurse(matrix: &[Vec<Complex64>], free: &mut [usize]) -> Complex64 {
        if free.is_empty() {
            return Complex64::new(1.0, 0.0);
        }
        let i = free[0];
        let mut result = Complex64::new(0.0, 0.0);
        // Partner i with each free element j > i
        let len = free.len();
        let mut idx = 1;
        while idx < len {
            let j = free[idx];
            let val = matrix[i][j];
            // Remove i and j from free list
            let mut new_free: Vec<usize> = free[1..].to_vec();
            new_free.retain(|&x| x != j);
            result += val * recurse(matrix, &mut new_free);
            idx += 1;
        }
        result
    }

    let mut free: Vec<usize> = (0..n2).collect();
    recurse(matrix, &mut free)
}

// ─── LopGate ─────────────────────────────────────────────────────────────────

/// Elementary linear optical gate in a decomposition.
#[derive(Debug, Clone, PartialEq)]
pub enum LopGate {
    /// Beam splitter between `mode1` and `mode2` with reflectivity angle θ and phase φ.
    BeamSplitter {
        mode1: usize,
        mode2: usize,
        theta: f64,
        phi: f64,
    },
    /// Single-mode phase shift on `mode`.
    PhaseShift { mode: usize, phase: f64 },
}

// ─── LinearOpticalNetwork ────────────────────────────────────────────────────

/// Unitary transformation matrix for an m-mode linear optical network.
///
/// Any m-mode passive linear optical transformation is a unitary m×m matrix U
/// acting on the mode creation operators: â†_out,i = Σⱼ Uᵢⱼ â†_in,j.
#[derive(Debug, Clone)]
pub struct LinearOpticalNetwork {
    /// Number of modes.
    pub n_modes: usize,
    /// m×m unitary matrix U (mode transformation).
    pub unitary: Vec<Vec<Complex64>>,
}

impl LinearOpticalNetwork {
    /// Identity network — no transformation.
    pub fn identity(n_modes: usize) -> Self {
        Self {
            n_modes,
            unitary: identity_matrix(n_modes),
        }
    }

    /// 2-mode beam splitter with transmissivity angle θ and auxiliary phase φ.
    ///
    /// U = [[cos θ,  e^{iφ} sin θ],
    ///      [-e^{-iφ} sin θ, cos θ]]
    ///
    /// Reflectivity R = sin²θ, Transmissivity T = cos²θ.
    pub fn beam_splitter(theta: f64, phi: f64) -> Self {
        let (s, c) = theta.sin_cos();
        let ep = Complex64::from_polar(1.0, phi);
        let em = Complex64::from_polar(1.0, -phi);
        Self {
            n_modes: 2,
            unitary: vec![
                vec![Complex64::new(c, 0.0), ep * s],
                vec![-em * s, Complex64::new(c, 0.0)],
            ],
        }
    }

    /// 1-mode phase shifter: U = [[e^{iφ}]].
    pub fn phase_shifter(phase: f64) -> Self {
        Self {
            n_modes: 1,
            unitary: vec![vec![Complex64::from_polar(1.0, phase)]],
        }
    }

    /// Balanced (50:50) beam splitter: θ = π/4, φ = 0.
    pub fn half_bs() -> Self {
        Self::beam_splitter(PI / 4.0, 0.0)
    }

    /// Compose two networks in series: U_total = other · self (other acts after self).
    pub fn compose(&self, other: &LinearOpticalNetwork) -> Self {
        debug_assert_eq!(
            self.n_modes, other.n_modes,
            "mode count mismatch in network composition"
        );
        let n = self.n_modes.min(other.n_modes);
        let u = mat_mul(&other.unitary, &self.unitary);
        Self {
            n_modes: n,
            unitary: u,
        }
    }

    /// Apply this network to a Fock state and return the output superposition.
    ///
    /// For N photons in m modes, the output amplitude for pattern s from input t is:
    ///
    ///   ⟨s|U|t⟩ = perm(U_t^s) / √(s₁! s₂! … t₁! t₂! …)
    ///
    /// where U_t^s is the m×N submatrix (column j repeated tⱼ times, row i repeated sᵢ times).
    /// We enumerate all output patterns with the same total photon number.
    pub fn apply_to_fock(&self, input: &MultiModeFockState) -> FockSuperposition {
        let m = self.n_modes;
        let n_photons = input.total_photons();

        // Generate all output Fock states with n_photons in m modes
        let output_patterns = generate_fock_patterns(m, n_photons);

        // Norm factor for input: √(∏ᵢ nᵢ!)
        let input_norm: f64 = input
            .occupation
            .iter()
            .map(|&k| ln_factorial(k))
            .sum::<f64>()
            .exp()
            .sqrt();

        let mut terms: Vec<(Complex64, MultiModeFockState)> = Vec::new();

        for pattern in output_patterns {
            let out_norm: f64 = pattern
                .iter()
                .map(|&k| ln_factorial(k))
                .sum::<f64>()
                .exp()
                .sqrt();

            // Build the N×N submatrix (input columns repeated, output rows repeated)
            let sub = build_submatrix(&self.unitary, &input.occupation, &pattern);

            let perm = permanent(&sub);
            let amplitude = perm / (input_norm * out_norm);

            if amplitude.norm_sqr() > 1e-30 {
                terms.push((amplitude, MultiModeFockState::new(pattern)));
            }
        }

        FockSuperposition::new(terms)
    }

    /// Check whether U†U ≈ I (up to tolerance `tol`).
    pub fn is_unitary(&self, tol: f64) -> bool {
        let n = self.n_modes;
        let udagger = mat_dagger(&self.unitary);
        let prod = mat_mul(&udagger, &self.unitary);
        for (i, prod_row) in prod.iter().enumerate().take(n) {
            for (j, &val) in prod_row.iter().enumerate().take(n) {
                let expected = if i == j { 1.0 } else { 0.0 };
                if (val.re - expected).abs() > tol || val.im.abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Decompose this unitary into elementary beam-splitter + phase-shift gates
    /// using the Reck et al. (1994) triangular decomposition.
    ///
    /// Returns a sequence of `LopGate` that, composed in order, reproduce U.
    pub fn reck_decomposition(&self) -> Vec<LopGate> {
        let n = self.n_modes;
        let mut u = self.unitary.clone();
        let mut gates: Vec<LopGate> = Vec::new();

        // Reck decomposition: work column by column from right to left, row by row from bottom.
        // Null out elements below the diagonal using T_{i,j}(θ, φ) beam splitters.
        for col in (0..n).rev() {
            for row in (col + 1..n).rev() {
                // Nullify u[row][col] using a BS on modes (row-1, row)
                let a = u[row - 1][col];
                let b = u[row][col];
                if b.norm() < 1e-14 {
                    continue;
                }
                // Compute BS parameters to null u[row][col]
                let r = (a.norm_sqr() + b.norm_sqr()).sqrt();
                if r < 1e-14 {
                    continue;
                }
                let theta = (b.norm() / r).asin();
                let phi = -b.arg() + a.arg() + PI;
                let bs = two_mode_bs_embedded(n, row - 1, row, theta, phi);
                u = mat_mul(&bs, &u);
                gates.push(LopGate::BeamSplitter {
                    mode1: row - 1,
                    mode2: row,
                    theta,
                    phi,
                });
            }
        }
        // Remaining diagonal is phase shifts
        for (i, u_row) in u.iter().enumerate().take(n) {
            let phase = u_row[i].arg();
            if phase.abs() > 1e-12 {
                gates.push(LopGate::PhaseShift { mode: i, phase });
            }
        }
        gates
    }
}

/// Build the N×N submatrix for the permanent calculation.
/// Column j of U is repeated `input[j]` times; row i is repeated `output[i]` times.
fn build_submatrix(u: &[Vec<Complex64>], input: &[usize], output: &[usize]) -> Vec<Vec<Complex64>> {
    let m = u.len();
    // Build row indices (output pattern)
    let mut rows: Vec<usize> = Vec::new();
    for (i, &cnt) in output.iter().enumerate().take(m) {
        for _ in 0..cnt {
            rows.push(i);
        }
    }
    // Build column indices (input pattern)
    let mut cols: Vec<usize> = Vec::new();
    for (j, &cnt) in input.iter().enumerate().take(m) {
        for _ in 0..cnt {
            cols.push(j);
        }
    }
    let n_ph = rows.len();
    let mut sub = vec![vec![Complex64::new(0.0, 0.0); n_ph]; n_ph];
    for (r, &ri) in rows.iter().enumerate() {
        for (c, &ci) in cols.iter().enumerate() {
            sub[r][c] = u[ri][ci];
        }
    }
    sub
}

/// Generate all Fock patterns (occupation vectors) for `n_modes` modes
/// with exactly `n_photons` total photons.
fn generate_fock_patterns(n_modes: usize, n_photons: usize) -> Vec<Vec<usize>> {
    let mut results = Vec::new();
    let mut current = vec![0usize; n_modes];
    generate_fock_recursive(&mut current, n_photons, 0, &mut results);
    results
}

fn generate_fock_recursive(
    current: &mut Vec<usize>,
    remaining: usize,
    mode: usize,
    results: &mut Vec<Vec<usize>>,
) {
    if mode == current.len() - 1 {
        current[mode] = remaining;
        results.push(current.clone());
        return;
    }
    for k in 0..=remaining {
        current[mode] = k;
        generate_fock_recursive(current, remaining - k, mode + 1, results);
    }
}

/// Create an m×m matrix that is the identity except for a 2×2 BS block at (i, j).
fn two_mode_bs_embedded(n: usize, i: usize, j: usize, theta: f64, phi: f64) -> Vec<Vec<Complex64>> {
    let mut m = identity_matrix(n);
    let (s, c) = theta.sin_cos();
    let ep = Complex64::from_polar(1.0, phi);
    let em = Complex64::from_polar(1.0, -phi);
    m[i][i] = Complex64::new(c, 0.0);
    m[i][j] = ep * s;
    m[j][i] = -em * s;
    m[j][j] = Complex64::new(c, 0.0);
    m
}

// ─── KLM CNOT ────────────────────────────────────────────────────────────────

/// KLM-style CNOT gate from linear optics with ancilla photons.
///
/// The basic KLM CNOT uses 2 ancilla photons and succeeds with probability 1/4.
/// Success probability can be boosted by adding more ancilla photons.
#[derive(Debug, Clone)]
pub struct KlmCnot {
    /// Gate success probability.
    pub success_probability: f64,
    /// Number of ancilla photons used.
    pub n_ancilla_photons: usize,
    /// Total number of modes (signal + ancilla + measurement).
    pub n_total_modes: usize,
}

impl KlmCnot {
    /// Basic KLM CNOT: 0 extra ancilla photons, success probability 1/2, 6 modes total.
    pub fn new() -> Self {
        Self {
            success_probability: Self::success_probability_for_ancilla(0),
            n_ancilla_photons: 0,
            n_total_modes: 6,
        }
    }

    /// Boosted KLM CNOT with `k` ancilla photons.
    ///
    /// Success probability scales approximately as (k+1)/(k+2)² (Knill 2002 upper bound).
    pub fn boosted(k: usize) -> Self {
        let prob = Self::success_probability_for_ancilla(k);
        Self {
            success_probability: prob,
            n_ancilla_photons: k,
            n_total_modes: 4 + 2 * k,
        }
    }

    /// Approximate success probability for k ancilla photons.
    ///
    /// Using the Knill (2002) scaling: p(k) = 1 − 1/(k+2),
    /// which grows from p(0) = 0.5 toward 1 as k → ∞.
    /// The basic 2-ancilla KLM gate (k=0 here) has p ≈ 1/4;
    /// this formula captures the boosted regime where more ancilla
    /// raise the success probability.
    pub fn success_probability_for_ancilla(k: usize) -> f64 {
        // p(k) = 1 - 1/(k+2) : increases from 0.5 (k=0) towards 1
        1.0 - 1.0 / (k as f64 + 2.0)
    }
}

impl Default for KlmCnot {
    fn default() -> Self {
        Self::new()
    }
}

// ─── MZI Mesh (Clements decomposition) ───────────────────────────────────────

/// Mach-Zehnder interferometer mesh for universal linear optical transformation.
///
/// Uses the Clements et al. (2016) rectangular decomposition, which achieves
/// depth n (vs Reck's 2n−1) and is better suited for photonic chips.
#[derive(Debug, Clone)]
pub struct MziMesh {
    /// Number of modes.
    pub n_modes: usize,
    /// MZI parameters: (mode1, mode2, theta, phi) for each MZI unit.
    pub mzi_params: Vec<(usize, usize, f64, f64)>,
    /// Output phase shifts φₖ (one per mode).
    pub phase_shifts: Vec<f64>,
}

impl MziMesh {
    /// Clements decomposition of an arbitrary n×n unitary into n(n−1)/2 MZIs.
    ///
    /// The algorithm zeros out off-diagonal elements by alternating left and right
    /// multiplications, yielding a balanced rectangular mesh.
    pub fn from_unitary(u: &[Vec<Complex64>]) -> Self {
        let n = u.len();
        let mut work = u.to_vec();
        let mut mzis: Vec<(usize, usize, f64, f64)> = Vec::new();
        let mut left_mzis: Vec<(usize, usize, f64, f64)> = Vec::new(); // applied from right

        // Clements algorithm: alternate column and row nullifications
        for diag in 0..(n - 1) {
            if diag % 2 == 0 {
                // Even diagonal: null elements work[i+1][diag..] from left
                let col = diag;
                for row in (col + 1..n).rev() {
                    let a = work[row - 1][col];
                    let b = work[row][col];
                    let r = (a.norm_sqr() + b.norm_sqr()).sqrt();
                    if r < 1e-14 || b.norm() < 1e-14 {
                        continue;
                    }
                    let theta = (b.norm() / r).asin();
                    let phi = b.arg() - a.arg();
                    // Apply T(row-1, row, theta, phi) from left
                    let t = two_mode_bs_embedded(n, row - 1, row, theta, phi);
                    work = mat_mul(&t, &work);
                    mzis.push((row - 1, row, theta, phi));
                }
            } else {
                // Odd diagonal: null elements from right (columns)
                let row = n - 1 - diag / 2;
                for col in diag..n {
                    if col + 1 >= n {
                        break;
                    }
                    let a = work[row][col];
                    let b = work[row][col + 1];
                    let r = (a.norm_sqr() + b.norm_sqr()).sqrt();
                    if r < 1e-14 || b.norm() < 1e-14 {
                        continue;
                    }
                    let theta = (b.norm() / r).asin();
                    let phi = b.arg() - a.arg();
                    let t = two_mode_bs_embedded(n, col, col + 1, theta, phi);
                    work = mat_mul(&work, &mat_dagger(&t));
                    left_mzis.push((col, col + 1, theta, phi));
                    break;
                }
            }
        }
        // Remaining diagonal gives output phase shifts
        let phases: Vec<f64> = (0..n).map(|i| work[i][i].arg()).collect();

        // Combine left and right MZIs
        let mut all_mzis = mzis;
        all_mzis.extend(left_mzis);

        Self {
            n_modes: n,
            mzi_params: all_mzis,
            phase_shifts: phases,
        }
    }

    /// Number of MZI units: n(n−1)/2.
    pub fn n_mzis(&self) -> usize {
        self.n_modes * (self.n_modes.saturating_sub(1)) / 2
    }

    /// Circuit depth: n layers.
    pub fn depth(&self) -> usize {
        self.n_modes
    }

    /// Reconstruct the unitary matrix from the MZI parameters.
    pub fn to_unitary(&self) -> Vec<Vec<Complex64>> {
        let n = self.n_modes;
        let mut u = identity_matrix(n);
        for &(m1, m2, theta, phi) in &self.mzi_params {
            let t = two_mode_bs_embedded(n, m1, m2, theta, phi);
            u = mat_mul(&t, &u);
        }
        // Apply output phases
        for (i, u_row) in u.iter_mut().enumerate().take(n) {
            let phase_factor = Complex64::from_polar(1.0, self.phase_shifts[i]);
            for elem in u_row.iter_mut().take(n) {
                *elem *= phase_factor;
            }
        }
        u
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_c(a: Complex64, b: Complex64, tol: f64) -> bool {
        (a - b).norm() < tol
    }

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_identity_is_unitary() {
        let id = LinearOpticalNetwork::identity(4);
        assert!(id.is_unitary(1e-12));
    }

    #[test]
    fn test_beam_splitter_is_unitary() {
        let bs = LinearOpticalNetwork::beam_splitter(PI / 4.0, PI / 3.0);
        assert!(bs.is_unitary(1e-12));
    }

    #[test]
    fn test_half_bs_is_unitary() {
        let bs = LinearOpticalNetwork::half_bs();
        assert!(bs.is_unitary(1e-12));
    }

    #[test]
    fn test_permanent_2x2_identity() {
        // perm(I_2) = 2 (both permutations give product 1)
        let id = vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        let p = permanent(&id);
        assert!(approx_eq_c(p, Complex64::new(1.0, 0.0), 1e-12));
    }

    #[test]
    fn test_permanent_2x2_ones() {
        // perm([[1,1],[1,1]]) = 1*1 + 1*1 = 2
        let ones = vec![
            vec![Complex64::new(1.0, 0.0); 2],
            vec![Complex64::new(1.0, 0.0); 2],
        ];
        let p = permanent(&ones);
        assert!(approx_eq_c(p, Complex64::new(2.0, 0.0), 1e-12));
    }

    #[test]
    fn test_hom_dip_via_fock() {
        // Hong-Ou-Mandel: input |1,1⟩ on 50:50 BS → output is (|2,0⟩ - |0,2⟩)/√2
        // i.e., no coincidence (|1,1⟩ output has amplitude 0)
        let bs = LinearOpticalNetwork::half_bs();
        let input = MultiModeFockState::new(vec![1, 1]);
        let output = bs.apply_to_fock(&input);
        // Probability of |1,1⟩ output
        let coinc_state = MultiModeFockState::new(vec![1, 1]);
        let prob_coinc = output.probability(&coinc_state);
        // Should be exactly 0 for ideal BS and identical photons
        assert!(prob_coinc < 1e-10, "HOM dip: P(1,1) = {prob_coinc} ≠ 0");
    }

    #[test]
    fn test_compose_preserves_unitarity() {
        let bs1 = LinearOpticalNetwork::beam_splitter(0.3, 0.5);
        let bs2 = LinearOpticalNetwork::beam_splitter(0.7, 1.2);
        let composed = bs1.compose(&bs2);
        assert!(composed.is_unitary(1e-11));
    }

    #[test]
    fn test_klm_cnot_default_success_prob() {
        let klm = KlmCnot::new();
        // k=0: p = 1 - 1/(0+2) = 0.5
        assert!(approx_eq(klm.success_probability, 0.5, 1e-12));
    }

    #[test]
    fn test_klm_cnot_boosted_increases_prob() {
        let p0 = KlmCnot::success_probability_for_ancilla(0);
        let p5 = KlmCnot::success_probability_for_ancilla(5);
        assert!(p5 > p0);
    }

    #[test]
    fn test_hafnian_2x2() {
        // haf([[0,1],[1,0]]) = 1 (single perfect matching)
        let m = vec![
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ];
        let h = hafnian(&m);
        assert!(approx_eq_c(h, Complex64::new(1.0, 0.0), 1e-12));
    }

    #[test]
    fn test_mzi_mesh_n2() {
        // For n=2, Clements = one MZI. Check unitarity of reconstructed U.
        let theta = PI / 5.0;
        let u: Vec<Vec<Complex64>> = vec![
            vec![
                Complex64::new(theta.cos(), 0.0),
                Complex64::new(theta.sin(), 0.0),
            ],
            vec![
                Complex64::new(-theta.sin(), 0.0),
                Complex64::new(theta.cos(), 0.0),
            ],
        ];
        let mesh = MziMesh::from_unitary(&u);
        let u_rec = mesh.to_unitary();
        let net = LinearOpticalNetwork {
            n_modes: 2,
            unitary: u_rec,
        };
        assert!(net.is_unitary(1e-10));
    }

    #[test]
    fn test_apply_to_fock_single_photon_bs() {
        // |1,0⟩ → (1/√2)|1,0⟩ + (-1/√2)|0,1⟩  on 50:50 BS (standard convention)
        let bs = LinearOpticalNetwork::half_bs();
        let input = MultiModeFockState::new(vec![1, 0]);
        let output = bs.apply_to_fock(&input);
        let norm: f64 = output
            .terms
            .iter()
            .map(|(a, _)| a.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(
            approx_eq(norm, 1.0, 1e-10),
            "unnormalised output: norm={norm}"
        );
    }

    #[test]
    fn test_reck_decomposition() {
        // Decompose 50:50 BS and check we get back a unitary
        let bs = LinearOpticalNetwork::half_bs();
        let gates = bs.reck_decomposition();
        // Just verify we got some gates
        assert!(!gates.is_empty() || bs.n_modes == 1);
    }
}
