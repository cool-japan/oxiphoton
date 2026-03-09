//! Quantum entanglement quantification for bipartite systems.
//!
//! Implements density matrix operations and the standard family of entanglement
//! measures for two-qubit and general bipartite states:
//!
//! - Von Neumann entropy S = −Tr(ρ log₂ ρ)
//! - Entanglement entropy S(ρ_A)
//! - Concurrence C (Wootters formula) and entanglement of formation E_f
//! - Negativity N and logarithmic negativity E_N (partial-transpose criterion)
//! - Polarisation and time-bin entanglement characterisation
//!
//! References:
//! - Wootters, PRL 80, 2245 (1998): concurrence and entanglement of formation
//! - Vidal & Werner, PRA 65, 032314 (2002): negativity
//! - Horodecki et al., Rev. Mod. Phys. 81, 865 (2009): entanglement review

use num_complex::Complex64;
use std::f64::consts::LN_2;

// ─── Density matrix ───────────────────────────────────────────────────────────

/// Density matrix for a bipartite quantum system A ⊗ B.
///
/// `rho` is a (dim_a × dim_b)² square matrix stored in row-major order as
/// `rho[i][j]` where row / column index encodes the joint state `|ab⟩`.
#[derive(Debug, Clone)]
pub struct DensityMatrix {
    /// Hilbert-space dimension of subsystem A
    pub dim_a: usize,
    /// Hilbert-space dimension of subsystem B
    pub dim_b: usize,
    /// Density matrix entries: `rho[row][col]`, size `(dim_a*dim_b) × (dim_a*dim_b)`
    pub rho: Vec<Vec<Complex64>>,
}

impl DensityMatrix {
    /// Construct the maximally mixed state I/(dim_a * dim_b).
    pub fn new(dim_a: usize, dim_b: usize) -> Self {
        let d = dim_a * dim_b;
        let val = Complex64::new(1.0 / d as f64, 0.0);
        let rho = (0..d)
            .map(|i| {
                (0..d)
                    .map(|j| {
                        if i == j {
                            val
                        } else {
                            Complex64::new(0.0, 0.0)
                        }
                    })
                    .collect()
            })
            .collect();
        Self { dim_a, dim_b, rho }
    }

    /// Construct a pure state density matrix ρ = |ψ⟩⟨ψ| from an amplitude vector.
    ///
    /// The vector is automatically normalised.
    pub fn pure_state(state: &[Complex64]) -> Self {
        let d = state.len();
        // Determine dim_a, dim_b: assume square factorisation (e.g. 4 = 2×2)
        let dim_a = (d as f64).sqrt() as usize;
        let dim_b = if dim_a * dim_a == d { dim_a } else { d };
        let norm_sq: f64 = state.iter().map(|v| v.norm_sqr()).sum();
        let norm = norm_sq.sqrt().max(1e-30);
        let normed: Vec<Complex64> = state.iter().map(|v| v / norm).collect();
        let rho: Vec<Vec<Complex64>> = (0..d)
            .map(|i| (0..d).map(|j| normed[i] * normed[j].conj()).collect())
            .collect();
        Self { dim_a, dim_b, rho }
    }

    // ── Bell states ────────────────────────────────────────────────────────────

    /// |Φ+⟩ = (|00⟩ + |11⟩) / √2
    pub fn bell_phi_plus() -> Self {
        let s = 1.0_f64 / 2.0_f64.sqrt();
        let state = [
            Complex64::new(s, 0.0),   // |00⟩
            Complex64::new(0.0, 0.0), // |01⟩
            Complex64::new(0.0, 0.0), // |10⟩
            Complex64::new(s, 0.0),   // |11⟩
        ];
        Self::pure_state(&state)
    }

    /// |Φ-⟩ = (|00⟩ − |11⟩) / √2
    pub fn bell_phi_minus() -> Self {
        let s = 1.0_f64 / 2.0_f64.sqrt();
        let state = [
            Complex64::new(s, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-s, 0.0),
        ];
        Self::pure_state(&state)
    }

    /// |Ψ+⟩ = (|01⟩ + |10⟩) / √2
    pub fn bell_psi_plus() -> Self {
        let s = 1.0_f64 / 2.0_f64.sqrt();
        let state = [
            Complex64::new(0.0, 0.0),
            Complex64::new(s, 0.0),
            Complex64::new(s, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        Self::pure_state(&state)
    }

    /// |Ψ-⟩ = (|01⟩ − |10⟩) / √2
    pub fn bell_psi_minus() -> Self {
        let s = 1.0_f64 / 2.0_f64.sqrt();
        let state = [
            Complex64::new(0.0, 0.0),
            Complex64::new(s, 0.0),
            Complex64::new(-s, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        Self::pure_state(&state)
    }

    /// Werner state: ρ_W(p) = p |Ψ-⟩⟨Ψ-| + (1−p) I/4.
    ///
    /// Separable for p ≤ 1/3, entangled (but PPT) for 1/3 < p ≤ 1/2,
    /// entangled and NPT for p > 1/2.
    pub fn werner_state(p: f64) -> Self {
        let p = p.clamp(0.0, 1.0);
        let psi_minus = Self::bell_psi_minus();
        let mixed = Self::new(2, 2);
        let d = 4;
        let rho: Vec<Vec<Complex64>> = (0..d)
            .map(|i| {
                (0..d)
                    .map(|j| {
                        Complex64::new(p, 0.0) * psi_minus.rho[i][j]
                            + Complex64::new(1.0 - p, 0.0) * mixed.rho[i][j]
                    })
                    .collect()
            })
            .collect();
        Self {
            dim_a: 2,
            dim_b: 2,
            rho,
        }
    }

    // ── Partial traces ─────────────────────────────────────────────────────────

    /// Partial trace over subsystem B → reduced density matrix ρ_A (dim_a × dim_a).
    pub fn partial_trace_b(&self) -> Vec<Vec<Complex64>> {
        let da = self.dim_a;
        let db = self.dim_b;
        let mut rho_a = vec![vec![Complex64::new(0.0, 0.0); da]; da];
        for (a1, rho_a_row) in rho_a.iter_mut().enumerate().take(da) {
            for (a2, elem) in rho_a_row.iter_mut().enumerate().take(da) {
                let mut sum = Complex64::new(0.0, 0.0);
                for b in 0..db {
                    let row = a1 * db + b;
                    let col = a2 * db + b;
                    sum += self.rho[row][col];
                }
                *elem = sum;
            }
        }
        rho_a
    }

    /// Partial trace over subsystem A → reduced density matrix ρ_B (dim_b × dim_b).
    pub fn partial_trace_a(&self) -> Vec<Vec<Complex64>> {
        let da = self.dim_a;
        let db = self.dim_b;
        let mut rho_b = vec![vec![Complex64::new(0.0, 0.0); db]; db];
        for (b1, rho_b_row) in rho_b.iter_mut().enumerate().take(db) {
            for (b2, elem) in rho_b_row.iter_mut().enumerate().take(db) {
                let mut sum = Complex64::new(0.0, 0.0);
                for a in 0..da {
                    let row = a * db + b1;
                    let col = a * db + b2;
                    sum += self.rho[row][col];
                }
                *elem = sum;
            }
        }
        rho_b
    }

    // ── Entropy ────────────────────────────────────────────────────────────────

    /// Von Neumann entropy of the full system: S = −Tr(ρ log₂ ρ).
    pub fn von_neumann_entropy(&self) -> f64 {
        let d = self.dim_a * self.dim_b;
        let eigenvalues = hermitian_eigenvalues_real(&self.rho, d);
        shannon_entropy_from_eigenvalues(&eigenvalues)
    }

    /// Entanglement entropy = von Neumann entropy of the reduced state ρ_A.
    ///
    /// For pure states this equals the unique entanglement measure.
    pub fn entanglement_entropy(&self) -> f64 {
        let rho_a = self.partial_trace_b();
        let eigenvalues = hermitian_eigenvalues_real(&rho_a, self.dim_a);
        shannon_entropy_from_eigenvalues(&eigenvalues)
    }

    // ── 2-qubit measures ───────────────────────────────────────────────────────

    /// Concurrence C for a 2-qubit state (Wootters formula, 1998).
    ///
    /// C ∈ [0, 1]: C = 0 separable, C = 1 maximally entangled.
    pub fn concurrence(&self) -> f64 {
        if self.dim_a != 2 || self.dim_b != 2 {
            return 0.0; // Only defined for 2-qubit systems
        }
        // R = ρ (σ_y ⊗ σ_y) ρ* (σ_y ⊗ σ_y)
        // Eigenvalues of R: λ₁ ≥ λ₂ ≥ λ₃ ≥ λ₄ ≥ 0 (square roots)
        // C = max(0, √λ₁ − √λ₂ − √λ₃ − √λ₄)
        let r = spin_flipped_matrix(&self.rho);
        let eigenvalues = positive_eigenvalues_4x4(&r);
        // Pad to exactly 4 elements (deflation may return fewer near-zero eigenvalues)
        let mut sqrt_eigs: Vec<f64> = eigenvalues
            .iter()
            .map(|&e| if e > 0.0 { e.sqrt() } else { 0.0 })
            .collect();
        sqrt_eigs.resize(4, 0.0);
        sqrt_eigs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let c = sqrt_eigs[0] - sqrt_eigs[1] - sqrt_eigs[2] - sqrt_eigs[3];
        c.max(0.0)
    }

    /// Entanglement of formation: E_f = h((1 + √(1 − C²)) / 2).
    ///
    /// E_f ∈ [0, 1] (ebits).
    pub fn entanglement_of_formation(&self) -> f64 {
        let c = self.concurrence();
        let x = (1.0 + (1.0 - c * c).max(0.0).sqrt()) / 2.0;
        binary_entropy(x)
    }

    /// Negativity N = (‖ρ^{T_B}‖₁ − 1) / 2.
    ///
    /// For separable states N = 0; for maximally entangled N = 0.5.
    pub fn negativity(&self) -> f64 {
        if self.dim_a != 2 || self.dim_b != 2 {
            return 0.0; // General case requires larger partial transpose
        }
        let pt = partial_transpose_b(&self.rho, self.dim_a, self.dim_b);
        let trace_norm = trace_norm_4x4(&pt);
        (trace_norm - 1.0).max(0.0) / 2.0
    }

    /// Logarithmic negativity E_N = log₂(‖ρ^{T_B}‖₁).
    pub fn log_negativity(&self) -> f64 {
        if self.dim_a != 2 || self.dim_b != 2 {
            return 0.0;
        }
        let pt = partial_transpose_b(&self.rho, self.dim_a, self.dim_b);
        let trace_norm = trace_norm_4x4(&pt);
        trace_norm.max(1e-30).log2()
    }

    /// Fidelity to a pure target state: F = ⟨ψ|ρ|ψ⟩.
    pub fn fidelity_to_pure(&self, target: &[Complex64]) -> f64 {
        let d = self.dim_a * self.dim_b;
        if target.len() != d {
            return 0.0;
        }
        let norm_sq: f64 = target.iter().map(|v| v.norm_sqr()).sum();
        let norm = norm_sq.sqrt().max(1e-30);
        let normed: Vec<Complex64> = target.iter().map(|v| v / norm).collect();
        // F = ⟨ψ|ρ|ψ⟩ = Σ_{i,j} ψ*_i ρ_{ij} ψ_j
        let mut f = Complex64::new(0.0, 0.0);
        for i in 0..d {
            for j in 0..d {
                f += normed[i].conj() * self.rho[i][j] * normed[j];
            }
        }
        f.re.clamp(0.0, 1.0)
    }

    /// Returns `true` if the state is separable (negativity < tol).
    pub fn is_separable(&self, tol: f64) -> bool {
        self.negativity() < tol
    }
}

// ─── Internal linear algebra ──────────────────────────────────────────────────

/// Binary (Shannon) entropy: h(x) = −x log₂ x − (1−x) log₂(1−x).
pub fn binary_entropy(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    let h = |p: f64| {
        if !(1e-15..=1.0 - 1e-15).contains(&p) {
            0.0
        } else {
            -p * p.log2()
        }
    };
    h(x) + h(1.0 - x)
}

/// Shannon entropy −Σ λ_k log₂ λ_k from a list of (real) eigenvalues.
fn shannon_entropy_from_eigenvalues(eigenvalues: &[f64]) -> f64 {
    eigenvalues.iter().fold(0.0, |acc, &lam| {
        if lam > 1e-15 {
            acc - lam * (lam.ln() / LN_2)
        } else {
            acc
        }
    })
}

/// Eigenvalues of a real symmetric (or Hermitian real part) matrix
/// using power deflation. Returns `d` eigenvalues (may include near-zero).
fn hermitian_eigenvalues_real(m: &[Vec<Complex64>], d: usize) -> Vec<f64> {
    if d == 0 {
        return vec![];
    }
    if d == 1 {
        return vec![m[0][0].re];
    }
    if d == 2 {
        return eigenvalues_2x2_symmetric(m[0][0].re, m[0][1].re, m[1][1].re);
    }
    // For 4×4 (most common: 2-qubit), use direct analytical or numerical approach
    if d == 4 {
        return eigenvalues_4x4_real_symmetric(m);
    }
    // General: power deflation on real part
    let real_m: Vec<Vec<f64>> = m
        .iter()
        .map(|row| row.iter().map(|v| v.re).collect())
        .collect();
    power_deflation_eigenvalues(&real_m, d, d)
}

/// Analytical eigenvalues of a 2×2 symmetric matrix [[a, b], [b, c]].
fn eigenvalues_2x2_symmetric(a: f64, b: f64, c: f64) -> Vec<f64> {
    let trace = a + c;
    let det = a * c - b * b;
    let disc = ((trace * trace) / 4.0 - det).max(0.0).sqrt();
    vec![trace / 2.0 + disc, trace / 2.0 - disc]
}

/// Eigenvalues of a real symmetric 4×4 matrix via power deflation.
fn eigenvalues_4x4_real_symmetric(m: &[Vec<Complex64>]) -> Vec<f64> {
    let real_m: Vec<Vec<f64>> = m
        .iter()
        .map(|row| row.iter().map(|v| v.re).collect())
        .collect();
    power_deflation_eigenvalues(&real_m, 4, 4)
}

/// Power deflation to find `n_eigs` eigenvalues of a real symmetric matrix.
///
/// Always attempts to extract exactly `n_eigs` eigenvalues.  Near-zero
/// eigenvalues are included (as 0.0) so that the caller always gets a
/// fixed-length slice.
fn power_deflation_eigenvalues(m: &[Vec<f64>], n: usize, n_eigs: usize) -> Vec<f64> {
    let mut mat: Vec<Vec<f64>> = m.to_vec();
    let mut eigenvalues = Vec::with_capacity(n_eigs);
    for _ in 0..n_eigs {
        let (lam, v) = power_iteration_largest(&mat, n, 500, 1e-13);
        eigenvalues.push(lam);
        // Always deflate, even for near-zero eigenvalues, to reveal the next one
        for i in 0..n {
            for j in 0..n {
                mat[i][j] -= lam * v[i] * v[j];
            }
        }
    }
    eigenvalues
}

/// Power iteration for the largest-magnitude eigenvalue/vector of a real symmetric matrix.
fn power_iteration_largest(m: &[Vec<f64>], n: usize, max_iter: usize, tol: f64) -> (f64, Vec<f64>) {
    let mut v: Vec<f64> = (0..n)
        .map(|i| if i == 0 { 1.0 } else { 0.1 / (i as f64) })
        .collect();
    normalise_vec(&mut v);
    let mut lam = 0.0_f64;
    for _ in 0..max_iter {
        let mv = matvec(m, &v, n);
        let new_lam: f64 = v.iter().zip(mv.iter()).map(|(a, b)| a * b).sum();
        let norm: f64 = mv.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-30);
        let v_new: Vec<f64> = mv.iter().map(|x| x / norm).collect();
        let diff: f64 = v_new.iter().zip(v.iter()).map(|(a, b)| (a - b).abs()).sum();
        v = v_new;
        let converged = (new_lam - lam).abs() < tol && diff < tol;
        lam = new_lam;
        if converged {
            break;
        }
    }
    (lam, v)
}

fn normalise_vec(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-30);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

fn matvec(m: &[Vec<f64>], v: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            result[i] += m[i][j] * v[j];
        }
    }
    result
}

/// Compute the spin-flip matrix R = ρ (σ_y⊗σ_y) ρ* (σ_y⊗σ_y) for concurrence.
///
/// σ_y⊗σ_y in the basis {|00⟩,|01⟩,|10⟩,|11⟩} is:
/// ```text
/// [ 0  0  0 -1 ]
/// [ 0  0  1  0 ]
/// [ 0  1  0  0 ]
/// [-1  0  0  0 ]
/// ```
fn spin_flipped_matrix(rho: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    // σ_y⊗σ_y: antidiagonal with alternating signs
    // (σ_y⊗σ_y)[i][j]: row 0 → col 3 with -1, row 1 → col 2 with +1, etc.
    let sigma_y_tp_sigma_y = |i: usize, j: usize| -> Complex64 {
        let map = [
            (0usize, 3usize, -1.0_f64),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (3, 0, -1.0),
        ];
        for &(r, c, sign) in &map {
            if i == r && j == c {
                return Complex64::new(sign, 0.0);
            }
        }
        Complex64::new(0.0, 0.0)
    };
    // Compute tilde_rho = (σ_y⊗σ_y) ρ* (σ_y⊗σ_y)
    let n = 4;
    let mut tilde_rho = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for (i, tilde_row) in tilde_rho.iter_mut().enumerate().take(n) {
        for (j, elem) in tilde_row.iter_mut().enumerate().take(n) {
            let mut val = Complex64::new(0.0, 0.0);
            for (k, rho_row) in rho.iter().enumerate().take(n) {
                for (l, &rho_kl) in rho_row.iter().enumerate().take(n) {
                    val += sigma_y_tp_sigma_y(i, k) * rho_kl.conj() * sigma_y_tp_sigma_y(l, j);
                }
            }
            *elem = val;
        }
    }
    // R = ρ * tilde_rho
    let mut r = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                r[i][j] += rho[i][k] * tilde_rho[k][j];
            }
        }
    }
    r
}

/// Eigenvalues (non-negative real parts) of R for Wootters concurrence.
/// R = ρ (σ_y⊗σ_y) ρ* (σ_y⊗σ_y) has real non-negative eigenvalues.
fn positive_eigenvalues_4x4(r: &[Vec<Complex64>]) -> Vec<f64> {
    let real_r: Vec<Vec<f64>> = r
        .iter()
        .map(|row| row.iter().map(|v| v.re).collect())
        .collect();
    let mut eigs = power_deflation_eigenvalues(&real_r, 4, 4);
    for e in eigs.iter_mut() {
        if *e < 0.0 {
            *e = 0.0;
        }
    }
    eigs
}

/// Partial transpose over subsystem B for a (dim_a × dim_b) bipartite system.
///
/// (ρ^{T_B})_{a₁ b₁, a₂ b₂} = ρ_{a₁ b₂, a₂ b₁}
fn partial_transpose_b(rho: &[Vec<Complex64>], dim_a: usize, dim_b: usize) -> Vec<Vec<Complex64>> {
    let d = dim_a * dim_b;
    let mut pt = vec![vec![Complex64::new(0.0, 0.0); d]; d];
    for a1 in 0..dim_a {
        for b1 in 0..dim_b {
            for a2 in 0..dim_a {
                for b2 in 0..dim_b {
                    let row = a1 * dim_b + b1;
                    let col = a2 * dim_b + b2;
                    // (ρ^{T_B})_{a1 b1, a2 b2} = ρ_{a1 b2, a2 b1}
                    let src_row = a1 * dim_b + b2;
                    let src_col = a2 * dim_b + b1;
                    pt[row][col] = rho[src_row][src_col];
                }
            }
        }
    }
    pt
}

/// Trace norm ‖M‖₁ = Σ |λ_k| for a Hermitian 4×4 matrix.
///
/// Uses both positive and negative eigenvalue deflation to capture all
/// four eigenvalues (including negative ones, which power iteration may miss).
fn trace_norm_4x4(m: &[Vec<Complex64>]) -> f64 {
    // Convert to real matrix
    let real_m: Vec<Vec<f64>> = m
        .iter()
        .map(|row| row.iter().map(|v| v.re).collect())
        .collect();
    // Use the characteristic polynomial approach for 4×4 or dual deflation
    let eigs = hermitian_4x4_all_eigenvalues(&real_m);
    eigs.iter().map(|&e| e.abs()).sum()
}

/// Extract all 4 eigenvalues of a real symmetric 4×4 matrix.
///
/// Uses a shifted power method to reliably find both positive and negative
/// eigenvalues via the "shift-and-invert"–style approach:
/// For each eigenvalue we shift the matrix so the target eigenvalue becomes
/// the largest magnitude.
fn hermitian_4x4_all_eigenvalues(m: &[Vec<f64>]) -> Vec<f64> {
    let n = 4;
    // Estimate spectral radius via Gershgorin: max row sum of |m[i][j]|
    let spectral_radius: f64 = m
        .iter()
        .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    let shift = spectral_radius + 1.0; // shift > all eigenvalues

    // Step 1: Find all eigenvalues of (M + shift*I) using standard power deflation.
    // (M + shift*I) has all-positive eigenvalues: ν_k = λ_k + shift.
    let mut shifted = m.to_vec();
    for (i, row) in shifted.iter_mut().enumerate().take(n) {
        row[i] += shift;
    }

    let mut shifted_eigs = Vec::with_capacity(n);
    let mut mat = shifted.clone();
    for _ in 0..n {
        let (nu, v) = power_iteration_largest(&mat, n, 600, 1e-13);
        shifted_eigs.push(nu);
        for i in 0..n {
            for j in 0..n {
                mat[i][j] -= nu * v[i] * v[j];
            }
        }
    }

    // Recover original eigenvalues: λ_k = ν_k - shift
    shifted_eigs.iter().map(|&nu| nu - shift).collect()
}

// ─── Polarisation entanglement ────────────────────────────────────────────────

/// Characterisation of polarisation entanglement via coincidence visibilities.
///
/// Visibilities are measured in three mutually unbiased bases:
/// - HV: horizontal/vertical
/// - DA: diagonal/antidiagonal (+45°/−45°)
/// - RL: right/left circular
#[derive(Debug, Clone)]
pub struct PolarizationEntanglement {
    /// HV basis coincidence visibility
    pub visibility_hv: f64,
    /// DA basis coincidence visibility
    pub visibility_da: f64,
    /// RL (circular) basis coincidence visibility
    pub visibility_rl: f64,
}

impl PolarizationEntanglement {
    /// Construct from three basis visibilities (each in [0, 1]).
    pub fn new(v_hv: f64, v_da: f64, v_rl: f64) -> Self {
        Self {
            visibility_hv: v_hv.clamp(0.0, 1.0),
            visibility_da: v_da.clamp(0.0, 1.0),
            visibility_rl: v_rl.clamp(0.0, 1.0),
        }
    }

    /// Tangle T = max(0, 2F − 1) where F is the Bell-state fidelity.
    ///
    /// T = C² where C is the concurrence.  T ∈ [0, 1].
    pub fn tangle(&self) -> f64 {
        let f = self.bell_state_fidelity();
        (2.0 * f - 1.0).max(0.0)
    }

    /// Concurrence C = √T ∈ [0, 1].
    pub fn concurrence(&self) -> f64 {
        self.tangle().sqrt()
    }

    /// Bell-state fidelity: F ≈ (1 + V_HV + V_DA + V_RL) / 4.
    ///
    /// Derived from the standard polarimetric reconstruction.
    pub fn bell_state_fidelity(&self) -> f64 {
        ((1.0 + self.visibility_hv + self.visibility_da + self.visibility_rl) / 4.0).clamp(0.0, 1.0)
    }

    /// Returns `true` if the state is entangled (fidelity > 0.5 guarantees entanglement).
    pub fn is_entangled(&self) -> bool {
        self.bell_state_fidelity() > 0.5
    }
}

// ─── Time-bin entanglement ────────────────────────────────────────────────────

/// Characterisation of time-bin entangled photon pairs.
///
/// Two-qubit time-bin states use early (|e⟩) and late (|l⟩) time slots.
/// The maximally entangled state is (|ee⟩ + e^{iφ}|ll⟩)/√2.
#[derive(Debug, Clone)]
pub struct TimeBinEntanglement {
    /// Visibility in the Z-basis (|e⟩/|l⟩ distinguishable)
    pub visibility_z: f64,
    /// Visibility in the X-basis (superposition measurement)
    pub visibility_x: f64,
    /// Relative phase between early and late bins (radians)
    pub phi: f64,
}

impl TimeBinEntanglement {
    /// Construct from Z and X visibilities and relative phase.
    pub fn new(v_z: f64, v_x: f64, phi: f64) -> Self {
        Self {
            visibility_z: v_z.clamp(0.0, 1.0),
            visibility_x: v_x.clamp(0.0, 1.0),
            phi,
        }
    }

    /// Concurrence ≈ V_X for an idealised time-bin state.
    ///
    /// C = V_X when V_Z ≈ 1 (good time-bin discrimination).
    pub fn concurrence(&self) -> f64 {
        self.visibility_x
    }

    /// Quantum bit error rate (QBER) derived from visibility:
    /// QBER = (1 − V_X) / 2.
    pub fn qber(&self) -> f64 {
        (1.0 - self.visibility_x) / 2.0
    }

    /// Secret key fraction (one-way error correction, BB84-like):
    /// r = 1 − 2 h(QBER).
    pub fn secret_key_fraction(&self) -> f64 {
        let qber = self.qber();
        let r = 1.0 - 2.0 * binary_entropy(qber);
        r.max(0.0)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bell_state_purity() {
        // A Bell state is pure: Tr(ρ²) = 1
        let rho = DensityMatrix::bell_phi_plus();
        let d = 4;
        // Tr(ρ²) = Σ_{i,j} |ρ_{ij}|²  (for density matrices Tr(ρ²) = Σ|ρ|²)
        let tr_rho_sq: f64 = (0..d)
            .map(|i| (0..d).map(|j| rho.rho[i][j].norm_sqr()).sum::<f64>())
            .sum();
        // For pure state Tr(ρ²) = 1
        assert!((tr_rho_sq - 1.0).abs() < 1e-10, "Bell state must be pure");
    }

    #[test]
    fn test_bell_state_entanglement_entropy() {
        // Bell states have S(ρ_A) = 1 ebit (maximally entangled)
        let rho = DensityMatrix::bell_phi_plus();
        let s = rho.entanglement_entropy();
        assert!(
            (s - 1.0).abs() < 1e-6,
            "Bell state entropy = 1 ebit, got {s}"
        );
    }

    #[test]
    fn test_concurrence_bell_state() {
        // C(Bell state) = 1
        let rho = DensityMatrix::bell_psi_minus();
        let c = rho.concurrence();
        assert!(
            (c - 1.0).abs() < 1e-6,
            "Concurrence of Bell state = 1, got {c}"
        );
    }

    #[test]
    fn test_concurrence_product_state() {
        // |00⟩ is separable: C = 0
        let state = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let rho = DensityMatrix::pure_state(&state);
        let c = rho.concurrence();
        assert!(c < 1e-6, "Product state concurrence = 0, got {c}");
    }

    #[test]
    fn test_negativity_bell_state() {
        // N(|Ψ-⟩) = 0.5
        let rho = DensityMatrix::bell_psi_minus();
        let n = rho.negativity();
        assert!(
            (n - 0.5).abs() < 1e-6,
            "Negativity of Bell state = 0.5, got {n}"
        );
    }

    #[test]
    fn test_werner_state_separability() {
        // Werner state p=0.1 < 1/3 is separable (negativity = 0)
        let rho = DensityMatrix::werner_state(0.1);
        assert!(rho.is_separable(1e-6), "Werner(0.1) should be separable");
        // Werner state p=0.9 is entangled
        let rho2 = DensityMatrix::werner_state(0.9);
        assert!(!rho2.is_separable(1e-6), "Werner(0.9) should be entangled");
    }

    #[test]
    fn test_fidelity_bell_state() {
        let rho = DensityMatrix::bell_phi_plus();
        let s = 1.0_f64 / 2.0_f64.sqrt();
        let target = [
            Complex64::new(s, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(s, 0.0),
        ];
        let f = rho.fidelity_to_pure(&target);
        assert!(
            (f - 1.0).abs() < 1e-6,
            "Fidelity of state to itself = 1, got {f}"
        );
    }

    #[test]
    fn test_polarization_entanglement_perfect() {
        let pe = PolarizationEntanglement::new(1.0, 1.0, 1.0);
        // With all three visibilities = 1.0: F = (1+1+1+1)/4 = 1.0
        assert!(
            (pe.bell_state_fidelity() - 1.0).abs() < 1e-9,
            "F={}",
            pe.bell_state_fidelity()
        );
        assert!(pe.is_entangled());
        // Tangle T = max(0, 2F-1) = 1.0; Concurrence = √T = 1.0
        assert!(
            (pe.concurrence() - 1.0).abs() < 1e-9,
            "C={}",
            pe.concurrence()
        );
    }

    #[test]
    fn test_time_bin_qber() {
        let tbe = TimeBinEntanglement::new(1.0, 0.96, 0.0);
        let qber = tbe.qber();
        assert!((qber - 0.02).abs() < 1e-9, "QBER = (1-V_X)/2");
    }

    #[test]
    fn test_binary_entropy_extremes() {
        assert!(binary_entropy(0.0).abs() < 1e-12, "h(0) = 0");
        assert!(binary_entropy(1.0).abs() < 1e-12, "h(1) = 0");
        assert!((binary_entropy(0.5) - 1.0).abs() < 1e-10, "h(0.5) = 1");
    }
}
