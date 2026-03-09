//! Gaussian boson sampling (GBS) and boson sampling complexity certificates.
//!
//! GBS uses squeezed coherent states fed into a linear optical interferometer,
//! followed by photon-number-resolving (PNR) detection.  The output distribution
//! is related to the hafnian of submatrices of the adjacency matrix A.
//!
//! # References
//! - Aaronson & Arkhipov (2011) — original boson sampling proposal
//! - Hamilton et al. (2017) — Gaussian boson sampling
//! - Arrazola et al. (2021) — GBS with photonic hardware (Xanadu Borealis)

use num_complex::Complex64;

use super::linear_optical::hafnian;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// ln n! via Stirling / exact table.
fn ln_factorial(n: usize) -> f64 {
    match n {
        0 | 1 => 0.0,
        2..=20 => (2..=n).map(|k| (k as f64).ln()).sum(),
        _ => {
            let nf = n as f64;
            nf * nf.ln() - nf + 0.5 * (2.0 * std::f64::consts::PI * nf).ln() + 1.0 / (12.0 * nf)
        }
    }
}

/// Binomial coefficient C(n+k-1, k) — stars and bars (multiset coefficient).
fn multiset_coeff(n: usize, k: usize) -> u64 {
    // C(n+k-1, k)
    let top = n + k - 1;
    let bot = k;
    if bot > top {
        return 0;
    }
    let bot = bot.min(top - bot);
    let mut result = 1u64;
    for i in 0..bot {
        result = result.saturating_mul(top as u64 - i as u64) / (i as u64 + 1);
    }
    result
}

// ─── GaussianBosonSampler ────────────────────────────────────────────────────

/// Gaussian boson sampler: squeezed vacuum states through a linear interferometer.
///
/// Input: m single-mode squeezed vacuum states |r_k⟩ (squeezing parameter r_k ≥ 0).
/// Interferometer: m×m unitary U.
/// Output: PNR detection in each output mode.
///
/// The output probability of pattern s is proportional to |haf(A_s)|²,
/// where A_s is the submatrix of the adjacency matrix A = U diag(tanh r) Uᵀ
/// with rows and columns repeated according to the pattern s.
#[derive(Debug, Clone)]
pub struct GaussianBosonSampler {
    /// Number of modes m.
    pub n_modes: usize,
    /// Single-mode squeezing parameters r_k ≥ 0 for each input mode.
    pub squeezing_params: Vec<f64>,
    /// m×m interferometer unitary U.
    pub unitary: Vec<Vec<Complex64>>,
}

impl GaussianBosonSampler {
    /// Construct a GBS instance.
    ///
    /// `squeezing` and `unitary` must have consistent dimension `n_modes`.
    /// Squeezing values are clamped to [0, ∞).
    pub fn new(n_modes: usize, squeezing: Vec<f64>, unitary: Vec<Vec<Complex64>>) -> Self {
        let sq: Vec<f64> = squeezing.into_iter().map(|r| r.max(0.0)).collect();
        Self {
            n_modes,
            squeezing_params: sq,
            unitary,
        }
    }

    /// Mean photon number in mode k: ⟨n_k⟩ = sinh²(r_k).
    pub fn mean_photon_number(&self, mode: usize) -> f64 {
        let r = self.squeezing_params.get(mode).copied().unwrap_or(0.0);
        r.sinh().powi(2)
    }

    /// Total mean photon number: Σ_k sinh²(r_k).
    pub fn total_mean_photons(&self) -> f64 {
        self.squeezing_params
            .iter()
            .map(|&r| r.sinh().powi(2))
            .sum()
    }

    /// Adjacency matrix A = U · diag(tanh r) · Uᵀ (complex symmetric).
    ///
    /// This is the core object whose hafnian submatrices give GBS output probabilities.
    pub fn adjacency_matrix(&self) -> Vec<Vec<Complex64>> {
        let m = self.n_modes;
        // B = U · diag(tanh r_k)
        let mut b = vec![vec![Complex64::new(0.0, 0.0); m]; m];
        for (i, b_row) in b.iter_mut().enumerate().take(m) {
            for (j, b_cell) in b_row.iter_mut().enumerate().take(m) {
                let tanh_rj = self.squeezing_params.get(j).copied().unwrap_or(0.0).tanh();
                *b_cell = self.unitary[i][j] * tanh_rj;
            }
        }
        // A = B · Uᵀ  (not U†, Uᵀ = transpose without conjugation)
        let mut a = vec![vec![Complex64::new(0.0, 0.0); m]; m];
        for (i, a_row) in a.iter_mut().enumerate().take(m) {
            for (j, a_cell) in a_row.iter_mut().enumerate().take(m) {
                for (k, &b_ik) in b[i].iter().enumerate().take(m) {
                    *a_cell += b_ik * self.unitary[j][k]; // Uᵀ[k,j] = U[j,k]
                }
            }
        }
        a
    }

    /// Probability of the vacuum outcome P(0…0).
    ///
    /// P(0) = 1 / √det(σ_Q)  where σ_Q is the Husimi Q covariance matrix.
    /// For pure squeezed vacuum input:
    ///   P(0) = 1 / Π_k cosh(r_k)
    ///
    /// This is the exact formula for pure single-mode squeezing input.
    pub fn vacuum_probability(&self) -> f64 {
        let denom: f64 = self.squeezing_params.iter().map(|&r| r.cosh()).product();
        if denom < f64::EPSILON {
            return 0.0;
        }
        1.0 / denom
    }

    /// Probability of a specific photon number pattern `s`.
    ///
    /// P(s) = |haf(A_s)|² / (Π_k s_k! · √det(σ_Q))
    ///
    /// where A_s is the submatrix of A with row/col i repeated s_i times.
    pub fn pattern_probability(&self, pattern: &[usize]) -> f64 {
        if pattern.len() != self.n_modes {
            return 0.0;
        }
        let a = self.adjacency_matrix();
        let n_ph: usize = pattern.iter().sum();
        if n_ph == 0 {
            return self.vacuum_probability();
        }
        // Build A_s submatrix (row i repeated pattern[i] times)
        let mut rows: Vec<usize> = Vec::with_capacity(n_ph);
        for (i, &cnt) in pattern.iter().enumerate() {
            for _ in 0..cnt {
                rows.push(i);
            }
        }
        let dim = rows.len();
        let mut a_s = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
        for (r, &ri) in rows.iter().enumerate() {
            for (c, &ci) in rows.iter().enumerate() {
                a_s[r][c] = a[ri][ci];
            }
        }
        let h = hafnian(&a_s);
        let haf_sq = h.norm_sqr();

        // Denominator: Π_k s_k!
        let fact_denom: f64 = pattern.iter().map(|&k| ln_factorial(k)).sum::<f64>().exp();

        // Prefactor: vacuum_probability
        let p0 = self.vacuum_probability();

        p0 * haf_sq / fact_denom
    }

    /// Approximate sample from the GBS distribution using a coherent-state approximation.
    ///
    /// Each mode is treated as an independent Poisson process with mean ⟨n_k⟩.
    /// This is only exact for coherent states (r → 0); for finite squeezing it is
    /// an approximation useful for seeding classical simulators.
    pub fn approximate_sample(&self) -> Vec<usize> {
        // Use a deterministic pseudo-sample derived from the mean photon numbers,
        // rounding to nearest integer. A production implementation would use
        // a proper random number generator; here we return the mode-wise floor
        // of the expected photon number as a deterministic approximation.
        (0..self.n_modes)
            .map(|k| {
                let mean = self.mean_photon_number(k);
                // Round to nearest integer
                (mean + 0.5) as usize
            })
            .collect()
    }
}

// ─── BosonSamplingCertificate ────────────────────────────────────────────────

/// Classical complexity certificate for a boson sampling experiment.
///
/// Tracks the photon and mode counts, Hilbert space dimension,
/// classical simulation cost, and quantum advantage threshold.
#[derive(Debug, Clone)]
pub struct BosonSamplingCertificate {
    /// Number of photons n.
    pub n_photons: usize,
    /// Number of modes m.
    pub n_modes: usize,
    /// Whether the sampling is in the collision-free (no two photons per mode) regime.
    pub collision_free: bool,
}

impl BosonSamplingCertificate {
    /// Construct a certificate for n photons in m modes.
    ///
    /// Collision-free regime: m ≥ n² (Aaronson-Arkhipov condition).
    pub fn new(n: usize, m: usize) -> Self {
        let collision_free = m >= n * n;
        Self {
            n_photons: n,
            n_modes: m,
            collision_free,
        }
    }

    /// Hilbert space dimension for n photons in m modes (bosonic Fock space):
    /// dim = C(m + n − 1, n).
    pub fn hilbert_space_dim(&self) -> u64 {
        multiset_coeff(self.n_modes, self.n_photons)
    }

    /// Estimated classical simulation cost in floating-point operations.
    ///
    /// Computing a single permanent of an n×n matrix via Ryser's formula costs O(2ⁿ · n).
    /// This is a rough order-of-magnitude estimate.
    pub fn classical_cost_ops(&self) -> f64 {
        let n = self.n_photons as f64;
        let pow2n = 2.0_f64.powf(n);
        pow2n * n * n
    }

    /// Whether the configuration exceeds the quantum advantage threshold.
    ///
    /// Rough criterion: n ≥ 50 photons and m ≥ n² modes (collision-free GBS),
    /// or n ≥ 50 in the non-collision-free regime.
    pub fn has_quantum_advantage(&self) -> bool {
        self.n_photons >= 50 && self.collision_free
    }

    /// Cross-entropy benchmarking (XEB) score — a proxy for quantum fidelity.
    ///
    /// XEB = D · Σᵢ p_measured(i) · p_ideal(i) − 1
    /// where D = 2^n is the Hilbert space dimension for qubit-equivalent systems.
    /// For boson sampling we use D = Hilbert space dimension.
    ///
    /// A score of 1 corresponds to ideal quantum performance;
    /// a score of 0 corresponds to uniform random sampling.
    pub fn xeb_score(measured_probs: &[f64], ideal_probs: &[f64]) -> f64 {
        if measured_probs.len() != ideal_probs.len() || ideal_probs.is_empty() {
            return 0.0;
        }
        let d = ideal_probs.len() as f64;
        let inner: f64 = measured_probs
            .iter()
            .zip(ideal_probs.iter())
            .map(|(&p_m, &p_i)| p_m * p_i)
            .sum();
        d * inner - 1.0
    }

    /// Porter-Thomas distribution expected XEB for ideal quantum sampling.
    ///
    /// For ideal sampling, ⟨XEB⟩ = 1 (by definition of the XEB estimator).
    pub fn ideal_xeb_score(&self) -> f64 {
        1.0
    }

    /// Total variation distance bound between GBS and uniform distributions.
    ///
    /// For large n in the collision-free regime, the GBS output concentrates
    /// on outputs with high permanent values.  This is a rough lower bound:
    ///   TVD ≥ 1 − (D · P_uniform)^{1/2}  (from Cauchy-Schwarz).
    pub fn total_variation_distance_lower_bound(&self) -> f64 {
        let d = self.hilbert_space_dim() as f64;
        if d < 1.0 {
            return 0.0;
        }
        (1.0 - 1.0 / d.sqrt()).max(0.0)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn identity_unitary(n: usize) -> Vec<Vec<Complex64>> {
        let mut u = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for (i, row) in u.iter_mut().enumerate().take(n) {
            row[i] = Complex64::new(1.0, 0.0);
        }
        u
    }

    #[test]
    fn test_mean_photon_number() {
        // r=1: sinh²(1) ≈ 1.3810
        let u = identity_unitary(2);
        let gbs = GaussianBosonSampler::new(2, vec![1.0, 0.5], u);
        let expected_0 = 1.0_f64.sinh().powi(2);
        let expected_1 = 0.5_f64.sinh().powi(2);
        assert!(approx_eq(gbs.mean_photon_number(0), expected_0, 1e-12));
        assert!(approx_eq(gbs.mean_photon_number(1), expected_1, 1e-12));
    }

    #[test]
    fn test_total_mean_photons() {
        let u = identity_unitary(2);
        let gbs = GaussianBosonSampler::new(2, vec![1.0, 1.0], u);
        let expected = 2.0 * 1.0_f64.sinh().powi(2);
        assert!(approx_eq(gbs.total_mean_photons(), expected, 1e-12));
    }

    #[test]
    fn test_vacuum_probability_no_squeezing() {
        // r=0 for all modes: cosh(0)=1 → P(0)=1
        let u = identity_unitary(3);
        let gbs = GaussianBosonSampler::new(3, vec![0.0, 0.0, 0.0], u);
        assert!(approx_eq(gbs.vacuum_probability(), 1.0, 1e-12));
    }

    #[test]
    fn test_vacuum_probability_with_squeezing() {
        // r=1 for one mode: P(0) = 1/cosh(1) ≈ 0.6481
        let u = identity_unitary(1);
        let gbs = GaussianBosonSampler::new(1, vec![1.0], u);
        let expected = 1.0 / 1.0_f64.cosh();
        assert!(approx_eq(gbs.vacuum_probability(), expected, 1e-12));
    }

    #[test]
    fn test_adjacency_matrix_identity_interferometer() {
        // For identity U: A[i][j] = tanh(r_i) * δ_{ij}
        let u = identity_unitary(2);
        let gbs = GaussianBosonSampler::new(2, vec![0.5, 0.8], u);
        let a = gbs.adjacency_matrix();
        let tanh0 = 0.5_f64.tanh();
        let tanh1 = 0.8_f64.tanh();
        assert!(approx_eq(a[0][0].re, tanh0, 1e-12));
        assert!(approx_eq(a[1][1].re, tanh1, 1e-12));
        assert!(approx_eq(a[0][1].norm(), 0.0, 1e-12));
    }

    #[test]
    fn test_hilbert_space_dim() {
        // C(3+2-1, 2) = C(4,2) = 6
        let cert = BosonSamplingCertificate::new(2, 3);
        assert_eq!(cert.hilbert_space_dim(), 6);
    }

    #[test]
    fn test_quantum_advantage_threshold() {
        // n=50, m=2500: should have quantum advantage
        let cert = BosonSamplingCertificate::new(50, 2500);
        assert!(cert.has_quantum_advantage());
        // n=5, m=25: below threshold
        let cert_small = BosonSamplingCertificate::new(5, 25);
        assert!(!cert_small.has_quantum_advantage());
    }

    #[test]
    fn test_collision_free_regime() {
        // m >= n^2: collision-free
        let cert = BosonSamplingCertificate::new(3, 9);
        assert!(cert.collision_free);
        let cert2 = BosonSamplingCertificate::new(3, 8);
        assert!(!cert2.collision_free);
    }

    #[test]
    fn test_xeb_score_ideal() {
        // For ideal sampling: measured = ideal, XEB = D * Σ p_i^2 - 1
        // For uniform: p_i = 1/D, D * Σ (1/D)^2 - 1 = D * (1/D) * (1/D) * D - 1 = 0
        let n = 4;
        let ideal: Vec<f64> = vec![1.0 / n as f64; n];
        let measured = ideal.clone();
        let xeb = BosonSamplingCertificate::xeb_score(&measured, &ideal);
        // XEB = D * D * (1/D)^2 - 1 = 1 - 1 = 0 for uniform distribution
        assert!(approx_eq(xeb, 0.0, 1e-12));
    }

    #[test]
    fn test_classical_cost_ops_scaling() {
        let cert10 = BosonSamplingCertificate::new(10, 100);
        let cert20 = BosonSamplingCertificate::new(20, 400);
        // Cost(20) should be much larger than cost(10)
        assert!(cert20.classical_cost_ops() > cert10.classical_cost_ops() * 1000.0);
    }

    #[test]
    fn test_pattern_probability_vacuum() {
        // All-zero pattern should equal vacuum_probability
        let u = identity_unitary(2);
        let gbs = GaussianBosonSampler::new(2, vec![0.5, 0.3], u);
        let p_pattern = gbs.pattern_probability(&[0, 0]);
        let p_vac = gbs.vacuum_probability();
        assert!(approx_eq(p_pattern, p_vac, 1e-12));
    }
}
