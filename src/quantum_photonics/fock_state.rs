//! Multi-mode Fock (photon number) state algebra.
//!
//! A Fock state |n₁, n₂, …, nₘ⟩ describes the photon occupation of m optical modes.
//! Superpositions of Fock states, partial-trace, entanglement entropy, and PNRD
//! measurement back-action are provided.

use num_complex::Complex64;
use std::f64::consts::PI;

// ─── Combinatorics helpers ────────────────────────────────────────────────────

/// Natural log of n! using exact summation up to n≤20, Stirling otherwise.
pub(crate) fn ln_factorial(n: usize) -> f64 {
    match n {
        0 | 1 => 0.0,
        2..=20 => {
            let mut acc = 0.0_f64;
            for k in 2..=n {
                acc += (k as f64).ln();
            }
            acc
        }
        _ => {
            let nf = n as f64;
            nf * nf.ln() - nf + 0.5 * (2.0 * PI * nf).ln() + 1.0 / (12.0 * nf)
        }
    }
}

/// Exact factorial for small n (returns None if n > 20 to avoid overflow).
#[allow(dead_code)]
pub(crate) fn factorial_small(n: usize) -> Option<u64> {
    const TABLE: [u64; 21] = [
        1,
        1,
        2,
        6,
        24,
        120,
        720,
        5040,
        40320,
        362880,
        3628800,
        39916800,
        479001600,
        6227020800,
        87178291200,
        1307674368000,
        20922789888000,
        355687428096000,
        6402373705728000,
        121645100408832000,
        2432902008176640000,
    ];
    TABLE.get(n).copied()
}

/// Binomial coefficient C(n, k).
pub(crate) fn binom(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k); // use symmetry
    let mut result = 1u64;
    for i in 0..k {
        result = result.saturating_mul(n as u64 - i as u64) / (i as u64 + 1);
    }
    result
}

// ─── BellState ───────────────────────────────────────────────────────────────

/// Polarisation-encoded Bell states in the dual-rail encoding.
///
/// Each state is encoded into four modes: (H₁, V₁, H₂, V₂).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BellState {
    /// (|H₁V₂⟩ + |V₁H₂⟩) / √2  — even-parity singlet
    PhiPlus,
    /// (|H₁V₂⟩ - |V₁H₂⟩) / √2
    PhiMinus,
    /// (|H₁H₂⟩ + |V₁V₂⟩) / √2
    PsiPlus,
    /// (|H₁H₂⟩ - |V₁V₂⟩) / √2
    PsiMinus,
}

// ─── FockState ───────────────────────────────────────────────────────────────

/// Multi-mode photon number state |n₁, n₂, …, nₘ⟩.
///
/// Each entry in `occupation` is the photon count in that mode.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MultiModeFockState {
    /// Photon occupation number per mode.
    pub occupation: Vec<usize>,
}

impl MultiModeFockState {
    /// Construct from an occupation vector.
    pub fn new(occupation: Vec<usize>) -> Self {
        Self { occupation }
    }

    /// Number of modes.
    #[inline]
    pub fn n_modes(&self) -> usize {
        self.occupation.len()
    }

    /// Total photon number N = Σᵢ nᵢ.
    pub fn total_photons(&self) -> usize {
        self.occupation.iter().sum()
    }

    /// Vacuum state |0, 0, …, 0⟩ with `n_modes` modes.
    pub fn vacuum(n_modes: usize) -> Self {
        Self {
            occupation: vec![0; n_modes],
        }
    }

    /// Single-photon state |0, …, 1, …, 0⟩ with a photon in `mode`.
    ///
    /// Modes are zero-indexed. `n_modes` is the total number of modes.
    ///
    /// # Panics (debug only)
    /// Panics in debug builds if `mode >= n_modes`.
    pub fn single_photon(mode: usize, n_modes: usize) -> Self {
        debug_assert!(mode < n_modes, "mode index {mode} >= n_modes {n_modes}");
        let mut occ = vec![0usize; n_modes];
        if mode < n_modes {
            occ[mode] = 1;
        }
        Self { occupation: occ }
    }

    /// Bell basis states in the dual-rail encoding.
    ///
    /// Modes are ordered (H_qubit0, V_qubit0, H_qubit1, V_qubit1), four modes total.
    ///
    /// Returns a superposition `Vec<(amplitude, state)>`.
    pub fn bell_basis_state(kind: BellState) -> Vec<(Complex64, MultiModeFockState)> {
        let inv_sqrt2 = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        let neg_inv_sqrt2 = -inv_sqrt2;

        // Dual-rail: mode 0=H0, 1=V0, 2=H1, 3=V1
        let hh = MultiModeFockState::new(vec![1, 0, 1, 0]); // |H0 H1⟩
        let vv = MultiModeFockState::new(vec![0, 1, 0, 1]); // |V0 V1⟩
        let hv = MultiModeFockState::new(vec![1, 0, 0, 1]); // |H0 V1⟩
        let vh = MultiModeFockState::new(vec![0, 1, 1, 0]); // |V0 H1⟩

        match kind {
            BellState::PhiPlus => vec![(inv_sqrt2, hh), (inv_sqrt2, vv)],
            BellState::PhiMinus => vec![(inv_sqrt2, hh), (neg_inv_sqrt2, vv)],
            BellState::PsiPlus => vec![(inv_sqrt2, hv), (inv_sqrt2, vh)],
            BellState::PsiMinus => vec![(inv_sqrt2, hv), (neg_inv_sqrt2, vh)],
        }
    }
}

// ─── FockSuperposition ───────────────────────────────────────────────────────

/// Coherent superposition Σᵢ αᵢ |ψᵢ⟩ of multi-mode Fock states.
#[derive(Debug, Clone)]
pub struct FockSuperposition {
    /// (amplitude, Fock state) pairs.  May contain duplicate states.
    pub terms: Vec<(Complex64, MultiModeFockState)>,
}

impl FockSuperposition {
    /// Construct from a list of (amplitude, state) pairs.
    pub fn new(terms: Vec<(Complex64, MultiModeFockState)>) -> Self {
        Self { terms }
    }

    /// ‖ψ‖ = √(Σᵢ |αᵢ|²) — does not account for state overlaps,
    /// but since Fock states with different occupation are orthogonal, this
    /// equals the true norm when all states are distinct.
    pub fn norm(&self) -> f64 {
        self.terms
            .iter()
            .map(|(a, _)| a.norm_sqr())
            .sum::<f64>()
            .sqrt()
    }

    /// Normalise in place so ‖ψ‖ = 1.
    pub fn normalize(&mut self) {
        let n = self.norm();
        if n > f64::EPSILON {
            for (amp, _) in &mut self.terms {
                *amp /= n;
            }
        }
    }

    /// Number of modes (from first term; 0 if empty).
    pub fn n_modes(&self) -> usize {
        self.terms.first().map(|(_, s)| s.n_modes()).unwrap_or(0)
    }

    /// Inner product ⟨self|other⟩ = Σᵢⱼ α*ᵢ βⱼ ⟨ψᵢ|φⱼ⟩.
    /// Since Fock states are orthonormal, this sums only matching occupation vectors.
    pub fn inner_product(&self, other: &Self) -> Complex64 {
        let mut result = Complex64::new(0.0, 0.0);
        for (a, sa) in &self.terms {
            for (b, sb) in &other.terms {
                if sa.occupation == sb.occupation {
                    result += a.conj() * b;
                }
            }
        }
        result
    }

    /// Probability of measuring a specific Fock state: P(s) = |⟨s|ψ⟩|².
    pub fn probability(&self, state: &MultiModeFockState) -> f64 {
        let amp: Complex64 = self
            .terms
            .iter()
            .filter(|(_, s)| s.occupation == state.occupation)
            .map(|(a, _)| *a)
            .sum();
        amp.norm_sqr()
    }

    /// Reduced density matrix ρ_A = Tr_B(|ψ⟩⟨ψ|) for a subset of modes.
    ///
    /// `keep_modes` is the list of mode indices to retain (mode A).
    /// The remaining modes are traced out (mode B).
    ///
    /// The returned matrix is indexed by distinct occupation vectors of the kept modes.
    /// Rows/columns correspond to the distinct marginal Fock configurations found in the state,
    /// in the order they first appear when scanning `self.terms`.
    pub fn partial_trace(&self, keep_modes: &[usize]) -> Vec<Vec<Complex64>> {
        // Enumerate all distinct marginal configurations (mode-A occupation patterns)
        let mut configs: Vec<Vec<usize>> = Vec::new();
        for (_, state) in &self.terms {
            let marginal: Vec<usize> = keep_modes
                .iter()
                .map(|&m| *state.occupation.get(m).unwrap_or(&0))
                .collect();
            if !configs.contains(&marginal) {
                configs.push(marginal);
            }
        }
        let dim = configs.len();
        let mut rho = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];

        // ρ_A[i,j] = Σ_{b} ⟨φᵢ, b|ψ⟩⟨ψ|φⱼ, b⟩
        // where b ranges over all environment configurations
        // We enumerate all environment (mode-B) configs present in the state.
        let all_modes = self.n_modes();
        let env_modes: Vec<usize> = (0..all_modes).filter(|m| !keep_modes.contains(m)).collect();

        let mut env_configs: Vec<Vec<usize>> = Vec::new();
        for (_, state) in &self.terms {
            let env_occ: Vec<usize> = env_modes
                .iter()
                .map(|&m| *state.occupation.get(m).unwrap_or(&0))
                .collect();
            if !env_configs.contains(&env_occ) {
                env_configs.push(env_occ);
            }
        }

        for env_occ in &env_configs {
            for (i, cfg_i) in configs.iter().enumerate() {
                // amplitude for system config i + env config
                let amp_i: Complex64 = self
                    .terms
                    .iter()
                    .filter(|(_, s)| {
                        let sys: Vec<usize> = keep_modes.iter().map(|&m| s.occupation[m]).collect();
                        let env: Vec<usize> = env_modes.iter().map(|&m| s.occupation[m]).collect();
                        sys == *cfg_i && env == *env_occ
                    })
                    .map(|(a, _)| *a)
                    .sum();

                for (j, cfg_j) in configs.iter().enumerate() {
                    let amp_j: Complex64 = self
                        .terms
                        .iter()
                        .filter(|(_, s)| {
                            let sys: Vec<usize> =
                                keep_modes.iter().map(|&m| s.occupation[m]).collect();
                            let env: Vec<usize> =
                                env_modes.iter().map(|&m| s.occupation[m]).collect();
                            sys == *cfg_j && env == *env_occ
                        })
                        .map(|(a, _)| *a)
                        .sum();

                    rho[i][j] += amp_i * amp_j.conj();
                }
            }
        }
        rho
    }

    /// Von Neumann entropy S = -Tr(ρ_A ln ρ_A) of the reduced state on `mode_a`.
    ///
    /// Computed from the eigenvalues of ρ_A via power-iteration for small matrices,
    /// using the Jacobi algorithm for the Hermitian matrix.
    pub fn entanglement_entropy(&self, mode_a: &[usize]) -> f64 {
        let rho = self.partial_trace(mode_a);
        let dim = rho.len();
        if dim == 0 {
            return 0.0;
        }
        // Eigenvalues of the Hermitian density matrix via Jacobi diagonalisation
        let eigenvalues = jacobi_eigenvalues_hermitian(&rho);
        eigenvalues
            .iter()
            .filter(|&&ev| ev > 1e-15)
            .map(|&ev| -ev * ev.ln())
            .sum()
    }
}

/// Jacobi eigenvalue algorithm for small Hermitian matrices.
/// Returns real eigenvalues (imaginary parts are zero for Hermitian).
fn jacobi_eigenvalues_hermitian(a: &[Vec<Complex64>]) -> Vec<f64> {
    let n = a.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![a[0][0].re];
    }
    // Work on the real symmetric part for the diagonal entries; for truly Hermitian
    // matrices we use a real Jacobi sweep on Re/Im components packed together.
    // For the sizes we expect (≤ few dozen), a power-method approximation suffices.
    // We implement the real Jacobi algorithm on the real part since ρ is Hermitian
    // and its eigenvalues are real.
    let mut diag: Vec<f64> = (0..n).map(|i| a[i][i].re).collect();
    // Off-diagonal entries — use a simple trace-based bound for trivially diagonal matrices
    let off_diag_sq: f64 = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .filter(|&(i, j)| i != j)
        .map(|(i, j)| a[i][j].norm_sqr())
        .sum();
    if off_diag_sq < 1e-28 {
        // Already diagonal
        return diag;
    }
    // Full Jacobi: convert to real symmetric by taking modulus of off-diag (approximate
    // for the entropy computation, since we only need eigenvalues).
    // Build real symmetric matrix |A|_sym
    let mut sym: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j {
                        a[i][j].re
                    } else {
                        // Off-diagonal: use real part (Hermitian: A[i][j] = conj(A[j][i]))
                        a[i][j].re
                    }
                })
                .collect()
        })
        .collect();

    // Classic Jacobi sweeps
    let max_iter = 200 * n * n;
    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0_f64;
        let (mut p, mut q) = (0, 1);
        for (i, row) in sym.iter().enumerate().take(n) {
            for (j, _) in row.iter().enumerate().take(n).skip(i + 1) {
                let v = sym[i][j].abs();
                if v > max_val {
                    max_val = v;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-14 {
            break;
        }
        let theta = 0.5 * (sym[q][q] - sym[p][p]).atan2(2.0 * sym[p][q]);
        let (s, c) = theta.sin_cos();
        // Apply Jacobi rotation
        let app = sym[p][p];
        let aqq = sym[q][q];
        let apq = sym[p][q];
        sym[p][p] = c * c * app + s * s * aqq - 2.0 * s * c * apq;
        sym[q][q] = s * s * app + c * c * aqq + 2.0 * s * c * apq;
        sym[p][q] = 0.0;
        sym[q][p] = 0.0;
        for (r, _) in (0..n).zip(std::iter::repeat(())) {
            if r != p && r != q {
                let arp = sym[r][p];
                let arq = sym[r][q];
                sym[r][p] = c * arp - s * arq;
                sym[p][r] = sym[r][p];
                sym[r][q] = s * arp + c * arq;
                sym[q][r] = sym[r][q];
            }
        }
    }
    diag = (0..n).map(|i| sym[i][i]).collect();
    diag
}

// ─── PnrdMeasurement ─────────────────────────────────────────────────────────

/// Photon-number-resolving detection (PNRD) model.
///
/// Models detection efficiency η and dark count rate (DCR).
pub struct PnrdMeasurement {
    /// Detection efficiency η ∈ [0, 1].
    pub detection_efficiency: f64,
    /// Dark count rate (counts per second).
    pub dark_count_rate: f64,
}

impl PnrdMeasurement {
    /// Construct a PNRD model.
    ///
    /// `eta` is clamped to [0, 1].
    pub fn new(eta: f64, dcr: f64) -> Self {
        Self {
            detection_efficiency: eta.clamp(0.0, 1.0),
            dark_count_rate: dcr.max(0.0),
        }
    }

    /// Probability of detecting `n_detected` photons given `n_incident` incident photons.
    ///
    /// P(n|m) = C(m, n) · ηⁿ · (1−η)^(m−n)  (binomial model, ignoring dark counts).
    pub fn detection_probability(&self, n_detected: usize, n_incident: usize) -> f64 {
        if n_detected > n_incident {
            return 0.0;
        }
        let eta = self.detection_efficiency;
        let c = binom(n_incident, n_detected) as f64;
        let k = n_detected as f64;
        let mk = (n_incident - n_detected) as f64;
        c * eta.powf(k) * (1.0 - eta).powf(mk)
    }

    /// Mean number of detected photons: ⟨n_det⟩ = η · n_incident.
    #[inline]
    pub fn mean_detected(&self, n_incident: usize) -> f64 {
        self.detection_efficiency * n_incident as f64
    }

    /// Post-measurement state after detecting `n_detected` photons in `mode`.
    ///
    /// Projects the superposition onto the subspace where mode `mode` has exactly
    /// `n_detected` photons, then renormalises.
    pub fn post_measurement_state(
        &self,
        state: &FockSuperposition,
        mode: usize,
        n_detected: usize,
    ) -> FockSuperposition {
        // Collect terms consistent with the detection outcome, weighted by detection amplitude.
        let terms: Vec<(Complex64, MultiModeFockState)> = state
            .terms
            .iter()
            .filter_map(|(amp, fock)| {
                let n_in_mode = *fock.occupation.get(mode)?;
                // Weight the amplitude by √P(n_detected | n_in_mode)
                let prob = self.detection_probability(n_detected, n_in_mode);
                if prob < f64::EPSILON {
                    return None;
                }
                Some((*amp * prob.sqrt(), fock.clone()))
            })
            .collect();

        let mut result = FockSuperposition::new(terms);
        result.normalize();
        result
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_vacuum_state() {
        let v = MultiModeFockState::vacuum(4);
        assert_eq!(v.n_modes(), 4);
        assert_eq!(v.total_photons(), 0);
    }

    #[test]
    fn test_single_photon_state() {
        let s = MultiModeFockState::single_photon(2, 5);
        assert_eq!(s.occupation[2], 1);
        assert_eq!(s.total_photons(), 1);
        for i in [0, 1, 3, 4] {
            assert_eq!(s.occupation[i], 0);
        }
    }

    #[test]
    fn test_bell_state_phi_plus_normalised() {
        let terms = MultiModeFockState::bell_basis_state(BellState::PhiPlus);
        let sup = FockSuperposition::new(terms);
        let norm = sup.norm();
        assert!(approx_eq(norm, 1.0, 1e-12));
    }

    #[test]
    fn test_bell_states_orthogonal() {
        let phi_plus =
            FockSuperposition::new(MultiModeFockState::bell_basis_state(BellState::PhiPlus));
        let phi_minus =
            FockSuperposition::new(MultiModeFockState::bell_basis_state(BellState::PhiMinus));
        let ip = phi_plus.inner_product(&phi_minus);
        assert!(ip.norm() < 1e-12);
    }

    #[test]
    fn test_fock_superposition_probability() {
        // |ψ⟩ = (1/√2)|1,0⟩ + (1/√2)|0,1⟩
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let s1 = MultiModeFockState::new(vec![1, 0]);
        let s2 = MultiModeFockState::new(vec![0, 1]);
        let sup = FockSuperposition::new(vec![
            (Complex64::new(inv_sqrt2, 0.0), s1.clone()),
            (Complex64::new(inv_sqrt2, 0.0), s2.clone()),
        ]);
        assert!(approx_eq(sup.probability(&s1), 0.5, 1e-12));
        assert!(approx_eq(sup.probability(&s2), 0.5, 1e-12));
    }

    #[test]
    fn test_pnrd_detection_probability_perfect() {
        let det = PnrdMeasurement::new(1.0, 0.0);
        // Perfect efficiency: P(n|n) = 1 for all n
        assert!(approx_eq(det.detection_probability(3, 3), 1.0, 1e-12));
        assert!(approx_eq(det.detection_probability(2, 3), 0.0, 1e-12));
        assert!(approx_eq(det.detection_probability(4, 3), 0.0, 1e-12));
    }

    #[test]
    fn test_pnrd_detection_probability_50pct() {
        let det = PnrdMeasurement::new(0.5, 0.0);
        // P(1|2) = C(2,1) * 0.5 * 0.5 = 0.5
        assert!(approx_eq(det.detection_probability(1, 2), 0.5, 1e-12));
        // P(0|2) = 0.25
        assert!(approx_eq(det.detection_probability(0, 2), 0.25, 1e-12));
    }

    #[test]
    fn test_pnrd_mean_detected() {
        let det = PnrdMeasurement::new(0.8, 0.0);
        assert!(approx_eq(det.mean_detected(5), 4.0, 1e-12));
    }

    #[test]
    fn test_entanglement_entropy_product_state() {
        // Product state |1,0⟩ ⊗ |0,1⟩ has zero entanglement
        let s = MultiModeFockState::new(vec![1, 0, 0, 1]);
        let sup = FockSuperposition::new(vec![(Complex64::new(1.0, 0.0), s)]);
        let entropy = sup.entanglement_entropy(&[0, 1]);
        assert!(entropy < 1e-10);
    }

    #[test]
    fn test_entanglement_entropy_bell_state() {
        // Bell state Φ+ has log(2) entanglement entropy
        let terms = MultiModeFockState::bell_basis_state(BellState::PhiPlus);
        let sup = FockSuperposition::new(terms);
        let entropy = sup.entanglement_entropy(&[0, 1]);
        // S = -2 * (0.5) * ln(0.5) = ln(2) ≈ 0.693
        assert!(approx_eq(entropy, 2.0_f64.ln(), 0.05));
    }

    #[test]
    fn test_binom() {
        assert_eq!(binom(5, 2), 10);
        assert_eq!(binom(10, 3), 120);
        assert_eq!(binom(0, 0), 1);
        assert_eq!(binom(3, 5), 0);
    }

    #[test]
    fn test_post_measurement_collapses_state() {
        // |ψ⟩ = (1/√2)|2,0⟩ + (1/√2)|0,1⟩; detect 0 photons in mode 0
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let s1 = MultiModeFockState::new(vec![2, 0]);
        let s2 = MultiModeFockState::new(vec![0, 1]);
        let sup = FockSuperposition::new(vec![
            (Complex64::new(inv_sqrt2, 0.0), s1),
            (Complex64::new(inv_sqrt2, 0.0), s2),
        ]);
        let det = PnrdMeasurement::new(1.0, 0.0); // perfect efficiency
        let post = det.post_measurement_state(&sup, 0, 0);
        // After detecting 0 photons in mode 0 with perfect detector,
        // only the |0,1⟩ term survives.
        let s2b = MultiModeFockState::new(vec![0, 1]);
        assert!(approx_eq(post.probability(&s2b), 1.0, 1e-10));
    }
}
