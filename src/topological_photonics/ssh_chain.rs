//! SSH (Su-Schrieffer-Heeger) model for 1D topological photonics.
//!
//! The SSH model describes a 1D chain with alternating hopping amplitudes t₁
//! (intra-cell) and t₂ (inter-cell).  The Bloch Hamiltonian is:
//!
//!   H(k) = (t₁ + t₂ e^{-ik}) σ₊ + (t₁ + t₂ e^{ik}) σ₋
//!
//! The topological invariant is the winding number W (equivalently, Zak phase γ = πW):
//!   W = 0  →  trivial    (t₁ > t₂)
//!   W = 1  →  topological (t₂ > t₁)
//!
//! In the topological phase a finite chain hosts two zero-energy edge modes
//! localised at the two ends with localisation length ξ = −1/ln(t₁/t₂).

use std::f64::consts::PI;

// ─── SSH chain ─────────────────────────────────────────────────────────────────

/// 1D Su-Schrieffer-Heeger (SSH) chain.
///
/// Bulk Hamiltonian (in k-space):
///   H(k) = [0, h(k); h*(k), 0]  with  h(k) = t₁ + t₂ e^{-ik}
///
/// Band energies: E±(k) = ±|h(k)| = ±√(t₁² + t₂² + 2 t₁ t₂ cos k)
#[derive(Debug, Clone)]
pub struct SshChain {
    /// Intra-cell hopping amplitude t₁ (same units as energy).
    pub t1: f64,
    /// Inter-cell hopping amplitude t₂.
    pub t2: f64,
    /// Number of unit cells N (chain has 2N sites).
    pub n_cells: usize,
}

impl SshChain {
    /// Create an SSH chain with given hopping parameters and number of unit cells.
    pub fn new(t1: f64, t2: f64, n_cells: usize) -> Self {
        Self { t1, t2, n_cells }
    }

    /// Returns `true` if the chain is in the topological phase (t₂ > t₁).
    ///
    /// Equivalently, the winding number W = 1 iff `t2 > t1`.
    pub fn is_topological(&self) -> bool {
        self.t2 > self.t1
    }

    /// Winding number W of the SSH chain.
    ///
    /// Computed by tracking the phase angle φ(k) = arg[h(k)] = arg[t₁ + t₂ e^{ik}]
    /// around the full Brillouin zone k ∈ [−π, π] and counting the net winding:
    ///   W = (1/2π) ∮ dφ/dk dk
    ///
    /// W = 0 (trivial) when t₁ > t₂; W = 1 (topological) when t₂ > t₁.
    pub fn winding_number(&self) -> i32 {
        let n_k = 1000;
        // Start angle at k = -π
        let k_start = -PI;
        let hx_start = self.t1 + self.t2 * k_start.cos();
        let hy_start = self.t2 * k_start.sin();
        let mut angle_prev = hy_start.atan2(hx_start);
        let mut total_winding = 0.0_f64;

        for i in 0..n_k {
            let k = -PI + 2.0 * PI * (i + 1) as f64 / n_k as f64;
            let hx = self.t1 + self.t2 * k.cos();
            let hy = self.t2 * k.sin();
            let angle = hy.atan2(hx);
            let mut delta = angle - angle_prev;
            // Wrap Δφ into (−π, π] to handle branch cuts
            while delta > PI {
                delta -= 2.0 * PI;
            }
            while delta < -PI {
                delta += 2.0 * PI;
            }
            total_winding += delta;
            angle_prev = angle;
        }
        (total_winding / (2.0 * PI)).round() as i32
    }

    /// Zak phase γ (rad): 0 for trivial, π for topological.
    ///
    /// γ = π × W
    pub fn zak_phase(&self) -> f64 {
        PI * self.winding_number() as f64
    }

    /// Band energies E±(k) at a given crystal momentum k.
    ///
    /// E±(k) = ±√(t₁² + t₂² + 2 t₁ t₂ cos k)
    ///
    /// Returns `(E_minus, E_plus)`.
    pub fn band_energies(&self, k: f64) -> (f64, f64) {
        let e_sq = self.t1 * self.t1 + self.t2 * self.t2 + 2.0 * self.t1 * self.t2 * k.cos();
        let e = e_sq.max(0.0).sqrt();
        (-e, e)
    }

    /// Full Brillouin-zone band structure sampled at `n_k` k-points.
    ///
    /// Returns a vector of `(k, E_minus, E_plus)` tuples with k ∈ [−π, π].
    pub fn dispersion(&self, n_k: usize) -> Vec<(f64, f64, f64)> {
        if n_k == 0 {
            return Vec::new();
        }
        (0..n_k)
            .map(|i| {
                let k = -PI + 2.0 * PI * i as f64 / (n_k.saturating_sub(1).max(1)) as f64;
                let (em, ep) = self.band_energies(k);
                (k, em, ep)
            })
            .collect()
    }

    /// Band gap at the zone boundary k = π: Eg = 2|t₁ − t₂|.
    pub fn band_gap(&self) -> f64 {
        2.0 * (self.t1 - self.t2).abs()
    }

    /// Eigenvalues of the finite SSH tight-binding chain.
    ///
    /// Constructs the 2N×2N symmetric tridiagonal Hamiltonian and diagonalises
    /// it using Sturm-sequence bisection (exact for symmetric tridiagonal matrices).
    ///
    /// Returns the 2N eigenvalues in ascending order.
    pub fn finite_chain_energies(&self) -> Vec<f64> {
        let n = 2 * self.n_cells;
        if n == 0 {
            return Vec::new();
        }
        // Diagonal is all zeros for the SSH model (equal on-site energies)
        let diag = vec![0.0f64; n];
        // Off-diagonal: t₁ for intra-cell bonds, t₂ for inter-cell bonds
        let offdiag: Vec<f64> = (0..n.saturating_sub(1))
            .map(|i| if i % 2 == 0 { self.t1 } else { self.t2 })
            .collect();
        sturm_bisect(&diag, &offdiag)
    }

    /// Edge state energies from the finite chain spectrum.
    ///
    /// In the topological phase the two eigenvalues closest to E = 0 correspond
    /// to the left and right edge modes.  Returns `Some((E_left, E_right))` when
    /// the chain is topological with at least two eigenvalues; `None` otherwise.
    pub fn edge_state_energies(&self) -> Option<(f64, f64)> {
        if !self.is_topological() {
            return None;
        }
        let mut energies = self.finite_chain_energies();
        if energies.len() < 2 {
            return None;
        }
        // Sort by |E| to find the two states closest to zero
        energies.sort_by(|a, b| {
            a.abs()
                .partial_cmp(&b.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Some((energies[0], energies[1]))
    }

    /// Localisation length ξ of the edge state (in units of unit cells).
    ///
    /// ξ = −1 / ln(t₁/t₂)
    ///
    /// Returns `None` for the trivial phase (t₁ ≥ t₂) where the expression
    /// diverges or is undefined.
    pub fn edge_state_localization(&self) -> Option<f64> {
        if !self.is_topological() {
            return None;
        }
        let ratio = self.t1 / self.t2;
        if ratio <= 0.0 || ratio >= 1.0 {
            return None;
        }
        Some(-1.0 / ratio.ln())
    }

    /// Electric polarisation P = W/2 (in units of the elementary charge × lattice constant).
    ///
    /// P = γ/(2π) = W/2.  Trivial: P = 0; topological: P = 1/2.
    pub fn polarization(&self) -> f64 {
        self.winding_number() as f64 / 2.0
    }

    /// Returns `true` if a domain-wall interface state exists between this SSH
    /// chain (left domain) and a right domain with hopping (`t1_right`, `t2_right`).
    ///
    /// An interface state is topologically protected whenever the winding numbers
    /// of the two domains differ.
    pub fn interface_state_exists(&self, t1_right: f64, t2_right: f64) -> bool {
        let w_left = self.winding_number();
        let right = SshChain::new(t1_right, t2_right, self.n_cells);
        let w_right = right.winding_number();
        w_left != w_right
    }
}

// ─── Photonic SSH resonator chain ──────────────────────────────────────────────

/// Photonic SSH chain realised with coupled optical resonators.
///
/// Each resonator has the same resonance frequency ω₀ and the coupling
/// alternates between κ₁ (intra-cell, e.g. small gap) and κ₂ (inter-cell,
/// e.g. large gap).  The physics maps exactly onto the electronic SSH model
/// with t₁ → κ₁ and t₂ → κ₂.
#[derive(Debug, Clone)]
pub struct PhotonicSshResonator {
    /// Underlying SSH tight-binding chain.
    pub ssh: SshChain,
    /// Common resonance frequency ω₀ of all resonators (rad/s).
    pub resonance_freq: f64,
}

impl PhotonicSshResonator {
    /// Construct a photonic SSH resonator chain.
    ///
    /// # Arguments
    /// * `kappa1`         – intra-cell coupling (rad/s)
    /// * `kappa2`         – inter-cell coupling (rad/s)
    /// * `n_cells`        – number of unit cells
    /// * `resonance_freq` – on-site resonance ω₀ (rad/s)
    pub fn new(kappa1: f64, kappa2: f64, n_cells: usize, resonance_freq: f64) -> Self {
        Self {
            ssh: SshChain::new(kappa1, kappa2, n_cells),
            resonance_freq,
        }
    }

    /// Frequencies of transmission minima (band-edge dips) in the output spectrum.
    ///
    /// At the zone boundary (k = π) the two band edges occur at:
    ///   ω± = ω₀ ± |t₁ − t₂|
    ///
    /// Returns `[ω_lower_edge, ω_upper_edge]`.
    pub fn transmission_dip_frequencies(&self) -> Vec<f64> {
        let half_gap = (self.ssh.t1 - self.ssh.t2).abs();
        vec![
            self.resonance_freq - half_gap,
            self.resonance_freq + half_gap,
        ]
    }

    /// Mid-gap frequency (resonance of the topological edge state).
    ///
    /// For the SSH model the edge state sits exactly at ω₀.
    pub fn midgap_frequency(&self) -> f64 {
        self.resonance_freq
    }

    /// `true` if the resonator chain hosts topological edge states.
    pub fn has_edge_states(&self) -> bool {
        self.ssh.is_topological()
    }

    /// Band gap: Δω = 2|κ₁ − κ₂|.
    pub fn band_gap(&self) -> f64 {
        self.ssh.band_gap()
    }
}

// ─── Internal: Sturm-sequence bisection ────────────────────────────────────────

/// Count the number of eigenvalues of a symmetric tridiagonal matrix that are
/// strictly less than `mu`, using the Sturm sequence recurrence.
///
/// `diag` are the diagonal elements, `offdiag` the sub-diagonal (length n−1).
fn sturm_count(diag: &[f64], offdiag: &[f64], mu: f64) -> usize {
    let n = diag.len();
    if n == 0 {
        return 0;
    }
    let mut count = 0usize;
    let mut prev = diag[0] - mu;
    if prev < 0.0 {
        count += 1;
    }
    for i in 1..n {
        let d = diag[i] - mu;
        let o = offdiag[i - 1];
        // Sturm recurrence; avoid division by near-zero
        let curr = if prev.abs() < 1e-30 {
            d - o * o * 1e30_f64.copysign(1.0)
        } else {
            d - o * o / prev
        };
        if curr < 0.0 {
            count += 1;
        }
        prev = curr;
    }
    count
}

/// Bisection eigenvalue solver for symmetric tridiagonal matrices.
///
/// Uses Gershgorin circle theorem for initial bounds and Sturm-sequence
/// counting for bisection.  Returns all eigenvalues in ascending order.
fn sturm_bisect(diag: &[f64], offdiag: &[f64]) -> Vec<f64> {
    let n = diag.len();
    if n == 0 {
        return Vec::new();
    }

    // Gershgorin bounds: each eigenvalue lies in [d_i − r_i, d_i + r_i]
    // where r_i = |off[i-1]| + |off[i]|.
    let mut mu_min = f64::INFINITY;
    let mut mu_max = f64::NEG_INFINITY;
    for i in 0..n {
        let r = if i == 0 {
            offdiag.first().map_or(0.0, |x| x.abs())
        } else if i == n - 1 {
            offdiag.last().map_or(0.0, |x| x.abs())
        } else {
            offdiag[i - 1].abs() + offdiag[i].abs()
        };
        let lo = diag[i] - r;
        let hi = diag[i] + r;
        if lo < mu_min {
            mu_min = lo;
        }
        if hi > mu_max {
            mu_max = hi;
        }
    }
    mu_min -= 1.0;
    mu_max += 1.0;

    let mut eigenvalues = Vec::with_capacity(n);
    for k in 0..n {
        // Find k-th eigenvalue (0-indexed) by bisection: locate interval where
        // the Sturm count crosses from ≤ k to > k.
        let mut lo = mu_min;
        let mut hi = mu_max;
        for _ in 0..60 {
            let mid = (lo + hi) / 2.0;
            if sturm_count(diag, offdiag, mid) <= k {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        eigenvalues.push((lo + hi) / 2.0);
    }
    eigenvalues
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssh_winding_trivial() {
        let ssh = SshChain::new(1.0, 0.5, 10);
        assert_eq!(ssh.winding_number(), 0);
        assert!(!ssh.is_topological());
    }

    #[test]
    fn ssh_winding_topological() {
        let ssh = SshChain::new(0.5, 1.0, 10);
        assert_eq!(ssh.winding_number(), 1);
        assert!(ssh.is_topological());
    }

    #[test]
    fn ssh_zak_phase_trivial_is_zero() {
        let ssh = SshChain::new(1.0, 0.5, 5);
        assert!(ssh.zak_phase().abs() < 1e-9);
    }

    #[test]
    fn ssh_zak_phase_topological_is_pi() {
        let ssh = SshChain::new(0.5, 1.0, 5);
        assert!((ssh.zak_phase() - PI).abs() < 1e-9);
    }

    #[test]
    fn ssh_band_gap_correct() {
        let ssh = SshChain::new(0.5, 1.0, 5);
        let expected = 2.0 * (0.5_f64 - 1.0).abs();
        assert!((ssh.band_gap() - expected).abs() < 1e-12);
    }

    #[test]
    fn ssh_band_energies_symmetric() {
        let ssh = SshChain::new(0.6, 0.8, 5);
        let (em, ep) = ssh.band_energies(0.3);
        assert!(
            (em + ep).abs() < 1e-12,
            "E+ + E- should be zero, got {}",
            em + ep
        );
        assert!(ep >= 0.0);
    }

    #[test]
    fn ssh_finite_chain_eigenvalues_count() {
        let ssh = SshChain::new(0.5, 1.0, 6);
        let eigs = ssh.finite_chain_energies();
        assert_eq!(eigs.len(), 12); // 2 × n_cells
    }

    #[test]
    fn ssh_edge_state_energies_near_zero() {
        let ssh = SshChain::new(0.1, 1.0, 10);
        let edge = ssh.edge_state_energies();
        assert!(edge.is_some(), "Topological chain should have edge states");
        let (e1, e2) = edge.unwrap_or((1.0, 1.0));
        assert!(
            e1.abs() < 0.1,
            "Edge state energy |E1| should be small, got {e1}"
        );
        assert!(
            e2.abs() < 0.1,
            "Edge state energy |E2| should be small, got {e2}"
        );
    }

    #[test]
    fn ssh_no_edge_state_trivial() {
        let ssh = SshChain::new(1.0, 0.5, 10);
        let edge = ssh.edge_state_energies();
        assert!(
            edge.is_none(),
            "Trivial chain should not report edge states"
        );
    }

    #[test]
    fn ssh_localization_length_topological() {
        let ssh = SshChain::new(0.3, 1.0, 10);
        let xi = ssh.edge_state_localization();
        assert!(xi.is_some());
        assert!(xi.unwrap_or(0.0) > 0.0);
    }

    #[test]
    fn ssh_localization_length_trivial_is_none() {
        let ssh = SshChain::new(1.0, 0.3, 10);
        assert!(ssh.edge_state_localization().is_none());
    }

    #[test]
    fn ssh_polarization_half_when_topological() {
        let ssh = SshChain::new(0.5, 1.0, 5);
        assert!((ssh.polarization() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn ssh_interface_state_different_topologies() {
        let ssh_left = SshChain::new(0.5, 1.0, 5); // topological
        assert!(ssh_left.interface_state_exists(1.0, 0.5)); // right is trivial
        assert!(!ssh_left.interface_state_exists(0.5, 1.0)); // right is also topological
    }

    #[test]
    fn photonic_ssh_resonator_dip_frequencies() {
        let res = PhotonicSshResonator::new(0.5, 1.0, 5, 200e12);
        let dips = res.transmission_dip_frequencies();
        assert_eq!(dips.len(), 2);
        assert!(dips[0] < res.midgap_frequency());
        assert!(dips[1] > res.midgap_frequency());
    }

    #[test]
    fn photonic_ssh_resonator_has_edge_states() {
        let res = PhotonicSshResonator::new(0.5, 1.0, 5, 200e12);
        assert!(res.has_edge_states());
    }
}
