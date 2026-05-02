//! Topological photonics models.
//!
//! Topological photonic systems exhibit edge states that are protected by
//! topological invariants (Chern number, Zak phase, etc.) against perturbations.
//!
//! ## SSH (Su-Schrieffer-Heeger) model
//!
//! The simplest 1D topological model with alternating coupling constants κ₁, κ₂.
//! The Zak phase φ_Zak = 0 (trivial) or π (topological).
//! Edge states exist when κ₁ < κ₂ (dimerization drives topology).
//!
//! Bulk-edge correspondence: topological phase ↔ localised edge mode at boundaries.
//!
//! Photonic SSH: alternating coupled resonator chain with hopping t₁, t₂.
//! Topological invariant: Zak phase = π when t₁ < t₂.

use num_complex::Complex64;
use oxiblas::prelude::{Mat, TridiagEvd};
use std::f64::consts::PI;

use crate::error::OxiPhotonError;

/// SSH (Su-Schrieffer-Heeger) photonic chain.
///
/// Alternating coupling constants κ₁ and κ₂ in a 1D resonator chain.
#[derive(Debug, Clone)]
pub struct SshChain {
    /// Intra-cell coupling κ₁ (Hz or normalised)
    pub kappa1: f64,
    /// Inter-cell coupling κ₂ (Hz or normalised)
    pub kappa2: f64,
    /// On-site resonance frequency ω₀ (rad/s or normalised)
    pub omega0: f64,
    /// Number of unit cells
    pub n_cells: usize,
}

impl SshChain {
    /// Create SSH chain.
    pub fn new(kappa1: f64, kappa2: f64, omega0: f64, n_cells: usize) -> Self {
        Self {
            kappa1,
            kappa2,
            omega0,
            n_cells,
        }
    }

    /// Topologically trivial SSH chain (κ₁ > κ₂).
    pub fn trivial(n_cells: usize) -> Self {
        Self::new(1.0, 0.3, 0.0, n_cells)
    }

    /// Topologically non-trivial SSH chain (κ₁ < κ₂).
    pub fn topological(n_cells: usize) -> Self {
        Self::new(0.3, 1.0, 0.0, n_cells)
    }

    /// True if in the topological phase (κ₁ < κ₂).
    pub fn is_topological(&self) -> bool {
        self.kappa1 < self.kappa2
    }

    /// Zak phase (rad): 0 for trivial, π for topological.
    ///
    /// Computed analytically from the winding number:
    ///   φ_Zak = π if κ₁ < κ₂, else 0
    pub fn zak_phase(&self) -> f64 {
        if self.is_topological() {
            PI
        } else {
            0.0
        }
    }

    /// Bulk band structure ω(k) for k ∈ [-π, π].
    ///
    /// SSH bands: ω± = ω₀ ± √(κ₁² + κ₂² + 2κ₁κ₂·cos(k·a))
    /// where a = 1 is the unit cell length.
    pub fn bulk_bands(&self, n_k: usize) -> Vec<(f64, f64, f64)> {
        (0..n_k)
            .map(|i| {
                let k = -PI + 2.0 * PI * i as f64 / (n_k - 1) as f64;
                let cos_k = k.cos();
                let e_sq = self.kappa1 * self.kappa1
                    + self.kappa2 * self.kappa2
                    + 2.0 * self.kappa1 * self.kappa2 * cos_k;
                let e = e_sq.sqrt();
                (k, self.omega0 - e, self.omega0 + e)
            })
            .collect()
    }

    /// Band gap Δω = 2·|κ₂ - κ₁|.
    pub fn bandgap(&self) -> f64 {
        2.0 * (self.kappa2 - self.kappa1).abs()
    }

    /// Edge state frequency (at ω₀ in the middle of the gap, for topological phase).
    ///
    /// In the ideal SSH model, edge states appear exactly at ω₀.
    pub fn edge_state_frequency(&self) -> Option<f64> {
        if self.is_topological() {
            Some(self.omega0)
        } else {
            None
        }
    }

    /// Localisation length ξ of the edge state (in units of unit cells).
    ///
    ///   ξ = -1 / ln(κ₁/κ₂)
    pub fn edge_state_localisation_length(&self) -> Option<f64> {
        if !self.is_topological() {
            return None;
        }
        let ratio = self.kappa1 / self.kappa2;
        if ratio <= 0.0 || ratio >= 1.0 {
            return None;
        }
        Some(-1.0 / ratio.ln())
    }

    /// Eigenmode spectrum of the finite SSH chain (tight-binding Hamiltonian).
    ///
    /// For a chain with N unit cells (2N sites), returns 2N eigenfrequencies.
    /// Edge modes appear at ω ≈ ω₀ for topological phase.
    ///
    /// The matrix is:
    ///   H = [[ω₀, κ₁, 0, κ₂, 0, ...],
    ///         [κ₁, ω₀, κ₂, 0, κ₁, ...], ...]
    pub fn eigenfrequencies(&self) -> Vec<f64> {
        let n = 2 * self.n_cells;
        let mut h = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            h[i][i] = self.omega0;
            // SSH coupling pattern: within cell (κ₁), between cells (κ₂)
            if i + 1 < n {
                let kappa = if i % 2 == 0 { self.kappa1 } else { self.kappa2 };
                h[i][i + 1] = kappa;
                h[i + 1][i] = kappa;
            }
        }
        // Simple power iteration for eigenvalues (tridiagonal → sorted)
        // For demonstration: return diagonal + coupling estimate
        // Full diagonalization via Jacobi would be O(n³)
        Self::tridiagonal_eigenvalues(&h, n)
    }

    /// Symmetric tridiagonal eigenvalue solver.
    ///
    /// Attempts `TridiagEvd` first for better accuracy; falls back to the
    /// legacy Sturm-bisection path if the eigenpair solver fails.
    fn tridiagonal_eigenvalues(h: &[Vec<f64>], n: usize) -> Vec<f64> {
        let diag: Vec<f64> = (0..n).map(|i| h[i][i]).collect();
        let offdiag: Vec<f64> = (0..n.saturating_sub(1)).map(|i| h[i][i + 1]).collect();
        match eigenpairs_inner(&diag, &offdiag) {
            Ok((eigs, _)) => eigs,
            Err(_) => sturm_bisect_eigenvalues(&diag, &offdiag, n),
        }
    }
}

/// Compute eigenpairs of a symmetric tridiagonal matrix via `TridiagEvd`.
///
/// Returns sorted eigenvalues and the corresponding column-major eigenvector
/// matrix.  Falls through to the existing Sturm-bisection path only for the
/// eigenvalue-only callers; the full eigenpair path always uses TridiagEvd.
fn eigenpairs_inner(
    diagonal: &[f64],
    off_diagonal: &[f64],
) -> Result<(Vec<f64>, Mat<f64>), OxiPhotonError> {
    let evd = TridiagEvd::compute(diagonal, off_diagonal)
        .map_err(|e| OxiPhotonError::NumericalError(format!("tridiag evd: {e:?}")))?;
    let eigs = evd.eigenvalues().to_vec();
    let vecs = evd
        .eigenvectors()
        .ok_or_else(|| {
            OxiPhotonError::NumericalError("TridiagEvd returned no eigenvectors".to_string())
        })?
        .clone();
    // Debug-only orthogonality check
    #[cfg(debug_assertions)]
    {
        let n = eigs.len();
        for i in 0..n {
            for j in 0..n {
                let dot: f64 = (0..n).map(|r| vecs[(r, i)] * vecs[(r, j)]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                debug_assert!(
                    (dot - expected).abs() < 1e-4,
                    "eigenvector orthogonality failed: <v{i}|v{j}> = {dot:.6}"
                );
            }
        }
    }
    Ok((eigs, vecs))
}

/// Sturm sequence bisection for symmetric tridiagonal eigenvalues.
///
/// Counts eigenvalues below μ using Sturm sequence.
fn sturm_count(diag: &[f64], offdiag: &[f64], mu: f64) -> usize {
    let n = diag.len();
    let mut count = 0usize;
    let mut prev = diag[0] - mu;
    if prev < 0.0 {
        count += 1;
    }
    for i in 1..n {
        let d = diag[i] - mu;
        let o = offdiag[i - 1];
        let curr = if prev.abs() < 1e-30 {
            d - o * o / 1e-30
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

fn sturm_bisect_eigenvalues(diag: &[f64], offdiag: &[f64], n: usize) -> Vec<f64> {
    // Gershgorin bounds
    let mu_min = diag
        .iter()
        .zip(offdiag.iter())
        .fold(f64::INFINITY, |m, (&d, &o)| (d - o.abs()).min(m))
        .min(*diag.last().unwrap_or(&0.0) - offdiag.last().unwrap_or(&0.0).abs());
    let mu_max = diag
        .iter()
        .zip(offdiag.iter())
        .fold(f64::NEG_INFINITY, |m, (&d, &o)| (d + o.abs()).max(m))
        .max(*diag.last().unwrap_or(&0.0) + offdiag.last().unwrap_or(&0.0).abs());

    let mut eigenvalues = Vec::with_capacity(n);
    for k in 0..n {
        // Find k-th eigenvalue by bisection
        let mut lo = mu_min - 1.0;
        let mut hi = mu_max + 1.0;
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

/// Chern number for 2D photonic topological insulator (conceptual model).
///
/// For a 2-band model with Bloch Hamiltonian H(k) = d(k)·σ,
/// the Chern number C = (1/4π) ∫ d̂·(∂d̂/∂kx × ∂d̂/∂ky) d²k.
///
/// Implemented for the simple 2-band model d(kx,ky) = [sin(kx), sin(ky), m+cos(kx)+cos(ky)].
pub fn chern_number_two_band(m_param: f64, n_k: usize) -> i32 {
    let mut chern = 0.0f64;
    let dk = 2.0 * PI / n_k as f64;
    for ix in 0..n_k {
        for iy in 0..n_k {
            let kx = -PI + ix as f64 * dk;
            let ky = -PI + iy as f64 * dk;
            let d = [kx.sin(), ky.sin(), m_param + kx.cos() + ky.cos()];
            let d_norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if d_norm < 1e-10 {
                continue;
            }
            let dhat = [d[0] / d_norm, d[1] / d_norm, d[2] / d_norm];

            // Numerical gradient d̂(kx + dk) and d̂(ky + dk)
            let kx2 = kx + dk;
            let ky2 = ky + dk;
            let d_x = [kx2.sin(), ky.sin(), m_param + kx2.cos() + ky.cos()];
            let d_y = [kx.sin(), ky2.sin(), m_param + kx.cos() + ky2.cos()];
            let dnx = (d_x[0] * d_x[0] + d_x[1] * d_x[1] + d_x[2] * d_x[2])
                .sqrt()
                .max(1e-30);
            let dny = (d_y[0] * d_y[0] + d_y[1] * d_y[1] + d_y[2] * d_y[2])
                .sqrt()
                .max(1e-30);
            let dhx = [d_x[0] / dnx, d_x[1] / dnx, d_x[2] / dnx];
            let dhy = [d_y[0] / dny, d_y[1] / dny, d_y[2] / dny];

            // ∂d̂/∂kx ≈ (dhx - dhat) / dk
            let ddkx = [
                (dhx[0] - dhat[0]) / dk,
                (dhx[1] - dhat[1]) / dk,
                (dhx[2] - dhat[2]) / dk,
            ];
            let ddky = [
                (dhy[0] - dhat[0]) / dk,
                (dhy[1] - dhat[1]) / dk,
                (dhy[2] - dhat[2]) / dk,
            ];

            // Cross product ddkx × ddky
            let cross = [
                ddkx[1] * ddky[2] - ddkx[2] * ddky[1],
                ddkx[2] * ddky[0] - ddkx[0] * ddky[2],
                ddkx[0] * ddky[1] - ddkx[1] * ddky[0],
            ];

            // Berry curvature: d̂ · cross
            let curvature = dhat[0] * cross[0] + dhat[1] * cross[1] + dhat[2] * cross[2];
            chern += curvature * dk * dk;
        }
    }
    chern /= 4.0 * PI;
    chern.round() as i32
}

// ─── Berry phase / Wilson loop ─────────────────────────────────────────────────

/// Berry phase accumulated along a k-space path, computed via the Wilson-loop
/// (link-product) method for a set of Bloch eigenstates.
///
/// The Wilson-loop Berry phase for band `n` along a closed path
/// {k₀, k₁, …, k_N = k₀} is:
///
///   γ_n = -Im ln ∏_j ⟨u_n(k_j) | u_n(k_{j+1})⟩
///
/// For a 1D Brillouin zone path this reduces to the Zak phase:
///   γ = 0 (trivial) or π (topological).
#[derive(Debug, Clone)]
pub struct BerryPhase {
    /// Ordered k-points along the integration path (each element is [kx, ky]).
    pub k_path: Vec<[f64; 2]>,
    /// Berry phase (rad) for each band, computed from the Wilson loop.
    pub berry_phases: Vec<f64>,
}

impl BerryPhase {
    /// Compute the Wilson-loop Berry phase for a set of Bloch state vectors.
    ///
    /// # Arguments
    /// * `bands`  – slice of per-band Bloch vectors; `bands[n][j]` is the
    ///   complex amplitude of band `n` at k-point `j`.
    /// * `k_path` – k-point path (for storage only; loop product uses indices).
    ///
    /// # Returns
    /// `BerryPhase` with one phase per band.
    pub fn compute_wilson_loop(bands: &[Vec<Complex64>], k_path: &[[f64; 2]]) -> Self {
        let n_bands = bands.len();
        let n_k = k_path.len();
        let mut berry_phases = Vec::with_capacity(n_bands);

        for band in bands.iter() {
            if band.len() < 2 {
                berry_phases.push(0.0);
                continue;
            }
            // Wilson-loop product: W = ∏_{j=0}^{N-1} ⟨u(k_j)|u(k_{j+1})⟩
            let n_steps = n_k.min(band.len());
            let mut w = Complex64::new(1.0, 0.0);
            for j in 0..n_steps {
                let j_next = (j + 1) % n_steps;
                let u_j = band[j];
                let u_next = band[j_next];
                // Inner product ⟨u_j|u_{j+1}⟩ = u_j*.u_{j+1}
                let overlap = u_j.conj() * u_next;
                let norm_sq = u_j.norm_sqr() * u_next.norm_sqr();
                if norm_sq > 1e-30 {
                    w *= overlap / norm_sq.sqrt();
                }
            }
            // γ = -Im ln W
            berry_phases.push(-w.arg());
        }

        Self {
            k_path: k_path.to_vec(),
            berry_phases,
        }
    }

    /// Zak phase for band `band` (rad), taken modulo 2π into [0, 2π).
    ///
    /// Returns 0.0 if the band index is out of range.
    pub fn zak_phase(&self, band: usize) -> f64 {
        match self.berry_phases.get(band) {
            None => 0.0,
            Some(&phi) => {
                // Normalise to [0, 2π)

                phi.rem_euclid(2.0 * PI)
            }
        }
    }

    /// Returns `true` if the Zak phase for `band` is close to π (topological).
    ///
    /// Tolerance: |γ - π| < 0.2 rad.
    pub fn is_topological(&self, band: usize) -> bool {
        let phi = self.zak_phase(band);
        // π is topological; 0 (mod 2π) is trivial
        (phi - PI).abs() < 0.2 || (phi - PI).abs() > 2.0 * PI - 0.2
    }
}

// ─── Chern number ─────────────────────────────────────────────────────────────

/// Chern number for a single photonic band, computed by integrating the
/// Berry curvature over the 2D Brillouin zone.
///
/// For a non-degenerate band the Chern number is an integer topological
/// invariant that counts the number of chiral edge modes.
#[derive(Debug, Clone, Copy)]
pub struct ChernNumber {
    /// Band index (0-based).
    pub band_index: usize,
    /// Integer Chern number C.
    pub value: i32,
}

impl ChernNumber {
    /// Compute the Chern number by summing a pre-computed Berry-curvature grid.
    ///
    /// # Arguments
    /// * `berry_curvature` – 2D grid of Berry curvature values Ω(kx, ky);
    ///   `berry_curvature[ix][iy]` is the value at grid point (ix, iy).
    /// * `dk`              – uniform k-space grid spacing (rad/m or 1/period).
    ///
    /// # Returns
    /// `ChernNumber` with rounded integer value.
    pub fn compute_from_berry_curvature(berry_curvature: &[Vec<f64>], dk: f64) -> Self {
        let mut integral = 0.0_f64;
        for row in berry_curvature.iter() {
            for &omega_k in row.iter() {
                integral += omega_k * dk * dk;
            }
        }
        let c = (integral / (2.0 * PI)).round() as i32;
        Self {
            band_index: 0,
            value: c,
        }
    }

    /// Numerically compute the Berry curvature Ω(kx, ky) at a single k-point.
    ///
    /// Uses the four-point plaquette formula:
    ///
    ///   Ω ≈ -2 Im[ ⟨u(kx,ky)|u(kx+dk,ky)⟩ × ⟨u(kx+dk,ky)|u(kx+dk,ky+dk)⟩
    ///              × ⟨u(kx+dk,ky+dk)|u(kx,ky+dk)⟩ × ⟨u(kx,ky+dk)|u(kx,ky)⟩ ]
    ///   / dk²
    ///
    /// This requires Bloch vectors at the four corners of the plaquette.
    /// The `bloch_vecs` parameter provides the four two-component vectors in
    /// order: (kx,ky), (kx+dk,ky), (kx+dk,ky+dk), (kx,ky+dk).
    ///
    /// # Arguments
    /// * `bloch_vecs` – array of 4 two-component Bloch vectors (corners)
    /// * `kx`, `ky`  – lower-left corner of the plaquette (unused except docs)
    /// * `dk`        – plaquette side length
    pub fn berry_curvature_at_k(bloch_vecs: &[[Complex64; 2]], _kx: f64, _ky: f64, dk: f64) -> f64 {
        if bloch_vecs.len() < 4 || dk.abs() < 1e-30 {
            return 0.0;
        }
        // Overlap between consecutive corners
        let overlap = |a: &[Complex64; 2], b: &[Complex64; 2]| -> Complex64 {
            let raw = a[0].conj() * b[0] + a[1].conj() * b[1];
            let norm = (a[0].norm_sqr() + a[1].norm_sqr()).sqrt().max(1e-30)
                * (b[0].norm_sqr() + b[1].norm_sqr()).sqrt().max(1e-30);
            raw / norm
        };
        let u00 = &bloch_vecs[0];
        let u10 = &bloch_vecs[1];
        let u11 = &bloch_vecs[2];
        let u01 = &bloch_vecs[3];

        let w = overlap(u00, u10) * overlap(u10, u11) * overlap(u11, u01) * overlap(u01, u00);
        // Ω ≈ -Im ln(W) / dk²
        -w.arg() / (dk * dk)
    }
}

// ─── Topological edge states ──────────────────────────────────────────────────

/// Direction of the edge termination in a 2D photonic crystal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeDirection {
    /// Zigzag edge termination of a honeycomb lattice.
    ZigZag,
    /// Armchair edge termination of a honeycomb lattice.
    Armchair,
    /// Bearded edge termination (less common).
    Bearded,
}

/// A topological edge state in a photonic crystal.
///
/// Topological edge states arise at the boundary between regions with
/// different topological invariants (Chern numbers or Zak phases).  Chiral
/// edge states are immune to backscattering by time-reversal-breaking
/// perturbations.
#[derive(Debug, Clone)]
pub struct TopologicalEdgeState {
    /// Edge termination direction.
    pub edge_direction: EdgeDirection,
    /// Normalised resonance frequency ωa/2πc.
    pub frequency_normalized: f64,
    /// Group velocity v_g (m/s).
    pub group_velocity: f64,
    /// 1/e penetration depth into the bulk (m).
    pub penetration_depth: f64,
    /// `true` for chiral (unidirectional) edge modes.
    pub is_chiral: bool,
}

impl TopologicalEdgeState {
    /// Construct a topological edge state.
    ///
    /// # Arguments
    /// * `edge`  – edge direction
    /// * `freq`  – normalised frequency a/λ
    /// * `v_g`   – group velocity (m/s)
    pub fn new(edge: EdgeDirection, freq: f64, v_g: f64) -> Self {
        Self {
            edge_direction: edge,
            frequency_normalized: freq,
            group_velocity: v_g,
            penetration_depth: 0.0,
            is_chiral: false,
        }
    }

    /// Set the penetration depth and chirality, returning `self` for chaining.
    pub fn with_penetration_depth(mut self, depth: f64) -> Self {
        self.penetration_depth = depth;
        self
    }

    /// Mark this edge state as chiral (and therefore backscattering-immune).
    pub fn with_chirality(mut self, chiral: bool) -> Self {
        self.is_chiral = chiral;
        self
    }

    /// Returns `true` if the edge state is immune to backscattering.
    ///
    /// Chiral edge states (arising in systems with broken time-reversal or
    /// in valley-Hall materials at specific interfaces) cannot backscatter
    /// because there is no counter-propagating mode at the same energy.
    pub fn backscattering_immune(&self) -> bool {
        self.is_chiral
    }

    /// Estimated transmission through a point defect on the edge.
    ///
    /// * Chiral mode: T ≈ 1.0 (no backscattering channel).
    /// * Trivial mode: T < 1.0, estimated from Anderson localisation:
    ///   T ≈ exp(-L / ξ) where L is taken as 10 unit cells and ξ = 3 unit cells.
    pub fn transmission_through_defect(&self) -> f64 {
        if self.is_chiral {
            1.0
        } else {
            // Trivial: exponential Anderson localisation
            let l_over_xi: f64 = 10.0 / 3.0; // 10 cells / 3-cell localisation length
            (-l_over_xi).exp()
        }
    }
}

// ─── Valley photonic crystal ──────────────────────────────────────────────────

/// Valley-Hall photonic crystal.
///
/// Valley photonics exploits the two inequivalent corners (K and K′) of the
/// honeycomb Brillouin zone.  When inversion symmetry is broken (e.g. by
/// making the two sublattice holes different sizes), each valley acquires an
/// opposite Berry curvature and a half-integer valley Chern number C_v = ±½.
///
/// An interface between two domains with opposite valley Chern numbers hosts
/// topologically protected valley-Hall edge states with high transmission
/// even around sharp bends.
#[derive(Debug, Clone)]
pub struct ValleyPhotonicCrystal {
    /// `true` if the crystal breaks inversion symmetry (e.g. r_A ≠ r_B).
    pub inversion_symmetry_broken: bool,
    /// `true` if the crystal has C_{3v} point-group symmetry.
    pub c3v_symmetry: bool,
    /// Valley Chern number C_v (typically ±1 for the full valley, ±½ per cone).
    pub valley_chern_number: i32,
}

impl ValleyPhotonicCrystal {
    /// Construct a honeycomb valley PhC with two different air-hole radii.
    ///
    /// Setting r_A ≠ r_B breaks the inversion symmetry of the honeycomb
    /// lattice and opens a gap at the Dirac point K/K′.  Each valley then
    /// carries a valley Chern number C_v = ±1/2 (here represented as ±1
    /// for the integer label used in the literature).
    ///
    /// # Arguments
    /// * `r_a` – radius of sublattice-A holes (m)
    /// * `r_b` – radius of sublattice-B holes (m)
    pub fn new_honeycomb(r_a: f64, r_b: f64) -> Self {
        let broken = (r_a - r_b).abs() > 1e-15;
        // C_v = +1 for r_A > r_B, -1 for r_A < r_B (K-valley sign convention)
        let c_v = if !broken {
            0
        } else if r_a > r_b {
            1
        } else {
            -1
        };
        Self {
            inversion_symmetry_broken: broken,
            c3v_symmetry: true, // honeycomb with equal-size sublattice circles
            valley_chern_number: c_v,
        }
    }

    /// Berry-curvature contrast between the K and K′ valleys.
    ///
    /// For an inversion-broken honeycomb:
    ///   Δ_BC ≈ |C_v(K) - C_v(K′)| = 2|C_v|
    ///
    /// Returns 0 when inversion is not broken.
    pub fn valley_contrast(&self) -> f64 {
        if !self.inversion_symmetry_broken {
            0.0
        } else {
            2.0 * self.valley_chern_number.abs() as f64
        }
    }

    /// Approximate normalised frequency of the valley-Hall edge state.
    ///
    /// The edge state exists inside the valley-induced gap, which sits near
    /// the Dirac frequency of the unperturbed honeycomb lattice:
    ///   f_edge ≈ 0.25 for a honeycomb lattice (a/λ units).
    ///
    /// # Arguments
    /// * `interface_type` – `"zigzag"` gives a slightly higher frequency;
    ///   `"armchair"` gives a slightly lower one.
    pub fn edge_state_frequency(&self, interface_type: &str) -> f64 {
        if !self.inversion_symmetry_broken {
            return 0.0;
        }
        match interface_type {
            "zigzag" => 0.255,
            "armchair" => 0.245,
            _ => 0.25,
        }
    }

    /// Transmission through a sharp kink (180° bend) in a valley-Hall
    /// waveguide, estimated from experiment / FDTD literature.
    ///
    /// * Topological (broken inversion): T ≈ 0.85–0.95 (literature range).
    /// * Trivial (inversion intact): T ≈ 0.3 (strong backscattering).
    pub fn transmission_kink_test(&self) -> f64 {
        if self.inversion_symmetry_broken {
            0.90
        } else {
            0.30
        }
    }
}

// ─── SSH photonic chain (extended interface for new module) ────────────────────

/// SSH (Su-Schrieffer-Heeger) 1D photonic crystal chain.
///
/// This is a re-export-friendly wrapper around the existing `SshChain`,
/// providing the naming convention requested in the new topology module.
/// All physics is identical to `SshChain`.
#[derive(Debug, Clone)]
pub struct SshPhotonicChain {
    /// Number of unit cells N.
    pub n_unit_cells: usize,
    /// Intra-cell coupling κ₁.
    pub kappa1: f64,
    /// Inter-cell coupling κ₂.
    pub kappa2: f64,
    /// On-site resonance frequency ω₀.
    pub omega0: f64,
}

impl SshPhotonicChain {
    /// Construct an SSH photonic chain.
    pub fn new(n: usize, kappa1: f64, kappa2: f64, omega0: f64) -> Self {
        Self {
            n_unit_cells: n,
            kappa1,
            kappa2,
            omega0,
        }
    }

    /// Winding number w: 0 (trivial) or 1 (topological).
    ///
    /// w = 1 when κ₂ > κ₁ (inter-cell coupling dominates).
    pub fn winding_number(&self) -> i32 {
        if self.kappa2 > self.kappa1 {
            1
        } else {
            0
        }
    }

    /// `true` when the chain is in the topological phase (κ₂ > κ₁).
    pub fn is_topological(&self) -> bool {
        self.kappa2 > self.kappa1
    }

    /// Edge-state frequency: `Some(ω₀)` for topological, `None` for trivial.
    pub fn edge_state_frequency(&self) -> Option<f64> {
        if self.is_topological() {
            Some(self.omega0)
        } else {
            None
        }
    }

    /// Bulk band gap Δω = 2|κ₂ − κ₁|.
    pub fn band_gap(&self) -> f64 {
        2.0 * (self.kappa2 - self.kappa1).abs()
    }

    /// Bulk dispersion relation: returns [ω₋(k), ω₊(k)].
    ///
    /// ω±(k) = ω₀ ± √(κ₁² + κ₂² + 2κ₁κ₂ cos k)
    pub fn eigenfrequencies(&self, k: f64) -> [f64; 2] {
        let e_sq = self.kappa1 * self.kappa1
            + self.kappa2 * self.kappa2
            + 2.0 * self.kappa1 * self.kappa2 * k.cos();
        let e = e_sq.max(0.0).sqrt();
        [self.omega0 - e, self.omega0 + e]
    }

    /// Bloch eigenvector at crystal momentum `k` for the given `band_index`.
    ///
    /// The SSH bulk Bloch Hamiltonian is:
    ///
    ///   H(k) = ⎡  0     h(k) ⎤   with h(k) = κ₁ + κ₂ exp(ik)
    ///          ⎣ h*(k)   0   ⎦
    ///
    /// Eigenvectors (in the sublattice basis):
    /// * Band 0 (lower):  (1, −h*(k)/|h|) / √2
    /// * Band 1 (upper):  (1, +h*(k)/|h|) / √2
    ///
    /// Returns `Err` for invalid `band_index` or a degenerate point (|h|=0).
    pub fn bloch_vector(
        &self,
        k: f64,
        band_index: usize,
    ) -> Result<Vec<Complex64>, OxiPhotonError> {
        if band_index > 1 {
            return Err(OxiPhotonError::NumericalError(format!(
                "band_index {band_index} out of range (SSH has 2 bands: 0 and 1)"
            )));
        }
        // Off-diagonal element of the Bloch Hamiltonian
        let h = Complex64::new(self.kappa1 + self.kappa2 * k.cos(), self.kappa2 * k.sin());
        let h_abs = h.norm();
        if h_abs < 1e-15 {
            return Err(OxiPhotonError::NumericalError(
                "degenerate k-point: |h(k)| = 0, Bloch vector undefined".to_string(),
            ));
        }
        // Phase factor e^{iφ} = h / |h|
        let h_phase = h / h_abs;
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        // Lower band sign = −1, upper band sign = +1
        let sign = if band_index == 0 {
            Complex64::new(-1.0, 0.0)
        } else {
            Complex64::new(1.0, 0.0)
        };
        // v = (1, sign * conj(h_phase)) / √2
        let bloch = vec![
            Complex64::new(inv_sqrt2, 0.0),
            sign * h_phase.conj() * inv_sqrt2,
        ];
        Ok(bloch)
    }

    /// Compute the Wilson-loop Berry (Zak) phase for `band_index` along a
    /// discretised 1D Brillouin-zone path.
    ///
    /// The SSH Bloch state at each k is a 2-component complex vector.  The
    /// full inner product ⟨u(k_j)|u(k_{j+1})⟩ = Σ_α u_α*(k_j) u_α(k_{j+1})
    /// is used for the Wilson-loop product, which then delegates to
    /// `BerryPhase::compute_wilson_loop` via the scalar-overlap convention.
    ///
    /// # Arguments
    /// * `k_path` – ordered k-points; the path is automatically closed
    ///   (wraps from the last point back to the first).
    /// * `band_index` – 0 (lower band) or 1 (upper band).
    ///
    /// # Returns
    /// `Ok(BerryPhase)` whose `berry_phases[0]` is the Zak phase, or `Err`
    /// if any `bloch_vector` call fails.
    pub fn wilson_loop_band_n(
        &self,
        k_path: &[f64],
        band_index: usize,
    ) -> Result<BerryPhase, OxiPhotonError> {
        let n_k = k_path.len();
        // Collect per-k Bloch vectors (2-component each).
        let bloch_vecs: Vec<Vec<Complex64>> = k_path
            .iter()
            .map(|&k| self.bloch_vector(k, band_index))
            .collect::<Result<Vec<_>, _>>()?;

        if n_k < 2 {
            let k_path_2d: Vec<[f64; 2]> = k_path.iter().map(|&k| [k, 0.0]).collect();
            return Ok(BerryPhase::compute_wilson_loop(
                &[vec![Complex64::new(1.0, 0.0)]],
                &k_path_2d,
            ));
        }

        // Compute full inner-product overlaps ⟨u(k_j)|u(k_{j+1})⟩ for all j.
        // Build a single scalar "band" of sequential overlap amplitudes so that
        // `compute_wilson_loop` can multiply them in a chain.
        // We encode each link as a single complex number whose argument is the
        // link phase; the product gives the Wilson loop.
        let mut link_phases: Vec<Complex64> = Vec::with_capacity(n_k);
        for j in 0..n_k {
            let j_next = (j + 1) % n_k;
            let u_j = &bloch_vecs[j];
            let u_next = &bloch_vecs[j_next];
            // ⟨u_j|u_{j+1}⟩ = Σ_α conj(u_j[α]) * u_{j+1}[α]
            let overlap: Complex64 = u_j
                .iter()
                .zip(u_next.iter())
                .map(|(&a, &b)| a.conj() * b)
                .sum();
            link_phases.push(overlap);
        }

        // Wilson-loop product W = ∏ (overlap / |overlap|)
        let mut w = Complex64::new(1.0, 0.0);
        for lp in &link_phases {
            let norm = lp.norm();
            if norm > 1e-30 {
                w *= lp / norm;
            }
        }
        let phase = -w.arg();

        let k_path_2d: Vec<[f64; 2]> = k_path.iter().map(|&k| [k, 0.0]).collect();
        Ok(BerryPhase {
            k_path: k_path_2d,
            berry_phases: vec![phase],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssh_topological_phase() {
        let chain = SshChain::topological(5);
        assert!(chain.is_topological());
        assert!((chain.zak_phase() - PI).abs() < 1e-10);
    }

    #[test]
    fn ssh_trivial_phase() {
        let chain = SshChain::trivial(5);
        assert!(!chain.is_topological());
        assert!(chain.zak_phase().abs() < 1e-10);
    }

    #[test]
    fn ssh_bandgap_positive() {
        let chain = SshChain::topological(5);
        assert!(chain.bandgap() > 0.0);
    }

    #[test]
    fn ssh_edge_state_exists_in_topological() {
        let chain = SshChain::topological(5);
        assert!(chain.edge_state_frequency().is_some());
    }

    #[test]
    fn ssh_no_edge_state_in_trivial() {
        let chain = SshChain::trivial(5);
        assert!(chain.edge_state_frequency().is_none());
    }

    #[test]
    fn ssh_localisation_length_positive() {
        let chain = SshChain::topological(5);
        let xi = chain.edge_state_localisation_length();
        assert!(xi.is_some() && xi.unwrap() > 0.0);
    }

    #[test]
    fn ssh_bulk_bands_count() {
        let chain = SshChain::topological(3);
        let bands = chain.bulk_bands(50);
        assert_eq!(bands.len(), 50);
    }

    #[test]
    fn ssh_eigenfrequencies_count() {
        let chain = SshChain::topological(4);
        let eigs = chain.eigenfrequencies();
        assert_eq!(eigs.len(), 8); // 2 × n_cells
    }

    #[test]
    fn chern_number_topological_phase() {
        // m=-1: two bands cross, Chern number = 1
        let c = chern_number_two_band(-1.0, 30);
        assert!(c.abs() == 1, "C={c}");
    }

    #[test]
    fn chern_number_trivial_phase() {
        // m=3: trivial, Chern number = 0
        let c = chern_number_two_band(3.0, 30);
        assert!(c == 0, "C={c}");
    }

    // ── BerryPhase tests ───────────────────────────────────────────────────

    #[test]
    fn berry_phase_wilson_loop_trivial() {
        // All-real positive vectors → overlap products all +1 → W = 1 → γ = 0
        let n_k = 8;
        let band: Vec<Complex64> = (0..n_k).map(|_| Complex64::new(1.0, 0.0)).collect();
        let k_path: Vec<[f64; 2]> = (0..n_k).map(|i| [i as f64 * 0.1, 0.0]).collect();
        let bp = BerryPhase::compute_wilson_loop(&[band], &k_path);
        let phi = bp.zak_phase(0);
        // Should be ≈ 0 (trivial)
        assert!(!(0.5..=2.0 * PI - 0.5).contains(&phi), "φ = {phi}");
    }

    #[test]
    fn berry_phase_empty_band_returns_zero() {
        let k_path = vec![[0.0, 0.0]];
        let bp = BerryPhase::compute_wilson_loop(&[vec![]], &k_path);
        assert_eq!(bp.zak_phase(0), 0.0);
    }

    #[test]
    fn berry_phase_out_of_bounds_band_returns_zero() {
        let k_path = vec![[0.0, 0.0], [1.0, 0.0]];
        let band = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
        let bp = BerryPhase::compute_wilson_loop(&[band], &k_path);
        // Band index 5 doesn't exist → 0
        assert_eq!(bp.zak_phase(5), 0.0);
    }

    // ── ChernNumber tests ──────────────────────────────────────────────────

    #[test]
    fn chern_number_from_uniform_curvature() {
        // A uniform Berry curvature Ω = 2π / (2π)² over the BZ integrates to 1
        let n_k = 10usize;
        let dk = 2.0 * PI / n_k as f64;
        // Total integral should give C = 1: Ω = 2π / (Nk * dk)²  = 2π / (2π*dk)
        // Choose Ω so that sum Ω dk² = 2π
        let omega = 2.0 * PI / ((n_k * n_k) as f64 * dk * dk);
        let grid: Vec<Vec<f64>> = (0..n_k).map(|_| vec![omega; n_k]).collect();
        let cn = ChernNumber::compute_from_berry_curvature(&grid, dk);
        assert_eq!(cn.value, 1, "C = {}", cn.value);
    }

    #[test]
    fn chern_number_zero_curvature() {
        let n_k = 5usize;
        let dk = 2.0 * PI / n_k as f64;
        let grid: Vec<Vec<f64>> = (0..n_k).map(|_| vec![0.0; n_k]).collect();
        let cn = ChernNumber::compute_from_berry_curvature(&grid, dk);
        assert_eq!(cn.value, 0);
    }

    #[test]
    fn berry_curvature_at_k_four_corners() {
        // Four identical vectors → Wilson loop = 1 → Ω = 0
        let u = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let vecs = [u, u, u, u];
        let omega = ChernNumber::berry_curvature_at_k(&vecs, 0.0, 0.0, 0.1);
        assert!(omega.abs() < 1e-6, "Ω = {omega}");
    }

    // ── TopologicalEdgeState tests ─────────────────────────────────────────

    #[test]
    fn chiral_edge_state_backscattering_immune() {
        let state =
            TopologicalEdgeState::new(EdgeDirection::ZigZag, 0.25, 1e5).with_chirality(true);
        assert!(state.backscattering_immune());
        assert!((state.transmission_through_defect() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn trivial_edge_state_not_backscattering_immune() {
        let state = TopologicalEdgeState::new(EdgeDirection::Armchair, 0.25, 1e5);
        assert!(!state.backscattering_immune());
        let t = state.transmission_through_defect();
        assert!(t < 1.0 && t > 0.0, "T = {t}");
    }

    #[test]
    fn edge_state_penetration_depth_set() {
        let state = TopologicalEdgeState::new(EdgeDirection::Bearded, 0.25, 5e4)
            .with_penetration_depth(300e-9);
        assert!((state.penetration_depth - 300e-9).abs() < 1e-15);
    }

    // ── ValleyPhotonicCrystal tests ────────────────────────────────────────

    #[test]
    fn valley_phc_broken_inversion_nonzero_contrast() {
        let vpc = ValleyPhotonicCrystal::new_honeycomb(130e-9, 100e-9);
        assert!(vpc.inversion_symmetry_broken);
        assert!(vpc.valley_contrast() > 0.0);
    }

    #[test]
    fn valley_phc_symmetric_zero_contrast() {
        let vpc = ValleyPhotonicCrystal::new_honeycomb(120e-9, 120e-9);
        assert!(!vpc.inversion_symmetry_broken);
        assert_eq!(vpc.valley_contrast(), 0.0);
    }

    #[test]
    fn valley_phc_kink_transmission_high() {
        let vpc = ValleyPhotonicCrystal::new_honeycomb(130e-9, 100e-9);
        let t = vpc.transmission_kink_test();
        assert!(t > 0.5, "T_kink = {t}");
    }

    #[test]
    fn valley_phc_edge_state_frequency_zigzag() {
        let vpc = ValleyPhotonicCrystal::new_honeycomb(130e-9, 100e-9);
        let f = vpc.edge_state_frequency("zigzag");
        assert!(f > 0.0, "f = {f}");
    }

    // ── SshPhotonicChain tests ─────────────────────────────────────────────

    #[test]
    fn ssh_photonic_chain_topological_winding() {
        let chain = SshPhotonicChain::new(5, 0.3, 1.0, 0.0);
        assert!(chain.is_topological());
        assert_eq!(chain.winding_number(), 1);
    }

    #[test]
    fn ssh_photonic_chain_trivial_winding() {
        let chain = SshPhotonicChain::new(5, 1.0, 0.3, 0.0);
        assert!(!chain.is_topological());
        assert_eq!(chain.winding_number(), 0);
    }

    #[test]
    fn ssh_photonic_chain_band_gap_positive() {
        let chain = SshPhotonicChain::new(5, 0.3, 1.0, 0.0);
        assert!(chain.band_gap() > 0.0);
    }

    #[test]
    fn ssh_photonic_chain_eigenfrequencies_ordered() {
        let chain = SshPhotonicChain::new(5, 0.5, 1.0, 2.0);
        let [omega_minus, omega_plus] = chain.eigenfrequencies(0.0);
        assert!(omega_plus >= omega_minus, "ω+ must be ≥ ω-");
    }

    #[test]
    fn ssh_photonic_chain_edge_state_at_omega0() {
        let chain = SshPhotonicChain::new(5, 0.3, 1.0, 3.0);
        let f_edge = chain.edge_state_frequency();
        assert_eq!(f_edge, Some(3.0));
    }
}
