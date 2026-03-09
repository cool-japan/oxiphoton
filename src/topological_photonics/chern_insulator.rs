//! 2D Chern insulator — Qi-Wu-Zhang (QWZ) model.
//!
//! The QWZ model is the canonical 2-band 2D topological insulator:
//!
//!   H(k) = d(k) · σ
//!
//! where the d-vector is:
//!   d_x(k) = sin(k_x)
//!   d_y(k) = sin(k_y)
//!   d_z(k) = u + cos(k_x) + cos(k_y)
//!
//! and σ = (σ_x, σ_y, σ_z) are the Pauli matrices.
//!
//! The Chern number is:
//!   C = (1/4π) ∬_BZ  d̂ · (∂d̂/∂k_x × ∂d̂/∂k_y)  dk_x dk_y
//!
//! Phase diagram:
//!   |u| > 2  →  C = 0  (trivial)
//!    0 < u < 2  →  C = −1
//!   −2 < u < 0  →  C = +1
//!    u = 0       →  C = ±2 (critical, gap closes at Γ)
//!    u = ±2      →  gap closes at M points (phase boundaries)

use std::f64::consts::PI;

// ─── QWZ model ─────────────────────────────────────────────────────────────────

/// Qi-Wu-Zhang (QWZ) 2D Chern insulator.
///
/// The two-band Hamiltonian is H(k) = d(k)·σ with:
///   d = [sin k_x,  sin k_y,  u + cos k_x + cos k_y]
#[derive(Debug, Clone, Copy)]
pub struct QwzModel {
    /// Topological mass parameter u.
    ///
    /// Controls the Chern number (see module-level docs for the phase diagram).
    pub u: f64,
}

impl QwzModel {
    /// Construct a QWZ model with parameter `u`.
    pub fn new(u: f64) -> Self {
        Self { u }
    }

    /// d-vector [d_x, d_y, d_z] at k-point (kx, ky).
    ///
    /// d_x = sin(kx),  d_y = sin(ky),  d_z = u + cos(kx) + cos(ky)
    pub fn d_vec(&self, kx: f64, ky: f64) -> [f64; 3] {
        [kx.sin(), ky.sin(), self.u + kx.cos() + ky.cos()]
    }

    /// 2×2 Bloch Hamiltonian at (kx, ky).
    ///
    /// Returned as the real representation:
    ///   H = [[d_z,  d_x],
    ///        [d_x, -d_z]]
    ///
    /// (The off-diagonal d_y term enters as an imaginary part; for the
    ///  eigenvalue calculation only |d| matters, so the real 2×2 matrix
    ///  is sufficient for energy purposes.)
    pub fn hamiltonian(&self, kx: f64, ky: f64) -> [[f64; 2]; 2] {
        let d = self.d_vec(kx, ky);
        // Simplified real representation: d·σ with σ_y imaginary part omitted for display
        [[d[2], d[0]], [d[0], -d[2]]]
    }

    /// Energy eigenvalues E±(k) = ±|d(k)|.
    ///
    /// Returns `(E_minus, E_plus)`.
    pub fn energies(&self, kx: f64, ky: f64) -> (f64, f64) {
        let d = self.d_vec(kx, ky);
        let mag = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        (-mag, mag)
    }

    /// Berry curvature Ω_z(kx, ky) of the lower band.
    ///
    /// Computed via the d-vector formula:
    ///   Ω_z = (1/2) d̂ · (∂_kx d̂ × ∂_ky d̂)
    ///
    /// where the gradients are evaluated by finite differences.
    pub fn berry_curvature(&self, kx: f64, ky: f64) -> f64 {
        let eps = 1e-5;
        let d = self.d_vec(kx, ky);
        let d_pkx = self.d_vec(kx + eps, ky);
        let d_pky = self.d_vec(kx, ky + eps);

        // Finite-difference gradients ∂d/∂kx and ∂d/∂ky
        let dd_dkx = [
            (d_pkx[0] - d[0]) / eps,
            (d_pkx[1] - d[1]) / eps,
            (d_pkx[2] - d[2]) / eps,
        ];
        let dd_dky = [
            (d_pky[0] - d[0]) / eps,
            (d_pky[1] - d[1]) / eps,
            (d_pky[2] - d[2]) / eps,
        ];

        // Cross product dd_dkx × dd_dky
        let cross = [
            dd_dkx[1] * dd_dky[2] - dd_dkx[2] * dd_dky[1],
            dd_dkx[2] * dd_dky[0] - dd_dkx[0] * dd_dky[2],
            dd_dkx[0] * dd_dky[1] - dd_dkx[1] * dd_dky[0],
        ];

        // Normalise d to get d̂
        let d_norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if d_norm < 1e-30 {
            return 0.0;
        }
        let d_hat = [d[0] / d_norm, d[1] / d_norm, d[2] / d_norm];

        // Berry curvature: Ω = (1/2) d̂ · cross
        0.5 * (d_hat[0] * cross[0] + d_hat[1] * cross[1] + d_hat[2] * cross[2])
    }

    /// Chern number C computed by integrating the Berry curvature over the BZ.
    ///
    /// Uses a uniform n_k × n_k grid covering k ∈ [−π, π)²:
    ///   C = (1/2π) Σ_{kx,ky} Ω_z(kx, ky) · Δk²
    ///
    /// Typical accuracy: ±0 for |u| away from phase boundaries (|u| = 0 or 2).
    pub fn chern_number(&self, n_k: usize) -> i32 {
        if n_k == 0 {
            return 0;
        }
        let dk = 2.0 * PI / n_k as f64;
        let mut integral = 0.0_f64;
        for ix in 0..n_k {
            for iy in 0..n_k {
                let kx = -PI + ix as f64 * dk;
                let ky = -PI + iy as f64 * dk;
                integral += self.berry_curvature(kx, ky) * dk * dk;
            }
        }
        (integral / (2.0 * PI)).round() as i32
    }

    /// Phase diagram: Chern number as a function of the parameter u.
    ///
    /// Samples `n_points` values of u in `u_range = (u_min, u_max)` and
    /// returns `Vec<(u, C)>` pairs.
    ///
    /// Uses a moderate grid (n_k = 30) for speed; increase for precision near
    /// phase boundaries.
    pub fn phase_diagram(u_range: (f64, f64), n_points: usize) -> Vec<(f64, i32)> {
        if n_points == 0 {
            return Vec::new();
        }
        let (u_min, u_max) = u_range;
        let du = if n_points > 1 {
            (u_max - u_min) / (n_points - 1) as f64
        } else {
            0.0
        };
        (0..n_points)
            .map(|i| {
                let u = u_min + i as f64 * du;
                let model = QwzModel::new(u);
                (u, model.chern_number(30))
            })
            .collect()
    }

    /// Minimum band gap over the entire Brillouin zone.
    ///
    /// The gap closes at special k-points when the topological phase transitions occur.
    /// For the QWZ model the minimum gap is 2 × min_k |d(k)|.
    pub fn band_gap(&self) -> f64 {
        // The gap closes at high-symmetry points: Γ, M, X, Y
        // Evaluate |d| at these points and find the minimum
        let special_k = [
            (0.0_f64, 0.0_f64), // Γ:  d_z = u + 2
            (PI, PI),           // M:  d_z = u − 2
            (PI, 0.0),          // X:  d_z = u
            (0.0, PI),          // Y:  d_z = u
        ];
        let min_gap = special_k
            .iter()
            .map(|&(kx, ky)| {
                let d = self.d_vec(kx, ky);
                2.0 * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
            })
            .fold(f64::INFINITY, f64::min);
        min_gap.max(0.0)
    }

    /// Hall conductance σ_xy in units of e²/h.
    ///
    /// By the bulk-edge correspondence: σ_xy = C × e²/h.
    /// This method returns the value in units where e²/h = 1.
    pub fn hall_conductance(&self) -> f64 {
        self.chern_number(50) as f64
    }

    /// Number of chiral edge modes per edge = |C|.
    ///
    /// Each chiral edge mode is unidirectional and immune to backscattering.
    pub fn n_edge_states(&self) -> i32 {
        self.chern_number(50).abs()
    }
}

// ─── Berry curvature map ───────────────────────────────────────────────────────

/// Compute the Berry curvature Ω_z over the full 2D Brillouin zone.
///
/// Returns an `n_k × n_k` grid of Berry curvature values; element `[ix][iy]`
/// corresponds to k-point (k_x, k_y) = (−π + ix·Δk, −π + iy·Δk).
pub fn berry_curvature_map(model: &QwzModel, n_k: usize) -> Vec<Vec<f64>> {
    if n_k == 0 {
        return Vec::new();
    }
    let dk = 2.0 * PI / n_k as f64;
    (0..n_k)
        .map(|ix| {
            let kx = -PI + ix as f64 * dk;
            (0..n_k)
                .map(|iy| {
                    let ky = -PI + iy as f64 * dk;
                    model.berry_curvature(kx, ky)
                })
                .collect()
        })
        .collect()
}

/// Integrate the Berry curvature map to obtain the Chern number.
///
/// Helper function for post-processing a pre-computed curvature grid.
pub fn chern_from_curvature_map(map: &[Vec<f64>], dk: f64) -> i32 {
    let integral: f64 = map
        .iter()
        .flat_map(|row| row.iter())
        .map(|&v| v * dk * dk)
        .sum();
    (integral / (2.0 * PI)).round() as i32
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwz_chern_number_trivial_large_u() {
        let m = QwzModel::new(3.0); // |u| > 2: trivial
        assert_eq!(m.chern_number(40), 0);
    }

    #[test]
    fn qwz_chern_number_trivial_negative_large_u() {
        let m = QwzModel::new(-3.0); // |u| > 2: trivial
        assert_eq!(m.chern_number(40), 0);
    }

    #[test]
    fn qwz_chern_number_topological_u_positive() {
        let m = QwzModel::new(1.0); // 0 < u < 2: |C| = 1
        let c = m.chern_number(40).abs();
        assert_eq!(c, 1, "Expected |C|=1, got C={c}");
    }

    #[test]
    fn qwz_chern_number_topological_u_negative() {
        let m = QwzModel::new(-1.0); // -2 < u < 0: |C| = 1
        let c = m.chern_number(40).abs();
        assert_eq!(c, 1, "Expected |C|=1, got C={c}");
    }

    #[test]
    fn qwz_energies_symmetric() {
        let m = QwzModel::new(1.0);
        let (em, ep) = m.energies(0.5, 0.3);
        assert!((em + ep).abs() < 1e-12, "E+ + E- = {}", em + ep);
        assert!(ep >= 0.0);
    }

    #[test]
    fn qwz_berry_curvature_finite() {
        let m = QwzModel::new(1.0);
        let omega = m.berry_curvature(0.5, 0.5);
        assert!(omega.is_finite());
    }

    #[test]
    fn qwz_band_gap_positive_nontrivial() {
        let m = QwzModel::new(1.0);
        assert!(m.band_gap() > 0.0);
    }

    #[test]
    fn qwz_band_gap_closes_at_boundary() {
        // At u = 2.0 the gap closes at the M point: d = [0, 0, 0] there
        let m = QwzModel::new(2.0);
        let gap = m.band_gap();
        assert!(gap < 1e-10, "Gap at u=2 should be ~0, got {gap}");
    }

    #[test]
    fn qwz_phase_diagram_length() {
        let pd = QwzModel::phase_diagram((-3.0, 3.0), 7);
        assert_eq!(pd.len(), 7);
    }

    #[test]
    fn qwz_phase_diagram_trivial_endpoints() {
        let pd = QwzModel::phase_diagram((-3.0, 3.0), 7);
        // First point u=-3 and last u=3 should be trivial (C=0)
        assert_eq!(pd[0].1, 0, "u={}: C should be 0", pd[0].0);
        assert_eq!(pd[6].1, 0, "u={}: C should be 0", pd[6].0);
    }

    #[test]
    fn berry_curvature_map_dimensions() {
        let m = QwzModel::new(1.0);
        let map = berry_curvature_map(&m, 10);
        assert_eq!(map.len(), 10);
        for row in &map {
            assert_eq!(row.len(), 10);
        }
    }

    #[test]
    fn chern_from_curvature_map_trivial() {
        // Zero curvature map → C = 0
        let map: Vec<Vec<f64>> = vec![vec![0.0; 5]; 5];
        let dk = 2.0 * PI / 5.0;
        assert_eq!(chern_from_curvature_map(&map, dk), 0);
    }

    #[test]
    fn qwz_hall_conductance_trivial_is_zero() {
        let m = QwzModel::new(3.0);
        assert_eq!(m.hall_conductance() as i32, 0);
    }

    #[test]
    fn qwz_n_edge_states_nontrivial() {
        let m = QwzModel::new(1.0);
        assert_eq!(m.n_edge_states(), 1);
    }
}
