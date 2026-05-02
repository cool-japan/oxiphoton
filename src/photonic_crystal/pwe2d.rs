/// 2D Plane-Wave Expansion (PWE) photonic crystal band-structure solver.
///
/// Implements the master eigenvalue equation for 2D photonic crystals
/// using the plane-wave expansion method. The convention throughout is:
///
///   - Lattice constant a = 1 (all lengths in units of a)
///   - k and G vectors in reduced units (units of 2π/a)
///     so the BZ boundary of a square lattice is at 0.5, not π
///   - Eigenvalues of the master matrix are (ω·a/(2πc))²
///   - Normalized frequencies returned are ω·a/(2πc)
///
/// References:
///   - Joannopoulos et al., "Photonic Crystals", 2nd ed.
///   - Ho, Chan, Soukoulis, Phys. Rev. Lett. 65, 3152 (1990)
use std::f64::consts::PI;

// ────────────────────────────────────────────────────────────────────
// 1a. Bessel J₁
// ────────────────────────────────────────────────────────────────────

/// Bessel function J₁(x) for x ≥ 0.
///
/// Power series for |x| < 5:
///   J₁(x) = (x/2) × Σ_{k=0}^∞ (-1)^k (x²/4)^k / (k! (k+1)!)
///
/// Hankel asymptotic for x ≥ 5 (simplified, accurate to ~0.1%):
///   J₁(x) ≈ sqrt(2/(π x)) × cos(x - 3π/4)
pub fn bessel_j1(x: f64) -> f64 {
    if x < 0.0 {
        return -bessel_j1(-x);
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < 5.0 {
        // Power series via recurrence: each term uses the previous.
        // T_k = (-1)^k (x/2)^{2k+1} / (k! (k+1)!)
        // T_{k+1} / T_k = -(x/2)² / ((k+1)(k+2))
        let x_half_sq = (x / 2.0) * (x / 2.0);
        let mut term = x / 2.0; // k=0: (x/2)^1 / (0! * 1!)
        let mut sum = term;
        for k in 1..=50_usize {
            term *= -x_half_sq / (k as f64 * (k + 1) as f64);
            sum += term;
            if term.abs() < 1e-16 * sum.abs() {
                break;
            }
        }
        sum
    } else {
        // Asymptotic expansion
        (2.0 / (PI * x)).sqrt() * (x - 3.0 * PI / 4.0).cos()
    }
}

// ────────────────────────────────────────────────────────────────────
// 1b. Reciprocal lattice and G-vector generation
// ────────────────────────────────────────────────────────────────────

/// Compute reciprocal lattice vectors b1, b2 from direct lattice a1, a2.
///
/// Uses the 2D formula:
///   b1 = 2π × (a2_perp) / |a1 × a2|
///   b2 = 2π × (a1_perp) / |a1 × a2|
///
/// where a2_perp = `[a2[1], -a2[0]]` and a1_perp = `[-a1[1], a1[0]]`.
/// Result is in units of 2π/a (Cartesian reciprocal-space coordinates).
pub fn recip_lattice(a1: [f64; 2], a2: [f64; 2]) -> ([f64; 2], [f64; 2]) {
    let cross = a1[0] * a2[1] - a1[1] * a2[0];
    // b1 perpendicular to a2, normalized
    let b1 = [2.0 * PI * a2[1] / cross, -2.0 * PI * a2[0] / cross];
    // b2 perpendicular to a1, normalized
    let b2 = [-2.0 * PI * a1[1] / cross, 2.0 * PI * a1[0] / cross];
    (b1, b2)
}

/// Generate G-vectors G = m·b1 + n·b2 with |m|,|n| ≤ n_max,
/// keeping those within a circular cutoff |G| ≤ G_max.
///
/// The G vectors are returned in **physical** Cartesian units (i.e.,
/// they are multiples of the reciprocal-lattice vectors, not reduced).
/// G_max = n_max × max(|b1|, |b2|).
pub fn gen_gvecs(b1: [f64; 2], b2: [f64; 2], n_max: i32) -> Vec<[f64; 2]> {
    let mag_b1 = (b1[0] * b1[0] + b1[1] * b1[1]).sqrt();
    let mag_b2 = (b2[0] * b2[0] + b2[1] * b2[1]).sqrt();
    let g_max = (n_max as f64) * mag_b1.max(mag_b2);
    let g_max_sq = g_max * g_max;

    let mut gvecs = Vec::new();
    for m in -n_max..=n_max {
        for n in -n_max..=n_max {
            let gx = m as f64 * b1[0] + n as f64 * b2[0];
            let gy = m as f64 * b1[1] + n as f64 * b2[1];
            if gx * gx + gy * gy <= g_max_sq + 1e-10 {
                gvecs.push([gx, gy]);
            }
        }
    }
    gvecs
}

// ────────────────────────────────────────────────────────────────────
// 1c. Permittivity Fourier coefficients (η = ε⁻¹)
// ────────────────────────────────────────────────────────────────────

/// Fourier coefficients of ε⁻¹ for a 2D unit cell with circular inclusions.
///
/// Convention (reduced units, k and G in units of 2π/a):
///   η(G=0,0) = f/eps_incl + (1-f)/eps_bg
///   η(G≠0)   = (1/eps_incl - 1/eps_bg) × 2f × J₁(2π·|G|·r̃) / (2π·|G|·r̃)
///
/// where r̃ = r/a is the normalized rod radius, and
/// the 2π factor comes from the Bessel argument when G is in reduced units.
///
/// The fill fraction f = π r̃² / (cell_area/a²), or equivalently
/// r̃ = sqrt(f × cell_area / π) where cell_area is in units of a².
///
/// Parameters:
///   eps_incl  : permittivity of circular inclusion
///   eps_bg    : permittivity of background
///   fill      : area fill fraction of inclusion in [0, 1)
///   cell_area : unit-cell area in units of a² (1.0 for square, sqrt(3)/2 for hex)
///   g_diff    : G - G' vector in physical (Cartesian) reciprocal-space units (2π/a)
pub fn eta_fourier_circular(
    eps_incl: f64,
    eps_bg: f64,
    fill: f64,
    cell_area: f64,
    g_diff: [f64; 2],
) -> f64 {
    let inv_incl = 1.0 / eps_incl;
    let inv_bg = 1.0 / eps_bg;
    // |G| in units of 2π/a
    let g_mag = (g_diff[0] * g_diff[0] + g_diff[1] * g_diff[1]).sqrt();
    // g_diff is already in physical units (2π/a from recip_lattice);
    // convert to reduced (divide by 2π) to get consistent unit with fill definition
    let g_red = g_mag / (2.0 * PI);

    if g_red < 1e-12 {
        // DC term (G = G')
        inv_incl * fill + inv_bg * (1.0 - fill)
    } else {
        // r̃ = r/a = sqrt(fill × cell_area / π)
        let r_tilde = (fill * cell_area / PI).sqrt();
        // Argument of J₁: 2π · |G_red| · r̃
        let arg = 2.0 * PI * g_red * r_tilde;
        let j1_over_arg = if arg < 1e-12 {
            // 2f·J₁(x)/x → f as x→0  (the function 2J₁(x)/x → 1 as x→0)
            0.5
        } else {
            bessel_j1(arg) / arg
        };
        (inv_incl - inv_bg) * 2.0 * fill * j1_over_arg
    }
}

// ────────────────────────────────────────────────────────────────────
// 1d. Master eigenvalue matrices
// ────────────────────────────────────────────────────────────────────

/// Build the TM (E-polarization) master matrix.
///
/// `M[i,j]` = |k + G_i| · |k + G_j| · η(G_i - G_j)
///
/// where k and G are in physical reciprocal-space coordinates (2π/a units).
/// Eigenvalues are (ω·a/(2πc))².
///
/// Returns a flat row-major n×n symmetric real matrix.
pub fn tm_matrix(k: [f64; 2], gvecs: &[[f64; 2]], eta: &dyn Fn([f64; 2]) -> f64) -> Vec<f64> {
    let n = gvecs.len();
    let mut mat = vec![0.0_f64; n * n];

    for i in 0..n {
        let kgi_x = k[0] + gvecs[i][0];
        let kgi_y = k[1] + gvecs[i][1];
        let kgi_mag = (kgi_x * kgi_x + kgi_y * kgi_y).sqrt();

        for j in 0..n {
            let kgj_x = k[0] + gvecs[j][0];
            let kgj_y = k[1] + gvecs[j][1];
            let kgj_mag = (kgj_x * kgj_x + kgj_y * kgj_y).sqrt();

            let g_diff = [gvecs[i][0] - gvecs[j][0], gvecs[i][1] - gvecs[j][1]];
            mat[i * n + j] = kgi_mag * kgj_mag * eta(g_diff);
        }
    }
    mat
}

/// Build the TE (H-polarization) master matrix.
///
/// `M[i,j]` = (k + G_i) · (k + G_j) · η(G_i - G_j)
///
/// (dot product, not magnitude product)
/// Eigenvalues are (ω·a/(2πc))².
///
/// Returns a flat row-major n×n symmetric real matrix.
pub fn te_matrix(k: [f64; 2], gvecs: &[[f64; 2]], eta: &dyn Fn([f64; 2]) -> f64) -> Vec<f64> {
    let n = gvecs.len();
    let mut mat = vec![0.0_f64; n * n];

    for i in 0..n {
        let kgi_x = k[0] + gvecs[i][0];
        let kgi_y = k[1] + gvecs[i][1];

        for j in 0..n {
            let kgj_x = k[0] + gvecs[j][0];
            let kgj_y = k[1] + gvecs[j][1];

            let dot = kgi_x * kgj_x + kgi_y * kgj_y;
            let g_diff = [gvecs[i][0] - gvecs[j][0], gvecs[i][1] - gvecs[j][1]];
            mat[i * n + j] = dot * eta(g_diff);
        }
    }
    mat
}

// ────────────────────────────────────────────────────────────────────
// 1e. Jacobi eigensolver for real symmetric matrices
// ────────────────────────────────────────────────────────────────────

/// Solve symmetric real eigenvalue problem A v = λ v via cyclic-by-row
/// Jacobi rotations.
///
/// Parameters:
///   a         : flat row-major n×n symmetric matrix (modified in place)
///   n         : matrix dimension
///   tol       : convergence threshold on max |off-diagonal element|
///   max_sweeps: maximum number of Jacobi sweeps (each sweep covers all n(n-1)/2 pairs)
///
/// Returns eigenvalues in ascending order.
/// The off-diagonal elements of `a` are zeroed, diagonal becomes eigenvalues.
pub fn jacobi_eigh(a: &mut [f64], n: usize, tol: f64, max_sweeps: usize) -> Vec<f64> {
    // Safeguard: trivial case
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![a[0]];
    }

    for _sweep in 0..max_sweeps {
        // Check convergence: find max off-diagonal element
        let mut max_off = 0.0_f64;
        for p in 0..n {
            for q in (p + 1)..n {
                let v = a[p * n + q].abs();
                if v > max_off {
                    max_off = v;
                }
            }
        }
        if max_off < tol {
            break;
        }

        // One sweep: iterate over all pairs (p,q) with p < q
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-15 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                // Standard Jacobi: tan(2θ) = 2·A[p,q] / (A[p,p] - A[q,q])
                // θ = 0.5 · atan2(2·A[p,q], A[p,p] - A[q,q])
                let theta = 0.5 * f64::atan2(2.0 * apq, app - aqq);
                let c = theta.cos();
                let s = theta.sin();

                // Update the 2×2 diagonal block (J^T A J):
                // new_pp = c²·app + 2cs·apq + s²·aqq
                // new_qq = s²·app - 2cs·apq + c²·aqq
                let new_app = c * c * app + 2.0 * c * s * apq + s * s * aqq;
                let new_aqq = s * s * app - 2.0 * c * s * apq + c * c * aqq;
                a[p * n + p] = new_app;
                a[q * n + q] = new_aqq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                // Update remaining rows/columns r ≠ p,q:
                // new_rp = c·arp + s·arq
                // new_rq = -s·arp + c·arq
                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = a[r * n + p];
                    let arq = a[r * n + q];
                    let new_arp = c * arp + s * arq;
                    let new_arq = -s * arp + c * arq;
                    a[r * n + p] = new_arp;
                    a[p * n + r] = new_arp;
                    a[r * n + q] = new_arq;
                    a[q * n + r] = new_arq;
                }
            }
        }
    }

    // Collect and sort diagonal elements
    let mut eigs: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    eigs.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    eigs
}

// ────────────────────────────────────────────────────────────────────
// 1f. Band structure structs and API
// ────────────────────────────────────────────────────────────────────

/// Polarization selector for 2D photonic crystal band structure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Polarization {
    /// Transverse Electric (H out-of-plane, E in-plane)
    TE,
    /// Transverse Magnetic (E out-of-plane, H in-plane)
    TM,
}

/// Band structure result along a k-path.
pub struct BandStructure {
    /// k-points along the path (in physical units, 2π/a)
    pub k_path: Vec<[f64; 2]>,
    /// `bands[band_index][k_index]` = ω·a/(2πc) (normalized frequency)
    pub bands: Vec<Vec<f64>>,
    /// Photonic band gaps: (lower_edge, upper_edge) in normalized freq units
    pub gaps: Vec<(f64, f64)>,
}

impl BandStructure {
    /// Find photonic band gaps between consecutive bands.
    ///
    /// A gap exists when the minimum of band n+1 is above the maximum of band n.
    pub fn find_gaps(&mut self) {
        self.gaps.clear();
        let nb = self.bands.len();
        for i in 0..(nb.saturating_sub(1)) {
            let lo = self.bands[i]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let hi = self.bands[i + 1]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            if hi > lo {
                self.gaps.push((lo, hi));
            }
        }
    }
}

/// 2D photonic crystal with circular inclusions.
pub struct PhCrystal2d {
    /// Direct lattice vector a1 (in units of a)
    pub a1: [f64; 2],
    /// Direct lattice vector a2 (in units of a)
    pub a2: [f64; 2],
    /// Permittivity of circular inclusion
    pub eps_incl: f64,
    /// Permittivity of background
    pub eps_bg: f64,
    /// Area fill fraction of inclusion (0 < fill < 1)
    pub fill: f64,
    /// Number of G-vector shells per direction (n_max in gen_gvecs)
    pub n_g: usize,
}

impl PhCrystal2d {
    /// Square lattice of dielectric rods in air (period a = 1).
    ///
    /// a1 = [1, 0], a2 = [0, 1].
    pub fn square_rods(eps_rod: f64, eps_bg: f64, fill: f64, n_g: usize) -> Self {
        Self {
            a1: [1.0, 0.0],
            a2: [0.0, 1.0],
            eps_incl: eps_rod,
            eps_bg,
            fill,
            n_g,
        }
    }

    /// Hexagonal lattice of air holes in dielectric (period a = 1).
    ///
    /// a1 = [1, 0], a2 = [0.5, sqrt(3)/2].
    pub fn hex_holes(eps_bg: f64, eps_hole: f64, fill: f64, n_g: usize) -> Self {
        Self {
            a1: [1.0, 0.0],
            a2: [0.5, 3.0_f64.sqrt() / 2.0],
            eps_incl: eps_hole,
            eps_bg,
            fill,
            n_g,
        }
    }

    /// Unit cell area |a1 × a2| (in units of a²).
    fn cell_area(&self) -> f64 {
        (self.a1[0] * self.a2[1] - self.a1[1] * self.a2[0]).abs()
    }

    /// Build reciprocal lattice and G-vector list for this crystal.
    fn build_gvecs(&self) -> Vec<[f64; 2]> {
        let (b1, b2) = recip_lattice(self.a1, self.a2);
        gen_gvecs(b1, b2, self.n_g as i32)
    }

    /// Create the η = ε⁻¹ closure for this crystal structure.
    fn eta_fn(&self) -> impl Fn([f64; 2]) -> f64 + '_ {
        let eps_incl = self.eps_incl;
        let eps_bg = self.eps_bg;
        let fill = self.fill;
        let cell_area = self.cell_area();
        move |g_diff: [f64; 2]| eta_fourier_circular(eps_incl, eps_bg, fill, cell_area, g_diff)
    }

    /// Convert k in reduced units [0..0.5] to physical units (multiply by 2π).
    fn k_to_phys(k_red: [f64; 2]) -> [f64; 2] {
        [k_red[0] * 2.0 * PI, k_red[1] * 2.0 * PI]
    }

    /// Solve TM eigenfrequencies at a single k-point (k in reduced units).
    ///
    /// Returns ω·a/(2πc) values sorted ascending.
    pub fn solve_tm(&self, k_red: [f64; 2]) -> Vec<f64> {
        let gvecs = self.build_gvecs();
        let eta = self.eta_fn();
        let k_phys = Self::k_to_phys(k_red);
        let mut mat = tm_matrix(k_phys, &gvecs, &eta);
        let n = gvecs.len();
        let raw = jacobi_eigh(&mut mat, n, 1e-10, 50);
        // Eigenvalues are (ω·a/(2πc))²; take sqrt, clamp negatives
        // Normalize by dividing by (2π)² since k, G were in physical units
        raw.into_iter()
            .map(|ev| {
                let ev_norm = ev / (4.0 * PI * PI);
                if ev_norm < 0.0 {
                    0.0
                } else {
                    ev_norm.sqrt()
                }
            })
            .collect()
    }

    /// Solve TE eigenfrequencies at a single k-point (k in reduced units).
    ///
    /// Returns ω·a/(2πc) values sorted ascending.
    pub fn solve_te(&self, k_red: [f64; 2]) -> Vec<f64> {
        let gvecs = self.build_gvecs();
        let eta = self.eta_fn();
        let k_phys = Self::k_to_phys(k_red);
        let mut mat = te_matrix(k_phys, &gvecs, &eta);
        let n = gvecs.len();
        let raw = jacobi_eigh(&mut mat, n, 1e-10, 50);
        raw.into_iter()
            .map(|ev| {
                let ev_norm = ev / (4.0 * PI * PI);
                if ev_norm < 0.0 {
                    0.0
                } else {
                    ev_norm.sqrt()
                }
            })
            .collect()
    }

    /// Compute band diagram along k_path for the given polarization.
    ///
    /// k_path: k-points in **reduced** units (e.g., X=[0.5,0] for square lattice).
    /// Returns BandStructure with n_bands = min(total_gvecs, 10) bands.
    pub fn band_diagram(&self, k_path: &[[f64; 2]], pol: Polarization) -> BandStructure {
        let gvecs = self.build_gvecs();
        let n_total = gvecs.len();
        let n_bands = n_total.min(10);
        let eta = self.eta_fn();

        let mut bands: Vec<Vec<f64>> = (0..n_bands).map(|_| Vec::new()).collect();

        for &k_red in k_path {
            let k_phys = Self::k_to_phys(k_red);
            let mut mat = match pol {
                Polarization::TM => tm_matrix(k_phys, &gvecs, &eta),
                Polarization::TE => te_matrix(k_phys, &gvecs, &eta),
            };
            let raw = jacobi_eigh(&mut mat, n_total, 1e-10, 50);
            for (band_idx, band) in bands.iter_mut().enumerate() {
                let ev = raw[band_idx];
                let ev_norm = ev / (4.0 * PI * PI);
                let freq = if ev_norm < 0.0 { 0.0 } else { ev_norm.sqrt() };
                band.push(freq);
            }
        }

        let mut bs = BandStructure {
            k_path: k_path.to_vec(),
            bands,
            gaps: Vec::new(),
        };
        bs.find_gaps();
        bs
    }
}

// ────────────────────────────────────────────────────────────────────
// 1g. k-path generation
// ────────────────────────────────────────────────────────────────────

/// Generate k-path for square lattice: Γ → X → M → Γ
///
/// High-symmetry points in reduced units (units of 2π/a):
///   Γ = [0.0, 0.0]
///   X = [0.5, 0.0]
///   M = [0.5, 0.5]
///
/// Returns n_per_segment × 3 + 1 points total.
pub fn kpath_square(n_per_segment: usize) -> Vec<[f64; 2]> {
    let gamma = [0.0_f64, 0.0];
    let x_pt = [0.5, 0.0];
    let m_pt = [0.5, 0.5];

    let mut path = Vec::with_capacity(n_per_segment * 3 + 1);

    // Γ → X
    for i in 0..n_per_segment {
        let t = i as f64 / n_per_segment as f64;
        path.push(lerp(gamma, x_pt, t));
    }
    // X → M
    for i in 0..n_per_segment {
        let t = i as f64 / n_per_segment as f64;
        path.push(lerp(x_pt, m_pt, t));
    }
    // M → Γ
    for i in 0..n_per_segment {
        let t = i as f64 / n_per_segment as f64;
        path.push(lerp(m_pt, gamma, t));
    }
    // Final Γ point
    path.push(gamma);
    path
}

/// Generate k-path for hexagonal lattice: Γ → M → K → Γ
///
/// High-symmetry points in reduced coordinates of the reciprocal lattice
/// (i.e., as `[α, β]` where k = α·b1 + β·b2, then projected to Cartesian):
///   Γ = `[0, 0]`
///   M = midpoint of BZ edge → `[0.5, 0]` in reduced
///   K = BZ corner → `[1/3, 1/3]` in reduced (or equivalently `[2/3, 1/3]`)
///
/// Returns n_per_segment × 3 + 1 points (in reduced 2π/a units converted to Cartesian).
///
/// For hex lattice with a1=`[1,0]`, a2=`[0.5, √3/2]`:
///   b1 = 2π·`[1, -1/√3]`, b2 = 2π·`[0, 2/√3]`
///   M_cart = 0.5·b1/(2π) = `[0.5, -1/(2√3)]`  → but convention varies.
///
/// We use reduced coordinates directly and return them as-is (our solve methods
/// accept reduced coordinates and convert internally).
pub fn kpath_hexagonal(n_per_segment: usize) -> Vec<[f64; 2]> {
    // High-symmetry points in reduced reciprocal-lattice coordinates
    // (fractions of b1, b2), then converted to Cartesian (2π/a = 1 units)
    // For hex: a1=[1,0], a2=[0.5, √3/2]
    // b1 = [1, -1/√3], b2 = [0, 2/√3] (in units of 2π/a)
    // Γ = (0,0): k=0
    // M = (1/2, 0): k = 0.5·b1 = [0.5, -0.5/√3]
    // K = (1/3, 1/3): k = (1/3)(b1+b2) = [1/3, 1/(3·√3)·(2-1)] = [1/3, 1/(3√3)]

    let sqrt3 = 3.0_f64.sqrt();
    let gamma = [0.0_f64, 0.0];
    // M: half of b1 in Cartesian (in reduced 2π/a units)
    let m_pt = [0.5_f64, -0.5 / sqrt3];
    // K: (1/3)·b1 + (1/3)·b2 in Cartesian
    // b1_red = [1, -1/√3], b2_red = [0, 2/√3]
    let k_pt = [1.0 / 3.0, -1.0 / (3.0 * sqrt3) + 2.0 / (3.0 * sqrt3)];
    // = [1/3, 1/(3√3)]

    let mut path = Vec::with_capacity(n_per_segment * 3 + 1);

    // Γ → M
    for i in 0..n_per_segment {
        let t = i as f64 / n_per_segment as f64;
        path.push(lerp(gamma, m_pt, t));
    }
    // M → K
    for i in 0..n_per_segment {
        let t = i as f64 / n_per_segment as f64;
        path.push(lerp(m_pt, k_pt, t));
    }
    // K → Γ
    for i in 0..n_per_segment {
        let t = i as f64 / n_per_segment as f64;
        path.push(lerp(k_pt, gamma, t));
    }
    // Final Γ
    path.push(gamma);
    path
}

/// Linear interpolation between two 2D points.
#[inline]
fn lerp(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}

// ────────────────────────────────────────────────────────────────────
// Unit tests
// ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bessel_j1_at_zero() {
        assert!((bessel_j1(0.0)).abs() < 1e-14);
    }

    #[test]
    fn bessel_j1_at_one() {
        let expected = 0.4400505857449335_f64;
        let got = bessel_j1(1.0);
        assert!(
            (got - expected).abs() < 1e-8,
            "J₁(1) = {} (expected {})",
            got,
            expected
        );
    }

    #[test]
    fn bessel_j1_first_zero() {
        // First zero of J₁ is near 3.8317
        let val = bessel_j1(3.8317);
        assert!(val.abs() < 1e-3, "J₁(3.8317) = {}", val);
    }

    #[test]
    fn recip_lattice_square() {
        let (b1, b2) = recip_lattice([1.0, 0.0], [0.0, 1.0]);
        // b1 should be 2π × [1, 0], b2 = 2π × [0, 1]
        assert!((b1[0] - 2.0 * PI).abs() < 1e-10);
        assert!(b1[1].abs() < 1e-10);
        assert!(b2[0].abs() < 1e-10);
        assert!((b2[1] - 2.0 * PI).abs() < 1e-10);
    }

    #[test]
    fn jacobi_eigh_2x2() {
        let mut a = vec![3.0_f64, 1.0, 1.0, 2.0];
        let eigs = jacobi_eigh(&mut a, 2, 1e-12, 50);
        // Eigenvalues of [[3,1],[1,2]]: (5 ± √5)/2
        let lam1 = (5.0 - 5.0_f64.sqrt()) / 2.0;
        let lam2 = (5.0 + 5.0_f64.sqrt()) / 2.0;
        assert!((eigs[0] - lam1).abs() < 1e-10, "eig[0]={}", eigs[0]);
        assert!((eigs[1] - lam2).abs() < 1e-10, "eig[1]={}", eigs[1]);
    }

    #[test]
    fn empty_lattice_lowest_tm_at_gamma() {
        // Uniform ε=1 crystal: TM lowest frequency at Γ should be 0
        let crystal = PhCrystal2d::square_rods(1.0, 1.0, 0.3, 3);
        let freqs = crystal.solve_tm([0.0, 0.0]);
        assert!(freqs[0].abs() < 1e-8, "Lowest TM at Γ = {}", freqs[0]);
    }
}
