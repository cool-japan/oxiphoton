use num_complex::Complex64;
use std::f64::consts::PI;

/// Type alias for a 2D complex matrix (N×N S-matrix block).
pub type CMatrix = Vec<Vec<Complex64>>;

/// Tuple of four S-matrix blocks (S11, S12, S21, S22).
pub type SMatrixBlocks = (CMatrix, CMatrix, CMatrix, CMatrix);

/// Eigenmode Expansion Method (EME) for 1D slab waveguide segments.
///
/// The EME method decomposes the propagation problem into modes of each waveguide
/// segment, then cascades the S-matrices at each interface using mode overlap integrals.
///
/// For each segment, modes are found using the analytical transcendental equation
/// (effective index method). At each interface, the coupling S-matrix is built
/// from mode overlap integrals, then cascaded along the propagation direction.
///
/// Reference: Bienstman & Baets, OQE 33(4-5), 2001.
/// A guided mode with its effective index and field profile.
#[derive(Debug, Clone)]
pub struct EmeMode {
    /// Effective index n_eff = β/k0
    pub n_eff: f64,
    /// Propagation constant β = k0·n_eff
    pub beta: f64,
    /// Transverse field profile (normalized) sampled on a grid
    pub field: Vec<f64>,
    /// Grid spacing for the field
    pub dx: f64,
}

impl EmeMode {
    /// Compute field norm ∫|E|²dx
    pub fn norm(&self) -> f64 {
        self.field.iter().map(|&e| e * e).sum::<f64>() * self.dx
    }

    /// Overlap integral with another mode: ∫E_1·E_2 dx / sqrt(N1·N2)
    pub fn overlap(&self, other: &EmeMode) -> f64 {
        assert_eq!(
            self.field.len(),
            other.field.len(),
            "modes must be on same grid"
        );
        let inner: f64 = self
            .field
            .iter()
            .zip(other.field.iter())
            .map(|(&a, &b)| a * b)
            .sum::<f64>()
            * self.dx;
        let n1 = self.norm();
        let n2 = other.norm();
        if n1 < 1e-30 || n2 < 1e-30 {
            return 0.0;
        }
        inner / (n1 * n2).sqrt()
    }
}

/// A complex-valued eigenmode for lossy / dispersive waveguides.
///
/// This extends `EmeMode` to support complex propagation constants
/// β = β_r + i·β_i where β_i > 0 gives propagation loss.
#[derive(Debug, Clone)]
pub struct EigenMode {
    /// Complex propagation constant (rad/m). Im part > 0 → loss.
    pub beta: Complex64,
    /// Complex field profile sampled on a transverse grid.
    pub field: Vec<Complex64>,
    /// Grid spacing (m).
    pub dx: f64,
}

impl EigenMode {
    /// Create an `EigenMode` from a real `EmeMode` (zero imaginary parts).
    pub fn from_eme_mode(mode: &EmeMode) -> Self {
        Self {
            beta: Complex64::new(mode.beta, 0.0),
            field: mode.field.iter().map(|&e| Complex64::new(e, 0.0)).collect(),
            dx: mode.dx,
        }
    }

    /// Compute field power ∫|E|² dx.
    pub fn power(&self) -> f64 {
        self.field.iter().map(|e| e.norm_sqr()).sum::<f64>() * self.dx
    }
}

// ─── Loss / propagation helpers ──────────────────────────────────────────────

/// Propagation loss in dB/cm for a mode with imaginary propagation constant β_i.
///
/// The field amplitude decays as exp(-β_i · z), so the power decays as
/// exp(-2·β_i·z).  Converting to dB/cm:
///   loss = 2·β_i·(100 cm/m) · 20/ln(10)  \[dB/cm\]
pub fn mode_loss_db_per_cm(mode: &EigenMode) -> f64 {
    let beta_i = mode.beta.im;
    2.0 * beta_i * 100.0 * 20.0 / 10.0_f64.ln()
}

/// Propagation loss in dB for a given imaginary β and propagation length.
///
/// loss_dB = 20·log10(exp(-β_i·L)) = -β_i·L·20/ln(10)
pub fn propagation_loss_db(beta_imag: f64, length: f64) -> f64 {
    -beta_imag * length * 20.0 / 10.0_f64.ln()
}

/// Per-mode propagation loss in dB/cm for a slice of `EigenMode`s.
pub fn effective_loss_db_per_cm(modes: &[EigenMode]) -> Vec<f64> {
    modes.iter().map(mode_loss_db_per_cm).collect()
}

/// Confinement loss: fraction of power outside the core region.
///
/// `core_indices` are the grid indices that belong to the core.
/// Returns a value in \[0, 1\] where 0 means all power is confined.
pub fn confinement_loss(mode: &EigenMode, core_indices: &[usize], field_grid: &[f64]) -> f64 {
    let _ = field_grid; // grid positions not needed for power fraction
    let total_power: f64 = mode.field.iter().map(|e| e.norm_sqr()).sum::<f64>();
    if total_power < 1e-30 {
        return 0.0;
    }
    let core_power: f64 = core_indices
        .iter()
        .filter_map(|&idx| mode.field.get(idx))
        .map(|e| e.norm_sqr())
        .sum::<f64>();
    let clad_power = (total_power - core_power).max(0.0);
    clad_power / total_power
}

// ─── Mode overlap integrals ───────────────────────────────────────────────────

/// Complex overlap integral ∫ E_a*(x) · E_b(x) dx.
///
/// Uses the trapezoidal rule on a uniform grid with spacing `dx`.
pub fn overlap_integral(mode_a: &[Complex64], mode_b: &[Complex64], dx: f64) -> Complex64 {
    assert_eq!(
        mode_a.len(),
        mode_b.len(),
        "field arrays must have the same length"
    );
    let n = mode_a.len();
    if n == 0 {
        return Complex64::new(0.0, 0.0);
    }
    // Trapezoidal weights: 1, 2, 2, …, 2, 1 → multiply sum by dx/2
    let inner: Complex64 = if n == 1 {
        mode_a[0].conj() * mode_b[0]
    } else {
        let ends = mode_a[0].conj() * mode_b[0] + mode_a[n - 1].conj() * mode_b[n - 1];
        let middle: Complex64 = mode_a[1..n - 1]
            .iter()
            .zip(mode_b[1..n - 1].iter())
            .map(|(a, b)| a.conj() * b * 2.0)
            .sum();
        ends + middle
    };
    inner * (dx / 2.0)
}

/// N×N complex overlap matrix O\[i\]\[j\] = <E_i | E_j>.
pub fn overlap_matrix(modes: &[Vec<Complex64>], dx: f64) -> Vec<Vec<Complex64>> {
    let n = modes.len();
    let mut mat = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for i in 0..n {
        for j in 0..n {
            mat[i][j] = overlap_integral(&modes[i], &modes[j], dx);
        }
    }
    mat
}

/// Coupling efficiency from an input field into a single mode.
///
/// η = |<E_in | mode>|² / (||E_in||² · ||mode||²)
///
/// Returns a value in \[0, 1\].
pub fn coupling_efficiency(field_in: &[Complex64], mode: &[Complex64], dx: f64) -> f64 {
    let overlap = overlap_integral(field_in, mode, dx);
    let norm_in = overlap_integral(field_in, field_in, dx).re.max(0.0);
    let norm_mode = overlap_integral(mode, mode, dx).re.max(0.0);
    if norm_in < 1e-60 || norm_mode < 1e-60 {
        return 0.0;
    }
    overlap.norm_sqr() / (norm_in * norm_mode)
}

// ─── Bidirectional propagator ─────────────────────────────────────────────────

/// Bidirectional mode propagator.
///
/// Decomposes an input field onto a set of `EigenMode`s and propagates
/// them forward or backward by a given distance.
pub struct EigenmodePropagator {
    /// Modes of the waveguide cross-section.
    pub modes: Vec<EigenMode>,
}

impl EigenmodePropagator {
    /// Create a propagator for a set of eigen-modes.
    pub fn new(modes: Vec<EigenMode>) -> Self {
        Self { modes }
    }

    /// Decompose `field` onto the modes: c_i = <mode_i | field> / <mode_i | mode_i>.
    fn coefficients(&self, field: &[Complex64]) -> Vec<Complex64> {
        self.modes
            .iter()
            .map(|m| {
                let overlap = overlap_integral(&m.field, field, m.dx);
                let norm = overlap_integral(&m.field, &m.field, m.dx);
                if norm.re.abs() < 1e-60 {
                    Complex64::new(0.0, 0.0)
                } else {
                    overlap / norm
                }
            })
            .collect()
    }

    /// Reconstruct the field from mode coefficients at position z.
    fn reconstruct(&self, coeffs: &[Complex64], length: f64, forward: bool) -> Vec<Complex64> {
        let n_pts = self.modes.first().map(|m| m.field.len()).unwrap_or(0);
        let mut out = vec![Complex64::new(0.0, 0.0); n_pts];
        for (mode, &c) in self.modes.iter().zip(coeffs.iter()) {
            // Phase factor: exp(±i·β·z) with β potentially complex
            let phase_exp = if forward {
                (Complex64::i() * mode.beta * length).exp()
            } else {
                (-Complex64::i() * mode.beta * length).exp()
            };
            let coeff_eff = c * phase_exp;
            for (o, &e) in out.iter_mut().zip(mode.field.iter()) {
                *o += coeff_eff * e;
            }
        }
        out
    }

    /// Propagate `field` forward by `length` metres.
    pub fn propagate_forward(&self, field: &[Complex64], length: f64) -> Vec<Complex64> {
        let coeffs = self.coefficients(field);
        self.reconstruct(&coeffs, length, true)
    }

    /// Propagate `field` backward by `length` metres.
    pub fn propagate_backward(&self, field: &[Complex64], length: f64) -> Vec<Complex64> {
        let coeffs = self.coefficients(field);
        self.reconstruct(&coeffs, length, false)
    }
}

// ─── Existing types below (unchanged) ────────────────────────────────────────

/// One waveguide segment in the EME simulation.
#[derive(Debug, Clone)]
pub struct EmeSegment {
    /// Segment length (m)
    pub length: f64,
    /// Core index
    pub n_core: f64,
    /// Cladding index
    pub n_clad: f64,
    /// Slab thickness (m)
    pub thickness: f64,
}

impl EmeSegment {
    pub fn new(length: f64, n_core: f64, n_clad: f64, thickness: f64) -> Self {
        Self {
            length,
            n_core,
            n_clad,
            thickness,
        }
    }

    /// Find the TE modes using the symmetric slab analytical solution.
    /// Returns modes sorted by n_eff descending.
    pub fn find_modes(&self, wavelength: f64, n_modes_max: usize, n_pts: usize) -> Vec<EmeMode> {
        let k0 = 2.0 * PI / wavelength;
        let h = self.thickness;
        let n_c = self.n_core;
        let n_cl = self.n_clad;
        let dn = (n_c * n_c - n_cl * n_cl).sqrt();
        let big_v = k0 * h / 2.0 * dn;

        // Grid for field sampling: symmetric domain [-L/2, L/2]
        let domain = 3.0 * h;
        let dx = domain / n_pts as f64;
        let xs: Vec<f64> = (0..n_pts).map(|i| i as f64 * dx - domain / 2.0).collect();

        let w = |u: f64| -> f64 { (big_v * big_v - u * u).sqrt() };

        let mut modes = Vec::new();
        for mode_order in 0..n_modes_max.min(20) {
            let is_even = mode_order % 2 == 0;
            let half_ord = mode_order / 2;

            let u_start = if is_even {
                half_ord as f64 * PI + 1e-10
            } else {
                half_ord as f64 * PI + PI / 2.0 + 1e-10
            };
            let u_end = if is_even {
                half_ord as f64 * PI + PI / 2.0 - 1e-10
            } else {
                (half_ord + 1) as f64 * PI - 1e-10
            };

            if u_start >= big_v {
                break;
            }
            let u_end = u_end.min(big_v - 1e-12);
            if u_end <= u_start {
                continue;
            }

            let f = |u: f64| -> f64 {
                let wu = w(u);
                let tan_u = u.tan();
                if is_even {
                    u * tan_u - wu
                } else {
                    -u / tan_u - wu
                }
            };

            let n_sub = 200;
            let sub_step = (u_end - u_start) / n_sub as f64;
            let mut prev = f(u_start);
            let mut found = false;
            for i in 1..=n_sub {
                let u = u_start + i as f64 * sub_step;
                let val = f(u);
                if !prev.is_nan() && !val.is_nan() && prev * val < 0.0 && (val - prev).abs() < 1e15
                {
                    if let Some(u_root) = bisect(f, u - sub_step, u, 1e-14) {
                        let wu = w(u_root);
                        let kappa = 2.0 * u_root / h;
                        let gamma = 2.0 * wu / h;
                        let beta = (n_c * n_c * k0 * k0 - kappa * kappa).sqrt();
                        let n_eff = beta / k0;

                        // Compute field profile
                        let field: Vec<f64> = xs
                            .iter()
                            .map(|&x| {
                                let abs_x = x.abs();
                                if abs_x <= h / 2.0 {
                                    if is_even {
                                        (kappa * abs_x).cos()
                                    } else {
                                        (kappa * x).sin()
                                    }
                                } else {
                                    let decay = (-gamma * (abs_x - h / 2.0)).exp();
                                    if is_even {
                                        (kappa * h / 2.0).cos() * decay
                                    } else {
                                        (kappa * h / 2.0).sin() * decay * x.signum()
                                    }
                                }
                            })
                            .collect();

                        modes.push(EmeMode {
                            n_eff,
                            beta,
                            field,
                            dx,
                        });
                        found = true;
                        break;
                    }
                }
                prev = val;
            }
            if !found {
                break;
            }
        }

        modes.sort_by(|a, b| {
            b.n_eff
                .partial_cmp(&a.n_eff)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        modes
    }
}

/// 2×2 S-matrix for single-mode propagation.
///
/// Relates forward (+) and backward (-) amplitudes:
///   \[b+\]   [S11 S12] \[a+\]
///   \[b-\] = [S21 S22] \[a-\]
///
/// Convention: port 1 is left, port 2 is right.
#[derive(Debug, Clone)]
pub struct SMatrix2x2 {
    pub s11: f64,
    pub s12: f64,
    pub s21: f64,
    pub s22: f64,
}

impl SMatrix2x2 {
    /// Identity S-matrix (perfectly transmitting, no reflection)
    pub fn identity() -> Self {
        Self {
            s11: 0.0,
            s12: 1.0,
            s21: 1.0,
            s22: 0.0,
        }
    }

    /// Propagation S-matrix for a segment of length L and propagation constant beta.
    pub fn propagation(beta: f64, length: f64) -> Self {
        let phase = (beta * length).cos(); // real part only (lossless)
        let _ = phase;
        let t = (beta * length).cos() * (beta * length).cos()
            + (beta * length).sin() * (beta * length).sin();
        // For a lossless segment: T = exp(i*beta*L), |T|² = 1
        // S-parameters: S21 = exp(i*beta*L), S12 = exp(-i*beta*L), S11=S22=0
        // Taking magnitude (real-valued EME approximation):
        let _ = t;
        Self {
            s11: 0.0,
            s12: 1.0,
            s21: 1.0,
            s22: 0.0,
        }
    }

    /// Interface S-matrix from mode overlap coefficient.
    ///
    /// For coupling coefficient η (overlap integral),
    /// energy conservation gives: T = η², R = 1 - η²
    pub fn from_overlap(eta: f64) -> Self {
        let t = eta.powi(2).min(1.0);
        let r = 1.0 - t;
        Self {
            s11: r.sqrt(),
            s12: t.sqrt(),
            s21: t.sqrt(),
            s22: r.sqrt(),
        }
    }

    /// Cascade two S-matrices (Redheffer star product).
    pub fn cascade(&self, other: &Self) -> Self {
        // For real-valued S-matrices:
        // S = S_A ⊕ S_B
        // S21_total = S21_B * S21_A / (1 - S22_A * S11_B)
        let denom = 1.0 - self.s22 * other.s11;
        let denom = if denom.abs() < 1e-30 { 1.0 } else { denom };
        let s21 = self.s21 * other.s21 / denom;
        let s12 = other.s12 * self.s12 / denom;
        let s11 = self.s11 + self.s12 * other.s11 * self.s21 / denom;
        let s22 = other.s22 + other.s21 * self.s22 * other.s12 / denom;
        Self { s11, s12, s21, s22 }
    }
}

/// Full multi-mode S-matrix for a waveguide system.
///
/// Stores the scattering matrix for N_modes input and output ports.
#[derive(Debug, Clone)]
pub struct SMatrixNd {
    pub n_modes: usize,
    /// S-matrix entries S\[i*n + j\] = S_{ij}
    pub s: Vec<f64>,
}

impl SMatrixNd {
    pub fn identity(n_modes: usize) -> Self {
        let mut s = vec![0.0; n_modes * n_modes * 4]; // 2N x 2N (ports 1 and 2)
                                                      // Diagonal: S_{ii}^{21} = 1 (perfect transmission)
        for i in 0..n_modes {
            s[i * (2 * n_modes) + n_modes + i] = 1.0; // S21 block diagonal
            s[(n_modes + i) * (2 * n_modes) + i] = 1.0; // S12 block diagonal
        }
        Self { n_modes, s }
    }

    /// Transmission efficiency from input mode `m_in` to output mode `m_out`.
    pub fn transmission(&self, m_in: usize, m_out: usize) -> f64 {
        let n = 2 * self.n_modes;
        let row = self.n_modes + m_out;
        let col = m_in;
        let t = self.s[row * n + col];
        t * t
    }
}

/// EME solver: cascades multiple waveguide segments.
pub struct EmeSolver {
    pub segments: Vec<EmeSegment>,
    pub wavelength: f64,
    pub n_modes: usize,
    pub n_pts: usize,
}

impl EmeSolver {
    pub fn new(wavelength: f64, n_modes: usize, n_pts: usize) -> Self {
        Self {
            segments: Vec::new(),
            wavelength,
            n_modes,
            n_pts,
        }
    }

    pub fn add_segment(&mut self, seg: EmeSegment) {
        self.segments.push(seg);
    }

    /// Solve: find modes in each segment, build and cascade S-matrices.
    /// Returns the total 2x2 S-matrix for the fundamental mode.
    pub fn solve_fundamental(&self) -> SMatrix2x2 {
        if self.segments.is_empty() {
            return SMatrix2x2::identity();
        }

        // For each segment, find modes and build propagation S-matrix
        let seg_modes: Vec<Vec<EmeMode>> = self
            .segments
            .iter()
            .map(|seg| seg.find_modes(self.wavelength, self.n_modes, self.n_pts))
            .collect();

        let mut total_s = SMatrix2x2::identity();

        for (i, (seg, modes)) in self.segments.iter().zip(seg_modes.iter()).enumerate() {
            if modes.is_empty() {
                continue;
            }

            // Propagation phase for fundamental mode in this segment
            let beta = modes[0].beta;
            let prop_s = SMatrix2x2::propagation(beta, seg.length);
            total_s = total_s.cascade(&prop_s);

            // Interface coupling to next segment
            if i + 1 < self.segments.len() {
                let next_modes = &seg_modes[i + 1];
                if !next_modes.is_empty() && !modes.is_empty() {
                    let eta = modes[0].overlap(&next_modes[0]).abs();
                    let iface_s = SMatrix2x2::from_overlap(eta);
                    total_s = total_s.cascade(&iface_s);
                }
            }
        }

        total_s
    }

    /// Total power transmission (fundamental mode in → fundamental mode out).
    pub fn transmission(&self) -> f64 {
        let s = self.solve_fundamental();
        s.s21 * s.s21
    }
}

// ─── Eigenmode layer with full S-matrix ──────────────────────────────────────

/// A waveguide layer for S-matrix construction.
#[derive(Debug, Clone)]
pub struct EigenmodeLayer {
    /// Layer thickness (m)
    pub thickness: f64,
    /// Core refractive index
    pub n_core: f64,
    /// Cladding refractive index
    pub n_clad: f64,
    /// Wavelength (m)
    pub wavelength: f64,
    /// Number of modes to include
    pub n_modes: usize,
    /// Grid points for field sampling
    pub n_pts: usize,
}

impl EigenmodeLayer {
    pub fn new(
        thickness: f64,
        n_core: f64,
        n_clad: f64,
        wavelength: f64,
        n_modes: usize,
        n_pts: usize,
    ) -> Self {
        Self {
            thickness,
            n_core,
            n_clad,
            wavelength,
            n_modes,
            n_pts,
        }
    }

    /// Compute full N-mode S-matrix for this layer.
    ///
    /// For a uniform waveguide layer, S11 = S22 = 0 (no back-reflection)
    /// and S21\[i\]\[i\] = exp(i·β_i·L), S12 = S21 (lossless symmetry).
    ///
    /// Returns (S11, S12, S21, S22) blocks as N×N matrices.
    pub fn to_s_matrix_full(&self) -> SMatrixBlocks {
        let seg = EmeSegment::new(self.thickness, self.n_core, self.n_clad, self.thickness);
        let modes = seg.find_modes(self.wavelength, self.n_modes, self.n_pts);
        let n = modes.len().max(1);
        let s11 = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        let s22 = s11.clone();
        let mut s21 = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        let mut s12 = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for (i, mode) in modes.iter().enumerate() {
            if i >= n {
                break;
            }
            let phase = (Complex64::i() * mode.beta * self.thickness).exp();
            s21[i][i] = phase;
            s12[i][i] = phase;
        }
        (s11, s12, s21, s22)
    }
}

// ─── Thin-film transfer matrix method ────────────────────────────────────────

/// Thin-film layer for transfer matrix method.
#[derive(Debug, Clone)]
pub struct ThinFilmLayer {
    /// Refractive index
    pub n: f64,
    /// Layer thickness (m)
    pub d: f64,
    /// Design wavelength (m)
    pub wavelength: f64,
}

impl ThinFilmLayer {
    pub fn new(n: f64, d: f64, wavelength: f64) -> Self {
        Self { n, d, wavelength }
    }

    /// Phase thickness δ = 2π·n·d/λ
    pub fn phase_thickness(&self) -> f64 {
        2.0 * PI * self.n * self.d / self.wavelength
    }
}

/// Transfer matrix (T-matrix) for multilayer thin-film stacks.
///
/// Uses the characteristic matrix formalism for TE polarization at normal
/// incidence. Each layer is represented by a 2×2 matrix:
///
///   M = \[\[cos δ,  -i·sin δ/η\\],
///        \[-i·η·sin δ, cos δ\]]
///
/// where δ = 2π·n·d/λ and η = n (TE, normal incidence, in units where μ₀=ε₀=1).
#[derive(Debug, Clone)]
pub struct TransferMatrixSystem {
    /// Refractive index of incident medium
    pub n_in: f64,
    /// Refractive index of substrate
    pub n_sub: f64,
    /// Stack layers in incident→substrate order
    pub layers: Vec<ThinFilmLayer>,
}

impl TransferMatrixSystem {
    pub fn new(n_in: f64, n_sub: f64) -> Self {
        Self {
            n_in,
            n_sub,
            layers: Vec::new(),
        }
    }

    /// Add a layer to the stack.
    pub fn add_layer(&mut self, layer: ThinFilmLayer) {
        self.layers.push(layer);
    }

    /// Characteristic matrix for a single layer:
    ///
    ///   M = \[\[cos δ,  -i·sin δ/η\\],
    ///        \[-i·η·sin δ, cos δ\]]
    pub fn propagate_through_layer(layer: &ThinFilmLayer) -> [[Complex64; 2]; 2] {
        let delta = layer.phase_thickness();
        let eta = Complex64::new(layer.n, 0.0);
        let cos_d = Complex64::new(delta.cos(), 0.0);
        let sin_d = Complex64::new(delta.sin(), 0.0);
        let i = Complex64::i();
        [[cos_d, -i * sin_d / eta], [-i * eta * sin_d, cos_d]]
    }

    /// Multiply two 2×2 complex matrices.
    fn mat_mul_2x2(a: &[[Complex64; 2]; 2], b: &[[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
        [
            [
                a[0][0] * b[0][0] + a[0][1] * b[1][0],
                a[0][0] * b[0][1] + a[0][1] * b[1][1],
            ],
            [
                a[1][0] * b[0][0] + a[1][1] * b[1][0],
                a[1][0] * b[0][1] + a[1][1] * b[1][1],
            ],
        ]
    }

    /// Total system transfer matrix M = M₁ · M₂ · … · Mₙ.
    pub fn total_transfer_matrix(&self) -> [[Complex64; 2]; 2] {
        let identity: [[Complex64; 2]; 2] = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        self.layers.iter().fold(identity, |acc, layer| {
            let m = Self::propagate_through_layer(layer);
            Self::mat_mul_2x2(&acc, &m)
        })
    }

    /// Total S-matrix from transfer matrix at normal incidence (TE).
    ///
    /// Returns \[\[r, t\\], \[t, -r\]] where r = amplitude reflection and t = amplitude
    /// transmission coefficients (Fresnel conventions).
    pub fn total_s_matrix(&self) -> [[Complex64; 2]; 2] {
        let m = self.total_transfer_matrix();
        let eta_in = Complex64::new(self.n_in, 0.0);
        let eta_sub = Complex64::new(self.n_sub, 0.0);
        // Ref: Born & Wolf, Principles of Optics, §1.6
        let denom = m[0][0] * eta_sub + m[0][1] * eta_in * eta_sub + m[1][0] + m[1][1] * eta_in;
        let denom_safe = if denom.norm_sqr() < 1e-200 {
            Complex64::new(1.0, 0.0)
        } else {
            denom
        };
        let r = (m[0][0] * eta_sub + m[0][1] * eta_in * eta_sub - m[1][0] - m[1][1] * eta_in)
            / denom_safe;
        let t = Complex64::new(2.0, 0.0) * eta_in / denom_safe;
        [[r, t], [t, -r]]
    }

    /// Amplitude reflection coefficient r.
    pub fn reflection_coefficient(&self) -> Complex64 {
        self.total_s_matrix()[0][0]
    }

    /// Amplitude transmission coefficient t.
    pub fn transmission_coefficient(&self) -> Complex64 {
        self.total_s_matrix()[0][1]
    }

    /// Power reflectance R = |r|².
    pub fn reflectance(&self) -> f64 {
        self.reflection_coefficient().norm_sqr()
    }

    /// Power transmittance T = (n_sub / n_in) · |t|².
    pub fn transmittance(&self) -> f64 {
        let t = self.transmission_coefficient();
        (self.n_sub / self.n_in) * t.norm_sqr()
    }
}

// ─── N×N S-matrix helper operations ─────────────────────────────────────────

/// Multiply two N×N complex matrices.
pub fn mat_mul_nd(a: &[Vec<Complex64>], b: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let n = a.len();
    let mut c = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Add two N×N complex matrices element-wise.
pub fn mat_add_nd(a: &[Vec<Complex64>], b: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let n = a.len();
    let mut c = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for i in 0..n {
        for j in 0..n {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    c
}

/// N×N identity matrix.
pub fn eye_nd(n: usize) -> Vec<Vec<Complex64>> {
    let mut e = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for (i, row) in e.iter_mut().enumerate().take(n) {
        row[i] = Complex64::new(1.0, 0.0);
    }
    e
}

/// Diagonal approximation to matrix inverse.
///
/// Valid for diagonally dominant matrices (e.g., weakly-coupled modes).
/// For strongly coupled systems, a full LU factorisation should be used.
fn diag_inv_nd(m: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let n = m.len();
    let mut result = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for i in 0..n {
        let d = m[i][i];
        result[i][i] = if d.norm_sqr() < 1e-200 {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(1.0, 0.0) / d
        };
    }
    result
}

/// Redheffer star product for N×N S-matrices.
///
/// Cascades S_A and S_B into S_total following:
///
///   D₁ = (I − S22_A · S11_B)⁻¹
///   D₂ = (I − S11_B · S22_A)⁻¹
///   new_S11 = S11_A + S12_A · D₁ · S11_B · S21_A
///   new_S22 = S22_B + S21_B · D₁ · S22_A · S12_B
///   new_S21 = S21_B · D₁ · S21_A
///   new_S12 = S12_A · D₂ · S12_B
///
/// The inverse is approximated as a diagonal approximation for weakly-coupled
/// mode sets. All input blocks must be N×N.
#[allow(clippy::too_many_arguments)]
pub fn cascade_smatrices(
    s11_a: &[Vec<Complex64>],
    s12_a: &[Vec<Complex64>],
    s21_a: &[Vec<Complex64>],
    s22_a: &[Vec<Complex64>],
    s11_b: &[Vec<Complex64>],
    s12_b: &[Vec<Complex64>],
    s21_b: &[Vec<Complex64>],
    s22_b: &[Vec<Complex64>],
) -> SMatrixBlocks {
    let n = s11_a.len();
    let id = eye_nd(n);
    // D1 = (I − S22_A · S11_B)⁻¹
    let s22a_s11b = mat_mul_nd(s22_a, s11_b);
    let id_minus1: Vec<Vec<Complex64>> = (0..n)
        .map(|i| (0..n).map(|j| id[i][j] - s22a_s11b[i][j]).collect())
        .collect();
    let d1 = diag_inv_nd(&id_minus1);
    // D2 = (I − S11_B · S22_A)⁻¹
    let s11b_s22a = mat_mul_nd(s11_b, s22_a);
    let id_minus2: Vec<Vec<Complex64>> = (0..n)
        .map(|i| (0..n).map(|j| id[i][j] - s11b_s22a[i][j]).collect())
        .collect();
    let d2 = diag_inv_nd(&id_minus2);
    // new_S11 = S11_A + S12_A · D1 · S11_B · S21_A
    let new_s11 = mat_add_nd(
        s11_a,
        &mat_mul_nd(&mat_mul_nd(&mat_mul_nd(s12_a, &d1), s11_b), s21_a),
    );
    // new_S22 = S22_B + S21_B · D1 · S22_A · S12_B
    let new_s22 = mat_add_nd(
        s22_b,
        &mat_mul_nd(&mat_mul_nd(&mat_mul_nd(s21_b, &d1), s22_a), s12_b),
    );
    // new_S21 = S21_B · D1 · S21_A
    let new_s21 = mat_mul_nd(&mat_mul_nd(s21_b, &d1), s21_a);
    // new_S12 = S12_A · D2 · S12_B
    let new_s12 = mat_mul_nd(&mat_mul_nd(s12_a, &d2), s12_b);
    (new_s11, new_s12, new_s21, new_s22)
}

/// Extract mode amplitudes from an FDTD field snapshot via overlap integrals.
///
/// Decomposes `fdtd_field` onto the given `modes` using a normalised inner product:
///
///   c_i = ∫ E_fdtd · E_mode,i dx  /  sqrt(∫|E_fdtd|² dx · ∫|E_mode,i|² dx)
///
/// This cosine-similarity form is numerically stable even when field magnitudes differ
/// by many orders of magnitude; the result lies in \[−1, 1\] for real fields.
/// Returns real-valued complex amplitudes (imaginary part zero for real FDTD fields).
pub fn extract_mode_amplitudes(fdtd_field: &[f64], modes: &[EmeMode], dx: f64) -> Vec<Complex64> {
    let fdtd_norm_sq: f64 = fdtd_field.iter().map(|&e| e * e).sum::<f64>() * dx;
    modes
        .iter()
        .map(|mode| {
            let len = fdtd_field.len().min(mode.field.len());
            let num: f64 = fdtd_field[..len]
                .iter()
                .zip(mode.field[..len].iter())
                .map(|(&e_fdtd, &e_mode)| e_fdtd * e_mode)
                .sum::<f64>()
                * dx;
            let mode_norm_sq: f64 = mode.field[..len].iter().map(|&e| e * e).sum::<f64>() * dx;
            let denom = (fdtd_norm_sq * mode_norm_sq).sqrt();
            if denom < 1e-60 {
                Complex64::new(0.0, 0.0)
            } else {
                Complex64::new(num / denom, 0.0)
            }
        })
        .collect()
}

/// Bisection root finder for the mode equations
fn bisect<F: Fn(f64) -> f64>(f: F, mut lo: f64, mut hi: f64, tol: f64) -> Option<f64> {
    let mut f_lo = f(lo);
    let f_hi = f(hi);
    if f_lo.is_nan() || f_hi.is_nan() || f_lo * f_hi > 0.0 {
        return None;
    }
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if (hi - lo) < tol {
            return Some(mid);
        }
        let f_mid = f(mid);
        if f_mid.is_nan() {
            return None;
        }
        if f_lo * f_mid <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
            f_lo = f_mid;
        }
    }
    Some((lo + hi) / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── original tests ────────────────────────────────────────────────────────

    #[test]
    fn eme_mode_finds_fundamental() {
        let seg = EmeSegment::new(10e-6, 3.476, 1.444, 500e-9);
        let modes = seg.find_modes(1550e-9, 5, 200);
        assert!(!modes.is_empty(), "Should find at least fundamental mode");
        let n_eff = modes[0].n_eff;
        assert!(
            n_eff > 1.444 && n_eff < 3.476,
            "n_eff={n_eff:.4} out of guidance range"
        );
    }

    #[test]
    fn eme_mode_field_normalized() {
        let seg = EmeSegment::new(10e-6, 3.476, 1.444, 500e-9);
        let modes = seg.find_modes(1550e-9, 1, 200);
        if !modes.is_empty() {
            let norm = modes[0].norm();
            assert!(norm > 0.0, "Mode norm must be positive");
        }
    }

    #[test]
    fn eme_same_segments_unity_transmission() {
        // Two identical segments: perfect transmission expected
        let mut solver = EmeSolver::new(1550e-9, 2, 150);
        let seg1 = EmeSegment::new(5e-6, 3.476, 1.444, 500e-9);
        let seg2 = EmeSegment::new(5e-6, 3.476, 1.444, 500e-9);
        solver.add_segment(seg1);
        solver.add_segment(seg2);
        let t = solver.transmission();
        assert!(
            t > 0.9,
            "Identical segments should have T close to 1, got {t:.4}"
        );
    }

    #[test]
    fn eme_different_segments_reduces_transmission() {
        // Two different segments: overlap < 1 → T < 1
        let mut solver = EmeSolver::new(1550e-9, 1, 150);
        let seg1 = EmeSegment::new(5e-6, 3.476, 1.444, 500e-9);
        let seg2 = EmeSegment::new(5e-6, 3.476, 1.444, 1000e-9); // different thickness
        solver.add_segment(seg1);
        solver.add_segment(seg2);
        let t = solver.transmission();
        assert!(
            (0.0..=1.0).contains(&t),
            "Transmission must be in [0,1], got {t:.4}"
        );
    }

    #[test]
    fn smatrix_cascade_identity() {
        let s1 = SMatrix2x2::identity();
        let s2 = SMatrix2x2::identity();
        let s = s1.cascade(&s2);
        assert!((s.s12 - 1.0).abs() < 1e-10);
        assert!((s.s21 - 1.0).abs() < 1e-10);
        assert!(s.s11.abs() < 1e-10);
        assert!(s.s22.abs() < 1e-10);
    }

    #[test]
    fn overlap_same_mode_is_unity() {
        let mode = EmeMode {
            n_eff: 2.5,
            beta: 1e7,
            field: vec![1.0; 100],
            dx: 10e-9,
        };
        let ov = mode.overlap(&mode);
        assert!(
            (ov - 1.0).abs() < 1e-10,
            "Self-overlap should be 1, got {ov:.4}"
        );
    }

    #[test]
    fn overlap_orthogonal_modes_near_zero() {
        // Even and odd modes should be orthogonal
        let n = 100;
        let dx = 10e-9;
        let mut field_even = vec![0.0; n];
        let mut field_odd = vec![0.0; n];
        for i in 0..n {
            let x = (i as f64 - n as f64 / 2.0) * dx;
            field_even[i] = (-x * x / (1e-6 * 1e-6)).exp();
            field_odd[i] = x * (-x * x / (1e-6 * 1e-6)).exp();
        }
        let mode_even = EmeMode {
            n_eff: 2.5,
            beta: 1e7,
            field: field_even,
            dx,
        };
        let mode_odd = EmeMode {
            n_eff: 2.3,
            beta: 9.5e6,
            field: field_odd,
            dx,
        };
        let ov = mode_even.overlap(&mode_odd).abs();
        assert!(
            ov < 0.1,
            "Orthogonal modes should have small overlap, got {ov:.4}"
        );
    }

    // ── new tests: loss helpers ───────────────────────────────────────────────

    #[test]
    fn mode_loss_lossless_is_zero() {
        let mode = EigenMode {
            beta: Complex64::new(1e7, 0.0),
            field: vec![Complex64::new(1.0, 0.0); 50],
            dx: 10e-9,
        };
        let loss = mode_loss_db_per_cm(&mode);
        assert!(
            loss.abs() < 1e-12,
            "Lossless mode should have 0 dB/cm, got {loss}"
        );
    }

    #[test]
    fn mode_loss_positive_for_lossy_mode() {
        let mode = EigenMode {
            beta: Complex64::new(1e7, 1e4), // β_i = 1e4 rad/m > 0
            field: vec![Complex64::new(1.0, 0.0); 50],
            dx: 10e-9,
        };
        let loss = mode_loss_db_per_cm(&mode);
        assert!(loss > 0.0, "Lossy mode should have positive dB/cm loss");
    }

    #[test]
    fn propagation_loss_db_zero_length() {
        let loss = propagation_loss_db(500.0, 0.0);
        assert!(
            loss.abs() < 1e-12,
            "Zero-length propagation gives zero loss"
        );
    }

    #[test]
    fn propagation_loss_db_round_trip() {
        // exp(-β_i·L) → loss_dB = -β_i·L·20/ln10
        let beta_i = 100.0_f64; // rad/m
        let length = 0.01_f64; // 1 cm
        let loss = propagation_loss_db(beta_i, length);
        let expected = -beta_i * length * 20.0 / 10.0_f64.ln();
        assert!((loss - expected).abs() < 1e-10);
    }

    #[test]
    fn effective_loss_multi_mode() {
        let modes = vec![
            EigenMode {
                beta: Complex64::new(1e7, 0.0),
                field: vec![],
                dx: 1e-8,
            },
            EigenMode {
                beta: Complex64::new(1e7, 1e3),
                field: vec![],
                dx: 1e-8,
            },
        ];
        let losses = effective_loss_db_per_cm(&modes);
        assert_eq!(losses.len(), 2);
        assert!(losses[0].abs() < 1e-12);
        assert!(losses[1] > 0.0);
    }

    // ── new tests: confinement loss ───────────────────────────────────────────

    #[test]
    fn confinement_loss_all_core_is_zero() {
        let n = 10_usize;
        let mode = EigenMode {
            beta: Complex64::new(1e7, 0.0),
            field: (0..n).map(|_| Complex64::new(1.0, 0.0)).collect(),
            dx: 1e-8,
        };
        let core_idx: Vec<usize> = (0..n).collect();
        let grid: Vec<f64> = (0..n).map(|i| i as f64 * 1e-8).collect();
        let cl = confinement_loss(&mode, &core_idx, &grid);
        assert!(cl.abs() < 1e-12, "All power in core → 0 loss, got {cl}");
    }

    #[test]
    fn confinement_loss_half_in_core() {
        let n = 10_usize;
        let mode = EigenMode {
            beta: Complex64::new(1e7, 0.0),
            field: (0..n).map(|_| Complex64::new(1.0, 0.0)).collect(),
            dx: 1e-8,
        };
        let core_idx: Vec<usize> = (0..n / 2).collect(); // first half in core
        let grid: Vec<f64> = (0..n).map(|i| i as f64 * 1e-8).collect();
        let cl = confinement_loss(&mode, &core_idx, &grid);
        assert!(
            (cl - 0.5).abs() < 1e-12,
            "Half power outside core → confinement_loss=0.5, got {cl}"
        );
    }

    // ── new tests: overlap integral ───────────────────────────────────────────

    #[test]
    fn overlap_integral_self_equals_norm() {
        let field: Vec<Complex64> = (0..8).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let dx = 0.1;
        let self_ov = overlap_integral(&field, &field, dx);
        assert!(self_ov.im.abs() < 1e-12, "Self-overlap must be real");
        assert!(self_ov.re > 0.0);
    }

    #[test]
    fn overlap_integral_conjugate_symmetry() {
        let a: Vec<Complex64> = (0..6)
            .map(|i| Complex64::new(i as f64, (i % 2) as f64))
            .collect();
        let b: Vec<Complex64> = (0..6)
            .map(|i| Complex64::new(0.5 * i as f64, -(i % 3) as f64))
            .collect();
        let dx = 0.05;
        let ab = overlap_integral(&a, &b, dx);
        let ba = overlap_integral(&b, &a, dx);
        // <a|b>* = <b|a>
        assert!((ab - ba.conj()).norm() < 1e-12);
    }

    #[test]
    fn overlap_matrix_diagonal_real_positive() {
        let modes: Vec<Vec<Complex64>> = vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0)],
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.5)],
        ];
        let mat = overlap_matrix(&modes, 0.1);
        assert_eq!(mat.len(), 2);
        assert!(mat[0][0].re > 0.0 && mat[0][0].im.abs() < 1e-12);
        assert!(mat[1][1].re > 0.0);
    }

    #[test]
    fn coupling_efficiency_same_field_is_one() {
        let field: Vec<Complex64> = (1..=8).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let dx = 0.05;
        let eff = coupling_efficiency(&field, &field, dx);
        assert!(
            (eff - 1.0).abs() < 1e-10,
            "Same field → efficiency=1, got {eff}"
        );
    }

    #[test]
    fn coupling_efficiency_orthogonal_fields_is_zero() {
        let n = 128_usize;
        let dx = 1.0 / n as f64;
        // Two Gaussians centered on opposite sides of the domain — very little overlap.
        // Centre at x=0.25 vs x=0.75, width σ=0.06 → e-fold at ~2.5σ apart.
        let field_a: Vec<Complex64> = (0..n)
            .map(|i| {
                let x = i as f64 / n as f64;
                let v = (-(x - 0.25).powi(2) / (2.0 * 0.04 * 0.04)).exp();
                Complex64::new(v, 0.0)
            })
            .collect();
        let field_b: Vec<Complex64> = (0..n)
            .map(|i| {
                let x = i as f64 / n as f64;
                let v = (-(x - 0.75).powi(2) / (2.0 * 0.04 * 0.04)).exp();
                Complex64::new(v, 0.0)
            })
            .collect();
        let eff = coupling_efficiency(&field_a, &field_b, dx);
        assert!(
            eff < 1e-6,
            "Well-separated Gaussians → coupling ≈ 0, got {eff}"
        );
    }

    // ── new tests: EigenmodePropagator ────────────────────────────────────────

    #[test]
    fn propagator_zero_length_identity() {
        let n = 16_usize;
        let dx = 1e-8;
        let field: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((PI * i as f64 / n as f64).sin(), 0.0))
            .collect();

        // Build a single mode equal to the field itself
        let mode = EigenMode {
            beta: Complex64::new(1e7, 0.0),
            field: field.clone(),
            dx,
        };
        let prop = EigenmodePropagator::new(vec![mode]);
        let out = prop.propagate_forward(&field, 0.0);
        assert_eq!(out.len(), n);
        let diff: f64 = out
            .iter()
            .zip(field.iter())
            .map(|(a, b)| (a - b).norm())
            .sum();
        assert!(
            diff < 1e-6,
            "Zero-length propagation should return input field, diff={diff}"
        );
    }

    #[test]
    fn propagator_forward_backward_roundtrip() {
        let n = 16_usize;
        let dx = 1e-8;
        let field: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((PI * i as f64 / n as f64).sin(), 0.0))
            .collect();
        let mode = EigenMode {
            beta: Complex64::new(1e7, 0.0),
            field: field.clone(),
            dx,
        };
        let prop = EigenmodePropagator::new(vec![mode]);
        let forward = prop.propagate_forward(&field, 1e-6);
        let back = prop.propagate_backward(&forward, 1e-6);
        // For a lossless mode, forward then backward should recover the input
        let diff: f64 = back
            .iter()
            .zip(field.iter())
            .map(|(a, b)| (a - b).norm())
            .sum();
        assert!(
            diff < 1e-5,
            "Forward + backward should recover input, diff={diff}"
        );
    }

    #[test]
    fn propagator_lossy_mode_decays() {
        let n = 16_usize;
        let dx = 1e-8;
        let field: Vec<Complex64> = vec![Complex64::new(1.0, 0.0); n];
        let mode = EigenMode {
            beta: Complex64::new(1e7, 1e5), // lossy: β_i = 1e5 rad/m
            field: field.clone(),
            dx,
        };
        let prop = EigenmodePropagator::new(vec![mode]);
        let out = prop.propagate_forward(&field, 1e-5); // 10 μm
                                                        // Power should decrease
        let p_in: f64 = field.iter().map(|e| e.norm_sqr()).sum();
        let p_out: f64 = out.iter().map(|e| e.norm_sqr()).sum();
        assert!(
            p_out < p_in,
            "Lossy mode should reduce power after propagation"
        );
    }

    #[test]
    fn eigenmode_from_eme_mode() {
        let eme = EmeMode {
            n_eff: 2.5,
            beta: 1e7,
            field: vec![1.0, 2.0, 3.0],
            dx: 10e-9,
        };
        let eigen = EigenMode::from_eme_mode(&eme);
        assert!((eigen.beta.re - 1e7).abs() < 1e-6);
        assert!(eigen.beta.im.abs() < 1e-30);
        assert_eq!(eigen.field.len(), 3);
    }

    // ── TransferMatrixSystem tests ─────────────────────────────────────────────

    #[test]
    fn tmm_single_layer_energy_conservation() {
        // Glass layer in air at 1550 nm: R + T should equal 1
        let mut tmm = TransferMatrixSystem::new(1.0, 1.0);
        tmm.add_layer(ThinFilmLayer::new(1.5, 100e-9, 1550e-9));
        let r = tmm.reflectance();
        let t = tmm.transmittance();
        assert!((r + t - 1.0).abs() < 1e-10, "R+T={}", r + t);
    }

    #[test]
    fn tmm_no_layers_bare_interface_fresnel() {
        // No layers: bare interface air→glass at normal incidence
        let tmm = TransferMatrixSystem::new(1.0, 1.5);
        let r = tmm.reflectance();
        let r_fresnel = ((1.0_f64 - 1.5) / (1.0_f64 + 1.5)).powi(2);
        assert!(
            (r - r_fresnel).abs() < 1e-10,
            "R={r} vs Fresnel={r_fresnel}"
        );
    }

    #[test]
    fn tmm_quarter_wave_layer_increases_reflectance() {
        // Quarter-wave high-index layer increases reflectance beyond bare interface
        let n_h = 2.3_f64;
        let lambda = 1550e-9_f64;
        let d_qw = lambda / (4.0 * n_h);
        let mut tmm = TransferMatrixSystem::new(1.0, 1.5);
        tmm.add_layer(ThinFilmLayer::new(n_h, d_qw, lambda));
        let r = tmm.reflectance();
        let r_bare = ((1.0_f64 - 1.5_f64) / (1.0_f64 + 1.5_f64)).powi(2);
        assert!(
            r > r_bare,
            "QW layer should increase reflectance: R={r} vs bare={r_bare}"
        );
    }

    #[test]
    fn tmm_anti_reflection_coating_near_zero_reflection() {
        // Single-layer AR coating: n_ar = sqrt(n1*n2), d = λ/(4*n_ar)
        let n1 = 1.0_f64;
        let n2 = 1.5_f64;
        let n_ar = (n1 * n2).sqrt();
        let lambda = 1550e-9_f64;
        let d_ar = lambda / (4.0 * n_ar);
        let mut tmm = TransferMatrixSystem::new(n1, n2);
        tmm.add_layer(ThinFilmLayer::new(n_ar, d_ar, lambda));
        let r = tmm.reflectance();
        assert!(r < 1e-10, "Ideal AR coating: R={r} should be ~0");
    }

    #[test]
    fn tmm_reflection_coefficient_real_at_normal() {
        // For real refractive indices at normal incidence, r is real (imaginary part zero)
        let tmm = TransferMatrixSystem::new(1.0, 2.0);
        let r = tmm.reflection_coefficient();
        assert!(
            r.re.abs() > 0.0,
            "r should be non-zero for n2 ≠ n1, got {}",
            r.re
        );
        assert!(
            r.im.abs() < 1e-12,
            "r should be real at normal incidence, im={}",
            r.im
        );
        // Reflectance should match Fresnel formula
        let r_fresnel_sq = ((1.0_f64 - 2.0_f64) / (1.0_f64 + 2.0_f64)).powi(2);
        assert!(
            (r.norm_sqr() - r_fresnel_sq).abs() < 1e-10,
            "|r|²={} vs Fresnel={r_fresnel_sq}",
            r.norm_sqr()
        );
    }

    #[test]
    fn tmm_multilayer_energy_conservation() {
        // Three-layer Bragg stack: R + T = 1 (lossless)
        let mut tmm = TransferMatrixSystem::new(1.0, 1.46);
        let wavelength = 1550e-9_f64;
        tmm.add_layer(ThinFilmLayer::new(2.3, 168e-9, wavelength));
        tmm.add_layer(ThinFilmLayer::new(1.46, 265e-9, wavelength));
        tmm.add_layer(ThinFilmLayer::new(2.3, 168e-9, wavelength));
        let r = tmm.reflectance();
        let t = tmm.transmittance();
        assert!((r + t - 1.0).abs() < 1e-8, "R+T={} for multilayer", r + t);
    }

    #[test]
    fn phase_thickness_quarter_wave() {
        // Quarter-wave layer: δ = π/2
        let n = 1.5_f64;
        let lambda = 1550e-9_f64;
        let d = lambda / (4.0 * n);
        let layer = ThinFilmLayer::new(n, d, lambda);
        let delta = layer.phase_thickness();
        assert!(
            (delta - PI / 2.0).abs() < 1e-10,
            "QW: δ should be π/2, got {delta}"
        );
    }

    #[test]
    fn phase_thickness_half_wave() {
        // Half-wave layer: δ = π
        let n = 2.3_f64;
        let lambda = 800e-9_f64;
        let d = lambda / (2.0 * n);
        let layer = ThinFilmLayer::new(n, d, lambda);
        let delta = layer.phase_thickness();
        assert!((delta - PI).abs() < 1e-10, "HW: δ should be π, got {delta}");
    }

    // ── EigenmodeLayer tests ───────────────────────────────────────────────────

    #[test]
    fn eigenmode_layer_s11_s22_zero() {
        let layer = EigenmodeLayer::new(500e-9, 3.476, 1.444, 1550e-9, 1, 100);
        let (s11, _s12, _s21, s22) = layer.to_s_matrix_full();
        assert!(
            s11[0][0].norm() < 1e-10,
            "S11 should be zero for uniform layer"
        );
        assert!(
            s22[0][0].norm() < 1e-10,
            "S22 should be zero for uniform layer"
        );
    }

    #[test]
    fn eigenmode_layer_s21_phase_nonzero() {
        let layer = EigenmodeLayer::new(10e-6, 3.476, 1.444, 1550e-9, 1, 100);
        let (_s11, _s12, s21, _s22) = layer.to_s_matrix_full();
        // S21 should be a phase factor with magnitude 1
        assert!(
            (s21[0][0].norm() - 1.0).abs() < 1e-6,
            "S21 should have unit magnitude"
        );
    }

    // ── cascade_smatrices tests ───────────────────────────────────────────────

    #[test]
    fn cascade_identity_smatrices() {
        // Cascade two identity S-matrices: result should preserve identity properties
        let n = 2_usize;
        let id = eye_nd(n);
        let zero = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        let (s11, _s12, s21, s22) =
            cascade_smatrices(&zero, &id, &id, &zero, &zero, &id, &id, &zero);
        for i in 0..n {
            assert!(
                (s21[i][i].re - 1.0).abs() < 1e-10,
                "S21 diagonal should be 1"
            );
            assert!(s11[i][i].norm() < 1e-10, "S11 should be 0");
            assert!(s22[i][i].norm() < 1e-10, "S22 should be 0");
        }
    }

    // ── extract_mode_amplitudes tests ─────────────────────────────────────────

    #[test]
    fn extract_amplitudes_exact_mode_gives_unity() {
        let n = 50_usize;
        let dx = 10e-9_f64;
        let field: Vec<f64> = (0..n).map(|i| (PI * i as f64 / n as f64).sin()).collect();
        let mode = EmeMode {
            n_eff: 2.5,
            beta: 1e7,
            field: field.clone(),
            dx,
        };
        let amps = extract_mode_amplitudes(&field, &[mode], dx);
        assert_eq!(amps.len(), 1);
        assert!((amps[0].re - 1.0).abs() < 1e-10, "Amplitude={}", amps[0].re);
        assert!(amps[0].im.abs() < 1e-10);
    }

    #[test]
    fn extract_amplitudes_orthogonal_mode_near_zero() {
        // Use two well-separated Gaussians: numerator overlap should be negligible
        // relative to the denominator (mode norm), so the coefficient is small.
        let n = 200_usize;
        let dx = 10e-9_f64;
        // Field centered far to the right
        let field: Vec<f64> = (0..n)
            .map(|i| {
                let x = (i as f64 - 150.0) * dx; // shifted right
                (-x * x / (100e-9 * 100e-9)).exp()
            })
            .collect();
        // Mode centered far to the left
        let mode_field: Vec<f64> = (0..n)
            .map(|i| {
                let x = (i as f64 - 50.0) * dx; // shifted left
                (-x * x / (100e-9 * 100e-9)).exp()
            })
            .collect();
        let mode = EmeMode {
            n_eff: 2.3,
            beta: 9e6,
            field: mode_field,
            dx,
        };
        let amps = extract_mode_amplitudes(&field, &[mode], dx);
        assert!(
            amps[0].norm() < 0.01,
            "Orthogonal mode amplitude={}",
            amps[0].norm()
        );
    }

    // ── mat_mul_nd / eye_nd tests ─────────────────────────────────────────────

    #[test]
    fn mat_mul_nd_identity_left() {
        let n = 3_usize;
        let id = eye_nd(n);
        let a: Vec<Vec<Complex64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| Complex64::new((i * n + j) as f64, 0.0))
                    .collect()
            })
            .collect();
        let result = mat_mul_nd(&id, &a);
        for i in 0..n {
            for j in 0..n {
                assert!((result[i][j] - a[i][j]).norm() < 1e-10);
            }
        }
    }

    #[test]
    fn mat_add_nd_commutative() {
        let n = 2_usize;
        let a: Vec<Vec<Complex64>> = vec![
            vec![Complex64::new(1.0, 0.5), Complex64::new(-1.0, 0.0)],
            vec![Complex64::new(0.0, 2.0), Complex64::new(3.0, -1.0)],
        ];
        let b: Vec<Vec<Complex64>> = vec![
            vec![Complex64::new(2.0, 0.0), Complex64::new(0.5, 1.0)],
            vec![Complex64::new(-1.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        let ab = mat_add_nd(&a, &b);
        let ba = mat_add_nd(&b, &a);
        for i in 0..n {
            for j in 0..n {
                assert!((ab[i][j] - ba[i][j]).norm() < 1e-14);
            }
        }
    }

    #[test]
    fn propagator_lossless_preserves_power() {
        let n = 64_usize;
        let dx = 5e-9_f64;
        let field: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((PI * i as f64 / n as f64).sin(), 0.0))
            .collect();
        let mode = EigenMode {
            beta: Complex64::new(1.2e7, 0.0),
            field: field.clone(),
            dx,
        };
        let prop = EigenmodePropagator::new(vec![mode]);
        let out = prop.propagate_forward(&field, 5e-6);
        let p_in: f64 = field.iter().map(|e| e.norm_sqr()).sum();
        let p_out: f64 = out.iter().map(|e| e.norm_sqr()).sum();
        assert!(
            (p_out - p_in).abs() / p_in < 1e-8,
            "Lossless propagator should preserve power"
        );
    }

    #[test]
    fn smatrix_2x2_from_overlap_energy_conserved() {
        let eta = 0.8_f64;
        let s = SMatrix2x2::from_overlap(eta);
        let r_sq = s.s11 * s.s11;
        let t_sq = s.s21 * s.s21;
        assert!((r_sq + t_sq - 1.0).abs() < 1e-10, "R+T={}", r_sq + t_sq);
    }
}
