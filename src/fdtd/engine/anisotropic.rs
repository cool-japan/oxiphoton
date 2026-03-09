//! Anisotropic and magnetic 3D FDTD engine.
//!
//! Supports diagonal ε/μ tensors, birefringent (uniaxial) crystals,
//! gyroelectric (Faraday-rotation) media, and double-negative (Veselago)
//! metamaterials.
//!
//! # Physical model
//!
//! For a diagonal anisotropic medium the constitutive relations are:
//!   D_x = ε₀ · ε_xx · E_x,  D_y = ε₀ · ε_yy · E_y,  D_z = ε₀ · ε_zz · E_z
//!   B_x = μ₀ · μ_xx · H_x,  B_y = μ₀ · μ_yy · H_y,  B_z = μ₀ · μ_zz · H_z
//!
//! The FDTD update equations become component-wise:
//!   E_x^{n+1} = C_a · E_x^n + C_b · (∂H_z/∂y − ∂H_y/∂z)
//! where C_a and C_b absorb ε₀, ε_xx, dt and σ_e.

use num_complex::Complex64;

use crate::error::OxiPhotonError;

// ─── Physical constants ───────────────────────────────────────────────────────

const EPS0: f64 = 8.854_187_812_8e-12;
const MU0: f64 = 1.256_637_062_12e-6;
const C0: f64 = 299_792_458.0;

// ─── AnisotropicFdtd3d ────────────────────────────────────────────────────────

/// Full anisotropic 3D FDTD solver with diagonal ε and μ tensors.
///
/// All six field arrays (E and H) are stored as flat `nx × ny × nz` vectors
/// with the index mapping `idx = k*(nx*ny) + j*nx + i`.
///
/// # Update equations (lossy, diagonal anisotropy)
///
/// ```text
/// E_x^{n+1}\[i,j,k\] = C_a_ex\[i,j,k\] · E_x^n  +  C_b_ex\[i,j,k\] ·
///                     ((H_z\[i,j,k\] − H_z\[i,j-1,k\])/dy
///                      − (H_y\[i,j,k\] − H_y\[i,j,k-1\])/dz)
/// ```
/// with
/// ```text
/// C_a = (1 − σ_e·dt/(2·ε₀·ε_xx)) / (1 + σ_e·dt/(2·ε₀·ε_xx))
/// C_b = (dt/(ε₀·ε_xx))            / (1 + σ_e·dt/(2·ε₀·ε_xx))
/// ```
/// and analogously for the other E and H components.
pub struct AnisotropicFdtd3d {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    pub dt: f64,

    // Electric field components
    pub ex: Vec<f64>,
    pub ey: Vec<f64>,
    pub ez: Vec<f64>,

    // Magnetic field components
    pub hx: Vec<f64>,
    pub hy: Vec<f64>,
    pub hz: Vec<f64>,

    // Anisotropic relative permittivity (diagonal tensor components)
    pub eps_xx: Vec<f64>,
    pub eps_yy: Vec<f64>,
    pub eps_zz: Vec<f64>,

    // Anisotropic relative permeability (diagonal tensor components)
    pub mu_xx: Vec<f64>,
    pub mu_yy: Vec<f64>,
    pub mu_zz: Vec<f64>,

    // Electric conductivity σ_e (S/m) — isotropic loss term
    pub sigma_e: Vec<f64>,

    // Magnetic conductivity σ_m (Ω/m·s) — isotropic magnetic loss
    pub sigma_m: Vec<f64>,

    pub time_step: usize,

    // Pre-computed update coefficients (computed lazily on first step)
    ca_ex: Vec<f64>,
    cb_ex: Vec<f64>,
    ca_ey: Vec<f64>,
    cb_ey: Vec<f64>,
    ca_ez: Vec<f64>,
    cb_ez: Vec<f64>,
    da_hx: Vec<f64>,
    db_hx: Vec<f64>,
    da_hy: Vec<f64>,
    db_hy: Vec<f64>,
    da_hz: Vec<f64>,
    db_hz: Vec<f64>,

    coeffs_ready: bool,
}

impl AnisotropicFdtd3d {
    /// Construct a new anisotropic FDTD grid.
    ///
    /// All fields are initialised to zero; all ε_ii = 1 (vacuum); all μ_ii = 1.
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64, dt: f64) -> Self {
        let n = nx * ny * nz;
        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            dt,
            ex: vec![0.0; n],
            ey: vec![0.0; n],
            ez: vec![0.0; n],
            hx: vec![0.0; n],
            hy: vec![0.0; n],
            hz: vec![0.0; n],
            eps_xx: vec![1.0; n],
            eps_yy: vec![1.0; n],
            eps_zz: vec![1.0; n],
            mu_xx: vec![1.0; n],
            mu_yy: vec![1.0; n],
            mu_zz: vec![1.0; n],
            sigma_e: vec![0.0; n],
            sigma_m: vec![0.0; n],
            time_step: 0,
            ca_ex: vec![0.0; n],
            cb_ex: vec![0.0; n],
            ca_ey: vec![0.0; n],
            cb_ey: vec![0.0; n],
            ca_ez: vec![0.0; n],
            cb_ez: vec![0.0; n],
            da_hx: vec![0.0; n],
            db_hx: vec![0.0; n],
            da_hy: vec![0.0; n],
            db_hy: vec![0.0; n],
            da_hz: vec![0.0; n],
            db_hz: vec![0.0; n],
            coeffs_ready: false,
        }
    }

    // ── Index helper ──────────────────────────────────────────────────────────

    /// Linearise grid indices: `(i, j, k)` → flat index.
    #[inline(always)]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        k * (self.nx * self.ny) + j * self.nx + i
    }

    // ── Material setters ──────────────────────────────────────────────────────

    /// Fill a rectangular box with diagonal permittivity tensor components.
    #[allow(clippy::too_many_arguments)]
    pub fn fill_eps_box(
        &mut self,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        k0: usize,
        k1: usize,
        exx: f64,
        eyy: f64,
        ezz: f64,
    ) {
        for k in k0..k1.min(self.nz) {
            for j in j0..j1.min(self.ny) {
                for i in i0..i1.min(self.nx) {
                    let idx = self.idx(i, j, k);
                    self.eps_xx[idx] = exx;
                    self.eps_yy[idx] = eyy;
                    self.eps_zz[idx] = ezz;
                }
            }
        }
        self.coeffs_ready = false;
    }

    /// Fill a rectangular box with diagonal permeability tensor components.
    #[allow(clippy::too_many_arguments)]
    pub fn fill_mu_box(
        &mut self,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        k0: usize,
        k1: usize,
        mxx: f64,
        myy: f64,
        mzz: f64,
    ) {
        for k in k0..k1.min(self.nz) {
            for j in j0..j1.min(self.ny) {
                for i in i0..i1.min(self.nx) {
                    let idx = self.idx(i, j, k);
                    self.mu_xx[idx] = mxx;
                    self.mu_yy[idx] = myy;
                    self.mu_zz[idx] = mzz;
                }
            }
        }
        self.coeffs_ready = false;
    }

    // ── Coefficient computation ───────────────────────────────────────────────

    /// Pre-compute all FDTD update coefficients from the current material maps.
    ///
    /// This is called automatically before the first time step, and must be
    /// re-called explicitly if you modify materials after stepping has begun.
    pub fn compute_coefficients(&mut self) {
        let dt = self.dt;
        let n = self.nx * self.ny * self.nz;

        for p in 0..n {
            // ── Electric field coefficients ─────────────────────────────────
            // For E_x: denom = 1 + σ_e·dt/(2·ε₀·ε_xx)
            let se = self.sigma_e[p];

            let denom_ex = 1.0 + se * dt / (2.0 * EPS0 * self.eps_xx[p]);
            self.ca_ex[p] = (1.0 - se * dt / (2.0 * EPS0 * self.eps_xx[p])) / denom_ex;
            self.cb_ex[p] = (dt / (EPS0 * self.eps_xx[p])) / denom_ex;

            let denom_ey = 1.0 + se * dt / (2.0 * EPS0 * self.eps_yy[p]);
            self.ca_ey[p] = (1.0 - se * dt / (2.0 * EPS0 * self.eps_yy[p])) / denom_ey;
            self.cb_ey[p] = (dt / (EPS0 * self.eps_yy[p])) / denom_ey;

            let denom_ez = 1.0 + se * dt / (2.0 * EPS0 * self.eps_zz[p]);
            self.ca_ez[p] = (1.0 - se * dt / (2.0 * EPS0 * self.eps_zz[p])) / denom_ez;
            self.cb_ez[p] = (dt / (EPS0 * self.eps_zz[p])) / denom_ez;

            // ── Magnetic field coefficients ─────────────────────────────────
            let sm = self.sigma_m[p];

            let denom_hx = 1.0 + sm * dt / (2.0 * MU0 * self.mu_xx[p]);
            self.da_hx[p] = (1.0 - sm * dt / (2.0 * MU0 * self.mu_xx[p])) / denom_hx;
            self.db_hx[p] = (dt / (MU0 * self.mu_xx[p])) / denom_hx;

            let denom_hy = 1.0 + sm * dt / (2.0 * MU0 * self.mu_yy[p]);
            self.da_hy[p] = (1.0 - sm * dt / (2.0 * MU0 * self.mu_yy[p])) / denom_hy;
            self.db_hy[p] = (dt / (MU0 * self.mu_yy[p])) / denom_hy;

            let denom_hz = 1.0 + sm * dt / (2.0 * MU0 * self.mu_zz[p]);
            self.da_hz[p] = (1.0 - sm * dt / (2.0 * MU0 * self.mu_zz[p])) / denom_hz;
            self.db_hz[p] = (dt / (MU0 * self.mu_zz[p])) / denom_hz;
        }

        self.coeffs_ready = true;
    }

    // ── Time-stepping ─────────────────────────────────────────────────────────

    /// Update magnetic field components (H) using curl of E.
    ///
    /// Yee-grid half-step for H (PEC boundaries — boundary H cells are left
    /// unmodified because the forward difference stencils are zero there).
    pub fn step_h(&mut self) {
        if !self.coeffs_ready {
            self.compute_coefficients();
        }

        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let dx = self.dx;
        let dy = self.dy;
        let dz = self.dz;

        // Iterate over interior cells only (avoid boundary overflows)
        for k in 0..nz.saturating_sub(1) {
            for j in 0..ny.saturating_sub(1) {
                for i in 0..nx.saturating_sub(1) {
                    let p = self.idx(i, j, k);

                    // curl E components (forward differences on Yee grid)
                    let dez_dy = (self.ez[self.idx(i, j + 1, k)] - self.ez[p]) / dy;
                    let dey_dz = (self.ey[self.idx(i, j, k + 1)] - self.ey[p]) / dz;
                    let dex_dz = (self.ex[self.idx(i, j, k + 1)] - self.ex[p]) / dz;
                    let dez_dx = (self.ez[self.idx(i + 1, j, k)] - self.ez[p]) / dx;
                    let dey_dx = (self.ey[self.idx(i + 1, j, k)] - self.ey[p]) / dx;
                    let dex_dy = (self.ex[self.idx(i, j + 1, k)] - self.ex[p]) / dy;

                    // H_x update:  ∂H_x/∂t = -(1/μ₀μ_xx)(∂E_z/∂y − ∂E_y/∂z)
                    self.hx[p] = self.da_hx[p] * self.hx[p] - self.db_hx[p] * (dez_dy - dey_dz);

                    // H_y update:  ∂H_y/∂t = -(1/μ₀μ_yy)(∂E_x/∂z − ∂E_z/∂x)
                    self.hy[p] = self.da_hy[p] * self.hy[p] - self.db_hy[p] * (dex_dz - dez_dx);

                    // H_z update:  ∂H_z/∂t = -(1/μ₀μ_zz)(∂E_y/∂x − ∂E_x/∂y)
                    self.hz[p] = self.da_hz[p] * self.hz[p] - self.db_hz[p] * (dey_dx - dex_dy);
                }
            }
        }
    }

    /// Update electric field components (E) using curl of H.
    ///
    /// Uses backward-difference stencils consistent with the Yee-grid
    /// leapfrog scheme.
    pub fn step_e(&mut self) {
        if !self.coeffs_ready {
            self.compute_coefficients();
        }

        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let dx = self.dx;
        let dy = self.dy;
        let dz = self.dz;

        // Iterate over interior cells (skip index 0 to avoid underflow)
        for k in 1..nz {
            for j in 1..ny {
                for i in 1..nx {
                    let p = self.idx(i, j, k);

                    // curl H backward differences
                    let dhz_dy = (self.hz[p] - self.hz[self.idx(i, j - 1, k)]) / dy;
                    let dhy_dz = (self.hy[p] - self.hy[self.idx(i, j, k - 1)]) / dz;
                    let dhx_dz = (self.hx[p] - self.hx[self.idx(i, j, k - 1)]) / dz;
                    let dhz_dx = (self.hz[p] - self.hz[self.idx(i - 1, j, k)]) / dx;
                    let dhy_dx = (self.hy[p] - self.hy[self.idx(i - 1, j, k)]) / dx;
                    let dhx_dy = (self.hx[p] - self.hx[self.idx(i, j - 1, k)]) / dy;

                    // E_x += (1/ε₀ε_xx)(∂H_z/∂y − ∂H_y/∂z)
                    self.ex[p] = self.ca_ex[p] * self.ex[p] + self.cb_ex[p] * (dhz_dy - dhy_dz);

                    // E_y += (1/ε₀ε_yy)(∂H_x/∂z − ∂H_z/∂x)
                    self.ey[p] = self.ca_ey[p] * self.ey[p] + self.cb_ey[p] * (dhx_dz - dhz_dx);

                    // E_z += (1/ε₀ε_zz)(∂H_y/∂x − ∂H_x/∂y)
                    self.ez[p] = self.ca_ez[p] * self.ez[p] + self.cb_ez[p] * (dhy_dx - dhx_dy);
                }
            }
        }
    }

    /// Perform one complete FDTD leapfrog time step (H half-step → E full-step).
    pub fn step(&mut self) {
        self.step_h();
        self.step_e();
        self.time_step += 1;
    }

    /// Inject a soft Ez point source at cell `(i, j, k)`.
    ///
    /// "Soft" means the source is additive, so waves can pass through the
    /// source point without reflection.
    pub fn set_point_source_ez(&mut self, i: usize, j: usize, k: usize, amplitude: f64) {
        if i < self.nx && j < self.ny && k < self.nz {
            let idx = self.idx(i, j, k);
            self.ez[idx] += amplitude;
        }
    }

    /// Return the current simulation time (s).
    pub fn current_time(&self) -> f64 {
        self.time_step as f64 * self.dt
    }
}

// ─── UniaxialCrystal ──────────────────────────────────────────────────────────

/// Birefringent uniaxial crystal described by ordinary (`no`) and
/// extraordinary (`ne`) refractive indices and an optic-axis direction.
///
/// The permittivity tensor is:
/// ```text
/// ε = no² · I  +  (ne² − no²) · (ĉ ⊗ ĉ)
/// ```
/// where `ĉ` is the unit vector along the optic axis.
#[derive(Debug, Clone)]
pub struct UniaxialCrystal {
    /// Ordinary refractive index (perpendicular to optic axis).
    pub no: f64,
    /// Extraordinary refractive index (along optic axis).
    pub ne: f64,
    /// Unit vector along the optic axis.
    pub optic_axis: [f64; 3],
}

impl UniaxialCrystal {
    /// Create a new uniaxial crystal.
    ///
    /// `optic_axis` need not be unit-length — it is normalised internally.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] if `optic_axis` has zero length
    /// or if `no` / `ne` are non-positive.
    pub fn new(no: f64, ne: f64, optic_axis: [f64; 3]) -> Result<Self, OxiPhotonError> {
        if no <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "ordinary index no={no} must be positive"
            )));
        }
        if ne <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "extraordinary index ne={ne} must be positive"
            )));
        }
        let mag = (optic_axis[0].powi(2) + optic_axis[1].powi(2) + optic_axis[2].powi(2)).sqrt();
        if mag < 1e-30 {
            return Err(OxiPhotonError::NumericalError(
                "optic_axis must be a non-zero vector".to_string(),
            ));
        }
        let axis = [
            optic_axis[0] / mag,
            optic_axis[1] / mag,
            optic_axis[2] / mag,
        ];
        Ok(Self {
            no,
            ne,
            optic_axis: axis,
        })
    }

    /// Full 3 × 3 permittivity tensor as a row-major array.
    ///
    /// ε_ij = no² · δ_ij  +  (ne² − no²) · c_i · c_j
    pub fn permittivity_tensor(&self) -> [[f64; 3]; 3] {
        let [cx, cy, cz] = self.optic_axis;
        let no2 = self.no * self.no;
        let delta = self.ne * self.ne - no2;
        [
            [no2 + delta * cx * cx, delta * cx * cy, delta * cx * cz],
            [delta * cy * cx, no2 + delta * cy * cy, delta * cy * cz],
            [delta * cz * cx, delta * cz * cy, no2 + delta * cz * cz],
        ]
    }

    /// Diagonal elements of the permittivity tensor.
    ///
    /// Only exact for axis-aligned optic axes where the off-diagonal terms
    /// vanish. For general orientations these are the diagonal entries of the
    /// full tensor (see \[`permittivity_tensor`\]).
    pub fn diagonal_eps(&self) -> [f64; 3] {
        let t = self.permittivity_tensor();
        [t[0][0], t[1][1], t[2][2]]
    }

    /// Phase velocity for the ordinary ray: `v = c / no`.
    pub fn phase_velocity_ordinary(&self) -> f64 {
        C0 / self.no
    }

    /// Phase velocity for the extraordinary ray: `v = c / ne`.
    pub fn phase_velocity_extraordinary(&self) -> f64 {
        C0 / self.ne
    }

    /// Birefringence magnitude `Δn = |ne − no|`.
    pub fn birefringence(&self) -> f64 {
        (self.ne - self.no).abs()
    }
}

// ─── GyroelectricMedium ───────────────────────────────────────────────────────

/// Gyroelectric medium supporting Faraday rotation.
///
/// The permittivity tensor (with gyration vector along z) has the
/// Hermitian anti-symmetric off-diagonal form:
/// ```text
/// ε = \[[ εd,  −iεg,  0 \],
///      \[ iεg,  εd,   0 \],
///      \[  0,    0,   εd\]]
/// ```
/// For a general gyration direction the tensor is rotated accordingly.
#[derive(Debug, Clone)]
pub struct GyroelectricMedium {
    /// Diagonal permittivity element ε_d.
    pub eps_d: f64,
    /// Gyrotropy (off-diagonal) parameter ε_g.
    pub eps_g: f64,
    /// Unit vector along the gyration (magnetisation) direction.
    pub g_vec: [f64; 3],
}

impl GyroelectricMedium {
    /// Create a new gyroelectric medium.
    ///
    /// `g_vec` is normalised internally.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] if `g_vec` has zero length.
    pub fn new(eps_d: f64, eps_g: f64, g_vec: [f64; 3]) -> Result<Self, OxiPhotonError> {
        let mag = (g_vec[0].powi(2) + g_vec[1].powi(2) + g_vec[2].powi(2)).sqrt();
        if mag < 1e-30 {
            return Err(OxiPhotonError::NumericalError(
                "g_vec must be a non-zero vector".to_string(),
            ));
        }
        let gn = [g_vec[0] / mag, g_vec[1] / mag, g_vec[2] / mag];
        Ok(Self {
            eps_d,
            eps_g,
            g_vec: gn,
        })
    }

    /// Full 3 × 3 complex permittivity tensor.
    ///
    /// For `g_vec = ẑ` this reduces to the standard gyrotropic form.
    /// The general form uses the rotation that maps ẑ → `g_vec`.
    ///
    /// The resulting tensor is Hermitian: ε_ij = ε_ji*.
    pub fn permittivity_tensor(&self) -> [[Complex64; 3]; 3] {
        // Build the gyrotropic tensor in the local frame (g along z)
        let eps_local: [[Complex64; 3]; 3] = [
            [
                Complex64::new(self.eps_d, 0.0),
                Complex64::new(0.0, -self.eps_g),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, self.eps_g),
                Complex64::new(self.eps_d, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(self.eps_d, 0.0),
            ],
        ];

        // Rotation matrix R that maps ẑ → g_vec
        let r = rotation_z_to(&self.g_vec);

        // ε_global = R · ε_local · R^†
        rotate_tensor_complex(&eps_local, &r)
    }

    /// Refractive indices for left (+) and right (−) circularly polarised waves.
    ///
    /// ```text
    /// n± = √(εd ± εg)
    /// ```
    ///
    /// Returns `(n_plus, n_minus)`.  If either quantity under the square-root
    /// is negative (strongly absorbing regime) the magnitude is returned with
    /// a note that the result may be complex.
    pub fn circular_indices(&self) -> (f64, f64) {
        let np = (self.eps_d + self.eps_g).abs().sqrt();
        let nm = (self.eps_d - self.eps_g).abs().sqrt();
        (np, nm)
    }

    /// Faraday rotation rate θ_F (rad/m) at angular frequency `omega` (rad/s).
    ///
    /// ```text
    /// θ_F = (ω / (2c)) · (n₊ − n₋)
    /// ```
    pub fn faraday_rotation_rate(&self, omega: f64) -> f64 {
        let (np, nm) = self.circular_indices();
        omega / (2.0 * C0) * (np - nm)
    }

    /// Estimate of the Verdet constant V (rad T⁻¹ m⁻¹) assuming the gyrotropy
    /// is proportional to the applied magnetic flux density `b_field` (T).
    ///
    /// ```text
    /// V = θ_F / b_field
    /// ```
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] if `b_field` is zero.
    pub fn verdet_constant(&self, omega: f64, b_field: f64) -> Result<f64, OxiPhotonError> {
        if b_field.abs() < 1e-30 {
            return Err(OxiPhotonError::NumericalError(
                "b_field must be non-zero to compute Verdet constant".to_string(),
            ));
        }
        Ok(self.faraday_rotation_rate(omega) / b_field)
    }
}

// ─── DoublNegativeMedium ──────────────────────────────────────────────────────

/// Double-negative (Veselago) metamaterial with ε_r < 0 and μ_r < 0.
///
/// In this medium the refractive index is chosen as the negative root:
/// `n = −√(|ε_r| · |μ_r|)`.
#[derive(Debug, Clone)]
pub struct DoublNegativeMedium {
    /// Relative permittivity (must be negative for DNG behaviour).
    pub eps_r: f64,
    /// Relative permeability (must be negative for DNG behaviour).
    pub mu_r: f64,
}

impl DoublNegativeMedium {
    /// Construct a double-negative medium.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] if either `eps_r` or `mu_r`
    /// is non-negative (which would make this an ordinary medium).
    pub fn new(eps_r: f64, mu_r: f64) -> Result<Self, OxiPhotonError> {
        if eps_r >= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "eps_r={eps_r} must be negative for a double-negative medium"
            )));
        }
        if mu_r >= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "mu_r={mu_r} must be negative for a double-negative medium"
            )));
        }
        Ok(Self { eps_r, mu_r })
    }

    /// Refractive index `n = −√(|ε_r| · |μ_r|)` (negative — backward wave).
    pub fn refractive_index(&self) -> f64 {
        -(self.eps_r.abs() * self.mu_r.abs()).sqrt()
    }

    /// Phase velocity `v_p = c / n` (negative for DNG — backward wave).
    pub fn phase_velocity(&self) -> f64 {
        C0 / self.refractive_index()
    }

    /// Estimate of the group velocity using a finite-difference approximation
    /// of `dω/dk` over a small frequency interval `domega`.
    ///
    /// For a non-dispersive DNG medium the group and phase velocities have
    /// equal magnitudes but opposite signs (group velocity is positive).
    ///
    /// For a realistic dispersive model you should supply the full dispersion
    /// relation; this method provides a first-order estimate using the formula
    ///
    /// ```text
    /// v_g ≈ domega / (dk)
    /// ```
    ///
    /// where `dk = (ω + dω) · n(ω + dω) / c − ω · n(ω) / c`.
    ///
    /// For a frequency-independent DNG this simplifies to `|v_p|`.
    pub fn group_velocity(&self, omega: f64, domega: f64) -> f64 {
        // k(ω) = ω·n/c  (n < 0, so k < 0)
        let n = self.refractive_index();
        let k1 = omega * n / C0;
        // For a non-dispersive medium n does not change with frequency:
        let k2 = (omega + domega) * n / C0;
        let dk = k2 - k1;
        if dk.abs() < 1e-30 {
            // Non-dispersive limit: |v_p|
            (C0 / n).abs()
        } else {
            (domega / dk).abs()
        }
    }
}

// ─── Helper: fill uniaxial crystal into AnisotropicFdtd3d ────────────────────

/// Fill a rectangular box of an [`AnisotropicFdtd3d`] grid with the diagonal
/// permittivity from a [`UniaxialCrystal`].
///
/// Only the diagonal components of the permittivity tensor are used, which is
/// exact when the optic axis is aligned with a coordinate axis and a reasonable
/// approximation otherwise (the off-diagonal terms couple field components that
/// the diagonal FDTD engine cannot represent).
#[allow(clippy::too_many_arguments)]
pub fn fill_uniaxial_crystal(
    fdtd: &mut AnisotropicFdtd3d,
    crystal: &UniaxialCrystal,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    k0: usize,
    k1: usize,
) {
    let [exx, eyy, ezz] = crystal.diagonal_eps();
    fdtd.fill_eps_box(i0, i1, j0, j1, k0, k1, exx, eyy, ezz);
}

// ─── Internal rotation helpers ────────────────────────────────────────────────

/// Build the 3 × 3 rotation matrix that rotates ẑ = (0,0,1) to `target`.
///
/// Uses Rodrigues' rotation formula.  Returns the identity if `target` is
/// already aligned with ẑ.
fn rotation_z_to(target: &[f64; 3]) -> [[f64; 3]; 3] {
    let z = [0.0_f64, 0.0, 1.0];
    let t = *target;

    // Cross product k = z × t
    let kx = z[1] * t[2] - z[2] * t[1];
    let ky = z[2] * t[0] - z[0] * t[2];
    let kz = z[0] * t[1] - z[1] * t[0];
    let sin_theta = (kx * kx + ky * ky + kz * kz).sqrt();
    let cos_theta = z[0] * t[0] + z[1] * t[1] + z[2] * t[2];

    if sin_theta < 1e-12 {
        // Already aligned (or anti-parallel): return identity or 180° rotation
        if cos_theta > 0.0 {
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        } else {
            // 180° rotation about x-axis
            return [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
        }
    }

    // Rodrigues: R = I + K + K²·(1-cos θ)/sin²θ
    // where K is the cross-product matrix of k̂ = (kx,ky,kz)/sin_theta
    let kxn = kx / sin_theta;
    let kyn = ky / sin_theta;
    let kzn = kz / sin_theta;
    let c = cos_theta;
    let s = sin_theta;
    let t1 = 1.0 - c;

    [
        [
            c + kxn * kxn * t1,
            kxn * kyn * t1 - kzn * s,
            kxn * kzn * t1 + kyn * s,
        ],
        [
            kyn * kxn * t1 + kzn * s,
            c + kyn * kyn * t1,
            kyn * kzn * t1 - kxn * s,
        ],
        [
            kzn * kxn * t1 - kyn * s,
            kzn * kyn * t1 + kxn * s,
            c + kzn * kzn * t1,
        ],
    ]
}

/// Rotate a 3×3 complex tensor: T_global = R · T_local · R^T (real R).
fn rotate_tensor_complex(t: &[[Complex64; 3]; 3], r: &[[f64; 3]; 3]) -> [[Complex64; 3]; 3] {
    // First compute tmp = R · T
    let mut tmp = [[Complex64::new(0.0, 0.0); 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            let mut sum = Complex64::new(0.0, 0.0);
            for m in 0..3 {
                sum += Complex64::new(r[row][m], 0.0) * t[m][col];
            }
            tmp[row][col] = sum;
        }
    }

    // Then compute result = tmp · R^T
    let mut result = [[Complex64::new(0.0, 0.0); 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            let mut sum = Complex64::new(0.0, 0.0);
            for m in 0..3 {
                // R^T[m][col] = R[col][m]
                sum += tmp[row][m] * Complex64::new(r[col][m], 0.0);
            }
            result[row][col] = sum;
        }
    }
    result
}

// ─── Extension methods for Fdtd3d ────────────────────────────────────────────
// (These live in fdtd_3d.rs; declared here as free functions for use from
//  the dims module.)

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const APPROX_TOL: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ── 1. Creation test ──────────────────────────────────────────────────────

    #[test]
    fn test_anisotropic_fdtd_creation() {
        let fdtd = AnisotropicFdtd3d::new(10, 10, 10, 1e-9, 1e-9, 1e-9, 1e-18);

        assert_eq!(fdtd.nx, 10);
        assert_eq!(fdtd.ny, 10);
        assert_eq!(fdtd.nz, 10);

        let n = 10 * 10 * 10;
        assert_eq!(fdtd.ex.len(), n);
        assert_eq!(fdtd.hz.len(), n);

        // All fields must be zero on construction
        for &v in &fdtd.ex {
            assert_eq!(v, 0.0);
        }
        for &v in &fdtd.ey {
            assert_eq!(v, 0.0);
        }
        for &v in &fdtd.ez {
            assert_eq!(v, 0.0);
        }
        for &v in &fdtd.hx {
            assert_eq!(v, 0.0);
        }
        for &v in &fdtd.hy {
            assert_eq!(v, 0.0);
        }
        for &v in &fdtd.hz {
            assert_eq!(v, 0.0);
        }

        // Default permittivity / permeability must be 1 (vacuum)
        for &v in &fdtd.eps_xx {
            assert_eq!(v, 1.0);
        }
        for &v in &fdtd.eps_yy {
            assert_eq!(v, 1.0);
        }
        for &v in &fdtd.eps_zz {
            assert_eq!(v, 1.0);
        }
        for &v in &fdtd.mu_xx {
            assert_eq!(v, 1.0);
        }
        for &v in &fdtd.mu_yy {
            assert_eq!(v, 1.0);
        }
        for &v in &fdtd.mu_zz {
            assert_eq!(v, 1.0);
        }
    }

    // ── 2. Uniaxial crystal tensor ────────────────────────────────────────────

    #[test]
    fn test_uniaxial_crystal_tensor() {
        // Optic axis along z → ε = diag(no², no², ne²)
        let crystal = UniaxialCrystal::new(1.5, 1.7, [0.0, 0.0, 1.0]).expect("valid crystal");

        let t = crystal.permittivity_tensor();

        let no2 = 1.5_f64 * 1.5;
        let ne2 = 1.7_f64 * 1.7;

        // Diagonal elements
        assert!(
            approx_eq(t[0][0], no2, APPROX_TOL),
            "ε_xx = no² failed: {}",
            t[0][0]
        );
        assert!(
            approx_eq(t[1][1], no2, APPROX_TOL),
            "ε_yy = no² failed: {}",
            t[1][1]
        );
        assert!(
            approx_eq(t[2][2], ne2, APPROX_TOL),
            "ε_zz = ne² failed: {}",
            t[2][2]
        );

        // Off-diagonal elements must be zero for z-axis optic axis
        assert!(approx_eq(t[0][1], 0.0, APPROX_TOL));
        assert!(approx_eq(t[0][2], 0.0, APPROX_TOL));
        assert!(approx_eq(t[1][2], 0.0, APPROX_TOL));

        // diagonal_eps() convenience function
        let d = crystal.diagonal_eps();
        assert!(approx_eq(d[0], no2, APPROX_TOL));
        assert!(approx_eq(d[1], no2, APPROX_TOL));
        assert!(approx_eq(d[2], ne2, APPROX_TOL));
    }

    // ── 3. Birefringence ──────────────────────────────────────────────────────

    #[test]
    fn test_birefringence() {
        let crystal = UniaxialCrystal::new(1.5, 1.7, [0.0, 0.0, 1.0]).expect("valid crystal");
        assert!(approx_eq(crystal.birefringence(), 0.2, APPROX_TOL));

        // Negative birefringence (ne < no)
        let crystal2 = UniaxialCrystal::new(1.7, 1.5, [0.0, 0.0, 1.0]).expect("valid crystal");
        assert!(approx_eq(crystal2.birefringence(), 0.2, APPROX_TOL));
    }

    // ── 4. Gyroelectric tensor — Hermitian property ───────────────────────────

    #[test]
    fn test_gyroelectric_tensor() {
        let medium = GyroelectricMedium::new(2.5, 0.3, [0.0, 0.0, 1.0]).expect("valid medium");
        let t = medium.permittivity_tensor();

        // Hermitian: t[i][j] == conj(t[j][i])
        for (i, row) in t.iter().enumerate() {
            for (j, _) in row.iter().enumerate() {
                let diff = (t[i][j] - t[j][i].conj()).norm();
                assert!(
                    diff < APPROX_TOL,
                    "Hermitian property violated at ({i},{j}): diff={diff}"
                );
            }
        }

        // Diagonal elements must be real and equal to eps_d
        for (k, _) in t.iter().enumerate() {
            assert!(approx_eq(t[k][k].re, 2.5, APPROX_TOL));
            assert!(approx_eq(t[k][k].im, 0.0, APPROX_TOL));
        }

        // Off-diagonal for z-alignment: t[0][1] = -i·eps_g, t[1][0] = i·eps_g
        assert!(approx_eq(t[0][1].re, 0.0, APPROX_TOL));
        assert!(approx_eq(t[0][1].im, -0.3, APPROX_TOL));
        assert!(approx_eq(t[1][0].re, 0.0, APPROX_TOL));
        assert!(approx_eq(t[1][0].im, 0.3, APPROX_TOL));
    }

    // ── 5. Faraday rotation rate sign ────────────────────────────────────────

    #[test]
    fn test_faraday_rotation_rate() {
        // Positive eps_g → n_+ > n_- → positive Faraday rotation
        let medium = GyroelectricMedium::new(2.5, 0.5, [0.0, 0.0, 1.0]).expect("valid medium");
        let omega = 2.0 * std::f64::consts::PI * 3e14; // ~infrared
        let rate = medium.faraday_rotation_rate(omega);
        assert!(
            rate > 0.0,
            "Faraday rotation rate must be positive for eps_g > 0: {rate}"
        );
    }

    // ── 6. Double-negative medium ─────────────────────────────────────────────

    #[test]
    fn test_double_negative_medium() {
        let dng = DoublNegativeMedium::new(-2.0, -1.5).expect("valid DNG");

        let n = dng.refractive_index();
        assert!(n < 0.0, "Refractive index must be negative for DNG: {n}");

        // |n| = sqrt(|eps| * |mu|) = sqrt(2.0 * 1.5) = sqrt(3)
        let expected_abs = (2.0_f64 * 1.5_f64).sqrt();
        assert!(
            approx_eq(n.abs(), expected_abs, APPROX_TOL),
            "|n| = {}, expected {}",
            n.abs(),
            expected_abs
        );

        // Phase velocity must be negative
        let vp = dng.phase_velocity();
        assert!(vp < 0.0, "Phase velocity must be negative for DNG: {vp}");

        // Group velocity must be positive (energy propagates forward)
        let omega = 2.0 * std::f64::consts::PI * 3e14;
        let vg = dng.group_velocity(omega, omega * 1e-6);
        assert!(vg > 0.0, "Group velocity must be positive for DNG: {vg}");

        // Positive eps_r rejected
        assert!(DoublNegativeMedium::new(2.0, -1.5).is_err());
        // Positive mu_r rejected
        assert!(DoublNegativeMedium::new(-2.0, 1.5).is_err());
    }

    // ── 7. fill_uniaxial_box ──────────────────────────────────────────────────

    #[test]
    fn test_fill_uniaxial_box() {
        let mut fdtd = AnisotropicFdtd3d::new(20, 20, 20, 10e-9, 10e-9, 10e-9, 1e-17);

        // Before: all eps = 1.0 (vacuum)
        let p_before = fdtd.idx(10, 10, 10);
        assert_eq!(fdtd.eps_xx[p_before], 1.0);

        let crystal = UniaxialCrystal::new(1.5, 1.7, [0.0, 0.0, 1.0]).expect("valid crystal");
        fill_uniaxial_crystal(&mut fdtd, &crystal, 5, 15, 5, 15, 5, 15);

        // After: centre cell should have ordinary permittivity (no²) for xx
        let p_after = fdtd.idx(10, 10, 10);
        let no2 = 1.5_f64 * 1.5_f64;
        let ne2 = 1.7_f64 * 1.7_f64;
        assert!(
            approx_eq(fdtd.eps_xx[p_after], no2, APPROX_TOL),
            "eps_xx should be no²={no2}, got {}",
            fdtd.eps_xx[p_after]
        );
        assert!(
            approx_eq(fdtd.eps_yy[p_after], no2, APPROX_TOL),
            "eps_yy should be no²={no2}, got {}",
            fdtd.eps_yy[p_after]
        );
        assert!(
            approx_eq(fdtd.eps_zz[p_after], ne2, APPROX_TOL),
            "eps_zz should be ne²={ne2}, got {}",
            fdtd.eps_zz[p_after]
        );

        // Corner outside box: should still be vacuum
        let p_out = fdtd.idx(0, 0, 0);
        assert_eq!(fdtd.eps_xx[p_out], 1.0);
    }

    // ── 8. Single time step without crash ─────────────────────────────────────

    #[test]
    fn test_anisotropic_step() {
        let dx = 20e-9_f64; // 20 nm cells
                            // Courant stable dt for vacuum 3D: dt = dx / (sqrt(3) * c)
        let dt = dx / (3.0_f64.sqrt() * C0) * 0.95;
        let mut fdtd = AnisotropicFdtd3d::new(16, 16, 16, dx, dx, dx, dt);

        // Fill a birefringent slab in the centre
        let crystal = UniaxialCrystal::new(1.5, 1.7, [0.0, 0.0, 1.0]).expect("valid crystal");
        fill_uniaxial_crystal(&mut fdtd, &crystal, 4, 12, 4, 12, 4, 12);

        // Inject a soft Ez source in the centre
        fdtd.set_point_source_ez(8, 8, 8, 1.0);

        // Run ten steps — must not panic or produce NaN/Inf
        for _ in 0..10 {
            fdtd.step();
        }

        assert_eq!(fdtd.time_step, 10);

        // Check no NaN or Inf in any field component
        for &v in fdtd.ez.iter().chain(&fdtd.ey).chain(&fdtd.ex) {
            assert!(v.is_finite(), "Ez field contains non-finite value: {v}");
        }
        for &v in fdtd.hz.iter().chain(&fdtd.hy).chain(&fdtd.hx) {
            assert!(v.is_finite(), "Hz field contains non-finite value: {v}");
        }

        // After injecting Ez and stepping, at least one field value should be
        // non-zero (energy must propagate)
        let max_e: f64 = fdtd.ez.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            max_e > 0.0,
            "Ez field should contain non-zero values after stepping"
        );
    }
}
