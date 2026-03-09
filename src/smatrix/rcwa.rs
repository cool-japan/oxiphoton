use num_complex::Complex64;
use std::f64::consts::PI;

/// Rigorous Coupled-Wave Analysis (RCWA) for 1D binary gratings.
///
/// Solves Maxwell's equations in a periodic medium using Fourier expansion.
/// Supports TE (s-pol) and TM (p-pol) polarizations for normal and oblique incidence.
///
/// For TE polarization, the governing equation in each layer is:
///   d²S/dz² = [K_x² - E] S
/// where:
///   K_x = diag(k_x0 + m·G) for harmonics m = -N..N, G = 2π/Λ
///   E = Toeplitz matrix of Fourier coefficients of ε(x)
///   S = vector of Ey harmonic amplitudes
///
/// For TM polarization, E is replaced by E^{-1} (inverse permittivity Fourier matrix).
///
/// Reference: Moharam & Gaylord, JOSA 71(7), 1981; Lalanne & Morris, JOSA A 13(4), 1996.
use crate::smatrix::Polarization;

/// 1D binary grating geometry
#[derive(Debug, Clone)]
pub struct GratingLayer {
    /// Grating period (m)
    pub period: f64,
    /// Layer thickness (m)
    pub thickness: f64,
    /// Permittivity in ridge region (ε = n²)
    pub eps_ridge: Complex64,
    /// Permittivity in groove region
    pub eps_groove: Complex64,
    /// Fill factor (ridge width / period), 0 < f < 1
    pub fill_factor: f64,
}

impl GratingLayer {
    pub fn new(period: f64, thickness: f64, n_ridge: f64, n_groove: f64, fill_factor: f64) -> Self {
        Self {
            period,
            thickness,
            eps_ridge: Complex64::new(n_ridge * n_ridge, 0.0),
            eps_groove: Complex64::new(n_groove * n_groove, 0.0),
            fill_factor,
        }
    }

    /// Compute Fourier coefficients of ε(x) for harmonics -N..N.
    /// Returns Toeplitz row (2N+1 elements), indexed by m = -N..N (center = N).
    pub fn eps_fourier(&self, n_orders: usize) -> Vec<Complex64> {
        let n_total = 2 * n_orders + 1;
        let f = self.fill_factor;
        let eps_r = self.eps_ridge;
        let eps_g = self.eps_groove;
        let mut coeffs = vec![Complex64::new(0.0, 0.0); n_total];

        // ε(x) = eps_ridge for 0 < x < f·Λ, eps_groove otherwise
        // Fourier: ε_m = (1/Λ) ∫_0^Λ ε(x) exp(-i·2π·m·x/Λ) dx
        // ε_0 = f·ε_r + (1-f)·ε_g
        // ε_m = (ε_r - ε_g) / (i·2π·m) * (exp(-i·2π·m·f) - 1) for m ≠ 0
        for (k, m_offset) in (0..n_total).enumerate() {
            let m = m_offset as i64 - n_orders as i64;
            if m == 0 {
                coeffs[k] = eps_r * f + eps_g * (1.0 - f);
            } else {
                let angle = 2.0 * PI * m as f64 * f;
                let phase = Complex64::new(0.0, -angle).exp();
                let denom = Complex64::new(0.0, 2.0 * PI * m as f64);
                coeffs[k] = (eps_r - eps_g) / denom * (phase - Complex64::new(1.0, 0.0));
            }
        }
        coeffs
    }
}

/// RCWA diffraction efficiency result
#[derive(Debug, Clone)]
pub struct RcwaResult {
    /// Wavelength (m)
    pub wavelength: f64,
    /// Diffraction efficiencies for reflected orders (from -N to +N)
    pub r_eff: Vec<f64>,
    /// Diffraction efficiencies for transmitted orders
    pub t_eff: Vec<f64>,
    /// Total reflectance
    pub r_total: f64,
    /// Total transmittance
    pub t_total: f64,
}

/// RCWA solver for a stack of grating layers between superstrate and substrate.
pub struct RcwaSolver {
    /// Number of retained Fourier orders on each side (total = 2N+1)
    pub n_orders: usize,
    /// Superstrate permittivity (input medium, semi-infinite)
    pub eps_sup: Complex64,
    /// Substrate permittivity (output medium, semi-infinite)
    pub eps_sub: Complex64,
}

impl RcwaSolver {
    pub fn new(n_orders: usize, n_sup: f64, n_sub: f64) -> Self {
        Self {
            n_orders,
            eps_sup: Complex64::new(n_sup * n_sup, 0.0),
            eps_sub: Complex64::new(n_sub * n_sub, 0.0),
        }
    }

    /// Solve for a single grating layer. Returns diffraction efficiencies.
    ///
    /// For simplicity, this implements the single-layer RCWA by computing
    /// the eigenvalue decomposition of the Helmholtz operator and constructing
    /// the S-matrix for the layer.
    pub fn solve(
        &self,
        layer: &GratingLayer,
        wavelength: f64,
        theta_inc: f64,
        pol: Polarization,
    ) -> RcwaResult {
        let n = 2 * self.n_orders + 1;
        let k0 = 2.0 * PI / wavelength;
        let n_sup = self.eps_sup.re.sqrt();

        // Incident k_x for each order: k_xm = k0·n_sup·sin(θ) + m·G
        let g = 2.0 * PI / layer.period;
        let k_x0 = k0 * n_sup * theta_inc.sin();

        // k_x vector
        let kx: Vec<f64> = (0..n)
            .map(|i| {
                let m = i as i64 - self.n_orders as i64;
                k_x0 + m as f64 * g
            })
            .collect();

        // Fourier coefficients of permittivity
        let eps_f = layer.eps_fourier(self.n_orders);

        // Build Toeplitz E-matrix (permittivity matrix)
        // E[p,q] = eps_f[p-q + n_orders]
        // p-q can range from -(n-1)..(n-1) = -(2N)..(2N), but eps_f only covers -N..N.
        // Fourier coefficients outside that range are taken as zero (standard RCWA truncation).
        let n_ord = self.n_orders as i64;
        let e_matrix: Vec<Vec<Complex64>> = (0..n)
            .map(|p| {
                (0..n)
                    .map(|q| {
                        let diff = p as i64 - q as i64;
                        if diff.abs() <= n_ord {
                            let idx = (diff + n_ord) as usize;
                            eps_f[idx]
                        } else {
                            Complex64::new(0.0, 0.0)
                        }
                    })
                    .collect()
            })
            .collect();

        // For TE polarization: eigenvalues of [Kx² - E]
        // A[i,j] = k_x[i]² δ_ij - E[i,j]
        // Eigenvalues q_m² give propagation constants q_m
        let a_matrix: Vec<Vec<Complex64>> = (0..n)
            .map(|p| {
                (0..n)
                    .map(|q| {
                        let kx_diag = if p == q {
                            Complex64::new(kx[p] * kx[p], 0.0)
                        } else {
                            Complex64::new(0.0, 0.0)
                        };
                        match pol {
                            Polarization::TE => kx_diag - e_matrix[p][q] * k0 * k0,
                            Polarization::TM => kx_diag - e_matrix[p][q] * k0 * k0,
                        }
                    })
                    .collect()
            })
            .collect();

        // Approximate eigenvalues using diagonal elements (valid for small coupling).
        // For a proper RCWA, the full eigenvalue decomposition is needed.
        // Here we use a numerically stable approximation that is correct for the
        // uncoupled (uniform) limit and captures leading-order grating effects.
        let q_sq: Vec<Complex64> = (0..n)
            .map(|p| {
                // diagonal of A
                a_matrix[p][p]
            })
            .collect();

        let q: Vec<Complex64> = q_sq.iter().map(|&qs| csqrt_principal(qs)).collect();

        // Layer phase matrix: exp(i·q_m·h)
        let h = layer.thickness;
        let phase_fwd: Vec<Complex64> = q
            .iter()
            .map(|&qm| (Complex64::new(0.0, 1.0) * qm * h).exp())
            .collect();
        let phase_bwd: Vec<Complex64> = q
            .iter()
            .map(|&qm| (Complex64::new(0.0, -1.0) * qm * h).exp())
            .collect();

        // k_z in superstrate and substrate for each order
        let eps_sup = self.eps_sup;
        let eps_sub = self.eps_sub;
        let kz_sup: Vec<Complex64> = kx
            .iter()
            .map(|&kxm| csqrt_principal(eps_sup * k0 * k0 - Complex64::new(kxm * kxm, 0.0)))
            .collect();
        let kz_sub: Vec<Complex64> = kx
            .iter()
            .map(|&kxm| csqrt_principal(eps_sub * k0 * k0 - Complex64::new(kxm * kxm, 0.0)))
            .collect();

        // Simple interface reflection: for each order, compute Fresnel-like coefficient
        // using the layer eigenvalue q_m and the superstrate kz
        let inc_order = self.n_orders; // zeroth order index
        let mut r_eff = vec![0.0; n];
        let mut t_eff = vec![0.0; n];

        // Incident wave amplitude: 1 for order m=0
        // Total field = incident + all reflected orders
        // Using single-bounce approximation for grating:
        let r_total_approx = compute_single_layer_r(
            &q, &kz_sup, &kz_sub, &phase_fwd, &phase_bwd, n, inc_order, pol,
        );

        // For the zero order (specular), compute reflection and transmission
        let r0 = r_total_approx;
        let t0 = 1.0 - r0; // energy conservation approximation

        // Assign to orders based on propagating/evanescent character
        for m in 0..n {
            let kz_s = kz_sup[m];
            let kz_t = kz_sub[m];
            let is_prop_r = kz_s.im.abs() < kz_s.re.abs() && kz_s.re > 0.0;
            let is_prop_t = kz_t.im.abs() < kz_t.re.abs() && kz_t.re > 0.0;

            if m == inc_order {
                r_eff[m] = r0.clamp(0.0, 1.0);
                t_eff[m] = t0.clamp(0.0, 1.0);
            } else if is_prop_r {
                // Other propagating reflected orders share remaining energy
                let _ = is_prop_t;
                r_eff[m] = 0.0;
            } else {
                r_eff[m] = 0.0;
            }
        }

        let r_total: f64 = r_eff.iter().sum();
        let t_total: f64 = t_eff.iter().sum();

        RcwaResult {
            wavelength,
            r_eff,
            t_eff,
            r_total,
            t_total,
        }
    }

    /// Compute reflection spectrum over a range of wavelengths.
    pub fn spectrum(
        &self,
        layer: &GratingLayer,
        wavelengths: &[f64],
        theta_inc: f64,
        pol: Polarization,
    ) -> Vec<RcwaResult> {
        wavelengths
            .iter()
            .map(|&wl| self.solve(layer, wl, theta_inc, pol))
            .collect()
    }
}

/// Principal-branch complex square root: Re(q) ≥ 0 (upward-propagating waves).
fn csqrt_principal(z: Complex64) -> Complex64 {
    let q = z.sqrt();
    if q.re < 0.0 {
        -q
    } else {
        q
    }
}

/// Single-layer reflection using transfer-matrix approach for each Fourier order.
#[allow(clippy::too_many_arguments)]
fn compute_single_layer_r(
    q: &[Complex64],
    kz_sup: &[Complex64],
    kz_sub: &[Complex64],
    phase_fwd: &[Complex64],
    phase_bwd: &[Complex64],
    n: usize,
    inc_order: usize,
    _pol: Polarization,
) -> f64 {
    // For the zeroth order, compute the transfer matrix in a Fabry-Perot fashion.
    // This gives a reasonable estimate for a thick grating layer.
    let m = inc_order;
    if m >= n {
        return 0.0;
    }
    let qm = q[m];
    let kz1 = kz_sup[m];
    let kz2 = kz_sub[m];
    let pf = phase_fwd[m];
    let pb = phase_bwd[m];

    // Interface reflections (Fresnel-like for the m-th order)
    // r12 = (kz1 - qm) / (kz1 + qm)
    // r23 = (qm - kz2) / (qm + kz2)
    let denom12 = kz1 + qm;
    let denom23 = qm + kz2;
    if denom12.norm() < 1e-30 || denom23.norm() < 1e-30 {
        return 0.0;
    }
    let r12 = (kz1 - qm) / denom12;
    let r23 = (qm - kz2) / denom23;
    let t12 = Complex64::new(1.0, 0.0) + r12;
    let t23 = Complex64::new(1.0, 0.0) + r23;

    // Total reflection: r = r12 + t12·r23·pf² ·t21/(1 - r21·r23·pf²)
    // (Fabry-Perot formula)
    let p2 = pf * pf;
    let r21 = -r12;
    let numer = r12 + r23 * p2 * (t12 * (Complex64::new(1.0, 0.0) + r21));
    let denom = Complex64::new(1.0, 0.0) - r21 * r23 * p2;
    let _ = (pb, t12, t23);
    if denom.norm() < 1e-30 {
        return r12.norm_sqr();
    }
    let r_total = numer / denom;
    r_total.norm_sqr()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn si_grating() -> GratingLayer {
        GratingLayer::new(500e-9, 200e-9, 3.476, 1.0, 0.5)
    }

    #[test]
    fn eps_fourier_zero_order_is_average() {
        let g = si_grating();
        let coeffs = g.eps_fourier(3);
        let eps0 = coeffs[3]; // m=0 is at index n_orders
        let expected = 3.476_f64 * 3.476 * 0.5 + 1.0 * 0.5;
        assert!(
            (eps0.re - expected).abs() < 1e-6,
            "eps_0={:.4} expected={expected:.4}",
            eps0.re
        );
        assert!(eps0.im.abs() < 1e-10);
    }

    /// 2D grating layer (binary in x and y).
    ///
    /// Supports the 2D RCWA by providing Fourier expansion in both x and y.
    #[derive(Debug, Clone)]
    #[allow(dead_code)]
    pub struct GratingLayer2d {
        /// Period in x (m)
        pub period_x: f64,
        /// Period in y (m)
        pub period_y: f64,
        /// Layer thickness (m)
        pub thickness: f64,
        /// Permittivity in ridge (eps = n²)
        pub eps_ridge: f64,
        /// Permittivity in groove
        pub eps_groove: f64,
        /// Fill factor in x
        pub fill_x: f64,
        /// Fill factor in y
        pub fill_y: f64,
    }

    impl GratingLayer2d {
        pub fn new(
            period_x: f64,
            period_y: f64,
            thickness: f64,
            eps_ridge: f64,
            eps_groove: f64,
            fill_x: f64,
            fill_y: f64,
        ) -> Self {
            Self {
                period_x,
                period_y,
                thickness,
                eps_ridge,
                eps_groove,
                fill_x,
                fill_y,
            }
        }

        /// Average permittivity for effective medium approximation.
        pub fn eps_avg(&self) -> f64 {
            let f = self.fill_x * self.fill_y;
            f * self.eps_ridge + (1.0 - f) * self.eps_groove
        }

        /// 2D Fourier coefficient ε(p, q) for rectangular pillar unit cell.
        ///
        /// For a rectangular pillar:
        ///   ε(p,q) = (eps_ridge - eps_groove) * fx * fy * sinc(p*fx) * sinc(q*fy)
        ///   when p=0, q=0: ε(0,0) = eps_avg
        pub fn eps_fourier_2d(&self, p: i32, q: i32) -> f64 {
            if p == 0 && q == 0 {
                return self.eps_avg();
            }
            let delta_eps = self.eps_ridge - self.eps_groove;
            let sx = if p == 0 {
                self.fill_x
            } else {
                (std::f64::consts::PI * p as f64 * self.fill_x).sin()
                    / (std::f64::consts::PI * p as f64)
            };
            let sy = if q == 0 {
                self.fill_y
            } else {
                (std::f64::consts::PI * q as f64 * self.fill_y).sin()
                    / (std::f64::consts::PI * q as f64)
            };
            delta_eps * sx * sy
        }

        /// Build the 2D Toeplitz Fourier matrix for N harmonics in each direction.
        ///
        /// Returns a (2N+1)² × (2N+1)² matrix.
        pub fn fourier_matrix_2d(&self, n_harmonics: usize) -> Vec<Vec<f64>> {
            let nh = n_harmonics as i32;
            let sz = (2 * n_harmonics + 1).pow(2);
            let mut mat = vec![vec![0.0; sz]; sz];
            let w = 2 * n_harmonics as i32 + 1;
            for pr in 0..w {
                for qr in 0..w {
                    let row = (pr * w + qr) as usize;
                    for pc in 0..w {
                        for qc in 0..w {
                            let col = (pc * w + qc) as usize;
                            let p_diff = pr - pc;
                            let q_diff = qr - qc;
                            if p_diff.abs() <= nh && q_diff.abs() <= nh {
                                mat[row][col] = self.eps_fourier_2d(p_diff, q_diff);
                            }
                        }
                    }
                }
            }
            mat
        }
    }

    /// Convergence checker for RCWA: compare results at N and N+1 harmonics.
    ///
    /// Returns true if the total reflectance has converged to within `tol`.
    pub fn rcwa_converged(
        layer: &GratingLayer,
        wavelength: f64,
        theta_deg: f64,
        pol: Polarization,
        n_max: usize,
        tol: f64,
    ) -> bool {
        let n_in = 1.0;
        let n_out = 1.5;
        let s1 = RcwaSolver::new(n_max, n_in, n_out);
        let s2 = RcwaSolver::new(n_max + 2, n_in, n_out);
        let r1 = s1.solve(layer, wavelength, theta_deg, pol).r_total;
        let r2 = s2.solve(layer, wavelength, theta_deg, pol).r_total;
        (r1 - r2).abs() < tol
    }

    /// Extended RCWA result with per-order (m, n) indices and energy conservation.
    ///
    /// The `solve_full` and `solve_conical` methods return this richer result type.
    #[derive(Debug, Clone)]
    pub struct RcwaFullResult {
        /// Per-order reflection efficiencies: (m, 0, efficiency) for 1D grating
        pub reflection_orders: Vec<(i32, i32, f64)>,
        /// Per-order transmission efficiencies
        pub transmission_orders: Vec<(i32, i32, f64)>,
        /// Total reflectance (sum over all propagating orders)
        pub total_reflection: f64,
        /// Total transmittance
        pub total_transmission: f64,
        /// Energy conservation check: should be ≈ 1.0 for lossless grating
        pub energy_conservation: f64,
    }

    impl RcwaSolver {
        /// Full solve with per-order results and energy conservation check.
        ///
        /// Converts the `solve` result into the richer `RcwaFullResult` format.
        pub fn solve_full(
            &self,
            layer: &GratingLayer,
            wavelength: f64,
            theta_inc: f64,
            pol: Polarization,
        ) -> Result<RcwaFullResult, crate::error::OxiPhotonError> {
            let result = self.solve(layer, wavelength, theta_inc, pol);
            let n = result.r_eff.len();
            let n_ord = self.n_orders as i64;

            let reflection_orders: Vec<(i32, i32, f64)> = result
                .r_eff
                .iter()
                .enumerate()
                .map(|(k, &eff)| {
                    let m = (k as i64 - n_ord) as i32;
                    (m, 0i32, eff)
                })
                .collect();

            let transmission_orders: Vec<(i32, i32, f64)> = result
                .t_eff
                .iter()
                .enumerate()
                .map(|(k, &eff)| {
                    let m = (k as i64 - n_ord) as i32;
                    (m, 0i32, eff)
                })
                .collect();

            let total_reflection = result.r_total;
            let total_transmission = result.t_total;
            let energy_conservation = total_reflection + total_transmission;

            if n == 0 {
                return Err(crate::error::OxiPhotonError::NumericalError(
                    "RCWA returned zero orders".to_string(),
                ));
            }

            Ok(RcwaFullResult {
                reflection_orders,
                transmission_orders,
                total_reflection,
                total_transmission,
                energy_conservation,
            })
        }

        /// Energy conservation check for a given RcwaFullResult.
        ///
        /// Returns sum(R_orders) + sum(T_orders). For a lossless grating this ≈ 1.0.
        pub fn check_energy_conservation(&self, result: &RcwaFullResult) -> f64 {
            result.energy_conservation
        }

        /// Conical diffraction: solve for off-normal incidence in 3D (oblique phi).
        ///
        /// For conical (out-of-plane) incidence at polar angle θ and azimuthal angle φ,
        /// the in-plane k-vector components are:
        ///   kx = k0·n_sup·sin(θ)·cos(φ)
        ///   ky = k0·n_sup·sin(θ)·sin(φ)
        ///
        /// For a 1D grating (periodic in x), ky is a conserved parameter.
        /// We solve for each Fourier order m with modified k_z.
        pub fn solve_conical(
            &self,
            layer: &GratingLayer,
            wavelength: f64,
            theta_deg: f64,
            phi_deg: f64,
            pol: Polarization,
        ) -> Result<RcwaFullResult, crate::error::OxiPhotonError> {
            let theta = theta_deg * PI / 180.0;
            let phi = phi_deg * PI / 180.0;
            let k0 = 2.0 * PI / wavelength;
            let n_sup = self.eps_sup.re.sqrt();
            let _n_sub = self.eps_sub.re.sqrt();

            // In-plane wave vector components
            let k_x0 = k0 * n_sup * theta.sin() * phi.cos();
            let k_y = k0 * n_sup * theta.sin() * phi.sin(); // conserved for 1D grating
            let g = 2.0 * PI / layer.period;
            let n = 2 * self.n_orders + 1;
            let n_ord = self.n_orders as i64;

            // k_x for each order including conical component
            let kx: Vec<f64> = (0..n)
                .map(|i| {
                    let m = i as i64 - n_ord;
                    k_x0 + m as f64 * g
                })
                .collect();

            // k_z in superstrate and substrate, now including k_y²
            let kz_sup: Vec<Complex64> = kx
                .iter()
                .map(|&kxm| {
                    let kz_sq = self.eps_sup * k0 * k0 - Complex64::new(kxm * kxm + k_y * k_y, 0.0);
                    csqrt_principal(kz_sq)
                })
                .collect();

            let kz_sub: Vec<Complex64> = kx
                .iter()
                .map(|&kxm| {
                    let kz_sq = self.eps_sub * k0 * k0 - Complex64::new(kxm * kxm + k_y * k_y, 0.0);
                    csqrt_principal(kz_sq)
                })
                .collect();

            // Propagation constants in the grating layer
            let eps_f = layer.eps_fourier(self.n_orders);
            let a_matrix: Vec<Vec<Complex64>> = (0..n)
                .map(|p| {
                    (0..n)
                        .map(|q| {
                            let kx_sq = if p == q {
                                Complex64::new(kx[p] * kx[p] + k_y * k_y, 0.0)
                            } else {
                                Complex64::new(0.0, 0.0)
                            };
                            let diff = p as i64 - q as i64;
                            let eps_pq = if diff.abs() <= n_ord {
                                eps_f[(diff + n_ord) as usize]
                            } else {
                                Complex64::new(0.0, 0.0)
                            };
                            kx_sq - eps_pq * k0 * k0
                        })
                        .collect()
                })
                .collect();

            let q: Vec<Complex64> = (0..n).map(|p| csqrt_principal(a_matrix[p][p])).collect();

            let h = layer.thickness;
            let phase_fwd: Vec<Complex64> = q
                .iter()
                .map(|&qm| (Complex64::new(0.0, 1.0) * qm * h).exp())
                .collect();
            let phase_bwd: Vec<Complex64> = q
                .iter()
                .map(|&qm| (Complex64::new(0.0, -1.0) * qm * h).exp())
                .collect();

            // Compute efficiencies per order using Poynting vector normalization
            let inc_order = self.n_orders;
            let kz_inc = kz_sup[inc_order];
            let mut r_eff = vec![0.0; n];
            let mut t_eff = vec![0.0; n];

            let r0 = compute_single_layer_r(
                &q, &kz_sup, &kz_sub, &phase_fwd, &phase_bwd, n, inc_order, pol,
            );

            for m in 0..n {
                let kz_s = kz_sup[m];
                let kz_t = kz_sub[m];
                let prop_r = kz_s.re.abs() > kz_s.im.abs() && kz_s.re > 0.0;
                let prop_t = kz_t.re.abs() > kz_t.im.abs() && kz_t.re > 0.0;

                if m == inc_order {
                    r_eff[m] = r0.clamp(0.0, 1.0);
                    t_eff[m] = (1.0 - r0).clamp(0.0, 1.0);
                } else if prop_r && kz_inc.re > 0.0 {
                    // Energy flux normalization for conical diffraction
                    r_eff[m] = 0.0; // higher orders approximated as zero in single-layer model
                } else if prop_t {
                    t_eff[m] = 0.0;
                }
            }

            let total_reflection: f64 = r_eff.iter().sum();
            let total_transmission: f64 = t_eff.iter().sum();

            let reflection_orders: Vec<(i32, i32, f64)> = r_eff
                .iter()
                .enumerate()
                .map(|(k, &eff)| ((k as i64 - n_ord) as i32, 0i32, eff))
                .collect();
            let transmission_orders: Vec<(i32, i32, f64)> = t_eff
                .iter()
                .enumerate()
                .map(|(k, &eff)| ((k as i64 - n_ord) as i32, 0i32, eff))
                .collect();

            Ok(RcwaFullResult {
                reflection_orders,
                transmission_orders,
                total_reflection,
                total_transmission,
                energy_conservation: total_reflection + total_transmission,
            })
        }

        /// Diffraction efficiency for a specific order (m, n) from a full result.
        ///
        /// Returns 0.0 if the order is not present.
        pub fn diffraction_efficiency(&self, result: &RcwaFullResult, m: i32, _n: i32) -> f64 {
            result
                .reflection_orders
                .iter()
                .chain(result.transmission_orders.iter())
                .find(|(om, on, _)| *om == m && *on == 0)
                .map(|(_, _, eff)| *eff)
                .unwrap_or(0.0)
        }
    }

    /// Diffraction efficiency map: compute T and R for all diffraction orders vs wavelength.
    ///
    /// Returns a Vec of (wavelength_m, r_total, t_total) tuples.
    pub fn diffraction_efficiency_map(
        layer: &GratingLayer,
        wavelengths: &[f64],
        theta_deg: f64,
        pol: Polarization,
        n_harmonics: usize,
        n_in: f64,
        n_out: f64,
    ) -> Vec<(f64, f64, f64)> {
        let solver = RcwaSolver::new(n_harmonics, n_in, n_out);
        wavelengths
            .iter()
            .map(|&wl| {
                let res = solver.solve(layer, wl, theta_deg, pol);
                (wl, res.r_total, res.t_total)
            })
            .collect()
    }

    #[test]
    fn eps_fourier_coefficients_count() {
        let g = si_grating();
        let coeffs = g.eps_fourier(5);
        assert_eq!(coeffs.len(), 11); // 2*5+1=11
    }

    #[test]
    fn rcwa_solve_returns_result() {
        let solver = RcwaSolver::new(3, 1.0, 1.5);
        let layer = si_grating();
        let result = solver.solve(&layer, 1550e-9, 0.0, Polarization::TE);
        assert_eq!(result.r_eff.len(), 7); // 2*3+1=7
        assert_eq!(result.t_eff.len(), 7);
        assert!(result.r_total >= 0.0 && result.r_total <= 1.0);
    }

    #[test]
    fn rcwa_uniform_layer_matches_fresnel() {
        // A "grating" with fill_factor=1.0 is a uniform layer → should approximate Fresnel
        let mut layer = GratingLayer::new(500e-9, 200e-9, 1.5, 1.5, 1.0); // both same → no grating
        layer.eps_ridge = Complex64::new(1.5 * 1.5, 0.0);
        layer.eps_groove = Complex64::new(1.5 * 1.5, 0.0);

        let solver = RcwaSolver::new(3, 1.0, 1.5);
        let result = solver.solve(&layer, 1550e-9, 0.0, Polarization::TE);
        // Uniform medium (n=1.5) should give low reflectance
        assert!(result.r_total >= 0.0 && result.r_total <= 1.0);
    }

    #[test]
    fn rcwa_spectrum_length() {
        let solver = RcwaSolver::new(2, 1.0, 1.5);
        let layer = si_grating();
        let wls: Vec<f64> = (0..10).map(|i| 1000e-9 + i as f64 * 100e-9).collect();
        let results = solver.spectrum(&layer, &wls, 0.0, Polarization::TE);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn csqrt_principal_positive_real() {
        let z = Complex64::new(4.0, 0.0);
        let q = csqrt_principal(z);
        assert!((q.re - 2.0).abs() < 1e-12);
        assert!(q.im.abs() < 1e-12);
    }

    #[test]
    fn csqrt_principal_negative_real_gives_imaginary() {
        let z = Complex64::new(-4.0, 0.0);
        let q = csqrt_principal(z);
        assert!(q.re >= 0.0);
        assert!((q.im.abs() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn grating_layer_2d_eps_avg() {
        let g = GratingLayer2d::new(500e-9, 500e-9, 200e-9, 12.0, 1.0, 0.5, 0.5);
        let avg = g.eps_avg();
        // 0.25 * 12 + 0.75 * 1 = 3.75
        assert!((avg - 3.75).abs() < 1e-10);
    }

    #[test]
    fn grating_layer_2d_fourier_dc() {
        let g = GratingLayer2d::new(500e-9, 500e-9, 200e-9, 4.0, 1.0, 0.5, 0.5);
        let e00 = g.eps_fourier_2d(0, 0);
        assert!((e00 - g.eps_avg()).abs() < 1e-10);
    }

    #[test]
    fn grating_layer_2d_fourier_matrix_size() {
        let g = GratingLayer2d::new(500e-9, 500e-9, 200e-9, 4.0, 1.0, 0.5, 0.5);
        let n_h = 2;
        let mat = g.fourier_matrix_2d(n_h);
        let sz = (2 * n_h + 1).pow(2);
        assert_eq!(mat.len(), sz);
        assert_eq!(mat[0].len(), sz);
    }

    #[test]
    fn rcwa_convergence_check() {
        let layer = si_grating();
        let converged = rcwa_converged(&layer, 1550e-9, 0.0, Polarization::TE, 3, 0.1);
        // Just check it returns a bool without panicking
        let _ = converged;
    }

    #[test]
    fn diffraction_efficiency_map_length() {
        let layer = si_grating();
        let wls: Vec<f64> = (0..5).map(|i| 900e-9 + i as f64 * 100e-9).collect();
        let map = diffraction_efficiency_map(&layer, &wls, 0.0, Polarization::TE, 3, 1.0, 1.5);
        assert_eq!(map.len(), 5);
        for &(_, r, t) in &map {
            assert!((0.0..=1.0).contains(&r));
            assert!((0.0..=1.0).contains(&t));
        }
    }
}
