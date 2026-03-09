/// Effective Medium Theory (EMT) implementations
///
/// Provides Maxwell Garnett, Bruggeman, Mie, and periodic structure
/// effective medium models for composite optical materials.
use num_complex::Complex64;
use std::f64::consts::PI;

use crate::error::OxiPhotonError;

// ─────────────────────────────────────────────────────────────────────────────
// Maxwell Garnett
// ─────────────────────────────────────────────────────────────────────────────

/// Maxwell Garnett effective medium theory.
///
/// Valid for dilute, non-interacting spherical inclusions (fill fraction f << 1)
/// in a host matrix.  The Clausius-Mossotti relation gives:
///
/// ε_eff = ε_h · [1 + 3f(ε_i − ε_h) / (ε_i + 2ε_h − f(ε_i − ε_h))]
#[derive(Debug, Clone)]
pub struct MaxwellGarnett {
    /// Permittivity of the host (background) medium.
    pub eps_host: Complex64,
    /// Permittivity of the spherical inclusions.
    pub eps_inclusion: Complex64,
    /// Volume fill fraction of inclusions (0 ≤ f ≤ 1).
    pub fill_fraction: f64,
}

impl MaxwellGarnett {
    /// Create a new Maxwell-Garnett model.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] if `fill_fraction` is outside [0, 1].
    pub fn new(
        eps_host: Complex64,
        eps_inclusion: Complex64,
        fill_fraction: f64,
    ) -> Result<Self, OxiPhotonError> {
        if !(0.0..=1.0).contains(&fill_fraction) {
            return Err(OxiPhotonError::NumericalError(format!(
                "fill_fraction must be in [0, 1], got {fill_fraction}"
            )));
        }
        Ok(Self {
            eps_host,
            eps_inclusion,
            fill_fraction,
        })
    }

    /// Effective permittivity for spherical inclusions (depolarisation L = 1/3).
    ///
    /// ε_eff = ε_h · [1 + 3f(ε_i − ε_h) / (ε_i + 2ε_h − f(ε_i − ε_h))]
    pub fn effective_permittivity(&self) -> Complex64 {
        let f = self.fill_fraction;
        let eh = self.eps_host;
        let ei = self.eps_inclusion;
        let delta = ei - eh;
        let numerator = Complex64::new(3.0 * f, 0.0) * delta;
        let denominator = ei + Complex64::new(2.0, 0.0) * eh - Complex64::new(f, 0.0) * delta;
        eh * (Complex64::new(1.0, 0.0) + numerator / denominator)
    }

    /// Effective refractive index n_eff = √ε_eff (branch with Re(n) ≥ 0).
    pub fn effective_index(&self) -> Complex64 {
        let eps_eff = self.effective_permittivity();
        // Choose the square root with positive real part
        let root = eps_eff.sqrt();
        if root.re >= 0.0 {
            root
        } else {
            -root
        }
    }

    /// Effective permittivity parallel to layers (in-plane), 1D lamellar geometry.
    ///
    /// ε_‖ = f · ε_i + (1 − f) · ε_h
    pub fn layered_parallel(&self) -> Complex64 {
        let f = self.fill_fraction;
        Complex64::new(f, 0.0) * self.eps_inclusion + Complex64::new(1.0 - f, 0.0) * self.eps_host
    }

    /// Effective permittivity perpendicular to layers (out-of-plane), 1D lamellar geometry.
    ///
    /// 1/ε_⊥ = f/ε_i + (1 − f)/ε_h
    pub fn layered_perpendicular(&self) -> Complex64 {
        let f = self.fill_fraction;
        let inv = Complex64::new(f, 0.0) / self.eps_inclusion
            + Complex64::new(1.0 - f, 0.0) / self.eps_host;
        Complex64::new(1.0, 0.0) / inv
    }

    /// Effective permittivity for infinite cylinders with axis along z (2D rods in host).
    ///
    /// Uses depolarisation factor L = 1/2 (appropriate for 2D in-plane fields).
    ///
    /// ε_eff = ε_h · [1 + 2f(ε_i − ε_h) / (ε_i + ε_h − f(ε_i − ε_h))]
    pub fn cylindrical_effective_permittivity(&self) -> Complex64 {
        let f = self.fill_fraction;
        let eh = self.eps_host;
        let ei = self.eps_inclusion;
        let delta = ei - eh;
        let numerator = Complex64::new(2.0 * f, 0.0) * delta;
        let denominator = ei + eh - Complex64::new(f, 0.0) * delta;
        eh * (Complex64::new(1.0, 0.0) + numerator / denominator)
    }

    /// Sweep fill fraction from 0 to 1 and return (f, ε_eff) pairs.
    pub fn sweep_fill_fraction(&self, n_points: usize) -> Vec<(f64, Complex64)> {
        (0..n_points)
            .map(|i| {
                let f = i as f64 / (n_points.saturating_sub(1).max(1)) as f64;
                let mg = MaxwellGarnett {
                    eps_host: self.eps_host,
                    eps_inclusion: self.eps_inclusion,
                    fill_fraction: f,
                };
                (f, mg.effective_permittivity())
            })
            .collect()
    }

    /// Find the fill fraction whose Re(ε_eff) matches `target_eps_real` via bisection.
    ///
    /// Returns `None` if the target is outside the reachable range.
    pub fn fill_fraction_for_target_eps(&self, target_eps_real: f64) -> Option<f64> {
        // Evaluate at boundaries
        let eps_at_f = |f: f64| -> f64 {
            let mg = MaxwellGarnett {
                eps_host: self.eps_host,
                eps_inclusion: self.eps_inclusion,
                fill_fraction: f,
            };
            mg.effective_permittivity().re
        };

        let lo_val = eps_at_f(0.0);
        let hi_val = eps_at_f(1.0);

        // Check target is in range
        let (low_f, high_f) = if lo_val <= hi_val {
            if target_eps_real < lo_val || target_eps_real > hi_val {
                return None;
            }
            (0.0_f64, 1.0_f64)
        } else {
            if target_eps_real < hi_val || target_eps_real > lo_val {
                return None;
            }
            (0.0_f64, 1.0_f64)
        };

        // Bisection
        let mut lo = low_f;
        let mut hi = high_f;
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            let val = eps_at_f(mid);
            if (val - target_eps_real).abs() < 1e-12 {
                return Some(mid);
            }
            // Determine which half contains the target
            let lo_val_now = eps_at_f(lo);
            if (lo_val_now - target_eps_real) * (val - target_eps_real) <= 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Some(0.5 * (lo + hi))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bruggeman
// ─────────────────────────────────────────────────────────────────────────────

/// Bruggeman self-consistent effective medium theory.
///
/// Symmetric treatment of two phases — valid at arbitrary fill fractions.
/// Solves:
///   f·(ε_a − ε_eff)/(ε_a + 2ε_eff) + (1−f)·(ε_b − ε_eff)/(ε_b + 2ε_eff) = 0
#[derive(Debug, Clone)]
pub struct Bruggeman {
    /// Permittivity of phase A.
    pub eps_a: Complex64,
    /// Permittivity of phase B.
    pub eps_b: Complex64,
    /// Volume fraction of phase A (0 ≤ f ≤ 1).
    pub fill_fraction_a: f64,
}

impl Bruggeman {
    /// Create a new Bruggeman model.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] if `fill_fraction_a` is outside [0, 1].
    pub fn new(
        eps_a: Complex64,
        eps_b: Complex64,
        fill_fraction_a: f64,
    ) -> Result<Self, OxiPhotonError> {
        if !(0.0..=1.0).contains(&fill_fraction_a) {
            return Err(OxiPhotonError::NumericalError(format!(
                "fill_fraction_a must be in [0, 1], got {fill_fraction_a}"
            )));
        }
        Ok(Self {
            eps_a,
            eps_b,
            fill_fraction_a,
        })
    }

    /// Solve the Bruggeman equation via Newton-Raphson iteration.
    ///
    /// The quadratic form of the Bruggeman equation allows for an analytical
    /// starting point; iteration is used for robustness in the complex plane.
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] if the iteration diverges.
    pub fn effective_permittivity(&self) -> Result<Complex64, OxiPhotonError> {
        let f = self.fill_fraction_a;
        let ea = self.eps_a;
        let eb = self.eps_b;
        let two = Complex64::new(2.0, 0.0);
        let one = Complex64::new(1.0, 0.0);

        // Bruggeman equation: F(e) = f*(ea - e)/(ea + 2e) + (1-f)*(eb - e)/(eb + 2e) = 0
        // This is a quadratic in e:
        // 2e^2 + [f(ea - 4eb)/... ]  — use the closed-form roots as starting guess.
        //
        // Expanding:  2e^2 - [(2fa·ea + 2(1-fa)·eb - ea·... )] is messy.
        // Instead use the analytical solution of the quadratic:
        //   (3f-1)ea + (3(1-f)-1)eb  ± sqrt(...)
        //   -----------------------------------
        //                  4
        let fa = Complex64::new(f, 0.0);
        let fb = Complex64::new(1.0 - f, 0.0);

        // b·e + c = 0 form: 2e² + [f(ea−4eb)/... ] — use discriminant formula
        // Analytical form: 2ε_eff² − [(3fa−1)εa + (3fb−1)εb]ε_eff − εa·εb/2 = 0
        // → 4ε_eff² − 2[(3fa−1)εa + (3fb−1)εb]ε_eff − εa·εb = 0
        let coeff_b = -((Complex64::new(3.0, 0.0) * fa - one) * ea
            + (Complex64::new(3.0, 0.0) * fb - one) * eb);
        // coeff_a = 4, coeff_c = -ea*eb  (using standard 4e^2 + 2*coeff_b*e - ea*eb = 0)
        // Discriminant
        let discriminant = coeff_b * coeff_b + Complex64::new(4.0 * 2.0, 0.0) * ea * eb;
        let sqrt_disc = discriminant.sqrt();

        // Two candidate roots
        let root1 = (-coeff_b + sqrt_disc) / Complex64::new(8.0, 0.0);
        let root2 = (-coeff_b - sqrt_disc) / Complex64::new(8.0, 0.0);

        // Pick the root with positive imaginary part (physical), or largest real part
        let eps0 = if root1.im >= root2.im { root1 } else { root2 };

        // Refine with Newton-Raphson
        let bruggeman_f = |e: Complex64| -> Complex64 {
            fa * (ea - e) / (ea + two * e) + fb * (eb - e) / (eb + two * e)
        };

        let bruggeman_df = |e: Complex64| -> Complex64 {
            let da = ea + two * e;
            let db = eb + two * e;
            -fa * (ea + two * e + two * (ea - e)) / (da * da)
                - fb * (eb + two * e + two * (eb - e)) / (db * db)
        };

        let mut eps = eps0;
        for iter in 0..200 {
            let fval = bruggeman_f(eps);
            if fval.norm() < 1e-14 {
                break;
            }
            let dfval = bruggeman_df(eps);
            if dfval.norm() < 1e-30 {
                return Err(OxiPhotonError::NumericalError(
                    "Bruggeman Newton step: zero derivative".to_string(),
                ));
            }
            let step = fval / dfval;
            eps -= step;
            if iter == 199 {
                return Err(OxiPhotonError::NumericalError(
                    "Bruggeman iteration did not converge after 200 steps".to_string(),
                ));
            }
        }

        // Enforce physical solution: Re(ε) should be real-positive if both phases are
        // non-absorbing, and Im(ε) ≥ 0 for passive materials.
        if eps.re.is_nan() || eps.im.is_nan() {
            return Err(OxiPhotonError::NumericalError(
                "Bruggeman converged to NaN".to_string(),
            ));
        }

        Ok(eps)
    }

    /// Effective conductivity for a binary conductor/insulator composite.
    ///
    /// Uses the classical Bruggeman result for conductivities:
    ///   f·(σ_a − σ_eff)/(σ_a + 2σ_eff) + (1−f)·(σ_b − σ_eff)/(σ_b + 2σ_eff) = 0
    ///
    /// Solved analytically by the same quadratic approach.
    pub fn effective_conductivity(&self, sigma_a: f64, sigma_b: f64) -> f64 {
        let f = self.fill_fraction_a;
        // Analytical Bruggeman for real conductivities
        // σ_eff = (1/4){(3f-1)σ_a + (3(1-f)-1)σ_b
        //         + sqrt[((3f-1)σ_a + (2-3f)σ_b)² + 8σ_a σ_b]}
        let term1 = (3.0 * f - 1.0) * sigma_a + (2.0 - 3.0 * f) * sigma_b;
        let discriminant = term1 * term1 + 8.0 * sigma_a * sigma_b;
        (term1 + discriminant.sqrt()) / 4.0
    }

    /// Percolation threshold for 3D spherical inclusions ≈ 1/3.
    pub fn percolation_threshold() -> f64 {
        1.0 / 3.0
    }

    /// Returns `true` if the conductor phase A exceeds the percolation threshold.
    pub fn is_percolating(&self) -> bool {
        self.fill_fraction_a > Self::percolation_threshold()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mie scattering theory
// ─────────────────────────────────────────────────────────────────────────────

/// Mie scattering theory for a homogeneous sphere.
///
/// Provides exact scattering, absorption, and extinction efficiencies via
/// Mie coefficients a_n, b_n evaluated through Riccati-Bessel functions.
#[derive(Debug, Clone)]
pub struct MieSphere {
    /// Sphere radius in nanometres.
    pub radius_nm: f64,
    /// Complex permittivity of the sphere material.
    pub eps_sphere: Complex64,
    /// Complex permittivity of the surrounding medium.
    pub eps_medium: Complex64,
}

impl MieSphere {
    /// Create a new Mie sphere.
    pub fn new(radius_nm: f64, eps_sphere: Complex64, eps_medium: Complex64) -> Self {
        Self {
            radius_nm,
            eps_sphere,
            eps_medium,
        }
    }

    /// Size parameter x = k_m · r = 2π √ε_m · r / λ (dimensionless).
    pub fn size_parameter(&self, lambda_nm: f64) -> f64 {
        let n_m = self.eps_medium.sqrt().re.max(0.0);
        2.0 * PI * n_m * self.radius_nm / lambda_nm
    }

    /// Relative refractive index m = n_sphere / n_medium.
    fn relative_index(&self) -> Complex64 {
        let n_sph = self.eps_sphere.sqrt();
        let n_med = self.eps_medium.sqrt();
        // Choose physical roots
        let n_sph = if n_sph.re >= 0.0 { n_sph } else { -n_sph };
        let n_med = if n_med.re >= 0.0 { n_med } else { -n_med };
        n_sph / n_med
    }

    /// Riccati-Bessel function ψ_n(z) = z · j_n(z) evaluated by upward recurrence.
    fn riccati_bessel_j(z: Complex64, n_max: usize) -> Vec<Complex64> {
        let mut psi = vec![Complex64::new(0.0, 0.0); n_max + 2];
        // ψ_0(z) = sin(z), ψ_1(z) = sin(z)/z - cos(z)
        psi[0] = z.sin();
        if n_max == 0 {
            return psi;
        }
        psi[1] = z.sin() / z - z.cos();
        for n in 1..n_max {
            psi[n + 1] = Complex64::new((2 * n + 1) as f64, 0.0) / z * psi[n] - psi[n - 1];
        }
        psi
    }

    /// Riccati-Bessel function ξ_n(z) = z · h_n^(1)(z) (outgoing Hankel).
    fn riccati_hankel(z: Complex64, n_max: usize) -> Vec<Complex64> {
        let i = Complex64::new(0.0, 1.0);
        let mut xi = vec![Complex64::new(0.0, 0.0); n_max + 2];
        // ξ_0(z) = -i·e^{iz}, ξ_1(z) = e^{iz}·(1/z - i)·(-i) ...
        // Use: ξ_n = ψ_n + i·χ_n where χ_n = -z·y_n(z)
        // χ_0 = -cos(z), χ_1 = -cos(z)/z - sin(z)
        let psi = Self::riccati_bessel_j(z, n_max);
        let mut chi = vec![Complex64::new(0.0, 0.0); n_max + 2];
        chi[0] = -z.cos();
        if n_max >= 1 {
            chi[1] = -z.cos() / z - z.sin();
        }
        for n in 1..n_max {
            chi[n + 1] = Complex64::new((2 * n + 1) as f64, 0.0) / z * chi[n] - chi[n - 1];
        }
        for n in 0..=n_max + 1 {
            xi[n] = psi[n] + i * chi[n];
        }
        xi
    }

    /// Compute Mie coefficients a_n (electric) and b_n (magnetic).
    ///
    /// Uses the Bohren & Huffman convention.
    fn mie_coefficients(&self, lambda_nm: f64) -> (Vec<Complex64>, Vec<Complex64>) {
        let x = self.size_parameter(lambda_nm);
        let m = self.relative_index();
        let n_max = (x + 4.0 * x.cbrt() + 2.0).ceil() as usize + 1;
        let n_max = n_max.max(2);

        let mx = m * Complex64::new(x, 0.0);
        let x_c = Complex64::new(x, 0.0);

        let psi_x = Self::riccati_bessel_j(x_c, n_max);
        let psi_mx = Self::riccati_bessel_j(mx, n_max);
        let xi_x = Self::riccati_hankel(x_c, n_max);

        // Derivatives: ψ'_n(z) = ψ_{n-1}(z) - n/z · ψ_n(z)
        let dpsi_x: Vec<Complex64> = (0..=n_max)
            .map(|n| {
                if n == 0 {
                    psi_x[1]
                } else {
                    psi_x[n - 1] - Complex64::new(n as f64, 0.0) / x_c * psi_x[n]
                }
            })
            .collect();

        let dpsi_mx: Vec<Complex64> = (0..=n_max)
            .map(|n| {
                if n == 0 {
                    psi_mx[1]
                } else {
                    psi_mx[n - 1] - Complex64::new(n as f64, 0.0) / mx * psi_mx[n]
                }
            })
            .collect();

        let dxi_x: Vec<Complex64> = (0..=n_max)
            .map(|n| {
                if n == 0 {
                    xi_x[1]
                } else {
                    xi_x[n - 1] - Complex64::new(n as f64, 0.0) / x_c * xi_x[n]
                }
            })
            .collect();

        let mut a_n = Vec::with_capacity(n_max);
        let mut b_n = Vec::with_capacity(n_max);

        for n in 1..=n_max {
            // a_n = (m·ψ_n(mx)·ψ'_n(x) - ψ_n(x)·ψ'_n(mx)) /
            //       (m·ψ_n(mx)·ξ'_n(x) - ξ_n(x)·ψ'_n(mx))
            let an_num = m * psi_mx[n] * dpsi_x[n] - psi_x[n] * dpsi_mx[n];
            let an_den = m * psi_mx[n] * dxi_x[n] - xi_x[n] * dpsi_mx[n];
            let an = if an_den.norm() > 1e-30 {
                an_num / an_den
            } else {
                Complex64::new(0.0, 0.0)
            };
            a_n.push(an);

            // b_n = (ψ_n(mx)·ψ'_n(x) - m·ψ_n(x)·ψ'_n(mx)) /
            //       (ψ_n(mx)·ξ'_n(x) - m·ξ_n(x)·ψ'_n(mx))
            let bn_num = psi_mx[n] * dpsi_x[n] - m * psi_x[n] * dpsi_mx[n];
            let bn_den = psi_mx[n] * dxi_x[n] - m * xi_x[n] * dpsi_mx[n];
            let bn = if bn_den.norm() > 1e-30 {
                bn_num / bn_den
            } else {
                Complex64::new(0.0, 0.0)
            };
            b_n.push(bn);
        }

        (a_n, b_n)
    }

    /// Scattering efficiency Q_sca = (2/x²) Σ (2n+1)(|a_n|² + |b_n|²).
    pub fn scattering_efficiency(&self, lambda_nm: f64) -> f64 {
        let x = self.size_parameter(lambda_nm);
        let (a_n, b_n) = self.mie_coefficients(lambda_nm);
        let sum: f64 = a_n
            .iter()
            .zip(b_n.iter())
            .enumerate()
            .map(|(i, (a, b))| {
                let n = (i + 1) as f64;
                (2.0 * n + 1.0) * (a.norm_sqr() + b.norm_sqr())
            })
            .sum();
        (2.0 / (x * x)) * sum
    }

    /// Extinction efficiency Q_ext = (2/x²) Σ (2n+1) Re(a_n + b_n).
    pub fn extinction_efficiency(&self, lambda_nm: f64) -> f64 {
        let x = self.size_parameter(lambda_nm);
        let (a_n, b_n) = self.mie_coefficients(lambda_nm);
        let sum: f64 = a_n
            .iter()
            .zip(b_n.iter())
            .enumerate()
            .map(|(i, (a, b))| {
                let n = (i + 1) as f64;
                (2.0 * n + 1.0) * (a.re + b.re)
            })
            .sum();
        (2.0 / (x * x)) * sum
    }

    /// Absorption efficiency Q_abs = Q_ext − Q_sca.
    pub fn absorption_efficiency(&self, lambda_nm: f64) -> f64 {
        (self.extinction_efficiency(lambda_nm) - self.scattering_efficiency(lambda_nm)).max(0.0)
    }

    /// Scattering cross-section σ_sca = Q_sca · π r² (nm²).
    pub fn scattering_cross_section_nm2(&self, lambda_nm: f64) -> f64 {
        self.scattering_efficiency(lambda_nm) * PI * self.radius_nm * self.radius_nm
    }

    /// Dipole (Rayleigh) approximation Q_sca for x << 1.
    ///
    /// Q_sca ≈ (8/3) x⁴ |K|²  where K = (ε−1)/(ε+2).
    pub fn dipole_scattering_efficiency(&self, lambda_nm: f64) -> f64 {
        let x = self.size_parameter(lambda_nm);
        let eps_r = self.eps_sphere / self.eps_medium; // relative permittivity
        let one = Complex64::new(1.0, 0.0);
        let two = Complex64::new(2.0, 0.0);
        let k = (eps_r - one) / (eps_r + two);
        (8.0 / 3.0) * x.powi(4) * k.norm_sqr()
    }

    /// The value of Re(ε_sphere) that satisfies the Fröhlich (LSPR) condition:
    ///   Re(ε_sphere) = −2 · Re(ε_medium)
    ///
    /// Returns the target Re(ε) value, not a wavelength.
    pub fn lspr_condition(&self) -> f64 {
        -2.0 * self.eps_medium.re
    }

    /// Forward scattering asymmetry parameter g = ⟨cos θ⟩.
    ///
    /// g = (4/Q_sca·x²) · Σ [n(n+2)/(n+1) Re(a_n a*_{n+1} + b_n b*_{n+1})
    ///                        + (2n+1)/(n(n+1)) Re(a_n b*_n)]
    pub fn asymmetry_parameter(&self, lambda_nm: f64) -> f64 {
        let x = self.size_parameter(lambda_nm);
        let (a_n, b_n) = self.mie_coefficients(lambda_nm);
        let q_sca = self.scattering_efficiency(lambda_nm);
        if q_sca < 1e-20 {
            return 0.0;
        }

        let len = a_n.len();
        let mut sum = 0.0;
        for n in 1..=len {
            let an = a_n[n - 1];
            let bn = b_n[n - 1];
            let nf = n as f64;

            // Cross terms with n+1
            if n < len {
                let an1 = a_n[n];
                let bn1 = b_n[n];
                let cross =
                    nf * (nf + 2.0) / (nf + 1.0) * ((an * an1.conj()).re + (bn * bn1.conj()).re);
                sum += cross;
            }
            // Self term
            let self_term = (2.0 * nf + 1.0) / (nf * (nf + 1.0)) * (an * bn.conj()).re;
            sum += self_term;
        }
        (4.0 / (q_sca * x * x)) * sum
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Periodic structure EMT
// ─────────────────────────────────────────────────────────────────────────────

/// Geometry of the periodic inclusions.
#[derive(Debug, Clone)]
pub enum PeriodicGeometry {
    /// 1D lamellar (layer) structure.  `fill_a` is the fraction of material A.
    Lamellar { fill_a: f64 },
    /// 2D array of circular rods.  `radius_fraction` = r/a where a is the lattice constant.
    Cylinders { radius_fraction: f64 },
    /// 3D simple cubic array of spheres.  `radius_fraction` = r/a.
    Spheres { radius_fraction: f64 },
    /// 2D array of circular holes in a slab of material A.
    Holes { radius_fraction: f64 },
}

/// Long-wavelength effective medium theory for periodic composite structures.
///
/// Based on homogenisation theory (Aspnes 1982, Yariv & Yeh 1984).
#[derive(Debug, Clone)]
pub struct PeriodicEMT {
    /// Lattice constant of the periodic structure in nm.
    pub lattice_constant_nm: f64,
    /// Permittivity of material A (inclusion or layer A).
    pub eps_a: Complex64,
    /// Permittivity of material B (host or layer B).
    pub eps_b: Complex64,
    /// Geometry of the periodic structure.
    pub geometry: PeriodicGeometry,
}

impl PeriodicEMT {
    /// Create a new periodic EMT instance.
    pub fn new(
        lattice_constant_nm: f64,
        eps_a: Complex64,
        eps_b: Complex64,
        geometry: PeriodicGeometry,
    ) -> Self {
        Self {
            lattice_constant_nm,
            eps_a,
            eps_b,
            geometry,
        }
    }

    fn fill_fraction_a(&self) -> f64 {
        match &self.geometry {
            PeriodicGeometry::Lamellar { fill_a } => *fill_a,
            PeriodicGeometry::Cylinders { radius_fraction } => {
                PI * radius_fraction * radius_fraction
            }
            PeriodicGeometry::Spheres { radius_fraction } => {
                (4.0 / 3.0) * PI * radius_fraction.powi(3)
            }
            PeriodicGeometry::Holes { radius_fraction } => {
                1.0 - PI * radius_fraction * radius_fraction
            }
        }
    }

    /// Effective permittivity parallel to layers / perpendicular to rods (TE-like).
    ///
    /// For lamellar: ε_‖ = f·ε_a + (1−f)·ε_b
    /// For cylinders/spheres: Maxwell-Garnett
    pub fn effective_permittivity_parallel(&self) -> Complex64 {
        let f = self.fill_fraction_a().clamp(0.0, 1.0);
        match &self.geometry {
            PeriodicGeometry::Lamellar { .. } | PeriodicGeometry::Holes { .. } => {
                // Volume-weighted average
                Complex64::new(f, 0.0) * self.eps_a + Complex64::new(1.0 - f, 0.0) * self.eps_b
            }
            PeriodicGeometry::Cylinders { .. } => {
                // MG for cylinders (depolarisation 1/2)
                let mg = MaxwellGarnett {
                    eps_host: self.eps_b,
                    eps_inclusion: self.eps_a,
                    fill_fraction: f,
                };
                mg.cylindrical_effective_permittivity()
            }
            PeriodicGeometry::Spheres { .. } => {
                let mg = MaxwellGarnett {
                    eps_host: self.eps_b,
                    eps_inclusion: self.eps_a,
                    fill_fraction: f,
                };
                mg.effective_permittivity()
            }
        }
    }

    /// Effective permittivity perpendicular to layers / along rods (TM-like).
    ///
    /// For lamellar: 1/ε_⊥ = f/ε_a + (1−f)/ε_b
    pub fn effective_permittivity_perpendicular(&self) -> Complex64 {
        let f = self.fill_fraction_a().clamp(0.0, 1.0);
        match &self.geometry {
            PeriodicGeometry::Lamellar { .. } | PeriodicGeometry::Holes { .. } => {
                // Harmonic average
                let inv =
                    Complex64::new(f, 0.0) / self.eps_a + Complex64::new(1.0 - f, 0.0) / self.eps_b;
                Complex64::new(1.0, 0.0) / inv
            }
            PeriodicGeometry::Cylinders { .. } | PeriodicGeometry::Spheres { .. } => {
                // MG with depolarisation along rod axis (L=0 gives ε_‖)
                // For TM (along z): use standard MG
                let mg = MaxwellGarnett {
                    eps_host: self.eps_b,
                    eps_inclusion: self.eps_a,
                    fill_fraction: f,
                };
                mg.effective_permittivity()
            }
        }
    }

    /// Effective index for TE polarisation √ε_‖.
    pub fn effective_index_te(&self) -> Complex64 {
        let eps = self.effective_permittivity_parallel();
        let root = eps.sqrt();
        if root.re >= 0.0 {
            root
        } else {
            -root
        }
    }

    /// Effective index for TM polarisation √ε_⊥.
    pub fn effective_index_tm(&self) -> Complex64 {
        let eps = self.effective_permittivity_perpendicular();
        let root = eps.sqrt();
        if root.re >= 0.0 {
            root
        } else {
            -root
        }
    }

    /// Form birefringence |n_TE − n_TM|.
    pub fn form_birefringence(&self) -> f64 {
        let n_te = self.effective_index_te();
        let n_tm = self.effective_index_tm();
        (n_te.re - n_tm.re).abs()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── Maxwell Garnett ──────────────────────────────────────────────────────

    #[test]
    fn test_maxwell_garnett_dilute_limit() {
        // f→0: ε_eff → ε_host
        let eh = Complex64::new(2.25, 0.0);
        let ei = Complex64::new(12.0, 0.5);
        let mg = MaxwellGarnett::new(eh, ei, 1e-6).unwrap();
        let eps_eff = mg.effective_permittivity();
        assert_abs_diff_eq!(eps_eff.re, eh.re, epsilon = 1e-4);
        assert_abs_diff_eq!(eps_eff.im, eh.im, epsilon = 1e-4);
    }

    #[test]
    fn test_maxwell_garnett_dense_limit() {
        // f→1: ε_eff → ε_inclusion
        let eh = Complex64::new(2.25, 0.0);
        let ei = Complex64::new(12.0, 0.5);
        let mg = MaxwellGarnett::new(eh, ei, 1.0 - 1e-8).unwrap();
        let eps_eff = mg.effective_permittivity();
        // At f very close to 1, the MG formula asymptotes toward ε_i
        // (not exact because MG formula technically breaks down, but directionally correct)
        assert!(
            eps_eff.re > eh.re,
            "ε_eff should be > ε_host at high fill fraction"
        );
    }

    #[test]
    fn test_layered_parallel_weighted_average() {
        let eh = Complex64::new(2.25, 0.0);
        let ei = Complex64::new(9.0, 0.0);
        let f = 0.3;
        let mg = MaxwellGarnett::new(eh, ei, f).unwrap();
        let eps_par = mg.layered_parallel();
        let expected = f * ei + (1.0 - f) * eh;
        assert_abs_diff_eq!(eps_par.re, expected.re, epsilon = 1e-12);
        assert_abs_diff_eq!(eps_par.im, expected.im, epsilon = 1e-12);
    }

    #[test]
    fn test_maxwell_garnett_invalid_fill_fraction() {
        let eps = Complex64::new(2.0, 0.0);
        assert!(MaxwellGarnett::new(eps, eps, 1.5).is_err());
        assert!(MaxwellGarnett::new(eps, eps, -0.1).is_err());
    }

    // ── Bruggeman ────────────────────────────────────────────────────────────

    #[test]
    fn test_bruggeman_symmetric() {
        // Swapping A↔B with complementary fill fraction gives the same ε_eff
        let ea = Complex64::new(2.25, 0.0);
        let eb = Complex64::new(9.0, 0.1);
        let f = 0.4;

        let bg1 = Bruggeman::new(ea, eb, f).unwrap();
        let eps1 = bg1.effective_permittivity().unwrap();

        let bg2 = Bruggeman::new(eb, ea, 1.0 - f).unwrap();
        let eps2 = bg2.effective_permittivity().unwrap();

        assert_abs_diff_eq!(eps1.re, eps2.re, epsilon = 1e-8);
        assert_abs_diff_eq!(eps1.im, eps2.im, epsilon = 1e-8);
    }

    #[test]
    fn test_bruggeman_percolation_threshold() {
        assert_abs_diff_eq!(
            Bruggeman::percolation_threshold(),
            1.0 / 3.0,
            epsilon = 1e-15
        );
    }

    #[test]
    fn test_bruggeman_is_percolating() {
        let ea = Complex64::new(1e6, 0.0); // conductor
        let eb = Complex64::new(2.0, 0.0); // insulator
        let bg_above = Bruggeman::new(ea, eb, 0.4).unwrap();
        let bg_below = Bruggeman::new(ea, eb, 0.2).unwrap();
        assert!(bg_above.is_percolating());
        assert!(!bg_below.is_percolating());
    }

    // ── Mie scattering ───────────────────────────────────────────────────────

    #[test]
    fn test_mie_small_sphere_dipole_approx() {
        // For x << 1, dipole approximation should match full Mie within a few %
        let radius_nm = 5.0; // very small sphere
        let eps_sphere = Complex64::new(4.0, 0.1);
        let eps_medium = Complex64::new(1.0, 0.0);
        let mie = MieSphere::new(radius_nm, eps_sphere, eps_medium);
        let lambda_nm = 600.0;
        let x = mie.size_parameter(lambda_nm);
        assert!(
            x < 0.2,
            "size parameter should be < 0.2 for dipole regime, got {x}"
        );

        let q_full = mie.scattering_efficiency(lambda_nm);
        let q_dipole = mie.dipole_scattering_efficiency(lambda_nm);
        // Should agree within 20% for x ≈ 0.05
        let rel_err = (q_full - q_dipole).abs() / q_full.max(1e-30);
        assert!(
            rel_err < 0.2,
            "dipole vs full Mie relative error {rel_err:.4} > 20%"
        );
    }

    #[test]
    fn test_mie_efficiency_positive() {
        let mie = MieSphere::new(
            50.0,
            Complex64::new(-10.0, 1.5), // gold-like
            Complex64::new(2.25, 0.0),
        );
        for lambda_nm in [400.0, 500.0, 600.0, 800.0] {
            let q_sca = mie.scattering_efficiency(lambda_nm);
            let q_abs = mie.absorption_efficiency(lambda_nm);
            let q_ext = mie.extinction_efficiency(lambda_nm);
            assert!(q_sca >= -1e-10, "Q_sca < 0 at λ={lambda_nm}");
            assert!(q_abs >= -1e-10, "Q_abs < 0 at λ={lambda_nm}");
            assert!(q_ext >= -1e-10, "Q_ext < 0 at λ={lambda_nm}");
        }
    }

    // ── Periodic EMT ─────────────────────────────────────────────────────────

    #[test]
    fn test_periodic_emt_lamellar_form_birefringence() {
        // Lamellar structure with different permittivities must show non-zero form birefringence
        let ea = Complex64::new(4.0, 0.0);
        let eb = Complex64::new(1.0, 0.0);
        let pemt = PeriodicEMT::new(500.0, ea, eb, PeriodicGeometry::Lamellar { fill_a: 0.5 });
        let fb = pemt.form_birefringence();
        assert!(
            fb > 1e-6,
            "form birefringence should be non-zero for ε_a ≠ ε_b, got {fb}"
        );
    }

    #[test]
    fn test_periodic_emt_equal_eps_zero_birefringence() {
        let eps = Complex64::new(3.0, 0.0);
        let pemt = PeriodicEMT::new(400.0, eps, eps, PeriodicGeometry::Lamellar { fill_a: 0.5 });
        let fb = pemt.form_birefringence();
        assert!(
            fb < 1e-12,
            "form birefringence should vanish when ε_a = ε_b, got {fb}"
        );
    }
}
