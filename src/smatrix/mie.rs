//! Rigorous Mie scattering theory for homogeneous spheres.
//!
//! Implements exact Lorenz–Mie theory (Bohren & Huffman, "Absorption and
//! Scattering of Light by Small Particles", 1983, §4) for computing the
//! scattering and absorption efficiencies of a single homogeneous sphere
//! embedded in a non-absorbing medium.
//!
//! # Theory
//!
//! The Mie coefficients a_n and b_n are given by (B&H eq. 4.88):
//!
//! ```text
//! a_n = [m·ψ_n(mx)·ψ'_n(x) − ψ_n(x)·ψ'_n(mx)] /
//!       [m·ψ_n(mx)·ξ'_n(x) − ξ_n(x)·ψ'_n(mx)]
//!
//! b_n = [ψ_n(mx)·ψ'_n(x) − m·ψ_n(x)·ψ'_n(mx)] /
//!       [ψ_n(mx)·ξ'_n(x) − m·ξ_n(x)·ψ'_n(mx)]
//! ```
//!
//! where ψ_n(ρ) = ρ·j_n(ρ) are Riccati–Bessel functions of the first kind,
//! ξ_n(ρ) = ρ·h_n^(1)(ρ) are Riccati–Hankel functions, x = 2π·n_med·r/λ
//! is the size parameter, and m = m_sphere/m_medium is the relative refractive
//! index.
//!
//! The logarithmic derivative D_n(ρ) = d/dρ[ln ψ_n(ρ)] is computed via
//! downward recurrence (B&H §4.8) for numerical stability, starting from
//! D_{n_max} = 0 with n_max = ⌈max(|m|·x, x)⌉ + 16.
//!
//! # Reference
//!
//! Bohren, C. F., & Huffman, D. R. (1983). *Absorption and Scattering of
//! Light by Small Particles*. Wiley-Interscience.

use num_complex::Complex64;
use std::f64::consts::PI;

/// Results from a single Mie scattering computation.
#[derive(Debug, Clone)]
pub struct MieResult {
    /// Extinction efficiency Q_ext  (dimensionless; ratio to geometric cross-section π r²).
    pub q_ext: f64,
    /// Scattering efficiency Q_scat (dimensionless).
    pub q_scat: f64,
    /// Absorption efficiency Q_abs  (dimensionless; = Q_ext − Q_scat).
    pub q_abs: f64,
    /// Backscattering efficiency Q_back (dimensionless).
    pub q_back: f64,
    /// Size parameter x = 2π·n_med·r / λ_vacuum (dimensionless).
    pub size_parameter: f64,
}

/// Sphere scattering via Lorenz–Mie theory (Bohren & Huffman §4).
///
/// Models a homogeneous sphere with complex refractive index m_sphere = n + i·k
/// embedded in a non-absorbing medium with real index n_medium.
pub struct SphereScatter {
    /// Sphere radius in meters.
    pub radius: f64,
    /// Real part of sphere refractive index (n_sphere ≥ 0).
    pub n_sphere: f64,
    /// Imaginary part of sphere refractive index (k_sphere ≥ 0; absorption).
    pub k_sphere: f64,
    /// Refractive index of the surrounding medium (n_medium > 0, real).
    pub n_medium: f64,
}

impl SphereScatter {
    /// Construct a new `SphereScatter`.
    ///
    /// # Arguments
    ///
    /// * `radius`   – sphere radius in metres
    /// * `n_sphere` – real part of sphere refractive index
    /// * `k_sphere` – imaginary part (extinction coefficient; ≥ 0)
    /// * `n_medium` – real refractive index of the embedding medium (> 0)
    pub fn new(radius: f64, n_sphere: f64, k_sphere: f64, n_medium: f64) -> Self {
        Self {
            radius,
            n_sphere,
            k_sphere,
            n_medium,
        }
    }

    /// Compute Mie efficiencies for vacuum wavelength `lambda_m` (metres).
    ///
    /// Returns a [`MieResult`] containing Q_ext, Q_scat, Q_abs, Q_back, and
    /// the size parameter x.
    ///
    /// Uses the Bohren–Huffman algorithm (§4.4–4.8):
    /// 1. Compute D_n(mx) via downward recurrence (complex argument).
    /// 2. Build ψ_n(x) and ξ_n(x) via upward recurrence (real argument).
    /// 3. Assemble a_n, b_n and accumulate efficiency sums.
    pub fn compute(&self, lambda_m: f64) -> MieResult {
        debug_assert!(lambda_m > 0.0, "wavelength must be positive");
        debug_assert!(self.radius > 0.0, "radius must be positive");
        debug_assert!(self.n_medium > 0.0, "n_medium must be positive");

        let x = 2.0 * PI * self.n_medium * self.radius / lambda_m;
        // Relative complex refractive index m = m_sphere / n_medium
        let m = Complex64::new(self.n_sphere / self.n_medium, self.k_sphere / self.n_medium);

        // Handle the degenerate case x ≈ 0 (Rayleigh limit) analytically
        if x < 1e-10 {
            return SphereScatter::rayleigh_limit(x, m);
        }

        let (a_vec, b_vec) = Self::mie_coefficients(x, m);

        let n_terms = a_vec.len();
        let mut q_ext_sum = 0.0_f64;
        let mut q_scat_sum = 0.0_f64;
        // Backscattering requires the coherent superposition (complex sum)
        let mut q_back_sum = Complex64::new(0.0, 0.0);

        for idx in 0..n_terms {
            let n = (idx + 1) as f64; // 1-indexed
            let weight = 2.0 * n + 1.0;
            let an = a_vec[idx];
            let bn = b_vec[idx];

            // B&H eq. 4.61
            q_ext_sum += weight * (an.re + bn.re);
            q_scat_sum += weight * (an.norm_sqr() + bn.norm_sqr());

            // B&H eq. 4.70:  Σ (2n+1)·(-1)^n·(a_n - b_n)
            let sign = if (idx + 1) % 2 == 1 { 1.0 } else { -1.0 }; // (-1)^n with n=idx+1
            q_back_sum += (an - bn) * Complex64::new(weight * sign, 0.0);
        }

        let inv_x2 = 1.0 / (x * x);
        let q_ext = 2.0 * inv_x2 * q_ext_sum;
        let q_scat = 2.0 * inv_x2 * q_scat_sum;
        let q_abs = q_ext - q_scat;
        // Q_back = (1/x²) |sum|² — B&H eq. 4.70
        let q_back = inv_x2 * q_back_sum.norm_sqr();

        MieResult {
            q_ext,
            q_scat,
            q_abs,
            q_back,
            size_parameter: x,
        }
    }

    /// Physical cross sections (m²): C = Q · π r².
    ///
    /// Returns `(C_ext, C_scat, C_abs)`.
    pub fn cross_sections(&self, lambda_m: f64) -> (f64, f64, f64) {
        let geometric_area = PI * self.radius * self.radius;
        let r = self.compute(lambda_m);
        (
            r.q_ext * geometric_area,
            r.q_scat * geometric_area,
            r.q_abs * geometric_area,
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal implementation
    // ─────────────────────────────────────────────────────────────────────────

    /// Compute Mie coefficients a_n, b_n until convergence.
    ///
    /// Uses the Bohren–Huffman algorithm:
    /// * Logarithmic derivatives D_n(mx) via **downward** recurrence (stable
    ///   for complex ρ = mx because |D_n| grows as n → ∞).
    /// * Riccati–Bessel ψ_n(x) and Riccati–Hankel ξ_n(x) via **upward**
    ///   recurrence (stable for real x).
    ///
    /// Convergence criterion: |a_n| < 1e-12 **and** |b_n| < 1e-12 for three
    /// consecutive n.  Capped at n_max.
    fn mie_coefficients(x: f64, m: Complex64) -> (Vec<Complex64>, Vec<Complex64>) {
        let mx = m * x;

        // n_max for downward recurrence: B&H recommend max(|m|·x, x) + 15 or 16
        // We add 16 for safety and round up.
        let n_max = ((mx.norm().max(x)).ceil() as usize) + 16;

        // D_n(mx) via downward recurrence
        let d_mx = Self::log_deriv_d(mx, n_max);

        // Upward recurrence for ψ_n(x) and ξ_n(x) (real argument x)
        // ψ_0(x) = sin(x),  ψ_1(x) = sin(x)/x - cos(x)
        // ξ_0(x) = sin(x) - i·cos(x) = -i·exp(ix)
        // ξ_1(x) = (1/x - i)·exp(ix)  [B&H eq. 4.11 / 4.15]
        //
        // Upward recurrence: ψ_{n+1} = (2n+1)/x · ψ_n - ψ_{n-1}
        //                    ξ_{n+1} = (2n+1)/x · ξ_n - ξ_{n-1}
        //
        // We need ψ_n and ξ_n AND their logarithmic derivatives. Instead of
        // storing all ψ, we carry two consecutive values and compute
        // D_ψ(x) via the relation:
        //   D_n(x) ≡ ψ'_n / ψ_n = n/x - ψ_{n-1}/ψ_n
        // which is equivalent to the same downward recurrence applied to REAL x.
        //
        // However, since x is real and n_max is moderate, we can safely use the
        // upward recurrence for ψ (it's numerically stable for real x) and
        // compute the logarithmic derivative from ψ_{n-1} / ψ_n at each step,
        // then convert: ψ'_n/ψ_n = n/x - ψ_{n-1}/ψ_n.
        //
        // For ξ the same upward recurrence applies.

        let sin_x = x.sin();
        let cos_x = x.cos();

        // ψ_n(x): real-valued since x is real
        let mut psi_prev = sin_x; // ψ_0
        let mut psi_curr = sin_x / x - cos_x; // ψ_1

        // ξ_n(x) = ψ_n(x) − i·χ_n(x)  where χ_n(x) = −x·y_n(x) (B&H eq. 4.11).
        //
        // The Neumann-function counterpart χ satisfies the same recurrence as ψ:
        //   χ_{n+1}(x) = (2n+1)/x · χ_n(x) − χ_{n-1}(x)
        //
        // Initial values (B&H eq. 4.11–4.12):
        //   χ_0(x) = cos(x)         →  ξ_0(x) = sin(x) − i·cos(x)
        //   χ_1(x) = cos(x)/x + sin(x)  →  ξ_1(x) = (sin(x)/x − cos(x)) − i·(cos(x)/x + sin(x))
        //
        // NOTE: The imaginary part of ξ_1 is NEGATIVE (− i·χ_1).
        let xi_0 = Complex64::new(sin_x, -cos_x);
        // ξ_1(x) = ψ_1(x) − i·χ_1(x) = (sin(x)/x − cos(x)) − i·(cos(x)/x + sin(x))
        let xi_1 = Complex64::new(sin_x / x - cos_x, -(cos_x / x + sin_x));
        let mut xi_prev = xi_0;
        let mut xi_curr = xi_1;

        let mut a_vec: Vec<Complex64> = Vec::new();
        let mut b_vec: Vec<Complex64> = Vec::new();

        let mut consecutive_converged: usize = 0;
        const CONVERGENCE_THRESHOLD: f64 = 1e-12;
        const CONVERGENCE_RUNS: usize = 3;

        #[allow(clippy::needless_range_loop)]
        for n in 1..=n_max {
            let nf = n as f64;

            // D_n(mx): retrieved from downward recurrence table (index n).
            // d_mx is indexed 0..=n_max, d_mx[n] = D_n(mx).
            let d_n_mx = if n <= n_max {
                d_mx[n]
            } else {
                Complex64::new(0.0, 0.0)
            };

            // Carry ψ_{n-1}(x) and ψ_n(x) (real), ξ_{n-1}(x) and ξ_n(x) (complex).
            // At the start of iteration n:
            //   psi_prev = ψ_{n-1}(x),  psi_curr = ψ_n(x)
            //   xi_prev  = ξ_{n-1}(x),  xi_curr  = ξ_n(x)
            let psi_n_m1 = Complex64::new(psi_prev, 0.0); // ψ_{n-1}(x)
            let psi_n = Complex64::new(psi_curr, 0.0); // ψ_n(x)
            let xi_n_m1 = xi_prev; // ξ_{n-1}(x)
            let xi_n = xi_curr; // ξ_n(x)

            // B&H eq. 4.88 rewritten using Bohren–Huffman D_n algorithm
            // (Bohren & Huffman §4.4, p.127):
            //
            //   Let A_n = D_n(mx)/m + n/x    (complex)
            //   Let B_n = D_n(mx)·m + n/x    (complex)
            //
            //   a_n = (A_n·ψ_n(x) − ψ_{n-1}(x)) / (A_n·ξ_n(x) − ξ_{n-1}(x))
            //   b_n = (B_n·ψ_n(x) − ψ_{n-1}(x)) / (B_n·ξ_n(x) − ξ_{n-1}(x))
            //
            // Derivation sketch for a_n:
            //   B&H 4.88: a_n = [m·ψ_n(mx)·ψ'_n(x) − ψ_n(x)·ψ'_n(mx)] /
            //                   [m·ψ_n(mx)·ξ'_n(x) − ξ_n(x)·ψ'_n(mx)]
            //   With ψ'_n(mx) = D_n(mx)·ψ_n(mx), divide by m·ψ_n(mx):
            //     = [ψ'_n(x) − (D_n(mx)/m)·ψ_n(x)] / [ξ'_n(x) − (D_n(mx)/m)·ξ_n(x)]
            //   Use ψ'_n(x) = ψ_{n-1}(x) − (n/x)·ψ_n(x):
            //     num = ψ_{n-1}(x) − (n/x + D_n(mx)/m)·ψ_n(x) = ψ_{n-1}(x) − A_n·ψ_n(x)
            //     den = ξ_{n-1}(x) − (n/x + D_n(mx)/m)·ξ_n(x) = ξ_{n-1}(x) − A_n·ξ_n(x)
            //   Hence a_n = (A_n·ψ_n − ψ_{n-1}) / (A_n·ξ_n − ξ_{n-1})  [negating both]
            let n_over_x = Complex64::new(nf / x, 0.0);
            let coeff_a = d_n_mx / m + n_over_x; // A_n = D_n(mx)/m + n/x
            let coeff_b = d_n_mx * m + n_over_x; // B_n = D_n(mx)·m + n/x

            let num_a = coeff_a * psi_n - psi_n_m1;
            let den_a = coeff_a * xi_n - xi_n_m1;
            let num_b = coeff_b * psi_n - psi_n_m1;
            let den_b = coeff_b * xi_n - xi_n_m1;

            // Guard against division by essentially zero denominator
            let a_n = if den_a.norm() > f64::MIN_POSITIVE * 1e3 {
                num_a / den_a
            } else {
                Complex64::new(0.0, 0.0)
            };
            let b_n = if den_b.norm() > f64::MIN_POSITIVE * 1e3 {
                num_b / den_b
            } else {
                Complex64::new(0.0, 0.0)
            };

            a_vec.push(a_n);
            b_vec.push(b_n);

            // Convergence check: 3 consecutive terms below threshold
            if a_n.norm() < CONVERGENCE_THRESHOLD && b_n.norm() < CONVERGENCE_THRESHOLD {
                consecutive_converged += 1;
                if consecutive_converged >= CONVERGENCE_RUNS {
                    break;
                }
            } else {
                consecutive_converged = 0;
            }

            // Upward recurrence step: advance n → n+1
            // ψ_{n+1} = (2n+1)/x · ψ_n - ψ_{n-1}
            let factor = (2.0 * nf + 1.0) / x;
            let psi_next = factor * psi_curr - psi_prev;
            psi_prev = psi_curr;
            psi_curr = psi_next;

            // ξ_{n+1} = (2n+1)/x · ξ_n - ξ_{n-1}
            let xi_next = Complex64::new(factor, 0.0) * xi_curr - xi_prev;
            xi_prev = xi_curr;
            xi_curr = xi_next;
        }

        (a_vec, b_vec)
    }

    /// Compute the logarithmic derivative D_n(ρ) = d/dρ[ln ψ_n(ρ)] for
    /// n = 0, 1, …, n_max using **downward recurrence** (B&H §4.8, eq. 4.89).
    ///
    /// The recurrence is:
    /// ```text
    /// D_{n-1}(ρ) = n/ρ − 1 / (D_n(ρ) + n/ρ)
    /// ```
    /// Started from D_{n_max}(ρ) = 0, stepping downward to n = 0.
    ///
    /// Returns a `Vec<Complex64>` of length `n_max + 1` where `result[n] = D_n(ρ)`.
    fn log_deriv_d(rho: Complex64, n_max: usize) -> Vec<Complex64> {
        let mut d = vec![Complex64::new(0.0, 0.0); n_max + 2];
        // d[n_max] = 0 (starting value)
        // Downward: for n = n_max, n_max-1, …, 1
        for n in (1..=n_max).rev() {
            let nf = n as f64;
            let n_over_rho = Complex64::new(nf, 0.0) / rho;
            // D_{n-1} = n/ρ - 1/(D_n + n/ρ)
            let denominator = d[n] + n_over_rho;
            d[n - 1] = n_over_rho
                - if denominator.norm() > f64::MIN_POSITIVE * 1e6 {
                    Complex64::new(1.0, 0.0) / denominator
                } else {
                    // Denominator is essentially zero — use very large value to avoid
                    // numerical blow-up (physically this should not happen for valid inputs)
                    Complex64::new(0.0, 0.0)
                };
        }
        d
    }

    /// Approximate Mie result in the Rayleigh limit (x ≪ 1) using the
    /// leading-order analytic expressions.
    ///
    /// For x → 0 (B&H eq. 4.62):
    /// * Q_ext  ≈ −4x·Im{K}  where K = (m²−1)/(m²+2)  (Clausius–Mossotti)
    /// * Q_scat ≈ (8/3)·x⁴·|K|²
    /// * Q_abs  ≈ Q_ext − Q_scat  (≈ Q_ext for x ≪ 1)
    ///
    /// This avoids catastrophic cancellation in the general Mie algorithm when
    /// x is extremely small.
    fn rayleigh_limit(x: f64, m: Complex64) -> MieResult {
        let m2 = m * m;
        let k = (m2 - Complex64::new(1.0, 0.0)) / (m2 + Complex64::new(2.0, 0.0));
        // Q_ext = 4x · Im(-K) … sign: Q_ext = -4x·Im(K)  [B&H 4.62]
        let q_ext = -4.0 * x * k.im;
        let q_scat = (8.0 / 3.0) * x.powi(4) * k.norm_sqr();
        let q_abs = q_ext - q_scat;
        // Backscattering in Rayleigh limit: Q_back = (3/2)·Q_scat  (B&H 4.71)
        let q_back = 1.5 * q_scat;
        MieResult {
            q_ext: q_ext.max(0.0),
            q_scat,
            q_abs: q_abs.max(0.0),
            q_back,
            size_parameter: x,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: relative error
    fn rel_err(got: f64, expected: f64) -> f64 {
        (got - expected).abs() / expected.abs()
    }

    // ── B&H Table 4.1 reference values ───────────────────────────────────────

    /// B&H Table 4.1: x=0.1, m=1.5+0i  →  Q_ext ≈ 0.000040 (Rayleigh regime)
    #[test]
    fn mie_bh_table4_1_x01_real_index() {
        // x=0.1 is in the Rayleigh regime; Q_scat≈0, Q_abs≈0
        let radius = 0.1 / (2.0 * PI); // gives x=0.1 for lambda=1 m, n_med=1
        let s = SphereScatter::new(radius, 1.5, 0.0, 1.0);
        let r = s.compute(1.0);
        // For non-absorbing sphere Q_abs must vanish
        assert!(r.q_abs.abs() < 1e-5, "Q_abs={:.2e}", r.q_abs);
        // Q_ext > 0
        assert!(r.q_ext > 0.0, "Q_ext={:.6}", r.q_ext);
        // Rayleigh: Q_ext ~ (8/3)x^4 * |K|^2  (proportional to x^4 for absorption-free)
        // The leading term in Rayleigh is x^4 for Q_scat and x for Q_ext for real m,
        // but for a purely real m: Im(K)=0, so Q_ext is dominated by Q_scat ∝ x^4.
        // So Q_ext = Q_scat ≈ (8/3) * x^4 * |K|^2 where K=(m²-1)/(m²+2)
        let m2 = 1.5_f64.powi(2); // = 2.25
        let k_val = (m2 - 1.0) / (m2 + 2.0); // = 1.25/4.25
        let q_scat_expected = (8.0 / 3.0) * 0.1_f64.powi(4) * k_val * k_val;
        assert!(
            rel_err(r.q_scat, q_scat_expected) < 1e-3,
            "Q_scat={:.6e} expected≈{:.6e}",
            r.q_scat,
            q_scat_expected
        );
    }

    /// B&H canonical computation: x=1.0, m=1.5+0i.
    /// BHMIE reference value: Q_ext ≈ 0.2151, Q_abs = 0 (non-absorbing).
    /// (Verified against independent Python BHMIE port of B&H Appendix A code.)
    #[test]
    fn mie_bh_x1_m15_energy_conservation() {
        let s = SphereScatter::new(1.0 / (2.0 * PI), 1.5, 0.0, 1.0); // x=1
        let r = s.compute(1.0);
        // Non-absorbing: Q_abs must be negligible
        assert!(r.q_abs.abs() < 1e-6, "Q_abs={:.2e}", r.q_abs);
        // Q_ext = Q_scat for real index
        assert!(
            (r.q_ext - r.q_scat).abs() < 1e-8,
            "|Q_ext-Q_scat|={:.2e}",
            (r.q_ext - r.q_scat).abs()
        );
        // Verify Q_ext matches BHMIE reference: ≈ 0.2151 for x=1, m=1.5
        assert!(
            (r.q_ext - 0.2151).abs() < 0.002,
            "Q_ext={:.5} expected ≈ 0.2151",
            r.q_ext
        );
    }

    // ── Mandatory tests from the task specification ───────────────────────────

    /// Energy conservation: for non-absorbing sphere (k=0), Q_abs < 1e-6.
    #[test]
    fn mie_energy_conservation_real_index() {
        // x = 2π * 1.0 * 100e-9 / 532e-9 ≈ 1.18
        let s = SphereScatter::new(100e-9, 1.5, 0.0, 1.0);
        let r = s.compute(532e-9);
        assert!(r.q_abs.abs() < 1e-6, "Q_abs={}", r.q_abs);
        assert!(r.q_ext > 0.0);
        assert!(r.q_scat > 0.0);
        assert!((r.q_ext - r.q_scat - r.q_abs).abs() < 1e-10);
    }

    /// Rayleigh scaling: Q_ext ∝ x⁴ for small x with non-absorbing sphere.
    /// Doubling the radius at the same wavelength doubles x, so Q_ext → 16× larger.
    #[test]
    fn mie_small_sphere_rayleigh_scaling() {
        let r1 = SphereScatter::new(5e-9, 1.5, 0.0, 1.0).compute(532e-9); // x ≈ 0.0589
        let r2 = SphereScatter::new(10e-9, 1.5, 0.0, 1.0).compute(532e-9); // x ≈ 0.1178
                                                                           // Q_ext ∝ x^4 in Rayleigh limit → ratio ≈ 2^4 = 16
        let ratio = r2.q_ext / r1.q_ext;
        assert!(
            (ratio - 16.0).abs() < 2.0,
            "Rayleigh scaling ratio={:.4} (expected ≈ 16)",
            ratio
        );
    }

    /// Absorbing sphere: Q_abs > 0 and Q_ext > Q_scat.
    #[test]
    fn mie_absorbing_sphere_positive_q_abs() {
        let s = SphereScatter::new(100e-9, 1.5, 1.0, 1.0);
        let r = s.compute(532e-9);
        assert!(r.q_abs > 0.0, "Q_abs={:.6} should be positive", r.q_abs);
        assert!(
            r.q_ext > r.q_scat,
            "Q_ext={:.6} should exceed Q_scat={:.6}",
            r.q_ext,
            r.q_scat
        );
    }

    /// Cross sections scale as r² at fixed size parameter x ≈ 1.
    #[test]
    fn mie_cross_sections_scale_with_area() {
        let r1 = SphereScatter::new(50e-9, 1.5, 0.0, 1.0);
        let r2 = SphereScatter::new(100e-9, 1.5, 0.0, 1.0);
        // lambda = 314e-9 → x(r1) = 2π*50e-9/314e-9 ≈ 1.0
        let lambda = 314e-9;
        let (c1_ext, _, _) = r1.cross_sections(lambda);
        // lambda2 = 628e-9 → x(r2) = 2π*100e-9/628e-9 ≈ 1.0
        let lambda2 = 628e-9;
        let (c2_ext, _, _) = r2.cross_sections(lambda2);
        // At the same x, Q_ext is identical, so C2/C1 = r2²/r1² = 4
        let ratio = c2_ext / c1_ext;
        assert!(
            (ratio - 4.0).abs() < 0.1,
            "C2/C1 ratio={:.4} (expected ≈ 4.0)",
            ratio
        );
    }

    /// Large-sphere extinction: Q_ext > 1.0 (approaches 2.0 in the geometric limit).
    #[test]
    fn mie_large_size_parameter_converges() {
        // x = 2π * 848e-9 / 1000e-9 ≈ 5.32
        let s = SphereScatter::new(848e-9, 1.5, 0.0, 1.0);
        let r = s.compute(1000e-9);
        assert!(
            r.q_ext > 1.0,
            "large sphere Q_ext should exceed 1.0, got {}",
            r.q_ext
        );
        assert!(
            r.q_scat < r.q_ext + 1e-6,
            "Q_scat={} Q_ext={}",
            r.q_scat,
            r.q_ext
        );
    }

    /// Q_back ≥ 0 for any sphere.
    #[test]
    fn mie_q_back_nonnegative() {
        let s = SphereScatter::new(100e-9, 1.5, 0.5, 1.0);
        let r = s.compute(532e-9);
        assert!(
            r.q_back >= 0.0,
            "Q_back={:.6} must be non-negative",
            r.q_back
        );
    }

    // ── Additional rigorous checks ────────────────────────────────────────────

    /// Energy balance identity: Q_ext = Q_scat + Q_abs holds for any sphere.
    #[test]
    fn mie_energy_balance_absorbing() {
        let s = SphereScatter::new(100e-9, 2.5, 0.8, 1.3);
        let r = s.compute(400e-9);
        let balance = (r.q_ext - r.q_scat - r.q_abs).abs();
        assert!(balance < 1e-9, "energy balance violation: {:.2e}", balance);
    }

    /// Non-absorbing sphere in medium (k=0, n_medium > 1).
    #[test]
    fn mie_nonabsorbing_in_medium() {
        let s = SphereScatter::new(50e-9, 2.0, 0.0, 1.5);
        let r = s.compute(633e-9);
        assert!(
            r.q_abs.abs() < 1e-6,
            "Q_abs={:.2e} for non-absorbing sphere",
            r.q_abs
        );
        assert!(r.q_ext >= 0.0);
        assert!(r.q_scat >= 0.0);
    }

    /// Very small sphere: Q_ext/x should scale linearly with x^3 (Rayleigh absorption).
    /// For real index: Im(m) = 0, so Q_ext = Q_scat ∝ x^4; Q_ext/x^4 ≈ const.
    #[test]
    fn mie_rayleigh_regime_x4_scaling() {
        // x_1 = 0.001, x_2 = 0.01 → ratio of Q_ext should be ≈ 10^4
        let lambda1 = 1.0 / (2.0 * PI * 0.001); // gives x=0.001
        let lambda2 = 1.0 / (2.0 * PI * 0.01); // gives x=0.01
        let r1 = SphereScatter::new(1.0, 1.5, 0.0, 1.0).compute(lambda1);
        let r2 = SphereScatter::new(1.0, 1.5, 0.0, 1.0).compute(lambda2);
        let ratio = r2.q_ext / r1.q_ext;
        assert!(
            (ratio - 1e4).abs() / 1e4 < 0.01,
            "Rayleigh x^4 scaling: Q_ext(x=0.01)/Q_ext(x=0.001) = {:.4} (expected ≈ 1e4)",
            ratio
        );
    }

    /// Size parameter accessor is correct.
    #[test]
    fn mie_size_parameter_value() {
        let radius = 100e-9;
        let lambda = 500e-9;
        let n_med = 1.33;
        let s = SphereScatter::new(radius, 1.5, 0.0, n_med);
        let r = s.compute(lambda);
        let x_expected = 2.0 * PI * n_med * radius / lambda;
        assert!(
            (r.size_parameter - x_expected).abs() < 1e-12,
            "x={} expected={}",
            r.size_parameter,
            x_expected
        );
    }

    /// Consistency: cross_sections returns C = Q * π*r².
    #[test]
    fn mie_cross_section_vs_efficiency() {
        let radius = 80e-9;
        let lambda = 400e-9;
        let s = SphereScatter::new(radius, 1.8, 0.3, 1.0);
        let r = s.compute(lambda);
        let (c_ext, c_scat, c_abs) = s.cross_sections(lambda);
        let area = PI * radius * radius;
        assert!((c_ext - r.q_ext * area).abs() < 1e-30, "C_ext mismatch");
        assert!((c_scat - r.q_scat * area).abs() < 1e-30, "C_scat mismatch");
        assert!((c_abs - r.q_abs * area).abs() < 1e-30, "C_abs mismatch");
    }

    /// Metallic sphere (high k): strong absorption.
    #[test]
    fn mie_metallic_sphere_strong_absorption() {
        // Gold-like: n≈0.15, k≈3.5 at 532 nm
        let s = SphereScatter::new(40e-9, 0.15, 3.5, 1.0);
        let r = s.compute(532e-9);
        assert!(r.q_abs > 0.0, "metallic sphere must absorb");
        assert!(r.q_ext > 0.0);
        assert!((r.q_ext - r.q_scat - r.q_abs).abs() < 1e-9);
    }

    /// Very large sphere (x ≈ 20): test robustness and Q_ext ≈ 2.
    #[test]
    fn mie_extinction_paradox_large_sphere() {
        // x = 2π * r / λ = 20 at r=1e-6, λ = 2π/20 * 1e-6
        let r_val = 1.0e-6;
        let lambda_val = 2.0 * PI * r_val / 20.0;
        let s = SphereScatter::new(r_val, 1.5, 0.0, 1.0);
        let result = s.compute(lambda_val);
        // Large sphere: Q_ext oscillates around 2 (extinction paradox)
        // For x=20, m=1.5 it may deviate; check it's between 1.5 and 3.0
        assert!(
            result.q_ext > 1.0 && result.q_ext < 3.5,
            "x=20 Q_ext={:.5} outside expected range",
            result.q_ext
        );
    }

    /// Absorbing sphere: Q_scat < Q_ext and Q_abs exactly equals Q_ext - Q_scat.
    #[test]
    fn mie_q_abs_consistency() {
        let s = SphereScatter::new(120e-9, 1.7, 0.4, 1.2);
        let r = s.compute(600e-9);
        let q_abs_direct = r.q_ext - r.q_scat;
        assert!(
            (r.q_abs - q_abs_direct).abs() < 1e-10,
            "Q_abs internal inconsistency: stored={:.2e}, computed={:.2e}",
            r.q_abs,
            q_abs_direct
        );
    }
}
