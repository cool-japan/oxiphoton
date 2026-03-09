//! LP Mode Solver for Step-Index Optical Fiber
//!
//! Solves for Linearly Polarized (LP) modes in a step-index fiber using the
//! characteristic equation:
//!   J_{l-1}(ua)/J_l(ua) = -(w/u) * K_{l-1}(wa)/K_l(wa)
//!
//! where:
//!   u = transverse wave number in core = sqrt(k0²*n1² - β²)
//!   w = transverse decay in cladding = sqrt(β² - k0²*n2²)
//!   a = core radius
//!   β = propagation constant
//!
//! Reference: Saleh & Teich, "Fundamentals of Photonics", §8.2

use std::f64::consts::PI;

// ── Bessel function implementations ──────────────────────────────────────────

/// Bessel function of the first kind J_n(x) via series expansion.
/// Valid for all x, converges rapidly for small x; uses forward recurrence
/// for large orders.
pub fn bessel_j(n: i32, x: f64) -> f64 {
    if x == 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    // Use Miller's backward recurrence for stability
    if n < 0 {
        // J_{-n}(x) = (-1)^n * J_n(x)
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        return sign * bessel_j(-n, x);
    }
    let n_u = n as usize;
    if n_u == 0 {
        return bessel_j0(x);
    }
    if n_u == 1 {
        return bessel_j1(x);
    }
    // Forward recurrence: J_{n+1}(x) = (2n/x)*J_n(x) - J_{n-1}(x)
    // Numerically stable only for n < x; use Miller's algorithm otherwise
    if x > n as f64 {
        let mut j_prev = bessel_j0(x);
        let mut j_curr = bessel_j1(x);
        for k in 1..n_u {
            let j_next = (2.0 * k as f64 / x) * j_curr - j_prev;
            j_prev = j_curr;
            j_curr = j_next;
        }
        j_curr
    } else {
        // Miller's backward recurrence
        let start = n_u + 40;
        let mut j_next = 0.0_f64;
        let mut j_curr = 1.0e-300_f64;
        let mut result = 0.0_f64;
        let mut found = false;
        for k in (0..start).rev() {
            let j_prev_m = if k == 0 {
                0.0
            } else {
                (2.0 * k as f64 / x) * j_curr - j_next
            };
            if k == n_u && !found {
                result = j_curr;
                found = true;
            }
            j_next = j_curr;
            j_curr = j_prev_m;
        }
        // Normalisation: sum of all J_n / J_0_true
        let j0_true = bessel_j0(x);
        let j0_miller = j_next; // after the last step, j_next is J_0 from Miller
        if j0_miller.abs() < 1.0e-300 {
            return 0.0;
        }
        result * j0_true / j0_miller
    }
}

/// J_0(x) via Chebyshev polynomial approximation (Abramowitz & Stegun §9.4)
fn bessel_j0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let p1 = 57568490574.0_f64;
        let p2 = -13362590354.0_f64;
        let p3 = 651619640.7_f64;
        let p4 = -11214424.18_f64;
        let p5 = 77392.33017_f64;
        let p6 = -184.9052456_f64;
        let q1 = 57568490411.0_f64;
        let q2 = 1029532985.0_f64;
        let q3 = 9494680.718_f64;
        let q4 = 59272.64853_f64;
        let q5 = 267.8532712_f64;
        (p1 + y * (p2 + y * (p3 + y * (p4 + y * (p5 + y * p6)))))
            / (q1 + y * (q2 + y * (q3 + y * (q4 + y * (q5 + y)))))
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - std::f64::consts::FRAC_PI_4;
        let p1 = 1.0 + y * (-0.001098628627 + y * (0.000027307302 + y * (-0.000002073370)));
        let q1 = -0.01562499995 + y * (0.000143048484 + y * (-0.000006911773 + y * 0.000000073986));
        (2.0 / (PI * ax)).sqrt() * (p1 * xx.cos() - z * q1 * xx.sin())
    }
}

/// J_1(x) via Chebyshev polynomial approximation
fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    if ax < 8.0 {
        let y = x * x;
        let p1 = 72362614232.0_f64;
        let p2 = -7895059235.0_f64;
        let p3 = 242396853.1_f64;
        let p4 = -2972611.439_f64;
        let p5 = 15704.48260_f64;
        let p6 = -30.16036606_f64;
        let q1 = 144725228442.0_f64;
        let q2 = 2300535178.0_f64;
        let q3 = 18583304.74_f64;
        let q4 = 99447.43394_f64;
        let q5 = 376.9991397_f64;
        sign * x * (p1 + y * (p2 + y * (p3 + y * (p4 + y * (p5 + y * p6)))))
            / (q1 + y * (q2 + y * (q3 + y * (q4 + y * (q5 + y)))))
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356_194_491;
        let p1 = 1.0 + y * (0.00183105 + y * (-0.00031689657 + y * 0.000082376154));
        let q1 =
            0.04687499995 + y * (-0.00200269402 + y * (0.000854411190 + y * (-0.000012244755)));
        sign * (2.0 / (PI * ax)).sqrt() * (p1 * xx.cos() - z * q1 * xx.sin())
    }
}

/// Modified Bessel function of the second kind K_n(x) for x > 0.
/// Uses recurrence: K_{n+1}(x) = (2n/x)*K_n(x) + K_{n-1}(x)
pub fn bessel_k(n: i32, x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    if n < 0 {
        return bessel_k(-n, x); // K_{-n}(x) = K_n(x)
    }
    let n_u = n as usize;
    let k0_val = k0(x);
    if n_u == 0 {
        return k0_val;
    }
    let k1_val = k1(x);
    if n_u == 1 {
        return k1_val;
    }
    // Forward recurrence — stable for K_n
    let mut k_prev = k0_val;
    let mut k_curr = k1_val;
    for k in 1..n_u {
        let k_next = (2.0 * k as f64 / x) * k_curr + k_prev;
        k_prev = k_curr;
        k_curr = k_next;
    }
    k_curr
}

/// K_0(x) — Abramowitz & Stegun 9.8.5 polynomial approximation.
fn k0(x: f64) -> f64 {
    if x <= 2.0 {
        let y = x * x / 4.0;
        let i0_val = { 1.0 + y * (1.0 + y * (0.25 + y * (0.027_778 + y * 0.001_736))) };
        let poly = -0.577_215_665
            + y * (0.422_784_335
                + y * (0.230_697_561
                    + y * (0.034_885_904
                        + y * (0.002_626_980 + y * (0.000_107_502 + y * 0.000_007_400)))));
        poly - i0_val * (x / 2.0).ln()
    } else {
        let y = 2.0 / x;
        ((-x).exp() / x.sqrt())
            * (1.253_314_137
                + y * (-0.078_323_580
                    + y * (0.021_895_682
                        + y * (-0.010_624_460
                            + y * (0.005_878_720 + y * (-0.002_515_400 + y * 0.000_532_080))))))
    }
}

/// K_1(x) approximation (Abramowitz & Stegun 9.8.7).
fn k1(x: f64) -> f64 {
    if x <= 2.0 {
        let y = x * x / 4.0;
        let p = 1.0
            + y * (0.154_431_44
                + y * (-0.672_785_79
                    + y * (-0.181_568_97
                        + y * (-0.019_194_02 + y * (-0.001_104_04 + y * (-0.000_046_86))))));
        let q = 0.5
            + y * (0.878_905_94
                + y * (0.514_988_69
                    + y * (0.150_849_34
                        + y * (0.026_587_33 + y * (0.003_015_32 + y * 0.000_324_11)))));
        x * p.ln() + q / x
    } else {
        let y = 2.0 / x;
        ((-x).exp() / x.sqrt())
            * (1.253_314_14
                + y * (0.234_986_19
                    + y * (-0.036_556_20
                        + y * (0.015_042_68
                            + y * (-0.007_803_53 + y * (0.003_256_14 + y * (-0.000_682_45)))))))
    }
}

// ── LP Mode types ─────────────────────────────────────────────────────────────

/// LP mode designation for a step-index fiber.
///
/// Each LP_{l,m} mode is characterised by:
/// - `l`: azimuthal index (0, 1, 2, …)
/// - `m`: radial index (1, 2, 3, …)
#[derive(Debug, Clone, PartialEq)]
pub struct LpMode {
    /// Azimuthal mode order (l = 0 → HE11 family; l ≥ 1 → TE/TM/HE pairs)
    pub l: usize,
    /// Radial mode order (m = 1 is the fundamental for each l)
    pub m: usize,
    /// Effective refractive index n_eff = β / k0
    pub n_eff: f64,
    /// V-number at which this mode becomes guided (cutoff V)
    pub cutoff_v: f64,
}

impl LpMode {
    /// Human-readable label, e.g. LP01, LP11, LP21, LP02.
    pub fn label(&self) -> String {
        format!("LP{}{}", self.l, self.m)
    }

    /// Degeneracy count.
    ///
    /// LP01 (l=0): 2 (two orthogonal polarisations of HE11)
    /// LP_{l≥1,m}: 4 (TE0m, TM0m, HE_even, HE_odd — two orientations × two pols)
    pub fn degeneracy(&self) -> usize {
        if self.l == 0 {
            2
        } else {
            4
        }
    }

    /// Number of distinct spatial intensity patterns.
    /// LP01 → 1; LP_{l≥1} → 2 (cos and sin azimuthal variants).
    pub fn n_spatial_modes(&self) -> usize {
        if self.l == 0 {
            1
        } else {
            2
        }
    }

    /// Normalised propagation constant b = (n_eff² - n2²) / (n1² - n2²) ∈ \[0, 1\].
    pub fn normalised_b(&self, n1: f64, n2: f64) -> f64 {
        let num = self.n_eff * self.n_eff - n2 * n2;
        let den = n1 * n1 - n2 * n2;
        if den.abs() < 1.0e-15 {
            0.0
        } else {
            num / den
        }
    }
}

// ── Step-index fiber LP mode solver ──────────────────────────────────────────

/// LP mode solver for a step-index cylindrical optical fiber.
///
/// # Physical background
/// The guidance condition comes from matching the tangential fields at the
/// core–cladding boundary.  For the LP approximation, the characteristic
/// equation for mode LP_{l,m} is:
///
///   u · J_{l-1}(u) · K_l(w) + w · K_{l-1}(w) · J_l(u) = 0
///
/// where:
///   u² + w² = V²   (V = 2π·a·NA/λ is the V-number)
///   u = a · sqrt(k0²·n1² − β²)
///   w = a · sqrt(β² − k0²·n2²)
pub struct StepIndexFiberModes {
    /// Core radius in micrometres.
    pub core_radius_um: f64,
    /// Core refractive index n₁.
    pub n_core: f64,
    /// Cladding refractive index n₂.
    pub n_clad: f64,
    /// Free-space wavelength in metres.
    pub wavelength: f64,
}

impl StepIndexFiberModes {
    /// Construct a new step-index fiber mode solver.
    pub fn new(core_um: f64, n_core: f64, n_clad: f64, wavelength: f64) -> Self {
        Self {
            core_radius_um: core_um,
            n_core,
            n_clad,
            wavelength,
        }
    }

    /// V-number: V = 2π · a · NA / λ.
    pub fn v_number(&self) -> f64 {
        let a_m = self.core_radius_um * 1.0e-6;
        2.0 * PI * a_m * self.numerical_aperture() / self.wavelength
    }

    /// Numerical aperture NA = sqrt(n1² − n2²).
    pub fn numerical_aperture(&self) -> f64 {
        let diff = self.n_core * self.n_core - self.n_clad * self.n_clad;
        if diff > 0.0 {
            diff.sqrt()
        } else {
            0.0
        }
    }

    /// Approximate number of guided modes: N ≈ V² / 2.
    pub fn n_guided_modes(&self) -> usize {
        let v = self.v_number();
        ((v * v) / 2.0).ceil() as usize
    }

    /// Characteristic equation value for mode order `l` at normalised transverse
    /// parameter `u` (with w = sqrt(V²−u²)).
    ///
    /// Returns f(u) = u·J_{l-1}(u)·K_l(w) + w·K_{l-1}(w)·J_l(u)
    /// Zero-crossings give guided modes.
    fn char_eq(&self, l: usize, u: f64) -> f64 {
        let v = self.v_number();
        let w2 = v * v - u * u;
        if w2 <= 0.0 {
            return f64::NAN;
        }
        let w = w2.sqrt();
        let li = l as i32;
        // J_{l-1}(u): for l=0 → J_{-1}(u) = -J_1(u)
        let jlm1 = if l == 0 {
            -bessel_j1(u)
        } else {
            bessel_j(li - 1, u)
        };
        let jl = bessel_j(li, u);
        // K_{l-1}(w): for l=0 → K_{-1}(w) = K_1(w)
        let klm1 = if l == 0 { k1(w) } else { bessel_k(li - 1, w) };
        let kl = bessel_k(li, w);
        u * jlm1 * kl + w * klm1 * jl
    }

    /// Find all guided LP modes via bisection on the characteristic equation.
    pub fn find_modes(&self) -> Vec<LpMode> {
        let v = self.v_number();
        let mut modes = Vec::new();
        // Scan azimuthal orders l = 0, 1, 2, …
        // For LP_{l,m} the cutoff V is approximately the m-th zero of J_{l-1}.
        // We stop when the first cutoff exceeds V.
        let mut l = 0usize;
        loop {
            let root = self.find_mode_roots(l);
            if root.is_empty() {
                break;
            }
            let mut any_guided = false;
            for (m_idx, u_root) in root.iter().enumerate() {
                let m = m_idx + 1;
                let w2 = v * v - u_root * u_root;
                if w2 <= 0.0 {
                    break;
                }
                any_guided = true;
                // n_eff from normalised propagation constant b = 1 - u²/V²
                let n_eff =
                    self.n_clad + (self.n_core - self.n_clad) * (1.0 - u_root * u_root / (v * v));
                // Cutoff: u_c at which w→0, i.e., the l-th Bessel zero
                // For LP_0m: cutoff V_c ≈ zeros of J_0 except LP_01 (V_c=0)
                let cutoff_v = if l == 0 && m == 1 {
                    0.0
                } else {
                    bessel_j_zero(if l == 0 { 0 } else { l - 1 }, m)
                };
                modes.push(LpMode {
                    l,
                    m,
                    n_eff,
                    cutoff_v,
                });
            }
            if !any_guided {
                break;
            }
            l += 1;
            if l > 20 {
                break; // safety cap
            }
        }
        // Sort by n_eff descending (fundamental mode first)
        modes.sort_by(|a, b| {
            b.n_eff
                .partial_cmp(&a.n_eff)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        modes
    }

    /// Find roots of char_eq for azimuthal order `l` in u ∈ (ε, V).
    fn find_mode_roots(&self, l: usize) -> Vec<f64> {
        let v = self.v_number();
        let n_scan = 2000usize;
        let eps = 1.0e-6;
        let du = (v - eps) / n_scan as f64;
        let mut roots = Vec::new();
        let mut u_prev = eps;
        let mut f_prev = self.char_eq(l, u_prev);
        for i in 1..=n_scan {
            let u_curr = eps + i as f64 * du;
            let f_curr = self.char_eq(l, u_curr);
            // Skip NaN
            if f_prev.is_nan() || f_curr.is_nan() {
                u_prev = u_curr;
                f_prev = f_curr;
                continue;
            }
            if f_prev * f_curr < 0.0 {
                // Sign change → bisection
                if let Some(root) = bisect(|u| self.char_eq(l, u), u_prev, u_curr, 60) {
                    roots.push(root);
                }
            }
            u_prev = u_curr;
            f_prev = f_curr;
        }
        roots
    }

    /// Solve for the effective index of LP_{l,m} directly.
    /// Returns `None` if the mode is not guided at the current V number.
    pub fn n_eff_lp(&self, l: usize, m: usize) -> Option<f64> {
        let modes = self.find_modes();
        modes
            .iter()
            .find(|md| md.l == l && md.m == m)
            .map(|md| md.n_eff)
    }

    /// Mode field amplitude Ψ(r) for LP_{l,m} (azimuthally averaged, real, cos part).
    ///
    /// Ψ(r) = J_l(u·r/a)            for r < a  (core)
    ///      = J_l(u)/K_l(w) · K_l(w·r/a)  for r ≥ a  (cladding)
    pub fn mode_field(&self, mode: &LpMode, r_um: f64) -> f64 {
        let v = self.v_number();
        let a = self.core_radius_um;
        // Reconstruct u from n_eff
        let k0 = 2.0 * PI / self.wavelength;
        let a_m = a * 1.0e-6;
        let beta = mode.n_eff * k0;
        let u2 = (self.n_core * self.n_core * k0 * k0 - beta * beta) * a_m * a_m;
        let w2 = v * v - u2;
        if u2 <= 0.0 || w2 <= 0.0 {
            return 0.0;
        }
        let u = u2.sqrt();
        let w = w2.sqrt();
        let li = mode.l as i32;
        let rho = r_um / a; // normalised radius
        if rho < 1.0 {
            bessel_j(li, u * rho)
        } else {
            let norm = bessel_j(li, u) / bessel_k(li, w);
            norm * bessel_k(li, w * rho)
        }
    }

    /// Effective area A_eff = (∫|Ψ|² dA)² / ∫|Ψ|⁴ dA  \[µm²\].
    ///
    /// Numerical integration in the radial direction (azimuthal integral gives 2π).
    pub fn effective_area_um2(&self, mode: &LpMode) -> f64 {
        let a = self.core_radius_um;
        let r_max = 5.0 * a; // integrate to 5 core radii
        let n = 2000usize;
        let dr = r_max / n as f64;
        let mut int2 = 0.0_f64;
        let mut int4 = 0.0_f64;
        for i in 0..n {
            let r = (i as f64 + 0.5) * dr;
            let psi = self.mode_field(mode, r);
            let psi2 = psi * psi;
            let psi4 = psi2 * psi2;
            // dA = 2π r dr → integrate r·Ψ² and r·Ψ⁴
            int2 += r * psi2 * dr;
            int4 += r * psi4 * dr;
        }
        // A_eff = (2π·int2)² / (2π·int4)  = 2π·int2²/int4
        if int4.abs() < 1.0e-60 {
            return 0.0;
        }
        2.0 * PI * int2 * int2 / int4
    }

    /// Group index n_g = n_eff − λ · dn_eff/dλ.
    /// Computed by finite difference over ±0.1% wavelength change.
    pub fn group_index(&self, mode: &LpMode) -> f64 {
        let delta = self.wavelength * 1.0e-3;
        let solver_p = StepIndexFiberModes {
            wavelength: self.wavelength + delta,
            ..*self
        };
        let solver_m = StepIndexFiberModes {
            wavelength: self.wavelength - delta,
            ..*self
        };
        let n_p = solver_p.n_eff_lp(mode.l, mode.m).unwrap_or(mode.n_eff);
        let n_m = solver_m.n_eff_lp(mode.l, mode.m).unwrap_or(mode.n_eff);
        let dn_dl = (n_p - n_m) / (2.0 * delta);
        mode.n_eff - self.wavelength * dn_dl
    }

    /// Chromatic dispersion D \[ps/(nm·km)\] = −(λ/c) · d²n_eff/dλ².
    pub fn dispersion_ps_per_nm_km(&self, mode: &LpMode) -> f64 {
        let delta = self.wavelength * 1.0e-3;
        let solver_p = StepIndexFiberModes {
            wavelength: self.wavelength + delta,
            ..*self
        };
        let solver_m = StepIndexFiberModes {
            wavelength: self.wavelength - delta,
            ..*self
        };
        let n_p = solver_p.n_eff_lp(mode.l, mode.m).unwrap_or(mode.n_eff);
        let n_m = solver_m.n_eff_lp(mode.l, mode.m).unwrap_or(mode.n_eff);
        let d2n = (n_p - 2.0 * mode.n_eff + n_m) / (delta * delta);
        // D = -(λ/c) * d²n/dλ², convert to ps/(nm·km)
        let c = 3.0e8_f64; // m/s
        let d = -self.wavelength / c * d2n;
        // d is in s/m² → convert: 1 s/m² = 1e6 ps/(nm·km)
        d * 1.0e6
    }

    /// Inter-mode coupling coefficient κ_{12} due to periodic perturbation.
    ///
    /// κ = (k0/2) · (n1² − n2²) · ∫ Ψ₁(r) · Ψ₂(r) · r dr / sqrt(N1·N2)
    ///
    /// Phase-matched condition: perturbation period Λ = 2π / |β1 − β2|.
    pub fn coupling_coefficient(
        &self,
        mode1: &LpMode,
        mode2: &LpMode,
        perturbation_period_mm: f64,
    ) -> f64 {
        let a = self.core_radius_um;
        let r_max = 3.0 * a;
        let n = 1000usize;
        let dr = r_max / n as f64;

        let mut overlap = 0.0_f64;
        let mut norm1 = 0.0_f64;
        let mut norm2 = 0.0_f64;
        for i in 0..n {
            let r = (i as f64 + 0.5) * dr;
            let psi1 = self.mode_field(mode1, r);
            let psi2 = self.mode_field(mode2, r);
            overlap += r * psi1 * psi2 * dr;
            norm1 += r * psi1 * psi1 * dr;
            norm2 += r * psi2 * psi2 * dr;
        }
        let denom = (norm1 * norm2).sqrt();
        if denom < 1.0e-30 {
            return 0.0;
        }
        let k0 = 2.0 * PI / self.wavelength;
        let dn2 = self.n_core * self.n_core - self.n_clad * self.n_clad;
        let kappa_raw = (k0 / 2.0) * dn2 * overlap / denom;
        // Phase-matching correction: sinc factor for period mismatch
        let delta_beta = (mode1.n_eff - mode2.n_eff) * k0;
        let lambda_m = perturbation_period_mm * 1.0e-3;
        let delta_beta_eff = delta_beta - 2.0 * PI / lambda_m;
        let phase_factor = if delta_beta_eff.abs() < 1.0e-6 {
            1.0
        } else {
            (delta_beta_eff / 2.0).sin() / (delta_beta_eff / 2.0)
        };
        kappa_raw * phase_factor.abs()
    }
}

// ── Numerical utilities ───────────────────────────────────────────────────────

/// Bisection root-finder for f on \[a, b\] to within relative tolerance 1e-10.
/// Returns None if no sign change or NaN encountered.
fn bisect<F: Fn(f64) -> f64>(f: F, mut a: f64, mut b: f64, max_iter: usize) -> Option<f64> {
    let fa = f(a);
    let fb = f(b);
    if fa.is_nan() || fb.is_nan() {
        return None;
    }
    if fa * fb > 0.0 {
        return None;
    }
    for _ in 0..max_iter {
        let mid = (a + b) / 2.0;
        let fm = f(mid);
        if fm.is_nan() {
            return None;
        }
        if (b - a).abs() < 1.0e-12 * (a.abs() + b.abs() + 1.0e-30) {
            return Some(mid);
        }
        if fa * fm <= 0.0 {
            b = mid;
        } else {
            a = mid;
        }
    }
    Some((a + b) / 2.0)
}

/// Approximate zeros of J_{l-1}(x) for the m-th root.
/// Used to estimate mode cutoff V numbers.
fn bessel_j_zero(l: usize, m: usize) -> f64 {
    // Tabulated first few zeros of J_l for l = 0..4
    // j_{l,m}: m-th positive zero of J_l(x)
    const ZEROS: [[f64; 5]; 5] = [
        // J_0
        [2.4048, 5.5201, 8.6537, 11.7915, 14.9309],
        // J_1
        [3.8317, 7.0156, 10.1735, 13.3237, 16.4706],
        // J_2
        [5.1356, 8.4172, 11.6198, 14.7960, 17.9598],
        // J_3
        [6.3802, 9.7610, 13.0152, 16.2235, 19.4094],
        // J_4
        [7.5883, 11.0647, 14.3725, 17.6160, 20.8269],
    ];
    if l < 5 && (1..=5).contains(&m) {
        ZEROS[l][m - 1]
    } else {
        // McMahon's asymptotic formula: j_{l,m} ≈ β_m - (μ-1)/(8*β_m)
        // where β_m = (m + l/2 - 1/4)*π, μ = 4l²
        let beta = (m as f64 + l as f64 / 2.0 - 0.25) * PI;
        let mu = 4.0 * (l * l) as f64;
        beta - (mu - 1.0) / (8.0 * beta)
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_v_number_smf28() {
        // Standard SMF-28: a=4.1µm, n1=1.4681, n2=1.4629, λ=1.55µm
        let fiber = StepIndexFiberModes::new(4.1, 1.4681, 1.4629, 1.55e-6);
        let v = fiber.v_number();
        // Expected V ≈ 2.1 (single-mode region)
        assert!(v > 1.5 && v < 3.0, "V-number should be ~2.1, got {}", v);
    }

    #[test]
    fn test_na_standard_fiber() {
        // NA = sqrt(1.4681² - 1.4629²)
        let fiber = StepIndexFiberModes::new(4.1, 1.4681, 1.4629, 1.55e-6);
        let na = fiber.numerical_aperture();
        // NA ≈ 0.123 for SMF-28
        assert!(na > 0.10 && na < 0.15, "NA should be ~0.123, got {}", na);
    }

    #[test]
    fn test_lp_mode_label() {
        let mode = LpMode {
            l: 0,
            m: 1,
            n_eff: 1.46,
            cutoff_v: 0.0,
        };
        assert_eq!(mode.label(), "LP01");
        let mode2 = LpMode {
            l: 1,
            m: 1,
            n_eff: 1.45,
            cutoff_v: 2.405,
        };
        assert_eq!(mode2.label(), "LP11");
    }

    #[test]
    fn test_lp_mode_degeneracy() {
        let lp01 = LpMode {
            l: 0,
            m: 1,
            n_eff: 1.46,
            cutoff_v: 0.0,
        };
        assert_eq!(lp01.degeneracy(), 2);
        let lp11 = LpMode {
            l: 1,
            m: 1,
            n_eff: 1.45,
            cutoff_v: 2.405,
        };
        assert_eq!(lp11.degeneracy(), 4);
        let lp21 = LpMode {
            l: 2,
            m: 1,
            n_eff: 1.44,
            cutoff_v: 3.832,
        };
        assert_eq!(lp21.degeneracy(), 4);
    }

    #[test]
    fn test_find_modes_single_mode_fiber() {
        // V < 2.405 → only LP01 guided
        let fiber = StepIndexFiberModes::new(4.1, 1.4681, 1.4629, 1.55e-6);
        let v = fiber.v_number();
        assert!(v < 2.405, "V={} should be below LP11 cutoff", v);
        let modes = fiber.find_modes();
        let lp01_count = modes.iter().filter(|m| m.l == 0 && m.m == 1).count();
        assert_eq!(lp01_count, 1, "SMF should have exactly one LP01 mode");
    }

    #[test]
    fn test_find_modes_few_mode_fiber() {
        // Larger core → few modes (V ~ 5)
        let fiber = StepIndexFiberModes::new(10.0, 1.455, 1.444, 1.55e-6);
        let v = fiber.v_number();
        assert!(
            v > 2.405,
            "Should have V > 2.405 for few modes, got V={}",
            v
        );
        let modes = fiber.find_modes();
        assert!(!modes.is_empty(), "Should find at least one mode");
        // LP01 should be the first (highest n_eff)
        assert_eq!(modes[0].l, 0);
        assert_eq!(modes[0].m, 1);
    }

    #[test]
    fn test_bessel_j0_values() {
        // J_0(0) = 1, J_0(2.4048) ≈ 0
        assert_abs_diff_eq!(bessel_j0(0.0), 1.0, epsilon = 1.0e-6);
        assert_abs_diff_eq!(bessel_j0(2.4048), 0.0, epsilon = 1.0e-3);
    }

    #[test]
    fn test_bessel_k_positive() {
        // K_0(1) ≈ 0.4210, K_1(1) ≈ 0.6019
        let k0_val = bessel_k(0, 1.0);
        let k1_val = bessel_k(1, 1.0);
        assert!(
            k0_val > 0.3 && k0_val < 0.55,
            "K_0(1) ≈ 0.421, got {}",
            k0_val
        );
        assert!(
            k1_val > 0.5 && k1_val < 0.75,
            "K_1(1) ≈ 0.602, got {}",
            k1_val
        );
    }

    #[test]
    fn test_effective_area_smf() {
        // SMF-28: A_eff typically 80–90 µm²
        let fiber = StepIndexFiberModes::new(4.1, 1.4681, 1.4629, 1.55e-6);
        let modes = fiber.find_modes();
        assert!(!modes.is_empty());
        let lp01 = modes.iter().find(|m| m.l == 0 && m.m == 1);
        if let Some(mode) = lp01 {
            let aeff = fiber.effective_area_um2(mode);
            assert!(
                aeff > 10.0 && aeff < 300.0,
                "A_eff should be realistic, got {} µm²",
                aeff
            );
        }
    }
}
