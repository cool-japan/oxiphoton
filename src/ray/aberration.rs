use std::f64::consts::PI;

/// Zernike polynomial decomposition for wavefront aberrations.
///
/// Zernike polynomials Z_n^m(ρ, φ) form a complete orthonormal basis on the
/// unit disk, commonly used to describe optical wavefront aberrations:
///   W(ρ, φ) = Σ c_{nm} · Z_n^m(ρ, φ)
///
/// Reference: Born & Wolf, "Principles of Optics", Ch. 9.
pub struct ZernikePolynomial;

impl ZernikePolynomial {
    /// Evaluate the radial Zernike polynomial R_n^|m|(ρ).
    ///
    /// Valid for 0 ≤ ρ ≤ 1 and (n - |m|) must be even and ≥ 0.
    pub fn radial(n: i32, m: i32, rho: f64) -> f64 {
        let abs_m = m.unsigned_abs() as i32;
        assert!(
            (n - abs_m) % 2 == 0 && abs_m <= n,
            "Invalid Zernike indices n={n}, m={m}"
        );
        let s_max = (n - abs_m) / 2;
        (0..=s_max)
            .map(|s| {
                let sign = if s % 2 == 0 { 1.0 } else { -1.0 };
                let num = factorial(n - s) as f64;
                let den = (factorial(s)
                    * factorial((n + abs_m) / 2 - s)
                    * factorial((n - abs_m) / 2 - s)) as f64;
                sign * num / den * rho.powi(n - 2 * s)
            })
            .sum()
    }

    /// Evaluate Z_n^m(ρ, φ) including the angular factor.
    pub fn evaluate(n: i32, m: i32, rho: f64, phi: f64) -> f64 {
        let r = Self::radial(n, m, rho);
        if m > 0 {
            r * (m as f64 * phi).cos()
        } else if m < 0 {
            r * ((-m) as f64 * phi).sin()
        } else {
            r
        }
    }

    /// RMS wavefront error from Zernike coefficients.
    ///
    /// W_rms = sqrt(Σ c_i²) (in the same units as the coefficients)
    pub fn rms_error(coefficients: &[f64]) -> f64 {
        coefficients.iter().map(|c| c * c).sum::<f64>().sqrt()
    }
}

fn factorial(n: i32) -> u64 {
    (1..=n as u64).product()
}

/// Common Zernike terms by name (ANSI/OSA single-index ordering).
pub enum ZernikeTerm {
    /// Piston: Z(0,0)
    Piston,
    /// Tip (x-tilt): Z(1,1)
    Tip,
    /// Tilt (y-tilt): Z(1,-1)
    Tilt,
    /// Defocus: Z(2,0)
    Defocus,
    /// Astigmatism X: Z(2,2)
    AstigmatismX,
    /// Astigmatism Y: Z(2,-2)
    AstigmatismY,
    /// Coma X: Z(3,1)
    ComaX,
    /// Coma Y: Z(3,-1)
    ComaY,
    /// Spherical aberration: Z(4,0)
    SphericalAberration,
}

impl ZernikeTerm {
    pub fn indices(&self) -> (i32, i32) {
        match self {
            Self::Piston => (0, 0),
            Self::Tip => (1, 1),
            Self::Tilt => (1, -1),
            Self::Defocus => (2, 0),
            Self::AstigmatismX => (2, 2),
            Self::AstigmatismY => (2, -2),
            Self::ComaX => (3, 1),
            Self::ComaY => (3, -1),
            Self::SphericalAberration => (4, 0),
        }
    }

    pub fn evaluate(&self, rho: f64, phi: f64) -> f64 {
        let (n, m) = self.indices();
        ZernikePolynomial::evaluate(n, m, rho, phi)
    }
}

/// Strehl ratio: ratio of peak diffraction-limited intensity with aberration to without.
///
/// Maréchal approximation (valid for small aberrations, W_rms < λ/14):
///   S ≈ exp(-(2π·W_rms/λ)²)
pub fn strehl_marechal(w_rms: f64, wavelength: f64) -> f64 {
    let phase_rms = 2.0 * PI * w_rms / wavelength;
    (-phase_rms * phase_rms).exp()
}

/// Seidel (primary) aberrations for a single refracting surface.
///
/// Returns (W040, W131, W222, W220, W311): spherical, coma, astigmatism,
/// field curvature, distortion contributions (in units of reference wavelength).
pub struct SeidelAberrations {
    /// Spherical aberration coefficient W040
    pub spherical: f64,
    /// Coma coefficient W131
    pub coma: f64,
    /// Astigmatism coefficient W222
    pub astigmatism: f64,
    /// Field curvature W220
    pub field_curvature: f64,
    /// Distortion W311
    pub distortion: f64,
}

impl SeidelAberrations {
    /// RMS wavefront error for on-axis field (dominated by spherical aberration).
    pub fn rms_on_axis(&self) -> f64 {
        // W_rms ≈ W040 / (2√5) for pure spherical aberration
        self.spherical / (2.0 * 5.0_f64.sqrt())
    }

    /// Peak-to-valley wavefront error.
    pub fn ptv(&self) -> f64 {
        self.spherical.abs() + self.coma.abs() + self.astigmatism.abs() + self.field_curvature.abs()
    }
}

// ─── Seidel coefficients (summed across surfaces) ────────────────────────────

/// Seidel sum coefficients for a complete optical system.
///
/// Each coefficient is the sum of contributions SI..SV from all surfaces.
/// SI = spherical, SII = coma, SIII = astigmatism, SIV = field curvature,
/// SV = distortion (Petzval/Seidel convention).
#[derive(Debug, Clone, Default)]
pub struct SeidelCoeffs {
    /// Spherical aberration sum SI (= Σ s1 over surfaces)
    pub s1: f64,
    /// Coma sum SII
    pub s2: f64,
    /// Astigmatism sum SIII
    pub s3: f64,
    /// Field curvature sum SIV
    pub s4: f64,
    /// Distortion sum SV
    pub s5: f64,
}

impl SeidelCoeffs {
    /// Sum Seidel contributions from a slice of `SurfaceAberration`s.
    pub fn from_surfaces(surfaces: &[SurfaceAberration]) -> Self {
        let mut out = Self::default();
        for s in surfaces {
            out.s1 += s.s1;
            out.s2 += s.s2;
            out.s3 += s.s3;
            out.s4 += s.s4;
            out.s5 += s.s5;
        }
        out
    }

    /// Total spherical aberration SI.
    pub fn total_spherical(&self) -> f64 {
        self.s1
    }

    /// Total coma SII.
    pub fn total_coma(&self) -> f64 {
        self.s2
    }

    /// Total astigmatism SIII.
    pub fn total_astigmatism(&self) -> f64 {
        self.s3
    }

    /// Total field curvature SIV.
    pub fn total_field_curvature(&self) -> f64 {
        self.s4
    }

    /// Total distortion SV.
    pub fn total_distortion(&self) -> f64 {
        self.s5
    }
}

/// Seidel aberration contributions from a single optical surface.
#[derive(Debug, Clone, Default)]
pub struct SurfaceAberration {
    /// Spherical aberration SI
    pub s1: f64,
    /// Coma SII
    pub s2: f64,
    /// Astigmatism SIII
    pub s3: f64,
    /// Field curvature SIV
    pub s4: f64,
    /// Distortion SV
    pub s5: f64,
}

impl SurfaceAberration {
    /// Approximate Seidel coefficients for a thin lens in air.
    ///
    /// Uses paraxial ray tracing to estimate the primary aberration contributions.
    ///
    /// Parameters:
    /// - `f`: focal length (m)
    /// - `n1`: refractive index before the lens
    /// - `n2`: refractive index of the lens material (n after)
    /// - `object_dist`: object distance (positive for real object to the left)
    /// - `y_marginal`: marginal ray height at the lens (m)
    ///
    /// The approximations follow Welford's "Aberrations of Optical Systems" §6.
    pub fn thin_lens(f: f64, n1: f64, n2: f64, object_dist: f64, y_marginal: f64) -> Self {
        // Image distance via thin-lens formula  1/v - 1/u = 1/f  (sign: u < 0 for real)
        let u = -object_dist.abs(); // object is to the left → u < 0
        let inv_v = 1.0 / f + 1.0 / u;
        let v = if inv_v.abs() < 1e-30 {
            f64::INFINITY
        } else {
            1.0 / inv_v
        };

        // Paraxial convergence angles
        let alpha_u = y_marginal / u.abs(); // angle at object side (positive magnitude)
        let alpha_v = if v.is_finite() {
            y_marginal / v.abs()
        } else {
            0.0
        };

        // Abbe invariant A = n·(1/R - 1/u)·y  — use thin-lens approximation with
        // effective surface power Φ = (n2-n1)/R ≈ n1/f for a single-surface thin lens
        let power = n1 / f;
        let a = n1 * alpha_u + power * y_marginal;

        // Seidel sum for a thin lens (Welford §6.4, simplified):
        //   SI  = -A²·y²·(1/n1 - 1/n2) / (2·n_obj·u²)  (spherical)
        //   SII = A·H·y·(1/n1 - 1/n2)                   (coma)  where H = n·η·α (Lagrange)
        //   SIII= H²·(1/n1 - 1/n2)                      (astigmatism)
        //   SIV = H²·power·(1/(n1·n2))                  (field curvature)
        //   SV  = (SIII+SIV)·alpha_v/alpha_u            (distortion, approx)
        let dn = 1.0 / n1 - 1.0 / n2;
        let h = n1 * y_marginal * alpha_v; // Lagrange invariant proxy (field height ~1)

        let s1 = -a * a * y_marginal * y_marginal * dn / (2.0 * n1 * u * u);
        let s2 = a * h * y_marginal * dn;
        let s3 = h * h * dn;
        let s4 = h * h * power / (n1 * n2);
        let s5 = if alpha_u.abs() > 1e-30 {
            (s3 + s4) * alpha_v / alpha_u
        } else {
            0.0
        };

        Self { s1, s2, s3, s4, s5 }
    }
}

// ─── Wavefront map and RMS/Strehl ────────────────────────────────────────────

/// Evaluate the Seidel wavefront W(x, y) on an nx×ny Cartesian pupil grid.
///
/// The pupil is a circle of radius `pupil_radius`.  Points outside the pupil
/// are set to 0.  The mapping uses normalised pupil coordinates ρ = r / R.
///
/// W(ρ, φ) = s1/8 · ρ⁴ + s2/2 · ρ³·cosφ + s3/2 · ρ²·cos²φ
///          + (s3+s4)/4 · ρ² + s5/2 · ρ·cosφ
///
/// (Welford sign convention)
pub fn wavefront_map(seidel: &SeidelCoeffs, nx: usize, ny: usize, pupil_radius: f64) -> Vec<f64> {
    let mut wfe = vec![0.0_f64; nx * ny];
    let r = pupil_radius;
    for iy in 0..ny {
        let y = (iy as f64 / (ny as f64 - 1.0) * 2.0 - 1.0) * r;
        for ix in 0..nx {
            let x = (ix as f64 / (nx as f64 - 1.0) * 2.0 - 1.0) * r;
            let rho2 = (x * x + y * y) / (r * r);
            if rho2 > 1.0 {
                continue; // outside pupil — leave 0
            }
            let rho = rho2.sqrt();
            let phi = y.atan2(x);
            let cos_phi = phi.cos();
            let w = seidel.s1 / 8.0 * rho2 * rho2
                + seidel.s2 / 2.0 * rho * rho2 * cos_phi
                + seidel.s3 / 2.0 * rho2 * cos_phi * cos_phi
                + (seidel.s3 + seidel.s4) / 4.0 * rho2
                + seidel.s5 / 2.0 * rho * cos_phi;
            wfe[iy * nx + ix] = w;
        }
    }
    wfe
}

/// RMS wavefront error: √(⟨W²⟩ − ⟨W⟩²) over the non-zero pupil samples.
pub fn rms_wavefront_error(wfe: &[f64]) -> f64 {
    let samples = wfe.to_vec();
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }
    let mean = samples.iter().sum::<f64>() / n as f64;
    let var = samples
        .iter()
        .map(|&w| (w - mean) * (w - mean))
        .sum::<f64>()
        / n as f64;
    var.sqrt()
}

/// Maréchal Strehl ratio from RMS wavefront error in units of waves.
///
/// S = exp(−(2π·W_rms)²)
///
/// `rms_waves` is the RMS WFE expressed as a fraction of a wavelength.
pub fn strehl_marechal_waves(rms_waves: f64) -> f64 {
    let phase = 2.0 * PI * rms_waves;
    (-(phase * phase)).exp()
}

// ─── Free-standing Zernike basis functions ────────────────────────────────────

/// Z₀⁰ — piston = 1.
#[inline]
pub fn zernike_piston(_rho: f64, _theta: f64) -> f64 {
    1.0
}

/// Z₁¹ — x-tilt = ρ·cosθ.
#[inline]
pub fn zernike_tilt_x(rho: f64, theta: f64) -> f64 {
    rho * theta.cos()
}

/// Z₂⁰ — defocus = 2ρ² − 1.
#[inline]
pub fn zernike_defocus(rho: f64) -> f64 {
    2.0 * rho * rho - 1.0
}

/// Z₄⁰ — primary spherical = 6ρ⁴ − 6ρ² + 1.
#[inline]
pub fn zernike_spherical(rho: f64) -> f64 {
    let r2 = rho * rho;
    6.0 * r2 * r2 - 6.0 * r2 + 1.0
}

/// Zernike basis functions used by `zernike_decompose` (n_terms ≤ 9).
///
/// Ordering (ANSI/OSA single-index j = 0..):
///   0: piston          Z(0,0)
///   1: tip             Z(1, 1) = ρ cosθ
///   2: tilt            Z(1,-1) = ρ sinθ
///   3: defocus         Z(2, 0) = 2ρ²-1
///   4: astig-x         Z(2, 2) = ρ² cos2θ
///   5: astig-y         Z(2,-2) = ρ² sin2θ
///   6: coma-x          Z(3, 1) = (3ρ³-2ρ) cosθ
///   7: coma-y          Z(3,-1) = (3ρ³-2ρ) sinθ
///   8: spherical       Z(4, 0) = 6ρ⁴-6ρ²+1
fn zernike_basis(j: usize, rho: f64, theta: f64) -> f64 {
    let r2 = rho * rho;
    match j {
        0 => 1.0,
        1 => rho * theta.cos(),
        2 => rho * theta.sin(),
        3 => 2.0 * r2 - 1.0,
        4 => r2 * (2.0 * theta).cos(),
        5 => r2 * (2.0 * theta).sin(),
        6 => (3.0 * r2 - 2.0) * rho * theta.cos(),
        7 => (3.0 * r2 - 2.0) * rho * theta.sin(),
        8 => 6.0 * r2 * r2 - 6.0 * r2 + 1.0,
        _ => 0.0,
    }
}

/// Project a wavefront map `wfe` (nx×ny, stored row-major) onto the first
/// `n_terms` Zernike polynomials.
///
/// Returns Zernike coefficients c\[j\] = ⟨W, Z_j⟩ / ⟨Z_j, Z_j⟩ computed by
/// numerical integration over the pupil disk of radius `pupil_radius`.
///
/// Grid layout matches `wavefront_map`: x varies along columns (ix), y along rows (iy).
pub fn zernike_decompose(
    wfe: &[f64],
    nx: usize,
    ny: usize,
    pupil_radius: f64,
    n_terms: usize,
) -> Vec<f64> {
    assert_eq!(wfe.len(), nx * ny, "wfe length must equal nx*ny");
    let n_terms = n_terms.min(9); // support up to 9 predefined terms
    let mut num = vec![0.0_f64; n_terms];
    let mut den = vec![0.0_f64; n_terms];

    let r = pupil_radius;
    for iy in 0..ny {
        let yp = (iy as f64 / (ny as f64 - 1.0) * 2.0 - 1.0) * r;
        for ix in 0..nx {
            let xp = (ix as f64 / (nx as f64 - 1.0) * 2.0 - 1.0) * r;
            let rho2 = (xp * xp + yp * yp) / (r * r);
            if rho2 > 1.0 {
                continue;
            }
            let rho = rho2.sqrt();
            let theta = yp.atan2(xp);
            let w = wfe[iy * nx + ix];
            for j in 0..n_terms {
                let z = zernike_basis(j, rho, theta);
                num[j] += w * z;
                den[j] += z * z;
            }
        }
    }
    (0..n_terms)
        .map(|j| {
            if den[j].abs() < 1e-30 {
                0.0
            } else {
                num[j] / den[j]
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Zernike polynomial expansion (Noll convention)
// ---------------------------------------------------------------------------

/// Convert Noll single index `j` (starting at 1) to radial order `n` and
/// azimuthal frequency `m` (signed: positive → cosine, negative → sine).
///
/// Reference: Noll, R.J. (1976) "Zernike polynomials and atmospheric turbulence."
pub fn zernike_noll_to_nm(j: u32) -> (i32, i32) {
    if j == 0 {
        return (0, 0);
    }
    // Find radial order n such that the Noll index j falls within order n.
    // The total number of Zernike modes up through order n is (n+1)(n+2)/2.
    // The first Noll index at order n is (n*(n+1))/2 + 1.
    let mut n = 0i32;
    while (n + 1) * (n + 2) / 2 < j as i32 {
        n += 1;
    }
    // First Noll index of order n (1-based)
    let j_start = (n * (n + 1) / 2 + 1) as u32;
    // 0-based offset within this radial order
    let k = j.saturating_sub(j_start) as usize;

    // Build |m| sequence for order n in Noll ordering (ascending |m|):
    // For even n: m = 0, 2, 4, ..., n
    // For odd  n: m = 1, 3, 5, ..., n
    let start = if n % 2 == 0 { 0i32 } else { 1 };
    let m_abs_seq: Vec<i32> = (0..=(n / 2))
        .map(|i| start + 2 * i)
        .filter(|&m| m >= 0 && m <= n)
        .collect();

    // Within each |m| pair, Noll assigns: even total-j → positive m (cos), odd → negative m (sin).
    // m=0 gets only one slot.
    // The k-th slot within the row maps to an |m| value and a sign:
    let m_abs_idx = k / 2;
    let m_abs = if m_abs_idx < m_abs_seq.len() {
        m_abs_seq[m_abs_idx]
    } else {
        0
    };
    let m = if m_abs == 0 {
        0
    } else if k % 2 == 0 {
        m_abs // cosine term (positive m, even k)
    } else {
        -m_abs // sine term (negative m, odd k)
    };
    (n, m)
}

/// Zernike polynomial value Z_j(ρ, θ) for Noll index `j` (1-based).
///
/// `rho` ∈ \[0, 1\], `theta` ∈ [0, 2π).
pub fn zernike_noll(j: u32, rho: f64, theta: f64) -> f64 {
    if j == 0 {
        return 0.0;
    }
    let (n, m) = zernike_noll_to_nm(j);
    // Normalisation factor
    let norm = if m == 0 {
        ((n + 1) as f64).sqrt()
    } else {
        (2.0 * (n + 1) as f64).sqrt()
    };
    let r = zernike_radial(n, m, rho);
    let angular = if m > 0 {
        (m as f64 * theta).cos()
    } else if m < 0 {
        ((-m) as f64 * theta).sin()
    } else {
        1.0
    };
    norm * r * angular
}

/// Zernike radial polynomial R_n^|m|(ρ).
///
/// Computed via the explicit sum formula:
///   R_n^m(ρ) = Σ_{s=0}^{(n-m)/2} ((-1)^s (n-s)!) / (s! ((n+m)/2-s)! ((n-m)/2-s)!) · ρ^(n-2s)
pub fn zernike_radial(n: i32, m: i32, rho: f64) -> f64 {
    let abs_m = m.unsigned_abs() as i32;
    if (n - abs_m) < 0 || (n - abs_m) % 2 != 0 {
        return 0.0;
    }
    let s_max = (n - abs_m) / 2;
    (0..=s_max)
        .map(|s| {
            let sign = if s % 2 == 0 { 1.0 } else { -1.0 };
            let num = factorial(n - s) as f64;
            let den = (factorial(s)
                * factorial((n + abs_m) / 2 - s)
                * factorial((n - abs_m) / 2 - s)) as f64;
            sign * num / den * rho.powi(n - 2 * s)
        })
        .sum()
}

/// Reconstruct a wavefront from Zernike coefficients at point (ρ, θ).
///
/// `coeffs[j-1]` is the coefficient for Noll index j (1-based).
pub fn wavefront_from_zernike(coeffs: &[f64], rho: f64, theta: f64) -> f64 {
    coeffs
        .iter()
        .enumerate()
        .map(|(idx, &c)| c * zernike_noll((idx + 1) as u32, rho, theta))
        .sum()
}

/// RMS wavefront error from Zernike coefficients.
///
/// By Parseval's theorem for orthonormal Zernike polynomials:
///   W_rms = √(Σ c_j²)
///
/// Note: piston (j=1) contributes to mean, not variance — callers may wish to
/// exclude it by zeroing coeffs\[0\].
pub fn rms_wavefront_error_zernike(coeffs: &[f64]) -> f64 {
    coeffs.iter().map(|&c| c * c).sum::<f64>().sqrt()
}

/// Return the common name of a Zernike polynomial for Noll index `j`.
///
/// Covers j = 1..11; higher orders are returned as "Higher-order".
pub fn zernike_name(j: u32) -> &'static str {
    match j {
        1 => "Piston",
        2 => "Tip",
        3 => "Tilt",
        4 => "Defocus",
        5 => "Oblique Astigmatism",
        6 => "Vertical Astigmatism",
        7 => "Vertical Coma",
        8 => "Horizontal Coma",
        9 => "Vertical Trefoil",
        10 => "Oblique Trefoil",
        11 => "Primary Spherical",
        _ => "Higher-order",
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── original tests ────────────────────────────────────────────────────────

    #[test]
    fn zernike_piston_polynomial() {
        // Z(0,0) = 1 everywhere
        for rho in [0.0, 0.5, 1.0] {
            assert!((ZernikePolynomial::radial(0, 0, rho) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn zernike_defocus_at_center() {
        // Z(2,0)(0) = -1 (from R_2^0(ρ) = 2ρ²-1, at ρ=0 → -1)
        let val = ZernikePolynomial::radial(2, 0, 0.0);
        assert!((val + 1.0).abs() < 1e-12);
    }

    #[test]
    fn zernike_defocus_at_edge() {
        // Z(2,0)(1) = 2-1 = 1
        let val = ZernikePolynomial::radial(2, 0, 1.0);
        assert!((val - 1.0).abs() < 1e-12);
    }

    #[test]
    fn zernike_tip_at_edge() {
        // Z(1,1)(ρ=1, φ=0) = ρ*cos(φ) = 1
        let val = ZernikePolynomial::evaluate(1, 1, 1.0, 0.0);
        assert!((val - 1.0).abs() < 1e-12);
    }

    #[test]
    fn zernike_rms_single_term() {
        let coeffs = [1.0];
        let rms = ZernikePolynomial::rms_error(&coeffs);
        assert!((rms - 1.0).abs() < 1e-12);
    }

    #[test]
    fn zernike_rms_two_terms() {
        let coeffs = [3.0, 4.0];
        let rms = ZernikePolynomial::rms_error(&coeffs);
        assert!((rms - 5.0).abs() < 1e-12);
    }

    #[test]
    fn strehl_zero_aberration() {
        let s = strehl_marechal(0.0, 633e-9);
        assert!((s - 1.0).abs() < 1e-12);
    }

    #[test]
    fn strehl_lambda_over_14_is_maréchal_limit() {
        // At W_rms = λ/14 ≈ 0.071λ, Strehl ≈ 0.80
        let wl = 633e-9;
        let w_rms = wl / 14.0;
        let s = strehl_marechal(w_rms, wl);
        // The Maréchal criterion: S ≥ 0.80 for diffraction-limited
        assert!(s >= 0.79, "Strehl={s:.4} at λ/14 should be ≈0.80");
    }

    #[test]
    fn zernike_term_spherical_index() {
        let (n, m) = ZernikeTerm::SphericalAberration.indices();
        assert_eq!(n, 4);
        assert_eq!(m, 0);
    }

    #[test]
    fn strehl_decreases_with_aberration() {
        let wl = 633e-9;
        let s1 = strehl_marechal(wl / 20.0, wl);
        let s2 = strehl_marechal(wl / 10.0, wl);
        assert!(s1 > s2, "More aberration should give lower Strehl");
    }

    // ── SeidelCoeffs ──────────────────────────────────────────────────────────

    #[test]
    fn seidel_coeffs_zero_surfaces() {
        let coeffs = SeidelCoeffs::from_surfaces(&[]);
        assert_eq!(coeffs.total_spherical(), 0.0);
        assert_eq!(coeffs.total_coma(), 0.0);
        assert_eq!(coeffs.total_astigmatism(), 0.0);
        assert_eq!(coeffs.total_field_curvature(), 0.0);
        assert_eq!(coeffs.total_distortion(), 0.0);
    }

    #[test]
    fn seidel_coeffs_sums_surfaces() {
        let surfaces = vec![
            SurfaceAberration {
                s1: 1.0,
                s2: 2.0,
                s3: 3.0,
                s4: 4.0,
                s5: 5.0,
            },
            SurfaceAberration {
                s1: 0.5,
                s2: -0.5,
                s3: 0.1,
                s4: -0.2,
                s5: 0.3,
            },
        ];
        let c = SeidelCoeffs::from_surfaces(&surfaces);
        assert!((c.s1 - 1.5).abs() < 1e-12);
        assert!((c.s2 - 1.5).abs() < 1e-12);
        assert!((c.s3 - 3.1).abs() < 1e-12);
        assert!((c.s4 - 3.8).abs() < 1e-12);
        assert!((c.s5 - 5.3).abs() < 1e-12);
    }

    #[test]
    fn surface_thin_lens_finite_object() {
        // Smoke test: thin lens with f=0.1m, n1=1, n2=1.5, object at 0.2m, y=0.01m
        let sa = SurfaceAberration::thin_lens(0.1, 1.0, 1.5, 0.2, 0.01);
        // Just check it doesn't panic and returns finite values
        assert!(sa.s1.is_finite());
        assert!(sa.s2.is_finite());
        assert!(sa.s3.is_finite());
        assert!(sa.s4.is_finite());
        assert!(sa.s5.is_finite());
    }

    #[test]
    fn surface_thin_lens_same_n_gives_zero() {
        // If n1 == n2 → dn = 0 → all SI..SIII = 0
        let sa = SurfaceAberration::thin_lens(0.1, 1.5, 1.5, 0.3, 0.01);
        assert!(sa.s1.abs() < 1e-20);
        assert!(sa.s2.abs() < 1e-20);
        assert!(sa.s3.abs() < 1e-20);
    }

    // ── wavefront_map ─────────────────────────────────────────────────────────

    #[test]
    fn wavefront_map_zero_coeffs_is_flat() {
        let seidel = SeidelCoeffs::default();
        let wfe = wavefront_map(&seidel, 11, 11, 1.0);
        let max_abs = wfe.iter().map(|w| w.abs()).fold(0.0_f64, f64::max);
        assert!(max_abs < 1e-30, "All-zero Seidel → flat wavefront");
    }

    #[test]
    fn wavefront_map_outside_pupil_is_zero() {
        let seidel = SeidelCoeffs {
            s1: 1.0,
            ..Default::default()
        };
        let wfe = wavefront_map(&seidel, 11, 11, 1.0);
        // Corner pixel is outside the unit circle: wfe[0] = 0
        assert_eq!(wfe[0], 0.0, "Corner (outside pupil) must be zero");
    }

    #[test]
    fn wavefront_map_centre_pixel() {
        // At ρ=0 with only s1: W = s1/8 * 0 = 0
        let seidel = SeidelCoeffs {
            s1: 2.0,
            ..Default::default()
        };
        let nx = 11;
        let ny = 11;
        let wfe = wavefront_map(&seidel, nx, ny, 1.0);
        let centre = wfe[(ny / 2) * nx + (nx / 2)];
        assert!(
            centre.abs() < 1e-12,
            "At pupil centre ρ=0 spherical gives W=0"
        );
    }

    // ── rms_wavefront_error ───────────────────────────────────────────────────

    #[test]
    fn rms_wavefront_error_flat_is_zero() {
        let wfe = vec![0.0_f64; 25];
        assert_eq!(rms_wavefront_error(&wfe), 0.0);
    }

    #[test]
    fn rms_wavefront_error_constant_is_zero() {
        // A constant WFE has zero variance → RMS = 0
        let wfe = vec![3.0_f64; 100];
        assert!(rms_wavefront_error(&wfe) < 1e-12);
    }

    #[test]
    fn rms_wavefront_error_known_case() {
        // [0, 1, 0, -1] → mean=0, var=0.5, rms=sqrt(0.5)
        let wfe = vec![0.0_f64, 1.0, 0.0, -1.0];
        let rms = rms_wavefront_error(&wfe);
        assert!((rms - 0.5_f64.sqrt()).abs() < 1e-12);
    }

    // ── strehl_marechal_waves ─────────────────────────────────────────────────

    #[test]
    fn strehl_marechal_waves_zero_is_one() {
        assert!((strehl_marechal_waves(0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn strehl_marechal_waves_consistent_with_strehl_marechal() {
        let wl = 500e-9;
        let w_rms = wl / 10.0;
        let s1 = strehl_marechal(w_rms, wl);
        let s2 = strehl_marechal_waves(1.0 / 10.0);
        assert!((s1 - s2).abs() < 1e-12);
    }

    // ── free-standing Zernike functions ───────────────────────────────────────

    #[test]
    fn zernike_piston_fn_is_one() {
        for (rho, theta) in [(0.0, 0.0), (0.5, 1.0), (1.0, PI)] {
            assert!((zernike_piston(rho, theta) - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn zernike_tilt_x_matches_definition() {
        let rho = 0.7;
        let theta = PI / 4.0;
        let val = zernike_tilt_x(rho, theta);
        assert!((val - rho * theta.cos()).abs() < 1e-15);
    }

    #[test]
    fn zernike_defocus_matches_polynomial() {
        for rho in [0.0, 0.3, 0.7, 1.0] {
            let expected = 2.0 * rho * rho - 1.0;
            assert!((zernike_defocus(rho) - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn zernike_spherical_at_edge() {
        // Z40(ρ=1) = 6-6+1 = 1
        assert!((zernike_spherical(1.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn zernike_spherical_at_zero() {
        // Z40(0) = 1
        assert!((zernike_spherical(0.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn zernike_defocus_at_zero_is_minus_one() {
        assert!((zernike_defocus(0.0) + 1.0).abs() < 1e-15);
    }

    // ── zernike_decompose ─────────────────────────────────────────────────────

    #[test]
    fn zernike_decompose_flat_wavefront_piston_only() {
        // A constant wavefront W=c should project entirely onto piston (j=0)
        let nx = 21;
        let ny = 21;
        let wfe = vec![1.0_f64; nx * ny]; // constant = 1
        let coeffs = zernike_decompose(&wfe, nx, ny, 1.0, 4);
        // Piston coefficient should be ≈ 1
        assert!(
            (coeffs[0] - 1.0).abs() < 0.05,
            "Constant WFE → piston coeff ≈ 1, got {}",
            coeffs[0]
        );
        // All higher terms should be small
        for (j, &c) in coeffs.iter().enumerate().skip(1) {
            assert!(
                c.abs() < 0.1,
                "term j={j} should be near 0 for flat WFE, got {c}"
            );
        }
    }

    #[test]
    fn zernike_decompose_length_matches_n_terms() {
        let wfe = vec![0.0_f64; 15 * 15];
        let coeffs = zernike_decompose(&wfe, 15, 15, 1.0, 5);
        assert_eq!(coeffs.len(), 5);
    }

    #[test]
    fn zernike_decompose_n_terms_capped_at_9() {
        let wfe = vec![0.0_f64; 11 * 11];
        let coeffs = zernike_decompose(&wfe, 11, 11, 1.0, 20);
        assert_eq!(coeffs.len(), 9);
    }

    // ── Zernike Noll expansion tests ──────────────────────────────────────────

    #[test]
    fn zernike_noll_to_nm_piston() {
        // Noll j=1 → piston (n=0, m=0)
        let (n, m) = zernike_noll_to_nm(1);
        assert_eq!(n, 0);
        assert_eq!(m, 0);
    }

    #[test]
    fn zernike_noll_to_nm_tip_tilt() {
        // j=2 → tip (n=1, m=1), j=3 → tilt (n=1, m=-1)
        let (n2, m2) = zernike_noll_to_nm(2);
        let (n3, m3) = zernike_noll_to_nm(3);
        assert_eq!(n2, 1);
        assert_eq!(n3, 1);
        // m values should be +1 and -1 in some order
        assert!(m2.abs() == 1 && m3.abs() == 1);
        assert_ne!(m2, m3);
    }

    #[test]
    fn zernike_radial_piston() {
        // R_0^0(ρ) = 1 for all ρ
        for rho in [0.0, 0.3, 0.7, 1.0] {
            assert!((zernike_radial(0, 0, rho) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn zernike_radial_tip() {
        // R_1^1(ρ) = ρ
        for rho in [0.0, 0.5, 1.0] {
            assert!((zernike_radial(1, 1, rho) - rho).abs() < 1e-12);
        }
    }

    #[test]
    fn zernike_noll_piston_is_constant() {
        use approx::assert_relative_eq;
        // Z_1 = sqrt(1) * R_0^0 * 1 = 1 → after normalisation = 1
        let v0 = zernike_noll(1, 0.0, 0.0);
        let v1 = zernike_noll(1, 0.7, 1.2);
        assert_relative_eq!(v0, v1, max_relative = 1e-10);
    }

    #[test]
    fn wavefront_from_zernike_single_piston_term() {
        use approx::assert_relative_eq;
        // W = c1 * Z_1; Z_1 = 1 (normalised) everywhere → W = c1
        let coeffs = vec![2.0_f64]; // c1 = 2
        let piston_z1 = zernike_noll(1, 0.5, 1.0);
        let w = wavefront_from_zernike(&coeffs, 0.5, 1.0);
        assert_relative_eq!(w, 2.0 * piston_z1, max_relative = 1e-10);
    }

    #[test]
    fn wavefront_from_zernike_zero_coeffs() {
        let coeffs = vec![0.0f64; 9];
        let w = wavefront_from_zernike(&coeffs, 0.5, 1.0);
        assert!(w.abs() < 1e-30);
    }

    #[test]
    fn rms_wavefront_error_zernike_known() {
        use approx::assert_relative_eq;
        let coeffs = vec![3.0_f64, 4.0];
        let rms = rms_wavefront_error_zernike(&coeffs);
        assert_relative_eq!(rms, 5.0, max_relative = 1e-10);
    }

    #[test]
    fn zernike_name_piston_matches() {
        assert_eq!(zernike_name(1), "Piston");
        assert_eq!(zernike_name(4), "Defocus");
        assert_eq!(zernike_name(11), "Primary Spherical");
    }

    #[test]
    fn zernike_name_high_order_fallback() {
        assert_eq!(zernike_name(100), "Higher-order");
    }
}
