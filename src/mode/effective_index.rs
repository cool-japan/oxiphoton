use std::f64::consts::PI;

/// Polarization for mode solving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarization {
    /// Transverse electric (E-field along x).
    TE,
    /// Transverse magnetic (H-field along x).
    TM,
}

/// Result of solving a slab waveguide mode.
#[derive(Debug, Clone)]
pub struct SlabMode {
    /// Effective index n_eff = β / k0.
    pub n_eff: f64,
    /// Mode order (0 = fundamental).
    pub order: usize,
    /// Polarization.
    pub polarization: Polarization,
}

/// Symmetric 3-layer slab waveguide effective index solver.
///
/// Uses the normalized u-w transcendental equations:
///   u = κ h/2,  w = γ h/2,  u² + w² = V²
///
/// Even TE modes: u·tan(u) = w
/// Odd TE modes:  -u·cot(u) = w
///
/// Solutions are found for u ∈ (0, V) in ascending order (= modes in
/// descending n_eff order, fundamental first).
pub struct SlabWaveguide {
    pub n_core: f64,
    pub n_clad: f64,
    pub thickness: f64,
}

impl SlabWaveguide {
    pub fn new(n_core: f64, n_clad: f64, thickness: f64) -> Self {
        assert!(n_core > n_clad, "n_core must be > n_clad for guidance");
        Self {
            n_core,
            n_clad,
            thickness,
        }
    }

    pub fn solve_te(&self, wavelength: f64) -> Vec<SlabMode> {
        self.solve_modes(wavelength, Polarization::TE)
    }

    pub fn solve_tm(&self, wavelength: f64) -> Vec<SlabMode> {
        self.solve_modes(wavelength, Polarization::TM)
    }

    fn solve_modes(&self, wavelength: f64, pol: Polarization) -> Vec<SlabMode> {
        let k0 = 2.0 * PI / wavelength;
        let h = self.thickness;
        let dn = (self.n_core * self.n_core - self.n_clad * self.n_clad).sqrt();
        let big_v = k0 * h / 2.0 * dn;

        // Polarization factor for TM modes: r = n_clad² / n_core²
        let pol_factor = match pol {
            Polarization::TE => 1.0,
            Polarization::TM => (self.n_clad / self.n_core).powi(2),
        };

        // Characteristic equations in u-space (u = κ h/2):
        // Even: f_e(u) = u·tan(u) - pol_factor * w(u) = 0, w = sqrt(V²-u²)
        // Odd:  f_o(u) = -u/tan(u) - pol_factor * w(u) = 0
        // Search: u ∈ (0, V), alternating even/odd modes

        let w = |u: f64| -> f64 { (big_v * big_v - u * u).sqrt() };

        // Find all roots in u-space in ascending order
        let mut u_roots: Vec<f64> = Vec::new();
        let n_scan = 2000;
        let du = big_v / n_scan as f64;

        // For mode of given parity in interval: skip over tan singularities
        // (which are at u = π/2, 3π/2, ... for even; u = 0, π, ... for odd)
        let f = |u: f64, is_even: bool| -> f64 {
            if u <= 0.0 || u >= big_v {
                return f64::NAN;
            }
            let wu = w(u);
            let tan_u = u.tan();
            if is_even {
                // f_e(u) = u·tan(u) - pol_factor·w  (valid in (kπ, kπ + π/2))
                u * tan_u - pol_factor * wu
            } else {
                // f_o(u) = -u·cot(u) - pol_factor·w  (valid in (kπ + π/2, (k+1)π))
                -u / tan_u - pol_factor * wu
            }
        };

        // Scan over all modes (interleaved even/odd)
        // Mode 0 is even (u in 0..π/2), mode 1 is odd (u in π/2..π), etc.
        for mode_order in 0..20 {
            let is_even = mode_order % 2 == 0;
            let half_ord = mode_order / 2;

            // Interval for this mode:
            // even mode k (0-based): u ∈ (k·π, k·π + π/2)
            // odd mode k:            u ∈ (k·π + π/2, (k+1)·π)
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
                break; // No more modes
            }
            let u_end = u_end.min(big_v - 1e-12);
            if u_end <= u_start {
                continue;
            }

            // Scan for root in this interval
            let n_sub = (((u_end - u_start) / du).ceil() as usize).max(10);
            let sub_step = (u_end - u_start) / n_sub as f64;

            let mut prev = f(u_start, is_even);
            for i in 1..=n_sub {
                let u = u_start + i as f64 * sub_step;
                let val = f(u, is_even);
                if !prev.is_nan() && !val.is_nan() && prev * val < 0.0 && (val - prev).abs() < 1e15
                // not a tan singularity
                {
                    if let Some(u_root) = bisect(|uu| f(uu, is_even), u - sub_step, u, 1e-14) {
                        u_roots.push(u_root);
                    }
                    break;
                }
                prev = val;
            }
        }

        // Convert u to n_eff and build mode list
        let mut modes: Vec<SlabMode> = u_roots
            .iter()
            .enumerate()
            .map(|(i, &u)| {
                let kappa = 2.0 * u / h;
                let beta = (self.n_core * self.n_core * k0 * k0 - kappa * kappa).sqrt();
                SlabMode {
                    n_eff: beta / k0,
                    order: i,
                    polarization: pol,
                }
            })
            .collect();

        // Sort by n_eff descending (fundamental first)
        modes.sort_by(|a, b| {
            b.n_eff
                .partial_cmp(&a.n_eff)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, m) in modes.iter_mut().enumerate() {
            m.order = i;
        }
        modes
    }
}

/// Asymmetric 3-layer slab waveguide (n_left | n_core | n_right).
pub struct AsymmetricSlab {
    pub n_left: f64,
    pub n_core: f64,
    pub n_right: f64,
    pub thickness: f64,
}

impl AsymmetricSlab {
    pub fn new(n_left: f64, n_core: f64, n_right: f64, thickness: f64) -> Self {
        assert!(
            n_core > n_left.max(n_right),
            "n_core must exceed both cladding indices"
        );
        Self {
            n_left,
            n_core,
            n_right,
            thickness,
        }
    }

    pub fn solve_te(&self, wavelength: f64) -> Vec<SlabMode> {
        self.solve_modes(wavelength, Polarization::TE)
    }

    pub fn solve_tm(&self, wavelength: f64) -> Vec<SlabMode> {
        self.solve_modes(wavelength, Polarization::TM)
    }

    fn solve_modes(&self, wavelength: f64, pol: Polarization) -> Vec<SlabMode> {
        let k0 = 2.0 * PI / wavelength;
        let h = self.thickness;
        let n1 = self.n_left;
        let n2 = self.n_core;
        let n3 = self.n_right;
        let n_max_clad = n1.max(n3);
        let beta_min = n_max_clad * k0 + 1e-10;
        let beta_max = n2 * k0 - 1e-10;

        if beta_min >= beta_max {
            return Vec::new();
        }

        // Characteristic equation for asymmetric TE slab (any order m):
        // κh = arctan(r1·γ1/κ) + arctan(r3·γ3/κ) + m·π
        // where for TE: r1 = r3 = 1; for TM: r1 = n2²/n1², r3 = n2²/n3²
        let (r1, r3) = match pol {
            Polarization::TE => (1.0, 1.0),
            Polarization::TM => (n2 * n2 / (n1 * n1), n2 * n2 / (n3 * n3)),
        };

        let char_eq = |beta: f64, m: usize| -> f64 {
            let kappa2 = n2 * n2 * k0 * k0 - beta * beta;
            let g1_2 = beta * beta - n1 * n1 * k0 * k0;
            let g3_2 = beta * beta - n3 * n3 * k0 * k0;
            if kappa2 <= 0.0 {
                return f64::NAN;
            }
            let kappa = kappa2.sqrt();
            let g1 = if g1_2 > 0.0 { g1_2.sqrt() } else { 0.0 };
            let g3 = if g3_2 > 0.0 { g3_2.sqrt() } else { 0.0 };
            kappa * h - (r1 * g1 / kappa).atan() - (r3 * g3 / kappa).atan() - m as f64 * PI
        };

        let max_modes = ((k0 * h * (n2 * n2 - n_max_clad * n_max_clad).sqrt() / PI) as usize) + 2;

        let n_scan = 1000;
        let step = (beta_max - beta_min) / n_scan as f64;

        let mut modes = Vec::new();
        for m in 0..max_modes {
            let mut prev = char_eq(beta_min, m);
            for i in 1..=n_scan {
                let beta = beta_min + i as f64 * step;
                let val = char_eq(beta, m);
                if !prev.is_nan() && !val.is_nan() && prev * val < 0.0 {
                    if let Some(beta_root) = bisect(|b| char_eq(b, m), beta - step, beta, 1e-14) {
                        modes.push(SlabMode {
                            n_eff: beta_root / k0,
                            order: 0,
                            polarization: pol,
                        });
                    }
                    break;
                }
                prev = val;
            }
        }

        // Sort by n_eff descending
        modes.sort_by(|a, b| {
            b.n_eff
                .partial_cmp(&a.n_eff)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, m) in modes.iter_mut().enumerate() {
            m.order = i;
        }
        modes
    }
}

/// Effective Index Method for 2D strip waveguide.
///
/// 2-step approximation:
/// 1. Solve vertical slab (height H) → n_eff_v
/// 2. Solve horizontal slab (width W, core = n_eff_v) → n_eff_strip
pub fn strip_waveguide_eim(
    n_core: f64,
    n_clad: f64,
    width: f64,
    height: f64,
    wavelength: f64,
    pol: Polarization,
) -> Option<f64> {
    let vert = SlabWaveguide::new(n_core, n_clad, height);
    let vert_modes = vert.solve_modes(wavelength, pol);
    let n_eff_v = vert_modes.first()?.n_eff;

    if n_eff_v <= n_clad {
        return None;
    }
    let horiz = SlabWaveguide::new(n_eff_v, n_clad, width);
    let horiz_modes = horiz.solve_modes(wavelength, pol);
    Some(horiz_modes.first()?.n_eff)
}

/// Bisection method for root finding in [lo, hi].
fn bisect<F: Fn(f64) -> f64>(f: F, mut lo: f64, mut hi: f64, tol: f64) -> Option<f64> {
    let mut f_lo = f(lo);
    let f_hi = f(hi);
    if f_lo.is_nan() || f_hi.is_nan() {
        return None;
    }
    if f_lo * f_hi > 0.0 {
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

    #[test]
    fn symmetric_slab_te0_guided() {
        let slab = SlabWaveguide::new(3.476, 1.444, 1000e-9);
        let modes = slab.solve_te(1550e-9);
        assert!(!modes.is_empty(), "Should find at least TE0");
        let n_eff = modes[0].n_eff;
        assert!(n_eff > 1.444 && n_eff < 3.476, "n_eff={n_eff} out of range");
    }

    #[test]
    fn symmetric_slab_500nm_si_te0() {
        // Si 500nm slab: n_eff should be well above cladding
        let slab = SlabWaveguide::new(3.476, 1.444, 500e-9);
        let modes = slab.solve_te(1550e-9);
        assert!(!modes.is_empty());
        let n0 = modes[0].n_eff;
        // From analytical calculation: n_eff ≈ 3.27 for V≈3.2
        assert!(n0 > 2.5 && n0 < 3.476, "TE0 n_eff={n0:.4} unexpected");
    }

    #[test]
    fn asymmetric_slab_te_guided() {
        let slab = AsymmetricSlab::new(1.444, 3.476, 1.0, 220e-9);
        let modes = slab.solve_te(1550e-9);
        assert!(!modes.is_empty(), "Should find TE0 mode");
        let n_eff = modes[0].n_eff;
        assert!(n_eff > 1.444 && n_eff < 3.476, "n_eff={n_eff}");
    }

    #[test]
    fn eim_strip_waveguide_si_1550() {
        let n_eff = strip_waveguide_eim(3.476, 1.444, 500e-9, 220e-9, 1550e-9, Polarization::TE);
        let n_eff = n_eff.expect("EIM should find a mode");
        assert!(n_eff > 1.8 && n_eff < 3.0, "EIM n_eff={n_eff:.4}");
    }

    #[test]
    fn mode_ordering_by_neff() {
        // Thick slab should have multiple modes in decreasing n_eff order
        let slab = SlabWaveguide::new(3.476, 1.444, 2000e-9);
        let modes = slab.solve_te(1550e-9);
        assert!(modes.len() >= 2, "Thick slab should have multiple modes");
        assert!(
            modes[0].n_eff > modes[1].n_eff,
            "Modes must be in descending n_eff order"
        );
    }
}
