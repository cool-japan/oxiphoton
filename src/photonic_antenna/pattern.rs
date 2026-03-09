//! Antenna radiation pattern analysis.
//!
//! Provides:
//! - Numerical directivity computation from a 2D pattern function
//! - Pattern metrics: HPBW, SLL, front-to-back ratio
//! - Effective aperture area from gain
//! - Friis transmission equation
//!
//! All angles in radians internally.

use std::f64::consts::PI;

// ─── directivity_from_pattern ─────────────────────────────────────────────────

/// Compute directivity D from a 2D radiation pattern function U(θ, φ).
///
/// Uses numerical integration over the sphere (rectangular quadrature):
///
///   D = 4π U_max / P_rad
///
/// where:
///
///   P_rad = ∫₀^π ∫₀^{2π} U(θ,φ) sin θ dθ dφ
///
/// * `pattern_fn` — radiation intensity U(θ, φ) (any units; can be normalised)
/// * `n_theta`    — number of quadrature points in θ ∈ (0, π)
/// * `n_phi`      — number of quadrature points in φ ∈ [0, 2π)
///
/// Returns the directivity as a dimensionless linear number (not in dBi).
pub fn directivity_from_pattern(
    pattern_fn: impl Fn(f64, f64) -> f64,
    n_theta: usize,
    n_phi: usize,
) -> f64 {
    if n_theta == 0 || n_phi == 0 {
        return 0.0;
    }

    let mut p_rad = 0.0_f64;
    let mut u_max = 0.0_f64;

    let dtheta = PI / n_theta as f64;
    let dphi = 2.0 * PI / n_phi as f64;

    for it in 0..n_theta {
        let theta = PI * (it as f64 + 0.5) / n_theta as f64;
        let sin_theta = theta.sin();
        for ip in 0..n_phi {
            let phi = 2.0 * PI * ip as f64 / n_phi as f64;
            let u = pattern_fn(theta, phi);
            if u > u_max {
                u_max = u;
            }
            p_rad += u * sin_theta * dtheta * dphi;
        }
    }

    if p_rad < f64::EPSILON {
        return 0.0;
    }
    4.0 * PI * u_max / p_rad
}

/// Directivity in dBi from a 2D pattern function.
pub fn directivity_dbi_from_pattern(
    pattern_fn: impl Fn(f64, f64) -> f64,
    n_theta: usize,
    n_phi: usize,
) -> f64 {
    let d = directivity_from_pattern(pattern_fn, n_theta, n_phi);
    if d < f64::MIN_POSITIVE {
        return f64::NEG_INFINITY;
    }
    10.0 * d.log10()
}

// ─── AntennaPatternMetrics ────────────────────────────────────────────────────

/// Aggregate metrics extracted from a sampled 1D antenna pattern cut.
///
/// All angle values are in degrees.
#[derive(Debug, Clone)]
pub struct AntennaPatternMetrics {
    /// Directivity at boresight in dBi
    pub directivity_dbi: f64,
    /// HPBW in the E-plane (degrees)
    pub hpbw_e_deg: f64,
    /// HPBW in the H-plane (degrees) — set to same as E if only 1D available
    pub hpbw_h_deg: f64,
    /// Side-lobe level relative to main lobe peak (dB, ≤ 0)
    pub side_lobe_level_db: f64,
    /// Front-to-back ratio (dB, ≥ 0)
    pub front_to_back_ratio_db: f64,
}

impl AntennaPatternMetrics {
    /// Compute metrics from a sampled 1D pattern array.
    ///
    /// * `pattern`         — slice of intensity samples (linear, not dB), length ≥ 2
    /// * `theta_deg_range` — (θ_start, θ_end) spanning the pattern in degrees
    ///
    /// The pattern is assumed to have its maximum near the centre.
    pub fn from_pattern_1d(pattern: &[f64], theta_deg_range: (f64, f64)) -> Self {
        let n = pattern.len();
        if n < 2 {
            return Self {
                directivity_dbi: 0.0,
                hpbw_e_deg: 180.0,
                hpbw_h_deg: 180.0,
                side_lobe_level_db: 0.0,
                front_to_back_ratio_db: 0.0,
            };
        }

        let theta_start = theta_deg_range.0;
        let theta_end = theta_deg_range.1;
        let theta_span = theta_end - theta_start;

        // Find global maximum
        let (peak_idx, &peak_val) = pattern
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((n / 2, &1.0));
        let peak_val = peak_val.max(f64::MIN_POSITIVE);

        // Angle of each sample
        let angle_of = |idx: usize| theta_start + theta_span * idx as f64 / (n - 1) as f64;

        // HPBW: find -3 dB (0.5 power) crossing on each side of the peak
        let half_power = peak_val * 0.5;

        // Left half-power crossing
        let left_idx = (0..peak_idx)
            .rev()
            .find(|&i| pattern[i] <= half_power)
            .unwrap_or(0);
        // Right half-power crossing
        let right_idx = ((peak_idx + 1)..n)
            .find(|&i| pattern[i] <= half_power)
            .unwrap_or(n - 1);

        let hpbw = (angle_of(right_idx) - angle_of(left_idx)).abs().max(0.0);

        // Side-lobe level: highest local maximum outside HPBW region
        let sll_val = find_side_lobe_level(pattern, left_idx, right_idx, peak_val);
        let side_lobe_level_db = if sll_val > f64::MIN_POSITIVE {
            10.0 * (sll_val / peak_val).log10()
        } else {
            -60.0 // floor
        };

        // Front-to-back: compare peak to pattern at ±180° (back direction)
        let back_idx = ((peak_idx + n / 2) % n).min(n - 1);
        let back_val = pattern[back_idx].max(f64::MIN_POSITIVE);
        let front_to_back_ratio_db = 10.0 * (peak_val / back_val).log10();

        // Directivity: use numerical integration over the 1D cut (approximation
        // assuming azimuthal symmetry → solid angle weight sin θ)
        let directivity_linear = compute_directivity_1d(pattern, theta_deg_range);
        let directivity_dbi = if directivity_linear > f64::MIN_POSITIVE {
            10.0 * directivity_linear.log10()
        } else {
            0.0
        };

        Self {
            directivity_dbi,
            hpbw_e_deg: hpbw,
            hpbw_h_deg: hpbw, // same cut assumed for both planes
            side_lobe_level_db,
            front_to_back_ratio_db: front_to_back_ratio_db.max(0.0),
        }
    }

    /// Boresight gain in dBi (alias for directivity_dbi, assuming η = 1)
    pub fn boresight_gain_dbi(&self) -> f64 {
        self.directivity_dbi
    }

    /// True if the pattern has no significant sidelobes (SLL < threshold).
    pub fn is_low_sidelobe(&self, threshold_db: f64) -> bool {
        self.side_lobe_level_db < threshold_db
    }
}

/// Find the peak value of sidelobes outside the main-lobe region [left, right].
fn find_side_lobe_level(pattern: &[f64], left: usize, right: usize, _peak: f64) -> f64 {
    let n = pattern.len();
    let mut max_side = 0.0_f64;

    // Local maxima to the left of the HPBW region
    for i in 1..left.min(n - 1) {
        if pattern[i] >= pattern[i - 1]
            && pattern[i] >= pattern[(i + 1).min(n - 1)]
            && pattern[i] > max_side
        {
            max_side = pattern[i];
        }
    }
    // Local maxima to the right
    for i in (right + 1).min(n - 1)..n - 1 {
        if pattern[i] >= pattern[i - 1] && pattern[i] >= pattern[i + 1] && pattern[i] > max_side {
            max_side = pattern[i];
        }
    }

    max_side
}

/// Approximate directivity from a 1D pattern cut, assuming azimuthal symmetry.
///
///   D = 2 U_max / ∫₀^π U(θ) sin θ dθ
fn compute_directivity_1d(pattern: &[f64], theta_deg_range: (f64, f64)) -> f64 {
    let n = pattern.len();
    if n < 2 {
        return 1.0;
    }
    let theta_start = theta_deg_range.0.to_radians();
    let theta_end = theta_deg_range.1.to_radians();
    let dtheta = (theta_end - theta_start) / (n - 1) as f64;

    let mut integral = 0.0_f64;
    let mut u_max = 0.0_f64;

    for (i, &u) in pattern.iter().enumerate() {
        let theta = theta_start + i as f64 * dtheta;
        integral += u * theta.sin() * dtheta;
        if u > u_max {
            u_max = u;
        }
    }

    if integral < f64::EPSILON {
        return 1.0;
    }
    2.0 * u_max / integral
}

// ─── Utility functions ────────────────────────────────────────────────────────

/// Effective aperture area from gain: A_eff = G λ² / (4π)
///
/// * `gain_linear`   — antenna gain (linear, not dB)
/// * `wavelength_m`  — free-space wavelength (m)
pub fn effective_aperture_m2(gain_linear: f64, wavelength_m: f64) -> f64 {
    gain_linear * wavelength_m * wavelength_m / (4.0 * PI)
}

/// Friis free-space transmission equation.
///
///   P_r / P_t = G_t G_r (λ / 4π R)²
///
/// * `gain_t`       — transmit antenna gain (linear)
/// * `gain_r`       — receive antenna gain (linear)
/// * `wavelength_m` — free-space wavelength (m)
/// * `range_m`      — link distance (m)
///
/// Returns the power ratio P_r / P_t (dimensionless, ≤ 1 in the far field).
pub fn friis_equation(gain_t: f64, gain_r: f64, wavelength_m: f64, range_m: f64) -> f64 {
    if range_m < f64::MIN_POSITIVE {
        return 0.0;
    }
    gain_t * gain_r * (wavelength_m / (4.0 * PI * range_m)).powi(2)
}

/// Free-space path loss in dB: FSPL = −10 log₁₀( (λ/4πR)² ).
pub fn free_space_path_loss_db(wavelength_m: f64, range_m: f64) -> f64 {
    if range_m < f64::MIN_POSITIVE || wavelength_m < f64::MIN_POSITIVE {
        return f64::INFINITY;
    }
    -20.0 * (wavelength_m / (4.0 * PI * range_m)).log10()
}

/// Convert linear gain to dBi.
pub fn gain_linear_to_dbi(gain_linear: f64) -> f64 {
    if gain_linear <= 0.0 {
        return f64::NEG_INFINITY;
    }
    10.0 * gain_linear.log10()
}

/// Convert dBi to linear gain.
pub fn gain_dbi_to_linear(gain_dbi: f64) -> f64 {
    10.0_f64.powf(gain_dbi / 10.0)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hertzian_dipole_directivity_numeric() {
        // Hertzian dipole pattern: U(θ,φ) = sin²θ → D = 1.5
        let d = directivity_from_pattern(|theta, _phi| theta.sin().powi(2), 500, 200);
        assert!(
            (d - 1.5).abs() < 0.02,
            "Dipole directivity: {d:.4} expected ~1.5"
        );
    }

    #[test]
    fn isotropic_radiator_directivity_unity() {
        // Isotropic: U = 1 → D = 1
        let d = directivity_from_pattern(|_theta, _phi| 1.0, 200, 100);
        assert!((d - 1.0).abs() < 0.01, "Isotropic directivity: {d:.4}");
    }

    #[test]
    fn friis_free_space_path_loss() {
        // λ=1550 nm, R=100 m, G_t=G_r=1 (isotropic antennas)
        // FSPL = 20 log10(4π R / λ) = 20 log10(4π × 100 / 1.55e-6) ≈ 178 dB
        let p_ratio = friis_equation(1.0, 1.0, 1550e-9, 100.0);
        let fspl_db = -10.0 * p_ratio.log10();
        assert!(
            fspl_db > 165.0 && fspl_db < 190.0,
            "FSPL = {fspl_db:.1} dB (expected ~178 dB)"
        );
    }

    #[test]
    fn effective_aperture_isotropic() {
        // Isotropic antenna (G=1) at 1550 nm
        let a = effective_aperture_m2(1.0, 1550.0e-9);
        // A = λ²/4π ≈ 3.01e-13 m²
        assert!(a > 1.0e-14 && a < 1.0e-12, "Isotropic aperture: {a:.3e} m²");
    }

    #[test]
    fn fspl_increases_with_range() {
        let loss1 = free_space_path_loss_db(1550.0e-9, 10.0);
        let loss2 = free_space_path_loss_db(1550.0e-9, 100.0);
        // Doubling R by ×10 increases FSPL by 20 dB
        assert!(
            loss2 > loss1,
            "FSPL must increase with range: {loss1:.1} → {loss2:.1}"
        );
        assert!(
            (loss2 - loss1 - 20.0).abs() < 0.01,
            "10× range → +20 dB: diff={:.3}",
            loss2 - loss1
        );
    }

    #[test]
    fn pattern_metrics_from_sinc_like_array() {
        // Create a sinc-like pattern: cos²(10 θ) × cos²(θ/2) in [−π/2, π/2]
        let n = 1001_usize;
        let pattern: Vec<f64> = (0..n)
            .map(|i| {
                let t = -PI / 2.0 + PI * i as f64 / (n - 1) as f64;
                (10.0 * t).cos().powi(2) * (t / 2.0).cos().powi(2)
            })
            .collect();
        let metrics = AntennaPatternMetrics::from_pattern_1d(&pattern, (-90.0, 90.0));
        // Main lobe HPBW should be narrow (< 15°)
        assert!(
            metrics.hpbw_e_deg < 15.0,
            "HPBW should be < 15°: {:.2}°",
            metrics.hpbw_e_deg
        );
        // SLL should be negative dB
        assert!(
            metrics.side_lobe_level_db < 0.0,
            "SLL must be negative: {:.2} dB",
            metrics.side_lobe_level_db
        );
    }

    #[test]
    fn gain_conversion_round_trip() {
        let g_dbi = 12.0_f64;
        let g_lin = gain_dbi_to_linear(g_dbi);
        let g_dbi2 = gain_linear_to_dbi(g_lin);
        assert!(
            (g_dbi2 - g_dbi).abs() < 1.0e-10,
            "Round-trip conversion: {g_dbi2}"
        );
    }

    #[test]
    fn directivity_dbi_positive_for_dipole() {
        let d_dbi = directivity_dbi_from_pattern(|theta, _phi| theta.sin().powi(2), 200, 100);
        // Dipole D = 1.5 → 1.76 dBi
        assert!(d_dbi > 1.5 && d_dbi < 2.0, "Dipole dBi: {d_dbi:.3}");
    }
}
