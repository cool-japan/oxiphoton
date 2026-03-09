//! ISO 11146 beam quality measurement protocol.
//!
//! ISO 11146-1:2021 (*Lasers and laser-related equipment — Test methods for laser beam
//! widths, divergence angles and beam propagation ratios*) specifies how to measure
//! the M² factor of a real laser beam from a set of second-moment width measurements
//! taken at different axial positions.
//!
//! Key requirements of the standard:
//! * Use the **D4σ** (second-moment, 1/e² diameter) width definition.
//! * At least **5 measurement planes** within one Rayleigh length on either side of the
//!   beam waist *and* at least **5 planes** in the far field (|z − z₀| ≥ 2 zR).
//! * Fit a parabola to the measured w²(z) values to extract M², w₀, and z₀.
//!
//! # Example
//! ```rust,ignore
//! use oxiphoton::beam_quality::iso_standard::{Iso11146Measurement, synthetic_gaussian_caustic};
//!
//! let lambda = 1064e-9;
//! let points = synthetic_gaussian_caustic(lambda, 1.2, 0.5e-3, 0.0, -0.3, 0.3, 20);
//! let mut meas = Iso11146Measurement::new(lambda);
//! for (z, w) in &points {
//!     meas.add_point(*z, 2.0 * w, 2.0 * w);  // D4σ = 2w
//! }
//! let result = meas.analyze().expect("ISO analysis");
//! println!("M²_x = {:.3}", result.m2_x);
//! ```

use std::f64::consts::PI;

use super::caustic::{fit_parabola, BeamCaustic, BeamMeasurement};
use super::m2_factor::BeamQuality;

// ─── Measurement container ────────────────────────────────────────────────────

/// ISO 11146 beam quality measurement session.
///
/// Input widths should be **D4σ diameters** (= 4 × second-moment radius = 2 × ISO w).
/// Internally the diameters are halved to convert to radii before fitting.
#[derive(Debug, Clone)]
pub struct Iso11146Measurement {
    /// Vacuum wavelength (m).
    pub wavelength_m: f64,
    /// Raw measurements stored as [`BeamMeasurement`] (radii, not diameters).
    pub measurements: Vec<BeamMeasurement>,
}

impl Iso11146Measurement {
    /// Create a new, empty measurement session for the given wavelength.
    pub fn new(wavelength_m: f64) -> Self {
        Self {
            wavelength_m,
            measurements: Vec::new(),
        }
    }

    /// Add a measurement point.
    ///
    /// # Arguments
    /// * `z_m` – Axial position (m).
    /// * `d4sigma_x_m` – D4σ beam diameter in x (m) = 4 × second-moment width.
    /// * `d4sigma_y_m` – D4σ beam diameter in y (m).
    pub fn add_point(&mut self, z_m: f64, d4sigma_x_m: f64, d4sigma_y_m: f64) {
        // ISO 11146: radii = D4σ / 2
        self.measurements.push(BeamMeasurement {
            z_m,
            radius_x_m: d4sigma_x_m / 2.0,
            radius_y_m: d4sigma_y_m / 2.0,
        });
    }

    /// Check whether the measurement set satisfies the ISO 11146 sampling requirements.
    ///
    /// The check estimates the Rayleigh range from a preliminary parabola fit and then
    /// verifies that ≥ 5 points lie within [z₀ − zR, z₀ + zR] and ≥ 5 outside that
    /// interval in the far field.
    ///
    /// Returns `true` if the requirements are met.
    pub fn check_sampling(&self) -> bool {
        if self.measurements.len() < 10 {
            return false;
        }
        let caustic = self.to_caustic();

        // Preliminary x-axis fit to estimate z₀ and zR
        let bq = match caustic.extract_beam_quality_x() {
            Some(b) => b,
            None => return false,
        };
        let z0 = bq.waist_position_m;
        let zr = bq.rayleigh_range_m();

        let near: usize = self
            .measurements
            .iter()
            .filter(|m| (m.z_m - z0).abs() <= zr)
            .count();
        let far: usize = self
            .measurements
            .iter()
            .filter(|m| (m.z_m - z0).abs() >= 2.0 * zr)
            .count();

        near >= 5 && far >= 5
    }

    /// Run the full ISO 11146 analysis.
    ///
    /// Fits parabolas to both x and y caustics, extracts beam quality parameters,
    /// and returns an [`Iso11146Result`].
    ///
    /// Returns `None` if the parabola fit fails for either axis.
    pub fn analyze(&self) -> Option<Iso11146Result> {
        let caustic = self.to_caustic();

        let bq_x = caustic.extract_beam_quality_x()?;
        let bq_y = caustic.extract_beam_quality_y()?;

        let astigmatic_difference_m = (bq_x.waist_position_m - bq_y.waist_position_m).abs();

        Some(Iso11146Result {
            m2_x: bq_x.m2,
            m2_y: bq_y.m2,
            waist_x_m: bq_x.beam_waist_m,
            waist_y_m: bq_y.beam_waist_m,
            // Average waist position (mean of the two principal planes)
            waist_z_m: 0.5 * (bq_x.waist_position_m + bq_y.waist_position_m),
            divergence_x_rad: bq_x.divergence_half_angle_rad(),
            divergence_y_rad: bq_y.divergence_half_angle_rad(),
            astigmatic_difference_m,
            beam_parameter_product_x: bq_x.beam_parameter_product(),
            beam_parameter_product_y: bq_y.beam_parameter_product(),
        })
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Convert to a [`BeamCaustic`] for fitting.
    fn to_caustic(&self) -> BeamCaustic {
        let mut caustic = BeamCaustic::new(self.wavelength_m);
        for m in &self.measurements {
            caustic.add_measurement(m.z_m, m.radius_x_m, m.radius_y_m);
        }
        caustic
    }
}

// ─── Result type ──────────────────────────────────────────────────────────────

/// Results of an ISO 11146 beam quality measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct Iso11146Result {
    /// M² factor along the x principal axis.
    pub m2_x: f64,
    /// M² factor along the y principal axis.
    pub m2_y: f64,
    /// Beam waist radius in x (m).
    pub waist_x_m: f64,
    /// Beam waist radius in y (m).
    pub waist_y_m: f64,
    /// Mean waist axial position (m).
    pub waist_z_m: f64,
    /// Far-field divergence half-angle in x (rad).
    pub divergence_x_rad: f64,
    /// Far-field divergence half-angle in y (rad).
    pub divergence_y_rad: f64,
    /// Astigmatic difference: |z₀_x − z₀_y| (m).
    pub astigmatic_difference_m: f64,
    /// Beam parameter product in x: w₀_x · θ_x (m·rad).
    pub beam_parameter_product_x: f64,
    /// Beam parameter product in y: w₀_y · θ_y (m·rad).
    pub beam_parameter_product_y: f64,
}

// ─── Synthetic data generation ────────────────────────────────────────────────

/// Generate synthetic caustic data for an ideal Gaussian beam for testing.
///
/// Produces `n_points` evenly spaced (z, w_x) pairs using the standard
/// Gaussian propagation formula with the specified M² and waist parameters.
///
/// # Arguments
/// * `wavelength_m` – Vacuum wavelength (m).
/// * `m2` – M² factor (1 = ideal Gaussian).
/// * `w0_m` – Beam waist radius at focus (m).
/// * `z0_m` – Axial position of the focus (m).
/// * `z_min`, `z_max` – Axial range to sample (m).
/// * `n_points` – Number of sample points.
///
/// # Returns
/// A `Vec<(z_m, w_x_m)>` of radius (not diameter) values.
pub fn synthetic_gaussian_caustic(
    wavelength_m: f64,
    m2: f64,
    w0_m: f64,
    z0_m: f64,
    z_min: f64,
    z_max: f64,
    n_points: usize,
) -> Vec<(f64, f64)> {
    let bq = BeamQuality::new(wavelength_m, m2, w0_m, z0_m);
    (0..n_points)
        .map(|i| {
            let t = if n_points > 1 {
                i as f64 / (n_points - 1) as f64
            } else {
                0.5
            };
            let z = z_min + t * (z_max - z_min);
            let w = bq.beam_radius_at(z);
            (z, w)
        })
        .collect()
}

/// Check the ISO 11146 sampling requirement directly on a `(z, w)` dataset.
///
/// This is a convenience function used in tests and standalone scripts.
///
/// Returns `(n_near, n_far)` counts relative to estimated `(z₀, zR)`.
pub fn count_near_far_samples(z: &[f64], w: &[f64], wavelength_m: f64) -> (usize, usize) {
    let w2: Vec<f64> = w.iter().map(|&wi| wi * wi).collect();
    let (a, b, c) = match fit_parabola(z, &w2) {
        Some(coeffs) => coeffs,
        None => return (0, 0),
    };
    if c <= 0.0 {
        return (0, 0);
    }
    let z0 = -b / (2.0 * c);
    let w0_sq = a - b * b / (4.0 * c);
    if w0_sq <= 0.0 {
        return (0, 0);
    }
    let w0 = w0_sq.sqrt();
    let m2 = (PI * w0 * c.sqrt() / wavelength_m).max(1.0);
    let zr = PI * w0 * w0 / (m2 * wavelength_m);

    let near = z.iter().filter(|&&zi| (zi - z0).abs() <= zr).count();
    let far = z.iter().filter(|&&zi| (zi - z0).abs() >= 2.0 * zr).count();
    (near, far)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Populate an `Iso11146Measurement` from synthetic Gaussian data and check M²
    fn build_measurement(lambda: f64, m2: f64, w0: f64, z0: f64) -> Iso11146Measurement {
        let bq = BeamQuality::new(lambda, m2, w0, z0);
        let zr = bq.rayleigh_range_m();
        let mut meas = Iso11146Measurement::new(lambda);
        // 6 near-waist + 6 far-field points
        for i in 0..6usize {
            let z = z0 - zr + 2.0 * zr * i as f64 / 5.0;
            let w = bq.beam_radius_at(z);
            meas.add_point(z, 2.0 * w, 2.0 * w);
        }
        for i in 0..6usize {
            let z = z0 + 2.5 * zr + 2.0 * zr * i as f64 / 5.0;
            let w = bq.beam_radius_at(z);
            meas.add_point(z, 2.0 * w, 2.0 * w);
        }
        meas
    }

    #[test]
    fn iso_analysis_recovers_m2() {
        let lambda = 1064e-9;
        let m2 = 1.5;
        let w0 = 0.5e-3;
        let meas = build_measurement(lambda, m2, w0, 0.0);
        let result = meas.analyze().expect("ISO analysis should succeed");
        assert!(
            (result.m2_x - m2).abs() / m2 < 0.01,
            "M²_x: got {}, expected {}",
            result.m2_x,
            m2
        );
    }

    #[test]
    fn iso_analysis_recovers_waist() {
        let lambda = 1064e-9;
        let w0 = 0.8e-3;
        let meas = build_measurement(lambda, 1.0, w0, 0.05);
        let result = meas.analyze().expect("ISO analysis should succeed");
        assert!(
            (result.waist_x_m - w0).abs() / w0 < 0.01,
            "waist_x: got {}, expected {}",
            result.waist_x_m,
            w0
        );
    }

    #[test]
    fn bpp_satisfies_relation() {
        let lambda = 532e-9;
        let m2 = 2.0;
        let meas = build_measurement(lambda, m2, 0.3e-3, 0.0);
        let result = meas.analyze().expect("ISO analysis should succeed");
        let expected_bpp = m2 * lambda / PI;
        assert!(
            (result.beam_parameter_product_x - expected_bpp).abs() / expected_bpp < 0.02,
            "BPP: got {}, expected {}",
            result.beam_parameter_product_x,
            expected_bpp
        );
    }

    #[test]
    fn synthetic_gaussian_has_correct_waist() {
        let lambda = 1064e-9;
        let w0 = 1e-3;
        let points = synthetic_gaussian_caustic(lambda, 1.0, w0, 0.0, -0.5, 0.5, 20);
        // The minimum radius should occur near z=0 and be close to w0
        let min_w = points.iter().map(|(_, w)| *w).fold(f64::INFINITY, f64::min);
        assert!(
            (min_w - w0).abs() / w0 < 0.05,
            "Min radius: got {min_w}, expected {w0}"
        );
    }

    #[test]
    fn d4sigma_radius_conversion() {
        // D4σ = 2w, so adding point with d4sigma = 2w should store radius = w
        let lambda = 1064e-9;
        let mut meas = Iso11146Measurement::new(lambda);
        meas.add_point(0.0, 2.0e-3, 3.0e-3);
        assert!((meas.measurements[0].radius_x_m - 1.0e-3).abs() < 1e-15);
        assert!((meas.measurements[0].radius_y_m - 1.5e-3).abs() < 1e-15);
    }

    #[test]
    fn astigmatism_zero_for_symmetric_beam() {
        let lambda = 1064e-9;
        let meas = build_measurement(lambda, 1.2, 0.4e-3, 0.0);
        let result = meas.analyze().expect("ISO analysis should succeed");
        // Symmetric beam: x and y waist positions are identical
        assert!(
            result.astigmatic_difference_m < 1e-9,
            "Astigmatism should be ~0 for symmetric beam, got {}",
            result.astigmatic_difference_m
        );
    }

    #[test]
    fn count_near_far_helper() {
        let lambda = 1064e-9;
        let w0 = 1e-3;
        // zR for M²=1, w0=1mm, λ=1064nm: zR = π(1e-3)² / 1064e-9 ≈ 2.95 m
        // Use range ±10 m to ensure both near-waist and far-field coverage
        let points = synthetic_gaussian_caustic(lambda, 1.0, w0, 0.0, -10.0, 10.0, 40);
        let z: Vec<f64> = points.iter().map(|(z, _)| *z).collect();
        let w: Vec<f64> = points.iter().map(|(_, w)| *w).collect();
        let (near, far) = count_near_far_samples(&z, &w, lambda);
        assert!(near > 0, "Should have near-waist samples, got {near}");
        assert!(far > 0, "Should have far-field samples, got {far}");
    }
}
