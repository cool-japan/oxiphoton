//! Beam caustic measurement and fitting.
//!
//! A *caustic* is the w(z) profile of a laser beam along the propagation axis.
//! ISO 11146 specifies that the caustic is characterised by fitting a parabola to
//! the squared beam-radius measurements:
//!
//! ```text
//!   w²(z) = a + b z + c z²
//! ```
//!
//! From the fitted coefficients (a, b, c) the beam quality parameters follow:
//!
//! ```text
//!   z₀ = −b / (2c)                      (waist position)
//!   w₀² = a − b² / (4c)                 (waist squared)
//!   M² = π w₀ √c / λ                    (beam quality factor)
//! ```
//!
//! Reference: ISO 11146-1:2021 Annex B

use std::f64::consts::PI;

use super::m2_factor::BeamQuality;

// ─── Data structures ──────────────────────────────────────────────────────────

/// A single beam-radius measurement at axial position `z_m`.
///
/// Radii should be the second-moment (D4σ/2) values, as required by ISO 11146.
#[derive(Debug, Clone, PartialEq)]
pub struct BeamMeasurement {
    /// Axial position (m).
    pub z_m: f64,
    /// Beam radius in x (second-moment / D4σ / 2) (m).
    pub radius_x_m: f64,
    /// Beam radius in y (second-moment / D4σ / 2) (m).
    pub radius_y_m: f64,
}

/// Beam caustic: collection of [`BeamMeasurement`] samples along the
/// propagation axis together with the vacuum wavelength.
///
/// Call [`BeamCaustic::add_measurement`] to populate the caustic, then use
/// [`BeamCaustic::extract_beam_quality_x`] / [`BeamCaustic::extract_beam_quality_y`]
/// to obtain the ISO-11146 beam quality parameters.
#[derive(Debug, Clone)]
pub struct BeamCaustic {
    /// All recorded measurements.
    pub measurements: Vec<BeamMeasurement>,
    /// Vacuum wavelength (m).
    pub wavelength_m: f64,
}

impl BeamCaustic {
    /// Create an empty caustic for the given wavelength.
    pub fn new(wavelength_m: f64) -> Self {
        Self {
            measurements: Vec::new(),
            wavelength_m,
        }
    }

    /// Append a measurement.
    ///
    /// `radius_x_m` and `radius_y_m` are the D4σ/2 radii in x and y.
    pub fn add_measurement(&mut self, z_m: f64, radius_x_m: f64, radius_y_m: f64) {
        self.measurements.push(BeamMeasurement {
            z_m,
            radius_x_m,
            radius_y_m,
        });
    }

    // ── Parabola fitting helpers ──────────────────────────────────────────────

    /// Fit `w²(z) = a + b z + c z²` to the x-axis radii.
    ///
    /// Returns `Some((a, b, c))` on success; `None` if fewer than 3 measurements
    /// are available or if the system is degenerate.
    pub fn fit_parabola_x(&self) -> Option<(f64, f64, f64)> {
        let z: Vec<f64> = self.measurements.iter().map(|m| m.z_m).collect();
        let w2: Vec<f64> = self
            .measurements
            .iter()
            .map(|m| m.radius_x_m * m.radius_x_m)
            .collect();
        fit_parabola(&z, &w2)
    }

    /// Fit `w²(z) = a + b z + c z²` to the y-axis radii.
    ///
    /// Returns `Some((a, b, c))` on success; `None` if fewer than 3 measurements
    /// are available or if the system is degenerate.
    pub fn fit_parabola_y(&self) -> Option<(f64, f64, f64)> {
        let z: Vec<f64> = self.measurements.iter().map(|m| m.z_m).collect();
        let w2: Vec<f64> = self
            .measurements
            .iter()
            .map(|m| m.radius_y_m * m.radius_y_m)
            .collect();
        fit_parabola(&z, &w2)
    }

    // ── Beam quality extraction ───────────────────────────────────────────────

    /// Extract ISO 11146 beam quality parameters from the x-axis caustic fit.
    ///
    /// Returns `None` if the fit fails or if the inferred waist is unphysical.
    pub fn extract_beam_quality_x(&self) -> Option<BeamQuality> {
        let (a, b, c) = self.fit_parabola_x()?;
        beam_quality_from_parabola(a, b, c, self.wavelength_m)
    }

    /// Extract ISO 11146 beam quality parameters from the y-axis caustic fit.
    ///
    /// Returns `None` if the fit fails or if the inferred waist is unphysical.
    pub fn extract_beam_quality_y(&self) -> Option<BeamQuality> {
        let (a, b, c) = self.fit_parabola_y()?;
        beam_quality_from_parabola(a, b, c, self.wavelength_m)
    }

    // ── Residuals ────────────────────────────────────────────────────────────

    /// RMS residual of the parabolic fit to x-axis w²(z).
    ///
    /// Returns 0 if fewer than 3 points or if the fit fails.
    pub fn fit_residual_x(&self) -> f64 {
        match self.fit_parabola_x() {
            None => 0.0,
            Some((a, b, c)) => {
                let n = self.measurements.len();
                if n == 0 {
                    return 0.0;
                }
                let sum_sq: f64 = self
                    .measurements
                    .iter()
                    .map(|m| {
                        let z = m.z_m;
                        let w2_meas = m.radius_x_m * m.radius_x_m;
                        let w2_fit = a + b * z + c * z * z;
                        let diff = w2_meas - w2_fit;
                        diff * diff
                    })
                    .sum();
                (sum_sq / n as f64).sqrt()
            }
        }
    }

    /// RMS residual of the parabolic fit to y-axis w²(z).
    ///
    /// Returns 0 if fewer than 3 points or if the fit fails.
    pub fn fit_residual_y(&self) -> f64 {
        match self.fit_parabola_y() {
            None => 0.0,
            Some((a, b, c)) => {
                let n = self.measurements.len();
                if n == 0 {
                    return 0.0;
                }
                let sum_sq: f64 = self
                    .measurements
                    .iter()
                    .map(|m| {
                        let z = m.z_m;
                        let w2_meas = m.radius_y_m * m.radius_y_m;
                        let w2_fit = a + b * z + c * z * z;
                        let diff = w2_meas - w2_fit;
                        diff * diff
                    })
                    .sum();
                (sum_sq / n as f64).sqrt()
            }
        }
    }

    /// Return a reference to the raw measurement list.
    pub fn to_vec(&self) -> &Vec<BeamMeasurement> {
        &self.measurements
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Least-squares parabolic fit: w²(z) = a + b z + c z².
///
/// Solves the 3×3 normal equations via partial-pivot Gaussian elimination:
///
/// ```text
///   ⎡ n    Σz   Σz²  ⎤ ⎡ a ⎤   ⎡ Σw²   ⎤
///   ⎢ Σz   Σz²  Σz³  ⎥ ⎢ b ⎥ = ⎢ Σz w² ⎥
///   ⎣ Σz²  Σz³  Σz⁴  ⎦ ⎣ c ⎦   ⎣ Σz²w² ⎦
/// ```
///
/// Returns `None` if fewer than 3 data points are provided or the matrix is
/// singular (pivot < 1e-300).
pub(crate) fn fit_parabola(z: &[f64], w2: &[f64]) -> Option<(f64, f64, f64)> {
    let n = z.len();
    if n < 3 || n != w2.len() {
        return None;
    }
    let nf = n as f64;

    // Accumulate moments
    let sz1: f64 = z.iter().sum();
    let sz2: f64 = z.iter().map(|&zi| zi * zi).sum();
    let sz3: f64 = z.iter().map(|&zi| zi * zi * zi).sum();
    let sz4: f64 = z.iter().map(|&zi| zi * zi * zi * zi).sum();
    let sw: f64 = w2.iter().sum();
    let szw: f64 = z.iter().zip(w2.iter()).map(|(&zi, &wi)| zi * wi).sum();
    let sz2w: f64 = z.iter().zip(w2.iter()).map(|(&zi, &wi)| zi * zi * wi).sum();

    // Augmented matrix [A | b] in row-major order, 3 rows × 4 cols
    let mut mat = [
        [nf, sz1, sz2, sw],
        [sz1, sz2, sz3, szw],
        [sz2, sz3, sz4, sz2w],
    ];

    // Gaussian elimination with partial pivoting
    for col in 0..3usize {
        // Find pivot row
        let mut max_val = mat[col][col].abs();
        let mut max_row = col;
        for (row, r_data) in mat.iter().enumerate().skip(col + 1) {
            if r_data[col].abs() > max_val {
                max_val = r_data[col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-300 {
            return None; // singular or near-singular
        }
        mat.swap(col, max_row);

        // Eliminate below
        let pivot = mat[col][col];
        for row in col + 1..3 {
            let factor = mat[row][col] / pivot;
            let pivot_row: Vec<f64> = mat[col][col..4].to_vec();
            for (k_off, &pv) in pivot_row.iter().enumerate() {
                mat[row][col + k_off] -= factor * pv;
            }
        }
    }

    // Back substitution
    let c = mat[2][3] / mat[2][2];
    let b = (mat[1][3] - mat[1][2] * c) / mat[1][1];
    let a = (mat[0][3] - mat[0][2] * c - mat[0][1] * b) / mat[0][0];

    Some((a, b, c))
}

/// Derive `BeamQuality` from parabola coefficients (a, b, c) and wavelength.
///
/// ```text
///   z₀  = −b / (2c)
///   w₀² = a − b² / (4c)    (must be > 0 for physical result)
///   M²  = π w₀ √c / λ
/// ```
fn beam_quality_from_parabola(a: f64, b: f64, c: f64, wavelength_m: f64) -> Option<BeamQuality> {
    if c <= 0.0 {
        return None; // beam does not converge — unphysical fit
    }
    let z0 = -b / (2.0 * c);
    let w0_sq = a - b * b / (4.0 * c);
    if w0_sq <= 0.0 {
        return None; // waist would be imaginary
    }
    let w0 = w0_sq.sqrt();
    let m2 = (PI * w0 * c.sqrt() / wavelength_m).max(1.0);

    Some(BeamQuality::new(wavelength_m, m2, w0, z0))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic caustic for an ideal Gaussian (M²=1) and check fit.
    fn synthetic_caustic(lambda: f64, m2: f64, w0: f64, z0: f64) -> BeamCaustic {
        let bq = BeamQuality::new(lambda, m2, w0, z0);
        let mut caustic = BeamCaustic::new(lambda);
        let zr = bq.rayleigh_range_m();
        // 10 points spanning ±3 zR
        let n = 10usize;
        for i in 0..n {
            let z = z0 - 3.0 * zr + 6.0 * zr * i as f64 / (n - 1) as f64;
            let w = bq.beam_radius_at(z);
            caustic.add_measurement(z, w, w);
        }
        caustic
    }

    #[test]
    fn fit_ideal_gaussian_recovers_parameters() {
        let lambda = 1064e-9;
        let w0 = 1e-3;
        let z0 = 0.05;
        let caustic = synthetic_caustic(lambda, 1.0, w0, z0);
        let bq = caustic
            .extract_beam_quality_x()
            .expect("fit should succeed");
        assert!(
            (bq.beam_waist_m - w0).abs() / w0 < 1e-6,
            "waist: got {}, expected {}",
            bq.beam_waist_m,
            w0
        );
        assert!(
            (bq.waist_position_m - z0).abs() < 1e-9,
            "z0: got {}, expected {}",
            bq.waist_position_m,
            z0
        );
        assert!(
            (bq.m2 - 1.0).abs() < 1e-4,
            "M²: got {}, expected 1.0",
            bq.m2
        );
    }

    #[test]
    fn fit_m2_2_beam() {
        let lambda = 1064e-9;
        let m2 = 2.0;
        let w0 = 0.5e-3;
        let z0 = 0.0;
        let caustic = synthetic_caustic(lambda, m2, w0, z0);
        let bq = caustic
            .extract_beam_quality_x()
            .expect("fit should succeed");
        assert!(
            (bq.m2 - m2).abs() / m2 < 0.01,
            "M²: got {}, expected {}",
            bq.m2,
            m2
        );
    }

    #[test]
    fn fit_parabola_exact_three_points() {
        // w²(z) = 1 + 2z + 3z² → a=1, b=2, c=3
        let z = vec![-1.0, 0.0, 1.0];
        let w2: Vec<f64> = z.iter().map(|&zi| 1.0 + 2.0 * zi + 3.0 * zi * zi).collect();
        let (a, b, c) = fit_parabola(&z, &w2).expect("fit should succeed");
        assert!((a - 1.0).abs() < 1e-10, "a={}", a);
        assert!((b - 2.0).abs() < 1e-10, "b={}", b);
        assert!((c - 3.0).abs() < 1e-10, "c={}", c);
    }

    #[test]
    fn fit_fails_with_too_few_points() {
        let result = fit_parabola(&[0.0, 1.0], &[1.0, 2.0]);
        assert!(result.is_none(), "Should fail with only 2 points");
    }

    #[test]
    fn residual_exact_data_is_zero() {
        let lambda = 1064e-9;
        let caustic = synthetic_caustic(lambda, 1.0, 1e-3, 0.0);
        let res = caustic.fit_residual_x();
        // Floating-point round-trip through Gaussian elimination gives ~1e-20 residual
        assert!(
            res < 1e-18,
            "Residual for exact data should be near zero, got {}",
            res
        );
    }

    #[test]
    fn add_measurement_and_retrieve() {
        let mut c = BeamCaustic::new(1550e-9);
        c.add_measurement(0.0, 1e-3, 1.1e-3);
        c.add_measurement(0.1, 1.2e-3, 1.3e-3);
        assert_eq!(c.to_vec().len(), 2);
        assert!((c.to_vec()[0].z_m - 0.0).abs() < 1e-15);
    }

    #[test]
    fn negative_z0_caustic() {
        let lambda = 532e-9;
        let caustic = synthetic_caustic(lambda, 1.5, 0.3e-3, -0.2);
        let bq = caustic
            .extract_beam_quality_x()
            .expect("fit should succeed");
        assert!(
            (bq.waist_position_m - (-0.2)).abs() < 1e-6,
            "z0: got {}",
            bq.waist_position_m
        );
    }
}
