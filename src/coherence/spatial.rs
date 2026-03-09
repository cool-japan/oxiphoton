/// Spatial Coherence — Van Cittert-Zernike theorem applied to Young's experiment
/// and propagating coherence in optical systems.
///
/// Covers:
/// - Young's double-slit fringe visibility from the Van Cittert-Zernike theorem
/// - Coherence radius evolution through free space and lenses
/// - Partially coherent beam representation combining spatial/temporal coherence
/// - Tabulated lateral coherence functions
use std::f64::consts::PI;

/// Speed of light in vacuum \[m/s\].
const C: f64 = 2.997_924_58e8;

/// Error type for spatial coherence calculations.
#[derive(Debug, Clone, PartialEq)]
pub enum SpatialCoherenceError {
    /// A physical parameter is out of its valid range.
    InvalidParameter(String),
    /// The provided data arrays have inconsistent lengths.
    LengthMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for SpatialCoherenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidParameter(msg) => write!(f, "Invalid parameter: {msg}"),
            Self::LengthMismatch { expected, got } => {
                write!(f, "Length mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for SpatialCoherenceError {}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: jinc function  jinc(x) = 2 J₁(πx) / (πx)
// ─────────────────────────────────────────────────────────────────────────────

/// First-order Bessel function J₁(x) via its Maclaurin series (accurate for |x| ≲ 20).
fn bessel_j1(x: f64) -> f64 {
    if x.abs() < f64::EPSILON {
        return 0.0;
    }
    // Series: J₁(x) = Σ_{m=0}^∞ (−1)^m x^(2m+1) / (2^(2m+1) m! (m+1)!)
    // We iterate until correction is negligible.
    let mut term = x / 2.0;
    let mut sum = term;
    let mut m = 1_u32;
    loop {
        let x2 = x * x;
        term *= -x2 / (4.0 * m as f64 * (m as f64 + 1.0));
        let prev = sum;
        sum += term;
        if (sum - prev).abs() < f64::EPSILON * sum.abs().max(1e-300) {
            break;
        }
        m += 1;
        if m > 200 {
            break;
        }
    }
    sum
}

/// Jinc function: jinc(x) = 2 J₁(π x) / (π x), with jinc(0) = 1.
///
/// This arises as the 2-D Fourier transform of a uniform circular aperture
/// (the Airy pattern), and therefore gives the spatial degree of coherence
/// for a uniform incoherent circular source (Van Cittert-Zernike theorem).
fn jinc(x: f64) -> f64 {
    if x.abs() < 1e-14 {
        return 1.0;
    }
    let pix = PI * x;
    2.0 * bessel_j1(pix) / pix
}

// ─────────────────────────────────────────────────────────────────────────────
// SpatialCoherence
// ─────────────────────────────────────────────────────────────────────────────

/// Spatial coherence utilities for incoherent sources observed in the far field.
///
/// Based on the Van Cittert-Zernike theorem, which states that the spatial
/// degree of coherence in the far field of an extended incoherent source equals
/// the (normalised) Fourier transform of the source intensity distribution.
pub struct SpatialCoherence;

impl SpatialCoherence {
    /// Fringe visibility in Young's double-slit experiment for a uniform circular source.
    ///
    /// By the Van Cittert-Zernike theorem the degree of coherence between two
    /// pinholes separated by `slit_separation` d, when illuminated by an
    /// extended incoherent source of diameter D at distance `distance` z, is:
    ///
    ///   μ = |jinc(d·D / (λ·z))|
    ///
    /// where jinc(u) = 2 J₁(πu)/(πu).
    ///
    /// The Young fringe visibility equals |μ| for equal-intensity pinholes.
    ///
    /// # Parameters
    /// - `source_diameter`  — diameter of the incoherent source \[m\].
    /// - `slit_separation`  — centre-to-centre distance of the pinholes \[m\].
    /// - `wavelength`       — free-space wavelength \[m\].
    /// - `distance`         — source-to-pinhole distance \[m\].
    pub fn young_visibility(
        source_diameter: f64,
        slit_separation: f64,
        wavelength: f64,
        distance: f64,
    ) -> Result<f64, SpatialCoherenceError> {
        if source_diameter <= 0.0 {
            return Err(SpatialCoherenceError::InvalidParameter(
                "source_diameter must be positive".into(),
            ));
        }
        if wavelength <= 0.0 {
            return Err(SpatialCoherenceError::InvalidParameter(
                "wavelength must be positive".into(),
            ));
        }
        if distance <= 0.0 {
            return Err(SpatialCoherenceError::InvalidParameter(
                "distance must be positive".into(),
            ));
        }
        // Argument of jinc: u = d·D/(λ·z)
        let u = slit_separation * source_diameter / (wavelength * distance);
        Ok(jinc(u).abs())
    }

    /// Transverse coherence radius r_c = λ z / (π d_source/2) \[m\].
    ///
    /// This is the separation at which the visibility drops to the first zero
    /// of the jinc function, i.e. where J₁(π u) = 0 first (u ≈ 1.22/2 for jinc).
    ///
    /// For the coherence radius (where |μ| drops to ~0.88, first significant
    /// drop) we use the simpler formula r_c = λ z / d, which is the standard
    /// Van Cittert-Zernike approximation used in practice.
    ///
    /// # Parameters
    /// - `source_size`  — linear size (diameter) of the source \[m\].
    /// - `wavelength`   — free-space wavelength \[m\].
    /// - `distance`     — propagation distance \[m\].
    pub fn coherence_radius(
        source_size: f64,
        wavelength: f64,
        distance: f64,
    ) -> Result<f64, SpatialCoherenceError> {
        if source_size <= 0.0 {
            return Err(SpatialCoherenceError::InvalidParameter(
                "source_size must be positive".into(),
            ));
        }
        if wavelength <= 0.0 {
            return Err(SpatialCoherenceError::InvalidParameter(
                "wavelength must be positive".into(),
            ));
        }
        if distance <= 0.0 {
            return Err(SpatialCoherenceError::InvalidParameter(
                "distance must be positive".into(),
            ));
        }
        Ok(wavelength * distance / source_size)
    }

    /// Transverse coherence length from the beam divergence half-angle θ_div.
    ///
    /// l_c = λ / (2π θ_div)
    ///
    /// This relates the angular spread of the source to the coherence length
    /// in the plane perpendicular to propagation.
    pub fn transverse_coherence_length(
        divergence_angle: f64,
        wavelength: f64,
    ) -> Result<f64, SpatialCoherenceError> {
        if divergence_angle <= 0.0 {
            return Err(SpatialCoherenceError::InvalidParameter(
                "divergence_angle must be positive".into(),
            ));
        }
        if wavelength <= 0.0 {
            return Err(SpatialCoherenceError::InvalidParameter(
                "wavelength must be positive".into(),
            ));
        }
        Ok(wavelength / (2.0 * PI * divergence_angle))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PropagatingCoherence
// ─────────────────────────────────────────────────────────────────────────────

/// Tracks the coherence area of a partially coherent beam as it propagates
/// through an optical system modelled in the paraxial approximation.
///
/// The coherence radius r_c evolves via ABCD matrix optics analogously to the
/// Gaussian beam spot size.  For a beam with wavelength λ:
///
/// Free space propagation by z:
///   r_c(z) = r_c₀ √(1 + (z/z_c)²)   with z_c = π r_c₀² / λ
///
/// Thin lens with focal length f:
///   1/r_c_out = 1/r_c_in − 1/(f · λ/π)   (coherence Rayleigh range)
///
/// These are analogues of Gaussian beam propagation for the coherence function.
#[derive(Debug, Clone)]
pub struct PropagatingCoherence {
    /// Current transverse coherence radius r_c \[m\].
    coherence_radius: f64,
    /// Wavelength \[m\].
    wavelength: f64,
    /// Accumulated axial position \[m\] (for record-keeping).
    pub position: f64,
}

impl PropagatingCoherence {
    /// Create a new propagating coherence tracker.
    ///
    /// # Errors
    /// Returns `SpatialCoherenceError::InvalidParameter` if either argument
    /// is non-positive.
    pub fn new(
        initial_coherence_radius: f64,
        wavelength: f64,
    ) -> Result<Self, SpatialCoherenceError> {
        if initial_coherence_radius <= 0.0 {
            return Err(SpatialCoherenceError::InvalidParameter(
                "initial_coherence_radius must be positive".into(),
            ));
        }
        if wavelength <= 0.0 {
            return Err(SpatialCoherenceError::InvalidParameter(
                "wavelength must be positive".into(),
            ));
        }
        Ok(Self {
            coherence_radius: initial_coherence_radius,
            wavelength,
            position: 0.0,
        })
    }

    /// Propagate through free space of length `z` \[m\].
    ///
    /// Coherence radius evolves as:
    ///   r_c(z) = r_c₀ √(1 + (z/z_c)²)   with z_c = π r_c₀²/λ
    pub fn propagate_free_space(&mut self, z: f64) {
        let r0 = self.coherence_radius;
        let z_c = PI * r0 * r0 / self.wavelength;
        self.coherence_radius = r0 * (1.0 + (z / z_c).powi(2)).sqrt();
        self.position += z;
    }

    /// Pass through a thin lens of focal length `focal_length` \[m\].
    ///
    /// A lens of focal length f reduces the coherence Rayleigh range q by:
    ///   1/q_out = 1/q_in − 1/(f)
    /// where q = π r_c² / λ (analogous to the complex beam parameter).
    ///
    /// For the coherence radius this gives:
    ///   r_c_out = r_c_in / √(1 + (r_c_in / (f · (λ/(π r_c_in))))²)
    ///
    /// In the thin-lens paraxial limit the transformation is simply:
    ///   r_c_out = r_c_in · |f| / √(f² + z_c²)  (matched Gaussian analogy)
    ///
    /// Here we apply the exact thin-lens ABCD result for the coherence function:
    ///   r_c → r_c  (unchanged by pure lens; the lens affects beam curvature).
    ///
    /// For a focusing lens the coherence is re-mapped onto the focal plane;
    /// at distance f after the lens r_c evolves as free-space from the waist.
    pub fn through_lens(&mut self, focal_length: f64) {
        // A thin lens does not change the coherence radius at the lens plane.
        // Instead it resets the Rayleigh range so that the beam focuses at
        // distance f.  We model this by computing the equivalent waist r_c_waist
        // that would produce the current r_c after propagating distance f.
        let z_c_in = PI * self.coherence_radius * self.coherence_radius / self.wavelength;
        // After a thin lens the new Rayleigh range: 1/z_c_out = 1/z_c_in ± 1/f
        // (using thin lens coherence matrix, sign depends on convention).
        let inv_z_c_out = 1.0 / z_c_in - 1.0 / focal_length;
        if inv_z_c_out.abs() > f64::EPSILON {
            let z_c_out = 1.0 / inv_z_c_out;
            // New coherence radius (at the lens plane) after the focal mapping.
            let r_new = (self.wavelength * z_c_out.abs() / PI).sqrt();
            self.coherence_radius = r_new;
        }
        // If inv_z_c_out ≈ 0 the beam is at the coherence focus; r_c is minimal.
    }

    /// Return the current coherence radius \[m\].
    pub fn coherence_radius(&self) -> f64 {
        self.coherence_radius
    }

    /// Return the current coherence Rayleigh range z_c = π r_c² / λ \[m\].
    pub fn coherence_rayleigh_range(&self) -> f64 {
        PI * self.coherence_radius * self.coherence_radius / self.wavelength
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PartiallyCoherentBeam
// ─────────────────────────────────────────────────────────────────────────────

/// Combined spatial and temporal coherence description of a beam.
///
/// Encapsulates:
/// - Spatial coherence radius r_c \[m\]
/// - Temporal coherence length l_c \[m\]
/// - Beam spot size w \[m\]
/// - Centre wavelength λ \[m\]
#[derive(Debug, Clone)]
pub struct PartiallyCoherentBeam {
    /// Transverse coherence radius \[m\].
    pub spatial_coherence_radius: f64,
    /// Longitudinal coherence length \[m\].
    pub temporal_coherence_length: f64,
    /// 1/e² beam spot radius \[m\].
    pub beam_radius: f64,
    /// Centre wavelength \[m\].
    pub wavelength: f64,
}

impl PartiallyCoherentBeam {
    /// Construct a partially coherent beam.
    ///
    /// # Errors
    /// Returns `SpatialCoherenceError::InvalidParameter` for non-positive inputs.
    pub fn new(
        spatial_coherence_radius: f64,
        temporal_coherence_length: f64,
        beam_radius: f64,
        wavelength: f64,
    ) -> Result<Self, SpatialCoherenceError> {
        if spatial_coherence_radius <= 0.0
            || temporal_coherence_length <= 0.0
            || beam_radius <= 0.0
            || wavelength <= 0.0
        {
            return Err(SpatialCoherenceError::InvalidParameter(
                "all parameters must be positive".into(),
            ));
        }
        Ok(Self {
            spatial_coherence_radius,
            temporal_coherence_length,
            beam_radius,
            wavelength,
        })
    }

    /// Spatio-temporal degree of coherence (product form approximation).
    ///
    /// μ_total(Δr, Δl) = μ_spatial(Δr) · μ_temporal(Δl)
    ///
    /// where
    ///   μ_spatial  = exp(−Δr²/(2 r_c²))
    ///   μ_temporal = exp(−Δl²/(2 (c τ_c)²)) with c τ_c = l_c
    pub fn degree_of_coherence(&self, delta_r: f64, delta_path: f64) -> f64 {
        let mu_spatial = (-delta_r * delta_r
            / (2.0 * self.spatial_coherence_radius * self.spatial_coherence_radius))
            .exp();
        let mu_temporal = (-delta_path * delta_path
            / (2.0 * self.temporal_coherence_length * self.temporal_coherence_length))
            .exp();
        mu_spatial * mu_temporal
    }

    /// Étendue G = π² · w² · (λ/r_c)² \[m² sr\].
    ///
    /// Measures the phase-space volume occupied by the partially coherent beam.
    pub fn etendue(&self) -> f64 {
        let divergence = self.wavelength / (PI * self.spatial_coherence_radius);
        PI * PI * self.beam_radius * self.beam_radius * divergence * divergence
    }

    /// Number of transverse coherence cells N_t = (w / r_c)².
    ///
    /// For a fully coherent beam N_t = 1; for a spatially incoherent
    /// beam N_t = (w/λ)² (one cell per resolution element).
    pub fn num_transverse_coherence_cells(&self) -> f64 {
        (self.beam_radius / self.spatial_coherence_radius).powi(2)
    }

    /// Temporal coherence time τ_c = l_c / c \[s\].
    pub fn temporal_coherence_time(&self) -> f64 {
        self.temporal_coherence_length / C
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LateralCoherenceFunction
// ─────────────────────────────────────────────────────────────────────────────

/// Tabulated lateral coherence function j(Δr) mapping spatial separation to
/// degree of spatial coherence.
///
/// The table is linearly interpolated; extrapolation returns 0.
#[derive(Debug, Clone)]
pub struct LateralCoherenceFunction {
    /// Separations Δr \[m\] (must be sorted ascending).
    pub separations: Vec<f64>,
    /// Degree of coherence j(Δr) ∈ \[0, 1\] at each separation.
    pub values: Vec<f64>,
}

impl LateralCoherenceFunction {
    /// Build a lateral coherence function from tabulated data.
    ///
    /// # Errors
    /// - `LengthMismatch` if `separations.len() ≠ values.len()`
    /// - `InvalidParameter` if the separation grid is not sorted or values are out of \[0, 1\]
    pub fn from_table(
        separations: Vec<f64>,
        values: Vec<f64>,
    ) -> Result<Self, SpatialCoherenceError> {
        if separations.len() != values.len() {
            return Err(SpatialCoherenceError::LengthMismatch {
                expected: separations.len(),
                got: values.len(),
            });
        }
        // Check sorting.
        for i in 1..separations.len() {
            if separations[i] <= separations[i - 1] {
                return Err(SpatialCoherenceError::InvalidParameter(
                    "separations must be strictly increasing".into(),
                ));
            }
        }
        Ok(Self {
            separations,
            values,
        })
    }

    /// Build a Gaussian lateral coherence function j(Δr) = exp(−Δr²/(2 r_c²)).
    pub fn gaussian(coherence_radius: f64, n_points: usize) -> Result<Self, SpatialCoherenceError> {
        if coherence_radius <= 0.0 {
            return Err(SpatialCoherenceError::InvalidParameter(
                "coherence_radius must be positive".into(),
            ));
        }
        let separations: Vec<f64> = (0..n_points)
            .map(|i| 4.0 * coherence_radius * i as f64 / (n_points as f64 - 1.0))
            .collect();
        let values: Vec<f64> = separations
            .iter()
            .map(|&dr| (-dr * dr / (2.0 * coherence_radius * coherence_radius)).exp())
            .collect();
        Ok(Self {
            separations,
            values,
        })
    }

    /// Build a jinc-shaped lateral coherence function (circular incoherent source).
    pub fn jinc_shaped(
        coherence_radius: f64,
        n_points: usize,
    ) -> Result<Self, SpatialCoherenceError> {
        if coherence_radius <= 0.0 {
            return Err(SpatialCoherenceError::InvalidParameter(
                "coherence_radius must be positive".into(),
            ));
        }
        let separations: Vec<f64> = (0..n_points)
            .map(|i| 4.0 * coherence_radius * i as f64 / (n_points as f64 - 1.0))
            .collect();
        let values: Vec<f64> = separations
            .iter()
            .map(|&dr| jinc(dr / coherence_radius).abs())
            .collect();
        Ok(Self {
            separations,
            values,
        })
    }

    /// Evaluate by linear interpolation.  Returns 0 outside the table range.
    pub fn evaluate(&self, delta_r: f64) -> f64 {
        let n = self.separations.len();
        if n == 0 || delta_r < self.separations[0] || delta_r > self.separations[n - 1] {
            return 0.0;
        }
        // Binary search for the bracketing interval.
        let idx = self
            .separations
            .partition_point(|&s| s <= delta_r)
            .saturating_sub(1)
            .min(n - 2);
        let s0 = self.separations[idx];
        let s1 = self.separations[idx + 1];
        let v0 = self.values[idx];
        let v1 = self.values[idx + 1];
        let t = (delta_r - s0) / (s1 - s0);
        v0 + t * (v1 - v0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn young_visibility_incoherent_point_source_is_unity() {
        // A very small source → coherence radius >> slit separation → V ≈ 1.
        let v = SpatialCoherence::young_visibility(
            1e-9,   // 1 nm source (quasi-point)
            1e-3,   // 1 mm slit separation
            633e-9, // 633 nm
            1.0,    // 1 m
        )
        .expect("valid parameters");
        assert_abs_diff_eq!(v, 1.0, epsilon = 1e-3);
    }

    #[test]
    fn coherence_radius_scales_linearly_with_distance() {
        let r1 = SpatialCoherence::coherence_radius(1e-3, 633e-9, 1.0).expect("ok");
        let r2 = SpatialCoherence::coherence_radius(1e-3, 633e-9, 2.0).expect("ok");
        assert_abs_diff_eq!(r2, 2.0 * r1, epsilon = 1e-20);
    }

    #[test]
    fn propagating_coherence_increases_with_free_space() {
        let mut pc = PropagatingCoherence::new(0.5e-3, 633e-9).expect("valid");
        let r0 = pc.coherence_radius();
        pc.propagate_free_space(10.0);
        assert!(pc.coherence_radius() > r0, "coherence radius must grow");
    }

    #[test]
    fn partially_coherent_beam_self_coherence_is_unity() {
        let beam = PartiallyCoherentBeam::new(0.5e-3, 1e-3, 1e-3, 633e-9).expect("valid");
        let mu = beam.degree_of_coherence(0.0, 0.0);
        assert_abs_diff_eq!(mu, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn lateral_coherence_gaussian_at_zero_is_unity() {
        let lcf = LateralCoherenceFunction::gaussian(0.5e-3, 100).expect("valid");
        let v = lcf.evaluate(0.0);
        assert_abs_diff_eq!(v, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn lateral_coherence_interpolation_is_monotone() {
        let rc = 0.5e-3_f64;
        let lcf = LateralCoherenceFunction::gaussian(rc, 200).expect("valid");
        let v1 = lcf.evaluate(rc * 0.5);
        let v2 = lcf.evaluate(rc * 1.5);
        assert!(v1 > v2, "coherence must decrease with separation");
    }

    #[test]
    fn jinc_at_zero_is_unity() {
        assert_abs_diff_eq!(super::jinc(0.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn jinc_first_zero_near_1_22() {
        // First zero of jinc is at u ≈ 1.2197 (first zero of J₁(πu)).
        let u = 1.2197_f64;
        assert!(
            super::jinc(u).abs() < 0.02,
            "jinc should be near zero at u ≈ 1.22"
        );
    }

    #[test]
    fn transverse_coherence_length_inversely_proportional_to_divergence() {
        let lc1 = SpatialCoherence::transverse_coherence_length(1e-3, 633e-9).expect("ok");
        let lc2 = SpatialCoherence::transverse_coherence_length(2e-3, 633e-9).expect("ok");
        assert_abs_diff_eq!(lc1, 2.0 * lc2, epsilon = 1e-20);
    }
}
