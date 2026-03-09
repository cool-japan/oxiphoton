/// Speckle statistics and reduction methods.
///
/// Fully developed speckle follows a negative-exponential (Rayleigh)
/// intensity probability distribution:
///
///   p(I) = (1/<I>) exp(−I/<I>)   for I ≥ 0
///
/// The speckle contrast C = σ_I / <I> = 1 for fully developed speckle.
/// After averaging N independent patterns C drops as 1/√N.
///
/// References:
///   Goodman, J. W. (2007). Speckle Phenomena in Optics. Roberts & Company.
use std::f64::consts::PI;

/// Error type for speckle calculations.
#[derive(Debug, Clone, PartialEq)]
pub enum SpeckleError {
    /// A physical parameter is out of its valid range.
    InvalidParameter(String),
    /// The provided arrays have inconsistent lengths.
    LengthMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for SpeckleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidParameter(msg) => write!(f, "Invalid parameter: {msg}"),
            Self::LengthMismatch { expected, got } => {
                write!(f, "Length mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for SpeckleError {}

// ─────────────────────────────────────────────────────────────────────────────
// SpeckleStatistics
// ─────────────────────────────────────────────────────────────────────────────

/// Statistical model for fully developed laser speckle.
///
/// The intensity follows a negative exponential distribution:
///   p(I) = (1/⟨I⟩) exp(−I/⟨I⟩)
///
/// The autocorrelation of the intensity pattern in the far field is:
///   C_I(Δr) = ⟨I⟩² (1 + |μ(Δr)|²)
///
/// where μ(Δr) is the degree of spatial coherence of the illuminating field.
#[derive(Debug, Clone)]
pub struct SpeckleStatistics {
    /// Mean intensity ⟨I⟩ \[W/m²\].
    pub mean_intensity: f64,
}

impl SpeckleStatistics {
    /// Create a fully-developed speckle model with given mean intensity.
    ///
    /// # Errors
    /// Returns `SpeckleError::InvalidParameter` if `mean_intensity ≤ 0`.
    pub fn new(mean_intensity: f64) -> Result<Self, SpeckleError> {
        if mean_intensity <= 0.0 {
            return Err(SpeckleError::InvalidParameter(
                "mean_intensity must be positive".into(),
            ));
        }
        Ok(Self { mean_intensity })
    }

    /// Probability density function of the intensity: p(I) = exp(−I/⟨I⟩)/⟨I⟩.
    ///
    /// Returns 0 for I < 0.
    pub fn pdf(&self, intensity: f64) -> f64 {
        if intensity < 0.0 {
            return 0.0;
        }
        let i_bar = self.mean_intensity;
        (-intensity / i_bar).exp() / i_bar
    }

    /// Cumulative distribution function P(I ≤ x) = 1 − exp(−x/⟨I⟩).
    pub fn cdf(&self, intensity: f64) -> f64 {
        if intensity < 0.0 {
            return 0.0;
        }
        1.0 - (-intensity / self.mean_intensity).exp()
    }

    /// Speckle contrast C = σ_I / ⟨I⟩.
    ///
    /// For fully developed speckle C = 1 (the standard deviation equals the mean).
    pub fn contrast(&self) -> f64 {
        // For a negative-exponential distribution: σ² = ⟨I⟩², so σ = ⟨I⟩.
        1.0
    }

    /// Intensity autocorrelation C_I(Δr) = ⟨I⟩² (1 + |μ(Δr)|²).
    ///
    /// Assumes that the underlying field has a Gaussian degree of coherence:
    ///   |μ(Δr)| = exp(−Δr²/(2 σ_c²))   where σ_c = coherence_area radius.
    ///
    /// # Parameters
    /// - `delta_r`       — transverse separation \[m\].
    /// - `coherence_area` — RMS radius of the coherence area \[m\].
    pub fn autocorrelation(&self, delta_r: f64, coherence_area: f64) -> Result<f64, SpeckleError> {
        if coherence_area <= 0.0 {
            return Err(SpeckleError::InvalidParameter(
                "coherence_area must be positive".into(),
            ));
        }
        let i_bar = self.mean_intensity;
        let mu_sq = (-delta_r * delta_r / (coherence_area * coherence_area)).exp();
        Ok(i_bar * i_bar * (1.0 + mu_sq))
    }

    /// Variance of intensity σ²_I = ⟨I⟩².
    pub fn variance(&self) -> f64 {
        self.mean_intensity * self.mean_intensity
    }

    /// Higher-order moment: ⟨Iⁿ⟩ = n! ⟨I⟩ⁿ  (Gamma distribution result).
    pub fn moment(&self, n: u32) -> f64 {
        let n_fac: f64 = (1..=(n as u64)).map(|k| k as f64).product();
        n_fac * self.mean_intensity.powi(n as i32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SpeckleReduction
// ─────────────────────────────────────────────────────────────────────────────

/// Speckle contrast reduction via averaging independent speckle patterns.
///
/// All methods return the residual speckle contrast C after the given
/// diversity strategy is applied.
pub struct SpeckleReduction;

impl SpeckleReduction {
    /// Spatial averaging of N spatially independent speckle patterns.
    ///
    /// C = 1 / √N
    ///
    /// Requires that the N patterns are uncorrelated (separated by more than
    /// a speckle size).
    pub fn spatial_averaging(n_patterns: usize) -> Result<f64, SpeckleError> {
        if n_patterns == 0 {
            return Err(SpeckleError::InvalidParameter(
                "n_patterns must be at least 1".into(),
            ));
        }
        Ok(1.0 / (n_patterns as f64).sqrt())
    }

    /// Polarization diversity: averaging two orthogonal polarization states.
    ///
    /// C = 1 / √2
    ///
    /// Two independent speckle patterns (one per polarization) are summed.
    pub fn polarization_diversity() -> f64 {
        1.0 / 2.0_f64.sqrt()
    }

    /// Frequency diversity: averaging M independent frequency bands.
    ///
    /// The number of independent speckle patterns M is determined by the
    /// path-length spread ΔL of the scattering medium relative to the
    /// temporal coherence length l_c:
    ///
    ///   M = 1 + ΔL / l_c
    ///
    /// Resulting contrast: C = 1 / √M.
    ///
    /// # Parameters
    /// - `bandwidth`         — optical bandwidth \[rad/s\].
    /// - `coherence_length`  — temporal coherence length l_c \[m\].
    /// - `path_length`       — path-length spread ΔL of the scatterer \[m\].
    pub fn frequency_diversity(
        bandwidth: f64,
        coherence_length: f64,
        path_length: f64,
    ) -> Result<f64, SpeckleError> {
        if bandwidth <= 0.0 {
            return Err(SpeckleError::InvalidParameter(
                "bandwidth must be positive".into(),
            ));
        }
        if coherence_length <= 0.0 {
            return Err(SpeckleError::InvalidParameter(
                "coherence_length must be positive".into(),
            ));
        }
        if path_length < 0.0 {
            return Err(SpeckleError::InvalidParameter(
                "path_length must be non-negative".into(),
            ));
        }
        let m = 1.0 + path_length / coherence_length;
        Ok(1.0 / m.sqrt())
    }

    /// Angular diversity: rotating diffuser with N independent angular positions.
    ///
    /// Equivalent to spatial averaging of N uncorrelated patterns.
    /// C = 1 / √N
    pub fn angular_diversity(n_angles: usize) -> Result<f64, SpeckleError> {
        Self::spatial_averaging(n_angles)
    }

    /// Combined diversity: spatial × frequency × polarization.
    ///
    /// C = 1 / √(N_spatial · M_freq · N_pol)
    pub fn combined(
        n_spatial: usize,
        m_frequency: f64,
        n_polarization: f64,
    ) -> Result<f64, SpeckleError> {
        if n_spatial == 0 {
            return Err(SpeckleError::InvalidParameter(
                "n_spatial must be at least 1".into(),
            ));
        }
        if m_frequency <= 0.0 {
            return Err(SpeckleError::InvalidParameter(
                "m_frequency must be positive".into(),
            ));
        }
        if n_polarization <= 0.0 {
            return Err(SpeckleError::InvalidParameter(
                "n_polarization must be positive".into(),
            ));
        }
        let total = n_spatial as f64 * m_frequency * n_polarization;
        Ok(1.0 / total.sqrt())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ObjectiveSpeckleSize
// ─────────────────────────────────────────────────────────────────────────────

/// Characteristic size of speckle patterns in image/far-field planes.
///
/// Two regimes:
/// - **Objective speckle** (image plane): size set by the aperture of the
///   imaging system (Airy disk).
/// - **Subjective speckle** (free-space far field): size set by the aperture
///   and propagation distance.
pub struct ObjectiveSpeckleSize;

impl ObjectiveSpeckleSize {
    /// Airy disk radius (first zero of J₁) in the image plane.
    ///
    ///   r_Airy = 1.22 · λ · f/#
    ///
    /// where f/# is the f-number (focal length / aperture diameter).
    ///
    /// # Parameters
    /// - `wavelength` — free-space wavelength \[m\].
    /// - `f_number`   — f-number of the imaging system (dimensionless).
    pub fn airy_disk_radius(wavelength: f64, f_number: f64) -> Result<f64, SpeckleError> {
        if wavelength <= 0.0 {
            return Err(SpeckleError::InvalidParameter(
                "wavelength must be positive".into(),
            ));
        }
        if f_number <= 0.0 {
            return Err(SpeckleError::InvalidParameter(
                "f_number must be positive".into(),
            ));
        }
        Ok(1.22 * wavelength * f_number)
    }

    /// Subjective speckle size in a free-space observation plane.
    ///
    /// The average speckle radius in the far field at distance `distance` from
    /// an aperture of size `aperture`:
    ///
    ///   r_speckle ≈ 1.22 · λ · z / D
    ///
    /// This is identical to the diffraction-limited Airy radius for an aperture
    /// of diameter D observed at distance z.
    ///
    /// # Parameters
    /// - `wavelength` — free-space wavelength \[m\].
    /// - `distance`   — propagation distance from the aperture to the observation plane \[m\].
    /// - `aperture`   — diameter of the illuminated aperture \[m\].
    pub fn speckle_size_subjective(
        wavelength: f64,
        distance: f64,
        aperture: f64,
    ) -> Result<f64, SpeckleError> {
        if wavelength <= 0.0 {
            return Err(SpeckleError::InvalidParameter(
                "wavelength must be positive".into(),
            ));
        }
        if distance <= 0.0 {
            return Err(SpeckleError::InvalidParameter(
                "distance must be positive".into(),
            ));
        }
        if aperture <= 0.0 {
            return Err(SpeckleError::InvalidParameter(
                "aperture must be positive".into(),
            ));
        }
        Ok(1.22 * wavelength * distance / aperture)
    }

    /// Mean speckle area in the far field \[m²\].
    ///
    /// A_speckle = π r_speckle²
    pub fn speckle_area(
        wavelength: f64,
        distance: f64,
        aperture: f64,
    ) -> Result<f64, SpeckleError> {
        let r = Self::speckle_size_subjective(wavelength, distance, aperture)?;
        Ok(PI * r * r)
    }

    /// Number of independent speckles in an observation area A_obs.
    ///
    ///   N = A_obs / A_speckle
    pub fn num_speckles_in_area(
        wavelength: f64,
        distance: f64,
        aperture: f64,
        observation_area: f64,
    ) -> Result<f64, SpeckleError> {
        if observation_area <= 0.0 {
            return Err(SpeckleError::InvalidParameter(
                "observation_area must be positive".into(),
            ));
        }
        let a_speckle = Self::speckle_area(wavelength, distance, aperture)?;
        Ok(observation_area / a_speckle)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SpeckleCorrelation
// ─────────────────────────────────────────────────────────────────────────────

/// Cross-correlation between two speckle intensity patterns.
///
/// For two speckle patterns I₁ and I₂ arising from similar but slightly
/// different illumination conditions (e.g., slightly different angles or
/// wavelengths), the normalised cross-correlation is:
///
///   C₁₂(Δr) = (⟨I₁(r) I₂(r+Δr)⟩ − ⟨I₁⟩⟨I₂⟩) / (σ₁ σ₂)
///
/// For fully developed speckle from the same source: C₁₂(0) = 1.
/// For independent speckle: C₁₂ = 0 everywhere.
#[derive(Debug, Clone)]
pub struct SpeckleCorrelation {
    /// Separations at which the cross-correlation is evaluated \[m\].
    pub separations: Vec<f64>,
    /// Normalised cross-correlation values C₁₂(Δr).
    pub correlation: Vec<f64>,
}

impl SpeckleCorrelation {
    /// Compute the cross-correlation between two sampled intensity patterns.
    ///
    /// Both patterns must be sampled on the same 1-D grid.  The result is
    /// evaluated at the shifts \[0, δx, 2δx, …\] where δx is the grid spacing.
    ///
    /// # Errors
    /// - `LengthMismatch` if the patterns have different lengths.
    /// - `InvalidParameter` if either pattern has zero variance.
    pub fn compute(
        pattern1: &[f64],
        pattern2: &[f64],
        grid_spacing: f64,
    ) -> Result<Self, SpeckleError> {
        let n = pattern1.len();
        if pattern2.len() != n {
            return Err(SpeckleError::LengthMismatch {
                expected: n,
                got: pattern2.len(),
            });
        }
        if n == 0 {
            return Err(SpeckleError::InvalidParameter(
                "patterns must not be empty".into(),
            ));
        }

        let mean1 = pattern1.iter().sum::<f64>() / n as f64;
        let mean2 = pattern2.iter().sum::<f64>() / n as f64;
        let var1 = pattern1.iter().map(|&x| (x - mean1).powi(2)).sum::<f64>() / n as f64;
        let var2 = pattern2.iter().map(|&x| (x - mean2).powi(2)).sum::<f64>() / n as f64;

        if var1 < f64::EPSILON || var2 < f64::EPSILON {
            return Err(SpeckleError::InvalidParameter(
                "patterns must have non-zero variance".into(),
            ));
        }

        let sigma1 = var1.sqrt();
        let sigma2 = var2.sqrt();

        // Compute cross-correlation for shifts 0..n/2.
        let n_shifts = n / 2;
        let mut separations = Vec::with_capacity(n_shifts);
        let mut correlation = Vec::with_capacity(n_shifts);

        for shift in 0..n_shifts {
            let cov: f64 = (0..(n - shift))
                .map(|i| (pattern1[i] - mean1) * (pattern2[i + shift] - mean2))
                .sum::<f64>()
                / (n - shift) as f64;
            separations.push(shift as f64 * grid_spacing);
            correlation.push(cov / (sigma1 * sigma2));
        }

        Ok(Self {
            separations,
            correlation,
        })
    }

    /// Model cross-correlation for two speckle patterns from sources with mutual
    /// coherence degree |μ_12|.
    ///
    /// C₁₂(0) = |μ_12|²
    ///
    /// The analytic Gaussian model:
    ///   C₁₂(Δr) = |μ_12|² exp(−Δr²/(2 σ_speckle²))
    pub fn analytic_gaussian(
        mutual_coherence: f64,
        speckle_size: f64,
        separations: Vec<f64>,
    ) -> Result<Self, SpeckleError> {
        if !(0.0..=1.0).contains(&mutual_coherence) {
            return Err(SpeckleError::InvalidParameter(
                "mutual_coherence must be in [0, 1]".into(),
            ));
        }
        if speckle_size <= 0.0 {
            return Err(SpeckleError::InvalidParameter(
                "speckle_size must be positive".into(),
            ));
        }
        let c0 = mutual_coherence * mutual_coherence;
        let correlation: Vec<f64> = separations
            .iter()
            .map(|&dr| c0 * (-dr * dr / (2.0 * speckle_size * speckle_size)).exp())
            .collect();
        Ok(Self {
            separations,
            correlation,
        })
    }

    /// Peak correlation value C₁₂(0).
    pub fn peak(&self) -> f64 {
        self.correlation.first().copied().unwrap_or(0.0)
    }

    /// Correlation length: separation at which C₁₂ drops to 1/e of its peak.
    ///
    /// Returns `None` if the correlation never drops to 1/e (e.g., flat distribution).
    pub fn correlation_length(&self) -> Option<f64> {
        let c0 = self.peak();
        if c0 < f64::EPSILON {
            return None;
        }
        let threshold = c0 / std::f64::consts::E;
        for i in 1..self.separations.len() {
            if self.correlation[i] <= threshold {
                // Linear interpolation.
                let s0 = self.separations[i - 1];
                let s1 = self.separations[i];
                let c_prev = self.correlation[i - 1];
                let c_next = self.correlation[i];
                let t = (threshold - c_prev) / (c_next - c_prev);
                return Some(s0 + t * (s1 - s0));
            }
        }
        None
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
    fn speckle_pdf_integrates_to_unity() {
        let stats = SpeckleStatistics::new(1.0).expect("valid");
        // Numerical integration over [0, 30*<I>] with trapezoidal rule.
        let n = 10_000_usize;
        let i_max = 30.0_f64;
        let di = i_max / (n as f64 - 1.0);
        let integral: f64 = (0..n)
            .map(|k| {
                let i = k as f64 * di;
                let w = if k == 0 || k == n - 1 { 0.5 } else { 1.0 };
                w * stats.pdf(i) * di
            })
            .sum();
        assert_abs_diff_eq!(integral, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn speckle_contrast_is_unity() {
        let stats = SpeckleStatistics::new(2.5).expect("valid");
        assert_abs_diff_eq!(stats.contrast(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn spatial_averaging_contrast_reduction() {
        let c = SpeckleReduction::spatial_averaging(4).expect("valid");
        assert_abs_diff_eq!(c, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn polarization_diversity_contrast() {
        let c = SpeckleReduction::polarization_diversity();
        assert_abs_diff_eq!(c, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn frequency_diversity_no_scatter_gives_unity_contrast() {
        // path_length = 0 → M = 1 → C = 1.
        let c = SpeckleReduction::frequency_diversity(1e12, 1e-3, 0.0).expect("valid");
        assert_abs_diff_eq!(c, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn airy_disk_radius_scales_with_f_number() {
        let r1 = ObjectiveSpeckleSize::airy_disk_radius(633e-9, 2.0).expect("ok");
        let r2 = ObjectiveSpeckleSize::airy_disk_radius(633e-9, 4.0).expect("ok");
        assert_abs_diff_eq!(r2, 2.0 * r1, epsilon = 1e-20);
    }

    #[test]
    fn speckle_correlation_self_is_unity() {
        // Autocorrelation of a non-trivial pattern at zero shift = 1.
        let pattern: Vec<f64> = (0..64).map(|i| (i as f64 * 0.3).sin() + 2.0).collect();
        let corr = SpeckleCorrelation::compute(&pattern, &pattern, 1e-6).expect("ok");
        assert_abs_diff_eq!(corr.peak(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn analytic_gaussian_correlation_peak_equals_mu_squared() {
        let mu = 0.7_f64;
        let seps: Vec<f64> = (0..50).map(|i| i as f64 * 1e-6).collect();
        let corr = SpeckleCorrelation::analytic_gaussian(mu, 5e-6, seps).expect("ok");
        assert_abs_diff_eq!(corr.peak(), mu * mu, epsilon = 1e-12);
    }

    #[test]
    fn num_speckles_in_area_positive() {
        let n = ObjectiveSpeckleSize::num_speckles_in_area(633e-9, 1.0, 1e-3, 1e-4).expect("ok");
        assert!(n > 0.0);
    }

    #[test]
    fn speckle_moment_n2_equals_two_i_bar_squared() {
        // ⟨I²⟩ = 2 ⟨I⟩² for the negative-exponential distribution.
        let i_bar = 3.0_f64;
        let stats = SpeckleStatistics::new(i_bar).expect("valid");
        assert_abs_diff_eq!(stats.moment(2), 2.0 * i_bar * i_bar, epsilon = 1e-10);
    }
}
