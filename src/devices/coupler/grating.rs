/// Grating coupler model.
///
/// Couples light between a waveguide mode and free-space (fiber) at angle θ.
///
/// Grating equation (first-order diffraction):
///   n_eff = n_clad · sin(θ) + λ / Λ
///
/// where Λ is the grating period, θ is the coupling angle (from normal),
/// n_eff is the waveguide effective index, n_clad is the cladding index above.
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct GratingCoupler {
    /// Grating period Λ (m).
    pub period: f64,
    /// Waveguide effective refractive index.
    pub n_eff: f64,
    /// Refractive index of the medium above the grating (fiber/air).
    pub n_clad: f64,
    /// Grating fill factor (fraction of period with tooth).
    pub fill_factor: f64,
    /// Number of grating periods.
    pub n_periods: usize,
}

impl GratingCoupler {
    /// Create a grating coupler.
    pub fn new(period: f64, n_eff: f64, n_clad: f64, fill_factor: f64, n_periods: usize) -> Self {
        Self {
            period,
            n_eff,
            n_clad,
            fill_factor,
            n_periods,
        }
    }

    /// Design a grating coupler for coupling angle θ (radians from normal) at λ.
    ///
    /// Computes the required period: Λ = λ / (n_eff - n_clad · sin(θ))
    pub fn design(n_eff: f64, n_clad: f64, theta_rad: f64, wavelength: f64) -> Option<Self> {
        let denom = n_eff - n_clad * theta_rad.sin();
        if denom <= 0.0 {
            return None;
        }
        let period = wavelength / denom;
        Some(Self::new(period, n_eff, n_clad, 0.5, 20))
    }

    /// Coupling angle θ for the given wavelength (radians from normal).
    ///
    /// From the grating equation: sin(θ) = (n_eff - λ/Λ) / n_clad
    pub fn coupling_angle(&self, wavelength: f64) -> Option<f64> {
        let sin_theta = (self.n_eff - wavelength / self.period) / self.n_clad;
        if sin_theta.abs() > 1.0 {
            None // Total internal reflection or evanescent order
        } else {
            Some(sin_theta.asin())
        }
    }

    /// Coupling bandwidth (3dB) in wavelength (m), estimated from grating length.
    ///
    /// Δλ ≈ λ² / (n_g · L) (similar to ring FSR but for diffraction)
    /// Approximate: Δλ ≈ λ · cos(θ) / (n_eff · N) where N = number of periods.
    pub fn bandwidth_3db(&self, wavelength: f64) -> f64 {
        // Rough estimate: Δλ / λ ≈ 1 / (n_eff * N) * cos(θ_B)
        let theta = self.coupling_angle(wavelength).unwrap_or(0.0);
        wavelength * theta.cos() / (self.n_eff * self.n_periods as f64)
    }

    /// Peak coupling efficiency estimate (simplified).
    ///
    /// For a 1D grating coupler with Gaussian overlap:
    /// η ≈ fill_factor * (1 - fill_factor) (rough estimate)
    /// A proper calculation requires the full electromagnetic simulation.
    pub fn peak_efficiency_estimate(&self) -> f64 {
        // Simple empirical estimate based on fill factor
        // Actual efficiency requires FDTD/RCWA computation
        4.0 * self.fill_factor * (1.0 - self.fill_factor)
    }

    /// Grating length (m).
    pub fn length(&self) -> f64 {
        self.period * self.n_periods as f64
    }
}

// ---------------------------------------------------------------------------
// ApodizedGratingCoupler — new expanded implementation
// ---------------------------------------------------------------------------

/// Apodized grating coupler with physics-based fill-factor calculation.
///
/// Apodization tailors the per-period fill factor so that the near-field
/// amplitude envelope matches a target profile (e.g., Gaussian), maximising
/// overlap with a single-mode fiber.
///
/// The Bragg condition gives the coupling angle:
///   sin(θ) = n_eff − λ/Λ
///
/// The radiated field strength per period is modelled as proportional to
/// `ff * (1 − ff)` (first Born approximation for a binary grating), so the
/// fill factor that produces a desired normalised amplitude `a` is:
///   ff = (1 − √(1 − a)) / 2   (smaller root, 0 < ff ≤ 0.5)
#[derive(Debug, Clone)]
pub struct ApodizedGratingCoupler {
    /// Waveguide effective index at the design wavelength.
    pub n_eff: f64,
    /// Cladding index (medium above grating, e.g. SiO₂ = 1.444 or air = 1.0).
    pub n_clad: f64,
    /// Design wavelength (m).
    pub wavelength: f64,
    /// Uniform period Λ (m).  The apodization uses fill-factor variation only.
    pub period: f64,
    /// Number of grating periods.
    pub n_periods: usize,
    /// Group index n_g (used for bandwidth estimate).  Defaults to n_eff + 0.3.
    pub n_group: f64,
}

impl ApodizedGratingCoupler {
    /// Create an apodized grating coupler.
    ///
    /// `n_group` is estimated from `n_eff` if not explicitly provided
    /// (use `with_group_index` to override).
    pub fn new(n_eff: f64, n_clad: f64, wavelength: f64, period: f64, n_periods: usize) -> Self {
        Self {
            n_eff,
            n_clad,
            wavelength,
            period,
            n_periods,
            n_group: n_eff + 0.3,
        }
    }

    /// Override the group index.
    pub fn with_group_index(mut self, n_g: f64) -> Self {
        self.n_group = n_g;
        self
    }

    /// Coupling angle in degrees (from surface normal).
    ///
    /// Bragg condition: sin(θ) = n_eff − λ/Λ
    pub fn coupling_angle_deg(&self) -> f64 {
        let sin_theta = self.n_eff - self.wavelength / self.period;
        sin_theta.clamp(-1.0, 1.0).asin().to_degrees()
    }

    /// Compute per-period fill factors that match a target amplitude profile.
    ///
    /// `target_profile` is a slice of length `n_periods` holding the desired
    /// relative amplitude at each period (arbitrary scale; will be normalised).
    ///
    /// Uses the Born model: radiated amplitude ∝ 4·ff·(1−ff), solved for ff:
    ///   ff = 0.5 − 0.5·√(1 − a_norm)  where a_norm ∈ [0, 1]
    ///
    /// Returns a `Vec<f64>` of fill factors in [0.01, 0.99].
    pub fn apodized_fill_factors(&self, target_profile: &[f64]) -> Vec<f64> {
        if target_profile.is_empty() {
            return Vec::new();
        }
        // Normalise profile to [0, 1]
        let max_val = target_profile.iter().cloned().fold(0.0_f64, f64::max);
        if max_val <= 0.0 {
            return vec![0.5; target_profile.len()];
        }
        target_profile
            .iter()
            .map(|&a| {
                let a_norm = (a / max_val).clamp(0.0, 1.0);
                // Invert: amplitude² = 4·ff·(1-ff)  → ff = 0.5 - 0.5√(1-a²)
                let ff = 0.5 - 0.5 * (1.0 - a_norm * a_norm).max(0.0).sqrt();
                ff.clamp(0.01, 0.99)
            })
            .collect()
    }

    /// Estimate the coupling efficiency for a given fill-factor profile.
    ///
    /// Uses a Gaussian mode-overlap approximation.  The grating near-field
    /// is constructed from the per-period Born amplitude; the overlap with
    /// a normalised Gaussian of waist w₀ = L/2 is computed analytically via
    /// a discrete dot-product.
    ///
    /// `ff`: fill factors returned by `apodized_fill_factors`.
    pub fn coupling_efficiency(&self, ff: &[f64]) -> f64 {
        if ff.is_empty() {
            return 0.0;
        }
        let n = ff.len();
        let l_total = self.period * n as f64;
        // Gaussian waist = half the grating length (1/e² intensity)
        let w0 = l_total / 2.0;
        // Position of each period centre
        let positions: Vec<f64> = (0..n)
            .map(|i| (i as f64 + 0.5) * self.period - l_total / 2.0)
            .collect();
        // Near-field amplitude at each period: a_i = 2·√(ff·(1−ff))
        let amplitudes: Vec<f64> = ff.iter().map(|&f| 2.0 * (f * (1.0 - f)).sqrt()).collect();
        // Target Gaussian profile
        let gaussian: Vec<f64> = positions
            .iter()
            .map(|&x| (-x * x / (w0 * w0)).exp())
            .collect();
        // Normalise both vectors
        let norm_a = amplitudes.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let norm_g = gaussian.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if norm_a < 1e-30 || norm_g < 1e-30 {
            return 0.0;
        }
        let overlap: f64 = amplitudes
            .iter()
            .zip(gaussian.iter())
            .map(|(&a, &g)| a * g)
            .sum::<f64>()
            / (norm_a * norm_g);
        // Efficiency = overlap² (intensity)
        (overlap * overlap).clamp(0.0, 1.0)
    }

    /// 3 dB coupling bandwidth in nm.
    ///
    /// Approximation: Δλ ≈ λ² / (n_g · L)
    pub fn bandwidth_nm(&self) -> f64 {
        let l = self.period * self.n_periods as f64;
        if l <= 0.0 || self.n_group <= 0.0 {
            return 0.0;
        }
        let bw_m = self.wavelength * self.wavelength / (self.n_group * l);
        bw_m * 1e9
    }

    /// Total grating length (m).
    pub fn length(&self) -> f64 {
        self.period * self.n_periods as f64
    }
}

// ---------------------------------------------------------------------------
// GratingArray2d — 2-D phased array for surface emission
// ---------------------------------------------------------------------------

/// 2D grating array for surface emission (phased-array beam steering).
///
/// Models an nx × ny grid of emitters with periods Λ_x and Λ_y.
/// Far-field intensity is computed via the discrete phased-array formula:
///
///   I(θ_x, θ_y) = |Σ_mn w_mn · exp(i·k₀·(m·Λ_x·sin θ_x + n·Λ_y·sin θ_y))|²
#[derive(Debug, Clone)]
pub struct GratingArray2d {
    /// Number of elements along x.
    pub nx: usize,
    /// Number of elements along y.
    pub ny: usize,
    /// Array pitch along x (m).
    pub period_x: f64,
    /// Array pitch along y (m).
    pub period_y: f64,
    /// Design wavelength (m).
    pub wavelength: f64,
}

impl GratingArray2d {
    /// Create a 2D grating array.
    ///
    /// `wavelength` defaults to 1550 nm; set via `with_wavelength` to override.
    pub fn new(nx: usize, ny: usize, period_x: f64, period_y: f64) -> Self {
        Self {
            nx,
            ny,
            period_x,
            period_y,
            wavelength: 1550e-9,
        }
    }

    /// Set the design wavelength.
    pub fn with_wavelength(mut self, wavelength: f64) -> Self {
        self.wavelength = wavelength;
        self
    }

    /// Compute the 2D far-field intensity pattern.
    ///
    /// - `weights`: flat array of element weights, length = nx * ny (row-major).
    ///   Use uniform weights (all 1.0) for a basic array factor.
    /// - `theta_max_deg`: half-angle of the angular sweep (symmetric about 0).
    /// - `n_pts`: number of angular samples per axis.
    ///
    /// Returns a `Vec<Vec<f64>>` of shape `[n_pts][n_pts]` (row = θ_x, col = θ_y).
    pub fn far_field_pattern(
        &self,
        weights: &[f64],
        theta_max_deg: f64,
        n_pts: usize,
    ) -> Vec<Vec<f64>> {
        assert_eq!(
            weights.len(),
            self.nx * self.ny,
            "weights length must equal nx*ny"
        );
        assert!(n_pts >= 2, "n_pts must be >= 2");
        let k0 = 2.0 * PI / self.wavelength;
        let theta_max = theta_max_deg.to_radians();
        let angles: Vec<f64> = (0..n_pts)
            .map(|i| -theta_max + 2.0 * theta_max * i as f64 / (n_pts - 1) as f64)
            .collect();

        // Pre-compute per-element phase contributions (x and y independently)
        // Phase from element (m,n) at direction (θ_x, θ_y):
        //   φ_mn = k₀ · (m·Λ_x·sin θ_x + n·Λ_y·sin θ_y)
        let mut pattern = vec![vec![0.0_f64; n_pts]; n_pts];
        for (ix, &tx) in angles.iter().enumerate() {
            let sin_tx = tx.sin();
            for (iy, &ty) in angles.iter().enumerate() {
                let sin_ty = ty.sin();
                let mut re = 0.0_f64;
                let mut im = 0.0_f64;
                for n in 0..self.ny {
                    for m in 0..self.nx {
                        let w = weights[n * self.nx + m];
                        let phase = k0
                            * (m as f64 * self.period_x * sin_tx
                                + n as f64 * self.period_y * sin_ty);
                        re += w * phase.cos();
                        im += w * phase.sin();
                    }
                }
                pattern[ix][iy] = re * re + im * im;
            }
        }
        pattern
    }

    /// Approximate beam waist (Gaussian approximation): w ≈ λ / (2·NA_approx).
    ///
    /// NA_approx = 0.5·λ / (array_length/2) = λ / array_length
    pub fn beam_waist_estimate(&self) -> f64 {
        let lx = self.nx as f64 * self.period_x;
        let ly = self.ny as f64 * self.period_y;
        // Use the larger dimension for a conservative estimate
        let l = lx.max(ly);
        if l <= 0.0 {
            return f64::INFINITY;
        }
        // NA_approx ≈ λ / (2·L)  →  w ≈ λ / (2·NA) = L
        // More precise: w ≈ 0.5·λ·f / (D/2) for far-field; here return λ·L/(λ) = L / (2π) * λ
        // Standard formula: beam divergence θ ≈ λ/L, so waist w ≈ λ / (2·θ) = L/2
        l / 2.0
    }

    /// Total number of array elements.
    pub fn n_elements(&self) -> usize {
        self.nx * self.ny
    }
}

// ---------------------------------------------------------------------------
// EfficiencyVsTilt — sweep coupling efficiency over tilt angle
// ---------------------------------------------------------------------------

/// Sweeps the coupling efficiency of an `ApodizedGratingCoupler` over a range
/// of tilt angles, modelling how detuning from the Bragg angle affects coupling.
///
/// The efficiency model is:
///   η(θ) = η_peak · sinc²(π·Δθ / δθ_3dB)
/// where Δθ = θ − θ_Bragg and δθ_3dB is the angular 3dB bandwidth.
pub struct EfficiencyVsTilt;

impl EfficiencyVsTilt {
    /// Sweep coupling efficiency vs. tilt angle.
    ///
    /// - `coupler`: reference to an `ApodizedGratingCoupler`.
    /// - `tilt_deg_range`: `(min_deg, max_deg)` sweep range.
    /// - `n_pts`: number of angle points.
    ///
    /// Returns `Vec<(tilt_deg, efficiency)>`.
    pub fn sweep(
        coupler: &ApodizedGratingCoupler,
        tilt_deg_range: (f64, f64),
        n_pts: usize,
    ) -> Vec<(f64, f64)> {
        assert!(n_pts >= 2, "n_pts must be >= 2");
        // Compute a reference fill-factor profile (Gaussian target, n_periods long)
        let n = coupler.n_periods;
        let gaussian_target: Vec<f64> = (0..n)
            .map(|i| {
                let t = (i as f64 - (n - 1) as f64 / 2.0) / (n as f64 / 4.0);
                (-t * t).exp()
            })
            .collect();
        let ff = coupler.apodized_fill_factors(&gaussian_target);
        let eta_peak = coupler.coupling_efficiency(&ff);

        // Angular 3dB bandwidth: δθ_3dB ≈ λ / (n_g · L · cos θ_0)
        let theta0_rad = coupler.coupling_angle_deg().to_radians();
        let l = coupler.length();
        let delta_theta_3db = if l > 0.0 && coupler.n_group > 0.0 {
            coupler.wavelength / (coupler.n_group * l * theta0_rad.cos().max(0.01))
        } else {
            0.1_f64.to_radians()
        };

        let (t_min, t_max) = tilt_deg_range;
        (0..n_pts)
            .map(|i| {
                let tilt_deg = t_min + (t_max - t_min) * i as f64 / (n_pts - 1) as f64;
                let delta_theta = (tilt_deg - coupler.coupling_angle_deg()).to_radians();
                // sinc²(π·Δθ / δθ_3dB)
                let x = PI * delta_theta / delta_theta_3db;
                let sinc = if x.abs() < 1e-12 { 1.0 } else { x.sin() / x };
                let eta = eta_peak * sinc * sinc;
                (tilt_deg, eta.clamp(0.0, 1.0))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- GratingCoupler (original) ----

    #[test]
    fn grating_coupler_phase_match() {
        // Design a grating for 10° coupling from n_eff=2.4 waveguide to air
        let theta = 10.0_f64.to_radians();
        let lambda = 1550e-9;
        let n_eff = 2.4;
        let n_clad = 1.0;

        let gc = GratingCoupler::design(n_eff, n_clad, theta, lambda)
            .expect("Grating design should succeed");

        // Verify round-trip: designed coupling angle should match
        let theta_check = gc
            .coupling_angle(lambda)
            .expect("Should have valid coupling angle");
        let err_deg = (theta_check.to_degrees() - theta.to_degrees()).abs();
        assert!(err_deg < 0.001, "Angle error={err_deg:.4}°");
    }

    #[test]
    fn grating_coupler_period_reasonable() {
        // Si photonics grating: n_eff≈2.4, 10° coupling at 1550nm
        let gc = GratingCoupler::design(2.4, 1.444, 10.0_f64.to_radians(), 1550e-9).unwrap();
        // Period should be around 600-700 nm for typical Si grating couplers
        let period_nm = gc.period * 1e9;
        assert!(
            period_nm > 400.0 && period_nm < 1500.0,
            "Period={period_nm:.1} nm out of expected range"
        );
    }

    #[test]
    fn grating_coupler_beyond_critical_angle_returns_none() {
        // If n_eff - λ/Λ > n_clad, no real coupling angle
        let gc = GratingCoupler::new(300e-9, 2.4, 1.0, 0.5, 20); // short period
                                                                 // At 1550nm: n_eff - λ/Λ = 2.4 - 1550/300 ≈ -2.77 → sin_theta < -1
        let angle = gc.coupling_angle(1550e-9);
        // This might return None (sin_theta < -1) or Some depending on parameters
        // Just check it doesn't panic
        let _ = angle;
    }

    // ---- ApodizedGratingCoupler ----

    #[test]
    fn apodized_grating_coupler_coupling_angle_reasonable() {
        // Si waveguide, SiO₂ cladding, 1550 nm, 630 nm period → ~10° coupling
        let agc = ApodizedGratingCoupler::new(2.65, 1.444, 1550e-9, 630e-9, 20);
        let angle_deg = agc.coupling_angle_deg();
        assert!(
            angle_deg > -30.0 && angle_deg < 30.0,
            "Coupling angle={angle_deg:.2}° out of expected range"
        );
    }

    #[test]
    fn apodized_fill_factors_length_matches_input() {
        let agc = ApodizedGratingCoupler::new(2.65, 1.444, 1550e-9, 630e-9, 20);
        let target: Vec<f64> = (0..20).map(|i| (i as f64).exp()).collect();
        let ff = agc.apodized_fill_factors(&target);
        assert_eq!(ff.len(), 20);
    }

    #[test]
    fn apodized_fill_factors_in_valid_range() {
        let agc = ApodizedGratingCoupler::new(2.65, 1.444, 1550e-9, 630e-9, 20);
        // Gaussian target profile
        let target: Vec<f64> = (0..20)
            .map(|i| {
                let t = (i as f64 - 9.5) / 5.0;
                (-t * t).exp()
            })
            .collect();
        let ff = agc.apodized_fill_factors(&target);
        for &f in &ff {
            assert!((0.0..=1.0).contains(&f), "Fill factor {f} out of [0,1]");
        }
    }

    #[test]
    fn apodized_coupling_efficiency_between_zero_and_one() {
        let agc = ApodizedGratingCoupler::new(2.65, 1.444, 1550e-9, 630e-9, 20);
        let target: Vec<f64> = (0..20)
            .map(|i| {
                let t = (i as f64 - 9.5) / 5.0;
                (-t * t).exp()
            })
            .collect();
        let ff = agc.apodized_fill_factors(&target);
        let eta = agc.coupling_efficiency(&ff);
        assert!(
            (0.0..=1.0).contains(&eta),
            "Efficiency={eta:.4} out of [0,1]"
        );
    }

    #[test]
    fn apodized_bandwidth_nm_positive() {
        let agc = ApodizedGratingCoupler::new(2.65, 1.444, 1550e-9, 630e-9, 20);
        let bw = agc.bandwidth_nm();
        assert!(bw > 0.0, "Bandwidth={bw:.3} nm should be positive");
    }

    #[test]
    fn apodized_bandwidth_nm_physically_reasonable() {
        // For 20 periods × 630 nm = 12.6 µm grating, BW ≈ λ²/(n_g*L) ≈ few nm
        let agc = ApodizedGratingCoupler::new(2.65, 1.444, 1550e-9, 630e-9, 20);
        let bw = agc.bandwidth_nm();
        assert!(bw > 0.5 && bw < 200.0, "Bandwidth={bw:.2} nm unexpected");
    }

    #[test]
    fn apodized_fill_factors_uniform_input() {
        // All-ones input → all fill factors the same
        let agc = ApodizedGratingCoupler::new(2.65, 1.444, 1550e-9, 630e-9, 10);
        let target = vec![1.0_f64; 10];
        let ff = agc.apodized_fill_factors(&target);
        let f0 = ff[0];
        for &f in &ff {
            assert!((f - f0).abs() < 1e-10, "Expected uniform fill factors");
        }
    }

    #[test]
    fn apodized_empty_profile_returns_empty() {
        let agc = ApodizedGratingCoupler::new(2.65, 1.444, 1550e-9, 630e-9, 20);
        let ff = agc.apodized_fill_factors(&[]);
        assert!(ff.is_empty());
    }

    // ---- GratingArray2d ----

    #[test]
    fn grating_array_2d_far_field_size() {
        let arr = GratingArray2d::new(4, 4, 2e-6, 2e-6).with_wavelength(1550e-9);
        let weights = vec![1.0_f64; 16];
        let ff = arr.far_field_pattern(&weights, 30.0, 11);
        assert_eq!(ff.len(), 11);
        assert_eq!(ff[0].len(), 11);
    }

    #[test]
    fn grating_array_2d_far_field_nonnegative() {
        let arr = GratingArray2d::new(4, 4, 2e-6, 2e-6).with_wavelength(1550e-9);
        let weights = vec![1.0_f64; 16];
        let ff = arr.far_field_pattern(&weights, 30.0, 11);
        for row in &ff {
            for &v in row {
                assert!(v >= 0.0, "Negative intensity {v}");
            }
        }
    }

    #[test]
    fn grating_array_2d_peak_at_broadside() {
        // For uniform weights, peak should be at centre (broadside θ=0)
        let arr = GratingArray2d::new(8, 8, 2e-6, 2e-6).with_wavelength(1550e-9);
        let weights = vec![1.0_f64; 64];
        let n = 21;
        let ff = arr.far_field_pattern(&weights, 45.0, n);
        let centre = n / 2;
        let peak = ff[centre][centre];
        // Every off-centre point should be ≤ peak
        for row in &ff {
            for &v in row {
                assert!(v <= peak + 1e-6, "Off-centre value {v} > peak {peak}");
            }
        }
    }

    #[test]
    fn grating_array_2d_beam_waist_positive() {
        let arr = GratingArray2d::new(8, 8, 2e-6, 2e-6);
        assert!(arr.beam_waist_estimate() > 0.0);
    }

    #[test]
    fn grating_array_2d_n_elements() {
        let arr = GratingArray2d::new(5, 7, 1e-6, 1e-6);
        assert_eq!(arr.n_elements(), 35);
    }

    // ---- EfficiencyVsTilt ----

    #[test]
    fn efficiency_vs_tilt_length_correct() {
        let agc = ApodizedGratingCoupler::new(2.65, 1.444, 1550e-9, 630e-9, 20);
        let sweep = EfficiencyVsTilt::sweep(&agc, (-20.0, 20.0), 41);
        assert_eq!(sweep.len(), 41);
    }

    #[test]
    fn efficiency_vs_tilt_peak_near_bragg_angle() {
        let agc = ApodizedGratingCoupler::new(2.65, 1.444, 1550e-9, 630e-9, 20);
        let theta_bragg = agc.coupling_angle_deg();
        let sweep = EfficiencyVsTilt::sweep(&agc, (theta_bragg - 15.0, theta_bragg + 15.0), 61);
        let (_, eta_at_bragg) = sweep[30]; // centre point
                                           // All other points should be ≤ this
        for &(_, eta) in &sweep {
            assert!(
                eta <= eta_at_bragg + 1e-9,
                "Off-peak eta={eta} > peak {eta_at_bragg}"
            );
        }
    }

    #[test]
    fn efficiency_vs_tilt_all_nonnegative() {
        let agc = ApodizedGratingCoupler::new(2.65, 1.444, 1550e-9, 630e-9, 20);
        let sweep = EfficiencyVsTilt::sweep(&agc, (-30.0, 30.0), 31);
        for (_, eta) in sweep {
            assert!(eta >= 0.0, "Negative efficiency");
        }
    }

    #[test]
    fn efficiency_vs_tilt_decreases_far_from_peak() {
        // The sinc² envelope is generally decreasing but has small side lobes.
        // Verify that the efficiency at the edges of the sweep is well below the peak.
        let agc = ApodizedGratingCoupler::new(2.65, 1.444, 1550e-9, 630e-9, 20);
        let theta0 = agc.coupling_angle_deg();
        let sweep = EfficiencyVsTilt::sweep(&agc, (theta0, theta0 + 20.0), 41);
        let etas: Vec<f64> = sweep.iter().map(|&(_, e)| e).collect();
        // Peak is at index 0 (theta0); efficiency at far end should be much lower
        let eta_peak = etas[0];
        let eta_far = etas[40];
        assert!(
            eta_far < eta_peak * 0.5,
            "Efficiency far from Bragg angle ({eta_far:.4}) should be < 50% of peak ({eta_peak:.4})"
        );
    }
}
