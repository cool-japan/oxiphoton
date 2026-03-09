//! Optical phased array (OPA) beam-steering theory.
//!
//! Covers:
//! - 1D and 2D uniform linear arrays with programmable phase control
//! - Grating-lobe analysis, scan range, HPBW, peak gain
//! - Phase quantisation error for B-bit DAC control
//!
//! All angles in radians internally; degrees available via helper methods.
//! All spatial quantities in metres.

use std::f64::consts::PI;

// ─── OpticalPhasedArray1d ────────────────────────────────────────────────────

/// One-dimensional optical phased array with N uniformly spaced emitters.
///
/// Each emitter has an independently programmable phase φ_n.  The far-field
/// array factor (AF) is the discrete-Fourier-transform relationship between
/// element phases and the angular spectrum:
///
///   AF(θ) = (1/N) Σ_{n=0}^{N-1} exp( i (φ_n + n k d sin θ) )
///
/// where k = 2π/λ and d is the element pitch.
#[derive(Debug, Clone)]
pub struct OpticalPhasedArray1d {
    /// Number of emitting elements
    pub n_elements: usize,
    /// Element pitch d (m) — should satisfy d ≤ λ/2 to suppress grating lobes
    pub pitch_m: f64,
    /// Free-space wavelength λ (m)
    pub wavelength_m: f64,
    /// Phase settings φ_n (rad) — length must equal n_elements
    pub phases: Vec<f64>,
}

impl OpticalPhasedArray1d {
    /// Construct an OPA with all phases initialised to zero (broadside beam).
    pub fn new(n_elements: usize, pitch_m: f64, wavelength_m: f64) -> Self {
        Self {
            n_elements,
            pitch_m,
            wavelength_m,
            phases: vec![0.0; n_elements],
        }
    }

    /// Wave number in free space: k = 2π / λ
    pub fn wave_number(&self) -> f64 {
        2.0 * PI / self.wavelength_m
    }

    /// Set phase taper to steer the main beam to angle θ (rad from boresight).
    ///
    /// The phase of element n is set to:
    ///
    ///   φ_n = −n k d sin θ
    ///
    /// This creates a linear progressive phase shift that shifts the array
    /// factor peak from broadside (θ = 0) to the desired steering angle.
    pub fn steer_to_angle(&mut self, theta_rad: f64) {
        let k = self.wave_number();
        let d = self.pitch_m;
        for (n, phi) in self.phases.iter_mut().enumerate() {
            *phi = -(n as f64) * k * d * theta_rad.sin();
        }
    }

    /// Array factor magnitude |AF(θ)| ∈ \[0, 1\].
    ///
    /// Computed by coherent summation of phasors from all N elements:
    ///
    ///   AF(θ) = |Σ_n exp(i (φ_n + n k d sin θ))| / N
    pub fn array_factor(&self, theta_rad: f64) -> f64 {
        if self.n_elements == 0 {
            return 0.0;
        }
        let k = self.wave_number();
        let d = self.pitch_m;
        let (mut re_sum, mut im_sum) = (0.0_f64, 0.0_f64);

        for (n, &phi_n) in self.phases.iter().enumerate() {
            let psi = phi_n + (n as f64) * k * d * theta_rad.sin();
            re_sum += psi.cos();
            im_sum += psi.sin();
        }

        let n = self.n_elements as f64;
        (re_sum * re_sum + im_sum * im_sum).sqrt() / n
    }

    /// Far-field beam intensity pattern (normalised).
    ///
    /// P(θ) = |AF(θ)|²
    pub fn beam_pattern(&self, theta_rad: f64) -> f64 {
        self.array_factor(theta_rad).powi(2)
    }

    /// Grating-lobe indicator: `true` when d > λ/2 (aliases will appear).
    pub fn has_grating_lobes(&self) -> bool {
        self.pitch_m > self.wavelength_m / 2.0
    }

    /// Half-power beamwidth (HPBW) in radians for a uniformly illuminated array.
    ///
    /// Δθ_HPBW ≈ 0.886 × λ / (N d)
    pub fn hpbw_rad(&self) -> f64 {
        0.886 * self.wavelength_m / (self.n_elements as f64 * self.pitch_m)
    }

    /// HPBW in degrees.
    pub fn hpbw_deg(&self) -> f64 {
        self.hpbw_rad().to_degrees()
    }

    /// Grating-lobe-free scan range ±|θ_max| (radians).
    ///
    ///   |θ_max| = arcsin( λ/(2d) − 1/N )
    ///
    /// Returns π/2 if the argument exceeds 1 (full hemispheric coverage).
    pub fn scan_range_rad(&self) -> f64 {
        let arg = self.wavelength_m / (2.0 * self.pitch_m) - 1.0 / self.n_elements as f64;
        if arg >= 1.0 {
            PI / 2.0
        } else if arg <= -1.0 {
            0.0
        } else {
            arg.asin()
        }
    }

    /// Scan range in degrees.
    pub fn scan_range_deg(&self) -> f64 {
        self.scan_range_rad().to_degrees()
    }

    /// Peak array gain: G_peak = N² × element_gain (for uniform excitation).
    ///
    /// * `element_gain` — gain of a single element (linear, not dB)
    pub fn peak_gain(&self, element_gain: f64) -> f64 {
        (self.n_elements as f64).powi(2) * element_gain
    }

    /// Side-lobe level (SLL) for uniform amplitude illumination: −13.3 dB.
    ///
    /// This is the classic Chebyshev / sinc sidelobe level for a rectangular
    /// aperture weighting (no taper).
    pub fn side_lobe_level_db(&self) -> f64 {
        -13.3
    }

    /// Phase quantisation step for a B-bit DAC: Δφ = 2π / 2^B.
    ///
    /// * `bits` — number of bits in the phase-shift DAC
    pub fn phase_quantization_error_rad(&self, bits: u32) -> f64 {
        if bits >= 63 {
            return f64::MIN_POSITIVE;
        }
        2.0 * PI / (1u64 << bits) as f64
    }

    /// Current beam-steering angle derived from the programmed phase gradient.
    ///
    ///   θ = arcsin( λ / (2π d) × Δφ/Δn )
    ///
    /// Assumes a linear phase taper; uses the first phase difference as the
    /// representative gradient.
    pub fn current_steering_angle_rad(&self) -> f64 {
        if self.phases.len() < 2 {
            return 0.0;
        }
        let dphi_dn = self.phases[1] - self.phases[0]; // linear gradient step
        let arg = (self.wavelength_m / (2.0 * PI * self.pitch_m) * dphi_dn).clamp(-1.0, 1.0);
        arg.asin()
    }

    /// Scan the beam across a set of angles and return (angle, intensity) pairs.
    ///
    /// * `n_points` — number of uniformly spaced angles in \[−π/2, +π/2\]
    pub fn scan_pattern(&self, n_points: usize) -> Vec<(f64, f64)> {
        if n_points == 0 {
            return Vec::new();
        }
        (0..n_points)
            .map(|i| {
                let theta = -PI / 2.0 + PI * i as f64 / (n_points - 1).max(1) as f64;
                (theta, self.beam_pattern(theta))
            })
            .collect()
    }

    /// Directivity of the phased array (assuming isotropic elements):
    ///
    ///   D ≈ 2 N d / λ  (endfire / broadside approximation)
    pub fn directivity_approx(&self) -> f64 {
        2.0 * self.n_elements as f64 * self.pitch_m / self.wavelength_m
    }
}

// ─── OpticalPhasedArray2d ────────────────────────────────────────────────────

/// Two-dimensional Nx × Ny optical phased array.
///
/// Extends the 1D formulation to two orthogonal directions.  The array factor
/// separates into a product of 1D array factors when the phase is set as a
/// separable function φ_{nx,ny} = φ_x(nx) + φ_y(ny):
///
///   AF(θ_x, θ_y) = AF_x(θ_x) × AF_y(θ_y)
#[derive(Debug, Clone)]
pub struct OpticalPhasedArray2d {
    /// Number of columns (x-direction)
    pub nx: usize,
    /// Number of rows (y-direction)
    pub ny: usize,
    /// Pitch in x (m)
    pub pitch_x_m: f64,
    /// Pitch in y (m)
    pub pitch_y_m: f64,
    /// Free-space wavelength (m)
    pub wavelength_m: f64,
    /// Phase array \[ny\]\[nx\] (radians)
    pub phases: Vec<Vec<f64>>,
}

impl OpticalPhasedArray2d {
    /// Construct a 2D OPA with all phases set to zero.
    pub fn new(nx: usize, ny: usize, pitch_x_m: f64, pitch_y_m: f64, wavelength_m: f64) -> Self {
        let phases = vec![vec![0.0_f64; nx]; ny];
        Self {
            nx,
            ny,
            pitch_x_m,
            pitch_y_m,
            wavelength_m,
            phases,
        }
    }

    /// Wave number k = 2π/λ
    pub fn wave_number(&self) -> f64 {
        2.0 * PI / self.wavelength_m
    }

    /// Set phases to steer the main beam toward (θ_x, θ_y).
    ///
    /// Phase of element (nx, ny):
    ///
    ///   φ(nx, ny) = − nx k dx sin(θ_x) − ny k dy sin(θ_y)
    pub fn steer_to_angle(&mut self, theta_x_rad: f64, theta_y_rad: f64) {
        let k = self.wave_number();
        let dx = self.pitch_x_m;
        let dy = self.pitch_y_m;
        for iy in 0..self.ny {
            for ix in 0..self.nx {
                self.phases[iy][ix] = -(ix as f64) * k * dx * theta_x_rad.sin()
                    - (iy as f64) * k * dy * theta_y_rad.sin();
            }
        }
    }

    /// 2D array factor magnitude.
    ///
    ///   AF(θ_x, θ_y) = |Σ_{nx,ny} exp(i (φ_{nx,ny} + nx k dx sin θ_x + ny k dy sin θ_y))| / (Nx Ny)
    pub fn array_factor(&self, theta_x: f64, theta_y: f64) -> f64 {
        if self.nx == 0 || self.ny == 0 {
            return 0.0;
        }
        let k = self.wave_number();
        let dx = self.pitch_x_m;
        let dy = self.pitch_y_m;
        let (mut re_sum, mut im_sum) = (0.0_f64, 0.0_f64);

        for (iy, row) in self.phases.iter().enumerate() {
            for (ix, &phi) in row.iter().enumerate() {
                let psi = phi
                    + (ix as f64) * k * dx * theta_x.sin()
                    + (iy as f64) * k * dy * theta_y.sin();
                re_sum += psi.cos();
                im_sum += psi.sin();
            }
        }

        let n_total = (self.nx * self.ny) as f64;
        (re_sum * re_sum + im_sum * im_sum).sqrt() / n_total
    }

    /// Total element count: Nx × Ny
    pub fn total_elements(&self) -> usize {
        self.nx * self.ny
    }

    /// HPBW in x (radians): Δθ_x ≈ 0.886 λ / (Nx dx)
    pub fn hpbw_x_rad(&self) -> f64 {
        0.886 * self.wavelength_m / (self.nx as f64 * self.pitch_x_m)
    }

    /// HPBW in y (radians): Δθ_y ≈ 0.886 λ / (Ny dy)
    pub fn hpbw_y_rad(&self) -> f64 {
        0.886 * self.wavelength_m / (self.ny as f64 * self.pitch_y_m)
    }

    /// Scan range in x (radians, grating-lobe free):
    ///   θ_max_x = arcsin( λ/(2 dx) − 1/Nx )
    pub fn scan_range_x_rad(&self) -> f64 {
        let arg = self.wavelength_m / (2.0 * self.pitch_x_m) - 1.0 / self.nx as f64;
        if arg >= 1.0 {
            PI / 2.0
        } else if arg <= -1.0 {
            0.0
        } else {
            arg.asin()
        }
    }

    /// Scan range in y (radians)
    pub fn scan_range_y_rad(&self) -> f64 {
        let arg = self.wavelength_m / (2.0 * self.pitch_y_m) - 1.0 / self.ny as f64;
        if arg >= 1.0 {
            PI / 2.0
        } else if arg <= -1.0 {
            0.0
        } else {
            arg.asin()
        }
    }

    /// Number of resolvable spots across the 2D scan field.
    ///
    ///   N_spots = (2 θ_max_x / Δθ_x) × (2 θ_max_y / Δθ_y)
    pub fn n_resolvable_spots(&self) -> usize {
        let nx = (2.0 * self.scan_range_x_rad() / self.hpbw_x_rad()).round() as usize;
        let ny = (2.0 * self.scan_range_y_rad() / self.hpbw_y_rad()).round() as usize;
        nx.max(1) * ny.max(1)
    }

    /// Grating-lobe indicator for x-direction
    pub fn has_grating_lobes_x(&self) -> bool {
        self.pitch_x_m > self.wavelength_m / 2.0
    }

    /// Grating-lobe indicator for y-direction
    pub fn has_grating_lobes_y(&self) -> bool {
        self.pitch_y_m > self.wavelength_m / 2.0
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opa_steer_to_10_degrees() {
        let mut opa = OpticalPhasedArray1d::new(32, 775.0e-9, 1550.0e-9); // d = λ/2
        let theta = 10_f64.to_radians();
        opa.steer_to_angle(theta);
        let af_at_theta = opa.array_factor(theta);
        let af_at_zero = opa.array_factor(0.0);
        assert!(
            af_at_theta > af_at_zero * 0.9,
            "Beam not at θ=10°: AF(θ)={:.4}, AF(0)={:.4}",
            af_at_theta,
            af_at_zero
        );
    }

    #[test]
    fn opa_broadside_array_factor_unity() {
        let opa = OpticalPhasedArray1d::new(16, 775.0e-9, 1550.0e-9);
        // All phases zero → broadside → AF(0) = 1.0
        let af = opa.array_factor(0.0);
        assert!((af - 1.0).abs() < 1.0e-10, "Broadside AF must be 1.0: {af}");
    }

    #[test]
    fn opa_no_grating_lobes_at_half_wavelength_pitch() {
        let opa = OpticalPhasedArray1d::new(32, 775.0e-9, 1550.0e-9);
        assert!(
            !opa.has_grating_lobes(),
            "d=λ/2 should not produce grating lobes"
        );
    }

    #[test]
    fn opa_grating_lobes_at_full_wavelength_pitch() {
        let opa = OpticalPhasedArray1d::new(32, 1550.0e-9, 1550.0e-9);
        assert!(opa.has_grating_lobes(), "d=λ should produce grating lobes");
    }

    #[test]
    fn opa_hpbw_decreases_with_more_elements() {
        let opa8 = OpticalPhasedArray1d::new(8, 775.0e-9, 1550.0e-9);
        let opa32 = OpticalPhasedArray1d::new(32, 775.0e-9, 1550.0e-9);
        assert!(
            opa32.hpbw_rad() < opa8.hpbw_rad(),
            "Larger array must have narrower beam"
        );
    }

    #[test]
    fn opa_peak_gain_quadratic_in_n() {
        let n = 16_usize;
        let opa = OpticalPhasedArray1d::new(n, 775.0e-9, 1550.0e-9);
        let g = opa.peak_gain(1.0);
        let expected = (n * n) as f64;
        assert!(
            (g - expected).abs() < 1.0e-10,
            "Peak gain = N²: {g} vs {expected}"
        );
    }

    #[test]
    fn opa2d_total_elements() {
        let opa = OpticalPhasedArray2d::new(8, 8, 775.0e-9, 775.0e-9, 1550.0e-9);
        assert_eq!(opa.total_elements(), 64);
    }

    #[test]
    fn opa2d_broadside_af_unity() {
        let opa = OpticalPhasedArray2d::new(8, 8, 775.0e-9, 775.0e-9, 1550.0e-9);
        let af = opa.array_factor(0.0, 0.0);
        assert!(
            (af - 1.0).abs() < 1.0e-10,
            "2D broadside AF must be 1.0: {af}"
        );
    }

    #[test]
    fn opa2d_steer_increases_af_at_target() {
        let mut opa = OpticalPhasedArray2d::new(16, 16, 775.0e-9, 775.0e-9, 1550.0e-9);
        let tx = 5_f64.to_radians();
        let ty = 5_f64.to_radians();
        opa.steer_to_angle(tx, ty);
        let af_target = opa.array_factor(tx, ty);
        let af_boresight = opa.array_factor(0.0, 0.0);
        assert!(
            af_target > af_boresight * 0.85,
            "2D beam not at target: AF(tgt)={:.4} AF(0,0)={:.4}",
            af_target,
            af_boresight
        );
    }
}
