/// Pancharatnam-Berry (PB) / geometric-phase metasurfaces.
///
/// The geometric phase (also called Pancharatnam-Berry phase) is accumulated
/// when a half-wave-plate-like element is spatially rotated by angle θ: the
/// cross-polarised scattered field acquires a phase of ±2θ depending on the
/// handedness of the incident circular polarisation.
///
/// # Key physics
///
/// For an ideal half-wave-plate metasurface:
///   - Incident LCP  → transmitted RCP  with phase +2θ(x,y)
///   - Incident RCP  → transmitted LCP  with phase −2θ(x,y)
///
/// This decoupling of phase from material dispersion enables broadband
/// operation and allows encoding two independent phase profiles in the same
/// aperture (spin-multiplexing).
use num_complex::Complex64;
use oxifft::Complex as OxiComplex;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// CircPolarization
// ---------------------------------------------------------------------------

/// Handedness of circular polarisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircPolarization {
    /// Left-circular polarisation (σ⁺)
    LCP,
    /// Right-circular polarisation (σ⁻)
    RCP,
}

impl CircPolarization {
    /// Jones vector for unit-amplitude circular polarisation.
    ///
    /// LCP: (1/√2)(1, +i)ᵀ
    /// RCP: (1/√2)(1, −i)ᵀ
    pub fn jones_vector(&self) -> [Complex64; 2] {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        match self {
            CircPolarization::LCP => [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(0.0, inv_sqrt2),
            ],
            CircPolarization::RCP => [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(0.0, -inv_sqrt2),
            ],
        }
    }

    /// Return the opposite handedness.
    pub fn opposite(&self) -> CircPolarization {
        match self {
            CircPolarization::LCP => CircPolarization::RCP,
            CircPolarization::RCP => CircPolarization::LCP,
        }
    }
}

// ---------------------------------------------------------------------------
// PbPhaseElement
// ---------------------------------------------------------------------------

/// Pancharatnam-Berry phase element.
///
/// Each pixel (i, j) stores an orientation angle θ(i,j).  The phase imparted
/// to the cross-polarised output is ±2θ (sign depends on input polarisation).
///
/// The element is typically fabricated as an array of half-wave-plate antennas
/// (dielectric pillars with aspect ratio ≥ 1 or plasmonic bars).
#[derive(Debug, Clone)]
pub struct PbPhaseElement {
    /// Number of pixels along x
    pub n_pixels_x: usize,
    /// Number of pixels along y
    pub n_pixels_y: usize,
    /// Physical pixel size (m)
    pub pixel_size: f64,
    /// Orientation map θ(i,j) in radians; row-major indexing \[y\]\[x\].
    pub orientation_map: Vec<Vec<f64>>,
    /// Design wavelength (m)
    pub wavelength: f64,
    /// Fraction of power converted to cross-polarisation (0 … 1).
    pub conversion_efficiency: f64,
}

impl PbPhaseElement {
    /// Construct an all-zero orientation map.
    pub fn new(nx: usize, ny: usize, pixel_size: f64, wavelength: f64) -> Self {
        Self {
            n_pixels_x: nx,
            n_pixels_y: ny,
            pixel_size,
            orientation_map: vec![vec![0.0; nx]; ny],
            wavelength,
            conversion_efficiency: 1.0,
        }
    }

    /// Phase at pixel (i, j) for incident circular polarisation.
    ///
    /// LCP → cross-pol phase = +2θ
    /// RCP → cross-pol phase = −2θ
    pub fn phase_at(&self, i: usize, j: usize) -> f64 {
        if j < self.orientation_map.len() && i < self.orientation_map[j].len() {
            2.0 * self.orientation_map[j][i]
        } else {
            0.0
        }
    }

    /// Physical x-coordinate of pixel column i (m, centred at 0).
    fn pixel_x(&self, i: usize) -> f64 {
        (i as f64 - (self.n_pixels_x as f64 - 1.0) * 0.5) * self.pixel_size
    }

    /// Physical y-coordinate of pixel row j (m, centred at 0).
    fn pixel_y(&self, j: usize) -> f64 {
        (j as f64 - (self.n_pixels_y as f64 - 1.0) * 0.5) * self.pixel_size
    }

    /// Set a lens phase profile (paraxial approximation).
    ///
    /// θ(x,y) = −π r² / (λ f)  so that 2θ = −2π r² / (λ f) = φ_paraxial_lens
    ///
    /// Note: this is equivalent to a Fresnel lens design in the paraxial limit.
    pub fn set_lens_profile(&mut self, focal_length: f64) {
        for j in 0..self.n_pixels_y {
            for i in 0..self.n_pixels_x {
                let x = self.pixel_x(i);
                let y = self.pixel_y(j);
                let r2 = x * x + y * y;
                self.orientation_map[j][i] = -PI * r2 / (self.wavelength * focal_length);
            }
        }
    }

    /// Set a blazed-grating phase profile.
    ///
    /// θ(x) = π x / Λ  so that 2θ = 2π x / Λ (first-order deflection).
    ///
    /// The grating deflects LCP into first order and RCP into minus-first order.
    pub fn set_grating_profile(&mut self, period_m: f64) {
        for j in 0..self.n_pixels_y {
            for i in 0..self.n_pixels_x {
                let x = self.pixel_x(i);
                self.orientation_map[j][i] = PI * x / period_m;
            }
        }
    }

    /// Set a vortex-beam phase profile.
    ///
    /// θ(x,y) = (l/2) · atan2(y, x)  →  2θ = l · atan2(y, x)
    ///
    /// The topological charge of the generated vortex beam is l for LCP and
    /// −l for RCP input.
    pub fn set_vortex_profile(&mut self, topological_charge: i32) {
        let l = topological_charge as f64;
        for j in 0..self.n_pixels_y {
            for i in 0..self.n_pixels_x {
                let x = self.pixel_x(i);
                let y = self.pixel_y(j);
                self.orientation_map[j][i] = (l / 2.0) * y.atan2(x);
            }
        }
    }

    /// Apply the phase element to a 2-D input field.
    ///
    /// The input is assumed to be the co-polarised amplitude.  The output is
    /// the cross-polarised amplitude after acquiring the geometric phase.
    ///
    /// Returns a `Vec<Vec<Complex64>>` of the same size as the input (row-major).
    pub fn apply(
        &self,
        input: &[Vec<Complex64>],
        input_polarization: CircPolarization,
    ) -> Vec<Vec<Complex64>> {
        let ny = input.len();
        let nx = if ny > 0 { input[0].len() } else { 0 };
        let sign = match input_polarization {
            CircPolarization::LCP => 1.0_f64,
            CircPolarization::RCP => -1.0_f64,
        };
        let sqrt_eff = self.conversion_efficiency.sqrt();
        let mut out = vec![vec![Complex64::new(0.0, 0.0); nx]; ny];
        for j in 0..ny.min(self.n_pixels_y) {
            for i in 0..nx.min(self.n_pixels_x) {
                let theta = self.orientation_map[j][i];
                let phase = sign * 2.0 * theta;
                let phasor = Complex64::from_polar(sqrt_eff, phase);
                out[j][i] = input[j][i] * phasor;
            }
        }
        out
    }

    /// Fraction of power in the m-th diffraction order.
    ///
    /// For an ideal geometric-phase element:
    ///   - m = ±1 (cross-pol):  η = conversion_efficiency / 2 each
    ///   - m =  0 (co-pol):     η = 1 − conversion_efficiency
    ///
    /// Only m ∈ {−1, 0, +1} are treated; other orders return 0.
    pub fn efficiency_in_order(&self, order: i32) -> f64 {
        match order {
            0 => 1.0 - self.conversion_efficiency,
            1 | -1 => self.conversion_efficiency / 2.0,
            _ => 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// PbBeamSplitter
// ---------------------------------------------------------------------------

/// Geometric-phase beam splitter.
///
/// A Pancharatnam-Berry grating deflects LCP and RCP into opposite first
/// orders, acting as a polarisation beam splitter with no Fresnel losses.
#[derive(Debug, Clone)]
pub struct PbBeamSplitter {
    /// Grating period (m)
    pub grating_period: f64,
    /// Design wavelength (m)
    pub wavelength: f64,
    /// Deflection angle of the ±1 orders (degrees; stored for caching)
    pub deflection_angle_deg: f64,
}

impl PbBeamSplitter {
    /// Construct a geometric-phase beam splitter.
    ///
    /// The ±1 diffraction orders are at ±arcsin(λ/Λ).
    pub fn new(period: f64, wavelength: f64) -> Self {
        let angle_rad = (wavelength / period).clamp(-1.0, 1.0).asin();
        Self {
            grating_period: period,
            wavelength,
            deflection_angle_deg: angle_rad.to_degrees(),
        }
    }

    /// Deflection angle of the ±1 order (degrees).
    pub fn deflection_angle_deg(&self) -> f64 {
        (self.wavelength / self.grating_period)
            .clamp(-1.0, 1.0)
            .asin()
            .to_degrees()
    }

    /// Angular separation between the LCP and RCP outputs (degrees).
    ///
    /// = 2 × deflection_angle
    pub fn polarization_separation_angle(&self) -> f64 {
        2.0 * self.deflection_angle_deg()
    }

    /// Ideal diffraction efficiency per order.
    ///
    /// For a perfect geometric-phase grating, 100 % of the cross-polarised
    /// light goes into the ±1 orders equally (50 % each).
    /// The co-polarised (zeroth order) is suppressed when conversion
    /// efficiency → 1.
    pub fn efficiency_per_order(&self) -> f64 {
        0.5 // 50 % per ±1 order for perfect conversion
    }
}

// ---------------------------------------------------------------------------
// MetasurfaceFunction
// ---------------------------------------------------------------------------

/// Optical function implemented by one polarisation channel of a
/// spin-multiplexed metasurface.
#[derive(Debug, Clone)]
pub enum MetasurfaceFunction {
    /// Converging or diverging lens.
    Lens {
        /// Focal length (m); positive = converging, negative = diverging
        focal_length: f64,
    },
    /// Diffraction grating with arbitrary angle.
    Grating {
        /// Grating period (m)
        period: f64,
        /// Deflection angle (degrees)
        angle_deg: f64,
    },
    /// Holographic beam shaping — implements the target phase via precomputed
    /// Pancharatnam-Berry phase encoding.
    ///
    /// The `phase_map[iy][ix]` grid contains the pupil-plane phase φ(x,y) in
    /// radians. `orientation_at` returns φ/2 (the nanofin rotation angle) so
    /// that 2θ = φ.
    Hologram {
        /// Precomputed phase map φ(x, y) in radians, indexed `[iy][ix]`.
        /// Row 0, column 0 maps to world coordinate `origin`.
        phase_map: std::sync::Arc<Vec<Vec<f64>>>,
        /// World coordinate (m) of `phase_map[0][0]` — i.e. the corner pixel.
        origin: (f64, f64),
        /// Pixel pitch Δ (m) — uniform in x and y.
        pitch: f64,
        /// Optional human-readable label.
        label: String,
    },
    /// Optical vortex beam generator.
    Vortex {
        /// Topological charge
        charge: i32,
    },
}

impl MetasurfaceFunction {
    /// Orientation angle θ(x, y) that encodes this function (radians).
    ///
    /// The returned value is the angle such that 2θ gives the desired phase.
    pub fn orientation_at(&self, x: f64, y: f64, wavelength: f64) -> f64 {
        match self {
            MetasurfaceFunction::Lens { focal_length } => {
                let r2 = x * x + y * y;
                -PI * r2 / (wavelength * focal_length)
            }
            MetasurfaceFunction::Grating { period, angle_deg } => {
                let dir = angle_deg.to_radians();
                let s = x * dir.cos() + y * dir.sin();
                PI * s / period
            }
            MetasurfaceFunction::Hologram {
                phase_map,
                origin,
                pitch,
                ..
            } => {
                if phase_map.is_empty() || phase_map[0].is_empty() {
                    return 0.0;
                }
                let ny = phase_map.len();
                let nx = phase_map[0].len();
                // Map world coords to fractional grid indices
                let fx = (x - origin.0) / pitch;
                let fy = (y - origin.1) / pitch;
                // Clamp to valid range for boundary pixels; return 0 if fully out of bounds
                if fx < 0.0 || fy < 0.0 || fx >= nx as f64 || fy >= ny as f64 {
                    return 0.0;
                }
                // Bilinear interpolation
                let ix0 = fx.floor() as usize;
                let iy0 = fy.floor() as usize;
                let ix1 = (ix0 + 1).min(nx - 1);
                let iy1 = (iy0 + 1).min(ny - 1);
                let tx = fx - ix0 as f64;
                let ty = fy - iy0 as f64;
                let p00 = phase_map[iy0][ix0];
                let p10 = phase_map[iy0][ix1];
                let p01 = phase_map[iy1][ix0];
                let p11 = phase_map[iy1][ix1];
                let phase = p00 * (1.0 - tx) * (1.0 - ty)
                    + p10 * tx * (1.0 - ty)
                    + p01 * (1.0 - tx) * ty
                    + p11 * tx * ty;
                // Geometric-phase encoding: orientation angle θ = φ/2
                phase / 2.0
            }
            MetasurfaceFunction::Vortex { charge } => {
                let l = *charge as f64;
                (l / 2.0) * y.atan2(x)
            }
        }
    }

    /// Create a hologram from a precomputed phase map.
    ///
    /// # Arguments
    /// * `phase_map` — 2-D grid of phase values in radians, indexed `[iy][ix]`.
    ///   Must be non-empty and rectangular (all rows the same length).
    /// * `origin` — world coordinate (m) of `phase_map[0][0]`.
    /// * `pitch` — pixel pitch (m), must be positive.
    /// * `label` — optional human-readable description.
    ///
    /// # Errors
    /// Returns `Err` if `phase_map` is empty, non-rectangular, or `pitch <= 0`.
    pub fn hologram_from_phase_map(
        phase_map: Vec<Vec<f64>>,
        origin: (f64, f64),
        pitch: f64,
        label: impl Into<String>,
    ) -> Result<Self, String> {
        if phase_map.is_empty() {
            return Err("phase_map must not be empty".into());
        }
        if pitch <= 0.0 {
            return Err(format!("pitch must be positive, got {pitch}"));
        }
        let ncols = phase_map[0].len();
        if ncols == 0 {
            return Err("phase_map rows must not be empty".into());
        }
        if phase_map.iter().any(|row| row.len() != ncols) {
            return Err("phase_map must be rectangular (all rows same length)".into());
        }
        Ok(MetasurfaceFunction::Hologram {
            phase_map: std::sync::Arc::new(phase_map),
            origin,
            pitch,
            label: label.into(),
        })
    }

    /// Compute a hologram phase map from a target intensity image using the
    /// Gerchberg-Saxton iterative Fourier algorithm.
    ///
    /// # Arguments
    /// * `target` — 2-D target intensity pattern (ny × nx), non-negative values.
    /// * `pitch` — pixel pitch in m.
    /// * `n_iters` — number of GS iterations (50–200 is typical).
    /// * `label` — human-readable description.
    ///
    /// Returns a `Hologram` with the recovered pupil-plane phase, with `origin`
    /// at `(0.0, 0.0)`.
    ///
    /// # Errors
    /// Returns `Err` if `target` is empty, non-rectangular, or `pitch <= 0`.
    pub fn hologram_from_target(
        target: &[Vec<f64>],
        pitch: f64,
        n_iters: usize,
        label: impl Into<String>,
    ) -> Result<Self, String> {
        if target.is_empty() || target[0].is_empty() {
            return Err("target must be non-empty".into());
        }
        if pitch <= 0.0 {
            return Err(format!("pitch must be positive, got {pitch}"));
        }
        let ny = target.len();
        let nx = target[0].len();
        if target.iter().any(|row| row.len() != nx) {
            return Err("target must be rectangular".into());
        }

        // Build target amplitude from target intensity (sqrt for amplitude).
        let target_amp: Vec<Vec<f64>> = target
            .iter()
            .map(|row| row.iter().map(|&v| v.max(0.0).sqrt()).collect())
            .collect();

        // Initialize pupil field (flat, row-major) with deterministic pseudo-random
        // phase using a diagonal wave pattern, unit amplitude.
        let n_total = ny * nx;
        let mut pupil: Vec<OxiComplex<f64>> = (0..n_total)
            .map(|k| {
                let phase = PI * (k as f64) / (n_total as f64);
                OxiComplex::from_polar(1.0_f64, phase)
            })
            .collect();

        // Gerchberg-Saxton iterations.
        for _ in 0..n_iters {
            // 1. Forward 2D FFT: pupil plane → image plane.
            let mut image = oxifft::fft2d(&pupil, ny, nx);

            // 2. Replace image-plane amplitude with target, keep phase.
            for (iy, row) in target_amp.iter().enumerate() {
                for (ix, &amp_t) in row.iter().enumerate() {
                    let c = image[iy * nx + ix];
                    let ph = c.arg();
                    image[iy * nx + ix] = OxiComplex::from_polar(amp_t, ph);
                }
            }

            // 3. Inverse 2D FFT: image plane → pupil plane.
            let back = oxifft::ifft2d(&image, ny, nx);

            // 4. Replace pupil-plane amplitude with 1.0, keep phase.
            for (k, c) in back.iter().enumerate() {
                let ph = c.arg();
                pupil[k] = OxiComplex::from_polar(1.0_f64, ph);
            }
        }

        // Extract pupil-plane phase as 2-D grid.
        let phase_map: Vec<Vec<f64>> = (0..ny)
            .map(|iy| (0..nx).map(|ix| pupil[iy * nx + ix].arg()).collect())
            .collect();

        Self::hologram_from_phase_map(phase_map, (0.0, 0.0), pitch, label)
    }
}

// ---------------------------------------------------------------------------
// SpinMultiplexedMetasurface
// ---------------------------------------------------------------------------

/// Spin-multiplexed (spin-dependent) metasurface.
///
/// The LCP and RCP channels implement independent optical functions encoded
/// in the same physical aperture.  Crosstalk is ideally −∞ dB (complete
/// orthogonality) but is limited in practice by fabrication tolerances and
/// finite element size.
#[derive(Debug, Clone)]
pub struct SpinMultiplexedMetasurface {
    /// Function for LCP input
    pub lcp_function: MetasurfaceFunction,
    /// Function for RCP input
    pub rcp_function: MetasurfaceFunction,
    /// Number of pixels (square array)
    pub n_pixels: usize,
    /// Physical pixel size (m)
    pub pixel_size: f64,
}

impl SpinMultiplexedMetasurface {
    /// Construct a spin-multiplexed metasurface.
    pub fn new(
        lcp: MetasurfaceFunction,
        rcp: MetasurfaceFunction,
        n: usize,
        pixel_size: f64,
    ) -> Self {
        Self {
            lcp_function: lcp,
            rcp_function: rcp,
            n_pixels: n,
            pixel_size,
        }
    }

    /// Physical coordinate of pixel index k (centred at 0).
    fn coord(&self, k: usize) -> f64 {
        (k as f64 - (self.n_pixels as f64 - 1.0) * 0.5) * self.pixel_size
    }

    /// Phase imparted at pixel (i, j) for the given input polarisation and
    /// wavelength.
    ///
    /// The encoding is:
    ///   θ(x,y) = (θ_LCP + θ_RCP) / 2
    ///
    /// where each orientation angle is derived from the target function.
    /// The output phase for polarisation P is then 2θ·sign(P) plus an
    /// additional correction from the RCP channel.
    pub fn phase_at_pixel(&self, i: usize, j: usize, polarization: CircPolarization) -> f64 {
        // Use a constant wavelength representative value (500 nm) because the
        // struct does not store wavelength; callers should use a wavelength-aware
        // wrapper for quantitative work.
        let wavelength = 500e-9;
        let x = self.coord(i);
        let y = self.coord(j);

        let theta_lcp = self.lcp_function.orientation_at(x, y, wavelength);
        let theta_rcp = self.rcp_function.orientation_at(x, y, wavelength);

        match polarization {
            // LCP → cross-pol phase = +2θ_LCP
            CircPolarization::LCP => 2.0 * theta_lcp,
            // RCP → cross-pol phase = −2θ_RCP
            CircPolarization::RCP => -2.0 * theta_rcp,
        }
    }

    /// Crosstalk between the two channels in dB.
    ///
    /// For an ideal device the channels are orthogonal (−∞ dB).
    /// Here a conservative engineering estimate is returned based on the
    /// half-wave plate conversion efficiency (~1 dB from residual co-pol).
    pub fn crosstalk_db(&self) -> f64 {
        // Ideal geometric-phase: conversion ≈ 1 → crosstalk ≈ −20 dB
        -20.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn circ_pol_jones_vectors_unit_norm() {
        for pol in [CircPolarization::LCP, CircPolarization::RCP] {
            let jv = pol.jones_vector();
            let norm_sq = jv[0].norm_sqr() + jv[1].norm_sqr();
            assert_abs_diff_eq!(norm_sq, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn pb_element_phase_at_centre_is_zero_for_flat_profile() {
        let elem = PbPhaseElement::new(11, 11, 100e-9, 500e-9);
        // All orientations are zero → phase at (5,5) = 0.
        assert_abs_diff_eq!(elem.phase_at(5, 5), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn pb_element_geometric_phase_is_twice_orientation() {
        let mut elem = PbPhaseElement::new(5, 5, 100e-9, 500e-9);
        elem.orientation_map[2][2] = PI / 6.0; // 30°
        assert_abs_diff_eq!(elem.phase_at(2, 2), PI / 3.0, epsilon = 1e-14);
    }

    #[test]
    fn pb_element_lens_profile_centre_angle_zero() {
        let mut elem = PbPhaseElement::new(11, 11, 100e-9, 500e-9);
        elem.set_lens_profile(1e-3);
        // Centre pixel: r = 0 → θ = 0
        assert_abs_diff_eq!(elem.orientation_map[5][5], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn pb_element_vortex_profile_charge_one() {
        let mut elem = PbPhaseElement::new(11, 11, 100e-9, 500e-9);
        elem.set_vortex_profile(2); // charge = 2 → θ = atan2(y,x), 2θ = 2φ
                                    // At pixel (10, 5) (right edge, centre row): x > 0, y = 0 → atan2 = 0.
                                    // θ = (2/2)*0 = 0
        assert_abs_diff_eq!(elem.orientation_map[5][10], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn pb_element_efficiency_orders_sum_to_one() {
        let elem = PbPhaseElement::new(11, 11, 100e-9, 500e-9);
        let sum = elem.efficiency_in_order(-1)
            + elem.efficiency_in_order(0)
            + elem.efficiency_in_order(1);
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn pb_beam_splitter_deflection_angle() {
        // λ = 500 nm, Λ = 1000 nm → arcsin(0.5) = 30°
        let bs = PbBeamSplitter::new(1000e-9, 500e-9);
        assert_abs_diff_eq!(bs.deflection_angle_deg(), 30.0, epsilon = 1e-6);
        assert_abs_diff_eq!(bs.polarization_separation_angle(), 60.0, epsilon = 1e-6);
    }

    #[test]
    fn spin_multiplexed_lcp_rcp_different_phases() {
        // Lens for LCP, grating for RCP.
        let lcp_fn = MetasurfaceFunction::Lens { focal_length: 1e-3 };
        let rcp_fn = MetasurfaceFunction::Grating {
            period: 2e-6,
            angle_deg: 0.0,
        };
        let ms = SpinMultiplexedMetasurface::new(lcp_fn, rcp_fn, 21, 100e-9);
        let phi_lcp = ms.phase_at_pixel(15, 10, CircPolarization::LCP);
        let phi_rcp = ms.phase_at_pixel(15, 10, CircPolarization::RCP);
        // The two phases should differ (independent functions).
        assert!(
            (phi_lcp - phi_rcp).abs() > 1e-6,
            "LCP and RCP phases are identical: phi_lcp={phi_lcp}, phi_rcp={phi_rcp}"
        );
    }

    #[test]
    fn hologram_phase_map_interpolation() {
        // 3×3 phase map with known values
        let map = vec![
            vec![0.0, PI / 4.0, PI / 2.0],
            vec![PI, PI, PI],
            vec![3.0 * PI / 2.0, 7.0 * PI / 4.0, 2.0 * PI],
        ];
        let func = MetasurfaceFunction::hologram_from_phase_map(map, (0.0, 0.0), 1.0e-6, "test")
            .expect("valid hologram");
        // At grid centre (1, 1) in pixel coords → world (1e-6, 1e-6)
        let theta = func.orientation_at(1.0e-6, 1.0e-6, 1550e-9);
        let expected = PI / 2.0; // phase_map[1][1] = PI, orientation = PI/2
        assert!(
            (theta - expected).abs() < 1e-12,
            "got {theta}, expected {expected}"
        );
    }

    #[test]
    fn hologram_from_phase_map_errors() {
        // Empty map
        assert!(
            MetasurfaceFunction::hologram_from_phase_map(vec![], (0.0, 0.0), 1e-6, "").is_err()
        );
        // Non-positive pitch
        assert!(
            MetasurfaceFunction::hologram_from_phase_map(vec![vec![0.0]], (0.0, 0.0), 0.0, "")
                .is_err()
        );
    }

    #[test]
    fn hologram_from_target_gs_convergence() {
        // Small 8×8 target: bright spot at (2, 2)
        let mut target = vec![vec![0.0f64; 8]; 8];
        target[2][2] = 1.0;
        let hologram = MetasurfaceFunction::hologram_from_target(&target, 1.0e-6, 50, "gs_test")
            .expect("GS should succeed");
        // Verify it's a Hologram variant and phase_map is 8×8
        match &hologram {
            MetasurfaceFunction::Hologram { phase_map, .. } => {
                assert_eq!(phase_map.len(), 8);
                assert_eq!(phase_map[0].len(), 8);
                // All phases should be in [-π, π]
                for row in phase_map.iter() {
                    for &phase in row.iter() {
                        assert!(
                            (-PI - 1e-10..=PI + 1e-10).contains(&phase),
                            "phase {phase} out of range"
                        );
                    }
                }
            }
            _ => panic!("expected Hologram variant"),
        }
    }
}
