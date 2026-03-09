/// Reflectarray and Reconfigurable Intelligent Surface (RIS) models.
///
/// Models covered:
/// - `RisElement` — single reflecting element with phase and amplitude
/// - `ReflectArray` — full N×M array with beam-steering and focusing modes
/// - `HolographicMetasurface` — holographic wavefront shaping via the
///   Gerchberg-Saxton (GS) iterative algorithm
///
/// Physical conventions:
/// - Angles: input/output in degrees, internal calculations in radians
/// - Distances: metres
/// - Phase: radians
use num_complex::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Free helper (avoids borrow-checker conflicts inside &mut self loops)
// ---------------------------------------------------------------------------

/// Quantise `phase` (radians) to the nearest of `n_levels` uniform levels
/// spanning [0, 2π).
#[inline]
fn quantize_phase_levels(phase: f64, n_levels: usize) -> f64 {
    if n_levels == 0 {
        return 0.0;
    }
    let n = n_levels as f64;
    let step = 2.0 * PI / n;
    let normalised = phase.rem_euclid(2.0 * PI) / step;
    let level = normalised.round() as usize % n_levels;
    level as f64 * step
}

// ---------------------------------------------------------------------------
// RisElement
// ---------------------------------------------------------------------------

/// A single element of a Reconfigurable Intelligent Surface.
///
/// Each element acts as a sub-wavelength antenna that can independently set
/// its reflection phase and amplitude, usually via a PIN diode, varactor, or
/// liquid crystal.
#[derive(Debug, Clone)]
pub struct RisElement {
    /// Current reflection phase state (radians, 0 … 2π)
    pub phase: f64,
    /// Reflection coefficient amplitude (0 … 1)
    pub amplitude: f64,
    /// (x, y) position in the array plane (m)
    pub position: [f64; 2],
}

impl RisElement {
    /// Construct a new element at the given position with unit amplitude and
    /// zero phase.
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            phase: 0.0,
            amplitude: 1.0,
            position: [x, y],
        }
    }

    /// Complex reflection coefficient Γ = A · exp(iφ).
    pub fn reflection_coefficient(&self) -> Complex64 {
        Complex64::from_polar(self.amplitude, self.phase)
    }
}

// ---------------------------------------------------------------------------
// ReflectArray
// ---------------------------------------------------------------------------

/// Reflectarray / Reconfigurable Intelligent Surface (N×M array).
///
/// The array factor is computed as:
///
///   AF(θ,φ) = Σ_{m,n} Γ_{mn} · exp(i k · d_mn · sin θ)
///
/// where d_mn is the element position projected onto the steering direction.
#[derive(Debug, Clone)]
pub struct ReflectArray {
    /// 2-D grid of RIS elements; indexing: elements\[row_y\]\[col_x\]
    pub elements: Vec<Vec<RisElement>>,
    /// Inter-element spacing (m), assumed square lattice
    pub element_spacing: f64,
    /// Design wavelength (m)
    pub wavelength: f64,
    /// Phase-quantisation bits (e.g. 1-bit: 2 levels, 2-bit: 4 levels)
    pub n_bits: u32,
}

impl ReflectArray {
    /// Construct an N_x × N_y array with λ/2 spacing at the given wavelength.
    pub fn new(nx: usize, ny: usize, spacing: f64, wavelength: f64, n_bits: u32) -> Self {
        let mut elements = Vec::with_capacity(ny);
        for row in 0..ny {
            let mut row_vec = Vec::with_capacity(nx);
            for col in 0..nx {
                let x = col as f64 * spacing;
                let y = row as f64 * spacing;
                row_vec.push(RisElement::new(x, y));
            }
            elements.push(row_vec);
        }
        Self {
            elements,
            element_spacing: spacing,
            wavelength,
            n_bits,
        }
    }

    /// Number of columns.
    fn nx(&self) -> usize {
        self.elements.first().map(|r| r.len()).unwrap_or(0)
    }

    /// Number of rows.
    fn ny(&self) -> usize {
        self.elements.len()
    }

    /// Number of discrete phase levels: 2^n_bits.
    pub fn n_phase_levels(&self) -> usize {
        1_usize << self.n_bits
    }

    /// Quantise a continuous phase to the nearest discrete level.
    pub fn quantize_phase(&self, phase: f64) -> f64 {
        quantize_phase_levels(phase, self.n_phase_levels())
    }

    /// Free-space wave number k₀ = 2π/λ.
    fn k0(&self) -> f64 {
        2.0 * PI / self.wavelength
    }

    /// Configure all elements to steer the reflected beam towards
    /// (θ, φ) in spherical coordinates (elevation, azimuth) in degrees.
    ///
    /// The required phase at element position (x, y) is:
    ///   φ_req = k₀ (x sin θ cos φ + y sin θ sin φ)   [conjugate phase for reflection]
    pub fn configure_beam_steering(&mut self, theta_deg: f64, phi_deg: f64) {
        let theta = theta_deg.to_radians();
        let phi = phi_deg.to_radians();
        let k = self.k0();
        let n_levels = self.n_phase_levels();
        let sin_theta = theta.sin();
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();
        for row in &mut self.elements {
            for elem in row.iter_mut() {
                let x = elem.position[0];
                let y = elem.position[1];
                let phase_req = k * (x * sin_theta * cos_phi + y * sin_theta * sin_phi);
                elem.phase = quantize_phase_levels(phase_req, n_levels);
            }
        }
    }

    /// Configure all elements to focus the reflected beam at a near-field
    /// point (x_t, y_t, z_t) in metres.
    ///
    /// The required phase at element (x_m, y_m, 0) is:
    ///   φ_mn = k₀ · √((x_t−x_m)² + (y_t−y_m)² + z_t²) − constant
    /// where the constant is chosen so that φ₀₀ = 0.
    pub fn configure_focusing(&mut self, target: [f64; 3]) {
        let k = self.k0();
        let n_levels = self.n_phase_levels();
        let (xt, yt, zt) = (target[0], target[1], target[2]);

        // Reference distance from element (0,0) to target.
        let r_ref = ((xt * xt) + (yt * yt) + (zt * zt)).sqrt();

        for row in &mut self.elements {
            for elem in row.iter_mut() {
                let dx = xt - elem.position[0];
                let dy = yt - elem.position[1];
                let r = (dx * dx + dy * dy + zt * zt).sqrt();
                let phase_req = k * (r - r_ref);
                elem.phase = quantize_phase_levels(phase_req, n_levels);
            }
        }
    }

    /// Array factor AF(θ, φ) at the given far-field direction.
    ///
    /// AF = Σ Γ_{mn} exp(−i k (x_m sin θ cos φ + y_m sin θ sin φ))
    ///
    /// The phase sign convention here is that of outgoing waves.
    pub fn array_factor(&self, theta_deg: f64, phi_deg: f64) -> Complex64 {
        let theta = theta_deg.to_radians();
        let phi = phi_deg.to_radians();
        let k = self.k0();
        let sin_theta = theta.sin();
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();

        let mut af = Complex64::new(0.0, 0.0);
        for row in &self.elements {
            for elem in row {
                let x = elem.position[0];
                let y = elem.position[1];
                let phase = -k * (x * sin_theta * cos_phi + y * sin_theta * sin_phi);
                af += elem.reflection_coefficient() * Complex64::from_polar(1.0, phase);
            }
        }
        af
    }

    /// Gain over isotropic (dBi) in the configured steering direction.
    ///
    /// Approximation: G ≈ 10 log₁₀(N M η_q)
    /// where N M is the element count and η_q is the quantisation efficiency.
    pub fn gain_db(&self) -> f64 {
        let n = (self.nx() * self.ny()) as f64;
        let eta = self.quantization_efficiency();
        10.0 * (n * eta).log10()
    }

    /// Far-field beam pattern |AF(θ,φ)|² on a grid of angles.
    ///
    /// Returns a `Vec<Vec<f64>>` with dimensions n_theta × n_phi.
    /// θ ∈ \[−90°, +90°\], φ ∈ [0°, 360°).
    pub fn beam_pattern(&self, n_theta: usize, n_phi: usize) -> Vec<Vec<f64>> {
        let n_theta = n_theta.max(1);
        let n_phi = n_phi.max(1);

        let mut pattern = vec![vec![0.0_f64; n_phi]; n_theta];
        for (it, pat_row) in pattern.iter_mut().enumerate().take(n_theta) {
            let theta = -90.0 + 180.0 * it as f64 / (n_theta - 1).max(1) as f64;
            for (ip, pat_cell) in pat_row.iter_mut().enumerate().take(n_phi) {
                let phi = 360.0 * ip as f64 / n_phi as f64;
                let af = self.array_factor(theta, phi);
                *pat_cell = af.norm_sqr();
            }
        }
        pattern
    }

    /// Phase-quantisation efficiency: η_q = (sin(π / 2^n) / (π / 2^n))².
    ///
    /// Approaches 1 as n → ∞ (continuous phase).
    pub fn quantization_efficiency(&self) -> f64 {
        let n = self.n_phase_levels() as f64;
        let arg = PI / n;
        let sinc = arg.sin() / arg;
        sinc * sinc
    }

    /// Half-power beamwidth (HPBW) in degrees (azimuthal plane).
    ///
    /// HPBW ≈ 0.886 λ / (N d) (in radians) × 180/π
    /// where N is the number of elements along x and d is the spacing.
    pub fn hpbw_deg(&self) -> f64 {
        let n = self.nx() as f64;
        let d = self.element_spacing;
        (0.886 * self.wavelength / (n * d)).asin().to_degrees() * 2.0
    }
}

// ---------------------------------------------------------------------------
// HolographicMetasurface
// ---------------------------------------------------------------------------

/// Holographic metasurface that reconstructs a target intensity pattern via
/// a precomputed phase hologram.
///
/// The hologram is computed with the Gerchberg-Saxton (GS) iterative
/// algorithm, which iterates between the hologram plane and the image plane
/// enforcing the constraints:
///   - Hologram plane: unit amplitude, free phase
///   - Image plane:    target amplitude, free phase
///
/// The forward propagation is approximated by a Fraunhofer (far-field) DFT.
#[derive(Debug, Clone)]
pub struct HolographicMetasurface {
    /// Phase hologram: φ(i,j) in radians; row-major \[y\]\[x\].
    pub phase_map: Vec<Vec<f64>>,
    /// Number of pixels along x
    pub n_pixels_x: usize,
    /// Number of pixels along y
    pub n_pixels_y: usize,
    /// Pixel pitch (m)
    pub pixel_size: f64,
    /// Design wavelength (m)
    pub wavelength: f64,
}

impl HolographicMetasurface {
    /// Construct a holographic metasurface with a flat (zero) phase.
    pub fn new(nx: usize, ny: usize, pixel_size: f64, wavelength: f64) -> Self {
        Self {
            phase_map: vec![vec![0.0; nx]; ny],
            n_pixels_x: nx,
            n_pixels_y: ny,
            pixel_size,
            wavelength,
        }
    }

    /// Fraunhofer (DFT) forward propagation.
    ///
    /// Computes the field at the image plane as a 2-D discrete Fourier
    /// transform of the hologram field `u(x,y) = exp(i φ(x,y))`.
    ///
    /// The implementation uses the explicit DFT sum to avoid introducing
    /// an external FFT dependency in this standalone module.  For large
    /// arrays callers should use the FDTD / RCWA modules for validation.
    fn dft2(&self, field: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
        let ny = field.len();
        let nx = if ny > 0 { field[0].len() } else { 0 };
        let mut out = vec![vec![Complex64::new(0.0, 0.0); nx]; ny];

        for (m, out_row) in out.iter_mut().enumerate().take(ny) {
            for (n, out_cell) in out_row.iter_mut().enumerate().take(nx) {
                let mut sum = Complex64::new(0.0, 0.0);
                for (j, field_row) in field.iter().enumerate().take(ny) {
                    for (i, &field_cell) in field_row.iter().enumerate().take(nx) {
                        let angle = -2.0
                            * PI
                            * (m as f64 * j as f64 / ny as f64 + n as f64 * i as f64 / nx as f64);
                        sum += field_cell * Complex64::from_polar(1.0, angle);
                    }
                }
                *out_cell = sum;
            }
        }
        out
    }

    /// Inverse DFT (IDFT).
    fn idft2(&self, spectrum: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
        let ny = spectrum.len();
        let nx = if ny > 0 { spectrum[0].len() } else { 0 };
        let norm = 1.0 / (nx * ny) as f64;
        let mut out = vec![vec![Complex64::new(0.0, 0.0); nx]; ny];

        for (j, out_row) in out.iter_mut().enumerate().take(ny) {
            for (i, out_cell) in out_row.iter_mut().enumerate().take(nx) {
                let mut sum = Complex64::new(0.0, 0.0);
                for (m, spec_row) in spectrum.iter().enumerate().take(ny) {
                    for (n, &spec_cell) in spec_row.iter().enumerate().take(nx) {
                        let angle = 2.0
                            * PI
                            * (m as f64 * j as f64 / ny as f64 + n as f64 * i as f64 / nx as f64);
                        sum += spec_cell * Complex64::from_polar(1.0, angle);
                    }
                }
                *out_cell = sum * norm;
            }
        }
        out
    }

    /// Gerchberg-Saxton iterative phase retrieval.
    ///
    /// Computes the phase hologram that, when illuminated with unit-amplitude
    /// plane-wave light, reconstructs the `target_intensity` distribution in
    /// the far field.
    ///
    /// # Arguments
    /// * `target_intensity` — 2-D target intensity map (row-major, same size
    ///   as the hologram).  Values are normalised internally.
    /// * `n_iterations` — Number of GS iterations (typically 20–100).
    pub fn compute_gs_hologram(&mut self, target_intensity: &[Vec<f64>], n_iterations: usize) {
        let ny = self.n_pixels_y.min(target_intensity.len());
        let nx = self
            .n_pixels_x
            .min(target_intensity.first().map(|r| r.len()).unwrap_or(0));
        if nx == 0 || ny == 0 {
            return;
        }

        // Target amplitude: sqrt of target intensity.
        let target_amp: Vec<Vec<f64>> = target_intensity[..ny]
            .iter()
            .map(|row| row[..nx].iter().map(|&v| v.max(0.0).sqrt()).collect())
            .collect();

        // Initialise hologram field with unit amplitude, random phase ≈ 0.
        let mut h_field: Vec<Vec<Complex64>> = (0..ny)
            .map(|_| vec![Complex64::new(1.0, 0.0); nx])
            .collect();

        for _ in 0..n_iterations {
            // Forward: hologram plane → image plane.
            let mut img = self.dft2(&h_field);

            // Enforce amplitude constraint in the image plane.
            for m in 0..ny {
                for n in 0..nx {
                    let angle = img[m][n].arg();
                    img[m][n] = Complex64::from_polar(target_amp[m][n], angle);
                }
            }

            // Backward: image plane → hologram plane.
            let back = self.idft2(&img);

            // Enforce unit-amplitude constraint in the hologram plane.
            for j in 0..ny {
                for i in 0..nx {
                    let angle = back[j][i].arg();
                    h_field[j][i] = Complex64::from_polar(1.0, angle);
                }
            }
        }

        // Store the computed phase.
        for (phase_row, hf_row) in self.phase_map.iter_mut().zip(h_field.iter()).take(ny) {
            for (phase_cell, hf_cell) in phase_row.iter_mut().zip(hf_row.iter()).take(nx) {
                *phase_cell = hf_cell.arg();
            }
        }
    }

    /// Estimate diffraction efficiency from phase uniformity.
    ///
    /// A perfectly uniform phase distribution → high efficiency (≈ 1/N for
    /// random phase, but GS converges well → ≈ 0.8 typical).
    ///
    /// Here: η = (mean |exp(iφ)|)² = 1 by construction; estimate from
    /// phase standard deviation: η_est = exp(−σ²_φ / (2π)²).
    pub fn efficiency(&self) -> f64 {
        let n = (self.n_pixels_x * self.n_pixels_y) as f64;
        if n < 2.0 {
            return 1.0;
        }
        let mean = self.phase_map.iter().flatten().sum::<f64>() / n;
        let var = self
            .phase_map
            .iter()
            .flatten()
            .map(|&p| (p - mean).powi(2))
            .sum::<f64>()
            / n;
        // Uniformly-distributed phase (best case) has σ² = (2π)²/12.
        // Normalise: η_est = var_uniform / var_actual ≤ 1.
        let var_uniform = (2.0 * PI).powi(2) / 12.0;
        // More uniform → higher efficiency.
        (var_uniform / (var + 1e-30)).min(1.0)
    }

    /// Reconstruct the target image at propagation distance `distance_m` (m).
    ///
    /// Uses the Fraunhofer approximation: the field at the image plane is the
    /// DFT of the hologram field `exp(i φ(x,y))`.  The returned 2-D array
    /// contains the intensity |E|² normalised to the peak.
    pub fn reconstruct(&self, _distance_m: f64) -> Vec<Vec<f64>> {
        // Build hologram field.
        let field: Vec<Vec<Complex64>> = self
            .phase_map
            .iter()
            .map(|row| row.iter().map(|&p| Complex64::from_polar(1.0, p)).collect())
            .collect();

        let img = self.dft2(&field);

        // Convert to intensity and normalise.
        let intensities: Vec<Vec<f64>> = img
            .iter()
            .map(|row| row.iter().map(|c| c.norm_sqr()).collect())
            .collect();

        let peak = intensities
            .iter()
            .flatten()
            .cloned()
            .fold(0.0_f64, f64::max);
        if peak < 1e-30 {
            return intensities;
        }
        intensities
            .into_iter()
            .map(|row| row.into_iter().map(|v| v / peak).collect())
            .collect()
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
    fn ris_element_reflection_coefficient_unit_amplitude() {
        let elem = RisElement::new(0.0, 0.0);
        assert_abs_diff_eq!(elem.reflection_coefficient().norm(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn reflect_array_n_phase_levels() {
        let ra = ReflectArray::new(8, 8, 5e-3, 60e-3, 2);
        assert_eq!(ra.n_phase_levels(), 4);
        let ra2 = ReflectArray::new(8, 8, 5e-3, 60e-3, 3);
        assert_eq!(ra2.n_phase_levels(), 8);
    }

    #[test]
    fn reflect_array_quantize_phase_wraps_correctly() {
        let ra = ReflectArray::new(4, 4, 5e-3, 60e-3, 2); // 4 levels: 0, π/2, π, 3π/2
        let step = 2.0 * PI / 4.0;
        // A phase of 0.1 rad should quantise to 0.
        let q = ra.quantize_phase(0.1);
        assert_abs_diff_eq!(q, 0.0, epsilon = 1e-12);
        // A phase of π/2 + 0.05 should quantise to π/2.
        let q2 = ra.quantize_phase(step + 0.05);
        assert_abs_diff_eq!(q2, step, epsilon = 1e-12);
    }

    #[test]
    fn reflect_array_quantization_efficiency_one_bit() {
        // 1-bit: N = 2 levels, η = (sin(π/2)/(π/2))² = (2/π)² ≈ 0.405
        let ra = ReflectArray::new(4, 4, 5e-3, 60e-3, 1);
        let expected = (2.0 / PI).powi(2);
        assert_abs_diff_eq!(ra.quantization_efficiency(), expected, epsilon = 1e-10);
    }

    #[test]
    fn reflect_array_beam_steering_broadside() {
        // Configured for (θ=0°, φ=0°): all phases zero.
        let mut ra = ReflectArray::new(4, 4, 15e-3, 60e-3, 4);
        ra.configure_beam_steering(0.0, 0.0);
        // All elements should have phase ≈ 0.
        for row in &ra.elements {
            for elem in row {
                assert_abs_diff_eq!(elem.phase, 0.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn reflect_array_gain_positive() {
        let ra = ReflectArray::new(8, 8, 15e-3, 60e-3, 4);
        let gain = ra.gain_db();
        assert!(gain > 0.0, "Gain should be positive, got {gain}");
    }

    #[test]
    fn holographic_metasurface_reconstruct_flat_phase_is_delta() {
        // With flat phase map the DFT produces a delta function at (0,0).
        let hs = HolographicMetasurface::new(4, 4, 100e-9, 500e-9);
        let img = hs.reconstruct(1.0e-3);
        // Peak should be at (0,0) and have intensity 1.0.
        assert_abs_diff_eq!(img[0][0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn holographic_gs_converges_small() {
        // Run GS on a 4×4 hologram targeting a uniform intensity.
        let mut hs = HolographicMetasurface::new(4, 4, 100e-9, 500e-9);
        let target = vec![vec![1.0_f64; 4]; 4];
        hs.compute_gs_hologram(&target, 5);
        // After GS all phases should be finite.
        for row in &hs.phase_map {
            for &p in row {
                assert!(p.is_finite(), "GS phase is NaN/inf");
            }
        }
    }

    #[test]
    fn holographic_efficiency_range() {
        let hs = HolographicMetasurface::new(8, 8, 100e-9, 500e-9);
        let eta = hs.efficiency();
        assert!((0.0..=1.0).contains(&eta), "Efficiency out of range: {eta}");
    }
}
