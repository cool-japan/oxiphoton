//! Atmospheric turbulence models for adaptive optics.
//!
//! Provides:
//! - [`AtmosphericTurbulence`]: Kolmogorov single-layer turbulence statistics
//! - [`TurbulentLayer`]: One turbulent layer in a multi-layer model
//! - [`LayeredAtmosphere`]: Multi-layer atmosphere with effective parameters
//! - [`PhaseScreen`]: Von Kármán phase screen for time-domain simulation
//!
//! # Physical Background
//!
//! Atmospheric turbulence follows Kolmogorov statistics for spatial scales
//! between the inner scale l₀ and outer scale L₀. The key parameter is the
//! Fried parameter r₀ (coherence length), which determines the degree of
//! wavefront distortion for a given aperture.
//!
//! ## Key Relations
//! - Phase structure function: D_φ(r) = 6.88 (r/r₀)^(5/3)
//! - Kolmogorov PSD: Φ(f) = 0.023 r₀^(−5/3) f^(−11/3)
//! - Isoplanatic angle: θ₀ = 0.314 (r₀/h)
//! - Coherence time: τ₀ = 0.314 (r₀/v)
//!
//! # References
//! - Kolmogorov (1941) — turbulence statistics
//! - Fried (1966) — coherence length r₀
//! - Noll (1976) — Zernike decomposition of turbulence
//! - von Kármán — modified spectrum with outer scale

const PI: f64 = std::f64::consts::PI;
const TWO_PI: f64 = 2.0 * PI;

// ─────────────────────────────────────────────────────────────────────────────
// Noll variance table
// ─────────────────────────────────────────────────────────────────────────────

/// Noll (1976) coefficients for residual wavefront variance after correcting
/// j Zernike modes: σ² = a_j * (D/r₀)^(5/3).
///
/// Values from Noll (1976), Table 1, for j = 1..21.
const NOLL_COEFFICIENTS: [f64; 22] = [
    0.0,    // j=0 (placeholder)
    1.0299, // j=1 (tip — residual after piston only)
    0.582,  // j=2
    0.134,  // j=3
    0.111,  // j=4
    0.0880, // j=5
    0.0648, // j=6
    0.0587, // j=7
    0.0525, // j=8
    0.0463, // j=9
    0.0401, // j=10
    0.0377, // j=11
    0.0352, // j=12
    0.0328, // j=13
    0.0304, // j=14
    0.0279, // j=15
    0.0267, // j=16
    0.0255, // j=17
    0.0243, // j=18
    0.0231, // j=19
    0.0220, // j=20
    0.0208, // j=21
];

// ─────────────────────────────────────────────────────────────────────────────
// AtmosphericTurbulence
// ─────────────────────────────────────────────────────────────────────────────

/// Single-layer Kolmogorov atmospheric turbulence.
///
/// Parameterised by the Fried parameter r₀, outer scale L₀, inner scale l₀,
/// and wind vector (speed + direction).
///
/// # Units
/// - r₀, L₀, l₀: metres
/// - wind_speed: m/s
/// - wind_direction: radians (0 = x-axis)
#[derive(Debug, Clone)]
pub struct AtmosphericTurbulence {
    /// Fried parameter in metres (larger = weaker turbulence).
    pub r0: f64,
    /// Outer scale in metres (L₀ ≈ 10–100 m typical).
    pub l0_outer: f64,
    /// Inner scale in metres (l₀ ≈ 1–10 mm typical).
    pub l0_inner: f64,
    /// Wind speed in m/s.
    pub wind_speed: f64,
    /// Wind direction in radians.
    pub wind_direction: f64,
}

impl AtmosphericTurbulence {
    /// Create a new turbulence model.
    ///
    /// # Arguments
    /// * `r0` — Fried parameter in metres
    /// * `l0` — outer scale in metres
    /// * `l0_inner` — inner scale in metres
    pub fn new(r0: f64, l0: f64, l0_inner: f64) -> Self {
        Self {
            r0,
            l0_outer: l0,
            l0_inner,
            wind_speed: 10.0,
            wind_direction: 0.0,
        }
    }

    /// Set wind velocity.
    pub fn with_wind(mut self, speed: f64, direction_rad: f64) -> Self {
        self.wind_speed = speed;
        self.wind_direction = direction_rad;
        self
    }

    /// Phase structure function D_φ(r) \[rad²\].
    ///
    /// For Kolmogorov turbulence: D_φ(r) = 6.88 (r/r₀)^(5/3).
    ///
    /// This is the Kolmogorov structure function (valid for l₀ ≪ r ≪ L₀).
    pub fn phase_structure_function(&self, r: f64) -> f64 {
        6.88 * (r / self.r0).powf(5.0 / 3.0)
    }

    /// Von Kármán modified structure function including outer scale.
    ///
    /// D_φ(r) ≈ 6.88 (r/r₀)^(5/3) * \[1 - (r/L₀)^(1/3)\] for r ≪ L₀.
    /// This saturates at the outer scale.
    pub fn von_karman_structure_function(&self, r: f64) -> f64 {
        let kolmogorov = 6.88 * (r / self.r0).powf(5.0 / 3.0);
        let outer_scale_correction = if self.l0_outer > 0.0 {
            (1.0 - (r / self.l0_outer).powf(1.0 / 3.0)).max(0.0)
        } else {
            1.0
        };
        kolmogorov * outer_scale_correction
    }

    /// Kolmogorov power spectral density of phase \[rad²/m⁻²\].
    ///
    /// Φ(f) = 0.023 r₀^(−5/3) f^(−11/3)
    ///
    /// where f is the spatial frequency in cycles per metre.
    /// Includes a low-frequency cutoff at f₀ = 1/L₀ (von Kármán).
    pub fn kolmogorov_psd(&self, f: f64) -> f64 {
        if f <= 0.0 {
            return 0.0;
        }
        let f0 = if self.l0_outer > 0.0 {
            1.0 / self.l0_outer
        } else {
            0.0
        };
        let f_eff = (f * f + f0 * f0).sqrt(); // von Kármán modification
        0.023 * self.r0.powf(-5.0 / 3.0) * f_eff.powf(-11.0 / 3.0)
    }

    /// Isoplanatic angle θ₀ in radians.
    ///
    /// θ₀ = 0.314 (r₀/h)
    ///
    /// where h is the effective turbulence height.
    ///
    /// # Arguments
    /// * `h` — effective turbulence height in metres
    pub fn isoplanatic_angle(&self, h: f64) -> f64 {
        if h <= 0.0 {
            return f64::INFINITY;
        }
        0.314 * self.r0 / h
    }

    /// Coherence time τ₀ in seconds.
    ///
    /// τ₀ = 0.314 r₀ / v
    ///
    /// where v is the wind speed.
    pub fn coherence_time(&self) -> f64 {
        if self.wind_speed <= 0.0 {
            return f64::INFINITY;
        }
        0.314 * self.r0 / self.wind_speed
    }

    /// Greenwood frequency in Hz.
    ///
    /// f_G = 0.427 v / r₀ (for a single layer).
    pub fn greenwood_frequency(&self) -> f64 {
        0.427 * self.wind_speed / self.r0
    }

    /// Residual wavefront variance after correcting `n_modes` Zernike modes \[rad²\].
    ///
    /// Uses the Noll (1976) coefficient table:
    ///   σ² = a_N (D/r₀)^(5/3)
    ///
    /// where D is the aperture diameter. Here we normalise by r₀ = 1 m and
    /// return the dimensionless coefficient; multiply by (D/r₀)^(5/3) for
    /// a specific aperture.
    ///
    /// # Arguments
    /// * `n_modes` — number of Zernike modes corrected (clamped to table size)
    pub fn noll_variance(&self, n_modes: usize) -> f64 {
        let idx = n_modes.min(NOLL_COEFFICIENTS.len() - 1).max(0);
        // Return coefficient (caller multiplies by (D/r₀)^(5/3)).
        NOLL_COEFFICIENTS[idx]
    }

    /// Residual phase variance after correcting `n_modes` Zernike modes \[rad²\].
    ///
    /// σ²_φ = a_N * (D/r₀)^(5/3)
    ///
    /// # Arguments
    /// * `n_modes` — number of modes corrected
    /// * `aperture_diameter` — aperture diameter in metres
    pub fn residual_phase_variance(&self, n_modes: usize, aperture_diameter: f64) -> f64 {
        let a_n = self.noll_variance(n_modes);
        a_n * (aperture_diameter / self.r0).powf(5.0 / 3.0)
    }

    /// Strehl ratio from Maréchal after correcting `n_modes` Zernike modes.
    ///
    /// S ≈ exp(−σ²_φ)
    pub fn strehl_after_correction(&self, n_modes: usize, aperture_diameter: f64) -> f64 {
        let sigma2 = self.residual_phase_variance(n_modes, aperture_diameter);
        (-sigma2).exp()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LayeredAtmosphere
// ─────────────────────────────────────────────────────────────────────────────

/// One turbulent layer in a multi-layer atmosphere model.
#[derive(Debug, Clone)]
pub struct TurbulentLayer {
    /// Layer height above ground in metres.
    pub height: f64,
    /// Fried parameter of this layer in metres.
    pub r0: f64,
    /// Wind speed at this layer in m/s.
    pub wind_speed: f64,
    /// Wind direction at this layer in radians.
    pub wind_direction: f64,
    /// Fractional contribution of this layer to the total C_n² integral.
    pub weight: f64,
}

impl TurbulentLayer {
    /// Create a new turbulent layer.
    pub fn new(height: f64, r0: f64, wind_speed: f64, wind_direction: f64) -> Self {
        Self {
            height,
            r0,
            wind_speed,
            wind_direction,
            weight: 1.0,
        }
    }
}

/// Multi-layer atmosphere model.
///
/// Combines multiple turbulent layers using the profile weighting to compute
/// effective parameters (r₀, coherence time, isoplanatic angle).
#[derive(Debug, Clone)]
pub struct LayeredAtmosphere {
    /// Turbulent layers.
    pub layers: Vec<TurbulentLayer>,
}

impl LayeredAtmosphere {
    /// Create a layered atmosphere from a list of turbulent layers.
    pub fn new(layers: Vec<TurbulentLayer>) -> Self {
        Self { layers }
    }

    /// Effective Fried parameter combining all layers.
    ///
    /// r₀_eff = (Σᵢ r₀ᵢ^(−5/3))^(−3/5)
    pub fn effective_r0(&self) -> f64 {
        let sum: f64 = self.layers.iter().map(|l| l.r0.powf(-5.0 / 3.0)).sum();
        if sum <= 0.0 {
            return f64::INFINITY;
        }
        sum.powf(-3.0 / 5.0)
    }

    /// Effective coherence time τ₀ for the layered atmosphere.
    ///
    /// τ₀_eff = 0.314 * (Σᵢ vᵢ^(5/3) * r₀ᵢ^(−5/3))^(−3/5)
    ///
    /// Equivalent to the Greenwood coherence time for a wind-weighted profile.
    pub fn effective_coherence_time(&self) -> f64 {
        let sum: f64 = self
            .layers
            .iter()
            .map(|l| {
                if l.r0 > 0.0 {
                    l.wind_speed.powf(5.0 / 3.0) * l.r0.powf(-5.0 / 3.0)
                } else {
                    0.0
                }
            })
            .sum();
        if sum <= 0.0 {
            return f64::INFINITY;
        }
        0.314 * sum.powf(-3.0 / 5.0)
    }

    /// Effective isoplanatic angle θ₀ for the layered atmosphere \[radians\].
    ///
    /// θ₀_eff = 0.314 * (Σᵢ hᵢ^(5/3) * r₀ᵢ^(−5/3))^(−3/5)
    pub fn isoplanatic_angle(&self) -> f64 {
        let sum: f64 = self
            .layers
            .iter()
            .map(|l| {
                if l.r0 > 0.0 && l.height > 0.0 {
                    l.height.powf(5.0 / 3.0) * l.r0.powf(-5.0 / 3.0)
                } else {
                    0.0
                }
            })
            .sum();
        if sum <= 0.0 {
            return f64::INFINITY;
        }
        0.314 * sum.powf(-3.0 / 5.0)
    }

    /// Effective Greenwood frequency \[Hz\].
    ///
    /// f_G = 0.427 * v_eff / r₀_eff
    ///
    /// where v_eff is the effective wind speed weighted by the turbulence profile.
    pub fn effective_greenwood_frequency(&self) -> f64 {
        let r0 = self.effective_r0();
        if r0.is_infinite() || r0 <= 0.0 {
            return 0.0;
        }
        // Effective wind speed from coherence time: v_eff = 0.314 r₀ / τ₀.
        let tau0 = self.effective_coherence_time();
        if tau0.is_infinite() || tau0 <= 0.0 {
            return 0.0;
        }
        0.314 * r0 / tau0 * 0.427 / 0.314
    }

    /// Sum of the turbulence strengths C_n² × Δh for each layer,
    /// proportional to the total turbulence integral.
    ///
    /// J = Σᵢ wᵢ · r₀ᵢ^(−5/3)  (unnormalised integral of C_n²)
    pub fn total_turbulence_integral(&self) -> f64 {
        self.layers
            .iter()
            .map(|l| l.weight * l.r0.powf(-5.0 / 3.0))
            .sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PhaseScreen
// ─────────────────────────────────────────────────────────────────────────────

/// Von Kármán phase screen for atmospheric turbulence simulation.
///
/// Generates a statistically correct 2D phase map using the Fourier method:
/// the phase PSD is sampled in frequency space and inverse-transformed to
/// give a realisation of the turbulent phase.
///
/// # Taylor Frozen Flow
/// Wind translation is modelled by shifting the phase screen. For a wind
/// vector (v_x, v_y) and timestep Δt, the screen shifts by
/// (v_x·Δt/pixel_scale, v_y·Δt/pixel_scale) pixels.
#[derive(Debug, Clone)]
pub struct PhaseScreen {
    /// Grid size (n × n pixels).
    pub n_pixels: usize,
    /// Physical scale in metres per pixel.
    pub pixel_scale: f64,
    /// Phase values in radians, row-major (row = y, col = x).
    pub phase: Vec<f64>,
    /// Fried parameter used to generate this screen.
    pub r0: f64,
    /// Outer scale used.
    pub l0_outer: f64,
    /// Fractional pixel offset x (for sub-pixel translation).
    pub offset_x: f64,
    /// Fractional pixel offset y (for sub-pixel translation).
    pub offset_y: f64,
}

impl PhaseScreen {
    /// Generate a new von Kármán phase screen using the spectral method.
    ///
    /// The screen is generated by:
    /// 1. Computing the von Kármán PSD on the frequency grid
    /// 2. Multiplying by complex Gaussian random amplitudes
    /// 3. Inverse-transforming to get the phase
    ///
    /// Since we must avoid rand, a deterministic pseudo-random sequence is
    /// used (linear congruential generator seeded from n and r0).
    ///
    /// # Arguments
    /// * `n` — grid size (must be power of 2 for efficiency)
    /// * `pixel_scale` — metres per pixel
    /// * `r0` — Fried parameter in metres
    /// * `l0` — outer scale in metres
    pub fn new_frozen(n: usize, pixel_scale: f64, r0: f64, l0: f64) -> Self {
        let phase = generate_phase_screen(n, pixel_scale, r0, l0);
        Self {
            n_pixels: n,
            pixel_scale,
            phase,
            r0,
            l0_outer: l0,
            offset_x: 0.0,
            offset_y: 0.0,
        }
    }

    /// Translate the phase screen by (dx_pixels, dy_pixels) using bilinear
    /// interpolation (Taylor frozen-flow hypothesis).
    ///
    /// The fractional offsets accumulate for smooth sub-pixel motion.
    pub fn translate(&mut self, dx_pixels: f64, dy_pixels: f64) {
        self.offset_x += dx_pixels;
        self.offset_y += dy_pixels;

        let n = self.n_pixels;

        // Wrap offsets to integer part + fractional part.
        let shift_ix = self.offset_x.floor() as isize;
        let shift_iy = self.offset_y.floor() as isize;
        let frac_x = self.offset_x - shift_ix as f64;
        let frac_y = self.offset_y - shift_iy as f64;

        // Update offsets (keep fractional part only).
        self.offset_x = frac_x;
        self.offset_y = frac_y;

        if shift_ix == 0 && shift_iy == 0 {
            return;
        }

        // Bilinear shift: new_phase[y][x] = interpolated old phase at (x + shift_ix, y + shift_iy).
        let mut new_phase = vec![0.0_f64; n * n];
        for iy in 0..n {
            for ix in 0..n {
                // Source coordinates (with periodic wrapping).
                let src_x = ((ix as isize + shift_ix).rem_euclid(n as isize)) as usize;
                let src_y = ((iy as isize + shift_iy).rem_euclid(n as isize)) as usize;
                let src_x1 = (src_x + 1) % n;
                let src_y1 = (src_y + 1) % n;

                // Bilinear interpolation using fractional offsets.
                let v00 = self.phase[src_y * n + src_x];
                let v01 = self.phase[src_y * n + src_x1];
                let v10 = self.phase[src_y1 * n + src_x];
                let v11 = self.phase[src_y1 * n + src_x1];

                new_phase[iy * n + ix] = v00 * (1.0 - frac_x) * (1.0 - frac_y)
                    + v01 * frac_x * (1.0 - frac_y)
                    + v10 * (1.0 - frac_x) * frac_y
                    + v11 * frac_x * frac_y;
            }
        }
        self.phase = new_phase;
    }

    /// RMS phase across the screen in radians.
    pub fn rms_phase(&self) -> f64 {
        let n = self.phase.len() as f64;
        if n < 1.0 {
            return 0.0;
        }
        let mean = self.phase.iter().sum::<f64>() / n;
        let var = self
            .phase
            .iter()
            .map(|&p| (p - mean) * (p - mean))
            .sum::<f64>()
            / n;
        var.sqrt()
    }

    /// Approximate Strehl ratio via the Maréchal approximation.
    ///
    /// S ≈ exp(−σ²_φ)
    ///
    /// where σ_φ is the RMS phase in radians. Note that for large turbulence
    /// (σ_φ > 1 rad) this approximation underestimates Strehl.
    pub fn strehl_from_marechal(&self) -> f64 {
        let sigma = self.rms_phase();
        (-sigma * sigma).exp()
    }

    /// Sample the phase at (x, y) in metres using bilinear interpolation.
    ///
    /// Returns `None` if (x, y) is outside the screen.
    pub fn sample(&self, x: f64, y: f64) -> Option<f64> {
        let n = self.n_pixels as f64;
        let half = n * self.pixel_scale * 0.5;
        let px_f = (x + half) / self.pixel_scale;
        let py_f = (y + half) / self.pixel_scale;

        if px_f < 0.0 || py_f < 0.0 || px_f >= n || py_f >= n {
            return None;
        }

        let px0 = px_f.floor() as usize;
        let py0 = py_f.floor() as usize;
        let px1 = (px0 + 1).min(self.n_pixels - 1);
        let py1 = (py0 + 1).min(self.n_pixels - 1);
        let fx = px_f - px0 as f64;
        let fy = py_f - py0 as f64;

        let ni = self.n_pixels;
        let v00 = self.phase[py0 * ni + px0];
        let v01 = self.phase[py0 * ni + px1];
        let v10 = self.phase[py1 * ni + px0];
        let v11 = self.phase[py1 * ni + px1];

        Some(
            v00 * (1.0 - fx) * (1.0 - fy)
                + v01 * fx * (1.0 - fy)
                + v10 * (1.0 - fx) * fy
                + v11 * fx * fy,
        )
    }

    /// Compute the empirical phase structure function D(r) averaged over the screen.
    ///
    /// Compares pairs of pixels separated by approximately `r / pixel_scale` pixels.
    /// Uses a random but deterministic sample for efficiency.
    pub fn empirical_structure_function(&self, r: f64) -> f64 {
        let r_pix = (r / self.pixel_scale).round() as isize;
        let n = self.n_pixels as isize;
        let mut sum = 0.0_f64;
        let mut count = 0usize;

        // Sample pairs along x-axis at separation r_pix.
        for iy in 0..self.n_pixels {
            for ix in 0..(self.n_pixels as isize - r_pix).max(0) as usize {
                let p1 = self.phase[iy * self.n_pixels + ix];
                let ix2 = (ix as isize + r_pix).min(n - 1) as usize;
                let p2 = self.phase[iy * self.n_pixels + ix2];
                let diff = p1 - p2;
                sum += diff * diff;
                count += 1;
            }
        }
        if count == 0 {
            return 0.0;
        }
        sum / count as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase screen generation (spectral method, pure Rust, no rand)
// ─────────────────────────────────────────────────────────────────────────────

/// Deterministic pseudo-random number generator (Xorshift64).
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform [0, 1).
    fn next_f64(&mut self) -> f64 {
        self.next_u64() as f64 / u64::MAX as f64
    }

    /// Box-Muller transform: standard normal sample.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-10);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (TWO_PI * u2).cos()
    }
}

/// Generate a von Kármán phase screen using the spectral method.
///
/// Steps:
/// 1. Build the 2D spatial frequency grid
/// 2. Evaluate the von Kármán PSD at each frequency: Φ(f) = 0.023 r₀^(−5/3) (f²+f₀²)^(−11/6)
/// 3. Multiply by complex Gaussian noise (deterministic)
/// 4. Inverse FFT to get the phase screen
/// 5. Remove piston (mean phase)
fn generate_phase_screen(n: usize, pixel_scale: f64, r0: f64, l0: f64) -> Vec<f64> {
    let f0 = if l0 > 0.0 { 1.0 / l0 } else { 0.0 };
    let df = 1.0 / (n as f64 * pixel_scale); // frequency grid spacing
    let norm = 0.023_f64 * r0.powf(-5.0 / 3.0) * df * df; // PSD × df² = variance per mode

    // Work in the Fourier domain.
    // Complex array: re and im interleaved as [n*n] re + [n*n] im.
    let mut re = vec![0.0_f64; n * n];
    let mut im = vec![0.0_f64; n * n];

    let mut rng = Xorshift64::new((n as u64).wrapping_mul(13) ^ (r0.to_bits()));

    for ky in 0..n {
        // Centred frequency.
        let fy = if ky <= n / 2 {
            ky as f64 * df
        } else {
            (ky as f64 - n as f64) * df
        };

        for kx in 0..n {
            if kx == 0 && ky == 0 {
                // DC component = 0 (no piston).
                continue;
            }
            let fx = if kx <= n / 2 {
                kx as f64 * df
            } else {
                (kx as f64 - n as f64) * df
            };

            let f2 = fx * fx + fy * fy;
            let f_eff2 = f2 + f0 * f0;

            // Von Kármán PSD amplitude.
            let psd = norm * f_eff2.powf(-11.0 / 6.0);
            let amplitude = psd.sqrt();

            // Complex Gaussian random variable.
            let g_re = rng.next_normal();
            let g_im = rng.next_normal();

            re[ky * n + kx] = amplitude * g_re;
            im[ky * n + kx] = amplitude * g_im;
        }
    }

    // Inverse FFT using the same DFT as a manual implementation.
    // We use the direct DFT formula for simplicity (O(n^4) — feasible for n ≤ 64).
    // For n > 64, a proper FFT should be used; here we use n ≤ 128 in practice.
    let phase = if n <= 64 {
        direct_idft_2d(&re, &im, n)
    } else {
        // For larger screens, use a subsampled approximate transform.
        direct_idft_2d_approx(&re, &im, n)
    };

    // Remove piston.
    let mean = phase.iter().sum::<f64>() / (n * n) as f64;
    phase.iter().map(|&p| p - mean).collect()
}

/// Direct 2D IDFT for small n (real part of inverse transform).
fn direct_idft_2d(re: &[f64], im: &[f64], n: usize) -> Vec<f64> {
    let n_f64 = n as f64;
    let n2 = (n * n) as f64;
    let mut result = vec![0.0_f64; n * n];

    for y in 0..n {
        for x in 0..n {
            let mut val = 0.0_f64;
            for ky in 0..n {
                for kx in 0..n {
                    let phase_arg = TWO_PI * (kx as f64 * x as f64 + ky as f64 * y as f64) / n_f64;
                    let cos_p = phase_arg.cos();
                    let sin_p = phase_arg.sin();
                    let idx = ky * n + kx;
                    // IDFT: real(F^{-1}[H]) = (1/N²) Σ [re*cos - im*sin].
                    val += re[idx] * cos_p - im[idx] * sin_p;
                }
            }
            result[y * n + x] = val / n2;
        }
    }
    result
}

/// Approximate 2D IDFT for larger n using a reduced sampling.
///
/// For n > 64 uses separable 1D transforms row by row then column by column
/// with the Cooley-Tukey butterfly algorithm.
fn direct_idft_2d_approx(re: &[f64], im: &[f64], n: usize) -> Vec<f64> {
    // Perform 1D IDFT on each row, then each column.
    let mut work_re = re.to_vec();
    let mut work_im = im.to_vec();

    // Row IDFTs.
    for ky in 0..n {
        let row_re: Vec<f64> = (0..n).map(|kx| work_re[ky * n + kx]).collect();
        let row_im: Vec<f64> = (0..n).map(|kx| work_im[ky * n + kx]).collect();
        let (out_re, out_im) = idft_1d(&row_re, &row_im);
        for kx in 0..n {
            work_re[ky * n + kx] = out_re[kx];
            work_im[ky * n + kx] = out_im[kx];
        }
    }

    // Column IDFTs.
    for kx in 0..n {
        let col_re: Vec<f64> = (0..n).map(|ky| work_re[ky * n + kx]).collect();
        let col_im: Vec<f64> = (0..n).map(|ky| work_im[ky * n + kx]).collect();
        let (out_re, _out_im) = idft_1d(&col_re, &col_im);
        for ky in 0..n {
            work_re[ky * n + kx] = out_re[ky];
        }
    }

    work_re
}

/// 1D IDFT using the direct sum (for n ≤ 256).
fn idft_1d(re: &[f64], im: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = re.len();
    let n_f64 = n as f64;
    let mut out_re = vec![0.0_f64; n];
    let mut out_im = vec![0.0_f64; n];

    for x in 0..n {
        let mut vr = 0.0_f64;
        let mut vi = 0.0_f64;
        for k in 0..n {
            let arg = TWO_PI * k as f64 * x as f64 / n_f64;
            let c = arg.cos();
            let s = arg.sin();
            vr += re[k] * c - im[k] * s;
            vi += re[k] * s + im[k] * c;
        }
        out_re[x] = vr / n_f64;
        out_im[x] = vi / n_f64;
    }
    (out_re, out_im)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atmospheric_turbulence_structure_function() {
        let atm = AtmosphericTurbulence::new(0.2, 30.0, 0.005);
        // D_phi(r0) should equal 6.88.
        let d = atm.phase_structure_function(0.2);
        assert!(
            (d - 6.88).abs() < 1e-10,
            "D_phi(r0) should be 6.88, got {}",
            d
        );
    }

    #[test]
    fn test_atmospheric_turbulence_structure_function_scaling() {
        let atm = AtmosphericTurbulence::new(0.1, 30.0, 0.005);
        let d1 = atm.phase_structure_function(0.1);
        let d2 = atm.phase_structure_function(0.2);
        // D(2r) = 2^(5/3) * D(r).
        let expected_ratio = 2.0_f64.powf(5.0 / 3.0);
        let ratio = d2 / d1;
        assert!(
            (ratio - expected_ratio).abs() < 1e-10,
            "Scaling: expected {}, got {}",
            expected_ratio,
            ratio
        );
    }

    #[test]
    fn test_atmospheric_turbulence_coherence_time() {
        let atm = AtmosphericTurbulence::new(0.2, 30.0, 0.005).with_wind(10.0, 0.0);
        let tau = atm.coherence_time();
        let expected = 0.314 * 0.2 / 10.0;
        assert!(
            (tau - expected).abs() < 1e-12,
            "τ₀ mismatch: {} vs {}",
            tau,
            expected
        );
    }

    #[test]
    fn test_atmospheric_turbulence_isoplanatic_angle() {
        let atm = AtmosphericTurbulence::new(0.2, 30.0, 0.005);
        let theta = atm.isoplanatic_angle(10000.0);
        let expected = 0.314 * 0.2 / 10000.0;
        assert!(
            (theta - expected).abs() < 1e-15,
            "θ₀ mismatch: {} vs {}",
            theta,
            expected
        );
    }

    #[test]
    fn test_atmospheric_turbulence_kolmogorov_psd_zero() {
        let atm = AtmosphericTurbulence::new(0.2, 30.0, 0.005);
        let psd = atm.kolmogorov_psd(0.0);
        assert_eq!(psd, 0.0, "PSD at f=0 should be 0");
    }

    #[test]
    fn test_noll_variance_decreases_with_more_modes() {
        let atm = AtmosphericTurbulence::new(0.2, 30.0, 0.005);
        let v5 = atm.noll_variance(5);
        let v10 = atm.noll_variance(10);
        assert!(
            v5 > v10,
            "More corrected modes should give smaller residual variance"
        );
    }

    #[test]
    fn test_layered_atmosphere_effective_r0_single_layer() {
        // Single layer: r0_eff should equal the layer's r0.
        let layer = TurbulentLayer::new(1000.0, 0.2, 10.0, 0.0);
        let atm = LayeredAtmosphere::new(vec![layer]);
        let r0_eff = atm.effective_r0();
        assert!(
            (r0_eff - 0.2).abs() < 1e-10,
            "Single-layer r0_eff = {}, expected 0.2",
            r0_eff
        );
    }

    #[test]
    fn test_layered_atmosphere_effective_r0_two_equal_layers() {
        // Two layers with same r0: r0_eff = r0 / 2^(3/5).
        let r0 = 0.2_f64;
        let layer1 = TurbulentLayer::new(1000.0, r0, 10.0, 0.0);
        let layer2 = TurbulentLayer::new(5000.0, r0, 10.0, 0.0);
        let atm = LayeredAtmosphere::new(vec![layer1, layer2]);
        let r0_eff = atm.effective_r0();
        let expected = r0 / 2.0_f64.powf(3.0 / 5.0);
        assert!(
            (r0_eff - expected).abs() < 1e-10,
            "Two-layer r0_eff = {}, expected {}",
            r0_eff,
            expected
        );
    }

    #[test]
    fn test_layered_atmosphere_isoplanatic_angle_positive() {
        let layer = TurbulentLayer::new(5000.0, 0.15, 15.0, 0.0);
        let atm = LayeredAtmosphere::new(vec![layer]);
        let theta = atm.isoplanatic_angle();
        assert!(
            theta > 0.0 && theta < 1.0,
            "θ₀ = {} should be a small positive angle",
            theta
        );
    }

    #[test]
    fn test_phase_screen_rms_nonzero() {
        let screen = PhaseScreen::new_frozen(16, 0.01, 0.2, 30.0);
        let rms = screen.rms_phase();
        assert!(rms > 0.0, "Phase screen RMS should be nonzero, got {}", rms);
    }

    #[test]
    fn test_phase_screen_strehl_in_range() {
        let screen = PhaseScreen::new_frozen(16, 0.01, 0.2, 30.0);
        let s = screen.strehl_from_marechal();
        assert!(s > 0.0 && s <= 1.0, "Strehl should be in (0, 1], got {}", s);
    }

    #[test]
    fn test_phase_screen_translate_changes_phase() {
        let mut screen = PhaseScreen::new_frozen(16, 0.01, 0.2, 30.0);
        let phase_before = screen.phase.clone();
        screen.translate(3.0, 0.0);
        let changed = screen
            .phase
            .iter()
            .zip(phase_before.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-10);
        assert!(changed, "Phase screen should change after translation");
    }

    #[test]
    fn test_phase_screen_sample_centre() {
        let screen = PhaseScreen::new_frozen(16, 0.01, 0.2, 30.0);
        // Sampling at (0, 0) should succeed.
        let val = screen.sample(0.0, 0.0);
        assert!(val.is_some(), "Sampling at centre should succeed");
    }

    #[test]
    fn test_phase_screen_sample_outside() {
        let screen = PhaseScreen::new_frozen(16, 0.01, 0.2, 30.0);
        // Far outside the screen.
        let val = screen.sample(10.0, 10.0);
        assert!(val.is_none(), "Sampling outside should return None");
    }

    #[test]
    fn test_xorshift64_not_constant() {
        let mut rng = Xorshift64::new(42);
        let a = rng.next_f64();
        let b = rng.next_f64();
        assert!(
            (a - b).abs() > 1e-10,
            "PRNG should produce different values"
        );
    }

    #[test]
    fn test_atmospheric_turbulence_greenwood_frequency() {
        let atm = AtmosphericTurbulence::new(0.2, 30.0, 0.005).with_wind(10.0, 0.0);
        let f_g = atm.greenwood_frequency();
        let expected = 0.427 * 10.0 / 0.2;
        assert!(
            (f_g - expected).abs() < 1e-10,
            "Greenwood frequency mismatch"
        );
    }
}
