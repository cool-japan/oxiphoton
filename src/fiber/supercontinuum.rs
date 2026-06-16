/// Supercontinuum generation model using the Generalised NLSE (GNLSE).
///
/// Implements the split-step Fourier (SSF) method with:
/// - Arbitrary-order dispersion (Taylor expansion of β(ω))
/// - Kerr SPM
/// - Raman response (delayed nonlinear response)
/// - Self-steepening (shock term)
/// - Power loss α
///
/// The GNLSE in the retarded frame (Dudley et al., Rev. Mod. Phys. 2006):
/// ```text
///   ∂A/∂z = D̂A + iγ(1 + i∂_T/ω₀) [A(z,T) ∫ R(T')|A(T-T')|² dT']
/// ```
/// where D̂ = Σₙ βₙ(iω)ⁿ/n! − α/2 in the frequency domain,
/// and R(t) = (1-fR)δ(t) + fR h_R(t) is the nonlinear response function.
///
/// The pure-Rust Cooley-Tukey radix-2 FFT is implemented internally.
use num_complex::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Pure-Rust radix-2 Cooley-Tukey FFT
// ---------------------------------------------------------------------------

/// In-place FFT (forward, sign convention: exp(−iωt)).
///
/// `buf` length must be a power of 2.  The result is the DFT of the input.
pub fn fft_inplace(buf: &mut [Complex64]) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }
    // Cooley-Tukey butterfly stages
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        let wlen = Complex64::new(ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let mut w = Complex64::new(1.0, 0.0);
            for jj in 0..(len / 2) {
                let u = buf[i + jj];
                let v = buf[i + jj + len / 2] * w;
                buf[i + jj] = u + v;
                buf[i + jj + len / 2] = u - v;
                w *= wlen;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// In-place IFFT (inverse FFT, normalised by 1/N).
///
/// Uses the identity: IFFT(x) = conj(FFT(conj(x)))/N.
pub fn ifft_inplace(buf: &mut [Complex64]) {
    // Conjugate
    for v in buf.iter_mut() {
        *v = v.conj();
    }
    fft_inplace(buf);
    let n = buf.len() as f64;
    for v in buf.iter_mut() {
        *v = v.conj() / n;
    }
}

// ---------------------------------------------------------------------------
// Helper: build angular frequency axis (unshifted, FFT-order)
// ---------------------------------------------------------------------------

fn omega_axis(n: usize, dt: f64) -> Vec<f64> {
    let dw = 2.0 * PI / (n as f64 * dt);
    (0..n)
        .map(|k| {
            if k < n / 2 {
                k as f64 * dw
            } else {
                (k as f64 - n as f64) * dw
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// GnlseSolver
// ---------------------------------------------------------------------------

/// Generalised NLSE solver for supercontinuum generation.
///
/// Uses the symmetric split-step Fourier method.  All higher-order dispersion
/// terms, Raman response, self-steepening, and linear loss are included.
#[derive(Debug, Clone)]
pub struct GnlseSolver {
    /// Total fiber length (m).
    pub fiber_length: f64,
    /// Propagation step size dz (m).
    pub dz: f64,
    /// Taylor coefficients [β₂, β₃, β₄, …] in SI units (s^n/m), starting at n=2.
    pub beta_coeffs: Vec<f64>,
    /// Nonlinear coefficient γ (1/(W·m)).
    pub gamma: f64,
    /// Power loss coefficient α (1/m).
    pub alpha: f64,
    /// Raman fraction fR (≈ 0.18 for silica).
    pub raman_fraction: f64,
    /// Raman response time τ₁ (s) ≈ 12.2 fs.
    pub raman_time1: f64,
    /// Raman response time τ₂ (s) ≈ 32 fs.
    pub raman_time2: f64,
    /// Number of time-domain grid points (power of 2 required).
    pub n_time_points: usize,
    /// Time step Δt (s).
    pub dt: f64,
}

impl GnlseSolver {
    /// Construct a GNLSE solver pre-configured for standard silica fiber.
    ///
    /// Default Raman parameters: fR = 0.18, τ₁ = 12.2 fs, τ₂ = 32 fs.
    /// The time grid is 4096 points × 10 fs per point.
    pub fn new_silica(fiber_length: f64, gamma: f64, beta2: f64, beta3: f64) -> Self {
        Self {
            fiber_length,
            dz: fiber_length / 1000.0_f64.min(fiber_length / 0.01).max(1.0),
            beta_coeffs: vec![beta2, beta3],
            gamma,
            alpha: 0.046e-3, // 0.2 dB/km
            raman_fraction: 0.18,
            raman_time1: 12.2e-15,
            raman_time2: 32.0e-15,
            n_time_points: 4096,
            dt: 10e-15,
        }
    }

    /// Raman response function h_R(t) (normalised, 1/s).
    ///
    /// ```text
    ///   h_R(t) = (τ₁² + τ₂²) / (τ₁ τ₂²) · exp(−t/τ₂) · sin(t/τ₁)  [t > 0]
    /// ```
    pub fn raman_response(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        let tau1 = self.raman_time1;
        let tau2 = self.raman_time2;
        let prefactor = (tau1 * tau1 + tau2 * tau2) / (tau1 * tau2 * tau2);
        prefactor * (-t / tau2).exp() * (t / tau1).sin()
    }

    /// Linear dispersion phase accumulated per step dz (rad).
    ///
    /// ```text
    ///   φ_D(ω) = dz/2 · [Σ_{n=2}^{N} βₙ (iω)ⁿ / n!] − α dz/2
    /// ```
    /// (The α/2 term is applied for a *half* step; callers use dz/2 for the
    /// symmetric SSF half-steps and dz for the single full dispersion step.)
    pub fn dispersion_phase(&self, omega: f64) -> f64 {
        let mut phase = 0.0_f64;
        let mut omega_n = omega * omega; // starts at ω²
        let mut factorial = 2.0_f64; // starts at 2!
        for (n_idx, &beta_n) in self.beta_coeffs.iter().enumerate() {
            // n = n_idx + 2  (first element is β₂)
            let n = n_idx + 2;
            phase += beta_n * omega_n / factorial;
            omega_n *= omega;
            factorial *= (n + 1) as f64;
        }
        phase
    }

    /// Nonlinear step: computes iγ [(1-fR)|A|²A + fR A*(hR⊗|A|²)] for a
    /// propagation distance `dz`.  Self-steepening is applied as a
    /// first-order correction in the frequency domain.
    ///
    /// Returns the updated field after the nonlinear phase.
    pub fn nonlinear_step(&self, field: &[Complex64], omega0: f64) -> Vec<Complex64> {
        let n = field.len();
        let dt = self.dt;
        let fr = self.raman_fraction;

        // Instantaneous (Kerr) contribution
        let kerr: Vec<Complex64> = field
            .iter()
            .map(|&a| a * ((1.0 - fr) * a.norm_sqr()))
            .collect();

        // Delayed Raman contribution: hR ⊗ |A|²
        // Compute h_R on the time grid, FFT it, multiply by FFT(|A|²), IFFT back.
        let intensity: Vec<Complex64> = field
            .iter()
            .map(|&a| Complex64::new(a.norm_sqr(), 0.0))
            .collect();

        let mut int_fft = intensity;
        fft_inplace(&mut int_fft);

        // Build Raman response in FFT order (causal, sampled at k·dt)
        let mut hr_buf: Vec<Complex64> = (0..n)
            .map(|k| {
                let t = k as f64 * dt;
                Complex64::new(self.raman_response(t), 0.0)
            })
            .collect();
        fft_inplace(&mut hr_buf);

        // Convolution in frequency domain
        let mut raman_conv: Vec<Complex64> = hr_buf
            .iter()
            .zip(int_fft.iter())
            .map(|(h, i)| h * i * dt) // include dt for proper integral scaling
            .collect();
        ifft_inplace(&mut raman_conv);

        // Total nonlinear polarisation: (1-fR)|A|²A + fR A*(hR⊗|A|²)
        // Then apply self-steepening in freq domain: multiply by (1 + ω/ω₀)
        let mut nl_field: Vec<Complex64> = field
            .iter()
            .zip(kerr.iter())
            .zip(raman_conv.iter())
            .map(|((&a, &k), &r)| a * fr * r + k)
            .collect();

        // Self-steepening: go to freq domain, apply (1 + ω/ω₀), come back
        fft_inplace(&mut nl_field);
        let omegas = omega_axis(n, dt);
        for (s, &om) in nl_field.iter_mut().zip(omegas.iter()) {
            *s *= 1.0 + om / omega0;
        }
        ifft_inplace(&mut nl_field);

        nl_field
    }

    /// Propagate the input field envelope over `fiber_length` using the
    /// symmetric split-step Fourier method.
    ///
    /// Steps:
    /// 1. Half-step dispersion in frequency domain.
    /// 2. Full nonlinear step in time domain.
    /// 3. Half-step dispersion + loss in frequency domain.
    ///
    /// Returns the output field A(L, t).
    pub fn propagate(&self, input_field: &[Complex64], omega0: f64) -> Vec<Complex64> {
        let n = self.n_time_points;
        assert!(
            input_field.len() == n,
            "input_field length {} must equal n_time_points {}",
            input_field.len(),
            n
        );

        let dz = self.dz;
        let n_steps = ((self.fiber_length / dz).ceil() as usize).max(1);

        let mut field: Vec<Complex64> = input_field.to_vec();
        let omegas = omega_axis(n, self.dt);

        for _step in 0..n_steps {
            // Current step might be shorter at the end of the fiber
            let step_dz = dz.min(self.fiber_length - _step as f64 * dz).max(0.0);
            if step_dz <= 0.0 {
                break;
            }

            // ── 1. Half dispersion step ────────────────────────────────────
            fft_inplace(&mut field);
            for (s, &om) in field.iter_mut().zip(omegas.iter()) {
                let phi = self.dispersion_phase(om) * step_dz;
                let loss = (-self.alpha * step_dz / 2.0).exp();
                *s *= Complex64::new(0.0, phi).exp() * loss;
            }
            ifft_inplace(&mut field);

            // ── 2. Full nonlinear step ─────────────────────────────────────
            let nl = self.nonlinear_step(&field, omega0);
            for (a, nl_a) in field.iter_mut().zip(nl.iter()) {
                let phi_nl = self.gamma * nl_a.norm() * step_dz;
                *a *= Complex64::new(0.0, phi_nl).exp();
            }

            // ── 3. Half dispersion step ────────────────────────────────────
            fft_inplace(&mut field);
            for (s, &om) in field.iter_mut().zip(omegas.iter()) {
                let phi = self.dispersion_phase(om) * step_dz;
                let loss = (-self.alpha * step_dz / 2.0).exp();
                *s *= Complex64::new(0.0, phi).exp() * loss;
            }
            ifft_inplace(&mut field);
        }

        field
    }

    /// Compute the output power spectrum |Ã(L, ω)|² from the input field.
    ///
    /// Returns a vector of spectral power values (W·s² = J·s) at each frequency
    /// bin (FFT order: 0, Δω, 2Δω, …, (N/2−1)Δω, −N/2·Δω, …, −Δω).
    pub fn output_spectrum(&self, input_field: &[Complex64], omega0: f64) -> Vec<f64> {
        let output = self.propagate(input_field, omega0);
        let mut spec: Vec<Complex64> = output;
        fft_inplace(&mut spec);
        spec.iter()
            .map(|s| s.norm_sqr() * self.dt * self.dt)
            .collect()
    }

    /// Apply the linear dispersion operator exp(D̂·h) in the frequency domain.
    ///
    /// D̂(ω) = i·φ_D(ω) − α/2, where φ_D(ω) is the dispersion phase per unit
    /// length (rad/m) as returned by `dispersion_phase`.  The complex exponential
    /// is applied in-place to `field_freq`, which must already be in FFT-order.
    fn apply_linear_propagator(&self, field_freq: &mut [Complex64], h: f64) {
        let n = field_freq.len();
        let omegas = omega_axis(n, self.dt);
        for (a, &om) in field_freq.iter_mut().zip(omegas.iter()) {
            let phi = self.dispersion_phase(om);
            let attenuation = (-self.alpha * h / 2.0).exp();
            *a *= Complex64::new(0.0, phi * h).exp() * attenuation;
        }
    }

    /// Compute the nonlinear operator N̂[A] = i·γ · (nonlinear polarisation).
    ///
    /// Returns i·γ times the nonlinear coupling field produced by
    /// `nonlinear_step`.  This operator is used as the right-hand-side source
    /// term in the RK4IP integrator.
    fn nl_operator(&self, field: &[Complex64], omega0: f64) -> Vec<Complex64> {
        let nl = self.nonlinear_step(field, omega0);
        nl.into_iter()
            .map(|v| v * Complex64::new(0.0, self.gamma))
            .collect()
    }

    /// Advance the field by one step of size `h` using the fourth-order
    /// Runge-Kutta in the interaction picture (RK4IP) method.
    ///
    /// Implements Table 1 of Hult, J. Lightw. Technol. 25, 3770 (2007).
    /// The interaction picture removes the stiff linear part so that the
    /// four-stage Runge-Kutta integration only acts on the nonlinear term.
    ///
    /// # Steps
    /// ```text
    ///   A_I  = IFFT[ exp(L̂·h/2) · FFT[A_n] ]   (half-step linear propagation)
    ///   k1   = h · N̂(A_I)
    ///   k2   = h · N̂( IFFT[exp(L̂·h/2)·FFT[A_I]] + k1/2 )
    ///   k3   = h · N̂( IFFT[exp(L̂·h/2)·FFT[A_I]] + k2/2 )
    ///   k4   = h · N̂( IFFT[exp(L̂·h) ·FFT[A_I + k3/h]] )
    ///   A_{n+1} = IFFT[exp(L̂·h/2) · FFT[A_I + k1/6 + k2/3 + k3/3]] + k4/6
    /// ```
    pub fn rk4ip_step(&self, field: &[Complex64], omega0: f64, h: f64) -> Vec<Complex64> {
        // Helper: apply L̂·h_half to a time-domain field and return time-domain result.
        let apply_half_disp = |f: &[Complex64]| -> Vec<Complex64> {
            let mut freq: Vec<Complex64> = f.to_vec();
            fft_inplace(&mut freq);
            self.apply_linear_propagator(&mut freq, h / 2.0);
            ifft_inplace(&mut freq);
            freq
        };

        let apply_full_disp = |f: &[Complex64]| -> Vec<Complex64> {
            let mut freq: Vec<Complex64> = f.to_vec();
            fft_inplace(&mut freq);
            self.apply_linear_propagator(&mut freq, h);
            ifft_inplace(&mut freq);
            freq
        };

        // A_I = exp(L̂·h/2) · A_n  — the interaction-picture field at z.
        let a_i = apply_half_disp(field);

        // k1 = h · N̂(A_I)
        let k1: Vec<Complex64> = self
            .nl_operator(&a_i, omega0)
            .into_iter()
            .map(|v| v * h)
            .collect();

        // k2 = h · N̂( exp(L̂·h/2)·A_I + k1/2 )
        let k2_input: Vec<Complex64> = apply_half_disp(&a_i)
            .into_iter()
            .zip(k1.iter())
            .map(|(a, &k)| a + k * 0.5)
            .collect();
        let k2: Vec<Complex64> = self
            .nl_operator(&k2_input, omega0)
            .into_iter()
            .map(|v| v * h)
            .collect();

        // k3 = h · N̂( exp(L̂·h/2)·A_I + k2/2 )
        let k3_input: Vec<Complex64> = apply_half_disp(&a_i)
            .into_iter()
            .zip(k2.iter())
            .map(|(a, &k)| a + k * 0.5)
            .collect();
        let k3: Vec<Complex64> = self
            .nl_operator(&k3_input, omega0)
            .into_iter()
            .map(|v| v * h)
            .collect();

        // k4 = h · N̂( exp(L̂·h) · (A_I + k3) )
        let k4_input: Vec<Complex64> = apply_full_disp(&a_i)
            .into_iter()
            .zip(k3.iter())
            .map(|(a, &k)| a + k)
            .collect();
        let k4: Vec<Complex64> = self
            .nl_operator(&k4_input, omega0)
            .into_iter()
            .map(|v| v * h)
            .collect();

        // A_{n+1} = exp(L̂·h/2) · (A_I + k1/6 + k2/3 + k3/3) + k4/6
        let combined: Vec<Complex64> = a_i
            .iter()
            .zip(k1.iter())
            .zip(k2.iter())
            .zip(k3.iter())
            .map(|(((a, k1v), k2v), k3v)| a + k1v / 6.0 + k2v / 3.0 + k3v / 3.0)
            .collect();
        let propagated = apply_half_disp(&combined);
        propagated
            .into_iter()
            .zip(k4.iter())
            .map(|(a, &k)| a + k / 6.0)
            .collect()
    }

    /// Propagate the input field adaptively using RK4IP with step-doubling
    /// error control (Sinkin et al., Opt. Express 11, 3514 (2003)).
    ///
    /// At each trial step of size `h`:
    /// 1. Take one RK4IP step of size `h`       → A_full.
    /// 2. Take two RK4IP steps of size `h/2`    → A_half (more accurate).
    /// 3. Local error = ||A_full − A_half||₂ / ||A_half||₂.
    /// 4. Accept if err < `tol`; use A_half as the new field.
    /// 5. Update h via the classical 5th-order control formula:
    ///    `h_new = h · min((tol/err)^0.2, h_max/h)`.
    /// 6. Reject if err ≥ tol (and h > h_min): halve h and retry.
    ///
    /// Returns `(output_field, number_of_accepted_steps)`.
    pub fn propagate_adaptive(
        &self,
        input_field: &[Complex64],
        omega0: f64,
        tol: f64,
    ) -> (Vec<Complex64>, usize) {
        let n = input_field.len();
        assert_eq!(
            n, self.n_time_points,
            "input_field length {} must equal n_time_points {}",
            n, self.n_time_points
        );

        let mut field = input_field.to_vec();
        let mut z = 0.0_f64;
        let z_end = self.fiber_length;
        let mut h = self.dz;
        let h_min = self.dz * 0.001;
        let h_max = self.dz * 16.0;
        let mut n_steps = 0_usize;

        while z < z_end {
            // Clamp h to remaining distance (never below h_min).
            h = h.min(z_end - z).max(h_min);

            // Full step with h.
            let a_full = self.rk4ip_step(&field, omega0, h);

            // Two half-steps with h/2 (higher-order accurate estimate).
            let a_half1 = self.rk4ip_step(&field, omega0, h / 2.0);
            let a_half = self.rk4ip_step(&a_half1, omega0, h / 2.0);

            // Local relative error estimate via step-doubling.
            let err_sq: f64 = a_full
                .iter()
                .zip(a_half.iter())
                .map(|(&af, &ah)| (af - ah).norm_sqr())
                .sum();
            let norm_sq: f64 = a_half.iter().map(|&a| a.norm_sqr()).sum();
            let err = if norm_sq > 1e-300 {
                (err_sq / norm_sq).sqrt()
            } else {
                0.0
            };

            if err < tol || h <= h_min {
                // Accept: advance with the more accurate two-half-step result.
                field = a_half;
                z += h;
                n_steps += 1;
                // Adjust step size (classical 5th-order control).
                if err > 0.0 {
                    h = (h * (tol / err).powf(0.2)).min(h_max);
                }
            } else {
                // Reject: shrink step and retry.
                h *= 0.5;
            }
        }

        (field, n_steps)
    }

    /// Estimate the 10 dB spectral bandwidth in nanometres.
    ///
    /// Converts the FFT-ordered spectrum to wavelength bins using λ = 2πc/(ω₀+Δω),
    /// finds the 10 dB threshold below the peak, and returns the width.
    pub fn spectral_bandwidth_nm(&self, spectrum: &[f64], center_lambda: f64) -> f64 {
        if spectrum.is_empty() {
            return 0.0;
        }
        let peak = spectrum.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if peak <= 0.0 {
            return 0.0;
        }
        let threshold = peak / 10.0_f64.powi(1); // 10 dB below peak
        let n = spectrum.len();
        let dw = 2.0 * PI / (n as f64 * self.dt);
        let c = 2.997_924_58e8_f64;
        let omega0 = 2.0 * PI * c / center_lambda;

        // Convert bins to wavelength; only keep bins where spectrum > threshold
        let mut lambdas_in_band: Vec<f64> = (0..n)
            .filter_map(|k| {
                if spectrum[k] < threshold {
                    return None;
                }
                let om = if k < n / 2 {
                    omega0 + k as f64 * dw
                } else {
                    omega0 + (k as f64 - n as f64) * dw
                };
                if om <= 0.0 {
                    return None;
                }
                Some(2.0 * PI * c / om * 1e9) // convert to nm
            })
            .collect();

        if lambdas_in_band.is_empty() {
            return 0.0;
        }
        lambdas_in_band.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lmin = lambdas_in_band[0];
        let lmax = *lambdas_in_band.last().unwrap_or(&lmin);
        lmax - lmin
    }
}

// ---------------------------------------------------------------------------
// ScFiberType
// ---------------------------------------------------------------------------

/// Fiber type determining the SC generation parameters.
#[derive(Debug, Clone)]
pub enum ScFiberType {
    /// Photonic crystal fiber — dominant platform for visible/NIR SC.
    Pcf {
        /// Hole pitch Λ (m).
        pitch: f64,
        /// Hole diameter d (m).
        hole_diameter: f64,
    },
    /// Standard Corning SMF-28 single-mode fiber.
    Smf28,
    /// Highly nonlinear fiber (HNLF) with tailored dispersion.
    Hnlf {
        /// Nonlinear coefficient γ (1/(W·m)).
        gamma: f64,
        /// GVD β₂ (s²/m).
        beta2: f64,
    },
    /// Chalcogenide glass fiber for mid-infrared SC.
    Chalcogenide {
        /// Nonlinear index n₂ (m²/W).
        n2: f64,
        /// GVD β₂ (s²/m).
        beta2: f64,
    },
}

impl ScFiberType {
    /// Effective nonlinear coefficient γ (1/(W·m)).
    pub fn gamma(&self) -> f64 {
        match self {
            ScFiberType::Pcf {
                pitch,
                hole_diameter,
            } => {
                // Empirical PCF nonlinearity (Agrawal §12):
                // γ ≈ 2πn₂/(λ A_eff) with A_eff ≈ pitch² · f(d/Λ)
                // For simplicity: γ ≈ 70 × (1 + d/pitch) × 10⁻³ /W/m
                let fill = hole_diameter / pitch;
                70e-3 * (1.0 + fill)
            }
            ScFiberType::Smf28 => 1.3e-3, // 1/(W·m) @ 1550 nm
            ScFiberType::Hnlf { gamma, .. } => *gamma,
            ScFiberType::Chalcogenide { n2, .. } => {
                // γ = 2π n₂ / (λ A_eff) with A_eff ≈ 10 µm², λ ≈ 2.5 µm
                let lambda = 2.5e-6;
                let a_eff = 10e-12; // 10 µm²
                2.0 * PI * n2 / (lambda * a_eff)
            }
        }
    }

    /// Group-velocity dispersion β₂ (s²/m) at the given wavelength.
    ///
    /// Sign convention: β₂ < 0 at λ > λ_ZDW (anomalous), β₂ > 0 at λ < λ_ZDW (normal).
    pub fn beta2_at(&self, wavelength: f64) -> f64 {
        match self {
            ScFiberType::Pcf {
                pitch,
                hole_diameter,
            } => {
                // PCF empirical model: β₂ ≈ S·(λ − λ_ZDW)
                // S ≈ -9e-20 s²/m² (negative slope: β₂ more negative for λ > λ_ZDW)
                let lambda_zdw = 1.05e-6 * (pitch / hole_diameter).sqrt().min(2.0);
                let slope = -9.0e-20; // s²/m² — anomalous for λ > λ_ZDW
                slope * (wavelength - lambda_zdw)
            }
            ScFiberType::Smf28 => {
                // SMF-28: ZDW ≈ 1.31 µm; β₂ ≈ −21.7 ps²/km @ 1550 nm
                // Slope S = β₂(1550) / (1550 - 1310 nm) ≈ -9.03e-20 s²/m²
                let lambda_zdw = 1.31e-6;
                let slope = -9.03e-20; // s²/m²
                slope * (wavelength - lambda_zdw)
            }
            ScFiberType::Hnlf { beta2, .. } => *beta2,
            ScFiberType::Chalcogenide { beta2, .. } => *beta2,
        }
    }

    /// Zero-dispersion wavelength λ_ZDW (m).
    pub fn zero_dispersion_wavelength(&self) -> f64 {
        match self {
            ScFiberType::Pcf {
                pitch,
                hole_diameter,
            } => 1.05e-6 * (pitch / hole_diameter).sqrt().min(2.0),
            ScFiberType::Smf28 => 1.31e-6,
            ScFiberType::Hnlf { beta2, .. } => {
                // Estimate ZDW: λ_ZDW = λ₀ − β₂/slope, around 1.55 µm
                let slope = -9.03e-20;
                let lambda0 = 1.55e-6;
                lambda0 - beta2 / slope
            }
            ScFiberType::Chalcogenide { beta2, .. } => {
                // Mid-IR chalcogenide: estimate ZDW around 2.5 µm
                let slope = -5.0e-20;
                let lambda0 = 2.5e-6;
                lambda0 - beta2 / slope
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PumpingRegime
// ---------------------------------------------------------------------------

/// Pumping regime for supercontinuum generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PumpingRegime {
    /// Anomalous GVD (β₂ < 0) at pump: soliton fission drives SC, typically
    /// incoherent for large soliton numbers.
    Anomalous,
    /// Normal GVD (β₂ > 0) at pump: SPM + optical wave breaking → coherent SC.
    Normal,
    /// Pump wavelength within 50 nm of ZDW: mixed or transitional regime.
    NearZdw,
}

// ---------------------------------------------------------------------------
// SupercontinuumSource
// ---------------------------------------------------------------------------

/// High-level descriptor of a supercontinuum source.
///
/// Provides empirical scaling-law estimates for bandwidth, soliton number,
/// and coherence without running a full GNLSE simulation.
#[derive(Debug, Clone)]
pub struct SupercontinuumSource {
    /// Pump centre wavelength (m).
    pub pump_wavelength: f64,
    /// Pump peak power P₀ (W).
    pub pump_power_peak: f64,
    /// Pump pulse duration (FWHM, s).
    pub pump_duration: f64,
    /// Fiber type.
    pub fiber_type: ScFiberType,
    /// Fiber length L (m).
    pub fiber_length: f64,
}

impl SupercontinuumSource {
    /// Construct a supercontinuum source.
    pub fn new(
        pump_wl: f64,
        peak_power: f64,
        duration: f64,
        fiber: ScFiberType,
        length: f64,
    ) -> Self {
        Self {
            pump_wavelength: pump_wl,
            pump_power_peak: peak_power,
            pump_duration: duration,
            fiber_type: fiber,
            fiber_length: length,
        }
    }

    /// Soliton number N = √(L_D / L_NL).
    ///
    /// The T₀ used is T₀ = T_FWHM / (2 ln(1+√2)) (sech-pulse convention).
    pub fn soliton_number(&self) -> f64 {
        let ln_factor = 2.0 * (1.0 + 2.0_f64.sqrt()).ln();
        let t0 = self.pump_duration / ln_factor;
        let beta2 = self.fiber_type.beta2_at(self.pump_wavelength);
        let gamma = self.fiber_type.gamma();
        if beta2.abs() < 1e-60 || gamma < 1e-30 || t0 < 1e-30 {
            return 0.0;
        }
        let ld = t0 * t0 / beta2.abs();
        let lnl = 1.0 / (gamma * self.pump_power_peak);
        (ld / lnl).sqrt()
    }

    /// Estimated SC 10 dB bandwidth in nm using an empirical scaling law.
    ///
    /// For anomalous pumping (soliton fission):
    ///   BW ≈ λ_pump² / (π c T₀) × N  (Dudley et al. 2006, Eq. 4.1)
    /// For normal pumping (OWB):
    ///   BW ≈ λ_pump² / (π c T₀) × √(N)
    pub fn estimated_bandwidth_nm(&self) -> f64 {
        let c = 2.997_924_58e8_f64;
        let n = self.soliton_number();
        if n < 1e-3 {
            return 0.0;
        }
        let ln_factor = 2.0 * (1.0 + 2.0_f64.sqrt()).ln();
        let t0 = self.pump_duration / ln_factor;
        // Base spectral extent from SPM
        let base_bw_hz = 1.0 / (PI * t0);
        let base_bw_nm = self.pump_wavelength * self.pump_wavelength / c * base_bw_hz * 1e9;
        match self.pumping_regime() {
            PumpingRegime::Anomalous => base_bw_nm * n,
            PumpingRegime::Normal => base_bw_nm * n.sqrt(),
            PumpingRegime::NearZdw => base_bw_nm * n * 0.7,
        }
    }

    /// Estimate temporal coherence length L_c of the SC output (s).
    ///
    /// Coherent (normal-pumping or low-N) SC: L_c ≈ pump duration T_FWHM.
    /// Incoherent (anomalous, large N) SC: L_c ≈ T_FWHM / N² (noise-seeded).
    pub fn temporal_coherence_estimate(&self) -> f64 {
        let n = self.soliton_number();
        match self.pumping_regime() {
            PumpingRegime::Normal => self.pump_duration,
            PumpingRegime::NearZdw => self.pump_duration / n.max(1.0),
            PumpingRegime::Anomalous => {
                // For large soliton numbers, MI-seeded noise dominates
                self.pump_duration / (n * n).max(1.0)
            }
        }
    }

    /// Simplified spectral power density (W/nm) at wavelength λ.
    ///
    /// Uses a super-Gaussian envelope centred at the pump, with width
    /// equal to the estimated bandwidth and peak power scaled by pulse energy.
    pub fn spectral_psd(&self, wavelength: f64) -> f64 {
        let bw_nm = self.estimated_bandwidth_nm();
        if bw_nm < 1e-6 {
            return 0.0;
        }
        let bw_m = bw_nm * 1e-9;
        let _c = 2.997_924_58e8_f64;
        let ln_factor = 2.0 * (1.0 + 2.0_f64.sqrt()).ln();
        let t0 = self.pump_duration / ln_factor;
        // Pulse energy ≈ 2 P₀ T₀ (sech pulse)
        let energy = 2.0 * self.pump_power_peak * t0;
        // Peak PSD = energy / bandwidth
        let peak_psd = energy / bw_m;
        // Super-Gaussian spectral envelope (order 2)
        let dl = wavelength - self.pump_wavelength;
        let sigma = bw_m / 2.355; // FWHM → σ
        peak_psd * (-0.5 * (dl / sigma).powi(2)).exp()
    }

    /// Determine the pumping regime from the GVD sign at the pump wavelength.
    pub fn pumping_regime(&self) -> PumpingRegime {
        let beta2 = self.fiber_type.beta2_at(self.pump_wavelength);
        let zdw = self.fiber_type.zero_dispersion_wavelength();
        let near_zdw_band = 50e-9; // 50 nm window
        if (self.pump_wavelength - zdw).abs() < near_zdw_band {
            PumpingRegime::NearZdw
        } else if beta2 < 0.0 {
            PumpingRegime::Anomalous
        } else {
            PumpingRegime::Normal
        }
    }
}

// ---------------------------------------------------------------------------
// OpticalWaveBreaking
// ---------------------------------------------------------------------------

/// Optical wave breaking (OWB) in the normal dispersion regime.
///
/// When a pulse propagates in normal GVD with strong SPM (N ≫ 1), the
/// frequency-chirped leading/trailing edges eventually overtake the pulse
/// wings, generating new spectral components — "wave breaking".
/// This is the dominant SC mechanism for normal-dispersion pumping.
///
/// Reference: Anderson et al., J. Opt. Soc. Am. B 9, 1358 (1992).
#[derive(Debug, Clone)]
pub struct OpticalWaveBreaking {
    /// GVD β₂ (s²/m), must be > 0 (normal dispersion).
    pub beta2: f64,
    /// Nonlinear coefficient γ (1/(W·m)).
    pub gamma: f64,
    /// 1/e half-width T₀ (s).
    pub t0: f64,
    /// Peak power P₀ (W).
    pub p0: f64,
}

impl OpticalWaveBreaking {
    /// Construct an OWB descriptor.
    ///
    /// # Panics
    /// Panics if `beta2` ≤ 0 (requires normal dispersion).
    pub fn new(beta2: f64, gamma: f64, t0: f64, p0: f64) -> Self {
        assert!(
            beta2 > 0.0,
            "OpticalWaveBreaking requires normal dispersion (beta2 > 0)"
        );
        Self {
            beta2,
            gamma,
            t0,
            p0,
        }
    }

    /// Dispersion length L_D = T₀² / β₂ (m).
    pub fn dispersion_length(&self) -> f64 {
        self.t0 * self.t0 / self.beta2
    }

    /// Nonlinear length L_NL = 1 / (γ P₀) (m).
    pub fn nonlinear_length(&self) -> f64 {
        if self.gamma < 1e-30 || self.p0 < 1e-30 {
            return f64::INFINITY;
        }
        1.0 / (self.gamma * self.p0)
    }

    /// Optical wave-breaking distance L_OWB (m).
    ///
    /// Anderson et al. give:
    /// ```text
    ///   L_OWB = L_D / √(exp(N²) − 1)
    /// ```
    /// For N ≫ 1 this simplifies to L_D / N.
    pub fn wave_breaking_distance(&self) -> f64 {
        let ld = self.dispersion_length();
        let lnl = self.nonlinear_length();
        if lnl.is_infinite() {
            return f64::INFINITY;
        }
        let n_sq = ld / lnl; // N² = L_D / L_NL
        let denom = (n_sq.exp() - 1.0).sqrt();
        if denom < 1e-15 {
            return f64::INFINITY;
        }
        ld / denom
    }

    /// FWHM temporal broadening factor at propagation distance z.
    ///
    /// Before OWB the pulse broadens as (Agrawal §4.2):
    /// ```text
    ///   T_FWHM(z) / T_FWHM(0) ≈ √(1 + (z/L_D)²)  [linear GVD only]
    /// ```
    /// With SPM, the effective broadening is accelerated:
    /// ```text
    ///   factor ≈ √(1 + (z/L_D + N² z²/(2L_D²))²)
    /// ```
    /// (First-order SPM correction to the chirp accumulation.)
    pub fn fwhm_broadening_factor(&self, z: f64) -> f64 {
        let ld = self.dispersion_length();
        let lnl = self.nonlinear_length();
        if ld < 1e-30 {
            return 1.0;
        }
        let n_sq = if lnl.is_finite() { ld / lnl } else { 0.0 };
        let xi = z / ld;
        // Linear GVD term + SPM-enhanced chirp
        let chirp = xi + n_sq * xi * xi / 2.0;
        (1.0 + chirp * chirp).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── FFT round-trip ────────────────────────────────────────────────────────

    #[test]
    fn fft_ifft_roundtrip_power_of_two() {
        let n = 128;
        let mut buf: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.1).sin(), 0.0))
            .collect();
        let original = buf.clone();
        fft_inplace(&mut buf);
        ifft_inplace(&mut buf);
        for (orig, rec) in original.iter().zip(buf.iter()) {
            assert_abs_diff_eq!(rec.re, orig.re, epsilon = 1e-10);
            assert_abs_diff_eq!(rec.im, orig.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn fft_single_tone() {
        // FFT of exp(i 2π k₀/N t) should be a delta at bin k₀
        let n = 64;
        let k0 = 5usize;
        let mut buf: Vec<Complex64> = (0..n)
            .map(|t| {
                let phase = 2.0 * PI * k0 as f64 * t as f64 / n as f64;
                Complex64::new(phase.cos(), phase.sin())
            })
            .collect();
        fft_inplace(&mut buf);
        // Bin k0 should dominate
        let peak_idx = buf
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.norm()
                    .partial_cmp(&b.norm())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(
            peak_idx, k0,
            "FFT peak should be at bin {k0}, got {peak_idx}"
        );
    }

    // ── GnlseSolver ──────────────────────────────────────────────────────────

    #[test]
    fn gnlse_dispersion_phase_zero_at_zero_omega() {
        let solver = GnlseSolver::new_silica(1.0, 1e-3, -20e-27, 0.1e-39);
        assert_abs_diff_eq!(solver.dispersion_phase(0.0), 0.0, epsilon = 1e-30);
    }

    #[test]
    fn gnlse_raman_response_zero_at_negative_t() {
        let solver = GnlseSolver::new_silica(1.0, 1e-3, -20e-27, 0.1e-39);
        assert_abs_diff_eq!(solver.raman_response(-1e-15), 0.0, epsilon = 1e-30);
    }

    #[test]
    fn gnlse_raman_response_positive_for_t_positive() {
        let solver = GnlseSolver::new_silica(1.0, 1e-3, -20e-27, 0.1e-39);
        // h_R(t) should be positive for small t > 0 (before first zero crossing at t = π τ₁)
        let t = 5e-15; // 5 fs < π × 12.2 fs ≈ 38 fs → in the positive lobe
        assert!(
            solver.raman_response(t) > 0.0,
            "h_R(5 fs) should be positive"
        );
    }

    #[test]
    fn gnlse_propagate_lossless_power_conservation() {
        // With α = 0, no Raman, no SPM (γ = 0), pure dispersion should conserve
        // total power (Parseval).
        let mut solver = GnlseSolver::new_silica(0.1, 0.0, -20e-27, 0.0);
        solver.alpha = 0.0;
        solver.raman_fraction = 0.0;
        let n = solver.n_time_points;
        // Gaussian pulse at centre
        let t0 = 50e-15;
        let field: Vec<Complex64> = (0..n)
            .map(|i| {
                let t = (i as f64 - n as f64 / 2.0) * solver.dt;
                Complex64::new((-0.5 * (t / t0).powi(2)).exp(), 0.0)
            })
            .collect();
        let power_in: f64 = field.iter().map(|a| a.norm_sqr()).sum::<f64>() * solver.dt;
        let output = solver.propagate(&field, 2.0 * PI * 2.998e8 / 1550e-9);
        let power_out: f64 = output.iter().map(|a| a.norm_sqr()).sum::<f64>() * solver.dt;
        let rel_err = (power_out - power_in).abs() / power_in;
        assert!(
            rel_err < 0.05,
            "Power not conserved (lossless, no NL): rel_err = {rel_err:.3e}"
        );
    }

    #[test]
    fn gnlse_output_spectrum_length_matches_input() {
        let solver = GnlseSolver::new_silica(0.01, 1e-3, -20e-27, 0.1e-39);
        let n = solver.n_time_points;
        let field: Vec<Complex64> = vec![Complex64::new(1.0, 0.0); n];
        let spec = solver.output_spectrum(&field, 1.21e15);
        assert_eq!(spec.len(), n);
    }

    // ── ScFiberType ───────────────────────────────────────────────────────────

    #[test]
    fn fiber_type_smf28_gamma() {
        let g = ScFiberType::Smf28.gamma();
        assert_abs_diff_eq!(g, 1.3e-3, epsilon = 1e-10);
    }

    #[test]
    fn fiber_type_hnlf_returns_given_gamma() {
        let g = 10e-3;
        let fiber = ScFiberType::Hnlf {
            gamma: g,
            beta2: -1e-27,
        };
        assert_abs_diff_eq!(fiber.gamma(), g, epsilon = 1e-20);
    }

    #[test]
    fn fiber_type_pcf_zdw_reasonable() {
        // For typical PCF (pitch = 2 µm, d = 1 µm) ZDW should be ≈ 1–1.5 µm
        let fiber = ScFiberType::Pcf {
            pitch: 2e-6,
            hole_diameter: 1e-6,
        };
        let zdw = fiber.zero_dispersion_wavelength();
        assert!(
            zdw > 0.5e-6 && zdw < 2e-6,
            "PCF ZDW = {zdw:.2e} should be ~1 µm"
        );
    }

    // ── SupercontinuumSource ─────────────────────────────────────────────────

    #[test]
    fn sc_source_soliton_number_positive_anomalous() {
        // SMF-28 pumped at 1550 nm (anomalous): N should be > 0
        let src = SupercontinuumSource::new(
            1550e-9,
            1000.0,  // 1 kW peak
            100e-15, // 100 fs
            ScFiberType::Smf28,
            1.0,
        );
        let n = src.soliton_number();
        assert!(n > 0.0, "Soliton number should be positive, got {n}");
    }

    #[test]
    fn sc_source_anomalous_regime_detection() {
        // SMF-28 @ 1550 nm: β₂ < 0 → anomalous
        let src = SupercontinuumSource::new(1550e-9, 1000.0, 100e-15, ScFiberType::Smf28, 1.0);
        assert_eq!(src.pumping_regime(), PumpingRegime::Anomalous);
    }

    #[test]
    fn sc_source_normal_regime_at_1300nm() {
        // SMF-28 @ 1300 nm: λ < ZDW (1310 nm) → β₂ < 0, but near ZDW
        // Pump at 1200 nm (well below ZDW) → β₂ > 0 → normal
        let src = SupercontinuumSource::new(1200e-9, 1000.0, 100e-15, ScFiberType::Smf28, 1.0);
        // β₂ @ 1200 nm: slope*(1200nm - 1310nm) > 0 → normal
        assert_eq!(src.pumping_regime(), PumpingRegime::Normal);
    }

    #[test]
    fn sc_source_estimated_bandwidth_positive() {
        let src = SupercontinuumSource::new(1550e-9, 1000.0, 100e-15, ScFiberType::Smf28, 1.0);
        let bw = src.estimated_bandwidth_nm();
        assert!(bw > 0.0, "Estimated bandwidth must be positive, got {bw}");
    }

    #[test]
    fn sc_source_spectral_psd_peak_at_pump() {
        let pump = 1550e-9;
        let src = SupercontinuumSource::new(pump, 1000.0, 100e-15, ScFiberType::Smf28, 1.0);
        let psd_at_pump = src.spectral_psd(pump);
        let psd_off = src.spectral_psd(pump + 200e-9);
        assert!(psd_at_pump > psd_off, "PSD should peak at pump wavelength");
    }

    // ── OpticalWaveBreaking ───────────────────────────────────────────────────

    #[test]
    fn owb_dispersion_length_positive() {
        let owb = OpticalWaveBreaking::new(20e-27, 1.3e-3, 1e-12, 1000.0);
        assert!(owb.dispersion_length() > 0.0);
    }

    #[test]
    fn owb_nonlinear_length_positive() {
        let owb = OpticalWaveBreaking::new(20e-27, 1.3e-3, 1e-12, 1000.0);
        assert!(owb.nonlinear_length().is_finite() && owb.nonlinear_length() > 0.0);
    }

    #[test]
    fn owb_wave_breaking_distance_finite_for_spm() {
        let owb = OpticalWaveBreaking::new(20e-27, 1.3e-3, 1e-12, 1000.0);
        let d = owb.wave_breaking_distance();
        assert!(
            d.is_finite() && d > 0.0,
            "OWB distance should be finite and positive: {d}"
        );
    }

    #[test]
    fn owb_broadening_factor_one_at_z_zero() {
        let owb = OpticalWaveBreaking::new(20e-27, 1.3e-3, 1e-12, 100.0);
        assert_abs_diff_eq!(owb.fwhm_broadening_factor(0.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn owb_broadening_increases_with_z() {
        let owb = OpticalWaveBreaking::new(20e-27, 1.3e-3, 1e-12, 100.0);
        let f1 = owb.fwhm_broadening_factor(10.0);
        let f2 = owb.fwhm_broadening_factor(20.0);
        assert!(
            f2 > f1,
            "Broadening should increase with z: f(10)={f1:.4}, f(20)={f2:.4}"
        );
    }
}
