/// THz time-domain spectroscopy (THz-TDS) and THz waveguide mode analysis.
use num_complex::Complex64;

use crate::error::OxiPhotonError;

// ─── Physical constants ────────────────────────────────────────────────────
const C0: f64 = 2.997_924_58e8; // m/s

// ─── THz-TDS system ───────────────────────────────────────────────────────

/// THz time-domain spectroscopy (THz-TDS) system description and analysis.
///
/// A THz-TDS system generates sub-ps THz pulses, transmits them through a
/// sample, and records the transmitted time-domain waveform.  Fourier
/// transformation of the sample and reference waveforms gives the complex
/// transfer function from which the complex refractive index is extracted.
#[derive(Debug, Clone)]
pub struct ThzTds {
    /// THz emitter.
    pub emitter: crate::thz::generation::PhotoconductiveAntenna,
    /// Measurable frequency range (THz): (f_min, f_max).
    pub frequency_range_thz: (f64, f64),
    /// Number of time samples N.
    pub n_time_points: usize,
    /// Time step Δt (ps).
    pub time_resolution_ps: f64,
    /// Typical peak dynamic range (dB).
    pub dynamic_range_db: f64,
}

impl ThzTds {
    /// General constructor.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        emitter: crate::thz::generation::PhotoconductiveAntenna,
        f_min_thz: f64,
        f_max_thz: f64,
        n_time_points: usize,
        time_resolution_ps: f64,
        dynamic_range_db: f64,
    ) -> Self {
        Self {
            emitter,
            frequency_range_thz: (f_min_thz, f_max_thz),
            n_time_points,
            time_resolution_ps,
            dynamic_range_db,
        }
    }

    /// A representative laboratory THz-TDS system.
    ///
    /// LT-GaAs emitter/detector pair, 0–3 THz range, 1024 points at 0.05 ps step.
    pub fn standard_system() -> Self {
        Self::new(
            crate::thz::generation::PhotoconductiveAntenna::lt_gaas_standard(),
            0.1, // THz
            3.0, // THz
            1024,
            0.05, // ps
            70.0, // dB
        )
    }

    /// Spectral resolution δf = 1 / (N · Δt) (GHz).
    pub fn spectral_resolution_ghz(&self) -> f64 {
        let total_time_ps = self.n_time_points as f64 * self.time_resolution_ps;
        1.0 / total_time_ps * 1e3 // (THz → GHz)
    }

    /// Usable bandwidth determined by the emitter bandwidth and the frequency
    /// range setting.
    pub fn usable_bandwidth_thz(&self) -> f64 {
        let emitter_bw = self.emitter.bandwidth_thz();
        let set_bw = self.frequency_range_thz.1 - self.frequency_range_thz.0;
        emitter_bw.min(set_bw)
    }

    /// Positive-frequency axis for the DFT output (THz).
    ///
    /// Returns N/2 values from 0 to the Nyquist frequency.
    pub fn frequency_axis_thz(&self) -> Vec<f64> {
        let n_pos = self.n_time_points / 2;
        let df_thz = 1.0 / (self.n_time_points as f64 * self.time_resolution_ps); // THz
        (0..n_pos).map(|k| k as f64 * df_thz).collect()
    }

    /// Time axis (ps).
    pub fn time_axis_ps(&self) -> Vec<f64> {
        (0..self.n_time_points)
            .map(|i| i as f64 * self.time_resolution_ps)
            .collect()
    }

    /// Extract the complex refractive index from sample and reference waveforms.
    ///
    /// The algorithm uses the standard THz-TDS inversion:
    ///
    /// ```text
    /// H(ω) = E_sam(ω) / E_ref(ω) = exp(i ω (n-1) d / c) · exp(-ω k d / c)
    /// ⟹  n(ω) = 1 + c·φ(ω) / (ω·d)
    ///     k(ω) = -c·ln|H(ω)| / (ω·d)
    /// ```
    ///
    /// Fabry-Pérot etalon reflections are neglected (thin-sample approximation).
    ///
    /// # Arguments
    /// * `e_sample`            — THz waveform recorded through the sample.
    /// * `e_reference`         — THz waveform recorded without the sample (air).
    /// * `sample_thickness_mm` — physical thickness of the sample (mm).
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::NumericalError`] if the waveforms have different
    /// lengths or if the sample thickness is non-positive.
    pub fn extract_complex_index(
        &self,
        e_sample: &[f64],
        e_reference: &[f64],
        sample_thickness_mm: f64,
    ) -> Result<Vec<(f64, Complex64)>, OxiPhotonError> {
        if e_sample.len() != e_reference.len() {
            return Err(OxiPhotonError::NumericalError(
                "Sample and reference waveforms must have equal length".into(),
            ));
        }
        if sample_thickness_mm <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "Sample thickness must be positive".into(),
            ));
        }
        let d_m = sample_thickness_mm * 1e-3;

        let e_sam_f = self.fft_waveform(e_sample);
        let e_ref_f = self.fft_waveform(e_reference);

        let freq_axis = self.frequency_axis_thz();
        let n_pos = freq_axis.len();

        let mut result = Vec::with_capacity(n_pos);
        for (k, &freq_thz) in freq_axis.iter().enumerate() {
            if freq_thz < 1e-6 {
                // DC — skip
                result.push((freq_thz, Complex64::new(1.0, 0.0)));
                continue;
            }
            let omega = 2.0 * std::f64::consts::PI * freq_thz * 1e12; // rad/s

            let h = if e_ref_f[k].norm() > 1e-60 {
                e_sam_f[k] / e_ref_f[k]
            } else {
                Complex64::new(1.0, 0.0)
            };

            // Phase of H gives n, amplitude gives k
            let phi = h.arg();
            let n_real = 1.0 + C0 * phi / (omega * d_m);
            let n_imag = if h.norm() > 1e-30 {
                -C0 * h.norm().ln() / (omega * d_m)
            } else {
                0.0
            };
            result.push((freq_thz, Complex64::new(n_real, n_imag)));
        }
        Ok(result)
    }

    /// Absorption coefficient α = 2 ω k / c  (cm⁻¹).
    ///
    /// # Arguments
    /// * `k`        — extinction coefficient (imaginary part of refractive index).
    /// * `freq_thz` — frequency (THz).
    pub fn absorption_coefficient(k: f64, freq_thz: f64) -> f64 {
        let omega = 2.0 * std::f64::consts::PI * freq_thz * 1e12;
        let alpha_m = 2.0 * omega * k / C0; // m⁻¹
        alpha_m * 1e-2 // m⁻¹ → cm⁻¹
    }

    /// DFT of a real-valued time-domain waveform via a plain O(N²) algorithm.
    ///
    /// Returns N complex coefficients (full spectrum); callers typically use
    /// only the first N/2 (positive frequencies).
    pub fn fft_waveform(&self, waveform: &[f64]) -> Vec<Complex64> {
        let n = waveform.len();
        if n == 0 {
            return Vec::new();
        }
        let two_pi_over_n = 2.0 * std::f64::consts::PI / n as f64;
        (0..n)
            .map(|k| {
                let mut re = 0.0_f64;
                let mut im = 0.0_f64;
                for (j, &s) in waveform.iter().enumerate() {
                    let angle = two_pi_over_n * k as f64 * j as f64;
                    re += s * angle.cos();
                    im -= s * angle.sin();
                }
                Complex64::new(re, im)
            })
            .collect()
    }

    /// Simulate the THz waveform transmitted through a sample with the given
    /// complex refractive index dispersion.
    ///
    /// The reference waveform is multiplied in the frequency domain by the
    /// complex transmission function T(ω) and then inverse-DFT'd.
    ///
    /// # Arguments
    /// * `reference_waveform` — THz time-domain waveform in air.
    /// * `n_complex`          — frequency-resolved complex index `(freq_THz, n+ik)`.
    /// * `thickness_mm`       — sample thickness (mm).
    pub fn simulate_transmission(
        &self,
        reference_waveform: &[f64],
        n_complex: &[(f64, Complex64)],
        thickness_mm: f64,
    ) -> Vec<f64> {
        let n_pts = reference_waveform.len();
        if n_pts == 0 || n_complex.is_empty() {
            return vec![0.0; n_pts];
        }
        let d_m = thickness_mm * 1e-3;

        let e_ref_f = self.fft_waveform(reference_waveform);

        // Build transmission-modified spectrum
        let mut e_sam_f = e_ref_f.clone();
        for (k, (freq_thz, n_c)) in n_complex.iter().enumerate() {
            if k >= n_pts {
                break;
            }
            if *freq_thz < 1e-6 {
                continue;
            }
            let omega = 2.0 * std::f64::consts::PI * freq_thz * 1e12;
            // H(ω) = exp(i ω (n-1) d / c) · exp(-ω k d / c)
            let phase = omega * (n_c.re - 1.0) * d_m / C0;
            let attenuation = (-omega * n_c.im * d_m / C0).exp();
            let transfer = Complex64::new(0.0, phase).exp() * attenuation;
            e_sam_f[k] *= transfer;
        }

        // Inverse DFT
        self.idft_to_real(&e_sam_f)
    }

    /// Signal-to-noise ratio at each frequency bin (dB).
    ///
    /// Returns `(freq_thz, snr_db)` pairs for the positive-frequency axis.
    pub fn snr_db(&self, signal: &[f64], noise: &[f64]) -> Vec<(f64, f64)> {
        let sig_f = self.fft_waveform(signal);
        let noi_f = self.fft_waveform(noise);
        let freq_axis = self.frequency_axis_thz();

        freq_axis
            .into_iter()
            .enumerate()
            .map(|(k, f)| {
                let s_pow = sig_f[k].norm_sqr();
                let n_pow = noi_f[k].norm_sqr();
                let snr = if n_pow > 1e-60 {
                    10.0 * (s_pow / n_pow).log10()
                } else {
                    f64::INFINITY
                };
                (f, snr)
            })
            .collect()
    }

    // ── Private helpers ──────────────────────────────────────────────────

    /// Plain O(N²) inverse DFT returning the real part of the time signal.
    fn idft_to_real(&self, spectrum: &[Complex64]) -> Vec<f64> {
        let n = spectrum.len();
        let scale = 1.0 / n as f64;
        let two_pi_over_n = 2.0 * std::f64::consts::PI / n as f64;
        (0..n)
            .map(|j| {
                let mut val = 0.0_f64;
                for (k, s) in spectrum.iter().enumerate() {
                    let angle = two_pi_over_n * k as f64 * j as f64;
                    val += s.re * angle.cos() - s.im * angle.sin();
                }
                val * scale
            })
            .collect()
    }
}

// ─── THz Waveguide ────────────────────────────────────────────────────────

/// Classification of THz waveguide geometry.
#[derive(Debug, Clone)]
pub enum ThzGuideType {
    /// Parallel-plate metallic waveguide (TE modes, very low ohmic loss).
    ParallelPlate,
    /// Rectangular metallic waveguide with a given height (mm).
    Rectangular {
        /// Inner height b (mm).
        height_mm: f64,
    },
    /// Circular metallic tube with inner radius (mm).
    CircularMetal {
        /// Inner radius r (mm).
        radius_mm: f64,
    },
    /// Dielectric rod waveguide with given core radius (mm).
    DielectricRod {
        /// Core radius (mm).
        radius_mm: f64,
    },
}

/// THz waveguide — supports mode analysis for common guide geometries.
#[derive(Debug, Clone)]
pub struct ThzWaveguide {
    /// Primary transverse dimension (plate separation or inner width) (mm).
    pub width_mm: f64,
    /// Guide axial length (mm).
    pub length_mm: f64,
    /// Geometry and secondary dimensions.
    pub guide_type: ThzGuideType,
    /// Refractive index of the filling medium (1.0 for air).
    pub material_index: f64,
}

impl ThzWaveguide {
    /// General constructor.
    pub fn new(
        width_mm: f64,
        length_mm: f64,
        guide_type: ThzGuideType,
        material_index: f64,
    ) -> Self {
        Self {
            width_mm,
            length_mm,
            guide_type,
            material_index,
        }
    }

    /// Cutoff frequency for the dominant (TE₁₀ / TE₁) mode (THz).
    ///
    /// - Parallel plate (TE₁):  f_c = c / (2 · a · n)
    /// - Rectangular TE₁₀:      same formula with `width_mm` as the broad wall
    /// - Circular TE₁₁:         f_c = 1.841 · c / (2π · r · n)
    /// - Dielectric rod HE₁₁:   f_c ≈ 0 (guided from DC in ideal rod)
    pub fn cutoff_frequency_thz(&self) -> f64 {
        let n = self.material_index;
        match &self.guide_type {
            ThzGuideType::ParallelPlate | ThzGuideType::Rectangular { .. } => {
                let a_m = self.width_mm * 1e-3;
                C0 / (2.0 * a_m * n) * 1e-12 // Hz → THz
            }
            ThzGuideType::CircularMetal { radius_mm } => {
                let r_m = radius_mm * 1e-3;
                1.841 * C0 / (2.0 * std::f64::consts::PI * r_m * n) * 1e-12
            }
            ThzGuideType::DielectricRod { .. } => {
                // HE₁₁ has no cut-off in a lossless unbounded rod
                0.0
            }
        }
    }

    /// Propagation constant β (rad m⁻¹) at frequency `freq_thz` (THz).
    ///
    /// β = (ω·n/c) · sqrt(1 − (f_c/f)²)
    ///
    /// Returns 0.0 below cut-off.
    pub fn propagation_constant(&self, freq_thz: f64) -> f64 {
        let f_c = self.cutoff_frequency_thz();
        if freq_thz <= f_c {
            return 0.0;
        }
        let omega = 2.0 * std::f64::consts::PI * freq_thz * 1e12;
        let ratio = f_c / freq_thz;
        omega * self.material_index / C0 * (1.0 - ratio * ratio).sqrt()
    }

    /// Group velocity v_g = c/n · sqrt(1 − (f_c/f)²)  (m s⁻¹).
    ///
    /// Returns 0.0 at or below cut-off.
    pub fn group_velocity(&self, freq_thz: f64) -> f64 {
        let f_c = self.cutoff_frequency_thz();
        if freq_thz <= f_c {
            return 0.0;
        }
        let ratio = f_c / freq_thz;
        C0 / self.material_index * (1.0 - ratio * ratio).sqrt()
    }

    /// Ohmic (wall-loss) attenuation coefficient α_c (dB cm⁻¹) for copper
    /// at room temperature.
    ///
    /// Based on the standard TE₁₀ parallel-plate / rectangular waveguide formula:
    /// α_c ≈ Rs / (η · a · sqrt(1 − (f_c/f)²))
    ///
    /// Where Rs = sqrt(π f μ₀ / σ_Cu) is the surface resistance and
    /// η = 377 Ω (free-space impedance, corrected by n and propagation factor).
    pub fn ohmic_loss_db_per_cm(&self, freq_thz: f64) -> f64 {
        let f_c = self.cutoff_frequency_thz();
        if freq_thz <= f_c {
            return f64::INFINITY;
        }

        // Copper conductivity σ ≈ 5.8 × 10⁷ S/m
        let sigma_cu = 5.8e7_f64; // S/m
        let mu0 = 4.0 * std::f64::consts::PI * 1e-7;

        let freq_hz = freq_thz * 1e12;
        // Surface resistance Rs = sqrt(π f μ₀ / σ)
        let rs = (std::f64::consts::PI * freq_hz * mu0 / sigma_cu).sqrt();

        let a_m = self.width_mm * 1e-3;
        let eta0 = 377.0_f64; // Ω — free-space impedance
        let eta = eta0 / self.material_index;

        let ratio = f_c / freq_thz;
        let te_factor = (1.0 - ratio * ratio).sqrt();

        // α in Np/m
        let alpha_np_per_m = rs / (eta * a_m * te_factor);
        // Convert Np/m → dB/cm: 1 Np = 8.686 dB, 1 m = 100 cm
        alpha_np_per_m * 8.686 / 100.0
    }

    /// Propagation time delay τ = L / v_g  (ps).
    ///
    /// Returns 0.0 below cut-off.
    pub fn propagation_delay_ps(&self, freq_thz: f64) -> f64 {
        let vg = self.group_velocity(freq_thz);
        if vg < 1e-10 {
            return 0.0;
        }
        let l_m = self.length_mm * 1e-3;
        l_m / vg * 1e12 // s → ps
    }

    /// Group delay dispersion (GDD) = d²β/dω²  (ps² mm⁻¹).
    ///
    /// For a metallic waveguide:
    /// β(ω) = (n/c) · sqrt(ω² − ω_c²)
    /// ⟹  d²β/dω² = -n · ω_c² / (c · (ω² − ω_c²)^(3/2))
    pub fn gdd_ps2_per_mm(&self, freq_thz: f64) -> f64 {
        let f_c = self.cutoff_frequency_thz();
        if freq_thz <= f_c {
            return f64::INFINITY;
        }
        let omega = 2.0 * std::f64::consts::PI * freq_thz * 1e12;
        let omega_c = 2.0 * std::f64::consts::PI * f_c * 1e12;
        let delta_sq = omega * omega - omega_c * omega_c;

        // d²β/dω² in s²/m
        let gdd_s2_per_m = -self.material_index * omega_c * omega_c / (C0 * delta_sq.powf(1.5));

        // Convert s²/m → ps²/mm
        // 1 s²/m = (1e12 ps)² / (1e3 mm) = 1e24/1e3 ps²/mm = 1e21 ps²/mm
        gdd_s2_per_m * 1e21
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn standard_tds() -> ThzTds {
        ThzTds::standard_system()
    }

    #[test]
    fn test_spectral_resolution() {
        let tds = standard_tds();
        // δf = 1/(N·Δt) where N=1024, Δt=0.05 ps → δf = 1/(51.2 ps) ≈ 19.5 GHz
        let delta_f = tds.spectral_resolution_ghz();
        let expected = 1.0 / (1024.0 * 0.05e-3); // THz → GHz: ×1000
                                                 // expected = 1/(0.0512 ps·1e-3) = ... let's compute directly
        let expected_ghz = 1.0 / (1024.0_f64 * 0.05) * 1e3; // ps×1e-3 → THz, then ×1e3 → GHz
        assert!(
            (delta_f - expected_ghz).abs() < 1e-6,
            "δf = {delta_f} GHz, expected ≈ {expected_ghz} GHz (alt {expected})"
        );
    }

    #[test]
    fn test_frequency_axis_length() {
        let tds = standard_tds();
        let freq = tds.frequency_axis_thz();
        assert_eq!(
            freq.len(),
            tds.n_time_points / 2,
            "frequency axis must have N/2 elements"
        );
    }

    #[test]
    fn test_absorption_coefficient_formula() {
        // α = 2ω k / c; at f=1 THz, k=0.1 → α in cm⁻¹
        let alpha = ThzTds::absorption_coefficient(0.1, 1.0);
        let omega = 2.0 * std::f64::consts::PI * 1e12;
        let expected = 2.0 * omega * 0.1 / C0 * 1e-2; // m⁻¹ → cm⁻¹
        assert!(
            (alpha - expected).abs() < 1e-6,
            "α = {alpha} cm⁻¹, expected {expected} cm⁻¹"
        );
    }

    #[test]
    fn test_waveguide_cutoff_frequency() {
        // Parallel-plate, a = 1 mm, n = 1 → f_c = c/(2×0.001) = 150 GHz = 0.15 THz
        let wg = ThzWaveguide::new(1.0, 50.0, ThzGuideType::ParallelPlate, 1.0);
        let f_c = wg.cutoff_frequency_thz();
        let expected = C0 / (2.0 * 1e-3) * 1e-12;
        assert!(
            (f_c - expected).abs() < 1e-6,
            "f_c = {f_c} THz, expected {expected} THz"
        );
    }

    #[test]
    fn test_group_velocity_below_c() {
        let wg = ThzWaveguide::new(1.0, 50.0, ThzGuideType::ParallelPlate, 1.0);
        // Well above cutoff (f_c ≈ 0.15 THz)
        let vg = wg.group_velocity(1.0);
        assert!(
            vg > 0.0 && vg < C0,
            "group velocity {vg} m/s must be in (0, c)"
        );
    }

    #[test]
    fn test_propagation_delay_positive() {
        let wg = ThzWaveguide::new(1.0, 50.0, ThzGuideType::ParallelPlate, 1.0);
        let delay = wg.propagation_delay_ps(1.0);
        assert!(
            delay > 0.0,
            "propagation delay must be positive, got {delay}"
        );
    }

    #[test]
    fn test_fft_waveform_length() {
        let tds = standard_tds();
        let waveform: Vec<f64> = (0..64).map(|i| (i as f64).sin()).collect();
        let spectrum = tds.fft_waveform(&waveform);
        assert_eq!(
            spectrum.len(),
            64,
            "FFT output must have the same length as input"
        );
    }
}
