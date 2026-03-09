//! Arrayed Waveguide Grating (AWG) demultiplexer models.
//!
//! Implements a phased-array WDM demultiplexer model using Gaussian spectral
//! channel profiles.  Also provides an ITU-T grid calculator for standard
//! WDM channel plans.
//!
//! # Physical model
//!
//! The AWG consists of:
//! 1. Input star coupler (free-propagation region)
//! 2. Array of waveguides with a constant path-length increment ΔL
//! 3. Output star coupler routing each wavelength to a different output port
//!
//! The grating order is `m = n_eff · ΔL / λ_center`.
//! The channel spacing satisfies: `Δλ = λ_center² / (m · n_g · ΔL)`.
//!
//! Channel transmission is approximated by a Gaussian profile:
//! ```text
//! T(λ) = exp(-(λ - λ_ch)² / (2 σ²))
//! ```
//! where `σ = channel_bandwidth_nm / (2 sqrt(2 ln 2))`.
//!
//! # References
//! - Smit & van Dam, "PHASAR-based WDM-devices", IEEE J. Sel. Topics Quantum
//!   Electron. 2 (1996)
//! - Okamoto, "Fundamentals of Optical Waveguides", 2nd ed. (2006)

use crate::error::{OxiPhotonError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Physical constants
// ─────────────────────────────────────────────────────────────────────────────

/// Speed of light in nm·THz = 10⁻⁶ m · 10¹² Hz = 10⁶ m·Hz = 10⁶ m/s × 10⁶
/// Actually: c = 299792.458 nm·THz (since 1 nm × 1 THz = 1 nm × 10¹² Hz = 10³ m/s × 10⁻⁶ = …)
/// Let's be precise: c [m/s] = λ[m] × f[Hz]
/// → c [nm × THz] = c [m/s] / 10³  (1 THz = 10¹² Hz, 1 nm = 10⁻⁹ m, so nm·THz = 10⁻⁹·10¹²=10³ m/s)
/// c = 2.99792458e8 m/s → c = 2.99792458e8 / 10³ nm·THz = 2.99792458e5 nm·THz
const C_NM_THZ: f64 = 299_792.458; // nm·THz

// ─────────────────────────────────────────────────────────────────────────────
// ArrayedWaveguideGrating
// ─────────────────────────────────────────────────────────────────────────────

/// Arrayed Waveguide Grating (AWG) demultiplexer.
///
/// Models both the spectral demultiplexing function and NxN routing capability.
#[derive(Debug, Clone)]
pub struct ArrayedWaveguideGrating {
    /// Number of output channels.
    pub n_channels: usize,
    /// Wavelength channel spacing (nm).
    pub channel_spacing_nm: f64,
    /// Center/reference wavelength (nm).
    pub center_wavelength_nm: f64,
    /// Effective refractive index of array waveguides.
    pub n_eff_array: f64,
    /// Group refractive index of array waveguides.
    pub n_g_array: f64,
    /// Path-length increment between adjacent array waveguides (μm).
    pub delta_l_um: f64,
    /// Total insertion loss (dB) — applied uniformly.
    pub insertion_loss_db: f64,
    /// Adjacent-channel crosstalk (dB, negative means isolation).
    pub crosstalk_db: f64,
    /// Number of array waveguides (affects sidelobe level).
    pub n_waveguides: usize,
}

impl ArrayedWaveguideGrating {
    /// Create a new AWG.
    ///
    /// # Arguments
    /// * `n_channels`          – number of WDM output channels
    /// * `channel_spacing_nm`  – channel spacing in nm
    /// * `center_wavelength_nm`– center wavelength in nm
    /// * `n_eff_array`         – effective index of array waveguides
    /// * `n_g_array`           – group index of array waveguides
    /// * `delta_l_um`          – path-length increment ΔL in μm
    /// * `n_waveguides`        – number of array waveguides
    /// * `insertion_loss_db`   – insertion loss in dB
    /// * `crosstalk_db`        – adjacent-channel crosstalk in dB
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_channels: usize,
        channel_spacing_nm: f64,
        center_wavelength_nm: f64,
        n_eff_array: f64,
        n_g_array: f64,
        delta_l_um: f64,
        n_waveguides: usize,
        insertion_loss_db: f64,
        crosstalk_db: f64,
    ) -> Self {
        Self {
            n_channels,
            channel_spacing_nm,
            center_wavelength_nm,
            n_eff_array,
            n_g_array,
            delta_l_um,
            insertion_loss_db,
            crosstalk_db,
            n_waveguides,
        }
    }

    /// Create a standard C-band 100 GHz AWG (ITU grid, 40 channels).
    pub fn c_band_100ghz(n_channels: usize) -> Self {
        // 100 GHz ≈ 0.8 nm at 1550 nm
        let channel_spacing_nm = 0.8;
        Self::new(
            n_channels,
            channel_spacing_nm,
            1550.0,
            2.4, // typical Si strip waveguide n_eff
            4.2, // typical Si strip waveguide n_g
            // ΔL from grating order: m = n_eff·ΔL/λ = n_g·λ/Δλ
            // ΔL = n_g·λ²/(n_eff·n_channels·Δλ)
            {
                let lambda = 1550.0_f64;
                let n_g = 4.2_f64;
                let n_eff = 2.4_f64;
                let delta_lambda = channel_spacing_nm;
                n_g * lambda * lambda / (n_eff * delta_lambda * 1_000.0)
            },
            100,   // 100 array waveguides
            3.0,   // 3 dB insertion loss
            -30.0, // -30 dB crosstalk
        )
    }

    /// Center wavelengths of all output channels (nm).
    ///
    /// Channel 0 is at `center_wavelength_nm - (n_channels/2) * channel_spacing_nm`.
    /// The center channel is at `center_wavelength_nm`.
    pub fn channel_wavelengths_nm(&self) -> Vec<f64> {
        let half = (self.n_channels as f64 - 1.0) / 2.0;
        (0..self.n_channels)
            .map(|i| self.center_wavelength_nm + (i as f64 - half) * self.channel_spacing_nm)
            .collect()
    }

    /// 3-dB bandwidth of each AWG channel (nm).
    ///
    /// Approximated as 50% of the channel spacing (Gaussian profile).
    pub fn channel_bandwidth_nm(&self) -> f64 {
        self.channel_spacing_nm * 0.5
    }

    /// Gaussian sigma for channel spectral profile (nm).
    fn channel_sigma_nm(&self) -> f64 {
        // FWHM = 2 sqrt(2 ln2) σ  ↔  σ = FWHM / (2 sqrt(2 ln2))
        let fwhm = self.channel_bandwidth_nm();
        fwhm / (2.0 * (2.0 * 2.0_f64.ln()).sqrt())
    }

    /// Intensity transmission to channel `ch_idx` at wavelength `lambda_nm`.
    ///
    /// Uses a Gaussian spectral profile centered on the channel wavelength,
    /// attenuated by the insertion loss factor.
    ///
    /// Returns 0 if `ch_idx >= n_channels`.
    pub fn channel_transmission(&self, lambda_nm: f64, ch_idx: usize) -> f64 {
        if ch_idx >= self.n_channels {
            return 0.0;
        }
        let channels = self.channel_wavelengths_nm();
        let lambda_ch = channels[ch_idx];
        let sigma = self.channel_sigma_nm();
        let delta = lambda_nm - lambda_ch;
        let gaussian = (-delta * delta / (2.0 * sigma * sigma)).exp();
        // Apply insertion loss
        let il_factor = 10.0_f64.powf(-self.insertion_loss_db / 10.0);
        gaussian * il_factor
    }

    /// Full transmission matrix: `T[ch_idx][pt_idx]` = transmission of channel
    /// `ch_idx` at wavelength point `pt_idx`.
    ///
    /// Returns a 2D vector of shape `[n_channels][n_pts]`.
    pub fn transmission_matrix(
        &self,
        lambda_start_nm: f64,
        lambda_end_nm: f64,
        n_pts: usize,
    ) -> Result<Vec<Vec<f64>>> {
        if n_pts < 2 {
            return Err(OxiPhotonError::NumericalError(
                "n_pts must be >= 2".to_owned(),
            ));
        }
        if lambda_start_nm >= lambda_end_nm || lambda_start_nm <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "invalid wavelength range [{lambda_start_nm}, {lambda_end_nm}]"
            )));
        }
        let step = (lambda_end_nm - lambda_start_nm) / (n_pts - 1) as f64;
        let matrix = (0..self.n_channels)
            .map(|ch| {
                (0..n_pts)
                    .map(|i| {
                        let lam = lambda_start_nm + i as f64 * step;
                        self.channel_transmission(lam, ch)
                    })
                    .collect::<Vec<f64>>()
            })
            .collect();
        Ok(matrix)
    }

    /// Grating order: m = n_eff · ΔL / λ_center.
    pub fn grating_order(&self) -> f64 {
        let delta_l_nm = self.delta_l_um * 1_000.0;
        self.n_eff_array * delta_l_nm / self.center_wavelength_nm
    }

    /// Free spectral range: FSR ≈ channel_spacing × n_channels.
    ///
    /// Also expressible as FSR = λ² / (m · n_g · ΔL).
    pub fn fsr_nm(&self) -> f64 {
        let m = self.grating_order();
        if m == 0.0 {
            return f64::INFINITY;
        }
        let delta_l_nm = self.delta_l_um * 1_000.0;
        self.center_wavelength_nm * self.center_wavelength_nm / (m * self.n_g_array * delta_l_nm)
    }

    /// NxN routing matrix at `lambda_nm`.
    ///
    /// Entry `[i][j]` is the transmission from input port `i` to output port `j`.
    /// The AWG routes `lambda_nm` from input `i` to output channel `j` determined
    /// by which channel wavelength `lambda_nm` is closest to (offset by `i`).
    ///
    /// Returns `Vec<Vec<f64>>` of shape `[n_channels][n_channels]`.
    pub fn routing_matrix_at(&self, lambda_nm: f64) -> Vec<Vec<f64>> {
        let channels = self.channel_wavelengths_nm();
        let n = self.n_channels;
        let mut matrix = vec![vec![0.0_f64; n]; n];
        for (input_port, row) in matrix.iter_mut().enumerate() {
            for (output_ch, entry) in row.iter_mut().enumerate() {
                // Routing: for input port `p`, the effective wavelength at
                // output channel `j` is shifted by `p * channel_spacing_nm`
                let effective_lambda = lambda_nm - input_port as f64 * self.channel_spacing_nm;
                let delta = effective_lambda - channels[output_ch];
                let sigma = self.channel_sigma_nm();
                let gaussian = (-delta * delta / (2.0 * sigma * sigma)).exp();
                let il_factor = 10.0_f64.powf(-self.insertion_loss_db / 10.0);
                *entry = gaussian * il_factor;
            }
        }
        matrix
    }

    /// Thermal sensitivity dλ/dT (nm/K).
    ///
    /// dλ/dT = λ_center · (dn_eff/dT) / n_g
    pub fn thermal_sensitivity_nm_per_k(&self, dn_dt: f64) -> f64 {
        self.center_wavelength_nm * dn_dt / self.n_g_array
    }

    /// Required heater power for one full FSR of thermal tuning (mW).
    ///
    /// Uses P = ΔT / R_thermal where ΔT = FSR / (dλ/dT).
    ///
    /// # Arguments
    /// * `r_thermal` – thermal resistance (K/mW)
    pub fn tuning_power_mw(&self, r_thermal: f64, dn_dt: f64) -> f64 {
        let sensitivity = self.thermal_sensitivity_nm_per_k(dn_dt);
        if sensitivity == 0.0 || r_thermal == 0.0 {
            return f64::INFINITY;
        }
        let delta_t_needed = self.fsr_nm() / sensitivity;
        delta_t_needed / r_thermal
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ItuGrid
// ─────────────────────────────────────────────────────────────────────────────

/// ITU-T WDM channel grid (G.694.1).
///
/// Defines equally spaced channels in the frequency domain around a reference
/// frequency of 193.1 THz (C-band) or 190.1 THz (L-band).
#[derive(Debug, Clone)]
pub struct ItuGrid {
    /// Reference/center frequency (THz).  ITU C-band = 193.1 THz.
    pub center_freq_thz: f64,
    /// Channel spacing (GHz).  Common: 100, 50, 25 GHz.
    pub channel_spacing_ghz: f64,
    /// Number of channels.
    pub n_channels: usize,
}

impl ItuGrid {
    /// ITU C-band grid at 100 GHz spacing (center 193.1 THz).
    pub fn c_band_100ghz(n_channels: usize) -> Self {
        Self {
            center_freq_thz: 193.1,
            channel_spacing_ghz: 100.0,
            n_channels,
        }
    }

    /// ITU C-band grid at 50 GHz spacing (center 193.1 THz).
    pub fn c_band_50ghz(n_channels: usize) -> Self {
        Self {
            center_freq_thz: 193.1,
            channel_spacing_ghz: 50.0,
            n_channels,
        }
    }

    /// ITU L-band grid at 100 GHz spacing (center 190.1 THz).
    pub fn l_band_100ghz(n_channels: usize) -> Self {
        Self {
            center_freq_thz: 190.1,
            channel_spacing_ghz: 100.0,
            n_channels,
        }
    }

    /// Channel frequencies in THz for all channels.
    ///
    /// Channels are symmetric around the center frequency.
    /// For even `n_channels`, the center falls between channels.
    pub fn channel_frequencies_thz(&self) -> Vec<f64> {
        let spacing_thz = self.channel_spacing_ghz * 1e-3; // GHz → THz
        let half = (self.n_channels as f64 - 1.0) / 2.0;
        (0..self.n_channels)
            .map(|i| self.center_freq_thz + (i as f64 - half) * spacing_thz)
            .collect()
    }

    /// Channel wavelengths in nm for all channels.
    pub fn channel_wavelengths_nm(&self) -> Vec<f64> {
        self.channel_frequencies_thz()
            .iter()
            .map(|&f| Self::frequency_to_wavelength_nm(f))
            .collect()
    }

    /// Index of the nearest channel to `lambda_nm`.
    ///
    /// Returns the index (0-based) of the channel closest in wavelength.
    pub fn nearest_channel(&self, lambda_nm: f64) -> usize {
        let freq_thz = Self::wavelength_to_frequency_thz(lambda_nm);
        let spacing_thz = self.channel_spacing_ghz * 1e-3;
        let half = (self.n_channels as f64 - 1.0) / 2.0;
        let raw_idx = (freq_thz - self.center_freq_thz) / spacing_thz + half;
        let idx = raw_idx.round() as isize;
        idx.clamp(0, self.n_channels as isize - 1) as usize
    }

    /// Convert frequency (THz) to wavelength (nm): λ = c / f.
    ///
    /// Uses `c = 299792.458 nm·THz`.
    pub fn frequency_to_wavelength_nm(freq_thz: f64) -> f64 {
        C_NM_THZ / freq_thz
    }

    /// Convert wavelength (nm) to frequency (THz): f = c / λ.
    pub fn wavelength_to_frequency_thz(lambda_nm: f64) -> f64 {
        C_NM_THZ / lambda_nm
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn standard_awg() -> ArrayedWaveguideGrating {
        // 8-channel AWG, 0.8 nm spacing, 1550 nm center
        ArrayedWaveguideGrating::new(
            8,
            0.8,
            1550.0,
            2.4,
            4.2,
            // ΔL designed so channel spacing matches: Δλ = λ²/(m·n_g·ΔL) = 0.8 nm
            // m·n_g·ΔL = λ²/Δλ → ΔL = λ²/(m·n_g·Δλ)
            // Use m ≈ n_eff·ΔL/λ: iterate or set ΔL directly
            // ΔL such that FSR ≈ n_channels * channel_spacing = 8 * 0.8 = 6.4 nm
            // FSR = λ²/(m·n_g·ΔL) and channel_spacing ≈ FSR/n_ch
            // → m·n_g·ΔL = λ²/(channel_spacing * n_ch) ... but m=n_eff·ΔL/λ
            // → n_eff·ΔL/λ · n_g · ΔL = λ²/(channel_spacing * n_ch)
            // → ΔL² = λ³ / (n_eff · n_g · channel_spacing · n_ch)
            {
                let lambda = 1550.0_f64;
                let n_eff = 2.4_f64;
                let n_g = 4.2_f64;
                let n_ch = 8_f64;
                let delta_lambda = 0.8_f64; // nm
                                            // ΔL in μm from: ΔL² [nm²] = λ³/(n_eff·n_g·Δλ·N)
                let delta_l_sq_nm2 = lambda.powi(3) / (n_eff * n_g * delta_lambda * n_ch);
                delta_l_sq_nm2.sqrt() / 1_000.0 // nm → μm
            },
            100,
            3.0,
            -30.0,
        )
    }

    #[test]
    fn test_awg_channel_wavelengths() {
        let awg = standard_awg();
        let channels = awg.channel_wavelengths_nm();
        assert_eq!(channels.len(), 8, "Should have 8 channels");
        // Spacing between adjacent channels should match design
        for i in 1..channels.len() {
            let spacing = (channels[i] - channels[i - 1]).abs();
            assert!(
                (spacing - awg.channel_spacing_nm).abs() / awg.channel_spacing_nm < 1e-10,
                "Channel spacing mismatch at ch {i}: {spacing:.6} vs {:.6}",
                awg.channel_spacing_nm
            );
        }
        // Center channel should be near center_wavelength_nm
        let center_ch = channels[channels.len() / 2];
        assert!(
            (center_ch - awg.center_wavelength_nm).abs() < awg.channel_spacing_nm,
            "Center channel {center_ch:.4} not near design center {:.4}",
            awg.center_wavelength_nm
        );
    }

    #[test]
    fn test_awg_fsr() {
        let awg = standard_awg();
        let fsr = awg.fsr_nm();
        let expected_approx = awg.n_channels as f64 * awg.channel_spacing_nm;
        // FSR should be approximately N × channel_spacing
        let rel_err = (fsr - expected_approx).abs() / expected_approx;
        assert!(
            rel_err < 0.15,
            "FSR {fsr:.4} nm should be close to {expected_approx:.4} nm (rel err {rel_err:.3})"
        );
    }

    #[test]
    fn test_awg_routing_matrix() {
        let awg = standard_awg();
        let channels = awg.channel_wavelengths_nm();
        // At each channel wavelength, the diagonal entry (input 0 → same output)
        // should have the highest transmission among outputs
        for (ch_idx, &lambda_ch) in channels.iter().enumerate() {
            let t_target = awg.channel_transmission(lambda_ch, ch_idx);
            // Compare against adjacent channels
            if ch_idx > 0 {
                let t_adj = awg.channel_transmission(lambda_ch, ch_idx - 1);
                assert!(
                    t_target > t_adj,
                    "Channel {ch_idx} at λ={lambda_ch:.2}: target T={t_target:.4} should be > adjacent T={t_adj:.4}"
                );
            }
            if ch_idx + 1 < channels.len() {
                let t_adj = awg.channel_transmission(lambda_ch, ch_idx + 1);
                assert!(
                    t_target > t_adj,
                    "Channel {ch_idx} at λ={lambda_ch:.2}: target T={t_target:.4} should be > adjacent T={t_adj:.4}"
                );
            }
        }
    }

    #[test]
    fn test_itu_grid_100ghz() {
        let grid = ItuGrid::c_band_100ghz(40);
        let wls = grid.channel_wavelengths_nm();
        assert_eq!(wls.len(), 40);
        // 100 GHz spacing at 1550 nm:
        // Δλ = λ²/c · Δf = (1550)² nm² / 299792.458 nm·THz × 0.1 THz
        let expected_spacing_nm = 1550.0_f64.powi(2) / 299_792.458 * 0.1;
        // Check spacing between adjacent channels
        for i in 1..wls.len() {
            let spacing = (wls[i - 1] - wls[i]).abs(); // shorter λ has higher freq, so wls is sorted desc
            assert!(
                (spacing - expected_spacing_nm).abs() / expected_spacing_nm < 0.05,
                "100 GHz channel spacing at 1550 nm: got {spacing:.4} nm, expected ~{expected_spacing_nm:.4} nm"
            );
        }
    }

    #[test]
    fn test_wavelength_frequency_conversion() {
        // f = c/λ, roundtrip
        let lambda = 1550.0_f64;
        let freq = ItuGrid::wavelength_to_frequency_thz(lambda);
        let lambda_back = ItuGrid::frequency_to_wavelength_nm(freq);
        assert!(
            (lambda_back - lambda).abs() < 1e-9,
            "Roundtrip conversion: {lambda} nm → {freq:.6} THz → {lambda_back:.10} nm"
        );
        // 1550 nm should give ~193.41 THz
        assert!(
            (freq - 193.41).abs() < 0.1,
            "1550 nm should be ~193.41 THz, got {freq:.4} THz"
        );
    }

    #[test]
    fn test_itu_nearest_channel() {
        let grid = ItuGrid::c_band_100ghz(40);
        let wls = grid.channel_wavelengths_nm();
        // Each channel wavelength should map back to itself
        for (i, &wl) in wls.iter().enumerate() {
            let nearest = grid.nearest_channel(wl);
            assert_eq!(
                nearest, i,
                "Channel {i} at λ={wl:.4} nm should map to itself, got {nearest}"
            );
        }
    }

    #[test]
    fn test_awg_grating_order_positive() {
        let awg = standard_awg();
        let m = awg.grating_order();
        assert!(m > 0.0, "Grating order should be positive, got {m}");
        // Typical silicon AWG grating orders are in the range 10-100
        assert!(
            m > 1.0 && m < 10_000.0,
            "Grating order out of typical range: {m}"
        );
    }

    #[test]
    fn test_awg_insertion_loss() {
        let awg = standard_awg();
        // On-channel transmission should be reduced by insertion loss
        let channels = awg.channel_wavelengths_nm();
        let t = awg.channel_transmission(channels[0], 0);
        let il_factor = 10.0_f64.powf(-awg.insertion_loss_db / 10.0);
        // Peak Gaussian = 1 × il_factor
        assert!(
            (t - il_factor).abs() / il_factor < 1e-10,
            "On-peak transmission should equal il_factor={il_factor:.4}, got {t:.4}"
        );
    }

    #[test]
    fn test_awg_transmission_matrix_shape() {
        let awg = standard_awg();
        let matrix = awg
            .transmission_matrix(1546.0, 1554.0, 100)
            .expect("matrix");
        assert_eq!(
            matrix.len(),
            awg.n_channels,
            "Matrix should have n_channels rows"
        );
        for row in &matrix {
            assert_eq!(row.len(), 100, "Each row should have 100 wavelength points");
        }
    }

    #[test]
    fn test_itu_grid_50ghz_has_double_channels() {
        let grid_100 = ItuGrid::c_band_100ghz(20);
        let grid_50 = ItuGrid::c_band_50ghz(40);
        // 50 GHz grid should have half the spacing
        let freqs_100 = grid_100.channel_frequencies_thz();
        let freqs_50 = grid_50.channel_frequencies_thz();
        let spacing_100 = (freqs_100[1] - freqs_100[0]).abs();
        let spacing_50 = (freqs_50[1] - freqs_50[0]).abs();
        assert!(
            (spacing_100 / spacing_50 - 2.0).abs() < 1e-10,
            "100 GHz spacing should be 2× 50 GHz spacing: {spacing_100} vs {spacing_50}"
        );
    }

    #[test]
    fn test_thermal_sensitivity() {
        let awg = standard_awg();
        // Silicon dn/dT ≈ 1.86e-4 K⁻¹ → positive sensitivity
        let sens = awg.thermal_sensitivity_nm_per_k(1.86e-4);
        assert!(sens > 0.0, "Thermal sensitivity should be positive: {sens}");
        // Typical value: ~0.05–0.1 nm/K for Si AWG
        assert!(
            sens > 0.01 && sens < 1.0,
            "Thermal sensitivity out of range: {sens} nm/K"
        );
    }
}
