/// High-level coating designs built on top of the transfer matrix method.
///
/// Provides factory methods for common thin film optical coatings:
/// - Anti-reflection (AR) coatings: single-layer, double-layer, broadband
/// - High-reflectance (HR) quarter-wave stacks
/// - Bandpass Fabry-Perot filters
/// - Longpass / shortpass edge filters
/// - Fabry-Perot etalon (analytical model)
///
/// Physical references:
/// - Macleod, "Thin-Film Optical Filters", 4th ed.
/// - Heavens, "Optical Properties of Thin Solid Films"
/// - Born & Wolf, "Principles of Optics"
use std::f64::consts::PI;

use num_complex::Complex64;

use crate::thin_film::transfer_matrix::{Layer, MultilayerStack, Polarization};

// ─── Anti-reflection coatings ─────────────────────────────────────────────────

/// Factory for anti-reflection coating designs.
///
/// Anti-reflection coatings minimise Fresnel reflection at an optical surface
/// by destructive interference of reflected partial waves.
pub struct AntiReflectionCoating;

impl AntiReflectionCoating {
    /// Single-layer AR coating: n_AR = √(n_inc · n_sub), d = λ/(4 n_AR).
    ///
    /// When n_AR² = n_inc · n_sub, perfect cancellation occurs at the design
    /// wavelength.  The `n_high` parameter is the *actual* coating index used
    /// (not necessarily the optimal value).
    pub fn single_layer(n_substrate: f64, n_high: f64, lambda_nm: f64) -> MultilayerStack {
        let mut stack = MultilayerStack::new(1.0_f64, n_substrate);
        stack.add_layer(Layer::quarter_wave(n_high, lambda_nm, "AR_single"));
        stack
    }

    /// Double-layer V-coat AR coating (very low R at design wavelength).
    ///
    /// Structure: Air | n_high (λ/4) | n_low (λ/4) | Substrate
    ///
    /// The two-layer design allows independent optimisation of both index
    /// and thickness to achieve near-zero reflectance over a narrow band.
    pub fn double_layer(
        n_substrate: f64,
        n_high: f64,
        n_low: f64,
        lambda_nm: f64,
    ) -> MultilayerStack {
        let mut stack = MultilayerStack::new(1.0_f64, n_substrate);
        stack.add_layer(Layer::quarter_wave(n_high, lambda_nm, "AR_high"));
        stack.add_layer(Layer::quarter_wave(n_low, lambda_nm, "AR_low"));
        stack
    }

    /// Broadband AR coating using a multi-period graded-index approximation.
    ///
    /// Uses a Rugate-style triplet structure (MgF2 / ZrO2 / MgF2) repeated
    /// `n_periods` times.  The period thickness is scaled to λ_center/4 per
    /// sub-layer, yielding a smoother spectral response than a simple V-coat.
    ///
    /// Approximate material indices:
    /// - MgF2 (low): 1.38
    /// - ZrO2 (high): 2.10
    pub fn broadband(n_substrate: f64, lambda_center_nm: f64, n_periods: usize) -> MultilayerStack {
        let n_mgf2 = 1.38_f64;
        let n_zro2 = 2.10_f64;
        let mut stack = MultilayerStack::new(1.0_f64, n_substrate);
        for _ in 0..n_periods {
            stack.add_layer(Layer::quarter_wave(n_mgf2, lambda_center_nm, "MgF2"));
            stack.add_layer(Layer::quarter_wave(n_zro2, lambda_center_nm, "ZrO2"));
            stack.add_layer(Layer::quarter_wave(n_mgf2, lambda_center_nm, "MgF2"));
        }
        stack
    }

    /// Single-layer reflectance at the design wavelength (analytic).
    ///
    /// R = ((n_inc · n_sub − n_ar²) / (n_inc · n_sub + n_ar²))²
    /// (normal incidence, quarter-wave layer)
    pub fn single_layer_r_at_design(n_inc: f64, n_ar: f64, n_sub: f64) -> f64 {
        let num = n_inc * n_sub - n_ar * n_ar;
        let den = n_inc * n_sub + n_ar * n_ar;
        (num / den) * (num / den)
    }

    /// Optimal coating index for zero reflectance: n_AR = √(n_inc · n_sub).
    pub fn optimal_n_ar(n_inc: f64, n_sub: f64) -> f64 {
        (n_inc * n_sub).sqrt()
    }
}

// ─── High-reflectance mirrors ─────────────────────────────────────────────────

/// Factory for high-reflectance (HR) dielectric mirror coatings.
///
/// HR mirrors use constructive interference of partial reflections from a
/// quarter-wave stack of alternating high- and low-index layers.
pub struct HighReflectanceMirror;

impl HighReflectanceMirror {
    /// Quarter-wave stack (H L)^N H at normal incidence.
    ///
    /// Structure: Air | H (λ/4) | L (λ/4) | … | H (λ/4) | Substrate
    ///
    /// The first and last layers are high-index to maximise interference.
    pub fn quarter_wave_stack(
        n_substrate: f64,
        n_high: f64,
        n_low: f64,
        n_periods: usize,
        lambda_nm: f64,
    ) -> MultilayerStack {
        let mut stack = MultilayerStack::new(1.0_f64, n_substrate);
        // Start with H layer, then (L H)^(N-1) pairs, end with H
        stack.add_layer(Layer::quarter_wave(n_high, lambda_nm, "H"));
        for _ in 0..n_periods.saturating_sub(1) {
            stack.add_layer(Layer::quarter_wave(n_low, lambda_nm, "L"));
            stack.add_layer(Layer::quarter_wave(n_high, lambda_nm, "H"));
        }
        stack
    }

    /// Analytic peak reflectance for N full H-L periods.
    ///
    /// For a (H L)^N H stack at normal incidence:
    /// R = ((1 − (n_H/n_L)^(2N) · n_sub/n_inc) / (1 + (n_H/n_L)^(2N) · n_sub/n_inc))²
    pub fn peak_reflectance(n_inc: f64, n_sub: f64, n_h: f64, n_l: f64, n_periods: usize) -> f64 {
        let ratio = (n_h / n_l).powi(2 * n_periods as i32) * n_sub / n_inc;
        let r = (1.0 - ratio) / (1.0 + ratio);
        r * r
    }

    /// Stop-band fractional width Δλ/λ₀.
    ///
    /// Δλ/λ₀ = (4/π) · arcsin((n_H − n_L)/(n_H + n_L))
    pub fn stopband_width_fraction(n_h: f64, n_l: f64) -> f64 {
        (4.0 / PI) * ((n_h - n_l) / (n_h + n_l)).asin()
    }

    /// Minimum number of periods to achieve a target reflectance R_target.
    ///
    /// Inverts the analytic formula; returns at least 1.
    pub fn periods_for_reflectance(
        target_r: f64,
        n_inc: f64,
        n_sub: f64,
        n_h: f64,
        n_l: f64,
    ) -> usize {
        // Solve R(N) >= target_r by incrementing N
        for n in 1..=200 {
            if Self::peak_reflectance(n_inc, n_sub, n_h, n_l, n) >= target_r {
                return n;
            }
        }
        200 // saturate at 200 if target is unreachable
    }

    /// Normal-incidence reflectance of a bare metal surface.
    ///
    /// R = |r|² where r = (n_inc − ñ_metal) / (n_inc + ñ_metal)
    pub fn metal_mirror(n_metal: Complex64, n_inc: f64) -> f64 {
        let ni = Complex64::new(n_inc, 0.0);
        let r = (ni - n_metal) / (ni + n_metal);
        r.norm_sqr()
    }
}

// ─── Bandpass filter ──────────────────────────────────────────────────────────

/// Factory for thin-film bandpass (narrowband transmission) filters.
///
/// The simplest design is a single-cavity Fabry-Perot: a spacer layer
/// sandwiched between two HR quarter-wave stacks.
pub struct BandpassFilter;

impl BandpassFilter {
    /// Fabry-Perot bandpass filter with dielectric HR mirror stacks.
    ///
    /// Structure: Air | (H L)^N_pairs | Spacer (half-wave) | (L H)^N_pairs | Sub
    ///
    /// The spacer is a half-wave optical thickness layer at the centre wavelength.
    pub fn fabry_perot(
        n_inc: f64,
        n_sub: f64,
        n_h: f64,
        n_l: f64,
        n_spacer: f64,
        n_pairs: usize,
        center_lambda_nm: f64,
    ) -> MultilayerStack {
        let mut stack = MultilayerStack::new(n_inc, n_sub);

        // Front HR stack: (H L)^N_pairs
        for _ in 0..n_pairs {
            stack.add_layer(Layer::quarter_wave(n_h, center_lambda_nm, "H_front"));
            stack.add_layer(Layer::quarter_wave(n_l, center_lambda_nm, "L_front"));
        }

        // Half-wave spacer (cavity)
        stack.add_layer(Layer::half_wave(n_spacer, center_lambda_nm, "Spacer"));

        // Back HR stack: (L H)^N_pairs (reversed for symmetry)
        for _ in 0..n_pairs {
            stack.add_layer(Layer::quarter_wave(n_l, center_lambda_nm, "L_back"));
            stack.add_layer(Layer::quarter_wave(n_h, center_lambda_nm, "H_back"));
        }

        stack
    }

    /// Bandwidth (FWHM) of a Fabry-Perot cavity filter (nm).
    ///
    /// δλ = λ² / (2 · n · d · F)  where F is the finesse.
    pub fn bandwidth_nm(
        center_lambda_nm: f64,
        spacer_n: f64,
        spacer_d_nm: f64,
        reflectance: f64,
    ) -> f64 {
        let f = Self::finesse(reflectance);
        center_lambda_nm * center_lambda_nm / (2.0 * spacer_n * spacer_d_nm * f)
    }

    /// Finesse of a Fabry-Perot cavity: F = π √R / (1 − R).
    pub fn finesse(reflectance: f64) -> f64 {
        PI * reflectance.sqrt() / (1.0 - reflectance)
    }

    /// Free spectral range (FSR) in nm: FSR = λ² / (2 · n · d).
    pub fn fsr_nm(center_lambda_nm: f64, spacer_n: f64, spacer_d_nm: f64) -> f64 {
        center_lambda_nm * center_lambda_nm / (2.0 * spacer_n * spacer_d_nm)
    }
}

// ─── Edge filters ─────────────────────────────────────────────────────────────

/// Factory for longpass and shortpass edge filters.
///
/// Edge filters use the stop-band of a quarter-wave stack to create a sharp
/// transmission edge.  By slightly detuning the QW design wavelength from the
/// desired edge wavelength, we can place the stop-band edge at the target.
pub struct EdgeFilter;

impl EdgeFilter {
    /// Longpass edge filter: transmits λ > λ_edge, blocks λ < λ_edge.
    ///
    /// The stop-band of a QW stack centred at λ_design ≈ λ_edge is used.
    /// A slight detuning shifts the short-wavelength stop-band edge to λ_edge.
    pub fn longpass(
        n_inc: f64,
        n_sub: f64,
        n_h: f64,
        n_l: f64,
        edge_wavelength_nm: f64,
        n_pairs: usize,
    ) -> MultilayerStack {
        // Design wavelength slightly shorter than edge to place the rising
        // edge of the stop-band at the edge wavelength.
        let stopband_frac = HighReflectanceMirror::stopband_width_fraction(n_h, n_l);
        // Shift design λ so that the short-wavelength edge lands at λ_edge
        // λ_design = λ_edge / (1 - stopband_frac/2)
        let lambda_design = edge_wavelength_nm / (1.0 - stopband_frac / 2.0);

        let mut stack = MultilayerStack::new(n_inc, n_sub);
        for _ in 0..n_pairs {
            stack.add_layer(Layer::quarter_wave(n_h, lambda_design, "H"));
            stack.add_layer(Layer::quarter_wave(n_l, lambda_design, "L"));
        }
        stack
    }

    /// Shortpass edge filter: transmits λ < λ_edge, blocks λ > λ_edge.
    ///
    /// Design wavelength is shifted to the long-wavelength side.
    pub fn shortpass(
        n_inc: f64,
        n_sub: f64,
        n_h: f64,
        n_l: f64,
        edge_wavelength_nm: f64,
        n_pairs: usize,
    ) -> MultilayerStack {
        let stopband_frac = HighReflectanceMirror::stopband_width_fraction(n_h, n_l);
        // Shift design λ so that the long-wavelength edge lands at λ_edge
        let lambda_design = edge_wavelength_nm / (1.0 + stopband_frac / 2.0);

        let mut stack = MultilayerStack::new(n_inc, n_sub);
        for _ in 0..n_pairs {
            stack.add_layer(Layer::quarter_wave(n_h, lambda_design, "H"));
            stack.add_layer(Layer::quarter_wave(n_l, lambda_design, "L"));
        }
        stack
    }

    /// Estimate the edge slope as the wavelength range for 10% → 90% transmission.
    ///
    /// Performs a coarse linear search around `edge_nm` over ±30% bandwidth.
    pub fn edge_slope_nm(stack: &MultilayerStack, edge_nm: f64) -> f64 {
        let n_pts = 500;
        let l_min = edge_nm * 0.7;
        let l_max = edge_nm * 1.3;

        let spec = stack.spectrum(l_min, l_max, n_pts, 0.0, Polarization::TE);

        let mut lambda_10 = l_min;
        let mut lambda_90 = l_max;
        let mut found_10 = false;

        for &(lambda, _r, t, _a) in &spec {
            if !found_10 && t >= 0.1 {
                lambda_10 = lambda;
                found_10 = true;
            }
            if t >= 0.9 {
                lambda_90 = lambda;
                break;
            }
        }

        (lambda_90 - lambda_10).abs()
    }
}

// ─── Fabry-Perot etalon ───────────────────────────────────────────────────────

/// Analytical Fabry-Perot etalon (solid or air-gap).
///
/// Models a parallel-plate resonator using the Airy function.  Both mirrors
/// are assumed identical with reflectance R.
///
/// Reference: Born & Wolf §7.6
#[derive(Debug, Clone)]
pub struct FabryPerotEtalon {
    /// Refractive index of the cavity medium.
    pub n_medium: f64,
    /// Physical thickness of the etalon in millimetres.
    pub thickness_mm: f64,
    /// Mirror reflectance (both mirrors equal, 0 < R < 1).
    pub reflectance: f64,
    /// Nominal centre wavelength used for FSR / linewidth estimates (nm).
    pub center_wavelength_nm: f64,
}

impl FabryPerotEtalon {
    /// Create a new Fabry-Perot etalon.
    pub fn new(
        n_medium: f64,
        thickness_mm: f64,
        reflectance: f64,
        center_wavelength_nm: f64,
    ) -> Self {
        Self {
            n_medium,
            thickness_mm,
            reflectance,
            center_wavelength_nm,
        }
    }

    /// Transmission at wavelength λ (nm) via the Airy function.
    ///
    /// T(φ) = 1 / (1 + F sin²(φ/2))
    /// where φ = 4π n d / λ  (round-trip phase),  F = 4R/(1−R)².
    pub fn transmission(&self, lambda_nm: f64) -> f64 {
        let d_nm = self.thickness_mm * 1.0e6; // mm → nm
        let phase = 4.0 * PI * self.n_medium * d_nm / lambda_nm;
        let f_coeff = 4.0 * self.reflectance / (1.0 - self.reflectance).powi(2);
        1.0 / (1.0 + f_coeff * (phase / 2.0).sin().powi(2))
    }

    /// Finesse: F = π √R / (1 − R).
    pub fn finesse(&self) -> f64 {
        PI * self.reflectance.sqrt() / (1.0 - self.reflectance)
    }

    /// Free spectral range (FSR) in GHz: Δν = c / (2 n d).
    pub fn fsr_ghz(&self) -> f64 {
        let c_mm_per_s = 2.99792458e11_f64; // mm/s
        c_mm_per_s / (2.0 * self.n_medium * self.thickness_mm * 1.0e9)
    }

    /// Free spectral range in nm (at the centre wavelength).
    ///
    /// FSR_nm ≈ λ² / (2 · n · d)
    pub fn fsr_nm(&self) -> f64 {
        let d_nm = self.thickness_mm * 1.0e6;
        self.center_wavelength_nm * self.center_wavelength_nm / (2.0 * self.n_medium * d_nm)
    }

    /// Linewidth (FWHM) in GHz: δν = FSR / F.
    pub fn linewidth_ghz(&self) -> f64 {
        self.fsr_ghz() / self.finesse()
    }

    /// Linewidth (FWHM) in nm: δλ = FSR_nm / F.
    pub fn linewidth_nm(&self) -> f64 {
        self.fsr_nm() / self.finesse()
    }

    /// Transmission spectrum as (lambda_nm, T) pairs.
    pub fn spectrum(
        &self,
        lambda_min_nm: f64,
        lambda_max_nm: f64,
        n_pts: usize,
    ) -> Vec<(f64, f64)> {
        if n_pts == 0 {
            return Vec::new();
        }
        (0..n_pts)
            .map(|i| {
                let t = i as f64 / (n_pts - 1).max(1) as f64;
                let lambda = lambda_min_nm + t * (lambda_max_nm - lambda_min_nm);
                (lambda, self.transmission(lambda))
            })
            .collect()
    }

    /// Resonance wavelengths within [lambda_min_nm, lambda_max_nm].
    ///
    /// Resonances occur where the round-trip phase is a multiple of 2π:
    /// λ_m = 2 n d / m
    pub fn resonances_nm(&self, lambda_min_nm: f64, lambda_max_nm: f64) -> Vec<f64> {
        let d_nm = self.thickness_mm * 1.0e6;
        let two_nd = 2.0 * self.n_medium * d_nm;

        // m_min: smallest integer such that λ = 2nd/m ≤ lambda_max
        let m_min = (two_nd / lambda_max_nm).ceil() as usize;
        let m_max = (two_nd / lambda_min_nm).floor() as usize;

        (m_min..=m_max)
            .filter_map(|m| {
                if m == 0 {
                    return None;
                }
                let lam = two_nd / m as f64;
                if lam >= lambda_min_nm && lam <= lambda_max_nm {
                    Some(lam)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Thermal sensitivity of resonance wavelength (nm/K).
    ///
    /// dλ/dT = λ · (α_thermal + (1/n) · dn/dT)
    ///
    /// where α_thermal is the coefficient of thermal expansion (1/K) and
    /// dn/dT is the thermo-optic coefficient (1/K).
    pub fn thermal_sensitivity_nm_per_k(&self, dn_dt: f64, alpha_thermal: f64) -> f64 {
        self.center_wavelength_nm * (alpha_thermal + dn_dt / self.n_medium)
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    const N_AIR: f64 = 1.0;
    const N_GLASS: f64 = 1.52;
    const N_TIO2: f64 = 2.35;
    const N_SIO2: f64 = 1.46;
    const LAMBDA: f64 = 550.0; // nm

    fn bare_glass_r(n_sub: f64) -> f64 {
        let r = (1.0 - n_sub) / (1.0 + n_sub);
        r * r
    }

    // ── test 1 ────────────────────────────────────────────────────────────────
    #[test]
    fn test_ar_single_layer_r_reduced() {
        let n_ar = AntiReflectionCoating::optimal_n_ar(N_AIR, N_GLASS);
        let stack = AntiReflectionCoating::single_layer(N_GLASS, n_ar, LAMBDA);

        let r_coated = stack.reflectance(LAMBDA, 0.0, Polarization::TE);
        let r_bare = bare_glass_r(N_GLASS);

        assert!(
            r_coated < r_bare,
            "AR single-layer should reduce R: r_coated={r_coated:.6} r_bare={r_bare:.6}"
        );
    }

    // ── test 2 ────────────────────────────────────────────────────────────────
    #[test]
    fn test_ar_optimal_n() {
        let n_opt = AntiReflectionCoating::optimal_n_ar(N_AIR, N_GLASS);
        let expected = (N_AIR * N_GLASS).sqrt();
        assert_abs_diff_eq!(n_opt, expected, epsilon = 1e-12);
    }

    // ── test 3 ────────────────────────────────────────────────────────────────
    #[test]
    fn test_hr_mirror_high_r() {
        let stack = HighReflectanceMirror::quarter_wave_stack(N_GLASS, N_TIO2, N_SIO2, 10, LAMBDA);
        let r = stack.reflectance(LAMBDA, 0.0, Polarization::TE);
        assert!(
            r > 0.99,
            "HR 10-pair mirror should have R > 0.99, got R={r:.6}"
        );
    }

    // ── test 4 ────────────────────────────────────────────────────────────────
    #[test]
    fn test_hr_stopband_width() {
        let w1 = HighReflectanceMirror::stopband_width_fraction(N_TIO2, N_SIO2);
        let w2 = HighReflectanceMirror::stopband_width_fraction(2.0, 1.5);
        assert!(w1 > 0.0, "Stopband width must be positive");
        // Higher contrast → wider stop-band
        assert!(
            w1 > w2,
            "Higher index contrast should give wider stop-band: {w1:.4} vs {w2:.4}"
        );
    }

    // ── test 5 ────────────────────────────────────────────────────────────────
    #[test]
    fn test_periods_for_target_r() {
        let n =
            HighReflectanceMirror::periods_for_reflectance(0.999, N_AIR, N_GLASS, N_TIO2, N_SIO2);
        assert!(n >= 1, "Must return at least 1 period");
        // Verify the result actually achieves the target
        let r = HighReflectanceMirror::peak_reflectance(N_AIR, N_GLASS, N_TIO2, N_SIO2, n);
        assert!(
            r >= 0.999,
            "Predicted R={r:.6} should be ≥ 0.999 for N={n} periods"
        );
    }

    // ── test 6 ────────────────────────────────────────────────────────────────
    #[test]
    fn test_fabry_perot_transmission_at_resonance() {
        // Lossless etalon: T=1 at resonance
        let etalon = FabryPerotEtalon::new(1.5, 1.0, 0.95, 500.0);
        // Find a resonance wavelength
        let resonances = etalon.resonances_nm(400.0, 600.0);
        assert!(
            !resonances.is_empty(),
            "Should have resonances in 400-600 nm"
        );

        let lam_res = resonances[0];
        let t = etalon.transmission(lam_res);
        assert!(
            (t - 1.0).abs() < 1e-6,
            "Transmission at resonance should be ≈1, got T={t:.8} at λ={lam_res:.4}"
        );
    }

    // ── test 7 ────────────────────────────────────────────────────────────────
    #[test]
    fn test_fp_fsr_formula() {
        let n = 1.5_f64;
        let d_mm = 0.5_f64;
        let lambda_nm = 600.0_f64;

        let etalon = FabryPerotEtalon::new(n, d_mm, 0.9, lambda_nm);
        let fsr = etalon.fsr_nm();

        // Analytic: FSR = λ² / (2 n d)
        let d_nm = d_mm * 1.0e6;
        let expected = lambda_nm * lambda_nm / (2.0 * n * d_nm);

        assert_abs_diff_eq!(fsr, expected, epsilon = 1e-10);
    }

    // ── test 8 ────────────────────────────────────────────────────────────────
    #[test]
    fn test_fp_finesse() {
        let r = 0.9_f64;
        let etalon = FabryPerotEtalon::new(1.0, 1.0, r, 550.0);
        let expected = PI * r.sqrt() / (1.0 - r);
        assert_abs_diff_eq!(etalon.finesse(), expected, epsilon = 1e-12);
    }

    // ── test 9 ────────────────────────────────────────────────────────────────
    #[test]
    fn test_etalon_linewidth_lt_fsr() {
        let etalon = FabryPerotEtalon::new(1.5, 1.0, 0.9, 550.0);
        let lw = etalon.linewidth_nm();
        let fsr = etalon.fsr_nm();
        assert!(lw < fsr, "Linewidth {lw:.6} must be less than FSR {fsr:.6}");
    }

    // ── test 10 ───────────────────────────────────────────────────────────────
    #[test]
    fn test_bandpass_peak_transmission() {
        // A well-designed Fabry-Perot bandpass filter should transmit near 1
        // at the centre wavelength (lossless materials).
        let stack = BandpassFilter::fabry_perot(N_AIR, N_GLASS, N_TIO2, N_SIO2, N_SIO2, 4, LAMBDA);
        let t = stack.transmittance(LAMBDA, 0.0, Polarization::TE);
        assert!(
            t > 0.5,
            "Bandpass filter should transmit significantly at centre wavelength: T={t:.4}"
        );
    }
}
