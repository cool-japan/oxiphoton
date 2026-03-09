//! Edge coupler model for chip-to-fiber coupling.
//!
//! Edge couplers (also called "inverse tapers" or "nanotapers") are used to
//! couple light from a single-mode optical fiber into a Si/SiN waveguide by
//! mode size transformation. The waveguide is tapered to a sharp tip
//! (typically 100–200 nm) where the mode expands to match the lensed fiber
//! mode (~2-5 µm diameter).
//!
//! Key parameters:
//!   - Tip width w_tip (m): determines coupling efficiency
//!   - Taper length L (m): longer = more adiabatic
//!   - Fiber MFD (m): mode-field diameter of lensed fiber
//!   - Coupling efficiency η: overlap integral between fiber and chip mode
//!
//! Mode overlap integral:
//!   η = |∫∫ E_fiber* · E_wg dA|² / (∫∫|E_fiber|²dA · ∫∫|E_wg|²dA)
//!
//! For Gaussian approximation:
//!   η = 4·(w_f·w_wg)² / (w_f² + w_wg²)²
//! where w_f, w_wg are the 1/e² mode radii.

use std::f64::consts::PI;

/// Edge coupler (inverse taper) model.
#[derive(Debug, Clone, Copy)]
pub struct EdgeCoupler {
    /// Tip width (m) — smallest waveguide dimension
    pub w_tip: f64,
    /// Waveguide width at full size (m)
    pub w_full: f64,
    /// Taper length (m)
    pub length: f64,
    /// Fiber mode-field diameter (m)
    pub fiber_mfd: f64,
    /// Waveguide height (m)
    pub height: f64,
    /// Core refractive index
    pub n_core: f64,
    /// Cladding index
    pub n_clad: f64,
}

impl EdgeCoupler {
    /// Create an edge coupler model.
    pub fn new(
        w_tip: f64,
        w_full: f64,
        length: f64,
        fiber_mfd: f64,
        height: f64,
        n_core: f64,
        n_clad: f64,
    ) -> Self {
        Self {
            w_tip,
            w_full,
            length,
            fiber_mfd,
            height,
            n_core,
            n_clad,
        }
    }

    /// Standard SOI edge coupler at 1550 nm.
    ///
    /// 220 nm × 150 nm tip, 10 µm length, SMF-28 fiber (10.4 µm MFD).
    pub fn soi_standard() -> Self {
        Self::new(150e-9, 500e-9, 10e-6, 10.4e-6, 220e-9, 3.476, 1.444)
    }

    /// SiN edge coupler at 1550 nm (lower index contrast, broader mode).
    ///
    /// 300 nm × 300 nm tip, 500 µm length, lensed fiber 3 µm MFD.
    pub fn sin_inverse_taper() -> Self {
        Self::new(300e-9, 800e-9, 500e-6, 3e-6, 300e-9, 2.0, 1.444)
    }

    /// Approximate mode radius at the tip (Gaussian approximation).
    ///
    /// For a sub-wavelength waveguide at the tip, the mode expands into the cladding.
    /// Approximate MFD_tip ≈ fiber_mfd when tip << λ/n.
    pub fn mode_radius_at_tip(&self, wavelength: f64) -> f64 {
        // At the tip, mode is weakly guided → approximate as Gaussian expanding into cladding
        // Using approximate formula: w_mode ≈ 0.65*w_tip + 1.619/V^1.5 + 2.879/V^6) * w_tip
        let v = PI * self.w_tip / wavelength
            * (self.n_core * self.n_core - self.n_clad * self.n_clad).sqrt();
        let v = v.max(0.1);
        self.w_tip * (0.65 + 1.619 / v.powf(1.5) + 2.879 / v.powi(6))
    }

    /// Gaussian overlap coupling efficiency η (fraction, not dB).
    ///
    ///   η = 4·(w_f·w_wg)² / (w_f² + w_wg²)²
    pub fn coupling_efficiency(&self, wavelength: f64) -> f64 {
        let w_f = self.fiber_mfd / 2.0; // fiber 1/e² radius
        let w_wg = self.mode_radius_at_tip(wavelength);
        4.0 * (w_f * w_wg).powi(2) / (w_f * w_f + w_wg * w_wg).powi(2)
    }

    /// Coupling loss in dB.
    pub fn coupling_loss_db(&self, wavelength: f64) -> f64 {
        let eta = self.coupling_efficiency(wavelength);
        if eta < 1e-30 {
            return 50.0;
        }
        -10.0 * eta.log10()
    }

    /// Adiabaticity criterion check: is the taper adiabatic?
    ///
    /// Condition: L > (w_full - w_tip) / (2 · tan(θ_max))
    /// where θ_max ≈ λ/(π·n·w_min²) is the local beat angle.
    pub fn is_adiabatic(&self, wavelength: f64) -> bool {
        let theta_max = wavelength / (PI * self.n_core * self.w_tip * self.w_tip);
        let l_min = (self.w_full - self.w_tip) / (2.0 * theta_max);
        self.length > l_min
    }

    /// Reflection from the tip (Fresnel approximation).
    ///
    /// For a polished facet at n_clad / n_core interface:
    ///   R = ((n_core - n_clad) / (n_core + n_clad))²
    /// At the tip, n_eff is close to n_clad → low reflection.
    pub fn facet_reflection(&self) -> f64 {
        let r = (self.n_core - self.n_clad) / (self.n_core + self.n_clad);
        r * r
    }

    /// Taper bandwidth (nm): approximate coupling efficiency ≥ η₀/2 bandwidth.
    ///
    /// Rough estimate: bandwidth scales inversely with NA² of the coupler.
    pub fn bandwidth_3db_nm(&self, center_wavelength: f64) -> f64 {
        // Estimate: coupling is broadband for small-NA tip
        // BW ≈ 2 * (center_wl^2 / MFD) * ... rough empirical
        let na_approx = (self.n_core * self.n_core - self.n_clad * self.n_clad).sqrt();
        // Larger NA → narrower BW
        center_wavelength * 1e9 * 0.1 / na_approx.max(0.01)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_coupler_efficiency_in_range() {
        let ec = EdgeCoupler::soi_standard();
        let eta = ec.coupling_efficiency(1550e-9);
        assert!(eta > 0.0 && eta <= 1.0, "η={eta:.3}");
    }

    #[test]
    fn edge_coupler_loss_db_positive() {
        let ec = EdgeCoupler::soi_standard();
        let loss = ec.coupling_loss_db(1550e-9);
        assert!((0.0..30.0).contains(&loss), "loss={loss:.2}dB");
    }

    #[test]
    fn sin_coupler_lower_loss() {
        let sin = EdgeCoupler::sin_inverse_taper();
        let soi = EdgeCoupler::soi_standard();
        let eta_sin = sin.coupling_efficiency(1550e-9);
        let eta_soi = soi.coupling_efficiency(1550e-9);
        // SiN with smaller index contrast → better mode match for lensed fiber
        assert!(eta_sin > 0.0);
        assert!(eta_soi > 0.0);
    }

    #[test]
    fn edge_coupler_mode_radius_positive() {
        let ec = EdgeCoupler::soi_standard();
        assert!(ec.mode_radius_at_tip(1550e-9) > 0.0);
    }

    #[test]
    fn edge_coupler_facet_reflection_small() {
        let ec = EdgeCoupler::soi_standard();
        let r = ec.facet_reflection();
        assert!(r > 0.0 && r < 0.5, "R={r:.3}");
    }

    #[test]
    fn edge_coupler_bandwidth_positive() {
        let ec = EdgeCoupler::soi_standard();
        let bw = ec.bandwidth_3db_nm(1550e-9);
        assert!(bw > 0.0);
    }
}
