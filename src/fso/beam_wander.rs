//! Beam wander and wandering-spot statistics for FSO propagation.
//!
//! Beam wander refers to random displacement of the instantaneous beam centroid
//! caused by large-scale turbulent eddies (scales > beam diameter).  The
//! long-term spot size grows substantially beyond the diffraction limit.
//!
//! # Key Relations (Andrews & Phillips 2005, Ch. 6)
//!
//! Short-term (instantaneous) spot size:
//!   W_ST² = W₀² [1 + (z/z_R)²] [1 + 1.33 σ²_R (2W₀/r₀)^{-1/3} / ...]
//!
//! Beam wander variance (Churnside 1989):
//!   ⟨r²⟩ = 2.42 C_n² L³ W₀^{-1/3}
//!
//! Long-term spot size:
//!   W_LT² = W_ST² + ⟨r²⟩
//!
//! Wander displacement follows a 2-D isotropic zero-mean Gaussian, so the
//! radial displacement has a Rayleigh distribution.
//!
//! # References
//! - Andrews & Phillips, "Laser Beam Propagation through Random Media", 2005
//! - Churnside, "Aperture averaging of optical scintillations", 1991

use super::turbulence::AtmosphericPath;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// BeamWander
// ─────────────────────────────────────────────────────────────────────────────

/// Beam wander analysis for a Gaussian beam propagating through turbulence.
#[derive(Debug, Clone)]
pub struct BeamWander {
    /// 1/e² intensity beam waist at the transmitter (m).
    pub beam_waist_m: f64,
    /// Optical wavelength (m).
    pub wavelength: f64,
    /// Atmospheric propagation path (turbulence model).
    pub path: AtmosphericPath,
    /// Link distance (km).
    pub link_distance_km: f64,
}

impl BeamWander {
    /// Construct a new BeamWander model.
    pub fn new(beam_waist: f64, wavelength: f64, path: AtmosphericPath, dist_km: f64) -> Self {
        Self {
            beam_waist_m: beam_waist.max(1e-6),
            wavelength,
            path,
            link_distance_km: dist_km.max(0.001),
        }
    }

    /// Rayleigh range z_R = π W₀² / λ (metres).
    fn rayleigh_range_m(&self) -> f64 {
        PI * self.beam_waist_m * self.beam_waist_m / self.wavelength
    }

    /// Link distance in metres.
    fn link_m(&self) -> f64 {
        self.link_distance_km * 1e3
    }

    /// Diffraction-only (vacuum) Gaussian beam radius at distance L:
    /// W_vac(L) = W₀ √(1 + (L/z_R)²).
    fn vacuum_spot_size_m(&self) -> f64 {
        let z = self.link_m();
        let zr = self.rayleigh_range_m();
        self.beam_waist_m * (1.0 + (z / zr).powi(2)).sqrt()
    }

    /// Short-term spot size (diffraction + small-scale turbulence, excluding wander).
    ///
    /// W_ST² ≈ W_vac² * (1 + ...)  using Andrews & Phillips (2005) Eq. 6.100:
    /// W_ST² = W_vac² + W₀² * T_turb
    /// where T_turb = 1.63 (σ²_R)^{6/5} (W₀/r₀)^{-1/3} is a turbulence broadening
    /// correction.
    pub fn short_term_spot_size_m(&self) -> f64 {
        let w_vac = self.vacuum_spot_size_m();
        let r0 = self.path.fried_parameter_m();
        let sr = self.path.rytov_variance();
        // Turbulence induced short-term broadening
        let w0 = self.beam_waist_m;
        let ratio = (w0 / r0).powf(-1.0 / 3.0); // dimensionless
        let t_turb = 1.63 * sr.powf(6.0 / 5.0) * ratio;
        let w_st2 = w_vac * w_vac + w0 * w0 * t_turb.max(0.0);
        w_st2.sqrt()
    }

    /// Beam wander variance ⟨r²⟩ (m²).
    ///
    /// Churnside (1989) formula for plane-wave limit:
    ///   ⟨r²⟩ = 0.54 (L/z_R)² (λ/2W₀)^{1/3} C_n² L² (2W₀)^{-1/3}
    ///
    /// Simplified Andrews form (valid for Gaussian beam, weak turbulence):
    ///   ⟨r²⟩ = 2.42 C_n² L³ W₀^{-1/3} [in metres]
    pub fn wander_variance_m2(&self) -> f64 {
        let cn2 = self.path.cn2_profile.cn2_at_height(self.path.h_start_m);
        let l = self.link_m();
        let w0 = self.beam_waist_m;
        // Equation from Andrews & Phillips (2005), Eq. 6.83 (corrected form)
        // ⟨r²⟩ = 2.42 Cn² k^{-1/3} L^3 W₀^{-1/3}  (k = 2π/λ)
        let k = 2.0 * PI / self.wavelength;
        2.42 * cn2 * k.powf(-1.0 / 3.0) * l.powi(3) * w0.powf(-1.0 / 3.0)
    }

    /// RMS beam wander displacement σ_wander = √⟨r²⟩ (metres).
    pub fn rms_wander_m(&self) -> f64 {
        self.wander_variance_m2().sqrt()
    }

    /// Long-term spot size (diffraction + small-scale + wander):
    /// W_LT² = W_ST² + ⟨r²⟩
    pub fn long_term_spot_size_m(&self) -> f64 {
        let w_st = self.short_term_spot_size_m();
        let wander = self.wander_variance_m2();
        (w_st * w_st + wander).sqrt()
    }

    /// Residual pointing error (at the receiver plane) after tip-tilt correction
    /// of `n_zernike_modes` Zernike modes.
    ///
    /// Without correction: σ_PE = σ_wander.
    /// With tip-tilt only (n≥2): σ_PE ≈ σ_wander * (1 − correction_factor)^{1/2}.
    /// With higher-order correction: residual decreases as ≈ 0.5^n.
    pub fn pointing_error_after_correction_m(&self, n_zernike_modes: usize) -> f64 {
        let sigma_wander = self.rms_wander_m();
        let correction_frac = match n_zernike_modes {
            0 => 0.0,
            1 => 0.0,      // piston only — no wander correction
            2 | 3 => 0.85, // tip/tilt modes
            4..=10 => 0.92,
            _ => 0.97,
        };
        sigma_wander * (1.0_f64 - correction_frac).sqrt()
    }

    /// Wander-induced fade depth (dB) for a receiver aperture of diameter `aperture_m`.
    ///
    /// When the beam centroid drifts by distance r from the aperture centre, the
    /// fraction of power captured by a receiver aperture of radius r_ap is:
    ///   η(r) = 1 − exp(−2 r_ap² / W_LT²)   (roughly, for Gaussian beam)
    ///
    /// The ensemble-average power loss due to wander is computed by integrating
    /// η(r) over the Rayleigh distribution of r.
    pub fn wander_induced_fade_db(&self, aperture_m: f64) -> f64 {
        let r_ap = aperture_m / 2.0;
        let sigma_sq = self.wander_variance_m2() / 2.0; // per-axis variance
        let w_lt = self.long_term_spot_size_m();

        if sigma_sq <= 0.0 || w_lt <= 0.0 {
            return 0.0;
        }

        // Fraction of Gaussian beam power captured by a circular aperture radius r_ap
        // when the beam centroid is displaced radially by r from the aperture centre.
        //
        // For a Gaussian beam with 1/e² waist W, the captured power (fraction of total)
        // is approximately: η_0 = 1 - exp(-2 r_ap² / W²)  (centred case).
        // When the centroid shifts by r, the captured power decreases; for r_ap ≫ W
        // the beam exits the aperture.  We use the power carried in the central disc
        // approximation:  η(r) ≈ η_0 * exp(-r²/r_ap²) (heuristic, monotone in r).
        //
        // This is a conservative estimate ensuring η(r) ≤ η(0) for all r > 0.
        let static_capture = 1.0 - (-2.0 * r_ap * r_ap / (w_lt * w_lt)).exp();
        let eta0 = static_capture;

        // Average captured fraction over Rayleigh distribution of centroid offset.
        let wander_capture = {
            let n = 400;
            let r_max = 6.0 * sigma_sq.sqrt();
            let dr = r_max / n as f64;
            let mut integral = 0.0;
            for i in 0..=n {
                let r = i as f64 * dr;
                let rayleigh = if i == 0 {
                    0.0
                } else {
                    r / sigma_sq * (-r * r / (2.0 * sigma_sq)).exp()
                };
                // η(r) = η_0 * exp(-r²/r_ap²): decreases monotonically with displacement
                let eta_r = eta0 * (-r * r / (r_ap * r_ap)).exp();
                let w = if i == 0 || i == n { 0.5 } else { 1.0 };
                integral += w * rayleigh * eta_r;
            }
            integral * dr
        };
        if static_capture <= 0.0 || wander_capture <= 0.0 {
            return 0.0;
        }
        // Fade = reduction in power due to wander (always ≥ 0)
        let ratio = wander_capture / static_capture;
        (-10.0 * ratio.log10()).max(0.0)
    }

    /// PDF of beam wander radial displacement (Rayleigh distribution).
    ///
    /// p(r) = r / σ² * exp(−r² / (2σ²))
    /// where σ² = ⟨r²⟩ / 2 (per-axis variance).
    pub fn wander_pdf(&self, displacement_m: f64) -> f64 {
        if displacement_m < 0.0 {
            return 0.0;
        }
        let sigma_sq = self.wander_variance_m2() / 2.0;
        if sigma_sq <= 0.0 {
            return 0.0;
        }
        let r = displacement_m;
        (r / sigma_sq) * (-r * r / (2.0 * sigma_sq)).exp()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TipTiltCorrection
// ─────────────────────────────────────────────────────────────────────────────

/// Models the effectiveness of a tip-tilt AO correction loop on beam wander.
#[derive(Debug, Clone)]
pub struct TipTiltCorrection {
    /// Underlying beam wander model.
    pub wander: BeamWander,
    /// AO loop bandwidth (Hz) — determines how quickly wander is tracked.
    pub update_rate_hz: f64,
    /// Residual uncorrected fraction of wander (0 = perfect, 1 = no correction).
    pub residual_error_fraction: f64,
}

impl TipTiltCorrection {
    /// Construct from a BeamWander model and the loop bandwidth.
    ///
    /// The residual error fraction is estimated from the Greenwood frequency
    /// f_G = 0.102 (k² C_n² v_w L)^{3/5} and the loop bandwidth:
    /// residual ≈ (f_G / BW)^{5/3} for a single integrator.
    pub fn new(wander: BeamWander, bandwidth_hz: f64) -> Self {
        let bw = bandwidth_hz.max(0.1);
        // Greenwood frequency (horizontal path, wind speed approximation)
        let cn2 = wander.path.cn2_profile.cn2_at_height(wander.path.h_start_m);
        let k = 2.0 * PI / wander.wavelength;
        let l = wander.link_m();
        let v_wind = 10.0_f64; // assumed 10 m/s wind speed for estimate
        let f_g = 0.102 * (k * k * cn2 * v_wind * l).powf(0.6);
        let residual = (f_g / bw).powf(5.0 / 3.0).min(1.0);
        Self {
            wander,
            update_rate_hz: bw,
            residual_error_fraction: residual,
        }
    }

    /// Residual beam wander RMS after correction (metres).
    pub fn residual_wander_m(&self) -> f64 {
        self.wander.rms_wander_m() * self.residual_error_fraction.sqrt()
    }

    /// Fraction of total wander variance that was corrected: 1 − residual_fraction.
    pub fn correction_efficiency(&self) -> f64 {
        (1.0 - self.residual_error_fraction).clamp(0.0, 1.0)
    }

    /// Signal power improvement (dB) versus no wander correction.
    ///
    /// Computed as the reduction in wander-induced fade depth.
    pub fn improvement_db(&self, aperture_m: f64) -> f64 {
        let fade_no_corr = self.wander.wander_induced_fade_db(aperture_m);
        // Build a corrected wander model with reduced variance
        let mut corrected_path = self.wander.path.clone();
        // Scale effective C_n² to match residual variance
        let cn2_eff =
            self.wander.path.cn2_profile.cn2_at_height(0.0) * self.residual_error_fraction;
        corrected_path.cn2_profile = super::turbulence::Cn2Profile::Constant(cn2_eff);
        let corrected_wander = BeamWander::new(
            self.wander.beam_waist_m,
            self.wander.wavelength,
            corrected_path,
            self.wander.link_distance_km,
        );
        let fade_corrected = corrected_wander.wander_induced_fade_db(aperture_m);
        (fade_no_corr - fade_corrected).max(0.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::turbulence::AtmosphericPath;
    use super::*;

    fn make_wander(cn2: f64, dist_km: f64) -> BeamWander {
        let path = AtmosphericPath::new_horizontal(dist_km, cn2, 1550e-9);
        BeamWander::new(0.05, 1550e-9, path, dist_km)
    }

    /// Long-term spot size must be ≥ short-term spot size.
    #[test]
    fn test_lt_ge_st_spot_size() {
        let bw = make_wander(1e-14, 2.0);
        assert!(
            bw.long_term_spot_size_m() >= bw.short_term_spot_size_m(),
            "LT = {} ST = {}",
            bw.long_term_spot_size_m(),
            bw.short_term_spot_size_m()
        );
    }

    /// Wander variance must be positive.
    #[test]
    fn test_wander_variance_positive() {
        let bw = make_wander(1e-14, 2.0);
        assert!(bw.wander_variance_m2() > 0.0);
    }

    /// Stronger turbulence → larger wander variance.
    #[test]
    fn test_wander_increases_with_cn2() {
        let bw_weak = make_wander(1e-17, 2.0);
        let bw_strong = make_wander(1e-13, 2.0);
        assert!(bw_strong.wander_variance_m2() > bw_weak.wander_variance_m2());
    }

    /// Rayleigh PDF must integrate to ~1 (check via quadrature).
    #[test]
    fn test_rayleigh_pdf_normalisation() {
        let bw = make_wander(1e-14, 2.0);
        let sigma = bw.rms_wander_m() / 2.0_f64.sqrt();
        let n = 1000;
        let r_max = 8.0 * sigma;
        let dr = r_max / n as f64;
        let mut integral = 0.0;
        for i in 0..=n {
            let r = i as f64 * dr;
            let w = if i == 0 || i == n { 0.5 } else { 1.0 };
            integral += w * bw.wander_pdf(r);
        }
        integral *= dr;
        assert!(
            (integral - 1.0).abs() < 0.02,
            "PDF integral = {integral:.4}"
        );
    }

    /// More tip-tilt correction modes → smaller residual error.
    #[test]
    fn test_correction_residual_decreases_with_modes() {
        let bw = make_wander(1e-14, 2.0);
        let e0 = bw.pointing_error_after_correction_m(0);
        let e2 = bw.pointing_error_after_correction_m(2);
        let e10 = bw.pointing_error_after_correction_m(10);
        assert!(e0 >= e2, "e0={e0:.4e} e2={e2:.4e}");
        assert!(e2 >= e10, "e2={e2:.4e} e10={e10:.4e}");
    }

    /// Tip-tilt correction with high bandwidth → high correction efficiency.
    #[test]
    fn test_tip_tilt_high_bandwidth() {
        let bw = make_wander(1e-14, 2.0);
        let tt = TipTiltCorrection::new(bw, 10_000.0);
        assert!(
            tt.correction_efficiency() > 0.5,
            "efficiency = {}",
            tt.correction_efficiency()
        );
    }

    /// Wander-induced fade must be non-negative.
    #[test]
    fn test_fade_non_negative() {
        let bw = make_wander(1e-14, 2.0);
        assert!(bw.wander_induced_fade_db(0.2) >= 0.0);
    }

    /// Vacuum (zero turbulence) beam: long-term ≈ short-term.
    #[test]
    fn test_vacuum_beam_no_wander() {
        let path = AtmosphericPath::new_horizontal(1.0, 1e-30, 1550e-9);
        let bw = BeamWander::new(0.05, 1550e-9, path, 1.0);
        let wander = bw.wander_variance_m2();
        assert!(
            wander < 1e-20,
            "Wander in vacuum should be ~0, got {wander:.4e}"
        );
    }
}
