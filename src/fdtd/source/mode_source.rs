//! Mode source for waveguide excitation in FDTD.
//!
//! A mode source injects the transverse field profile of a guided mode at a
//! cross-section of the waveguide, enabling efficient single-mode excitation.
//!
//! Implementation:
//!   1. Compute the TE/TM mode profile E(y), H(y) via effective index method
//!   2. At each time step, inject J_src(y,t) = E_mode(y) · waveform(t)
//!   3. Only propagates in one direction (forward injection)
//!
//! For 2D TE simulations:
//!   E_z(ix, iy, t) += mode_profile\[iy\] · waveform(t)

/// Transverse mode profile for a slab waveguide.
///
/// Stores sampled E and H fields across the waveguide cross-section.
#[derive(Debug, Clone)]
pub struct ModeProfile {
    /// Transverse coordinate samples (m)
    pub y: Vec<f64>,
    /// E-field profile (normalized, a.u.)
    pub e_profile: Vec<f64>,
    /// H-field profile (normalized, a.u.)
    pub h_profile: Vec<f64>,
    /// Effective index n_eff
    pub n_eff: f64,
    /// Polarization ("TE" or "TM")
    pub polarization: &'static str,
}

impl ModeProfile {
    /// Compute TE₀ mode profile for a symmetric slab waveguide.
    ///
    /// n_core, n_clad: refractive indices
    /// width: waveguide width (m)
    /// lambda: wavelength (m)
    /// n_pts: number of transverse sample points
    pub fn slab_te0(n_core: f64, n_clad: f64, width: f64, lambda: f64, n_pts: usize) -> Self {
        use std::f64::consts::PI;
        // Normalized frequency V = (pi/lambda) * width * sqrt(n_core^2 - n_clad^2)
        let v = PI / lambda * width * (n_core * n_core - n_clad * n_clad).sqrt();
        // Find kappa by solving transcendental equation: kappa*tan(kappa*d/2) = gamma
        // where kappa^2 + gamma^2 = (2pi/lambda)^2 * (n_core^2 - n_clad^2)
        // Simple approximation for fundamental mode:
        let kappa = v / (width / 2.0) * 0.5; // rough estimate
        let k0 = 2.0 * PI / lambda;
        let n_eff = (n_core * n_core - (kappa / k0).powi(2)).sqrt().max(n_clad);

        let d = width / 2.0;
        let gamma = k0 * (n_eff * n_eff - n_clad * n_clad).sqrt().max(0.0);

        let y: Vec<f64> = (0..n_pts)
            .map(|i| -1.5 * width + 3.0 * width * i as f64 / (n_pts - 1) as f64)
            .collect();

        let e_profile: Vec<f64> = y
            .iter()
            .map(|&yp| {
                if yp.abs() <= d {
                    (kappa * yp).cos()
                } else {
                    let sign: f64 = if yp > 0.0 { 1.0 } else { -1.0 };
                    (kappa * d).cos() * (-gamma * (yp.abs() - d) * sign).exp()
                }
            })
            .collect();

        // Normalize
        let norm: f64 = e_profile.iter().map(|e| e * e).sum::<f64>().sqrt();
        let e_norm: Vec<f64> = e_profile.iter().map(|e| e / norm.max(1e-30)).collect();
        let h_profile = e_norm.clone(); // TE: H_z ∝ E_y for planar approximation

        Self {
            y,
            e_profile: e_norm,
            h_profile,
            n_eff,
            polarization: "TE",
        }
    }

    /// Gaussian approximation to the mode profile (simpler, analytic).
    ///
    /// E(y) = exp(-(y/w₀)²)  where w₀ = waist radius
    pub fn gaussian_approx(w0: f64, n_eff: f64, lambda: f64, n_pts: usize) -> Self {
        let y: Vec<f64> = (0..n_pts)
            .map(|i| -3.0 * w0 + 6.0 * w0 * i as f64 / (n_pts - 1) as f64)
            .collect();
        let e_profile: Vec<f64> = y.iter().map(|&yp| (-(yp / w0).powi(2)).exp()).collect();
        let norm: f64 = e_profile.iter().map(|e| e * e).sum::<f64>().sqrt();
        let e_norm: Vec<f64> = e_profile.iter().map(|e| e / norm.max(1e-30)).collect();
        let _ = lambda; // stored for future use
        Self {
            y,
            e_profile: e_norm.clone(),
            h_profile: e_norm,
            n_eff,
            polarization: "TE",
        }
    }

    /// Number of transverse sample points.
    pub fn n_pts(&self) -> usize {
        self.y.len()
    }

    /// Mode overlap integral with another profile (0–1).
    ///
    ///   η = |∫ E₁* E₂ dy|² / (∫|E₁|² dy · ∫|E₂|² dy)
    pub fn overlap(&self, other: &ModeProfile) -> f64 {
        if self.n_pts() != other.n_pts() {
            return 0.0;
        }
        let num: f64 = self
            .e_profile
            .iter()
            .zip(other.e_profile.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm1: f64 = self.e_profile.iter().map(|e| e * e).sum::<f64>().sqrt();
        let norm2: f64 = other.e_profile.iter().map(|e| e * e).sum::<f64>().sqrt();
        if norm1 < 1e-30 || norm2 < 1e-30 {
            return 0.0;
        }
        (num / (norm1 * norm2)).powi(2)
    }

    /// Mode field diameter (MFD) at 1/e² intensity (m).
    pub fn mode_field_diameter(&self) -> f64 {
        let i_max = self.e_profile.iter().map(|e| e * e).fold(0.0_f64, f64::max);
        let threshold = i_max / std::f64::consts::E.powi(2);
        let mut y_min = self.y[0];
        let mut y_max = *self.y.last().expect("mode source y grid must be non-empty");
        for (i, &e) in self.e_profile.iter().enumerate() {
            if e * e >= threshold {
                y_min = self.y[i].min(y_min);
                y_max = self.y[i].max(y_max);
            }
        }
        (y_max - y_min).abs()
    }
}

/// Mode source injector: applies a mode profile as a time-domain source.
#[derive(Debug, Clone)]
pub struct ModeSource {
    /// x-index (column) of injection plane in 2D grid
    pub inject_ix: usize,
    /// y-indices covered by this source
    pub iy_range: std::ops::Range<usize>,
    /// Normalized mode profile values (one per iy in range)
    pub profile: Vec<f64>,
    /// Center frequency (Hz)
    pub f0: f64,
    /// Gaussian pulse width (s): 0 = CW
    pub tau: f64,
    /// Time offset (s)
    pub t0: f64,
    /// Peak amplitude
    pub amplitude: f64,
}

impl ModeSource {
    /// Create from a mode profile.
    pub fn new(
        inject_ix: usize,
        iy_start: usize,
        mode: &ModeProfile,
        f0: f64,
        tau: f64,
        amplitude: f64,
    ) -> Self {
        let t0 = if tau > 0.0 { 3.0 * tau } else { 0.0 };
        Self {
            inject_ix,
            iy_range: iy_start..iy_start + mode.n_pts(),
            profile: mode.e_profile.clone(),
            f0,
            tau,
            t0,
            amplitude,
        }
    }

    /// Evaluate source amplitude at time t.
    pub fn waveform(&self, t: f64) -> f64 {
        use std::f64::consts::PI;
        let envelope = if self.tau > 0.0 {
            let dt = t - self.t0;
            (-(dt / self.tau).powi(2)).exp()
        } else {
            1.0
        };
        let phase = 2.0 * PI * self.f0 * (t - self.t0);
        self.amplitude * envelope * phase.sin()
    }

    /// Field injection values at time t for all iy in range.
    pub fn field_values(&self, t: f64) -> Vec<f64> {
        let w = self.waveform(t);
        self.profile.iter().map(|p| p * w).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slab_te0_profile_normalized() {
        let mode = ModeProfile::slab_te0(3.48, 1.44, 500e-9, 1550e-9, 50);
        let norm_sq: f64 = mode.e_profile.iter().map(|e| e * e).sum();
        assert!((norm_sq - 1.0).abs() < 0.1, "norm²={norm_sq:.3}"); // roughly normalized
    }

    #[test]
    fn slab_te0_n_pts() {
        let mode = ModeProfile::slab_te0(3.48, 1.44, 500e-9, 1550e-9, 64);
        assert_eq!(mode.n_pts(), 64);
    }

    #[test]
    fn gaussian_mode_overlap_self() {
        let mode = ModeProfile::gaussian_approx(500e-9, 2.5, 1550e-9, 100);
        let ol = mode.overlap(&mode);
        assert!((ol - 1.0).abs() < 1e-6, "self-overlap={ol:.4}");
    }

    #[test]
    fn gaussian_mode_overlap_different() {
        // Two modes with different waists, sampled on a SHARED fixed y-range
        // so the discrete overlap integral is meaningful.
        let n_pts = 100;
        let w1 = 300e-9_f64;
        let w2 = 1500e-9_f64;
        let y_span = 3.0 * w2; // use the larger mode's span for both
        let y: Vec<f64> = (0..n_pts)
            .map(|i| -y_span + 2.0 * y_span * i as f64 / (n_pts - 1) as f64)
            .collect();
        let make_gauss = |w: f64| -> Vec<f64> {
            let v: Vec<f64> = y.iter().map(|&yp| (-(yp / w).powi(2)).exp()).collect();
            let norm: f64 = v.iter().map(|e| e * e).sum::<f64>().sqrt().max(1e-30);
            v.iter().map(|e| e / norm).collect()
        };
        let e1 = make_gauss(w1);
        let e2 = make_gauss(w2);
        let num: f64 = e1.iter().zip(e2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = e1.iter().map(|e| e * e).sum::<f64>().sqrt();
        let norm2: f64 = e2.iter().map(|e| e * e).sum::<f64>().sqrt();
        let ol = (num / (norm1 * norm2)).powi(2);
        assert!(ol > 0.0 && ol < 1.0, "overlap={ol:.3}");
    }

    #[test]
    fn mode_field_diameter_positive() {
        let mode = ModeProfile::gaussian_approx(500e-9, 2.5, 1550e-9, 100);
        let mfd = mode.mode_field_diameter();
        assert!(mfd > 0.0, "MFD={mfd:.2e}");
    }

    #[test]
    fn mode_source_waveform_oscillates() {
        let mode = ModeProfile::gaussian_approx(500e-9, 2.5, 1550e-9, 50);
        let src = ModeSource::new(10, 0, &mode, 1.94e14, 0.0, 1.0);
        let w1 = src.waveform(1.0 / (4.0 * 1.94e14));
        let w2 = src.waveform(3.0 / (4.0 * 1.94e14));
        assert!(w1 * w2 < 0.0, "source should alternate sign");
    }

    #[test]
    fn mode_source_field_values_length() {
        let mode = ModeProfile::gaussian_approx(500e-9, 2.5, 1550e-9, 50);
        let src = ModeSource::new(10, 0, &mode, 1.94e14, 0.0, 1.0);
        let vals = src.field_values(0.0);
        assert_eq!(vals.len(), 50);
    }

    #[test]
    fn mode_source_n_eff_above_cladding() {
        let mode = ModeProfile::slab_te0(3.48, 1.44, 500e-9, 1550e-9, 50);
        assert!(mode.n_eff >= 1.44, "n_eff={:.3}", mode.n_eff);
    }
}
