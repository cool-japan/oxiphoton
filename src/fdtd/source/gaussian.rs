//! Gaussian pulse and Gaussian beam sources for FDTD.
//!
//! Provides:
//! - `GaussianPulse` — simple Gaussian envelope waveform
//! - `GaussianBeam3d` — 3D Gaussian beam source with full paraxial optics
//!
//! The Gaussian beam model follows the standard paraxial approximation:
//!   E(r, z) = E₀ · (w₀/w(z)) · exp(-r²/w(z)²) · exp(-i(kz + k·r²/(2R(z)) - ψ(z)))
//!
//! where:
//!   w(z) = w₀ · sqrt(1 + (z/z_R)²)   — beam radius
//!   R(z) = z + z_R²/z               — radius of curvature
//!   ψ(z) = atan(z/z_R)               — Gouy phase
//!   z_R  = π·w₀²/λ                  — Rayleigh range

use crate::fdtd::source::plane_wave::{Polarization3d, PropagationAxis};
use std::f64::consts::PI;

/// Gaussian pulse waveform parameters
#[derive(Debug, Clone, Copy)]
pub struct GaussianPulse {
    pub t0: f64,
    pub tau: f64,
}

impl GaussianPulse {
    /// Create Gaussian pulse with peak at t0 and width tau (seconds)
    pub fn new(t0: f64, tau: f64) -> Self {
        Self { t0, tau }
    }

    /// Compute amplitude at time t
    pub fn amplitude(&self, t: f64) -> f64 {
        let dt = t - self.t0;
        (-(dt / self.tau).powi(2)).exp()
    }
}

/// 3D Gaussian beam source for FDTD injection.
///
/// Models a paraxial Gaussian beam with:
/// - A focus position at (focus_x, focus_y, focus_z)
/// - A propagation direction given by `propagation`
/// - A transverse beam profile injected at the plane `k = plane_k` (for Z propagation)
///
/// The beam is injected as a soft source on a 2D injection plane.
/// The field at each transverse point is weighted by the Gaussian profile.
#[derive(Debug, Clone)]
pub struct GaussianBeam3d {
    /// Free-space wavelength (m)
    pub wavelength: f64,
    /// Beam waist radius w₀ at focus (m)
    pub beam_waist: f64,
    /// Focus position x (m)
    pub focus_x: f64,
    /// Focus position y (m)
    pub focus_y: f64,
    /// Focus position z (m)
    pub focus_z: f64,
    /// Propagation direction
    pub propagation: PropagationAxis,
    /// E-field polarization
    pub polarization: Polarization3d,
    /// Peak E-field amplitude (V/m)
    pub amplitude: f64,
    /// Phase offset (rad)
    pub phase: f64,
}

impl GaussianBeam3d {
    /// Create a new Gaussian beam source.
    ///
    /// Defaults focus at origin, zero phase.
    pub fn new(
        wavelength: f64,
        beam_waist: f64,
        propagation: PropagationAxis,
        polarization: Polarization3d,
    ) -> Self {
        Self {
            wavelength,
            beam_waist,
            focus_x: 0.0,
            focus_y: 0.0,
            focus_z: 0.0,
            propagation,
            polarization,
            amplitude: 1.0,
            phase: 0.0,
        }
    }

    /// Set the focus position.
    pub fn at_focus(mut self, x: f64, y: f64, z: f64) -> Self {
        self.focus_x = x;
        self.focus_y = y;
        self.focus_z = z;
        self
    }

    /// Set peak amplitude (V/m).
    pub fn with_amplitude(mut self, amp: f64) -> Self {
        self.amplitude = amp;
        self
    }

    /// Set phase offset (rad).
    pub fn with_phase(mut self, phase: f64) -> Self {
        self.phase = phase;
        self
    }

    /// Rayleigh range z_R = π · w₀² / λ (m).
    pub fn rayleigh_range(&self) -> f64 {
        PI * self.beam_waist * self.beam_waist / self.wavelength
    }

    /// Beam radius at propagation distance z from the focus (m).
    ///
    /// w(z) = w₀ · sqrt(1 + (z/z_R)²)
    pub fn beam_radius(&self, z: f64) -> f64 {
        let zr = self.rayleigh_range();
        self.beam_waist * (1.0 + (z / zr).powi(2)).sqrt()
    }

    /// Radius of curvature of the phase front at distance z from focus (m).
    ///
    /// R(z) = z · (1 + (z_R/z)²)
    /// Returns f64::INFINITY at z=0 (flat wave front at focus).
    pub fn radius_of_curvature(&self, z: f64) -> f64 {
        if z.abs() < 1e-30 {
            return f64::INFINITY;
        }
        let zr = self.rayleigh_range();
        z * (1.0 + (zr / z).powi(2))
    }

    /// Gouy phase at propagation distance z from focus (rad).
    ///
    /// ψ(z) = atan(z / z_R)
    pub fn gouy_phase(&self, z: f64) -> f64 {
        let zr = self.rayleigh_range();
        (z / zr).atan()
    }

    /// Compute the real E-field amplitude at transverse position (x, y) relative to beam axis,
    /// propagation distance z from focus, and time t with angular frequency omega.
    ///
    /// Implements the paraxial Gaussian beam:
    ///   E = A · (w₀/w(z)) · exp(-r²/w²(z)) · cos(ω·t - k·z - k·r²/(2R(z)) + ψ(z) + phase)
    ///
    /// # Arguments
    /// * `x`, `y` — transverse coordinates relative to beam axis (m)
    /// * `z` — propagation distance from focus (m, signed: negative = before focus)
    /// * `t` — simulation time (s)
    /// * `omega` — angular frequency (rad/s)
    pub fn field_at(&self, x: f64, y: f64, z: f64, t: f64, omega: f64) -> f64 {
        let w = self.beam_radius(z);
        let zr = self.rayleigh_range();
        let k = 2.0 * PI / self.wavelength;
        let r2 = x * x + y * y;

        // Amplitude profile
        let profile = self.amplitude * (self.beam_waist / w) * (-r2 / (w * w)).exp();

        // Phase: propagation + wavefront curvature + Gouy + user phase
        let r_curv = self.radius_of_curvature(z);
        let curvature_phase = if r_curv.is_finite() && r_curv.abs() > 1e-30 {
            k * r2 / (2.0 * r_curv)
        } else {
            0.0
        };
        let _ = zr; // used indirectly via gouy_phase
        let total_phase = omega * t - k * z - curvature_phase + self.gouy_phase(z) + self.phase;

        profile * total_phase.cos()
    }

    /// Compute beam intensity |E|² at transverse position (x, y) and axial position z.
    ///
    /// I(r,z) = I₀ · (w₀/w(z))² · exp(-2r²/w²(z))
    pub fn intensity_at(&self, x: f64, y: f64, z: f64) -> f64 {
        let w = self.beam_radius(z);
        let r2 = x * x + y * y;
        let i0 = self.amplitude * self.amplitude;
        i0 * (self.beam_waist / w).powi(2) * (-2.0 * r2 / (w * w)).exp()
    }

    /// Compute the beam power (total integrated intensity over the transverse plane).
    ///
    /// P = π/2 · I₀ · w₀²  (in appropriate units)
    pub fn total_power(&self) -> f64 {
        // Integral of I₀ * exp(-2r²/w₀²) * 2πr dr from 0 to ∞ = π/2 * I₀ * w₀²
        let i0 = self.amplitude * self.amplitude;
        PI / 2.0 * i0 * self.beam_waist * self.beam_waist
    }

    /// Apply this Gaussian beam source to an injection plane in the 3D FDTD grid.
    ///
    /// Injects the Gaussian beam profile onto the specified 2D plane.
    /// The plane is perpendicular to the propagation axis.
    ///
    /// # Arguments
    /// * `t` — current simulation time (s)
    /// * `omega` — angular frequency (rad/s)
    /// * `ex`, `ey`, `ez` — mutable 3D E-field arrays
    /// * `nx`, `ny`, `nz` — grid dimensions
    /// * `plane_k` — plane index along the propagation axis
    /// * `dx`, `dy` — transverse cell spacings (m)
    /// * `x0`, `y0` — physical origin of the injection plane (m)
    /// * `nx_grid`, `ny_grid` — transverse grid dimensions on the injection plane
    #[allow(clippy::too_many_arguments)]
    pub fn apply_to_plane(
        &self,
        t: f64,
        omega: f64,
        ex: &mut [f64],
        ey: &mut [f64],
        ez: &mut [f64],
        nx: usize,
        ny: usize,
        nz: usize,
        plane_k: usize,
        dx: f64,
        dy: f64,
        x0: f64,
        y0: f64,
    ) {
        // Determine propagation distance z from focus based on axis
        match self.propagation.axis_index() {
            2 => {
                // Z propagation: injection plane at k = plane_k
                if plane_k >= nz {
                    return;
                }
                // Physical z of injection plane (z origin assumed at 0.0)
                let dz = dx; // assume isotropic for z offset
                let z_plane = 0.0_f64 + plane_k as f64 * dz;
                let z_from_focus = z_plane - self.focus_z;
                let sign = if self.propagation.is_positive() {
                    1.0
                } else {
                    -1.0
                };

                for i in 0..nx {
                    for j in 0..ny {
                        let xp = x0 + i as f64 * dx - self.focus_x;
                        let yp = y0 + j as f64 * dy - self.focus_y;
                        let val = self.field_at(xp, yp, sign * z_from_focus, t, omega);
                        let idx = i * ny * nz + j * nz + plane_k;
                        match self.polarization {
                            Polarization3d::Ex => {
                                if idx < ex.len() {
                                    ex[idx] += val;
                                }
                            }
                            Polarization3d::Ey => {
                                if idx < ey.len() {
                                    ey[idx] += val;
                                }
                            }
                            Polarization3d::Ez => {
                                if idx < ez.len() {
                                    ez[idx] += val;
                                }
                            }
                        }
                    }
                }
            }
            1 => {
                // Y propagation: injection plane at j = plane_k
                if plane_k >= ny {
                    return;
                }
                let dy_prop = dy;
                let y_plane = y0 + plane_k as f64 * dy_prop;
                let y_from_focus = y_plane - self.focus_y;
                let sign = if self.propagation.is_positive() {
                    1.0
                } else {
                    -1.0
                };

                for i in 0..nx {
                    for k in 0..nz {
                        let xp = x0 + i as f64 * dx - self.focus_x;
                        let zp = k as f64 * dx - self.focus_z;
                        let val = self.field_at(xp, zp, sign * y_from_focus, t, omega);
                        let idx = i * ny * nz + plane_k * nz + k;
                        match self.polarization {
                            Polarization3d::Ex => {
                                if idx < ex.len() {
                                    ex[idx] += val;
                                }
                            }
                            Polarization3d::Ey => {
                                if idx < ey.len() {
                                    ey[idx] += val;
                                }
                            }
                            Polarization3d::Ez => {
                                if idx < ez.len() {
                                    ez[idx] += val;
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                // X propagation: injection plane at i = plane_k
                if plane_k >= nx {
                    return;
                }
                let x_plane = x0 + plane_k as f64 * dx;
                let x_from_focus = x_plane - self.focus_x;
                let sign = if self.propagation.is_positive() {
                    1.0
                } else {
                    -1.0
                };

                for j in 0..ny {
                    for k in 0..nz {
                        let yp = y0 + j as f64 * dy - self.focus_y;
                        let zp = k as f64 * dy - self.focus_z;
                        let val = self.field_at(yp, zp, sign * x_from_focus, t, omega);
                        let idx = plane_k * ny * nz + j * nz + k;
                        match self.polarization {
                            Polarization3d::Ex => {
                                if idx < ex.len() {
                                    ex[idx] += val;
                                }
                            }
                            Polarization3d::Ey => {
                                if idx < ey.len() {
                                    ey[idx] += val;
                                }
                            }
                            Polarization3d::Ez => {
                                if idx < ez.len() {
                                    ez[idx] += val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Compute the normalized 2D intensity pattern on the injection plane.
    ///
    /// Returns a flat Vec of intensities with shape [nx, ny] (row-major).
    #[allow(clippy::too_many_arguments)]
    pub fn transverse_intensity_map(
        &self,
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
        x0: f64,
        y0: f64,
        z_from_focus: f64,
    ) -> Vec<f64> {
        let mut out = vec![0.0f64; nx * ny];
        for i in 0..nx {
            for j in 0..ny {
                let x = x0 + i as f64 * dx - self.focus_x;
                let y = y0 + j as f64 * dy - self.focus_y;
                out[i * ny + j] = self.intensity_at(x, y, z_from_focus);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_pulse_peak_at_t0() {
        let gp = GaussianPulse::new(30e-15, 10e-15);
        let amp_at_peak = gp.amplitude(30e-15);
        assert!(
            (amp_at_peak - 1.0).abs() < 1e-12,
            "Peak should be 1.0: {amp_at_peak}"
        );
        let amp_before = gp.amplitude(0.0);
        assert!(
            amp_before < amp_at_peak,
            "Amplitude before peak should be less than 1"
        );
    }

    #[test]
    fn gaussian_pulse_decays_away_from_peak() {
        let gp = GaussianPulse::new(30e-15, 10e-15);
        let a_center = gp.amplitude(30e-15);
        let a_far = gp.amplitude(100e-15);
        assert!(
            a_far < a_center * 0.01,
            "Far from peak amplitude should be very small"
        );
    }

    #[test]
    fn gaussian_beam_rayleigh_range() {
        // For w0=1µm, λ=1.55µm: z_R = π*w₀²/λ = π*(1e-6)²/(1.55e-6) ≈ 2.02 µm
        let beam = GaussianBeam3d::new(1.55e-6, 1e-6, PropagationAxis::PlusZ, Polarization3d::Ex);
        let zr = beam.rayleigh_range();
        let expected = PI * 1e-12 / 1.55e-6;
        assert!(
            (zr - expected).abs() < 1e-14,
            "Rayleigh range: {zr:.3e} vs {expected:.3e}"
        );
    }

    #[test]
    fn gaussian_beam_radius_at_focus_is_waist() {
        let w0 = 2e-6;
        let beam = GaussianBeam3d::new(1.55e-6, w0, PropagationAxis::PlusZ, Polarization3d::Ex);
        let w_at_focus = beam.beam_radius(0.0);
        assert!(
            (w_at_focus - w0).abs() < 1e-15,
            "w(0) should equal w0: {w_at_focus:.3e}"
        );
    }

    #[test]
    fn gaussian_beam_radius_grows_beyond_rayleigh() {
        let w0 = 1e-6;
        let beam = GaussianBeam3d::new(1.55e-6, w0, PropagationAxis::PlusZ, Polarization3d::Ex);
        let zr = beam.rayleigh_range();
        let w_at_zr = beam.beam_radius(zr);
        let expected = w0 * 2.0_f64.sqrt();
        assert!(
            (w_at_zr - expected).abs() < 1e-14,
            "w(z_R) = w0*sqrt(2): {w_at_zr:.3e} vs {expected:.3e}"
        );
    }

    #[test]
    fn gaussian_beam_gouy_phase_at_rayleigh() {
        let beam = GaussianBeam3d::new(1.55e-6, 1e-6, PropagationAxis::PlusZ, Polarization3d::Ex);
        let zr = beam.rayleigh_range();
        let psi = beam.gouy_phase(zr);
        assert!(
            (psi - PI / 4.0).abs() < 1e-10,
            "Gouy phase at z_R should be π/4: {psi}"
        );
    }

    #[test]
    fn gaussian_beam_intensity_peak_at_focus_center() {
        let beam = GaussianBeam3d::new(1.55e-6, 1e-6, PropagationAxis::PlusZ, Polarization3d::Ex)
            .with_amplitude(2.0);
        let i_center = beam.intensity_at(0.0, 0.0, 0.0);
        let i_off = beam.intensity_at(2e-6, 0.0, 0.0);
        assert!(i_center > i_off, "Intensity should peak at center");
        assert!(
            (i_center - 4.0).abs() < 1e-10,
            "I_center = A² = 4: {i_center}"
        );
    }

    #[test]
    fn gaussian_beam_field_at_on_axis() {
        let beam = GaussianBeam3d::new(1.55e-6, 2e-6, PropagationAxis::PlusZ, Polarization3d::Ex)
            .with_amplitude(1.0);
        let omega = 2.0 * PI * 2.998e8 / 1.55e-6;
        let t = 0.0;
        // On-axis at focus: field = A * cos(phase)
        let f = beam.field_at(0.0, 0.0, 0.0, t, omega);
        assert!(
            f.abs() <= 1.0 + 1e-10,
            "On-axis field should not exceed amplitude: {f}"
        );
    }

    #[test]
    fn gaussian_beam_apply_to_plane_modifies_fields() {
        let beam = GaussianBeam3d::new(1.55e-6, 5e-6, PropagationAxis::PlusZ, Polarization3d::Ex)
            .at_focus(40e-9 * 10.0, 40e-9 * 10.0, 40e-9 * 5.0)
            .with_amplitude(1.0);
        let omega = 2.0 * PI * 2.998e8 / 1.55e-6;
        let nx = 20;
        let ny = 20;
        let nz = 12;
        let n = nx * ny * nz;
        let mut ex = vec![0.0f64; n];
        let mut ey = vec![0.0f64; n];
        let mut ez = vec![0.0f64; n];
        let t = 0.0;
        beam.apply_to_plane(
            t, omega, &mut ex, &mut ey, &mut ez, nx, ny, nz, 5, 40e-9, 40e-9, 0.0, 0.0,
        );
        let any_nonzero = ex.iter().any(|&v| v.abs() > 0.0);
        assert!(any_nonzero, "apply_to_plane should modify Ex field");
    }

    #[test]
    fn gaussian_beam_total_power_positive() {
        let beam = GaussianBeam3d::new(1.55e-6, 1e-6, PropagationAxis::PlusZ, Polarization3d::Ex)
            .with_amplitude(1.0);
        let p = beam.total_power();
        assert!(p > 0.0, "Total power should be positive: {p:.3e}");
    }

    #[test]
    fn gaussian_beam_transverse_intensity_map_shape() {
        let beam = GaussianBeam3d::new(1.55e-6, 2e-6, PropagationAxis::PlusZ, Polarization3d::Ex);
        let nx = 10;
        let ny = 8;
        let map = beam.transverse_intensity_map(nx, ny, 1e-6, 1e-6, -5e-6, -4e-6, 0.0);
        assert_eq!(map.len(), nx * ny, "Map should have nx*ny elements");
        assert!(
            map.iter().all(|&v| v >= 0.0),
            "All intensities should be non-negative"
        );
    }
}
