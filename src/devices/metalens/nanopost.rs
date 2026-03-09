//! Metalens nanopost unit cell model.
//!
//! A metalens consists of an array of sub-wavelength nanoposts (pillars) whose
//! diameter determines the local phase shift of transmitted light. By spatially
//! varying the post diameter, an arbitrary phase profile is imprinted.
//!
//! Phase shift vs. diameter is computed from Mie theory or RCWA for a given:
//!   - Post material (e.g., TiO₂, GaN, Si)
//!   - Post height H
//!   - Wavelength λ
//!   - Array pitch Λ
//!
//! Transmission amplitude and phase: T(D) = |T|·exp(iφ) where:
//!   - φ ∈ [0, 2π] (full phase coverage) for optimized height
//!   - |T| ≈ 1 (high-efficiency condition: near-unity transmission amplitude)
//!
//! Design flow:
//!   1. Sweep diameter D from 0 to Λ
//!   2. For each D, compute φ(D) and |T(D)| via RCWA or Mie
//!   3. Create lookup table φ → D (inverse map)
//!   4. For each lens pixel, look up required D from phase profile

use std::f64::consts::PI;

/// Nanopost unit cell with diameter-phase lookup table.
#[derive(Debug, Clone)]
pub struct NanopostLibrary {
    /// Array pitch Λ (m)
    pub pitch: f64,
    /// Post height H (m)
    pub height: f64,
    /// Post material refractive index
    pub n_post: f64,
    /// Substrate index
    pub n_sub: f64,
    /// Superstrate (cover) index
    pub n_sup: f64,
    /// Design wavelength (m)
    pub wavelength: f64,
    /// Diameter sweep (m)
    pub diameters: Vec<f64>,
    /// Transmission amplitude for each diameter (0–1)
    pub transmittance: Vec<f64>,
    /// Phase (rad) for each diameter (0–2π)
    pub phase: Vec<f64>,
}

impl NanopostLibrary {
    /// Create a nanopost library with an analytic phase model.
    ///
    /// Uses a simplified waveguide phase model:
    ///   φ(D) = (β(D) - β_sub) · H
    ///   β(D) ≈ k₀ · n_eff(D)  (effective index of the nanopost as a waveguide)
    ///
    /// This is an approximation; accurate results require RCWA.
    pub fn new(pitch: f64, height: f64, n_post: f64, n_sub: f64, wavelength: f64) -> Self {
        let n_pts = 50;
        let d_min = 0.05 * pitch;
        let d_max = 0.85 * pitch;
        let k0 = 2.0 * PI / wavelength;

        let diameters: Vec<f64> = (0..n_pts)
            .map(|i| d_min + (d_max - d_min) * i as f64 / (n_pts - 1) as f64)
            .collect();

        let phase: Vec<f64> = diameters
            .iter()
            .map(|&d| {
                // Approximate n_eff via EIM for a cylindrical post:
                // Fill fraction f = π(d/2)²/Λ²
                let f = PI * (d / 2.0).powi(2) / (pitch * pitch);
                let f = f.clamp(0.0, 0.95);
                let n_eff = (f * n_post * n_post + (1.0 - f) * n_sub * n_sub).sqrt();
                // Phase accumulation
                let phi = k0 * (n_eff - n_sub) * height;
                phi.rem_euclid(2.0 * PI)
            })
            .collect();

        // Assume high transmission (simplified model)
        let transmittance = vec![0.95; n_pts];

        Self {
            pitch,
            height,
            n_post,
            n_sub,
            n_sup: 1.0,
            wavelength,
            diameters,
            transmittance,
            phase,
        }
    }

    /// TiO₂ nanopost on glass at 532 nm (visible metalens).
    ///
    /// Pitch = 350 nm, height = 600 nm, n_TiO₂ ≈ 2.35 at 532 nm.
    pub fn tio2_532nm() -> Self {
        Self::new(350e-9, 600e-9, 2.35, 1.46, 532e-9)
    }

    /// GaN nanopost at 405 nm (UV-vis metalens).
    ///
    /// Pitch = 280 nm, height = 500 nm, n_GaN ≈ 2.55.
    pub fn gan_405nm() -> Self {
        Self::new(280e-9, 500e-9, 2.55, 1.46, 405e-9)
    }

    /// Si nanopost at 1550 nm (NIR metalens).
    ///
    /// Pitch = 700 nm, height = 600 nm, n_Si ≈ 3.48.
    pub fn si_1550nm() -> Self {
        Self::new(700e-9, 600e-9, 3.48, 1.46, 1550e-9)
    }

    /// Phase range covered (rad): max - min.
    pub fn phase_range(&self) -> f64 {
        let max = self.phase.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = self.phase.iter().cloned().fold(f64::INFINITY, f64::min);
        max - min
    }

    /// Does the library cover a full 2π phase range?
    pub fn has_full_phase_coverage(&self) -> bool {
        self.phase_range() >= 1.8 * PI // require 0.9 × 2π
    }

    /// Find diameter (m) for a target phase (rad) by linear interpolation.
    ///
    /// Returns None if phase is outside the library range.
    pub fn diameter_for_phase(&self, target_phase: f64) -> Option<f64> {
        let target = target_phase.rem_euclid(2.0 * PI);
        // Find two adjacent phase points that bracket target
        for i in 0..self.phase.len().saturating_sub(1) {
            let p0 = self.phase[i];
            let p1 = self.phase[i + 1];
            if (p0 <= target && target <= p1) || (p1 <= target && target <= p0) {
                let t = if (p1 - p0).abs() < 1e-10 {
                    0.0
                } else {
                    (target - p0) / (p1 - p0)
                };
                let d = self.diameters[i] + t * (self.diameters[i + 1] - self.diameters[i]);
                return Some(d);
            }
        }
        None
    }

    /// Average transmittance across the library.
    pub fn average_transmittance(&self) -> f64 {
        self.transmittance.iter().sum::<f64>() / self.transmittance.len() as f64
    }

    /// Number of distinct diameter entries.
    pub fn n_diameters(&self) -> usize {
        self.diameters.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nanopost_library_tio2_created() {
        let lib = NanopostLibrary::tio2_532nm();
        assert!(lib.n_diameters() > 0);
    }

    #[test]
    fn nanopost_phase_range_positive() {
        let lib = NanopostLibrary::si_1550nm();
        assert!(
            lib.phase_range() > 0.0,
            "phase range={:.2}",
            lib.phase_range()
        );
    }

    #[test]
    fn nanopost_diameters_sorted() {
        let lib = NanopostLibrary::tio2_532nm();
        for i in 1..lib.diameters.len() {
            assert!(lib.diameters[i] > lib.diameters[i - 1]);
        }
    }

    #[test]
    fn nanopost_average_transmittance_in_range() {
        let lib = NanopostLibrary::tio2_532nm();
        let t = lib.average_transmittance();
        assert!(t > 0.0 && t <= 1.0, "T_avg={t:.3}");
    }

    #[test]
    fn nanopost_phase_in_0_2pi() {
        let lib = NanopostLibrary::tio2_532nm();
        for &p in &lib.phase {
            assert!((0.0..=2.0 * PI + 1e-10).contains(&p), "phase={p:.3}");
        }
    }
}
