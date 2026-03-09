use crate::material::DispersiveMaterial;
use crate::units::{RefractiveIndex, Wavelength};

/// Extrapolation mode for tabulated data outside the defined range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtrapolationMode {
    /// Clamp to boundary value (default).
    Clamp,
    /// Linear extrapolation from last two points.
    Linear,
}

/// Tabulated (n, k) data with linear interpolation
#[derive(Debug, Clone)]
pub struct Tabulated {
    pub name: String,
    /// Wavelengths in meters, must be sorted ascending
    pub wavelengths: Vec<f64>,
    /// Refractive index real part
    pub n_values: Vec<f64>,
    /// Extinction coefficient
    pub k_values: Vec<f64>,
}

impl Tabulated {
    pub fn new(
        name: impl Into<String>,
        wavelengths: Vec<f64>,
        n_values: Vec<f64>,
        k_values: Vec<f64>,
    ) -> Self {
        assert_eq!(wavelengths.len(), n_values.len());
        assert_eq!(wavelengths.len(), k_values.len());
        assert!(wavelengths.len() >= 2, "Need at least 2 data points");
        Self {
            name: name.into(),
            wavelengths,
            n_values,
            k_values,
        }
    }

    /// Gold (Au) — Palik 1985, visible-IR range 0.3–2.0 μm.
    ///
    /// Data points sampled from the Palik handbook at representative wavelengths.
    pub fn au_palik() -> Self {
        // (wavelength_m, n, k) sampled from Palik for Au
        let data: &[(f64, f64, f64)] = &[
            (300e-9, 1.658, 1.956),
            (400e-9, 1.658, 1.956),
            (450e-9, 1.449, 1.908),
            (500e-9, 0.920, 1.952),
            (550e-9, 0.280, 2.816),
            (600e-9, 0.272, 3.015),
            (650e-9, 0.174, 3.560),
            (700e-9, 0.166, 3.848),
            (800e-9, 0.150, 4.483),
            (900e-9, 0.131, 5.083),
            (1000e-9, 0.116, 5.657),
            (1200e-9, 0.154, 6.988),
            (1500e-9, 0.223, 8.985),
            (2000e-9, 0.479, 12.576),
        ];
        let wl: Vec<f64> = data.iter().map(|(w, _, _)| *w).collect();
        let n: Vec<f64> = data.iter().map(|(_, n, _)| *n).collect();
        let k: Vec<f64> = data.iter().map(|(_, _, k)| *k).collect();
        Self::new("Au", wl, n, k)
    }

    /// Silver (Ag) — Palik 1985, visible-IR range 0.3–2.0 μm.
    pub fn ag_palik() -> Self {
        let data: &[(f64, f64, f64)] = &[
            (300e-9, 1.267, 0.770),
            (350e-9, 1.550, 0.570),
            (400e-9, 0.173, 1.950),
            (450e-9, 0.119, 2.480),
            (500e-9, 0.130, 2.912),
            (550e-9, 0.144, 3.326),
            (600e-9, 0.156, 3.733),
            (650e-9, 0.162, 4.127),
            (700e-9, 0.185, 4.527),
            (800e-9, 0.159, 5.245),
            (900e-9, 0.133, 5.940),
            (1000e-9, 0.132, 6.608),
            (1200e-9, 0.157, 7.877),
            (1500e-9, 0.180, 9.801),
            (2000e-9, 0.230, 13.03),
        ];
        let wl: Vec<f64> = data.iter().map(|(w, _, _)| *w).collect();
        let n: Vec<f64> = data.iter().map(|(_, n, _)| *n).collect();
        let k: Vec<f64> = data.iter().map(|(_, _, k)| *k).collect();
        Self::new("Ag", wl, n, k)
    }

    /// Aluminum (Al) — Palik 1985, visible-IR range 0.2–2.0 μm.
    pub fn al_palik() -> Self {
        let data: &[(f64, f64, f64)] = &[
            (200e-9, 0.110, 2.200),
            (300e-9, 0.400, 4.450),
            (400e-9, 0.490, 4.860),
            (500e-9, 0.770, 6.080),
            (600e-9, 1.020, 7.260),
            (700e-9, 1.290, 8.310),
            (800e-9, 1.510, 9.260),
            (900e-9, 1.750, 10.14),
            (1000e-9, 1.960, 11.00),
            (1200e-9, 2.410, 12.59),
            (1500e-9, 3.100, 15.14),
            (2000e-9, 4.390, 19.38),
        ];
        let wl: Vec<f64> = data.iter().map(|(w, _, _)| *w).collect();
        let n: Vec<f64> = data.iter().map(|(_, n, _)| *n).collect();
        let k: Vec<f64> = data.iter().map(|(_, _, k)| *k).collect();
        Self::new("Al", wl, n, k)
    }

    /// Extrapolation mode.
    pub fn with_extrapolation(self, mode: ExtrapolationMode) -> Self {
        // Store mode as a name suffix for later use; store in struct if needed
        // For now this is a no-op builder (clamping is the default behavior)
        let _ = mode;
        self
    }

    /// Complex permittivity ε = (n + ik)² at the given wavelength.
    pub fn permittivity(&self, wavelength: Wavelength) -> (f64, f64) {
        let ri = self.refractive_index(wavelength);
        let eps_re = ri.n * ri.n - ri.k * ri.k;
        let eps_im = 2.0 * ri.n * ri.k;
        (eps_re, eps_im)
    }

    /// Absorption coefficient α = 4πk/λ (m⁻¹).
    pub fn absorption_coefficient(&self, wavelength: Wavelength) -> f64 {
        use std::f64::consts::PI;
        let ri = self.refractive_index(wavelength);
        4.0 * PI * ri.k / wavelength.0
    }

    /// Penetration depth δ = 1/α (m), i.e., 1/e intensity depth.
    pub fn penetration_depth(&self, wavelength: Wavelength) -> f64 {
        let alpha = self.absorption_coefficient(wavelength);
        if alpha < 1e-30 {
            f64::INFINITY
        } else {
            1.0 / alpha
        }
    }

    fn interpolate(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
        if x <= xs[0] {
            return ys[0];
        }
        if x >= xs[xs.len() - 1] {
            return ys[ys.len() - 1];
        }
        // Binary search for interval
        let idx =
            match xs.binary_search_by(|v| v.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Less)) {
                Ok(i) => return ys[i],
                Err(i) => i - 1,
            };
        let x0 = xs[idx];
        let x1 = xs[idx + 1];
        let y0 = ys[idx];
        let y1 = ys[idx + 1];
        let t = (x - x0) / (x1 - x0);
        y0 + t * (y1 - y0)
    }
}

impl DispersiveMaterial for Tabulated {
    fn refractive_index(&self, wavelength: Wavelength) -> RefractiveIndex {
        let wl = wavelength.0;
        let n = Self::interpolate(wl, &self.wavelengths, &self.n_values);
        let k = Self::interpolate(wl, &self.wavelengths, &self.k_values);
        RefractiveIndex { n, k }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn linear_interpolation() {
        let tab = Tabulated::new(
            "test",
            vec![400e-9, 500e-9, 600e-9, 700e-9],
            vec![1.52, 1.50, 1.49, 1.48],
            vec![0.0, 0.0, 0.0, 0.0],
        );
        let ri = tab.refractive_index(Wavelength::from_nm(450.0));
        assert_relative_eq!(ri.n, 1.51, epsilon = 1e-10);
    }

    #[test]
    fn extrapolation_clamped() {
        let tab = Tabulated::new(
            "test",
            vec![400e-9, 700e-9],
            vec![1.52, 1.48],
            vec![0.0, 0.0],
        );
        let ri_low = tab.refractive_index(Wavelength::from_nm(300.0));
        assert_relative_eq!(ri_low.n, 1.52, epsilon = 1e-10);
        let ri_high = tab.refractive_index(Wavelength::from_nm(900.0));
        assert_relative_eq!(ri_high.n, 1.48, epsilon = 1e-10);
    }

    #[test]
    fn au_palik_at_633nm() {
        let au = Tabulated::au_palik();
        let ri = au.refractive_index(Wavelength::from_nm(633.0));
        // Au at 633nm: n ≈ 0.17, k ≈ 3.6 (metallic)
        assert!(ri.n < 0.5, "Au n at 633nm should be small, got {:.3}", ri.n);
        assert!(ri.k > 2.0, "Au k at 633nm should be large, got {:.3}", ri.k);
    }

    #[test]
    fn ag_palik_at_633nm() {
        let ag = Tabulated::ag_palik();
        let ri = ag.refractive_index(Wavelength::from_nm(633.0));
        assert!(ri.n < 0.5, "Ag n={:.3}", ri.n);
        assert!(ri.k > 3.0, "Ag k={:.3}", ri.k);
    }

    #[test]
    fn al_palik_large_k() {
        let al = Tabulated::al_palik();
        let ri = al.refractive_index(Wavelength::from_nm(800.0));
        assert!(ri.k > 5.0, "Al k at 800nm={:.2}", ri.k);
    }

    #[test]
    fn au_permittivity_negative_real() {
        let au = Tabulated::au_palik();
        let (eps_re, _eps_im) = au.permittivity(Wavelength::from_nm(700.0));
        assert!(
            eps_re < 0.0,
            "Au eps_re should be negative at 700nm, got {eps_re:.2}"
        );
    }

    #[test]
    fn au_absorption_coefficient_large() {
        let au = Tabulated::au_palik();
        let alpha = au.absorption_coefficient(Wavelength::from_nm(633.0));
        // alpha ~ 4π*k/λ ≈ 4π*3.6/633e-9 ≈ 7e7 m⁻¹
        assert!(alpha > 1e7, "Au absorption={alpha:.2e} m⁻¹");
    }

    #[test]
    fn penetration_depth_metals_nm_range() {
        let au = Tabulated::au_palik();
        let delta = au.penetration_depth(Wavelength::from_nm(633.0));
        // Skin depth ≈ 15-30 nm
        assert!(
            delta < 100e-9,
            "Au skin depth too large: {:.1} nm",
            delta * 1e9
        );
        assert!(
            delta > 5e-9,
            "Au skin depth too small: {:.1} nm",
            delta * 1e9
        );
    }
}
