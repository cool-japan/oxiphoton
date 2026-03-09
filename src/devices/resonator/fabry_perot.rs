use num_complex::Complex64;
/// Fabry-Pérot resonator model.
///
/// A Fabry-Pérot (FP) cavity consists of two partially reflective mirrors
/// separated by a medium of length L and effective index n_eff.
///
/// Round-trip condition: 2 · β · L = 2π · m  (m integer)
/// Resonance wavelengths: λ_m = 2 · n_eff · L / m
///
/// Key figures of merit:
///   FSR (Free Spectral Range):  FSR = λ² / (2 · n_g · L)
///   Finesse: F = π·√R / (1 - R)   (for equal mirrors R₁ = R₂ = R)
///   Linewidth: Δλ = FSR / F
///   Quality factor: Q = λ / Δλ
use std::f64::consts::PI;

/// Fabry-Pérot resonator model.
#[derive(Debug, Clone, Copy)]
pub struct FabryPerot {
    /// Cavity length (m)
    pub length: f64,
    /// Effective refractive index of the cavity medium
    pub n_eff: f64,
    /// Group index n_g = n_eff - λ · dn_eff/dλ
    pub n_g: f64,
    /// Mirror 1 power reflectivity R₁ ∈ \[0,1\]
    pub r1: f64,
    /// Mirror 2 power reflectivity R₂ ∈ \[0,1\]
    pub r2: f64,
    /// Round-trip internal loss (power) α_rt ∈ \[0,1\]
    pub round_trip_loss: f64,
}

impl FabryPerot {
    pub fn new(length: f64, n_eff: f64, n_g: f64, r1: f64, r2: f64) -> Self {
        Self {
            length,
            n_eff,
            n_g,
            r1,
            r2,
            round_trip_loss: 0.0,
        }
    }

    /// Simple symmetric FP cavity (R₁ = R₂ = R).
    pub fn symmetric(length: f64, n_eff: f64, n_g: f64, r: f64) -> Self {
        Self::new(length, n_eff, n_g, r, r)
    }

    /// Semiconductor Fabry-Pérot laser cavity (cleaved facets, R≈0.32).
    ///
    /// For GaAs/InP: R_facet = ((n-1)/(n+1))² ≈ 0.32 at n≈3.5.
    pub fn semiconductor_laser(length: f64, n_eff: f64, n_g: f64) -> Self {
        let n = n_eff;
        let r = ((n - 1.0) / (n + 1.0)).powi(2);
        Self::symmetric(length, n_eff, n_g, r)
    }

    /// High-finesse Fabry-Pérot etalon (e.g., optical frequency standard).
    pub fn high_finesse_etalon(length: f64, n_eff: f64, r: f64) -> Self {
        Self::symmetric(length, n_eff, n_eff, r)
    }

    /// Free spectral range (m) in wavelength.
    ///   FSR = λ₀² / (2 · n_g · L)
    pub fn free_spectral_range(&self, wavelength: f64) -> f64 {
        wavelength * wavelength / (2.0 * self.n_g * self.length)
    }

    /// Free spectral range in frequency (Hz).
    ///   FSR_f = c / (2 · n_g · L)
    pub fn fsr_frequency(&self) -> f64 {
        2.998e8 / (2.0 * self.n_g * self.length)
    }

    /// Effective mirror reflectivity: R_eff = √(R₁ · R₂ · (1 - α_rt)).
    fn r_eff(&self) -> f64 {
        (self.r1 * self.r2 * (1.0 - self.round_trip_loss)).sqrt()
    }

    /// Cavity finesse: F = π · √R_eff / (1 - R_eff).
    pub fn finesse(&self) -> f64 {
        let r = self.r_eff();
        PI * r.sqrt() / (1.0 - r)
    }

    /// Resonance linewidth (FWHM) in wavelength (m).
    ///   Δλ = FSR / F
    pub fn linewidth_fwhm(&self, wavelength: f64) -> f64 {
        self.free_spectral_range(wavelength) / self.finesse()
    }

    /// Resonance linewidth (FWHM) in frequency (Hz).
    pub fn linewidth_fwhm_hz(&self) -> f64 {
        self.fsr_frequency() / self.finesse()
    }

    /// Quality factor Q = λ / Δλ = F · λ / FSR.
    pub fn quality_factor(&self, wavelength: f64) -> f64 {
        wavelength / self.linewidth_fwhm(wavelength)
    }

    /// Resonance wavelengths near target_wavelength.
    ///
    /// Returns (wavelength, mode_number) for modes within ± n_modes × FSR.
    pub fn resonances_near(&self, target: f64, n_modes: usize) -> Vec<(f64, u64)> {
        // Mode number m at target: m ≈ 2 n_eff L / λ
        let m_center = (2.0 * self.n_eff * self.length / target).round() as i64;
        let fsr = self.free_spectral_range(target);
        let mut modes = Vec::new();
        for dm in -(n_modes as i64)..=(n_modes as i64) {
            let m = m_center + dm;
            if m <= 0 {
                continue;
            }
            let lambda_m = 2.0 * self.n_eff * self.length / m as f64;
            if (lambda_m - target).abs() <= n_modes as f64 * fsr + fsr {
                modes.push((lambda_m, m as u64));
            }
        }
        modes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        modes
    }

    /// Transmission spectrum T(λ) — Airy function.
    ///
    ///   T(λ) = (1-R₁)(1-R₂) / |1 - √(R₁R₂)·exp(i·2βL)|²
    ///
    /// where β = 2π·n_eff/λ.
    pub fn transmission(&self, wavelength: f64) -> f64 {
        let beta = 2.0 * PI * self.n_eff / wavelength;
        let phase = 2.0 * beta * self.length;
        let r1 = self.r1;
        let r2 = self.r2;
        let r12 = (r1 * r2 * (1.0 - self.round_trip_loss)).sqrt();
        let e: Complex64 = Complex64::new(0.0, phase).exp();
        let denom = Complex64::new(1.0, 0.0) - r12 * e;
        let t_max = (1.0 - r1) * (1.0 - r2);
        t_max / denom.norm_sqr()
    }

    /// Reflection spectrum R(λ).
    pub fn reflection(&self, wavelength: f64) -> f64 {
        let beta = 2.0 * PI * self.n_eff / wavelength;
        let phase = 2.0 * beta * self.length;
        let r1 = self.r1;
        let r2 = self.r2;
        let r12 = (r1 * r2 * (1.0 - self.round_trip_loss)).sqrt();
        let e: Complex64 = Complex64::new(0.0, phase).exp();
        // r(λ) = (√R₁ - √R₂·exp(i·2βL)) / (1 - √(R₁R₂)·exp(i·2βL))
        let sqrt_r1: Complex64 = r1.sqrt().into();
        let sqrt_r2: Complex64 = r2.sqrt().into();
        let num = sqrt_r1 - sqrt_r2 * e;
        let den = Complex64::new(1.0, 0.0) - r12 * e;
        (num / den).norm_sqr()
    }

    /// Threshold gain coefficient g_th (m⁻¹) for lasing.
    ///
    ///   g_th = α_int + (1/(2L)) · ln(1/(R₁·R₂))
    pub fn threshold_gain(&self, alpha_internal: f64) -> f64 {
        let mirror_loss = (1.0 / (self.r1 * self.r2)).ln() / (2.0 * self.length);
        alpha_internal + mirror_loss
    }

    /// Effective cavity loss rate (s⁻¹): γ_c = FSR_f / F.
    pub fn cavity_decay_rate(&self) -> f64 {
        self.fsr_frequency() / self.finesse()
    }

    /// Photon lifetime (s): τ_ph = 1/(2π · γ_c).
    pub fn photon_lifetime(&self) -> f64 {
        1.0 / (2.0 * PI * self.cavity_decay_rate())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsr_physical_glass_etalon() {
        // 1mm BK7 etalon (n=1.515, n_g≈1.520): FSR ≈ λ²/(2·n_g·L)
        let fp = FabryPerot::high_finesse_etalon(1e-3, 1.515, 0.99);
        let fsr = fp.free_spectral_range(1550e-9);
        // FSR = 1550²e-18 / (2×1.515×1e-3) ≈ 793 pm
        assert!(fsr > 500e-12 && fsr < 2e-9, "FSR={fsr:.2e}");
    }

    #[test]
    fn finesse_high_for_high_r() {
        let fp = FabryPerot::high_finesse_etalon(1e-3, 1.5, 0.999);
        let f = fp.finesse();
        assert!(f > 1000.0, "Finesse={f:.1}");
    }

    #[test]
    fn finesse_low_for_low_r() {
        let fp = FabryPerot::semiconductor_laser(500e-6, 3.5, 4.0);
        let f = fp.finesse();
        assert!(f > 0.0 && f < 20.0, "Finesse={f:.2}");
    }

    #[test]
    fn transmission_peaks_at_resonance() {
        let fp = FabryPerot::symmetric(500e-6, 3.5, 3.8, 0.32);
        // Find resonance near 1550nm
        let modes = fp.resonances_near(1550e-9, 2);
        assert!(!modes.is_empty());
        let (wl_res, _) = modes[modes.len() / 2];
        let t_res = fp.transmission(wl_res);
        let t_off = fp.transmission(wl_res + fp.free_spectral_range(wl_res) * 0.5);
        assert!(t_res > t_off, "T at resonance should be maximum");
    }

    #[test]
    fn quality_factor_positive() {
        let fp = FabryPerot::high_finesse_etalon(1e-3, 1.5, 0.99);
        let q = fp.quality_factor(1550e-9);
        assert!(q > 100.0, "Q={q:.0}");
    }

    #[test]
    fn threshold_gain_positive() {
        let fp = FabryPerot::semiconductor_laser(500e-6, 3.5, 4.0);
        let g_th = fp.threshold_gain(20.0);
        assert!(g_th > 20.0); // > internal loss
    }

    #[test]
    fn resonances_sorted() {
        let fp = FabryPerot::symmetric(500e-6, 3.5, 3.8, 0.5);
        let modes = fp.resonances_near(1550e-9, 3);
        for i in 1..modes.len() {
            assert!(modes[i].0 >= modes[i - 1].0);
        }
    }

    #[test]
    fn photon_lifetime_positive() {
        let fp = FabryPerot::high_finesse_etalon(1e-3, 1.5, 0.99);
        let tau = fp.photon_lifetime();
        assert!(tau > 0.0);
    }
}
