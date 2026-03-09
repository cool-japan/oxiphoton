use std::f64::consts::PI;

/// Ring resonator model based on transfer matrix / coupled-resonator equations.
///
/// Geometry: bus waveguide — coupling region — ring (radius R).
///
/// Transfer function (single bus, add-drop geometry):
///
/// Through port: T = (a² - 2ra cos(φ) + r²) / (1 - 2ra cos(φ) + r²a²)
///
/// where:
/// - r = sqrt(1 - κ²): field transmission (amplitude), κ = power coupling coefficient
/// - a = exp(-α L / 2): round-trip field loss (L = 2πR)
/// - φ = k₀ n_eff · 2πR: round-trip phase
#[derive(Debug, Clone)]
pub struct RingResonator {
    /// Ring radius (m).
    pub radius: f64,
    /// Effective refractive index (at center wavelength).
    pub n_eff: f64,
    /// Group refractive index (for FSR calculation).
    pub n_group: f64,
    /// Power coupling coefficient κ² (dimensionless, 0–1).
    pub kappa_sq: f64,
    /// Round-trip power loss coefficient α (1/m).
    pub alpha: f64,
}

impl RingResonator {
    /// Create a ring resonator.
    ///
    /// # Arguments
    /// - `radius`: ring radius (m)
    /// - `n_eff`: effective index at center wavelength
    /// - `n_group`: group index (for FSR)
    /// - `kappa_sq`: power coupling coefficient (0 = no coupling, 1 = full coupling)
    /// - `alpha`: propagation loss coefficient (1/m); 0 = lossless
    pub fn new(radius: f64, n_eff: f64, n_group: f64, kappa_sq: f64, alpha: f64) -> Self {
        Self {
            radius,
            n_eff,
            n_group,
            kappa_sq,
            alpha,
        }
    }

    /// Ring circumference (m).
    pub fn circumference(&self) -> f64 {
        2.0 * PI * self.radius
    }

    /// Free spectral range (m) at center wavelength λ.
    ///
    /// FSR = λ² / (n_g · L)  where L = 2πR.
    pub fn fsr(&self, wavelength: f64) -> f64 {
        wavelength * wavelength / (self.n_group * self.circumference())
    }

    /// Resonance wavelengths nearest to λ.
    ///
    /// Returns the m resonances centered around wavelength, with FSR spacing.
    pub fn resonances(&self, wavelength: f64, n_resonances: usize) -> Vec<f64> {
        let l = self.circumference();
        // Find mode order m such that λ_m = n_eff · L / m
        let m_center = (self.n_eff * l / wavelength).round() as i64;
        let half = (n_resonances as i64) / 2;
        (m_center - half..m_center - half + n_resonances as i64)
            .filter(|&m| m > 0)
            .map(|m| self.n_eff * l / m as f64)
            .collect()
    }

    /// Through-port transmission spectrum.
    ///
    /// Returns transmission T_through(λ) for each wavelength.
    pub fn transmission_through(&self, wavelengths: &[f64]) -> Vec<f64> {
        let l = self.circumference();
        let r = (1.0 - self.kappa_sq).sqrt(); // field transmission
        let a_rt = (-self.alpha * l / 2.0).exp(); // round-trip field attenuation

        wavelengths
            .iter()
            .map(|&wl| {
                let k0 = 2.0 * PI / wl;
                let phi = k0 * self.n_eff * l; // round-trip phase

                let cos_phi = phi.cos();
                let num = a_rt * a_rt - 2.0 * r * a_rt * cos_phi + r * r;
                let den = 1.0 - 2.0 * r * a_rt * cos_phi + r * r * a_rt * a_rt;

                if den.abs() < 1e-30 {
                    0.0
                } else {
                    num / den
                }
            })
            .collect()
    }

    /// Drop-port transmission spectrum (requires second coupling κ_drop).
    ///
    /// For a symmetric add-drop ring (κ_input = κ_drop = kappa_sq):
    /// T_drop = (1-r²)(1-r²)·a² / (1 - 2ra cos(φ) + r²a²)²
    pub fn transmission_drop(&self, wavelengths: &[f64]) -> Vec<f64> {
        let l = self.circumference();
        let r = (1.0 - self.kappa_sq).sqrt();
        let a_rt = (-self.alpha * l / 2.0).exp();
        let kappa_factor = 1.0 - r * r; // = kappa_sq

        wavelengths
            .iter()
            .map(|&wl| {
                let k0 = 2.0 * PI / wl;
                let phi = k0 * self.n_eff * l;
                let cos_phi = phi.cos();

                let den = 1.0 - 2.0 * r * a_rt * cos_phi + r * r * a_rt * a_rt;
                if den.abs() < 1e-30 {
                    return 0.0;
                }
                kappa_factor * kappa_factor * a_rt * a_rt / (den * den)
            })
            .collect()
    }

    /// Quality factor Q at a resonance wavelength.
    ///
    /// Q = λ / δλ where δλ is the resonance linewidth (FWHM).
    pub fn quality_factor(&self, wavelength: f64) -> f64 {
        // Q = ω₀ / Δω = π * sqrt(r * a) / (1 - r * a) * λ_res / FSR
        let l = self.circumference();
        let r = (1.0 - self.kappa_sq).sqrt();
        let a_rt = (-self.alpha * l / 2.0).exp();
        let fsr = self.fsr(wavelength);
        let finesse = PI * (r * a_rt).sqrt() / (1.0 - r * a_rt);
        finesse * wavelength / fsr
    }

    /// Finesse of the resonator.
    pub fn finesse(&self) -> f64 {
        let l = self.circumference();
        let r = (1.0 - self.kappa_sq).sqrt();
        let a_rt = (-self.alpha * l / 2.0).exp();
        PI * (r * a_rt).sqrt() / (1.0 - r * a_rt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_resonator_fsr() {
        // R = 5μm, n_group = 4.2, λ = 1550nm
        // FSR = λ² / (n_g * 2πR) = (1550e-9)² / (4.2 * 2π * 5e-6)
        let ring = RingResonator::new(5e-6, 2.4, 4.2, 0.1, 0.0);
        let fsr = ring.fsr(1550e-9);
        // Expected: ~18.2nm
        let expected = (1550e-9_f64).powi(2) / (4.2 * 2.0 * PI * 5e-6);
        let err_nm = (fsr - expected).abs() * 1e9;
        assert!(
            err_nm < 0.1,
            "FSR error = {err_nm:.4} nm (got {:.4} nm, expected {:.4} nm)",
            fsr * 1e9,
            expected * 1e9
        );
    }

    #[test]
    fn ring_resonator_through_at_resonance() {
        // For a lossless all-pass ring, |T|=1 for all wavelengths.
        // To observe a dip, we need propagation loss (a < 1).
        // With loss alpha=5000/m (≈21.7 dB/cm), a_rt < 1 → dip at resonance.
        let ring = RingResonator::new(5e-6, 2.4, 4.2, 0.1, 5000.0); // lossy ring
        let wl = 1550e-9;

        let reso = ring.resonances(wl, 1);
        let t_at_res = ring.transmission_through(&reso);
        let t_off_res = ring.transmission_through(&[reso[0] + ring.fsr(wl) / 2.0]);

        assert!(
            t_at_res[0] < t_off_res[0],
            "Through port should dip at resonance: T_res={:.4} T_off={:.4}",
            t_at_res[0],
            t_off_res[0]
        );
    }

    #[test]
    fn ring_resonator_fsr_si_waveguide() {
        // Si strip waveguide ring R=5μm: n_eff≈2.4, n_g≈4.2
        let ring = RingResonator::new(5e-6, 2.4, 4.2, 0.05, 100.0); // 100/m loss
        let fsr_nm = ring.fsr(1550e-9) * 1e9;
        // Analytical: (1550e-9)² / (4.2 * 2π * 5e-6) * 1e9 ≈ 18.2 nm
        let analytical_nm = (1550e-9_f64).powi(2) / (4.2 * 2.0 * PI * 5e-6) * 1e9;
        assert!(
            (fsr_nm - analytical_nm).abs() < 0.1,
            "FSR={fsr_nm:.3} nm, analytical={analytical_nm:.3} nm"
        );
    }
}
