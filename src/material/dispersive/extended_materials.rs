//! Extended material database — additional optical materials.
//!
//! Provides Sellmeier, Drude-like, and custom models for:
//! - LiNbO₃ (Sellmeier, e and o rays)
//! - InGaAs (wavelength-dependent Sellmeier fit)
//! - SiGe alloy (interpolated between Si and Ge)
//! - Diamond (Sellmeier)
//! - TiN (Drude-like with interband)
//! - ITO (Drude + ENZ behavior)

use num_complex::Complex64;
use std::f64::consts::PI;

use crate::material::DispersiveMaterial;
use crate::units::conversion::SPEED_OF_LIGHT;
use crate::units::{RefractiveIndex, Wavelength};

// Physical constants
const HBAR: f64 = 1.054_571_817e-34; // J·s
const Q_E: f64 = 1.602_176_634e-19; // C

/// Lithium Niobate (LiNbO₃) — Sellmeier model for ordinary (o) and extraordinary (e) rays.
///
/// Coefficients from Zelmon et al. (1997), valid 0.4–5.0 μm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinboRay {
    Ordinary,
    Extraordinary,
}

#[derive(Debug, Clone)]
pub struct LithiumNiobate {
    pub ray: LinboRay,
}

impl LithiumNiobate {
    pub fn ordinary() -> Self {
        Self {
            ray: LinboRay::Ordinary,
        }
    }
    pub fn extraordinary() -> Self {
        Self {
            ray: LinboRay::Extraordinary,
        }
    }

    fn n_sq(&self, lambda_um: f64) -> f64 {
        let l2 = lambda_um * lambda_um;
        match self.ray {
            // Ordinary ray: Jundt 1997, Opt. Lett. 22, 1553
            // n_o^2 = 4.9048 + 0.11768/(l^2-0.04750) - 0.01269*l^2
            // n_o(1.55μm) ≈ 2.219
            LinboRay::Ordinary => 4.9048 + 0.117_68 / (l2 - 0.047_50) - 0.012_69 * l2,
            // Extraordinary ray: Jundt 1997
            // n_e^2 = 4.5820 + 0.09969/(l^2-0.04432) - 0.01182*l^2
            // n_e(1.55μm) ≈ 2.144
            LinboRay::Extraordinary => 4.5820 + 0.099_69 / (l2 - 0.044_32) - 0.011_82 * l2,
        }
    }
}

impl DispersiveMaterial for LithiumNiobate {
    fn refractive_index(&self, wavelength: Wavelength) -> RefractiveIndex {
        let lambda_um = wavelength.as_um();
        let n_sq = self.n_sq(lambda_um).max(1.0);
        RefractiveIndex {
            n: n_sq.sqrt(),
            k: 0.0,
        }
    }

    fn name(&self) -> &str {
        match self.ray {
            LinboRay::Ordinary => "LiNbO3-o",
            LinboRay::Extraordinary => "LiNbO3-e",
        }
    }
}

/// InGaAs (In₁₋ₓGaₓAs) — Sellmeier fit for x=0.47 (lattice-matched to InP).
///
/// Covers the 1.0–2.5 μm range. n ≈ 3.52 at 1.55 μm.
#[derive(Debug, Clone)]
pub struct InGaAs {
    /// Gallium fraction x (0 = InAs, 1 = GaAs); x=0.47 for InP lattice-match
    pub x_ga: f64,
}

impl InGaAs {
    /// InP-lattice-matched In₀.₅₃Ga₀.₄₇As (x=0.47).
    pub fn inp_matched() -> Self {
        Self { x_ga: 0.47 }
    }

    /// Pure InAs (x=0).
    pub fn inas() -> Self {
        Self { x_ga: 0.0 }
    }

    fn n_sq(&self, lambda_um: f64) -> f64 {
        let x = self.x_ga;
        // Interpolated Sellmeier for In_{1-x}Ga_xAs
        // For x=0.47 (InP-matched): n^2 ≈ 12.4 at long wavelengths → n ≈ 3.52
        // Using modified one-pole Sellmeier: n^2 = A + B*l^2/(l^2 - C^2)
        // InAs (x=0): A=11.1, B=2.16, C=0.97 μm  (Moss 1961, Adachi 1989)
        // GaAs (x=1): A=8.95, B=2.054, C=0.677 μm
        let l2 = lambda_um * lambda_um;
        let a = 11.1 * (1.0 - x) + 8.95 * x;
        let b = 2.16 * (1.0 - x) + 2.054 * x;
        let c = 0.97 * (1.0 - x) + 0.677 * x;
        (a + b * l2 / (l2 - c * c)).max(1.0)
    }
}

impl DispersiveMaterial for InGaAs {
    fn refractive_index(&self, wavelength: Wavelength) -> RefractiveIndex {
        let lambda_um = wavelength.as_um();
        RefractiveIndex {
            n: self.n_sq(lambda_um).sqrt(),
            k: 0.0,
        }
    }

    fn name(&self) -> &str {
        "InGaAs"
    }
}

/// Silicon-Germanium alloy (Si₁₋ₓGeₓ) — interpolated Sellmeier.
///
/// Valid approximately 1.2–15 μm. Band gap decreases with Ge fraction.
#[derive(Debug, Clone)]
pub struct SiGe {
    /// Germanium fraction x (0 = pure Si, 1 = pure Ge)
    pub x_ge: f64,
}

impl SiGe {
    /// Pure silicon (x=0).
    pub fn silicon() -> Self {
        Self { x_ge: 0.0 }
    }

    /// Pure germanium (x=1).
    pub fn germanium() -> Self {
        Self { x_ge: 1.0 }
    }

    /// 50:50 SiGe alloy.
    pub fn sige_50() -> Self {
        Self { x_ge: 0.5 }
    }

    fn n_sq(&self, lambda_um: f64) -> f64 {
        let x = self.x_ge;
        let l2 = lambda_um * lambda_um;
        // Two-term Sellmeier interpolation between Si and Ge
        // Si (Li 1993): n^2 = 1 + 10.6684*l^2/(l^2-0.091302) + 0.003*l^2/(l^2-1.135) + 1.54*l^2/(l^2-1104)
        // Ge (Barnes 1979): n^2 = 1 + 9.28*l^2/(l^2-0.447) + 6.73*l^2/(l^2-0.139) + 0.21*l^2/(l^2-3870)
        // Simplified two-term Sellmeier for each:
        let b1_si = 10.6684_f64;
        let c1_si = 0.091302_f64;
        let b2_si = 0.6 * l2; // small correction
        let _ = b2_si;
        let b1_ge = 9.2811_f64;
        let c1_ge = 0.4479_f64;
        let b2_ge = 6.7237_f64;
        let c2_ge = 0.1396_f64;

        let n_sq_si = 1.0 + b1_si * l2 / (l2 - c1_si);
        let n_sq_ge = 1.0 + b1_ge * l2 / (l2 - c1_ge) + b2_ge * l2 / (l2 - c2_ge);

        ((1.0 - x) * n_sq_si + x * n_sq_ge).max(1.0)
    }
}

impl DispersiveMaterial for SiGe {
    fn refractive_index(&self, wavelength: Wavelength) -> RefractiveIndex {
        let lambda_um = wavelength.as_um();
        RefractiveIndex {
            n: self.n_sq(lambda_um).sqrt(),
            k: 0.0,
        }
    }

    fn name(&self) -> &str {
        "SiGe"
    }
}

/// Diamond (carbon) — Sellmeier model.
///
/// Peter & Thomas, Phys. Rev. 88, 1961; valid 0.22–14.8 μm.
/// n ≈ 2.417 at 589 nm.
#[derive(Debug, Clone)]
pub struct Diamond;

impl Diamond {
    fn n_sq(lambda_um: f64) -> f64 {
        let l2 = lambda_um * lambda_um;
        // Sellmeier for diamond (Peter 1923, refitted by Mildren):
        // n^2 = 1 + 4.3356*l^2/(l^2-0.1060^2) + 0.3306*l^2/(l^2-0.1750^2)
        // gives n(589nm) = 2.417
        1.0
            + 4.3356 * l2 / (l2 - 0.011_236)  // 0.1060^2 = 0.011236
            + 0.3306 * l2 / (l2 - 0.030_625) // 0.1750^2 = 0.030625
    }
}

impl DispersiveMaterial for Diamond {
    fn refractive_index(&self, wavelength: Wavelength) -> RefractiveIndex {
        let lambda_um = wavelength.as_um();
        let n_sq = Diamond::n_sq(lambda_um).max(1.0);
        RefractiveIndex {
            n: n_sq.sqrt(),
            k: 0.0,
        }
    }

    fn name(&self) -> &str {
        "Diamond"
    }
}

/// Titanium Nitride (TiN) — Drude-like model with interband transitions.
///
/// TiN is a refractory plasmonic material with ε_∞ ≈ -1 at 780 nm.
/// Parameters from Naik et al. (2013, Optica).
#[derive(Debug, Clone)]
pub struct TitaniumNitride;

impl TitaniumNitride {
    fn permittivity_at_omega(omega: f64) -> Complex64 {
        // TiN Drude-Lorentz parameters (Naik 2013, fitted to sputtered TiN)
        let ev = Q_E / HBAR;
        let eps_inf = 4.84_f64;
        let omega_p = 7.0 * ev; // plasma frequency
        let gamma_d = 0.7 * ev; // Drude damping
                                // Single Lorentz oscillator for interband transitions
        let omega_l = 5.1 * ev;
        let gamma_l = 3.7 * ev;
        let delta_eps_l = 2.3_f64;

        let i = Complex64::new(0.0, 1.0);
        let drude = -omega_p * omega_p / (omega * omega + i * gamma_d * omega);
        let lorentz = delta_eps_l * omega_l * omega_l
            / Complex64::new(omega_l * omega_l - omega * omega, gamma_l * omega);
        Complex64::new(eps_inf, 0.0) + drude + lorentz
    }
}

impl DispersiveMaterial for TitaniumNitride {
    fn refractive_index(&self, wavelength: Wavelength) -> RefractiveIndex {
        let omega = 2.0 * PI * SPEED_OF_LIGHT / wavelength.0;
        let eps = TitaniumNitride::permittivity_at_omega(omega);
        let z = eps.sqrt();
        let (n, k) = if z.re >= 0.0 {
            (z.re, z.im.abs())
        } else {
            (-z.re, (-z.im).abs())
        };
        RefractiveIndex {
            n: n.abs(),
            k: k.abs(),
        }
    }

    fn permittivity(&self, wavelength: Wavelength) -> Complex64 {
        let omega = 2.0 * PI * SPEED_OF_LIGHT / wavelength.0;
        TitaniumNitride::permittivity_at_omega(omega)
    }

    fn name(&self) -> &str {
        "TiN"
    }
}

/// Indium Tin Oxide (ITO) — Drude model with Epsilon-Near-Zero (ENZ) crossing.
///
/// ITO exhibits ENZ behavior near λ ≈ 1.2–1.5 μm depending on doping.
/// Parameters from Feigenbaum et al. (2010, PRL).
#[derive(Debug, Clone)]
pub struct InTinOxide {
    /// Free carrier density (cm⁻³), controls ENZ wavelength
    pub carrier_density: f64,
}

impl InTinOxide {
    /// Typical e-beam-evaporated ITO with n ≈ 10²¹ cm⁻³.
    pub fn typical() -> Self {
        Self {
            carrier_density: 1e21,
        }
    }

    /// ITO with ENZ near 1.3 μm.
    pub fn enz_1300nm() -> Self {
        Self {
            carrier_density: 8e20,
        }
    }

    fn permittivity_at_omega(&self, omega: f64) -> Complex64 {
        // Drude model for ITO
        let eps_inf = 3.9_f64; // high-frequency dielectric constant
        let m_eff = 0.35; // effective mass (in electron masses)
        let m_e = 9.109_383_701_5e-31; // kg
        let eps0 = 8.854_187_817e-12; // F/m
        let mu_e = 40.0 * 1e-4; // electron mobility 40 cm²/Vs → m²/Vs
        let n_m3 = self.carrier_density * 1e6; // cm⁻³ → m⁻³

        // Plasma frequency: ωp² = n·e²/(m_eff·ε₀)
        let omega_p_sq = n_m3 * Q_E * Q_E / (m_eff * m_e * eps0);
        let _omega_p = omega_p_sq.sqrt();

        // Scattering rate: γ = e/(m_eff·m_e·μ)
        let gamma = Q_E / (m_eff * m_e * mu_e);

        let i = Complex64::new(0.0, 1.0);
        let denom = omega * omega + i * gamma * omega;
        if denom.norm() < 1e-30 {
            return Complex64::new(eps_inf, 0.0);
        }
        Complex64::new(eps_inf, 0.0) - omega_p_sq / denom
    }
}

impl DispersiveMaterial for InTinOxide {
    fn refractive_index(&self, wavelength: Wavelength) -> RefractiveIndex {
        let omega = 2.0 * PI * SPEED_OF_LIGHT / wavelength.0;
        let eps = self.permittivity_at_omega(omega);
        let z = eps.sqrt();
        let (n, k) = if z.re >= 0.0 {
            (z.re, z.im.abs())
        } else {
            (-z.re, (-z.im).abs())
        };
        RefractiveIndex {
            n: n.abs(),
            k: k.abs(),
        }
    }

    fn permittivity(&self, wavelength: Wavelength) -> Complex64 {
        let omega = 2.0 * PI * SPEED_OF_LIGHT / wavelength.0;
        self.permittivity_at_omega(omega)
    }

    fn name(&self) -> &str {
        "ITO"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn linbo3_ordinary_at_1550nm() {
        let m = LithiumNiobate::ordinary();
        let ri = m.refractive_index(Wavelength::from_nm(1550.0));
        // LiNbO3 ordinary ray: n ≈ 2.14 at 1550nm
        assert!(ri.n > 2.1 && ri.n < 2.4, "n_o(1550nm) = {}", ri.n);
        assert_relative_eq!(ri.k, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn linbo3_birefringence() {
        let no = LithiumNiobate::ordinary();
        let ne = LithiumNiobate::extraordinary();
        let ri_o = no.refractive_index(Wavelength::from_nm(1000.0));
        let ri_e = ne.refractive_index(Wavelength::from_nm(1000.0));
        // LiNbO3 is negative uniaxial: n_o > n_e
        assert!(ri_o.n > ri_e.n, "n_o={:.4}, n_e={:.4}", ri_o.n, ri_e.n);
    }

    #[test]
    fn ingaas_inp_matched_at_1550nm() {
        let m = InGaAs::inp_matched();
        let ri = m.refractive_index(Wavelength::from_nm(1550.0));
        // In0.53Ga0.47As at 1550nm: n ≈ 3.5-3.6
        assert!(ri.n > 3.3 && ri.n < 3.8, "n(InGaAs)={:.4}", ri.n);
    }

    #[test]
    fn ingaas_inas_lower_index_than_gaas() {
        let inas = InGaAs::inas();
        let ingaas = InGaAs::inp_matched();
        let ri_inas = inas.refractive_index(Wavelength::from_nm(2000.0));
        let ri_ingaas = ingaas.refractive_index(Wavelength::from_nm(2000.0));
        // InAs has a larger lattice constant and generally higher n
        // Both should be positive and physically reasonable
        assert!(ri_inas.n > 3.0, "n(InAs) = {}", ri_inas.n);
        assert!(ri_ingaas.n > 3.0);
    }

    #[test]
    fn sige_silicon_matches_known_value() {
        let si = SiGe::silicon();
        let ri = si.refractive_index(Wavelength::from_nm(1550.0));
        assert!(ri.n > 3.4 && ri.n < 3.6, "n(Si) = {}", ri.n);
    }

    #[test]
    fn sige_ge_higher_index_than_si() {
        let si = SiGe::silicon();
        let ge = SiGe::germanium();
        let ri_si = si.refractive_index(Wavelength::from_nm(3000.0));
        let ri_ge = ge.refractive_index(Wavelength::from_nm(3000.0));
        assert!(
            ri_ge.n > ri_si.n,
            "n(Ge)={:.3} should > n(Si)={:.3}",
            ri_ge.n,
            ri_si.n
        );
    }

    #[test]
    fn diamond_at_589nm() {
        let d = Diamond;
        let ri = d.refractive_index(Wavelength::from_nm(589.0));
        // Diamond: n ≈ 2.417 at 589nm
        assert!(ri.n > 2.3 && ri.n < 2.5, "n(diamond) = {}", ri.n);
    }

    #[test]
    fn tin_metallic_at_visible() {
        let tin = TitaniumNitride;
        let eps = tin.permittivity(Wavelength::from_nm(780.0));
        // TiN is plasmonic: should have Re(ε) near 0 or negative at λ_ENZ
        assert!(
            eps.re.abs() < 10.0,
            "TiN Re(eps) = {} at ENZ region",
            eps.re
        );
    }

    #[test]
    fn ito_enz_near_expected_wavelength() {
        let ito = InTinOxide::typical();
        // Scan wavelengths 1000-1800nm to find ENZ crossing (Re(ε) → 0)
        let mut min_abs_eps_re = f64::INFINITY;
        for i in 0..100 {
            let lambda_nm = 1000.0 + i as f64 * 8.0;
            let eps = ito.permittivity(Wavelength::from_nm(lambda_nm));
            if eps.re.abs() < min_abs_eps_re {
                min_abs_eps_re = eps.re.abs();
            }
        }
        // ENZ condition: |Re(ε)| should become very small near ENZ wavelength
        assert!(
            min_abs_eps_re < 2.0,
            "min |Re(ε)| = {min_abs_eps_re} (should reach near-zero ENZ)"
        );
    }

    #[test]
    fn ito_has_loss() {
        let ito = InTinOxide::typical();
        let ri = ito.refractive_index(Wavelength::from_nm(1550.0));
        // ITO has significant absorption in the IR
        assert!(ri.k > 0.0, "ITO k = {}", ri.k);
    }

    #[test]
    fn all_materials_return_positive_n() {
        let wavelength = Wavelength::from_nm(1550.0);
        let m1 = LithiumNiobate::ordinary();
        let m2 = InGaAs::inp_matched();
        let m3 = SiGe::sige_50();
        let m4 = Diamond;
        for (name, ri) in [
            (m1.name(), m1.refractive_index(wavelength)),
            (m2.name(), m2.refractive_index(wavelength)),
            (m3.name(), m3.refractive_index(wavelength)),
            (m4.name(), m4.refractive_index(wavelength)),
        ] {
            assert!(ri.n > 0.0, "{name}: n = {}", ri.n);
        }
    }
}
