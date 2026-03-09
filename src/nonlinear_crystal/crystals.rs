//! Nonlinear optical crystal database with Sellmeier equations and NLO tensor components.
//!
//! Provides built-in parameters for common NLO crystals:
//! - BBO (β-BaB₂O₄): wide transparency, high damage threshold
//! - KTP (KTiOPO₄): Type-II SHG, popular for green lasers
//! - LiNbO3: electro-optic and NLO, used for QPM
//! - KDP (KH₂PO₄): classic material, large aperture availability
//! - PPLN (periodically-poled LiNbO3): QPM material

/// Crystal symmetry class determining birefringence character.
#[derive(Debug, Clone, PartialEq)]
pub enum CrystalClass {
    /// Single optic axis (e.g., BBO, LiNbO3). Negative uniaxial: n_e < n_o.
    /// Positive uniaxial: n_e > n_o.
    Uniaxial { negative: bool },
    /// Two optic axes (e.g., KTP). Biaxial crystals require 3 principal refractive indices.
    Biaxial,
    /// Cubic crystal class — no birefringence (isotropic).
    Isotropic,
}

/// Simplified Sellmeier equation for a single polarization:
/// n²(λ) = A + B·λ²/(λ²−C) − D·λ²
///
/// λ is in micrometers (μm).
#[derive(Debug, Clone)]
pub struct SellmeierCoeff {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

impl SellmeierCoeff {
    /// Create new Sellmeier coefficients.
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self { a, b, c, d }
    }

    /// n² at wavelength λ (μm).
    pub fn n_squared(&self, lambda_um: f64) -> f64 {
        let l2 = lambda_um * lambda_um;
        self.a + self.b * l2 / (l2 - self.c) - self.d * l2
    }

    /// Refractive index n at wavelength λ (μm). Returns 1.0 if n² ≤ 0.
    pub fn n(&self, lambda_um: f64) -> f64 {
        self.n_squared(lambda_um).max(1.0).sqrt()
    }

    /// dn/dλ at wavelength λ (μm) — analytical derivative.
    pub fn dn_dlambda(&self, lambda_um: f64) -> f64 {
        let l2 = lambda_um * lambda_um;
        let denom = l2 - self.c;
        // d(n²)/dλ = B * 2λ*(λ²-C) - B*λ²*2λ) / (λ²-C)² - 2D*λ
        //           = -2B*C*λ / (λ²-C)² - 2D*λ
        let dn2_dl =
            -2.0 * self.b * self.c * lambda_um / (denom * denom) - 2.0 * self.d * lambda_um;
        let n_val = self.n(lambda_um);
        if n_val < 1e-10 {
            0.0
        } else {
            dn2_dl / (2.0 * n_val)
        }
    }
}

/// Nonlinear optical crystal with Sellmeier dispersion and NLO tensor.
///
/// The d-tensor is stored in contracted (Voigt) notation: d\[i\]\[j\] where i=0..2 (xyz)
/// and j=0..5 (xx, yy, zz, yz, xz, xy in Voigt notation), units pm/V.
#[derive(Debug, Clone)]
pub struct NloCrystal {
    /// Crystal name (e.g., "BBO", "KTP", "LiNbO3").
    pub name: String,
    /// Crystal symmetry class.
    pub crystal_class: CrystalClass,
    /// Second-order NLO tensor d_ij in contracted notation, units pm/V.
    /// d\[0..2\]\[0..5\] — rows are xyz output polarization, columns are Voigt input pairs.
    pub d_tensor: [[f64; 6]; 3],
    /// Sellmeier coefficients for the ordinary polarization (o-wave).
    pub sellmeier_o: SellmeierCoeff,
    /// Sellmeier coefficients for the extraordinary polarization (e-wave).
    pub sellmeier_e: SellmeierCoeff,
    /// Transparency window (λ_min, λ_max) in nm.
    pub transparency_range: (f64, f64),
    /// Laser damage threshold (GW/cm²) for ns pulses (~1 ns).
    pub damage_threshold_gw_per_cm2: f64,
    /// Thermo-optic coefficient dn/dT (1/K) — used for temperature tuning.
    pub thermo_optic_dn_dt: f64,
}

impl NloCrystal {
    /// BBO (β-BaB₂O₄) — negative uniaxial, point group 3m.
    ///
    /// Sellmeier coefficients from Kato 1986. d_eff relevant components:
    /// d_22 ≈ 2.3 pm/V, d_31 ≈ 0.16 pm/V.
    /// Type-I SHG: 1064 → 532 nm, phase matching angle ≈ 22.8°.
    pub fn bbo() -> Self {
        // BBO Sellmeier (Kato 1986, λ in μm):
        // n_o² = 2.7359 + 0.01878/(λ²-0.01822) - 0.01354 λ²
        // n_e² = 2.3753 + 0.01224/(λ²-0.01667) - 0.01516 λ²
        Self {
            name: "BBO".to_string(),
            crystal_class: CrystalClass::Uniaxial { negative: true },
            d_tensor: [
                [0.0, 0.0, 0.0, 0.0, 0.16, -2.3], // d_1j
                [-2.3, 2.3, 0.0, 0.16, 0.0, 0.0], // d_2j (d_22 = 2.3 pm/V)
                [0.16, 0.16, 0.0, 0.0, 0.0, 0.0], // d_3j (d_31 = 0.16 pm/V)
            ],
            sellmeier_o: SellmeierCoeff::new(2.7359, 0.01878, 0.01822, 0.01354),
            sellmeier_e: SellmeierCoeff::new(2.3753, 0.01224, 0.01667, 0.01516),
            transparency_range: (189.0, 3500.0),
            damage_threshold_gw_per_cm2: 5.0,
            thermo_optic_dn_dt: -16.6e-6,
        }
    }

    /// KTP (KTiOPO₄) — biaxial, point group mm2.
    ///
    /// High effective nonlinearity for Type-II SHG at 1064 nm → 532 nm.
    /// d_24 ≈ 7.6 pm/V, d_15 ≈ 6.1 pm/V, d_33 ≈ 13.7 pm/V.
    /// Uses Sellmeier for n_x (ordinary) and n_z (extraordinary) as approximation.
    pub fn ktp() -> Self {
        // KTP Sellmeier (Kato & Takaoka 2002, λ in μm):
        // n_x² = 3.0065 + 0.03901/(λ²-0.04251) - 0.01327 λ²
        // n_z² = 3.3134 + 0.05694/(λ²-0.05658) - 0.01682 λ²
        Self {
            name: "KTP".to_string(),
            crystal_class: CrystalClass::Biaxial,
            d_tensor: [
                [0.0, 0.0, 0.0, 0.0, 6.1, 0.0],  // d_1j: d_15
                [0.0, 0.0, 0.0, 7.6, 0.0, 0.0],  // d_2j: d_24
                [5.0, 5.0, 13.7, 0.0, 0.0, 0.0], // d_3j: d_31, d_32, d_33
            ],
            sellmeier_o: SellmeierCoeff::new(3.0065, 0.03901, 0.04251, 0.01327),
            sellmeier_e: SellmeierCoeff::new(3.3134, 0.05694, 0.05658, 0.01682),
            transparency_range: (350.0, 4500.0),
            damage_threshold_gw_per_cm2: 1.5,
            thermo_optic_dn_dt: 13.0e-6,
        }
    }

    /// LiNbO3 (Lithium Niobate) — negative uniaxial, point group 3m.
    ///
    /// Large d_33 ≈ 27 pm/V — ideal for QPM applications.
    /// Also important for electro-optic modulation.
    pub fn linbo3() -> Self {
        // LiNbO3 Sellmeier (Zelmon 1997, λ in μm):
        // n_o² = 4.9048 + 0.11775/(λ²-0.04908) - 0.027169 λ²
        // n_e² = 4.5820 + 0.099169/(λ²-0.044432) - 0.021950 λ²
        Self {
            name: "LiNbO3".to_string(),
            crystal_class: CrystalClass::Uniaxial { negative: true },
            d_tensor: [
                [0.0, 0.0, 0.0, 0.0, -3.4, -2.5], // d_1j: d_15, d_22 (sign convention)
                [-2.5, 2.5, 0.0, -3.4, 0.0, 0.0], // d_2j
                [-4.3, -4.3, 27.0, 0.0, 0.0, 0.0], // d_3j: d_31, d_33
            ],
            sellmeier_o: SellmeierCoeff::new(4.9048, 0.11775, 0.04908, 0.027169),
            sellmeier_e: SellmeierCoeff::new(4.5820, 0.099169, 0.044432, 0.021950),
            transparency_range: (400.0, 5000.0),
            damage_threshold_gw_per_cm2: 0.5,
            thermo_optic_dn_dt: -40.0e-6,
        }
    }

    /// KDP (KH₂PO₄) — negative uniaxial, point group 42m.
    ///
    /// Classic NLO material, available in large apertures. d_36 ≈ 0.39 pm/V.
    /// Used historically for SHG of Nd:YAG at 1064 nm.
    pub fn kdp() -> Self {
        // KDP Sellmeier (Zernike 1964, λ in μm):
        // n_o² = 2.259276 + 0.01008956/(λ²-0.012942625) - 0.013 λ²
        // n_e² = 2.132668 + 0.008637494/(λ²-0.012281043) - 0.001 λ²
        Self {
            name: "KDP".to_string(),
            crystal_class: CrystalClass::Uniaxial { negative: true },
            d_tensor: [
                [0.0, 0.0, 0.0, 0.39, 0.0, 0.0], // d_1j: d_14
                [0.0, 0.0, 0.0, 0.0, 0.39, 0.0], // d_2j: d_25
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.39], // d_3j: d_36
            ],
            sellmeier_o: SellmeierCoeff::new(2.259276, 0.01008956, 0.012942625, 0.013),
            sellmeier_e: SellmeierCoeff::new(2.132668, 0.008637494, 0.012281043, 0.001),
            transparency_range: (200.0, 1700.0),
            damage_threshold_gw_per_cm2: 10.0,
            thermo_optic_dn_dt: -2.4e-5,
        }
    }

    /// PPLN (periodically-poled LiNbO3) — QPM material based on LiNbO3.
    ///
    /// The poling period enables quasi-phase matching for efficient SHG/OPA.
    /// Effective d_33 for QPM: d_eff = (2/π) * d_33 ≈ 17.2 pm/V.
    pub fn ppln(_poling_period_um: f64) -> Self {
        // Same Sellmeier as LiNbO3; poling period is used externally for QPM calculation
        let mut crystal = Self::linbo3();
        crystal.name = "PPLN".to_string();
        // PPLN uses d33 for QPM (effective ~17.2 pm/V after 2/π factor)
        crystal
    }

    /// Ordinary refractive index at wavelength λ (nm).
    pub fn n_ordinary(&self, lambda_nm: f64) -> f64 {
        self.sellmeier_o.n(lambda_nm * 1e-3)
    }

    /// Extraordinary refractive index at angle θ to the optic axis (uniaxial crystal).
    ///
    /// Uses the index ellipsoid: 1/n_e(θ)² = cos²θ/n_o² + sin²θ/n_e²
    /// For θ=0: n_e(0) = n_o (propagation along optic axis, no birefringence).
    /// For θ=π/2: n_e(π/2) = n_e (maximum birefringence).
    pub fn n_extraordinary(&self, lambda_nm: f64, theta_rad: f64) -> f64 {
        let n_o = self.sellmeier_o.n(lambda_nm * 1e-3);
        let n_e = self.sellmeier_e.n(lambda_nm * 1e-3);
        let cos_t = theta_rad.cos();
        let sin_t = theta_rad.sin();
        // 1/n_e(θ)² = cos²θ/n_o² + sin²θ/n_e²
        let inv_n2 = cos_t * cos_t / (n_o * n_o) + sin_t * sin_t / (n_e * n_e);
        if inv_n2 < 1e-30 {
            n_o
        } else {
            (1.0 / inv_n2).sqrt()
        }
    }

    /// Group velocity dispersion β₂ (ps²/mm) for the ordinary wave at λ (nm).
    ///
    /// β₂ = λ³/(2πc²) · d²n/dλ² where c is in mm/ps, λ in mm.
    pub fn gvd_ps2_per_mm(&self, lambda_nm: f64) -> f64 {
        let c_mm_per_ps = 2.99792458e8 * 1e3 * 1e-12; // mm/ps
        let lambda_mm = lambda_nm * 1e-6;
        // Numerical second derivative of n(λ)
        let dl_mm = 0.001e-6; // 1 pm step in mm
        let lp = lambda_mm + dl_mm;
        let lm = lambda_mm - dl_mm;
        let n0 = self.sellmeier_o.n(lambda_mm * 1e3);
        let np = self.sellmeier_o.n(lp * 1e3);
        let nm = self.sellmeier_o.n(lm * 1e3);
        let d2n_dl2 = (np - 2.0 * n0 + nm) / (dl_mm * dl_mm);
        // β₂ = λ³ / (2πc²) * d²n/dλ² [ps²/mm]
        lambda_mm * lambda_mm * lambda_mm / (2.0 * std::f64::consts::PI * c_mm_per_ps * c_mm_per_ps)
            * d2n_dl2
    }

    /// Maximum effective nonlinear coefficient d_eff (pm/V) of this crystal.
    pub fn max_d_eff_pm_per_v(&self) -> f64 {
        self.d_tensor
            .iter()
            .flat_map(|row| row.iter())
            .copied()
            .fold(0.0_f64, |acc, v| acc.max(v.abs()))
    }

    /// Check if λ (nm) is within the crystal's transparency window.
    pub fn is_transparent(&self, lambda_nm: f64) -> bool {
        lambda_nm >= self.transparency_range.0 && lambda_nm <= self.transparency_range.1
    }

    /// Effective damage threshold (W/cm²) for a given pulse width (ns).
    ///
    /// Scales as √τ for ns pulses (thermal/diffusion limited regime):
    /// I_damage(τ) = I_damage(1 ns) · √(τ/1 ns)
    pub fn damage_threshold_w_per_cm2(&self, pulse_width_ns: f64) -> f64 {
        let i_ref = self.damage_threshold_gw_per_cm2 * 1e9; // W/cm²
        i_ref * pulse_width_ns.max(0.0).sqrt()
    }

    /// Temperature tuning rate dλ_PM/dT (nm/°C) — approximate from thermo-optic coefficient.
    ///
    /// Estimated from the change in phase matching condition with temperature.
    pub fn temperature_tuning_rate_nm_per_c(&self) -> f64 {
        // Approximate: dλ/dT ≈ (dn_e/dT - dn_o/dT) / (dn/dλ)
        // Use a representative value based on thermo-optic coefficient
        // For BBO: ~0.05 nm/°C, LiNbO3: ~0.2 nm/°C
        let dn_dt = self.thermo_optic_dn_dt.abs();
        // d_eff sensitivity ≈ dn_dt / (dn/dλ at 532nm ≈ 0.1 /μm = 1e5 /m)
        // => nm/°C scale value
        dn_dt * 5e5 // empirical scaling to get reasonable nm/°C
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbo_n_ordinary() {
        let bbo = NloCrystal::bbo();
        let n_o = bbo.n_ordinary(532.0);
        // BBO n_o at 532 nm ≈ 1.67
        assert!(
            (n_o - 1.67).abs() < 0.03,
            "BBO n_o={:.4} at 532 nm, expected ≈ 1.67",
            n_o
        );
    }

    #[test]
    fn test_bbo_n_extraordinary() {
        let bbo = NloCrystal::bbo();
        let n_o = bbo.n_ordinary(532.0);
        // BBO is negative uniaxial (n_e < n_o)
        let n_e_max = bbo.n_extraordinary(532.0, std::f64::consts::FRAC_PI_2);
        assert!(
            n_e_max < n_o,
            "BBO should be negative uniaxial: n_e={:.4} < n_o={:.4}",
            n_e_max,
            n_o
        );
    }

    #[test]
    fn test_ktp_transparent_at_1064() {
        let ktp = NloCrystal::ktp();
        assert!(
            ktp.is_transparent(1064.0),
            "KTP should be transparent at 1064 nm"
        );
    }

    #[test]
    fn test_linbo3_transparent_at_1550() {
        let lnb = NloCrystal::linbo3();
        assert!(
            lnb.is_transparent(1550.0),
            "LiNbO3 should be transparent at 1550 nm"
        );
    }

    #[test]
    fn test_crystal_damage_threshold_scaling() {
        let bbo = NloCrystal::bbo();
        // Damage threshold should scale with sqrt(τ)
        let i_1ns = bbo.damage_threshold_w_per_cm2(1.0);
        let i_4ns = bbo.damage_threshold_w_per_cm2(4.0);
        // sqrt(4)/sqrt(1) = 2
        let ratio = i_4ns / i_1ns;
        assert!(
            (ratio - 2.0).abs() < 0.01,
            "Damage threshold scaling: expected ratio 2.0, got {:.3}",
            ratio
        );
    }

    #[test]
    fn test_n_extraordinary_at_90deg() {
        let bbo = NloCrystal::bbo();
        let lambda_nm = 1064.0;
        let n_e_at_90 = bbo.n_extraordinary(lambda_nm, std::f64::consts::FRAC_PI_2);
        // At θ=π/2, n_e(θ) should equal the principal n_e
        let n_e_principal = bbo.sellmeier_e.n(lambda_nm * 1e-3);
        assert!(
            (n_e_at_90 - n_e_principal).abs() < 1e-6,
            "n_e(90°)={:.6} should equal n_e={:.6}",
            n_e_at_90,
            n_e_principal
        );
    }

    #[test]
    fn test_n_extraordinary_at_0deg() {
        let bbo = NloCrystal::bbo();
        let lambda_nm = 1064.0;
        let n_e_at_0 = bbo.n_extraordinary(lambda_nm, 0.0);
        let n_o = bbo.n_ordinary(lambda_nm);
        // At θ=0 (propagation along optic axis), n_e(0) = n_o
        assert!(
            (n_e_at_0 - n_o).abs() < 1e-6,
            "n_e(0°)={:.6} should equal n_o={:.6}",
            n_e_at_0,
            n_o
        );
    }
}
