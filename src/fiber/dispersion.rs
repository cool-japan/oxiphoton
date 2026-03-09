use num_complex::Complex64;
/// Fiber dispersion models and dispersion-management structures.
///
/// Implements the group-velocity dispersion (GVD) operator used in
/// split-step Fourier propagation, standard fibre presets (SMF-28, DSF, HNLF),
/// and a `DispersionMap` for multi-span dispersion-managed transmission links.
///
/// # Unit conventions
///
/// | Quantity | Symbol | SI unit stored internally |
/// |----------|--------|--------------------------|
/// | GVD      | β₂     | s²/m                     |
/// | TOD      | β₃     | s³/m                     |
/// | FOD      | β₄     | s⁴/m                     |
/// | Dispersion parameter | D | ps/(nm·km)        |
/// | Dispersion slope | S  | ps/(nm²·km)           |
///
/// The user-facing fields are in *engineering* units (ps²/km, ps³/km, ps⁴/km)
/// to match standard datasheet values.
use std::f64::consts::PI;

const C0: f64 = 2.997_924_58e8; // m/s

// ---------------------------------------------------------------------------
// FiberDispersion
// ---------------------------------------------------------------------------

/// Dispersion parameters of a single-mode optical fibre.
///
/// All public fields use engineering units (ps²/km, ps³/km, ps⁴/km) for
/// convenience; conversion to SI (s²/m) is performed internally.
#[derive(Debug, Clone, PartialEq)]
pub struct FiberDispersion {
    /// Group-velocity dispersion β₂ (ps²/km).
    /// Negative ↔ anomalous dispersion (signal travels faster at shorter λ).
    pub beta2_ps2_per_km: f64,
    /// Third-order dispersion β₃ (ps³/km).
    pub beta3_ps3_per_km: f64,
    /// Fourth-order dispersion β₄ (ps⁴/km).
    pub beta4_ps4_per_km: f64,
    /// Reference wavelength λ₀ (nm) at which the parameters are specified.
    pub center_wavelength_nm: f64,
}

impl FiberDispersion {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Create from explicit dispersion coefficients.
    pub fn new(
        beta2_ps2_per_km: f64,
        beta3_ps3_per_km: f64,
        beta4_ps4_per_km: f64,
        center_wavelength_nm: f64,
    ) -> Self {
        Self {
            beta2_ps2_per_km,
            beta3_ps3_per_km,
            beta4_ps4_per_km,
            center_wavelength_nm,
        }
    }

    /// Corning SMF-28 standard single-mode fibre at 1550 nm.
    ///
    /// Reference: Agrawal, "Nonlinear Fiber Optics", 6th ed., Appendix B.
    ///   β₂ = −21.7 ps²/km, β₃ = 0.12 ps³/km, β₄ ≈ 0 ps⁴/km.
    pub fn smf28() -> Self {
        Self {
            beta2_ps2_per_km: -21.7,
            beta3_ps3_per_km: 0.12,
            beta4_ps4_per_km: 0.0,
            center_wavelength_nm: 1550.0,
        }
    }

    /// Dispersion-shifted fibre (DSF / ITU-T G.653) at 1550 nm.
    ///
    /// Near-zero GVD: β₂ ≈ 0, small TOD.
    pub fn dsf() -> Self {
        Self {
            beta2_ps2_per_km: 0.0,
            beta3_ps3_per_km: 0.06,
            beta4_ps4_per_km: 0.0,
            center_wavelength_nm: 1550.0,
        }
    }

    /// Highly nonlinear fibre (HNLF) at 1550 nm.
    ///
    /// Small anomalous GVD for broadband nonlinear interactions.
    pub fn hnlf() -> Self {
        Self {
            beta2_ps2_per_km: -1.0,
            beta3_ps3_per_km: 0.03,
            beta4_ps4_per_km: 0.0,
            center_wavelength_nm: 1550.0,
        }
    }

    // -----------------------------------------------------------------------
    // Unit conversion helpers
    // -----------------------------------------------------------------------

    /// β₂ in SI units (s²/m).
    pub fn beta2_s2_per_m(&self) -> f64 {
        // ps²/km = 10⁻²⁴ s²/m * (10⁻¹² / 10³) = 10⁻²⁴ / 10³ → 1e-27 s²/m
        self.beta2_ps2_per_km * 1.0e-27
    }

    /// β₃ in SI units (s³/m).
    pub fn beta3_s3_per_m(&self) -> f64 {
        // ps³/km = (10⁻¹²)³ / 10³ s³/m = 10⁻³⁹ s³/m
        self.beta3_ps3_per_km * 1.0e-39
    }

    /// β₄ in SI units (s⁴/m).
    pub fn beta4_s4_per_m(&self) -> f64 {
        // ps⁴/km = 10⁻⁵¹ s⁴/m
        self.beta4_ps4_per_km * 1.0e-51
    }

    // -----------------------------------------------------------------------
    // Engineering quantities
    // -----------------------------------------------------------------------

    /// Chromatic dispersion parameter D = −(2πc/λ²) β₂ (ps/nm/km).
    ///
    /// Positive D ↔ anomalous dispersion in the conventional telecom convention.
    pub fn dispersion_ps_per_nm_km(&self) -> f64 {
        let lambda_m = self.center_wavelength_nm * 1.0e-9;
        let beta2_si = self.beta2_s2_per_m();
        // D = -(2πc/λ²) β₂   [s/m²] → convert to ps/(nm·km)
        // 1 s/m² = 10¹² ps / (10⁹ nm · 10⁻³ km)   wait, careful:
        //   D [ps/(nm·km)] = D [s/m²] * (1e12 ps/s) / (1e9 nm/m) * 1e3 m/km
        //                  = D [s/m²] * 1e12 / 1e9 * 1e3 = D * 1e6
        // But it's cleaner to work in SI then convert:
        //   D [s/m²] = -(2πc/λ²) β₂ [s/m²]  where β₂ is in s²/m
        let d_si = -(2.0 * PI * C0 / (lambda_m * lambda_m)) * beta2_si; // s/m²
                                                                        // 1 s/m² = 1e6 ps/(nm·km)
        d_si * 1.0e6
    }

    /// Dispersion slope S = dD/dλ (ps/nm²/km).
    ///
    /// Derived from β₃ using S = (2πc/λ²)² β₃ + (4πc/λ³) β₂.
    pub fn slope_ps_per_nm2_km(&self) -> f64 {
        let lambda_m = self.center_wavelength_nm * 1.0e-9;
        let beta2_si = self.beta2_s2_per_m();
        let beta3_si = self.beta3_s3_per_m();
        // dD/dλ  (in s/m³):
        // D = -(2πc/λ²)·β₂  ⟹  dD/dλ = (4πc/λ³)·β₂ - (2πc/λ²)·(dβ₂/dλ)
        // dβ₂/dλ = -(2πc/λ²)·β₃   (chain rule with ω=2πc/λ, dω/dλ=-2πc/λ²)
        let s_si = (4.0 * PI * C0 / lambda_m.powi(3)) * beta2_si
            + (2.0 * PI * C0 / lambda_m.powi(2)).powi(2) * beta3_si;
        // 1 s/m³ = 1e3 ps/(nm²·km)
        s_si * 1.0e3
    }

    /// Zero-dispersion wavelength (ZDW) in nm, estimated from a linear model
    /// β₂(λ) ≈ β₂₀ + S · (λ − λ₀) where S = dβ₂/dλ.
    ///
    /// Returns `None` if the slope is negligible (i.e. dispersion-flat fibre).
    pub fn zdw_nm(&self) -> Option<f64> {
        // Linear model: β₂(λ) = β₂₀ + (dβ₂/dλ)(λ − λ₀)
        // ZDW: λ_ZDW = λ₀ − β₂₀ / (dβ₂/dλ)
        // dβ₂/dλ from β₃: dβ₂/dλ = -(2πc/λ₀²)·β₃  (first-order)
        let lambda0_m = self.center_wavelength_nm * 1.0e-9;
        let dbeta2_dlambda = -(2.0 * PI * C0 / (lambda0_m * lambda0_m)) * self.beta3_s3_per_m();
        if dbeta2_dlambda.abs() < 1.0e-60 {
            return None;
        }
        let delta_lambda_m = -self.beta2_s2_per_m() / dbeta2_dlambda;
        let zdw_m = lambda0_m + delta_lambda_m;
        if zdw_m > 0.0 && zdw_m.is_finite() {
            Some(zdw_m * 1.0e9) // m → nm
        } else {
            None
        }
    }

    /// Dispersion operator phase accumulated over a step `dz_m` (m):
    ///
    ///   D̂(ω) = exp[i(β₂/2·ω² + β₃/6·ω³ + β₄/24·ω⁴)·dz]
    ///
    /// The result is a complex multiplier array of length `omega.len()`.
    pub fn dispersion_operator(&self, omega: &[f64], dz_m: f64) -> Vec<Complex64> {
        let b2 = self.beta2_s2_per_m();
        let b3 = self.beta3_s3_per_m();
        let b4 = self.beta4_s4_per_m();
        omega
            .iter()
            .map(|&w| {
                let phase =
                    (b2 / 2.0 * w * w + b3 / 6.0 * w * w * w + b4 / 24.0 * w.powi(4)) * dz_m;
                Complex64::new(0.0, phase).exp()
            })
            .collect()
    }

    /// Dispersion length L_D = T₀² / |β₂| (m).
    ///
    /// `fwhm_ps` is the 1/e half-width T₀ = FWHM / (2√(ln 2)) of a Gaussian
    /// pulse.  Returns infinity for zero-dispersion fibre.
    pub fn gvd_length_m(&self, fwhm_ps: f64) -> f64 {
        let b2_abs = self.beta2_s2_per_m().abs();
        if b2_abs < 1.0e-60 {
            return f64::INFINITY;
        }
        let t0_s = fwhm_ps * 1.0e-12 / (2.0 * 2.0_f64.ln().sqrt());
        t0_s * t0_s / b2_abs
    }

    /// Pulse broadening factor for a Gaussian pulse after propagation
    /// distance `z_km` (km):
    ///
    ///   B(z) = [1 + (z/L_D)²]^(1/2)
    ///
    /// where L_D = T₀²/|β₂|.  Returns 1.0 for a non-dispersive fibre.
    pub fn broadening_factor(&self, z_km: f64, fwhm_ps: f64) -> f64 {
        let ld_m = self.gvd_length_m(fwhm_ps);
        if ld_m.is_infinite() || ld_m < 1.0e-30 {
            return 1.0;
        }
        let z_m = z_km * 1.0e3;
        let ratio = z_m / ld_m;
        (1.0 + ratio * ratio).sqrt()
    }
}

// ---------------------------------------------------------------------------
// DispersionMap
// ---------------------------------------------------------------------------

/// Multi-span dispersion map for dispersion-managed optical transmission.
///
/// Each span is a `(FiberDispersion, length_km)` pair.  The map accumulates
/// the net dispersion over all spans and provides compensation diagnostics.
#[derive(Debug, Clone, Default)]
pub struct DispersionMap {
    /// Ordered list of (fibre dispersion, span length in km) pairs.
    pub spans: Vec<(FiberDispersion, f64)>,
}

impl DispersionMap {
    /// Create an empty dispersion map.
    pub fn new() -> Self {
        Self { spans: Vec::new() }
    }

    /// Append a span with the given dispersion parameters and length.
    pub fn add_span(&mut self, dispersion: FiberDispersion, length_km: f64) {
        self.spans.push((dispersion, length_km));
    }

    /// Net accumulated dispersion Σᵢ β₂ᵢ · Lᵢ (ps²), using engineering units.
    pub fn net_dispersion_ps2(&self) -> f64 {
        self.spans
            .iter()
            .map(|(disp, length)| disp.beta2_ps2_per_km * length)
            .sum()
    }

    /// Average dispersion D̄ = Σ(β₂ᵢ Lᵢ) / Σ Lᵢ (ps²/km).
    ///
    /// Returns 0 if the total length is zero.
    pub fn average_dispersion_ps2_per_km(&self) -> f64 {
        let total_l = self.total_length_km();
        if total_l < 1.0e-30 {
            return 0.0;
        }
        self.net_dispersion_ps2() / total_l
    }

    /// Total fibre length Σ Lᵢ (km).
    pub fn total_length_km(&self) -> f64 {
        self.spans.iter().map(|(_, l)| l).sum()
    }

    /// Check whether the map is dispersion-compensated: |net| < `threshold_ps2`.
    pub fn is_compensated(&self, threshold_ps2: f64) -> bool {
        self.net_dispersion_ps2().abs() < threshold_ps2
    }

    /// Accumulated dispersion map: vector of (position_km, net_dispersion_ps2)
    /// at each span boundary.  Useful for visualisation.
    pub fn accumulated_dispersion_profile(&self) -> Vec<(f64, f64)> {
        let mut position = 0.0_f64;
        let mut net = 0.0_f64;
        let mut profile = vec![(0.0, 0.0)];
        for (disp, length) in &self.spans {
            position += length;
            net += disp.beta2_ps2_per_km * length;
            profile.push((position, net));
        }
        profile
    }

    /// Maximum excursion of the accumulated dispersion from zero (ps²).
    pub fn peak_excursion_ps2(&self) -> f64 {
        self.accumulated_dispersion_profile()
            .iter()
            .map(|(_, d)| d.abs())
            .fold(0.0_f64, f64::max)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_smf28_dispersion() {
        // SMF-28 at 1550 nm: D ≈ −17 ps/(nm·km) in Agrawal convention
        // (anomalous: D > 0 per the -(2πc/λ²)β₂ formula with β₂ < 0)
        let fiber = FiberDispersion::smf28();
        let d = fiber.dispersion_ps_per_nm_km();
        // β₂ = -21.7 ps²/km: D = -(2πc/λ²)·β₂ > 0
        assert!(
            d > 0.0,
            "SMF-28 anomalous dispersion should give D > 0, got {d}"
        );
        // Magnitude should be in the 15–20 ps/(nm·km) range
        assert!(
            d > 14.0 && d < 22.0,
            "SMF-28 D expected ~17 ps/(nm·km), got {d:.3}"
        );
    }

    #[test]
    fn test_dispersion_operator_shape() {
        let fiber = FiberDispersion::smf28();
        let n = 256_usize;
        let omega: Vec<f64> = (0..n)
            .map(|i| (i as f64 - n as f64 / 2.0) * 1.0e12)
            .collect();
        let op = fiber.dispersion_operator(&omega, 100.0);
        assert_eq!(
            op.len(),
            n,
            "Dispersion operator output length must match input omega length"
        );
        // Each element must be a unit-magnitude complex number
        for v in &op {
            assert_relative_eq!(v.norm(), 1.0, max_relative = 1.0e-12);
        }
    }

    #[test]
    fn test_broadening_factor_increases_with_z() {
        let fiber = FiberDispersion::smf28();
        let fwhm_ps = 1.0;
        let b1 = fiber.broadening_factor(1.0, fwhm_ps);
        let b2 = fiber.broadening_factor(10.0, fwhm_ps);
        let b3 = fiber.broadening_factor(100.0, fwhm_ps);
        assert!(
            b1 > 1.0 && b2 > b1 && b3 > b2,
            "Broadening factor must increase monotonically: {b1:.3} < {b2:.3} < {b3:.3}"
        );
    }

    #[test]
    fn test_dispersion_map_total_length() {
        let mut map = DispersionMap::new();
        map.add_span(FiberDispersion::smf28(), 80.0);
        map.add_span(FiberDispersion::new(350.0, 0.0, 0.0, 1550.0), 4.0); // DCF
        assert_relative_eq!(map.total_length_km(), 84.0, max_relative = 1.0e-10);
    }

    #[test]
    fn test_dispersion_map_net_dispersion() {
        let mut map = DispersionMap::new();
        // 80 km SMF-28 (β₂ = −21.7 ps²/km) + DCF to compensate
        map.add_span(FiberDispersion::smf28(), 80.0);
        // Net after SMF: -21.7 * 80 = -1736 ps²
        // Add a perfect compensator:
        map.add_span(FiberDispersion::new(21.7, 0.0, 0.0, 1550.0), 80.0);
        let net = map.net_dispersion_ps2();
        assert_relative_eq!(net, 0.0, epsilon = 1.0e-6);
        assert!(map.is_compensated(1.0));
    }

    #[test]
    fn test_zdw_nm_smf28() {
        let fiber = FiberDispersion::smf28();
        // SMF-28 ZDW is around 1310 nm but computed from β₂/β₃ at 1550 nm
        let zdw = fiber.zdw_nm();
        assert!(zdw.is_some(), "SMF-28 should have a finite ZDW");
        let zdw_val = zdw.expect("zdw should be Some");
        assert!(
            zdw_val > 100.0 && zdw_val < 5000.0,
            "ZDW out of plausible range: {zdw_val}"
        );
    }

    #[test]
    fn test_gvd_length_scaling() {
        // L_D ∝ T₀² → doubling FWHM quadruples L_D
        let fiber = FiberDispersion::smf28();
        let ld1 = fiber.gvd_length_m(1.0);
        let ld2 = fiber.gvd_length_m(2.0);
        assert_relative_eq!(ld2 / ld1, 4.0, max_relative = 1.0e-9);
    }

    #[test]
    fn test_dsf_near_zero_dispersion() {
        let fiber = FiberDispersion::dsf();
        let d = fiber.dispersion_ps_per_nm_km().abs();
        assert!(
            d < 1.0,
            "DSF dispersion should be near zero, got {d:.3} ps/(nm·km)"
        );
    }

    #[test]
    fn test_hnlf_small_anomalous() {
        let fiber = FiberDispersion::hnlf();
        assert!(
            fiber.beta2_ps2_per_km < 0.0,
            "HNLF should be anomalous (β₂ < 0)"
        );
        assert!(
            fiber.beta2_ps2_per_km.abs() < 5.0,
            "HNLF |β₂| should be small (< 5 ps²/km)"
        );
    }
}
