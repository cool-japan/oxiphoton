/// Slot waveguide model.
///
/// A slot waveguide consists of two high-index rails separated by a narrow
/// low-index slot. The fundamental quasi-TE mode has its peak field intensity
/// in the low-index slot — ideal for sensing, nonlinear optics, and
/// electro-optic modulation.
///
/// Structure (cross-section):
///   [substrate (n_sub)] / [oxide (n_ox)] / [rail 1 (n_rail, w_rail)] +
///   [slot (n_slot, w_slot)] + [rail 2 (n_rail, w_rail)] / [oxide (n_ox)]
use std::f64::consts::PI;

/// Slot waveguide geometry and optical properties.
#[derive(Debug, Clone, Copy)]
pub struct SlotWaveguide {
    /// Rail refractive index (e.g., Si ≈ 3.48)
    pub n_rail: f64,
    /// Slot (gap) refractive index (e.g., SiO₂ ≈ 1.44 or air ≈ 1.0)
    pub n_slot: f64,
    /// Substrate/cladding refractive index
    pub n_clad: f64,
    /// Rail width (m)
    pub w_rail: f64,
    /// Slot width (m)
    pub w_slot: f64,
    /// Rail height (m)
    pub height: f64,
}

impl SlotWaveguide {
    pub fn new(
        n_rail: f64,
        n_slot: f64,
        n_clad: f64,
        w_rail: f64,
        w_slot: f64,
        height: f64,
    ) -> Self {
        Self {
            n_rail,
            n_slot,
            n_clad,
            w_rail,
            w_slot,
            height,
        }
    }

    /// Silicon-on-insulator (SOI) slot waveguide at 1550nm.
    ///
    /// Standard parameters: 220nm tall, 180nm rails, 100nm slot.
    pub fn soi_standard() -> Self {
        Self {
            n_rail: 3.476,
            n_slot: 1.444, // SiO₂ slot fill
            n_clad: 1.444,
            w_rail: 180e-9,
            w_slot: 100e-9,
            height: 220e-9,
        }
    }

    /// Air-clad SOI slot waveguide (slot filled with analyte, n≈1).
    pub fn soi_air_slot() -> Self {
        Self {
            n_rail: 3.476,
            n_slot: 1.0,
            n_clad: 1.0,
            w_rail: 200e-9,
            w_slot: 120e-9,
            height: 220e-9,
        }
    }

    /// Total waveguide width (m): w_total = 2·w_rail + w_slot.
    pub fn total_width(&self) -> f64 {
        2.0 * self.w_rail + self.w_slot
    }

    /// Effective index approximation using slot-waveguide EIM.
    ///
    /// Approximate quasi-TE n_eff using the effective index of an
    /// equivalent slab with averaged transverse mode profile.
    ///
    /// Uses the Xu et al. (2004) approximation:
    ///   n_eff ≈ n_rail · f_rail + n_slot · f_slot   (weighted average)
    ///
    /// where f_rail, f_slot are fill fractions weighted by field enhancement.
    pub fn effective_index_approx(&self, wavelength: f64) -> f64 {
        let w_tot = self.total_width();
        let f_slot = self.w_slot / w_tot;
        let f_rail = 1.0 - f_slot;
        // Field in slot is enhanced by (n_rail/n_slot)² relative to rail
        let enhancement = (self.n_rail / self.n_slot).powi(2);
        let norm = f_rail + f_slot * enhancement;
        let n_eff_sq = (f_rail * self.n_rail * self.n_rail
            + f_slot * self.n_slot * self.n_slot * enhancement)
            / norm;
        // Apply height confinement correction (slab approximation)
        let k0 = 2.0 * PI / wavelength;
        let v_h =
            k0 * self.height * (self.n_rail * self.n_rail - self.n_clad * self.n_clad).sqrt() / 2.0;
        let correction = 1.0 - 0.5 / (v_h * v_h + 1.0).max(1.0);
        (n_eff_sq * correction * correction).sqrt().min(self.n_rail)
    }

    /// Confinement factor in the slot region Γ_slot.
    ///
    /// Uses the field enhancement principle for quasi-TE slot modes:
    ///   Γ_slot ≈ f_slot · (n_rail/n_slot)² / [1 + f_slot·((n_rail/n_slot)² - 1)]
    pub fn slot_confinement_factor(&self) -> f64 {
        let w_tot = self.total_width();
        let f_slot = self.w_slot / w_tot;
        let enhancement = (self.n_rail / self.n_slot).powi(2);
        f_slot * enhancement / (1.0 + f_slot * (enhancement - 1.0))
    }

    /// Group velocity dispersion β₂ (s²/m) — approximate waveguide contribution.
    ///
    /// Uses numerical differentiation of n_eff(λ).
    pub fn beta2_approx(&self, wavelength: f64) -> f64 {
        let dl = wavelength * 1e-4; // 0.01% wavelength step
        let c = 2.998e8;
        let n_p = self.effective_index_approx(wavelength + dl);
        let n_0 = self.effective_index_approx(wavelength);
        let n_m = self.effective_index_approx(wavelength - dl);
        // β₂ = λ³/(2πc²) · d²n_eff/dλ²
        let d2n_dl2 = (n_p - 2.0 * n_0 + n_m) / (dl * dl);
        wavelength * wavelength * wavelength / (2.0 * PI * c * c) * d2n_dl2
    }

    /// Nonlinear coefficient γ (rad/(W·m)) given n₂ of rail and mode area.
    ///
    ///   γ = (ω/c) · n₂_eff / A_eff
    ///
    /// n₂_eff = Γ_slot · n₂_slot + (1-Γ_slot) · n₂_rail
    pub fn nonlinear_coefficient(&self, wavelength: f64, n2_rail: f64, n2_slot: f64) -> f64 {
        let c = 2.998e8;
        let omega = 2.0 * PI * c / wavelength;
        let gamma_slot = self.slot_confinement_factor();
        let n2_eff = gamma_slot * n2_slot + (1.0 - gamma_slot) * n2_rail;
        // Effective mode area A_eff ≈ w_total × height (rectangular approximation)
        let a_eff = self.total_width() * self.height;
        let n_eff = self.effective_index_approx(wavelength);
        omega * n2_eff / (c * n_eff * a_eff)
    }

    /// Sensing figure of merit for refractive index sensing.
    ///
    ///   FOM = Γ_slot / A_eff  (m⁻²)
    ///
    /// Higher FOM → greater sensitivity to analyte index changes in the slot.
    pub fn sensing_figure_of_merit(&self) -> f64 {
        let gamma = self.slot_confinement_factor();
        let a_eff = self.total_width() * self.height;
        gamma / a_eff
    }

    /// Waveguide loss sensitivity (dB/RIU — refractive index unit).
    ///
    /// Change in effective index per RIU change in slot index:
    ///   dn_eff/dn_slot = Γ_slot
    pub fn index_sensitivity(&self) -> f64 {
        self.slot_confinement_factor()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soi_slot_geometry() {
        let wg = SlotWaveguide::soi_standard();
        let w = wg.total_width();
        assert!((w - 460e-9).abs() < 1e-12);
    }

    #[test]
    fn effective_index_between_clad_and_rail() {
        let wg = SlotWaveguide::soi_standard();
        let n_eff = wg.effective_index_approx(1550e-9);
        assert!(n_eff > wg.n_clad, "n_eff={n_eff:.3} should be > n_clad");
        assert!(n_eff < wg.n_rail, "n_eff={n_eff:.3} should be < n_rail");
    }

    #[test]
    fn slot_confinement_factor_positive() {
        let wg = SlotWaveguide::soi_standard();
        let gamma = wg.slot_confinement_factor();
        assert!(gamma > 0.0 && gamma < 1.0, "Γ_slot={gamma:.3}");
    }

    #[test]
    fn slot_confinement_enhanced_over_fill_fraction() {
        let wg = SlotWaveguide::soi_standard();
        let gamma = wg.slot_confinement_factor();
        let f_slot = wg.w_slot / wg.total_width();
        // Slot confinement should exceed fill fraction due to field enhancement
        assert!(
            gamma > f_slot,
            "Γ={gamma:.3} should exceed f_slot={f_slot:.3}"
        );
    }

    #[test]
    fn nonlinear_coefficient_positive() {
        let wg = SlotWaveguide::soi_standard();
        let n2_si = 6e-18; // Si n₂ (m²/W)
        let n2_sio2 = 2.2e-20; // SiO₂ n₂
        let gamma = wg.nonlinear_coefficient(1550e-9, n2_si, n2_sio2);
        assert!(gamma > 0.0);
        // Should be in range typical for SOI slot waveguides (~100-1000 /W/m)
        assert!(gamma > 10.0 && gamma < 1e5, "γ={gamma:.2e}");
    }

    #[test]
    fn sensing_fom_positive() {
        let wg = SlotWaveguide::soi_air_slot();
        let fom = wg.sensing_figure_of_merit();
        assert!(fom > 0.0);
    }

    #[test]
    fn air_slot_higher_confinement_than_oxide() {
        let wg_ox = SlotWaveguide::soi_standard();
        let wg_air = SlotWaveguide::soi_air_slot();
        // Air slot has lower n_slot → larger (n_rail/n_slot)² → larger Γ
        assert!(wg_air.slot_confinement_factor() > wg_ox.slot_confinement_factor());
    }
}
