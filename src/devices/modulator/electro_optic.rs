//! Electro-optic (Pockels) effect modulators.
//!
//! The linear electro-optic (Pockels) effect changes the refractive index in
//! proportion to an applied electric field:
//!
//!   Δ(1/n²) = r·E  →  Δn ≈ -(1/2)·n³·r·E
//!
//! where `r` is the relevant element of the EO tensor (m/V) and `E = V/d` is
//! the field across an electrode gap `d`.
//!
//! Half-wave voltage:  V_π = λ·d / (n³·r·L)
//! Phase shift:        Δφ = π·n³·r·E·L / λ = π·V / V_π
//!
//! Common materials: LiNbO₃ (r₃₃ = 30.8 pm/V), KTP (r₃₃ = 35 pm/V),
//! BBO (r₂₂ = 2.2 pm/V), GaAs (r₄₁ = 1.5 pm/V).

use std::f64::consts::PI;

/// Electro-optic crystal material with Pockels coefficients.
#[derive(Debug, Clone)]
pub struct EoCrystal {
    /// Refractive index at operating wavelength
    pub n: f64,
    /// Dominant EO coefficient r (m/V)
    pub r_eff: f64,
    /// Material name
    pub name: &'static str,
}

impl EoCrystal {
    /// Lithium niobate (LiNbO₃): r₃₃ = 30.8 pm/V, n_e ≈ 2.17 @ 1550 nm.
    pub fn lithium_niobate() -> Self {
        Self {
            n: 2.17,
            r_eff: 30.8e-12,
            name: "LiNbO3",
        }
    }

    /// Potassium titanyl phosphate (KTP): r₃₃ = 35 pm/V, n_z ≈ 1.74 @ 1064 nm.
    pub fn ktp() -> Self {
        Self {
            n: 1.74,
            r_eff: 35.0e-12,
            name: "KTP",
        }
    }

    /// Beta barium borate (BBO): r₂₂ = 2.2 pm/V, n_o ≈ 1.67 @ 532 nm.
    pub fn bbo() -> Self {
        Self {
            n: 1.67,
            r_eff: 2.2e-12,
            name: "BBO",
        }
    }

    /// Gallium arsenide (GaAs): r₄₁ = 1.5 pm/V, n ≈ 3.37 @ 1064 nm.
    pub fn gaas() -> Self {
        Self {
            n: 3.37,
            r_eff: 1.5e-12,
            name: "GaAs",
        }
    }

    /// Refractive index change for applied field E (V/m).
    ///   Δn = -(1/2)·n³·r·E
    pub fn delta_n(&self, e_field: f64) -> f64 {
        -0.5 * self.n.powi(3) * self.r_eff * e_field
    }
}

/// Longitudinal Pockels cell (field along optical axis, z-cut LiNbO₃).
///
/// In longitudinal configuration the electrode gap equals the interaction
/// length L, so V_π is independent of L:
///   V_π = λ / (n³·r)
#[derive(Debug, Clone)]
pub struct LongitudinalPockelsCell {
    /// Crystal parameters
    pub crystal: EoCrystal,
    /// Wavelength (m)
    pub wavelength: f64,
}

impl LongitudinalPockelsCell {
    pub fn new(crystal: EoCrystal, wavelength: f64) -> Self {
        Self {
            crystal,
            wavelength,
        }
    }

    /// Half-wave voltage V_π (V): V_π = λ / (n³·r).
    pub fn v_pi(&self) -> f64 {
        self.wavelength / (self.crystal.n.powi(3) * self.crystal.r_eff)
    }

    /// Phase retardation for applied voltage V (V): Γ = π·V / V_π.
    pub fn phase_retardation(&self, voltage: f64) -> f64 {
        PI * voltage / self.v_pi()
    }

    /// Intensity transmission for crossed polarizers with retardation Γ.
    ///   T = sin²(Γ/2)
    pub fn transmission(&self, voltage: f64) -> f64 {
        (self.phase_retardation(voltage) / 2.0).sin().powi(2)
    }
}

/// Transverse Pockels cell (field perpendicular to optical axis).
///
/// In transverse configuration: V_π = λ·d / (n³·r·L)
/// where d = electrode gap, L = interaction length.
#[derive(Debug, Clone)]
pub struct TransversePockelsCell {
    /// Crystal parameters
    pub crystal: EoCrystal,
    /// Wavelength (m)
    pub wavelength: f64,
    /// Electrode gap d (m)
    pub gap: f64,
    /// Interaction length L (m)
    pub length: f64,
}

impl TransversePockelsCell {
    pub fn new(crystal: EoCrystal, wavelength: f64, gap: f64, length: f64) -> Self {
        Self {
            crystal,
            wavelength,
            gap,
            length,
        }
    }

    /// Half-wave voltage V_π (V) = λ·d / (n³·r·L).
    pub fn v_pi(&self) -> f64 {
        self.wavelength * self.gap / (self.crystal.n.powi(3) * self.crystal.r_eff * self.length)
    }

    /// Phase shift for applied voltage V: Δφ = π·V / V_π.
    pub fn phase_shift(&self, voltage: f64) -> f64 {
        PI * voltage / self.v_pi()
    }

    /// Electric field in the crystal (V/m) for applied voltage V.
    pub fn e_field(&self, voltage: f64) -> f64 {
        voltage / self.gap
    }

    /// Index change Δn for applied voltage V.
    pub fn delta_n(&self, voltage: f64) -> f64 {
        self.crystal.delta_n(self.e_field(voltage))
    }
}

/// Electro-optic phase modulator bandwidth model.
///
/// The 3-dB electrical bandwidth is limited by the RC time constant:
///   f₃dB = 1 / (2π·R·C)
/// where C = ε₀·ε_r·A/d is the electrode capacitance.
#[derive(Debug, Clone)]
pub struct EoModulatorBandwidth {
    /// Series resistance Ω
    pub resistance: f64,
    /// Electrode area A (m²)
    pub electrode_area: f64,
    /// Electrode gap d (m)
    pub gap: f64,
    /// Relative permittivity of crystal at RF frequency
    pub eps_r_rf: f64,
}

impl EoModulatorBandwidth {
    const EPS0: f64 = 8.854_187_817e-12;

    /// LiNbO₃ travelling-wave modulator: Z=50 Ω, ε_r(RF)≈28, 1cm×10μm electrode.
    pub fn linbo3_travelling_wave() -> Self {
        Self {
            resistance: 50.0,
            electrode_area: 1e-2 * 10e-6,
            gap: 10e-6,
            eps_r_rf: 28.0,
        }
    }

    /// Electrode capacitance C (F) = ε₀·ε_r·A/d.
    pub fn capacitance(&self) -> f64 {
        Self::EPS0 * self.eps_r_rf * self.electrode_area / self.gap
    }

    /// 3-dB bandwidth (Hz) = 1 / (2π·R·C).
    pub fn bandwidth_hz(&self) -> f64 {
        1.0 / (2.0 * PI * self.resistance * self.capacitance())
    }

    /// Modulation efficiency V_π·L (V·cm) — lower is better.
    ///
    /// For a transverse modulator: V_π·L = λ·d / (n³·r).
    pub fn vpi_l_product(crystal: &EoCrystal, wavelength: f64, gap: f64) -> f64 {
        wavelength * gap / (crystal.n.powi(3) * crystal.r_eff) * 1e2 // convert to V·cm
    }
}

/// Silicon photonics plasma-dispersion EO phase shifter.
///
/// Based on Soref & Bennett (1987) empirical relations:
///   Δn_e = -8.8×10⁻²²·ΔN - 8.5×10⁻¹⁸·ΔP^0.8
///   Δα   =  8.5×10⁻¹⁸·ΔN + 6.0×10⁻¹⁸·ΔP     (cm⁻¹)
/// at λ = 1550 nm (ΔN = free electrons /cm³, ΔP = holes /cm³).
#[derive(Debug, Clone, Copy)]
pub struct SiPlasmaDispersion {
    /// Free-carrier change ΔN (electrons/m³)
    pub delta_n_carriers: f64,
    /// Hole change ΔP (holes/m³)
    pub delta_p_carriers: f64,
    /// Phase shifter length L (m)
    pub length: f64,
}

impl SiPlasmaDispersion {
    /// Phase shift Δφ (rad) at 1550 nm.
    pub fn phase_shift(&self) -> f64 {
        let wavelength = 1550e-9;
        let dn = self.delta_n_1550();
        2.0 * PI * dn * self.length / wavelength
    }

    /// Index change Δn at 1550 nm (Soref & Bennett).
    pub fn delta_n_1550(&self) -> f64 {
        let dn_m3 = self.delta_n_carriers * 1e-6; // /m³ → /cm³
        let dp_m3 = self.delta_p_carriers * 1e-6;
        -(8.8e-22 * dn_m3 + 8.5e-18 * dp_m3.powf(0.8))
    }

    /// Absorption change Δα (m⁻¹) at 1550 nm.
    pub fn delta_alpha_1550(&self) -> f64 {
        let dn_cm = self.delta_n_carriers * 1e-6;
        let dp_cm = self.delta_p_carriers * 1e-6;
        (8.5e-18 * dn_cm + 6.0e-18 * dp_cm) * 100.0 // cm⁻¹ → m⁻¹
    }

    /// Voltage-length product V_π·L for a p-n junction with 1V swing.
    /// Approximated for a 500 nm × 220 nm rib waveguide.
    pub fn vpi_l_pm(&self) -> f64 {
        let dphase = self.phase_shift().abs();
        if dphase > 0.0 {
            PI * self.length / dphase // length for π shift in meters
        } else {
            f64::INFINITY
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linbo3_vpi_reasonable() {
        let cell = LongitudinalPockelsCell::new(EoCrystal::lithium_niobate(), 1550e-9);
        let vpi = cell.v_pi();
        // V_π for longitudinal LiNbO3 @ 1550 nm: λ/(n³·r₃₃) ≈ 1550e-9/(2.17³×30.8e-12)
        // ≈ 1550e-9 / (10.2·30.8e-12) ≈ 1550e-9 / 314e-12 ≈ 4935 V
        assert!(vpi > 1000.0 && vpi < 20000.0, "V_pi={vpi}");
    }

    #[test]
    fn transverse_vpi_lower() {
        let crystal = EoCrystal::lithium_niobate();
        let cell = TransversePockelsCell::new(crystal, 1550e-9, 10e-6, 1e-2);
        let vpi = cell.v_pi();
        // Transverse with 1cm interaction, 10µm gap
        assert!(vpi > 0.0 && vpi < 50.0, "V_pi transverse={vpi}");
    }

    #[test]
    fn transmission_at_vpi_is_one() {
        let cell = LongitudinalPockelsCell::new(EoCrystal::lithium_niobate(), 1550e-9);
        let vpi = cell.v_pi();
        let t = cell.transmission(vpi);
        // At V_π, phase retardation = π, T = sin²(π/2) = 1
        assert!((t - 1.0).abs() < 1e-10, "T at Vpi={t}");
    }

    #[test]
    fn transmission_at_zero_is_zero() {
        let cell = LongitudinalPockelsCell::new(EoCrystal::lithium_niobate(), 1550e-9);
        let t = cell.transmission(0.0);
        assert!(t < 1e-10, "T at 0={t}");
    }

    #[test]
    fn eo_bandwidth_linbo3() {
        let bw = EoModulatorBandwidth::linbo3_travelling_wave();
        let f3db = bw.bandwidth_hz();
        // Should be in GHz range for travelling-wave modulator
        assert!(f3db > 1e9, "bandwidth too low: {f3db}");
    }

    #[test]
    fn ktp_crystal_vpi_different_from_linbo3() {
        let ln = LongitudinalPockelsCell::new(EoCrystal::lithium_niobate(), 1550e-9);
        let ktp = LongitudinalPockelsCell::new(EoCrystal::ktp(), 1550e-9);
        assert!((ln.v_pi() - ktp.v_pi()).abs() > 100.0);
    }

    #[test]
    fn si_plasma_dispersion_phase_shift() {
        let ps = SiPlasmaDispersion {
            delta_n_carriers: 1e23, // 1e17 /cm³ in /m³
            delta_p_carriers: 1e23,
            length: 1e-3, // 1 mm
        };
        let dphase = ps.phase_shift();
        // Should be non-zero
        assert!(dphase.abs() > 0.0);
    }

    #[test]
    fn delta_n_negative_for_free_carriers() {
        let ps = SiPlasmaDispersion {
            delta_n_carriers: 1e23,
            delta_p_carriers: 1e23,
            length: 1e-3,
        };
        // Free carriers reduce index (plasma dispersion)
        assert!(ps.delta_n_1550() < 0.0);
    }

    #[test]
    fn vpi_l_product_linbo3() {
        let crystal = EoCrystal::lithium_niobate();
        let vpi_l = EoModulatorBandwidth::vpi_l_product(&crystal, 1550e-9, 10e-6);
        // For LiNbO3 transverse: should be a few V·cm
        assert!(vpi_l > 0.0 && vpi_l < 100.0, "V_pi*L = {vpi_l} V·cm");
    }
}
