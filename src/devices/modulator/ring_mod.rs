/// Ring resonator electro-optic modulator model.
///
/// A ring modulator consists of a ring resonator coupled to a bus waveguide.
/// The electro-optic effect (plasma dispersion in Si, or Pockels effect)
/// shifts the resonance wavelength, modulating the transmission.
///
/// For silicon ring modulators, the plasma dispersion effect gives:
///   Δn ≈ -8.8×10⁻²² ΔN_e - 8.5×10⁻¹⁸ ΔN_h^0.8  (Soref-Bennett, 1987)
///   Δα ≈ 8.5×10⁻¹⁸ ΔN_e + 6.0×10⁻¹⁸ ΔN_h   (cm⁻¹, for cm units)
use std::f64::consts::PI;

/// Silicon ring modulator via plasma dispersion effect.
#[derive(Debug, Clone, Copy)]
pub struct SiliconRingModulator {
    /// Ring radius (m)
    pub radius: f64,
    /// Effective refractive index at operating point
    pub n_eff: f64,
    /// Group index n_g
    pub n_g: f64,
    /// Bus-to-ring power coupling coefficient κ² (intensity coupling, 0-1)
    pub kappa_sq: f64,
    /// Round-trip power loss (internal) a² ∈ (0, 1]
    pub a_sq: f64,
    /// Operating wavelength (m)
    pub wavelength: f64,
    /// Optical confinement factor in doped region Γ
    pub confinement: f64,
}

impl SiliconRingModulator {
    pub fn new(
        radius: f64,
        n_eff: f64,
        n_g: f64,
        kappa_sq: f64,
        a_sq: f64,
        wavelength: f64,
    ) -> Self {
        Self {
            radius,
            n_eff,
            n_g,
            kappa_sq,
            a_sq,
            wavelength,
            confinement: 0.8,
        }
    }

    /// Standard silicon ring modulator (5μm radius, 1550nm).
    pub fn standard_si_ring() -> Self {
        Self {
            radius: 5e-6,
            n_eff: 2.44,
            n_g: 4.18,
            kappa_sq: 0.05, // ~5% coupling per pass
            a_sq: 0.99,     // 1% round-trip loss
            wavelength: 1550e-9,
            confinement: 0.85,
        }
    }

    /// Free spectral range (m).
    pub fn fsr(&self) -> f64 {
        self.wavelength * self.wavelength / (2.0 * PI * self.radius * self.n_g)
    }

    /// Resonance wavelength λ_res = 2π·R·n_eff / m (nearest to operating λ).
    pub fn resonance_wavelength(&self) -> f64 {
        let circumference = 2.0 * PI * self.radius;
        let m = (circumference * self.n_eff / self.wavelength).round();
        circumference * self.n_eff / m
    }

    /// Transmission (all-pass ring) at wavelength λ.
    ///
    ///   T(λ) = (a² - 2·a·√(1-κ²)·cos(φ) + (1-κ²)) /
    ///           (1 - 2·a·√(1-κ²)·cos(φ) + a²·(1-κ²))
    ///
    /// where φ = 2π·n_eff·(2πR)/λ is the round-trip phase.
    pub fn transmission(&self, wavelength: f64, delta_n: f64) -> f64 {
        let n_eff_mod = self.n_eff + delta_n * self.confinement;
        let circumference = 2.0 * PI * self.radius;
        let phi = 2.0 * PI * n_eff_mod * circumference / wavelength;
        let a2 = self.a_sq;
        let t2 = 1.0 - self.kappa_sq; // through coupling power
        let a = a2.sqrt();
        let t = t2.sqrt();
        let cos_phi = phi.cos();
        let num = a2 - 2.0 * a * t * cos_phi + t2;
        let den = 1.0 - 2.0 * a * t * cos_phi + a2 * t2;
        num / den
    }

    /// Resonance wavelength shift per unit carrier density change (m per cm⁻³).
    ///
    /// Uses Soref-Bennett plasma dispersion: Δn ≈ -8.8e-22 ΔN (electrons at 1550nm)
    pub fn resonance_shift_per_carrier(&self) -> f64 {
        // Δλ_res = λ_res · Γ · Δn / n_g
        let dn_per_carrier = -8.8e-22; // m⁻³ → Δn (Soref-Bennett, 1550nm)
        let lambda_res = self.resonance_wavelength();
        lambda_res * self.confinement * dn_per_carrier / self.n_g
    }

    /// Extinction ratio (dB) — max/min transmission swing.
    ///
    /// At resonance: T_min → with large Δn: T_max = 1.0 (off resonance).
    pub fn extinction_ratio_db(&self) -> f64 {
        let t_min = self.transmission(self.resonance_wavelength(), 0.0);
        let t_max = 1.0;
        if t_min < 1e-20 {
            return 100.0; // critically coupled → ∞ ER
        }
        10.0 * (t_max / t_min).log10()
    }

    /// 3dB modulation bandwidth (Hz) due to photon lifetime in ring.
    ///
    ///   f_3dB = FSR / (π · F)  where F is finesse
    pub fn bandwidth_3db_hz(&self) -> f64 {
        let fsr_hz = 2.998e8 / (2.0 * PI * self.radius * self.n_g);
        let r_eff = (self.a_sq * (1.0 - self.kappa_sq)).sqrt();
        let finesse = PI * r_eff.sqrt() / (1.0 - r_eff);
        fsr_hz / finesse
    }

    /// Energy per bit (J) estimate based on V_π·C / 4.
    ///
    /// For depletion-mode Si modulator: E_bit = C·V² / 4 ≈ 10-100 fJ/bit.
    pub fn energy_per_bit_joules(&self, v_drive: f64, capacitance: f64) -> f64 {
        capacitance * v_drive * v_drive / 4.0
    }

    /// Carrier density change ΔN (cm⁻³) needed for λ_res shift of Δλ (m).
    pub fn carrier_density_for_shift(&self, delta_lambda: f64) -> f64 {
        let shift_per_carrier = self.resonance_shift_per_carrier();
        delta_lambda / shift_per_carrier
    }

    /// Finesse of the ring resonator.
    pub fn finesse(&self) -> f64 {
        let r_eff = (self.a_sq * (1.0 - self.kappa_sq)).sqrt();
        PI * r_eff.sqrt() / (1.0 - r_eff)
    }

    /// Q factor of the ring resonator.
    pub fn q_factor(&self) -> f64 {
        let lambda_res = self.resonance_wavelength();
        let fsr = self.fsr();
        let finesse = self.finesse();
        lambda_res * finesse / fsr
    }
}

/// GeSi electro-absorption (EA) ring modulator.
/// Uses Franz-Keldysh effect: absorption edge shifts with applied field.
#[derive(Debug, Clone, Copy)]
pub struct ElectroAbsorptionRing {
    /// Underlying silicon ring geometry
    pub ring: SiliconRingModulator,
    /// Extinction coefficient change per V/m applied field Δk per V/m
    pub dk_per_field: f64,
    /// Ge layer thickness (m) inside ring
    pub ge_thickness: f64,
}

impl ElectroAbsorptionRing {
    /// Standard GeSi EA ring at 1310nm.
    pub fn gesi_1310nm() -> Self {
        let ring = SiliconRingModulator {
            radius: 10e-6,
            n_eff: 3.6,
            n_g: 4.5,
            kappa_sq: 0.08,
            a_sq: 0.95,
            wavelength: 1310e-9,
            confinement: 0.4,
        };
        Self {
            ring,
            dk_per_field: 1e-7,
            ge_thickness: 100e-9,
        }
    }

    /// Loss change (m⁻¹) per volt for EA ring.
    pub fn loss_per_volt(&self, v: f64) -> f64 {
        let field = v / self.ge_thickness;
        let dk = self.dk_per_field * field;
        4.0 * PI * dk / self.ring.wavelength
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsr_physical_range() {
        let m = SiliconRingModulator::standard_si_ring();
        let fsr = m.fsr();
        // R=5μm, n_g=4.18: FSR = λ²/(2π·R·n_g) ≈ 11.5nm
        assert!(fsr > 5e-9 && fsr < 30e-9, "FSR={:.2e}", fsr);
    }

    #[test]
    fn resonance_wavelength_near_operating() {
        let m = SiliconRingModulator::standard_si_ring();
        let lres = m.resonance_wavelength();
        assert!((lres - m.wavelength).abs() < m.fsr());
    }

    #[test]
    fn transmission_at_resonance_below_unity() {
        let m = SiliconRingModulator::standard_si_ring();
        let lres = m.resonance_wavelength();
        let t = m.transmission(lres, 0.0);
        assert!((0.0..1.0).contains(&t), "T={t:.4}");
    }

    #[test]
    fn transmission_off_resonance_near_unity() {
        let m = SiliconRingModulator::standard_si_ring();
        let lres = m.resonance_wavelength();
        // Half-FSR away from resonance
        let t = m.transmission(lres + m.fsr() * 0.5, 0.0);
        assert!(t > 0.9, "T off-res={t:.4}");
    }

    #[test]
    fn extinction_ratio_positive() {
        let m = SiliconRingModulator::standard_si_ring();
        let er = m.extinction_ratio_db();
        assert!(er > 0.0);
    }

    #[test]
    fn bandwidth_3db_positive() {
        let m = SiliconRingModulator::standard_si_ring();
        let bw = m.bandwidth_3db_hz();
        assert!(bw > 1e6); // > 1 MHz
    }

    #[test]
    fn q_factor_positive() {
        let m = SiliconRingModulator::standard_si_ring();
        let q = m.q_factor();
        assert!(q > 100.0, "Q={q:.0}");
    }

    #[test]
    fn energy_per_bit_picojoules_range() {
        let m = SiliconRingModulator::standard_si_ring();
        // 50fF junction, 2V drive: E = 50e-15 * 4 / 4 = 50 fJ
        let e = m.energy_per_bit_joules(2.0, 50e-15);
        assert!(e > 1e-15 && e < 1e-9, "E={e:.2e} J");
    }
}
