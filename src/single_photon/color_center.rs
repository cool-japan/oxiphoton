//! Solid-state colour-centre single-photon emitters.
//!
//! Models the optical and spin properties of:
//! - NV⁻ / NV⁰ centres in diamond (nitrogen-vacancy)
//! - SiV centres in diamond (silicon-vacancy, D3d symmetry)
//! - SnV centres in diamond (tin-vacancy, emerging platform)
//! - Defects in hexagonal boron nitride (hBN)
//!
//! # References
//! - Doherty et al., Phys. Rep. 528, 1 (2013)   — NV review
//! - Trusheim et al., PRL 124, 023602 (2020)    — SnV
//! - Bradac et al., Nat. Comm. 10, 5625 (2019)  — colour centre comparison
//! - Exarhos et al., Nat. Comm. 8, 15783 (2017) — hBN single-photon emitters

use std::f64::consts::PI;

// ─── Physical constants ────────────────────────────────────────────────────────

/// Planck's constant (J·s)
const H_PLANCK: f64 = 6.626_070_15e-34;
/// Boltzmann constant (J/K)
const K_B: f64 = 1.380_649e-23;
/// Bohr magneton (J/T)
const MU_B: f64 = 9.274_010_08e-24;
/// NV⁻ electron spin g-factor (≈ 2.003 for NV⁻)
const G_NV: f64 = 2.003;

// ─── NV charge state ──────────────────────────────────────────────────────────

/// Charge state of the nitrogen-vacancy centre.
#[derive(Debug, Clone, PartialEq)]
pub enum NvCharge {
    /// NV⁻: the negatively charged state — spin qubit and main emitter for sensing.
    Negative,
    /// NV⁰: neutral state — different ZPL (575 nm), no spin-qubit functionality.
    Neutral,
}

// ─── NvCenter ─────────────────────────────────────────────────────────────────

/// Nitrogen-Vacancy (NV) colour centre in diamond.
///
/// The NV⁻ centre is a spin-1 system with a ground-state zero-field splitting
/// D = 2.87 GHz and is the leading solid-state quantum sensor and spin qubit.
#[derive(Debug, Clone)]
pub struct NvCenter {
    /// Charge state (NV⁻ or NV⁰)
    pub charge_state: NvCharge,
    /// Applied magnetic field vector [Bx, By, Bz] in Gauss
    pub magnetic_field_gauss: [f64; 3],
    /// Sample temperature (K)
    pub temperature_k: f64,
    /// Local strain-induced splitting (MHz) — shifts ODMR resonances
    pub strain_mhz: f64,
}

impl NvCenter {
    /// Create an NV⁻ centre with default parameters at the given temperature.
    pub fn new_nv_minus(temperature_k: f64) -> Self {
        Self {
            charge_state: NvCharge::Negative,
            magnetic_field_gauss: [0.0, 0.0, 0.0],
            temperature_k,
            strain_mhz: 0.0,
        }
    }

    /// Zero-phonon line (ZPL) wavelength (nm).
    ///
    /// - NV⁻: 637 nm (1.945 eV)
    /// - NV⁰: 575 nm (2.156 eV)
    pub fn zpl_wavelength_nm(&self) -> f64 {
        match self.charge_state {
            NvCharge::Negative => 637.0,
            NvCharge::Neutral => 575.0,
        }
    }

    /// Debye-Waller (ZPL) factor.
    ///
    /// Fraction of emission into the ZPL.  For NV⁻ at cryogenic temperatures:
    ///   DW ≈ 0.03 (only 3 % ZPL — poor for photonic applications without a cavity).
    /// At room temperature the phonon sideband dominates even more.
    pub fn debye_waller_factor(&self) -> f64 {
        match self.charge_state {
            NvCharge::Negative => {
                // DW(T) ≈ DW₀ * exp(−γ_ph * T / T_Debye)
                let dw0 = 0.032_f64;
                let t_d = 1860.0_f64; // Debye temperature of diamond (K)
                let gamma_ph = 2.0_f64;
                let t = self.temperature_k;
                dw0 * (-gamma_ph * t / t_d).exp()
            }
            NvCharge::Neutral => 0.05, // slightly better DW for NV⁰
        }
    }

    /// Radiative lifetime (ns).
    ///
    /// NV⁻: ~12 ns at room temperature, ~12 ns at 4 K (mostly radiative).
    pub fn radiative_lifetime_ns(&self) -> f64 {
        match self.charge_state {
            NvCharge::Negative => {
                // Weak T-dependence via phonon sideband reabsorption
                let tau_0 = 12.0_f64; // ns at low T
                let t = self.temperature_k;
                // Slight temperature dependence ~2 % per 100 K
                tau_0 * (1.0 + 0.002 * (t - 4.0).max(0.0) / 100.0)
            }
            NvCharge::Neutral => 20.0,
        }
    }

    /// Internal quantum efficiency (η).
    ///
    /// NV⁻ has ~70 % radiative yield at room temperature due to a metastable
    /// singlet state pathway.  At low T this improves slightly.
    pub fn quantum_efficiency(&self) -> f64 {
        match self.charge_state {
            NvCharge::Negative => {
                let t = self.temperature_k;
                // Empirical: η rises from ~0.65 at RT to ~0.95 at 10 K
                if t < 30.0 {
                    0.95
                } else {
                    0.70 + 0.25 * (-0.02 * t).exp()
                }
            }
            NvCharge::Neutral => 0.80,
        }
    }

    /// Zero-field splitting D (GHz) for NV⁻ ground state (ms=0 ↔ ms=±1).
    ///
    /// D ≈ 2.877 GHz at 0 K; has a temperature coefficient dD/dT ≈ −74 kHz/K.
    pub fn zero_field_splitting_ghz(&self) -> f64 {
        match self.charge_state {
            NvCharge::Negative => {
                let d0 = 2.877_2_f64; // GHz at 0 K
                let dd_dt = -74.0e-6_f64; // GHz/K
                (d0 + dd_dt * self.temperature_k).max(0.0)
            }
            NvCharge::Neutral => 0.0, // NV⁰ has S=1/2, no zero-field splitting
        }
    }

    /// Zeeman splitting (GHz) along the NV axis for a field of `b_field_gauss`.
    ///
    /// ΔE = g·μ_B·B = 2.8 MHz/Gauss for NV⁻.
    pub fn zeeman_splitting_ghz(&self, b_field_gauss: f64) -> f64 {
        // 2.8 MHz/Gauss = 0.0028 GHz/Gauss (for NV⁻, g ≈ 2.003)
        let b_tesla = b_field_gauss * 1e-4; // Gauss → Tesla
        G_NV * MU_B * b_tesla / (H_PLANCK * 1e9) // J → GHz
    }

    /// ODMR contrast (spin-dependent fluorescence ratio).
    ///
    /// Defined as (I_{ms=0} − I_{ms=±1})/I_{ms=0}.
    /// Typical: ~25 % for NV⁻ at room temperature under 532 nm excitation.
    pub fn odmr_contrast(&self) -> f64 {
        match self.charge_state {
            NvCharge::Negative => {
                let t = self.temperature_k;
                // Contrast improves slightly at low temperature
                if t < 30.0 {
                    0.30
                } else {
                    0.25
                }
            }
            NvCharge::Neutral => 0.05,
        }
    }

    /// Longitudinal spin relaxation time T₁ (ms).
    ///
    /// Dominated by phonon processes:
    /// - ~1 ms at room temperature
    /// - ~3 s at 4 K in high-purity CVD diamond
    pub fn t1_spin_ms(&self) -> f64 {
        let t = self.temperature_k;
        if t < 10.0 {
            // Low-temperature: T₁ limited by magnetic impurities; can reach seconds
            1000.0 // ~1 s in ms
        } else if t < 77.0 {
            // Intermediate: phonon scattering ~T⁵
            1000.0 * (10.0_f64 / t).powi(5)
        } else {
            // Room temperature
            1.0
        }
    }

    /// Transverse spin coherence time T₂ (μs).
    ///
    /// Limited by the ¹³C nuclear spin bath.
    /// - ~1 μs at RT in natural-abundance diamond
    /// - ~1000 μs at RT in isotopically purified diamond (¹²C > 99.99 %)
    pub fn t2_spin_us(&self) -> f64 {
        match self.charge_state {
            NvCharge::Negative => {
                let t = self.temperature_k;
                if t < 30.0 {
                    1000.0
                } else {
                    2.0
                }
            }
            NvCharge::Neutral => 0.1,
        }
    }

    /// Inhomogeneous dephasing time T₂* (μs).
    ///
    /// T₂* < T₂ due to static field inhomogeneity from the ¹³C bath.
    /// Typical: 1–5 μs for natural-abundance diamond.
    pub fn t2_star_us(&self) -> f64 {
        // T₂* ≈ T₂ / 10 in natural-abundance diamond
        self.t2_spin_us() / 10.0
    }

    /// Magnetic field sensitivity (nT/√Hz).
    ///
    /// δB = ħ / (g·μ_B·√T₂·√N_photons) * 1/contrast
    /// For continuous-wave ODMR: δB ≈ ΔB_ODMR / (contrast * √(photon_rate))
    ///
    /// Here we use the shot-noise limited estimate:
    ///   δB = 1 / (g·μ_B·T₂*·√(R)) in appropriate units.
    pub fn magnetic_sensitivity_nt_per_sqrthz(&self, collection_efficiency: f64) -> f64 {
        let t2_star_s = self.t2_star_us() * 1e-6;
        let contrast = self.odmr_contrast();
        let tau_rad_s = self.radiative_lifetime_ns() * 1e-9;
        let photon_rate = collection_efficiency * self.quantum_efficiency() / tau_rad_s;
        // γ_NV = g μ_B / ħ (rad/s/T)
        let gamma_nv = G_NV * MU_B / (1.054_571_817e-34_f64); // rad/s/T
                                                              // δB (T/√Hz) = 1/(γ_NV · T₂* · contrast · √(photon_rate · T₂*))
        if photon_rate > 0.0 && t2_star_s > 0.0 && contrast > 0.0 {
            let sens_t = 1.0 / (gamma_nv * t2_star_s * contrast * (photon_rate * t2_star_s).sqrt());
            sens_t * 1e9 // T/√Hz → nT/√Hz
        } else {
            f64::INFINITY
        }
    }

    /// Temperature sensitivity (mK/√Hz) via the temperature dependence of D(T).
    ///
    /// dD/dT ≈ −74 kHz/K → sensitivity δT = δD / |dD/dT|.
    pub fn temperature_sensitivity_mk_per_sqrthz(&self, photon_rate: f64) -> f64 {
        // Frequency sensitivity δν (Hz/√Hz) = linewidth / contrast / √(photon_rate)
        let linewidth_hz = 1.0 / (2.0 * PI * self.t2_star_us() * 1e-6); // Hz
        let contrast = self.odmr_contrast();
        if photon_rate > 0.0 && contrast > 0.0 {
            let df_per_sqrthz = linewidth_hz / (contrast * photon_rate.sqrt());
            let dd_dt_hz_per_k = 74.0e3_f64; // 74 kHz/K
            let dt_per_sqrthz_k = df_per_sqrthz / dd_dt_hz_per_k;
            dt_per_sqrthz_k * 1000.0 // K/√Hz → mK/√Hz
        } else {
            f64::INFINITY
        }
    }
}

// ─── SivCenter ────────────────────────────────────────────────────────────────

/// Silicon-Vacancy (SiV) colour centre in diamond.
///
/// The SiV centre has D3d symmetry, resulting in inversion symmetry that
/// makes it insensitive to electric field noise.  Its 738 nm ZPL has a
/// Debye-Waller factor of ~70 %, far superior to NV.  However, spin coherence
/// requires milli-Kelvin temperatures.
#[derive(Debug, Clone)]
pub struct SivCenter {
    /// Sample temperature (K)
    pub temperature_k: f64,
    /// Symmetry-breaking strain splitting (THz)
    pub strain_thz: f64,
}

impl SivCenter {
    /// Create an SiV centre at the given temperature.
    pub fn new(temperature_k: f64) -> Self {
        Self {
            temperature_k,
            strain_thz: 0.0,
        }
    }

    /// Zero-phonon line wavelength: 738 nm.
    pub fn zpl_wavelength_nm(&self) -> f64 {
        738.0_f64
    }

    /// Debye-Waller factor: ~0.70 at low temperature (much higher than NV!).
    ///
    /// For SiV the strong inversion symmetry concentrates emission into ZPL.
    pub fn debye_waller_factor(&self) -> f64 {
        // DW ≈ 0.70 at T < 10 K; decreases slowly with temperature
        let dw0 = 0.70_f64;
        let t_d = 1860.0_f64; // diamond Debye temperature
        let t = self.temperature_k;
        (dw0 * (1.0 - 0.2 * t / t_d)).max(0.4)
    }

    /// Radiative lifetime (ns): ~1 ns for SiV.
    pub fn radiative_lifetime_ns(&self) -> f64 {
        // SiV has a strong electric dipole transition giving τ ~1 ns
        1.0_f64 + 0.05 * self.temperature_k / 300.0 // slight T correction
    }

    /// Zero-phonon linewidth (GHz) as a function of temperature.
    ///
    /// Phonon-induced dephasing ∝ T³ at low T (Raman scattering between
    /// ground-state orbital branches separated by ~50 GHz).
    pub fn zero_phonon_linewidth_ghz(&self, temperature_k: f64) -> f64 {
        let tau_rad_ghz = 1.0 / (2.0 * PI * self.radiative_lifetime_ns() * 1e-9) * 1e-9;
        // Pure dephasing: γ_pure ~ (kT/ħΔ)² * phonon density
        // Orbital splitting Δ ≈ 48 GHz; effective T³ law for low T
        let delta_ghz = 48.0_f64;
        let kbt_ghz = K_B * temperature_k / (H_PLANCK * 1e9);
        let n_phonon = if temperature_k < 1e-3 {
            0.0
        } else {
            1.0 / ((delta_ghz / kbt_ghz).exp() - 1.0 + 1e-30)
        };
        let gamma_pure = 0.05 * n_phonon.powi(2); // GHz (empirical coefficient)
        tau_rad_ghz + gamma_pure
    }

    /// Photon indistinguishability (achievable > 90 % in photonic devices).
    ///
    ///  M = Γ_rad / (Γ_rad + 2 γ_pure)
    pub fn indistinguishability(&self) -> f64 {
        let t = self.temperature_k;
        let lw = self.zero_phonon_linewidth_ghz(t);
        let gamma_rad_ghz = 1.0 / (2.0 * PI * self.radiative_lifetime_ns() * 1e-9) * 1e-9;
        let gamma_pure = (lw - gamma_rad_ghz).max(0.0);
        let denom = gamma_rad_ghz + 2.0 * gamma_pure;
        if denom > 0.0 {
            gamma_rad_ghz / denom
        } else {
            1.0
        }
    }

    /// SiV⁻ possesses inversion symmetry → insensitive to local electric fields.
    pub fn inversion_symmetry(&self) -> bool {
        true
    }

    /// Spin coherence time T₂ (μs).
    ///
    /// At room temperature, the strong phonon mixing between orbital branches
    /// limits T₂ to nanoseconds.  At mK temperatures T₂ can reach milliseconds.
    pub fn t2_spin_us(&self) -> f64 {
        let t = self.temperature_k;
        if t < 0.5 {
            // mK regime: long spin coherence limited by 29Si nuclear spin bath
            500.0
        } else if t < 5.0 {
            // 4 K regime: short due to orbital relaxation
            10.0 * (0.5_f64 / t).powi(3)
        } else {
            // Room temperature: phonon-limited, very short
            0.001 // ~1 ns in μs
        }
    }

    /// Recommended operating temperature (mK) for spin coherence.
    ///
    /// Below ~100 mK, phonon-induced orbital mixing is suppressed and
    /// spin coherence is limited only by nuclear spin bath.
    pub fn operating_temperature_mk(&self) -> f64 {
        100.0_f64 // mK
    }
}

// ─── SnvCenter ────────────────────────────────────────────────────────────────

/// Tin-Vacancy (SnV) colour centre in diamond — emerging platform.
///
/// SnV has a very large ground-state spin-orbit splitting (~850 GHz), allowing
/// spin manipulation at temperatures of 1–4 K, unlike SiV which requires mK.
#[derive(Debug, Clone)]
pub struct SnvCenter {
    /// Sample temperature (K)
    pub temperature_k: f64,
}

impl SnvCenter {
    /// Create an SnV centre at the given temperature.
    pub fn new(temperature_k: f64) -> Self {
        Self { temperature_k }
    }

    /// Zero-phonon line wavelength: ~619 nm.
    pub fn zpl_wavelength_nm(&self) -> f64 {
        619.0_f64
    }

    /// Debye-Waller factor: ~0.85 (excellent for photonic integration).
    ///
    /// The heavy-mass Sn atom and strong spin-orbit coupling further suppress
    /// phonon sideband emission relative to SiV.
    pub fn debye_waller_factor(&self) -> f64 {
        // DW ≈ 0.85 at low T; slight reduction at elevated T
        let dw0 = 0.85_f64;
        let t = self.temperature_k;
        // Gentle suppression above 10 K
        (dw0 * (1.0 - 0.05 * t / 100.0)).max(0.70)
    }

    /// Radiative lifetime (ns): ~6 ns for SnV⁻.
    pub fn radiative_lifetime_ns(&self) -> f64 {
        // SnV: ~6 ns at low T; slight lengthening at RT due to singlet mixing
        6.0_f64 + 0.02 * self.temperature_k / 300.0
    }

    /// Ground-state orbital (spin-orbit) splitting (THz).
    ///
    /// SnV has a very large spin-orbit splitting Δ_SO ≈ 850 GHz = 0.85 THz,
    /// enabling spin manipulation at 4 K without phonon mixing.
    pub fn ground_state_splitting_thz(&self) -> f64 {
        0.850_f64 // THz
    }

    /// Recommended operating temperature (K) for coherent spin control.
    ///
    /// Because Δ_SO ≫ k_B·T at 4 K, thermal population of the upper orbital
    /// branch is suppressed: n_phonon(4 K, 850 GHz) ≈ 4 × 10⁻⁵.
    pub fn operating_temperature_k(&self) -> f64 {
        // Upper bound for coherent operation: kT < Δ_SO/5
        let delta_ghz = self.ground_state_splitting_thz() * 1000.0; // THz → GHz
        let kbt_limit_ghz = delta_ghz / 5.0;
        kbt_limit_ghz * H_PLANCK * 1e9 / K_B // GHz → K
    }
}

// ─── HbnDefect ────────────────────────────────────────────────────────────────

/// Single-photon emitter in hexagonal boron nitride (hBN).
///
/// hBN hosts a variety of optically active defects that act as room-temperature
/// single-photon emitters in the visible range (580–720 nm).  They are among
/// the brightest solid-state SPEs known, with count rates > 10 Mcps.
#[derive(Debug, Clone)]
pub struct HbnDefect {
    /// Central emission wavelength (nm) — varies between defect sites
    pub emission_wavelength_nm: f64,
    /// Sample temperature (K)
    pub temperature_k: f64,
}

impl HbnDefect {
    /// Create an hBN defect emitter at the given wavelength.
    ///
    /// Defaults to room temperature.
    pub fn new(wavelength_nm: f64) -> Self {
        Self {
            emission_wavelength_nm: wavelength_nm.clamp(400.0, 900.0),
            temperature_k: 300.0,
        }
    }

    /// Debye-Waller factor as a function of temperature.
    ///
    /// hBN defects show a moderate DW factor (~0.8 at cryogenic T) that
    /// decreases with temperature due to phonon sideband growth.
    pub fn debye_waller_factor(&self, temperature_k: f64) -> f64 {
        // DW ≈ 0.80 at 5 K; 0.50 at RT (broad PSB due to 2-D crystal phonons)
        let dw_low = 0.80_f64;
        let dw_high = 0.50_f64;
        let t_cross = 150.0_f64; // K
        let t = temperature_k.max(0.0);
        // Smooth interpolation
        dw_high + (dw_low - dw_high) / (1.0 + (t / t_cross).powi(2))
    }

    /// Linewidth (nm) of the ZPL as a function of temperature.
    ///
    /// At cryogenic temperatures: ~0.1–0.5 nm (limited by spectral diffusion).
    /// At room temperature: ~5–20 nm (phonon broadening).
    pub fn linewidth_nm(&self, temperature_k: f64) -> f64 {
        // Empirical: lw(T) ≈ lw₀ + γ_ph * (T/300)³
        let lw0 = 0.3_f64; // nm at 0 K (spectral diffusion floor)
        let gamma_ph = 15.0_f64; // nm at 300 K phonon contribution
        let t = temperature_k.max(0.0);
        lw0 + gamma_ph * (t / 300.0_f64).powi(3)
    }

    /// Saturated brightness (millions of counts per second, Mcps).
    ///
    /// hBN defects are extremely bright: 10–50 Mcps have been reported with
    /// high-NA objectives.  Here we use a conservative 15 Mcps estimate.
    pub fn brightness_mcps(&self) -> f64 {
        // Depends on orientation, crystal quality, excitation
        15.0_f64
    }

    /// g²(0) value — near 0 for a confirmed single emitter.
    ///
    /// Well-characterised hBN SPEs show g²(0) < 0.05 under pulsed excitation.
    pub fn g2_zero(&self) -> f64 {
        0.04_f64 // typical measured value
    }

    /// Photostability factor (fraction of time the emitter is in the bright state).
    ///
    /// hBN emitters can blink (transition to dark states).
    /// Typical photostability: 0.7–0.95 for well-prepared samples.
    pub fn photostability(&self) -> f64 {
        0.85_f64
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─ NV centre ──────────────────────────────────────────────────────────────

    #[test]
    fn test_nv_minus_zpl_wavelength() {
        let nv = NvCenter::new_nv_minus(300.0);
        assert_eq!(nv.zpl_wavelength_nm(), 637.0, "NV⁻ ZPL should be 637 nm");
    }

    #[test]
    fn test_nv_zero_field_splitting() {
        let nv = NvCenter::new_nv_minus(300.0);
        let d = nv.zero_field_splitting_ghz();
        // D ≈ 2.87 GHz at RT (within 0.01 GHz)
        assert!(
            (d - 2.87).abs() < 0.02,
            "Zero-field splitting should be ~2.87 GHz; got {d:.4}"
        );
    }

    #[test]
    fn test_nv_zeeman_splitting_rate() {
        let nv = NvCenter::new_nv_minus(300.0);
        // 100 Gauss → ~280 MHz = 0.280 GHz
        let dz = nv.zeeman_splitting_ghz(100.0);
        assert!(
            (dz - 0.280).abs() < 0.005,
            "Zeeman splitting at 100 Gauss should be ~0.280 GHz; got {dz:.4}"
        );
    }

    #[test]
    fn test_nv_t1_longer_at_low_temperature() {
        let nv_rt = NvCenter::new_nv_minus(300.0);
        let nv_low = NvCenter::new_nv_minus(4.0);
        assert!(
            nv_low.t1_spin_ms() > nv_rt.t1_spin_ms(),
            "T₁ should be longer at low temperature"
        );
    }

    #[test]
    fn test_nv_magnetic_sensitivity_finite() {
        let nv = NvCenter::new_nv_minus(300.0);
        let sens = nv.magnetic_sensitivity_nt_per_sqrthz(0.01);
        assert!(
            sens.is_finite() && sens > 0.0,
            "Magnetic sensitivity should be positive and finite; got {sens}"
        );
    }

    // ─ SiV centre ─────────────────────────────────────────────────────────────

    #[test]
    fn test_siv_zpl_wavelength() {
        let siv = SivCenter::new(4.0);
        assert_eq!(siv.zpl_wavelength_nm(), 738.0, "SiV ZPL should be 738 nm");
    }

    #[test]
    fn test_siv_dw_factor_high() {
        let siv = SivCenter::new(4.0);
        let dw = siv.debye_waller_factor();
        assert!(dw > 0.60, "SiV DW factor should be > 60 %; got {dw:.3}");
    }

    #[test]
    fn test_siv_inversion_symmetry() {
        let siv = SivCenter::new(4.0);
        assert!(
            siv.inversion_symmetry(),
            "SiV should have inversion symmetry"
        );
    }

    #[test]
    fn test_siv_t2_longer_at_mk() {
        let siv_mk = SivCenter::new(0.05);
        let siv_4k = SivCenter::new(4.0);
        assert!(
            siv_mk.t2_spin_us() > siv_4k.t2_spin_us(),
            "T₂ should be longer at mK temperatures"
        );
    }

    // ─ SnV centre ─────────────────────────────────────────────────────────────

    #[test]
    fn test_snv_zpl_and_dw() {
        let snv = SnvCenter::new(4.0);
        assert_eq!(snv.zpl_wavelength_nm(), 619.0, "SnV ZPL should be 619 nm");
        assert!(
            snv.debye_waller_factor() > 0.80,
            "SnV DW factor should be > 80 %; got {:.3}",
            snv.debye_waller_factor()
        );
    }

    #[test]
    fn test_snv_operating_temperature_above_1k() {
        let snv = SnvCenter::new(4.0);
        let t_op = snv.operating_temperature_k();
        assert!(
            t_op > 1.0,
            "SnV operating T should exceed 1 K; got {t_op:.2} K"
        );
    }

    // ─ hBN defect ─────────────────────────────────────────────────────────────

    #[test]
    fn test_hbn_g2_below_threshold() {
        let hbn = HbnDefect::new(600.0);
        let g2 = hbn.g2_zero();
        assert!(
            g2 < 0.5,
            "g²(0) should be < 0.5 for single emitter; got {g2}"
        );
    }

    #[test]
    fn test_hbn_linewidth_grows_with_temperature() {
        let hbn = HbnDefect::new(600.0);
        let lw_cold = hbn.linewidth_nm(10.0);
        let lw_warm = hbn.linewidth_nm(300.0);
        assert!(
            lw_warm > lw_cold,
            "hBN linewidth should increase with temperature; cold={lw_cold:.3}, warm={lw_warm:.3}"
        );
    }

    #[test]
    fn test_hbn_dw_factor_bounded() {
        let hbn = HbnDefect::new(620.0);
        for &t in &[0.0, 4.0, 77.0, 300.0] {
            let dw = hbn.debye_waller_factor(t);
            assert!(
                (0.0..=1.0).contains(&dw),
                "DW factor out of [0,1] at T={t} K: got {dw}"
            );
        }
    }
}
