/// Frequency comb-based spectroscopy and attosecond science.
///
/// Covers:
/// - Dual-comb spectroscopy (DCS): simultaneous broadband absorption spectroscopy
///   with comb-tooth-resolved spectral resolution
/// - Direct frequency comb spectroscopy (DFCS) with enhancement cavities
/// - High-harmonic generation (HHG) and attosecond pulse characterization
use super::comb::{FrequencyComb, C0};

// ─── DualCombSpectroscopy ────────────────────────────────────────────────────

/// Dual-comb spectroscopy (DCS) system.
///
/// Two frequency combs with slightly different repetition rates f_rep1 and
/// f_rep2 interfere on a photodetector.  The resulting RF beat comb has teeth
/// spaced by |Δf_rep| and maps the optical spectrum onto the RF domain through
/// a Doppler-free down-conversion.
///
/// The system achieves simultaneously high spectral resolution (limited by f_rep),
/// broad spectral coverage (limited by comb bandwidth), and microsecond-scale
/// acquisition times.
#[derive(Debug, Clone)]
pub struct DualCombSpectroscopy {
    /// Signal comb — interrogates the sample.
    pub comb1: FrequencyComb,
    /// Local oscillator comb — provides the reference.
    pub comb2: FrequencyComb,
    /// Repetition rate difference |f_rep1 − f_rep2| (Hz).
    pub delta_frep: f64,
}

impl DualCombSpectroscopy {
    /// Construct a DCS system from two combs.
    ///
    /// The repetition rate difference is computed automatically as |f_rep1 − f_rep2|.
    /// The combs must have similar center wavelengths for the technique to work.
    ///
    /// # Arguments
    /// * `comb1` — signal comb
    /// * `comb2` — local oscillator comb
    pub fn new(comb1: FrequencyComb, comb2: FrequencyComb) -> Self {
        let delta_frep = (comb1.f_rep - comb2.f_rep).abs();
        Self {
            comb1,
            comb2,
            delta_frep,
        }
    }

    /// RF comb tooth spacing at the detector: Δf_rep (Hz).
    pub fn rf_comb_spacing_hz(&self) -> f64 {
        self.delta_frep
    }

    /// Acquisition time for one complete spectral sweep: T_acq = 1 / Δf_rep (s).
    ///
    /// A single interferogram of duration T_acq contains one complete mapping
    /// of the optical comb spectrum onto the RF comb.
    pub fn acquisition_time_s(&self) -> f64 {
        if self.delta_frep <= 0.0 {
            return f64::INFINITY;
        }
        1.0 / self.delta_frep
    }

    /// Number of resolved comb tooth pairs (useful spectral points).
    ///
    /// Limited by the Nyquist condition: the RF comb must not alias, so the
    /// number of resolvable teeth is N = f_rep / (2 · Δf_rep).
    pub fn n_resolved_teeth(&self) -> usize {
        if self.delta_frep <= 0.0 {
            return 0;
        }
        let n = (self.comb1.f_rep / (2.0 * self.delta_frep)).floor() as usize;
        n.max(1)
    }

    /// Spectral resolution (GHz) — set by the repetition rate f_rep.
    ///
    /// Each comb tooth corresponds to one resolved spectral channel with
    /// frequency spacing f_rep.
    pub fn spectral_resolution_ghz(&self) -> f64 {
        self.comb1.f_rep * 1e-9 // Hz → GHz
    }

    /// Minimum detectable absorbance per unit integration time.
    ///
    /// Uses the noise-equivalent absorbance formula for shot-noise limited DCS:
    /// A_min = 1 / (SNR · √(N · T_acq · B))
    ///
    /// where N is the number of teeth, B is the bandwidth per tooth, and the
    /// factor √(N) accounts for simultaneous multi-channel detection.
    ///
    /// # Arguments
    /// * `integration_time_s` — total averaging time (s)
    /// * `snr_per_comb`       — signal-to-noise ratio per comb tooth per acquisition
    pub fn min_detectable_absorbance(&self, integration_time_s: f64, snr_per_comb: f64) -> f64 {
        let n_teeth = self.n_resolved_teeth() as f64;
        let t_acq = self.acquisition_time_s();
        // Number of independent averages
        let n_avg = (integration_time_s / t_acq.max(1e-12)).max(1.0);
        // A_min = 1 / (SNR * sqrt(N * n_avg))
        1.0 / (snr_per_comb * (n_teeth * n_avg).sqrt())
    }

    /// Figure of merit: M = N · T_acq · SNR² (teeth × seconds × sensitivity²).
    ///
    /// This metric quantifies the advantage of DCS over single-channel methods.
    /// Larger M indicates more simultaneous spectral points per unit acquisition time.
    ///
    /// # Arguments
    /// * `snr_per_comb` — signal-to-noise ratio per tooth
    pub fn figure_of_merit(&self, snr_per_comb: f64) -> f64 {
        let n = self.n_resolved_teeth() as f64;
        let t = self.acquisition_time_s();
        n * t * snr_per_comb * snr_per_comb
    }

    /// Mutual coherence requirement for the two combs (Hz).
    ///
    /// The linewidths of both combs must satisfy:
    /// δν_1, δν_2  <  Δf_rep² / f_rep
    ///
    /// to avoid aliasing of adjacent RF comb teeth.
    /// Returns the maximum allowable individual comb linewidth (Hz).
    pub fn coherence_requirement_hz(&self) -> f64 {
        let f_rep = self.comb1.f_rep;
        if f_rep <= 0.0 {
            return 0.0;
        }
        self.delta_frep * self.delta_frep / f_rep
    }

    /// Instantaneous spectral coverage (nm) set by the comb bandwidth.
    pub fn spectral_coverage_nm(&self) -> f64 {
        self.comb1.bandwidth_nm
    }

    /// Sampling theorem check: returns true if the two combs satisfy the
    /// condition for alias-free mapping of the optical spectrum onto RF.
    ///
    /// Condition: Δf_rep < f_rep / N_teeth — i.e., the RF beat notes fit
    /// within the Nyquist window without overlap.
    pub fn is_alias_free(&self) -> bool {
        let n = self.n_resolved_teeth() as f64;
        self.delta_frep * n < self.comb1.f_rep / 2.0
    }
}

// ─── DirectCombSpectroscopy ──────────────────────────────────────────────────

/// Direct frequency comb spectroscopy (DFCS) with a passive enhancement cavity.
///
/// The comb is mode-matched and coupled into a high-finesse Fabry-Pérot cavity.
/// Each comb mode resonates independently (cavity-comb locking), multiplying
/// the single-pass absorption by the effective finesse.
#[derive(Debug, Clone)]
pub struct DirectCombSpectroscopy {
    /// Frequency comb source.
    pub comb: FrequencyComb,
    /// Cavity finesse F.
    pub cavity_finesse: f64,
    /// Physical path length of the enhancement cavity (m).
    pub interaction_length_m: f64,
}

impl DirectCombSpectroscopy {
    /// Construct a DFCS system.
    ///
    /// # Arguments
    /// * `comb`    — frequency comb source
    /// * `finesse` — cavity finesse (dimensionless)
    pub fn new(comb: FrequencyComb, finesse: f64) -> Self {
        // Interaction length defaults to the cavity round-trip path length
        let interaction_length_m = C0 / (2.0 * comb.f_rep); // half-wave round trip
        Self {
            comb,
            cavity_finesse: finesse,
            interaction_length_m,
        }
    }

    /// Power enhancement factor inside the cavity: F_enh = F / π.
    pub fn enhancement_factor(&self) -> f64 {
        self.cavity_finesse / std::f64::consts::PI
    }

    /// Effective optical path length (m): L_eff = 2 F L / π.
    ///
    /// Both the forward and backward passes are counted; the total equals
    /// 2 cavity lengths × enhancement factor.
    pub fn effective_path_length_m(&self) -> f64 {
        2.0 * self.cavity_finesse * self.interaction_length_m / std::f64::consts::PI
    }

    /// Minimum detectable concentration (molecules/cm³).
    ///
    /// From Beer-Lambert law: A_min = σ · N · L_eff
    /// Solve for N: N_min = A_min / (σ · L_eff)
    ///
    /// Assumes a minimum detectable absorbance of 10⁻⁶ per tooth (shot-noise).
    ///
    /// # Arguments
    /// * `cross_section_cm2` — absorption cross-section of the target molecule (cm²)
    pub fn min_detectable_concentration(&self, cross_section_cm2: f64) -> f64 {
        let a_min = 1e-6; // dimensionless minimum detectable absorbance per tooth
        let l_eff_cm = self.effective_path_length_m() * 100.0; // m → cm
        if cross_section_cm2 <= 0.0 || l_eff_cm <= 0.0 {
            return f64::INFINITY;
        }
        a_min / (cross_section_cm2 * l_eff_cm) // molecules/cm³
    }

    /// Cavity free spectral range (Hz): FSR = c / (2 L).
    pub fn cavity_fsr_hz(&self) -> f64 {
        C0 / (2.0 * self.interaction_length_m)
    }

    /// Check that the comb repetition rate matches the cavity FSR.
    ///
    /// Returns the fractional mismatch |f_rep − FSR| / FSR.
    pub fn rep_rate_mismatch(&self) -> f64 {
        let fsr = self.cavity_fsr_hz();
        if fsr <= 0.0 {
            return f64::INFINITY;
        }
        (self.comb.f_rep - fsr).abs() / fsr
    }
}

// ─── HhgGas ─────────────────────────────────────────────────────────────────

/// Noble gas target for high-harmonic generation (HHG).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HhgGas {
    /// Helium — highest cutoff, lowest cross-section.
    Helium,
    /// Neon — high cutoff, moderate efficiency.
    Neon,
    /// Argon — most commonly used; good balance of cutoff and efficiency.
    Argon,
    /// Krypton — lower cutoff, higher efficiency than Ar.
    Krypton,
    /// Xenon — lowest cutoff, highest HHG efficiency.
    Xenon,
}

impl HhgGas {
    /// Ionization potential (eV) — determines the HHG minimum photon energy.
    pub fn ionization_potential_ev(&self) -> f64 {
        match self {
            HhgGas::Helium => 24.59,
            HhgGas::Neon => 21.56,
            HhgGas::Argon => 15.76,
            HhgGas::Krypton => 14.00,
            HhgGas::Xenon => 12.13,
        }
    }

    /// Estimate of the highest significant harmonic order.
    ///
    /// Uses the three-step model cutoff law:
    /// E_cut = Ip + 3.17 · Up  where  Up ∝ I · λ²
    ///
    /// Returns the harmonic order `n = E_cut / ħω`.
    ///
    /// # Arguments
    /// * `intensity_wcm2` — peak laser intensity (W/cm²)
    /// * `wavelength_nm`  — driving laser wavelength (nm)
    pub fn cutoff_harmonic_estimate(&self, intensity_wcm2: f64, wavelength_nm: f64) -> usize {
        let ip_ev = self.ionization_potential_ev();
        // Up [eV] = 9.33e-14 · I[W/cm²] · (λ[μm])²
        let lambda_um = wavelength_nm * 1e-3; // nm → μm
        let up_ev = 9.33e-14 * intensity_wcm2 * lambda_um * lambda_um;
        let e_cut_ev = ip_ev + 3.17 * up_ev;
        // Photon energy ħω [eV] = h c / λ = 1240 / λ[nm]
        let h_omega_ev = 1240.0 / wavelength_nm;
        if h_omega_ev <= 0.0 {
            return 1;
        }
        let harmonic = (e_cut_ev / h_omega_ev).floor() as usize;
        harmonic.max(1)
    }
}

// ─── HhgSource ───────────────────────────────────────────────────────────────

/// High-harmonic generation (HHG) attosecond pulse source.
///
/// HHG is a strong-field process where an intense femtosecond laser pulse
/// ionizes a noble gas, the freed electron is accelerated in the laser field,
/// and recollides with the parent ion emitting an attosecond XUV burst.
///
/// The three-step model predicts a plateau of harmonics up to a cutoff energy
/// E_cut = Ip + 3.17 Up, where Up is the ponderomotive energy.
#[derive(Debug, Clone)]
pub struct HhgSource {
    /// Driving laser wavelength (nm). Commonly 800 nm (Ti:Sa) or 1030 nm (Yb fiber).
    pub drive_wavelength_nm: f64,
    /// Peak laser intensity at the focus (W/cm²).
    pub drive_intensity_wcm2: f64,
    /// Noble gas target for HHG.
    pub target_gas: HhgGas,
}

impl HhgSource {
    /// Construct an HHG source.
    ///
    /// # Arguments
    /// * `wavelength_nm`   — driving laser wavelength (nm)
    /// * `intensity_wcm2`  — peak focus intensity (W/cm²)
    /// * `gas`             — target noble gas
    pub fn new(wavelength_nm: f64, intensity_wcm2: f64, gas: HhgGas) -> Self {
        Self {
            drive_wavelength_nm: wavelength_nm,
            drive_intensity_wcm2: intensity_wcm2,
            target_gas: gas,
        }
    }

    /// Standard 800 nm / Argon HHG setup for ≤100 eV attosecond pulses.
    pub fn new_ti_sa_argon() -> Self {
        Self::new(800.0, 2e14, HhgGas::Argon)
    }

    /// Ponderomotive energy Up (eV).
    ///
    /// Up = e² E₀² / (4 mₑ ω²)  →  Up \[eV\] = 9.33 × 10⁻¹⁴ · I \[W/cm²\] · λ \[μm\]²
    pub fn ponderomotive_energy_ev(&self) -> f64 {
        let lambda_um = self.drive_wavelength_nm * 1e-3;
        9.33e-14 * self.drive_intensity_wcm2 * lambda_um * lambda_um
    }

    /// HHG cutoff photon energy (eV): E_cut = Ip + 3.17 · Up.
    pub fn cutoff_energy_ev(&self) -> f64 {
        let ip = self.target_gas.ionization_potential_ev();
        let up = self.ponderomotive_energy_ev();
        ip + 3.17 * up
    }

    /// Highest harmonic order within the plateau.
    pub fn max_harmonic_order(&self) -> usize {
        self.target_gas
            .cutoff_harmonic_estimate(self.drive_intensity_wcm2, self.drive_wavelength_nm)
    }

    /// Estimated attosecond pulse duration (as).
    ///
    /// The intrinsic duration of an isolated attosecond pulse generated near the
    /// HHG cutoff is approximately:
    /// τ_as ≈ (1/3) · T_laser / n_cutoff
    ///
    /// where T_laser = λ/c and n_cutoff is the harmonic order at the cutoff.
    /// Returns the duration in attoseconds (as).
    pub fn attosecond_pulse_duration_as(&self) -> f64 {
        let n_cut = self.max_harmonic_order().max(1) as f64;
        // T_laser in attoseconds: T = λ[nm] / c[nm/as] where c ≈ 0.3 nm/as
        let t_laser_as = self.drive_wavelength_nm / (C0 * 1e9 / 1e18); // nm / (nm/as) = as
                                                                       // τ_as ≈ T_laser / (3 * n_cut)
        t_laser_as / (3.0 * n_cut)
    }

    /// Phase-matching pressure estimate (mbar).
    ///
    /// Optimal phase matching in the plateau occurs when the neutral gas
    /// dispersion balances the plasma dispersion plus the geometric (Gouy) phase.
    /// An approximate empirical formula for Ar near 800 nm gives:
    /// P_match ≈ 20 mbar · (I / 2e14)^0.5
    pub fn phase_matching_pressure_mbar(&self) -> f64 {
        let i_ref = 2e14_f64; // W/cm²
        let base_pressure = match self.target_gas {
            HhgGas::Helium => 200.0,
            HhgGas::Neon => 80.0,
            HhgGas::Argon => 20.0,
            HhgGas::Krypton => 10.0,
            HhgGas::Xenon => 5.0,
        };
        base_pressure * (self.drive_intensity_wcm2 / i_ref).sqrt()
    }

    /// Single attosecond pulse (SAP) or attosecond pulse train?
    ///
    /// A single attosecond pulse can be isolated by:
    /// - Polarization gating (few-cycle pulses < 2 optical cycles)
    /// - Ionization gating
    /// - Amplitude gating (half-cycle-cutoff)
    ///
    /// Returns true if the driving pulse duration is short enough to produce a SAP.
    /// Threshold: pulse duration < 2 optical cycles.
    ///
    /// # Arguments
    /// * `drive_duration_fs` — driving pulse duration FWHM (fs)
    pub fn is_single_attosecond_pulse(&self, drive_duration_fs: f64) -> bool {
        // 2 optical cycles duration
        let t_laser_fs = self.drive_wavelength_nm * 1e-9 / C0 * 1e15; // optical period in fs
        drive_duration_fs < 2.0 * t_laser_fs
    }

    /// XUV photon flux (photons/s/eV) at the cutoff energy.
    ///
    /// Uses an empirical scaling: Φ ≈ 10^10 photons/s/eV for Ar at 2×10¹⁴ W/cm².
    /// Scales as I^(n/2) with n ≈ 4 for below-threshold harmonics.
    ///
    /// # Arguments
    /// * `photon_energy_ev` — XUV photon energy of interest (eV)
    pub fn xuv_photon_flux(&self, photon_energy_ev: f64) -> f64 {
        let e_cut = self.cutoff_energy_ev();
        let ip = self.target_gas.ionization_potential_ev();
        if photon_energy_ev > e_cut || photon_energy_ev < ip {
            return 0.0;
        }
        // Plateau flux (roughly constant between Ip and E_cut)
        let base_flux = 1e10_f64; // photons/s/eV (typical)
                                  // Scale with intensity relative to reference
        let i_ref = 2e14_f64;
        base_flux * (self.drive_intensity_wcm2 / i_ref).sqrt()
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_dcs_rf_spacing_is_delta_frep() {
        let c1 = FrequencyComb::new_ti_sapphire(100_000_000.0, 10e6);
        let c2 = FrequencyComb::new_ti_sapphire(100_001_000.0, 11e6); // Δf_rep = 1 kHz
        let dcs = DualCombSpectroscopy::new(c1, c2);
        assert_abs_diff_eq!(dcs.rf_comb_spacing_hz(), 1000.0, epsilon = 0.01);
    }

    #[test]
    fn test_dcs_acquisition_time() {
        let c1 = FrequencyComb::new_ti_sapphire(100e6, 10e6);
        let c2 = FrequencyComb::new_ti_sapphire(100e6 + 100.0, 11e6); // Δf_rep = 100 Hz
        let dcs = DualCombSpectroscopy::new(c1, c2);
        // T_acq = 1 / 100 Hz = 10 ms
        assert_abs_diff_eq!(dcs.acquisition_time_s(), 0.01, epsilon = 1e-6);
    }

    #[test]
    fn test_dcs_coherence_requirement_positive() {
        let c1 = FrequencyComb::new_erbium_fiber(250e6, 0.0);
        let c2 = FrequencyComb::new_erbium_fiber(250e6 + 500.0, 0.0);
        let dcs = DualCombSpectroscopy::new(c1, c2);
        let req = dcs.coherence_requirement_hz();
        assert!(req > 0.0, "coherence requirement must be positive: {req}");
        // Δf_rep² / f_rep = 500² / 250e6 = 250000 / 250e6 = 1 mHz
        let expected = 500.0 * 500.0 / 250e6;
        assert_abs_diff_eq!(req, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_dfcs_enhancement_factor() {
        let comb = FrequencyComb::new_ti_sapphire(100e6, 0.0);
        let dfcs = DirectCombSpectroscopy::new(comb, 10_000.0); // F = 10,000
        let enh = dfcs.enhancement_factor();
        assert_abs_diff_eq!(enh, 10_000.0 / std::f64::consts::PI, epsilon = 0.001);
    }

    #[test]
    fn test_dfcs_effective_path_length_greater_than_physical() {
        let comb = FrequencyComb::new_ti_sapphire(100e6, 0.0);
        let dfcs = DirectCombSpectroscopy::new(comb, 1000.0);
        assert!(
            dfcs.effective_path_length_m() > dfcs.interaction_length_m,
            "effective path {:.3} m must exceed physical length {:.3} m",
            dfcs.effective_path_length_m(),
            dfcs.interaction_length_m
        );
    }

    #[test]
    fn test_hhg_ponderomotive_energy_argon_800nm() {
        // Standard case: I = 2e14 W/cm², λ = 800 nm → Up = 9.33e-14 * 2e14 * 0.64 ≈ 11.95 eV
        let src = HhgSource::new_ti_sa_argon();
        let up = src.ponderomotive_energy_ev();
        let expected = 9.33e-14 * 2e14 * 0.8_f64.powi(2); // λ in μm: 0.8 μm
        assert_abs_diff_eq!(up, expected, epsilon = 0.01);
    }

    #[test]
    fn test_hhg_cutoff_energy_exceeds_ip() {
        let src = HhgSource::new_ti_sa_argon();
        let e_cut = src.cutoff_energy_ev();
        let ip = src.target_gas.ionization_potential_ev();
        assert!(e_cut > ip, "cutoff {e_cut} eV must exceed Ip={ip} eV");
    }

    #[test]
    fn test_hhg_gas_ionization_potentials_ordering() {
        // He > Ne > Ar > Kr > Xe
        let gases = [
            HhgGas::Helium,
            HhgGas::Neon,
            HhgGas::Argon,
            HhgGas::Krypton,
            HhgGas::Xenon,
        ];
        for pair in gases.windows(2) {
            assert!(
                pair[0].ionization_potential_ev() > pair[1].ionization_potential_ev(),
                "{:?} Ip should exceed {:?} Ip",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn test_hhg_attosecond_duration_positive() {
        let src = HhgSource::new_ti_sa_argon();
        let tau = src.attosecond_pulse_duration_as();
        assert!(tau > 0.0, "attosecond duration must be positive: {tau} as");
        // Should be in the range 10 – 1000 as for typical conditions
        assert!(tau < 1000.0, "duration {tau} as unreasonably large");
    }
}
