//! Optical channel model: crosstalk, dispersion, and noise.
//!
//! Models the impairments accumulated as light propagates from transmitter
//! to receiver in a photonic interconnect:
//!
//!   1. Chromatic dispersion: D (ps/nm/km) causes pulse broadening
//!   2. Polarisation mode dispersion (PMD): DGD ~ PMD_coeff × √L
//!   3. Optical amplifier noise: ASE noise from EDFAs
//!   4. Crosstalk: XT between WDM channels or waveguide crossings
//!   5. Nonlinear impairments: SPM, XPM, FWM
//!
//! Noise-signal ratio (NSR) accumulates as:
//!   OSNR = P_signal / P_ASE  (optical SNR, usually measured in 0.1 nm BW)

/// Optical channel model.
#[derive(Debug, Clone)]
pub struct OpticalChannel {
    /// Total fiber length (km)
    pub length_km: f64,
    /// Chromatic dispersion D (ps/nm/km)
    pub dispersion_ps_nm_km: f64,
    /// Dispersion slope S (ps/nm²/km)
    pub dispersion_slope_ps_nm2_km: f64,
    /// Fiber attenuation α (dB/km)
    pub attenuation_db_per_km: f64,
    /// PMD coefficient (ps/√km)
    pub pmd_coeff_ps_sqrt_km: f64,
    /// Number of optical amplifiers
    pub n_amplifiers: usize,
    /// EDFA noise figure (dB)
    pub edfa_nf_db: f64,
    /// EDFA gain (dB) — one per span
    pub edfa_gain_db: f64,
    /// Crosstalk level (linear, -∞ to 0)
    pub crosstalk_linear: f64,
}

impl OpticalChannel {
    /// Create a channel model.
    pub fn new(length_km: f64, dispersion_ps_nm_km: f64, attenuation_db_per_km: f64) -> Self {
        Self {
            length_km,
            dispersion_ps_nm_km,
            dispersion_slope_ps_nm2_km: 0.06,
            attenuation_db_per_km,
            pmd_coeff_ps_sqrt_km: 0.04,
            n_amplifiers: 0,
            edfa_nf_db: 5.0,
            edfa_gain_db: 20.0,
            crosstalk_linear: 0.0,
        }
    }

    /// Standard SMF-28 channel, 80 km unamplified.
    pub fn smf28_80km() -> Self {
        Self::new(80.0, 17.0, 0.2)
    }

    /// Metro link with EDFA amplification (4 × 80 km spans).
    pub fn metro_4_span() -> Self {
        let mut c = Self::new(320.0, 17.0, 0.2);
        c.n_amplifiers = 3;
        c.edfa_gain_db = 16.0; // 0.2 dB/km × 80 km = 16 dB
        c
    }

    /// On-chip SiPh channel (short distance, high loss).
    pub fn on_chip_siph_10mm() -> Self {
        let mut c = Self::new(0.01e-3, 1000.0, 2000.0); // 2 dB/cm = 200,000 dB/km
        c.pmd_coeff_ps_sqrt_km = 0.0; // negligible PMD on-chip
        c
    }

    /// Total chromatic dispersion (ps/nm).
    pub fn total_dispersion_ps_nm(&self) -> f64 {
        self.dispersion_ps_nm_km * self.length_km
    }

    /// Accumulated dispersion at a channel offset Δλ (nm) from center.
    ///
    ///   D(λ) = D(λ₀) + S·Δλ  (linear approximation)
    pub fn dispersion_at_offset(&self, delta_lambda_nm: f64) -> f64 {
        let d = self.dispersion_ps_nm_km + self.dispersion_slope_ps_nm2_km * delta_lambda_nm;
        d * self.length_km
    }

    /// Differential group delay DGD (ps) — average PMD.
    ///
    ///   DGD = PMD_coeff × √L
    pub fn dgd_ps(&self) -> f64 {
        self.pmd_coeff_ps_sqrt_km * self.length_km.sqrt()
    }

    /// Total attenuation (dB).
    pub fn total_attenuation_db(&self) -> f64 {
        self.attenuation_db_per_km * self.length_km
    }

    /// Optical signal-to-noise ratio (OSNR in dB) after N amplifiers.
    ///
    /// OSNR = P_signal - 10·log10(n_amp · P_ASE_per_amp · B_ref)
    ///
    /// For cascaded EDFAs with equal gain/loss: OSNR ≈ P_launch - NF - 10·log10(n_amp) - α·L_span + G
    /// Simplified formula (per-span EDFA system):
    ///   OSNR ≈ P_launch_dBm - NF_dB - 10·log10(h·ν·B_ref) - 10·log10(n_amp) - α·L_span_dB
    pub fn osnr_db(&self, p_launch_dbm: f64, wavelength_nm: f64) -> f64 {
        if self.n_amplifiers == 0 {
            // Unamplified: no ASE noise, OSNR → ∞ (thermal noise dominates)
            return 100.0;
        }
        let h = 6.626e-34;
        let c = 3e8;
        let nu = c / (wavelength_nm * 1e-9); // Hz
        let b_ref = 12.5e9; // 0.1 nm reference BW at 1550 nm
        let nf = 10.0_f64.powf(self.edfa_nf_db / 10.0);
        let p_ase_per_amp = nf * h * nu * b_ref; // W
        let p_ase_per_amp_dbm = 10.0 * p_ase_per_amp.log10() + 30.0;
        let n_amp_db = 10.0 * (self.n_amplifiers as f64).log10();
        p_launch_dbm - p_ase_per_amp_dbm - n_amp_db
    }

    /// ISI (inter-symbol interference) penalty (dB) from chromatic dispersion.
    ///
    /// For NRZ at bit rate B (Gbps):
    ///   penalty ≈ -5·log₁₀(1 - (D·L·Δλ·B)²)
    pub fn isi_penalty_db(&self, data_rate_gbps: f64, source_linewidth_nm: f64) -> f64 {
        let d_total = self.total_dispersion_ps_nm() * 1e-12; // s/nm
        let dl = source_linewidth_nm; // nm
        let b = data_rate_gbps * 1e9;
        let eps = (d_total * dl * b).powi(2);
        if eps >= 1.0 {
            return 20.0;
        } // very large penalty
        -5.0 * (1.0 - eps).log10()
    }

    /// Crosstalk penalty (dB): penalty for linear power crosstalk XT_linear.
    ///
    ///   penalty ≈ -10·log10(1 - XT)
    pub fn crosstalk_penalty_db(&self) -> f64 {
        if self.crosstalk_linear <= 0.0 {
            return 0.0;
        }
        if self.crosstalk_linear >= 1.0 {
            return 30.0;
        }
        -10.0 * (1.0 - self.crosstalk_linear).log10()
    }

    /// Q-factor from OSNR (for NRZ): Q ≈ √(OSNR × 2·B_opt/B_elec).
    pub fn q_from_osnr_db(&self, osnr_db: f64, b_opt_ghz: f64, b_elec_ghz: f64) -> f64 {
        let osnr = 10.0_f64.powf(osnr_db / 10.0);
        (osnr * 2.0 * b_opt_ghz / b_elec_ghz).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_total_dispersion_scales_with_length() {
        let c = OpticalChannel::smf28_80km();
        let d = c.total_dispersion_ps_nm();
        assert!((d - 17.0 * 80.0).abs() < 1e-6);
    }

    #[test]
    fn channel_dgd_positive() {
        let c = OpticalChannel::smf28_80km();
        assert!(c.dgd_ps() > 0.0);
    }

    #[test]
    fn channel_attenuation_positive() {
        let c = OpticalChannel::smf28_80km();
        assert!(c.total_attenuation_db() > 0.0);
    }

    #[test]
    fn channel_osnr_unamplified_high() {
        let c = OpticalChannel::smf28_80km();
        let osnr = c.osnr_db(0.0, 1550.0);
        assert!(osnr > 50.0, "OSNR={osnr:.1}dB");
    }

    #[test]
    fn channel_osnr_with_amps_reasonable() {
        let c = OpticalChannel::metro_4_span();
        let osnr = c.osnr_db(0.0, 1550.0);
        assert!(osnr > 10.0 && osnr < 50.0, "OSNR={osnr:.1}dB");
    }

    #[test]
    fn channel_isi_penalty_zero_for_zero_dispersion() {
        let mut c = OpticalChannel::smf28_80km();
        c.dispersion_ps_nm_km = 0.0;
        let penalty = c.isi_penalty_db(10.0, 0.01);
        assert!(penalty.abs() < 0.01);
    }

    #[test]
    fn channel_crosstalk_penalty_zero_when_no_xt() {
        let c = OpticalChannel::smf28_80km();
        assert!(c.crosstalk_penalty_db() == 0.0);
    }

    #[test]
    fn channel_dispersion_slope_at_offset() {
        let c = OpticalChannel::smf28_80km();
        let d0 = c.dispersion_at_offset(0.0);
        let d1 = c.dispersion_at_offset(10.0);
        assert!(
            d1 > d0,
            "Positive slope: d(Δλ=10nm)={d1:.1} > d(Δλ=0)={d0:.1}"
        );
    }
}
