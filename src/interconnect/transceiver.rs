//! Optical transceiver models for photonic interconnects.
//!
//! A transceiver includes:
//!   - Transmitter (TX): laser + modulator → optical signal
//!   - Receiver (RX): photodetector + TIA + decision circuit
//!
//! Modulation formats:
//!   - NRZ (Non-Return-to-Zero): 2-level, simplest
//!   - PAM-4 (4-level Pulse Amplitude Modulation): 2 bits/symbol, 2× spectral efficiency
//!   - PAM-8: 3 bits/symbol (emerging for >100G)
//!   - Coherent DP-QPSK: 4 bits/symbol (long-haul)
//!
//! Key performance metrics:
//!   - Data rate B (Gb/s)
//!   - Bit error rate (BER): target 1e-12 (pre-FEC) or 1e-4 (post-FEC)
//!   - Power consumption P (W)
//!   - Energy efficiency E = P/B (pJ/bit)

use std::f64::consts::PI;

/// Modulation format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModulationFormat {
    /// 2-level NRZ (1 bit/symbol)
    Nrz,
    /// 4-level PAM-4 (2 bits/symbol)
    Pam4,
    /// 8-level PAM-8 (3 bits/symbol)
    Pam8,
    /// QPSK (2 bits/symbol, coherent)
    Qpsk,
}

impl ModulationFormat {
    /// Bits per symbol.
    pub fn bits_per_symbol(&self) -> u32 {
        match self {
            ModulationFormat::Nrz => 1,
            ModulationFormat::Pam4 => 2,
            ModulationFormat::Pam8 => 3,
            ModulationFormat::Qpsk => 2,
        }
    }

    /// Symbol rate given data rate (GBaud from Gb/s).
    pub fn symbol_rate_gbaud(&self, data_rate_gbps: f64) -> f64 {
        data_rate_gbps / self.bits_per_symbol() as f64
    }

    /// Required SNR for BER = 1e-12 (approximate, AWGN).
    pub fn required_snr_db(&self) -> f64 {
        match self {
            ModulationFormat::Nrz => 17.0,  // ~OMA-based
            ModulationFormat::Pam4 => 24.0, // 9.5 dB penalty vs NRZ
            ModulationFormat::Pam8 => 30.0, // ~15 dB penalty vs NRZ
            ModulationFormat::Qpsk => 16.0, // coherent advantage
        }
    }
}

/// Optical transmitter model.
#[derive(Debug, Clone, Copy)]
pub struct OpticalTransmitter {
    /// Output optical power (dBm)
    pub tx_power_dbm: f64,
    /// Extinction ratio (dB): ratio of 1 to 0 level
    pub extinction_ratio_db: f64,
    /// Chirp parameter α (Henry factor)
    pub chirp_alpha: f64,
    /// Modulation bandwidth 3dB (GHz)
    pub bandwidth_ghz: f64,
    /// Modulation format
    pub format: ModulationFormat,
    /// Power consumption (mW)
    pub power_mw: f64,
}

impl OpticalTransmitter {
    /// Silicon ring modulator transmitter at 50 Gb/s NRZ.
    pub fn si_ring_nrz_50g() -> Self {
        Self {
            tx_power_dbm: 0.0,
            extinction_ratio_db: 7.0,
            chirp_alpha: 0.0, // ring modulator
            bandwidth_ghz: 35.0,
            format: ModulationFormat::Nrz,
            power_mw: 5.0,
        }
    }

    /// MZM transmitter at 100 Gb/s PAM-4.
    pub fn mzm_pam4_100g() -> Self {
        Self {
            tx_power_dbm: 3.0,
            extinction_ratio_db: 10.0,
            chirp_alpha: -0.5,
            bandwidth_ghz: 30.0,
            format: ModulationFormat::Pam4,
            power_mw: 25.0,
        }
    }

    /// Effective optical modulation amplitude (OMA) in dB.
    ///
    ///   OMA = P_1 - P_0 where P_1 = P_tx, P_0 = P_tx / ER
    pub fn oma_db(&self) -> f64 {
        let p_tx = 10.0_f64.powf(self.tx_power_dbm / 10.0);
        let er = 10.0_f64.powf(self.extinction_ratio_db / 10.0);
        let p1 = p_tx;
        let p0 = p_tx / er;
        10.0 * (p1 - p0).log10()
    }

    /// Energy per bit (pJ/bit).
    pub fn energy_per_bit_pj(&self, data_rate_gbps: f64) -> f64 {
        self.power_mw / data_rate_gbps // mW / Gbps = pJ/bit
    }

    /// Dispersion power penalty (dB) for fiber length L (m) and dispersion D (ps/nm/km).
    ///
    ///   PP ≈ 2·(π·D·L·Δλ·B·c/λ²)²  (NRZ formula)
    ///   where B is data rate (Gbps), Δλ is spectral width.
    pub fn dispersion_penalty_db(
        &self,
        data_rate_gbps: f64,
        fiber_length_km: f64,
        d_ps_nm_km: f64,
        wavelength_nm: f64,
    ) -> f64 {
        let b = data_rate_gbps * 1e9;
        let d_total = d_ps_nm_km * fiber_length_km * 1e-12; // s/m
        let lambda = wavelength_nm * 1e-9;
        let c = 3e8;
        // Spectral width Δλ ≈ λ²·B/c for NRZ
        let dl = lambda * lambda * b / c;
        let pp = 2.0 * (PI * d_total * dl * b).powi(2);
        10.0 * (1.0 + pp).log10() // dB
    }
}

/// Optical receiver model.
#[derive(Debug, Clone, Copy)]
pub struct OpticalReceiver {
    /// Sensitivity at target BER (dBm)
    pub sensitivity_dbm: f64,
    /// 3 dB electrical bandwidth (GHz)
    pub bandwidth_ghz: f64,
    /// Modulation format
    pub format: ModulationFormat,
    /// Power consumption (mW)
    pub power_mw: f64,
    /// Responsivity (A/W)
    pub responsivity: f64,
    /// TIA trans-impedance gain (Ω)
    pub tia_gain_ohm: f64,
}

impl OpticalReceiver {
    /// InGaAs photodetector + TIA for 100 Gb/s PAM-4.
    pub fn ingaas_pam4_100g() -> Self {
        Self {
            sensitivity_dbm: -18.0,
            bandwidth_ghz: 65.0,
            format: ModulationFormat::Pam4,
            power_mw: 150.0,
            responsivity: 0.9,
            tia_gain_ohm: 500.0,
        }
    }

    /// Si Ge photodetector for on-chip 50 Gb/s NRZ.
    pub fn sige_nrz_50g() -> Self {
        Self {
            sensitivity_dbm: -15.0,
            bandwidth_ghz: 35.0,
            format: ModulationFormat::Nrz,
            power_mw: 50.0,
            responsivity: 0.8,
            tia_gain_ohm: 200.0,
        }
    }

    /// Minimum detectable optical power (mW).
    pub fn sensitivity_mw(&self) -> f64 {
        10.0_f64.powf(self.sensitivity_dbm / 10.0)
    }

    /// Energy per bit at receiver (pJ/bit).
    pub fn energy_per_bit_pj(&self, data_rate_gbps: f64) -> f64 {
        self.power_mw / data_rate_gbps
    }

    /// TIA output voltage swing (V) for optical power P_opt (W).
    pub fn tia_output_mv(&self, p_opt_w: f64) -> f64 {
        self.responsivity * p_opt_w * self.tia_gain_ohm * 1e3
    }
}

/// Full transceiver model (TX + RX).
#[derive(Debug, Clone)]
pub struct Transceiver {
    /// Transmitter
    pub tx: OpticalTransmitter,
    /// Receiver
    pub rx: OpticalReceiver,
    /// Data rate (Gb/s)
    pub data_rate_gbps: f64,
}

impl Transceiver {
    /// Link power budget (dB).
    pub fn link_budget_db(&self) -> f64 {
        self.tx.tx_power_dbm - self.rx.sensitivity_dbm
    }

    /// Total power consumption (mW).
    pub fn total_power_mw(&self) -> f64 {
        self.tx.power_mw + self.rx.power_mw
    }

    /// System energy efficiency (pJ/bit).
    pub fn energy_efficiency_pj_per_bit(&self) -> f64 {
        self.total_power_mw() / self.data_rate_gbps
    }

    /// Achievable reach (km) for dispersion-limited link.
    pub fn max_reach_km(&self, d_ps_nm_km: f64, wavelength_nm: f64, penalty_budget_db: f64) -> f64 {
        // Solve for L such that dispersion_penalty_db = penalty_budget_db
        // Binary search
        let mut lo = 0.0f64;
        let mut hi = 1e4f64;
        for _ in 0..50 {
            let mid = (lo + hi) / 2.0;
            let pp =
                self.tx
                    .dispersion_penalty_db(self.data_rate_gbps, mid, d_ps_nm_km, wavelength_nm);
            if pp < penalty_budget_db {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (lo + hi) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modulation_pam4_bits_per_symbol() {
        assert_eq!(ModulationFormat::Pam4.bits_per_symbol(), 2);
    }

    #[test]
    fn tx_oma_db_finite() {
        let tx = OpticalTransmitter::si_ring_nrz_50g();
        // OMA = 10*log10(P1 - P0); with 0 dBm TX and 10 dB ER, OMA ≈ -0.46 dBm
        assert!(tx.oma_db().is_finite());
        assert!(tx.oma_db() > -20.0, "OMA={:.2} dBm", tx.oma_db());
    }

    #[test]
    fn tx_energy_per_bit_positive() {
        let tx = OpticalTransmitter::si_ring_nrz_50g();
        assert!(tx.energy_per_bit_pj(50.0) > 0.0);
    }

    #[test]
    fn rx_sensitivity_mw_positive() {
        let rx = OpticalReceiver::ingaas_pam4_100g();
        assert!(rx.sensitivity_mw() > 0.0);
    }

    #[test]
    fn transceiver_link_budget_positive() {
        let xcvr = Transceiver {
            tx: OpticalTransmitter::mzm_pam4_100g(),
            rx: OpticalReceiver::ingaas_pam4_100g(),
            data_rate_gbps: 100.0,
        };
        assert!(xcvr.link_budget_db() > 0.0);
    }

    #[test]
    fn transceiver_energy_efficiency_positive() {
        let xcvr = Transceiver {
            tx: OpticalTransmitter::si_ring_nrz_50g(),
            rx: OpticalReceiver::sige_nrz_50g(),
            data_rate_gbps: 50.0,
        };
        assert!(xcvr.energy_efficiency_pj_per_bit() > 0.0);
    }

    #[test]
    fn symbol_rate_pam4_half_data_rate() {
        let fmt = ModulationFormat::Pam4;
        assert!((fmt.symbol_rate_gbaud(100.0) - 50.0).abs() < 1e-10);
    }
}
