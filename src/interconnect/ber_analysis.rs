//! BER vs OSNR characterisation for optical links.
//!
//! Provides [`BerOsnrCurve`] for sweeping OSNR and computing BER for a given
//! modulation format, and [`LinkPerformanceAnalysis`] for full system margin
//! calculation including cascaded-EDFA OSNR.

#![cfg(feature = "interconnect")]

use crate::comms::metrics::{BerCalculator, OsnrAnalysis};
use crate::comms::modulation::ModulationFormat;
use crate::error::OxiPhotonError;
use crate::interconnect::sparam_link::SiPhLink;

// Physical constants
const H_PLANCK: f64 = 6.626_070_15e-34; // J·s
const C_LIGHT: f64 = 2.997_924_58e8; // m/s

// Reference bandwidth for OSNR: 0.1 nm at 1550 nm ≈ 12.5 GHz
const B_REF_HZ: f64 = 12.5e9; // Hz

// ─────────────────────────────────────────────────────────────────────────────
// BerOsnrCurve
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a BER-vs-OSNR sweep for a given modulation format.
///
/// OSNR is measured in a 0.1 nm reference bandwidth (12.5 GHz at 1550 nm).
#[derive(Debug, Clone)]
pub struct BerOsnrCurve {
    /// OSNR values (dB, in 0.1 nm reference bandwidth)
    pub osnr_db: Vec<f64>,
    /// BER at each OSNR point
    pub ber: Vec<f64>,
    /// Modulation format used for the computation
    pub modulation: ModulationFormat,
    /// Bit rate (Gb/s)
    pub bit_rate_gbps: f64,
    /// Required OSNR for BER = 1e-3 (pre-FEC threshold), in dB
    pub osnr_required_db: f64,
    /// Required OSNR for BER = 1e-12 (error-free threshold), in dB
    pub osnr_required_ef_db: f64,
}

impl BerOsnrCurve {
    /// Compute a BER-vs-OSNR curve for a given modulation format.
    ///
    /// # Arguments
    ///
    /// * `modulation` — modulation format
    /// * `bit_rate_gbps` — data rate (Gb/s)
    /// * `osnr_range_db` — OSNR values to sweep (dB, in 0.1 nm bandwidth)
    /// * `dispersion_penalty_db` — additional OSNR penalty from chromatic
    ///   dispersion (0 = ideal back-to-back)
    pub fn compute(
        modulation: ModulationFormat,
        bit_rate_gbps: f64,
        osnr_range_db: &[f64],
        dispersion_penalty_db: f64,
    ) -> Self {
        let baud_rate_hz = bit_rate_gbps * 1e9 / modulation.bits_per_symbol() as f64;

        let ber: Vec<f64> = osnr_range_db
            .iter()
            .map(|&osnr_db| {
                // Apply dispersion penalty by subtracting it from effective OSNR
                let effective_osnr_db = osnr_db - dispersion_penalty_db;
                Self::ber_from_osnr_db(&modulation, effective_osnr_db, baud_rate_hz)
            })
            .collect();

        // Find required OSNR for pre-FEC threshold (BER = 1e-3)
        let osnr_required_db =
            Self::find_required_osnr(&modulation, 1e-3, baud_rate_hz, dispersion_penalty_db);

        // Find required OSNR for error-free (BER = 1e-12)
        let osnr_required_ef_db =
            Self::find_required_osnr(&modulation, 1e-12, baud_rate_hz, dispersion_penalty_db);

        Self {
            osnr_db: osnr_range_db.to_vec(),
            ber,
            modulation,
            bit_rate_gbps,
            osnr_required_db,
            osnr_required_ef_db,
        }
    }

    /// Compute BER from OSNR (dB) for a given modulation format and baud rate.
    ///
    /// Converts OSNR → Eb/N₀ → BER using the analytic formula for the format.
    fn ber_from_osnr_db(modulation: &ModulationFormat, osnr_db: f64, baud_rate_hz: f64) -> f64 {
        // Convert OSNR (dB in 0.1 nm) to linear
        let osnr_linear = 10.0_f64.powf(osnr_db / 10.0);

        // Convert OSNR_linear → Eb/N₀_linear
        // Eb/N₀ = OSNR_linear × B_ref / (2 × Rs)
        // where B_ref = 12.5 GHz and Rs = baud_rate
        let eb_n0_linear =
            BerCalculator::osnr_to_eb_n0(osnr_linear, baud_rate_hz.max(1e3), B_REF_HZ);

        // Compute BER from Eb/N₀ for each format
        match modulation {
            ModulationFormat::Ook => {
                // OOK: Q ≈ sqrt(2 * Eb/N0)
                let q = (2.0 * eb_n0_linear.max(0.0)).sqrt();
                BerCalculator::ook_direct_ber(q)
            }
            ModulationFormat::Bpsk | ModulationFormat::Dpsk => {
                BerCalculator::bpsk_ber(eb_n0_linear)
            }
            ModulationFormat::Qpsk | ModulationFormat::Dqpsk => {
                BerCalculator::qpsk_ber(eb_n0_linear)
            }
            ModulationFormat::Qam16 => BerCalculator::qam16_ber(eb_n0_linear),
            ModulationFormat::Qam64 => BerCalculator::qam64_ber(eb_n0_linear),
            ModulationFormat::Qam256 => {
                let arg = (eb_n0_linear.max(0.0) / 85.0).sqrt();
                (15.0 / 64.0) * BerCalculator::erfc(arg)
            }
            ModulationFormat::Pam4 => {
                let arg = (eb_n0_linear.max(0.0) / 5.0).sqrt();
                (3.0 / 4.0) * BerCalculator::erfc(arg)
            }
        }
    }

    /// Binary search for the OSNR (dB) required to achieve a target BER.
    ///
    /// Searches over [−10, 50] dB OSNR range.
    fn find_required_osnr(
        modulation: &ModulationFormat,
        ber_target: f64,
        baud_rate_hz: f64,
        dispersion_penalty_db: f64,
    ) -> f64 {
        // BER decreases monotonically with OSNR, so binary search is valid
        let mut lo = -10.0_f64;
        let mut hi = 60.0_f64;

        // If BER at lo is already below target, required OSNR is very low
        let ber_at_hi =
            Self::ber_from_osnr_db(modulation, hi - dispersion_penalty_db, baud_rate_hz);
        if ber_at_hi > ber_target {
            return hi; // Cannot reach target even at high OSNR
        }

        let ber_at_lo =
            Self::ber_from_osnr_db(modulation, lo - dispersion_penalty_db, baud_rate_hz);
        if ber_at_lo <= ber_target {
            return lo; // Target met even at low OSNR
        }

        for _ in 0..80 {
            let mid = (lo + hi) / 2.0;
            let ber_mid =
                Self::ber_from_osnr_db(modulation, mid - dispersion_penalty_db, baud_rate_hz);
            if ber_mid > ber_target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (lo + hi) / 2.0 + dispersion_penalty_db
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BER-vs-OSNR sweep driven by a SiPhLink
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a [`BerOsnrCurve`] for a physical [`SiPhLink`], using the link's
/// insertion loss at the signal centre frequency as the dispersion penalty.
///
/// The insertion loss (dB) is obtained from
/// [`SiPhLink::insertion_loss_db`] evaluated at `f_center_hz`.  It is passed
/// directly as the `dispersion_penalty_db` argument to
/// [`BerOsnrCurve::compute`], which subtracts it from the effective OSNR at
/// each sweep point.
///
/// # Arguments
///
/// * `link`          — the SiPh link whose on-chip loss drives the penalty
/// * `f_center_hz`   — signal centre frequency (Hz)
/// * `modulation`    — modulation format
/// * `bit_rate_gbps` — data rate (Gb/s)
/// * `osnr_db_grid`  — OSNR sweep points (dB, in 0.1 nm reference bandwidth)
///
/// # Errors
///
/// Returns [`OxiPhotonError::NumericalError`] if `insertion_loss_db` returns
/// an empty vector (should not happen for a well-formed link).
pub fn ber_vs_osnr_sweep_for_link(
    link: &SiPhLink,
    f_center_hz: f64,
    modulation: ModulationFormat,
    bit_rate_gbps: f64,
    osnr_db_grid: &[f64],
) -> Result<BerOsnrCurve, OxiPhotonError> {
    let il_db = link
        .insertion_loss_db(&[f_center_hz])
        .first()
        .copied()
        .ok_or_else(|| {
            OxiPhotonError::NumericalError(
                "insertion_loss_db returned empty vec for ber_vs_osnr_sweep_for_link".to_string(),
            )
        })?;

    Ok(BerOsnrCurve::compute(
        modulation,
        bit_rate_gbps,
        osnr_db_grid,
        il_db,
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// LinkPerformanceAnalysis
// ─────────────────────────────────────────────────────────────────────────────

/// Full link performance analysis: SiPh link + amplified fiber span.
///
/// Models a SiPh photonic integrated circuit link followed by one or more
/// fiber spans with EDFA amplification.  Computes receiver OSNR and system
/// margin against a BER threshold.
// Note: no #[derive(Debug)] because SiPhLink is not Debug
pub struct LinkPerformanceAnalysis {
    /// On-chip SiPh link
    pub link: SiPhLink,
    /// Modulation format
    pub modulation: ModulationFormat,
    /// Data rate (Gb/s)
    pub bit_rate_gbps: f64,
    /// TX launch power into the first fiber span (dBm)
    pub launch_power_dbm: f64,
    /// EDFA noise figure (dB)
    pub amplifier_noise_figure_db: f64,
    /// Span length (km, used only for reference; does not change OSNR formula)
    pub span_length_km: f64,
    /// Number of amplifier spans
    pub n_spans: usize,
}

impl LinkPerformanceAnalysis {
    /// Create a new analysis with sensible defaults.
    ///
    /// Default: 0 dBm launch, 5 dB NF, 1 span of 80 km.
    pub fn new(link: SiPhLink, modulation: ModulationFormat, bit_rate_gbps: f64) -> Self {
        Self {
            link,
            modulation,
            bit_rate_gbps,
            launch_power_dbm: 0.0,
            amplifier_noise_figure_db: 5.0,
            span_length_km: 80.0,
            n_spans: 1,
        }
    }

    /// Total on-chip link insertion loss at the signal wavelength (dB).
    ///
    /// # Arguments
    /// * `wavelength_nm` — signal wavelength (nm)
    pub fn link_loss_db(&self, wavelength_nm: f64) -> f64 {
        let freq_hz = C_LIGHT / (wavelength_nm * 1e-9);
        let il = self.link.insertion_loss_db(&[freq_hz]);
        il.into_iter().next().unwrap_or(0.0)
    }

    /// Receiver OSNR (dB) using the cascaded-amplifier OSNR formula.
    ///
    /// Formula:
    /// ```text
    /// OSNR_linear = P_launch / (N_spans · h·ν · NF_linear · B_ref)
    /// ```
    /// where `B_ref = 12.5 GHz` (0.1 nm at 1550 nm), `h` = Planck constant,
    /// `ν = c/λ` at 1550 nm.
    pub fn receiver_osnr_db(&self) -> f64 {
        if self.n_spans == 0 {
            return f64::INFINITY;
        }

        let nu = C_LIGHT / 1550e-9; // frequency at 1550 nm reference
        let p_launch_w = 10.0_f64.powf(self.launch_power_dbm / 10.0) * 1e-3;
        let nf_linear = 10.0_f64.powf(self.amplifier_noise_figure_db / 10.0);

        let ase_per_span_w = H_PLANCK * nu * nf_linear * B_REF_HZ;
        let total_ase_w = self.n_spans as f64 * ase_per_span_w;

        if total_ase_w <= 0.0 {
            return f64::INFINITY;
        }

        let osnr_linear = p_launch_w / total_ase_w;
        10.0 * osnr_linear.max(1e-40).log10()
    }

    /// System margin (dB) = receiver OSNR − required OSNR for target BER.
    ///
    /// Positive margin means the link is viable.
    pub fn system_margin_db(&self, target_ber: f64) -> f64 {
        let baud_rate_hz = self.bit_rate_gbps * 1e9 / self.modulation.bits_per_symbol() as f64;
        let receiver_osnr = self.receiver_osnr_db();
        let required_osnr =
            BerOsnrCurve::find_required_osnr(&self.modulation, target_ber, baud_rate_hz, 0.0);
        receiver_osnr - required_osnr
    }

    /// Available link power budget (dB): launch power − receiver sensitivity.
    ///
    /// Receiver sensitivity is computed from the target BER and the modulation
    /// format via `comms::metrics::OsnrAnalysis::required_osnr_db`.
    pub fn power_margin_db(&self, target_ber: f64) -> f64 {
        let required_osnr = OsnrAnalysis::required_osnr_db(&self.modulation, target_ber);
        let link_loss = self.link_loss_db(1550.0);
        self.launch_power_dbm - required_osnr - link_loss
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ber_decreases_monotonically_with_osnr_ook() {
        let osnr = vec![5.0, 8.0, 11.0, 14.0, 17.0, 20.0];
        let curve = BerOsnrCurve::compute(ModulationFormat::Ook, 10.0, &osnr, 0.0);
        for i in 1..curve.ber.len() {
            assert!(
                curve.ber[i] <= curve.ber[i - 1] + 1e-15,
                "BER not monotonic at index {i}: {:.3e} > {:.3e}",
                curve.ber[i],
                curve.ber[i - 1]
            );
        }
    }

    #[test]
    fn required_osnr_in_physical_range_ook() {
        let osnr: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let curve = BerOsnrCurve::compute(ModulationFormat::Ook, 10.0, &osnr, 0.0);
        assert!(
            curve.osnr_required_db >= 5.0 && curve.osnr_required_db <= 25.0,
            "Required OSNR for OOK BER=1e-3 should be in [5,25] dB, got {:.2}",
            curve.osnr_required_db
        );
    }

    #[test]
    fn link_performance_margin_is_finite() {
        let link = SiPhLink::new();
        let analysis = LinkPerformanceAnalysis::new(link, ModulationFormat::Ook, 10.0);
        let margin = analysis.system_margin_db(1e-3);
        assert!(
            margin.is_finite(),
            "System margin must be finite, got {margin}"
        );
    }
}
