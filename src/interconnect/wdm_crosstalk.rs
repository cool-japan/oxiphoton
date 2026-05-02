//! WDM crosstalk matrix for N WDM channels sharing a SiPh link.
//!
//! Models the inter-channel interference arising from the frequency roll-off
//! of optical elements in the link.  The crosstalk from channel j into channel
//! i is estimated from the filter rejection applied at the wavelength separation
//! between channels i and j.

#![cfg(feature = "interconnect")]

use crate::interconnect::sparam_link::SiPhLink;

const C_LIGHT: f64 = 2.997_924_58e8; // m/s

// ─────────────────────────────────────────────────────────────────────────────
// WdmCh
// ─────────────────────────────────────────────────────────────────────────────

/// A single WDM channel defined by centre wavelength and launch power.
#[derive(Debug, Clone)]
pub struct WdmCh {
    /// Centre wavelength (nm)
    pub wavelength_nm: f64,
    /// Launch power (dBm)
    pub power_dbm: f64,
}

impl WdmCh {
    /// Centre frequency (Hz).
    pub fn frequency_hz(&self) -> f64 {
        C_LIGHT / (self.wavelength_nm * 1e-9)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WdmCrosstalkMatrix
// ─────────────────────────────────────────────────────────────────────────────

/// Crosstalk matrix for N WDM channels through a shared SiPh link.
///
/// Entry `crosstalk_matrix_db[i][j]` is the crosstalk power (dB) that leaks
/// from channel j into channel i.  The diagonal is set to 0 dB by convention
/// (the direct signal path, not cross-talk in the interference sense).
///
/// # Crosstalk model
///
/// Each optical element in the link acts as a bandpass filter with frequency
/// selectivity.  For a simple first-order model, the adjacent-channel
/// isolation is approximated by a linear wavelength rejection:
///
/// ```text
/// XT(Δλ) = -filter_rejection_db_per_nm × |λ_i - λ_j|   (dB)
/// ```
///
/// The link's own insertion loss at each channel frequency is computed from
/// the S-matrix cascade, so on-chip element losses are included.
#[derive(Debug, Clone)]
pub struct WdmCrosstalkMatrix {
    /// WDM channels
    pub channels: Vec<WdmCh>,
    /// Crosstalk matrix (dB): `[i][j]` = power from ch j into ch i
    pub crosstalk_matrix_db: Vec<Vec<f64>>,
}

impl WdmCrosstalkMatrix {
    /// Build a crosstalk matrix from a [`SiPhLink`] and WDM channel plan.
    ///
    /// # Arguments
    /// * `link` — the shared SiPh link
    /// * `channels` — WDM channel plan (centre wavelengths and launch powers)
    /// * `filter_rejection_db_per_nm` — first-order frequency selectivity of
    ///   the link elements (dB/nm); a higher value means better isolation
    pub fn from_link(
        link: &SiPhLink,
        channels: Vec<WdmCh>,
        filter_rejection_db_per_nm: f64,
    ) -> Self {
        let n = channels.len();

        // Evaluate the link S-parameter at each channel's centre frequency
        let freqs: Vec<f64> = channels.iter().map(|ch| ch.frequency_hz()).collect();
        let sp = if freqs.is_empty() {
            Vec::new()
        } else {
            link.cascade(&freqs)
        };

        // On-chip link insertion loss at each channel (dB)
        let link_il_db: Vec<f64> = sp
            .iter()
            .map(|p| {
                let mag = p[1].norm();
                if mag < 1e-40 {
                    400.0
                } else {
                    -20.0 * mag.log10()
                }
            })
            .collect();

        // Build the N×N crosstalk matrix
        let mut matrix: Vec<Vec<f64>> = vec![vec![f64::NEG_INFINITY; n]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    // Diagonal: self-signal (not crosstalk)
                    matrix[i][j] = 0.0;
                } else {
                    let delta_lambda_nm =
                        (channels[i].wavelength_nm - channels[j].wavelength_nm).abs();
                    // Crosstalk isolation from filter roll-off
                    let isolation_db = filter_rejection_db_per_nm * delta_lambda_nm;
                    // Power of ch j that leaks into ch i:
                    // P_xt = P_j_launch - link_IL_j - isolation
                    // expressed as a ratio relative to the signal power at ch i:
                    let signal_power_dbm = channels[i].power_dbm - link_il_db[i];
                    let interferer_power_dbm = channels[j].power_dbm - link_il_db[j] - isolation_db;
                    // Crosstalk in dB: relative to signal at ch i
                    matrix[i][j] = interferer_power_dbm - signal_power_dbm;
                }
            }
        }

        Self {
            channels,
            crosstalk_matrix_db: matrix,
        }
    }

    /// Worst-case OSNR penalty (dB) from all other channels leaking into channel i.
    ///
    /// Penalty is approximated as:
    /// ```text
    /// penalty ≈ -10·log10(1 - Σ_{j≠i} 10^(XT[i][j]/10))
    /// ```
    /// This is the standard optical crosstalk penalty formula.
    ///
    /// Returns 0.0 for channels with no significant crosstalk.
    pub fn crosstalk_penalty_db(&self, channel_idx: usize) -> f64 {
        let n = self.channels.len();
        if channel_idx >= n || n < 2 {
            return 0.0;
        }

        // Sum all cross-channel interference in linear scale
        let total_xt_linear: f64 = (0..n)
            .filter(|&j| j != channel_idx)
            .map(|j| {
                let xt_db = self.crosstalk_matrix_db[channel_idx][j];
                if xt_db.is_finite() {
                    10.0_f64.powf(xt_db / 10.0)
                } else {
                    0.0 // treat -inf as zero
                }
            })
            .sum();

        if total_xt_linear <= 0.0 {
            return 0.0;
        }

        // Clamp to avoid log of non-positive number
        if total_xt_linear >= 1.0 {
            return 30.0; // saturate at 30 dB penalty
        }

        -10.0 * (1.0 - total_xt_linear).log10()
    }

    /// Number of channels in the WDM plan.
    pub fn n_channels(&self) -> usize {
        self.channels.len()
    }

    /// Total interference power (dBm) received by channel i from all interferers.
    ///
    /// Uses logarithmic power addition of all off-diagonal entries in row i.
    pub fn total_interference_dbm(&self, channel_idx: usize) -> f64 {
        let n = self.channels.len();
        if channel_idx >= n {
            return f64::NEG_INFINITY;
        }

        let signal_power_dbm = self.channels[channel_idx].power_dbm;

        let sum_linear: f64 = (0..n)
            .filter(|&j| j != channel_idx)
            .map(|j| {
                let xt_db = self.crosstalk_matrix_db[channel_idx][j];
                // XT is relative to signal; convert to absolute dBm and then linear
                let xt_abs_dbm = signal_power_dbm + xt_db;
                if xt_abs_dbm.is_finite() {
                    10.0_f64.powf(xt_abs_dbm / 10.0)
                } else {
                    0.0
                }
            })
            .sum();

        if sum_linear <= 0.0 {
            return f64::NEG_INFINITY;
        }
        10.0 * sum_linear.log10()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interconnect::sparam_link::SiPhLink;

    #[test]
    fn crosstalk_matrix_is_n_by_n() {
        let channels = vec![
            WdmCh {
                wavelength_nm: 1550.0,
                power_dbm: 0.0,
            },
            WdmCh {
                wavelength_nm: 1550.8,
                power_dbm: 0.0,
            },
            WdmCh {
                wavelength_nm: 1551.6,
                power_dbm: 0.0,
            },
        ];
        let link = SiPhLink::new();
        let mat = WdmCrosstalkMatrix::from_link(&link, channels, 20.0);
        for row in &mat.crosstalk_matrix_db {
            assert_eq!(row.len(), 3);
        }
        assert_eq!(mat.crosstalk_matrix_db.len(), 3);
    }

    #[test]
    fn diagonal_is_zero() {
        let channels = vec![
            WdmCh {
                wavelength_nm: 1549.2,
                power_dbm: 0.0,
            },
            WdmCh {
                wavelength_nm: 1550.0,
                power_dbm: 0.0,
            },
            WdmCh {
                wavelength_nm: 1550.8,
                power_dbm: 0.0,
            },
        ];
        let link = SiPhLink::new();
        let mat = WdmCrosstalkMatrix::from_link(&link, channels, 20.0);
        for i in 0..3 {
            assert!(
                (mat.crosstalk_matrix_db[i][i] - 0.0).abs() < 1e-12,
                "Diagonal[{i}] should be 0, got {}",
                mat.crosstalk_matrix_db[i][i]
            );
        }
    }

    #[test]
    fn crosstalk_decreases_with_channel_separation() {
        // ch 0 at 1550, ch 1 at 1550.8 (closer), ch 2 at 1552.0 (farther)
        let channels = vec![
            WdmCh {
                wavelength_nm: 1550.0,
                power_dbm: 0.0,
            },
            WdmCh {
                wavelength_nm: 1550.8,
                power_dbm: 0.0,
            },
            WdmCh {
                wavelength_nm: 1552.0,
                power_dbm: 0.0,
            },
        ];
        let link = SiPhLink::new();
        let mat = WdmCrosstalkMatrix::from_link(&link, channels, 20.0);
        // Crosstalk from ch1→ch0 (Δλ=0.8 nm) should be less negative than ch2→ch0 (Δλ=2.0 nm)
        // i.e., ch1 leaks more into ch0 than ch2 does
        let xt_near = mat.crosstalk_matrix_db[0][1];
        let xt_far = mat.crosstalk_matrix_db[0][2];
        assert!(
            xt_near > xt_far,
            "Near channel crosstalk should be higher (less suppressed): xt_near={xt_near:.1}, xt_far={xt_far:.1}"
        );
    }

    #[test]
    fn penalty_zero_for_single_channel() {
        let channels = vec![WdmCh {
            wavelength_nm: 1550.0,
            power_dbm: 0.0,
        }];
        let link = SiPhLink::new();
        let mat = WdmCrosstalkMatrix::from_link(&link, channels, 20.0);
        let penalty = mat.crosstalk_penalty_db(0);
        assert!(
            penalty.abs() < 1e-10,
            "Single channel should have zero penalty, got {penalty}"
        );
    }
}
