//! End-to-end optical interconnect integration tests.
//!
//! Feature-gated under `interconnect`.

#[cfg(feature = "interconnect")]
mod tests {
    use oxiphoton::comms::modulation::ModulationFormat;
    use oxiphoton::interconnect::ber_analysis::{BerOsnrCurve, LinkPerformanceAnalysis};
    use oxiphoton::interconnect::sparam_link::{
        DirectionalCoupler, SiPhLink, Splitter50_50, WaveguideSection,
    };
    use oxiphoton::interconnect::wdm_crosstalk::{WdmCh, WdmCrosstalkMatrix};

    // ─────────────────────────────────────────────────────────────────────────
    // WaveguideSection tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn waveguide_insertion_loss_matches_theory() {
        // 1 cm waveguide at 3 dB/cm → 3 dB insertion loss
        let wg = WaveguideSection {
            length_um: 10_000.0,
            loss_db_per_cm: 3.0,
            group_index: 4.2,
            dispersion_ps_per_nm_per_km: 1000.0,
        };
        let link = SiPhLink::new().push(wg);
        let freqs = [193.41e12_f64]; // ~1550 nm
        let il = link.insertion_loss_db(&freqs);
        assert!(
            (il[0] - 3.0).abs() < 0.01,
            "Expected 3 dB IL, got {:.4}",
            il[0]
        );
    }

    #[test]
    fn cascade_two_waveguides_doubles_loss() {
        let wg1 = WaveguideSection {
            length_um: 5_000.0,
            loss_db_per_cm: 2.0,
            group_index: 4.2,
            dispersion_ps_per_nm_per_km: 0.0,
        };
        let wg2 = WaveguideSection {
            length_um: 5_000.0,
            loss_db_per_cm: 2.0,
            group_index: 4.2,
            dispersion_ps_per_nm_per_km: 0.0,
        };
        let link = SiPhLink::new().push(wg1).push(wg2);
        let freqs = [193.41e12_f64];
        let il = link.insertion_loss_db(&freqs);
        // Each: 2 dB/cm × 0.5 cm = 1 dB; total = 2 dB
        assert!(
            (il[0] - 2.0).abs() < 0.01,
            "Expected 2 dB, got {:.4}",
            il[0]
        );
    }

    #[test]
    fn group_delay_increases_with_length() {
        let short_wg = WaveguideSection {
            length_um: 1_000.0,
            loss_db_per_cm: 0.0,
            group_index: 4.2,
            dispersion_ps_per_nm_per_km: 0.0,
        };
        let long_wg = WaveguideSection {
            length_um: 10_000.0,
            loss_db_per_cm: 0.0,
            group_index: 4.2,
            dispersion_ps_per_nm_per_km: 0.0,
        };
        // Use many closely-spaced frequency points for stable numerical derivative
        let freqs: Vec<f64> = (0..50).map(|i| 193.0e12 + i as f64 * 10e9).collect();
        let gd_short = SiPhLink::new().push(short_wg).group_delay_ps(&freqs);
        let gd_long = SiPhLink::new().push(long_wg).group_delay_ps(&freqs);

        // Long waveguide should have more group delay at every point
        assert!(
            gd_long[0] > gd_short[0],
            "Longer waveguide should have more group delay: long={:.2} ps, short={:.2} ps",
            gd_long[0],
            gd_short[0]
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BER vs OSNR tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn ber_decreases_with_osnr_nrz_ook() {
        let osnr = vec![10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
        let curve = BerOsnrCurve::compute(ModulationFormat::Ook, 100.0, &osnr, 0.0);
        // BER should strictly decrease with increasing OSNR
        for i in 1..curve.ber.len() {
            assert!(
                curve.ber[i] <= curve.ber[i - 1] + 1e-15,
                "BER should decrease with OSNR: BER[{}]={:.2e} > BER[{}]={:.2e}",
                i,
                curve.ber[i],
                i - 1,
                curve.ber[i - 1]
            );
        }
    }

    #[test]
    fn ber_required_osnr_is_physical() {
        let osnr: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let curve = BerOsnrCurve::compute(ModulationFormat::Ook, 100.0, &osnr, 0.0);
        // Required OSNR for NRZ-OOK BER=1e-3 should be physically reasonable (5–25 dB range)
        assert!(
            curve.osnr_required_db >= 5.0 && curve.osnr_required_db <= 25.0,
            "Required OSNR for NRZ-OOK BER=1e-3 should be 5–25 dB, got {:.2}",
            curve.osnr_required_db
        );
    }

    #[test]
    fn link_performance_analysis_positive_margin() {
        // Empty link (0 dB IL), NRZ-OOK, 10 Gb/s
        let link = SiPhLink::new();
        let analysis = LinkPerformanceAnalysis::new(link, ModulationFormat::Ook, 10.0);
        // For a reasonable launch power, system margin should be finite
        let margin = analysis.system_margin_db(1e-3);
        assert!(
            margin.is_finite(),
            "System margin must be finite, got {margin}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // WDM crosstalk tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn wdm_crosstalk_matrix_diagonal_zero() {
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
        for i in 0..3 {
            assert_eq!(
                mat.crosstalk_matrix_db[i].len(),
                3,
                "Row {i} should have 3 entries"
            );
            assert!(
                (mat.crosstalk_matrix_db[i][i] - 0.0).abs() < 1e-12,
                "Diagonal[{i}][{i}] should be 0, got {}",
                mat.crosstalk_matrix_db[i][i]
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Splitter and DirectionalCoupler tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn splitter_50_50_loss_is_3db() {
        let link = SiPhLink::new().push(Splitter50_50);
        let freqs = [193.41e12_f64];
        let il = link.insertion_loss_db(&freqs);
        // Ideal 50:50 split = 3.0103 dB
        assert!(
            (il[0] - 3.0103).abs() < 0.01,
            "Splitter IL should be ~3.01 dB, got {:.4}",
            il[0]
        );
    }

    #[test]
    fn directional_coupler_through_port_loss() {
        // 50:50 coupler with 1 dB excess loss → through port loss = 3 dB + 1 dB = 4 dB
        let dc = DirectionalCoupler {
            coupling_ratio: 0.5,
            excess_loss_db: 1.0,
        };
        let link = SiPhLink::new().push(dc);
        let freqs = [193.41e12_f64];
        let il = link.insertion_loss_db(&freqs);
        // |S21|^2 = (1 - 0.5) * 10^(-1/10) ≈ 0.5 * 0.794 = 0.397
        // IL = -10*log10(0.397) ≈ 4.01 dB
        assert!(
            (il[0] - 4.01).abs() < 0.1,
            "DC through-port IL should be ~4 dB, got {:.4}",
            il[0]
        );
    }
}
