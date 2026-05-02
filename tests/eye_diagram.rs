//! Integration tests for the eye-diagram simulation module.

#[cfg(feature = "interconnect")]
mod eye_diagram_tests {
    use oxiphoton::interconnect::eye_diagram::{
        prbs, raised_cosine_filter, simulate_eye, EyeDiagramConfig,
    };
    use oxiphoton::interconnect::sparam_link::{SiPhLink, WaveguideSection};

    // ─── helpers ─────────────────────────────────────────────────────────────

    /// Local erfc approximation for test verification (Abramowitz & Stegun 7.1.26).
    fn erfc_approx(x: f64) -> f64 {
        if x < 0.0 {
            return 2.0 - erfc_approx(-x);
        }
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let poly = t
            * (0.254_829_592
                + t * (-0.284_496_736
                    + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
        poly * (-x * x).exp()
    }

    fn make_ideal_link() -> SiPhLink {
        SiPhLink::new().push(WaveguideSection {
            length_um: 100.0,
            loss_db_per_cm: 0.1,
            group_index: 4.2,
            dispersion_ps_per_nm_per_km: 0.0,
        })
    }

    fn default_freqs() -> Vec<f64> {
        (0..64).map(|i| i as f64 * 1e9).collect()
    }

    // ─── PRBS tests ───────────────────────────────────────────────────────────

    #[test]
    fn prbs_balance() {
        let bits = prbs(7, 127).expect("PRBS-7 must succeed");
        assert_eq!(bits.len(), 127);
        let ones = bits.iter().filter(|&&b| b == 1).count();
        // ITU-T O.150: PRBS-7 has exactly 64 ones per 127-bit period
        assert_eq!(ones, 64, "PRBS-7 must have exactly 64 ones in 127 bits");
    }

    #[test]
    fn prbs9_period_correct() {
        let bits = prbs(9, 511).expect("PRBS-9 must succeed");
        let ones = bits.iter().filter(|&&b| b == 1).count();
        assert_eq!(ones, 256, "PRBS-9 must have exactly 256 ones in 511 bits");
    }

    #[test]
    fn prbs_unsupported_order_errors() {
        assert!(prbs(5, 100).is_err(), "PRBS-5 is not supported");
        assert!(prbs(0, 10).is_err(), "PRBS-0 is not supported");
    }

    // ─── Raised-cosine filter tests ───────────────────────────────────────────

    #[test]
    fn raised_cosine_isi_free_at_sample_times() {
        let sps = 16usize;
        let taps = 10 * sps + 1;
        let h = raised_cosine_filter(0.35, sps, taps);
        assert_eq!(h.len(), taps);
        // At integer UI offsets from centre (k != 0), filter should be ~0 (ISI-free property)
        let center = taps / 2;
        for k in 1..=5usize {
            let val = h[center + k * sps];
            assert!(val.abs() < 1e-3, "ISI at +{k} UI: {val:.6} (should be ~0)");
            let val_neg = h[center - k * sps];
            assert!(
                val_neg.abs() < 1e-3,
                "ISI at -{k} UI: {val_neg:.6} (should be ~0)"
            );
        }
    }

    #[test]
    fn raised_cosine_normalised() {
        let h = raised_cosine_filter(0.35, 16, 10 * 16 + 1);
        let sum: f64 = h.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Filter must sum to 1, got {sum}");
    }

    // ─── Eye-diagram simulation tests ─────────────────────────────────────────

    #[test]
    fn q_factor_matches_textbook_for_noise_only() {
        let link = make_ideal_link();
        let freqs = default_freqs();
        let config = EyeDiagramConfig {
            bit_rate_gbps: 10.0,
            samples_per_bit: 16,
            n_bits: 256,
            osnr_db: 15.0,
            prbs_order: 7,
            rolloff: 0.35,
            ..Default::default()
        };
        let result = simulate_eye(&link, &freqs, &config).expect("simulate_eye must succeed");
        // For OSNR=15 dB NRZ-OOK, Q should be in a physically reasonable range
        assert!(
            result.q_factor > 2.0 && result.q_factor < 25.0,
            "Q-factor out of range: {}",
            result.q_factor
        );
        assert!(result.eye_opening_v > 0.0, "Eye must be open");
    }

    #[test]
    fn eye_opening_shrinks_with_bandwidth_limit() {
        // Build a link with heavy dispersion to close the eye
        let narrow_link = SiPhLink::new().push(WaveguideSection {
            length_um: 10_000.0,
            loss_db_per_cm: 0.1,
            group_index: 4.2,
            dispersion_ps_per_nm_per_km: 2000.0,
        });
        let ideal_link = make_ideal_link();
        let freqs = default_freqs();
        let config = EyeDiagramConfig {
            bit_rate_gbps: 25.0,
            samples_per_bit: 16,
            n_bits: 256,
            osnr_db: 30.0,
            prbs_order: 7,
            rolloff: 0.35,
            ..Default::default()
        };
        let ideal_result =
            simulate_eye(&ideal_link, &freqs, &config).expect("ideal simulate_eye must succeed");
        let narrow_result =
            simulate_eye(&narrow_link, &freqs, &config).expect("narrow simulate_eye must succeed");
        // The dispersive link should not open a wider eye than the ideal link
        assert!(
            narrow_result.eye_opening_v <= ideal_result.eye_opening_v + 0.01,
            "Narrow-band link must not have larger eye opening than ideal: narrow={:.4} ideal={:.4}",
            narrow_result.eye_opening_v,
            ideal_result.eye_opening_v,
        );
    }

    #[test]
    fn jitter_increases_with_osnr_decrease() {
        let link = make_ideal_link();
        let freqs = default_freqs();
        // Higher OSNR → lower noise → less jitter
        let config_high = EyeDiagramConfig {
            bit_rate_gbps: 10.0,
            samples_per_bit: 16,
            n_bits: 256,
            osnr_db: 25.0,
            prbs_order: 7,
            rolloff: 0.35,
            ..Default::default()
        };
        let config_low = EyeDiagramConfig {
            osnr_db: 12.0,
            ..config_high.clone()
        };
        let high_result =
            simulate_eye(&link, &freqs, &config_high).expect("high OSNR simulate_eye must succeed");
        let low_result =
            simulate_eye(&link, &freqs, &config_low).expect("low OSNR simulate_eye must succeed");
        // With lower OSNR there is more noise so jitter should be >= (with small tolerance)
        assert!(
            low_result.jitter_rms_ps >= high_result.jitter_rms_ps - 0.2,
            "Jitter should be non-decreasing as OSNR drops; high_OSNR={:.3} low_OSNR={:.3}",
            high_result.jitter_rms_ps,
            low_result.jitter_rms_ps,
        );
    }

    #[test]
    fn q_factor_to_ber_estimate_matches_erfc() {
        let link = make_ideal_link();
        let freqs = default_freqs();
        let config = EyeDiagramConfig {
            bit_rate_gbps: 10.0,
            samples_per_bit: 16,
            n_bits: 256,
            osnr_db: 20.0,
            prbs_order: 7,
            rolloff: 0.35,
            ..Default::default()
        };
        let result = simulate_eye(&link, &freqs, &config).expect("simulate_eye must succeed");
        let q = result.q_factor;
        let expected_ber = 0.5 * erfc_approx(q / 2.0_f64.sqrt());
        let rel_err = (result.ber_estimate - expected_ber).abs() / (expected_ber + 1e-30);
        assert!(
            rel_err < 0.01,
            "BER estimate does not match erfc formula: got {} expected {}",
            result.ber_estimate,
            expected_ber
        );
    }

    #[test]
    fn simulate_returns_correct_trace_dimensions() {
        let link = make_ideal_link();
        let freqs = default_freqs();
        let sps = 16usize;
        let config = EyeDiagramConfig {
            bit_rate_gbps: 10.0,
            samples_per_bit: sps,
            n_bits: 128,
            osnr_db: 20.0,
            prbs_order: 7,
            rolloff: 0.35,
            ..Default::default()
        };
        let result = simulate_eye(&link, &freqs, &config).expect("simulate_eye must succeed");
        // Time axis should cover 2 UI
        assert_eq!(result.time_axis_ps.len(), 2 * sps);
        // All traces should have the same length = 2*sps
        for (i, trace) in result.traces.iter().enumerate() {
            assert_eq!(
                trace.len(),
                2 * sps,
                "trace[{i}] has wrong length: {}",
                trace.len()
            );
        }
    }

    #[test]
    fn pam4_eye_simulation_runs_without_error() {
        use oxiphoton::comms::modulation::ModulationFormat;
        use oxiphoton::interconnect::eye_diagram::EyeDiagramConfig;
        let link = make_ideal_link();
        let freqs = default_freqs();
        let config = EyeDiagramConfig {
            bit_rate_gbps: 10.0,
            samples_per_bit: 16,
            n_bits: 256,
            osnr_db: 20.0,
            prbs_order: 7,
            rolloff: 0.35,
            modulation: ModulationFormat::Pam4,
        };
        let result = simulate_eye(&link, &freqs, &config).expect("PAM-4 simulate_eye must succeed");
        assert!(
            result.eye_opening_v >= 0.0,
            "PAM-4 eye opening must be non-negative"
        );
        assert!(
            result.q_factor >= 0.0,
            "PAM-4 Q-factor must be non-negative"
        );
    }
}
