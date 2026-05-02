//! Integration tests for `chip_to_chip_link_response` and
//! `ber_vs_osnr_sweep_for_link`.
//!
//! Feature-gated under `interconnect`.

#[cfg(feature = "interconnect")]
mod tests {
    use num_complex::Complex64;
    use oxiphoton::comms::modulation::ModulationFormat;
    use oxiphoton::interconnect::ber_analysis::ber_vs_osnr_sweep_for_link;
    use oxiphoton::interconnect::sparam_link::{
        chip_to_chip_link_response, SiPhElement, SiPhLink, WaveguideSection,
    };

    // ─────────────────────────────────────────────────────────────────────────
    // Helper: build a realistic waveguide-only link
    // ─────────────────────────────────────────────────────────────────────────

    fn make_waveguide_link(length_um: f64, loss_db_per_cm: f64, group_index: f64) -> SiPhLink {
        SiPhLink::new().push(WaveguideSection {
            length_um,
            loss_db_per_cm,
            group_index,
            dispersion_ps_per_nm_per_km: 0.0,
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Custom band-pass element for bandwidth-narrowing test
    // ─────────────────────────────────────────────────────────────────────────

    /// A Lorentzian bandpass filter modelled as a 2-port element:
    /// ```text
    /// S21(f) = S12(f) = 1 / sqrt(1 + ((f - f0) / bw)^2)
    /// S11 = S22 = 0
    /// ```
    /// with centre frequency `f0_hz` and half-bandwidth `bw_hz`.
    struct LorentzianBandpass {
        f0_hz: f64,
        bw_hz: f64,
    }

    impl SiPhElement for LorentzianBandpass {
        fn s_params(&self, freq_hz: &[f64]) -> Vec<[Complex64; 4]> {
            freq_hz
                .iter()
                .map(|&f| {
                    let x = (f - self.f0_hz) / self.bw_hz;
                    let mag = 1.0 / (1.0 + x * x).sqrt();
                    let s21 = Complex64::new(mag, 0.0);
                    [
                        Complex64::new(0.0, 0.0), // S11
                        s21,                      // S21
                        s21,                      // S12
                        Complex64::new(0.0, 0.0), // S22
                    ]
                })
                .collect()
        }
    }

    fn make_bandpass_link(f0_hz: f64, bw_hz: f64) -> SiPhLink {
        SiPhLink::new().push(LorentzianBandpass { f0_hz, bw_hz })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Frequency grid helper
    // ─────────────────────────────────────────────────────────────────────────

    fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
        if n < 2 {
            return vec![start];
        }
        let step = (stop - start) / (n - 1) as f64;
        (0..n).map(|i| start + step * i as f64).collect()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 1: cascade_identity_recovers_single_link
    // ─────────────────────────────────────────────────────────────────────────

    /// `chip_to_chip_link_response` with a single stage must return the same
    /// S-matrices as calling `link.cascade()` directly.
    #[test]
    fn cascade_identity_recovers_single_link() {
        let link = make_waveguide_link(5_000.0, 2.0, 4.2);
        let freq_grid = linspace(192.0e12, 194.0e12, 50);

        let from_cascade = link.cascade(&freq_grid);
        let from_wrapper = chip_to_chip_link_response(&[&link], &freq_grid);

        assert_eq!(
            from_cascade.len(),
            from_wrapper.len(),
            "Output length mismatch"
        );

        for (i, (&direct, &wrapped)) in from_cascade.iter().zip(from_wrapper.iter()).enumerate() {
            for k in 0..4 {
                let diff = (direct[k] - wrapped[k]).norm();
                assert!(
                    diff < 1e-12,
                    "S-matrix element [{k}] mismatch at freq index {i}: direct={:.6e}, wrapped={:.6e}, diff={:.3e}",
                    direct[k].norm(),
                    wrapped[k].norm(),
                    diff
                );
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2: cascade_two_lossless_links_doubles_phase
    // ─────────────────────────────────────────────────────────────────────────

    /// Two identical waveguide sections: the S21 phase of the cascade is
    /// approximately 2× that of a single section.
    ///
    /// The WaveguideSection model gives `S21 = A·exp(-i·φ)` with φ proportional
    /// to frequency and path length, so doubling the path (two identical links)
    /// doubles the phase.  Tested at a single frequency to avoid wrap-around
    /// ambiguity.
    #[test]
    fn cascade_two_lossless_links_doubles_phase() {
        // Very short, near-zero loss waveguide so amplitude stays ~1.0
        let f_test = 193.41e12_f64; // Hz (~1550 nm)
        let freq_grid = [f_test];

        let link = make_waveguide_link(1_000.0, 0.0, 4.0); // 1 µm, lossless, n_g=4
        let cascade_single = link.cascade(&freq_grid);
        let single_phase = cascade_single[0][1].arg(); // arg(S21) of one link

        let cascade_double = chip_to_chip_link_response(&[&link, &link], &freq_grid);
        let double_phase = cascade_double[0][1].arg(); // arg(S21) of two links

        // Phase difference between double and single should equal single_phase
        // i.e., double ≈ 2 * single  (modulo 2π — here magnitudes are tiny so
        // we compare the actual complex products instead)
        let expected = cascade_single[0][1] * cascade_single[0][1]; // single^2
        let actual = cascade_double[0][1];
        let diff = (actual - expected).norm();
        assert!(
            diff < 1e-12,
            "S21 of two identical phases should equal S21_single², got diff={diff:.3e} (single_phase={single_phase:.6}, double_phase={double_phase:.6})"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 3: cascade_reduces_bandwidth
    // ─────────────────────────────────────────────────────────────────────────

    /// Two identical Lorentzian bandpass stages cascaded: the resulting |S21|²
    /// drops to half-peak at a frequency closer to the centre than the single
    /// stage, i.e., the 3-dB bandwidth is narrower.
    ///
    /// For a Lorentzian of half-bandwidth `bw`: |S21|²(f) = 1/(1 + ((f-f0)/bw)²)
    /// which reaches 0.5 at |f-f0| = bw.  The squared cascade is
    /// |S21|⁴(f) = 1/(1 + ((f-f0)/bw)²)² which reaches 0.5 at |f-f0| = bw/√(√2−1).
    /// Since √(√2−1) < 1, the 3-dB bandwidth of the cascade is *narrower*.
    #[test]
    fn cascade_reduces_bandwidth() {
        let f0 = 193.41e12_f64;
        let bw = 50.0e9_f64; // 50 GHz half-bandwidth

        // Fine frequency grid around the passband
        let freq_grid = linspace(f0 - 5.0 * bw, f0 + 5.0 * bw, 1001);

        let link = make_bandpass_link(f0, bw);

        let single_s = chip_to_chip_link_response(&[&link], &freq_grid);
        let cascade_s = chip_to_chip_link_response(&[&link, &link], &freq_grid);

        // Find 3-dB bandwidth of single stage
        let single_bw = find_3db_bandwidth_hz(&single_s, &freq_grid);
        let cascade_bw = find_3db_bandwidth_hz(&cascade_s, &freq_grid);

        assert!(
            single_bw > 0.0,
            "Single stage should have a measurable 3-dB bandwidth (got {single_bw:.3e})"
        );
        assert!(
            cascade_bw > 0.0,
            "Cascade should have a measurable 3-dB bandwidth (got {cascade_bw:.3e})"
        );
        assert!(
            cascade_bw < single_bw,
            "Cascade bandwidth ({cascade_bw:.3e} Hz) should be narrower than single ({single_bw:.3e} Hz)"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 4: link_response_length_matches_freq_grid
    // ─────────────────────────────────────────────────────────────────────────

    /// Output length equals the length of the frequency grid, regardless of
    /// the number of stages.
    #[test]
    fn link_response_length_matches_freq_grid() {
        let link_a = make_waveguide_link(3_000.0, 1.0, 4.2);
        let link_b = make_waveguide_link(5_000.0, 2.0, 4.0);
        let freq_grid = linspace(190.0e12, 196.0e12, 64);

        let result = chip_to_chip_link_response(&[&link_a, &link_b], &freq_grid);
        assert_eq!(
            result.len(),
            freq_grid.len(),
            "Output length {}, expected {}",
            result.len(),
            freq_grid.len()
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 5: ber_sweep_higher_loss_gives_worse_ber
    // ─────────────────────────────────────────────────────────────────────────

    /// A link with higher insertion loss (used as dispersion penalty) gives
    /// equal-or-worse BER at the same OSNR point.
    #[test]
    fn ber_sweep_higher_loss_gives_worse_ber() {
        // Low-loss link: 5 mm at 2 dB/cm → 1 dB IL
        let low_loss_link = make_waveguide_link(5_000.0, 2.0, 4.2);
        // High-loss link: 5 mm at 12 dB/cm → 6 dB IL
        let high_loss_link = make_waveguide_link(5_000.0, 12.0, 4.2);

        let f_center = 193.41e12_f64;
        let osnr_grid: Vec<f64> = (0..30).map(|i| i as f64).collect();

        let curve_low = ber_vs_osnr_sweep_for_link(
            &low_loss_link,
            f_center,
            ModulationFormat::Bpsk,
            100.0,
            &osnr_grid,
        )
        .expect("ber_vs_osnr_sweep_for_link failed for low-loss link");

        let curve_high = ber_vs_osnr_sweep_for_link(
            &high_loss_link,
            f_center,
            ModulationFormat::Bpsk,
            100.0,
            &osnr_grid,
        )
        .expect("ber_vs_osnr_sweep_for_link failed for high-loss link");

        // At the midrange OSNR point, higher loss must give equal-or-worse BER
        let mid = osnr_grid.len() / 2;
        let ber_low = curve_low.ber[mid];
        let ber_high = curve_high.ber[mid];

        assert!(
            ber_high >= ber_low,
            "Higher-loss link should have ≥ BER at midrange OSNR: high_loss_BER={ber_high:.3e}, low_loss_BER={ber_low:.3e}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 6: ber_sweep_length_matches_osnr_grid
    // ─────────────────────────────────────────────────────────────────────────

    /// The returned `BerOsnrCurve` has the same number of BER values as
    /// OSNR grid points.
    #[test]
    fn ber_sweep_length_matches_osnr_grid() {
        let link = make_waveguide_link(2_000.0, 3.0, 4.2);
        let f_center = 193.41e12_f64;
        let osnr_db_grid: Vec<f64> = linspace(5.0, 25.0, 41);

        let curve = ber_vs_osnr_sweep_for_link(
            &link,
            f_center,
            ModulationFormat::Qpsk,
            200.0,
            &osnr_db_grid,
        )
        .expect("ber_vs_osnr_sweep_for_link should succeed");

        assert_eq!(
            curve.ber.len(),
            osnr_db_grid.len(),
            "BER vec length {} != OSNR grid length {}",
            curve.ber.len(),
            osnr_db_grid.len()
        );
        assert_eq!(
            curve.osnr_db.len(),
            osnr_db_grid.len(),
            "osnr_db vec length {} != OSNR grid length {}",
            curve.osnr_db.len(),
            osnr_db_grid.len()
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Utility: compute 3-dB bandwidth (Hz) from |S21|² via S-matrix slice
    // ─────────────────────────────────────────────────────────────────────────

    /// Estimate the 3-dB bandwidth of `|S21|²` using linear interpolation.
    ///
    /// Returns 0.0 if no half-power point is found.
    fn find_3db_bandwidth_hz(s_mats: &[[Complex64; 4]], freq_hz: &[f64]) -> f64 {
        let powers: Vec<f64> = s_mats.iter().map(|m| m[1].norm_sqr()).collect();
        let peak = powers.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if peak <= 0.0 {
            return 0.0;
        }
        let half_peak = peak / 2.0;

        // Index of peak
        let peak_idx = match powers
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        {
            Some((i, _)) => i,
            None => return 0.0,
        };

        // First crossing to the right
        let right_idx = match (peak_idx + 1..freq_hz.len()).find(|&i| powers[i] < half_peak) {
            Some(i) => i,
            None => return 0.0,
        };

        let p_lo = powers[right_idx - 1];
        let p_hi = powers[right_idx];
        if (p_hi - p_lo).abs() < 1e-40 {
            return 0.0;
        }
        let t = (half_peak - p_lo) / (p_hi - p_lo);
        let f_3db_right =
            freq_hz[right_idx - 1] + t * (freq_hz[right_idx] - freq_hz[right_idx - 1]);
        2.0 * (f_3db_right - freq_hz[peak_idx]).abs()
    }
}
