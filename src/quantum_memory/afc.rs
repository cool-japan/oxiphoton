//! Atomic Frequency Comb (AFC) quantum memory protocol.
//!
//! The AFC protocol imprints a periodic absorption comb (spacing Δ) into an
//! inhomogeneously broadened atomic ensemble via spectral hole-burning.  An
//! absorbed photon re-emits as an echo at t = 1/Δ owing to the periodic
//! interference of the atomic dipoles.
//!
//! References:
//! - Afzelius et al., PRA 79, 052329 (2009): AFC multimode quantum memory
//! - de Riedmatten et al., Nature 456, 773 (2008): Solid-state quantum memory
//! - Tittel et al., Laser & Photon. Rev. 4, 244 (2010): AFC review
//! - Jobez et al., PRL 114, 230502 (2015): AFC + spin-wave storage

use std::f64::consts::PI;

// ─── Atomic frequency comb ────────────────────────────────────────────────────

/// Atomic Frequency Comb quantum memory.
///
/// An AFC consists of a train of N narrow absorption peaks (width γ, spacing Δ)
/// created by optical pumping into a dark state at selected frequency positions.
/// The finesse F = Δ/γ controls the echo efficiency.
#[derive(Debug, Clone)]
pub struct AtomicFrequencyComb {
    /// Comb-tooth spacing Δ  (Hz)
    pub peak_spacing_hz: f64,
    /// Individual tooth linewidth γ  (Hz)
    pub peak_width_hz: f64,
    /// Finesse F = Δ/γ  (dimensionless; > 1 for well-resolved teeth)
    pub finesse: f64,
    /// Resonant optical depth per comb tooth
    pub optical_depth: f64,
    /// Number of comb teeth N
    pub n_peaks: usize,
    /// Centre frequency of the comb  (Hz)
    pub center_frequency_hz: f64,
}

impl AtomicFrequencyComb {
    /// Echo time t_echo = 1 / Δ  (s).
    ///
    /// The periodic comb rephases all absorbed dipoles at this time.
    #[inline]
    pub fn echo_time_s(&self) -> f64 {
        1.0 / self.peak_spacing_hz.max(f64::MIN_POSITIVE)
    }

    /// AFC absorption efficiency.
    ///
    /// For F ≫ 1 and peak OD d the efficiency is:
    ///
    /// η_AFC = exp(−d/F) × (1 − exp(−d × F / 4))²
    ///
    /// Optimal d·F/4 ~ 1 gives η_AFC ~ 54 %; with impedance matching > 90 %.
    pub fn absorption_efficiency(&self) -> f64 {
        let d = self.optical_depth;
        let f = self.finesse.max(f64::MIN_POSITIVE);
        let forward_loss = (-d / f).exp();
        let absorption = 1.0 - (-d * f / 4.0).exp();
        forward_loss * absorption * absorption
    }

    /// Absorption coefficient α(ν) at frequency `freq_hz`.
    ///
    /// Computed as a sum of Lorentzian peaks centred at n × Δ
    /// (measured from `center_frequency_hz`), n = −N/2 … N/2.
    pub fn absorption_at(&self, freq_hz: f64) -> f64 {
        let half_width = self.peak_width_hz / 2.0;
        let n_half = (self.n_peaks / 2) as isize;
        let delta_f = freq_hz - self.center_frequency_hz;

        ((-n_half)..=(n_half as isize))
            .map(|n| {
                let center = n as f64 * self.peak_spacing_hz;
                let detuning = delta_f - center;
                // Lorentzian: A(ν) = γ/2 / (detuning² + (γ/2)²) normalised to peak OD
                let lorentzian = (half_width * half_width)
                    / (detuning * detuning + half_width * half_width);
                self.optical_depth * lorentzian
            })
            .sum()
    }

    /// Total memory bandwidth B ≈ N × Δ  (Hz).
    #[inline]
    pub fn bandwidth_hz(&self) -> f64 {
        self.n_peaks as f64 * self.peak_spacing_hz
    }

    /// Multimode storage capacity: N_modes = B × t_echo  (dimensionless).
    ///
    /// The time-bandwidth product gives the number of distinguishable
    /// temporal modes that can be stored simultaneously.
    #[inline]
    pub fn multimode_capacity(&self) -> f64 {
        self.bandwidth_hz() * self.echo_time_s()
    }

    /// Echo amplitude A(t) for a unit-amplitude input photon.
    ///
    /// The Fourier sum over comb teeth gives:
    ///
    /// A(t) = Σ_{n=−N/2}^{N/2} sinc(n π / F) × exp(i 2π n Δ t)
    ///
    /// Here we return the real part magnitude (observable amplitude).
    pub fn echo_amplitude_at(&self, t_s: f64) -> f64 {
        let n_half = (self.n_peaks / 2) as isize;
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;

        for n in (-n_half)..=(n_half as isize) {
            let n_f = n as f64;
            // sinc envelope from finite tooth linewidth
            let sinc_arg = n_f * PI / self.finesse.max(f64::MIN_POSITIVE);
            let sinc_val = if sinc_arg.abs() < 1e-12 {
                1.0
            } else {
                sinc_arg.sin() / sinc_arg
            };
            let phase = 2.0 * PI * n_f * self.peak_spacing_hz * t_s;
            re += sinc_val * phase.cos();
            im += sinc_val * phase.sin();
        }

        // Normalise by number of teeth
        let norm = self.n_peaks as f64;
        (re * re + im * im).sqrt() / norm
    }

    /// Minimum optical depth required to achieve `target_efficiency`.
    ///
    /// Solved numerically by scanning d from 0.1 to 20 at the current finesse.
    pub fn required_od_for_efficiency(&self, target_efficiency: f64) -> f64 {
        let f = self.finesse.max(f64::MIN_POSITIVE);
        let target = target_efficiency.clamp(0.0, 0.99);
        let mut probe = self.clone();

        let n_scan = 2000_usize;
        let d_max = 20.0_f64;
        let mut best_d = d_max;

        for i in 1..=n_scan {
            let d = i as f64 * d_max / n_scan as f64;
            probe.optical_depth = d;
            let eta = probe.absorption_efficiency();
            if eta >= target {
                best_d = d;
                // Refine with bisection
                let mut lo = d - d_max / n_scan as f64;
                let mut hi = d;
                for _ in 0..40 {
                    let mid = (lo + hi) / 2.0;
                    probe.optical_depth = mid;
                    if probe.absorption_efficiency() >= target {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                best_d = hi;
                break;
            }
            let _ = f; // suppress unused warning
        }
        best_d
    }

    /// AFC + spin-wave efficiency.
    ///
    /// After AFC pre-absorption the excitation is transferred to a long-lived
    /// spin wave by a control pulse with efficiency η_control (each direction).
    ///
    /// η_SW = η_AFC × η_control²
    pub fn spin_wave_efficiency(&self, control_efficiency: f64) -> f64 {
        let eta_afc = self.absorption_efficiency();
        let eta_ctrl = control_efficiency.clamp(0.0, 1.0);
        eta_afc * eta_ctrl * eta_ctrl
    }
}

// ─── AFC preparation ──────────────────────────────────────────────────────────

/// Spectral tailoring of an AFC via optical pumping / hole burning.
///
/// Starting from a thermally distributed inhomogeneous absorption profile,
/// atoms at comb frequencies are pumped to a dark state, leaving periodic
/// absorption peaks.
#[derive(Debug, Clone)]
pub struct AfcPreparation {
    /// Inhomogeneous absorption width (Doppler / crystal field)  (GHz)
    pub inhomogeneous_width_ghz: f64,
    /// Comb-tooth spacing  (Hz)
    pub comb_frequency_hz: f64,
    /// Number of comb teeth to prepare
    pub n_teeth: usize,
}

impl AfcPreparation {
    /// Peak optical depth after spectral tailoring.
    ///
    /// Each tooth captures a fraction Γ_inh / (N_teeth × γ) of the total OD.
    pub fn peak_optical_depth(&self, total_od: f64) -> f64 {
        let gamma_inh_hz = self.inhomogeneous_width_ghz * 1e9;
        let n_teeth_f = self.n_teeth.max(1) as f64;
        // tooth linewidth derived from finesse = comb_spacing / gamma_tooth ≈ 10
        let gamma_tooth_hz = self.comb_frequency_hz / 10.0;
        total_od * gamma_tooth_hz / (gamma_inh_hz / n_teeth_f)
    }

    /// Spectral preparation fidelity.
    ///
    /// Limited by the laser coherence time relative to the comb period:
    ///
    /// F_prep = exp(−1 / (laser_coherence_time_s × comb_frequency_hz))
    pub fn preparation_fidelity(&self, laser_coherence_time_s: f64) -> f64 {
        let coherence_product = laser_coherence_time_s * self.comb_frequency_hz;
        (-1.0 / coherence_product.max(f64::MIN_POSITIVE)).exp()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_afc() -> AtomicFrequencyComb {
        AtomicFrequencyComb {
            peak_spacing_hz: 1e9,   // 1 GHz
            peak_width_hz: 1e8,     // 100 MHz
            finesse: 10.0,
            optical_depth: 3.0,
            n_peaks: 10,
            center_frequency_hz: 3e14, // ~780 nm Rb D2
        }
    }

    #[test]
    fn afc_echo_time_ghz_comb() {
        let afc = test_afc();
        let t_echo = afc.echo_time_s();
        // 1/1GHz = 1 ns
        assert!(
            (t_echo - 1e-9).abs() < 1e-12,
            "t_echo={} s (expected 1 ns)",
            t_echo
        );
    }

    #[test]
    fn afc_absorption_efficiency_physical_range() {
        let afc = test_afc();
        let eta = afc.absorption_efficiency();
        assert!(eta > 0.0 && eta <= 1.0, "η_AFC={}", eta);
    }

    #[test]
    fn afc_absorption_peak_at_center() {
        let afc = test_afc();
        // At n=0 frequency the absorption should be maximal (OD contribution)
        let alpha_center = afc.absorption_at(afc.center_frequency_hz);
        let alpha_off = afc.absorption_at(afc.center_frequency_hz + 5e9); // far off
        assert!(
            alpha_center > alpha_off,
            "Center absorption {} should exceed off-resonance {}",
            alpha_center,
            alpha_off
        );
    }

    #[test]
    fn afc_bandwidth_hz() {
        let afc = test_afc();
        // 10 teeth × 1 GHz = 10 GHz
        assert!(
            (afc.bandwidth_hz() - 1e10).abs() < 1.0,
            "bandwidth={}",
            afc.bandwidth_hz()
        );
    }

    #[test]
    fn afc_multimode_capacity_integer_like() {
        let afc = test_afc();
        let n_modes = afc.multimode_capacity();
        // B×t_echo = N_teeth = 10
        assert!(
            (n_modes - 10.0).abs() < 1e-9,
            "N_modes={}",
            n_modes
        );
    }

    #[test]
    fn afc_echo_amplitude_peak_at_echo_time() {
        let afc = test_afc();
        let t_echo = afc.echo_time_s();
        let a_echo = afc.echo_amplitude_at(t_echo);
        let a_half = afc.echo_amplitude_at(t_echo / 2.0);
        // Echo amplitude at t_echo should exceed mid-point
        assert!(
            a_echo > a_half,
            "Echo amplitude at t_echo ({}) should exceed mid-point ({})",
            a_echo,
            a_half
        );
    }

    #[test]
    fn afc_required_od_for_50pct() {
        let mut afc = test_afc();
        afc.finesse = 5.0;
        let d_req = afc.required_od_for_efficiency(0.5);
        assert!(d_req > 0.0 && d_req <= 20.0, "d_req={}", d_req);
        // Verify it actually achieves ≥50 %
        afc.optical_depth = d_req;
        let eta = afc.absorption_efficiency();
        assert!(eta >= 0.499, "η={} < 50%", eta);
    }

    #[test]
    fn afc_spin_wave_efficiency_less_than_afc() {
        let afc = test_afc();
        let eta_sw = afc.spin_wave_efficiency(0.9);
        let eta_afc = afc.absorption_efficiency();
        assert!(eta_sw <= eta_afc, "η_SW must be ≤ η_AFC");
    }

    #[test]
    fn afc_preparation_fidelity_range() {
        let prep = AfcPreparation {
            inhomogeneous_width_ghz: 1.0,
            comb_frequency_hz: 1e9,
            n_teeth: 10,
        };
        // With 1 ms laser coherence → coherence_product = 1e6 → F ≈ 1
        let f = prep.preparation_fidelity(1e-3);
        assert!(f > 0.0 && f <= 1.0, "fidelity={}", f);
    }
}
