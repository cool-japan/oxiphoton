//! Coupled resonator optical waveguides (CROW) and coupled ring filters.
//!
//! Implements the tight-binding model for chains of coupled ring resonators
//! and the add-drop configuration for ring filters.

use num_complex::Complex64;
use std::f64::consts::PI;

/// Coupled Resonator Optical Waveguide (CROW) — N coupled ring resonators in series.
///
/// Based on the tight-binding model:
///   Pass-band bandwidth B = (2κ/π) · FSR
///   Group delay τ_g = N · (dφ/dω) at band center
#[derive(Debug, Clone)]
pub struct CoupledResonatorOW {
    /// Number of resonators
    pub n_resonators: usize,
    /// Free spectral range (Hz) of each resonator
    pub fsr: f64,
    /// Quality factor of each resonator
    pub q_factor: f64,
    /// Inter-resonator power coupling coefficient κ (dimensionless, 0 to 1)
    pub coupling_coeff: f64,
    /// Center operating wavelength (m)
    pub wavelength: f64,
}

impl CoupledResonatorOW {
    /// Create a new CROW structure.
    pub fn new(
        n_resonators: usize,
        fsr: f64,
        q_factor: f64,
        coupling_coeff: f64,
        wavelength: f64,
    ) -> Self {
        Self {
            n_resonators,
            fsr,
            q_factor,
            coupling_coeff: coupling_coeff.clamp(0.0, 1.0),
            wavelength,
        }
    }

    /// Pass-band center frequency (Hz).
    pub fn center_frequency(&self) -> f64 {
        // Speed of light / wavelength
        2.997_924_58e8 / self.wavelength
    }

    /// Pass-band bandwidth (Hz) of the CROW.
    ///
    /// In the tight-binding approximation:
    ///   B = (2/π) · κ · FSR
    pub fn bandwidth(&self) -> f64 {
        2.0 / PI * self.coupling_coeff * self.fsr
    }

    /// Group delay (s) at frequency f.
    ///
    /// In the tight-binding model at band center:
    ///   τ_g(ω) = N / FSR · 1/sqrt(1 - (Δω/B_half)²)
    /// where B_half = κ · FSR / π is the half-bandwidth.
    pub fn group_delay(&self, f: f64) -> f64 {
        let f0 = self.center_frequency();
        let df = f - f0;
        let b_half = self.coupling_coeff * self.fsr / PI;
        let t0 = self.n_resonators as f64 / self.fsr; // baseline group delay

        // Near band center, dispersion is minimal
        let normalized_df = if b_half > 0.0 { df / b_half } else { 0.0 };
        let denom = (1.0 - normalized_df * normalized_df).max(0.01).sqrt();
        t0 / denom
    }

    /// Transmission spectrum of the CROW (N resonators in series).
    ///
    /// Returns Vec of (frequency_Hz, transmission_linear).
    pub fn transmission_spectrum(
        &self,
        f_center: f64,
        span: f64,
        n_points: usize,
    ) -> Vec<(f64, f64)> {
        if n_points == 0 {
            return Vec::new();
        }
        (0..n_points)
            .map(|i| {
                let f = f_center - span / 2.0 + i as f64 / (n_points - 1).max(1) as f64 * span;
                let tm = self.total_transfer_matrix(f);
                // Transmission = |t_21|² or 1/|m_11|²
                let t = 1.0 / tm[0][0].norm_sqr().max(1e-30);
                (f, t.min(1.0))
            })
            .collect()
    }

    /// Transfer matrix for a single resonator at frequency f.
    ///
    /// Uses the all-pass ring resonator transfer matrix:
    ///   M = 1/(t_c) · [[1, -r_c·exp(iφ)], [r_c·exp(-iφ), -exp(iφ)]]
    /// where φ = 2πf/FSR (round-trip phase) and r_c, t_c are the field coupling coefficients.
    fn single_resonator_tm(&self, f: f64) -> [[Complex64; 2]; 2] {
        let f0 = self.center_frequency();
        let phi = 2.0 * PI * (f - f0) / self.fsr;
        let kappa_field = self.coupling_coeff.sqrt(); // power to field
        let tau_field = (1.0 - self.coupling_coeff).sqrt();
        // Loss per round trip from Q factor
        let omega0 = 2.0 * PI * f0;
        let delta_omega = if self.q_factor > 0.0 {
            omega0 / self.q_factor
        } else {
            0.0
        };
        let alpha_rt = (-PI * delta_omega / self.fsr / omega0).exp(); // amplitude loss per RT
        let exp_phi = Complex64::new(0.0, phi).exp();
        let exp_phi_neg = Complex64::new(0.0, -phi).exp();
        let i_c = Complex64::new(0.0, kappa_field);

        // Through-port transfer matrix for all-pass ring
        let denom = Complex64::new(tau_field, 0.0);
        let a_rt = alpha_rt * exp_phi;
        let m11 = (Complex64::new(1.0, 0.0) - denom * a_rt) / denom;
        let m12 = -i_c * a_rt / denom;
        let m21 = i_c * exp_phi_neg / denom;
        let m22 = Complex64::new(-1.0, 0.0) / denom;

        [[m11, m12], [m21, m22]]
    }

    /// Total transfer matrix for N coupled resonators (cascade).
    pub fn total_transfer_matrix(&self, f: f64) -> [[Complex64; 2]; 2] {
        let mut total = identity_2x2();
        for _ in 0..self.n_resonators {
            let m = self.single_resonator_tm(f);
            total = mat_mul_2x2(total, m);
        }
        total
    }

    /// Group velocity dispersion of the CROW (s²/rad).
    ///
    /// β₂ = d²β/dω² ≈ (1/FSR)² · κ⁻² · N at band center.
    pub fn group_velocity_dispersion(&self) -> f64 {
        if self.coupling_coeff <= 0.0 || self.fsr <= 0.0 {
            return f64::INFINITY;
        }
        // d²τ_g/dω at band center gives β₂ per unit length (using resonator as length unit)
        let omega_fsr = 2.0 * PI * self.fsr;
        1.0 / (omega_fsr * omega_fsr * self.coupling_coeff * self.coupling_coeff)
    }

    /// Slow-light factor ng/n_eff at the band edge.
    ///
    /// At the band edge of the CROW, ng/n ≈ π/(2κ·FSR·τ_0)
    /// where τ_0 = 1/FSR is the single-resonator round-trip time.
    pub fn slow_light_factor(&self) -> f64 {
        if self.coupling_coeff <= 0.0 {
            return f64::INFINITY;
        }
        PI / (2.0 * self.coupling_coeff)
    }

    /// Insertion loss (dB) for the CROW.
    ///
    /// Each resonator contributes a loss from finite Q:
    ///   IL = N · 10·log10(1 - π·Δf_res/FSR)
    /// where Δf_res = f0/Q is the resonator linewidth.
    pub fn insertion_loss_db(&self) -> f64 {
        let f0 = self.center_frequency();
        let linewidth = if self.q_factor > 0.0 {
            f0 / self.q_factor
        } else {
            0.0
        };
        let loss_per_resonator = PI * linewidth / self.fsr;
        let total_loss = 1.0 - (1.0 - loss_per_resonator.min(0.999)).powi(self.n_resonators as i32);
        if total_loss <= 0.0 {
            return 0.0;
        }
        -10.0 * (1.0 - total_loss).max(1e-30).log10()
    }
}

/// 2×2 identity matrix.
fn identity_2x2() -> [[Complex64; 2]; 2] {
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    [[one, zero], [zero, one]]
}

/// 2×2 complex matrix multiplication.
fn mat_mul_2x2(a: [[Complex64; 2]; 2], b: [[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
    let mut c = [[Complex64::new(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Coupled ring resonator filter (add-drop configuration).
///
/// A single ring with input and drop couplers, characterized by:
///   through-port: T = |Et_thru|²
///   drop-port:    T = |Et_drop|²
#[derive(Debug, Clone)]
pub struct CoupledRingFilter {
    /// Ring radius (m)
    pub radius: f64,
    /// Effective mode index
    pub n_eff: f64,
    /// Input coupler power coupling coefficient κ₁ (0 to 1)
    pub kappa1: f64,
    /// Drop coupler power coupling coefficient κ₂ (0 to 1)
    pub kappa2: f64,
    /// Round-trip amplitude loss (Np/m, power loss α = 2·loss_per_m)
    pub alpha: f64,
    /// Operating wavelength (m)
    pub wavelength: f64,
}

impl CoupledRingFilter {
    /// Create a new coupled ring filter.
    pub fn new(
        radius: f64,
        n_eff: f64,
        kappa1: f64,
        kappa2: f64,
        alpha: f64,
        wavelength: f64,
    ) -> Self {
        Self {
            radius,
            n_eff,
            kappa1: kappa1.clamp(0.0, 1.0),
            kappa2: kappa2.clamp(0.0, 1.0),
            alpha,
            wavelength,
        }
    }

    /// Round-trip phase φ = 2π·n_eff·L/λ where L = 2πR.
    pub fn round_trip_phase(&self) -> f64 {
        let circumference = 2.0 * PI * self.radius;
        2.0 * PI * self.n_eff * circumference / self.wavelength
    }

    /// Round-trip amplitude transmission (including loss).
    fn round_trip_amplitude(&self) -> f64 {
        let circumference = 2.0 * PI * self.radius;
        (-self.alpha * circumference).exp()
    }

    /// Finesse of the ring resonator.
    ///
    /// F = π · sqrt(a · t1 · t2) / (1 - a · t1 · t2)
    /// where a is the round-trip amplitude, t1 = sqrt(1-κ1), t2 = sqrt(1-κ2).
    pub fn finesse(&self) -> f64 {
        let a = self.round_trip_amplitude();
        let t1 = (1.0 - self.kappa1).sqrt();
        let t2 = (1.0 - self.kappa2).sqrt();
        let r = a * t1 * t2;
        if r >= 1.0 {
            return f64::INFINITY;
        }
        PI * r.sqrt() / (1.0 - r)
    }

    /// Through-port and drop-port power transmission at the resonance detuning.
    ///
    /// Uses the standard coupled-resonator transfer matrix.
    /// Returns (T_through, T_drop) as power fractions.
    pub fn transmission(&self) -> (f64, f64) {
        let a = self.round_trip_amplitude();
        let t1 = (1.0 - self.kappa1).sqrt();
        let t2 = (1.0 - self.kappa2).sqrt();
        let phi = self.round_trip_phase();
        let exp_phi = Complex64::new(0.0, phi).exp();

        // Through-port field transfer function (add-drop ring):
        // E_t / E_in = (t1 - a·t2·exp(iφ)) / (1 - a·t1·t2·exp(iφ))
        let numer_t = Complex64::new(t1, 0.0) - a * t2 * exp_phi;
        let denom = Complex64::new(1.0, 0.0) - a * t1 * t2 * exp_phi;

        if denom.norm() < 1e-30 {
            return (0.0, 0.0);
        }

        let t_through = (numer_t / denom).norm_sqr();

        // Drop-port: E_d / E_in = -kappa1·kappa2·sqrt(a)·exp(iφ/2) / denom
        let k1_amp = self.kappa1.sqrt();
        let k2_amp = self.kappa2.sqrt();
        let numer_d = -k1_amp * k2_amp * a.sqrt() * Complex64::new(0.0, phi / 2.0).exp();
        let t_drop = (numer_d / denom).norm_sqr();

        (t_through.clamp(0.0, 1.0), t_drop.clamp(0.0, 1.0))
    }

    /// Extinction ratio of the through port (dB).
    ///
    /// ER = 10·log10(T_max / T_min) over one FSR.
    pub fn extinction_ratio_db(&self) -> f64 {
        let (t_off, _) = self.transmission(); // at current phase (could be on-resonance)
                                              // Maximum transmission is ≈ 1 (off resonance); find minimum
        let fsr = self.free_spectral_range();
        let n_scan = 200usize;
        let mut t_min = 1.0_f64;
        for i in 0..n_scan {
            let delta_lambda = i as f64 / n_scan as f64 * self.wavelength * self.wavelength / fsr;
            let detuned = CoupledRingFilter::new(
                self.radius,
                self.n_eff,
                self.kappa1,
                self.kappa2,
                self.alpha,
                self.wavelength + delta_lambda,
            );
            let (t, _) = detuned.transmission();
            if t < t_min {
                t_min = t;
            }
        }
        if t_min <= 0.0 {
            return 60.0;
        }
        let t_max = t_off.max(1.0 - t_min); // approximate max from symmetry
        if t_max / t_min.max(1e-10) <= 1.0 {
            return 0.0;
        }
        10.0 * (t_max / t_min.max(1e-10)).log10()
    }

    /// Free spectral range (Hz).
    ///
    /// FSR = c / (n_g · 2πR) ≈ c / (n_eff · 2πR) (ignoring dispersion).
    pub fn free_spectral_range(&self) -> f64 {
        let c = 2.997_924_58e8;
        let circumference = 2.0 * PI * self.radius;
        c / (self.n_eff * circumference)
    }

    /// Resonance linewidth (Hz) = FSR / Finesse.
    pub fn linewidth_hz(&self) -> f64 {
        let finesse = self.finesse();
        if finesse <= 0.0 || finesse.is_infinite() {
            return 0.0;
        }
        self.free_spectral_range() / finesse
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn crow_4() -> CoupledResonatorOW {
        // 4 coupled resonators, FSR = 100 GHz, Q = 1e4, κ = 0.1
        CoupledResonatorOW::new(4, 100e9, 1e4, 0.1, 1.55e-6)
    }

    fn ring_filter() -> CoupledRingFilter {
        // R = 5 μm, n_eff = 2.4, κ1 = κ2 = 0.1, α = 10 /m
        CoupledRingFilter::new(5e-6, 2.4, 0.1, 0.1, 10.0, 1.55e-6)
    }

    #[test]
    fn crow_center_frequency_positive() {
        let crow = crow_4();
        assert!(crow.center_frequency() > 0.0);
    }

    #[test]
    fn crow_bandwidth_positive() {
        let crow = crow_4();
        assert!(crow.bandwidth() > 0.0);
    }

    #[test]
    fn crow_bandwidth_less_than_fsr() {
        let crow = crow_4();
        assert!(crow.bandwidth() < crow.fsr);
    }

    #[test]
    fn crow_group_delay_at_center_finite() {
        let crow = crow_4();
        let f0 = crow.center_frequency();
        let tau = crow.group_delay(f0);
        assert!(tau.is_finite() && tau > 0.0, "group delay = {tau}");
    }

    #[test]
    fn crow_transmission_spectrum_length() {
        let crow = crow_4();
        let f0 = crow.center_frequency();
        let spec = crow.transmission_spectrum(f0, 200e9, 100);
        assert_eq!(spec.len(), 100);
    }

    #[test]
    fn crow_total_tm_is_2x2() {
        let crow = crow_4();
        let f0 = crow.center_frequency();
        let tm = crow.total_transfer_matrix(f0);
        for row in &tm {
            for &c in row {
                assert!(c.re.is_finite() && c.im.is_finite());
            }
        }
    }

    #[test]
    fn crow_gvd_positive() {
        let crow = crow_4();
        let gvd = crow.group_velocity_dispersion();
        assert!(gvd > 0.0, "GVD = {gvd}");
    }

    #[test]
    fn crow_slow_light_greater_than_one() {
        let crow = crow_4();
        assert!(crow.slow_light_factor() > 1.0);
    }

    #[test]
    fn crow_insertion_loss_non_negative() {
        let crow = crow_4();
        assert!(crow.insertion_loss_db() >= 0.0);
    }

    #[test]
    fn ring_round_trip_phase_positive() {
        let r = ring_filter();
        assert!(r.round_trip_phase() > 0.0);
    }

    #[test]
    fn ring_finesse_positive() {
        let r = ring_filter();
        assert!(r.finesse() > 0.0);
    }

    #[test]
    fn ring_transmission_sum_leq_one() {
        let r = ring_filter();
        let (t_thru, t_drop) = r.transmission();
        // Due to round-trip loss, T_thru + T_drop ≤ 1
        assert!(
            t_thru + t_drop <= 1.0 + 1e-10,
            "T_thru={t_thru:.4}, T_drop={t_drop:.4}"
        );
        assert!(t_thru >= 0.0 && t_drop >= 0.0);
    }

    #[test]
    fn ring_fsr_positive() {
        let r = ring_filter();
        assert!(r.free_spectral_range() > 0.0);
    }

    #[test]
    fn ring_linewidth_less_than_fsr() {
        let r = ring_filter();
        assert!(r.linewidth_hz() < r.free_spectral_range());
    }

    #[test]
    fn ring_q_increases_finesse() {
        let r1 = CoupledRingFilter::new(5e-6, 2.4, 0.05, 0.05, 5.0, 1.55e-6);
        let r2 = CoupledRingFilter::new(5e-6, 2.4, 0.1, 0.1, 10.0, 1.55e-6);
        // Lower loss + lower coupling → higher finesse
        assert!(
            r1.finesse() > r2.finesse(),
            "F1={:.1}, F2={:.1}",
            r1.finesse(),
            r2.finesse()
        );
    }

    #[test]
    fn mat_mul_identity() {
        let i = identity_2x2();
        let m = [
            [Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
            [Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)],
        ];
        let result = mat_mul_2x2(i, m);
        for r in 0..2 {
            for c in 0..2 {
                assert_relative_eq!(result[r][c].re, m[r][c].re, epsilon = 1e-12);
                assert_relative_eq!(result[r][c].im, m[r][c].im, epsilon = 1e-12);
            }
        }
    }
}
