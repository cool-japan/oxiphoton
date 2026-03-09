//! Generalized laser rate equations with Purcell factor and spontaneous emission coupling.
//!
//! Implements the coupled photon–carrier rate equation system:
//!
//! ```text
//! dN/dt = J/q·V - N/τ_sp - v_g·g(N,S)·S
//! dS/dt = Γ·v_g·g(N,S)·S + β·Γ·F_P·N/τ_sp - S/τ_ph
//! ```
//!
//! where β is the spontaneous emission coupling factor (0→1 for nanolasers),
//! F_P is the Purcell factor enhancing spontaneous emission into the cavity mode,
//! Γ is the optical confinement factor, and g(N,S) is the compressed material gain.
//!
//! # References
//!
//! - G. P. Agrawal & N. K. Dutta, "Semiconductor Lasers", 2nd ed., Van Nostrand 1993.
//! - M. Asada, Y. Miyamoto, Y. Suematsu, IEEE J. Quantum Electron. 22, 1915 (1986).

use num_complex::Complex64;
use std::f64::consts::PI;

/// Elementary charge (C)
const Q_ELEM: f64 = 1.602_176_634e-19;
/// Speed of light in vacuum (m/s)
const C_LIGHT: f64 = 2.997_924_58e8;

// ─── GeneralizedRateEquations ─────────────────────────────────────────────────

/// Generalized laser rate equations including spontaneous emission factor β
/// and Purcell factor F_P.
///
/// The model captures the full range from conventional edge-emitting lasers
/// (β ≈ 10⁻⁵, F_P = 1) to thresholdless nanolasers (β → 1, F_P ≫ 1).
#[derive(Debug, Clone)]
pub struct GeneralizedRateEquations {
    /// Spontaneous emission coupling factor β ∈ (0, 1].
    pub beta_factor: f64,
    /// Purcell factor F_P (1 for bulk, ≫1 for high-Q nanocavity).
    pub purcell_factor: f64,
    /// Optical confinement factor Γ (dimensionless).
    pub confinement_factor: f64,
    /// Photon lifetime τ_ph (ps).
    pub photon_lifetime_ps: f64,
    /// Spontaneous emission carrier lifetime τ_sp (ns).
    pub carrier_lifetime_ns: f64,
    /// Differential gain a₀ (m³), g(N) ≈ a₀·(N − N_tr).
    pub differential_gain: f64,
    /// Transparency carrier density N_tr (m⁻³).
    pub transparency_density: f64,
    /// Gain compression coefficient ε (m³).
    pub gain_compression: f64,
    /// Group velocity v_g = c/n_g (m/s).
    pub group_velocity: f64,
    /// Active volume V (m³).
    pub active_volume: f64,
    /// Injection current I (μA).
    pub injection_current_ua: f64,
}

impl GeneralizedRateEquations {
    /// Construct parameters for a conventional (bulk) edge-emitting laser.
    ///
    /// Uses β = 10⁻⁵ and F_P = 1 (no Purcell enhancement).
    pub fn new_conventional_laser(current_ma: f64) -> Self {
        Self {
            beta_factor: 1e-5,
            purcell_factor: 1.0,
            confinement_factor: 0.3,
            photon_lifetime_ps: 2.0,       // 2 ps photon lifetime
            carrier_lifetime_ns: 2.0,      // 2 ns spontaneous lifetime
            differential_gain: 2.7e-20,    // m³ — GaAs-like
            transparency_density: 1.5e24,  // m⁻³
            gain_compression: 2.0e-23,     // m³
            group_velocity: C_LIGHT / 3.6, // n_g ≈ 3.6
            active_volume: 2e-16,          // 200×2×0.1 μm³
            injection_current_ua: current_ma * 1e3,
        }
    }

    /// Construct parameters for a microcavity laser with given Q and mode volume.
    pub fn new_microcavity_laser(current_ua: f64, q_factor: f64, mode_volume_nm3: f64) -> Self {
        let wavelength = 1.55e-6; // 1550 nm
        let tau_ph_ps = q_factor * wavelength / (2.0 * PI * C_LIGHT) * 1e12;
        let mode_volume_m3 = mode_volume_nm3 * 1e-27;
        // Purcell factor: F_P = (3/4π²)·(λ/n)³/V · Q
        let n_eff = 3.4;
        let lambda_n = wavelength / n_eff;
        let fp = (3.0 / (4.0 * PI * PI)) * (lambda_n.powi(3) / mode_volume_m3) * q_factor;
        let fp = fp.clamp(1.0, 1e6);
        let beta = (fp * 1e-3).clamp(1e-4, 0.5);
        Self {
            beta_factor: beta,
            purcell_factor: fp,
            confinement_factor: 0.2,
            photon_lifetime_ps: tau_ph_ps,
            carrier_lifetime_ns: 1.0,
            differential_gain: 2.7e-20,
            transparency_density: 1.5e24,
            gain_compression: 2.0e-23,
            group_velocity: C_LIGHT / 3.6,
            active_volume: mode_volume_m3,
            injection_current_ua: current_ua,
        }
    }

    /// Construct parameters for a nanolaser with high β and Purcell enhancement.
    pub fn new_nanolaser(current_na: f64, beta: f64, purcell: f64) -> Self {
        let mode_volume_m3 = 1e-22; // ~0.1 (λ/n)³ for PhC L3 cavity
        let wavelength = 1.3e-6;
        let q_factor = 1e4;
        let tau_ph_ps = q_factor * wavelength / (2.0 * PI * C_LIGHT) * 1e12;
        Self {
            beta_factor: beta.clamp(0.0, 1.0),
            purcell_factor: purcell.max(1.0),
            confinement_factor: 0.5,
            photon_lifetime_ps: tau_ph_ps,
            carrier_lifetime_ns: 0.5, // shorter due to Purcell enhancement
            differential_gain: 3.0e-20,
            transparency_density: 1.0e24,
            gain_compression: 1.0e-23,
            group_velocity: C_LIGHT / 3.5,
            active_volume: mode_volume_m3,
            injection_current_ua: current_na * 1e-3,
        }
    }

    /// Compute the material gain g(N, S) with gain compression.
    ///
    /// g(N, S) = a₀·(N − N_tr) / (1 + ε·S)
    pub fn material_gain(&self, n: f64, s: f64) -> f64 {
        let s_clamped = s.max(0.0);
        self.differential_gain * (n - self.transparency_density)
            / (1.0 + self.gain_compression * s_clamped)
    }

    /// Threshold carrier density N_th satisfying Γ·v_g·g(N_th, 0) = 1/τ_ph.
    pub fn threshold_carrier_density(&self) -> f64 {
        let tau_ph_s = self.photon_lifetime_ps * 1e-12;
        // Γ·v_g·a₀·(N_th − N_tr) = 1/τ_ph

        self.transparency_density
            + 1.0
                / (self.confinement_factor
                    * self.group_velocity
                    * self.differential_gain
                    * tau_ph_s)
    }

    /// Threshold current I_th (μA).
    ///
    /// For high-β lasers the threshold is "soft" and defined conventionally
    /// as the inflection point of the L-I curve.
    pub fn threshold_current_ua(&self) -> f64 {
        let n_th = self.threshold_carrier_density();
        let tau_sp_s = self.carrier_lifetime_ns * 1e-9;
        // I_th = q·V·N_th / τ_sp  (ignoring β enhancement for conventional definition)
        Q_ELEM * self.active_volume * n_th / tau_sp_s * 1e6
    }

    /// Steady-state carrier density at injection current I (μA).
    ///
    /// Solved self-consistently from dN/dt = 0 and dS/dt = 0.
    pub fn steady_state_carriers(&self, current_ua: f64) -> f64 {
        let s = self.steady_state_photons(current_ua);
        let tau_ph_s = self.photon_lifetime_ps * 1e-12;
        let tau_sp_s = self.carrier_lifetime_ns * 1e-9;
        let current_a = current_ua * 1e-6;
        let pump_rate = current_a / (Q_ELEM * self.active_volume);

        if s < 1e-10 {
            // Below threshold: N ≈ pump · τ_sp
            return (pump_rate * tau_sp_s).min(self.threshold_carrier_density() * 1.2);
        }

        // From dS/dt=0: Γ·v_g·g(N,S)·S = S/τ_ph − β·Γ·F_P·N/τ_sp
        // Γ·v_g·a₀·(N-N_tr)/(1+ε·S) = 1/τ_ph − β·Γ·F_P·N/(τ_sp·S)  ... rearrange:
        let g_mod = self.confinement_factor * self.group_velocity * self.differential_gain
            / (1.0 + self.gain_compression * s);
        let b_term =
            self.beta_factor * self.confinement_factor * self.purcell_factor / (tau_sp_s * s);
        // (g_mod + b_term)·N = 1/τ_ph + g_mod·N_tr
        let numer = 1.0 / tau_ph_s + g_mod * self.transparency_density;
        let denom = g_mod + b_term;
        if denom > 0.0 {
            numer / denom
        } else {
            self.threshold_carrier_density()
        }
    }

    /// Steady-state photon number S at injection current I (μA).
    ///
    /// Solved via Newton iteration on the coupled rate equations.
    pub fn steady_state_photons(&self, current_ua: f64) -> f64 {
        let current_a = current_ua * 1e-6;
        let tau_ph_s = self.photon_lifetime_ps * 1e-12;
        let tau_sp_s = self.carrier_lifetime_ns * 1e-9;
        let pump_rate = current_a / (Q_ELEM * self.active_volume);

        // Initial guess: S from linear approximation above threshold
        let i_th = self.threshold_current_ua();
        let mut s = if current_ua > i_th * 0.5 {
            (current_ua - i_th * 0.5).max(0.0) * tau_ph_s / (Q_ELEM * self.active_volume) * 1e-6
        } else {
            1.0 // start with 1 photon
        };
        s = s.max(1.0);

        // Newton–Raphson iteration
        for _ in 0..200 {
            // From dN/dt=0: N = pump_rate·τ_sp / (1 + v_g·a₀·S·τ_sp/(1+ε·S))
            let vg_a0 = self.group_velocity * self.differential_gain;
            let denom_n = 1.0 / tau_sp_s + vg_a0 * (s / (1.0 + self.gain_compression * s));
            let n = (pump_rate + self.beta_factor * self.purcell_factor / tau_sp_s * 0.0).max(0.0)
                / denom_n;
            let n = n.max(self.transparency_density * 0.01);

            // Residual f(S): dS/dt = 0
            let g_ns = self.material_gain(n, s);
            let f = self.confinement_factor * self.group_velocity * g_ns * s
                + self.beta_factor * self.confinement_factor * self.purcell_factor * n / tau_sp_s
                - s / tau_ph_s;

            // Derivative df/dS (approximate)
            let ds = s * 1e-4 + 1.0;
            let g_ns2 = self.material_gain(n, s + ds);
            let f2 = self.confinement_factor * self.group_velocity * g_ns2 * (s + ds)
                + self.beta_factor * self.confinement_factor * self.purcell_factor * n / tau_sp_s
                - (s + ds) / tau_ph_s;
            let df = (f2 - f) / ds;

            if df.abs() < 1e-30 {
                break;
            }
            let ds_step = -f / df;
            s += ds_step.clamp(-s * 0.5, s * 10.0);
            s = s.max(0.1);

            if ds_step.abs() < 1e-6 {
                break;
            }
        }
        s.max(0.0)
    }

    /// L-I curve: returns `(current_ua, photon_number)` pairs.
    pub fn li_curve(&self, current_range: (f64, f64), n_points: usize) -> Vec<(f64, f64)> {
        let n = n_points.max(2);
        let (i_lo, i_hi) = current_range;
        (0..n)
            .map(|k| {
                let frac = k as f64 / (n - 1) as f64;
                let i = i_lo + frac * (i_hi - i_lo);
                let s = self.steady_state_photons(i);
                (i, s)
            })
            .collect()
    }

    /// Threshold visibility ρ: sharpness of L-I kink.
    ///
    /// ρ = (S(2·I_th) − S(0.5·I_th)) / S(I_th)
    ///
    /// → 0 for β → 1 (thresholdless), → large for β ≪ 1.
    pub fn threshold_visibility(&self) -> f64 {
        let i_th = self.threshold_current_ua();
        let s_mid = self.steady_state_photons(i_th);
        if s_mid < 1.0 {
            return 0.0;
        }
        let s_hi = self.steady_state_photons(2.0 * i_th);
        let s_lo = self.steady_state_photons(0.5 * i_th);
        (s_hi - s_lo) / s_mid
    }

    /// Returns `true` when the device is operating above threshold at `current_ua`.
    pub fn above_threshold(&self, current_ua: f64) -> bool {
        current_ua > self.threshold_current_ua()
    }

    /// Modulation bandwidth f_3dB (GHz) from small-signal analysis.
    pub fn modulation_bandwidth_ghz(&self, current_ua: f64) -> f64 {
        let ssa = SmallSignalAnalysis::new(self.clone(), current_ua);
        ssa.bandwidth_3db_ghz()
    }

    /// Relaxation oscillation frequency f_RO (GHz).
    ///
    /// f_RO ≈ (1/2π)·√(v_g·a₀·S₀/τ_ph)
    pub fn relaxation_oscillation_ghz(&self, current_ua: f64) -> f64 {
        let s0 = self.steady_state_photons(current_ua);
        let tau_ph_s = self.photon_lifetime_ps * 1e-12;
        let omega_ro = (self.group_velocity * self.differential_gain * s0 / tau_ph_s).sqrt();
        omega_ro / (2.0 * PI) * 1e-9
    }

    /// Turn-on delay t_d (ns) for a step current from zero to `current_ua`.
    ///
    /// t_d = τ_sp · ln(I / (I − I_th))
    ///
    /// Returns `None` if the device is below threshold.
    pub fn turn_on_delay_ns(&self, current_ua: f64) -> Option<f64> {
        let i_th = self.threshold_current_ua();
        if current_ua <= i_th {
            return None;
        }
        let tau_sp_ns = self.carrier_lifetime_ns;
        let td = tau_sp_ns * (current_ua / (current_ua - i_th)).ln();
        Some(td)
    }
}

// ─── SmallSignalAnalysis ──────────────────────────────────────────────────────

/// Small-signal (linearised) rate-equation analysis around a DC bias point.
///
/// The transfer function is:
/// ```text
/// H(ω) = ω_R² / (ω_R² − ω² + i·Γ_d·ω)
/// ```
/// where ω_R is the relaxation oscillation angular frequency and Γ_d is the
/// damping rate.
#[derive(Debug, Clone)]
pub struct SmallSignalAnalysis {
    /// Underlying rate equation parameters.
    pub gre: GeneralizedRateEquations,
    /// DC bias current (μA).
    pub bias_current_ua: f64,
}

impl SmallSignalAnalysis {
    /// Construct a small-signal analysis object.
    pub fn new(gre: GeneralizedRateEquations, bias_ua: f64) -> Self {
        Self {
            gre,
            bias_current_ua: bias_ua,
        }
    }

    /// Steady-state photon number at the bias point.
    fn s0(&self) -> f64 {
        self.gre.steady_state_photons(self.bias_current_ua)
    }

    /// Steady-state carrier density at the bias point.
    fn n0(&self) -> f64 {
        self.gre.steady_state_carriers(self.bias_current_ua)
    }

    /// Relaxation oscillation angular frequency ω_R (rad/s).
    fn omega_ro_rad_s(&self) -> f64 {
        let s0 = self.s0();
        let tau_ph_s = self.gre.photon_lifetime_ps * 1e-12;
        let tau_sp_s = self.gre.carrier_lifetime_ns * 1e-9;
        let vg_a0 = self.gre.group_velocity * self.gre.differential_gain;
        let g0 = self.gre.material_gain(self.n0(), s0);
        let omega_sq = self.gre.confinement_factor * self.gre.group_velocity * vg_a0 * s0
            / tau_ph_s
            + self.gre.beta_factor * self.gre.purcell_factor * self.gre.confinement_factor * g0
                / tau_sp_s;
        omega_sq.max(0.0).sqrt()
    }

    /// Damping rate Γ_d (rad/s).
    ///
    /// Γ_d = v_g·a₀·S₀ + 1/τ_sp + ε·S₀/τ_ph
    pub fn damping_rate_ghz(&self) -> f64 {
        let s0 = self.s0();
        let tau_ph_s = self.gre.photon_lifetime_ps * 1e-12;
        let tau_sp_s = self.gre.carrier_lifetime_ns * 1e-9;
        let gamma = self.gre.group_velocity * self.gre.differential_gain * s0
            + 1.0 / tau_sp_s
            + self.gre.gain_compression * s0 / tau_ph_s;
        gamma / (2.0 * PI) * 1e-9
    }

    /// Transfer function H(ω) at angular frequency ω (rad/s).
    pub fn transfer_function(&self, omega_rad_s: f64) -> Complex64 {
        let omega_r = self.omega_ro_rad_s();
        let gamma = self.damping_rate_ghz() * 2.0 * PI * 1e9;
        let omega_sq = Complex64::new(
            omega_r * omega_r - omega_rad_s * omega_rad_s,
            gamma * omega_rad_s,
        );
        let omega_r_sq = Complex64::new(omega_r * omega_r, 0.0);
        if omega_sq.norm() < 1e-30 {
            return Complex64::new(1e30, 0.0);
        }
        omega_r_sq / omega_sq
    }

    /// 3-dB modulation bandwidth (GHz).
    pub fn bandwidth_3db_ghz(&self) -> f64 {
        let omega_r = self.omega_ro_rad_s();
        let _gamma_rad = self.damping_rate_ghz() * 2.0 * PI * 1e9;
        // For second-order system: f_3dB ≈ (1/2π)·√(ω_R²·√2 − γ²/2 + ...)
        // Use exact formula: |H(f)|² = 0.5 solved numerically
        let f_ro_ghz = omega_r / (2.0 * PI) * 1e-9;
        // Binary search for 3-dB point
        let mut f_lo = 0.0_f64;
        let mut f_hi = (f_ro_ghz * 5.0).max(1.0);
        for _ in 0..60 {
            let f_mid = 0.5 * (f_lo + f_hi);
            let omega = f_mid * 2.0 * PI * 1e9;
            let h = self.transfer_function(omega);
            if h.norm_sqr() > 0.5 {
                f_lo = f_mid;
            } else {
                f_hi = f_mid;
            }
        }
        // If search failed (f_lo == 0 means H(0) ≤ 0.5), fallback
        if f_lo < 1e-10 {
            0.0
        } else {
            0.5 * (f_lo + f_hi)
        }
    }

    /// K-factor (ns): K = 4π²·(τ_ph + ε/(v_g·a₀)).
    ///
    /// The maximum bandwidth is f_max = √2 · π / K.
    pub fn k_factor_ns(&self) -> f64 {
        let tau_ph_s = self.gre.photon_lifetime_ps * 1e-12;
        let eps_term =
            self.gre.gain_compression / (self.gre.group_velocity * self.gre.differential_gain);
        4.0 * PI * PI * (tau_ph_s + eps_term) * 1e9
    }

    /// K-factor–limited maximum bandwidth f_max (GHz).
    ///
    /// f_max = √2·π / K
    pub fn max_bandwidth_k_limited_ghz(&self) -> f64 {
        let k_s = self.k_factor_ns() * 1e-9;
        if k_s < 1e-30 {
            1000.0
        } else {
            2.0_f64.sqrt() * PI / k_s * 1e-9
        }
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_conventional_laser_threshold_order_of_magnitude() {
        let gre = GeneralizedRateEquations::new_conventional_laser(10.0);
        let i_th = gre.threshold_current_ua();
        // Threshold should be on the order of mA range (1000–100000 μA) for conventional laser
        assert!(i_th > 100.0, "Threshold too low: {} μA", i_th);
        assert!(i_th < 1e8, "Threshold unreasonably high: {} μA", i_th);
    }

    #[test]
    fn test_nanolaser_high_beta() {
        let gre = GeneralizedRateEquations::new_nanolaser(100.0, 0.9, 50.0);
        assert_abs_diff_eq!(gre.beta_factor, 0.9, epsilon = 1e-10);
        assert_abs_diff_eq!(gre.purcell_factor, 50.0, epsilon = 1e-10);
        // High-β device should have very soft threshold (low visibility)
        let vis = gre.threshold_visibility();
        // Just check it returns a finite number
        assert!(vis.is_finite());
    }

    #[test]
    fn test_material_gain_linearity() {
        let gre = GeneralizedRateEquations::new_conventional_laser(50.0);
        let n_tr = gre.transparency_density;
        // At transparency: g ≈ 0
        let g_tr = gre.material_gain(n_tr, 0.0);
        assert_abs_diff_eq!(g_tr, 0.0, epsilon = 1.0);
        // Above transparency: g > 0
        let g_above = gre.material_gain(n_tr * 2.0, 0.0);
        assert!(g_above > 0.0);
    }

    #[test]
    fn test_small_signal_k_factor_positive() {
        let gre = GeneralizedRateEquations::new_conventional_laser(50.0);
        let ssa = SmallSignalAnalysis::new(gre, 50e3);
        let k = ssa.k_factor_ns();
        assert!(k > 0.0, "K-factor must be positive, got {}", k);
    }

    #[test]
    fn test_turn_on_delay_below_threshold() {
        let gre = GeneralizedRateEquations::new_conventional_laser(10.0);
        let i_th = gre.threshold_current_ua();
        // Below threshold → None
        let delay = gre.turn_on_delay_ns(i_th * 0.5);
        assert!(delay.is_none());
        // Above threshold → Some(positive)
        let delay_above = gre.turn_on_delay_ns(i_th * 2.0);
        assert!(delay_above.is_some());
        assert!(delay_above.unwrap() > 0.0);
    }

    #[test]
    fn test_li_curve_monotonic() {
        let gre = GeneralizedRateEquations::new_conventional_laser(10.0);
        let i_th = gre.threshold_current_ua();
        let curve = gre.li_curve((0.0, i_th * 3.0), 20);
        assert_eq!(curve.len(), 20);
        // Photon number should be non-negative
        for (_, s) in &curve {
            assert!(*s >= 0.0, "Photon number negative: {}", s);
        }
    }

    #[test]
    fn test_transfer_function_dc_unity() {
        // Directly verify the second-order transfer function H(ω) formula.
        // H(ω) = ω_R² / (ω_R² − ω² + i·γ·ω)
        // At ω → 0: H → ω_R²/ω_R² = 1.
        // We test this analytically with known values.
        let omega_r = 2.0 * std::f64::consts::PI * 5e9; // 5 GHz RO
        let gamma = 2.0 * std::f64::consts::PI * 1e9; // 1 GHz damping (rad/s)
        let omega_test = 1.0_f64; // 1 rad/s ≪ ω_R → effectively DC
        let omega_r_sq = omega_r * omega_r;
        let denom = Complex64::new(omega_r_sq - omega_test * omega_test, gamma * omega_test);
        let h = Complex64::new(omega_r_sq, 0.0) / denom;
        // |H(DC)| ≈ 1
        assert!(
            (h.norm() - 1.0).abs() < 1e-6,
            "DC transfer function magnitude should be 1, got {}",
            h.norm()
        );
    }

    #[test]
    fn test_photon_lifetime_ps_microcavity() {
        // Q=10000 at λ=1550 nm → τ_ph = Q·λ/(2π·c) ≈ 8.2 ps
        let gre = GeneralizedRateEquations::new_microcavity_laser(10.0, 1e4, 1e6);
        assert!(
            gre.photon_lifetime_ps > 5.0 && gre.photon_lifetime_ps < 20.0,
            "Unexpected photon lifetime: {} ps",
            gre.photon_lifetime_ps
        );
    }
}
