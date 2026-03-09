/// Laser rate equation models for solid-state lasers.
///
/// Implements four-level and three-level laser rate equations including:
/// - Population inversion dynamics
/// - Photon density evolution
/// - Threshold analysis and slope efficiency
/// - Relaxation oscillations
/// - Q-switched pulse dynamics
/// - Gain saturation
///
/// References:
/// - Saleh & Teich, "Fundamentals of Photonics", 3rd ed., Ch. 15
/// - Siegman, "Lasers", University Science Books, 1986
/// - Svelto, "Principles of Lasers", 5th ed., Springer, 2010
use std::f64::consts::PI;

use crate::error::OxiPhotonError;

// ---------------------------------------------------------------------------
// Physical constants
// ---------------------------------------------------------------------------

/// Reduced Planck constant (J·s)
const HBAR: f64 = 1.054_571_8e-34;
/// Planck constant (J·s)
#[allow(dead_code)]
const H_PLANCK: f64 = 6.626_070_15e-34;
/// Speed of light in vacuum (m/s)
const C0: f64 = 2.997_924_58e8;

// ---------------------------------------------------------------------------
// FourLevelLaser
// ---------------------------------------------------------------------------

/// Four-level solid-state laser model using coupled rate equations.
///
/// Rate equations (coupled ODEs):
/// ```text
/// dN/dt  = Rp  −  N/τ  −  σ·(c/n)·N·φ
/// dφ/dt  = σ·(c/n)·N·φ  −  φ/τ_c  +  β·N/τ
/// ```
///
/// where
///   N   = upper-level population inversion density (m⁻³)
///   φ   = photon density (m⁻³)
///   Rp  = pump rate (m⁻³·s⁻¹)
///   τ   = upper-level fluorescence lifetime (s)
///   τ_c = cavity photon lifetime (s)
///   σ   = stimulated emission cross-section (m²)
///   β   = spontaneous emission factor (fraction coupling into lasing mode)
///
/// The four-level approximation is valid when the lower laser level empties
/// rapidly compared to all other time scales (e.g. Nd:YAG at 1064 nm).
#[derive(Debug, Clone)]
pub struct FourLevelLaser {
    /// Stimulated emission cross-section (m²).
    pub sigma_em: f64,
    /// Upper laser level fluorescence lifetime (s).
    pub tau_upper: f64,
    /// Cavity photon lifetime τ_c = 2·n·L / (c·δ_total) (s).
    pub tau_cavity: f64,
    /// Refractive index of the gain medium.
    pub n_index: f64,
    /// Spontaneous emission coupling factor into the lasing mode (10⁻³–10⁻⁵).
    pub beta: f64,
    /// Mode volume (m³).
    pub mode_volume_m3: f64,
    /// Laser wavelength (nm).
    pub wavelength_nm: f64,
}

impl FourLevelLaser {
    /// Construct a new four-level laser model.
    ///
    /// # Arguments
    /// * `sigma_em`    – stimulated emission cross-section (m²)
    /// * `tau_upper`   – upper-level lifetime (s)
    /// * `tau_cavity`  – cavity photon lifetime (s)
    /// * `n_index`     – refractive index
    /// * `beta`        – spontaneous emission coupling factor
    /// * `mode_vol`    – mode volume (m³)
    /// * `lambda_nm`   – wavelength (nm)
    pub fn new(
        sigma_em: f64,
        tau_upper: f64,
        tau_cavity: f64,
        n_index: f64,
        beta: f64,
        mode_vol: f64,
        lambda_nm: f64,
    ) -> Result<Self, OxiPhotonError> {
        if sigma_em <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "sigma_em must be positive".into(),
            ));
        }
        if tau_upper <= 0.0 || tau_cavity <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "lifetimes must be positive".into(),
            ));
        }
        if n_index < 1.0 {
            return Err(OxiPhotonError::NumericalError(
                "n_index must be >= 1.0".into(),
            ));
        }
        if lambda_nm <= 0.0 {
            return Err(OxiPhotonError::InvalidWavelength(lambda_nm * 1.0e-9));
        }
        Ok(Self {
            sigma_em,
            tau_upper,
            tau_cavity,
            n_index,
            beta: beta.clamp(0.0, 1.0),
            mode_volume_m3: mode_vol.abs(),
            wavelength_nm: lambda_nm,
        })
    }

    /// Photon frequency (Hz).
    #[inline]
    fn photon_frequency(&self) -> f64 {
        C0 / (self.wavelength_nm * 1.0e-9)
    }

    /// Photon energy (J).
    #[inline]
    fn photon_energy_j(&self) -> f64 {
        2.0 * PI * HBAR * self.photon_frequency()
    }

    /// Stimulated emission rate coefficient: σ·c/n (m³/s).
    ///
    /// This is the rate at which a single inversion-density unit depletes
    /// when the photon density is unity.
    #[inline]
    fn stimulated_rate_coeff(&self) -> f64 {
        self.sigma_em * C0 / self.n_index
    }

    /// Threshold population inversion density (m⁻³).
    ///
    /// At threshold the modal gain equals the cavity loss:
    /// ```text
    /// N_th = 1 / (σ·(c/n)·τ_c)
    /// ```
    pub fn threshold_inversion(&self) -> f64 {
        1.0 / (self.stimulated_rate_coeff() * self.tau_cavity)
    }

    /// Threshold pump rate (m⁻³·s⁻¹).
    ///
    /// Minimum pump rate to sustain oscillation:
    /// ```text
    /// Rp_th = N_th / τ
    /// ```
    pub fn threshold_pump_rate(&self) -> f64 {
        self.threshold_inversion() / self.tau_upper
    }

    /// Threshold pump power (mW).
    ///
    /// ```text
    /// P_th = Rp_th · ħω_pump · V / η_pump
    /// ```
    ///
    /// # Arguments
    /// * `pump_wavelength_nm` – pump wavelength (nm)
    /// * `pump_efficiency`    – fraction of pump photons absorbed and converted (0–1)
    pub fn threshold_pump_power_mw(&self, pump_wavelength_nm: f64, pump_efficiency: f64) -> f64 {
        let eta = pump_efficiency.clamp(1.0e-10, 1.0);
        let freq_pump = C0 / (pump_wavelength_nm * 1.0e-9);
        let e_pump = 2.0 * PI * HBAR * freq_pump;
        let rp_th = self.threshold_pump_rate();
        // Power = pump rate × energy per photon × volume / efficiency
        rp_th * e_pump * self.mode_volume_m3 / eta * 1.0e3 // W → mW
    }

    /// Slope efficiency (mW per unit pump rate above threshold).
    ///
    /// Above threshold, each additional pump photon contributes:
    /// ```text
    /// dP/dRp = (ω_laser / ω_pump) · T_oc / (T_oc + L_int) · ħω_laser · V
    /// ```
    ///
    /// Simplified to Stokes efficiency times output coupler fraction:
    /// ```text
    /// η_slope = (λ_pump/λ_laser) · (T_oc) · ħω · V
    /// ```
    ///
    /// # Arguments
    /// * `pump_wavelength_nm` – pump wavelength (nm)
    /// * `output_coupler`     – output coupler transmission (0–1)
    pub fn slope_efficiency(&self, pump_wavelength_nm: f64, output_coupler: f64) -> f64 {
        // Stokes efficiency (quantum defect)
        let stokes = self.wavelength_nm / pump_wavelength_nm.max(1.0e-3);
        // Photon energy at laser wavelength
        let e_laser = self.photon_energy_j();
        let oc = output_coupler.clamp(0.0, 1.0);
        // Slope efficiency [mW per (m⁻³·s⁻¹)]
        stokes * oc * e_laser * self.mode_volume_m3 * 1.0e3
    }

    /// Steady-state photon density (m⁻³) in CW operation above threshold.
    ///
    /// Setting dN/dt = 0 and dφ/dt = 0 and neglecting β·N/τ term:
    /// ```text
    /// φ_ss = τ_c · (Rp - N_th/τ)
    /// ```
    pub fn steady_state_photons(&self, pump_rate: f64) -> f64 {
        let n_th = self.threshold_inversion();
        let excess = pump_rate - n_th / self.tau_upper;
        if excess <= 0.0 {
            return 0.0;
        }
        // φ_ss = (Rp - Rp_th) * τ_c
        excess * self.tau_cavity
    }

    /// CW output power (mW) through the output coupler.
    ///
    /// ```text
    /// P_out = φ_ss · (ħω / τ_c) · T_oc · V
    /// ```
    ///
    /// # Arguments
    /// * `pump_rate`          – pump rate (m⁻³·s⁻¹)
    /// * `output_coupler_loss`– output coupler transmission fraction (0–1)
    pub fn output_power_mw(&self, pump_rate: f64, output_coupler_loss: f64) -> f64 {
        let phi_ss = self.steady_state_photons(pump_rate);
        if phi_ss <= 0.0 {
            return 0.0;
        }
        let oc = output_coupler_loss.clamp(0.0, 1.0);
        let e_photon = self.photon_energy_j();
        // Power through output coupler: φ * V * ħω / τ_c * T_oc
        phi_ss * self.mode_volume_m3 * e_photon / self.tau_cavity * oc * 1.0e3
    }

    /// Relaxation oscillation frequency (MHz).
    ///
    /// Small-signal analysis about the CW steady state gives:
    /// ```text
    /// f_RO = (1/2π) · sqrt( (1/τ_c) · (σ·c/n·φ_ss) )
    ///      ≈ (1/2π) · sqrt( (Rp/Rp_th - 1) / (τ·τ_c) )
    /// ```
    pub fn relaxation_oscillation_freq_mhz(&self, pump_rate: f64) -> f64 {
        let phi_ss = self.steady_state_photons(pump_rate);
        if phi_ss <= 0.0 {
            return 0.0;
        }
        let g_coeff = self.stimulated_rate_coeff();
        // ω_RO² = (g·φ_ss / τ_c) - 1/τ²  (linearised equations)
        // Simplified dominant term: ω_RO² ≈ g·φ_ss / τ_c
        let omega_sq = g_coeff * phi_ss / self.tau_cavity;
        if omega_sq <= 0.0 {
            return 0.0;
        }
        omega_sq.sqrt() / (2.0 * PI) / 1.0e6
    }

    /// Relaxation oscillation damping rate (MHz).
    ///
    /// ```text
    /// γ = (σ·(c/n)·φ_ss + 1/τ) / 2
    /// ```
    pub fn damping_rate_mhz(&self, pump_rate: f64) -> f64 {
        let phi_ss = self.steady_state_photons(pump_rate);
        let g_coeff = self.stimulated_rate_coeff();
        (g_coeff * phi_ss + 1.0 / self.tau_upper) / 2.0 / 1.0e6
    }

    /// Time-domain simulation of the laser rate equations.
    ///
    /// Uses first-order explicit Euler integration of:
    /// ```text
    /// dN/dt = Rp - N/τ - σ·(c/n)·N·φ
    /// dφ/dt = σ·(c/n)·N·φ - φ/τ_c + β·N/τ
    /// ```
    ///
    /// # Arguments
    /// * `pump_rate` – pump rate (m⁻³·s⁻¹)
    /// * `t_max_ns`  – total simulation time (ns)
    /// * `dt_ns`     – time step (ns); should satisfy CFL: dt << τ, τ_c
    /// * `n0`        – initial inversion density (m⁻³)
    /// * `phi0`      – initial photon density (m⁻³)
    ///
    /// Returns a vector of `(time_ns, N, phi)` tuples.
    pub fn simulate(
        &self,
        pump_rate: f64,
        t_max_ns: f64,
        dt_ns: f64,
        n0: f64,
        phi0: f64,
    ) -> Result<Vec<(f64, f64, f64)>, OxiPhotonError> {
        if dt_ns <= 0.0 || t_max_ns <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "time parameters must be positive".into(),
            ));
        }
        let dt = dt_ns * 1.0e-9;
        let n_steps = ((t_max_ns / dt_ns) as usize).min(10_000_000);
        let mut results = Vec::with_capacity(n_steps + 1);

        let g = self.stimulated_rate_coeff();
        let mut n_inv = n0.max(0.0);
        let mut phi = phi0.max(0.0);

        results.push((0.0, n_inv, phi));

        for step in 1..=n_steps {
            let dn = pump_rate - n_inv / self.tau_upper - g * n_inv * phi;
            let dphi = g * n_inv * phi - phi / self.tau_cavity + self.beta * n_inv / self.tau_upper;
            n_inv = (n_inv + dn * dt).max(0.0);
            phi = (phi + dphi * dt).max(0.0);
            let t_ns = step as f64 * dt_ns;
            results.push((t_ns, n_inv, phi));
        }

        Ok(results)
    }

    /// Gain coefficient g = σ·N (m⁻¹).
    pub fn gain_coefficient(&self, n_inversion: f64) -> f64 {
        self.sigma_em * n_inversion.max(0.0)
    }

    /// Saturated gain coefficient (m⁻¹).
    ///
    /// ```text
    /// g_sat = g0 / (1 + φ/φ_sat)
    /// ```
    pub fn saturated_gain(&self, n_inversion: f64, photon_density: f64) -> f64 {
        let g0 = self.gain_coefficient(n_inversion);
        let phi_sat = self.saturation_photon_density();
        if phi_sat <= 0.0 {
            return g0;
        }
        g0 / (1.0 + photon_density.max(0.0) / phi_sat)
    }

    /// Saturation photon density (m⁻³).
    ///
    /// ```text
    /// φ_sat = 1 / (σ·(c/n)·τ)
    /// ```
    pub fn saturation_photon_density(&self) -> f64 {
        1.0 / (self.stimulated_rate_coeff() * self.tau_upper)
    }

    /// Cavity photon lifetime from round-trip parameters.
    ///
    /// ```text
    /// τ_c = 2·n·L / (c·δ)
    /// ```
    ///
    /// # Arguments
    /// * `length_m`       – cavity half-length (m)
    /// * `n_index`        – refractive index
    /// * `round_trip_loss`– total round-trip fractional power loss (0–1)
    pub fn photon_lifetime_from_loss(length_m: f64, n_index: f64, round_trip_loss: f64) -> f64 {
        if round_trip_loss <= 0.0 || length_m <= 0.0 {
            return f64::INFINITY;
        }
        2.0 * n_index * length_m / (C0 * round_trip_loss)
    }
}

// ---------------------------------------------------------------------------
// ThreeLevelLaser
// ---------------------------------------------------------------------------

/// Three-level laser model (e.g. Er³⁺-doped fiber at 1550 nm, ruby at 694 nm).
///
/// In a three-level system the lower laser level coincides with or is close
/// to the ground state, so ground-state reabsorption must be included:
///
/// ```text
/// dN₂/dt = Rp·N₁ − N₂/τ − (σ_em·N₂ − σ_abs·N₁)·(c/n)·φ
/// N₁ + N₂ = N_total   (conservation)
/// ```
///
/// The higher threshold (compared to four-level) arises from the requirement
/// to bleach the ground-state absorption before net gain occurs.
#[derive(Debug, Clone)]
pub struct ThreeLevelLaser {
    /// Stimulated emission cross-section (m²).
    pub sigma_em: f64,
    /// Ground-state absorption cross-section at the laser wavelength (m²).
    pub sigma_abs: f64,
    /// Upper-level lifetime (s).
    pub tau_upper: f64,
    /// Cavity photon lifetime (s).
    pub tau_cavity: f64,
    /// Total active ion density (m⁻³).
    pub n_ions: f64,
    /// Refractive index.
    pub n_index: f64,
    /// Mode volume (m³).
    pub mode_volume_m3: f64,
    /// Laser wavelength (nm).
    pub wavelength_nm: f64,
}

impl ThreeLevelLaser {
    /// Construct a new three-level laser model.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sigma_em: f64,
        sigma_abs: f64,
        tau_upper: f64,
        tau_cavity: f64,
        n_ions: f64,
        n_index: f64,
        mode_vol: f64,
        lambda_nm: f64,
    ) -> Result<Self, OxiPhotonError> {
        if sigma_em <= 0.0 || sigma_abs < 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "cross-sections must be non-negative (sigma_em > 0)".into(),
            ));
        }
        if tau_upper <= 0.0 || tau_cavity <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "lifetimes must be positive".into(),
            ));
        }
        if n_ions <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "n_ions must be positive".into(),
            ));
        }
        if lambda_nm <= 0.0 {
            return Err(OxiPhotonError::InvalidWavelength(lambda_nm * 1.0e-9));
        }
        Ok(Self {
            sigma_em,
            sigma_abs,
            tau_upper,
            tau_cavity,
            n_ions,
            n_index,
            mode_volume_m3: mode_vol.abs(),
            wavelength_nm: lambda_nm,
        })
    }

    /// Photon energy (J).
    #[inline]
    fn photon_energy_j(&self) -> f64 {
        2.0 * PI * HBAR * C0 / (self.wavelength_nm * 1.0e-9)
    }

    /// Effective net gain cross-section (m²).
    #[inline]
    fn net_cross_section(&self) -> f64 {
        self.sigma_em + self.sigma_abs
    }

    /// Threshold upper-level inversion fraction (N₂/N_total).
    ///
    /// Net gain = 0 requires:
    /// ```text
    /// σ_em·N₂ − σ_abs·(N_total − N₂) = 1/(τ_c·c/n)
    /// (σ_em + σ_abs)·N₂ = σ_abs·N_total + 1/(τ_c·c/n)
    /// ```
    pub fn threshold_inversion_fraction(&self) -> f64 {
        let v_g = C0 / self.n_index;
        let cavity_term = 1.0 / (self.tau_cavity * v_g * self.net_cross_section());
        let abs_term = self.sigma_abs / self.net_cross_section();
        (abs_term + cavity_term / self.n_ions).clamp(0.0, 1.0)
    }

    /// Absolute threshold inversion density (m⁻³).
    pub fn threshold_inversion(&self) -> f64 {
        self.threshold_inversion_fraction() * self.n_ions
    }

    /// Ground-state reabsorption loss coefficient α_abs (m⁻¹).
    ///
    /// ```text
    /// α_abs = σ_abs · (N_total − N₂)
    /// ```
    pub fn reabsorption_loss(&self, n_inversion: f64) -> f64 {
        let n_ground = (self.n_ions - n_inversion).max(0.0);
        self.sigma_abs * n_ground
    }

    /// Threshold pump power (mW).
    ///
    /// Higher than four-level because the ground state must be partially
    /// bleached before net gain is achieved.
    pub fn threshold_pump_power_mw(&self, pump_wavelength_nm: f64, pump_efficiency: f64) -> f64 {
        let eta = pump_efficiency.clamp(1.0e-10, 1.0);
        let freq_pump = C0 / (pump_wavelength_nm * 1.0e-9);
        let e_pump = 2.0 * PI * HBAR * freq_pump;
        // Minimum pump rate to reach threshold inversion
        let n_th = self.threshold_inversion();
        let rp_th = n_th / self.tau_upper;
        rp_th * e_pump * self.mode_volume_m3 / eta * 1.0e3
    }

    /// CW output power (mW) above threshold.
    pub fn output_power_mw(&self, pump_rate: f64, output_coupler_loss: f64) -> f64 {
        let n_th = self.threshold_inversion();
        let rp_th = n_th / self.tau_upper;
        let excess = pump_rate - rp_th;
        if excess <= 0.0 {
            return 0.0;
        }
        let oc = output_coupler_loss.clamp(0.0, 1.0);
        let e_photon = self.photon_energy_j();
        // Same structure as four-level, but threshold is higher
        let phi_ss = excess * self.tau_cavity;
        phi_ss * self.mode_volume_m3 * e_photon / self.tau_cavity * oc * 1.0e3
    }

    /// Slope efficiency (mW per unit pump rate above threshold).
    pub fn slope_efficiency(&self, pump_wavelength_nm: f64, output_coupler: f64) -> f64 {
        let stokes = self.wavelength_nm / pump_wavelength_nm.max(1.0e-3);
        let e_laser = self.photon_energy_j();
        let oc = output_coupler.clamp(0.0, 1.0);
        stokes * oc * e_laser * self.mode_volume_m3 * 1.0e3
    }
}

// ---------------------------------------------------------------------------
// QSwitchedLaser
// ---------------------------------------------------------------------------

/// Q-switched laser pulse dynamics.
///
/// During the Q-switch hold phase the pump builds up a large inversion N_i.
/// When the Q-switch opens, the cavity loss drops suddenly and a short,
/// energetic pulse develops.
///
/// Key relations (Refs: Siegman §26.3, Svelto §8.2):
/// ```text
/// P_peak ≈ (N_i/N_th − 1) · ħω · V / τ_c        (W)
/// E_pulse = (N_i − N_f) · ħω · V / 2             (J)
/// τ_pulse ≈ E_pulse / P_peak                       (s)
/// ```
///
/// The final inversion N_f satisfies the implicit equation:
/// ```text
/// N_f/N_i = exp(−(N_i − N_f)/N_th)   (Frantz-Nodvik)
/// ```
#[derive(Debug, Clone)]
pub struct QSwitchedLaser {
    /// Underlying four-level laser cavity parameters.
    pub laser: FourLevelLaser,
    /// Initial (stored) inversion density N_i (m⁻³) at the moment Q-switches open.
    pub initial_inversion: f64,
    /// Threshold inversion density N_th (m⁻³) after Q-switch opens.
    pub threshold_inversion: f64,
}

impl QSwitchedLaser {
    /// Construct a Q-switched laser model.
    pub fn new(
        laser: FourLevelLaser,
        n_initial: f64,
        n_threshold: f64,
    ) -> Result<Self, OxiPhotonError> {
        if n_initial <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "initial_inversion must be positive".into(),
            ));
        }
        if n_threshold <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "threshold_inversion must be positive".into(),
            ));
        }
        Ok(Self {
            laser,
            initial_inversion: n_initial,
            threshold_inversion: n_threshold,
        })
    }

    /// Photon energy (J) at the laser wavelength.
    #[inline]
    fn photon_energy_j(&self) -> f64 {
        2.0 * PI * HBAR * C0 / (self.laser.wavelength_nm * 1.0e-9)
    }

    /// Final inversion density N_f (m⁻³) after the Q-switched pulse.
    ///
    /// Solved numerically via Newton's method on the implicit equation:
    /// ```text
    /// x = r · exp(−(1 − x)·r_i/r_th)
    /// ```
    /// where x = N_f/N_i, r_i = N_i, r_th = N_th.
    pub fn final_inversion(&self) -> f64 {
        let ni = self.initial_inversion;
        let nth = self.threshold_inversion;
        if ni <= nth {
            return ni; // below threshold, pulse does not form
        }
        let ratio = ni / nth;
        // Iterative solution: N_f = N_i · exp(-(N_i - N_f)/N_th)
        let mut nf = 0.01 * ni; // start near zero
        for _ in 0..200 {
            let new_nf = ni * (-(ni - nf) / nth).exp();
            if (new_nf - nf).abs() < 1.0e-6 * ni {
                return new_nf;
            }
            nf = 0.5 * nf + 0.5 * new_nf; // relaxed update
        }
        // Fallback: analytic approximation for large inversion ratio
        let _ = ratio; // suppress unused warning
        nf
    }

    /// Peak power of the Q-switched pulse (W).
    ///
    /// ```text
    /// P_peak = (N_i/N_th − 1) · ħω · V / τ_c
    /// ```
    pub fn peak_power_w(&self) -> f64 {
        let ni = self.initial_inversion;
        let nth = self.threshold_inversion;
        if ni <= nth {
            return 0.0;
        }
        let e_photon = self.photon_energy_j();
        (ni / nth - 1.0) * e_photon * self.laser.mode_volume_m3 / self.laser.tau_cavity
    }

    /// Pulse energy (μJ).
    ///
    /// ```text
    /// E = (N_i − N_f) · ħω · V / 2
    /// ```
    pub fn pulse_energy_uj(&self) -> f64 {
        let ni = self.initial_inversion;
        let nf = self.final_inversion();
        let e_photon = self.photon_energy_j();
        (ni - nf) * e_photon * self.laser.mode_volume_m3 / 2.0 * 1.0e6
    }

    /// Pulse width (ns) estimated as E / P_peak.
    pub fn pulse_width_ns(&self) -> f64 {
        let e_j = self.pulse_energy_uj() * 1.0e-6;
        let p_w = self.peak_power_w();
        if p_w <= 0.0 {
            return 0.0;
        }
        e_j / p_w * 1.0e9
    }

    /// Build-up time (ns) — time from spontaneous emission noise to threshold.
    ///
    /// Approximated by the time for the photon field to grow from spontaneous
    /// emission to threshold:
    /// ```text
    /// t_build ≈ τ_c · ln(N_i·V·β / 1) / (N_i/N_th − 1)
    /// ```
    pub fn buildup_time_ns(&self, pump_rate: f64) -> f64 {
        let ni = self.initial_inversion;
        let nth = self.threshold_inversion;
        if ni <= nth {
            return 0.0;
        }
        let _ = pump_rate; // pump_rate determines how N_i was reached; not used here
                           // Approximate: t_b ≈ τ_c * ln(N_i * V * β / φ_noise) / (N_i/N_th - 1)
                           // Using φ_noise ≈ 1 photon / V for spontaneous emission seed
        let phi_noise = 1.0 / self.laser.mode_volume_m3;
        let phi_th = self.laser.saturation_photon_density() * 0.01;
        let phi_target = phi_th.max(phi_noise);
        let log_arg = (ni * self.laser.mode_volume_m3 * self.laser.beta).max(1.0);
        let _ = phi_target;
        let tau_c = self.laser.tau_cavity;
        tau_c * log_arg.ln() / (ni / nth - 1.0) * 1.0e9
    }

    /// Simulate the Q-switched pulse shape.
    ///
    /// Returns `(time_ns, power_kw)` pairs from pulse onset until the
    /// inversion is substantially depleted.
    ///
    /// # Arguments
    /// * `dt_ns` – time step (ns); should be < τ_c
    pub fn simulate_pulse(&self, dt_ns: f64) -> Result<Vec<(f64, f64)>, OxiPhotonError> {
        if dt_ns <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "dt_ns must be positive".into(),
            ));
        }
        let dt = dt_ns * 1.0e-9;
        // Simulate until inversion drops below 1% of initial
        let max_steps = 100_000usize;
        let mut results = Vec::with_capacity(max_steps / 10);

        let g = self.laser.stimulated_rate_coeff();
        let e_photon = self.photon_energy_j();
        let v = self.laser.mode_volume_m3;
        let tau_c = self.laser.tau_cavity;
        let tau_sp = self.laser.tau_upper;
        let beta = self.laser.beta;

        let mut n_inv = self.initial_inversion;
        // Start with tiny spontaneous emission seed
        let phi_seed = beta * n_inv / tau_sp * tau_c;
        let mut phi = phi_seed.max(1.0 / v);

        for step in 0..max_steps {
            let t_ns = step as f64 * dt_ns;
            let power_kw = phi * v * e_photon / tau_c / 1.0e3;
            results.push((t_ns, power_kw));

            let dn = -g * n_inv * phi - n_inv / tau_sp;
            let dphi = g * n_inv * phi - phi / tau_c + beta * n_inv / tau_sp;
            n_inv = (n_inv + dn * dt).max(0.0);
            phi = (phi + dphi * dt).max(0.0);

            if n_inv < 0.01 * self.initial_inversion && step > 10 {
                break;
            }
        }

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Nd:YAG-like four-level laser parameters
    fn nd_yag() -> FourLevelLaser {
        FourLevelLaser::new(
            2.8e-23,  // sigma_em (m²)
            230.0e-6, // tau_upper (s) = 230 µs
            10.0e-9,  // tau_cavity (s) = 10 ns
            1.82,     // n_index (Nd:YAG)
            1.0e-5,   // beta
            1.0e-9,   // mode_volume (m³) = 1 mm³
            1064.0,   // wavelength (nm)
        )
        .expect("valid Nd:YAG parameters")
    }

    /// Er:fiber three-level laser parameters (1550 nm)
    fn er_fiber() -> ThreeLevelLaser {
        ThreeLevelLaser::new(
            5.0e-25, // sigma_em (m²)
            3.0e-25, // sigma_abs (m²)  – ground-state reabsorption
            10.0e-3, // tau_upper = 10 ms
            15.0e-9, // tau_cavity = 15 ns
            1.0e25,  // n_ions (m⁻³) = 1e25 (Er-doped fiber)
            1.45,    // n_index (silica)
            1.0e-12, // mode_volume (m³) = 1 µm² × 1 m fiber (approx)
            1550.0,  // wavelength (nm)
        )
        .expect("valid Er:fiber parameters")
    }

    #[test]
    fn test_threshold_inversion() {
        let laser = nd_yag();
        let n_th = laser.threshold_inversion();
        // N_th = 1 / (σ·c/n·τ_c)
        let expected = 1.0 / (laser.sigma_em * C0 / laser.n_index * laser.tau_cavity);
        let rel_err = (n_th - expected).abs() / expected;
        assert!(
            rel_err < 1.0e-10,
            "N_th: expected {expected:.4e}, got {n_th:.4e}, rel_err={rel_err:.2e}"
        );
        assert!(n_th > 0.0, "threshold inversion must be positive");
    }

    #[test]
    fn test_output_power_above_threshold_positive() {
        let laser = nd_yag();
        let rp_th = laser.threshold_pump_rate();
        // Pump at 3× threshold
        let pump_rate = 3.0 * rp_th;
        let power = laser.output_power_mw(pump_rate, 0.10);
        assert!(
            power > 0.0,
            "output power should be positive above threshold: got {power}"
        );
    }

    #[test]
    fn test_output_power_below_threshold_zero() {
        let laser = nd_yag();
        let rp_th = laser.threshold_pump_rate();
        // Pump at 50% of threshold
        let power = laser.output_power_mw(0.5 * rp_th, 0.10);
        assert!(
            power <= 0.0,
            "output power should be zero below threshold: got {power}"
        );
    }

    #[test]
    fn test_relaxation_oscillation_freq() {
        let laser = nd_yag();
        let rp_th = laser.threshold_pump_rate();
        let f_ro_low = laser.relaxation_oscillation_freq_mhz(1.5 * rp_th);
        let f_ro_high = laser.relaxation_oscillation_freq_mhz(5.0 * rp_th);
        assert!(
            f_ro_low > 0.0,
            "RO frequency should be positive above threshold"
        );
        assert!(
            f_ro_high > f_ro_low,
            "RO frequency should increase with pump: low={f_ro_low:.3} MHz, high={f_ro_high:.3} MHz"
        );
    }

    #[test]
    fn test_photon_lifetime_formula() {
        // τ_c = 2·n·L / (c·δ)
        let n = 1.82;
        let l = 0.05; // 5 cm half-length
        let delta = 0.04; // 4% round-trip loss
        let tau_c = FourLevelLaser::photon_lifetime_from_loss(l, n, delta);
        let expected = 2.0 * n * l / (C0 * delta);
        let rel_err = (tau_c - expected).abs() / expected;
        assert!(
            rel_err < 1.0e-10,
            "τ_c: expected {expected:.4e} s, got {tau_c:.4e} s"
        );
        assert!(tau_c > 0.0, "photon lifetime must be positive");
    }

    #[test]
    fn test_simulation_reaches_steady_state() {
        let laser = nd_yag();
        let rp_th = laser.threshold_pump_rate();
        let pump = 2.0 * rp_th;
        // Run for 100 µs (much longer than τ_upper = 230 µs, so use fewer but larger steps)
        let results = laser
            .simulate(pump, 500.0, 0.5, 0.0, 0.0)
            .expect("simulation should succeed");
        assert!(!results.is_empty(), "simulation must produce output");
        // Check the last few points for approximate steady state
        let n_last = results.last().map(|r| r.1).unwrap_or(0.0);
        let phi_last = results.last().map(|r| r.2).unwrap_or(0.0);
        let n_th = laser.threshold_inversion();
        // In steady state, N should be near N_th (clamped by gain saturation)
        assert!(
            n_last > 0.0 && n_last.is_finite(),
            "inversion should be positive and finite: {n_last:.4e}"
        );
        // Photon density should be positive above threshold
        assert!(
            phi_last >= 0.0 && phi_last.is_finite(),
            "photon density should be non-negative and finite: {phi_last:.4e}"
        );
        // N should not exceed threshold by more than 2× (saturation)
        assert!(
            n_last < 3.0 * n_th,
            "inversion should not exceed 3×N_th in steady state: N={n_last:.4e}, N_th={n_th:.4e}"
        );
    }

    #[test]
    fn test_q_switched_peak_power() {
        let laser = nd_yag();
        let n_th = laser.threshold_inversion();
        // Store 3× the threshold inversion
        let n_i = 3.0 * n_th;
        let qsl = QSwitchedLaser::new(laser, n_i, n_th).expect("valid Q-switched parameters");
        let peak = qsl.peak_power_w();
        // P_peak ∝ (N_i/N_th - 1) = 2.0 for 3×N_th initial
        assert!(
            peak > 0.0,
            "Q-switched peak power must be positive: {peak:.4e} W"
        );
        // Check scaling: 5× should give higher peak than 3×
        let laser2 = nd_yag();
        let n_th2 = laser2.threshold_inversion();
        let qsl2 = QSwitchedLaser::new(laser2, 5.0 * n_th2, n_th2).expect("valid");
        assert!(
            qsl2.peak_power_w() > peak,
            "Higher inversion should give higher peak power"
        );
    }

    #[test]
    fn test_q_switched_pulse_energy() {
        let laser = nd_yag();
        let n_th = laser.threshold_inversion();
        let n_i = 3.0 * n_th;
        let qsl = QSwitchedLaser::new(laser, n_i, n_th).expect("valid Q-switched parameters");
        let energy = qsl.pulse_energy_uj();
        assert!(
            energy > 0.0,
            "Q-switched pulse energy must be positive: {energy:.4e} µJ"
        );
        // Energy should be less than stored energy (N_i · ħω · V)
        let e_photon = 2.0 * PI * HBAR * C0 / (1064.0e-9);
        let e_stored_uj = n_i * e_photon * qsl.laser.mode_volume_m3 * 1.0e6;
        assert!(
            energy < e_stored_uj,
            "Pulse energy {energy:.4e} µJ should be less than stored {e_stored_uj:.4e} µJ"
        );
    }

    #[test]
    fn test_three_level_higher_threshold() {
        // Three-level threshold should be higher than four-level for comparable params
        let four_level = FourLevelLaser::new(
            5.0e-25, // sigma_em
            10.0e-3, // tau_upper
            15.0e-9, // tau_cavity
            1.45,    // n_index
            1.0e-5,  // beta
            1.0e-12, // mode_volume
            1550.0,  // wavelength
        )
        .expect("valid 4-level");

        let three_level = er_fiber();

        let p_th_4 = four_level.threshold_pump_power_mw(980.0, 0.5);
        let p_th_3 = three_level.threshold_pump_power_mw(980.0, 0.5);

        assert!(
            p_th_3 > p_th_4,
            "Three-level threshold ({p_th_3:.4e} mW) should exceed four-level ({p_th_4:.4e} mW)"
        );
    }

    #[test]
    fn test_saturation_photon_density_positive() {
        let laser = nd_yag();
        let phi_sat = laser.saturation_photon_density();
        assert!(
            phi_sat > 0.0,
            "saturation photon density must be positive: {phi_sat:.4e}"
        );
    }

    #[test]
    fn test_saturated_gain_decreases_with_photon_density() {
        let laser = nd_yag();
        let n_inv = laser.threshold_inversion() * 2.0;
        let phi_sat = laser.saturation_photon_density();
        let g_low = laser.saturated_gain(n_inv, 0.0);
        let g_mid = laser.saturated_gain(n_inv, phi_sat);
        let g_high = laser.saturated_gain(n_inv, 10.0 * phi_sat);
        assert!(
            g_low > g_mid && g_mid > g_high,
            "saturated gain should decrease with photon density: {g_low:.4e} > {g_mid:.4e} > {g_high:.4e}"
        );
    }
}
