/// Semiconductor laser (diode) rate equation models.
///
/// Implements:
/// - Edge-emitting diode laser (Fabry-Perot)
/// - VCSEL (vertical-cavity surface-emitting laser)
/// - DFB (distributed feedback) laser
///
/// Rate equations for carrier density N and photon density S:
/// ```text
/// dN/dt = J/(e·d) − N/τ_n − v_g·g(N)·S
/// dS/dt = Γ·v_g·g(N)·S − S/τ_p + Γ·β·N/τ_n
/// dφ/dt = (1/2)·α_H·(Γ·v_g·g(N) − 1/τ_p)   [optical phase, for chirp]
/// ```
///
/// References:
/// - Coldren, Corzine & Mašanović, "Diode Lasers and Photonic Integrated Circuits", 2nd ed. (2012)
/// - Agrawal & Dutta, "Semiconductor Lasers", 2nd ed. (1993)
/// - Chuang, "Physics of Optoelectronic Devices", 2nd ed. (2009)
use std::f64::consts::PI;

use crate::error::OxiPhotonError;

// ---------------------------------------------------------------------------
// Physical constants
// ---------------------------------------------------------------------------

/// Reduced Planck constant (J·s)
const HBAR: f64 = 1.054_571_8e-34;
/// Speed of light in vacuum (m/s)
const C0: f64 = 2.997_924_58e8;
/// Elementary charge (C)
const E_CHARGE: f64 = 1.602_176_634e-19;

// ---------------------------------------------------------------------------
// SemiconductorLaser
// ---------------------------------------------------------------------------

/// Edge-emitting semiconductor laser (Fabry-Perot diode).
///
/// Uses a linear material gain model: g(N) = g₀·(N − N₀).
///
/// The modal gain is Γ·g(N), and lasing threshold requires
/// Γ·g(N_th) = α_i + α_m.
#[derive(Debug, Clone)]
pub struct SemiconductorLaser {
    /// Laser wavelength (nm).
    pub lambda_nm: f64,
    /// Active layer thickness d (nm).
    pub active_layer_thickness_nm: f64,
    /// Active region area W·L (µm²).
    pub active_area_um2: f64,
    /// Cavity length L (µm).
    pub cavity_length_um: f64,
    /// Effective refractive index of the mode.
    pub n_index: f64,
    /// Optical confinement factor Γ (0–1).
    pub confinement_factor: f64,
    /// Transparency carrier density N₀ (m⁻³).
    pub transparency_density: f64,
    /// Differential gain coefficient g₀ (m⁻¹ / m⁻³) = m².
    ///
    /// So g(N) = g0 * (N - N0) has units of m⁻¹ when N is m⁻³ and g0 is m³·m⁻¹ = m².
    /// Typical value: 2.5e-20 m² (InGaAsP at 1550 nm).
    pub gain_coefficient: f64,
    /// Carrier (electron-hole) lifetime τ_n (ns).
    pub carrier_lifetime_ns: f64,
    /// Photon lifetime τ_p (ps) — used as an override if set non-zero.
    /// If zero, computed from cavity parameters via `photon_lifetime_s()`.
    pub photon_lifetime_ps: f64,
    /// Spontaneous emission coupling factor β.
    pub beta: f64,
    /// Henry linewidth enhancement factor α_H.
    pub alpha_h: f64,
    /// Internal (material + scattering) loss α_i (m⁻¹).
    pub internal_loss: f64,
    /// Facet reflectivity R = sqrt(R₁·R₂).
    pub mirror_reflectivity: f64,
}

impl SemiconductorLaser {
    /// Construct a semiconductor laser with explicit parameters.
    ///
    /// # Arguments
    /// * `lambda_nm`                  – wavelength (nm)
    /// * `thickness_nm`               – active layer thickness d (nm)
    /// * `area_um2`                   – active area W×L (µm²)
    /// * `length_um`                  – cavity length L (µm)
    /// * `n_index`                    – effective refractive index
    /// * `confinement`                – optical confinement factor Γ
    /// * `n_transp`                   – transparency carrier density (m⁻³)
    /// * `g0`                         – differential gain (m²)
    /// * `tau_n_ns`                   – carrier lifetime (ns)
    /// * `tau_p_ps`                   – photon lifetime override (ps); 0 = auto
    /// * `beta`                       – spontaneous emission coupling factor
    /// * `alpha_h`                    – Henry factor
    /// * `alpha_i`                    – internal loss (m⁻¹)
    /// * `r_mirror`                   – mirror reflectivity R = sqrt(R1·R2)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lambda_nm: f64,
        thickness_nm: f64,
        area_um2: f64,
        length_um: f64,
        n_index: f64,
        confinement: f64,
        n_transp: f64,
        g0: f64,
        tau_n_ns: f64,
        tau_p_ps: f64,
        beta: f64,
        alpha_h: f64,
        alpha_i: f64,
        r_mirror: f64,
    ) -> Result<Self, OxiPhotonError> {
        if lambda_nm <= 0.0 {
            return Err(OxiPhotonError::InvalidWavelength(lambda_nm * 1.0e-9));
        }
        if length_um <= 0.0 || thickness_nm <= 0.0 || area_um2 <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "cavity dimensions must be positive".into(),
            ));
        }
        if confinement <= 0.0 || confinement > 1.0 {
            return Err(OxiPhotonError::NumericalError(
                "confinement factor must be in (0, 1]".into(),
            ));
        }
        if r_mirror <= 0.0 || r_mirror >= 1.0 {
            return Err(OxiPhotonError::NumericalError(
                "mirror reflectivity must be in (0, 1)".into(),
            ));
        }
        Ok(Self {
            lambda_nm,
            active_layer_thickness_nm: thickness_nm,
            active_area_um2: area_um2,
            cavity_length_um: length_um,
            n_index,
            confinement_factor: confinement,
            transparency_density: n_transp,
            gain_coefficient: g0,
            carrier_lifetime_ns: tau_n_ns,
            photon_lifetime_ps: tau_p_ps,
            beta: beta.clamp(0.0, 1.0),
            alpha_h,
            internal_loss: alpha_i.max(0.0),
            mirror_reflectivity: r_mirror,
        })
    }

    /// Typical InGaAsP/InP edge-emitting laser at 1550 nm.
    ///
    /// Parameters representative of a 300 µm buried heterostructure diode.
    pub fn ingaasp_1550() -> Self {
        // These are typical mid-range values for an InGaAsP/InP FP laser
        Self {
            lambda_nm: 1550.0,
            active_layer_thickness_nm: 100.0, // d = 100 nm (double-heterostructure)
            active_area_um2: 300.0 * 2.0,     // L × W = 300 µm × 2 µm
            cavity_length_um: 300.0,
            n_index: 3.5,
            confinement_factor: 0.3,
            transparency_density: 1.5e24, // N₀ = 1.5×10²⁴ m⁻³
            gain_coefficient: 2.5e-20,    // g₀ = 2.5×10⁻²⁰ m² (differential gain)
            carrier_lifetime_ns: 2.0,     // τ_n = 2 ns
            photon_lifetime_ps: 0.0,      // auto-computed
            beta: 1.0e-4,
            alpha_h: 4.0,                         // typical for InGaAsP
            internal_loss: 2.0e3,                 // α_i = 20 cm⁻¹ = 2000 m⁻¹
            mirror_reflectivity: 0.32_f64.sqrt(), // R ≈ 0.566 (both facets cleaved at 0.32)
        }
    }

    // -----------------------------------------------------------------------
    // Basic derived quantities
    // -----------------------------------------------------------------------

    /// Group velocity v_g = c/n (m/s).
    pub fn group_velocity(&self) -> f64 {
        C0 / self.n_index
    }

    /// Mirror loss α_m = ln(1/R²) / (2·L)  (m⁻¹).
    ///
    /// For a Fabry-Perot cavity with identical facet reflectivities:
    /// α_m = (1/L)·ln(1/R) where R = sqrt(R1·R2).
    pub fn mirror_loss(&self) -> f64 {
        let l_m = self.cavity_length_um * 1.0e-6;
        -self.mirror_reflectivity.ln() / l_m
    }

    /// Total round-trip loss: α_tot = α_i + α_m (m⁻¹).
    pub fn total_loss(&self) -> f64 {
        self.internal_loss + self.mirror_loss()
    }

    /// Photon lifetime τ_p (s).
    ///
    /// If `photon_lifetime_ps > 0`, that value is used directly.
    /// Otherwise computed from cavity parameters:
    /// ```text
    /// τ_p = 1 / (v_g · α_tot)
    /// ```
    pub fn photon_lifetime_s(&self) -> f64 {
        if self.photon_lifetime_ps > 0.0 {
            return self.photon_lifetime_ps * 1.0e-12;
        }
        1.0 / (self.group_velocity() * self.total_loss())
    }

    // -----------------------------------------------------------------------
    // Threshold analysis
    // -----------------------------------------------------------------------

    /// Threshold carrier density N_th (m⁻³).
    ///
    /// From Γ·g₀·(N_th − N₀) = α_tot:
    /// ```text
    /// N_th = N₀ + α_tot / (Γ·g₀)
    /// ```
    pub fn threshold_carrier_density(&self) -> f64 {
        let alpha = self.total_loss();
        self.transparency_density + alpha / (self.confinement_factor * self.gain_coefficient)
    }

    /// Active volume V = d × W × L (m³).
    fn active_volume_m3(&self) -> f64 {
        let d = self.active_layer_thickness_nm * 1.0e-9;
        let wl = self.active_area_um2 * 1.0e-12; // µm² → m²
        d * wl
    }

    /// Threshold current density J_th (A/m²).
    ///
    /// ```text
    /// J_th = e · d · N_th / τ_n
    /// ```
    pub fn threshold_current_density(&self) -> f64 {
        let d = self.active_layer_thickness_nm * 1.0e-9;
        let tau_n = self.carrier_lifetime_ns * 1.0e-9;
        E_CHARGE * d * self.threshold_carrier_density() / tau_n
    }

    /// Threshold current I_th (mA).
    ///
    /// ```text
    /// I_th = J_th · W · L
    /// ```
    pub fn threshold_current_ma(&self) -> f64 {
        let area_m2 = self.active_area_um2 * 1.0e-12;
        self.threshold_current_density() * area_m2 * 1.0e3
    }

    // -----------------------------------------------------------------------
    // Above-threshold behaviour
    // -----------------------------------------------------------------------

    /// External differential quantum efficiency η_d.
    ///
    /// ```text
    /// η_d = η_inj · α_m / α_tot
    /// ```
    pub fn differential_efficiency(&self, injection_efficiency: f64) -> f64 {
        injection_efficiency.clamp(0.0, 1.0) * self.mirror_loss() / self.total_loss()
    }

    /// Slope efficiency dP/dI (mW/mA).
    ///
    /// ```text
    /// η_slope = (ħω/e) · η_d   [W/A]  →  mW/mA (numerically equal)
    /// ```
    pub fn slope_efficiency_mw_per_ma(&self, injection_efficiency: f64) -> f64 {
        let e_photon = 2.0 * PI * HBAR * C0 / (self.lambda_nm * 1.0e-9);
        let eta_d = self.differential_efficiency(injection_efficiency);
        // (J/photon / C) = V = W/A = mW/mA
        (e_photon / E_CHARGE) * eta_d
    }

    /// Output power P_out (mW) versus injected current.
    ///
    /// Linear model above threshold:
    /// ```text
    /// P_out = η_slope · (I − I_th)    for I > I_th
    /// P_out ≈ 0                        for I ≤ I_th
    /// ```
    pub fn output_power_mw(&self, current_ma: f64, injection_efficiency: f64) -> f64 {
        let i_th = self.threshold_current_ma();
        if current_ma <= i_th {
            // Below threshold: spontaneous emission only (very small)
            return 0.0_f64.max(0.0);
        }
        let slope = self.slope_efficiency_mw_per_ma(injection_efficiency);
        slope * (current_ma - i_th)
    }

    // -----------------------------------------------------------------------
    // Gain model
    // -----------------------------------------------------------------------

    /// Material gain g(N) = g₀·(N − N₀) (m⁻¹).
    ///
    /// Linear approximation valid near transparency.
    pub fn material_gain(&self, carrier_density: f64) -> f64 {
        let delta_n = carrier_density - self.transparency_density;
        (self.gain_coefficient * delta_n).max(0.0)
    }

    /// Modal gain Γ·g(N) (m⁻¹).
    pub fn modal_gain(&self, carrier_density: f64) -> f64 {
        self.confinement_factor * self.material_gain(carrier_density)
    }

    // -----------------------------------------------------------------------
    // Small-signal dynamics
    // -----------------------------------------------------------------------

    /// Steady-state photon density S_ss (m⁻³) at a given injection current.
    ///
    /// Above threshold, the excess carrier injection converts to photons:
    /// S_ss = Γ · η_inj · (I − I_th) · τ_p / (e · V)
    ///
    /// Returns 0 below threshold.
    pub fn photon_density_ss(&self, current_ma: f64, injection_efficiency: f64) -> f64 {
        let i_th = self.threshold_current_ma();
        if current_ma <= i_th {
            return 0.0;
        }
        let eta_inj = injection_efficiency.clamp(0.0, 1.0);
        let i_a = current_ma * 1.0e-3;
        let i_th_a = i_th * 1.0e-3;
        let v_act = self.active_volume_m3();
        let tau_p = self.photon_lifetime_s();
        // In steady state above threshold, the excess carrier injection rate
        // converts directly to photons:
        // η_inj·(I - I_th)/(e·V) = S_ss / (Γ·τ_p)   [photons/s per volume]
        // Therefore: S_ss = Γ·η_inj·(I - I_th)·τ_p / (e·V)
        self.confinement_factor * eta_inj * (i_a - i_th_a) * tau_p / (E_CHARGE * v_act)
    }

    /// Relaxation oscillation frequency f_RO (GHz).
    ///
    /// ```text
    /// f_RO = (1/2π) · sqrt( v_g·g_th·S_ss / τ_p )
    ///      = (1/2π) · sqrt( (I/I_th - 1) / (τ_n · τ_p) )
    /// ```
    pub fn relaxation_oscillation_ghz(&self, current_ma: f64, injection_efficiency: f64) -> f64 {
        let i_th = self.threshold_current_ma();
        if current_ma <= i_th {
            return 0.0;
        }
        let tau_n = self.carrier_lifetime_ns * 1.0e-9;
        let tau_p = self.photon_lifetime_s();
        // f_RO² = (1/2π)² · (I/I_th - 1) / (τ_n · τ_p)
        let _ = injection_efficiency; // efficiency does not change f_RO formula
        let ratio = current_ma / i_th - 1.0;
        let omega_sq = ratio / (tau_n * tau_p);
        if omega_sq <= 0.0 {
            return 0.0;
        }
        omega_sq.sqrt() / (2.0 * PI) / 1.0e9
    }

    /// Linewidth Δν (MHz) using extended Schawlow-Townes formula with Henry factor.
    ///
    /// ```text
    /// Δν = (1 + α_H²) · ħω · v_g · α_m · n_sp / (4π · P_out · τ_p)
    /// ```
    ///
    /// Simplified (Agrawal model) inversely proportional to output power:
    /// ```text
    /// Δν ∝ (1 + α_H²) / P_out
    /// ```
    pub fn linewidth_mhz(&self, power_mw: f64) -> f64 {
        if power_mw <= 0.0 {
            return f64::INFINITY;
        }
        let e_photon = 2.0 * PI * HBAR * C0 / (self.lambda_nm * 1.0e-9);
        let v_g = self.group_velocity();
        let alpha_m = self.mirror_loss();
        let tau_p = self.photon_lifetime_s();
        // Spontaneous emission factor n_sp ≈ 2 (population inversion parameter)
        let n_sp = 2.0_f64;
        let p_out = power_mw * 1.0e-3; // mW → W
                                       // Δν = (1 + α_H²) · ħω · v_g · α_m · n_sp / (4π · P · τ_p)
        let delta_nu = (1.0 + self.alpha_h * self.alpha_h) * e_photon * v_g * alpha_m * n_sp
            / (4.0 * PI * p_out * tau_p);
        delta_nu / 1.0e6 // Hz → MHz
    }

    /// Simulate the turn-on transient using Euler integration.
    ///
    /// Integrates the coupled carrier–photon rate equations from t = 0.
    ///
    /// # Arguments
    /// * `current_ma`         – step-function injected current (mA)
    /// * `t_max_ns`           – simulation duration (ns)
    /// * `dt_ps`              – time step (ps); should be << τ_p
    /// * `injection_efficiency` – carrier injection efficiency η_inj
    ///
    /// Returns `(time_ps, N [m⁻³], S [m⁻³])` tuples.
    pub fn simulate_turn_on(
        &self,
        current_ma: f64,
        t_max_ns: f64,
        dt_ps: f64,
        injection_efficiency: f64,
    ) -> Result<Vec<(f64, f64, f64)>, OxiPhotonError> {
        if dt_ps <= 0.0 || t_max_ns <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "time parameters must be positive".into(),
            ));
        }
        let dt = dt_ps * 1.0e-12; // ps → s
        let n_steps = ((t_max_ns * 1.0e3 / dt_ps) as usize).min(5_000_000);
        let mut results = Vec::with_capacity(n_steps + 1);

        let v_g = self.group_velocity();
        let tau_n = self.carrier_lifetime_ns * 1.0e-9;
        let tau_p = self.photon_lifetime_s();
        let d = self.active_layer_thickness_nm * 1.0e-9;
        let v_act = self.active_volume_m3();
        let eta_inj = injection_efficiency.clamp(0.0, 1.0);
        let i_a = current_ma * 1.0e-3;
        // Current injection rate into active volume: η·J/(e·d) = η·I/(e·V)
        let pump_rate = eta_inj * i_a / (E_CHARGE * v_act);
        let _ = d; // d folded into v_act

        let mut n_carr = self.transparency_density; // start at transparency
        let mut s_phot = self.beta * n_carr / tau_n * tau_p; // small seed

        results.push((0.0, n_carr, s_phot));

        for step in 1..=n_steps {
            let g = self.material_gain(n_carr);
            let g_modal = self.confinement_factor * g;

            let dn = pump_rate - n_carr / tau_n - v_g * g * s_phot;
            let ds = g_modal * v_g * s_phot - s_phot / tau_p
                + self.confinement_factor * self.beta * n_carr / tau_n;

            n_carr = (n_carr + dn * dt).max(0.0);
            s_phot = (s_phot + ds * dt).max(0.0);

            let t_ps = step as f64 * dt_ps;
            results.push((t_ps, n_carr, s_phot));
        }

        Ok(results)
    }

    /// Relative intensity noise spectral density RIN (dB/Hz) at frequency f.
    ///
    /// ```text
    /// RIN(f) = 2 · (f_RO⁴ + Γ²·f²) / ((f_RO² − f²)² + Γ²·f²) · S_sp / S²
    /// ```
    ///
    /// Simplified formula: near the relaxation oscillation peak,
    /// RIN peaks and then rolls off.
    ///
    /// # Arguments
    /// * `current_ma` – operating current (mA)
    /// * `freq_ghz`   – noise frequency (GHz)
    pub fn rin_db_per_hz(&self, current_ma: f64, freq_ghz: f64) -> f64 {
        let f_ro = self.relaxation_oscillation_ghz(current_ma, 0.9);
        if f_ro <= 0.0 {
            return -100.0; // well below threshold
        }
        let tau_n = self.carrier_lifetime_ns * 1.0e-9;
        let tau_p = self.photon_lifetime_s();
        let f = freq_ghz;
        let gamma_damp = (1.0 / tau_n + 1.0 / tau_p) / (2.0 * PI * 1.0e9); // GHz
        let f_ro2 = f_ro * f_ro;
        let f2 = f * f;
        let g2 = gamma_damp * gamma_damp;
        // Noise transfer function (dimensionless, relative)
        let numerator = 2.0 * (f_ro2 * f_ro2 + g2 * f2);
        let denominator = (f_ro2 - f2) * (f_ro2 - f2) + g2 * f2;
        if denominator < 1.0e-20 {
            return -100.0;
        }
        let rin_linear = numerator / denominator * 1.0e-15; // normalised
        10.0 * rin_linear.log10()
    }

    /// Small-signal 3-dB modulation bandwidth (GHz).
    ///
    /// ```text
    /// f_3dB ≈ sqrt(3) · f_RO
    /// ```
    pub fn modulation_bandwidth_3db_ghz(&self, current_ma: f64) -> f64 {
        3.0_f64.sqrt() * self.relaxation_oscillation_ghz(current_ma, 0.9)
    }

    /// Instantaneous optical frequency chirp due to current modulation (GHz).
    ///
    /// ```text
    /// Δν(t) = α_H / (4π) · d(ln P)/dt = α_H / (4π) · (1/P) · dP/dt
    /// ```
    ///
    /// # Arguments
    /// * `dp_dt`    – rate of change of power (mW/s)
    /// * `power_mw` – instantaneous output power (mW)
    pub fn frequency_chirp_ghz(&self, dp_dt: f64, power_mw: f64) -> f64 {
        if power_mw <= 0.0 {
            return 0.0;
        }
        let p_w = power_mw * 1.0e-3;
        let dp_dt_w_per_s = dp_dt * 1.0e-3; // mW/s → W/s
        self.alpha_h / (4.0 * PI) * dp_dt_w_per_s / p_w / 1.0e9
    }
}

// ---------------------------------------------------------------------------
// Vcsel
// ---------------------------------------------------------------------------

/// Vertical-cavity surface-emitting laser (VCSEL).
///
/// VCSELs have very short cavities (λ-scale) formed by two distributed Bragg
/// reflector (DBR) mirror stacks. Key differences from edge emitters:
/// - Circular beam (circular symmetry → Gaussian far field)
/// - Very low threshold current (< 1 mA typical)
/// - High mirror reflectivity (> 99%) → very low mirror loss
/// - Short cavity → wide longitudinal mode spacing (no multimode issue)
///
/// This model uses a simplified lumped-parameter description suitable for
/// system-level calculations.
#[derive(Debug, Clone)]
pub struct Vcsel {
    /// Wavelength (nm).
    pub lambda_nm: f64,
    /// Mesa (active region) diameter (µm).
    pub mesa_diameter_um: f64,
    /// Number of DBR mirror pairs per mirror stack.
    pub dbr_periods: usize,
    /// Optical confinement factor Γ.
    pub confinement_factor: f64,
    /// Threshold current I_th (mA).
    pub threshold_current_ma: f64,
    /// Slope efficiency dP/dI (mW/mA).
    pub slope_efficiency_mw_per_ma: f64,
    /// Small-signal 3-dB modulation bandwidth (GHz).
    pub bandwidth_ghz: f64,
}

impl Vcsel {
    /// Construct a VCSEL with explicit parameters.
    pub fn new(
        lambda_nm: f64,
        diameter_um: f64,
        dbr_periods: usize,
        confinement: f64,
        i_th_ma: f64,
        slope_mw_per_ma: f64,
        bw_ghz: f64,
    ) -> Result<Self, OxiPhotonError> {
        if lambda_nm <= 0.0 {
            return Err(OxiPhotonError::InvalidWavelength(lambda_nm * 1.0e-9));
        }
        if i_th_ma <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "threshold current must be positive".into(),
            ));
        }
        Ok(Self {
            lambda_nm,
            mesa_diameter_um: diameter_um.max(1.0),
            dbr_periods,
            confinement_factor: confinement.clamp(0.0, 1.0),
            threshold_current_ma: i_th_ma,
            slope_efficiency_mw_per_ma: slope_mw_per_ma.max(0.0),
            bandwidth_ghz: bw_ghz.max(0.0),
        })
    }

    /// Typical 850 nm VCSEL (data-center interconnect standard).
    pub fn vcsel_850() -> Self {
        Self {
            lambda_nm: 850.0,
            mesa_diameter_um: 10.0,
            dbr_periods: 20,
            confinement_factor: 0.04, // short cavity → low Γ
            threshold_current_ma: 1.5,
            slope_efficiency_mw_per_ma: 0.5,
            bandwidth_ghz: 20.0,
        }
    }

    /// Typical 1550 nm VCSEL (telecom / LiDAR applications).
    pub fn vcsel_1550() -> Self {
        Self {
            lambda_nm: 1550.0,
            mesa_diameter_um: 8.0,
            dbr_periods: 30,
            confinement_factor: 0.03,
            threshold_current_ma: 1.0,
            slope_efficiency_mw_per_ma: 0.3,
            bandwidth_ghz: 15.0,
        }
    }

    /// Output power P_out (mW) versus injected current.
    pub fn output_power_mw(&self, current_ma: f64) -> f64 {
        if current_ma <= self.threshold_current_ma {
            return 0.0;
        }
        self.slope_efficiency_mw_per_ma * (current_ma - self.threshold_current_ma)
    }

    /// Wall-plug efficiency η_wp = P_out / (I · V).
    pub fn wall_plug_efficiency(&self, current_ma: f64, voltage_v: f64) -> f64 {
        if voltage_v <= 0.0 || current_ma <= 0.0 {
            return 0.0;
        }
        let p_out = self.output_power_mw(current_ma) * 1.0e-3; // W
        let p_in = current_ma * 1.0e-3 * voltage_v; // W
        (p_out / p_in).clamp(0.0, 1.0)
    }

    /// Far-field full-angle divergence (degrees).
    ///
    /// Gaussian beam divergence half-angle: θ₁/₂ = λ/(π·w₀)
    /// Full angle (degrees): 2·θ₁/₂·(180/π)
    pub fn divergence_angle_deg(&self) -> f64 {
        let lambda_m = self.lambda_nm * 1.0e-9;
        let w0_m = self.mesa_diameter_um * 0.5 * 1.0e-6; // half-diameter as beam waist
        if w0_m <= 0.0 {
            return 90.0;
        }
        let half_angle_rad = lambda_m / (PI * w0_m);
        2.0 * half_angle_rad * 180.0 / PI
    }
}

// ---------------------------------------------------------------------------
// DfbLaser
// ---------------------------------------------------------------------------

/// Distributed-feedback (DFB) semiconductor laser.
///
/// A periodic grating etched into or adjacent to the active layer provides
/// wavelength-selective feedback through Bragg diffraction. Key advantages:
/// - Single longitudinal mode (high SMSR > 30 dB typical)
/// - Narrow linewidth (< 1 MHz)
/// - Stable wavelength (temperature-tunable)
///
/// The Bragg condition: λ_B = 2·n_eff·Λ  (Λ = grating period).
/// The coupling coefficient κ (m⁻¹) determines the grating strength, and
/// the product κ·L characterises the single-mode stability.
#[derive(Debug, Clone)]
pub struct DfbLaser {
    /// Emission wavelength (nm).
    pub lambda_nm: f64,
    /// Cavity length L (µm).
    pub cavity_length_um: f64,
    /// Grating coupling coefficient κ (m⁻¹).
    pub coupling_coefficient_per_m: f64,
    /// Threshold current (mA).
    pub threshold_current_ma: f64,
    /// Slope efficiency (mW/mA).
    pub slope_efficiency_mw_per_ma: f64,
    /// Side-mode suppression ratio (dB).
    pub side_mode_suppression_db: f64,
}

impl DfbLaser {
    /// Construct a DFB laser.
    pub fn new(
        lambda_nm: f64,
        length_um: f64,
        kappa: f64,
        i_th: f64,
        slope: f64,
        smsr: f64,
    ) -> Result<Self, OxiPhotonError> {
        if lambda_nm <= 0.0 {
            return Err(OxiPhotonError::InvalidWavelength(lambda_nm * 1.0e-9));
        }
        if length_um <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "cavity length must be positive".into(),
            ));
        }
        if kappa <= 0.0 {
            return Err(OxiPhotonError::NumericalError(
                "coupling coefficient κ must be positive".into(),
            ));
        }
        Ok(Self {
            lambda_nm,
            cavity_length_um: length_um,
            coupling_coefficient_per_m: kappa,
            threshold_current_ma: i_th.max(0.0),
            slope_efficiency_mw_per_ma: slope.max(0.0),
            side_mode_suppression_db: smsr.max(0.0),
        })
    }

    /// Typical telecom DFB at 1550 nm (C-band).
    pub fn dfb_1550() -> Self {
        Self {
            lambda_nm: 1550.0,
            cavity_length_um: 300.0,
            coupling_coefficient_per_m: 1.0e4, // κ = 100 cm⁻¹ = 10⁴ m⁻¹
            threshold_current_ma: 10.0,
            slope_efficiency_mw_per_ma: 0.2,
            side_mode_suppression_db: 40.0,
        }
    }

    /// Output power (mW).
    pub fn output_power_mw(&self, current_ma: f64) -> f64 {
        if current_ma <= self.threshold_current_ma {
            return 0.0;
        }
        self.slope_efficiency_mw_per_ma * (current_ma - self.threshold_current_ma)
    }

    /// κ·L product.
    ///
    /// This dimensionless product determines the strength of the grating:
    /// - κ·L < 0.5 : weak grating (requires facet coating)
    /// - κ·L ≈ 1   : moderate grating (single-mode, preferred)
    /// - κ·L > 3   : strong grating (spatial hole burning risk)
    pub fn kappa_l_product(&self) -> f64 {
        let l_m = self.cavity_length_um * 1.0e-6;
        self.coupling_coefficient_per_m * l_m
    }

    /// Returns `true` if the DFB operates in a robust single-mode regime.
    ///
    /// Empirical criterion: κ·L > 1 ensures adequate side-mode suppression
    /// without requiring anti-reflection coatings on both facets.
    pub fn is_single_mode(&self) -> bool {
        self.kappa_l_product() > 1.0
    }

    /// Stopband width Δλ (nm).
    ///
    /// The photonic stopband of the grating:
    /// ```text
    /// Δλ = κ·λ² / (π·n_eff)
    /// ```
    ///
    /// Using n_eff ≈ 3.5 (typical InGaAsP effective index).
    pub fn stopband_width_nm(&self) -> f64 {
        let n_eff = 3.5_f64; // typical effective index
        let lambda_m = self.lambda_nm * 1.0e-9;
        self.coupling_coefficient_per_m * lambda_m * lambda_m / (PI * n_eff) * 1.0e9
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ingaasp() -> SemiconductorLaser {
        SemiconductorLaser::ingaasp_1550()
    }

    #[test]
    fn test_threshold_current_ingaasp() {
        let laser = ingaasp();
        let i_th = laser.threshold_current_ma();
        // Typical InGaAsP FP laser: threshold 10–50 mA for 300 µm × 2 µm
        assert!(
            i_th > 1.0 && i_th < 200.0,
            "InGaAsP threshold should be 1–200 mA, got {i_th:.2} mA"
        );
    }

    #[test]
    fn test_output_power_above_threshold() {
        let laser = ingaasp();
        let i_th = laser.threshold_current_ma();
        let power = laser.output_power_mw(2.0 * i_th, 0.9);
        assert!(
            power > 0.0,
            "output power should be positive above threshold, got {power:.4} mW"
        );
    }

    #[test]
    fn test_output_power_below_threshold_small() {
        let laser = ingaasp();
        let i_th = laser.threshold_current_ma();
        let power = laser.output_power_mw(0.5 * i_th, 0.9);
        // Below threshold, our model returns 0 (no spontaneous output modelled)
        assert!(
            power <= 0.0,
            "output power should be zero below threshold, got {power:.4e} mW"
        );
    }

    #[test]
    fn test_slope_efficiency_positive() {
        let laser = ingaasp();
        let slope = laser.slope_efficiency_mw_per_ma(0.9);
        assert!(
            slope > 0.0,
            "slope efficiency must be positive: {slope:.4e} mW/mA"
        );
        // Typical range: 0.05–0.5 mW/mA for InGaAsP
        assert!(
            slope < 1.0,
            "slope efficiency > 1 mW/mA is unrealistically high: {slope:.4e}"
        );
    }

    #[test]
    fn test_relaxation_oscillation_positive() {
        let laser = ingaasp();
        let i_th = laser.threshold_current_ma();
        let f_ro = laser.relaxation_oscillation_ghz(3.0 * i_th, 0.9);
        assert!(
            f_ro > 0.0,
            "relaxation oscillation frequency must be positive: {f_ro:.3} GHz"
        );
        // Typical range: 1–20 GHz for diode lasers
        assert!(
            f_ro < 100.0,
            "f_RO > 100 GHz is unrealistically high: {f_ro:.3} GHz"
        );
    }

    #[test]
    fn test_modulation_bandwidth_3db() {
        let laser = ingaasp();
        let i_th = laser.threshold_current_ma();
        let bw = laser.modulation_bandwidth_3db_ghz(3.0 * i_th);
        let f_ro = laser.relaxation_oscillation_ghz(3.0 * i_th, 0.9);
        // f_3dB = sqrt(3) * f_RO
        let expected = 3.0_f64.sqrt() * f_ro;
        let rel_err = (bw - expected).abs() / (expected + 1.0e-20);
        assert!(
            rel_err < 1.0e-6,
            "3-dB bandwidth: expected {expected:.4} GHz, got {bw:.4} GHz"
        );
    }

    #[test]
    fn test_linewidth_decreases_with_power() {
        let laser = ingaasp();
        let lw_low = laser.linewidth_mhz(1.0); // 1 mW
        let lw_high = laser.linewidth_mhz(10.0); // 10 mW
        assert!(
            lw_low > lw_high,
            "linewidth should decrease with power: {lw_low:.4} MHz @ 1 mW > {lw_high:.4} MHz @ 10 mW"
        );
    }

    #[test]
    fn test_vcsel_output_power() {
        let vcsel = Vcsel::vcsel_850();
        let p_below = vcsel.output_power_mw(0.5 * vcsel.threshold_current_ma);
        let p_above = vcsel.output_power_mw(3.0 * vcsel.threshold_current_ma);
        assert_eq!(p_below, 0.0, "VCSEL below threshold should give 0 mW");
        assert!(
            p_above > 0.0,
            "VCSEL above threshold should give positive power: {p_above:.4} mW"
        );
    }

    #[test]
    fn test_dfb_single_mode_condition() {
        // DFB with κ·L > 1 should be single-mode
        let dfb_strong = DfbLaser::new(1550.0, 300.0, 1.0e4, 10.0, 0.2, 40.0).expect("valid DFB");
        // κ·L = 10⁴ m⁻¹ × 300e-6 m = 3.0 > 1 → single mode
        assert!(
            dfb_strong.is_single_mode(),
            "DFB with κL={:.2} should be single-mode",
            dfb_strong.kappa_l_product()
        );

        // Weak grating: κ·L < 1
        let dfb_weak = DfbLaser::new(1550.0, 300.0, 2.0e3, 10.0, 0.2, 40.0).expect("valid DFB");
        // κ·L = 2000 × 300e-6 = 0.6 < 1 → not robustly single-mode
        assert!(
            !dfb_weak.is_single_mode(),
            "DFB with κL={:.2} should not be single-mode",
            dfb_weak.kappa_l_product()
        );
    }

    #[test]
    fn test_dfb_stopband_width() {
        let dfb = DfbLaser::dfb_1550();
        let delta_lambda = dfb.stopband_width_nm();
        // Δλ = κ·λ²/(π·n) = 10⁴ × (1550e-9)² / (π × 3.5) × 10⁹ nm
        // ≈ 10⁴ × 2.4025e-12 / 10.996 × 10⁹ ≈ 2.19 nm
        assert!(
            delta_lambda > 0.1 && delta_lambda < 10.0,
            "DFB stopband should be 0.1–10 nm for typical parameters, got {delta_lambda:.4} nm"
        );
    }

    #[test]
    fn test_vcsel_divergence_angle() {
        let vcsel = Vcsel::vcsel_850();
        let theta = vcsel.divergence_angle_deg();
        // For w0 = 5 µm at 850 nm: θ_full = 2·λ/(π·w0)·(180/π) ≈ 6.2°
        assert!(
            theta > 0.0 && theta < 60.0,
            "VCSEL divergence should be 0–60°, got {theta:.2}°"
        );
    }

    #[test]
    fn test_wall_plug_efficiency_bounded() {
        let vcsel = Vcsel::vcsel_1550();
        let eta = vcsel.wall_plug_efficiency(5.0, 2.0); // 5 mA, 2 V
        assert!(
            (0.0..=1.0).contains(&eta),
            "wall-plug efficiency must be in [0, 1]: {eta:.4}"
        );
    }

    #[test]
    fn test_photon_lifetime_from_cavity() {
        let laser = ingaasp();
        let tau_p = laser.photon_lifetime_s();
        // Typical range for InGaAsP: 1–10 ps
        assert!(
            tau_p > 0.1e-12 && tau_p < 50.0e-12,
            "photon lifetime should be 0.1–50 ps, got {:.4e} s",
            tau_p
        );
    }

    #[test]
    fn test_material_gain_positive_above_transparency() {
        let laser = ingaasp();
        let n_th = laser.threshold_carrier_density();
        let g = laser.material_gain(n_th);
        assert!(
            g > 0.0,
            "material gain above transparency must be positive: {g:.4e}"
        );
    }

    #[test]
    fn test_mirror_loss_positive() {
        let laser = ingaasp();
        let alpha_m = laser.mirror_loss();
        assert!(
            alpha_m > 0.0,
            "mirror loss must be positive: {alpha_m:.4e} m⁻¹"
        );
    }

    #[test]
    fn photon_density_ss_above_threshold() {
        let laser = ingaasp();
        let ith = laser.threshold_current_ma();
        let s = laser.photon_density_ss(ith * 2.0, 0.8);
        assert!(
            s > 0.0,
            "photon density should be positive above threshold, got {s}"
        );
        let s_below = laser.photon_density_ss(ith * 0.5, 0.8);
        assert_eq!(s_below, 0.0, "photon density should be 0 below threshold");
    }
}
