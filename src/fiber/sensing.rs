use crate::error::{OxiPhotonError, Result};

/// Speed of light in vacuum (m/s)
const C0: f64 = 2.99792458e8;
/// Planck's constant (J·s)
const H_PLANCK: f64 = 6.62607015e-34;
/// Boltzmann's constant (J/K)
const K_BOLTZMANN: f64 = 1.380649e-23;

// ---------------------------------------------------------------------------
// OtdrEvent
// ---------------------------------------------------------------------------

/// A discrete perturbation on the fiber that affects the OTDR trace.
///
/// Can represent splices, connectors, mechanical faults, or the fiber end.
#[derive(Debug, Clone)]
pub struct OtdrEvent {
    /// Position along the fiber (km)
    pub position_km: f64,
    /// Insertion loss at this point (dB) — positive value means attenuation
    pub loss_db: f64,
    /// Back-reflection / return loss at this point (dB).
    /// 0 → no reflection; -14 → Fresnel; -60 → very low.
    /// Stored as a *positive* dB magnitude (power ratio: 10^(-reflectance_db/10)).
    pub reflectance_db: f64,
}

impl OtdrEvent {
    /// Fusion splice: low loss (0.1 dB typical), negligible reflection.
    pub fn splice(position_km: f64, loss_db: f64) -> Self {
        Self {
            position_km,
            loss_db,
            reflectance_db: -80.0,
        }
    }

    /// Mechanical connector: higher loss, Fresnel reflection (≈ -14 dB for open PC).
    pub fn connector(position_km: f64, loss_db: f64) -> Self {
        Self {
            position_km,
            loss_db,
            reflectance_db: -14.0,
        }
    }

    /// Fiber end (cleaved): Fresnel reflection from glass-air interface (≈ -14.3 dB).
    pub fn fiber_end(position_km: f64) -> Self {
        Self {
            position_km,
            loss_db: 100.0,
            reflectance_db: -14.3,
        }
    }

    /// Mechanical fault: high insertion loss and anomalous reflection.
    pub fn fault(position_km: f64) -> Self {
        Self {
            position_km,
            loss_db: 10.0,
            reflectance_db: -40.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Otdr
// ---------------------------------------------------------------------------

/// OTDR (Optical Time-Domain Reflectometry) instrument model.
///
/// The OTDR sends short optical pulses into the fiber and detects the
/// continuously returning Rayleigh backscattered power as a function of time,
/// which is converted to distance via z = c·t/(2·n).
///
/// The returned backscatter trace is:
///   P_bs(z) \[dBm\] = P_launch \[dBm\] + S \[dB\] − 2·α \[dB/km\]·z \[km\]
///
/// where S is the Rayleigh backscatter coefficient and α is the fiber loss.
#[derive(Debug, Clone)]
pub struct Otdr {
    /// Probe wavelength (nm)
    pub wavelength_nm: f64,
    /// Optical pulse width τ (ns) — determines spatial resolution
    pub pulse_width_ns: f64,
    /// Launch (peak) power into the fiber (mW)
    pub peak_power_mw: f64,
    /// Fiber attenuation coefficient α (dB/km)
    pub fiber_loss_db_per_km: f64,
    /// Rayleigh backscatter coefficient S (dB) relative to launch power.
    /// Typical value for SMF-28 at 1550 nm: ≈ −77 dB.
    pub backscatter_coefficient_db: f64,
    /// Receiver noise floor (dBm) — sets the detection limit
    pub receiver_noise_floor_dbm: f64,
    /// Number of pulse averages for SNR improvement
    pub averaging_factor: usize,
    /// Effective group index (used for time→distance conversion)
    pub group_index: f64,
}

impl Otdr {
    /// Construct a new OTDR model.
    ///
    /// # Arguments
    /// * `lambda_nm`       — Wavelength (nm)
    /// * `pulse_ns`        — Pulse width (ns)
    /// * `power_mw`        — Launch power (mW)
    /// * `loss_db_km`      — Fiber loss (dB/km)
    /// * `backscatter_db`  — Backscatter coefficient (dB) — typically negative, e.g. −77
    /// * `noise_floor_dbm` — Noise floor (dBm) — typically negative, e.g. −45
    pub fn new(
        lambda_nm: f64,
        pulse_ns: f64,
        power_mw: f64,
        loss_db_km: f64,
        backscatter_db: f64,
        noise_floor_dbm: f64,
    ) -> Result<Self> {
        if lambda_nm <= 0.0 || !lambda_nm.is_finite() {
            return Err(OxiPhotonError::InvalidWavelength(lambda_nm * 1e-9));
        }
        if pulse_ns <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "pulse_width_ns must be positive, got {pulse_ns}"
            )));
        }
        if power_mw <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "peak_power_mw must be positive, got {power_mw}"
            )));
        }
        if loss_db_km <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "fiber_loss_db_per_km must be positive, got {loss_db_km}"
            )));
        }
        Ok(Self {
            wavelength_nm: lambda_nm,
            pulse_width_ns: pulse_ns,
            peak_power_mw: power_mw,
            fiber_loss_db_per_km: loss_db_km,
            backscatter_coefficient_db: backscatter_db,
            receiver_noise_floor_dbm: noise_floor_dbm,
            averaging_factor: 1,
            group_index: 1.4682,
        })
    }

    /// Standard SMF-28 OTDR configuration at 1550 nm.
    ///
    /// Typical field-instrument parameters:
    ///   τ = 100 ns, P = 1 mW, α = 0.2 dB/km, S = −77 dB, noise = −45 dBm.
    pub fn smf28_1550() -> Result<Self> {
        Self::new(1550.0, 100.0, 1.0, 0.2, -77.0, -45.0)
    }

    /// Spatial resolution δz = c·τ / (2·n_g) \[m\].
    ///
    /// Two closely-spaced events can be resolved only if their separation > δz.
    pub fn spatial_resolution_m(&self) -> f64 {
        let tau_s = self.pulse_width_ns * 1e-9;
        C0 * tau_s / (2.0 * self.group_index)
    }

    /// Instrument dynamic range DR = (P_launch_dBm − P_noise_dBm) + SNR_averaging \[dB\].
    ///
    /// Averaging improves the noise floor by 5 dB per decade of averages
    /// (10·log10(N_avg)/2 due to voltage averaging).
    pub fn dynamic_range_db(&self) -> f64 {
        let p_launch_dbm = 10.0 * (self.peak_power_mw).log10();
        let snr_avg = if self.averaging_factor > 1 {
            5.0 * (self.averaging_factor as f64).log10()
        } else {
            0.0
        };
        p_launch_dbm - self.receiver_noise_floor_dbm + snr_avg
    }

    /// Maximum measurement distance z_max = DR / (2·α) \[km\].
    ///
    /// The factor of 2 accounts for the round-trip path of the signal.
    pub fn max_distance_km(&self) -> f64 {
        self.dynamic_range_db() / (2.0 * self.fiber_loss_db_per_km)
    }

    /// Backscattered signal level at distance z (km), in dBm.
    ///
    ///   P_bs(z) \[dBm\] = P_launch \[dBm\] + S \[dB\] − 2·α \[dB/km\] · z \[km\]
    pub fn backscattered_power_dbm(&self, distance_km: f64) -> f64 {
        let p_launch_dbm = 10.0 * self.peak_power_mw.log10();
        p_launch_dbm + self.backscatter_coefficient_db
            - 2.0 * self.fiber_loss_db_per_km * distance_km
    }

    /// Simulate a complete OTDR trace for a fiber with discrete events.
    ///
    /// Returns `(distance_m, power_dBm)` pairs.  At each event position a
    /// Fresnel reflection spike is added with the specified reflectance, and the
    /// cumulative insertion losses are tracked.
    ///
    /// The continuous Rayleigh backscatter baseline decays at −2α per km.
    pub fn simulate_trace(
        &self,
        fiber_length_km: f64,
        events: &[OtdrEvent],
        n_pts: usize,
    ) -> Vec<(f64, f64)> {
        if n_pts == 0 || fiber_length_km <= 0.0 {
            return Vec::new();
        }
        let p_launch_dbm = 10.0 * self.peak_power_mw.log10();
        let _dead_zone_km = self.dead_zone_m() / 1000.0;

        // Sort events by position
        let mut sorted_events = events.to_vec();
        sorted_events.sort_by(|a, b| {
            a.position_km
                .partial_cmp(&b.position_km)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Pre-compute cumulative losses at each event
        let mut cum_loss_db = 0.0_f64;
        let mut event_cum_losses: Vec<f64> = Vec::with_capacity(sorted_events.len());
        for ev in &sorted_events {
            cum_loss_db += ev.loss_db;
            event_cum_losses.push(cum_loss_db);
        }

        let dead_zone_m = self.dead_zone_m();
        let _ = event_cum_losses; // pre-computed structure no longer needed

        (0..n_pts)
            .map(|i| {
                let z_km = fiber_length_km * i as f64 / (n_pts - 1).max(1) as f64;
                let z_m = z_km * 1000.0;

                // Continuous Rayleigh backscatter baseline (round-trip loss = 2α)
                let mut power_dbm = p_launch_dbm + self.backscatter_coefficient_db
                    - 2.0 * self.fiber_loss_db_per_km * z_km;

                // Cumulative insertion loss from all events at or before z
                let cumulative_event_loss: f64 = sorted_events
                    .iter()
                    .filter(|ev| ev.position_km <= z_km)
                    .map(|ev| ev.loss_db)
                    .sum();
                power_dbm -= cumulative_event_loss;

                // Reflection spikes at events within the dead zone
                let spike_mw: f64 = sorted_events
                    .iter()
                    .filter(|ev| {
                        let dz_m = (ev.position_km - z_km).abs() * 1000.0;
                        dz_m < dead_zone_m && ev.reflectance_db > -70.0
                    })
                    .map(|ev| {
                        let one_way_loss_db = 2.0 * self.fiber_loss_db_per_km * ev.position_km;
                        let p_spike_dbm = p_launch_dbm + ev.reflectance_db - one_way_loss_db;
                        10.0_f64.powf(p_spike_dbm / 10.0)
                    })
                    .sum();

                let bs_mw = 10.0_f64.powf(power_dbm / 10.0);
                let total_mw = bs_mw + spike_mw;
                let total_dbm = 10.0 * total_mw.max(1e-30).log10();
                (z_m, total_dbm)
            })
            .collect()
    }

    /// Locate positions of reflection events (spikes) in a trace.
    ///
    /// Returns distances (m) where the trace exceeds the baseline by more than
    /// `threshold_db` dB.  Uses a simple peak-finding approach.
    pub fn locate_reflection(&self, trace: &[(f64, f64)], threshold_db: f64) -> Vec<f64> {
        if trace.len() < 3 {
            return Vec::new();
        }
        // Compute a smoothed baseline using a running average over 5 points
        let n = trace.len();
        let mut peaks = Vec::new();

        for i in 2..n - 2 {
            let (z, p) = trace[i];
            // Local neighbours
            let left_avg = (trace[i - 2].1 + trace[i - 1].1) / 2.0;
            let right_avg = (trace[i + 1].1 + trace[i + 2].1) / 2.0;
            let baseline = (left_avg + right_avg) / 2.0;
            if p - baseline > threshold_db && p >= trace[i - 1].1 && p >= trace[i + 1].1 {
                peaks.push(z);
            }
        }
        peaks
    }

    /// Compute fiber loss between two points using the two-point method.
    ///
    ///   α = (P1 − P2) / (2·(z2 − z1))  \[dB/km\]
    ///
    /// where z1 < z2 are in km.
    pub fn two_point_loss_db_per_km(
        &self,
        p1_dbm: f64,
        p2_dbm: f64,
        z1_km: f64,
        z2_km: f64,
    ) -> f64 {
        let dz = z2_km - z1_km;
        if dz.abs() < 1e-9 {
            return 0.0;
        }
        // One-way slope from backscatter = 2α (round trip), so α = (P1-P2)/(2*dz)
        (p1_dbm - p2_dbm) / (2.0 * dz)
    }

    /// Event dead zone (m): the distance over which the receiver is blinded after
    /// a strong reflection. Approximately equal to the pulse spatial extent.
    ///
    ///   dz_dead ≈ c · τ / n_g
    pub fn dead_zone_m(&self) -> f64 {
        let tau_s = self.pulse_width_ns * 1e-9;
        C0 * tau_s / self.group_index
    }
}

// ---------------------------------------------------------------------------
// BotdaSensor
// ---------------------------------------------------------------------------

/// BOTDA (Brillouin Optical Time-Domain Analysis) distributed sensor.
///
/// In BOTDA, a pulsed pump and a CW probe are launched from opposite ends.
/// Stimulated Brillouin scattering (SBS) transfers power from pump to probe
/// when the frequency offset equals the local Brillouin shift ν_B.
/// By scanning the probe frequency, the spatial profile of ν_B is recovered,
/// yielding temperature and strain with metre-scale resolution.
///
/// Brillouin frequency shift:
///   ν_B(T, ε) = ν_B0 + C_T·ΔT + C_ε·ε
///
/// For standard SMF-28 at 1550 nm:
///   ν_B0 ≈ 10.83 GHz, C_T ≈ 1.07 MHz/°C, C_ε ≈ 0.05 MHz/με.
#[derive(Debug, Clone)]
pub struct BotdaSensor {
    /// Total fiber sensing length (km)
    pub fiber_length_km: f64,
    /// Free-running Brillouin frequency shift ν_B0 (GHz) at reference T and ε = 0
    pub brillouin_shift_ghz: f64,
    /// Peak Brillouin gain coefficient g_B (m/W) — ≈ 5×10⁻¹¹ m/W for SMF
    pub brillouin_gain_coefficient: f64,
    /// Spatial resolution (m), determined by pump pulse width
    pub spatial_resolution_m: f64,
    /// Fiber one-way loss α (dB/km)
    pub fiber_loss_db_per_km: f64,
    /// Brillouin gain linewidth Δν_B (MHz, FWHM) — ≈ 30 MHz for SMF at room T
    pub linewidth_mhz: f64,
    /// Temperature coefficient C_T (MHz/°C)
    pub strain_coefficient_mhz_per_microstrain: f64,
    /// Strain coefficient C_ε (MHz/με)
    pub temperature_coefficient_mhz_per_c: f64,
}

impl BotdaSensor {
    /// Construct a BOTDA sensor with user-specified parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        length_km: f64,
        brillouin_shift_ghz: f64,
        gain_coeff: f64,
        resolution_m: f64,
        loss_db_km: f64,
        linewidth_mhz: f64,
        strain_coeff: f64,
        temp_coeff: f64,
    ) -> Result<Self> {
        if length_km <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "fiber_length_km must be positive, got {length_km}"
            )));
        }
        if resolution_m <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "spatial_resolution_m must be positive, got {resolution_m}"
            )));
        }
        Ok(Self {
            fiber_length_km: length_km,
            brillouin_shift_ghz,
            brillouin_gain_coefficient: gain_coeff,
            spatial_resolution_m: resolution_m,
            fiber_loss_db_per_km: loss_db_km,
            linewidth_mhz,
            strain_coefficient_mhz_per_microstrain: strain_coeff,
            temperature_coefficient_mhz_per_c: temp_coeff,
        })
    }

    /// Standard SMF-28 BOTDA sensor with literature parameters.
    ///
    /// ν_B0 = 10.83 GHz, g_B = 5×10⁻¹¹ m/W, Δν_B = 30 MHz,
    /// C_T = 1.07 MHz/°C, C_ε = 0.05 MHz/με, α = 0.2 dB/km.
    pub fn smf28_standard(length_km: f64, resolution_m: f64) -> Result<Self> {
        Self::new(
            length_km,
            10.83, // ν_B0 (GHz)
            5e-11, // g_B (m/W)
            resolution_m,
            0.2,  // α (dB/km)
            30.0, // Δν_B (MHz)
            0.05, // C_ε (MHz/με)
            1.07, // C_T (MHz/°C)
        )
    }

    /// Brillouin frequency shift due to temperature change.
    ///
    ///   Δν_B(T) = C_T · ΔT  \[GHz\]
    pub fn frequency_shift_from_temperature(&self, delta_t_c: f64) -> f64 {
        self.temperature_coefficient_mhz_per_c * delta_t_c * 1e-3 // MHz → GHz
    }

    /// Brillouin frequency shift due to strain.
    ///
    ///   Δν_B(ε) = C_ε · ε  \[GHz\]
    pub fn frequency_shift_from_strain(&self, strain_microstrain: f64) -> f64 {
        self.strain_coefficient_mhz_per_microstrain * strain_microstrain * 1e-3 // MHz → GHz
    }

    /// Recover temperature from a measured Brillouin frequency shift (strain-free).
    ///
    ///   ΔT = Δν_B / C_T
    pub fn temperature_from_shift(&self, delta_nu_ghz: f64) -> f64 {
        let c_t_ghz = self.temperature_coefficient_mhz_per_c * 1e-3;
        if c_t_ghz.abs() < 1e-30 {
            return 0.0;
        }
        delta_nu_ghz / c_t_ghz
    }

    /// Recover strain from a measured Brillouin frequency shift (temperature-compensated).
    ///
    ///   ε = Δν_B / C_ε
    pub fn strain_from_shift(&self, delta_nu_ghz: f64) -> f64 {
        let c_e_ghz = self.strain_coefficient_mhz_per_microstrain * 1e-3;
        if c_e_ghz.abs() < 1e-30 {
            return 0.0;
        }
        delta_nu_ghz / c_e_ghz
    }

    /// Brillouin gain spectrum — Lorentzian profile.
    ///
    ///   g_B(ν) = g_B0 · (Δν_B/2)² / ((ν − ν_center)² + (Δν_B/2)²)
    ///
    /// # Arguments
    /// * `freq_ghz`   — Evaluation frequency (GHz)
    /// * `center_ghz` — Centre frequency of the Brillouin peak (GHz)
    ///
    /// Returns the normalised gain (dimensionless, max = 1.0 at centre).
    pub fn gain_spectrum(&self, freq_ghz: f64, center_ghz: f64) -> f64 {
        let half_bw = self.linewidth_mhz / 2.0 * 1e-3; // GHz
        let detuning = freq_ghz - center_ghz;
        half_bw * half_bw / (detuning * detuning + half_bw * half_bw)
    }

    /// Accumulated Brillouin gain for the pump-probe interaction over distance z.
    ///
    /// Simplified model: gain ∝ g_B · P_pump · Aeff⁻¹ · L_eff
    /// where L_eff = (1 − e^{−αz}) / α.
    ///
    /// Returns the *linear* gain factor (not in dB).
    fn brillouin_gain_linear(&self, z_km: f64, pump_power_w: f64) -> f64 {
        // Convert dB/km to Neper/m: α_Np/m = α_dB/km * ln(10) / 10_000
        let alpha_np = self.fiber_loss_db_per_km * std::f64::consts::LN_10 / 10_000.0;
        let z_m = z_km * 1e3;
        let l_eff = if alpha_np.abs() < 1e-30 {
            z_m
        } else {
            (1.0 - (-alpha_np * z_m).exp()) / alpha_np
        };
        let aeff = 80e-12; // m² (typical SMF effective mode area)
        (self.brillouin_gain_coefficient * pump_power_w * l_eff / aeff).exp()
    }

    /// Signal-to-noise ratio at position z for a given pump power.
    ///
    /// Approximate model based on signal gain vs. noise figure.
    /// SNR (linear) = G_B(z) / F_noise where F_noise ≈ 2 for Brillouin amplifier.
    pub fn snr_at_distance(&self, z_km: f64, pump_power_w: f64) -> f64 {
        let gain = self.brillouin_gain_linear(z_km, pump_power_w);
        // Round-trip propagation loss
        let alpha_db_per_m = self.fiber_loss_db_per_km / 1000.0;
        let loss_factor = 10.0_f64.powf(-2.0 * alpha_db_per_m * z_km * 1000.0 / 10.0);
        let noise_figure = 2.0_f64;
        gain * loss_factor / noise_figure
    }

    /// Temperature uncertainty (°C) from measurement SNR.
    ///
    /// δT = Δν_B / (C_T · √SNR)  — uncertainty scales as 1/√SNR.
    pub fn temperature_uncertainty_c(&self, snr_linear: f64) -> f64 {
        if snr_linear <= 0.0 {
            return f64::INFINITY;
        }
        let linewidth_ghz = self.linewidth_mhz * 1e-3;
        let c_t_ghz = self.temperature_coefficient_mhz_per_c * 1e-3;
        if c_t_ghz.abs() < 1e-30 {
            return f64::INFINITY;
        }
        linewidth_ghz / (c_t_ghz * snr_linear.sqrt())
    }

    /// Strain uncertainty (με) from measurement SNR.
    ///
    /// δε = Δν_B / (C_ε · √SNR)
    pub fn strain_uncertainty_microstrain(&self, snr_linear: f64) -> f64 {
        if snr_linear <= 0.0 {
            return f64::INFINITY;
        }
        let linewidth_ghz = self.linewidth_mhz * 1e-3;
        let c_e_ghz = self.strain_coefficient_mhz_per_microstrain * 1e-3;
        if c_e_ghz.abs() < 1e-30 {
            return f64::INFINITY;
        }
        linewidth_ghz / (c_e_ghz * snr_linear.sqrt())
    }

    /// Maximum sensing range (km) for a required SNR threshold and pump power.
    ///
    /// Solves G_B(z) · loss(z) / F_noise ≥ SNR_req iteratively.
    /// Uses bisection search over \[0, fiber_length_km\].
    pub fn max_range_km(&self, required_snr_db: f64, pump_power_w: f64) -> f64 {
        let snr_req = 10.0_f64.powf(required_snr_db / 10.0);
        let mut lo = 0.0_f64;
        let mut hi = self.fiber_length_km;

        if self.snr_at_distance(hi, pump_power_w) >= snr_req {
            return hi;
        }
        if self.snr_at_distance(lo, pump_power_w) < snr_req {
            return 0.0;
        }

        for _ in 0..60 {
            let mid = (lo + hi) / 2.0;
            if self.snr_at_distance(mid, pump_power_w) >= snr_req {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (lo + hi) / 2.0
    }
}

// ---------------------------------------------------------------------------
// RamanDts
// ---------------------------------------------------------------------------

/// Distributed Temperature Sensing (DTS) via spontaneous Raman backscattering.
///
/// The temperature is derived from the ratio of anti-Stokes to Stokes Raman
/// intensities.  The Stokes component is nearly temperature-independent, while
/// the anti-Stokes component is strongly temperature-dependent:
///
///   I_AS / I_S = C · exp(−hcΔν̃ / kT)
///
/// where Δν̃ is the Raman shift in wavenumbers, h is Planck's constant,
/// c is the speed of light, k is Boltzmann's constant, and T is temperature (K).
///
/// The proportionality constant C accounts for differential attenuation between
/// the Stokes and anti-Stokes wavelengths.
///
/// References:
/// - Dakin et al., Electron. Lett. 21(13), 1985.
/// - Grattan & Sun, Sensor. Actuators 82(1-3), 40-61 (2000).
#[derive(Debug, Clone)]
pub struct RamanDts {
    /// Total sensing fiber length (km)
    pub fiber_length_km: f64,
    /// Pump wavelength (nm) — determines Stokes/anti-Stokes wavelengths
    pub wavelength_nm: f64,
    /// Raman shift Δν̃ (cm⁻¹) — ≈ 440 cm⁻¹ for germano-silicate fiber
    pub raman_shift_cm: f64,
    /// Spatial resolution (m)
    pub spatial_resolution_m: f64,
    /// Fiber loss at the pump wavelength (dB/km)
    pub fiber_loss_db_per_km: f64,
    /// Temperature measurement accuracy (°C, 1σ)
    pub temperature_accuracy_c: f64,
    /// Signal integration time (s)
    pub integration_time_s: f64,
    /// Differential attenuation between Stokes and anti-Stokes (dB/km)
    pub differential_loss_db_per_km: f64,
}

impl RamanDts {
    /// Construct a Raman DTS system.
    pub fn new(length_km: f64, lambda_nm: f64, resolution_m: f64) -> Result<Self> {
        if length_km <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "fiber_length_km must be positive, got {length_km}"
            )));
        }
        if lambda_nm <= 0.0 || !lambda_nm.is_finite() {
            return Err(OxiPhotonError::InvalidWavelength(lambda_nm * 1e-9));
        }
        if resolution_m <= 0.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "spatial_resolution_m must be positive, got {resolution_m}"
            )));
        }
        Ok(Self {
            fiber_length_km: length_km,
            wavelength_nm: lambda_nm,
            raman_shift_cm: 440.0,
            spatial_resolution_m: resolution_m,
            fiber_loss_db_per_km: 0.35,
            temperature_accuracy_c: 1.0,
            integration_time_s: 60.0,
            differential_loss_db_per_km: 0.05,
        })
    }

    /// Silica fiber DTS system pumped at 1064 nm (Nd:YAG).
    ///
    /// Standard parameters: α = 0.35 dB/km, Δν̃ = 440 cm⁻¹.
    pub fn silica_fiber_1064(length_km: f64, resolution_m: f64) -> Result<Self> {
        let mut sys = Self::new(length_km, 1064.0, resolution_m)?;
        sys.fiber_loss_db_per_km = 0.35;
        sys.raman_shift_cm = 440.0;
        Ok(sys)
    }

    /// Anti-Stokes wavelength (nm).
    ///
    /// The anti-Stokes photon is blue-shifted (higher frequency) relative to the pump:
    ///   ν_AS = ν_pump + Δν̃  →  1/λ_AS = 1/λ_pump + Δν̃
    ///
    /// (1 cm⁻¹ = 100 m⁻¹; wavenumbers add linearly for frequency-domain shifts.)
    pub fn anti_stokes_wavelength_nm(&self) -> f64 {
        let nu_pump_per_m = 1.0 / (self.wavelength_nm * 1e-9); // m⁻¹
        let raman_per_m = self.raman_shift_cm * 100.0; // cm⁻¹ → m⁻¹
        let nu_as = nu_pump_per_m + raman_per_m; // higher frequency
        1.0 / nu_as * 1e9 // m → nm (shorter wavelength)
    }

    /// Stokes wavelength (nm).
    ///
    /// The Stokes photon is red-shifted (lower frequency) relative to the pump:
    ///   ν_S = ν_pump − Δν̃  →  1/λ_S = 1/λ_pump − Δν̃
    pub fn stokes_wavelength_nm(&self) -> f64 {
        let nu_pump_per_m = 1.0 / (self.wavelength_nm * 1e-9);
        let raman_per_m = self.raman_shift_cm * 100.0;
        let nu_s = nu_pump_per_m - raman_per_m; // lower frequency
        1.0 / nu_s * 1e9 // longer wavelength
    }

    /// Convert Stokes and anti-Stokes intensity ratio to temperature (K).
    ///
    ///   R = I_AS / I_S ∝ exp(−hcΔν̃ / kT)
    ///   ⟹  T = −hcΔν̃ / (k · ln(I_AS/I_S / C_ref))
    ///
    /// Here we use C_ref = 1.0 (calibrated system assumption).
    /// Returns temperature in Kelvin; returns `f64::NAN` if ratio is non-positive.
    pub fn temperature_from_ratio(&self, stokes_intensity: f64, anti_stokes_intensity: f64) -> f64 {
        if stokes_intensity <= 0.0 || anti_stokes_intensity <= 0.0 {
            return f64::NAN;
        }
        let ratio = anti_stokes_intensity / stokes_intensity;
        if ratio <= 0.0 {
            return f64::NAN;
        }
        // hcΔν̃ / k in Kelvin
        let raman_per_m = self.raman_shift_cm * 100.0;
        let hc_delta_nu_over_k = H_PLANCK * C0 * raman_per_m / K_BOLTZMANN;
        -hc_delta_nu_over_k / ratio.ln()
    }

    /// Expected Stokes-to-anti-Stokes intensity ratio at temperature T (K).
    ///
    ///   R(T) = C · exp(+hcΔν̃ / kT)
    ///
    /// where C = 1 for a calibrated system (differential loss corrected).
    pub fn expected_ratio(&self, temperature_k: f64) -> f64 {
        if temperature_k <= 0.0 {
            return f64::NAN;
        }
        let raman_per_m = self.raman_shift_cm * 100.0;
        let hc_delta_nu_over_k = H_PLANCK * C0 * raman_per_m / K_BOLTZMANN;
        (hc_delta_nu_over_k / temperature_k).exp()
    }

    /// Temperature sensitivity dR/dT of the Stokes/anti-Stokes ratio.
    ///
    ///   dR/dT = −R(T) · hcΔν̃ / (kT²)
    pub fn ratio_temperature_sensitivity(&self, temperature_k: f64) -> f64 {
        if temperature_k <= 0.0 {
            return f64::NAN;
        }
        let raman_per_m = self.raman_shift_cm * 100.0;
        let hc_delta_nu_over_k = H_PLANCK * C0 * raman_per_m / K_BOLTZMANN;
        let ratio = self.expected_ratio(temperature_k);
        -ratio * hc_delta_nu_over_k / (temperature_k * temperature_k)
    }

    /// Approximate maximum sensing range (km).
    ///
    /// The measurement range is limited by differential loss between the
    /// Stokes and anti-Stokes channels.  The useful range is:
    ///   z_max ≈ T_meas_accuracy / |dR/dT| / R(T_ref) · 1 / (α_diff · ln(10)/10)
    ///
    /// A simpler engineering approximation: z_max = 3 / α_diff \[km\],
    /// where α_diff is the differential loss (dB/km) ≈ 0.05–0.1 dB/km for SMF.
    pub fn max_range_km(&self) -> f64 {
        if self.differential_loss_db_per_km.abs() < 1e-12 {
            return self.fiber_length_km;
        }
        // 3 dB differential attenuation → factor-of-2 imbalance limit
        let z_diff = 3.0 / self.differential_loss_db_per_km;
        // Also limited by total fiber loss and detection SNR
        // Use 20 dB dynamic range / (2 * alpha) as upper bound
        let z_loss = 20.0 / (2.0 * self.fiber_loss_db_per_km);
        z_diff.min(z_loss).min(self.fiber_length_km)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ── OTDR tests ──────────────────────────────────────────────────────────

    fn smf28_otdr() -> Otdr {
        Otdr::smf28_1550().expect("SMF-28 OTDR construction should succeed")
    }

    #[test]
    fn test_otdr_spatial_resolution() {
        // δz = c · τ / (2·n_g)
        let otdr = smf28_otdr();
        let expected = C0 * 100e-9 / (2.0 * 1.4682);
        assert_relative_eq!(otdr.spatial_resolution_m(), expected, max_relative = 1e-9);
    }

    #[test]
    fn test_otdr_dynamic_range_positive() {
        let otdr = smf28_otdr();
        let dr = otdr.dynamic_range_db();
        assert!(dr > 0.0, "Dynamic range should be positive, got {dr}");
        // Typical OTDR DR ≈ 10–40 dB
        assert!(dr < 100.0, "Dynamic range suspiciously large: {dr} dB");
    }

    #[test]
    fn test_otdr_backscattered_power_decreases() {
        let otdr = smf28_otdr();
        let p1 = otdr.backscattered_power_dbm(1.0); // 1 km
        let p2 = otdr.backscattered_power_dbm(5.0); // 5 km
        let p3 = otdr.backscattered_power_dbm(10.0); // 10 km
        assert!(
            p1 > p2,
            "Backscatter should decrease with distance: P(1km)={p1} P(5km)={p2}"
        );
        assert!(
            p2 > p3,
            "Backscatter should decrease with distance: P(5km)={p2} P(10km)={p3}"
        );
    }

    #[test]
    fn test_otdr_two_point_loss() {
        let otdr = smf28_otdr();
        // Simulate two points on a 0.2 dB/km fiber
        let p1 = otdr.backscattered_power_dbm(1.0);
        let p2 = otdr.backscattered_power_dbm(3.0);
        let alpha = otdr.two_point_loss_db_per_km(p1, p2, 1.0, 3.0);
        // Should recover 0.2 dB/km
        assert_relative_eq!(alpha, 0.2, max_relative = 1e-9);
    }

    #[test]
    fn test_otdr_dead_zone_proportional_to_pulse() {
        let otdr_short =
            Otdr::new(1550.0, 10.0, 1.0, 0.2, -77.0, -45.0).expect("OTDR construction");
        let otdr_long =
            Otdr::new(1550.0, 100.0, 1.0, 0.2, -77.0, -45.0).expect("OTDR construction");
        // Dead zone should scale with pulse width
        assert!(
            otdr_long.dead_zone_m() > otdr_short.dead_zone_m(),
            "Longer pulse → larger dead zone"
        );
        assert_relative_eq!(
            otdr_long.dead_zone_m() / otdr_short.dead_zone_m(),
            10.0,
            max_relative = 1e-9
        );
    }

    #[test]
    fn test_otdr_trace_length() {
        let otdr = smf28_otdr();
        let trace = otdr.simulate_trace(10.0, &[], 500);
        assert_eq!(trace.len(), 500);
    }

    #[test]
    fn test_otdr_trace_distances_start_at_zero() {
        let otdr = smf28_otdr();
        let trace = otdr.simulate_trace(5.0, &[], 100);
        assert!(!trace.is_empty());
        assert_relative_eq!(trace[0].0, 0.0, max_relative = 1e-9);
    }

    #[test]
    fn test_otdr_max_distance_positive() {
        let otdr = smf28_otdr();
        let z_max = otdr.max_distance_km();
        assert!(z_max > 0.0, "Max distance should be positive, got {z_max}");
    }

    // ── BOTDA tests ─────────────────────────────────────────────────────────

    fn smf28_botda() -> BotdaSensor {
        BotdaSensor::smf28_standard(25.0, 1.0).expect("BOTDA construction should succeed")
    }

    #[test]
    fn test_botda_frequency_shift_temperature() {
        // C_T = 1.07 MHz/°C = 1.07e-3 GHz/°C
        let botda = smf28_botda();
        let delta_nu = botda.frequency_shift_from_temperature(100.0); // 100 °C
        let expected = 1.07e-3 * 100.0; // GHz
        assert_relative_eq!(delta_nu, expected, max_relative = 1e-9);
    }

    #[test]
    fn test_botda_temperature_from_shift() {
        let botda = smf28_botda();
        let dt = 50.0; // °C
        let shift = botda.frequency_shift_from_temperature(dt);
        let dt_recovered = botda.temperature_from_shift(shift);
        assert_relative_eq!(dt_recovered, dt, max_relative = 1e-9);
    }

    #[test]
    fn test_botda_gain_spectrum_lorentzian() {
        // Gain should be maximum at center frequency
        let botda = smf28_botda();
        let center = botda.brillouin_shift_ghz;
        let g_center = botda.gain_spectrum(center, center);
        let g_off = botda.gain_spectrum(center + 0.1, center); // 100 MHz off
        assert_relative_eq!(g_center, 1.0, max_relative = 1e-9);
        assert!(g_off < g_center, "Off-center gain should be less than peak");
        assert!(g_off > 0.0, "Off-center gain should be positive");
    }

    #[test]
    fn test_botda_strain_frequency_shift() {
        // C_ε = 0.05 MHz/με
        let botda = smf28_botda();
        let shift = botda.frequency_shift_from_strain(100.0); // 100 με
        let expected = 0.05e-3 * 100.0; // GHz
        assert_relative_eq!(shift, expected, max_relative = 1e-9);
    }

    #[test]
    fn test_botda_temperature_uncertainty_decreases_with_snr() {
        let botda = smf28_botda();
        let unc_low = botda.temperature_uncertainty_c(10.0);
        let unc_high = botda.temperature_uncertainty_c(100.0);
        assert!(
            unc_low > unc_high,
            "Higher SNR should give lower temperature uncertainty"
        );
    }

    // ── Raman DTS tests ──────────────────────────────────────────────────────

    fn silica_dts() -> RamanDts {
        RamanDts::silica_fiber_1064(10.0, 1.0).expect("Raman DTS construction")
    }

    #[test]
    fn test_raman_dts_stokes_anti_stokes_wavelengths() {
        let dts = silica_dts();
        let lambda_as = dts.anti_stokes_wavelength_nm();
        let lambda_s = dts.stokes_wavelength_nm();
        let lambda_p = dts.wavelength_nm;

        // Anti-Stokes: shorter wavelength (blue-shifted from pump)
        assert!(
            lambda_as < lambda_p,
            "Anti-Stokes ({lambda_as:.2} nm) should be shorter than pump ({lambda_p} nm)"
        );
        // Stokes: longer wavelength (red-shifted from pump)
        assert!(
            lambda_s > lambda_p,
            "Stokes ({lambda_s:.2} nm) should be longer than pump ({lambda_p} nm)"
        );
        // Symmetric shift in wavenumber space
        let nu_pump = 1.0 / (lambda_p * 1e-9);
        let nu_as = 1.0 / (lambda_as * 1e-9);
        let nu_s = 1.0 / (lambda_s * 1e-9);
        let shift_as = (nu_pump - nu_as).abs();
        let shift_s = (nu_s - nu_pump).abs();
        assert_relative_eq!(shift_as, shift_s, max_relative = 1e-9);
    }

    #[test]
    fn test_raman_temperature_from_ratio() {
        let dts = silica_dts();
        let t_ref = 300.0_f64; // K
                               // Compute expected ratio at T_ref, then invert
        let ratio = dts.expected_ratio(t_ref);
        // R = I_S/I_AS; temperature_from_ratio uses I_AS/I_S
        let t_recovered = dts.temperature_from_ratio(ratio, 1.0);
        // R = exp(hcΔν/kT) → I_S/I_AS; temperature_from_ratio inverts I_AS/I_S
        // So feed I_AS=1, I_S = ratio → ratio = exp(hcΔν/kT)
        // temperature_from_ratio(stokes, anti_stokes) = -hcΔν/k / ln(anti/stokes)
        // If we feed (stokes=ratio, anti_stokes=1.0):
        //   ln(1/ratio) = -hcΔν/kT → T = hcΔν/(k * ln(ratio)) = T_ref ✓
        let t2 = dts.temperature_from_ratio(ratio, 1.0);
        // ln(1.0/ratio) = ln(1) - ln(ratio) = -ln(ratio) = -hcΔν/kT → T = +hcΔν/k/ln(ratio)
        // But our function does: T = -hcΔν/k / ln(I_AS/I_S)
        // With I_AS=1, I_S=ratio: ln(1/ratio) = -ln(ratio)
        // T = -hcΔν/k / (-ln(ratio)) = hcΔν / (k*ln(ratio)) = T_ref  ✓
        let _ = t_recovered;
        // Use expected_ratio as a cross-check: we round-trip through (I_S, I_AS)
        let raman_per_m = dts.raman_shift_cm * 100.0;
        let hc_dk = H_PLANCK * C0 * raman_per_m / K_BOLTZMANN;
        // At T=300K: ratio = exp(hc/kT) means I_S/I_AS = ratio, so I_AS/I_S = 1/ratio
        let t_check = dts.temperature_from_ratio(1.0, 1.0 / ratio);
        // ln((1/ratio)/1) = -ln(ratio) = -hcΔν/kT → T = hcΔν / (k·ln(ratio))
        // = hc_dk / ln(ratio) where ratio = exp(hc_dk/T) → ln(ratio) = hc_dk/T
        // → T_check = hc_dk / (hc_dk/T) = T ✓
        let _ = hc_dk;
        let _ = t2;
        assert!(
            (t_check - t_ref).abs() < 0.1,
            "Round-trip temperature mismatch: expected {t_ref} K, got {t_check} K"
        );
    }

    #[test]
    fn test_raman_expected_ratio_increases_with_temp() {
        // Stokes/anti-Stokes ratio R = exp(hcΔν/kT) increases as T decreases
        // (colder → lower thermal population of upper Raman level → higher ratio)
        let dts = silica_dts();
        let r_cold = dts.expected_ratio(250.0); // 250 K (cold)
        let r_warm = dts.expected_ratio(350.0); // 350 K (warm)
        assert!(
            r_cold > r_warm,
            "Ratio R=I_S/I_AS should be larger at lower temperature: R(250K)={r_cold}, R(350K)={r_warm}"
        );
    }

    #[test]
    fn test_raman_ratio_sensitivity_negative() {
        // dR/dT = -R * hcΔν/kT² < 0 (ratio decreases with increasing T)
        let dts = silica_dts();
        let sens = dts.ratio_temperature_sensitivity(300.0);
        assert!(
            sens < 0.0,
            "dR/dT should be negative (ratio decreases with heating), got {sens}"
        );
    }

    #[test]
    fn test_raman_max_range_positive() {
        let dts = silica_dts();
        let z_max = dts.max_range_km();
        assert!(z_max > 0.0, "Max range should be positive, got {z_max}");
        assert!(
            z_max <= dts.fiber_length_km,
            "Max range should not exceed fiber length"
        );
    }

    #[test]
    fn test_botda_gain_spectrum_at_zero_detuning() {
        let botda = smf28_botda();
        let center = botda.brillouin_shift_ghz;
        let g = botda.gain_spectrum(center, center);
        assert_relative_eq!(g, 1.0, max_relative = 1e-10);
    }
}
