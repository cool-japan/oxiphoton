//! Free-space optical (FSO) link budget analysis.
//!
//! Models the end-to-end received power, fading statistics, and average bit-error
//! rate for terrestrial and near-Earth FSO communication links.
//!
//! # Link Budget Equation
//!
//! P_rx (dBm) = P_tx − L_geo − L_atm − L_pointing + G_rx
//!
//! where the geometric loss combines free-space spreading and the finite receive
//! aperture, and the atmospheric loss includes Rayleigh scattering, Mie aerosol
//! extinction, and molecular absorption.
//!
//! # References
//! - Khalighi & Uysal, "Survey on Free Space Optical Communication", IEEE 2014
//! - Andrews & Phillips, "Laser Beam Propagation through Random Media", 2005
//! - Kruse, McGlauchlin & McQuistan, "Elements of Infrared Technology", 1962

use super::turbulence::{
    erfc_approx, AtmosphericPath, GammaGammaDistribution, LogNormalScintillation, TurbulenceRegime,
};
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// FsoModulation
// ─────────────────────────────────────────────────────────────────────────────

/// Modulation format used by the FSO link.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FsoModulation {
    /// On-Off Keying with intensity modulation / direct detection.
    OokIm,
    /// Pulse-Position Modulation with M slots.
    Bppm { m: usize },
    /// Coherent dual-polarisation QPSK.
    DpQpsk,
}

// ─────────────────────────────────────────────────────────────────────────────
// AerosolType
// ─────────────────────────────────────────────────────────────────────────────

/// Aerosol type affecting the extinction coefficient wavelength scaling.
#[derive(Debug, Clone)]
pub enum AerosolType {
    /// Maritime aerosol (humidity, sea spray).
    Maritime,
    /// Continental background aerosol.
    Continental,
    /// Urban industrial aerosol.
    Urban,
    /// Desert dust aerosol.
    Desert,
    /// Rain (Carbonneau 1998 model).
    Rain {
        /// Rainfall rate in mm/hr.
        rate_mm_per_hr: f64,
    },
    /// Radiation fog or advection fog.
    Fog {
        /// Liquid water content in g/m³.
        liquid_water_content: f64,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// AtmosphericExtinction
// ─────────────────────────────────────────────────────────────────────────────

/// Atmospheric extinction model combining Rayleigh scattering and aerosol Mie
/// scattering via the Kruse–Kim–Middleton model.
#[derive(Debug, Clone)]
pub struct AtmosphericExtinction {
    /// Meteorological visibility in km.
    pub visibility_km: f64,
    /// Optical wavelength in metres.
    pub wavelength: f64,
    /// Aerosol type that determines the q exponent.
    pub aerosol_type: AerosolType,
}

impl AtmosphericExtinction {
    /// Construct a new extinction model.
    pub fn new(visibility_km: f64, wavelength: f64, aerosol: AerosolType) -> Self {
        Self {
            visibility_km: visibility_km.max(0.01),
            wavelength,
            aerosol_type: aerosol,
        }
    }

    /// Kim–Middleton exponent q for the Kruse model.
    ///
    /// V is the meteorological visibility in km; λ is in µm.
    pub fn q_exponent(&self) -> f64 {
        let v = self.visibility_km;
        match &self.aerosol_type {
            AerosolType::Rain { rate_mm_per_hr } => {
                // Carbonneau model for rain: β = 0.000365 * R^0.63 (km^{-1}) at 830 nm
                // approximate q ≈ 0 (wavelength independent for large drops)
                let _ = rate_mm_per_hr;
                0.0
            }
            AerosolType::Fog { .. } => {
                // Mie scattering in dense fog: q ≈ 0
                if v < 0.2 {
                    0.0
                } else {
                    0.585 * v.powf(1.0 / 3.0)
                }
            }
            _ => {
                // Kruse model (1962) / Kim (1998)
                if v < 0.5 {
                    0.0
                } else if v < 1.0 {
                    0.585 * v.powf(1.0 / 3.0)
                } else if v < 6.0 {
                    1.3
                } else {
                    1.6
                }
            }
        }
    }

    /// Extinction coefficient β_ext (km^{-1}).
    ///
    /// For visibility model: β = (3.91 / V) * (λ_ref / λ)^q
    /// where λ_ref = 0.55 µm.
    pub fn extinction_coefficient_per_km(&self) -> f64 {
        let v = self.visibility_km;
        let lambda_um = self.wavelength * 1e6; // convert m → µm
        let q = self.q_exponent();
        match &self.aerosol_type {
            AerosolType::Rain { rate_mm_per_hr } => {
                // Carbonneau: β = a * R^b km^{-1}; coefficients for ~1550 nm
                let a = 1.076e-4;
                let b = 0.67;
                a * rate_mm_per_hr.powf(b)
            }
            AerosolType::Fog {
                liquid_water_content,
            } => {
                // Kunkel (1984): β = A * lwc^B km^{-1}
                let lwc = liquid_water_content.max(1e-6);
                0.144 * lwc.powf(0.88)
            }
            _ => (3.91 / v) * (0.55 / lambda_um).powf(q),
        }
    }

    /// Power transmission over distance L (km): T = exp(−β L).
    pub fn transmission(&self, distance_km: f64) -> f64 {
        let beta = self.extinction_coefficient_per_km();
        (-beta * distance_km).exp()
    }

    /// Atmospheric loss in dB/km: α = 4.343 β.
    pub fn loss_db_per_km(&self) -> f64 {
        4.343 * self.extinction_coefficient_per_km()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FsoLink
// ─────────────────────────────────────────────────────────────────────────────

/// Complete FSO link budget.
#[derive(Debug, Clone)]
pub struct FsoLink {
    /// Optical wavelength (m).
    pub wavelength: f64,
    /// Transmit power (dBm).
    pub tx_power_dbm: f64,
    /// Transmitter aperture diameter (m).
    pub tx_aperture_m: f64,
    /// Receiver aperture diameter (m).
    pub rx_aperture_m: f64,
    /// Link distance (km).
    pub link_distance_km: f64,
    /// Full-angle beam divergence (mrad). Use 0 for diffraction-limited.
    pub tx_divergence_mrad: f64,
    /// Fixed pointing loss (dB) — separate from tracking jitter.
    pub pointing_loss_db: f64,
    /// Atmospheric propagation path (turbulence model).
    pub atmospheric: AtmosphericPath,
}

impl FsoLink {
    /// Create a new FSO link with sensible defaults (diffraction-limited divergence,
    /// horizontal path at 1550 nm).
    pub fn new(wavelength: f64, tx_dbm: f64, tx_diam: f64, rx_diam: f64, dist_km: f64) -> Self {
        let cn2_typical = 1e-15; // moderate turbulence
        let atm = AtmosphericPath::new_horizontal(dist_km, cn2_typical, wavelength);
        let div_dl = 2.44 * wavelength / tx_diam; // diffraction-limited full angle
        Self {
            wavelength,
            tx_power_dbm: tx_dbm,
            tx_aperture_m: tx_diam,
            rx_aperture_m: rx_diam,
            link_distance_km: dist_km,
            tx_divergence_mrad: div_dl * 1e3, // → mrad
            pointing_loss_db: 0.0,
            atmospheric: atm,
        }
    }

    /// Diffraction-limited full-angle divergence θ_DL = 2.44 λ / D_tx (radians).
    pub fn diffraction_limited_divergence_rad(&self) -> f64 {
        2.44 * self.wavelength / self.tx_aperture_m
    }

    /// Effective beam divergence (radians): max of requested and diffraction limit.
    fn effective_divergence_rad(&self) -> f64 {
        let dl = self.diffraction_limited_divergence_rad();
        let req = self.tx_divergence_mrad * 1e-3;
        req.max(dl)
    }

    /// Free-space path loss (dB) using the Friis equation:
    /// L_fs = 20 log₁₀(4πR/λ) − G_tx − G_rx
    ///
    /// For a Gaussian beam: G_tx = 0, G_rx = (π D_rx / λ)².
    /// This method returns the combined spreading + aperture loss.
    pub fn free_space_loss_db(&self) -> f64 {
        let r = self.link_distance_km * 1e3;
        let lambda = self.wavelength;
        // Friis free-space path loss (isotropic → isotropic)

        // Receive aperture gain: A_rx = π(D_rx/2)²; G_rx = (π D_rx / λ)² * (λ/4π)² → separate
        20.0 * (4.0 * PI * r / lambda).log10()
    }

    /// Geometric loss from beam spreading: L_geo = 20 log₁₀(θ*R / D_rx) \[dB\].
    ///
    /// Assumes a top-hat approximation: all power falls within spot radius θ*R,
    /// and the receiver captures (D_rx / (2*θ*R))² of it.
    pub fn geometric_loss_db(&self) -> f64 {
        let theta = self.effective_divergence_rad(); // full angle
        let r = self.link_distance_km * 1e3; // m
        let spot_radius = theta * r / 2.0; // half-angle * distance
        let rx_radius = self.rx_aperture_m / 2.0;
        if rx_radius >= spot_radius {
            return 0.0; // full capture
        }
        -20.0 * (rx_radius / spot_radius).log10()
    }

    /// Atmospheric attenuation over the link (dB).
    pub fn atmospheric_loss_db(&self, visibility_km: f64) -> f64 {
        let ext =
            AtmosphericExtinction::new(visibility_km, self.wavelength, AerosolType::Continental);
        ext.loss_db_per_km() * self.link_distance_km
    }

    /// Received power (dBm): P_rx = P_tx − L_geo − L_atm − L_pointing.
    pub fn received_power_dbm(&self, visibility_km: f64) -> f64 {
        self.tx_power_dbm
            - self.geometric_loss_db()
            - self.atmospheric_loss_db(visibility_km)
            - self.pointing_loss_db
    }

    /// Link margin (dB): M = P_rx − P_sensitivity.
    pub fn link_margin_db(&self, sensitivity_dbm: f64, visibility_km: f64) -> f64 {
        self.received_power_dbm(visibility_km) - sensitivity_dbm
    }

    /// Aperture averaging factor A(D_rx) that reduces the effective scintillation
    /// index when the receiver aperture is large.
    ///
    /// Andrews (1998) plane-wave approximation:
    /// A(D) = \[1 + 1.062*(D_rx²/(4*λ*L))^{7/6}\]^{-1}
    pub fn aperture_averaging_factor(&self) -> f64 {
        let l = self.link_distance_km * 1e3;
        let d = self.rx_aperture_m;
        let lambda = self.wavelength;
        let phi = d * d / (4.0 * lambda * l); // dimensionless aperture parameter
        1.0 / (1.0 + 1.062 * phi.powf(7.0 / 6.0))
    }

    /// Effective scintillation index with aperture averaging.
    pub fn effective_scintillation_index(&self) -> f64 {
        let si = self.atmospheric.scintillation_index();
        let a = self.aperture_averaging_factor();
        si * a
    }

    /// Probability that the instantaneous received power fades below `fade_depth_db`
    /// below the mean received power.
    pub fn fade_probability(&self, fade_depth_db: f64, visibility_km: f64) -> f64 {
        let mean_p_dbm = self.received_power_dbm(visibility_km);
        let threshold_dbm = mean_p_dbm - fade_depth_db;
        self.outage_probability(threshold_dbm)
    }

    /// Outage probability: P(P_rx < P_threshold).
    ///
    /// Uses the log-normal model in weak turbulence and Gamma-Gamma otherwise.
    pub fn outage_probability(&self, sensitivity_dbm: f64) -> f64 {
        let eff_si = self.effective_scintillation_index();
        let mean_p_dbm = self.received_power_dbm(23.0); // use 23 km clear-air visibility
        let mean_p_w = 1e-3 * 10.0_f64.powf(mean_p_dbm / 10.0);
        let threshold_w = 1e-3 * 10.0_f64.powf(sensitivity_dbm / 10.0);

        match self.atmospheric.regime() {
            TurbulenceRegime::Weak => {
                let ln_scint = LogNormalScintillation::new(eff_si, mean_p_w);
                ln_scint.cdf(threshold_w)
            }
            _ => {
                let gg = GammaGammaDistribution::from_scintillation_index(eff_si, mean_p_w);
                gg.outage_probability(threshold_w)
            }
        }
    }

    /// Average BER over the fading channel distribution.
    pub fn average_ber(&self, sensitivity_dbm: f64, modulation: FsoModulation) -> f64 {
        let eff_si = self.effective_scintillation_index();
        let mean_p_dbm = self.received_power_dbm(23.0);
        let snr_db = mean_p_dbm - sensitivity_dbm;

        // For OOK-IM: BER = 0.5 erfc(√(SNR/2)) in AWGN; average over fading
        match modulation {
            FsoModulation::OokIm => match self.atmospheric.regime() {
                TurbulenceRegime::Weak => {
                    let mean_p_w = 1e-3 * 10.0_f64.powf(mean_p_dbm / 10.0);
                    let ln_scint = LogNormalScintillation::new(eff_si, mean_p_w);
                    // Approximate: use penalty from scintillation
                    let penalty = ln_scint.mean_snr_penalty_db();
                    let effective_snr_db = snr_db - penalty;
                    let snr = 10.0_f64.powf(effective_snr_db / 10.0);
                    0.5 * erfc_approx((snr / 2.0).sqrt())
                }
                _ => {
                    let mean_p_w = 1e-3 * 10.0_f64.powf(mean_p_dbm / 10.0);
                    let gg = GammaGammaDistribution::from_scintillation_index(eff_si, mean_p_w);
                    gg.mean_ber(snr_db)
                }
            },
            FsoModulation::Bppm { m } => {
                // M-PPM BER in AWGN: BER ≈ (M/2) erfc(√(log2(M) * SNR / M))
                let m_f = m as f64;
                let snr = 10.0_f64.powf(snr_db / 10.0);
                let arg = (m_f.log2() * snr / m_f).sqrt();
                (m_f / 2.0) * erfc_approx(arg)
            }
            FsoModulation::DpQpsk => {
                // DP-QPSK BER: BER ≈ erfc(√SNR)
                let snr = 10.0_f64.powf(snr_db / 10.0);
                erfc_approx(snr.sqrt())
            }
        }
    }

    /// Maximum range in km for which link_margin ≥ min_margin_db.
    ///
    /// Binary searches over \[0.1 km, 10 000 km\].
    pub fn max_range_km(&self, min_margin_db: f64, visibility_km: f64) -> f64 {
        let mut lo = 0.1_f64;
        let mut hi = 10_000.0_f64;
        for _ in 0..60 {
            let mid = (lo + hi) / 2.0;
            let mut trial = self.clone();
            trial.link_distance_km = mid;
            trial.atmospheric.length_km = mid;
            let margin = trial.link_margin_db(
                self.tx_power_dbm - 60.0, // sensitivity = tx - 60 dB by default
                visibility_km,
            );
            if margin >= min_margin_db {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_link() -> FsoLink {
        FsoLink::new(1550e-9, 30.0, 0.1, 0.2, 1.0)
    }

    /// Geometric loss must be non-negative.
    #[test]
    fn test_geometric_loss_non_negative() {
        let link = default_link();
        assert!(link.geometric_loss_db() >= 0.0);
    }

    /// Received power must be less than transmitted power.
    #[test]
    fn test_received_power_less_than_tx() {
        let link = default_link();
        assert!(link.received_power_dbm(23.0) < link.tx_power_dbm);
    }

    /// Atmospheric loss increases with distance.
    #[test]
    fn test_atmospheric_loss_vs_distance() {
        let link1 = FsoLink::new(1550e-9, 30.0, 0.1, 0.2, 1.0);
        let link5 = FsoLink::new(1550e-9, 30.0, 0.1, 0.2, 5.0);
        assert!(link5.atmospheric_loss_db(10.0) > link1.atmospheric_loss_db(10.0));
    }

    /// Aperture averaging factor must be in (0, 1].
    #[test]
    fn test_aperture_averaging_factor_range() {
        let link = default_link();
        let a = link.aperture_averaging_factor();
        assert!(a > 0.0 && a <= 1.0, "A = {a:.4}");
    }

    /// Continental extinction at 1550 nm, 10 km visibility: low loss.
    #[test]
    fn test_continental_extinction_clear() {
        let ext = AtmosphericExtinction::new(10.0, 1550e-9, AerosolType::Continental);
        let loss = ext.loss_db_per_km();
        assert!(
            loss < 1.0,
            "Loss = {loss:.3} dB/km (expected < 1 in clear air)"
        );
    }

    /// Rain extinction increases with rain rate.
    #[test]
    fn test_rain_extinction() {
        let ext_light = AtmosphericExtinction::new(
            5.0,
            1550e-9,
            AerosolType::Rain {
                rate_mm_per_hr: 1.0,
            },
        );
        let ext_heavy = AtmosphericExtinction::new(
            5.0,
            1550e-9,
            AerosolType::Rain {
                rate_mm_per_hr: 50.0,
            },
        );
        assert!(
            ext_heavy.extinction_coefficient_per_km() > ext_light.extinction_coefficient_per_km()
        );
    }

    /// Diffraction-limited divergence for 10 cm aperture at 1550 nm ≈ 37.6 µrad.
    #[test]
    fn test_diffraction_limited_divergence() {
        let link = FsoLink::new(1550e-9, 30.0, 0.1, 0.2, 1.0);
        let div_rad = link.diffraction_limited_divergence_rad();
        // 2.44 * 1550e-9 / 0.1 = 37.78e-6 rad
        assert!((div_rad - 37.78e-6).abs() < 1e-7, "div = {div_rad:.4e} rad");
    }

    /// Link margin decreases with distance.
    #[test]
    fn test_margin_vs_distance() {
        let link_near = FsoLink::new(1550e-9, 30.0, 0.1, 0.2, 1.0);
        let link_far = FsoLink::new(1550e-9, 30.0, 0.1, 0.2, 5.0);
        let m_near = link_near.link_margin_db(-40.0, 23.0);
        let m_far = link_far.link_margin_db(-40.0, 23.0);
        assert!(m_near > m_far);
    }
}
