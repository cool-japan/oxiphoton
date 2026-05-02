//! Diffraction gratings: ruled, holographic, Echelle, volume Bragg, and Dammann gratings.
//!
//! All grating equation calculations follow the standard sign convention:
//!   n_in · sin(θ_i) + m · λ/Λ = n_out · sin(θ_m)
//! where θ is measured from the grating normal.
//!
//! References:
//! - Palmer, C. "Diffraction Grating Handbook" (Newport Corp., 7th ed., 2014)
//! - Kogelnik, H. "Coupled Wave Theory for Thick Hologram Gratings" (1969)

use crate::error::OxiPhotonError;
use std::f64::consts::PI;

// Speed of light (m/s) — kept for dimensional-analysis comments
#[allow(dead_code)]
const C0: f64 = 2.99792458e8;

// ---------------------------------------------------------------------------
// DiffractionGrating
// ---------------------------------------------------------------------------

/// Type of diffraction grating.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GratingType {
    /// Transmission grating (light passes through)
    Transmission,
    /// Reflection grating (light reflected from grooved surface)
    Reflection,
    /// Echelle grating — high-angle, high-order reflection for large dispersion
    Echelle,
    /// Volume Bragg grating (thick, index-modulated medium)
    Volume,
}

/// Ruled or holographic diffraction grating (amplitude or phase).
///
/// The grating equation (transmission):
///   n_in · sin θ_i + m · λ/Λ = n_out · sin θ_m
///
/// For a reflection grating in the same medium (n_in = n_out = n):
///   n · sin θ_i + m · λ/Λ = n · sin θ_m
#[derive(Debug, Clone)]
pub struct DiffractionGrating {
    /// Grating period Λ (μm)
    pub period_um: f64,
    /// Refractive index of the ambient medium (input side)
    pub n_ambient: f64,
    /// Refractive index of the substrate (output side for transmission)
    pub n_substrate: f64,
    /// Blaze angle (deg); 0 = sinusoidal/unblazed
    pub blaze_angle_deg: f64,
    /// Grating type
    pub grating_type: GratingType,
    /// Groove depth (nm)
    pub groove_depth_nm: f64,
}

impl DiffractionGrating {
    /// Create a new diffraction grating.
    ///
    /// # Errors
    /// - `InvalidLayer` if period_um ≤ 0 or groove_depth_nm < 0.
    /// - `InvalidRefractiveIndex` if any index ≤ 0.
    pub fn new(
        period_um: f64,
        n_medium: f64,
        blaze_deg: f64,
        grating_type: GratingType,
        groove_depth_nm: f64,
    ) -> Result<Self, OxiPhotonError> {
        if period_um <= 0.0 {
            return Err(OxiPhotonError::InvalidLayer(format!(
                "grating period must be > 0, got {period_um} μm"
            )));
        }
        if n_medium <= 0.0 {
            return Err(OxiPhotonError::InvalidRefractiveIndex {
                n: n_medium,
                k: 0.0,
            });
        }
        if groove_depth_nm < 0.0 {
            return Err(OxiPhotonError::InvalidLayer(format!(
                "groove depth must be ≥ 0, got {groove_depth_nm} nm"
            )));
        }
        Ok(Self {
            period_um,
            n_ambient: n_medium,
            n_substrate: n_medium,
            blaze_angle_deg: blaze_deg,
            grating_type,
            groove_depth_nm,
        })
    }

    /// Set the substrate refractive index (relevant for transmission gratings).
    pub fn with_substrate_index(mut self, n_sub: f64) -> Self {
        self.n_substrate = n_sub;
        self
    }

    // -----------------------------------------------------------------------
    // Core grating physics
    // -----------------------------------------------------------------------

    /// Diffraction angle for order `m` (rad).
    ///
    /// Grating equation: n_out · sin θ_m = n_in · sin θ_i + m · λ/Λ
    ///
    /// # Errors
    /// - `InvalidWavelength` if lambda_nm ≤ 0.
    /// - `NumericalError` if the argument of asin is outside [-1, 1] (evanescent).
    pub fn diffraction_angle_rad(
        &self,
        lambda_nm: f64,
        order: i32,
        incident_angle_rad: f64,
    ) -> Result<f64, OxiPhotonError> {
        if lambda_nm <= 0.0 {
            return Err(OxiPhotonError::InvalidWavelength(lambda_nm * 1e-9));
        }
        let lambda_um = lambda_nm * 1e-3;
        let n_in = self.n_ambient;
        let n_out = match self.grating_type {
            GratingType::Transmission => self.n_substrate,
            _ => self.n_ambient, // reflection: same medium
        };
        let sin_theta_m =
            (n_in * incident_angle_rad.sin() + order as f64 * lambda_um / self.period_um) / n_out;
        if sin_theta_m.abs() > 1.0 {
            return Err(OxiPhotonError::NumericalError(format!(
                "order m={order} is evanescent at λ={lambda_nm} nm (sin θ_m = {sin_theta_m:.4})"
            )));
        }
        Ok(sin_theta_m.asin())
    }

    /// All propagating (non-evanescent) diffraction orders for the given wavelength.
    pub fn propagating_orders(&self, lambda_nm: f64, incident_angle_rad: f64) -> Vec<i32> {
        let lambda_um = lambda_nm * 1e-3;
        // Maximum possible |sin| is 1, so |m| ≤ Λ*(n_in*(1+|sin θ_i|))/λ
        let max_m = ((self.period_um * (self.n_ambient + 1.0) / lambda_um).ceil() as i32) + 1;
        let mut orders = Vec::new();
        for m in -max_m..=max_m {
            if self
                .diffraction_angle_rad(lambda_nm, m, incident_angle_rad)
                .is_ok()
            {
                orders.push(m);
            }
        }
        orders
    }

    // -----------------------------------------------------------------------
    // Efficiency (scalar diffraction theory)
    // -----------------------------------------------------------------------

    /// Blaze efficiency for order `m` using scalar diffraction theory.
    ///
    /// For a blazed grating:
    ///   η_m = sinc²(m - Λ sin θ_B / λ)
    /// where sinc(x) = sin(πx)/(πx).
    pub fn blaze_efficiency(&self, lambda_nm: f64, order: i32, incident_angle_rad: f64) -> f64 {
        let blaze_rad = self.blaze_angle_deg.to_radians();
        let lambda_um = lambda_nm * 1e-3;
        // Diffraction angle at design
        let sin_theta_d =
            self.n_ambient * incident_angle_rad.sin() + order as f64 * lambda_um / self.period_um;
        // Phase mismatch argument: β = m - (Λ/λ)·(sin θ_i + sin θ_B)
        // Blazed grating optimum: Λ(sin θ_i + sin θ_d) = 2 Λ sin θ_B (Littrow blaze)
        // Simplified scalar theory efficiency
        let blaze_lambda_um = self.period_um * (incident_angle_rad.sin() + blaze_rad.sin());
        let x = order as f64 - blaze_lambda_um / lambda_um;
        // sinc² function
        let eta = sinc_sq(x);
        // If the grating is unblazed (blaze_angle = 0), zero-order gets all power
        if self.blaze_angle_deg.abs() < 1e-9 && order == 0 {
            return 1.0;
        }
        // Modulate by groove shadow factor for reflection gratings
        let shadow = match self.grating_type {
            GratingType::Reflection | GratingType::Echelle => {
                // Geometric shadowing reduces efficiency for large blaze angles
                let cos_b = blaze_rad.cos().max(1e-9);
                let cos_d = (sin_theta_d.clamp(-1.0, 1.0)).asin().cos().abs();
                (cos_d / cos_b).clamp(0.0, 1.0)
            }
            _ => 1.0,
        };
        eta * shadow
    }

    /// Wavelength of maximum blaze efficiency (nm).
    ///
    /// For Littrow mounting: λ_blaze = 2 Λ sin θ_B / m
    /// For general: λ_blaze = Λ (sin θ_i + sin θ_B) / m
    pub fn blaze_wavelength_nm(&self, incident_angle_rad: f64) -> f64 {
        let blaze_rad = self.blaze_angle_deg.to_radians();
        // First order (m=1) blaze wavelength
        let lambda_um = self.period_um * (incident_angle_rad.sin() + blaze_rad.sin());
        lambda_um * 1e3 // nm
    }

    /// Littrow mounting incident angle for retroreflection (rad).
    ///
    /// In Littrow mounting the diffracted beam of order `m` exactly retraces the
    /// incident beam.  In our grating-equation sign convention
    ///   sin θ_m = sin θ_i + m·λ/Λ
    /// retroreflection requires sin θ_m = -sin θ_i (opposite side of normal), giving
    ///   θ_i = -arcsin(m·λ / (2·Λ·n))
    ///
    /// The returned angle is negative for positive orders (incident beam on the
    /// left of the grating normal) and vice-versa.
    /// Returns NaN if the condition is evanescent (|m·λ/(2·Λ·n)| > 1).
    pub fn littrow_angle_rad(&self, lambda_nm: f64, order: i32) -> f64 {
        let lambda_um = lambda_nm * 1e-3;
        let arg = order as f64 * lambda_um / (2.0 * self.period_um * self.n_ambient);
        if arg.abs() > 1.0 {
            return f64::NAN;
        }
        // Negate so that the diffracted angle equals -θ_i (retroreflection)
        -arg.asin()
    }

    /// Magnitude of the Littrow angle (rad) — always non-negative.
    ///
    /// Equivalent to arcsin(|m|·λ / (2·Λ·n)).  Use this when only the physical
    /// tilt of the grating is needed (e.g., for mechanical design).
    pub fn littrow_angle_magnitude_rad(&self, lambda_nm: f64, order: i32) -> f64 {
        self.littrow_angle_rad(lambda_nm, order).abs()
    }

    // -----------------------------------------------------------------------
    // Dispersion and resolution
    // -----------------------------------------------------------------------

    /// Angular dispersion dθ/dλ (rad/nm) for diffraction order `m`.
    ///
    /// dθ_m/dλ = m / (Λ · cos θ_m · n_out)
    ///
    /// Returns 0.0 if the order is evanescent.
    pub fn angular_dispersion_rad_per_nm(
        &self,
        lambda_nm: f64,
        order: i32,
        incident_angle_rad: f64,
    ) -> f64 {
        match self.diffraction_angle_rad(lambda_nm, order, incident_angle_rad) {
            Err(_) => 0.0,
            Ok(theta_m) => {
                let _lambda_um = lambda_nm * 1e-3;
                let n_out = match self.grating_type {
                    GratingType::Transmission => self.n_substrate,
                    _ => self.n_ambient,
                };
                // dθ/dλ [rad/nm]: convert period from μm→nm (×1e3) and factor n_out
                let cos_theta = theta_m.cos().abs().max(1e-12);
                order as f64 / (self.period_um * 1e3 * cos_theta * n_out)
            }
        }
    }

    /// Linear dispersion dx/dλ = f · dθ/dλ (mm/nm) for camera focal length `f` (mm).
    pub fn linear_dispersion_mm_per_nm(
        &self,
        lambda_nm: f64,
        order: i32,
        focal_length_mm: f64,
        incident_angle_rad: f64,
    ) -> f64 {
        self.angular_dispersion_rad_per_nm(lambda_nm, order, incident_angle_rad) * focal_length_mm
    }

    /// Resolving power R = m · N (dimensionless).
    ///
    /// N is the total number of illuminated grooves.
    pub fn resolving_power(&self, order: i32, n_grooves: usize) -> f64 {
        (order.abs() as f64) * n_grooves as f64
    }

    /// Minimum resolvable wavelength difference δλ = λ/R (nm).
    pub fn resolution_nm(&self, lambda_nm: f64, order: i32, n_grooves: usize) -> f64 {
        let r = self.resolving_power(order, n_grooves);
        if r < 1e-30 {
            return f64::INFINITY;
        }
        lambda_nm / r
    }

    /// Free spectral range FSR = λ / |m| (nm).
    ///
    /// Spectral range before orders m and m+1 overlap.
    pub fn free_spectral_range_nm(&self, lambda_nm: f64, order: i32) -> f64 {
        if order == 0 {
            return f64::INFINITY;
        }
        lambda_nm / (order.abs() as f64)
    }

    /// Useful bandwidth Δλ = λ_center / |m| (nm) — alias for FSR for consistency.
    pub fn useful_bandwidth_nm(&self, lambda_center_nm: f64, order: i32) -> f64 {
        self.free_spectral_range_nm(lambda_center_nm, order)
    }

    // -----------------------------------------------------------------------
    // Anomalies
    // -----------------------------------------------------------------------

    /// Rayleigh / Wood's anomaly wavelengths (nm).
    ///
    /// Occur when a diffraction order becomes grazing (sin θ = ±1):
    ///   λ_W = Λ · (n ± sin θ_i) / |m|
    ///
    /// Returns wavelengths for orders m = ±1 in both + and − families.
    pub fn woods_anomaly_wavelength_nm(&self, incident_angle_rad: f64) -> Vec<f64> {
        let sin_i = incident_angle_rad.sin();
        let mut anomalies = Vec::new();
        // For each integer order m ≠ 0, anomaly when sin θ_m = ±1
        // λ = Λ·(n·(±1) - n·sin θ_i) / m  = Λ·n·(±1 - sin θ_i)/m
        for m in [-3i32, -2, -1, 1, 2, 3] {
            for sign in [1.0_f64, -1.0] {
                let lambda_um = self.period_um * self.n_ambient * (sign - sin_i) / (m as f64);
                if lambda_um > 0.0 {
                    anomalies.push(lambda_um * 1e3); // nm
                }
            }
        }
        anomalies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        anomalies
    }
}

// ---------------------------------------------------------------------------
// HolographicGrating
// ---------------------------------------------------------------------------

/// Sinusoidal holographic grating (surface or volume type).
///
/// Characterized by the Raman-Nath parameter Q:
///   Q = 2π·λ·d / (n·Λ²)
/// - Q < 1: thin grating (Raman-Nath regime)
/// - Q > 10: thick grating (Bragg regime)
#[derive(Debug, Clone)]
pub struct HolographicGrating {
    /// Grating period Λ (μm)
    pub period_um: f64,
    /// Modulation depth: Δn for volume hologram, or h/Λ (groove depth/period) for surface
    pub modulation_depth: f64,
    /// Average refractive index of the medium
    pub n_medium: f64,
}

impl HolographicGrating {
    /// Create a holographic grating.
    pub fn new(period_um: f64, modulation: f64, n_medium: f64) -> Self {
        Self {
            period_um,
            modulation_depth: modulation,
            n_medium,
        }
    }

    /// First-order diffraction efficiency in the thin (Raman-Nath) regime.
    ///
    /// η₁ = J₁²(π·modulation) ≈ (π·modulation/2)² for small modulation.
    ///
    /// For surface gratings, uses sinusoidal scalar theory:
    ///   η₁ = (π·h/λ·cos θ)² in scalar limit
    /// Simplified here as η₁ = (π·Δn·d/λ)² / 4 (first-order Bessel approx).
    pub fn first_order_efficiency_thin(&self, lambda_nm: f64) -> f64 {
        let lambda_um = lambda_nm * 1e-3;
        // Phase modulation amplitude: φ = π·Δn·Λ/λ (surface depth ~ period)
        let phi = PI * self.modulation_depth * self.period_um / lambda_um;
        // η₁ = J₁²(φ); Bessel J₁(x) ≈ x/2 for x << 1
        // Use approximation valid for moderate φ:
        bessel_j1_sq(phi)
    }

    /// Raman-Nath parameter Q = 2π·λ·d / (n·Λ²).
    ///
    /// Q < 1: thin (multi-order) regime.
    /// Q > 10: thick (Bragg) regime.
    pub fn raman_nath_parameter(&self, lambda_nm: f64, thickness_um: f64) -> f64 {
        let lambda_um = lambda_nm * 1e-3;
        2.0 * PI * lambda_um * thickness_um / (self.n_medium * self.period_um * self.period_um)
    }

    /// Returns true if the Bragg condition is met for the given wavelength and incidence angle.
    ///
    /// Bragg condition: 2 · n · Λ · sin θ_B = λ
    pub fn bragg_condition_met(&self, lambda_nm: f64, angle_rad: f64) -> bool {
        let lambda_um = lambda_nm * 1e-3;
        let bragg_lambda = 2.0 * self.n_medium * self.period_um * angle_rad.sin();
        (bragg_lambda - lambda_um).abs() < 0.05 * lambda_um // within 5%
    }

    /// Diffraction efficiency for a thick (Bragg-regime) hologram using Kogelnik coupled-wave theory.
    ///
    /// For lossless reflection hologram at Bragg resonance:
    ///   η = sin²(π · Δn · d / (λ · cos θ))
    ///
    /// For transmission hologram:
    ///   η = sin²(π · Δn · d / (λ · cos θ_B))
    pub fn diffraction_efficiency_thick(&self, lambda_nm: f64, thickness_um: f64) -> f64 {
        let lambda_um = lambda_nm * 1e-3;
        // Assume normal incidence at Bragg angle for simplicity
        let bragg_angle = (lambda_um / (2.0 * self.n_medium * self.period_um))
            .clamp(-1.0, 1.0)
            .asin();
        let cos_theta = bragg_angle.cos().max(1e-12);
        let nu = PI * self.modulation_depth * thickness_um / (lambda_um * cos_theta);
        nu.sin().powi(2)
    }
}

// ---------------------------------------------------------------------------
// GratingSpectrometer
// ---------------------------------------------------------------------------

/// Grating-based spectrometer design and performance estimation.
///
/// Models a spectrometer with:
/// - Collimating + camera optics (focal length f)
/// - Diffraction grating (ruled or holographic)
/// - Linear detector array (CCD/InGaAs)
#[derive(Debug, Clone)]
pub struct GratingSpectrometer {
    /// Diffraction grating
    pub grating: DiffractionGrating,
    /// Camera (focusing) lens focal length (mm)
    pub focal_length_mm: f64,
    /// Detector active width (mm)
    pub detector_width_mm: f64,
    /// Number of detector pixels
    pub n_pixels: usize,
    /// Diffraction order used
    pub order: i32,
    /// Center wavelength (nm)
    pub center_wavelength_nm: f64,
    /// Angle of incidence on grating (rad)
    pub incident_angle_rad: f64,
}

impl GratingSpectrometer {
    /// Create a new spectrometer design.
    pub fn new(
        grating: DiffractionGrating,
        focal_mm: f64,
        det_width_mm: f64,
        n_pixels: usize,
        order: i32,
        center_nm: f64,
        angle_rad: f64,
    ) -> Self {
        Self {
            grating,
            focal_length_mm: focal_mm,
            detector_width_mm: det_width_mm,
            n_pixels,
            order,
            center_wavelength_nm: center_nm,
            incident_angle_rad: angle_rad,
        }
    }

    /// Wavelength range at detector edges: (λ_min, λ_max) in nm.
    ///
    /// Δλ = detector_width / linear_dispersion
    pub fn wavelength_range_nm(&self) -> (f64, f64) {
        let disp = self.grating.linear_dispersion_mm_per_nm(
            self.center_wavelength_nm,
            self.order,
            self.focal_length_mm,
            self.incident_angle_rad,
        );
        if disp.abs() < 1e-30 {
            return (self.center_wavelength_nm, self.center_wavelength_nm);
        }
        let half_range = self.detector_width_mm / (2.0 * disp.abs());
        (
            self.center_wavelength_nm - half_range,
            self.center_wavelength_nm + half_range,
        )
    }

    /// Dispersion in nm per pixel.
    pub fn nm_per_pixel(&self) -> f64 {
        if self.n_pixels == 0 {
            return 0.0;
        }
        let (lmin, lmax) = self.wavelength_range_nm();
        (lmax - lmin) / self.n_pixels as f64
    }

    /// Pixel-size-limited spectral resolution δλ = pixel_size × (dλ/dx) (nm).
    pub fn pixel_limited_resolution_nm(&self) -> f64 {
        self.nm_per_pixel()
    }

    /// Étendue A·Ω = (beam_width × grating_width) × (beam_solid_angle) (mm² · sr).
    ///
    /// Approximated as A_beam × λ/D for a diffraction-limited beam.
    pub fn etendue_mm2_sr(&self, beam_width_mm: f64) -> f64 {
        // Solid angle of the collimated beam ≈ (beam_width/focal_length)²
        let omega = (beam_width_mm / self.focal_length_mm).powi(2);
        beam_width_mm * beam_width_mm * omega
    }

    /// Wavelength axis: returns the center wavelength of each pixel (nm).
    pub fn wavelength_axis_nm(&self) -> Vec<f64> {
        if self.n_pixels == 0 {
            return Vec::new();
        }
        let (lmin, lmax) = self.wavelength_range_nm();
        let dlambda = (lmax - lmin) / self.n_pixels as f64;
        (0..self.n_pixels)
            .map(|i| lmin + (i as f64 + 0.5) * dlambda)
            .collect()
    }

    /// Stray light suppression ratio (estimated, dB).
    ///
    /// Typical ruled grating: −30 to −40 dB; holographic: −40 to −60 dB.
    /// This model returns −40 dB as a conservative estimate for high-quality gratings.
    pub fn stray_light_db(&self) -> f64 {
        // Reflection/Echelle gratings have somewhat more stray light than transmission
        match self.grating.grating_type {
            GratingType::Transmission => -45.0,
            GratingType::Volume => -55.0,
            _ => -38.0,
        }
    }
}

// ---------------------------------------------------------------------------
// VolumeBraggGrating
// ---------------------------------------------------------------------------

/// Narrowband volume Bragg grating (VBG) wavelength filter / beam combiner.
///
/// The peak reflectivity and bandwidth are derived from Kogelnik's coupled-wave theory:
///   R_peak = tanh²(κ · d)   where κ = π · Δn / λ
///   δλ = λ² / (Δn · d)      (FWHM bandwidth, approximate)
#[derive(Debug, Clone)]
pub struct VolumeBraggGrating {
    /// Bragg (design) wavelength λ_B (nm)
    pub bragg_wavelength_nm: f64,
    /// Average refractive index of the recording medium
    pub refractive_index: f64,
    /// Peak refractive-index modulation amplitude Δn
    pub delta_n: f64,
    /// Grating physical thickness d (mm)
    pub thickness_mm: f64,
    /// 1/e² beam diameter (mm) — used for diffraction-limited estimates
    pub beam_diameter_mm: f64,
}

impl VolumeBraggGrating {
    /// Create a new volume Bragg grating.
    pub fn new(
        lambda_b_nm: f64,
        n_avg: f64,
        delta_n: f64,
        thickness_mm: f64,
        beam_mm: f64,
    ) -> Self {
        Self {
            bragg_wavelength_nm: lambda_b_nm,
            refractive_index: n_avg,
            delta_n,
            thickness_mm,
            beam_diameter_mm: beam_mm,
        }
    }

    /// Peak reflectivity R = tanh²(π · Δn · d / λ_B).
    ///
    /// Returns a value in [0, 1].
    pub fn peak_reflectivity(&self) -> f64 {
        let lambda_um = self.bragg_wavelength_nm * 1e-3;
        let thickness_um = self.thickness_mm * 1e3;
        let kappa_d = PI * self.delta_n * thickness_um / lambda_um;
        kappa_d.tanh().powi(2)
    }

    /// Null-to-null reflection bandwidth δλ = λ_B² / (Δn · d) (nm).
    ///
    /// This is the spectral distance between the first zeros of the reflectivity.
    pub fn bandwidth_nm(&self) -> f64 {
        let thickness_um = self.thickness_mm * 1e3;
        let lambda_sq = self.bragg_wavelength_nm * self.bragg_wavelength_nm;
        lambda_sq / (self.delta_n * thickness_um * 1e3)
    }

    /// Angular bandwidth in mrad.
    ///
    /// δθ ≈ λ / (n · Λ_z) where Λ_z = d (grating thickness in propagation direction).
    /// More precisely: δθ_mrad ≈ λ_B / (n · d · tan θ_B) × 1e3
    pub fn angular_bandwidth_mrad(&self) -> f64 {
        let lambda_um = self.bragg_wavelength_nm * 1e-3;
        let thickness_um = self.thickness_mm * 1e3;
        // Period of index modulation from Bragg condition: Λ = λ/(2n)
        let period_um = lambda_um / (2.0 * self.refractive_index);
        let bragg_angle = (lambda_um / (2.0 * self.refractive_index * period_um))
            .clamp(-1.0, 1.0)
            .asin();
        let tan_b = bragg_angle.tan().abs().max(1e-12);
        // δθ = λ / (n · d · tan θ_B)
        let delta_theta_rad = lambda_um / (self.refractive_index * thickness_um * tan_b);
        delta_theta_rad * 1e3 // mrad
    }

    /// Reflection spectrum R(λ) over [lambda_min, lambda_max] with `n` points.
    ///
    /// Uses Kogelnik sinc-like formula:
    ///   R(Δλ) = tanh²(√(κ²d² - (π Δn d δ/λ²)²)) / (1 + (δ/κ·tanh...)²)
    /// Simplified to:
    ///   R(λ) ≈ tanh²(κ·d) · sinc²(Δλ / δλ_null)
    ///
    /// where δλ_null = λ²/(Δn·d).
    pub fn reflection_spectrum(
        &self,
        lambda_min_nm: f64,
        lambda_max_nm: f64,
        n: usize,
    ) -> Vec<(f64, f64)> {
        if n == 0 {
            return Vec::new();
        }
        let r_peak = self.peak_reflectivity();
        let bw_nm = self.bandwidth_nm();
        let dlambda = if n > 1 {
            (lambda_max_nm - lambda_min_nm) / (n - 1) as f64
        } else {
            0.0
        };
        (0..n)
            .map(|i| {
                let lam = lambda_min_nm + i as f64 * dlambda;
                let delta_lam = lam - self.bragg_wavelength_nm;
                let x = delta_lam / bw_nm;
                let reflectivity = r_peak * sinc_sq(x);
                (lam, reflectivity.clamp(0.0, 1.0))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// DammannGrating
// ---------------------------------------------------------------------------

/// Dammann grating: binary phase grating for uniform multi-spot beam splitting.
///
/// Designed for splitting a single beam into N equally spaced, equally intense spots.
/// Transition points (phase-flip positions within one period `[0,1]`) are found by a
/// Levenberg–Marquardt optimiser that minimises diffraction-order intensity variance.
///
/// The optimiser enforces half-wave antisymmetry (f(x+0.5) = −f(x)), which
/// guarantees that all even-order Fourier coefficients vanish.  It operates on
/// K = `n_orders` free parameters in [0, 0.5) and mirrors them to form the full
/// 2K-transition set stored in `optimised_transitions`.
///
/// After calling [`optimize_transitions`](DammannGrating::optimize_transitions),
/// [`efficiency`](DammannGrating::efficiency) and
/// [`uniformity`](DammannGrating::uniformity) are based on Fourier coefficients
/// of the optimised profile.  Before optimisation they fall back to the hard-coded
/// analytical table for N ≤ 5.
#[derive(Debug, Clone)]
pub struct DammannGrating {
    /// Grating period Λ (μm)
    pub period_um: f64,
    /// Target number of output spots (e.g., 5 for 5 spots in one dimension)
    pub n_spots: usize,
    /// Design wavelength (nm)
    pub wavelength_nm: f64,
    /// Optimised full-period transition points from `optimize_transitions()`.
    /// `None` until the LM optimiser has been run.
    optimised_transitions: Option<Vec<f64>>,
    /// Number of orders used in the last `optimize_transitions()` call.
    /// Stored so that `efficiency()` / `uniformity()` can refer to the same
    /// set of orders without requiring a parameter.
    optimised_n_orders: usize,
}

impl DammannGrating {
    /// Create a new Dammann grating.
    pub fn new(period_um: f64, n_spots: usize, lambda_nm: f64) -> Self {
        Self {
            period_um,
            n_spots: n_spots.max(1),
            wavelength_nm: lambda_nm,
            optimised_transitions: None,
            optimised_n_orders: 0,
        }
    }

    /// Normalised transition points within one period [0, 1].
    ///
    /// Returns the LM-optimised transitions if `optimize_transitions()` has
    /// been called, otherwise the hard-coded analytical solutions for N ≤ 5
    /// or a uniform-spacing approximation for larger N.
    ///
    /// Reference: Dammann & Görtler, Opt. Commun. 3 (1971) 312.
    pub fn transition_points(&self) -> Vec<f64> {
        if let Some(ref t) = self.optimised_transitions {
            return t.clone();
        }
        self.table_transition_points()
    }

    /// Hard-coded / analytical table of transition points.
    fn table_transition_points(&self) -> Vec<f64> {
        match self.n_spots {
            1 => vec![],
            2 => vec![0.5],
            3 => vec![1.0 / 6.0, 1.0 / 2.0, 5.0 / 6.0],
            4 => vec![0.125, 0.375, 0.625, 0.875],
            5 => {
                // Optimized for 5-spot: approximate known solution
                vec![0.1, 0.35, 0.5, 0.65, 0.9]
            }
            n => {
                // Uniform approximation: transition every 1/n
                (1..n).map(|k| k as f64 / n as f64).collect()
            }
        }
    }

    /// Return a reference to the optimised transition points, if available.
    pub fn get_optimised_transitions(&self) -> Option<&[f64]> {
        self.optimised_transitions.as_deref()
    }

    // -----------------------------------------------------------------------
    // Fourier-coefficient machinery
    // -----------------------------------------------------------------------

    /// Compute the complex Fourier coefficients c_0, c_1, …, c_{m_max} of the
    /// binary ±1 grating profile defined by `transitions`.
    ///
    /// The profile is +1 on [0, transitions[0]), −1 on [transitions[0], transitions[1]),
    /// and so on, alternating sign at each transition.
    ///
    /// The coefficients are computed by direct integration of each strip:
    ///
    ///   c_m = Σ_i sign_i · ∫_{a_i}^{b_i} e^{−j2πmx} dx
    ///       = Σ_i sign_i · (e^{−j2πm a_i} − e^{−j2πm b_i}) / (j2πm)
    ///
    /// DC (m = 0): c_0 = Σ_i sign_i · (b_i − a_i).
    fn fourier_coefficients_static(
        transitions: &[f64],
        m_max: usize,
    ) -> Vec<num_complex::Complex64> {
        // Build strip boundary array: [0, t_1, t_2, …, t_K, 1]
        // The transitions are assumed sorted.
        let boundaries: Vec<f64> = {
            let mut b = Vec::with_capacity(transitions.len() + 2);
            b.push(0.0_f64);
            b.extend_from_slice(transitions);
            b.push(1.0_f64);
            b
        };
        let n_strips = boundaries.len() - 1;

        let mut coeffs = Vec::with_capacity(m_max + 1);

        // DC coefficient: c_0 = integral of the profile = Σ sign_i * width_i
        let c0_real: f64 = (0..n_strips)
            .map(|i| {
                let sign = if i % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
                sign * (boundaries[i + 1] - boundaries[i])
            })
            .sum();
        coeffs.push(num_complex::Complex64::new(c0_real, 0.0));

        // Non-zero orders m = 1 … m_max.
        for m in 1..=m_max {
            let mf = m as f64;
            let two_pi_m = 2.0 * PI * mf;
            let mut c_re = 0.0_f64;
            let mut c_im = 0.0_f64;

            for i in 0..n_strips {
                let sign = if i % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
                let angle_a = two_pi_m * boundaries[i];
                let angle_b = two_pi_m * boundaries[i + 1];
                // ∫_a^b e^{-j2πmx} dx = (e^{-ja} - e^{-jb}) / (j*2πm)
                // numerator: (cos a - cos b) - j*(sin a - sin b)
                let num_re = angle_a.cos() - angle_b.cos();
                let num_im = -(angle_a.sin() - angle_b.sin());
                // divide by j*two_pi_m: (re + j*im)/(j*d) = im/d - j*re/d
                c_re += sign * (num_im / two_pi_m);
                c_im += sign * (-num_re / two_pi_m);
            }
            coeffs.push(num_complex::Complex64::new(c_re, c_im));
        }
        coeffs
    }

    /// Expand K half-period transition points to the full 2K-point set by
    /// mirroring: [t_1,…,t_K] → [t_1,…,t_K, t_1+0.5,…,t_K+0.5] (sorted).
    ///
    /// This enforces half-wave antisymmetry f(x+0.5) = −f(x), which causes
    /// all even-order Fourier coefficients to vanish exactly.
    fn half_to_full_transitions(half: &[f64]) -> Vec<f64> {
        let mut full = Vec::with_capacity(2 * half.len());
        for &t in half {
            full.push(t);
        }
        for &t in half {
            full.push(t + 0.5);
        }
        full.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        full
    }

    /// Compute the complex Fourier coefficients c_0, c_1, …, c_{m_max} of the
    /// grating's binary phase profile.
    ///
    /// Uses optimised transition points if available, otherwise falls back to
    /// the hard-coded table.
    pub fn fourier_coefficients(&self, m_max: usize) -> Vec<num_complex::Complex64> {
        let tp = self.transition_points();
        Self::fourier_coefficients_static(&tp, m_max)
    }

    // -----------------------------------------------------------------------
    // LM optimiser internals
    // -----------------------------------------------------------------------

    /// Cost function and gradient for the *half-period* transition vector `half`.
    ///
    /// Internally expands to a full 2K-transition set (half-wave symmetric).
    /// The cost targets the first `n_orders` *odd* diffraction orders
    /// (m = 1, 3, 5, …, 2·n_orders−1), minimising their intensity variance:
    ///
    ///   J = (1/N) · Σ_{k=1..N} (|c_{2k−1}|² − target)²  +  10³·Σ mono_violations²
    ///
    /// where target = Σ|c_{2k−1}|² / N.
    ///
    /// Gradient computed with 5-point central-difference stencil (h = 1e-6).
    fn cost_and_grad_half(half: &[f64], n_orders: usize) -> (f64, Vec<f64>) {
        let m_max_odd = 2 * n_orders - 1; // largest odd order we care about

        let cost = |h_pts: &[f64]| -> f64 {
            let full = Self::half_to_full_transitions(h_pts);
            let coeffs = Self::fourier_coefficients_static(&full, m_max_odd);
            // Power in each target odd order m = 1, 3, ..., 2*n-1
            let powers: Vec<f64> = (0..n_orders)
                .map(|k| coeffs[2 * k + 1].norm_sqr())
                .collect();
            let target = powers.iter().sum::<f64>() / n_orders as f64;
            let variance =
                powers.iter().map(|&p| (p - target).powi(2)).sum::<f64>() / n_orders as f64;
            // Soft monotonicity penalty in half-period domain
            let mono_pen: f64 = h_pts
                .windows(2)
                .map(|w| (w[0] - w[1]).max(0.0).powi(2))
                .sum();
            variance + 1e3 * mono_pen
        };

        let j0 = cost(half);
        let h = 1e-6_f64;
        let mut grad = vec![0.0_f64; half.len()];
        let mut h_pert = half.to_vec();
        for i in 0..half.len() {
            h_pert[i] = half[i] + 2.0 * h;
            let jp2 = cost(&h_pert);
            h_pert[i] = half[i] + h;
            let jp1 = cost(&h_pert);
            h_pert[i] = half[i] - h;
            let jm1 = cost(&h_pert);
            h_pert[i] = half[i] - 2.0 * h;
            let jm2 = cost(&h_pert);
            h_pert[i] = half[i]; // restore
            grad[i] = (-jp2 + 8.0 * jp1 - 8.0 * jm1 + jm2) / (12.0 * h);
        }
        (j0, grad)
    }

    /// Run a Levenberg–Marquardt optimiser to find transition points that
    /// uniformly distribute power across `n_orders` diffraction orders.
    ///
    /// The optimiser works on K = `n_orders` *half-period* free parameters in
    /// [0, 0.5) and mirrors them to produce the full 2K-transition set.  This
    /// enforces half-wave antisymmetry, making all even-order coefficients zero.
    ///
    /// A 5-restart multi-start strategy is used to reduce the risk of local
    /// minima.  The optimised transitions are stored and used by
    /// [`efficiency`](DammannGrating::efficiency),
    /// [`uniformity`](DammannGrating::uniformity), and
    /// [`transition_points`](DammannGrating::transition_points).
    ///
    /// Returns the final cost J (0 = perfect uniform splitter).
    ///
    /// # Errors
    /// Returns [`OxiPhotonError::Convergence`] if `n_orders` is 0.
    pub fn optimize_transitions(
        &mut self,
        n_orders: usize,
    ) -> Result<f64, crate::error::OxiPhotonError> {
        if n_orders == 0 {
            return Err(crate::error::OxiPhotonError::Convergence(
                "n_orders must be ≥ 1".to_owned(),
            ));
        }

        let k = n_orders; // free parameters in [0, 0.5)
        let default_seed: Vec<f64> = (1..=k).map(|i| i as f64 / (2 * k + 2) as f64).collect();

        let mut best_half = default_seed.clone();
        let mut best_j = f64::INFINITY;

        for restart in 0..5_usize {
            let mut x: Vec<f64> = if restart == 0 {
                default_seed.clone()
            } else {
                // Deterministic perturbation without external RNG.
                let seed = restart as f64 * 0.09;
                (1..=k)
                    .map(|i| {
                        let base = i as f64 / (2 * k + 2) as f64;
                        (base + seed * (i as f64 * 1.618).sin()).clamp(0.001, 0.498)
                    })
                    .collect()
            };
            // Ensure initial point is sorted (monotone).
            x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let mut mu = 1e-2_f64; // LM damping parameter

            for _iter in 0..300_usize {
                let (j, grad) = Self::cost_and_grad_half(&x, n_orders);
                let gnorm = grad.iter().map(|g| g.abs()).fold(0.0_f64, f64::max);
                if gnorm < 1e-12 {
                    break;
                }

                // Gradient-descent step with LM damping.
                let x_new: Vec<f64> = x
                    .iter()
                    .zip(grad.iter())
                    .map(|(&xi, &gi)| (xi - gi / (1.0 + mu)).clamp(0.001, 0.498))
                    .collect();
                let (j_new, _) = Self::cost_and_grad_half(&x_new, n_orders);
                if j_new < j {
                    x = x_new;
                    // Keep x sorted to maintain monotonicity in the half domain.
                    x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    mu *= 0.5_f64.max(1e-12);
                } else {
                    mu *= 2.0;
                    if mu > 1e12 {
                        break;
                    }
                }
            }

            let (j_final, _) = Self::cost_and_grad_half(&x, n_orders);
            if j_final < best_j {
                best_j = j_final;
                best_half = x;
            }
        }

        // Expand half-period to full period and store.
        let full = Self::half_to_full_transitions(&best_half);
        self.optimised_transitions = Some(full);
        self.optimised_n_orders = n_orders;
        Ok(best_j)
    }

    // -----------------------------------------------------------------------
    // Metrics
    // -----------------------------------------------------------------------

    /// Total diffraction efficiency: fraction of input power in all non-DC
    /// diffraction orders (both positive and negative).
    ///
    /// For a real grating c_{−m} = c_m*, so |c_{−m}|² = |c_m|².  Efficiency
    /// is therefore 2·Σ_{m=1..N}|c_m|² where N = `optimised_n_orders` (when
    /// optimised) or `n_spots` (otherwise).
    ///
    /// Uses optimised transitions when available.
    pub fn efficiency(&self) -> f64 {
        let tp = self.transition_points();
        let n = if self.optimised_transitions.is_some() {
            self.optimised_n_orders.max(1)
        } else {
            self.n_spots
        };
        // Use enough harmonics to capture all target orders.
        let m_max = if self.optimised_transitions.is_some() {
            2 * n - 1 // highest odd order for half-wave symmetric design
        } else {
            n
        };
        let coeffs = Self::fourier_coefficients_static(&tp, m_max);
        // Bilateral: include both +m and −m orders.
        2.0 * coeffs[1..].iter().map(|c| c.norm_sqr()).sum::<f64>()
    }

    /// Uniformity metric: min(|c_m|²) / max(|c_m|²) over the target non-DC
    /// diffraction orders.
    ///
    /// Returns 1.0 (perfect) when all target-order Fourier powers are equal,
    /// and 0.0 when max power is negligible.
    ///
    /// For the optimised half-wave design, only odd orders m = 1, 3, …,
    /// 2·n_orders−1 are considered (even orders are identically zero).
    ///
    /// Uses optimised transitions when available.
    pub fn uniformity(&self) -> f64 {
        let tp = self.transition_points();
        let n = if self.optimised_transitions.is_some() {
            self.optimised_n_orders.max(1)
        } else {
            self.n_spots
        };
        let m_max = if self.optimised_transitions.is_some() {
            2 * n - 1
        } else {
            n
        };
        let coeffs = Self::fourier_coefficients_static(&tp, m_max);
        // For optimised design, compare only odd orders (non-zero by construction).
        let powers: Vec<f64> = if self.optimised_transitions.is_some() {
            (0..n).map(|k| coeffs[2 * k + 1].norm_sqr()).collect()
        } else {
            coeffs[1..].iter().map(|c| c.norm_sqr()).collect()
        };
        if powers.is_empty() {
            return 1.0;
        }
        let min = powers.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = powers.iter().cloned().fold(0.0_f64, f64::max);
        if max < 1e-30 {
            0.0
        } else {
            min / max
        }
    }

    /// Non-uniformity metric: 1 − uniformity().
    ///
    /// Returns 0.0 for a perfect grating and approaches 1.0 for a highly
    /// non-uniform design.
    pub fn nonuniformity_metric(&self, _n_orders: usize) -> f64 {
        1.0 - self.uniformity()
    }

    /// Diffraction angle for spot index (signed order m) (rad).
    ///
    /// θ_m = arcsin(m · λ / Λ) for a 1D grating in air.
    pub fn spot_angle_rad(&self, spot_idx: i32) -> f64 {
        let lambda_um = self.wavelength_nm * 1e-3;
        let arg = spot_idx as f64 * lambda_um / self.period_um;
        if arg.abs() > 1.0 {
            return f64::NAN;
        }
        arg.asin()
    }
}

// ---------------------------------------------------------------------------
// VolumeGrating
// ---------------------------------------------------------------------------

/// Volume holographic transmission grating (index-modulated thick medium).
///
/// Models a transmission-mode volume phase hologram whose diffraction is
/// described by Raman-Nath theory (thin regime) **and** by Kogelnik's
/// coupled-wave theory (thick / off-Bragg regime).
///
/// Unit conventions: all distances in metres (SI).
///
/// References:
/// - Kogelnik, H. "Coupled Wave Theory for Thick Hologram Gratings" Bell Syst. Tech. J. 48, 2909 (1969)
/// - Abramowitz & Stegun, "Handbook of Mathematical Functions", §9.1 Bessel recurrences
#[derive(Debug, Clone)]
pub struct VolumeGrating {
    /// Peak refractive-index modulation amplitude Δn (dimensionless)
    pub delta_n: f64,
    /// Physical thickness of the holographic layer d (m)
    pub thickness_m: f64,
    /// Bragg (design) wavelength λ_B (m)
    pub bragg_wavelength_m: f64,
    /// Grating period Λ_g (m)
    pub period_m: f64,
    /// Bragg angle θ_B (rad) — angle of incidence inside the medium at Bragg resonance
    pub bragg_angle_rad: f64,
}

impl VolumeGrating {
    /// Create a new volume grating.
    ///
    /// # Errors
    /// - `InvalidLayer` if any distance/period is non-positive.
    /// - `InvalidRefractiveIndex` if delta_n is negative.
    pub fn new(
        delta_n: f64,
        thickness_m: f64,
        bragg_wavelength_m: f64,
        period_m: f64,
        bragg_angle_rad: f64,
    ) -> Result<Self, OxiPhotonError> {
        if thickness_m <= 0.0 {
            return Err(OxiPhotonError::InvalidLayer(format!(
                "VolumeGrating thickness must be > 0, got {thickness_m} m"
            )));
        }
        if period_m <= 0.0 {
            return Err(OxiPhotonError::InvalidLayer(format!(
                "VolumeGrating period must be > 0, got {period_m} m"
            )));
        }
        if bragg_wavelength_m <= 0.0 {
            return Err(OxiPhotonError::InvalidWavelength(bragg_wavelength_m));
        }
        if delta_n < 0.0 {
            return Err(OxiPhotonError::InvalidRefractiveIndex { n: delta_n, k: 0.0 });
        }
        Ok(Self {
            delta_n,
            thickness_m,
            bragg_wavelength_m,
            period_m,
            bragg_angle_rad,
        })
    }

    // -----------------------------------------------------------------------
    // Raman-Nath regime
    // -----------------------------------------------------------------------

    /// Raman-Nath phase modulation strength ν = 2π·Δn·d / λ.
    ///
    /// This is the argument to J_m in the Raman-Nath diffraction formula.
    pub fn raman_nath_modulation(&self, wavelength_m: f64) -> f64 {
        2.0 * PI * self.delta_n * self.thickness_m / wavelength_m
    }

    /// Raman-Nath diffraction order spectrum.
    ///
    /// Returns `(order, efficiency)` pairs for m ∈ {-m_max, ..., m_max}
    /// where efficiency η_m = J_m²(ν) and ν = 2π·Δn·d / λ.
    ///
    /// Uses Miller's downward recurrence (A&S 9.1.27) for numerical stability.
    pub fn diffraction_orders(&self, m_max: usize, wavelength_m: f64) -> Vec<(i32, f64)> {
        let nu = self.raman_nath_modulation(wavelength_m);
        let j_vals = bessel_j_integer(m_max, nu);
        let mut orders = Vec::with_capacity(2 * m_max + 1);
        for m in 0..=m_max as i32 {
            let eta = j_vals[m as usize].powi(2);
            if m == 0 {
                orders.push((0, eta));
            } else {
                orders.push((m, eta));
                // J_{-m}(x) = (-1)^m J_m(x), so J_{-m}² = J_m²
                orders.push((-m, eta));
            }
        }
        orders
    }

    /// First-order (m=1) diffraction efficiency in the thin (Raman-Nath) regime.
    ///
    /// Computes η₁ = J₁²(ν) exactly via Miller's downward recurrence.
    ///
    /// This supersedes the former approximate formula η₁ ≈ ν²/4 (valid only for ν ≪ 1).
    pub fn first_order_efficiency_thin(&self, wavelength_m: f64) -> f64 {
        let nu = self.raman_nath_modulation(wavelength_m);
        let j_vals = bessel_j_integer(1, nu);
        // j_vals[1] = J₁(ν); efficiency = J₁²(ν)
        j_vals[1].powi(2)
    }

    // -----------------------------------------------------------------------
    // Kogelnik thick-grating regime
    // -----------------------------------------------------------------------

    /// Kogelnik coupling constant κ = π·Δn / λ_B.
    ///
    /// At Bragg resonance, R(λ_B) = tanh²(κ·d).
    fn coupling_constant(&self) -> f64 {
        PI * self.delta_n / self.bragg_wavelength_m
    }

    /// Linear off-Bragg detuning parameter δ(λ).
    ///
    /// Derived from the on-Bragg condition λ_B = 2·Λ_g·cos θ_B:
    ///   δ(λ) = (π·cos θ_B / (Λ_g·λ)) · (λ − λ_B)  × correction factor
    ///
    /// Standard Kogelnik linearisation (first order in Δλ = λ − λ_B):
    ///   δ ≈ (2π·n·cos θ_B / λ_B²) · (λ_B − λ)
    ///     = −(2π·n·cos θ_B / λ_B²) · (λ − λ_B)
    ///
    /// Canonical Kogelnik 1969 detuning for a reflection-grating, derived from
    /// the Bragg condition λ_B = 2·n·Λ_g·cos θ_B, which lets us eliminate n:
    ///
    ///   δ = π·(λ_B − λ) / (λ_B · Λ_g)
    ///
    /// This is the per-unit-length detuning (m⁻¹); ξ = δ·d is used in the formula.
    fn detuning(&self, wavelength_m: f64) -> f64 {
        // Kogelnik 1969 reflection-grating detuning, derived from Bragg condition
        // λ_B = 2·n·Λ_g·cos θ_B  →  eliminate n to get δ = π·(λ_B−λ)/(λ_B·Λ_g)
        // Sign convention: δ > 0 when λ < λ_B (below Bragg wavelength)
        PI * (self.bragg_wavelength_m - wavelength_m) / (self.bragg_wavelength_m * self.period_m)
    }

    /// Reflection spectrum R(λ) using Kogelnik 1969 coupled-wave theory.
    ///
    /// Returns diffraction efficiency in [0, 1] for a reflection-mode volume hologram.
    ///
    /// Kogelnik's canonical result (eq. 4.19) with:
    ///   ν = κ·d  (normalised coupling),  ξ = δ·d  (normalised detuning)
    ///
    /// Branch A (ν² ≥ ξ²) — over-coupled / near Bragg:
    ///   Γ = √(ν² − ξ²)
    ///   η = sinh²(Γ) / (cosh²(Γ) − ξ²/ν²)
    ///
    /// Branch B (ξ² > ν²) — off-Bragg / under-coupled:
    ///   Φ = √(ξ² − ν²)
    ///   η = sin²(Φ) / (sin²(Φ) + ξ²/ν² − 1)
    ///
    /// Both branches reduce to tanh²(ν) at ξ = 0, and agree at the branch
    /// boundary ξ = ν where both give ν²/(1 + ν²) [using L'Hôpital].
    pub fn reflection_spectrum(&self, wavelength_m: f64) -> f64 {
        let kappa = self.coupling_constant();
        let d = self.thickness_m;
        let delta = self.detuning(wavelength_m);

        let nu = kappa * d; // normalised coupling ν = κ·d
        let xi = delta * d; // normalised detuning ξ = δ·d

        // Guard: κ = 0 means no hologram recorded
        if nu.abs() < 1e-30 {
            return 0.0;
        }

        let nu2 = nu * nu;
        let xi2 = xi * xi;
        let ratio2 = xi2 / nu2; // (ξ/ν)²

        if nu2 >= xi2 {
            // Branch A: Γ = √(ν² − ξ²) — near-Bragg / over-coupled
            let gamma = (nu2 - xi2).sqrt();
            if gamma < 1e-15 {
                // Boundary ξ = ν: lim_{Γ→0} sinh²(Γ)/(cosh²(Γ) − ratio²)
                // ≈ Γ² / (1 + Γ² − ratio²) → ν²/(1 + ν² − ν²) = ν²
                // More precisely at ξ=ν: η = tanh²(ν) / (1 − ratio²·sech²(ν) + ...)
                // Use numerically safe tanh² at exact Bragg
                return nu.tanh().powi(2);
            }
            let sh = gamma.sinh();
            let ch = gamma.cosh();
            let numerator = sh * sh;
            let denominator = ch * ch - ratio2;
            if denominator < 1e-15 {
                // Shouldn't occur for valid κ,δ — defensive clamp
                return 1.0_f64.min(numerator / 1e-15);
            }
            (numerator / denominator).clamp(0.0, 1.0)
        } else {
            // Branch B: Φ = √(ξ² − ν²) — off-Bragg / under-coupled
            let phi = (xi2 - nu2).sqrt();
            let sn = phi.sin();
            let numerator = sn * sn;
            // denominator = sin²(Φ) + ξ²/ν² − 1
            let denominator = numerator + ratio2 - 1.0;
            if denominator.abs() < 1e-15 {
                // Only occurs if sin²(Φ) ≈ 1 − ξ²/ν²; clamp to near-zero
                return 0.0;
            }
            if denominator <= 0.0 {
                // Numerical artefact near the branch boundary — return 0
                return 0.0;
            }
            (numerator / denominator).clamp(0.0, 1.0)
        }
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Bessel functions J₀(x), J₁(x), …, J_m_max(x) via Miller's downward recurrence.
///
/// Algorithm (Abramowitz & Stegun 9.1.27):
/// 1. Start at order M = m_max + max(3·|x|, 30) + 20 with seed f[M-1]=1, f[M]=0.
/// 2. Downward recurrence: f[m-1] = (2m/x)·f[m] − f[m+1].
/// 3. Normalise using Bessel completeness: Σ_{m=-∞}^{∞} J_m²(x) = 1,
///    equivalently  f[0]² + 2·Σ_{m=1}^{M} f[m]² = norm².
///
/// Edge cases: x ≈ 0 → J₀=1, J_m=0 for m≥1.
fn bessel_j_integer(m_max: usize, x: f64) -> Vec<f64> {
    if x.abs() < 1e-15 {
        let mut result = vec![0.0f64; m_max + 1];
        result[0] = 1.0;
        return result;
    }

    // Choose starting order M large enough for convergence.
    let m_extra = (x.abs() as usize).saturating_mul(3).max(30) + 20;
    let m_start = m_max + m_extra;

    // Allocate recurrence array with two guard slots beyond m_start.
    let mut f = vec![0.0f64; m_start + 2];
    // Seed: f[m_start] = 0, f[m_start-1] = 1 (unnormalised).
    f[m_start - 1] = 1.0;

    // Downward recurrence: f[m-1] = (2m/x)·f[m] − f[m+1]
    for m in (1..=m_start - 1).rev() {
        f[m - 1] = 2.0 * m as f64 / x * f[m] - f[m + 1];
        // Renormalise on overflow to prevent floating-point overflow.
        if f[m - 1].abs() > 1e150 {
            let scale = 1.0 / f[m - 1].abs();
            for fi in f.iter_mut() {
                *fi *= scale;
            }
        }
    }

    // Normalise using the Bessel completeness relation:
    //   J₀²(x) + 2·Σ_{m=1}^{∞} J_m²(x) = 1
    let sum_sq: f64 = f[0].powi(2) + 2.0 * f[1..=m_start].iter().map(|v| v.powi(2)).sum::<f64>();
    let norm = sum_sq.sqrt();

    if norm < 1e-300 {
        // Degenerate: return J₀=1 fallback
        let mut result = vec![0.0f64; m_max + 1];
        result[0] = 1.0;
        return result;
    }

    f[..=m_max].iter().map(|v| v / norm).collect()
}

/// sinc²(x) = (sin(πx)/(πx))², with sinc(0) = 1.
fn sinc_sq(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        return 1.0;
    }
    let pix = PI * x;
    (pix.sin() / pix).powi(2)
}

/// Approximation of J₁²(x) — Bessel function of the first kind, order 1, squared.
///
/// Uses the power-series for |x| < 8 and asymptotic formula for larger |x|.
fn bessel_j1_sq(x: f64) -> f64 {
    let j1 = bessel_j1(x);
    j1 * j1
}

/// Bessel J₁(x) via recurrence / series.
fn bessel_j1(x: f64) -> f64 {
    // Polynomial approximation (Abramowitz & Stegun 9.4.4, accurate to ~1e-7)
    // For |x| ≤ 3:
    let ax = x.abs();
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    if ax < 1e-12 {
        return 0.0;
    }
    if ax <= 3.0 {
        let t = x / 3.0;
        let t2 = t * t;
        // From A&S table 9.8 coefficients
        sign * (0.5 - 0.0625 * t2 + 0.002604_167 * t2 * t2 - 6.510_417e-5 * t2.powi(3)
            + 1.0286_458e-6 * t2.powi(4)
            - 1.0942_96e-8 * t2.powi(5))
            * x
    } else {
        // Asymptotic: J₁(x) ≈ sqrt(2/(πx)) * cos(x - 3π/4)
        (2.0 / (PI * ax)).sqrt() * (ax - 3.0 * PI / 4.0).cos() * sign
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grating(period_um: f64) -> DiffractionGrating {
        DiffractionGrating::new(period_um, 1.0, 0.0, GratingType::Reflection, 0.0).unwrap()
    }

    // 1. m=0 order: θ_out = θ_in regardless of wavelength
    #[test]
    fn test_grating_equation_zero_order() {
        let g = make_grating(1.0); // 1 μm period
        let theta_in = 0.3_f64.to_radians(); // 0.3 rad incidence
        let theta_out = g.diffraction_angle_rad(500.0, 0, theta_in).unwrap();
        // For m=0 in same medium: sin θ_out = sin θ_in → θ_out = θ_in
        let diff = (theta_out - theta_in).abs();
        assert!(diff < 1e-10, "m=0 should give θ_out = θ_in, diff={diff}");
    }

    // 2. m=1 first-order check against known geometry
    #[test]
    fn test_grating_equation_first_order() {
        // 1 μm period, 500 nm wavelength, normal incidence
        // sin θ_1 = 0 + 1·(0.5/1.0) = 0.5 → θ_1 = 30°
        let g = make_grating(1.0);
        let theta_out = g.diffraction_angle_rad(500.0, 1, 0.0).unwrap();
        let expected = (30.0_f64).to_radians();
        assert!(
            (theta_out - expected).abs() < 1e-8,
            "θ_1 should be 30°, got {:.4}°",
            theta_out.to_degrees()
        );
    }

    // 3. Littrow: retroreflection — diffracted angle is the mirror of incident angle.
    //
    // In our sign convention (both angles from grating normal, positive = same side):
    //   sin θ_m = sin θ_i + m·λ/Λ
    // Retroreflection requires sin θ_m = -sin θ_i, i.e. θ_m = -θ_i.
    // `littrow_angle_rad` returns θ_i (negative for positive m), so we expect θ_d = -θ_i.
    #[test]
    fn test_littrow_symmetry() {
        let g = make_grating(1.6); // 1.6 μm period
        let lambda_nm = 800.0;
        let order = 1;
        let theta_i = g.littrow_angle_rad(lambda_nm, order); // negative for m=+1
        assert!(theta_i.is_finite(), "Littrow angle should be finite");
        // Verify retroreflection: θ_d = -θ_i
        let theta_d = g.diffraction_angle_rad(lambda_nm, order, theta_i).unwrap();
        assert!(
            (theta_d + theta_i).abs() < 1e-8,
            "Littrow retroreflection not satisfied: θ_i={theta_i:.6}, θ_d={theta_d:.6}, expected θ_d=-θ_i"
        );
        // Verify the magnitude matches the textbook formula sin|θ_L| = |m|·λ/(2·Λ)
        let expected_mag = (order.abs() as f64 * lambda_nm * 1e-3 / (2.0 * 1.6)).asin();
        assert!(
            (theta_i.abs() - expected_mag).abs() < 1e-8,
            "|θ_L| mismatch: got {:.6}, expected {expected_mag:.6}",
            theta_i.abs()
        );
    }

    // 4. Angular dispersion positive for m=+1
    #[test]
    fn test_angular_dispersion_positive() {
        let g = make_grating(1.0);
        let disp = g.angular_dispersion_rad_per_nm(500.0, 1, 0.0);
        assert!(disp > 0.0, "dθ/dλ should be positive for m=+1, got {disp}");
    }

    // 5. Resolving power R = |m|·N
    #[test]
    fn test_resolving_power() {
        let g = make_grating(1.0);
        let r = g.resolving_power(2, 1000);
        assert_eq!(r, 2000.0, "R = |m|·N = 2·1000 = 2000");
        let r2 = g.resolving_power(-3, 500);
        assert_eq!(r2, 1500.0, "R = |m|·N = 3·500 = 1500");
    }

    // 6. FSR = λ/|m|
    #[test]
    fn test_fsr_formula() {
        let g = make_grating(1.0);
        let fsr = g.free_spectral_range_nm(500.0, 2);
        let expected = 500.0 / 2.0;
        assert!(
            (fsr - expected).abs() < 1e-10,
            "FSR={fsr}, expected {expected}"
        );
        let fsr_inf = g.free_spectral_range_nm(500.0, 0);
        assert!(fsr_inf.is_infinite(), "FSR for m=0 must be infinite");
    }

    // 7. Blaze efficiency maximum at design wavelength (m=1)
    #[test]
    fn test_blaze_efficiency_at_design_wavelength() {
        let g = DiffractionGrating::new(
            1.0,
            1.0,
            30.0, // 30° blaze angle
            GratingType::Reflection,
            200.0,
        )
        .unwrap();
        let lambda_blaze = g.blaze_wavelength_nm(0.0);
        let eta_blaze = g.blaze_efficiency(lambda_blaze, 1, 0.0);
        // At blaze wavelength, efficiency should be maximum (near 1)
        assert!(
            eta_blaze > 0.8,
            "Blaze efficiency at design wavelength should be high, got {eta_blaze:.4}"
        );
    }

    // 8. Wood's anomaly wavelengths exist and are positive
    #[test]
    fn test_woods_anomaly_exists() {
        let g = make_grating(1.0);
        let anomalies = g.woods_anomaly_wavelength_nm(0.0);
        assert!(
            !anomalies.is_empty(),
            "Should have Wood's anomaly wavelengths"
        );
        for &lam in &anomalies {
            assert!(lam > 0.0, "Anomaly wavelength must be positive, got {lam}");
        }
    }

    // 9. VBG peak reflectivity in [0, 1]
    #[test]
    fn test_vbg_peak_reflectivity_range() {
        let vbg = VolumeBraggGrating::new(1064.0, 1.487, 1e-3, 5.0, 2.0);
        let r = vbg.peak_reflectivity();
        assert!(
            (0.0..=1.0).contains(&r),
            "Reflectivity must be in [0,1], got {r}"
        );
    }

    // 10. VBG bandwidth positive
    #[test]
    fn test_vbg_bandwidth_positive() {
        let vbg = VolumeBraggGrating::new(1064.0, 1.487, 1e-3, 5.0, 2.0);
        let bw = vbg.bandwidth_nm();
        assert!(bw > 0.0, "VBG bandwidth must be positive, got {bw}");
    }

    // 11. Spectrometer wavelength range is ordered and non-trivial
    #[test]
    fn test_spectrometer_wavelength_range() {
        let g = DiffractionGrating::new(
            1.0,
            1.0,
            26.74, // Littrow at 532 nm, m=1
            GratingType::Reflection,
            150.0,
        )
        .unwrap();
        let spec = GratingSpectrometer::new(g, 100.0, 25.0, 2048, 1, 532.0, 0.0);
        let (lmin, lmax) = spec.wavelength_range_nm();
        assert!(lmin < lmax, "λ_min < λ_max required");
        assert!(lmax - lmin > 1.0, "Wavelength range should be > 1 nm");
    }
}
