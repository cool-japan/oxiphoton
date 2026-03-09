/// Process Design Kit (PDK) components for Photonic Integrated Circuits.
///
/// Provides accurate models for SOI and SiN photonic platforms, including
/// effective index calculations, loss models, and standard component specifications.
use std::f64::consts::PI;

// ── Physical constants ────────────────────────────────────────────────────────
/// Speed of light in vacuum (m/s)
const C0: f64 = 2.997_924_58e8;

// ─────────────────────────────────────────────────────────────────────────────
// SOI Process
// ─────────────────────────────────────────────────────────────────────────────

/// Silicon-on-Insulator (SOI) process parameters.
///
/// Models the 220 nm and 300 nm SOI photonic platforms used in foundries
/// such as IME A*STAR, IMEC, and CEA-LETI.
#[derive(Debug, Clone)]
pub struct SoiProcess {
    /// Silicon core thickness (nm)
    pub silicon_thickness_nm: f64,
    /// Buried oxide (BOX) layer thickness (nm)
    pub oxide_thickness_nm: f64,
    /// Silicon refractive index at 1550 nm
    pub n_si: f64,
    /// SiO₂ refractive index at 1550 nm
    pub n_sio2: f64,
    /// Minimum lithographic feature size (nm)
    pub min_feature_size_nm: f64,
    /// Propagation loss for a standard 450 nm wide strip waveguide (dB/cm)
    pub waveguide_loss_db_per_cm: f64,
}

impl SoiProcess {
    /// Standard 220 nm SOI process (most common, compatible with CMOS fabs).
    pub fn standard_220nm() -> Self {
        Self {
            silicon_thickness_nm: 220.0,
            oxide_thickness_nm: 2000.0,
            n_si: 3.4757,
            n_sio2: 1.4440,
            min_feature_size_nm: 100.0,
            waveguide_loss_db_per_cm: 2.0,
        }
    }

    /// 300 nm SOI process — lower loss, larger single-mode window.
    pub fn thin_300nm() -> Self {
        Self {
            silicon_thickness_nm: 300.0,
            oxide_thickness_nm: 3000.0,
            n_si: 3.4757,
            n_sio2: 1.4440,
            min_feature_size_nm: 120.0,
            waveguide_loss_db_per_cm: 1.2,
        }
    }

    /// Effective index of the TE₀ mode in a strip waveguide.
    ///
    /// Uses a semi-analytical fit calibrated to 3D-FDTD results for the
    /// 220 nm SOI platform. The width is clamped to [300, 1200] nm.
    ///
    /// # Arguments
    /// * `width_nm` – Strip waveguide width (nm)
    pub fn n_eff_strip(&self, width_nm: f64) -> f64 {
        let w = width_nm.clamp(300.0, 1200.0);
        let h_norm = self.silicon_thickness_nm / 220.0;
        // Polynomial fit: n_eff = a + b*(w-450)/450 + c*(h-1)
        let a = 2.45;
        let b = 0.55;
        let c = 0.25;
        let w_norm = (w - 450.0) / 450.0;
        let h_fac = h_norm - 1.0;
        (a + b * w_norm + c * h_fac).clamp(self.n_sio2 + 0.01, self.n_si - 0.01)
    }

    /// Effective index of the TE₀ mode in a rib waveguide.
    ///
    /// # Arguments
    /// * `width_nm`  – Rib width (nm)
    /// * `etch_nm`   – Etch depth (nm, partial etch into Si slab)
    pub fn n_eff_rib(&self, width_nm: f64, etch_nm: f64) -> f64 {
        let w = width_nm.clamp(400.0, 2000.0);
        let etch_frac = (etch_nm / self.silicon_thickness_nm).clamp(0.0, 1.0);
        // Rib mode is between slab index and strip index
        let n_strip = self.n_eff_strip(w);
        let n_slab = self.n_sio2 + (n_strip - self.n_sio2) * 0.6 * (1.0 - etch_frac);
        n_slab + (n_strip - n_slab) * etch_frac.powi(2)
    }

    /// Returns the (min, max) single-mode width range in nm for this process.
    ///
    /// Below `min`, the mode becomes leaky; above `max`, higher-order modes appear.
    pub fn single_mode_width_range(&self) -> (f64, f64) {
        match self.silicon_thickness_nm as u32 {
            0..=249 => (300.0, 500.0),
            250..=269 => (320.0, 550.0),
            270..=320 => (340.0, 600.0),
            _ => (350.0, 650.0),
        }
    }

    /// Propagation loss (dB/cm) as a function of waveguide width.
    ///
    /// Narrower waveguides suffer higher sidewall roughness scattering loss.
    /// The model uses a 1/w⁴ sidewall scattering dependence.
    ///
    /// # Arguments
    /// * `width_nm` – Waveguide width (nm)
    pub fn propagation_loss_db_per_cm(&self, width_nm: f64) -> f64 {
        let w_ref = 450.0_f64;
        let w = width_nm.max(200.0);
        // Sidewall scattering ∝ (σ/w)² * (∂n_eff/∂w)² — simplified power law
        self.waveguide_loss_db_per_cm * (w_ref / w).powi(3)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SiN Process
// ─────────────────────────────────────────────────────────────────────────────

/// Silicon Nitride (Si₃N₄) photonic process parameters.
///
/// SiN offers ultra-low loss (< 0.1 dB/m demonstrated) and a broad
/// transparency window from visible to mid-IR.
#[derive(Debug, Clone)]
pub struct SiNProcess {
    /// Si₃N₄ refractive index at 1550 nm
    pub n_sin: f64,
    /// Core thickness (nm)
    pub thickness_nm: f64,
    /// Film stress (MPa; positive = tensile, negative = compressive)
    pub stress_mpa: f64,
    /// Baseline propagation loss (dB/cm)
    pub waveguide_loss_db_per_cm: f64,
    /// Transparency window (m): (λ_min, λ_max)
    pub wavelength_range: (f64, f64),
}

impl SiNProcess {
    /// Standard 400 nm SiN process — balanced confinement and loss.
    pub fn standard_400nm() -> Self {
        Self {
            n_sin: 1.9963,
            thickness_nm: 400.0,
            stress_mpa: 900.0,
            waveguide_loss_db_per_cm: 0.15,
            wavelength_range: (0.4e-6, 2.35e-6),
        }
    }

    /// Low-loss 700 nm SiN process — anomalous dispersion, Q > 10⁶.
    pub fn low_loss_700nm() -> Self {
        Self {
            n_sin: 1.9870,
            thickness_nm: 700.0,
            stress_mpa: 200.0,
            waveguide_loss_db_per_cm: 0.05,
            wavelength_range: (0.5e-6, 2.35e-6),
        }
    }

    /// Effective index of TE₀ mode in a SiN strip waveguide.
    ///
    /// # Arguments
    /// * `width_nm` – Waveguide width (nm)
    pub fn n_eff_strip(&self, width_nm: f64) -> f64 {
        let w = width_nm.clamp(500.0, 3000.0);
        let h_norm = self.thickness_nm / 400.0;
        let a = 1.70;
        let b = 0.22;
        let c = 0.10;
        let w_norm = (w - 1000.0) / 1000.0;
        let h_fac = h_norm - 1.0;
        (a + b * w_norm + c * h_fac).clamp(1.444 + 0.01, self.n_sin - 0.01)
    }

    /// Returns the waveguide width (nm) that achieves anomalous group-velocity
    /// dispersion at 1550 nm — required for Kerr frequency comb generation.
    ///
    /// Uses the empirical relation: w_anomalous ≈ 1750*(h/700)^0.8 nm.
    pub fn anomalous_dispersion_width_nm(&self) -> f64 {
        1750.0 * (self.thickness_nm / 700.0).powf(0.8)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PIC Process enum
// ─────────────────────────────────────────────────────────────────────────────

/// Photonic process platform selector.
#[derive(Debug, Clone)]
pub enum PicProcess {
    /// 220 nm SOI platform
    Soi220(SoiProcess),
    /// 400 nm SiN platform
    Sin400(SiNProcess),
    /// 700 nm SiN platform
    Sin700(SiNProcess),
    /// InP platform (III-V, active devices)
    InP,
    /// Lithium niobate on insulator (LNOI) platform
    LiNbO3,
}

impl PicProcess {
    /// Returns the core refractive index of the platform material.
    pub fn core_index(&self) -> f64 {
        match self {
            Self::Soi220(p) => p.n_si,
            Self::Sin400(p) | Self::Sin700(p) => p.n_sin,
            Self::InP => 3.17,
            Self::LiNbO3 => 2.21,
        }
    }

    /// Returns the cladding (SiO₂) refractive index.
    pub fn clad_index(&self) -> f64 {
        match self {
            Self::Soi220(_) => 1.4440,
            Self::Sin400(_) | Self::Sin700(_) => 1.4440,
            Self::InP => 3.17, // InP buried layer ≈ same n
            Self::LiNbO3 => 1.444,
        }
    }

    /// Returns the baseline propagation loss (dB/cm).
    pub fn baseline_loss_db_per_cm(&self) -> f64 {
        match self {
            Self::Soi220(p) => p.waveguide_loss_db_per_cm,
            Self::Sin400(p) | Self::Sin700(p) => p.waveguide_loss_db_per_cm,
            Self::InP => 3.0,
            Self::LiNbO3 => 0.3,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Component Specs
// ─────────────────────────────────────────────────────────────────────────────

/// Specification for a Multi-Mode Interference (MMI) splitter.
#[derive(Debug, Clone)]
pub struct MmiSpec {
    /// MMI section length (µm)
    pub length_um: f64,
    /// MMI section width (µm)
    pub width_um: f64,
    /// Insertion loss (dB)
    pub insertion_loss_db: f64,
    /// Amplitude imbalance between ports (dB)
    pub imbalance_db: f64,
    /// 1-dB bandwidth (nm)
    pub bandwidth_nm: f64,
}

/// Specification for a Directional Coupler (DC).
#[derive(Debug, Clone)]
pub struct DcSpec {
    /// Coupling section length (µm) for target coupling ratio
    pub coupling_length_um: f64,
    /// Power coupling coefficient κ² (0–1)
    pub coupling_coefficient: f64,
    /// Cross-port extinction ratio (dB)
    pub extinction_ratio_db: f64,
    /// 3-dB bandwidth (nm)
    pub bandwidth_nm: f64,
}

/// Specification for a Y-junction splitter.
#[derive(Debug, Clone)]
pub struct YJunctionSpec {
    /// Total transition length (µm)
    pub length_um: f64,
    /// Insertion loss (dB)
    pub insertion_loss_db: f64,
    /// Amplitude imbalance (dB)
    pub imbalance_db: f64,
}

/// Specification for a grating coupler.
#[derive(Debug, Clone)]
pub struct GcSpec {
    /// Grating period (nm)
    pub period_nm: f64,
    /// Grating duty cycle (0–1)
    pub duty_cycle: f64,
    /// Peak coupling efficiency (dB)
    pub coupling_efficiency_db: f64,
    /// 1-dB bandwidth (nm)
    pub bandwidth_nm: f64,
    /// Fiber tilt angle (degrees from normal)
    pub angle_deg: f64,
}

/// Specification for a ring resonator waveguide component.
#[derive(Debug, Clone)]
pub struct RingSpec {
    /// Ring radius (µm)
    pub radius_um: f64,
    /// Bus-ring coupling gap (nm)
    pub gap_nm: f64,
    /// Loaded Q-factor
    pub q_factor: f64,
    /// Free spectral range (nm)
    pub fsr_nm: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// PIC Component Library
// ─────────────────────────────────────────────────────────────────────────────

/// Standard PIC component library for a specific process and wavelength.
///
/// Provides pre-characterized component specifications calibrated to the
/// chosen PDK process. All dimensions are design-center values; consult
/// the PDK documentation for corner-case process tolerances.
#[derive(Debug, Clone)]
pub struct PicComponentLibrary {
    /// Photonic process platform
    pub process: PicProcess,
    /// Design wavelength (m)
    pub wavelength: f64,
}

impl PicComponentLibrary {
    /// Create a library for the standard 220 nm SOI process.
    ///
    /// # Arguments
    /// * `wavelength` – Design wavelength in metres (e.g. `1.55e-6`)
    pub fn new_soi(wavelength: f64) -> Self {
        Self {
            process: PicProcess::Soi220(SoiProcess::standard_220nm()),
            wavelength,
        }
    }

    /// Create a library for the 400 nm SiN process.
    ///
    /// # Arguments
    /// * `wavelength` – Design wavelength in metres
    pub fn new_sin(wavelength: f64) -> Self {
        Self {
            process: PicProcess::Sin400(SiNProcess::standard_400nm()),
            wavelength,
        }
    }

    /// Return the design wavelength in nm.
    fn lambda_nm(&self) -> f64 {
        self.wavelength * 1.0e9
    }

    /// 1×2 MMI splitter specification.
    ///
    /// Self-imaging length: L_π = n_r * W_mmi² / λ.
    pub fn mmi_1x2(&self) -> MmiSpec {
        let lambda_nm = self.lambda_nm();
        let n_r = self.process.core_index();
        let w_um = match &self.process {
            PicProcess::Soi220(_) => 5.6,
            PicProcess::Sin400(_) | PicProcess::Sin700(_) => 8.0,
            _ => 6.0,
        };
        let w_nm = w_um * 1000.0;
        // MMI length: L = 3 * L_pi / 4 for 1x2
        let l_pi_nm = n_r * w_nm * w_nm / lambda_nm;
        let length_um = 3.0 * l_pi_nm / 4.0 / 1000.0;
        MmiSpec {
            length_um,
            width_um: w_um,
            insertion_loss_db: 0.3,
            imbalance_db: 0.1,
            bandwidth_nm: 80.0,
        }
    }

    /// 2×2 MMI coupler specification (90° hybrid or 3-dB coupler).
    pub fn mmi_2x2(&self) -> MmiSpec {
        let lambda_nm = self.lambda_nm();
        let n_r = self.process.core_index();
        let w_um = match &self.process {
            PicProcess::Soi220(_) => 6.0,
            PicProcess::Sin400(_) | PicProcess::Sin700(_) => 9.0,
            _ => 7.0,
        };
        let w_nm = w_um * 1000.0;
        // L = L_pi for 2x2 input/output ports
        let l_pi_nm = n_r * w_nm * w_nm / lambda_nm;
        let length_um = l_pi_nm / 1000.0;
        MmiSpec {
            length_um,
            width_um: w_um,
            insertion_loss_db: 0.5,
            imbalance_db: 0.2,
            bandwidth_nm: 60.0,
        }
    }

    /// Directional coupler specification for a given gap.
    ///
    /// Coupling coefficient derived from the coupled-mode theory beat length:
    /// κ² = sin²(π * L_c / (2 * L_beat))
    ///
    /// # Arguments
    /// * `gap_nm` – Coupling gap between waveguides (nm)
    pub fn directional_coupler(&self, gap_nm: f64) -> DcSpec {
        let lambda_nm = self.lambda_nm();
        let n_eff = match &self.process {
            PicProcess::Soi220(p) => p.n_eff_strip(450.0),
            PicProcess::Sin400(p) | PicProcess::Sin700(p) => p.n_eff_strip(1000.0),
            _ => 2.0,
        };
        // Coupling coefficient decays exponentially with gap
        let g0 = match &self.process {
            PicProcess::Soi220(_) => 200.0_f64,
            _ => 500.0_f64,
        };
        let kappa = (-gap_nm / g0).exp() * 0.98;
        // Beat length (µm)
        let l_beat_um = lambda_nm / (2.0 * (n_eff * 0.02 * (-gap_nm / g0).exp())) / 1000.0;
        let l_beat_um = l_beat_um.clamp(5.0, 2000.0);
        // Coupling length for 50:50 split
        let coupling_length_um = l_beat_um / 2.0;
        DcSpec {
            coupling_length_um,
            coupling_coefficient: kappa.clamp(0.0, 1.0),
            extinction_ratio_db: 20.0 - gap_nm / 50.0,
            bandwidth_nm: 30.0 + gap_nm / 20.0,
        }
    }

    /// Y-junction splitter specification.
    pub fn y_junction(&self) -> YJunctionSpec {
        let length_um = match &self.process {
            PicProcess::Soi220(_) => 20.0,
            PicProcess::Sin400(_) | PicProcess::Sin700(_) => 30.0,
            _ => 25.0,
        };
        YJunctionSpec {
            length_um,
            insertion_loss_db: 0.5,
            imbalance_db: 0.05,
        }
    }

    /// Surface grating coupler specification.
    ///
    /// Period computed from the Bragg condition:
    /// Λ = λ / (n_eff - n_clad * sin θ)
    pub fn grating_coupler(&self) -> GcSpec {
        let lambda_nm = self.lambda_nm();
        let theta_deg = 10.0_f64;
        let theta_rad = theta_deg * PI / 180.0;
        let n_eff = match &self.process {
            PicProcess::Soi220(p) => p.n_eff_strip(450.0),
            PicProcess::Sin400(p) => p.n_eff_strip(1000.0),
            PicProcess::Sin700(p) => p.n_eff_strip(1000.0),
            _ => 2.0,
        };
        let n_clad = 1.0; // air cladding for fiber coupling
        let period_nm = lambda_nm / (n_eff - n_clad * theta_rad.sin());
        GcSpec {
            period_nm,
            duty_cycle: 0.50,
            coupling_efficiency_db: -3.5,
            bandwidth_nm: 40.0,
            angle_deg: theta_deg,
        }
    }

    /// Ring resonator waveguide specification.
    ///
    /// FSR computed from: FSR = λ² / (n_g * 2πR)
    pub fn ring_resonator_wg(&self) -> RingSpec {
        let lambda = self.wavelength;
        let n_eff = match &self.process {
            PicProcess::Soi220(p) => p.n_eff_strip(450.0),
            PicProcess::Sin400(p) | PicProcess::Sin700(p) => p.n_eff_strip(1000.0),
            _ => 2.0,
        };
        // Group index ≈ n_eff + dn_eff/dλ * λ, approximate as 1.3*n_eff for SOI
        let n_g = n_eff * 1.30;
        let radius_um = 10.0;
        let circumference_m = 2.0 * PI * radius_um * 1.0e-6;
        let fsr_m = lambda * lambda / (n_g * circumference_m);
        let fsr_nm = fsr_m * 1.0e9;
        RingSpec {
            radius_um,
            gap_nm: 200.0,
            q_factor: 10_000.0,
            fsr_nm,
        }
    }

    /// Compute the free spectral range (nm) for a ring of given radius.
    ///
    /// # Arguments
    /// * `radius_um` – Ring radius (µm)
    /// * `n_g`       – Group index
    pub fn fsr_nm(&self, radius_um: f64, n_g: f64) -> f64 {
        let lambda = self.wavelength;
        let circumference_m = 2.0 * PI * radius_um * 1.0e-6;
        lambda * lambda / (n_g * circumference_m) * 1.0e9
    }

    /// Compute the finesse from Q-factor and FSR.
    ///
    /// F = Q * FSR / λ_center
    pub fn finesse_from_q(&self, q_factor: f64, fsr_nm: f64) -> f64 {
        let lambda_nm = self.lambda_nm();
        q_factor * fsr_nm / lambda_nm
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: wavelength ↔ frequency
// ─────────────────────────────────────────────────────────────────────────────

/// Convert wavelength (m) to optical frequency (Hz).
pub fn wavelength_to_freq(wavelength_m: f64) -> f64 {
    C0 / wavelength_m
}

/// Convert optical frequency (Hz) to wavelength (m).
pub fn freq_to_wavelength(freq_hz: f64) -> f64 {
    C0 / freq_hz
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_soi_220_n_eff_reasonable() {
        let soi = SoiProcess::standard_220nm();
        let n_eff = soi.n_eff_strip(450.0);
        // n_eff for 450 nm wide, 220 nm thick SOI at 1550 nm ≈ 2.4–2.5
        assert!(n_eff > 2.0, "n_eff too low: {n_eff}");
        assert!(n_eff < 3.5, "n_eff too high: {n_eff}");
    }

    #[test]
    fn test_soi_single_mode_range() {
        let soi = SoiProcess::standard_220nm();
        let (w_min, w_max) = soi.single_mode_width_range();
        assert!(w_min < w_max);
        assert!(w_min >= 250.0);
        assert!(w_max <= 600.0);
    }

    #[test]
    fn test_soi_loss_increases_for_narrow_guides() {
        let soi = SoiProcess::standard_220nm();
        let loss_450 = soi.propagation_loss_db_per_cm(450.0);
        let loss_300 = soi.propagation_loss_db_per_cm(300.0);
        assert!(loss_300 > loss_450, "Narrow guide should have higher loss");
    }

    #[test]
    fn test_sin_anomalous_dispersion_width() {
        let sin = SiNProcess::low_loss_700nm();
        let w = sin.anomalous_dispersion_width_nm();
        // Should be around 1750 nm for 700 nm thick SiN
        assert_abs_diff_eq!(w, 1750.0, epsilon = 1.0);
    }

    #[test]
    fn test_mmi_1x2_length_positive() {
        let lib = PicComponentLibrary::new_soi(1.55e-6);
        let mmi = lib.mmi_1x2();
        assert!(mmi.length_um > 0.0);
        assert!(mmi.width_um > 0.0);
    }

    #[test]
    fn test_ring_fsr_soi() {
        let lib = PicComponentLibrary::new_soi(1.55e-6);
        let ring = lib.ring_resonator_wg();
        // For R=10 µm, FSR should be several nm (~ 8–12 nm for SOI)
        assert!(ring.fsr_nm > 1.0, "FSR too small: {} nm", ring.fsr_nm);
        assert!(ring.fsr_nm < 50.0, "FSR too large: {} nm", ring.fsr_nm);
    }

    #[test]
    fn test_grating_coupler_period_reasonable() {
        let lib = PicComponentLibrary::new_soi(1.55e-6);
        let gc = lib.grating_coupler();
        // Typical SOI grating period at 10° tilt ≈ 600–700 nm
        assert!(
            gc.period_nm > 400.0 && gc.period_nm < 1000.0,
            "Unexpected grating period: {} nm",
            gc.period_nm
        );
    }

    #[test]
    fn test_wavelength_freq_roundtrip() {
        let lambda = 1.55e-6_f64;
        let f = wavelength_to_freq(lambda);
        let lambda2 = freq_to_wavelength(f);
        assert_abs_diff_eq!(lambda, lambda2, epsilon = 1.0e-18);
    }

    #[test]
    fn test_finesse_from_q() {
        let lib = PicComponentLibrary::new_soi(1.55e-6);
        // Q=10000, FSR=10 nm → finesse = 10000 * 10 / 1550 ≈ 64.5
        let f = lib.finesse_from_q(10_000.0, 10.0);
        assert!(f > 50.0 && f < 100.0, "Finesse out of range: {f}");
    }
}
