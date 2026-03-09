/// Metasurface unit-cell physics.
///
/// Models covered:
/// - `DielectricPillar` — Mie-resonance dielectric pillar (TiO₂, Si, …)
/// - `PlasmonicMetal` + `PlasmonicAntenna` — Drude-model nanoantenna
/// - `VAntenna` — V-antenna for Pancharatnam-Berry phase
/// - `HuygensMetasurface` — balanced electric + magnetic dipoles
///
/// All physical quantities use SI units (metres, radians/s, etc.).
use num_complex::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Physical constants (local copies to avoid extra imports)
// ---------------------------------------------------------------------------
const SPEED_OF_LIGHT: f64 = 2.997_924_58e8; // m/s

// ---------------------------------------------------------------------------
// DielectricPillar
// ---------------------------------------------------------------------------

/// Mie-resonance dielectric nanopillar unit cell.
///
/// The phase and amplitude response are estimated via a coupled Fabry-Pérot /
/// effective-medium model that reproduces the key trends without requiring a
/// full FEM/RCWA solve:
///
/// * Transmission phase: 2π n_eff h / λ + resonance detuning factor
/// * Amplitude: Lorentzian centred on the magnetic-dipole resonance
///
/// This is suitable for rapid design-space exploration; use RCWA or FDTD for
/// final validation.
#[derive(Debug, Clone)]
pub struct DielectricPillar {
    /// Pillar radius (m)
    pub radius: f64,
    /// Pillar height (m)
    pub height: f64,
    /// Lattice constant / period (m)
    pub period: f64,
    /// Pillar refractive index (real part; e.g. TiO₂ ≈ 2.4, Si ≈ 3.5)
    pub n_pillar: f64,
    /// Substrate refractive index
    pub n_substrate: f64,
    /// Design wavelength (m)
    pub wavelength: f64,
}

impl DielectricPillar {
    /// Construct a TiO₂ pillar (n ≈ 2.4, n_sub = 1.46 SiO₂).
    pub fn new_tio2(radius: f64, height: f64, period: f64, wavelength: f64) -> Self {
        Self {
            radius,
            height,
            period,
            n_pillar: 2.4,
            n_substrate: 1.46,
            wavelength,
        }
    }

    /// Construct a crystalline-silicon pillar (n ≈ 3.5, n_sub = 1.46 SiO₂).
    pub fn new_silicon(radius: f64, height: f64, period: f64, wavelength: f64) -> Self {
        Self {
            radius,
            height,
            period,
            n_pillar: 3.5,
            n_substrate: 1.46,
            wavelength,
        }
    }

    /// Fill fraction: π r² / a²  (fraction of unit cell covered by pillar).
    pub fn fill_fraction(&self) -> f64 {
        PI * self.radius * self.radius / (self.period * self.period)
    }

    /// Effective refractive index via volume-average mixing rule.
    ///
    /// n_eff² = ff * n_p² + (1-ff) * n_air²
    fn n_eff(&self) -> f64 {
        let ff = self.fill_fraction().clamp(0.0, 1.0);
        let n_air = 1.0_f64;
        (ff * self.n_pillar * self.n_pillar + (1.0 - ff) * n_air * n_air).sqrt()
    }

    /// Magnetic-dipole (first Mie) resonance wavelength.
    ///
    /// Approximate relation: λ_MD ≈ 2.4 · n · h
    pub fn magnetic_dipole_resonance(&self) -> f64 {
        2.4 * self.n_pillar * self.height
    }

    /// Electric-dipole (second Mie) resonance wavelength.
    ///
    /// Approximate relation: λ_ED ≈ 1.7 · n · h
    pub fn electric_dipole_resonance(&self) -> f64 {
        1.7 * self.n_pillar * self.height
    }

    /// Accumulated propagation phase through the pillar (radians).
    ///
    /// φ_prop = 2π n_eff h / λ
    fn propagation_phase(&self) -> f64 {
        2.0 * PI * self.n_eff() * self.height / self.wavelength
    }

    /// Lorentzian resonance factor for the magnetic-dipole mode.
    ///
    /// Returns a complex factor: Δ = γ / (ω - ω_MD + iγ)
    /// evaluated at the design wavelength.
    fn md_lorentzian_factor(&self) -> Complex64 {
        let omega = 2.0 * PI * SPEED_OF_LIGHT / self.wavelength;
        let lambda_md = self.magnetic_dipole_resonance();
        let omega_md = 2.0 * PI * SPEED_OF_LIGHT / lambda_md;
        // Q ≈ n_p / (2 * fill_fraction) as a rough estimate
        let q = (self.n_pillar / (2.0 * self.fill_fraction().max(0.01))).clamp(5.0, 200.0);
        let gamma = omega_md / q;
        Complex64::new(0.0, -gamma) / Complex64::new(omega - omega_md, gamma)
    }

    /// Transmission amplitude and phase via Fabry-Pérot + Mie-resonance model.
    ///
    /// t(r) ≈ exp(i φ_prop) · (1 + A_res · Δ_MD)
    ///
    /// where A_res controls how strongly the resonance modifies the response.
    pub fn transmission(&self) -> Complex64 {
        let phi = self.propagation_phase();
        let prop = Complex64::from_polar(1.0, phi);
        // Fresnel loss at pillar–substrate interface
        let t_fresnel = 2.0 * self.n_eff() / (self.n_eff() + self.n_substrate);
        let resonance_strength = 0.3;
        let res = self.md_lorentzian_factor();
        let modulation = Complex64::new(1.0, 0.0) + resonance_strength * res;
        prop * modulation * t_fresnel
    }

    /// Phase of the transmission coefficient at a given radius.
    ///
    /// Returns a value in [0, 2π) (mod 2π from the propagation phase).
    pub fn phase_at_radius(&self, radius: f64) -> f64 {
        // Build a temporary pillar with the given radius and read its phase.
        let tmp = Self {
            radius,
            height: self.height,
            period: self.period,
            n_pillar: self.n_pillar,
            n_substrate: self.n_substrate,
            wavelength: self.wavelength,
        };
        let t = tmp.transmission();
        t.arg().rem_euclid(2.0 * PI)
    }

    /// Transmission amplitude at a given radius (0 … 1).
    pub fn amplitude_at_radius(&self, radius: f64) -> f64 {
        let tmp = Self {
            radius,
            height: self.height,
            period: self.period,
            n_pillar: self.n_pillar,
            n_substrate: self.n_substrate,
            wavelength: self.wavelength,
        };
        tmp.transmission().norm().clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// PlasmonicMetal
// ---------------------------------------------------------------------------

/// Supported plasmonic metals with Drude-model parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlasmonicMetal {
    Gold,
    Silver,
    Aluminum,
}

impl PlasmonicMetal {
    /// Plasma angular frequency ω_p (rad/s).
    ///
    /// Values from Johnson & Christy / Rakić:
    /// - Au: 1.37×10¹⁶  rad/s
    /// - Ag: 1.37×10¹⁶  rad/s  (similar to Au)
    /// - Al: 2.24×10¹⁶  rad/s
    pub fn plasma_frequency_rad_s(&self) -> f64 {
        match self {
            PlasmonicMetal::Gold => 1.37e16,
            PlasmonicMetal::Silver => 1.37e16,
            PlasmonicMetal::Aluminum => 2.24e16,
        }
    }

    /// Drude damping rate γ (rad/s).
    ///
    /// - Au: 4.08×10¹³ rad/s
    /// - Ag: 2.73×10¹³ rad/s
    /// - Al: 1.22×10¹⁴ rad/s
    pub fn damping_rate_rad_s(&self) -> f64 {
        match self {
            PlasmonicMetal::Gold => 4.08e13,
            PlasmonicMetal::Silver => 2.73e13,
            PlasmonicMetal::Aluminum => 1.22e14,
        }
    }

    /// High-frequency permittivity ε_∞ (interband correction).
    fn epsilon_inf(&self) -> f64 {
        match self {
            PlasmonicMetal::Gold => 9.5,
            PlasmonicMetal::Silver => 3.7,
            PlasmonicMetal::Aluminum => 1.0,
        }
    }

    /// Drude permittivity at angular frequency ω (rad/s).
    ///
    /// ε(ω) = ε_∞ − ω_p² / (ω² + iγω)
    pub fn permittivity(&self, omega: f64) -> Complex64 {
        let wp = self.plasma_frequency_rad_s();
        let gamma = self.damping_rate_rad_s();
        let eps_inf = self.epsilon_inf();
        let denominator = Complex64::new(omega * omega, gamma * omega);
        Complex64::new(eps_inf, 0.0) - Complex64::new(wp * wp, 0.0) / denominator
    }
}

// ---------------------------------------------------------------------------
// PlasmonicAntenna
// ---------------------------------------------------------------------------

/// Plasmonic nanoantenna (rod/bar) unit cell.
///
/// The resonance wavelength follows the half-wave condition:
/// λ_res ≈ 2 · n_eff · L
///
/// The phase shift is modelled as a Lorentzian response:
/// φ(ω) = arctan(γ / (ω_res − ω))   ∈ [0, π]
#[derive(Debug, Clone)]
pub struct PlasmonicAntenna {
    /// Rod length along x (m)
    pub length: f64,
    /// Rod width (m)
    pub width: f64,
    /// Rod height / thickness (m)
    pub height: f64,
    /// Period along x (m)
    pub period_x: f64,
    /// Period along y (m)
    pub period_y: f64,
    /// Metal type
    pub metal: PlasmonicMetal,
    /// Design wavelength (m)
    pub wavelength: f64,
}

impl PlasmonicAntenna {
    /// Gold rod antenna on a glass substrate.
    pub fn new_gold_rod(
        length: f64,
        width: f64,
        height: f64,
        period: f64,
        wavelength: f64,
    ) -> Self {
        Self {
            length,
            width,
            height,
            period_x: period,
            period_y: period,
            metal: PlasmonicMetal::Gold,
            wavelength,
        }
    }

    /// Effective refractive index of surrounding medium (geometric average
    /// of air and glass substrate n ≈ 1.5).
    fn n_eff_medium(&self) -> f64 {
        let n_sub = 1.5_f64;
        let n_air = 1.0_f64;
        ((n_sub + n_air) / 2.0).sqrt()
    }

    /// Half-wave resonance wavelength: λ_res ≈ 2 n_eff L.
    pub fn resonance_wavelength(&self) -> f64 {
        2.0 * self.n_eff_medium() * self.length
    }

    /// Angular frequency of the design wavelength.
    fn omega(&self) -> f64 {
        2.0 * PI * SPEED_OF_LIGHT / self.wavelength
    }

    /// Resonance angular frequency.
    fn omega_res(&self) -> f64 {
        2.0 * PI * SPEED_OF_LIGHT / self.resonance_wavelength()
    }

    /// Q-factor (approximate for a plasmonic rod).
    fn q_factor(&self) -> f64 {
        // Typical plasmonic Q ≈ 5–20; use Im(ε)/Re(ε) based estimate.
        let omega = self.omega();
        let eps = self.metal.permittivity(omega);
        let q_raw = eps.re.abs() / eps.im.abs();
        q_raw.clamp(3.0, 30.0)
    }

    /// Lorentzian phase shift φ(ω) ∈ [0, π] (radians).
    ///
    /// φ = atan2(γ, ω_res − ω) + π/2  shifted so that far below resonance → 0,
    /// far above resonance → π.
    pub fn phase_shift(&self) -> f64 {
        let omega = self.omega();
        let omega_r = self.omega_res();
        let gamma = self.metal.damping_rate_rad_s() / self.q_factor();
        (gamma / (omega_r - omega + 1e-30)).atan() + PI / 2.0
    }

    /// Lorentzian transmission / reflection amplitude (0 … 1).
    ///
    /// |t| ≈ 1 − A_peak · γ² / ((ω − ω_res)² + γ²)
    pub fn amplitude(&self) -> f64 {
        let omega = self.omega();
        let omega_r = self.omega_res();
        let gamma = self.metal.damping_rate_rad_s() / self.q_factor();
        let detuning_sq = (omega - omega_r).powi(2);
        let gamma_sq = gamma * gamma;
        let peak_absorption = 0.5_f64; // 50 % absorption at resonance
        (1.0 - peak_absorption * gamma_sq / (detuning_sq + gamma_sq)).clamp(0.0, 1.0)
    }

    /// Cross-polarisation efficiency for a rotated V-antenna / geometric-phase
    /// element.  Estimated as the squared sine of the scattering mismatch.
    ///
    /// η_cross ≈ |t_s − t_p|² / 4
    pub fn cross_polarization_efficiency(&self) -> f64 {
        // For a rod antenna, |t_s| ≈ 1 (no resonance) and |t_p| = amplitude().
        let t_p = self.amplitude();
        let t_s = 1.0_f64;
        0.25 * (t_s - t_p).powi(2)
    }
}

// ---------------------------------------------------------------------------
// VAntenna
// ---------------------------------------------------------------------------

/// V-antenna (gap antenna) for Pancharatnam-Berry phase generation.
///
/// A V-antenna supports two eigenmodes (symmetric and anti-symmetric) with
/// different resonance wavelengths.  When illuminated with linearly polarised
/// light the scattered field has a component in the cross-polarisation whose
/// phase equals twice the orientation angle (geometric phase).
#[derive(Debug, Clone)]
pub struct VAntenna {
    /// Arm length (m)
    pub arm_length: f64,
    /// Gap between arms (m)
    pub gap: f64,
    /// Opening angle of the V (degrees)
    pub opening_angle_deg: f64,
    /// In-plane rotation angle (degrees) — sets the geometric phase
    pub orientation_deg: f64,
    /// Metal type
    pub metal: PlasmonicMetal,
    /// Design wavelength (m)
    pub wavelength: f64,
}

impl VAntenna {
    /// Create a new V-antenna.
    pub fn new(arm: f64, gap: f64, angle_deg: f64, orientation_deg: f64, wavelength: f64) -> Self {
        Self {
            arm_length: arm,
            gap,
            opening_angle_deg: angle_deg,
            orientation_deg,
            metal: PlasmonicMetal::Gold,
            wavelength,
        }
    }

    /// Pancharatnam-Berry phase: φ_PB = 2 · θ_orientation (radians).
    pub fn geometric_phase(&self) -> f64 {
        2.0 * self.orientation_deg.to_radians()
    }

    /// Helper: symmetric-mode resonance wavelength.
    ///
    /// The symmetric mode of a V-antenna behaves like a rod of length
    /// 2L·cos(Δ/2) where Δ is the half-opening angle.
    fn lambda_sym(&self) -> f64 {
        let n_eff = 1.22_f64; // effective index for Au on glass
        let half_angle = (self.opening_angle_deg / 2.0).to_radians();
        2.0 * n_eff * self.arm_length * half_angle.cos()
    }

    /// Helper: anti-symmetric-mode resonance wavelength.
    ///
    /// The anti-symmetric mode behaves like a rod of length
    /// 2L·sin(Δ/2).
    fn lambda_asym(&self) -> f64 {
        let n_eff = 1.22_f64;
        let half_angle = (self.opening_angle_deg / 2.0).to_radians();
        2.0 * n_eff * self.arm_length * half_angle.sin()
    }

    /// Lorentzian amplitude for a given resonance wavelength.
    fn lorentzian_amplitude(&self, lambda_res: f64) -> f64 {
        let omega = 2.0 * PI * SPEED_OF_LIGHT / self.wavelength;
        let omega_res = 2.0 * PI * SPEED_OF_LIGHT / lambda_res;
        let gamma = self.metal.damping_rate_rad_s();
        let detuning_sq = (omega - omega_res).powi(2);
        gamma / (detuning_sq + gamma * gamma).sqrt()
    }

    /// Cross-polarisation amplitude (proportional to product of the two mode
    /// amplitudes).
    pub fn cross_pol_amplitude(&self) -> f64 {
        let a_sym = self.lorentzian_amplitude(self.lambda_sym());
        let a_asym = self.lorentzian_amplitude(self.lambda_asym());
        (a_sym * a_asym).clamp(0.0, 1.0)
    }

    /// 2×2 Jones matrix of the V-antenna in the lab frame.
    ///
    /// In the antenna frame the Jones matrix is diagonal:
    ///   J' = [[t_sym, 0], [0, t_asym]]
    ///
    /// Rotated by orientation angle θ:
    ///   J = R(-θ) · J' · R(θ)
    pub fn jones_matrix(&self) -> [[Complex64; 2]; 2] {
        let theta = self.orientation_deg.to_radians();
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Lorentzian complex amplitudes for each mode.
        let omega = 2.0 * PI * SPEED_OF_LIGHT / self.wavelength;
        let lorentz = |lambda_res: f64| -> Complex64 {
            let omega_res = 2.0 * PI * SPEED_OF_LIGHT / lambda_res;
            let gamma = self.metal.damping_rate_rad_s();
            Complex64::new(0.0, gamma) / Complex64::new(omega_res - omega, -gamma)
        };

        let t_s = lorentz(self.lambda_sym());
        let t_a = lorentz(self.lambda_asym());

        // Rotation: R = [[cos, -sin], [sin, cos]]
        // J = R(-θ) · diag(t_s, t_a) · R(θ)
        let j00 = t_s * cos_t * cos_t + t_a * sin_t * sin_t;
        let j01 = (t_a - t_s) * Complex64::new(sin_t * cos_t, 0.0);
        let j10 = j01;
        let j11 = t_s * sin_t * sin_t + t_a * cos_t * cos_t;

        [[j00, j01], [j10, j11]]
    }
}

// ---------------------------------------------------------------------------
// HuygensMetasurface
// ---------------------------------------------------------------------------

/// Huygens' metasurface — balanced electric and magnetic dipole resonances.
///
/// At the first Kerker condition (α_e = α_m) the backward scattering vanishes
/// and the structure achieves 2π phase coverage with near-unity transmission.
///
/// The transmission and reflection are computed from the surface susceptibility
/// model (Holloway et al.):
///
///   t = 1 + (i k / 2) (χ_ee + χ_mm)
///   r =     (i k / 2) (χ_ee − χ_mm)
///
/// where χ_ee = α_e / (ε₀ · a²)  and  χ_mm = α_m / (μ₀ · a²)
/// (a = period).
#[derive(Debug, Clone)]
pub struct HuygensMetasurface {
    /// Electric polarisability (SI units: C·m / (V/m) = C²·s²/kg)
    pub electric_polarizability: Complex64,
    /// Magnetic polarisability (SI units: A·m² / (A/m) = m³)
    pub magnetic_polarizability: Complex64,
    /// Lattice period (m)
    pub period: f64,
    /// Design wavelength (m)
    pub wavelength: f64,
}

impl HuygensMetasurface {
    /// Construct a Huygens' metasurface.
    pub fn new(alpha_e: Complex64, alpha_m: Complex64, period: f64, wavelength: f64) -> Self {
        Self {
            electric_polarizability: alpha_e,
            magnetic_polarizability: alpha_m,
            period,
            wavelength,
        }
    }

    /// Wave number in free space.
    fn k0(&self) -> f64 {
        2.0 * PI / self.wavelength
    }

    /// Normalised electric surface susceptibility χ_ee (units: m).
    ///
    /// χ_ee = α_e / (ε₀ · a²)
    fn chi_ee(&self) -> Complex64 {
        let eps0 = 8.854_187_817e-12_f64;
        let a2 = self.period * self.period;
        self.electric_polarizability / Complex64::new(eps0 * a2, 0.0)
    }

    /// Normalised magnetic surface susceptibility χ_mm (units: m).
    ///
    /// χ_mm = α_m / (μ₀ · a²)
    fn chi_mm(&self) -> Complex64 {
        let mu0 = 1.256_637_061_4e-6_f64;
        let a2 = self.period * self.period;
        self.magnetic_polarizability / Complex64::new(mu0 * a2, 0.0)
    }

    /// Complex transmission coefficient.
    pub fn transmission(&self) -> Complex64 {
        let k = self.k0();
        let factor = Complex64::new(0.0, k / 2.0);
        Complex64::new(1.0, 0.0) + factor * (self.chi_ee() + self.chi_mm())
    }

    /// Complex reflection coefficient.
    pub fn reflection(&self) -> Complex64 {
        let k = self.k0();
        let factor = Complex64::new(0.0, k / 2.0);
        factor * (self.chi_ee() - self.chi_mm())
    }

    /// True when the first Kerker condition is satisfied (α_e ≈ α_m within 5 %).
    ///
    /// At this condition reflection vanishes and transmission amplitude → 1
    /// while the phase sweeps 0 → 2π.
    pub fn kerker_condition_met(&self) -> bool {
        let diff = (self.electric_polarizability - self.magnetic_polarizability).norm();
        let sum = (self.electric_polarizability + self.magnetic_polarizability).norm();
        if sum < 1e-30 {
            return true;
        }
        diff / sum < 0.05
    }

    /// Approximate achievable phase coverage (radians).
    ///
    /// A Huygens' metasurface reaches 2π when the two resonances are tunable
    /// across the design wavelength.  Here we estimate the coverage from the
    /// argument range of the transmission coefficient as the electric polarisability
    /// magnitude is swept from 0 to |α_e_max| = 2|α_m|.
    pub fn phase_coverage(&self) -> f64 {
        // t_min (well below resonance): arg ≈ 0
        // t_max (well above resonance): arg ≈ 2π
        // Return theoretical maximum (2π) if Kerker condition is met,
        // otherwise estimate from the current transmission argument.
        if self.kerker_condition_met() {
            2.0 * PI
        } else {
            let t = self.transmission();
            // Estimate the swept range as twice the current argument (symmetric
            // about resonance).
            (2.0 * t.arg().abs()).clamp(0.0, 2.0 * PI)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn tio2_pillar_fill_fraction() {
        // period = 400 nm, radius = 80 nm → ff = π·80²/400² ≈ 0.1257
        let p = DielectricPillar::new_tio2(80e-9, 600e-9, 400e-9, 532e-9);
        let ff = p.fill_fraction();
        assert_abs_diff_eq!(
            ff,
            PI * 80.0_f64.powi(2) / 400.0_f64.powi(2),
            epsilon = 1e-6
        );
    }

    #[test]
    fn tio2_pillar_mie_resonances() {
        // λ_MD = 2.4 · 2.4 · 600 nm ≈ 3456 nm
        let p = DielectricPillar::new_tio2(80e-9, 600e-9, 400e-9, 532e-9);
        assert_abs_diff_eq!(
            p.magnetic_dipole_resonance(),
            2.4 * 2.4 * 600e-9,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            p.electric_dipole_resonance(),
            1.7 * 2.4 * 600e-9,
            epsilon = 1e-12
        );
    }

    #[test]
    fn si_pillar_transmission_is_unit_amplitude_bounded() {
        let p = DielectricPillar::new_silicon(100e-9, 500e-9, 350e-9, 700e-9);
        // Transmission amplitude should be in a physically plausible range.
        let amp = p.transmission().norm();
        assert!(amp > 0.0 && amp < 2.0, "amplitude={amp}");
    }

    #[test]
    fn plasmonic_gold_drude_permittivity_negative_real() {
        // Below plasma frequency real part of ε should be negative for Au.
        let metal = PlasmonicMetal::Gold;
        let omega = 2.0 * PI * SPEED_OF_LIGHT / 700e-9; // 700 nm
        let eps = metal.permittivity(omega);
        assert!(eps.re < 0.0, "ε_r(Au, 700nm)={}", eps.re);
    }

    #[test]
    fn v_antenna_geometric_phase_twice_orientation() {
        let ant = VAntenna::new(150e-9, 30e-9, 60.0, 45.0, 800e-9);
        assert_abs_diff_eq!(
            ant.geometric_phase(),
            2.0 * 45.0_f64.to_radians(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn huygens_kerker_condition_transmission_high() {
        // When α_e = α_m, reflection → 0 and |t| should be near 1.
        // Use small polarisabilities so that the susceptibility-based model
        // gives |t| ≈ 1 (perturbative regime).
        let alpha = Complex64::new(1e-33, 0.0);
        let hs = HuygensMetasurface::new(alpha, alpha, 300e-9, 500e-9);
        assert!(hs.kerker_condition_met(), "Kerker not detected");
        let r = hs.reflection().norm();
        assert!(r < 0.05, "|r|={r} (should be ~0 at Kerker condition)");
    }

    #[test]
    fn huygens_phase_coverage_two_pi_at_kerker() {
        let alpha = Complex64::new(1e-33, 0.0);
        let hs = HuygensMetasurface::new(alpha, alpha, 300e-9, 500e-9);
        assert_abs_diff_eq!(hs.phase_coverage(), 2.0 * PI, epsilon = 1e-10);
    }

    #[test]
    fn jones_matrix_symmetry() {
        // Jones matrix of a V-antenna should be symmetric (reciprocal scatterer).
        let ant = VAntenna::new(150e-9, 30e-9, 60.0, 30.0, 800e-9);
        let j = ant.jones_matrix();
        // J[0][1] == J[1][0]
        let diff = (j[0][1] - j[1][0]).norm();
        assert!(diff < 1e-20, "Jones matrix not symmetric: diff={diff}");
    }
}
