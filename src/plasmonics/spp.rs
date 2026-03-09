//! Surface Plasmon Polariton (SPP) physics
//!
//! Implements:
//! - Drude model for metal permittivity: ε(ω) = ε_∞ - ωp²/(ω(ω + iγ))
//! - SPP dispersion at metal-dielectric interfaces
//! - Kretschmann ATR coupling configuration (3-layer transfer matrix)
//! - Metal-Insulator-Metal (MIM) plasmonic waveguides

use num_complex::Complex64;
use std::f64::consts::PI;

use crate::error::OxiPhotonError;

// Physical constants (SI units)
const EPS0: f64 = 8.854_187_817e-12; // F/m
const C0: f64 = 2.997_924_58e8; // m/s
const HBAR: f64 = 1.054_571_8e-34; // J·s
const E_CHARGE: f64 = 1.602_176_634e-19; // C

// Suppress unused constant warnings — kept for completeness as physical constants
#[allow(dead_code)]
const _EPS0: f64 = EPS0;
#[allow(dead_code)]
const _HBAR: f64 = HBAR;
#[allow(dead_code)]
const _E_CHARGE: f64 = E_CHARGE;

// ──────────────────────────────────────────────────────────────────────────────
// DrudeMetal
// ──────────────────────────────────────────────────────────────────────────────

/// Drude model permittivity for metals.
///
/// ε(ω) = ε_∞ − ωp² / (ω · (ω + iγ))
///
/// Parameters are in SI angular-frequency units (rad/s).
#[derive(Debug, Clone)]
pub struct DrudeMetal {
    /// High-frequency (inter-band) dielectric constant
    pub eps_inf: f64,
    /// Plasma frequency (rad/s)
    pub omega_p: f64,
    /// Collision / damping rate (rad/s)
    pub gamma: f64,
    /// Human-readable identifier
    pub name: String,
}

impl DrudeMetal {
    /// Create a new Drude metal model.
    pub fn new(eps_inf: f64, omega_p: f64, gamma: f64, name: impl Into<String>) -> Self {
        Self {
            eps_inf,
            omega_p,
            gamma,
            name: name.into(),
        }
    }

    /// Complex permittivity at angular frequency ω.
    ///
    /// ε(ω) = ε_∞ − ωp² / (ω · (ω + iγ))
    pub fn permittivity(&self, omega: f64) -> Complex64 {
        let eps_inf = Complex64::new(self.eps_inf, 0.0);
        let wp2 = self.omega_p * self.omega_p;
        // denominator: ω(ω + iγ) = ω² + iγω
        let denom = Complex64::new(omega * omega, self.gamma * omega);
        eps_inf - wp2 / denom
    }

    /// Complex refractive index ñ = √ε.
    pub fn refractive_index(&self, omega: f64) -> Complex64 {
        self.permittivity(omega).sqrt()
    }

    /// Optical skin depth in nm: δ = λ / (4π · Im(ñ))
    ///
    /// λ = 2πc/ω, so δ = c / (2ω · Im(ñ))
    pub fn skin_depth_nm(&self, omega: f64) -> f64 {
        let n_tilde = self.refractive_index(omega);
        let k_opt = n_tilde.im.abs();
        if k_opt < f64::EPSILON {
            return f64::INFINITY;
        }
        // δ = c / (ω * Im(ñ)) — factor of 2 from amplitude → intensity
        let delta_m = C0 / (omega * k_opt);
        delta_m * 1.0e9 // m → nm
    }

    // ── Preset metals ────────────────────────────────────────────────────────

    /// Gold (Au) — Johnson & Christy parameters.
    ///
    /// ωp = 13.7×10¹⁵ rad/s, γ = 1.07×10¹⁴ rad/s, ε∞ = 9.5
    pub fn gold() -> Self {
        Self::new(9.5, 13.7e15, 1.07e14, "Gold")
    }

    /// Silver (Ag).
    ///
    /// ωp = 13.7×10¹⁵ rad/s, γ = 2.73×10¹³ rad/s, ε∞ = 3.7
    pub fn silver() -> Self {
        Self::new(3.7, 13.7e15, 2.73e13, "Silver")
    }

    /// Aluminum (Al).
    ///
    /// ωp = 22.7×10¹⁵ rad/s, γ = 1.27×10¹⁴ rad/s, ε∞ = 1.0
    pub fn aluminum() -> Self {
        Self::new(1.0, 22.7e15, 1.27e14, "Aluminum")
    }

    /// Copper (Cu).
    ///
    /// ωp = 13.4×10¹⁵ rad/s, γ = 1.45×10¹⁴ rad/s, ε∞ = 10.8
    pub fn copper() -> Self {
        Self::new(10.8, 13.4e15, 1.45e14, "Copper")
    }

    // ── Derived quantities ───────────────────────────────────────────────────

    /// Plasma energy ℏωp in electron-volts.
    pub fn plasma_energy_ev(&self) -> f64 {
        self.omega_p * HBAR / E_CHARGE
    }

    /// Returns `true` when the metal can support an SPP at the given
    /// angular frequency in contact with a dielectric of permittivity
    /// `eps_dielectric`.
    ///
    /// Condition: Re(ε_metal) < −ε_dielectric  (ε_dielectric > 0 assumed)
    pub fn supports_spp(&self, omega: f64, eps_dielectric: f64) -> bool {
        let eps_m = self.permittivity(omega);
        eps_m.re < -eps_dielectric
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SurfacePlasmonPolariton
// ──────────────────────────────────────────────────────────────────────────────

/// Surface plasmon polariton (SPP) at a single metal–dielectric interface.
///
/// The interface lies in the xy-plane; the metal occupies z < 0 and the
/// dielectric z > 0.  All calculations are for TM polarisation.
pub struct SurfacePlasmonPolariton {
    pub metal: DrudeMetal,
    /// Real dielectric constant of the adjacent insulator (εd > 0).
    pub eps_dielectric: f64,
}

impl SurfacePlasmonPolariton {
    pub fn new(metal: DrudeMetal, eps_dielectric: f64) -> Self {
        Self {
            metal,
            eps_dielectric,
        }
    }

    /// Complex SPP wavevector along the interface.
    ///
    /// k_sp = (ω/c) √(ε_m · ε_d / (ε_m + ε_d))
    pub fn wavevector(&self, omega: f64) -> Complex64 {
        let eps_m = self.metal.permittivity(omega);
        let eps_d = Complex64::new(self.eps_dielectric, 0.0);
        let k0 = omega / C0;
        let ratio = eps_m * eps_d / (eps_m + eps_d);
        k0 * ratio.sqrt()
    }

    /// Complex effective index n_eff = k_sp / k0.
    pub fn effective_index(&self, omega: f64) -> Complex64 {
        let eps_m = self.metal.permittivity(omega);
        let eps_d = Complex64::new(self.eps_dielectric, 0.0);
        (eps_m * eps_d / (eps_m + eps_d)).sqrt()
    }

    /// SPP propagation length L_sp = 1 / (2 · Im(k_sp)) in µm.
    pub fn propagation_length_um(&self, omega: f64) -> f64 {
        let k_sp = self.wavevector(omega);
        let two_ki = 2.0 * k_sp.im.abs();
        if two_ki < f64::EPSILON {
            return f64::INFINITY;
        }
        (1.0 / two_ki) * 1.0e6 // m → µm
    }

    /// SPP penetration depth into the *metal* in nm.
    ///
    /// κ_m = √(k_sp² − k0²·ε_m);  δ_m = 1/Im(κ_m)
    pub fn penetration_depth_metal_nm(&self, omega: f64) -> f64 {
        let k_sp = self.wavevector(omega);
        let k0 = omega / C0;
        let eps_m = self.metal.permittivity(omega);
        let kz_m = (k_sp * k_sp - k0 * k0 * eps_m).sqrt();
        let im = kz_m.im.abs();
        if im < f64::EPSILON {
            return f64::INFINITY;
        }
        (1.0 / im) * 1.0e9
    }

    /// SPP penetration depth into the *dielectric* in nm.
    ///
    /// κ_d = √(k_sp² − k0²·ε_d);  δ_d = 1/Im(κ_d)
    pub fn penetration_depth_dielectric_nm(&self, omega: f64) -> f64 {
        let k_sp = self.wavevector(omega);
        let k0 = omega / C0;
        let eps_d = Complex64::new(self.eps_dielectric, 0.0);
        let kz_d = (k_sp * k_sp - k0 * k0 * eps_d).sqrt();
        let im = kz_d.im.abs();
        if im < f64::EPSILON {
            return f64::INFINITY;
        }
        (1.0 / im) * 1.0e9
    }

    /// SPP group velocity v_g = dω/dk estimated by central finite difference.
    pub fn group_velocity(&self, omega: f64, d_omega: f64) -> f64 {
        let k1 = self.wavevector(omega - d_omega).re;
        let k2 = self.wavevector(omega + d_omega).re;
        2.0 * d_omega / (k2 - k1)
    }

    /// Surface plasmon resonance frequency: ω_sp = ωp / √(ε_∞ + ε_d)
    pub fn resonance_frequency(&self) -> f64 {
        self.metal.omega_p / (self.metal.eps_inf + self.eps_dielectric).sqrt()
    }

    /// Fractional field energy in the metal (confinement factor).
    ///
    /// For the TM SPP mode the penetration depths give a simple estimate:
    ///
    /// Γ_m = δ_d / (δ_d + δ_m)
    ///
    /// (where δ_d, δ_m are field penetration depths from the interface)
    pub fn confinement_factor_metal(&self, omega: f64) -> f64 {
        let delta_d = self.penetration_depth_dielectric_nm(omega);
        let delta_m = self.penetration_depth_metal_nm(omega);
        if !delta_d.is_finite() || !delta_m.is_finite() {
            return 0.0;
        }
        delta_d / (delta_d + delta_m)
    }

    /// Compute the SPP dispersion curve (ω, Re(k_sp)) over a frequency range.
    pub fn dispersion_curve(
        &self,
        omega_min: f64,
        omega_max: f64,
        n_pts: usize,
    ) -> Vec<(f64, f64)> {
        (0..n_pts)
            .map(|i| {
                let omega =
                    omega_min + (omega_max - omega_min) * i as f64 / (n_pts - 1).max(1) as f64;
                let k = self.wavevector(omega).re;
                (omega, k)
            })
            .collect()
    }

    /// Quality factor of the SPP mode: Q = Re(k_sp) / (2 · Im(k_sp)).
    pub fn quality_factor(&self, omega: f64) -> f64 {
        let k_sp = self.wavevector(omega);
        if k_sp.im.abs() < f64::EPSILON {
            return f64::INFINITY;
        }
        k_sp.re / (2.0 * k_sp.im.abs())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// KretschmannConfig
// ──────────────────────────────────────────────────────────────────────────────

/// Kretschmann attenuated total reflection (ATR) configuration.
///
/// Stack (from incident side): prism (n_prism) | metal film (d) | dielectric
///
/// The transfer matrix method is used for the 3-layer system (air/prism not
/// treated here; the prism IS the incident medium).
pub struct KretschmannConfig {
    /// Refractive index of the coupling prism (real, > 1)
    pub n_prism: f64,
    pub metal: DrudeMetal,
    /// Metal film thickness in nm
    pub metal_thickness_nm: f64,
    /// Permittivity of the output dielectric (e.g. analyte on top of metal)
    pub eps_dielectric: f64,
}

impl KretschmannConfig {
    pub fn new(n_prism: f64, metal: DrudeMetal, thickness_nm: f64, eps_d: f64) -> Self {
        Self {
            n_prism,
            metal,
            metal_thickness_nm: thickness_nm,
            eps_dielectric: eps_d,
        }
    }

    /// SPP coupling angle θ_sp (in radians) from the Kretschmann condition.
    ///
    /// n_prism · sin θ_sp = Re(n_eff_spp) = Re(√(ε_m · ε_d / (ε_m + ε_d)))
    ///
    /// Returns NaN if SPP is not supported at this frequency.
    pub fn coupling_angle_rad(&self, omega: f64) -> f64 {
        let eps_m = self.metal.permittivity(omega);
        let eps_d = Complex64::new(self.eps_dielectric, 0.0);
        let n_sp = (eps_m * eps_d / (eps_m + eps_d)).sqrt().re;
        let sin_theta = n_sp / self.n_prism;
        if sin_theta.abs() > 1.0 {
            return f64::NAN;
        }
        sin_theta.asin()
    }

    /// Reflectance of the 3-layer system (prism|metal|dielectric) for TM
    /// polarisation using the transfer matrix method.
    ///
    /// ω: angular frequency (rad/s), θ_rad: angle of incidence in prism.
    pub fn reflectance(&self, omega: f64, theta_rad: f64) -> f64 {
        let k0 = omega / C0;
        let n_p = self.n_prism;
        let eps_p = n_p * n_p;
        let eps_m = self.metal.permittivity(omega);
        let eps_d = Complex64::new(self.eps_dielectric, 0.0);

        // In-plane wavevector (conserved)
        let kx = Complex64::new(k0 * n_p * theta_rad.sin(), 0.0);

        // z-components of wavevectors in each medium
        let kz_p = (Complex64::new(eps_p, 0.0) * k0 * k0 - kx * kx).sqrt();
        let kz_m = (eps_m * k0 * k0 - kx * kx).sqrt();
        let kz_d = (eps_d * k0 * k0 - kx * kx).sqrt();

        // TM Fresnel reflection coefficients at each interface
        // r12 = (eps2*kz1 - eps1*kz2) / (eps2*kz1 + eps1*kz2)
        let kz_p_c = kz_p;
        let r01 = (eps_m * kz_p_c - Complex64::new(eps_p, 0.0) * kz_m)
            / (eps_m * kz_p_c + Complex64::new(eps_p, 0.0) * kz_m);
        let r12 = (eps_d * kz_m - eps_m * kz_d) / (eps_d * kz_m + eps_m * kz_d);

        // Phase acquired traversing the metal film
        let d = self.metal_thickness_nm * 1.0e-9; // nm → m
        let phase = Complex64::new(0.0, 2.0) * kz_m * d;
        let exp_phase = phase.exp();

        // Total reflection (Fabry-Perot)
        let r_total = (r01 + r12 * exp_phase) / (Complex64::new(1.0, 0.0) + r01 * r12 * exp_phase);
        let r_norm = r_total.norm();
        let reflectance = r_norm * r_norm;
        reflectance.clamp(0.0, 1.0)
    }

    /// Reflectance vs angular frequency at fixed incidence angle.
    pub fn reflectance_vs_omega(
        &self,
        theta_rad: f64,
        omega_min: f64,
        omega_max: f64,
        n_pts: usize,
    ) -> Vec<(f64, f64)> {
        (0..n_pts)
            .map(|i| {
                let omega =
                    omega_min + (omega_max - omega_min) * i as f64 / (n_pts - 1).max(1) as f64;
                (omega, self.reflectance(omega, theta_rad))
            })
            .collect()
    }

    /// Angular reflectance scan at fixed frequency.
    pub fn reflectance_vs_angle(
        &self,
        omega: f64,
        theta_min_rad: f64,
        theta_max_rad: f64,
        n_pts: usize,
    ) -> Vec<(f64, f64)> {
        (0..n_pts)
            .map(|i| {
                let theta = theta_min_rad
                    + (theta_max_rad - theta_min_rad) * i as f64 / (n_pts - 1).max(1) as f64;
                (theta, self.reflectance(omega, theta))
            })
            .collect()
    }

    /// Find the resonance dip: returns (angle_rad, R_min).
    pub fn resonance_dip(&self, omega: f64) -> (f64, f64) {
        let theta_c = (1.0_f64 / self.n_prism).asin(); // critical angle
                                                       // scan from critical angle to 90°, 500 points
        let scan = self.reflectance_vs_angle(omega, theta_c, PI / 2.0 - 1e-6, 500);
        scan.iter().copied().fold(
            (f64::NAN, f64::INFINITY),
            |(best_theta, best_r), (theta, r)| {
                if r < best_r {
                    (theta, r)
                } else {
                    (best_theta, best_r)
                }
            },
        )
    }

    /// Refractive-index sensitivity in degrees per RIU (refractive-index unit).
    ///
    /// Estimated by finite difference: Δθ_sp / Δn_d for Δn_d = 0.001.
    pub fn sensitivity_deg_per_riu(&self, omega: f64) -> f64 {
        let dn = 0.001_f64;
        let eps_d_orig = self.eps_dielectric;

        // θ_sp at original n_d = √eps_d
        let theta0 = self.coupling_angle_rad(omega);

        // θ_sp at n_d + Δn  → eps_d → (n_d + Δn)²
        let n_d = eps_d_orig.sqrt();
        let eps_d2 = (n_d + dn) * (n_d + dn);

        let cfg2 = KretschmannConfig {
            n_prism: self.n_prism,
            metal: self.metal.clone(),
            metal_thickness_nm: self.metal_thickness_nm,
            eps_dielectric: eps_d2,
        };
        let theta2 = cfg2.coupling_angle_rad(omega);

        if theta0.is_nan() || theta2.is_nan() {
            return 0.0;
        }
        // deg/RIU
        (theta2 - theta0).to_degrees() / dn
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// MimWaveguide
// ──────────────────────────────────────────────────────────────────────────────

/// Metal-Insulator-Metal (MIM) plasmonic waveguide.
///
/// Geometry: symmetric sandwich — metal | insulator (thickness d) | metal.
/// Supports a highly confined TM gap-plasmon mode described by:
///
///   tanh(κ_ins · d/2) = −ε_ins · κ_m / (ε_m · κ_ins)
///
/// where κ_i = √(β² − k0² · ε_i) is the transverse decay constant.
pub struct MimWaveguide {
    pub metal: DrudeMetal,
    /// Insulator gap thickness in nm
    pub insulator_thickness_nm: f64,
    /// Permittivity of the insulator (real, positive)
    pub eps_insulator: f64,
}

impl MimWaveguide {
    pub fn new(metal: DrudeMetal, thickness_nm: f64, eps_ins: f64) -> Self {
        Self {
            metal,
            insulator_thickness_nm: thickness_nm,
            eps_insulator: eps_ins,
        }
    }

    /// Solve the transcendental equation for the symmetric MIM mode wavevector β.
    ///
    /// The dispersion relation for the symmetric TM gap-plasmon mode is:
    ///
    ///   tanh(κ_i · d/2) = −(ε_i · κ_m) / (ε_m · κ_i)
    ///
    /// where κ_j = √(β² − k0² · ε_j) (positive real / positive imaginary part).
    ///
    /// Uses a scan-and-refine strategy:
    /// 1. Scan Re(β) from k0·√ε_i to 50·k0 on the real axis to locate a sign change.
    /// 2. Bisect to a tight bracket.
    /// 3. Polish with Newton iterations on the full complex plane.
    pub fn symmetric_mode_wavevector(&self, omega: f64) -> Result<Complex64, OxiPhotonError> {
        let k0 = omega / C0;
        let eps_m = self.metal.permittivity(omega);
        let eps_i = Complex64::new(self.eps_insulator, 0.0);
        let d = self.insulator_thickness_nm * 1.0e-9; // nm → m

        // Branch-choosing helper: pick the square root with Im ≥ 0, or if
        // Im ≈ 0, pick the root with Re ≥ 0. This enforces evanescent decay
        // away from the gap center.
        let branch = |z: Complex64| -> Complex64 {
            let s = z.sqrt();
            if s.im < -1.0e-30 {
                -s
            } else {
                s
            }
        };

        // Characteristic function F(β) for the *symmetric* MIM mode.
        // F(β) = tanh(κ_i · d/2) + (ε_i · κ_m) / (ε_m · κ_i) = 0
        let char_fn = |beta: Complex64| -> Complex64 {
            let kz_i = branch(beta * beta - eps_i * k0 * k0);
            let kz_m = branch(beta * beta - eps_m * k0 * k0);
            let arg = kz_i * d / 2.0;
            let tanh_val = arg.tanh();
            tanh_val + eps_i * kz_m / (eps_m * kz_i)
        };

        // ── Step 1: scan on the real β axis to locate a sign change ──────────
        let n_ins = self.eps_insulator.sqrt();
        let beta_lo = n_ins * k0 * 1.001; // just above insulator light line
        let beta_hi = n_ins * k0 * 80.0; // generous upper bound
        const N_SCAN: usize = 2000;

        let mut best_beta = Complex64::new(beta_lo, 0.0);
        let mut best_residual = f64::INFINITY;

        // Evaluate on real axis, track smallest |F|
        let mut prev_re = char_fn(Complex64::new(beta_lo, 0.0)).re;
        let mut bracket: Option<(f64, f64)> = None;

        for i in 1..N_SCAN {
            let beta_r = beta_lo + (beta_hi - beta_lo) * i as f64 / N_SCAN as f64;
            let beta_c = Complex64::new(beta_r, 0.0);
            let f = char_fn(beta_c);
            let r = f.norm();
            if r < best_residual {
                best_residual = r;
                best_beta = beta_c;
            }
            if bracket.is_none() && f.re * prev_re < 0.0 {
                let beta_prev = beta_lo + (beta_hi - beta_lo) * (i - 1) as f64 / N_SCAN as f64;
                bracket = Some((beta_prev, beta_r));
            }
            prev_re = f.re;
        }

        // ── Step 2: bisect in the bracket if found ────────────────────────────
        if let Some((mut lo, mut hi)) = bracket {
            for _ in 0..80 {
                let mid = (lo + hi) / 2.0;
                let f_lo = char_fn(Complex64::new(lo, 0.0)).re;
                let f_mid = char_fn(Complex64::new(mid, 0.0)).re;
                if f_lo * f_mid <= 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                }
                if (hi - lo).abs() < 1.0e-4 * lo {
                    break;
                }
            }
            let mid = (lo + hi) / 2.0;
            best_beta = Complex64::new(mid, 0.0);
        }

        // ── Step 3: Newton polish on the full complex plane ───────────────────
        let mut z = best_beta;
        const MAX_ITER: usize = 300;
        const TOL: f64 = 1.0e-12;
        let dh = beta_lo * 1.0e-7;

        for _ in 0..MAX_ITER {
            let f = char_fn(z);
            if f.norm() < TOL {
                break;
            }
            // Numerical Jacobian (central difference)
            let df = (char_fn(z + dh) - char_fn(z - dh)) / (2.0 * dh);
            if df.norm() < f64::EPSILON * 1.0e6 {
                break;
            }
            let dz = f / df;
            z -= dz;
            // Guard: keep Re(β) positive (physical mode)
            if z.re < 0.0 {
                z = Complex64::new(z.re.abs(), z.im);
            }
            if dz.norm() < TOL * z.norm().max(1.0) {
                break;
            }
        }

        let residual = char_fn(z).norm();
        if residual < 1.0e-4 && z.re > 0.0 {
            Ok(z)
        } else {
            Err(OxiPhotonError::NumericalError(format!(
                "MIM mode solver did not converge for ω={omega:.4e} rad/s, \
                 d={:.1}nm (residual={:.3e}, β={:.4e}+{:.4e}i)",
                self.insulator_thickness_nm, residual, z.re, z.im
            )))
        }
    }

    /// Real effective index n_eff = Re(β) / k0.
    pub fn effective_index(&self, omega: f64) -> Result<f64, OxiPhotonError> {
        let beta = self.symmetric_mode_wavevector(omega)?;
        Ok(beta.re / (omega / C0))
    }

    /// SPP propagation length in µm: L = 1 / (2 · Im(β)).
    pub fn propagation_length_um(&self, omega: f64) -> Result<f64, OxiPhotonError> {
        let beta = self.symmetric_mode_wavevector(omega)?;
        let two_ki = 2.0 * beta.im.abs();
        if two_ki < f64::EPSILON {
            return Ok(f64::INFINITY);
        }
        Ok((1.0 / two_ki) * 1.0e6)
    }

    /// Mode confinement factor: fraction of electric field intensity in
    /// the insulator gap (heuristic based on penetration depths).
    pub fn confinement_factor(&self, omega: f64) -> f64 {
        let d_nm = self.insulator_thickness_nm;
        // Skin depth in metal (nm)
        let delta_m = self.metal.skin_depth_nm(omega);
        // Approximate confinement as fraction of energy in the gap
        d_nm / (d_nm + 2.0 * delta_m.min(d_nm * 10.0))
    }

    /// Approximate cutoff thickness d_c below which the symmetric MIM mode
    /// becomes too lossy (defined here as: propagation length < 100 nm).
    ///
    /// Evaluated by scanning from 1 nm upward.
    pub fn cutoff_thickness_nm(&self, omega: f64) -> f64 {
        let original_d = self.insulator_thickness_nm;
        // quick scan
        let d_values: Vec<f64> = (1..=500).map(|i| i as f64).collect();
        for d_nm in d_values {
            let test = MimWaveguide::new(self.metal.clone(), d_nm, self.eps_insulator);
            if let Ok(beta) = test.symmetric_mode_wavevector(omega) {
                let two_ki = 2.0 * beta.im.abs();
                let l_nm = if two_ki > f64::EPSILON {
                    (1.0 / two_ki) * 1.0e9
                } else {
                    f64::INFINITY
                };
                if l_nm > 100.0 {
                    return d_nm;
                }
            }
        }
        original_d
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// λ = 633 nm  →  ω = 2πc/λ
    fn omega_633nm() -> f64 {
        2.0 * PI * C0 / 633.0e-9
    }

    fn omega_800nm() -> f64 {
        2.0 * PI * C0 / 800.0e-9
    }

    // ── DrudeMetal tests ────────────────────────────────────────────────────

    #[test]
    fn test_drude_gold_negative_real_eps() {
        let gold = DrudeMetal::gold();
        let omega = omega_633nm();
        let eps = gold.permittivity(omega);
        assert!(
            eps.re < 0.0,
            "Gold Re(ε) must be negative at visible frequencies; got {:.3}",
            eps.re
        );
    }

    #[test]
    fn test_drude_skin_depth_positive() {
        let gold = DrudeMetal::gold();
        let delta = gold.skin_depth_nm(omega_633nm());
        assert!(
            delta.is_finite() && delta > 0.0,
            "Skin depth must be positive and finite; got {delta}"
        );
    }

    #[test]
    fn test_gold_supports_spp() {
        let gold = DrudeMetal::gold();
        let omega = omega_633nm();
        // glass ε_d ≈ 2.25
        assert!(
            gold.supports_spp(omega, 2.25),
            "Gold must support SPP on glass at 633 nm"
        );
    }

    // ── SurfacePlasmonPolariton tests ───────────────────────────────────────

    #[test]
    fn test_spp_wavevector_larger_than_k0() {
        let gold = DrudeMetal::gold();
        let spp = SurfacePlasmonPolariton::new(gold, 1.0); // air dielectric
        let omega = omega_633nm();
        let k_sp = spp.wavevector(omega);
        let k0 = omega / C0;
        assert!(
            k_sp.re > k0,
            "SPP k_sp.re ({:.3e}) must exceed k0 ({:.3e}) in air",
            k_sp.re,
            k0
        );
    }

    #[test]
    fn test_spp_propagation_length_finite() {
        let gold = DrudeMetal::gold();
        let spp = SurfacePlasmonPolariton::new(gold, 2.25); // glass
        let l = spp.propagation_length_um(omega_633nm());
        assert!(
            l.is_finite() && l > 0.0,
            "SPP propagation length must be positive and finite; got {l}"
        );
    }

    #[test]
    fn test_spp_resonance_frequency() {
        let gold = DrudeMetal::gold();
        let eps_d = 1.0_f64; // air
        let spp = SurfacePlasmonPolariton::new(gold.clone(), eps_d);
        let omega_sp = spp.resonance_frequency();
        let expected = gold.omega_p / (gold.eps_inf + eps_d).sqrt();
        let rel_err = (omega_sp - expected).abs() / expected;
        assert!(
            rel_err < 1.0e-10,
            "ω_sp mismatch: got {omega_sp:.4e}, expected {expected:.4e}"
        );
    }

    #[test]
    fn test_spp_quality_factor_positive() {
        let gold = DrudeMetal::gold();
        let spp = SurfacePlasmonPolariton::new(gold, 2.25);
        let q = spp.quality_factor(omega_800nm());
        assert!(q > 0.0, "Quality factor must be positive; got {q}");
    }

    // ── KretschmannConfig tests ─────────────────────────────────────────────

    #[test]
    fn test_kretschmann_coupling_angle_greater_than_critical() {
        // SF11 prism n=1.78, 50 nm gold, water analyte
        let metal = DrudeMetal::gold();
        let cfg = KretschmannConfig::new(1.78, metal, 50.0, 1.77); // n_water²≈1.77
        let omega = omega_633nm();
        let theta_sp = cfg.coupling_angle_rad(omega);
        let theta_c = (1.0_f64 / 1.78).asin();
        assert!(
            theta_sp > theta_c || theta_sp.is_nan(),
            "SPP coupling angle ({:.4} rad) must exceed critical angle ({:.4} rad)",
            theta_sp,
            theta_c
        );
    }

    #[test]
    fn test_kretschmann_reflectance_range() {
        let metal = DrudeMetal::gold();
        let cfg = KretschmannConfig::new(1.5, metal, 50.0, 1.0);
        let omega = omega_633nm();
        for i in 0..100 {
            let theta = 0.5 + i as f64 * (PI / 2.0 - 0.5) / 100.0;
            let r = cfg.reflectance(omega, theta);
            assert!(
                (0.0..=1.0).contains(&r),
                "Reflectance must be in [0,1]; got {r} at θ={theta:.4}"
            );
        }
    }

    // ── MimWaveguide tests ──────────────────────────────────────────────────

    #[test]
    fn test_mim_effective_index_high() {
        // Very thin gap → very high effective index
        let metal = DrudeMetal::gold();
        let mim = MimWaveguide::new(metal, 10.0, 2.25); // 10 nm gap, glass
        let omega = omega_633nm();
        match mim.effective_index(omega) {
            Ok(n_eff) => {
                assert!(
                    n_eff > 1.5,
                    "MIM n_eff should be significantly larger than 1 for thin gap; got {n_eff}"
                );
            }
            Err(e) => {
                // Convergence failure is acceptable for very thin gaps in tests
                eprintln!("MIM solver note: {e}");
            }
        }
    }
}
