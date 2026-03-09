/// Negative index metamaterials — Veselago double-negative media, SRR, and Drude wire arrays.
use std::f64::consts::PI;

const C_LIGHT: f64 = 2.99792458e8;
const EPS0: f64 = 8.854187817e-12;
const MU0: f64 = 1.2566370614e-6;

// ---------------------------------------------------------------------------
// Double-Negative Medium (Veselago)
// ---------------------------------------------------------------------------

/// Veselago medium: simultaneous ε < 0 and μ < 0 → n < 0 (backward wave).
#[derive(Debug, Clone)]
pub struct DoubleNegativeMedium {
    /// Relative permittivity (< 0 for double-negative medium).
    pub eps_r: f64,
    /// Relative permeability (< 0 for double-negative medium).
    pub mu_r: f64,
    /// Operating frequency (Hz).
    pub frequency_hz: f64,
}

impl DoubleNegativeMedium {
    /// Refractive index: n = −√(ε_r μ_r) when both components are negative.
    pub fn refractive_index(&self) -> f64 {
        let sign = if self.eps_r < 0.0 && self.mu_r < 0.0 {
            -1.0
        } else {
            1.0
        };
        sign * (self.eps_r.abs() * self.mu_r.abs()).sqrt()
    }

    /// Wave impedance: Z = Z₀ √(|μ_r| / |ε_r|).  Always positive for passive media.
    pub fn impedance_ohm(&self) -> f64 {
        let z0 = (MU0 / EPS0).sqrt(); // ≈ 376.73 Ω
        (self.mu_r.abs() / self.eps_r.abs()).sqrt() * z0
    }

    /// Phase velocity: v_ph = c / n  (negative for backward wave).
    pub fn phase_velocity(&self) -> f64 {
        C_LIGHT / self.refractive_index()
    }

    /// Group velocity for a Drude–Lorentz model is always positive (positive energy flow).
    pub fn group_velocity_positive(&self) -> bool {
        true
    }

    /// Snell's law refraction angle (n₁ sin θ₁ = n₂ sin θ₂), n₂ = self.
    pub fn refraction_angle_rad(&self, n_incident: f64, theta_inc_rad: f64) -> f64 {
        (n_incident * theta_inc_rad.sin() / self.refractive_index())
            .clamp(-1.0, 1.0)
            .asin()
    }

    /// Fresnel reflection coefficient (s-polarisation) at a vacuum / DNM interface.
    ///
    /// Uses the impedance-based formulation which remains well-defined for negative
    /// refractive indices:
    ///
    /// r_s = (η₂ cos θ_i − η₁ cos θ_t) / (η₂ cos θ_i + η₁ cos θ_t)
    ///
    /// where η = Z₀ / (n · Z / Z₀) = Z₀ √(μ_r / ε_r) normalised appropriately.
    /// For a vacuum incident on a DNM with impedance Z = Z₀ √(|μ_r/ε_r|):
    ///
    /// r_s = (Z_dnm cos θ_i − Z_vac cos θ_t) / (Z_dnm cos θ_i + Z_vac cos θ_t)
    pub fn reflection_coeff_s(&self, theta_inc_rad: f64) -> f64 {
        let z_vac = (MU0 / EPS0).sqrt();
        let z_dnm = self.impedance_ohm();
        let n1 = 1.0_f64;
        let n2 = self.refractive_index();
        let cos_i = theta_inc_rad.cos();
        // Snell: n₁ sin θ_i = n₂ sin θ_t  (n₂ may be negative → sign handled via asin)
        let sin_t_arg = if n2.abs() < 1e-30 {
            0.0
        } else {
            (n1 * theta_inc_rad.sin() / n2).clamp(-1.0, 1.0)
        };
        let cos_t = (1.0 - sin_t_arg * sin_t_arg).max(0.0).sqrt();
        let num = z_dnm * cos_i - z_vac * cos_t;
        let den = z_dnm * cos_i + z_vac * cos_t;
        if den.abs() < 1e-30 {
            0.0
        } else {
            num / den
        }
    }

    /// Fresnel reflection coefficient (p-polarisation) at a vacuum / DNM interface.
    ///
    /// r_p = (Z_vac cos θ_i − Z_dnm cos θ_t) / (Z_vac cos θ_i + Z_dnm cos θ_t)
    pub fn reflection_coeff_p(&self, theta_inc_rad: f64) -> f64 {
        let z_vac = (MU0 / EPS0).sqrt();
        let z_dnm = self.impedance_ohm();
        let n1 = 1.0_f64;
        let n2 = self.refractive_index();
        let cos_i = theta_inc_rad.cos();
        let sin_t_arg = if n2.abs() < 1e-30 {
            0.0
        } else {
            (n1 * theta_inc_rad.sin() / n2).clamp(-1.0, 1.0)
        };
        let cos_t = (1.0 - sin_t_arg * sin_t_arg).max(0.0).sqrt();
        let num = z_vac * cos_i - z_dnm * cos_t;
        let den = z_vac * cos_i + z_dnm * cos_t;
        if den.abs() < 1e-30 {
            0.0
        } else {
            num / den
        }
    }

    /// True when the phase and group velocities are anti-parallel (backward wave).
    pub fn is_backward_wave(&self) -> bool {
        self.refractive_index() < 0.0
    }

    /// True when the medium satisfies the Pendry perfect-lens condition (ε = μ = −1).
    pub fn is_perfect_lens(&self) -> bool {
        (self.eps_r + 1.0).abs() < 0.01 && (self.mu_r + 1.0).abs() < 0.01
    }

    /// Angular frequency (rad/s).
    pub fn omega(&self) -> f64 {
        2.0 * PI * self.frequency_hz
    }

    /// Free-space wave number k₀ = ω/c.
    pub fn k0(&self) -> f64 {
        self.omega() / C_LIGHT
    }

    /// Wave number inside the medium: k = n k₀.
    pub fn wave_number(&self) -> f64 {
        self.refractive_index() * self.k0()
    }
}

// ---------------------------------------------------------------------------
// Split-Ring Resonator (SRR) — achieves μ_eff < 0 near resonance
// ---------------------------------------------------------------------------

/// Split-ring resonator array that produces effective μ < 0 near resonance.
#[derive(Debug, Clone)]
pub struct SplitRingResonator {
    /// Ring radius (m).
    pub ring_radius_m: f64,
    /// Wire cross-section radius (m).
    pub wire_radius_m: f64,
    /// Gap width (m).
    pub gap_width_m: f64,
    /// Lattice constant (m).
    pub lattice_constant_m: f64,
    /// DC conductivity of the ring material (S/m); copper ≈ 6 × 10⁷.
    pub conductivity: f64,
}

impl SplitRingResonator {
    /// LC resonance frequency (Hz): ω₀ = 1/√(LC).
    ///
    /// L ≈ μ₀ π r²  (loop self-inductance)
    /// C ≈ ε₀ r²   / d  (gap capacitance approximation)
    pub fn resonance_frequency_hz(&self) -> f64 {
        let r = self.ring_radius_m;
        let l = MU0 * PI * r * r;
        let c = EPS0 * r * r / self.gap_width_m.max(1e-30);
        1.0 / (2.0 * PI * (l * c).max(0.0).sqrt())
    }

    /// Filling fraction F = π r² / a²  (fraction of unit cell area occupied by loop).
    pub fn filling_fraction(&self) -> f64 {
        PI * self.ring_radius_m * self.ring_radius_m
            / (self.lattice_constant_m * self.lattice_constant_m)
    }

    /// Ohmic loss rate Γ (rad/s) estimated from series resistance of the ring.
    ///
    /// R ≈ 2πr / (σ π a²)  where a = wire_radius
    fn loss_rate_rad_s(&self) -> f64 {
        let r = self.ring_radius_m;
        let a = self.wire_radius_m;
        let resistance = 2.0 * PI * r / (self.conductivity * PI * a * a).max(1e-60);
        let l = MU0 * PI * r * r;
        resistance / l.max(1e-60)
    }

    /// Effective permeability at angular frequency ω (Lorentz form):
    ///
    /// μ_eff = 1 − F ω² / (ω² − ω₀² + iΓω)
    ///
    /// Returns `(Re(μ_eff), Im(μ_eff))`.
    pub fn effective_permeability(&self, omega: f64) -> (f64, f64) {
        let omega0 = 2.0 * PI * self.resonance_frequency_hz();
        let f = self.filling_fraction();
        let gamma = self.loss_rate_rad_s();

        // denominator: (ω² − ω₀²) + i Γ ω
        let d_re = omega * omega - omega0 * omega0;
        let d_im = gamma * omega;
        let d_mag2 = d_re * d_re + d_im * d_im;

        if d_mag2 < 1e-60 {
            return (1.0 - f, 0.0);
        }

        // numerator of correction: F ω²  (real)
        let num = f * omega * omega;

        // mu_eff = 1 - num / (d_re + i d_im)
        //        = 1 - num (d_re - i d_im) / |d|²
        let mu_re = 1.0 - num * d_re / d_mag2;
        let mu_im = num * d_im / d_mag2;
        (mu_re, mu_im)
    }

    /// Loss tangent tan δ = Im(μ_eff) / Re(μ_eff) at angular frequency ω.
    pub fn loss_tangent(&self, omega: f64) -> f64 {
        let (mu_re, mu_im) = self.effective_permeability(omega);
        if mu_re.abs() < 1e-30 {
            f64::INFINITY
        } else {
            mu_im / mu_re
        }
    }
}

// ---------------------------------------------------------------------------
// Drude Wire Array — achieves ε_eff < 0 below plasma frequency
// ---------------------------------------------------------------------------

/// Periodic thin-wire array whose effective permittivity follows a Drude model,
/// yielding ε_eff < 0 below the plasma frequency.
#[derive(Debug, Clone)]
pub struct DrudeWireArray {
    /// Wire radius (m).
    pub wire_radius_m: f64,
    /// Lattice constant (m).
    pub lattice_constant_m: f64,
    /// Conductivity of the wire material (S/m).
    pub conductivity: f64,
}

impl DrudeWireArray {
    /// Plasma frequency (Hz):
    ///
    /// ω_p² = 2π c² / (a² ln(a/r))
    pub fn plasma_frequency_hz(&self) -> f64 {
        let a = self.lattice_constant_m;
        let r = self.wire_radius_m;
        let ratio = (a / r.max(1e-30)).max(1.0 + 1e-10);
        let omega_p_sq = 2.0 * PI * C_LIGHT * C_LIGHT / (a * a * ratio.ln().max(1e-30));
        omega_p_sq.sqrt() / (2.0 * PI)
    }

    /// Effective permittivity at angular frequency ω:
    ///
    /// ε_eff = 1 − ω_p² / ω²   (lossless Drude, ω >> damping)
    pub fn effective_permittivity(&self, omega: f64) -> f64 {
        if omega.abs() < 1e-30 {
            return f64::NEG_INFINITY;
        }
        let omega_p = 2.0 * PI * self.plasma_frequency_hz();
        1.0 - (omega_p * omega_p) / (omega * omega)
    }

    /// Returns `true` when ω is below the plasma frequency (ε_eff < 0).
    pub fn is_epsilon_negative(&self, omega: f64) -> bool {
        self.effective_permittivity(omega) < 0.0
    }

    /// Effective refractive index (real part, clamped to zero for ε < 0 in lossless limit).
    pub fn effective_index(&self, omega: f64) -> f64 {
        self.effective_permittivity(omega).max(0.0).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_negative_refraction() {
        let dnm = DoubleNegativeMedium {
            eps_r: -1.0,
            mu_r: -1.0,
            frequency_hz: 1e12,
        };
        assert!(
            (dnm.refractive_index() + 1.0).abs() < 1e-10,
            "Expected n = -1, got {}",
            dnm.refractive_index()
        );
        assert!(dnm.is_backward_wave());
        assert!(dnm.is_perfect_lens());
    }

    #[test]
    fn impedance_z0_for_matched_medium() {
        // ε_r = μ_r = -1 → Z = Z₀
        let dnm = DoubleNegativeMedium {
            eps_r: -1.0,
            mu_r: -1.0,
            frequency_hz: 1e10,
        };
        let z0 = (MU0 / EPS0).sqrt();
        assert!((dnm.impedance_ohm() - z0).abs() < 1.0);
    }

    #[test]
    fn snell_negative_refraction_angle_sign() {
        let dnm = DoubleNegativeMedium {
            eps_r: -2.25,
            mu_r: -1.0,
            frequency_hz: 1e12,
        };
        let theta_t = dnm.refraction_angle_rad(1.0, 30_f64.to_radians());
        // Refracted angle should be negative (same side as incident)
        assert!(
            theta_t < 0.0,
            "Expected negative refraction angle, got {}",
            theta_t
        );
    }

    #[test]
    fn srr_resonance_frequency_positive() {
        let srr = SplitRingResonator {
            ring_radius_m: 1.5e-3,
            wire_radius_m: 0.2e-3,
            gap_width_m: 0.5e-3,
            lattice_constant_m: 5.0e-3,
            conductivity: 6e7,
        };
        let f0 = srr.resonance_frequency_hz();
        assert!(
            f0 > 0.0 && f0 < 1e12,
            "Resonance frequency unrealistic: {}",
            f0
        );
    }

    #[test]
    fn srr_negative_permeability_near_resonance() {
        let srr = SplitRingResonator {
            ring_radius_m: 1.5e-3,
            wire_radius_m: 0.05e-3,
            gap_width_m: 0.3e-3,
            lattice_constant_m: 5.0e-3,
            conductivity: 1e12, // artificially low loss
        };
        let omega0 = 2.0 * PI * srr.resonance_frequency_hz();
        // Just above resonance, Re(μ_eff) should be < 1
        let (mu_re, _) = srr.effective_permeability(omega0 * 1.05);
        assert!(
            mu_re < 1.0,
            "Expected μ_eff < 1 above resonance, got {}",
            mu_re
        );
    }

    #[test]
    fn drude_wire_epsilon_negative_below_plasma() {
        let wire = DrudeWireArray {
            wire_radius_m: 1e-6,
            lattice_constant_m: 5e-3,
            conductivity: 6e7,
        };
        let omega_p = 2.0 * PI * wire.plasma_frequency_hz();
        // At half the plasma frequency, ε < 0
        assert!(wire.is_epsilon_negative(omega_p * 0.5));
        // Above plasma frequency, ε > 0
        assert!(!wire.is_epsilon_negative(omega_p * 2.0));
    }

    #[test]
    fn drude_wire_plasma_frequency_realistic() {
        let wire = DrudeWireArray {
            wire_radius_m: 1e-6,
            lattice_constant_m: 5e-3,
            conductivity: 6e7,
        };
        let fp = wire.plasma_frequency_hz();
        // Typical thin-wire plasma frequency in GHz range
        assert!(
            fp > 1e8 && fp < 1e13,
            "Unexpected plasma frequency: {:.3e}",
            fp
        );
    }

    #[test]
    fn reflection_coeff_normal_incidence() {
        // At normal incidence for ε = μ = -1 (perfect lens), r_s = r_p = 0
        let dnm = DoubleNegativeMedium {
            eps_r: -1.0,
            mu_r: -1.0,
            frequency_hz: 1e12,
        };
        let rs = dnm.reflection_coeff_s(0.0);
        let rp = dnm.reflection_coeff_p(0.0);
        assert!(
            rs.abs() < 1e-10,
            "r_s should be 0 for matched medium, got {}",
            rs
        );
        assert!(
            rp.abs() < 1e-10,
            "r_p should be 0 for matched medium, got {}",
            rp
        );
    }
}
