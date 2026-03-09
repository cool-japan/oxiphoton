//! Planar microcavity design: distributed Bragg reflectors and cavity modes.
//!
//! A semiconductor microcavity consists of:
//! - A top DBR (N pairs of high-n / low-n quarter-wave layers)
//! - A spacer (cavity) layer of thickness L_c ≈ λ_c / (2 n_s)
//! - One or more quantum wells at the antinode positions
//! - A bottom DBR
//!
//! The cavity confines the photon vertically to a length ~ λ, yielding a very
//! small effective mass m_eff ≈ ħ ω_c n²/(c² L_eff) ~ 10⁻⁵ m_e.
//!
//! # References
//! - C. Weisbuch & B. Vinter, "Quantum Semiconductor Structures", Academic Press 1991
//! - G. Björk & Y. Yamamoto, IEEE J. Quantum Electron. 27, 2386 (1991)
//! - R. Houdré et al., Phys. Rev. Lett. 73, 2043 (1994)

use std::f64::consts::PI;

/// Speed of light in vacuum (m/s)
const C_LIGHT: f64 = 2.997_924_58e8;

// ─── DbMirror ───────────────────────────────────────────────────────────────

/// Distributed Bragg Reflector: N quarter-wave layer pairs.
///
/// Each bilayer consists of a high-index (n_H, thickness λ₀/(4 n_H)) and a
/// low-index (n_L, thickness λ₀/(4 n_L)) layer tuned to the centre wavelength.
#[derive(Debug, Clone)]
pub struct DbMirror {
    /// High refractive index n_H (e.g., GaAs ≈ 3.6, TiO₂ ≈ 2.3).
    pub n_high: f64,
    /// Low refractive index n_L (e.g., AlAs ≈ 2.9, SiO₂ ≈ 1.46).
    pub n_low: f64,
    /// Number of high/low bilayer pairs N.
    pub n_pairs: usize,
    /// Design (centre) wavelength λ₀ (m).
    pub center_wavelength_m: f64,
}

impl DbMirror {
    /// Normal-incidence power reflectivity of the DBR.
    ///
    /// For a free-standing N-pair DBR (air/DBR/substrate), the reflectivity is:
    ///
    /// ```text
    /// R = [(n_H/n_L)^(2N) − 1]² / [(n_H/n_L)^(2N) + 1]²
    /// ```
    ///
    /// (exact for n_sub = n_air = 1, quarter-wave design).  In practice this is
    /// an upper bound; the formula is standard and approaches 1 exponentially.
    pub fn reflectivity(&self) -> f64 {
        let ratio = self.n_high / self.n_low;
        let rho_2n = ratio.powi(2 * self.n_pairs as i32);
        let r = ((rho_2n - 1.0) / (rho_2n + 1.0)).powi(2);
        r.min(1.0)
    }

    /// Effective penetration depth into the DBR.
    ///
    /// Photons evanescently penetrate the DBR by approximately:
    ///
    /// ```text
    /// L_pen = λ₀ / (4(n_H − n_L)) × tanh(N ln(n_H/n_L))
    /// ```
    ///
    /// For large N: L_pen → λ₀ / (4(n_H − n_L)).
    pub fn penetration_depth_m(&self) -> f64 {
        let dn = self.n_high - self.n_low;
        if dn <= 0.0 {
            return 0.0;
        }
        let l_pen_inf = self.center_wavelength_m / (4.0 * dn);
        let n_eff = self.n_pairs as f64 * (self.n_high / self.n_low).ln();
        l_pen_inf * n_eff.tanh()
    }

    /// Photon lifetime τ_ph (s) for the given cavity round-trip length.
    ///
    /// ```text
    /// τ_ph = 2 L_eff / (c × T)
    /// ```
    ///
    /// where T = 1 − R is the power transmission and L_eff includes both
    /// mirror penetration depths.
    pub fn photon_lifetime_s(&self, cavity_length_m: f64) -> f64 {
        let r = self.reflectivity();
        let t = 1.0 - r;
        if t < f64::EPSILON {
            return f64::INFINITY;
        }
        let l_eff = cavity_length_m + 2.0 * self.penetration_depth_m();
        2.0 * l_eff / (C_LIGHT * t)
    }

    /// Cavity quality factor Q = ω₀ τ_ph.
    pub fn q_factor(&self, cavity_length_m: f64) -> f64 {
        let omega_0 = 2.0 * PI * C_LIGHT / self.center_wavelength_m;
        omega_0 * self.photon_lifetime_s(cavity_length_m)
    }

    /// DBR stop-band width Δλ (m) centred on λ₀.
    ///
    /// ```text
    /// Δλ/λ₀ = (4/π) arcsin((n_H − n_L) / (n_H + n_L))
    /// ```
    pub fn stop_band_width_m(&self) -> f64 {
        let ratio = (self.n_high - self.n_low) / (self.n_high + self.n_low);
        let arcsin = ratio.clamp(-1.0, 1.0).asin();
        (4.0 / PI) * arcsin * self.center_wavelength_m
    }

    /// Quarter-wave optical thickness of the high-index layer (m).
    pub fn high_layer_thickness_m(&self) -> f64 {
        self.center_wavelength_m / (4.0 * self.n_high)
    }

    /// Quarter-wave optical thickness of the low-index layer (m).
    pub fn low_layer_thickness_m(&self) -> f64 {
        self.center_wavelength_m / (4.0 * self.n_low)
    }
}

// ─── PlanarMicrocavity ──────────────────────────────────────────────────────

/// Full planar microcavity: top DBR + spacer + quantum wells + bottom DBR.
///
/// The cavity supports a discrete set of longitudinal modes.  The fundamental
/// mode (m=1) is chosen when the spacer thickness equals λ_c / (2 n_s).
#[derive(Debug, Clone)]
pub struct PlanarMicrocavity {
    /// Top (output) distributed Bragg reflector.
    pub top_mirror: DbMirror,
    /// Bottom (back) distributed Bragg reflector.
    pub bottom_mirror: DbMirror,
    /// Spacer (cavity) layer physical thickness L_c (m).
    pub spacer_thickness_m: f64,
    /// Spacer layer refractive index n_s.
    pub spacer_n: f64,
    /// Fractional z-positions of quantum wells inside the cavity (0 = bottom, 1 = top).
    pub qw_positions: Vec<f64>,
}

impl PlanarMicrocavity {
    /// Cavity resonance wavelength λ_c = 2 n_s L_eff / m (m, with m=1).
    pub fn cavity_wavelength_m(&self) -> f64 {
        let l_eff = self.effective_length_m();
        2.0 * self.spacer_n * l_eff
    }

    /// Effective optical path length including mirror penetration depths.
    ///
    /// ```text
    /// L_eff = L_c + L_pen,top + L_pen,bot
    /// ```
    pub fn effective_length_m(&self) -> f64 {
        self.spacer_thickness_m
            + self.top_mirror.penetration_depth_m()
            + self.bottom_mirror.penetration_depth_m()
    }

    /// Combined cavity finesse from both mirrors.
    ///
    /// ```text
    /// F = π · (R_t R_b)^(1/4) / (1 − (R_t R_b)^(1/2))
    /// ```
    pub fn finesse(&self) -> f64 {
        let r_t = self.top_mirror.reflectivity();
        let r_b = self.bottom_mirror.reflectivity();
        let sqrt_r = (r_t * r_b).sqrt();
        if (1.0 - sqrt_r).abs() < f64::EPSILON {
            return f64::INFINITY;
        }
        PI * sqrt_r.sqrt() / (1.0 - sqrt_r)
    }

    /// Effective mode volume V = A_eff × L_eff (m³).
    ///
    /// For a planar cavity illuminated by a Gaussian beam with mode area A_eff = π w₀²/2,
    /// the mode volume is set by the beam waist.  Here we return the 1D volume
    /// (per unit area) using the effective cavity length — multiply by the beam area
    /// externally if needed.
    ///
    /// As a useful approximation (per λ² area), V ≈ (λ/n)² × L_eff.
    pub fn mode_volume_m3(&self) -> f64 {
        let lambda_n = self.top_mirror.center_wavelength_m / self.spacer_n;
        let l_eff = self.effective_length_m();
        lambda_n * lambda_n * l_eff
    }

    /// Total cavity Q factor from both mirrors.
    ///
    /// ```text
    /// 1/Q_total = 1/Q_top + 1/Q_bot
    /// ```
    ///
    /// where Q_top = F × ω₀ τ_ph,top.  Approximately:
    ///
    /// ```text
    /// Q_total = ω₀ L_eff / (c × (T_t + T_b)/2)
    /// ```
    pub fn total_q(&self) -> f64 {
        let lambda_c = self.top_mirror.center_wavelength_m;
        let omega_0 = 2.0 * PI * C_LIGHT / lambda_c;
        let r_t = self.top_mirror.reflectivity();
        let r_b = self.bottom_mirror.reflectivity();
        let t_eff = (1.0 - r_t + 1.0 - r_b) / 2.0;
        if t_eff < f64::EPSILON {
            return f64::INFINITY;
        }
        let l_eff = self.effective_length_m();
        omega_0 * l_eff / (C_LIGHT * t_eff)
    }

    /// Electric field intensity enhancement at a quantum-well position.
    ///
    /// The standing-wave pattern in the cavity gives a sinusoidal field profile.
    /// The enhancement factor η relative to the antinode (where η = 1) is:
    ///
    /// ```text
    /// η(z) = sin²(π z / L_c × m)    where m = 1 for the fundamental mode
    /// ```
    ///
    /// At the antinode (z/L_c = 0.5): η = 1.
    /// At the node (z/L_c = 0): η = 0.
    ///
    /// `qw_pos_fraction` — fractional position in [0, 1] (0 = bottom, 1 = top).
    pub fn qw_field_enhancement(&self, qw_pos_fraction: f64) -> f64 {
        let z_frac = qw_pos_fraction.clamp(0.0, 1.0);
        // Standing wave: E(z) ∝ sin(π z / L_c) for the fundamental mode
        (PI * z_frac).sin().powi(2)
    }

    /// Total coupling enhancement for all quantum wells.
    ///
    /// Sums the field enhancement factors at each QW position.
    /// Maximum coupling is achieved when all QWs are at antinodes (z/L_c = 0.5).
    pub fn total_qw_coupling_factor(&self) -> f64 {
        self.qw_positions
            .iter()
            .map(|&z| self.qw_field_enhancement(z))
            .sum()
    }

    /// Photon effective mass m_eff (kg).
    ///
    /// For a planar Fabry-Pérot microcavity, the in-plane dispersion gives:
    ///
    /// ```text
    /// m_eff = ħ ω₀ n_s² / c²
    /// ```
    ///
    /// This is typically ~ 10⁻⁵ m_e, making the polariton dispersion much
    /// lighter than the exciton.
    pub fn photon_effective_mass_kg(&self) -> f64 {
        const HBAR: f64 = 1.054_571_817e-34;
        let omega_0 = 2.0 * PI * C_LIGHT / self.top_mirror.center_wavelength_m;
        let l_eff = self.effective_length_m();
        // m_eff = ħ / (c × L_eff) × n_s ... more precisely:
        // From E_c(k) ≈ ħc k_z / n_s + ħ²k‖²/(2 m_c), m_c = ħ ω_c n_s² / c²
        HBAR * omega_0 * self.spacer_n * self.spacer_n / (C_LIGHT * C_LIGHT)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// GaAs/AlAs DBR at 850 nm, 25 pairs (top mirror of typical VCSEL).
    fn gaas_dbr_top() -> DbMirror {
        DbMirror {
            n_high: 3.6,    // GaAs
            n_low: 2.95,    // AlAs
            n_pairs: 25,
            center_wavelength_m: 850e-9,
        }
    }

    fn gaas_dbr_bot() -> DbMirror {
        DbMirror {
            n_high: 3.6,
            n_low: 2.95,
            n_pairs: 30,    // bottom mirror has more pairs → higher R
            center_wavelength_m: 850e-9,
        }
    }

    fn gaas_microcavity() -> PlanarMicrocavity {
        // λ/n_s cavity: L_c = λ₀ / (2 n_s) for fundamental mode
        let n_s = 3.6_f64;
        let lambda_0 = 850e-9_f64;
        let l_c = lambda_0 / (2.0 * n_s); // ~ 118 nm
        PlanarMicrocavity {
            top_mirror: gaas_dbr_top(),
            bottom_mirror: gaas_dbr_bot(),
            spacer_thickness_m: l_c,
            spacer_n: n_s,
            qw_positions: vec![0.5], // QW at antinode
        }
    }

    #[test]
    fn dbr_reflectivity_approaches_unity() {
        let dbr = gaas_dbr_top();
        let r = dbr.reflectivity();
        assert!(r > 0.99, "25-pair GaAs/AlAs DBR should have R > 99%, got R={:.4}", r);
        assert!(r <= 1.0, "Reflectivity cannot exceed 1, got {}", r);
    }

    #[test]
    fn penetration_depth_positive() {
        let dbr = gaas_dbr_top();
        let l_pen = dbr.penetration_depth_m();
        assert!(l_pen > 0.0 && l_pen.is_finite(), "Penetration depth should be finite positive, got {}", l_pen);
    }

    #[test]
    fn stop_band_width_physical() {
        let dbr = gaas_dbr_top();
        let bw = dbr.stop_band_width_m();
        // For GaAs/AlAs: Δλ/λ ≈ (4/π) arcsin(0.65/6.55) ≈ 0.082 → Δλ ≈ 70 nm
        assert!(bw > 20e-9 && bw < 200e-9, "Stop band width should be 20–200 nm, got {} nm", bw * 1e9);
    }

    #[test]
    fn microcavity_finesse_high() {
        let mc = gaas_microcavity();
        let f = mc.finesse();
        assert!(f > 100.0 && f.is_finite(), "GaAs microcavity finesse should be >100, got {}", f);
    }

    #[test]
    fn qw_at_antinode_maximum_coupling() {
        let mc = gaas_microcavity();
        let eta_antinode = mc.qw_field_enhancement(0.5);
        let eta_node = mc.qw_field_enhancement(0.0);
        assert!((eta_antinode - 1.0).abs() < 1e-10, "Antinode enhancement should be 1, got {}", eta_antinode);
        assert!(eta_node.abs() < 1e-10, "Node enhancement should be 0, got {}", eta_node);
    }

    #[test]
    fn effective_length_larger_than_spacer() {
        let mc = gaas_microcavity();
        let l_eff = mc.effective_length_m();
        assert!(l_eff > mc.spacer_thickness_m, "L_eff should exceed spacer thickness");
    }

    #[test]
    fn total_q_positive_finite() {
        let mc = gaas_microcavity();
        let q = mc.total_q();
        assert!(q > 0.0 && q.is_finite(), "Total Q should be finite positive, got {}", q);
        // GaAs microcavity Q ~ 1000–10000
        assert!(q > 100.0 && q < 1e8, "GaAs microcavity Q out of range: {}", q);
    }

    #[test]
    fn photon_effective_mass_small() {
        let mc = gaas_microcavity();
        let m_eff = mc.photon_effective_mass_kg();
        const ME: f64 = 9.109_383_701_5e-31;
        let ratio = m_eff / ME;
        // Should be ~ 10^-5 m_e for typical microcavity
        assert!(
            ratio > 1e-7 && ratio < 1e-3,
            "Photon effective mass fraction should be ~1e-5, got {:.2e}",
            ratio
        );
    }
}
