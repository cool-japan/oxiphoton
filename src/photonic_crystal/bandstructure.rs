use std::f64::consts::PI;

/// Plane-wave expansion (PWE) method for 1D photonic crystal band structure.
///
/// For a 1D periodic structure with period Λ, the Maxwell eigenvalue problem
/// in the plane-wave basis reduces to:
///
///   Σ_G' [|k+G|² δ_{GG'} / ε(G-G')] e_{G'} = (ω/c)² e_G
///
/// where G = m·(2π/Λ) are reciprocal lattice vectors and ε(G) are Fourier
/// coefficients of the permittivity profile.
///
/// For 1D binary (two-layer) crystals, the analytic band structure can also
/// be obtained from the transfer matrix dispersion relation.
///
/// Reference: Joannopoulos et al., "Photonic Crystals", 2nd ed., Ch. 2.
///
/// Fourier coefficients of the inverse permittivity 1/ε(x) for a 1D binary crystal.
///
/// Layer 1: ε₁, width d₁. Layer 2: ε₂, width d₂ = Λ - d₁.
/// f = d₁/Λ (fill factor of layer 1).
pub fn inverse_eps_fourier(eps1: f64, eps2: f64, fill: f64, n_orders: usize) -> Vec<f64> {
    let n = 2 * n_orders + 1;
    let f = fill;
    let inv_eps1 = 1.0 / eps1;
    let inv_eps2 = 1.0 / eps2;

    (0..n)
        .map(|k| {
            let m = k as i64 - n_orders as i64;
            if m == 0 {
                f * inv_eps1 + (1.0 - f) * inv_eps2
            } else {
                // Fourier coeff: (inv_eps1 - inv_eps2) * f * sinc(m*f)
                let arg = PI * m as f64 * f;
                let sinc = if arg.abs() < 1e-12 {
                    1.0
                } else {
                    arg.sin() / arg
                };
                (inv_eps1 - inv_eps2) * f * sinc
            }
        })
        .collect()
}

/// 1D photonic crystal band structure via the transfer matrix dispersion relation.
///
/// For a 1D layered medium (period Λ = d₁ + d₂):
///   cos(k_B·Λ) = cos(k₁d₁)cos(k₂d₂) - ½(Z₁/Z₂ + Z₂/Z₁)sin(k₁d₁)sin(k₂d₂)
///
/// where k₁ = n₁·ω/c, k₂ = n₂·ω/c, Z_i = n_i⁻¹ (for TE waves).
/// This gives the dispersion relation k_B(ω) for allowed/forbidden bands.
#[derive(Debug, Clone)]
pub struct PhCrystal1d {
    /// Period length (m)
    pub period: f64,
    /// Index of layer 1
    pub n1: f64,
    /// Thickness of layer 1 (m)
    pub d1: f64,
    /// Index of layer 2
    pub n2: f64,
    /// Thickness of layer 2 (m)
    pub d2: f64,
}

impl PhCrystal1d {
    pub fn new(n1: f64, d1: f64, n2: f64, d2: f64) -> Self {
        Self {
            period: d1 + d2,
            n1,
            d1,
            n2,
            d2,
        }
    }

    /// Quarter-wave stack (optimal for high-reflectance mirror):
    /// d_i = λ_c / (4·n_i)
    pub fn quarter_wave(n1: f64, n2: f64, lambda_c: f64) -> Self {
        let d1 = lambda_c / (4.0 * n1);
        let d2 = lambda_c / (4.0 * n2);
        Self::new(n1, d1, n2, d2)
    }

    /// Compute the dispersion relation: cos(k_B·Λ) at a given normalized frequency ω·Λ/c.
    ///
    /// Returns cos(k_B·Λ). Values in [-1, 1] → propagating (allowed band).
    /// Values outside → band gap (evanescent).
    pub fn cos_kbloch(&self, omega: f64) -> f64 {
        use crate::units::conversion::SPEED_OF_LIGHT;
        let k1 = self.n1 * omega / SPEED_OF_LIGHT;
        let k2 = self.n2 * omega / SPEED_OF_LIGHT;
        let z1 = 1.0 / self.n1;
        let z2 = 1.0 / self.n2;
        let cos1 = (k1 * self.d1).cos();
        let cos2 = (k2 * self.d2).cos();
        let sin1 = (k1 * self.d1).sin();
        let sin2 = (k2 * self.d2).sin();
        cos1 * cos2 - 0.5 * (z1 / z2 + z2 / z1) * sin1 * sin2
    }

    /// Check if frequency ω is in a photonic band gap (|cos(k_B·Λ)| > 1).
    pub fn is_band_gap(&self, omega: f64) -> bool {
        self.cos_kbloch(omega).abs() > 1.0
    }

    /// Compute the Bloch wavevector k_B at given frequency ω (if in allowed band).
    /// Returns None if in band gap.
    pub fn bloch_k(&self, omega: f64) -> Option<f64> {
        let cos_kb = self.cos_kbloch(omega);
        if cos_kb.abs() > 1.0 {
            None
        } else {
            Some(cos_kb.acos() / self.period)
        }
    }

    /// Scan frequencies and return allowed bands as (omega_start, omega_end) pairs.
    ///
    /// `n_scan`: number of frequency points to scan over [0, omega_max].
    pub fn find_bands(&self, omega_max: f64, n_scan: usize) -> Vec<(f64, f64)> {
        let domega = omega_max / n_scan as f64;
        let mut bands = Vec::new();
        let mut in_band = false;
        let mut band_start = 0.0;

        for i in 0..=n_scan {
            let omega = i as f64 * domega;
            let allowed = !self.is_band_gap(omega);
            if allowed && !in_band {
                band_start = omega;
                in_band = true;
            } else if !allowed && in_band {
                bands.push((band_start, omega - domega));
                in_band = false;
            }
        }
        if in_band {
            bands.push((band_start, omega_max));
        }
        bands
    }

    /// Compute the photonic density of states (DoS) via dω/dk from dispersion.
    ///
    /// DoS(ω) ∝ 1 / |dω/dk|. Returns (omega, dos) pairs.
    pub fn density_of_states(&self, omega_max: f64, n_pts: usize) -> Vec<(f64, f64)> {
        let domega = omega_max / n_pts as f64;
        let mut result = Vec::with_capacity(n_pts);
        for i in 1..n_pts {
            let omega = i as f64 * domega;
            // Numerical derivative dk/dω
            let cos_prev = self.cos_kbloch(omega - domega);
            let cos_next = self.cos_kbloch(omega + domega);
            if cos_prev.abs() <= 1.0 && cos_next.abs() <= 1.0 {
                let k_prev = cos_prev.acos() / self.period;
                let k_next = cos_next.acos() / self.period;
                let dk_domega = (k_next - k_prev) / (2.0 * domega);
                let dos = dk_domega.abs();
                result.push((omega, dos));
            } else {
                result.push((omega, 0.0)); // in gap
            }
        }
        result
    }
}

/// Band gap center frequency and width for a quarter-wave stack.
///
/// For a quarter-wave stack: ω_c = π·c/(2·Λ), gap_width = 4ω_c/π · arcsin(|n₁-n₂|/(n₁+n₂))
pub fn quarter_wave_gap(n1: f64, n2: f64, period: f64) -> (f64, f64) {
    use crate::units::conversion::SPEED_OF_LIGHT;
    let omega_c = PI * SPEED_OF_LIGHT / (2.0 * period);
    let gap_frac = 4.0 / PI * ((n1 - n2).abs() / (n1 + n2)).asin();
    let gap_width = gap_frac * omega_c;
    (omega_c, gap_width)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::conversion::SPEED_OF_LIGHT;

    #[test]
    fn quarter_wave_stack_has_band_gap() {
        // SiO2/TiO2 quarter-wave stack at 550nm
        let pc = PhCrystal1d::quarter_wave(1.46, 2.35, 550e-9);
        // At center frequency, should be in gap
        let omega_c = PI * SPEED_OF_LIGHT / (2.0 * pc.period);
        assert!(
            pc.is_band_gap(omega_c),
            "Quarter-wave stack should have gap at center"
        );
    }

    #[test]
    fn uniform_medium_no_gap() {
        // Uniform medium (n1=n2): no photonic band gap
        let pc = PhCrystal1d::new(1.5, 100e-9, 1.5, 100e-9);
        let omega = 2.0 * PI * SPEED_OF_LIGHT / 800e-9;
        assert!(!pc.is_band_gap(omega), "Uniform medium should have no gap");
    }

    #[test]
    fn bloch_k_returns_none_in_gap() {
        let pc = PhCrystal1d::quarter_wave(1.46, 2.35, 550e-9);
        let omega_c = PI * SPEED_OF_LIGHT / (2.0 * pc.period);
        let k = pc.bloch_k(omega_c);
        assert!(k.is_none(), "Should return None in band gap");
    }

    #[test]
    fn bloch_k_returns_some_in_band() {
        let pc = PhCrystal1d::quarter_wave(1.46, 2.35, 550e-9);
        // Low frequency (deep in first band)
        let omega_low = 0.1 * PI * SPEED_OF_LIGHT / pc.period;
        let k = pc.bloch_k(omega_low);
        assert!(k.is_some(), "Should find Bloch k in allowed band");
    }

    #[test]
    fn inverse_eps_fourier_dc_term() {
        let eps1 = 1.46 * 1.46;
        let eps2 = 2.35 * 2.35;
        let fill = 0.5;
        let coeffs = inverse_eps_fourier(eps1, eps2, fill, 3);
        let dc = coeffs[3]; // m=0
        let expected = 0.5 / eps1 + 0.5 / eps2;
        assert!((dc - expected).abs() < 1e-8);
    }

    #[test]
    fn quarter_wave_gap_formula() {
        let n1 = 1.46;
        let n2 = 2.35;
        let period = 200e-9;
        let (omega_c, gap_width) = quarter_wave_gap(n1, n2, period);
        assert!(omega_c > 0.0);
        assert!(gap_width > 0.0);
        assert!(
            gap_width < omega_c,
            "Gap width should be < center frequency"
        );
    }

    #[test]
    fn find_bands_returns_at_least_one_band() {
        let pc = PhCrystal1d::quarter_wave(1.46, 2.35, 550e-9);
        let omega_max = 6.0 * PI * SPEED_OF_LIGHT / pc.period;
        let bands = pc.find_bands(omega_max, 2000);
        assert!(!bands.is_empty(), "Should find at least one allowed band");
    }

    #[test]
    fn density_of_states_nonnegative() {
        let pc = PhCrystal1d::quarter_wave(1.46, 2.35, 550e-9);
        let omega_max = 3.0 * PI * SPEED_OF_LIGHT / pc.period;
        let dos = pc.density_of_states(omega_max, 100);
        assert!(dos.iter().all(|(_, d)| *d >= 0.0));
    }
}
