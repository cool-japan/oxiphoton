use std::f64::consts::PI;

/// Directional coupler based on coupled mode theory.
///
/// Two parallel waveguides exchange power periodically:
///   P₁(z) = P₀ cos²(κ z)
///   P₂(z) = P₀ sin²(κ z)
///
/// where κ is the coupling coefficient (1/m) and the coupling length
/// L_c = π / (2κ) gives full power transfer.
#[derive(Debug, Clone)]
pub struct DirectionalCoupler {
    /// Coupling coefficient κ (1/m).
    pub kappa: f64,
    /// Propagation constant of each waveguide β (1/m).
    pub beta: f64,
    /// Coupling region length (m).
    pub length: f64,
}

impl DirectionalCoupler {
    /// Create a directional coupler.
    ///
    /// # Arguments
    /// - `kappa`: coupling coefficient (1/m); determines how fast power transfers
    /// - `beta`: propagation constant of each waveguide (assumes identical guides)
    /// - `length`: coupler length (m)
    pub fn new(kappa: f64, beta: f64, length: f64) -> Self {
        Self {
            kappa,
            beta,
            length,
        }
    }

    /// Create coupler from coupling length (length for full power transfer).
    ///
    /// L_c = π / (2κ)  →  κ = π / (2 L_c)
    pub fn from_coupling_length(lc: f64, beta: f64, length: f64) -> Self {
        let kappa = PI / (2.0 * lc);
        Self::new(kappa, beta, length)
    }

    /// Coupling length L_c = π / (2κ).
    pub fn coupling_length(&self) -> f64 {
        PI / (2.0 * self.kappa)
    }

    /// Through-port power ratio (input in port 1, output in port 1).
    /// P₁/P₀ = cos²(κ L)
    pub fn through_power(&self) -> f64 {
        (self.kappa * self.length).cos().powi(2)
    }

    /// Cross-port power ratio (input in port 1, output in port 2).
    /// P₂/P₀ = sin²(κ L)
    pub fn cross_power(&self) -> f64 {
        (self.kappa * self.length).sin().powi(2)
    }

    /// Power splitting ratio (cross / through).
    pub fn splitting_ratio(&self) -> f64 {
        self.cross_power() / self.through_power().max(1e-30)
    }

    /// Transfer matrix of the coupler for complex field amplitudes.
    ///
    /// Returns \[[a₁₁, a₁₂\], \[a₂₁, a₂₂\]] where:
    ///   \[E₁_out\]   \[a₁₁  a₁₂\] \[E₁_in\]
    ///   \[E₂_out\] = \[a₂₁  a₂₂\] \[E₂_in\]
    pub fn transfer_matrix(&self) -> [[num_complex::Complex64; 2]; 2] {
        use num_complex::Complex64;
        let kl = self.kappa * self.length;
        let cos_kl = kl.cos();
        let sin_kl = kl.sin();
        let phase = Complex64::new(0.0, self.beta * self.length).exp();
        // For identical waveguides (phase-matched):
        // T = exp(iβL) [[cos(κL), i·sin(κL)],
        //               [i·sin(κL), cos(κL)]]
        [
            [phase * cos_kl, phase * Complex64::new(0.0, sin_kl)],
            [phase * Complex64::new(0.0, sin_kl), phase * cos_kl],
        ]
    }

    /// Output fields given input complex fields \[E₁_in, E₂_in\].
    pub fn propagate(&self, e_in: [num_complex::Complex64; 2]) -> [num_complex::Complex64; 2] {
        let t = self.transfer_matrix();
        [
            t[0][0] * e_in[0] + t[0][1] * e_in[1],
            t[1][0] * e_in[0] + t[1][1] * e_in[1],
        ]
    }
}

/// 50/50 directional coupler (beam splitter).
///
/// Sets length = L_c / 2 so cos²(κ·L) = sin²(κ·L) = 0.5.
pub fn half_coupler(kappa: f64, beta: f64) -> DirectionalCoupler {
    let lc = PI / (2.0 * kappa);
    DirectionalCoupler::new(kappa, beta, lc / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn full_coupling_at_lc() {
        // At L = L_c, all power transfers from port 1 to port 2
        let kappa = 1000.0; // 1000 /m
        let lc = PI / (2.0 * kappa);
        let coupler = DirectionalCoupler::new(kappa, 1.0e7, lc);
        assert_relative_eq!(coupler.cross_power(), 1.0, max_relative = 1e-10);
        assert_relative_eq!(coupler.through_power(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn half_coupling() {
        // At L = L_c / 2, 50/50 splitting
        let kappa = 1000.0;
        let coupler = half_coupler(kappa, 1.0e7);
        assert_relative_eq!(coupler.through_power(), 0.5, max_relative = 1e-10);
        assert_relative_eq!(coupler.cross_power(), 0.5, max_relative = 1e-10);
    }

    #[test]
    fn power_conservation() {
        // P_through + P_cross = 1 (lossless)
        let kappa = 500.0;
        let coupler = DirectionalCoupler::new(kappa, 1.0e7, 1e-3);
        let total = coupler.through_power() + coupler.cross_power();
        assert_relative_eq!(total, 1.0, max_relative = 1e-10);
    }

    #[test]
    fn transfer_matrix_power_conservation() {
        use num_complex::Complex64;
        let kappa = 1000.0;
        let lc = PI / (2.0 * kappa);
        let coupler = DirectionalCoupler::new(kappa, 1.5e7, lc * 0.3);
        let e_in = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let e_out = coupler.propagate(e_in);
        let p_out = e_out[0].norm_sqr() + e_out[1].norm_sqr();
        assert_relative_eq!(p_out, 1.0, max_relative = 1e-10);
    }
}
