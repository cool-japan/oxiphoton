use crate::units::conversion::{EPSILON_0, Z0};

/// Convolutional PML (CPML) coefficients for one axis
///
/// Reference: Roden & Gedney, "Convolutional PML implementation using FDTD method",
/// Microwave and Optical Technology Letters, 27(5), 2000.
#[derive(Debug, Clone)]
pub struct Cpml {
    /// Number of PML cells on each side
    pub thickness: usize,
    /// b coefficients for E-field update (size = total_cells)
    pub b_e: Vec<f64>,
    /// c coefficients for E-field update (size = total_cells)
    pub c_e: Vec<f64>,
    /// b coefficients for H-field update (size = total_cells)
    pub b_h: Vec<f64>,
    /// c coefficients for H-field update (size = total_cells)
    pub c_h: Vec<f64>,
    /// kappa for E-field (size = total_cells)
    pub kappa_e: Vec<f64>,
    /// kappa for H-field (size = total_cells)
    pub kappa_h: Vec<f64>,
}

impl Cpml {
    /// Build CPML coefficients for a 1D axis of `total_cells` cells with `pml_cells` on each side.
    ///
    /// Parameters:
    /// - `total_cells`: total number of cells along this axis
    /// - `pml_cells`: number of PML cells on each side
    /// - `d`: cell spacing (m)
    /// - `dt`: time step (s)
    /// - `m`: polynomial grading order (typically 3-4)
    /// - `r0`: target reflection coefficient (e.g. 1e-8)
    pub fn new(total_cells: usize, pml_cells: usize, d: f64, dt: f64, m: f64, r0: f64) -> Self {
        let n = total_cells;
        let d_pml = pml_cells as f64 * d;

        // sigma_max from the polynomial grading relation (Taflove & Hagness §7.10.2)
        // σ_max = (m+1) * ln(1/R0) / (2 * η₀ * d_pml)  where η₀ = Z0 ≈ 377 Ω
        let sigma_max = -(m + 1.0) * r0.ln() / (2.0 * d_pml * Z0);
        let kappa_max = 1.0;
        // CFS-PML alpha grading: prevents DC-mode late-time instability.
        // Linearly graded from alpha_max (interior interface) to 0 (outer boundary).
        // alpha_max = 0.05 * sigma_max is a practical stable default.
        let alpha_max = 0.05 * sigma_max;

        let mut b_e = vec![1.0; n];
        let mut c_e = vec![0.0; n];
        let mut b_h = vec![1.0; n];
        let mut c_h = vec![0.0; n];
        let mut kappa_e = vec![1.0; n];
        let mut kappa_h = vec![1.0; n];

        // Left PML region: cells [0, pml_cells)
        for i in 0..pml_cells {
            // E-field at position (pml_cells - i - 0.5) from boundary
            let rho_e = (pml_cells - i) as f64 / pml_cells as f64;
            let sigma_e = sigma_max * rho_e.powf(m);
            let kappa = 1.0 + (kappa_max - 1.0) * rho_e.powf(m);
            let alpha = alpha_max * (1.0 - rho_e).max(0.0);
            let (b, c) = cpml_bc(sigma_e, kappa, alpha, dt, EPSILON_0);
            b_e[i] = b;
            c_e[i] = c;
            kappa_e[i] = kappa;

            // H-field at position (pml_cells - i - 0.5) from boundary
            let rho_h = (pml_cells as f64 - i as f64 - 0.5) / pml_cells as f64;
            let sigma_h = sigma_max * rho_h.powf(m);
            let kappa_h_val = 1.0 + (kappa_max - 1.0) * rho_h.powf(m);
            let alpha_h = alpha_max * (1.0 - rho_h).max(0.0);
            let (bh, ch) = cpml_bc(sigma_h, kappa_h_val, alpha_h, dt, EPSILON_0);
            b_h[i] = bh;
            c_h[i] = ch;
            kappa_h[i] = kappa_h_val;
        }

        // Right PML region: cells [n-pml_cells, n)
        for i in 0..pml_cells {
            let gi = n - pml_cells + i;

            let rho_e = (i + 1) as f64 / pml_cells as f64;
            let sigma_e = sigma_max * rho_e.powf(m);
            let kappa = 1.0 + (kappa_max - 1.0) * rho_e.powf(m);
            let alpha = alpha_max * (1.0 - rho_e).max(0.0);
            let (b, c) = cpml_bc(sigma_e, kappa, alpha, dt, EPSILON_0);
            b_e[gi] = b;
            c_e[gi] = c;
            kappa_e[gi] = kappa;

            let rho_h = (i as f64 + 0.5) / pml_cells as f64;
            let sigma_h = sigma_max * rho_h.powf(m);
            let kappa_h_val = 1.0 + (kappa_max - 1.0) * rho_h.powf(m);
            let alpha_h = alpha_max * (1.0 - rho_h).max(0.0);
            let (bh, ch) = cpml_bc(sigma_h, kappa_h_val, alpha_h, dt, EPSILON_0);
            b_h[gi] = bh;
            c_h[gi] = ch;
            kappa_h[gi] = kappa_h_val;
        }

        Self {
            thickness: pml_cells,
            b_e,
            c_e,
            b_h,
            c_h,
            kappa_e,
            kappa_h,
        }
    }
}

/// Compute CPML b and c coefficients
fn cpml_bc(sigma: f64, kappa: f64, alpha: f64, dt: f64, eps_or_mu: f64) -> (f64, f64) {
    let b = (-(sigma / kappa + alpha) * dt / eps_or_mu).exp();
    let denom = kappa * (sigma + kappa * alpha);
    let c = if denom.abs() < 1e-30 {
        0.0
    } else {
        sigma / denom * (b - 1.0)
    };
    (b, c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpml_interior_is_identity() {
        let pml = Cpml::new(100, 10, 10e-9, 1.67e-17, 3.5, 1e-8);
        // Interior cells (not in PML) should have b=1, c=0
        let mid = 50;
        assert!((pml.b_e[mid] - 1.0).abs() < 1e-12);
        assert!(pml.c_e[mid].abs() < 1e-12);
        assert!((pml.kappa_e[mid] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cpml_pml_cells_have_nonzero_coefficients() {
        let pml = Cpml::new(100, 10, 10e-9, 1.67e-17, 3.5, 1e-8);
        // PML cells should have b < 1 (damping)
        assert!(pml.b_e[0] < 1.0);
        assert!(pml.b_e[99] < 1.0);
    }
}
