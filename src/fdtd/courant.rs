use crate::fdtd::config::{Dimensions, GridSpacing};
use crate::units::conversion::SPEED_OF_LIGHT;

/// Compute maximum stable Courant time step for the given grid
pub fn courant_dt(dims: Dimensions, spacing: GridSpacing, n_min: f64) -> f64 {
    let c = SPEED_OF_LIGHT / n_min;
    match dims {
        Dimensions::OneD { .. } => spacing.dz / c,
        Dimensions::TwoD { .. } => {
            let inv_sum = 1.0 / (spacing.dx * spacing.dx) + 1.0 / (spacing.dy * spacing.dy);
            1.0 / (c * inv_sum.sqrt())
        }
        Dimensions::ThreeD { .. } => {
            let inv_sum = 1.0 / (spacing.dx * spacing.dx)
                + 1.0 / (spacing.dy * spacing.dy)
                + 1.0 / (spacing.dz * spacing.dz);
            1.0 / (c * inv_sum.sqrt())
        }
    }
}

/// Courant stability number S = c * dt / dx (should be <= 1 for 1D)
pub fn courant_number(dt: f64, dx: f64, n_min: f64) -> f64 {
    SPEED_OF_LIGHT / n_min * dt / dx
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn courant_1d_vacuum() {
        let spacing = GridSpacing::uniform(10e-9); // 10nm
        let dt = courant_dt(Dimensions::OneD { nz: 100 }, spacing, 1.0);
        let s = courant_number(dt, spacing.dz, 1.0);
        // For 1D: S = 1 exactly at Courant limit
        assert_relative_eq!(s, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn courant_2d_vacuum() {
        let spacing = GridSpacing::uniform(10e-9);
        let dt = courant_dt(Dimensions::TwoD { nx: 100, ny: 100 }, spacing, 1.0);
        let s = courant_number(dt, spacing.dx, 1.0);
        // For 2D: S = 1/sqrt(2) ~ 0.707
        assert_relative_eq!(s, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-12);
    }
}
