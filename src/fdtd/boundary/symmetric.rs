//! Symmetric and antisymmetric boundary conditions for FDTD.
//!
//! Symmetry BCs exploit the mirror symmetry of a structure to reduce
//! the computational domain by 2× or 4×.
//!
//! For TE polarization (E_z, H_x, H_y in 2D):
//!   Perfect Electric Conductor (PEC) wall:  E_z = 0  at boundary (odd BC for E)
//!   Perfect Magnetic Conductor (PMC) wall:  H_t = 0  at boundary (even BC for E)
//!
//! For a symmetric structure:
//!   - Even modes: use PMC at mirror plane (E_z symmetric, H_n = 0)
//!   - Odd modes:  use PEC at mirror plane (E_z antisymmetric, E_z = 0)

/// Type of symmetric boundary condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetryBc {
    /// Perfect Electric Conductor: E_tangential = 0, H_normal = 0 at wall.
    /// Selects odd (antisymmetric) E-field modes.
    Pec,
    /// Perfect Magnetic Conductor: H_tangential = 0, E_normal = 0 at wall.
    /// Selects even (symmetric) E-field modes.
    Pmc,
}

impl SymmetryBc {
    /// Sign applied to field when mirroring across the symmetry plane.
    ///
    /// PEC: E is antisymmetric → sign = -1
    /// PMC: E is symmetric   → sign = +1
    pub fn mirror_sign(&self) -> f64 {
        match self {
            SymmetryBc::Pec => -1.0,
            SymmetryBc::Pmc => 1.0,
        }
    }

    /// Apply BC to a 1D field slice at index `i_wall`.
    ///
    /// Sets field\[i_wall\] = 0 for PEC (Dirichlet zero BC).
    /// For PMC, the field derivative is zero, which is handled implicitly.
    pub fn apply_1d(&self, field: &mut [f64], i_wall: usize) {
        match self {
            SymmetryBc::Pec => {
                if i_wall < field.len() {
                    field[i_wall] = 0.0;
                }
            }
            SymmetryBc::Pmc => {
                // PMC: enforce zero normal derivative by mirroring
                // field[-1] = +field[1] (ghost cell approach)
                // Implemented in update equations, nothing to zero here
            }
        }
    }

    /// Apply 1D symmetry BC to both left and right walls.
    pub fn apply_both_1d(&self, field: &mut [f64], i_left: usize, i_right: usize) {
        self.apply_1d(field, i_left);
        self.apply_1d(field, i_right);
    }
}

/// Symmetry BC configuration for a 2D FDTD grid.
#[derive(Debug, Clone, Copy)]
pub struct SymmetryBc2d {
    /// BC on x-left wall (None = no BC / absorbing)
    pub x_left: Option<SymmetryBc>,
    /// BC on x-right wall
    pub x_right: Option<SymmetryBc>,
    /// BC on y-bottom wall
    pub y_bottom: Option<SymmetryBc>,
    /// BC on y-top wall
    pub y_top: Option<SymmetryBc>,
}

impl SymmetryBc2d {
    /// No symmetry BCs on any wall.
    pub fn none() -> Self {
        Self {
            x_left: None,
            x_right: None,
            y_bottom: None,
            y_top: None,
        }
    }

    /// PEC on all four walls (metallic box).
    pub fn pec_box() -> Self {
        Self {
            x_left: Some(SymmetryBc::Pec),
            x_right: Some(SymmetryBc::Pec),
            y_bottom: Some(SymmetryBc::Pec),
            y_top: Some(SymmetryBc::Pec),
        }
    }

    /// PMC on x-walls (symmetric about y-axis), PEC on y-walls.
    pub fn even_x_pec_y() -> Self {
        Self {
            x_left: Some(SymmetryBc::Pmc),
            x_right: Some(SymmetryBc::Pmc),
            y_bottom: Some(SymmetryBc::Pec),
            y_top: Some(SymmetryBc::Pec),
        }
    }

    /// Apply all symmetry BCs to a 2D E-field (Ez component), stored as \[ix\]\[iy\] row-major.
    pub fn apply_ez(&self, ez: &mut [f64], nx: usize, ny: usize) {
        // x-left boundary: all iy at ix=0
        if let Some(bc) = self.x_left {
            for iy in 0..ny {
                bc.apply_1d(ez, iy); // ez[0*ny + iy] = ez[iy]
            }
        }
        // x-right boundary: all iy at ix=nx-1
        if let Some(bc) = self.x_right {
            for iy in 0..ny {
                bc.apply_1d(ez, (nx - 1) * ny + iy);
            }
        }
        // y-bottom boundary: all ix at iy=0
        if let Some(bc) = self.y_bottom {
            for ix in 0..nx {
                bc.apply_1d(ez, ix * ny);
            }
        }
        // y-top boundary: all ix at iy=ny-1
        if let Some(bc) = self.y_top {
            for ix in 0..nx {
                bc.apply_1d(ez, ix * ny + (ny - 1));
            }
        }
    }

    /// Number of walls with a BC applied.
    pub fn n_active_walls(&self) -> usize {
        [self.x_left, self.x_right, self.y_bottom, self.y_top]
            .iter()
            .filter(|b| b.is_some())
            .count()
    }
}

/// Mirror field across symmetry plane for even mode excitation.
///
/// For a 1D field of length n with mirror at index `i_mirror`:
/// - Even (PMC): field\[i_mirror - k\] = field\[i_mirror + k\]
/// - Odd (PEC): field\[i_mirror - k\] = -field\[i_mirror + k\]
pub fn apply_mirror_bc_1d(field: &mut [f64], i_mirror: usize, bc: SymmetryBc) {
    let sign = bc.mirror_sign();
    let n = field.len();
    let max_k = i_mirror.min(n - 1 - i_mirror);
    for k in 1..=max_k {
        let lo = i_mirror - k;
        let hi = i_mirror + k;
        if hi < n {
            // Mirror from right (TF region) to left (ghost region)
            field[lo] = sign * field[hi];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pec_zero_at_wall() {
        let mut field = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        SymmetryBc::Pec.apply_1d(&mut field, 2);
        assert_eq!(field[2], 0.0);
        assert_eq!(field[0], 1.0); // unchanged
    }

    #[test]
    fn pmc_noop_on_field() {
        let mut field = vec![1.0, 2.0, 3.0];
        SymmetryBc::Pmc.apply_1d(&mut field, 1);
        assert_eq!(field[1], 2.0); // unchanged (PMC handled in updates)
    }

    #[test]
    fn pec_mirror_sign_negative() {
        assert_eq!(SymmetryBc::Pec.mirror_sign(), -1.0);
    }

    #[test]
    fn pmc_mirror_sign_positive() {
        assert_eq!(SymmetryBc::Pmc.mirror_sign(), 1.0);
    }

    #[test]
    fn symmetry_bc_2d_none_no_walls() {
        let s = SymmetryBc2d::none();
        assert_eq!(s.n_active_walls(), 0);
    }

    #[test]
    fn symmetry_bc_2d_pec_box_4_walls() {
        let s = SymmetryBc2d::pec_box();
        assert_eq!(s.n_active_walls(), 4);
    }

    #[test]
    fn apply_ez_pec_zeros_boundary() {
        let nx = 4;
        let ny = 4;
        let mut ez = vec![1.0_f64; nx * ny];
        let bc = SymmetryBc2d::pec_box();
        bc.apply_ez(&mut ez, nx, ny);
        // Check x-left (ix=0): ez[0..ny] should be 0
        for (iy, val) in ez.iter().enumerate().take(ny) {
            assert_eq!(*val, 0.0, "x-left boundary at iy={iy}");
        }
        // Check y-bottom (iy=0): ez[ix*ny + 0] should be 0
        for ix in 0..nx {
            assert_eq!(ez[ix * ny], 0.0, "y-bottom boundary at ix={ix}");
        }
    }

    #[test]
    fn mirror_bc_1d_even() {
        let mut field = vec![0.0, 0.0, 0.0, 1.0, 2.0];
        apply_mirror_bc_1d(&mut field, 2, SymmetryBc::Pmc);
        // field[1] = +field[3] = 1.0
        // field[0] = +field[4] = 2.0
        assert_eq!(field[1], 1.0);
        assert_eq!(field[0], 2.0);
    }

    #[test]
    fn mirror_bc_1d_odd() {
        let mut field = vec![0.0, 0.0, 0.0, 1.0, 2.0];
        apply_mirror_bc_1d(&mut field, 2, SymmetryBc::Pec);
        assert_eq!(field[1], -1.0);
        assert_eq!(field[0], -2.0);
    }
}
