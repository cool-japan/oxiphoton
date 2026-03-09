//! Symmetry operations and exploitation for computational reduction.
//!
//! Symmetry reduces problem size by restricting computations to a fundamental
//! domain and applying boundary conditions:
//!
//!   - Mirror (reflection): E_x(x,y) = ±E_x(-x,y) depending on field parity
//!   - Rotational: C₄ symmetry in square lattices
//!   - Translational: periodic boundary conditions
//!   - Time-reversal: real vs complex field representations
//!
//! Computational savings:
//!   - 1 mirror plane → 2× speed-up
//!   - 2 mirror planes → 4× speed-up
//!   - C₄ rotation → 8× speed-up (with mirror)

/// Symmetry type for a field component or geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Symmetry {
    /// No symmetry — full domain required
    None,
    /// Even (symmetric) about the symmetry axis: f(-x) = +f(x)
    Even,
    /// Odd (antisymmetric) about the symmetry axis: f(-x) = -f(x)
    Odd,
}

/// Mirror symmetry plane (axis of reflection).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirrorPlane {
    /// Mirror in x (reflect about x = 0)
    X,
    /// Mirror in y (reflect about y = 0)
    Y,
    /// Mirror in z (reflect about z = 0)
    Z,
}

/// A symmetry group element for a 2D domain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Symmetry2d {
    /// Symmetry about the x-axis (x=0 plane, y-domain halved)
    pub x_sym: Symmetry,
    /// Symmetry about the y-axis (y=0 plane, x-domain halved)
    pub y_sym: Symmetry,
}

impl Symmetry2d {
    /// Full symmetry (even–even): halves domain in both x and y.
    pub fn even_even() -> Self {
        Self {
            x_sym: Symmetry::Even,
            y_sym: Symmetry::Even,
        }
    }

    /// Full antisymmetry (odd–odd).
    pub fn odd_odd() -> Self {
        Self {
            x_sym: Symmetry::Odd,
            y_sym: Symmetry::Odd,
        }
    }

    /// No symmetry.
    pub fn none() -> Self {
        Self {
            x_sym: Symmetry::None,
            y_sym: Symmetry::None,
        }
    }

    /// Domain reduction factor (1, 2, or 4).
    pub fn reduction_factor(&self) -> usize {
        let x = if self.x_sym != Symmetry::None { 2 } else { 1 };
        let y = if self.y_sym != Symmetry::None { 2 } else { 1 };
        x * y
    }

    /// Apply x-symmetry: reflect field value at index i across x=0.
    ///
    /// For an even field: f\[-i\] = +f\[i\]
    /// For an odd field:  f\[-i\] = -f\[i\]
    pub fn apply_x(&self, value: f64) -> f64 {
        match self.x_sym {
            Symmetry::Even => value,
            Symmetry::Odd => -value,
            Symmetry::None => value,
        }
    }

    /// Apply y-symmetry: reflect field value at index j across y=0.
    pub fn apply_y(&self, value: f64) -> f64 {
        match self.y_sym {
            Symmetry::Even => value,
            Symmetry::Odd => -value,
            Symmetry::None => value,
        }
    }
}

/// Point group symmetries for a 2D lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointGroup2d {
    /// C₁: no rotational symmetry
    C1,
    /// C₂: 180° rotation symmetry
    C2,
    /// C₄: 90° rotation symmetry (square lattice)
    C4,
    /// C₆: 60° rotation symmetry (hexagonal lattice)
    C6,
}

impl PointGroup2d {
    /// Number of equivalent k-points reduced by this symmetry.
    pub fn reduction_factor(&self) -> usize {
        match self {
            PointGroup2d::C1 => 1,
            PointGroup2d::C2 => 2,
            PointGroup2d::C4 => 4,
            PointGroup2d::C6 => 6,
        }
    }

    /// Irreducible Brillouin zone fraction.
    pub fn ibz_fraction(&self) -> f64 {
        1.0 / self.reduction_factor() as f64
    }

    /// Rotation angle in degrees.
    pub fn rotation_angle_deg(&self) -> f64 {
        match self {
            PointGroup2d::C1 => 360.0,
            PointGroup2d::C2 => 180.0,
            PointGroup2d::C4 => 90.0,
            PointGroup2d::C6 => 60.0,
        }
    }
}

/// Apply a 1D mirror symmetry to a field array.
///
/// Extends the half-domain `half` (size n) to the full domain (size 2n)
/// by appending the mirror image with the appropriate sign.
pub fn expand_mirror_1d(half: &[f64], sym: Symmetry) -> Vec<f64> {
    let sign = match sym {
        Symmetry::Even => 1.0,
        Symmetry::Odd => -1.0,
        Symmetry::None => return half.to_vec(),
    };
    let n = half.len();
    let mut full = Vec::with_capacity(2 * n);
    full.extend_from_slice(half);
    for i in (0..n).rev() {
        full.push(sign * half[i]);
    }
    full
}

/// Restrict a full 1D field to the symmetry half-domain.
///
/// Returns the first `n/2` elements (assuming the domain is centered).
pub fn restrict_mirror_1d(full: &[f64]) -> Vec<f64> {
    let n = full.len() / 2;
    full[..n].to_vec()
}

/// Enforce x-mirror symmetry on a 2D grid (in-place).
///
/// Grid is (nx × ny), stored row-major. Enforces field\[ix\]\[iy\] = ±field\[nx-1-ix\]\[iy\].
pub fn enforce_x_symmetry_2d(field: &mut [f64], nx: usize, ny: usize, sym: Symmetry) {
    if sym == Symmetry::None {
        return;
    }
    let sign = if sym == Symmetry::Even { 1.0 } else { -1.0 };
    for iy in 0..ny {
        for ix in 0..nx / 2 {
            let i_lo = ix * ny + iy;
            let i_hi = (nx - 1 - ix) * ny + iy;
            let avg = (field[i_lo] + sign * field[i_hi]) / 2.0;
            field[i_lo] = avg;
            field[i_hi] = sign * avg;
        }
    }
}

/// Enforce y-mirror symmetry on a 2D grid (in-place).
pub fn enforce_y_symmetry_2d(field: &mut [f64], nx: usize, ny: usize, sym: Symmetry) {
    if sym == Symmetry::None {
        return;
    }
    let sign = if sym == Symmetry::Even { 1.0 } else { -1.0 };
    for ix in 0..nx {
        for iy in 0..ny / 2 {
            let i_lo = ix * ny + iy;
            let i_hi = ix * ny + (ny - 1 - iy);
            let avg = (field[i_lo] + sign * field[i_hi]) / 2.0;
            field[i_lo] = avg;
            field[i_hi] = sign * avg;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetry2d_reduction_none() {
        let s = Symmetry2d::none();
        assert_eq!(s.reduction_factor(), 1);
    }

    #[test]
    fn symmetry2d_reduction_one_axis() {
        let s = Symmetry2d {
            x_sym: Symmetry::Even,
            y_sym: Symmetry::None,
        };
        assert_eq!(s.reduction_factor(), 2);
    }

    #[test]
    fn symmetry2d_reduction_both_axes() {
        let s = Symmetry2d::even_even();
        assert_eq!(s.reduction_factor(), 4);
    }

    #[test]
    fn expand_mirror_even() {
        let half = vec![1.0, 2.0, 3.0];
        let full = expand_mirror_1d(&half, Symmetry::Even);
        assert_eq!(full, vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn expand_mirror_odd() {
        let half = vec![1.0, 2.0, 3.0];
        let full = expand_mirror_1d(&half, Symmetry::Odd);
        assert_eq!(full, vec![1.0, 2.0, 3.0, -3.0, -2.0, -1.0]);
    }

    #[test]
    fn expand_mirror_none_passthrough() {
        let half = vec![1.0, 2.0, 3.0];
        let full = expand_mirror_1d(&half, Symmetry::None);
        assert_eq!(full, half);
    }

    #[test]
    fn restrict_mirror_halves() {
        let full = vec![1.0, 2.0, 3.0, 4.0];
        let half = restrict_mirror_1d(&full);
        assert_eq!(half, vec![1.0, 2.0]);
    }

    #[test]
    fn point_group_c4_reduction() {
        assert_eq!(PointGroup2d::C4.reduction_factor(), 4);
        assert!((PointGroup2d::C4.ibz_fraction() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn point_group_c6_angle() {
        assert!((PointGroup2d::C6.rotation_angle_deg() - 60.0).abs() < 1e-10);
    }

    #[test]
    fn enforce_x_symmetry_even() {
        let nx = 4;
        let ny = 2;
        // Row-major: field[ix * ny + iy]
        let mut field = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        enforce_x_symmetry_2d(&mut field, nx, ny, Symmetry::Even);
        // ix=0 and ix=3 should be averaged: (1+7)/2=4, (2+8)/2=5
        // ix=1 and ix=2 should be averaged: (3+5)/2=4, (4+6)/2=5
        assert!((field[0] - field[6]).abs() < 1e-10);
        assert!((field[1] - field[7]).abs() < 1e-10);
    }

    #[test]
    fn symmetry_apply_x_odd() {
        let s = Symmetry2d {
            x_sym: Symmetry::Odd,
            y_sym: Symmetry::None,
        };
        assert_eq!(s.apply_x(3.0), -3.0);
    }
}
