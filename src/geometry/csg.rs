//! Constructive Solid Geometry (CSG) operations.
//!
//! CSG allows complex 3D shapes to be built by Boolean operations on primitives:
//!   - Union (∪): all points in A or B
//!   - Intersection (∩): all points in both A and B
//!   - Difference (A\B): all points in A but not B
//!
//! Shapes are represented as signed distance functions (SDF):
//!   SDF(p) < 0: point is inside the shape
//!   SDF(p) = 0: point is on the boundary
//!   SDF(p) > 0: point is outside the shape
//!
//! CSG operations:
//!   Union:        SDF_union(p) = min(SDF_A(p), SDF_B(p))
//!   Intersection: SDF_inter(p) = max(SDF_A(p), SDF_B(p))
//!   Difference:   SDF_diff(p)  = max(SDF_A(p), -SDF_B(p))

use crate::geometry::primitives::Shape2d;

/// A 3D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Distance to another point.
    pub fn dist(&self, other: &Point3) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Trait for signed distance function shapes.
pub trait Sdf: Send + Sync {
    /// Signed distance from point p to this shape's surface.
    /// Negative inside, positive outside.
    fn sdf(&self, p: Point3) -> f64;

    /// True if point p is inside the shape.
    fn contains(&self, p: Point3) -> bool {
        self.sdf(p) <= 0.0
    }

    /// Bounding box (axis-aligned): (min, max) corners.
    fn bounding_box(&self) -> (Point3, Point3);
}

/// Sphere SDF: SDF(p) = |p - center| - radius.
#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub center: Point3,
    pub radius: f64,
}

impl Sphere {
    pub fn new(cx: f64, cy: f64, cz: f64, r: f64) -> Self {
        Self {
            center: Point3::new(cx, cy, cz),
            radius: r,
        }
    }
}

impl Sdf for Sphere {
    fn sdf(&self, p: Point3) -> f64 {
        p.dist(&self.center) - self.radius
    }

    fn bounding_box(&self) -> (Point3, Point3) {
        let r = self.radius;
        let c = self.center;
        (
            Point3::new(c.x - r, c.y - r, c.z - r),
            Point3::new(c.x + r, c.y + r, c.z + r),
        )
    }
}

/// Axis-aligned box SDF.
#[derive(Debug, Clone, Copy)]
pub struct Box3d {
    pub min: Point3,
    pub max: Point3,
}

impl Box3d {
    pub fn new(min: Point3, max: Point3) -> Self {
        Self { min, max }
    }

    pub fn cube(center: Point3, half_size: f64) -> Self {
        let h = half_size;
        Self::new(
            Point3::new(center.x - h, center.y - h, center.z - h),
            Point3::new(center.x + h, center.y + h, center.z + h),
        )
    }
}

impl Sdf for Box3d {
    fn sdf(&self, p: Point3) -> f64 {
        let qx = p.x.max(self.min.x).min(self.max.x) - p.x;
        let qy = p.y.max(self.min.y).min(self.max.y) - p.y;
        let qz = p.z.max(self.min.z).min(self.max.z) - p.z;
        // Interior: max of signed distances to each slab
        let dx = (p.x - self.min.x).min(self.max.x - p.x);
        let dy = (p.y - self.min.y).min(self.max.y - p.y);
        let dz = (p.z - self.min.z).min(self.max.z - p.z);
        let inside = dx >= 0.0 && dy >= 0.0 && dz >= 0.0;
        if inside {
            -dx.min(dy).min(dz) // negative inside
        } else {
            (qx * qx + qy * qy + qz * qz).sqrt()
        }
    }

    fn bounding_box(&self) -> (Point3, Point3) {
        (self.min, self.max)
    }
}

/// Cylinder SDF (aligned along z-axis).
#[derive(Debug, Clone, Copy)]
pub struct Cylinder {
    pub center: Point3,
    pub radius: f64,
    pub half_height: f64,
}

impl Cylinder {
    pub fn new(cx: f64, cy: f64, cz: f64, r: f64, half_h: f64) -> Self {
        Self {
            center: Point3::new(cx, cy, cz),
            radius: r,
            half_height: half_h,
        }
    }
}

impl Sdf for Cylinder {
    fn sdf(&self, p: Point3) -> f64 {
        let dx = p.x - self.center.x;
        let dy = p.y - self.center.y;
        let dz = (p.z - self.center.z).abs() - self.half_height;
        let dr = (dx * dx + dy * dy).sqrt() - self.radius;
        dr.max(dz).min(0.0) + (dr.max(0.0).powi(2) + dz.max(0.0).powi(2)).sqrt()
    }

    fn bounding_box(&self) -> (Point3, Point3) {
        let r = self.radius;
        let h = self.half_height;
        let c = self.center;
        (
            Point3::new(c.x - r, c.y - r, c.z - h),
            Point3::new(c.x + r, c.y + r, c.z + h),
        )
    }
}

/// CSG Union of two shapes.
pub struct CsgUnion {
    pub a: Box<dyn Sdf>,
    pub b: Box<dyn Sdf>,
}

impl CsgUnion {
    pub fn new(a: impl Sdf + 'static, b: impl Sdf + 'static) -> Self {
        Self {
            a: Box::new(a),
            b: Box::new(b),
        }
    }
}

impl Sdf for CsgUnion {
    fn sdf(&self, p: Point3) -> f64 {
        self.a.sdf(p).min(self.b.sdf(p))
    }

    fn bounding_box(&self) -> (Point3, Point3) {
        let (a_min, a_max) = self.a.bounding_box();
        let (b_min, b_max) = self.b.bounding_box();
        (
            Point3::new(
                a_min.x.min(b_min.x),
                a_min.y.min(b_min.y),
                a_min.z.min(b_min.z),
            ),
            Point3::new(
                a_max.x.max(b_max.x),
                a_max.y.max(b_max.y),
                a_max.z.max(b_max.z),
            ),
        )
    }
}

/// CSG Intersection of two shapes.
pub struct CsgIntersection {
    pub a: Box<dyn Sdf>,
    pub b: Box<dyn Sdf>,
}

impl CsgIntersection {
    pub fn new(a: impl Sdf + 'static, b: impl Sdf + 'static) -> Self {
        Self {
            a: Box::new(a),
            b: Box::new(b),
        }
    }
}

impl Sdf for CsgIntersection {
    fn sdf(&self, p: Point3) -> f64 {
        self.a.sdf(p).max(self.b.sdf(p))
    }

    fn bounding_box(&self) -> (Point3, Point3) {
        let (a_min, a_max) = self.a.bounding_box();
        let (b_min, b_max) = self.b.bounding_box();
        (
            Point3::new(
                a_min.x.max(b_min.x),
                a_min.y.max(b_min.y),
                a_min.z.max(b_min.z),
            ),
            Point3::new(
                a_max.x.min(b_max.x),
                a_max.y.min(b_max.y),
                a_max.z.min(b_max.z),
            ),
        )
    }
}

/// CSG Difference: A minus B.
pub struct CsgDifference {
    pub a: Box<dyn Sdf>,
    pub b: Box<dyn Sdf>,
}

impl CsgDifference {
    pub fn new(a: impl Sdf + 'static, b: impl Sdf + 'static) -> Self {
        Self {
            a: Box::new(a),
            b: Box::new(b),
        }
    }
}

impl Sdf for CsgDifference {
    fn sdf(&self, p: Point3) -> f64 {
        self.a.sdf(p).max(-self.b.sdf(p))
    }

    fn bounding_box(&self) -> (Point3, Point3) {
        self.a.bounding_box() // difference can't be larger than A
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Grid rasterization
// ─────────────────────────────────────────────────────────────────────────────

/// Rasterize a 2D shape onto an nx × ny grid within `bounds`.
///
/// `bounds` is `[[x_min, y_min], [x_max, y_max]]`.
/// Returns a flat row-major `Vec<bool>` of length nx × ny;
/// element `[j * nx + i]` is `true` if the cell centre lies inside the shape.
pub fn voxelize(shape: &dyn Shape2d, nx: usize, ny: usize, bounds: [[f64; 2]; 2]) -> Vec<bool> {
    let x_min = bounds[0][0];
    let y_min = bounds[0][1];
    let x_max = bounds[1][0];
    let y_max = bounds[1][1];
    let dx = (x_max - x_min) / nx as f64;
    let dy = (y_max - y_min) / ny as f64;
    let mut out = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        let y = y_min + (j as f64 + 0.5) * dy;
        for i in 0..nx {
            let x = x_min + (i as f64 + 0.5) * dx;
            out.push(shape.contains(x, y));
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Distance-field free functions
// ─────────────────────────────────────────────────────────────────────────────

/// SDF union: min(a, b).
#[inline]
pub fn df_union(a: f64, b: f64) -> f64 {
    a.min(b)
}

/// SDF intersection: max(a, b).
#[inline]
pub fn df_intersection(a: f64, b: f64) -> f64 {
    a.max(b)
}

/// SDF difference: max(a, −b).
#[inline]
pub fn df_difference(a: f64, b: f64) -> f64 {
    a.max(-b)
}

/// Smooth SDF union via polynomial smoothing (Inigo Quilez).
///
/// k controls the blend radius; k = 0 degenerates to min(a, b).
pub fn df_smooth_union(a: f64, b: f64, k: f64) -> f64 {
    if k <= 0.0 {
        return df_union(a, b);
    }
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    a * h + b * (1.0 - h) - k * h * (1.0 - h)
}

/// Linear blend between two SDF values.
///
/// t = 0 → a, t = 1 → b.
#[inline]
pub fn df_blend(a: f64, b: f64, t: f64) -> f64 {
    a * (1.0 - t) + b * t
}

/// Evaluate a `Shape2d`'s membership (0/1) on an nx × ny grid and return
/// the result as a signed distance field approximation using a simple
/// inside/outside sign convention.
///
/// Since `Shape2d` only provides `contains()` (not a true SDF), we produce a
/// pseudo-SDF: −1 inside, +1 outside.  For a true SDF use the `Sdf` trait.
pub fn sdf_grid_from_shape(
    shape: &dyn Shape2d,
    nx: usize,
    ny: usize,
    bounds: [[f64; 2]; 2],
) -> Vec<f64> {
    voxelize(shape, nx, ny, bounds)
        .into_iter()
        .map(|inside| if inside { -1.0 } else { 1.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::primitives::{Circle2d, Rect2d};

    #[test]
    fn sphere_inside() {
        let s = Sphere::new(0.0, 0.0, 0.0, 1.0);
        assert!(s.contains(Point3::new(0.0, 0.0, 0.0)));
        assert!(!s.contains(Point3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn sphere_sdf_at_surface_zero() {
        let s = Sphere::new(0.0, 0.0, 0.0, 1.0);
        assert!((s.sdf(Point3::new(1.0, 0.0, 0.0))).abs() < 1e-10);
    }

    #[test]
    fn box_inside() {
        let b = Box3d::cube(Point3::new(0.0, 0.0, 0.0), 1.0);
        assert!(b.contains(Point3::new(0.0, 0.0, 0.0)));
        assert!(!b.contains(Point3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn csg_union_contains_both() {
        let u = CsgUnion::new(
            Sphere::new(-1.0, 0.0, 0.0, 1.0),
            Sphere::new(1.0, 0.0, 0.0, 1.0),
        );
        assert!(u.contains(Point3::new(-0.5, 0.0, 0.0)));
        assert!(u.contains(Point3::new(0.5, 0.0, 0.0)));
        assert!(!u.contains(Point3::new(5.0, 0.0, 0.0)));
    }

    #[test]
    fn csg_intersection_inside_both() {
        let inter = CsgIntersection::new(
            Sphere::new(0.0, 0.0, 0.0, 2.0),
            Box3d::cube(Point3::new(0.0, 0.0, 0.0), 1.0),
        );
        assert!(inter.contains(Point3::new(0.5, 0.5, 0.5)));
        assert!(!inter.contains(Point3::new(1.5, 0.0, 0.0))); // in sphere, outside box
    }

    #[test]
    fn csg_difference_a_minus_b() {
        let diff = CsgDifference::new(
            Sphere::new(0.0, 0.0, 0.0, 2.0),
            Sphere::new(0.0, 0.0, 0.0, 1.0),
        );
        assert!(!diff.contains(Point3::new(0.5, 0.0, 0.0))); // inside B, should be excluded
        assert!(diff.contains(Point3::new(1.5, 0.0, 0.0))); // in A, outside B
    }

    #[test]
    fn cylinder_inside() {
        let c = Cylinder::new(0.0, 0.0, 0.0, 1.0, 1.0);
        assert!(c.contains(Point3::new(0.5, 0.0, 0.5)));
        assert!(!c.contains(Point3::new(2.0, 0.0, 0.0)));
    }

    // ── voxelize ────────────────────────────────────────────────────────────

    #[test]
    fn voxelize_circle_centre_is_true() {
        let c = Circle2d::new(0.0, 0.0, 0.5);
        let grid = voxelize(&c, 10, 10, [[-1.0, -1.0], [1.0, 1.0]]);
        assert_eq!(grid.len(), 100);
        // Centre cell (4,4) or (5,5) should be inside
        let centre = grid[5 * 10 + 5];
        assert!(centre);
    }

    #[test]
    fn voxelize_rect_fills_interior() {
        let r = Rect2d::new(-0.5, 0.5, -0.5, 0.5);
        let grid = voxelize(&r, 4, 4, [[-1.0, -1.0], [1.0, 1.0]]);
        // The inner 2×2 block (cells 1,2 in each axis) should be inside
        assert!(grid[4 + 1]);
        assert!(grid[4 + 2]);
        assert!(grid[2 * 4 + 1]);
        assert!(grid[2 * 4 + 2]);
        // Corner cells should be outside
        assert!(!grid[0]);
        assert!(!grid[3 * 4 + 3]);
    }

    // ── df_ free functions ───────────────────────────────────────────────────

    #[test]
    fn df_union_is_min() {
        assert_eq!(df_union(-1.0, 0.5), -1.0);
        assert_eq!(df_union(0.5, -1.0), -1.0);
    }

    #[test]
    fn df_intersection_is_max() {
        assert_eq!(df_intersection(-1.0, 0.5), 0.5);
    }

    #[test]
    fn df_difference_excludes_b() {
        // a inside (-0.5), b inside (-0.3) → difference = max(-0.5, 0.3) = 0.3 (outside)
        let d = df_difference(-0.5, -0.3);
        assert!((d - 0.3).abs() < 1e-12);
    }

    #[test]
    fn df_smooth_union_between_min_and_values() {
        let a = 0.3_f64;
        let b = -0.2_f64;
        let k = 0.5;
        let su = df_smooth_union(a, b, k);
        // Smooth union should be <= min(a,b) by at most k/4
        assert!(su <= b.min(a) + k * 0.25 + 1e-10);
    }

    #[test]
    fn df_smooth_union_zero_k_equals_min() {
        assert!((df_smooth_union(0.3, -0.2, 0.0) - df_union(0.3, -0.2)).abs() < 1e-12);
    }

    #[test]
    fn df_blend_at_zero_is_a() {
        assert!((df_blend(3.0, 7.0, 0.0) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn df_blend_at_one_is_b() {
        assert!((df_blend(3.0, 7.0, 1.0) - 7.0).abs() < 1e-12);
    }

    #[test]
    fn df_blend_midpoint() {
        assert!((df_blend(0.0, 1.0, 0.5) - 0.5).abs() < 1e-12);
    }

    // ── sdf_grid_from_shape ─────────────────────────────────────────────────

    #[test]
    fn sdf_grid_inside_is_negative() {
        let c = Circle2d::new(0.0, 0.0, 0.9);
        let grid = sdf_grid_from_shape(&c, 3, 3, [[-1.0, -1.0], [1.0, 1.0]]);
        // Centre cell [1*3+1]
        assert!(grid[3 + 1] < 0.0);
    }

    #[test]
    fn sdf_grid_outside_is_positive() {
        let c = Circle2d::new(0.0, 0.0, 0.1);
        let grid = sdf_grid_from_shape(&c, 3, 3, [[-1.0, -1.0], [1.0, 1.0]]);
        // Corner cell [0*3+0] is far outside
        assert!(grid[0] > 0.0);
    }
}
