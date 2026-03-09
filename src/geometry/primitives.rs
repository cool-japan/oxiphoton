/// Axis-aligned bounding box in 2D
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox2d {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

/// 2D shape trait
pub trait Shape2d: Send + Sync {
    /// Return true if point (x, y) is inside the shape
    fn contains(&self, x: f64, y: f64) -> bool;
    /// Bounding box of the shape
    fn bounding_box(&self) -> BoundingBox2d;
}

/// 2D rectangle
#[derive(Debug, Clone, Copy)]
pub struct Rect2d {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

/// 2D circle
#[derive(Debug, Clone, Copy)]
pub struct Circle2d {
    pub cx: f64,
    pub cy: f64,
    pub radius: f64,
}

impl Rect2d {
    pub fn new(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Self {
        Self {
            x_min,
            x_max,
            y_min,
            y_max,
        }
    }
}

impl Circle2d {
    pub fn new(cx: f64, cy: f64, radius: f64) -> Self {
        Self { cx, cy, radius }
    }
}

impl Shape2d for Rect2d {
    fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.x_min && x <= self.x_max && y >= self.y_min && y <= self.y_max
    }

    fn bounding_box(&self) -> BoundingBox2d {
        BoundingBox2d {
            x_min: self.x_min,
            x_max: self.x_max,
            y_min: self.y_min,
            y_max: self.y_max,
        }
    }
}

impl Shape2d for Circle2d {
    fn contains(&self, x: f64, y: f64) -> bool {
        let dx = x - self.cx;
        let dy = y - self.cy;
        dx * dx + dy * dy <= self.radius * self.radius
    }

    fn bounding_box(&self) -> BoundingBox2d {
        BoundingBox2d {
            x_min: self.cx - self.radius,
            x_max: self.cx + self.radius,
            y_min: self.cy - self.radius,
            y_max: self.cy + self.radius,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3D Axis-aligned bounding box
// ─────────────────────────────────────────────────────────────────────────────

/// 3D axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct Aabb3d {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Aabb3d {
    pub fn new(min: [f64; 3], max: [f64; 3]) -> Self {
        Self { min, max }
    }

    pub fn contains(&self, p: [f64; 3]) -> bool {
        p[0] >= self.min[0]
            && p[0] <= self.max[0]
            && p[1] >= self.min[1]
            && p[1] <= self.max[1]
            && p[2] >= self.min[2]
            && p[2] <= self.max[2]
    }

    pub fn intersects(&self, other: &Aabb3d) -> bool {
        self.min[0] <= other.max[0]
            && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1]
            && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2]
            && self.max[2] >= other.min[2]
    }

    pub fn union(&self, other: &Aabb3d) -> Aabb3d {
        Aabb3d {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    pub fn volume(&self) -> f64 {
        (self.max[0] - self.min[0]).max(0.0)
            * (self.max[1] - self.min[1]).max(0.0)
            * (self.max[2] - self.min[2]).max(0.0)
    }

    pub fn surface_area(&self) -> f64 {
        let dx = (self.max[0] - self.min[0]).max(0.0);
        let dy = (self.max[1] - self.min[1]).max(0.0);
        let dz = (self.max[2] - self.min[2]).max(0.0);
        2.0 * (dx * dy + dy * dz + dz * dx)
    }

    pub fn center(&self) -> [f64; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    pub fn expand_by(&self, delta: f64) -> Aabb3d {
        Aabb3d {
            min: [
                self.min[0] - delta,
                self.min[1] - delta,
                self.min[2] - delta,
            ],
            max: [
                self.max[0] + delta,
                self.max[1] + delta,
                self.max[2] + delta,
            ],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3D Sphere
// ─────────────────────────────────────────────────────────────────────────────

/// 3D sphere.
#[derive(Debug, Clone, Copy)]
pub struct Sphere3d {
    pub center: [f64; 3],
    pub radius: f64,
}

impl Sphere3d {
    pub fn new(center: [f64; 3], radius: f64) -> Self {
        Self { center, radius }
    }

    pub fn contains(&self, p: [f64; 3]) -> bool {
        let dx = p[0] - self.center[0];
        let dy = p[1] - self.center[1];
        let dz = p[2] - self.center[2];
        dx * dx + dy * dy + dz * dz <= self.radius * self.radius
    }

    pub fn volume(&self) -> f64 {
        4.0 / 3.0 * std::f64::consts::PI * self.radius.powi(3)
    }

    pub fn surface_area(&self) -> f64 {
        4.0 * std::f64::consts::PI * self.radius * self.radius
    }

    pub fn bounding_box(&self) -> Aabb3d {
        let r = self.radius;
        Aabb3d {
            min: [self.center[0] - r, self.center[1] - r, self.center[2] - r],
            max: [self.center[0] + r, self.center[1] + r, self.center[2] + r],
        }
    }

    /// Returns true if the sphere intersects or touches the AABB.
    pub fn intersects_aabb(&self, aabb: &Aabb3d) -> bool {
        // Square distance from sphere center to nearest point in AABB
        let sq_dist: f64 = (0..3)
            .map(|i| {
                let v = self.center[i].max(aabb.min[i]).min(aabb.max[i]);
                let d = self.center[i] - v;
                d * d
            })
            .sum();
        sq_dist <= self.radius * self.radius
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3D Cylinder (Z-axis aligned)
// ─────────────────────────────────────────────────────────────────────────────

/// 3D cylinder aligned with the Z-axis.
#[derive(Debug, Clone, Copy)]
pub struct Cylinder3d {
    pub center_x: f64,
    pub center_y: f64,
    pub z_min: f64,
    pub z_max: f64,
    pub radius: f64,
}

impl Cylinder3d {
    pub fn new(center_x: f64, center_y: f64, z_min: f64, z_max: f64, radius: f64) -> Self {
        Self {
            center_x,
            center_y,
            z_min,
            z_max,
            radius,
        }
    }

    pub fn contains(&self, p: [f64; 3]) -> bool {
        let dx = p[0] - self.center_x;
        let dy = p[1] - self.center_y;
        dx * dx + dy * dy <= self.radius * self.radius && p[2] >= self.z_min && p[2] <= self.z_max
    }

    pub fn height(&self) -> f64 {
        (self.z_max - self.z_min).max(0.0)
    }

    pub fn volume(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius * self.height()
    }

    pub fn lateral_area(&self) -> f64 {
        2.0 * std::f64::consts::PI * self.radius * self.height()
    }

    pub fn bounding_box(&self) -> Aabb3d {
        Aabb3d {
            min: [
                self.center_x - self.radius,
                self.center_y - self.radius,
                self.z_min,
            ],
            max: [
                self.center_x + self.radius,
                self.center_y + self.radius,
                self.z_max,
            ],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3D Torus (waveguide ring geometry)
// ─────────────────────────────────────────────────────────────────────────────

/// 3D torus.
///
/// Parameterised by major radius R (center of tube to center of ring) and
/// minor radius r (tube radius).  The torus is oriented so that its axis of
/// symmetry points along the direction `normal`.  For convenience,
/// `new_z_axis` creates a torus with a Z-axis normal.
#[derive(Debug, Clone, Copy)]
pub struct Torus3d {
    /// Major radius R (m)
    pub major_radius: f64,
    /// Minor radius r (m)
    pub minor_radius: f64,
    /// Center of the torus
    pub center: [f64; 3],
    /// Axis of symmetry (unit vector)
    pub normal: [f64; 3],
}

impl Torus3d {
    pub fn new_z_axis(center: [f64; 3], major_radius: f64, minor_radius: f64) -> Self {
        Self {
            major_radius,
            minor_radius,
            center,
            normal: [0.0, 0.0, 1.0],
        }
    }

    /// Approximate containment test (valid for Z-axis tori).
    ///
    /// Treats the point as lying in the torus cross-section plane and checks
    /// whether the distance to the tube centre circle is ≤ minor_radius.
    pub fn contains_approx(&self, p: [f64; 3]) -> bool {
        let dx = p[0] - self.center[0];
        let dy = p[1] - self.center[1];
        let dz = p[2] - self.center[2];
        let rho = (dx * dx + dy * dy).sqrt();
        let dist_tube = ((rho - self.major_radius).powi(2) + dz * dz).sqrt();
        dist_tube <= self.minor_radius
    }

    /// Volume = 2π²·R·r²
    pub fn volume(&self) -> f64 {
        2.0 * std::f64::consts::PI
            * std::f64::consts::PI
            * self.major_radius
            * self.minor_radius.powi(2)
    }

    /// Surface area = 4π²·R·r
    pub fn surface_area(&self) -> f64 {
        4.0 * std::f64::consts::PI * std::f64::consts::PI * self.major_radius * self.minor_radius
    }

    /// Axis-aligned bounding box (valid for Z-axis tori only).
    pub fn bounding_box(&self) -> Aabb3d {
        let extent_xy = self.major_radius + self.minor_radius;
        Aabb3d {
            min: [
                self.center[0] - extent_xy,
                self.center[1] - extent_xy,
                self.center[2] - self.minor_radius,
            ],
            max: [
                self.center[0] + extent_xy,
                self.center[1] + extent_xy,
                self.center[2] + self.minor_radius,
            ],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Waveguide cross-section profiles
// ─────────────────────────────────────────────────────────────────────────────

/// Photonic waveguide cross-section profile.
#[derive(Debug, Clone)]
pub enum WaveguideProfile {
    /// Simple strip waveguide
    Strip { width: f64, height: f64 },
    /// Ridge waveguide with slab
    Ridge {
        top_width: f64,
        slab_width: f64,
        core_height: f64,
        slab_height: f64,
    },
    /// Slot waveguide
    Slot {
        total_width: f64,
        slot_width: f64,
        height: f64,
    },
    /// Circular (fiber-like) cross-section
    Circular { radius: f64 },
    /// Elliptical cross-section
    Elliptical { a: f64, b: f64 },
}

impl WaveguideProfile {
    /// Cross-sectional area (m²).
    pub fn area(&self) -> f64 {
        match self {
            WaveguideProfile::Strip { width, height } => width * height,
            WaveguideProfile::Ridge {
                top_width,
                slab_width,
                core_height,
                slab_height,
            } => top_width * core_height + slab_width * slab_height,
            WaveguideProfile::Slot {
                total_width,
                slot_width,
                height,
            } => (total_width - slot_width) * height,
            WaveguideProfile::Circular { radius } => std::f64::consts::PI * radius * radius,
            WaveguideProfile::Elliptical { a, b } => std::f64::consts::PI * a * b,
        }
    }

    /// Perimeter of the cross-section (m).
    pub fn perimeter(&self) -> f64 {
        match self {
            WaveguideProfile::Strip { width, height } => 2.0 * (width + height),
            WaveguideProfile::Ridge {
                top_width,
                slab_width,
                core_height,
                slab_height,
            } => {
                // Approximate: outer perimeter of ridge + slab
                2.0 * (slab_width + slab_height) + 2.0 * top_width + 2.0 * core_height
            }
            WaveguideProfile::Slot {
                total_width,
                slot_width: _,
                height,
            } => {
                // Two rails: each has its own perimeter (simplified)
                2.0 * (total_width + height)
            }
            WaveguideProfile::Circular { radius } => 2.0 * std::f64::consts::PI * radius,
            WaveguideProfile::Elliptical { a, b } => {
                // Ramanujan approximation
                std::f64::consts::PI * (3.0 * (a + b) - ((3.0 * a + b) * (a + 3.0 * b)).sqrt())
            }
        }
    }

    /// 2D containment test for the cross-section.
    pub fn contains(&self, x: f64, y: f64) -> bool {
        match self {
            WaveguideProfile::Strip { width, height } => {
                x.abs() <= width * 0.5 && y >= 0.0 && y <= *height
            }
            WaveguideProfile::Ridge {
                top_width,
                slab_width,
                core_height,
                slab_height,
            } => {
                let in_slab = x.abs() <= slab_width * 0.5 && y >= 0.0 && y <= *slab_height;
                let in_ridge = x.abs() <= top_width * 0.5
                    && y >= *slab_height
                    && y <= slab_height + core_height;
                in_slab || in_ridge
            }
            WaveguideProfile::Slot {
                total_width,
                slot_width,
                height,
            } => {
                let rail_half = (total_width - slot_width) * 0.5;
                let in_left_rail =
                    x >= -total_width * 0.5 && x <= -(slot_width * 0.5) && y >= 0.0 && y <= *height;
                let in_right_rail =
                    x >= slot_width * 0.5 && x <= total_width * 0.5 && y >= 0.0 && y <= *height;
                let _ = rail_half;
                in_left_rail || in_right_rail
            }
            WaveguideProfile::Circular { radius } => x * x + y * y <= radius * radius,
            WaveguideProfile::Elliptical { a, b } => (x / a).powi(2) + (y / b).powi(2) <= 1.0,
        }
    }

    /// 2D bounding box: (x_min, x_max, y_min, y_max).
    pub fn bounding_box_2d(&self) -> (f64, f64, f64, f64) {
        match self {
            WaveguideProfile::Strip { width, height } => (-width * 0.5, width * 0.5, 0.0, *height),
            WaveguideProfile::Ridge {
                top_width: _,
                slab_width,
                core_height,
                slab_height,
            } => (
                -slab_width * 0.5,
                slab_width * 0.5,
                0.0,
                slab_height + core_height,
            ),
            WaveguideProfile::Slot {
                total_width,
                slot_width: _,
                height,
            } => (-total_width * 0.5, total_width * 0.5, 0.0, *height),
            WaveguideProfile::Circular { radius } => (-radius, *radius, -radius, *radius),
            WaveguideProfile::Elliptical { a, b } => (-a, *a, -b, *b),
        }
    }

    /// Aspect ratio (width / height or major / minor axis).
    pub fn aspect_ratio(&self) -> f64 {
        match self {
            WaveguideProfile::Strip { width, height } => width / height.max(1e-30),
            WaveguideProfile::Ridge {
                top_width,
                slab_width: _,
                core_height,
                slab_height,
            } => top_width / (core_height + slab_height).max(1e-30),
            WaveguideProfile::Slot {
                total_width,
                slot_width: _,
                height,
            } => total_width / height.max(1e-30),
            WaveguideProfile::Circular { radius: _ } => 1.0,
            WaveguideProfile::Elliptical { a, b } => a / b.max(1e-30),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2D Polygon
// ─────────────────────────────────────────────────────────────────────────────

/// 2D polygon (for complex waveguide shapes).
#[derive(Debug, Clone)]
pub struct Polygon2d {
    pub vertices: Vec<[f64; 2]>,
}

impl Polygon2d {
    /// Create a polygon from a list of vertices.
    pub fn new(vertices: Vec<[f64; 2]>) -> Self {
        Self { vertices }
    }

    /// Signed area via the shoelace formula.  Positive for CCW orientation.
    pub fn signed_area(&self) -> f64 {
        let n = self.vertices.len();
        if n < 3 {
            return 0.0;
        }
        let mut sum = 0.0_f64;
        for i in 0..n {
            let j = (i + 1) % n;
            sum += self.vertices[i][0] * self.vertices[j][1];
            sum -= self.vertices[j][0] * self.vertices[i][1];
        }
        sum * 0.5
    }

    /// Area (always positive).
    pub fn area(&self) -> f64 {
        self.signed_area().abs()
    }

    /// Perimeter (sum of edge lengths).
    pub fn perimeter(&self) -> f64 {
        let n = self.vertices.len();
        if n < 2 {
            return 0.0;
        }
        (0..n)
            .map(|i| {
                let j = (i + 1) % n;
                let dx = self.vertices[j][0] - self.vertices[i][0];
                let dy = self.vertices[j][1] - self.vertices[i][1];
                (dx * dx + dy * dy).sqrt()
            })
            .sum()
    }

    /// Ray casting algorithm for point-in-polygon.
    pub fn contains(&self, p: [f64; 2]) -> bool {
        let n = self.vertices.len();
        if n < 3 {
            return false;
        }
        let mut inside = false;
        let (px, py) = (p[0], p[1]);
        let mut j = n - 1;
        for i in 0..n {
            let (xi, yi) = (self.vertices[i][0], self.vertices[i][1]);
            let (xj, yj) = (self.vertices[j][0], self.vertices[j][1]);
            if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi + 1e-300) + xi) {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// Centroid of the polygon.
    pub fn centroid(&self) -> [f64; 2] {
        let n = self.vertices.len();
        if n == 0 {
            return [0.0, 0.0];
        }
        if n == 1 {
            return self.vertices[0];
        }
        let signed_area = self.signed_area();
        if signed_area.abs() < 1e-300 {
            // Degenerate: return mean of vertices
            let cx = self.vertices.iter().map(|v| v[0]).sum::<f64>() / n as f64;
            let cy = self.vertices.iter().map(|v| v[1]).sum::<f64>() / n as f64;
            return [cx, cy];
        }
        let mut cx = 0.0_f64;
        let mut cy = 0.0_f64;
        for i in 0..n {
            let j = (i + 1) % n;
            let cross = self.vertices[i][0] * self.vertices[j][1]
                - self.vertices[j][0] * self.vertices[i][1];
            cx += (self.vertices[i][0] + self.vertices[j][0]) * cross;
            cy += (self.vertices[i][1] + self.vertices[j][1]) * cross;
        }
        let factor = 1.0 / (6.0 * signed_area);
        [cx * factor, cy * factor]
    }

    /// Axis-aligned bounding box: ([x_min, y_min], [x_max, y_max]).
    pub fn bounding_box(&self) -> ([f64; 2], [f64; 2]) {
        if self.vertices.is_empty() {
            return ([0.0, 0.0], [0.0, 0.0]);
        }
        let x_min = self
            .vertices
            .iter()
            .map(|v| v[0])
            .fold(f64::INFINITY, f64::min);
        let x_max = self
            .vertices
            .iter()
            .map(|v| v[0])
            .fold(f64::NEG_INFINITY, f64::max);
        let y_min = self
            .vertices
            .iter()
            .map(|v| v[1])
            .fold(f64::INFINITY, f64::min);
        let y_max = self
            .vertices
            .iter()
            .map(|v| v[1])
            .fold(f64::NEG_INFINITY, f64::max);
        ([x_min, y_min], [x_max, y_max])
    }

    /// Check if the polygon is convex.
    pub fn is_convex(&self) -> bool {
        let n = self.vertices.len();
        if n < 3 {
            return true;
        }
        let mut sign = 0i32;
        for i in 0..n {
            let a = self.vertices[i];
            let b = self.vertices[(i + 1) % n];
            let c = self.vertices[(i + 2) % n];
            let cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
            let s = if cross > 0.0 {
                1i32
            } else if cross < 0.0 {
                -1
            } else {
                0
            };
            if s != 0 {
                if sign == 0 {
                    sign = s;
                } else if sign != s {
                    return false;
                }
            }
        }
        true
    }

    /// Convex hull using Graham scan.
    ///
    /// Returns a new Polygon2d representing the convex hull of the vertices.
    pub fn convex_hull(&self) -> Polygon2d {
        let mut pts = self.vertices.clone();
        let n = pts.len();
        if n < 3 {
            return Polygon2d::new(pts);
        }
        // Find lowest point (then leftmost)
        let pivot_idx = pts
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a[1].partial_cmp(&b[1])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal))
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        pts.swap(0, pivot_idx);
        let pivot = pts[0];
        // Sort by polar angle around pivot
        pts[1..].sort_by(|a, b| {
            let ax = a[0] - pivot[0];
            let ay = a[1] - pivot[1];
            let bx = b[0] - pivot[0];
            let by = b[1] - pivot[1];
            let cross = ax * by - ay * bx;
            if cross.abs() < 1e-300 {
                // Collinear: sort by distance
                let da = ax * ax + ay * ay;
                let db = bx * bx + by * by;
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                // Negative cross = clockwise = comes later
                (-cross)
                    .partial_cmp(&0.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        });
        // Graham scan
        let mut hull: Vec<[f64; 2]> = Vec::with_capacity(n);
        for p in &pts {
            while hull.len() >= 2 {
                let a = hull[hull.len() - 2];
                let b = hull[hull.len() - 1];
                let cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
                if cross <= 0.0 {
                    hull.pop();
                } else {
                    break;
                }
            }
            hull.push(*p);
        }
        Polygon2d::new(hull)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_contains() {
        let c = Circle2d::new(0.0, 0.0, 1.0);
        assert!(c.contains(0.0, 0.0));
        assert!(c.contains(0.5, 0.5));
        assert!(!c.contains(1.0, 1.0));
    }

    #[test]
    fn rect_contains() {
        let r = Rect2d::new(-1.0, 1.0, -1.0, 1.0);
        assert!(r.contains(0.0, 0.0));
        assert!(r.contains(1.0, 1.0));
        assert!(!r.contains(1.5, 0.0));
    }

    // ── Aabb3d tests ─────────────────────────────────────────────────────────

    #[test]
    fn aabb3d_contains_center() {
        let b = Aabb3d::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(b.contains([0.5, 0.5, 0.5]));
        assert!(!b.contains([1.5, 0.5, 0.5]));
    }

    #[test]
    fn aabb3d_volume_correct() {
        let b = Aabb3d::new([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]);
        assert!((b.volume() - 24.0).abs() < 1e-10);
    }

    #[test]
    fn aabb3d_surface_area_correct() {
        let b = Aabb3d::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!((b.surface_area() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn aabb3d_center_correct() {
        let b = Aabb3d::new([0.0, 0.0, 0.0], [4.0, 6.0, 8.0]);
        let c = b.center();
        assert!((c[0] - 2.0).abs() < 1e-10);
        assert!((c[1] - 3.0).abs() < 1e-10);
        assert!((c[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn aabb3d_intersects_overlapping() {
        let a = Aabb3d::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let b = Aabb3d::new([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]);
        assert!(a.intersects(&b));
    }

    #[test]
    fn aabb3d_intersects_disjoint() {
        let a = Aabb3d::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = Aabb3d::new([2.0, 2.0, 2.0], [3.0, 3.0, 3.0]);
        assert!(!a.intersects(&b));
    }

    // ── Sphere3d tests ────────────────────────────────────────────────────────

    #[test]
    fn sphere_volume_correct() {
        let s = Sphere3d::new([0.0, 0.0, 0.0], 1.0);
        let expected = 4.0 / 3.0 * std::f64::consts::PI;
        assert!((s.volume() - expected).abs() < 1e-10);
    }

    #[test]
    fn sphere_surface_area_correct() {
        let s = Sphere3d::new([0.0, 0.0, 0.0], 1.0);
        let expected = 4.0 * std::f64::consts::PI;
        assert!((s.surface_area() - expected).abs() < 1e-10);
    }

    #[test]
    fn sphere_contains_center() {
        let s = Sphere3d::new([1.0, 2.0, 3.0], 0.5);
        assert!(s.contains([1.0, 2.0, 3.0]));
        assert!(!s.contains([2.0, 2.0, 3.0]));
    }

    #[test]
    fn sphere_intersects_aabb() {
        let s = Sphere3d::new([0.0, 0.0, 0.0], 1.0);
        let aabb = Aabb3d::new([0.5, 0.5, 0.5], [2.0, 2.0, 2.0]);
        assert!(s.intersects_aabb(&aabb));
        let far_aabb = Aabb3d::new([3.0, 3.0, 3.0], [4.0, 4.0, 4.0]);
        assert!(!s.intersects_aabb(&far_aabb));
    }

    // ── Cylinder3d tests ──────────────────────────────────────────────────────

    #[test]
    fn cylinder_volume_correct() {
        let c = Cylinder3d::new(0.0, 0.0, 0.0, 2.0, 1.0);
        let expected = std::f64::consts::PI * 2.0;
        assert!((c.volume() - expected).abs() < 1e-10);
    }

    #[test]
    fn cylinder_contains() {
        let c = Cylinder3d::new(0.0, 0.0, 0.0, 5.0, 1.0);
        assert!(c.contains([0.0, 0.0, 2.5]));
        assert!(!c.contains([0.0, 0.0, 6.0]));
        assert!(!c.contains([1.5, 0.0, 2.5]));
    }

    // ── Torus3d tests ─────────────────────────────────────────────────────────

    #[test]
    fn torus_volume_correct() {
        let t = Torus3d::new_z_axis([0.0, 0.0, 0.0], 5.0, 1.0);
        let expected = 2.0 * std::f64::consts::PI * std::f64::consts::PI * 5.0 * 1.0;
        assert!((t.volume() - expected).abs() < 1e-8);
    }

    #[test]
    fn torus_contains_on_ring() {
        let t = Torus3d::new_z_axis([0.0, 0.0, 0.0], 5.0, 1.0);
        // Point at (5, 0, 0) — exactly on the tube centre circle → inside
        assert!(t.contains_approx([5.0, 0.0, 0.0]));
        // Point far away → outside
        assert!(!t.contains_approx([10.0, 0.0, 0.0]));
    }

    // ── WaveguideProfile tests ────────────────────────────────────────────────

    #[test]
    fn strip_profile_area() {
        let p = WaveguideProfile::Strip {
            width: 0.5e-6,
            height: 0.22e-6,
        };
        let expected = 0.5e-6 * 0.22e-6;
        assert!((p.area() - expected).abs() < 1e-20);
    }

    #[test]
    fn circular_profile_area() {
        let r = 1.0e-6_f64;
        let p = WaveguideProfile::Circular { radius: r };
        let expected = std::f64::consts::PI * r * r;
        assert!((p.area() - expected).abs() < 1e-25);
    }

    #[test]
    fn strip_profile_contains() {
        let p = WaveguideProfile::Strip {
            width: 1.0,
            height: 1.0,
        };
        assert!(p.contains(0.0, 0.5));
        assert!(!p.contains(1.0, 0.5)); // on edge: x = 0.5*width
    }

    #[test]
    fn strip_aspect_ratio() {
        let p = WaveguideProfile::Strip {
            width: 2.0,
            height: 1.0,
        };
        assert!((p.aspect_ratio() - 2.0).abs() < 1e-10);
    }

    // ── Polygon2d tests ───────────────────────────────────────────────────────

    #[test]
    fn polygon_area_unit_square() {
        let sq = Polygon2d::new(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        assert!((sq.area() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn polygon_perimeter_unit_square() {
        let sq = Polygon2d::new(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        assert!((sq.perimeter() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn polygon_contains_interior() {
        let sq = Polygon2d::new(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        assert!(sq.contains([0.5, 0.5]));
        assert!(!sq.contains([1.5, 0.5]));
    }

    #[test]
    fn polygon_centroid_unit_square() {
        let sq = Polygon2d::new(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let c = sq.centroid();
        assert!((c[0] - 0.5).abs() < 1e-10);
        assert!((c[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn polygon_convex_check_square() {
        let sq = Polygon2d::new(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        assert!(sq.is_convex());
    }

    #[test]
    fn polygon_not_convex_concave() {
        // Star-like concave polygon
        let poly = Polygon2d::new(vec![
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, 0.0],
            [1.5, 1.0],
            [1.0, 2.0],
            [0.5, 1.0],
        ]);
        assert!(!poly.is_convex());
    }

    #[test]
    fn polygon_convex_hull_of_square() {
        // Add interior points; hull should be the square corners
        let poly = Polygon2d::new(vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5], // interior
        ]);
        let hull = poly.convex_hull();
        assert!(hull.vertices.len() >= 4);
        // Hull area should be 1.0
        assert!(
            (hull.area() - 1.0).abs() < 1e-8,
            "hull area={}",
            hull.area()
        );
    }
}
