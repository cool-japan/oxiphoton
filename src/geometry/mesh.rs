//! Unstructured mesh for FEM and finite-volume methods.
//!
//! Represents a 2D triangular mesh or 3D tetrahedral mesh. The mesh stores:
//!   - Node coordinates
//!   - Element connectivity (triangles or tetrahedra)
//!   - Boundary edges/faces
//!   - Material labels per element

/// A 2D mesh node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Node2d {
    pub x: f64,
    pub y: f64,
}

impl Node2d {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// A triangular mesh element (3 node indices).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Triangle {
    pub nodes: [usize; 3],
    /// Material label (index into material list)
    pub material: usize,
}

impl Triangle {
    pub fn new(n0: usize, n1: usize, n2: usize) -> Self {
        Self {
            nodes: [n0, n1, n2],
            material: 0,
        }
    }

    pub fn with_material(n0: usize, n1: usize, n2: usize, mat: usize) -> Self {
        Self {
            nodes: [n0, n1, n2],
            material: mat,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mesh quality helpers (free functions)
// ─────────────────────────────────────────────────────────────────────────────

/// Edge length between two 2D vertices.
#[inline]
fn edge_len(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt()
}

/// Aspect ratio of a triangle: longest-edge / shortest-altitude.
///
/// A perfect equilateral triangle has aspect ratio 1 (all altitudes equal
/// to `sqrt(3)/2 * side`).  Degenerate (flat) triangles → ∞.
pub fn aspect_ratio(v0: [f64; 2], v1: [f64; 2], v2: [f64; 2]) -> f64 {
    let a = edge_len(v1, v2);
    let b = edge_len(v0, v2);
    let c = edge_len(v0, v1);
    let longest = a.max(b).max(c);
    // Cross product magnitude = 2 · triangle_area
    let two_area = ((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])).abs();
    if two_area < f64::EPSILON {
        return f64::INFINITY;
    }
    // altitude_i = 2·triangle_area / opposite_edge = two_area / opposite_edge
    let alt_a = two_area / a.max(f64::EPSILON);
    let alt_b = two_area / b.max(f64::EPSILON);
    let alt_c = two_area / c.max(f64::EPSILON);
    let shortest_alt = alt_a.min(alt_b).min(alt_c);
    if shortest_alt < f64::EPSILON {
        return f64::INFINITY;
    }
    longest / shortest_alt
}

/// Skewness of a triangle: 0 = equilateral, 1 = degenerate.
///
/// Uses the equilateral-normalised formula:
///   skewness = (θ_max − θ_eq) / (π − θ_eq)
/// where θ_eq = π/3 (60°).
pub fn skewness(v0: [f64; 2], v1: [f64; 2], v2: [f64; 2]) -> f64 {
    let a = edge_len(v1, v2);
    let b = edge_len(v0, v2);
    let c = edge_len(v0, v1);
    // Compute angles via law of cosines
    let angle = |opp: f64, s1: f64, s2: f64| -> f64 {
        if s1 < f64::EPSILON || s2 < f64::EPSILON {
            return std::f64::consts::PI;
        }
        let cos_val = (s1 * s1 + s2 * s2 - opp * opp) / (2.0 * s1 * s2);
        cos_val.clamp(-1.0, 1.0).acos()
    };
    let theta_a = angle(a, b, c);
    let theta_b = angle(b, a, c);
    let theta_c = angle(c, a, b);
    let theta_max = theta_a.max(theta_b).max(theta_c);
    let theta_min = theta_a.min(theta_b).min(theta_c);
    let theta_eq = std::f64::consts::PI / 3.0;
    let skew_max = (theta_max - theta_eq) / (std::f64::consts::PI - theta_eq);
    let skew_min = (theta_eq - theta_min) / theta_eq;
    skew_max.max(skew_min).clamp(0.0, 1.0)
}

/// Midpoint of the longest edge of a triangle.
pub fn longest_edge_midpoint(v0: [f64; 2], v1: [f64; 2], v2: [f64; 2]) -> [f64; 2] {
    let l01 = edge_len(v0, v1);
    let l12 = edge_len(v1, v2);
    let l20 = edge_len(v2, v0);
    let mid = |a: [f64; 2], b: [f64; 2]| [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
    if l01 >= l12 && l01 >= l20 {
        mid(v0, v1)
    } else if l12 >= l20 {
        mid(v1, v2)
    } else {
        mid(v2, v0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mesh quality report
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics for mesh quality.
#[derive(Debug, Clone)]
pub struct MeshQualityReport {
    pub min_aspect_ratio: f64,
    pub max_aspect_ratio: f64,
    pub mean_skewness: f64,
    /// Number of triangles with skewness > 0.9 (near-degenerate).
    pub n_degenerate: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// AABB / BVH
// ─────────────────────────────────────────────────────────────────────────────

/// Axis-aligned bounding box in 2D.
#[derive(Debug, Clone, Copy)]
pub struct Aabb2d {
    pub min: [f64; 2],
    pub max: [f64; 2],
}

impl Aabb2d {
    /// Create a new AABB.
    pub fn new(min: [f64; 2], max: [f64; 2]) -> Self {
        Self { min, max }
    }

    /// Build AABB enclosing a single triangle.
    pub fn from_triangle(v0: [f64; 2], v1: [f64; 2], v2: [f64; 2]) -> Self {
        Self {
            min: [v0[0].min(v1[0]).min(v2[0]), v0[1].min(v1[1]).min(v2[1])],
            max: [v0[0].max(v1[0]).max(v2[0]), v0[1].max(v1[1]).max(v2[1])],
        }
    }

    /// True if point p lies inside (or on the boundary of) this AABB.
    pub fn contains(&self, p: [f64; 2]) -> bool {
        p[0] >= self.min[0] && p[0] <= self.max[0] && p[1] >= self.min[1] && p[1] <= self.max[1]
    }

    /// True if this AABB overlaps with another.
    pub fn intersects(&self, other: &Aabb2d) -> bool {
        self.min[0] <= other.max[0]
            && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1]
            && self.max[1] >= other.min[1]
    }

    /// Return a new AABB expanded outward by `margin` on all sides.
    pub fn expand(&self, margin: f64) -> Aabb2d {
        Aabb2d {
            min: [self.min[0] - margin, self.min[1] - margin],
            max: [self.max[0] + margin, self.max[1] + margin],
        }
    }

    /// Merge two AABBs into one that encloses both.
    fn merge(a: &Aabb2d, b: &Aabb2d) -> Aabb2d {
        Aabb2d {
            min: [a.min[0].min(b.min[0]), a.min[1].min(b.min[1])],
            max: [a.max[0].max(b.max[0]), a.max[1].max(b.max[1])],
        }
    }

    /// Centre of this AABB along `axis` (0=x, 1=y).
    fn centre(&self, axis: usize) -> f64 {
        (self.min[axis] + self.max[axis]) * 0.5
    }
}

/// A node in a bounding-volume hierarchy over 2D triangles.
pub enum BvhNode {
    Leaf {
        tri_idx: usize,
        aabb: Aabb2d,
    },
    Branch {
        aabb: Aabb2d,
        left: Box<BvhNode>,
        right: Box<BvhNode>,
    },
}

impl BvhNode {
    fn aabb(&self) -> &Aabb2d {
        match self {
            BvhNode::Leaf { aabb, .. } => aabb,
            BvhNode::Branch { aabb, .. } => aabb,
        }
    }
}

/// Bounding-volume hierarchy for fast point-in-triangle queries.
pub struct Bvh2d {
    root: Option<BvhNode>,
}

impl Bvh2d {
    /// Build a BVH from triangle connectivity and vertex positions.
    pub fn build(triangles: &[[usize; 3]], vertices: &[[f64; 2]]) -> Self {
        if triangles.is_empty() {
            return Self { root: None };
        }
        // Collect (aabb, index) pairs
        let mut items: Vec<(usize, Aabb2d)> = triangles
            .iter()
            .enumerate()
            .map(|(i, tri)| {
                let v0 = vertices[tri[0]];
                let v1 = vertices[tri[1]];
                let v2 = vertices[tri[2]];
                (i, Aabb2d::from_triangle(v0, v1, v2))
            })
            .collect();
        let root = Self::build_recursive(&mut items);
        Self { root: Some(root) }
    }

    fn build_recursive(items: &mut [(usize, Aabb2d)]) -> BvhNode {
        if items.len() == 1 {
            return BvhNode::Leaf {
                tri_idx: items[0].0,
                aabb: items[0].1,
            };
        }
        // Compute combined AABB
        let combined = items
            .iter()
            .fold(items[0].1, |acc, (_, b)| Aabb2d::merge(&acc, b));
        // Split on the longer axis
        let axis = if (combined.max[0] - combined.min[0]) >= (combined.max[1] - combined.min[1]) {
            0
        } else {
            1
        };
        items.sort_by(|(_, a), (_, b)| {
            a.centre(axis)
                .partial_cmp(&b.centre(axis))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mid = items.len() / 2;
        let left = Self::build_recursive(&mut items[..mid]);
        let right = Self::build_recursive(&mut items[mid..]);
        BvhNode::Branch {
            aabb: combined,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Query: return triangle indices whose AABB contains point p.
    pub fn query_point(&self, p: [f64; 2]) -> Vec<usize> {
        let mut result = Vec::new();
        if let Some(root) = &self.root {
            Self::query_recursive(root, p, &mut result);
        }
        result
    }

    fn query_recursive(node: &BvhNode, p: [f64; 2], out: &mut Vec<usize>) {
        if !node.aabb().contains(p) {
            return;
        }
        match node {
            BvhNode::Leaf { tri_idx, .. } => out.push(*tri_idx),
            BvhNode::Branch { left, right, .. } => {
                Self::query_recursive(left, p, out);
                Self::query_recursive(right, p, out);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TriMesh2d
// ─────────────────────────────────────────────────────────────────────────────

/// A 2D triangular mesh.
#[derive(Debug, Clone, Default)]
pub struct TriMesh2d {
    /// Node coordinates
    pub nodes: Vec<Node2d>,
    /// Triangle elements
    pub elements: Vec<Triangle>,
    /// Boundary edge node pairs
    pub boundary_edges: Vec<[usize; 2]>,
    /// Material list (names)
    pub materials: Vec<String>,
}

impl TriMesh2d {
    /// Create an empty mesh.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a uniform rectangular mesh over \[0,Lx\] × \[0,Ly\].
    ///
    /// Grid: nx × ny quads, each split into 2 triangles.
    pub fn rectangular(lx: f64, ly: f64, nx: usize, ny: usize) -> Self {
        let mut mesh = Self::new();
        // Create nodes
        for j in 0..=ny {
            for i in 0..=nx {
                mesh.nodes.push(Node2d::new(
                    lx * i as f64 / nx as f64,
                    ly * j as f64 / ny as f64,
                ));
            }
        }
        // Create triangles (split each quad into 2)
        let row_len = nx + 1;
        for j in 0..ny {
            for i in 0..nx {
                let n00 = j * row_len + i;
                let n10 = n00 + 1;
                let n01 = n00 + row_len;
                let n11 = n01 + 1;
                mesh.elements.push(Triangle::new(n00, n10, n11));
                mesh.elements.push(Triangle::new(n00, n11, n01));
            }
        }
        // Boundary edges (perimeter)
        for i in 0..nx {
            mesh.boundary_edges.push([i, i + 1]);
        } // bottom
        for j in 0..ny {
            mesh.boundary_edges
                .push([j * row_len + nx, (j + 1) * row_len + nx]);
        } // right
        for i in (0..nx).rev() {
            mesh.boundary_edges
                .push([ny * row_len + i + 1, ny * row_len + i]);
        } // top
        for j in (0..ny).rev() {
            mesh.boundary_edges.push([(j + 1) * row_len, j * row_len]);
        } // left
        mesh
    }

    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of triangular elements.
    pub fn n_elements(&self) -> usize {
        self.elements.len()
    }

    /// Area of element i (m²).
    pub fn element_area(&self, i: usize) -> f64 {
        let [n0, n1, n2] = self.elements[i].nodes;
        let (x0, y0) = (self.nodes[n0].x, self.nodes[n0].y);
        let (x1, y1) = (self.nodes[n1].x, self.nodes[n1].y);
        let (x2, y2) = (self.nodes[n2].x, self.nodes[n2].y);
        ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)).abs() / 2.0
    }

    /// Total mesh area (sum of element areas).
    pub fn total_area(&self) -> f64 {
        (0..self.n_elements()).map(|i| self.element_area(i)).sum()
    }

    /// Centroid of element i.
    pub fn element_centroid(&self, i: usize) -> Node2d {
        let [n0, n1, n2] = self.elements[i].nodes;
        let x = (self.nodes[n0].x + self.nodes[n1].x + self.nodes[n2].x) / 3.0;
        let y = (self.nodes[n0].y + self.nodes[n1].y + self.nodes[n2].y) / 3.0;
        Node2d::new(x, y)
    }

    /// Find element containing point (x, y) via barycentric coordinates.
    ///
    /// Returns Some(element_index) or None if outside mesh.
    pub fn find_element(&self, x: f64, y: f64) -> Option<usize> {
        for (i, tri) in self.elements.iter().enumerate() {
            let [n0, n1, n2] = tri.nodes;
            let (x0, y0) = (self.nodes[n0].x, self.nodes[n0].y);
            let (x1, y1) = (self.nodes[n1].x, self.nodes[n1].y);
            let (x2, y2) = (self.nodes[n2].x, self.nodes[n2].y);
            let denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
            if denom.abs() < 1e-30 {
                continue;
            }
            let lambda1 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom;
            let lambda2 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom;
            let lambda3 = 1.0 - lambda1 - lambda2;
            if lambda1 >= -1e-10 && lambda2 >= -1e-10 && lambda3 >= -1e-10 {
                return Some(i);
            }
        }
        None
    }

    /// Mesh quality: minimum angle across all triangles (degrees).
    pub fn min_angle_deg(&self) -> f64 {
        let mut min_angle = f64::INFINITY;
        for tri in &self.elements {
            let [n0, n1, n2] = tri.nodes;
            let (x0, y0) = (self.nodes[n0].x, self.nodes[n0].y);
            let (x1, y1) = (self.nodes[n1].x, self.nodes[n1].y);
            let (x2, y2) = (self.nodes[n2].x, self.nodes[n2].y);
            let sides = [
                ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt(),
                ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt(),
                ((x0 - x2).powi(2) + (y0 - y2).powi(2)).sqrt(),
            ];
            // Law of cosines for each angle
            for i in 0..3 {
                let a = sides[i];
                let b = sides[(i + 1) % 3];
                let c = sides[(i + 2) % 3];
                if a < 1e-30 || b < 1e-30 {
                    continue;
                }
                let cos_angle = (b * b + c * c - a * a) / (2.0 * b * c);
                let angle_deg = cos_angle.clamp(-1.0, 1.0).acos().to_degrees();
                min_angle = min_angle.min(angle_deg);
            }
        }
        min_angle
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Quality metrics
    // ─────────────────────────────────────────────────────────────────────────

    /// Compute mesh quality metrics for all elements.
    pub fn quality_metrics(&self) -> MeshQualityReport {
        if self.elements.is_empty() {
            return MeshQualityReport {
                min_aspect_ratio: 0.0,
                max_aspect_ratio: 0.0,
                mean_skewness: 0.0,
                n_degenerate: 0,
            };
        }
        let mut min_ar = f64::INFINITY;
        let mut max_ar = f64::NEG_INFINITY;
        let mut skew_sum = 0.0;
        let mut n_degenerate = 0usize;
        for tri in &self.elements {
            let [n0, n1, n2] = tri.nodes;
            let v0 = [self.nodes[n0].x, self.nodes[n0].y];
            let v1 = [self.nodes[n1].x, self.nodes[n1].y];
            let v2 = [self.nodes[n2].x, self.nodes[n2].y];
            let ar = aspect_ratio(v0, v1, v2);
            let sk = skewness(v0, v1, v2);
            min_ar = min_ar.min(ar);
            max_ar = max_ar.max(ar);
            skew_sum += sk;
            if sk > 0.9 {
                n_degenerate += 1;
            }
        }
        MeshQualityReport {
            min_aspect_ratio: min_ar,
            max_aspect_ratio: max_ar,
            mean_skewness: skew_sum / self.elements.len() as f64,
            n_degenerate,
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Refinement
    // ─────────────────────────────────────────────────────────────────────────

    /// Refine the `n_to_refine` triangles with the highest aspect ratios by
    /// splitting each at the midpoint of its longest edge.
    ///
    /// Each refined triangle is replaced by two children that share the new
    /// midpoint node.
    pub fn refine_worst(&mut self, n_to_refine: usize) {
        if n_to_refine == 0 || self.elements.is_empty() {
            return;
        }
        // Rank elements by aspect ratio (highest first)
        let mut ranked: Vec<(usize, f64)> = self
            .elements
            .iter()
            .enumerate()
            .map(|(i, tri)| {
                let [n0, n1, n2] = tri.nodes;
                let v0 = [self.nodes[n0].x, self.nodes[n0].y];
                let v1 = [self.nodes[n1].x, self.nodes[n1].y];
                let v2 = [self.nodes[n2].x, self.nodes[n2].y];
                (i, aspect_ratio(v0, v1, v2))
            })
            .collect();
        ranked.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let to_refine_count = n_to_refine.min(self.elements.len());
        // Collect indices to refine (highest AR first), sorted descending so
        // that we can remove them without shifting earlier indices.
        let mut indices: Vec<usize> = ranked[..to_refine_count].iter().map(|(i, _)| *i).collect();
        indices.sort_by(|a, b| b.cmp(a)); // descending order for safe swap-remove

        let mut new_tris: Vec<Triangle> = Vec::new();

        for &idx in &indices {
            let tri = self.elements[idx];
            let [n0, n1, n2] = tri.nodes;
            let v0 = [self.nodes[n0].x, self.nodes[n0].y];
            let v1 = [self.nodes[n1].x, self.nodes[n1].y];
            let v2 = [self.nodes[n2].x, self.nodes[n2].y];

            let mp = longest_edge_midpoint(v0, v1, v2);
            let mid_idx = self.nodes.len();
            self.nodes.push(Node2d::new(mp[0], mp[1]));

            // Determine which edge is longest and split accordingly
            let l01 = edge_len(v0, v1);
            let l12 = edge_len(v1, v2);
            let l20 = edge_len(v2, v0);
            let mat = tri.material;
            let (t0, t1) = if l01 >= l12 && l01 >= l20 {
                // split edge (n0,n1)
                (
                    Triangle::with_material(n0, mid_idx, n2, mat),
                    Triangle::with_material(mid_idx, n1, n2, mat),
                )
            } else if l12 >= l20 {
                // split edge (n1,n2)
                (
                    Triangle::with_material(n0, n1, mid_idx, mat),
                    Triangle::with_material(n0, mid_idx, n2, mat),
                )
            } else {
                // split edge (n2,n0)
                (
                    Triangle::with_material(n0, n1, mid_idx, mat),
                    Triangle::with_material(mid_idx, n1, n2, mat),
                )
            };
            // Replace the original slot with t0, push t1 later
            self.elements[idx] = t0;
            new_tris.push(t1);
        }
        self.elements.extend(new_tris);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BVH convenience
    // ─────────────────────────────────────────────────────────────────────────

    /// Build a BVH over this mesh for fast point queries.
    pub fn build_bvh(&self) -> Bvh2d {
        let tris: Vec<[usize; 3]> = self.elements.iter().map(|t| t.nodes).collect();
        let verts: Vec<[f64; 2]> = self.nodes.iter().map(|n| [n.x, n.y]).collect();
        Bvh2d::build(&tris, &verts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trimesh_rectangular_node_count() {
        let m = TriMesh2d::rectangular(1.0, 1.0, 4, 4);
        assert_eq!(m.n_nodes(), 25); // (4+1)*(4+1)
    }

    #[test]
    fn trimesh_rectangular_element_count() {
        let m = TriMesh2d::rectangular(1.0, 1.0, 4, 4);
        assert_eq!(m.n_elements(), 32); // 4*4*2
    }

    #[test]
    fn trimesh_total_area_correct() {
        let m = TriMesh2d::rectangular(2.0, 3.0, 4, 6);
        assert!((m.total_area() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn trimesh_find_element_center() {
        let m = TriMesh2d::rectangular(1.0, 1.0, 4, 4);
        assert!(m.find_element(0.5, 0.5).is_some());
    }

    #[test]
    fn trimesh_find_element_outside() {
        let m = TriMesh2d::rectangular(1.0, 1.0, 4, 4);
        assert!(m.find_element(2.0, 0.5).is_none());
    }

    #[test]
    fn trimesh_min_angle_positive() {
        let m = TriMesh2d::rectangular(1.0, 1.0, 4, 4);
        assert!(m.min_angle_deg() > 0.0);
        assert!(m.min_angle_deg() <= 90.0);
    }

    #[test]
    fn trimesh_centroid_inside() {
        let m = TriMesh2d::rectangular(1.0, 1.0, 4, 4);
        let c = m.element_centroid(0);
        assert!(m.find_element(c.x, c.y).is_some());
    }

    // ── aspect_ratio / skewness ─────────────────────────────────────────────

    #[test]
    fn aspect_ratio_equilateral() {
        // Equilateral triangle: AR = 2/sqrt(3) ≈ 1.1547
        let v0 = [0.0_f64, 0.0];
        let v1 = [1.0, 0.0];
        let v2 = [0.5, 3.0_f64.sqrt() / 2.0];
        let ar = aspect_ratio(v0, v1, v2);
        // All edges equal ⟹ longest = 1, all altitudes = sqrt(3)/2
        // AR = 1 / (sqrt(3)/2) = 2/sqrt(3)
        let expected = 2.0 / 3.0_f64.sqrt();
        assert!((ar - expected).abs() < 1e-9, "AR={ar}, expected≈{expected}");
    }

    #[test]
    fn aspect_ratio_right_isosceles() {
        // Right isosceles: legs 1, hypotenuse sqrt(2)
        let ar = aspect_ratio([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]);
        assert!(ar > 1.0, "AR should be > 1 for non-equilateral, got {ar}");
    }

    #[test]
    fn skewness_equilateral_near_zero() {
        let v0 = [0.0_f64, 0.0];
        let v1 = [1.0, 0.0];
        let v2 = [0.5, 3.0_f64.sqrt() / 2.0];
        let sk = skewness(v0, v1, v2);
        assert!(sk < 1e-9, "skewness of equilateral should be ~0, got {sk}");
    }

    #[test]
    fn skewness_degenerate_near_one() {
        // Very flat triangle
        let sk = skewness([0.0, 0.0], [1.0, 0.0], [0.5, 1e-10]);
        assert!(
            sk > 0.9,
            "skewness of flat triangle should be >0.9, got {sk}"
        );
    }

    // ── longest_edge_midpoint ───────────────────────────────────────────────

    #[test]
    fn longest_edge_midpoint_correct() {
        // Edge (0,0)-(2,0) has length 2, others <2
        let mp = longest_edge_midpoint([0.0, 0.0], [2.0, 0.0], [1.0, 0.01]);
        assert!((mp[0] - 1.0).abs() < 1e-12);
        assert!((mp[1] - 0.0).abs() < 1e-12);
    }

    // ── quality_metrics ─────────────────────────────────────────────────────

    #[test]
    fn quality_metrics_rectangular_mesh() {
        let m = TriMesh2d::rectangular(1.0, 1.0, 4, 4);
        let q = m.quality_metrics();
        assert!(q.min_aspect_ratio > 0.0);
        assert!(q.max_aspect_ratio >= q.min_aspect_ratio);
        assert!(q.mean_skewness >= 0.0 && q.mean_skewness <= 1.0);
        assert_eq!(q.n_degenerate, 0); // rectangular mesh has no degenerate tris
    }

    // ── refine_worst ────────────────────────────────────────────────────────

    #[test]
    fn refine_worst_increases_element_count() {
        let mut m = TriMesh2d::rectangular(1.0, 1.0, 2, 2);
        let before = m.n_elements();
        m.refine_worst(2);
        // Each refined triangle is split into 2, net +1 per triangle
        assert_eq!(m.n_elements(), before + 2);
    }

    #[test]
    fn refine_worst_zero_noops() {
        let mut m = TriMesh2d::rectangular(1.0, 1.0, 2, 2);
        let before = m.n_elements();
        m.refine_worst(0);
        assert_eq!(m.n_elements(), before);
    }

    // ── Aabb2d ──────────────────────────────────────────────────────────────

    #[test]
    fn aabb_contains_centre() {
        let b = Aabb2d::new([0.0, 0.0], [2.0, 2.0]);
        assert!(b.contains([1.0, 1.0]));
        assert!(!b.contains([3.0, 1.0]));
    }

    #[test]
    fn aabb_intersects() {
        let a = Aabb2d::new([0.0, 0.0], [2.0, 2.0]);
        let b = Aabb2d::new([1.0, 1.0], [3.0, 3.0]);
        let c = Aabb2d::new([5.0, 5.0], [6.0, 6.0]);
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn aabb_expand() {
        let b = Aabb2d::new([1.0, 1.0], [2.0, 2.0]);
        let e = b.expand(0.5);
        assert!((e.min[0] - 0.5).abs() < 1e-12);
        assert!((e.max[0] - 2.5).abs() < 1e-12);
    }

    // ── Bvh2d ───────────────────────────────────────────────────────────────

    #[test]
    fn bvh_query_finds_triangle() {
        let verts: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let tris: Vec<[usize; 3]> = vec![[0, 1, 2]];
        let bvh = Bvh2d::build(&tris, &verts);
        let hits = bvh.query_point([0.1, 0.1]);
        assert!(hits.contains(&0));
    }

    #[test]
    fn bvh_query_misses_far_point() {
        let verts: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let tris: Vec<[usize; 3]> = vec![[0, 1, 2]];
        let bvh = Bvh2d::build(&tris, &verts);
        let hits = bvh.query_point([5.0, 5.0]);
        assert!(hits.is_empty());
    }

    #[test]
    fn bvh_build_on_mesh() {
        let m = TriMesh2d::rectangular(1.0, 1.0, 4, 4);
        let bvh = m.build_bvh();
        let hits = bvh.query_point([0.5, 0.5]);
        assert!(!hits.is_empty());
    }
}
