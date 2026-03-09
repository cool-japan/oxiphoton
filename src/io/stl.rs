//! STL (STereoLithography) mesh format for 3D geometry export.
//!
//! STL stores a triangular mesh as:
//!   - ASCII format: human readable
//!   - Binary format: compact (80-byte header + n_tri×50 bytes)
//!
//! This module implements ASCII STL export and a simple triangle mesh builder.
//!
//! STL is used to export photonic device geometries for:
//!   - 3D printing (prototyping)
//!   - FEM meshing tools
//!   - Visualization in Paraview/Blender

/// A 3D vertex.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vertex3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Compute unit normal via cross product (v1-v0) × (v2-v0).
    pub fn normal(v0: Self, v1: Self, v2: Self) -> Self {
        let ax = v1.x - v0.x;
        let ay = v1.y - v0.y;
        let az = v1.z - v0.z;
        let bx = v2.x - v0.x;
        let by = v2.y - v0.y;
        let bz = v2.z - v0.z;
        let nx = ay * bz - az * by;
        let ny = az * bx - ax * bz;
        let nz = ax * by - ay * bx;
        let len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-30);
        Self {
            x: nx / len,
            y: ny / len,
            z: nz / len,
        }
    }
}

/// A single STL triangle (facet) with normal and three vertices.
#[derive(Debug, Clone, Copy)]
pub struct StlTriangle {
    pub normal: Vertex3,
    pub v0: Vertex3,
    pub v1: Vertex3,
    pub v2: Vertex3,
}

impl StlTriangle {
    /// Create a triangle, computing the normal automatically.
    pub fn new(v0: Vertex3, v1: Vertex3, v2: Vertex3) -> Self {
        let normal = Vertex3::normal(v0, v1, v2);
        Self { normal, v0, v1, v2 }
    }

    /// Area of this triangle.
    pub fn area(&self) -> f32 {
        let ax = self.v1.x - self.v0.x;
        let ay = self.v1.y - self.v0.y;
        let az = self.v1.z - self.v0.z;
        let bx = self.v2.x - self.v0.x;
        let by = self.v2.y - self.v0.y;
        let bz = self.v2.z - self.v0.z;
        let cx = ay * bz - az * by;
        let cy = az * bx - ax * bz;
        let cz = ax * by - ay * bx;
        0.5 * (cx * cx + cy * cy + cz * cz).sqrt()
    }
}

/// An STL mesh: a collection of triangles.
#[derive(Debug, Clone, Default)]
pub struct StlMesh {
    pub triangles: Vec<StlTriangle>,
    pub name: String,
}

impl StlMesh {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            triangles: Vec::new(),
            name: name.into(),
        }
    }

    pub fn add_triangle(&mut self, tri: StlTriangle) {
        self.triangles.push(tri);
    }

    /// Add a rectangular face (two triangles) at z = z_val.
    pub fn add_rect_face(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, z: f32) {
        let v00 = Vertex3::new(x0, y0, z);
        let v10 = Vertex3::new(x1, y0, z);
        let v11 = Vertex3::new(x1, y1, z);
        let v01 = Vertex3::new(x0, y1, z);
        self.add_triangle(StlTriangle::new(v00, v10, v11));
        self.add_triangle(StlTriangle::new(v00, v11, v01));
    }

    /// Add an extruded rectangular prism (box).
    pub fn add_box(&mut self, x0: f32, y0: f32, z0: f32, x1: f32, y1: f32, z1: f32) {
        // Top and bottom faces
        self.add_rect_face(x0, y0, x1, y1, z1);
        self.add_rect_face(x1, y0, x0, y1, z0); // reversed for outward normal

        // Side faces
        // Front (y=y0)
        let v000 = Vertex3::new(x0, y0, z0);
        let v100 = Vertex3::new(x1, y0, z0);
        let v101 = Vertex3::new(x1, y0, z1);
        let v001 = Vertex3::new(x0, y0, z1);
        self.add_triangle(StlTriangle::new(v000, v101, v100));
        self.add_triangle(StlTriangle::new(v000, v001, v101));

        // Back (y=y1)
        let v010 = Vertex3::new(x0, y1, z0);
        let v110 = Vertex3::new(x1, y1, z0);
        let v111 = Vertex3::new(x1, y1, z1);
        let v011 = Vertex3::new(x0, y1, z1);
        self.add_triangle(StlTriangle::new(v010, v110, v111));
        self.add_triangle(StlTriangle::new(v010, v111, v011));

        // Left (x=x0)
        self.add_triangle(StlTriangle::new(v000, v010, v011));
        self.add_triangle(StlTriangle::new(v000, v011, v001));

        // Right (x=x1)
        self.add_triangle(StlTriangle::new(v100, v101, v111));
        self.add_triangle(StlTriangle::new(v100, v111, v110));
    }

    pub fn n_triangles(&self) -> usize {
        self.triangles.len()
    }

    /// Total surface area of the mesh.
    pub fn total_area(&self) -> f32 {
        self.triangles.iter().map(|t| t.area()).sum()
    }
}

/// ASCII STL writer.
pub struct StlWriter;

impl StlWriter {
    /// Write mesh to ASCII STL format string.
    pub fn write_ascii(mesh: &StlMesh) -> String {
        let mut out = String::new();
        out.push_str(&format!("solid {}\n", mesh.name));
        for tri in &mesh.triangles {
            out.push_str(&format!(
                "  facet normal {:.6e} {:.6e} {:.6e}\n",
                tri.normal.x, tri.normal.y, tri.normal.z
            ));
            out.push_str("    outer loop\n");
            out.push_str(&format!(
                "      vertex {:.6e} {:.6e} {:.6e}\n",
                tri.v0.x, tri.v0.y, tri.v0.z
            ));
            out.push_str(&format!(
                "      vertex {:.6e} {:.6e} {:.6e}\n",
                tri.v1.x, tri.v1.y, tri.v1.z
            ));
            out.push_str(&format!(
                "      vertex {:.6e} {:.6e} {:.6e}\n",
                tri.v2.x, tri.v2.y, tri.v2.z
            ));
            out.push_str("    endloop\n");
            out.push_str("  endfacet\n");
        }
        out.push_str(&format!("endsolid {}\n", mesh.name));
        out
    }

    /// Write as bytes.
    pub fn write_bytes(mesh: &StlMesh) -> Vec<u8> {
        Self::write_ascii(mesh).into_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stl_triangle_area_unit() {
        // Right triangle with legs 1,1 in z=0 plane
        let t = StlTriangle::new(
            Vertex3::new(0.0, 0.0, 0.0),
            Vertex3::new(1.0, 0.0, 0.0),
            Vertex3::new(0.0, 1.0, 0.0),
        );
        assert!((t.area() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn stl_mesh_box_12_triangles() {
        let mut mesh = StlMesh::new("box");
        mesh.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
        assert_eq!(mesh.n_triangles(), 12);
    }

    #[test]
    fn stl_mesh_total_area_box() {
        let mut mesh = StlMesh::new("box");
        mesh.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
        // Surface area of unit cube = 6
        assert!((mesh.total_area() - 6.0).abs() < 0.01);
    }

    #[test]
    fn stl_writer_ascii_contains_solid() {
        let mut mesh = StlMesh::new("test_solid");
        mesh.add_rect_face(0.0, 0.0, 1.0, 1.0, 0.0);
        let txt = StlWriter::write_ascii(&mesh);
        assert!(txt.starts_with("solid test_solid"));
        assert!(txt.contains("endsolid test_solid"));
        assert!(txt.contains("facet normal"));
        assert!(txt.contains("vertex"));
    }

    #[test]
    fn stl_writer_bytes_nonempty() {
        let mut mesh = StlMesh::new("test");
        mesh.add_rect_face(0.0, 0.0, 1.0, 1.0, 0.0);
        let bytes = StlWriter::write_bytes(&mesh);
        assert!(!bytes.is_empty());
    }

    #[test]
    fn vertex_normal_z_axis() {
        let v0 = Vertex3::new(0.0, 0.0, 0.0);
        let v1 = Vertex3::new(1.0, 0.0, 0.0);
        let v2 = Vertex3::new(0.0, 1.0, 0.0);
        let n = Vertex3::normal(v0, v1, v2);
        assert!(
            n.z.abs() > 0.99,
            "normal should be ~z: ({},{},{})",
            n.x,
            n.y,
            n.z
        );
    }
}
