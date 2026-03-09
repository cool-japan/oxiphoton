//! VTK legacy format writer for FDTD field visualization.
//!
//! VTK (Visualization Toolkit) legacy ASCII format:
//!   - Structured Points: regular 2D/3D grid
//!   - Point/Cell data: scalar or vector fields
//!
//! Example output:
//! ```text
//! # vtk DataFile Version 3.0
//! OxiPhoton FDTD Ez field
//! ASCII
//! DATASET STRUCTURED_POINTS
//! DIMENSIONS 100 100 1
//! ORIGIN 0 0 0
//! SPACING 10e-9 10e-9 1
//! POINT_DATA 10000
//! SCALARS Ez float 1
//! LOOKUP_TABLE default
//! 0.0 0.1 0.2 ...
//! ```

/// A 2D scalar field for VTK export.
#[derive(Debug, Clone)]
pub struct VtkField2d {
    /// Field values, stored row-major: field[ix * ny + iy]
    pub data: Vec<f64>,
    /// Grid size (nx, ny)
    pub nx: usize,
    pub ny: usize,
    /// Grid spacing (m)
    pub dx: f64,
    pub dy: f64,
    /// Field name (e.g. "Ez")
    pub name: String,
}

impl VtkField2d {
    /// Create a zero-initialized 2D field.
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64, name: impl Into<String>) -> Self {
        Self {
            data: vec![0.0; nx * ny],
            nx,
            ny,
            dx,
            dy,
            name: name.into(),
        }
    }

    /// Set field value at (ix, iy).
    pub fn set(&mut self, ix: usize, iy: usize, value: f64) {
        if ix < self.nx && iy < self.ny {
            self.data[ix * self.ny + iy] = value;
        }
    }

    /// Get field value at (ix, iy).
    pub fn get(&self, ix: usize, iy: usize) -> f64 {
        self.data[ix * self.ny + iy]
    }

    /// Total number of points.
    pub fn n_points(&self) -> usize {
        self.nx * self.ny
    }

    /// Maximum field value.
    pub fn max_value(&self) -> f64 {
        self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Minimum field value.
    pub fn min_value(&self) -> f64 {
        self.data.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// RMS field value.
    pub fn rms(&self) -> f64 {
        let n = self.data.len() as f64;
        if n == 0.0 {
            return 0.0;
        }
        (self.data.iter().map(|v| v * v).sum::<f64>() / n).sqrt()
    }
}

/// VTK legacy format writer.
pub struct VtkWriter;

impl VtkWriter {
    /// Write a 2D field to VTK structured points ASCII format.
    pub fn write_2d(field: &VtkField2d, title: &str) -> String {
        let mut out = String::new();
        out.push_str("# vtk DataFile Version 3.0\n");
        out.push_str(&format!("{title}\n"));
        out.push_str("ASCII\n");
        out.push_str("DATASET STRUCTURED_POINTS\n");
        out.push_str(&format!("DIMENSIONS {} {} 1\n", field.nx, field.ny));
        out.push_str("ORIGIN 0 0 0\n");
        out.push_str(&format!("SPACING {:.6e} {:.6e} 1\n", field.dx, field.dy));
        out.push_str(&format!("POINT_DATA {}\n", field.n_points()));
        out.push_str(&format!("SCALARS {} float 1\n", field.name));
        out.push_str("LOOKUP_TABLE default\n");

        // Write data in VTK order (y varies fastest for STRUCTURED_POINTS)
        let mut count = 0;
        for ix in 0..field.nx {
            for iy in 0..field.ny {
                out.push_str(&format!("{:.6e}", field.get(ix, iy)));
                count += 1;
                if count % 6 == 0 {
                    out.push('\n');
                } else {
                    out.push(' ');
                }
            }
        }
        if count % 6 != 0 {
            out.push('\n');
        }
        out
    }

    /// Write multiple 2D fields to a single VTK file (multi-scalar).
    pub fn write_2d_multi(fields: &[&VtkField2d], title: &str) -> String {
        if fields.is_empty() {
            return String::new();
        }
        let f0 = fields[0];
        let mut out = String::new();
        out.push_str("# vtk DataFile Version 3.0\n");
        out.push_str(&format!("{title}\n"));
        out.push_str("ASCII\n");
        out.push_str("DATASET STRUCTURED_POINTS\n");
        out.push_str(&format!("DIMENSIONS {} {} 1\n", f0.nx, f0.ny));
        out.push_str("ORIGIN 0 0 0\n");
        out.push_str(&format!("SPACING {:.6e} {:.6e} 1\n", f0.dx, f0.dy));
        out.push_str(&format!("POINT_DATA {}\n", f0.n_points()));

        for field in fields {
            out.push_str(&format!("SCALARS {} float 1\n", field.name));
            out.push_str("LOOKUP_TABLE default\n");
            let mut count = 0;
            for ix in 0..field.nx {
                for iy in 0..field.ny {
                    out.push_str(&format!("{:.6e}", field.get(ix, iy)));
                    count += 1;
                    if count % 6 == 0 {
                        out.push('\n');
                    } else {
                        out.push(' ');
                    }
                }
            }
            if count % 6 != 0 {
                out.push('\n');
            }
        }
        out
    }

    /// Write bytes.
    pub fn write_bytes(field: &VtkField2d, title: &str) -> Vec<u8> {
        Self::write_2d(field, title).into_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vtk_field_2d_set_get() {
        let mut f = VtkField2d::new(10, 10, 1e-9, 1e-9, "Ez");
        f.set(3, 5, 1.5);
        assert!((f.get(3, 5) - 1.5).abs() < 1e-10);
    }

    #[test]
    fn vtk_field_max_min() {
        let mut f = VtkField2d::new(5, 5, 1e-9, 1e-9, "Ez");
        f.set(0, 0, -2.0);
        f.set(4, 4, 3.0);
        assert!((f.max_value() - 3.0).abs() < 1e-10);
        assert!((f.min_value() - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn vtk_field_rms() {
        let mut f = VtkField2d::new(2, 2, 1e-9, 1e-9, "Ez");
        f.set(0, 0, 1.0);
        f.set(0, 1, 1.0);
        f.set(1, 0, 1.0);
        f.set(1, 1, 1.0);
        assert!((f.rms() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn vtk_writer_header() {
        let f = VtkField2d::new(10, 10, 5e-9, 5e-9, "Ez");
        let txt = VtkWriter::write_2d(&f, "Test FDTD");
        assert!(txt.starts_with("# vtk DataFile Version 3.0"));
        assert!(txt.contains("Test FDTD"));
        assert!(txt.contains("STRUCTURED_POINTS"));
        assert!(txt.contains("DIMENSIONS 10 10 1"));
    }

    #[test]
    fn vtk_writer_contains_data() {
        let mut f = VtkField2d::new(3, 3, 5e-9, 5e-9, "Hy");
        f.set(1, 1, 0.5);
        let txt = VtkWriter::write_2d(&f, "Hy field");
        assert!(txt.contains("SCALARS Hy float 1"));
        assert!(txt.contains("5.000000e-1") || txt.contains("5.0e-1") || txt.contains("0.5"));
    }

    #[test]
    fn vtk_writer_multi_fields() {
        let f1 = VtkField2d::new(4, 4, 5e-9, 5e-9, "Ez");
        let f2 = VtkField2d::new(4, 4, 5e-9, 5e-9, "Hy");
        let txt = VtkWriter::write_2d_multi(&[&f1, &f2], "EM fields");
        assert!(txt.contains("SCALARS Ez float 1"));
        assert!(txt.contains("SCALARS Hy float 1"));
    }

    #[test]
    fn vtk_writer_bytes_nonempty() {
        let f = VtkField2d::new(5, 5, 5e-9, 5e-9, "Ez");
        let bytes = VtkWriter::write_bytes(&f, "test");
        assert!(!bytes.is_empty());
    }

    #[test]
    fn vtk_field_n_points() {
        let f = VtkField2d::new(10, 20, 5e-9, 5e-9, "Ez");
        assert_eq!(f.n_points(), 200);
    }
}
