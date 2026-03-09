//! GDSII layout format support for photonic device design.
//!
//! GDSII (Graphic Database System II) is the de-facto standard for IC/photonic
//! mask layout. A GDS file contains:
//!   - Library with metadata (database units, user units)
//!   - Structures (cells) containing elements:
//!     - Boundary: filled polygon
//!     - Path: stroked polygon with width
//!     - SREF: structure reference (cell placement)
//!     - AREF: array reference (periodic placement)
//!     - Text: labels
//!
//! This module provides:
//!   - In-memory representation: `GdsLibrary`, `GdsCell`, `GdsElement`
//!   - Simple ASCII-style writer (OASIS is binary; this uses text DSL for portability)
//!   - Reader stub for common patterns

/// A 2D integer coordinate in GDSII units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdsPoint {
    pub x: i32,
    pub y: i32,
}

impl GdsPoint {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// Create from floating-point user units given database unit (e.g. 1 nm = 1).
    pub fn from_um(x_um: f64, y_um: f64, db_per_um: f64) -> Self {
        Self {
            x: (x_um * db_per_um).round() as i32,
            y: (y_um * db_per_um).round() as i32,
        }
    }
}

/// A GDS layer/datatype pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdsLayer {
    pub layer: u16,
    pub datatype: u16,
}

impl GdsLayer {
    pub fn new(layer: u16, datatype: u16) -> Self {
        Self { layer, datatype }
    }
}

/// A GDSII polygon boundary element.
#[derive(Debug, Clone)]
pub struct GdsBoundary {
    pub layer: GdsLayer,
    pub points: Vec<GdsPoint>,
}

impl GdsBoundary {
    /// Create a rectangular boundary.
    pub fn rectangle(layer: GdsLayer, x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        Self {
            layer,
            points: vec![
                GdsPoint::new(x0, y0),
                GdsPoint::new(x1, y0),
                GdsPoint::new(x1, y1),
                GdsPoint::new(x0, y1),
                GdsPoint::new(x0, y0), // closed
            ],
        }
    }
}

/// A GDSII path element.
#[derive(Debug, Clone)]
pub struct GdsPath {
    pub layer: GdsLayer,
    pub width: i32,
    pub points: Vec<GdsPoint>,
}

/// A GDSII structure reference (cell placement).
#[derive(Debug, Clone)]
pub struct GdsSref {
    /// Referenced cell name.
    pub sname: String,
    pub origin: GdsPoint,
    /// Rotation angle in degrees (counter-clockwise).
    pub angle_deg: f64,
    /// Magnification (1.0 = no scaling).
    pub magnification: f64,
    /// If true, reflect about x-axis before rotation.
    pub x_reflection: bool,
}

impl GdsSref {
    pub fn new(sname: impl Into<String>, origin: GdsPoint) -> Self {
        Self {
            sname: sname.into(),
            origin,
            angle_deg: 0.0,
            magnification: 1.0,
            x_reflection: false,
        }
    }
}

/// A GDSII text label.
#[derive(Debug, Clone)]
pub struct GdsText {
    pub layer: GdsLayer,
    pub string: String,
    pub origin: GdsPoint,
    pub height: i32,
}

/// Union of GDSII element types.
#[derive(Debug, Clone)]
pub enum GdsElement {
    Boundary(GdsBoundary),
    Path(GdsPath),
    Sref(GdsSref),
    Text(GdsText),
}

/// A GDSII cell (structure).
#[derive(Debug, Clone)]
pub struct GdsCell {
    pub name: String,
    pub elements: Vec<GdsElement>,
}

impl GdsCell {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            elements: Vec::new(),
        }
    }

    pub fn add(&mut self, element: GdsElement) -> &mut Self {
        self.elements.push(element);
        self
    }

    /// Add a rectangle on the given layer.
    pub fn add_rect(&mut self, layer: GdsLayer, x0: i32, y0: i32, x1: i32, y1: i32) -> &mut Self {
        self.add(GdsElement::Boundary(GdsBoundary::rectangle(
            layer, x0, y0, x1, y1,
        )))
    }

    /// Add a cell reference.
    pub fn add_sref(&mut self, sname: impl Into<String>, origin: GdsPoint) -> &mut Self {
        self.add(GdsElement::Sref(GdsSref::new(sname, origin)))
    }

    pub fn n_elements(&self) -> usize {
        self.elements.len()
    }
}

/// A GDSII library.
#[derive(Debug, Clone)]
pub struct GdsLibrary {
    pub name: String,
    /// Database unit in meters (e.g. 1e-9 for 1 nm grid).
    pub db_unit_m: f64,
    /// User unit in meters (e.g. 1e-6 for µm display).
    pub user_unit_m: f64,
    pub cells: Vec<GdsCell>,
}

impl GdsLibrary {
    /// Create a new library with 1 nm database unit and 1 µm user unit.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            db_unit_m: 1e-9,
            user_unit_m: 1e-6,
            cells: Vec::new(),
        }
    }

    /// Database units per micron.
    pub fn db_per_um(&self) -> f64 {
        self.user_unit_m / self.db_unit_m
    }

    pub fn add_cell(&mut self, cell: GdsCell) -> &mut Self {
        self.cells.push(cell);
        self
    }

    pub fn find_cell(&self, name: &str) -> Option<&GdsCell> {
        self.cells.iter().find(|c| c.name == name)
    }

    pub fn n_cells(&self) -> usize {
        self.cells.len()
    }

    /// Total number of elements across all cells.
    pub fn total_elements(&self) -> usize {
        self.cells.iter().map(|c| c.n_elements()).sum()
    }
}

/// Simple text-based GDS writer (produces a human-readable representation).
///
/// Real GDS binary writing would require the full GDSII stream format.
/// This implementation writes a structured text format for debug/review.
pub struct GdsWriter {
    output: String,
}

impl GdsWriter {
    pub fn new() -> Self {
        Self {
            output: String::new(),
        }
    }

    /// Write library to text format, returning the string.
    pub fn write_library(&mut self, lib: &GdsLibrary) -> &str {
        self.output.clear();
        self.output.push_str(&format!(
            "LIBRARY {} db_unit={:.2e}m user_unit={:.2e}m\n",
            lib.name, lib.db_unit_m, lib.user_unit_m
        ));
        for cell in &lib.cells {
            self.write_cell(cell);
        }
        self.output.push_str("ENDLIB\n");
        &self.output
    }

    fn write_cell(&mut self, cell: &GdsCell) {
        self.output.push_str(&format!("  CELL {}\n", cell.name));
        for elem in &cell.elements {
            match elem {
                GdsElement::Boundary(b) => {
                    self.output.push_str(&format!(
                        "    BOUNDARY layer={}/{} pts={}\n",
                        b.layer.layer,
                        b.layer.datatype,
                        b.points.len()
                    ));
                }
                GdsElement::Path(p) => {
                    self.output.push_str(&format!(
                        "    PATH layer={}/{} width={} pts={}\n",
                        p.layer.layer,
                        p.layer.datatype,
                        p.width,
                        p.points.len()
                    ));
                }
                GdsElement::Sref(s) => {
                    self.output.push_str(&format!(
                        "    SREF {} at ({},{})\n",
                        s.sname, s.origin.x, s.origin.y
                    ));
                }
                GdsElement::Text(t) => {
                    self.output.push_str(&format!(
                        "    TEXT \"{}\" at ({},{})\n",
                        t.string, t.origin.x, t.origin.y
                    ));
                }
            }
        }
        self.output.push_str("  ENDCELL\n");
    }

    pub fn result(&self) -> &str {
        &self.output
    }
}

impl Default for GdsWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple GDS layout builder for silicon photonics.
pub struct SiPhLayout {
    pub lib: GdsLibrary,
}

impl SiPhLayout {
    /// Si waveguide layer (layer 1, datatype 0).
    pub const WG_LAYER: GdsLayer = GdsLayer {
        layer: 1,
        datatype: 0,
    };
    /// Oxide cladding layer (layer 2, datatype 0).
    pub const CLAD_LAYER: GdsLayer = GdsLayer {
        layer: 2,
        datatype: 0,
    };
    /// Metal contact layer (layer 10, datatype 0).
    pub const METAL_LAYER: GdsLayer = GdsLayer {
        layer: 10,
        datatype: 0,
    };

    pub fn new(name: impl Into<String>) -> Self {
        Self {
            lib: GdsLibrary::new(name),
        }
    }

    /// Add a straight waveguide rectangle (in nm coordinates).
    pub fn add_waveguide(
        &mut self,
        cell_name: &str,
        x0_nm: i32,
        y0_nm: i32,
        length_nm: i32,
        width_nm: i32,
    ) {
        if let Some(cell) = self.lib.cells.iter_mut().find(|c| c.name == cell_name) {
            cell.add_rect(
                Self::WG_LAYER,
                x0_nm,
                y0_nm - width_nm / 2,
                x0_nm + length_nm,
                y0_nm + width_nm / 2,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gds_library_new() {
        let lib = GdsLibrary::new("test_lib");
        assert_eq!(lib.name, "test_lib");
        assert_eq!(lib.n_cells(), 0);
        assert!((lib.db_per_um() - 1000.0).abs() < 1.0); // 1µm / 1nm = 1000
    }

    #[test]
    fn gds_cell_add_rect() {
        let mut cell = GdsCell::new("TOP");
        let layer = GdsLayer::new(1, 0);
        cell.add_rect(layer, 0, 0, 1000, 500);
        assert_eq!(cell.n_elements(), 1);
    }

    #[test]
    fn gds_boundary_rectangle_closed() {
        let layer = GdsLayer::new(1, 0);
        let b = GdsBoundary::rectangle(layer, 0, 0, 100, 50);
        assert_eq!(b.points.len(), 5);
        assert_eq!(b.points[0], b.points[4]); // closed polygon
    }

    #[test]
    fn gds_library_find_cell() {
        let mut lib = GdsLibrary::new("test");
        lib.add_cell(GdsCell::new("CELL_A"));
        assert!(lib.find_cell("CELL_A").is_some());
        assert!(lib.find_cell("CELL_B").is_none());
    }

    #[test]
    fn gds_library_total_elements() {
        let mut lib = GdsLibrary::new("test");
        let mut cell = GdsCell::new("A");
        cell.add_rect(GdsLayer::new(1, 0), 0, 0, 100, 100);
        cell.add_rect(GdsLayer::new(2, 0), 200, 0, 300, 100);
        lib.add_cell(cell);
        assert_eq!(lib.total_elements(), 2);
    }

    #[test]
    fn gds_writer_produces_output() {
        let mut lib = GdsLibrary::new("phot_lib");
        let mut cell = GdsCell::new("TOP");
        cell.add_rect(GdsLayer::new(1, 0), 0, 0, 500, 500);
        cell.add_sref("SUB_CELL", GdsPoint::new(100, 200));
        lib.add_cell(cell);

        let mut writer = GdsWriter::new();
        let txt = writer.write_library(&lib).to_string();
        assert!(txt.contains("LIBRARY phot_lib"));
        assert!(txt.contains("CELL TOP"));
        assert!(txt.contains("BOUNDARY"));
        assert!(txt.contains("SREF"));
        assert!(txt.contains("ENDLIB"));
    }

    #[test]
    fn gds_point_from_um() {
        let p = GdsPoint::from_um(1.5, 2.0, 1000.0);
        assert_eq!(p.x, 1500);
        assert_eq!(p.y, 2000);
    }

    #[test]
    fn gds_sref_default_transform() {
        let s = GdsSref::new("SUB", GdsPoint::new(0, 0));
        assert_eq!(s.magnification, 1.0);
        assert_eq!(s.angle_deg, 0.0);
        assert!(!s.x_reflection);
    }
}
