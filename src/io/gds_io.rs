//! GDS I/O utilities (text-based representation).
//!
//! Exports simulation geometry to a human-readable GDS-like text format.
//! For full binary GDSII, an external tool (gdspy, KLayout) would be used.

use crate::geometry::gds::{GdsLibrary, GdsWriter};

/// Exports a GDS library to a text string.
pub struct GdsTextExporter;

impl GdsTextExporter {
    /// Export `lib` to text format, returning the string.
    pub fn export(lib: &GdsLibrary) -> String {
        let mut writer = GdsWriter::new();
        writer.write_library(lib).to_string()
    }

    /// Export to a `Vec<u8>` (UTF-8 bytes) for file writing.
    pub fn export_bytes(lib: &GdsLibrary) -> Vec<u8> {
        Self::export(lib).into_bytes()
    }

    /// Count total polygons (boundaries) across all cells.
    pub fn count_polygons(lib: &GdsLibrary) -> usize {
        use crate::geometry::gds::GdsElement;
        lib.cells
            .iter()
            .flat_map(|c| &c.elements)
            .filter(|e| matches!(e, GdsElement::Boundary(_)))
            .count()
    }

    /// Estimate file size (bytes) for the text export.
    pub fn estimated_size_bytes(lib: &GdsLibrary) -> usize {
        // Rough estimate: ~100 bytes per element
        lib.total_elements() * 100 + lib.n_cells() * 50
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::gds::{GdsCell, GdsLayer, GdsLibrary};

    #[test]
    fn gds_exporter_nonempty() {
        let mut lib = GdsLibrary::new("test");
        let mut cell = GdsCell::new("TOP");
        cell.add_rect(GdsLayer::new(1, 0), 0, 0, 100, 100);
        lib.add_cell(cell);
        let txt = GdsTextExporter::export(&lib);
        assert!(!txt.is_empty());
        assert!(txt.contains("LIBRARY test"));
    }

    #[test]
    fn gds_exporter_count_polygons() {
        let mut lib = GdsLibrary::new("test");
        let mut cell = GdsCell::new("TOP");
        cell.add_rect(GdsLayer::new(1, 0), 0, 0, 100, 100);
        cell.add_rect(GdsLayer::new(2, 0), 200, 0, 300, 100);
        lib.add_cell(cell);
        assert_eq!(GdsTextExporter::count_polygons(&lib), 2);
    }

    #[test]
    fn gds_exporter_bytes_nonempty() {
        let lib = GdsLibrary::new("empty");
        let bytes = GdsTextExporter::export_bytes(&lib);
        assert!(!bytes.is_empty());
    }

    #[test]
    fn gds_estimated_size() {
        let mut lib = GdsLibrary::new("test");
        let mut cell = GdsCell::new("TOP");
        for _ in 0..10 {
            cell.add_rect(GdsLayer::new(1, 0), 0, 0, 100, 100);
        }
        lib.add_cell(cell);
        let sz = GdsTextExporter::estimated_size_bytes(&lib);
        assert!(sz > 0);
    }
}
