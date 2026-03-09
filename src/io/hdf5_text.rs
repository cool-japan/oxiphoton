//! HDF5-like text format for structured simulation data.
//!
//! HDF5 is a hierarchical binary format common in scientific computing.
//! Since linking to the HDF5 C library adds complexity, this module provides
//! a text-based equivalent for development/testing purposes.
//!
//! Format:
//! ```text
//! HDF5_TEXT v1.0
//! GROUP /
//!   DATASET Ez
//!     DIMS 100 100
//!     DTYPE f64
//!     DATA
//!     0.0 0.1 0.2 ...
//!   END_DATASET
//! END_GROUP
//! ```
//!
//! This is not binary HDF5 but follows the same conceptual hierarchy.

/// A single dataset (named array of f64).
#[derive(Debug, Clone)]
pub struct Hdf5TextDataset {
    pub name: String,
    pub dims: Vec<usize>,
    pub data: Vec<f64>,
    pub attributes: Vec<(String, String)>,
}

impl Hdf5TextDataset {
    /// Create a 1D dataset.
    pub fn new_1d(name: impl Into<String>, data: Vec<f64>) -> Self {
        let n = data.len();
        Self {
            name: name.into(),
            dims: vec![n],
            data,
            attributes: Vec::new(),
        }
    }

    /// Create a 2D dataset (row-major).
    pub fn new_2d(name: impl Into<String>, data: Vec<f64>, nx: usize, ny: usize) -> Self {
        Self {
            name: name.into(),
            dims: vec![nx, ny],
            data,
            attributes: Vec::new(),
        }
    }

    /// Add a metadata attribute (key, value).
    pub fn add_attr(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.attributes.push((key.into(), value.into()));
    }

    /// Total number of elements.
    pub fn n_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// Get element at flat index.
    pub fn get_flat(&self, i: usize) -> f64 {
        self.data[i]
    }

    /// Get element at 2D index (ix, iy) for 2D datasets.
    pub fn get_2d(&self, ix: usize, iy: usize) -> f64 {
        let ny = if self.dims.len() >= 2 {
            self.dims[1]
        } else {
            1
        };
        self.data[ix * ny + iy]
    }

    /// Serialize to text.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        let dims_str: Vec<String> = self.dims.iter().map(|d| d.to_string()).collect();
        out.push_str(&format!("  DATASET {}\n", self.name));
        out.push_str(&format!("    DIMS {}\n", dims_str.join(" ")));
        out.push_str("    DTYPE f64\n");
        for (k, v) in &self.attributes {
            out.push_str(&format!("    ATTR {} = {}\n", k, v));
        }
        out.push_str("    DATA\n");
        let mut count = 0;
        out.push_str("    ");
        for &v in &self.data {
            out.push_str(&format!("{:.6e} ", v));
            count += 1;
            if count % 8 == 0 {
                out.push('\n');
                out.push_str("    ");
            }
        }
        if count % 8 != 0 {
            out.push('\n');
        }
        out.push_str("  END_DATASET\n");
        out
    }
}

/// An HDF5-like text file with grouped datasets.
#[derive(Debug, Clone, Default)]
pub struct Hdf5TextFile {
    pub datasets: Vec<Hdf5TextDataset>,
    pub global_attrs: Vec<(String, String)>,
}

impl Hdf5TextFile {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_dataset(&mut self, ds: Hdf5TextDataset) {
        self.datasets.push(ds);
    }

    pub fn add_global_attr(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.global_attrs.push((key.into(), value.into()));
    }

    pub fn find_dataset(&self, name: &str) -> Option<&Hdf5TextDataset> {
        self.datasets.iter().find(|ds| ds.name == name)
    }

    pub fn n_datasets(&self) -> usize {
        self.datasets.len()
    }

    /// Serialize entire file to text.
    pub fn to_text(&self) -> String {
        let mut out = String::from("HDF5_TEXT v1.0\n");
        out.push_str("GROUP /\n");
        for (k, v) in &self.global_attrs {
            out.push_str(&format!("  ATTR {} = {}\n", k, v));
        }
        for ds in &self.datasets {
            out.push_str(&ds.to_text());
        }
        out.push_str("END_GROUP\n");
        out
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.to_text().into_bytes()
    }

    /// Parse a simple value from the text (reads dataset by name, returns first value).
    /// This is a minimal reader for round-trip testing.
    pub fn read_first_value(&self, name: &str) -> Option<f64> {
        self.find_dataset(name)?.data.first().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_1d_n_elements() {
        let ds = Hdf5TextDataset::new_1d("wavelength", vec![1.0, 1.1, 1.2]);
        assert_eq!(ds.n_elements(), 3);
    }

    #[test]
    fn dataset_2d_get_2d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ds = Hdf5TextDataset::new_2d("Ez", data, 2, 3);
        assert!((ds.get_2d(0, 0) - 1.0).abs() < 1e-10);
        assert!((ds.get_2d(1, 2) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn dataset_to_text_contains_name() {
        let ds = Hdf5TextDataset::new_1d("reflectance", vec![0.1, 0.2]);
        let txt = ds.to_text();
        assert!(txt.contains("DATASET reflectance"));
        assert!(txt.contains("DIMS 2"));
    }

    #[test]
    fn dataset_attr_in_text() {
        let mut ds = Hdf5TextDataset::new_1d("spectrum", vec![1.0]);
        ds.add_attr("units", "m");
        let txt = ds.to_text();
        assert!(txt.contains("ATTR units = m"));
    }

    #[test]
    fn hdf5_file_add_find() {
        let mut file = Hdf5TextFile::new();
        file.add_dataset(Hdf5TextDataset::new_1d("time", vec![0.0, 1e-15, 2e-15]));
        assert_eq!(file.n_datasets(), 1);
        assert!(file.find_dataset("time").is_some());
        assert!(file.find_dataset("missing").is_none());
    }

    #[test]
    fn hdf5_file_to_text_structure() {
        let mut file = Hdf5TextFile::new();
        file.add_global_attr("author", "oxiphoton");
        file.add_dataset(Hdf5TextDataset::new_1d("Ez", vec![0.0, 1.0]));
        let txt = file.to_text();
        assert!(txt.starts_with("HDF5_TEXT v1.0"));
        assert!(txt.contains("GROUP /"));
        assert!(txt.contains("ATTR author = oxiphoton"));
        assert!(txt.contains("DATASET Ez"));
        assert!(txt.contains("END_GROUP"));
    }

    #[test]
    fn hdf5_file_bytes_nonempty() {
        let file = Hdf5TextFile::new();
        let bytes = file.to_bytes();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn hdf5_read_first_value() {
        let mut file = Hdf5TextFile::new();
        file.add_dataset(Hdf5TextDataset::new_1d(
            "test",
            vec![std::f64::consts::PI, 2.71],
        ));
        let v = file.read_first_value("test").unwrap();
        assert!((v - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn hdf5_read_missing_returns_none() {
        let file = Hdf5TextFile::new();
        assert!(file.read_first_value("missing").is_none());
    }
}
