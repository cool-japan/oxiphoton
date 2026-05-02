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
//!   - Text-format reader: `GdsReader::parse()` — round-trips with `GdsWriter`
//!     (note: binary GDSII stream parsing is not yet supported)

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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
pub struct GdsPath {
    pub layer: GdsLayer,
    pub width: i32,
    pub points: Vec<GdsPoint>,
}

/// A GDSII structure reference (cell placement).
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
pub struct GdsText {
    pub layer: GdsLayer,
    pub string: String,
    pub origin: GdsPoint,
    pub height: i32,
}

/// A GDSII array reference (periodic placement of a cell).
#[derive(Debug, Clone, PartialEq)]
pub struct GdsAref {
    /// Referenced cell name.
    pub ref_name: String,
    /// Number of columns.
    pub cols: u16,
    /// Number of rows.
    pub rows: u16,
    /// Rotation angle in degrees (counter-clockwise).
    pub angle_deg: f64,
    /// Magnification (1.0 = no scaling).
    pub magnification: f64,
    /// If true, reflect about x-axis before rotation.
    pub x_reflection: bool,
    /// Exactly 3 points: [origin, col-displacement, row-displacement].
    pub xy: Vec<GdsPoint>,
}

impl GdsAref {
    /// Create a new AREF with default transform (no rotation, no magnification).
    pub fn new(
        ref_name: impl Into<String>,
        cols: u16,
        rows: u16,
        origin: GdsPoint,
        col_pt: GdsPoint,
        row_pt: GdsPoint,
    ) -> Self {
        Self {
            ref_name: ref_name.into(),
            cols,
            rows,
            angle_deg: 0.0,
            magnification: 1.0,
            x_reflection: false,
            xy: vec![origin, col_pt, row_pt],
        }
    }
}

/// Union of GDSII element types.
#[derive(Debug, Clone, PartialEq)]
pub enum GdsElement {
    Boundary(GdsBoundary),
    Path(GdsPath),
    Sref(GdsSref),
    Aref(GdsAref),
    Text(GdsText),
}

/// A GDSII cell (structure).
#[derive(Debug, Clone, PartialEq)]
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

    /// Add an array reference.
    pub fn add_aref(&mut self, aref: GdsAref) {
        self.elements.push(GdsElement::Aref(aref));
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

impl PartialEq for GdsLibrary {
    fn eq(&self, other: &Self) -> bool {
        if self.name != other.name || self.cells != other.cells {
            return false;
        }
        // Unit values are stored in IBM real8 (hex float) which introduces tiny
        // rounding errors relative to IEEE 754 binary. Allow 1e-12 relative tolerance.
        let units_eq = |a: f64, b: f64| -> bool {
            if a == 0.0 && b == 0.0 {
                return true;
            }
            let denom = a.abs().max(b.abs());
            if denom == 0.0 {
                return false;
            }
            (a - b).abs() / denom < 1e-12
        };
        units_eq(self.db_unit_m, other.db_unit_m) && units_eq(self.user_unit_m, other.user_unit_m)
    }
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

/// Error type for GDS text-format parsing.
#[derive(Debug, thiserror::Error)]
pub enum GdsParseError {
    #[error("unexpected end of input while parsing {context}")]
    UnexpectedEof { context: String },
    #[error("unexpected line {line:?} while parsing {context}")]
    UnexpectedLine { line: String, context: String },
    #[error("malformed field {field:?}: {reason}")]
    MalformedField { field: String, reason: String },
}

/// Simple text-based GDS writer (produces a human-readable representation).
///
/// Real GDS binary writing would require the full GDSII stream format.
/// This implementation writes a structured text format for debug/review.
/// All geometry fields (including transform attributes) are preserved,
/// enabling lossless round-trips via [`GdsReader`].
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
                    let pts: String = b
                        .points
                        .iter()
                        .map(|p| format!("({},{})", p.x, p.y))
                        .collect::<Vec<_>>()
                        .join(" ");
                    self.output.push_str(&format!(
                        "    BOUNDARY layer={}/{} points={}\n",
                        b.layer.layer, b.layer.datatype, pts
                    ));
                }
                GdsElement::Path(p) => {
                    let pts: String = p
                        .points
                        .iter()
                        .map(|pt| format!("({},{})", pt.x, pt.y))
                        .collect::<Vec<_>>()
                        .join(" ");
                    self.output.push_str(&format!(
                        "    PATH layer={}/{} width={} points={}\n",
                        p.layer.layer, p.layer.datatype, p.width, pts
                    ));
                }
                GdsElement::Sref(s) => {
                    self.output.push_str(&format!(
                        "    SREF {} at ({},{}) angle={} mag={} xrefl={}\n",
                        s.sname,
                        s.origin.x,
                        s.origin.y,
                        s.angle_deg,
                        s.magnification,
                        s.x_reflection as u8,
                    ));
                }
                GdsElement::Aref(a) => {
                    let origin = a.xy.first().map_or(GdsPoint::new(0, 0), |p| *p);
                    let col_pt = a.xy.get(1).map_or(GdsPoint::new(0, 0), |p| *p);
                    let row_pt = a.xy.get(2).map_or(GdsPoint::new(0, 0), |p| *p);
                    self.output.push_str(&format!(
                        "    AREF {} cols={} rows={} at ({},{}) col_pt=({},{}) row_pt=({},{}) angle={} mag={} xrefl={}\n",
                        a.ref_name,
                        a.cols,
                        a.rows,
                        origin.x,
                        origin.y,
                        col_pt.x,
                        col_pt.y,
                        row_pt.x,
                        row_pt.y,
                        a.angle_deg,
                        a.magnification,
                        a.x_reflection as u8,
                    ));
                }
                GdsElement::Text(t) => {
                    self.output.push_str(&format!(
                        "    TEXT \"{}\" layer={}/{} at ({},{}) height={}\n",
                        t.string, t.layer.layer, t.layer.datatype, t.origin.x, t.origin.y, t.height,
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

/// Text-format GDS reader (parses the output of [`GdsWriter`]).
///
/// This reader handles the structured text format produced by `GdsWriter`,
/// not binary GDSII stream format (binary parsing is handled by
/// `crate::io::gds_io::GdsBinaryReader`).
pub struct GdsReader;

impl GdsReader {
    /// Parse a GDS text representation into a [`GdsLibrary`].
    pub fn parse(text: &str) -> Result<GdsLibrary, GdsParseError> {
        let mut lines = text.lines().peekable();

        // Parse LIBRARY header
        let header = lines.next().ok_or_else(|| GdsParseError::UnexpectedEof {
            context: "LIBRARY header".into(),
        })?;
        let (name, db_unit_m, user_unit_m) = parse_library_header(header)?;

        let mut lib = GdsLibrary::new(name);
        lib.db_unit_m = db_unit_m;
        lib.user_unit_m = user_unit_m;

        // Parse CELLs until ENDLIB
        while let Some(raw) = lines.next() {
            let line = raw.trim().to_string();
            if line == "ENDLIB" {
                break;
            }
            if let Some(cell_name) = line.strip_prefix("CELL ") {
                let cell = parse_cell(cell_name.trim(), &mut lines)?;
                lib.cells.push(cell);
            } else if !line.is_empty() {
                return Err(GdsParseError::UnexpectedLine {
                    line,
                    context: "top level".into(),
                });
            }
        }

        Ok(lib)
    }
}

// ─── Private parsing helpers ──────────────────────────────────────────────────

fn parse_library_header(line: &str) -> Result<(String, f64, f64), GdsParseError> {
    // Format: "LIBRARY <name> db_unit=<val>m user_unit=<val>m"
    let line = line.trim();
    let rest = line
        .strip_prefix("LIBRARY ")
        .ok_or_else(|| GdsParseError::MalformedField {
            field: "LIBRARY".into(),
            reason: format!("expected LIBRARY prefix, got: {line}"),
        })?;
    let parts: Vec<&str> = rest.splitn(3, ' ').collect();
    if parts.len() < 3 {
        return Err(GdsParseError::MalformedField {
            field: "LIBRARY header".into(),
            reason: format!("expected name + db_unit + user_unit, got: {rest}"),
        });
    }
    let name = parts[0].to_string();
    let db_unit_m = parse_unit_field(parts[1], "db_unit")?;
    let user_unit_m = parse_unit_field(parts[2], "user_unit")?;
    Ok((name, db_unit_m, user_unit_m))
}

fn parse_unit_field(s: &str, key: &str) -> Result<f64, GdsParseError> {
    // Format: "db_unit=1.00e-9m" or "user_unit=1.00e-6m"
    let prefix = format!("{key}=");
    let val_str = s
        .strip_prefix(&prefix)
        .ok_or_else(|| GdsParseError::MalformedField {
            field: key.into(),
            reason: format!("expected {key}=..., got {s}"),
        })?;
    // Strip trailing 'm'
    let val_str = val_str.strip_suffix('m').unwrap_or(val_str);
    val_str
        .parse::<f64>()
        .map_err(|e| GdsParseError::MalformedField {
            field: key.into(),
            reason: format!("could not parse float {val_str}: {e}"),
        })
}

fn parse_cell<'a>(
    name: &str,
    lines: &mut impl Iterator<Item = &'a str>,
) -> Result<GdsCell, GdsParseError> {
    let mut cell = GdsCell::new(name);
    for raw_line in lines.by_ref() {
        let line = raw_line.trim();
        if line == "ENDCELL" {
            return Ok(cell);
        }
        if let Some(rest) = line.strip_prefix("BOUNDARY ") {
            cell.elements
                .push(GdsElement::Boundary(parse_boundary(rest)?));
        } else if let Some(rest) = line.strip_prefix("PATH ") {
            cell.elements.push(GdsElement::Path(parse_path(rest)?));
        } else if let Some(rest) = line.strip_prefix("SREF ") {
            cell.elements.push(GdsElement::Sref(parse_sref(rest)?));
        } else if let Some(rest) = line.strip_prefix("AREF ") {
            cell.elements.push(GdsElement::Aref(parse_aref(rest)?));
        } else if let Some(rest) = line.strip_prefix("TEXT ") {
            cell.elements.push(GdsElement::Text(parse_text(rest)?));
        } else if !line.is_empty() {
            return Err(GdsParseError::UnexpectedLine {
                line: line.into(),
                context: format!("CELL {name}"),
            });
        }
    }
    Err(GdsParseError::UnexpectedEof {
        context: format!("CELL {name} — missing ENDCELL"),
    })
}

/// Parse `layer=L/D` prefix from a token string, returning `(layer, datatype, remainder)`.
fn parse_layer_prefix(s: &str) -> Result<(u16, u16, &str), GdsParseError> {
    // s starts with "layer=L/D ..."
    let rest = s
        .strip_prefix("layer=")
        .ok_or_else(|| GdsParseError::MalformedField {
            field: "layer".into(),
            reason: format!("expected 'layer=', got: {s}"),
        })?;
    // Find the space that ends the layer spec
    let (layer_part, remainder) = if let Some(idx) = rest.find(' ') {
        (&rest[..idx], rest[idx + 1..].trim_start())
    } else {
        (rest, "")
    };
    let mut ld = layer_part.splitn(2, '/');
    let l_str = ld.next().ok_or_else(|| GdsParseError::MalformedField {
        field: "layer".into(),
        reason: format!("missing layer number in {layer_part}"),
    })?;
    let d_str = ld.next().ok_or_else(|| GdsParseError::MalformedField {
        field: "datatype".into(),
        reason: format!("missing datatype in {layer_part}"),
    })?;
    let layer = l_str
        .parse::<u16>()
        .map_err(|e| GdsParseError::MalformedField {
            field: "layer".into(),
            reason: format!("{e}"),
        })?;
    let datatype = d_str
        .parse::<u16>()
        .map_err(|e| GdsParseError::MalformedField {
            field: "datatype".into(),
            reason: format!("{e}"),
        })?;
    Ok((layer, datatype, remainder))
}

fn parse_points(s: &str) -> Result<Vec<GdsPoint>, GdsParseError> {
    // s looks like: "(0,0) (100,0) (100,50) (0,50)"
    let mut points = Vec::new();
    for token in s.split_whitespace() {
        let inner = token.trim_matches(|c| c == '(' || c == ')');
        let mut parts = inner.splitn(2, ',');
        let x_str = parts.next().ok_or_else(|| GdsParseError::MalformedField {
            field: "point x".into(),
            reason: format!("missing x in {token}"),
        })?;
        let y_str = parts.next().ok_or_else(|| GdsParseError::MalformedField {
            field: "point y".into(),
            reason: format!("missing y in {token}"),
        })?;
        let x = x_str
            .parse::<i32>()
            .map_err(|e| GdsParseError::MalformedField {
                field: "point x".into(),
                reason: format!("{e}"),
            })?;
        let y = y_str
            .parse::<i32>()
            .map_err(|e| GdsParseError::MalformedField {
                field: "point y".into(),
                reason: format!("{e}"),
            })?;
        points.push(GdsPoint::new(x, y));
    }
    Ok(points)
}

/// Parse a single `(x,y)` point.
fn parse_point_token(token: &str) -> Result<GdsPoint, GdsParseError> {
    let inner = token.trim_matches(|c| c == '(' || c == ')');
    let mut parts = inner.splitn(2, ',');
    let x_str = parts.next().ok_or_else(|| GdsParseError::MalformedField {
        field: "point x".into(),
        reason: format!("missing x in {token}"),
    })?;
    let y_str = parts.next().ok_or_else(|| GdsParseError::MalformedField {
        field: "point y".into(),
        reason: format!("missing y in {token}"),
    })?;
    let x = x_str
        .parse::<i32>()
        .map_err(|e| GdsParseError::MalformedField {
            field: "point x".into(),
            reason: format!("{e}"),
        })?;
    let y = y_str
        .parse::<i32>()
        .map_err(|e| GdsParseError::MalformedField {
            field: "point y".into(),
            reason: format!("{e}"),
        })?;
    Ok(GdsPoint::new(x, y))
}

/// Extract `key=(x,y)` from a space-separated token stream.
fn extract_point_field<'a>(
    tokens: &mut impl Iterator<Item = &'a str>,
    key: &str,
) -> Result<GdsPoint, GdsParseError> {
    let prefix = format!("{key}=");
    let tok = tokens.next().ok_or_else(|| GdsParseError::UnexpectedEof {
        context: format!("field {key}"),
    })?;
    let val = tok
        .strip_prefix(&prefix)
        .ok_or_else(|| GdsParseError::MalformedField {
            field: key.into(),
            reason: format!("expected '{key}=(x,y)', got '{tok}'"),
        })?;
    parse_point_token(val)
}

/// Extract `key=<value>` from a space-separated token stream.
fn extract_kv<'a>(
    tokens: &mut impl Iterator<Item = &'a str>,
    key: &str,
) -> Result<&'a str, GdsParseError> {
    let prefix = format!("{key}=");
    let tok = tokens.next().ok_or_else(|| GdsParseError::UnexpectedEof {
        context: format!("field {key}"),
    })?;
    tok.strip_prefix(&prefix)
        .ok_or_else(|| GdsParseError::MalformedField {
            field: key.into(),
            reason: format!("expected '{key}=...', got '{tok}'"),
        })
}

fn parse_boundary(rest: &str) -> Result<GdsBoundary, GdsParseError> {
    // Format: "layer=L/D points=(x0,y0) (x1,y1) ..."
    let (layer_n, datatype, pts_str) = parse_layer_prefix(rest)?;
    // pts_str starts with "points=(x0,y0) ..."
    let pts_data =
        pts_str
            .strip_prefix("points=")
            .ok_or_else(|| GdsParseError::MalformedField {
                field: "points".into(),
                reason: format!("expected 'points=', got: {pts_str}"),
            })?;
    let points = parse_points(pts_data)?;
    Ok(GdsBoundary {
        layer: GdsLayer::new(layer_n, datatype),
        points,
    })
}

fn parse_path(rest: &str) -> Result<GdsPath, GdsParseError> {
    // Format: "layer=L/D width=W points=(x0,y0) (x1,y1) ..."
    let (layer_n, datatype, after_layer) = parse_layer_prefix(rest)?;
    // after_layer: "width=W points=..."
    let width_end = after_layer
        .find(' ')
        .ok_or_else(|| GdsParseError::MalformedField {
            field: "width".into(),
            reason: format!("missing space after width in: {after_layer}"),
        })?;
    let width_token = &after_layer[..width_end];
    let pts_str = after_layer[width_end + 1..].trim_start();
    let width_str =
        width_token
            .strip_prefix("width=")
            .ok_or_else(|| GdsParseError::MalformedField {
                field: "width".into(),
                reason: format!("expected 'width=', got '{width_token}'"),
            })?;
    let width = width_str
        .parse::<i32>()
        .map_err(|e| GdsParseError::MalformedField {
            field: "width".into(),
            reason: format!("{e}"),
        })?;
    let pts_data =
        pts_str
            .strip_prefix("points=")
            .ok_or_else(|| GdsParseError::MalformedField {
                field: "points".into(),
                reason: format!("expected 'points=', got: {pts_str}"),
            })?;
    let points = parse_points(pts_data)?;
    Ok(GdsPath {
        layer: GdsLayer::new(layer_n, datatype),
        width,
        points,
    })
}

fn parse_sref(rest: &str) -> Result<GdsSref, GdsParseError> {
    // Format: "<sname> at (ox,oy) angle=A mag=M xrefl=R"
    let mut tokens = rest.split_whitespace();
    let sname = tokens
        .next()
        .ok_or_else(|| GdsParseError::UnexpectedEof {
            context: "SREF name".into(),
        })?
        .to_string();
    // "at"
    let _at = tokens.next().ok_or_else(|| GdsParseError::UnexpectedEof {
        context: "SREF 'at' keyword".into(),
    })?;
    let origin_tok = tokens.next().ok_or_else(|| GdsParseError::UnexpectedEof {
        context: "SREF origin".into(),
    })?;
    let origin = parse_point_token(origin_tok)?;

    let angle_str = extract_kv(&mut tokens, "angle")?;
    let angle_deg = angle_str
        .parse::<f64>()
        .map_err(|e| GdsParseError::MalformedField {
            field: "angle".into(),
            reason: format!("{e}"),
        })?;

    let mag_str = extract_kv(&mut tokens, "mag")?;
    let magnification = mag_str
        .parse::<f64>()
        .map_err(|e| GdsParseError::MalformedField {
            field: "mag".into(),
            reason: format!("{e}"),
        })?;

    let xrefl_str = extract_kv(&mut tokens, "xrefl")?;
    let xrefl_n = xrefl_str
        .parse::<u8>()
        .map_err(|e| GdsParseError::MalformedField {
            field: "xrefl".into(),
            reason: format!("{e}"),
        })?;

    Ok(GdsSref {
        sname,
        origin,
        angle_deg,
        magnification,
        x_reflection: xrefl_n != 0,
    })
}

fn parse_aref(rest: &str) -> Result<GdsAref, GdsParseError> {
    // Format: "<ref_name> cols=C rows=R at (ox,oy) col_pt=(cx,cy) row_pt=(rx,ry) angle=A mag=M xrefl=R"
    let mut tokens = rest.split_whitespace();
    let ref_name = tokens
        .next()
        .ok_or_else(|| GdsParseError::UnexpectedEof {
            context: "AREF name".into(),
        })?
        .to_string();

    let cols_str = extract_kv(&mut tokens, "cols")?;
    let cols = cols_str
        .parse::<u16>()
        .map_err(|e| GdsParseError::MalformedField {
            field: "cols".into(),
            reason: format!("{e}"),
        })?;

    let rows_str = extract_kv(&mut tokens, "rows")?;
    let rows = rows_str
        .parse::<u16>()
        .map_err(|e| GdsParseError::MalformedField {
            field: "rows".into(),
            reason: format!("{e}"),
        })?;

    // "at"
    let _at = tokens.next().ok_or_else(|| GdsParseError::UnexpectedEof {
        context: "AREF 'at' keyword".into(),
    })?;
    let origin_tok = tokens.next().ok_or_else(|| GdsParseError::UnexpectedEof {
        context: "AREF origin".into(),
    })?;
    let origin = parse_point_token(origin_tok)?;

    let col_pt = extract_point_field(&mut tokens, "col_pt")?;
    let row_pt = extract_point_field(&mut tokens, "row_pt")?;

    let angle_str = extract_kv(&mut tokens, "angle")?;
    let angle_deg = angle_str
        .parse::<f64>()
        .map_err(|e| GdsParseError::MalformedField {
            field: "angle".into(),
            reason: format!("{e}"),
        })?;

    let mag_str = extract_kv(&mut tokens, "mag")?;
    let magnification = mag_str
        .parse::<f64>()
        .map_err(|e| GdsParseError::MalformedField {
            field: "mag".into(),
            reason: format!("{e}"),
        })?;

    let xrefl_str = extract_kv(&mut tokens, "xrefl")?;
    let xrefl_n = xrefl_str
        .parse::<u8>()
        .map_err(|e| GdsParseError::MalformedField {
            field: "xrefl".into(),
            reason: format!("{e}"),
        })?;

    Ok(GdsAref {
        ref_name,
        cols,
        rows,
        angle_deg,
        magnification,
        x_reflection: xrefl_n != 0,
        xy: vec![origin, col_pt, row_pt],
    })
}

fn parse_text(rest: &str) -> Result<GdsText, GdsParseError> {
    // Format: "\"<string>\" layer=L/D at (ox,oy) height=H"
    // The string is quoted and may contain spaces; everything between first " and
    // matching closing " is the text string.
    let rest = rest.trim();
    if !rest.starts_with('"') {
        return Err(GdsParseError::MalformedField {
            field: "TEXT string".into(),
            reason: format!("expected opening quote, got: {rest}"),
        });
    }
    let after_open = &rest[1..]; // skip leading "
    let close_idx = after_open
        .find('"')
        .ok_or_else(|| GdsParseError::MalformedField {
            field: "TEXT string".into(),
            reason: "missing closing quote".into(),
        })?;
    let string = after_open[..close_idx].to_string();
    // Remainder after the closing quote
    let tail = after_open[close_idx + 1..].trim_start();
    // tail: "layer=L/D at (ox,oy) height=H"
    let (layer_n, datatype, after_layer) = parse_layer_prefix(tail)?;
    // after_layer: "at (ox,oy) height=H"
    let mut tokens = after_layer.split_whitespace();
    let _at = tokens.next().ok_or_else(|| GdsParseError::UnexpectedEof {
        context: "TEXT 'at' keyword".into(),
    })?;
    let origin_tok = tokens.next().ok_or_else(|| GdsParseError::UnexpectedEof {
        context: "TEXT origin".into(),
    })?;
    let origin = parse_point_token(origin_tok)?;

    let height_str = extract_kv(&mut tokens, "height")?;
    let height = height_str
        .parse::<i32>()
        .map_err(|e| GdsParseError::MalformedField {
            field: "height".into(),
            reason: format!("{e}"),
        })?;

    Ok(GdsText {
        layer: GdsLayer::new(layer_n, datatype),
        string,
        origin,
        height,
    })
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

    #[test]
    fn gds_writer_emits_full_boundary_points() {
        let mut lib = GdsLibrary::new("ptlib");
        let mut cell = GdsCell::new("C");
        cell.elements.push(GdsElement::Boundary(GdsBoundary {
            layer: GdsLayer::new(3, 1),
            points: vec![GdsPoint::new(10, 20), GdsPoint::new(30, 40)],
        }));
        lib.add_cell(cell);
        let mut w = GdsWriter::new();
        let txt = w.write_library(&lib).to_string();
        assert!(txt.contains("points=(10,20) (30,40)"), "got: {txt}");
        assert!(txt.contains("layer=3/1"), "got: {txt}");
    }

    #[test]
    fn gds_writer_emits_full_path_points() {
        let mut lib = GdsLibrary::new("pathlib");
        let mut cell = GdsCell::new("C");
        cell.elements.push(GdsElement::Path(GdsPath {
            layer: GdsLayer::new(2, 0),
            width: 15,
            points: vec![GdsPoint::new(0, 25), GdsPoint::new(100, 25)],
        }));
        lib.add_cell(cell);
        let mut w = GdsWriter::new();
        let txt = w.write_library(&lib).to_string();
        assert!(txt.contains("width=15"), "got: {txt}");
        assert!(txt.contains("points=(0,25) (100,25)"), "got: {txt}");
    }

    #[test]
    fn gds_writer_reader_round_trip() {
        let mut lib = GdsLibrary::new("testlib");
        lib.db_unit_m = 1e-9;
        lib.user_unit_m = 1e-6;

        let mut cell = GdsCell::new("cell1");

        // Boundary
        cell.elements.push(GdsElement::Boundary(GdsBoundary {
            layer: GdsLayer::new(1, 0),
            points: vec![
                GdsPoint::new(0, 0),
                GdsPoint::new(100, 0),
                GdsPoint::new(100, 50),
                GdsPoint::new(0, 50),
            ],
        }));

        // Path
        cell.elements.push(GdsElement::Path(GdsPath {
            layer: GdsLayer::new(2, 0),
            width: 10,
            points: vec![GdsPoint::new(0, 25), GdsPoint::new(100, 25)],
        }));

        // Sref
        cell.elements.push(GdsElement::Sref(GdsSref {
            sname: "subcell".into(),
            origin: GdsPoint::new(50, 50),
            angle_deg: 0.0,
            magnification: 1.0,
            x_reflection: false,
        }));

        // Aref
        cell.elements.push(GdsElement::Aref(GdsAref {
            ref_name: "subcell".into(),
            cols: 3,
            rows: 2,
            angle_deg: 0.0,
            magnification: 1.0,
            x_reflection: false,
            xy: vec![
                GdsPoint::new(0, 0),
                GdsPoint::new(300, 0),
                GdsPoint::new(0, 200),
            ],
        }));

        // Text
        cell.elements.push(GdsElement::Text(GdsText {
            layer: GdsLayer::new(5, 0),
            string: "TE".into(),
            origin: GdsPoint::new(10, 10),
            height: 50,
        }));

        lib.cells.push(cell);

        // Write → Read → compare
        let mut writer = GdsWriter::new();
        let text = writer.write_library(&lib).to_string();
        let lib2 = GdsReader::parse(&text).expect("parse should succeed");

        assert_eq!(lib2.name, lib.name);
        assert!((lib2.db_unit_m - lib.db_unit_m).abs() < 1e-20);
        assert_eq!(lib2.cells.len(), lib.cells.len());
        let c2 = &lib2.cells[0];
        assert_eq!(c2.elements.len(), 5);

        // Check boundary points round-tripped
        match &c2.elements[0] {
            GdsElement::Boundary(b) => {
                assert_eq!(b.points.len(), 4);
                assert_eq!(b.points[0], GdsPoint::new(0, 0));
                assert_eq!(b.points[2], GdsPoint::new(100, 50));
            }
            _ => panic!("expected Boundary"),
        }

        // Check path round-tripped
        match &c2.elements[1] {
            GdsElement::Path(p) => {
                assert_eq!(p.width, 10);
                assert_eq!(p.points.len(), 2);
                assert_eq!(p.points[1], GdsPoint::new(100, 25));
            }
            _ => panic!("expected Path"),
        }

        // Check sref round-tripped
        match &c2.elements[2] {
            GdsElement::Sref(s) => {
                assert_eq!(s.sname, "subcell");
                assert_eq!(s.origin, GdsPoint::new(50, 50));
                assert!((s.magnification - 1.0).abs() < 1e-10);
            }
            _ => panic!("expected Sref"),
        }

        // Check aref round-tripped
        match &c2.elements[3] {
            GdsElement::Aref(a) => {
                assert_eq!(a.ref_name, "subcell");
                assert_eq!(a.cols, 3);
                assert_eq!(a.rows, 2);
                assert_eq!(a.xy.len(), 3);
                assert_eq!(a.xy[0], GdsPoint::new(0, 0));
                assert_eq!(a.xy[1], GdsPoint::new(300, 0));
                assert_eq!(a.xy[2], GdsPoint::new(0, 200));
            }
            _ => panic!("expected Aref"),
        }

        // Check text round-tripped
        match &c2.elements[4] {
            GdsElement::Text(t) => {
                assert_eq!(t.string, "TE");
                assert_eq!(t.origin, GdsPoint::new(10, 10));
                assert_eq!(t.height, 50);
                assert_eq!(t.layer.layer, 5);
            }
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn gds_reader_parse_error_on_bad_input() {
        let result = GdsReader::parse("NOT_A_LIBRARY");
        assert!(result.is_err(), "expected parse error for bad input");
    }

    #[test]
    fn gds_reader_empty_input_error() {
        let result = GdsReader::parse("");
        assert!(result.is_err(), "expected parse error for empty input");
    }

    #[test]
    fn gds_writer_reader_sref_transform_round_trip() {
        let mut lib = GdsLibrary::new("xflib");
        lib.db_unit_m = 1e-9;
        lib.user_unit_m = 1e-6;
        let mut cell = GdsCell::new("TOP");
        cell.elements.push(GdsElement::Sref(GdsSref {
            sname: "SUB".into(),
            origin: GdsPoint::new(100, 200),
            angle_deg: 90.0,
            magnification: 2.0,
            x_reflection: true,
        }));
        lib.add_cell(cell);

        let mut writer = GdsWriter::new();
        let text = writer.write_library(&lib).to_string();
        let lib2 = GdsReader::parse(&text).expect("parse should succeed");

        match &lib2.cells[0].elements[0] {
            GdsElement::Sref(s) => {
                assert!((s.angle_deg - 90.0).abs() < 1e-10);
                assert!((s.magnification - 2.0).abs() < 1e-10);
                assert!(s.x_reflection);
            }
            _ => panic!("expected Sref"),
        }
    }
}
