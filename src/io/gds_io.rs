//! GDS I/O utilities: text-based and binary GDSII stream format.
//!
//! This module provides:
//! - `GdsTextExporter`: human-readable text export (original functionality)
//! - `GdsBinaryWriter`: spec-compliant binary GDSII stream writer (feature = "io-gds")
//! - `GdsBinaryReader`: spec-compliant binary GDSII stream reader (feature = "io-gds")

use crate::error::OxiPhotonError;
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

// ─── Binary GDSII implementation ─────────────────────────────────────────────

/// GDSII record type constants (first byte of the 2-byte record token).
mod rec {
    pub const HEADER: u8 = 0x00;
    pub const BGNLIB: u8 = 0x01;
    pub const LIBNAME: u8 = 0x02;
    pub const UNITS: u8 = 0x03;
    pub const ENDLIB: u8 = 0x04;
    pub const BGNSTR: u8 = 0x05;
    pub const STRNAME: u8 = 0x06;
    pub const ENDSTR: u8 = 0x07;
    pub const BOUNDARY: u8 = 0x08;
    pub const PATH: u8 = 0x09;
    pub const SREF: u8 = 0x0A;
    pub const AREF: u8 = 0x0B;
    pub const TEXT: u8 = 0x0C;
    pub const LAYER: u8 = 0x0D;
    pub const DATATYPE: u8 = 0x0E;
    pub const WIDTH: u8 = 0x0F;
    pub const XY: u8 = 0x10;
    pub const ENDEL: u8 = 0x11;
    pub const SNAME: u8 = 0x12;
    pub const COLROW: u8 = 0x13;
    pub const TEXTTYPE: u8 = 0x16;
    pub const STRING: u8 = 0x19;
    pub const STRANS: u8 = 0x1A;
    pub const MAG: u8 = 0x1B;
    pub const ANGLE: u8 = 0x1C;
}

/// GDSII data type constants (second byte of the 2-byte record token).
mod dt {
    pub const NO_DATA: u8 = 0x00;
    pub const BITARRAY: u8 = 0x01;
    pub const INT16: u8 = 0x02;
    pub const INT32: u8 = 0x03;
    pub const REAL8: u8 = 0x05;
    pub const ASCII: u8 = 0x06;
}

/// STRANS flags (u16, big-endian).
const STRANS_X_REFLECTION: u16 = 0x8000; // bit 15

// ─── IBM real8 (Calma HEX float) ─────────────────────────────────────────────

/// Encode an `f64` to IBM hexadecimal floating-point (8 bytes, big-endian).
///
/// Format: [sign(1)|exponent(7)][mantissa(56 bits = 7 bytes)]
/// The exponent is excess-64 in base-16; the mantissa is a base-16 fraction
/// in the range [1/16, 1).
pub fn real8_encode(value: f64) -> [u8; 8] {
    if value == 0.0 {
        return [0u8; 8];
    }

    let sign: u8 = if value < 0.0 { 1 } else { 0 };
    let mut x = value.abs();

    // Normalize: find exponent E such that 1/16 <= x < 1
    // stored_exp = E (where actual base-16 exponent = E - 64)
    let mut stored_exp: i32 = 64;
    while x >= 1.0 {
        x /= 16.0;
        stored_exp += 1;
    }
    while x < 1.0 / 16.0 {
        x *= 16.0;
        stored_exp -= 1;
    }

    // Mantissa as 56-bit integer: M = round(x * 2^56)
    // M fits in 56 bits; store as 7 big-endian bytes (the low 7 bytes of a u64 BE).
    let mantissa = (x * (1u64 << 56) as f64).round() as u64;

    // Pack byte[0] = (sign << 7) | stored_exp
    let byte0 = (sign << 7) | (stored_exp as u8 & 0x7F);

    // M fits in bits 55..0.  to_be_bytes() puts the MSB at index 0:
    //   [0x00, high7..low0]
    // We want the 7-byte big-endian representation, so skip the leading 0x00.
    let m_bytes = mantissa.to_be_bytes(); // [byte0(MSB)..byte7(LSB)]
    [
        byte0, m_bytes[1], m_bytes[2], m_bytes[3], m_bytes[4], m_bytes[5], m_bytes[6], m_bytes[7],
    ]
}

/// Decode an IBM real8 (8 bytes, big-endian) to `f64`.
pub fn real8_decode(bytes: &[u8; 8]) -> f64 {
    let sign = (bytes[0] >> 7) & 1;
    let stored_exp = (bytes[0] & 0x7F) as i32;

    // Reconstruct 56-bit mantissa: 7-byte big-endian value, prepend 0x00 to make u64.
    let mantissa = u64::from_be_bytes([
        0, bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);

    if stored_exp == 0 && mantissa == 0 {
        return 0.0;
    }

    let m_f64 = mantissa as f64 / (1u64 << 56) as f64;
    let exp = stored_exp - 64;
    let magnitude = m_f64 * 16.0_f64.powi(exp);

    if sign == 1 {
        -magnitude
    } else {
        magnitude
    }
}

// ─── Low-level record builders ────────────────────────────────────────────────

/// Append a GDSII record to `buf`.
///
/// `record_type` and `data_type` are single bytes. `payload` is the data bytes
/// (may be empty for no-data records). The 2-byte length field is
/// `4 + payload.len()`.
fn push_record(buf: &mut Vec<u8>, record_type: u8, data_type: u8, payload: &[u8]) {
    let length = 4u16 + payload.len() as u16;
    buf.extend_from_slice(&length.to_be_bytes());
    buf.push(record_type);
    buf.push(data_type);
    buf.extend_from_slice(payload);
}

/// Record with no payload.
fn push_no_data(buf: &mut Vec<u8>, record_type: u8) {
    push_record(buf, record_type, dt::NO_DATA, &[]);
}

/// Record with a single i16 value.
fn push_i16(buf: &mut Vec<u8>, record_type: u8, value: i16) {
    push_record(buf, record_type, dt::INT16, &value.to_be_bytes());
}

/// Record with two i16 values.
fn push_i16x2(buf: &mut Vec<u8>, record_type: u8, a: i16, b: i16) {
    let mut payload = [0u8; 4];
    payload[0..2].copy_from_slice(&a.to_be_bytes());
    payload[2..4].copy_from_slice(&b.to_be_bytes());
    push_record(buf, record_type, dt::INT16, &payload);
}

/// Record with 12 zero i16 values (for BGNLIB, BGNSTR timestamps).
fn push_i16x12_zeros(buf: &mut Vec<u8>, record_type: u8) {
    push_record(buf, record_type, dt::INT16, &[0u8; 24]);
}

/// Record with a single i32 value.
fn push_i32(buf: &mut Vec<u8>, record_type: u8, value: i32) {
    push_record(buf, record_type, dt::INT32, &value.to_be_bytes());
}

/// Record with XY coordinate pairs (i32 big-endian).
fn push_xy(buf: &mut Vec<u8>, points: &[crate::geometry::gds::GdsPoint]) {
    let mut payload = Vec::with_capacity(points.len() * 8);
    for p in points {
        payload.extend_from_slice(&p.x.to_be_bytes());
        payload.extend_from_slice(&p.y.to_be_bytes());
    }
    push_record(buf, rec::XY, dt::INT32, &payload);
}

/// Record with a single real8 value.
fn push_real8(buf: &mut Vec<u8>, record_type: u8, value: f64) {
    push_record(buf, record_type, dt::REAL8, &real8_encode(value));
}

/// Record with two real8 values.
fn push_real8x2(buf: &mut Vec<u8>, record_type: u8, a: f64, b: f64) {
    let mut payload = [0u8; 16];
    payload[0..8].copy_from_slice(&real8_encode(a));
    payload[8..16].copy_from_slice(&real8_encode(b));
    push_record(buf, record_type, dt::REAL8, &payload);
}

/// Record with an ASCII string (even-padded with null if odd length).
fn push_string(buf: &mut Vec<u8>, record_type: u8, s: &str) {
    let bytes = s.as_bytes();
    if bytes.len() % 2 == 0 {
        push_record(buf, record_type, dt::ASCII, bytes);
    } else {
        let mut padded = bytes.to_vec();
        padded.push(0u8);
        push_record(buf, record_type, dt::ASCII, &padded);
    }
}

/// Record with a u16 bitarray (STRANS flags).
fn push_bitarray(buf: &mut Vec<u8>, record_type: u8, flags: u16) {
    push_record(buf, record_type, dt::BITARRAY, &flags.to_be_bytes());
}

// ─── Element writers ──────────────────────────────────────────────────────────

fn write_boundary(buf: &mut Vec<u8>, b: &crate::geometry::gds::GdsBoundary) {
    push_no_data(buf, rec::BOUNDARY);
    push_i16(buf, rec::LAYER, b.layer.layer as i16);
    push_i16(buf, rec::DATATYPE, b.layer.datatype as i16);
    push_xy(buf, &b.points);
    push_no_data(buf, rec::ENDEL);
}

fn write_path(buf: &mut Vec<u8>, p: &crate::geometry::gds::GdsPath) {
    push_no_data(buf, rec::PATH);
    push_i16(buf, rec::LAYER, p.layer.layer as i16);
    push_i16(buf, rec::DATATYPE, p.layer.datatype as i16);
    push_i32(buf, rec::WIDTH, p.width);
    push_xy(buf, &p.points);
    push_no_data(buf, rec::ENDEL);
}

fn write_strans(buf: &mut Vec<u8>, x_reflection: bool, magnification: f64, angle_deg: f64) {
    let mut flags: u16 = 0;
    if x_reflection {
        flags |= STRANS_X_REFLECTION;
    }
    push_bitarray(buf, rec::STRANS, flags);
    if (magnification - 1.0).abs() > f64::EPSILON {
        push_real8(buf, rec::MAG, magnification);
    }
    if angle_deg.abs() > f64::EPSILON {
        push_real8(buf, rec::ANGLE, angle_deg);
    }
}

fn write_sref(buf: &mut Vec<u8>, s: &crate::geometry::gds::GdsSref) {
    push_no_data(buf, rec::SREF);
    push_string(buf, rec::SNAME, &s.sname);
    write_strans(buf, s.x_reflection, s.magnification, s.angle_deg);
    push_xy(buf, std::slice::from_ref(&s.origin));
    push_no_data(buf, rec::ENDEL);
}

fn write_aref(buf: &mut Vec<u8>, a: &crate::geometry::gds::GdsAref) {
    push_no_data(buf, rec::AREF);
    push_string(buf, rec::SNAME, &a.ref_name);
    write_strans(buf, a.x_reflection, a.magnification, a.angle_deg);
    push_i16x2(buf, rec::COLROW, a.cols as i16, a.rows as i16);
    push_xy(buf, &a.xy);
    push_no_data(buf, rec::ENDEL);
}

fn write_text(buf: &mut Vec<u8>, t: &crate::geometry::gds::GdsText) {
    push_no_data(buf, rec::TEXT);
    push_i16(buf, rec::LAYER, t.layer.layer as i16);
    push_i16(buf, rec::TEXTTYPE, 0i16);
    push_xy(buf, std::slice::from_ref(&t.origin));
    push_string(buf, rec::STRING, &t.string);
    push_no_data(buf, rec::ENDEL);
}

// ─── GdsBinaryWriter ─────────────────────────────────────────────────────────

/// Writes a [`GdsLibrary`] to binary GDSII stream format.
pub struct GdsBinaryWriter;

impl GdsBinaryWriter {
    /// Serialize `lib` to a `Vec<u8>` containing a valid binary GDSII stream.
    pub fn to_bytes(lib: &GdsLibrary) -> Vec<u8> {
        let mut buf = Vec::new();

        // HEADER record: version 600 (0x0258)
        push_i16(&mut buf, rec::HEADER, 600i16);

        // BGNLIB (12 × i16 zeros — modification/access time)
        push_i16x12_zeros(&mut buf, rec::BGNLIB);

        // LIBNAME
        push_string(&mut buf, rec::LIBNAME, &lib.name);

        // UNITS: [db_unit_m / user_unit_m, db_unit_m]
        let units0 = lib.db_unit_m / lib.user_unit_m;
        push_real8x2(&mut buf, rec::UNITS, units0, lib.db_unit_m);

        // Cells
        for cell in &lib.cells {
            // BGNSTR (12 × i16 zeros)
            push_i16x12_zeros(&mut buf, rec::BGNSTR);
            // STRNAME
            push_string(&mut buf, rec::STRNAME, &cell.name);

            for elem in &cell.elements {
                use crate::geometry::gds::GdsElement;
                match elem {
                    GdsElement::Boundary(b) => write_boundary(&mut buf, b),
                    GdsElement::Path(p) => write_path(&mut buf, p),
                    GdsElement::Sref(s) => write_sref(&mut buf, s),
                    GdsElement::Aref(a) => write_aref(&mut buf, a),
                    GdsElement::Text(t) => write_text(&mut buf, t),
                }
            }

            // ENDSTR
            push_no_data(&mut buf, rec::ENDSTR);
        }

        // ENDLIB
        push_no_data(&mut buf, rec::ENDLIB);

        buf
    }

    /// Write `lib` as binary GDSII to `path`.
    pub fn to_path(
        lib: &GdsLibrary,
        path: &std::path::Path,
    ) -> std::result::Result<(), OxiPhotonError> {
        use std::io::Write;
        let bytes = Self::to_bytes(lib);
        let mut file = std::fs::File::create(path)?;
        file.write_all(&bytes)?;
        Ok(())
    }
}

// ─── GdsBinaryReader ─────────────────────────────────────────────────────────

/// A record parsed from the binary GDSII stream.
struct GdsRecord {
    record_type: u8,
    _data_type: u8,
    payload: Vec<u8>,
}

impl GdsRecord {
    /// Read a single record from a byte cursor, advancing the cursor.
    fn read_from(cursor: &mut &[u8]) -> std::result::Result<Self, OxiPhotonError> {
        if cursor.len() < 4 {
            return Err(OxiPhotonError::Gds(
                "truncated GDSII stream: not enough bytes for record header".into(),
            ));
        }
        let length = u16::from_be_bytes([cursor[0], cursor[1]]) as usize;
        if length < 4 {
            return Err(OxiPhotonError::Gds(format!(
                "invalid record length {length} (minimum 4)"
            )));
        }
        if cursor.len() < length {
            return Err(OxiPhotonError::Gds(format!(
                "truncated GDSII stream: record header says {length} bytes but only {} remain",
                cursor.len()
            )));
        }
        let record_type = cursor[2];
        let _data_type = cursor[3];
        let payload = cursor[4..length].to_vec();
        *cursor = &cursor[length..];
        Ok(GdsRecord {
            record_type,
            _data_type,
            payload,
        })
    }

    /// Parse payload as a single i16.
    fn as_i16(&self) -> std::result::Result<i16, OxiPhotonError> {
        if self.payload.len() < 2 {
            return Err(OxiPhotonError::Gds(format!(
                "record 0x{:02X}: expected i16 payload (>=2 bytes), got {}",
                self.record_type,
                self.payload.len()
            )));
        }
        Ok(i16::from_be_bytes([self.payload[0], self.payload[1]]))
    }

    /// Parse payload as two i16 values.
    fn as_i16x2(&self) -> std::result::Result<(i16, i16), OxiPhotonError> {
        if self.payload.len() < 4 {
            return Err(OxiPhotonError::Gds(format!(
                "record 0x{:02X}: expected 2×i16 payload (>=4 bytes), got {}",
                self.record_type,
                self.payload.len()
            )));
        }
        let a = i16::from_be_bytes([self.payload[0], self.payload[1]]);
        let b = i16::from_be_bytes([self.payload[2], self.payload[3]]);
        Ok((a, b))
    }

    /// Parse payload as a single i32.
    fn as_i32(&self) -> std::result::Result<i32, OxiPhotonError> {
        if self.payload.len() < 4 {
            return Err(OxiPhotonError::Gds(format!(
                "record 0x{:02X}: expected i32 payload (>=4 bytes), got {}",
                self.record_type,
                self.payload.len()
            )));
        }
        Ok(i32::from_be_bytes([
            self.payload[0],
            self.payload[1],
            self.payload[2],
            self.payload[3],
        ]))
    }

    /// Parse payload as an ASCII string (strips trailing null bytes).
    fn as_string(&self) -> String {
        let s = std::str::from_utf8(&self.payload).unwrap_or("");
        s.trim_end_matches('\0').to_string()
    }

    /// Parse payload as a u16 bitarray.
    fn as_u16(&self) -> std::result::Result<u16, OxiPhotonError> {
        if self.payload.len() < 2 {
            return Err(OxiPhotonError::Gds(format!(
                "record 0x{:02X}: expected u16 payload (>=2 bytes), got {}",
                self.record_type,
                self.payload.len()
            )));
        }
        Ok(u16::from_be_bytes([self.payload[0], self.payload[1]]))
    }

    /// Parse payload as two real8 values.
    fn as_real8x2(&self) -> std::result::Result<(f64, f64), OxiPhotonError> {
        if self.payload.len() < 16 {
            return Err(OxiPhotonError::Gds(format!(
                "record 0x{:02X}: expected 2×real8 payload (>=16 bytes), got {}",
                self.record_type,
                self.payload.len()
            )));
        }
        let a = real8_decode(
            &self.payload[0..8]
                .try_into()
                .map_err(|_| OxiPhotonError::Gds("real8 slice conversion failed".into()))?,
        );
        let b = real8_decode(
            &self.payload[8..16]
                .try_into()
                .map_err(|_| OxiPhotonError::Gds("real8 slice conversion failed".into()))?,
        );
        Ok((a, b))
    }

    /// Parse payload as a single real8 value.
    fn as_real8(&self) -> std::result::Result<f64, OxiPhotonError> {
        if self.payload.len() < 8 {
            return Err(OxiPhotonError::Gds(format!(
                "record 0x{:02X}: expected real8 payload (>=8 bytes), got {}",
                self.record_type,
                self.payload.len()
            )));
        }
        Ok(real8_decode(&self.payload[0..8].try_into().map_err(
            |_| OxiPhotonError::Gds("real8 slice conversion failed".into()),
        )?))
    }

    /// Parse payload as XY coordinate pairs (i32 pairs, big-endian).
    fn as_xy(&self) -> std::result::Result<Vec<crate::geometry::gds::GdsPoint>, OxiPhotonError> {
        use crate::geometry::gds::GdsPoint;
        if self.payload.len() % 8 != 0 {
            return Err(OxiPhotonError::Gds(format!(
                "XY record payload length {} is not a multiple of 8",
                self.payload.len()
            )));
        }
        let n = self.payload.len() / 8;
        let mut pts = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * 8;
            let x = i32::from_be_bytes([
                self.payload[base],
                self.payload[base + 1],
                self.payload[base + 2],
                self.payload[base + 3],
            ]);
            let y = i32::from_be_bytes([
                self.payload[base + 4],
                self.payload[base + 5],
                self.payload[base + 6],
                self.payload[base + 7],
            ]);
            pts.push(GdsPoint::new(x, y));
        }
        Ok(pts)
    }
}

// Tracking which kind of element is being built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingKind {
    Boundary,
    Path,
    Sref,
    Aref,
    Text,
}

/// Mutable state held while parsing a single element.
struct ElementBuilder {
    kind: PendingKind,
    layer: u16,
    datatype: u16,
    width: i32,
    xy: Vec<crate::geometry::gds::GdsPoint>,
    sname: String,
    strans_flags: u16,
    magnification: f64,
    angle_deg: f64,
    cols: u16,
    rows: u16,
    text_string: String,
}

impl ElementBuilder {
    fn new(kind: PendingKind) -> Self {
        Self {
            kind,
            layer: 0,
            datatype: 0,
            width: 0,
            xy: Vec::new(),
            sname: String::new(),
            strans_flags: 0,
            magnification: 1.0,
            angle_deg: 0.0,
            cols: 1,
            rows: 1,
            text_string: String::new(),
        }
    }

    fn finish(self) -> std::result::Result<crate::geometry::gds::GdsElement, OxiPhotonError> {
        use crate::geometry::gds::{
            GdsAref, GdsBoundary, GdsElement, GdsLayer, GdsPath, GdsSref, GdsText,
        };

        let x_reflection = (self.strans_flags & STRANS_X_REFLECTION) != 0;
        let layer = GdsLayer::new(self.layer, self.datatype);

        match self.kind {
            PendingKind::Boundary => Ok(GdsElement::Boundary(GdsBoundary {
                layer,
                points: self.xy,
            })),
            PendingKind::Path => Ok(GdsElement::Path(GdsPath {
                layer,
                width: self.width,
                points: self.xy,
            })),
            PendingKind::Sref => {
                let origin = self
                    .xy
                    .into_iter()
                    .next()
                    .ok_or_else(|| OxiPhotonError::Gds("SREF missing XY record".into()))?;
                Ok(GdsElement::Sref(GdsSref {
                    sname: self.sname,
                    origin,
                    angle_deg: self.angle_deg,
                    magnification: self.magnification,
                    x_reflection,
                }))
            }
            PendingKind::Aref => {
                if self.xy.len() < 3 {
                    return Err(OxiPhotonError::Gds(format!(
                        "AREF requires exactly 3 XY points, got {}",
                        self.xy.len()
                    )));
                }
                Ok(GdsElement::Aref(GdsAref {
                    ref_name: self.sname,
                    cols: self.cols,
                    rows: self.rows,
                    angle_deg: self.angle_deg,
                    magnification: self.magnification,
                    x_reflection,
                    xy: self.xy,
                }))
            }
            PendingKind::Text => {
                let origin = self
                    .xy
                    .into_iter()
                    .next()
                    .ok_or_else(|| OxiPhotonError::Gds("TEXT missing XY record".into()))?;
                Ok(GdsElement::Text(GdsText {
                    layer,
                    string: self.text_string,
                    origin,
                    height: 0,
                }))
            }
        }
    }
}

/// Reads a binary GDSII stream and deserializes it into a [`GdsLibrary`].
pub struct GdsBinaryReader;

impl GdsBinaryReader {
    /// Deserialize a binary GDSII stream from a byte slice.
    pub fn from_bytes(data: &[u8]) -> std::result::Result<GdsLibrary, OxiPhotonError> {
        use crate::geometry::gds::{GdsCell, GdsLibrary};

        let mut cursor: &[u8] = data;

        let mut lib_name = String::new();
        let mut db_unit_m = 1e-9_f64;
        let mut user_unit_m = 1e-6_f64;
        let mut cells: Vec<GdsCell> = Vec::new();

        let mut current_cell: Option<GdsCell> = None;
        let mut current_elem: Option<ElementBuilder> = None;

        loop {
            let rec = GdsRecord::read_from(&mut cursor)?;

            match rec.record_type {
                rec::HEADER => {
                    // version — nothing to store
                }
                rec::BGNLIB => {
                    // timestamps — skip
                }
                rec::LIBNAME => {
                    lib_name = rec.as_string();
                }
                rec::UNITS => {
                    let (units0, units1) = rec.as_real8x2()?;
                    // units0 = db_unit_m / user_unit_m
                    // units1 = db_unit_m
                    db_unit_m = units1;
                    // user_unit_m = db_unit_m / units0
                    if units0.abs() > f64::EPSILON {
                        user_unit_m = units1 / units0;
                    }
                }
                rec::BGNSTR => {
                    // Start a new cell
                    current_cell = Some(GdsCell::new(""));
                }
                rec::STRNAME => {
                    if let Some(ref mut cell) = current_cell {
                        cell.name = rec.as_string();
                    }
                }
                rec::ENDSTR => {
                    if let Some(cell) = current_cell.take() {
                        cells.push(cell);
                    }
                }
                rec::BOUNDARY => {
                    current_elem = Some(ElementBuilder::new(PendingKind::Boundary));
                }
                rec::PATH => {
                    current_elem = Some(ElementBuilder::new(PendingKind::Path));
                }
                rec::SREF => {
                    current_elem = Some(ElementBuilder::new(PendingKind::Sref));
                }
                rec::AREF => {
                    current_elem = Some(ElementBuilder::new(PendingKind::Aref));
                }
                rec::TEXT => {
                    current_elem = Some(ElementBuilder::new(PendingKind::Text));
                }
                rec::LAYER => {
                    if let Some(ref mut b) = current_elem {
                        b.layer = rec.as_i16()? as u16;
                    }
                }
                rec::DATATYPE => {
                    if let Some(ref mut b) = current_elem {
                        b.datatype = rec.as_i16()? as u16;
                    }
                }
                rec::WIDTH => {
                    if let Some(ref mut b) = current_elem {
                        b.width = rec.as_i32()?;
                    }
                }
                rec::XY => {
                    if let Some(ref mut b) = current_elem {
                        b.xy = rec.as_xy()?;
                    }
                }
                rec::SNAME => {
                    if let Some(ref mut b) = current_elem {
                        b.sname = rec.as_string();
                    }
                }
                rec::COLROW => {
                    if let Some(ref mut b) = current_elem {
                        let (cols, rows) = rec.as_i16x2()?;
                        b.cols = cols as u16;
                        b.rows = rows as u16;
                    }
                }
                rec::STRANS => {
                    if let Some(ref mut b) = current_elem {
                        b.strans_flags = rec.as_u16()?;
                    }
                }
                rec::MAG => {
                    if let Some(ref mut b) = current_elem {
                        b.magnification = rec.as_real8()?;
                    }
                }
                rec::ANGLE => {
                    if let Some(ref mut b) = current_elem {
                        b.angle_deg = rec.as_real8()?;
                    }
                }
                rec::TEXTTYPE => {
                    // ignore value
                }
                rec::STRING => {
                    if let Some(ref mut b) = current_elem {
                        b.text_string = rec.as_string();
                    }
                }
                rec::ENDEL => {
                    if let Some(builder) = current_elem.take() {
                        let elem = builder.finish()?;
                        if let Some(ref mut cell) = current_cell {
                            cell.elements.push(elem);
                        }
                    }
                }
                rec::ENDLIB => {
                    break;
                }
                _ => {
                    // Unknown record type — skip gracefully
                }
            }
        }

        Ok(GdsLibrary {
            name: lib_name,
            db_unit_m,
            user_unit_m,
            cells,
        })
    }

    /// Read and deserialize a binary GDSII file at `path`.
    pub fn from_path(path: &std::path::Path) -> std::result::Result<GdsLibrary, OxiPhotonError> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Self::from_bytes(&data)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

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
