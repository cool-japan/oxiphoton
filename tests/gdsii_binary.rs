//! Integration tests for the binary GDSII writer and reader.
//!
//! These tests verify spec-compliant round-trip serialisation and specific
//! binary encoding properties of the `GdsBinaryWriter` / `GdsBinaryReader`
//! pair as well as the IBM real8 floating-point codec.

#![cfg(feature = "io-gds")]

use oxiphoton::geometry::gds::{GdsAref, GdsCell, GdsLayer, GdsLibrary, GdsPoint, GdsSref};
use oxiphoton::io::gds_io::{real8_decode, real8_encode, GdsBinaryReader, GdsBinaryWriter};
use std::env::temp_dir;

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn tmp(name: &str) -> std::path::PathBuf {
    temp_dir().join(name)
}

/// Build a minimal library: one cell, one rectangle.
fn minimal_library() -> GdsLibrary {
    let mut lib = GdsLibrary::new("testlib");
    let mut cell = GdsCell::new("TOP");
    cell.add_rect(GdsLayer::new(1, 0), 0, 0, 1000, 500);
    lib.add_cell(cell);
    lib
}

// ─── Test 1: real8 encode/decode round-trip ───────────────────────────────────

#[test]
fn real8_encode_decode_roundtrip() {
    let values: &[f64] = &[
        0.0,
        1.0,
        -1.0,
        1e-9,
        7.777_777,
        1.5e8,
        1.23456789e-15,
        -42.0,
        0.5,
    ];
    for &x in values {
        let encoded = real8_encode(x);
        let decoded = real8_decode(&encoded);
        if x == 0.0 {
            assert_eq!(
                decoded, 0.0,
                "real8 round-trip for 0.0 should be exactly 0.0"
            );
        } else {
            // Allow at most 1 ULP of relative error (IBM real8 has 56-bit mantissa,
            // f64 has 52-bit mantissa — encoding is lossless for these values).
            let rel_err = (decoded - x).abs() / x.abs();
            assert!(
                rel_err < 1e-14,
                "real8 round-trip for {x}: got {decoded}, relative error {rel_err}"
            );
        }
    }
}

// ─── Test 2: real8 known value (1.0) ─────────────────────────────────────────

#[test]
fn real8_known_value_one() {
    let encoded = real8_encode(1.0);
    // byte[0]: sign=0, stored_exp=65 → 0x41
    // byte[1]: mantissa high byte = 0x10 (1/16 × 2^56 = 2^52; high byte of 7 = 0x10)
    // bytes[2..8]: 0
    assert_eq!(encoded[0], 0x41, "real8_encode(1.0) byte[0] should be 0x41");
    assert_eq!(encoded[1], 0x10, "real8_encode(1.0) byte[1] should be 0x10");
    for (offset, &byte) in encoded[2..8].iter().enumerate() {
        assert_eq!(
            byte,
            0x00,
            "real8_encode(1.0) byte[{}] should be 0x00",
            offset + 2
        );
    }
}

// ─── Test 3: single boundary round-trip ───────────────────────────────────────

#[test]
fn single_boundary_roundtrip() {
    let orig = minimal_library();
    let bytes = GdsBinaryWriter::to_bytes(&orig);
    let decoded =
        GdsBinaryReader::from_bytes(&bytes).expect("round-trip from_bytes should not fail");
    assert_eq!(
        orig, decoded,
        "round-tripped library should equal the original"
    );
}

// ─── Test 4: hierarchical SREF + AREF round-trip ─────────────────────────────

#[test]
fn hierarchical_sref_aref_roundtrip() {
    let mut lib = GdsLibrary::new("hierlib");

    // Sub-cell with one rectangle
    let mut sub = GdsCell::new("SUB");
    sub.add_rect(GdsLayer::new(2, 0), 0, 0, 200, 100);
    lib.add_cell(sub);

    // Top cell referencing sub via SREF and AREF
    let mut top = GdsCell::new("TOP");
    top.add_sref("SUB", GdsPoint::new(500, 500));

    let aref = GdsAref::new(
        "SUB",
        4u16,
        3u16,
        GdsPoint::new(0, 0),
        GdsPoint::new(4 * 300, 0),
        GdsPoint::new(0, 3 * 200),
    );
    top.add_aref(aref);
    lib.add_cell(top);

    let bytes = GdsBinaryWriter::to_bytes(&lib);
    let decoded =
        GdsBinaryReader::from_bytes(&bytes).expect("round-trip from_bytes should not fail");
    assert_eq!(
        lib, decoded,
        "hierarchical library should survive binary round-trip"
    );
}

// ─── Test 5: UNITS record decode ─────────────────────────────────────────────

#[test]
fn units_record_correct() {
    let mut lib = GdsLibrary::new("units_test");
    lib.db_unit_m = 1e-9;
    lib.user_unit_m = 1e-6;

    let bytes = GdsBinaryWriter::to_bytes(&lib);
    let decoded =
        GdsBinaryReader::from_bytes(&bytes).expect("from_bytes should not fail for units test");

    assert!(
        (decoded.db_unit_m - 1e-9).abs() < 1e-20,
        "db_unit_m round-trip: expected ~1e-9, got {}",
        decoded.db_unit_m
    );
    assert!(
        (decoded.user_unit_m - 1e-6).abs() < 1e-17,
        "user_unit_m round-trip: expected ~1e-6, got {}",
        decoded.user_unit_m
    );
}

// ─── Test 6: first record bytes ──────────────────────────────────────────────

#[test]
fn header_first_record() {
    let lib = GdsLibrary::new("hdr_test");
    let bytes = GdsBinaryWriter::to_bytes(&lib);

    // First 4 bytes: length=6 (0x00, 0x06), record_type=HEADER=0x00, data_type=0x02
    assert!(bytes.len() >= 6, "output must be at least 6 bytes");
    assert_eq!(bytes[0], 0x00, "byte[0] should be 0x00 (length hi)");
    assert_eq!(bytes[1], 0x06, "byte[1] should be 0x06 (length lo = 6)");
    assert_eq!(
        bytes[2], 0x00,
        "byte[2] should be 0x00 (HEADER record type)"
    );
    assert_eq!(bytes[3], 0x02, "byte[3] should be 0x02 (INT16 data type)");
    // Bytes 4-5: i16 big-endian 600 = 0x0258
    assert_eq!(bytes[4], 0x02, "byte[4] should be 0x02 (600 hi byte)");
    assert_eq!(bytes[5], 0x58, "byte[5] should be 0x58 (600 lo byte)");
}

// ─── Test 7: truncated input returns error ────────────────────────────────────

#[test]
fn truncated_input_returns_error() {
    let lib = minimal_library();
    let bytes = GdsBinaryWriter::to_bytes(&lib);
    let half = bytes.len() / 2;
    let truncated = &bytes[..half];

    let result = GdsBinaryReader::from_bytes(truncated);
    assert!(
        result.is_err(),
        "truncated GDSII input must return Err, not Ok"
    );
}

// ─── Test 8: file write/read round-trip ──────────────────────────────────────

#[test]
fn file_roundtrip() {
    let orig = minimal_library();
    let path = tmp("oxiphoton_test_gdsii_binary_roundtrip.gds");

    GdsBinaryWriter::to_path(&orig, &path).expect("to_path should succeed");

    let decoded = GdsBinaryReader::from_path(&path).expect("from_path should succeed");

    // Clean up
    let _ = std::fs::remove_file(&path);

    assert_eq!(orig, decoded, "file round-trip should preserve the library");
}

// ─── Test 9: SREF with non-default transform ─────────────────────────────────

#[test]
fn sref_transform_roundtrip() {
    let mut lib = GdsLibrary::new("sref_transform");
    let sub = GdsCell::new("SUB");
    lib.add_cell(sub);

    let mut top = GdsCell::new("TOP");
    let mut sref = GdsSref::new("SUB", GdsPoint::new(100, 200));
    sref.angle_deg = 90.0;
    sref.magnification = 2.0;
    sref.x_reflection = true;
    top.elements
        .push(oxiphoton::geometry::gds::GdsElement::Sref(sref));
    lib.add_cell(top);

    let bytes = GdsBinaryWriter::to_bytes(&lib);
    let decoded = GdsBinaryReader::from_bytes(&bytes)
        .expect("from_bytes should not fail for sref transform test");
    assert_eq!(
        lib, decoded,
        "SREF with transform should round-trip correctly"
    );
}
