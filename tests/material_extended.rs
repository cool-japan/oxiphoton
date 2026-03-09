use oxiphoton::material::gain::GainMedium;
/// Extended material tests — Sellmeier glasses, tabulated metals,
/// MaterialDatabase, gain media, and GVD.
use oxiphoton::material::{DispersiveMaterial, MaterialDatabase, Sellmeier, Tabulated};
use oxiphoton::units::Wavelength;

// ── Sellmeier: N-BK7 ──────────────────────────────────────────────────────────

#[test]
fn nbk7_n_at_1550nm_about_1_50() {
    let mat = Sellmeier::nbk7();
    let n = mat.refractive_index(Wavelength(1550e-9)).n;
    // N-BK7 at 1550nm: n ≈ 1.499–1.502
    assert!(n > 1.48 && n < 1.52, "N-BK7 n at 1550nm: {n:.4}");
}

#[test]
fn nbk7_n_at_589nm_near_catalog() {
    let mat = Sellmeier::nbk7();
    let n = mat.refractive_index(Wavelength(589e-9)).n;
    // Catalog value: 1.5168
    assert!(
        (n - 1.5168).abs() < 0.005,
        "N-BK7 n at 589nm should be ~1.517: {n:.5}"
    );
}

#[test]
fn nbk7_n_increases_toward_shorter_wavelength() {
    let mat = Sellmeier::nbk7();
    let n_vis = mat.refractive_index(Wavelength(400e-9)).n;
    let n_ir = mat.refractive_index(Wavelength(1000e-9)).n;
    assert!(n_vis > n_ir, "Normal dispersion: n should increase at shorter wavelength: n_vis={n_vis:.4} n_ir={n_ir:.4}");
}

// ── Sellmeier: N-SF11 ─────────────────────────────────────────────────────────

#[test]
fn nsf11_n_at_1550nm_about_1_75() {
    let mat = Sellmeier::nsf11();
    let n = mat.refractive_index(Wavelength(1550e-9)).n;
    // N-SF11 is dense flint glass, n > 1.7 at IR
    assert!(n > 1.73 && n < 1.80, "N-SF11 n at 1550nm: {n:.4}");
}

#[test]
fn nsf11_n_at_589nm_near_1_78() {
    let mat = Sellmeier::nsf11();
    let n = mat.refractive_index(Wavelength(589e-9)).n;
    // Catalog: n_d ≈ 1.7847
    assert!(
        (n - 1.7847).abs() < 0.01,
        "N-SF11 n at 589nm should be ~1.785: {n:.4}"
    );
}

// ── Sellmeier: Germanium ──────────────────────────────────────────────────────

#[test]
fn ge_n_at_10um_about_4() {
    let mat = Sellmeier::ge();
    let n = mat.refractive_index(Wavelength(10e-6)).n;
    // Ge at 10 μm (IR optics): n ≈ 4.0
    assert!(n > 3.8 && n < 4.2, "Ge n at 10μm should be ~4.0: {n:.4}");
}

#[test]
fn ge_is_valid_at_10um() {
    let mat = Sellmeier::ge();
    assert!(mat.is_wavelength_valid(10e-6));
}

// ── Sellmeier: Sapphire ───────────────────────────────────────────────────────

#[test]
fn sapphire_n_at_800nm_about_1_76() {
    let mat = Sellmeier::sapphire();
    let n = mat.refractive_index(Wavelength(800e-9)).n;
    // Sapphire ordinary ray at 800nm: n ≈ 1.76
    assert!(n > 1.74 && n < 1.78, "Sapphire n at 800nm: {n:.4}");
}

#[test]
fn sapphire_n_at_589nm_about_1_77() {
    let mat = Sellmeier::sapphire();
    let n = mat.refractive_index(Wavelength(589e-9)).n;
    // Sapphire n_d ≈ 1.7681
    assert!(
        (n - 1.768).abs() < 0.01,
        "Sapphire n at 589nm should be ~1.768: {n:.4}"
    );
}

// ── Tabulated: Au Palik ───────────────────────────────────────────────────────

#[test]
fn au_palik_n_at_500nm_positive() {
    let mat = Tabulated::au_palik();
    let ri = mat.refractive_index(Wavelength(500e-9));
    assert!(ri.n > 0.0, "Au n at 500nm should be positive: {}", ri.n);
}

#[test]
fn au_palik_k_at_500nm_positive() {
    let mat = Tabulated::au_palik();
    let ri = mat.refractive_index(Wavelength(500e-9));
    // Au extinction coefficient k > 1 in visible
    assert!(ri.k > 0.0, "Au k at 500nm should be > 0: {}", ri.k);
}

#[test]
fn au_palik_name_is_au() {
    let mat = Tabulated::au_palik();
    assert_eq!(mat.name(), "Au");
}

// ── Tabulated: Ag Palik ───────────────────────────────────────────────────────

#[test]
fn ag_palik_eps_real_negative_in_visible() {
    let mat = Tabulated::ag_palik();
    let ri = mat.refractive_index(Wavelength(600e-9));
    // For metals: ε = (n + ik)² = n² - k² + 2ink
    // Real part: n² - k² < 0 for Ag at 600nm (Drude metal)
    let eps_real = ri.n * ri.n - ri.k * ri.k;
    assert!(
        eps_real < 0.0,
        "Ag real permittivity at 600nm should be negative: {eps_real}"
    );
}

#[test]
fn ag_palik_k_at_500nm_positive() {
    let mat = Tabulated::ag_palik();
    let ri = mat.refractive_index(Wavelength(500e-9));
    assert!(ri.k > 0.0, "Ag k at 500nm should be > 0: {}", ri.k);
}

// ── MaterialDatabase ──────────────────────────────────────────────────────────

#[test]
fn material_database_load_default_non_empty() {
    let db = MaterialDatabase::load_default();
    // Should have at least Si and SiO2
    assert!(db.get("Si").is_some(), "Database should contain Si");
    assert!(db.get("SiO2").is_some(), "Database should contain SiO2");
}

#[test]
fn material_database_search_silicon() {
    let db = MaterialDatabase::load_default();
    let results = db.search("si");
    assert!(
        !results.is_empty(),
        "Search for 'si' should find at least one material"
    );
}

#[test]
fn material_database_search_nbk7() {
    let db = MaterialDatabase::load_default();
    // N-BK7 is stored with its Sellmeier name "N-BK7"
    let results = db.search("bk7");
    assert!(
        !results.is_empty(),
        "Search for 'bk7' should find N-BK7: {:?}",
        results
    );
}

#[test]
fn material_database_get_missing_returns_none() {
    let db = MaterialDatabase::load_default();
    assert!(db.get("NonExistentMaterial_XYZ").is_none());
}

#[test]
fn material_database_si_n_eff_at_1550nm() {
    let db = MaterialDatabase::load_default();
    let si = db.get("Si").unwrap();
    let n = si.refractive_index(Wavelength(1550e-9)).n;
    assert!(
        (n - 3.476).abs() < 0.02,
        "Si n at 1550nm should be ~3.476: {n:.4}"
    );
}

// ── GainMedium ────────────────────────────────────────────────────────────────

#[test]
fn gain_medium_edfa_g0_positive() {
    let gm = GainMedium::edfa_c_band();
    assert!(
        gm.g0 > 0.0,
        "EDFA small-signal gain should be positive: {}",
        gm.g0
    );
}

#[test]
fn gain_medium_edfa_above_threshold() {
    let gm = GainMedium::edfa_c_band();
    assert!(gm.above_threshold(), "EDFA should be above threshold");
}

#[test]
fn gain_medium_edfa_spectral_gain_at_center() {
    let gm = GainMedium::edfa_c_band();
    let g_center = gm.spectral_gain(gm.lambda_center);
    assert!(
        (g_center - gm.g0).abs() < 1e-12,
        "Gain at center should equal g0: {g_center}"
    );
}

#[test]
fn gain_medium_edfa_spectral_gain_decreases_away_from_center() {
    let gm = GainMedium::edfa_c_band();
    let g_center = gm.spectral_gain(gm.lambda_center);
    let g_edge = gm.spectral_gain(gm.lambda_center + 30e-9); // 30 nm away
    assert!(
        g_center > g_edge,
        "Gain should decrease away from center: {g_center} vs {g_edge}"
    );
}

#[test]
fn gain_medium_edfa_small_signal_power_gain_positive() {
    let gm = GainMedium::edfa_c_band();
    let g = gm.small_signal_power_gain(1.0); // 1 m
    assert!(
        g > 1.0,
        "Small-signal power gain over 1m should be > 1: {g}"
    );
}

#[test]
fn gain_medium_soa_bandwidth_larger_than_edfa() {
    let edfa = GainMedium::edfa_c_band();
    let soa = GainMedium::soa_inp();
    assert!(
        soa.bandwidth > edfa.bandwidth,
        "SOA bandwidth should be wider than EDFA"
    );
}

// ── Sellmeier GVD ─────────────────────────────────────────────────────────────

#[test]
fn silica_gvd_at_1550nm_is_anomalous() {
    // SiO2 at 1550nm has anomalous dispersion: D > 0 (ps/nm/km)
    let mat = Sellmeier::sio2();
    let wl = Wavelength(1550e-9);
    let d = mat.gvd_ps_per_nm_km(wl);
    // Standard single-mode fiber D ≈ 17 ps/(nm·km) at 1550nm
    assert!(
        d > 0.0,
        "Silica D at 1550nm should be positive (anomalous): {d:.2}"
    );
    assert!(d < 30.0, "Silica D at 1550nm too large: {d:.2}");
}

#[test]
fn silica_gvd_at_800nm_is_normal() {
    // Below ZDW (~1270nm), silica has normal dispersion: D < 0
    let mat = Sellmeier::sio2();
    let wl = Wavelength(800e-9);
    let d = mat.gvd_ps_per_nm_km(wl);
    assert!(
        d < 0.0,
        "Silica D at 800nm should be negative (normal): {d:.2}"
    );
}

#[test]
fn group_index_silica_at_1550nm_physical() {
    let mat = Sellmeier::sio2();
    let ng = mat.group_index(Wavelength(1550e-9));
    // Silica group index ~1.467 at 1550nm
    assert!(
        ng > 1.4 && ng < 1.6,
        "Silica group index at 1550nm: {ng:.4}"
    );
}

#[test]
fn sellmeier_name_matches_material() {
    let si3n4 = Sellmeier::si3n4();
    assert_eq!(si3n4.name, "Si3N4");
    let sio2 = Sellmeier::sio2();
    assert_eq!(sio2.name, "SiO2");
}
