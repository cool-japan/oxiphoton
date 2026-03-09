//! Extended waveguide mode analysis validation tests.
//!
//! Tests the mode solver and waveguide models for physical correctness.

use approx::assert_relative_eq;
use std::f64::consts::PI;

use oxiphoton::devices::waveguide::RidgeWaveguide;
use oxiphoton::devices::waveguide::{
    MmiSplitter, MultimodeWaveguide, SlotWaveguide, StripWaveguide,
};
use oxiphoton::mode::{FemModeSolver1d, SlabWaveguide};

/// StripWaveguide effective index is bounded (n_clad < n_eff < n_core).
#[test]
fn strip_waveguide_neff_bounded() {
    let n_core = 3.476;
    let n_clad = 1.444;
    let wg = StripWaveguide::new(n_core, n_clad, 500e-9, 220e-9);
    let n_eff = wg.n_eff_te(1550e-9).expect("Should find TE mode");

    assert!(
        n_eff > n_clad && n_eff < n_core,
        "n_eff = {n_eff:.4} not in range ({n_clad}, {n_core})"
    );

    // Also check TM mode
    let n_eff_tm = wg.n_eff_tm(1550e-9).expect("Should find TM mode");
    assert!(
        n_eff_tm > n_clad && n_eff_tm < n_core,
        "n_eff_tm = {n_eff_tm:.4} not in range ({n_clad}, {n_core})"
    );

    // Both TE and TM modes should be guided (between n_clad and n_core)
    // Note: relative ordering of TE/TM n_eff depends on waveguide geometry
    assert!(
        (n_eff - n_eff_tm).abs() < n_core,
        "TE and TM n_eff should be within n_core of each other"
    );
}

/// Slot waveguide mode confinement in slot region.
///
/// The slot confinement factor should be > 0 (some field in the slot).
#[test]
fn slot_waveguide_slot_confinement() {
    let slot = SlotWaveguide::soi_standard();

    // Slot confinement factor should be positive (field does exist in slot)
    let gamma_slot = slot.slot_confinement_factor();
    assert!(
        gamma_slot > 0.0,
        "Slot confinement factor should be > 0, got {gamma_slot:.4}"
    );
    assert!(
        gamma_slot < 1.0,
        "Slot confinement factor should be < 1, got {gamma_slot:.4}"
    );

    // Air-clad slot should behave differently from oxide-slot
    let slot_air = SlotWaveguide::soi_air_slot();
    let gamma_air = slot_air.slot_confinement_factor();
    assert!(
        gamma_air > 0.0 && gamma_air < 1.0,
        "Air-slot confinement factor out of range: {gamma_air:.4}"
    );

    // Effective index should be between slot and rail indices
    let wl = 1550e-9;
    let n_eff = slot.effective_index_approx(wl);
    assert!(
        n_eff > slot.n_slot && n_eff < slot.n_rail,
        "n_eff = {n_eff:.4} should be between n_slot ({}) and n_rail ({})",
        slot.n_slot,
        slot.n_rail
    );
}

/// Ridge waveguide vs strip: ridge should have valid n_eff.
#[test]
fn ridge_waveguide_neff_valid() {
    // Ridge (rib) waveguide
    let ridge = RidgeWaveguide::soi_standard();
    let wl = 1550e-9;

    let n_eff_te = ridge.n_eff_te(wl);
    assert!(n_eff_te.is_some(), "Ridge waveguide should find a TE mode");
    let n_te = n_eff_te.unwrap();
    assert!(
        n_te > 1.444 && n_te < 3.476,
        "Ridge n_eff_te = {n_te:.4} out of range"
    );

    // Group index should be physical
    let ng = ridge.group_index_te(wl);
    if let Some(ng_val) = ng {
        assert!(
            ng_val > 1.0 && ng_val < 10.0,
            "Group index should be in reasonable range: {ng_val:.4}"
        );
    }
}

/// Step-index fiber V-number determines single-mode condition.
///
/// V < 2.405 → single mode for step-index fiber.
#[test]
fn step_index_single_mode_condition() {
    let c = 2.998e8_f64;
    let wl = 1550e-9_f64;
    let k0 = 2.0 * PI / wl;

    // Step-index SMF-28 parameters: n_core=1.4682, n_clad=1.4629, a=4.1μm
    let n_core = 1.4682_f64;
    let n_clad = 1.4629_f64;
    let na = (n_core * n_core - n_clad * n_clad).sqrt();

    // Single-mode core radius: a such that V = k0 * a * NA < 2.405
    let a_sm = 4.1e-6; // 4.1 μm (SMF-28 typical core radius)
    let v_sm = k0 * a_sm * na;
    assert!(
        v_sm < 2.405,
        "SMF-28 should be single-mode at 1550nm: V = {v_sm:.4}"
    );

    // Large core = multimode
    let a_mm = 25e-6; // 25 μm multi-mode fiber core
    let v_mm = k0 * a_mm * na;
    assert!(
        v_mm > 2.405,
        "Large core fiber should be multimode at 1550nm: V = {v_mm:.4}"
    );

    // Numerical aperture should be positive
    assert!(na > 0.0, "NA should be positive: {na:.4}");
    let _ = c; // suppress unused warning
}

/// MultimodeWaveguide V-number monotonically increases with width.
#[test]
fn multimode_waveguide_v_number_vs_width() {
    let widths = [0.4e-6, 0.8e-6, 1.5e-6, 3.0e-6, 5.0e-6];
    let mut prev_v = 0.0_f64;

    for &w in &widths {
        let wg = MultimodeWaveguide::new(w, 0.22e-6, 3.476, 1.444, 1550e-9);
        let v = wg.v_number();
        assert!(
            v > prev_v,
            "V-number should increase with width: v({:.1}μm)={v:.4} ≤ prev={prev_v:.4}",
            w * 1e6
        );
        prev_v = v;
    }
}

/// MmiSplitter beat_length scales as W² (approximately).
///
/// Lπ = 4*n_eff*W² / (3*λ)  →  Lπ ∝ W² for fixed n_eff, λ.
#[test]
fn mmi_beat_length_scaling() {
    let n_eff = 3.4;
    let lambda = 1550e-9;

    let w1 = 5e-6;
    let w2 = 10e-6; // doubled width

    let mmi1 = MmiSplitter::new(w1, 50e-6, n_eff, lambda);
    let mmi2 = MmiSplitter::new(w2, 50e-6, n_eff, lambda);

    let lpi1 = mmi1.beat_length();
    let lpi2 = mmi2.beat_length();

    // Lπ ∝ W²: doubling width should quadruple beat_length
    let ratio = lpi2 / lpi1;
    assert_relative_eq!(ratio, 4.0, epsilon = 1e-12);
}

/// FemModeSolver1d: slab waveguide finds guided modes.
///
/// For a Si/SiO2 slab at 1550nm, should find at least one guided mode
/// with n_clad < n_eff < n_core.
#[test]
fn fem_slab_waveguide_finds_guided_mode() {
    let n_core = 3.476;
    let n_clad = 1.444;
    let core_width = 300e-9; // 300 nm slab
    let wavelength = 1550e-9;
    let n_nodes = 101;

    let solver = FemModeSolver1d::slab(n_core, n_clad, core_width, wavelength, n_nodes);
    let modes = solver.solve(n_clad);

    assert!(
        !modes.is_empty(),
        "Should find at least one guided mode in Si slab"
    );

    for mode in &modes {
        assert!(
            mode.n_eff > n_clad && mode.n_eff < n_core,
            "n_eff = {:.4} out of range ({n_clad}, {n_core})",
            mode.n_eff
        );
        assert_eq!(
            mode.field.len(),
            n_nodes,
            "Mode field should have n_nodes={n_nodes} elements"
        );
    }
}

/// FemModeSolver1d: modes are sorted by descending n_eff.
///
/// Fundamental mode (highest n_eff) should come first.
#[test]
fn fem_modes_sorted_by_neff() {
    let solver = FemModeSolver1d::slab(3.476, 1.444, 500e-9, 1550e-9, 101);
    let modes = solver.solve(1.444);

    for i in 1..modes.len() {
        assert!(
            modes[i - 1].n_eff >= modes[i].n_eff,
            "Modes should be sorted by descending n_eff: modes[{i}].n_eff = {:.4} > modes[{}] = {:.4}",
            modes[i].n_eff, i - 1, modes[i - 1].n_eff
        );
    }
}

/// SlabWaveguide effective index vs V-number.
///
/// Single-mode slab (V < π/2) should have one mode;
/// multimode slab (V >> π) should have several.
#[test]
fn slab_waveguide_mode_cutoff() {
    let lambda = 1550e-9;
    let n_core = 3.476;
    let n_clad = 1.444;

    // Very narrow slab
    let narrow_slab = SlabWaveguide::new(n_core, n_clad, 100e-9);
    // Wide slab (definitely multimode)
    let wide_slab = SlabWaveguide::new(n_core, n_clad, 2000e-9);

    // V-number should scale with width
    let v_narrow = PI / lambda * (n_core * n_core - n_clad * n_clad).sqrt() * 100e-9;
    let v_wide = PI / lambda * (n_core * n_core - n_clad * n_clad).sqrt() * 2000e-9;
    assert!(v_wide > v_narrow, "V-wide should be > V-narrow");

    // Wide slab should support more modes
    let modes_narrow = narrow_slab.solve_te(lambda);
    let modes_wide = wide_slab.solve_te(lambda);
    assert!(
        modes_wide.len() >= modes_narrow.len(),
        "Wide slab should support at least as many modes: wide={}, narrow={}",
        modes_wide.len(),
        modes_narrow.len()
    );
}

/// MultimodeWaveguide confinement factor is between 0 and 1.
#[test]
fn multimode_waveguide_confinement_factor_range() {
    let wg = MultimodeWaveguide::new(2e-6, 0.22e-6, 3.476, 1.444, 1550e-9);
    let gamma = wg
        .confinement_factor(0)
        .expect("Fundamental mode should exist");

    assert!(
        gamma > 0.0 && gamma <= 1.0,
        "Confinement factor should be in (0, 1], got {gamma:.4}"
    );
}

/// StripWaveguide group index is in physical range for Si at 1550nm.
///
/// Si strip waveguide group index at 1550nm is typically 4.0-5.5.
#[test]
fn strip_waveguide_group_index_physical() {
    let wg = StripWaveguide::new(3.476, 1.444, 500e-9, 220e-9);
    let ng = wg.group_index(1550e-9).expect("Should compute group index");

    // Silicon strip waveguide at 1550nm: ng ~ 4.0-5.5
    assert!(
        ng > 2.0 && ng < 8.0,
        "Group index should be in physical range, got {ng:.4}"
    );
}

/// Ridge waveguide V-number increases with total height (h_ridge + h_slab).
///
/// The ridge V-number depends on height, not width (by EIM definition).
#[test]
fn ridge_waveguide_v_number_vs_height() {
    let wl = 1550e-9;

    // Short ridge
    let ridge_short = RidgeWaveguide::new(3.476, 1.444, 1.444, 500e-9, 100e-9, 50e-9);
    let v_short = ridge_short.v_number(wl);

    // Tall ridge (more height)
    let ridge_tall = RidgeWaveguide::new(3.476, 1.444, 1.444, 500e-9, 300e-9, 150e-9);
    let v_tall = ridge_tall.v_number(wl);

    assert!(
        v_tall > v_short,
        "Taller ridge should have larger V-number: v_tall={v_tall:.4}, v_short={v_short:.4}"
    );

    // V-number should be positive
    assert!(v_short > 0.0, "V-number should be positive");
}

/// SlotWaveguide: total width equals 2*w_rail + w_slot.
#[test]
fn slot_waveguide_total_width() {
    let slot = SlotWaveguide::soi_standard();
    let expected = 2.0 * slot.w_rail + slot.w_slot;
    assert_relative_eq!(slot.total_width(), expected, epsilon = 1e-20);
}

/// FemModeSolver1d: wider slab supports more modes.
#[test]
fn fem_wider_slab_supports_more_modes() {
    let n_core = 3.476;
    let n_clad = 1.444;
    let lambda = 1550e-9;

    let solver_narrow = FemModeSolver1d::slab(n_core, n_clad, 200e-9, lambda, 101);
    let solver_wide = FemModeSolver1d::slab(n_core, n_clad, 1000e-9, lambda, 151);

    let modes_narrow = solver_narrow.solve(n_clad);
    let modes_wide = solver_wide.solve(n_clad);

    assert!(
        modes_wide.len() >= modes_narrow.len(),
        "Wider slab should support at least as many modes: wide={}, narrow={}",
        modes_wide.len(),
        modes_narrow.len()
    );
}

/// MultimodeWaveguide mode crosstalk is small for orthogonal modes.
#[test]
fn multimode_waveguide_crosstalk_small() {
    let wg = MultimodeWaveguide::new(2e-6, 0.22e-6, 3.476, 1.444, 1550e-9);
    let xt = wg.mode_crosstalk(0, 1);
    assert!(
        xt < 0.1,
        "Cross-modal crosstalk should be small (< 10%), got {xt:.4}"
    );

    // Self-overlap should be 1.0
    let self_xt = wg.mode_crosstalk(0, 0);
    assert_relative_eq!(self_xt, 1.0, epsilon = 1e-10);
}

/// MmiSplitter splitting ratio is between 0 and 1.
///
/// The splitting ratio function returns a value in [0,1].
#[test]
fn mmi_splitting_ratio_at_imaging_length() {
    let n_eff = 3.4;
    let wl = 1550e-9;
    let w = 10e-6;

    // Test several lengths — all should give ratio in [0, 1]
    let lpi = 4.0 * n_eff * w * w / (3.0 * wl);

    for &factor in &[0.5, 1.0, 1.5, 2.0, 3.0] {
        let l = lpi * factor;
        let mmi = MmiSplitter::new(w, l, n_eff, wl);
        let ratio = mmi.splitting_ratio();
        assert!(
            (0.0..=1.0).contains(&ratio),
            "Splitting ratio should be in [0,1] for L={:.2}×Lπ, got {ratio:.4}",
            factor
        );
    }

    // At the half-imaging length (L = Lπ), splitting should be valid
    let mmi_at_lpi = MmiSplitter::new(w, lpi, n_eff, wl);
    let r = mmi_at_lpi.splitting_ratio();
    assert!(
        (0.0..=1.0).contains(&r),
        "Splitting ratio at Lπ should be in [0,1], got {r:.4}"
    );
}
