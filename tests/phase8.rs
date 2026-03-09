/// Phase 8 integration tests — new grid, monitor, S-matrix, FDTD, and WDM modules.
use oxiphoton::fdtd::{
    BoundaryConfig, Dimensions, EUpdateCoeffs, Fdtd2dTm, FieldMonitor1d, FieldMonitor2d,
    HUpdateCoeffs, SimulationConfig,
};
use oxiphoton::geometry::grid::{GridSpec1d, GridSpec2d, GridSpec3d};
use oxiphoton::interconnect::wdm::{CrosstalkMatrix, DwdmChannelPlan, OpticalAddDropMux};

// ── GridSpec1d ────────────────────────────────────────────────────────────────

#[test]
fn grid1d_uniform_cell_count() {
    let g = GridSpec1d::uniform(0.0, 1e-6, 100);
    assert_eq!(g.n_cells(), 100);
}

#[test]
fn grid1d_uniform_dx_correct() {
    let g = GridSpec1d::uniform(0.0, 1e-6, 100);
    let dx = g.spacings()[0];
    assert!((dx - 10e-9).abs() < 1e-18, "dx={dx:.3e}");
}

#[test]
fn grid1d_nonuniform_cell_count() {
    let edges: Vec<f64> = (0..=10).map(|i| i as f64 * 1e-7).collect();
    let g = GridSpec1d::nonuniform(edges);
    assert_eq!(g.n_cells(), 10);
}

#[test]
fn grid1d_xcoord_center_in_range() {
    let start = 0.0_f64;
    let end = 1e-6_f64;
    let g = GridSpec1d::uniform(start, end, 50);
    for &c in &g.centers {
        assert!(c > start && c < end);
    }
}

#[test]
fn grid1d_find_cell_at_center() {
    let g = GridSpec1d::uniform(0.0, 1.0, 10);
    // Center of cell 5 is at 0.55
    let cell = g.find_cell(0.55);
    assert_eq!(cell, 5);
}

// ── GridSpec2d ────────────────────────────────────────────────────────────────

#[test]
fn grid2d_uniform_dimensions() {
    let g = GridSpec2d::uniform(0.0, 4e-6, 40, 0.0, 3e-6, 30);
    assert_eq!(g.nx(), 40);
    assert_eq!(g.ny(), 30);
}

#[test]
fn grid2d_idx_row_major() {
    let g = GridSpec2d::uniform(0.0, 1.0, 5, 0.0, 1.0, 4);
    // idx(2, 3) = 2*4 + 3 = 11
    assert_eq!(g.idx(2, 3), 11);
}

#[test]
fn grid2d_yee_e_positions_count() {
    let g = GridSpec2d::uniform(0.0, 1e-6, 10, 0.0, 1e-6, 10);
    let positions = g.yee_e_positions();
    assert_eq!(positions.len(), 100);
}

// ── GridSpec3d ────────────────────────────────────────────────────────────────

#[test]
fn grid3d_uniform_dimensions() {
    let g = GridSpec3d::uniform(0.0, 1e-6, 10, 0.0, 1e-6, 10, 0.0, 1e-6, 10);
    assert_eq!(g.nx(), 10);
    assert_eq!(g.ny(), 10);
    assert_eq!(g.nz(), 10);
    assert_eq!(g.n_total(), 1000);
}

#[test]
fn grid3d_fill_box_material() {
    let g = GridSpec3d::uniform(0.0, 1.0, 10, 0.0, 1.0, 10, 0.0, 1.0, 10);
    let mut map = vec![0_usize; g.n_total()];
    g.fill_box_material(&mut map, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 1);
    // The interior cells should be labelled 1
    let center_idx = g.idx(5, 5, 5);
    assert_eq!(map[center_idx], 1);
    // Corner cells should remain 0
    assert_eq!(map[g.idx(0, 0, 0)], 0);
}

// ── FieldMonitor1d ────────────────────────────────────────────────────────────

#[test]
fn field_monitor_1d_n_snapshots_at_interval() {
    let mut mon = FieldMonitor1d::new(50, 10);
    let ex = vec![1.0_f64; 50];
    let hy = vec![0.5_f64; 50];
    for step in 0..25usize {
        mon.record(step, step as f64 * 1e-15, &ex, &hy);
    }
    // Steps 0, 10, 20 → 3 snapshots
    assert_eq!(mon.n_snapshots(), 3);
}

#[test]
fn field_monitor_1d_time_averaged_intensity_positive() {
    let mut mon = FieldMonitor1d::new(20, 1);
    let ex: Vec<f64> = (0..20).map(|i| (i as f64) * 0.1).collect();
    let hy = vec![1.0_f64; 20];
    mon.record(0, 0.0, &ex, &hy);
    let avg = mon.time_averaged_intensity();
    assert!(avg.iter().sum::<f64>() > 0.0);
}

#[test]
fn field_monitor_1d_empty_intensity_zero() {
    let mon = FieldMonitor1d::new(30, 5);
    let avg = mon.time_averaged_intensity();
    assert!(avg.iter().all(|&v| v == 0.0));
}

// ── FieldMonitor2d ────────────────────────────────────────────────────────────

#[test]
fn field_monitor_2d_n_snapshots() {
    let mut mon = FieldMonitor2d::new(8, 8, 5);
    let n = 64;
    let hz = vec![1.0_f64; n];
    let ex = vec![0.5_f64; n];
    let ey = vec![0.5_f64; n];
    for step in 0..15usize {
        mon.record(step, step as f64 * 1e-15, &hz, &ex, &ey);
    }
    // Steps 0, 5, 10 → 3 snapshots
    assert_eq!(mon.n_snapshots(), 3);
}

#[test]
fn field_monitor_2d_time_averaged_intensity_positive() {
    let mut mon = FieldMonitor2d::new(4, 4, 1);
    let n = 16;
    let hz = vec![0.0_f64; n];
    let ex = vec![2.0_f64; n];
    let ey = vec![1.0_f64; n];
    mon.record(0, 0.0, &hz, &ex, &ey);
    let avg = mon.time_averaged_intensity();
    // intensity = ex² + ey² = 4 + 1 = 5 for every cell
    assert!((avg[0] - 5.0).abs() < 1e-12, "avg[0]={}", avg[0]);
}

// ── SimulationConfig ──────────────────────────────────────────────────────────

#[test]
fn simulation_config_auto_validates() {
    let dims = Dimensions::TwoD { nx: 80, ny: 80 };
    let cfg = SimulationConfig::auto(
        dims,
        oxiphoton::units::electromagnetic::Wavelength(1550e-9),
        3.5,
        20,
    );
    assert!(cfg.validate().is_ok());
    assert!(cfg.grid.dx > 0.0);
}

#[test]
fn simulation_config_optimal_dt_positive() {
    let dims = Dimensions::OneD { nz: 200 };
    let cfg = SimulationConfig::auto(
        dims,
        oxiphoton::units::electromagnetic::Wavelength(1550e-9),
        1.5,
        20,
    );
    let dt = cfg.optimal_dt();
    assert!(dt > 0.0 && dt < 1e-12);
}

// ── EUpdateCoeffs / HUpdateCoeffs ────────────────────────────────────────────

#[test]
fn eupdate_coeffs_lossless_ca_is_one() {
    let eps_r = vec![1.0_f64; 50];
    let dt = 1e-16_f64;
    let dz = 10e-9_f64;
    let coeffs = EUpdateCoeffs::lossless_1d(&eps_r, dt, dz);
    assert_eq!(coeffs.n, 50);
    assert!(!coeffs.ca.is_empty());
    assert!(!coeffs.cb.is_empty());
    // For lossless sigma=0: Ca = 1.0
    for &ca in &coeffs.ca {
        assert!((ca - 1.0).abs() < 1e-12, "ca={ca}");
    }
}

#[test]
fn eupdate_coeffs_with_loss_ca_less_one() {
    let eps_r = vec![2.25_f64; 30];
    let sigma_e = vec![10.0_f64; 30]; // lossy
    let dt = 1e-16_f64;
    let dz = 10e-9_f64;
    let coeffs = EUpdateCoeffs::new_1d(&eps_r, &sigma_e, dt, dz);
    for &ca in &coeffs.ca {
        assert!(ca < 1.0);
    }
}

#[test]
fn hupdate_coeffs_lossless_da_is_one() {
    let n = 40_usize;
    let dt = 1e-16_f64;
    let dz = 10e-9_f64;
    let coeffs = HUpdateCoeffs::lossless_1d(n, dt, dz);
    assert_eq!(coeffs.n, n);
    assert!(!coeffs.da.is_empty());
    assert!(!coeffs.db.is_empty());
    for &da in &coeffs.da {
        assert!((da - 1.0).abs() < 1e-12);
    }
}

#[test]
fn hupdate_coeffs_db_positive() {
    let n = 20_usize;
    let dt = 1e-16_f64;
    let dz = 10e-9_f64;
    let coeffs = HUpdateCoeffs::lossless_1d(n, dt, dz);
    for &db in &coeffs.db {
        assert!(db > 0.0);
    }
}

// ── Fdtd2dTm ─────────────────────────────────────────────────────────────────

#[test]
fn fdtd2d_tm_create_and_step_once() {
    let bc = BoundaryConfig::pml(5);
    let mut sim = Fdtd2dTm::new(20, 20, 50e-9, 50e-9, &bc);
    // All fields start at zero
    assert!(sim.ez.iter().all(|&v| v == 0.0));
    // Inject a pulse at center
    sim.inject_ez(10, 10, 1.0);
    sim.step();
    // After one step, time advances and some fields non-zero
    assert!(sim.time_step == 1);
    // Ez at injection point may have changed; at least no panic
    let sum: f64 = sim.ez.iter().map(|v| v.abs()).sum();
    assert!(sum > 0.0);
}

#[test]
fn fdtd2d_tm_fill_eps_box() {
    let bc = BoundaryConfig::pml(5);
    let mut sim = Fdtd2dTm::new(30, 30, 50e-9, 50e-9, &bc);
    sim.fill_eps_box(5, 25, 5, 25, 12.25); // Si eps = 3.5^2
    let idx = 10 * 30 + 10;
    assert!((sim.eps_r[idx] - 12.25).abs() < 1e-10);
    // Outside the box should still be 1.0
    assert!((sim.eps_r[0] - 1.0).abs() < 1e-10);
}

// ── DwdmChannelPlan ───────────────────────────────────────────────────────────

#[test]
fn dwdm_channel_plan_8ch_count() {
    let plan = DwdmChannelPlan::new(1550.0, 100.0, 8);
    assert_eq!(plan.n_channels, 8);
}

#[test]
fn dwdm_channel_plan_wavelengths_span() {
    let plan = DwdmChannelPlan::new(1550.0, 100.0, 8);
    let wavelengths = plan.channel_wavelengths_nm();
    assert_eq!(wavelengths.len(), 8);
    let min_wl = wavelengths.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_wl = wavelengths.iter().cloned().fold(0.0_f64, f64::max);
    // 8 channels × 100 GHz ≈ 0.8 nm spacing → total span ~5.6 nm
    assert!(max_wl > min_wl, "channels must span some wavelength range");
    // All wavelengths should be in the C-band neighborhood
    assert!(min_wl > 1500.0 && max_wl < 1600.0);
}

#[test]
fn dwdm_channel_plan_find_center_channel() {
    let plan = DwdmChannelPlan::new(1550.0, 100.0, 8);
    // The center wavelength should be findable
    let idx = plan.channel_index_for_wavelength(1550.0);
    assert!(idx.is_some(), "should find a channel near 1550 nm");
}

// ── CrosstalkMatrix ───────────────────────────────────────────────────────────

#[test]
fn crosstalk_matrix_set_get() {
    let mut xt = CrosstalkMatrix::new(4);
    xt.set_crosstalk(0, 1, -30.0); // -30 dB
                                   // Verify the diagonal crosstalk is effectively 0 (or not set to that value)
                                   // and off-diagonal is set
    let total = xt.total_interference_db(1);
    // With one interferer at -30 dB, total interference should be around -30 dB
    assert!(total < 0.0);
}

#[test]
fn crosstalk_matrix_default_crosstalk_very_low() {
    // Default initializes all off-diagonal entries to -60 dB
    let xt = CrosstalkMatrix::new(4);
    let total = xt.total_interference_db(0);
    // Sum of 3 channels × 10^(-6) → total ≈ -55 dB (very low but not -inf)
    assert!(
        total < -50.0,
        "Default crosstalk should be very low: {total:.2}"
    );
}

// ── OpticalAddDropMux ─────────────────────────────────────────────────────────

#[test]
fn oadm_add_and_drop_channel() {
    let plan = DwdmChannelPlan::new(1550.0, 100.0, 4);
    let mut oadm = OpticalAddDropMux::new(plan);
    oadm.add_channel(0, 0.0); // add channel 0 at 0 dBm
    let dropped = oadm.drop_channel(0);
    assert!(dropped.is_some());
}

#[test]
fn oadm_express_channels_excludes_dropped() {
    let plan = DwdmChannelPlan::new(1550.0, 100.0, 4);
    let mut oadm = OpticalAddDropMux::new(plan);
    // Add channels 0,1,2,3
    for i in 0..4 {
        oadm.add_channel(i, 0.0);
    }
    // Drop channel 1
    let _ = oadm.drop_channel(1);
    let express = oadm.express_channels();
    // Express should not include dropped channel
    let ch_indices: Vec<usize> = express.iter().map(|(i, _)| *i).collect();
    assert!(!ch_indices.contains(&1));
}

#[test]
fn oadm_insertion_loss_positive() {
    let plan = DwdmChannelPlan::new(1550.0, 100.0, 4);
    let oadm = OpticalAddDropMux::new(plan);
    let loss = oadm.insertion_loss_db();
    assert!(loss >= 0.0);
}
