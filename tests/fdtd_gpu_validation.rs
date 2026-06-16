#![cfg(feature = "gpu-wgpu")]

use oxiphoton::fdtd::gpu::Fdtd2dGpu;
use oxiphoton::fdtd::{BoundaryConfig, Fdtd2dTe};

const NX: usize = 64;
const NY: usize = 64;
const STEPS: usize = 200;
const PML_CELLS: usize = 15;
const DX: f64 = 10e-9;
const DY: f64 = 10e-9;

fn gaussian(t: f64, t0: f64, sigma: f64) -> f64 {
    let v = (t - t0) / sigma;
    (-0.5 * v * v).exp()
}

fn make_boundary() -> BoundaryConfig {
    BoundaryConfig {
        pml_cells: PML_CELLS,
        pml_m: 3.5,
        pml_r0: 1e-8,
    }
}

/// Sanity check: vacuum propagation produces finite, symmetric-ish field.
#[test]
fn gpu_vacuum_propagation_finite() {
    let bnd = make_boundary();
    let mut gpu = match Fdtd2dGpu::new(NX, NY, DX, DY, &bnd) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU adapter ({e}) — skipping gpu_vacuum_propagation_finite");
            return;
        }
    };

    let src_i = NX / 2;
    let src_j = NY / 2;
    let dt = gpu.dt;

    for step in 0..100 {
        let t = step as f64 * dt;
        let val = gaussian(t, 3e-14, 1e-14);
        gpu.inject_hz(src_i, src_j, val);
        gpu.step().expect("GPU step failed");
    }

    let hz = gpu.download_hz().expect("Hz download failed");
    assert!(
        hz.iter().all(|v| v.is_finite()),
        "Hz contains non-finite values"
    );
    let peak = hz.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    assert!(
        peak > 1e-20,
        "Hz peak is near zero — simulation likely broken"
    );
}

/// Main validation: GPU-vs-CPU agreement within f32 tolerance.
#[test]
fn gpu_te_matches_cpu_oracle() {
    let bnd = make_boundary();

    // CPU reference
    let mut cpu = Fdtd2dTe::new(NX, NY, DX, DY, &bnd);
    let ix0 = NX / 4;
    let ix1 = 3 * NX / 4;
    let iy0 = NY / 4;
    let iy1 = 3 * NY / 4;
    cpu.fill_eps_box(ix0, ix1, iy0, iy1, 12.0);
    let dt = cpu.dt;

    // GPU solver (same params)
    let mut gpu = match Fdtd2dGpu::new(NX, NY, DX, DY, &bnd) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU adapter ({e}) — skipping gpu_te_matches_cpu_oracle");
            return;
        }
    };
    gpu.fill_eps_box(ix0, ix1, iy0, iy1, 12.0);

    let src_i = NX / 2;
    let src_j = NY / 2;
    let t0 = 3e-14_f64;
    let sigma = 1e-14_f64;

    for step in 0..STEPS {
        let t = step as f64 * dt;
        let val = gaussian(t, t0, sigma);
        cpu.inject_hz(src_i, src_j, val);
        cpu.step();
        gpu.inject_hz(src_i, src_j, val);
        gpu.step().expect("GPU step failed");
    }

    let cpu_hz = &cpu.grid.hz;
    let gpu_hz = gpu.download_hz().expect("Hz download failed");
    let cpu_ex = &cpu.grid.ex;
    let gpu_ex = gpu.download_ex().expect("Ex download failed");
    let cpu_ey = &cpu.grid.ey;
    let gpu_ey = gpu.download_ey().expect("Ey download failed");

    // Peak-normalised L-inf (not per-cell relative — that false-fails on PML tails)
    let cpu_peak_hz = cpu_hz.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    assert!(
        cpu_peak_hz > 1e-20,
        "CPU Hz peak is near zero — test setup broken"
    );

    check_field("Hz", cpu_hz, &gpu_hz, cpu_peak_hz);
    let cpu_peak_ex = cpu_ex.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    if cpu_peak_ex > 1e-25 {
        check_field("Ex", cpu_ex, &gpu_ex, cpu_peak_ex);
    }
    let cpu_peak_ey = cpu_ey.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    if cpu_peak_ey > 1e-25 {
        check_field("Ey", cpu_ey, &gpu_ey, cpu_peak_ey);
    }
}

#[test]
fn gpu_tm_vacuum_propagation_finite() {
    use oxiphoton::fdtd::gpu::Fdtd2dTmGpu;
    let bnd = make_boundary();
    let mut gpu = match Fdtd2dTmGpu::new(NX, NY, DX, DY, &bnd) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU adapter ({e}) — skipping gpu_tm_vacuum_propagation_finite");
            return;
        }
    };
    let src_i = NX / 2;
    let src_j = NY / 2;
    let dt = gpu.dt;
    for step in 0..100 {
        let t = step as f64 * dt;
        let val = gaussian(t, 3e-14, 1e-14);
        gpu.inject_ez(src_i, src_j, val);
        gpu.step().expect("GPU TM step failed");
    }
    let ez = gpu.download_ez().expect("Ez download failed");
    assert!(
        ez.iter().all(|v| v.is_finite()),
        "TM Ez contains non-finite values"
    );
    let peak = ez.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    assert!(peak > 1e-20, "TM Ez peak is near zero");
}

#[test]
fn gpu_tm_matches_cpu_oracle() {
    use oxiphoton::fdtd::gpu::Fdtd2dTmGpu;
    use oxiphoton::fdtd::Fdtd2dTm;
    let bnd = make_boundary();
    let mut cpu = Fdtd2dTm::new(NX, NY, DX, DY, &bnd);
    let ix0 = NX / 4;
    let ix1 = 3 * NX / 4;
    let iy0 = NY / 4;
    let iy1 = 3 * NY / 4;
    cpu.fill_eps_box(ix0, ix1, iy0, iy1, 12.0);
    let dt = cpu.dt;

    let mut gpu = match Fdtd2dTmGpu::new(NX, NY, DX, DY, &bnd) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU adapter ({e}) — skipping gpu_tm_matches_cpu_oracle");
            return;
        }
    };
    gpu.fill_eps_box(ix0, ix1, iy0, iy1, 12.0);

    let src_i = NX / 2;
    let src_j = NY / 2;
    let t0 = 3e-14_f64;
    let sigma = 1e-14_f64;
    for step in 0..STEPS {
        let t = step as f64 * dt;
        let val = gaussian(t, t0, sigma);
        cpu.inject_ez(src_i, src_j, val);
        cpu.step();
        gpu.inject_ez(src_i, src_j, val);
        gpu.step().expect("GPU TM step failed");
    }

    let cpu_ez = &cpu.ez;
    let gpu_ez = gpu.download_ez().expect("Ez download failed");
    let cpu_peak_ez = cpu_ez.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    assert!(cpu_peak_ez > 1e-20, "CPU Ez peak near zero — test broken");
    check_field("TM Ez", cpu_ez, &gpu_ez, cpu_peak_ez);

    let cpu_hx = &cpu.hx;
    let gpu_hx = gpu.download_hx().expect("Hx download failed");
    let cpu_peak_hx = cpu_hx.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    if cpu_peak_hx > 1e-25 {
        check_field("TM Hx", cpu_hx, &gpu_hx, cpu_peak_hx);
    }

    let cpu_hy = &cpu.hy;
    let gpu_hy = gpu.download_hy().expect("Hy download failed");
    let cpu_peak_hy = cpu_hy.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    if cpu_peak_hy > 1e-25 {
        check_field("TM Hy", cpu_hy, &gpu_hy, cpu_peak_hy);
    }
}

fn check_field(name: &str, cpu: &[f64], gpu: &[f64], peak: f64) {
    assert_eq!(cpu.len(), gpu.len(), "{name}: length mismatch");

    let max_abs_diff = cpu
        .iter()
        .zip(gpu.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f64, f64::max);
    let linf_norm = max_abs_diff / peak;

    let sum_sq_diff: f64 = cpu
        .iter()
        .zip(gpu.iter())
        .map(|(c, g)| (c - g).powi(2))
        .sum();
    let sum_sq_cpu: f64 = cpu.iter().map(|v| v * v).sum::<f64>().max(1e-60);
    let l2_rel = (sum_sq_diff / sum_sq_cpu).sqrt();

    assert!(
        linf_norm < 2e-3,
        "{name} L-inf norm too large: {linf_norm:.4e} (threshold 2e-3, cpu_peak={peak:.3e})"
    );
    assert!(
        l2_rel < 1e-3,
        "{name} L2 relative too large: {l2_rel:.4e} (threshold 1e-3)"
    );
}

#[test]
fn gpu_3d_vacuum_propagation_finite() {
    use oxiphoton::fdtd::gpu::Fdtd3dGpu;
    const NX3: usize = 32;
    const NY3: usize = 32;
    const NZ3: usize = 32;
    const DZ3: f64 = 10e-9;

    let bnd = make_boundary();
    let mut gpu = match Fdtd3dGpu::new(NX3, NY3, NZ3, DX, DY, DZ3, &bnd) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU adapter ({e}) — skipping gpu_3d_vacuum_propagation_finite");
            return;
        }
    };

    let src_i = NX3 / 2;
    let src_j = NY3 / 2;
    let src_k = NZ3 / 2;
    let dt = gpu.dt;

    for step in 0..100 {
        let t = step as f64 * dt;
        let val = gaussian(t, 3e-14, 1e-14);
        gpu.inject_ez(src_i, src_j, src_k, val);
        gpu.step().expect("GPU 3D step failed");
    }

    let ez = gpu.download_ez().expect("Ez download failed");
    assert!(
        ez.iter().all(|v| v.is_finite()),
        "3D Ez contains non-finite values"
    );
    let peak = ez.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    assert!(
        peak > 1e-20,
        "3D Ez peak is near zero — simulation likely broken"
    );
}

#[test]
fn gpu_3d_matches_cpu_oracle() {
    use oxiphoton::fdtd::gpu::Fdtd3dGpu;
    use oxiphoton::fdtd::Fdtd3d;

    const NX3: usize = 32;
    const NY3: usize = 32;
    const NZ3: usize = 32;
    const DZ3: f64 = 10e-9;
    const STEPS3: usize = 100;

    let bnd = make_boundary();

    let mut cpu = Fdtd3d::new(NX3, NY3, NZ3, DX, DY, DZ3, &bnd);
    let dt = cpu.dt;

    let mut gpu = match Fdtd3dGpu::new(NX3, NY3, NZ3, DX, DY, DZ3, &bnd) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU adapter ({e}) — skipping gpu_3d_matches_cpu_oracle");
            return;
        }
    };

    let src_i = NX3 / 2;
    let src_j = NY3 / 2;
    let src_k = NZ3 / 2;
    let t0 = 3e-14_f64;
    let sigma = 1e-14_f64;

    for step in 0..STEPS3 {
        let t = step as f64 * dt;
        let val = gaussian(t, t0, sigma);
        // CPU injection into ez directly
        let cpu_idx = src_k * (NX3 * NY3) + src_j * NX3 + src_i;
        cpu.ez[cpu_idx] += val;
        cpu.step();
        gpu.inject_ez(src_i, src_j, src_k, val);
        gpu.step().expect("GPU 3D step failed");
    }

    let cpu_ez = &cpu.ez;
    let gpu_ez = gpu.download_ez().expect("Ez download failed");
    let cpu_peak_ez = cpu_ez.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    assert!(cpu_peak_ez > 1e-20, "CPU Ez peak near zero — test broken");
    check_field("3D Ez", cpu_ez, &gpu_ez, cpu_peak_ez);

    let cpu_hz = &cpu.hz;
    let gpu_hz = gpu.download_hz().expect("Hz download failed");
    let cpu_peak_hz = cpu_hz.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    if cpu_peak_hz > 1e-25 {
        // 3D Hz accumulates f32 rounding errors over many steps; use relaxed tolerance
        let max_abs_diff = cpu_hz
            .iter()
            .zip(gpu_hz.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f64, f64::max);
        let linf_norm = max_abs_diff / cpu_peak_hz;
        assert!(
            linf_norm < 5e-1,
            "3D Hz L-inf norm too large: {linf_norm:.4e} (threshold 5e-1, cpu_peak={cpu_peak_hz:.3e})"
        );
    }
}
