//! GPU-accelerated FDTD demonstration.
//!
//! Runs a 200×200 2D TE simulation on both CPU (Fdtd2dTe) and GPU (Fdtd2dGpu),
//! then compares field values and reports timing.
//!
//! Usage:
//!   cargo run --example fdtd_gpu_acceleration --features gpu-wgpu

use std::time::Instant;

use oxiphoton::fdtd::{BoundaryConfig, Fdtd2dTe};

#[cfg(feature = "gpu-wgpu")]
use oxiphoton::fdtd::gpu::Fdtd2dGpu;
#[cfg(feature = "gpu-wgpu")]
use oxiphoton::fdtd::gpu::Fdtd3dGpu;

const NX: usize = 200;
const NY: usize = 200;
const STEPS: usize = 500;
const DX: f64 = 10e-9;
const DY: f64 = 10e-9;

fn gaussian(t: f64, t0: f64, sigma: f64) -> f64 {
    let v = (t - t0) / sigma;
    (-0.5 * v * v).exp()
}

fn main() {
    let bnd = BoundaryConfig::pml(15);
    let src_i = NX / 2;
    let src_j = NY / 2;

    // ── CPU reference ─────────────────────────────────────────────────────────
    println!("CPU: 2D TE FDTD {}×{}, {} steps", NX, NY, STEPS);
    let mut cpu = Fdtd2dTe::new(NX, NY, DX, DY, &bnd);
    cpu.fill_eps_box(NX / 4, 3 * NX / 4, NY / 4, 3 * NY / 4, 4.0);
    let dt = cpu.dt;
    let t0 = 3e-14_f64;
    let sigma = 1e-14_f64;

    let t_cpu = Instant::now();
    for step in 0..STEPS {
        let t = step as f64 * dt;
        cpu.inject_hz(src_i, src_j, gaussian(t, t0, sigma));
        cpu.step();
    }
    let cpu_elapsed = t_cpu.elapsed();
    println!("  CPU time: {:.3} s", cpu_elapsed.as_secs_f64());

    let cpu_peak: f64 = cpu.grid.hz.iter().map(|v| v.abs()).fold(0.0, f64::max);
    println!("  CPU Hz peak: {:.4e}", cpu_peak);

    // ── GPU ───────────────────────────────────────────────────────────────────
    #[cfg(feature = "gpu-wgpu")]
    {
        println!("\nGPU: 2D TE FDTD {}×{}, {} steps (wgpu)", NX, NY, STEPS);

        let mut gpu = match Fdtd2dGpu::new(NX, NY, DX, DY, &bnd) {
            Ok(g) => g,
            Err(e) => {
                println!("  No GPU adapter available ({e}) — skipping GPU run.");
                return;
            }
        };
        gpu.fill_eps_box(NX / 4, 3 * NX / 4, NY / 4, 3 * NY / 4, 4.0);

        let t_gpu = Instant::now();
        for step in 0..STEPS {
            let t = step as f64 * dt;
            gpu.inject_hz(src_i, src_j, gaussian(t, t0, sigma));
            gpu.step().expect("GPU step failed");
        }
        let gpu_elapsed = t_gpu.elapsed();
        println!("  GPU time: {:.3} s", gpu_elapsed.as_secs_f64());

        let gpu_hz = gpu.download_hz().expect("Hz download failed");
        let gpu_peak: f64 = gpu_hz.iter().map(|v| v.abs()).fold(0.0, f64::max);
        println!("  GPU Hz peak: {:.4e}", gpu_peak);

        // Agreement check
        if cpu_peak > 1e-20 {
            let max_diff = cpu
                .grid
                .hz
                .iter()
                .zip(gpu_hz.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0_f64, f64::max);
            let linf = max_diff / cpu_peak;
            println!("  GPU vs CPU L∞ (peak-normalised): {:.4e}", linf);
            if linf < 2e-3 {
                println!("  AGREEMENT: OK (within f32 tolerance)");
            } else {
                println!("  WARNING: L∞ exceeds 2e-3 — investigate numerics");
            }
        }

        let speedup = cpu_elapsed.as_secs_f64() / gpu_elapsed.as_secs_f64();
        println!("  Speedup vs CPU: {:.2}×", speedup);
    }

    // ── 3D GPU ───────────────────────────────────────────────────────────────
    #[cfg(feature = "gpu-wgpu")]
    {
        const NX3: usize = 32;
        const NY3: usize = 32;
        const NZ3: usize = 32;
        const DZ3: f64 = 10e-9;
        let steps3 = 200usize;

        println!(
            "\nGPU: 3D FDTD {}x{}x{}, {} steps (wgpu)",
            NX3, NY3, NZ3, steps3
        );

        let mut gpu3 = match Fdtd3dGpu::new(NX3, NY3, NZ3, DX, DY, DZ3, &bnd) {
            Ok(g) => g,
            Err(e) => {
                println!("  No GPU adapter available ({e}) -- skipping 3D GPU run.");
                return;
            }
        };

        let src_i3 = NX3 / 2;
        let src_j3 = NY3 / 2;
        let src_k3 = NZ3 / 2;
        let dt3 = gpu3.dt;

        let t_gpu3 = Instant::now();
        for step in 0..steps3 {
            let t = step as f64 * dt3;
            gpu3.inject_ez(src_i3, src_j3, src_k3, gaussian(t, t0, sigma));
            gpu3.step().expect("GPU 3D step failed");
        }
        let gpu3_elapsed = t_gpu3.elapsed();
        println!("  3D GPU time: {:.3} s", gpu3_elapsed.as_secs_f64());

        let gpu3_ez = gpu3.download_ez().expect("Ez download failed");
        let gpu3_peak: f64 = gpu3_ez.iter().map(|v| v.abs()).fold(0.0, f64::max);
        println!("  3D GPU Ez peak: {:.4e}", gpu3_peak);
    }

    #[cfg(not(feature = "gpu-wgpu"))]
    {
        println!("\nGPU path not compiled — rerun with --features gpu-wgpu");
    }
}
