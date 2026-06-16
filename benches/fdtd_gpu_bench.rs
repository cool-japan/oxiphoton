use criterion::{criterion_group, criterion_main, Criterion};
use oxiphoton::fdtd::{BoundaryConfig, Fdtd2dTe};
use std::hint::black_box;

#[cfg(feature = "gpu-wgpu")]
use oxiphoton::fdtd::gpu::Fdtd2dGpu;
#[cfg(feature = "gpu-wgpu")]
use oxiphoton::fdtd::gpu::Fdtd3dGpu;

const NX: usize = 200;
const NY: usize = 200;
const STEPS: usize = 100;
const DX: f64 = 10e-9;
const DY: f64 = 10e-9;

fn bench_cpu_te_200x200(c: &mut Criterion) {
    let bnd = BoundaryConfig::pml(15);
    c.bench_function("CPU TE 200×200 × 100 steps", |b| {
        b.iter(|| {
            let mut solver = Fdtd2dTe::new(
                black_box(NX),
                black_box(NY),
                black_box(DX),
                black_box(DY),
                &bnd,
            );
            solver.run(black_box(STEPS));
            solver
        })
    });
}

#[cfg(feature = "gpu-wgpu")]
fn bench_gpu_te_200x200(c: &mut Criterion) {
    let bnd = BoundaryConfig::pml(15);
    let mut gpu = match Fdtd2dGpu::new(NX, NY, DX, DY, &bnd) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU adapter ({e}) — skipping GPU bench");
            return;
        }
    };
    c.bench_function("GPU TE 200×200 × 100 steps", |b| {
        b.iter(|| {
            gpu.run(black_box(STEPS)).expect("GPU run failed");
        })
    });
}

#[cfg(feature = "gpu-wgpu")]
fn bench_gpu_3d_32x32x32(c: &mut Criterion) {
    let bnd = BoundaryConfig::pml(6);
    let mut gpu = match Fdtd3dGpu::new(32, 32, 32, 10e-9, 10e-9, 10e-9, &bnd) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU adapter ({e}) -- skipping 3D GPU bench");
            return;
        }
    };
    c.bench_function("GPU 3D 32x32x32 x 100 steps", |b| {
        b.iter(|| {
            gpu.run(black_box(STEPS)).expect("GPU 3D run failed");
        })
    });
}

#[cfg(feature = "gpu-wgpu")]
criterion_group!(
    benches,
    bench_cpu_te_200x200,
    bench_gpu_te_200x200,
    bench_gpu_3d_32x32x32
);

#[cfg(not(feature = "gpu-wgpu"))]
criterion_group!(benches, bench_cpu_te_200x200);

criterion_main!(benches);
