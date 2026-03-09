use criterion::{criterion_group, criterion_main, Criterion};
use oxiphoton::bpm::{FdBpm1d, FftBpm1d, FftBpm2d};
use std::hint::black_box;

fn bench_fft_bpm_1d(c: &mut Criterion) {
    let nx = 512;
    let dx = 20e-9;
    let wavelength = 1550e-9;
    let n_ref = 1.5;
    let waist = 2e-6;
    let dz = 100e-9;
    let n_steps = 200;
    let xc = (nx as f64 / 2.0) * dx;

    c.bench_function("FFT-BPM 1D 512pt x 200 steps", |b| {
        b.iter(|| {
            let mut solver = FftBpm1d::new(
                black_box(nx),
                black_box(dx),
                black_box(n_ref),
                black_box(wavelength),
            );
            solver.set_gaussian_input(1.0, xc, waist);
            for _ in 0..n_steps {
                solver.step(black_box(dz));
            }
            solver
        })
    });
}

fn bench_fft_bpm_2d(c: &mut Criterion) {
    let nx = 64;
    let ny = 64;
    let dx = 50e-9;
    let dy = 50e-9;
    let wavelength = 1550e-9;
    let n_ref = 1.5;
    let dz = 200e-9;
    let n_steps = 50;
    let xc = (nx as f64 / 2.0) * dx;
    let yc = (ny as f64 / 2.0) * dy;
    let waist = 1.5e-6;

    c.bench_function("FFT-BPM 2D 64x64 x 50 steps", |b| {
        b.iter(|| {
            let mut solver = FftBpm2d::new(
                black_box(nx),
                black_box(ny),
                black_box(dx),
                black_box(dy),
                black_box(n_ref),
                black_box(wavelength),
            );
            solver.set_gaussian_input(1.0, xc, yc, waist);
            for _ in 0..n_steps {
                solver.step(black_box(dz));
            }
            solver
        })
    });
}

fn bench_fd_bpm_1d(c: &mut Criterion) {
    let nx = 256;
    let dx = 20e-9;
    let wavelength = 1550e-9;
    let n_ref = 1.5;
    let waist = 2e-6;
    let dz = 100e-9;
    let n_steps = 200;
    let xc = (nx as f64 / 2.0) * dx;

    c.bench_function("FD-BPM 1D 256pt x 200 steps", |b| {
        b.iter(|| {
            let mut solver = FdBpm1d::new(
                black_box(nx),
                black_box(dx),
                black_box(n_ref),
                black_box(wavelength),
            );
            solver.set_gaussian_input(1.0, xc, waist);
            for _ in 0..n_steps {
                solver.step(black_box(dz));
            }
            solver
        })
    });
}

criterion_group!(benches, bench_fft_bpm_1d, bench_fft_bpm_2d, bench_fd_bpm_1d);
criterion_main!(benches);
