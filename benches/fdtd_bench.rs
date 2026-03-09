use criterion::{criterion_group, criterion_main, Criterion};
use oxiphoton::fdtd::*;
use std::hint::black_box;

fn bench_1d_fdtd_1000_steps(c: &mut Criterion) {
    let nz = 500;
    let dz = 10e-9;
    let bnd = BoundaryConfig::pml(20);
    let mut solver = Fdtd1d::new(nz, dz, &bnd);
    let pulse = GaussianEnvelope::new(30.0 * solver.dt, 8.0 * solver.dt);
    solver.add_source(PlaneWaveSource::new(100, Box::new(pulse)));

    c.bench_function("1D FDTD 500 cells x 1000 steps", |b| {
        b.iter(|| {
            let mut s = Fdtd1d::new(black_box(nz), black_box(dz), &bnd);
            let pulse2 = GaussianEnvelope::new(30.0 * s.dt, 8.0 * s.dt);
            s.add_source(PlaneWaveSource::new(100, Box::new(pulse2)));
            s.run(black_box(1000));
            s
        })
    });
}

fn bench_2d_fdtd_1000_steps(c: &mut Criterion) {
    let d = 10e-9;
    let bnd = BoundaryConfig::pml(15);

    c.bench_function("2D TE FDTD 200x200 x 1000 steps", |b| {
        b.iter(|| {
            let mut solver = Fdtd2dTe::new(
                black_box(200),
                black_box(200),
                black_box(d),
                black_box(d),
                &bnd,
            );
            solver.run(black_box(1000));
            solver
        })
    });
}

criterion_group!(benches, bench_1d_fdtd_1000_steps, bench_2d_fdtd_1000_steps);
criterion_main!(benches);
