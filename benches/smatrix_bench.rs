use criterion::{criterion_group, criterion_main, Criterion};
use oxiphoton::prelude::*;
use std::hint::black_box;

fn build_100_layer_stack() -> Vec<Layer> {
    let mut layers = Vec::with_capacity(100);
    let n_high = 2.45;
    let n_low = 1.46;
    let design_wl = 550e-9;
    let d_high = design_wl / (4.0 * n_high);
    let d_low = design_wl / (4.0 * n_low);

    for _ in 0..50 {
        layers.push(Layer::from_boxed(
            Box::new(ConstantMaterial::from_n("H", n_high)),
            d_high,
        ));
        layers.push(Layer::from_boxed(
            Box::new(ConstantMaterial::from_n("L", n_low)),
            d_low,
        ));
    }
    layers
}

fn bench_100_layer_1000_wavelengths(c: &mut Criterion) {
    let layers = build_100_layer_stack();
    let wavelengths: Vec<Wavelength> = (400..=1400)
        .step_by(1)
        .take(1000)
        .map(|nm| Wavelength::from_nm(nm as f64))
        .collect();

    c.bench_function("TMM 100-layer x 1000 wavelengths", |b| {
        b.iter(|| {
            TransferMatrix::spectrum(
                black_box(&layers),
                RefractiveIndex::real(1.0),
                RefractiveIndex::real(1.52),
                black_box(&wavelengths),
                Angle(0.0),
                Polarization::TE,
            )
        })
    });
}

fn bench_single_wavelength_100_layers(c: &mut Criterion) {
    let layers = build_100_layer_stack();

    c.bench_function("TMM 100-layer single wavelength", |b| {
        b.iter(|| {
            TransferMatrix::solve(
                black_box(&layers),
                RefractiveIndex::real(1.0),
                RefractiveIndex::real(1.52),
                Wavelength::from_nm(550.0),
                Angle(0.0),
                Polarization::TE,
            )
        })
    });
}

fn bench_bragg_dispersive(c: &mut Criterion) {
    let sio2 = Sellmeier::sio2();
    let tio2 = Sellmeier::tio2();
    let design_wl = 550e-9;

    let n_h = tio2.refractive_index(Wavelength(design_wl)).n;
    let n_l = sio2.refractive_index(Wavelength(design_wl)).n;
    let d_h = design_wl / (4.0 * n_h);
    let d_l = design_wl / (4.0 * n_l);

    let mut layers = Vec::new();
    for _ in 0..50 {
        layers.push(Layer::from_boxed(Box::new(tio2.clone()), d_h));
        layers.push(Layer::from_boxed(Box::new(sio2.clone()), d_l));
    }

    let wavelengths: Vec<Wavelength> = (400..=1400)
        .step_by(1)
        .take(1000)
        .map(|nm| Wavelength::from_nm(nm as f64))
        .collect();

    c.bench_function("TMM 100-layer dispersive x 1000 wl", |b| {
        b.iter(|| {
            TransferMatrix::spectrum(
                black_box(&layers),
                RefractiveIndex::real(1.0),
                RefractiveIndex::real(1.52),
                black_box(&wavelengths),
                Angle(0.0),
                Polarization::TE,
            )
        })
    });
}

criterion_group!(
    benches,
    bench_100_layer_1000_wavelengths,
    bench_single_wavelength_100_layers,
    bench_bragg_dispersive
);
criterion_main!(benches);
