//! Integration tests for Bloch boundary conditions and parameter sweep (Wave 5).
//!
//! Tests cover: Gamma-point trivial phase, X-point phase, ParamSweep linspace,
//! ParamSweep run, ParamGrid 2D product, WavelengthSweep conversion, frequencies
//! order, ConvergenceSweep constant, BandStructureCalc path endpoints, and
//! find_resonances peak detection.

#[cfg(feature = "fdtd")]
mod bloch_bc_tests {
    use oxiphoton::fdtd::{
        BandStructureCalc, BlochBc3d, ConvergenceSweep, ParamGrid, ParamSweep, WavelengthSweep,
    };
    use std::f64::consts::PI;

    const TOL: f64 = 1e-10;

    // ── test 1: Gamma point phase factors are 1+0i ────────────────────────────

    /// At k = (0,0,0) all phase factors must equal 1 + 0i.
    #[test]
    fn test_gamma_point_trivial() {
        let bc = BlochBc3d::new(0.0, 0.0, 0.0, 16, 16, 16, 10e-9, 10e-9, 10e-9);

        let pfx = bc.phase_factor_x();
        let pfy = bc.phase_factor_y();
        let pfz = bc.phase_factor_z();

        assert!(
            (pfx.re - 1.0).abs() < TOL,
            "Gamma pfx.re should be 1, got {}",
            pfx.re
        );
        assert!(
            pfx.im.abs() < TOL,
            "Gamma pfx.im should be 0, got {}",
            pfx.im
        );

        assert!(
            (pfy.re - 1.0).abs() < TOL,
            "Gamma pfy.re should be 1, got {}",
            pfy.re
        );
        assert!(
            pfy.im.abs() < TOL,
            "Gamma pfy.im should be 0, got {}",
            pfy.im
        );

        assert!(
            (pfz.re - 1.0).abs() < TOL,
            "Gamma pfz.re should be 1, got {}",
            pfz.re
        );
        assert!(
            pfz.im.abs() < TOL,
            "Gamma pfz.im should be 0, got {}",
            pfz.im
        );
    }

    // ── test 2: X point for square lattice ───────────────────────────────────

    /// At the X point (kx = π/a, ky = kz = 0) for a square lattice with
    /// period a, the x phase factor must be exp(iπ) = −1 + 0i.
    #[test]
    fn test_x_point_phase() {
        let nx = 32usize;
        let dx = 10e-9_f64;
        let a = nx as f64 * dx; // lattice constant = full domain length
        let kx = PI / a; // X point: kx · Lx = π

        let bc = BlochBc3d::new(kx, 0.0, 0.0, nx, 32, 32, dx, dx, dx);
        let pfx = bc.phase_factor_x();

        // exp(i·π) = -1 + 0i
        assert!(
            (pfx.re - (-1.0)).abs() < TOL,
            "X-point pfx.re should be -1, got {}",
            pfx.re
        );
        assert!(
            pfx.im.abs() < TOL,
            "X-point pfx.im should be 0, got {}",
            pfx.im
        );

        // y and z factors must be 1 (ky = kz = 0)
        let pfy = bc.phase_factor_y();
        let pfz = bc.phase_factor_z();
        assert!(
            (pfy.re - 1.0).abs() < TOL,
            "X-point pfy.re should be 1, got {}",
            pfy.re
        );
        assert!(
            (pfz.re - 1.0).abs() < TOL,
            "X-point pfz.re should be 1, got {}",
            pfz.re
        );
    }

    // ── test 3: ParamSweep linspace values ───────────────────────────────────

    /// linspace("x", 0, 4, 5) should produce [0, 1, 2, 3, 4].
    #[test]
    fn test_param_sweep_linspace_values() {
        let sweep = ParamSweep::linspace("wavelength_nm", 1000.0, 1600.0, 7);
        assert_eq!(sweep.values.len(), 7, "linspace should produce 7 points");
        assert!(
            (sweep.values[0] - 1000.0).abs() < TOL,
            "first value should be 1000"
        );
        assert!(
            (sweep.values[6] - 1600.0).abs() < TOL,
            "last value should be 1600"
        );

        // Step should be 100 nm
        let expected_step = 100.0;
        assert!(
            (sweep.values[1] - sweep.values[0] - expected_step).abs() < TOL,
            "step should be {expected_step} nm"
        );
    }

    // ── test 4: ParamSweep run squares each value ─────────────────────────────

    /// ParamSweep::run with f(x) = x² should yield [1, 4, 9, 16, 25].
    #[test]
    fn test_param_sweep_run_squares() {
        let sweep = ParamSweep::linspace("x", 1.0, 5.0, 5);
        let results: Vec<f64> = sweep.run(|x| x * x);
        let expected = [1.0_f64, 4.0, 9.0, 16.0, 25.0];
        assert_eq!(results.len(), expected.len());
        for (i, (&r, &e)) in results.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < TOL, "sweep.run[{i}]: got {r}, expected {e}");
        }
    }

    // ── test 5: ParamGrid 2×2 product ────────────────────────────────────────

    /// ParamGrid with 2 values each should produce a 2×2 result matrix.
    #[test]
    fn test_param_grid_2x2() {
        let grid = ParamGrid::new("a", vec![2.0, 3.0], "b", vec![10.0, 100.0]);
        let results = grid.run(|a, b| a * b);

        assert_eq!(results.len(), 2, "outer dimension should be 2");
        assert_eq!(results[0].len(), 2, "inner dimension should be 2");

        // results[0][0] = 2 * 10 = 20
        assert!(
            (results[0][0] - 20.0).abs() < TOL,
            "grid[0][0] should be 20"
        );
        // results[0][1] = 2 * 100 = 200
        assert!(
            (results[0][1] - 200.0).abs() < TOL,
            "grid[0][1] should be 200"
        );
        // results[1][0] = 3 * 10 = 30
        assert!(
            (results[1][0] - 30.0).abs() < TOL,
            "grid[1][0] should be 30"
        );
        // results[1][1] = 3 * 100 = 300
        assert!(
            (results[1][1] - 300.0).abs() < TOL,
            "grid[1][1] should be 300"
        );
    }

    // ── test 6: WavelengthSweep converts nm to m correctly ───────────────────

    /// WavelengthSweep stores λ in metres; wavelengths_nm() should invert that.
    #[test]
    fn test_wavelength_sweep_conversion() {
        let sweep = WavelengthSweep::new(1000.0, 1600.0, 61);

        // Internal storage in metres
        assert!(
            (sweep.lambda_min_m - 1e-6).abs() < 1e-15,
            "lambda_min_m should be 1e-6, got {}",
            sweep.lambda_min_m
        );
        assert!(
            (sweep.lambda_max_m - 1.6e-6).abs() < 1e-15,
            "lambda_max_m should be 1.6e-6, got {}",
            sweep.lambda_max_m
        );

        // Round-trip via wavelengths_nm()
        let nm = sweep.wavelengths_nm();
        assert_eq!(nm.len(), 61);
        assert!(
            (nm[0] - 1000.0).abs() < 1e-6,
            "first nm should be 1000, got {}",
            nm[0]
        );
        assert!(
            (nm[60] - 1600.0).abs() < 1e-6,
            "last nm should be 1600, got {}",
            nm[60]
        );
    }

    // ── test 7: WavelengthSweep frequencies in decreasing order ─────────────

    /// Since freq = c/λ and λ increases, frequencies must be strictly decreasing.
    #[test]
    fn test_wavelength_sweep_frequencies_decreasing() {
        let sweep = WavelengthSweep::new(1000.0, 1600.0, 21);
        let freqs = sweep.frequencies_hz();
        assert_eq!(freqs.len(), 21);

        // Frequencies must be strictly decreasing (longer λ → lower f)
        for i in 1..freqs.len() {
            assert!(
                freqs[i] < freqs[i - 1],
                "frequencies should be decreasing: freqs[{i}]={} >= freqs[{}]={}",
                freqs[i],
                i - 1,
                freqs[i - 1]
            );
        }

        // First frequency should correspond to λ_min = 1000 nm
        const C: f64 = 2.997_924_58e8;
        let expected_f_first = C / 1000e-9;
        assert!(
            (freqs[0] - expected_f_first).abs() / expected_f_first < 1e-6,
            "first frequency should be c/1000nm, got {}",
            freqs[0]
        );
    }

    // ── test 8: ConvergenceSweep converges on constant function ──────────────

    /// For a constant function f(x) = 42.0, any relative change is 0 → instant convergence.
    #[test]
    fn test_convergence_sweep_constant() {
        let sweep = ConvergenceSweep::new("n_cells", 16.0, 10, 1e-6);
        let result = sweep.run(|_n| 42.0_f64);
        assert!(
            result.is_ok(),
            "ConvergenceSweep on constant function should converge immediately: {:?}",
            result
        );
        let (_param, val) = result.expect("convergence success");
        assert!(
            (val - 42.0).abs() < 1e-12,
            "converged value should be 42.0, got {val}"
        );
    }

    // ── test 9: BandStructureCalc square path has correct endpoints ───────────

    /// The Γ–X–M–Γ path must start and end at Γ = (0,0,0) and contain the
    /// X and M high-symmetry points.
    #[test]
    fn test_band_structure_path_endpoints() {
        let a = 500e-9_f64; // lattice constant 500 nm
        let n_k = 30;
        let path = BandStructureCalc::square_lattice_path(n_k, a);

        assert!(!path.is_empty(), "path should not be empty");

        // First point: Γ = (0, 0, 0)
        let first = path[0];
        assert!(
            first[0].abs() < TOL,
            "path starts at Γ: kx should be 0, got {}",
            first[0]
        );
        assert!(
            first[1].abs() < TOL,
            "path starts at Γ: ky should be 0, got {}",
            first[1]
        );
        assert!(
            first[2].abs() < TOL,
            "path starts at Γ: kz should be 0, got {}",
            first[2]
        );

        // Last point: Γ = (0, 0, 0)  (M → Γ segment)
        let last = *path.last().expect("path has at least one point");
        assert!(
            last[0].abs() < TOL,
            "path ends at Γ: kx should be 0, got {}",
            last[0]
        );
        assert!(
            last[1].abs() < TOL,
            "path ends at Γ: ky should be 0, got {}",
            last[1]
        );

        // X point (π/a, 0, 0) must appear in the path
        let tol_kpt = 1e-5 * PI / a;
        let has_x = path
            .iter()
            .any(|&[kx, ky, _]| (kx - PI / a).abs() < tol_kpt && ky.abs() < tol_kpt);
        assert!(has_x, "X point (π/a, 0, 0) must appear in the Γ–X–M–Γ path");

        // M point (π/a, π/a, 0) must appear in the path
        let has_m = path
            .iter()
            .any(|&[kx, ky, _]| (kx - PI / a).abs() < tol_kpt && (ky - PI / a).abs() < tol_kpt);
        assert!(
            has_m,
            "M point (π/a, π/a, 0) must appear in the Γ–X–M–Γ path"
        );
    }

    // ── test 10: find_resonances detects known frequency ─────────────────────

    /// A pure sinusoidal signal at a known frequency must produce a resonance
    /// peak close to that frequency (within 5× the frequency bin width).
    #[test]
    fn test_find_resonances_known_frequency() {
        let dt = 1e-15_f64; // 1 fs time step
        let f_target = 3.0e14_f64; // 300 THz

        // Generate N cycles of the sine
        let n_samples = 4096;
        let signal: Vec<f64> = (0..n_samples)
            .map(|n| (2.0 * PI * f_target * n as f64 * dt).sin())
            .collect();

        let freq_range = (1.0e14, 6.0e14);
        let n_freqs = 512;
        let peaks = BandStructureCalc::find_resonances(&signal, dt, freq_range, n_freqs);

        assert!(
            !peaks.is_empty(),
            "find_resonances must detect at least one peak in a sine signal"
        );

        let df = (freq_range.1 - freq_range.0) / (n_freqs - 1) as f64;
        let closest = peaks
            .iter()
            .map(|&f| (f - f_target).abs())
            .fold(f64::INFINITY, f64::min);

        assert!(
            closest < 5.0 * df,
            "closest resonance is {closest:.3e} Hz from target {f_target:.3e} Hz \
             (threshold = {} × df = {:.3e} Hz)",
            5,
            5.0 * df
        );
    }
}
