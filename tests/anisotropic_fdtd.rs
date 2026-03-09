//! Integration tests for anisotropic FDTD engine (Wave 5).
//!
//! Tests cover: birefringent waveguide simulation, uniaxial crystal permittivity
//! tensor, Faraday rotation sign, double-negative medium properties, isotropic
//! limit, fill_uniaxial_crystal helper, and Fdtd3d tensor fill.

#[cfg(feature = "fdtd")]
mod anisotropic_fdtd_tests {
    use oxiphoton::fdtd::{
        fill_uniaxial_crystal, AnisotropicFdtd3d, DoublNegativeMedium, GyroelectricMedium,
        UniaxialCrystal,
    };

    const TOL: f64 = 1e-10;

    // ── test 1: birefringent waveguide runs without panic ─────────────────────

    /// Create a 10×10×10 AnisotropicFdtd3d, fill half with a z-axis uniaxial
    /// crystal, inject a soft source, run 10 steps and verify field array sizes
    /// and finite field values.
    #[test]
    fn test_birefringent_waveguide_runs() {
        const NX: usize = 10;
        const NY: usize = 10;
        const NZ: usize = 10;
        const DX: f64 = 20e-9; // 20 nm cells
                               // Courant-stable dt for vacuum 3D: dt = dx/(sqrt(3)*c)*0.9
        const C0: f64 = 2.997_924_58e8;
        let dt = DX / (3.0_f64.sqrt() * C0) * 0.9;

        let mut fdtd = AnisotropicFdtd3d::new(NX, NY, NZ, DX, DX, DX, dt);

        // Fill lower half (k in 0..5) with uniaxial crystal, optic axis along z
        let crystal =
            UniaxialCrystal::new(1.5, 1.7, [0.0, 0.0, 1.0]).expect("valid uniaxial crystal");
        fill_uniaxial_crystal(&mut fdtd, &crystal, 0, NX, 0, NY, 0, NZ / 2);

        // Inject a soft Ez source at the centre
        fdtd.set_point_source_ez(5, 5, 5, 1.0);

        // Run 10 steps
        for _ in 0..10 {
            fdtd.step();
        }

        // Verify time step counter
        assert_eq!(fdtd.time_step, 10, "time_step should be 10 after 10 steps");

        // Verify field array sizes
        let expected_n = NX * NY * NZ;
        assert_eq!(fdtd.ex.len(), expected_n, "ex has wrong length");
        assert_eq!(fdtd.ey.len(), expected_n, "ey has wrong length");
        assert_eq!(fdtd.ez.len(), expected_n, "ez has wrong length");
        assert_eq!(fdtd.hx.len(), expected_n, "hx has wrong length");
        assert_eq!(fdtd.hy.len(), expected_n, "hy has wrong length");
        assert_eq!(fdtd.hz.len(), expected_n, "hz has wrong length");

        // No NaN or Inf in any field component
        for &v in fdtd
            .ex
            .iter()
            .chain(&fdtd.ey)
            .chain(&fdtd.ez)
            .chain(&fdtd.hx)
            .chain(&fdtd.hy)
            .chain(&fdtd.hz)
        {
            assert!(v.is_finite(), "field contains non-finite value: {v}");
        }
    }

    // ── test 2: uniaxial crystal permittivity tensor (z-axis) ─────────────────

    /// For optic axis along ẑ the tensor must satisfy:
    ///   ε_xx = ε_yy = no²,  ε_zz = ne²,  all off-diagonal = 0.
    #[test]
    fn test_uniaxial_tensor_z_axis() {
        let no = 1.5_f64;
        let ne = 1.7_f64;
        let crystal = UniaxialCrystal::new(no, ne, [0.0, 0.0, 1.0]).expect("valid z-axis crystal");

        let t = crystal.permittivity_tensor();
        let no2 = no * no;
        let ne2 = ne * ne;

        assert!(
            (t[0][0] - no2).abs() < TOL,
            "ε_xx = no² failed: got {}",
            t[0][0]
        );
        assert!(
            (t[1][1] - no2).abs() < TOL,
            "ε_yy = no² failed: got {}",
            t[1][1]
        );
        assert!(
            (t[2][2] - ne2).abs() < TOL,
            "ε_zz = ne² failed: got {}",
            t[2][2]
        );

        // Off-diagonal elements must be zero
        assert!(t[0][1].abs() < TOL, "ε_xy != 0: {}", t[0][1]);
        assert!(t[0][2].abs() < TOL, "ε_xz != 0: {}", t[0][2]);
        assert!(t[1][2].abs() < TOL, "ε_yz != 0: {}", t[1][2]);
        assert!(t[1][0].abs() < TOL, "ε_yx != 0: {}", t[1][0]);
        assert!(t[2][0].abs() < TOL, "ε_zx != 0: {}", t[2][0]);
        assert!(t[2][1].abs() < TOL, "ε_zy != 0: {}", t[2][1]);

        // Also verify diagonal_eps convenience method
        let d = crystal.diagonal_eps();
        assert!((d[0] - no2).abs() < TOL, "diagonal_eps[0] != no²");
        assert!((d[1] - no2).abs() < TOL, "diagonal_eps[1] != no²");
        assert!((d[2] - ne2).abs() < TOL, "diagonal_eps[2] != ne²");
    }

    // ── test 3: Faraday rotation direction ────────────────────────────────────

    /// eps_g > 0 should give a positive Faraday rotation rate at positive omega.
    #[test]
    fn test_faraday_rotation_sign() {
        let medium =
            GyroelectricMedium::new(2.5, 0.5, [0.0, 0.0, 1.0]).expect("valid gyroelectric medium");
        let omega = 2.0 * std::f64::consts::PI * 3.0e14; // ~infrared
        let rate = medium.faraday_rotation_rate(omega);
        assert!(
            rate > 0.0,
            "Faraday rotation rate must be positive for eps_g > 0 at positive omega: {rate}"
        );
    }

    // ── test 4: double-negative medium properties ─────────────────────────────

    /// DNG medium: n < 0, phase velocity < 0, group velocity > 0.
    #[test]
    fn test_double_negative_properties() {
        let dng = DoublNegativeMedium::new(-2.0, -1.5).expect("valid double-negative medium");

        let n = dng.refractive_index();
        assert!(n < 0.0, "DNG refractive index must be negative: {n}");

        // |n| = sqrt(|eps|·|mu|) = sqrt(2.0 × 1.5) = sqrt(3)
        let expected_abs = (2.0_f64 * 1.5_f64).sqrt();
        assert!(
            (n.abs() - expected_abs).abs() < TOL,
            "|n| should be {expected_abs:.10}, got {:.10}",
            n.abs()
        );

        // Phase velocity must be negative
        let vp = dng.phase_velocity();
        assert!(vp < 0.0, "DNG phase velocity must be negative: {vp}");

        // Group velocity must be positive (energy propagates forward)
        let omega = 2.0 * std::f64::consts::PI * 3.0e14;
        let vg = dng.group_velocity(omega, omega * 1e-6);
        assert!(vg > 0.0, "DNG group velocity must be positive: {vg}");

        // Error cases: non-negative eps_r or mu_r rejected
        assert!(
            DoublNegativeMedium::new(2.0, -1.5).is_err(),
            "Positive eps_r should produce an error"
        );
        assert!(
            DoublNegativeMedium::new(-2.0, 1.5).is_err(),
            "Positive mu_r should produce an error"
        );
    }

    // ── test 5: isotropic limit ───────────────────────────────────────────────

    /// With eps_xx = eps_yy = eps_zz = n², birefringence should be 0.
    #[test]
    fn test_isotropic_limit() {
        let n = 1.5_f64;
        // Optic axis along x with no = ne = n: purely isotropic
        let crystal = UniaxialCrystal::new(n, n, [1.0, 0.0, 0.0]).expect("valid isotropic crystal");

        assert!(
            crystal.birefringence().abs() < TOL,
            "Birefringence should be 0 for isotropic (no=ne) crystal: {}",
            crystal.birefringence()
        );

        let n2 = n * n;
        let d = crystal.diagonal_eps();
        for (i, &di) in d.iter().enumerate() {
            assert!(
                (di - n2).abs() < TOL,
                "eps diagonal[{i}] should be n²={n2} for isotropic crystal, got {di}"
            );
        }
    }

    // ── test 6: fill_uniaxial_crystal integrates correctly ────────────────────

    /// Fill a box and verify that eps_xx != eps_zz for a z-axis crystal
    /// with no != ne, while cells outside the box remain at vacuum (eps = 1).
    #[test]
    fn test_fill_uniaxial_crystal() {
        let mut fdtd = AnisotropicFdtd3d::new(20, 20, 20, 10e-9, 10e-9, 10e-9, 1e-17);

        let no = 1.5_f64;
        let ne = 1.7_f64;
        let crystal = UniaxialCrystal::new(no, ne, [0.0, 0.0, 1.0]).expect("valid crystal");

        // Fill a 5×5×5 sub-box in the interior
        fill_uniaxial_crystal(&mut fdtd, &crystal, 5, 10, 5, 10, 5, 10);

        let no2 = no * no;
        let ne2 = ne * ne;

        // Check a cell inside the filled region
        let p_in = fdtd.idx(7, 7, 7);
        assert!(
            (fdtd.eps_xx[p_in] - no2).abs() < TOL,
            "eps_xx inside box should be no²={no2}, got {}",
            fdtd.eps_xx[p_in]
        );
        assert!(
            (fdtd.eps_yy[p_in] - no2).abs() < TOL,
            "eps_yy inside box should be no²={no2}, got {}",
            fdtd.eps_yy[p_in]
        );
        assert!(
            (fdtd.eps_zz[p_in] - ne2).abs() < TOL,
            "eps_zz inside box should be ne²={ne2}, got {}",
            fdtd.eps_zz[p_in]
        );

        // eps_xx must differ from eps_zz (birefringent crystal)
        assert!(
            (fdtd.eps_xx[p_in] - fdtd.eps_zz[p_in]).abs() > 0.01,
            "eps_xx and eps_zz should differ for birefringent crystal"
        );

        // Check a cell outside the filled region: must still be vacuum
        let p_out = fdtd.idx(0, 0, 0);
        assert_eq!(
            fdtd.eps_xx[p_out], 1.0,
            "eps_xx outside box should remain 1 (vacuum)"
        );
        assert_eq!(
            fdtd.eps_yy[p_out], 1.0,
            "eps_yy outside box should remain 1 (vacuum)"
        );
        assert_eq!(
            fdtd.eps_zz[p_out], 1.0,
            "eps_zz outside box should remain 1 (vacuum)"
        );
    }

    // ── test 7: AnisotropicFdtd3d compute_coefficients and fill_eps_box ───────

    /// fill_eps_box sets eps correctly and compute_coefficients runs without panic.
    #[test]
    fn test_fdtd3d_fill_tensor_eps() {
        const DX: f64 = 50e-9;
        const C0: f64 = 2.997_924_58e8;
        let dt = DX / (3.0_f64.sqrt() * C0) * 0.9;

        let mut fdtd = AnisotropicFdtd3d::new(8, 8, 8, DX, DX, DX, dt);

        // Fill a 4×4×4 region with anisotropic eps (exx=2, eyy=3, ezz=4)
        fdtd.fill_eps_box(2, 6, 2, 6, 2, 6, 2.0, 3.0, 4.0);

        // Verify values in the filled region
        let p = fdtd.idx(4, 4, 4);
        assert!(
            (fdtd.eps_xx[p] - 2.0).abs() < TOL,
            "eps_xx should be 2.0, got {}",
            fdtd.eps_xx[p]
        );
        assert!(
            (fdtd.eps_yy[p] - 3.0).abs() < TOL,
            "eps_yy should be 3.0, got {}",
            fdtd.eps_yy[p]
        );
        assert!(
            (fdtd.eps_zz[p] - 4.0).abs() < TOL,
            "eps_zz should be 4.0, got {}",
            fdtd.eps_zz[p]
        );

        // Compute coefficients should succeed
        fdtd.compute_coefficients();

        // Run a single step — must not panic
        fdtd.step();

        assert_eq!(fdtd.time_step, 1, "time_step should be 1 after one step");
    }
}
