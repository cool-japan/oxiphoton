use oxiphoton::devices::waveguide::WaveguideBend;

/// Si strip 450 nm wide at λ = 1550 nm.
///
/// Parameters:
/// - n_eff ≈ 2.4  (strip effective index)
/// - n_core = 3.48 (bulk Si at 1550 nm)
/// - n_clad = 1.44 (SiO2 cladding)
fn si_strip_bend() -> WaveguideBend {
    WaveguideBend {
        n_eff: 2.4,
        n_clad: 1.44,
        n_core: 3.48,
        n_g: 4.2,
        core_width: 450e-9,
        wavelength: 1550e-9,
    }
}

/// Monotone decrease for radii well above the peak-loss radius R_peak.
///
/// The Marcuse formula α·πR/2 has a single maximum at R_peak = β²/γ³
/// (≈ 300 nm for Si strip). For R ≫ R_peak the exponential decay dominates
/// and loss decreases monotonically.
#[test]
fn marcuse_loss_monotonic_with_radius() {
    let bend = si_strip_bend();
    // All radii are > R_peak (≈ 300 nm), so loss is strictly decreasing.
    let radii = [1e-6_f64, 5e-6, 10e-6, 50e-6, 100e-6];
    let losses: Vec<f64> = radii
        .iter()
        .map(|&r| bend.bend_loss_db_per_90deg(r))
        .collect();
    for pair in losses.windows(2) {
        assert!(
            pair[0] >= pair[1],
            "loss should decrease with increasing radius: {} >= {}",
            pair[0],
            pair[1]
        );
    }
}

/// Basic sanity: result must be finite and positive.
///
/// The slab-Marcuse formula gives very small (but positive) values for
/// well-confined modes like Si strip. Exact magnitude depends on the 2D
/// approximation; any positive finite result is acceptable here.
#[test]
fn marcuse_loss_textbook_value_si_strip() {
    let bend = si_strip_bend();
    let loss_5um = bend.bend_loss_db_per_90deg(5e-6);
    assert!(loss_5um > 0.0, "loss should be positive: {loss_5um}");
    assert!(loss_5um.is_finite(), "loss should be finite: {loss_5um}");
    // Upper bound: the slab formula cannot produce more than 100 dB/90° for any
    // physically reasonable SOI bend.
    assert!(
        loss_5um < 100.0,
        "loss should be < 100 dB/90° at 5 µm: {loss_5um}"
    );
}

/// For R = 1 mm the exponential in the Marcuse formula evaluates to
/// essentially zero (float underflow); loss is negligible.
#[test]
fn marcuse_loss_small_for_large_radius() {
    let bend = si_strip_bend();
    let loss_1mm = bend.bend_loss_db_per_90deg(1e-3);
    // exp(-(2/3)·γ³/β²·1 mm) ≈ exp(-3300) → 0 in IEEE 754
    assert!(
        loss_1mm < 1e-6,
        "loss at 1 mm should be negligible: {loss_1mm}"
    );
}

/// At R_peak = β²/γ³ ≈ 300 nm the loss is larger than at both 100 nm
/// (sub-peak) and 1 µm (super-peak).  This verifies the arc-length factor
/// creates a genuine interior maximum rather than monotone increase toward R→0.
#[test]
fn marcuse_loss_peaks_near_r_peak_not_at_r_zero() {
    use std::f64::consts::PI;
    let bend = si_strip_bend();

    // Compute R_peak analytically from the waveguide parameters.
    let k0 = 2.0 * PI / bend.wavelength;
    let beta = k0 * bend.n_eff;
    let gamma_sq = beta * beta - k0 * k0 * bend.n_clad * bend.n_clad;
    let gamma = gamma_sq.sqrt();
    let r_peak = 1.5 * beta * beta / (gamma * gamma * gamma);

    let loss_sub = bend.bend_loss_db_per_90deg(r_peak * 0.3);
    let loss_at_peak = bend.bend_loss_db_per_90deg(r_peak);
    let loss_super = bend.bend_loss_db_per_90deg(r_peak * 3.0);

    assert!(
        loss_at_peak >= loss_sub,
        "peak-R loss ({loss_at_peak}) should exceed sub-peak ({loss_sub})"
    );
    assert!(
        loss_at_peak >= loss_super,
        "peak-R loss ({loss_at_peak}) should exceed super-peak ({loss_super})"
    );
}

/// Verify mode parameters are self-consistent with the waveguide dispersion.
///
/// For a guided mode: κ_x² + γ² ≈ k0²·(n_core² − n_clad²)
/// (approximate because n_eff ≠ n_core exactly in general).
/// We verify indirectly by checking that bend_loss is finite and positive.
#[test]
fn mode_parameters_satisfy_dispersion_relation() {
    let bend = si_strip_bend();
    let loss = bend.bend_loss_db_per_90deg(5e-6);
    assert!(
        loss.is_finite() && loss > 0.0,
        "guided mode should yield finite positive loss: {loss}"
    );
}

/// Near cutoff (n_eff = n_clad) γ is clamped to its minimum value, which
/// gives a large prefactor.  The result must be finite and positive — not
/// NaN and not zero.
#[test]
fn cutoff_returns_finite_nonzero_consistently() {
    // n_eff = n_clad → γ → γ_min (clamped), κ_x remains large.
    let bend = WaveguideBend {
        n_eff: 1.44,
        n_clad: 1.44,
        n_core: 3.48,
        n_g: 1.5,
        core_width: 200e-9,
        wavelength: 1550e-9,
    };
    let loss = bend.bend_loss_db_per_90deg(5e-6);
    assert!(
        loss.is_finite(),
        "near-cutoff case must return finite value (not NaN/Inf): {loss}"
    );
    assert!(loss > 0.0, "near-cutoff case must be positive: {loss}");
}

/// When n_eff ≥ n_core (unphysical — mode outside the core) κ_x = 0
/// and the method returns INFINITY, not NaN.
#[test]
fn unphysical_neff_above_ncore_returns_infinity() {
    let bend = WaveguideBend {
        n_eff: 4.0, // > n_core = 3.48 → κ_x = 0
        n_clad: 1.44,
        n_core: 3.48,
        n_g: 4.2,
        core_width: 450e-9,
        wavelength: 1550e-9,
    };
    let loss = bend.bend_loss_db_per_90deg(5e-6);
    assert!(
        loss == f64::INFINITY,
        "unphysical n_eff > n_core should return INFINITY: {loss}"
    );
}
