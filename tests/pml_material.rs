/// Integration tests for the CFS-PML material model (`src/material/pml.rs`).
///
/// Covers profile monotonicity, boundary values, the Bérenger–Roden optimal
/// σ_max formula, cell-profile lengths, lossy effective permittivity, legacy
/// constructor semantics, and the `DispersiveMaterial` trait interface.
use oxiphoton::material::pml::{Pml, PmlCellProfiles};
use oxiphoton::material::DispersiveMaterial;
use oxiphoton::units::Wavelength;

// ─── Profile monotonicity ─────────────────────────────────────────────────────

#[test]
fn polynomial_grading_monotone() {
    let pml = Pml::new_optimal(1e-6, 1.0);
    let sig_max = pml.sigma_max_optimal(1e-8);

    let vals: Vec<f64> = (0..=10)
        .map(|i| pml.sigma(i as f64 / 10.0, sig_max))
        .collect();
    for w in vals.windows(2) {
        assert!(
            w[1] >= w[0],
            "σ must be non-decreasing: {:.6e} then {:.6e}",
            w[0],
            w[1]
        );
    }

    let kap_vals: Vec<f64> = (0..=10).map(|i| pml.kappa(i as f64 / 10.0)).collect();
    for w in kap_vals.windows(2) {
        assert!(
            w[1] >= w[0],
            "κ must be non-decreasing: {:.6e} then {:.6e}",
            w[0],
            w[1]
        );
    }
}

// ─── Boundary values ──────────────────────────────────────────────────────────

#[test]
fn boundary_values() {
    let pml = Pml::new_optimal(1e-6, 1.0);
    let sig_max = pml.sigma_max_optimal(1e-8);

    // σ(0) = 0
    assert!(
        pml.sigma(0.0, sig_max).abs() < 1e-30,
        "σ(0) must be zero, got {:.3e}",
        pml.sigma(0.0, sig_max)
    );
    // σ(1) = σ_max
    let rel_err = (pml.sigma(1.0, sig_max) - sig_max).abs() / sig_max;
    assert!(rel_err < 1e-12, "σ(1) relative error {rel_err:.3e}");

    // κ(0) = 1
    assert!(
        (pml.kappa(0.0) - 1.0).abs() < 1e-12,
        "κ(0) must be 1.0, got {:.6}",
        pml.kappa(0.0)
    );
    // κ(1) = κ_max
    assert!(
        (pml.kappa(1.0) - pml.kappa_max).abs() < 1e-12,
        "κ(1) must equal κ_max {:.2}, got {:.6}",
        pml.kappa_max,
        pml.kappa(1.0)
    );

    // α(0) = α_max
    let alp_max = pml.alpha_max;
    let alp_tol = 1e-30 * (alp_max.abs() + 1e-30);
    assert!(
        (pml.alpha(0.0) - alp_max).abs() < alp_tol + 1e-40,
        "α(0) must equal α_max {:.3e}, got {:.3e}",
        alp_max,
        pml.alpha(0.0)
    );
    // α(1) = 0
    assert!(
        pml.alpha(1.0).abs() < 1e-30,
        "α(1) must be 0, got {:.3e}",
        pml.alpha(1.0)
    );
}

// ─── Optimal σ_max formula ────────────────────────────────────────────────────

#[test]
fn optimal_sigma_max_matches_berenger_roden() {
    // m = 3, ε_r = 1, dx = 1 nm → σ_opt = 4 / (150π · 1e-9)
    let pml = Pml::new_optimal(1e-6, 1.0);
    let dx = 1e-9;
    let sigma_opt = pml.sigma_max_optimal(dx);
    let expected = 4.0 / (150.0 * std::f64::consts::PI * dx);
    let rel_err = (sigma_opt - expected).abs() / expected;
    assert!(
        rel_err < 0.01,
        "Relative error {rel_err:.4} exceeds 1%; got {sigma_opt:.6e}, expected {expected:.6e}"
    );
}

// ─── Cell profiles ────────────────────────────────────────────────────────────

#[test]
fn cell_profiles_length_and_continuity() {
    let pml = Pml::new_optimal(1e-6, 1.0);
    let profiles: PmlCellProfiles = pml.cell_profiles(10, 1e-8);

    assert_eq!(profiles.sigma.len(), 10, "sigma vector length");
    assert_eq!(profiles.kappa.len(), 10, "kappa vector length");
    assert_eq!(profiles.alpha.len(), 10, "alpha vector length");

    // σ increases from inner to outer cell
    assert!(
        profiles.sigma[9] > profiles.sigma[0],
        "σ must increase toward outer wall; first={:.3e} last={:.3e}",
        profiles.sigma[0],
        profiles.sigma[9]
    );
}

// ─── Complex effective permittivity ──────────────────────────────────────────

#[test]
fn complex_eps_eff_attenuates_at_normal_incidence() {
    let pml = Pml::new_optimal(1e-6, 1.0);
    let omega = 2.0 * std::f64::consts::PI * 3e8 / 1550e-9; // 1550 nm

    let eps_eff = pml.complex_eps_eff(0.5, omega, 1e-8);

    // n_eff = sqrt(eps_eff); choose branch with Im(n) ≥ 0
    let n_eff = eps_eff.sqrt();
    let n_absorbing = if n_eff.im >= 0.0 { n_eff } else { -n_eff };

    assert!(
        n_absorbing.im > 0.0,
        "PML must be absorbing: Im(n) = {:.6e}",
        n_absorbing.im
    );
}

// ─── Legacy constructor ───────────────────────────────────────────────────────

#[test]
fn legacy_constructor_zero_alpha_kappa_one() {
    let pml = Pml::new(1e-6, 1e6);

    // κ_max must be exactly 1 (no real-axis stretch in classical Bérenger PML)
    assert!(
        (pml.kappa_max - 1.0).abs() < 1e-12,
        "Classical Bérenger: kappa_max must be 1.0, got {:.6}",
        pml.kappa_max
    );
    // α_max must be exactly 0 (no CFS frequency shift)
    assert!(
        pml.alpha_max.abs() < 1e-30,
        "Classical Bérenger: alpha_max must be 0, got {:.3e}",
        pml.alpha_max
    );
    // sigma_max must be the supplied value
    let stored_sigma = pml
        .sigma_max
        .expect("sigma_max must be Some for legacy constructor");
    assert!(
        (stored_sigma - 1e6).abs() < 1e-9 * 1e6,
        "sigma_max must be 1e6, got {:.3e}",
        stored_sigma
    );
}

// ─── DispersiveMaterial trait ─────────────────────────────────────────────────

#[test]
fn dispersive_material_trait_returns_lossy_index() {
    let pml = Pml::new_optimal(1e-6, 1.0);
    let wl = Wavelength::from_nm(1550.0);
    let ri = pml.refractive_index(wl);

    // The real part of the refractive index must be positive.
    assert!(ri.n > 0.9, "Refractive index real part n = {:.6}", ri.n);

    // A CFS-PML at the midpoint must be absorbing (k > 0).
    assert!(
        ri.k > 0.0,
        "PML must be absorbing: extinction coefficient k = {:.6e}",
        ri.k
    );

    // The stub returned exactly n=1, k=0; the real implementation must differ.
    let is_stub = (ri.n - 1.0).abs() < 1e-9 && ri.k.abs() < 1e-9;
    assert!(
        !is_stub,
        "refractive_index must not return the stub value n=1, k=0"
    );
}
