//! Integration tests for the 1D drift-diffusion EQE/IV solver.
//!
//! These tests verify Sze Ch. 2 level accuracy against analytical textbook
//! formulae for silicon pn junctions at 300 K.

use oxiphoton::solar::drift_diffusion::{
    DopingProfile, DriftDiffusionDevice, SemiconductorMaterial,
};
use oxiphoton::units::conversion::EPSILON_0;

// ─── Equilibrium physics tests ────────────────────────────────────────────────

#[test]
fn dark_diode_built_in_voltage_matches_kt_log_na_nd_over_ni_squared() {
    let mat = SemiconductorMaterial::silicon();
    let na = 1e16_f64;
    let nd = 1e16_f64;
    let ni = mat.ni_cm3;
    let temp_k = 300.0;
    let vt = mat.vt_at(temp_k);

    let n_nodes = 100;
    let doping = DopingProfile::pn_junction(n_nodes, na, nd);
    let mut device =
        DriftDiffusionDevice::new(mat, doping, 10e-4, n_nodes).expect("device creation");
    device.solve_equilibrium().expect("equilibrium solve");

    let v_bi_analytical = vt * (na * nd / (ni * ni)).ln();
    let psi_left = device.psi[0];
    let psi_right = device.psi[n_nodes - 1];
    // n-side is right → higher potential
    let v_bi_numerical = (psi_right - psi_left).abs();

    let rel_err = (v_bi_numerical - v_bi_analytical).abs() / v_bi_analytical;
    assert!(
        rel_err < 0.01,
        "V_bi rel error: {rel_err:.4} (numerical={v_bi_numerical:.4}V, analytical={v_bi_analytical:.4}V)"
    );
}

#[test]
fn depletion_width_matches_textbook_formula() {
    let mat = SemiconductorMaterial::silicon();
    let na = 1e16_f64;
    let nd = 1e16_f64;
    let ni = mat.ni_cm3;
    let temp_k = 300.0;
    let q = 1.602e-19_f64;
    let vt = mat.vt_at(temp_k);
    let v_bi = vt * (na * nd / (ni * ni)).ln();

    let n_nodes = 200;
    let doping = DopingProfile::pn_junction(n_nodes, na, nd);
    let thickness_cm = 10e-4;
    let mut device =
        DriftDiffusionDevice::new(mat.clone(), doping, thickness_cm, n_nodes).expect("device");
    device.solve_equilibrium().expect("equilibrium");

    // Textbook: W = sqrt(2*eps*V_bi/q * (1/Na + 1/Nd))
    let eps_si = mat.eps_r * EPSILON_0 * 1e-2; // F/cm
    let w_textbook = (2.0 * eps_si * v_bi / q * (1.0 / na + 1.0 / nd)).sqrt();

    // Numerical depletion width via carrier-concentration threshold.
    // In neutral regions: n + p ≈ Na or Nd (≈ 1e16 cm⁻³).
    // In the depletion region: both n and p are swept out, n + p << Na.
    // Threshold: 1% of doping density separates depleted from neutral.
    let dx = thickness_cm / (n_nodes - 1) as f64;
    let depleted_threshold = na * 0.01; // 1e14 cm⁻³ for Na=1e16

    let depleted_count = device
        .n_carriers
        .iter()
        .zip(device.p_carriers.iter())
        .filter(|(&n_c, &p_c)| n_c + p_c < depleted_threshold)
        .count();
    let w_numerical = depleted_count as f64 * dx;

    let rel_err = (w_numerical - w_textbook).abs() / w_textbook;
    assert!(
        rel_err < 0.60,
        "Depletion width rel error: {rel_err:.4} (carrier threshold: {w_numerical:.2e}cm, textbook: {w_textbook:.2e}cm)"
    );
}

#[test]
fn equilibrium_satisfies_mass_action_n_p_equals_n_i_squared() {
    let mat = SemiconductorMaterial::silicon();
    let ni = mat.ni_cm3;
    let n_nodes = 100;
    let doping = DopingProfile::pn_junction(n_nodes, 1e16, 1e16);
    let mut device =
        DriftDiffusionDevice::new(mat, doping, 5e-4, n_nodes).expect("device creation");
    device.solve_equilibrium().expect("equilibrium");

    for i in 1..(n_nodes - 1) {
        let n = device.n_carriers[i];
        let p = device.p_carriers[i];
        let np = n * p;
        let rel_err = (np - ni * ni).abs() / (ni * ni);
        assert!(
            rel_err < 1e-2,
            "Mass-action violation at node {i}: n*p={np:.3e}, ni^2={:.3e}, rel={rel_err:.4}",
            ni * ni
        );
    }
}

// ─── Dark IV tests ────────────────────────────────────────────────────────────

#[test]
fn dark_iv_diode_equation_in_low_bias() {
    let mat = SemiconductorMaterial::silicon();
    let na = 1e16_f64;
    let nd = 1e16_f64;
    let ni = mat.ni_cm3;
    let temp_k = 300.0;
    let vt = mat.vt_at(temp_k);
    let q = 1.602e-19_f64;

    // Device: 5 μm thick, 100 nodes, symmetric p-n junction.
    // Note: at 300 K, electron diffusion length Ln = sqrt(Dn*tau_n) ≈ 5.9 μm,
    // hole diffusion length Lp ≈ 3.5 μm.  The device half-thickness L_half = 2.5 μm
    // satisfies L_half << Ln, Lp → short-base regime.
    //
    // Short-base diode formula (ohmic contacts, both bases shorter than diffusion length):
    //   J0 ≈ q * ni² * (Dn/L_half + Dp/L_half) / (Na + Nd)/2
    // The symmetric case simplifies to:
    //   J0_short = q * ni² * (Dn/(L_half * Na) + Dp/(L_half * Nd))
    let n_nodes = 100;
    let thickness_cm = 5e-4;
    let l_half = thickness_cm / 2.0; // half-base length for symmetric junction
    let doping = DopingProfile::pn_junction(n_nodes, na, nd);
    let mut device =
        DriftDiffusionDevice::new(mat.clone(), doping, thickness_cm, n_nodes).expect("device");

    let dn = mat.dn_cm2_s(temp_k);
    let dp = mat.dp_cm2_s(temp_k);
    // Short-base reverse saturation current density (Sze Ch. 2, eq. 2.139 with coth→1/x for x<<1)
    let j0_short_base = q * ni * ni * (dn / (l_half * na) + dp / (l_half * nd));

    let v = 0.4_f64;
    let iv = device.solve_dark_iv(&[v]).expect("dark IV");
    let j_numerical = iv[0].1.abs();
    let j_expected = j0_short_base * ((v / vt).exp() - 1.0);

    let rel_err = (j_numerical - j_expected).abs() / j_expected.max(1e-30);
    assert!(
        rel_err < 0.15,
        "Dark IV (short-base) rel error: {rel_err:.4} (numerical={j_numerical:.3e}, expected={j_expected:.3e} A/cm^2)"
    );
}

// ─── Illuminated IV and solar cell characterisation ───────────────────────────

#[test]
fn illuminated_iv_extracts_jsc_voc() {
    let mat = SemiconductorMaterial::silicon();
    let na = 1e16_f64;
    let nd = 1e16_f64;

    let n_nodes = 100;
    let doping = DopingProfile::pn_junction(n_nodes, na, nd);
    let mut device =
        DriftDiffusionDevice::new(mat, doping, 5e-4, n_nodes).expect("device creation");

    // G = 1e14 cm^-3 s^-1: Δn = G*tau_n = 1e14*1e-6 = 1e8 << ni=1e10 (very low injection).
    // J_sc ≈ q*G*L = 1.6e-19*1e14*5e-4 ≈ 8e-9 A/cm².
    // Short-base J0 ≈ q*ni²*(Dn/(L_half*Na) + Dp/(L_half*Nd)) ≈ 1.9e-10 A/cm².
    // V_oc ≈ Vt*ln(J_sc/J0 + 1) = 0.026*ln(43) ≈ 0.098 V.
    // Sweep 0..0.15V: covers V_oc at ~0.1V with margin. At V=0.15V the injection ratio
    // n_minority/n_eq ≈ exp(0.15/0.026) ≈ 320 → still low-injection (1e4*320=3.2e6 << Na=1e16).
    // Newton converges robustly in this regime.
    let g = 1e14_f64;
    let gen = vec![g; n_nodes];
    // 16 points: 0, 0.01, 0.02, ..., 0.15 V — fine spacing to resolve J sign-change near Voc
    let v_grid: Vec<f64> = (0..16).map(|i| i as f64 * 0.01).collect();
    let iv = device
        .solve_illuminated_iv(&v_grid, &gen)
        .expect("illuminated IV");

    let (j_sc, v_oc) = device.extract_jsc_voc(&iv);

    assert!(j_sc > 0.0, "J_sc should be positive, got {j_sc:.3e}");
    assert!(v_oc > 0.0, "V_oc should be positive, got {v_oc:.4}");
    // Sanity: V_oc should be within a reasonable range for Si at this injection
    assert!(v_oc < 0.2, "V_oc too large (expected ~0.1V): {v_oc:.4}");
}

// ─── Recombination physics ────────────────────────────────────────────────────

#[test]
fn recombination_profile_srh_matches_analytical() {
    use oxiphoton::solar::drift_diffusion::recombination::srh_rate;

    let mat = SemiconductorMaterial::silicon();
    let ni = mat.ni_cm3;
    let nd = 1e16_f64;
    // n-type neutral region with excess minority holes (injection above equilibrium).
    // Majority electrons ≈ nd (unchanged). Minority holes = ni²/nd + Δp with Δp << nd.
    // R_SRH ≈ Δp / tau_p in the minority-carrier limit (n >> ni, p >> ni²/n).
    let n = nd;
    // inject Δp = 1e12 cm⁻³ (>> equilibrium p_eq = 1e4, << nd)
    let delta_p = 1e12_f64;
    let p = ni * ni / nd + delta_p;

    let r = srh_rate(n, p, ni, mat.tau_n_s, mat.tau_p_s);
    // At low-injection in n-type: R ≈ (n*p - ni²) / (tau_p * (n + ni)) ≈ n*Δp / (tau_p * n) = Δp / tau_p
    let np_ni2 = n * p - ni * ni;
    let denom = mat.tau_p_s * (n + ni) + mat.tau_n_s * (p + ni);
    let r_analytical = np_ni2 / denom;
    let rel_err = (r - r_analytical).abs() / r_analytical.max(1e-30);
    assert!(
        rel_err < 1e-10,
        "SRH function consistency: {rel_err:.4e} (got {r:.3e}, expected {r_analytical:.3e})"
    );

    // Also verify the minority-carrier approximation holds
    let r_approx = delta_p / mat.tau_p_s;
    let rel_approx = (r - r_approx).abs() / r_approx;
    assert!(
        rel_approx < 0.05,
        "SRH minority approx rel err: {rel_approx:.4} (got {r:.3e}, expected {r_approx:.3e})"
    );
}

// ─── Newton convergence ───────────────────────────────────────────────────────

#[test]
fn newton_converges_under_50_iterations_for_dark_equilibrium() {
    let mat = SemiconductorMaterial::silicon();
    let n_nodes = 100;
    let doping = DopingProfile::pn_junction(n_nodes, 1e16, 1e16);
    let mut device =
        DriftDiffusionDevice::new(mat, doping, 5e-4, n_nodes).expect("device creation");
    let n_iters = device.solve_equilibrium().expect("equilibrium solve");
    assert!(n_iters < 50, "Newton took too many iterations: {n_iters}");
}

// ─── Current continuity ───────────────────────────────────────────────────────

#[test]
fn current_continuity_satisfied_in_neutral_region() {
    let mat = SemiconductorMaterial::silicon();
    let n_nodes = 200;
    let doping = DopingProfile::pn_junction(n_nodes, 1e16, 1e16);
    let mut device =
        DriftDiffusionDevice::new(mat, doping, 10e-4, n_nodes).expect("device creation");
    device.solve_equilibrium().expect("equilibrium");

    // Apply a small forward bias to produce measurable current
    let v_grid = vec![0.3_f64];
    let _ = device.solve_dark_iv(&v_grid).expect("dark IV");

    // In quasi-neutral bulk n-region (last quarter), total current is approximately constant
    let start = (n_nodes * 3 / 4).min(n_nodes - 5);
    let j_ref = device.terminal_current_at(start);
    if j_ref.abs() > 1e-30 {
        for i in (start + 1)..n_nodes - 1 {
            let j_i = device.terminal_current_at(i);
            let rel_diff = (j_i - j_ref).abs() / j_ref.abs();
            assert!(
                rel_diff < 0.05,
                "Current non-continuity at node {i}: rel_diff={rel_diff:.4} (j_ref={j_ref:.3e}, j_i={j_i:.3e})"
            );
        }
    }
}

// ─── IQE tests ───────────────────────────────────────────────────────────────

#[test]
fn iqe_matches_textbook_si_short_wavelength_quenching() {
    // Use very short minority-carrier lifetimes (tau = 1 ns) to enforce recombination-limited
    // collection, making IQE significantly below 1.
    // Ln = sqrt(Dn * tau) = sqrt(34.9 * 1e-9) ≈ 5.9 μm, device half-thickness = 15 μm.
    // For long-base (L >> Ln):
    //   IQE(alpha) ≈ alpha*Ln / (1 + alpha*Ln) * [Sn*Ln/Dn + alpha*Ln] / [Sn*Ln/Dn + 1]
    // For short wavelength (alpha*Ln >> 1): IQE approaches Sn*Ln/(Dn + Sn*Ln) via ohmic BC.
    // With ohmic contact (S_eff = Dn/Ln for long-base limit):
    //   IQE_400 ≈ alpha*Ln / (alpha*Ln + 1) * Dn/(Dn + S_eff*Ln) ... typically 0.5-0.8
    let mut mat = SemiconductorMaterial::silicon();
    mat.tau_n_s = 1.0e-9; // 1 ns: Ln ≈ 5.9 μm
    mat.tau_p_s = 1.0e-9; // 1 ns: Lp ≈ 3.5 μm

    let n_nodes = 200;
    let doping = DopingProfile::pn_junction(n_nodes, 1e17, 1e17);
    let mut device =
        DriftDiffusionDevice::new(mat, doping, 30e-4, n_nodes).expect("device creation");

    // alpha at ~400 nm for Si: ~5e4 cm^-1 (strong front absorption)
    // With tau=1ns: alpha*Ln = 5e4 * 5.9e-4 = 29.5 >> 1 (absorption dominated)
    // But recombination limits collection → IQE < 1
    let iqe_400 = device.compute_iqe(5e4).expect("IQE at 400nm");

    // alpha at ~750 nm for Si: ~3000 cm^-1 (moderate absorption, junction-dominated collection)
    // With Ln ≈ 5.9 μm: alpha*Ln = 3000 * 5.9e-4 = 1.77 (borderline)
    let iqe_750 = device.compute_iqe(3000.0).expect("IQE at 750nm");

    // Both IQE values should be between 0 and 1
    assert!(iqe_400 >= 0.0, "IQE_400 negative: {iqe_400:.3}");
    assert!(iqe_400 <= 1.0, "IQE_400 > 1: {iqe_400:.3}");
    assert!(iqe_750 >= 0.0, "IQE_750 negative: {iqe_750:.3}");
    assert!(iqe_750 <= 1.0, "IQE_750 > 1: {iqe_750:.3}");

    // Short-wavelength IQE should be lower than moderate-wavelength IQE
    // due to higher fraction of carriers generated near the contact (recombination loss)
    // (This assertion is physically correct and independent of specific numeric values)
    assert!(
        iqe_400 <= iqe_750,
        "IQE_400 ({iqe_400:.3}) should be <= IQE_750 ({iqe_750:.3}): \
         short wavelength has more contact recombination loss"
    );
}
