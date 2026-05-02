//! Tests for Fermi-Dirac statistics wiring into the drift-diffusion solver.
//!
//! Verifies that:
//! * Silicon defaults to Boltzmann statistics (backward-compatible).
//! * GaAs defaults to FermiDirac statistics.
//! * The FD diffusivity exceeds the Boltzmann value at high carrier degeneracy,
//!   consistent with the modified Einstein relation (Blakemore 1982).

use oxiphoton::solar::drift_diffusion::{SemiconductorMaterial, StatisticsModel};

#[test]
fn boltzmann_default_unchanged() {
    // silicon() must still default to Boltzmann for backward compatibility.
    let mat = SemiconductorMaterial::silicon();
    assert_eq!(
        mat.statistics,
        StatisticsModel::Boltzmann,
        "silicon() statistics model should be Boltzmann by default"
    );
}

#[test]
fn gaas_default_uses_fermi_dirac() {
    // GaAs emitters routinely operate in the degenerate regime, so FD is the
    // physically correct default.
    let mat = SemiconductorMaterial::gaas();
    assert_eq!(
        mat.statistics,
        StatisticsModel::FermiDirac,
        "gaas() statistics model should be FermiDirac by default"
    );
}

#[test]
fn fd_diffusivity_factor_above_unity_at_degeneracy() {
    // At n = 1e20 cm⁻³ (heavily degenerate, n >> N_c = 4.7e17 for GaAs),
    // the modified Einstein relation gives D_n,FD > D_n,Boltzmann.
    //
    // Physically: in the degenerate limit the Fermi pressure term increases
    // the diffusivity as D_n → μ_n·(2/3)·E_F/q, which exceeds μ_n·k_B·T/q
    // by a factor of order (2/3)·E_F/(k_B·T) >> 1 at n=1e20.
    let mat = SemiconductorMaterial::gaas();
    let temp_k = 300.0;
    let n_deg = 1e20_f64; // highly degenerate

    let d_fd = mat.dn_cm2_s_fd(temp_k, n_deg);
    let d_bolt = mat.dn_cm2_s(temp_k);

    assert!(
        d_fd > d_bolt,
        "FD diffusivity {d_fd:.4e} cm²/s should exceed Boltzmann diffusivity {d_bolt:.4e} cm²/s at n=1e20 cm⁻³"
    );
}

#[test]
fn fd_diffusivity_recovers_boltzmann_at_low_density() {
    // At low carrier density (n << N_c), FD must converge to the Boltzmann limit.
    // GaAs N_c = 4.7e17; at n = 1e14 (intrinsic regime), u = n/N_c ≈ 2e-4 << 0.1.
    let mat = SemiconductorMaterial::gaas();
    let temp_k = 300.0;
    let n_low = 1e14_f64; // well below degeneracy

    let d_fd = mat.dn_cm2_s_fd(temp_k, n_low);
    let d_bolt = mat.dn_cm2_s(temp_k);

    // Should agree within 0.1% (the fallback threshold is n < 0.1·N_c → exact equality).
    let rel_err = (d_fd - d_bolt).abs() / d_bolt;
    assert!(
        rel_err < 1e-3,
        "FD diffusivity {d_fd:.6e} should match Boltzmann {d_bolt:.6e} at n=1e14 (rel_err={rel_err:.2e})"
    );
}

#[test]
fn statistics_model_can_be_set_on_silicon() {
    // Users should be able to override the default statistics model.
    let mut mat = SemiconductorMaterial::silicon();
    mat.statistics = StatisticsModel::FermiDirac;
    assert_eq!(mat.statistics, StatisticsModel::FermiDirac);

    mat.statistics = StatisticsModel::Boltzmann;
    assert_eq!(mat.statistics, StatisticsModel::Boltzmann);
}
