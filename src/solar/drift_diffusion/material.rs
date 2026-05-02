//! Semiconductor material parameters for drift-diffusion simulation.
//!
//! All parameters at 300 K unless otherwise noted. Values from Sze & Ng,
//! "Physics of Semiconductor Devices", 3rd ed. (2006), Appendices.

use crate::solar::drift_diffusion::bandgap_narrowing;
use crate::units::conversion::{BOLTZMANN, ELECTRON_CHARGE};

pub use crate::solar::drift_diffusion::bandgap_narrowing::BgnModel;

/// Carrier statistics model used by the drift-diffusion solver.
///
/// Boltzmann statistics are accurate for doping below ~1e18 cm⁻³. For
/// degenerate conditions (heavily-doped emitters or high carrier injection),
/// Fermi-Dirac statistics are required to correctly capture the modified
/// Einstein relation and the associated increase in diffusivity.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum StatisticsModel {
    /// Classical Maxwell-Boltzmann (non-degenerate) statistics.
    /// D_n = μ_n·V_T, D_p = μ_p·V_T everywhere.
    #[default]
    Boltzmann,
    /// Fermi-Dirac (degenerate) statistics with modified Einstein relation.
    /// D_n = μ_n·V_T·(2·F_{1/2}(η)/F_{-1/2}(η)) at each node.
    FermiDirac,
}

/// Boltzmann constant (J/K), re-exported for internal use.
pub const K_B: f64 = BOLTZMANN;

/// Elementary charge (C), re-exported for internal use.
pub const Q: f64 = ELECTRON_CHARGE;

/// Semiconductor material parameters for 1D drift-diffusion simulation.
///
/// All mobility and diffusion values assume Boltzmann statistics (non-degenerate
/// doping below ~1e19 cm⁻³). For degenerate doping use the Fermi-Dirac extension.
#[derive(Debug, Clone)]
pub struct SemiconductorMaterial {
    /// Band gap energy (eV) at 300 K.
    pub band_gap_ev: f64,
    /// Intrinsic carrier density (cm⁻³) at 300 K.
    pub ni_cm3: f64,
    /// Effective density of states, conduction band (cm⁻³).
    pub nc_cm3: f64,
    /// Effective density of states, valence band (cm⁻³).
    pub nv_cm3: f64,
    /// Electron mobility (cm²/(V·s)).
    pub mu_n_cm2_vs: f64,
    /// Hole mobility (cm²/(V·s)).
    pub mu_p_cm2_vs: f64,
    /// SRH electron minority-carrier lifetime (s).
    pub tau_n_s: f64,
    /// SRH hole minority-carrier lifetime (s).
    pub tau_p_s: f64,
    /// Radiative recombination coefficient B (cm³/s).
    pub b_rad_cm3_s: f64,
    /// Auger coefficient for electrons (cm⁶/s).
    pub cn_auger_cm6_s: f64,
    /// Auger coefficient for holes (cm⁶/s).
    pub cp_auger_cm6_s: f64,
    /// Relative permittivity (dimensionless).
    pub eps_r: f64,
    /// Bandgap-narrowing model to apply for this material.
    ///
    /// Defaults to `BgnModel::None` for backward-compatible behaviour.
    /// Set to `BgnModel::Slotboom` or `BgnModel::Klaassen` for heavily-doped
    /// silicon emitters where BGN is significant (N > ~1e18 cm⁻³).
    pub bgn_model: BgnModel,
    /// Carrier statistics model for this material.
    ///
    /// Defaults to `StatisticsModel::Boltzmann` for silicon. Use
    /// `StatisticsModel::FermiDirac` for GaAs or degenerate emitters where
    /// the modified Einstein relation is necessary.
    pub statistics: StatisticsModel,
}

impl SemiconductorMaterial {
    /// Thermal voltage V_T = k_B T / q (V) at temperature `temp_k` (K).
    pub fn vt_at(&self, temp_k: f64) -> f64 {
        K_B * temp_k / Q
    }

    /// Electron diffusion coefficient D_n = μ_n · V_T (cm²/s) at `temp_k` (K).
    pub fn dn_cm2_s(&self, temp_k: f64) -> f64 {
        self.mu_n_cm2_vs * self.vt_at(temp_k)
    }

    /// Hole diffusion coefficient D_p = μ_p · V_T (cm²/s) at `temp_k` (K).
    pub fn dp_cm2_s(&self, temp_k: f64) -> f64 {
        self.mu_p_cm2_vs * self.vt_at(temp_k)
    }

    /// Effective density of states in the conduction band at temperature `temp_k` (cm⁻³).
    ///
    /// Scales as (T/300)^{3/2} from the 300 K reference value.
    pub fn nc_at(&self, temp_k: f64) -> f64 {
        self.nc_cm3 * (temp_k / 300.0).powf(1.5)
    }

    /// Effective density of states in the valence band at temperature `temp_k` (cm⁻³).
    ///
    /// Scales as (T/300)^{3/2} from the 300 K reference value.
    pub fn nv_at(&self, temp_k: f64) -> f64 {
        self.nv_cm3 * (temp_k / 300.0).powf(1.5)
    }

    /// Return `true` if any node has doping that puts the Fermi level inside or
    /// above the relevant band (i.e. n > 0.5·N_c or p > 0.5·N_v at `temp_k`).
    pub fn is_degenerate_for_doping(&self, temp_k: f64, nd_max: f64, na_max: f64) -> bool {
        nd_max > 0.5 * self.nc_at(temp_k) || na_max > 0.5 * self.nv_at(temp_k)
    }

    /// Electron diffusion coefficient corrected for Fermi-Dirac statistics.
    ///
    /// Uses the modified Einstein relation (Blakemore 1982):
    ///   D_n = μ_n · V_T · (F_{1/2}(η) / F_{-1/2}(η))
    ///
    /// where η satisfies `n = N_c · (2/√π)·F_{1/2}(η)`, i.e. in the Blakemore
    /// normalisation u = n/N_c so F_{1/2}(η) = u·(√π/2).
    ///
    /// Falls back to the Boltzmann value `μ_n·V_T` for non-degenerate conditions
    /// (`n_local_cm3 < 0.1·N_c`).
    pub fn dn_cm2_s_fd(&self, temp_k: f64, n_local_cm3: f64) -> f64 {
        use super::fermi_dirac::{f_half, f_minus_half, joyce_dixon_eta};
        let vt = self.vt_at(temp_k);
        let nc = self.nc_at(temp_k);
        if n_local_cm3 < 0.1 * nc {
            // Boltzmann (non-degenerate) limit
            self.mu_n_cm2_vs * vt
        } else {
            // Blakemore u = n / N_c; then F_{1/2}(η) = u·√π/2
            let u = n_local_cm3 / nc;
            let eta = joyce_dixon_eta(u);
            let f12 = f_half(eta);
            let fm12 = f_minus_half(eta).max(1e-30);
            // Modified Einstein relation (derived from d/dη[n(η)] = N_c·(1/√π)·F_{-1/2}):
            //   D_n = μ_n·V_T·n/(dn/dη) = μ_n·V_T · 2·F_{1/2}(η)/F_{-1/2}(η)
            // Factor 2 arises because d/dη F_{1/2} = (1/2)·F_{-1/2} (integration-by-parts
            // identity for un-normalised integrals), so n/(dn/dη) = F_{1/2}/((1/2)·F_{-1/2}).
            // Non-degenerate limit: F_{1/2}→(√π/2)eη, F_{-1/2}→√π·eη, ratio→1. Correct.
            self.mu_n_cm2_vs * vt * 2.0 * f12 / fm12
        }
    }

    /// Hole diffusion coefficient corrected for Fermi-Dirac statistics.
    ///
    /// Same modified Einstein relation as `dn_cm2_s_fd` but for holes:
    ///   D_p = μ_p · V_T · (F_{1/2}(η_v) / F_{-1/2}(η_v))
    ///
    /// Falls back to the Boltzmann value `μ_p·V_T` for non-degenerate conditions
    /// (`p_local_cm3 < 0.1·N_v`).
    pub fn dp_cm2_s_fd(&self, temp_k: f64, p_local_cm3: f64) -> f64 {
        use super::fermi_dirac::{f_half, f_minus_half, joyce_dixon_eta};
        let vt = self.vt_at(temp_k);
        let nv = self.nv_at(temp_k);
        if p_local_cm3 < 0.1 * nv {
            self.mu_p_cm2_vs * vt
        } else {
            let u = p_local_cm3 / nv;
            let eta = joyce_dixon_eta(u);
            let f12 = f_half(eta);
            let fm12 = f_minus_half(eta).max(1e-30);
            // Same modified Einstein relation: D_p = μ_p·V_T · 2·F_{1/2}(η)/F_{-1/2}(η)
            self.mu_p_cm2_vs * vt * 2.0 * f12 / fm12
        }
    }

    /// Effective intrinsic carrier concentration squared (cm⁻⁶), accounting for BGN.
    ///
    /// Returns `ni_cm3²` when `bgn_model = BgnModel::None`. For Slotboom/Klaassen,
    /// returns `ni_cm3² · exp(ΔEg(nd + na) / V_T(temp_k))`. For Harmon1994,
    /// the doping type (n vs p) is inferred from `nd_cm3 >= na_cm3`.
    ///
    /// # Arguments
    /// * `temp_k`  — device temperature (K)
    /// * `nd_cm3`  — ionised donor concentration at this node (cm⁻³)
    /// * `na_cm3`  — ionised acceptor concentration at this node (cm⁻³)
    pub fn n_ie_squared(&self, temp_k: f64, nd_cm3: f64, na_cm3: f64) -> f64 {
        let ni2 = self.ni_cm3 * self.ni_cm3;
        let n_total = nd_cm3 + na_cm3;
        match self.bgn_model {
            BgnModel::None => ni2,
            BgnModel::Slotboom => {
                let de_g = bandgap_narrowing::slotboom_delta_eg_ev(n_total);
                bandgap_narrowing::ni_eff_squared_cm6(self.ni_cm3, de_g, self.vt_at(temp_k))
            }
            BgnModel::Klaassen => {
                let de_g = bandgap_narrowing::klaassen_delta_eg_ev(n_total);
                bandgap_narrowing::ni_eff_squared_cm6(self.ni_cm3, de_g, self.vt_at(temp_k))
            }
            BgnModel::Harmon1994 => {
                let is_n_type = nd_cm3 >= na_cm3;
                let de_g = bandgap_narrowing::harmon1994_gaas_delta_eg_ev(n_total, is_n_type);
                bandgap_narrowing::ni_eff_squared_cm6(self.ni_cm3, de_g, self.vt_at(temp_k))
            }
        }
    }

    /// Silicon parameters (Sze & Ng App. D, 300 K).
    ///
    /// Valid for doping up to ~1e18 cm⁻³ (Boltzmann approximation).
    ///
    /// BGN disabled by default; enable via `mat.bgn_model = BgnModel::Slotboom`
    pub fn silicon() -> Self {
        SemiconductorMaterial {
            band_gap_ev: 1.12,
            ni_cm3: 1.0e10,
            nc_cm3: 2.8e19,
            nv_cm3: 1.04e19,
            mu_n_cm2_vs: 1350.0,
            mu_p_cm2_vs: 480.0,
            tau_n_s: 1.0e-6,
            tau_p_s: 1.0e-6,
            b_rad_cm3_s: 2.0e-15,
            cn_auger_cm6_s: 2.8e-31,
            cp_auger_cm6_s: 9.9e-32,
            eps_r: 11.7,
            bgn_model: BgnModel::None,
            statistics: StatisticsModel::Boltzmann,
        }
    }

    /// GaAs parameters (Sze & Ng App. D, 300 K).
    ///
    /// Uses Fermi-Dirac statistics and the Harmon 1994 effective BGN model by default.
    /// BGN is appropriate for GaAs solar-cell emitters where heavily-doped regions
    /// (N > 1e17 cm⁻³) produce significant effective bandgap shrinkage.
    ///
    /// Source for BGN: Harmon, Melloch & Lundstrom (1994), Appl. Phys. Lett. 64(4):502.
    pub fn gaas() -> Self {
        SemiconductorMaterial {
            band_gap_ev: 1.42,
            ni_cm3: 1.79e6,
            nc_cm3: 4.7e17,
            nv_cm3: 7.0e18,
            mu_n_cm2_vs: 8500.0,
            mu_p_cm2_vs: 400.0,
            tau_n_s: 1.0e-8,
            tau_p_s: 1.0e-8,
            b_rad_cm3_s: 1.0e-10,
            cn_auger_cm6_s: 1.0e-30,
            cp_auger_cm6_s: 1.0e-30,
            eps_r: 12.9,
            bgn_model: BgnModel::Harmon1994,
            statistics: StatisticsModel::FermiDirac,
        }
    }
}
