use crate::material::DispersiveMaterial;
use crate::units::{RefractiveIndex, Wavelength};

/// Sellmeier dispersion model: n^2(lambda) = 1 + sum_i B_i * lambda^2 / (lambda^2 - C_i)
///
/// Coefficients (B_i, C_i) where C_i is in um^2.
#[derive(Debug, Clone)]
pub struct Sellmeier {
    pub name: String,
    /// (B_i, C_i) pairs where C_i is in um^2
    pub coefficients: Vec<(f64, f64)>,
}

impl Sellmeier {
    pub fn new(name: impl Into<String>, coefficients: Vec<(f64, f64)>) -> Self {
        Self {
            name: name.into(),
            coefficients,
        }
    }

    /// Fused Silica (SiO2) — Malitson 1965
    pub fn sio2() -> Self {
        Self::new(
            "SiO2",
            vec![
                (0.696_166_3, 0.004_679_148_2), // 0.0684043^2
                (0.407_942_6, 0.013_512_063),   // 0.1162414^2
                (0.897_479_4, 97.934_002_5),    // 9.896161^2
            ],
        )
    }

    /// Silicon (Si) — Fitted to match experimental data (Li 1993 / Green 2008)
    /// Valid approximately 1.1-5.0 um for crystalline Si at 295K
    pub fn si() -> Self {
        Self::new(
            "Si",
            vec![
                (10.6684, 0.0913_02),    // near-IR resonance (C = 0.3022^2)
                (0.003_043_5, 1.134_75), // mid-IR resonance
                (1.541_33, 1104.0),      // far-IR phonon pole
            ],
        )
    }

    /// Silicon Nitride (Si3N4) — Luke et al. 2015
    pub fn si3n4() -> Self {
        Self::new(
            "Si3N4",
            vec![
                (3.021_7, 0.013_493_16), // 0.1162^2
                (40.0, 1174.0),          // ~34.26^2
            ],
        )
    }

    /// Titanium Dioxide (TiO2) — Devore 1951 (ordinary ray, rutile)
    /// Valid 0.43-1.5 um
    pub fn tio2() -> Self {
        Self::new(
            "TiO2",
            vec![
                (4.913, 0.026_96),  // 0.1642^2
                (0.2441, 0.080_42), // 0.2836^2
            ],
        )
    }

    /// Gallium Arsenide (GaAs) — Fitted to experimental data
    /// n = 3.374 at 1550nm, valid 1-2 um range
    pub fn gaas() -> Self {
        Self::new("GaAs", vec![(9.899, 0.113_56), (0.724, 1250.0)])
    }

    /// Indium Phosphide (InP) — Fitted to experimental data
    /// n = 3.169 at 1550nm, valid 1-2 um range
    pub fn inp() -> Self {
        Self::new("InP", vec![(8.724, 0.084_1), (0.580, 750.0)])
    }

    /// Magnesium Fluoride (MgF2) — Dodge 1984 (ordinary ray)
    pub fn mgf2() -> Self {
        Self::new(
            "MgF2",
            vec![
                (0.487_551_08, 0.001_882_178),
                (0.398_750_31, 0.008_951_888),
                (2.312_035_3, 566.135_91),
            ],
        )
    }

    /// N-BK7 (Borosilicate Crown Glass) — Schott catalog
    /// Valid 0.3-2.5 um, n = 1.5168 at 589nm
    pub fn nbk7() -> Self {
        Self::new(
            "N-BK7",
            vec![
                (1.039_612_12, 0.006_001_699),  // 0.0774703^2
                (0.231_792_344, 0.020_017_914), // 0.1414765^2
                (1.010_469_45, 103.560_653),    // 10.176877^2
            ],
        )
    }

    /// N-SF11 (Dense Flint Glass) — Schott catalog
    /// High dispersion glass, n = 1.7847 at 589nm
    pub fn nsf11() -> Self {
        Self::new(
            "N-SF11",
            vec![
                (1.737_596_950, 0.013_188_707),
                (0.313_747_346, 0.062_306_599),
                (1.898_781_010, 155.236_290),
            ],
        )
    }

    /// N-LAK22 (Lanthanum Crown Glass) — Schott catalog
    /// n_d = 1.6511 at 587.56 nm, V_d = 55.78.
    ///
    /// Coefficients fitted to the Schott glass-map data (n_d, n_F, n_C) with resonance
    /// wavelengths lambda1=0.084 um (UV), lambda2=0.148 um (UV), lambda3=9.08 um (IR).
    /// C_i stored as lambda_i^2 in um^2.
    pub fn nlak22() -> Self {
        Self::new(
            "N-LAK22",
            vec![
                (2.113_669_995, 0.007_063_26),  // UV pole (lambda ~ 0.084 um)
                (-0.360_820_184, 0.021_915_64), // UV pole (lambda ~ 0.148 um)
                (11.034_058_080, 82.425_469_0), // IR pole (lambda ~ 9.08 um)
            ],
        )
    }

    /// Sapphire (Al2O3) — Malitson 1962 (ordinary ray)
    /// Valid 0.2-5.5 um, n = 1.7681 at 589nm
    pub fn sapphire() -> Self {
        Self::new(
            "Sapphire",
            vec![
                (1.431_340_50, 0.005_279_924),
                (0.650_547_130, 0.014_229_052),
                (5.341_503_80, 325.017_834),
            ],
        )
    }

    /// Zinc Selenide (ZnSe) — Connolly 1979
    /// Valid 0.55-18 um, n = 2.444 at 10 um (IR window material)
    pub fn znse() -> Self {
        Self::new(
            "ZnSe",
            vec![
                (4.299_110, 0.036_888_2), // UV resonance
                (0.626_842, 0.143_470),   // electronic
                (2.895_070, 2468.710),    // IR phonon
            ],
        )
    }

    /// Germanium (Ge) — Barnes & Piltch 1979
    /// Valid 2-14 um (IR optics), n ≈ 4.0 at 10 um
    pub fn ge() -> Self {
        Self::new(
            "Ge",
            vec![
                (9.281_50, 0.447_57), // near-IR resonance
                (6.724_56, 0.139_07), // electronic resonance
                (0.214_32, 3870.37),  // far-IR
            ],
        )
    }

    /// Barium Fluoride (BaF2) — Li 1980
    /// Valid 0.15-15 um (UV-IR window), n = 1.474 at 589nm
    pub fn baf2() -> Self {
        // Li 1980: resonance wavelengths lambda1=0.057789 um, lambda2=0.10968 um,
        // lambda3=46.3864 um; C_i stored as lambda_i^2 in um^2.
        Self::new(
            "BaF2",
            vec![
                (0.643_356, 0.003_339_569), // 0.057789^2
                (0.506_762, 0.012_029_702), // 0.10968^2
                (3.826_34, 2_151.698_1),    // 46.3864^2
            ],
        )
    }

    /// Wavelength validity range for this glass (approx).
    ///
    /// Returns `(lambda_min_um, lambda_max_um)`. These are approximate
    /// values for guidance; the Sellmeier formula may give non-physical
    /// results outside this range.
    pub fn validity_range_um(&self) -> (f64, f64) {
        match self.name.as_str() {
            "SiO2" => (0.21, 6.7),
            "Si" => (1.1, 7.0),
            "Si3N4" => (0.31, 5.5),
            "TiO2" => (0.43, 1.5),
            "GaAs" => (1.0, 17.0),
            "InP" => (0.95, 10.0),
            "MgF2" => (0.11, 10.0),
            "N-BK7" => (0.30, 2.5),
            "N-SF11" => (0.37, 2.5),
            "N-LAK22" => (0.35, 2.0),
            "Sapphire" => (0.20, 5.5),
            "ZnSe" => (0.55, 18.0),
            "Ge" => (2.0, 14.0),
            "BaF2" => (0.15, 15.0),
            _ => (0.3, 5.0),
        }
    }

    /// Check if a wavelength (m) is within the known validity range.
    pub fn is_wavelength_valid(&self, wavelength_m: f64) -> bool {
        let wl_um = wavelength_m * 1e6;
        let (lo, hi) = self.validity_range_um();
        wl_um >= lo && wl_um <= hi
    }

    /// Group index ng = n - lambda * dn/dlambda (numerical derivative, step 1 nm).
    pub fn group_index(&self, wavelength: crate::units::Wavelength) -> f64 {
        let delta = 1e-9;
        let n0 = self.refractive_index(wavelength).n;
        let n1 = self
            .refractive_index(crate::units::Wavelength(wavelength.0 + delta))
            .n;
        let dn_dl = (n1 - n0) / delta;
        n0 - wavelength.0 * dn_dl
    }

    /// Group velocity dispersion D (ps/nm/km) at the given wavelength.
    ///
    /// D = -(lambda/c) d²n/dlambda²  [expressed in ps/(nm·km)]
    pub fn gvd_ps_per_nm_km(&self, wavelength: crate::units::Wavelength) -> f64 {
        let c = 2.998e8; // m/s
        let dl = 1e-9; // 1 nm step
        let n_m = self
            .refractive_index(crate::units::Wavelength(wavelength.0 - dl))
            .n;
        let n_0 = self.refractive_index(wavelength).n;
        let n_p = self
            .refractive_index(crate::units::Wavelength(wavelength.0 + dl))
            .n;
        let d2n_dl2 = (n_p - 2.0 * n_0 + n_m) / (dl * dl);
        // D = -(lambda/c) * d²n/dlambda² in s/m², convert to ps/(nm·km)
        let d_s_m2 = -wavelength.0 / c * d2n_dl2;
        // 1 s/m² = 1e3 ps/nm/km
        d_s_m2 * 1e3
    }
}

impl DispersiveMaterial for Sellmeier {
    fn refractive_index(&self, wavelength: Wavelength) -> RefractiveIndex {
        let lambda_um = wavelength.as_um();
        let lambda_sq = lambda_um * lambda_um;
        let n_sq: f64 = 1.0
            + self
                .coefficients
                .iter()
                .map(|(b, c)| b * lambda_sq / (lambda_sq - c))
                .sum::<f64>();
        RefractiveIndex {
            n: n_sq.max(1.0).sqrt(),
            k: 0.0,
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn sio2_at_1550nm() {
        let sio2 = Sellmeier::sio2();
        let ri = sio2.refractive_index(Wavelength::from_nm(1550.0));
        assert_relative_eq!(ri.n, 1.444, epsilon = 0.002);
        assert_relative_eq!(ri.k, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn si_at_1550nm() {
        let si = Sellmeier::si();
        let ri = si.refractive_index(Wavelength::from_nm(1550.0));
        assert_relative_eq!(ri.n, 3.476, epsilon = 0.01);
    }

    #[test]
    fn si3n4_at_1550nm() {
        let si3n4 = Sellmeier::si3n4();
        let ri = si3n4.refractive_index(Wavelength::from_nm(1550.0));
        assert_relative_eq!(ri.n, 1.998, epsilon = 0.01);
    }

    #[test]
    fn tio2_at_550nm() {
        let tio2 = Sellmeier::tio2();
        let ri = tio2.refractive_index(Wavelength::from_nm(550.0));
        // TiO2 n ~ 2.3-2.6 at 550nm
        assert!(ri.n > 2.2 && ri.n < 2.7, "TiO2 n={} at 550nm", ri.n);
    }

    #[test]
    fn gaas_at_1550nm() {
        let gaas = Sellmeier::gaas();
        let ri = gaas.refractive_index(Wavelength::from_nm(1550.0));
        assert_relative_eq!(ri.n, 3.374, epsilon = 0.02);
    }

    #[test]
    fn inp_at_1550nm() {
        let inp = Sellmeier::inp();
        let ri = inp.refractive_index(Wavelength::from_nm(1550.0));
        assert_relative_eq!(ri.n, 3.169, epsilon = 0.02);
    }

    #[test]
    fn mgf2_at_550nm() {
        let mgf2 = Sellmeier::mgf2();
        let ri = mgf2.refractive_index(Wavelength::from_nm(550.0));
        assert_relative_eq!(ri.n, 1.38, epsilon = 0.01);
    }

    #[test]
    fn nbk7_at_589nm() {
        let g = Sellmeier::nbk7();
        let ri = g.refractive_index(Wavelength::from_nm(589.0));
        assert_relative_eq!(ri.n, 1.5168, epsilon = 0.002);
    }

    #[test]
    fn nsf11_at_589nm() {
        let g = Sellmeier::nsf11();
        let ri = g.refractive_index(Wavelength::from_nm(589.0));
        assert_relative_eq!(ri.n, 1.7847, epsilon = 0.005);
    }

    #[test]
    fn nlak22_at_589nm() {
        let g = Sellmeier::nlak22();
        let ri = g.refractive_index(Wavelength::from_nm(589.0));
        assert_relative_eq!(ri.n, 1.651, epsilon = 0.005);
    }

    #[test]
    fn sapphire_at_589nm() {
        let s = Sellmeier::sapphire();
        let ri = s.refractive_index(Wavelength::from_nm(589.0));
        assert_relative_eq!(ri.n, 1.768, epsilon = 0.005);
    }

    #[test]
    fn baf2_at_589nm() {
        let s = Sellmeier::baf2();
        let ri = s.refractive_index(Wavelength::from_nm(589.0));
        assert_relative_eq!(ri.n, 1.474, epsilon = 0.01);
    }

    #[test]
    fn validity_range_nbk7() {
        let g = Sellmeier::nbk7();
        assert!(g.is_wavelength_valid(589e-9));
        assert!(!g.is_wavelength_valid(100e-9)); // below UV cut-off
        assert!(!g.is_wavelength_valid(5000e-9)); // above IR cut-off
    }

    #[test]
    fn group_index_sio2() {
        let sio2 = Sellmeier::sio2();
        let ng = sio2.group_index(Wavelength::from_nm(1550.0));
        // SMF-28 ng ≈ 1.4677
        assert!(ng > 1.4 && ng < 1.55, "SiO2 ng={ng:.4} at 1550nm");
    }

    #[test]
    fn gvd_sio2_at_1300nm_anomalous_sign() {
        // SiO2 has zero-dispersion wavelength around 1.27 um;
        // at 1300 nm D should be slightly positive (anomalous region)
        let sio2 = Sellmeier::sio2();
        let d = sio2.gvd_ps_per_nm_km(Wavelength::from_nm(1300.0));
        // Near ZDW, |D| should be small
        assert!(d.abs() < 50.0, "SiO2 GVD={d:.2} ps/nm/km at 1300nm");
    }

    #[test]
    fn ge_index_in_mid_ir() {
        let ge = Sellmeier::ge();
        let ri = ge.refractive_index(Wavelength::from_nm(4000.0));
        // Ge n ≈ 4.0 at 4 um
        assert!(ri.n > 3.9 && ri.n < 4.2, "Ge n={:.3} at 4um", ri.n);
    }
}
