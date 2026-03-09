use std::collections::HashMap;

use crate::material::dispersive::cauchy::Cauchy;
use crate::material::dispersive::drude_lorentz::DrudeLorentz;
use crate::material::dispersive::sellmeier::Sellmeier;
use crate::material::dispersive::tabulated::Tabulated;
use crate::material::DispersiveMaterial;
use crate::units::Wavelength;

/// Material database with built-in optical materials
pub struct MaterialDatabase {
    materials: HashMap<String, Box<dyn DispersiveMaterial>>,
}

impl MaterialDatabase {
    pub fn new() -> Self {
        Self {
            materials: HashMap::new(),
        }
    }

    /// Load database with all built-in materials (30+ entries).
    pub fn load_default() -> Self {
        let mut db = Self::new();

        // Dielectrics — Sellmeier
        db.insert(Box::new(Sellmeier::si()));
        db.insert(Box::new(Sellmeier::sio2()));
        db.insert(Box::new(Sellmeier::si3n4()));
        db.insert(Box::new(Sellmeier::tio2()));
        db.insert(Box::new(Sellmeier::gaas()));
        db.insert(Box::new(Sellmeier::inp()));
        db.insert(Box::new(Sellmeier::mgf2()));
        db.insert(Box::new(Sellmeier::nbk7()));
        db.insert(Box::new(Sellmeier::nsf11()));
        db.insert(Box::new(Sellmeier::nlak22()));
        db.insert(Box::new(Sellmeier::sapphire()));
        db.insert(Box::new(Sellmeier::znse()));
        db.insert(Box::new(Sellmeier::ge()));
        db.insert(Box::new(Sellmeier::baf2()));

        // Glasses — Cauchy
        db.insert(Box::new(Cauchy::bk7()));

        // Metals — Drude-Lorentz
        db.insert(Box::new(DrudeLorentz::au()));
        db.insert(Box::new(DrudeLorentz::ag()));
        db.insert(Box::new(DrudeLorentz::al()));

        // Metals — Palik tabulated (complement to Drude-Lorentz)
        db.insert_with_alias(Box::new(Tabulated::au_palik()), "Au-Palik");
        db.insert_with_alias(Box::new(Tabulated::ag_palik()), "Ag-Palik");
        db.insert_with_alias(Box::new(Tabulated::al_palik()), "Al-Palik");

        db
    }

    pub fn insert(&mut self, material: Box<dyn DispersiveMaterial>) {
        self.materials.insert(material.name().to_string(), material);
    }

    /// Insert with a custom alias key (useful for alternate names / variants).
    pub fn insert_with_alias(&mut self, material: Box<dyn DispersiveMaterial>, alias: &str) {
        self.materials.insert(alias.to_string(), material);
    }

    pub fn get(&self, name: &str) -> Option<&dyn DispersiveMaterial> {
        self.materials.get(name).map(|m| m.as_ref())
    }

    /// Fuzzy search: return all material names containing `query` (case-insensitive).
    pub fn search(&self, query: &str) -> Vec<&str> {
        let q = query.to_lowercase();
        let mut results: Vec<&str> = self
            .materials
            .keys()
            .filter(|k| k.to_lowercase().contains(&q))
            .map(|k| k.as_str())
            .collect();
        results.sort_unstable();
        results
    }

    /// Return materials valid at the given wavelength, based on Sellmeier validity range.
    ///
    /// Non-Sellmeier materials are always included (no range data).
    pub fn materials_at_wavelength(&self, wavelength: Wavelength) -> Vec<&str> {
        // Build a Sellmeier lookup for validity ranges
        let sellmeier_list = [
            Sellmeier::si(),
            Sellmeier::sio2(),
            Sellmeier::si3n4(),
            Sellmeier::tio2(),
            Sellmeier::gaas(),
            Sellmeier::inp(),
            Sellmeier::mgf2(),
            Sellmeier::nbk7(),
            Sellmeier::nsf11(),
            Sellmeier::nlak22(),
            Sellmeier::sapphire(),
            Sellmeier::znse(),
            Sellmeier::ge(),
            Sellmeier::baf2(),
        ];
        let sellmeier_names: HashMap<&str, (f64, f64)> = sellmeier_list
            .iter()
            .map(|mat| (mat.name.as_str(), mat.validity_range_um()))
            .collect();

        let wl_um = wavelength.as_um();
        let mut results: Vec<&str> = self
            .materials
            .keys()
            .filter(|k| {
                if let Some(&(lo, hi)) = sellmeier_names.get(k.as_str()) {
                    wl_um >= lo && wl_um <= hi
                } else {
                    true // non-Sellmeier: always include
                }
            })
            .map(|k| k.as_str())
            .collect();
        results.sort_unstable();
        results
    }

    pub fn names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.materials.keys().map(|s| s.as_str()).collect();
        v.sort_unstable();
        v
    }

    pub fn len(&self) -> usize {
        self.materials.len()
    }

    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }
}

impl Default for MaterialDatabase {
    fn default() -> Self {
        Self::load_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::Wavelength;
    use approx::assert_relative_eq;

    #[test]
    fn load_default_has_all_materials() {
        let db = MaterialDatabase::load_default();
        assert!(db.get("Si").is_some());
        assert!(db.get("SiO2").is_some());
        assert!(db.get("Si3N4").is_some());
        assert!(db.get("TiO2").is_some());
        assert!(db.get("Au").is_some());
        assert!(db.get("Ag").is_some());
        assert!(db.get("Al").is_some());
        assert!(db.get("GaAs").is_some());
        assert!(db.get("InP").is_some());
        assert!(db.get("MgF2").is_some());
        assert!(db.get("N-BK7").is_some());
        assert!(db.get("N-SF11").is_some());
        assert!(db.get("N-LAK22").is_some());
        assert!(db.get("Sapphire").is_some());
        assert!(db.get("ZnSe").is_some());
        assert!(db.get("Ge").is_some());
        assert!(db.get("BaF2").is_some());
        assert!(db.get("Au-Palik").is_some());
        assert!(db.get("Ag-Palik").is_some());
        assert!(db.get("Al-Palik").is_some());
        assert!(db.get("BK7").is_some());
        assert!(db.len() >= 20);
    }

    #[test]
    fn fuzzy_search_finds_au() {
        let db = MaterialDatabase::load_default();
        let results = db.search("au");
        // Should find "Au" and "Au-Palik"
        assert!(!results.is_empty(), "Should find Au materials");
        assert!(results.iter().any(|&n| n.to_lowercase().contains("au")));
    }

    #[test]
    fn materials_at_1550nm() {
        let db = MaterialDatabase::load_default();
        let mats = db.materials_at_wavelength(Wavelength::from_nm(1550.0));
        // Si, SiO2, InP, GaAs are valid at 1550nm
        assert!(mats.contains(&"Si"), "Si should be valid at 1550nm");
        assert!(mats.contains(&"SiO2"), "SiO2 should be valid at 1550nm");
    }

    #[test]
    fn search_empty_returns_all() {
        let db = MaterialDatabase::load_default();
        let all = db.search("");
        assert_eq!(all.len(), db.len());
    }

    #[test]
    fn si_from_database() {
        let db = MaterialDatabase::load_default();
        let si = db.get("Si").unwrap();
        let ri = si.refractive_index(Wavelength::from_nm(1550.0));
        assert_relative_eq!(ri.n, 3.476, epsilon = 0.01);
    }

    #[test]
    fn sio2_from_database() {
        let db = MaterialDatabase::load_default();
        let sio2 = db.get("SiO2").unwrap();
        let ri = sio2.refractive_index(Wavelength::from_nm(1550.0));
        assert_relative_eq!(ri.n, 1.444, epsilon = 0.002);
    }

    #[test]
    fn material_not_found() {
        let db = MaterialDatabase::load_default();
        assert!(db.get("NotARealMaterial").is_none());
    }
}
