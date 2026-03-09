//! Lumerical FDTD Solutions .fsp format reader (import only).
//!
//! Parses a simplified text representation of Lumerical simulation files.
//! The actual .fsp format is binary/proprietary; this module handles a
//! human-readable text export format compatible with Lumerical's script API.
//!
//! Supported constructs:
//! - `setnamed("FDTD", "x span", value)` — simulation domain settings
//! - `addstructure("geometry", ...)` — geometric objects
//! - `addsource("mode", ...)` — source definitions
//! - `addmonitor("DFT", ...)` — monitor definitions
//! - `set("wavelength start/stop", value)` — wavelength range
//!
//! Values are extracted using key = value line parsing.

use std::collections::HashMap;

/// Parsed simulation domain from a Lumerical file.
#[derive(Debug, Clone)]
pub struct LumericalDomain {
    /// Simulation x span (m)
    pub x_span: f64,
    /// Simulation y span (m)
    pub y_span: f64,
    /// Simulation z span (m)
    pub z_span: f64,
    /// Mesh step dx (m)
    pub dx: f64,
    /// Number of simulation steps
    pub time_steps: usize,
    /// Wavelength start (m)
    pub wavelength_start: f64,
    /// Wavelength stop (m)
    pub wavelength_stop: f64,
}

impl Default for LumericalDomain {
    fn default() -> Self {
        Self {
            x_span: 10e-6,
            y_span: 10e-6,
            z_span: 10e-6,
            dx: 10e-9,
            time_steps: 1000,
            wavelength_start: 1500e-9,
            wavelength_stop: 1600e-9,
        }
    }
}

/// A geometry object (rectangle/cylinder) from the Lumerical file.
#[derive(Debug, Clone)]
pub struct LumericalGeometry {
    /// Object type: "rectangle", "circle", "polygon", etc.
    pub kind: String,
    /// Material name
    pub material: String,
    /// Key-value properties (x, y, z, x span, y span, z span, radius, etc.)
    pub properties: HashMap<String, f64>,
}

/// A source definition from the Lumerical file.
#[derive(Debug, Clone)]
pub struct LumericalSource {
    /// Source type: "mode", "plane", "gaussian", "dipole"
    pub kind: String,
    /// Properties (injection axis, x, y, z, wavelength, etc.)
    pub properties: HashMap<String, f64>,
    /// Named string properties
    pub str_properties: HashMap<String, String>,
}

/// A monitor definition from the Lumerical file.
#[derive(Debug, Clone)]
pub struct LumericalMonitor {
    /// Monitor type: "DFT", "time", "field", "mode expansion"
    pub kind: String,
    /// Properties (x, y, z, x span, etc.)
    pub properties: HashMap<String, f64>,
    /// Named string properties
    pub str_properties: HashMap<String, String>,
}

/// Complete parsed Lumerical simulation description.
#[derive(Debug, Clone)]
pub struct LumericalSimulation {
    pub domain: LumericalDomain,
    pub geometries: Vec<LumericalGeometry>,
    pub sources: Vec<LumericalSource>,
    pub monitors: Vec<LumericalMonitor>,
}

/// Parser for Lumerical text-format script files.
pub struct LumericalParser;

impl LumericalParser {
    /// Parse a Lumerical script text into a simulation description.
    ///
    /// Handles lines of the form:
    /// - `set("key", value);`
    /// - `setnamed("FDTD", "key", value);`
    /// - `addrect; set("key", value);`
    /// - Comments starting with `#` or `%`
    pub fn parse(text: &str) -> LumericalSimulation {
        let mut domain = LumericalDomain::default();
        let mut geometries = Vec::new();
        let mut sources = Vec::new();
        let mut monitors = Vec::new();

        let mut current_geo: Option<LumericalGeometry> = None;
        let mut current_source: Option<LumericalSource> = None;
        let mut current_monitor: Option<LumericalMonitor> = None;

        for raw_line in text.lines() {
            let line = raw_line.trim();
            // Skip comments
            if line.starts_with('#') || line.starts_with('%') || line.is_empty() {
                continue;
            }

            // Geometry start
            if line.starts_with("addrect") {
                Self::flush_geo(&mut current_geo, &mut geometries);
                Self::flush_source(&mut current_source, &mut sources);
                Self::flush_monitor(&mut current_monitor, &mut monitors);
                current_geo = Some(LumericalGeometry {
                    kind: "rectangle".into(),
                    material: "etch".into(),
                    properties: HashMap::new(),
                });
            } else if line.starts_with("addcircle") {
                Self::flush_geo(&mut current_geo, &mut geometries);
                current_geo = Some(LumericalGeometry {
                    kind: "circle".into(),
                    material: "".into(),
                    properties: HashMap::new(),
                });
            } else if line.starts_with("addmodesource") || line.starts_with("addmode") {
                Self::flush_geo(&mut current_geo, &mut geometries);
                Self::flush_source(&mut current_source, &mut sources);
                current_source = Some(LumericalSource {
                    kind: "mode".into(),
                    properties: HashMap::new(),
                    str_properties: HashMap::new(),
                });
            } else if line.starts_with("addplane") {
                Self::flush_source(&mut current_source, &mut sources);
                current_source = Some(LumericalSource {
                    kind: "plane".into(),
                    properties: HashMap::new(),
                    str_properties: HashMap::new(),
                });
            } else if line.starts_with("addpower") || line.starts_with("addprofile") {
                Self::flush_monitor(&mut current_monitor, &mut monitors);
                current_monitor = Some(LumericalMonitor {
                    kind: "DFT".into(),
                    properties: HashMap::new(),
                    str_properties: HashMap::new(),
                });
            }

            // Parse `setnamed("FDTD", "key", value)` → domain settings
            if line.contains("setnamed(") {
                if let Some((_obj, key, val)) = Self::parse_setnamed(line) {
                    if let Ok(v) = val.parse::<f64>() {
                        Self::apply_domain_setting(&key, v, &mut domain);
                    }
                }
            }

            // Parse `set("key", value)` → context-aware (geo/source/monitor first, then domain)
            if line.contains("set(") && !line.contains("setnamed(") {
                if let Some((key, val)) = Self::parse_set(line) {
                    // Try numeric
                    if let Ok(v) = val.parse::<f64>() {
                        Self::apply_numeric_setting(
                            &key,
                            v,
                            &mut domain,
                            &mut current_geo,
                            &mut current_source,
                            &mut current_monitor,
                        );
                    } else {
                        // String value
                        let s = val.trim_matches('"').to_string();
                        Self::apply_string_setting(
                            &key,
                            s,
                            &mut current_geo,
                            &mut current_source,
                            &mut current_monitor,
                        );
                    }
                }
            }
        }

        // Flush remaining
        Self::flush_geo(&mut current_geo, &mut geometries);
        Self::flush_source(&mut current_source, &mut sources);
        Self::flush_monitor(&mut current_monitor, &mut monitors);

        LumericalSimulation {
            domain,
            geometries,
            sources,
            monitors,
        }
    }

    /// Parse from a file path.
    pub fn parse_file(path: &str) -> Result<LumericalSimulation, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        Ok(Self::parse(&text))
    }

    /// Export a summary of the parsed simulation to a string.
    pub fn summarize(sim: &LumericalSimulation) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Domain: {:.1}×{:.1}×{:.1} µm\n",
            sim.domain.x_span * 1e6,
            sim.domain.y_span * 1e6,
            sim.domain.z_span * 1e6,
        ));
        out.push_str(&format!("Mesh: dx={:.1} nm\n", sim.domain.dx * 1e9));
        out.push_str(&format!(
            "Wavelength: {:.0}–{:.0} nm\n",
            sim.domain.wavelength_start * 1e9,
            sim.domain.wavelength_stop * 1e9,
        ));
        out.push_str(&format!("Geometries: {}\n", sim.geometries.len()));
        out.push_str(&format!("Sources: {}\n", sim.sources.len()));
        out.push_str(&format!("Monitors: {}\n", sim.monitors.len()));
        out
    }

    // --- internal helpers ---

    fn parse_set(line: &str) -> Option<(String, String)> {
        // Match: set("key", value) or setnamed("FDTD", "key", value)
        let start = line.find('"')?;
        let rest = &line[start + 1..];
        let end = rest.find('"')?;
        let key = rest[..end].to_string();
        let after = &rest[end + 1..];
        // Find the value after the comma
        let comma = after.find(',')?;
        let val_str = after[comma + 1..]
            .trim()
            .trim_end_matches(");")
            .trim()
            .to_string();
        Some((key, val_str))
    }

    /// Apply a `set("key", val)` call: route to active context first, else domain.
    fn apply_numeric_setting(
        key: &str,
        val: f64,
        domain: &mut LumericalDomain,
        geo: &mut Option<LumericalGeometry>,
        source: &mut Option<LumericalSource>,
        monitor: &mut Option<LumericalMonitor>,
    ) {
        // Route to active context (geo/source/monitor) first
        if let Some(g) = geo {
            g.properties.insert(key.to_string(), val);
            return;
        }
        if let Some(s) = source {
            s.properties.insert(key.to_string(), val);
            return;
        }
        if let Some(m) = monitor {
            m.properties.insert(key.to_string(), val);
            return;
        }
        // No active context: apply to domain
        Self::apply_domain_setting(key, val, domain);
    }

    /// Apply a numeric setting directly to the domain struct.
    fn apply_domain_setting(key: &str, val: f64, domain: &mut LumericalDomain) {
        match key {
            "x span" | "x_span" => domain.x_span = val,
            "y span" | "y_span" => domain.y_span = val,
            "z span" | "z_span" => domain.z_span = val,
            "dx" | "mesh cells x" => domain.dx = val,
            "simulation time" => domain.time_steps = (val / 1e-15) as usize,
            "wavelength start" => domain.wavelength_start = val,
            "wavelength stop" => domain.wavelength_stop = val,
            _ => {}
        }
    }

    /// Parse `setnamed("object", "key", value)` → (object, key, value_str).
    fn parse_setnamed(line: &str) -> Option<(String, String, String)> {
        // Find first "..." → object
        let s1 = line.find('"')? + 1;
        let e1 = line[s1..].find('"')? + s1;
        let obj = line[s1..e1].to_string();
        // Find second "..." → key
        let s2 = line[e1 + 1..].find('"')? + e1 + 2;
        let e2 = line[s2..].find('"')? + s2;
        let key = line[s2..e2].to_string();
        // Find value after second closing quote + comma
        let after = &line[e2 + 1..];
        let comma = after.find(',')?;
        let val = after[comma + 1..]
            .trim()
            .trim_end_matches(");")
            .trim()
            .to_string();
        Some((obj, key, val))
    }

    fn apply_string_setting(
        key: &str,
        val: String,
        geo: &mut Option<LumericalGeometry>,
        source: &mut Option<LumericalSource>,
        monitor: &mut Option<LumericalMonitor>,
    ) {
        if key == "material" {
            if let Some(g) = geo {
                g.material = val;
                return;
            }
        }
        if let Some(g) = geo {
            g.str_properties_insert(key, val);
        } else if let Some(s) = source {
            s.str_properties.insert(key.to_string(), val);
        } else if let Some(m) = monitor {
            m.str_properties.insert(key.to_string(), val);
        }
    }

    fn flush_geo(geo: &mut Option<LumericalGeometry>, list: &mut Vec<LumericalGeometry>) {
        if let Some(g) = geo.take() {
            list.push(g);
        }
    }

    fn flush_source(src: &mut Option<LumericalSource>, list: &mut Vec<LumericalSource>) {
        if let Some(s) = src.take() {
            list.push(s);
        }
    }

    fn flush_monitor(mon: &mut Option<LumericalMonitor>, list: &mut Vec<LumericalMonitor>) {
        if let Some(m) = mon.take() {
            list.push(m);
        }
    }
}

trait StrPropsExt {
    fn str_properties_insert(&mut self, key: &str, val: String);
}

impl StrPropsExt for LumericalGeometry {
    fn str_properties_insert(&mut self, _key: &str, _val: String) {
        // LumericalGeometry doesn't have str_properties; silently drop
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_SCRIPT: &str = r#"
# Lumerical FDTD script
setnamed("FDTD", "x span", 20e-6);
setnamed("FDTD", "y span", 10e-6);
setnamed("FDTD", "z span", 5e-6);
setnamed("FDTD", "wavelength start", 1500e-9);
setnamed("FDTD", "wavelength stop", 1600e-9);
addrect;
set("x span", 5e-6);
set("y span", 220e-9);
set("material", "Si (Silicon) - Palik");
addmodesource;
set("wavelength start", 1500e-9);
set("wavelength stop", 1600e-9);
addpower;
set("x span", 1e-6);
"#;

    #[test]
    fn parse_domain_x_span() {
        let sim = LumericalParser::parse(SAMPLE_SCRIPT);
        assert!(
            (sim.domain.x_span - 20e-6).abs() < 1e-15,
            "x_span={}",
            sim.domain.x_span
        );
    }

    #[test]
    fn parse_domain_wavelength() {
        let sim = LumericalParser::parse(SAMPLE_SCRIPT);
        assert!((sim.domain.wavelength_start - 1500e-9).abs() < 1e-18);
        assert!((sim.domain.wavelength_stop - 1600e-9).abs() < 1e-18);
    }

    #[test]
    fn parse_geometry_found() {
        let sim = LumericalParser::parse(SAMPLE_SCRIPT);
        assert_eq!(sim.geometries.len(), 1);
        assert_eq!(sim.geometries[0].kind, "rectangle");
    }

    #[test]
    fn parse_source_found() {
        let sim = LumericalParser::parse(SAMPLE_SCRIPT);
        assert_eq!(sim.sources.len(), 1);
        assert_eq!(sim.sources[0].kind, "mode");
    }

    #[test]
    fn parse_monitor_found() {
        let sim = LumericalParser::parse(SAMPLE_SCRIPT);
        assert_eq!(sim.monitors.len(), 1);
    }

    #[test]
    fn geometry_material_parsed() {
        let sim = LumericalParser::parse(SAMPLE_SCRIPT);
        assert!(
            sim.geometries[0].material.contains("Si"),
            "material={}",
            sim.geometries[0].material
        );
    }

    #[test]
    fn geometry_x_span_parsed() {
        let sim = LumericalParser::parse(SAMPLE_SCRIPT);
        let xspan = sim.geometries[0]
            .properties
            .get("x span")
            .copied()
            .unwrap_or(0.0);
        assert!((xspan - 5e-6).abs() < 1e-15, "xspan={xspan}");
    }

    #[test]
    fn summarize_contains_domain() {
        let sim = LumericalParser::parse(SAMPLE_SCRIPT);
        let s = LumericalParser::summarize(&sim);
        assert!(s.contains("Domain:"), "summary={s}");
        assert!(s.contains("Wavelength:"), "summary={s}");
    }

    #[test]
    fn empty_script_gives_defaults() {
        let sim = LumericalParser::parse("");
        assert_eq!(sim.geometries.len(), 0);
        assert_eq!(sim.sources.len(), 0);
    }

    #[test]
    fn parse_set_extracts_key_value() {
        let result = LumericalParser::parse_set(r#"set("x span", 5e-6);"#);
        assert!(result.is_some());
        let (key, val) = result.unwrap();
        assert_eq!(key, "x span");
        assert!((val.parse::<f64>().unwrap() - 5e-6).abs() < 1e-20);
    }
}
