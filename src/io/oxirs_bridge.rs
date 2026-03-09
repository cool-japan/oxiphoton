//! OxiRS knowledge graph bridge for photonic simulation data.
//!
//! Exports simulation results as RDF-compatible triples for integration with
//! the oxirs knowledge graph. The output format uses a simple N-Triples-like
//! text representation:
//!
//!   `<subject> <predicate> <object> .`
//!
//! Subjects are simulation entities (waveguide, source, monitor, result).
//! Predicates are photonic properties (wavelength, n_eff, transmission, etc.).
//! Objects are typed literals (numeric, string, unit-annotated).
//!
//! This module provides stub structures for future full oxirs integration.

use std::fmt;

/// An RDF-like triple: subject → predicate → object.
#[derive(Debug, Clone, PartialEq)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: RdfObject,
}

/// RDF object: literal (numeric or string) or URI reference.
#[derive(Debug, Clone, PartialEq)]
pub enum RdfObject {
    /// Numeric value with SI unit annotation
    Numeric { value: f64, unit: String },
    /// Plain string literal
    Literal(String),
    /// URI reference (another entity)
    Uri(String),
}

impl fmt::Display for RdfObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RdfObject::Numeric { value, unit } => write!(f, "\"{value}\"^^<{unit}>"),
            RdfObject::Literal(s) => write!(f, "\"{s}\""),
            RdfObject::Uri(u) => write!(f, "<{u}>"),
        }
    }
}

impl Triple {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: RdfObject,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object,
        }
    }

    /// Format as N-Triple line.
    pub fn to_n_triple(&self) -> String {
        format!("<{}> <{}> {} .", self.subject, self.predicate, self.object)
    }
}

/// A collection of triples representing a simulation knowledge graph.
#[derive(Debug, Clone, Default)]
pub struct KnowledgeGraph {
    pub triples: Vec<Triple>,
    /// Base URI prefix for entity identifiers
    pub base_uri: String,
}

impl KnowledgeGraph {
    /// Create a new graph with a base URI.
    pub fn new(base_uri: impl Into<String>) -> Self {
        Self {
            triples: Vec::new(),
            base_uri: base_uri.into(),
        }
    }

    /// Add a triple to the graph.
    pub fn add(&mut self, triple: Triple) {
        self.triples.push(triple);
    }

    /// Add a numeric property triple.
    pub fn add_numeric(&mut self, subject: &str, predicate: &str, value: f64, unit: &str) {
        self.add(Triple::new(
            format!("{}{}", self.base_uri, subject),
            format!("https://oxirs.io/photonics#{predicate}"),
            RdfObject::Numeric {
                value,
                unit: unit.to_string(),
            },
        ));
    }

    /// Add a string property triple.
    pub fn add_literal(&mut self, subject: &str, predicate: &str, value: impl Into<String>) {
        self.add(Triple::new(
            format!("{}{}", self.base_uri, subject),
            format!("https://oxirs.io/photonics#{predicate}"),
            RdfObject::Literal(value.into()),
        ));
    }

    /// Add a relation between two entities.
    pub fn add_relation(&mut self, subject: &str, predicate: &str, object: &str) {
        self.add(Triple::new(
            format!("{}{}", self.base_uri, subject),
            format!("https://oxirs.io/photonics#{predicate}"),
            RdfObject::Uri(format!("{}{}", self.base_uri, object)),
        ));
    }

    /// Export graph as N-Triples text.
    pub fn to_n_triples(&self) -> String {
        self.triples
            .iter()
            .map(|t| t.to_n_triple() + "\n")
            .collect()
    }

    /// Number of triples.
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Query triples by predicate fragment.
    pub fn query_predicate(&self, predicate_fragment: &str) -> Vec<&Triple> {
        self.triples
            .iter()
            .filter(|t| t.predicate.contains(predicate_fragment))
            .collect()
    }

    /// Query triples by subject fragment.
    pub fn query_subject(&self, subject_fragment: &str) -> Vec<&Triple> {
        self.triples
            .iter()
            .filter(|t| t.subject.contains(subject_fragment))
            .collect()
    }
}

/// Builder for photonic simulation knowledge graph entries.
pub struct PhotonicSimExporter {
    graph: KnowledgeGraph,
    sim_id: String,
}

impl PhotonicSimExporter {
    /// Create exporter with simulation identifier.
    pub fn new(sim_id: impl Into<String>) -> Self {
        let id: String = sim_id.into();
        Self {
            graph: KnowledgeGraph::new("https://oxirs.io/sim/"),
            sim_id: id,
        }
    }

    /// Record waveguide properties.
    pub fn add_waveguide(&mut self, name: &str, n_eff: f64, wavelength_m: f64) {
        let entity = format!("{}/{}", self.sim_id, name);
        self.graph.add_literal(&entity, "type", "Waveguide");
        self.graph
            .add_numeric(&entity, "effectiveIndex", n_eff, "dimensionless");
        self.graph
            .add_numeric(&entity, "wavelength", wavelength_m, "m");
        self.graph.add_relation(&entity, "belongsTo", &self.sim_id);
    }

    /// Record transmission spectrum result.
    pub fn add_transmission(&mut self, monitor_name: &str, wavelength_m: f64, transmission: f64) {
        let entity = format!(
            "{}/{}/{:.0}nm",
            self.sim_id,
            monitor_name,
            wavelength_m * 1e9
        );
        self.graph
            .add_literal(&entity, "type", "TransmissionResult");
        self.graph
            .add_numeric(&entity, "wavelength", wavelength_m, "m");
        self.graph
            .add_numeric(&entity, "transmission", transmission, "dimensionless");
    }

    /// Record resonator characteristics.
    pub fn add_resonator(&mut self, name: &str, q_factor: f64, fsr_m: f64, wavelength_m: f64) {
        let entity = format!("{}/{}", self.sim_id, name);
        self.graph.add_literal(&entity, "type", "Resonator");
        self.graph
            .add_numeric(&entity, "qualityFactor", q_factor, "dimensionless");
        self.graph
            .add_numeric(&entity, "freeSpectralRange", fsr_m, "m");
        self.graph
            .add_numeric(&entity, "resonanceWavelength", wavelength_m, "m");
    }

    /// Record material.
    pub fn add_material(&mut self, name: &str, n_real: f64, n_imag: f64, wavelength_m: f64) {
        let entity = format!("material/{name}");
        self.graph.add_literal(&entity, "type", "Material");
        self.graph
            .add_numeric(&entity, "refractiveIndexReal", n_real, "dimensionless");
        self.graph
            .add_numeric(&entity, "refractiveIndexImag", n_imag, "dimensionless");
        self.graph
            .add_numeric(&entity, "wavelength", wavelength_m, "m");
    }

    /// Get the completed knowledge graph.
    pub fn into_graph(self) -> KnowledgeGraph {
        self.graph
    }

    /// Export N-Triples text.
    pub fn export(&self) -> String {
        self.graph.to_n_triples()
    }
}

/// oxirs connection stub (placeholder for future real oxirs API integration).
#[derive(Debug, Clone)]
pub struct OxirsConnection {
    /// Remote endpoint URL
    pub endpoint: String,
    /// Authentication token (placeholder)
    pub token: Option<String>,
}

impl OxirsConnection {
    /// Create connection to a SPARQL-compatible endpoint.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            token: None,
        }
    }

    /// Set authentication token.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Simulate uploading triples (stub — returns byte count).
    pub fn upload_graph(&self, graph: &KnowledgeGraph) -> Result<usize, String> {
        let data = graph.to_n_triples();
        // In a real implementation, this would POST to `self.endpoint`
        // For now, return the number of bytes that would be sent
        Ok(data.len())
    }

    /// Simulate a SPARQL SELECT query (stub — returns empty results).
    pub fn query(&self, _sparql: &str) -> Result<Vec<Vec<String>>, String> {
        // Stub: real implementation would HTTP GET to endpoint
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triple_n_triple_format() {
        let t = Triple::new(
            "sim/001",
            "photon:wavelength",
            RdfObject::Numeric {
                value: 1550e-9,
                unit: "m".into(),
            },
        );
        let s = t.to_n_triple();
        assert!(s.starts_with("<sim/001>"), "got: {s}");
        assert!(s.ends_with(" ."), "got: {s}");
    }

    #[test]
    fn graph_add_and_count() {
        let mut g = KnowledgeGraph::new("https://oxirs.io/");
        g.add_numeric("wg1", "n_eff", 2.5, "dimensionless");
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn graph_to_n_triples() {
        let mut g = KnowledgeGraph::new("https://oxirs.io/");
        g.add_literal("device1", "type", "Waveguide");
        let text = g.to_n_triples();
        assert!(text.contains("Waveguide"), "text={text}");
        assert!(text.ends_with(".\n"), "text={text}");
    }

    #[test]
    fn query_by_predicate() {
        let mut g = KnowledgeGraph::new("https://oxirs.io/");
        g.add_numeric("wg1", "wavelength", 1550e-9, "m");
        g.add_numeric("wg1", "n_eff", 2.5, "dimensionless");
        let results = g.query_predicate("wavelength");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn query_by_subject() {
        let mut g = KnowledgeGraph::new("https://oxirs.io/");
        g.add_literal("wg1", "type", "Waveguide");
        g.add_literal("mon1", "type", "Monitor");
        let results = g.query_subject("wg1");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn exporter_adds_waveguide() {
        let mut exp = PhotonicSimExporter::new("sim001");
        exp.add_waveguide("wg_te0", 2.5, 1550e-9);
        let g = exp.into_graph();
        assert!(!g.is_empty());
        let wg = g.query_predicate("effectiveIndex");
        assert!(!wg.is_empty());
    }

    #[test]
    fn exporter_adds_resonator() {
        let mut exp = PhotonicSimExporter::new("sim002");
        exp.add_resonator("ring1", 10000.0, 8e-9, 1550e-9);
        let text = exp.export();
        assert!(text.contains("qualityFactor"), "text={text}");
    }

    #[test]
    fn exporter_transmission() {
        let mut exp = PhotonicSimExporter::new("sim003");
        exp.add_transmission("T_port", 1550e-9, 0.95);
        let g = exp.into_graph();
        let t = g.query_predicate("transmission");
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn connection_upload_stub() {
        let mut g = KnowledgeGraph::new("https://oxirs.io/");
        g.add_literal("dev", "type", "Test");
        let conn = OxirsConnection::new("https://oxirs.io/sparql");
        let bytes = conn.upload_graph(&g);
        assert!(bytes.is_ok());
        assert!(bytes.unwrap() > 0);
    }

    #[test]
    fn rdf_object_display_numeric() {
        let obj = RdfObject::Numeric {
            value: 1.55e-6,
            unit: "m".into(),
        };
        let s = format!("{obj}");
        assert!(s.contains("1.55e-6") || s.contains("0.00000155"), "s={s}");
    }

    #[test]
    fn rdf_object_display_literal() {
        let obj = RdfObject::Literal("Waveguide".into());
        assert_eq!(format!("{obj}"), "\"Waveguide\"");
    }

    #[test]
    fn graph_is_empty() {
        let g = KnowledgeGraph::new("https://oxirs.io/");
        assert!(g.is_empty());
    }
}
