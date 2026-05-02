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
//! Real HTTP connectivity to a SPARQL endpoint is provided by [`OxirsConnection`],
//! which is available when the `io-oxirs` feature flag is enabled.

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

/// Real SPARQL-over-HTTP connection to an oxirs-fuseki or compatible endpoint.
///
/// Requires the `io-oxirs` feature flag.
///
/// # Example
///
/// ```rust,ignore
/// # #[cfg(feature = "io-oxirs")]
/// let conn = OxirsConnection::new("http://localhost:3030/dataset/sparql");
/// let rows = conn.query("SELECT * WHERE { ?s ?p ?o } LIMIT 10")?;
/// ```
#[cfg(feature = "io-oxirs")]
#[derive(Debug, Clone)]
pub struct OxirsConnection {
    /// Remote SPARQL endpoint URL (e.g. `http://localhost:3030/dataset`).
    pub endpoint: String,
    /// Optional Bearer authentication token.
    pub token: Option<String>,
}

#[cfg(feature = "io-oxirs")]
impl OxirsConnection {
    /// Create a connection to a SPARQL-compatible endpoint.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            token: None,
        }
    }

    /// Attach a Bearer authentication token to all requests from this connection.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Upload a knowledge graph as N-Triples to the SPARQL endpoint.
    ///
    /// Sends a POST request with `Content-Type: application/n-triples`.
    /// Returns the number of bytes sent on success.
    pub fn upload_graph(&self, graph: &KnowledgeGraph) -> Result<usize, String> {
        let data = graph.to_n_triples();
        let byte_count = data.len();

        let mut req = ureq::post(&self.endpoint).header("Content-Type", "application/n-triples");

        if let Some(token) = &self.token {
            req = req.header("Authorization", format!("Bearer {token}"));
        }

        req.send(data.as_bytes())
            .map_err(|e| format!("upload_graph HTTP error: {e}"))?;

        Ok(byte_count)
    }

    /// Execute a SPARQL SELECT query against the endpoint.
    ///
    /// Sends a GET request with the query URL-encoded as `?query=...`
    /// and `Accept: application/sparql-results+json`.
    ///
    /// Returns a list of result rows; each row is a list of binding values
    /// (one per variable, in the order returned by the endpoint).
    pub fn query(&self, sparql: &str) -> Result<Vec<Vec<String>>, String> {
        let mut req = ureq::get(&self.endpoint)
            .query("query", sparql)
            .header("Accept", "application/sparql-results+json");

        if let Some(token) = &self.token {
            req = req.header("Authorization", format!("Bearer {token}"));
        }

        let body = req
            .call()
            .map_err(|e| format!("query HTTP error: {e}"))?
            .body_mut()
            .read_to_string()
            .map_err(|e| format!("query read error: {e}"))?;

        parse_sparql_json(&body)
    }
}

/// Parse a SPARQL 1.1 Query Results JSON Format response body.
///
/// Expects the standard structure:
/// ```json
/// {"head":{"vars":["v1","v2"]},"results":{"bindings":[{"v1":{"type":"literal","value":"x"}}]}}
/// ```
///
/// Returns one `Vec<String>` per result row, with values ordered by the
/// variable list in `head.vars`. Missing bindings in a row produce an empty string.
#[cfg(feature = "io-oxirs")]
fn parse_sparql_json(body: &str) -> Result<Vec<Vec<String>>, String> {
    use serde_json::Value;

    let json: Value =
        serde_json::from_str(body).map_err(|e| format!("query JSON parse error: {e}"))?;

    let vars: Vec<&str> = json
        .get("head")
        .and_then(|h| h.get("vars"))
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    let bindings = json
        .get("results")
        .and_then(|r| r.get("bindings"))
        .and_then(|b| b.as_array())
        .ok_or_else(|| "query: unexpected SPARQL JSON structure".to_string())?;

    let rows: Vec<Vec<String>> = bindings
        .iter()
        .map(|binding| {
            vars.iter()
                .map(|var| {
                    binding
                        .get(var)
                        .and_then(|b| b.get("value"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string()
                })
                .collect()
        })
        .collect();

    Ok(rows)
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

    // --- io-oxirs feature tests ---

    #[cfg(feature = "io-oxirs")]
    #[test]
    fn oxirs_connection_upload_returns_error_on_unreachable() {
        let conn = OxirsConnection::new("http://127.0.0.1:1/sparql");
        let mut g = KnowledgeGraph::new("https://test/");
        g.add_literal("s", "p", "o");
        let result = conn.upload_graph(&g);
        assert!(result.is_err(), "should fail to connect to port 1");
    }

    #[cfg(feature = "io-oxirs")]
    #[test]
    fn oxirs_connection_query_returns_error_on_unreachable() {
        let conn = OxirsConnection::new("http://127.0.0.1:1/sparql");
        let result = conn.query("SELECT * WHERE { ?s ?p ?o }");
        assert!(result.is_err(), "should fail to connect to port 1");
    }

    #[cfg(feature = "io-oxirs")]
    #[test]
    fn parse_sparql_json_single_variable() {
        let body = r#"{"head":{"vars":["name"]},"results":{"bindings":[{"name":{"type":"literal","value":"Alice"}},{"name":{"type":"literal","value":"Bob"}}]}}"#;
        let rows = parse_sparql_json(body).expect("parse");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec!["Alice"]);
        assert_eq!(rows[1], vec!["Bob"]);
    }

    #[cfg(feature = "io-oxirs")]
    #[test]
    fn parse_sparql_json_multiple_variables() {
        let body = r#"{"head":{"vars":["s","p","o"]},"results":{"bindings":[{"s":{"type":"uri","value":"http://example.org/s"},"p":{"type":"uri","value":"http://example.org/p"},"o":{"type":"literal","value":"hello"}}]}}"#;
        let rows = parse_sparql_json(body).expect("parse");
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0],
            vec!["http://example.org/s", "http://example.org/p", "hello"]
        );
    }

    #[cfg(feature = "io-oxirs")]
    #[test]
    fn parse_sparql_json_empty_results() {
        let body = r#"{"head":{"vars":["x"]},"results":{"bindings":[]}}"#;
        let rows = parse_sparql_json(body).expect("parse");
        assert!(rows.is_empty());
    }

    #[cfg(feature = "io-oxirs")]
    #[test]
    fn parse_sparql_json_missing_binding_yields_empty_string() {
        // Row has ?s but not ?o — missing binding should become "".
        let body = r#"{"head":{"vars":["s","o"]},"results":{"bindings":[{"s":{"type":"literal","value":"only_s"}}]}}"#;
        let rows = parse_sparql_json(body).expect("parse");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0], vec!["only_s", ""]);
    }

    #[cfg(feature = "io-oxirs")]
    #[test]
    fn parse_sparql_json_invalid_json_returns_error() {
        let result = parse_sparql_json("not json at all");
        assert!(result.is_err());
    }

    #[cfg(feature = "io-oxirs")]
    #[test]
    fn parse_sparql_json_missing_results_key_returns_error() {
        let body = r#"{"head":{"vars":["x"]}}"#;
        let result = parse_sparql_json(body);
        assert!(result.is_err());
    }
}
