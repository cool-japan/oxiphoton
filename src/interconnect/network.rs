//! Photonic network topology models.
//!
//! Models multi-node photonic networks such as bus, ring, and mesh topologies
//! commonly used in optical interconnects and photonic computing fabrics.

/// Network topology type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkTopology {
    /// Linear bus: all nodes on a single waveguide
    Bus,
    /// Ring: nodes connected in a loop
    Ring,
    /// Mesh/Grid: 2D array (for future extension)
    Mesh,
}

/// Single node in a photonic network.
#[derive(Debug, Clone)]
pub struct NetworkNode {
    /// Node identifier
    pub id: usize,
    /// Drop/insert loss per node (dB)
    pub loss_db: f64,
}

impl NetworkNode {
    pub fn new(id: usize, loss_db: f64) -> Self {
        Self { id, loss_db }
    }

    /// Standard Si microring add-drop node (~3dB drop loss).
    pub fn ring_add_drop(id: usize) -> Self {
        Self::new(id, 3.0)
    }
}

/// Photonic network model.
#[derive(Debug, Clone)]
pub struct PhotonicNetwork {
    /// Topology
    pub topology: NetworkTopology,
    /// List of nodes
    pub nodes: Vec<NetworkNode>,
    /// Waveguide propagation loss between adjacent nodes (dB)
    pub link_loss_db: f64,
    /// Total TX power (dBm) at the source
    pub tx_power_dbm: f64,
}

impl PhotonicNetwork {
    pub fn new(topology: NetworkTopology, tx_power_dbm: f64, link_loss_db: f64) -> Self {
        Self {
            topology,
            nodes: Vec::new(),
            link_loss_db,
            tx_power_dbm,
        }
    }

    /// Add a node to the network.
    pub fn add_node(&mut self, node: NetworkNode) {
        self.nodes.push(node);
    }

    /// Create a bus network with n identical nodes.
    pub fn bus(n_nodes: usize, tx_power_dbm: f64, link_loss_db: f64, node_loss_db: f64) -> Self {
        let mut net = Self::new(NetworkTopology::Bus, tx_power_dbm, link_loss_db);
        for i in 0..n_nodes {
            net.add_node(NetworkNode::new(i, node_loss_db));
        }
        net
    }

    /// Create a ring network with n identical nodes.
    pub fn ring(n_nodes: usize, tx_power_dbm: f64, link_loss_db: f64, node_loss_db: f64) -> Self {
        let mut net = Self::new(NetworkTopology::Ring, tx_power_dbm, link_loss_db);
        for i in 0..n_nodes {
            net.add_node(NetworkNode::new(i, node_loss_db));
        }
        net
    }

    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Power received (dBm) at node k (0-indexed) for bus topology.
    ///
    /// Accounts for all preceding node losses and link losses.
    pub fn power_at_node_bus(&self, k: usize) -> f64 {
        if k >= self.nodes.len() {
            return f64::NEG_INFINITY;
        }
        let mut power = self.tx_power_dbm;
        // Loss from source through k preceding nodes + k link segments
        for i in 0..k {
            power -= self.link_loss_db;
            power -= self.nodes[i].loss_db;
        }
        power -= self.link_loss_db; // last link to node k
        power
    }

    /// Power received (dBm) at each node in the bus topology.
    pub fn power_profile_bus(&self) -> Vec<f64> {
        (0..self.nodes.len())
            .map(|k| self.power_at_node_bus(k))
            .collect()
    }

    /// Maximum number of nodes supportable for a given receiver sensitivity.
    pub fn max_nodes_bus(&self, rx_sensitivity_dbm: f64, node_loss_db: f64) -> usize {
        let loss_per_hop = self.link_loss_db + node_loss_db;
        let budget = self.tx_power_dbm - rx_sensitivity_dbm;
        if loss_per_hop <= 0.0 {
            return usize::MAX;
        }
        (budget / loss_per_hop).floor() as usize
    }

    /// Total optical power injected into the network (mW).
    pub fn tx_power_mw(&self) -> f64 {
        10.0_f64.powf(self.tx_power_dbm / 10.0)
    }

    /// Network power efficiency: total power consumed by nodes / TX power.
    ///
    /// For bus topology, each node consumes the drop fraction of arriving power.
    /// Efficiency ∈ [0, 1].
    pub fn power_efficiency(&self) -> f64 {
        match self.topology {
            NetworkTopology::Bus => {
                let tx_mw = self.tx_power_mw();
                if tx_mw <= 0.0 {
                    return 0.0;
                }
                let mut power_mw = tx_mw;
                let mut total_consumed_mw = 0.0f64;
                let link_factor = 10.0_f64.powf(-self.link_loss_db / 10.0);
                for node in &self.nodes {
                    power_mw *= link_factor; // link loss before this node
                    let node_factor = 10.0_f64.powf(-node.loss_db / 10.0);
                    total_consumed_mw += power_mw * (1.0 - node_factor);
                    power_mw *= node_factor; // node through loss
                }
                (total_consumed_mw / tx_mw).min(1.0)
            }
            _ => 0.0, // stub for other topologies
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PhotonicTopology — graph-level topology models
// ─────────────────────────────────────────────────────────────────────────────

/// A node in the photonic topology graph.
#[derive(Debug, Clone)]
pub struct TopologyNode {
    /// Unique node index
    pub id: usize,
    /// Human-readable label (e.g. "router-A")
    pub label: String,
}

/// A directed edge in the photonic topology graph.
#[derive(Debug, Clone)]
pub struct TopologyEdge {
    /// Source node index
    pub src: usize,
    /// Destination node index
    pub dst: usize,
    /// Fiber length (km)
    pub length_km: f64,
    /// Fiber loss per km (dB/km)
    pub loss_db_per_km: f64,
}

impl TopologyEdge {
    /// Total propagation loss for this edge (dB).
    pub fn total_loss_db(&self) -> f64 {
        self.length_km * self.loss_db_per_km
    }
}

/// Photonic network topology graph.
///
/// Represents the physical connectivity as an undirected weighted graph where
/// edge weights are propagation losses (dB).  Shortest-path queries use
/// Dijkstra's algorithm on the loss graph.
#[derive(Debug, Clone)]
pub struct PhotonicTopology {
    /// Nodes in the topology
    pub nodes: Vec<TopologyNode>,
    /// Edges (stored as undirected: each edge appears once)
    pub edges: Vec<TopologyEdge>,
    /// Default fiber loss (dB/km) used by factory methods
    pub default_loss_db_per_km: f64,
}

impl PhotonicTopology {
    /// Create an empty topology.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            default_loss_db_per_km: 0.2,
        }
    }

    /// Add a node and return its index.
    pub fn add_node(&mut self, label: &str) -> usize {
        let id = self.nodes.len();
        self.nodes.push(TopologyNode {
            id,
            label: label.to_owned(),
        });
        id
    }

    /// Add an undirected link between two nodes.
    pub fn add_link(&mut self, src: usize, dst: usize, length_km: f64) {
        self.edges.push(TopologyEdge {
            src,
            dst,
            length_km,
            loss_db_per_km: self.default_loss_db_per_km,
        });
    }

    /// Ring topology: n nodes arranged in a loop with equal link lengths.
    pub fn ring_topology(n_nodes: usize, link_length_km: f64) -> Self {
        let mut topo = Self::new();
        for i in 0..n_nodes {
            topo.add_node(&format!("node-{i}"));
        }
        for i in 0..n_nodes {
            let next = (i + 1) % n_nodes;
            topo.add_link(i, next, link_length_km);
        }
        topo
    }

    /// Star topology: one hub node connected to `n_leaf` leaf nodes.
    pub fn star_topology(n_leaf: usize, link_length_km: f64) -> Self {
        let mut topo = Self::new();
        topo.add_node("hub");
        for i in 0..n_leaf {
            topo.add_node(&format!("leaf-{i}"));
            topo.add_link(0, i + 1, link_length_km);
        }
        topo
    }

    /// Rectangular mesh topology: `rows × cols` nodes on a grid.
    ///
    /// Horizontal and vertical links have equal length `link_length_km`.
    pub fn mesh_topology(rows: usize, cols: usize, link_length_km: f64) -> Self {
        let mut topo = Self::new();
        for r in 0..rows {
            for c in 0..cols {
                topo.add_node(&format!("n{r}-{c}"));
            }
        }
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                if c + 1 < cols {
                    topo.add_link(idx, idx + 1, link_length_km);
                }
                if r + 1 < rows {
                    topo.add_link(idx, idx + cols, link_length_km);
                }
            }
        }
        topo
    }

    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Build adjacency list: adj[u] = Vec<(v, loss_db)>.
    fn adjacency_list(&self) -> Vec<Vec<(usize, f64)>> {
        let n = self.nodes.len();
        let mut adj = vec![Vec::new(); n];
        for e in &self.edges {
            let loss = e.total_loss_db();
            if e.src < n && e.dst < n {
                adj[e.src].push((e.dst, loss));
                adj[e.dst].push((e.src, loss));
            }
        }
        adj
    }

    /// Shortest path loss (dB) from `src` to `dst` using Dijkstra on the loss graph.
    ///
    /// Returns `None` if `dst` is unreachable.
    pub fn shortest_path_loss(&self, src: usize, dst: usize) -> Option<f64> {
        let n = self.nodes.len();
        if src >= n || dst >= n {
            return None;
        }
        if src == dst {
            return Some(0.0);
        }
        let adj = self.adjacency_list();
        // Dijkstra with a simple Vec-based priority queue (adequate for small N)
        let mut dist = vec![f64::INFINITY; n];
        dist[src] = 0.0;
        // Queue entries: (loss, node_index)
        let mut queue: Vec<(f64, usize)> = Vec::with_capacity(n);
        queue.push((0.0, src));
        while !queue.is_empty() {
            // Find the minimum-cost entry
            let min_pos = queue
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)?;
            let (d, u) = queue.remove(min_pos);
            if d > dist[u] {
                continue;
            }
            if u == dst {
                return Some(d);
            }
            for &(v, w) in &adj[u] {
                let nd = d + w;
                if nd < dist[v] {
                    dist[v] = nd;
                    queue.push((nd, v));
                }
            }
        }
        if dist[dst].is_finite() {
            Some(dist[dst])
        } else {
            None
        }
    }
}

impl Default for PhotonicTopology {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PhotonicNetwork extensions
// ─────────────────────────────────────────────────────────────────────────────

/// A wavelength-routed path through the network.
#[derive(Debug, Clone)]
pub struct WavelengthPath {
    /// Source node
    pub src: usize,
    /// Destination node
    pub dst: usize,
    /// Wavelength (nm)
    pub wavelength_nm: f64,
    /// Data rate (Gb/s)
    pub data_rate_gbps: f64,
}

impl PhotonicNetwork {
    /// Add a wavelength path from `src` to `dst`.
    ///
    /// Returns an error if either node is out of range.
    pub fn add_wavelength_path(
        &mut self,
        _src: usize,
        _dst: usize,
        _lambda_nm: f64,
    ) -> crate::error::Result<WavelengthPath> {
        // Validate node indices
        let n = self.nodes.len();
        if _src >= n || _dst >= n {
            return Err(crate::error::OxiPhotonError::NumericalError(format!(
                "node index out of range: src={_src}, dst={_dst}, n_nodes={n}"
            )));
        }
        Ok(WavelengthPath {
            src: _src,
            dst: _dst,
            wavelength_nm: _lambda_nm,
            data_rate_gbps: 100.0, // default
        })
    }

    /// Aggregate capacity (Gb/s) using the Shannon formula.
    ///
    /// Assumes each link operates at the Shannon capacity with SNR derived from
    /// the link loss and a fixed launch power of 0 dBm.
    pub fn compute_capacity_gbps(&self) -> f64 {
        // SNR ≈ (TX power) / (noise floor); use simplified model
        // C = B · log2(1 + SNR) summed over all nodes
        let noise_floor_dbm = -30.0_f64; // thermal floor
        self.nodes
            .iter()
            .enumerate()
            .map(|(k, _)| {
                let p_dbm = self.power_at_node_bus(k);
                let snr_db = p_dbm - noise_floor_dbm;
                let snr_linear = 10.0_f64.powf(snr_db / 10.0);
                let bw_ghz = 100.0_f64; // assume 100 GHz channel bandwidth
                bw_ghz * (1.0 + snr_linear).log2()
            })
            .sum()
    }

    /// Spectral efficiency (bits/s/Hz) averaged across all nodes.
    pub fn spectral_efficiency(&self) -> f64 {
        let n = self.nodes.len();
        if n == 0 {
            return 0.0;
        }
        let noise_floor_dbm = -30.0_f64;
        let avg_snr_linear: f64 = self
            .nodes
            .iter()
            .enumerate()
            .map(|(k, _)| {
                let p_dbm = self.power_at_node_bus(k);
                let snr_db = p_dbm - noise_floor_dbm;
                10.0_f64.powf(snr_db / 10.0)
            })
            .sum::<f64>()
            / n as f64;
        (1.0 + avg_snr_linear).log2()
    }

    /// Shortest-path loss (dB) from node `src` to node `dst`.
    ///
    /// Uses Dijkstra on the loss graph of the bus/ring network.
    /// For bus topology, loss is proportional to hop count.
    pub fn shortest_path_loss(&self, src: usize, dst: usize) -> Option<f64> {
        let n = self.nodes.len();
        if src >= n || dst >= n {
            return None;
        }
        if src == dst {
            return Some(0.0);
        }
        let hops = (dst as isize - src as isize).unsigned_abs();
        match self.topology {
            NetworkTopology::Bus => {
                let loss = hops as f64
                    * (self.link_loss_db + self.nodes.first().map(|nd| nd.loss_db).unwrap_or(0.0));
                Some(loss)
            }
            NetworkTopology::Ring => {
                let forward = hops;
                let backward = n - hops;
                let min_hops = forward.min(backward);
                let loss = min_hops as f64
                    * (self.link_loss_db + self.nodes.first().map(|nd| nd.loss_db).unwrap_or(0.0));
                Some(loss)
            }
            NetworkTopology::Mesh => None, // not supported without full graph
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bus_power_decreases_with_distance() {
        let net = PhotonicNetwork::bus(5, 0.0, 1.0, 1.0);
        let profile = net.power_profile_bus();
        for i in 1..profile.len() {
            assert!(
                profile[i] < profile[i - 1],
                "Power should decrease: p[{}]={:.1} p[{}]={:.1}",
                i - 1,
                profile[i - 1],
                i,
                profile[i]
            );
        }
    }

    #[test]
    fn bus_first_node_max_power() {
        let net = PhotonicNetwork::bus(4, 0.0, 0.5, 1.0);
        let profile = net.power_profile_bus();
        for &p in &profile[1..] {
            assert!(profile[0] >= p);
        }
    }

    #[test]
    fn max_nodes_finite_and_positive() {
        let net = PhotonicNetwork::bus(10, 0.0, 0.5, 1.0);
        let n = net.max_nodes_bus(-20.0, 1.0);
        assert!(n > 0 && n < 1000);
    }

    #[test]
    fn network_power_efficiency_between_0_and_1() {
        let net = PhotonicNetwork::bus(4, 0.0, 0.5, 1.0);
        let eta = net.power_efficiency();
        assert!((0.0..=1.0).contains(&eta), "η={eta:.4}");
    }

    #[test]
    fn ring_network_has_correct_node_count() {
        let net = PhotonicNetwork::ring(8, 3.0, 0.3, 0.5);
        assert_eq!(net.n_nodes(), 8);
    }

    #[test]
    fn tx_power_mw_conversion() {
        let net = PhotonicNetwork::bus(1, 0.0, 0.0, 0.0);
        assert!((net.tx_power_mw() - 1.0).abs() < 1e-10); // 0 dBm = 1 mW
    }

    #[test]
    fn power_at_node_zero_is_near_tx() {
        let net = PhotonicNetwork::bus(3, 10.0, 0.0, 0.0);
        // With zero losses, node 0 gets TX power
        let p = net.power_at_node_bus(0);
        assert!((p - 10.0).abs() < 1e-10);
    }

    // ── PhotonicTopology tests ───────────────────────────────────────────────

    #[test]
    fn ring_topology_node_count() {
        let topo = PhotonicTopology::ring_topology(6, 10.0);
        assert_eq!(topo.n_nodes(), 6);
        assert_eq!(topo.n_edges(), 6); // ring has N edges
    }

    #[test]
    fn star_topology_node_count() {
        let topo = PhotonicTopology::star_topology(4, 5.0);
        assert_eq!(topo.n_nodes(), 5); // hub + 4 leaves
        assert_eq!(topo.n_edges(), 4);
    }

    #[test]
    fn mesh_topology_node_count() {
        let topo = PhotonicTopology::mesh_topology(3, 4, 2.0);
        assert_eq!(topo.n_nodes(), 12); // 3×4
                                        // Edges: (rows-1)*cols horizontal + rows*(cols-1) vertical = 2*4 + 3*3 = 17
        assert_eq!(topo.n_edges(), 17);
    }

    #[test]
    fn shortest_path_loss_same_node_zero() {
        let topo = PhotonicTopology::ring_topology(5, 10.0);
        let loss = topo.shortest_path_loss(2, 2);
        assert_eq!(loss, Some(0.0));
    }

    #[test]
    fn shortest_path_loss_adjacent_nodes() {
        let topo = PhotonicTopology::ring_topology(5, 10.0);
        // Adjacent: 1 link × 10 km × 0.2 dB/km = 2 dB
        let loss = topo.shortest_path_loss(0, 1).expect("should find path");
        assert!((loss - 2.0).abs() < 1e-10, "loss={loss}");
    }

    #[test]
    fn shortest_path_loss_unreachable_returns_none() {
        let topo = PhotonicTopology::new(); // empty
        let loss = topo.shortest_path_loss(0, 1);
        assert!(loss.is_none());
    }

    #[test]
    fn ring_shortest_path_uses_shorter_arc() {
        // In a 5-node ring, node 0 to node 4 is 1 hop going backward
        let topo = PhotonicTopology::ring_topology(5, 10.0);
        let loss_0_4 = topo.shortest_path_loss(0, 4).expect("path exists");
        let loss_0_2 = topo.shortest_path_loss(0, 2).expect("path exists");
        // 2 hops is more expensive than 1 hop
        assert!(
            loss_0_4 < loss_0_2,
            "shorter arc should have less loss: {loss_0_4} vs {loss_0_2}"
        );
    }

    // ── PhotonicNetwork extension tests ──────────────────────────────────────

    #[test]
    fn compute_capacity_gbps_positive() {
        let net = PhotonicNetwork::bus(4, 0.0, 0.5, 1.0);
        let cap = net.compute_capacity_gbps();
        assert!(cap > 0.0, "capacity should be positive");
    }

    #[test]
    fn spectral_efficiency_positive() {
        let net = PhotonicNetwork::bus(4, 0.0, 0.5, 1.0);
        let se = net.spectral_efficiency();
        assert!(se > 0.0);
    }

    #[test]
    fn add_wavelength_path_valid() {
        let mut net = PhotonicNetwork::bus(4, 0.0, 0.5, 1.0);
        let path = net.add_wavelength_path(0, 2, 1550.0).expect("valid nodes");
        assert_eq!(path.src, 0);
        assert_eq!(path.dst, 2);
    }

    #[test]
    fn add_wavelength_path_invalid_node_error() {
        let mut net = PhotonicNetwork::bus(4, 0.0, 0.5, 1.0);
        let result = net.add_wavelength_path(0, 10, 1550.0); // node 10 out of range
        assert!(result.is_err());
    }

    #[test]
    fn network_shortest_path_loss_bus_same_node() {
        let net = PhotonicNetwork::bus(5, 0.0, 1.0, 1.0);
        assert_eq!(net.shortest_path_loss(2, 2), Some(0.0));
    }

    #[test]
    fn network_shortest_path_loss_bus_adjacent() {
        let net = PhotonicNetwork::bus(5, 0.0, 1.0, 1.0);
        // 1 hop: link_loss + node_loss = 1 + 1 = 2 dB
        let loss = net.shortest_path_loss(0, 1).expect("path exists");
        assert!((loss - 2.0).abs() < 1e-10, "loss={loss}");
    }
}
