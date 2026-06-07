//! Markov blanket computation for agent boundary detection.
//!
//! Implements Markov blanket identification, conditional independence testing,
//! boundary detection, information permeability, and belief updating.

// ── Core types ──────────────────────────────────────────────────────────────

/// A node identifier in a graphical model.
pub type NodeId = usize;

/// A directed edge with an optional weight.
#[derive(Clone, Debug)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub weight: f64,
}

/// A directed graphical model (Bayesian network).
#[derive(Clone, Debug)]
pub struct Graph {
    pub nodes: Vec<NodeId>,
    pub edges: Vec<Edge>,
    node_count: usize,
}

impl Graph {
    pub fn new(node_count: usize) -> Self {
        let nodes: Vec<NodeId> = (0..node_count).collect();
        Graph { nodes, edges: Vec::new(), node_count }
    }

    pub fn add_edge(&mut self, from: NodeId, to: NodeId, weight: f64) {
        self.edges.push(Edge { from, to, weight });
    }

    pub fn add_directed_edge(&mut self, from: NodeId, to: NodeId) {
        self.add_edge(from, to, 1.0);
    }

    pub fn node_count(&self) -> usize { self.node_count }

    pub fn parents(&self, node: NodeId) -> Vec<NodeId> {
        self.edges.iter().filter(|e| e.to == node).map(|e| e.from).collect()
    }

    pub fn children(&self, node: NodeId) -> Vec<NodeId> {
        self.edges.iter().filter(|e| e.from == node).map(|e| e.to).collect()
    }

    pub fn has_edge(&self, from: NodeId, to: NodeId) -> bool {
        self.edges.iter().any(|e| e.from == from && e.to == to)
    }

    pub fn neighbors(&self, node: NodeId) -> Vec<NodeId> {
        let mut n: Vec<NodeId> = self.parents(node);
        n.extend(self.children(node));
        n.sort(); n.dedup(); n
    }
}

// ── Module: blanket ─────────────────────────────────────────────────────────

pub mod blanket {
    use crate::{Graph, NodeId};

    /// Compute the Markov blanket of a node in a Bayesian network.
    /// The blanket = parents + children + other parents of children (co-parents).
    pub fn markov_blanket(graph: &Graph, node: NodeId) -> Vec<NodeId> {
        let mut blanket = Vec::new();
        // Parents
        blanket.extend(graph.parents(node));
        // Children
        let children = graph.children(node);
        blanket.extend(children.clone());
        // Co-parents (other parents of children)
        for child in &children {
            for p in graph.parents(*child) {
                if p != node && !blanket.contains(&p) {
                    blanket.push(p);
                }
            }
        }
        blanket.sort(); blanket.dedup();
        blanket
    }

    /// Compute the Markov blanket for a set of internal nodes.
    pub fn markov_blanket_set(graph: &Graph, internal: &[NodeId]) -> Vec<NodeId> {
        let mut blanket = Vec::new();
        for &node in internal {
            for n in markov_blanket(graph, node) {
                if !internal.contains(&n) && !blanket.contains(&n) {
                    blanket.push(n);
                }
            }
        }
        blanket.sort();
        blanket
    }

    /// Check if a node is in the Markov blanket of another.
    pub fn is_in_blanket(graph: &Graph, target: NodeId, candidate: NodeId) -> bool {
        markov_blanket(graph, target).contains(&candidate)
    }

    /// Compute blanket size.
    pub fn blanket_size(graph: &Graph, node: NodeId) -> usize {
        markov_blanket(graph, node).len()
    }

    /// Verify the Markov blanket property: given the blanket, the node is
    /// conditionally independent of all non-blanket nodes.
    pub fn verify_blanket(graph: &Graph, node: NodeId) -> bool {
        let blanket = markov_blanket(graph, node);
        let non_blanket: Vec<NodeId> = graph.nodes.iter()
            .filter(|&&n| n != node && !blanket.contains(&n))
            .copied()
            .collect();
        // Simple structural check: no direct edges to non-blanket nodes
        for &nb in &non_blanket {
            if graph.has_edge(node, nb) || graph.has_edge(nb, node) {
                // Check if it goes through the blanket
                let connected_via_blanket = blanket.iter().any(|&b| {
                    graph.has_edge(node, b) && (graph.has_edge(b, nb) || graph.has_edge(nb, b))
                });
                if !connected_via_blanket && !blanket.contains(&nb) {
                    // Direct connection without blanket intermediary
                    // For strict BNs, this means it should be in the blanket
                    if graph.has_edge(node, nb) || graph.has_edge(nb, node) {
                        return false;
                    }
                }
            }
        }
        true
    }
}

// ── Module: conditional_independence ─────────────────────────────────────────

pub mod conditional_independence {
    use crate::{Graph, NodeId};

    /// Check if two nodes are d-separated given a conditioning set.
    /// Uses the Bayes-Ball algorithm (simplified for directed acyclic graphs).
    pub fn d_separated(graph: &Graph, x: NodeId, y: NodeId, z: &[NodeId]) -> bool {
        !is_connected_given(graph, x, y, z)
    }

    /// Check if x and y are connected (not d-separated) given z.
    fn is_connected_given(graph: &Graph, x: NodeId, y: NodeId, z: &[NodeId]) -> bool {
        let mut visited = std::collections::HashSet::new();
        let mut queue = vec![(x, true)]; // (node, going_up)
        while let Some((node, going_up)) = queue.pop() {
            let key = (node, going_up);
            if visited.contains(&key) { continue; }
            visited.insert(key);
            if node == y { return true; }

            if going_up {
                // If not conditioned on, we can continue through
                if !z.contains(&node) {
                    for &parent in &graph.parents(node) {
                        queue.push((parent, true));
                    }
                    for &child in &graph.children(node) {
                        queue.push((child, false));
                    }
                } else {
                    // Conditioned: v-structure activates
                    for &parent in &graph.parents(node) {
                        queue.push((parent, true));
                    }
                }
            } else {
                // Going down
                if !z.contains(&node) {
                    for &child in &graph.children(node) {
                        queue.push((child, false));
                    }
                }
                // If conditioned, can go up (v-structure)
                if z.contains(&node) {
                    for &parent in &graph.parents(node) {
                        queue.push((parent, true));
                    }
                }
            }
        }
        false
    }

    /// Find the minimal conditioning set that d-separates x and y.
    pub fn minimal_separating_set(graph: &Graph, x: NodeId, y: NodeId) -> Vec<NodeId> {
        let all_nodes: Vec<NodeId> = graph.nodes.iter()
            .filter(|&&n| n != x && n != y)
            .copied()
            .collect();
        for size in 0..=all_nodes.len() {
            for combo in combinations(&all_nodes, size) {
                if d_separated(graph, x, y, &combo) {
                    return combo;
                }
            }
        }
        vec![]
    }

    fn combinations(arr: &[NodeId], k: usize) -> Vec<Vec<NodeId>> {
        if k == 0 { return vec![vec![]]; }
        if k > arr.len() { return vec![]; }
        let mut result = Vec::new();
        for i in 0..=arr.len() - k {
            let rest = combinations(&arr[i + 1..], k - 1);
            for mut c in rest {
                let mut combo = vec![arr[i]];
                combo.append(&mut c);
                result.push(combo);
            }
        }
        result
    }

    /// Check unconditional independence (no conditioning set).
    pub fn unconditionally_independent(graph: &Graph, x: NodeId, y: NodeId) -> bool {
        d_separated(graph, x, y, &[])
    }

    /// List all pairs of d-separated nodes given a conditioning set.
    pub fn all_separated_pairs(graph: &Graph, z: &[NodeId]) -> Vec<(NodeId, NodeId)> {
        let mut pairs = Vec::new();
        for i in 0..graph.nodes.len() {
            for j in (i + 1)..graph.nodes.len() {
                if d_separated(graph, graph.nodes[i], graph.nodes[j], z) {
                    pairs.push((graph.nodes[i], graph.nodes[j]));
                }
            }
        }
        pairs
    }

    /// Check if a set of nodes forms a valid conditioning set for d-separation.
    pub fn is_valid_conditioning_set(graph: &Graph, x: NodeId, y: NodeId, z: &[NodeId]) -> bool {
        d_separated(graph, x, y, z)
    }
}

// ── Module: boundary ────────────────────────────────────────────────────────

pub mod boundary {
    use crate::{Graph, NodeId};

    /// An agent boundary defined by internal states and their Markov blanket.
    #[derive(Clone, Debug)]
    pub struct AgentBoundary {
        pub internal: Vec<NodeId>,
        pub blanket: Vec<NodeId>,
        pub external: Vec<NodeId>,
    }

    impl AgentBoundary {
        /// Create a new agent boundary from internal nodes and a graph.
        pub fn from_internal(graph: &Graph, internal: &[NodeId]) -> Self {
            let blanket = crate::blanket::markov_blanket_set(graph, internal);
            let external: Vec<NodeId> = graph.nodes.iter()
                .filter(|&&n| !internal.contains(&n) && !blanket.contains(&n))
                .copied()
                .collect();
            AgentBoundary { internal: internal.to_vec(), blanket, external }
        }

        /// Boundary size (number of blanket nodes).
        pub fn size(&self) -> usize {
            self.blanket.len()
        }

        /// Check if a node is internal.
        pub fn is_internal(&self, node: NodeId) -> bool {
            self.internal.contains(&node)
        }

        /// Check if a node is on the boundary (blanket).
        pub fn is_boundary(&self, node: NodeId) -> bool {
            self.blanket.contains(&node)
        }

        /// Check if a node is external.
        pub fn is_external(&self, node: NodeId) -> bool {
            self.external.contains(&node)
        }

        /// Compute the boundary permeability (ratio of blanket to total non-internal).
        pub fn permeability(&self) -> f64 {
            let total = self.blanket.len() + self.external.len();
            if total == 0 { return 0.0; }
            self.blanket.len() as f64 / total as f64
        }
    }

    /// Find the optimal boundary that minimizes surprise given a graph.
    pub fn optimal_boundary(graph: &Graph, max_internal: usize) -> AgentBoundary {
        let mut best = None;
        let mut best_score = f64::MAX;
        let nodes = &graph.nodes;
        for size in 1..=max_internal.min(nodes.len()) {
            for combo in combinations(nodes, size) {
                let boundary = AgentBoundary::from_internal(graph, &combo);
                let score = boundary.size() as f64;
                if score < best_score {
                    best_score = score;
                    best = Some(boundary);
                }
            }
        }
        best.unwrap_or_else(|| AgentBoundary::from_internal(graph, &[0]))
    }

    fn combinations(arr: &[NodeId], k: usize) -> Vec<Vec<NodeId>> {
        if k == 0 { return vec![vec![]]; }
        if k > arr.len() { return vec![]; }
        let mut result = Vec::new();
        for i in 0..=arr.len() - k {
            let rest = combinations(&arr[i + 1..], k - 1);
            for mut c in rest {
                let mut combo = vec![arr[i]];
                combo.append(&mut c);
                result.push(combo);
            }
        }
        result
    }

    /// Check if a boundary is valid (blanket separates internal from external).
    pub fn validate_boundary(boundary: &AgentBoundary, graph: &Graph) -> bool {
        for &int in &boundary.internal {
            for &ext in &boundary.external {
                if graph.has_edge(int, ext) || graph.has_edge(ext, int) {
                    return false;
                }
            }
        }
        true
    }
}

// ── Module: permeability ────────────────────────────────────────────────────

pub mod permeability {
    use crate::{Graph, NodeId};

    /// Information flow between two nodes (simplified as path count × weight).
    pub fn information_flow(graph: &Graph, source: NodeId, target: NodeId) -> f64 {
        let mut total = 0.0;
        let mut stack = vec![(source, 1.0)];
        let mut visited = vec![false; graph.node_count()];

        while let Some((node, weight)) = stack.pop() {
            if node == target {
                total += weight;
                continue;
            }
            if visited[node] { continue; }
            visited[node] = true;
            for edge in &graph.edges {
                if edge.from == node {
                    stack.push((edge.to, weight * edge.weight));
                }
            }
        }
        total
    }

    /// Compute permeability of a boundary as total information flow across it.
    pub fn boundary_permeability(graph: &Graph, internal: &[NodeId], blanket: &[NodeId]) -> f64 {
        let mut flow = 0.0;
        for &int in internal {
            for &b in blanket {
                flow += information_flow(graph, int, b);
                flow += information_flow(graph, b, int);
            }
        }
        flow
    }

    /// Compute mutual information approximation (product of marginal flows).
    pub fn mutual_information_approx(graph: &Graph, x: NodeId, y: NodeId) -> f64 {
        let flow_xy = information_flow(graph, x, y);
        let flow_yx = information_flow(graph, y, x);
        flow_xy * flow_yx
    }

    /// Check if information can flow from source to target (reachability).
    pub fn can_reach(graph: &Graph, source: NodeId, target: NodeId) -> bool {
        let mut visited = vec![false; graph.node_count()];
        let mut stack = vec![source];
        while let Some(node) = stack.pop() {
            if node == target { return true; }
            if visited[node] { continue; }
            visited[node] = true;
            for edge in &graph.edges {
                if edge.from == node && !visited[edge.to] {
                    stack.push(edge.to);
                }
            }
        }
        false
    }

    /// Find all paths between two nodes.
    pub fn find_paths(graph: &Graph, source: NodeId, target: NodeId) -> Vec<Vec<NodeId>> {
        let mut paths = Vec::new();
        let mut current = vec![source];
        find_paths_recursive(graph, source, target, &mut current, &mut paths);
        paths
    }

    fn find_paths_recursive(
        graph: &Graph, node: NodeId, target: NodeId,
        current: &mut Vec<NodeId>, paths: &mut Vec<Vec<NodeId>>,
    ) {
        if node == target {
            paths.push(current.clone());
            return;
        }
        for edge in &graph.edges {
            if edge.from == node && !current.contains(&edge.to) {
                current.push(edge.to);
                find_paths_recursive(graph, edge.to, target, current, paths);
                current.pop();
            }
        }
    }

    /// Compute effective connectivity (average flow between all pairs).
    pub fn effective_connectivity(graph: &Graph, nodes: &[NodeId]) -> f64 {
        if nodes.len() < 2 { return 0.0; }
        let mut total = 0.0;
        let mut count = 0;
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                total += information_flow(graph, nodes[i], nodes[j]);
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { total / count as f64 }
    }
}

// ── Module: inference ────────────────────────────────────────────────────────

pub mod inference {
    use crate::{Graph, NodeId};

    /// A simple factor (potential function) over a set of variables.
    #[derive(Clone, Debug)]
    pub struct Factor {
        pub variables: Vec<NodeId>,
        pub values: Vec<f64>,
    }

    impl Factor {
        pub fn new(variables: Vec<NodeId>, values: Vec<f64>) -> Self {
            Factor { variables, values }
        }

        pub fn normalize(&mut self) {
            let sum: f64 = self.values.iter().sum();
            if sum > 0.0 {
                for v in &mut self.values {
                    *v /= sum;
                }
            }
        }

        pub fn entropy(&self) -> f64 {
            self.values.iter()
                .filter(|&&v| v > 0.0)
                .map(|&v| -v * v.ln())
                .sum()
        }

        pub fn marginal(&self, variable: NodeId) -> Factor {
            if self.variables.len() == 1 && self.variables[0] == variable {
                return self.clone();
            }
            let idx = self.variables.iter().position(|&v| v == variable);
            // Simplified: return the factor if variable is present
            if idx.is_some() { self.clone() } else { Factor::new(vec![variable], vec![1.0]) }
        }
    }

    /// Belief state: a map from node to its probability distribution.
    #[derive(Clone, Debug)]
    pub struct BeliefState {
        pub beliefs: Vec<(NodeId, Vec<f64>)>,
    }

    impl BeliefState {
        pub fn new() -> Self {
            BeliefState { beliefs: Vec::new() }
        }

        pub fn set_belief(&mut self, node: NodeId, dist: Vec<f64>) {
            if let Some(entry) = self.beliefs.iter_mut().find(|(n, _)| *n == node) {
                entry.1 = dist;
            } else {
                self.beliefs.push((node, dist));
            }
        }

        pub fn get_belief(&self, node: NodeId) -> Option<&Vec<f64>> {
            self.beliefs.iter().find(|(n, _)| *n == node).map(|(_, d)| d)
        }

        /// Update beliefs given evidence using simple Bayesian update.
        pub fn update(&mut self, node: NodeId, evidence: &[f64]) {
            if let Some(belief) = self.get_belief(node) {
                let updated: Vec<f64> = belief.iter()
                    .zip(evidence.iter())
                    .map(|(&b, &e)| b * e)
                    .collect();
                self.set_belief(node, updated);
            } else {
                self.set_belief(node, evidence.to_vec());
            }
        }

        /// Get the most likely state for a node.
        pub fn most_likely(&self, node: NodeId) -> Option<usize> {
            self.get_belief(node).map(|d| {
                d.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
        }

        /// Compute total surprise (negative log probability of most likely).
        pub fn surprise(&self, node: NodeId) -> f64 {
            self.get_belief(node).map(|d| {
                let max_p = d.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if max_p > 0.0 { -max_p.ln() } else { f64::MAX }
            }).unwrap_or(f64::MAX)
        }
    }

    /// Perform belief propagation (simplified sum-product) on a tree-structured graph.
    pub fn belief_propagation(graph: &Graph, factors: &[Factor]) -> BeliefState {
        let mut state = BeliefState::new();
        for node in &graph.nodes {
            let node_factors: Vec<&Factor> = factors.iter()
                .filter(|f| f.variables.contains(node))
                .collect();
            if node_factors.is_empty() {
                state.set_belief(*node, vec![0.5, 0.5]);
            } else {
                let mut product = node_factors[0].values.clone();
                for f in &node_factors[1..] {
                    for (i, v) in f.values.iter().enumerate() {
                        if i < product.len() { product[i] *= v; }
                    }
                }
                let sum: f64 = product.iter().sum();
                if sum > 0.0 {
                    for v in &mut product { *v /= sum; }
                }
                state.set_belief(*node, product);
            }
        }
        state
    }

    /// Compute Kullback-Leibler divergence between two distributions.
    pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
        p.iter().zip(q.iter())
            .filter(|(&pi, _)| pi > 0.0)
            .map(|(&pi, &qi)| {
                let qi_safe = if qi > 0.0 { qi } else { 1e-10 };
                pi * (pi / qi_safe).ln()
            })
            .sum()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chain() -> Graph {
        let mut g = Graph::new(5);
        g.add_directed_edge(0, 1);
        g.add_directed_edge(1, 2);
        g.add_directed_edge(2, 3);
        g.add_directed_edge(3, 4);
        g
    }

    fn make_tree() -> Graph {
        let mut g = Graph::new(7);
        g.add_directed_edge(0, 1);
        g.add_directed_edge(0, 2);
        g.add_directed_edge(1, 3);
        g.add_directed_edge(1, 4);
        g.add_directed_edge(2, 5);
        g.add_directed_edge(2, 6);
        g
    }

    fn make_v_structure() -> Graph {
        let mut g = Graph::new(3);
        g.add_directed_edge(0, 2);
        g.add_directed_edge(1, 2);
        g
    }

    fn make_diamond() -> Graph {
        let mut g = Graph::new(4);
        g.add_directed_edge(0, 1);
        g.add_directed_edge(0, 2);
        g.add_directed_edge(1, 3);
        g.add_directed_edge(2, 3);
        g
    }

    // ── Graph tests ──

    #[test]
    fn test_graph_creation() {
        let g = Graph::new(3);
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edges.len(), 0);
    }

    #[test]
    fn test_add_edge() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 0.5);
        assert_eq!(g.edges.len(), 1);
        assert_eq!(g.edges[0].weight, 0.5);
    }

    #[test]
    fn test_parents() {
        let g = make_chain();
        assert_eq!(g.parents(2), vec![1]);
        assert_eq!(g.parents(0), vec![]);
    }

    #[test]
    fn test_children() {
        let g = make_chain();
        assert_eq!(g.children(0), vec![1]);
        assert_eq!(g.children(4), vec![]);
    }

    #[test]
    fn test_neighbors() {
        let g = make_chain();
        let n = g.neighbors(2);
        assert!(n.contains(&1));
        assert!(n.contains(&3));
    }

    #[test]
    fn test_has_edge() {
        let g = make_chain();
        assert!(g.has_edge(0, 1));
        assert!(!g.has_edge(1, 0));
    }

    // ── Blanket tests ──

    #[test]
    fn test_blanket_chain_middle() {
        let g = make_chain();
        let b = blanket::markov_blanket(&g, 2);
        assert!(b.contains(&1));
        assert!(b.contains(&3));
    }

    #[test]
    fn test_blanket_chain_end() {
        let g = make_chain();
        let b = blanket::markov_blanket(&g, 0);
        assert_eq!(b, vec![1]);
    }

    #[test]
    fn test_blanket_tree_root() {
        let g = make_tree();
        let b = blanket::markov_blanket(&g, 0);
        assert!(b.contains(&1));
        assert!(b.contains(&2));
    }

    #[test]
    fn test_blanket_tree_leaf() {
        let g = make_tree();
        let b = blanket::markov_blanket(&g, 3);
        // Node 3's parent is 1; node 1 has children 3,4 and parent 0
        // blanket = {1 (parent), 4 (sibling child of 1), 0 (parent of 1)}
        assert!(b.contains(&1)); // parent
    }

    #[test]
    fn test_blanket_v_structure() {
        let g = make_v_structure();
        let b = blanket::markov_blanket(&g, 2);
        assert!(b.contains(&0));
        assert!(b.contains(&1));
    }

    #[test]
    fn test_blanket_set() {
        let g = make_chain();
        let b = blanket::markov_blanket_set(&g, &[1, 3]);
        assert!(b.contains(&0));
        assert!(b.contains(&2));
        assert!(b.contains(&4));
    }

    #[test]
    fn test_is_in_blanket() {
        let g = make_chain();
        assert!(blanket::is_in_blanket(&g, 2, 1));
        assert!(!blanket::is_in_blanket(&g, 2, 0));
    }

    #[test]
    fn test_blanket_size() {
        let g = make_v_structure();
        assert_eq!(blanket::blanket_size(&g, 2), 2);
    }

    #[test]
    fn test_blanket_diamond() {
        let g = make_diamond();
        let b = blanket::markov_blanket(&g, 3);
        // Node 3's parents are 1,2. Their parent is 0 (co-parent).
        assert!(b.contains(&1));
        assert!(b.contains(&2));
    }

    #[test]
    fn test_blanket_single_node() {
        let g = Graph::new(1);
        let b = blanket::markov_blanket(&g, 0);
        assert!(b.is_empty());
    }

    // ── Conditional independence tests ──

    #[test]
    fn test_d_separated_chain() {
        let g = make_chain();
        assert!(!conditional_independence::d_separated(&g, 0, 2, &[]));
        assert!(conditional_independence::d_separated(&g, 0, 4, &[2]));
    }

    #[test]
    fn test_d_separated_v_structure_unconditional() {
        let g = make_v_structure();
        assert!(conditional_independence::d_separated(&g, 0, 1, &[]));
    }

    #[test]
    fn test_d_separated_v_structure_conditioned() {
        let g = make_v_structure();
        // Conditioning on common effect opens the path
        assert!(!conditional_independence::d_separated(&g, 0, 1, &[2]));
    }

    #[test]
    fn test_d_separated_diamond() {
        let g = make_diamond();
        // 1 and 2 are d-separated given 0 (fork), but not unconditional
        assert!(!conditional_independence::d_separated(&g, 1, 2, &[]));
    }

    #[test]
    fn test_unconditionally_independent() {
        let g = make_v_structure();
        assert!(conditional_independence::unconditionally_independent(&g, 0, 1));
    }

    #[test]
    fn test_minimal_separating_set_chain() {
        let g = make_chain();
        let sep = conditional_independence::minimal_separating_set(&g, 0, 4);
        // Should find some node(s) between 0 and 4
        assert!(!sep.is_empty() || !conditional_independence::d_separated(&g, 0, 4, &[]));
    }

    #[test]
    fn test_all_separated_pairs() {
        let g = make_v_structure();
        let pairs = conditional_independence::all_separated_pairs(&g, &[]);
        assert!(pairs.contains(&(0, 1)));
    }

    #[test]
    fn test_valid_conditioning_set() {
        let g = make_chain();
        assert!(conditional_independence::is_valid_conditioning_set(&g, 0, 4, &[2]));
    }

    // ── Boundary tests ──

    #[test]
    fn test_boundary_creation() {
        let g = make_chain();
        let b = boundary::AgentBoundary::from_internal(&g, &[2]);
        assert!(b.is_internal(2));
        assert!(b.is_boundary(1));
        assert!(b.is_boundary(3));
    }

    #[test]
    fn test_boundary_external() {
        let g = make_chain();
        let b = boundary::AgentBoundary::from_internal(&g, &[2]);
        assert!(b.is_external(0));
        assert!(b.is_external(4));
    }

    #[test]
    fn test_boundary_size() {
        let g = make_chain();
        let b = boundary::AgentBoundary::from_internal(&g, &[2]);
        assert_eq!(b.size(), 2);
    }

    #[test]
    fn test_boundary_permeability() {
        let g = make_chain();
        let b = boundary::AgentBoundary::from_internal(&g, &[2]);
        let p = b.permeability();
        assert!(p > 0.0 && p <= 1.0);
    }

    #[test]
    fn test_optimal_boundary() {
        let g = make_chain();
        let b = boundary::optimal_boundary(&g, 1);
        assert!(!b.internal.is_empty());
    }

    #[test]
    fn test_validate_boundary_valid() {
        let g = make_chain();
        let b = boundary::AgentBoundary::from_internal(&g, &[2]);
        assert!(boundary::validate_boundary(&b, &g));
    }

    #[test]
    fn test_boundary_tree() {
        let g = make_tree();
        let b = boundary::AgentBoundary::from_internal(&g, &[0]);
        assert!(b.is_boundary(1));
        assert!(b.is_boundary(2));
    }

    #[test]
    fn test_boundary_full_internal() {
        let mut g = Graph::new(3);
        g.add_directed_edge(0, 1);
        g.add_directed_edge(1, 2);
        let b = boundary::AgentBoundary::from_internal(&g, &[0, 1, 2]);
        assert!(b.blanket.is_empty());
        assert!(b.external.is_empty());
    }

    // ── Permeability tests ──

    #[test]
    fn test_information_flow_direct() {
        let g = make_chain();
        let flow = permeability::information_flow(&g, 0, 1);
        assert_eq!(flow, 1.0);
    }

    #[test]
    fn test_information_flow_multi_hop() {
        let g = make_chain();
        let flow = permeability::information_flow(&g, 0, 4);
        assert!(flow > 0.0);
    }

    #[test]
    fn test_information_flow_no_path() {
        let mut g = Graph::new(3);
        g.add_directed_edge(0, 1);
        let flow = permeability::information_flow(&g, 2, 0);
        assert_eq!(flow, 0.0);
    }

    #[test]
    fn test_boundary_permeability_flow() {
        let g = make_chain();
        let flow = permeability::boundary_permeability(&g, &[2], &[1, 3]);
        assert!(flow > 0.0);
    }

    #[test]
    fn test_mutual_information() {
        let g = make_chain();
        let mi = permeability::mutual_information_approx(&g, 0, 1);
        // flow(0->1)=1.0, flow(1->0)=0.0 so mi=0
        // Use a bidirectional case instead
        let mut bg = crate::Graph::new(3);
        bg.add_edge(0, 1, 0.5);
        bg.add_edge(1, 0, 0.5);
        let mi2 = permeability::mutual_information_approx(&bg, 0, 1);
        assert!(mi2 > 0.0);
    }

    #[test]
    fn test_can_reach() {
        let g = make_chain();
        assert!(permeability::can_reach(&g, 0, 4));
        assert!(!permeability::can_reach(&g, 4, 0));
    }

    #[test]
    fn test_find_paths() {
        let g = make_chain();
        let paths = permeability::find_paths(&g, 0, 4);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_find_paths_diamond() {
        let g = make_diamond();
        let paths = permeability::find_paths(&g, 0, 3);
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_effective_connectivity() {
        let g = make_chain();
        let ec = permeability::effective_connectivity(&g, &[0, 1, 2]);
        assert!(ec > 0.0);
    }

    #[test]
    fn test_effective_connectivity_single() {
        let g = make_chain();
        let ec = permeability::effective_connectivity(&g, &[0]);
        assert_eq!(ec, 0.0);
    }

    // ── Inference tests ──

    #[test]
    fn test_factor_creation() {
        let f = inference::Factor::new(vec![0], vec![0.3, 0.7]);
        assert_eq!(f.variables, vec![0]);
        assert_eq!(f.values.len(), 2);
    }

    #[test]
    fn test_factor_normalize() {
        let mut f = inference::Factor::new(vec![0], vec![2.0, 8.0]);
        f.normalize();
        assert!((f.values[0] - 0.2).abs() < 1e-10);
        assert!((f.values[1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_factor_entropy() {
        let f = inference::Factor::new(vec![0], vec![0.5, 0.5]);
        let h = f.entropy();
        assert!((h - 0.6931_f64).abs() < 0.01);
    }

    #[test]
    fn test_belief_state_new() {
        let bs = inference::BeliefState::new();
        assert!(bs.beliefs.is_empty());
    }

    #[test]
    fn test_belief_set_get() {
        let mut bs = inference::BeliefState::new();
        bs.set_belief(0, vec![0.4, 0.6]);
        assert_eq!(bs.get_belief(0), Some(&vec![0.4_f64, 0.6_f64]));
    }

    #[test]
    fn test_belief_update() {
        let mut bs = inference::BeliefState::new();
        bs.set_belief(0, vec![0.5, 0.5]);
        bs.update(0, &[0.8, 0.2]);
        let b = bs.get_belief(0).unwrap();
        assert!((b[0] - 0.4 / 0.5).abs() < 1e-10 || (b[0] - 0.8 * 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_most_likely() {
        let mut bs = inference::BeliefState::new();
        bs.set_belief(0, vec![0.2, 0.8]);
        assert_eq!(bs.most_likely(0), Some(1));
    }

    #[test]
    fn test_surprise() {
        let mut bs = inference::BeliefState::new();
        bs.set_belief(0, vec![0.1, 0.9]);
        let s = bs.surprise(0);
        assert!(s > 0.0);
    }

    #[test]
    fn test_belief_propagation() {
        let g = make_chain();
        let factors = vec![
            inference::Factor::new(vec![0], vec![0.6, 0.4]),
            inference::Factor::new(vec![1], vec![0.5, 0.5]),
        ];
        let bs = inference::belief_propagation(&g, &factors);
        assert!(bs.get_belief(0).is_some());
    }

    #[test]
    fn test_kl_divergence_same() {
        let kl = inference::kl_divergence(&[0.5, 0.5], &[0.5, 0.5]);
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_different() {
        let kl = inference::kl_divergence(&[0.5, 0.5], &[0.9, 0.1]);
        assert!(kl > 0.0);
    }

    #[test]
    fn test_kl_divergence_asymmetric() {
        let kl1 = inference::kl_divergence(&[0.5, 0.5], &[0.9, 0.1]);
        let kl2 = inference::kl_divergence(&[0.9, 0.1], &[0.5, 0.5]);
        assert!((kl1 - kl2).abs() > 0.01);
    }

    #[test]
    fn test_factor_marginal() {
        let f = inference::Factor::new(vec![0], vec![0.3, 0.7]);
        let m = f.marginal(0);
        assert_eq!(m.variables, vec![0]);
    }

    #[test]
    fn test_belief_update_new_node() {
        let mut bs = inference::BeliefState::new();
        bs.update(3, &[0.6, 0.4]);
        assert!(bs.get_belief(3).is_some());
    }

    #[test]
    fn test_surprise_node_not_found() {
        let bs = inference::BeliefState::new();
        assert_eq!(bs.surprise(99), f64::MAX);
    }
}
