//! DAG builder: topological sort and resource validation.

use std::collections::HashSet;

use super::ids::{GraphNodeId, PassId, SubgraphId, SubgraphLabel};
use super::pass_trait::RenderPass;
use super::resources::{GraphBuildError, PassResources, ResourceSlot};
use super::runtime::{ExecutionUnit, LabeledSubgraph, RenderGraph, declared_writes_recursive};

/// Topological node in a [`GraphBuilder`]: either a pass index or a subgraph index.
enum GraphBuilderNode {
    /// Index into [`GraphBuilder::passes`].
    Pass(usize),
    /// Index into [`GraphBuilder::subgraphs`].
    Subgraph(usize),
}

/// Builder for a DAG of render passes and optional subgraphs. Declare nodes and edges, then call
/// [`GraphBuilder::build`] to topologically sort and produce a [`RenderGraph`].
pub struct GraphBuilder {
    passes: Vec<Box<dyn RenderPass>>,
    subgraphs: Vec<(SubgraphLabel, RenderGraph)>,
    /// One entry per [`add_pass`](Self::add_pass) / [`add_subgraph`](Self::add_subgraph), in order.
    nodes: Vec<GraphBuilderNode>,
    /// `pass_id_to_node_index[pass_index] =` index into `nodes`.
    pass_id_to_node_index: Vec<usize>,
    /// `subgraph_id_to_node_index[subgraph_index] =` index into `nodes`.
    subgraph_id_to_node_index: Vec<usize>,
    edges: Vec<(usize, usize)>,
}

impl GraphBuilder {
    /// Creates an empty graph builder.
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            subgraphs: Vec::new(),
            nodes: Vec::new(),
            pass_id_to_node_index: Vec::new(),
            subgraph_id_to_node_index: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Adds a pass to the graph. Returns a [`PassId`] for declaring edges and special pass ids.
    pub fn add_pass(&mut self, pass: Box<dyn RenderPass>) -> PassId {
        let pass_idx = self.passes.len();
        self.passes.push(pass);
        let node_idx = self.nodes.len();
        self.pass_id_to_node_index.push(node_idx);
        self.nodes.push(GraphBuilderNode::Pass(pass_idx));
        PassId(pass_idx)
    }

    /// Adds a nested [`RenderGraph`] as a single scheduled node.
    ///
    /// The subgraph owns its own passes, resource metadata, and RTAO MRT cache. Future
    /// multi-viewport rendering can schedule several subgraphs (e.g. per camera) with edges between
    /// them or root-level passes.
    pub fn add_subgraph(
        &mut self,
        label: impl Into<SubgraphLabel>,
        subgraph: RenderGraph,
    ) -> SubgraphId {
        let sg_idx = self.subgraphs.len();
        self.subgraphs.push((label.into(), subgraph));
        let node_idx = self.nodes.len();
        self.subgraph_id_to_node_index.push(node_idx);
        self.nodes.push(GraphBuilderNode::Subgraph(sg_idx));
        SubgraphId(sg_idx)
    }

    /// Adds a pass only when `condition` is true.
    ///
    /// Use this for graph variants (e.g. RTAO passes only when ray tracing and config allow it).
    /// Returns [`Some`](PassId) with the new id when the pass was added, [`None`] otherwise.
    pub fn add_pass_if(&mut self, condition: bool, pass: Box<dyn RenderPass>) -> Option<PassId> {
        if condition {
            Some(self.add_pass(pass))
        } else {
            None
        }
    }

    fn node_index_for(&self, id: GraphNodeId) -> usize {
        match id {
            GraphNodeId::Pass(PassId(i)) => *self
                .pass_id_to_node_index
                .get(i)
                .expect("PassId from this builder"),
            GraphNodeId::Subgraph(SubgraphId(i)) => *self
                .subgraph_id_to_node_index
                .get(i)
                .expect("SubgraphId from this builder"),
        }
    }

    /// Declares that `from` runs before `to`. Accepts [`PassId`] and/or [`SubgraphId`] via
    /// [`GraphNodeId`].
    pub fn add_edge(&mut self, from: impl Into<GraphNodeId>, to: impl Into<GraphNodeId>) {
        let from_n = self.node_index_for(from.into());
        let to_n = self.node_index_for(to.into());
        self.edges.push((from_n, to_n));
    }

    /// Topologically sorts nodes, validates no cycles, and returns a [`RenderGraph`] with execution
    /// units in sorted order.
    pub fn build(self) -> Result<RenderGraph, GraphBuildError> {
        self.build_with_special_passes(None, None)
    }

    /// Like [`build`](Self::build), but records which [`PassId`]s correspond to the composite
    /// and overlay passes. Used to switch render target to surface for those passes and to
    /// decide whether to run the copy fallback when composite is absent.
    pub fn build_with_special_passes(
        self,
        composite_pass_id: Option<PassId>,
        overlay_pass_id: Option<PassId>,
    ) -> Result<RenderGraph, GraphBuildError> {
        let n = self.nodes.len();
        let mut in_degree = vec![0usize; n];
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];

        for &(from_idx, to_idx) in &self.edges {
            if from_idx >= n || to_idx >= n {
                return Err(GraphBuildError::CycleDetected);
            }
            if from_idx != to_idx {
                neighbors[from_idx].push(to_idx);
                in_degree[to_idx] += 1;
            }
        }

        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut sorted = Vec::with_capacity(n);

        while let Some(node) = queue.pop() {
            sorted.push(node);
            for &neighbor in &neighbors[node] {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 {
                    queue.push(neighbor);
                }
            }
        }

        if sorted.len() != n {
            return Err(GraphBuildError::CycleDetected);
        }

        let mut cumulative_writes: HashSet<ResourceSlot> = HashSet::new();
        for &node_idx in &sorted {
            match self.nodes[node_idx] {
                GraphBuilderNode::Pass(pass_idx) => {
                    let resources = self.passes[pass_idx].resources();
                    for &slot in &resources.reads {
                        if !cumulative_writes.contains(&slot) {
                            return Err(GraphBuildError::MissingDependency {
                                pass: PassId(pass_idx),
                                slot,
                            });
                        }
                    }
                    cumulative_writes.extend(resources.writes.iter().copied());
                }
                GraphBuilderNode::Subgraph(sg_idx) => {
                    cumulative_writes.extend(declared_writes_recursive(&self.subgraphs[sg_idx].1));
                }
            }
        }

        let mut pass_take: Vec<Option<Box<dyn RenderPass>>> =
            self.passes.into_iter().map(Some).collect();
        let mut subgraph_take: Vec<Option<(SubgraphLabel, RenderGraph)>> =
            self.subgraphs.into_iter().map(Some).collect();

        let mut execution: Vec<ExecutionUnit> = Vec::with_capacity(n);
        let mut execution_order_pass_ids: Vec<PassId> = Vec::new();
        let mut pass_resources: Vec<PassResources> = Vec::new();

        for &node_idx in &sorted {
            match self.nodes[node_idx] {
                GraphBuilderNode::Pass(pass_idx) => {
                    let p = pass_take[pass_idx]
                        .take()
                        .expect("pass taken once from builder");
                    let resources = p.resources();
                    pass_resources.push(resources.clone());
                    execution_order_pass_ids.push(PassId(pass_idx));
                    execution.push(ExecutionUnit::Pass { pass: p, resources });
                }
                GraphBuilderNode::Subgraph(sg_idx) => {
                    let (label, graph) = subgraph_take[sg_idx]
                        .take()
                        .expect("subgraph taken once from builder");
                    execution.push(ExecutionUnit::Subgraph(LabeledSubgraph {
                        label,
                        graph: Box::new(graph),
                    }));
                }
            }
        }

        Ok(RenderGraph {
            execution,
            execution_order_pass_ids,
            pass_resources,
            composite_pass_id,
            overlay_pass_id,
            rtao_mrt_cache: None,
        })
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}
