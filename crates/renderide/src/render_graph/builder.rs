//! DAG builder: topological sort, cycle detection, and producer/consumer validation.

use std::collections::HashSet;

use super::compiled::{CompileStats, CompiledRenderGraph};
use super::error::GraphBuildError;
use super::ids::PassId;
use super::pass::RenderPass;
use super::resources::ResourceSlot;

/// Builder for a directed acyclic graph of render passes.
///
/// Declare passes and edges (`from` before `to`), then call [`Self::build`] to obtain an immutable
/// [`CompiledRenderGraph`]. Use [`Self::add_pass_if`] to omit optional branches without placeholder
/// passes.
pub struct GraphBuilder {
    passes: Vec<Box<dyn RenderPass>>,
    edges: Vec<(usize, usize)>,
}

impl GraphBuilder {
    /// Empty builder with no passes or edges.
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Appends a pass; returns its [`PassId`] for [`Self::add_edge`].
    pub fn add_pass(&mut self, pass: Box<dyn RenderPass>) -> PassId {
        let id = self.passes.len();
        self.passes.push(pass);
        PassId(id)
    }

    /// Appends a pass only when `condition` is true (optional RTAO, debug overlays, etc.).
    pub fn add_pass_if(&mut self, condition: bool, pass: Box<dyn RenderPass>) -> Option<PassId> {
        if condition {
            Some(self.add_pass(pass))
        } else {
            None
        }
    }

    /// Ensures `from` is scheduled before `to` (both are pass indices in this builder).
    pub fn add_edge(&mut self, from: PassId, to: PassId) {
        self.edges.push((from.0, to.0));
    }

    /// Topologically sorts passes, validates resource flow, and transfers ownership into a graph.
    pub fn build(self) -> Result<CompiledRenderGraph, GraphBuildError> {
        let n = self.passes.len();
        if n == 0 {
            return Ok(CompiledRenderGraph {
                passes: Vec::new(),
                needs_surface_acquire: false,
                compile_stats: CompileStats {
                    pass_count: 0,
                    topo_levels: 0,
                },
            });
        }

        let mut in_degree = vec![0usize; n];
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];

        for &(from, to) in &self.edges {
            if from >= n || to >= n {
                return Err(GraphBuildError::CycleDetected);
            }
            if from != to {
                neighbors[from].push(to);
                in_degree[to] += 1;
            }
        }

        let mut current: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut sorted: Vec<usize> = Vec::with_capacity(n);
        let mut topo_levels = 0usize;

        while !current.is_empty() {
            topo_levels += 1;
            let mut next_level: Vec<usize> = Vec::new();
            for &node in &current {
                sorted.push(node);
                for &neighbor in &neighbors[node] {
                    in_degree[neighbor] -= 1;
                    if in_degree[neighbor] == 0 {
                        next_level.push(neighbor);
                    }
                }
            }
            current = next_level;
        }

        if sorted.len() != n {
            return Err(GraphBuildError::CycleDetected);
        }

        let mut cumulative_writes: HashSet<ResourceSlot> = HashSet::new();
        for &pass_idx in &sorted {
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

        let needs_surface_acquire = self
            .passes
            .iter()
            .any(|p| p.resources().writes.contains(&ResourceSlot::Backbuffer));

        let mut pass_take: Vec<Option<Box<dyn RenderPass>>> =
            self.passes.into_iter().map(Some).collect();
        let mut ordered_passes: Vec<Box<dyn RenderPass>> = Vec::with_capacity(n);
        for &idx in &sorted {
            let p = pass_take[idx]
                .take()
                .expect("pass index taken once during build");
            ordered_passes.push(p);
        }

        Ok(CompiledRenderGraph {
            passes: ordered_passes,
            needs_surface_acquire,
            compile_stats: CompileStats {
                pass_count: n,
                topo_levels,
            },
        })
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_graph::context::RenderPassContext;
    use crate::render_graph::error::RenderPassError;
    use crate::render_graph::resources::PassResources;

    struct TestPass {
        name: &'static str,
        resources: PassResources,
    }

    impl RenderPass for TestPass {
        fn name(&self) -> &str {
            self.name
        }

        fn resources(&self) -> PassResources {
            self.resources.clone()
        }

        fn execute(&mut self, _ctx: &mut RenderPassContext<'_>) -> Result<(), RenderPassError> {
            Ok(())
        }
    }

    fn pass_w(name: &'static str, writes: &[ResourceSlot]) -> Box<dyn RenderPass> {
        Box::new(TestPass {
            name,
            resources: PassResources {
                reads: Vec::new(),
                writes: writes.to_vec(),
            },
        })
    }

    fn pass_rw(
        name: &'static str,
        reads: &[ResourceSlot],
        writes: &[ResourceSlot],
    ) -> Box<dyn RenderPass> {
        Box::new(TestPass {
            name,
            resources: PassResources {
                reads: reads.to_vec(),
                writes: writes.to_vec(),
            },
        })
    }

    #[test]
    fn linear_chain_validates() {
        let mut b = GraphBuilder::new();
        let a = b.add_pass(pass_w("a", &[ResourceSlot::Color]));
        let b_id = b.add_pass(pass_rw(
            "b",
            &[ResourceSlot::Color],
            &[ResourceSlot::Surface],
        ));
        b.add_edge(a, b_id);
        let g = b.build().expect("build");
        assert_eq!(g.compile_stats.pass_count, 2);
        assert_eq!(g.compile_stats.topo_levels, 2);
        assert!(!g.needs_surface_acquire);
    }

    #[test]
    fn missing_dependency_errs() {
        let mut b = GraphBuilder::new();
        b.add_pass(pass_rw("orphan_reader", &[ResourceSlot::Color], &[]));
        assert!(matches!(
            b.build(),
            Err(GraphBuildError::MissingDependency {
                slot: ResourceSlot::Color,
                ..
            })
        ));
    }

    #[test]
    fn cycle_detected() {
        let mut b = GraphBuilder::new();
        let p0 = b.add_pass(pass_w("p0", &[]));
        let p1 = b.add_pass(pass_w("p1", &[]));
        b.add_edge(p0, p1);
        b.add_edge(p1, p0);
        assert!(matches!(b.build(), Err(GraphBuildError::CycleDetected)));
    }

    #[test]
    fn add_pass_if_skips_when_false() {
        let mut b = GraphBuilder::new();
        b.add_pass(pass_w("root", &[ResourceSlot::Color]));
        assert!(b
            .add_pass_if(false, pass_w("skipped", &[ResourceSlot::Surface]))
            .is_none());
        let g = b.build().expect("build");
        assert_eq!(g.compile_stats.pass_count, 1);
    }

    #[test]
    fn forward_without_mesh_deform_producer_missing_dependency() {
        let mut b = GraphBuilder::new();
        let clustered = b.add_pass(pass_w(
            "clustered_like",
            &[ResourceSlot::ClusterBuffers, ResourceSlot::LightBuffer],
        ));
        let forward_like = b.add_pass(pass_rw(
            "forward_like",
            &[
                ResourceSlot::ClusterBuffers,
                ResourceSlot::LightBuffer,
                ResourceSlot::MeshDeformOutputs,
            ],
            &[ResourceSlot::Backbuffer],
        ));
        b.add_edge(clustered, forward_like);
        assert!(matches!(
            b.build(),
            Err(GraphBuildError::MissingDependency {
                slot: ResourceSlot::MeshDeformOutputs,
                ..
            })
        ));
    }

    #[test]
    fn parallel_passes_single_level() {
        let mut b = GraphBuilder::new();
        let a = b.add_pass(pass_w("a", &[ResourceSlot::Color]));
        let c = b.add_pass(pass_w("c", &[ResourceSlot::Depth]));
        let b_id = b.add_pass(pass_rw(
            "b",
            &[ResourceSlot::Color],
            &[ResourceSlot::Surface],
        ));
        let d = b.add_pass(pass_rw(
            "d",
            &[ResourceSlot::Depth],
            &[ResourceSlot::Surface],
        ));
        b.add_edge(a, b_id);
        b.add_edge(c, d);
        let g = b.build().expect("build");
        assert_eq!(g.compile_stats.topo_levels, 2);
        assert_eq!(g.compile_stats.pass_count, 4);
    }
}
