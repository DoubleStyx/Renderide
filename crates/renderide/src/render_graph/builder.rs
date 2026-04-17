//! DAG builder: topological sort, cycle detection, auto-derived edges, and producer/consumer validation.

use std::collections::{HashMap, HashSet};

use super::compiled::{CompileStats, CompiledRenderGraph};
use super::error::GraphBuildError;
use super::handles::ResourceId;
use super::ids::PassId;
use super::module::RenderModule;
use super::pass::RenderPass;
use super::resources::PassResources;

/// Builder for a directed acyclic graph of render passes.
///
/// Declare logical resources with [`Self::import`] / [`Self::create_transient`], register passes,
/// then [`Self::build`]. Edges are **auto-derived** from read/write declarations (last writer →
/// reader); use [`Self::add_edge`] only when explicit ordering is required beyond resource flow.
pub struct GraphBuilder {
    passes: Vec<Box<dyn RenderPass>>,
    edges: Vec<(usize, usize)>,
    resources: Vec<super::handles::ResourceDesc>,
}

impl GraphBuilder {
    /// Empty builder with no passes, edges, or resource registry entries.
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            edges: Vec::new(),
            resources: Vec::new(),
        }
    }

    fn alloc_resource(&mut self, desc: super::handles::ResourceDesc) -> ResourceId {
        let next = (self.resources.len() + 1) as u32;
        self.resources.push(desc);
        ResourceId::from_index_one_based(next)
    }

    /// Registers an externally owned logical resource (swapchain, depth, frame buffer).
    pub fn import(&mut self, desc: super::handles::ResourceDesc) -> ResourceId {
        self.alloc_resource(desc)
    }

    /// Registers a transient logical resource (metadata only until an allocator exists).
    pub fn create_transient(&mut self, desc: super::handles::ResourceDesc) -> ResourceId {
        self.alloc_resource(desc)
    }

    /// Runs [`RenderModule::register`] for `module`.
    pub fn register_module(
        &mut self,
        module: Box<dyn RenderModule>,
        handles: &super::handles::SharedRenderHandles,
    ) {
        module.register(self, handles);
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

    fn validate_pass_resources(&self, res: &PassResources) -> Result<(), GraphBuildError> {
        for &r in res.reads.iter().chain(res.writes.iter()) {
            if r.index() >= self.resources.len() {
                return Err(GraphBuildError::UnknownResource(r));
            }
        }
        Ok(())
    }

    fn resource_name(&self, id: ResourceId) -> &'static str {
        self.resources[id.index()].name
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

        for i in 0..n {
            self.validate_pass_resources(&self.passes[i].resources())?;
        }

        let mut last_writer: HashMap<ResourceId, usize> = HashMap::new();
        let mut auto_edges: Vec<(usize, usize)> = Vec::new();
        for pass_idx in 0..n {
            let pr = self.passes[pass_idx].resources();
            for &read in &pr.reads {
                if let Some(&w) = last_writer.get(&read) {
                    if w != pass_idx {
                        auto_edges.push((w, pass_idx));
                    }
                }
            }
            for &wrote in &pr.writes {
                last_writer.insert(wrote, pass_idx);
            }
        }

        let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
        for e in self.edges.iter().copied().chain(auto_edges) {
            if e.0 < n && e.1 < n && e.0 != e.1 {
                edge_set.insert(e);
            }
        }
        let merged_edges: Vec<(usize, usize)> = edge_set.into_iter().collect();

        let mut in_degree = vec![0usize; n];
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];

        for &(from, to) in &merged_edges {
            if from >= n || to >= n {
                return Err(GraphBuildError::CycleDetected);
            }
            neighbors[from].push(to);
            in_degree[to] += 1;
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

        let mut cumulative_writes: HashSet<ResourceId> = HashSet::new();
        for &pass_idx in &sorted {
            let resources = self.passes[pass_idx].resources();
            for &slot in &resources.reads {
                if !cumulative_writes.contains(&slot) {
                    return Err(GraphBuildError::MissingDependency {
                        pass: PassId(pass_idx),
                        resource: slot,
                        name: self.resource_name(slot),
                    });
                }
            }
            cumulative_writes.extend(resources.writes.iter().copied());
        }

        let needs_surface_acquire = self.passes.iter().any(|p| {
            p.resources()
                .writes
                .iter()
                .any(|&rid| self.resource_name(rid) == "backbuffer")
        });

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
    use crate::render_graph::cache::GraphCacheKey;
    use crate::render_graph::context::RenderPassContext;
    use crate::render_graph::error::RenderPassError;
    use crate::render_graph::handles::{ResourceDesc, SharedRenderHandles};
    use wgpu::{TextureFormat, TextureUsages};

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

    fn dummy_key() -> GraphCacheKey {
        GraphCacheKey {
            surface_extent: (800, 600),
            msaa_sample_count: 1,
            multiview_stereo: false,
            surface_format: TextureFormat::Bgra8UnormSrgb,
        }
    }

    fn pass_w(name: &'static str, writes: &[ResourceId]) -> Box<dyn RenderPass> {
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
        reads: &[ResourceId],
        writes: &[ResourceId],
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
    fn linear_chain_auto_edges_validate() {
        let mut b = GraphBuilder::new();
        let color = b.import(ResourceDesc::transient_texture("color"));
        let surface = b.import(ResourceDesc::imported_texture(
            "surface",
            Some(TextureFormat::Rgba8Unorm),
            None,
            TextureUsages::RENDER_ATTACHMENT,
        ));
        b.add_pass(pass_w("a", &[color]));
        b.add_pass(pass_rw("b", &[color], &[surface]));
        let g = b.build().expect("build");
        assert_eq!(g.compile_stats.pass_count, 2);
        assert_eq!(g.compile_stats.topo_levels, 2);
        assert!(!g.needs_surface_acquire());
    }

    #[test]
    fn missing_dependency_errs() {
        let mut b = GraphBuilder::new();
        let color = b.import(ResourceDesc::transient_texture("color"));
        b.add_pass(pass_rw("orphan_reader", &[color], &[]));
        assert!(matches!(
            b.build(),
            Err(GraphBuildError::MissingDependency { name: "color", .. })
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
        let color = b.import(ResourceDesc::transient_texture("color"));
        b.add_pass(pass_w("root", &[color]));
        let surface = b.import(ResourceDesc::imported_texture(
            "surface",
            None,
            None,
            TextureUsages::RENDER_ATTACHMENT,
        ));
        assert!(b
            .add_pass_if(false, pass_w("skipped", &[surface]))
            .is_none());
        let g = b.build().expect("build");
        assert_eq!(g.compile_stats.pass_count, 1);
    }

    #[test]
    fn forward_without_mesh_deform_producer_missing_dependency() {
        let mut b = GraphBuilder::new();
        let key = dummy_key();
        let h = SharedRenderHandles::declare(&mut b, key);
        let clustered = b.add_pass(pass_w(
            "clustered_like",
            &[h.cluster_buffers, h.light_buffer],
        ));
        let forward_like = b.add_pass(pass_rw(
            "forward_like",
            &[h.cluster_buffers, h.light_buffer, h.mesh_deform_outputs],
            &[h.backbuffer],
        ));
        b.add_edge(clustered, forward_like);
        assert!(matches!(
            b.build(),
            Err(GraphBuildError::MissingDependency {
                name: "mesh_deform_outputs",
                ..
            })
        ));
    }

    #[test]
    fn parallel_passes_single_level() {
        let mut b = GraphBuilder::new();
        let color = b.import(ResourceDesc::transient_texture("color"));
        let depth = b.import(ResourceDesc::transient_texture("depth"));
        let surface = b.import(ResourceDesc::imported_texture(
            "surface",
            None,
            None,
            TextureUsages::RENDER_ATTACHMENT,
        ));
        let a = b.add_pass(pass_w("a", &[color]));
        let c = b.add_pass(pass_w("c", &[depth]));
        let b_id = b.add_pass(pass_rw("b", &[color], &[surface]));
        let d = b.add_pass(pass_rw("d", &[depth], &[surface]));
        b.add_edge(a, b_id);
        b.add_edge(c, d);
        let g = b.build().expect("build");
        assert_eq!(g.compile_stats.topo_levels, 2);
        assert_eq!(g.compile_stats.pass_count, 4);
    }

    #[test]
    fn auto_edge_writer_to_reader_without_manual_edge() {
        let mut b = GraphBuilder::new();
        let x = b.import(ResourceDesc::transient_texture("x"));
        b.add_pass(pass_w("producer", &[x]));
        b.add_pass(pass_rw("consumer", &[x], &[]));
        let g = b.build().expect("build");
        assert_eq!(g.compile_stats.pass_count, 2);
        assert_eq!(g.compile_stats.topo_levels, 2);
    }
}
