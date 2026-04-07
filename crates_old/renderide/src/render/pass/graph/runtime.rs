//! Graph schedule types and recursive write tracking for build-time validation.

use std::collections::HashSet;

use super::ids::{PassId, SubgraphLabel};
use super::pass_trait::RenderPass;
use super::resources::{PassResources, ResourceSlot};

/// Nested [`RenderGraph`] with the label from [`crate::render::pass::graph::GraphBuilder::add_subgraph`].
///
/// Stored behind [`Box`] so [`ExecutionUnit`] and [`RenderGraph`] can be mutually recursive.
pub struct LabeledSubgraph {
    /// Debug / scheduling label (e.g. `"main_view"`).
    pub label: SubgraphLabel,
    /// Nested graph run as a single step in the parent schedule.
    pub graph: Box<RenderGraph>,
}

/// One step in a [`RenderGraph`] schedule: a leaf pass or a nested subgraph.
pub enum ExecutionUnit {
    /// Leaf [`RenderPass`] with [`PassResources`] snapshot from build time.
    Pass {
        /// Pass implementation.
        pass: Box<dyn RenderPass>,
        /// Declared reads and writes (from [`RenderPass::resources`] when the graph was built).
        resources: PassResources,
    },
    /// Nested [`RenderGraph`] (see [`LabeledSubgraph`]); owns its own passes, slots, and RTAO cache.
    Subgraph(LabeledSubgraph),
}

/// Graph of render passes (and optional subgraphs) executed each frame.
///
/// RTAO MRT textures are owned on **this** graph instance when it contains passes that use them;
/// nested [`ExecutionUnit::Subgraph`] graphs keep separate caches. [`execute`](RenderGraph::execute) on
/// the root records one [`wgpu::CommandEncoder`] per frame; subgraphs append to the same encoder.
pub struct RenderGraph {
    /// Resource cache for RTAO slots ([`ResourceSlot::Color`], [`ResourceSlot::Position`], etc.):
    /// one bundled [`crate::gpu::rtao_textures::RtaoTextureCache`], recreated when viewport or
    /// [`GpuState::config`](crate::gpu::GpuState::config) color format changes. Cleared when `enable_rtao_mrt` is false.
    pub(crate) rtao_mrt_cache: Option<crate::gpu::rtao_textures::RtaoTextureCache>,
    /// Topological execution order: passes and subgraph nodes.
    pub(crate) execution: Vec<ExecutionUnit>,
    /// [`PassId`]s in execution order for **leaf passes only** at this level (for special-pass ids).
    #[allow(dead_code)]
    pub(crate) execution_order_pass_ids: Vec<PassId>,
    /// Resource declarations for each leaf pass at this level, same order as `execution_order_pass_ids`.
    /// Populated at build time; read by the `pass_resources` test helper. Retained for introspection.
    #[allow(dead_code)]
    pub(crate) pass_resources: Vec<PassResources>,
    /// PassId of the composite pass, if present. Exposed for tests.
    #[allow(dead_code)]
    pub(crate) composite_pass_id: Option<PassId>,
    /// PassId of the overlay pass, if present. Exposed for tests.
    #[allow(dead_code)]
    pub(crate) overlay_pass_id: Option<PassId>,
}

/// Union of all resource slots written anywhere inside `graph` (including nested subgraphs).
///
/// Used at build time so a pass after a subgraph can validate reads against the subgraph’s outputs.
pub(super) fn declared_writes_recursive(graph: &RenderGraph) -> HashSet<ResourceSlot> {
    graph
        .execution
        .iter()
        .fold(HashSet::new(), |mut acc, unit| {
            match unit {
                ExecutionUnit::Pass { resources, .. } => {
                    acc.extend(resources.writes.iter().copied());
                }
                ExecutionUnit::Subgraph(labeled) => {
                    acc.extend(declared_writes_recursive(&labeled.graph));
                }
            }
            acc
        })
}
