//! Render graph: [`RenderPass`], [`GraphBuilder`], [`RenderGraph`], and pass implementations.
//!
//! All frame rendering for the main window and offscreen tasks goes through [`RenderGraph::execute`]
//! (see [`crate::render::RenderLoop`]). This module defines the DAG, resource declarations,
//! build-time validation, and execution order.
//!
//! # DAG structure
//!
//! - **Nodes:** Each [`GraphBuilder::add_pass`] adds a leaf pass and returns a [`PassId`].
//!   [`GraphBuilder::add_subgraph`] adds a nested [`RenderGraph`] as one node and returns a
//!   [`SubgraphId`].
//! - **Edges:** [`GraphBuilder::add_edge`] takes two [`GraphNodeId`] values ([`PassId`] and/or
//!   [`SubgraphId`] via [`From`]).
//! - **Topological order:** [`GraphBuilder::build`] runs Kahn’s algorithm over the mixed node list.
//!   If not all nodes can be ordered, the build returns [`GraphBuildError::CycleDetected`].
//! - **Construction:** Schedulable graphs with correct edges and validation must be produced by
//!   [`GraphBuilder`] (or helpers such as [`build_main_render_graph`]). Use [`RenderGraph::new`]
//!   only for an empty graph (e.g. as a building block); do not append passes without the builder.
//!
//! # Resource slots
//!
//! [`ResourceSlot`] describes abstract inputs and outputs (G-buffer color, depth, surface, AO
//! textures, clustered buffers, light buffer). Passes implement [`RenderPass::resources`] with
//! [`PassResources`] (reads / writes). Texture-backed slots drive attachment and sampling wiring
//! (see [Resource barriers](#resource-barriers-between-passes) below).
//! [`ResourceSlot::ClusterBuffers`]
//! and [`ResourceSlot::LightBuffer`] are logical slots; GPU handles live in caches and are not
//! passed through [`wgpu::CommandEncoder::transition_resources`].
//!
//! # Build-time validation
//!
//! After sorting, the builder walks nodes in execution order and tracks cumulative **writes**.
//! For each **leaf** pass, every slot in `reads` must already appear in that set; otherwise the
//! build returns [`GraphBuildError::MissingDependency`]. A **subgraph** node contributes the union
//! of all slots written anywhere inside it (`declared_writes_recursive`), so a pass after a
//! subgraph can legally read outputs produced inside the nested graph.
//!
//! # Execution order
//!
//! [`RenderGraph::execute`] creates one [`wgpu::CommandEncoder`] per graph invocation, prepares mesh
//! draws and TLAS as needed, acquires a ring-buffer frame index via [`PipelineManager::acquire_frame_index`],
//! then runs the internal schedule walker, which
//! walks [`ExecutionUnit`] in order: each [`ExecutionUnit::Pass`] runs [`RenderPass::execute`]; each
//! [`ExecutionUnit::Subgraph`] recurses into the nested [`LabeledSubgraph::graph`] on the **same**
//! encoder.
//! Each [`RenderGraph`] instance may own an RTAO MRT cache when [`RenderGraphContext::enable_rtao_mrt`]
//! is true; after its units run, if MRT color exists and no pass in that graph writes
//! [`ResourceSlot::Surface`], an MRT→target copy is recorded (RTAO path, not a graph bypass).
//! [`GraphBuilder::build_with_special_passes`] records composite and overlay
//! [`PassId`]s for attachment routing.
//!
//! # Subgraphs
//!
//! See [`GraphBuilder::add_subgraph`]. Nested graphs keep their own passes, slot declarations, and
//! RTAO cache; each root [`RenderGraph::execute`] still performs a single queue submit for that graph.
//!
//! # Resource barriers between passes
//!
//! All passes for a frame record into **one** [`wgpu::CommandEncoder`]. Wgpu infers texture layouts
//! from render pass and compute pass descriptors within that encoder (same as the pre–render-graph
//! loop). [`PassResources`] are used for build-time dependency checks and for wiring
//! [`RenderTargetViews`], not for inserting [`wgpu::CommandEncoder::transition_resources`] between
//! passes—manual transitions here previously forced depth into states incompatible with the overlay
//! pass (depth as a load/store attachment after compute had sampled the same texture).
//!
//! The MRT→surface [`wgpu::CommandEncoder::copy_texture_to_texture`] runs when RTAO MRT color exists
//! and no pass in that graph wrote [`ResourceSlot::Surface`], as described under
//! [Execution order](#execution-order).
//!
//! Cluster and light buffers are ordered by pass sequence and wgpu’s buffer tracking.

mod error;
mod mesh_draw;
mod mesh_prep;

mod clustered_light;
mod composite;
mod mesh_pass;
mod overlay_pass;
mod projection;
mod rtao_blur;
mod rtao_compute;

mod graph;

pub use error::RenderPassError;
pub use graph::{
    ExecutionUnit, GraphBuildError, GraphBuilder, GraphNodeId, LabeledSubgraph, MrtViews, PassId,
    PassResources, RenderGraph, RenderGraphContext, RenderPass, RenderPassContext,
    RenderTargetViews, ResourceSlot, SubgraphId, SubgraphLabel, build_main_render_graph,
};
pub use mesh_prep::{MeshDrawPrepStats, PreCollectedFrameData, prepare_mesh_draws_for_view};

pub use clustered_light::ClusteredLightPass;
pub use composite::CompositePass;
pub use mesh_pass::MeshRenderPass;
pub use overlay_pass::OverlayRenderPass;
pub use projection::{
    orthographic_projection_reverse_z, projection_for_params, reverse_z_projection,
};
pub use rtao_blur::RtaoBlurPass;
pub use rtao_compute::RtaoComputePass;
