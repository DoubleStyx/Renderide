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

mod clustered_light;
mod composite;
mod mesh_draw;
mod mesh_pass;
mod overlay_pass;
mod projection;
mod rtao_blur;
mod rtao_compute;

use std::borrow::Cow;
use std::collections::HashSet;

use nalgebra::Matrix4;

use super::SpaceDrawBatch;
use super::target::RenderTarget;
use super::view::ViewParams;
use crate::session::Session;
use mesh_draw::{CollectMeshDrawsContext, collect_mesh_draws};

pub use clustered_light::ClusteredLightPass;
pub use composite::CompositePass;
pub use mesh_pass::MeshRenderPass;
pub use overlay_pass::OverlayRenderPass;
pub use projection::{
    orthographic_projection_reverse_z, projection_for_params, reverse_z_projection,
};
pub use rtao_blur::RtaoBlurPass;
pub use rtao_compute::RtaoComputePass;

/// Cached mesh draws: (non_overlay_skinned, overlay_skinned, non_overlay_non_skinned, overlay_non_skinned).
pub(crate) type CachedMeshDraws = (
    Vec<mesh_draw::SkinnedBatchedDraw>,
    Vec<mesh_draw::SkinnedBatchedDraw>,
    Vec<mesh_draw::BatchedDraw>,
    Vec<mesh_draw::BatchedDraw>,
);

/// CPU mesh-draw prep counters for one frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeshDrawPrepStats {
    /// Total draws visited across all batches before mesh/GPU validation.
    pub total_input_draws: usize,
    /// Total non-skinned draws visited.
    pub rigid_input_draws: usize,
    /// Total skinned draws visited.
    pub skinned_input_draws: usize,
    /// Submitted rigid draws after CPU culling/validation.
    pub submitted_rigid_draws: usize,
    /// Submitted skinned draws after validation.
    pub submitted_skinned_draws: usize,
    /// Rigid draws rejected by CPU frustum culling.
    pub frustum_culled_rigid_draws: usize,
    /// Skinned draws rejected by CPU frustum culling (bone-position AABB test).
    pub frustum_culled_skinned_draws: usize,
    /// Rigid draws kept because upload bounds were degenerate, so culling was skipped.
    pub skipped_cull_degenerate_bounds: usize,
    /// Draws skipped because `mesh_asset_id < 0`.
    pub skipped_invalid_mesh_asset_id: usize,
    /// Draws skipped because the mesh asset was not found.
    pub skipped_missing_mesh_asset: usize,
    /// Draws skipped because the mesh had no vertices or indices.
    pub skipped_empty_mesh: usize,
    /// Draws skipped because GPU buffers were not resident.
    pub skipped_missing_gpu_buffers: usize,
    /// Skinned draws skipped because bind poses were missing.
    pub skipped_skinned_missing_bind_poses: usize,
    /// Skinned draws skipped because bone IDs were missing or empty.
    pub skipped_skinned_missing_bone_ids: usize,
    /// Skinned draws skipped because bone ID count exceeded bind pose count.
    pub skipped_skinned_id_count_mismatch: usize,
    /// Skinned draws skipped because the skinned vertex buffer was missing.
    pub skipped_skinned_missing_vertex_buffer: usize,
}

impl MeshDrawPrepStats {
    /// Total draws submitted after prep.
    pub fn submitted_draws(&self) -> usize {
        self.submitted_rigid_draws + self.submitted_skinned_draws
    }

    fn accumulate(&mut self, other: &Self) {
        self.total_input_draws += other.total_input_draws;
        self.rigid_input_draws += other.rigid_input_draws;
        self.skinned_input_draws += other.skinned_input_draws;
        self.submitted_rigid_draws += other.submitted_rigid_draws;
        self.submitted_skinned_draws += other.submitted_skinned_draws;
        self.frustum_culled_rigid_draws += other.frustum_culled_rigid_draws;
        self.frustum_culled_skinned_draws += other.frustum_culled_skinned_draws;
        self.skipped_cull_degenerate_bounds += other.skipped_cull_degenerate_bounds;
        self.skipped_invalid_mesh_asset_id += other.skipped_invalid_mesh_asset_id;
        self.skipped_missing_mesh_asset += other.skipped_missing_mesh_asset;
        self.skipped_empty_mesh += other.skipped_empty_mesh;
        self.skipped_missing_gpu_buffers += other.skipped_missing_gpu_buffers;
        self.skipped_skinned_missing_bind_poses += other.skipped_skinned_missing_bind_poses;
        self.skipped_skinned_missing_bone_ids += other.skipped_skinned_missing_bone_ids;
        self.skipped_skinned_id_count_mismatch += other.skipped_skinned_id_count_mismatch;
        self.skipped_skinned_missing_vertex_buffer += other.skipped_skinned_missing_vertex_buffer;
    }
}

/// Reference to cached mesh draws for render pass context.
pub(crate) type CachedMeshDrawsRef<'a> = (
    &'a [mesh_draw::SkinnedBatchedDraw],
    &'a [mesh_draw::SkinnedBatchedDraw],
    &'a [mesh_draw::BatchedDraw],
    &'a [mesh_draw::BatchedDraw],
);

/// Runs mesh-draw CPU collection for the main view and graph fallback paths.
///
/// [`Session`] is not [`Sync`] today (IPC queues), so per-batch worker threads cannot safely share
/// `&Session` yet. [`RenderConfig::parallel_mesh_draw_prep_batches`] is reserved for when prep uses
/// owned snapshots or the session becomes shareable for read-only prep.
fn run_collect_mesh_draws(
    session: &Session,
    draw_batches: &[SpaceDrawBatch],
    gpu: &crate::gpu::GpuState,
    proj: Matrix4<f32>,
    overlay_projection_override: Option<ViewParams>,
) -> (CachedMeshDraws, MeshDrawPrepStats) {
    let collect_ctx = CollectMeshDrawsContext {
        session,
        draw_batches,
        gpu,
        proj,
        overlay_projection_override,
    };
    let (non_overlay_skinned, overlay_skinned, non_overlay_non_skinned, overlay_non_skinned, stats) =
        collect_mesh_draws(&collect_ctx);
    (
        (
            non_overlay_skinned,
            overlay_skinned,
            non_overlay_non_skinned,
            overlay_non_skinned,
        ),
        stats,
    )
}

/// Pre-collected mesh draws and view parameters for the main view.
///
/// Produced by [`prepare_mesh_draws_for_view`] during the collect phase for the same
/// render extent as the [`crate::render::RenderTarget`] passed into [`RenderLoop::render_frame`]
/// (typically the acquired swapchain texture size, not window client area alone).
pub struct PreCollectedFrameData {
    /// Primary projection matrix for the main view.
    pub proj: Matrix4<f32>,
    /// Overlay projection override when overlays use orthographic.
    pub overlay_projection_override: Option<ViewParams>,
    /// Cached mesh draws for mesh and overlay passes.
    pub(crate) cached_mesh_draws: CachedMeshDraws,
    /// CPU-side mesh draw preparation counters for diagnostics.
    pub prep_stats: MeshDrawPrepStats,
}

/// Prepares mesh draws for the main view during the collect phase.
///
/// `viewport` must match the width and height of the swapchain (or other color target)
/// that will be rendered to in the same frame, so projection and cached draws agree
/// with the GPU viewport.
///
/// Runs [`ensure_mesh_buffers`] and [`run_collect_mesh_draws`] so this CPU work
/// is measured in the collect phase rather than the render phase.
pub fn prepare_mesh_draws_for_view(
    gpu: &mut crate::gpu::GpuState,
    session: &Session,
    draw_batches: &[SpaceDrawBatch],
    viewport: (u32, u32),
) -> PreCollectedFrameData {
    ensure_mesh_buffers(gpu, session, draw_batches);
    let (width, height) = viewport;
    let aspect = width as f32 / height.max(1) as f32;
    let view_params = ViewParams::perspective_from_session(session, aspect);
    let proj = view_params.to_projection_matrix();
    let overlay_projection_override =
        ViewParams::overlay_projection_for_frame(session, draw_batches, aspect);
    let (cached_mesh_draws, prep_stats) = run_collect_mesh_draws(
        session,
        draw_batches,
        gpu,
        proj,
        overlay_projection_override.clone(),
    );
    PreCollectedFrameData {
        proj,
        overlay_projection_override,
        cached_mesh_draws,
        prep_stats,
    }
}

/// Errors that can occur during render pass execution.
#[derive(Debug)]
pub enum RenderPassError {
    /// Wrapper for wgpu surface errors when acquiring the current texture.
    Surface(wgpu::SurfaceError),
    /// Cached mesh draws were not provided to the pass.
    MissingCachedMeshDraws,
    /// MRT views were required but not provided.
    MissingMrtViews,
}

impl From<wgpu::SurfaceError> for RenderPassError {
    fn from(e: wgpu::SurfaceError) -> Self {
        RenderPassError::Surface(e)
    }
}

/// Opaque identifier for a pass in the graph. Used for declaring edges.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PassId(usize);

/// Human-readable label for a subgraph node (e.g. `"main_view"`, `"reflection_probe"`).
///
/// Used for debugging and for namespaced pass names in tests (`label/pass_name`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SubgraphLabel(Cow<'static, str>);

impl SubgraphLabel {
    /// Returns the label as a string slice.
    pub fn as_str(&self) -> &str {
        self.0.as_ref()
    }
}

impl From<&'static str> for SubgraphLabel {
    fn from(s: &'static str) -> Self {
        Self(Cow::Borrowed(s))
    }
}

impl From<String> for SubgraphLabel {
    fn from(s: String) -> Self {
        Self(Cow::Owned(s))
    }
}

/// Opaque identifier returned by [`GraphBuilder::add_subgraph`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubgraphId(pub usize);

/// Endpoint for [`GraphBuilder::add_edge`]: a root-level pass or a subgraph instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GraphNodeId {
    /// A pass added with [`GraphBuilder::add_pass`].
    Pass(PassId),
    /// A subgraph added with [`GraphBuilder::add_subgraph`].
    Subgraph(SubgraphId),
}

impl From<PassId> for GraphNodeId {
    fn from(value: PassId) -> Self {
        GraphNodeId::Pass(value)
    }
}

impl From<SubgraphId> for GraphNodeId {
    fn from(value: SubgraphId) -> Self {
        GraphNodeId::Subgraph(value)
    }
}

/// Resource slot identifier for pass resource declarations.
///
/// Passes declare which slots they read and write; the graph can use this for
/// validation and scheduling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceSlot {
    /// Clustered shading cluster buffers (counts, indices).
    ClusterBuffers,
    /// Light buffer for clustered shading.
    LightBuffer,
    /// MRT color texture (mesh pass output, composite input).
    Color,
    /// MRT position G-buffer.
    Position,
    /// MRT normal G-buffer.
    Normal,
    /// Raw AO texture (RTAO compute output, blur input).
    AoRaw,
    /// Blurred AO texture (blur output, composite input).
    Ao,
    /// Final surface output.
    Surface,
    /// Depth buffer.
    Depth,
}

/// Declared reads and writes for a render pass.
#[derive(Clone, Debug, Default)]
pub struct PassResources {
    /// Resource slots this pass reads from.
    pub reads: Vec<ResourceSlot>,
    /// Resource slots this pass writes to.
    pub writes: Vec<ResourceSlot>,
}

/// Errors that can occur when building a render graph.
#[derive(Debug, thiserror::Error)]
pub enum GraphBuildError {
    /// The graph contains a cycle; topological sort is impossible.
    #[error("cycle detected in render graph")]
    CycleDetected,

    /// A pass reads a resource slot that no earlier pass produces.
    #[error("pass {pass:?} reads {slot:?} but no earlier pass writes it")]
    MissingDependency {
        /// Pass that requires the missing dependency.
        pass: PassId,
        /// Resource slot that has no producer.
        slot: ResourceSlot,
    },
}

/// Maps resource slots to texture views for pass execution.
///
/// Built from target, MRT views, and depth. Slots may be `None` when RTAO is disabled.
struct SlotMap<'a> {
    color: Option<&'a wgpu::TextureView>,
    position: Option<&'a wgpu::TextureView>,
    normal: Option<&'a wgpu::TextureView>,
    ao_raw: Option<&'a wgpu::TextureView>,
    ao: Option<&'a wgpu::TextureView>,
    surface: &'a wgpu::TextureView,
    depth: Option<&'a wgpu::TextureView>,
}

/// Builds a slot-to-view map from the render target, MRT views, and depth override.
fn build_slot_map<'a>(
    target: &'a RenderTarget,
    mrt_views: Option<&'a MrtViews<'a>>,
    depth_view_override: Option<&'a wgpu::TextureView>,
) -> SlotMap<'a> {
    let surface = target.color_view();
    let depth = target.depth_view().or(depth_view_override);
    match mrt_views {
        Some(mrt) => SlotMap {
            color: Some(mrt.color_view),
            position: Some(mrt.position_view),
            normal: Some(mrt.normal_view),
            ao_raw: Some(mrt.ao_raw_view),
            ao: Some(mrt.ao_view),
            surface,
            depth,
        },
        None => SlotMap {
            color: None,
            position: None,
            normal: None,
            ao_raw: None,
            ao: None,
            surface,
            depth,
        },
    }
}

/// Computes [`RenderTargetViews`] for a pass from its resource declarations and the slot map.
fn render_target_views_for_pass<'a>(
    slot_map: &SlotMap<'a>,
    resources: Option<&PassResources>,
) -> RenderTargetViews<'a> {
    let uses = |slot: ResourceSlot| {
        resources.is_some_and(|r| r.reads.contains(&slot) || r.writes.contains(&slot))
    };
    let writes = |slot: ResourceSlot| resources.is_some_and(|r| r.writes.contains(&slot));

    let color_view = if writes(ResourceSlot::Surface) {
        slot_map.surface
    } else {
        slot_map.color.unwrap_or(slot_map.surface)
    };

    RenderTargetViews {
        color_view,
        depth_view: if uses(ResourceSlot::Depth) {
            slot_map.depth
        } else {
            None
        },
        mrt_position_view: if uses(ResourceSlot::Position) {
            slot_map.position
        } else {
            None
        },
        mrt_normal_view: if uses(ResourceSlot::Normal) {
            slot_map.normal
        } else {
            None
        },
        mrt_ao_raw_view: if uses(ResourceSlot::AoRaw) {
            slot_map.ao_raw
        } else {
            None
        },
        mrt_ao_view: if uses(ResourceSlot::Ao) {
            slot_map.ao
        } else {
            None
        },
        mrt_color_input_view: if resources.is_some_and(|r| r.reads.contains(&ResourceSlot::Color)) {
            slot_map.color
        } else {
            None
        },
    }
}

/// Target [`wgpu::TextureUses`] when the current pass reads `slot` as input (after a prior write).
///
/// Used by unit tests documenting intended transition semantics if explicit barriers are reintroduced
/// (e.g. for multi-submit batching).
#[cfg(test)]
fn texture_read_target_uses(slot: ResourceSlot, curr: &PassResources) -> Option<wgpu::TextureUses> {
    match slot {
        ResourceSlot::Color
        | ResourceSlot::Position
        | ResourceSlot::Normal
        | ResourceSlot::AoRaw
        | ResourceSlot::Ao => Some(wgpu::TextureUses::RESOURCE),
        ResourceSlot::Depth => {
            if curr.writes.contains(&ResourceSlot::Surface)
                && !curr.writes.contains(&ResourceSlot::Depth)
            {
                Some(wgpu::TextureUses::DEPTH_STENCIL_WRITE)
            } else {
                Some(wgpu::TextureUses::DEPTH_STENCIL_READ)
            }
        }
        ResourceSlot::Surface => Some(wgpu::TextureUses::RESOURCE),
        ResourceSlot::ClusterBuffers | ResourceSlot::LightBuffer => None,
    }
}

/// Color and optional depth texture views for the current render pass.
pub struct RenderTargetViews<'a> {
    /// Color attachment view (output for this pass).
    pub color_view: &'a wgpu::TextureView,
    /// Optional depth attachment view.
    pub depth_view: Option<&'a wgpu::TextureView>,
    /// When RTAO is enabled, position G-buffer view for MRT mesh pass.
    pub mrt_position_view: Option<&'a wgpu::TextureView>,
    /// When RTAO is enabled, normal G-buffer view for MRT mesh pass.
    pub mrt_normal_view: Option<&'a wgpu::TextureView>,
    /// When RTAO is enabled, raw AO texture view (RTAO output, blur input).
    pub mrt_ao_raw_view: Option<&'a wgpu::TextureView>,
    /// When RTAO is enabled, AO texture view for blur output and composite.
    pub mrt_ao_view: Option<&'a wgpu::TextureView>,
    /// When RTAO is enabled, mesh color input for composite pass (MRT color texture).
    pub mrt_color_input_view: Option<&'a wgpu::TextureView>,
}

/// Per-pass context passed to `RenderPass::execute`.
pub struct RenderPassContext<'a> {
    /// GPU state including device, queue, mesh cache, and depth texture.
    pub gpu: &'a mut crate::gpu::GpuState,
    /// Session for scene, assets, and view state.
    pub session: &'a Session,
    /// Draw batches for this frame.
    pub draw_batches: &'a [SpaceDrawBatch],
    /// Pipeline manager for mesh pipelines.
    pub pipeline_manager: &'a mut crate::gpu::PipelineManager,
    /// Frame index for ring buffer offset; advanced once per frame by the graph.
    pub frame_index: u64,
    /// Viewport dimensions (width, height).
    pub viewport: (u32, u32),
    /// Primary projection matrix; passes build view-proj per batch as needed.
    pub proj: Matrix4<f32>,
    /// Optional overlay projection override. When `Some`, overlay batches use this instead of
    /// `proj` (e.g. orthographic for screen-space UI). Future: set from RenderConfig or host data.
    pub overlay_projection_override: Option<ViewParams>,
    /// Current color and depth attachments.
    pub render_target: RenderTargetViews<'a>,
    /// Command encoder for this frame; pass records into this.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// Optional timestamp query set for GPU pass timing.
    pub timestamp_query_set: Option<&'a wgpu::QuerySet>,
    /// Cached mesh draws from a single collect per frame. Mesh and overlay passes use this.
    pub(crate) cached_mesh_draws: Option<CachedMeshDrawsRef<'a>>,
}

/// MRT (Multiple Render Target) views for RTAO pass.
///
/// When RTAO is enabled, the mesh pass renders to these instead of the surface.
pub struct MrtViews<'a> {
    /// Color attachment view (matches surface format for copy-back).
    pub color_view: &'a wgpu::TextureView,
    /// Color texture for copy to surface (same as color_view's texture).
    pub color_texture: &'a wgpu::Texture,
    /// Position G-buffer view (Rgba16Float, camera-relative `world - view_position`).
    pub position_view: &'a wgpu::TextureView,
    /// Position G-buffer texture (for [`wgpu::CommandEncoder::transition_resources`]).
    pub position_texture: &'a wgpu::Texture,
    /// Normal G-buffer view (Rgba16Float).
    pub normal_view: &'a wgpu::TextureView,
    /// Normal G-buffer texture.
    pub normal_texture: &'a wgpu::Texture,
    /// Raw AO view (Rgba8Unorm). Written by RTAO compute, read by blur pass.
    pub ao_raw_view: &'a wgpu::TextureView,
    /// Raw AO texture.
    pub ao_raw_texture: &'a wgpu::Texture,
    /// AO output view (Rgba8Unorm). Written by blur pass, read by composite.
    pub ao_view: &'a wgpu::TextureView,
    /// Blurred AO texture.
    pub ao_texture: &'a wgpu::Texture,
}

/// Frame-level context created when executing the main-view render graph.
pub struct RenderGraphContext<'a> {
    /// GPU state.
    pub gpu: &'a mut crate::gpu::GpuState,
    /// Session.
    pub session: &'a Session,
    /// Draw batches.
    pub draw_batches: &'a [SpaceDrawBatch],
    /// Pipeline manager.
    pub pipeline_manager: &'a mut crate::gpu::PipelineManager,
    /// Render target (surface or offscreen).
    pub target: &'a RenderTarget,
    /// Depth view for Surface targets; Offscreen provides its own. Dimensions must match target.
    pub depth_view_override: Option<&'a wgpu::TextureView>,
    /// Viewport (width, height); must match target dimensions.
    pub viewport: (u32, u32),
    /// Primary projection matrix.
    pub proj: Matrix4<f32>,
    /// Optional overlay projection override. When `Some`, overlay pass uses this instead of `proj`.
    pub overlay_projection_override: Option<ViewParams>,
    /// Optional timestamp query set for GPU pass timing.
    pub timestamp_query_set: Option<&'a wgpu::QuerySet>,
    /// Optional resolve buffer for timestamp readback.
    pub timestamp_resolve_buffer: Option<&'a wgpu::Buffer>,
    /// Optional staging buffer for timestamp readback.
    pub timestamp_staging_buffer: Option<&'a wgpu::Buffer>,
    /// When true, the graph allocates or reuses RTAO MRT textures from
    /// [`RenderGraph::rtao_mrt_cache`] using [`GpuState::config`] color format and [`viewport`](Self::viewport).
    /// Set false for offscreen paths that render without RTAO (e.g. [`crate::render::RenderLoop::render_to_target`]).
    pub enable_rtao_mrt: bool,
    /// Pre-collected mesh draws from the collect phase. When `Some`, skips collect in execute.
    pub(crate) pre_collected: Option<&'a CachedMeshDraws>,
    /// When set, invoked on the graph encoder after passes (and optional timestamp resolve) and before
    /// [`wgpu::Queue::submit`], so extra commands share the same submission (e.g. camera-task readback copy).
    pub before_submit: Option<&'a mut dyn FnMut(&mut wgpu::CommandEncoder)>,
}

/// Trait for render passes that can be executed by the render graph.
pub trait RenderPass {
    /// Human-readable name for debugging.
    fn name(&self) -> &str;

    /// Declares which resource slots this pass reads and writes.
    /// Default: empty reads and writes.
    fn resources(&self) -> PassResources {
        PassResources::default()
    }

    /// Executes the pass, recording commands into the context's encoder.
    fn execute(&mut self, ctx: &mut RenderPassContext) -> Result<(), RenderPassError>;
}

/// Nested [`RenderGraph`] with the label from [`GraphBuilder::add_subgraph`].
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
/// nested [`ExecutionUnit::Subgraph`] graphs keep separate caches. [`execute`](Self::execute) on
/// the root records one [`wgpu::CommandEncoder`] per frame; subgraphs append to the same encoder.
pub struct RenderGraph {
    /// Resource cache for RTAO slots ([`ResourceSlot::Color`], [`ResourceSlot::Position`], etc.):
    /// one bundled [`crate::gpu::rtao_textures::RtaoTextureCache`], recreated when viewport or
    /// [`GpuState::config`] color format changes. Cleared when `enable_rtao_mrt` is false.
    rtao_mrt_cache: Option<crate::gpu::rtao_textures::RtaoTextureCache>,
    /// Topological execution order: passes and subgraph nodes.
    execution: Vec<ExecutionUnit>,
    /// [`PassId`]s in execution order for **leaf passes only** at this level (for special-pass ids).
    #[allow(dead_code)]
    execution_order_pass_ids: Vec<PassId>,
    /// Resource declarations for each leaf pass at this level, same order as `execution_order_pass_ids`.
    /// Populated at build time; read by the `pass_resources` test helper. Retained for introspection.
    #[allow(dead_code)]
    pass_resources: Vec<PassResources>,
    /// PassId of the composite pass, if present. Exposed for tests.
    #[allow(dead_code)]
    composite_pass_id: Option<PassId>,
    /// PassId of the overlay pass, if present. Exposed for tests.
    #[allow(dead_code)]
    overlay_pass_id: Option<PassId>,
}

/// Union of all resource slots written anywhere inside `graph` (including nested subgraphs).
///
/// Used at build time so a pass after a subgraph can validate reads against the subgraph’s outputs.
fn declared_writes_recursive(graph: &RenderGraph) -> HashSet<ResourceSlot> {
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

/// Builds the main window render graph for the RTAO MRT variant or the direct-to-surface variant.
///
/// When `rtao_mrt_graph` is true, inserts [`RtaoComputePass`], [`RtaoBlurPass`], and
/// [`CompositePass`] between mesh and overlay, and uses [`MeshRenderPass::with_rtao_mrt_graph`]
/// with `true`. When false, the mesh pass uses `false` and writes color and depth to the
/// surface and edges mesh directly to overlay; RTAO passes are omitted.
///
/// [`RenderGraphContext::enable_rtao_mrt`] must be set to match this variant at execute time so
/// MRT textures are allocated only when the graph expects them.
pub fn build_main_render_graph(rtao_mrt_graph: bool) -> Result<RenderGraph, GraphBuildError> {
    let mut builder = GraphBuilder::new();
    let clustered = builder.add_pass(Box::new(ClusteredLightPass::new()));
    let mesh = builder.add_pass(Box::new(MeshRenderPass::with_rtao_mrt_graph(
        rtao_mrt_graph,
    )));
    let overlay = builder.add_pass(Box::new(OverlayRenderPass::new()));
    builder.add_edge(clustered, mesh);

    if rtao_mrt_graph {
        let rtao = builder.add_pass(Box::new(RtaoComputePass::new()));
        let rtao_blur = builder.add_pass(Box::new(RtaoBlurPass::new()));
        let composite = builder.add_pass(Box::new(CompositePass::new()));
        builder.add_edge(mesh, rtao);
        builder.add_edge(rtao, rtao_blur);
        builder.add_edge(rtao_blur, composite);
        builder.add_edge(composite, overlay);
        builder.build_with_special_passes(Some(composite), Some(overlay))
    } else {
        builder.add_edge(mesh, overlay);
        builder.build_with_special_passes(None, Some(overlay))
    }
}

impl RenderGraph {
    /// Creates an empty render graph.
    ///
    /// Schedulable graphs must be built with [`GraphBuilder`] (or [`build_main_render_graph`]) so
    /// edges and [`GraphBuildError`] validation apply. An empty graph is only useful as a nested
    /// placeholder or before the builder moves content in; it has no composite/overlay [`PassId`]
    /// metadata.
    pub fn new() -> Self {
        Self {
            rtao_mrt_cache: None,
            execution: Vec::new(),
            execution_order_pass_ids: Vec::new(),
            pass_resources: Vec::new(),
            composite_pass_id: None,
            overlay_pass_id: None,
        }
    }

    fn any_pass_writes_surface(&self) -> bool {
        self.execution.iter().any(|unit| match unit {
            ExecutionUnit::Pass { resources, .. } => {
                resources.writes.contains(&ResourceSlot::Surface)
            }
            ExecutionUnit::Subgraph(labeled) => labeled.graph.any_pass_writes_surface(),
        })
    }

    /// Runs this graph’s [`ExecutionUnit`] sequence on `encoder`. Used by [`execute`](Self::execute)
    /// and recursively for subgraphs.
    fn execute_scheduled_units(
        &mut self,
        ctx: &mut RenderGraphContext<'_>,
        encoder: &mut wgpu::CommandEncoder,
        frame_index: u64,
        cached_mesh_draws: Option<CachedMeshDrawsRef<'_>>,
    ) -> Result<(), RenderPassError> {
        let (width, height) = ctx.viewport;
        let color_format = ctx.gpu.config.format;
        if ctx.enable_rtao_mrt {
            let recreate = self
                .rtao_mrt_cache
                .as_ref()
                .is_none_or(|c| !c.matches_key(width, height, color_format));
            if recreate {
                self.rtao_mrt_cache = Some(crate::gpu::rtao_textures::RtaoTextureCache::create(
                    &ctx.gpu.device,
                    width,
                    height,
                    color_format,
                ));
            }
        } else {
            self.rtao_mrt_cache = None;
        }

        let mrt_views = self.rtao_mrt_cache.as_ref().map(|c| MrtViews {
            color_view: &c.color_view,
            color_texture: &c.color_texture,
            position_view: &c.position_view,
            position_texture: &c.position_texture,
            normal_view: &c.normal_view,
            normal_texture: &c.normal_texture,
            ao_raw_view: &c.ao_raw_view,
            ao_raw_texture: &c.ao_raw_texture,
            ao_view: &c.ao_view,
            ao_texture: &c.ao_texture,
        });

        for unit in &mut self.execution {
            match unit {
                ExecutionUnit::Pass { pass, resources } => {
                    let slot_map =
                        build_slot_map(ctx.target, mrt_views.as_ref(), ctx.depth_view_override);
                    let render_target = render_target_views_for_pass(&slot_map, Some(resources));

                    let mut pass_ctx = RenderPassContext {
                        gpu: ctx.gpu,
                        session: ctx.session,
                        draw_batches: ctx.draw_batches,
                        pipeline_manager: ctx.pipeline_manager,
                        frame_index,
                        viewport: ctx.viewport,
                        proj: ctx.proj,
                        overlay_projection_override: ctx.overlay_projection_override.clone(),
                        render_target,
                        encoder,
                        timestamp_query_set: ctx.timestamp_query_set,
                        cached_mesh_draws,
                    };
                    pass.execute(&mut pass_ctx)?;
                }
                ExecutionUnit::Subgraph(labeled) => {
                    labeled.graph.execute_scheduled_units(
                        ctx,
                        encoder,
                        frame_index,
                        cached_mesh_draws,
                    )?;
                }
            }
        }

        if let Some(mrt) = mrt_views.as_ref()
            && !self.any_pass_writes_surface()
        {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: mrt.color_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: ctx.target.texture(),
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }

        Ok(())
    }

    /// Executes all passes in order, recording into a new command encoder.
    pub fn execute(&mut self, ctx: &mut RenderGraphContext) -> Result<(), RenderPassError> {
        let mut encoder = ctx
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        // Cluster counts and light_count are NOT reset here. ClusteredLightPass sets them when it
        // runs. If it skips, we keep the previous frame's values so the mesh pass can still use
        // cluster buffers from the last successful run (avoids "lights flash then vanish" when
        // clustered_light occasionally skips).

        let overlay_count = ctx.draw_batches.iter().filter(|b| b.is_overlay).count();
        let non_overlay_count = ctx.draw_batches.len().saturating_sub(overlay_count);
        logger::trace!(
            "render frame batches: {} overlay, {} non-overlay (clustered_light needs non-overlay)",
            overlay_count,
            non_overlay_count
        );

        if ctx.pre_collected.is_none() {
            ensure_mesh_buffers(ctx.gpu, ctx.session, ctx.draw_batches);
        }

        let computed;
        let cached_mesh_draws = match ctx.pre_collected {
            Some(pc) => Some((&pc.0[..], &pc.1[..], &pc.2[..], &pc.3[..])),
            None => {
                computed = run_collect_mesh_draws(
                    ctx.session,
                    ctx.draw_batches,
                    &*ctx.gpu,
                    ctx.proj,
                    ctx.overlay_projection_override.clone(),
                );
                let cached = &computed.0;
                Some((&cached.0[..], &cached.1[..], &cached.2[..], &cached.3[..]))
            }
        };

        if let Some(ref mut ray_tracing) = ctx.gpu.ray_tracing_state
            && let Some(ref accel) = ctx.gpu.accel_cache
        {
            crate::gpu::update_tlas(
                &ctx.gpu.device,
                &mut encoder,
                ray_tracing,
                accel,
                ctx.draw_batches,
                &ctx.proj,
                ctx.overlay_projection_override.as_ref(),
                ctx.session.asset_registry(),
                ctx.session.render_config().frustum_culling,
            );
        }

        let frame_index = ctx.pipeline_manager.acquire_frame_index(&ctx.gpu.device);

        self.execute_scheduled_units(ctx, &mut encoder, frame_index, cached_mesh_draws)?;

        if let (Some(query_set), Some(resolve_buffer), Some(staging_buffer)) = (
            ctx.timestamp_query_set,
            ctx.timestamp_resolve_buffer,
            ctx.timestamp_staging_buffer,
        ) {
            encoder.resolve_query_set(query_set, 0..2, resolve_buffer, 0);
            encoder.copy_buffer_to_buffer(
                resolve_buffer,
                0,
                staging_buffer,
                0,
                resolve_buffer.size(),
            );
        }

        if let Some(hook) = ctx.before_submit.as_mut() {
            hook(&mut encoder);
        }

        let submission = ctx.gpu.queue.submit(std::iter::once(encoder.finish()));
        ctx.pipeline_manager
            .record_submission(submission, frame_index);
        Ok(())
    }
}

impl Default for RenderGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl RenderGraph {
    /// Returns pass names in execution order (depth-first; subgraph passes are prefixed with
    /// `label/`). For tests only.
    fn pass_names(&self) -> Vec<String> {
        let mut out = Vec::new();
        for unit in &self.execution {
            match unit {
                ExecutionUnit::Pass { pass, .. } => out.push(pass.name().to_string()),
                ExecutionUnit::Subgraph(labeled) => {
                    let prefix = labeled.label.as_str();
                    for n in labeled.graph.pass_names() {
                        out.push(format!("{prefix}/{n}"));
                    }
                }
            }
        }
        out
    }

    /// Returns composite and overlay PassIds for tests. For tests only.
    fn special_pass_ids(&self) -> (Option<PassId>, Option<PassId>) {
        (self.composite_pass_id, self.overlay_pass_id)
    }

    /// Returns pass resource declarations in execution order. For tests only.
    fn pass_resources(&self) -> &[PassResources] {
        &self.pass_resources
    }
}

/// Ensures all meshes referenced by draw batches are in the GPU mesh buffer cache.
///
/// When ray tracing is available, BLAS builds are submitted as separate queue submissions
/// (one per new mesh). After all builds, waits for those submissions to complete so the
/// TLAS build in the same frame (`update_tlas`) can safely reference the BLASes.
fn ensure_mesh_buffers(
    gpu: &mut crate::gpu::GpuState,
    session: &crate::session::Session,
    draw_batches: &[SpaceDrawBatch],
) {
    let mesh_assets = session.asset_registry();
    let mut built_any_blas = false;
    for batch in draw_batches {
        for d in &batch.draws {
            if d.mesh_asset_id < 0 {
                continue;
            }
            let Some(mesh) = mesh_assets.get_mesh(d.mesh_asset_id) else {
                continue;
            };
            if mesh.vertex_count <= 0 || mesh.index_count <= 0 {
                continue;
            }
            if !gpu.mesh_buffer_cache.contains_key(&d.mesh_asset_id) {
                let stride = crate::assets::compute_vertex_stride(&mesh.vertex_attributes) as usize;
                let stride = if stride > 0 {
                    stride
                } else {
                    crate::gpu::compute_vertex_stride_from_mesh(mesh)
                };
                let ray_tracing = gpu.ray_tracing_available;
                if let Some(b) =
                    crate::gpu::create_mesh_buffers(&gpu.device, mesh, stride, ray_tracing)
                {
                    gpu.mesh_buffer_cache.insert(d.mesh_asset_id, b.clone());
                    if let Some(ref mut accel) = gpu.accel_cache
                        && let Some(blas) =
                            crate::gpu::build_blas_for_mesh(&gpu.device, &gpu.queue, mesh, &b)
                    {
                        accel.insert(d.mesh_asset_id, blas);
                        built_any_blas = true;
                    }
                }
            }
        }
    }

    // Each build_blas_for_mesh call submits a separate queue submission. The TLAS build
    // (update_tlas) in the same frame records into the main encoder and references these
    // BLASes. Without waiting, the GPU may still be executing the BLAS build submissions
    // when the TLAS build tries to read them, causing a GPU fault / TDR crash on large scenes.
    if built_any_blas {
        let _ = gpu.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
    }
}

#[cfg(test)]
/// Unit tests for graph build errors, execution metadata, and pass helpers.
mod tests {
    use super::*;

    struct TestPass {
        name: String,
    }

    impl RenderPass for TestPass {
        fn name(&self) -> &str {
            &self.name
        }
        fn execute(&mut self, _ctx: &mut RenderPassContext) -> Result<(), RenderPassError> {
            Ok(())
        }
    }

    #[test]
    fn graph_builder_add_pass_if_skips_when_false() {
        let mut builder = GraphBuilder::new();
        let _a = builder.add_pass(Box::new(TestPass {
            name: "a".to_string(),
        }));
        let opt = builder.add_pass_if(
            false,
            Box::new(TestPass {
                name: "b".to_string(),
            }),
        );
        assert!(opt.is_none());
        let graph = builder.build().expect("single pass");
        assert_eq!(graph.pass_names(), &["a"]);
    }

    #[test]
    fn graph_builder_add_pass_if_adds_when_true() {
        let mut builder = GraphBuilder::new();
        let a = builder.add_pass(Box::new(TestPass {
            name: "a".to_string(),
        }));
        let b = builder
            .add_pass_if(
                true,
                Box::new(TestPass {
                    name: "b".to_string(),
                }),
            )
            .expect("condition true");
        builder.add_edge(a, b);
        let graph = builder.build().expect("chain");
        assert_eq!(graph.pass_names(), &["a", "b"]);
    }

    #[test]
    fn main_render_graph_no_rtao_mesh_writes_color_depth() {
        let graph = build_main_render_graph(false).expect("graph");
        assert_eq!(graph.pass_names(), &["clustered_light", "mesh", "overlay"]);
        let (comp, overlay) = graph.special_pass_ids();
        assert!(comp.is_none());
        assert!(overlay.is_some());
        let res = graph.pass_resources();
        assert!(res[1].writes.contains(&ResourceSlot::Color));
        assert!(res[1].writes.contains(&ResourceSlot::Depth));
        assert!(!res[1].writes.contains(&ResourceSlot::Position));
    }

    #[test]
    fn main_render_graph_rtao_includes_compute_blur_composite() {
        let graph = build_main_render_graph(true).expect("graph");
        let names = graph.pass_names();
        assert_eq!(names.len(), 6);
        assert!(names.iter().any(|n| n == "rtao_compute"));
        assert!(names.iter().any(|n| n == "rtao_blur"));
        assert!(names.iter().any(|n| n == "composite"));
        let (comp, overlay) = graph.special_pass_ids();
        assert!(comp.is_some());
        assert!(overlay.is_some());
        let res = graph.pass_resources();
        assert!(res[1].writes.contains(&ResourceSlot::Position));
    }

    /// [`RtaoBlurPass`] samples the normal G-buffer; its [`RenderPass::resources`] must declare
    /// [`ResourceSlot::Normal`] or per-pass [`RenderTargetViews`] omit `mrt_normal_view` and blur
    /// never runs (breaking composite AO).
    #[test]
    fn rtao_blur_pass_declares_normal_read_for_slot_map() {
        let pass = RtaoBlurPass::new();
        let r = pass.resources();
        assert!(
            r.reads.contains(&ResourceSlot::Normal),
            "blur execute() needs mrt_normal_view from slot map"
        );
    }

    #[test]
    fn graph_builder_subgraph_wraps_main_view_flat_graph() {
        let inner = build_main_render_graph(false).expect("inner");
        let mut builder = GraphBuilder::new();
        let _main = builder.add_subgraph("main_view", inner);
        let graph = builder.build().expect("single subgraph");
        let names = graph.pass_names();
        assert!(names.iter().any(|n| n == "main_view/clustered_light"));
        assert!(names.iter().any(|n| n == "main_view/mesh"));
        assert!(names.iter().any(|n| n == "main_view/overlay"));
    }

    #[test]
    fn graph_builder_edge_pass_to_subgraph() {
        let inner = build_main_render_graph(false).expect("inner");
        let mut builder = GraphBuilder::new();
        let pre = builder.add_pass(Box::new(TestPass {
            name: "pre".to_string(),
        }));
        let sg = builder.add_subgraph("main", inner);
        builder.add_edge(pre, sg);
        let graph = builder.build().expect("dag");
        let names = graph.pass_names();
        assert_eq!(names[0], "pre");
        assert!(names[1].starts_with("main/"));
    }

    /// A valid acyclic graph topologically sorts to the unique order `a → b → c`.
    #[test]
    fn graph_builder_valid_graph_produces_expected_pass_order() {
        let mut builder = GraphBuilder::new();
        let a = builder.add_pass(Box::new(TestPass {
            name: "a".to_string(),
        }));
        let b = builder.add_pass(Box::new(TestPass {
            name: "b".to_string(),
        }));
        let c = builder.add_pass(Box::new(TestPass {
            name: "c".to_string(),
        }));
        builder.add_edge(a, b);
        builder.add_edge(b, c);
        let graph = builder.build().expect("linear chain has no cycle");
        assert_eq!(graph.pass_names(), &["a", "b", "c"]);
    }

    /// An edge cycle makes topological sort impossible; build returns [`GraphBuildError::CycleDetected`].
    #[test]
    fn graph_builder_cycle_returns_cycle_detected() {
        let mut builder = GraphBuilder::new();
        let a = builder.add_pass(Box::new(TestPass {
            name: "a".to_string(),
        }));
        let b = builder.add_pass(Box::new(TestPass {
            name: "b".to_string(),
        }));
        builder.add_edge(a, b);
        builder.add_edge(b, a);
        let result = builder.build();
        assert!(matches!(result, Err(GraphBuildError::CycleDetected)));
    }

    #[test]
    fn graph_builder_dag_branching() {
        let mut builder = GraphBuilder::new();
        let a = builder.add_pass(Box::new(TestPass {
            name: "a".to_string(),
        }));
        let b = builder.add_pass(Box::new(TestPass {
            name: "b".to_string(),
        }));
        let c = builder.add_pass(Box::new(TestPass {
            name: "c".to_string(),
        }));
        let d = builder.add_pass(Box::new(TestPass {
            name: "d".to_string(),
        }));
        builder.add_edge(a, b);
        builder.add_edge(a, c);
        builder.add_edge(b, d);
        builder.add_edge(c, d);
        let graph = builder.build().expect("DAG has no cycle");
        let names = graph.pass_names();
        assert_eq!(names.len(), 4);
        assert_eq!(names[0], "a");
        assert_eq!(names[3], "d");
        assert!(names.contains(&"b".to_string()));
        assert!(names.contains(&"c".to_string()));
    }

    #[test]
    fn graph_builder_special_passes_recorded() {
        let mut builder = GraphBuilder::new();
        let _clustered = builder.add_pass(Box::new(ClusteredLightPass::new()));
        let _mesh = builder.add_pass(Box::new(MeshRenderPass::with_rtao_mrt_graph(true)));
        let _rtao = builder.add_pass(Box::new(RtaoComputePass::new()));
        let _rtao_blur = builder.add_pass(Box::new(RtaoBlurPass::new()));
        let composite = builder.add_pass(Box::new(CompositePass::new()));
        let overlay = builder.add_pass(Box::new(OverlayRenderPass::new()));
        builder.add_edge(_clustered, _mesh);
        builder.add_edge(_mesh, _rtao);
        builder.add_edge(_rtao, _rtao_blur);
        builder.add_edge(_rtao_blur, composite);
        builder.add_edge(composite, overlay);
        let graph = builder
            .build_with_special_passes(Some(composite), Some(overlay))
            .expect("graph has no cycle");
        let (comp_id, overlay_id) = graph.special_pass_ids();
        assert!(comp_id.is_some(), "composite PassId should be recorded");
        assert!(overlay_id.is_some(), "overlay PassId should be recorded");
        assert_eq!(comp_id, Some(composite));
        assert_eq!(overlay_id, Some(overlay));
    }

    #[test]
    fn graph_builder_stores_pass_resources() {
        let mut builder = GraphBuilder::new();
        let _clustered = builder.add_pass(Box::new(ClusteredLightPass::new()));
        let _mesh = builder.add_pass(Box::new(MeshRenderPass::with_rtao_mrt_graph(true)));
        let _rtao = builder.add_pass(Box::new(RtaoComputePass::new()));
        let _rtao_blur = builder.add_pass(Box::new(RtaoBlurPass::new()));
        let composite = builder.add_pass(Box::new(CompositePass::new()));
        let overlay = builder.add_pass(Box::new(OverlayRenderPass::new()));
        builder.add_edge(_clustered, _mesh);
        builder.add_edge(_mesh, _rtao);
        builder.add_edge(_rtao, _rtao_blur);
        builder.add_edge(_rtao_blur, composite);
        builder.add_edge(composite, overlay);
        let graph = builder
            .build_with_special_passes(Some(composite), Some(overlay))
            .expect("graph has no cycle");
        let resources = graph.pass_resources();
        assert_eq!(resources.len(), 6);
        assert!(resources[0].writes.contains(&ResourceSlot::ClusterBuffers));
        assert!(resources[1].reads.contains(&ResourceSlot::ClusterBuffers));
        assert!(resources[2].writes.contains(&ResourceSlot::AoRaw));
        assert!(resources[3].writes.contains(&ResourceSlot::Ao));
        assert!(resources[4].writes.contains(&ResourceSlot::Surface));
        assert!(resources[5].writes.contains(&ResourceSlot::Surface));
    }

    /// A pass that reads a slot no predecessor writes fails with [`GraphBuildError::MissingDependency`].
    #[test]
    fn graph_builder_missing_read_returns_missing_dependency() {
        let mut builder = GraphBuilder::new();
        let a = builder.add_pass(Box::new(TestPass {
            name: "a".to_string(),
        }));
        let composite = builder.add_pass(Box::new(CompositePass::new()));
        builder.add_edge(a, composite);
        let result = builder.build();
        assert!(
            matches!(
                result,
                Err(GraphBuildError::MissingDependency {
                    slot: ResourceSlot::Color,
                    ..
                })
            ),
            "composite reads Color but no earlier pass produces it"
        );
    }

    #[test]
    fn texture_read_target_uses_depth_overlay_vs_blur() {
        let overlay = PassResources {
            reads: vec![ResourceSlot::Depth],
            writes: vec![ResourceSlot::Surface],
        };
        assert_eq!(
            super::texture_read_target_uses(ResourceSlot::Depth, &overlay),
            Some(wgpu::TextureUses::DEPTH_STENCIL_WRITE)
        );
        let blur = PassResources {
            reads: vec![
                ResourceSlot::AoRaw,
                ResourceSlot::Depth,
                ResourceSlot::Normal,
            ],
            writes: vec![ResourceSlot::Ao],
        };
        assert_eq!(
            super::texture_read_target_uses(ResourceSlot::Depth, &blur),
            Some(wgpu::TextureUses::DEPTH_STENCIL_READ)
        );
    }
}
