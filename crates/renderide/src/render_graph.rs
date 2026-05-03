//! Compile-time validated **render graph** with typed handles, setup-time access declarations,
//! pass culling, and transient alias planning. Per-frame command recording may use **several**
//! [`wgpu::CommandEncoder`]s, then submit the assembled command buffers once for the tick (see
//! [`CompiledRenderGraph::execute_multi_view`]).
//!
//! **Hi-Z-related code:** CPU helpers for mip layout, depth readback unpacking, and screen-space
//! occlusion tests live in [`crate::occlusion::cpu`]. GPU pyramid build, staging, and pipelines
//! live in [`crate::occlusion::gpu`].
//!
//! ## Portability
//!
//! [`TextureAccess`] and [`BufferAccess`] describe resource usage for ordering and validation. If
//! this project ever targets a lower-level API than wgpu's automatic barriers, the same access
//! metadata is the natural input for barrier and layout transition planning.
//!
//! ## Responsibilities
//!
//! - **[`GraphBuilder`]** declares transient resources/imports, groups, and [`RenderPass`] nodes,
//!   then calls each pass's setup hook to derive resource-ordering edges.
//! - **[`CompiledRenderGraph`]** -- immutable flattened pass list in dependency order with
//!   transient usage unions and lifetime-based alias slots. At run time,
//!   [`CompiledRenderGraph::execute`] / [`CompiledRenderGraph::execute_multi_view`] may acquire the
//!   swapchain once when any pass writes the logical `backbuffer` resource, then present after the
//!   last GPU work for that frame. Encoding is **not** "one encoder for the whole graph":
//!   multi-view records [`PassPhase::FrameGlobal`] passes in a dedicated encoder, then
//!   **one encoder per [`FrameView`]** for [`PassPhase::PerView`] passes. Deferred
//!   [`wgpu::Queue::write_buffer`] updates are drained before the single submit; see
//!   [`CompiledRenderGraph::execute_multi_view`]. Before the per-view loop, transient resources,
//!   per-view per-draw / frame state ([`crate::backend::FrameResourceManager`]), and the material
//!   pipeline cache are pre-warmed once across all views so the per-view record path no longer
//!   pays lazy `&mut` allocation costs (also a structural prerequisite for the parallel record
//!   path; see [`record_parallel`]).
//! - **[`GraphCache`]** memoizes a compiled graph by [`GraphCacheKey`] (surface extent, MSAA,
//!   multiview, surface format, scene HDR format) so the backend rebuilds only when one of those inputs changes.
//!
//! [`CompileStats`] field `topo_levels` counts Kahn-style **parallel waves** in the DAG at compile
//! time; the executor still walks passes in a **single flat order** (waves are not a separate
//! runtime schedule). The debug HUD surfaces this value next to pass count as a scheduling /
//! future-parallelism hint.
//!
//! ## Frame pipeline
//!
//! Runtime and passes combine to the following **logical** phases each frame (some CPU-side,
//! some GPU passes in [`passes`]):
//!
//! 1. **LightPrep** -- [`crate::backend::FrameResourceManager::prepare_lights_from_scene`] packs
//!    clustered lights (see [`crate::world_mesh::cluster_frame_params`]); at most one full pack per winit tick (coalesced across graph entry points).
//! 2. **Camera / cluster params** -- [`frame_params::GraphPassFrame`] + [`crate::world_mesh::cluster_frame_params`] from
//!    host camera and [`HostCameraFrame`].
//! 3. **Cull** -- frustum and Hi-Z occlusion via [`crate::world_mesh::build_world_mesh_cull_proj_params`] and
//!    [`crate::world_mesh::capture_hi_z_temporal`] (inputs to forward pass).
//! 4. **Sort** -- [`crate::world_mesh::collect_and_sort_draws`] builds draw order and batch keys.
//! 5. **DrawPrep** -- per-draw uniforms and material resolution inside [`passes::WorldMeshForwardPreparePass`].
//! 6. **RenderPasses** -- [`CompiledRenderGraph`] runs mesh deform (logical deform outputs producer),
//!    clustered lights, then forward (see [`default_graph_tests`] / [`build_main_graph`]); frame-global
//!    deform runs before per-view passes at execute time ([`CompiledRenderGraph::execute_multi_view`]).
//! 7. **HiZ** -- [`passes::HiZBuildPass`] after depth is written; CPU readback feeds next frame's cull
//!    ([`crate::render_graph::occlusion`]).
//! 8. **SceneColorCompose** -- [`passes::SceneColorComposePass`] copies HDR scene color into the swapchain
//!    / XR / offscreen output (hook for future post-processing).
//! 9. **FrameEnd** -- submit, optional debug HUD composite, present, Hi-Z frame bookkeeping.

pub(crate) mod blackboard;
pub(crate) mod builder;
pub(crate) mod cache;
pub(crate) mod compiled;
pub(crate) mod context;
pub(crate) mod error;
pub(crate) mod frame_params;
pub(crate) mod frame_upload_batch;
pub(crate) mod gpu_cache;
pub(crate) mod ids;
pub mod main_graph;
pub mod pass;
pub(crate) mod pool;
pub mod post_processing;
mod record_parallel;
pub(crate) mod resources;
pub(crate) mod schedule;
pub(crate) mod secondary_camera;
pub(crate) mod swapchain_scope;

#[doc(hidden)]
pub mod test_fixtures;

pub use blackboard::{Blackboard, BlackboardSlot, FrameMotionVectorsSlot};
pub use builder::GraphBuilder;
pub use cache::{GraphCache, GraphCacheKey};
pub use compiled::{
    ColorAttachmentTemplate, CompileStats, CompiledRenderGraph, DepthAttachmentTemplate, DotFormat,
    ExternalFrameTargets, ExternalOffscreenTargets, FrameView, FrameViewTarget, RenderPassTemplate,
    WorldMeshDrawPlan,
};
pub use context::{
    CallbackCtx, ComputePassCtx, CopyPassCtx, GraphRasterPassContext, GraphResolvedResources,
    PostSubmitContext, RasterPassCtx, RenderPassContext, ResolvedGraphBuffer, ResolvedGraphTexture,
    ResolvedImportedBuffer, ResolvedImportedTexture,
};
pub use error::{GraphBuildError, GraphExecuteError, RenderPassError, SetupError};
pub use frame_params::{FrameViewClear, GraphPassFrame, PerViewFramePlan, PerViewFramePlanSlot};
pub use ids::{GroupId, PassId};
pub use main_graph::{build_default_main_graph, build_default_main_graph_with, build_main_graph};
pub use pass::{
    CallbackPass, ComputePass, CopyPass, GroupScope, PassBuilder, PassKind, PassMergeHint,
    PassNode, PassPhase, RasterPass, RasterPassBuilder,
};
pub use pool::{BufferKey, TextureKey, TransientPool, TransientPoolError, TransientPoolMetrics};
pub use resources::{
    BackendFrameBufferKind, BufferAccess, BufferHandle, BufferImportSource, BufferSizePolicy,
    FrameTargetRole, HistorySlotId, ImportSource, ImportedBufferDecl, ImportedBufferHandle,
    ImportedTextureDecl, ImportedTextureHandle, StorageAccess, SubresourceHandle, TextureAccess,
    TextureAttachmentResolve, TextureAttachmentTarget, TextureHandle, TextureResourceHandle,
    TransientArrayLayers, TransientBufferDesc, TransientExtent, TransientSampleCount,
    TransientSubresourceDesc, TransientTextureDesc, TransientTextureFormat,
};
pub use schedule::{FrameSchedule, ScheduleHudSnapshot, ScheduleStep, ScheduleValidationError};
pub use secondary_camera::{camera_state_enabled, host_camera_frame_for_render_texture};
pub use swapchain_scope::{SwapchainEnterOutcome, SwapchainScope};
