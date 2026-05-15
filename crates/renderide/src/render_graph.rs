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
//!   **one encoder per [`FrameView`]** for [`PassPhase::PerView`] passes. Deferred graph upload
//!   writes are drained before the single submit; see
//!   [`CompiledRenderGraph::execute_multi_view`]. Before the per-view loop, transient resources,
//!   graph-facing frame resources, and world-mesh draw packets are prepared once across all views
//!   so the per-view record path no longer pays lazy `&mut` allocation costs (also a structural
//!   prerequisite for the parallel record path; see [`record_parallel`]).
//! - **[`GraphCache`]** memoizes compiled graph variants by [`GraphCacheKey`] (surface extent,
//!   MSAA, multiview, surface format, scene HDR format) so the backend rebuilds only when an
//!   uncached variant is requested.
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
//! 1. **LightPrep** -- the backend packs clustered lights into graph-facing frame resources; at
//!    most one full pack per winit tick (coalesced across graph entry points).
//! 2. **Camera / cluster params** -- [`frame_params::GraphPassFrame`] carries host camera and
//!    per-view frame state to passes.
//! 3. **Cull / sort** -- runtime extraction prepares caller-owned draw packets before graph entry.
//! 4. **View prep** -- backend-specific blackboard preparation packs per-draw uniforms and resolves
//!    material packets before graph pass-node recording.
//! 5. **RenderPasses** -- [`CompiledRenderGraph`] runs mesh deform (logical deform outputs producer),
//!    clustered lights, then forward (see [`default_graph_tests`] / [`build_main_graph`]); frame-global
//!    deform runs before per-view passes at execute time ([`CompiledRenderGraph::execute_multi_view`]).
//! 6. **HiZ** -- [`passes::HiZBuildPass`] after depth is written; CPU readback feeds next frame's cull.
//! 7. **SceneColorCompose** -- [`passes::SceneColorComposePass`] copies HDR scene color into the swapchain
//!    / XR / offscreen output (hook for future post-processing).
//! 8. **FrameEnd** -- submit, optional debug HUD composite, present, Hi-Z frame bookkeeping.

pub(crate) mod blackboard;
pub(crate) mod builder;
pub(crate) mod compiled;
pub(crate) mod context;
pub(crate) mod error;
pub(crate) mod execution_backend;
pub(crate) mod frame_params;
pub(crate) mod frame_upload_batch;
pub(crate) mod gpu_cache;
pub(crate) mod history;
pub(crate) mod ids;
pub(crate) mod pass;
mod pool;
pub(crate) mod post_process_chain;
pub(crate) mod post_process_settings;
mod record_parallel;
pub(crate) mod resources;
mod schedule;
mod swapchain_scope;
pub(crate) mod upload_arena;

pub(crate) use compiled::cache::{GraphCache, GraphCacheEnsureResult, GraphCacheKey};
pub(crate) use compiled::{
    ExternalFrameTargets, ExternalOffscreenTargets, FrameView, FrameViewResourceHints,
    FrameViewTarget, OffscreenSampleCountPolicy, ViewPostProcessing,
};
pub(crate) use error::GraphExecuteError;
pub(crate) use execution_backend::{
    GraphAssetResources, GraphExecutionBackend, GraphFrameResources,
};
pub(crate) use frame_params::FrameViewClear;
pub(crate) use history::{HistoryRegistry, HistoryRegistryError, HistoryTextureMipViews};
pub(crate) use pool::TransientPool;
pub(crate) use resources::HistorySlotId;
