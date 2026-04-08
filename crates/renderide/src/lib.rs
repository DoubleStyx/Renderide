//! Renderide: host–renderer IPC, window loop, and GPU presentation (skeleton).
//!
//! The library exposes [`run`] for the `renderide` binary. Shared IPC types live in [`shared`] and
//! are generated; do not edit `shared/shared.rs` by hand.
//!
//! ## Layering
//!
//! - **[`frontend`]** — IPC queues, shared memory accessor, init handshake, lock-step frame gating,
//!   and window [`input`](crate::frontend::input) (winit to [`InputState`](crate::shared::InputState)).
//! - **[`scene`]** — Render spaces, transforms, mesh renderables, host light cache (no wgpu).
//! - **[`backend`]** — GPU device usage, mesh/texture pools, material property store, uploads,
//!   [`MeshPreprocessPipelines`](crate::gpu::MeshPreprocessPipelines), and the compiled
//!   [`render_graph`](crate::render_graph).
//!
//! [`RendererRuntime`](crate::runtime::RendererRuntime) composes these three; prefer adding new
//! logic in the appropriate module rather than growing the façade.

pub mod app;
pub mod assets;
/// GPU resource pools, material tables, mesh/texture uploads, preprocess pipelines — **backend** layer.
pub mod backend;
/// Optional `config.ini` loading and [`config::RendererSettings`] (process-wide defaults).
pub mod config;
pub mod connection;
/// Developer overlay: Dear ImGui frame snapshot + HUD ([`diagnostics::DebugHud`], feature `debug-hud`).
pub mod diagnostics;
/// Host IPC, shared memory, init, lock-step — **frontend** layer.
pub mod frontend;
pub mod gpu;

/// Composed WGSL targets from `build.rs` (`shaders/target/*.wgsl`).
#[doc(hidden)]
pub mod embedded_shaders {
    include!(concat!(env!("OUT_DIR"), "/embedded_shaders.rs"));
}

pub mod ipc;
pub mod materials;
/// Host `HeadOutputDevice` → VR / OpenXR GPU path ([`output_device::head_output_device_wants_openxr`]).
pub mod output_device;
pub mod pipelines;
pub mod present;
/// Desktop vs OpenXR presentation (see plan).
pub mod presentation;
pub mod render_graph;
pub mod resources;
pub mod runtime;
/// Transforms, render spaces, mesh renderables — **scene** layer (no wgpu).
pub mod scene;

pub mod shared;

pub mod xr;

pub use assets::material::{
    parse_materials_update_batch_into_store, MaterialBatchBlobLoader, MaterialDictionary,
    MaterialPropertyLookupIds, MaterialPropertySemanticHook, MaterialPropertyStore,
    MaterialPropertyValue, ParseMaterialBatchOptions, PropertyIdRegistry,
};
pub use backend::{order_lights_for_clustered_shading, GpuLight, RenderBackend, MAX_LIGHTS};
pub use config::{
    load_renderer_settings, log_config_resolve_trace, resolve_save_path, save_renderer_settings,
    save_renderer_settings_from_load, settings_handle_from, ConfigLoadResult, ConfigResolveOutcome,
    ConfigSource, DebugSettings, DisplaySettings, IniDocument, ParseWarning,
    PowerPreferenceSetting, RendererSettings, RendererSettingsHandle, RenderingSettings,
};
pub use connection::{
    get_connection_parameters, try_claim_renderer_singleton, ConnectionParams, InitError,
    DEFAULT_QUEUE_CAPACITY,
};
pub use frontend::RendererFrontend;
pub use gpu::{FrameGpuUniforms, MeshPreprocessPipelines};
pub use ipc::DualQueueIpc;
pub use materials::{
    compose_wgsl, reflect_raster_material_wgsl, resolve_raster_family, DebugWorldNormalsFamily,
    ManifestStemMaterialFamily, MaterialFamilyId, MaterialPipelineCache, MaterialPipelineCacheKey,
    MaterialPipelineDesc, MaterialPipelineFamily, MaterialPropertyGpuLayout, MaterialRegistry,
    MaterialRouter, ReflectError, ReflectedRasterLayout, SolidColorFamily, StemResolver, WgslPatch,
    WorldUnlitPropertyIds, DEBUG_WORLD_NORMALS_FAMILY_ID, MANIFEST_RASTER_FAMILY_ID,
    SOLID_COLOR_FAMILY_ID,
};
pub use render_graph::{
    build_default_main_graph, passes::MeshDeformPass, passes::SwapchainClearPass,
    passes::WorldMeshForwardPass, CompileStats, CompiledRenderGraph, FrameRenderParams,
    GraphBuildError, GraphBuilder, GraphExecuteError, HostCameraFrame, PassId, PassResources,
    RenderPass, RenderPassContext, RenderPassError, ResourceSlot,
};
pub use resources::{
    GpuResource, GpuTexture2d, MeshPool, MeshResidencyMeta, NoopStreamingPolicy, ResidencyTier,
    StreamingPolicy, TexturePool, TextureResidencyMeta, VramAccounting, VramResourceKind,
};
pub use runtime::{InitState, RendererRuntime};
pub use scene::{
    light_casts_shadows, CachedLight, LightCache, MeshMaterialSlot, RenderSpaceId, ResolvedLight,
    SceneCoordinator, SkinnedMeshRenderer, StaticMeshRenderer, TransformRemovalEvent,
};

/// Runs the renderer process: logging, optional IPC, winit loop, and wgpu presentation.
///
/// Returns [`None`] when the event loop exits without a host-requested exit code; otherwise
/// returns an exit code for [`std::process::exit`].
pub fn run() -> Option<i32> {
    app::run()
}
