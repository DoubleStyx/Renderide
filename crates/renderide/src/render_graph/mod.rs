//! Compile-time validated **render graph** with typed handles, setup-time access declarations,
//! pass culling, and transient alias planning. Per-frame command recording may use **several**
//! [`wgpu::CommandEncoder`]s and **several** [`wgpu::Queue::submit`] calls in one tick (see
//! [`CompiledRenderGraph::execute_multi_view`]).
//!
//! **Hi-Z-related code:** CPU helpers for mip layout, depth readback unpacking, and screen-space
//! occlusion tests live in [`hi_z_cpu`] and [`hi_z_occlusion`]. GPU pyramid build, staging, and
//! pipelines are under [`crate::render_graph::occlusion`].
//!
//! ## Portability
//!
//! [`TextureAccess`] and [`BufferAccess`] describe resource usage for ordering and validation. If
//! this project ever targets a lower-level API than wgpu’s automatic barriers, the same access
//! metadata is the natural input for barrier and layout transition planning.
//!
//! ## Responsibilities
//!
//! - **[`GraphBuilder`]** declares transient resources/imports, groups, and [`RenderPass`] nodes,
//!   then calls each pass's setup hook to derive resource-ordering edges.
//! - **[`CompiledRenderGraph`]** — immutable flattened pass list in dependency order with
//!   transient usage unions and lifetime-based alias slots. At run time,
//!   [`CompiledRenderGraph::execute`] / [`CompiledRenderGraph::execute_multi_view`] may acquire the
//!   swapchain once when any pass writes the logical `backbuffer` resource, then present after the
//!   last GPU work for that frame. Encoding is **not** "one encoder for the whole graph":
//!   multi-view runs [`PassPhase::FrameGlobal`] passes in a dedicated encoder + submit, then
//!   **one encoder + submit per [`FrameView`]** for [`PassPhase::PerView`] passes so per-view
//!   [`wgpu::Queue::write_buffer`] updates are visible before each view's commands; see
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
//! 1. **LightPrep** — [`crate::backend::FrameResourceManager::prepare_lights_from_scene`] packs
//!    clustered lights (see [`cluster_frame`]); at most one full pack per winit tick (coalesced across graph entry points).
//! 2. **Camera / cluster params** — [`frame_params::FrameRenderParams`] + [`cluster_frame`] from
//!    host camera and [`HostCameraFrame`].
//! 3. **Cull** — frustum and Hi-Z occlusion in [`world_mesh_cull`] (inputs to forward pass).
//! 4. **Sort** — [`world_mesh_draw_prep`] builds draw order and batch keys.
//! 5. **DrawPrep** — per-draw uniforms and material resolution inside [`passes::WorldMeshForwardPreparePass`].
//! 6. **RenderPasses** — [`CompiledRenderGraph`] runs mesh deform (logical deform outputs producer),
//!    clustered lights, then forward (see [`default_graph_tests`] / [`build_main_graph`]); frame-global
//!    deform runs before per-view passes at execute time ([`CompiledRenderGraph::execute_multi_view`]).
//! 7. **HiZ** — [`passes::HiZBuildPass`] after depth is written; CPU readback feeds next frame’s cull
//!    ([`crate::render_graph::occlusion`]).
//! 8. **SceneColorCompose** — [`passes::SceneColorComposePass`] copies HDR scene color into the swapchain
//!    / XR / offscreen output (hook for future post-processing).
//! 9. **FrameEnd** — submit, optional debug HUD composite, present, Hi-Z frame bookkeeping.

mod blackboard;
mod builder;
mod cache;
mod camera;
mod cluster_frame;
mod compiled;
mod context;
mod error;
mod frame_params;
mod frame_upload_batch;
mod frustum;
mod hi_z_cpu;
mod hi_z_occlusion;
mod ids;
pub mod occlusion;
mod output_depth_mode;
pub mod pass;
pub mod post_processing;
mod record_parallel;
mod resources;
mod reverse_z_depth;
mod schedule;
mod secondary_camera;
mod skinning_palette;
mod swapchain_scope;
mod transient_pool;
mod world_mesh_cull;
mod world_mesh_cull_eval;
mod world_mesh_draw_prep;
mod world_mesh_draw_stats;

#[cfg(test)]
pub(crate) mod test_fixtures;

pub mod passes;

pub use world_mesh_draw_prep::{
    build_instance_batches, collect_and_sort_world_mesh_draws,
    collect_and_sort_world_mesh_draws_with_parallelism, draw_filter_from_camera_entry,
    resolved_material_slots, sort_world_mesh_draws, CameraTransformDrawFilter,
    DrawCollectionContext, FrameMaterialBatchCache, FramePreparedRenderables, InstanceBatch,
    MaterialDrawBatchKey, WorldMeshDrawCollectParallelism, WorldMeshDrawCollection,
    WorldMeshDrawItem,
};
pub use world_mesh_draw_stats::{
    world_mesh_draw_state_rows_from_sorted, world_mesh_draw_stats_from_sorted,
    WorldMeshDrawStateRow, WorldMeshDrawStats,
};

pub use blackboard::{Blackboard, BlackboardSlot};
pub use builder::GraphBuilder;
pub use cache::{GraphCache, GraphCacheKey};
pub use camera::{
    apply_view_handedness_fix, clamp_desktop_fov_degrees, effective_head_output_clip_planes,
    reverse_z_orthographic, reverse_z_perspective, reverse_z_perspective_openxr_fov,
    view_matrix_for_world_mesh_render_space, view_matrix_from_render_transform,
};
pub use camera::{DESKTOP_FOV_DEGREES_MAX, DESKTOP_FOV_DEGREES_MIN};
pub use cluster_frame::{cluster_frame_params, cluster_frame_params_stereo, ClusterFrameParams};
pub use compiled::{
    ColorAttachmentTemplate, CompileStats, CompiledRenderGraph, DepthAttachmentTemplate,
    ExternalFrameTargets, ExternalOffscreenTargets, FrameView, FrameViewTarget, RenderPassTemplate,
};
pub use context::{
    CallbackCtx, ComputePassCtx, CopyPassCtx, GraphRasterPassContext, GraphResolvedResources,
    PostSubmitContext, RasterPassCtx, RenderPassContext, ResolvedGraphBuffer, ResolvedGraphTexture,
    ResolvedImportedBuffer, ResolvedImportedTexture,
};
pub use error::{GraphBuildError, GraphExecuteError, RenderPassError, SetupError};
pub use frame_params::{
    FrameRenderParams, HostCameraFrame, OcclusionViewId, PerViewFramePlan, PerViewFramePlanSlot,
    PerViewHudConfig, PerViewHudOutputs, PerViewHudOutputsSlot, PrecomputedMaterialBind,
    PrefetchedWorldMeshDrawsSlot, PreparedWorldMeshForwardFrame, WorldMeshForwardPipelineState,
    WorldMeshForwardPlanSlot,
};
pub use frustum::{
    mesh_bounds_degenerate_for_cull, world_aabb_from_local_bounds,
    world_aabb_visible_in_homogeneous_clip, Frustum, Plane, HOMOGENEOUS_CLIP_EPS,
};
pub use hi_z_cpu::{
    hi_z_pyramid_dimensions, hi_z_snapshot_from_linear_linear, mip_dimensions,
    mip_levels_for_extent, unpack_linear_rows_to_mips, HiZCpuSnapshot, HiZCullData,
    HiZStereoCpuSnapshot, HI_Z_PYRAMID_MAX_LONG_EDGE,
};
pub use hi_z_occlusion::{
    hi_z_view_proj_matrices, mesh_fully_occluded_in_hiz, stereo_hiz_keeps_draw,
};
pub use ids::{GroupId, PassId};
pub use output_depth_mode::{OutputDepthMode, OutputDepthModeError};
pub use pass::{
    CallbackPass, ComputePass, CopyPass, GroupScope, PassBuilder, PassKind, PassNode, PassPhase,
    RasterPass, RasterPassBuilder,
};
pub use resources::{
    BackendFrameBufferKind, BufferAccess, BufferHandle, BufferImportSource, BufferSizePolicy,
    FrameTargetRole, HistorySlotId, ImportSource, ImportedBufferDecl, ImportedBufferHandle,
    ImportedTextureDecl, ImportedTextureHandle, StorageAccess, TextureAccess,
    TextureAttachmentResolve, TextureAttachmentTarget, TextureHandle, TextureResourceHandle,
    TransientArrayLayers, TransientBufferDesc, TransientExtent, TransientSampleCount,
    TransientTextureDesc, TransientTextureFormat,
};
pub use reverse_z_depth::{
    main_forward_depth_stencil_format, MAIN_FORWARD_DEPTH_CLEAR, MAIN_FORWARD_DEPTH_COMPARE,
};
pub use schedule::{FrameSchedule, ScheduleHudSnapshot, ScheduleStep, ScheduleValidationError};
pub use secondary_camera::{camera_state_enabled, host_camera_frame_for_render_texture};
pub use skinning_palette::{build_skinning_palette, SkinningPaletteParams};
pub use swapchain_scope::{SwapchainEnterOutcome, SwapchainScope};
pub use transient_pool::{
    BufferKey, TextureKey, TransientPool, TransientPoolError, TransientPoolMetrics,
};
pub use world_mesh_cull::{
    build_world_mesh_cull_proj_params, capture_hi_z_temporal, HiZTemporalState, WorldMeshCullInput,
    WorldMeshCullProjParams,
};

/// Imported buffers/transients wired into [`build_main_graph`].
struct MainGraphHandles {
    color: ImportedTextureHandle,
    depth: ImportedTextureHandle,
    hi_z_current: ImportedTextureHandle,
    lights: ImportedBufferHandle,
    cluster_light_counts: ImportedBufferHandle,
    cluster_light_indices: ImportedBufferHandle,
    per_draw_slab: ImportedBufferHandle,
    frame_uniforms: ImportedBufferHandle,
    cluster_params: BufferHandle,
    hi_z_readback: BufferHandle,
    /// Single-sample HDR scene color (forward resolve target + compose input).
    scene_color_hdr: TextureHandle,
    /// Multisampled HDR scene color for forward when MSAA is active.
    scene_color_hdr_msaa: TextureHandle,
    forward_msaa_depth: TextureHandle,
    forward_msaa_depth_r32: TextureHandle,
}

/// Handles for imported backend buffers (lights, cluster tables, per-draw slab, frame uniforms).
struct MainGraphBufferImports {
    lights: ImportedBufferHandle,
    cluster_light_counts: ImportedBufferHandle,
    cluster_light_indices: ImportedBufferHandle,
    per_draw_slab: ImportedBufferHandle,
    frame_uniforms: ImportedBufferHandle,
}

fn import_main_graph_textures(
    builder: &mut GraphBuilder,
) -> (
    ImportedTextureHandle,
    ImportedTextureHandle,
    ImportedTextureHandle,
) {
    let color = builder.import_texture(ImportedTextureDecl {
        label: "frame_color",
        source: ImportSource::FrameTarget(FrameTargetRole::ColorAttachment),
        initial_access: TextureAccess::ColorAttachment {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
            resolve_to: None,
        },
        final_access: TextureAccess::Present,
    });
    let depth = builder.import_texture(ImportedTextureDecl {
        label: "frame_depth",
        source: ImportSource::FrameTarget(FrameTargetRole::DepthAttachment),
        initial_access: TextureAccess::DepthAttachment {
            depth: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
            stencil: None,
        },
        final_access: TextureAccess::Sampled {
            stages: wgpu::ShaderStages::COMPUTE,
        },
    });
    let hi_z_current = builder.import_texture(ImportedTextureDecl {
        label: "hi_z_current",
        source: ImportSource::PingPong(HistorySlotId::HiZ),
        initial_access: TextureAccess::Storage {
            stages: wgpu::ShaderStages::COMPUTE,
            access: StorageAccess::WriteOnly,
        },
        final_access: TextureAccess::Storage {
            stages: wgpu::ShaderStages::COMPUTE,
            access: StorageAccess::WriteOnly,
        },
    });
    (color, depth, hi_z_current)
}

fn import_main_graph_buffers(builder: &mut GraphBuilder) -> MainGraphBufferImports {
    let lights = builder.import_buffer(ImportedBufferDecl {
        label: "lights",
        source: BufferImportSource::BackendFrameResource(BackendFrameBufferKind::Lights),
        initial_access: BufferAccess::Storage {
            stages: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
            access: StorageAccess::ReadOnly,
        },
        final_access: BufferAccess::Storage {
            stages: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
            access: StorageAccess::ReadOnly,
        },
    });
    let cluster_light_counts = builder.import_buffer(ImportedBufferDecl {
        label: "cluster_light_counts",
        source: BufferImportSource::BackendFrameResource(
            BackendFrameBufferKind::ClusterLightCounts,
        ),
        initial_access: BufferAccess::Storage {
            stages: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
            access: StorageAccess::WriteOnly,
        },
        final_access: BufferAccess::Storage {
            stages: wgpu::ShaderStages::FRAGMENT,
            access: StorageAccess::ReadOnly,
        },
    });
    let cluster_light_indices = builder.import_buffer(ImportedBufferDecl {
        label: "cluster_light_indices",
        source: BufferImportSource::BackendFrameResource(
            BackendFrameBufferKind::ClusterLightIndices,
        ),
        initial_access: BufferAccess::Storage {
            stages: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
            access: StorageAccess::WriteOnly,
        },
        final_access: BufferAccess::Storage {
            stages: wgpu::ShaderStages::FRAGMENT,
            access: StorageAccess::ReadOnly,
        },
    });
    let per_draw_slab = builder.import_buffer(ImportedBufferDecl {
        label: "per_draw_slab",
        source: BufferImportSource::BackendFrameResource(BackendFrameBufferKind::PerDrawSlab),
        initial_access: BufferAccess::Storage {
            stages: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            access: StorageAccess::ReadOnly,
        },
        final_access: BufferAccess::Storage {
            stages: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            access: StorageAccess::ReadOnly,
        },
    });
    let frame_uniforms = builder.import_buffer(ImportedBufferDecl {
        label: "frame_uniforms",
        source: BufferImportSource::BackendFrameResource(BackendFrameBufferKind::FrameUniforms),
        initial_access: BufferAccess::Uniform {
            stages: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            dynamic_offset: false,
        },
        final_access: BufferAccess::Uniform {
            stages: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            dynamic_offset: false,
        },
    });
    MainGraphBufferImports {
        lights,
        cluster_light_counts,
        cluster_light_indices,
        per_draw_slab,
        frame_uniforms,
    }
}

/// Declares cluster/Hi-Z staging buffers and HDR forward transients for [`build_main_graph`].
///
/// Forward MSAA depth targets use [`TransientArrayLayers::Frame`] (not a fixed layer count from
/// [`GraphCacheKey::multiview_stereo`]) so the same compiled graph can run mono desktop and stereo
/// OpenXR without mismatched multiview attachment layers.
fn create_main_graph_transient_resources(
    builder: &mut GraphBuilder,
) -> (
    BufferHandle,
    BufferHandle,
    TextureHandle,
    TextureHandle,
    TextureHandle,
    TextureHandle,
) {
    let cluster_params = builder.create_buffer(TransientBufferDesc {
        label: "cluster_params",
        size_policy: BufferSizePolicy::Fixed(crate::backend::CLUSTER_PARAMS_UNIFORM_SIZE * 2),
        base_usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        alias: true,
    });
    let hi_z_readback = builder.create_buffer(TransientBufferDesc {
        label: "hi_z_readback_staging",
        size_policy: BufferSizePolicy::PerViewport { bytes_per_px: 4 },
        base_usage: wgpu::BufferUsages::COPY_DST,
        alias: true,
    });
    // Use [`TransientExtent::Backbuffer`] for forward MSAA targets: [`build_default_main_graph`]
    // uses a placeholder [`GraphCacheKey::surface_extent`]; baking that into `Custom` extent would
    // allocate 1×1 textures while resolve / imported frame color stay at the real swapchain size.
    // Execute-time resolution uses each view's viewport (see [`crate::render_graph::compiled::helpers::resolve_transient_extent`]).
    //
    // Multisampled forward attachments use [`TransientSampleCount::Frame`] so pool allocations match
    // the live MSAA tier; [`GraphCacheKey::msaa_sample_count`] still invalidates [`GraphCache`].
    let extent_backbuffer = TransientExtent::Backbuffer;
    // HDR scene color uses [`TransientTextureFormat::SceneColorHdr`]; the resolved format comes from
    // [`crate::config::RenderingSettings::scene_color_format`] at execute time
    // ([`TransientTextureResolveSurfaceParams::scene_color_format`]).
    let scene_color_hdr = builder.create_texture(TransientTextureDesc {
        label: "scene_color_hdr",
        format: TransientTextureFormat::SceneColorHdr,
        extent: extent_backbuffer,
        mip_levels: 1,
        sample_count: TransientSampleCount::Fixed(1),
        dimension: wgpu::TextureDimension::D2,
        array_layers: TransientArrayLayers::Frame,
        base_usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        alias: true,
    });
    let scene_color_hdr_msaa = builder.create_texture(TransientTextureDesc {
        label: "scene_color_hdr_msaa",
        format: TransientTextureFormat::SceneColorHdr,
        extent: extent_backbuffer,
        mip_levels: 1,
        sample_count: TransientSampleCount::Frame,
        dimension: wgpu::TextureDimension::D2,
        array_layers: TransientArrayLayers::Frame,
        base_usage: wgpu::TextureUsages::empty(),
        alias: true,
    });
    let mut forward_msaa_depth = TransientTextureDesc::frame_depth_stencil_sampled_texture_2d(
        "forward_msaa_depth",
        extent_backbuffer,
        wgpu::TextureUsages::empty(),
    );
    forward_msaa_depth.sample_count = TransientSampleCount::Frame;
    // Same layer policy as scene color MSAA: execute-time stereo (e.g. OpenXR) must not disagree
    // with a graph built under a mono [`GraphCacheKey`].
    forward_msaa_depth.array_layers = TransientArrayLayers::Frame;
    let forward_msaa_depth = builder.create_texture(forward_msaa_depth);
    let forward_msaa_depth_r32 = builder.create_texture(
        TransientTextureDesc::texture_2d(
            "forward_msaa_depth_r32",
            wgpu::TextureFormat::R32Float,
            extent_backbuffer,
            1,
            wgpu::TextureUsages::empty(),
        )
        .with_frame_array_layers(),
    );
    (
        cluster_params,
        hi_z_readback,
        scene_color_hdr,
        scene_color_hdr_msaa,
        forward_msaa_depth,
        forward_msaa_depth_r32,
    )
}

/// Wires imported frame targets and main-graph transients into `builder` for [`build_main_graph`].
fn import_main_graph_resources(builder: &mut GraphBuilder) -> MainGraphHandles {
    let (color, depth, hi_z_current) = import_main_graph_textures(builder);
    let buf = import_main_graph_buffers(builder);
    let (
        cluster_params,
        hi_z_readback,
        scene_color_hdr,
        scene_color_hdr_msaa,
        forward_msaa_depth,
        forward_msaa_depth_r32,
    ) = create_main_graph_transient_resources(builder);
    MainGraphHandles {
        color,
        depth,
        hi_z_current,
        lights: buf.lights,
        cluster_light_counts: buf.cluster_light_counts,
        cluster_light_indices: buf.cluster_light_indices,
        per_draw_slab: buf.per_draw_slab,
        frame_uniforms: buf.frame_uniforms,
        cluster_params,
        hi_z_readback,
        scene_color_hdr,
        scene_color_hdr_msaa,
        forward_msaa_depth,
        forward_msaa_depth_r32,
    }
}

fn add_main_graph_passes_and_edges(
    mut builder: GraphBuilder,
    h: MainGraphHandles,
    post_processing: &crate::config::PostProcessingSettings,
) -> Result<CompiledRenderGraph, GraphBuildError> {
    let deform = builder.add_compute_pass(Box::new(passes::MeshDeformPass::new()));
    let clustered = builder.add_compute_pass(Box::new(passes::ClusteredLightPass::new(
        passes::ClusteredLightGraphResources {
            lights: h.lights,
            cluster_light_counts: h.cluster_light_counts,
            cluster_light_indices: h.cluster_light_indices,
            params: h.cluster_params,
        },
    )));
    let forward_resources = passes::WorldMeshForwardGraphResources {
        scene_color_hdr: h.scene_color_hdr,
        scene_color_hdr_msaa: h.scene_color_hdr_msaa,
        depth: h.depth,
        msaa_depth: h.forward_msaa_depth,
        msaa_depth_r32: h.forward_msaa_depth_r32,
        cluster_light_counts: h.cluster_light_counts,
        cluster_light_indices: h.cluster_light_indices,
        lights: h.lights,
        per_draw_slab: h.per_draw_slab,
        frame_uniforms: h.frame_uniforms,
    };
    let forward_prepare = builder.add_callback_pass(Box::new(
        passes::WorldMeshForwardPreparePass::new(forward_resources),
    ));
    let forward_opaque = builder.add_raster_pass(Box::new(
        passes::WorldMeshForwardOpaquePass::new(forward_resources),
    ));
    let depth_snapshot = builder.add_compute_pass(Box::new(
        passes::WorldMeshDepthSnapshotPass::new(forward_resources),
    ));
    let forward_intersect = builder.add_raster_pass(Box::new(
        passes::WorldMeshForwardIntersectPass::new(forward_resources),
    ));
    let depth_resolve = builder.add_compute_pass(Box::new(
        passes::WorldMeshForwardDepthResolvePass::new(forward_resources),
    ));
    let hiz = builder.add_compute_pass(Box::new(passes::HiZBuildPass::new(
        passes::HiZBuildGraphResources {
            depth: h.depth,
            hi_z_current: h.hi_z_current,
            readback_staging: h.hi_z_readback,
        },
    )));

    let chain = build_default_post_processing_chain(&h, post_processing);
    let chain_output = chain.build_into_graph(&mut builder, h.scene_color_hdr, post_processing);
    let compose_input = chain_output.final_handle();

    let compose = builder.add_raster_pass(Box::new(passes::SceneColorComposePass::new(
        passes::SceneColorComposeGraphResources {
            scene_color_hdr: compose_input,
            frame_color: h.color,
        },
    )));
    builder.add_edge(deform, clustered);
    builder.add_edge(clustered, forward_prepare);
    builder.add_edge(forward_prepare, forward_opaque);
    builder.add_edge(forward_opaque, depth_snapshot);
    builder.add_edge(depth_snapshot, forward_intersect);
    builder.add_edge(forward_intersect, depth_resolve);
    builder.add_edge(depth_resolve, hiz);
    if let Some((first_post, last_post)) = chain_output.pass_range() {
        builder.add_edge(hiz, first_post);
        builder.add_edge(last_post, compose);
    } else {
        builder.add_edge(hiz, compose);
    }
    builder.build()
}

/// Builds the canonical post-processing chain shipped with the renderer.
///
/// Execution order is GTAO → ACES tonemap. GTAO must run first so ambient occlusion modulates
/// linear HDR light before tonemap compresses it. Each effect gates itself via
/// [`PostProcessEffect::is_enabled`] against the live [`crate::config::PostProcessingSettings`].
///
/// `GtaoEffect` is parameterised with the current [`crate::config::GtaoSettings`] snapshot and
/// the imported `frame_uniforms` handle (used to access per-eye projection coefficients and the
/// frame index at record time).
fn build_default_post_processing_chain(
    h: &MainGraphHandles,
    post_processing: &crate::config::PostProcessingSettings,
) -> post_processing::PostProcessChain {
    let mut chain = post_processing::PostProcessChain::new();
    chain.push(Box::new(passes::GtaoEffect {
        settings: post_processing.gtao,
        depth: h.depth,
        frame_uniforms: h.frame_uniforms,
    }));
    chain.push(Box::new(passes::AcesTonemapEffect));
    chain
}

/// Builds the main frame graph: mesh deform compute, clustered lights, world forward, Hi-Z readback,
/// then HDR scene-color compose into the display target.
///
/// Forward MSAA transients use [`TransientExtent::Backbuffer`] and [`TransientSampleCount::Frame`] so
/// sizes match the current view at execute time (the graph is often built with
/// [`build_default_main_graph`]'s placeholder [`GraphCacheKey::surface_extent`]). HDR scene color
/// uses [`TransientTextureFormat::SceneColorHdr`]; the resolved format follows
/// [`crate::config::RenderingSettings::scene_color_format`] at execute time (see
/// [`GraphCacheKey::scene_color_format`] for [`GraphCache`] identity). `key` still drives
/// [`GraphCache`] identity ([`GraphCacheKey::surface_format`], [`GraphCacheKey::multiview_stereo`],
/// [`GraphCacheKey::msaa_sample_count`]). Imported sources resolve at execute time via
/// [`crate::backend::FrameResourceManager`].
pub fn build_main_graph(
    key: GraphCacheKey,
    post_processing: &crate::config::PostProcessingSettings,
) -> Result<CompiledRenderGraph, GraphBuildError> {
    logger::info!(
        "main render graph: scene color HDR format = {:?}, post-processing = {} effect(s)",
        key.scene_color_format,
        key.post_processing.active_count()
    );
    let mut builder = GraphBuilder::new();
    let handles = import_main_graph_resources(&mut builder);
    let msaa_handles = [
        handles.scene_color_hdr_msaa,
        handles.forward_msaa_depth,
        handles.forward_msaa_depth_r32,
    ];
    let mut graph = add_main_graph_passes_and_edges(builder, handles, post_processing)?;
    graph.main_graph_msaa_transient_handles = Some(msaa_handles);
    Ok(graph)
}

/// Builds the main graph with a placeholder cache key for callers that still compile it once at attach.
///
/// Uses [`crate::config::PostProcessingSettings::default`] (chain disabled), yielding a graph with
/// an empty post-processing chain. Pass live settings via [`build_default_main_graph_with`] when
/// the chain should be applied.
pub fn build_default_main_graph() -> Result<CompiledRenderGraph, GraphBuildError> {
    build_default_main_graph_with(&crate::config::PostProcessingSettings::default())
}

/// Builds the main graph with a placeholder cache key but applies `post_processing` so the chain
/// is wired into the graph at attach time.
pub fn build_default_main_graph_with(
    post_processing: &crate::config::PostProcessingSettings,
) -> Result<CompiledRenderGraph, GraphBuildError> {
    let key = GraphCacheKey {
        surface_extent: (1, 1),
        msaa_sample_count: 1,
        multiview_stereo: false,
        surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
        scene_color_format: wgpu::TextureFormat::Rgba16Float,
        post_processing: post_processing::PostProcessChainSignature::from_settings(post_processing),
    };
    build_main_graph(key, post_processing)
}

#[cfg(test)]
mod default_graph_tests {
    use wgpu::TextureFormat;

    use super::*;
    use crate::config::{PostProcessingSettings, TonemapMode, TonemapSettings};
    use crate::render_graph::post_processing::PostProcessChainSignature;

    fn smoke_key() -> GraphCacheKey {
        GraphCacheKey {
            surface_extent: (1280, 720),
            msaa_sample_count: 1,
            multiview_stereo: false,
            surface_format: TextureFormat::Bgra8UnormSrgb,
            scene_color_format: TextureFormat::Rgba16Float,
            post_processing: PostProcessChainSignature::default(),
        }
    }

    fn no_post() -> PostProcessingSettings {
        PostProcessingSettings::default()
    }

    fn aces_enabled_post() -> PostProcessingSettings {
        PostProcessingSettings {
            enabled: true,
            tonemap: TonemapSettings {
                mode: TonemapMode::AcesFitted,
            },
            ..Default::default()
        }
    }

    #[test]
    fn default_main_needs_surface_and_nine_passes() {
        let g = build_main_graph(smoke_key(), &no_post()).expect("default graph");
        assert!(g.needs_surface_acquire());
        assert_eq!(g.pass_count(), 9);
        assert_eq!(g.compile_stats.topo_levels, 9);
        assert_eq!(g.compile_stats.transient_texture_count, 4);
    }

    #[test]
    fn enabling_aces_adds_a_pass_and_a_transient() {
        let g_off = build_main_graph(smoke_key(), &no_post()).expect("default graph");
        let mut key_on = smoke_key();
        key_on.post_processing = PostProcessChainSignature::from_settings(&aces_enabled_post());
        let g_on = build_main_graph(key_on, &aces_enabled_post()).expect("aces graph");
        assert_eq!(g_on.pass_count(), g_off.pass_count() + 1);
        assert!(g_on.needs_surface_acquire());
        assert!(
            g_on.compile_stats.transient_texture_count
                >= g_off.compile_stats.transient_texture_count
        );
    }

    #[test]
    fn graph_cache_reuses_when_key_unchanged() {
        let key = smoke_key();
        let post = no_post();
        let mut cache = GraphCache::default();
        cache
            .ensure(key, || build_main_graph(key, &post))
            .expect("first build");
        let n = cache.pass_count();
        let mut build_called = false;
        cache
            .ensure(key, || {
                build_called = true;
                build_main_graph(key, &post)
            })
            .expect("second ensure");
        assert!(!build_called);
        assert_eq!(cache.pass_count(), n);
    }

    #[test]
    fn graph_cache_rebuilds_when_scene_color_format_changes() {
        let mut a = smoke_key();
        a.scene_color_format = TextureFormat::Rgba16Float;
        let mut b = smoke_key();
        b.scene_color_format = TextureFormat::Rg11b10Ufloat;
        let post = no_post();
        let mut cache = GraphCache::default();
        cache
            .ensure(a, || build_main_graph(a, &post))
            .expect("first build");
        let mut build_called = false;
        cache
            .ensure(b, || {
                build_called = true;
                build_main_graph(b, &post)
            })
            .expect("second ensure");
        assert!(build_called);
    }

    /// MSAA depth transients must follow [`TransientArrayLayers::Frame`] so stereo execution matches
    /// HDR color even when [`GraphCacheKey::multiview_stereo`] was `false` at compile time.
    #[test]
    fn forward_msaa_depth_uses_frame_array_layers_with_mono_cache_key() {
        let mut key = smoke_key();
        key.multiview_stereo = false;
        let g = build_main_graph(key, &no_post()).expect("default graph");
        let forward_depth = g
            .transient_textures
            .iter()
            .find(|t| t.desc.label == "forward_msaa_depth")
            .expect("forward_msaa_depth transient");
        assert_eq!(forward_depth.desc.array_layers, TransientArrayLayers::Frame);
        let r32 = g
            .transient_textures
            .iter()
            .find(|t| t.desc.label == "forward_msaa_depth_r32")
            .expect("forward_msaa_depth_r32 transient");
        assert_eq!(r32.desc.array_layers, TransientArrayLayers::Frame);
    }

    #[test]
    fn graph_cache_rebuilds_when_post_processing_signature_changes() {
        let mut a = smoke_key();
        a.post_processing = PostProcessChainSignature::default();
        let mut b = smoke_key();
        b.post_processing = PostProcessChainSignature::from_settings(&aces_enabled_post());
        let mut cache = GraphCache::default();
        cache
            .ensure(a, || build_main_graph(a, &no_post()))
            .expect("first build");
        let mut build_called = false;
        cache
            .ensure(b, || {
                build_called = true;
                build_main_graph(b, &aces_enabled_post())
            })
            .expect("second ensure");
        assert!(build_called);
    }
}
