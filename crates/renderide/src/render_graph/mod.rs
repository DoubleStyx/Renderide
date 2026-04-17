//! Compile-time validated **render graph** with typed handles, setup-time access declarations,
//! pass culling, and transient alias planning.
//!
//! **Hi-Z-related code:** CPU helpers for mip layout, depth readback unpacking, and screen-space
//! occlusion tests live in [`hi_z_cpu`] and [`hi_z_occlusion`]. GPU pyramid build, staging, and
//! pipelines are under [`crate::render_graph::occlusion`].
//!
//! ## Responsibilities
//!
//! - **[`GraphBuilder`]** declares transient resources/imports, groups, and [`RenderPass`] nodes,
//!   then calls each pass's setup hook to derive resource-ordering edges.
//! - **[`CompiledRenderGraph`]** stores the retained schedule, transient usage unions,
//!   lifetime-based alias slots, and the existing frame execution entry points.
//! - **[`GraphCache`]** memoizes a compiled graph by [`GraphCacheKey`] (surface extent, MSAA,
//!   multiview, surface format) so the backend rebuilds only when one of those inputs changes.
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
//! 5. **DrawPrep** — per-draw uniforms and material resolution inside [`passes::WorldMeshForwardPass`].
//! 6. **RenderPasses** — [`CompiledRenderGraph`] runs mesh deform (logical deform outputs producer),
//!    clustered lights, then forward (see [`default_graph_tests`] / [`build_main_graph`]); frame-global
//!    deform runs before per-view passes at execute time ([`CompiledRenderGraph::execute_multi_view`]).
//! 7. **HiZ** — [`passes::HiZBuildPass`] after depth is written; CPU readback feeds next frame’s cull
//!    ([`crate::render_graph::occlusion`]).
//! 8. **FrameEnd** — submit, optional debug HUD composite, present, Hi-Z frame bookkeeping.

mod builder;
mod cache;
mod camera;
mod cluster_frame;
mod compiled;
mod context;
mod error;
mod frame_params;
mod frustum;
mod hi_z_cpu;
mod hi_z_occlusion;
mod ids;
pub mod occlusion;
mod output_depth_mode;
mod pass;
mod resources;
mod reverse_z_depth;
mod secondary_camera;
mod skinning_palette;
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
    DrawCollectionContext, InstanceBatch, MaterialDrawBatchKey, WorldMeshDrawCollectParallelism,
    WorldMeshDrawCollection, WorldMeshDrawItem,
};
pub use world_mesh_draw_stats::{
    world_mesh_draw_state_rows_from_sorted, world_mesh_draw_stats_from_sorted,
    WorldMeshDrawStateRow, WorldMeshDrawStats,
};

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
    ExternalFrameTargets, ExternalOffscreenTargets, FrameView, FrameViewTarget,
    OffscreenSingleViewExecuteSpec, RenderPassTemplate,
};
pub use context::{
    GraphRasterPassContext, GraphResolvedResources, RenderPassContext, ResolvedGraphBuffer,
    ResolvedGraphTexture, ResolvedImportedBuffer, ResolvedImportedTexture,
};
pub use error::{GraphBuildError, GraphExecuteError, RenderPassError, SetupError};
pub use frame_params::{FrameRenderParams, HostCameraFrame, OcclusionViewId};
pub use frustum::{
    mesh_bounds_degenerate_for_cull, mesh_bounds_max_half_extent, world_aabb_from_local_bounds,
    world_aabb_from_skinned_bone_origins, world_aabb_visible_in_homogeneous_clip, Frustum, Plane,
    HOMOGENEOUS_CLIP_EPS,
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
pub use output_depth_mode::OutputDepthMode;
pub use pass::{GroupScope, PassBuilder, PassKind, PassPhase, RenderPass};
pub use resources::{
    BufferAccess, BufferHandle, BufferImportSource, BufferSizePolicy, FrameTargetRole,
    HistorySlotId, ImportSource, ImportedBufferDecl, ImportedBufferHandle, ImportedTextureDecl,
    ImportedTextureHandle, StorageAccess, TextureAccess, TextureAttachmentResolve,
    TextureAttachmentTarget, TextureHandle, TextureResourceHandle, TransientArrayLayers,
    TransientBufferDesc, TransientExtent, TransientSampleCount, TransientTextureDesc,
    TransientTextureFormat,
};
pub use reverse_z_depth::{
    main_forward_depth_stencil_format, MAIN_FORWARD_DEPTH_CLEAR, MAIN_FORWARD_DEPTH_COMPARE,
};
pub use secondary_camera::{camera_state_enabled, host_camera_frame_for_render_texture};
pub use skinning_palette::{build_skinning_palette, SkinningPaletteParams};
pub use transient_pool::{BufferKey, TextureKey, TransientPool, TransientPoolMetrics};
pub use world_mesh_cull::{
    build_world_mesh_cull_proj_params, capture_hi_z_temporal, HiZTemporalState, WorldMeshCullInput,
    WorldMeshCullProjParams,
};

/// Builds the main frame graph: mesh deform compute, clustered lights, world forward, then Hi-Z readback.
///
/// `key` is reserved for future surface-driven resource descriptor selection (e.g. swapchain
/// format). Imported sources resolve at execute time via [`crate::backend::FrameResourceManager`],
/// so the typed declarations below are not yet keyed off `key`. Use [`GraphCacheKey`] from
/// [`crate::gpu::GpuContext`] state when compiling through [`GraphCache`].
pub fn build_main_graph(key: GraphCacheKey) -> Result<CompiledRenderGraph, GraphBuildError> {
    let _ = key;
    let mut builder = GraphBuilder::new();
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
    let lights = builder.import_buffer(ImportedBufferDecl {
        label: "lights",
        source: BufferImportSource::BackendFrameResource("lights"),
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
        source: BufferImportSource::BackendFrameResource("cluster_light_counts"),
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
        source: BufferImportSource::BackendFrameResource("cluster_light_indices"),
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
        source: BufferImportSource::BackendFrameResource("per_draw_slab"),
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
        source: BufferImportSource::BackendFrameResource("frame_uniforms"),
        initial_access: BufferAccess::Uniform {
            stages: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            dynamic_offset: false,
        },
        final_access: BufferAccess::Uniform {
            stages: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            dynamic_offset: false,
        },
    });
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
    let forward_msaa_color = builder.create_texture(
        TransientTextureDesc::frame_color_sampled_texture_2d(
            "forward_msaa_color",
            TransientExtent::Backbuffer,
            wgpu::TextureUsages::empty(),
        )
        .with_frame_array_layers(),
    );
    let forward_msaa_depth = builder.create_texture(
        TransientTextureDesc::frame_sampled_texture_2d(
            "forward_msaa_depth",
            wgpu::TextureFormat::Depth32Float,
            TransientExtent::Backbuffer,
            wgpu::TextureUsages::empty(),
        )
        .with_frame_array_layers(),
    );
    let forward_msaa_depth_r32 = builder.create_texture(
        TransientTextureDesc::texture_2d(
            "forward_msaa_depth_r32",
            wgpu::TextureFormat::R32Float,
            TransientExtent::Backbuffer,
            1,
            wgpu::TextureUsages::empty(),
        )
        .with_frame_array_layers(),
    );
    let deform = builder.add_pass(Box::new(passes::MeshDeformPass::new()));
    let clustered = builder.add_pass(Box::new(passes::ClusteredLightPass::new(
        passes::ClusteredLightGraphResources {
            lights,
            cluster_light_counts,
            cluster_light_indices,
            params: cluster_params,
        },
    )));
    let forward_resources = passes::WorldMeshForwardGraphResources {
        color,
        depth,
        msaa_color: forward_msaa_color,
        msaa_depth: forward_msaa_depth,
        msaa_depth_r32: forward_msaa_depth_r32,
        cluster_light_counts,
        cluster_light_indices,
        lights,
        per_draw_slab,
        frame_uniforms,
    };
    let forward_prepare = builder.add_pass(Box::new(passes::WorldMeshForwardPreparePass::new(
        forward_resources,
    )));
    let forward_opaque = builder.add_pass(Box::new(passes::WorldMeshForwardOpaquePass::new(
        forward_resources,
    )));
    let depth_snapshot = builder.add_pass(Box::new(passes::WorldMeshDepthSnapshotPass::new(
        forward_resources,
    )));
    let forward_intersect = builder.add_pass(Box::new(passes::WorldMeshForwardIntersectPass::new(
        forward_resources,
    )));
    let depth_resolve = builder.add_pass(Box::new(passes::WorldMeshForwardDepthResolvePass::new(
        forward_resources,
    )));
    let hiz = builder.add_pass(Box::new(passes::HiZBuildPass::new(
        passes::HiZBuildGraphResources {
            depth,
            hi_z_current,
            readback_staging: hi_z_readback,
        },
    )));
    builder.add_edge(deform, clustered);
    builder.add_edge(clustered, forward_prepare);
    builder.add_edge(forward_prepare, forward_opaque);
    builder.add_edge(forward_opaque, depth_snapshot);
    builder.add_edge(depth_snapshot, forward_intersect);
    builder.add_edge(forward_intersect, depth_resolve);
    builder.add_edge(depth_resolve, hiz);
    builder.build()
}

#[cfg(test)]
mod default_graph_tests {
    use wgpu::TextureFormat;

    use super::*;

    fn smoke_key() -> GraphCacheKey {
        GraphCacheKey {
            surface_extent: (1280, 720),
            msaa_sample_count: 1,
            multiview_stereo: false,
            surface_format: TextureFormat::Bgra8UnormSrgb,
        }
    }

    #[test]
    fn default_main_needs_surface_and_eight_passes() {
        let g = build_main_graph(smoke_key()).expect("default graph");
        assert!(g.needs_surface_acquire());
        assert_eq!(g.pass_count(), 8);
        assert_eq!(g.compile_stats.topo_levels, 8);
        assert_eq!(g.compile_stats.transient_texture_count, 3);
    }

    #[test]
    fn graph_cache_reuses_when_key_unchanged() {
        let key = smoke_key();
        let mut cache = GraphCache::default();
        cache
            .ensure(key, || build_main_graph(key))
            .expect("first build");
        let n = cache.pass_count();
        let mut build_called = false;
        cache
            .ensure(key, || {
                build_called = true;
                build_main_graph(key)
            })
            .expect("second ensure");
        assert!(!build_called);
        assert_eq!(cache.pass_count(), n);
    }
}
