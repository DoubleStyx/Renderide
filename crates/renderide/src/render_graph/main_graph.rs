//! Canonical main render graph: imports, transient declarations, pass topology, and the default
//! post-processing chain.
//!
//! This module is the *application* of the render-graph framework: it wires the renderer's
//! built-in passes together to produce the frame the host expects (mesh deform -> clustered lights
//! -> forward -> Hi-Z -> post-processing -> HDR scene-color compose). The framework primitives it
//! consumes (builder, compiled graph, resources, post-processing chain) live in their respective
//! sibling modules.

use super::builder::GraphBuilder;
use super::cache::GraphCacheKey;
use super::compiled::CompiledRenderGraph;
use super::error::GraphBuildError;
use super::post_processing;
use super::resources::{
    BackendFrameBufferKind, BufferAccess, BufferHandle, BufferImportSource, BufferSizePolicy,
    FrameTargetRole, HistorySlotId, ImportSource, ImportedBufferDecl, ImportedBufferHandle,
    ImportedTextureDecl, ImportedTextureHandle, StorageAccess, TextureAccess, TextureHandle,
    TransientArrayLayers, TransientBufferDesc, TransientExtent, TransientSampleCount,
    TransientTextureDesc, TransientTextureFormat,
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
        source: ImportSource::Frame(FrameTargetRole::ColorAttachment),
        initial_access: TextureAccess::ColorAttachment {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
            resolve_to: None,
        },
        final_access: TextureAccess::Present,
    });
    let depth = builder.import_texture(ImportedTextureDecl {
        label: "frame_depth",
        source: ImportSource::Frame(FrameTargetRole::DepthAttachment),
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
        source: ImportSource::PingPong(HistorySlotId::HI_Z),
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
        source: BufferImportSource::Frame(BackendFrameBufferKind::Lights),
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
        source: BufferImportSource::Frame(BackendFrameBufferKind::ClusterLightCounts),
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
        source: BufferImportSource::Frame(BackendFrameBufferKind::ClusterLightIndices),
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
        source: BufferImportSource::Frame(BackendFrameBufferKind::PerDrawSlab),
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
        source: BufferImportSource::Frame(BackendFrameBufferKind::FrameUniforms),
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

/// Declares cluster buffers and HDR forward transients for [`build_main_graph`].
///
/// Forward MSAA depth targets use [`TransientArrayLayers::Frame`] (not a fixed layer count from
/// [`GraphCacheKey::multiview_stereo`]) so the same compiled graph can run mono desktop and stereo
/// OpenXR without mismatched multiview attachment layers.
fn create_main_graph_transient_resources(
    builder: &mut GraphBuilder,
) -> (
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
    // Use [`TransientExtent::Backbuffer`] for forward MSAA targets: [`build_default_main_graph`]
    // uses a placeholder [`GraphCacheKey::surface_extent`]; baking that into `Custom` extent would
    // allocate 1x1 textures while resolve / imported frame color stay at the real swapchain size.
    // Execute-time resolution uses each view's viewport (see [`crate::render_graph::compiled::helpers::resolve_transient_extent`]).
    //
    // Multisampled forward attachments use [`TransientSampleCount::Frame`] so pool allocations match
    // the live MSAA tier; [`GraphCacheKey::msaa_sample_count`] still invalidates [`super::cache::GraphCache`].
    let extent_backbuffer = TransientExtent::Backbuffer;
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
        scene_color_hdr,
        scene_color_hdr_msaa,
        forward_msaa_depth,
        forward_msaa_depth_r32,
    )
}

fn gtao_post_processing_active(settings: &crate::config::PostProcessingSettings) -> bool {
    settings.enabled && settings.gtao.enabled
}

fn create_gtao_view_normal_transients(
    builder: &mut GraphBuilder,
) -> (TextureHandle, TextureHandle) {
    let extent = TransientExtent::Backbuffer;
    let normals = builder.create_texture(TransientTextureDesc {
        label: "gtao_view_normals",
        format: TransientTextureFormat::Fixed(crate::passes::GTAO_VIEW_NORMAL_FORMAT),
        extent,
        mip_levels: 1,
        sample_count: TransientSampleCount::Fixed(1),
        dimension: wgpu::TextureDimension::D2,
        array_layers: TransientArrayLayers::Frame,
        base_usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        alias: true,
    });
    let normals_msaa = builder.create_texture(TransientTextureDesc {
        label: "gtao_view_normals_msaa",
        format: TransientTextureFormat::Fixed(crate::passes::GTAO_VIEW_NORMAL_FORMAT),
        extent,
        mip_levels: 1,
        sample_count: TransientSampleCount::Frame,
        dimension: wgpu::TextureDimension::D2,
        array_layers: TransientArrayLayers::Frame,
        base_usage: wgpu::TextureUsages::empty(),
        alias: true,
    });
    (normals, normals_msaa)
}

struct GtaoNormalPrepassNode {
    view_normals: TextureHandle,
    pass: crate::render_graph::ids::PassId,
}

fn main_forward_resources(h: &MainGraphHandles) -> crate::passes::WorldMeshForwardGraphResources {
    crate::passes::WorldMeshForwardGraphResources {
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
    }
}

fn add_gtao_normal_prepass_if_active(
    builder: &mut GraphBuilder,
    h: &MainGraphHandles,
    post_processing_settings: &crate::config::PostProcessingSettings,
) -> Option<GtaoNormalPrepassNode> {
    if !gtao_post_processing_active(post_processing_settings) {
        return None;
    }
    let (view_normals, normals_msaa) = create_gtao_view_normal_transients(builder);
    let pass = builder.add_raster_pass(Box::new(crate::passes::WorldMeshForwardNormalPass::new(
        crate::passes::WorldMeshForwardNormalGraphResources {
            normals: view_normals,
            normals_msaa,
            depth: h.depth,
            msaa_depth: h.forward_msaa_depth,
            per_draw_slab: h.per_draw_slab,
        },
    )));
    Some(GtaoNormalPrepassNode { view_normals, pass })
}

/// Wires imported frame targets and main-graph transients into `builder` for [`build_main_graph`].
fn import_main_graph_resources(builder: &mut GraphBuilder) -> MainGraphHandles {
    let (color, depth, hi_z_current) = import_main_graph_textures(builder);
    let buf = import_main_graph_buffers(builder);
    let (
        cluster_params,
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
        scene_color_hdr,
        scene_color_hdr_msaa,
        forward_msaa_depth,
        forward_msaa_depth_r32,
    }
}

fn add_main_graph_passes_and_edges(
    mut builder: GraphBuilder,
    h: MainGraphHandles,
    post_processing_settings: &crate::config::PostProcessingSettings,
    msaa_sample_count: u8,
    cluster_assignment: crate::config::ClusterAssignmentMode,
) -> Result<CompiledRenderGraph, GraphBuildError> {
    let deform = builder.add_compute_pass(Box::new(crate::passes::MeshDeformPass::new()));
    let clustered = builder.add_compute_pass(Box::new(crate::passes::ClusteredLightPass::new(
        crate::passes::ClusteredLightGraphResources {
            lights: h.lights,
            cluster_light_counts: h.cluster_light_counts,
            cluster_light_indices: h.cluster_light_indices,
            params: h.cluster_params,
        },
        cluster_assignment,
    )));
    let forward_resources = main_forward_resources(&h);
    let forward_prepare = builder.add_callback_pass(Box::new(
        crate::passes::WorldMeshForwardPreparePass::new(forward_resources),
    ));
    let forward_opaque = builder.add_raster_pass(Box::new(
        crate::passes::WorldMeshForwardOpaquePass::new(forward_resources),
    ));
    let gtao_normals =
        add_gtao_normal_prepass_if_active(&mut builder, &h, post_processing_settings);
    let depth_snapshot = builder.add_compute_pass(Box::new(
        crate::passes::WorldMeshDepthSnapshotPass::new(forward_resources),
    ));
    let forward_intersect = builder.add_raster_pass(Box::new(
        crate::passes::WorldMeshForwardIntersectPass::new(forward_resources),
    ));
    // Color resolve replaces the wgpu automatic linear `resolve_target`. The pre-grab resolve
    // makes a single-sample HDR snapshot available to grab-pass shaders; the final resolve moves
    // any grab-pass transparent MSAA color back into the single-sample HDR target consumed by
    // post-processing. In 1x mode each forward pass writes `scene_color_hdr` directly.
    let color_resolve_resources = crate::passes::WorldMeshForwardColorResolveGraphResources {
        scene_color_hdr_msaa: h.scene_color_hdr_msaa,
        scene_color_hdr: h.scene_color_hdr,
    };
    let pre_grab_color_resolve = (msaa_sample_count > 1).then(|| {
        builder.add_raster_pass(Box::new(
            crate::passes::WorldMeshForwardColorResolvePass::new_pre_grab(color_resolve_resources),
        ))
    });
    let color_snapshot = builder.add_compute_pass(Box::new(
        crate::passes::WorldMeshColorSnapshotPass::new(forward_resources),
    ));
    let forward_transparent = builder.add_raster_pass(Box::new(
        crate::passes::WorldMeshForwardTransparentPass::new(forward_resources),
    ));
    let final_color_resolve = (msaa_sample_count > 1).then(|| {
        builder.add_raster_pass(Box::new(
            crate::passes::WorldMeshForwardColorResolvePass::new_final(color_resolve_resources),
        ))
    });
    let depth_resolve = builder.add_compute_pass(Box::new(
        crate::passes::WorldMeshForwardDepthResolvePass::new(forward_resources),
    ));
    let hiz = builder.add_compute_pass(Box::new(crate::passes::HiZBuildPass::new(
        crate::passes::HiZBuildGraphResources {
            depth: h.depth,
            hi_z_current: h.hi_z_current,
        },
    )));

    let chain = build_default_post_processing_chain(
        &h,
        post_processing_settings,
        gtao_normals.as_ref().map(|node| node.view_normals),
    );
    let chain_output =
        chain.build_into_graph(&mut builder, h.scene_color_hdr, post_processing_settings);
    let compose_input = chain_output.final_handle();

    let compose = builder.add_raster_pass(Box::new(crate::passes::SceneColorComposePass::new(
        crate::passes::SceneColorComposeGraphResources {
            scene_color_hdr: compose_input,
            frame_color: h.color,
        },
    )));
    builder.add_edge(deform, clustered);
    builder.add_edge(clustered, forward_prepare);
    builder.add_edge(forward_prepare, forward_opaque);
    if let Some(gtao_normals) = gtao_normals {
        builder.add_edge(forward_opaque, gtao_normals.pass);
        builder.add_edge(gtao_normals.pass, depth_snapshot);
    } else {
        builder.add_edge(forward_opaque, depth_snapshot);
    }
    builder.add_edge(depth_snapshot, forward_intersect);
    if let Some(pre_grab_color_resolve) = pre_grab_color_resolve {
        builder.add_edge(forward_intersect, pre_grab_color_resolve);
        builder.add_edge(pre_grab_color_resolve, color_snapshot);
    } else {
        builder.add_edge(forward_intersect, color_snapshot);
    }
    builder.add_edge(color_snapshot, forward_transparent);
    if let Some(final_color_resolve) = final_color_resolve {
        builder.add_edge(forward_transparent, final_color_resolve);
        builder.add_edge(final_color_resolve, depth_resolve);
    } else {
        builder.add_edge(forward_transparent, depth_resolve);
    }
    builder.add_edge(depth_resolve, hiz);
    // Sequence post-processing after the final forward HDR target is available.
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
/// Execution order is GTAO -> bloom -> ACES tonemap. GTAO runs first so ambient occlusion
/// modulates linear HDR light before bloom scatter; bloom runs in HDR-linear space so its
/// dual-filter pyramid operates on scene-referred radiance; then ACES compresses the combined
/// HDR signal to display-referred `[0, 1]`. Each effect gates itself via
/// [`super::post_processing::PostProcessEffect::is_enabled`] against the live
/// [`crate::config::PostProcessingSettings`].
///
/// `GtaoEffect` is parameterised with the current [`crate::config::GtaoSettings`] snapshot and
/// the imported `frame_uniforms` handle (used to access per-eye projection coefficients and the
/// frame index at record time). It is registered only when the graph also created the matching
/// view-normal texture. `BloomEffect` captures a [`crate::config::BloomSettings`] snapshot for its
/// shared params UBO and per-mip blend constants.
fn build_default_post_processing_chain(
    h: &MainGraphHandles,
    post_processing_settings: &crate::config::PostProcessingSettings,
    gtao_view_normals: Option<TextureHandle>,
) -> post_processing::PostProcessChain {
    let mut chain = post_processing::PostProcessChain::new();
    if let Some(view_normals) = gtao_view_normals {
        chain.push(Box::new(crate::passes::GtaoEffect {
            settings: post_processing_settings.gtao,
            depth: h.depth,
            view_normals,
            frame_uniforms: h.frame_uniforms,
        }));
    }
    chain.push(Box::new(crate::passes::BloomEffect {
        settings: post_processing_settings.bloom,
    }));
    chain.push(Box::new(crate::passes::AcesTonemapEffect));
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
/// [`GraphCacheKey::scene_color_format`] for graph-cache identity). `key` still drives graph-cache
/// identity ([`GraphCacheKey::surface_format`], [`GraphCacheKey::multiview_stereo`],
/// [`GraphCacheKey::msaa_sample_count`]). Imported sources resolve at execute time via
/// [`crate::backend::FrameResourceManager`].
pub fn build_main_graph(
    key: GraphCacheKey,
    post_processing_settings: &crate::config::PostProcessingSettings,
) -> Result<CompiledRenderGraph, GraphBuildError> {
    logger::info!(
        "main render graph: scene color HDR format = {:?}, post-processing = {} effect(s)",
        key.scene_color_format,
        key.post_processing.active_count()
    );
    let mut builder = GraphBuilder::new();
    let handles = import_main_graph_resources(&mut builder);
    let msaa_handles = [handles.forward_msaa_depth, handles.forward_msaa_depth_r32];
    let mut graph = add_main_graph_passes_and_edges(
        builder,
        handles,
        post_processing_settings,
        key.msaa_sample_count,
        key.cluster_assignment,
    )?;
    graph.main_graph_msaa_transient_handles = Some(msaa_handles);
    Ok(graph)
}

/// Builds the main graph with a placeholder cache key for callers that still compile it once at attach.
///
/// Uses [`crate::config::PostProcessingSettings::default`], yielding a graph with the built-in
/// default post-processing chain. Pass live settings via [`build_default_main_graph_with`] when
/// the chain should mirror a resolved config value.
pub fn build_default_main_graph() -> Result<CompiledRenderGraph, GraphBuildError> {
    build_default_main_graph_with(&crate::config::PostProcessingSettings::default(), 1)
}

/// Builds the main graph with a placeholder cache key but applies `post_processing_settings` so
/// the chain is wired into the graph at attach time. `msaa_sample_count` selects whether the
/// HDR-aware MSAA color resolve passes are included; pass `1` when MSAA is off.
pub fn build_default_main_graph_with(
    post_processing_settings: &crate::config::PostProcessingSettings,
    msaa_sample_count: u8,
) -> Result<CompiledRenderGraph, GraphBuildError> {
    let key = GraphCacheKey {
        surface_extent: (1, 1),
        msaa_sample_count,
        multiview_stereo: false,
        surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
        scene_color_format: wgpu::TextureFormat::Rgba16Float,
        post_processing: post_processing::PostProcessChainSignature::from_settings(
            post_processing_settings,
        ),
        cluster_assignment: crate::config::ClusterAssignmentMode::Auto,
    };
    build_main_graph(key, post_processing_settings)
}

#[cfg(test)]
mod tests {
    use wgpu::TextureFormat;

    use super::*;
    use crate::config::{
        BloomSettings, GtaoSettings, PostProcessingSettings, TonemapMode, TonemapSettings,
    };
    use crate::render_graph::cache::GraphCache;
    use crate::render_graph::post_processing::PostProcessChainSignature;

    fn smoke_key() -> GraphCacheKey {
        GraphCacheKey {
            surface_extent: (1280, 720),
            msaa_sample_count: 1,
            multiview_stereo: false,
            surface_format: TextureFormat::Bgra8UnormSrgb,
            scene_color_format: TextureFormat::Rgba16Float,
            post_processing: PostProcessChainSignature::default(),
            cluster_assignment: crate::config::ClusterAssignmentMode::Auto,
        }
    }

    fn no_post() -> PostProcessingSettings {
        PostProcessingSettings {
            enabled: false,
            ..Default::default()
        }
    }

    fn aces_enabled_post() -> PostProcessingSettings {
        PostProcessingSettings {
            enabled: true,
            gtao: GtaoSettings {
                enabled: false,
                ..Default::default()
            },
            bloom: BloomSettings {
                enabled: false,
                ..Default::default()
            },
            tonemap: TonemapSettings {
                mode: TonemapMode::AcesFitted,
            },
        }
    }

    fn gtao_enabled_post() -> PostProcessingSettings {
        PostProcessingSettings {
            enabled: true,
            gtao: GtaoSettings {
                enabled: true,
                ..Default::default()
            },
            bloom: BloomSettings {
                enabled: false,
                ..Default::default()
            },
            tonemap: TonemapSettings {
                mode: TonemapMode::None,
            },
        }
    }

    #[test]
    fn default_main_needs_surface_and_eleven_passes() {
        let g = build_main_graph(smoke_key(), &no_post()).expect("default graph");
        assert!(g.needs_surface_acquire());
        assert_eq!(g.pass_count(), 11);
        assert_eq!(g.compile_stats.topo_levels, 11);
        assert_eq!(g.compile_stats.transient_texture_count, 4);
    }

    #[test]
    fn msaa_main_graph_brackets_grab_pass_with_color_resolves() {
        let mut key = smoke_key();
        key.msaa_sample_count = 4;
        let g = build_main_graph(key, &no_post()).expect("MSAA graph");
        let pass_names: Vec<&str> = g.pass_info.iter().map(|p| p.name.as_str()).collect();
        let pre_grab_resolve_pos = pass_names
            .iter()
            .position(|name| *name == "WorldMeshForwardColorResolvePreGrab")
            .expect("pre-grab color resolve pass");
        let snapshot_pos = pass_names
            .iter()
            .position(|name| *name == "WorldMeshColorSnapshot")
            .expect("color snapshot pass");
        let transparent_pos = pass_names
            .iter()
            .position(|name| *name == "WorldMeshForwardTransparent")
            .expect("transparent pass");
        let final_resolve_pos = pass_names
            .iter()
            .position(|name| *name == "WorldMeshForwardColorResolveFinal")
            .expect("final color resolve pass");

        assert!(pre_grab_resolve_pos < snapshot_pos);
        assert!(snapshot_pos < transparent_pos);
        assert!(transparent_pos < final_resolve_pos);
        assert_eq!(g.pass_count(), 13);
        assert_eq!(g.compile_stats.topo_levels, 13);
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
    fn enabling_gtao_adds_normal_prepass_before_gtao_main() {
        let post = gtao_enabled_post();
        let mut key = smoke_key();
        key.post_processing = PostProcessChainSignature::from_settings(&post);
        let g = build_main_graph(key, &post).expect("gtao graph");
        let pass_names: Vec<&str> = g.pass_info.iter().map(|p| p.name.as_str()).collect();
        let normal_pos = pass_names
            .iter()
            .position(|name| *name == "WorldMeshForwardNormals")
            .expect("GTAO normal prepass");
        let depth_snapshot_pos = pass_names
            .iter()
            .position(|name| *name == "WorldMeshDepthSnapshot")
            .expect("depth snapshot pass");
        let gtao_main_pos = pass_names
            .iter()
            .position(|name| *name == "GtaoMain")
            .expect("GTAO main pass");

        assert!(normal_pos < depth_snapshot_pos);
        assert!(depth_snapshot_pos < gtao_main_pos);
        assert!(
            g.transient_textures
                .iter()
                .any(|t| t.desc.label == "gtao_view_normals")
        );
        assert!(
            g.transient_textures
                .iter()
                .any(|t| t.desc.label == "gtao_view_normals_msaa")
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
