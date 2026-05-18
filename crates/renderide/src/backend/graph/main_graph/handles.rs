//! Imported texture/buffer handles and transient resource declarations for the main render graph.

use crate::render_graph::builder::GraphBuilder;
use crate::render_graph::resources::{
    BackendFrameBufferKind, BufferAccess, BufferHandle, BufferImportSource, BufferSizePolicy,
    FrameTargetRole, HistorySlotId, ImportSource, ImportedBufferDecl, ImportedBufferHandle,
    ImportedTextureDecl, ImportedTextureHandle, StorageAccess, TextureAccess, TextureHandle,
    TransientArrayLayers, TransientBufferDesc, TransientExtent, TransientSampleCount,
    TransientTextureDesc, TransientTextureFormat,
};

/// Long-lived resources shared by post-processing passes across main-graph rebuilds.
#[derive(Clone, Default)]
pub(crate) struct MainGraphPostProcessingResources {
    auto_exposure_state_cache:
        std::sync::Arc<crate::passes::post_processing::AutoExposureStateCache>,
    motion_blur_state_cache: std::sync::Arc<crate::passes::post_processing::MotionBlurStateCache>,
}

impl MainGraphPostProcessingResources {
    /// Shared auto-exposure state cache used by graph instances built for the same backend.
    pub(crate) fn auto_exposure_state_cache(
        &self,
    ) -> std::sync::Arc<crate::passes::post_processing::AutoExposureStateCache> {
        std::sync::Arc::clone(&self.auto_exposure_state_cache)
    }

    /// Shared motion-blur state cache used by graph instances built for the same backend.
    pub(crate) fn motion_blur_state_cache(
        &self,
    ) -> std::sync::Arc<crate::passes::post_processing::MotionBlurStateCache> {
        std::sync::Arc::clone(&self.motion_blur_state_cache)
    }

    /// Releases view-scoped post-processing resources for views that are no longer active.
    pub(crate) fn retire_views(&self, retired_views: &[crate::camera::ViewId]) {
        self.auto_exposure_state_cache.retire_views(retired_views);
        self.motion_blur_state_cache.retire_views(retired_views);
    }
}

/// Imported buffers/transients wired into the main render graph.
pub(super) struct MainGraphHandles {
    pub(super) color: ImportedTextureHandle,
    pub(super) depth: ImportedTextureHandle,
    pub(super) hi_z_current: ImportedTextureHandle,
    pub(super) lights: ImportedBufferHandle,
    pub(super) cluster_light_counts: ImportedBufferHandle,
    pub(super) cluster_light_indices: ImportedBufferHandle,
    pub(super) per_draw_slab: ImportedBufferHandle,
    pub(super) frame_uniforms: ImportedBufferHandle,
    pub(super) cluster_params: BufferHandle,
    /// Single-sample HDR scene color (forward resolve target + compose input).
    pub(super) scene_color_hdr: TextureHandle,
    /// Multisampled HDR scene color for forward when MSAA is active.
    pub(super) scene_color_hdr_msaa: TextureHandle,
    pub(super) forward_msaa_depth: TextureHandle,
    pub(super) forward_msaa_depth_r32: TextureHandle,
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

/// Declares cluster buffers and HDR forward transients for the main render graph.
///
/// Forward MSAA depth targets use [`TransientArrayLayers::Frame`] (not a fixed layer count from
/// `GraphCacheKey::multiview_stereo`) so the same compiled graph can run mono desktop and stereo
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
        size_policy: BufferSizePolicy::Fixed(crate::gpu::CLUSTER_PARAMS_UNIFORM_SIZE * 2),
        base_usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        alias: true,
    });
    // Backbuffer extent + Frame sample count keep forward MSAA targets aligned to the live
    // swapchain even when the graph is compiled with a placeholder cache key.
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

/// Wires imported frame targets and main-graph transients into `builder`.
pub(super) fn import_main_graph_resources(builder: &mut GraphBuilder) -> MainGraphHandles {
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
