//! GTAO opaque-only pass wiring for the main render graph.

use crate::render_graph::builder::GraphBuilder;
use crate::render_graph::ids::PassId;
use crate::render_graph::resources::{
    TextureHandle, TransientArrayLayers, TransientExtent, TransientSampleCount,
    TransientTextureDesc, TransientTextureFormat,
};

/// Pass ids and resources produced by [`add_gtao_if_active`].
pub(super) struct GtaoNode {
    /// Optional MSAA depth resolve that refreshes single-sample frame depth before GTAO samples it.
    pub(super) pre_depth_resolve: Option<PassId>,
    /// Raster pass that writes the smooth view-normal target consumed by GTAO.
    pub(super) normal_pass: PassId,
    /// First and last passes of the GTAO compute/raster subchain.
    pub(super) range: crate::passes::GtaoPassRange,
}

/// Returns true when the live settings enable both post-processing and the GTAO effect.
pub(super) fn gtao_post_processing_active(
    settings: &crate::config::PostProcessingSettings,
) -> bool {
    settings.enabled && settings.gtao.enabled
}

fn create_gtao_view_normal_transients(
    builder: &mut GraphBuilder,
    msaa_enabled: bool,
) -> (TextureHandle, Option<TextureHandle>) {
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
    let normals_msaa = msaa_enabled.then(|| {
        builder.create_texture(TransientTextureDesc {
            label: "gtao_view_normals_msaa",
            format: TransientTextureFormat::Fixed(crate::passes::GTAO_VIEW_NORMAL_FORMAT),
            extent,
            mip_levels: 1,
            sample_count: TransientSampleCount::Frame,
            dimension: wgpu::TextureDimension::D2,
            array_layers: TransientArrayLayers::Frame,
            base_usage: wgpu::TextureUsages::empty(),
            alias: true,
        })
    });
    (normals, normals_msaa)
}

/// Registers GTAO normal, depth-prefilter, AO, denoise, and opaque-composite passes when GTAO is
/// enabled. Returns `None` when post-processing or GTAO is disabled.
pub(super) fn add_gtao_if_active(
    builder: &mut GraphBuilder,
    forward_resources: crate::passes::WorldMeshForwardGraphResources,
    post_processing_settings: &crate::config::PostProcessingSettings,
    multiview_stereo: bool,
) -> Option<GtaoNode> {
    if !gtao_post_processing_active(post_processing_settings) {
        return None;
    }
    let pre_depth_resolve = forward_resources.msaa.map(|_| {
        builder.add_encoder_pass(Box::new(
            crate::passes::WorldMeshForwardGtaoDepthResolvePass::new(forward_resources),
        ))
    });
    let (view_normals, normals_msaa) =
        create_gtao_view_normal_transients(builder, forward_resources.msaa_enabled());
    let normal_pass =
        builder.add_raster_pass(Box::new(crate::passes::WorldMeshForwardNormalPass::new(
            crate::passes::WorldMeshForwardNormalGraphResources {
                normals: view_normals,
                normals_msaa,
                depth: forward_resources.depth,
                msaa_depth: forward_resources.msaa.map(|msaa| msaa.depth),
                per_draw_slab: forward_resources.per_draw_slab,
            },
        )));
    let range = crate::passes::GtaoEffect {
        settings: post_processing_settings.gtao,
        resources: crate::passes::GtaoGraphResources {
            depth: forward_resources.depth,
            view_normals,
            frame_uniforms: forward_resources.frame_uniforms,
            scene_color_hdr: forward_resources.scene_color_hdr,
            scene_color_hdr_msaa: forward_resources.msaa.map(|msaa| msaa.scene_color_hdr),
            multiview_stereo,
        },
    }
    .register(builder);
    Some(GtaoNode {
        pre_depth_resolve,
        normal_pass,
        range,
    })
}
