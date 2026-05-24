//! GTAO opaque-only pass wiring for the main render graph.

use crate::render_graph::builder::GraphBuilder;
use crate::render_graph::resources::{
    TextureHandle, TransientArrayLayers, TransientExtent, TransientSampleCount,
    TransientTextureDesc, TransientTextureFormat,
};

use super::handles::MainGraphHandles;

/// Pass ids and resources produced by [`add_gtao_if_active`].
pub(super) struct GtaoNode {
    /// Raster pass that writes the smooth view-normal target consumed by GTAO.
    pub(super) normal_pass: crate::render_graph::ids::PassId,
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

/// Registers GTAO normal, depth-prefilter, AO, denoise, and opaque-composite passes when GTAO is
/// enabled. Returns `None` when post-processing or GTAO is disabled.
pub(super) fn add_gtao_if_active(
    builder: &mut GraphBuilder,
    h: &MainGraphHandles,
    post_processing_settings: &crate::config::PostProcessingSettings,
    msaa_enabled: bool,
    multiview_stereo: bool,
) -> Option<GtaoNode> {
    if !gtao_post_processing_active(post_processing_settings) {
        return None;
    }
    let (view_normals, normals_msaa) = create_gtao_view_normal_transients(builder);
    let normal_pass =
        builder.add_raster_pass(Box::new(crate::passes::WorldMeshForwardNormalPass::new(
            crate::passes::WorldMeshForwardNormalGraphResources {
                normals: view_normals,
                normals_msaa,
                depth: h.depth,
                msaa_depth: h.forward_msaa_depth,
                msaa_enabled,
                per_draw_slab: h.per_draw_slab,
            },
        )));
    let range = crate::passes::GtaoEffect {
        settings: post_processing_settings.gtao,
        resources: crate::passes::GtaoGraphResources {
            depth: h.depth,
            view_normals,
            frame_uniforms: h.frame_uniforms,
            scene_color_hdr: h.scene_color_hdr,
            scene_color_hdr_msaa: h.scene_color_hdr_msaa,
            msaa_enabled,
            multiview_stereo,
        },
    }
    .register(builder);
    Some(GtaoNode { normal_pass, range })
}
