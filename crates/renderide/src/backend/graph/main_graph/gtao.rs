//! GTAO normal prepass wiring for the main render graph.

use crate::render_graph::builder::GraphBuilder;
use crate::render_graph::ids::PassId;
use crate::render_graph::resources::{
    TextureHandle, TransientArrayLayers, TransientExtent, TransientSampleCount,
    TransientTextureDesc, TransientTextureFormat,
};

use super::handles::MainGraphHandles;

/// Pass id plus the view-normals attachment produced by [`add_gtao_normal_prepass_if_active`].
pub(super) struct GtaoNormalPrepassNode {
    pub(super) view_normals: TextureHandle,
    pub(super) pass: PassId,
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

/// Registers the GTAO normal prepass and its view-normal transients when GTAO is enabled in the
/// supplied [`crate::config::PostProcessingSettings`]. Returns the prepass id plus the view-normals
/// texture handle for downstream wiring; returns `None` otherwise.
pub(super) fn add_gtao_normal_prepass_if_active(
    builder: &mut GraphBuilder,
    h: &MainGraphHandles,
    post_processing_settings: &crate::config::PostProcessingSettings,
    msaa_enabled: bool,
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
            msaa_enabled,
            per_draw_slab: h.per_draw_slab,
        },
    )));
    Some(GtaoNormalPrepassNode { view_normals, pass })
}
