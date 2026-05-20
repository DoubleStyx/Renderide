//! Resolves MSAA / multiview attachment views from compiled transient textures.

use crate::graph_inputs::{GraphPassFrame, MsaaViews};
use crate::render_graph::context::{GraphResolvedResources, ResolvedGraphTexture};
use crate::render_graph::resources::TextureHandle;

fn first_two_layer_views(texture: &ResolvedGraphTexture) -> Option<[wgpu::TextureView; 2]> {
    Some([
        texture.layer_views.first()?.clone(),
        texture.layer_views.get(1)?.clone(),
    ])
}

/// Creates a single-layer `D2` view of `texture` with `DepthOnly` aspect, suitable for sampling
/// the multisampled depth attachment in the depth-resolve compute shader.
fn depth_sample_view(texture: &ResolvedGraphTexture, layer: Option<u32>) -> wgpu::TextureView {
    let view = texture.texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("forward-msaa-depth-sample-view"),
        dimension: Some(wgpu::TextureViewDimension::D2),
        base_array_layer: layer.unwrap_or(0),
        array_layer_count: Some(1),
        aspect: wgpu::TextureAspect::DepthOnly,
        ..Default::default()
    });
    crate::profiling::note_resource_churn!(
        TextureView,
        "render_graph::forward_msaa_depth_sample_view"
    );
    view
}

fn first_two_depth_sample_layer_views(
    texture: &ResolvedGraphTexture,
) -> Option<[wgpu::TextureView; 2]> {
    if texture.layer_views.len() < 2 {
        return None;
    }
    Some([
        depth_sample_view(texture, Some(0)),
        depth_sample_view(texture, Some(1)),
    ])
}

/// Resolves MSAA attachment views from graph transient textures for the main graph.
///
/// Returns `None` when MSAA is inactive (`sample_count <= 1`) or the transient handles are
/// unavailable. The executor inserts the returned value into the
/// per-view [`crate::render_graph::blackboard::Blackboard`] as a
/// [`crate::graph_inputs::MsaaViewsSlot`]. Depth views are produced with
/// `DepthOnly` aspect so they are directly bindable as `texture_multisampled_2d<f32>` in the
/// depth-resolve compute shader.
pub(in crate::render_graph::compiled) fn resolve_forward_msaa_views_from_graph_resources(
    frame: &GraphPassFrame<'_>,
    graph_resources: &GraphResolvedResources,
    msaa_handles: Option<[TextureHandle; 2]>,
) -> Option<MsaaViews> {
    let handles = msaa_handles?;
    let [depth_h, r32_h] = handles;
    if frame.view.sample_count <= 1 {
        return None;
    }
    let depth = graph_resources.transient_texture(depth_h)?;
    let r32 = graph_resources.transient_texture(r32_h)?;
    let depth_view = depth_sample_view(depth, None);

    if frame.view.multiview_stereo {
        let depth_layers = first_two_depth_sample_layer_views(depth)?;
        let r32_layers = first_two_layer_views(r32)?;
        Some(MsaaViews {
            msaa_depth_view: depth_view,
            msaa_depth_resolve_r32_view: r32.view.clone(),
            msaa_depth_is_array: true,
            msaa_stereo_depth_layer_views: Some(depth_layers),
            msaa_stereo_r32_layer_views: Some(r32_layers),
        })
    } else {
        Some(MsaaViews {
            msaa_depth_view: depth_view,
            msaa_depth_resolve_r32_view: r32.view.clone(),
            msaa_depth_is_array: false,
            msaa_stereo_depth_layer_views: None,
            msaa_stereo_r32_layer_views: None,
        })
    }
}
