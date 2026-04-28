//! MSAA depth resolve and scene-depth snapshot helpers for the graph-managed
//! world-mesh forward pass.

use crate::gpu::{
    MsaaDepthResolveMonoTargets, MsaaDepthResolveResources, MsaaDepthResolveStereoTargets,
};
use crate::render_graph::context::{GraphResolvedResources, ResolvedGraphTexture};
use crate::render_graph::frame_params::{FrameRenderParams, PreparedWorldMeshForwardFrame};

use super::super::WorldMeshForwardGraphResources;

/// Resolves MSAA depth when needed, then copies the single-sample frame depth into the
/// sampled scene-depth snapshot used by intersection materials.
pub(crate) fn encode_world_mesh_forward_depth_snapshot(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    frame: &mut FrameRenderParams<'_>,
    prepared: &PreparedWorldMeshForwardFrame,
    msaa_views: Option<&ForwardMsaaResolvedViews>,
    msaa_depth_resolve: Option<&MsaaDepthResolveResources>,
) -> bool {
    if prepared.plan.intersect_groups.is_empty() {
        return false;
    }

    if frame.view.sample_count > 1 {
        if let (Some(msaa_views), Some(res)) = (msaa_views, msaa_depth_resolve) {
            encode_msaa_depth_resolve_for_frame(device, encoder, frame, msaa_views, res);
        }
    }

    if frame.shared.frame_resources.frame_gpu().is_none() {
        return false;
    }
    frame
        .shared
        .frame_resources
        .copy_scene_depth_snapshot_for_view(
            frame.view.view_id,
            encoder,
            frame.view.depth_texture,
            frame.view.viewport_px,
            prepared.pipeline.use_multiview,
        );
    true
}

/// After a clear-only MSAA pass, resolves multisampled depth to the single-sample depth used by Hi-Z.
pub(crate) fn encode_msaa_depth_resolve_after_clear_only(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    frame: &FrameRenderParams<'_>,
    msaa_views: Option<&ForwardMsaaResolvedViews>,
    msaa_depth_resolve: Option<&MsaaDepthResolveResources>,
) {
    if frame.view.sample_count <= 1 {
        return;
    }
    let (Some(msaa_views), Some(res)) = (msaa_views, msaa_depth_resolve) else {
        return;
    };
    encode_msaa_depth_resolve_for_frame(device, encoder, frame, msaa_views, res);
}

/// Dispatches the desktop (`D2`) or stereo (`D2Array` multiview) depth-resolve path based on
/// [`ForwardMsaaResolvedViews::is_array`].
fn encode_msaa_depth_resolve_for_frame(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    frame: &FrameRenderParams<'_>,
    msaa: &ForwardMsaaResolvedViews,
    resolve: &MsaaDepthResolveResources,
) {
    let Some(limits) = frame.view.gpu_limits.as_ref() else {
        logger::warn!("MSAA depth resolve: gpu_limits missing; skipping resolve");
        return;
    };
    let limits = limits.as_ref();
    if msaa.is_array {
        let (Some(msaa_layers), Some(r32_layers)) = (
            msaa.stereo_depth_layer_views.as_ref(),
            msaa.stereo_r32_layer_views.as_ref(),
        ) else {
            return;
        };
        resolve.encode_resolve_stereo(
            device,
            encoder,
            frame.view.viewport_px,
            MsaaDepthResolveStereoTargets {
                msaa_depth_layer_views: [&msaa_layers[0], &msaa_layers[1]],
                r32_layer_views: [&r32_layers[0], &r32_layers[1]],
                r32_array_view: &msaa.depth_resolve_r32_view,
                dst_depth_view: frame.view.depth_view,
                dst_depth_format: frame.view.depth_texture.format(),
            },
            limits,
        );
    } else {
        resolve.encode_resolve(
            device,
            encoder,
            frame.view.viewport_px,
            MsaaDepthResolveMonoTargets {
                msaa_depth_view: &msaa.depth_view,
                r32_view: &msaa.depth_resolve_r32_view,
                dst_depth_view: frame.view.depth_view,
                dst_depth_format: frame.view.depth_texture.format(),
            },
            limits,
        );
    }
}

/// MSAA views resolved from the graph's transient resources for one forward pass execution.
pub(crate) struct ForwardMsaaResolvedViews {
    /// Depth-only multisampled view used by the compute depth resolve shader.
    pub depth_view: wgpu::TextureView,
    /// R32Float intermediate used by the MSAA depth resolve shader.
    pub depth_resolve_r32_view: wgpu::TextureView,
    /// `true` when [`Self::depth_view`] is a 2-layer `D2Array` (stereo multiview MSAA).
    pub is_array: bool,
    /// Per-eye `D2` single-layer views of the multisampled depth texture (stereo path only).
    pub stereo_depth_layer_views: Option<[wgpu::TextureView; 2]>,
    /// Per-eye `D2` single-layer views of the R32Float resolve temp (stereo path only).
    pub stereo_r32_layer_views: Option<[wgpu::TextureView; 2]>,
}

/// Resolves the MSAA transient textures for a forward pass when MSAA is active.
pub(crate) fn resolve_forward_msaa_views(
    graph_resources: Option<&GraphResolvedResources>,
    resources: WorldMeshForwardGraphResources,
    sample_count: u32,
    multiview_stereo: bool,
) -> Option<ForwardMsaaResolvedViews> {
    if sample_count <= 1 {
        return None;
    }
    let graph_resources = graph_resources?;
    graph_resources.transient_texture(resources.scene_color_hdr_msaa)?;
    let depth = graph_resources.transient_texture(resources.msaa_depth)?;
    let r32 = graph_resources.transient_texture(resources.msaa_depth_r32)?;
    let depth_view = depth_sample_view(depth, None);

    if multiview_stereo {
        let depth_layers = first_two_depth_sample_layer_views(depth)?;
        let r32_layers = first_two_layer_views(r32)?;
        Some(ForwardMsaaResolvedViews {
            depth_view,
            depth_resolve_r32_view: r32.view.clone(),
            is_array: true,
            stereo_depth_layer_views: Some(depth_layers),
            stereo_r32_layer_views: Some(r32_layers),
        })
    } else {
        Some(ForwardMsaaResolvedViews {
            depth_view,
            depth_resolve_r32_view: r32.view.clone(),
            is_array: false,
            stereo_depth_layer_views: None,
            stereo_r32_layer_views: None,
        })
    }
}

fn first_two_layer_views(texture: &ResolvedGraphTexture) -> Option<[wgpu::TextureView; 2]> {
    Some([
        texture.layer_views.first()?.clone(),
        texture.layer_views.get(1)?.clone(),
    ])
}

fn depth_sample_view(texture: &ResolvedGraphTexture, layer: Option<u32>) -> wgpu::TextureView {
    texture.texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("forward-msaa-depth-sample-view"),
        dimension: Some(wgpu::TextureViewDimension::D2),
        base_array_layer: layer.unwrap_or(0),
        array_layer_count: Some(1),
        aspect: wgpu::TextureAspect::DepthOnly,
        ..Default::default()
    })
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
