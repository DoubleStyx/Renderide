//! Shared helpers for [`super::CompiledRenderGraph`] execution (resolution, raster templates).

use std::sync::Arc;

use crate::backend::RenderBackend;
use crate::gpu::{GpuLimits, MsaaDepthResolveResources};
use crate::scene::SceneCoordinator;

use super::super::context::{GraphResolvedResources, RasterPassCtx, ResolvedGraphTexture};
use super::super::error::GraphExecuteError;
use super::super::frame_params::HostCameraFrame;
use super::super::frame_params::{FrameRenderParams, FrameRenderParamsView, FrameSystemsShared};
use super::super::pass::PassNode;
use super::super::resources::{
    BufferSizePolicy, TextureAttachmentResolve, TextureAttachmentTarget, TextureHandle,
    TextureResourceHandle, TransientExtent,
};
use super::super::transient_pool::TextureKey;
use super::super::world_mesh_draw_prep::CameraTransformDrawFilter;

use super::{CompiledPassInfo, RenderPassTemplate, ResolvedView};

/// Per-view inputs for [`frame_render_params_from_shared`].
///
/// Groups the view-side data that would otherwise inflate the builder's parameter list: the
/// resolved surface handles, host camera, per-view overrides, and the GPU / MSAA / Hi-Z resources
/// scoped to this view.
pub(super) struct FrameRenderParamsViewInputs<'a, 'r> {
    /// Resolved surface targets, viewport, and view flags for this view.
    pub resolved: &'r ResolvedView<'a>,
    /// Scene color format used by the render graph.
    pub scene_color_format: wgpu::TextureFormat,
    /// Host camera inputs forwarded to per-pass logic.
    pub host_camera: HostCameraFrame,
    /// Optional per-camera draw-list filter applied before world-mesh recording.
    pub transform_draw_filter: Option<CameraTransformDrawFilter>,
    /// GPU capability limits, shared with passes that need to clamp against them.
    pub gpu_limits: Option<Arc<GpuLimits>>,
    /// MSAA depth resolve helpers when MSAA is active.
    pub msaa_depth_resolve: Option<Arc<MsaaDepthResolveResources>>,
    /// Per-camera Hi-Z state slot.
    pub hi_z_slot: Arc<parking_lot::Mutex<crate::render_graph::occlusion::HiZGpuState>>,
}

/// Builds [`FrameRenderParams`] from pre-split shared backend slices and per-view surface state.
pub(super) fn frame_render_params_from_shared<'a>(
    shared: FrameSystemsShared<'a>,
    view_inputs: FrameRenderParamsViewInputs<'a, '_>,
) -> FrameRenderParams<'a> {
    let FrameRenderParamsViewInputs {
        resolved,
        scene_color_format,
        host_camera,
        transform_draw_filter,
        gpu_limits,
        msaa_depth_resolve,
        hi_z_slot,
    } = view_inputs;
    let depth_sample_view = resolved
        .depth_texture
        .create_view(&wgpu::TextureViewDescriptor {
            label: Some("depth_sample"),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
    FrameRenderParams {
        shared,
        view: FrameRenderParamsView {
            depth_texture: resolved.depth_texture,
            depth_view: resolved.depth_view,
            depth_sample_view: Some(depth_sample_view),
            surface_format: resolved.surface_format,
            scene_color_format,
            viewport_px: resolved.viewport_px,
            host_camera,
            multiview_stereo: resolved.multiview_stereo,
            transform_draw_filter,
            offscreen_write_render_texture_asset_id: resolved
                .offscreen_write_render_texture_asset_id,
            occlusion_view: resolved.occlusion_view,
            hi_z_slot,
            sample_count: resolved.sample_count,
            gpu_limits,
            msaa_depth_resolve,
            // MSAA views now live in the per-view blackboard (MsaaViewsSlot), resolved from
            // graph transient textures by the executor via resolve_forward_msaa_views_from_graph_resources.
        },
    }
}

/// Builds [`FrameRenderParams`] from a resolved target and per-view host/IPC fields.
pub(super) fn frame_render_params_from_resolved<'a>(
    scene: &'a SceneCoordinator,
    backend: &'a mut RenderBackend,
    resolved: &ResolvedView<'a>,
    host_camera: HostCameraFrame,
    transform_draw_filter: Option<CameraTransformDrawFilter>,
) -> FrameRenderParams<'a> {
    let scene_color_format = backend.scene_color_format_wgpu();
    let (
        occlusion,
        frame_resources,
        materials,
        asset_transfers,
        mesh_preprocess,
        mesh_deform_scratch,
        skin_cache,
        gpu_limits,
        msaa_depth_resolve,
        debug_hud,
    ) = backend.split_for_graph_frame_params();
    let hi_z_slot = occlusion.ensure_hi_z_state(resolved.occlusion_view);
    frame_render_params_from_shared(
        FrameSystemsShared {
            scene,
            occlusion,
            frame_resources,
            materials,
            asset_transfers,
            mesh_preprocess,
            mesh_deform_scratch,
            mesh_deform_skin_cache: skin_cache,
            skin_cache: None,
            debug_hud,
        },
        FrameRenderParamsViewInputs {
            resolved,
            scene_color_format,
            host_camera,
            transform_draw_filter,
            gpu_limits,
            msaa_depth_resolve,
            hi_z_slot,
        },
    )
}

fn first_two_layer_views(texture: &ResolvedGraphTexture) -> Option<[wgpu::TextureView; 2]> {
    Some([
        texture.layer_views.first()?.clone(),
        texture.layer_views.get(1)?.clone(),
    ])
}

/// Resolves MSAA attachment views from graph transient textures for the main graph.
///
/// Returns `None` when MSAA is inactive (`sample_count <= 1`), graph resources are missing,
/// or the transient handles are unavailable. The executor inserts the returned value into the
/// per-view [`super::super::blackboard::Blackboard`] as a
/// [`super::super::frame_params::MsaaViewsSlot`].
pub(super) fn resolve_forward_msaa_views_from_graph_resources(
    frame: &FrameRenderParams<'_>,
    graph_resources: Option<&GraphResolvedResources>,
    msaa_handles: Option<[TextureHandle; 3]>,
) -> Option<super::super::frame_params::MsaaViews> {
    let handles = msaa_handles?;
    let [color_h, depth_h, r32_h] = handles;
    let graph_resources = graph_resources?;
    if frame.view.sample_count <= 1 {
        return None;
    }
    let color = graph_resources.transient_texture(color_h)?;
    let depth = graph_resources.transient_texture(depth_h)?;
    let r32 = graph_resources.transient_texture(r32_h)?;

    if frame.view.multiview_stereo {
        let depth_layers = first_two_layer_views(depth)?;
        let r32_layers = first_two_layer_views(r32)?;
        Some(super::super::frame_params::MsaaViews {
            msaa_color_view: color.view.clone(),
            msaa_depth_view: depth.view.clone(),
            msaa_depth_resolve_r32_view: r32.view.clone(),
            msaa_depth_is_array: true,
            msaa_stereo_depth_layer_views: Some(depth_layers),
            msaa_stereo_r32_layer_views: Some(r32_layers),
        })
    } else {
        Some(super::super::frame_params::MsaaViews {
            msaa_color_view: color.view.clone(),
            msaa_depth_view: depth.view.clone(),
            msaa_depth_resolve_r32_view: r32.view.clone(),
            msaa_depth_is_array: false,
            msaa_stereo_depth_layer_views: None,
            msaa_stereo_r32_layer_views: None,
        })
    }
}

pub(super) fn resolve_transient_extent(
    extent: TransientExtent,
    viewport_px: (u32, u32),
    array_layers: u32,
) -> TransientExtent {
    match extent {
        TransientExtent::Backbuffer if array_layers > 1 => TransientExtent::MultiLayer {
            width: viewport_px.0.max(1),
            height: viewport_px.1.max(1),
            layers: array_layers,
        },
        TransientExtent::Backbuffer => TransientExtent::Custom {
            width: viewport_px.0.max(1),
            height: viewport_px.1.max(1),
        },
        TransientExtent::BackbufferScaledMip { max_dim, mip } => {
            let (vw, vh) = (viewport_px.0.max(1), viewport_px.1.max(1));
            let ratio = f64::from(max_dim.max(1)) / f64::from(vh);
            let base_w = ((f64::from(vw) * ratio).round() as u32).max(1);
            let base_h = ((f64::from(vh) * ratio).round() as u32).max(1);
            let w = (base_w >> mip).max(1);
            let h = (base_h >> mip).max(1);
            if array_layers > 1 {
                TransientExtent::MultiLayer {
                    width: w,
                    height: h,
                    layers: array_layers,
                }
            } else {
                TransientExtent::Custom {
                    width: w,
                    height: h,
                }
            }
        }
        other => other,
    }
}

/// Clamps viewport dimensions to [`wgpu::Limits::max_texture_dimension_2d`] before transient texture
/// or buffer allocation from viewport-derived sizes.
pub(super) fn clamp_viewport_for_transient_alloc(
    viewport_px: (u32, u32),
    max_texture_dimension_2d: u32,
) -> (u32, u32) {
    let ow = viewport_px.0.max(1);
    let oh = viewport_px.1.max(1);
    let w = ow.min(max_texture_dimension_2d);
    let h = oh.min(max_texture_dimension_2d);
    if w != ow || h != oh {
        logger::warn!(
            "transient alloc: viewport {}×{} clamped to {}×{} (max_texture_dimension_2d={max_texture_dimension_2d})",
            ow,
            oh,
            w,
            h,
        );
    }
    (w, h)
}

pub(super) fn resolve_buffer_size(size_policy: BufferSizePolicy, viewport_px: (u32, u32)) -> u64 {
    match size_policy {
        BufferSizePolicy::Fixed(size) => size.max(1),
        BufferSizePolicy::PerViewport { bytes_per_px } => u64::from(viewport_px.0.max(1))
            .saturating_mul(u64::from(viewport_px.1.max(1)))
            .saturating_mul(bytes_per_px)
            .max(1),
    }
}

pub(super) fn create_transient_layer_views(
    texture: &wgpu::Texture,
    key: TextureKey,
) -> Vec<wgpu::TextureView> {
    if key.dimension != wgpu::TextureDimension::D2 || key.array_layers <= 1 {
        return Vec::new();
    }
    (0..key.array_layers)
        .map(|layer| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("render-graph-transient-layer"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            })
        })
        .collect()
}

pub(super) fn pass_info_raster_template(
    pass_info: &[CompiledPassInfo],
    pass_idx: usize,
) -> Result<RenderPassTemplate, GraphExecuteError> {
    let Some(info) = pass_info.get(pass_idx) else {
        return Err(GraphExecuteError::MissingRasterTemplate {
            pass: format!("pass#{pass_idx}"),
        });
    };
    info.raster_template
        .clone()
        .ok_or_else(|| GraphExecuteError::MissingRasterTemplate {
            pass: info.name.clone(),
        })
}

pub(super) fn frame_sample_count_from_raster_ctx(ctx: &RasterPassCtx<'_, '_>) -> u32 {
    ctx.frame
        .as_ref()
        .map(|frame| frame.view.sample_count.max(1))
        .unwrap_or(1)
}

pub(super) fn resolve_attachment_target(
    target: TextureAttachmentTarget,
    sample_count: u32,
) -> TextureResourceHandle {
    match target {
        TextureAttachmentTarget::Resource(handle) => handle,
        TextureAttachmentTarget::FrameSampled {
            single_sample,
            multisampled,
        } => {
            if sample_count > 1 {
                multisampled
            } else {
                single_sample
            }
        }
    }
}

pub(super) fn resolve_attachment_resolve_target(
    target: TextureAttachmentResolve,
    sample_count: u32,
) -> Option<TextureResourceHandle> {
    match target {
        TextureAttachmentResolve::Always(handle) => Some(handle),
        TextureAttachmentResolve::FrameMultisampled(handle) => (sample_count > 1).then_some(handle),
    }
}

/// Opens a graph-managed raster render pass for a [`PassNode::Raster`] variant and calls
/// [`super::super::pass::PassNode::record_raster`].
///
/// This is the primary path for raster passes in the new pass-node system.
pub(super) fn execute_graph_raster_pass_node(
    pass: &PassNode,
    template: &RenderPassTemplate,
    graph_resources: &GraphResolvedResources,
    encoder: &mut wgpu::CommandEncoder,
    ctx: &mut RasterPassCtx<'_, '_>,
) -> Result<(), GraphExecuteError> {
    let sample_count = frame_sample_count_from_raster_ctx(ctx);
    let pass_name = pass.name();

    let color_attachments = {
        profiling::scope!("graph::raster::resolve_color_attachments");
        let mut color_attachments = Vec::with_capacity(template.color_attachments.len());
        for color in &template.color_attachments {
            let target = resolve_attachment_target(color.target, sample_count);
            let view = graph_resources.texture_view(target).ok_or_else(|| {
                GraphExecuteError::MissingGraphAttachment {
                    pass: pass_name.to_owned(),
                    resource: format!("{target:?}"),
                }
            })?;
            let resolve_target = match color
                .resolve_to
                .and_then(|t| resolve_attachment_resolve_target(t, sample_count))
            {
                Some(target) => Some(graph_resources.texture_view(target).ok_or_else(|| {
                    GraphExecuteError::MissingGraphAttachment {
                        pass: pass_name.to_owned(),
                        resource: format!("{target:?}"),
                    }
                })?),
                None => None,
            };
            color_attachments.push(Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target,
                ops: wgpu::Operations {
                    load: color.load,
                    store: color.store,
                },
                depth_slice: None,
            }));
        }
        color_attachments
    };

    let depth_stencil_attachment = if let Some(depth) = &template.depth_stencil_attachment {
        let target = resolve_attachment_target(depth.target, sample_count);
        let view = graph_resources.texture_view(target).ok_or_else(|| {
            GraphExecuteError::MissingGraphAttachment {
                pass: pass_name.to_owned(),
                resource: format!("{target:?}"),
            }
        })?;
        let stencil_ops = pass.stencil_ops_override(ctx, depth);
        Some(wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(depth.depth),
            stencil_ops,
        })
    } else {
        None
    };

    let multiview_mask = pass.multiview_mask_override(ctx, template);
    let pass_query = ctx.profiler.map(|p| p.begin_pass_query(pass_name, encoder));
    let timestamp_writes = crate::profiling::render_pass_timestamp_writes(pass_query.as_ref());
    let mut rpass = {
        profiling::scope!("graph::raster::begin_render_pass");
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render-graph-raster"),
            color_attachments: &color_attachments,
            depth_stencil_attachment,
            occlusion_query_set: None,
            timestamp_writes,
            multiview_mask,
        })
    };
    {
        profiling::scope!("graph::raster::record_draws");
        pass.record_raster(ctx, &mut rpass)
            .map_err(GraphExecuteError::Pass)?;
    }
    {
        profiling::scope!("graph::raster::end_render_pass");
        drop(rpass);
    }
    if let (Some(p), Some(q)) = (ctx.profiler, pass_query) {
        p.end_query(encoder, q);
    }
    Ok(())
}
