//! Shared helpers for [`super::CompiledRenderGraph`] execution (resolution, raster templates).

use std::sync::Arc;

use crate::backend::BackendGraphAccess;
use crate::gpu::{GpuLimits, MsaaDepthResolveResources};
use crate::scene::SceneCoordinator;

use super::super::context::{GraphResolvedResources, RasterPassCtx, ResolvedGraphTexture};
use super::super::error::GraphExecuteError;
use super::super::frame_params::FrameViewClear;
use super::super::frame_params::{FrameSystemsShared, GraphPassFrame, GraphPassFrameView};
use super::super::pass::PassNode;
use super::super::pool::TextureKey;
use super::super::resources::{
    BufferSizePolicy, TextureAttachmentResolve, TextureAttachmentTarget, TextureHandle,
    TextureResourceHandle, TransientExtent,
};
use crate::camera::HostCameraFrame;
use crate::world_mesh::draw_prep::CameraTransformDrawFilter;

use super::{CompiledPassInfo, RenderPassTemplate, ResolvedView};

/// Per-view inputs for [`frame_render_params_from_shared`].
///
/// Groups the view-side data that would otherwise inflate the builder's parameter list: the
/// resolved surface handles, host camera, per-view overrides, and the GPU / MSAA / Hi-Z resources
/// scoped to this view.
pub(super) struct GraphPassFrameViewInputs<'a, 'r> {
    /// Resolved surface targets, viewport, and view flags for this view.
    pub resolved: &'r ResolvedView<'a>,
    /// Scene color format used by the render graph.
    pub scene_color_format: wgpu::TextureFormat,
    /// Host camera inputs forwarded to per-pass logic.
    pub host_camera: HostCameraFrame,
    /// Optional per-camera draw-list filter applied before world-mesh recording.
    pub transform_draw_filter: Option<CameraTransformDrawFilter>,
    /// Background clear/skybox behavior for this view.
    pub clear: FrameViewClear,
    /// GPU capability limits, shared with passes that need to clamp against them.
    pub gpu_limits: Option<Arc<GpuLimits>>,
    /// MSAA depth resolve helpers when MSAA is active.
    pub msaa_depth_resolve: Option<Arc<MsaaDepthResolveResources>>,
    /// Per-camera Hi-Z state slot.
    pub hi_z_slot: Arc<parking_lot::Mutex<crate::occlusion::gpu::HiZGpuState>>,
}

/// Builds [`GraphPassFrame`] from pre-split shared backend slices and per-view surface state.
pub(super) fn frame_render_params_from_shared<'a>(
    shared: FrameSystemsShared<'a>,
    view_inputs: GraphPassFrameViewInputs<'a, '_>,
) -> GraphPassFrame<'a> {
    let GraphPassFrameViewInputs {
        resolved,
        scene_color_format,
        host_camera,
        transform_draw_filter,
        clear,
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
    GraphPassFrame {
        shared,
        view: GraphPassFrameView {
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
            view_id: resolved.view_id,
            hi_z_slot,
            sample_count: resolved.sample_count,
            gpu_limits,
            msaa_depth_resolve,
            clear,
            // MSAA views now live in the per-view blackboard (MsaaViewsSlot), resolved from
            // graph transient textures by the executor via resolve_forward_msaa_views_from_graph_resources.
        },
    }
}

/// Builds [`GraphPassFrame`] from a resolved target and per-view host/IPC fields.
pub(super) fn frame_render_params_from_resolved<'a>(
    scene: &'a SceneCoordinator,
    backend: &'a mut BackendGraphAccess<'_>,
    resolved: &ResolvedView<'a>,
    host_camera: HostCameraFrame,
    transform_draw_filter: Option<CameraTransformDrawFilter>,
    clear: FrameViewClear,
) -> GraphPassFrame<'a> {
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
    let hi_z_slot = occlusion.ensure_hi_z_state(resolved.view_id);
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
        GraphPassFrameViewInputs {
            resolved,
            scene_color_format,
            host_camera,
            transform_draw_filter,
            clear,
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
/// Returns `None` when MSAA is inactive (`sample_count <= 1`) or the transient handles are
/// unavailable. The executor inserts the returned value into the
/// per-view [`super::super::blackboard::Blackboard`] as a
/// [`super::super::frame_params::MsaaViewsSlot`].
pub(super) fn resolve_forward_msaa_views_from_graph_resources(
    frame: &GraphPassFrame<'_>,
    graph_resources: &GraphResolvedResources,
    msaa_handles: Option<[TextureHandle; 3]>,
) -> Option<super::super::frame_params::MsaaViews> {
    let handles = msaa_handles?;
    let [color_h, depth_h, r32_h] = handles;
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
            resolve_backbuffer_scaled_mip_extent(max_dim, mip, viewport_px, array_layers)
        }
        other => other,
    }
}

/// Resolves a bloom-style backbuffer-relative mip extent without exceeding the current viewport.
fn resolve_backbuffer_scaled_mip_extent(
    max_dim: u32,
    mip: u32,
    viewport_px: (u32, u32),
    array_layers: u32,
) -> TransientExtent {
    let (vw, vh) = (viewport_px.0.max(1), viewport_px.1.max(1));
    let requested_h = power_of_two_at_or_below(max_dim.max(1));
    let viewport_h = power_of_two_at_or_below(vh);
    let base_h = requested_h.min(viewport_h).max(1);
    let ratio = f64::from(base_h) / f64::from(vh);
    let base_w = ((f64::from(vw) * ratio).round() as u32).max(1).min(vw);
    let w = mip_axis_extent(base_w, mip);
    let h = mip_axis_extent(base_h, mip);
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

/// Returns the largest power of two less than or equal to `value`.
fn power_of_two_at_or_below(value: u32) -> u32 {
    let value = value.max(1);
    1_u32 << (u32::BITS - value.leading_zeros() - 1)
}

/// Halves a base extent by `mip`, clamping degenerate high mips to one pixel.
fn mip_axis_extent(base: u32, mip: u32) -> u32 {
    match base.checked_shr(mip) {
        Some(size) => size.max(1),
        None => 1,
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
            "transient alloc: viewport {}x{} clamped to {}x{} (max_texture_dimension_2d={max_texture_dimension_2d})",
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
    ctx.pass_frame.view.sample_count.max(1)
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

/// Resolves the raster pass template's color attachments to live `wgpu::TextureView`s.
fn resolve_color_attachments<'a>(
    pass_name: &str,
    template: &RenderPassTemplate,
    graph_resources: &'a GraphResolvedResources,
    sample_count: u32,
) -> Result<Vec<Option<wgpu::RenderPassColorAttachment<'a>>>, GraphExecuteError> {
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
    Ok(color_attachments)
}

/// Resolves the raster pass template's depth/stencil attachment, if any, to a live view.
fn resolve_depth_attachment<'a>(
    pass: &PassNode,
    template: &RenderPassTemplate,
    graph_resources: &'a GraphResolvedResources,
    sample_count: u32,
    ctx: &RasterPassCtx<'_, '_>,
) -> Result<Option<wgpu::RenderPassDepthStencilAttachment<'a>>, GraphExecuteError> {
    let Some(depth) = &template.depth_stencil_attachment else {
        return Ok(None);
    };
    let target = resolve_attachment_target(depth.target, sample_count);
    let view = graph_resources.texture_view(target).ok_or_else(|| {
        GraphExecuteError::MissingGraphAttachment {
            pass: pass.name().to_owned(),
            resource: format!("{target:?}"),
        }
    })?;
    let stencil_ops = pass.stencil_ops_override(ctx, depth);
    Ok(Some(wgpu::RenderPassDepthStencilAttachment {
        view,
        depth_ops: Some(depth.depth),
        stencil_ops,
    }))
}

/// Opens a graph-managed raster render pass for a [`PassNode::Raster`] variant and calls
/// [`super::super::pass::PassNode::record_raster`].
///
/// This is the primary path for raster passes in the new pass-node system. Attachment resolution
/// is delegated to [`resolve_color_attachments`] and [`resolve_depth_attachment`]; the body here
/// owns only the wgpu render-pass lifecycle and the GPU-profiler query bracketing.
pub(super) fn execute_graph_raster_pass_node(
    pass: &PassNode,
    template: &RenderPassTemplate,
    graph_resources: &GraphResolvedResources,
    encoder: &mut wgpu::CommandEncoder,
    ctx: &mut RasterPassCtx<'_, '_>,
) -> Result<(), GraphExecuteError> {
    if !pass
        .should_record_raster(ctx)
        .map_err(GraphExecuteError::Pass)?
    {
        return Ok(());
    }

    let sample_count = frame_sample_count_from_raster_ctx(ctx);
    let color_attachments =
        resolve_color_attachments(pass.name(), template, graph_resources, sample_count)?;
    let depth_stencil_attachment =
        resolve_depth_attachment(pass, template, graph_resources, sample_count, ctx)?;
    let multiview_mask = pass.multiview_mask_override(ctx, template);

    let pass_profile_label = pass.profiling_label();
    let pass_query = ctx
        .profiler
        .map(|p| p.begin_pass_query(pass_profile_label, encoder));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backbuffer_scaled_mip_clamps_high_max_dimension_to_viewport_pot() {
        assert_eq!(
            resolve_transient_extent(
                TransientExtent::BackbufferScaledMip {
                    max_dim: 2048,
                    mip: 0,
                },
                (1920, 1080),
                1,
            ),
            TransientExtent::Custom {
                width: 1820,
                height: 1024,
            }
        );
    }

    #[test]
    fn backbuffer_scaled_mip_preserves_lower_configured_dimension() {
        assert_eq!(
            resolve_transient_extent(
                TransientExtent::BackbufferScaledMip {
                    max_dim: 512,
                    mip: 0,
                },
                (1920, 1080),
                1,
            ),
            TransientExtent::Custom {
                width: 910,
                height: 512,
            }
        );
    }

    #[test]
    fn backbuffer_scaled_mip_halves_from_clamped_base() {
        assert_eq!(
            resolve_transient_extent(
                TransientExtent::BackbufferScaledMip {
                    max_dim: 2048,
                    mip: 1,
                },
                (800, 600),
                1,
            ),
            TransientExtent::Custom {
                width: 341,
                height: 256,
            }
        );
    }

    #[test]
    fn backbuffer_scaled_mip_handles_tiny_viewports_and_high_mips() {
        assert_eq!(
            resolve_transient_extent(
                TransientExtent::BackbufferScaledMip {
                    max_dim: 2048,
                    mip: 5,
                },
                (32, 18),
                1,
            ),
            TransientExtent::Custom {
                width: 1,
                height: 1,
            }
        );
    }

    #[test]
    fn backbuffer_scaled_mip_preserves_multiview_layers() {
        assert_eq!(
            resolve_transient_extent(
                TransientExtent::BackbufferScaledMip {
                    max_dim: 2048,
                    mip: 0,
                },
                (1920, 1080),
                2,
            ),
            TransientExtent::MultiLayer {
                width: 1820,
                height: 1024,
                layers: 2,
            }
        );
    }
}
