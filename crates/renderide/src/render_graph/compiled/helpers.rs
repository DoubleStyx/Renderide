//! Shared helpers for [`super::CompiledRenderGraph`] execution (resolution, raster templates).

use winit::window::Window;

use crate::backend::RenderBackend;
use crate::gpu::GpuContext;
use crate::present::{acquire_surface_outcome, SurfaceFrameOutcome};
use crate::scene::SceneCoordinator;

use super::super::context::{GraphRasterPassContext, GraphResolvedResources, ResolvedGraphTexture};
use super::super::error::GraphExecuteError;
use super::super::frame_params::FrameRenderParams;
use super::super::frame_params::HostCameraFrame;
use super::super::pass::RenderPass;
use super::super::resources::{
    BufferSizePolicy, TextureAttachmentResolve, TextureAttachmentTarget, TextureHandle,
    TextureResourceHandle, TransientExtent,
};
use super::super::transient_pool::TextureKey;
use super::super::world_mesh_draw_prep::{CameraTransformDrawFilter, WorldMeshDrawCollection};

use super::{CompiledPassInfo, RenderPassTemplate, ResolvedView};

/// Builds [`FrameRenderParams`] from a resolved target and per-view host/IPC fields.
pub(super) fn frame_render_params_from_resolved<'a>(
    scene: &'a SceneCoordinator,
    backend: &'a mut RenderBackend,
    resolved: &ResolvedView<'a>,
    host_camera: HostCameraFrame,
    transform_draw_filter: Option<CameraTransformDrawFilter>,
    prefetched_world_mesh_draws: Option<WorldMeshDrawCollection>,
) -> FrameRenderParams<'a> {
    let depth_sample_view = resolved
        .depth_texture
        .create_view(&wgpu::TextureViewDescriptor {
            label: Some("depth_sample"),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
    FrameRenderParams {
        scene,
        backend,
        depth_texture: resolved.depth_texture,
        depth_view: resolved.depth_view,
        depth_sample_view: Some(depth_sample_view),
        surface_format: resolved.surface_format,
        viewport_px: resolved.viewport_px,
        host_camera,
        multiview_stereo: resolved.multiview_stereo,
        transform_draw_filter,
        offscreen_write_render_texture_asset_id: resolved.offscreen_write_render_texture_asset_id,
        prefetched_world_mesh_draws,
        prepared_world_mesh_forward: None,
        occlusion_view: resolved.occlusion_view,
        sample_count: resolved.sample_count,
        msaa_color_view: resolved.msaa_color_view.clone(),
        msaa_depth_view: resolved.msaa_depth_view.clone(),
        msaa_depth_resolve_r32_view: resolved.msaa_depth_resolve_r32_view.clone(),
        msaa_depth_is_array: resolved.msaa_depth_is_array,
        msaa_stereo_depth_layer_views: resolved.msaa_stereo_depth_layer_views.clone(),
        msaa_stereo_r32_layer_views: resolved.msaa_stereo_r32_layer_views.clone(),
    }
}

fn first_two_layer_views(texture: &ResolvedGraphTexture) -> Option<[wgpu::TextureView; 2]> {
    Some([
        texture.layer_views.first()?.clone(),
        texture.layer_views.get(1)?.clone(),
    ])
}

/// Fills [`FrameRenderParams`] MSAA attachment views from resolved graph transients (main graph only).
pub(super) fn populate_forward_msaa_from_graph_resources(
    frame: &mut FrameRenderParams<'_>,
    graph_resources: Option<&GraphResolvedResources>,
    msaa_handles: Option<[TextureHandle; 3]>,
) {
    let Some(handles) = msaa_handles else {
        return;
    };
    let [color_h, depth_h, r32_h] = handles;
    let Some(graph_resources) = graph_resources else {
        return;
    };
    if frame.sample_count <= 1 {
        return;
    }
    let (Some(color), Some(depth), Some(r32)) = (
        graph_resources.transient_texture(color_h),
        graph_resources.transient_texture(depth_h),
        graph_resources.transient_texture(r32_h),
    ) else {
        return;
    };

    if frame.multiview_stereo {
        let (Some(depth_layers), Some(r32_layers)) =
            (first_two_layer_views(depth), first_two_layer_views(r32))
        else {
            return;
        };
        frame.msaa_color_view = Some(color.view.clone());
        frame.msaa_depth_view = Some(depth.view.clone());
        frame.msaa_depth_resolve_r32_view = Some(r32.view.clone());
        frame.msaa_depth_is_array = true;
        frame.msaa_stereo_depth_layer_views = Some(depth_layers);
        frame.msaa_stereo_r32_layer_views = Some(r32_layers);
    } else {
        frame.msaa_color_view = Some(color.view.clone());
        frame.msaa_depth_view = Some(depth.view.clone());
        frame.msaa_depth_resolve_r32_view = Some(r32.view.clone());
        frame.msaa_depth_is_array = false;
        frame.msaa_stereo_depth_layer_views = None;
        frame.msaa_stereo_r32_layer_views = None;
    }
}

/// Outcome of swapchain acquisition for [`CompiledRenderGraph::execute_multi_view`].
pub(super) enum MultiViewSwapchainAcquire {
    /// No swapchain view required (no swapchain pass, or graph does not bind the backbuffer).
    NotNeeded,
    /// Skip this frame’s GPU work (timeout, occluded, or swapchain reconfigured).
    SkipPresent,
    /// Surface texture and default view for per-view and present.
    Acquired {
        /// Surface texture presented at the end of multi-view execution when present.
        frame: wgpu::SurfaceTexture,
        /// View used as the swapchain color attachment across views.
        backbuffer_view: wgpu::TextureView,
    },
}

/// Acquires the window swapchain when any [`FrameView`] targets [`FrameViewTarget::Swapchain`].
pub(super) fn acquire_swapchain_for_multi_view_if_needed(
    needs_swapchain: bool,
    needs_surface_acquire: bool,
    gpu: &mut GpuContext,
    window: &Window,
) -> Result<MultiViewSwapchainAcquire, GraphExecuteError> {
    if !needs_swapchain {
        return Ok(MultiViewSwapchainAcquire::NotNeeded);
    }
    if !needs_surface_acquire {
        return Ok(MultiViewSwapchainAcquire::NotNeeded);
    }
    match acquire_surface_outcome(gpu, window)? {
        SurfaceFrameOutcome::Skip | SurfaceFrameOutcome::Reconfigured => {
            Ok(MultiViewSwapchainAcquire::SkipPresent)
        }
        SurfaceFrameOutcome::Acquired(tex) => {
            let backbuffer_view = tex
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            Ok(MultiViewSwapchainAcquire::Acquired {
                frame: tex,
                backbuffer_view,
            })
        }
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

pub(super) fn frame_sample_count(ctx: &GraphRasterPassContext<'_, '_>) -> u32 {
    ctx.frame
        .as_ref()
        .map(|frame| frame.sample_count.max(1))
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

pub(super) fn execute_graph_managed_raster_pass(
    pass: &mut dyn RenderPass,
    template: &RenderPassTemplate,
    graph_resources: &GraphResolvedResources,
    encoder: &mut wgpu::CommandEncoder,
    ctx: &mut GraphRasterPassContext<'_, '_>,
) -> Result<(), GraphExecuteError> {
    let sample_count = frame_sample_count(ctx);
    let mut color_attachments = Vec::with_capacity(template.color_attachments.len());
    for color in &template.color_attachments {
        let target = resolve_attachment_target(color.target, sample_count);
        let view = graph_resources.texture_view(target).ok_or_else(|| {
            GraphExecuteError::MissingGraphAttachment {
                pass: pass.name().to_string(),
                resource: format!("{target:?}"),
            }
        })?;
        let resolve_target = match color
            .resolve_to
            .and_then(|target| resolve_attachment_resolve_target(target, sample_count))
        {
            Some(target) => Some(graph_resources.texture_view(target).ok_or_else(|| {
                GraphExecuteError::MissingGraphAttachment {
                    pass: pass.name().to_string(),
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

    let depth_stencil_attachment = if let Some(depth) = &template.depth_stencil_attachment {
        let target = resolve_attachment_target(depth.target, sample_count);
        let view = graph_resources.texture_view(target).ok_or_else(|| {
            GraphExecuteError::MissingGraphAttachment {
                pass: pass.name().to_string(),
                resource: format!("{target:?}"),
            }
        })?;
        Some(wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(depth.depth),
            stencil_ops: pass.graph_raster_stencil_ops(ctx, depth),
        })
    } else {
        None
    };

    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("render-graph-raster"),
        color_attachments: &color_attachments,
        depth_stencil_attachment,
        occlusion_query_set: None,
        timestamp_writes: None,
        multiview_mask: pass.graph_raster_multiview_mask(ctx, template),
    });
    pass.execute_graph_raster(ctx, &mut rpass)?;
    Ok(())
}
