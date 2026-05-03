//! Per-frame compute + render-pass encoders for MSAA depth resolve.
//!
//! Free functions consumed by [`super::MsaaDepthResolveResources`]; they take everything
//! they need by reference so the encoder paths stay independent of the resource bundle's
//! field shape.

use crate::gpu::depth::MAIN_FORWARD_DEPTH_CLEAR;
use crate::gpu::limits::GpuLimits;
use crate::profiling::{
    GpuProfilerHandle, compute_pass_timestamp_writes, render_pass_timestamp_writes,
};

use super::MsaaDepthResolveResources;
use super::targets::{MsaaDepthResolveMonoTargets, MsaaDepthResolveStereoTargets};

/// Encodes the desktop (non-multiview) MSAA depth resolve into `encoder`.
///
/// Skips both passes when the 8x8-tiled compute dispatch would exceed
/// [`GpuLimits::compute_dispatch_fits`] (logged), or when the destination format has no
/// blit pipeline (logged).
pub(super) fn encode_resolve(
    resources: &MsaaDepthResolveResources,
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    extent: (u32, u32),
    targets: MsaaDepthResolveMonoTargets<'_>,
    limits: &GpuLimits,
    profiler: Option<&GpuProfilerHandle>,
) {
    let MsaaDepthResolveMonoTargets {
        msaa_depth_view,
        r32_view,
        dst_depth_view,
        dst_depth_format,
    } = targets;
    let (w, h) = (extent.0.max(1), extent.1.max(1));
    let gx = w.div_ceil(8);
    let gy = h.div_ceil(8);
    if !limits.compute_dispatch_fits(gx, gy, 1) {
        logger::warn!(
            "MSAA depth resolve: dispatch {}x{}x1 exceeds max_compute_workgroups_per_dimension ({})",
            gx,
            gy,
            limits.max_compute_workgroups_per_dimension()
        );
        return;
    }

    let compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("msaa_depth_resolve_compute_bg"),
        layout: resources.compute_bgl(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(msaa_depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(r32_view),
            },
        ],
    });

    let blit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("msaa_depth_blit_bg"),
        layout: resources.blit_bgl(),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(r32_view),
        }],
    });

    let compute_query = profiler.map(|p| p.begin_pass_query("msaa_depth_resolve.compute", encoder));
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("msaa-depth-resolve-r32"),
            timestamp_writes: compute_pass_timestamp_writes(compute_query.as_ref()),
        });
        cpass.set_pipeline(resources.compute_pipeline());
        cpass.set_bind_group(0, &compute_bg, &[]);
        cpass.dispatch_workgroups(gx, gy, 1);
    };
    if let (Some(q), Some(p)) = (compute_query, profiler) {
        p.end_query(encoder, q);
    }

    let Some(blit_pipeline) = resources.blit_pipeline_for_format(dst_depth_format) else {
        logger::warn!(
            "MSAA depth resolve: mono blit pipeline missing for {:?} (DEPTH32FLOAT_STENCIL8 feature unavailable?)",
            dst_depth_format
        );
        return;
    };
    let blit_query = profiler.map(|p| p.begin_pass_query("msaa_depth_resolve.blit", encoder));
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("msaa-depth-blit-r32-to-depth"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: dst_depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(MAIN_FORWARD_DEPTH_CLEAR),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: render_pass_timestamp_writes(blit_query.as_ref()),
            multiview_mask: None,
        });
        rpass.set_pipeline(blit_pipeline);
        rpass.set_bind_group(0, &blit_bg, &[]);
        rpass.draw(0..3, 0..1);
    };
    if let (Some(q), Some(p)) = (blit_query, profiler) {
        p.end_query(encoder, q);
    }
}

/// Encodes the stereo (OpenXR multiview) MSAA depth resolve into `encoder`.
///
/// Issues two compute dispatches (one per eye) because WGSL lacks
/// `texture_depth_multisampled_2d_array` today, then one multiview blit pass
/// (`multiview_mask = 0b11`) that writes both depth layers via `@builtin(view_index)`.
/// Does nothing when [`wgpu::Features::MULTIVIEW`] was unavailable at construction.
pub(super) fn encode_resolve_stereo(
    resources: &MsaaDepthResolveResources,
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    extent: (u32, u32),
    targets: MsaaDepthResolveStereoTargets<'_>,
    limits: &GpuLimits,
    profiler: Option<&GpuProfilerHandle>,
) {
    let MsaaDepthResolveStereoTargets {
        msaa_depth_layer_views,
        r32_layer_views,
        r32_array_view,
        dst_depth_view,
        dst_depth_format,
    } = targets;
    let Some(blit_stereo_pipeline) = resources.stereo_blit_pipeline_for_format(dst_depth_format)
    else {
        return;
    };
    let Some(blit_stereo_bgl) = resources.blit_stereo_bgl() else {
        return;
    };
    let (w, h) = (extent.0.max(1), extent.1.max(1));
    let gx = w.div_ceil(8);
    let gy = h.div_ceil(8);
    if !limits.compute_dispatch_fits(gx, gy, 1) {
        logger::warn!(
            "MSAA depth resolve (stereo): dispatch {}x{}x1 exceeds max_compute_workgroups_per_dimension ({})",
            gx,
            gy,
            limits.max_compute_workgroups_per_dimension()
        );
        return;
    }

    for eye in 0..2 {
        let compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("msaa_depth_resolve_compute_bg_stereo"),
            layout: resources.compute_bgl(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(msaa_depth_layer_views[eye]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(r32_layer_views[eye]),
                },
            ],
        });
        let label = if eye == 0 {
            "msaa_depth_resolve.compute_stereo_eye0"
        } else {
            "msaa_depth_resolve.compute_stereo_eye1"
        };
        let compute_query = profiler.map(|p| p.begin_pass_query(label, encoder));
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("msaa-depth-resolve-r32-stereo"),
                timestamp_writes: compute_pass_timestamp_writes(compute_query.as_ref()),
            });
            cpass.set_pipeline(resources.compute_pipeline());
            cpass.set_bind_group(0, &compute_bg, &[]);
            cpass.dispatch_workgroups(gx, gy, 1);
        };
        if let (Some(q), Some(p)) = (compute_query, profiler) {
            p.end_query(encoder, q);
        }
    }

    let blit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("msaa_depth_blit_bg_stereo"),
        layout: blit_stereo_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(r32_array_view),
        }],
    });

    let blit_query =
        profiler.map(|p| p.begin_pass_query("msaa_depth_resolve.blit_stereo", encoder));
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("msaa-depth-blit-r32-to-depth-stereo"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: dst_depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(MAIN_FORWARD_DEPTH_CLEAR),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: render_pass_timestamp_writes(blit_query.as_ref()),
            multiview_mask: std::num::NonZeroU32::new(3),
        });
        rpass.set_pipeline(blit_stereo_pipeline);
        rpass.set_bind_group(0, &blit_bg, &[]);
        rpass.draw(0..3, 0..1);
    };
    if let (Some(q), Some(p)) = (blit_query, profiler) {
        p.end_query(encoder, q);
    }
}
