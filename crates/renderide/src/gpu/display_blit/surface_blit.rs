//! Per-frame "host texture -> swapchain" blit for the local-user `BlitToDisplay` renderable.
//!
//! Clears the surface to `BlitToDisplayState.background_color`, restricts rasterization to the
//! centered fitted rect via [`wgpu::RenderPass::set_viewport`], and draws a single fullscreen
//! triangle that samples the source texture with optional axis flips. Letterbox bars stay in the
//! cleared color.
//!
//! Used by the app driver when the scene reports an active blit for the desktop window's display
//! index. Composes any debug HUD overlay on the same encoder, the same way [`super::super::vr_mirror`]
//! does for the VR mirror path.

use glam::Vec4;

use crate::gpu::GpuContext;
use crate::gpu::blit_kit::layout::sampled_2d_filtered_uv_layout;
use crate::gpu::blit_kit::pipeline::UvUniformBuffer;
use crate::gpu::blit_kit::sampler::linear_clamp_sampler;
use crate::present::{
    PresentClearError, SurfaceAcquireTrace, SurfaceFrameOutcome, SurfaceSubmitTrace,
    acquire_surface_outcome_traced, submit_surface_frame_traced,
};

use super::fit::{FittedRectPx, fit_rect_px, flip_uv_params};
use super::resources::DisplayBlitResources;

/// Source texture sampled by [`DisplayBlitResources::present_blit_to_surface`].
///
/// The view must be a 2D filterable float view (sRGB or linear) that lives at least as long as
/// this frame's `Queue::submit`. Pool entries already satisfy that.
#[derive(Clone, Copy, Debug)]
pub struct DisplayBlitSource<'a> {
    /// Sampleable color view (Texture2D, RenderTexture color, VideoTexture, or Desktop).
    pub view: &'a wgpu::TextureView,
    /// Width in texels, used for the fit rect aspect ratio.
    pub width: u32,
    /// Height in texels, used for the fit rect aspect ratio.
    pub height: u32,
    /// `BlitToDisplayState.flipHorizontally` flag (bit 0 of `_flags`).
    pub flip_horizontally: bool,
    /// `BlitToDisplayState.flipVertically` flag (bit 1 of `_flags`).
    pub flip_vertically: bool,
    /// `BlitToDisplayState.background_color`; clears the swapchain (and letterbox bars).
    ///
    /// Ignored when this source is used as a load/blend overlay.
    pub background_color: Vec4,
}

impl DisplayBlitResources {
    /// Acquires the desktop swapchain, clears it to `background_color`, and blits `source` into
    /// the centered fitted rect. Composes `overlay` (typically the debug HUD) on the same encoder.
    ///
    /// Returns `Ok(())` for both the success path and the no-op cases (`SkipFrame` /
    /// `Reconfigured` from surface acquire) so the caller does not need to distinguish between
    /// them; only catastrophic acquire/submit failures bubble up as [`PresentClearError`].
    pub fn present_blit_to_surface<F, E>(
        &mut self,
        gpu: &mut GpuContext,
        source: DisplayBlitSource<'_>,
        overlay: F,
    ) -> Result<(), PresentClearError>
    where
        F: FnOnce(&mut wgpu::CommandEncoder, &wgpu::TextureView, &mut GpuContext) -> Result<(), E>,
        E: std::fmt::Display,
    {
        self.present_blit_to_surface_traced(
            gpu,
            source,
            SurfaceAcquireTrace::DesktopBlitToDisplay,
            SurfaceSubmitTrace::DesktopBlitToDisplay,
            overlay,
        )
    }

    /// Acquires the desktop swapchain, blits `source`, and presents under source-specific traces.
    pub fn present_blit_to_surface_traced<F, E>(
        &mut self,
        gpu: &mut GpuContext,
        source: DisplayBlitSource<'_>,
        acquire_trace: SurfaceAcquireTrace,
        submit_trace: SurfaceSubmitTrace,
        overlay: F,
    ) -> Result<(), PresentClearError>
    where
        F: FnOnce(&mut wgpu::CommandEncoder, &wgpu::TextureView, &mut GpuContext) -> Result<(), E>,
        E: std::fmt::Display,
    {
        self.present_blit_to_surface_traced_with_overlay(
            gpu,
            source,
            None,
            acquire_trace,
            submit_trace,
            overlay,
        )
    }

    /// Acquires the desktop swapchain, blits `source`, optionally alpha-composites `surface_overlay`,
    /// and presents under source-specific traces.
    pub fn present_blit_to_surface_traced_with_overlay<F, E>(
        &mut self,
        gpu: &mut GpuContext,
        source: DisplayBlitSource<'_>,
        surface_overlay: Option<DisplayBlitSource<'_>>,
        acquire_trace: SurfaceAcquireTrace,
        submit_trace: SurfaceSubmitTrace,
        overlay: F,
    ) -> Result<(), PresentClearError>
    where
        F: FnOnce(&mut wgpu::CommandEncoder, &wgpu::TextureView, &mut GpuContext) -> Result<(), E>,
        E: std::fmt::Display,
    {
        profiling::scope!("display_blit::present");
        let frame = match acquire_surface_outcome_traced(gpu, acquire_trace)? {
            SurfaceFrameOutcome::Skip | SurfaceFrameOutcome::Reconfigured => return Ok(()),
            SurfaceFrameOutcome::Acquired(f) => f,
        };

        let surface_format = gpu.config_format();
        let (sw, sh) = gpu.surface_extent_px();
        let sw = sw.max(1);
        let sh = sh.max(1);
        let rect = fit_rect_px(source.width, source.height, sw, sh);

        // Clone the device Arc so `device` doesn't hold a borrow on `gpu`; the GPU profiler below
        // needs `gpu.gpu_profiler_mut()` which is `&mut self`.
        let device_arc = gpu.device().clone();
        let device = device_arc.as_ref();
        self.ensure_uniform(device);
        let Some(uniform_buf) = self.uniform().get() else {
            logger::warn!("display_blit: uniform buffer missing after ensure_uniform");
            submit_surface_frame_traced(gpu, Vec::new(), frame, submit_trace);
            return Ok(());
        };
        write_source_uv(gpu, self.uniform(), source);

        let surface_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        crate::profiling::note_resource_churn!(TextureView, "gpu::display_blit_surface_view");
        let sampler = linear_clamp_sampler(device);

        let bind_group = create_surface_blit_bind_group(
            device,
            "display_blit_surface",
            source.view,
            sampler,
            uniform_buf,
        );
        crate::profiling::note_resource_churn!(BindGroup, "gpu::display_blit_bind_group");
        let prepared_base = PreparedBaseBlit {
            bind_group,
            rect,
            background_color: source.background_color,
        };
        let prepared_overlay = surface_overlay.and_then(|source| {
            prepare_surface_overlay_source(self, gpu, device, sampler, source, sw, sh)
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("display_blit_surface"),
        });
        let outer_query = gpu
            .gpu_profiler_mut()
            .map(|p| p.begin_query("graph::display_blit.surface", &mut encoder));
        encode_display_blit_passes(
            self,
            gpu,
            &mut encoder,
            SurfaceBlitPassCtx {
                device,
                surface_format,
                surface_view: &surface_view,
            },
            &prepared_base,
            prepared_overlay.as_ref(),
        );

        if let Err(e) = overlay(&mut encoder, &surface_view, gpu) {
            logger::warn!("debug HUD overlay (display blit): {e}");
        }
        if let Some(query) = outer_query
            && let Some(prof) = gpu.gpu_profiler_mut()
        {
            prof.end_query(&mut encoder, query);
            prof.resolve_queries(&mut encoder);
        }

        // Hand the surface texture to the driver thread alongside the command buffer so the real
        // `Queue::submit` runs **before** `SurfaceTexture::present`. Same constraint as the VR
        // mirror blit path -- presenting before submit destroys the texture and drops our work.
        let command_buffer = {
            profiling::scope!("CommandEncoder::finish::display_blit_surface");
            encoder.finish()
        };
        submit_surface_frame_traced(gpu, vec![command_buffer], frame, submit_trace);
        Ok(())
    }
}

struct PreparedBaseBlit {
    bind_group: wgpu::BindGroup,
    rect: FittedRectPx,
    background_color: Vec4,
}

struct PreparedOverlayBlit {
    bind_group: wgpu::BindGroup,
    rect: FittedRectPx,
}

#[derive(Clone, Copy)]
struct SurfaceBlitPassCtx<'a> {
    device: &'a wgpu::Device,
    surface_format: wgpu::TextureFormat,
    surface_view: &'a wgpu::TextureView,
}

fn encode_display_blit_passes(
    resources: &mut DisplayBlitResources,
    gpu: &mut GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    ctx: SurfaceBlitPassCtx<'_>,
    base: &PreparedBaseBlit,
    overlay: Option<&PreparedOverlayBlit>,
) {
    let blit_query = gpu
        .gpu_profiler_mut()
        .map(|p| p.begin_pass_query("graph::display_blit.surface.pass", encoder));
    let blit_timestamp_writes = crate::profiling::render_pass_timestamp_writes(blit_query.as_ref());
    let pipeline = resources.pipeline_for_format(ctx.device, ctx.surface_format);
    encode_display_blit_pass(
        encoder,
        ctx.surface_view,
        pipeline,
        &base.bind_group,
        base.rect,
        base.background_color,
        blit_timestamp_writes,
    );
    if let Some(query) = blit_query
        && let Some(prof) = gpu.gpu_profiler_mut()
    {
        prof.end_query(encoder, query);
    }
    if let Some(prepared) = overlay {
        encode_display_overlay_blit(resources, gpu, encoder, ctx, prepared);
    }
}

fn encode_display_overlay_blit(
    resources: &mut DisplayBlitResources,
    gpu: &mut GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    ctx: SurfaceBlitPassCtx<'_>,
    prepared: &PreparedOverlayBlit,
) {
    let overlay_query = gpu
        .gpu_profiler_mut()
        .map(|p| p.begin_pass_query("graph::display_blit.dashboard_overlay.pass", encoder));
    let overlay_timestamp_writes =
        crate::profiling::render_pass_timestamp_writes(overlay_query.as_ref());
    let pipeline = resources.overlay_pipeline_for_format(ctx.device, ctx.surface_format);
    encode_display_overlay_blit_pass(
        encoder,
        ctx.surface_view,
        pipeline,
        &prepared.bind_group,
        prepared.rect,
        overlay_timestamp_writes,
    );
    if let Some(query) = overlay_query
        && let Some(prof) = gpu.gpu_profiler_mut()
    {
        prof.end_query(encoder, query);
    }
}

fn prepare_surface_overlay_source(
    resources: &mut DisplayBlitResources,
    gpu: &GpuContext,
    device: &wgpu::Device,
    sampler: &wgpu::Sampler,
    source: DisplayBlitSource<'_>,
    surface_width: u32,
    surface_height: u32,
) -> Option<PreparedOverlayBlit> {
    resources.ensure_overlay_uniform(device);
    let Some(uniform_buf) = resources.overlay_uniform().get() else {
        logger::warn!("display_blit: overlay uniform buffer missing after ensure_overlay_uniform");
        return None;
    };
    write_source_uv(gpu, resources.overlay_uniform(), source);
    let bind_group = create_surface_blit_bind_group(
        device,
        "display_blit_dashboard_overlay",
        source.view,
        sampler,
        uniform_buf,
    );
    crate::profiling::note_resource_churn!(BindGroup, "gpu::display_blit_overlay_bind_group");
    Some(PreparedOverlayBlit {
        bind_group,
        rect: fit_rect_px(source.width, source.height, surface_width, surface_height),
    })
}

fn write_source_uv(gpu: &GpuContext, uniform: &UvUniformBuffer, source: DisplayBlitSource<'_>) {
    let uv_params = flip_uv_params(source.flip_horizontally, source.flip_vertically);
    uniform.write(gpu.queue(), bytemuck::bytes_of(&uv_params));
}

fn create_surface_blit_bind_group(
    device: &wgpu::Device,
    label: &'static str,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    uniform_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: sampled_2d_filtered_uv_layout(device),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buf.as_entire_binding(),
            },
        ],
    })
}

/// Clears the swapchain to `background_color`, restricts the viewport to the fitted rect, and
/// draws the fullscreen-triangle blit.
///
/// Areas outside the viewport keep the cleared `background_color` (letterbox bars).
fn encode_display_blit_pass(
    encoder: &mut wgpu::CommandEncoder,
    surface_view: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    rect: FittedRectPx,
    background_color: Vec4,
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
) {
    let bg = wgpu::Color {
        r: background_color.x as f64,
        g: background_color.y as f64,
        b: background_color.z as f64,
        a: background_color.w as f64,
    };
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("display_blit_surface"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: surface_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(bg),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        occlusion_query_set: None,
        timestamp_writes,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_viewport(
        rect.x as f32,
        rect.y as f32,
        rect.w.max(1) as f32,
        rect.h.max(1) as f32,
        0.0,
        1.0,
    );
    pass.draw(0..3, 0..1);
}

fn encode_display_overlay_blit_pass(
    encoder: &mut wgpu::CommandEncoder,
    surface_view: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    rect: FittedRectPx,
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("display_blit_dashboard_overlay"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: surface_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        occlusion_query_set: None,
        timestamp_writes,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_viewport(
        rect.x as f32,
        rect.y as f32,
        rect.w.max(1) as f32,
        rect.h.max(1) as f32,
        0.0,
        1.0,
    );
    pass.draw(0..3, 0..1);
}
