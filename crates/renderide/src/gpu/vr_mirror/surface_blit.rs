//! Staging->surface blit for the VR desktop mirror.
//!
//! Per-frame effect: blits the persistent staging texture (filled by
//! [`super::resources::VrMirrorBlitResources::submit_eye_to_staging`]) into the swapchain
//! using **cover** UV mapping (fills the window, crops staging center). Optionally
//! composites an overlay (e.g. Dear ImGui) on the same encoder.

use crate::gpu::GpuContext;
use crate::present::{
    PresentClearError, SurfaceAcquireTrace, SurfaceFrameOutcome, SurfaceSubmitTrace,
    acquire_surface_outcome_traced, submit_surface_frame_traced,
};

use super::cover::cover_uv_params;
use super::pipelines::{linear_sampler, surface_bind_group_layout};
use super::resources::VrMirrorBlitResources;

impl VrMirrorBlitResources {
    /// Blits staging to the window with **cover** mapping (fills the window, crops staging as needed).
    /// No-op when [`Self::staging_valid`] is false
    /// (caller may [`crate::present::present_clear_frame`] instead).
    pub fn present_staging_to_surface(
        &mut self,
        gpu: &mut GpuContext,
    ) -> Result<(), PresentClearError> {
        self.present_staging_to_surface_overlay(gpu, |_, _, _| Ok::<(), String>(()))
    }

    /// Same as [`Self::present_staging_to_surface`], then runs `overlay` on the same encoder and swapchain view
    /// (e.g. Dear ImGui with `LoadOp::Load` over the mirror image).
    pub fn present_staging_to_surface_overlay<F, E>(
        &mut self,
        gpu: &mut GpuContext,
        overlay: F,
    ) -> Result<(), PresentClearError>
    where
        F: FnOnce(&mut wgpu::CommandEncoder, &wgpu::TextureView, &mut GpuContext) -> Result<(), E>,
        E: std::fmt::Display,
    {
        profiling::scope!("vr::mirror_blit_encode");
        if !self.staging_valid() {
            return Ok(());
        }
        if self.staging_texture().is_none() {
            return Ok(());
        }

        let frame = match acquire_surface_outcome_traced(gpu, SurfaceAcquireTrace::VrMirror)? {
            SurfaceFrameOutcome::Skip | SurfaceFrameOutcome::Reconfigured => return Ok(()),
            SurfaceFrameOutcome::Acquired(f) => f,
        };

        let surface_format = gpu.config_format();
        let (sw, sh) = gpu.surface_extent_px();
        let sw = sw.max(1);
        let sh = sh.max(1);
        let staging_extent = self.staging_extent();
        let (ew, eh) = (staging_extent.0.max(1), staging_extent.1.max(1));

        let u = cover_uv_params(ew, eh, sw, sh);
        let uniform_bytes = bytemuck::bytes_of(&u);
        // Clone the device Arc so `device` doesn't hold a borrow on `gpu`; the GPU profiler
        // wrappers below need `gpu.gpu_profiler_mut()` which is `&mut self`.
        let device_arc = gpu.device().clone();
        let device = device_arc.as_ref();
        self.ensure_surface_uniform(device);
        let Some(uniform_buf) = self.surface_uniform_buffer() else {
            logger::warn!("vr_mirror: surface uniform buffer missing after ensure_surface_uniform");
            frame.present();
            return Ok(());
        };
        gpu.queue().write_buffer(uniform_buf, 0, uniform_bytes);

        let Some(staging_tex) = self.staging_texture() else {
            frame.present();
            return Ok(());
        };
        let staging_view = staging_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let surface_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = linear_sampler(device);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vr_mirror_surface"),
            layout: surface_bind_group_layout(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&staging_view),
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
        });

        let pipeline = self.surface_pipeline_for_format(device, surface_format);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vr_mirror_surface"),
        });
        let outer_query = gpu
            .gpu_profiler_mut()
            .map(|p| p.begin_query("graph::vr_mirror.staging_to_surface", &mut encoder));
        encode_vr_mirror_cover_blit_pass(&mut encoder, &surface_view, pipeline, &bind_group);

        if let Err(e) = overlay(&mut encoder, &surface_view, gpu) {
            logger::warn!("debug HUD overlay (VR mirror): {e}");
        }
        if let Some(query) = outer_query
            && let Some(prof) = gpu.gpu_profiler_mut()
        {
            prof.end_query(&mut encoder, query);
            prof.resolve_queries(&mut encoder);
        }

        // Hand the surface texture to the driver thread along with the command buffer so the
        // real `Queue::submit` runs **before** `SurfaceTexture::present`. Calling `present()` on
        // the main thread immediately after `submit_tracked_frame_commands` (which only enqueues
        // on the driver) destroys the surface texture, after which the driver's deferred
        // `Queue::submit` rejects the command buffer with: "Texture with '<Surface Texture>'
        // label has been destroyed".
        let command_buffer = {
            profiling::scope!("CommandEncoder::finish::vr_mirror_surface");
            encoder.finish()
        };
        submit_surface_frame_traced(
            gpu,
            vec![command_buffer],
            frame,
            SurfaceSubmitTrace::VrMirror,
        );
        Ok(())
    }
}

/// Clears the swapchain to black, then draws a fullscreen triangle using the mirror bind group.
fn encode_vr_mirror_cover_blit_pass(
    encoder: &mut wgpu::CommandEncoder,
    surface_view: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("vr_mirror_surface"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: surface_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        occlusion_query_set: None,
        timestamp_writes: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}
