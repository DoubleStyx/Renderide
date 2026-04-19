//! Staging texture and submit/present paths for the VR desktop mirror.

use crate::gpu::GpuContext;
use crate::present::{acquire_surface_outcome, PresentClearError, SurfaceFrameOutcome};
use crate::xr::XR_COLOR_FORMAT;

use super::cover::cover_uv_params;
use super::pipelines::{
    eye_bind_group_layout, eye_pipeline, linear_sampler, surface_bind_group_layout,
    surface_pipeline,
};

/// GPU resources for VR mirror blit (staging texture + pipelines).
pub struct VrMirrorBlitResources {
    staging_texture: Option<wgpu::Texture>,
    staging_extent: (u32, u32),
    /// `true` after a successful eye→staging copy this session.
    pub staging_valid: bool,
    surface_uniform: Option<wgpu::Buffer>,
    surface_pipeline: Option<(wgpu::TextureFormat, wgpu::RenderPipeline)>,
}

impl Default for VrMirrorBlitResources {
    fn default() -> Self {
        Self::new()
    }
}

impl VrMirrorBlitResources {
    /// Empty resources; staging is allocated on first successful HMD frame.
    pub fn new() -> Self {
        Self {
            staging_texture: None,
            staging_extent: (0, 0),
            staging_valid: false,
            surface_uniform: None,
            surface_pipeline: None,
        }
    }

    fn ensure_staging(&mut self, device: &wgpu::Device, extent: (u32, u32)) {
        if self.staging_extent == extent && self.staging_texture.is_some() {
            return;
        }
        let w = extent.0.max(1);
        let h = extent.1.max(1);
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vr_mirror_staging"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: XR_COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.staging_texture = Some(tex);
        self.staging_extent = extent;
    }

    fn ensure_surface_uniform(&mut self, device: &wgpu::Device) {
        if self.surface_uniform.is_some() {
            return;
        }
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vr_mirror_surface_uv"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.surface_uniform = Some(buf);
    }

    fn surface_pipeline_for_format(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> &wgpu::RenderPipeline {
        let entry = self
            .surface_pipeline
            .get_or_insert_with(|| (format, surface_pipeline(device, format)));
        if entry.0 != format {
            *entry = (format, surface_pipeline(device, format));
        }
        &entry.1
    }

    /// Copies the acquired swapchain eye layer into the staging texture and submits GPU work.
    ///
    /// Call after the multiview render graph submit, before [`openxr::Swapchain::release_image`].
    pub fn submit_eye_to_staging(
        &mut self,
        gpu: &GpuContext,
        eye_extent: (u32, u32),
        source_layer_view: &wgpu::TextureView,
    ) {
        let device = gpu.device().as_ref();
        self.ensure_staging(device, eye_extent);
        self.ensure_surface_uniform(device);

        let Some(staging_tex) = self.staging_texture.as_ref() else {
            return;
        };
        let staging_view = staging_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = linear_sampler(device);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vr_mirror_eye_to_staging"),
            layout: eye_bind_group_layout(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_layer_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vr_mirror_eye_to_staging"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vr_mirror_eye_to_staging"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &staging_view,
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
            pass.set_pipeline(eye_pipeline(device));
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        gpu.submit_tracked_frame_commands(encoder.finish());
        self.staging_valid = true;
    }

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
        if !self.staging_valid {
            return Ok(());
        }
        if self.staging_texture.is_none() {
            return Ok(());
        }

        let frame = match acquire_surface_outcome(gpu)? {
            SurfaceFrameOutcome::Skip | SurfaceFrameOutcome::Reconfigured => return Ok(()),
            SurfaceFrameOutcome::Acquired(f) => f,
        };

        let surface_format = gpu.config_format();
        let (sw, sh) = gpu.surface_extent_px();
        let sw = sw.max(1);
        let sh = sh.max(1);
        let (ew, eh) = (self.staging_extent.0.max(1), self.staging_extent.1.max(1));

        let u = cover_uv_params(ew, eh, sw, sh);
        let uniform_bytes = bytemuck::bytes_of(&u);
        let device = gpu.device().as_ref();
        self.ensure_surface_uniform(device);
        let Some(uniform_buf) = self.surface_uniform.as_ref() else {
            logger::warn!("vr_mirror: surface uniform buffer missing after ensure_surface_uniform");
            frame.present();
            return Ok(());
        };
        gpu.queue().write_buffer(uniform_buf, 0, uniform_bytes);

        let Some(staging_tex) = self.staging_texture.as_ref() else {
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
        encode_vr_mirror_cover_blit_pass(&mut encoder, &surface_view, pipeline, &bind_group);

        if let Err(e) = overlay(&mut encoder, &surface_view, gpu) {
            logger::warn!("debug HUD overlay (VR mirror): {e}");
        }

        gpu.submit_tracked_frame_commands(encoder.finish());
        frame.present();
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
