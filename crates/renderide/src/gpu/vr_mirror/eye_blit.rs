//! Eye->staging blit for the VR desktop mirror.
//!
//! Per-frame effect: copies one HMD eye into the persistent staging texture owned by
//! [`super::resources::VrMirrorBlitResources`] and submits the resulting command buffer
//! through the renderer's frame-tracked path.

use crate::gpu::GpuContext;

use super::pipelines::{eye_bind_group_layout, eye_pipeline, linear_sampler};
use super::resources::VrMirrorBlitResources;

impl VrMirrorBlitResources {
    /// Copies the acquired swapchain eye layer into the staging texture and submits GPU work.
    ///
    /// Call after the multiview render graph submit, before [`openxr::Swapchain::release_image`].
    pub fn submit_eye_to_staging(
        &mut self,
        gpu: &mut GpuContext,
        eye_extent: (u32, u32),
        source_layer_view: &wgpu::TextureView,
    ) {
        // Clone the device Arc so `device` doesn't hold a borrow on `gpu`; the GPU profiler
        // wrappers below need `gpu.gpu_profiler_mut()` which is `&mut self`.
        let device_arc = gpu.device().clone();
        let device = device_arc.as_ref();
        let limits = gpu.limits().clone();
        self.ensure_staging(device, &limits, eye_extent);
        self.ensure_surface_uniform(device);

        let Some(staging_tex) = self.staging_texture() else {
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
        let outer_query = gpu
            .gpu_profiler_mut()
            .map(|p| p.begin_query("graph::vr_mirror.eye_to_staging", &mut encoder));
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
        };
        if let Some(query) = outer_query
            && let Some(prof) = gpu.gpu_profiler_mut()
        {
            prof.end_query(&mut encoder, query);
            prof.resolve_queries(&mut encoder);
        }

        gpu.submit_tracked_frame_commands(encoder.finish());
        self.mark_staging_valid();
    }
}
