//! Final HMD copy for the VR desktop mirror.
//!
//! Per-frame effect: copies renderer-owned stereo color into the acquired OpenXR swapchain and
//! copies the owned left-eye view into the persistent desktop mirror staging texture.

use crate::gpu::GpuContext;
use crate::gpu::blit_kit::layout::{sampled_2d_array_filtered_layout, sampled_2d_filtered_layout};
use crate::gpu::blit_kit::sampler::linear_clamp_sampler;

use super::pipelines::{eye_pipeline, stereo_to_openxr_pipeline};
use super::resources::VrMirrorBlitResources;

impl VrMirrorBlitResources {
    /// Encodes the final HMD copy batch: owned stereo color into OpenXR, and owned left eye into
    /// desktop mirror staging.
    pub fn encode_owned_hmd_to_openxr_and_staging(
        &mut self,
        gpu: &mut GpuContext,
        eye_extent: (u32, u32),
        source_stereo_view: &wgpu::TextureView,
        source_mirror_eye_view: &wgpu::TextureView,
        openxr_target_view: &wgpu::TextureView,
    ) -> wgpu::CommandBuffer {
        let device_arc = gpu.device().clone();
        let device = device_arc.as_ref();
        let limits = gpu.limits().clone();
        self.ensure_staging(device, &limits, eye_extent);

        let sampler = linear_clamp_sampler(device);
        let stereo_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vr_mirror_stereo_to_openxr"),
            layout: sampled_2d_array_filtered_layout(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_stereo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        crate::profiling::note_resource_churn!(
            BindGroup,
            "gpu::vr_mirror_stereo_to_openxr_bind_group"
        );

        let staging_view_and_bind_group = self.staging_texture().map(|staging_tex| {
            let staging_view = staging_tex.create_view(&wgpu::TextureViewDescriptor::default());
            crate::profiling::note_resource_churn!(TextureView, "gpu::vr_mirror_eye_staging_view");
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vr_mirror_eye_to_staging"),
                layout: sampled_2d_filtered_layout(device),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(source_mirror_eye_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            crate::profiling::note_resource_churn!(BindGroup, "gpu::vr_mirror_eye_bind_group");
            (staging_view, bind_group)
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vr_mirror_hmd_final_copy"),
        });
        let outer_query = gpu
            .gpu_profiler_mut()
            .map(|p| p.begin_query("graph::vr_mirror.hmd_final_copy", &mut encoder));

        encode_hmd_stereo_to_openxr_pass(
            &mut encoder,
            openxr_target_view,
            stereo_to_openxr_pipeline(device),
            &stereo_bind_group,
        );

        if let Some((staging_view, mirror_bind_group)) = staging_view_and_bind_group.as_ref() {
            encode_mirror_eye_to_staging_pass(
                &mut encoder,
                staging_view,
                eye_pipeline(device),
                mirror_bind_group,
            );
            self.mark_staging_valid();
        } else {
            logger::debug!(
                "vr mirror eye staging texture unavailable; submitting OpenXR copy only"
            );
        }

        if let Some(query) = outer_query
            && let Some(prof) = gpu.gpu_profiler_mut()
        {
            prof.end_query(&mut encoder, query);
            prof.resolve_queries(&mut encoder);
        }

        profiling::scope!("CommandEncoder::finish::vr_mirror_hmd_final_copy");
        encoder.finish()
    }
}

/// Encodes the renderer-owned stereo-color to OpenXR swapchain copy pass.
fn encode_hmd_stereo_to_openxr_pass(
    encoder: &mut wgpu::CommandEncoder,
    openxr_target_view: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("vr_mirror_stereo_to_openxr"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: openxr_target_view,
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
        multiview_mask: std::num::NonZeroU32::new(0b11),
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

/// Encodes the renderer-owned left-eye to desktop mirror staging pass.
fn encode_mirror_eye_to_staging_pass(
    encoder: &mut wgpu::CommandEncoder,
    staging_view: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("vr_mirror_eye_to_staging"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: staging_view,
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

#[cfg(test)]
mod tests {
    #[test]
    fn stereo_openxr_copy_uses_two_view_mask() {
        assert_eq!(std::num::NonZeroU32::new(0b11).map(|v| v.get()), Some(3));
    }
}
