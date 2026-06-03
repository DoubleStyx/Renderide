//! Final HMD copy for the VR desktop mirror.
//!
//! Per-frame effect: copies renderer-owned stereo color into the acquired OpenXR swapchain and
//! copies the owned left-eye view into the persistent desktop mirror staging texture.

use std::num::NonZeroU32;

use crate::gpu::GpuContext;
use crate::gpu::blit_kit::layout::{sampled_2d_array_filtered_layout, sampled_2d_filtered_layout};
use crate::gpu::blit_kit::sampler::linear_clamp_sampler;

use super::pipelines::{eye_pipeline, openxr_multiview_pipeline};
use super::resources::VrMirrorBlitResources;

/// Debug label for the final multiview OpenXR color-copy render pass.
const OPENXR_COPY_PASS_LABEL: &str = "vr_mirror_hmd_to_openxr_multiview";
/// GPU-profiler pass label for the final multiview OpenXR color-copy render pass.
const OPENXR_COPY_QUERY_LABEL: &str = "graph::vr_mirror.hmd_to_openxr_multiview";

impl VrMirrorBlitResources {
    /// Encodes the final HMD copy batch: owned stereo color into OpenXR, and owned left eye into
    /// desktop mirror staging.
    pub fn encode_owned_hmd_to_openxr_and_staging(
        &mut self,
        gpu: &mut GpuContext,
        eye_extent: (u32, u32),
        source_color_array_view: &wgpu::TextureView,
        openxr_target_array_view: &wgpu::TextureView,
        source_mirror_eye_view: &wgpu::TextureView,
    ) -> wgpu::CommandBuffer {
        let device_arc = gpu.device().clone();
        let device = device_arc.as_ref();
        let limits = gpu.limits().clone();
        self.ensure_staging(device, &limits, eye_extent);

        let sampler = linear_clamp_sampler(device);
        let openxr_bind_group =
            create_openxr_multiview_bind_group(device, sampler, source_color_array_view);
        let staging_view_and_bind_group = self.staging_texture().map(|staging_tex| {
            create_mirror_staging_bind_group(staging_tex, device, sampler, source_mirror_eye_view)
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vr_mirror_hmd_final_copy"),
        });
        let outer_query = gpu
            .gpu_profiler_mut()
            .map(|p| p.begin_query("graph::vr_mirror.hmd_final_copy", &mut encoder));

        encode_openxr_multiview_copy(
            gpu,
            device,
            &mut encoder,
            openxr_target_array_view,
            &openxr_bind_group,
        );

        if encode_mirror_eye_to_staging_copy(
            gpu,
            device,
            &mut encoder,
            staging_view_and_bind_group.as_ref(),
        ) {
            self.mark_staging_valid();
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

/// Creates the bind group for the final multiview OpenXR swapchain copy.
fn create_openxr_multiview_bind_group(
    device: &wgpu::Device,
    sampler: &wgpu::Sampler,
    source_color_array_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vr_mirror_hmd_to_openxr_multiview"),
        layout: sampled_2d_array_filtered_layout(device),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source_color_array_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    crate::profiling::note_resource_churn!(BindGroup, "gpu::vr_mirror_hmd_to_openxr_multiview");
    bind_group
}

/// Creates the desktop mirror staging view and bind group.
fn create_mirror_staging_bind_group(
    staging_tex: &wgpu::Texture,
    device: &wgpu::Device,
    sampler: &wgpu::Sampler,
    source_eye_view: &wgpu::TextureView,
) -> (wgpu::TextureView, wgpu::BindGroup) {
    let staging_view = staging_tex.create_view(&wgpu::TextureViewDescriptor::default());
    crate::profiling::note_resource_churn!(TextureView, "gpu::vr_mirror_eye_staging_view");
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vr_mirror_eye_to_staging"),
        layout: sampled_2d_filtered_layout(device),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source_eye_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    crate::profiling::note_resource_churn!(BindGroup, "gpu::vr_mirror_eye_bind_group");
    (staging_view, bind_group)
}

/// Encodes the final OpenXR color copy as one multiview pass.
fn encode_openxr_multiview_copy(
    gpu: &mut GpuContext,
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    openxr_target_array_view: &wgpu::TextureView,
    bind_group: &wgpu::BindGroup,
) {
    let query = gpu
        .gpu_profiler_mut()
        .map(|p| p.begin_pass_query(OPENXR_COPY_QUERY_LABEL, encoder));
    let timestamp_writes = crate::profiling::render_pass_timestamp_writes(query.as_ref());
    encode_openxr_multiview_copy_pass(
        encoder,
        openxr_target_array_view,
        openxr_multiview_pipeline(device),
        bind_group,
        timestamp_writes,
    );
    if let Some(query) = query
        && let Some(prof) = gpu.gpu_profiler_mut()
    {
        prof.end_query(encoder, query);
    }
}

/// Encodes the desktop mirror staging copy if staging is available.
fn encode_mirror_eye_to_staging_copy(
    gpu: &mut GpuContext,
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    staging_view_and_bind_group: Option<&(wgpu::TextureView, wgpu::BindGroup)>,
) -> bool {
    let Some((staging_view, mirror_bind_group)) = staging_view_and_bind_group else {
        logger::debug!("vr mirror eye staging texture unavailable; submitting OpenXR copy only");
        return false;
    };

    let staging_query = gpu
        .gpu_profiler_mut()
        .map(|p| p.begin_pass_query("graph::vr_mirror.eye_to_staging", encoder));
    let staging_timestamp_writes =
        crate::profiling::render_pass_timestamp_writes(staging_query.as_ref());
    encode_mirror_eye_to_staging_pass(
        encoder,
        staging_view,
        eye_pipeline(device),
        mirror_bind_group,
        staging_timestamp_writes,
    );
    if let Some(query) = staging_query
        && let Some(prof) = gpu.gpu_profiler_mut()
    {
        prof.end_query(encoder, query);
    }
    true
}

/// Encodes renderer-owned stereo color to the OpenXR swapchain color array.
fn encode_openxr_multiview_copy_pass(
    encoder: &mut wgpu::CommandEncoder,
    openxr_array_view: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(OPENXR_COPY_PASS_LABEL),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: openxr_array_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        occlusion_query_set: None,
        timestamp_writes,
        multiview_mask: openxr_copy_multiview_mask(),
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
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
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
        timestamp_writes,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

/// Stereo mask used by the final OpenXR color-copy pass.
fn openxr_copy_multiview_mask() -> Option<NonZeroU32> {
    NonZeroU32::new(3)
}

#[cfg(test)]
mod tests {
    use super::openxr_copy_multiview_mask;

    #[test]
    fn openxr_copy_uses_stereo_multiview_mask() {
        assert_eq!(openxr_copy_multiview_mask().map(|mask| mask.get()), Some(3));
    }
}
