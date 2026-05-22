//! Final HMD copy for the VR desktop mirror.
//!
//! Per-frame effect: copies renderer-owned stereo color into the acquired OpenXR swapchain and
//! copies the owned left-eye view into the persistent desktop mirror staging texture.

use crate::gpu::GpuContext;
use crate::gpu::blit_kit::layout::sampled_2d_filtered_layout;
use crate::gpu::blit_kit::sampler::linear_clamp_sampler;

use super::pipelines::{eye_pipeline, openxr_eye_pipeline};
use super::resources::VrMirrorBlitResources;

/// Number of eye layers copied into the OpenXR stereo swapchain.
const OPENXR_EYE_COPY_COUNT: usize = 2;
/// Debug labels for the two final OpenXR eye-copy render passes.
const OPENXR_EYE_COPY_PASS_LABELS: [&str; OPENXR_EYE_COPY_COUNT] = [
    "vr_mirror_left_eye_to_openxr",
    "vr_mirror_right_eye_to_openxr",
];
/// GPU-profiler pass labels for the two final OpenXR eye-copy render passes.
const OPENXR_EYE_COPY_QUERY_LABELS: [&str; OPENXR_EYE_COPY_COUNT] = [
    "graph::vr_mirror.left_eye_to_openxr",
    "graph::vr_mirror.right_eye_to_openxr",
];

impl VrMirrorBlitResources {
    /// Encodes the final HMD copy batch: owned stereo color into OpenXR, and owned left eye into
    /// desktop mirror staging.
    pub fn encode_owned_hmd_to_openxr_and_staging(
        &mut self,
        gpu: &mut GpuContext,
        eye_extent: (u32, u32),
        source_eye_views: [&wgpu::TextureView; OPENXR_EYE_COPY_COUNT],
        openxr_target_eye_views: [&wgpu::TextureView; OPENXR_EYE_COPY_COUNT],
    ) -> wgpu::CommandBuffer {
        let device_arc = gpu.device().clone();
        let device = device_arc.as_ref();
        let limits = gpu.limits().clone();
        self.ensure_staging(device, &limits, eye_extent);

        let sampler = linear_clamp_sampler(device);
        let eye_bind_groups = create_openxr_eye_bind_groups(device, sampler, source_eye_views);
        let staging_view_and_bind_group = self.staging_texture().map(|staging_tex| {
            create_mirror_staging_bind_group(
                staging_tex,
                device,
                sampler,
                source_eye_views[crate::gpu::VR_MIRROR_EYE_LAYER as usize],
            )
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vr_mirror_hmd_final_copy"),
        });
        let outer_query = gpu
            .gpu_profiler_mut()
            .map(|p| p.begin_query("graph::vr_mirror.hmd_final_copy", &mut encoder));

        encode_openxr_eye_copies(
            gpu,
            device,
            &mut encoder,
            openxr_target_eye_views,
            &eye_bind_groups,
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

/// Creates per-eye bind groups for final OpenXR swapchain copies.
fn create_openxr_eye_bind_groups(
    device: &wgpu::Device,
    sampler: &wgpu::Sampler,
    source_eye_views: [&wgpu::TextureView; OPENXR_EYE_COPY_COUNT],
) -> [wgpu::BindGroup; OPENXR_EYE_COPY_COUNT] {
    source_eye_views.map(|source_view| {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vr_mirror_eye_to_openxr"),
            layout: sampled_2d_filtered_layout(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        crate::profiling::note_resource_churn!(BindGroup, "gpu::vr_mirror_eye_to_openxr");
        bind_group
    })
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

/// Encodes both final OpenXR eye-layer copy passes.
fn encode_openxr_eye_copies(
    gpu: &mut GpuContext,
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    openxr_target_eye_views: [&wgpu::TextureView; OPENXR_EYE_COPY_COUNT],
    eye_bind_groups: &[wgpu::BindGroup; OPENXR_EYE_COPY_COUNT],
) {
    for eye in 0..OPENXR_EYE_COPY_COUNT {
        let eye_query = gpu
            .gpu_profiler_mut()
            .map(|p| p.begin_pass_query(OPENXR_EYE_COPY_QUERY_LABELS[eye], encoder));
        let eye_timestamp_writes =
            crate::profiling::render_pass_timestamp_writes(eye_query.as_ref());
        encode_openxr_eye_copy_pass(
            encoder,
            OPENXR_EYE_COPY_PASS_LABELS[eye],
            openxr_target_eye_views[eye],
            openxr_eye_pipeline(device),
            &eye_bind_groups[eye],
            eye_timestamp_writes,
        );
        if let Some(query) = eye_query
            && let Some(prof) = gpu.gpu_profiler_mut()
        {
            prof.end_query(encoder, query);
        }
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

/// Encodes one renderer-owned eye color view to one OpenXR swapchain eye layer.
fn encode_openxr_eye_copy_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &'static str,
    openxr_eye_view: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: openxr_eye_view,
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
        multiview_mask: openxr_eye_copy_multiview_mask(),
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

/// Multiview is intentionally disabled for final OpenXR eye-layer copies.
fn openxr_eye_copy_multiview_mask() -> Option<std::num::NonZeroU32> {
    None
}

#[cfg(test)]
mod tests {
    use super::{OPENXR_EYE_COPY_COUNT, openxr_eye_copy_multiview_mask};

    #[test]
    fn openxr_copy_uses_two_single_view_eye_passes() {
        assert_eq!(OPENXR_EYE_COPY_COUNT, 2);
        assert!(openxr_eye_copy_multiview_mask().is_none());
    }
}
