//! HDR-aware MSAA color resolve for the world-mesh forward path.
//!
//! Replaces wgpu's automatic linear `resolve_target` average. A linear average of HDR samples is
//! perceptually wrong at stark contrast edges: a pixel partially covered by a very bright sample
//! (specular spark, emissive, sun) and partially by a dark sample averages to a value that, after
//! tonemapping, looks too bright (or too dark in the inverse case), producing visibly aliased
//! silhouettes between bright and dark surfaces even with high MSAA.
//!
//! This pass implements a Karis bracket: each sample is compressed by `x / (1 + max3(x))`, the
//! compressed values are linearly averaged, and
//! the result is decompressed by `y / (1 - max3(y))`. The compress / average / uncompress
//! sandwich approximates "tonemap each sample, average, untonemap" while keeping an HDR result
//! for downstream bloom and tonemap to consume.
//!
//! The transparent sequence pass uses the cached fullscreen resolve internally before each
//! grab-pass snapshot and for the final scene-color handoff, including opaque-only MSAA frames.
//! The main graph no longer brackets grab materials with one global pre-grab snapshot. When MSAA is
//! off, forward passes write directly to the single-sample `scene_color_hdr` and the resolve draw is
//! skipped.

mod pipeline;

use std::sync::LazyLock;

use pipeline::{MsaaResolveHdrPipelineCache, ResolveParamsUbo};

use crate::frame_upload_batch::GraphUploadSink;
use crate::render_graph::context::{GraphResolvedResources, PassFrameContext};
use crate::render_graph::error::RenderPassError;
use crate::render_graph::gpu_cache::stereo_mask_or_template;
use crate::render_graph::resources::TextureHandle;

/// Graph handles for the HDR-aware color resolve used by the transparent sequence.
#[derive(Clone, Copy, Debug)]
pub struct WorldMeshForwardColorResolveGraphResources {
    /// Multisampled HDR scene-color source produced by the forward opaque + intersect passes.
    pub scene_color_hdr_msaa: TextureHandle,
    /// Single-sample HDR destination consumed by post-processing and scene compose.
    pub scene_color_hdr: TextureHandle,
}

/// Inputs required to encode one HDR-aware color resolve draw.
pub(super) struct WorldMeshForwardColorResolveEncodeContext<'a, 'encoder, 'frame, 'pass> {
    /// WGPU device used for pipeline and bind-group cache lookup.
    pub(super) device: &'a wgpu::Device,
    /// Resolved graph resources for this recording scope.
    pub(super) graph_resources: &'a GraphResolvedResources,
    /// Command encoder receiving the resolve pass.
    pub(super) encoder: &'encoder mut wgpu::CommandEncoder,
    /// Per-view frame data.
    pub(super) frame: &'frame PassFrameContext<'a, 'pass>,
    /// Deferred graph upload sink for resolve uniforms.
    pub(super) uploads: GraphUploadSink<'frame>,
    /// Graph handles for resolve source and destination.
    pub(super) resources: WorldMeshForwardColorResolveGraphResources,
    /// Optional GPU profiler for pass timestamp queries.
    pub(super) profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
    /// Debug/profiler label for this resolve pass.
    pub(super) label: &'static str,
}

fn pipeline_cache() -> &'static MsaaResolveHdrPipelineCache {
    static CACHE: LazyLock<MsaaResolveHdrPipelineCache> =
        LazyLock::new(MsaaResolveHdrPipelineCache::default);
    &CACHE
}

/// Encodes an HDR-aware MSAA color resolve into `scene_color_hdr` using a caller-owned encoder.
pub(in crate::passes::world_mesh_forward) fn encode_world_mesh_forward_msaa_color_resolve(
    ctx: WorldMeshForwardColorResolveEncodeContext<'_, '_, '_, '_>,
) -> Result<bool, RenderPassError> {
    let WorldMeshForwardColorResolveEncodeContext {
        device,
        graph_resources,
        encoder,
        frame,
        uploads,
        resources,
        profiler,
        label,
    } = ctx;

    profiling::scope!("world_mesh_forward::manual_color_resolve_record");
    let sample_count = frame.view.sample_count;
    if sample_count <= 1 {
        return Ok(false);
    }

    let Some(src) = graph_resources.transient_texture(resources.scene_color_hdr_msaa) else {
        return Err(RenderPassError::FrameParamsRequired {
            pass: format!(
                "{label} (missing transient scene_color_hdr_msaa {:?})",
                resources.scene_color_hdr_msaa
            ),
        });
    };
    let Some(dst) = graph_resources.transient_texture(resources.scene_color_hdr) else {
        return Err(RenderPassError::FrameParamsRequired {
            pass: format!(
                "{label} (missing transient scene_color_hdr {:?})",
                resources.scene_color_hdr
            ),
        });
    };

    let multiview_stereo = frame.view.multiview_stereo;
    let pipelines = pipeline_cache();
    let pipeline = pipelines.pipeline(device, dst.texture.format(), multiview_stereo);
    let params = ResolveParamsUbo {
        sample_count,
        _pad: [0; 3],
    };
    let params_ubo = pipelines.params_ubo(device);
    uploads.write_buffer(params_ubo, 0, bytemuck::bytes_of(&params));
    let bind_group = pipelines.bind_group(device, &src.texture, params_ubo, multiview_stereo);

    let color_attachments = [Some(wgpu::RenderPassColorAttachment {
        view: &dst.view,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        },
        depth_slice: None,
    })];
    let pass_query = profiler.map(|p| p.begin_pass_query(label, encoder));
    let timestamp_writes = crate::profiling::render_pass_timestamp_writes(pass_query.as_ref());
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &color_attachments,
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes,
            multiview_mask: stereo_mask_or_template(multiview_stereo, None),
        });
        rpass.set_pipeline(pipeline.as_ref());
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }
    if let (Some(p), Some(q)) = (profiler, pass_query) {
        p.end_query(encoder, q);
    }
    Ok(true)
}
