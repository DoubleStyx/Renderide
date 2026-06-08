//! Offscreen color-copy recording for per-view graph execution.

use std::time::Instant;

use super::super::super::ResolvedOffscreenColorCopy;
use super::super::super::elapsed_ms;

/// Records the final scratch-to-render-texture copy for a partial offscreen viewport.
pub(super) fn record_offscreen_color_copy(
    encoder: &mut wgpu::CommandEncoder,
    copy: Option<&ResolvedOffscreenColorCopy>,
    profiler: Option<&crate::profiling::GpuProfilerHandle>,
) -> bool {
    let Some(copy) = copy else {
        return false;
    };
    if copy.extent_px.0 == 0 || copy.extent_px.1 == 0 {
        return false;
    }
    profiling::scope!("graph::per_view::offscreen_color_copy");
    let copy_query =
        profiler.map(|p| p.begin_query("graph::per_view::offscreen_color_copy", encoder));
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &copy.source_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: &copy.destination_texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: copy.destination_origin_px.0,
                y: copy.destination_origin_px.1,
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width: copy.extent_px.0,
            height: copy.extent_px.1,
            depth_or_array_layers: 1,
        },
    );
    if let Some(query) = copy_query
        && let Some(profiler) = profiler
    {
        profiler.end_query(encoder, query);
    }
    true
}

/// Records an offscreen copy into a dedicated command buffer.
pub(super) fn record_offscreen_color_copy_command(
    device: &wgpu::Device,
    copy: Option<&ResolvedOffscreenColorCopy>,
    profiler: Option<&crate::profiling::GpuProfilerHandle>,
) -> Option<(wgpu::CommandBuffer, bool, f64, f64)> {
    let copy = copy?;
    if copy.extent_px.0 == 0 || copy.extent_px.1 == 0 {
        return None;
    }
    let encode_start = Instant::now();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("render-graph-per-view-offscreen-copy"),
    });
    let recorded = record_offscreen_color_copy(&mut encoder, Some(copy), profiler);
    let encode_ms = elapsed_ms(encode_start);
    let finish_start = Instant::now();
    let command_buffer = encoder.finish();
    let finish_ms = elapsed_ms(finish_start);
    Some((command_buffer, recorded, encode_ms, finish_ms))
}
