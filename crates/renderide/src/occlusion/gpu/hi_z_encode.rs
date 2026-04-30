//! Hi-Z pyramid compute dispatch and copy-to-staging encoding.

use bytemuck::{Pod, Zeroable};

use crate::backend::HistoryTextureMipViews;
use crate::gpu::OutputDepthMode;
use crate::occlusion::{hi_z_pyramid_dimensions, mip_dimensions, mip_levels_for_extent};

use super::hi_z_gpu::{HIZ_MAX_MIPS, HiZGpuScratch, HiZGpuState};
use super::hi_z_pipelines::HiZPipelines;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LayerUniform {
    layer: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DownsampleUniform {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
}

#[derive(Clone, Copy)]
enum DepthBinding {
    D2,
    D2Array { layer: u32 },
}

/// Which history texture layer the current mip0 + downsample call should target.
///
/// Controls which cache slots [`HiZBindGroupCache`] reuses or rebuilds.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PyramidSide {
    /// Desktop (non-stereo) or stereo-left pyramid.
    DesktopOrLeft,
    /// Stereo-right layer in a stereo history pyramid.
    Right,
}

/// Device, encoder, and source/destination views for a single Hi-Z mip0 dispatch from depth.
struct HiZMip0EncodeContext<'a> {
    /// Device for bind group creation.
    device: &'a wgpu::Device,
    /// Queue for uniform writes (`layer_uniform`, `downsample_uniform`).
    queue: &'a wgpu::Queue,
    /// Active command encoder receiving the mip0 + downsample compute passes.
    encoder: &'a mut wgpu::CommandEncoder,
    /// Source depth view (sampled in the mip0 pass).
    depth_view: &'a wgpu::TextureView,
    /// Scratch buffers and viewports (extent, mip count, uniforms) plus cached bind groups.
    scratch: &'a mut HiZGpuScratch,
    /// Compiled Hi-Z pipelines (mip0 desktop/stereo + downsample).
    pipes: &'a HiZPipelines,
    /// Views for each pyramid mip level (written by mip0, read/written by downsample).
    pyramid_views: &'a [wgpu::TextureView],
    /// Binding flavour for the mip0 pass (D2 vs D2Array with layer).
    depth_bind: DepthBinding,
    /// Which pyramid (desktop/left vs right) this call targets; selects the cache slot.
    side: PyramidSide,
    /// GPU profiler for per-dispatch pass-level timestamp queries; [`None`] when disabled.
    profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
}

/// Resets slot validity, invalidates cache, ensures [`HiZGpuScratch`] matches `extent` / stereo layout.
///
/// Returns `false` when encoding must abort (zero extent, missing scratch, or GPU not ready).
fn reset_and_prepare_hi_z_scratch(
    device: &wgpu::Device,
    limits: &crate::gpu::GpuLimits,
    extent: (u32, u32),
    mode: OutputDepthMode,
    state: &mut HiZGpuState,
) -> bool {
    state.clear_encoded_slot();
    state.invalidate_if_needed(extent, mode);

    let (full_w, full_h) = extent;
    if full_w == 0 || full_h == 0 {
        return false;
    }

    let (bw, bh) = hi_z_pyramid_dimensions(full_w, full_h);
    if bw == 0 || bh == 0 {
        return false;
    }

    let stereo = matches!(mode, OutputDepthMode::StereoArray { .. });
    let mip_levels = mip_levels_for_extent(bw, bh, HIZ_MAX_MIPS);
    if state.scratch.as_ref().map(|s| (s.extent, s.mip_levels)) != Some(((bw, bh), mip_levels))
        || state.scratch.as_ref().map(|s| s.staging_r.is_some()) != Some(stereo)
    {
        state.scratch = HiZGpuScratch::new(device, limits, (bw, bh), stereo);
        state.clear_pending();
        state.set_secondary_readback_enabled(stereo);
    }
    state.set_secondary_readback_enabled(stereo);
    let Some(scratch_ref) = state.scratch.as_ref() else {
        return false;
    };

    state.can_encode_hi_z(scratch_ref)
}

/// GPU handles recorded into for one [`encode_hi_z_build`] call (device + limits + queue + encoder).
pub struct HiZBuildRecord<'a> {
    /// Device for pipeline cache and bind group creation.
    pub device: &'a wgpu::Device,
    /// Effective device caps used to validate scratch allocations and dispatches.
    pub limits: &'a crate::gpu::GpuLimits,
    /// Queue for uniform writes (`layer_uniform`, `downsample_uniform`).
    pub queue: &'a wgpu::Queue,
    /// Command encoder receiving the mip0, downsample, and staging copy commands.
    pub encoder: &'a mut wgpu::CommandEncoder,
}

/// Registry-owned Hi-Z pyramid selected for this view and ping-pong half.
pub struct HiZHistoryTarget<'a> {
    /// Backing history texture that receives mip writes and is copied to readback staging.
    pub texture: &'a wgpu::Texture,
    /// Per-layer/per-mip texture views for writing the current view's pyramid.
    pub mip_views: &'a HistoryTextureMipViews,
}

/// Per-layer mip chains selected from a registry-owned Hi-Z history texture.
struct HiZHistoryViews<'a> {
    /// Desktop or stereo-left mip chain.
    left: &'a [wgpu::TextureView],
    /// Stereo-right mip chain when the depth target is a two-layer array.
    right: Option<&'a [wgpu::TextureView]>,
}

/// Records Hi-Z build + copy-to-staging into the state's current readback slot.
///
/// Claims the staging slot at encode time so two consecutive frames can never aim the
/// same buffer even if the prior frame's `on_submitted_work_done` callback has not yet fired.
///
/// The claimed slot is stored as a transient handoff for the main-thread submit path to bake into
/// a [`wgpu::Queue::on_submitted_work_done`] closure, so the slot travels with the closure by value
/// and a late-firing callback cannot consume a newer frame's slot.
///
/// Call [`HiZGpuState::begin_frame_readback`] at the **start** of the next frame to drain
/// completed maps.
pub fn encode_hi_z_build(
    record: HiZBuildRecord<'_>,
    depth_view: &wgpu::TextureView,
    history: HiZHistoryTarget<'_>,
    extent: (u32, u32),
    mode: OutputDepthMode,
    state: &mut HiZGpuState,
    profiler: Option<&crate::profiling::GpuProfilerHandle>,
) {
    let HiZBuildRecord {
        device,
        limits,
        queue,
        encoder,
    } = record;
    if !reset_and_prepare_hi_z_scratch(device, limits, extent, mode, state) {
        return;
    }

    let ws = state.next_write_slot();
    let Some(scratch) = state.scratch.as_mut() else {
        return;
    };

    let (bw, bh) = scratch.extent;
    let pipes = HiZPipelines::get(device);
    let Some(history_views) = resolve_history_views(history.mip_views, mode, scratch.mip_levels)
    else {
        return;
    };

    scratch
        .bind_groups
        .invalidate_mip0_if_depth_changed(depth_view);
    scratch.bind_groups.invalidate_pyramid_if_target_changed(
        &history_views.left[0],
        history_views.right.map(|views| &views[0]),
    );

    match mode {
        OutputDepthMode::DesktopSingle => {
            dispatch_mip0_and_downsample(HiZMip0EncodeContext {
                device,
                queue,
                encoder,
                depth_view,
                scratch,
                pipes,
                pyramid_views: history_views.left,
                depth_bind: DepthBinding::D2,
                side: PyramidSide::DesktopOrLeft,
                profiler,
            });
            copy_current_layer_to_staging(encoder, history.texture, 0, bw, bh, scratch, ws);
        }
        OutputDepthMode::StereoArray { .. } => {
            if scratch.staging_r.is_none() {
                return;
            }
            let Some(views_right) = history_views.right else {
                return;
            };
            dispatch_mip0_and_downsample(HiZMip0EncodeContext {
                device,
                queue,
                encoder,
                depth_view,
                scratch,
                pipes,
                pyramid_views: history_views.left,
                depth_bind: DepthBinding::D2Array { layer: 0 },
                side: PyramidSide::DesktopOrLeft,
                profiler,
            });
            dispatch_mip0_and_downsample(HiZMip0EncodeContext {
                device,
                queue,
                encoder,
                depth_view,
                scratch,
                pipes,
                pyramid_views: views_right,
                depth_bind: DepthBinding::D2Array { layer: 1 },
                side: PyramidSide::Right,
                profiler,
            });
            copy_current_layer_to_staging(encoder, history.texture, 0, bw, bh, scratch, ws);
            if let Some(staging_r) = scratch.staging_r.as_ref() {
                copy_pyramid_to_staging(
                    encoder,
                    history.texture,
                    1,
                    bw,
                    bh,
                    scratch.mip_levels,
                    &staging_r[ws],
                );
            }
        }
    }

    let claimed_ws = state.claim_encoded_slot();
    debug_assert_eq!(claimed_ws, ws);
}

/// Resolves the history texture layer/mip chains required by the current depth mode.
fn resolve_history_views(
    mip_views: &HistoryTextureMipViews,
    mode: OutputDepthMode,
    required_mips: u32,
) -> Option<HiZHistoryViews<'_>> {
    let Some(left) = history_layer_mip_views(mip_views, 0, required_mips) else {
        logger::warn!("hi_z history texture missing layer 0 mip views; skipping encode");
        return None;
    };
    let right = match mode {
        OutputDepthMode::DesktopSingle => None,
        OutputDepthMode::StereoArray { .. } => {
            if let Some(views) = history_layer_mip_views(mip_views, 1, required_mips) {
                Some(views)
            } else {
                logger::warn!(
                    "hi_z stereo history texture missing layer 1 mip views; skipping encode"
                );
                return None;
            }
        }
    };
    Some(HiZHistoryViews { left, right })
}

/// Returns the mip view chain for `layer` when it covers every mip the current encode needs.
fn history_layer_mip_views(
    mip_views: &HistoryTextureMipViews,
    layer: u32,
    required_mips: u32,
) -> Option<&[wgpu::TextureView]> {
    let views = mip_views.layer_mip_views(layer)?;
    if views.len() < required_mips as usize {
        return None;
    }
    Some(&views[..required_mips as usize])
}

/// Fills Hi-Z mip0 from a depth texture (desktop 2D view or one layer of a stereo depth array).
fn dispatch_hi_z_mip0_from_depth(args: &mut HiZMip0EncodeContext<'_>) {
    match args.depth_bind {
        DepthBinding::D2 => dispatch_hi_z_mip0_desktop(args),
        DepthBinding::D2Array { layer } => dispatch_hi_z_mip0_stereo(args, layer),
    }
}

/// Mip0 dispatch for the desktop (non-stereo) 2D depth view.
fn dispatch_hi_z_mip0_desktop(args: &mut HiZMip0EncodeContext<'_>) {
    let device = args.device;
    let depth_view = args.depth_view;
    let pyramid_views = args.pyramid_views;
    let layout = &args.pipes.bgl_mip0_desktop;
    let bg = args.scratch.bind_groups.mip0_desktop_or_build(|| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hi_z_mip0_d_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&pyramid_views[0]),
                },
            ],
        })
    });
    let pass_query = args
        .profiler
        .map(|p| p.begin_pass_query("hi_z_mip0_desktop", args.encoder));
    let timestamp_writes = crate::profiling::compute_pass_timestamp_writes(pass_query.as_ref());
    {
        let mut pass = args
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hi_z_mip0_desktop"),
                timestamp_writes,
            });
        pass.set_pipeline(&args.pipes.mip0_desktop);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(
            args.scratch.extent.0.div_ceil(8),
            args.scratch.extent.1.div_ceil(8),
            1,
        );
    };
    if let (Some(p), Some(q)) = (args.profiler, pass_query) {
        p.end_query(args.encoder, q);
    }
}

/// Mip0 dispatch for one array layer of a stereo depth target.
fn dispatch_hi_z_mip0_stereo(args: &mut HiZMip0EncodeContext<'_>, layer: u32) {
    let layer_u = LayerUniform {
        layer,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    args.queue
        .write_buffer(&args.scratch.layer_uniform, 0, bytemuck::bytes_of(&layer_u));
    let device = args.device;
    let depth_view = args.depth_view;
    let pyramid_views = args.pyramid_views;
    let layout = &args.pipes.bgl_mip0_stereo;
    let layer_uniform = args.scratch.layer_uniform.clone();
    let bg = args.scratch.bind_groups.mip0_stereo_or_build(layer, || {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hi_z_mip0_s_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: layer_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&pyramid_views[0]),
                },
            ],
        })
    });
    let pass_query = args
        .profiler
        .map(|p| p.begin_pass_query("hi_z_mip0_stereo", args.encoder));
    let timestamp_writes = crate::profiling::compute_pass_timestamp_writes(pass_query.as_ref());
    {
        let mut pass = args
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hi_z_mip0_stereo"),
                timestamp_writes,
            });
        pass.set_pipeline(&args.pipes.mip0_stereo);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(
            args.scratch.extent.0.div_ceil(8),
            args.scratch.extent.1.div_ceil(8),
            1,
        );
    };
    if let (Some(p), Some(q)) = (args.profiler, pass_query) {
        p.end_query(args.encoder, q);
    }
}

/// Bundle of handles used by [`dispatch_hi_z_downsample_mips`].
struct HiZDownsampleContext<'a> {
    /// Device for on-demand bind-group creation.
    device: &'a wgpu::Device,
    /// Queue for `downsample_uniform` writes that carry per-mip extents.
    queue: &'a wgpu::Queue,
    /// Encoder receiving each mip's compute pass.
    encoder: &'a mut wgpu::CommandEncoder,
    /// Scratch providing both the `downsample_uniform` buffer and the cached bind groups.
    scratch: &'a mut HiZGpuScratch,
    /// Compiled downsample pipeline + bind-group layout.
    pipes: &'a HiZPipelines,
    /// Per-mip pyramid views for the active pyramid side.
    pyramid_views: &'a [wgpu::TextureView],
    /// Which pyramid's bind-group cache slot to read/write.
    side: PyramidSide,
    /// GPU profiler for per-dispatch timestamp queries; [`None`] when disabled.
    profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
}

/// Max-reduction chain from mip0 through the rest of the R32F pyramid.
fn dispatch_hi_z_downsample_mips(args: &mut HiZDownsampleContext<'_>) {
    let (bw, bh) = args.scratch.extent;
    for mip in 0..args.scratch.mip_levels.saturating_sub(1) {
        let (sw, sh) = mip_dimensions(bw, bh, mip).unwrap_or((1, 1));
        let (dw, dh) = mip_dimensions(bw, bh, mip + 1).unwrap_or((1, 1));
        let du = DownsampleUniform {
            src_w: sw,
            src_h: sh,
            dst_w: dw,
            dst_h: dh,
        };
        args.queue
            .write_buffer(&args.scratch.downsample_uniform, 0, bytemuck::bytes_of(&du));
        let device = args.device;
        let layout = &args.pipes.bgl_downsample;
        let downsample_uniform = args.scratch.downsample_uniform.clone();
        let pyramid_views = args.pyramid_views;
        let build = || {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hi_z_ds_bg"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&pyramid_views[mip as usize]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &pyramid_views[mip as usize + 1],
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: downsample_uniform.as_entire_binding(),
                    },
                ],
            })
        };
        let bg = match args.side {
            PyramidSide::DesktopOrLeft => args
                .scratch
                .bind_groups
                .downsample_desktop_or_build(mip, build),
            PyramidSide::Right => args
                .scratch
                .bind_groups
                .downsample_right_or_build(mip, build),
        };
        let pass_query = args
            .profiler
            .map(|p| p.begin_pass_query("hi_z_downsample", args.encoder));
        let timestamp_writes = crate::profiling::compute_pass_timestamp_writes(pass_query.as_ref());
        {
            let mut pass = args
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hi_z_downsample"),
                    timestamp_writes,
                });
            pass.set_pipeline(&args.pipes.downsample);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dw.div_ceil(8), dh.div_ceil(8), 1);
        };
        if let (Some(p), Some(q)) = (args.profiler, pass_query) {
            p.end_query(args.encoder, q);
        }
    }
}

/// Depth mip0 copy plus hierarchical downsample for one pyramid view chain (desktop or one array layer).
fn dispatch_mip0_and_downsample(mut args: HiZMip0EncodeContext<'_>) {
    dispatch_hi_z_mip0_from_depth(&mut args);
    dispatch_hi_z_downsample_mips(&mut HiZDownsampleContext {
        device: args.device,
        queue: args.queue,
        encoder: args.encoder,
        scratch: args.scratch,
        pipes: args.pipes,
        pyramid_views: args.pyramid_views,
        side: args.side,
        profiler: args.profiler,
    });
}

/// Copies the current write slot for one history layer into readback staging.
fn copy_current_layer_to_staging(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    array_layer: u32,
    base_w: u32,
    base_h: u32,
    scratch: &HiZGpuScratch,
    write_slot: usize,
) {
    copy_pyramid_to_staging(
        encoder,
        texture,
        array_layer,
        base_w,
        base_h,
        scratch.mip_levels,
        &scratch.staging_desktop[write_slot],
    );
}

/// Copies all mips for one history texture array layer into the selected readback staging buffer.
fn copy_pyramid_to_staging(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    array_layer: u32,
    base_w: u32,
    base_h: u32,
    mip_levels: u32,
    staging: &wgpu::Buffer,
) {
    let mut offset = 0u64;
    for mip in 0..mip_levels {
        let (w, h) = mip_dimensions(base_w, base_h, mip).unwrap_or((1, 1));
        let row_pitch = wgpu::util::align_to(w * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: mip,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: array_layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset,
                    bytes_per_row: Some(row_pitch),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        offset += u64::from(row_pitch) * u64::from(h);
    }
}
