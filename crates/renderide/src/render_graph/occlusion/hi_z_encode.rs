//! Hi-Z pyramid compute dispatch and copy-to-staging encoding.

use bytemuck::{Pod, Zeroable};

use crate::render_graph::{
    hi_z_pyramid_dimensions, mip_dimensions, mip_levels_for_extent, OutputDepthMode,
};

use super::hi_z_gpu::{
    pending_none_array, pending_submit_default, HiZGpuScratch, HiZGpuState, HIZ_MAX_MIPS,
    HIZ_STAGING_RING,
};
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
    /// Scratch buffers and viewports (extent, mip count, uniforms).
    scratch: &'a HiZGpuScratch,
    /// Compiled Hi-Z pipelines (mip0 desktop/stereo + downsample).
    pipes: &'a HiZPipelines,
    /// Views for each pyramid mip level (written by mip0, read/written by downsample).
    pyramid_views: &'a [wgpu::TextureView],
    /// Binding flavour for the mip0 pass (D2 vs D2Array with layer).
    depth_bind: DepthBinding,
    /// GPU profiler for per-dispatch pass-level timestamp queries; [`None`] when disabled.
    profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
}

/// Resets slot validity, invalidates cache, ensures [`HiZGpuScratch`] matches `extent` / stereo layout.
///
/// Returns `false` when encoding must abort (zero extent, missing scratch, or GPU not ready).
fn reset_and_prepare_hi_z_scratch(
    device: &wgpu::Device,
    extent: (u32, u32),
    mode: OutputDepthMode,
    state: &mut HiZGpuState,
) -> bool {
    state.hi_z_encoded_slot = None;
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
        || state.scratch.as_ref().map(|s| s.pyramid_r.is_some()) != Some(stereo)
    {
        state.scratch = HiZGpuScratch::new(device, (bw, bh), stereo);
        state.desktop_pending = pending_none_array();
        state.stereo_left_stash = pending_none_array();
        state.stereo_right_stash = pending_none_array();
        state.write_idx = 0;
        state.hi_z_encoded_slot = None;
        state.pending_submit = pending_submit_default();
        if stereo {
            state.right_pending = Some(pending_none_array());
        } else {
            state.right_pending = None;
        }
    }
    let Some(scratch_ref) = state.scratch.as_ref() else {
        return false;
    };

    if stereo && state.right_pending.is_none() {
        state.right_pending = Some(pending_none_array());
    }
    if !stereo {
        state.right_pending = None;
    }

    state.can_encode_hi_z(scratch_ref)
}

/// Records Hi-Z build + copy-to-staging into [`HiZGpuState::write_idx`].
///
/// Claims the staging slot (advances [`HiZGpuState::write_idx`] and marks
/// [`HiZGpuState::pending_submit`]) at encode time so two consecutive frames can never aim the
/// same buffer even if the prior frame's `on_submitted_work_done` callback has not yet fired.
///
/// The claimed slot is stashed on [`HiZGpuState::hi_z_encoded_slot`] as a transient handoff for
/// the main-thread submit path to bake into a [`wgpu::Queue::on_submitted_work_done`] closure
/// via [`HiZGpuState::hi_z_encoded_slot`]`.take()` — so the slot travels with the closure by
/// value, and a late-firing callback cannot consume a newer frame's slot.
///
/// Call [`HiZGpuState::begin_frame_readback`] at the **start** of the next frame to drain
/// completed maps.
#[expect(
    clippy::too_many_arguments,
    reason = "Hi-Z build pass needs depth/pyramid/staging/uniform resources"
)]
pub fn encode_hi_z_build(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    depth_view: &wgpu::TextureView,
    extent: (u32, u32),
    mode: OutputDepthMode,
    state: &mut HiZGpuState,
    profiler: Option<&crate::profiling::GpuProfilerHandle>,
) {
    if !reset_and_prepare_hi_z_scratch(device, extent, mode, state) {
        return;
    }

    let Some(scratch) = state.scratch.as_mut() else {
        return;
    };

    let (bw, bh) = scratch.extent;
    let ws = state.write_idx;
    let pipes = HiZPipelines::get(device);

    match mode {
        OutputDepthMode::DesktopSingle => {
            dispatch_mip0_and_downsample(HiZMip0EncodeContext {
                device,
                queue,
                encoder,
                depth_view,
                scratch,
                pipes,
                pyramid_views: &scratch.views,
                depth_bind: DepthBinding::D2,
                profiler,
            });
            copy_pyramid_to_staging(
                encoder,
                &scratch.pyramid,
                bw,
                bh,
                scratch.mip_levels,
                &scratch.staging_desktop[ws],
            );
        }
        OutputDepthMode::StereoArray { .. } => {
            let Some((ref pyr_r, ref views_r)) = scratch.pyramid_r else {
                return;
            };
            let Some(staging_r) = scratch.staging_r.as_ref() else {
                return;
            };
            dispatch_mip0_and_downsample(HiZMip0EncodeContext {
                device,
                queue,
                encoder,
                depth_view,
                scratch,
                pipes,
                pyramid_views: &scratch.views,
                depth_bind: DepthBinding::D2Array { layer: 0 },
                profiler,
            });
            dispatch_mip0_and_downsample(HiZMip0EncodeContext {
                device,
                queue,
                encoder,
                depth_view,
                scratch,
                pipes,
                pyramid_views: views_r,
                depth_bind: DepthBinding::D2Array { layer: 1 },
                profiler,
            });
            copy_pyramid_to_staging(
                encoder,
                &scratch.pyramid,
                bw,
                bh,
                scratch.mip_levels,
                &scratch.staging_desktop[ws],
            );
            copy_pyramid_to_staging(encoder, pyr_r, bw, bh, scratch.mip_levels, &staging_r[ws]);
        }
    }

    state.pending_submit[ws] = true;
    state.write_idx = (ws + 1) % HIZ_STAGING_RING;
    state.hi_z_encoded_slot = Some(ws);
}

/// Fills Hi-Z mip0 from a depth texture (desktop 2D view or one layer of a stereo depth array).
fn dispatch_hi_z_mip0_from_depth(args: &mut HiZMip0EncodeContext<'_>) {
    match args.depth_bind {
        DepthBinding::D2 => {
            let bg = args.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hi_z_mip0_d_bg"),
                layout: &args.pipes.bgl_mip0_desktop,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(args.depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&args.pyramid_views[0]),
                    },
                ],
            });
            let pass_query = args
                .profiler
                .map(|p| p.begin_pass_query("hi_z_mip0_desktop", args.encoder));
            let timestamp_writes =
                crate::profiling::compute_pass_timestamp_writes(pass_query.as_ref());
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
            }
            if let (Some(p), Some(q)) = (args.profiler, pass_query) {
                p.end_query(args.encoder, q);
            }
        }
        DepthBinding::D2Array { layer } => {
            let layer_u = LayerUniform {
                layer,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            args.queue
                .write_buffer(&args.scratch.layer_uniform, 0, bytemuck::bytes_of(&layer_u));
            let bg = args.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hi_z_mip0_s_bg"),
                layout: &args.pipes.bgl_mip0_stereo,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(args.depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: args.scratch.layer_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&args.pyramid_views[0]),
                    },
                ],
            });
            let pass_query = args
                .profiler
                .map(|p| p.begin_pass_query("hi_z_mip0_stereo", args.encoder));
            let timestamp_writes =
                crate::profiling::compute_pass_timestamp_writes(pass_query.as_ref());
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
            }
            if let (Some(p), Some(q)) = (args.profiler, pass_query) {
                p.end_query(args.encoder, q);
            }
        }
    }
}

/// Max-reduction chain from mip0 through the rest of the R32F pyramid.
fn dispatch_hi_z_downsample_mips(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    scratch: &HiZGpuScratch,
    pipes: &HiZPipelines,
    pyramid_views: &[wgpu::TextureView],
    profiler: Option<&crate::profiling::GpuProfilerHandle>,
) {
    let (bw, bh) = scratch.extent;
    for mip in 0..scratch.mip_levels.saturating_sub(1) {
        let (sw, sh) = mip_dimensions(bw, bh, mip).unwrap_or((1, 1));
        let (dw, dh) = mip_dimensions(bw, bh, mip + 1).unwrap_or((1, 1));
        let du = DownsampleUniform {
            src_w: sw,
            src_h: sh,
            dst_w: dw,
            dst_h: dh,
        };
        queue.write_buffer(&scratch.downsample_uniform, 0, bytemuck::bytes_of(&du));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hi_z_ds_bg"),
            layout: &pipes.bgl_downsample,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&pyramid_views[mip as usize]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&pyramid_views[mip as usize + 1]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scratch.downsample_uniform.as_entire_binding(),
                },
            ],
        });
        let pass_query = profiler.map(|p| p.begin_pass_query("hi_z_downsample", encoder));
        let timestamp_writes = crate::profiling::compute_pass_timestamp_writes(pass_query.as_ref());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hi_z_downsample"),
                timestamp_writes,
            });
            pass.set_pipeline(&pipes.downsample);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dw.div_ceil(8), dh.div_ceil(8), 1);
        }
        if let (Some(p), Some(q)) = (profiler, pass_query) {
            p.end_query(encoder, q);
        }
    }
}

/// Depth mip0 copy plus hierarchical downsample for one pyramid view chain (desktop or one array layer).
fn dispatch_mip0_and_downsample(mut args: HiZMip0EncodeContext<'_>) {
    dispatch_hi_z_mip0_from_depth(&mut args);
    dispatch_hi_z_downsample_mips(
        args.device,
        args.queue,
        args.encoder,
        args.scratch,
        args.pipes,
        args.pyramid_views,
        args.profiler,
    );
}

fn copy_pyramid_to_staging(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    base_w: u32,
    base_h: u32,
    mip_levels: u32,
    staging: &wgpu::Buffer,
) {
    let mut offset = 0u64;
    for mip in 0..mip_levels {
        let (w, h) = mip_dimensions(base_w, base_h, mip).unwrap_or((1, 1));
        let row_pitch = wgpu::util::align_to(w * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) as u32;
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: mip,
                origin: wgpu::Origin3d::ZERO,
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
