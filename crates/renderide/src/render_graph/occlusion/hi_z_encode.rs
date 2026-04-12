//! Hi-Z pyramid compute dispatch and copy-to-staging encoding.

use bytemuck::{Pod, Zeroable};

use crate::render_graph::{
    hi_z_pyramid_dimensions, mip_dimensions, mip_levels_for_extent, OutputDepthMode,
};

use super::hi_z_gpu::{pending_none_array, HiZGpuScratch, HiZGpuState, HIZ_MAX_MIPS};
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

/// Records Hi-Z build + copy-to-staging into [`HiZGpuState::write_idx`].
///
/// Call [`HiZGpuState::on_frame_submitted`] after [`wgpu::Queue::submit`]. Call
/// [`HiZGpuState::begin_frame_readback`] at the **start** of the next frame to drain completed maps.
pub fn encode_hi_z_build(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    depth_view: &wgpu::TextureView,
    extent: (u32, u32),
    mode: OutputDepthMode,
    state: &mut HiZGpuState,
) {
    state.hi_z_encoded_slot = None;
    state.invalidate_if_needed(extent, mode);

    let (full_w, full_h) = extent;
    if full_w == 0 || full_h == 0 {
        return;
    }

    let (bw, bh) = hi_z_pyramid_dimensions(full_w, full_h);
    if bw == 0 || bh == 0 {
        return;
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
        if stereo {
            state.right_pending = Some(pending_none_array());
        } else {
            state.right_pending = None;
        }
    }
    let Some(scratch_ref) = state.scratch.as_ref() else {
        return;
    };

    if stereo && state.right_pending.is_none() {
        state.right_pending = Some(pending_none_array());
    }
    if !stereo {
        state.right_pending = None;
    }

    if !state.can_encode_hi_z(scratch_ref) {
        return;
    }

    let Some(scratch) = state.scratch.as_mut() else {
        return;
    };

    let ws = state.write_idx;
    let pipes = HiZPipelines::get(device);

    let dispatch_mip0_and_downsample =
        |encoder: &mut wgpu::CommandEncoder,
         pyramid_views: &[wgpu::TextureView],
         depth_bind: DepthBinding| {
            match depth_bind {
                DepthBinding::D2 => {
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("hi_z_mip0_d_bg"),
                        layout: &pipes.bgl_mip0_desktop,
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
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("hi_z_mip0_desktop"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&pipes.mip0_desktop);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.dispatch_workgroups(
                            scratch.extent.0.div_ceil(8),
                            scratch.extent.1.div_ceil(8),
                            1,
                        );
                    }
                }
                DepthBinding::D2Array { layer } => {
                    let layer_u = LayerUniform {
                        layer,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    };
                    queue.write_buffer(&scratch.layer_uniform, 0, bytemuck::bytes_of(&layer_u));
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("hi_z_mip0_s_bg"),
                        layout: &pipes.bgl_mip0_stereo,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(depth_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: scratch.layer_uniform.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&pyramid_views[0]),
                            },
                        ],
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("hi_z_mip0_stereo"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&pipes.mip0_stereo);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.dispatch_workgroups(
                            scratch.extent.0.div_ceil(8),
                            scratch.extent.1.div_ceil(8),
                            1,
                        );
                    }
                }
            }

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
                            resource: wgpu::BindingResource::TextureView(
                                &pyramid_views[mip as usize],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &pyramid_views[mip as usize + 1],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: scratch.downsample_uniform.as_entire_binding(),
                        },
                    ],
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("hi_z_downsample"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipes.downsample);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(dw.div_ceil(8), dh.div_ceil(8), 1);
                }
            }
        };

    match mode {
        OutputDepthMode::DesktopSingle => {
            dispatch_mip0_and_downsample(encoder, &scratch.views, DepthBinding::D2);
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
            dispatch_mip0_and_downsample(
                encoder,
                &scratch.views,
                DepthBinding::D2Array { layer: 0 },
            );
            dispatch_mip0_and_downsample(encoder, views_r, DepthBinding::D2Array { layer: 1 });
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

    state.hi_z_encoded_slot = Some(ws);
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
