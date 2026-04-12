//! GPU hierarchical depth pyramid build and CPU readback for occlusion tests.

use std::sync::mpsc;

use crate::render_graph::{
    hi_z_snapshot_from_linear_linear, mip_dimensions, mip_levels_for_extent,
    unpack_linear_rows_to_mips, HiZCpuSnapshot, HiZStereoCpuSnapshot, HiZTemporalState,
    OutputDepthMode,
};

pub(crate) const HIZ_MAX_MIPS: u32 = 8;

/// Triple-buffered staging so a slot is not reused until prior `map_async` completes (non-blocking).
pub(crate) const HIZ_STAGING_RING: usize = 3;

pub(crate) type MapRecv = mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>;

pub(crate) const fn pending_none_array<T>() -> [Option<T>; HIZ_STAGING_RING] {
    [None, None, None]
}

/// GPU + CPU Hi-Z state owned by [`crate::backend::OcclusionSystem`].
pub struct HiZGpuState {
    /// Last successfully read desktop pyramid (previous frame).
    pub desktop: Option<HiZCpuSnapshot>,
    /// Last successfully read stereo pyramids (previous frame).
    pub stereo: Option<HiZStereoCpuSnapshot>,
    /// View/projection snapshot for the frame that produced [`Self::desktop`] / [`Self::stereo`].
    pub temporal: Option<HiZTemporalState>,
    pub(crate) scratch: Option<HiZGpuScratch>,
    last_extent: (u32, u32),
    last_mode: OutputDepthMode,
    /// Next ring index for Hi-Z encode copy targets (0..[`HIZ_STAGING_RING`]).
    pub(crate) write_idx: usize,
    /// Staging slot written in the current encode (consumed by [`Self::on_frame_submitted`]).
    pub(crate) hi_z_encoded_slot: Option<usize>,
    /// Pending `map_async` callbacks per desktop / left-eye staging buffer.
    pub(crate) desktop_pending: [Option<MapRecv>; HIZ_STAGING_RING],
    /// Pending `map_async` per right-eye buffer when stereo; `None` when desktop-only.
    pub(crate) right_pending: Option<[Option<MapRecv>; HIZ_STAGING_RING]>,
    /// Partial stereo CPU bytes until both eyes for the same ring slot complete.
    pub(crate) stereo_left_stash: [Option<Vec<u8>>; HIZ_STAGING_RING],
    pub(crate) stereo_right_stash: [Option<Vec<u8>>; HIZ_STAGING_RING],
}

impl Default for HiZGpuState {
    fn default() -> Self {
        Self {
            desktop: None,
            stereo: None,
            temporal: None,
            scratch: None,
            last_extent: (0, 0),
            last_mode: OutputDepthMode::DesktopSingle,
            write_idx: 0,
            hi_z_encoded_slot: None,
            desktop_pending: pending_none_array(),
            right_pending: None,
            stereo_left_stash: pending_none_array(),
            stereo_right_stash: pending_none_array(),
        }
    }
}

impl HiZGpuState {
    /// Drops GPU scratch and CPU snapshots when resolution or depth mode changes.
    pub fn invalidate_if_needed(&mut self, extent: (u32, u32), mode: OutputDepthMode) {
        if self.last_extent != extent || self.last_mode != mode {
            self.desktop = None;
            self.stereo = None;
            self.temporal = None;
            self.scratch = None;
            self.write_idx = 0;
            self.hi_z_encoded_slot = None;
            self.desktop_pending = pending_none_array();
            self.right_pending = None;
            self.stereo_left_stash = pending_none_array();
            self.stereo_right_stash = pending_none_array();
        }
        self.last_extent = extent;
        self.last_mode = mode;
    }

    /// Clears ring readback state without mapping (e.g. device loss).
    pub fn clear_pending(&mut self) {
        self.write_idx = 0;
        self.hi_z_encoded_slot = None;
        self.desktop_pending = pending_none_array();
        self.right_pending = None;
        self.stereo_left_stash = pending_none_array();
        self.stereo_right_stash = pending_none_array();
    }

    /// Drains completed `map_async` work into [`Self::desktop`] / [`Self::stereo`] without blocking.
    ///
    /// Call at the **start** of each frame (before encoding the render graph). Uses at most one
    /// [`wgpu::Device::poll`] to advance callbacks; if a read is not ready, prior snapshots are kept.
    pub fn begin_frame_readback(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::Poll);

        let Some(scratch) = self.scratch.as_ref() else {
            return;
        };
        let extent = scratch.extent;
        let mip_levels = scratch.mip_levels;
        let stereo = scratch.staging_r.is_some();

        for i in 0..HIZ_STAGING_RING {
            if let Some(recv) = self.desktop_pending[i].as_mut() {
                match recv.try_recv() {
                    Ok(Ok(())) => {
                        let buf = &scratch.staging_desktop[i];
                        let Some(raw) = read_mapped_buffer(buf) else {
                            self.desktop_pending[i] = None;
                            continue;
                        };
                        self.desktop_pending[i] = None;
                        if stereo {
                            self.stereo_left_stash[i] = Some(raw);
                        } else if let Some(snap) = unpack_desktop_snapshot(extent, mip_levels, &raw)
                        {
                            self.desktop = Some(snap);
                            self.stereo = None;
                        }
                    }
                    Ok(Err(_)) => {
                        scratch.staging_desktop[i].unmap();
                        self.desktop_pending[i] = None;
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => {
                        self.desktop_pending[i] = None;
                    }
                }
            }
        }

        if stereo {
            if let Some(ref staging_r) = scratch.staging_r {
                let right_pending = self.right_pending.get_or_insert_with(pending_none_array);
                for i in 0..HIZ_STAGING_RING {
                    if let Some(recv) = right_pending[i].as_mut() {
                        match recv.try_recv() {
                            Ok(Ok(())) => {
                                let buf = &staging_r[i];
                                let Some(raw) = read_mapped_buffer(buf) else {
                                    right_pending[i] = None;
                                    continue;
                                };
                                right_pending[i] = None;
                                self.stereo_right_stash[i] = Some(raw);
                            }
                            Ok(Err(_)) => {
                                staging_r[i].unmap();
                                right_pending[i] = None;
                            }
                            Err(mpsc::TryRecvError::Empty) => {}
                            Err(mpsc::TryRecvError::Disconnected) => {
                                right_pending[i] = None;
                            }
                        }
                    }
                }
            }
        }

        if stereo {
            for i in 0..HIZ_STAGING_RING {
                if self.stereo_left_stash[i].is_some() && self.stereo_right_stash[i].is_some() {
                    let left_raw = self.stereo_left_stash[i].take();
                    let right_raw = self.stereo_right_stash[i].take();
                    if let (Some(left_raw), Some(right_raw)) = (left_raw, right_raw) {
                        if let Some(stereo_snap) =
                            unpack_stereo_snapshot(extent, mip_levels, &left_raw, &right_raw)
                        {
                            self.stereo = Some(stereo_snap);
                            self.desktop = None;
                        }
                    }
                }
            }
        }
    }

    /// Starts `map_async` on the staging buffer(s) written this frame. Call **after**
    /// [`wgpu::Queue::submit`] for the command buffer that contains the Hi-Z copies.
    pub fn on_frame_submitted(&mut self, _device: &wgpu::Device) {
        let Some(ws) = self.hi_z_encoded_slot.take() else {
            return;
        };
        debug_assert!(ws < HIZ_STAGING_RING);
        let Some(scratch) = self.scratch.as_ref() else {
            return;
        };
        debug_assert!(self.desktop_pending[ws].is_none());

        let slice = scratch.staging_desktop[ws].slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.desktop_pending[ws] = Some(rx);

        if let Some(ref staging_r) = scratch.staging_r {
            if self.right_pending.is_none() {
                self.right_pending = Some(pending_none_array());
            }
            if let Some(rp) = self.right_pending.as_mut() {
                debug_assert!(rp[ws].is_none());
                let slice_r = staging_r[ws].slice(..);
                let (tx_r, rx_r) = mpsc::channel();
                slice_r.map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx_r.send(r);
                });
                rp[ws] = Some(rx_r);
            }
        }

        self.write_idx = (self.write_idx + 1) % HIZ_STAGING_RING;
    }

    pub(crate) fn can_encode_hi_z(&self, scratch: &HiZGpuScratch) -> bool {
        let idx = self.write_idx;
        if self.desktop_pending[idx].is_some() {
            return false;
        }
        if scratch.staging_r.is_some() {
            if let Some(ref rp) = self.right_pending {
                if rp[idx].is_some() {
                    return false;
                }
            }
        }
        true
    }
}

fn read_mapped_buffer(buf: &wgpu::Buffer) -> Option<Vec<u8>> {
    let range = buf.slice(..).get_mapped_range().to_vec();
    buf.unmap();
    Some(range)
}

fn unpack_desktop_snapshot(
    extent: (u32, u32),
    mip_levels: u32,
    raw: &[u8],
) -> Option<HiZCpuSnapshot> {
    let mips = match unpack_linear_rows_to_mips(extent.0, extent.1, mip_levels, raw) {
        Some(m) => m,
        None => {
            logger::warn!("Hi-Z desktop readback unpack failed");
            return None;
        }
    };
    match hi_z_snapshot_from_linear_linear(extent.0, extent.1, mip_levels, mips) {
        Some(s) => Some(s),
        None => {
            logger::warn!("Hi-Z desktop snapshot validation failed");
            None
        }
    }
}

fn unpack_stereo_snapshot(
    extent: (u32, u32),
    mip_levels: u32,
    left_raw: &[u8],
    right_raw: &[u8],
) -> Option<HiZStereoCpuSnapshot> {
    let mips_l = match unpack_linear_rows_to_mips(extent.0, extent.1, mip_levels, left_raw) {
        Some(m) => m,
        None => {
            logger::warn!("Hi-Z stereo left readback unpack failed");
            return None;
        }
    };
    let left = match hi_z_snapshot_from_linear_linear(extent.0, extent.1, mip_levels, mips_l) {
        Some(s) => s,
        None => {
            logger::warn!("Hi-Z stereo left snapshot validation failed");
            return None;
        }
    };
    let mips_r = match unpack_linear_rows_to_mips(extent.0, extent.1, mip_levels, right_raw) {
        Some(m) => m,
        None => {
            logger::warn!("Hi-Z stereo right readback unpack failed");
            return None;
        }
    };
    let right = match hi_z_snapshot_from_linear_linear(extent.0, extent.1, mip_levels, mips_r) {
        Some(s) => s,
        None => {
            logger::warn!("Hi-Z stereo right snapshot validation failed");
            return None;
        }
    };
    Some(HiZStereoCpuSnapshot { left, right })
}

/// Transient GPU resources reused while extent and mip count stay stable.
pub(crate) struct HiZGpuScratch {
    pub extent: (u32, u32),
    pub mip_levels: u32,
    pub pyramid: wgpu::Texture,
    pub views: Vec<wgpu::TextureView>,
    pub pyramid_r: Option<(wgpu::Texture, Vec<wgpu::TextureView>)>,
    /// Triple-buffered staging for async readback (see [`HiZGpuState::write_idx`]).
    pub staging_desktop: [wgpu::Buffer; HIZ_STAGING_RING],
    pub staging_r: Option<[wgpu::Buffer; HIZ_STAGING_RING]>,
    pub layer_uniform: wgpu::Buffer,
    pub downsample_uniform: wgpu::Buffer,
}

fn staging_size_pyramid(base_w: u32, base_h: u32, mip_levels: u32) -> u64 {
    let mut total = 0u64;
    for mip in 0..mip_levels {
        let (w, h) = mip_dimensions(base_w, base_h, mip).unwrap_or((0, 0));
        let row_pitch = wgpu::util::align_to(w * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) as u64;
        total += row_pitch * u64::from(h);
    }
    total
}

fn make_staging_ring(
    device: &wgpu::Device,
    staging_size: u64,
    label_prefix: &str,
) -> [wgpu::Buffer; HIZ_STAGING_RING] {
    std::array::from_fn(|i| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix}_{i}")),
            size: staging_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    })
}

impl HiZGpuScratch {
    pub(crate) fn new(device: &wgpu::Device, extent: (u32, u32), stereo: bool) -> Option<Self> {
        let (bw, bh) = extent;
        if bw == 0 || bh == 0 {
            return None;
        }
        let mip_levels = mip_levels_for_extent(bw, bh, HIZ_MAX_MIPS);
        if mip_levels == 0 {
            return None;
        }

        let make_pyramid = || -> (wgpu::Texture, Vec<wgpu::TextureView>) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("hi_z_pyramid"),
                size: wgpu::Extent3d {
                    width: bw,
                    height: bh,
                    depth_or_array_layers: 1,
                },
                mip_level_count: mip_levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let mut views = Vec::with_capacity(mip_levels as usize);
            for m in 0..mip_levels {
                let v = tex.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("hi_z_pyramid_mip"),
                    format: Some(wgpu::TextureFormat::R32Float),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: m,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: None,
                    ..Default::default()
                });
                views.push(v);
            }
            (tex, views)
        };

        let (pyramid, views) = make_pyramid();
        let staging_size = staging_size_pyramid(bw, bh, mip_levels);
        let staging_desktop = make_staging_ring(device, staging_size, "hi_z_staging_desktop");

        let (pyramid_r, staging_r) = if stereo {
            let (t, v) = make_pyramid();
            let buf = make_staging_ring(device, staging_size, "hi_z_staging_r");
            (Some((t, v)), Some(buf))
        } else {
            (None, None)
        };

        let layer_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hi_z_layer_uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let downsample_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hi_z_downsample_uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Some(Self {
            extent: (bw, bh),
            mip_levels,
            pyramid,
            views,
            pyramid_r,
            staging_desktop,
            staging_r,
            layer_uniform,
            downsample_uniform,
        })
    }
}
