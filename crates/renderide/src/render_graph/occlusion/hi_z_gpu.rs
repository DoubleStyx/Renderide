//! GPU hierarchical depth pyramid build and CPU readback for occlusion tests.

use crossbeam_channel as mpsc;

use crate::render_graph::{
    hi_z_snapshot_from_linear_linear, mip_dimensions, mip_levels_for_extent,
    unpack_linear_rows_to_mips, HiZCpuSnapshot, HiZStereoCpuSnapshot, HiZTemporalState,
    OutputDepthMode,
};

pub(crate) const HIZ_MAX_MIPS: u32 = 8;

/// Triple-buffered staging so a slot is not reused until prior `map_async` completes (non-blocking).
pub(crate) const HIZ_STAGING_RING: usize = 3;

/// `crossbeam_channel::Receiver` is `Send + Sync`, which lets [`HiZGpuState`] (and transitively
/// [`crate::backend::OcclusionSystem`]) be `Sync` so cull-snapshot reads can fan out across rayon
/// workers. `std::sync::mpsc::Receiver` is only `Send`, which was the historical root cause of
/// `OcclusionSystem` being `!Sync` and secondary-camera Hi-Z gathering having to stay serial.
pub(crate) type MapRecv = mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>;

pub(crate) const fn pending_none_array<T>() -> [Option<T>; HIZ_STAGING_RING] {
    [None, None, None]
}

pub(crate) const fn pending_submit_default() -> [bool; HIZ_STAGING_RING] {
    [false; HIZ_STAGING_RING]
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
    /// Transient handoff set by [`crate::render_graph::occlusion::encode_hi_z_build`] naming the
    /// slot to be mapped by the subsequent [`wgpu::Queue::on_submitted_work_done`] callback.
    /// Consumed (taken) on the main thread when the callback closure is constructed so the slot
    /// travels with the closure, not through shared state.
    pub(crate) hi_z_encoded_slot: Option<usize>,
    /// Slots whose copy-to-staging command has been recorded but whose
    /// [`wgpu::Queue::on_submitted_work_done`] callback has not yet fired. Guards
    /// [`Self::can_encode_hi_z`] from picking a slot that the driver thread is about to consume.
    pub(crate) pending_submit: [bool; HIZ_STAGING_RING],
    /// Slots whose `on_submitted_work_done` callback has fired (submit confirmed complete) but
    /// whose `map_async` has not yet been issued. Promoted to [`Self::desktop_pending`] by
    /// [`Self::start_ready_maps`] on the main thread — never inside a device-poll callback, so no
    /// wgpu call runs from within the callback's execution context.
    pub(crate) submit_done: [bool; HIZ_STAGING_RING],
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
            pending_submit: pending_submit_default(),
            submit_done: pending_submit_default(),
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
            self.pending_submit = pending_submit_default();
            self.submit_done = pending_submit_default();
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
        self.pending_submit = pending_submit_default();
        self.submit_done = pending_submit_default();
        self.desktop_pending = pending_none_array();
        self.right_pending = None;
        self.stereo_left_stash = pending_none_array();
        self.stereo_right_stash = pending_none_array();
    }

    /// Drains completed `map_async` work into [`Self::desktop`] / [`Self::stereo`] and promotes
    /// any newly-`submit_done` slots into fresh `map_async` requests. Non-blocking.
    ///
    /// Call at the **start** of each frame (before encoding the render graph). Uses at most one
    /// [`wgpu::Device::poll`] to advance callbacks; if a read is not ready, prior snapshots are kept.
    ///
    /// ### Re-entrance
    ///
    /// [`crate::backend::OcclusionSystem::hi_z_begin_frame_readback`] drains
    /// `on_submitted_work_done` callbacks via [`wgpu::Device::poll`] **before** locking this
    /// state, so the [`Self::mark_submit_done`] callback does not re-enter the mutex.
    /// This helper polls-then-locks itself and is meant for direct callers (mainly tests).
    pub fn begin_frame_readback(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::Poll);
        self.drain_completed_map_async();
        self.start_ready_maps();
    }

    /// Non-polling variant of [`Self::begin_frame_readback`] used when the caller has already
    /// drained completed queue callbacks via [`wgpu::Device::poll`] outside any
    /// [`HiZGpuState`] mutex (see [`crate::backend::OcclusionSystem::hi_z_begin_frame_readback`]).
    pub(crate) fn drain_completed_map_async(&mut self) {
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

    /// Records that the driver-thread submit carrying the copy-to-staging for `ws` has
    /// completed. Does not touch wgpu — [`Self::start_ready_maps`] promotes the slot to a real
    /// `map_async` on the main thread. Keeping this callback pure (just a flag flip) avoids
    /// running any wgpu call from inside a [`wgpu::Device::poll`] callback, which can hold
    /// wgpu-internal locks that also serialize [`wgpu::Queue::write_texture`] and would
    /// otherwise risk a futex-wait deadlock with the asset-upload path on the main thread.
    pub fn mark_submit_done(&mut self, ws: usize) {
        debug_assert!(ws < HIZ_STAGING_RING);
        self.submit_done[ws] = true;
    }

    /// Issues `map_async` for every slot whose submit has completed since the last call.
    /// Runs on the main thread from [`crate::backend::OcclusionSystem::hi_z_begin_frame_readback`]
    /// after `device.poll` has flushed completion callbacks into [`Self::submit_done`].
    pub(crate) fn start_ready_maps(&mut self) {
        let Some(scratch) = self.scratch.as_ref() else {
            for flag in &mut self.submit_done {
                *flag = false;
            }
            for flag in &mut self.pending_submit {
                *flag = false;
            }
            return;
        };

        for ws in 0..HIZ_STAGING_RING {
            if !self.submit_done[ws] {
                continue;
            }
            self.submit_done[ws] = false;
            self.pending_submit[ws] = false;

            if self.desktop_pending[ws].is_some() {
                continue;
            }

            let slice = scratch.staging_desktop[ws].slice(..);
            let (tx, rx) = mpsc::bounded::<Result<(), wgpu::BufferAsyncError>>(1);
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            self.desktop_pending[ws] = Some(rx);

            if let Some(ref staging_r) = scratch.staging_r {
                if self.right_pending.is_none() {
                    self.right_pending = Some(pending_none_array());
                }
                if let Some(rp) = self.right_pending.as_mut() {
                    if rp[ws].is_none() {
                        let slice_r = staging_r[ws].slice(..);
                        let (tx_r, rx_r) = mpsc::bounded::<Result<(), wgpu::BufferAsyncError>>(1);
                        slice_r.map_async(wgpu::MapMode::Read, move |r| {
                            let _ = tx_r.send(r);
                        });
                        rp[ws] = Some(rx_r);
                    }
                }
            }
        }
    }

    pub(crate) fn can_encode_hi_z(&self, scratch: &HiZGpuScratch) -> bool {
        let idx = self.write_idx;
        if self.pending_submit[idx] || self.submit_done[idx] || self.desktop_pending[idx].is_some()
        {
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
    let Some(mips) = unpack_linear_rows_to_mips(extent.0, extent.1, mip_levels, raw) else {
        logger::warn!("Hi-Z desktop readback unpack failed");
        return None;
    };
    match hi_z_snapshot_from_linear_linear(extent.0, extent.1, mip_levels, mips) {
        Some(s) => Some(s),
        None => {
            logger::warn!("Hi-Z desktop snapshot validation failed");
            None
        }
    }
}

/// Unpacks the per-eye CPU snapshots in parallel via [`rayon::join`].
///
/// Each eye performs an independent O(W·H·mips) byte-to-`f32` walk over its own staging buffer
/// (see [`unpack_linear_rows_to_mips`]), then validates dimensions through
/// [`hi_z_snapshot_from_linear_linear`]. The two walks share no state, so fan-out is straightforward
/// and roughly halves stereo Hi-Z readback wall time on multi-core hosts.
fn unpack_stereo_snapshot(
    extent: (u32, u32),
    mip_levels: u32,
    left_raw: &[u8],
    right_raw: &[u8],
) -> Option<HiZStereoCpuSnapshot> {
    let unpack_eye = |label: &'static str, raw: &[u8]| -> Option<HiZCpuSnapshot> {
        let Some(mips) = unpack_linear_rows_to_mips(extent.0, extent.1, mip_levels, raw) else {
            logger::warn!("Hi-Z stereo {label} readback unpack failed");
            return None;
        };
        match hi_z_snapshot_from_linear_linear(extent.0, extent.1, mip_levels, mips) {
            Some(s) => Some(s),
            None => {
                logger::warn!("Hi-Z stereo {label} snapshot validation failed");
                None
            }
        }
    };

    let (left, right) = rayon::join(
        || unpack_eye("left", left_raw),
        || unpack_eye("right", right_raw),
    );
    Some(HiZStereoCpuSnapshot {
        left: left?,
        right: right?,
    })
}

/// Transient GPU resources reused while extent and mip count stay stable.
pub(crate) struct HiZGpuScratch {
    /// Pyramid base extent `(width, height)` in texels.
    pub extent: (u32, u32),
    /// Total mip count (mip0 through `mip_levels - 1`).
    pub mip_levels: u32,
    /// Desktop / stereo-left pyramid texture.
    pub pyramid: wgpu::Texture,
    /// Per-mip views for the desktop / stereo-left pyramid.
    pub views: Vec<wgpu::TextureView>,
    /// Stereo-right pyramid (texture + per-mip views), populated only in stereo modes.
    pub pyramid_r: Option<(wgpu::Texture, Vec<wgpu::TextureView>)>,
    /// Triple-buffered staging for async readback (see [`HiZGpuState::write_idx`]).
    pub staging_desktop: [wgpu::Buffer; HIZ_STAGING_RING],
    /// Triple-buffered staging for the stereo-right pyramid.
    pub staging_r: Option<[wgpu::Buffer; HIZ_STAGING_RING]>,
    /// Uniform buffer carrying the current target array layer for stereo mip0 dispatches.
    pub layer_uniform: wgpu::Buffer,
    /// Uniform buffer carrying src/dst extents for the active downsample dispatch.
    pub downsample_uniform: wgpu::Buffer,
    /// Cached bind groups for this scratch's pipelines. Invalidated alongside the scratch itself
    /// (i.e. when `extent` / `mip_levels` / stereo layout changes trigger a fresh allocation).
    pub bind_groups: HiZBindGroupCache,
}

/// Cached Hi-Z encode bind groups whose bindings are stable for the lifetime of a
/// [`HiZGpuScratch`]. Built lazily on first use so the cache is both cheap to initialise and
/// self-invalidating: recreating the scratch wipes every slot.
pub(crate) struct HiZBindGroupCache {
    /// `depth_view` last bound into the mip0 slots. Mip0 bindings are rebuilt when this changes
    /// (e.g. depth target reallocation between frames).
    mip0_depth_view: Option<wgpu::TextureView>,
    /// Mip0 bind group for desktop (non-stereo) dispatches.
    mip0_desktop: Option<wgpu::BindGroup>,
    /// Mip0 bind groups for stereo dispatches, indexed by array layer (`[layer0, layer1]`).
    mip0_stereo: [Option<wgpu::BindGroup>; 2],
    /// Downsample bind groups for the desktop / stereo-left pyramid, one per mip transition.
    downsample_desktop: Vec<Option<wgpu::BindGroup>>,
    /// Downsample bind groups for the stereo-right pyramid, one per mip transition.
    downsample_right: Vec<Option<wgpu::BindGroup>>,
}

impl HiZBindGroupCache {
    /// Creates an empty cache sized for `mip_levels` transitions; allocates a right-eye slot set
    /// only when `stereo` is true.
    fn with_shape(mip_levels: u32, stereo: bool) -> Self {
        let n = (mip_levels.saturating_sub(1)) as usize;
        Self {
            mip0_depth_view: None,
            mip0_desktop: None,
            mip0_stereo: [None, None],
            downsample_desktop: vec![None; n],
            downsample_right: if stereo { vec![None; n] } else { Vec::new() },
        }
    }

    /// Drops the mip0 slots whenever the caller-provided `depth_view` differs from the one used
    /// to build the cached entries.
    pub(crate) fn invalidate_mip0_if_depth_changed(&mut self, depth_view: &wgpu::TextureView) {
        if self.mip0_depth_view.as_ref() != Some(depth_view) {
            self.mip0_depth_view = Some(depth_view.clone());
            self.mip0_desktop = None;
            self.mip0_stereo = [None, None];
        }
    }

    /// Returns a clone of the cached mip0 desktop bind group, building it via `build` on miss.
    pub(crate) fn mip0_desktop_or_build<F: FnOnce() -> wgpu::BindGroup>(
        &mut self,
        build: F,
    ) -> wgpu::BindGroup {
        self.mip0_desktop.get_or_insert_with(build).clone()
    }

    /// Returns a clone of the cached mip0 stereo bind group for `layer`, building via `build` on miss.
    pub(crate) fn mip0_stereo_or_build<F: FnOnce() -> wgpu::BindGroup>(
        &mut self,
        layer: u32,
        build: F,
    ) -> wgpu::BindGroup {
        let idx = (layer as usize).min(1);
        self.mip0_stereo[idx].get_or_insert_with(build).clone()
    }

    /// Returns a clone of the desktop downsample bind group at `mip`, building via `build` on miss.
    pub(crate) fn downsample_desktop_or_build<F: FnOnce() -> wgpu::BindGroup>(
        &mut self,
        mip: u32,
        build: F,
    ) -> wgpu::BindGroup {
        let idx = mip as usize;
        self.downsample_desktop[idx]
            .get_or_insert_with(build)
            .clone()
    }

    /// Returns a clone of the stereo-right downsample bind group at `mip`, building via `build` on miss.
    pub(crate) fn downsample_right_or_build<F: FnOnce() -> wgpu::BindGroup>(
        &mut self,
        mip: u32,
        build: F,
    ) -> wgpu::BindGroup {
        let idx = mip as usize;
        self.downsample_right[idx].get_or_insert_with(build).clone()
    }
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

        let bind_groups = HiZBindGroupCache::with_shape(mip_levels, stereo);
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
            bind_groups,
        })
    }
}
