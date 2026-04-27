//! GPU hierarchical depth pyramid build and CPU readback for occlusion tests.

use crossbeam_channel as mpsc;

use crate::render_graph::{
    hi_z_snapshot_from_linear_linear, mip_dimensions, mip_levels_for_extent,
    unpack_linear_rows_to_mips, HiZCpuSnapshot, HiZStereoCpuSnapshot, HiZTemporalState,
    OutputDepthMode,
};

use super::readback_ring::{pending_none_array, GpuReadbackRing};

/// Maximum number of mip levels retained in each Hi-Z pyramid.
pub(crate) const HIZ_MAX_MIPS: u32 = 8;

/// Triple-buffered staging so a slot is not reused until prior `map_async` completes (non-blocking).
pub(crate) use super::readback_ring::HIZ_STAGING_RING;

/// GPU + CPU Hi-Z state owned by [`crate::backend::OcclusionSystem`].
pub struct HiZGpuState {
    /// Last successfully read desktop pyramid (previous frame).
    pub desktop: Option<HiZCpuSnapshot>,
    /// Last successfully read stereo pyramids (previous frame).
    pub stereo: Option<HiZStereoCpuSnapshot>,
    /// View/projection snapshot for the frame that produced [`Self::desktop`] / [`Self::stereo`].
    pub temporal: Option<HiZTemporalState>,
    /// GPU scratch resources reused while the pyramid extent and stereo layout are unchanged.
    pub(crate) scratch: Option<HiZGpuScratch>,
    /// Last framebuffer extent used to validate [`Self::scratch`] and CPU snapshots.
    last_extent: (u32, u32),
    /// Last depth output mode used to validate [`Self::scratch`] and CPU snapshots.
    last_mode: OutputDepthMode,
    /// Submit, map, and staging-slot ownership for the Hi-Z readback ring.
    pub(crate) readback: GpuReadbackRing,
    /// Partial stereo CPU bytes until both eyes for the same ring slot complete.
    pub(crate) stereo_left_stash: [Option<Vec<u8>>; HIZ_STAGING_RING],
    /// Partial stereo CPU bytes for right-eye readbacks until the matching left eye completes.
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
            readback: GpuReadbackRing::default(),
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
            self.readback.reset();
            self.stereo_left_stash = pending_none_array();
            self.stereo_right_stash = pending_none_array();
        }
        self.last_extent = extent;
        self.last_mode = mode;
    }

    /// Clears ring readback state without mapping (e.g. device loss).
    pub fn clear_pending(&mut self) {
        self.readback.reset();
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
            let recv_result = self
                .readback
                .primary_pending_mut(i)
                .as_mut()
                .map(|recv| recv.try_recv());
            let Some(recv_result) = recv_result else {
                continue;
            };
            match recv_result {
                Ok(Ok(())) => {
                    let buf = &scratch.staging_desktop[i];
                    let Some(raw) = read_mapped_buffer(buf) else {
                        *self.readback.primary_pending_mut(i) = None;
                        continue;
                    };
                    *self.readback.primary_pending_mut(i) = None;
                    if stereo {
                        self.stereo_left_stash[i] = Some(raw);
                    } else if let Some(snap) = unpack_desktop_snapshot(extent, mip_levels, &raw) {
                        self.desktop = Some(snap);
                        self.stereo = None;
                    }
                }
                Ok(Err(_)) => {
                    scratch.staging_desktop[i].unmap();
                    *self.readback.primary_pending_mut(i) = None;
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    *self.readback.primary_pending_mut(i) = None;
                }
            }
        }

        if stereo {
            if let Some(ref staging_r) = scratch.staging_r {
                self.readback.set_secondary_enabled(true);
                for (i, right_staging_buffer) in staging_r.iter().enumerate().take(HIZ_STAGING_RING)
                {
                    let recv_result = self
                        .readback
                        .secondary_pending_mut(i)
                        .and_then(|pending| pending.as_mut().map(|recv| recv.try_recv()));
                    let Some(recv_result) = recv_result else {
                        continue;
                    };
                    match recv_result {
                        Ok(Ok(())) => {
                            let Some(raw) = read_mapped_buffer(right_staging_buffer) else {
                                if let Some(right_pending) = self.readback.secondary_pending_mut(i)
                                {
                                    *right_pending = None;
                                }
                                continue;
                            };
                            if let Some(right_pending) = self.readback.secondary_pending_mut(i) {
                                *right_pending = None;
                            }
                            self.stereo_right_stash[i] = Some(raw);
                        }
                        Ok(Err(_)) => {
                            right_staging_buffer.unmap();
                            if let Some(right_pending) = self.readback.secondary_pending_mut(i) {
                                *right_pending = None;
                            }
                        }
                        Err(mpsc::TryRecvError::Empty) => {}
                        Err(mpsc::TryRecvError::Disconnected) => {
                            if let Some(right_pending) = self.readback.secondary_pending_mut(i) {
                                *right_pending = None;
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
        self.readback.mark_submit_done(ws);
    }

    /// Issues `map_async` for every slot whose submit has completed since the last call.
    /// Runs on the main thread from [`crate::backend::OcclusionSystem::hi_z_begin_frame_readback`].
    pub(crate) fn start_ready_maps(&mut self) {
        let primary_staging = self
            .scratch
            .as_ref()
            .map(|scratch| &scratch.staging_desktop);
        let secondary_staging = self
            .scratch
            .as_ref()
            .and_then(|scratch| scratch.staging_r.as_ref());
        self.readback
            .start_ready_maps(primary_staging, secondary_staging);
    }

    /// Returns the staging slot that the next successful Hi-Z encode should target.
    pub(crate) fn next_write_slot(&self) -> usize {
        self.readback.next_write_slot()
    }

    /// Claims the current staging slot after Hi-Z encode commands were recorded successfully.
    pub(crate) fn claim_encoded_slot(&mut self) -> usize {
        self.readback.claim_next_slot()
    }

    /// Clears any stale encoded-slot handoff before attempting a new encode.
    pub(crate) fn clear_encoded_slot(&mut self) {
        self.readback.clear_encoded_slot();
    }

    /// Takes the encoded-slot handoff for queue-submit callback installation.
    pub(crate) fn take_encoded_slot(&mut self) -> Option<usize> {
        self.readback.take_encoded_slot()
    }

    /// Ensures the readback ring has the pending-slot shape required by the current output mode.
    pub(crate) fn set_secondary_readback_enabled(&mut self, enabled: bool) {
        self.readback.set_secondary_enabled(enabled);
    }

    /// Returns true when the readback ring can safely accept another Hi-Z staging copy.
    pub(crate) fn can_encode_hi_z(&self, scratch: &HiZGpuScratch) -> bool {
        self.readback
            .can_claim_next_slot(scratch.staging_r.is_some())
    }
}

/// Reads and unmaps a completed staging buffer into CPU-owned bytes.
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
    /// Triple-buffered staging for async readback.
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
    /// Mip0 view last used as the desktop/left pyramid output.
    pyramid_left_mip0_view: Option<wgpu::TextureView>,
    /// Mip0 view last used as the stereo-right pyramid output.
    pyramid_right_mip0_view: Option<wgpu::TextureView>,
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
            pyramid_left_mip0_view: None,
            pyramid_right_mip0_view: None,
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

    /// Drops bind groups that reference the destination pyramid when the ping-pong half changes.
    pub(crate) fn invalidate_pyramid_if_target_changed(
        &mut self,
        left_mip0_view: &wgpu::TextureView,
        right_mip0_view: Option<&wgpu::TextureView>,
    ) {
        if self.pyramid_left_mip0_view.as_ref() == Some(left_mip0_view)
            && self.pyramid_right_mip0_view.as_ref() == right_mip0_view
        {
            return;
        }
        self.pyramid_left_mip0_view = Some(left_mip0_view.clone());
        self.pyramid_right_mip0_view = right_mip0_view.cloned();
        self.mip0_desktop = None;
        self.mip0_stereo = [None, None];
        for slot in &mut self.downsample_desktop {
            *slot = None;
        }
        for slot in &mut self.downsample_right {
            *slot = None;
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
    pub(crate) fn new(
        device: &wgpu::Device,
        limits: &crate::gpu::GpuLimits,
        extent: (u32, u32),
        stereo: bool,
    ) -> Option<Self> {
        let (bw, bh) = extent;
        if bw == 0 || bh == 0 {
            return None;
        }
        if !limits.texture_2d_fits(bw, bh) {
            logger::warn!(
                "hi_z scratch: pyramid extent {bw}x{bh} exceeds max_texture_dimension_2d={}; skipping",
                limits.max_texture_dimension_2d()
            );
            return None;
        }
        let mip_levels = mip_levels_for_extent(bw, bh, HIZ_MAX_MIPS);
        if mip_levels == 0 {
            return None;
        }
        let staging_size = staging_size_pyramid(bw, bh, mip_levels);
        if !limits.buffer_size_fits(staging_size) {
            logger::warn!(
                "hi_z scratch: staging size {staging_size} exceeds max_buffer_size={}; skipping",
                limits.max_buffer_size()
            );
            return None;
        }

        let staging_desktop = make_staging_ring(device, staging_size, "hi_z_staging_desktop");

        let staging_r = if stereo {
            Some(make_staging_ring(device, staging_size, "hi_z_staging_r"))
        } else {
            None
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
            staging_desktop,
            staging_r,
            layer_uniform,
            downsample_uniform,
            bind_groups,
        })
    }
}
