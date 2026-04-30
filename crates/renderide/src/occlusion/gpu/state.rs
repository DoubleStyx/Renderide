//! Per-view Hi-Z GPU state: CPU snapshots, GPU scratch, and the readback ring lifecycle.
//!
//! [`HiZGpuState`] is the single piece of state guarded by a `parking_lot::Mutex` per view in
//! [`crate::occlusion::OcclusionSystem`]. It deliberately avoids heavy logic — encode (in
//! [`super::encode`]) and readback drain (in [`super::readback`]) operate on `&mut HiZGpuState`
//! through narrow entry points so the lock-held footprint stays small.

use crate::gpu::OutputDepthMode;
use crate::occlusion::cpu::snapshot::{HiZCpuSnapshot, HiZStereoCpuSnapshot};
use crate::world_mesh::HiZTemporalState;

use super::readback::StereoStash;
use super::readback_ring::GpuReadbackRing;
use super::scratch::HiZGpuScratch;

/// GPU + CPU Hi-Z state owned by [`crate::occlusion::OcclusionSystem`].
pub struct HiZGpuState {
    /// Last successfully read desktop pyramid (previous frame).
    pub desktop: Option<HiZCpuSnapshot>,
    /// Last successfully read stereo pyramids (previous frame).
    pub stereo: Option<HiZStereoCpuSnapshot>,
    /// View/projection snapshot for the frame that produced [`Self::desktop`] / [`Self::stereo`].
    pub temporal: Option<HiZTemporalState>,
    /// GPU scratch resources reused while the pyramid extent and stereo layout are unchanged.
    pub(super) scratch: Option<HiZGpuScratch>,
    /// Last framebuffer extent used to validate [`Self::scratch`] and CPU snapshots.
    last_extent: (u32, u32),
    /// Last depth output mode used to validate [`Self::scratch`] and CPU snapshots.
    last_mode: OutputDepthMode,
    /// Submit, map, and staging-slot ownership for the Hi-Z readback ring.
    pub(super) readback: GpuReadbackRing,
    /// Partial stereo CPU bytes pending pairing into [`HiZStereoCpuSnapshot`].
    pub(super) stereo_stash: StereoStash,
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
            stereo_stash: StereoStash::default(),
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
            self.stereo_stash.clear();
        }
        self.last_extent = extent;
        self.last_mode = mode;
    }

    /// Clears ring readback state without mapping (e.g. device loss).
    pub fn clear_pending(&mut self) {
        self.readback.reset();
        self.stereo_stash.clear();
    }

    /// Drains completed `map_async` work into [`Self::desktop`] / [`Self::stereo`] and promotes
    /// any newly-`submit_done` slots into fresh `map_async` requests. Non-blocking.
    ///
    /// Call at the **start** of each frame (before encoding the render graph). Uses at most one
    /// [`wgpu::Device::poll`] to advance callbacks; if a read is not ready, prior snapshots are kept.
    ///
    /// ### Re-entrance
    ///
    /// [`crate::occlusion::OcclusionSystem::hi_z_begin_frame_readback`] drains
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
    /// [`HiZGpuState`] mutex (see [`crate::occlusion::OcclusionSystem::hi_z_begin_frame_readback`]).
    pub(crate) fn drain_completed_map_async(&mut self) {
        super::readback::drain(self);
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
    /// Runs on the main thread from [`crate::occlusion::OcclusionSystem::hi_z_begin_frame_readback`].
    pub(crate) fn start_ready_maps(&mut self) {
        super::readback::start_ready_maps(self);
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
        self.readback.can_claim_next_slot(scratch.is_stereo())
    }

    /// Replaces the GPU scratch for this view, clearing any stale ring state and stashed bytes.
    pub(super) fn replace_scratch(&mut self, scratch: Option<HiZGpuScratch>) {
        self.scratch = scratch;
        self.clear_pending();
    }

    /// Returns immutable access to the active GPU scratch for callers in the encode path.
    pub(super) fn scratch(&self) -> Option<&HiZGpuScratch> {
        self.scratch.as_ref()
    }

    /// Returns mutable access to the active GPU scratch for callers in the encode path.
    pub(super) fn scratch_mut(&mut self) -> Option<&mut HiZGpuScratch> {
        self.scratch.as_mut()
    }
}
