//! Hi-Z readback drain: turns completed `map_async` callbacks into [`HiZCpuSnapshot`]s.
//!
//! Owns the stereo pairing buffer ([`StereoStash`]) and the snapshot decoders. The
//! [`HiZGpuState`] owner forwards `drain` and `start_ready_maps` here so the state struct stays
//! focused on lifecycle.

use crossbeam_channel as mpsc;

use crate::occlusion::cpu::readback::{
    hi_z_snapshot_from_linear_linear, unpack_linear_rows_to_mips,
};
use crate::occlusion::cpu::snapshot::{HiZCpuSnapshot, HiZStereoCpuSnapshot};

use super::readback_ring::{HIZ_STAGING_RING, pending_none_array};
use super::state::HiZGpuState;

/// Per-slot left/right byte buffers that pair stereo readbacks before decode.
///
/// Both rings flow into [`StereoStash`]; pairs are taken only after both eyes for the same slot
/// have completed. A reset clears both halves; this lives in the readback module so the state
/// struct does not own the pairing rules.
pub(super) struct StereoStash {
    left: [Option<Vec<u8>>; HIZ_STAGING_RING],
    right: [Option<Vec<u8>>; HIZ_STAGING_RING],
}

impl Default for StereoStash {
    fn default() -> Self {
        Self {
            left: pending_none_array(),
            right: pending_none_array(),
        }
    }
}

impl StereoStash {
    /// Drops every stashed buffer (used on invalidation / device loss).
    pub(super) fn clear(&mut self) {
        self.left = pending_none_array();
        self.right = pending_none_array();
    }

    /// Stashes the just-read left-eye bytes for `slot`.
    fn set_left(&mut self, slot: usize, raw: Vec<u8>) {
        self.left[slot] = Some(raw);
    }

    /// Stashes the just-read right-eye bytes for `slot`.
    fn set_right(&mut self, slot: usize, raw: Vec<u8>) {
        self.right[slot] = Some(raw);
    }

    /// Takes both halves for `slot` if both have arrived.
    fn take_pair(&mut self, slot: usize) -> Option<(Vec<u8>, Vec<u8>)> {
        if self.left[slot].is_some() && self.right[slot].is_some() {
            Some((self.left[slot].take()?, self.right[slot].take()?))
        } else {
            None
        }
    }
}

/// Drains completed primary / secondary `map_async` work into desktop / stereo CPU snapshots.
///
/// Caller must already have run [`wgpu::Device::poll`] **outside** any [`HiZGpuState`] mutex (see
/// [`crate::occlusion::OcclusionSystem::hi_z_begin_frame_readback`]).
pub(super) fn drain(state: &mut HiZGpuState) {
    profiling::scope!("hi_z::readback_drain_state");
    let Some(shape) = ScratchShape::from(state) else {
        return;
    };

    {
        profiling::scope!("hi_z::readback_drain_primary_slots");
        for slot in 0..HIZ_STAGING_RING {
            drain_primary_slot(state, slot, shape);
        }
    }

    if shape.stereo {
        state.readback.set_secondary_enabled(true);
        {
            profiling::scope!("hi_z::readback_drain_secondary_slots");
            for slot in 0..HIZ_STAGING_RING {
                drain_secondary_slot(state, slot);
            }
        }
        combine_paired_stereo(state, shape);
    }
}

/// Issues `map_async` for every slot whose submit has completed since the last call.
pub(super) fn start_ready_maps(state: &mut HiZGpuState) {
    profiling::scope!("hi_z::start_ready_maps");
    let primary_staging = state
        .scratch
        .as_ref()
        .map(|scratch| &scratch.staging_desktop);
    let secondary_staging = state
        .scratch
        .as_ref()
        .and_then(|scratch| scratch.staging_right());
    state
        .readback
        .start_ready_maps(primary_staging, secondary_staging);
}

#[derive(Clone, Copy)]
struct ScratchShape {
    extent: (u32, u32),
    mip_levels: u32,
    stereo: bool,
}

impl ScratchShape {
    fn from(state: &HiZGpuState) -> Option<Self> {
        let scratch = state.scratch.as_ref()?;
        Some(Self {
            extent: scratch.extent,
            mip_levels: scratch.mip_levels,
            stereo: scratch.is_stereo(),
        })
    }
}

fn drain_primary_slot(state: &mut HiZGpuState, slot: usize, shape: ScratchShape) {
    let recv_result = state
        .readback
        .primary_pending_mut(slot)
        .as_mut()
        .map(|recv| recv.try_recv());
    let Some(recv_result) = recv_result else {
        return;
    };

    match recv_result {
        Ok(Ok(())) => {
            profiling::scope!("hi_z::readback_primary_slot");
            let buf = primary_staging_buffer(state, slot);
            let Some(raw) = buf.and_then(read_mapped_buffer) else {
                *state.readback.primary_pending_mut(slot) = None;
                return;
            };
            *state.readback.primary_pending_mut(slot) = None;
            apply_primary_bytes(state, slot, shape, raw);
        }
        Ok(Err(_)) => {
            if let Some(buf) = primary_staging_buffer(state, slot) {
                buf.unmap();
            }
            *state.readback.primary_pending_mut(slot) = None;
        }
        Err(mpsc::TryRecvError::Empty) => {}
        Err(mpsc::TryRecvError::Disconnected) => {
            *state.readback.primary_pending_mut(slot) = None;
        }
    }
}

fn apply_primary_bytes(state: &mut HiZGpuState, slot: usize, shape: ScratchShape, raw: Vec<u8>) {
    if shape.stereo {
        state.stereo_stash.set_left(slot, raw);
    } else if let Some(snap) = unpack_desktop_snapshot(shape.extent, shape.mip_levels, &raw) {
        state.desktop = Some(snap);
        state.stereo = None;
    }
}

fn drain_secondary_slot(state: &mut HiZGpuState, slot: usize) {
    let recv_result = state
        .readback
        .secondary_pending_mut(slot)
        .and_then(|pending| pending.as_mut().map(|recv| recv.try_recv()));
    let Some(recv_result) = recv_result else {
        return;
    };

    match recv_result {
        Ok(Ok(())) => {
            profiling::scope!("hi_z::readback_secondary_slot");
            let buf = secondary_staging_buffer(state, slot);
            let Some(raw) = buf.and_then(read_mapped_buffer) else {
                if let Some(pending) = state.readback.secondary_pending_mut(slot) {
                    *pending = None;
                }
                return;
            };
            if let Some(pending) = state.readback.secondary_pending_mut(slot) {
                *pending = None;
            }
            state.stereo_stash.set_right(slot, raw);
        }
        Ok(Err(_)) => {
            if let Some(buf) = secondary_staging_buffer(state, slot) {
                buf.unmap();
            }
            if let Some(pending) = state.readback.secondary_pending_mut(slot) {
                *pending = None;
            }
        }
        Err(mpsc::TryRecvError::Empty) => {}
        Err(mpsc::TryRecvError::Disconnected) => {
            if let Some(pending) = state.readback.secondary_pending_mut(slot) {
                *pending = None;
            }
        }
    }
}

fn combine_paired_stereo(state: &mut HiZGpuState, shape: ScratchShape) {
    profiling::scope!("hi_z::combine_paired_stereo");
    for slot in 0..HIZ_STAGING_RING {
        let Some((left_raw, right_raw)) = state.stereo_stash.take_pair(slot) else {
            continue;
        };
        if let Some(stereo_snap) =
            unpack_stereo_snapshot(shape.extent, shape.mip_levels, &left_raw, &right_raw)
        {
            state.stereo = Some(stereo_snap);
            state.desktop = None;
        }
    }
}

fn primary_staging_buffer(state: &HiZGpuState, slot: usize) -> Option<&wgpu::Buffer> {
    state
        .scratch
        .as_ref()
        .map(|scratch| &scratch.staging_desktop[slot])
}

fn secondary_staging_buffer(state: &HiZGpuState, slot: usize) -> Option<&wgpu::Buffer> {
    state
        .scratch
        .as_ref()
        .and_then(|scratch| scratch.staging_right().map(|ring| &ring[slot]))
}

/// Reads and unmaps a completed staging buffer into CPU-owned bytes.
fn read_mapped_buffer(buf: &wgpu::Buffer) -> Option<Vec<u8>> {
    profiling::scope!("hi_z::read_mapped_buffer");
    let range = buf.slice(..).get_mapped_range().to_vec();
    buf.unmap();
    Some(range)
}

fn unpack_desktop_snapshot(
    extent: (u32, u32),
    mip_levels: u32,
    raw: &[u8],
) -> Option<HiZCpuSnapshot> {
    profiling::scope!("hi_z::unpack_desktop_snapshot");
    let Some(mips) = unpack_linear_rows_to_mips(extent.0, extent.1, mip_levels, raw) else {
        logger::warn!("Hi-Z desktop readback unpack failed");
        return None;
    };
    if let Some(s) = hi_z_snapshot_from_linear_linear(extent.0, extent.1, mip_levels, mips) {
        Some(s)
    } else {
        logger::warn!("Hi-Z desktop snapshot validation failed");
        None
    }
}

/// Unpacks the per-eye CPU snapshots in parallel via [`rayon::join`].
///
/// Each eye performs an independent O(W*H*mips) byte-to-`f32` walk over its own staging buffer
/// (see [`unpack_linear_rows_to_mips`]), then validates dimensions through
/// [`hi_z_snapshot_from_linear_linear`]. The two walks share no state, so fan-out is straightforward
/// and roughly halves stereo Hi-Z readback wall time on multi-core hosts.
fn unpack_stereo_snapshot(
    extent: (u32, u32),
    mip_levels: u32,
    left_raw: &[u8],
    right_raw: &[u8],
) -> Option<HiZStereoCpuSnapshot> {
    profiling::scope!("hi_z::unpack_stereo_snapshot");
    let unpack_eye = |label: &'static str, raw: &[u8]| -> Option<HiZCpuSnapshot> {
        profiling::scope!("hi_z::unpack_stereo_eye", label);
        let Some(mips) = unpack_linear_rows_to_mips(extent.0, extent.1, mip_levels, raw) else {
            logger::warn!("Hi-Z stereo {label} readback unpack failed");
            return None;
        };
        if let Some(s) = hi_z_snapshot_from_linear_linear(extent.0, extent.1, mip_levels, mips) {
            Some(s)
        } else {
            logger::warn!("Hi-Z stereo {label} snapshot validation failed");
            None
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
