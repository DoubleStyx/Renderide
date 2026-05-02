//! Lockstep loop: drains both `...S` queues, replies to every `FrameStartData` with the current
//! `FrameSubmitData`, and lets callers inspect drained messages for asset acks (e.g.
//! `MeshUploadResult`).
//!
//! `frame_index` starts at 0 and increments on each submit, mirroring the production C# host's
//! monotonically-incrementing per-tick counter.

use renderide_shared::ipc::HostDualQueueIpc;
use renderide_shared::shared::{FrameSubmitData, RenderSpaceUpdate, RendererCommand};

/// Per-tick lockstep state: how many `FrameStartData` messages we observed and the full drained
/// command batch (for callers that want to look for asset acks like `MeshUploadResult`).
#[derive(Debug, Default)]
pub struct TickResult {
    /// Number of `FrameStartData` messages consumed this tick.
    pub frame_starts: u32,
    /// Number of `FrameSubmitData` replies actually enqueued (may be `< frame_starts` if the
    /// publisher rejected the send because the queue was full).
    pub frame_submits_sent: u32,
    /// All non-`FrameStartData` messages drained this tick (asset acks, lights consumption, etc.).
    pub other_messages: Vec<RendererCommand>,
}

/// Lockstep driver shared across the harness. Owns the current scene state plus the per-tick
/// frame counter.
pub struct LockstepDriver {
    /// Frame index for the next `FrameSubmitData` we send.
    frame_index: i32,
    /// Fixed scalar fields applied to every `FrameSubmitData` (clip planes, FOV, `vr_active` flag).
    pub frame_scalars: FrameSubmitScalars,
    /// Current scene state. `None` during the handshake / asset-upload phases (no render-space yet);
    /// switch to `Some(...)` once the sphere has been uploaded and we want it on screen.
    pub current_render_space: Option<RenderSpaceUpdate>,
}

/// Static per-frame scalar fields that the harness sets once and reuses across every submit.
#[derive(Clone, Copy, Debug)]
pub struct FrameSubmitScalars {
    /// Near-clip plane in meters.
    pub near_clip: f32,
    /// Far-clip plane in meters.
    pub far_clip: f32,
    /// Vertical FOV in **degrees** (matches `HostCameraFrame::desktop_fov_degrees`).
    pub desktop_fov: f32,
    /// `true` when the host is in VR mode. Always `false` for the headless integration test.
    pub vr_active: bool,
    /// Toggle for the renderer's debug-log frame snapshot. Default `false`.
    pub debug_log: bool,
}

impl Default for FrameSubmitScalars {
    fn default() -> Self {
        Self {
            near_clip: 0.1,
            far_clip: 100.0,
            desktop_fov: 60.0,
            vr_active: false,
            debug_log: false,
        }
    }
}

impl LockstepDriver {
    /// Creates a driver with `frame_index = 0` and no scene attached yet (handshake / upload
    /// phase).
    pub const fn new(frame_scalars: FrameSubmitScalars) -> Self {
        Self {
            frame_index: 0,
            frame_scalars,
            current_render_space: None,
        }
    }

    /// Latches a render-space update so future `FrameSubmitData` messages embed the sphere scene.
    /// Pass `None` to revert to "empty" frame submits.
    pub fn set_render_space(&mut self, update: Option<RenderSpaceUpdate>) {
        self.current_render_space = update;
    }

    /// Current frame index that will be assigned to the **next** outgoing `FrameSubmitData`.
    pub const fn current_frame_index(&self) -> i32 {
        self.frame_index
    }

    /// Drains both `...S` queues and replies to every `FrameStartData` with a `FrameSubmitData`
    /// built from the current scene state.
    pub(super) fn tick(&mut self, queues: &mut HostDualQueueIpc) -> TickResult {
        let mut drained = Vec::new();
        queues.poll_into(&mut drained);
        let mut result = TickResult::default();
        for cmd in drained {
            match cmd {
                RendererCommand::FrameStartData(_) => {
                    result.frame_starts += 1;
                    if self.send_frame_submit(queues) {
                        result.frame_submits_sent += 1;
                    }
                }
                other => result.other_messages.push(other),
            }
        }
        result
    }

    fn send_frame_submit(&mut self, queues: &mut HostDualQueueIpc) -> bool {
        let submit = build_frame_submit_data(
            self.frame_index,
            &self.frame_scalars,
            self.current_render_space.as_ref(),
        );
        let sent = queues.send_primary(RendererCommand::FrameSubmitData(submit));
        if sent {
            self.frame_index = self.frame_index.wrapping_add(1);
        } else {
            logger::warn!(
                "Lockstep: failed to send FrameSubmitData (queue full?); frame_index unchanged"
            );
        }
        sent
    }
}

/// Builds a [`FrameSubmitData`] message with the supplied frame index, scalar fields, and
/// optional render-space. Pure: no I/O, no clock, no global state.
///
/// Extracted from [`LockstepDriver::send_frame_submit`] so unit tests can verify the construction
/// rules (frame index propagation, scalar plumbing, render-space `Vec` shape) without standing up
/// a real Cloudtoid queue.
pub fn build_frame_submit_data(
    frame_index: i32,
    scalars: &FrameSubmitScalars,
    render_space: Option<&RenderSpaceUpdate>,
) -> FrameSubmitData {
    let render_spaces = render_space.map(|rs| vec![rs.clone()]).unwrap_or_default();
    FrameSubmitData {
        frame_index,
        debug_log: scalars.debug_log,
        vr_active: scalars.vr_active,
        near_clip: scalars.near_clip,
        far_clip: scalars.far_clip,
        desktop_fov: scalars.desktop_fov,
        output_state: None,
        render_spaces,
        render_tasks: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_result_default_is_empty() {
        let r = TickResult::default();
        assert_eq!(r.frame_starts, 0);
        assert_eq!(r.frame_submits_sent, 0);
        assert!(r.other_messages.is_empty());
    }

    #[test]
    fn driver_starts_at_frame_index_zero() {
        let d = LockstepDriver::new(FrameSubmitScalars::default());
        assert_eq!(d.current_frame_index(), 0);
        assert!(d.current_render_space.is_none());
    }

    #[test]
    fn frame_submit_scalars_default_values() {
        let s = FrameSubmitScalars::default();
        assert!((s.near_clip - 0.1).abs() < 1e-6);
        assert!((s.far_clip - 100.0).abs() < 1e-3);
        assert!((s.desktop_fov - 60.0).abs() < 1e-6);
        assert!(!s.vr_active);
        assert!(!s.debug_log);
    }

    #[test]
    fn set_render_space_to_some_then_none_round_trip() {
        let mut d = LockstepDriver::new(FrameSubmitScalars::default());
        d.set_render_space(Some(RenderSpaceUpdate::default()));
        assert!(d.current_render_space.is_some());
        d.set_render_space(None);
        assert!(d.current_render_space.is_none());
    }

    #[test]
    fn current_frame_index_unchanged_by_set_render_space() {
        let mut d = LockstepDriver::new(FrameSubmitScalars::default());
        let before = d.current_frame_index();
        d.set_render_space(Some(RenderSpaceUpdate::default()));
        d.set_render_space(None);
        assert_eq!(d.current_frame_index(), before);
    }

    #[test]
    fn build_frame_submit_data_with_no_render_space_yields_empty_render_spaces() {
        let scalars = FrameSubmitScalars::default();
        let submit = build_frame_submit_data(7, &scalars, None);
        assert_eq!(submit.frame_index, 7);
        assert!(submit.render_spaces.is_empty());
        assert!(submit.render_tasks.is_empty());
        assert!(submit.output_state.is_none());
    }

    #[test]
    fn build_frame_submit_data_with_some_render_space_yields_one_entry() {
        let scalars = FrameSubmitScalars::default();
        let rs = RenderSpaceUpdate::default();
        let submit = build_frame_submit_data(0, &scalars, Some(&rs));
        assert_eq!(submit.render_spaces.len(), 1);
    }

    #[test]
    fn build_frame_submit_data_uses_provided_frame_index_and_scalars() {
        let scalars = FrameSubmitScalars {
            near_clip: 0.25,
            far_clip: 5000.0,
            desktop_fov: 90.0,
            vr_active: true,
            debug_log: true,
        };
        let submit = build_frame_submit_data(42, &scalars, None);
        assert_eq!(submit.frame_index, 42);
        assert!((submit.near_clip - 0.25).abs() < 1e-6);
        assert!((submit.far_clip - 5000.0).abs() < 1e-3);
        assert!((submit.desktop_fov - 90.0).abs() < 1e-6);
        assert!(submit.vr_active);
        assert!(submit.debug_log);
    }
}
