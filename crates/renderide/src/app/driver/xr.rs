//! OpenXR frame begin/end, HMD multiview submission, and IPC input cache for the app driver.

use glam::{Quat, Vec3};

use crate::diagnostics::gpu_flight_recorder::GpuFlightRecorder;
use crate::frontend::input::vr_inputs_for_session;
use crate::gpu::GpuQueueAccessGate;
use crate::shared::{HandState, HeadOutputDevice, OutputState, VRControllerState, VRInputsState};
use crate::xr::{OpenxrFrameTick, synthesize_hand_states};

use super::AppDriver;
use std::sync::Arc;

/// Latest OpenXR input state sampled for host IPC.
#[derive(Debug, Default)]
pub(super) struct XrInputCache {
    head_pose: Option<(Vec3, Quat)>,
    controllers: Vec<VRControllerState>,
    hands: Vec<HandState>,
}

impl XrInputCache {
    /// Builds host-facing VR input for the current session output device.
    pub(super) fn build_vr_input(&self, output_device: HeadOutputDevice) -> Option<VRInputsState> {
        let hands = self.hand_states_for_ipc();
        vr_inputs_for_session(output_device, self.head_pose, &self.controllers, hands)
    }

    /// Selects real OpenXR hand-joint data when available, otherwise falls back to synthesis.
    fn hand_states_for_ipc(&self) -> Vec<HandState> {
        if self.hands.is_empty() {
            log_synthetic_hand_source_once();
            synthesize_hand_states(&self.controllers)
        } else {
            self.hands.clone()
        }
    }
}

impl AppDriver {
    /// Runs OpenXR wait/locate and samples input before lock-step IPC input is built.
    pub(super) fn xr_begin_tick(&mut self) -> Option<OpenxrFrameTick> {
        profiling::scope!("tick::xr_begin_tick");
        super::tick_phase_trace("xr_begin_tick");
        let gpu_queue_access_gate = self
            .target
            .as_ref()
            .map(|target| target.gpu().gpu_queue_access_gate().clone())?;
        let flight_recorder = self
            .target
            .as_ref()
            .map(|target| Arc::clone(target.gpu().gpu_flight_recorder()))?;
        let tick = self.begin_openxr_frame_tick(&gpu_queue_access_gate, flight_recorder.as_ref());
        if let Some(ref tick) = tick {
            self.update_xr_input_cache(tick);
        }
        tick
    }

    fn begin_openxr_frame_tick(
        &mut self,
        gpu_queue_access_gate: &GpuQueueAccessGate,
        flight_recorder: &GpuFlightRecorder,
    ) -> Option<OpenxrFrameTick> {
        let target = self.target.as_mut()?;
        let session = target.xr_session_mut()?;
        crate::xr::openxr_begin_frame_tick(
            &mut session.handles,
            &mut self.runtime,
            gpu_queue_access_gate,
            flight_recorder,
        )
    }

    fn update_xr_input_cache(&mut self, tick: &OpenxrFrameTick) {
        crate::xr::OpenxrInput::log_stereo_view_order_once(&tick.views);
        self.sample_openxr_controllers(tick);
        self.xr_input_cache.head_pose =
            crate::xr::headset_center_pose_from_stereo_views(tick.views.as_slice());
        if let Some(head_pose) = self.xr_input_cache.head_pose {
            trace_head_pose(tick, head_pose);
        }
    }

    fn sample_openxr_controllers(&mut self, tick: &OpenxrFrameTick) {
        let Some(target) = self.target.as_ref() else {
            return;
        };
        let Some(session) = target.xr_session() else {
            return;
        };
        let Some(input) = session.handles.openxr_input.as_ref() else {
            return;
        };
        if !session.handles.xr_session.session_running() {
            return;
        }

        match input.sync_and_sample(
            session.handles.xr_session.xr_vulkan_session(),
            session.handles.xr_session.stage_space(),
            tick.predicted_display_time,
        ) {
            Ok(sample) => {
                self.xr_input_cache.controllers = sample.controllers;
                self.xr_input_cache.hands = sample.hands;
            }
            Err(error) => logger::trace!("OpenXR input sync: {error:?}"),
        }
    }

    /// Applies host-requested VR haptics to OpenXR when the current target owns an active session.
    pub(super) fn apply_host_vr_haptics(&mut self, output_state: &OutputState) {
        let Some(target) = self.target.as_ref() else {
            return;
        };
        let Some(session) = target.xr_session() else {
            return;
        };
        if !session.handles.xr_session.session_running() {
            return;
        }
        let Some(input) = session.handles.openxr_input.as_ref() else {
            return;
        };

        self.xr_haptics.apply_output_state(
            input,
            session.handles.xr_session.xr_vulkan_session(),
            output_state.vr.as_ref(),
        );
    }

    /// Renders the HMD stereo view when an OpenXR tick is active.
    pub(super) fn try_hmd_multiview_submit(
        &mut self,
        xr_tick: Option<&OpenxrFrameTick>,
    ) -> crate::xr::HmdSubmitOutcome {
        let Some(tick) = xr_tick else {
            return crate::xr::HmdSubmitOutcome::SkippedBeforeRender;
        };
        let Some(target) = self.target.as_mut() else {
            return crate::xr::HmdSubmitOutcome::SkippedBeforeRender;
        };
        let Some((gpu, session)) = target.openxr_parts_mut() else {
            return crate::xr::HmdSubmitOutcome::SkippedBeforeRender;
        };
        profiling::scope!("xr::hmd_multiview_submit");
        crate::xr::try_openxr_hmd_multiview_submit(gpu, session, &mut self.runtime, tick)
    }

    /// Ends the OpenXR frame with an empty projection layer when one is still open but the
    /// renderer did not submit HMD content this tick (e.g., shutdown, fatal IPC, or graph error).
    pub(super) fn queue_empty_openxr_frame_if_needed(&mut self, xr_tick: Option<OpenxrFrameTick>) {
        let Some(tick) = xr_tick else {
            return;
        };
        let Some(target) = self.target.as_mut() else {
            return;
        };
        let Some((gpu, session)) = target.openxr_parts_mut() else {
            return;
        };
        // Atomic check is intentional: the driver thread clears `frame_open` from a deferred
        // finalize, so this read is safe without holding any session mutex.
        if !session.handles.xr_session.frame_open() {
            return;
        }
        profiling::scope!("xr::end_frame_if_open");
        let (finalize, rx) = session
            .handles
            .xr_session
            .build_empty_finalize(tick.predicted_display_time);
        gpu.submit_finalize_only(finalize);
        session.handles.xr_session.set_pending_finalize(rx);
    }
}

fn trace_head_pose(tick: &OpenxrFrameTick, (ipc_p, ipc_q): (Vec3, Quat)) {
    let (Some(v0), Some(v1)) = (tick.views.first(), tick.views.get(1)) else {
        return;
    };
    let rp0 = &v0.pose.position;
    let rp1 = &v1.pose.position;
    let render_center_x = (rp0.x + rp1.x) * 0.5;
    let render_center_y = (rp0.y + rp1.y) * 0.5;
    let render_center_z = (rp0.z + rp1.z) * 0.5;
    logger::trace!(
        "HEAD POS | render(OpenXR RH): ({:.3},{:.3},{:.3}) | ipc->host(Unity LH): ({:.3},{:.3},{:.3}) | ipc_quat: ({:.4},{:.4},{:.4},{:.4})",
        render_center_x,
        render_center_y,
        render_center_z,
        ipc_p.x,
        ipc_p.y,
        ipc_p.z,
        ipc_q.x,
        ipc_q.y,
        ipc_q.z,
        ipc_q.w,
    );
}

fn log_synthetic_hand_source_once() {
    use std::sync::atomic::{AtomicBool, Ordering};

    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        logger::info!("OpenXR hand input: using synthesized controller poses for IPC hand poses");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{BodyNode, Chirality, IndexControllerState, TouchControllerState};

    fn tracked_touch_controller(side: Chirality) -> VRControllerState {
        VRControllerState::TouchControllerState(TouchControllerState {
            side,
            body_node: match side {
                Chirality::Left => BodyNode::LeftController,
                Chirality::Right => BodyNode::RightController,
            },
            is_device_active: true,
            is_tracking: true,
            rotation: Quat::IDENTITY,
            battery_level: 1.0,
            ..Default::default()
        })
    }

    fn tracked_index_controller(side: Chirality) -> VRControllerState {
        VRControllerState::IndexControllerState(IndexControllerState {
            side,
            body_node: match side {
                Chirality::Left => BodyNode::LeftController,
                Chirality::Right => BodyNode::RightController,
            },
            is_device_active: true,
            is_tracking: true,
            rotation: Quat::IDENTITY,
            battery_level: 1.0,
            ..Default::default()
        })
    }

    fn tracked_hand(unique_id: &str, side: Chirality) -> HandState {
        HandState {
            unique_id: Some(unique_id.to_string()),
            chirality: side,
            is_device_active: true,
            is_tracking: true,
            tracks_metacarpals: true,
            confidence: 1.0,
            segment_positions: vec![Vec3::ZERO; 24],
            segment_rotations: vec![Quat::IDENTITY; 24],
            ..Default::default()
        }
    }

    #[test]
    fn real_hand_states_win_over_controller_synthesis() {
        let cache = XrInputCache {
            controllers: vec![tracked_touch_controller(Chirality::Left)],
            hands: vec![tracked_hand("tracked-left", Chirality::Left)],
            ..Default::default()
        };

        let hands = cache.hand_states_for_ipc();

        assert_eq!(hands.len(), 1);
        assert_eq!(hands[0].unique_id.as_deref(), Some("tracked-left"));
        assert!(hands[0].tracks_metacarpals);
    }

    #[test]
    fn empty_real_hands_use_controller_synthesis() {
        let cache = XrInputCache {
            controllers: vec![tracked_touch_controller(Chirality::Right)],
            ..Default::default()
        };

        let hands = cache.hand_states_for_ipc();

        assert_eq!(hands.len(), 1);
        assert_eq!(hands[0].chirality, Chirality::Right);
        assert!(!hands[0].tracks_metacarpals);
    }

    #[test]
    fn index_controller_synthesis_is_suppressed_without_real_hands() {
        let cache = XrInputCache {
            controllers: vec![tracked_index_controller(Chirality::Left)],
            ..Default::default()
        };

        let hands = cache.hand_states_for_ipc();

        assert!(
            hands.is_empty(),
            "Index fallback must wait for real OpenXR hand joints"
        );
    }
}
