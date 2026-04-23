//! OpenXR action set, interaction profile bindings, and per-frame VR controller sampling.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};

use glam::{Quat, Vec2, Vec3};
use openxr as xr;

use crate::shared::{Chirality, VRControllerState};

use super::bindings::ProfileExtensionGates;
use super::frame::{resolve_controller_frame, ControllerFrame};
use super::manifest::Manifest;
use super::openxr_actions::{
    create_openxr_input_parts, OpenxrInputActions, OpenxrInputParts, ResolvedProfilePaths,
};
use super::pose::pose_from_location;
use super::profile::{
    decode_profile_code, is_concrete_profile, log_profile_transition, profile_code,
    ActiveControllerProfile,
};
use super::state::{build_controller_state, OpenxrControllerRawInputs};

/// OpenXR [`xr::Action::state`] snapshot for one hand (all channels consumed by IPC mapping).
struct PolledHandStates {
    trigger: xr::ActionState<f32>,
    trigger_touch: xr::ActionState<bool>,
    trigger_click: xr::ActionState<bool>,
    squeeze: xr::ActionState<f32>,
    squeeze_click: xr::ActionState<bool>,
    thumbstick: xr::ActionState<xr::Vector2f>,
    thumbstick_touch: xr::ActionState<bool>,
    thumbstick_click: xr::ActionState<bool>,
    trackpad: xr::ActionState<xr::Vector2f>,
    trackpad_touch: xr::ActionState<bool>,
    trackpad_click: xr::ActionState<bool>,
    trackpad_force: xr::ActionState<f32>,
    primary: xr::ActionState<bool>,
    secondary: xr::ActionState<bool>,
    primary_touch: xr::ActionState<bool>,
    secondary_touch: xr::ActionState<bool>,
    menu: xr::ActionState<bool>,
    thumbrest_touch: xr::ActionState<bool>,
    select: xr::ActionState<bool>,
}

impl PolledHandStates {
    fn thumbstick_vec(&self) -> Vec2 {
        Vec2::new(
            self.thumbstick.current_state.x,
            self.thumbstick.current_state.y,
        )
    }

    fn trackpad_vec(&self) -> Vec2 {
        Vec2::new(self.trackpad.current_state.x, self.trackpad.current_state.y)
    }
}

/// Fallback [`ControllerFrame`] when [`resolve_controller_frame`] returns [`None`].
fn placeholder_controller_frame() -> ControllerFrame {
    ControllerFrame {
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        has_bound_hand: false,
        hand_position: Vec3::ZERO,
        hand_rotation: Quat::IDENTITY,
    }
}

/// Maps a resolved pose frame (if any) plus analog/digital samples into a host-facing [`VRControllerState`].
fn ipc_vr_controller_from_polled(
    profile: ActiveControllerProfile,
    side: Chirality,
    resolved_frame: Option<ControllerFrame>,
    polled: &PolledHandStates,
) -> VRControllerState {
    let tracking_valid = resolved_frame.is_some();
    let frame = resolved_frame.unwrap_or_else(placeholder_controller_frame);
    build_controller_state(OpenxrControllerRawInputs {
        profile,
        side,
        is_tracking: tracking_valid,
        frame,
        trigger: polled.trigger.current_state,
        trigger_touch: polled.trigger_touch.current_state,
        trigger_click: polled.trigger_click.current_state,
        squeeze: polled.squeeze.current_state,
        squeeze_click: polled.squeeze_click.current_state,
        thumbstick: polled.thumbstick_vec(),
        thumbstick_touch: polled.thumbstick_touch.current_state,
        thumbstick_click: polled.thumbstick_click.current_state,
        trackpad: polled.trackpad_vec(),
        trackpad_touch: polled.trackpad_touch.current_state,
        trackpad_click: polled.trackpad_click.current_state,
        trackpad_force: polled.trackpad_force.current_state,
        primary: polled.primary.current_state,
        secondary: polled.secondary.current_state,
        primary_touch: polled.primary_touch.current_state,
        secondary_touch: polled.secondary_touch.current_state,
        menu: polled.menu.current_state,
        thumbrest_touch: polled.thumbrest_touch.current_state,
        select: polled.select.current_state,
    })
}

/// OpenXR actions and derived spaces for headset/controller input used by the renderer IPC path.
pub struct OpenxrInput {
    action_set: xr::ActionSet,
    left_user_path: xr::Path,
    right_user_path: xr::Path,
    profile_paths: ResolvedProfilePaths,
    left_profile_cache: AtomicU8,
    right_profile_cache: AtomicU8,
    actions: OpenxrInputActions,
    left_space: xr::Space,
    right_space: xr::Space,
    left_aim_space: xr::Space,
    right_aim_space: xr::Space,
}

impl OpenxrInput {
    /// Loads the XR action manifest, creates the action set, suggests bindings, and builds spaces.
    ///
    /// `gates` must reflect which OpenXR extensions were enabled on the instance; binding
    /// suggestions for profiles whose extension is disabled are skipped so runtimes that do not
    /// recognise those paths do not emit errors.
    pub fn new(
        instance: &xr::Instance,
        session: &xr::Session<xr::Vulkan>,
        gates: &ProfileExtensionGates,
        manifest: &Manifest,
    ) -> Result<Self, xr::sys::Result> {
        let parts = create_openxr_input_parts(instance, session, gates, manifest)?;
        Ok(Self::from_parts(parts))
    }

    fn from_parts(parts: OpenxrInputParts) -> Self {
        Self {
            action_set: parts.action_set,
            left_user_path: parts.left_user_path,
            right_user_path: parts.right_user_path,
            profile_paths: parts.profile_paths,
            left_profile_cache: parts.left_profile_cache,
            right_profile_cache: parts.right_profile_cache,
            actions: parts.actions,
            left_space: parts.left_space,
            right_space: parts.right_space,
            left_aim_space: parts.left_aim_space,
            right_aim_space: parts.right_aim_space,
        }
    }

    /// Triggers a haptic pulse on the requested hand via
    /// [`openxr::Action::apply_feedback`]. `amplitude` is `0..=1`; `duration_nanos` of
    /// `xr::Duration::MIN_HAPTIC` lets the runtime choose a minimum; `frequency_hz` uses
    /// `xr::FREQUENCY_UNSPECIFIED` when left-to-runtime.
    pub fn apply_haptic(
        &self,
        session: &xr::Session<xr::Vulkan>,
        side: Chirality,
        amplitude: f32,
        duration_nanos: i64,
        frequency_hz: f32,
    ) -> Result<(), xr::sys::Result> {
        let action = match side {
            Chirality::Left => &self.actions.left_haptic,
            Chirality::Right => &self.actions.right_haptic,
        };
        let event = xr::HapticVibration::new()
            .amplitude(amplitude)
            .duration(xr::Duration::from_nanos(duration_nanos))
            .frequency(frequency_hz);
        action.apply_feedback(session, xr::Path::NULL, &event)
    }

    fn detect_profile(
        &self,
        session: &xr::Session<xr::Vulkan>,
        hand_user_path: xr::Path,
    ) -> ActiveControllerProfile {
        let Ok(profile) = session.current_interaction_profile(hand_user_path) else {
            return ActiveControllerProfile::Generic;
        };
        let paths = &self.profile_paths;
        if profile == paths.oculus_touch
            || profile == paths.meta_touch_pro
            || profile == paths.meta_touch_plus
        {
            ActiveControllerProfile::Touch
        } else if profile == paths.pico4_controller {
            ActiveControllerProfile::Pico4
        } else if profile == paths.pico_neo3_controller {
            ActiveControllerProfile::PicoNeo3
        } else if profile == paths.valve_index {
            ActiveControllerProfile::Index
        } else if profile == paths.htc_vive {
            ActiveControllerProfile::Vive
        } else if profile == paths.htc_vive_cosmos {
            ActiveControllerProfile::ViveCosmos
        } else if profile == paths.htc_vive_focus3 {
            ActiveControllerProfile::ViveFocus3
        } else if profile == paths.hp_reverb_g2 {
            ActiveControllerProfile::HpReverbG2
        } else if profile == paths.microsoft_motion || profile == paths.samsung_odyssey {
            ActiveControllerProfile::WindowsMr
        } else if profile == paths.generic_controller {
            ActiveControllerProfile::Generic
        } else if profile == paths.simple_controller || profile == xr::Path::NULL {
            ActiveControllerProfile::Simple
        } else {
            ActiveControllerProfile::Generic
        }
    }

    fn active_profile(
        &self,
        session: &xr::Session<xr::Vulkan>,
        hand_user_path: xr::Path,
        side: Chirality,
    ) -> ActiveControllerProfile {
        let live = self.detect_profile(session, hand_user_path);
        let cache = match side {
            Chirality::Left => &self.left_profile_cache,
            Chirality::Right => &self.right_profile_cache,
        };
        if is_concrete_profile(live) {
            cache.store(profile_code(live), Ordering::Relaxed);
            return live;
        }
        decode_profile_code(cache.load(Ordering::Relaxed))
            .filter(|cached| is_concrete_profile(*cached))
            .unwrap_or(live)
    }

    /// Samples every bound action for the given hand after [`xr::Session::sync_actions`].
    fn poll_hand_action_states(
        &self,
        session: &xr::Session<xr::Vulkan>,
        side: Chirality,
    ) -> Result<PolledHandStates, xr::sys::Result> {
        let a = &self.actions;
        match side {
            Chirality::Left => Ok(PolledHandStates {
                trigger: a.left_trigger.state(session, xr::Path::NULL)?,
                trigger_touch: a.left_trigger_touch.state(session, xr::Path::NULL)?,
                trigger_click: a.left_trigger_click.state(session, xr::Path::NULL)?,
                squeeze: a.left_squeeze.state(session, xr::Path::NULL)?,
                squeeze_click: a.left_squeeze_click.state(session, xr::Path::NULL)?,
                thumbstick: a.left_thumbstick.state(session, xr::Path::NULL)?,
                thumbstick_touch: a.left_thumbstick_touch.state(session, xr::Path::NULL)?,
                thumbstick_click: a.left_thumbstick_click.state(session, xr::Path::NULL)?,
                trackpad: a.left_trackpad.state(session, xr::Path::NULL)?,
                trackpad_touch: a.left_trackpad_touch.state(session, xr::Path::NULL)?,
                trackpad_click: a.left_trackpad_click.state(session, xr::Path::NULL)?,
                trackpad_force: a.left_trackpad_force.state(session, xr::Path::NULL)?,
                primary: a.left_primary.state(session, xr::Path::NULL)?,
                secondary: a.left_secondary.state(session, xr::Path::NULL)?,
                primary_touch: a.left_primary_touch.state(session, xr::Path::NULL)?,
                secondary_touch: a.left_secondary_touch.state(session, xr::Path::NULL)?,
                menu: a.left_menu.state(session, xr::Path::NULL)?,
                thumbrest_touch: a.left_thumbrest_touch.state(session, xr::Path::NULL)?,
                select: a.left_select.state(session, xr::Path::NULL)?,
            }),
            Chirality::Right => Ok(PolledHandStates {
                trigger: a.right_trigger.state(session, xr::Path::NULL)?,
                trigger_touch: a.right_trigger_touch.state(session, xr::Path::NULL)?,
                trigger_click: a.right_trigger_click.state(session, xr::Path::NULL)?,
                squeeze: a.right_squeeze.state(session, xr::Path::NULL)?,
                squeeze_click: a.right_squeeze_click.state(session, xr::Path::NULL)?,
                thumbstick: a.right_thumbstick.state(session, xr::Path::NULL)?,
                thumbstick_touch: a.right_thumbstick_touch.state(session, xr::Path::NULL)?,
                thumbstick_click: a.right_thumbstick_click.state(session, xr::Path::NULL)?,
                trackpad: a.right_trackpad.state(session, xr::Path::NULL)?,
                trackpad_touch: a.right_trackpad_touch.state(session, xr::Path::NULL)?,
                trackpad_click: a.right_trackpad_click.state(session, xr::Path::NULL)?,
                trackpad_force: a.right_trackpad_force.state(session, xr::Path::NULL)?,
                primary: a.right_primary.state(session, xr::Path::NULL)?,
                secondary: a.right_secondary.state(session, xr::Path::NULL)?,
                primary_touch: a.right_primary_touch.state(session, xr::Path::NULL)?,
                secondary_touch: a.right_secondary_touch.state(session, xr::Path::NULL)?,
                menu: a.right_menu.state(session, xr::Path::NULL)?,
                thumbrest_touch: a.right_thumbrest_touch.state(session, xr::Path::NULL)?,
                select: a.right_select.state(session, xr::Path::NULL)?,
            }),
        }
    }

    /// Syncs actions, samples poses and digital/analog state, and returns left/right [`VRControllerState`] values.
    pub fn sync_and_sample(
        &self,
        session: &xr::Session<xr::Vulkan>,
        stage: &xr::Space,
        predicted_time: xr::Time,
    ) -> Result<Vec<VRControllerState>, xr::sys::Result> {
        profiling::scope!("xr::input_sync_and_sample");
        session.sync_actions(&[xr::ActiveActionSet::new(&self.action_set)])?;
        let left_loc = self.left_space.locate(stage, predicted_time)?;
        let right_loc = self.right_space.locate(stage, predicted_time)?;
        let left_aim_loc = self.left_aim_space.locate(stage, predicted_time)?;
        let right_aim_loc = self.right_aim_space.locate(stage, predicted_time)?;
        let left_grip_pose = pose_from_location(&left_loc);
        let right_grip_pose = pose_from_location(&right_loc);
        let left_aim_pose = pose_from_location(&left_aim_loc);
        let right_aim_pose = pose_from_location(&right_aim_loc);

        let left_polled = self.poll_hand_action_states(session, Chirality::Left)?;
        let right_polled = self.poll_hand_action_states(session, Chirality::Right)?;

        let left_profile = self.active_profile(session, self.left_user_path, Chirality::Left);
        let right_profile = self.active_profile(session, self.right_user_path, Chirality::Right);
        log_profile_transition(Chirality::Left, left_profile);
        log_profile_transition(Chirality::Right, right_profile);
        Self::log_grip_missing_aim_valid_throttled(Chirality::Left, left_grip_pose, left_aim_pose);
        Self::log_grip_missing_aim_valid_throttled(
            Chirality::Right,
            right_grip_pose,
            right_aim_pose,
        );
        let left_frame =
            resolve_controller_frame(left_profile, Chirality::Left, left_grip_pose, left_aim_pose);
        let right_frame = resolve_controller_frame(
            right_profile,
            Chirality::Right,
            right_grip_pose,
            right_aim_pose,
        );
        let left =
            ipc_vr_controller_from_polled(left_profile, Chirality::Left, left_frame, &left_polled);
        let right = ipc_vr_controller_from_polled(
            right_profile,
            Chirality::Right,
            right_frame,
            &right_polled,
        );
        Ok(vec![left, right])
    }

    /// Logs at most once every 300 frames per hand when the grip space is untracked but the aim space is valid.
    ///
    /// Confirms on-device that [`super::frame::resolve_controller_frame`] can use the aim fallback path.
    fn log_grip_missing_aim_valid_throttled(
        side: Chirality,
        grip_pose: Option<(Vec3, Quat)>,
        aim_pose: Option<(Vec3, Quat)>,
    ) {
        if grip_pose.is_some() || aim_pose.is_none() {
            return;
        }
        static LEFT: AtomicU32 = AtomicU32::new(0);
        static RIGHT: AtomicU32 = AtomicU32::new(0);
        let slot = match side {
            Chirality::Left => &LEFT,
            Chirality::Right => &RIGHT,
        };
        let n = slot.fetch_add(1, Ordering::Relaxed);
        if n % 300 == 0 {
            logger::debug!(
                "OpenXR {side:?}: grip pose invalid or untracked but aim pose valid; resolving controller frame from aim for IPC"
            );
        }
    }

    /// Logs once (at trace level) if stereo view array order may not match left-then-right pose ordering.
    pub fn log_stereo_view_order_once(views: &[xr::View]) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if views.len() < 2 || ONCE.swap(true, Ordering::Relaxed) {
            return;
        }
        let x0 = views[0].pose.position.x;
        let x1 = views[1].pose.position.x;
        if x0 > x1 + 0.02 {
            logger::trace!(
                "OpenXR stereo: views[0].pose.x ({x0}) > views[1].pose.x ({x1}); runtime may use right-then-left ordering - verify eye mapping."
            );
        }
    }
}
