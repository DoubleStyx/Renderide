//! VR [`VRInputsState`](crate::shared::VRInputsState) for host lock-step [`InputState`](crate::shared::InputState).
//!
//! The host creates a headset device only when `headset_state` is present. The desktop accumulator
//! leaves `InputState.vr` empty; this module supplies a minimal headset snapshot for VR
//! [`HeadOutputDevice`](crate::shared::HeadOutputDevice) sessions so VR input initialization is safe.
//! Before the first OpenXR view sample arrives, we keep the headset present but **not tracking**
//! rather than inventing a zero/identity tracked pose.
//! OpenXR supplies controller snapshots via `openxr_controllers` when the runtime has bound actions.

use glam::{Quat, Vec3};

use crate::output_device::head_output_device_is_vr;
use crate::shared::{
    HeadOutputDevice, HeadsetConnection, HeadsetState, VRControllerState, VRInputsState,
};

/// Builds VR input for the host when the session targets a VR [`HeadOutputDevice`].
///
/// `head_pose` is the center-eye pose from the last [`crate::xr::headset_center_pose_from_stereo_views`]
/// update ([`crate::xr::openxr_pose_to_host_tracking`], same idea as Unity XR tracking), or `None`
/// before the first XR tick. When `None`, the headset IPC object is still present but marked
/// `is_tracking = false` so the host can allocate the device without consuming fake origin poses.
///
/// On the FrooxEngine side, **TrackedObject.Position** may differ from **RawPosition** when a
/// **TrackingSpace** applies position/rotation offsets; compare IPC trace logs to **RawPosition**
/// when debugging avatar alignment.
/// `openxr_controllers` is filled from the same XR tick’s `sync_actions` before `pre_frame` runs.
pub fn vr_inputs_for_session(
    session_output_device: HeadOutputDevice,
    head_pose: Option<(Vec3, Quat)>,
    openxr_controllers: &[VRControllerState],
) -> Option<VRInputsState> {
    if !head_output_device_is_vr(session_output_device) {
        return None;
    }
    let is_tracking = head_pose.is_some();
    let (position, rotation) = head_pose.unwrap_or((Vec3::ZERO, Quat::IDENTITY));
    Some(VRInputsState {
        user_present_in_headset: true,
        dashboard_open: false,
        headset_state: Some(HeadsetState {
            is_tracking,
            position,
            rotation,
            battery_level: 1.0,
            battery_charging: false,
            connection_type: HeadsetConnection::wired,
            headset_manufacturer: Some("Renderide".to_string()),
            headset_model: Some("SteamVR".to_string()),
        }),
        controllers: openxr_controllers.to_vec(),
        trackers: Vec::new(),
        tracking_references: Vec::new(),
        hands: Vec::new(),
        vive_hand_tracking: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::HeadOutputDevice;

    #[test]
    fn non_vr_session_returns_none() {
        assert!(vr_inputs_for_session(HeadOutputDevice::screen, None, &[]).is_none());
        assert!(vr_inputs_for_session(HeadOutputDevice::unknown, None, &[]).is_none());
    }

    #[test]
    fn steam_vr_includes_headset_and_wired_connection() {
        let vr = vr_inputs_for_session(HeadOutputDevice::steam_vr, None, &[]).expect("vr session");
        assert!(vr.user_present_in_headset);
        let hs = vr.headset_state.expect("headset");
        assert!(!hs.is_tracking);
        assert_eq!(hs.connection_type, HeadsetConnection::wired);
        assert_eq!(hs.headset_model.as_deref(), Some("SteamVR"));
        assert_eq!(hs.position, Vec3::ZERO);
        assert_eq!(hs.rotation, Quat::IDENTITY);
    }

    #[test]
    fn steam_vr_accepts_cached_pose() {
        let pos = Vec3::new(1.0, 2.0, 3.0);
        let rot = Quat::from_rotation_x(0.5);
        let vr =
            vr_inputs_for_session(HeadOutputDevice::steam_vr, Some((pos, rot)), &[]).expect("vr");
        let hs = vr.headset_state.expect("headset");
        assert!(hs.is_tracking);
        assert_eq!(hs.position, pos);
        assert_eq!(hs.rotation, rot);
    }
}
